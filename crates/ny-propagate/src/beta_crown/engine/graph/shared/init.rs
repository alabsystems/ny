// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph-BaB intermediate-bounds bootstrap.
//!
//! Unifies the alpha/IBP/CROWN-IBP mode selection shared by the graph BaB
//! entry points before each path computes its mode-specific root objective
//! bounds.
//!
//! Design: `designs/2026-03-14-issue-1860-graph-bab-service-convergence.md`
//! Issue: #1860 (Packet C)

use std::collections::HashMap;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::beta_crown::config::BetaCrownConfig;
use crate::bounds::GraphAlphaState;
use crate::network::{GraphNetworkCrownExt, SpecCrownRequest};
use crate::{AlphaCrownConfig, GraphNetwork, MulBinaryRelaxationMode};

/// Shared graph-BaB bootstrap state produced before root objective evaluation.
#[must_use]
pub(crate) struct GraphBabBootstrap {
    pub(crate) initial_node_bounds: HashMap<String, BoundedTensor>,
    pub(crate) root_alpha_state: Option<GraphAlphaState>,
    pub(crate) alpha_config: AlphaCrownConfig,
}

fn resolve_graph_output_bounds<'a>(
    graph: &'a GraphNetwork,
    node_bounds: &'a HashMap<String, BoundedTensor>,
) -> Result<&'a BoundedTensor> {
    let output_node_name = if graph.output_name().is_empty() {
        graph
            .exec_order()?
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
            .as_str()
    } else {
        graph.output_name()
    };

    node_bounds.get(output_node_name).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Graph BaB bootstrap missing output node bounds for '{output_node_name}'"
        ))
    })
}

/// True when `err` is a deadline-exceeded signal from an intermediate-bounds sweep.
fn is_deadline_exceeded(err: &NyError) -> bool {
    matches!(err, NyError::DeadlineExceeded(_))
}

/// Deadline-free plain-IBP intermediate bounds: the cheapest sound fallback when
/// a non-alpha warmup collection (CROWN-IBP / forward-linear / deadline-checked
/// IBP) exhausts the warmup budget (#4260). Plain IBP is O(L) and always sound
/// (looser, never unsound), so these root-fallback paths can always make forward
/// progress without an external timeout kill.
///
/// Deliberately NOT used by:
///   * the large-conv path — there IBP itself is the expensive sweep that must
///     hard-bail to emit a verdict (#4321);
///   * the α-CROWN path — its DeadlineExceeded is translated by the BaB alpha
///     entry points into an explicit warmup-cap Unknown (#4413).
fn ibp_fallback_node_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<HashMap<String, BoundedTensor>> {
    info!("Warmup deadline exceeded; falling back to plain-IBP intermediate bounds (#4260).");
    graph.collect_node_bounds_with_engine(input, engine)
}

/// Compute the shared intermediate-bounds bootstrap for graph BaB engines.
pub(crate) fn compute_graph_bab_bootstrap(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<GraphBabBootstrap> {
    // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): bracket the
    // whole warmup + intermediate-bounds collection so the phase is priceable
    // from a log. Gate-off is a cached-bool load — byte-identical output.
    crate::phase_telemetry::phase_marker("graph-bab-bootstrap start");
    config.validate()?;

    let alpha_config = {
        let mut alpha_config = config.alpha_config.clone();
        alpha_config.deadline = deadline;
        // #fit-100s (dark, `NY_WARMUP_ITERS=k`, default absent = preset value):
        // cap the root α-CROWN warmup iteration count. On cifar100_2024 @100s
        // the warmup runs ~7 iterations at ~1.5s each before its own early
        // exit; a small cap trades a slightly looser root for BaB time — the
        // per-domain wide-α/β ascent and the interm-refine lane recover the
        // tightness where it matters (measured direction, task #fit). Only
        // ever LOWERS the configured count (min), never raises it.
        if let Some(cap) = std::env::var("NY_WARMUP_ITERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            alpha_config.iterations = alpha_config.iterations.min(cap);
        }
        // #root-alpha-true (dark, `NY_ROOT_ALPHA_ITERS=k`, default absent): OVERRIDE
        // (set, can RAISE — unlike NY_WARMUP_ITERS which only lowers) the root
        // α-CROWN warmup iteration count. Pairs with `NY_ROOT_ALPHA_TRUE=1`, which
        // swaps the warmup's wrong local-rule gradient (`pre_lower·Σmax(ν,0)`) for
        // the true chain-rule gradient (`max(ν,0)·ĥ(x*)`): the right direction plus
        // enough iters (αβ runs ~20 root α iters) to converge the root relaxation
        // toward its own LP optimum. Applied AFTER the NY_WARMUP_ITERS cap so it
        // wins. Absent ⇒ preset value unchanged (byte-identical). Sound: any
        // α∈[0,1] the ascent visits is a valid relaxation; iteration count only
        // schedules optimization work.
        if let Some(iters) = std::env::var("NY_ROOT_ALPHA_ITERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            alpha_config.iterations = iters;
        }
        // #wall-affordability knob (dark, `NY_ROOT_ALPHA_CAP_SECS=k`, default
        // absent ⇒ byte-identical): cap the root α-CROWN warmup's wall clock by
        // tightening its deadline (min with any existing). Deadlines only
        // schedule work — on expiry every path falls back to the sound
        // reference bounds — so the knob carries no soundness obligation. The
        // completed wall matrix (2026-07-20) prices root-α at ~460s of a >850s
        // root pipeline against the ~50s an official 100s budget allows; this
        // knob is what the tightness-vs-cost affordability curve is measured
        // with.
        if let Some(cap_secs) = std::env::var("NY_ROOT_ALPHA_CAP_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            let capped = Instant::now() + std::time::Duration::from_secs(cap_secs);
            alpha_config.deadline = Some(alpha_config.deadline.map_or(capped, |d| d.min(capped)));
        }
        alpha_config
    };

    // Large convolutional graphs: per-node CROWN-IBP / α-CROWN intermediate-bound
    // collection is O(L²) dense conv-transpose GEMMs at full spatial resolution,
    // which burns the entire timeout before BaB even starts (TinyYOLO: 31 nodes
    // but ~8k-element image input stalled ~50s pre-domain). For these, fall back
    // to cheap O(L) IBP intermediate bounds (sound, just looser) so the budget is
    // spent on BaB search instead. Small models (e.g. cifar100, 3072 inputs) keep
    // the tighter α-CROWN path. Gated on conv presence + input volume to avoid any
    // regression on the categories α-CROWN already solves. (#perf-conv-ibp)
    const LARGE_CONV_INPUT_NUMEL: usize = 5000;
    // #conv-patches-collect (metaroom/cifar100 lever): the large-conv gate below
    // routes deep conv DAGs to PLAIN IBP intermediate bounds to dodge the O(L²)
    // dense CROWN-IBP conv stall. But the CROWN-IBP collector's patches-start
    // path (`crown_ibp_target_can_start_in_patches` + collector override) keeps
    // the deep spatial conv targets in the memory-light PATCHES representation,
    // so the dense OOM the gate protects against never materializes for the
    // patches-eligible nodes (the rest degrade to sound IBP per node). This
    // default-OFF env lifts the gate for conv graphs so those spatial targets
    // tighten via patches-mode conv CROWN instead of staying pure-IBP loose.
    // Env-UNSET is byte-identical (the extra `&& !...` cannot change the bool).
    // Sound either way: the collector INTERSECTS CROWN with IBP per node and
    // any per-node patches failure falls back to that node's IBP bound.
    let force_conv_patches_collect =
        std::env::var_os("NY_CONV_PATCHES_COLLECT").is_some_and(|v| v != "0" && !v.is_empty());
    let large_conv_graph = graph.has_conv_layers()
        && input.len() > LARGE_CONV_INPUT_NUMEL
        && !force_conv_patches_collect;

    let (initial_node_bounds, root_alpha_state) = if force_conv_patches_collect
        && graph.has_conv_layers()
        && input.len() > LARGE_CONV_INPUT_NUMEL
    {
        info!(
            "#conv-patches-collect: lifting the large-conv IBP gate ({} input elements); \
             running CROWN-IBP intermediate collection with patches-start conv targets.",
            input.len()
        );
        match graph.collect_crown_ibp_bounds_dag_with_deadline_and_engine(input, deadline, engine) {
            Ok(bounds) => (bounds, None),
            Err(e) if is_deadline_exceeded(&e) => {
                (ibp_fallback_node_bounds(graph, input, engine)?, None)
            }
            Err(e) => return Err(e),
        }
    } else if large_conv_graph {
        info!(
            "Large conv graph ({} input elements): using IBP intermediate bounds to avoid \
             O(L²) per-node CROWN-IBP conv stall; reserving budget for BaB.",
            input.len()
        );
        // Thread the global deadline (#4321): without it, the IBP intermediate
        // sweep over a deep conv DAG (TinyImageNet ResNet) runs unbounded and
        // overruns the verifier timeout, getting killed externally with no JSON
        // verdict. collect_node_bounds_core checks the deadline between nodes.
        // This path keeps the hard deadline bail: on a deep conv DAG IBP itself
        // is the expensive sweep that must abort to emit a verdict.
        (
            graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline)?,
            None,
        )
    } else if config.use_alpha_crown {
        info!(
            "Computing α-CROWN initial bounds ({} iterations, fix_interm_bounds={})...",
            config.alpha_config.iterations, config.alpha_config.fix_interm_bounds
        );
        // α-CROWN deliberately propagates DeadlineExceeded: the GPU/heap BaB
        // alpha entry points translate it into an explicit "warmup exceeded its
        // deadline cap" Unknown (#4413) so per-domain budget is preserved. Do NOT
        // swallow it into an IBP fallback here.
        let (bounds, alpha) =
            graph.collect_alpha_crown_bounds_dag_with_engine(input, &alpha_config, engine)?;
        (bounds, Some(alpha))
    } else if config.use_forward_bounds {
        info!("Computing forward-linear initial bounds...");
        match graph
            .collect_forward_linear_bounds_dag_with_engine_and_deadline(input, engine, deadline)
        {
            Ok(bounds) => (bounds, None),
            Err(e) if is_deadline_exceeded(&e) => {
                (ibp_fallback_node_bounds(graph, input, engine)?, None)
            }
            Err(e) => return Err(e),
        }
    } else if config.alpha_config.fix_interm_bounds {
        info!("Computing IBP initial bounds...");
        match graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline) {
            Ok(bounds) => (bounds, None),
            Err(e) if is_deadline_exceeded(&e) => {
                (ibp_fallback_node_bounds(graph, input, engine)?, None)
            }
            Err(e) => return Err(e),
        }
    } else {
        info!("Computing CROWN-IBP initial bounds...");
        match graph.collect_crown_ibp_bounds_dag_with_deadline_and_engine(input, deadline, engine) {
            Ok(bounds) => (bounds, None),
            Err(e) if is_deadline_exceeded(&e) => {
                (ibp_fallback_node_bounds(graph, input, engine)?, None)
            }
            Err(e) => return Err(e),
        }
    };

    // #phase-telemetry: end of the warmup+collect phase (error exits above
    // abort the pipeline, so start-without-end in a log reads as "bootstrap
    // did not complete").
    crate::phase_telemetry::phase_marker("graph-bab-bootstrap end");
    Ok(GraphBabBootstrap {
        initial_node_bounds,
        root_alpha_state,
        alpha_config,
    })
}

/// Compute full-output root bounds for graph BaB entry points from a shared bootstrap.
///
/// `forward+crown` reuses the bootstrap's forward-linear node map through an
/// identity-spec CROWN request instead of silently reverting to the plain
/// DAG-CROWN intermediate-bound source.
pub(crate) fn compute_graph_root_output_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
    bootstrap: &GraphBabBootstrap,
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    if config.use_alpha_crown {
        return graph.propagate_alpha_crown_with_config_and_engine(
            input,
            &bootstrap.alpha_config,
            engine,
        );
    }

    if config.use_forward_bounds {
        let output_bounds = resolve_graph_output_bounds(graph, &bootstrap.initial_node_bounds)?;
        let output_shape = output_bounds.shape().to_vec();
        let identity_spec = ndarray::Array2::<f32>::eye(output_bounds.len());
        let output = SpecCrownRequest::new(graph, input, &identity_spec, engine)
            .node_bounds(&bootstrap.initial_node_bounds)
            .deadline_opt(deadline)
            .truncate_after_opt(config.crown_backward_layers)
            .run()?;
        return output.reshape(&output_shape);
    }

    GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation(
        graph,
        input,
        engine,
        MulBinaryRelaxationMode::default(),
        deadline,
        config.crown_backward_layers,
    )
    .map(|result| result.bounds)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    use super::*;
    use crate::beta_crown::config::BetaCrownConfig;
    use crate::layers::{AddLayer, Conv2dLayer, Layer, LinearLayer, ReLULayer, ReduceSumLayer};
    use crate::network::GraphNode;

    fn build_test_graph() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();

        let linear1 = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.25_f32, 0.75]]),
            Some(arr1(&[0.0_f32, 0.1])),
        )
        .expect("test graph linear1 should be valid");
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));

        let linear2 = LinearLayer::new(arr2(&[[1.0_f32, -1.0]]), Some(arr1(&[0.0_f32]))).unwrap();
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");

        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("test input bounds should be valid");

        (graph, input)
    }

    fn build_residual_dag_4404() -> (GraphNetwork, BoundedTensor) {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 2, 1, 1]), vec![0.9_f32, -0.35, -0.45, 0.8])
            .expect("valid Conv2d kernel");
        let bias = arr1(&[0.05_f32, -0.1]);
        let conv = Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 2, 2)
            .expect("valid Conv2d params");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "residual",
            Layer::Add(AddLayer),
            "relu",
            crate::NETWORK_INPUT,
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::ReduceSum(ReduceSumLayer::new(vec![0, 1, 2], false)),
            vec!["residual".to_string()],
        ));
        graph.set_output("out");

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[2, 2, 2]),
                vec![-1.0_f32, -0.6, 0.1, -0.3, -0.5, -0.2, 0.0, -0.4],
            )
            .expect("valid lower input shape"),
            ArrayD::from_shape_vec(
                IxDyn(&[2, 2, 2]),
                vec![1.2_f32, 0.7, 0.9, 0.6, 0.8, 0.5, 1.0, 0.4],
            )
            .expect("valid upper input shape"),
        )
        .expect("residual DAG input bounds should be valid");

        (graph, input)
    }

    fn node_bounds_max_abs_diff_4404(actual: &BoundedTensor, expected: &BoundedTensor) -> f32 {
        actual
            .lower()
            .iter()
            .zip(expected.lower().iter())
            .chain(actual.upper().iter().zip(expected.upper().iter()))
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f32, f32::max)
    }

    fn assert_node_bounds_match_4404(
        actual: &HashMap<String, BoundedTensor>,
        expected: &HashMap<String, BoundedTensor>,
        node_name: &str,
        label: &str,
    ) {
        let actual_bounds = actual
            .get(node_name)
            .unwrap_or_else(|| panic!("{label}: missing actual node bounds for '{node_name}'"));
        let expected_bounds = expected
            .get(node_name)
            .unwrap_or_else(|| panic!("{label}: missing expected node bounds for '{node_name}'"));
        assert_eq!(
            actual_bounds.shape(),
            expected_bounds.shape(),
            "{label}: node '{node_name}' shape mismatch"
        );
        for (actual_value, expected_value) in actual_bounds
            .lower()
            .iter()
            .zip(expected_bounds.lower().iter())
        {
            assert!(
                (actual_value - expected_value).abs() <= 1e-6,
                "{label}: node '{node_name}' lower mismatch actual={actual_value}, expected={expected_value}"
            );
        }
        for (actual_value, expected_value) in actual_bounds
            .upper()
            .iter()
            .zip(expected_bounds.upper().iter())
        {
            assert!(
                (actual_value - expected_value).abs() <= 1e-6,
                "{label}: node '{node_name}' upper mismatch actual={actual_value}, expected={expected_value}"
            );
        }
    }

    #[test]
    fn test_compute_graph_bab_bootstrap_ibp_path_preserves_deadline() {
        let (graph, input) = build_test_graph();
        let deadline = Some(Instant::now() + Duration::from_secs(1));
        let config = BetaCrownConfig {
            use_alpha_crown: false,
            ..Default::default()
        };

        let bootstrap = compute_graph_bab_bootstrap(&graph, &input, &config, None, deadline)
            .expect("IBP bootstrap should succeed on the toy graph");

        assert!(bootstrap.root_alpha_state.is_none());
        assert_eq!(bootstrap.alpha_config.deadline, deadline);
        assert!(
            bootstrap.initial_node_bounds.contains_key("relu"),
            "IBP bootstrap should collect intermediate bounds"
        );
    }

    #[test]
    fn test_compute_graph_bab_bootstrap_alpha_path_returns_root_alpha() {
        let (graph, input) = build_test_graph();
        let deadline = Some(Instant::now() + Duration::from_secs(1));
        let mut config = BetaCrownConfig {
            use_alpha_crown: true,
            ..Default::default()
        };
        config.alpha_config.iterations = 1;
        config.alpha_config.adaptive_skip = false;
        config.alpha_config.adaptive_skip_pilot = false;

        let bootstrap = compute_graph_bab_bootstrap(&graph, &input, &config, None, deadline)
            .expect("α-CROWN bootstrap should succeed on the toy graph");

        assert!(bootstrap.root_alpha_state.is_some());
        assert_eq!(bootstrap.alpha_config.deadline, deadline);
        assert!(
            bootstrap.initial_node_bounds.contains_key("linear2"),
            "α-CROWN bootstrap should collect output-adjacent bounds"
        );
    }

    #[test]
    fn test_compute_graph_bab_bootstrap_forward_path_uses_forward_linear_bounds_4354() {
        let (graph, input) = build_test_graph();
        let deadline = Some(Instant::now() + Duration::from_secs(1));
        let config = BetaCrownConfig {
            use_alpha_crown: false,
            use_forward_bounds: true,
            ..Default::default()
        };

        let expected = graph
            .collect_forward_linear_bounds_dag_with_engine(&input, None)
            .expect("forward-linear collection should succeed on the toy graph");
        let bootstrap = compute_graph_bab_bootstrap(&graph, &input, &config, None, deadline)
            .expect("forward bootstrap should succeed on the toy graph");

        assert!(bootstrap.root_alpha_state.is_none());
        assert_eq!(bootstrap.alpha_config.deadline, deadline);
        let actual_relu = bootstrap
            .initial_node_bounds
            .get("relu")
            .expect("bootstrap should include relu bounds");
        let expected_relu = expected
            .get("relu")
            .expect("forward-linear bounds should include relu");
        assert_eq!(actual_relu.shape(), expected_relu.shape());
        for (actual, expected) in actual_relu.lower().iter().zip(expected_relu.lower().iter()) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "forward bootstrap lower mismatch: actual={actual}, expected={expected}"
            );
        }
        for (actual, expected) in actual_relu.upper().iter().zip(expected_relu.upper().iter()) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "forward bootstrap upper mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_compute_graph_bab_bootstrap_forward_path_falls_back_to_ibp_on_expired_deadline_4260() {
        // Contract change (#4260): a forward-linear warmup that exhausts its
        // deadline no longer aborts the whole bootstrap with DeadlineExceeded.
        // Instead it falls back to plain-IBP intermediate bounds — the cheapest
        // SOUND collector (looser, never unsound) — so a non-alpha root can still
        // make forward progress instead of being killed externally with no
        // verdict. Plain IBP over-approximates the forward-linear bounds, so the
        // fallback can only make the verifier MORE conservative, never wrongly
        // "verified". This supersedes the obsolete hard-fail contract that the
        // pre-#4260 test pinned.
        let (graph, input) = build_test_graph();
        let config = BetaCrownConfig {
            use_alpha_crown: false,
            use_forward_bounds: true,
            ..Default::default()
        };
        let expired_deadline = Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap(),
        );

        let bootstrap =
            compute_graph_bab_bootstrap(&graph, &input, &config, None, expired_deadline)
                .expect("expired forward-linear warmup should fall back to plain IBP, not abort");

        assert!(
            bootstrap.root_alpha_state.is_none(),
            "IBP fallback bootstrap carries no alpha state"
        );

        // The fallback must reproduce the plain-IBP intermediate bounds exactly.
        let ibp_reference = graph
            .collect_node_bounds_with_engine(&input, None)
            .expect("plain IBP reference should succeed on the toy graph");
        for node in ["relu", "linear2"] {
            assert_node_bounds_match_4404(
                &bootstrap.initial_node_bounds,
                &ibp_reference,
                node,
                "forward-deadline IBP fallback",
            );
        }
    }

    #[test]
    fn test_compute_graph_bab_bootstrap_alpha_path_uses_ibp_intermediates_when_fix_interm_bounds_true_4404(
    ) {
        let (graph, input) = build_residual_dag_4404();
        let ibp_bounds = graph
            .collect_node_bounds_with_engine(&input, None)
            .expect("IBP reference bounds should succeed on the residual DAG");
        let crown_ibp_bounds = graph
            .collect_crown_ibp_bounds_dag(&input)
            .expect("CROWN-IBP reference bounds should succeed on the residual DAG");

        let residual_ibp = ibp_bounds
            .get("residual")
            .expect("IBP bounds should include the residual node");
        let residual_crown_ibp = crown_ibp_bounds
            .get("residual")
            .expect("CROWN-IBP bounds should include the residual node");
        let output_ibp = ibp_bounds
            .get("out")
            .expect("IBP bounds should include the output node");
        let output_crown_ibp = crown_ibp_bounds
            .get("out")
            .expect("CROWN-IBP bounds should include the output node");
        assert!(
            node_bounds_max_abs_diff_4404(residual_ibp, residual_crown_ibp)
                .max(node_bounds_max_abs_diff_4404(output_ibp, output_crown_ibp))
                > 1e-5,
            "#4404 oracle graph must distinguish DAG IBP from DAG CROWN-IBP at bootstrap"
        );

        let deadline = Some(Instant::now() + Duration::from_secs(1));
        let mut config = BetaCrownConfig {
            use_alpha_crown: true,
            ..Default::default()
        };
        config.alpha_config.iterations = 0;
        config.alpha_config.fix_interm_bounds = true;
        config.alpha_config.adaptive_skip = false;
        config.alpha_config.adaptive_skip_pilot = false;

        let bootstrap = compute_graph_bab_bootstrap(&graph, &input, &config, None, deadline)
            .expect("alpha bootstrap should succeed on the residual DAG");

        assert!(bootstrap.root_alpha_state.is_some());
        assert_node_bounds_match_4404(
            &bootstrap.initial_node_bounds,
            &ibp_bounds,
            "residual",
            "#4404 alpha bootstrap should reuse IBP residual bounds",
        );
        assert_node_bounds_match_4404(
            &bootstrap.initial_node_bounds,
            &ibp_bounds,
            "out",
            "#4404 alpha bootstrap should reuse IBP output bounds",
        );
    }
}
