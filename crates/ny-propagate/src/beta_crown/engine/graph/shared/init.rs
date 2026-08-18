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
use crate::network::{GraphAlphaCollectionOutcome, GraphNetworkCrownExt, SpecCrownRequest};
use crate::{AlphaCrownConfig, GraphNetwork, MulBinaryRelaxationMode};

/// Shared graph-BaB bootstrap state produced before root objective evaluation.
#[must_use]
pub(crate) struct GraphBabBootstrap {
    pub(crate) initial_node_bounds: HashMap<String, BoundedTensor>,
    pub(crate) root_alpha_state: Option<GraphAlphaState>,
    pub(crate) alpha_config: AlphaCrownConfig,
    /// The explicit typed cGAN request matched the same exact forward-linear
    /// predicate used by Graph-CROWN Step 1. Its returned map/state can be
    /// evaluated directly; running alpha initialization again would only
    /// repeat the root transaction.
    typed_cgan_root_reusable: bool,
    /// Exact number of optimizer updates represented by the alpha state when a
    /// multi-objective caller retained a DAG-alpha phase checkpoint. `None`
    /// means the bootstrap completed normally. This is scheduling/telemetry
    /// state, never verdict authority.
    pub(crate) phase_cap_optimizer_updates: Option<usize>,
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

/// A local phase checkpoint may continue only under an explicit, strictly
/// live outer verifier deadline.  `None` is not authority to extend work: the
/// multi-objective caller always owns a concrete effective deadline and any
/// future caller must do the same deliberately.
fn phase_checkpoint_authority_live(authority: Option<Instant>, now: Instant) -> bool {
    authority.is_some_and(|deadline| now < deadline)
}

/// Tighten a scheduling deadline with a local alpha cap and report whether the
/// cap, rather than an equal/earlier caller boundary, is the actual limiter.
fn apply_local_phase_cap(
    prior_deadline: Option<Instant>,
    capped_deadline: Instant,
) -> (Option<Instant>, bool) {
    (
        Some(prior_deadline.map_or(capped_deadline, |prior| prior.min(capped_deadline))),
        prior_deadline.is_none_or(|prior| capped_deadline < prior),
    )
}

/// Compute the shared intermediate-bounds bootstrap for graph BaB engines.
pub(crate) fn compute_graph_bab_bootstrap(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<GraphBabBootstrap> {
    compute_graph_bab_bootstrap_with_policy(graph, input, config, engine, deadline, None, false)
}

/// Multi-objective-only bootstrap seam for the dark phase-cap checkpoint
/// policy.  Scalar/GPU callers retain the exact legacy error mapping through
/// [`compute_graph_bab_bootstrap`].
pub(crate) fn compute_graph_bab_bootstrap_with_phase_cap_checkpoint(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
    bootstrap_deadline: Option<Instant>,
    checkpoint_authority_deadline: Option<Instant>,
) -> Result<GraphBabBootstrap> {
    compute_graph_bab_bootstrap_with_policy(
        graph,
        input,
        config,
        engine,
        bootstrap_deadline,
        checkpoint_authority_deadline,
        true,
    )
}

fn compute_graph_bab_bootstrap_with_policy(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    checkpoint_authority_deadline: Option<Instant>,
    allow_phase_cap_checkpoint: bool,
) -> Result<GraphBabBootstrap> {
    // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): bracket the
    // whole warmup + intermediate-bounds collection so the phase is priceable
    // from a log. Gate-off is a cached-bool load — byte-identical output.
    crate::phase_telemetry::phase_marker("graph-bab-bootstrap start");
    config.validate()?;
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "graph BaB bootstrap: deadline exceeded before graph setup".to_string(),
        ));
    }

    let mut local_phase_cap_applied = false;
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
        // Config-driven default (`bab.root_alpha_cap_secs`); the env var below
        // still overrides it. See BetaCrownConfig::root_alpha_cap_secs for the
        // measurement that motivates capping the root on deep conv DAGs.
        //
        // The env override REPLACES the config cap rather than min-composing
        // with it. Both used to min-compose, which made the doc comment above
        // ("the env var below still overrides it") false: `NY_ROOT_ALPHA_CAP_SECS`
        // could only ever TIGHTEN, never widen, so every experiment that raised
        // it silently ran the config's window anyway. Measured on cifar100_2024,
        // whose preset pins `root_alpha_cap_secs: 40`: setting the env to 400
        // still produced a 40 s warmup.
        //
        // The resolved cap is still min-composed with the ledger deadline
        // below, so widening can borrow from the phase budget but never from
        // time the instance does not have.
        let resolved_cap_secs = std::env::var("NY_ROOT_ALPHA_CAP_SECS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .or(config.root_alpha_cap_secs);
        // #root-alpha-tail-reserve: TRIED AND REFUTED (2026-08-06), knob removed.
        //
        // Hypothesis: the cap must cover the alpha ASCENT plus the post-ascent
        // work in the same bootstrap, so stamping the whole cap on the ascent
        // leaves the tail already expired -- and reserving a fraction for the
        // tail would let the bootstrap COMPLETE and thereby reach the dense-head
        // tightener at root.rs:1966.
        //
        // Measured on cifar100_2024 prop_7500 at the official 100 s budget,
        // reserving 0%, 30% and 45% of the window:
        //
        //     frac   alpha loop-exit   bootstrap completed   global remaining
        //     0.00   t=40.0s           no                    46.4s
        //     0.30   t=28.1s           no                    58.3s
        //     0.45   t=22.4s           no                    63.9s
        //
        // The bootstrap fails ~4.6 s after loop-exit in EVERY case, with up to
        // 64 s of global budget still live. So the failure is not a shortage of
        // time for the tail -- the tail is gated on the CAP deadline itself,
        // which the ascent leaves expired by construction because it runs until
        // its deadline. Shortening the ascent cannot help; the tail has to be
        // re-based onto the global deadline instead.
        //
        // (The one configuration where the bootstrap does complete is a cap wide
        // enough that the ascent stops for its OWN reason before the deadline --
        // NY_ROOT_ALPHA_CAP_SECS=120 exits at t=63.2 s with the cap still live.)
        if let Some(cap_secs) = resolved_cap_secs {
            if cap_secs.is_finite() && cap_secs > 0.0 {
                let now = Instant::now();
                let capped = now
                    .checked_add(std::time::Duration::from_secs_f64(cap_secs))
                    .unwrap_or(now);
                (alpha_config.deadline, local_phase_cap_applied) =
                    apply_local_phase_cap(alpha_config.deadline, capped);
            }
        }
        // A fixed-intermediate root bootstrap keeps the reference node map and
        // only consumes the initialized alpha state. With zero requested
        // updates, the DAG optimizer's separate initial output CROWN pass is
        // therefore dead work. Arm the narrow collection-only hint here,
        // after iteration env overrides, rather than changing the public
        // `iterations == 0` bounds-only contract.
        alpha_config.skip_zero_iteration_collection_initial_bound =
            alpha_config.iterations == 0 && alpha_config.fix_interm_bounds;
        alpha_config
    };
    let typed_cgan_root_reusable = if config.use_alpha_crown {
        let exec_order = graph.exec_order()?;
        graph.cgan_complete_crown_ibp_root_eligible(&alpha_config, exec_order)
            || graph.cgan_sparse_target_complete_root_eligible(&alpha_config, exec_order)
    } else {
        false
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
    // default-ON feature lifts the gate for conv graphs so those spatial targets
    // tighten via patches-mode conv CROWN instead of staying pure-IBP loose.
    // `NY_CONV_PATCHES_COLLECT=0` restores the pre-feature behavior exactly.
    // Sound either way: the collector INTERSECTS CROWN with IBP per node and
    // any per-node patches failure falls back to that node's IBP bound.
    let force_conv_patches_collect = crate::util::conv_patches_collect_enabled();
    let large_conv_graph = graph.has_conv_layers()
        && input.len() > LARGE_CONV_INPUT_NUMEL
        && !force_conv_patches_collect;

    let mut phase_cap_optimizer_updates = None;
    let (initial_node_bounds, root_alpha_state) = if force_conv_patches_collect
        && graph.has_conv_layers()
        && input.len() > LARGE_CONV_INPUT_NUMEL
    {
        info!(
            "#conv-patches-collect: lifting the large-conv IBP gate ({} input elements); \
             running CROWN-IBP intermediate collection with patches-start conv targets.",
            input.len()
        );
        match graph
            .collect_crown_ibp_bounds_dag_with_hard_deadline_and_engine(input, deadline, engine)
        {
            Ok(bounds) => (bounds, None),
            Err(e) if is_deadline_exceeded(&e) => {
                // This branch is itself the large-conv path. Falling back to a
                // deadline-free plain-IBP sweep here can strand the verifier
                // long after its authoritative wall-clock budget expires
                // (#4321), exactly like the pre-deadline implementation. Let
                // the root coordinator translate expiry into a sound Timeout
                // verdict instead.
                return Err(e);
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
        let outcome = if allow_phase_cap_checkpoint && local_phase_cap_applied {
            graph.collect_alpha_crown_bounds_dag_with_engine_phase_cap_checkpoint(
                input,
                &alpha_config,
                engine,
            )?
        } else {
            let result =
                graph.collect_alpha_crown_bounds_dag_with_engine(input, &alpha_config, engine)?;
            GraphAlphaCollectionOutcome::Complete(result)
        };
        let (bounds, alpha) = match outcome {
            GraphAlphaCollectionOutcome::Complete(result) => result,
            GraphAlphaCollectionOutcome::PhaseCapCheckpoint {
                result,
                completed_iterations,
                optimizer_updates_completed,
            } => {
                if !phase_checkpoint_authority_live(checkpoint_authority_deadline, Instant::now()) {
                    return Err(NyError::DeadlineExceeded(
                        "DAG alpha phase checkpoint has no live outer verifier authority"
                            .to_string(),
                    ));
                }
                phase_cap_optimizer_updates = Some(optimizer_updates_completed);
                info!(
                    completed_iterations,
                    optimizer_updates_completed,
                    verdict_authority = false,
                    "#root-alpha-phase-checkpoint: consumed completed DAG-alpha artifact; \
                     skipping expired post-loop reference recollection"
                );
                crate::phase_telemetry::phase_marker(
                    "graph-bab-bootstrap phase-cap-checkpoint consumed",
                );
                result
            }
        };
        (bounds, Some(alpha))
    } else if config.use_forward_bounds {
        info!("Computing forward-linear initial bounds...");
        (
            graph.collect_forward_linear_bounds_dag_with_engine_and_deadline(
                input, engine, deadline,
            )?,
            None,
        )
    } else if config.alpha_config.fix_interm_bounds {
        info!("Computing IBP initial bounds...");
        (
            graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline)?,
            None,
        )
    } else {
        info!("Computing CROWN-IBP initial bounds...");
        (
            graph.collect_crown_ibp_bounds_dag_with_deadline_and_engine(input, deadline, engine)?,
            None,
        )
    };

    // #phase-telemetry: end of the warmup+collect phase (error exits above
    // abort the pipeline, so start-without-end in a log reads as "bootstrap
    // did not complete").
    crate::phase_telemetry::phase_marker("graph-bab-bootstrap end");
    Ok(GraphBabBootstrap {
        initial_node_bounds,
        root_alpha_state,
        alpha_config,
        typed_cgan_root_reusable,
        phase_cap_optimizer_updates,
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
        if bootstrap.typed_cgan_root_reusable {
            if let Some(alpha_state) = bootstrap.root_alpha_state.as_ref() {
                let output_bounds =
                    resolve_graph_output_bounds(graph, &bootstrap.initial_node_bounds)?;
                let output_shape = output_bounds.shape().to_vec();
                let identity_spec = ndarray::Array2::<f32>::eye(output_bounds.len());
                let output = SpecCrownRequest::new(graph, input, &identity_spec, engine)
                    .node_bounds(&bootstrap.initial_node_bounds)
                    .alpha_state_opt(Some(alpha_state))
                    .deadline_opt(deadline)
                    .truncate_after_opt(config.crown_backward_layers)
                    .run()?;
                return output.reshape(&output_shape);
            }
        }
        // The caller's explicit root-objective deadline owns this phase. A
        // retained root-alpha checkpoint necessarily embeds its expired LOCAL
        // warmup cap; feeding that stale config directly into the ordinary
        // alpha path would make the certified fallback refuse immediately even
        // while the outer verifier deadline remains live. Borrow on the common
        // equal-deadline path and clone only when rebasing is required.
        let alpha_config = if bootstrap.alpha_config.deadline == deadline {
            std::borrow::Cow::Borrowed(&bootstrap.alpha_config)
        } else {
            let mut rebased = bootstrap.alpha_config.clone();
            rebased.deadline = deadline;
            std::borrow::Cow::Owned(rebased)
        };
        return graph.propagate_alpha_crown_with_config_and_engine(input, &alpha_config, engine);
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
    use std::collections::{BTreeMap, HashMap};
    use std::time::Duration;

    use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

    use super::*;
    use crate::beta_crown::config::BetaCrownConfig;
    use crate::layers::{
        AddLayer, Conv2dLayer, ConvTranspose2dLayer, Layer, LinearLayer, ReLULayer, ReduceSumLayer,
    };
    use crate::network::GraphNode;

    #[test]
    fn phase_checkpoint_requires_explicit_strictly_live_outer_authority() {
        let now = Instant::now();
        assert!(!phase_checkpoint_authority_live(None, now));
        assert!(!phase_checkpoint_authority_live(Some(now), now));
        assert!(!phase_checkpoint_authority_live(
            now.checked_sub(Duration::from_nanos(1)),
            now,
        ));
        assert!(phase_checkpoint_authority_live(
            now.checked_add(Duration::from_nanos(1)),
            now,
        ));
    }

    #[test]
    fn only_a_strictly_earlier_local_cap_authorizes_checkpoint_recovery() {
        let now = Instant::now();
        let earlier = now.checked_sub(Duration::from_secs(1)).expect("earlier");
        let later = now.checked_add(Duration::from_secs(1)).expect("later");

        assert_eq!(apply_local_phase_cap(None, now), (Some(now), true));
        assert_eq!(apply_local_phase_cap(Some(later), now), (Some(now), true));
        assert_eq!(apply_local_phase_cap(Some(now), now), (Some(now), false));
        assert_eq!(
            apply_local_phase_cap(Some(earlier), now),
            (Some(earlier), false)
        );
    }

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

    fn build_typed_cgan_bootstrap_graph() -> (GraphNetwork, BoundedTensor) {
        let transpose_kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -0.5, 0.25, 0.75])
                .expect("transpose kernel");
        let transpose = ConvTranspose2dLayer::with_input_shape(
            transpose_kernel,
            Some(arr1(&[0.1_f32])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("conv transpose");
        let conv = Conv2dLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![0.75_f32]).expect("conv kernel"),
            Some(arr1(&[-0.2_f32])),
            (1, 1),
            (0, 0),
            3,
            3,
        )
        .expect("conv");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "convt",
            Layer::ConvTranspose2d(transpose),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["convt".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "conv",
            Layer::Conv2d(conv),
            vec!["relu".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::ReLU(ReLULayer),
            vec!["conv".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32),
        )
        .expect("input");
        (graph, input)
    }

    #[ntest::timeout(10000)]
    #[test]
    fn typed_cgan_bootstrap_root_output_reuses_map_and_state_without_recollection() {
        use crate::network::CganCompleteCollectionEntryCounter;

        ny_test_utils::env::with_env_edits(|env| {
            for key in [
                "NY_NO_FORWARD_LINEAR_REF",
                "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
                "NY_CROWN_IBP_SPARSE_RELU_ROWS",
                "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
                "NY_CROWN_DEADLINE_CHUNK_SALVAGE",
            ] {
                env.remove(key);
            }
            let (graph, input) = build_typed_cgan_bootstrap_graph();
            let mut config = BetaCrownConfig {
                use_alpha_crown: true,
                ..BetaCrownConfig::default()
            };
            config.alpha_config.iterations = 0;
            config.alpha_config.gradient_method = crate::bounds::GradientMethod::AnalyticChain;
            config.alpha_config.fix_interm_bounds = true;
            config.alpha_config.adaptive_skip = false;
            config.alpha_config.cgan_complete_crown_ibp_root = true;

            let entries = CganCompleteCollectionEntryCounter::start();
            let bootstrap = compute_graph_bab_bootstrap(&graph, &input, &config, None, None)
                .expect("typed bootstrap");
            assert!(
                bootstrap.phase_cap_optimizer_updates.is_none(),
                "the legacy bootstrap wrapper must never mint a phase checkpoint"
            );
            let output =
                compute_graph_root_output_bounds(&graph, &input, &config, None, &bootstrap, None)
                    .expect("root output from bootstrap map/state");
            assert_eq!(output.len(), 9);
            assert_eq!(
                entries.entries(),
                1,
                "root output evaluation must not start the typed transaction again"
            );
        });
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
                (actual_value - expected_value).abs() <= 1e-6 * (1.0 + expected_value.abs()),
                "{label}: node '{node_name}' lower mismatch actual={actual_value}, expected={expected_value}"
            );
        }
        for (actual_value, expected_value) in actual_bounds
            .upper()
            .iter()
            .zip(expected_bounds.upper().iter())
        {
            assert!(
                // Scale-aware: the certified-f64 deadline conv IBP (2026-08-11)
                // is ulp-tighter than the None-arm reference at magnitude ~12;
                // the pin's purpose (bounds REUSED, not recomputed from a
                // different source) survives at relative closeness.
                (actual_value - expected_value).abs() <= 1e-6 * (1.0 + expected_value.abs()),
                "{label}: node '{node_name}' upper mismatch actual={actual_value}, expected={expected_value}"
            );
        }
    }

    fn bound_map_bits(
        map: &HashMap<String, BoundedTensor>,
    ) -> BTreeMap<String, (Vec<u32>, Vec<u32>)> {
        map.iter()
            .map(|(name, bound)| {
                (
                    name.clone(),
                    (
                        bound.lower().iter().map(|value| value.to_bits()).collect(),
                        bound.upper().iter().map(|value| value.to_bits()).collect(),
                    ),
                )
            })
            .collect()
    }

    fn alpha_vector_map_bits(map: &BTreeMap<String, Array1<f32>>) -> BTreeMap<String, Vec<u32>> {
        map.iter()
            .map(|(name, values)| {
                (
                    name.clone(),
                    values.iter().map(|value| value.to_bits()).collect(),
                )
            })
            .collect()
    }

    fn alpha_matrix_map_bits(
        map: &BTreeMap<String, Array2<f32>>,
    ) -> BTreeMap<String, (Vec<usize>, Vec<u32>)> {
        map.iter()
            .map(|(name, values)| {
                (
                    name.clone(),
                    (
                        values.shape().to_vec(),
                        values.iter().map(|value| value.to_bits()).collect(),
                    ),
                )
            })
            .collect()
    }

    fn assert_alpha_state_bits_equal(actual: &GraphAlphaState, expected: &GraphAlphaState) {
        for (label, actual_bits, expected_bits) in [
            (
                "alphas",
                alpha_vector_map_bits(&actual.alphas),
                alpha_vector_map_bits(&expected.alphas),
            ),
            (
                "alphas_upper",
                alpha_vector_map_bits(&actual.alphas_upper),
                alpha_vector_map_bits(&expected.alphas_upper),
            ),
            (
                "velocity",
                alpha_vector_map_bits(&actual.velocity),
                alpha_vector_map_bits(&expected.velocity),
            ),
            (
                "adam_m",
                alpha_vector_map_bits(&actual.adam_m),
                alpha_vector_map_bits(&expected.adam_m),
            ),
            (
                "adam_v",
                alpha_vector_map_bits(&actual.adam_v),
                alpha_vector_map_bits(&expected.adam_v),
            ),
            (
                "velocity_upper",
                alpha_vector_map_bits(&actual.velocity_upper),
                alpha_vector_map_bits(&expected.velocity_upper),
            ),
            (
                "adam_m_upper",
                alpha_vector_map_bits(&actual.adam_m_upper),
                alpha_vector_map_bits(&expected.adam_m_upper),
            ),
            (
                "adam_v_upper",
                alpha_vector_map_bits(&actual.adam_v_upper),
                alpha_vector_map_bits(&expected.adam_v_upper),
            ),
        ] {
            assert_eq!(
                actual_bits, expected_bits,
                "{label} must remain bit-identical"
            );
        }
        assert_eq!(actual.unstable_mask, expected.unstable_mask);
        assert_eq!(actual.spatial_shapes, expected.spatial_shapes);
        assert_eq!(actual.spec_slot_rows, expected.spec_slot_rows);
        assert_eq!(
            alpha_matrix_map_bits(&actual.spec_deltas),
            alpha_matrix_map_bits(&expected.spec_deltas)
        );
        assert_eq!(
            alpha_matrix_map_bits(&actual.spec_adam_m),
            alpha_matrix_map_bits(&expected.spec_adam_m)
        );
        assert_eq!(
            alpha_matrix_map_bits(&actual.spec_adam_v),
            alpha_matrix_map_bits(&expected.spec_adam_v)
        );
        assert!(actual.monotone_s_shaped_alphas.is_empty());
        assert!(expected.monotone_s_shaped_alphas.is_empty());
        assert!(actual.sqrt_alphas.is_empty());
        assert!(expected.sqrt_alphas.is_empty());
        assert!(actual.reciprocal_alphas.is_empty());
        assert!(expected.reciprocal_alphas.is_empty());
        assert_eq!(
            *actual.gpu_suffix_ineligible.read().expect("actual cache"),
            *expected
                .gpu_suffix_ineligible
                .read()
                .expect("expected cache")
        );
    }

    #[ntest::timeout(30000)]
    #[test]
    fn checkpoint_policy_normal_completion_matches_legacy_bootstrap_bits() {
        let (graph, input) = build_residual_dag_4404();
        let mut config = BetaCrownConfig {
            use_alpha_crown: true,
            root_alpha_cap_secs: Some(10.0),
            ..BetaCrownConfig::default()
        };
        config.alpha_config.iterations = 1;
        config.alpha_config.gradient_method = crate::bounds::GradientMethod::AnalyticChain;
        config.alpha_config.fix_interm_bounds = true;
        config.alpha_config.adaptive_skip = false;
        config.alpha_config.adaptive_skip_pilot = false;

        let now = Instant::now();
        let bootstrap_deadline = now.checked_add(Duration::from_secs(30));
        let authority_deadline = now.checked_add(Duration::from_mins(1));
        let legacy = compute_graph_bab_bootstrap(&graph, &input, &config, None, bootstrap_deadline)
            .expect("legacy bootstrap completes before the local cap");
        let enabled = compute_graph_bab_bootstrap_with_phase_cap_checkpoint(
            &graph,
            &input,
            &config,
            None,
            bootstrap_deadline,
            authority_deadline,
        )
        .expect("checkpoint-policy bootstrap completes before the local cap");

        assert!(legacy.phase_cap_optimizer_updates.is_none());
        assert!(enabled.phase_cap_optimizer_updates.is_none());
        assert_eq!(
            bound_map_bits(&enabled.initial_node_bounds),
            bound_map_bits(&legacy.initial_node_bounds),
            "enabled-policy normal completion must preserve every certified node bound bit"
        );
        assert_alpha_state_bits_equal(
            enabled
                .root_alpha_state
                .as_ref()
                .expect("enabled alpha state"),
            legacy
                .root_alpha_state
                .as_ref()
                .expect("legacy alpha state"),
        );
    }

    #[ntest::timeout(30000)]
    #[test]
    fn root_output_alpha_path_honors_explicit_rebased_deadline() {
        let (graph, input) = build_test_graph();
        let mut config = BetaCrownConfig {
            use_alpha_crown: true,
            ..BetaCrownConfig::default()
        };
        config.alpha_config.iterations = 0;
        config.alpha_config.fix_interm_bounds = true;
        config.alpha_config.adaptive_skip = false;
        config.alpha_config.adaptive_skip_pilot = false;

        let mut bootstrap = compute_graph_bab_bootstrap(&graph, &input, &config, None, None)
            .expect("bootstrap without a deadline");
        bootstrap.alpha_config.deadline = Instant::now().checked_sub(Duration::from_secs(1));
        let live = Instant::now().checked_add(Duration::from_secs(10));
        let output =
            compute_graph_root_output_bounds(&graph, &input, &config, None, &bootstrap, live)
                .expect("explicit live root deadline must replace the expired warmup cap");
        assert!(output
            .lower()
            .iter()
            .chain(output.upper().iter())
            .all(|value| value.is_finite()));
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
        assert!(bootstrap.phase_cap_optimizer_updates.is_none());
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
        assert!(
            bootstrap.phase_cap_optimizer_updates.is_none(),
            "legacy alpha bootstrap behavior must remain complete-only"
        );
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
    fn test_compute_graph_bab_bootstrap_forward_path_preserves_expired_deadline_4260() {
        // The bootstrap has no precollected output map at entry. Once its
        // authority is already expired, starting a fresh plain-IBP sweep would
        // violate the same hard wall-clock contract as continuing forward
        // linear propagation.
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

        let error =
            match compute_graph_bab_bootstrap(&graph, &input, &config, None, expired_deadline) {
                Err(error) => error,
                Ok(_) => panic!("expired bootstrap must not launch a fresh IBP fallback"),
            };
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
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
        assert!(
            bootstrap
                .alpha_config
                .skip_zero_iteration_collection_initial_bound,
            "zero-update fixed-intermediate root bootstrap must arm the narrow collection skip"
        );
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
