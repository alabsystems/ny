// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pre-loop setup helpers for spec-guided CROWN backward propagation.
//!
//! Isolates the "prepare state for the backward loop" responsibility from the
//! loop itself: intermediate bounds collection, output node resolution, and
//! spec-column contract validation. Split from `core.rs` as part of #3960.

use crate::network::core::{GraphNetwork, GraphNode, GraphTargetShapeContract};

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::info;

/// Collect intermediate node bounds when no pre-computed bounds are provided.
///
/// Selects between per-node CROWN-IBP (O(N²) backward per graph model) and
/// simple IBP forward collection based on graph structure heuristics.
pub(crate) fn collect_intermediate_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    deadline: Option<Instant>,
    engine: Option<&dyn GemmEngine>,
) -> Result<std::collections::HashMap<String, BoundedTensor>> {
    // Image forward-linear intermediates (#vnncomp-image-forward-linear):
    // same shared policy and disable flags as the matching propagation path.
    // This includes the default-enabled sequential ConvTranspose2d+Conv2d
    // cGAN surface as well as the historical non-sequential Conv-DAG route.
    // Fail closed to the existing selection on any collector refusal.
    if graph.should_collect_forward_linear_intermediate_reference() {
        match graph.collect_forward_linear_bounds_dag_cached(input, engine, deadline) {
            Ok(bounds) => {
                info!(
                    "GraphNetwork spec-CROWN: forward-linear intermediates (image graph, cached)"
                );
                return Ok((*bounds).clone());
            }
            Err(error @ NyError::DeadlineExceeded(_))
                if graph.forward_linear_deadline_fallback_to_ibp =>
            {
                // #cgan-forward-deadline-ibp: mirror alpha-reference setup.
                // A plain IBP map is a certified (possibly looser)
                // intermediate enclosure, so opting out of a doomed
                // CROWN-IBP endgame changes schedule/tightness, never proof
                // authority.
                info!(
                    "GraphNetwork spec-CROWN: forward-linear intermediates unavailable \
                     ({error}); using direct IBP deadline fallback"
                );
                return graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline);
            }
            Err(
                error @ (NyError::UnsupportedOp(_)
                | NyError::UnsupportedConfiguration(_)
                | NyError::DeadlineExceeded(_)
                | NyError::ShapeMismatch { .. }
                | NyError::CpuMemoryExceeded { .. }),
            ) => {
                info!(
                    "GraphNetwork spec-CROWN: forward-linear intermediates unavailable \
                     ({error}); falling back (fail-closed)"
                );
            }
            Err(error) => return Err(error),
        }
    }

    let use_crown_ibp = graph.should_use_crown_ibp_intermediates();
    let use_per_node_crown_ibp = graph.should_collect_per_node_crown_ibp_intermediates();
    if use_per_node_crown_ibp {
        // Pass deadline to CROWN-IBP collection so the O(N²) per-node backward
        // passes respect the verification timeout. Without this, large CNN DAGs
        // (e.g., metaroom 6cnn_ry_49_8) run CROWN-IBP unbounded. Fixed in #3397.
        Ok(graph
            .collect_crown_ibp_bounds_dag_with_status_and_deadline(input, deadline, engine)?
            .bounds)
    } else {
        if use_crown_ibp {
            info!(
                "GraphNetwork spec-CROWN: {} nodes exceeds per-node CROWN-IBP threshold {}, using IBP intermediates for final backward pass",
                graph.nodes.len(),
                crate::network::core::graph::CROWN_IBP_PER_NODE_THRESHOLD
            );
        }
        // Thread the deadline (#4321): the IBP intermediate sweep over a deep conv
        // DAG can overrun the verifier timeout. collect_node_bounds_core checks the
        // deadline between nodes.
        graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline)
    }
}

/// Resolve the output node name and validate the spec-column contract.
///
/// Returns the output node name after verifying that the spec matrix columns
/// match the output node's shape.
pub(super) fn resolve_output_contract<'a>(
    graph: &'a GraphNetwork,
    exec_order: &'a [String],
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    spec_output_dim: usize,
) -> Result<&'a str> {
    let output_node_name = if graph.output_node.is_empty() {
        exec_order
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
            .as_str()
    } else {
        &graph.output_node
    };

    let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
        NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
    })?;
    let output_contract = GraphTargetShapeContract::from_bounds(output_node_name, output_bounds);
    output_contract.validate_spec_cols(spec_output_dim, "Spec-guided CROWN spec columns")?;

    Ok(output_node_name)
}

/// Build exec-order-indexed node references for the hot backward loop.
pub(super) fn collect_nodes_by_idx<'a>(
    graph: &'a GraphNetwork,
    exec_order: &[String],
) -> Result<Vec<&'a GraphNode>> {
    exec_order
        .iter()
        .map(|name| {
            graph
                .nodes
                .get(name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {name}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{
        BatchNormLayer, Conv2dLayer, ConvTranspose2dLayer, Layer, ReLULayer, SigmoidLayer,
    };
    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use std::collections::HashMap;

    fn sequential_cgan_fixture(unsupported_tail: bool) -> (GraphNetwork, BoundedTensor) {
        let conv_transpose = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_shape_vec(
                IxDyn(&[1, 2, 2, 2]),
                vec![0.7, -0.4, 0.3, 0.8, -0.2, 0.6, -0.9, 0.5],
            )
            .expect("valid ConvTranspose2d kernel"),
            Some(Array1::from_vec(vec![0.1, -0.15])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("valid ConvTranspose2d layer");
        let batch_norm = BatchNormLayer::from_scale_bias(
            Array1::from_vec(vec![1.2, -0.7]).into_dyn(),
            Array1::from_vec(vec![-0.05, 0.2]).into_dyn(),
        )
        .expect("valid BatchNorm layer");
        let conv = Conv2dLayer::with_input_shape(
            ArrayD::from_shape_vec(
                IxDyn(&[1, 2, 2, 2]),
                vec![0.4, -0.8, 0.5, 0.2, -0.3, 0.7, -0.6, 0.9],
            )
            .expect("valid Conv2d kernel"),
            Some(Array1::from_vec(vec![0.03])),
            (1, 1),
            (0, 0),
            3,
            3,
        )
        .expect("valid Conv2d layer");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "conv_transpose",
            Layer::ConvTranspose2d(conv_transpose),
        ));
        graph.add_node(GraphNode::new(
            "batch_norm",
            Layer::BatchNorm(batch_norm),
            vec!["conv_transpose".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["batch_norm".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "conv",
            Layer::Conv2d(conv),
            vec!["relu".to_string()],
        ));
        if unsupported_tail {
            graph.add_node(GraphNode::new(
                "sigmoid",
                Layer::Sigmoid(SigmoidLayer::new()),
                vec!["conv".to_string()],
            ));
            graph.set_output("sigmoid");
        } else {
            graph.set_output("conv");
        }

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0, -0.4, 0.1, -0.8])
                .expect("valid input lower"),
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.6, 0.9, 0.75, 0.35])
                .expect("valid input upper"),
        )
        .expect("valid bounded input");
        (graph, input)
    }

    fn assert_bound_maps_eq(
        actual: &HashMap<String, BoundedTensor>,
        expected: &HashMap<String, BoundedTensor>,
        context: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{context}: map size");
        for (name, expected_bounds) in expected {
            let actual_bounds = actual
                .get(name)
                .unwrap_or_else(|| panic!("{context}: missing node '{name}'"));
            assert_eq!(
                actual_bounds.lower(),
                expected_bounds.lower(),
                "{context}: node '{name}' lower"
            );
            assert_eq!(
                actual_bounds.upper(),
                expected_bounds.upper(),
                "{context}: node '{name}' upper"
            );
        }
    }

    #[ntest::timeout(30000)]
    #[test]
    fn sequential_cgan_forward_linear_route_is_default_enabled() {
        let (graph, input) = sequential_cgan_fixture(false);
        let order = graph.exec_order().expect("valid execution order");
        assert!(
            graph.is_sequential_graph(order),
            "fixture must exercise the sequential route"
        );

        crate::tests::with_env_edits(|env| {
            env.remove("NY_NO_FORWARD_LINEAR_REF");
            env.remove("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF");
            assert!(
                graph.should_collect_forward_linear_intermediate_reference(),
                "default policy must admit a sequential ConvTranspose2d+Conv2d chain"
            );
            assert!(
                graph.should_collect_forward_linear_image_reference(),
                "root/image policy must agree for the sequential cGAN surface"
            );

            let expected = graph
                .collect_forward_linear_bounds_dag_cached(&input, None, None)
                .expect("forward-linear reference collection");
            let actual =
                collect_intermediate_bounds(&graph, &input, None, None).expect("setup collection");
            assert_bound_maps_eq(&actual, &expected, "default-enabled forward-linear route");
        });
    }

    #[ntest::timeout(30000)]
    #[test]
    fn sequential_cgan_kill_switches_fall_back_to_crown_ibp() {
        let (graph, input) = sequential_cgan_fixture(false);

        crate::tests::with_env_edits(|env| {
            env.remove("NY_NO_FORWARD_LINEAR_REF");
            env.set("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "1");
            env.set("NY_DISABLE_CROWN_COLLECTION_CACHE", "1");
            assert!(
                !graph.should_collect_forward_linear_intermediate_reference(),
                "ConvTranspose-specific kill switch must disable the sequential route"
            );
            assert!(
                !graph.should_collect_forward_linear_image_reference(),
                "root/image policy must honor the ConvTranspose-specific kill switch"
            );

            let expected = graph
                .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, None, None)
                .expect("CROWN-IBP fallback")
                .bounds;
            let conv_transpose_disabled =
                collect_intermediate_bounds(&graph, &input, None, None).expect("setup fallback");
            assert_bound_maps_eq(
                &conv_transpose_disabled,
                &expected,
                "ConvTranspose kill-switch fallback",
            );

            env.set("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "0");
            env.set("NY_NO_FORWARD_LINEAR_REF", "1");
            assert!(
                !graph.should_collect_forward_linear_intermediate_reference(),
                "shared reference kill switch must disable the sequential route"
            );
            assert!(
                !graph.should_collect_forward_linear_image_reference(),
                "root/image policy must honor the shared reference kill switch"
            );
            let all_references_disabled =
                collect_intermediate_bounds(&graph, &input, None, None).expect("setup fallback");
            assert_bound_maps_eq(
                &all_references_disabled,
                &expected,
                "shared kill-switch fallback",
            );
        });
    }

    /// A sub-30-second deadline refuses a cold forward-linear image map before
    /// work begins. The cGAN opt-in must use plain IBP directly, preserve the
    /// certified specification enclosure and its threshold verdict, and avoid
    /// entering the historical CROWN-IBP collection route.
    #[ntest::timeout(30000)]
    #[test]
    fn sequential_cgan_deadline_refusal_routes_directly_to_sound_ibp() {
        let spec = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0_f32, -1.0, 0.0, 0.0, //
                0.0, 1.0, -1.0, 0.0, //
                -0.5, 0.0, 0.25, 1.0,
            ],
        )
        .expect("valid three-row output specification");

        crate::tests::with_env_edits(|env| {
            for key in [
                "NY_NO_FORWARD_LINEAR_REF",
                "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
                "NY_DISABLE_CROWN_COLLECTION_CACHE",
                "NY_CROWN_SERVE_TRUNCATED_CACHE",
            ] {
                env.remove(key);
            }

            let (mut graph, input) = sequential_cgan_fixture(false);
            graph.set_forward_linear_deadline_fallback_to_ibp(true);
            let expected_nodes = graph
                .collect_node_bounds_with_engine(&input, None)
                .expect("plain IBP node bounds");
            let actual_nodes = collect_intermediate_bounds(
                &graph,
                &input,
                Some(Instant::now() + std::time::Duration::from_secs(5)),
                None,
            )
            .expect("direct deadline fallback");
            // The finite-deadline arm now runs the certified-f64 conv IBP
            // (tighter than the engine route's f32 gamma*S widening — the
            // 2026-08-11 cgan floor fix), so bit-equality with the None-arm
            // reference no longer holds BY DESIGN. The route claim stays:
            // same nodes, and the deadline arm is NEVER LOOSER per element.
            assert_eq!(
                actual_nodes.len(),
                expected_nodes.len(),
                "deadline-refusal direct IBP route: map size"
            );
            for (name, exp) in &expected_nodes {
                let act = actual_nodes
                    .get(name)
                    .unwrap_or_else(|| panic!("route: missing node '{name}'"));
                for (a, e) in act.lower().iter().zip(exp.lower().iter()) {
                    assert!(
                        a >= e,
                        "node '{name}': deadline-arm lower {a} looser than reference {e}"
                    );
                }
                for (a, e) in act.upper().iter().zip(exp.upper().iter()) {
                    assert!(
                        a <= e,
                        "node '{name}': deadline-arm upper {a} looser than reference {e}"
                    );
                }
            }
            assert_eq!(
                graph.crown_ibp_collection_cache_hits(),
                0,
                "the direct route must not consult the CROWN-IBP collection cache"
            );

            let actual_spec = graph
                .propagate_crown_with_specs_fallback_ibp(&input, &spec, &actual_nodes, "conv")
                .expect("deadline-fallback specification enclosure");
            let expected_spec = graph
                .propagate_crown_with_specs_fallback_ibp(&input, &spec, &expected_nodes, "conv")
                .expect("plain-IBP specification enclosure");
            // Tighter node bounds feed a tighter spec enclosure (never
            // looser); the corner sweep below independently pins soundness.
            for (a, e) in actual_spec.lower().iter().zip(expected_spec.lower().iter()) {
                assert!(a >= e, "spec lower {a} looser than reference {e}");
            }
            for (a, e) in actual_spec.upper().iter().zip(expected_spec.upper().iter()) {
                assert!(a <= e, "spec upper {a} looser than reference {e}");
            }

            // Degenerate-box propagation independently evaluates every input
            // corner plus the midpoint. Every value must remain enclosed by
            // the routed fallback result.
            let input_lower: Vec<f32> = input.lower().iter().copied().collect();
            let input_upper: Vec<f32> = input.upper().iter().copied().collect();
            let corner_count = 1usize << input_lower.len();
            for sample in 0..=corner_count {
                let values: Vec<f32> = input_lower
                    .iter()
                    .zip(input_upper.iter())
                    .enumerate()
                    .map(|(index, (&lower, &upper))| {
                        if sample == corner_count {
                            f32::midpoint(lower, upper)
                        } else if sample & (1usize << index) == 0 {
                            lower
                        } else {
                            upper
                        }
                    })
                    .collect();
                let point = BoundedTensor::concrete(
                    ArrayD::from_shape_vec(IxDyn(input.shape()), values)
                        .expect("point shape matches input"),
                )
                .expect("valid concrete input");
                let point_nodes = graph
                    .collect_node_bounds_with_engine(&point, None)
                    .expect("concrete point evaluation");
                let point_spec = graph
                    .propagate_crown_with_specs_fallback_ibp(&point, &spec, &point_nodes, "conv")
                    .expect("concrete specification evaluation");
                for row in 0..spec.nrows() {
                    let value = f64::midpoint(
                        point_spec.lower()[row] as f64,
                        point_spec.upper()[row] as f64,
                    );
                    let slack = 1.0e-4 * (1.0 + value.abs());
                    assert!(
                        actual_spec.lower()[row] as f64 - slack <= value
                            && value <= actual_spec.upper()[row] as f64 + slack,
                        "deadline fallback row {row} excludes sample {sample}: \
                         value={value}, bounds=[{}, {}]",
                        actual_spec.lower()[row],
                        actual_spec.upper()[row],
                    );
                }
            }

            let thresholds = [0.0_f32, -0.25, 0.5];
            let verdict = |bounds: &BoundedTensor| {
                bounds
                    .lower()
                    .iter()
                    .zip(thresholds)
                    .all(|(&lower, threshold)| lower > threshold)
            };
            assert_eq!(
                verdict(&actual_spec),
                verdict(&expected_spec),
                "routing must preserve the direct-IBP threshold verdict"
            );

            // A fresh default-off graph must still enter and cache the legacy
            // CROWN-IBP fallback when the forward-linear collector refuses.
            // The refusal is FORCED via the ConvTranspose kill switch: the
            // certified-f64 deadline conv IBP (2026-08-11 cgan floor fix) made
            // the collector fast enough that the fixture's old "doomed
            // endgame" deadline no longer expires organically — a strictly
            // better outcome; the historical-route pin keeps its meaning
            // through the explicit disable instead.
            env.set("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "1");
            let (historical_graph, historical_input) = sequential_cgan_fixture(false);
            // deadline=None: the route pin needs determinism, not a timer —
            // a live Instant differs per call and can perturb the collection
            // coverage descriptor the cache keys on.
            for _ in 0..2 {
                collect_intermediate_bounds(&historical_graph, &historical_input, None, None)
                    .expect("historical CROWN-IBP fallback");
            }
            env.remove("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF");
            assert!(
                historical_graph.crown_ibp_collection_cache_hits() > 0,
                "default-off must retain the historical CROWN-IBP route"
            );
        });
    }

    /// End-to-end moat for the default-enabled sequential cGAN route.
    ///
    /// The default result must be exactly the certified forward-linear
    /// C-margin after the production output-box intersection. Both independent
    /// reference kill switches must take the full fallback propagation path,
    /// retain a sound enclosure, and preserve the bound-derived verdict.
    #[ntest::timeout(30000)]
    #[test]
    fn sequential_cgan_spec_propagation_preserves_enclosure_and_verdict_across_kill_switches() {
        let (graph, input) = sequential_cgan_fixture(false);
        let spec = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0_f32, -1.0, 0.0, 0.0, //
                0.0, 1.0, -1.0, 0.0, //
                -0.5, 0.0, 0.25, 1.0,
            ],
        )
        .expect("valid three-row output specification");

        crate::tests::with_env_edits(|env| {
            for key in [
                "NY_NO_FORWARD_LINEAR_REF",
                "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
                "NY_SPEC_ROOT_MARGIN",
                "NY_FORWARD_LINEAR_SEQ_CONV_REF",
            ] {
                env.remove(key);
            }
            // Isolate the route under test: the fixed forward-linear C-margin
            // remains default-enabled, while unrelated root candidates cannot
            // contribute another enclosure to the final intersection.
            env.set("NY_SPEC_ROOT_GPU", "0");
            env.set("NY_SPEC_ROOT_ALPHA", "0");
            env.set("NY_DISABLE_CROWN_COLLECTION_CACHE", "1");

            let forward_reference = collect_intermediate_bounds(&graph, &input, None, None)
                .expect("default forward-linear intermediate reference");
            let raw_margin = graph
                .forward_linear_spec_margin_bounds(&input, &spec, None, None)
                .expect("certified forward-linear C-margin");
            let reference_projection = graph
                .propagate_crown_with_specs_fallback_ibp(&input, &spec, &forward_reference, "conv")
                .expect("output-box specification enclosure");
            let expected_default = crate::network::tighten_crown_output(
                raw_margin.clone(),
                &reference_projection,
                "sequential cGAN route moat",
            )
            .expect("C-margin/output-box intersection");
            let default_route = graph
                .propagate_crown_with_specs_and_engine(&input, &spec, None)
                .expect("default production spec propagation");
            assert_eq!(
                default_route.lower(),
                expected_default.lower(),
                "default lower bounds must come through the production C-margin intersection"
            );
            assert_eq!(
                default_route.upper(),
                expected_default.upper(),
                "default upper bounds must come through the production C-margin intersection"
            );

            env.set("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "1");
            let conv_transpose_killed = graph
                .propagate_crown_with_specs_and_engine(&input, &spec, None)
                .expect("ConvTranspose kill-switch fallback propagation");

            env.remove("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF");
            env.set("NY_NO_FORWARD_LINEAR_REF", "1");
            let shared_reference_killed = graph
                .propagate_crown_with_specs_and_engine(&input, &spec, None)
                .expect("shared reference kill-switch fallback propagation");

            assert_eq!(
                conv_transpose_killed.lower(),
                shared_reference_killed.lower(),
                "both kill switches must select the same fallback lower enclosure"
            );
            assert_eq!(
                conv_transpose_killed.upper(),
                shared_reference_killed.upper(),
                "both kill switches must select the same fallback upper enclosure"
            );

            let plain_ibp_nodes = graph
                .collect_node_bounds_with_engine(&input, None)
                .expect("plain IBP node bounds");
            let plain_ibp_spec = graph
                .propagate_crown_with_specs_fallback_ibp(&input, &spec, &plain_ibp_nodes, "conv")
                .expect("plain IBP specification enclosure");
            for (label, bounds) in [
                ("default", &default_route),
                ("ConvTranspose kill switch", &conv_transpose_killed),
                ("shared kill switch", &shared_reference_killed),
            ] {
                for row in 0..spec.nrows() {
                    assert!(
                        bounds.lower()[row].is_finite()
                            && bounds.upper()[row].is_finite()
                            && bounds.lower()[row] <= bounds.upper()[row],
                        "{label} row {row} must be a finite, ordered enclosure"
                    );
                    assert!(
                        bounds.lower()[row] >= plain_ibp_spec.lower()[row]
                            && bounds.upper()[row] <= plain_ibp_spec.upper()[row],
                        "{label} row {row} must retain the production final IBP intersection: \
                         got [{}, {}], IBP [{}, {}]",
                        bounds.lower()[row],
                        bounds.upper()[row],
                        plain_ibp_spec.lower()[row],
                        plain_ibp_spec.upper()[row],
                    );
                }
            }

            // Exercise the enclosures against every input-box corner and its
            // midpoint. Degenerate-box propagation is an independent concrete
            // evaluator for this deterministic fixture.
            let input_lower: Vec<f32> = input.lower().iter().copied().collect();
            let input_upper: Vec<f32> = input.upper().iter().copied().collect();
            let corner_count = 1usize << input_lower.len();
            for sample in 0..=corner_count {
                let values: Vec<f32> = input_lower
                    .iter()
                    .zip(input_upper.iter())
                    .enumerate()
                    .map(|(index, (&lower, &upper))| {
                        if sample == corner_count {
                            f32::midpoint(lower, upper)
                        } else if sample & (1usize << index) == 0 {
                            lower
                        } else {
                            upper
                        }
                    })
                    .collect();
                let point = BoundedTensor::concrete(
                    ArrayD::from_shape_vec(IxDyn(input.shape()), values)
                        .expect("point shape matches the input"),
                )
                .expect("valid concrete input");
                let point_nodes = graph
                    .collect_node_bounds_with_engine(&point, None)
                    .expect("concrete point evaluation");
                let point_spec = graph
                    .propagate_crown_with_specs_fallback_ibp(&point, &spec, &point_nodes, "conv")
                    .expect("concrete specification evaluation");

                for row in 0..spec.nrows() {
                    let value = f64::midpoint(
                        point_spec.lower()[row] as f64,
                        point_spec.upper()[row] as f64,
                    );
                    let slack = 1.0e-4 * (1.0 + value.abs());
                    for (label, bounds) in [
                        ("raw C-margin", &raw_margin),
                        ("default", &default_route),
                        ("ConvTranspose kill switch", &conv_transpose_killed),
                        ("shared kill switch", &shared_reference_killed),
                    ] {
                        assert!(
                            bounds.lower()[row] as f64 - slack <= value
                                && value <= bounds.upper()[row] as f64 + slack,
                            "{label} row {row} excludes concrete sample {sample}: \
                             value={value}, bounds=[{}, {}]",
                            bounds.lower()[row],
                            bounds.upper()[row],
                        );
                    }
                }
            }

            let verified_at_zero =
                |bounds: &BoundedTensor| bounds.lower().iter().all(|&lower| lower > 0.0);
            assert_eq!(
                verified_at_zero(&default_route),
                verified_at_zero(&conv_transpose_killed),
                "ConvTranspose kill switch must preserve the zero-threshold verdict"
            );
            assert_eq!(
                verified_at_zero(&default_route),
                verified_at_zero(&shared_reference_killed),
                "shared reference kill switch must preserve the zero-threshold verdict"
            );
        });
    }

    #[ntest::timeout(30000)]
    #[test]
    fn sequential_cgan_unsupported_op_falls_back_to_crown_ibp() {
        let (mut graph, input) = sequential_cgan_fixture(true);
        graph.set_forward_linear_deadline_fallback_to_ibp(true);

        crate::tests::with_env_edits(|env| {
            env.remove("NY_NO_FORWARD_LINEAR_REF");
            env.remove("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF");
            env.set("NY_DISABLE_CROWN_COLLECTION_CACHE", "1");
            assert!(
                graph.should_collect_forward_linear_intermediate_reference(),
                "shape policy should attempt the image route before its op allowlist refuses"
            );
            assert!(
                graph.should_collect_forward_linear_image_reference(),
                "unsupported ops must be refused by the collector, not eligibility"
            );

            let error = graph
                .collect_forward_linear_bounds_dag_with_engine(&input, None)
                .expect_err("Sigmoid is outside the certified image forward-linear surface");
            assert!(
                matches!(error, NyError::UnsupportedConfiguration(_)),
                "unsupported image op must refuse fail-closed, got {error:?}"
            );

            let expected = graph
                .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, None, None)
                .expect("CROWN-IBP fallback")
                .bounds;
            let actual =
                collect_intermediate_bounds(&graph, &input, None, None).expect("setup fallback");
            assert_bound_maps_eq(&actual, &expected, "unsupported-op fallback");
        });
    }
}
