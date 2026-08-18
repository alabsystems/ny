// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn scalar_component_ibp_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    label: &str,
) -> (f32, f32) {
    let ibp = graph
        .propagate_ibp(input)
        .unwrap_or_else(|_| panic!("{label} pure IBP should succeed"));
    let flat = ibp.flatten();
    assert_eq!(
        flat.lower().len(),
        1,
        "{label} pure IBP should stay scalar, got shape {:?}",
        flat.lower().shape()
    );
    let lower = flat.lower()[0];
    let upper = flat.upper()[0];
    assert!(
        lower.is_finite() && upper.is_finite(),
        "{label} pure IBP bounds should be finite: [{lower}, {upper}]"
    );
    (lower, upper)
}

fn assert_scalar_component_crown_exercises_path_without_loosening(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    label: &str,
) -> ((f32, f32), (f32, f32), bool) {
    let ibp = scalar_component_ibp_bounds(graph, input, &format!("{label} + pure IBP"));
    let ibp_bounds = graph
        .collect_node_bounds(input)
        .unwrap_or_else(|_| panic!("{label} IBP node-bound collection should succeed"));

    // Diagnostic: use _and_linear variant to check if CROWN backward produced
    // linear bounds (Some) or fell back entirely to IBP (None). #3596
    let spec = ndarray::arr2(&[[1.0_f32]]);
    let (crown_output, linear_bounds) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear(input, &spec, None, &ibp_bounds)
        .unwrap_or_else(|e| panic!("{label} spec-guided CROWN should succeed: {e}"));
    let crown_backward_ran = linear_bounds.is_some();
    let a_matrix_nonzero = linear_bounds
        .as_ref()
        .map(|lb| lb.lower_a().iter().any(|&v| v != 0.0) || lb.upper_a().iter().any(|&v| v != 0.0))
        .unwrap_or(false);
    eprintln!(
        "{label} CROWN diagnostic: backward_ran={crown_backward_ran}, a_matrix_nonzero={a_matrix_nonzero}"
    );

    let flat = crown_output.flatten();
    let crown_bounds = (flat.lower()[0], flat.upper()[0]);

    let scale = ibp
        .0
        .abs()
        .max(ibp.1.abs())
        .max(crown_bounds.0.abs())
        .max(crown_bounds.1.abs())
        .max(1.0);
    let bound_tol = 1e-6 * scale;
    assert!(
        crown_bounds.0 >= ibp.0 - bound_tol,
        "{label} lower should not loosen with CROWN output over IBP intermediates: CROWN={}, IBP={}",
        crown_bounds.0,
        ibp.0
    );
    assert!(
        crown_bounds.1 <= ibp.1 + bound_tol,
        "{label} upper should not loosen with CROWN output over IBP intermediates: CROWN={}, IBP={}",
        crown_bounds.1,
        ibp.1
    );
    let ibp_width = scalar_width(ibp.0, ibp.1);
    let crown_width = scalar_width(crown_bounds.0, crown_bounds.1);
    let width_tol = 1e-6 * ibp_width.max(crown_width).max(1.0);
    let strict_improvement = crown_width < ibp_width - width_tol;
    eprintln!(
        "{label} output bounds: pure_ibp=[{}, {}] width={ibp_width}; crown=[{}, {}] width={crown_width}; strict_improvement={strict_improvement}",
        ibp.0,
        ibp.1,
        crown_bounds.0,
        crown_bounds.1,
    );

    (ibp, crown_bounds, strict_improvement)
}

fn collect_layer_type_counts(graph: &GraphNetwork) -> HashMap<String, usize> {
    let mut layer_type_counts = HashMap::new();
    for name in graph.node_names() {
        if let Some(node) = graph.node(name) {
            *layer_type_counts
                .entry(node.layer().layer_type().to_string())
                .or_insert(0) += 1;
        }
    }
    layer_type_counts
}

fn log_layer_inventory(graph: &GraphNetwork, layer_type_counts: &HashMap<String, usize>) {
    let mut sorted: Vec<_> = layer_type_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("dot_graph layer inventory ({} nodes):", graph.num_nodes());
    for (layer_type, count) in &sorted {
        eprintln!("  {}: {}", layer_type, count);
    }
}

fn log_crown_backward_gaps(graph: &GraphNetwork, layer_type_counts: &HashMap<String, usize>) {
    let unsupported_in_dispatch = ["Div", "MinBinary", "MaxBinary", "Where", "ScatterNd"];
    let conditionally_unsupported = ["Sqrt", "Softmax", "InstanceNorm1d"];

    for layer_type in unsupported_in_dispatch {
        if let Some(count) = layer_type_counts.get(layer_type) {
            eprintln!(
                "  WARNING: {} appears {} times — returns Unsupported in backward dispatch",
                layer_type, count
            );
        }
    }
    for layer_type in conditionally_unsupported {
        if let Some(count) = layer_type_counts.get(layer_type) {
            eprintln!(
                "  NOTE: {} appears {} times — conditionally supported (may error on wide intervals)",
                layer_type, count
            );
        }
    }

    for name in graph.node_names() {
        if let Some(node) = graph.node(name) {
            let layer_type = node.layer().layer_type();
            if unsupported_in_dispatch.contains(&layer_type)
                || conditionally_unsupported.contains(&layer_type)
            {
                eprintln!(
                    "  DETAIL: {} (type={}) inputs={:?}",
                    name,
                    layer_type,
                    node.inputs(),
                );
            }
        }
    }
}

/// The real speaker dot-component graph must retain the explicit three-node
/// cosine head after ONNX lowering. The inventory and backward-gap log remain
/// useful diagnostics, while the topology assertions prevent an empty or
/// miswired graph from passing as coverage. #3499
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_speaker_dot_graph_inventory_covers_cosine_head_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let (dot_graph, _, _) = cosine_head::build_speaker_cosine_component_graphs();
    let layer_type_counts = collect_layer_type_counts(&dot_graph);
    assert_eq!(
        layer_type_counts.values().sum::<usize>(),
        dot_graph.num_nodes(),
        "every dot-graph node must contribute exactly once to the layer inventory"
    );
    assert_eq!(dot_graph.output_name(), "cosine_dot");
    for (name, expected_type, expected_input) in [
        ("cosine_embed_scaled", "MulConstant", None),
        (
            "cosine_dot_terms",
            "MulConstant",
            Some("cosine_embed_scaled"),
        ),
        ("cosine_dot", "ReduceSum", Some("cosine_dot_terms")),
    ] {
        let node = dot_graph
            .node(name)
            .unwrap_or_else(|| panic!("dot graph missing required cosine-head node {name}"));
        assert_eq!(
            node.layer().layer_type(),
            expected_type,
            "cosine-head node {name} changed layer type"
        );
        assert_eq!(
            node.inputs().len(),
            1,
            "cosine-head node {name} must remain unary"
        );
        if let Some(expected_input) = expected_input {
            assert_eq!(node.inputs()[0], expected_input);
        }
    }
    log_layer_inventory(&dot_graph, &layer_type_counts);
    log_crown_backward_gaps(&dot_graph, &layer_type_counts);
}

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_speaker_cosine_components_use_tighter_crown_ibp_intermediates_3596() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let (dot_graph, norm_sq_graph, _) = cosine_head::build_speaker_cosine_component_graphs();
    let model = shared::avoice_speaker_encoder();
    let input = shared::bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        shared::SPEAKER_ENCODER_EPSILON,
    );

    let (dot_ibp, dot_crown, dot_strict_improvement) =
        assert_scalar_component_crown_exercises_path_without_loosening(&dot_graph, &input, "dot");
    let (norm_sq_ibp, norm_sq_crown, norm_sq_strict_improvement) =
        assert_scalar_component_crown_exercises_path_without_loosening(
            &norm_sq_graph,
            &input,
            "norm squared",
        );

    let (ibp_distance_upper, ibp_nonvacuous) =
        cosine_head::speaker_cosine_distance_upper(dot_ibp.0, norm_sq_ibp.1);
    let (crown_distance_upper, crown_nonvacuous) =
        cosine_head::speaker_cosine_distance_upper(dot_crown.0, norm_sq_crown.1);
    eprintln!(
        "speaker cosine distance upper: pure_ibp={ibp_distance_upper} \
         (nonvacuous={ibp_nonvacuous}); \
         crown={crown_distance_upper} (nonvacuous={crown_nonvacuous}); \
         strict_improvement(dot={dot_strict_improvement}, norm_sq={norm_sq_strict_improvement})"
    );
}

/// Compute cosine distance bounds using pre-computed encoder node bounds.
///
/// Extends encoder bounds to component heads, runs spec-guided CROWN on each,
/// and computes the analytical distance upper bound.
pub(super) struct CosineDistanceResult {
    pub(super) dot_lower: f32,
    pub(super) dot_upper: f32,
    pub(super) nsq_lower: f32,
    pub(super) nsq_upper: f32,
    pub(super) distance_upper: f32,
    pub(super) nonvacuous: bool,
}

pub(super) fn cosine_distance_from_encoder_bounds(
    dot_graph: &GraphNetwork,
    norm_sq_graph: &GraphNetwork,
    input: &BoundedTensor,
    encoder_bounds: &HashMap<String, BoundedTensor>,
    encoder_output_name: &str,
    label: &str,
) -> CosineDistanceResult {
    let dot_node_bounds =
        cosine_head::build_component_node_bounds(dot_graph, encoder_bounds, encoder_output_name);
    let nsq_node_bounds = cosine_head::build_component_node_bounds(
        norm_sq_graph,
        encoder_bounds,
        encoder_output_name,
    );
    let (dot_lower, dot_upper) = scalar_spec_bounds_with_node_bounds(
        dot_graph,
        input,
        &dot_node_bounds,
        &format!("{label} dot"),
    );
    let (nsq_lower, nsq_upper) = scalar_spec_bounds_with_node_bounds(
        norm_sq_graph,
        input,
        &nsq_node_bounds,
        &format!("{label} norm_sq"),
    );
    let (distance_upper, nonvacuous) =
        cosine_head::speaker_cosine_distance_upper(dot_lower, nsq_upper);
    CosineDistanceResult {
        dot_lower,
        dot_upper,
        nsq_lower,
        nsq_upper,
        distance_upper,
        nonvacuous,
    }
}
