// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2};
use ny_propagate::layers::{ExpLayer, LinearLayer, ReciprocalLayer};
use ny_propagate::{AlphaCrownConfig, GradientMethod, GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

fn build_reciprocal_exp_alpha_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let hidden_w = arr2(&[[1.15_f32, 0.45], [0.35, 1.05]]);
    let hidden_b = arr1(&[0.25_f32, 0.20]);
    graph.add_node(GraphNode::from_input(
        "linear_hidden",
        Layer::Linear(
            LinearLayer::new(hidden_w, Some(hidden_b)).expect("linear layer should build"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "reciprocal_hidden",
        Layer::Reciprocal(ReciprocalLayer::new()),
        vec!["linear_hidden".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "exp_hidden",
        Layer::Exp(ExpLayer::new()),
        vec!["reciprocal_hidden".to_string()],
    ));

    let out_w = arr2(&[[1.8_f32, -2.25]]);
    let out_b = arr1(&[0.1_f32]);
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(LinearLayer::new(out_w, Some(out_b)).expect("output layer should build")),
        vec!["exp_hidden".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("bounded reciprocal input should be valid");

    (graph, input)
}

fn reciprocal_alpha_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        spsa_samples: 4,
        sparse_ratio: 1.0,
        ..AlphaCrownConfig::default()
    }
}

fn assert_scalar_bounds_sound_by_sampling(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    lower: f32,
    upper: f32,
) {
    let input_lower = input.lower().as_slice().expect("contiguous").to_vec();
    let input_upper = input.upper().as_slice().expect("contiguous").to_vec();
    for sample_idx in 0..20 {
        let t = sample_idx as f32 / 19.0;
        let concrete: Vec<f32> = input_lower
            .iter()
            .zip(input_upper.iter())
            .enumerate()
            .map(|(j, (&lo, &hi))| {
                let phase = ((t + j as f32 * 0.31) % 1.0).clamp(0.0, 1.0);
                lo + phase * (hi - lo)
            })
            .collect();
        let point = arr1(&concrete).into_dyn();
        let concrete_bt =
            BoundedTensor::new(point.clone(), point).expect("point bounds should build");
        let value = graph
            .propagate_ibp(&concrete_bt)
            .expect("concrete graph evaluation should succeed")
            .flatten()
            .lower()[0];
        assert!(
            value >= lower - 1e-4 && value <= upper + 1e-4,
            "#4399 reciprocal alpha soundness violation: sample={sample_idx}, concrete={value}, bounds=[{lower}, {upper}]"
        );
    }
}

#[ntest::timeout(60000)]
#[test]
fn test_reciprocal_alpha_tightens_public_dag_bounds_4399() {
    let (graph, input) = build_reciprocal_exp_alpha_dag();

    let baseline = graph
        .collect_crown_ibp_bounds_dag(&input)
        .expect("#4399 baseline CROWN-IBP should succeed");
    let baseline_output = baseline
        .get(graph.output_name())
        .expect("#4399 baseline output bounds should exist")
        .flatten();
    let baseline_lower = baseline_output.lower()[0];
    let baseline_upper = baseline_output.upper()[0];
    let baseline_width = baseline_upper - baseline_lower;

    let (alpha_bounds, _alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &reciprocal_alpha_config(8))
        .expect("#4399 reciprocal alpha-CROWN should succeed");
    let alpha_output = alpha_bounds
        .get(graph.output_name())
        .expect("#4399 alpha output bounds should exist")
        .flatten();
    let alpha_lower = alpha_output.lower()[0];
    let alpha_upper = alpha_output.upper()[0];
    let alpha_width = alpha_upper - alpha_lower;

    assert!(
        alpha_lower.is_finite() && alpha_upper.is_finite() && alpha_lower <= alpha_upper,
        "#4399 alpha output bounds must be finite and ordered, got [{alpha_lower}, {alpha_upper}]"
    );
    assert!(
        alpha_lower >= baseline_lower - 1e-5,
        "#4399 reciprocal alpha lower bound loosened: baseline={baseline_lower}, alpha={alpha_lower}"
    );
    assert!(
        alpha_upper <= baseline_upper + 1e-5,
        "#4399 reciprocal alpha upper bound loosened: baseline={baseline_upper}, alpha={alpha_upper}"
    );
    assert!(
        alpha_width + 1e-5 < baseline_width,
        "#4399 reciprocal alpha should beat fixed-slope CROWN. baseline=[{baseline_lower}, {baseline_upper}] width={baseline_width:.6}, alpha=[{alpha_lower}, {alpha_upper}] width={alpha_width:.6}"
    );

    assert_scalar_bounds_sound_by_sampling(&graph, &input, alpha_lower, alpha_upper);
}
