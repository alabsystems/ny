// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::AlphaCrownConfig;
use crate::layers::{
    AddConstantLayer, AddLayer, Conv1dLayer, Layer, MulBinaryLayer, ReciprocalLayer,
    ReduceSumLayer, SqrtLayer,
};
use crate::network::core::{GraphNetwork, GraphNode};
use crate::types::BoundsProvenance;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

fn kokoro_alpha_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method: crate::bounds::GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        spsa_samples: 4,
        sparse_ratio: 1.0,
        ..AlphaCrownConfig::default()
    }
}

fn total_width(bounds: &std::collections::HashMap<String, BoundedTensor>, node_name: &str) -> f32 {
    let output = bounds
        .get(node_name)
        .unwrap_or_else(|| panic!("missing bounds for node '{node_name}'"))
        .flatten();
    output
        .upper()
        .iter()
        .zip(output.lower().iter())
        .map(|(upper, lower)| upper - lower)
        .sum()
}

fn build_kokoro_alpha_regression_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let main_kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.45_f32, -0.35]).expect("valid kernel");
    let main_bias = arr1(&[0.15_f32, -0.05]);
    let main_conv =
        Conv1dLayer::with_input_length(main_kernel, Some(main_bias), 1, 0, 4).expect("conv1d");
    graph.add_node(GraphNode::from_input("main_conv", Layer::Conv1d(main_conv)));
    graph.add_node(GraphNode::new(
        "main_shift",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
            IxDyn(&[2, 4]),
            2.4_f32,
        ))),
        vec!["main_conv".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sqrt_hidden",
        Layer::Sqrt(SqrtLayer::new()),
        vec!["main_shift".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "reciprocal_hidden",
        Layer::Reciprocal(ReciprocalLayer::new()),
        vec!["sqrt_hidden".to_string()],
    ));

    let gate_kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.8_f32, 0.55]).expect("valid kernel");
    let gate_bias = arr1(&[0.35_f32, 0.2]);
    let gate_conv =
        Conv1dLayer::with_input_length(gate_kernel, Some(gate_bias), 1, 0, 4).expect("conv1d");
    graph.add_node(GraphNode::from_input("gate_conv", Layer::Conv1d(gate_conv)));
    graph.add_node(GraphNode::binary(
        "gated",
        Layer::MulBinary(MulBinaryLayer),
        "reciprocal_hidden",
        "gate_conv",
    ));

    let skip_kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.25_f32, -0.1]).expect("valid kernel");
    let skip_bias = arr1(&[0.05_f32, 0.08]);
    let skip_conv =
        Conv1dLayer::with_input_length(skip_kernel, Some(skip_bias), 1, 0, 4).expect("conv1d");
    graph.add_node(GraphNode::from_input("skip_conv", Layer::Conv1d(skip_conv)));
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "gated",
        "skip_conv",
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::ReduceSum(ReduceSumLayer::new(vec![0, 1], false)),
        vec!["residual".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-0.6_f32, -0.2, 0.1, -0.3])
            .expect("valid lower input shape"),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.8_f32, 0.45, 1.1, 0.6])
            .expect("valid upper input shape"),
    )
    .expect("kokoro alpha regression input should be valid");

    (graph, input)
}

#[ntest::timeout(30000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_tightens_kokoro_style_conv1d_residual_graph_4400() {
    let (graph, input) = build_kokoro_alpha_regression_graph();

    let baseline_status = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .expect("#4400 baseline CROWN-IBP should succeed");
    // The graph output must tighten through CROWN backward — this is the
    // verification-relevant bound and is always in the demand set.
    assert_eq!(
        baseline_status.provenance.get("out"),
        Some(&BoundsProvenance::Crown),
        "#4400 output node 'out' should tighten through CROWN without forward fallback"
    );
    // The MulBinary/Reciprocal/Sqrt *inputs* feed nonlinear relaxations, so the
    // demand-driven selector (#3775) must tighten them through CROWN. 'gated' and
    // 'residual' are intentionally NOT asserted here: their only consumers (Add,
    // ReduceSum) are exact linear ops that list no required input-bound indices,
    // so demand-driven correctly leaves them on the sound IBP path. Asserting CROWN
    // on them would contradict the #3775 design.
    for node_name in [
        "reciprocal_hidden",
        "gate_conv",
        "sqrt_hidden",
        "main_shift",
    ] {
        assert_eq!(
            baseline_status.provenance.get(node_name),
            Some(&BoundsProvenance::Crown),
            "#4400 nonlinear-relaxation input '{node_name}' should tighten through CROWN"
        );
    }
    let (baseline_alpha_bounds, baseline_alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &kokoro_alpha_config(0))
        .expect("#4400 zero-iteration alpha-CROWN should succeed");
    let (optimized_bounds, optimized_alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &kokoro_alpha_config(8))
        .expect("#4400 optimized alpha-CROWN should succeed");

    let baseline_width = total_width(&baseline_alpha_bounds, graph.output_name());
    let optimized_width = total_width(&optimized_bounds, graph.output_name());
    let baseline_gated_width = total_width(&baseline_alpha_bounds, "gated");
    let optimized_gated_width = total_width(&optimized_bounds, "gated");
    let baseline_residual_width = total_width(&baseline_alpha_bounds, "residual");
    let optimized_residual_width = total_width(&optimized_bounds, "residual");
    assert!(
        optimized_width <= baseline_width + 1e-5,
        "#4400 optimized alpha width must not loosen baseline: baseline={baseline_width}, optimized={optimized_width}"
    );
    assert!(
        optimized_width + 1e-4 < baseline_width
            || optimized_gated_width + 1e-4 < baseline_gated_width
            || optimized_residual_width + 1e-4 < baseline_residual_width,
        "#4400 optimized alpha should tighten at least one Kokoro-style target width: out {baseline_width}->{optimized_width}, gated {baseline_gated_width}->{optimized_gated_width}, residual {baseline_residual_width}->{optimized_residual_width}"
    );

    let baseline_sqrt: Vec<String> = baseline_alpha_state.sqrt_alpha_names().cloned().collect();
    let baseline_reciprocal: Vec<String> = baseline_alpha_state
        .reciprocal_alpha_names()
        .cloned()
        .collect();
    assert_eq!(
        baseline_sqrt,
        vec!["sqrt_hidden".to_string()],
        "#4400 zero-iteration run should initialize sqrt alpha state for the decomposed norm path"
    );
    assert_eq!(
        baseline_reciprocal,
        vec!["reciprocal_hidden".to_string()],
        "#4400 zero-iteration run should initialize reciprocal alpha state for the decomposed norm path"
    );

    let optimized_sqrt: Vec<String> = optimized_alpha_state.sqrt_alpha_names().cloned().collect();
    let optimized_reciprocal: Vec<String> = optimized_alpha_state
        .reciprocal_alpha_names()
        .cloned()
        .collect();
    assert_eq!(optimized_sqrt, vec!["sqrt_hidden".to_string()]);
    assert_eq!(optimized_reciprocal, vec!["reciprocal_hidden".to_string()]);
}
