// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::bounds::GradientMethod;
use crate::layers::{AddLayer, Conv2dLayer, ReLULayer, ReduceSumLayer};
use crate::network::core::{GraphNode, NETWORK_INPUT};
use ndarray::{arr1, ArrayD, IxDyn};

fn build_channel_only_conv_residual_dag_4404() -> (GraphNetwork, BoundedTensor) {
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
        NETWORK_INPUT,
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
    .expect("valid channel-only DAG input");

    (graph, input)
}

fn narrowed_channel_only_input_4404() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![-0.6_f32, -0.3, 0.15, -0.15, -0.2, -0.05, 0.1, -0.2],
        )
        .expect("valid warm-start lower input shape"),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![0.8_f32, 0.35, 0.75, 0.4, 0.55, 0.3, 0.7, 0.2],
        )
        .expect("valid warm-start upper input shape"),
    )
    .expect("valid warm-start input")
}

fn channel_only_config_4404(
    iterations: usize,
    gradient_method: GradientMethod,
) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method,
        full_conv_alpha: false,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    }
}

fn assert_channel_only_optimizer_state_updated_4404(
    alpha_state: &GraphAlphaState,
    relu_name: &str,
    label: &str,
) {
    let alpha = alpha_state
        .alpha(relu_name)
        .expect("channel-only ReLU alpha should exist");
    assert_eq!(
        alpha.len(),
        2,
        "{label}: full_conv_alpha=false must reduce Conv2d ReLU alpha to per-channel length"
    );

    let spatial_shape = alpha_state
        .spatial_shape(relu_name)
        .expect("channel-only ReLU should record its original spatial shape");
    assert_eq!(
        spatial_shape,
        &[2, 2, 2],
        "{label}: expected stored [C,H,W] spatial shape for channel-only alpha"
    );

    let adam_m = alpha_state
        .adam_m
        .get(relu_name)
        .expect("channel-only ReLU Adam first moment should exist");
    let adam_v = alpha_state
        .adam_v
        .get(relu_name)
        .expect("channel-only ReLU Adam second moment should exist");
    assert!(
        adam_m.iter().all(|value| value.is_finite())
            && adam_v.iter().all(|value| value.is_finite()),
        "{label}: optimizer state must stay finite for channel-only alpha"
    );
    assert!(
        adam_m.iter().any(|value| value.abs() > 1e-8),
        "{label}: lower-path Adam state stayed zero, so reduce_gradient/update_adam did not run"
    );
    assert!(
        adam_v.iter().any(|value| value.abs() > 1e-10),
        "{label}: second-moment state stayed zero, so optimizer update was skipped"
    );

    let adam_m_upper = alpha_state
        .adam_m_upper
        .get(relu_name)
        .expect("channel-only ReLU upper-path Adam first moment should exist");
    assert!(
        adam_m_upper.iter().any(|value| value.abs() > 1e-8),
        "{label}: upper-path Adam state stayed zero, so upper alpha updates were skipped"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_channel_only_updates_dag_optimizer_state_4404() {
    let (graph, input) = build_channel_only_conv_residual_dag_4404();
    let (_bounds, alpha_state) = graph
        .collect_alpha_crown_bounds_dag(
            &input,
            &channel_only_config_4404(2, GradientMethod::AnalyticChain),
        )
        .expect("DAG alpha-CROWN collection should succeed");

    assert_channel_only_optimizer_state_updated_4404(&alpha_state, "relu", "#4404 DAG optimizer");
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_channel_only_updates_spsa_state_4404() {
    let (graph, input) = build_channel_only_conv_residual_dag_4404();
    let (_bounds, alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &channel_only_config_4404(2, GradientMethod::Spsa))
        .expect("SPSA alpha-CROWN collection should succeed");

    assert_channel_only_optimizer_state_updated_4404(&alpha_state, "relu", "#4404 SPSA root");
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_warm_channel_only_updates_state_4404() {
    let (graph, root_input) = build_channel_only_conv_residual_dag_4404();
    let (_root_bounds, root_alpha_state) = graph
        .collect_alpha_crown_bounds_dag(
            &root_input,
            &channel_only_config_4404(2, GradientMethod::Spsa),
        )
        .expect("root alpha-CROWN collection should succeed");

    let child_input = narrowed_channel_only_input_4404();
    let (_warm_bounds, warm_alpha_state) = graph
        .collect_alpha_crown_bounds_dag_warm(
            &child_input,
            &channel_only_config_4404(2, GradientMethod::Spsa),
            &root_alpha_state,
        )
        .expect("warm-start alpha-CROWN collection should succeed");

    assert_channel_only_optimizer_state_updated_4404(&warm_alpha_state, "relu", "#4404 warm-start");
}

#[ntest::timeout(10000)]
#[test]
fn test_optimize_alpha_for_spec_objective_channel_only_updates_state_4404() {
    let (graph, input) = build_channel_only_conv_residual_dag_4404();
    let config = channel_only_config_4404(2, GradientMethod::Spsa);
    let ibp_bounds = graph
        .collect_crown_ibp_bounds_dag(&input)
        .expect("CROWN-IBP bounds should succeed");
    let (_, initial_alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &channel_only_config_4404(0, GradientMethod::Spsa))
        .expect("zero-iteration alpha collection should initialize channel-only state");

    let optimized_alpha_state = graph
        .optimize_alpha_for_spec_objective(
            &input,
            &ibp_bounds,
            &initial_alpha_state,
            &config,
            &[1.0_f32],
            None,
        )
        .expect("spec-objective alpha optimization should succeed");

    assert_channel_only_optimizer_state_updated_4404(
        &optimized_alpha_state,
        "relu",
        "#4404 spec-objective",
    );
}
