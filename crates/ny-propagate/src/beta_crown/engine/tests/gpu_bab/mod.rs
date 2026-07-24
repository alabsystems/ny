// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for GPU BaB verify_graph_gpu_domain_list.
//! Part of #1518: Add test coverage for GPU BaB execution path.
//!
//! Split into submodules by test category:
//! - smoke: Immediate verification/violation, result structure tests
//! - modes: Configuration mode tests (alpha-CROWN, IBP, warm-start, beta)
//! - branching: Branch exploration and domain processing
//! - convergence: Cross-path convergence diagnostics
//! - input_split: Input-split parity and regression coverage
//! - regressions: Architecture-specific regressions

mod branching;
mod convergence;
mod input_split;
mod kfsb_parity;
mod modes;
mod regressions;
mod smoke;

use super::prelude::*;

/// Create a simple GraphNetwork for testing: Linear -> ReLU -> Linear.
pub(super) fn simple_graph_network() -> GraphNetwork {
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[[1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// Create a GeLU network for GenBaB testing: Linear -> GeLU -> Linear.
pub(super) fn gelu_graph_network() -> GraphNetwork {
    use crate::layers::GELULayer;

    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // Identity
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[[1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "gelu1",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// Create a CNN-like GraphNetwork: Conv2d -> ReLU -> Flatten -> Linear.
///
/// This exercises the Conv2d and Flatten backward propagation paths in the
/// lA-capture variant, which were identity pass-through before #1811.
pub(super) fn conv_graph_network() -> GraphNetwork {
    use crate::layers::FlattenLayer;

    // Conv2d: 1 input channel, 2 output channels, 2x2 kernel
    // Input is [1, 3, 3], output after conv is [2, 2, 2]
    let mut kernel = ArrayD::zeros(IxDyn(&[2, 1, 2, 2]));
    // Channel 0: averaging filter
    kernel[[0, 0, 0, 0]] = 0.5;
    kernel[[0, 0, 0, 1]] = 0.5;
    kernel[[0, 0, 1, 0]] = 0.5;
    kernel[[0, 0, 1, 1]] = 0.5;
    // Channel 1: edge detection
    kernel[[1, 0, 0, 0]] = 1.0;
    kernel[[1, 0, 0, 1]] = -1.0;
    kernel[[1, 0, 1, 0]] = -1.0;
    kernel[[1, 0, 1, 1]] = 1.0;

    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 3, 3).unwrap();
    let flatten = FlattenLayer::new(0);
    // After conv: [2, 2, 2] = 8 elements, after flatten: [8]
    let w_out = Array2::from_shape_vec((1, 8), vec![1.0; 8]).unwrap();
    let linear_out = LinearLayer::new(w_out, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "flatten1",
        Layer::Flatten(flatten),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(linear_out),
        vec!["flatten1".to_string()],
    ));
    graph.set_output("linear_out");
    graph
}

/// Create a deeper network (ACAS-Xu-like): 3 hidden layers with 10 neurons each.
///
/// Structure: Linear(2->10) -> ReLU -> Linear(10->10) -> ReLU -> Linear(10->10) -> ReLU -> Linear(10->1)
///
/// Uses random-ish weights that create a non-trivial verification problem.
pub(super) fn deep_graph_network() -> GraphNetwork {
    // Layer 1: 2 -> 10
    let mut w1_data = vec![0.0f32; 10 * 2];
    for i in 0..10 {
        for j in 0..2 {
            // Alternating positive/negative pattern
            w1_data[i * 2 + j] = if (i + j) % 2 == 0 { 0.5 } else { -0.3 };
        }
    }
    let w1 = Array2::from_shape_vec((10, 2), w1_data).unwrap();
    let b1 = Array1::from_vec(vec![
        0.1, -0.1, 0.2, -0.2, 0.0, 0.0, 0.15, -0.15, 0.05, -0.05,
    ]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();

    // Layer 2: 10 -> 10
    let mut w2_data = vec![0.0f32; 10 * 10];
    for i in 0..10 {
        for j in 0..10 {
            if i == j {
                w2_data[i * 10 + j] = 0.6;
            } else if (i as i32 - j as i32).abs() == 1 {
                w2_data[i * 10 + j] = -0.2;
            }
        }
    }
    let w2 = Array2::from_shape_vec((10, 10), w2_data).unwrap();
    let linear2 = LinearLayer::new(w2, None).unwrap();

    // Layer 3: 10 -> 10
    let mut w3_data = vec![0.0f32; 10 * 10];
    for i in 0..10 {
        for j in 0..10 {
            if i == j {
                w3_data[i * 10 + j] = 0.5;
            } else if (i + j) % 3 == 0 {
                w3_data[i * 10 + j] = 0.1;
            }
        }
    }
    let w3 = Array2::from_shape_vec((10, 10), w3_data).unwrap();
    let linear3 = LinearLayer::new(w3, None).unwrap();

    // Output layer: 10 -> 1
    let w4 = Array2::from_shape_vec(
        (1, 10),
        vec![0.3, -0.2, 0.4, -0.1, 0.2, -0.3, 0.1, 0.2, -0.1, 0.3],
    )
    .unwrap();
    let linear4 = LinearLayer::new(w4, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear4",
        Layer::Linear(linear4),
        vec!["relu3".to_string()],
    ));
    graph.set_output("linear4");
    graph
}

/// Build equivalent sequential Network from deep_graph_network.
pub(super) fn deep_sequential_network() -> Network {
    // Must match deep_graph_network() exactly
    let mut w1_data = vec![0.0f32; 10 * 2];
    for i in 0..10 {
        for j in 0..2 {
            w1_data[i * 2 + j] = if (i + j) % 2 == 0 { 0.5 } else { -0.3 };
        }
    }
    let w1 = Array2::from_shape_vec((10, 2), w1_data).unwrap();
    let b1 = Array1::from_vec(vec![
        0.1, -0.1, 0.2, -0.2, 0.0, 0.0, 0.15, -0.15, 0.05, -0.05,
    ]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let mut w2_data = vec![0.0f32; 10 * 10];
    for i in 0..10 {
        for j in 0..10 {
            if i == j {
                w2_data[i * 10 + j] = 0.6;
            } else if (i as i32 - j as i32).abs() == 1 {
                w2_data[i * 10 + j] = -0.2;
            }
        }
    }
    let w2 = Array2::from_shape_vec((10, 10), w2_data).unwrap();
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut w3_data = vec![0.0f32; 10 * 10];
    for i in 0..10 {
        for j in 0..10 {
            if i == j {
                w3_data[i * 10 + j] = 0.5;
            } else if (i + j) % 3 == 0 {
                w3_data[i * 10 + j] = 0.1;
            }
        }
    }
    let w3 = Array2::from_shape_vec((10, 10), w3_data).unwrap();
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let w4 = Array2::from_shape_vec(
        (1, 10),
        vec![0.3, -0.2, 0.4, -0.1, 0.2, -0.3, 0.1, 0.2, -0.1, 0.3],
    )
    .unwrap();
    let linear4 = LinearLayer::new(w4, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear4));
    network
}
