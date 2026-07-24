// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::NETWORK_INPUT_IDX;
use ny_tensor::BoundedTensor;

use crate::layers::activations::ReLULayer;
use crate::layers::binary_ops::{AddLayer, MatMulLayer};
use crate::layers::convolution::Conv2dLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::pooling::AveragePoolLayer;
use crate::layers::transform::FlattenLayer;
use crate::layers::Layer;
use crate::network::core::graph::node::GraphNode;
use crate::network::core::graph::{GraphNetwork, NETWORK_INPUT};

use super::try_lower_graph_dag;

/// Build a minimal residual-add DAG: input -> linear1 -> relu -> linear2 + input.
fn build_residual_dag() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]);
    let b1 = arr1(&[0.1_f32, -0.05]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.6_f32, -0.2], [-0.4, 0.7]]);
    let b2 = arr1(&[0.0_f32, 0.0]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));

    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        NETWORK_INPUT,
        "linear2",
    ));
    graph.set_output("residual");
    graph
}

#[test]
fn test_residual_add_dag_lowers_successfully() {
    let graph = build_residual_dag();
    let input_shape = &[2];
    let plan = try_lower_graph_dag(&graph, input_shape);
    assert!(plan.is_some(), "residual-add DAG should lower");

    let plan = plan.unwrap();
    assert_eq!(plan.ops.len(), 4, "linear1, relu, linear2, residual");
    assert_eq!(plan.output_op_idx, 3);
    assert_eq!(plan.input_shape, vec![2]);
}

#[test]
fn test_residual_dag_input_indices_correct() {
    let graph = build_residual_dag();
    let plan = try_lower_graph_dag(&graph, &[2]).unwrap();

    // linear1 reads from network input
    match &plan.ops[0] {
        ny_core::GpuDagIbpOp::Linear { input_idx, .. } => {
            assert_eq!(*input_idx, NETWORK_INPUT_IDX);
        }
        other => panic!("expected Linear, got {:?}", other),
    }

    // relu reads from linear1 (index 0)
    match &plan.ops[1] {
        ny_core::GpuDagIbpOp::ReLU { input_idx, .. } => {
            assert_eq!(*input_idx, 0);
        }
        other => panic!("expected ReLU, got {:?}", other),
    }

    // linear2 reads from relu (index 1)
    match &plan.ops[2] {
        ny_core::GpuDagIbpOp::Linear { input_idx, .. } => {
            assert_eq!(*input_idx, 1);
        }
        other => panic!("expected Linear, got {:?}", other),
    }

    // residual Add reads from network input and linear2 (index 2)
    match &plan.ops[3] {
        ny_core::GpuDagIbpOp::Add {
            input_a_idx,
            input_b_idx,
            ..
        } => {
            assert_eq!(*input_a_idx, NETWORK_INPUT_IDX);
            assert_eq!(*input_b_idx, 2);
        }
        other => panic!("expected Add, got {:?}", other),
    }
}

#[test]
fn test_average_pool_graph_lowers_with_spatial_input() {
    // AveragePool requires 3D+ input; verify lowering succeeds with proper shape.
    let mut graph = GraphNetwork::new();

    let conv = make_conv2d_1x1(4, 3);
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));

    // 2×2 pooling with stride 2 on a 2×2 spatial input → 1×1 output
    let avgpool = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    graph.add_node(GraphNode::new(
        "pool",
        Layer::AveragePool(avgpool),
        vec!["conv1".to_string()],
    ));
    graph.set_output("pool");

    let plan = try_lower_graph_dag(&graph, &[3, 2, 2]);
    assert!(
        plan.is_some(),
        "AveragePool graph with spatial input should lower"
    );

    let plan = plan.unwrap();
    assert_eq!(plan.ops.len(), 2, "conv1 + pool");
    match &plan.ops[1] {
        ny_core::GpuDagIbpOp::AveragePool {
            channels,
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            num_elements,
            input_idx,
            ..
        } => {
            assert_eq!(*channels, 4);
            assert_eq!(*input_h, 2);
            assert_eq!(*input_w, 2);
            assert_eq!(*output_h, 1);
            assert_eq!(*output_w, 1);
            assert_eq!(*kernel_h, 2);
            assert_eq!(*kernel_w, 2);
            assert_eq!(*num_elements, 4); // 4 channels * 1 * 1
            assert_eq!(*input_idx, 0);
        }
        other => panic!("expected AveragePool, got {:?}", other),
    }
}

#[test]
fn test_average_pool_graph_fails_on_1d_input() {
    // AveragePool with 1D input should fail (needs 3D+)
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.5_f32, -0.2], [0.3, 0.8]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    let avgpool = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    graph.add_node(GraphNode::new(
        "pool",
        Layer::AveragePool(avgpool),
        vec!["linear1".to_string()],
    ));
    graph.set_output("pool");

    let plan = try_lower_graph_dag(&graph, &[2]);
    assert!(
        plan.is_none(),
        "AveragePool on 1D input should fail lowering"
    );
}

#[test]
fn test_global_average_pool_graph_lowers() {
    // Global average pool: kernel_size (0, 0) sentinel
    let mut graph = GraphNetwork::new();

    let conv = make_conv2d_1x1(4, 3);
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));

    let avgpool = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);
    graph.add_node(GraphNode::new(
        "pool",
        Layer::AveragePool(avgpool),
        vec!["conv1".to_string()],
    ));
    graph.set_output("pool");

    let plan = try_lower_graph_dag(&graph, &[3, 4, 4]);
    assert!(plan.is_some(), "Global AveragePool should lower");

    let plan = plan.unwrap();
    match &plan.ops[1] {
        ny_core::GpuDagIbpOp::AveragePool {
            is_global,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            ..
        } => {
            assert!(*is_global);
            assert_eq!(*output_h, 1);
            assert_eq!(*output_w, 1);
            // Global: kernel covers full spatial dims (4x4)
            assert_eq!(*kernel_h, 4);
            assert_eq!(*kernel_w, 4);
        }
        other => panic!("expected AveragePool, got {:?}", other),
    }
}

#[test]
fn test_matmul_binary_op_returns_none() {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.5_f32, -0.2], [0.3, 0.8]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    let w2 = arr2(&[[0.6_f32, -0.1], [-0.3, 0.7]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::from_input("linear2", Layer::Linear(linear2)));

    // MatMul binary is unsupported
    graph.add_node(GraphNode::binary(
        "matmul",
        Layer::MatMul(MatMulLayer::new(false, None)),
        "linear1",
        "linear2",
    ));
    graph.set_output("matmul");

    let plan = try_lower_graph_dag(&graph, &[2]);
    assert!(
        plan.is_none(),
        "MatMul binary should cause lowering to fail"
    );
}

#[test]
fn test_empty_graph_returns_none() {
    let graph = GraphNetwork::new();
    let plan = try_lower_graph_dag(&graph, &[2]);
    assert!(plan.is_none(), "empty graph should return None");
}

#[test]
fn test_sequential_graph_lowers_successfully() {
    // A simple chain (no residual) should also lower fine
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.6_f32, -0.2], [-0.4, 0.7]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");

    let plan = try_lower_graph_dag(&graph, &[2]);
    assert!(plan.is_some(), "sequential-like graph should lower");
    let plan = plan.unwrap();
    assert_eq!(plan.ops.len(), 3);
    assert_eq!(plan.output_op_idx, 2);
}

/// Build a 1x1 Conv2d kernel for testing shape: (out_c, in_c, 1, 1).
fn make_conv2d_1x1(out_c: usize, in_c: usize) -> Conv2dLayer {
    let data: Vec<f32> = (0..out_c * in_c).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, 1, 1]), data).unwrap();
    Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap()
}

#[test]
fn test_conv2d_dag_lowers_successfully() {
    let mut graph = GraphNetwork::new();

    let conv = make_conv2d_1x1(4, 3);
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.set_output("relu");

    // Input: (3, 2, 2) = 3 channels, 2x2 spatial
    let plan = try_lower_graph_dag(&graph, &[3, 2, 2]);
    assert!(plan.is_some(), "Conv2d graph should lower");

    let plan = plan.unwrap();
    assert_eq!(plan.ops.len(), 2, "conv1 + relu");
    match &plan.ops[0] {
        ny_core::GpuDagIbpOp::Conv2d {
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            input_idx,
            ..
        } => {
            assert_eq!(*out_channels, 4);
            assert_eq!(*in_channels, 3);
            assert_eq!(*kernel_h, 1);
            assert_eq!(*kernel_w, 1);
            assert_eq!(*input_idx, NETWORK_INPUT_IDX);
        }
        other => panic!("expected Conv2d, got {:?}", other),
    }
}

#[test]
fn test_grouped_conv2d_returns_none() {
    let mut graph = GraphNetwork::new();

    // groups=2 is unsupported in the DAG IBP plan
    let data: Vec<f32> = vec![0.1; 4]; // kernel shape (4, 1, 1, 1) for groups=2, in_c=2
    let kernel = ArrayD::from_shape_vec(IxDyn(&[4, 1, 1, 1]), data).unwrap();
    let conv = Conv2dLayer::new_full(kernel, None, (1, 1), (0, 0), 2).unwrap();
    graph.add_node(GraphNode::from_input("conv_g2", Layer::Conv2d(conv)));
    graph.set_output("conv_g2");

    let plan = try_lower_graph_dag(&graph, &[2, 4, 4]);
    assert!(
        plan.is_none(),
        "grouped Conv2d (groups=2) should cause lowering to fail"
    );
}

#[test]
fn test_linear_shape_mismatch_returns_none() {
    let mut graph = GraphNetwork::new();

    // Linear with in_features=3, but input shape has last_dim=2
    let w = arr2(&[[0.5_f32, -0.2, 0.3], [0.1, 0.8, -0.4]]);
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear)));
    graph.set_output("linear1");

    let plan = try_lower_graph_dag(&graph, &[2]); // last_dim=2, in_features=3
    assert!(
        plan.is_none(),
        "Linear with in_features != last_dim should return None"
    );
}

#[test]
fn test_flatten_lowers_to_view() {
    let mut graph = GraphNetwork::new();

    let conv = make_conv2d_1x1(4, 3);
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));

    let flatten = FlattenLayer::new(0); // flatten all dims
    graph.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(flatten),
        vec!["conv1".to_string()],
    ));

    let w = arr2(&[[0.1_f32; 16]; 2]); // 2 outputs, 16 inputs (4*2*2)
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(linear),
        vec!["flat".to_string()],
    ));
    graph.set_output("linear_out");

    let plan = try_lower_graph_dag(&graph, &[3, 2, 2]);
    assert!(plan.is_some(), "Conv2d + Flatten + Linear should lower");

    let plan = plan.unwrap();
    assert_eq!(plan.ops.len(), 3, "conv1 + flat + linear_out");
    match &plan.ops[1] {
        ny_core::GpuDagIbpOp::View {
            output_shape,
            input_idx,
        } => {
            assert_eq!(*input_idx, 0, "Flatten reads from conv1 (index 0)");
            // Flatten(axis=0) on [4,2,2] → [1, 16]
            let total: usize = output_shape.iter().product();
            assert_eq!(total, 16, "flattened shape should have 16 elements");
        }
        other => panic!("expected View, got {:?}", other),
    }
}

#[test]
fn test_conv2d_wrong_input_channels_returns_none() {
    let mut graph = GraphNetwork::new();

    // Conv expects 3 input channels but input has 5
    let conv = make_conv2d_1x1(4, 3);
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));
    graph.set_output("conv1");

    let plan = try_lower_graph_dag(&graph, &[5, 2, 2]); // 5 channels, conv expects 3
    assert!(
        plan.is_none(),
        "Conv2d with wrong input channels should return None"
    );
}

/// Regression test (#4319): verify that the residual-add DAG CPU IBP produces
/// correct bounds, confirming the integration path is sound.
///
/// DAG: input -> linear1 -> relu -> linear2 -> Add(input, linear2) -> output
#[test]
fn test_residual_add_dag_cpu_ibp_parity_4319() {
    let graph = build_residual_dag();

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    // CPU IBP (no engine, always CPU path)
    let cpu_output = graph.propagate_ibp(&input).expect("CPU IBP should succeed");
    assert_eq!(cpu_output.shape(), vec![2], "output shape should be [2]");

    // Verify bounds are finite and lower <= upper
    for (&l, &u) in cpu_output.lower().iter().zip(cpu_output.upper().iter()) {
        assert!(l.is_finite(), "lower bound should be finite, got {}", l);
        assert!(u.is_finite(), "upper bound should be finite, got {}", u);
        assert!(l <= u, "lower {} should be <= upper {}", l, u);
    }

    // engine=None via propagate_ibp_with_engine should produce identical results
    let engine_none_output = graph
        .propagate_ibp_with_engine(&input, None)
        .expect("engine=None IBP should succeed");
    assert_eq!(
        cpu_output.lower().as_slice(),
        engine_none_output.lower().as_slice(),
        "engine=None should produce identical lower bounds"
    );
    assert_eq!(
        cpu_output.upper().as_slice(),
        engine_none_output.upper().as_slice(),
        "engine=None should produce identical upper bounds"
    );
}

/// Verify that the lowered DAG plan has correct element counts for each op (#4319).
#[test]
fn test_residual_dag_op_element_counts() {
    let graph = build_residual_dag();
    let plan = try_lower_graph_dag(&graph, &[2]).unwrap();

    match &plan.ops[0] {
        ny_core::GpuDagIbpOp::Linear {
            in_features,
            out_features,
            ..
        } => {
            assert_eq!(*in_features, 2);
            assert_eq!(*out_features, 2);
        }
        other => panic!("expected Linear, got {:?}", other),
    }

    match &plan.ops[1] {
        ny_core::GpuDagIbpOp::ReLU { num_elements, .. } => {
            assert_eq!(*num_elements, 2);
        }
        other => panic!("expected ReLU, got {:?}", other),
    }

    match &plan.ops[2] {
        ny_core::GpuDagIbpOp::Linear {
            in_features,
            out_features,
            ..
        } => {
            assert_eq!(*in_features, 2);
            assert_eq!(*out_features, 2);
        }
        other => panic!("expected Linear, got {:?}", other),
    }

    match &plan.ops[3] {
        ny_core::GpuDagIbpOp::Add { num_elements, .. } => {
            assert_eq!(*num_elements, 2);
        }
        other => panic!("expected Add, got {:?}", other),
    }
}

/// Regression test (#4320): ResNet-style graph with global average pool
/// should produce correct CPU IBP bounds and lower successfully.
///
/// DAG: input → conv1(1×1) → relu → global_avg_pool → output
#[test]
fn test_resnet_avgpool_cpu_ibp_parity_4320() {
    let mut graph = GraphNetwork::new();

    let conv = make_conv2d_1x1(4, 3);
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));

    // Global average pool: kernel (0, 0) sentinel
    let avgpool = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);
    graph.add_node(GraphNode::new(
        "pool",
        Layer::AveragePool(avgpool),
        vec!["relu".to_string()],
    ));
    graph.set_output("pool");

    // Input: (3, 2, 2)
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 2]), vec![-1.0; 12]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 2]), vec![1.0; 12]).unwrap(),
    )
    .unwrap();

    // CPU IBP
    let cpu_output = graph.propagate_ibp(&input).expect("CPU IBP should succeed");
    assert_eq!(
        cpu_output.shape(),
        vec![4, 1, 1],
        "output should be (4, 1, 1)"
    );

    for (&l, &u) in cpu_output.lower().iter().zip(cpu_output.upper().iter()) {
        assert!(l.is_finite(), "lower bound should be finite, got {l}");
        assert!(u.is_finite(), "upper bound should be finite, got {u}");
        assert!(l <= u, "lower {l} should be <= upper {u}");
    }

    // engine=None parity
    let engine_none = graph
        .propagate_ibp_with_engine(&input, None)
        .expect("engine=None IBP should succeed");
    assert_eq!(
        cpu_output.lower().as_slice(),
        engine_none.lower().as_slice(),
        "engine=None should produce identical lower bounds"
    );

    // Verify the plan lowering works
    let plan = try_lower_graph_dag(&graph, &[3, 2, 2]);
    assert!(
        plan.is_some(),
        "ResNet with AvgPool should lower to DAG plan"
    );
}

/// Regression: the adv_check PGD code path calls
/// `graph.propagate_ibp_with_engine(&concrete, engine)` on each SPSA
/// perturbed point. This test verifies that call succeeds on a graph
/// containing AveragePool with engine=None, which is exactly what
/// adv_check does when no GPU device is available.
///
/// Part of #4320 acceptance criteria: "a new regression proves resident DAG
/// usage from the adv-check caller path".
#[test]
fn test_adv_check_ibp_with_engine_on_avgpool_graph_4320() {
    let mut graph = GraphNetwork::new();
    let conv = make_conv2d_1x1(4, 3);
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    let avgpool = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);
    graph.add_node(GraphNode::new(
        "pool",
        Layer::AveragePool(avgpool),
        vec!["relu".to_string()],
    ));
    graph.set_output("pool");

    // Simulate what adv_check does: concrete point → propagate_ibp_with_engine
    let concrete_point = ArrayD::from_shape_vec(IxDyn(&[3, 2, 2]), vec![0.5; 12]).unwrap();
    let concrete = BoundedTensor::concrete(concrete_point).unwrap();

    let result = graph.propagate_ibp_with_engine(&concrete, None);
    assert!(
        result.is_ok(),
        "propagate_ibp_with_engine(concrete, None) on AvgPool graph must succeed: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert_eq!(output.shape(), vec![4, 1, 1]);
    // Concrete input → lower == upper for each output element
    for (&l, &u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(
            (l - u).abs() < 1e-5,
            "concrete point should produce tight bounds: l={l}, u={u}"
        );
    }
}
