// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork IBP propagation tests.
use crate::domain_clip::DomainClipper;
use crate::*;
use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};
use ny_core::NyError;

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_soundness() {
    // Test soundness: sample points should be within computed bounds
    let mut graph = GraphNetwork::new();

    // Build: input -> proj -> relu -> output
    let weight = arr2(&[[1.0_f32, -1.0], [-1.0, 1.0]]);
    let bias = arr1(&[0.5_f32, -0.5]);
    let proj = LinearLayer::new(weight, Some(bias)).unwrap();
    graph.add_node(GraphNode::from_input("proj", Layer::Linear(proj)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["proj".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_ibp(&input).unwrap();

    // Sample test points and verify they're within bounds
    let test_points = vec![
        arr1(&[-1.0_f32, -1.0]),
        arr1(&[1.0_f32, 1.0]),
        arr1(&[0.0_f32, 0.0]),
        arr1(&[-0.5_f32, 0.5]),
        arr1(&[0.5_f32, -0.5]),
    ];

    for point in test_points {
        // Linear: W @ x + b = [x0 - x1 + 0.5, -x0 + x1 - 0.5]
        let linear_out = arr1(&[
            point[[0]] - point[[1]] + 0.5,
            -point[[0]] + point[[1]] - 0.5,
        ]);

        // ReLU
        let relu_out = linear_out.mapv(|v| v.max(0.0));

        // Check bounds
        for i in 0..2 {
            assert!(
                relu_out[[i]] >= bounds.lower()[[i]] - 1e-5,
                "Point {:?}: output[{}]={} < lower={}",
                point,
                i,
                relu_out[[i]],
                bounds.lower()[[i]]
            );
            assert!(
                relu_out[[i]] <= bounds.upper()[[i]] + 1e-5,
                "Point {:?}: output[{}]={} > upper={}",
                point,
                i,
                relu_out[[i]],
                bounds.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_matches_collect_node_bounds_on_branching_concat_3500() {
    let lin_a = LinearLayer::new(
        arr2(&[[1.0_f32, -0.5], [0.25, 1.5]]),
        Some(arr1(&[0.1_f32, -0.2])),
    )
    .unwrap();
    let lin_b = LinearLayer::new(
        arr2(&[[0.4_f32, 0.75], [-1.0, 0.5]]),
        Some(arr1(&[-0.3_f32, 0.05])),
    )
    .unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("lin_a", Layer::Linear(lin_a)));
    graph.add_node(GraphNode::from_input("lin_b", Layer::Linear(lin_b)));
    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "lin_a",
        "lin_b",
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["lin_a".to_string(), "relu".to_string()],
    ));
    graph.set_output("concat");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.25]).into_dyn(),
        arr1(&[2.0_f32, 1.5]).into_dyn(),
    )
    .unwrap();

    let collected = graph.collect_node_bounds(&input).unwrap();
    let output = graph.propagate_ibp(&input).unwrap();
    let expected = collected
        .get("concat")
        .expect("collect_node_bounds should include the output node");

    assert_eq!(output.shape(), expected.shape());
    assert_eq!(output.lower(), expected.lower());
    assert_eq!(output.upper(), expected.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_skip_merge_multi_input_errors() {
    let mut graph = GraphNetwork::new();

    let weight = arr2(&[[1.0_f32]]);
    let bias = arr1(&[0.0_f32]);
    let linear_a = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();
    let linear_b = LinearLayer::new(weight, Some(bias)).unwrap();

    graph.add_node(GraphNode::from_input("a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::new(
        "skip_merge",
        Layer::SkipMerge(SkipMergeLayer::new()),
        vec!["a".to_string(), "b".to_string()],
    ));
    graph.set_output("skip_merge");

    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let err = graph.propagate_ibp(&input).unwrap_err();
    assert!(
        err.to_string().contains("SkipMerge node"),
        "expected SkipMerge multi-input error, got: {}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_where_wrong_arity_returns_invalid_spec_2633() {
    let mut graph = GraphNetwork::new();

    let linear = LinearLayer::new(arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]), None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "where",
        Layer::Where(WhereLayer::new()),
        vec!["linear".to_string()],
    ));
    graph.set_output("where");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let err = graph
        .propagate_ibp(&input)
        .expect_err("malformed Where arity should return error");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("where") && msg.contains("3 inputs"),
        "expected node name and ternary arity diagnostic, got: {msg}"
    );
}

/// #2991: Concat with single input is now caught at construction time by
/// GraphNode::try_new() arity validation (#2481, #2686).
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_concat_requires_two_inputs() {
    let err = GraphNode::try_new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["source".to_string()],
    )
    .expect_err("single-input Concat should return InvalidSpec at construction");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("concat") && msg.contains("2 input"),
        "expected concat arity diagnostic, got: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_detailed_concat_requires_two_inputs() {
    let err = GraphNode::try_new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["source".to_string()],
    )
    .expect_err("single-input Concat should return InvalidSpec at construction");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_collect_node_bounds_concat_requires_two_inputs() {
    let err = GraphNode::try_new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["source".to_string()],
    )
    .expect_err("single-input Concat should return InvalidSpec at construction");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
}

/// #4112 follow-up: fully-constant Concat may have zero graph-edge inputs, but
/// an empty constant list is still malformed and must fail at construction.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_concat_empty_constant_list_requires_input_4112() {
    let err = GraphNode::try_new(
        "concat",
        Layer::Concat(ConcatLayer::with_constants(0, vec![], vec![])),
        vec![],
    )
    .expect_err("empty-arity Concat should return InvalidSpec at construction");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("concat") && msg.contains("1 input"),
        "expected concat minimum-arity diagnostic, got: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_detailed_supports_self_attention_2472() {
    let mut graph = GraphNetwork::new();
    let zero = ArrayD::zeros(IxDyn(&[1]));
    let passthrough = Layer::AddConstant(AddConstantLayer::new(zero));

    graph.add_node(GraphNode::from_input("q", passthrough.clone()));
    graph.add_node(GraphNode::from_input("k", passthrough.clone()));
    graph.add_node(GraphNode::from_input("v", passthrough));
    graph.add_node(GraphNode::new(
        "attn",
        Layer::SelfAttention(SelfAttentionLayer::standard()),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    ));
    graph.set_output("attn");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 1.0_f32),
    )
    .expect("valid input bounds");

    let detailed = graph
        .propagate_ibp_detailed(&input, 0.1)
        .expect("detailed IBP must support SelfAttention ternary dispatch");
    let last = detailed
        .nodes
        .last()
        .expect("detailed run should contain at least one node");
    assert_eq!(last.layer_type, "SelfAttention");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_with_clipper_supports_nary_concat_2472() {
    let mut graph = GraphNetwork::new();
    let zero = ArrayD::zeros(IxDyn(&[1]));
    let passthrough = Layer::AddConstant(AddConstantLayer::new(zero));

    graph.add_node(GraphNode::from_input("a", passthrough.clone()));
    graph.add_node(GraphNode::from_input("b", passthrough.clone()));
    graph.add_node(GraphNode::from_input("c", passthrough));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    ));
    graph.set_output("concat");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.5]).into_dyn(),
        arr1(&[1.0_f32, 0.5]).into_dyn(),
    )
    .expect("valid input bounds");
    let mut clipper = DomainClipper::default();

    graph
        .collect_activation_statistics(&input, &mut clipper)
        .expect("activation statistics should collect for n-ary concat graph");
    let output = graph
        .propagate_ibp_with_clipper(&input, &mut clipper)
        .expect("clipper IBP must support n-ary concat");
    assert_eq!(output.shape(), &[6]);
}

#[test]
fn test_block_index_parsing() {
    assert_eq!(GraphNetwork::parse_block_index("layer0_attn_norm"), Some(0));
    assert_eq!(GraphNetwork::parse_block_index("layer12"), Some(12));
    assert_eq!(GraphNetwork::parse_block_index("layer"), None);
    assert_eq!(GraphNetwork::parse_block_index("layernorm"), None);
    assert_eq!(GraphNetwork::parse_block_index("other_layer3"), None);
}

#[ntest::timeout(10000)]
#[test]
fn test_matmul_ibp_soundness() {
    // Test that MatMul IBP bounds are sound (contain actual outputs)
    let matmul = MatMulLayer::new(false, None);

    // A: 2x3, B: 3x2 -> C: 2x2
    let a_lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let a_upper = arr2(&[[1.0_f32, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    let input_a = BoundedTensor::new(a_lower.into_dyn(), a_upper.into_dyn()).unwrap();

    let b_lower = arr2(&[[0.0_f32, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    let b_upper = arr2(&[[1.0_f32, 1.0], [1.0, 1.0], [1.0, 1.0]]);
    let input_b = BoundedTensor::new(b_lower.into_dyn(), b_upper.into_dyn()).unwrap();

    // Compute IBP bounds
    let ibp_bounds = matmul.propagate_ibp_binary(&input_a, &input_b).unwrap();

    // Test fixed sample points
    let test_points: Vec<(Array2<f32>, Array2<f32>)> = vec![
        // All zeros
        (Array2::zeros((2, 3)), Array2::zeros((3, 2))),
        // All ones
        (Array2::ones((2, 3)), Array2::ones((3, 2))),
        // Center
        (
            Array2::from_elem((2, 3), 0.5),
            Array2::from_elem((3, 2), 0.5),
        ),
        // Lower bounds
        (
            arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            arr2(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        ),
        // Upper bounds
        (
            arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            arr2(&[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        ),
    ];

    for (a, b) in test_points {
        let c = a.dot(&b);

        // Check bounds contain the result
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    c[[i, j]] >= ibp_bounds.lower()[[i, j]] - 1e-5,
                    "IBP lower bound violation: c[{},{}]={} < lower={}",
                    i,
                    j,
                    c[[i, j]],
                    ibp_bounds.lower()[[i, j]]
                );
                assert!(
                    c[[i, j]] <= ibp_bounds.upper()[[i, j]] + 1e-5,
                    "IBP upper bound violation: c[{},{}]={} > upper={}",
                    i,
                    j,
                    c[[i, j]],
                    ibp_bounds.upper()[[i, j]]
                );
            }
        }
    }
}
