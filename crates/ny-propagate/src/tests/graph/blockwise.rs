// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork block-wise verification tests.
use std::{cell::RefCell, path::PathBuf};

use crate::types::{BlockBoundsInfo, VerificationCheckpoint};
use crate::*;
use ndarray::{Array1, Array2, ArrayD, IxDyn};

fn add_transformer_block(
    graph: &mut GraphNetwork,
    block_index: usize,
    hidden: usize,
    q_weight: &Array2<f32>,
    k_weight: &Array2<f32>,
) {
    let attn_norm = format!("layer{}_attn_norm", block_index);
    if block_index == 0 {
        graph.add_node(GraphNode::from_input(
            attn_norm.as_str(),
            Layer::LayerNorm(
                LayerNormLayer::new(Array1::ones(hidden), Array1::zeros(hidden), 1e-5).unwrap(),
            ),
        ));
    } else {
        let prev_attn_norm = format!("layer{}_attn_norm", block_index - 1);
        graph.add_node(GraphNode::new(
            attn_norm.as_str(),
            Layer::LayerNorm(
                LayerNormLayer::new(Array1::ones(hidden), Array1::zeros(hidden), 1e-5).unwrap(),
            ),
            vec![prev_attn_norm],
        ));
    }

    let q_proj = format!("layer{}_q_proj", block_index);
    graph.add_node(GraphNode::new(
        q_proj.as_str(),
        Layer::Linear(LinearLayer::new(q_weight.clone(), None).unwrap()),
        vec![attn_norm.clone()],
    ));

    let k_proj = format!("layer{}_k_proj", block_index);
    graph.add_node(GraphNode::new(
        k_proj.as_str(),
        Layer::Linear(LinearLayer::new(k_weight.clone(), None).unwrap()),
        vec![attn_norm],
    ));

    let qk_matmul = format!("layer{}_qk_matmul", block_index);
    graph.add_node(GraphNode::new(
        qk_matmul.as_str(),
        Layer::MatMul(MatMulLayer::new(true, Some(1.0))),
        vec![q_proj, k_proj],
    ));
}

fn make_relu_checkpoint_graph(num_blocks: usize) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    for block_idx in 0..num_blocks {
        graph.add_node(GraphNode::from_input(
            format!("layer{block_idx}_relu"),
            Layer::ReLU(ReLULayer),
        ));
    }
    graph
}

fn scalar_input_bounds() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .expect("valid scalar input bounds")
}

fn scalar_block_reset_bounds(epsilon: f32) -> BoundedTensor {
    BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[1])), epsilon)
        .expect("valid scalar block-reset bounds")
}

fn assert_checkpoint_callback_matches_blocks(
    callbacks: &[(usize, String, usize, f32, f32, u64, usize)],
    result: &BlockWiseResult,
    expected_total_blocks: usize,
) {
    assert_eq!(
        callbacks.len(),
        result.blocks.len(),
        "expected one checkpoint callback per completed block"
    );
    for (callback, block) in callbacks.iter().zip(&result.blocks) {
        assert_eq!(
            callback.0, block.block_index,
            "callback block index mismatch"
        );
        assert_eq!(callback.1, block.block_name, "callback block name mismatch");
        assert_eq!(
            callback.2,
            block.nodes.len(),
            "callback node-count mismatch"
        );
        assert_eq!(
            callback.6, expected_total_blocks,
            "callback total_blocks should be graph-wide"
        );
        assert!(
            (callback.3 - block.input_width).abs() < f32::EPSILON,
            "callback input width mismatch: {} vs {}",
            callback.3,
            block.input_width
        );
        assert!(
            (callback.4 - block.output_width).abs() < f32::EPSILON,
            "callback output width mismatch: {} vs {}",
            callback.4,
            block.output_width
        );
    }
}

fn checkpoint_after_first_block(
    result: &BlockWiseResult,
    elapsed_ms: u64,
) -> VerificationCheckpoint {
    let mut checkpoint = VerificationCheckpoint::new(
        PathBuf::from("tests/models/dummy.onnx"),
        "0".repeat(64),
        0.1,
        "ibp",
        "cpu",
        result.total_blocks,
    );
    checkpoint.update(result.blocks[0].clone(), elapsed_ms);
    checkpoint
}

fn block_node_info<'a>(result: &'a BlockWiseResult, node_name: &str) -> &'a NodeBoundsInfo {
    result
        .blocks
        .iter()
        .flat_map(|block| block.nodes.iter())
        .find(|node| node.name == node_name)
        .unwrap_or_else(|| panic!("missing block-wise node info for {node_name}"))
}

fn assert_scalar_node_matches_bounds(
    result: &BlockWiseResult,
    node_name: &str,
    expected: &BoundedTensor,
) {
    let node = block_node_info(result, node_name);
    let expected_lower = expected
        .lower()
        .iter()
        .next()
        .copied()
        .expect("expected scalar lower bound");
    let expected_upper = expected
        .upper()
        .iter()
        .next()
        .copied()
        .expect("expected scalar upper bound");

    assert_eq!(
        node.output_shape,
        vec![1],
        "{node_name} should remain scalar in block-wise IBP"
    );

    if expected_lower.is_finite() {
        assert!(
            (node.min_bound - expected_lower).abs() < 1e-6,
            "{node_name} lower bound mismatch: {} vs {}",
            node.min_bound,
            expected_lower
        );
    } else {
        assert!(
            node.min_bound.is_infinite() && node.min_bound.is_sign_negative(),
            "{node_name} lower bound should be -inf, got {}",
            node.min_bound
        );
    }

    if expected_upper.is_finite() {
        assert!(
            (node.max_bound - expected_upper).abs() < 1e-6,
            "{node_name} upper bound mismatch: {} vs {}",
            node.max_bound,
            expected_upper
        );
    } else {
        assert!(
            node.max_bound.is_infinite() && node.max_bound.is_sign_positive(),
            "{node_name} upper bound should be +inf, got {}",
            node.max_bound
        );
    }

    let expected_width = expected.max_width();
    if expected_width.is_finite() {
        assert!(
            (node.output_width - expected_width).abs() < 1e-6,
            "{node_name} output width mismatch: {} vs {}",
            node.output_width,
            expected_width
        );
    } else {
        assert!(
            node.output_width.is_infinite(),
            "{node_name} output width should be infinite, got {}",
            node.output_width
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_verification() {
    // Test block-wise verification with a simple 2-block transformer-like graph.
    // Each block has: attn_norm -> q_proj -> k_proj -> qk_matmul
    // Output ends at qk_matmul to avoid shape mismatch issues
    let mut graph = GraphNetwork::new();
    let seq = 4;
    let hidden = 8;
    let epsilon = 0.1;

    // Block 0
    graph.add_node(GraphNode::from_input(
        "layer0_attn_norm",
        Layer::LayerNorm(
            LayerNormLayer::new(Array1::ones(hidden), Array1::zeros(hidden), 1e-5).unwrap(),
        ),
    ));

    let q_weight = Array2::from_shape_fn((hidden, hidden), |(i, j)| if i == j { 0.1 } else { 0.0 });
    graph.add_node(GraphNode::new(
        "layer0_q_proj",
        Layer::Linear(LinearLayer::new(q_weight.clone(), None).unwrap()),
        vec!["layer0_attn_norm".to_string()],
    ));

    let k_weight = Array2::from_shape_fn((hidden, hidden), |(i, j)| if i == j { 0.1 } else { 0.0 });
    graph.add_node(GraphNode::new(
        "layer0_k_proj",
        Layer::Linear(LinearLayer::new(k_weight.clone(), None).unwrap()),
        vec!["layer0_attn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer0_qk_matmul",
        Layer::MatMul(MatMulLayer::new(true, Some(1.0))),
        vec!["layer0_q_proj".to_string(), "layer0_k_proj".to_string()],
    ));

    // Block 1 - depends on block 0's LayerNorm output (via _input)
    graph.add_node(GraphNode::new(
        "layer1_attn_norm",
        Layer::LayerNorm(
            LayerNormLayer::new(Array1::ones(hidden), Array1::zeros(hidden), 1e-5).unwrap(),
        ),
        vec!["layer0_attn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer1_q_proj",
        Layer::Linear(LinearLayer::new(q_weight, None).unwrap()),
        vec!["layer1_attn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer1_k_proj",
        Layer::Linear(LinearLayer::new(k_weight, None).unwrap()),
        vec!["layer1_attn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer1_qk_matmul",
        Layer::MatMul(MatMulLayer::new(true, Some(1.0))),
        vec!["layer1_q_proj".to_string(), "layer1_k_proj".to_string()],
    ));

    graph.set_output("layer1_qk_matmul");

    // Create input
    let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[seq, hidden])), epsilon).unwrap();

    // Run block-wise verification
    let result = graph.propagate_ibp_block_wise(&input, epsilon).unwrap();

    // Should detect 2 blocks
    assert_eq!(
        result.total_blocks, 2,
        "Expected 2 blocks, got {}",
        result.total_blocks
    );
    assert_eq!(result.blocks.len(), 2);

    // Block 0 should have qk_matmul with zonotope tightening
    assert_eq!(result.blocks[0].block_name, "layer0");
    assert!(
        result.blocks[0].qk_matmul_width.is_some(),
        "Block 0 should have Q@K^T width"
    );

    // Block 1 should also have qk_matmul
    assert_eq!(result.blocks[1].block_name, "layer1");
    assert!(
        result.blocks[1].qk_matmul_width.is_some(),
        "Block 1 should have Q@K^T width"
    );

    // Q@K^T bounds should be finite (not NaN or inf)
    let qk0 = result.blocks[0].qk_matmul_width.unwrap();
    let qk1 = result.blocks[1].qk_matmul_width.unwrap();

    // Bounds should be finite - exact tightness depends on zonotope path detection
    // which may not trigger for all graph structures
    assert!(qk0.is_finite(), "Q@K^T should be finite, got {}", qk0);
    assert!(qk1.is_finite(), "Q@K^T should be finite, got {}", qk1);
    assert!(qk0 < 1e10, "Q@K^T should not saturate, got {}", qk0);
    assert!(qk1 < 1e10, "Q@K^T should not saturate, got {}", qk1);

    // No degradation (NaN/inf) expected for small epsilon
    assert_eq!(result.degraded_blocks, 0, "Expected no degraded blocks");
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_verification_four_blocks() {
    let mut graph = GraphNetwork::new();
    let seq = 4;
    let hidden = 8;
    let epsilon = 0.1;

    let q_weight = Array2::from_shape_fn((hidden, hidden), |(i, j)| if i == j { 0.1 } else { 0.0 });
    let k_weight = Array2::from_shape_fn((hidden, hidden), |(i, j)| if i == j { 0.1 } else { 0.0 });

    for block_index in 0..4 {
        add_transformer_block(&mut graph, block_index, hidden, &q_weight, &k_weight);
    }

    graph.set_output("layer3_qk_matmul");

    let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[seq, hidden])), epsilon).unwrap();
    let result = graph.propagate_ibp_block_wise(&input, epsilon).unwrap();

    assert_eq!(
        result.total_blocks, 4,
        "Expected 4 blocks, got {}",
        result.total_blocks
    );
    assert_eq!(result.blocks.len(), 4);
    assert_eq!(result.degraded_blocks, 0, "Expected no degraded blocks");
    assert!(
        result.max_sensitivity.is_finite(),
        "Max sensitivity should be finite"
    );

    for (idx, block) in result.blocks.iter().enumerate() {
        let qk_width = block.qk_matmul_width.unwrap_or_else(|| {
            panic!("Block {} should have Q@K^T width", idx);
        });
        assert!(
            qk_width.is_finite(),
            "Block {} Q@K^T width was {}",
            idx,
            qk_width
        );
        assert!(
            qk_width < 1e10,
            "Block {} Q@K^T saturated at {}",
            idx,
            qk_width
        );
        assert!(!block.degraded, "Block {} degraded unexpectedly", idx);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_checkpoint_callback_reports_each_completed_block_2519() {
    let graph = make_relu_checkpoint_graph(2);
    let input = scalar_input_bounds();
    let callbacks = RefCell::new(Vec::<(usize, String, usize, f32, f32, u64, usize)>::new());

    let result = graph
        .propagate_ibp_block_wise_with_checkpoint(
            &input,
            0.1,
            None::<fn(BlockProgress)>,
            Some(
                |block: &BlockBoundsInfo, elapsed_ms: u64, total_blocks: usize| {
                    callbacks.borrow_mut().push((
                        block.block_index,
                        block.block_name.clone(),
                        block.nodes.len(),
                        block.input_width,
                        block.output_width,
                        elapsed_ms,
                        total_blocks,
                    ));
                },
            ),
            0,
            None,
        )
        .expect("checkpoint-enabled block-wise IBP should succeed");

    let callbacks = callbacks.into_inner();
    assert_eq!(
        callbacks.iter().map(|entry| entry.0).collect::<Vec<_>>(),
        vec![0, 1],
        "callbacks should follow block execution order"
    );
    assert!(
        callbacks[1].5 >= callbacks[0].5,
        "elapsed_ms should be nondecreasing across callbacks"
    );
    assert_checkpoint_callback_matches_blocks(&callbacks, &result, 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_checkpoint_resume_skips_completed_blocks_2519() {
    let graph = make_relu_checkpoint_graph(3);
    let input = scalar_input_bounds();
    let baseline = graph
        .propagate_ibp_block_wise(&input, 0.1)
        .expect("baseline block-wise IBP should succeed");
    let checkpoint = checkpoint_after_first_block(&baseline, 123);
    let resumed_callbacks = RefCell::new(Vec::<(usize, u64, usize)>::new());

    let resumed = graph
        .propagate_ibp_block_wise_with_checkpoint(
            &input,
            0.1,
            None::<fn(BlockProgress)>,
            Some(
                |block: &BlockBoundsInfo, elapsed_ms: u64, total_blocks: usize| {
                    resumed_callbacks.borrow_mut().push((
                        block.block_index,
                        elapsed_ms,
                        total_blocks,
                    ));
                },
            ),
            0,
            Some(&checkpoint),
        )
        .expect("resumed block-wise IBP should succeed");

    let resumed_callbacks = resumed_callbacks.into_inner();
    assert_eq!(
        resumed_callbacks
            .iter()
            .map(|entry| entry.0)
            .collect::<Vec<_>>(),
        vec![1, 2],
        "resume should skip blocks already present in the checkpoint"
    );
    assert!(
        resumed_callbacks
            .iter()
            .all(|(_, elapsed_ms, total_blocks)| *elapsed_ms >= 123 && *total_blocks == 3),
        "resume callbacks should include prior elapsed_ms and full block count"
    );
    assert_eq!(
        resumed
            .blocks
            .iter()
            .map(|block| block.block_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2],
        "resumed result should contain checkpointed and newly computed blocks"
    );
    assert_eq!(
        resumed.blocks[0].block_name, baseline.blocks[0].block_name,
        "checkpointed block should be preserved in the resumed result"
    );
    assert!(
        (resumed.max_sensitivity - baseline.max_sensitivity).abs() < f32::EPSILON,
        "resumed max sensitivity {} should match baseline {}",
        resumed.max_sensitivity,
        baseline.max_sensitivity
    );
    assert_eq!(
        resumed.degraded_blocks, baseline.degraded_blocks,
        "resume should preserve degraded-block accounting"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_where_matches_scalar_union_bounds_2519() {
    let mut graph = GraphNetwork::new();
    let epsilon = 0.1;
    let passthrough = Layer::AddConstant(AddConstantLayer::new(ArrayD::zeros(IxDyn(&[1]))));

    graph.add_node(GraphNode::from_input("layer0_x", passthrough));
    graph.add_node(GraphNode::new(
        "layer0_y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["layer0_x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "layer0_cond",
        Layer::ReLU(ReLULayer),
        vec!["layer0_x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "layer0_where",
        Layer::Where(WhereLayer::new()),
        vec![
            "layer0_cond".to_string(),
            "layer0_x".to_string(),
            "layer0_y".to_string(),
        ],
    ));
    graph.set_output("layer0_where");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    )
    .expect("valid scalar input");
    let reset_input = scalar_block_reset_bounds(epsilon);

    let baseline = graph
        .propagate_ibp(&reset_input)
        .expect("reset-input IBP should succeed on scalar Where");
    let result = graph
        .propagate_ibp_block_wise(&input, epsilon)
        .expect("block-wise IBP should succeed on scalar Where");

    assert_eq!(result.total_blocks, 1, "Where graph should form one block");
    assert_eq!(
        result.degraded_blocks, 0,
        "Where union bounds should stay finite"
    );
    assert_scalar_node_matches_bounds(&result, "layer0_where", &baseline);
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_skip_merge_preserves_scalar_bounds_2519() {
    let mut graph = GraphNetwork::new();
    let epsilon = 0.1;
    let passthrough = Layer::AddConstant(AddConstantLayer::new(ArrayD::zeros(IxDyn(&[1]))));

    graph.add_node(GraphNode::from_input("layer0_input", passthrough));
    graph.add_node(GraphNode::new(
        "layer0_skip",
        Layer::SkipMerge(SkipMergeLayer::new()),
        vec!["layer0_input".to_string()],
    ));
    graph.set_output("layer0_skip");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.75]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.25]).unwrap(),
    )
    .expect("valid scalar input");
    let reset_input = scalar_block_reset_bounds(epsilon);

    let baseline = graph
        .propagate_ibp(&reset_input)
        .expect("reset-input IBP should succeed on SkipMerge");
    let result = graph
        .propagate_ibp_block_wise(&input, epsilon)
        .expect("block-wise IBP should succeed on SkipMerge");

    assert_eq!(
        result.total_blocks, 1,
        "SkipMerge graph should form one block"
    );
    assert_eq!(
        result.degraded_blocks, 0,
        "SkipMerge should not degrade bounds"
    );
    assert_scalar_node_matches_bounds(&result, "layer0_skip", &baseline);
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_opaque_skip_preserves_unbounded_scalar_semantics_2519() {
    let mut graph = GraphNetwork::new();
    let epsilon = 0.1;
    let passthrough = Layer::AddConstant(AddConstantLayer::new(ArrayD::zeros(IxDyn(&[1]))));

    graph.add_node(GraphNode::from_input("layer0_input", passthrough));
    graph.add_node(GraphNode::new(
        "layer0_opaque",
        Layer::OpaqueSkip(OpaqueSkipLayer::new()),
        vec!["layer0_input".to_string()],
    ));
    graph.set_output("layer0_opaque");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.75]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.25]).unwrap(),
    )
    .expect("valid scalar input");
    let reset_input = scalar_block_reset_bounds(epsilon);

    let baseline = graph
        .propagate_ibp(&reset_input)
        .expect("reset-input IBP should succeed on OpaqueSkip");
    let result = graph
        .propagate_ibp_block_wise(&input, epsilon)
        .expect("block-wise IBP should succeed on OpaqueSkip");
    let node = block_node_info(&result, "layer0_opaque");

    assert_eq!(
        result.total_blocks, 1,
        "OpaqueSkip graph should form one block"
    );
    assert_eq!(
        result.degraded_blocks, 1,
        "OpaqueSkip should mark the block degraded because bounds are infinite"
    );
    assert!(
        result.blocks[0].degraded,
        "OpaqueSkip block should be flagged as degraded"
    );
    assert!(
        node.has_infinite,
        "OpaqueSkip node should report infinite bounds"
    );
    assert_scalar_node_matches_bounds(&result, "layer0_opaque", &baseline);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_block_index() {
    // Test block index parsing from node names
    assert_eq!(GraphNetwork::parse_block_index("layer0_attn_norm"), Some(0));
    assert_eq!(GraphNetwork::parse_block_index("layer12_q_proj"), Some(12));
    assert_eq!(GraphNetwork::parse_block_index("layer127_add2"), Some(127));
    assert_eq!(GraphNetwork::parse_block_index("embedding"), None);
    assert_eq!(GraphNetwork::parse_block_index("output_norm"), None);
    assert_eq!(GraphNetwork::parse_block_index("layernorm"), None); // No number
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_supports_self_attention_options_and_checkpoint_2472() {
    let mut graph = GraphNetwork::new();
    let passthrough = Layer::AddConstant(AddConstantLayer::new(ArrayD::zeros(IxDyn(&[1]))));

    graph.add_node(GraphNode::from_input("layer0_q", passthrough.clone()));
    graph.add_node(GraphNode::from_input("layer0_k", passthrough.clone()));
    graph.add_node(GraphNode::from_input("layer0_v", passthrough));
    graph.add_node(GraphNode::new(
        "layer0_attn",
        Layer::SelfAttention(SelfAttentionLayer::standard()),
        vec![
            "layer0_q".to_string(),
            "layer0_k".to_string(),
            "layer0_v".to_string(),
        ],
    ));
    graph.set_output("layer0_attn");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 1.0_f32),
    )
    .expect("valid input bounds");

    let options_result = graph
        .propagate_ibp_block_wise_with_options(&input, 0.1, None::<fn(BlockProgress)>, 0)
        .expect("block-wise options path must support SelfAttention");
    assert_eq!(options_result.total_blocks, 1);
    assert!(options_result.blocks[0]
        .nodes
        .iter()
        .any(|n| n.layer_type == "SelfAttention"));

    let checkpoint_result = graph
        .propagate_ibp_block_wise_with_checkpoint(
            &input,
            0.1,
            None::<fn(BlockProgress)>,
            None::<fn(&crate::types::BlockBoundsInfo, u64, usize)>,
            0,
            None,
        )
        .expect("block-wise checkpoint path must support SelfAttention");
    assert_eq!(checkpoint_result.total_blocks, 1);
    assert!(checkpoint_result.blocks[0]
        .nodes
        .iter()
        .any(|n| n.layer_type == "SelfAttention"));
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_supports_nary_concat_2472() {
    let mut graph = GraphNetwork::new();
    let passthrough = Layer::AddConstant(AddConstantLayer::new(ArrayD::zeros(IxDyn(&[1]))));

    graph.add_node(GraphNode::from_input("layer0_a", passthrough.clone()));
    graph.add_node(GraphNode::from_input("layer0_b", passthrough.clone()));
    graph.add_node(GraphNode::from_input("layer0_c", passthrough));
    graph.add_node(GraphNode::new(
        "layer0_concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec![
            "layer0_a".to_string(),
            "layer0_b".to_string(),
            "layer0_c".to_string(),
        ],
    ));
    graph.set_output("layer0_concat");

    let input = BoundedTensor::new(
        ndarray::arr1(&[-1.0_f32, -0.5]).into_dyn(),
        ndarray::arr1(&[1.0_f32, 0.5]).into_dyn(),
    )
    .expect("valid input bounds");

    let result = graph
        .propagate_ibp_block_wise(&input, 0.1)
        .expect("block-wise propagation must support n-ary concat");
    assert_eq!(result.total_blocks, 1);
    let concat_info = result.blocks[0]
        .nodes
        .iter()
        .find(|n| n.name == "layer0_concat")
        .expect("concat node info must exist");
    assert_eq!(concat_info.output_shape, vec![6]);
}
