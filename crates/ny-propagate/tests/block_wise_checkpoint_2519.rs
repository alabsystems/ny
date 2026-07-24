// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for block-wise checkpoint resume and callback paths (#2519).
//!
//! Covers:
//! - Resume-from-checkpoint skipping completed blocks (block_wise.rs:141-151)
//! - Checkpoint callback invocation with correct arguments (block_wise.rs:246-249)

use ndarray::arr1;
use ny_propagate::layers::ReLULayer;
use ny_propagate::{
    BlockBoundsInfo, BlockProgress, GraphNetwork, GraphNode, Layer, VerificationCheckpoint,
    NETWORK_INPUT,
};
use ny_tensor::BoundedTensor;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Build a 2-block graph: layer0_relu and layer1_relu, both taking _input.
/// Each block is a single ReLU node, which is the simplest IBP-propagable layer.
fn make_two_block_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "layer0_relu",
        Layer::ReLU(ReLULayer::new()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "layer1_relu",
        Layer::ReLU(ReLULayer::new()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph
}

/// #2519: Resume-from-checkpoint skips already-completed blocks.
///
/// Verifies the code path at block_wise.rs:141-151 where `resume_from`
/// populates `(blocks, max_sensitivity, degraded_blocks, skip_blocks)`.
#[test]
fn test_block_wise_resume_from_checkpoint_skips_completed_blocks_2519() {
    let graph = make_two_block_graph();
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds");

    // First: run full block-wise to get reference results for both blocks.
    let full_result = graph
        .propagate_ibp_block_wise(&input, 0.1)
        .expect("full block-wise should succeed");
    assert_eq!(
        full_result.blocks.len(),
        2,
        "expected 2 blocks, got {}",
        full_result.blocks.len()
    );

    // Build a checkpoint that says block 0 is done, resume from block 1.
    let block0 = full_result.blocks[0].clone();
    let checkpoint = VerificationCheckpoint {
        version: VerificationCheckpoint::VERSION,
        model_path: PathBuf::from("test"),
        model_hash: String::new(),
        epsilon: 0.1,
        method: "ibp".to_string(),
        backend: "cpu".to_string(),
        start_time: String::new(),
        checkpoint_time: String::new(),
        elapsed_ms: 100,
        completed_blocks: vec![block0.clone()],
        max_sensitivity: block0.sensitivity,
        degraded_blocks: 0,
        total_blocks: 2,
        next_block_index: 1,
    };

    // Resume from checkpoint — should process only block 1.
    let resumed_result = graph
        .propagate_ibp_block_wise_with_checkpoint(
            &input,
            0.1,
            None::<fn(BlockProgress)>,
            None::<fn(&BlockBoundsInfo, u64, usize)>,
            0,
            Some(&checkpoint),
        )
        .expect("resumed block-wise should succeed");

    // Resumed result should have 2 blocks: 1 from checkpoint + 1 freshly computed.
    assert_eq!(
        resumed_result.blocks.len(),
        2,
        "resumed should have 2 blocks (1 checkpointed + 1 new)"
    );
    // Block 0 should be the checkpoint's copy.
    assert_eq!(resumed_result.blocks[0].block_index, block0.block_index);
    assert_eq!(resumed_result.blocks[0].block_name, block0.block_name);
    // Block 1 should match the full run's block 1.
    assert_eq!(
        resumed_result.blocks[1].block_index,
        full_result.blocks[1].block_index
    );
    assert_eq!(
        resumed_result.blocks[1].block_name,
        full_result.blocks[1].block_name
    );
    // Sensitivity should match (same graph, same epsilon, same blocks).
    assert!(
        (resumed_result.blocks[1].sensitivity - full_result.blocks[1].sensitivity).abs() < 1e-6,
        "block 1 sensitivity mismatch: resumed {} vs full {}",
        resumed_result.blocks[1].sensitivity,
        full_result.blocks[1].sensitivity
    );
}

/// #2519: Checkpoint callback is invoked with correct BlockBoundsInfo and elapsed_ms.
///
/// Verifies the code path at block_wise.rs:246-249 where the checkpoint_callback
/// receives (block_info, elapsed_ms, total_blocks) after each block.
#[test]
fn test_block_wise_checkpoint_callback_invoked_2519() {
    let graph = make_two_block_graph();
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds");

    let captured: Arc<Mutex<Vec<(BlockBoundsInfo, u64, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);

    let checkpoint_cb = move |info: &BlockBoundsInfo, elapsed_ms: u64, total: usize| {
        captured_clone
            .lock()
            .unwrap()
            .push((info.clone(), elapsed_ms, total));
    };

    let result = graph
        .propagate_ibp_block_wise_with_checkpoint(
            &input,
            0.1,
            None::<fn(BlockProgress)>,
            Some(checkpoint_cb),
            0,
            None,
        )
        .expect("block-wise with checkpoint callback should succeed");

    let calls = captured.lock().unwrap();
    assert_eq!(
        calls.len(),
        2,
        "checkpoint callback should be called once per block, got {} calls",
        calls.len()
    );

    // First call: block 0
    assert_eq!(
        calls[0].0.block_index, 0,
        "first callback should be block 0"
    );
    assert_eq!(calls[0].2, 2, "total_blocks should be 2");

    // Second call: block 1
    assert_eq!(
        calls[1].0.block_index, 1,
        "second callback should be block 1"
    );
    assert_eq!(calls[1].2, 2, "total_blocks should be 2");
    // elapsed_ms should be monotonically non-decreasing
    assert!(
        calls[1].1 >= calls[0].1,
        "elapsed_ms should be monotonically non-decreasing: {} < {}",
        calls[1].1,
        calls[0].1
    );

    // Result should have the same blocks as without callback.
    assert_eq!(result.blocks.len(), 2);
}
