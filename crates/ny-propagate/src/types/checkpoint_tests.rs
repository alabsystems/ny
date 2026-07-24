// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`VerificationCheckpoint`].

use super::VerificationCheckpoint;
use crate::types::BlockBoundsInfo;

fn make_block_info_at(block_index: usize, sensitivity: f32, degraded: bool) -> BlockBoundsInfo {
    BlockBoundsInfo {
        block_index,
        block_name: format!("block{}", block_index),
        nodes: vec![],
        input_width: 0.1,
        output_width: sensitivity * 0.1,
        sensitivity,
        qk_matmul_width: None,
        swiglu_width: None,
        degraded,
    }
}

fn make_block_info(sensitivity: f32, degraded: bool) -> BlockBoundsInfo {
    BlockBoundsInfo {
        block_index: 0,
        block_name: "block0".to_string(),
        nodes: vec![],
        input_width: 0.1,
        output_width: sensitivity * 0.1,
        sensitivity,
        qk_matmul_width: None,
        swiglu_width: None,
        degraded,
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_new() {
    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );
    assert_eq!(checkpoint.version, VerificationCheckpoint::VERSION);
    assert_eq!(checkpoint.epsilon, 0.01);
    assert_eq!(checkpoint.method, "alpha");
    assert_eq!(checkpoint.backend, "cpu");
    assert_eq!(checkpoint.total_blocks, 10);
    assert_eq!(checkpoint.next_block_index, 0);
    assert!(checkpoint.completed_blocks.is_empty());
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_update() {
    let mut checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );

    let block = make_block_info(50.0, false);
    checkpoint.update(block, 1000);

    assert_eq!(checkpoint.completed_blocks.len(), 1);
    assert_eq!(checkpoint.next_block_index, 1);
    assert!((checkpoint.max_sensitivity - 50.0).abs() < 1e-6);
    assert_eq!(checkpoint.degraded_blocks, 0);
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_update_degraded() {
    let mut checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );

    let block = make_block_info(50.0, true);
    checkpoint.update(block, 1000);

    assert_eq!(checkpoint.degraded_blocks, 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_is_complete() {
    let mut checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        2,
    );

    assert!(!checkpoint.is_complete());

    checkpoint.next_block_index = 2;
    assert!(checkpoint.is_complete());
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_into_result() {
    let mut checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        2,
    );

    // Use update() to properly advance next_block_index
    checkpoint.update(make_block_info_at(0, 10.0, false), 100);
    checkpoint.update(make_block_info_at(1, 20.0, false), 200);
    assert!(checkpoint.is_complete());

    let result = checkpoint
        .into_result()
        .expect("complete checkpoint should convert");
    assert_eq!(result.blocks.len(), 2);
    assert_eq!(result.total_blocks, 2);
    assert!((result.max_sensitivity - 20.0).abs() < 1e-6);
}

/// Regression test for #2808: into_result() must return Err on incomplete checkpoints.
/// Before fix, into_result() silently produced a BlockWiseResult with
/// blocks.len() < total_blocks, giving callers inconsistent data.
#[ntest::timeout(5000)]
#[test]
fn test_into_result_returns_err_on_incomplete_checkpoint_2808() {
    let mut checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        3, // 3 total blocks
    );

    // Only complete 1 of 3 blocks
    checkpoint.update(make_block_info(10.0, false), 100);
    assert!(!checkpoint.is_complete());

    // Must return Err — converting an incomplete checkpoint is invalid
    let result = checkpoint.into_result();
    assert!(result.is_err(), "incomplete checkpoint must return Err");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("incomplete checkpoint"),
        "error should mention 'incomplete checkpoint', got: {err_msg}"
    );
}

/// Audit finding: out-of-order update() can advance next_block_index past
/// total_blocks without processing all blocks. The second check in
/// into_result() catches this (completed_blocks.len() != total_blocks).
#[ntest::timeout(5000)]
#[test]
fn test_into_result_returns_err_on_out_of_order_update_2808() {
    let mut checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        3, // 3 total blocks (indices 0, 1, 2)
    );

    // Only process block 2 (skipping 0 and 1) — next_block_index jumps to 3
    checkpoint.update(make_block_info_at(2, 10.0, false), 100);
    // is_complete() returns true (3 >= 3) but only 1 block was processed!
    assert!(checkpoint.is_complete());
    assert_eq!(checkpoint.completed_blocks.len(), 1);

    // Must return Err: completed_blocks.len() (1) != total_blocks (3)
    let result = checkpoint.into_result();
    assert!(result.is_err(), "out-of-order update must return Err");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("block count mismatch"),
        "error should mention 'block count mismatch', got: {err_msg}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_validate_success() {
    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );

    assert!(checkpoint
        .validate(
            std::path::Path::new("/model.onnx"),
            "abc123",
            0.01,
            "alpha",
            "cpu"
        )
        .is_ok());
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_validate_path_mismatch() {
    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );

    let result = checkpoint.validate(
        std::path::Path::new("/other.onnx"),
        "abc123",
        0.01,
        "alpha",
        "cpu",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("model path"));
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_validate_hash_mismatch() {
    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );

    let result = checkpoint.validate(
        std::path::Path::new("/model.onnx"),
        "different",
        0.01,
        "alpha",
        "cpu",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hash"));
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_validate_epsilon_mismatch() {
    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );

    let result = checkpoint.validate(
        std::path::Path::new("/model.onnx"),
        "abc123",
        0.02, // Different epsilon
        "alpha",
        "cpu",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("epsilon"));
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_validate_method_mismatch() {
    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );

    let result = checkpoint.validate(
        std::path::Path::new("/model.onnx"),
        "abc123",
        0.01,
        "ibp", // Different method
        "cpu",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("method"));
}

#[ntest::timeout(5000)]
#[test]
fn test_verification_checkpoint_validate_backend_mismatch() {
    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        10,
    );

    let result = checkpoint.validate(
        std::path::Path::new("/model.onnx"),
        "abc123",
        0.01,
        "alpha",
        "wgpu", // Different backend
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("backend"));
}

/// Regression: NaN sensitivity must propagate through checkpoint.update(),
/// not be silently absorbed by `>` comparison (IEEE 754: NaN > x == false).
#[ntest::timeout(5000)]
#[test]
fn test_checkpoint_update_nan_sensitivity_propagates() {
    let mut checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("/model.onnx"),
        "abc123".to_string(),
        0.01,
        "alpha",
        "cpu",
        3,
    );

    // First block: normal sensitivity
    checkpoint.update(make_block_info(5.0, false), 100);
    assert!((checkpoint.max_sensitivity - 5.0).abs() < 1e-6);

    // Second block: NaN sensitivity (corrupted propagation)
    checkpoint.update(make_block_info(f32::NAN, false), 200);
    assert!(
        checkpoint.max_sensitivity.is_nan(),
        "NaN sensitivity must propagate, not be absorbed: got {}",
        checkpoint.max_sensitivity
    );

    // Third block: normal — NaN must persist (once corrupted, stays corrupted)
    checkpoint.update(make_block_info(10.0, false), 300);
    assert!(
        checkpoint.max_sensitivity.is_nan(),
        "NaN must persist after further updates: got {}",
        checkpoint.max_sensitivity
    );
}
