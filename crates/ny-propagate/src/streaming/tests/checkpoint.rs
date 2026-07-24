// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::streaming::*;

fn constant_bounds(dim: usize, lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[dim]), lower),
        ArrayD::from_elem(ndarray::IxDyn(&[dim]), upper),
    )
    .unwrap()
}

fn assert_uniform_bounds(bounds: &BoundedTensor, lower: f32, upper: f32) {
    for (&actual_lower, &actual_upper) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert_eq!(
            actual_lower, lower,
            "expected uniform lower bound {lower}, got {actual_lower}"
        );
        assert_eq!(
            actual_upper, upper,
            "expected uniform upper bound {upper}, got {actual_upper}"
        );
    }
}

/// Like `assert_uniform_bounds` but tolerates subnormal FP noise (|diff| < f32::MIN_POSITIVE).
/// Use for bounds that come from network propagation rather than hand-constructed fixtures.
fn assert_uniform_bounds_near(bounds: &BoundedTensor, lower: f32, upper: f32) {
    let tol = f32::MIN_POSITIVE; // ≈1.175e-38, just above subnormal range
    for (&actual_lower, &actual_upper) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(
            (actual_lower - lower).abs() < tol,
            "expected lower bound ~{lower}, got {actual_lower} (diff: {})",
            actual_lower - lower
        );
        assert!(
            (actual_upper - upper).abs() < tol,
            "expected upper bound ~{upper}, got {actual_upper} (diff: {})",
            actual_upper - upper
        );
        assert!(
            actual_lower <= actual_upper,
            "inverted bounds: lower {actual_lower} > upper {actual_upper}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_basic() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 20);

    // Add checkpoints at layers 4, 9, 14, 19
    for i in [4, 9, 14, 19] {
        let bounds = create_input(10);
        checkpointed.add_checkpoint(i, bounds);
    }

    assert_eq!(checkpointed.num_checkpoints(), 4);
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_checkpoints() {
    let network = create_test_network(20, 10, 10);
    let input = create_input(10);

    let config = StreamingConfig {
        checkpoint_interval: 5,
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // With interval=5 and 20 layers, checkpoints at: 4, 9, 14, 19
    assert_eq!(checkpointed.num_checkpoints(), 4);
}

#[ntest::timeout(10000)]
#[test]
fn test_find_nearest_checkpoint() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

    // Add checkpoints at layers 4, 9, 14
    checkpointed.add_checkpoint(4, constant_bounds(10, -4.0, 4.0));
    checkpointed.add_checkpoint(9, constant_bounds(10, -9.0, 9.0));
    checkpointed.add_checkpoint(14, constant_bounds(10, -14.0, 14.0));

    // Test find_nearest_checkpoint_before
    // Layer 6 should find checkpoint at 4
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(6).unwrap();
    assert_eq!(idx, 4);
    assert_uniform_bounds(&bounds, -4.0, 4.0);

    // Layer 12 should find checkpoint at 9
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(12).unwrap();
    assert_eq!(idx, 9);
    assert_uniform_bounds(&bounds, -9.0, 9.0);

    // Layer 2 should find no checkpoint (returns -1)
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(2).unwrap();
    assert_eq!(idx, -1);
    assert_eq!(bounds.lower(), input.lower());
    assert_eq!(bounds.upper(), input.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_is_compressed_f32() {
    let input = create_input(10);
    let checkpointed = CheckpointedBounds::new(input, 10);
    assert!(!checkpointed.is_compressed());
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_compression_stats_f32() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 10);
    checkpointed.add_checkpoint(0, create_input(10));

    // f32 storage should return None for compression stats
    assert!(checkpointed.compression_stats().is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_empty_checkpoints() {
    let input = create_input(10);
    let checkpointed = CheckpointedBounds::new(input, 10);

    assert_eq!(checkpointed.num_checkpoints(), 0);
    assert!(checkpointed.last_checkpoint().unwrap().is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_last_checkpoint_f32() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 10);

    let bounds1 = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[10]), -1.0_f32),
        ArrayD::from_elem(ndarray::IxDyn(&[10]), 1.0_f32),
    )
    .unwrap();
    let bounds2 = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[10]), -2.0_f32),
        ArrayD::from_elem(ndarray::IxDyn(&[10]), 2.0_f32),
    )
    .unwrap();

    checkpointed.add_checkpoint(3, bounds1);
    checkpointed.add_checkpoint(7, bounds2);

    let last = checkpointed.last_checkpoint().unwrap();
    assert!(last.is_some());
    let last_bounds = last.unwrap();
    // Should be bounds2 since it's at the highest index
    assert!((last_bounds.lower()[0] - (-2.0)).abs() < 1e-6);
    assert!((last_bounds.upper()[0] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_get_bounds_at_invalid_layer() {
    let input = create_input(10);
    let checkpointed = CheckpointedBounds::new(input, 5);
    let network = create_test_network(5, 10, 10);

    // Layer 10 is out of range (max is 4)
    let result = checkpointed.bounds_at(10, &network);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_get_bounds_at_exact_checkpoint() {
    let input = create_input(8);
    let network = create_test_network(10, 8, 8);

    let config = StreamingConfig {
        checkpoint_interval: 3,
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);
    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Layer 2 should be a checkpoint (index 2, which is (2+1)%3 == 0)
    // Actually with interval=3, checkpoints at 2, 5, 8, 9
    let bounds = checkpointed.bounds_at(2, &network).unwrap();

    // With zero-weight network and [-1, 1] input, all layer outputs are mathematically zero.
    // Multi-layer propagation introduces subnormal FP noise (≤1e-45), so use near-zero check.
    assert_eq!(bounds.lower().shape(), &[8]);
    assert_uniform_bounds_near(&bounds, 0.0, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_get_bounds_at_recomputes_after_checkpoint_boundary() {
    let input = create_input(8);
    let network = create_test_network(10, 8, 8);

    let config = StreamingConfig {
        checkpoint_interval: 3,
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);
    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Layer 3 sits immediately after checkpoint layer 2, so this must use the
    // recomputation path instead of the exact-checkpoint fast path.
    let (start_idx, start_bounds) = checkpointed.find_nearest_checkpoint_before(3).unwrap();
    assert_eq!(start_idx, 2);
    assert_uniform_bounds_near(&start_bounds, 0.0, 0.0);

    let recomputed = checkpointed.bounds_at(3, &network).unwrap();
    assert_uniform_bounds_near(&recomputed, 0.0, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_find_nearest_checkpoint_at_exact() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 20);

    checkpointed.add_checkpoint(5, constant_bounds(10, -5.0, 5.0));
    checkpointed.add_checkpoint(10, constant_bounds(10, -10.0, 10.0));

    // At exact checkpoint should return that checkpoint
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(5).unwrap();
    assert_eq!(idx, 5);
    assert_uniform_bounds(&bounds, -5.0, 5.0);

    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(10).unwrap();
    assert_eq!(idx, 10);
    assert_uniform_bounds(&bounds, -10.0, 10.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_find_nearest_checkpoint_at_layer_zero() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

    checkpointed.add_checkpoint(5, constant_bounds(10, -5.0, 5.0));

    // Layer 0 should return -1 (no checkpoint before layer 0)
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(0).unwrap();
    assert_eq!(idx, -1);
    assert_eq!(bounds.lower(), input.lower());
    assert_eq!(bounds.upper(), input.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_memory_bytes_empty() {
    let input = create_input(10);
    let checkpointed = CheckpointedBounds::new(input, 10);

    // Just input memory (10 elements * 4 bytes * 2 (lower+upper))
    let mem = checkpointed.memory_bytes();
    assert_eq!(mem, 10 * 4 * 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_memory_bytes_with_checkpoints() {
    let input = create_input(100);
    let mut checkpointed = CheckpointedBounds::new(input, 10);

    for i in 0..5 {
        checkpointed.add_checkpoint(i, create_input(100));
    }

    // input + 5 checkpoints, each 100 elements * 4 bytes * 2
    let expected = (1 + 5) * 100 * 4 * 2;
    assert_eq!(checkpointed.memory_bytes(), expected);
}

#[ntest::timeout(10000)]
#[test]
fn test_add_checkpoint_maintains_sort_order() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 20);

    // Add checkpoints out of order
    checkpointed.add_checkpoint(10, constant_bounds(10, -10.0, 10.0));
    checkpointed.add_checkpoint(5, constant_bounds(10, -5.0, 5.0));
    checkpointed.add_checkpoint(15, constant_bounds(10, -15.0, 15.0));
    checkpointed.add_checkpoint(0, constant_bounds(10, 0.0, 0.0));

    assert_eq!(checkpointed.num_checkpoints(), 4);

    // Verify ordering by finding nearest checkpoint
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(3).unwrap();
    assert_eq!(idx, 0);
    assert_uniform_bounds(&bounds, 0.0, 0.0);

    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(7).unwrap();
    assert_eq!(idx, 5);
    assert_uniform_bounds(&bounds, -5.0, 5.0);

    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(12).unwrap();
    assert_eq!(idx, 10);
    assert_uniform_bounds(&bounds, -10.0, 10.0);

    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(20).unwrap();
    assert_eq!(idx, 15);
    assert_uniform_bounds(&bounds, -15.0, 15.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpoint_interval_zero_clamped() {
    let network = create_test_network(5, 8, 8);
    let input = create_input(8);

    let config = StreamingConfig {
        checkpoint_interval: 0, // Should be clamped to 1
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // With interval clamped to 1, should have 5 checkpoints (one per layer)
    assert_eq!(checkpointed.num_checkpoints(), 5);
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_checkpoints_empty_network() {
    let network = Network::new();
    let input = create_input(10);

    let config = StreamingConfig::default();
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    assert_eq!(checkpointed.num_checkpoints(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpoint_interval_larger_than_network() {
    let network = create_test_network(3, 8, 8);
    let input = create_input(8);

    let config = StreamingConfig {
        checkpoint_interval: 100, // Much larger than network size
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Should have exactly 1 checkpoint (at the last layer)
    assert_eq!(checkpointed.num_checkpoints(), 1);
}
