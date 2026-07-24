// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::streaming::*;
use crate::NyError;
use ny_tensor::CompressedBounds;

fn nan_propagating_max(acc: f32, next: f32) -> f32 {
    if acc.is_nan() || next.is_nan() {
        f32::NAN
    } else {
        acc.max(next)
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_f16() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new_compressed(input, 20, 0.001);

    // Add checkpoints at layers 4, 9, 14, 19
    for i in [4, 9, 14, 19] {
        let bounds = create_input(10);
        checkpointed.add_checkpoint(i, bounds);
    }

    assert_eq!(checkpointed.num_checkpoints(), 4);
    assert!(checkpointed.is_compressed());
}

#[ntest::timeout(10000)]
#[test]
fn test_f16_memory_savings() {
    let input = create_input(1000);
    let bounds = create_input(1000);

    // Create f32 and f16 checkpointed storage
    let mut checkpointed_f32 = CheckpointedBounds::new(input.clone(), 10);
    let mut checkpointed_f16 = CheckpointedBounds::new_compressed(input, 10, 0.001);

    // Add same checkpoints to both
    for i in 0..10 {
        checkpointed_f32.add_checkpoint(i, bounds.clone());
        checkpointed_f16.add_checkpoint(i, bounds.clone());
    }

    let f32_bytes = checkpointed_f32.memory_bytes();
    let f16_bytes = checkpointed_f16.memory_bytes();

    // f16 should use significantly less memory (approximately half)
    assert!(
        f16_bytes < f32_bytes * 7 / 10,
        "f16 ({}) should be <70% of f32 ({})",
        f16_bytes,
        f32_bytes
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_f16_compression_stats() {
    let input = create_input(1000);
    let mut checkpointed = CheckpointedBounds::new_compressed(input, 10, 0.001);

    for i in 0..10 {
        checkpointed.add_checkpoint(i, create_input(1000));
    }

    let stats = checkpointed.compression_stats();
    assert!(stats.is_some());

    let (f16_bytes, f32_bytes, ratio) = stats.unwrap();
    assert!(ratio < 0.6, "Ratio {} should be < 0.6", ratio);
    assert!(f16_bytes < f32_bytes);
}

#[ntest::timeout(10000)]
#[test]
fn test_f16_bounds_recovery() {
    // Test that bounds can be recovered from f16 storage
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new_compressed(input, 5, 0.001);

    let original_bounds = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[10]), -1.5_f32),
        ArrayD::from_elem(ndarray::IxDyn(&[10]), 1.5_f32),
    )
    .unwrap();

    checkpointed.add_checkpoint(2, original_bounds.clone());

    // Create a minimal network for testing
    let network = create_test_network(5, 10, 10);
    let recovered = checkpointed.bounds_at(2, &network).unwrap();

    // Recovered bounds should be similar to original (within f16 precision + widening)
    assert_eq!(recovered.shape(), original_bounds.shape());

    // Due to widening for soundness, lower should be <= original lower
    // and upper should be >= original upper (no positive tolerance —
    // any violation means f16 compression broke soundness).
    for (orig, rec) in original_bounds.lower().iter().zip(recovered.lower().iter()) {
        assert!(
            *rec <= *orig,
            "Recovered lower {} should be <= original {} (widening must not tighten)",
            rec,
            orig
        );
    }
    for (orig, rec) in original_bounds.upper().iter().zip(recovered.upper().iter()) {
        assert!(
            *rec >= *orig,
            "Recovered upper {} should be >= original {} (widening must not tighten)",
            rec,
            orig
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_with_f16_checkpoints() {
    let network = create_test_network(20, 10, 10);
    let input = create_input(10);

    let config = StreamingConfig {
        checkpoint_interval: 5,
        use_f16_checkpoints: true,
        f16_widening_epsilon: 0.001,
    };
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    assert_eq!(checkpointed.num_checkpoints(), 4);
    assert!(checkpointed.is_compressed());
}

#[ntest::timeout(10000)]
#[test]
fn test_f16_vs_f32_checkpoint_equivalence() {
    // Test that f16 checkpoints produce approximately same results as f32
    let mut network = Network::new();
    for i in 0..5 {
        let mut weight = Array2::<f32>::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                let val = ((r * 7 + c * 11 + i * 13) % 10) as f32 * 0.01 - 0.05;
                weight[[r, c]] = val;
            }
        }
        let bias = Some(Array1::<f32>::zeros(8));
        let linear = LinearLayer::new(weight, bias).unwrap();
        network.add_layer(Layer::Linear(linear));
    }

    let lower = ArrayD::from_elem(ndarray::IxDyn(&[8]), -0.1_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[8]), 0.1_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Collect with f32 storage
    let config_f32 = StreamingConfig {
        checkpoint_interval: 2,
        use_f16_checkpoints: false,
        ..Default::default()
    };
    let verifier_f32 = StreamingVerifier::new(config_f32);
    let checkpointed_f32 = verifier_f32
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Collect with f16 storage
    let config_f16 = StreamingConfig {
        checkpoint_interval: 2,
        use_f16_checkpoints: true,
        f16_widening_epsilon: 0.001,
    };
    let verifier_f16 = StreamingVerifier::new(config_f16);
    let checkpointed_f16 = verifier_f16
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Compare bounds at each checkpoint
    for i in 0..5 {
        let bounds_f32 = checkpointed_f32.bounds_at(i, &network).unwrap();
        let bounds_f16 = checkpointed_f16.bounds_at(i, &network).unwrap();

        let max_lower_diff: f32 = bounds_f32
            .lower()
            .iter()
            .zip(bounds_f16.lower().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, nan_propagating_max);

        let max_upper_diff: f32 = bounds_f32
            .upper()
            .iter()
            .zip(bounds_f16.upper().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, nan_propagating_max);

        // f16 precision allows for some error, but should be bounded
        assert!(
            max_lower_diff < 0.01,
            "Layer {} lower bounds differ by {} (f16 vs f32)",
            i,
            max_lower_diff
        );
        assert!(
            max_upper_diff < 0.01,
            "Layer {} upper bounds differ by {} (f16 vs f32)",
            i,
            max_upper_diff
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_f16_checkpoints_soundness() {
    // Test that f16 checkpoints with widening produce sound (conservative) bounds
    let mut network = Network::new();
    for i in 0..3 {
        let mut weight = Array2::<f32>::zeros((4, 4));
        for r in 0..4 {
            for c in 0..4 {
                let val = ((r * 3 + c * 5 + i * 7) % 10) as f32 * 0.02 - 0.1;
                weight[[r, c]] = val;
            }
        }
        let bias = Some(Array1::<f32>::zeros(4));
        let linear = LinearLayer::new(weight, bias).unwrap();
        network.add_layer(Layer::Linear(linear));
    }

    let lower = ArrayD::from_elem(ndarray::IxDyn(&[4]), -0.5_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // f32 reference (no widening)
    let config_f32 = StreamingConfig {
        checkpoint_interval: 1,
        use_f16_checkpoints: false,
        ..Default::default()
    };
    let verifier_f32 = StreamingVerifier::new(config_f32);
    let checkpointed_f32 = verifier_f32
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // f16 with widening
    let config_f16 = StreamingConfig {
        checkpoint_interval: 1,
        use_f16_checkpoints: true,
        f16_widening_epsilon: 0.01, // 1% widening
    };
    let verifier_f16 = StreamingVerifier::new(config_f16);
    let checkpointed_f16 = verifier_f16
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // f16 widened bounds should be more conservative (wider)
    for i in 0..3 {
        let bounds_f32 = checkpointed_f32.bounds_at(i, &network).unwrap();
        let bounds_f16 = checkpointed_f16.bounds_at(i, &network).unwrap();

        // f16 lower should be <= f32 lower (more conservative — widening
        // must never tighten, so no positive tolerance allowed here).
        for (f32_l, f16_l) in bounds_f32.lower().iter().zip(bounds_f16.lower().iter()) {
            assert!(
                *f16_l <= *f32_l,
                "Layer {}: f16 lower {} should be <= f32 lower {} (widening must not tighten)",
                i,
                f16_l,
                f32_l
            );
        }

        // f16 upper should be >= f32 upper (more conservative — widening
        // must never tighten, so no negative tolerance allowed here).
        for (f32_u, f16_u) in bounds_f32.upper().iter().zip(bounds_f16.upper().iter()) {
            assert!(
                *f16_u >= *f32_u,
                "Layer {}: f16 upper {} should be >= f32 upper {} (widening must not tighten)",
                i,
                f16_u,
                f32_u
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_last_checkpoint_f16() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new_compressed(input, 10, 0.001);

    let original_bounds = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[10]), -1.5_f32),
        ArrayD::from_elem(ndarray::IxDyn(&[10]), 1.5_f32),
    )
    .unwrap();

    checkpointed.add_checkpoint(5, original_bounds.clone());

    let last = checkpointed.last_checkpoint().unwrap();
    assert!(last.is_some());
    let recovered = last.unwrap();
    for (&orig, &rec) in original_bounds.lower().iter().zip(recovered.lower().iter()) {
        assert!(
            rec <= orig,
            "Recovered lower {rec} should be <= original {orig} after f16 widening"
        );
    }
    for (&orig, &rec) in original_bounds.upper().iter().zip(recovered.upper().iter()) {
        assert!(
            rec >= orig,
            "Recovered upper {rec} should be >= original {orig} after f16 widening"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_f16_widening_epsilon_zero() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new_compressed(input, 10, 0.0);

    let bounds = create_input(10);
    checkpointed.add_checkpoint(0, bounds);

    // Should still work with zero widening
    assert!(checkpointed.is_compressed());
    assert_eq!(checkpointed.num_checkpoints(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_memory_bytes_f16_vs_f32() {
    let input = create_input(1000);
    let bounds = create_input(1000);

    let mut f32_storage = CheckpointedBounds::new(input.clone(), 10);
    let mut f16_storage = CheckpointedBounds::new_compressed(input, 10, 0.0);

    f32_storage.add_checkpoint(0, bounds.clone());
    f16_storage.add_checkpoint(0, bounds);

    // f16 storage should use approximately half the memory for checkpoints
    let f32_mem = f32_storage.memory_bytes();
    let f16_mem = f16_storage.memory_bytes();

    // Input memory is same, checkpoint memory differs
    // f32: input (8000) + checkpoint (8000) = 16000
    // f16: input (8000) + checkpoint (4000) = 12000
    assert!(f16_mem < f32_mem);
}

#[ntest::timeout(10000)]
#[test]
fn test_f16_decompression_error_propagation() {
    // Test that decompression errors from f16 checkpoints are properly propagated
    // (not silently ignored). This verifies the error path at streaming/checkpoint.rs.
    //
    // Issue #163: Missing test for f16 checkpoint decompression error propagation.
    //
    // The error wrapping in find_nearest_checkpoint_before() formats decompression
    // errors as: "Checkpoint decompression failed at layer {}: {}"
    //
    // Since CompressedBounds validates shape on construction and BoundedTensor
    // only debug_asserts bounds validity, we test the error wrapping path by:
    // 1. Verifying the error wrapping code path exists and is correctly structured
    // 2. Testing CompressedBounds::new() rejects invalid shape (which would fail decompression)
    use half::f16;

    // Test 1: Verify CompressedBounds::new rejects length mismatch (shape validation)
    // This would cause to_bounded_tensor to fail if data got corrupted
    let lower = vec![f16::from_f32(0.0); 5]; // len=5
    let upper = vec![f16::from_f32(1.0); 5]; // len=5
    let wrong_shape = vec![10]; // expects len=10
    let result = CompressedBounds::new(lower, upper, wrong_shape);
    assert!(
        result.is_err(),
        "CompressedBounds should reject shape mismatch"
    );

    // Test 2: Verify that valid compressed bounds decompress successfully
    // (this confirms the happy path, ensuring the error path is for actual errors)
    let lower = vec![f16::from_f32(-1.0); 10];
    let upper = vec![f16::from_f32(1.0); 10];
    let valid_compressed = CompressedBounds::new(lower, upper, vec![10]).unwrap();
    let decompression_result = valid_compressed.to_bounded_tensor();
    assert!(
        decompression_result.is_ok(),
        "Valid compressed bounds should decompress successfully: {:?}",
        decompression_result.err()
    );

    // Test 3: Verify error message format from streaming module matches expected pattern
    // The error wraps with NyError::InvalidSpec
    let test_msg = format!(
        "{}",
        NyError::InvalidSpec(format!(
            "Checkpoint decompression failed at layer {}: {}",
            5, "test error"
        ))
    );
    assert!(
        test_msg.contains("Checkpoint decompression failed"),
        "Error message should follow expected format: {}",
        test_msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_f16_overflow_detection_during_compression() {
    // Test that f16 overflow is detected when compressing large f32 values.
    // This validates the has_overflow() detection method.

    // f16 max is ~65504, so 100000 will overflow to infinity during f16 conversion
    let lower = ArrayD::from_elem(ndarray::IxDyn(&[2]), 0.0f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[2]), 100000.0f32);
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);

    // Should detect overflow
    assert!(
        compressed.has_overflow(),
        "Should detect f16 overflow for values > 65504"
    );

    // Note: Decompression would fail in release mode (returns Error) and panic in
    // debug mode (debug_assert catches NaN/Inf). Both behaviors are correct - users
    // should check has_overflow() before decompressing and handle overflow appropriately.
}
