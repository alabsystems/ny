// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `CompressedBounds` — compressed f16 bound storage.

use super::*;
use approx::assert_relative_eq;
use ndarray::{arr1, arr2};
use serde::Serialize;

#[test]
fn test_compress_decompress_basic() {
    let lower = arr1(&[-1.0f32, 0.0, 0.5]).into_dyn();
    let upper = arr1(&[1.0f32, 0.5, 1.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    let restored = compressed.to_bounded_tensor().unwrap();

    assert_eq!(restored.shape(), bounds.shape());

    // f16 precision is about 3 decimal places, so tolerance is 1e-3
    for (orig, rest) in bounds.lower().iter().zip(restored.lower().iter()) {
        assert_relative_eq!(orig, rest, max_relative = 1e-3);
    }
    for (orig, rest) in bounds.upper().iter().zip(restored.upper().iter()) {
        assert_relative_eq!(orig, rest, max_relative = 1e-3);
    }
}

#[test]
fn test_compress_2d_tensor() {
    let lower = arr2(&[[-1.0f32, -0.5], [0.0, 0.25]]).into_dyn();
    let upper = arr2(&[[1.0f32, 0.5], [0.5, 1.0]]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    assert_eq!(compressed.shape(), &[2, 2]);
    assert_eq!(compressed.len(), 4);

    let restored = compressed.to_bounded_tensor().unwrap();
    assert_eq!(restored.shape(), &[2, 2]);
}

#[test]
fn test_memory_savings() {
    let n = 10000;
    let lower = ArrayD::from_elem(IxDyn(&[n]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n]), 1.0f32);
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    let (compressed_bytes, f32_bytes, ratio) = compressed.compression_stats();

    // Should be approximately 50% compression
    assert!(ratio < 0.6, "Expected ~50% compression, got {}", ratio);
    assert!(ratio > 0.4, "Compression too aggressive: {}", ratio);
    assert!(
        compressed_bytes < f32_bytes,
        "Compressed size ({compressed_bytes}) should be smaller than f32 size ({f32_bytes})"
    );
}

#[test]
fn test_precision_loss() {
    // Test with values that have more precision than f16 can represent
    let lower = arr1(&[1.234567f32, -0.00012345, 100.123]).into_dyn();
    let upper = arr1(&[1.234568f32, -0.00012344, 100.124]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    let (max_lower_err, max_upper_err) = CompressedBounds::max_precision_loss(&bounds, &compressed);

    // f16 has ~3 decimal digits, so error should be < 1% of value
    // For small values like 1.234567, error should be < 0.01
    assert!(
        max_lower_err < 0.1,
        "Excessive lower precision loss: {}",
        max_lower_err
    );
    assert!(
        max_upper_err < 0.1,
        "Excessive upper precision loss: {}",
        max_upper_err
    );
}

#[test]
fn test_overflow_detection() {
    // f16 max is ~65504, so 100000 should overflow
    let lower = arr1(&[1.0f32, -100000.0]).into_dyn();
    let upper = arr1(&[2.0f32, 100000.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    assert!(compressed.has_overflow(), "Should detect overflow");
}

#[test]
fn test_no_overflow_in_normal_range() {
    let lower = arr1(&[-1000.0f32, -500.0, 0.0]).into_dyn();
    let upper = arr1(&[1000.0f32, 500.0, 100.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    assert!(
        !compressed.has_overflow(),
        "Should not overflow in normal range"
    );
}

#[test]
fn test_new_rejects_mismatched_lengths() {
    // Test that CompressedBounds::new rejects mismatched lengths
    let lower = vec![f16::from_f32(0.0), f16::from_f32(1.0)]; // len=2
    let upper = vec![f16::from_f32(1.0)]; // len=1
    let shape = vec![2];

    let result = CompressedBounds::new(lower, upper, shape);
    assert!(result.is_err(), "Should reject mismatched lengths");
}

#[test]
fn test_new_rejects_shape_mismatch() {
    // Test that CompressedBounds::new rejects shape/data mismatch
    let lower = vec![f16::from_f32(0.0), f16::from_f32(1.0)]; // len=2
    let upper = vec![f16::from_f32(1.0), f16::from_f32(2.0)]; // len=2
    let shape = vec![3]; // expects len=3

    let result = CompressedBounds::new(lower, upper, shape);
    assert!(result.is_err(), "Should reject shape/data mismatch");
}

#[test]
fn test_serde_round_trip_preserves_valid_compressed_bounds() {
    let compressed = CompressedBounds::new(
        vec![f16::from_f32(-1.0), f16::from_f32(0.0)],
        vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        vec![2],
    )
    .expect("valid compressed bounds");

    let encoded = serde_json::to_string(&compressed).expect("serialize");
    let decoded: CompressedBounds = serde_json::from_str(&encoded).expect("deserialize");

    assert_eq!(decoded.lower_raw(), compressed.lower_raw());
    assert_eq!(decoded.upper_raw(), compressed.upper_raw());
    assert_eq!(decoded.shape(), compressed.shape());
}

#[test]
fn test_serde_rejects_shape_data_mismatch() {
    #[derive(Serialize)]
    struct RawCompressedBounds {
        lower: Vec<f16>,
        upper: Vec<f16>,
        shape: Vec<usize>,
    }

    let raw = RawCompressedBounds {
        lower: vec![f16::from_f32(0.0)],
        upper: vec![f16::from_f32(1.0)],
        shape: vec![2],
    };
    let encoded = serde_json::to_string(&raw).expect("serialize malformed fixture");

    assert!(serde_json::from_str::<CompressedBounds>(&encoded).is_err());
}

#[test]
fn test_new_and_serde_reject_invalid_interval_endpoints() {
    assert!(
        CompressedBounds::new(vec![f16::NAN], vec![f16::ONE], vec![1]).is_err(),
        "NaN lower endpoints must be rejected"
    );
    assert!(
        CompressedBounds::new(vec![f16::ZERO], vec![f16::NAN], vec![1]).is_err(),
        "NaN upper endpoints must be rejected"
    );
    assert!(
        CompressedBounds::new(vec![f16::ONE], vec![f16::ZERO], vec![1]).is_err(),
        "ordinary inverted intervals must be rejected"
    );
    CompressedBounds::new(vec![f16::INFINITY], vec![f16::NEG_INFINITY], vec![1])
        .expect("canonical infeasible sentinel remains representable");

    #[derive(Serialize)]
    struct RawCompressedBounds {
        lower: Vec<f16>,
        upper: Vec<f16>,
        shape: Vec<usize>,
    }
    for raw in [
        RawCompressedBounds {
            lower: vec![f16::NAN],
            upper: vec![f16::ONE],
            shape: vec![1],
        },
        RawCompressedBounds {
            lower: vec![f16::ONE],
            upper: vec![f16::ZERO],
            shape: vec![1],
        },
    ] {
        let encoded = serde_json::to_string(&raw).expect("serialize malformed fixture");
        assert!(
            serde_json::from_str::<CompressedBounds>(&encoded).is_err(),
            "deserialization must apply constructor invariants"
        );
    }
}

#[test]
fn test_widen_for_soundness() {
    let lower = arr1(&[-1.0f32, 0.0, 0.5]).into_dyn();
    let upper = arr1(&[1.0f32, 0.5, 1.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let mut compressed = CompressedBounds::from_bounded_tensor(&bounds);
    compressed.widen_for_soundness(0.001); // 0.1% widening

    let restored = compressed.to_bounded_tensor().unwrap();

    // After widening, restored lower should be <= original lower
    // and restored upper should be >= original upper
    for (orig, rest) in bounds.lower().iter().zip(restored.lower().iter()) {
        assert!(
            rest <= orig,
            "Lower bound {} not conservative (original {})",
            rest,
            orig
        );
    }
    for (orig, rest) in bounds.upper().iter().zip(restored.upper().iter()) {
        assert!(
            rest >= orig,
            "Upper bound {} not conservative (original {})",
            rest,
            orig
        );
    }
}

#[test]
fn test_compression_stats() {
    let n = 1000;
    let lower = ArrayD::from_elem(IxDyn(&[n]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n]), 1.0f32);
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    let stats = CompressionStats::from_compression(&bounds, &compressed);

    assert!(
        stats.memory_savings_percent() > 40.0,
        "Expected memory savings above 40%, got {}",
        stats.memory_savings_percent()
    );
    assert!(
        stats.memory_savings_percent() < 60.0,
        "Expected memory savings below 60%, got {}",
        stats.memory_savings_percent()
    );
    assert!(
        !stats.has_overflow,
        "Compression should not overflow for [-1, 1] range"
    );
    assert!(
        stats.max_lower_error < 0.01,
        "Lower error {} exceeds 0.01 threshold",
        stats.max_lower_error
    );
    assert!(
        stats.max_upper_error < 0.01,
        "Upper error {} exceeds 0.01 threshold",
        stats.max_upper_error
    );
}

#[test]
fn test_empty_tensor() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    assert!(
        compressed.is_empty(),
        "Compressed empty tensor should report empty"
    );
    assert_eq!(compressed.len(), 0);

    let restored = compressed.to_bounded_tensor().unwrap();
    assert!(
        restored.is_empty(),
        "Restored empty tensor should report empty"
    );
}

#[test]
fn test_large_tensor() {
    // Test with realistic transformer-sized tensor
    let n = 768 * 512; // hidden_dim * seq_len
    let lower = ArrayD::from_elem(IxDyn(&[1, 512, 768]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 512, 768]), 10.0f32);
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    assert_eq!(compressed.len(), n);

    let (compressed_bytes, f32_bytes, _) = compressed.compression_stats();

    // Verify significant memory savings
    assert!(
        compressed_bytes < f32_bytes * 6 / 10,
        "Expected >40% savings: {} vs {}",
        compressed_bytes,
        f32_bytes
    );
}

// ========================================
// Mutation-killing tests for CompressedBounds
// ========================================

#[test]
fn test_is_empty_exact() {
    // Non-empty should return false, not true
    let lower = arr1(&[1.0f32, 2.0]).into_dyn();
    let upper = arr1(&[3.0f32, 4.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();
    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    assert!(
        !compressed.is_empty(),
        "Non-empty compressed bounds should not report empty"
    );

    // Empty should return true
    let empty = CompressedBounds::new(vec![], vec![], vec![0]).unwrap();
    assert!(
        empty.is_empty(),
        "Explicit empty compressed bounds should be empty"
    );
}

#[test]
fn test_memory_bytes_exact_computation() {
    // memory_bytes = lower.len() * 2 * 2 + shape.len() * 8
    // For 3 elements with 1D shape: 3*4 + 1*8 = 20
    let lower = arr1(&[1.0f32, 2.0, 3.0]).into_dyn();
    let upper = arr1(&[4.0f32, 5.0, 6.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();
    let compressed = CompressedBounds::from_bounded_tensor(&bounds);

    // 3 elements * 2 bytes * 2 (lower + upper) + 1 dimension * 8 bytes = 20
    assert_eq!(compressed.memory_bytes(), 3 * 2 * 2 + 8);

    // For 2D: 2x3=6 elements with 2D shape: 6*4 + 2*8 = 40
    let lower2d = arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
    let upper2d = arr2(&[[7.0f32, 8.0, 9.0], [10.0, 11.0, 12.0]]).into_dyn();
    let bounds2d = BoundedTensor::new(lower2d, upper2d).unwrap();
    let compressed2d = CompressedBounds::from_bounded_tensor(&bounds2d);
    assert_eq!(compressed2d.memory_bytes(), 6 * 2 * 2 + 2 * 8);
}

#[test]
fn test_lower_raw_upper_raw_nonempty() {
    // These functions should return non-empty slices when data exists
    let lower = arr1(&[1.0f32, 2.0, 3.0]).into_dyn();
    let upper = arr1(&[4.0f32, 5.0, 6.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();
    let compressed = CompressedBounds::from_bounded_tensor(&bounds);

    let lower_raw = compressed.lower_raw();
    let upper_raw = compressed.upper_raw();

    // Should have 3 elements, not 0 or 1
    assert_eq!(lower_raw.len(), 3);
    assert_eq!(upper_raw.len(), 3);

    // Values should be approximately correct
    assert!(
        (lower_raw[0].to_f32() - 1.0).abs() < 0.01,
        "lower_raw[0] should be approximately 1.0, got {}",
        lower_raw[0].to_f32()
    );
    assert!(
        (lower_raw[1].to_f32() - 2.0).abs() < 0.01,
        "lower_raw[1] should be approximately 2.0, got {}",
        lower_raw[1].to_f32()
    );
    assert!(
        (upper_raw[2].to_f32() - 6.0).abs() < 0.01,
        "upper_raw[2] should be approximately 6.0, got {}",
        upper_raw[2].to_f32()
    );
}

#[test]
fn test_widen_for_soundness_actually_widens() {
    // Verify widening actually changes values (not a no-op)
    let lower = arr1(&[10.0f32, -10.0, 0.5]).into_dyn();
    let upper = arr1(&[20.0f32, 10.0, 1.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let mut compressed = CompressedBounds::from_bounded_tensor(&bounds);
    let lower_before: Vec<f32> = compressed.lower.iter().map(|v| v.to_f32()).collect();
    let upper_before: Vec<f32> = compressed.upper.iter().map(|v| v.to_f32()).collect();

    compressed.widen_for_soundness(0.01); // 1% widening

    let lower_after: Vec<f32> = compressed.lower.iter().map(|v| v.to_f32()).collect();
    let upper_after: Vec<f32> = compressed.upper.iter().map(|v| v.to_f32()).collect();

    // Lower should decrease (become more negative/smaller)
    for i in 0..3 {
        assert!(
            lower_after[i] < lower_before[i],
            "Lower bound {} should decrease: {} -> {}",
            i,
            lower_before[i],
            lower_after[i]
        );
    }

    // Upper should increase
    for i in 0..3 {
        assert!(
            upper_after[i] > upper_before[i],
            "Upper bound {} should increase: {} -> {}",
            i,
            upper_before[i],
            upper_after[i]
        );
    }
}

#[test]
fn test_has_overflow_distinguish_lower_upper() {
    // Test that we can detect overflow in lower bounds only
    let lower = arr1(&[f32::NEG_INFINITY, 1.0]).into_dyn();
    let upper = arr1(&[1.0f32, 2.0]).into_dyn();
    let bounds_lower_inf = BoundedTensor::new_unchecked(lower, upper).unwrap();
    let compressed_lower = CompressedBounds::from_bounded_tensor(&bounds_lower_inf);
    assert!(
        compressed_lower.has_overflow(),
        "Should detect lower bound overflow"
    );

    // Test that we can detect overflow in upper bounds only
    let lower2 = arr1(&[0.0f32, 1.0]).into_dyn();
    let upper2 = arr1(&[1.0f32, f32::INFINITY]).into_dyn();
    let bounds_upper_inf = BoundedTensor::new_unchecked(lower2, upper2).unwrap();
    let compressed_upper = CompressedBounds::from_bounded_tensor(&bounds_upper_inf);
    assert!(
        compressed_upper.has_overflow(),
        "Should detect upper bound overflow"
    );

    // Test normal values have no overflow
    let lower3 = arr1(&[0.0f32, 1.0]).into_dyn();
    let upper3 = arr1(&[1.0f32, 2.0]).into_dyn();
    let bounds_normal = BoundedTensor::new(lower3, upper3).unwrap();
    let compressed_normal = CompressedBounds::from_bounded_tensor(&bounds_normal);
    assert!(
        !compressed_normal.has_overflow(),
        "Normal values should not overflow"
    );
}

#[test]
fn test_max_precision_loss_returns_correct_tuple() {
    // Verify both tuple elements are computed correctly, not just (0,0) or (-1,-1)
    // Use values that are not exactly representable in f16 so the error is non-zero.
    let lower = arr1(&[0.1f32, 0.2, 0.3]).into_dyn();
    let upper = arr1(&[0.15f32, 0.25, 0.35]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();
    let compressed = CompressedBounds::from_bounded_tensor(&bounds);

    let (lower_err, upper_err) = CompressedBounds::max_precision_loss(&bounds, &compressed);

    // Errors should be >= 0 (absolute values)
    assert!(
        lower_err >= 0.0,
        "Lower precision loss should be non-negative, got {lower_err}"
    );
    assert!(
        upper_err >= 0.0,
        "Upper precision loss should be non-negative, got {upper_err}"
    );

    // Since at least one value is not representable in f16, max error should be > 0.
    assert!(lower_err > 0.0, "Expected non-zero lower error");
    assert!(upper_err > 0.0, "Expected non-zero upper error");

    // For f16 conversion of small values, error should still be small.
    assert!(lower_err < 1.0, "Lower error too large: {}", lower_err);
    assert!(upper_err < 1.0, "Upper error too large: {}", upper_err);
}

#[test]
fn test_widen_for_soundness_expected_magnitude() {
    // This test targets common mutation survivors in widen_for_soundness:
    // - abs sign handling (negative values must widen proportionally)
    // - multiplication vs addition/division in delta computation
    // - min_delta fallback must not override proportional widening
    let lower = vec![
        f16::from_f32(-10.0), // large magnitude negative
        f16::from_f32(-6.0),  // negative
        f16::from_f32(0.0),   // triggers min_delta
    ];
    let upper = vec![
        f16::from_f32(20.0), // large magnitude positive
        f16::from_f32(-5.0), // negative upper bound (must still widen up)
        f16::from_f32(0.0),  // triggers min_delta
    ];

    let mut compressed = CompressedBounds::new(lower.clone(), upper.clone(), vec![3]).unwrap();
    compressed.widen_for_soundness(0.01);

    let lower_after: Vec<f32> = compressed.lower_raw().iter().map(|v| v.to_f32()).collect();
    let upper_after: Vec<f32> = compressed.upper_raw().iter().map(|v| v.to_f32()).collect();

    let lower_before: Vec<f32> = lower.iter().map(|v| v.to_f32()).collect();
    let upper_before: Vec<f32> = upper.iter().map(|v| v.to_f32()).collect();

    let lower_delta_0 = lower_before[0] - lower_after[0];
    let upper_delta_0 = upper_after[0] - upper_before[0];
    assert!(
        (0.08..0.12).contains(&lower_delta_0),
        "Expected ~0.1 lower widening for -10.0, got {}",
        lower_delta_0
    );
    assert!(
        (0.18..0.22).contains(&upper_delta_0),
        "Expected ~0.2 upper widening for 20.0, got {}",
        upper_delta_0
    );

    let lower_delta_1 = lower_before[1] - lower_after[1];
    let upper_delta_1 = upper_after[1] - upper_before[1];
    assert!(
        (0.04..0.08).contains(&lower_delta_1),
        "Expected ~0.06 lower widening for -6.0, got {}",
        lower_delta_1
    );
    assert!(
        (0.03..0.07).contains(&upper_delta_1),
        "Expected ~0.05 upper widening for -5.0, got {}",
        upper_delta_1
    );

    let min_delta_f32 = f16::from_f32(1e-6).to_f32();
    let lower_delta_2 = lower_before[2] - lower_after[2];
    let upper_delta_2 = upper_after[2] - upper_before[2];
    assert!(
        (lower_delta_2 - min_delta_f32).abs() <= min_delta_f32,
        "Expected min_delta widening for 0.0 lower, got {} (min_delta {})",
        lower_delta_2,
        min_delta_f32
    );
    assert!(
        (upper_delta_2 - min_delta_f32).abs() <= min_delta_f32,
        "Expected min_delta widening for 0.0 upper, got {} (min_delta {})",
        upper_delta_2,
        min_delta_f32
    );
}

#[test]
fn test_to_bounded_tensor_error_on_corrupt_shape() {
    // Test that to_bounded_tensor() returns error when shape doesn't match data
    // This exercises the error path at lines 108-111 and 114-116.
    // Part of #163: verify decompression error propagation
    let lower = vec![f16::from_f32(0.0), f16::from_f32(1.0)]; // len=2
    let upper = vec![f16::from_f32(1.0), f16::from_f32(2.0)]; // len=2
    let shape = vec![3]; // expects len=3, but we only have 2 elements

    // Use new_unchecked to bypass validation and create corrupt state
    let corrupt = CompressedBounds::new_unchecked(lower, upper, shape);

    // to_bounded_tensor() should fail with InvalidSpec error
    let result = corrupt.to_bounded_tensor();
    assert!(result.is_err(), "Should return error on shape mismatch");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("Failed to reshape"),
        "Error message should mention reshape failure: {}",
        err
    );
}

/// Directed rounding: from_bounded_tensor must round lower bounds DOWN
/// and upper bounds UP, even when the f32 value is not exactly representable
/// in f16. (#3039)
#[test]
fn test_from_bounded_tensor_directed_rounding() {
    // 1000.123 is not exactly representable in f16.
    // f16::from_f32(1000.123) rounds to nearest (1000.0 in f16).
    // Directed rounding should give:
    //   lower: round DOWN → value <= 1000.123
    //   upper: round UP   → value >= 1000.123
    let val = 1000.123f32;
    let lower = arr1(&[-val]).into_dyn();
    let upper = arr1(&[val]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    let restored = compressed.to_bounded_tensor().unwrap();

    // Lower bound must not be tighter (must be <= original)
    assert!(
        restored.lower()[[0]] <= -val,
        "Lower bound tightened: restored {} > original {}",
        restored.lower()[[0]],
        -val
    );
    // Upper bound must not be tighter (must be >= original)
    assert!(
        restored.upper()[[0]] >= val,
        "Upper bound tightened: restored {} < original {}",
        restored.upper()[[0]],
        val
    );
}

/// f16 overflow: values > 65504 become ±Inf in f16. to_bounded_tensor() must
/// succeed (not reject Inf) because directed rounding ensures only sound
/// infinities appear: lower bounds get -Inf (widened), upper bounds get +Inf
/// (widened). (#2358)
#[test]
fn test_to_bounded_tensor_handles_f16_overflow_2358() {
    // Create bounds that exceed f16::MAX (~65504) — these overflow to ±Inf in f16.
    let lower = arr1(&[-100_000.0f32, 1.0, 70_000.0]).into_dyn();
    let upper = arr1(&[100_000.0f32, 2.0, 80_000.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    assert!(compressed.has_overflow(), "Should detect f16 overflow");

    // CRITICAL: decompression must NOT return Err. Before #2358 fix,
    // this would fail with NumericalInstability because BoundedTensor::new
    // rejects Inf values.
    let restored = compressed.to_bounded_tensor();
    assert!(
        restored.is_ok(),
        "to_bounded_tensor should handle f16 overflow: {:?}",
        restored.err()
    );

    let restored = restored.unwrap();

    // Soundness: restored bounds must be a superset of originals.
    // Lower bounds: -100000 → compressed to -Inf → decompressed to -Inf (widened, sound)
    assert!(
        restored.lower()[[0]] <= -100_000.0,
        "Lower[0] should be <= -100000: got {}",
        restored.lower()[[0]]
    );
    // Upper bounds: 100000 → compressed to +Inf → decompressed to +Inf (widened, sound)
    assert!(
        restored.upper()[[0]] >= 100_000.0,
        "Upper[0] should be >= 100000: got {}",
        restored.upper()[[0]]
    );

    // Normal-range values should still roundtrip correctly
    assert!(
        (restored.lower()[[1]] - 1.0).abs() < 0.01,
        "Lower[1] should be ~1.0: got {}",
        restored.lower()[[1]]
    );
    assert!(
        (restored.upper()[[1]] - 2.0).abs() < 0.01,
        "Upper[1] should be ~2.0: got {}",
        restored.upper()[[1]]
    );

    // Positive lower bound > 65504: from_f32_down rounds to f16::MAX (65504), not +Inf
    // This tests that directed rounding correctly avoids unsound +Inf lower bounds.
    let lower_2 = restored.lower()[[2]];
    assert!(
        lower_2.is_finite(),
        "Lower[2] should be finite (f16::MAX via from_f32_down), not Inf: got {}",
        lower_2
    );
    assert!(
        lower_2 <= 70_000.0,
        "Lower[2] should be <= 70000: got {}",
        lower_2
    );
}

/// Proptest: from_bounded_tensor → to_bounded_tensor round-trip always produces
/// bounds ⊇ originals (soundness property). (#3039)
///
/// For every element: restored_lower <= original_lower AND restored_upper >= original_upper.
/// This is guaranteed by directed rounding (from_f32_down for lower, from_f32_up for upper).
mod proptest_roundtrip {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]
        #[test]
        fn compressed_roundtrip_sound(
            // Values in f16 range to avoid Inf (f16 max ≈ 65504)
            lo in -65000.0f32..0.0,
            spread in 0.0f32..65000.0,
        ) {
            let hi = lo + spread;

            let lower = arr1(&[lo]).into_dyn();
            let upper = arr1(&[hi]).into_dyn();
            let bounds = BoundedTensor::new(lower, upper).unwrap();

            let compressed = CompressedBounds::from_bounded_tensor(&bounds);
            let restored = compressed.to_bounded_tensor().unwrap();

            // Soundness: restored bounds must be a superset of originals.
            // restored_lower <= original_lower
            for (orig, rest) in bounds.lower().iter().zip(restored.lower().iter()) {
                prop_assert!(
                    rest <= orig || orig.is_nan(),
                    "Soundness violation: restored lower {} > original lower {}",
                    rest, orig
                );
            }
            // restored_upper >= original_upper
            for (orig, rest) in bounds.upper().iter().zip(restored.upper().iter()) {
                prop_assert!(
                    rest >= orig || orig.is_nan(),
                    "Soundness violation: restored upper {} < original upper {}",
                    rest, orig
                );
            }
        }

        /// Test directed rounding at values that are NOT exactly representable in f16.
        /// Uses values with more precision than f16's ~3 decimal digits.
        #[test]
        fn compressed_roundtrip_nonexact_values(
            // Use small fractional values where f16 rounding is non-trivial
            lo_frac in 0.001f32..1.0,
            hi_frac in 0.001f32..1.0,
        ) {
            // Construct bounds where lo < hi and neither is exactly representable
            let lo = -lo_frac * 1000.123;
            let hi = hi_frac * 1000.456;

            let lower = arr1(&[lo]).into_dyn();
            let upper = arr1(&[hi]).into_dyn();
            let bounds = BoundedTensor::new(lower, upper).unwrap();

            let compressed = CompressedBounds::from_bounded_tensor(&bounds);
            let restored = compressed.to_bounded_tensor().unwrap();

            for (orig, rest) in bounds.lower().iter().zip(restored.lower().iter()) {
                prop_assert!(
                    rest <= orig,
                    "Lower bound tightened: restored {} > original {}",
                    rest, orig
                );
            }
            for (orig, rest) in bounds.upper().iter().zip(restored.upper().iter()) {
                prop_assert!(
                    rest >= orig,
                    "Upper bound tightened: restored {} < original {}",
                    rest, orig
                );
            }
        }
    }
}

/// max_precision_loss must propagate NaN errors, not absorb them.
/// f32::max absorbs NaN per IEEE 754 maxNum semantics — nan_propagating_max
/// correctly returns NaN when either argument is NaN. (#3291 F2)
#[test]
fn test_max_precision_loss_nan_propagation() {
    // Create a BoundedTensor with a NaN in lower bounds
    let lower = arr1(&[1.0f32, f32::NAN, 3.0]).into_dyn();
    let upper = arr1(&[2.0f32, 4.0, 5.0]).into_dyn();
    // Use new_unchecked to bypass BoundedTensor validation
    let bounds = BoundedTensor::new_unchecked(lower, upper).unwrap();

    // Create compressed bounds with normal values (simulate corruption in original)
    let compressed = CompressedBounds::from_bounded_tensor(&bounds);
    let (max_lower_err, _max_upper_err) =
        CompressedBounds::max_precision_loss(&bounds, &compressed);

    // The NaN in lower[1] should produce a NaN error that propagates to the max.
    // Before fix: f32::max(0.0, NaN) == 0.0 (NaN absorbed, reports finite error)
    // After fix: nan_propagating_max(0.0, NaN) == NaN (corruption detected)
    assert!(
        max_lower_err.is_nan(),
        "max_precision_loss must propagate NaN from corrupted original bounds, got {}",
        max_lower_err
    );
}
