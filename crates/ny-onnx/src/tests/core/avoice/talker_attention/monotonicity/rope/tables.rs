// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Phase 4: Real RoPE frequency tables (#3497)
//
// The tests above use identity RoPE (cos=1, sin=0), which verifies
// content-similarity-only monotonicity. Phase 4 uses precomputed Qwen3-TTS
// RoPE frequency tables with the current avoice `rope_theta=1_000_000`,
// runtime `head_dim=128`, and exported cos/sin width `rope_dim=64` to verify
// monotonicity with actual positional encoding.
//
// Reference: designs/2026-03-11-issue-3497-centroid-monotonicity-verification-path.md §Phase 4
// ---------------------------------------------------------------------------

fn assert_rope_tables_finite_and_bounded(cos: &ArrayD<f32>, sin: &ArrayD<f32>) {
    let seq_len = cos.shape()[0];
    let rope_dim = cos.shape()[1];
    for pos in 0..seq_len {
        for d in 0..rope_dim {
            let c = cos[[pos, d]];
            let s = sin[[pos, d]];
            assert!(c.is_finite(), "cos[{pos}, {d}] not finite: {c}");
            assert!(s.is_finite(), "sin[{pos}, {d}] not finite: {s}");
            assert!(c.abs() <= 1.0 + 1e-6, "cos[{pos}, {d}] out of range: {c}");
            assert!(s.abs() <= 1.0 + 1e-6, "sin[{pos}, {d}] out of range: {s}");
        }
    }
}

fn assert_rope_tables_repeated_layout(cos: &ArrayD<f32>, sin: &ArrayD<f32>) {
    let seq_len = cos.shape()[0];
    let num_pairs = cos.shape()[1] / 2;
    for pos in 0..seq_len {
        for i in 0..num_pairs {
            assert_eq!(
                cos[[pos, i]],
                cos[[pos, i + num_pairs]],
                "cos repeated layout violated at pos={pos}, pair={i}"
            );
            assert_eq!(
                sin[[pos, i]],
                sin[[pos, i + num_pairs]],
                "sin repeated layout violated at pos={pos}, pair={i}"
            );
        }
    }
}

/// Sanity check: RoPE tables have expected mathematical properties.
#[test]
fn test_qwen3_rope_table_sanity_3497() {
    let (cos_table, sin_table) = compute_qwen3_rope_tables(
        TALKER_ATTENTION_SEQ_LEN,
        TALKER_ATTENTION_ROPE_DIM,
        QWEN3_TTS_ROPE_BASE,
    );

    assert_eq!(
        cos_table.shape(),
        &[TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_ROPE_DIM]
    );
    assert_eq!(
        sin_table.shape(),
        &[TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_ROPE_DIM]
    );

    // Position 0: angle = 0 for all frequencies -> cos=1, sin=0.
    for d in 0..TALKER_ATTENTION_ROPE_DIM {
        let cos_val = cos_table[[0, d]];
        let sin_val = sin_table[[0, d]];
        assert!(
            (cos_val - 1.0).abs() < 1e-6,
            "cos[0, {d}] should be 1.0 at position 0, got {cos_val}"
        );
        assert!(
            sin_val.abs() < 1e-6,
            "sin[0, {d}] should be 0.0 at position 0, got {sin_val}"
        );
    }

    assert_rope_tables_finite_and_bounded(&cos_table, &sin_table);
    assert_rope_tables_repeated_layout(&cos_table, &sin_table);

    // Verify frequency variation: position 1 should differ from position 0.
    let cos_diff: f32 = (0..TALKER_ATTENTION_ROPE_DIM)
        .map(|d| (cos_table[[1, d]] - cos_table[[0, d]]).abs())
        .sum();
    let sin_diff: f32 = (0..TALKER_ATTENTION_ROPE_DIM)
        .map(|d| (sin_table[[1, d]] - sin_table[[0, d]]).abs())
        .sum();
    assert!(
        cos_diff > 1e-4 || sin_diff > 1e-4,
        "RoPE tables should vary across positions: cos_diff={cos_diff}, sin_diff={sin_diff}"
    );
}

#[test]
fn test_qwen3_rope_table_matches_current_avoice_theta_4180() {
    let (cos_table, sin_table) = compute_qwen3_rope_tables(
        TALKER_ATTENTION_SEQ_LEN,
        TALKER_ATTENTION_ROPE_DIM,
        QWEN3_TTS_ROPE_BASE,
    );

    // These literals pin the current avoice rope_theta=1_000_000.0 at
    // position 1 for representative frequency pairs. They intentionally
    // distinguish the live contract from the stale 10000.0 fallback.
    let expected_entries = [
        (1usize, 0.796_457_9f32, 0.604_694f32),
        (2usize, 0.912_395_83f32, 0.409_308_9f32),
        (15usize, 0.999_998_8f32, 0.001_539_925_9f32),
    ];

    for (pair_idx, expected_cos, expected_sin) in expected_entries {
        let repeated_idx = pair_idx + (TALKER_ATTENTION_ROPE_DIM / 2);
        for dim_idx in [pair_idx, repeated_idx] {
            let actual_cos = cos_table[[1, dim_idx]];
            let actual_sin = sin_table[[1, dim_idx]];
            assert!(
                (actual_cos - expected_cos).abs() < 1e-6,
                "cos[1, {dim_idx}] should pin rope_theta=1_000_000.0: expected {expected_cos}, got {actual_cos}"
            );
            assert!(
                (actual_sin - expected_sin).abs() < 1e-6,
                "sin[1, {dim_idx}] should pin rope_theta=1_000_000.0: expected {expected_sin}, got {actual_sin}"
            );
        }
    }
}
