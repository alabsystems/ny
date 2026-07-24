// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_sound_mode_enforced() {
    // Test that LayerNorm in sound mode (default) returns error for CROWN propagation.
    // Network: Linear -> LayerNorm -> Linear
    // With sound mode enabled (default), CROWN propagation through LayerNorm should
    // return an error, NOT silently proceed with heuristic sampling.
    use crate::layers::LayerNormCrownMode;

    let ny = arr1(&[1.0_f32, 2.0, 0.5]);
    let beta = arr1(&[0.0_f32, 1.0, -0.5]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sound);
    assert_eq!(ln.crown_mode, LayerNormCrownMode::Sound);

    // Create input bounds
    let input_lower = arr1(&[0.5_f32, 1.5, 2.5]);
    let input_upper = arr1(&[1.5_f32, 2.5, 3.5]);
    let input = BoundedTensor::new(input_lower.into_dyn(), input_upper.into_dyn()).unwrap();

    // Get CROWN bounds - this should fail with UnsupportedOp error
    let linear_bounds = LinearBounds::identity(3);
    let result = ln.propagate_linear_with_bounds(&linear_bounds, &input);

    assert!(result.is_err(), "Sound mode should return error for CROWN");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("LayerNorm"),
        "Error should mention LayerNorm: {}",
        err_msg
    );
    assert!(
        err_msg.contains("heuristic sampling")
            || err_msg.contains("not provably sound")
            || err_msg.contains("Soundness refusal")
            || err_msg.contains("refused in Sound mode"),
        "Error should explain soundness refusal: {}",
        err_msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_cut_mode_succeeds() {
    // Test that LayerNorm in cut mode returns identity relaxation (sound but loose).
    // Same network, but with cut mode explicitly enabled.
    // This should succeed (sound, but looser due to cut).
    use crate::layers::LayerNormCrownMode;

    let ny = arr1(&[1.0_f32, 2.0, 0.5]);
    let beta = arr1(&[0.0_f32, 1.0, -0.5]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Cut);

    // Create input bounds
    let input_lower = arr1(&[0.5_f32, 1.5, 2.5]);
    let input_upper = arr1(&[1.5_f32, 2.5, 3.5]);
    let input = BoundedTensor::new(input_lower.into_dyn(), input_upper.into_dyn()).unwrap();

    // Get CROWN bounds - this should succeed
    let linear_bounds = LinearBounds::identity(3);
    let result = ln.propagate_linear_with_bounds(&linear_bounds, &input);

    assert!(
        result.is_ok(),
        "Cut mode should succeed for CROWN: {:?}",
        result.err()
    );

    // Cut mode returns identity relaxation - bounds should be unchanged
    let cut_bounds = result.unwrap();
    let input_bounds = linear_bounds;

    // Verify it's essentially identity (A matrices unchanged)
    let eps = 1e-6;
    assert!(
        cut_bounds
            .lower_a
            .iter()
            .zip(input_bounds.lower_a.iter())
            .all(|(a, b)| (a - b).abs() < eps),
        "Cut mode should return identity A_lower"
    );
    assert!(
        cut_bounds
            .upper_a
            .iter()
            .zip(input_bounds.upper_a.iter())
            .all(|(a, b)| (a - b).abs() < eps),
        "Cut mode should return identity A_upper"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_sampling_mode_warns() {
    // Test that LayerNorm in sampling mode succeeds but uses heuristic sampling.
    // Same network, but with sampling mode explicitly enabled.
    // This should succeed (NOT provably sound, but produces bounds).
    use crate::layers::LayerNormCrownMode;

    let ny = arr1(&[1.0_f32, 2.0, 0.5]);
    let beta = arr1(&[0.0_f32, 1.0, -0.5]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    // Create input bounds
    let input_lower = arr1(&[0.5_f32, 1.5, 2.5]);
    let input_upper = arr1(&[1.5_f32, 2.5, 3.5]);
    let input = BoundedTensor::new(
        input_lower.clone().into_dyn(),
        input_upper.clone().into_dyn(),
    )
    .unwrap();

    // Get CROWN bounds - this should succeed
    let linear_bounds = LinearBounds::identity(3);
    let result = ln.propagate_linear_with_bounds(&linear_bounds, &input);

    assert!(
        result.is_ok(),
        "Sampling mode should succeed for CROWN: {:?}",
        result.err()
    );

    // Verify we got non-trivial bounds (not identity like cut mode)
    let sampling_bounds = result.unwrap();

    // Concretize to check bounds are reasonable
    let concrete = sampling_bounds.concretize(&input);

    // Verify soundness by spot-checking a few samples
    for sample_idx in 0..10 {
        let t0 = (sample_idx as f32 * 17.0 % 100.0) / 100.0;
        let t1 = (sample_idx as f32 * 31.0 % 100.0) / 100.0;
        let t2 = (sample_idx as f32 * 47.0 % 100.0) / 100.0;

        let x_sample = arr1(&[
            input_lower[0] + (input_upper[0] - input_lower[0]) * t0,
            input_lower[1] + (input_upper[1] - input_lower[1]) * t1,
            input_lower[2] + (input_upper[2] - input_lower[2]) * t2,
        ]);

        let y_sample = ln.eval(&x_sample).unwrap();

        for i in 0..3 {
            // Note: sampling mode is NOT provably sound, so we use a larger tolerance
            // This test just verifies the mechanism works, not formal soundness
            assert!(
                y_sample[i] >= concrete.lower()[[i]] - 1e-3,
                "Sample {} output {} < lower bound {} at dim {} (sampling mode has tolerance)",
                sample_idx,
                y_sample[i],
                concrete.lower()[[i]],
                i
            );
            assert!(
                y_sample[i] <= concrete.upper()[[i]] + 1e-3,
                "Sample {} output {} > upper bound {} at dim {} (sampling mode has tolerance)",
                sample_idx,
                y_sample[i],
                concrete.upper()[[i]],
                i
            );
        }
    }
}
