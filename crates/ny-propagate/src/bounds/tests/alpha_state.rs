// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for AlphaState.

use super::checked_bounds;
use crate::bounds::AlphaState;
use crate::BoundedTensor;
use ndarray::array;

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_all_positive() {
    // All neurons always positive: no unstable
    let bounds = vec![checked_bounds(
        array![1.0_f32, 2.0, 3.0].into_dyn(),
        array![4.0_f32, 5.0, 6.0].into_dyn(),
    )];
    let relu_indices = vec![0];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    assert_eq!(state.num_unstable(), 0);
    // All alphas should be 1.0 (identity for positive)
    assert_eq!(state.alphas[0].as_slice().unwrap(), &[1.0, 1.0, 1.0]);
    // Mask should be all false
    assert_eq!(
        state.unstable_mask[0].as_slice().unwrap(),
        &[false, false, false]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_all_negative() {
    // All neurons always negative: no unstable
    let bounds = vec![checked_bounds(
        array![-6.0_f32, -5.0, -4.0].into_dyn(),
        array![-3.0_f32, -2.0, -1.0].into_dyn(),
    )];
    let relu_indices = vec![0];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    assert_eq!(state.num_unstable(), 0);
    // All alphas should be 0.0 (zero for negative)
    assert_eq!(state.alphas[0].as_slice().unwrap(), &[0.0, 0.0, 0.0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_mixed_bounds() {
    // Mix of positive, negative, and crossing
    let bounds = vec![checked_bounds(
        array![1.0_f32, -5.0, -2.0].into_dyn(), // always pos, always neg, crossing
        array![3.0_f32, -1.0, 4.0].into_dyn(),
    )];
    let relu_indices = vec![0];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    assert_eq!(state.num_unstable(), 1); // Only neuron 2 is unstable

    // Neuron 0: always positive -> alpha=1, mask=false
    assert_eq!(state.alphas[0][0], 1.0);
    assert!(!state.unstable_mask[0][0]);

    // Neuron 1: always negative -> alpha=0, mask=false
    assert_eq!(state.alphas[0][1], 0.0);
    assert!(!state.unstable_mask[0][1]);

    // Neuron 2: crossing [-2, 4], u=4 > -l=2 -> alpha=1, mask=true
    assert_eq!(state.alphas[0][2], 1.0);
    assert!(state.unstable_mask[0][2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_equal_crossing_bounds() {
    // Crossing with equal positive/negative magnitude: u == -l -> alpha = 0
    let bounds = vec![checked_bounds(
        array![-2.0_f32].into_dyn(),
        array![2.0_f32].into_dyn(),
    )];
    let relu_indices = vec![0];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    assert_eq!(state.num_unstable(), 1);
    assert_eq!(state.alphas[0][0], 0.0);
    assert!(state.unstable_mask[0][0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_equal_crossing_bounds_multi() {
    // Multiple equal-magnitude crossings should all pick alpha = 0.
    let bounds = vec![checked_bounds(
        array![-3.0_f32, -1.5_f32].into_dyn(),
        array![3.0_f32, 1.5_f32].into_dyn(),
    )];
    let relu_indices = vec![0];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    assert_eq!(state.num_unstable(), 2);
    assert_eq!(state.alphas[0].len(), 2);
    assert_eq!(state.alphas[0].as_slice().unwrap(), &[0.0, 0.0]);
    assert_eq!(state.unstable_mask[0].as_slice().unwrap(), &[true, true]);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_crossing_more_negative() {
    // Crossing neuron with more negative area
    let bounds = vec![checked_bounds(
        array![-4.0_f32].into_dyn(), // crossing with u < -l
        array![1.0_f32].into_dyn(),
    )];
    let relu_indices = vec![0];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    // u=1 < -l=4 -> alpha=0 (adaptive heuristic)
    assert_eq!(state.alphas[0][0], 0.0);
    assert!(state.unstable_mask[0][0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_get_alpha() {
    let bounds = vec![checked_bounds(
        array![-1.0_f32, -1.0].into_dyn(),
        array![1.0_f32, 1.0].into_dyn(),
    )];
    let relu_indices = vec![0];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    assert!(state.alpha(0).is_some());
    assert!(state.alpha(1).is_none()); // Out of range
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_without_momentum() {
    // Use asymmetric bounds where u > -l, so alpha initializes to 1
    let bounds = vec![checked_bounds(
        array![-1.0_f32, -1.0].into_dyn(),
        array![2.0_f32, 2.0].into_dyn(), // u=2 > -l=1
    )];
    let relu_indices = vec![0];

    let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    // Both neurons unstable with alpha=1 (u > -l)
    // Verify initial state before update
    assert_eq!(state.alphas[0][0], 1.0);
    assert_eq!(state.alphas[0][1], 1.0);

    // Apply gradient descent (gradient points up, so alpha decreases)
    let gradient = array![0.5_f32, 0.5_f32];
    state.update(0, &gradient, 0.1, 0.0);

    // alpha -= lr * gradient = 1.0 - 0.1 * 0.5 = 0.95
    assert!((state.alphas[0][0] - 0.95).abs() < 1e-6);
    assert!((state.alphas[0][1] - 0.95).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_with_clamping() {
    // Use asymmetric bounds where u > -l, so alpha initializes to 1
    let bounds = vec![checked_bounds(
        array![-1.0_f32].into_dyn(),
        array![2.0_f32].into_dyn(), // u=2 > -l=1
    )];
    let relu_indices = vec![0];

    let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();
    // Alpha starts at 1.0 (since u > -l)
    assert_eq!(state.alphas[0][0], 1.0);

    // Large positive gradient should clamp alpha to 0
    let gradient = array![100.0_f32];
    state.update(0, &gradient, 1.0, 0.0);

    assert_eq!(state.alphas[0][0], 0.0); // Clamped to 0

    // Large negative gradient should clamp alpha to 1
    let gradient = array![-100.0_f32];
    state.update(0, &gradient, 1.0, 0.0);

    assert_eq!(state.alphas[0][0], 1.0); // Clamped to 1
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_skips_stable() {
    // Use asymmetric bounds for unstable neuron: u=2 > -l=1 so alpha=1
    let bounds = vec![checked_bounds(
        array![1.0_f32, -1.0].into_dyn(), // First stable (positive), second unstable
        array![2.0_f32, 2.0].into_dyn(),  // Second neuron: u=2 > -l=1
    )];
    let relu_indices = vec![0];

    let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    // First neuron is stable (alpha=1, mask=false)
    // Second neuron is unstable with alpha=1 (u=2 > -l=1)
    let initial_stable = state.alphas[0][0];
    assert_eq!(state.alphas[0][1], 1.0);

    // Update shouldn't change stable neuron
    let gradient = array![10.0_f32, 0.5_f32];
    state.update(0, &gradient, 0.1, 0.0);

    assert_eq!(state.alphas[0][0], initial_stable); // Unchanged
    assert!((state.alphas[0][1] - 0.95).abs() < 1e-6); // Updated
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_invalid_index() {
    let bounds = vec![checked_bounds(
        array![-1.0_f32].into_dyn(),
        array![1.0_f32].into_dyn(),
    )];
    let relu_indices = vec![0];

    let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();
    let alpha_before = state.alphas[0][0];

    // Invalid relu index - should silently return
    let gradient = array![1.0_f32];
    state.update(99, &gradient, 0.1, 0.0);

    // State unchanged
    assert_eq!(state.alphas[0][0], alpha_before);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_num_unstable_multiple_layers() {
    let bounds = vec![
        checked_bounds(
            array![-1.0_f32, 1.0, -1.0].into_dyn(), // 2 unstable
            array![1.0_f32, 2.0, 1.0].into_dyn(),
        ),
        checked_bounds(
            array![-1.0_f32, -1.0].into_dyn(), // 2 unstable
            array![1.0_f32, 1.0].into_dyn(),
        ),
    ];
    let relu_indices = vec![0, 1];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    assert_eq!(state.num_unstable(), 4); // 2 + 2
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_empty_layers() {
    let bounds: Vec<BoundedTensor> = vec![];
    let relu_indices: Vec<usize> = vec![];

    let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    assert_eq!(state.num_unstable(), 0);
    assert!(state.alphas.is_empty());
}

/// Regression test for #1937: gradient-length mismatch must not panic.
///
/// When AnalyticChain falls back to a zero gradient with wrong length,
/// the update methods must skip the update instead of indexing out-of-bounds.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_gradient_length_mismatch_no_panic_1937() {
    let bounds = vec![checked_bounds(
        array![-1.0_f32, -0.5, -2.0].into_dyn(), // 3 unstable neurons
        array![1.0_f32, 0.5, 2.0].into_dyn(),
    )];
    let relu_indices = vec![0];

    let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();
    assert_eq!(state.alphas[0].len(), 3);
    let alpha_before = state.alphas[0].clone();

    // Gradient with wrong length (1 vs 3) — this is the bug scenario from #1937.
    let wrong_gradient = array![0.5_f32];
    state.update(0, &wrong_gradient, 0.1, 0.0);

    // Alpha must be unchanged — the mismatch guard should skip the update.
    assert_eq!(state.alphas[0], alpha_before);
}

/// Regression test for #1937: Adam update with wrong gradient length.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_adam_gradient_length_mismatch_no_panic_1937() {
    let bounds = vec![checked_bounds(
        array![-1.0_f32, -0.5].into_dyn(), // 2 unstable neurons
        array![1.0_f32, 0.5].into_dyn(),
    )];
    let relu_indices = vec![0];

    let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();
    let alpha_before = state.alphas[0].clone();

    let wrong_gradient = array![0.5_f32]; // length 1 vs 2
    let params = crate::bounds::alpha::AdamParams::new(0.01, 1);
    state.update_adam(0, &wrong_gradient, &params);

    assert_eq!(state.alphas[0], alpha_before);
}

/// Regression test for #2025: NaN gradient must not permanently corrupt alpha/m/v state.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_adam_nan_gradient_sanitized_2025() {
    let bounds = vec![checked_bounds(
        array![-1.0_f32, -0.5].into_dyn(),
        array![1.0_f32, 0.5].into_dyn(),
    )];
    let relu_indices = vec![0];

    let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    // Inject NaN gradient — before fix, this would permanently corrupt alpha state
    let nan_gradient = array![f32::NAN, f32::NAN];
    let params = crate::bounds::alpha::AdamParams::new(0.01, 1);
    state.update_adam(0, &nan_gradient, &params);

    // After NaN sanitization: alpha should be 0.5, m and v should be reset to 0
    for &a in state.alphas[0].iter() {
        assert!(
            a.is_finite(),
            "alpha must be finite after NaN gradient, got {}",
            a
        );
    }
    for &mi in state.adam_m[0].iter() {
        assert!(
            mi.is_finite(),
            "adam_m must be finite after NaN gradient, got {}",
            mi
        );
    }
    for &vi in state.adam_v[0].iter() {
        assert!(
            vi.is_finite(),
            "adam_v must be finite after NaN gradient, got {}",
            vi
        );
    }

    // Subsequent update with valid gradient should work normally
    let valid_gradient = array![0.1_f32, -0.1];
    let params2 = crate::bounds::alpha::AdamParams::new(0.01, 2);
    state.update_adam(0, &valid_gradient, &params2);

    for &a in state.alphas[0].iter() {
        assert!(
            a.is_finite(),
            "alpha must remain finite after recovery, got {}",
            a
        );
        assert!(
            (0.0..=1.0).contains(&a),
            "alpha must be in [0,1], got {}",
            a
        );
    }
}

/// Regression test for #2025: NaN gradient in SGD update must not corrupt alpha/velocity state.
/// The Adam path was tested by test_alpha_state_update_adam_nan_gradient_sanitized_2025;
/// this test covers the SGD path (AlphaState::update with momentum).
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_sgd_nan_gradient_sanitized_2025() {
    let bounds = vec![checked_bounds(
        array![-1.0_f32, -0.5].into_dyn(),
        array![1.0_f32, 0.5].into_dyn(),
    )];
    let relu_indices = vec![0];

    let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices).unwrap();

    // Inject NaN gradient via SGD path with momentum
    let nan_gradient = array![f32::NAN, f32::NAN];
    state.update(0, &nan_gradient, 0.01, 0.9);

    // After NaN sanitization: alpha should be 0.5, velocity should be reset to 0
    for (idx, &a) in state.alphas[0].iter().enumerate() {
        assert!(
            a.is_finite(),
            "alpha must be finite after NaN SGD gradient, got {}",
            a
        );
        assert_eq!(
            a, 0.5,
            "alpha[{idx}] must reset to 0.5 after NaN SGD gradient, got {a}"
        );
    }
    for (idx, &v) in state.velocity[0].iter().enumerate() {
        assert!(
            v.is_finite(),
            "velocity must be finite after NaN SGD gradient, got {}",
            v
        );
        assert_eq!(
            v, 0.0,
            "velocity[{idx}] must reset to 0.0 after NaN SGD gradient, got {v}"
        );
    }

    // Subsequent update with valid gradient should work normally
    let valid_gradient = array![0.1_f32, -0.1];
    state.update(0, &valid_gradient, 0.01, 0.9);

    for &a in state.alphas[0].iter() {
        assert!(
            a.is_finite(),
            "alpha must remain finite after SGD recovery, got {}",
            a
        );
        assert!(
            (0.0..=1.0).contains(&a),
            "alpha must be in [0,1] after SGD recovery, got {}",
            a
        );
    }
    for &v in state.velocity[0].iter() {
        assert!(
            v.is_finite(),
            "velocity must remain finite after SGD recovery, got {}",
            v
        );
    }
    assert!(
        state.alphas[0][0] < 0.5 && state.alphas[0][1] > 0.5,
        "alpha should move opposite gradient signs after recovery, got {:?}",
        state.alphas[0]
    );
    assert!(
        state.velocity[0][0] < 0.0 && state.velocity[0][1] > 0.0,
        "velocity should move opposite gradient signs after recovery, got {:?}",
        state.velocity[0]
    );
}
