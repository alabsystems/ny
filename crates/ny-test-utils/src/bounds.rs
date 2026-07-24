// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_tensor::BoundedTensor;

/// Use for tiny CPU-vs-GPU parity checks on small deterministic examples.
pub const GPU_REGRESSION_TINY_EPSILON: f32 = 1e-5;
/// Use for near-exact CPU-vs-GPU parity checks on simple extracted models.
pub const GPU_REGRESSION_NEAR_EXACT_EPSILON: f32 = 1e-6;
/// Use for strict parity checks where only a few fused operations accumulate error.
pub const GPU_REGRESSION_STRICT_EPSILON: f32 = 1e-4;
/// Use for larger GPU regression checks that intentionally tolerate fused-kernel noise.
pub const GPU_REGRESSION_RELAXED_EPSILON: f32 = 1e-3;

/// Assert two `f32` slices match elementwise within `epsilon`.
pub fn assert_slice_close(actual: &[f32], expected: &[f32], epsilon: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );

    for (idx, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual_value - expected_value).abs();
        assert!(
            diff <= epsilon,
            "{label}[{idx}] actual={actual_value} expected={expected_value} diff={diff} epsilon={epsilon}"
        );
    }
}

/// Assert two bounded tensors match elementwise within `epsilon`.
pub fn assert_bounded_tensor_close(
    actual: &BoundedTensor,
    expected: &BoundedTensor,
    epsilon: f32,
    label: &str,
) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape mismatch actual={:?} expected={:?}",
        actual.shape(),
        expected.shape()
    );

    for (idx, (&actual_lower, &expected_lower)) in actual
        .lower()
        .iter()
        .zip(expected.lower().iter())
        .enumerate()
    {
        let diff = (actual_lower - expected_lower).abs();
        assert!(
            diff <= epsilon,
            "{label}: lower[{idx}] actual={actual_lower} expected={expected_lower} diff={diff} epsilon={epsilon}"
        );
    }

    for (idx, (&actual_upper, &expected_upper)) in actual
        .upper()
        .iter()
        .zip(expected.upper().iter())
        .enumerate()
    {
        let diff = (actual_upper - expected_upper).abs();
        assert!(
            diff <= epsilon,
            "{label}: upper[{idx}] actual={actual_upper} expected={expected_upper} diff={diff} epsilon={epsilon}"
        );
    }
}

/// Assert two `f32` slices match elementwise within a relative tolerance.
///
/// For each pair, the tolerance is `rel_epsilon * max(|a|, |e|, 1.0)`.
/// This handles values near zero (floor of 1.0) and scales with magnitude.
pub fn assert_slice_close_relative(
    actual: &[f32],
    expected: &[f32],
    rel_epsilon: f32,
    label: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );

    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = a.abs().max(e.abs()).max(1.0);
        let tol = rel_epsilon * scale;
        assert!(
            (a - e).abs() <= tol,
            "{label}[{idx}] actual={a:.6e} expected={e:.6e} tol={tol:.6e}"
        );
    }
}

/// Assert that `current` bounds are at least as tight as `baseline` (within tolerance).
///
/// Checks `current_lower >= baseline_lower - tol` and `current_upper <= baseline_upper + tol`.
/// Useful for verifying that alpha-CROWN or iterative refinement never loosens bounds
/// relative to IBP or a previous iteration.
pub fn assert_bounds_do_not_loosen(
    current: &BoundedTensor,
    baseline: &BoundedTensor,
    tol: f32,
    label: &str,
) {
    assert_eq!(
        current.shape(),
        baseline.shape(),
        "{label}: shape mismatch {:?} vs {:?}",
        current.shape(),
        baseline.shape()
    );
    for (idx, ((&cur_l, &cur_u), (&base_l, &base_u))) in current
        .lower()
        .iter()
        .zip(current.upper().iter())
        .zip(baseline.lower().iter().zip(baseline.upper().iter()))
        .enumerate()
    {
        assert!(
            cur_l >= base_l - tol,
            "{label}: lower[{idx}] loosened: current={cur_l}, baseline={base_l}"
        );
        assert!(
            cur_u <= base_u + tol,
            "{label}: upper[{idx}] loosened: current={cur_u}, baseline={base_u}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_slice_close_accepts_matching_values() {
        assert_slice_close(
            &[1.0, 2.0],
            &[1.0, 2.0 + 5e-5],
            GPU_REGRESSION_STRICT_EPSILON,
            "ok",
        );
    }

    #[test]
    #[should_panic(expected = "bad[1]")]
    fn test_assert_slice_close_rejects_large_difference() {
        assert_slice_close(
            &[1.0, 2.0],
            &[1.0, 2.1],
            GPU_REGRESSION_STRICT_EPSILON,
            "bad",
        );
    }
}
