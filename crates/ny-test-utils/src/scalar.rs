// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Assert two `f32` values are within `tolerance`.
pub fn assert_f32_close(actual: f32, expected: f32, tolerance: f32, context: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tolerance,
        "{context}: expected {actual} to be within {tolerance} of {expected} (diff {diff})"
    );
}

/// Assert an `f32` is NaN with a contextual failure message.
pub fn assert_f32_nan(value: f32, context: &str) {
    assert!(value.is_nan(), "{context}: expected NaN, got {value}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_f32_close_accepts_close_values() {
        assert_f32_close(1.0 + 5e-7, 1.0, 1e-6, "close");
    }

    #[test]
    #[should_panic(expected = "far")]
    fn test_assert_f32_close_rejects_far_values() {
        assert_f32_close(1.1, 1.0, 1e-3, "far");
    }

    #[test]
    fn test_assert_f32_nan_accepts_nan() {
        assert_f32_nan(f32::NAN, "nan");
    }

    #[test]
    #[should_panic(expected = "not-nan")]
    fn test_assert_f32_nan_rejects_finite_values() {
        assert_f32_nan(1.0, "not-nan");
    }
}
