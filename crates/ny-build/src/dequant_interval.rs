// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SOUND dequantization to f32 intervals for quantized weights (P8).
//!
//! Quantized formats (ONNX DequantizeLinear, GGUF Q-blocks) reconstruct a real
//! weight from an integer code via an affine map. On real hardware that map and
//! its inputs (`scale`, `zero_point`/min) are themselves finite-precision (often
//! f16 block deltas), so the *exact* real value is bracketed by an interval, not
//! a single f32. These helpers return a `[lo, hi]` interval that is guaranteed to
//! CONTAIN the true dequantized value under directed rounding, so a verdict over
//! the interval is valid for the deployed weights.
//!
//! SOUNDNESS: every floating-point operation here is widened outward with the
//! directed-rounding primitives from [`ny_tensor::rounding`]. The returned
//! interval is therefore a superset of the real-arithmetic result; it is never
//! narrowed. Pure helpers with no global state — full GGUF wiring (feeding these
//! into `WeightStore`) is a follow-on (see crate integration notes).

use ny_tensor::rounding::{next_down_f32, next_up_f32};

/// SOUND interval for a single ONNX-style dequantized value.
///
/// Computes a `[lo, hi]` interval guaranteed to contain the real value
/// `(q - zero_point) * scale` evaluated in exact arithmetic, accounting for the
/// rounding error of the two f32 operations (subtract, then multiply).
///
/// This matches ONNX `DequantizeLinear` semantics (`y = (x - zero_point) * scale`),
/// which is also the affine form used by GGUF `Q*_0` (zero_point packed into the
/// code, `zero_point = 0`) and the per-group base of K-quants.
///
/// # Soundness
/// Let `r = (q - zero_point) * scale` be the ideal real value. Each f32 op can
/// err by at most one ULP of its rounded result, so we evaluate the chain in f32
/// and then push the lower end down one ULP and the upper end up one ULP at each
/// rounding site. The result satisfies `lo <= r <= hi`. NaN/Inf inputs produce a
/// fully-unbounded `[-inf, +inf]` interval (maximally conservative).
///
/// # Arguments
/// - `q`: the integer quantization code (as `i32`).
/// - `scale`: the (already-dequantized-to-f32) block scale `d`.
/// - `zero_point`: the quantization zero point (`0.0` for symmetric formats).
#[must_use]
pub fn dequant_interval(q: i32, scale: f32, zero_point: f32) -> (f32, f32) {
    if !scale.is_finite() || !zero_point.is_finite() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    // q is exactly representable in f32 for the |q| ranges quantization uses
    // (4/5/8-bit codes are tiny); guard anyway by widening the centered code.
    let q_f = q as f32;
    // `(q as f32)` is exact for |q| <= 2^24; all real quant codes are far below
    // that, so no widening is needed on q itself. The affine ops below are the
    // only rounding sites.

    // Step 1: centered = q - zero_point. Widen both directions by one ULP.
    let centered = q_f - zero_point;
    if !centered.is_finite() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }
    let centered_lo = next_down_f32(centered);
    let centered_hi = next_up_f32(centered);

    // Step 2: multiply the widened centered interval by `scale`, taking the
    // extreme products over the interval endpoints x sign of scale, then widen
    // each multiplication outward by one more ULP.
    interval_scale(centered_lo, centered_hi, scale)
}

/// SOUND interval for a GGUF affine block value `q * scale + min`.
///
/// GGUF `Q4_1`/`Q5_1`/`Q8_1` and the per-group terms of K-quants use the form
/// `y = q * scale + min` (where `min` may be `-dmin * group_min`). Returns a
/// `[lo, hi]` interval containing the exact real value, widening every f32
/// rounding site outward. `q` is treated as an exact non-negative code.
///
/// # Soundness
/// Same one-ULP-per-op argument as [`dequant_interval`]: multiply then add, each
/// widened outward. NaN/Inf inputs yield `[-inf, +inf]`.
#[must_use]
pub fn dequant_block_affine_interval(q: i32, scale: f32, min: f32) -> (f32, f32) {
    if !scale.is_finite() || !min.is_finite() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }
    let q_f = q as f32;

    // Step 1: q * scale, widened outward.
    let (mut lo, mut hi) = interval_scale(q_f, q_f, scale);

    // Step 2: + min, widened outward.
    lo = next_down_f32(lo + min);
    hi = next_up_f32(hi + min);
    // A lower endpoint at +inf or an upper endpoint at -inf is an inverted/empty
    // interval (overflow on the wrong side); collapse to the maximally
    // conservative full interval rather than return an unsound inverted bound.
    let lo_overflowed_up = lo == f32::INFINITY;
    let hi_overflowed_down = hi == f32::NEG_INFINITY;
    if lo_overflowed_up || hi_overflowed_down || lo > hi {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }
    (lo, hi)
}

/// Multiply the interval `[x_lo, x_hi]` by a scalar `scale`, returning a SOUND
/// `[lo, hi]` widened outward by one ULP per multiplication.
///
/// Handles the sign of `scale` by selecting the correct endpoint products.
fn interval_scale(x_lo: f32, x_hi: f32, scale: f32) -> (f32, f32) {
    debug_assert!(x_lo <= x_hi, "interval_scale requires x_lo <= x_hi");
    let p1 = x_lo * scale;
    let p2 = x_hi * scale;
    let raw_lo = p1.min(p2);
    let raw_hi = p1.max(p2);
    let lo = next_down_f32(raw_lo);
    let hi = next_up_f32(raw_hi);
    if lo.is_nan() || hi.is_nan() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact real value of an ONNX dequant, computed in f64 for reference.
    fn exact_onnx(q: i32, scale: f32, zero_point: f32) -> f64 {
        (q as f64 - zero_point as f64) * scale as f64
    }

    /// The exact real value of a GGUF affine block, in f64.
    fn exact_affine(q: i32, scale: f32, min: f32) -> f64 {
        q as f64 * scale as f64 + min as f64
    }

    #[test]
    fn dequant_interval_contains_exact_value_symmetric() {
        // Symmetric Q8_0-style: zero_point = 0.
        for &q in &[-128, -7, -1, 0, 1, 42, 127] {
            for &scale in &[0.0, 0.5, 1.0, 0.013_671_875_f32, 12.5] {
                let (lo, hi) = dequant_interval(q, scale, 0.0);
                let exact = exact_onnx(q, scale, 0.0);
                assert!(
                    (lo as f64) <= exact && exact <= (hi as f64),
                    "q={q} scale={scale}: exact {exact} not in [{lo}, {hi}]"
                );
                assert!(lo <= hi, "interval inverted for q={q} scale={scale}");
            }
        }
    }

    #[test]
    fn dequant_interval_contains_exact_value_asymmetric() {
        // Asymmetric uint8 quant: zero_point in [0, 255].
        for &q in &[0, 1, 128, 200, 255] {
            for &zp in &[0.0_f32, 1.0, 128.0, 255.0] {
                for &scale in &[0.1_f32, 0.003_141_5, 7.25] {
                    let (lo, hi) = dequant_interval(q, scale, zp);
                    let exact = exact_onnx(q, scale, zp);
                    assert!(
                        (lo as f64) <= exact && exact <= (hi as f64),
                        "q={q} zp={zp} scale={scale}: exact {exact} not in [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn dequant_interval_brackets_value_that_is_inexact_in_f32() {
        // Choose scale so (q - zp) * scale is not exactly representable in f32,
        // forcing a nonzero rounding gap that the interval must straddle.
        let q = 100;
        let zp = 0.0_f32;
        let scale = 1.0_f32 / 3.0; // not representable; product rounds.
        let (lo, hi) = dequant_interval(q, scale, zp);
        let exact = exact_onnx(q, scale, zp);
        assert!(
            (lo as f64) <= exact && exact <= (hi as f64),
            "exact {exact} not bracketed by [{lo}, {hi}]"
        );
        assert!(
            lo < hi,
            "interval should have positive width for inexact product"
        );
    }

    #[test]
    fn dequant_block_affine_contains_exact_value() {
        // GGUF Q4_1-style: nibble code in [0,15], f16-ish scale and min.
        for q in 0..16 {
            for &scale in &[0.0_f32, 0.25, 0.012_5, 3.5] {
                for &min in &[-2.0_f32, -0.5, 0.0, 1.0, 8.25] {
                    let (lo, hi) = dequant_block_affine_interval(q, scale, min);
                    let exact = exact_affine(q, scale, min);
                    assert!(
                        (lo as f64) <= exact && exact <= (hi as f64),
                        "q={q} scale={scale} min={min}: exact {exact} not in [{lo}, {hi}]"
                    );
                    assert!(lo <= hi);
                }
            }
        }
    }

    #[test]
    fn non_finite_scale_yields_full_interval() {
        let (lo, hi) = dequant_interval(5, f32::NAN, 0.0);
        assert_eq!((lo, hi), (f32::NEG_INFINITY, f32::INFINITY));
        let (lo, hi) = dequant_interval(5, f32::INFINITY, 0.0);
        assert_eq!((lo, hi), (f32::NEG_INFINITY, f32::INFINITY));
        let (lo, hi) = dequant_block_affine_interval(5, f32::NAN, 0.0);
        assert_eq!((lo, hi), (f32::NEG_INFINITY, f32::INFINITY));
    }

    #[test]
    fn non_finite_zero_point_yields_full_interval() {
        let (lo, hi) = dequant_interval(5, 1.0, f32::INFINITY);
        assert_eq!((lo, hi), (f32::NEG_INFINITY, f32::INFINITY));
    }

    #[test]
    fn zero_scale_is_exact_zero_interval() {
        // scale = 0 => value is exactly 0; interval must contain 0 (it brackets
        // it via subnormal ULP widening around 0).
        let (lo, hi) = dequant_interval(127, 0.0, 3.0);
        assert!(
            (lo as f64) <= 0.0 && 0.0 <= (hi as f64),
            "[{lo},{hi}] must contain 0"
        );
    }

    #[test]
    fn negative_scale_selects_correct_endpoints() {
        // Negative scale flips ordering of endpoint products; result must still
        // be a valid, containing interval.
        let q = 50;
        let zp = 10.0_f32;
        let scale = -0.7_f32;
        let (lo, hi) = dequant_interval(q, scale, zp);
        let exact = exact_onnx(q, scale, zp);
        assert!(lo <= hi, "interval must be ordered even for negative scale");
        assert!(
            (lo as f64) <= exact && exact <= (hi as f64),
            "exact {exact} not in [{lo}, {hi}]"
        );
    }
}
