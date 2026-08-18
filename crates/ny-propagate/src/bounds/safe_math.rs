// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use ndarray::ArrayD;
#[cfg(test)]
use ny_core::{NyError, Result};

// Re-export NaN-propagating helpers from ny_core (canonical home per #2654).
pub use ny_core::{nan_propagating_max_zero, nan_propagating_min_zero};

pub(crate) use ny_core::f32_to_f64_exact as f32_to_f64_exact_for_bounds;
pub(crate) use ny_core::{
    f64_to_f32_down as f64_to_f32_down_for_bounds, f64_to_f32_up as f64_to_f32_up_for_bounds,
};

/// Safe multiplication for bound computation.
///
/// In interval arithmetic, a coefficient of 0 means no contribution,
/// so 0 * inf = 0 (not NaN). This prevents NaN propagation when
/// computing bounds with saturated (infinite) input bounds.
#[inline]
pub fn safe_mul_for_bounds(a: f32, x: f32) -> f32 {
    // Handle 0 * inf = 0 for both cases
    if a == 0.0 || x == 0.0 {
        0.0
    } else if a.is_nan() || x.is_nan() {
        // Propagate NaN explicitly to avoid hiding issues
        // NaN in coefficients means the linear bound is invalid
        f32::NAN
    } else {
        a * x
    }
}

/// f64 variant of [`safe_mul_for_bounds`] for f64-intermediate accumulation.
///
/// Same contract as the f32 version: a coefficient of exactly 0 means no
/// contribution, so 0 * inf = 0 (not NaN). NaN inputs are propagated. Used by
/// CROWN affine-substitution paths (e.g. BatchNorm bias accumulation) where a
/// degenerate Inf/NaN parameter (var+eps ~= 0) would otherwise produce a NaN
/// from 0 * inf and abort downstream `BoundedTensor::new` construction.
#[inline]
pub fn safe_mul_for_bounds_f64(a: f64, x: f64) -> f64 {
    if a == 0.0 || x == 0.0 {
        0.0
    } else if a.is_nan() || x.is_nan() {
        f64::NAN
    } else {
        a * x
    }
}

/// Safe addition for bound computation that handles inf + (-inf) = conservative bound.
///
/// When summing bound contributions, inf + (-inf) should produce:
/// - For lower bounds being computed: -inf (conservative, sound)
/// - For upper bounds being computed: +inf (conservative, sound)
///
/// The is_lower parameter indicates which bound is being computed.
///
/// REQUIRES: None (handles all `f32` values, including NaN/inf).
/// ENSURES: If `sum + term` is finite or ±inf, returns that value.
/// ENSURES: If `sum + term` is NaN due to `(+inf) + (-inf)` (or vice versa), returns:
///   - `-inf` when `is_lower == true` (sound lower bound),
///   - `+inf` when `is_lower == false` (sound upper bound).
///     ENSURES: If `sum` or `term` is NaN (not from inf + (-inf)), returns NaN (propagates invalid input).
#[inline]
pub fn safe_add_for_bounds_with_polarity(sum: f32, term: f32, is_lower: bool) -> f32 {
    let result = sum + term;
    // NOTE: We only "repair" NaN when it came from (+inf) + (-inf) (or vice versa).
    // If NaN is present in inputs, we propagate it to avoid hiding invalid bounds.
    if result.is_nan()
        && sum.is_infinite()
        && term.is_infinite()
        && (sum.is_sign_positive() != term.is_sign_positive())
    {
        if is_lower {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else {
        result
    }
}

/// Safe addition for bound computation when the polarity isn't known (e.g., intermediate
/// computations).
///
/// Repairs only NaN produced by `(+inf) + (-inf)` by choosing the conservative upper bound.
/// If NaN is present in the inputs, it is propagated.
#[cfg(test)]
#[inline]
pub fn safe_add_for_bounds(sum: f32, term: f32) -> f32 {
    safe_add_for_bounds_with_polarity(sum, term, false) // default to upper bound (conservative)
}

/// Safe element-wise array addition that handles inf + (-inf).
///
/// For lower bounds, NaN from inf + (-inf) becomes -inf (sound lower).
/// For upper bounds, NaN from inf + (-inf) becomes +inf (sound upper).
///
/// Thin wrapper around `safe_array_add_checked`. Retained for test convenience.
///
/// # Errors
/// - `NyError::ShapeMismatch` if shapes cannot be broadcast to a common shape.
#[cfg(test)]
pub fn safe_array_add(a: &ArrayD<f32>, b: &ArrayD<f32>, is_lower: bool) -> Result<ArrayD<f32>> {
    safe_array_add_checked(a, b, is_lower)
}

/// Checked variant of `safe_array_add` that returns `Result` on shape mismatch.
///
/// # Errors
/// - `NyError::ShapeMismatch` if shapes cannot be broadcast to a common shape.
#[cfg(test)]
pub fn safe_array_add_checked(
    a: &ArrayD<f32>,
    b: &ArrayD<f32>,
    is_lower: bool,
) -> Result<ArrayD<f32>> {
    use ndarray::{IxDyn, Zip};

    let target_shape = crate::shape::broadcast_shapes(a.shape(), b.shape())
        .ok_or_else(|| NyError::shape_mismatch(a.shape().to_vec(), b.shape().to_vec()))?;
    let target_dim = IxDyn(&target_shape);

    let a_bc = a
        .broadcast(target_dim.clone())
        .ok_or_else(|| NyError::shape_mismatch(target_shape.clone(), a.shape().to_vec()))?;
    let b_bc = b
        .broadcast(target_dim.clone())
        .ok_or_else(|| NyError::shape_mismatch(target_shape.clone(), b.shape().to_vec()))?;

    let mut result = ArrayD::<f32>::zeros(target_dim);
    Zip::from(&mut result)
        .and(a_bc)
        .and(b_bc)
        .for_each(|r, &av, &bv| {
            *r = safe_add_for_bounds_with_polarity(av, bv, is_lower);
        });
    Ok(result)
}

/// Safe multiplication for bound composition where 0 * inf = 0.
///
/// In interval arithmetic for bound propagation, a coefficient of 0 means
/// no contribution from that term, so 0 * inf should equal 0 (not NaN).
/// This helper is used by interval multiplication.
///
/// ENSURES: Returns 0.0 if either operand is zero.
/// ENSURES: Otherwise returns `a * b`.
#[inline]
pub fn safe_mul_pair_for_bounds(a: f32, b: f32) -> f32 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        a * b
    }
}

/// Binary64 product of two binary32 bit patterns, including subnormals.
///
/// Every finite binary32 product is exact in binary64 (at most 48 significant
/// bits), and the smallest nonzero product, 2^-298, is a normal binary64
/// number. Exact-zero detection is bit based so DAZ cannot reinterpret a
/// nonzero binary32 subnormal as zero.
#[inline]
fn safe_mul_pair_f64_exact_for_bounds(a: f32, b: f32) -> f64 {
    let a_is_zero = a.to_bits() & 0x7fff_ffff == 0;
    let b_is_zero = b.to_bits() & 0x7fff_ffff == 0;
    if a_is_zero || b_is_zero {
        0.0
    } else {
        f32_to_f64_exact_for_bounds(a) * f32_to_f64_exact_for_bounds(b)
    }
}

/// Interval multiplication for bound composition.
///
/// Computes the interval [lower, upper] that bounds all products a * b
/// where a ∈ [a_l, a_u] and b ∈ [b_l, b_u].
///
/// REQUIRES: `a_l <= a_u` and `b_l <= b_u` (well-formed intervals), OR any input is NaN.
/// ENSURES: Returns (lower, upper) such that for all a ∈ [a_l, a_u] and b ∈ [b_l, b_u]:
///   `lower <= a * b <= upper` — with `lower`/`upper` rounded OUTWARD so the bound is
///   sound even though each corner product `a*b` rounds to nearest in f32.
/// ENSURES: Handles 0 * inf = 0 via safe multiplication (no NaN propagation).
/// ENSURES: Returns (-inf, +inf) if any input is NaN (conservative fallback).
/// ENSURES: Result is always a well-formed interval (`lower <= upper`).
///
/// SOUNDNESS (#concretize-soundness-hardening): the four binary32 corner bit
/// patterns are decoded directly and multiplied exactly in binary64. Direct
/// decoding is required because DAZ can otherwise erase a binary32 subnormal
/// before conversion or multiplication (for example 2^-149 * 2^120 = 2^-29).
/// The exact binary64 endpoints are converted outward to binary32 and widened by
/// one additional binary32 ULP, preserving the historical conservative contract.
/// This matters because finite verdict-path callers exist — the bilinear N-D CROWN compose
/// (`layers/binary_ops/bilinear/nd_compose.rs`) and the bilinear interval relaxation
/// (`layers/binary_ops/bilinear/relaxation/interval.rs`) feed the returned products
/// into verdict bias bounds.
#[inline]
pub fn interval_mul_for_bounds(a_l: f32, a_u: f32, b_l: f32, b_u: f32) -> (f32, f32) {
    if a_l.is_nan() || a_u.is_nan() || b_l.is_nan() || b_u.is_nan() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    let products = [
        safe_mul_pair_f64_exact_for_bounds(a_l, b_l),
        safe_mul_pair_f64_exact_for_bounds(a_l, b_u),
        safe_mul_pair_f64_exact_for_bounds(a_u, b_l),
        safe_mul_pair_f64_exact_for_bounds(a_u, b_u),
    ];

    // Treat all-infinite products as unbounded to avoid +inf lower bounds in compositions.
    if products.iter().all(|p| p.is_infinite()) {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    let mut lower = products[0];
    let mut upper = products[0];

    for &p in &products[1..] {
        if p < lower {
            lower = p;
        }
        if p > upper {
            upper = p;
        }
    }

    // Handle NaN from unexpected sources (the input check above should catch all).
    if lower.is_nan() || upper.is_nan() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    // First convert the exact binary64 products outward without relying on a
    // binary32 subnormal result, then retain the helper's historical extra ULP
    // of widening. next_*_f32 are no-ops on infinities.
    (
        ny_tensor::next_down_f32(f64_to_f32_down_for_bounds(lower)),
        ny_tensor::next_up_f32(f64_to_f32_up_for_bounds(upper)),
    )
}

/// Sign-split linear composition of a known downstream coefficient with a
/// McCormick relaxation plane (lower/upper plane slopes).
///
/// This is the tight, sound composition rule used by the production bilinear
/// broadcast path (`bilinear/relaxation/broadcast.rs:174-194`): the downstream
/// A-matrix entries `ds_l` (lower direction) and `ds_u` (upper direction) are
/// *known* coefficients for their respective bound directions, NOT an interval.
/// Composing them with the McCormick lower/upper plane slopes `c_l`/`c_u` is
/// therefore a sign-split, not a 4-corner interval product:
///
/// - Lower direction (uses `ds_l`):
///   `lower = max(ds_l, 0) * c_l + min(ds_l, 0) * c_u`
///   (where `ds_l >= 0` the lower plane `c_l` produces the smaller value; where
///   `ds_l < 0` the upper plane `c_u` produces the smaller value once negated.)
/// - Upper direction (uses `ds_u`):
///   `upper = max(ds_u, 0) * c_u + min(ds_u, 0) * c_l`
///
/// This is `<=` the width of [`interval_mul_for_bounds`] (which over-covers by
/// treating both `[ds_l, ds_u]` and `[c_l, c_u]` as simultaneous intervals)
/// while remaining a valid over-approximation.
///
/// REQUIRES: `c_l <= c_u` (well-formed plane: lower plane <= upper plane), OR
///   any input is NaN.
/// ENSURES: Returns (lower, upper) such that the composition encloses the true
///   contribution for the lower and upper bound directions respectively.
/// ENSURES: Handles 0 * inf = 0 via safe multiplication (no spurious NaN).
/// ENSURES: Returns (-inf, +inf) if any input is NaN, and widens an individual
///   side to -inf / +inf if opposing infinite terms cancel to NaN.
#[inline]
pub fn sign_split_compose_for_bounds(ds_l: f32, ds_u: f32, c_l: f32, c_u: f32) -> (f32, f32) {
    if ds_l.is_nan() || ds_u.is_nan() || c_l.is_nan() || c_u.is_nan() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    // Lower-bound direction: known coefficient ds_l selects the lower plane
    // where it is non-negative and the upper plane where it is negative.
    let ds_l_pos = ds_l.max(0.0);
    let ds_l_neg = ds_l.min(0.0);
    let lower = safe_mul_pair_for_bounds(ds_l_pos, c_l) + safe_mul_pair_for_bounds(ds_l_neg, c_u);

    // Upper-bound direction: known coefficient ds_u selects the upper plane
    // where it is non-negative and the lower plane where it is negative.
    let ds_u_pos = ds_u.max(0.0);
    let ds_u_neg = ds_u.min(0.0);
    let upper = safe_mul_pair_for_bounds(ds_u_pos, c_u) + safe_mul_pair_for_bounds(ds_u_neg, c_l);

    // Opposing infinite terms (e.g. +inf + -inf) produce NaN; widen to the
    // sound saturating bound for that side rather than leaking NaN.
    let lower = if lower.is_nan() {
        f32::NEG_INFINITY
    } else {
        lower
    };
    let upper = if upper.is_nan() { f32::INFINITY } else { upper };
    (lower, upper)
}

// Re-export NaN-propagating min/max from ny_core (canonical home per #2654).
pub use ny_core::{nan_propagating_max, nan_propagating_min};

/// Safe addition for lower bounds (NaN → -inf).
///
/// When computing lower bounds, NaN results (e.g., from inf + (-inf))
/// should be replaced with -inf for sound over-approximation.
///
/// ENSURES: Returns `a + b` if the result is not NaN.
/// ENSURES: Returns `-inf` if the result is NaN.
#[inline]
pub fn safe_add_lower_for_bounds(a: f32, b: f32) -> f32 {
    let s = a + b;
    if s.is_nan() {
        f32::NEG_INFINITY
    } else {
        s
    }
}

/// Safe addition for upper bounds (NaN → +inf).
///
/// When computing upper bounds, NaN results (e.g., from inf + (-inf))
/// should be replaced with +inf for sound over-approximation.
///
/// ENSURES: Returns `a + b` if the result is not NaN.
/// ENSURES: Returns `+inf` if the result is NaN.
#[inline]
pub fn safe_add_upper_for_bounds(a: f32, b: f32) -> f32 {
    let s = a + b;
    if s.is_nan() {
        f32::INFINITY
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests for nan_propagating_{max,min}_zero and nan_propagating_{min,max}
    // live in ny-core/src/nan_math.rs (canonical home per #2654).
    // This module only tests safe_math-specific functions and composition.

    /// Verify that NaN propagation through the full safe_mul_for_bounds chain works.
    /// This is the end-to-end test for the #2415 fix: NaN coefficient → NaN output.
    #[test]
    fn test_nan_coefficient_propagates_through_safe_mul() {
        // With the old code: NaN.max(0.0) = 0.0, safe_mul(0.0, 1.0) = 0.0 → NaN absorbed
        // With the fix: nan_propagating_max_zero(NaN) = NaN, safe_mul(NaN, 1.0) = NaN ✓
        let nan_coeff = f32::NAN;
        let result = safe_mul_for_bounds(nan_propagating_max_zero(nan_coeff), 1.0);
        assert!(
            result.is_nan(),
            "NaN coefficient must propagate through safe_mul"
        );

        let result_min = safe_mul_for_bounds(nan_propagating_min_zero(nan_coeff), 1.0);
        assert!(
            result_min.is_nan(),
            "NaN coefficient must propagate through min path too"
        );
    }

    #[test]
    fn test_safe_mul_for_bounds_f64_zero_times_inf_is_zero() {
        assert_eq!(safe_mul_for_bounds_f64(0.0, f64::INFINITY), 0.0);
        assert_eq!(safe_mul_for_bounds_f64(f64::INFINITY, 0.0), 0.0);
        assert_eq!(safe_mul_for_bounds_f64(0.0, f64::NEG_INFINITY), 0.0);
        // Zero short-circuits even when the other operand is NaN.
        assert_eq!(safe_mul_for_bounds_f64(0.0, f64::NAN), 0.0);
    }

    #[test]
    fn test_safe_mul_for_bounds_f64_propagates_nan_and_inf() {
        assert!(safe_mul_for_bounds_f64(1.0, f64::NAN).is_nan());
        assert_eq!(safe_mul_for_bounds_f64(2.0, f64::INFINITY), f64::INFINITY);
        assert_eq!(
            safe_mul_for_bounds_f64(-2.0, f64::INFINITY),
            f64::NEG_INFINITY
        );
        assert_eq!(safe_mul_for_bounds_f64(3.0, 4.0), 12.0);
    }

    #[test]
    fn interval_mul_decodes_subnormal_operands_bit_exactly() {
        let tiny = f32::from_bits(1);
        let large = 2.0_f32.powi(120);
        let exact_tiny = 2.0_f64.powi(-149);
        let exact_product = 2.0_f64.powi(-29);

        assert_eq!(f32_to_f64_exact_for_bounds(tiny), exact_tiny);

        let (lower, upper) = interval_mul_for_bounds(tiny, tiny, large, large);
        assert!(
            f32_to_f64_exact_for_bounds(lower) <= exact_product,
            "lower {lower:e} excludes exact product {exact_product:e}"
        );
        assert!(
            f32_to_f64_exact_for_bounds(upper) >= exact_product,
            "upper {upper:e} excludes exact product {exact_product:e}"
        );
    }

    /// NaN in any input widens to the saturating interval (matches
    /// interval_mul_for_bounds conservative fallback).
    #[test]
    fn test_sign_split_compose_nan_widens() {
        assert_eq!(
            sign_split_compose_for_bounds(f32::NAN, 1.0, 0.5, 1.5),
            (f32::NEG_INFINITY, f32::INFINITY)
        );
        assert_eq!(
            sign_split_compose_for_bounds(1.0, 1.0, f32::NAN, 1.5),
            (f32::NEG_INFINITY, f32::INFINITY)
        );
    }

    /// 0 * inf = 0 (no spurious NaN) — a zero downstream coefficient drops the
    /// term even with an infinite plane slope.
    #[test]
    fn test_sign_split_compose_zero_times_inf() {
        assert_eq!(
            sign_split_compose_for_bounds(0.0, 0.0, f32::INFINITY, f32::INFINITY),
            (0.0, 0.0)
        );
    }

    /// Opposing infinite plane contributions cancel to NaN per-side; widen to
    /// the sound saturating bound rather than leaking NaN. Mirrors the N-D
    /// compose NaN-fallback regression (#4204).
    #[test]
    fn test_sign_split_compose_opposing_infinities_widen() {
        // ds_l > 0 selects c_l = -inf for lower; ds_u > 0 selects c_u = +inf.
        // Each side is a single finite*inf term here, so no cancellation:
        let (lo, hi) = sign_split_compose_for_bounds(2.0, 2.0, f32::NEG_INFINITY, f32::INFINITY);
        assert_eq!((lo, hi), (f32::NEG_INFINITY, f32::INFINITY));
    }

    /// Sign-split matches the production broadcast.rs rule exactly:
    /// lower = pos(ds_l)*c_l + neg(ds_l)*c_u,
    /// upper = pos(ds_u)*c_u + neg(ds_u)*c_l.
    #[test]
    fn test_sign_split_compose_matches_broadcast_rule() {
        for &(ds_l, ds_u, c_l, c_u) in &[
            (1.0_f32, 2.0_f32, 0.5_f32, 1.5_f32),
            (-1.0, 2.0, 0.5, 1.5),
            (-3.0, -1.0, -2.0, 0.0),
            (0.0, 1.0, -1.0, 1.0),
        ] {
            let expected_lo = ds_l.max(0.0) * c_l + ds_l.min(0.0) * c_u;
            let expected_hi = ds_u.max(0.0) * c_u + ds_u.min(0.0) * c_l;
            let (lo, hi) = sign_split_compose_for_bounds(ds_l, ds_u, c_l, c_u);
            assert!(
                (lo - expected_lo).abs() < 1e-6,
                "lower mismatch: got {lo}, want {expected_lo}"
            );
            assert!(
                (hi - expected_hi).abs() < 1e-6,
                "upper mismatch: got {hi}, want {expected_hi}"
            );
        }
    }

    proptest::proptest! {
        /// SOUNDNESS + TIGHTNESS: `ds_l` / `ds_u` are the KNOWN downstream
        /// coefficients for the lower / upper bound directions (NOT a single
        /// interval), and `[c_l, c_u]` is a McCormick plane (lower <= upper).
        /// The two returned values are accumulated into *separate* lower / upper
        /// bound accumulators (as in nd_compose / broadcast.rs), so they are
        /// per-direction bounds and need not be mutually ordered. The sign-split
        /// result must:
        ///   (a) ENCLOSE the true contribution: for every plane value
        ///       c in [c_l, c_u], `ds_l * c >= ss_lo` and `ds_u * c <= ss_hi`
        ///       (soundness of the lower / upper directions), AND
        ///   (b) be at least as tight as the loose 4-corner interval product:
        ///       `ss_lo >= iv_lo` (tighter lower) and `ss_hi <= iv_hi`
        ///       (tighter upper) — contained within interval_mul_for_bounds.
        #[test]
        fn proptest_sign_split_encloses_true_and_is_tighter(
            ds_l in -10.0_f32..10.0,
            ds_u in -10.0_f32..10.0,
            c_l in -10.0_f32..10.0,
            c_u in -10.0_f32..10.0,
        ) {
            // Well-formed plane (lower <= upper). ds_l / ds_u are independent
            // per-direction coefficients, not an ordered interval.
            let (c_l, c_u) = (c_l.min(c_u), c_l.max(c_u));

            let (ss_lo, ss_hi) = sign_split_compose_for_bounds(ds_l, ds_u, c_l, c_u);
            let (iv_lo, iv_hi) = interval_mul_for_bounds(ds_l, ds_u, c_l, c_u);

            // (a) Enclosure: the lower direction uses the KNOWN coefficient ds_l,
            // so for any plane value c in [c_l, c_u], ds_l*c must be >= ss_lo;
            // the upper direction uses ds_u, so ds_u*c must be <= ss_hi.
            for &c in &[c_l, c_u, f32::midpoint(c_l, c_u)] {
                let lower_dir = ds_l * c;
                let upper_dir = ds_u * c;
                proptest::prop_assert!(
                    lower_dir >= ss_lo - 1e-3,
                    "sign-split lower {ss_lo} must enclose ds_l*c = {lower_dir}"
                );
                proptest::prop_assert!(
                    upper_dir <= ss_hi + 1e-3,
                    "sign-split upper {ss_hi} must enclose ds_u*c = {upper_dir}"
                );
            }

            // (b) Tightness vs the loose 4-corner product: the sign-split lower
            // direction is >= the loose lower, and the upper direction is <= the
            // loose upper (each side contained within interval_mul_for_bounds).
            proptest::prop_assert!(
                ss_lo >= iv_lo - 1e-3,
                "sign-split lower {ss_lo} must be >= interval-mul lower {iv_lo}"
            );
            proptest::prop_assert!(
                ss_hi <= iv_hi + 1e-3,
                "sign-split upper {ss_hi} must be <= interval-mul upper {iv_hi}"
            );
        }
    }
}
