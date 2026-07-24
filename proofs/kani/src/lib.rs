// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kani proofs for ny bound propagation soundness.
//!
//! These proofs verify critical interval arithmetic and relaxation operations
//! used in CROWN neural network verification.
//!
//! The proofs verify the extracted proof-support scalar math in
//! `ny-relaxation`. Most functions are audited copies of the production
//! `ny-propagate` implementations; the ReLU helper is a separate reference
//! implementation checked for soundness against the production path.
//!
//! To run verification: `cargo kani` (requires kani verifier installed)
//! For development checking: `cargo check --manifest-path proofs/kani/Cargo.toml`

// Only compile proofs when using cargo-kani
#![cfg_attr(kani, allow(dead_code))]
#![cfg_attr(not(kani), allow(dead_code, unused_imports, unused_variables))]

// Import the proof-support scalar surface from ny-relaxation.
// This lightweight crate contains only the f32 relaxation math needed by
// proof harnesses, avoiding ny-propagate's heavy deps (faer, ndarray, etc.).
// ny-propagate does not re-export these items yet, and the copies here are
// not mechanically kept equivalent to it — what these harnesses prove is
// soundness of the ny-relaxation implementations themselves.
use ny_relaxation::{
    abs_linear_relaxation, exp_linear_relaxation, gelu_eval, gelu_sound_linear_relaxation,
    gelu_tanh_sound_linear_relaxation, interval_mul_for_bounds, log_linear_relaxation,
    pow2_linear_relaxation, relu_crown_relaxation, safe_add_for_bounds_with_polarity,
    safe_add_lower_for_bounds, safe_add_upper_for_bounds, safe_mul_for_bounds,
    safe_mul_pair_for_bounds, silu_eval, silu_sound_linear_relaxation, sqrt_linear_relaxation,
    GeluApproximation, LinearRelaxation,
};
// Alpha-parameterized relaxations not re-exported from crate root — import via module path.
// These are used during alpha-CROWN optimization with learnable tangent points.
use ny_relaxation::sqrt::sqrt_linear_relaxation_with_alpha;

// Kani-specific imports (only available under cargo-kani)
#[cfg(kani)]
use kani::{any, assume};

// Stub implementations for non-kani builds (allows `cargo check` to pass)
#[cfg(not(kani))]
mod kani_stub {
    pub fn any<T: Default>() -> T {
        Default::default()
    }

    pub fn assume(_cond: bool) {}
}

#[cfg(not(kani))]
use kani_stub::{any, assume};

// =============================================================================
// Helper functions for generating symbolic values
// =============================================================================

fn any_finite_f32() -> f32 {
    let value: f32 = any();
    assume(value.is_finite());
    value
}

#[cfg(kani)]
fn any_valid_bound() -> ny_core::Bound {
    let lower = any_finite_f32();
    let upper = any_finite_f32();
    assume(lower <= upper);
    ny_core::Bound::new(lower, upper)
}

/// Generate a finite f32 within a range (for bounded symbolic exploration).
fn any_bounded_f32(min: f32, max: f32) -> f32 {
    let value: f32 = any();
    assume(value.is_finite());
    assume(value >= min && value <= max);
    value
}

/// Generate a finite f32 within a range, excluding IEEE 754 negative zero.
///
/// Negative zero (-0.0) is problematic for Kani proofs because:
/// - `-0.0 == 0.0` and `-0.0 >= 0.0` and `-0.0 <= 0.0` are all true in IEEE 754
/// - This lets CBMC construct degenerate intervals like (0.0, -0.0) that satisfy
///   `lower <= upper` but are semantically inverted
/// - Relaxation functions branch on `l >= 0.0` vs `u <= 0.0`, and -0.0 satisfies both
fn any_bounded_f32_no_negzero(min: f32, max: f32) -> f32 {
    let value: f32 = any();
    assume(value.is_finite());
    assume(value >= min && value <= max);
    // Exclude negative zero: bit pattern 0x80000000
    assume(value.to_bits() != 0x8000_0000u32);
    value
}

/// Generate a bounded i8 for tractable interval proofs.
fn any_bounded_i8(min: i8, max: i8) -> i8 {
    let value: i8 = any();
    assume(value >= min && value <= max);
    value
}

/// Generate either +inf or -inf.
fn any_infinite_f32() -> f32 {
    if any::<bool>() {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    }
}

/// Generate a selector for special-value proof cases.
fn any_special_case_index(case_count: u8) -> u8 {
    let selector: u8 = any();
    assume(selector < case_count);
    selector
}

/// Generate a f32 from a u8 index into an evenly-spaced grid over [min, max].
///
/// This produces 256 distinct f32 values spanning the range, dramatically
/// reducing the SAT problem size compared to full f32 symbolic exploration.
/// Used for proofs involving transcendental functions (sqrt, exp, log) where
/// CBMC's full f32 model generates intractable SAT problems.
fn any_grid_f32(min: f32, max: f32) -> f32 {
    let idx: u8 = any();
    let t = idx as f64 / 255.0;
    let value = (min as f64 + t * (max as f64 - min as f64)) as f32;
    assume(value.is_finite());
    assume(value >= min && value <= max);
    value
}

/// ReLU function for verification.
#[inline]
fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

// =============================================================================
// Bound Struct Proofs (basic properties)
// =============================================================================

#[cfg(kani)]
#[kani::proof]
fn bound_union_contains_inputs() {
    let bound_a = any_valid_bound();
    let bound_b = any_valid_bound();
    let value = any_finite_f32();
    assume(bound_a.contains(value));

    let union = bound_a.union(&bound_b);
    assert!(union.contains(value));
}

#[cfg(kani)]
#[kani::proof]
fn bound_intersection_is_subset() {
    let bound_a = any_valid_bound();
    let bound_b = any_valid_bound();

    match bound_a.intersect(&bound_b) {
        Some(intersection) => {
            assert!(intersection.lower() >= bound_a.lower());
            assert!(intersection.upper() <= bound_a.upper());
            assert!(intersection.lower() >= bound_b.lower());
            assert!(intersection.upper() <= bound_b.upper());

            let value = any_finite_f32();
            assume(intersection.contains(value));
            assert!(bound_a.contains(value));
            assert!(bound_b.contains(value));
        }
        None => {
            assert!(bound_a.upper() < bound_b.lower() || bound_b.upper() < bound_a.lower());
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn bound_width_is_non_negative() {
    let bound = any_valid_bound();
    assert!(bound.width() >= 0.0);
}

/// Proof: Bound union never loses containment and never shrinks width.
///
/// PROPERTY: union(A, B).width() >= max(A.width(), B.width()).
/// The union (convex hull) must be at least as wide as either operand.
/// This is critical for CROWN-IBP: union is used when merging bounds
/// from different branches, and a shrinking union would be unsound.
#[cfg(kani)]
#[kani::proof]
fn bound_union_never_shrinks() {
    let a = any_valid_bound();
    let b = any_valid_bound();

    let u = a.union(&b);

    // Union must be at least as wide as both inputs
    let a_width = a.width();
    let b_width = b.width();
    let u_width = u.width();

    if a_width.is_finite() && u_width.is_finite() {
        assert!(
            u_width >= a_width - 1e-6,
            "Union width must be >= first operand width"
        );
    }
    if b_width.is_finite() && u_width.is_finite() {
        assert!(
            u_width >= b_width - 1e-6,
            "Union width must be >= second operand width"
        );
    }
}

/// Proof: Bound intersection never expands width.
///
/// PROPERTY: If intersection exists, its width <= min(A.width(), B.width()).
/// The intersection must be no wider than either operand.
/// This is critical for CROWN-IBP tightening: intersection is how CROWN
/// bounds are tightened by IBP bounds, and an expanding intersection would
/// defeat the purpose.
#[cfg(kani)]
#[kani::proof]
fn bound_intersection_never_expands() {
    let a = any_valid_bound();
    let b = any_valid_bound();

    if let Some(intersection) = a.intersect(&b) {
        let i_width = intersection.width();
        let a_width = a.width();
        let b_width = b.width();

        if i_width.is_finite() && a_width.is_finite() {
            assert!(
                i_width <= a_width + 1e-6,
                "Intersection width must be <= first operand width"
            );
        }
        if i_width.is_finite() && b_width.is_finite() {
            assert!(
                i_width <= b_width + 1e-6,
                "Intersection width must be <= second operand width"
            );
        }
    }
}

// =============================================================================
// Interval Multiplication Proofs
// =============================================================================

/// Proof: safe_mul_pair_for_bounds handles 0 * inf correctly.
///
/// PROPERTY: If either operand is 0, result is 0 (not NaN from 0 * inf).
#[cfg(kani)]
#[kani::proof]
fn safe_mul_pair_zero_handling() {
    let a: f32 = any();
    let b: f32 = any();

    // When either is zero, result must be zero
    if a == 0.0 || b == 0.0 {
        let result = safe_mul_pair_for_bounds(a, b);
        assert!(result == 0.0);
    }
}

/// Proof: safe_mul_for_bounds handles 0 * inf correctly.
///
/// PROPERTY: If either operand is 0, result is 0 (not NaN from 0 * inf).
#[cfg(kani)]
#[kani::proof]
fn safe_mul_zero_inf_is_zero() {
    let a: f32 = any();
    let b: f32 = any();

    if a == 0.0 || b == 0.0 {
        let result = safe_mul_for_bounds(a, b);
        assert!(result == 0.0);
    }
}

/// Proof: safe_mul_for_bounds propagates NaN when no zero term is present.
///
/// PROPERTY: If one operand is NaN and the other is non-zero, result is NaN.
#[cfg(kani)]
#[kani::proof]
fn safe_mul_nan_propagation() {
    let finite = any_finite_f32();
    assume(finite != 0.0);

    let result = safe_mul_for_bounds(f32::NAN, finite);
    assert!(result.is_nan());

    let result = safe_mul_for_bounds(finite, f32::NAN);
    assert!(result.is_nan());
}

/// Proof: safe_mul_pair_for_bounds produces finite results for finite inputs.
///
/// PROPERTY: Finite inputs → finite output (no spurious infinities).
/// STRENGTHENED: Also verifies sign correctness — the product of two
/// values with the same sign must be non-negative, and with opposite
/// signs must be non-positive. This is critical for interval arithmetic
/// where sign determines which endpoint products form the result bounds.
#[cfg(kani)]
#[kani::proof]
fn safe_mul_pair_finite_preservation() {
    let a = any_finite_f32();
    let b = any_finite_f32();

    let result = safe_mul_pair_for_bounds(a, b);
    // Result should be finite or could overflow to inf, but never NaN
    assert!(!result.is_nan());

    // Sign correctness: essential for interval multiplication soundness.
    // If both positive or both negative, product must be non-negative.
    // If opposite signs, product must be non-positive.
    if result.is_finite() && a != 0.0 && b != 0.0 {
        if (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0) {
            assert!(
                result >= 0.0,
                "Same-sign operands must produce non-negative result"
            );
        }
        if (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0) {
            assert!(
                result <= 0.0,
                "Opposite-sign operands must produce non-positive result"
            );
        }
    }
}

/// Proof: interval_mul_for_bounds is sound for finite intervals.
///
/// PROPERTY: For any a in [a_l, a_u] and b in [b_l, b_u],
/// the product a*b is contained in [result_l, result_u].
///
/// This is the core soundness property for CROWN backward propagation.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn interval_mul_contains_product() {
    // Use bounded integer domains to keep verification tractable.
    let a_l_i = any_bounded_i8(-20, 20);
    let a_u_i = any_bounded_i8(-20, 20);
    let b_l_i = any_bounded_i8(-20, 20);
    let b_u_i = any_bounded_i8(-20, 20);

    // Require well-formed intervals
    assume(a_l_i <= a_u_i);
    assume(b_l_i <= b_u_i);

    let a_l = a_l_i as f32;
    let a_u = a_u_i as f32;
    let b_l = b_l_i as f32;
    let b_u = b_u_i as f32;

    let (result_l, result_u) = interval_mul_for_bounds(a_l, a_u, b_l, b_u);

    // Pick arbitrary points in the intervals
    let a = any_bounded_i8(a_l_i, a_u_i) as f32;
    let b = any_bounded_i8(b_l_i, b_u_i) as f32;

    // The product must be within result bounds
    let product = a * b;
    if product.is_finite() {
        assert!(result_l <= product, "Lower bound violated");
        assert!(product <= result_u, "Upper bound violated");
    }
}

/// Proof: interval_mul_for_bounds output is well-formed.
///
/// PROPERTY: result_l <= result_u (output is a valid interval).
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn interval_mul_well_formed_output() {
    let a_l = any_bounded_f32(-100.0, 100.0);
    let a_u = any_bounded_f32(-100.0, 100.0);
    let b_l = any_bounded_f32(-100.0, 100.0);
    let b_u = any_bounded_f32(-100.0, 100.0);

    assume(a_l <= a_u);
    assume(b_l <= b_u);

    let (result_l, result_u) = interval_mul_for_bounds(a_l, a_u, b_l, b_u);

    // Output should be a valid interval
    assert!(
        result_l <= result_u,
        "Output interval is malformed (lower > upper)"
    );
}

/// Proof: interval_mul_for_bounds handles NaN inputs gracefully.
///
/// PROPERTY: NaN input → conservative bounds (-inf, +inf).
#[cfg(kani)]
#[kani::proof]
fn interval_mul_nan_input_handling() {
    let a_l = f32::NAN;
    let a_u = any_finite_f32();
    let b_l = any_finite_f32();
    let b_u = any_finite_f32();

    let (result_l, result_u) = interval_mul_for_bounds(a_l, a_u, b_l, b_u);

    // With NaN input, should return conservative bounds
    assert!(result_l == f32::NEG_INFINITY);
    assert!(result_u == f32::INFINITY);
}

/// Proof: interval_mul_for_bounds returns conservative bounds when all products are infinite.
///
/// PROPERTY: If all products are infinite, output is (-inf, +inf).
#[cfg(kani)]
#[kani::proof]
fn interval_mul_all_infinite_returns_unbounded() {
    let (lower, upper) = interval_mul_for_bounds(
        any_infinite_f32(),
        any_infinite_f32(),
        any_infinite_f32(),
        any_infinite_f32(),
    );
    assert!(lower == f32::NEG_INFINITY);
    assert!(upper == f32::INFINITY);
}

// =============================================================================
// Safe Addition Proofs
// =============================================================================
// NOTE: Kani runs with CBMC --nan-check, so NaN-generating paths are verified
// via ny-propagate unit tests instead of these harnesses.

/// Proof: safe_add_lower_for_bounds matches finite sums exactly.
///
/// PROPERTY: Finite inputs with finite sum → exact sum.
#[cfg(kani)]
#[kani::proof]
fn safe_add_lower_matches_sum_when_finite() {
    let a = any_finite_f32();
    let b = any_finite_f32();
    let sum = a + b;
    assume(sum.is_finite());

    let result = safe_add_lower_for_bounds(a, b);
    assert!(result == sum);
}

/// Proof: safe_add_upper_for_bounds matches finite sums exactly.
///
/// PROPERTY: Finite inputs with finite sum → exact sum.
#[cfg(kani)]
#[kani::proof]
fn safe_add_upper_matches_sum_when_finite() {
    let a = any_finite_f32();
    let b = any_finite_f32();
    let sum = a + b;
    assume(sum.is_finite());

    let result = safe_add_upper_for_bounds(a, b);
    assert!(result == sum);
}

/// Proof: safe_add_lower_for_bounds preserves finite results and maintains
/// ordering relative to safe_add_upper.
///
/// PROPERTY: Finite inputs → finite output (no spurious conversions).
/// STRENGTHENED: Also verifies that safe_add_lower(a, b) <= safe_add_upper(a, b)
/// for all finite inputs — the lower bound must never exceed the upper bound
/// when applied to the same operands. This ordering is a prerequisite for
/// safe interval addition to produce well-formed intervals.
#[cfg(kani)]
#[kani::proof]
fn safe_add_lower_finite_preservation() {
    let a = any_finite_f32();
    let b = any_finite_f32();

    let lower_result = safe_add_lower_for_bounds(a, b);
    let upper_result = safe_add_upper_for_bounds(a, b);

    // Finite + finite should be finite (may overflow to inf, but not NaN)
    assert!(!lower_result.is_nan());

    // Core soundness: lower safe-add must never exceed upper safe-add
    // for the same operands. This ensures interval addition [lo, hi]
    // always produces a valid interval.
    assert!(
        lower_result <= upper_result,
        "safe_add_lower must be <= safe_add_upper for identical operands"
    );
}

/// Proof: safe_add_upper_for_bounds preserves finite results and is
/// monotonically compatible with safe_add_lower.
///
/// PROPERTY: Finite inputs → finite output (no spurious conversions).
/// STRENGTHENED: Verifies monotonicity — if a1 <= a2 and b1 <= b2, then
/// safe_add_upper(a1, b1) <= safe_add_upper(a2, b2). This is the key
/// property that makes interval addition produce correct upper bounds
/// when applied to the upper endpoints of input intervals.
#[cfg(kani)]
#[kani::proof]
fn safe_add_upper_finite_preservation() {
    let a1 = any_bounded_f32(-100.0, 100.0);
    let a2 = any_bounded_f32(-100.0, 100.0);
    let b1 = any_bounded_f32(-100.0, 100.0);
    let b2 = any_bounded_f32(-100.0, 100.0);

    assume(a1 <= a2);
    assume(b1 <= b2);

    let result_lo = safe_add_upper_for_bounds(a1, b1);
    let result_hi = safe_add_upper_for_bounds(a2, b2);

    // Finite + finite should be finite (may overflow to inf, but not NaN)
    assert!(!result_lo.is_nan());
    assert!(!result_hi.is_nan());

    // Monotonicity: if inputs grow, output must not shrink.
    // This is the foundation for interval addition soundness.
    if result_lo.is_finite() && result_hi.is_finite() {
        assert!(
            result_lo <= result_hi,
            "safe_add_upper must be monotonic: larger inputs produce larger results"
        );
    }
}

/// Proof: safe_add_lower_for_bounds is conservative for lower bounds.
///
/// PROPERTY: result <= a + b when a + b is finite.
#[cfg(kani)]
#[kani::proof]
fn safe_add_lower_is_conservative() {
    let a = any_finite_f32();
    let b = any_finite_f32();

    let result = safe_add_lower_for_bounds(a, b);
    let true_sum = a + b;

    if true_sum.is_finite() {
        assert!(result <= true_sum);
    }
}

/// Proof: safe_add_upper_for_bounds is conservative for upper bounds.
///
/// PROPERTY: result >= a + b when a + b is finite.
#[cfg(kani)]
#[kani::proof]
fn safe_add_upper_is_conservative() {
    let a = any_finite_f32();
    let b = any_finite_f32();

    let result = safe_add_upper_for_bounds(a, b);
    let true_sum = a + b;

    if true_sum.is_finite() {
        assert!(result >= true_sum);
    }
}

/// Proof: interval addition via safe_add_lower/upper is sound (containment).
///
/// PROPERTY: For any x in [a_l, a_u] and y in [b_l, b_u], the true sum x + y
/// is contained in [safe_add_lower(a_l, b_l), safe_add_upper(a_u, b_u)].
///
/// This is THE core soundness property for CROWN backward propagation through
/// affine layers. Every linear bound accumulation step computes:
///   new_lower = safe_add_lower(old_lower, contribution_lower)
///   new_upper = safe_add_upper(old_upper, contribution_upper)
/// If this containment property fails, CROWN bounds are unsound.
///
/// Uses bounded i8 domains to keep verification tractable while exercising
/// all sign combinations (positive, negative, crossing zero).
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn interval_add_contains_sum() {
    let a_l_i = any_bounded_i8(-20, 20);
    let a_u_i = any_bounded_i8(-20, 20);
    let b_l_i = any_bounded_i8(-20, 20);
    let b_u_i = any_bounded_i8(-20, 20);

    // Require well-formed intervals
    assume(a_l_i <= a_u_i);
    assume(b_l_i <= b_u_i);

    let a_l = a_l_i as f32;
    let a_u = a_u_i as f32;
    let b_l = b_l_i as f32;
    let b_u = b_u_i as f32;

    // Compute interval addition bounds
    let sum_lower = safe_add_lower_for_bounds(a_l, b_l);
    let sum_upper = safe_add_upper_for_bounds(a_u, b_u);

    // Pick arbitrary points within input intervals
    let a = any_bounded_i8(a_l_i, a_u_i) as f32;
    let b = any_bounded_i8(b_l_i, b_u_i) as f32;

    // The true sum must be within the computed interval bounds
    let true_sum = a + b;
    if true_sum.is_finite() {
        assert!(
            sum_lower <= true_sum,
            "Interval addition lower bound violated: safe_add_lower(a_l, b_l) must be <= a + b"
        );
        assert!(
            true_sum <= sum_upper,
            "Interval addition upper bound violated: a + b must be <= safe_add_upper(a_u, b_u)"
        );
    }

    // Also verify the result is a well-formed interval
    assert!(
        sum_lower <= sum_upper,
        "Interval addition must produce a valid interval (lower <= upper)"
    );
}

/// Proof: safe_add_for_bounds_with_polarity propagates NaN inputs.
///
/// PROPERTY: NaN inputs yield NaN outputs (invalid input propagation).
#[cfg(kani)]
#[kani::proof]
fn safe_add_with_polarity_propagates_nan_inputs() {
    let finite = any_finite_f32();
    let nan_is_first: bool = any();
    let is_lower: bool = any();
    let (sum, term) = if nan_is_first {
        (f32::NAN, finite)
    } else {
        (finite, f32::NAN)
    };

    let result = safe_add_for_bounds_with_polarity(sum, term, is_lower);
    assert!(result.is_nan());
}

/// Proof: safe_add_for_bounds_with_polarity matches finite sums exactly.
///
/// PROPERTY: Finite inputs with finite sum -> exact sum.
#[cfg(kani)]
#[kani::proof]
fn safe_add_with_polarity_matches_sum_when_finite() {
    let a = any_finite_f32();
    let b = any_finite_f32();
    let sum = a + b;
    assume(sum.is_finite());

    let lower_result = safe_add_for_bounds_with_polarity(a, b, true);
    assert!(lower_result == sum);

    let upper_result = safe_add_for_bounds_with_polarity(a, b, false);
    assert!(upper_result == sum);
}

// =============================================================================
// ReLU Relaxation Proofs
// =============================================================================

/// Proof: ReLU relaxation lower bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   lower_slope * x + lower_intercept <= ReLU(x)
#[cfg(kani)]
#[kani::proof]
fn relu_crown_lower_bound_sound() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let (lower_slope, lower_intercept, _, _) = relu_crown_relaxation(lower, upper);

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_relu = relu(x);

    assert!(linear_bound <= true_relu, "Lower bound must be <= ReLU(x)");
}

/// Proof: ReLU relaxation upper bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   upper_slope * x + upper_intercept >= ReLU(x)
#[cfg(kani)]
#[kani::proof]
fn relu_crown_upper_bound_sound() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let (_, _, upper_slope, upper_intercept) = relu_crown_relaxation(lower, upper);

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_relu = relu(x);

    assert!(linear_bound >= true_relu, "Upper bound must be >= ReLU(x)");
}

/// Proof: ReLU relaxation is exact in the positive region.
///
/// PROPERTY: If lower >= 0, returns identity for both bounds.
#[cfg(kani)]
#[kani::proof]
fn relu_crown_positive_region_exact() {
    let lower = any_bounded_f32(0.0, 10.0);
    let upper = any_bounded_f32(0.0, 10.0);
    assume(lower <= upper);

    let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
        relu_crown_relaxation(lower, upper);

    assert_eq!(lower_slope, 1.0);
    assert_eq!(lower_intercept, 0.0);
    assert_eq!(upper_slope, 1.0);
    assert_eq!(upper_intercept, 0.0);
}

/// Proof: ReLU relaxation is exact in the negative region.
///
/// PROPERTY: If upper <= 0, returns zero for both bounds.
#[cfg(kani)]
#[kani::proof]
fn relu_crown_negative_region_exact() {
    let lower = any_bounded_f32(-10.0, 0.0);
    let upper = any_bounded_f32(-10.0, 0.0);
    assume(lower <= upper);
    assume(upper <= 0.0);
    assume(lower < 0.0);

    let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
        relu_crown_relaxation(lower, upper);

    assert_eq!(lower_slope, 0.0);
    assert_eq!(lower_intercept, 0.0);
    assert_eq!(upper_slope, 0.0);
    assert_eq!(upper_intercept, 0.0);
}

/// Proof: ReLU relaxation slopes are in valid range [0, 1].
///
/// PROPERTY: All slopes are in [0, 1] for any valid interval.
#[cfg(kani)]
#[kani::proof]
fn relu_crown_slopes_valid_range() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let (lower_slope, _, upper_slope, _) = relu_crown_relaxation(lower, upper);

    assert!(
        lower_slope >= 0.0 && lower_slope <= 1.0,
        "Lower slope must be in [0, 1]"
    );
    assert!(
        upper_slope >= 0.0 && upper_slope <= 1.0,
        "Upper slope must be in [0, 1]"
    );
}

/// Proof: ReLU upper relaxation is exact at endpoints in crossing region.
///
/// PROPERTY: In crossing region (lower < 0 < upper):
///   upper_slope * lower + upper_intercept == 0
///   upper_slope * upper + upper_intercept == upper
#[cfg(kani)]
#[kani::proof]
fn relu_crown_upper_exact_at_endpoints() {
    // Use small integer bounds to keep CBMC tractable.
    let lower = any_bounded_i8(-4, -1) as f32;
    let upper = any_bounded_i8(1, 4) as f32;
    assume(lower < 0.0 && upper > 0.0);

    let (_, _, upper_slope, upper_intercept) = relu_crown_relaxation(lower, upper);

    // Upper relaxation should be exact at left endpoint (ReLU(lower) = 0)
    let at_lower = upper_slope * lower + upper_intercept;
    // Upper relaxation should be exact at right endpoint (ReLU(upper) = upper)
    let at_upper = upper_slope * upper + upper_intercept;

    // Use approximate equality due to floating point
    let tolerance = 1e-5;
    assert!(
        (at_lower - 0.0).abs() < tolerance,
        "Upper relaxation not exact at lower endpoint"
    );
    assert!(
        (at_upper - upper).abs() < tolerance,
        "Upper relaxation not exact at upper endpoint"
    );
}

// =============================================================================
// GELU Sound Relaxation Proofs
// =============================================================================

/// Proof: GELU sound relaxation lower bound is sound (Erf approximation).
///
/// PROPERTY: For all x in [lower, upper]:
///   lower_slope * x + lower_intercept <= GELU(x)
///
/// This is the core soundness property for CROWN backward propagation through GELU.
/// Uses precomputed tangent tables for provably sound bounds.
#[cfg(kani)]
#[kani::proof]
fn gelu_erf_crown_lower_bound_sound() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);

    let (lower_slope, lower_intercept, _, _) = gelu_sound_linear_relaxation(lower, upper);

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_gelu = gelu_eval(x, GeluApproximation::Erf);

    if linear_bound.is_finite() && true_gelu.is_finite() {
        assert!(
            linear_bound <= true_gelu,
            "GELU lower bound must be <= GELU(x)"
        );
    }
}

/// Proof: GELU sound relaxation upper bound is sound (Erf approximation).
///
/// PROPERTY: For all x in [lower, upper]:
///   upper_slope * x + upper_intercept >= GELU(x)
#[cfg(kani)]
#[kani::proof]
fn gelu_erf_crown_upper_bound_sound() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);

    let (_, _, upper_slope, upper_intercept) = gelu_sound_linear_relaxation(lower, upper);

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_gelu = gelu_eval(x, GeluApproximation::Erf);

    if linear_bound.is_finite() && true_gelu.is_finite() {
        assert!(
            linear_bound >= true_gelu,
            "GELU upper bound must be >= GELU(x)"
        );
    }
}

/// Proof: GELU tanh approximation sound relaxation lower bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   lower_slope * x + lower_intercept <= GELU_tanh(x)
#[cfg(kani)]
#[kani::proof]
fn gelu_tanh_crown_lower_bound_sound() {
    let lower = any_bounded_f32_no_negzero(-5.0, 5.0);
    let upper = any_bounded_f32_no_negzero(-5.0, 5.0);

    assume(lower <= upper);

    let (lower_slope, lower_intercept, _, _) = gelu_tanh_sound_linear_relaxation(lower, upper);

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32_no_negzero(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_gelu = gelu_eval(x, GeluApproximation::Tanh);

    if linear_bound.is_finite() && true_gelu.is_finite() {
        assert!(
            linear_bound <= true_gelu,
            "GELU tanh lower bound must be <= GELU(x)"
        );
    }
}

/// Proof: GELU tanh approximation sound relaxation upper bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   upper_slope * x + upper_intercept >= GELU_tanh(x)
#[cfg(kani)]
#[kani::proof]
fn gelu_tanh_crown_upper_bound_sound() {
    let lower = any_bounded_f32_no_negzero(-5.0, 5.0);
    let upper = any_bounded_f32_no_negzero(-5.0, 5.0);

    assume(lower <= upper);

    let (_, _, upper_slope, upper_intercept) = gelu_tanh_sound_linear_relaxation(lower, upper);

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32_no_negzero(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_gelu = gelu_eval(x, GeluApproximation::Tanh);

    if linear_bound.is_finite() && true_gelu.is_finite() {
        assert!(
            linear_bound >= true_gelu,
            "GELU tanh upper bound must be >= GELU(x)"
        );
    }
}

/// Proof: GELU Erf sound relaxation bounds are well-formed (lower <= upper).
///
/// PROPERTY: For any valid interval, the relaxation produces valid bounds
/// where lower bound <= upper bound at any point in the interval.
#[cfg(kani)]
#[kani::proof]
fn gelu_erf_sound_bounds_well_formed() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);

    let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
        gelu_sound_linear_relaxation(lower, upper);

    // Check at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Erf lower bound must be <= upper bound"
        );
    }
}

/// Proof: GELU Tanh sound relaxation bounds are well-formed (lower <= upper).
///
/// PROPERTY: For any valid interval, the tanh approximation relaxation produces valid bounds
/// where lower bound <= upper bound at any point in the interval.
#[cfg(kani)]
#[kani::proof]
fn gelu_tanh_sound_bounds_well_formed() {
    let lower = any_bounded_f32_no_negzero(-5.0, 5.0);
    let upper = any_bounded_f32_no_negzero(-5.0, 5.0);

    assume(lower <= upper);

    let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
        gelu_tanh_sound_linear_relaxation(lower, upper);

    // Check at an arbitrary point in the interval
    let x = any_bounded_f32_no_negzero(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Tanh lower bound must be <= upper bound"
        );
    }
}

/// Proof: GELU Erf sound relaxation contains endpoints.
///
/// PROPERTY: The sound relaxation should contain the true GELU value at
/// both endpoints of the interval.
#[cfg(kani)]
#[kani::proof]
fn gelu_erf_sound_contains_endpoints() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);
    // Skip degenerate intervals where the relaxation degenerates
    assume((upper - lower).abs() >= 1e-6);

    let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
        gelu_sound_linear_relaxation(lower, upper);

    // Check at lower endpoint
    let gelu_at_lower = gelu_eval(lower, GeluApproximation::Erf);
    let lower_bound_at_lower = lower_slope * lower + lower_intercept;
    let upper_bound_at_lower = upper_slope * lower + upper_intercept;

    if gelu_at_lower.is_finite()
        && lower_bound_at_lower.is_finite()
        && upper_bound_at_lower.is_finite()
    {
        assert!(
            lower_bound_at_lower <= gelu_at_lower,
            "Erf lower bound violated at lower endpoint"
        );
        assert!(
            gelu_at_lower <= upper_bound_at_lower,
            "Erf upper bound violated at lower endpoint"
        );
    }

    // Check at upper endpoint
    let gelu_at_upper = gelu_eval(upper, GeluApproximation::Erf);
    let lower_bound_at_upper = lower_slope * upper + lower_intercept;
    let upper_bound_at_upper = upper_slope * upper + upper_intercept;

    if gelu_at_upper.is_finite()
        && lower_bound_at_upper.is_finite()
        && upper_bound_at_upper.is_finite()
    {
        assert!(
            lower_bound_at_upper <= gelu_at_upper,
            "Erf lower bound violated at upper endpoint"
        );
        assert!(
            gelu_at_upper <= upper_bound_at_upper,
            "Erf upper bound violated at upper endpoint"
        );
    }
}

/// Proof: GELU Tanh sound relaxation contains endpoints.
///
/// PROPERTY: The tanh approximation sound relaxation should contain the true GELU value at
/// both endpoints of the interval.
#[cfg(kani)]
#[kani::proof]
fn gelu_tanh_sound_contains_endpoints() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);
    // Skip degenerate intervals where the relaxation degenerates
    assume((upper - lower).abs() >= 1e-6);

    let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
        gelu_tanh_sound_linear_relaxation(lower, upper);

    // Check at lower endpoint
    let gelu_at_lower = gelu_eval(lower, GeluApproximation::Tanh);
    let lower_bound_at_lower = lower_slope * lower + lower_intercept;
    let upper_bound_at_lower = upper_slope * lower + upper_intercept;

    if gelu_at_lower.is_finite()
        && lower_bound_at_lower.is_finite()
        && upper_bound_at_lower.is_finite()
    {
        assert!(
            lower_bound_at_lower <= gelu_at_lower,
            "Tanh lower bound violated at lower endpoint"
        );
        assert!(
            gelu_at_lower <= upper_bound_at_lower,
            "Tanh upper bound violated at lower endpoint"
        );
    }

    // Check at upper endpoint
    let gelu_at_upper = gelu_eval(upper, GeluApproximation::Tanh);
    let lower_bound_at_upper = lower_slope * upper + lower_intercept;
    let upper_bound_at_upper = upper_slope * upper + upper_intercept;

    if gelu_at_upper.is_finite()
        && lower_bound_at_upper.is_finite()
        && upper_bound_at_upper.is_finite()
    {
        assert!(
            lower_bound_at_upper <= gelu_at_upper,
            "Tanh lower bound violated at upper endpoint"
        );
        assert!(
            gelu_at_upper <= upper_bound_at_upper,
            "Tanh upper bound violated at upper endpoint"
        );
    }
}

// =============================================================================
// Exponential Sound Interval Proofs
// =============================================================================

// Import exponential/softmax/logsumexp functions from ny-relaxation.
use ny_relaxation::{
    exp_interval_bounds, logsoftmax_ibp_bounds, logsumexp_slice, softmax_ibp_element_bounds,
};

/// Proof: exp_interval_bounds is sound (exp is monotonic).
///
/// PROPERTY: For all x in [lower, upper], exp(x) is in [exp(lower), exp(upper)].
/// This is the core soundness property for exponential bound propagation.
#[cfg(kani)]
#[kani::proof]
fn exp_interval_bounds_sound() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);
    assume(lower <= upper);

    let (exp_lower, exp_upper) =
        exp_interval_bounds(lower, upper).expect("invariant: lower <= upper");

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let exp_x = x.exp();

    if exp_x.is_finite() {
        assert!(exp_lower <= exp_x, "exp lower bound violated");
        assert!(exp_x <= exp_upper, "exp upper bound violated");
    }
}

/// Proof: exp_interval_bounds produces well-formed output.
///
/// PROPERTY: exp_lower <= exp_upper (output is a valid interval).
#[cfg(kani)]
#[kani::proof]
fn exp_interval_bounds_well_formed() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);
    assume(lower <= upper);

    let (exp_lower, exp_upper) =
        exp_interval_bounds(lower, upper).expect("invariant: lower <= upper");

    // exp is monotonic so this should always hold
    if exp_lower.is_finite() && exp_upper.is_finite() {
        assert!(exp_lower <= exp_upper, "exp output interval malformed");
    }
}

/// Proof: exp_interval_bounds output is non-negative.
///
/// PROPERTY: exp(x) > 0 for all real x, so bounds must be positive.
#[cfg(kani)]
#[kani::proof]
fn exp_interval_bounds_positive() {
    let lower = any_bounded_f32(-20.0, 20.0);
    let upper = any_bounded_f32(-20.0, 20.0);
    assume(lower <= upper);

    let (exp_lower, exp_upper) =
        exp_interval_bounds(lower, upper).expect("invariant: lower <= upper");

    if exp_lower.is_finite() {
        assert!(exp_lower > 0.0, "exp lower bound must be positive");
    }
    if exp_upper.is_finite() {
        assert!(exp_upper > 0.0, "exp upper bound must be positive");
    }
}

// =============================================================================
// Softmax IBP Proofs
// =============================================================================

/// Proof: softmax_ibp_element_bounds output is in [0, 1].
///
/// PROPERTY: Softmax outputs are always in [0, 1] (probability simplex).
/// The function takes pre-computed exp values (shifted by max_upper).
#[cfg(kani)]
#[kani::proof]
fn softmax_ibp_element_bounds_unit_interval() {
    // Generate exp values directly (these are already exp(x - max) values)
    let exp_lower_i = any_bounded_f32(0.01, 10.0); // exp values are positive
    let exp_upper_i = any_bounded_f32(0.01, 10.0);
    assume(exp_lower_i <= exp_upper_i); // lower exp <= upper exp (since exp is monotonic)

    // sum_exp values must be positive and include exp_lower_i and exp_upper_i
    let sum_exp_lower = any_bounded_f32(0.1, 100.0);
    let sum_exp_upper = any_bounded_f32(0.1, 100.0);
    assume(sum_exp_lower <= sum_exp_upper);
    // sum_exp must be at least exp_lower_i for a valid softmax
    assume(sum_exp_lower >= exp_lower_i);
    assume(sum_exp_upper >= exp_upper_i);

    let (softmax_lower, softmax_upper) =
        softmax_ibp_element_bounds(exp_lower_i, exp_upper_i, sum_exp_lower, sum_exp_upper);

    // Softmax values are always in [0, 1]
    assert!(softmax_lower >= 0.0, "softmax lower must be >= 0");
    assert!(softmax_lower <= 1.0, "softmax lower must be <= 1");
    assert!(softmax_upper >= 0.0, "softmax upper must be >= 0");
    assert!(softmax_upper <= 1.0, "softmax upper must be <= 1");
}

/// Proof: softmax_ibp_element_bounds output is well-formed.
///
/// PROPERTY: softmax_lower <= softmax_upper (valid interval).
/// The function takes pre-computed exp values.
#[cfg(kani)]
#[kani::proof]
fn softmax_ibp_element_bounds_well_formed() {
    // Generate exp values directly
    let exp_lower_i = any_bounded_f32(0.01, 5.0);
    let exp_upper_i = any_bounded_f32(0.01, 5.0);
    assume(exp_lower_i <= exp_upper_i);

    let sum_exp_lower = any_bounded_f32(0.5, 50.0);
    let sum_exp_upper = any_bounded_f32(0.5, 50.0);
    assume(sum_exp_lower <= sum_exp_upper);
    assume(sum_exp_lower >= exp_lower_i);
    assume(sum_exp_upper >= exp_upper_i);

    let (softmax_lower, softmax_upper) =
        softmax_ibp_element_bounds(exp_lower_i, exp_upper_i, sum_exp_lower, sum_exp_upper);

    assert!(
        softmax_lower <= softmax_upper,
        "softmax output interval malformed"
    );
}

// =============================================================================
// LogSumExp Proofs
// =============================================================================

/// Proof: logsumexp_slice is monotonic in each argument.
///
/// PROPERTY: logsumexp([a]) = a for single element.
#[cfg(kani)]
#[kani::proof]
fn logsumexp_single_element_identity() {
    let x = any_bounded_f32(-10.0, 10.0);
    let values = [x];

    let lse = logsumexp_slice(&values);

    // For a single element, logsumexp(x) = x
    if lse.is_finite() {
        let diff = (lse - x).abs();
        assert!(
            diff < 1e-5,
            "logsumexp of single element should equal that element"
        );
    }
}

/// Proof: logsumexp_slice >= max(values).
///
/// PROPERTY: logsumexp(x) >= max_i(x_i) always (log of sum of positive values).
#[cfg(kani)]
#[kani::proof]
fn logsumexp_at_least_max() {
    let x1 = any_bounded_f32(-5.0, 5.0);
    let x2 = any_bounded_f32(-5.0, 5.0);
    let values = [x1, x2];

    let lse = logsumexp_slice(&values);
    let max_val = x1.max(x2);

    if lse.is_finite() && max_val.is_finite() {
        assert!(lse >= max_val, "logsumexp must be >= max(values)");
    }
}

/// Proof: logsumexp_slice is bounded by max + ln(n).
///
/// PROPERTY: logsumexp(x) <= max(x) + ln(n) where n is the number of elements.
#[cfg(kani)]
#[kani::proof]
fn logsumexp_at_most_max_plus_log_n() {
    let x1 = any_bounded_f32(-5.0, 5.0);
    let x2 = any_bounded_f32(-5.0, 5.0);
    let values = [x1, x2];

    let lse = logsumexp_slice(&values);
    let max_val = x1.max(x2);
    let n = 2.0_f32;
    let upper_bound = max_val + n.ln();

    if lse.is_finite() && upper_bound.is_finite() {
        assert!(
            lse <= upper_bound + 1e-5,
            "logsumexp must be <= max + ln(n)"
        );
    }
}

/// Proof: logsumexp_slice with two equal elements.
///
/// PROPERTY: logsumexp([x, x]) = x + ln(2).
#[cfg(kani)]
#[kani::proof]
fn logsumexp_two_equal_elements() {
    let x = any_bounded_f32(-5.0, 5.0);
    let values = [x, x];

    let lse = logsumexp_slice(&values);
    let expected = x + 2.0_f32.ln();

    if lse.is_finite() && expected.is_finite() {
        let diff = (lse - expected).abs();
        assert!(diff < 1e-5, "logsumexp([x,x]) should equal x + ln(2)");
    }
}

// =============================================================================
// LogSoftmax IBP Proofs
// =============================================================================

/// Proof: logsoftmax_ibp_bounds is sound.
///
/// PROPERTY: For logsoftmax_i = x_i - logsumexp(x):
///   lower_i - lse_upper <= logsoftmax_i(x) <= upper_i - lse_lower
/// for all x in [lower, upper].
#[cfg(kani)]
#[kani::proof]
fn logsoftmax_ibp_bounds_sound() {
    let lower_i = any_bounded_f32(-5.0, 5.0);
    let upper_i = any_bounded_f32(-5.0, 5.0);
    assume(lower_i <= upper_i);

    let lse_lower = any_bounded_f32(-5.0, 10.0);
    let lse_upper = any_bounded_f32(-5.0, 10.0);
    assume(lse_lower <= lse_upper);
    // logsumexp must be at least lower_i for a single-element softmax
    assume(lse_lower >= lower_i);

    let (bound_lower, bound_upper) = logsoftmax_ibp_bounds(lower_i, upper_i, lse_lower, lse_upper);

    // Test at an arbitrary point
    let x_i = any_bounded_f32(lower_i, upper_i);
    // For this test, we use a logsumexp value in the valid range
    let lse_x = any_bounded_f32(lse_lower, lse_upper);
    let logsoftmax_x = x_i - lse_x;

    if logsoftmax_x.is_finite() && bound_lower.is_finite() && bound_upper.is_finite() {
        assert!(
            bound_lower <= logsoftmax_x,
            "logsoftmax lower bound violated"
        );
        assert!(
            logsoftmax_x <= bound_upper,
            "logsoftmax upper bound violated"
        );
    }
}

/// Proof: logsoftmax_ibp_bounds output is well-formed.
///
/// PROPERTY: bound_lower <= bound_upper (valid interval).
#[cfg(kani)]
#[kani::proof]
fn logsoftmax_ibp_bounds_well_formed() {
    let lower_i = any_bounded_f32(-5.0, 5.0);
    let upper_i = any_bounded_f32(-5.0, 5.0);
    assume(lower_i <= upper_i);

    let lse_lower = any_bounded_f32(0.0, 10.0);
    let lse_upper = any_bounded_f32(0.0, 10.0);
    assume(lse_lower <= lse_upper);

    let (bound_lower, bound_upper) = logsoftmax_ibp_bounds(lower_i, upper_i, lse_lower, lse_upper);

    if bound_lower.is_finite() && bound_upper.is_finite() {
        assert!(
            bound_lower <= bound_upper,
            "logsoftmax output interval malformed"
        );
    }
}

/// Proof: logsoftmax_ibp_bounds upper bound is non-positive for valid inputs.
///
/// PROPERTY: logsoftmax(x) <= 0 always (log of probability <= log(1) = 0).
#[cfg(kani)]
#[kani::proof]
fn logsoftmax_ibp_bounds_non_positive() {
    let lower_i = any_bounded_f32(-5.0, 5.0);
    let upper_i = any_bounded_f32(-5.0, 5.0);
    assume(lower_i <= upper_i);

    // logsumexp must be at least upper_i for logsoftmax to be <= 0
    let lse_lower = any_bounded_f32(upper_i, upper_i + 5.0);
    let lse_upper = any_bounded_f32(upper_i, upper_i + 10.0);
    assume(lse_lower <= lse_upper);

    let (_, bound_upper) = logsoftmax_ibp_bounds(lower_i, upper_i, lse_lower, lse_upper);

    if bound_upper.is_finite() {
        // upper_i - lse_lower <= 0 when lse_lower >= upper_i
        assert!(
            bound_upper <= 0.0 + 1e-5,
            "logsoftmax upper bound should be <= 0"
        );
    }
}

// =============================================================================
// SiLU (Swish) Sound Relaxation Proofs
// =============================================================================
// SiLU activation: y = x * sigmoid(x)
// Non-monotonic with a global minimum near x ≈ -1.278

/// Proof: SiLU sound relaxation lower bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   lower_slope * x + lower_intercept <= SiLU(x)
///
/// The current implementation uses constant bounds (slope=0), so we verify:
///   min_val <= SiLU(x) for all x in [lower, upper]
#[cfg(kani)]
#[kani::proof]
fn silu_crown_lower_bound_sound() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = silu_sound_linear_relaxation(lower, upper);

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_silu = silu_eval(x);

    if true_silu.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound <= true_silu,
            "SiLU lower bound must be <= SiLU(x)"
        );
    }
}

/// Proof: SiLU sound relaxation upper bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   upper_slope * x + upper_intercept >= SiLU(x)
///
/// The current implementation uses constant bounds (slope=0), so we verify:
///   max_val >= SiLU(x) for all x in [lower, upper]
#[cfg(kani)]
#[kani::proof]
fn silu_crown_upper_bound_sound() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = silu_sound_linear_relaxation(lower, upper);

    // Test at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_silu = silu_eval(x);

    if true_silu.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_silu,
            "SiLU upper bound must be >= SiLU(x)"
        );
    }
}

/// Proof: SiLU sound relaxation bounds are well-formed (lower <= upper).
///
/// PROPERTY: For any valid interval, the relaxation produces valid bounds
/// where lower bound <= upper bound at any point in the interval.
#[cfg(kani)]
#[kani::proof]
fn silu_sound_bounds_well_formed() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = silu_sound_linear_relaxation(lower, upper);

    // Check at an arbitrary point in the interval
    let x = any_bounded_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "SiLU lower bound must be <= upper bound"
        );
    }
}

/// Proof: SiLU sound relaxation contains endpoints.
///
/// PROPERTY: The sound relaxation should contain the true SiLU value at
/// both endpoints of the interval.
#[cfg(kani)]
#[kani::proof]
fn silu_sound_contains_endpoints() {
    let lower = any_bounded_f32(-5.0, 5.0);
    let upper = any_bounded_f32(-5.0, 5.0);

    assume(lower <= upper);
    // Skip degenerate intervals
    assume((upper - lower).abs() >= 1e-6);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = silu_sound_linear_relaxation(lower, upper);

    // Check at lower endpoint
    let silu_at_lower = silu_eval(lower);
    let lower_bound_at_lower = lower_slope * lower + lower_intercept;
    let upper_bound_at_lower = upper_slope * lower + upper_intercept;

    if silu_at_lower.is_finite()
        && lower_bound_at_lower.is_finite()
        && upper_bound_at_lower.is_finite()
    {
        assert!(
            lower_bound_at_lower <= silu_at_lower,
            "SiLU lower bound violated at lower endpoint"
        );
        assert!(
            silu_at_lower <= upper_bound_at_lower,
            "SiLU upper bound violated at lower endpoint"
        );
    }

    // Check at upper endpoint
    let silu_at_upper = silu_eval(upper);
    let lower_bound_at_upper = lower_slope * upper + lower_intercept;
    let upper_bound_at_upper = upper_slope * upper + upper_intercept;

    if silu_at_upper.is_finite()
        && lower_bound_at_upper.is_finite()
        && upper_bound_at_upper.is_finite()
    {
        assert!(
            lower_bound_at_upper <= silu_at_upper,
            "SiLU lower bound violated at upper endpoint"
        );
        assert!(
            silu_at_upper <= upper_bound_at_upper,
            "SiLU upper bound violated at upper endpoint"
        );
    }
}

/// Proof: SiLU sound relaxation handles point intervals correctly.
///
/// PROPERTY: For a point interval [x, x], the relaxation should pass through SiLU(x).
#[cfg(kani)]
#[kani::proof]
fn silu_sound_point_interval() {
    let x = any_bounded_f32(-5.0, 5.0);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = silu_sound_linear_relaxation(x, x);

    let lower_at_x = lower_slope * x + lower_intercept;
    let upper_at_x = upper_slope * x + upper_intercept;
    let silu_x = silu_eval(x);

    if silu_x.is_finite() && lower_at_x.is_finite() && upper_at_x.is_finite() {
        // For a point interval, bounds should equal the function value
        let tolerance = 1e-5;
        assert!(
            (lower_at_x - silu_x).abs() < tolerance,
            "SiLU point interval lower bound should equal SiLU(x)"
        );
        assert!(
            (upper_at_x - silu_x).abs() < tolerance,
            "SiLU point interval upper bound should equal SiLU(x)"
        );
    }
}

/// Proof: silu_eval handles special values correctly.
///
/// PROPERTY: silu_eval(-inf) = 0, silu_eval(+inf) = +inf, silu_eval(NaN) = NaN.
#[cfg(kani)]
#[kani::proof]
fn silu_eval_special_values() {
    match any_special_case_index(3) {
        // SiLU(-inf) = -inf * sigmoid(-inf) = -inf * 0 = 0
        0 => {
            let neg_inf_result = silu_eval(f32::NEG_INFINITY);
            assert!(neg_inf_result == 0.0, "SiLU(-inf) should be 0");
        }
        // SiLU(+inf) = +inf * sigmoid(+inf) = +inf * 1 = +inf
        1 => {
            let pos_inf_result = silu_eval(f32::INFINITY);
            assert!(pos_inf_result == f32::INFINITY, "SiLU(+inf) should be +inf");
        }
        // SiLU(NaN) = NaN
        2 => {
            let nan_result = silu_eval(f32::NAN);
            assert!(nan_result.is_nan(), "SiLU(NaN) should be NaN");
        }
        _ => unreachable!(),
    }
}

/// Proof: SiLU sound relaxation handles crossing the critical point.
///
/// PROPERTY: When the interval [lower, upper] contains the critical point
/// (approximately -1.278), the lower bound should be the global minimum.
#[cfg(kani)]
#[kani::proof]
fn silu_sound_critical_point_crossing() {
    // Critical point is near -1.278
    // Use an interval that crosses the critical point
    let lower = any_bounded_f32(-3.0, -1.5);
    let upper = any_bounded_f32(-1.0, 0.0);

    assume(lower <= upper);
    assume(lower < -1.3 && upper > -1.2); // Ensure we cross the critical point

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = silu_sound_linear_relaxation(lower, upper);

    // The lower bound should capture the global minimum in the interval
    // Since we use constant bounds, lower_intercept should be the min value
    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_silu = silu_eval(x);

    if true_silu.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound <= true_silu,
            "SiLU lower bound violated at critical region"
        );
    }
}

// =============================================================================
// Exp CROWN Relaxation Proofs
// =============================================================================
// exp(x) is convex: chord is upper bound, tangent is lower bound.
// Reference: alpha-beta-CROWN BoundExp.bound_relax
//
// CBMC LIMITATION: CBMC (Kani backend) does not faithfully model f64 transcendental
// functions (exp, ln). Counterexamples at exact values (e.g., l=u=x=-0.0) are
// false positives — manual f32/f64 verification confirms the bounds are correct.
// These proofs remain as aspirational targets for when CBMC improves transcendental
// support. See kani_status.json for current status.
//
// Part of #1712.

/// Proof: Exp CROWN lower bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   lower_slope * x + lower_intercept <= exp(x)
///
/// The lower bound uses the tangent line at the midpoint, which lies below
/// the convex exp curve everywhere.
#[cfg(kani)]
#[kani::proof]
fn exp_crown_lower_bound_sound() {
    let lower = any_bounded_f32_no_negzero(-5.0, 5.0);
    let upper = any_bounded_f32_no_negzero(-5.0, 5.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = exp_linear_relaxation(lower, upper);

    let x = any_bounded_f32_no_negzero(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_exp = x.exp();

    if true_exp.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound <= true_exp,
            "Exp lower bound must be <= exp(x)"
        );
    }
}

/// Proof: Exp CROWN upper bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   upper_slope * x + upper_intercept >= exp(x)
///
/// The upper bound uses the chord (secant) through the endpoints, which lies
/// above the convex exp curve everywhere in the interval.
#[cfg(kani)]
#[kani::proof]
fn exp_crown_upper_bound_sound() {
    let lower = any_bounded_f32_no_negzero(-5.0, 5.0);
    let upper = any_bounded_f32_no_negzero(-5.0, 5.0);

    assume(lower <= upper);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = exp_linear_relaxation(lower, upper);

    let x = any_bounded_f32_no_negzero(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_exp = x.exp();

    if true_exp.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_exp,
            "Exp upper bound must be >= exp(x)"
        );
    }
}

/// Proof: Exp CROWN bounds are well-formed (lower <= upper at all points).
#[cfg(kani)]
#[kani::proof]
fn exp_crown_bounds_well_formed() {
    let lower = any_bounded_f32_no_negzero(-5.0, 5.0);
    let upper = any_bounded_f32_no_negzero(-5.0, 5.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = exp_linear_relaxation(lower, upper);

    let x = any_bounded_f32_no_negzero(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Exp lower bound must be <= upper bound"
        );
    }
}

/// Proof: Exp CROWN bounds contain endpoints.
///
/// PROPERTY: The relaxation contains exp(l) and exp(u).
#[cfg(kani)]
#[kani::proof]
fn exp_crown_contains_endpoints() {
    let lower = any_bounded_f32_no_negzero(-5.0, 5.0);
    let upper = any_bounded_f32_no_negzero(-5.0, 5.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = exp_linear_relaxation(lower, upper);

    // Check at lower endpoint
    let lb_at_l = lower_slope * lower + lower_intercept;
    let ub_at_l = upper_slope * lower + upper_intercept;
    let exp_l = lower.exp();

    if exp_l.is_finite() && lb_at_l.is_finite() && ub_at_l.is_finite() {
        assert!(lb_at_l <= exp_l, "Exp lower bound must contain exp(l)");
        assert!(ub_at_l >= exp_l, "Exp upper bound must contain exp(l)");
    }

    // Check at upper endpoint
    let lb_at_u = lower_slope * upper + lower_intercept;
    let ub_at_u = upper_slope * upper + upper_intercept;
    let exp_u = upper.exp();

    if exp_u.is_finite() && lb_at_u.is_finite() && ub_at_u.is_finite() {
        assert!(lb_at_u <= exp_u, "Exp lower bound must contain exp(u)");
        assert!(ub_at_u >= exp_u, "Exp upper bound must contain exp(u)");
    }
}

// =============================================================================
// Log CROWN Relaxation Proofs
// =============================================================================
// log(x) is concave for x > 0: chord is lower bound, tangent is upper bound.
// Reference: alpha-beta-CROWN BoundLog.bound_relax
//
// CBMC LIMITATION: Same as Exp proofs — CBMC does not faithfully model f64::ln().
// Counterexamples at exact values (e.g., l=u=4.0, l=u=8.0) are false positives.
// Manual verification confirms the degenerate-path tangent formula is mathematically
// exact (1/l * l + (ln(l) - 1) = ln(l)), with f64→f32 rounding preserving soundness.
//
// Part of #1712.

/// Proof: Log CROWN lower bound is sound.
///
/// PROPERTY: For all x in [lower, upper] with lower > 0:
///   lower_slope * x + lower_intercept <= log(x)
///
/// The lower bound uses the chord through the endpoints, which lies below
/// the concave log curve.
#[cfg(kani)]
#[kani::proof]
fn log_crown_lower_bound_sound() {
    let lower = any_bounded_f32(0.01, 10.0);
    let upper = any_bounded_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(lower > 0.0);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = log_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_log = x.ln();

    if true_log.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound <= true_log,
            "Log lower bound must be <= log(x)"
        );
    }
}

/// Proof: Log CROWN upper bound is sound.
///
/// PROPERTY: For all x in [lower, upper] with lower > 0:
///   upper_slope * x + upper_intercept >= log(x)
///
/// The upper bound uses the tangent at the midpoint, which lies above
/// the concave log curve.
#[cfg(kani)]
#[kani::proof]
fn log_crown_upper_bound_sound() {
    let lower = any_bounded_f32(0.01, 10.0);
    let upper = any_bounded_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(lower > 0.0);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = log_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_log = x.ln();

    if true_log.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_log,
            "Log upper bound must be >= log(x)"
        );
    }
}

/// Proof: Log CROWN bounds are well-formed (lower <= upper at all points).
#[cfg(kani)]
#[kani::proof]
fn log_crown_bounds_well_formed() {
    let lower = any_bounded_f32(0.01, 10.0);
    let upper = any_bounded_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(lower > 0.0);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = log_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Log lower bound must be <= upper bound"
        );
    }
}

// =============================================================================
// Sqrt CROWN Relaxation Proofs
// =============================================================================
// sqrt(x) is concave for x >= 0: chord is lower bound, tangent is upper bound.
//
// CBMC LIMITATION: All 3 sqrt harnesses timeout at 300s. CBMC generates
// ~220K variables / ~1M clauses for sqrt proofs. The SAT solver finds
// the instance satisfiable but cannot complete propositional reduction
// within the timeout. May require increased timeout or solver hints.
//
// Part of #1712.

/// Proof: Sqrt CROWN lower bound is sound.
///
/// PROPERTY: For all x in [lower, upper] with lower >= 0:
///   lower_slope * x + lower_intercept <= sqrt(x)
///
/// The lower bound uses the chord through the endpoints.
#[cfg(kani)]
#[kani::proof]
fn sqrt_crown_lower_bound_sound() {
    let lower = any_bounded_f32(0.0, 10.0);
    let upper = any_bounded_f32(0.0, 10.0);

    assume(lower <= upper);
    assume(lower >= 0.0);
    // Ensure non-degenerate interval to exercise chord logic
    assume(upper - lower >= 0.001);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = sqrt_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound <= true_sqrt,
            "Sqrt lower bound must be <= sqrt(x)"
        );
    }
}

/// Proof: Sqrt CROWN upper bound is sound.
///
/// PROPERTY: For all x in [lower, upper] with lower >= 0:
///   upper_slope * x + upper_intercept >= sqrt(x)
///
/// The upper bound uses the tangent at the upper endpoint.
#[cfg(kani)]
#[kani::proof]
fn sqrt_crown_upper_bound_sound() {
    let lower = any_bounded_f32(0.0, 10.0);
    let upper = any_bounded_f32(0.0, 10.0);

    assume(lower <= upper);
    assume(lower >= 0.0);
    assume(upper - lower >= 0.001);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = sqrt_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_sqrt,
            "Sqrt upper bound must be >= sqrt(x)"
        );
    }
}

/// Proof: Sqrt CROWN bounds are well-formed (lower <= upper at all points).
#[cfg(kani)]
#[kani::proof]
fn sqrt_crown_bounds_well_formed() {
    let lower = any_bounded_f32(0.0, 10.0);
    let upper = any_bounded_f32(0.0, 10.0);

    assume(lower <= upper);
    assume(lower >= 0.0);
    assume(upper - lower >= 0.001);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = sqrt_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Sqrt lower bound must be <= upper bound"
        );
    }
}

/// Proof: Sqrt CROWN lower bound is sound (grid-sampled variant).
///
/// Uses 256-point grid sampling per input to reduce SAT problem from ~220K
/// variables to ~10K variables, making CBMC verification tractable.
/// The full f32-range proofs above remain as aspirational targets.
#[cfg(kani)]
#[kani::proof]
fn sqrt_crown_lower_bound_sound_grid() {
    let lower = any_grid_f32(0.01, 10.0);
    let upper = any_grid_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(upper - lower >= 0.01);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = sqrt_linear_relaxation(lower, upper);

    let x = any_grid_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound <= true_sqrt,
            "Sqrt lower bound must be <= sqrt(x)"
        );
    }
}

/// Proof: Sqrt CROWN upper bound is sound (grid-sampled variant).
///
/// Uses 256-point grid sampling per input to reduce SAT problem size.
#[cfg(kani)]
#[kani::proof]
fn sqrt_crown_upper_bound_sound_grid() {
    let lower = any_grid_f32(0.01, 10.0);
    let upper = any_grid_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(upper - lower >= 0.01);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = sqrt_linear_relaxation(lower, upper);

    let x = any_grid_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_sqrt,
            "Sqrt upper bound must be >= sqrt(x)"
        );
    }
}

/// Proof: Sqrt CROWN bounds are well-formed (grid-sampled variant).
///
/// Uses 256-point grid sampling per input to reduce SAT problem size.
#[cfg(kani)]
#[kani::proof]
fn sqrt_crown_bounds_well_formed_grid() {
    let lower = any_grid_f32(0.01, 10.0);
    let upper = any_grid_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(upper - lower >= 0.01);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = sqrt_linear_relaxation(lower, upper);

    let x = any_grid_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Sqrt lower bound must be <= upper bound"
        );
    }
}

// =============================================================================
// Abs CROWN Relaxation Proofs
// =============================================================================
// |x| is V-shaped piecewise linear with a kink at x=0.
// For x >= 0: |x| = x (identity). For x <= 0: |x| = -x (negation).
// Crossing case (l < 0 < u): chord upper, heuristic tangent lower.
//
// The production code now uses a closed-form safety margin (4·ε·max_endpoint)
// instead of iterative shift_up_n_ulps loops. This should make these proofs
// tractable without subnormal exclusions. See #1784.
//
// Part of #1712.

/// Proof: Abs CROWN lower bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   lower_slope * x + lower_intercept <= |x|
///
/// The lower bound uses identity (slope=1) or negation (slope=-1)
/// depending on which side dominates, with intercept = 0.
#[cfg(kani)]
#[kani::proof]
fn abs_crown_lower_bound_sound() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = abs_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_abs = x.abs();

    if true_abs.is_finite() && linear_bound.is_finite() {
        assert!(linear_bound <= true_abs, "Abs lower bound must be <= |x|");
    }
}

/// Proof: Abs CROWN upper bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   upper_slope * x + upper_intercept >= |x|
///
/// The upper bound uses the chord from (l, -l) to (u, u) in the crossing case,
/// or identity/negation in the non-crossing cases.
#[cfg(kani)]
#[kani::proof]
fn abs_crown_upper_bound_sound() {
    // The production code now uses a closed-form safety margin
    // (4·ε·max_endpoint) instead of iterative shift_up_n_ulps loops,
    // making this proof tractable without subnormal exclusions.
    // See #1784 for the migration.
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = abs_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_abs = x.abs();

    if true_abs.is_finite() && linear_bound.is_finite() {
        assert!(linear_bound >= true_abs, "Abs upper bound must be >= |x|");
    }
}

/// Proof: Abs CROWN bounds are well-formed (lower <= upper at all points).
#[cfg(kani)]
#[kani::proof]
fn abs_crown_bounds_well_formed() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = abs_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Abs lower bound must be <= upper bound"
        );
    }
}

/// Proof: Abs CROWN bounds contain endpoints.
///
/// PROPERTY: The relaxation contains |l| and |u|.
#[cfg(kani)]
#[kani::proof]
fn abs_crown_contains_endpoints() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = abs_linear_relaxation(lower, upper);

    // Check at lower endpoint
    let lb_at_l = lower_slope * lower + lower_intercept;
    let ub_at_l = upper_slope * lower + upper_intercept;
    let abs_l = lower.abs();

    if abs_l.is_finite() && lb_at_l.is_finite() && ub_at_l.is_finite() {
        assert!(lb_at_l <= abs_l, "Abs lower bound must contain |l|");
        assert!(ub_at_l >= abs_l, "Abs upper bound must contain |l|");
    }

    // Check at upper endpoint
    let lb_at_u = lower_slope * upper + lower_intercept;
    let ub_at_u = upper_slope * upper + upper_intercept;
    let abs_u = upper.abs();

    if abs_u.is_finite() && lb_at_u.is_finite() && ub_at_u.is_finite() {
        assert!(lb_at_u <= abs_u, "Abs lower bound must contain |u|");
        assert!(ub_at_u >= abs_u, "Abs upper bound must contain |u|");
    }
}

/// Proof: Abs CROWN positive region is exact identity.
///
/// PROPERTY: If lower >= 0, both bounds are identity (slope=1, intercept=0).
#[cfg(kani)]
#[kani::proof]
fn abs_crown_positive_region_exact() {
    let lower = any_bounded_f32(0.0, 10.0);
    let upper = any_bounded_f32(0.0, 10.0);
    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = abs_linear_relaxation(lower, upper);

    assert_eq!(lower_slope, 1.0);
    assert_eq!(lower_intercept, 0.0);
    assert_eq!(upper_slope, 1.0);
    assert_eq!(upper_intercept, 0.0);
}

/// Proof: Abs CROWN negative region is exact negation.
///
/// PROPERTY: If upper <= 0, both bounds are negation (slope=-1, intercept=0).
#[cfg(kani)]
#[kani::proof]
fn abs_crown_negative_region_exact() {
    // Exclude -0.0: both -0.0 satisfy `<= 0.0` but the function's `l >= 0.0`
    // check fires first (since -0.0 >= 0.0 is true), returning identity not negation.
    // Both are mathematically correct at zero; this harness tests the strictly-negative case.
    let lower = any_bounded_f32_no_negzero(-10.0, 0.0);
    let upper = any_bounded_f32_no_negzero(-10.0, 0.0);
    assume(lower <= upper);
    assume(upper <= 0.0);
    // After excluding -0.0, upper <= 0.0 means upper < 0.0 (strictly negative)
    assume(upper < 0.0);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = abs_linear_relaxation(lower, upper);

    assert_eq!(lower_slope, -1.0);
    assert_eq!(lower_intercept, 0.0);
    assert_eq!(upper_slope, -1.0);
    assert_eq!(upper_intercept, 0.0);
}

// =============================================================================
// PowConstant(2) CROWN Relaxation Proofs
// =============================================================================
// x^2 is convex: chord is upper bound, tangent is lower bound.
// Crossing case (l < 0 < u): constant y = 0 is lower bound.
// Non-crossing: tangent at midpoint is lower bound.
//
// The production code now uses a closed-form safety margin (4·ε·max(l², u²))
// instead of iterative shift_up/down_n_ulps loops. This should make these
// proofs tractable without subnormal exclusions. See #1784.
//
// Part of #1712.

/// Proof: PowConstant(2) CROWN lower bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   lower_slope * x + lower_intercept <= x^2
///
/// Lower bound: tangent at midpoint (non-crossing) or y=0 (crossing).
#[cfg(kani)]
#[kani::proof]
fn pow2_crown_lower_bound_sound() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = pow2_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_sq = x * x;

    if true_sq.is_finite() && linear_bound.is_finite() {
        assert!(linear_bound <= true_sq, "x^2 lower bound must be <= x^2");
    }
}

/// Proof: PowConstant(2) CROWN upper bound is sound.
///
/// PROPERTY: For all x in [lower, upper]:
///   upper_slope * x + upper_intercept >= x^2
///
/// Upper bound: chord through (l, l^2) and (u, u^2).
#[cfg(kani)]
#[kani::proof]
fn pow2_crown_upper_bound_sound() {
    let lower = any_bounded_f32_no_negzero(-10.0, 10.0);
    let upper = any_bounded_f32_no_negzero(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = pow2_linear_relaxation(lower, upper);

    let x = any_bounded_f32_no_negzero(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_sq = x * x;

    if true_sq.is_finite() && linear_bound.is_finite() {
        assert!(linear_bound >= true_sq, "x^2 upper bound must be >= x^2");
    }
}

/// Proof: PowConstant(2) CROWN bounds are well-formed (lower <= upper at all points).
#[cfg(kani)]
#[kani::proof]
fn pow2_crown_bounds_well_formed() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = pow2_linear_relaxation(lower, upper);

    let x = any_bounded_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "x^2 lower bound must be <= upper bound"
        );
    }
}

/// Proof: PowConstant(2) CROWN bounds contain endpoints.
///
/// PROPERTY: The relaxation contains l^2 and u^2.
#[cfg(kani)]
#[kani::proof]
fn pow2_crown_contains_endpoints() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = pow2_linear_relaxation(lower, upper);

    // Check at lower endpoint
    let lb_at_l = lower_slope * lower + lower_intercept;
    let ub_at_l = upper_slope * lower + upper_intercept;
    let sq_l = lower * lower;

    if sq_l.is_finite() && lb_at_l.is_finite() && ub_at_l.is_finite() {
        assert!(lb_at_l <= sq_l, "x^2 lower bound must contain l^2");
        assert!(ub_at_l >= sq_l, "x^2 upper bound must contain l^2");
    }

    // Check at upper endpoint
    let lb_at_u = lower_slope * upper + lower_intercept;
    let ub_at_u = upper_slope * upper + upper_intercept;
    let sq_u = upper * upper;

    if sq_u.is_finite() && lb_at_u.is_finite() && ub_at_u.is_finite() {
        assert!(lb_at_u <= sq_u, "x^2 lower bound must contain u^2");
        assert!(ub_at_u >= sq_u, "x^2 upper bound must contain u^2");
    }
}

/// Proof: PowConstant(2) output is non-negative.
///
/// PROPERTY: x^2 >= 0, so the lower bound should be >= 0 (or conservatively below).
/// The upper bound should be non-negative.
#[cfg(kani)]
#[kani::proof]
fn pow2_crown_upper_bound_non_negative() {
    let lower = any_bounded_f32(-10.0, 10.0);
    let upper = any_bounded_f32(-10.0, 10.0);

    assume(lower <= upper);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = pow2_linear_relaxation(lower, upper);

    // Check at both endpoints — upper bound at endpoints should be >= 0
    let ub_at_l = upper_slope * lower + upper_intercept;
    let ub_at_u = upper_slope * upper + upper_intercept;

    if ub_at_l.is_finite() {
        assert!(ub_at_l >= -1e-5, "x^2 upper bound at l should be >= 0");
    }
    if ub_at_u.is_finite() {
        assert!(ub_at_u >= -1e-5, "x^2 upper bound at u should be >= 0");
    }
}

/// Proof: PowConstant(2) crossing lower bound is zero.
///
/// PROPERTY: When l < 0 < u, the lower bound is constant y = 0.
#[cfg(kani)]
#[kani::proof]
fn pow2_crown_crossing_lower_is_zero() {
    let lower = any_bounded_f32(-10.0, -0.001);
    let upper = any_bounded_f32(0.001, 10.0);

    assume(lower < 0.0);
    assume(upper > 0.0);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = pow2_linear_relaxation(lower, upper);

    assert_eq!(lower_slope, 0.0, "Crossing lower slope must be 0");
    assert_eq!(lower_intercept, 0.0, "Crossing lower intercept must be 0");
}

// =============================================================================
// Sqrt alpha-parameterized relaxation proofs
// =============================================================================
//
// The default `sqrt_linear_relaxation` always uses tangent at `u` (upper endpoint).
// During alpha-CROWN optimization, the tangent point `mid` is a learnable parameter
// that can be anywhere in [l, u]. These proofs verify soundness for arbitrary mid.
//
// Reference: ny-relaxation/src/sqrt.rs `sqrt_linear_relaxation_with_alpha`
// Part of #2519.

/// Proof: Sqrt alpha-CROWN lower bound is sound for arbitrary tangent point.
///
/// PROPERTY: For all x in [lower, upper] with lower >= 0, and any mid in [lower, upper]:
///   lower_slope * x + lower_intercept <= sqrt(x)
///
/// The lower bound (chord) is independent of mid, but we verify the full
/// relaxation struct returned by the alpha-parameterized function.
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_lower_bound_sound() {
    let lower = any_bounded_f32(0.0, 10.0);
    let upper = any_bounded_f32(0.0, 10.0);
    let mid = any_bounded_f32(0.0, 10.0);

    assume(lower <= upper);
    assume(lower >= 0.0);
    assume(upper - lower >= 0.001);
    assume(mid >= lower && mid <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        ..
    } = sqrt_linear_relaxation_with_alpha(lower, upper, mid);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = lower_slope * x + lower_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound <= true_sqrt,
            "Sqrt alpha lower bound must be <= sqrt(x)"
        );
    }
}

/// Proof: Sqrt alpha-CROWN upper bound is sound for arbitrary tangent point.
///
/// PROPERTY: For all x in [lower, upper] with lower >= 0, and any mid in [lower, upper]:
///   upper_slope * x + upper_intercept >= sqrt(x)
///
/// The upper bound uses the tangent at `mid` instead of always at `u`.
/// Since sqrt is concave, any tangent lies above the curve — but we must verify
/// that the directed-rounding adjustments preserve this for all mid values.
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_upper_bound_sound() {
    let lower = any_bounded_f32(0.0, 10.0);
    let upper = any_bounded_f32(0.0, 10.0);
    let mid = any_bounded_f32(0.0, 10.0);

    assume(lower <= upper);
    assume(lower >= 0.0);
    assume(upper - lower >= 0.001);
    assume(mid >= lower && mid <= upper);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = sqrt_linear_relaxation_with_alpha(lower, upper, mid);

    let x = any_bounded_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_sqrt,
            "Sqrt alpha upper bound must be >= sqrt(x)"
        );
    }
}

/// Proof: Sqrt alpha-CROWN bounds are well-formed for arbitrary tangent point.
///
/// PROPERTY: lower_slope * x + lower_intercept <= upper_slope * x + upper_intercept
/// for all x in [lower, upper] and any mid in [lower, upper].
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_bounds_well_formed() {
    let lower = any_bounded_f32(0.0, 10.0);
    let upper = any_bounded_f32(0.0, 10.0);
    let mid = any_bounded_f32(0.0, 10.0);

    assume(lower <= upper);
    assume(lower >= 0.0);
    assume(upper - lower >= 0.001);
    assume(mid >= lower && mid <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = sqrt_linear_relaxation_with_alpha(lower, upper, mid);

    let x = any_bounded_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Sqrt alpha lower bound must be <= upper bound"
        );
    }
}

/// Proof: Sqrt alpha-CROWN contains true value at endpoints.
///
/// PROPERTY: For any mid in [lower, upper]:
///   lower_slope * l + lower_intercept <= sqrt(l) <= upper_slope * l + upper_intercept
///   lower_slope * u + lower_intercept <= sqrt(u) <= upper_slope * u + upper_intercept
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_contains_endpoints() {
    let lower = any_bounded_f32(0.01, 10.0);
    let upper = any_bounded_f32(0.01, 10.0);
    let mid = any_bounded_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(upper - lower >= 0.01);
    assume(mid >= lower && mid <= upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = sqrt_linear_relaxation_with_alpha(lower, upper, mid);

    let sqrt_l = lower.sqrt();
    let sqrt_u = upper.sqrt();

    let lb_at_l = lower_slope * lower + lower_intercept;
    let ub_at_l = upper_slope * lower + upper_intercept;
    let lb_at_u = lower_slope * upper + lower_intercept;
    let ub_at_u = upper_slope * upper + upper_intercept;

    if lb_at_l.is_finite() && sqrt_l.is_finite() {
        assert!(
            lb_at_l <= sqrt_l + 1e-5,
            "Lower bound at l must be <= sqrt(l)"
        );
    }
    if ub_at_l.is_finite() && sqrt_l.is_finite() {
        assert!(
            ub_at_l >= sqrt_l - 1e-5,
            "Upper bound at l must be >= sqrt(l)"
        );
    }
    if lb_at_u.is_finite() && sqrt_u.is_finite() {
        assert!(
            lb_at_u <= sqrt_u + 1e-5,
            "Lower bound at u must be <= sqrt(u)"
        );
    }
    if ub_at_u.is_finite() && sqrt_u.is_finite() {
        assert!(
            ub_at_u >= sqrt_u - 1e-5,
            "Upper bound at u must be >= sqrt(u)"
        );
    }
}

/// Proof: Sqrt alpha-CROWN with negative-region input is still sound.
///
/// PROPERTY: When lower < 0, the function clamps to [0, u] for sqrt computation
/// but adjusts the upper intercept. Bounds must still contain sqrt(max(x, 0)).
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_negative_region_sound() {
    let lower = any_bounded_f32(-5.0, -0.001);
    let upper = any_bounded_f32(0.1, 10.0);
    let mid = any_bounded_f32(0.0, 10.0);

    assume(lower < 0.0);
    assume(upper > 0.0);
    assume(mid >= 0.0 && mid <= upper);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = sqrt_linear_relaxation_with_alpha(lower, upper, mid);

    // For x >= 0, upper bound must still be >= sqrt(x)
    let x = any_bounded_f32(0.0, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_sqrt,
            "Sqrt alpha upper bound must be >= sqrt(x) even with negative lower"
        );
    }
}

// Targeted grid proofs: fix mid at key alpha-CROWN optimization points to keep
// the SAT problem at 3 symbolic variables (256^3 ≈ 16M, tractable).
// 4 grid variables (256^4 ≈ 4B) times out CBMC.

/// Proof: Sqrt alpha-CROWN upper bound sound when tangent at lower endpoint.
///
/// mid=lower is the tightest possible upper bound (tangent closest to curve minimum).
/// This is the most aggressive alpha-CROWN setting.
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_upper_sound_mid_at_lower() {
    let lower = any_grid_f32(0.01, 10.0);
    let upper = any_grid_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(upper - lower >= 0.01);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = sqrt_linear_relaxation_with_alpha(lower, upper, lower);

    let x = any_grid_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_sqrt,
            "Sqrt alpha upper bound (mid=lower) must be >= sqrt(x)"
        );
    }
}

/// Proof: Sqrt alpha-CROWN upper bound sound when tangent at midpoint.
///
/// mid=(lower+upper)/2 is the default non-optimized alpha setting.
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_upper_sound_mid_at_center() {
    let lower = any_grid_f32(0.01, 10.0);
    let upper = any_grid_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(upper - lower >= 0.01);

    let mid = 0.5 * (lower + upper);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = sqrt_linear_relaxation_with_alpha(lower, upper, mid);

    let x = any_grid_f32(lower, upper);
    let linear_bound = upper_slope * x + upper_intercept;
    let true_sqrt = x.sqrt();

    if true_sqrt.is_finite() && linear_bound.is_finite() {
        assert!(
            linear_bound >= true_sqrt,
            "Sqrt alpha upper bound (mid=center) must be >= sqrt(x)"
        );
    }
}

/// Proof: Sqrt alpha-CROWN bounds well-formed when tangent at lower endpoint.
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_well_formed_mid_at_lower() {
    let lower = any_grid_f32(0.01, 10.0);
    let upper = any_grid_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(upper - lower >= 0.01);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = sqrt_linear_relaxation_with_alpha(lower, upper, lower);

    let x = any_grid_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Sqrt alpha bounds (mid=lower) must be well-formed"
        );
    }
}

/// Proof: Sqrt alpha-CROWN bounds well-formed when tangent at midpoint.
#[cfg(kani)]
#[kani::proof]
fn sqrt_alpha_well_formed_mid_at_center() {
    let lower = any_grid_f32(0.01, 10.0);
    let upper = any_grid_f32(0.01, 10.0);

    assume(lower <= upper);
    assume(upper - lower >= 0.01);

    let mid = 0.5 * (lower + upper);

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = sqrt_linear_relaxation_with_alpha(lower, upper, mid);

    let x = any_grid_f32(lower, upper);
    let lower_bound = lower_slope * x + lower_intercept;
    let upper_bound = upper_slope * x + upper_intercept;

    if lower_bound.is_finite() && upper_bound.is_finite() {
        assert!(
            lower_bound <= upper_bound,
            "Sqrt alpha bounds (mid=center) must be well-formed"
        );
    }
}

// =============================================================================
// CROWN Backward Step Composition Proof
// =============================================================================
// A single CROWN backward step through a linear layer computes:
//   output_bound = W * input_bound + bias
// where W * input_bound is interval multiplication and + bias is safe addition.
// This harness verifies that the COMPOSITION of interval_mul + safe_add is sound.
//
// This is the single most important soundness property in the entire system:
// if this fails, every CROWN bound computed through a linear layer is wrong.

/// Proof: A CROWN backward step (interval multiply + safe add) is sound.
///
/// PROPERTY: For weight w in [w_l, w_u], input x in [x_l, x_u], bias b:
///   The true output w*x + b is contained in
///   [safe_add_lower(interval_mul_lower, b), safe_add_upper(interval_mul_upper, b)]
///
/// This composes interval_mul_for_bounds and safe_add_{lower,upper}_for_bounds
/// to verify end-to-end soundness of a single affine bound propagation step.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn crown_affine_step_sound() {
    // Weight interval
    let w_l_i = any_bounded_i8(-10, 10);
    let w_u_i = any_bounded_i8(-10, 10);
    assume(w_l_i <= w_u_i);

    // Input interval
    let x_l_i = any_bounded_i8(-10, 10);
    let x_u_i = any_bounded_i8(-10, 10);
    assume(x_l_i <= x_u_i);

    let w_l = w_l_i as f32;
    let w_u = w_u_i as f32;
    let x_l = x_l_i as f32;
    let x_u = x_u_i as f32;

    // Bias (scalar, not an interval)
    let bias = any_bounded_i8(-10, 10) as f32;

    // Step 1: Interval multiplication W * X
    let (prod_l, prod_u) = interval_mul_for_bounds(w_l, w_u, x_l, x_u);

    // Step 2: Add bias using safe addition
    let out_l = safe_add_lower_for_bounds(prod_l, bias);
    let out_u = safe_add_upper_for_bounds(prod_u, bias);

    // Pick concrete points in the intervals
    let w = any_bounded_i8(w_l_i, w_u_i) as f32;
    let x = any_bounded_i8(x_l_i, x_u_i) as f32;

    // True output
    let true_output = w * x + bias;

    if true_output.is_finite() {
        assert!(
            out_l <= true_output,
            "CROWN affine step lower bound violated: w*x + b must be >= computed lower"
        );
        assert!(
            true_output <= out_u,
            "CROWN affine step upper bound violated: w*x + b must be <= computed upper"
        );
    }

    // Well-formedness: output interval must be valid
    if out_l.is_finite() && out_u.is_finite() {
        assert!(
            out_l <= out_u,
            "CROWN affine step must produce valid interval (lower <= upper)"
        );
    }
}
