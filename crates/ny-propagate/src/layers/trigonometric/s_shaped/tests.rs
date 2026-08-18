// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::arctan::arctan_linear_relaxation;
use super::shared::{s_shaped_finalize, SShapedPrecomputeTables};
use super::sigmoid::{
    sigmoid_crossing_default_tangents, sigmoid_f64, sigmoid_linear_relaxation,
    sigmoid_linear_relaxation_with_alpha,
};
use super::tanh::{
    tanh_crossing_default_tangents, tanh_d_f64, tanh_f64, tanh_linear_relaxation,
    tanh_linear_relaxation_with_alpha,
};
use crate::bounds::MonotoneSShapedPathAlpha;
use crate::layers::activations::LinearRelaxation;
use crate::tests::assert_relaxation_sound;
use proptest::prelude::*;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[ntest::timeout(10000)]
#[test]
fn test_tanh_linear_relaxation_sound() {
    let intervals = [(-4.0, -1.0), (-1.5, 0.5), (-0.5, 2.0), (0.5, 3.0)];
    for (l, u) in intervals {
        let relaxation = tanh_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::tanh, 1e-4, "tanh");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sigmoid_linear_relaxation_sound() {
    let intervals = [(-6.0, -2.0), (-1.0, 1.0), (-0.5, 2.5), (1.0, 6.0)];
    for (l, u) in intervals {
        let relaxation = sigmoid_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, sigmoid, 1e-4, "sigmoid");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_arctan_linear_relaxation_sound() {
    let intervals = [(-6.0, -1.5), (-1.0, 1.0), (0.5, 4.0)];
    for (l, u) in intervals {
        let relaxation = arctan_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::atan, 1e-4, "atan");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    #[ntest::timeout(10000)]
    #[test]
    fn prop_tanh_linear_relaxation_sound(l in -10.0f32..10.0, u in -10.0f32..10.0) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = tanh_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::tanh, 1e-4, "tanh");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn prop_sigmoid_linear_relaxation_sound(l in -10.0f32..10.0, u in -10.0f32..10.0) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = sigmoid_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, sigmoid, 1e-4, "sigmoid");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn prop_arctan_linear_relaxation_sound(l in -10.0f32..10.0, u in -10.0f32..10.0) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = arctan_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::atan, 1e-4, "atan");
    }
}

// ========================================================================
// Large-input overflow guard tests (#2625)
//
// When l or u is near f32::MAX, the slope-rounding error adjustment
// (slope_error * max_abs_x) can overflow and produce Inf/vacuous
// intercepts. The clamp in s_shaped_finalize prevents this.
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_tanh_large_input_no_inf() {
    let half_max = f32::MAX / 2.0;
    let max = f32::MAX;
    let relaxation = tanh_linear_relaxation(half_max, max);
    assert!(
        relaxation.lower_slope.is_finite(),
        "tanh lower slope should be finite, got {}",
        relaxation.lower_slope
    );
    assert!(
        relaxation.lower_intercept.is_finite(),
        "tanh lower intercept should be finite, got {}",
        relaxation.lower_intercept
    );
    assert!(
        relaxation.upper_slope.is_finite(),
        "tanh upper slope should be finite, got {}",
        relaxation.upper_slope
    );
    assert!(
        relaxation.upper_intercept.is_finite(),
        "tanh upper intercept should be finite, got {}",
        relaxation.upper_intercept
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sigmoid_large_input_no_inf() {
    let half_max = f32::MAX / 2.0;
    let max = f32::MAX;
    let relaxation = sigmoid_linear_relaxation(half_max, max);
    assert!(
        relaxation.lower_slope.is_finite(),
        "sigmoid lower slope should be finite, got {}",
        relaxation.lower_slope
    );
    assert!(
        relaxation.lower_intercept.is_finite(),
        "sigmoid lower intercept should be finite, got {}",
        relaxation.lower_intercept
    );
    assert!(
        relaxation.upper_slope.is_finite(),
        "sigmoid upper slope should be finite, got {}",
        relaxation.upper_slope
    );
    assert!(
        relaxation.upper_intercept.is_finite(),
        "sigmoid upper intercept should be finite, got {}",
        relaxation.upper_intercept
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_arctan_large_input_no_inf() {
    let half_max = f32::MAX / 2.0;
    let max = f32::MAX;
    let relaxation = arctan_linear_relaxation(half_max, max);
    assert!(
        relaxation.lower_slope.is_finite(),
        "arctan lower slope should be finite, got {}",
        relaxation.lower_slope
    );
    assert!(
        relaxation.lower_intercept.is_finite(),
        "arctan lower intercept should be finite, got {}",
        relaxation.lower_intercept
    );
    assert!(
        relaxation.upper_slope.is_finite(),
        "arctan upper slope should be finite, got {}",
        relaxation.upper_slope
    );
    assert!(
        relaxation.upper_intercept.is_finite(),
        "arctan upper intercept should be finite, got {}",
        relaxation.upper_intercept
    );
}

/// Test crossing-origin interval with large magnitude — this exercises the
/// tangent-point table path where slopes are non-trivial (#2625).
/// Uses ±1000 (beyond table's X_LIMIT=500 range) to exercise the default
/// tangent point fallback without triggering isize overflow in table index.
#[ntest::timeout(10000)]
#[test]
fn test_tanh_large_crossing_no_inf() {
    let relaxation = tanh_linear_relaxation(-1000.0, 1000.0);
    assert!(
        relaxation.lower_slope.is_finite() && relaxation.lower_intercept.is_finite(),
        "tanh crossing lower should be finite: slope={}, intercept={}",
        relaxation.lower_slope,
        relaxation.lower_intercept
    );
    assert!(
        relaxation.upper_slope.is_finite() && relaxation.upper_intercept.is_finite(),
        "tanh crossing upper should be finite: slope={}, intercept={}",
        relaxation.upper_slope,
        relaxation.upper_intercept
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_s_shaped_retrieve_large_bound_returns_default() {
    let tables = SShapedPrecomputeTables::new(tanh_f64, tanh_d_f64);
    let default_d = 12_345.0;
    let retrieved = tables.retrieve(&tables.d_lower, 1e30, default_d);
    assert_eq!(
        retrieved, default_d,
        "large bound should use default tangent point fallback"
    );
    assert_ne!(
        retrieved, tables.d_lower[0],
        "large bound should not wrap to table[0]"
    );
}

// ========================================================================
// Certified table roots and affine publication (#nonlinear-relaxation-audit)
// ========================================================================

fn sigmoid_d_for_table(x: f64) -> f64 {
    let y = sigmoid_f64(x);
    y * (1.0 - y)
}

fn arctan_for_table(x: f64) -> f64 {
    x.atan()
}

fn arctan_d_for_table(x: f64) -> f64 {
    1.0 / (1.0 + x * x)
}

fn assert_table_entry_valid(
    tables: &SShapedPrecomputeTables,
    index: usize,
    func: fn(f64) -> f64,
    dfunc: fn(f64) -> f64,
    label: &str,
) {
    // This is exactly the grid coordinate used by table construction.  Check
    // the defining tangent inequalities with no tolerance: root storage is
    // directed toward the valid side and must not rely on the later 1e-6
    // relaxation pad for validity.
    let q = f64::from(0.01_f32) * index as f64;
    let d_lower = f64::from(tables.d_lower[index]);
    let lower_at_q = dfunc(d_lower) * (q - d_lower) + func(d_lower);
    assert!(
        lower_at_q <= func(q),
        "{label} lower table[{index}] invalid: tangent({q})={lower_at_q} > f({q})={} by {}",
        func(q),
        lower_at_q - func(q),
    );

    let d_upper = f64::from(tables.d_upper[index]);
    let upper_at_neg_q = dfunc(d_upper) * (-q - d_upper) + func(d_upper);
    assert!(
        upper_at_neg_q >= func(-q),
        "{label} upper table[{index}] invalid: tangent({})={upper_at_neg_q} < f({})={} by {}",
        -q,
        -q,
        func(-q),
        func(-q) - upper_at_neg_q,
    );
}

#[test]
fn test_s_shaped_precomputed_roots_are_stored_on_valid_side() {
    // These are the worst nearest-cast counterexamples found by an exhaustive
    // scan of all 50_005 entries before the directed-root fix:
    //   tanh    lower/upper miss ~= 4.753512e-7 at 49_089,
    //   sigmoid lower/upper miss ~= 2.368261e-7 at 44_867,
    //   arctan  lower/upper miss ~= 3.527136e-7 at 18_872.
    let tanh_tables = SShapedPrecomputeTables::new(tanh_f64, tanh_d_f64);
    for index in 0..tanh_tables.d_lower.len() {
        assert_table_entry_valid(&tanh_tables, index, tanh_f64, tanh_d_f64, "tanh");
    }

    let sigmoid_tables = SShapedPrecomputeTables::new(sigmoid_f64, sigmoid_d_for_table);
    for index in 0..sigmoid_tables.d_lower.len() {
        assert_table_entry_valid(
            &sigmoid_tables,
            index,
            sigmoid_f64,
            sigmoid_d_for_table,
            "sigmoid",
        );
    }

    let arctan_tables = SShapedPrecomputeTables::new(arctan_for_table, arctan_d_for_table);
    for index in 0..arctan_tables.d_lower.len() {
        assert_table_entry_valid(
            &arctan_tables,
            index,
            arctan_for_table,
            arctan_d_for_table,
            "arctan",
        );
    }
}

fn assert_finalized_lines_enclose_sources(l: f32, u: f32) {
    let ulp_at_one = f64::from(f32::EPSILON);
    // Lower rounds upward to 1+ulp; upper rounds downward to 1.  On a large
    // positive interval both are the unsafe slope-cast directions.  The
    // negative-domain caller swaps the fractions below to exercise the
    // opposite unsafe directions.
    let (lower_slope, upper_slope) = if l >= 0.0 {
        (1.0 + 0.75 * ulp_at_one, 1.0 + 0.25 * ulp_at_one)
    } else {
        (1.0 + 0.25 * ulp_at_one, 1.0 + 0.75 * ulp_at_one)
    };
    let lower_intercept = 0.125_f64;
    let upper_intercept = -0.25_f64;
    let relaxation = s_shaped_finalize(
        l,
        u,
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    );

    for x in [
        f64::from(l),
        f64::midpoint(f64::from(l), f64::from(u)),
        f64::from(u),
    ] {
        let published_lower =
            f64::from(relaxation.lower_slope) * x + f64::from(relaxation.lower_intercept);
        let source_lower = lower_slope * x + lower_intercept;
        assert!(
            published_lower <= source_lower,
            "published lower line exceeded its f64 source at x={x}: {published_lower} > {source_lower}"
        );

        let published_upper =
            f64::from(relaxation.upper_slope) * x + f64::from(relaxation.upper_intercept);
        let source_upper = upper_slope * x + upper_intercept;
        assert!(
            published_upper >= source_upper,
            "published upper line fell below its f64 source at x={x}: {published_upper} < {source_upper}"
        );
    }
}

#[test]
fn test_s_shaped_finalize_preserves_lines_on_extreme_finite_domains() {
    // Before endpoint compensation, the 1e20 max-|x| clamp undercharged the
    // slope-cast displacement here by ten orders of magnitude.
    assert_finalized_lines_enclose_sources(1.0e30, 2.0e30);
    assert_finalized_lines_enclose_sources(-2.0e30, -1.0e30);
}

#[test]
fn test_s_shaped_finalize_nonfinite_inputs_fail_closed() {
    for relaxation in [
        s_shaped_finalize(0.0, 1.0, f64::NAN, 0.0, 1.0, 0.0),
        s_shaped_finalize(0.0, 1.0, 1.0, f64::INFINITY, 1.0, 0.0),
        s_shaped_finalize(0.0, 1.0, 1.0, 0.0, f64::NEG_INFINITY, 0.0),
        s_shaped_finalize(0.0, f32::INFINITY, 1.0, 0.0, 1.0, 0.0),
        s_shaped_finalize(1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        // Finite f64 slope that cannot be represented as a finite f32 slope.
        s_shaped_finalize(0.0, 1.0, f64::MAX, 0.0, 1.0, 0.0),
    ] {
        assert_eq!(relaxation.lower_slope, 0.0);
        assert_eq!(relaxation.lower_intercept, f32::NEG_INFINITY);
        assert_eq!(relaxation.upper_slope, 0.0);
        assert_eq!(relaxation.upper_intercept, f32::INFINITY);
    }
}

fn strict_f64_grid_enclosure(
    l: f32,
    u: f32,
    relaxation: LinearRelaxation,
    func: fn(f64) -> f64,
    label: &str,
) {
    assert!(
        relaxation.lower_slope.is_finite()
            && relaxation.lower_intercept.is_finite()
            && relaxation.upper_slope.is_finite()
            && relaxation.upper_intercept.is_finite(),
        "{label} unexpectedly returned a non-finite relaxation on [{l}, {u}]: {relaxation:?}"
    );
    for i in 0..=2_000 {
        let t = i as f64 / 2_000.0;
        let x = f64::from(l) + t * (f64::from(u) - f64::from(l));
        let y = func(x);
        let lower = f64::from(relaxation.lower_slope) * x + f64::from(relaxation.lower_intercept);
        let upper = f64::from(relaxation.upper_slope) * x + f64::from(relaxation.upper_intercept);
        assert!(
            lower <= y,
            "{label} lower alpha line unsound at x={x} on [{l}, {u}]: {lower} > {y}"
        );
        assert!(
            upper >= y,
            "{label} upper alpha line unsound at x={x} on [{l}, {u}]: {upper} < {y}"
        );
    }
}

fn alpha_bundle(
    tp_pos: f32,
    tp_neg: f32,
    tp_both_lower: f32,
    tp_both_upper: f32,
    d_lower: f32,
    d_upper: f32,
) -> MonotoneSShapedPathAlpha {
    MonotoneSShapedPathAlpha {
        tp_pos,
        tp_neg,
        tp_both_lower,
        tp_both_upper,
        d_lower,
        d_upper,
    }
}

#[test]
fn test_tanh_and_sigmoid_alpha_tangents_are_strictly_sound_on_adversarial_grid() {
    for &(l, u) in &[(-8.0_f32, -0.125_f32), (0.125, 8.0)] {
        for tangent in [l, f32::midpoint(l, u), u] {
            let alpha = alpha_bundle(tangent, tangent, l, u, l, u);
            strict_f64_grid_enclosure(
                l,
                u,
                tanh_linear_relaxation_with_alpha(l, u, alpha),
                tanh_f64,
                "tanh same-sign alpha",
            );
            strict_f64_grid_enclosure(
                l,
                u,
                sigmoid_linear_relaxation_with_alpha(l, u, alpha),
                sigmoid_f64,
                "sigmoid same-sign alpha",
            );
        }
    }

    for &(l, u) in &[(-8.0_f32, 0.125_f32), (-0.125, 8.0), (-3.0, 3.0)] {
        let (tanh_dl, tanh_du) = tanh_crossing_default_tangents(l, u);
        let (sigmoid_dl, sigmoid_du) = sigmoid_crossing_default_tangents(l, u);
        for distance in [0.0_f32, 0.25, 4.0, 1_000.0] {
            let tanh_alpha = alpha_bundle(
                0.0,
                0.0,
                tanh_dl - distance,
                tanh_du + distance,
                tanh_dl,
                tanh_du,
            );
            strict_f64_grid_enclosure(
                l,
                u,
                tanh_linear_relaxation_with_alpha(l, u, tanh_alpha),
                tanh_f64,
                "tanh crossing alpha",
            );

            let sigmoid_alpha = alpha_bundle(
                0.0,
                0.0,
                sigmoid_dl - distance,
                sigmoid_du + distance,
                sigmoid_dl,
                sigmoid_du,
            );
            strict_f64_grid_enclosure(
                l,
                u,
                sigmoid_linear_relaxation_with_alpha(l, u, sigmoid_alpha),
                sigmoid_f64,
                "sigmoid crossing alpha",
            );
        }
    }
}

#[test]
fn test_s_shaped_alpha_nonfinite_tangent_fails_closed() {
    let alpha = alpha_bundle(f32::NAN, 0.0, -1.0, 1.0, -1.0, 1.0);
    for relaxation in [
        tanh_linear_relaxation_with_alpha(0.0, 1.0, alpha),
        sigmoid_linear_relaxation_with_alpha(0.0, 1.0, alpha),
    ] {
        assert_eq!(relaxation, LinearRelaxation::nan_fallback());
    }
}

// ========================================================================
// CROWN backward soundness tests (#2292)
//
// These test propagate_linear_with_bounds (the full CROWN backward path),
// not just the per-element relaxation functions. They exercise:
//   - coefficient matrix multiplication via crown_elementwise_backward
//   - sign-dependent slope/intercept swapping for negative A coefficients
//   - BoundedTensor → LinearBounds extraction
// ========================================================================

// assert_crown_backward_sound extracted to crate::tests::assert_crown_backward_sound (#2307)
use crate::tests::assert_crown_backward_sound;

#[ntest::timeout(10000)]
#[test]
fn test_tanh_crown_backward_soundness() {
    let layer = super::tanh::TanhLayer;
    // Crossing interval + asymmetric ranges
    let intervals = [(-3.0, 3.0), (-4.0, -1.0), (-1.5, 0.5), (0.5, 3.0)];
    assert_crown_backward_sound(&layer, &intervals, f32::tanh);
}

#[ntest::timeout(10000)]
#[test]
fn test_sigmoid_crown_backward_soundness() {
    let layer = super::sigmoid::SigmoidLayer;
    // Crossing interval + asymmetric ranges
    let intervals = [(-3.0, 3.0), (-6.0, -2.0), (-1.0, 1.0), (1.0, 6.0)];
    assert_crown_backward_sound(&layer, &intervals, sigmoid);
}

#[ntest::timeout(10000)]
#[test]
fn test_arctan_crown_backward_soundness() {
    let layer = super::arctan::ArctanLayer;
    // Crossing interval + asymmetric ranges
    let intervals = [(-3.0, 3.0), (-6.0, -1.5), (-1.0, 1.0), (0.5, 4.0)];
    assert_crown_backward_sound(&layer, &intervals, f32::atan);
}

// ── Strict zero-tolerance CROWN relaxation proptests (#3292) ─────────
//
// Pattern from #3285: f64-evaluated reference with zero tolerance catches
// f32 cancellation bugs invisible to magnitude-scaled tolerance tests.
// Ref: silu/tests.rs proptest_silu_relaxation_strict_soundness

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Strict soundness proptest for sigmoid CROWN relaxation.
    /// Uses f64 reference (sigmoid_f64) with zero tolerance on 200-point grid.
    /// Ref: alpha-beta-CROWN auto_LiRPA sigmoid relaxation, #3292.
    #[test]
    fn proptest_sigmoid_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = sigmoid_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        // Skip NaN fallback (infinite bounds).
        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = sigmoid_f64(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "sigmoid lower bound UNSOUND at x={x}: {lower_val} > sigmoid({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "sigmoid upper bound UNSOUND at x={x}: {upper_val} < sigmoid({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }

    /// Strict soundness proptest for tanh CROWN relaxation.
    /// Uses f64 reference (tanh_f64) with zero tolerance on 200-point grid.
    /// Ref: alpha-beta-CROWN auto_LiRPA tanh relaxation, #3292.
    #[test]
    fn proptest_tanh_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = tanh_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = tanh_f64(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "tanh lower bound UNSOUND at x={x}: {lower_val} > tanh({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "tanh upper bound UNSOUND at x={x}: {upper_val} < tanh({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }

    /// Strict soundness proptest for arctan CROWN relaxation.
    /// Uses f64 reference (f64::atan) with zero tolerance on 200-point grid.
    /// Mirrors the sigmoid/tanh strict-soundness guards (#3292); added alongside
    /// the parallel-to-chord single-convexity tangent tightening so the
    /// arctan path has the same zero-tolerance enclosure check.
    #[test]
    fn proptest_arctan_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = arctan_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = (x).atan();

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "arctan lower bound UNSOUND at x={x}: {lower_val} > atan({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "arctan upper bound UNSOUND at x={x}: {upper_val} < atan({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }
}

// ── Parallel-to-chord single-convexity tangent: tightness regression ─────
//
// The single-convexity (all-negative convex / all-positive concave) tangent
// was switched from the MIDPOINT tangent to the tangent PARALLEL TO THE CHORD.
// Soundness is covered by the strict proptests above (every tangent of a
// convex/concave f is a valid global lower/upper bound). These tests assert the
// new line is at least as tight as the old midpoint tangent on wide single-
// convexity intervals, and strictly tighter on at least one wide interval.

/// Worst-case gap of a line g(x)=slope*x+intercept against f over [l,u].
fn max_gap(l: f64, u: f64, f: fn(f64) -> f64, slope: f64, intercept: f64, is_lower: bool) -> f64 {
    let mut worst = 0.0f64;
    for k in 0..=4000 {
        let x = l + (u - l) * (k as f64 / 4000.0);
        let fx = f(x);
        let line = slope * x + intercept;
        // For a lower line the gap is f - line; for an upper line it is line - f.
        let gap = if is_lower { fx - line } else { line - fx };
        worst = worst.max(gap);
    }
    worst
}

/// Old midpoint tangent (slope, intercept) — the bound this change replaces.
fn midpoint_tangent(l: f64, u: f64, f: fn(f64) -> f64, df: fn(f64) -> f64) -> (f64, f64) {
    let m = f64::midpoint(l, u);
    let k = df(m);
    (k, f(m) - k * m)
}

#[test]
fn test_s_shaped_parallel_tangent_tighter_than_midpoint() {
    // Convex regime (u <= 0): the parallel-to-chord tangent is the LOWER bound.
    // Concave regime (l >= 0): it is the UPPER bound.
    let tanh = |x: f64| x.tanh();
    let tanh_d = |x: f64| {
        let t = x.tanh();
        1.0 - t * t
    };
    let atan = |x: f64| x.atan();
    let atan_d = |x: f64| 1.0 / (1.0 + x * x);

    // (l, u, is_lower=convex/all-neg)
    let convex_cases = [(-6.0_f32, -0.5_f32), (-4.0, -1.0), (-3.0, -0.2)];
    let concave_cases = [(0.5_f32, 6.0_f32), (1.0, 4.0), (0.2, 3.0)];

    let mut saw_strictly_tighter = false;

    for &(l, u) in &convex_cases {
        // tanh
        let r = tanh_linear_relaxation(l, u);
        let new_gap = max_gap(
            l as f64,
            u as f64,
            tanh,
            r.lower_slope as f64,
            r.lower_intercept as f64,
            true,
        );
        let (mk, mb) = midpoint_tangent(l as f64, u as f64, tanh, tanh_d);
        let old_gap = max_gap(l as f64, u as f64, tanh, mk, mb, true);
        assert!(
            new_gap <= old_gap + 1e-5,
            "tanh convex [{l},{u}] not tighter: new {new_gap} > old {old_gap}"
        );
        if new_gap < old_gap - 1e-4 {
            saw_strictly_tighter = true;
        }

        // arctan
        let r = arctan_linear_relaxation(l, u);
        let new_gap = max_gap(
            l as f64,
            u as f64,
            atan,
            r.lower_slope as f64,
            r.lower_intercept as f64,
            true,
        );
        let (mk, mb) = midpoint_tangent(l as f64, u as f64, atan, atan_d);
        let old_gap = max_gap(l as f64, u as f64, atan, mk, mb, true);
        assert!(
            new_gap <= old_gap + 1e-5,
            "atan convex [{l},{u}] not tighter: new {new_gap} > old {old_gap}"
        );
        if new_gap < old_gap - 1e-4 {
            saw_strictly_tighter = true;
        }
    }

    for &(l, u) in &concave_cases {
        // tanh upper
        let r = tanh_linear_relaxation(l, u);
        let new_gap = max_gap(
            l as f64,
            u as f64,
            tanh,
            r.upper_slope as f64,
            r.upper_intercept as f64,
            false,
        );
        let (mk, mb) = midpoint_tangent(l as f64, u as f64, tanh, tanh_d);
        let old_gap = max_gap(l as f64, u as f64, tanh, mk, mb, false);
        assert!(
            new_gap <= old_gap + 1e-5,
            "tanh concave [{l},{u}] not tighter: new {new_gap} > old {old_gap}"
        );
        if new_gap < old_gap - 1e-4 {
            saw_strictly_tighter = true;
        }

        // arctan upper
        let r = arctan_linear_relaxation(l, u);
        let new_gap = max_gap(
            l as f64,
            u as f64,
            atan,
            r.upper_slope as f64,
            r.upper_intercept as f64,
            false,
        );
        let (mk, mb) = midpoint_tangent(l as f64, u as f64, atan, atan_d);
        let old_gap = max_gap(l as f64, u as f64, atan, mk, mb, false);
        assert!(
            new_gap <= old_gap + 1e-5,
            "atan concave [{l},{u}] not tighter: new {new_gap} > old {old_gap}"
        );
        if new_gap < old_gap - 1e-4 {
            saw_strictly_tighter = true;
        }
    }

    assert!(
        saw_strictly_tighter,
        "parallel-to-chord tangent should be strictly tighter than midpoint on at least one wide interval"
    );
}
