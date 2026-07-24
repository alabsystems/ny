// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::math::{silu_critical_point, silu_inflection_points, silu_min_max};
use super::*;
use crate::layers::activations::LinearRelaxation;
use crate::LinearBounds;
use proptest::prelude::*;

/// Helper: verify relaxation soundness at sampled points.
fn assert_sound(l: f32, u: f32, ls: f32, li: f32, us: f32, ui: f32) {
    let n = 100;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = (l + t * (u - l)).clamp(l, u);
        let fx = silu_eval(x);
        let lower = ls * x + li;
        let upper = us * x + ui;
        assert!(
            lower <= fx + 1e-5,
            "Lower bound violated at x={x}: lower={lower} > SiLU({x})={fx}, \
             interval=[{l}, {u}]"
        );
        assert!(
            upper >= fx - 1e-5,
            "Upper bound violated at x={x}: upper={upper} < SiLU({x})={fx}, \
             interval=[{l}, {u}]"
        );
    }
}

#[test]
fn test_nan_returns_maximally_loose() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(f32::NAN, 1.0);
    assert_eq!(ls, 0.0);
    assert_eq!(li, f32::NEG_INFINITY);
    assert_eq!(us, 0.0);
    assert_eq!(ui, f32::INFINITY);

    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(0.0, f32::NAN);
    assert_eq!(ls, 0.0);
    assert_eq!(li, f32::NEG_INFINITY);
    assert_eq!(us, 0.0);
    assert_eq!(ui, f32::INFINITY);
}

#[test]
fn test_positive_infinity_returns_maximally_loose() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(0.0, f32::INFINITY);
    assert_eq!(ls, 0.0);
    assert_eq!(li, f32::NEG_INFINITY);
    assert_eq!(us, 0.0);
    assert_eq!(ui, f32::INFINITY);
}

#[test]
fn test_negative_infinity_returns_constant_bounds() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(f32::NEG_INFINITY, 0.0);
    assert_eq!(ls, 0.0);
    assert_eq!(us, 0.0);
    // Lower bound should be at most the SiLU minimum (~-0.278)
    assert!(li <= silu_eval(silu_critical_point()) + 1e-6);
    // Upper bound should be at least 0 (SiLU(0) = 0, SiLU(-∞) = 0)
    assert!(ui >= -1e-6);
}

#[test]
fn test_inf_nan_not_identity() {
    // The old bug: identity relaxation (1, 0, 1, 0) for Inf/NaN.
    // SiLU(x) ≠ x, so identity is UNSOUND.
    let cases = [
        (f32::NEG_INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, 0.0),
        (0.0, f32::INFINITY),
        (f32::NAN, 0.0),
    ];
    for (l, u) in cases {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = silu_sound_linear_relaxation(l, u);
        assert!(
            !(ls == 1.0 && li == 0.0 && us == 1.0 && ui == 0.0),
            "Identity relaxation returned for [{l}, {u}] — SiLU(x) ≠ x"
        );
    }
}

#[test]
fn test_point_interval() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(1.0, 1.0);
    let y = silu_eval(1.0);
    assert_eq!(ls, 0.0);
    assert!((li - y).abs() < 1e-6);
    assert_eq!(us, 0.0);
    assert!((ui - y).abs() < 1e-6);
}

#[test]
fn test_convex_region_uses_tangent_lower() {
    // Interval [0, 2] is entirely in convex region [p1≈-2.4, p2≈2.4].
    // Should use tangent lower (non-zero slope), chord upper.
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(0.0, 2.0);
    // In the convex region, we expect non-zero slopes (tighter than constant).
    assert!(
        ls != 0.0 || us != 0.0,
        "Expected non-constant bounds in convex region [0, 2]"
    );
    assert_sound(0.0, 2.0, ls, li, us, ui);
}

#[test]
fn test_concave_left_region() {
    // Interval [-5, -3] is entirely in left concave region (< p1≈-2.4).
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(-5.0, -3.0);
    assert_sound(-5.0, -3.0, ls, li, us, ui);
}

#[test]
fn test_concave_right_region() {
    // Interval [3, 5] is entirely in right concave region (> p2≈2.4).
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(3.0, 5.0);
    assert_sound(3.0, 5.0, ls, li, us, ui);
}

#[test]
fn test_crossing_interval_soundness() {
    // Interval [-3, 3] crosses both inflection points.
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(-3.0, 3.0);
    assert_sound(-3.0, 3.0, ls, li, us, ui);
}

#[test]
fn test_wide_interval_soundness() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(-10.0, 10.0);
    assert_sound(-10.0, 10.0, ls, li, us, ui);
}

#[test]
fn test_critical_point_interval_soundness() {
    // Interval spanning the SiLU minimum near -1.28.
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(-2.0, 0.0);
    assert_sound(-2.0, 0.0, ls, li, us, ui);
}

#[test]
fn test_inflection_points_computed_correctly() {
    let (p1, p2) = silu_inflection_points();
    assert!(
        (p1 - (-2.3994)).abs() < 0.01,
        "Left inflection {p1} not near -2.3994"
    );
    assert!(
        (p2 - 2.3994).abs() < 0.01,
        "Right inflection {p2} not near 2.3994"
    );
    assert!(
        (p1.abs() - p2.abs()).abs() < 0.001,
        "Inflection points not symmetric"
    );
}

#[test]
fn test_tightness_improved_over_constant() {
    // For the convex region [0, 2], tangent+chord should be tighter
    // than constant bounds.
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(0.0, 2.0);
    let (min_val, max_val) = silu_min_max(0.0, 2.0);

    let constant_area = (max_val - min_val) * 2.0;
    let n = 100;
    let mut new_area = 0.0_f32;
    for i in 0..n {
        let x = 2.0 * i as f32 / n as f32;
        let upper = us * x + ui;
        let lower = ls * x + li;
        new_area += (upper - lower) * (2.0 / n as f32);
    }
    assert!(
        new_area < constant_area + 1e-3,
        "New relaxation ({new_area}) not tighter than constant ({constant_area})"
    );
}

#[test]
fn test_crossing_intervals_produce_nonconstant_bounds() {
    // Acceptance criterion: crossing intervals should produce non-constant
    // bounds (non-zero slopes) rather than collapsing CROWN to IBP.
    //
    // Previous implementation always fell back to constant bounds for
    // crossing intervals due to the sampling-based validator rejecting
    // valid chord/tangent bounds.
    let cases = [
        (-3.0_f32, 3.0_f32, "both inflections"),
        (-3.0, 1.0, "left inflection"),
        (0.5, 5.0, "right inflection"),
        (-5.0, 5.0, "wide crossing"),
        (-1.0, 10.0, "asymmetric right"),
        (-10.0, 1.0, "asymmetric left"),
    ];
    for (l, u, desc) in cases {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = silu_sound_linear_relaxation(l, u);
        // At least one of lower/upper should have non-zero slope
        assert!(
            ls != 0.0 || us != 0.0,
            "Crossing interval [{l}, {u}] ({desc}) returned constant bounds \
             ({ls}, {li}, {us}, {ui}) — CROWN collapsed to IBP"
        );
        // Verify soundness
        assert_sound(l, u, ls, li, us, ui);
    }
}

#[test]
fn test_proptest_regression_crossing_right() {
    // Regression test for proptest failure on [0.42, 20.08]:
    // Chord was used as upper bound but SiLU exceeds chord in right
    // concave tail. Fixed by using tangent from concave region.
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(0.42, 20.08);
    assert_sound(0.42, 20.08, ls, li, us, ui);
}

#[test]
fn test_proptest_regression_crossing_left() {
    // Regression test for proptest failure on [-2.62, 11.73]:
    // Midpoint tangent produced lower bound that exceeded SiLU near
    // left inflection point. Fixed by binary search for tangent point.
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = silu_sound_linear_relaxation(-2.62, 11.73);
    assert_sound(-2.62, 11.73, ls, li, us, ui);
}

/// Regression test for #2874: narrow crossing intervals near inflection points
/// exercise the f64-intermediate chord computation in `silu_relaxation_crossing()`.
/// The inline f32 chord `(fu - fl) / (u - l)` suffers catastrophic cancellation
/// when `u - l` is small; the fix delegates to `silu_chord()` which uses f64.
///
/// This test verifies soundness of crossing relaxation at various widths.
/// The existing `proptest_silu_chord_f64_precision` tests the chord helper
/// in isolation but not the integration through `silu_relaxation_crossing()`.
#[test]
fn test_narrow_crossing_interval_soundness_2874() {
    // Intervals straddling inflection points that enter silu_relaxation_crossing().
    // Left inflection point p1 ≈ -2.3994, right p2 ≈ 2.3994.
    let cases: &[(f32, f32)] = &[
        (-2.401, -2.399),   // width 0.002, straddles p1
        (-2.4005, -2.3985), // width 0.002, straddles p1
        (-2.41, -2.39),     // width 0.02
        (-2.5, -2.3),       // width 0.2
        (-3.0, -2.0),       // width 1.0 (wider crossing)
        (2.399, 2.401),     // width 0.002, straddles p2
        (2.39, 2.41),       // width 0.02
    ];

    for &(l, u) in cases {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = silu_sound_linear_relaxation(l, u);
        // Soundness: linear bounds must contain SiLU(x) at all sampled points
        assert_sound(l, u, ls, li, us, ui);
    }
}

// The precondition is a debug_assert — it cannot fire in release builds.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "silu_chord requires a non-point interval")]
fn test_silu_chord_panics_on_point_interval_precondition_violation() {
    let _ = math::silu_chord(1.0, 1.0);
}

// ── CROWN backward tests ───────────────────────────────────────────

#[test]
fn test_crown_backward_crossing_soundness() {
    use ndarray::arr1;
    let layer = SiLULayer::new();
    let l = -3.0_f32;
    let u = 3.0_f32;
    let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = silu_eval(x);
        assert!(
            la * x + lb <= y + 1e-5,
            "SiLU CROWN lb violated at x={x}: {} > {y}",
            la * x + lb
        );
        assert!(
            ua * x + ub >= y - 1e-5,
            "SiLU CROWN ub violated at x={x}: {} < {y}",
            ua * x + ub
        );
    }
}

#[test]
fn test_crown_backward_positive_region() {
    use ndarray::arr1;
    let layer = SiLULayer::new();
    let pre = BoundedTensor::new(arr1(&[2.0_f32]).into_dyn(), arr1(&[6.0_f32]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for k in 0..=50 {
        let x = 2.0 + 4.0 * (k as f32 / 50.0);
        let y = silu_eval(x);
        assert!(
            result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-5,
            "SiLU positive lb violated at x={x}"
        );
        assert!(
            result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-5,
            "SiLU positive ub violated at x={x}"
        );
    }
}

#[test]
fn test_crown_backward_near_minimum() {
    use ndarray::arr1;
    // SiLU minimum is near x ≈ -1.28
    let layer = SiLULayer::new();
    let pre =
        BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for k in 0..=50 {
        let x = -2.0 + 2.0 * (k as f32 / 50.0);
        let y = silu_eval(x);
        assert!(
            result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-5,
            "SiLU near-min lb violated at x={x}"
        );
        assert!(
            result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-5,
            "SiLU near-min ub violated at x={x}"
        );
    }
}

#[test]
fn test_crown_backward_multi_neuron() {
    use ndarray::arr1;
    let layer = SiLULayer::new();
    let pre = BoundedTensor::new(
        arr1(&[-3.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 5.0]).into_dyn(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for neuron in 0..2 {
        let la = result.lower_a[[neuron, neuron]];
        let lb = result.lower_b[neuron];
        let ua = result.upper_a[[neuron, neuron]];
        let ub = result.upper_b[neuron];
        let lo = pre.lower()[neuron];
        let hi = pre.upper()[neuron];

        for k in 0..=20 {
            let x = lo + (hi - lo) * (k as f32 / 20.0);
            let y = silu_eval(x);
            assert!(
                la * x + lb <= y + 1e-5,
                "neuron {neuron} lb violated at x={x}"
            );
            assert!(
                ua * x + ub >= y - 1e-5,
                "neuron {neuron} ub violated at x={x}"
            );
        }
    }
}

#[test]
fn test_propagate_linear_requires_preact() {
    let layer = SiLULayer::new();
    let bounds = LinearBounds::identity(1);
    assert!(
        layer.propagate_linear(&bounds).is_err(),
        "SiLU CROWN without pre-activation bounds should fail"
    );
    assert!(layer.requires_pre_activation_bounds());
}

// ── f64 chord precision proptest (#2624) ─────────────────────────────

/// Reference SiLU in f64, independent of the crate implementation.
fn silu_f64_reference(x: f64) -> f64 {
    let sigmoid = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    };
    x * sigmoid
}

/// ULP distance between two f32 values, handling sign correctly.
/// Maps f32 bits to a linear ordering where adjacent floats differ by 1.
fn ulp_distance(a: f32, b: f32) -> u64 {
    fn to_ordered(x: f32) -> i64 {
        let bits = x.to_bits() as i32;
        if bits < 0 {
            // Negative floats: flip all bits except sign to get linear ordering
            (0x8000_0000_u32 as i32 - bits) as i64
        } else {
            bits as i64
        }
    }
    (to_ordered(a) - to_ordered(b)).unsigned_abs()
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #2624: Verify silu_chord produces f64-quality results for narrow intervals.
    /// For intervals with width in [1e-8, 1e-4], the chord slope must match
    /// a pure f64 reference within 1 ULP.
    #[test]
    fn proptest_silu_chord_f64_precision(l in -10.0f32..10.0, width_exp in -8.0f64..-4.0) {
        let delta = 10.0_f64.powf(width_exp) as f32;
        let u = l + delta;
        prop_assume!(u > l); // skip if delta vanishes in f32
        prop_assume!(
            (u - l).abs() >= 1.0e-8,
            "skip rounded point intervals that violate silu_chord precondition"
        );

        let (slope, lower_intercept, upper_intercept) = math::silu_chord(l, u);

        // Independent f64 reference
        let l64 = l as f64;
        let u64 = u as f64;
        let fl64 = silu_f64_reference(l64);
        let fu64 = silu_f64_reference(u64);
        let ref_slope64 = (fu64 - fl64) / (u64 - l64);
        let ref_intercept64 = fl64 - ref_slope64 * l64;
        let ref_slope = ref_slope64 as f32;
        let ref_intercept = ref_intercept64 as f32;

        let slope_ulps = ulp_distance(slope, ref_slope);

        prop_assert!(
            slope_ulps <= 1,
            "SiLU chord slope not within 1 ULP: got {slope} vs ref {ref_slope} \
             ({slope_ulps} ULPs apart) for [{l}, {u}]"
        );
        // With directed rounding, lower_intercept <= ref <= upper_intercept.
        prop_assert!(
            lower_intercept <= ref_intercept,
            "SiLU chord lower_intercept {lower_intercept} > ref {ref_intercept} for [{l}, {u}]"
        );
        prop_assert!(
            upper_intercept >= ref_intercept,
            "SiLU chord upper_intercept {upper_intercept} < ref {ref_intercept} for [{l}, {u}]"
        );
    }

    /// #2434: Verify silu_sound_linear_relaxation produces strictly sound bounds.
    /// For random intervals, the lower bound must satisfy
    ///   lower_slope * x + lower_intercept <= SiLU(x)  for all x in [l, u]
    /// and the upper bound must satisfy
    ///   upper_slope * x + upper_intercept >= SiLU(x)  for all x in [l, u]
    /// with NO positive tolerance. Directed rounding (#3146) guarantees this.
    ///
    /// Evaluation is in f64 to test the mathematical relationship between the
    /// f32 coefficients and SiLU. The f32 evaluation of `a*x + b` introduces
    /// ~1 ULP rounding that is separate from relaxation soundness — it's a
    /// CROWN propagation concern, not a per-layer relaxation concern.
    ///
    /// Ref: alpha-beta-CROWN check_lower/check_upper use exact <= / >= comparisons.
    #[test]
    fn proptest_silu_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = silu_sound_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        // Skip NaN fallback (infinite bounds).
        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        // Dense grid: 200 points, evaluated in f64 for mathematical precision.
        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = silu_f64_reference(x);

            // Evaluate bounds in f64: exact representation of f32 coefficients
            // applied to the test point. This verifies the mathematical
            // relationship, not f32 evaluation rounding.
            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "SiLU lower bound UNSOUND at x={x}: {lower_val} > SiLU({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "SiLU upper bound UNSOUND at x={x}: {upper_val} < SiLU({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }

    /// #2846: Verify silu_tangent produces f64-quality results at large |d|.
    /// When |d| is large, SiLU(d) ≈ d and slope*d ≈ d, so the subtraction
    /// silu_eval(d) - slope*d loses most significant digits in f32. The f64
    /// intermediate upgrade must preserve intercept precision within 1 ULP.
    #[test]
    fn proptest_silu_tangent_f64_precision(d in -20.0f32..20.0) {
        // Use max_abs_x = |d| + 1 as a reasonable evaluation range.
        let max_abs_x = d.abs() + 1.0;
        let (slope, lower_intercept, upper_intercept) =
            math::silu_tangent(d, max_abs_x);

        // Independent f64 reference for SiLU derivative and tangent
        let d64 = d as f64;
        let s64 = if d64 >= 0.0 {
            1.0_f64 / (1.0 + (-d64).exp())
        } else {
            let ex = d64.exp();
            ex / (1.0 + ex)
        };
        let ref_slope64 = s64 * (1.0 + d64 * (1.0 - s64));
        let ref_eval64 = d64 * s64;
        let ref_intercept64 = ref_eval64 - ref_slope64 * d64;
        let ref_slope = ref_slope64 as f32;
        let ref_intercept = ref_intercept64 as f32;

        let slope_ulps = ulp_distance(slope, ref_slope);

        prop_assert!(
            slope_ulps <= 1,
            "SiLU tangent slope not within 1 ULP: got {slope} vs ref {ref_slope} \
             ({slope_ulps} ULPs apart) at d={d}"
        );
        // With directed rounding, lower_intercept <= ref <= upper_intercept.
        prop_assert!(
            lower_intercept <= ref_intercept,
            "SiLU tangent lower_intercept {lower_intercept} > ref {ref_intercept} at d={d}"
        );
        prop_assert!(
            upper_intercept >= ref_intercept,
            "SiLU tangent upper_intercept {upper_intercept} < ref {ref_intercept} at d={d}"
        );
    }
}
