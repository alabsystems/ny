// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::common::TRIG_RELAX_EPS;
use super::cos::cos_linear_relaxation;
use super::sin::sin_linear_relaxation;
use super::tan::tan_linear_relaxation;
use crate::tests::{assert_relaxation_sound, assert_relaxation_sound_exact};
use proptest::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_sin_linear_relaxation_concave() {
    let relaxation = sin_linear_relaxation(0.1, 1.0);
    assert_relaxation_sound(0.1, 1.0, relaxation, f32::sin, 1e-4, "sin");
}

#[ntest::timeout(10000)]
#[test]
fn test_sin_linear_relaxation_convex() {
    let relaxation = sin_linear_relaxation(3.4, 5.0);
    assert_relaxation_sound(3.4, 5.0, relaxation, f32::sin, 1e-4, "sin");
}

#[ntest::timeout(10000)]
#[test]
fn test_sin_linear_relaxation_cross_inflection_falls_back() {
    let r = sin_linear_relaxation(2.5, 4.0);
    assert!(r.lower_slope.abs() < 1e-6 && r.upper_slope.abs() < 1e-6);
    assert!(
        r.lower_intercept <= -1.0 - TRIG_RELAX_EPS * 0.5
            && r.upper_intercept >= 1.0 + TRIG_RELAX_EPS * 0.5
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_linear_relaxation_concave() {
    let relaxation = cos_linear_relaxation(0.1, 1.0);
    assert_relaxation_sound(0.1, 1.0, relaxation, f32::cos, 1e-4, "cos");
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_linear_relaxation_convex() {
    let relaxation = cos_linear_relaxation(2.0, 3.0);
    assert_relaxation_sound(2.0, 3.0, relaxation, f32::cos, 1e-4, "cos");
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_linear_relaxation_cross_inflection_falls_back() {
    let r = cos_linear_relaxation(1.0, 2.0);
    assert!(r.lower_slope.abs() < 1e-6 && r.upper_slope.abs() < 1e-6);
    assert!(
        r.lower_intercept <= -1.0 - TRIG_RELAX_EPS * 0.5
            && r.upper_intercept >= 1.0 + TRIG_RELAX_EPS * 0.5
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_tan_linear_relaxation_sound() {
    for (l, u) in [(-1.2, -0.2), (-0.5, 0.3), (0.2, 1.0)] {
        let relaxation = tan_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::tan, 1e-4, "tan");
    }
}

/// ZERO-tolerance f64 pin on tan's NON-PRINCIPAL branches.
///
/// `tan_interval_has_asymptote` admits every branch `(m*PI - PI/2, m*PI + PI/2)`,
/// but the concavity test compared against `0.0`, which is only that branch's
/// inflection point when `m == 0`. For `m != 0` the tangent line (an UPPER bound
/// on a concave arc) was published as the certified LOWER bound and the secant
/// as the certified UPPER, so both bounds were false simultaneously.
///
/// Each case below is an interval the OLD code accepted and got wrong; the
/// failure margins are enormous, not marginal, so this test discriminates the
/// two versions with no tolerance question at all:
///
///   [ 2.0,  3.0 ]  m =  1   old lower 3.7318958 ABOVE tan at x = 3.0
///   [ 1.6,  1.61]  m =  1   old lower 2.9907299 ABOVE tan at x = 1.61
///   [-3.0, -2.0 ]  m = -1   old upper 3.7318958 BELOW tan at x = -3.0
///   [ 5.0,  6.0 ]  m =  2   (same geometry as [2,3], one branch further out)
///   [ 8.0,  9.0 ]  m =  3
///
/// The samples INCLUDE both endpoints on purpose: on [2, 3] the largest lower
/// violation sits exactly at `x = u`, which a uniform interior grid can miss.
///
/// Non-vacuity is asserted first. `assert_relaxation_sound_exact` deliberately
/// accepts `(0, -inf, 0, +inf)`, so without this a regression that routed these
/// intervals to the constant fallback would pass silently — and the fallback is
/// precisely what the fix uses for the straddling case, so it is a live risk.
#[ntest::timeout(10000)]
#[test]
fn tan_off_principal_branches_are_exact_in_f64() {
    for (l, u) in [
        (2.0f32, 3.0f32),
        (1.6, 1.61),
        (-3.0, -2.0),
        (5.0, 6.0),
        (8.0, 9.0),
    ] {
        let relaxation = tan_linear_relaxation(l, u);
        assert!(
            relaxation.lower_slope.is_finite()
                && relaxation.lower_intercept.is_finite()
                && relaxation.upper_slope.is_finite()
                && relaxation.upper_intercept.is_finite(),
            "tan [{l}, {u}] fell back to constant bounds — this case is supposed \
             to exercise the tangent/secant branch, so a vacuous relaxation \
             would make the soundness check below meaningless"
        );
        let mut samples = vec![f64::from(l), f64::from(u)];
        for i in 0..=200 {
            samples.push(f64::from(l) + (f64::from(u) - f64::from(l)) * f64::from(i) / 200.0);
        }
        assert_relaxation_sound_exact(
            l,
            u,
            relaxation,
            f64::tan,
            &samples,
            &format!("tan off-branch [{l}, {u}]"),
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_tan_linear_relaxation_asymptote_fallback() {
    let r = tan_linear_relaxation(1.4, 1.8);
    assert!(r.lower_slope.abs() < 1e-6 && r.upper_slope.abs() < 1e-6);
    assert!(r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative());
    assert!(r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive());
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    #[ntest::timeout(10000)]
    #[test]
    fn prop_sin_linear_relaxation_sound(l in -20.0f32..20.0, u in -20.0f32..20.0) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = sin_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::sin, 1e-4, "sin");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn prop_cos_linear_relaxation_sound(l in -20.0f32..20.0, u in -20.0f32..20.0) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = cos_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::cos, 1e-4, "cos");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn prop_tan_linear_relaxation_sound(l in -1.4f32..1.4, u in -1.4f32..1.4) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = tan_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::tan, 1e-4, "tan");
    }

    /// The generator range, not the tolerance, was the blind spot.
    ///
    /// Every tan case in this crate was clamped to the principal branch —
    /// `prop_tan_linear_relaxation_sound` above draws `-1.4..1.4`, and the fixed
    /// intervals in `test_tan_linear_relaxation_sound` and (until this change)
    /// `test_tan_crown_backward_soundness` all sat inside `(-PI/2, PI/2)`, as do
    /// `crown_trig_reciprocal.rs`'s `-1.4..1.4` and `-1.2..1.2` draws. The clamp
    /// was chosen to dodge the POLES — `test_tan_crown_backward_soundness` said
    /// so in as many words — and silently also dodged every branch on which the
    /// concavity test was wrong. The violations there ran to 7.2e+06, so no
    /// tolerance was ever the issue.
    ///
    /// This draws `-20.0..20.0`, which spans branches `m = -6 ..= 6`, and
    /// compares in f64 with ZERO tolerance. f64 is required, not a preference:
    /// near a pole the coefficients are large (slope 1172, intercept -1910 on
    /// [1.6, 1.61]), so evaluating `slope * x + intercept` in f32 carries about
    /// 2.4e-04 of the test's OWN rounding — more than the 1e-4 tolerance the
    /// f32 helper uses, which would make a widened f32 check flaky rather than
    /// sensitive. Verified against a 60-dps reference over 18768 finite
    /// relaxations in this range: worst violation 0.000e+00 on both sides.
    ///
    /// Intervals containing a pole are NOT filtered out: they return the
    /// `(0, -inf, 0, +inf)` fallback, which passes trivially and is exactly the
    /// behaviour worth pinning for them.
    #[ntest::timeout(20000)]
    #[test]
    fn prop_tan_linear_relaxation_sound_exact_off_branch(
        l in -20.0f32..20.0,
        u in -20.0f32..20.0,
    ) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = tan_linear_relaxation(l, u);
        // Endpoints included: on [2, 3] the largest lower violation was at
        // x = u exactly, which an interior-only grid can step over.
        let mut samples = vec![f64::from(l), f64::from(u)];
        for i in 0..=64 {
            samples.push(f64::from(l) + (f64::from(u) - f64::from(l)) * f64::from(i) / 64.0);
        }
        assert_relaxation_sound_exact(l, u, relaxation, f64::tan, &samples, "tan wide");
    }
}

// ========================================================================
// CROWN backward soundness tests
//
// These test propagate_linear_with_bounds (the full CROWN backward path),
// not just the per-element relaxation functions. They exercise:
//   - coefficient matrix multiplication via crown_elementwise_backward
//   - sign-dependent slope/intercept swapping for negative A coefficients
//   - BoundedTensor → LinearBounds extraction
//
// Interval choices:
//   Sin: concave region [0.1, 1.0], convex region [3.4, 5.0],
//        crossing inflection [-1.0, 1.0] (falls back to constant bounds)
//   Cos: concave region [0.1, 1.0], convex region [2.0, 3.0],
//        crossing [-0.5, 1.5]
//   Tan: within half-period [-1.2, -0.2], crossing zero [-0.5, 0.3],
//        positive [0.2, 1.0]; all avoid asymptotes
// ========================================================================

// assert_crown_backward_sound extracted to crate::tests::assert_crown_backward_sound
use crate::tests::assert_crown_backward_sound;

#[ntest::timeout(10000)]
#[test]
fn test_sin_crown_backward_soundness() {
    let layer = super::SinLayer::new();
    // Concave region, convex region, and crossing-inflection (constant fallback)
    let intervals = [(0.1, 1.0), (3.4, 5.0), (-1.0, 1.0)];
    assert_crown_backward_sound(&layer, &intervals, f32::sin);
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_crown_backward_soundness() {
    let layer = super::CosLayer::new();
    // Concave region, convex region, and crossing
    let intervals = [(0.1, 1.0), (2.0, 3.0), (-0.5, 1.5)];
    assert_crown_backward_sound(&layer, &intervals, f32::cos);
}

#[ntest::timeout(10000)]
#[test]
fn test_tan_crown_backward_soundness() {
    let layer = super::TanLayer::new();
    // The first three are within (-π/2, π/2) to avoid asymptotes.
    //
    // The last three are asymptote-free intervals on OTHER branches, added
    // because "avoid asymptotes" was being satisfied by never leaving the
    // principal branch — which is also the only branch on which the old
    // `u <= 0.0` / `l >= 0.0` concavity test was right. On [2, 3] the old code
    // published a lower bound 3.73 ABOVE tan and an upper bound 0.46 BELOW it;
    // on [-3, -2] the same pair mirrored. Both are three orders above this
    // helper's 1e-3 tolerance, and all three added intervals are far enough
    // from a pole (coefficients O(10)) that the f32 evaluation here carries no
    // meaningful cancellation error of its own.
    let intervals = [
        (-1.2, -0.2),
        (-0.5, 0.3),
        (0.2, 1.0),
        (2.0, 3.0),
        (-3.0, -2.0),
        (5.0, 6.0),
    ];
    assert_crown_backward_sound(&layer, &intervals, f32::tan);
}

/// Regression: sound-mode Sin path uses constant bounds built from
/// `concretize_sound()`, which now repairs non-finite/inverted elements internally.
#[test]
fn test_sin_sound_mode_constant_bounds_repairs_invalid_concretization() {
    use crate::LinearBounds;
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;

    let layer = super::SinLayer::new().with_sound_mode(true);
    let pre = BoundedTensor::new(arr1(&[5.0]).into_dyn(), arr1(&[5.0]).into_dyn())
        .expect("invariant: valid singleton pre-activation interval");

    // Asymmetric coefficients can induce concretization inversions.
    let bounds = LinearBounds::new(
        arr2(&[[-10.0]]),
        arr1(&[100.0]),
        arr2(&[[10.0]]),
        arr1(&[-100.0]),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre)
        .expect("sound-mode Sin should return conservative constant bounds");

    assert!(
        result.lower_a.iter().all(|&v| v == 0.0),
        "sound-mode constant bounds must zero lower slopes"
    );
    assert!(
        result.upper_a.iter().all(|&v| v == 0.0),
        "sound-mode constant bounds must zero upper slopes"
    );
    assert!(
        result.lower_b[0] <= result.upper_b[0],
        "sound-mode constant bounds must not invert: {} > {}",
        result.lower_b[0],
        result.upper_b[0]
    );
}
