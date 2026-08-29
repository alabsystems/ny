// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::activations::LinearRelaxation;
use crate::layers::{
    gelu_eval, gelu_sound_linear_relaxation, gelu_tanh_inflection_point,
    gelu_tanh_sound_linear_relaxation, pow2_linear_relaxation, GeluApproximation,
};
use crate::tests::assert_relaxation_sound_exact;
use proptest::prelude::*;

/// Independent f64 GELU (erf) reference for strict proptest. (#3292)
/// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
fn gelu_erf_f64_reference(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let inv_sqrt2: f64 = 1.0 / 2.0_f64.sqrt();
    0.5 * x * (1.0 + libm::erf(x * inv_sqrt2))
}

/// Independent f64 GELU (tanh approximation) reference for strict proptest. (#3292)
/// GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_tanh_f64_reference(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

/// ZERO-tolerance f64 pin on GELU's CONSTANT / SATURATING branch, both
/// approximations.
///
/// `gelu_infinite_bounds_relaxation` (eval.rs) publishes, for `l = -inf` and
/// finite `u`, the constants `min(f_crit, f(u), 0)` and `max(f(u), 0)`. The two
/// existing strict proptests in this file draw finite `l, u` and then
/// `prop_assume!` the coefficients finite, so neither can enter this branch; the
/// tolerant grid helpers compare f32 against `gelu_eval`, the f32 evaluator, so
/// neither can see a real-arithmetic violation. Nothing was checking these lines.
///
/// What this pins, with margin: on `(-inf, u]` with `u <= -1` the certified
/// lower bound is `f_crit` — GELU at its own minimiser, about -0.16997 (erf) and
/// -0.17004 (tanh) — while the tightest the function gets on that domain is
/// GELU(-1), about -0.15866. The margin is 1.1e-02, five orders above any f32
/// rounding, so a failure here would mean a structural break (a wrong constant,
/// a wrong branch, a vacuity regression), not a marginal ULP.
///
/// COVERAGE PAST THE MINIMISER — the gap this doc used to name is now closed.
///
/// The former gap was that every case had `u <= -1`, so neither of the two
/// round-to-nearest constants could bind: `f_crit` only binds once the minimiser
/// is inside `(-inf, u]`, and `gelu_eval(u)` only binds as an upper bound for
/// `u > 0`. Both are now computed as f64 witnesses and rounded outward per
/// direction (`gelu/eval.rs`, commit 31741b290), so the `u > -1` cases below are
/// meaningful rather than vacuous, and they are the only cases in this file that
/// exercise those two constants at all.
///
/// The duplicate in `crates/ny-relaxation/src/gelu/eval.rs` has now had the same
/// hunk applied, so the two crates no longer diverge on this soundness property.
/// `ny-relaxation/tests/drift.rs` binds them: `drift_gelu_sound` compares the
/// bands bit-for-bit, and
/// `drift_gelu_infinite_lower_branch_is_exact_at_the_subnormal_endpoint` pins
/// the real-arithmetic fact on the mirror side so the two cannot drift
/// TOGETHER.
///
/// The witness that settled it needs no assumption about f32 `erff` at all:
/// at `u = f32::from_bits(1) = 1.401298464324817e-45` the erff argument is
/// subnormal, so `1.0f32 + erff(..) == 1.0f32` exactly under any
/// implementation, and `0.5f32 * f32::from_bits(1)` is exactly half an ULP and
/// rounds to +0.0. The old upper intercept was therefore `max(0.0, 0.0) = 0.0`
/// against a true `GELU(u) = 7.0064923216240854e-46 > 0`.
///
/// STILL NOT COVERED, and named so it is not mistaken for settled: the f64
/// evaluator itself loses the erf term for tiny `u`. At
/// `u = 5.877471754111438e-39`, `1.0 + erf(u/sqrt(2))` rounds to `1.0` in f64,
/// so `gelu_eval_f64(u)` returns exactly `0.5 * u`, which is
/// 1.3781331109198995e-77 BELOW the true GELU(u); the directed cast cannot
/// recover an error already made in the witness. That residual is present in
/// BOTH crates identically (so it is invisible to a drift test), is ~4.7e-39 in
/// relative terms, and is a separate finding from the one this test pins.
#[test]
fn gelu_saturating_branches_are_exact_in_f64() {
    let deep_to_1: &[f64] = &[-1000.0, -100.0, -50.0, -20.0, -5.0, -2.0, -1.0];
    let deep_to_5: &[f64] = &[-1000.0, -100.0, -50.0, -20.0, -5.0];

    let check = |l: f32,
                 u: f32,
                 coefficients: (f32, f32, f32, f32),
                 reference: fn(f64) -> f64,
                 samples: &[f64],
                 label: &str| {
        let r: LinearRelaxation = coefficients.into();
        // Non-vacuity: `assert_relaxation_sound_exact` deliberately does not
        // assert it, and this very function returns the (0, -inf, 0, +inf)
        // plane for `u = +inf`, which would pass every sample.
        assert_eq!(
            r.lower_slope, 0.0,
            "{label}: expected a CONSTANT lower bound"
        );
        assert_eq!(
            r.upper_slope, 0.0,
            "{label}: expected a CONSTANT upper bound"
        );
        assert!(
            r.lower_intercept.is_finite() && r.upper_intercept.is_finite(),
            "{label}: relaxation went vacuous ({r:?})"
        );
        assert_relaxation_sound_exact(l, u, r, reference, samples, label);
    };

    check(
        f32::NEG_INFINITY,
        -1.0,
        gelu_sound_linear_relaxation(f32::NEG_INFINITY, -1.0),
        gelu_erf_f64_reference,
        deep_to_1,
        "GELU(erf) [-inf, -1]",
    );
    check(
        f32::NEG_INFINITY,
        -5.0,
        gelu_sound_linear_relaxation(f32::NEG_INFINITY, -5.0),
        gelu_erf_f64_reference,
        deep_to_5,
        "GELU(erf) [-inf, -5]",
    );
    check(
        f32::NEG_INFINITY,
        -1.0,
        gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, -1.0),
        gelu_tanh_f64_reference,
        deep_to_1,
        "GELU(tanh) [-inf, -1]",
    );
    check(
        f32::NEG_INFINITY,
        -5.0,
        gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, -5.0),
        gelu_tanh_f64_reference,
        deep_to_5,
        "GELU(tanh) [-inf, -5]",
    );

    // PAST THE MINIMISER. Everything above has `u <= -1`, where the binding
    // lower bound is GELU(u) itself and the two round-to-nearest constants the
    // old code published never had to be right. These three do bind them.
    //
    // The erf minimiser is included as a sample in every case because that is
    // where the LOWER intercept is tight: GELU(x*) = -0.1699712074799036617 at
    // x* = -0.7517915246935644575. Sampling only a uniform grid would step over
    // it, and `l = -inf` makes a uniform grid impossible anyway.
    let x_star = -0.751_791_524_693_564_5_f64;

    // u = -0.5: the minimiser is inside (-inf, u], so `f_crit` binds as the
    // certified lower bound and nothing else does.
    let past_min: &[f64] = &[-1000.0, -100.0, -5.0, -2.0, x_star, -0.6, -0.5];
    check(
        f32::NEG_INFINITY,
        -0.5,
        gelu_sound_linear_relaxation(f32::NEG_INFINITY, -0.5),
        gelu_erf_f64_reference,
        past_min,
        "GELU(erf) [-inf, -0.5]",
    );
    check(
        f32::NEG_INFINITY,
        -0.5,
        gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, -0.5),
        gelu_tanh_f64_reference,
        past_min,
        "GELU(tanh) [-inf, -0.5]",
    );

    // u = 1.0 > 0: now the UPPER intercept is GELU(u) rather than 0, so the
    // second of the two constants binds — at the interval's own endpoint, which
    // is why `1.0` is in the sample list.
    let past_zero: &[f64] = &[-1000.0, -5.0, x_star, -0.1, 0.0, 0.5, 1.0];
    check(
        f32::NEG_INFINITY,
        1.0,
        gelu_sound_linear_relaxation(f32::NEG_INFINITY, 1.0),
        gelu_erf_f64_reference,
        past_zero,
        "GELU(erf) [-inf, 1]",
    );
    check(
        f32::NEG_INFINITY,
        1.0,
        gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, 1.0),
        gelu_tanh_f64_reference,
        past_zero,
        "GELU(tanh) [-inf, 1]",
    );

    // u = the smallest positive subnormal f32. This is the case that needs no
    // assumption about f32 `erff`/`tanh` rounding, and it is what settled the
    // question this test's doc used to leave open: `gelu_eval(u)` is forced to
    // +0.0 there (`1.0f32 + subnormal == 1.0f32`, then `0.5f32 * u` is exactly
    // half an ULP and rounds to even), so the old upper intercept was 0.0 while
    // GELU(u) = 7.0064923216240854e-46 is strictly positive. Sampling `u` itself
    // is the whole point — the violation is at the endpoint and nowhere else.
    let tiny_u = f32::from_bits(1);
    let to_subnormal: &[f64] = &[-1000.0, -5.0, x_star, -1e-10, 0.0, f64::from(tiny_u)];
    check(
        f32::NEG_INFINITY,
        tiny_u,
        gelu_sound_linear_relaxation(f32::NEG_INFINITY, tiny_u),
        gelu_erf_f64_reference,
        to_subnormal,
        "GELU(erf) [-inf, f32::from_bits(1)]",
    );
    check(
        f32::NEG_INFINITY,
        tiny_u,
        gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, tiny_u),
        gelu_tanh_f64_reference,
        to_subnormal,
        "GELU(tanh) [-inf, f32::from_bits(1)]",
    );

    // u = +inf is the deliberate vacuous fallback: GELU(x) -> x, so no finite
    // affine upper envelope exists on [l, +inf). Pin the SHAPE, since the
    // soundness of a +-inf plane is trivial and would mask a regression.
    for (label, coefficients) in [
        (
            "GELU(erf) [-5, +inf]",
            gelu_sound_linear_relaxation(-5.0, f32::INFINITY),
        ),
        (
            "GELU(tanh) [-5, +inf]",
            gelu_tanh_sound_linear_relaxation(-5.0, f32::INFINITY),
        ),
    ] {
        let r: LinearRelaxation = coefficients.into();
        assert!(
            r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative(),
            "{label}: expected the vacuous -inf lower plane, got {}",
            r.lower_intercept
        );
        assert!(
            r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive(),
            "{label}: expected the vacuous +inf upper plane, got {}",
            r.upper_intercept
        );
    }
}

fn assert_sound_interval(l: f32, u: f32) {
    let (ls, li, us, ui) = gelu_sound_linear_relaxation(l, u);

    // Probe a dense-ish grid; the implementation is intended to be provably sound,
    // but this test is a pragmatic regression check.
    let n = 2000;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = l + (u - l) * t;
        let y = gelu_eval(x, GeluApproximation::Erf);
        let lo = ls * x + li;
        let up = us * x + ui;
        assert!(
            lo <= y + 1e-5,
            "lower violated at x={x}: lo={lo} > y={y} on [{l}, {u}]"
        );
        assert!(
            y <= up + 1e-5,
            "upper violated at x={x}: y={y} > up={up} on [{l}, {u}]"
        );
    }
}

fn assert_sound_interval_tanh(l: f32, u: f32) {
    let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(l, u);

    let n = 2000;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = l + (u - l) * t;
        let y = gelu_eval(x, GeluApproximation::Tanh);
        let lo = ls * x + li;
        let up = us * x + ui;
        assert!(
            lo <= y + 1e-5,
            "tanh lower violated at x={x}: lo={lo} > y={y} on [{l}, {u}]"
        );
        assert!(
            y <= up + 1e-5,
            "tanh upper violated at x={x}: y={y} > up={up} on [{l}, {u}]"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_sound_relaxation_smoke() {
    let intervals = [
        // Entirely negative intervals
        (-5.0, -3.0),
        (-2.0, -1.0),
        (-1.0, -0.5),
        // Entirely positive intervals
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        // Cross-zero intervals (all sub-regions)
        (-1.0, 1.0),
        (-3.0, 0.5),
        (-0.5, 3.0),
        (-3.0, 3.0),
        (-2.0, 1.2),
    ];

    for (l, u) in intervals {
        assert_sound_interval(l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_sound_relaxation_edge_cases() {
    let edge_cases = [
        // Near sqrt(2)
        (1.41, 1.42),
        (1.414, 1.415),
        // Near -sqrt(2)
        (-1.42, -1.41),
        (-1.415, -1.414),
        // Near zero
        (-0.1, 0.1),
        (-0.01, 0.01),
        // Very small intervals
        (0.5, 0.501),
        (-0.5, -0.499),
        // Around the GELU minimizer
        (-0.8, -0.7),
        (-1.0, -0.5),
    ];

    for (l, u) in edge_cases {
        assert_sound_interval(l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_tanh_sound_relaxation_smoke() {
    let intervals = [
        (-5.0, -3.0),
        (-2.0, -1.0),
        (-1.0, -0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (-1.0, 1.0),
        (-3.0, 0.5),
        (-0.5, 3.0),
        (-3.0, 3.0),
        (-2.0, 1.2),
    ];

    for (l, u) in intervals {
        assert_sound_interval_tanh(l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_tanh_sound_relaxation_edge_cases() {
    let split = gelu_tanh_inflection_point();
    let delta = 5e-4;
    let edge_cases = [
        // Near tanh-approx inflection point
        (split - 2.0 * delta, split - delta),
        (split - delta, split + delta),
        (-split - delta, -split + delta),
        (-split + delta, -split + 2.0 * delta),
        // Near zero
        (-0.1, 0.1),
        (-0.01, 0.01),
        // Very small intervals
        (0.5, 0.501),
        (-0.5, -0.499),
        // Around the GELU minimizer
        (-0.8, -0.7),
        (-1.0, -0.5),
    ];

    for (l, u) in edge_cases {
        assert_sound_interval_tanh(l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_tanh_inflection_point_sane() {
    let split = gelu_tanh_inflection_point();
    assert!(split.is_finite());
    assert!(
        (1.0..2.0).contains(&split),
        "inflection point out of expected range: {split}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn gelu_sound_relaxation_degenerate_and_nonfinite() {
    // Point interval should produce identical lower/upper lines.
    let (ls, li, us, ui) = gelu_sound_linear_relaxation(1.0, 1.0);
    assert!((ls - us).abs() < 1e-6);
    assert!((li - ui).abs() < 1e-6);

    // Infinite/NaN bounds return maximally loose (NOT identity — identity is unsound, #1837).
    assert_eq!(
        gelu_sound_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY),
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
    );
    assert_eq!(
        gelu_sound_linear_relaxation(f32::NAN, 1.0),
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
    );
}

#[ntest::timeout(10000)]
#[test]
fn gelu_tanh_sound_relaxation_degenerate_and_nonfinite() {
    let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(1.0, 1.0);
    assert!((ls - us).abs() < 1e-6);
    assert!((li - ui).abs() < 1e-6);

    // Infinite/NaN bounds return maximally loose (NOT identity — #1837).
    assert_eq!(
        gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY),
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
    );
    assert_eq!(
        gelu_tanh_sound_linear_relaxation(f32::NAN, 1.0),
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
    );
}

#[ntest::timeout(10000)]
#[test]
fn pow2_subnormal_kani_counterexample_regression_1795() {
    // Values from issue #1795 Kani counterexample.
    let l = -3.743_392e-23f32;
    let u = -1.175_494e-38f32;
    let x = -1.871_839e-23f32;

    assert!(l < u, "counterexample interval must satisfy l < u");
    assert!(
        (l as f64) * (l as f64) < f64::from(f32::MIN_POSITIVE),
        "counterexample should be in subnormal x^2 regime"
    );

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = pow2_linear_relaxation(l, u);

    // Guard behavior: subnormal intervals use a constant zero lower bound.
    assert_eq!(
        lower_slope, 0.0,
        "expected zero lower slope, got {lower_slope:e}"
    );
    assert_eq!(
        lower_intercept, 0.0,
        "expected zero lower intercept, got {lower_intercept:e}"
    );

    let fx = x * x; // underflows to 0.0 in f32 for this counterexample
    let lb = lower_slope * x + lower_intercept;
    let ub = upper_slope * x + upper_intercept;
    assert!(
        lb <= fx,
        "subnormal lower bound must be sound: lb={lb:e}, x2={fx:e}, slope={lower_slope:e}, intercept={lower_intercept:e}"
    );
    assert!(
        ub >= fx,
        "subnormal upper bound must be sound: ub={ub:e}, x2={fx:e}, slope={upper_slope:e}, intercept={upper_intercept:e}"
    );
}

// ── Strict zero-tolerance CROWN relaxation proptests (#3292) ─────────────
//
// Pattern from #3285: f64-evaluated reference with zero tolerance catches
// f32 cancellation bugs invisible to magnitude-scaled tolerance tests.
// GELU (erf) reference: eval::gelu_erf_f64, GELU (tanh) reference: eval::gelu_tanh_f64.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Strict soundness proptest for GELU (erf) CROWN relaxation.
    /// Uses f64 reference (gelu_erf_f64) with zero tolerance on 200-point grid.
    /// Ref: alpha-beta-CROWN auto_LiRPA GELU relaxation, #3292.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_gelu_erf_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let (ls, li, us, ui) = gelu_sound_linear_relaxation(l, u);

        // Skip NaN fallback (infinite bounds).
        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = gelu_erf_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "GELU(erf) lower bound UNSOUND at x={x}: {lower_val} > GELU({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "GELU(erf) upper bound UNSOUND at x={x}: {upper_val} < GELU({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }

    /// Strict soundness proptest for GELU (tanh) CROWN relaxation.
    /// Uses f64 reference (gelu_tanh_f64) with zero tolerance on 200-point grid.
    /// Ref: alpha-beta-CROWN auto_LiRPA GELU tanh approximation, #3292.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_gelu_tanh_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(l, u);

        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = gelu_tanh_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "GELU(tanh) lower bound UNSOUND at x={x}: {lower_val} > GELU_tanh({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "GELU(tanh) upper bound UNSOUND at x={x}: {upper_val} < GELU_tanh({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }
}
