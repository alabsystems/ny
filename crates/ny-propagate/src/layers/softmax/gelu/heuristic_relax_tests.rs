// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::tests::assert_relaxation_sound;

// =========================================================================
// Chord relaxation
// =========================================================================

/// Chord relaxation should produce sound bounds on a typical interval.
#[test]
fn test_chord_relaxation_soundness() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let r = gelu_linear_relaxation(-1.5, 1.5, approx);
        assert_relaxation_sound(-1.5, 1.5, r.into(), |x| gelu_eval(x, approx), 1e-4, "Chord");
    }
}

/// Chord on a purely positive interval.
#[test]
fn test_chord_relaxation_positive_interval() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let r = gelu_linear_relaxation(0.5, 3.0, approx);
        assert_relaxation_sound(0.5, 3.0, r.into(), |x| gelu_eval(x, approx), 1e-4, "Chord+");
    }
}

/// Chord on a purely negative interval.
#[test]
fn test_chord_relaxation_negative_interval() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let r = gelu_linear_relaxation(-3.0, -0.5, approx);
        assert_relaxation_sound(
            -3.0,
            -0.5,
            r.into(),
            |x| gelu_eval(x, approx),
            1e-4,
            "Chord-",
        );
    }
}

/// Point interval should return sound derivative-based relaxation.
/// After directed rounding (#3329), lower/upper intercepts intentionally
/// differ by 2*slope_err + 2 ULP. Test soundness at zero tolerance.
#[test]
fn test_chord_relaxation_point_interval() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let r = gelu_linear_relaxation(1.0, 1.0, approx);
        // Slopes must be equal (same tangent slope for both bounds).
        assert!(
            (r.0 - r.2).abs() < 1e-6,
            "{approx:?}: slopes differ: ls={}, us={}",
            r.0,
            r.2,
        );
        // Soundness: zero tolerance — directed rounding alone suffices.
        assert_relaxation_sound(
            1.0,
            1.0,
            r.into(),
            |x| gelu_eval(x, approx),
            0.0,
            &format!("{approx:?} point"),
        );
    }
}

// =========================================================================
// Tangent relaxation
// =========================================================================

/// Tangent relaxation via adaptive dispatch.
#[test]
fn test_tangent_relaxation_soundness() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let r = adaptive_gelu_linear_relaxation(-1.0, 1.0, approx, RelaxationMode::Tangent);
        assert_relaxation_sound(
            -1.0,
            1.0,
            r.into(),
            |x| gelu_eval(x, approx),
            1e-4,
            "Tangent",
        );
    }
}

// =========================================================================
// TwoSlope relaxation
// =========================================================================

/// TwoSlope relaxation should be sound.
#[test]
fn test_two_slope_relaxation_soundness() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let r = adaptive_gelu_linear_relaxation(-2.0, 2.0, approx, RelaxationMode::TwoSlope);
        assert_relaxation_sound(
            -2.0,
            2.0,
            r.into(),
            |x| gelu_eval(x, approx),
            1e-4,
            "TwoSlope",
        );
    }
}

// =========================================================================
// Adaptive relaxation
// =========================================================================

/// Adaptive mode should be sound.
#[test]
fn test_adaptive_relaxation_soundness() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let r = adaptive_gelu_linear_relaxation(-1.5, 1.5, approx, RelaxationMode::Adaptive);
        assert_relaxation_sound(
            -1.5,
            1.5,
            r.into(),
            |x| gelu_eval(x, approx),
            1e-4,
            "Adaptive",
        );
    }
}

/// Adaptive mode should produce bounds at least as tight as chord.
#[test]
fn test_adaptive_at_least_as_tight_as_chord() {
    let l = -1.0;
    let u = 1.0;
    let c = f32::midpoint(l, u);
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let chord = gelu_linear_relaxation(l, u, approx);
        let adaptive = adaptive_gelu_linear_relaxation(l, u, approx, RelaxationMode::Adaptive);

        let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
        let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);
        assert!(
            adaptive_width <= chord_width + 1e-5,
            "{approx:?}: adaptive width {adaptive_width} > chord width {chord_width}"
        );
    }
}

// =========================================================================
// Infinite/NaN bounds
// =========================================================================

/// All modes should handle infinite bounds without panicking.
#[test]
fn test_all_modes_handle_infinite_bounds() {
    for mode in [
        RelaxationMode::Chord,
        RelaxationMode::Tangent,
        RelaxationMode::TwoSlope,
        RelaxationMode::Adaptive,
    ] {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let (ls, _li, us, _ui) =
                adaptive_gelu_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY, approx, mode);
            assert!(
                !ls.is_nan(),
                "{mode:?}/{approx:?} (-inf, +inf): lower slope is NaN"
            );
            assert!(
                !us.is_nan(),
                "{mode:?}/{approx:?} (-inf, +inf): upper slope is NaN"
            );
        }
    }
}

/// NaN bounds should return maximally loose for all modes.
#[test]
fn test_all_modes_handle_nan_bounds() {
    for mode in [
        RelaxationMode::Chord,
        RelaxationMode::Tangent,
        RelaxationMode::TwoSlope,
        RelaxationMode::Adaptive,
    ] {
        let (ls, li, us, ui) =
            adaptive_gelu_linear_relaxation(f32::NAN, 1.0, GeluApproximation::Erf, mode);
        assert_eq!(ls, 0.0, "{mode:?}: NaN lower slope");
        assert_eq!(li, f32::NEG_INFINITY, "{mode:?}: NaN lower intercept");
        assert_eq!(us, 0.0, "{mode:?}: NaN upper slope");
        assert_eq!(ui, f32::INFINITY, "{mode:?}: NaN upper intercept");
    }
}

// =========================================================================
// Various intervals for coverage
// =========================================================================

/// Sweep multiple intervals across the real line.
#[test]
fn test_chord_soundness_sweep() {
    let intervals: Vec<(f32, f32)> = vec![
        (-5.0, -3.0),
        (-3.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 3.0),
        (3.0, 5.0),
        (-0.5, 0.5),
        (-2.0, 0.0),
        (0.0, 2.0),
    ];
    for (l, u) in intervals {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let r = gelu_linear_relaxation(l, u, approx);
            assert_relaxation_sound(
                l,
                u,
                r.into(),
                |x| gelu_eval(x, approx),
                1e-4,
                &format!("Chord [{l},{u}]"),
            );
        }
    }
}

// =========================================================================
// Directed rounding regression (#3329)
// =========================================================================

/// Narrow intervals just above the 1e-8 early-return threshold trigger
/// the chord computation (gu - gl) / (u - l). In f32, both numerator and
/// denominator approach 0, causing catastrophic cancellation.
/// The f64 upgrade + directed rounding (#3329) prevents this.
#[test]
fn test_heuristic_narrow_interval_directed_rounding() {
    let test_centers: &[f32] = &[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    let half_width: f32 = 5e-6;
    for &c in test_centers {
        let l = c - half_width;
        let u = c + half_width;
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            for mode in [
                RelaxationMode::Chord,
                RelaxationMode::Tangent,
                RelaxationMode::TwoSlope,
            ] {
                let (ls, li, us, ui) = adaptive_gelu_linear_relaxation(l, u, approx, mode);
                assert!(
                    ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite(),
                    "{mode:?}/{approx:?} narrow [{l},{u}]: NaN/Inf: \
                     ls={ls}, li={li}, us={us}, ui={ui}"
                );
                // Verify soundness with zero tolerance — directed rounding
                // must make bounds provably contain GELU without tolerance.
                assert_relaxation_sound(
                    l,
                    u,
                    (ls, li, us, ui).into(),
                    |x| gelu_eval(x, approx),
                    0.0,
                    &format!("{mode:?}/{approx:?} narrow [{l},{u}]"),
                );
            }
        }
    }
}

/// Soundness sweep for ALL modes across varied intervals, including
/// asymmetric intervals that `test_chord_soundness_sweep` does not cover.
#[test]
fn test_all_modes_soundness_sweep() {
    let intervals: &[(f32, f32)] = &[
        (-5.0, -3.0),
        (-3.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 3.0),
        (3.0, 5.0),
        (-0.5, 0.5),
        (-2.0, 0.0),
        (0.0, 2.0),
        (-1.0, 2.0),
        (-2.0, 1.0),
    ];
    for &(l, u) in intervals {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            for mode in [
                RelaxationMode::Chord,
                RelaxationMode::Tangent,
                RelaxationMode::TwoSlope,
                RelaxationMode::Adaptive,
            ] {
                let r = adaptive_gelu_linear_relaxation(l, u, approx, mode);
                assert_relaxation_sound(
                    l,
                    u,
                    r.into(),
                    |x| gelu_eval(x, approx),
                    1e-4,
                    &format!("{mode:?}/{approx:?} [{l},{u}]"),
                );
            }
        }
    }
}

// =========================================================================
// Proptest soundness envelope (#2161)
// =========================================================================
//
// Heuristic relaxations use sampling + epsilon for soundness approximation.
// They are NOT mathematically sound — the doc says "heuristic, not a proof
// of global soundness." These proptests verify that the sampling provides
// sufficient coverage for practical use, catching gross regressions.
//
// Known precision limits from proptest discovery (worst case, 500 trials):
//   Chord (100 samples):  ~1.2e-3 gap on 10-unit intervals
//   Tangent (50 samples): ~5.9e-4 gap on 10-unit intervals
//   TwoSlope (30 samples): ~6.1e-3 gap on 9.5-unit Tanh intervals
//   Adaptive: inherits from selected strategy
//
// Gap grows with interval width (fewer samples per unit of width).
// Tests use width <= 10 (practical CROWN range). For wider intervals
// the sampling density drops below useful levels.
// For mathematically sound bounds, use gelu_sound_linear_relaxation.

use super::super::eval::{gelu_erf_f64, gelu_tanh_f64};
use proptest::prelude::*;

/// Per-mode tolerances based on observed worst-case gaps (width <= 10):
///   Chord (100 samples):  ~1.2e-3 → 2.5e-3  (~2x headroom)
///   Tangent (50 samples): ~1.63e-3 → 2.5e-3  (~1.5x headroom)
///   TwoSlope (30 samples): ~8.42e-3 → 1.2e-2  (~1.4x headroom)
///   Adaptive: inherits worst case (TwoSlope) → 1.2e-2
/// Per-mode tolerances catch regressions that a single tolerance would miss.
const CHORD_TOL: f64 = 2.5e-3;
const TANGENT_TOL: f64 = 2.5e-3;
const TWO_SLOPE_TOL: f64 = 1.2e-2;
const ADAPTIVE_TOL: f64 = 1.2e-2;

/// Independent f64 GELU reference dispatch for proptest.
fn gelu_f64_ref(x: f64, approx: GeluApproximation) -> f64 {
    match approx {
        GeluApproximation::Erf => gelu_erf_f64(x),
        GeluApproximation::Tanh => gelu_tanh_f64(x),
    }
}

/// Assert heuristic relaxation bounds on a 200-point f64 probe grid.
/// `tol` = mode-specific tolerance for sampling approximation gaps.
/// `bounds` = (lower_slope, lower_intercept, upper_slope, upper_intercept).
fn assert_heuristic_soundness_f64(
    l: f32,
    u: f32,
    bounds: (f32, f32, f32, f32),
    approx: GeluApproximation,
    tol: f64,
    label: &str,
) -> Result<(), TestCaseError> {
    let (ls, li, us, ui) = bounds;
    for k in 0..=200 {
        let t = k as f64 / 200.0;
        let x = (l as f64) + t * (u as f64 - l as f64);
        let x = x.clamp(l as f64, u as f64);
        let fx = gelu_f64_ref(x, approx);

        let lower_val = ls as f64 * x + li as f64;
        prop_assert!(
            lower_val <= fx + tol,
            "{label} lower UNSOUND at x={x}: {lower_val} > GELU({x})={fx} + tol({tol}), \
             interval=[{l}, {u}], gap={}",
            lower_val - fx
        );

        let upper_val = us as f64 * x + ui as f64;
        prop_assert!(
            upper_val >= fx - tol,
            "{label} upper UNSOUND at x={x}: {upper_val} < GELU({x})={fx} - tol({tol}), \
             interval=[{l}, {u}], gap={}",
            fx - upper_val
        );
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// #2161: Chord (heuristic) relaxation soundness envelope.
    #[test]
    fn proptest_heuristic_chord_soundness(
        l in -10.0f32..10.0,
        width in 0.001f32..10.0,
    ) {
        let u = l + width;
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let bounds = gelu_linear_relaxation(l, u, approx);
            prop_assume!(bounds.0.is_finite() && bounds.1.is_finite() && bounds.2.is_finite() && bounds.3.is_finite());
            assert_heuristic_soundness_f64(
                l, u, bounds, approx, CHORD_TOL,
                &format!("Chord/{approx:?}"),
            )?;
        }
    }

    /// #2161: Chord soundness on asymmetric intervals near GELU inflection.
    #[test]
    fn proptest_heuristic_chord_asymmetric_soundness(
        center in -3.0f32..3.0,
        half_width in 0.001f32..5.0,
        skew in -0.9f32..0.9,
    ) {
        let l = center - half_width * (1.0 - skew);
        let u = center + half_width * (1.0 + skew);
        prop_assume!(u > l);
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let bounds = gelu_linear_relaxation(l, u, approx);
            prop_assume!(bounds.0.is_finite() && bounds.1.is_finite() && bounds.2.is_finite() && bounds.3.is_finite());
            assert_heuristic_soundness_f64(
                l, u, bounds, approx, CHORD_TOL,
                &format!("Chord-asym/{approx:?}"),
            )?;
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// #2161: Tangent relaxation soundness envelope.
    #[test]
    fn proptest_heuristic_tangent_soundness(
        l in -10.0f32..10.0,
        width in 0.001f32..10.0,
    ) {
        let u = l + width;
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let bounds = gelu_tangent_relaxation(l, u, approx);
            prop_assume!(bounds.0.is_finite() && bounds.1.is_finite() && bounds.2.is_finite() && bounds.3.is_finite());
            assert_heuristic_soundness_f64(
                l, u, bounds, approx, TANGENT_TOL,
                &format!("Tangent/{approx:?}"),
            )?;
        }
    }

    /// #2161: TwoSlope relaxation soundness envelope.
    #[test]
    fn proptest_heuristic_two_slope_soundness(
        l in -10.0f32..10.0,
        width in 0.001f32..10.0,
    ) {
        let u = l + width;
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let bounds = gelu_two_slope_relaxation(l, u, approx);
            prop_assume!(bounds.0.is_finite() && bounds.1.is_finite() && bounds.2.is_finite() && bounds.3.is_finite());
            assert_heuristic_soundness_f64(
                l, u, bounds, approx, TWO_SLOPE_TOL,
                &format!("TwoSlope/{approx:?}"),
            )?;
        }
    }

    /// #2161: Adaptive mode selection soundness envelope.
    #[test]
    fn proptest_heuristic_adaptive_soundness(
        l in -10.0f32..10.0,
        width in 0.001f32..10.0,
    ) {
        let u = l + width;
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let bounds = adaptive_gelu_linear_relaxation(
                l, u, approx, RelaxationMode::Adaptive,
            );
            prop_assume!(bounds.0.is_finite() && bounds.1.is_finite() && bounds.2.is_finite() && bounds.3.is_finite());
            assert_heuristic_soundness_f64(
                l, u, bounds, approx, ADAPTIVE_TOL,
                &format!("Adaptive/{approx:?}"),
            )?;
        }
    }

    /// #2161: Adaptive produces bounds no wider than chord at center.
    #[test]
    fn proptest_adaptive_no_wider_than_chord(
        l in -10.0f32..10.0,
        width in 0.01f32..10.0,
    ) {
        let u = l + width;
        let c = f32::midpoint(l, u);
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let chord = gelu_linear_relaxation(l, u, approx);
            let adaptive = adaptive_gelu_linear_relaxation(
                l, u, approx, RelaxationMode::Adaptive,
            );
            prop_assume!(
                chord.0.is_finite() && chord.1.is_finite()
                && chord.2.is_finite() && chord.3.is_finite()
            );
            prop_assume!(
                adaptive.0.is_finite() && adaptive.1.is_finite()
                && adaptive.2.is_finite() && adaptive.3.is_finite()
            );

            let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
            let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);
            prop_assert!(
                adaptive_width <= chord_width + 1e-5,
                "Adaptive/{approx:?} wider than chord: {adaptive_width} > {chord_width} \
                 on [{l}, {u}]"
            );
        }
    }
}
