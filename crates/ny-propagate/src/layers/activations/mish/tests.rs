// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::arr1;
use proptest::prelude::*;

/// Check that linear bounds contain Mish(x) at all grid points in [l, u].
fn assert_mish_relaxation_soundness(l: f32, u: f32, tol: f32) {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = mish_linear_relaxation(l, u);
    let n = 101;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = (l + t * (u - l)).clamp(l, u);
        let y = mish_eval(x);
        let lb = ls * x + li;
        let ub = us * x + ui;
        assert!(
            lb <= y + tol,
            "Mish lower bound violated at x={}: lb={} > y={} (l={}, u={})",
            x,
            lb,
            y,
            l,
            u
        );
        assert!(
            ub >= y - tol,
            "Mish upper bound violated at x={}: ub={} < y={} (l={}, u={})",
            x,
            ub,
            y,
            l,
            u
        );
    }
}

// ========== Soundness grid tests for CROWN linear relaxation ==========

#[test]
fn mish_relaxation_positive_soundness() {
    // Purely positive: Mish ≈ x for large positive
    assert_mish_relaxation_soundness(0.5, 5.0, 1e-5);
    assert_mish_relaxation_soundness(1.0, 10.0, 1e-3);
    assert_mish_relaxation_soundness(0.0, 2.0, 1e-5);
}

#[test]
fn mish_relaxation_negative_soundness() {
    // Purely negative: Mish → 0 for large negative
    assert_mish_relaxation_soundness(-5.0, -1.0, 1e-5);
    assert_mish_relaxation_soundness(-10.0, -5.0, 1e-5);
    assert_mish_relaxation_soundness(-2.0, -0.5, 1e-5);
}

#[test]
fn mish_relaxation_crossing_soundness() {
    // Crossing zero: hardest case — non-monotonic near minimum
    assert_mish_relaxation_soundness(-2.0, 2.0, 1e-5);
    assert_mish_relaxation_soundness(-1.0, 1.0, 1e-5);
    assert_mish_relaxation_soundness(-5.0, 5.0, 1e-3);
    assert_mish_relaxation_soundness(-3.0, 1.0, 1e-3);
}

#[test]
fn mish_relaxation_near_minimum_soundness() {
    // Mish minimum is near x ≈ -0.31 (value ≈ -0.309)
    assert_mish_relaxation_soundness(-1.0, 0.0, 1e-5);
    assert_mish_relaxation_soundness(-0.5, -0.1, 1e-5);
    assert_mish_relaxation_soundness(-0.4, -0.2, 1e-5);
}

#[test]
fn mish_relaxation_wide_interval_soundness() {
    assert_mish_relaxation_soundness(-10.0, 10.0, 1e-2);
    assert_mish_relaxation_soundness(-20.0, 20.0, 1e-1);
}

#[test]
fn mish_relaxation_asymmetric_soundness() {
    assert_mish_relaxation_soundness(-0.01, 10.0, 1e-3);
    assert_mish_relaxation_soundness(-10.0, 0.01, 1e-3);
}

#[test]
fn mish_relaxation_point_interval_soundness() {
    for x in [-5.0f32, -1.0, -0.31, 0.0, 0.5, 2.0, 5.0] {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = mish_linear_relaxation(x, x);
        let y = mish_eval(x);
        let lb = ls * x + li;
        let ub = us * x + ui;
        assert!(
            lb <= y + 1e-3,
            "Point lower violated at x={}: lb={} > y={}",
            x,
            lb,
            y
        );
        assert!(
            ub >= y - 1e-3,
            "Point upper violated at x={}: ub={} < y={}",
            x,
            ub,
            y
        );
    }
}

fn f32_any_with_specials() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(f32::NEG_INFINITY),
        Just(f32::INFINITY),
        Just(0.0_f32),
        Just(f32::NAN),
        prop::num::f32::ANY
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]

    /// #1836 acceptance: exercise Mish eval on ANY f32 plus explicit IEEE corner values.
    #[test]
    fn proptest_mish_eval_handles_special_values(x in f32_any_with_specials()) {
        let y = mish_eval(x);
        if x.is_nan() {
            prop_assert!(y.is_nan(), "mish_eval(NaN) should be NaN, got {y}");
        } else if x == f32::NEG_INFINITY {
            prop_assert_eq!(y, 0.0, "mish_eval(-inf) should be 0.0, got {}", y);
        } else if x == f32::INFINITY {
            prop_assert_eq!(y, f32::INFINITY, "mish_eval(+inf) should be +inf, got {}", y);
        } else {
            prop_assert!(!y.is_nan(), "mish_eval({x}) should not be NaN, got {y}");
        }
    }

    /// #1836 acceptance: finite-width IBP intervals must not produce NaN bounds.
    #[test]
    fn proptest_mish_ibp_no_nan_for_finite_intervals(a in prop::num::f32::ANY, b in prop::num::f32::ANY) {
        prop_assume!(a.is_finite() && b.is_finite());
        prop_assume!(a.abs() <= 1.0e6 && b.abs() <= 1.0e6);
        let (l, u) = (a.min(b), a.max(b));
        let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let layer = MishLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        prop_assert!(!lower.is_nan(), "Mish IBP lower is NaN for [{l}, {u}]");
        prop_assert!(!upper.is_nan(), "Mish IBP upper is NaN for [{l}, {u}]");
        prop_assert!(lower <= upper, "Mish IBP bounds inverted for [{l}, {u}]: {lower} > {upper}");
    }
}

/// Regression test for #1836: mish_eval(-inf) must return 0, not NaN.
/// Mish(x) = x * tanh(softplus(x)); at x = -inf: (-inf)*tanh(0) = (-inf)*0 = NaN
/// without the guard. Correct limit: Mish(-inf) = 0.
#[test]
fn test_mish_eval_neg_infinity_returns_zero() {
    let result = mish_eval(f32::NEG_INFINITY);
    assert_eq!(result, 0.0, "mish_eval(-inf) should be 0.0, got {result}");
}

/// Regression test for #1836: mish_eval(+inf) must return +inf.
#[test]
fn test_mish_eval_pos_infinity_returns_pos_infinity() {
    let result = mish_eval(f32::INFINITY);
    assert_eq!(
        result,
        f32::INFINITY,
        "mish_eval(+inf) should be +inf, got {result}"
    );
}

/// Regression test for #1836: mish_eval(NaN) must return NaN.
#[test]
fn test_mish_eval_nan_returns_nan() {
    let result = mish_eval(f32::NAN);
    assert!(
        result.is_nan(),
        "mish_eval(NaN) should be NaN, got {result}"
    );
}

#[test]
fn test_mish_ibp_non_finite_input_falls_back_to_infinite_bounds() -> Result<()> {
    let input =
        BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    let layer = MishLayer::new();
    let output = layer.propagate_ibp(&input)?;

    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];
    assert!(lower.is_infinite() && lower.is_sign_negative());
    assert!(upper.is_infinite() && upper.is_sign_positive());
    Ok(())
}

// ── CROWN backward tests ───────────────────────────────────────────

#[test]
fn test_crown_backward_crossing_soundness() {
    let layer = MishLayer::new();
    let l = -3.0_f32;
    let u = 2.0_f32;
    let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = mish_eval(x);
        assert!(
            la * x + lb <= y + 1e-3,
            "Mish CROWN lower bound violated at x={}: {} > {}",
            x,
            la * x + lb,
            y
        );
        assert!(
            ua * x + ub >= y - 1e-3,
            "Mish CROWN upper bound violated at x={}: {} < {}",
            x,
            ua * x + ub,
            y
        );
    }
}

#[test]
fn test_crown_backward_positive_region() {
    let layer = MishLayer::new();
    let pre = BoundedTensor::new(arr1(&[2.0_f32]).into_dyn(), arr1(&[5.0_f32]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    // In the positive region, Mish ≈ x, so slopes should be near 1
    let la = result.lower_a[[0, 0]];
    let ua = result.upper_a[[0, 0]];
    assert!(
        la > 0.5,
        "Mish positive region lower slope should be positive, got {la}"
    );
    assert!(
        ua > 0.5,
        "Mish positive region upper slope should be positive, got {ua}"
    );

    // Grid soundness check
    for k in 0..=50 {
        let x = 2.0 + 3.0 * (k as f32 / 50.0);
        let y = mish_eval(x);
        assert!(
            result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-3,
            "positive CROWN lb violated at x={x}"
        );
        assert!(
            result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-3,
            "positive CROWN ub violated at x={x}"
        );
    }
}

#[test]
fn test_crown_backward_near_minimum() {
    // Mish minimum is near x ≈ -1.19
    let layer = MishLayer::new();
    let l = -2.0_f32;
    let u = 0.0_f32;
    let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = mish_eval(x);
        assert!(la * x + lb <= y + 1e-3, "near-minimum lb violated at x={x}");
        assert!(ua * x + ub >= y - 1e-3, "near-minimum ub violated at x={x}");
    }
}

#[test]
fn test_crown_backward_multi_neuron() {
    let layer = MishLayer::new();
    let pre = BoundedTensor::new(
        arr1(&[-2.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 4.0]).into_dyn(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    // Check each neuron independently
    for neuron in 0..2 {
        let la = result.lower_a[[neuron, neuron]];
        let lb = result.lower_b[neuron];
        let ua = result.upper_a[[neuron, neuron]];
        let ub = result.upper_b[neuron];
        let lo = pre.lower()[neuron];
        let hi = pre.upper()[neuron];

        for k in 0..=20 {
            let x = lo + (hi - lo) * (k as f32 / 20.0);
            let y = mish_eval(x);
            assert!(
                la * x + lb <= y + 1e-3,
                "neuron {neuron} lb violated at x={x}"
            );
            assert!(
                ua * x + ub >= y - 1e-3,
                "neuron {neuron} ub violated at x={x}"
            );
        }
    }
}

#[test]
fn test_propagate_linear_requires_preact() {
    let layer = MishLayer::new();
    let bounds = LinearBounds::identity(1);
    assert!(
        layer.propagate_linear(&bounds).is_err(),
        "Mish CROWN without pre-activation bounds should fail"
    );
    assert!(layer.requires_pre_activation_bounds());
}

// ── CROWN backward with non-identity incoming bounds ──────────────

/// Tests that CROWN backward correctly handles negative incoming coefficients.
/// With A = [[-1]], the sign-swap logic in crown_elementwise_backward must use
/// the upper relaxation for the lower bound and vice versa.
#[test]
fn test_crown_backward_negative_coeff_soundness() -> Result<()> {
    use ndarray::Array1;

    let layer = MishLayer::new();
    // Crossing interval spanning the Mish minimum region
    let pre = BoundedTensor::new(arr1(&[-3.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())?;

    let neg_bounds = LinearBounds::new(
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_linear_with_bounds(&neg_bounds, &pre)?;

    let l = -3.0_f32;
    let u = 2.0_f32;
    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = -(mish_eval(x));
        let bound_lo = la.max(0.0) * l + la.min(0.0) * u + lb;
        let bound_hi = ua.max(0.0) * u + ua.min(0.0) * l + ub;
        assert!(
            bound_lo <= y + 1e-3,
            "negative coeff: lower {} > -mish({}) = {} at x={}",
            bound_lo,
            x,
            y,
            x
        );
        assert!(
            bound_hi >= y - 1e-3,
            "negative coeff: upper {} < -mish({}) = {} at x={}",
            bound_hi,
            x,
            y,
            x
        );
    }
    Ok(())
}

/// Tests CROWN backward with a 2-neuron, 2-output non-identity coefficient matrix
/// A = [[1, -1], [0.5, 0.5]], verifying that the composed output A @ Mish(x)
/// is soundly bounded. Exercises both positive and negative coefficient branches
/// simultaneously on crossing intervals.
#[test]
fn test_crown_backward_non_identity_bounds() -> Result<()> {
    use ndarray::Array1;

    let layer = MishLayer::new();
    // Neuron 0: crossing with minimum [-2, 1], Neuron 1: positive region [1, 4]
    let pre = BoundedTensor::new(
        arr1(&[-2.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 4.0]).into_dyn(),
    )?;

    let a = ndarray::Array2::from_shape_vec((2, 2), vec![1.0_f32, -1.0, 0.5, 0.5])
        .expect("invariant: static 2x2 shape matches 4 elements");
    let bounds = LinearBounds::new(a.clone(), Array1::zeros(2), a, Array1::zeros(2)).unwrap();
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    let l0 = -2.0_f32;
    let u0 = 1.0_f32;
    let l1 = 1.0_f32;
    let u1 = 4.0_f32;
    let lowers = [l0, l1];
    let uppers = [u0, u1];

    for k0 in 0..=10 {
        for k1 in 0..=10 {
            let x0 = l0 + (u0 - l0) * (k0 as f32 / 10.0);
            let x1 = l1 + (u1 - l1) * (k1 as f32 / 10.0);
            let r0 = mish_eval(x0);
            let r1 = mish_eval(x1);

            let y0 = r0 - r1;
            let y1 = 0.5 * r0 + 0.5 * r1;

            for i in 0..2 {
                let y = if i == 0 { y0 } else { y1 };
                let mut lo = result.lower_b[i];
                let mut hi = result.upper_b[i];
                for j in 0..2 {
                    let la = result.lower_a[[i, j]];
                    let ua = result.upper_a[[i, j]];
                    lo += la.max(0.0) * lowers[j] + la.min(0.0) * uppers[j];
                    hi += ua.max(0.0) * uppers[j] + ua.min(0.0) * lowers[j];
                }
                assert!(
                    lo <= y + 1e-2,
                    "output {} lower {} > true {} at ({}, {})",
                    i,
                    lo,
                    y,
                    x0,
                    x1
                );
                assert!(
                    hi >= y - 1e-2,
                    "output {} upper {} < true {} at ({}, {})",
                    i,
                    hi,
                    y,
                    x0,
                    x1
                );
            }
        }
    }
    Ok(())
}

// ── f64 chord precision proptest (#2624) ─────────────────────────────

/// Reference Mish in f64, independent of the crate implementation.
fn mish_f64_reference(x: f64) -> f64 {
    let softplus = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f64 + x.exp()).ln()
    };
    x * softplus.tanh()
}

/// ULP distance between two f32 values, handling sign correctly.
fn ulp_distance(a: f32, b: f32) -> u64 {
    fn to_ordered(x: f32) -> i64 {
        let bits = x.to_bits() as i32;
        if bits < 0 {
            (0x8000_0000_u32 as i32 - bits) as i64
        } else {
            bits as i64
        }
    }
    (to_ordered(a) - to_ordered(b)).unsigned_abs()
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #2624, #3146: Verify mish_chord_f64 produces f64-quality results with directed rounding.
    /// For intervals with width in [1e-8, 1e-4], the chord slope must match
    /// a pure f64 reference within 1 ULP, and directed rounding bounds must
    /// bracket the reference intercept: lower_intercept <= ref <= upper_intercept.
    #[test]
    fn proptest_mish_chord_f64_precision(l in -10.0f32..10.0, width_exp in -8.0f64..-4.0) {
        let delta = 10.0_f64.powf(width_exp) as f32;
        let u = l + delta;
        prop_assume!(u > l); // skip if delta vanishes in f32

        let (slope, lower_intercept, upper_intercept) = mish_chord_f64(l, u);

        // Independent f64 reference
        let l64 = l as f64;
        let u64 = u as f64;
        let fl64 = mish_f64_reference(l64);
        let fu64 = mish_f64_reference(u64);
        let ref_slope64 = (fu64 - fl64) / (u64 - l64);
        let ref_intercept64 = fl64 - ref_slope64 * l64;
        let ref_slope = ref_slope64 as f32;
        let ref_intercept = ref_intercept64 as f32;

        let slope_ulps = ulp_distance(slope, ref_slope);

        prop_assert!(
            slope_ulps <= 1,
            "Mish chord slope not within 1 ULP: got {slope} vs ref {ref_slope} \
             ({slope_ulps} ULPs apart) for [{l}, {u}]"
        );
        // Directed rounding: lower_intercept <= ref <= upper_intercept
        prop_assert!(
            lower_intercept <= ref_intercept,
            "Mish chord lower_intercept {lower_intercept} > ref {ref_intercept} for [{l}, {u}]"
        );
        prop_assert!(
            upper_intercept >= ref_intercept,
            "Mish chord upper_intercept {upper_intercept} < ref {ref_intercept} for [{l}, {u}]"
        );
    }
}

/// Regression test for #1836: IBP with infinite input bounds must not produce NaN
/// in the computed values. The IBP output constructor (BoundedTensor::new) rejects
/// infinite bounds, so we verify the error is NumericalInstability (not NaN corruption).
/// The eval function fix ensures the intermediate values are correct (0, +inf) not NaN.
#[test]
fn test_mish_ibp_infinite_bounds_no_nan() {
    use ndarray::{ArrayD, IxDyn};
    let layer = MishLayer::new();
    // Use new_unchecked since BoundedTensor::new rejects infinite inputs
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, -10.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 10.0]).unwrap(),
    )
    .unwrap();
    // IBP output may error because mish(+inf) = +inf gets rejected by BoundedTensor::new.
    // The key property: no NaN corruption. Either it succeeds with valid bounds or errors cleanly.
    match layer.propagate_ibp(&input) {
        Ok(result) => {
            for &v in result.lower().iter() {
                assert!(!v.is_nan(), "Mish IBP lower must not be NaN, got {v}");
            }
            for &v in result.upper().iter() {
                assert!(!v.is_nan(), "Mish IBP upper must not be NaN, got {v}");
            }
        }
        Err(e) => {
            // NumericalInstability from BoundedTensor::new rejecting +inf is acceptable
            let msg = format!("{e}");
            assert!(
                msg.contains("NaN or Inf"),
                "Expected NumericalInstability error for inf bounds, got: {msg}"
            );
        }
    }
}

// ── NaN propagation regression tests (#2714) ──────────────────────────

#[test]
fn test_relaxation_nan_lower_returns_nan_fallback_2714() {
    // NaN lower bound must produce nan_fallback intercepts (±inf),
    // not 0.0 from silently absorbed NaN.
    let r = mish_linear_relaxation(f32::NAN, 1.0);
    assert!(
        r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative(),
        "NaN lower should trigger nan_fallback, got lower_intercept={}",
        r.lower_intercept
    );
    assert!(
        r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive(),
        "NaN lower should trigger nan_fallback, got upper_intercept={}",
        r.upper_intercept
    );
}

#[test]
fn test_relaxation_nan_upper_returns_nan_fallback_2714() {
    let r = mish_linear_relaxation(-1.0, f32::NAN);
    assert!(
        r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative(),
        "NaN upper should trigger nan_fallback, got lower_intercept={}",
        r.lower_intercept
    );
}

/// Exhaustive deterministic soundness sweep: every (l, u) on a fine mesh over
/// [-6, 6], 200-point f64 grid per interval, ZERO tolerance. Complements the
/// randomized proptests with a deterministic, reproducible enclosure check that
/// blankets the inflection-crossing region. (~0.9s; ~5.8M point checks.)
#[test]
// The f32-accumulated `li_ += step` mesh IS the documented reproducible sweep;
// re-indexing on integers would shift the mesh points by accumulation rounding.
#[allow(clippy::while_float)]
fn exhaustive_mish_soundness_sweep() {
    let mut checked = 0u64;
    let mut violations = 0u64;
    let step = 0.05_f32;
    let mut li_ = -6.0_f32;
    while li_ <= 6.0 {
        let mut ui_ = li_ + step;
        while ui_ <= 6.0 {
            let r = mish_linear_relaxation(li_, ui_);
            if r.lower_slope.is_finite()
                && r.lower_intercept.is_finite()
                && r.upper_slope.is_finite()
                && r.upper_intercept.is_finite()
            {
                for k in 0..=200 {
                    let t = k as f64 / 200.0;
                    let x =
                        (li_ as f64 + t * (ui_ as f64 - li_ as f64)).clamp(li_ as f64, ui_ as f64);
                    let fx = mish_f64_reference(x);
                    let lo = r.lower_slope as f64 * x + r.lower_intercept as f64;
                    let hi = r.upper_slope as f64 * x + r.upper_intercept as f64;
                    if lo > fx || hi < fx {
                        violations += 1;
                        if violations <= 20 {
                            println!(
                                "VIOLATION [{:.3},{:.3}] x={:.4}: lo={:.6} hi={:.6} mish={:.6}",
                                li_, ui_, x, lo, hi, fx
                            );
                        }
                    }
                    checked += 1;
                }
            }
            ui_ += step;
        }
        li_ += step;
    }
    println!("exhaustive sweep: checked {checked} points, {violations} violations");
    assert_eq!(
        violations, 0,
        "exhaustive sweep found {violations} unsound points"
    );
}

/// Mean linear-band width of a relaxation over [l, u], measured on a fine f64
/// grid. Used to compare tightness of two SOUND relaxations.
fn mean_band_width(r: &LinearRelaxation, l: f32, u: f32) -> f64 {
    let n = 400;
    let mut acc = 0.0f64;
    for k in 0..=n {
        let x = l as f64 + (u as f64 - l as f64) * k as f64 / n as f64;
        let w = (r.upper_slope as f64 * x + r.upper_intercept as f64)
            - (r.lower_slope as f64 * x + r.lower_intercept as f64);
        acc += w;
    }
    acc / (n as f64 + 1.0)
}

/// The region-classified relaxation must never be looser than the historical
/// chord±deviation band: it picks the tighter of {verified region line, band
/// line} per bound, so for every interval its mean width must be <= the band's
/// (modulo a tiny f32 rounding allowance).
#[test]
fn mish_relaxation_never_worse_than_band() {
    let cases = [
        (-5.0_f32, -3.0),
        (-3.0, -1.0),
        (-1.5, 1.0),
        (0.0, 1.0),
        (2.0, 5.0),
        (-4.0, 0.5),
        (0.5, 3.0),
        (1.0, 4.0),
        (-3.0, 2.0),
        (-2.4, 2.0),
        (-6.0, 6.0),
        (-2.256, -1.0),
        (-1.0, 1.491),
    ];
    for (l, u) in cases {
        let new = mish_linear_relaxation(l, u);
        let band = mish_fallback_band(l, u);
        let wn = mean_band_width(&new, l, u);
        let wb = mean_band_width(&band, l, u);
        assert!(
            wn <= wb + 1e-4,
            "region-classified relaxation looser than band on [{l}, {u}]: \
             new={wn} > band={wb}"
        );
    }
}

/// Tightening regression: on an inflection-crossing interval that straddles the
/// right inflection p2 ≈ +1.49 (the cross_right case where the endpoint chord is
/// NOT a valid upper bound), the SiLU-style region-classified relaxation must be
/// STRICTLY tighter than the historical chord±deviation band.
///
/// This is the load-bearing "tighter than before" evidence: [0.5, 3.0] crosses
/// p2, the chord fails as an upper bound, and the right-concave tangent (or the
/// convex-region lower tangent) yields a measurably narrower band.
#[test]
fn mish_crossing_tighter_than_band() {
    let (l, u) = (0.5_f32, 3.0_f32);
    // Confirm this interval really crosses the right inflection point.
    let (_p1, p2) = mish_inflection_points();
    assert!(l < p2 && u > p2, "test interval must straddle p2={p2}");

    let new = mish_linear_relaxation(l, u);
    let band = mish_fallback_band(l, u);
    let wn = mean_band_width(&new, l, u);
    let wb = mean_band_width(&band, l, u);
    // Require a real (>= 2%) improvement, not just noise.
    assert!(
        wn < wb * 0.98,
        "expected crossing interval [{l}, {u}] to be >=2% tighter: new={wn}, band={wb} (ratio {})",
        wn / wb
    );
}

// ── CROWN relaxation soundness proptest (#3285) ─────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #3285: Verify mish_linear_relaxation produces strictly sound bounds.
    /// For random intervals, the lower bound must satisfy
    ///   lower_slope * x + lower_intercept <= Mish(x)  for all x in [l, u]
    /// and the upper bound must satisfy
    ///   upper_slope * x + upper_intercept >= Mish(x)  for all x in [l, u]
    /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
    ///
    /// Ref: SiLU proptest_silu_relaxation_strict_soundness (silu/tests.rs:553).
    #[test]
    fn proptest_mish_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = mish_linear_relaxation(l, u);
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
            let fx = mish_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "Mish lower bound UNSOUND at x={}: {} > Mish({})={}, \
                 interval=[{}, {}], gap={}", x, lower_val, x, fx, l, u, lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "Mish upper bound UNSOUND at x={}: {} < Mish({})={}, \
                 interval=[{}, {}], gap={}", x, upper_val, x, fx, l, u, fx - upper_val
            );
        }
    }
}

// ── Inflection-crossing strict-soundness proptest (region-classified relax) ──
//
// The region-classified relaxation (SiLU-style tangent/chord construction) is
// only sound if every emitted line stays on the correct side of Mish across
// BOTH curvature regions when an interval straddles an inflection point. This
// proptest deliberately forces intervals that straddle p1 ≈ -2.256 and/or
// p2 ≈ +1.491, evaluating a 200-point f64 grid with ZERO tolerance. It is the
// gate that catches any unsound crossing-region line.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(2000) })]

    /// Strict soundness on intervals that STRADDLE the inflection points.
    /// The interval is constructed to contain a chosen pivot point near an
    /// inflection (p1 ≈ -2.256 or p2 ≈ +1.491), with the left/right extents
    /// drawn so the interval always crosses it. No `prop_assume` rejection: l < u
    /// holds by construction. Covers cross_left, cross_right, and cross_both over
    /// roughly [-6, 6].
    #[test]
    fn proptest_mish_crossing_strict_soundness(
        // Pick which inflection to straddle, plus how far below/above to extend.
        pivot_sel in prop::bool::ANY,
        below in 0.05f32..4.0,
        above in 0.05f32..4.0,
    ) {
        // p1 ≈ -2.256, p2 ≈ +1.491. Straddle one of them.
        let pivot = if pivot_sel { -2.256_f32 } else { 1.491_f32 };
        let l = pivot - below;
        let u = pivot + above;

        let relax = mish_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;
        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = (l as f64 + t * (u as f64 - l as f64)).clamp(l as f64, u as f64);
            let fx = mish_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "Mish crossing lower UNSOUND at x={}: {} > Mish={} on [{}, {}], gap={}",
                x, lower_val, fx, l, u, lower_val - fx
            );
            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "Mish crossing upper UNSOUND at x={}: {} < Mish={} on [{}, {}], gap={}",
                x, upper_val, fx, l, u, fx - upper_val
            );
        }
    }

    /// Strict soundness over wide intervals spanning the full [-6, 6] window,
    /// always crossing both inflection points.
    #[test]
    fn proptest_mish_wide_crossing_strict_soundness(
        l in -6.0f32..-2.5,
        u in 2.0f32..6.0,
    ) {
        let relax = mish_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;
        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = (l as f64 + t * (u as f64 - l as f64)).clamp(l as f64, u as f64);
            let fx = mish_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(lower_val <= fx,
                "wide-crossing lower UNSOUND at x={}: {} > {} on [{}, {}]", x, lower_val, fx, l, u);
            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(upper_val >= fx,
                "wide-crossing upper UNSOUND at x={}: {} < {} on [{}, {}]", x, upper_val, fx, l, u);
        }
    }
}
