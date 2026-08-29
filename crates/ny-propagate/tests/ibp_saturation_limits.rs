// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Saturated (±inf) IBP inputs must SATURATE per layer, never hard-error —
//! sibling coverage to `deep_ibp_overflow_saturates.rs` (the dc9da0fde
//! LayerNorm/GELU fix) for the remaining guard-dead-codes-its-own-fallback
//! sites:
//!
//! - RMSNorm / InstanceNorm1d / GroupNorm: the entry guard refused ALL
//!   non-finite input bounds, dead-coding each layer's own pre-existing
//!   analytic fallback enclosure (|z| ≤ √n for RMSNorm, beta ± ny·√(n−1) for
//!   the mean-subtract norms). Now NaN-only fails closed; a ±inf (NaN-free)
//!   saturated box routes to that fallback.
//! - Bounded / limit-correct activations (Abs, HardSigmoid, SELU,
//!   ThresholdedRelu): their interval analyses are limit-correct at ±inf, so
//!   the non-finite refusal is gone; endpoints map to the exact limits.
//! - Exp: monotone limit-aware — exp(-inf) = 0, exp(+inf) = +inf, and a
//!   FINITE upper endpoint beyond the f32 range saturates to +inf instead of
//!   refusing. LogSumExp: ±inf inputs take the exact limits (+inf element →
//!   +inf; all-(-inf) slice → -inf).
//!
//! NaN must STILL fail closed everywhere: NaN carries no bound information,
//! and a silently repaired NaN could mask a real numerical bug.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::NyError;
use ny_propagate::layers::{
    AbsLayer, ExpLayer, GroupNormLayer, HardSigmoidLayer, InstanceNorm1dLayer, LogSumExpLayer,
    RmsNormLayer, SeluLayer, ThresholdedReluLayer,
};
use ny_propagate::BoundPropagation;
use ny_tensor::{cast_f64_to_f32_down, BoundedTensor};

/// A fully saturated box [-inf, +inf] of the given shape.
fn saturated_box(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(shape), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(shape), f32::INFINITY),
    )
    .expect("valid saturated box")
}

/// A box with a single NaN endpoint (test-utils constructor bypasses validation).
fn nan_box(shape: &[usize]) -> BoundedTensor {
    let mut upper = ArrayD::from_elem(IxDyn(shape), 1.0_f32);
    if let Some(first) = upper.iter_mut().next() {
        *first = f32::NAN;
    }
    BoundedTensor::new_unchecked(ArrayD::from_elem(IxDyn(shape), -1.0_f32), upper)
        .expect("test-only NaN box")
}

// ── RMSNorm ────────────────────────────────────────────────────────────────

/// A saturated box takes RMSNorm's own |y_i| ≤ |ny_i|·√n fallback enclosure —
/// finite, symmetric, and wide enough to contain the eval of arbitrarily
/// large FINITE inputs (which is what the saturated box over-approximates).
#[ntest::timeout(60000)]
#[test]
fn rms_norm_saturated_box_takes_own_fallback() {
    let n = 4usize;
    let layer = RmsNormLayer::new_default(n, 1e-5).unwrap();
    let out = layer
        .propagate_ibp(&saturated_box(&[2, n]))
        .expect("saturated box must take the fallback, not error");

    assert_eq!(out.shape(), &[2, n]);
    let sqrt_n = (n as f32).sqrt();
    for (l, u) in out.lower().iter().zip(out.upper().iter()) {
        assert!(l.is_finite() && u.is_finite(), "fallback box is finite");
        assert!(
            *l <= -sqrt_n && *u >= sqrt_n,
            "must enclose ±√n = ±{sqrt_n}"
        );
        assert!(*u <= sqrt_n * 1.01, "fallback stays the analytic √n box");
    }

    // Real-propagation check: the fallback must contain eval at an extreme
    // finite input — the kind of value a saturated interval over-approximates.
    let x = Array1::from_elem(n, 1.0e18_f32);
    let y = layer.eval(&x).expect("eval of finite input");
    for (i, yi) in y.iter().enumerate() {
        assert!(
            out.lower()[[0, i]] <= *yi && *yi <= out.upper()[[0, i]],
            "eval(1e18)[{i}] = {yi} escapes the fallback box [{}, {}]",
            out.lower()[[0, i]],
            out.upper()[[0, i]]
        );
    }
}

/// Partial saturation (a single ±inf endpoint) routes to the same fallback.
#[ntest::timeout(60000)]
#[test]
fn rms_norm_partially_saturated_box_takes_fallback() {
    let layer = RmsNormLayer::new_default(3, 1e-5).unwrap();
    let mut upper = ArrayD::from_elem(IxDyn(&[3]), 1.0_f32);
    upper[[0]] = f32::INFINITY;
    let input =
        BoundedTensor::new_allow_infinite(ArrayD::from_elem(IxDyn(&[3]), -1.0_f32), upper).unwrap();
    let out = layer
        .propagate_ibp(&input)
        .expect("one saturated endpoint must take the fallback, not error");
    assert!(out
        .lower()
        .iter()
        .chain(out.upper().iter())
        .all(|v| v.is_finite()));
}

/// Forward mode routes a saturated box to the same fallback (the route the
/// non-finite-center path already used), and NaN fails closed in BOTH modes.
#[ntest::timeout(60000)]
#[test]
fn rms_norm_forward_mode_saturates_and_nan_fails_closed() {
    let conservative = RmsNormLayer::new_default(4, 1e-5).unwrap();
    let forward = RmsNormLayer::new_default(4, 1e-5)
        .unwrap()
        .with_forward_mode(true);

    forward
        .propagate_ibp(&saturated_box(&[4]))
        .expect("forward mode must saturate, not error");

    for (name, layer) in [("conservative", &conservative), ("forward", &forward)] {
        let err = layer
            .propagate_ibp(&nan_box(&[4]))
            .err()
            .unwrap_or_else(|| panic!("RMSNorm {name} mode: NaN must fail closed"));
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "RMSNorm {name} NaN rejection must be NumericalInstability, got {err:?}"
        );
    }
}

// ── InstanceNorm1d ─────────────────────────────────────────────────────────

/// A saturated box takes InstanceNorm1d's own beta ± ny·√(T−1) fallback,
/// sign-aware per channel.
#[ntest::timeout(60000)]
#[test]
fn instance_norm_saturated_box_takes_own_fallback() {
    let time_len = 3usize;
    let layer = InstanceNorm1dLayer::new(
        Array1::from_vec(vec![2.0, -1.0]),
        Array1::from_vec(vec![0.5, -0.5]),
        1e-5,
    )
    .unwrap();
    let out = layer
        .propagate_ibp(&saturated_box(&[2, time_len]))
        .expect("saturated box must take the fallback, not error");

    assert_eq!(out.shape(), &[2, time_len]);
    let z = ((time_len as f32) - 1.0).sqrt();
    // Channel 0: g = 2, b = 0.5 → [0.5 - 2z, 0.5 + 2z] (within rounding).
    for t in 0..time_len {
        assert!(out.lower()[[0, t]] <= 0.5 - 2.0 * z && out.lower()[[0, t]].is_finite());
        assert!(out.upper()[[0, t]] >= 0.5 + 2.0 * z && out.upper()[[0, t]].is_finite());
        // Channel 1: g = -1 (sign-aware swap), b = -0.5 → [-0.5 - z, -0.5 + z].
        assert!(out.lower()[[1, t]] <= -0.5 - z && out.lower()[[1, t]].is_finite());
        assert!(out.upper()[[1, t]] >= -0.5 + z && out.upper()[[1, t]].is_finite());
    }
}

#[ntest::timeout(60000)]
#[test]
fn instance_norm_nan_fails_closed_both_modes() {
    let conservative = InstanceNorm1dLayer::new_default(2, 1e-5).unwrap();
    let forward = InstanceNorm1dLayer::new_default(2, 1e-5)
        .unwrap()
        .with_forward_mode(true);
    for (name, layer) in [("conservative", &conservative), ("forward", &forward)] {
        let err = layer
            .propagate_ibp(&nan_box(&[2, 3]))
            .err()
            .unwrap_or_else(|| panic!("InstanceNorm1d {name}: NaN must fail closed"));
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "InstanceNorm1d {name} NaN rejection must be NumericalInstability, got {err:?}"
        );
    }
    // Forward mode also saturates rather than erroring.
    forward
        .propagate_ibp(&saturated_box(&[2, 3]))
        .expect("forward mode must saturate, not error");
}

// ── GroupNorm ──────────────────────────────────────────────────────────────

/// A saturated box takes GroupNorm's own beta ± ny·√(n−1) fallback
/// (n = channels-per-group × T).
#[ntest::timeout(60000)]
#[test]
fn group_norm_saturated_box_takes_own_fallback() {
    let layer = GroupNormLayer::new_default(4, 2, 1e-5).unwrap();
    let out = layer
        .propagate_ibp(&saturated_box(&[4, 2]))
        .expect("saturated box must take the fallback, not error");

    assert_eq!(out.shape(), &[4, 2]);
    // group size n = (4/2) * 2 = 4 → z_max = √3; default ny = 1, beta = 0.
    let z = 3.0_f32.sqrt();
    for (l, u) in out.lower().iter().zip(out.upper().iter()) {
        assert!(l.is_finite() && u.is_finite(), "fallback box is finite");
        assert!(*l <= -z && *u >= z, "must enclose ±√(n-1) = ±{z}");
        assert!(*u <= z * 1.01, "fallback stays the analytic √(n-1) box");
    }
}

#[ntest::timeout(60000)]
#[test]
fn group_norm_nan_fails_closed_both_modes() {
    let conservative = GroupNormLayer::new_default(4, 2, 1e-5).unwrap();
    let forward = GroupNormLayer::new_default(4, 2, 1e-5)
        .unwrap()
        .with_forward_mode(true);
    for (name, layer) in [("conservative", &conservative), ("forward", &forward)] {
        let err = layer
            .propagate_ibp(&nan_box(&[4, 2]))
            .err()
            .unwrap_or_else(|| panic!("GroupNorm {name}: NaN must fail closed"));
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "GroupNorm {name} NaN rejection must be NumericalInstability, got {err:?}"
        );
    }
    forward
        .propagate_ibp(&saturated_box(&[4, 2]))
        .expect("forward mode must saturate, not error");
}

// ── Abs ────────────────────────────────────────────────────────────────────

/// Abs is limit-correct at ±inf: Abs(-inf) = Abs(+inf) = +inf, and the
/// crossing case on a saturated box is exactly [0, +inf].
#[ntest::timeout(60000)]
#[test]
fn abs_saturated_limits_are_exact() {
    let layer = AbsLayer;

    // Crossing: [-inf, +inf] → [0, +inf].
    let out = layer.propagate_ibp(&saturated_box(&[2])).unwrap();
    assert_eq!(out.lower()[[0]], 0.0);
    assert!(out.upper()[[0]].is_infinite() && out.upper()[[0]].is_sign_positive());

    // Negative branch: [-inf, -3] → [3, +inf] (Abs(-inf) = +inf exactly).
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1]), -3.0_f32),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.lower()[[0]], 3.0);
    assert!(out.upper()[[0]].is_infinite() && out.upper()[[0]].is_sign_positive());

    // Positive branch: [5, +inf] passes through (Abs(+inf) = +inf exactly).
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[1]), 5.0_f32),
        ArrayD::from_elem(IxDyn(&[1]), f32::INFINITY),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.lower()[[0]], 5.0);
    assert!(out.upper()[[0]].is_infinite() && out.upper()[[0]].is_sign_positive());
}

#[ntest::timeout(60000)]
#[test]
fn abs_nan_still_fails_closed() {
    let err = AbsLayer
        .propagate_ibp(&nan_box(&[2]))
        .expect_err("Abs: NaN must fail closed");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

// ── HardSigmoid / SELU / ThresholdedRelu (limit pins) ──────────────────────

/// HardSigmoid maps a saturated box to its exact range: [0, 1].
#[ntest::timeout(60000)]
#[test]
fn hard_sigmoid_saturated_box_is_exactly_unit_interval() {
    let out = HardSigmoidLayer::default_params()
        .propagate_ibp(&saturated_box(&[3]))
        .expect("saturated box must propagate");
    for (l, u) in out.lower().iter().zip(out.upper().iter()) {
        assert_eq!(*l, 0.0, "HardSigmoid(-inf) = 0 exactly");
        assert_eq!(*u, 1.0, "HardSigmoid(+inf) = 1 exactly");
    }

    let err = HardSigmoidLayer::default_params()
        .propagate_ibp(&nan_box(&[3]))
        .expect_err("HardSigmoid: NaN must fail closed");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

/// SELU maps a saturated box to [−λα, +inf] with the lower asymptote exact.
#[ntest::timeout(60000)]
#[test]
fn selu_saturated_box_hits_exact_asymptote() {
    let out = SeluLayer::new()
        .propagate_ibp(&saturated_box(&[2]))
        .expect("saturated box must propagate");
    // OUTWARD-rounded, and that is the entire point. The f32 round-to-nearest
    // product `-(LAMBDA * ALPHA)` sits ~2.9e-8 ABOVE the true asymptote, so
    // pinning it raises the lower bound above the real SELU(x) for every
    // x < -17 — precisely the false proof 5d48d59e9 was written to remove.
    // That commit fixed `selu_floor` and ADDED this file in the same change,
    // still asserting the round-to-nearest value, so this gate has been RED
    // since the moment it landed. Repairing it by making production match the
    // old expectation would reintroduce the false proof; the direction
    // assertion below exists so that can never happen silently again.
    let floor =
        cast_f64_to_f32_down(-(f64::from(SeluLayer::LAMBDA) * f64::from(SeluLayer::ALPHA)));
    assert_eq!(
        out.lower()[[0]],
        floor,
        "SELU(-inf) saturates at the OUTWARD-rounded -lambda*alpha asymptote"
    );
    assert!(
        f64::from(out.lower()[[0]])
            <= -(f64::from(SeluLayer::LAMBDA) * f64::from(SeluLayer::ALPHA)),
        "the SELU floor must sit at or BELOW the true asymptote, never above it"
    );
    assert!(out.upper()[[0]].is_infinite() && out.upper()[[0]].is_sign_positive());

    let err = SeluLayer::new()
        .propagate_ibp(&nan_box(&[2]))
        .expect_err("SELU: NaN must fail closed");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

/// ThresholdedRelu maps a saturated box to [0, +inf] (alpha >= 0).
#[ntest::timeout(60000)]
#[test]
fn thresholded_relu_saturated_box_limits() {
    let out = ThresholdedReluLayer::new(1.0)
        .propagate_ibp(&saturated_box(&[2]))
        .expect("saturated box must propagate");
    assert_eq!(out.lower()[[0]], 0.0, "TRelu(-inf) = 0 exactly");
    assert!(out.upper()[[0]].is_infinite() && out.upper()[[0]].is_sign_positive());

    let err = ThresholdedReluLayer::new(1.0)
        .propagate_ibp(&nan_box(&[2]))
        .expect_err("ThresholdedRelu: NaN must fail closed");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

// ── Exp ────────────────────────────────────────────────────────────────────

/// Exp is monotone limit-aware: exp(-inf) = 0 and exp(+inf) = +inf exactly,
/// and a FINITE upper endpoint beyond the f32 range saturates to +inf while a
/// finite lower endpoint NEVER becomes +inf (that would be a false proof).
#[ntest::timeout(60000)]
#[test]
fn exp_limits_and_overflow_saturation() {
    let layer = ExpLayer::new();

    let out = layer.propagate_ibp(&saturated_box(&[1])).unwrap();
    assert_eq!(out.lower()[[0]], 0.0, "exp(-inf) = 0 exactly");
    assert!(out.upper()[[0]].is_infinite() && out.upper()[[0]].is_sign_positive());

    // Finite input above the old refusal threshold: saturates, no error.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[1]), 100.0_f32),
    )
    .unwrap();
    let out = layer
        .propagate_ibp(&input)
        .expect("finite overflow-range input must saturate, not refuse");
    assert!((out.lower()[[0]] - 1.0).abs() < 1e-5);
    assert!(out.upper()[[0]].is_infinite() && out.upper()[[0]].is_sign_positive());

    // Finite lower endpoint beyond even f64 exp range: floors at f32::MAX.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 800.0_f32),
        ArrayD::from_elem(IxDyn(&[1]), 900.0_f32),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(
        out.lower()[[0]],
        f32::MAX,
        "finite lower endpoint must stay finite (f32::MAX), never +inf"
    );

    let err = layer
        .propagate_ibp(&nan_box(&[1]))
        .expect_err("Exp: NaN must fail closed");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

// ── LogSumExp ──────────────────────────────────────────────────────────────

/// LogSumExp admits ±inf inputs with the exact limits: a +inf element makes
/// the reduction +inf; an all-(-inf) slice reduces to -inf; a -inf element
/// among finite ones contributes exp(-inf) = 0 exactly.
#[ntest::timeout(60000)]
#[test]
fn logsumexp_saturated_inputs_take_exact_limits() {
    let layer = LogSumExpLayer::new(vec![-1], false);

    // Fully saturated box: lower reduce sees all -inf → -inf; upper reduce
    // sees all +inf → +inf. A sound (vacuous) box, no NaN.
    let out = layer.propagate_ibp(&saturated_box(&[3])).unwrap();
    let (l, u) = (
        *out.lower().iter().next().unwrap(),
        *out.upper().iter().next().unwrap(),
    );
    assert!(
        l.is_infinite() && l.is_sign_negative(),
        "LSE(all -inf) = -inf"
    );
    assert!(
        u.is_infinite() && u.is_sign_positive(),
        "LSE(.., +inf, ..) = +inf"
    );
    assert!(!l.is_nan() && !u.is_nan());

    // Mixed: -inf among finite values drops out exactly.
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    // Lower: LSE(-inf, 0) = ln(e^0) = 0.
    assert!(
        out.lower().iter().next().unwrap().abs() < 1e-5,
        "LSE(-inf, 0) = 0"
    );

    let err = layer
        .propagate_ibp(&nan_box(&[3]))
        .expect_err("LogSumExp: NaN must fail closed");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

// ---------------------------------------------------------------------------
// Fallback-enclosure rounding (adversarial review 2026-08-26)
// ---------------------------------------------------------------------------

/// The saturated-box fallback `beta + gamma*z` MUST be rounded outward at the
/// PRODUCT's magnitude, not the sum's.
///
/// Computed in f32, `next_up_f32(beta + gamma * z_max)` takes its one-ULP step
/// at the magnitude of the (cancelled) sum, which can be hundreds of ULPs
/// finer than the product's own round-to-nearest error — so the "bound"
/// excluded reachable outputs. This is the reviewer's exact counterexample:
/// gamma = 1.1550927, beta = -2.0006793, T = 4. The true supremum of the
/// channel's output is `gamma*sqrt(3) + beta = 1.7408794e-8` (attained in the
/// limit by x = [1e5, 0, 0, 0], whose z hits sqrt(n-1)); the f32 form returned
/// an upper bound of 1e-45, excluding it by 37 orders of magnitude.
#[test]
fn instance_norm_saturated_fallback_encloses_the_cancellation_counterexample() {
    let gamma = f32::from_bits(0x3f93_da14); // 1.1550927
    let beta = f32::from_bits(0xc000_0b21); // -2.0006793
    let layer = InstanceNorm1dLayer::new(
        Array1::from_vec(vec![gamma]),
        Array1::from_vec(vec![beta]),
        1e-5,
    )
    .expect("valid instance norm");

    let shape = IxDyn(&[1, 1, 4]);
    let saturated = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(shape.clone(), f32::NEG_INFINITY),
        ArrayD::from_elem(shape, f32::INFINITY),
    )
    .expect("saturated box");
    let out = layer
        .propagate_ibp(&saturated)
        .expect("saturated input must take the conservative fallback");

    // The exact real supremum / infimum of gamma*z + beta over |z| <= sqrt(n-1),
    // computed in f64 (the enclosure's own claim). The fallback must cover BOTH.
    let z_max = 3.0_f64.sqrt();
    let sup = f64::from(gamma).mul_add(z_max, f64::from(beta));
    let inf = f64::from(gamma).mul_add(-z_max, f64::from(beta));

    for (&lo, &hi) in out.lower().iter().zip(out.upper().iter()) {
        assert!(
            f64::from(hi) >= sup,
            "fallback upper {hi:e} must cover the true supremum {sup:e}"
        );
        assert!(
            f64::from(lo) <= inf,
            "fallback lower {lo:e} must cover the true infimum {inf:e}"
        );
    }
}
