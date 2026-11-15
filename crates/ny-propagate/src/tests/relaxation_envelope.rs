// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Envelope soundness of the production scalar relaxations used by CROWN.
//!
//! For every activation `f`, the relaxation is a pair of lines
//!
//! ```text
//!   lower_slope * x + lower_intercept <= f(x) <= upper_slope * x + upper_intercept
//! ```
//!
//! that must hold for all `x` in `[l, u]`. An unsound envelope (the true value
//! escapes it anywhere in the interval) lets the verifier emit a wrong
//! "verified" verdict. Two tiers check it:
//!
//! * Dense sampling (the first half): each curated or random interval is
//!   sampled densely at a few-ULP tolerance, and `lower(x) <= upper(x)` is
//!   checked too.
//! * The adversarial grid (the second half): every ordered pair from a grid of
//!   signed zeros, subnormals, branch points, threshold neighbours and
//!   extremes, with the band evaluated in f32 the way callers evaluate it and
//!   compared at near-zero slack against INDEPENDENT f64 references, never
//!   against this crate's own `silu_eval`/`gelu_eval`.
//!
//! These checks used to run against `ny-relaxation`, a scalar mirror of these
//! functions that existed only so Kani harnesses could compile them, and that
//! mirror was bound back to this crate by a bit-exact drift test whose
//! production-side soundness asserts are the second tier here. The mirror and
//! the harnesses are gone; the checks now call the shipped functions directly,
//! so there is no copy for them to drift from. The ReLU cases at the bottom
//! replay the inputs on which Kani found counterexamples in an earlier ReLU
//! relaxation against the production `relu_linear_relaxation`.

use ndarray::{ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::layers::activations::relu::relu_linear_relaxation;
use crate::layers::arithmetic::sqrt::sqrt_linear_relaxation_with_alpha;
use crate::layers::{
    abs_linear_relaxation, exp_linear_relaxation, gelu_eval, gelu_sound_linear_relaxation,
    gelu_tanh_sound_linear_relaxation, log_linear_relaxation, pow2_linear_relaxation, silu_eval,
    silu_sound_linear_relaxation, sqrt_linear_relaxation, GeluApproximation, LinearRelaxation,
    ReLULayer,
};
use crate::LinearBounds;

/// A relaxation's four coefficients widened to f64, whichever shape the API
/// returns.
#[derive(Clone, Copy, Debug)]
struct Lines {
    lower_slope: f64,
    lower_intercept: f64,
    upper_slope: f64,
    upper_intercept: f64,
}

impl From<LinearRelaxation> for Lines {
    fn from(r: LinearRelaxation) -> Self {
        Lines {
            lower_slope: r.lower_slope as f64,
            lower_intercept: r.lower_intercept as f64,
            upper_slope: r.upper_slope as f64,
            upper_intercept: r.upper_intercept as f64,
        }
    }
}

impl From<(f32, f32, f32, f32)> for Lines {
    fn from(t: (f32, f32, f32, f32)) -> Self {
        Lines {
            lower_slope: t.0 as f64,
            lower_intercept: t.1 as f64,
            upper_slope: t.2 as f64,
            upper_intercept: t.3 as f64,
        }
    }
}

impl Lines {
    // An f32 * f32 product is exact in f64, so these evaluate the stored lines
    // up to one f64 rounding of the sum.
    fn lower_at(&self, x: f64) -> f64 {
        self.lower_slope * x + self.lower_intercept
    }
    fn upper_at(&self, x: f64) -> f64 {
        self.upper_slope * x + self.upper_intercept
    }
}

/// Slack for comparing an f64 reference `f(x)` against the lines.
///
/// The reference transcendental is evaluated in f64 while the relaxation was
/// built from f32 evaluations, so a tiny relative slack absorbs the gap. A
/// genuine unsoundness (a chord on the wrong side of a convex or concave
/// region) is gross and trips this easily.
fn tol(x: f64, fx: f64, line: f64) -> f64 {
    let scale = x.abs().max(fx.abs()).max(line.abs()).max(1.0);
    // ~4 f32 ULPs of relative slack plus a small absolute floor.
    scale * 6.0e-6 + 1.0e-6
}

/// Assert the envelope contains `f` at densely sampled points of `[l, u]`.
fn assert_envelope_contains<F>(name: &str, l: f32, u: f32, lines: Lines, f: F)
where
    F: Fn(f64) -> f64,
{
    assert!(l <= u, "{name}: bad interval [{l}, {u}]");

    // Every fixture here has a finite, supported domain. A non-finite
    // relaxation must not read as trivially sound: NaN is not the
    // authenticated [-inf, +inf] fallback, and an unexpected infinity means
    // the useful relaxation disappeared.
    assert!(
        lines.lower_slope.is_finite()
            && lines.lower_intercept.is_finite()
            && lines.upper_slope.is_finite()
            && lines.upper_intercept.is_finite(),
        "{name}: finite interval [{l}, {u}] produced a non-finite envelope: {lines:?}"
    );

    let l64 = l as f64;
    let u64 = u as f64;

    // A uniform grid plus the exact endpoints, where chord and tangent
    // relaxations are tightest and an off-by-rounding error shows first.
    const N: usize = 401;
    let mut samples: Vec<f64> = Vec::with_capacity(N + 7);
    for i in 0..N {
        let t = i as f64 / (N as f64 - 1.0);
        samples.push(l64 + t * (u64 - l64));
    }
    for &t in &[0.0, 1.0, 0.5, 0.25, 0.75, 1.0 / 3.0, 2.0 / 3.0] {
        samples.push(l64 + t * (u64 - l64));
    }

    for &x in &samples {
        let fx = f(x);
        if !fx.is_finite() {
            continue;
        }
        let lower = lines.lower_at(x);
        let upper = lines.upper_at(x);
        let t = tol(x, fx, lower.abs().max(upper.abs()));

        assert!(
            lower <= fx + t,
            "{name}: LOWER unsound on [{l}, {u}] at x={x}: \
             lower={lower} > f(x)={fx} (tol={t})\n  lines={lines:?}"
        );
        assert!(
            fx <= upper + t,
            "{name}: UPPER unsound on [{l}, {u}] at x={x}: \
             f(x)={fx} > upper={upper} (tol={t})\n  lines={lines:?}"
        );
        assert!(
            lower <= upper + t,
            "{name}: lower line above upper line on [{l}, {u}] at x={x}: \
             lower={lower} > upper={upper}\n  lines={lines:?}"
        );
    }
}

/// Every regime: deep-negative, deep-positive, straddling zero, wide, narrow,
/// near-point, and asymmetric.
fn general_intervals() -> Vec<(f32, f32)> {
    vec![
        (-5.0, -1.0),   // negative
        (1.0, 5.0),     // positive
        (-3.0, 3.0),    // straddles zero, symmetric
        (-0.5, 2.0),    // straddles zero, asymmetric
        (-2.0, 0.3),    // straddles zero, asymmetric (other side)
        (-10.0, 10.0),  // wide
        (-0.01, 0.01),  // narrow straddle
        (0.5, 0.5001),  // near-point positive
        (-4.0, -3.999), // near-point negative
        (0.0, 6.0),     // zero lower bound
        (-6.0, 0.0),    // zero upper bound
        (-8.0, -7.0),   // far-negative narrow
        (7.0, 8.0),     // far-positive narrow
        (-1.5, 1.5),    // moderate straddle
    ]
}

/// Intervals with `l >= 0`: right for `sqrt` (defined at 0), wrong for `log`
/// (see [`strictly_positive_intervals`]).
fn positive_intervals() -> Vec<(f32, f32)> {
    vec![
        (0.01, 1.0),
        (1.0, 5.0),
        (0.5, 10.0),
        (2.0, 2.0001),
        (1e-3, 1.0),
        (0.0, 4.0),
        (3.0, 12.0),
        (1e-6, 1e-3),
        (4.0, 4.5),
    ]
}

/// `log`'s domain is strictly `x > 0`. On `(0.0, 4.0)` the only sound lower
/// bound is `-inf`, which production returns and `assert_envelope_contains`
/// rejects as non-finite; [`log_refuses_an_interval_touching_zero`] pins that
/// edge instead.
fn strictly_positive_intervals() -> Vec<(f32, f32)> {
    positive_intervals()
        .into_iter()
        .filter(|&(l, _)| l > 0.0)
        .collect()
}

#[test]
fn relu_relaxation_is_sound() {
    for (l, u) in general_intervals() {
        let lines: Lines = relu_linear_relaxation(l, u).into();
        assert_envelope_contains("relu", l, u, lines, |x| x.max(0.0));
    }
}

#[test]
fn abs_relaxation_is_sound() {
    for (l, u) in general_intervals() {
        let lines: Lines = abs_linear_relaxation(l, u).into();
        assert_envelope_contains("abs", l, u, lines, |x| x.abs());
    }
}

#[test]
fn exp_relaxation_is_sound() {
    // exp grows fast; bounded magnitudes keep the f32/f64 gap tiny.
    let intervals = [
        (-5.0, -1.0),
        (-2.0, 2.0),
        (0.0, 3.0),
        (1.0, 4.0),
        (-1.0, 0.0),
        (-0.5, 0.5),
        (2.0, 2.0001),
        (-3.0, 1.0),
        (-4.0, -3.999),
    ];
    for (l, u) in intervals {
        let lines: Lines = exp_linear_relaxation(l, u).into();
        assert_envelope_contains("exp", l, u, lines, |x| x.exp());
    }
}

#[test]
fn pow2_relaxation_is_sound() {
    for (l, u) in general_intervals() {
        let lines: Lines = pow2_linear_relaxation(l, u).into();
        assert_envelope_contains("pow2", l, u, lines, |x| x * x);
    }
}

#[test]
fn sqrt_relaxation_is_sound() {
    for (l, u) in positive_intervals() {
        let lines: Lines = sqrt_linear_relaxation(l, u).into();
        assert_envelope_contains("sqrt", l, u, lines, |x| x.max(0.0).sqrt());
    }
}

#[test]
fn log_relaxation_is_sound() {
    for (l, u) in strictly_positive_intervals() {
        // Compare against the TRUE `ln`: `log` does not clamp its domain (a
        // 1e-10 clamp was removed as unsound, #log-epsilon-nonenclosing).
        let lines: Lines = log_linear_relaxation(l, u).into();
        assert_envelope_contains("log", l, u, lines, |x| x.ln());
    }
}

/// An interval touching 0 is out of `log`'s domain, so the sound answer is the
/// maximally loose envelope. A finite lower bound here would claim `ln` is
/// bounded below on `(0, 4]`, which is false.
#[test]
fn log_refuses_an_interval_touching_zero() {
    let lines: Lines = log_linear_relaxation(0.0, 4.0).into();
    assert!(
        lines.lower_intercept == f64::NEG_INFINITY,
        "ln(0) = -inf, so no finite lower bound on [0, 4] can be sound: {lines:?}"
    );
}

#[test]
fn silu_relaxation_is_sound() {
    for (l, u) in general_intervals() {
        let lines: Lines = silu_sound_linear_relaxation(l, u).into();
        assert_envelope_contains("silu", l, u, lines, |x| silu_eval(x as f32) as f64);
    }
}

#[test]
fn gelu_erf_relaxation_is_sound() {
    for (l, u) in general_intervals() {
        let lines: Lines = gelu_sound_linear_relaxation(l, u).into();
        assert_envelope_contains("gelu_erf", l, u, lines, |x| {
            gelu_eval(x as f32, GeluApproximation::Erf) as f64
        });
    }
}

#[test]
fn gelu_tanh_relaxation_is_sound() {
    for (l, u) in general_intervals() {
        let lines: Lines = gelu_tanh_sound_linear_relaxation(l, u).into();
        assert_envelope_contains("gelu_tanh", l, u, lines, |x| {
            gelu_eval(x as f32, GeluApproximation::Tanh) as f64
        });
    }
}

/// Many deterministically generated intervals per activation, covering far
/// more of the input space than the curated lists (xorshift; no dependency).
#[test]
fn relaxations_sound_on_dense_random_intervals() {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let uniform = |lo: f32, hi: f32, r: u64| -> f32 {
        let frac = (r >> 11) as f64 / (1u64 << 53) as f64;
        (lo as f64 + frac * (hi as f64 - lo as f64)) as f32
    };
    let ordered = |a: f32, b: f32| if a <= b { (a, b) } else { (b, a) };

    for _ in 0..2000 {
        let (l, u) = ordered(uniform(-6.0, 6.0, next()), uniform(-6.0, 6.0, next()));
        assert_envelope_contains("relu", l, u, relu_linear_relaxation(l, u).into(), |x| {
            x.max(0.0)
        });
        assert_envelope_contains("abs", l, u, abs_linear_relaxation(l, u).into(), |x| x.abs());
        assert_envelope_contains("pow2", l, u, pow2_linear_relaxation(l, u).into(), |x| x * x);
        assert_envelope_contains(
            "silu",
            l,
            u,
            silu_sound_linear_relaxation(l, u).into(),
            |x| silu_eval(x as f32) as f64,
        );
        assert_envelope_contains(
            "gelu_erf",
            l,
            u,
            gelu_sound_linear_relaxation(l, u).into(),
            |x| gelu_eval(x as f32, GeluApproximation::Erf) as f64,
        );
        assert_envelope_contains(
            "gelu_tanh",
            l,
            u,
            gelu_tanh_sound_linear_relaxation(l, u).into(),
            |x| gelu_eval(x as f32, GeluApproximation::Tanh) as f64,
        );

        let (el, eu) = ordered(uniform(-4.0, 3.0, next()), uniform(-4.0, 3.0, next()));
        assert_envelope_contains("exp", el, eu, exp_linear_relaxation(el, eu).into(), |x| {
            x.exp()
        });

        let (pl, pu) = ordered(uniform(1e-3, 10.0, next()), uniform(1e-3, 10.0, next()));
        assert_envelope_contains("sqrt", pl, pu, sqrt_linear_relaxation(pl, pu).into(), |x| {
            x.sqrt()
        });
        assert_envelope_contains("log", pl, pu, log_linear_relaxation(pl, pu).into(), |x| {
            x.ln()
        });
    }
}

// =========================================================================
// The adversarial grid, at the slacks the deleted drift test used
// =========================================================================

/// A band as the API returns it: (lower_slope, lower_intercept, upper_slope,
/// upper_intercept).
type Band = (f32, f32, f32, f32);

fn band(r: LinearRelaxation) -> Band {
    (
        r.lower_slope,
        r.lower_intercept,
        r.upper_slope,
        r.upper_intercept,
    )
}

/// Signed zeros, subnormals, the narrow-interval thresholds (1e-8, 1e-12),
/// the branch-point neighbourhoods of the nonlinear relaxations (silu
/// inflection and critical points, the gelu minimiser, the legacy exp cap at
/// 0.99/1.98), ULP neighbours of anchors, a dense sweep of [-6, 6], and the
/// extremes.
fn adversarial_grid() -> Vec<f32> {
    let mut v: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::from_bits(1), // smallest positive subnormal
        -f32::from_bits(1),
        f32::from_bits(0x0040_0000), // mid subnormal
        -f32::from_bits(0x0040_0000),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-38,
        -1e-38,
        1e-30,
        -1e-30,
        1e-12,
        -1e-12,
        2e-12,
        5e-9,
        -5e-9,
        1e-8,
        -1e-8,
        2e-8,
        -2e-8,
        1e-6,
        -1e-6,
        1e-3,
        -1e-3,
        0.05,
        -0.05,
        0.1,
        -0.1,
        0.5,
        -0.5,
        std::f32::consts::LN_2,
        -std::f32::consts::LN_2,
        -0.751_791_6, // gelu minimiser neighbourhood
        -0.751_791_5,
        0.99,
        -0.99,
        1.0,
        -1.0,
        1.0 + f32::EPSILON,
        -1.0 - f32::EPSILON,
        1.5,
        -1.5,
        1.98,
        -1.98,
        2.0,
        -2.0,
        -1.278_46, // silu critical point / inflection neighbourhoods
        -1.278_464_5,
        2.399,
        -2.399,
        2.3994,
        -2.3994,
        2.4,
        -2.4,
        std::f32::consts::E,
        -std::f32::consts::E,
        3.0,
        -3.0,
        4.0,
        -4.0,
        5.0,
        -5.0,
        8.0,
        -8.0,
        10.0,
        -10.0,
        19.25,
        -19.25,
        20.0,
        -20.0,
        50.0,
        -50.0,
        88.0,
        -88.0,
        100.0,
        -100.0,
        709.0,
        -709.0,
        1e4,
        -1e4,
        1e8,
        -1e8,
        1e20,
        -1e20,
        3e38,
        -3e38,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    for a in [0.0f32, 1.0, -1.0, 2.0, -2.0, 0.05, 1e-8, -1e-8] {
        v.push(next_up_f32(a));
        v.push(next_down_f32(a));
    }
    let n = 64;
    for i in 0..=n {
        v.push(-6.0 + 12.0 * (i as f32) / (n as f32));
    }
    v
}

/// Every ordered pair `(l, u)`, `l <= u`, of the grid.
fn adversarial_intervals() -> Vec<(f32, f32)> {
    let g = adversarial_grid();
    let mut out = Vec::new();
    for &l in &g {
        for &u in &g {
            if l <= u {
                out.push((l, u));
            }
        }
    }
    out
}

fn finite_intervals() -> impl Iterator<Item = (f32, f32)> {
    adversarial_intervals()
        .into_iter()
        .filter(|(l, u)| l.is_finite() && u.is_finite())
}

/// Sample points of [l, u] as a caller would produce them in f32.
fn f32_samples(l: f32, u: f32) -> Vec<f32> {
    let mut xs = vec![l, u];
    for i in 1..32 {
        let x = l + (u - l) * (i as f32) / 32.0;
        if x.is_finite() && x >= l && x <= u {
            xs.push(x);
        }
    }
    xs
}

/// Assert that `band`, evaluated in f32 as `slope * x + intercept`, contains
/// the f64 reference `f` on [l, u]. `rel_slack * (1 + |f(x)|)` absorbs only
/// the reference's own evaluation error; there is no absolute slack, so a
/// subnormal-scale violation is not hidden.
fn assert_band_sound(name: &str, l: f32, u: f32, b: Band, f: &dyn Fn(f64) -> f64, rel_slack: f64) {
    for x in f32_samples(l, u) {
        let fx = f(x as f64);
        if !fx.is_finite() {
            continue;
        }
        let slack = rel_slack * (1.0 + fx.abs());
        let lo = b.0 * x + b.1;
        if lo.is_finite() {
            assert!(
                (lo as f64) <= fx + slack,
                "{name} LOWER line above f: l={l:e} u={u:e} x={x:e} line={lo:e} f={fx:e} band={b:?}"
            );
        }
        let hi = b.2 * x + b.3;
        if hi.is_finite() {
            assert!(
                (hi as f64) >= fx - slack,
                "{name} UPPER line below f: l={l:e} u={u:e} x={x:e} line={hi:e} f={fx:e} band={b:?}"
            );
        }
    }
}

fn sqrt_ref(x: f64) -> f64 {
    x.max(0.0).sqrt()
}

#[test]
fn adversarial_grid_log_band_is_sound() {
    let ln = |x: f64| x.ln();
    for (l, u) in finite_intervals().filter(|&(l, _)| l > 0.0) {
        assert_band_sound("log", l, u, band(log_linear_relaxation(l, u)), &ln, 1e-12);
    }
    // Tiny and narrow intervals. An absolute-width lower tangent used to be
    // unsound on the third by 1.1e-5.
    for (l, u) in [
        (f32::from_bits(1), f32::from_bits(2)),
        (1e-12, 5e-9),
        (1e-6, 1e-6 + 5e-9),
        (1e-12, 2e-12),
    ] {
        assert_band_sound("log", l, u, band(log_linear_relaxation(l, u)), &ln, 1e-12);
    }
}

#[test]
fn adversarial_grid_sqrt_band_is_sound() {
    for (l, u) in finite_intervals() {
        let b = band(sqrt_linear_relaxation(l, u));
        assert_band_sound("sqrt", l, u, b, &sqrt_ref, 1e-13);
    }
}

/// alpha-CROWN calls `sqrt_linear_relaxation_with_alpha` with a LEARNED
/// tangent point, so the band must be sound for every `mid`, not only for the
/// default t*. The sweep covers mid across [max(l, 0), u], at both endpoints,
/// and outside the valid range on every side the clamp handles (negative,
/// below `SQRT_ALPHA_MIN_MID`, above `u`); intervals with `l < 0` reach the
/// `min_intercept` branch.
#[test]
fn sqrt_alpha_band_is_sound_for_any_tangent_point() {
    let mut checked = 0usize;
    for (l, u) in finite_intervals().filter(|&(_, u)| u > 0.0) {
        let lo = l.max(0.0);
        let mut mids = vec![
            l,
            u,
            0.0,
            f32::midpoint(l, u),
            -1.0,
            2.0 * u + 1.0,
            f32::MIN_POSITIVE,
            1e-7,
            1e-6,
        ];
        mids.extend((0..=16).map(|i| lo + (u - lo) * (i as f32) / 16.0));
        for mid in mids.into_iter().filter(|m| m.is_finite()) {
            let b = band(sqrt_linear_relaxation_with_alpha(l, u, mid));
            assert_band_sound("sqrt_alpha", l, u, b, &sqrt_ref, 1e-13);
            checked += 1;
        }
    }
    assert!(
        checked > 100_000,
        "sweep collapsed: {checked} bands checked"
    );
}

#[test]
fn adversarial_grid_silu_band_is_sound() {
    let silu = |x: f64| x / (1.0 + (-x).exp());
    for (l, u) in finite_intervals() {
        let b = band(silu_sound_linear_relaxation(l, u));
        assert_band_sound("silu", l, u, b, &silu, 1e-11);
    }
}

#[test]
fn adversarial_grid_gelu_bands_are_sound() {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let gelu_erf = move |x: f64| 0.5 * x * (1.0 + libm::erf(x * inv_sqrt2));
    let c: f64 = (2.0 / std::f64::consts::PI).sqrt();
    let gelu_tanh = move |x: f64| 0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
    for (l, u) in finite_intervals() {
        let erf_band = gelu_sound_linear_relaxation(l, u);
        assert_band_sound("gelu_erf", l, u, erf_band, &gelu_erf, 1e-11);
        let tanh_band = gelu_tanh_sound_linear_relaxation(l, u);
        assert_band_sound("gelu_tanh", l, u, tanh_band, &gelu_tanh, 1e-11);
    }
}

/// The scalar ReLU relaxation through the public `ReLULayer` CROWN backward
/// pass with a one-neuron identity seed, so the check covers the composition
/// and its directed rounding too. The stored coefficients carry per-entry
/// error that `concretize_sound` consumes, so soundness is asserted on the
/// concretized interval (the form verdict paths read), which must enclose
/// [ReLU(l), ReLU(u)] exactly.
#[test]
fn adversarial_grid_relu_layer_concretizes_soundly() {
    let layer = ReLULayer::new();
    for (l, u) in finite_intervals() {
        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), l),
            ArrayD::from_elem(IxDyn(&[1]), u),
        )
        .expect("valid pre-activation bounds");
        let out = layer
            .propagate_linear_with_bounds(&LinearBounds::identity(1), &pre)
            .expect("ReLU CROWN backward");
        let conc = out.concretize_sound(&pre);
        let (relu_l, relu_u) = (l.max(0.0), u.max(0.0));
        assert!(
            conc.lower()[[0]] <= relu_l && conc.upper()[[0]] >= relu_u,
            "relu concretized bound [{:e}, {:e}] does not enclose [{relu_l:e}, {relu_u:e}] \
             on l={l:e} u={u:e}",
            conc.lower()[[0]],
            conc.upper()[[0]],
        );
    }
}

/// ZERO-tolerance pin on the `l = -inf` GELU branch at the one `u` where the
/// verdict needs no assumption about `erff`: at u = f32::from_bits(1) the true
/// GELU(u) = 7.006e-46 > 0, while `0.5f32 * u` rounds to +0.0 under any erff
/// (half an ULP, ties to even). A relaxation that published
/// `max(gelu_eval(u), 0.0) = 0.0` as the upper intercept was a false bound at
/// the interval's own endpoint. The grid tests cannot reach this: they skip
/// infinite `l`, and their relative slack dwarfs 7e-46.
#[test]
fn gelu_infinite_lower_branch_is_exact_at_the_subnormal_endpoint() {
    let u = f32::from_bits(1);
    let true_gelu =
        0.5 * f64::from(u) * (1.0 + libm::erf(f64::from(u) * std::f64::consts::FRAC_1_SQRT_2));
    assert!(true_gelu > 0.0, "precondition: GELU({u:e}) = {true_gelu:e}");

    let b = gelu_sound_linear_relaxation(f32::NEG_INFINITY, u);
    assert_eq!(b.0, 0.0, "lower slope must be 0 on the constant branch");
    assert_eq!(b.2, 0.0, "upper slope must be 0 on the constant branch");
    assert!(
        f64::from(b.3) >= true_gelu,
        "GELU(x) <= {:e} claimed on (-inf, {u:e}], but GELU({u:e}) = {true_gelu:e}",
        b.3,
    );
    // GELU's global minimum, attained inside (-inf, u]. The certified constant
    // bounds GELU at the f32 bisected critical point, about 1e-15 above the
    // true minimum, and sits one downward f32 step (~1.5e-8) below that; this
    // is what notices if that margin ever stops covering the gap.
    let true_min = -0.169_971_207_479_903_66_f64;
    assert!(
        f64::from(b.1) <= true_min,
        "GELU(x) >= {:e} claimed on (-inf, {u:e}], but GELU attains {true_min:e} there",
        b.1,
    );
}

/// Assert the ReLU envelope holds at `x` with NO tolerance.
///
/// The general tolerance's 1e-6 absolute floor would make every check on these
/// subnormal-scale intervals vacuous. f64 represents every f32 product exactly
/// (including subnormal ones), so this compares the stored lines to `ReLU(x)`
/// without any slack.
fn assert_relu_envelope_exact(l: f32, u: f32, x: f32) {
    let lines: Lines = relu_linear_relaxation(l, u).into();
    let x64 = x as f64;
    let relu_x = x64.max(0.0);
    let lower = lines.lower_at(x64);
    let upper = lines.upper_at(x64);
    assert!(
        lower <= relu_x,
        "relu LOWER unsound on [{l:e}, {u:e}] at x={x:e}: {lower:e} > {relu_x:e}\n  {lines:?}"
    );
    assert!(
        relu_x <= upper,
        "relu UPPER unsound on [{l:e}, {u:e}] at x={x:e}: {upper:e} < {relu_x:e}\n  {lines:?}"
    );
}

/// The inputs on which Kani refuted an earlier ReLU relaxation: an
/// extreme endpoint ratio, a concrete-playback interior point, and subnormal
/// endpoints; plus extreme ratios on both sides of that relaxation's 1e-30
/// ratio threshold.
#[test]
fn relu_envelope_holds_on_the_335_counterexample_inputs() {
    let tiny = f32::from_bits(1); // smallest positive subnormal
    let small_normal = f32::MIN_POSITIVE * 2.0;
    let intervals: &[(f32, f32)] = &[
        (-4.411234e-36, 8.625892e-39),
        (f32::from_bits(0xa03b_f0eb), f32::from_bits(0x20b2_1f9a)),
        (-tiny, tiny),
        (-tiny, small_normal),
        (-small_normal, tiny),
        (-tiny * 100.0, tiny),
        (-tiny, tiny * 100.0),
        (-1e38, 1e-38),
        (-1e-38, 1e38),
        (-1e-30, 1e-35),
        (-1e-35, 1e-30),
        (-1.0, 1e-38),
        (-1e-38, 1.0),
        (-1.0, 1e-29),
        (-1.0, 1e-31),
        (-1e-29, 1.0),
        (-1e-31, 1.0),
    ];
    for &(l, u) in intervals {
        for x in [l, l / 2.0, 0.0, u / 2.0, u] {
            assert_relu_envelope_exact(l, u, x);
        }
    }
    // The concrete-playback point: one f32 ULP below the upper endpoint.
    assert_relu_envelope_exact(
        f32::from_bits(0xa03b_f0eb),
        f32::from_bits(0x20b2_1f9a),
        f32::from_bits(0x20b2_1f99),
    );
}
