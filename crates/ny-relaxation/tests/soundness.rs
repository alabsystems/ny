// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness tests for the scalar convex relaxations used by CROWN.
//!
//! For every activation `f`, the crate produces a linear relaxation
//!
//! ```text
//!   lower_slope * x + lower_intercept <= f(x) <= upper_slope * x + upper_intercept
//! ```
//!
//! valid for all `x` in `[l, u]`. An *unsound* relaxation (where the true
//! value escapes the linear envelope at some `x` in the interval) would let
//! the verifier emit a wrong "verified" verdict. These tests densely sample
//! each interval and assert the envelope contains `f(x)` at every sample, plus
//! `lower(x) <= upper(x)` everywhere.
//!
//! Sampling is deterministic (a fixed grid plus the interval endpoints and
//! known critical points), so failures are reproducible. We compare in f64 and
//! allow a small relative tolerance to absorb the f32 evaluation noise the
//! production code itself accounts for via directed rounding; the relaxations
//! are designed to be sound including those rounding terms, so the tolerance
//! is intentionally tiny.

use ny_relaxation::{
    abs_linear_relaxation, exp_linear_relaxation, gelu_eval, gelu_sound_linear_relaxation,
    gelu_tanh_sound_linear_relaxation, log_linear_relaxation, pow2_linear_relaxation,
    relu_crown_relaxation, silu_eval, silu_sound_linear_relaxation, sqrt_linear_relaxation,
    GeluApproximation, LinearRelaxation,
};

/// A linear relaxation expressed as the four scalar coefficients, regardless of
/// whether the underlying API returns a tuple or a [`LinearRelaxation`].
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
    fn lower_at(&self, x: f64) -> f64 {
        self.lower_slope * x + self.lower_intercept
    }
    fn upper_at(&self, x: f64) -> f64 {
        self.upper_slope * x + self.upper_intercept
    }
}

/// Tolerance for the soundness comparison.
///
/// The production relaxations evaluate in f32 and add directed-rounding
/// margins, so the *intended* envelope is sound even after f32 noise. We
/// compare the true f64 value against the f32-evaluated line; a tiny relative
/// slack absorbs the difference between "evaluated in f32 by the verifier" and
/// "evaluated in f64 by this test". A genuine unsoundness (the chord on the
/// wrong side of a convex/concave region) is gross and trips this easily.
fn tol(x: f64, fx: f64, line: f64) -> f64 {
    let scale = x.abs().max(fx.abs()).max(line.abs()).max(1.0);
    // ~4 f32 ULPs of relative slack plus a small absolute floor.
    scale * 6.0e-6 + 1.0e-6
}

/// Assert the linear envelope contains `f` at densely-sampled points in `[l, u]`.
fn assert_envelope_contains<F>(name: &str, l: f32, u: f32, lines: Lines, f: F)
where
    F: Fn(f64) -> f64,
{
    assert!(l <= u, "{name}: bad interval [{l}, {u}]");

    // Every fixture in this suite has a finite, supported domain. Treating a
    // non-finite relaxation as "trivially sound" used to turn regressions into
    // vacuous passes: NaN is not an authenticated [-inf,+inf] fallback, and an
    // unexpected infinity means the useful relaxation disappeared. Pin the
    // production contract instead.
    assert!(
        lines.lower_slope.is_finite()
            && lines.lower_intercept.is_finite()
            && lines.upper_slope.is_finite()
            && lines.upper_intercept.is_finite(),
        "{name}: finite interval [{l}, {u}] produced a non-finite envelope: {lines:?}"
    );

    let l64 = l as f64;
    let u64 = u as f64;

    // Build a dense, deterministic sample set: a uniform grid plus the exact
    // endpoints (where chord/tangent relaxations are tightest and most likely
    // to be violated by an off-by-rounding error).
    const N: usize = 401;
    let mut samples: Vec<f64> = Vec::with_capacity(N + 8);
    for i in 0..N {
        let t = i as f64 / (N as f64 - 1.0);
        samples.push(l64 + t * (u64 - l64));
    }
    // Extra emphasis on the endpoints and a few interior fractions.
    for &t in &[0.0, 1.0, 0.5, 0.25, 0.75, 1.0 / 3.0, 2.0 / 3.0] {
        samples.push(l64 + t * (u64 - l64));
    }

    for &x in &samples {
        if !x.is_finite() {
            continue;
        }
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

/// Representative intervals exercising every regime: deep-negative, deep-positive,
/// straddling zero, wide, narrow, near-point, and asymmetric.
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

/// Positive-domain intervals for functions defined only on x > 0 (sqrt, log).
/// Intervals with `l >= 0`. Suitable for `sqrt` (defined AT 0) but NOT for
/// `log`, whose domain is strictly `x > 0` — see [`strictly_positive_intervals`].
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

#[test]
fn relu_relaxation_is_sound() {
    for (l, u) in general_intervals() {
        let lines: Lines = relu_crown_relaxation(l, u).into();
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
    // exp grows fast; keep magnitudes bounded so f32 vs f64 noise stays tiny.
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

/// `log`'s domain is STRICTLY `x > 0`, so it gets its own fixture list.
///
/// `positive_intervals()` includes `(0.0, 4.0)`, which is correct for `sqrt`
/// (defined at 0) and out of domain for `log`: `ln(0) = -inf`, so the only sound
/// lower bound there is `-inf` and no finite envelope exists. Production
/// correctly returns the authenticated maximally-loose fallback for `l <= 0`,
/// which `assert_envelope_contains` then rejects for being non-finite — by
/// design, since a vacuous infinite envelope must not read as "sound".
fn strictly_positive_intervals() -> Vec<(f32, f32)> {
    positive_intervals()
        .into_iter()
        .filter(|&(l, _)| l > 0.0)
        .collect()
}

#[test]
fn log_relaxation_is_sound() {
    for (l, u) in strictly_positive_intervals() {
        // NOTE: `log` does NOT clamp its domain. This comment used to say it
        // clamped to >= 1e-10 and evaluated the reference as `x.max(1e-10).ln()`
        // to match — but that clamp was REMOVED as unsound
        // (#log-epsilon-nonenclosing): raising `l` MOVES THE DOMAIN, so the
        // relaxation built for `[1e-10, u]` sits ABOVE `ln` on `(l, 1e-10)` and
        // stops enclosing. `envelope_audit_expologpow` measured 10,484,626 of
        // 37,861,389 sampled points violating the lower line, worst violation
        // 64.47, every one of them inside the clamped region. Compare against
        // the TRUE `ln`, which is what production now bounds.
        let lines: Lines = log_linear_relaxation(l, u).into();
        assert_envelope_contains("log", l, u, lines, |x| x.ln());
    }
}

/// The domain edge itself, pinned rather than skipped: an interval touching 0 is
/// OUT OF DOMAIN for `log`, and the sound answer is the maximally-loose
/// envelope, not a finite one. This is what keeps the filter above honest — if
/// production ever started returning a finite lower bound here it would be
/// claiming `ln` is bounded below on `(0, 4]`, which is false.
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

/// Stress test: many randomly-but-deterministically generated intervals per
/// activation. A linear-congruential generator keeps this dependency-free and
/// reproducible while covering far more of the input space than the curated
/// interval lists.
#[test]
fn relaxations_sound_on_dense_random_intervals() {
    // Deterministic xorshift-style PRNG; no external crate needed.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Uniform f32 in [lo, hi].
    let uniform = |lo: f32, hi: f32, r: u64| -> f32 {
        let frac = (r >> 11) as f64 / (1u64 << 53) as f64;
        (lo as f64 + frac * (hi as f64 - lo as f64)) as f32
    };

    for _ in 0..2000 {
        let a = uniform(-6.0, 6.0, next());
        let b = uniform(-6.0, 6.0, next());
        let (l, u) = if a <= b { (a, b) } else { (b, a) };

        assert_envelope_contains("relu", l, u, relu_crown_relaxation(l, u).into(), |x| {
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

        // exp on a tamer range to keep magnitudes bounded.
        let ea = uniform(-4.0, 3.0, next());
        let eb = uniform(-4.0, 3.0, next());
        let (el, eu) = if ea <= eb { (ea, eb) } else { (eb, ea) };
        assert_envelope_contains("exp", el, eu, exp_linear_relaxation(el, eu).into(), |x| {
            x.exp()
        });

        // sqrt / log on the positive domain.
        let pa = uniform(1e-3, 10.0, next());
        let pb = uniform(1e-3, 10.0, next());
        let (pl, pu) = if pa <= pb { (pa, pb) } else { (pb, pa) };
        assert_envelope_contains("sqrt", pl, pu, sqrt_linear_relaxation(pl, pu).into(), |x| {
            x.max(0.0).sqrt()
        });
        assert_envelope_contains("log", pl, pu, log_linear_relaxation(pl, pu).into(), |x| {
            x.max(1e-10).ln()
        });
    }
}
