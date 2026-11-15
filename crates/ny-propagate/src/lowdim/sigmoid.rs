// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lowdim-bundle (1) — rigorous linear planes of the logistic sigmoid for the
//! lowdim lane's [`super::Op::Sigmoid`] (dist_shift_2023's
//! `Gemm -> ReLU -> Gemm -> Sigmoid -> ...` generator).
//!
//! # Soundness
//!
//! A plane is `(a, b)` with `a z + b <= sigmoid(z)` (LOWER) or
//! `sigmoid(z) <= a z + b` (UPPER) for EVERY `z` in the source's rigorous
//! concrete box `[l, u]`. The slope `a` is a heuristic choice (chord, tangent
//! or zero); the intercept is always a rigorous bound of the extreme of
//! `h(z) = sigmoid(z) - a z` over `[l, u]`, so a slope that is "wrong" only
//! loosens the plane, never invalidates it:
//!
//! * `sigmoid` is CONVEX on `(-inf, 0]` and CONCAVE on `[0, inf)`, so `h` is
//!   too (`-a z` is linear). `[l, u]` is split at 0.
//! * The minimum of a CONCAVE `h` (and the maximum of a CONVEX `h`) on an
//!   interval is attained at an endpoint: exactly `min(h(p), h(q))`.
//! * For the other extreme the tangent inequality of `h` at any `t` in the
//!   piece is used: convex `h(z) >= h(t) + h'(t)(z - t)`, concave
//!   `h(z) <= h(t) + h'(t)(z - t)`. `h'(t) = sigmoid'(t) - a` is enclosed in
//!   `[s(1-s) - a -/+ DERIV_ERR]` (see [`DERIV_ERR`]) and `z - t` in the
//!   outward-rounded `[p - t, q - t]`; the bilinear extreme over that box is
//!   at a corner, each corner product rounded outward. `t` is the stationary
//!   point of `h` on the piece when one exists (then the charged term is
//!   ~`1e-12 * width`), else the endpoint where the extreme lies.
//! * `sigmoid(t)` itself is enclosed by [`sig_bounds`]: the f64 evaluation
//!   `1 / (1 + exp(-t))` carries a few ulps of relative error (libm `exp`,
//!   one add, one divide); `1e-12` relative plus one outward step dominates
//!   it by four orders of magnitude, and the absolute `1e-300` covers the
//!   subnormal range (the same margins as [`super::sigmoid_down`]).
//! * Every `h(t)` evaluation charges its product and difference by `gslack`
//!   and rounds outward.
//!
//! The constant planes `(0, sigmoid_down(l))` and `(0, sigmoid_up(u))` are
//! always valid (sigmoid is increasing); they are candidates, and the only
//! planes on a box whose magnitude exceeds [`MAG_CAP`].

use super::{gslack, next_down, next_up};

/// Absolute bound on `|sigmoid'(t) - s (1 - s)|` for `s` the f64 sigmoid at
/// `t`: `s` is within ~`4 * 2^-53` (relative, so absolute, as `s <= 1`) of the
/// true value, `1 - s` and the product add one rounding each, so the total is
/// below `1e-15`; `1e-12` dominates it by three orders of magnitude.
pub(super) const DERIV_ERR: f64 = 1e-12;

/// Above this magnitude a box end yields only the constant planes (the
/// tangent algebra would multiply huge operands for no tightness: sigmoid is
/// flat to within `1e-300` of `0` or `1` far before it).
pub(super) const MAG_CAP: f64 = 1e30;

/// The plain f64 sigmoid (never a bound by itself).
#[inline]
pub(super) fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `(lower, upper)` rigorous bounds of `sigmoid(x)` for a FINITE `x`, both in
/// `[0, 1]`. The lower formula is exactly [`super::sigmoid_down`]'s.
#[inline]
pub(super) fn sig_bounds(x: f64) -> (f64, f64) {
    let s = sig(x);
    let lo = (next_down(s * (1.0 - 1e-12)) - 1e-300).max(0.0);
    let hi = (next_up(s * (1.0 + 1e-12)) + 1e-300).min(1.0);
    (lo, hi)
}

/// Rigorous upper bound of `sigmoid(x)` for a finite `x`.
#[inline]
pub(super) fn sig_up(x: f64) -> f64 {
    sig_bounds(x).1
}

/// Rigorous lower bound of `sigmoid(x)` for a finite `x`.
#[inline]
pub(super) fn sig_down(x: f64) -> f64 {
    sig_bounds(x).0
}

/// The f64 derivative `s (1 - s)` (within [`DERIV_ERR`] of the true one).
#[inline]
fn deriv(x: f64) -> f64 {
    let s = sig(x);
    s * (1.0 - s)
}

/// Rigorous `[lo, hi]` of `h(t) = sigmoid(t) - a t`.
fn h_at(t: f64, a: f64) -> (f64, f64) {
    let (sd, su) = sig_bounds(t);
    let at = a * t;
    let lo = next_down(sd - at - gslack(3, sd.abs() + at.abs()));
    let hi = next_up(su - at + gslack(3, su.abs() + at.abs()));
    (lo, hi)
}

/// The stationary point of `h(z) = sigmoid(z) - a z` on the CONVEX half
/// (`z <= 0`, `concave == false`) or the CONCAVE half (`z >= 0`): where
/// `sigmoid'(z) = s (1 - s) = a`, i.e. `s = (1 -/+ sqrt(1 - 4a)) / 2`, the
/// smaller root written as `2a / (1 + sqrt(1 - 4a))` (no cancellation).
/// `None` when `a` is outside `(0, 1/4)` (no stationary point) or the
/// logit is not finite. A heuristic only: soundness never depends on it.
fn stationary(a: f64, concave: bool) -> Option<f64> {
    if !(a > 0.0 && a < 0.25) {
        return None;
    }
    let r = (1.0 - 4.0 * a).sqrt();
    let small = 2.0 * a / (1.0 + r);
    let s = if concave { 1.0 - small } else { small };
    let z = (s / (1.0 - s)).ln();
    z.is_finite().then_some(z)
}

/// Where to take the tangent of `h` on `[p, q]` (one half): the stationary
/// point when it exists, else the endpoint where the wanted extreme lies (a
/// convex MIN or a concave MAX). Always inside `[p, q]`.
fn tangent_point(p: f64, q: f64, a: f64, concave: bool) -> f64 {
    let t = match stationary(a, concave) {
        Some(z) => z,
        // The stationary point exists but lies beyond f64's logit range
        // (`a` below ~1e-16 on the concave half): past `q` on the concave
        // half (h increases on `[p, q]`), before `p` on the convex one.
        None if a > 0.0 && a < 0.25 => {
            if concave {
                q
            } else {
                p
            }
        }
        // No stationary point: `h' = sigmoid' - a` has one sign on the half.
        // `a <= 0`: h increases (convex min at p, concave max at q);
        // `a >= 1/4`: h decreases (convex min at q, concave max at p).
        None => match (a <= 0.0, concave) {
            (true, false) | (false, true) => p,
            (true, true) | (false, false) => q,
        },
    };
    if t.is_nan() {
        p
    } else {
        t.max(p).min(q)
    }
}

/// Enclosure `[g_lo, g_hi]` of `h'(t) = sigmoid'(t) - a`.
fn slope_enclosure(t: f64, a: f64) -> (f64, f64) {
    let d = deriv(t);
    (
        next_down(next_down(d - DERIV_ERR) - a),
        next_up(next_up(d + DERIV_ERR) - a),
    )
}

/// Rigorous lower bound of `min h` over `[p, q]`, `q <= 0` (convex half).
fn convex_min(p: f64, q: f64, a: f64) -> f64 {
    let t = tangent_point(p, q, a, false);
    let (h_lo, _) = h_at(t, a);
    let (g_lo, g_hi) = slope_enclosure(t, a);
    let (w_lo, w_hi) = (next_down(p - t), next_up(q - t));
    let m = [
        next_down(g_lo * w_lo),
        next_down(g_lo * w_hi),
        next_down(g_hi * w_lo),
        next_down(g_hi * w_hi),
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min);
    next_down(h_lo + m)
}

/// Rigorous upper bound of `max h` over `[p, q]`, `p >= 0` (concave half).
fn concave_max(p: f64, q: f64, a: f64) -> f64 {
    let t = tangent_point(p, q, a, true);
    let (_, h_hi) = h_at(t, a);
    let (g_lo, g_hi) = slope_enclosure(t, a);
    let (w_lo, w_hi) = (next_down(p - t), next_up(q - t));
    let m = [
        next_up(g_lo * w_lo),
        next_up(g_lo * w_hi),
        next_up(g_hi * w_lo),
        next_up(g_hi * w_hi),
    ]
    .into_iter()
    .fold(f64::NEG_INFINITY, f64::max);
    next_up(h_hi + m)
}

/// Rigorous lower bound of `min_{z in [l, u]} sigmoid(z) - a z`.
pub(super) fn h_min(l: f64, u: f64, a: f64) -> f64 {
    let concave_min = |p: f64, q: f64| h_at(p, a).0.min(h_at(q, a).0);
    if u <= 0.0 {
        convex_min(l, u, a)
    } else if l >= 0.0 {
        concave_min(l, u)
    } else {
        convex_min(l, 0.0, a).min(concave_min(0.0, u))
    }
}

/// Rigorous upper bound of `max_{z in [l, u]} sigmoid(z) - a z`.
pub(super) fn h_max(l: f64, u: f64, a: f64) -> f64 {
    let convex_max = |p: f64, q: f64| h_at(p, a).1.max(h_at(q, a).1);
    if u <= 0.0 {
        convex_max(l, u)
    } else if l >= 0.0 {
        concave_max(l, u, a)
    } else {
        convex_max(l, 0.0).max(concave_max(0.0, u, a))
    }
}

/// `[a_lo, b_lo, a_hi, b_hi]`: `a_lo z + b_lo <= sigmoid(z) <= a_hi z + b_hi`
/// for every `z` in `[l, u]`, both slopes in `[0, 1/4]`. `None` for a
/// non-finite or unordered box (the caller poisons the domain).
///
/// Candidates: in the convex region the tangent at the midpoint (lower) and
/// the chord (upper); in the concave region the reverse; across 0 the chord
/// and the end tangents; always the constant planes. Among the valid
/// candidates the LOWER plane with the highest value at the midpoint (and the
/// UPPER plane with the lowest) is kept — a tightness choice only.
pub(super) fn planes(l: f64, u: f64) -> Option<[f64; 4]> {
    if !(l.is_finite() && u.is_finite() && l <= u) {
        return None;
    }
    let (c_lo, _) = sig_bounds(l);
    let (_, c_hi) = sig_bounds(u);
    if l.abs() > MAG_CAP || u.abs() > MAG_CAP {
        return Some([0.0, c_lo, 0.0, c_hi]);
    }
    let mid = 0.5 * l + 0.5 * u;
    let w = u - l;
    let chord = if w > 0.0 && w.is_finite() {
        let k = (sig(u) - sig(l)) / w;
        if k.is_finite() {
            k.clamp(0.0, 0.25)
        } else {
            0.0
        }
    } else {
        deriv(mid)
    };
    let tan_mid = deriv(mid);
    let (lo_cands, hi_cands) = if u <= 0.0 {
        ([tan_mid, chord], [chord, chord])
    } else if l >= 0.0 {
        ([chord, chord], [tan_mid, chord])
    } else {
        ([chord, deriv(l)], [chord, deriv(u)])
    };
    let (mut lo, mut lo_at) = ((0.0, c_lo), c_lo);
    for a in lo_cands {
        if !(a.is_finite() && a >= 0.0) {
            continue;
        }
        let b = h_min(l, u, a);
        let at = a * mid + b;
        if b.is_finite() && at > lo_at {
            lo = (a, b);
            lo_at = at;
        }
    }
    let (mut hi, mut hi_at) = ((0.0, c_hi), c_hi);
    for a in hi_cands {
        if !(a.is_finite() && a >= 0.0) {
            continue;
        }
        let b = h_max(l, u, a);
        let at = a * mid + b;
        if b.is_finite() && at < hi_at {
            hi = (a, b);
            hi_at = at;
        }
    }
    Some([lo.0, lo.1, hi.0, hi.1])
}
