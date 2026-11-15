// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The checker's own f64 arithmetic: directed rounding by one-ulp steps and
//! a-priori Higham charges. No fused multiply-add anywhere (cross-arch
//! reproducibility; ratcheted), no rounding-mode changes, no libm.
//!
//! # The error model
//!
//! IEEE-754 binary64 with round-to-nearest-even: every `+ - * / sqrt` returns
//! the correctly rounded result, so
//!
//! * `fl(a op b)` lies within half an ulp of the exact result (or within
//!   `2^-1075` when the result is subnormal or underflows to zero), hence
//!   `down(fl(a op b)) <= a op b <= up(fl(a op b))` for finite results;
//! * a dot product `fl(Σ_k a_k b_k)` of `n` terms, evaluated in ANY order,
//!   differs from the exact sum by at most `γ_n Σ_k |a_k b_k| + n·2^-1074`
//!   (Higham, Accuracy and Stability, Thm 3.1, plus the underflow term:
//!   products lose at most `2^-1075` absolutely and subnormal additions are
//!   exact), with `γ_n = n·u / (1 - n·u)`, `u = 2^-53`.
//!
//! [`gamma_up`] returns `2(n+2)u`, which dominates `γ_n` by a factor of about
//! two for every `n` with `n·u <= 0.01` (i.e. `n` below 9·10^13, far above any
//! fan-in this checker sees); the slack also absorbs the relative error of the
//! magnitude sums the charge multiplies, which are themselves accumulated with
//! upward rounding ([`sum_up`], [`dot_abs_up`]).

/// The unit roundoff `u = 2^-53`.
pub const U: f64 = f64::EPSILON / 2.0;

/// The smallest positive subnormal, `2^-1074`.
pub const TINY: f64 = f64::from_bits(1);

/// Next representable value toward `+inf` (`+inf` and NaN are fixed points).
#[inline]
#[must_use]
pub fn up(x: f64) -> f64 {
    x.next_up()
}

/// Next representable value toward `-inf`.
#[inline]
#[must_use]
pub fn down(x: f64) -> f64 {
    x.next_down()
}

/// An upper bound on Higham's `γ_n` for a sum of `n` rounded products:
/// `2(n+2)u`, itself rounded up.
#[must_use]
pub fn gamma_up(n: usize) -> f64 {
    up(2.0 * (n as f64 + 2.0) * U)
}

/// The a-priori charge for accumulating `n` rounded terms per element of a
/// vector whose magnitude-weighted term sum is at most `s_up` (that is,
/// `Σ_i M_i Σ_k |term_k,i| <= s_up`), over a tensor whose magnitude total is
/// at most `msum_up`: `γ_n·S + n·2^-1074·Σ M`, every step rounded up.
#[must_use]
pub fn charge(n: usize, s_up: f64, msum_up: f64) -> f64 {
    let rel = up(gamma_up(n) * s_up);
    let underflow = up(up((n as f64 + 1.0) * msum_up) * TINY);
    up(rel + underflow)
}

/// `a + b` rounded toward `+inf`.
#[inline]
#[must_use]
pub fn add_up(a: f64, b: f64) -> f64 {
    up(a + b)
}

/// `a + b` rounded toward `-inf`.
#[inline]
#[must_use]
pub fn add_down(a: f64, b: f64) -> f64 {
    down(a + b)
}

/// `a * b` rounded toward `+inf`.
#[inline]
#[must_use]
pub fn mul_up(a: f64, b: f64) -> f64 {
    up(a * b)
}

/// `a * b` rounded toward `-inf`.
#[inline]
#[must_use]
pub fn mul_down(a: f64, b: f64) -> f64 {
    down(a * b)
}

/// `Σ |a_i| * b_i` for `b_i >= 0`, rounded up after every operation.
#[must_use]
pub fn dot_abs_up(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for (x, m) in a.iter().zip(b) {
        s = up(s + up(x.abs() * *m));
    }
    s
}

/// `Σ x_i` of non-negative values, rounded up after every addition.
#[must_use]
pub fn sum_up(xs: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for x in xs {
        s = up(s + *x);
    }
    s
}

/// A lower bound on `Σ a_i * b_i`, every product and sum rounded down.
#[must_use]
pub fn dot_down(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        s = down(s + down(*x * *y));
    }
    s
}

/// A lower bound on `min_{x in [lo, hi]} Σ c_i x_i` (every step rounded
/// down). Returns NaN if any input is NaN, so a caller testing `> 0`
/// rejects it.
#[must_use]
pub fn concretize_down(c: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for ((ci, l), h) in c.iter().zip(lo).zip(hi) {
        let term = if *ci >= 0.0 {
            down(*ci * *l)
        } else {
            down(*ci * *h)
        };
        s = down(s + term);
    }
    s
}

/// An outward-rounded interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Iv {
    /// Lower end.
    pub lo: f64,
    /// Upper end.
    pub hi: f64,
}

/// Sum, rounded outward.
impl std::ops::Add for Iv {
    type Output = Self;

    fn add(self, o: Self) -> Self {
        Self {
            lo: down(self.lo + o.lo),
            hi: up(self.hi + o.hi),
        }
    }
}

impl std::ops::AddAssign for Iv {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

/// Difference, rounded outward.
impl std::ops::Sub for Iv {
    type Output = Self;

    fn sub(self, o: Self) -> Self {
        Self {
            lo: down(self.lo - o.hi),
            hi: up(self.hi - o.lo),
        }
    }
}

/// Product of two intervals, rounded outward.
impl std::ops::Mul for Iv {
    type Output = Self;

    fn mul(self, o: Self) -> Self {
        let c = [
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for v in c {
            if v.is_nan() {
                return Self {
                    lo: f64::NAN,
                    hi: f64::NAN,
                };
            }
            lo = lo.min(v);
            hi = hi.max(v);
        }
        Self {
            lo: down(lo),
            hi: up(hi),
        }
    }
}

impl Iv {
    /// The degenerate interval `[x, x]`.
    #[must_use]
    pub const fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    /// `[lo, hi]` as given.
    #[must_use]
    pub const fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Product with a scalar, rounded outward.
    #[must_use]
    pub fn scale(self, w: f64) -> Self {
        let a = w * self.lo;
        let b = w * self.hi;
        if w >= 0.0 {
            Self {
                lo: down(a),
                hi: up(b),
            }
        } else {
            Self {
                lo: down(b),
                hi: up(a),
            }
        }
    }

    /// `max(x, 0)`, exact. A NaN end stays NaN (`f64::max` alone would
    /// launder it into `0`).
    #[must_use]
    pub fn relu(self) -> Self {
        let r = |v: f64| if v.is_nan() { v } else { v.max(0.0) };
        Self {
            lo: r(self.lo),
            hi: r(self.hi),
        }
    }

    /// Whether both ends are finite and ordered.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
}

/// The ReLU upper line over `[l, u]` with `l < 0 < u`: slope `s` clamped to
/// `[0, 1]` and an intercept `t` rounded so the line `s·z + t` is `>=` relu at
/// BOTH endpoints (relu is convex, so it is then `>=` relu on all of `[l, u]`):
///
/// * at `z = l`: `s·l + t >= 0` because `t >= up(-s·l) >= -s·l`;
/// * at `z = u`: `s·u + t >= u` because `t >= up(u - down(s·u)) >= u - s·u`.
///
/// `None` for non-finite or unordered input.
#[must_use]
pub fn relu_chord(l: f64, u: f64) -> Option<(f64, f64)> {
    if !(l.is_finite() && u.is_finite() && l < 0.0 && u > 0.0) {
        return None;
    }
    let den = u - l;
    let raw = u / den;
    let s = if raw.is_nan() {
        1.0
    } else {
        raw.clamp(0.0, 1.0)
    };
    let t0 = up(-(s * l));
    let t1 = up(u - down(s * u));
    let t = t0.max(t1).max(0.0);
    if !t.is_finite() {
        return None;
    }
    Some((s, t))
}

/// The upper and lower f64 neighbours of a decimal string, and whether the
/// decimal is exactly representable. `None` if the string is not a finite
/// decimal (Rust's correctly rounded `f64::from_str` does the parsing).
#[must_use]
pub fn decimal_bounds(text: &str, exact: bool) -> Option<(f64, f64)> {
    let v: f64 = text.parse().ok()?;
    if !v.is_finite() {
        return None;
    }
    if exact {
        Some((v, v))
    } else {
        Some((down(v), up(v)))
    }
}

#[cfg(test)]
mod tests {
    use super::{charge, down, gamma_up, relu_chord, up, Iv, U};

    #[test]
    fn directed_steps_bracket() {
        let x = 0.1f64;
        assert!(down(x) < x && x < up(x));
        assert_eq!(up(0.0), f64::from_bits(1));
        assert!(gamma_up(1) >= 3.0 * U);
        assert!(charge(4, 1.0, 1.0) > 0.0);
    }

    #[test]
    fn interval_mul_brackets() {
        let a = Iv::new(-1.0, 2.0);
        let b = Iv::new(3.0, 4.0);
        let c = a * b;
        assert!(c.lo <= -4.0 && c.hi >= 8.0);
    }

    #[test]
    fn chord_is_above_endpoints() {
        let (s, t) = relu_chord(-1.0, 3.0).expect("chord");
        assert!(-s + t >= 0.0);
        assert!(s * 3.0 + t >= 3.0);
    }
}
