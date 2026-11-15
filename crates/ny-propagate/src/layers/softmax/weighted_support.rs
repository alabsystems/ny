// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Checked scalar softmax support. Search proposes a threshold; an outward
//! interval scan checks `sum min((c_i-tau) exp(l_i), (c_i-tau) exp(u_i)) >= 0`.
//! Only the checked scan authorizes the lower bound. Masked entries must be
//! removed by the caller; finite visible entries are retained even on underflow.

/// Binary64 interval operations charge each elementary rounding separately.
/// These enclose the mathematical real operator, not an unspecified device
/// implementation of softmax. Execution error belongs to the frontend contract.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Interval {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

pub(crate) fn down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    f64::from_bits(if x > 0.0 {
        x.to_bits() - 1
    } else {
        x.to_bits() + 1
    })
}
pub(crate) fn up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(if x > 0.0 {
        x.to_bits() + 1
    } else {
        x.to_bits() - 1
    })
}

impl Interval {
    pub(crate) fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    pub(crate) fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }
    pub(crate) fn add(self, other: Self) -> Self {
        if other.lo == 0.0 && other.hi == 0.0 {
            return self;
        }
        if self.lo == 0.0 && self.hi == 0.0 {
            return other;
        }
        Self::new(down(self.lo + other.lo), up(self.hi + other.hi))
    }
    pub(crate) fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }
    pub(crate) fn sub(self, other: Self) -> Self {
        if self.lo == self.hi && other.lo == other.hi && self.lo == other.lo {
            return Self::point(0.0);
        }
        self.add(other.neg())
    }
    pub(crate) fn mul(self, other: Self) -> Self {
        if other.lo == 1.0 && other.hi == 1.0 {
            return self;
        }
        if self.lo == 1.0 && self.hi == 1.0 {
            return other;
        }
        if (self.lo == 0.0 && self.hi == 0.0) || (other.lo == 0.0 && other.hi == 0.0) {
            return Self::point(0.0);
        }
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if products.iter().any(|p| p.is_nan()) {
            return Self::new(f64::NAN, f64::NAN);
        }
        Self::new(
            down(products.into_iter().fold(f64::INFINITY, f64::min)),
            up(products.into_iter().fold(f64::NEG_INFINITY, f64::max)),
        )
    }
    pub(crate) fn div_positive(self, other: Self) -> Self {
        if other.lo <= 0.0 || !other.finite() {
            return Self::new(f64::NEG_INFINITY, f64::INFINITY);
        }
        self.mul(Self::new(down(1.0 / other.hi), up(1.0 / other.lo)))
    }
    pub(crate) fn abs_max(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
    pub(crate) fn midpoint(self) -> f64 {
        self.lo * 0.5 + self.hi * 0.5
    }
    pub(crate) fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
}

/// Checked exp(x) for x <= 0. Range reduction makes t=-x/2^k <= 1/8.
/// Positive Taylor terms enclose exp(t); the tail after term 18 is at most
/// `2*t^19/19!`, since exp(t) < 1/(1-t) < 2. Reciprocal and k squarings
/// recover exp(x). This uses no assumptions about platform libm accuracy.
pub(crate) fn exp_nonpositive(x: f64) -> Interval {
    if x == 0.0 {
        return Interval::point(1.0);
    }
    if !x.is_finite() || x > 0.0 {
        return Interval::new(0.0, f64::INFINITY);
    }
    // Monotonicity gives a checked normal upper bound without executing any
    // subnormal exp arithmetic. The recursive endpoint takes the finite branch.
    if x < -708.0 {
        return Interval::new(0.0, exp_nonpositive(-708.0).hi);
    }
    // Exact halvings of t = -x in (0, 708] down to (0, 0.125]: at most 13,
    // so the explicit 64-step bound never cuts the reduction short.
    let mut t = -x;
    let mut squares = 0;
    for _ in 0..64 {
        if t <= 0.125 {
            break;
        }
        t *= 0.5;
        squares += 1;
    }
    let t = Interval::point(t);
    let mut term = Interval::point(1.0);
    let mut sum = term;
    for j in 1..=18 {
        term = term.mul(t).div_positive(Interval::point(j as f64));
        sum = sum.add(term);
    }
    let tail = term
        .mul(t)
        .div_positive(Interval::point(19.0))
        .mul(Interval::point(2.0));
    sum.hi = up(sum.hi + tail.hi);
    let mut result = Interval::point(1.0).div_positive(sum);
    for _ in 0..squares {
        result = result.mul(result);
    }
    Interval::new(result.lo.max(0.0), result.hi.min(1.0))
}

/// Upper bound on the total variation of a softmax under a score tilt whose
/// oscillation is at most `score_osc`. The likelihood-ratio range is at most
/// exp(score_osc), giving TV <= tanh(score_osc/4). Only a checked exp lower
/// endpoint authorizes `(1-e)/(1+e)`; libm supplies no authority here.
///
/// Multiplication by one half must be outward even for subnormals. Rounding
/// `-score_osc/2` to negative zero would incorrectly claim zero variation for
/// a positive real tilt. Unknown/nonfinite ranges conservatively return one.
pub(crate) fn softmax_tilt_tv_upper(score_osc: f64) -> f64 {
    if score_osc == 0.0 {
        return 0.0;
    }
    if !score_osc.is_finite() || score_osc < 0.0 {
        return 1.0;
    }
    let half = Interval::point(score_osc).mul(Interval::point(0.5));
    if !half.finite() {
        return 1.0;
    }
    let exponential = exp_nonpositive(-half.hi);
    if !exponential.finite() {
        return 1.0;
    }
    // h(e)=(1-e)/(1+e) is decreasing for e>=0, so a checked lower
    // exponential endpoint and outward ratio bound the desired quantity.
    // `exponential` is checked finite above, so clamp cannot see a NaN.
    let e = Interval::point(exponential.lo.clamp(0.0, 1.0));
    let ratio = Interval::point(1.0)
        .sub(e)
        .div_positive(Interval::point(1.0).add(e));
    if !ratio.finite() {
        return 1.0;
    }
    // `ratio` is checked finite above.
    ratio.hi.clamp(0.0, 1.0)
}

/// Shared exp endpoint cache. Construction O(T), threshold search O(T log T),
/// checked scans O(T), memory O(T). A common score shift changes no softmax.
pub(crate) struct WeightedScoreBox {
    lower_exp: Vec<Interval>,
    upper_exp: Vec<Interval>,
    denominator_lower: f64,
}

impl WeightedScoreBox {
    pub(crate) fn new(lower: &[f64], upper: &[f64]) -> Option<Self> {
        if lower.is_empty()
            || lower.len() != upper.len()
            || lower
                .iter()
                .zip(upper)
                .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
        {
            return None;
        }
        let shift = upper.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut lower_exp = Vec::with_capacity(lower.len());
        let mut upper_exp = Vec::with_capacity(lower.len());
        let mut denominator = Interval::point(0.0);
        for (index, (&l, &u)) in lower.iter().zip(upper).enumerate() {
            if index % 64 == 0
                && ny_core::instance_budget::deadline()
                    .is_some_and(|d| std::time::Instant::now() >= d)
            {
                return None;
            }
            let l = Interval::point(l).sub(Interval::point(shift));
            let u = Interval::point(u).sub(Interval::point(shift));
            let le = Interval::new(exp_nonpositive(l.lo).lo, exp_nonpositive(l.hi.min(0.0)).hi);
            let ue = Interval::new(exp_nonpositive(u.lo).lo, exp_nonpositive(u.hi.min(0.0)).hi);
            denominator = denominator.add(le);
            lower_exp.push(le);
            upper_exp.push(ue);
        }
        Some(Self {
            lower_exp,
            upper_exp,
            denominator_lower: denominator.lo.max(0.0),
        })
    }

    /// Returns a checked lower support for fixed coefficients. If construction
    /// or checking is ill conditioned, the convex hull remains a valid fallback.
    pub(crate) fn lower(&self, c: &[f64]) -> f64 {
        if c.len() != self.lower_exp.len() || c.iter().any(|x| !x.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let hull = c.iter().copied().fold(f64::INFINITY, f64::min);
        if c.iter().all(|&v| v == hull) {
            return hull;
        }
        let mut order: Vec<usize> = (0..c.len()).collect();
        order.sort_unstable_by(|&i, &j| c[i].total_cmp(&c[j]).then(i.cmp(&j)));
        // This floating computation proposes a candidate only. Cancellation,
        // exponent underflow, and endpoint errors cannot authorize a result.
        let mut mass: f64 = self.lower_exp.iter().map(|e| e.midpoint()).sum();
        let mut moment: f64 = c
            .iter()
            .zip(&self.lower_exp)
            .map(|(&v, e)| v * e.midpoint())
            .sum();
        let mut candidate = hull;
        for cut in 0..=order.len() {
            let tau = moment / mass;
            let low = if cut == 0 {
                f64::NEG_INFINITY
            } else {
                c[order[cut - 1]]
            };
            let high = if cut == order.len() {
                f64::INFINITY
            } else {
                c[order[cut]]
            };
            if tau.is_finite() && low <= tau && tau <= high {
                candidate = tau;
                break;
            }
            if cut < order.len() {
                let i = order[cut];
                let delta = self.upper_exp[i].midpoint() - self.lower_exp[i].midpoint();
                mass += delta;
                moment += delta * c[i];
            }
        }
        // Each attempt repeats the independent interval certificate. The factor
        // two gives the proposal room for fresh scan rounding: landing exactly
        // on the previous scan's enclosure boundary can repeatedly fail replay.
        // This correction proposes a threshold; only check_lower authorizes it.
        for _ in 0..3 {
            if candidate <= hull {
                return hull;
            }
            let check = self.check_lower(c, candidate);
            if check >= 0.0 {
                return candidate;
            }
            if self.denominator_lower <= 0.0 || !check.is_finite() {
                return hull;
            }
            let correction = Interval::point(check)
                .div_positive(Interval::point(self.denominator_lower))
                .mul(Interval::point(2.0));
            candidate = Interval::point(candidate).add(correction).lo;
        }
        hull
    }

    pub(crate) fn upper(&self, c: &[f64]) -> f64 {
        let negative: Vec<f64> = c.iter().map(|&v| -v).collect();
        -self.lower(&negative)
    }

    fn check_lower(&self, c: &[f64], tau: f64) -> f64 {
        let mut sum = Interval::point(0.0);
        for (i, &ci) in c.iter().enumerate() {
            let coefficient = if ci == tau {
                Interval::point(0.0)
            } else {
                Interval::point(ci).sub(Interval::point(tau))
            };
            let endpoint = if ci >= tau {
                self.lower_exp[i]
            } else {
                self.upper_exp[i]
            };
            sum = sum.add(coefficient.mul(endpoint));
        }
        sum.lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_tilt_encloses_tiny_and_wide_reference_values() {
        for range in [
            f64::from_bits(1),
            f64::from_bits(2),
            1e-20,
            1e-12,
            1e-6,
            0.125,
            1.5,
            4.0,
            100.0,
            1500.0,
            f64::MAX,
        ] {
            let bound = softmax_tilt_tv_upper(range);
            let reference = (range * 0.25).tanh();
            assert!(
                bound >= reference && bound <= 1.0,
                "range={range}: bound={bound},reference={reference}"
            );
            assert!(
                bound > 0.0,
                "a positive real score tilt cannot have zero TV authority"
            );
        }
        assert_eq!(softmax_tilt_tv_upper(0.0), 0.0);
        for range in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -1.0] {
            assert_eq!(softmax_tilt_tv_upper(range), 1.0);
        }
    }

    #[test]
    fn checked_tilt_bounds_independent_rational_two_point_witness_strictly_below_linear() {
        use num_rational::BigRational;
        use num_traits::{One, Zero};
        // p=[2/3,1/3], tilt weights=[1,4], p'=[1/3,2/3], TV=1/3.
        // Exact rational Taylor arithmetic establishes exp(3/2)>4, hence
        // osc(log tilt)=log(4)<3/2 without reusing the exp authority.
        let x = BigRational::new(3.into(), 2.into());
        let mut term = BigRational::one();
        let mut sum = BigRational::zero();
        for k in 0..=16 {
            if k > 0 {
                term = term.clone() * x.clone() / BigRational::from_integer(k.into());
            }
            sum += term.clone();
        }
        assert!(sum > BigRational::from_integer(4.into()));
        let upper = softmax_tilt_tv_upper(1.5);
        assert!(BigRational::from_float(upper).unwrap() >= BigRational::new(1.into(), 3.into()));
        // 0.36 is tighter than the analytic 1.5 / 4 = 0.375, so it alone carries
        // both claims (clippy::redundant_comparisons).
        assert!(
            upper < 0.36,
            "tilt TV upper {upper} must stay below 0.36 < 1.5 / 4"
        );
    }

    #[test]
    fn checked_exp_encloses_reference_and_underflow() {
        for x in [
            0.0, -1e-10, -0.125, -0.5, -1.0, -8.0, -80.0, -708.0, -709.0, -1000.0,
        ] {
            let e = exp_nonpositive(x);
            assert!(e.lo <= x.exp() && x.exp() <= e.hi, "x={x} {e:?}");
            assert!(e.finite() && e.lo >= 0.0 && e.hi <= 1.0);
        }
    }

    #[test]
    fn weighted_support_recovers_fixed_uniform_rational_mean() {
        let score = WeightedScoreBox::new(&[0.0; 4], &[0.0; 4]).unwrap();
        let c = [-3.0, 1.0, 2.0, 8.0];
        let lo = score.lower(&c);
        let hi = score.upper(&c);
        assert!(lo <= 2.0 && hi >= 2.0);
        assert!(
            lo > 1.99999999999 && hi < 2.00000000001,
            "checked bounds: {lo}, {hi}"
        );
    }

    #[test]
    fn uniform_support_keeps_verified_tightness_with_signed_cancellation() {
        for (coefficients, mean) in [
            ([-9.0, 2.0, 7.0, 16.0], 4.0),
            ([-16.0, -7.0, -2.0, 9.0], -4.0),
            ([-1024.0, -1.0, 1.0, 1024.0], 0.0),
        ] {
            let score = WeightedScoreBox::new(&[17.0; 4], &[17.0; 4]).unwrap();
            let lo = score.lower(&coefficients);
            let hi = score.upper(&coefficients);
            assert!(lo <= mean && mean <= hi);
            assert!(mean - lo < 1e-10 && hi - mean < 1e-10, "{lo} {hi}");
        }
    }

    #[test]
    fn weighted_support_checks_all_small_box_vertices_and_shift() {
        let l = [-1.0, -0.3, -2.0];
        let u = [0.7, 0.1, 0.4];
        let c = [-2.0, 3.0, 0.5];
        let b = WeightedScoreBox::new(&l, &u).unwrap();
        let lo = b.lower(&c);
        let hi = b.upper(&c);
        assert!(lo > -2.0 && hi < 3.0);
        for vertex in 0..8 {
            let e: Vec<f64> = (0..3)
                .map(|i| {
                    if vertex & (1 << i) == 0 {
                        l[i].exp()
                    } else {
                        u[i].exp()
                    }
                })
                .collect();
            let value = c.iter().zip(&e).map(|(c, e)| c * e).sum::<f64>() / e.iter().sum::<f64>();
            assert!(lo <= value && value <= hi);
        }
        let shifted = WeightedScoreBox::new(&l.map(|v| v + 128.0), &u.map(|v| v + 128.0)).unwrap();
        assert!((lo - shifted.lower(&c)).abs() < 1e-11);
    }

    #[test]
    fn malformed_and_underflow_keep_convex_hull() {
        assert!(WeightedScoreBox::new(&[f64::NAN], &[0.0]).is_none());
        assert!(WeightedScoreBox::new(&[1.0], &[0.0]).is_none());
        let b = WeightedScoreBox::new(&[-1e6, -1e6], &[1e6, 1e6]).unwrap();
        assert_eq!(b.lower(&[-4.0, 7.0]), -4.0);
        assert_eq!(b.upper(&[-4.0, 7.0]), 7.0);
    }
}
