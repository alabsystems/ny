// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LSE relaxations built from checked transcendental enclosures. Stored f32
//! slopes are recentered and their derivative deviations paid over this box.

use super::weighted_support::{exp_nonpositive, Interval as I};
use ndarray::{Array1, Array2};
use ny_tensor::{next_down_f32, next_up_f32};

fn log_point(x: f64) -> I {
    if !x.is_finite() || x <= 0.0 {
        return I::new(f64::NEG_INFINITY, f64::INFINITY);
    }
    if x == 1.0 {
        return I::point(0.0);
    }
    // Exact binary scaling of a finite positive `x` into `m` in [1, 2): at
    // most 1023 halvings (x < 2^1024) or 1074 doublings (x >= 2^-1074), so the
    // explicit 1100-step bounds never cut a scaling short.
    let (mut m, mut exponent) = (x, 0_i32);
    for _ in 0..1100 {
        if m < 2.0 {
            break;
        }
        m *= 0.5;
        exponent += 1;
    }
    for _ in 0..1100 {
        if m >= 1.0 {
            break;
        }
        m *= 2.0;
        exponent -= 1;
    }
    fn series(m: f64) -> I {
        if m == 1.0 {
            return I::point(0.0);
        }
        let t = I::point(m)
            .sub(I::point(1.0))
            .div_positive(I::point(m).add(I::point(1.0)));
        let square = t.mul(t);
        let mut power = t;
        let mut sum = I::point(0.0);
        for k in 0..34 {
            sum = sum.add(power.div_positive(I::point((2 * k + 1) as f64)));
            power = power.mul(square);
        }
        // Positive geometric majorant for all omitted odd terms.
        let tail = power
            .div_positive(I::point(69.0))
            .div_positive(I::point(1.0).sub(square));
        I::new(sum.lo, sum.add(tail).hi).mul(I::point(2.0))
    }
    static LOG_TWO: std::sync::OnceLock<I> = std::sync::OnceLock::new();
    let ln2 = *LOG_TWO.get_or_init(|| series(2.0));
    series(m).add(ln2.mul(I::point(exponent as f64)))
}

fn log_interval(x: I) -> I {
    I::new(log_point(x.lo).lo, log_point(x.hi).hi)
}

fn shifted_exp(x: I) -> I {
    I::new(exp_nonpositive(x.lo).lo, exp_nonpositive(x.hi.min(0.0)).hi)
}

fn lse(x: &[f64]) -> Option<I> {
    let shift = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = I::point(0.0);
    for &v in x {
        sum = sum.add(shifted_exp(I::point(v).sub(I::point(shift))));
    }
    let result = log_interval(sum).add(I::point(shift));
    result.finite().then_some(result)
}

/// An arbitrary stored slope with an intercept above both checked endpoints
/// is above the true exponential chord throughout the interval by convexity.
fn exp_chord(l: f64, u: f64) -> Option<(f64, I)> {
    let el = exp_nonpositive(l);
    let eu = exp_nonpositive(u);
    let slope = if l == u {
        0.0
    } else {
        eu.sub(el)
            .div_positive(I::point(u).sub(I::point(l)))
            .midpoint()
            .max(0.0) as f32 as f64
    };
    if !slope.is_finite() {
        return None;
    }
    let a = el.sub(I::point(slope).mul(I::point(l)));
    let b = eu.sub(I::point(slope).mul(I::point(u)));
    Some((slope, I::point(a.hi.max(b.hi))))
}

#[allow(clippy::type_complexity)]
pub(super) fn affine_bounds(
    lower: &Array1<f32>,
    upper: &Array1<f32>,
) -> Option<(Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>)> {
    let n = lower.len();
    if n == 0
        || upper.len() != n
        || lower
            .iter()
            .zip(upper)
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return None;
    }
    let l: Vec<f64> = lower.iter().map(|&x| x as f64).collect();
    let u: Vec<f64> = upper.iter().map(|&x| x as f64).collect();
    let center: Vec<f64> = l.iter().zip(&u).map(|(&l, &u)| l + (u - l) * 0.5).collect();
    let radius: Vec<f64> = l
        .iter()
        .zip(&u)
        .zip(&center)
        .map(|((&l, &u), &c)| {
            I::point(l)
                .sub(I::point(c))
                .abs_max()
                .max(I::point(u).sub(I::point(c)).abs_max())
        })
        .collect();
    let shift = u.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut chord_slope = Vec::with_capacity(n);
    let mut denominator = I::point(0.0);
    for j in 0..n {
        let shifted_l = I::point(l[j]).sub(I::point(shift));
        let shifted_u = I::point(u[j]).sub(I::point(shift));
        let (slope, _) = exp_chord(shifted_l.lo, shifted_u.hi.min(0.0))?;
        // Recalculate at original endpoints so cancellation in l-shift cannot
        // shrink an endpoint value or move the chord's asserted domain.
        let bl = shifted_exp(shifted_l).sub(I::point(slope).mul(I::point(l[j])));
        let bu = shifted_exp(shifted_u).sub(I::point(slope).mul(I::point(u[j])));
        let b = bl.hi.max(bu.hi);
        chord_slope.push(slope);
        denominator = denominator.add(I::point(slope).mul(I::point(center[j])).add(I::point(b)));
    }
    if !denominator.finite() || denominator.lo <= 0.0 {
        return None;
    }
    let exp_center: Vec<I> = center
        .iter()
        .map(|&c| shifted_exp(I::point(c).sub(I::point(shift))))
        .collect();
    let sum_center = exp_center.iter().fold(I::point(0.0), |s, &e| s.add(e));
    let probabilities: Vec<I> = exp_center
        .iter()
        .map(|e| e.div_positive(sum_center))
        .collect();
    let lse_lower = lse(&l)?;
    let lse_upper = lse(&u)?;
    let lse_center = lse(&center)?;
    let mut lower_a = Array2::<f32>::zeros((n, n));
    let mut upper_a = Array2::<f32>::zeros((n, n));
    let mut lower_b = Array1::<f32>::zeros(n);
    let mut upper_b = Array1::<f32>::zeros(n);
    for i in 0..n {
        if i % 16 == 0
            && ny_core::instance_budget::deadline().is_some_and(|d| std::time::Instant::now() >= d)
        {
            return None;
        }
        let lower_value = exp_center[i].div_positive(denominator);
        let mut lower_dot = I::point(0.0);
        let mut lower_error = I::point(0.0);
        let rlo = I::point(l[i]).sub(lse_upper).lo;
        let rhi = I::point(u[i]).sub(lse_lower).hi.min(0.0);
        let (upper_slope, upper_constant) = exp_chord(rlo, rhi)?;
        let upper_value = I::point(upper_slope)
            .mul(I::point(center[i]).sub(lse_center))
            .add(upper_constant);
        let mut upper_dot = I::point(0.0);
        let mut upper_error = I::point(0.0);
        for j in 0..n {
            let identity = I::point(if i == j { 1.0 } else { 0.0 });
            let lower_gradient =
                lower_value.mul(identity.sub(I::point(chord_slope[j]).div_positive(denominator)));
            let upper_gradient = I::point(upper_slope).mul(identity.sub(probabilities[j]));
            let la = lower_gradient.midpoint() as f32;
            let ua = upper_gradient.midpoint() as f32;
            if !la.is_finite() || !ua.is_finite() {
                return None;
            }
            lower_a[[i, j]] = la;
            upper_a[[i, j]] = ua;
            lower_dot = lower_dot.add(I::point(la as f64).mul(I::point(center[j])));
            upper_dot = upper_dot.add(I::point(ua as f64).mul(I::point(center[j])));
            lower_error = lower_error.add(
                I::point(lower_gradient.sub(I::point(la as f64)).abs_max())
                    .mul(I::point(radius[j])),
            );
            upper_error = upper_error.add(
                I::point(upper_gradient.sub(I::point(ua as f64)).abs_max())
                    .mul(I::point(radius[j])),
            );
        }
        let lb = lower_value.sub(lower_dot).sub(lower_error);
        let ub = upper_value.sub(upper_dot).add(upper_error);
        if !lb.finite() || !ub.finite() {
            return None;
        }
        lower_b[i] = next_down_f32(lb.lo as f32);
        upper_b[i] = next_up_f32(ub.hi as f32);
        if !lower_b[i].is_finite() || !upper_b[i].is_finite() {
            return None;
        }
    }
    Some((lower_a, lower_b, upper_a, upper_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_log_encloses_independent_reference() {
        for x in [f64::MIN_POSITIVE, 0.125, 0.5, 1.0, 2.0, 3.0, 128.0, 1e30] {
            let bound = log_point(x);
            assert!(bound.lo <= x.ln() && x.ln() <= bound.hi, "{x}: {bound:?}");
        }
    }
}
