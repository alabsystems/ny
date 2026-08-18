// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Small directed-binary64 transcendental kernel for the one-axis checker.
//!
//! These routines deliberately do not use the platform `exp` or `ln` as
//! certificate authority.  They enclose elementary arithmetic by one adjacent
//! binary64 step and use positive-term series with an explicit geometric tail.
//! Every non-finite intermediate, range refusal, or expired deadline returns
//! `None`.

use std::time::Instant;

use num_rational::BigRational;
use num_traits::ToPrimitive;
use ny_core::dd::{next_down_f64, next_up_f64};

/// Closed binary64 interval whose endpoints enclose real values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DirectedInterval {
    pub(super) lower: f64,
    pub(super) upper: f64,
}

impl DirectedInterval {
    pub(super) fn point(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            lower: value,
            upper: value,
        })
    }

    pub(super) fn new(lower: f64, upper: f64) -> Option<Self> {
        (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Self { lower, upper })
    }

    pub(super) fn add(self, rhs: Self) -> Option<Self> {
        Self::new(
            next_down_f64(self.lower + rhs.lower),
            next_up_f64(self.upper + rhs.upper),
        )
    }

    pub(super) fn sub(self, rhs: Self) -> Option<Self> {
        Self::new(
            next_down_f64(self.lower - rhs.upper),
            next_up_f64(self.upper - rhs.lower),
        )
    }

    pub(super) fn mul(self, rhs: Self) -> Option<Self> {
        let products = [
            self.lower * rhs.lower,
            self.lower * rhs.upper,
            self.upper * rhs.lower,
            self.upper * rhs.upper,
        ];
        if products.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let lower = products.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::new(next_down_f64(lower), next_up_f64(upper))
    }

    pub(super) fn div(self, rhs: Self) -> Option<Self> {
        if rhs.lower <= 0.0 && rhs.upper >= 0.0 {
            return None;
        }
        let reciprocal = Self::new(next_down_f64(1.0 / rhs.upper), next_up_f64(1.0 / rhs.lower))?;
        self.mul(reciprocal)
    }

    fn scale_nonnegative(self, factor: f64) -> Option<Self> {
        if !factor.is_finite() || factor < 0.0 || self.lower < 0.0 {
            return None;
        }
        Self::new(
            next_down_f64(self.lower * factor).max(0.0),
            next_up_f64(self.upper * factor),
        )
    }
}

/// Enclose an exact rational by finite binary64 endpoints.
pub(super) fn rational_enclosure(value: &BigRational) -> Option<DirectedInterval> {
    let nearest = value.to_f64()?;
    if !nearest.is_finite() {
        return None;
    }
    let nearest_exact = BigRational::from_float(nearest)?;
    let mut lower = nearest;
    let mut upper = nearest;
    if nearest_exact > *value {
        lower = next_down_f64(lower);
    }
    if nearest_exact < *value {
        upper = next_up_f64(upper);
    }
    let lower_exact = BigRational::from_float(lower)?;
    let upper_exact = BigRational::from_float(upper)?;
    if lower_exact > *value || upper_exact < *value {
        return None;
    }
    DirectedInterval::new(lower, upper)
}

fn expired(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

/// Positive-term enclosure of `exp(x)` for finite `x`.
fn exp_point(value: f64, deadline: Instant) -> Option<DirectedInterval> {
    if !value.is_finite() || value.abs() > 700.0 || expired(deadline) {
        return None;
    }
    if value < 0.0 {
        let positive = exp_point(-value, deadline)?;
        return DirectedInterval::point(1.0)?.div(positive);
    }
    if value == 0.0 {
        return DirectedInterval::point(1.0);
    }

    // Multiplication by a power of two is exact in this normal range.
    let mut reduced = value;
    let mut squarings = 0usize;
    for _ in 0..16 {
        if reduced <= 0.0625 {
            break;
        }
        if expired(deadline) {
            return None;
        }
        reduced *= 0.5;
        squarings += 1;
    }
    if reduced > 0.0625 {
        return None;
    }

    let x = DirectedInterval::point(reduced)?;
    let mut term = DirectedInterval::point(1.0)?;
    let mut sum = term;
    const TERMS: usize = 48;
    for degree in 1..=TERMS {
        if degree % 16 == 0 && expired(deadline) {
            return None;
        }
        term = term.mul(x)?.div(DirectedInterval::point(degree as f64)?)?;
        sum = sum.add(term)?;
    }

    // For k >= TERMS+1, successive terms have ratio <= x/(TERMS+2).
    let next = term
        .mul(x)?
        .div(DirectedInterval::point((TERMS + 1) as f64)?)?;
    let ratio = x.div(DirectedInterval::point((TERMS + 2) as f64)?)?;
    if ratio.lower < 0.0 || ratio.upper >= 1.0 {
        return None;
    }
    let remaining = DirectedInterval::point(1.0)?.sub(ratio)?;
    if remaining.lower <= 0.0 {
        return None;
    }
    let tail_upper = next.div(remaining)?.upper;
    sum = sum.add(DirectedInterval::new(0.0, tail_upper)?)?;

    for _ in 0..squarings {
        if expired(deadline) {
            return None;
        }
        sum = sum.mul(sum)?;
    }
    Some(sum)
}

#[cfg(test)]
pub(super) fn exp_enclosure(
    input: DirectedInterval,
    deadline: Instant,
) -> Option<DirectedInterval> {
    DirectedInterval::new(
        exp_point(input.lower, deadline)?.lower,
        exp_point(input.upper, deadline)?.upper,
    )
}

fn atanh_positive_series(u: DirectedInterval, deadline: Instant) -> Option<DirectedInterval> {
    if u.lower < 0.0 || u.upper >= 0.5 || expired(deadline) {
        return None;
    }
    let u2 = u.mul(u)?;
    let mut power = u;
    let mut sum = u;
    const TERMS: usize = 96;
    for index in 1..=TERMS {
        if index % 16 == 0 && expired(deadline) {
            return None;
        }
        power = power.mul(u2)?;
        let denominator = DirectedInterval::point((2 * index + 1) as f64)?;
        sum = sum.add(power.div(denominator)?)?;
    }
    let next_power = power.mul(u2)?;
    let next = next_power.div(DirectedInterval::point((2 * TERMS + 3) as f64)?)?;
    let remaining = DirectedInterval::point(1.0)?.sub(u2)?;
    if remaining.lower <= 0.0 {
        return None;
    }
    let tail_upper = next.div(remaining)?.upper;
    sum = sum.add(DirectedInterval::new(0.0, tail_upper)?)?;
    sum.scale_nonnegative(2.0)
}

fn ln_two(deadline: Instant) -> Option<DirectedInterval> {
    let u = DirectedInterval::point(1.0)?.div(DirectedInterval::point(3.0)?)?;
    atanh_positive_series(u, deadline)
}

fn log_point(value: f64, deadline: Instant) -> Option<DirectedInterval> {
    if !value.is_finite()
        || value <= 0.0
        || !(f64::from_bits((1023_u64 - 512) << 52)..=f64::from_bits((1023_u64 + 512) << 52))
            .contains(&value)
        || expired(deadline)
    {
        return None;
    }

    let mut reduced = value;
    let mut exponent = 0i32;
    for _ in 0..512 {
        if reduced >= 1.0 {
            break;
        }
        if expired(deadline) {
            return None;
        }
        reduced *= 2.0;
        exponent -= 1;
    }
    for _ in 0..512 {
        if reduced < 2.0 {
            break;
        }
        if expired(deadline) {
            return None;
        }
        reduced *= 0.5;
        exponent += 1;
    }
    if !(1.0..2.0).contains(&reduced) {
        return None;
    }

    let reduced_log = if reduced == 1.0 {
        DirectedInterval::point(0.0)?
    } else {
        let y = DirectedInterval::point(reduced)?;
        let u = y
            .sub(DirectedInterval::point(1.0)?)?
            .div(y.add(DirectedInterval::point(1.0)?)?)?;
        atanh_positive_series(u, deadline)?
    };
    let exponent_interval = DirectedInterval::point(f64::from(exponent))?;
    reduced_log.add(ln_two(deadline)?.mul(exponent_interval)?)
}

pub(super) fn log_enclosure(
    input: DirectedInterval,
    deadline: Instant,
) -> Option<DirectedInterval> {
    DirectedInterval::new(
        log_point(input.lower, deadline)?.lower,
        log_point(input.upper, deadline)?.upper,
    )
}

pub(super) fn sigmoid_enclosure(
    input: DirectedInterval,
    deadline: Instant,
) -> Option<DirectedInterval> {
    fn point(value: f64, deadline: Instant) -> Option<DirectedInterval> {
        if value >= 0.0 {
            DirectedInterval::point(1.0)?
                .div(DirectedInterval::point(1.0)?.add(exp_point(-value, deadline)?)?)
        } else {
            let exponential = exp_point(value, deadline)?;
            exponential.div(DirectedInterval::point(1.0)?.add(exponential)?)
        }
    }
    DirectedInterval::new(
        point(input.lower, deadline)?.lower,
        point(input.upper, deadline)?.upper,
    )
}

pub(super) fn logit_enclosure(
    probability: DirectedInterval,
    deadline: Instant,
) -> Option<DirectedInterval> {
    if probability.lower <= 0.0 || probability.upper >= 1.0 {
        return None;
    }
    let one = DirectedInterval::point(1.0)?;
    log_enclosure(probability.div(one.sub(probability)?)?, deadline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::time::Duration;

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    #[test]
    fn exact_rational_conversion_is_outward() {
        for (numerator, denominator) in [(1, 10), (-1, 10), (1, 3), (9_007_199_254_740_993_i64, 1)]
        {
            let exact = BigRational::new(numerator.into(), denominator.into());
            let enclosure = rational_enclosure(&exact).expect("finite enclosure");
            assert!(BigRational::from_float(enclosure.lower).unwrap() <= exact);
            assert!(BigRational::from_float(enclosure.upper).unwrap() >= exact);
        }
    }

    #[test]
    fn elementary_series_enclose_oracles() {
        for value in [-20.0_f64, -3.0, -0.125, 0.0, 0.125, 3.0, 20.0] {
            let exp = exp_enclosure(DirectedInterval::point(value).unwrap(), deadline()).unwrap();
            assert!(exp.lower <= value.exp() && value.exp() <= exp.upper);
            let sigmoid =
                sigmoid_enclosure(DirectedInterval::point(value).unwrap(), deadline()).unwrap();
            let oracle = 1.0 / (1.0 + (-value).exp());
            assert!(sigmoid.lower <= oracle && oracle <= sigmoid.upper);
        }
        for value in [0.125_f64, 0.5, 1.0, 2.0, 10.0] {
            let log = log_enclosure(DirectedInterval::point(value).unwrap(), deadline()).unwrap();
            assert!(log.lower <= value.ln() && value.ln() <= log.upper);
        }
        for probability in [0.01_f64, 0.125, 0.5, 0.875, 0.99] {
            let interval = DirectedInterval::point(probability).unwrap();
            let logit = logit_enclosure(interval, deadline()).unwrap();
            let oracle = (probability / (1.0 - probability)).ln();
            assert!(logit.lower <= oracle && oracle <= logit.upper);
        }
    }

    #[test]
    fn expired_and_extreme_requests_fail_closed() {
        assert!(exp_enclosure(DirectedInterval::point(1.0).unwrap(), Instant::now()).is_none());
        assert!(log_enclosure(DirectedInterval::point(0.0).unwrap(), deadline()).is_none());
        assert!(logit_enclosure(DirectedInterval::point(1.0).unwrap(), deadline()).is_none());
    }

    proptest! {
        #[test]
        fn directed_basic_arithmetic_contains_exact_binary64_results(
            left in -1.0e100_f64..1.0e100,
            right in -1.0e100_f64..1.0e100,
        ) {
            let left_interval = DirectedInterval::point(left).unwrap();
            let right_interval = DirectedInterval::point(right).unwrap();
            let left_exact = BigRational::from_float(left).unwrap();
            let right_exact = BigRational::from_float(right).unwrap();
            for (interval, exact) in [
                (left_interval.add(right_interval).unwrap(), &left_exact + &right_exact),
                (left_interval.sub(right_interval).unwrap(), &left_exact - &right_exact),
                (left_interval.mul(right_interval).unwrap(), &left_exact * &right_exact),
            ] {
                prop_assert!(BigRational::from_float(interval.lower).unwrap() <= exact);
                prop_assert!(BigRational::from_float(interval.upper).unwrap() >= exact);
            }
        }

        #[test]
        fn directed_division_contains_exact_binary64_quotient(
            numerator in -1.0e100_f64..1.0e100,
            magnitude in 1.0e-100_f64..1.0e100,
            negative in any::<bool>(),
        ) {
            let denominator = if negative { -magnitude } else { magnitude };
            let interval = DirectedInterval::point(numerator)
                .unwrap()
                .div(DirectedInterval::point(denominator).unwrap())
                .unwrap();
            let exact = BigRational::from_float(numerator).unwrap()
                / BigRational::from_float(denominator).unwrap();
            prop_assert!(BigRational::from_float(interval.lower).unwrap() <= exact);
            prop_assert!(BigRational::from_float(interval.upper).unwrap() >= exact);
        }

        #[test]
        fn directed_transcendentals_contain_finite_oracles(value in -50.0_f64..50.0) {
            let point = DirectedInterval::point(value).unwrap();
            let exp = exp_enclosure(point, deadline()).unwrap();
            prop_assert!(exp.lower <= value.exp() && value.exp() <= exp.upper);
            let sigmoid = sigmoid_enclosure(point, deadline()).unwrap();
            let sigmoid_oracle = 1.0 / (1.0 + (-value).exp());
            prop_assert!(
                sigmoid.lower <= sigmoid_oracle && sigmoid_oracle <= sigmoid.upper
            );
        }

        #[test]
        fn directed_log_contains_finite_oracle(value in 1.0e-6_f64..1.0e6) {
            let log = log_enclosure(DirectedInterval::point(value).unwrap(), deadline()).unwrap();
            prop_assert!(log.lower <= value.ln() && value.ln() <= log.upper);
        }
    }
}
