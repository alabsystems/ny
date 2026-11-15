// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Checked coordinate extrema in O(K), without a total-minus-own subtraction.
//! For each endpoint family the prefix/suffix sums enclose the excluded mass.
//! Removing a unique maximum uses one separately shifted O(K) sum, at most
//! twice per row. All exponent arguments are nonpositive certified differences.
//! These caches belong to one immutable row/mask; they authorize no KV reuse.

use super::super::weighted_support::{exp_nonpositive, Interval as I};
use ny_core::{f64_to_f32_down, f64_to_f32_up, NyError, Result};

fn exp_difference(x: f64, shift: f64) -> I {
    debug_assert!(x <= shift && x.is_finite() && shift.is_finite());
    let d = I::point(x).sub(I::point(shift));
    I::new(exp_nonpositive(d.lo).lo, exp_nonpositive(d.hi.min(0.0)).hi)
}

fn nonnegative(i: I) -> I {
    I::new(i.lo.max(0.0), i.hi)
}

struct EndpointMass {
    shift: f64,
    prefix: Vec<I>,
    suffix: Vec<I>,
    // A common shift by the excluded maximum could underflow every surviving
    // key. Keep its independently checked local scale, never an epsilon.
    excluded_max: Option<(usize, f64, I)>,
}

impl EndpointMass {
    fn new(values: &[f64], poll: &mut impl FnMut() -> Result<()>) -> Result<Self> {
        let mut shift = f64::NEG_INFINITY;
        let (mut unique, mut maximum) = (false, 0usize);
        for (i, &v) in values.iter().enumerate() {
            if i % 64 == 0 {
                poll()?;
            }
            if v > shift {
                shift = v;
                maximum = i;
                unique = true;
            } else if v == shift {
                unique = false;
            }
        }
        poll()?;
        let mut exponentials = Vec::with_capacity(values.len());
        let mut prefix = Vec::with_capacity(values.len() + 1);
        prefix.push(I::point(0.0));
        for (i, &value) in values.iter().enumerate() {
            if i % 64 == 0 {
                poll()?;
            }
            let e = exp_difference(value, shift);
            prefix.push(nonnegative(prefix[i].add(e)));
            exponentials.push(e);
        }
        let mut suffix = vec![I::point(0.0); values.len() + 1];
        for i in (0..values.len()).rev() {
            if i % 64 == 0 {
                poll()?;
            }
            suffix[i] = nonnegative(exponentials[i].add(suffix[i + 1]));
        }
        let excluded_max = if unique && values.len() > 1 {
            let mut other_shift = f64::NEG_INFINITY;
            for (i, &value) in values.iter().enumerate() {
                if i % 64 == 0 {
                    poll()?;
                }
                if i != maximum {
                    other_shift = other_shift.max(value);
                }
            }
            let mut mass = I::point(0.0);
            for (i, &value) in values.iter().enumerate() {
                if i % 64 == 0 {
                    poll()?;
                }
                if i != maximum {
                    mass = nonnegative(mass.add(exp_difference(value, other_shift)));
                }
            }
            Some((maximum, other_shift, mass))
        } else {
            None
        };
        poll()?;
        Ok(Self {
            shift,
            prefix,
            suffix,
            excluded_max,
        })
    }

    fn excluding(&self, index: usize) -> (f64, I) {
        if let Some((maximum, shift, mass)) = self.excluded_max {
            if index == maximum {
                return (shift, mass);
            }
        }
        (
            self.shift,
            nonnegative(self.prefix[index].add(self.suffix[index + 1])),
        )
    }

    fn ratio(&self, index: usize, own: f64, lower: bool) -> f64 {
        let (mass_shift, mass) = self.excluding(index);
        let shift = own.max(mass_shift);
        let numerator = exp_difference(own, shift);
        let others = nonnegative(mass.mul(exp_difference(mass_shift, shift)));
        // a/(a+b) increases with a and decreases with b for a,b>=0.
        // Use fixed authorized endpoints for the monotone substitution rather
        // than dividing an uncertain numerator by its correlated denominator.
        let (a, b) = if lower {
            (numerator.lo, others.hi)
        } else {
            (numerator.hi, others.lo)
        };
        if !a.is_finite() || !b.is_finite() || a <= 0.0 || b < 0.0 {
            return if lower { 0.0 } else { 1.0 };
        }
        if !lower && b == 0.0 {
            return 1.0;
        }
        let ratio = I::point(a).div_positive(I::point(a).add(I::point(b)));
        if !ratio.finite() {
            return if lower { 0.0 } else { 1.0 };
        }
        (if lower { ratio.lo } else { ratio.hi }).clamp(0.0, 1.0)
    }
}

pub(super) fn bounds(
    lower: &[f64],
    upper: &[f64],
    poll: &mut impl FnMut() -> Result<()>,
) -> Result<Vec<(f32, f32)>> {
    poll()?;
    if lower.len() != upper.len() {
        return Err(NyError::InvalidSpec(
            "checked causal row endpoint length mismatch".into(),
        ));
    }
    // Nonfinite active scores retain the original conservative profile. The
    // caller removed all masked entries and validated row lengths beforehand.
    for (i, (&l, &u)) in lower.iter().zip(upper).enumerate() {
        if i % 64 == 0 {
            poll()?;
        }
        if !l.is_finite() || !u.is_finite() || l > u {
            return Ok(vec![(0.0, 1.0); lower.len()]);
        }
    }
    if lower.len() == 1 {
        return Ok(vec![(1.0, 1.0)]);
    }
    if lower.is_empty() {
        return Ok(Vec::new());
    }
    let lower_mass = EndpointMass::new(lower, poll)?;
    let upper_mass = EndpointMass::new(upper, poll)?;
    let mut output = Vec::with_capacity(lower.len());
    for i in 0..lower.len() {
        if i % 64 == 0 {
            poll()?;
        }
        // Coordinate monotonicity: own lower/other upper for min, and own
        // upper/other lower for max. The true corner is covered independently.
        let lo = f64_to_f32_down(upper_mass.ratio(i, lower[i], true)).max(0.0);
        let hi = f64_to_f32_up(lower_mass.ratio(i, upper[i], false)).min(1.0);
        output.push(if lo <= hi { (lo, hi) } else { (0.0, 1.0) });
    }
    poll()?;
    Ok(output)
}

#[cfg(test)]
mod tests;
