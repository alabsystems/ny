// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Small outward scalar checker, deliberately independent of search hints.
use super::{AffineHead, DecoderRefusal, HeadArithmetic, HiddenBox};
use crate::zono_core::rounding::{gamma, next_down, next_up, TINY};

#[derive(Debug, Clone, Copy)]
pub(super) struct Interval {
    pub lo: f64,
    pub hi: f64,
}
fn finite(v: f64) -> Result<f64, DecoderRefusal> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(DecoderRefusal::NonFinite("outward arithmetic"))
    }
}
pub(super) fn up_add(a: f64, b: f64) -> Result<f64, DecoderRefusal> {
    finite(next_up(a + b))
}
pub(super) fn up_mul(a: f64, b: f64) -> Result<f64, DecoderRefusal> {
    finite(next_up(a * b))
}
pub(super) fn difference(a: f64, b: f64) -> Result<Interval, DecoderRefusal> {
    if a == b {
        return Ok(Interval { lo: 0.0, hi: 0.0 });
    }
    Ok(Interval {
        lo: finite(next_down(a - b))?,
        hi: finite(next_up(a - b))?,
    })
}
pub(super) fn product(a: Interval, b: Interval) -> Result<Interval, DecoderRefusal> {
    let values = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
    for &v in &values {
        finite(v)?;
    }
    Ok(Interval {
        lo: finite(next_down(
            values.iter().copied().fold(f64::INFINITY, f64::min),
        ))?,
        hi: finite(next_up(
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ))?,
    })
}
pub(super) fn abs_upper(v: Interval) -> f64 {
    v.lo.abs().max(v.hi.abs())
}
pub(super) fn sqrt_up(x: f64) -> Result<f64, DecoderRefusal> {
    if x == 0.0 {
        return Ok(0.0);
    }
    if !x.is_finite() || x < 0.0 {
        return Err(DecoderRefusal::NonFinite("norm square"));
    }
    let mut y = finite(next_up(x).sqrt())?;
    // Verify the square enclosure, rather than trusting sqrt accuracy alone.
    for _ in 0..4 {
        let lower_square = next_down(y * y);
        if lower_square.is_finite() && lower_square >= x {
            return Ok(y);
        }
        y = finite(next_up(y))?;
    }
    Err(DecoderRefusal::NonFinite("sqrt enclosure"))
}
pub(super) fn radius(hidden: HiddenBox<'_>, anchor: &[f64]) -> Result<Vec<f64>, DecoderRefusal> {
    hidden
        .lo
        .iter()
        .zip(hidden.hi)
        .zip(anchor)
        .map(
            |((&lo, &hi), &a)| Ok(abs_upper(difference(lo, a)?).max(abs_upper(difference(hi, a)?))),
        )
        .collect()
}
pub(super) fn execution_error(
    head: AffineHead<'_>,
    row: usize,
    hidden: HiddenBox<'_>,
) -> Result<f64, DecoderRefusal> {
    if head.arithmetic == HeadArithmetic::ExactReal {
        return Ok(0.0);
    }
    let mut absolute = head.bias[row].abs();
    for ((&w, &lo), &hi) in head.row(row).iter().zip(hidden.lo).zip(hidden.hi) {
        absolute = up_add(absolute, up_mul(w.abs(), lo.abs().max(hi.abs()))?)?;
    }
    arithmetic_error(head.arithmetic, absolute, head.hidden_size)
}
pub(super) fn arithmetic_error(
    arithmetic: HeadArithmetic,
    absolute: f64,
    dim: usize,
) -> Result<f64, DecoderRefusal> {
    match arithmetic {
        HeadArithmetic::ExactReal => Ok(0.0),
        HeadArithmetic::SequentialBinary64 => sequential_error(absolute, dim),
        HeadArithmetic::Binary32 { .. } => super::execution::binary32_error(absolute, dim),
    }
}
pub(super) fn sequential_error(absolute: f64, dim: usize) -> Result<f64, DecoderRefusal> {
    let operations = dim
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(DecoderRefusal::Resource("head operation count"))?;
    let error = up_add(
        up_mul(gamma(operations), absolute)?,
        TINY * operations as f64,
    )?;
    // The error theorem requires absence of overflow at every partial sum.
    // Their magnitudes are bounded by the entire absolute sum plus this error.
    up_add(absolute, error)?;
    Ok(error)
}
/// Direct box support of `w_j-w_target`, never separate logit intervals.
pub(super) fn dense_upper(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    target: usize,
    row: usize,
) -> Result<f64, DecoderRefusal> {
    let mut out = difference(head.bias[row], head.bias[target])?.hi;
    for (((&w, &t), &lo), &hi) in head
        .row(row)
        .iter()
        .zip(head.row(target))
        .zip(hidden.lo)
        .zip(hidden.hi)
    {
        out = up_add(out, product(difference(w, t)?, Interval { lo, hi })?.hi)?;
    }
    up_add(
        up_add(out, execution_error(head, row, hidden)?)?,
        execution_error(head, target, hidden)?,
    )
}
pub(super) fn anchor_margin(
    head: AffineHead<'_>,
    anchor: &[f64],
    target: usize,
    row: usize,
) -> Result<f64, DecoderRefusal> {
    let mut out = difference(head.bias[row], head.bias[target])?.hi;
    for ((&w, &t), &a) in head.row(row).iter().zip(head.row(target)).zip(anchor) {
        out = up_add(
            out,
            product(difference(w, t)?, Interval { lo: a, hi: a })?.hi,
        )?;
    }
    Ok(out)
}
pub(super) fn anchor_support(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    group_anchor: usize,
) -> Result<f64, DecoderRefusal> {
    let mut out = 0.0;
    for ((((&w, &t), &lo), &hi), &a) in head
        .row(group_anchor)
        .iter()
        .zip(head.row(target))
        .zip(hidden.lo)
        .zip(hidden.hi)
        .zip(anchor)
    {
        let dl = difference(lo, a)?.lo;
        let du = difference(hi, a)?.hi;
        out = up_add(
            out,
            product(difference(w, t)?, Interval { lo: dl, hi: du })?.hi,
        )?;
    }
    Ok(out)
}
