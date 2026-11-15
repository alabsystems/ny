// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cooperative numerical dual adapter. The caller owns the original-domain
//! halfspaces and admits the complete request on that domain's existing
//! ledger before entering this helper. Neither the callback nor a numerical
//! result authenticates supplied rows or mints a proof owner.
//!
//! The nominal coordinate sweep preserves the legacy arithmetic order. This
//! entry sorts equal breakpoints in original-coordinate order; the legacy
//! unstable sort has no specified tie order. Distinct-breakpoint proposals
//! retain legacy bits. Every finite nonnegative proposal is independently
//! enclosed by the same directed weak-duality calculation.

use super::{clip_add_down, clip_add_up, clip_mul_bounds, DualOutwardBound, Halfspace};
use ny_core::{NyError, Result};

const POLL_UNITS: usize = 256;
const META_BYTES: usize = 4096;

/// Complete simultaneously live local buffers. Halfspaces, the objective,
/// domain arrays, caller checkpoint work and other consumers are additional
/// caller-owned overlap. This estimate alone grants no memory authority.
#[derive(Clone, Copy, Debug)]
pub(in crate::margin_row) struct DualArithmeticRequest {
    pub(in crate::margin_row) scratch_bytes: usize,
    pub(in crate::margin_row) work: u64,
}

fn invalid(message: &'static str) -> NyError {
    NyError::InvalidSpec(message.into())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("cooperative dual size overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("cooperative dual size overflow"))
}

/// O(1) quote derived from the loops below. Four coefficient planes use 32N
/// bytes and two breakpoint planes use 32N bytes. No library sort scratch or
/// per-halfspace allocation is hidden. Each merge level visits at most N
/// records and copies at most N records back, at <=96 units/record including
/// run/index bookkeeping and the live environment probes between blocks.
/// There are ceil(log2 N) levels per halfspace.
///
/// The remaining quote is 1280N + 768K(N+1) + 1024 units: input validation,
/// six buffer initializations, per-halfspace validation/proposal/sweep/fold,
/// and the final directed box evaluation. Fixed-size directed primitives,
/// finite tests, indexing and polling bookkeeping fit the charged blocks
/// below. Each directed add/multiply is separately charged 64 units: a
/// TwoSum has six scalar additions/subtractions, eight finite tests and at
/// most one adjacent-float conversion; a product has one multiplication,
/// finite/sign/exact-zero/unit tests and at most two adjacent conversions.
/// The nominal proposal coordinate costs at most 64 units, a sweep record
/// 32, and a merge record 32. Each live environment probe is also charged 64.
/// An arbitrary caller callback's own work needs its own admission.
pub(in crate::margin_row) fn dual_lower_outward_request(
    n_in: usize,
    n_halfspaces: usize,
) -> Result<DualArithmeticRequest> {
    let levels = if n_in <= 1 {
        0
    } else {
        (usize::BITS - (n_in - 1).leading_zeros()) as usize
    };
    let width = add(n_in, 1)?;
    let per_halfspace = add(mul(768, width)?, mul(mul(96, width)?, levels)?)?;
    let units = add(
        add(1024, mul(1280, n_in)?)?,
        mul(n_halfspaces, per_halfspace)?,
    )?;
    Ok(DualArithmeticRequest {
        scratch_bytes: add(META_BYTES, mul(64, n_in)?)?,
        work: u64::try_from(units).map_err(|_| invalid("cooperative dual work overflow"))?,
    })
}

struct DualPoll<'a> {
    check: &'a dyn Fn() -> Result<()>,
    pending: usize,
    #[cfg(test)]
    total: usize,
    #[cfg(test)]
    max_gap: usize,
}
impl<'a> DualPoll<'a> {
    fn new(check: &'a dyn Fn() -> Result<()>) -> Result<Self> {
        let mut out = Self {
            check,
            pending: 0,
            #[cfg(test)]
            total: 0,
            #[cfg(test)]
            max_gap: 0,
        };
        out.checkpoint()?;
        Ok(out)
    }
    /// Charge BEFORE each bounded block. The largest block is 64 units;
    /// no loop, including sort movement/initialization, bypasses this poll.
    fn step(&mut self, units: usize) -> Result<()> {
        debug_assert!(units <= POLL_UNITS - 64);
        if self.pending > POLL_UNITS - units {
            self.checkpoint()?;
        }
        self.pending += units;
        #[cfg(test)]
        {
            self.total += units;
            self.max_gap = self.max_gap.max(self.pending);
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<()> {
        (self.check)()?;
        // A callback may run other numerical work. Requalify the live thread
        // after it returns, before any interval primitive runs.
        // The pinned probe executes five scalar operations, seven black-box
        // inputs and five bit comparisons. 64 units covers it and bookkeeping.
        // Callback work is external and must be priced by the caller.
        self.pending = 64;
        #[cfg(test)]
        {
            self.total += 64;
            self.max_gap = self.max_gap.max(self.pending);
        }
        ny_core::require_f64_interval_proof_environment()
    }
}

fn buffer<T: Copy>(n: usize, value: T, poll: &mut DualPoll<'_>) -> Result<Vec<T>> {
    poll.checkpoint()?;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| invalid("cooperative dual buffer allocation refused"))?;
    poll.checkpoint()?;
    if out.capacity() != n {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: out.capacity().saturating_mul(size_of::<T>()),
            budget_bytes: n.saturating_mul(size_of::<T>()),
            site: "cooperative dual buffer actual capacity",
        });
    }
    for _ in 0..n {
        poll.step(8)?;
        out.push(value);
    }
    Ok(out)
}

/// Bottom-up stable merge sort with no recursive or library-sort allocation.
/// Every compare/copy and every run boundary is cooperatively charged.
fn sort_breakpoints(
    points: &mut [(f64, f64)],
    tmp: &mut [(f64, f64)],
    poll: &mut DualPoll<'_>,
) -> Result<()> {
    debug_assert_eq!(points.len(), tmp.len());
    let n = points.len();
    let mut width = 1;
    while width < n {
        poll.step(16)?;
        let mut start = 0;
        while start < n {
            poll.step(16)?;
            let mid = start.saturating_add(width).min(n);
            let end = mid.saturating_add(width).min(n);
            let (mut left, mut right) = (start, mid);
            for out in start..end {
                poll.step(32)?;
                let take_left = left < mid
                    && (right == end || points[left].0.total_cmp(&points[right].0).is_le());
                if take_left {
                    tmp[out] = points[left];
                    left += 1;
                } else {
                    tmp[out] = points[right];
                    right += 1;
                }
            }
            start = end;
        }
        for j in 0..n {
            poll.step(8)?;
            points[j] = tmp[j];
        }
        width = width.saturating_mul(2);
    }
    Ok(())
}

fn argmax(
    work: &[f64],
    h: &Halfspace,
    mid: &[f64],
    rad: &[f64],
    points: &mut [(f64, f64)],
    tmp: &mut [(f64, f64)],
    poll: &mut DualPoll<'_>,
) -> Result<f64> {
    poll.step(32)?;
    let mut value = 0.0;
    let mut slope = h.d;
    let mut count = 0;
    for j in 0..work.len() {
        poll.step(64)?;
        let (w, a, x, r) = (work[j], h.a[j], mid[j], rad[j]);
        value += w.mul_add(x, -(w.abs() * r));
        let positive = if w > 0.0 {
            true
        } else if w < 0.0 {
            false
        } else {
            a > 0.0
        };
        slope += a * if positive { x - r } else { x + r };
        if a != 0.0 {
            let beta = -w / a;
            if beta.is_finite() && beta > 0.0 {
                points[count] = (beta, -2.0 * a.abs() * r);
                count += 1;
            }
        }
    }
    poll.step(32)?;
    if !value.is_finite() || !slope.is_finite() || slope <= 0.0 {
        return Ok(0.0);
    }
    sort_breakpoints(&mut points[..count], &mut tmp[..count], poll)?;
    let (mut current_beta, mut current_value) = (0.0, value);
    let (mut best_beta, mut best_value) = (0.0, value);
    for &(beta, delta_slope) in &points[..count] {
        poll.step(32)?;
        if slope <= 0.0 {
            break;
        }
        current_value += slope * (beta - current_beta);
        current_beta = beta;
        slope += delta_slope;
        if current_value > best_value && current_value.is_finite() {
            best_value = current_value;
            best_beta = current_beta;
        }
    }
    Ok(if slope > 0.0 { 0.0 } else { best_beta })
}

fn product(a: f64, b: f64, poll: &mut DualPoll<'_>) -> Result<Option<(f64, f64)>> {
    poll.step(64)?;
    Ok(clip_mul_bounds(a, b))
}
fn add_lower(a: f64, b: f64, poll: &mut DualPoll<'_>) -> Result<Option<f64>> {
    poll.step(64)?;
    Ok(clip_add_down(a, b))
}
fn add_upper(a: f64, b: f64, poll: &mut DualPoll<'_>) -> Result<Option<f64>> {
    poll.step(64)?;
    Ok(clip_add_up(a, b))
}

fn calculate(
    obj: &[f64],
    hs: &[Halfspace],
    mid: &[f64],
    rad: &[f64],
    poll: &mut DualPoll<'_>,
) -> Result<Option<DualOutwardBound>> {
    if obj.len() != mid.len() || obj.len() != rad.len() {
        return Ok(None);
    }
    for j in 0..obj.len() {
        poll.step(32)?;
        if !obj[j].is_finite() || !mid[j].is_finite() || !rad[j].is_finite() || rad[j] < 0.0 {
            return Ok(None);
        }
    }
    for h in hs {
        poll.step(16)?;
        if h.a.len() != obj.len() || !h.d.is_finite() {
            return Ok(None);
        }
        for &a in &h.a {
            poll.step(8)?;
            if !a.is_finite() {
                return Ok(None);
            }
        }
    }
    let n = obj.len();
    let mut work = buffer(n, 0.0, poll)?;
    let mut lo_work = buffer(n, 0.0, poll)?;
    let mut hi_work = buffer(n, 0.0, poll)?;
    let mut mag_work = buffer(n, 0.0, poll)?;
    let mut points = buffer(n, (0.0, 0.0), poll)?;
    let mut tmp = buffer(n, (0.0, 0.0), poll)?;
    for j in 0..n {
        poll.step(16)?;
        work[j] = obj[j];
        lo_work[j] = obj[j];
        hi_work[j] = obj[j];
        mag_work[j] = obj[j].abs();
    }
    let (mut nominal_bias, mut lo_bias, mut mag_bias) = (0.0, 0.0, 0.0);
    for h in hs {
        poll.step(16)?;
        let beta = argmax(&work, h, mid, rad, &mut points, &mut tmp, poll)?;
        if !beta.is_finite() || beta < 0.0 {
            return Ok(None);
        }
        if beta == 0.0 {
            continue;
        }
        for j in 0..n {
            poll.step(16)?;
            work[j] = beta.mul_add(h.a[j], work[j]);
            if !work[j].is_finite() {
                return Ok(None);
            }
            let Some((pl, pu)) = product(beta, h.a[j], poll)? else {
                return Ok(None);
            };
            let Some(lo) = add_lower(lo_work[j], pl, poll)? else {
                return Ok(None);
            };
            let Some(hi) = add_upper(hi_work[j], pu, poll)? else {
                return Ok(None);
            };
            let Some((_, mp)) = product(beta, h.a[j].abs(), poll)? else {
                return Ok(None);
            };
            let Some(mag) = add_upper(mag_work[j], mp, poll)? else {
                return Ok(None);
            };
            poll.step(16)?;
            lo_work[j] = lo;
            hi_work[j] = hi;
            mag_work[j] = mag;
        }
        poll.step(16)?;
        nominal_bias += beta * h.d;
        if !nominal_bias.is_finite() {
            return Ok(None);
        }
        let Some((bias_product, _)) = product(beta, h.d, poll)? else {
            return Ok(None);
        };
        let Some(bias) = add_lower(lo_bias, bias_product, poll)? else {
            return Ok(None);
        };
        let Some((_, mag_product)) = product(beta, h.d.abs(), poll)? else {
            return Ok(None);
        };
        let Some(mag) = add_upper(mag_bias, mag_product, poll)? else {
            return Ok(None);
        };
        lo_bias = bias;
        mag_bias = mag;
    }
    let (mut lower, mut nominal, mut magnitude) = (lo_bias, nominal_bias, mag_bias);
    for j in 0..n {
        poll.step(16)?;
        let Some(xl) = add_lower(mid[j], -rad[j], poll)? else {
            return Ok(None);
        };
        let Some(xu) = add_upper(mid[j], rad[j], poll)? else {
            return Ok(None);
        };
        let mut product_min = f64::INFINITY;
        for (coefficient, endpoint) in [
            (lo_work[j], xl),
            (lo_work[j], xu),
            (hi_work[j], xl),
            (hi_work[j], xu),
        ] {
            let Some((product, _)) = product(coefficient, endpoint, poll)? else {
                return Ok(None);
            };
            product_min = product_min.min(product);
        }
        let Some(value) = add_lower(lower, product_min, poll)? else {
            return Ok(None);
        };
        lower = value;
        // Preserve nominal operation order; this value has no proof authority.
        poll.step(16)?;
        nominal += work[j].mul_add(mid[j], -(work[j].abs() * rad[j]));
        let Some(abs_endpoint) = add_upper(mid[j].abs(), rad[j], poll)? else {
            return Ok(None);
        };
        let Some((_, product)) = product(mag_work[j], abs_endpoint, poll)? else {
            return Ok(None);
        };
        let Some(mag) = add_upper(magnitude, product, poll)? else {
            return Ok(None);
        };
        magnitude = mag;
    }
    Ok(nominal.is_finite().then_some(DualOutwardBound {
        lower,
        nominal,
        magnitude,
    }))
}

/// Independently enclose the entire Lagrangian at the nominal nonnegative
/// multipliers. Coefficient/bias addition dust, signed products, four endpoint
/// corners and gradual underflow all remain paid by the directed primitives.
/// Exceptional numerical candidates yield None; cancellation/deadline/FPU
/// refusal propagates as an error. Inputs and prior certified bounds remain
/// unchanged. A consumer combines valid candidates with its previous bound.
pub(in crate::margin_row) fn dual_lower_outward_with_checkpoint(
    obj: &[f64],
    hs: &[Halfspace],
    mid: &[f64],
    rad: &[f64],
    check: &dyn Fn() -> Result<()>,
) -> Result<Option<DualOutwardBound>> {
    // Validate arithmetic of the complete O(1) quote even for direct adapter
    // calls. Admission and the callback's own cost remain caller obligations.
    let _request = dual_lower_outward_request(obj.len(), hs.len())?;
    let mut poll = DualPoll::new(check)?;
    let result = calculate(obj, hs, mid, rad, &mut poll)?;
    poll.checkpoint()?;
    Ok(result)
}

#[cfg(test)]
#[path = "clip_cooperative_tests.rs"]
mod tests;
