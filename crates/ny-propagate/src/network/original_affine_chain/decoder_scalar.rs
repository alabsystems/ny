// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Paid scalar arithmetic for an exact-real decoder interpreter. Numeric
//! intervals are public data; they carry no model or reachable-state authority.
//! GELU reuses the qualified private kernel; trig uses checked Machin pi and
//! Taylor's derivative remainder, independently of every host trig/libm value.
//! The series are those in <https://dlmf.nist.gov/4.19>; the remainder below
//! additionally uses the global absolute derivative bound1 for sine/cosine.

use super::*;
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecoderInterval {
    pub lo: f64,
    pub hi: f64,
}
impl DecoderInterval {
    pub const fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    fn internal(self) -> I {
        I::new(self.lo, self.hi)
    }
    fn from_internal(value: I) -> Self {
        Self {
            lo: value.lo,
            hi: value.hi,
        }
    }
}

/// Conservative primitive capacities, supplied before any execution. Actual
/// arithmetic spends the prepaid phase; exhausting it is an explicit refusal.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecoderMathCounts {
    pub elementary: u64,
    pub gelu: u64,
    pub trig: u64,
    pub roots: u64,
    pub exponentials: u64,
}
impl DecoderMathCounts {
    pub fn request(self) -> Result<WorkRequest> {
        let mut work = 128 * 1024u64;
        for (count, unit) in [
            (self.elementary, 256u64),
            (self.gelu, 64 * 1024),
            (self.trig, 32 * 1024),
            (self.roots, 32 * 1024),
            (self.exponentials, 16 * 1024),
        ] {
            work = work
                .checked_add(
                    count
                        .checked_mul(unit)
                        .ok_or_else(|| invalid("decoder scalar work overflow"))?,
                )
                .ok_or_else(|| invalid("decoder scalar work overflow"))?;
        }
        Ok(WorkRequest {
            scratch_bytes: SCRATCH,
            persistent_bytes: 4096,
            work,
            proposals: 0,
        })
    }
}

pub struct CheckedDecoderMath {
    phase: OriginalAffinePhase,
    kernel: CheckedActivation,
    trig_tail: I,
    deadline: Option<Instant>,
}
impl CheckedDecoderMath {
    pub fn new(
        budget: &WorkBudget,
        requested: Option<Instant>,
        counts: DecoderMathCounts,
    ) -> Result<Self> {
        let deadline = [requested, ny_core::instance_budget::deadline()]
            .into_iter()
            .flatten()
            .min();
        let mut phase =
            OriginalAffinePhase::admit(budget, LeaseClass::Fallback, counts.request()?, deadline)?;
        let kernel = CheckedActivation::prepare(MathematicalActivation::GeluErf, &mut phase)?;
        let mut math = Math::new(&mut phase)?;
        // Both sin degree49 and cos degree48 have a derivative remainder
        // <=4^49/49! on [-4,4]; sin's degree50 remainder is smaller.
        let mut tail = I::point(1.0);
        for n in 1..=49 {
            tail = math.mul(tail, I::point(4.0))?;
            tail = math.div(tail, I::point(n as f64))?;
        }
        let trig_tail = math.finite(tail)?;
        phase.checkpoint()?;
        let deadline = phase.deadline();
        Ok(Self {
            phase,
            kernel,
            trig_tail,
            deadline,
        })
    }
    pub fn checkpoint(&self) -> Result<()> {
        ny_core::require_f64_interval_proof_environment()?;
        self.phase.checkpoint()
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    /// Charge bounded caller-side coordinate/lookup bookkeeping on this phase.
    /// No call may hide an unbounded loop behind a bulk credit deduction.
    pub fn bookkeeping(&mut self, units: u64) -> Result<()> {
        if units > 256 {
            return Err(invalid("decoder bookkeeping quantum"));
        }
        self.phase.charge_work(units)?;
        self.phase.checkpoint()
    }
    fn unary(&mut self, value: DecoderInterval) -> Result<I> {
        let mut math = Math::new(&mut self.phase)?;
        math.finite(value.internal())
    }
    pub fn add(&mut self, a: DecoderInterval, b: DecoderInterval) -> Result<DecoderInterval> {
        let (a, b) = (self.unary(a)?, self.unary(b)?);
        let mut math = Math::new(&mut self.phase)?;
        math.pay(64)?;
        let value = math.add(a, b)?;
        Ok(DecoderInterval::from_internal(math.finite(value)?))
    }
    pub fn sub(&mut self, a: DecoderInterval, b: DecoderInterval) -> Result<DecoderInterval> {
        let (a, b) = (self.unary(a)?, self.unary(b)?);
        let mut math = Math::new(&mut self.phase)?;
        math.pay(64)?;
        let value = math.sub(a, b)?;
        Ok(DecoderInterval::from_internal(math.finite(value)?))
    }
    pub fn mul(&mut self, a: DecoderInterval, b: DecoderInterval) -> Result<DecoderInterval> {
        let (a, b) = (self.unary(a)?, self.unary(b)?);
        let mut math = Math::new(&mut self.phase)?;
        math.pay(64)?;
        let value = math.mul(a, b)?;
        Ok(DecoderInterval::from_internal(math.finite(value)?))
    }
    pub fn div_positive(
        &mut self,
        a: DecoderInterval,
        b: DecoderInterval,
    ) -> Result<DecoderInterval> {
        let (a, b) = (self.unary(a)?, self.unary(b)?);
        let mut math = Math::new(&mut self.phase)?;
        math.pay(64)?;
        let value = math.div(a, b)?;
        Ok(DecoderInterval::from_internal(math.finite(value)?))
    }
    pub fn sqrt(&mut self, x: DecoderInterval) -> Result<DecoderInterval> {
        let x = self.unary(x)?;
        if x.lo < 0.0 {
            return Err(invalid("decoder sqrt domain"));
        }
        let mut math = Math::new(&mut self.phase)?;
        Ok(DecoderInterval::from_internal(checked_sqrt(x, &mut math)?))
    }
    pub fn exponential_nonpositive(&mut self, x: DecoderInterval) -> Result<DecoderInterval> {
        let x = self.unary(x)?;
        if x.hi > 0.0 {
            return Err(invalid("decoder exp requires nonpositive argument"));
        }
        let mut math = Math::new(&mut self.phase)?;
        let result = math.exponential(x)?;
        Ok(DecoderInterval::from_internal(math.finite(result)?))
    }
    pub fn gelu(&mut self, x: DecoderInterval) -> Result<DecoderInterval> {
        let x = self.unary(x)?;
        let mut math = Math::new(&mut self.phase)?;
        self.kernel.validate(&mut math)?;
        math.pay(64)?;
        let proposal = 0.5 * x.lo + 0.5 * x.hi;
        let c = if proposal.is_finite() && x.lo <= proposal && proposal <= x.hi {
            proposal
        } else {
            x.lo
        };
        let (value, derivative) = self.kernel.value_derivative(c, &mut math)?;
        let delta = math.sub(x, I::point(c))?;
        let radius = I::point(delta.abs_max());
        let radius2 = math.mul(radius, radius)?;
        let error = math.mul(self.kernel.half_global_curvature(), radius2)?;
        let linear = math.mul(derivative, delta)?;
        let image = math.add(value, linear)?;
        let image = math.add(image, I::new(-error.hi, error.hi))?;
        let image = math.finite(image)?;
        let image = math.finite(I::new(
            image.lo.max(x.lo.min(0.0)),
            image.hi.min(x.hi.max(0.0)),
        ))?;
        Ok(DecoderInterval::from_internal(image))
    }
    /// Mathematical sin/cos of the ENTIRE finite original angle interval.
    /// Any integer range-reduction proposal is valid by exact periodicity;
    /// its checked remainder must lie in [-4,4] before a Taylor bound is used.
    pub fn sin_cos(
        &mut self,
        angle: DecoderInterval,
    ) -> Result<(DecoderInterval, DecoderInterval)> {
        let angle = self.unary(angle)?;
        if angle.abs_max() > 4096.0 {
            return Err(invalid("decoder trig angle cap"));
        }
        let mut math = Math::new(&mut self.phase)?;
        self.kernel.validate(&mut math)?;
        math.pay(128)?;
        let center = angle.midpoint();
        let pi = self.kernel.pi;
        let q = ((center + pi.midpoint()) / (2.0 * pi.midpoint())).floor();
        if !q.is_finite() || q.abs() > 1024.0 {
            return Err(invalid("decoder trig integer reduction"));
        }
        let period = math.mul(pi, I::point(2.0 * q))?;
        let reduced = math.sub(angle, period)?;
        if reduced.abs_max() > 4.0 {
            return Err(invalid("decoder trig reduction range"));
        }
        let square = math.mul(reduced, reduced)?;
        let mut sine = reduced;
        let mut cosine = I::point(1.0);
        let mut st = reduced;
        let mut ct = I::point(1.0);
        for n in 1..=24usize {
            let s = math.mul(st, square)?;
            st = math.div(s.neg(), I::point((2 * n * (2 * n + 1)) as f64))?;
            sine = math.add(sine, st)?;
            let c = math.mul(ct, square)?;
            ct = math.div(c.neg(), I::point(((2 * n - 1) * 2 * n) as f64))?;
            cosine = math.add(cosine, ct)?;
        }
        let tail = I::new(-self.trig_tail.hi, self.trig_tail.hi);
        let s = math.add(sine, tail)?;
        let c = math.add(cosine, tail)?;
        let s = math.finite(I::new(s.lo.max(-1.0), s.hi.min(1.0)))?;
        let c = math.finite(I::new(c.lo.max(-1.0), c.hi.min(1.0)))?;
        math.phase.checkpoint()?;
        Ok((
            DecoderInterval::from_internal(s),
            DecoderInterval::from_internal(c),
        ))
    }
    pub fn finish(self) -> Result<PersistentReservation> {
        self.checkpoint()?;
        self.phase.commit()
    }
}

#[cfg(test)]
#[path = "decoder_scalar_tests.rs"]
mod tests;
