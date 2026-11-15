// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Directed, same-source head-box lower bound. Ordinary advice is never proof.
//! The private entry requires fresh local certificates and a validated carrier.
//! It does not remove the original prefix, change the query, or publish a verdict.
use super::head_source::{resolve, target_slopes, Slopes, Source};
use super::*;
use ny_core::dd::{next_down_f64, next_up_f64};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Quote {
    dimensions: [usize; 3],
    pub(super) work: u64,
    pub(super) scratch_bytes: usize,
}

pub(super) fn quote(e: usize, d: usize, o: usize) -> Option<Quote> {
    if !(1..=65536).contains(&e)
        || !(2..=8192).contains(&d)
        || !(1..=8192).contains(&o)
        || d.checked_mul(o)? > 1 << 20
    {
        return None;
    }
    let (ee, dd, oo) = (
        u64::try_from(e).ok()?,
        u64::try_from(d).ok()?,
        u64::try_from(o).ok()?,
    );
    Some(Quote {
        dimensions: [e, d, o],
        work: 128u64
            .checked_mul(dd)?
            .checked_mul(ee.checked_add(oo)?.checked_add(1)?)?
            .checked_add(2048u64.checked_mul(ee.checked_add(dd)?.checked_add(1)?)?)?
            .checked_add(2 << 20)?,
        scratch_bytes: 16usize.checked_mul(e)?.checked_add(32 << 10)?,
    })
}

#[derive(Clone, Copy, Debug)]
struct I {
    lo: f64,
    hi: f64,
}
const ZERO: I = I { lo: 0., hi: 0. };
impl I {
    fn checked(lo: f64, hi: f64) -> Option<Self> {
        (ny_core::is_crown_coeff_safe_f64(lo) && ny_core::is_crown_coeff_safe_f64(hi) && lo <= hi)
            .then_some(Self { lo, hi })
    }
    fn point(v: f32) -> Option<Self> {
        ny_core::is_crown_coeff_safe(v).then_some(Self {
            lo: f64::from(v),
            hi: f64::from(v),
        })
    }
    fn zero(self) -> bool {
        self.lo == 0. && self.hi == 0.
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn add(self, rhs: Self) -> Option<Self> {
        if self.zero() {
            return Some(rhs);
        }
        if rhs.zero() {
            return Some(self);
        }
        Self::checked(
            next_down_f64(self.lo + rhs.lo),
            next_up_f64(self.hi + rhs.hi),
        )
    }
    fn mul(self, rhs: Self) -> Option<Self> {
        if self.zero() || rhs.zero() {
            return Some(ZERO);
        }
        let p = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        if p.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Self::checked(
            next_down_f64(p.iter().copied().fold(f64::INFINITY, f64::min)),
            next_up_f64(p.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        )
    }
    fn hull(self, rhs: Self) -> Self {
        Self {
            lo: self.lo.min(rhs.lo),
            hi: self.hi.max(rhs.hi),
        }
    }
    fn abs(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
}
fn poll(guard: &mut Poll<'_>, i: usize) -> Option<()> {
    (!i.is_multiple_of(32) || guard.check()).then_some(())
}

// Coefficient uncertainty may straddle zero. In that case enclose BOTH valid
// ReLU branches; guessing a rounded sign could select an invalid minorant.
fn relax(slopes: Slopes<'_>, j: usize, c: I) -> Option<(I, I)> {
    let positive = || -> Option<(I, I)> {
        Some((
            c.mul(I::point(slopes.positive[j])?)?,
            c.mul(
                slopes
                    .positive_bias
                    .map_or(Some(ZERO), |v| I::point(v[j]))?,
            )?,
        ))
    };
    let negative = || -> Option<(I, I)> {
        Some((
            c.mul(I::point(slopes.negative[j])?)?,
            c.mul(I::point(slopes.negative_bias[j])?)?,
        ))
    };
    if c.lo >= 0. {
        positive()
    } else if c.hi <= 0. {
        negative()
    } else {
        let (pa, pb) = positive()?;
        let (na, nb) = negative()?;
        Some((pa.hull(na), pb.hull(nb)))
    }
}

#[allow(clippy::too_many_arguments)]
fn lower(
    s: &Source<'_>,
    slopes: Slopes<'_>,
    paid: Quote,
    active: [usize; 2],
    ids: [[usize; 8]; 2],
    weights: [[f32; 8]; 2],
    multipliers: [f32; 2],
    bounds: [f32; 2],
    guard: &mut Poll<'_>,
) -> Option<f32> {
    ny_core::require_f64_interval_proof_environment().ok()?;
    if !dd_selfcheck_ok() || !guard.check() {
        return None;
    }
    let [e, d, o] = paid.dimensions;
    if quote(e, d, o) != Some(paid)
        || !valid_selection(active, ids, weights, d)
        || s.matrix.len() != d.checked_mul(e)?
        || s.lower.len() != e
        || s.upper.len() != e
        || s.final_matrix.len() != o.checked_mul(d)?
        || s.objective.len() != o
        || s.bias.is_some_and(|v| v.len() != d)
        || s.final_bias.is_some_and(|v| v.len() != o)
        || slopes.positive.len() != d
        || slopes.negative.len() != d
        || slopes.negative_bias.len() != d
        || slopes.positive_bias.is_some_and(|v| v.len() != d)
        || s.bn.is_some_and(|v| {
            [
                v.scale.len(),
                v.shift.len(),
                v.scale_error.len(),
                v.bias_error.len(),
                v.lower.len(),
                v.upper.len(),
            ]
            .iter()
            .any(|n| *n != d)
        })
    {
        return None;
    }
    let mut a = Vec::new();
    a.try_reserve_exact(e).ok()?;
    if a.capacity()
        .checked_mul(size_of::<I>())?
        .checked_add(32 << 10)?
        > paid.scratch_bytes
    {
        return None;
    }
    a.resize(e, ZERO);
    let mut b = ZERO;
    for k in 0..o {
        poll(guard, k)?;
        b = b.add(
            I::point(s.objective[k])?.mul(s.final_bias.map_or(Some(ZERO), |v| I::point(v[k]))?)?,
        )?;
    }
    let lambda = [I::point(multipliers[0])?, I::point(multipliers[1])?];
    for g in 0..2 {
        if multipliers[g] <= 0. || bounds[g] < 0. {
            return None;
        }
        b = b.add(lambda[g].mul(I::point(bounds[g])?)?.neg())?;
    }
    for j in 0..d {
        poll(guard, j)?;
        let mut c = ZERO;
        for k in 0..o {
            poll(guard, k)?;
            c = c.add(I::point(s.objective[k])?.mul(I::point(s.final_matrix[k * d + j])?)?)?;
        }
        let mut pre = ZERO;
        for g in 0..2 {
            for i in 0..active[g] {
                if ids[g][i] == j {
                    let product = lambda[g].mul(I::point(weights[g][i])?)?;
                    pre = pre.add(product.neg())?;
                    c = c.add(product.mul(I::point(2.)?)?)?;
                }
            }
        }
        let (q, intercept) = relax(slopes, j, c)?;
        let q = q.add(pre)?;
        b = b.add(intercept)?;
        let (scale, shift) = if let Some(bn) = s.bn {
            let l = I::point(bn.lower[j])?;
            let u = I::point(bn.upper[j])?;
            let se = I::point(bn.scale_error[j])?;
            let be = I::point(bn.bias_error[j])?;
            if l.lo > u.hi || se.lo < 0. || be.lo < 0. {
                return None;
            }
            let magnitude = I::checked(0., l.abs().max(u.abs()))?;
            let radius = se.mul(magnitude)?.add(be)?;
            let penalty = I::checked(0., q.abs())?.mul(I::checked(0., radius.hi)?)?;
            b = b.add(I::checked(-penalty.hi, 0.)?)?;
            (I::point(bn.scale[j])?, I::point(bn.shift[j])?)
        } else {
            (I::point(1.)?, ZERO)
        };
        let bias = s.bias.map_or(Some(ZERO), |v| I::point(v[j]))?;
        b = b.add(q.mul(scale.mul(bias)?.add(shift)?)?)?;
        let scaled = q.mul(scale)?;
        for k in 0..e {
            poll(guard, k)?;
            a[k] = a[k].add(scaled.mul(I::point(s.matrix[j * e + k])?)?)?;
        }
    }
    for k in 0..e {
        poll(guard, k)?;
        let l = I::point(s.lower[k])?;
        let u = I::point(s.upper[k])?;
        b = b.add(a[k].mul(I::checked(l.lo, u.hi)?)?)?;
    }
    let result = ny_core::f64_to_f32_down(b.lo);
    (ny_core::is_crown_coeff_safe(result) && guard.check()).then_some(result)
}

/// Move-only completion, still tied to this exact context and generation.
pub(super) struct Completed<'c> {
    seal: &'c ResidentCutCallSeal,
    generations: ResidentCutSnapshotGenerations,
    lower: f32,
}
impl Completed<'_> {
    pub(super) fn consume(self, context: &TwoLocalContext<'_, '_>) -> Option<f32> {
        (std::ptr::eq(self.seal, &raw const context.base.seal)
            && self.generations == context.base.generations.get()
            && Instant::now() < context.base.deadline)
            .then_some(self.lower)
    }
}

pub(super) fn evaluate<'c>(
    context: &'c TwoLocalContext<'_, '_>,
    carrier: &TwoLocalCarrier<'_>,
    fresh: &FreshTwoLocalSupport<'_, '_>,
    paid: Quote,
    guard: &mut Poll<'_>,
) -> Option<Completed<'c>> {
    if !guard.check() || paid.dimensions != context.quote.source_shape {
        return None;
    }
    let generation = context.base.generations.get();
    // Rebuild from the actual fresh proofs: neither advisory scores nor a
    // caller scalar/transport can declare the cut's support bound.
    let expected = context
        .build_with_scale(fresh, carrier.selected_scale)
        .ok()?;
    if !std::ptr::eq(carrier.seal, &raw const context.base.seal)
        || carrier.generations != context.base.generations.get()
        || carrier.active != expected.active
        || carrier.neurons != expected.neurons
        || !same_weights(carrier.weights, expected.weights)
        || !same_multipliers(carrier.multipliers, expected.multipliers)
        || carrier
            .bounds
            .iter()
            .zip(expected.bounds)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return None;
    }
    let source = resolve(&context.quote.base, paid.dimensions, guard)?;
    let slopes = target_slopes(context, paid.dimensions[1], guard)?;
    let value = lower(
        &source,
        slopes,
        paid,
        carrier.active,
        carrier.neurons,
        carrier.weights,
        carrier.multipliers,
        carrier.bounds,
        guard,
    )?;
    if generation != context.base.generations.get() || Instant::now() >= context.base.deadline {
        return None;
    }
    Some(Completed {
        seal: &context.base.seal,
        generations: generation,
        lower: value,
    })
}

#[cfg(test)]
#[path = "head_box_floor/tests.rs"]
mod tests;
