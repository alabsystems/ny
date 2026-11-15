// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Nominal neutral direction sweeps. Only the existing full-q replay decides.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedDirectionReport {
    pub attempts: usize,
    pub completed_sweeps: usize,
    pub coordinates: usize,
    pub cached_masks: usize,
    pub checked_candidates: usize,
    pub accepted_candidates: usize,
    pub work_units: usize,
    pub cache_bytes: usize,
    pub stop_reason: &'static str,
}
impl Default for SignedDirectionReport {
    fn default() -> Self {
        Self {
            attempts: 0,
            completed_sweeps: 0,
            coordinates: 0,
            cached_masks: 0,
            checked_candidates: 0,
            accepted_candidates: 0,
            work_units: 0,
            cache_bytes: 0,
            stop_reason: "not_attempted",
        }
    }
}
fn reason(b: &Budget, value: &'static str) {
    if let Ok(mut c) = b.state.lock() {
        c.direction.stop_reason = value;
    }
}
fn poll_live(b: &Budget, poll: &mut dyn FnMut() -> bool) -> Option<()> {
    if !live(b.search_end) {
        reason(b, "deadline");
        return None;
    }
    if !poll() {
        reason(b, "poll");
        return None;
    }
    Some(())
}
fn charge(b: &Budget, work: usize) -> Option<()> {
    if b.charge(work).is_none() {
        reason(b, "work");
        return None;
    }
    b.state.lock().ok()?.direction.work_units += work;
    Some(())
}
pub(super) struct Quote {
    pub(super) work: usize,
    pub(super) bytes: usize,
}
pub(super) fn quote(p: &HullMarginProblem, plan: &Plan) -> Option<Quote> {
    if !capacity(p)
        || p.input_dim == 0
        || plan.components.is_empty()
        || plan.components.len() > p.hidden_dim
        || plan.leftover.len() > p.hidden_dim
    {
        return None;
    }
    let mut masks = 0usize;
    let mut work =
        64usize.checked_mul(p.input_dim.checked_mul(p.hidden_dim)? + p.hidden_dim + 1)?;
    let mut alpha = 0usize;
    for c in &plan.components {
        if c.negative.is_empty()
            || c.negative.len() > 8
            || c.positive.len() > MAX_POSITIVE
            || c.positive_units.len() != c.positive.len()
            || c.positive_units.iter().any(|u| !matches!(u, 1 | 2))
            || c.g.len() != p.input_dim
        {
            return None;
        }
        let m = 1usize.checked_shl(c.negative.len() as u32)?;
        if c.alpha.len() != m.checked_mul(c.positive.len())? {
            return None;
        }
        masks = masks.checked_add(m)?;
        alpha = alpha.checked_add(c.alpha.len())?;
        work = work.checked_add(
            m.checked_mul(p.input_dim + 1)?
                .checked_mul(c.positive.len() + c.negative.len() + 1)?
                .checked_mul(8)?,
        )?;
    }
    if masks > MAX_MASKS {
        return None;
    }
    let cells = masks.checked_mul(p.input_dim)?;
    // Two full-plan coordinate sweeps; the second rebases cached sums.
    work = work
        .checked_add(cells.checked_mul(2 * 64 + 8)?)?
        .checked_add(
            (alpha + plan.components.len() * p.input_dim + p.hidden_dim + 1).checked_mul(32)?,
        )?;
    // Three Vec payloads/cache plus a 2x allowance and descriptor margin.
    let bytes = masks
        .checked_mul(p.input_dim + 2)?
        .checked_mul(16)?
        .checked_add(plan.components.len().checked_mul(128)?)?;
    Some(Quote { work, bytes })
}
struct Cache {
    coeff: Vec<f64>,
    constant: Vec<f64>,
    value: Vec<f64>,
}
fn minimum(v: f64, l: f64, u: f64) -> Option<f64> {
    finite((finite(v * l)?).min(finite(v * u)?))
}
impl Cache {
    fn new(
        cuts: &HullRootMipCuts<'_>,
        c: &Component,
        b: &Budget,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        let p = cuts.problem;
        let masks = 1usize << c.negative.len();
        let mut out = Self {
            coeff: vec![0.; masks * p.input_dim],
            constant: vec![0.; masks],
            value: vec![0.; masks],
        };
        for mask in 0..masks {
            poll_live(b, poll)?;
            let alpha = &c.alpha[mask * c.positive.len()..(mask + 1) * c.positive.len()];
            for (slot, (&j, &a)) in c.positive.iter().zip(alpha).enumerate() {
                poll_live(b, poll)?;
                let k = finite(c.nominal_coefficient(p, slot)? * a)?;
                out.constant[mask] = finite(out.constant[mask] + finite(k * p.b1[j])?)?;
                for i in 0..p.input_dim {
                    let index = mask * p.input_dim + i;
                    out.coeff[index] = finite(out.coeff[index] + finite(k * p.w_row(j)[i])?)?;
                }
            }
            for (bit, &j) in c.negative.iter().enumerate() {
                if mask & (1 << bit) == 0 {
                    continue;
                }
                poll_live(b, poll)?;
                out.constant[mask] = finite(out.constant[mask] + finite(p.c[j] * p.b1[j])?)?;
                for i in 0..p.input_dim {
                    let index = mask * p.input_dim + i;
                    out.coeff[index] = finite(out.coeff[index] + finite(p.c[j] * p.w_row(j)[i])?)?;
                }
            }
            out.rebase_mask(p, c, mask)?;
            poll_live(b, poll)?;
            b.state.lock().ok()?.direction.cached_masks += 1;
        }
        Some(out)
    }
    fn rebase_mask(&mut self, p: &HullMarginProblem, c: &Component, mask: usize) -> Option<()> {
        let mut value = self.constant[mask];
        for i in 0..p.input_dim {
            value = finite(
                value
                    + minimum(
                        finite(self.coeff[mask * p.input_dim + i] - c.g[i])?,
                        p.lower[i],
                        p.upper[i],
                    )?,
            )?;
        }
        self.value[mask] = value;
        Some(())
    }
    fn coordinate(
        &mut self,
        p: &HullMarginProblem,
        c: &mut Component,
        i: usize,
        b: &Budget,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        let (l, u) = (p.lower[i], p.upper[i]);
        if l == u {
            return Some(());
        }
        let width = finite(u - l)?;
        if width <= 0. {
            return None;
        }
        let old = c.g[i];
        let (mut left, mut right) = (f64::INFINITY, f64::INFINITY);
        for mask in 0..self.value.len() {
            poll_live(b, poll)?;
            let v = self.coeff[mask * p.input_dim + i];
            let a = finite(self.value[mask] - minimum(finite(v - old)?, l, u)?)?;
            left = left.min(finite(a + finite(v * l)?)?);
            right = right.min(finite(a + finite(v * u)?)?);
        }
        // This intrinsic breakpoint is ALWAYS on the exact whole-root
        // plateau, independent of other components. Rounded caches are advice.
        let point = finite(finite(right - left)? / width)?;
        for mask in 0..self.value.len() {
            poll_live(b, poll)?;
            let v = self.coeff[mask * p.input_dim + i];
            let a = finite(self.value[mask] - minimum(finite(v - old)?, l, u)?)?;
            self.value[mask] = finite(a + minimum(finite(v - point)?, l, u)?)?;
        }
        poll_live(b, poll)?;
        c.g[i] = point;
        b.state.lock().ok()?.direction.coordinates += 1;
        Some(())
    }
}
fn nominal(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    next_cost: usize,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Plan> {
    {
        let mut counts = b.state.lock().ok()?;
        if counts.direction.attempts != 0 {
            return None;
        }
        counts.direction.attempts = 1;
        counts.direction.stop_reason = "search";
    }
    let p = cuts.problem;
    if b.source != std::ptr::from_ref(p) as usize
        || b.authority > cuts.deadline
        || p.c.len() != p.hidden_dim
        || b.dimensions
            != (
                p.input_dim,
                p.hidden_dim,
                p.c.iter().filter(|v| **v > 0.).count(),
            )
    {
        reason(b, "source");
        return None;
    }
    if !b.can_check() {
        reason(b, "checks");
        return None;
    }
    poll_live(b, poll)?;
    charge(
        b,
        16usize.checked_mul(
            p.hidden_dim
                .checked_add(plan.components.len())?
                .checked_add(p.input_dim)?
                .checked_add(1)?,
        )?,
    )?;
    let Some(q) = quote(p, plan) else {
        reason(b, "shape");
        return None;
    };
    if b.payload_bytes()?.checked_add(q.bytes)? > MAX_BYTES {
        reason(b, "payload");
        return None;
    }
    if b.remaining()?
        .checked_sub(next_cost)
        .is_none_or(|room| q.work > room)
    {
        reason(b, "work");
        return None;
    }
    charge(b, q.work)?;
    b.state.lock().ok()?.direction.cache_bytes = q.bytes;
    shape(cuts)?;
    plan.valid(cuts)?;
    require_hull_environment().ok()?;
    if p.lower.iter().zip(&p.upper).all(|(l, u)| l == u) {
        reason(b, "fixed_box");
        return None;
    }
    let mut trial = plan.clone();
    let mut caches = Vec::with_capacity(trial.components.len());
    for component in &trial.components {
        caches.push(Cache::new(cuts, component, b, poll)?);
    }
    for sweep in 0..2 {
        for (component, cache) in trial.components.iter_mut().zip(&mut caches) {
            if sweep > 0 {
                for mask in 0..cache.value.len() {
                    poll_live(b, poll)?;
                    cache.rebase_mask(p, component, mask)?;
                }
            }
            for i in 0..p.input_dim {
                cache.coordinate(p, component, i, b, poll)?;
            }
        }
        poll_live(b, poll)?;
        b.state.lock().ok()?.direction.completed_sweeps += 1;
    }
    if trial
        .components
        .iter()
        .zip(&plan.components)
        .all(|(a, old)| {
            a.g.iter()
                .map(|v| v.to_bits())
                .eq(old.g.iter().map(|v| v.to_bits()))
        })
    {
        reason(b, "unchanged");
        return None;
    }
    Some(trial)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn refine(
    cuts: &HullRootMipCuts<'_>,
    plan: &mut Plan,
    current: &mut f64,
    best: &mut Option<SignedMajorant>,
    floor: f64,
    cost: usize,
    b: &Budget,
    seal: &Arc<RetainedWords>,
    poll: &mut dyn FnMut() -> bool,
) {
    if *current > 0. {
        return;
    }
    let Some(trial) = nominal(cuts, plan, b, cost, poll) else {
        if b.state
            .lock()
            .is_ok_and(|c| c.direction.stop_reason == "search")
        {
            reason(b, "refused");
        }
        return;
    };
    if b.proof(cost, false).is_none() {
        reason(b, "check_budget");
        return;
    }
    b.state
        .lock()
        .expect("private signed ledger")
        .direction
        .work_units += cost;
    let Some(checked) = certificate::check(cuts, &trial, b, b.search_end, poll) else {
        reason(b, "source_refused");
        return;
    };
    b.state
        .lock()
        .expect("private signed ledger")
        .direction
        .checked_candidates += 1;
    if checked.root_lower <= *current {
        reason(b, "checked_no_gain");
        return;
    }
    *current = checked.root_lower;
    *plan = trial;
    b.state
        .lock()
        .expect("private signed ledger")
        .direction
        .accepted_candidates += 1;
    reason(
        b,
        if *current > 0. {
            "positive"
        } else {
            "accepted"
        },
    );
    if *current > floor {
        *best = Some(row(cuts, plan, checked, cost, b, seal));
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_direction_tests.rs"]
mod tests;
