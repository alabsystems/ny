// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Whole-negative-mass geometry; optimizer values are never authority.
use super::*;
use std::sync::Arc;
#[path = "single_hidden_hull_mip_sdp_budget.rs"]
mod budget;
use budget::*;
#[path = "single_hidden_hull_mip_sdp_certificate.rs"]
mod certificate;
use certificate::System;
#[path = "single_hidden_hull_mip_sdp_search.rs"]
mod search;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalSdpReport {
    pub dimension: usize,
    pub active_dimension: usize,
    pub optimizer_iterations: usize,
    pub factorizations: usize,
    pub checked_candidates: usize,
    pub source_replays: usize,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub correction: f64,
    pub lower: f64,
}
#[derive(Debug)]
pub(super) struct GlobalMajorant {
    source: usize,
    signature: Arc<RetainedWords>,
    g: Vec<f64>,
    lambda: Vec<f64>,
    l: Vec<f64>,
    upper: f64,
    lower: f64,
    correction: f64,
    n: usize,
    budget: Budget,
}
#[derive(Debug)]
pub(super) struct Replay {
    pub g: Vec<f64>,
    pub upper: f64,
    pub lower: f64,
}
fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}
fn signature_values<'a>(cuts: &'a HullRootMipCuts<'_>) -> impl Iterator<Item = u64> + 'a {
    let p = cuts.problem;
    [
        p.input_dim as u64,
        p.hidden_dim as u64,
        p.w1.len() as u64,
        p.b1.len() as u64,
        p.c.len() as u64,
        p.lower.len() as u64,
        p.upper.len() as u64,
        cuts.multipliers.alpha.len() as u64,
        cuts.multipliers.g.len() as u64,
    ]
    .into_iter()
    .chain(
        p.w1.iter()
            .chain(&p.b1)
            .chain(&p.c)
            .chain([&p.d])
            .chain(&p.lower)
            .chain(&p.upper)
            .chain(&cuts.multipliers.alpha)
            .chain(&cuts.multipliers.g)
            .map(|x| x.to_bits()),
    )
}
fn shape(cuts: &HullRootMipCuts<'_>) -> Option<usize> {
    let p = cuts.problem;
    let cells = p.input_dim.checked_mul(p.hidden_dim)?;
    if !capacity(p)
        || p.input_dim == 0
        || p.hidden_dim == 0
        || p.w1.len() != cells
        || p.b1.len() != p.hidden_dim
        || p.c.len() != p.hidden_dim
        || p.lower.len() != p.input_dim
        || p.upper.len() != p.input_dim
        || cuts.multipliers.alpha.len() != p.hidden_dim
        || cuts.multipliers.g.len() != cells
        || !cuts.checked_lower.is_finite()
        || p.w1
            .iter()
            .chain(&p.b1)
            .chain(&p.c)
            .chain([&p.d])
            .chain(&p.lower)
            .chain(&p.upper)
            .chain(&cuts.multipliers.alpha)
            .chain(&cuts.multipliers.g)
            .any(|v| !v.is_finite())
        || p.lower.iter().zip(&p.upper).any(|(l, u)| l > u)
    {
        return None;
    }
    1usize
        .checked_add(p.input_dim)?
        .checked_add(p.c.iter().filter(|c| **c < 0.).count())
}
fn positive_point(cuts: &HullRootMipCuts<'_>) -> Option<Vec<f64>> {
    let p = cuts.problem;
    if cuts.multipliers.alpha.len() != p.hidden_dim {
        return None;
    }
    let mut g = vec![0.; p.input_dim];
    for j in 0..p.hidden_dim {
        if p.c[j] <= 0. {
            continue;
        }
        let k = finite(p.c[j] * sanitize_alpha(cuts.multipliers.alpha[j]))?;
        for i in 0..p.input_dim {
            g[i] = finite(g[i] + k * p.w_row(j)[i])?;
        }
    }
    Some(g)
}
/// Independent full original-q lower replay. The point v never silently
/// replaces exact positive weighted sums; their complete residual remains.
fn root_lower(
    cuts: &HullRootMipCuts<'_>,
    g: &[f64],
    upper: f64,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    require_hull_environment().ok()?;
    let p = cuts.problem;
    if g.len() != p.input_dim || cuts.multipliers.alpha.len() != p.hidden_dim {
        return None;
    }
    let mut coeff: Vec<_> = g
        .iter()
        .map(|v| certificate::I::point(-v))
        .collect::<Option<_>>()?;
    let mut constant = certificate::I::point(p.d)?.sub(certificate::I::point(upper)?)?;
    for j in 0..p.hidden_dim {
        if !live(end) || !poll() {
            return None;
        }
        if p.c[j] <= 0. {
            continue;
        }
        let k = certificate::I::point(p.c[j])?.scale(sanitize_alpha(cuts.multipliers.alpha[j]))?;
        constant = constant.add(k.scale(p.b1[j])?)?;
        for i in 0..p.input_dim {
            coeff[i] = coeff[i].add(k.scale(p.w_row(j)[i])?)?;
        }
    }
    let mut lower = constant.lo;
    for i in 0..p.input_dim {
        let a = coeff[i];
        let (l, u) = (p.lower[i], p.upper[i]);
        let v = mul_down(a.lo, l)
            .min(mul_down(a.lo, u))
            .min(mul_down(a.hi, l))
            .min(mul_down(a.hi, u));
        lower = finite(add_down(lower, finite(v)?))?;
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some(lower)
}
impl GlobalMajorant {
    pub(super) fn g(&self) -> &[f64] {
        &self.g
    }
    pub(super) fn upper(&self) -> f64 {
        self.upper
    }
    pub(super) fn proposed_lower(&self) -> f64 {
        self.lower
    }
    pub(super) fn authority(&self) -> Instant {
        self.budget.authority
    }
    pub(super) fn stats(&self) -> GlobalSdpReport {
        let c = self
            .budget
            .counts
            .lock()
            .expect("private SDP ledger poisoned");
        GlobalSdpReport {
            dimension: self.n,
            active_dimension: c.active,
            optimizer_iterations: c.iterations,
            factorizations: c.factors,
            checked_candidates: c.checked,
            source_replays: c.completed_replays,
            work_units: c.work,
            admitted_bytes: self.budget.admitted_bytes,
            correction: self.correction,
            lower: self.lower,
        }
    }
    pub(super) fn replay(
        &self,
        cuts: &HullRootMipCuts<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Replay> {
        let end = end.min(self.budget.authority).min(cuts.deadline);
        if !live(end) || !poll() || self.source != std::ptr::from_ref(cuts.problem) as usize {
            return None;
        }
        self.budget.proof(true)?;
        if shape(cuts)? != self.n || !signature_values(cuts).eq(self.signature.iter().copied()) {
            return None;
        }
        let system = certificate::system(cuts, self.g(), end, poll)?;
        let (upper, correction) = certificate::certify(&system, &self.lambda, &self.l, end, poll)?;
        if upper.to_bits() != self.upper().to_bits()
            || correction.to_bits() != self.correction.to_bits()
        {
            return None;
        }
        let lower = root_lower(cuts, self.g(), upper, end, poll)?;
        if lower.to_bits() != self.lower.to_bits() {
            return None;
        }
        self.budget.counts.lock().ok()?.completed_replays += 1;
        Some(Replay {
            g: self.g.clone(),
            upper,
            lower,
        })
    }
    pub(super) fn prepare_rows(
        &self,
        cuts: &HullRootMipCuts<'_>,
        parts: &MipParts,
        columns: &[crate::ir::Col],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Vec<RowSpec>> {
        let row = self.replay(cuts, end, poll)?;
        let p = cuts.problem;
        let mut coeffs = Vec::new();
        for (j, &c) in p.c.iter().enumerate() {
            if c < 0. {
                coeffs.push((columns.get(j)?.0, -c));
            }
        }
        for (i, &g) in row.g.iter().enumerate() {
            if !live(end) || !poll() {
                return None;
            }
            if g != 0. {
                coeffs.push((parts.input_vars.get(i)?.0, -g));
            }
        }
        (live(end) && poll()).then_some(vec![RowSpec {
            lb: f64::NEG_INFINITY,
            ub: row.upper,
            coeffs,
        }])
    }
}
pub(super) fn propose(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> Option<GlobalMajorant> {
    require_hull_environment().ok()?;
    let p = cuts.problem;
    let n = shape(cuts)?;
    if n > MAX_N {
        return None;
    }
    let budget = Budget::new(cuts, authority, reserve, n)?;
    budget.charge(budget.proof_cost, false)?;
    let mut floor = cuts.checked_lower;
    if let Some(row) = cuts.compact.as_ref() {
        floor = floor.max(compact::root_lower(
            cuts,
            row,
            budget.search_end,
            &mut || true,
        )?);
    }
    if floor > 0. {
        return None;
    }
    let g = positive_point(cuts)?;
    let system = certificate::system(cuts, &g, budget.search_end, &mut || true)?;
    let seal = Arc::new(signature_values(cuts).collect::<RetainedWords>());
    let mut best: Option<GlobalMajorant> = None;
    let mut consider = |lambda: Vec<f64>, l: Vec<f64>| {
        if budget.proof(false).is_none() {
            return;
        }
        let Some((upper, correction)) =
            certificate::certify(&system, &lambda, &l, budget.search_end, &mut || true)
        else {
            return;
        };
        let Some(lower) = root_lower(cuts, &g, upper, budget.search_end, &mut || true) else {
            return;
        };
        if let Ok(mut c) = budget.counts.lock() {
            c.checked += 1;
        } else {
            return;
        }
        if lower <= floor || best.as_ref().is_some_and(|r| lower <= r.lower) {
            return;
        }
        best = Some(GlobalMajorant {
            source: std::ptr::from_ref(p) as usize,
            signature: seal.clone(),
            g: g.clone(),
            lambda,
            l,
            upper,
            lower,
            correction,
            n,
            budget: budget.clone(),
        });
    };
    let _ = search::run(&system, &budget, &mut consider);
    if !live(budget.authority) {
        return None;
    }
    best
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_sdp_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_sdp_retention_tests.rs"]
mod retention_tests;
