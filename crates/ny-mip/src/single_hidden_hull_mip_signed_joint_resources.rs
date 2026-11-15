// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Paid post-optimization selection; no credit or publication hold is renewed.
use super::*;
pub(super) const MAX_JOINT_WORK: usize = 4_000_000;
pub(super) const MAX_PAIRS: usize = 2;
const MAX_COMPONENTS: usize = 64;
#[derive(Debug, Clone)]
pub(super) struct Choice {
    pub left: usize,
    pub right: usize,
    pub partial: bool,
    pub masks: usize,
    pub work: usize,
    pub bytes: usize,
    pub score: f64,
}
pub(super) fn setup(row: &SignedMajorant, cuts: &HullRootMipCuts<'_>) -> Option<(usize, usize)> {
    let p = cuts.problem;
    let c = row.plan.components.len();
    if !(2..=MAX_COMPONENTS).contains(&c) {
        return None;
    }
    let cells = c
        .checked_add(p.hidden_dim.checked_mul(2)?)?
        .checked_mul(p.input_dim.checked_add(1)?)?;
    let pairs = c.checked_mul(c)?;
    let work = cells
        .checked_add(pairs.checked_mul(p.input_dim.checked_add(4)?)?)?
        .checked_add(p.hidden_dim)?
        .checked_add(1)?
        .checked_mul(16)?
        .checked_add(
            row.witness
                .len()
                .checked_add(row.signature.len())?
                .checked_mul(8)?,
        )?;
    let bytes = c
        .checked_mul(p.input_dim.checked_add(1)?)?
        .checked_add(p.hidden_dim)?
        .checked_add(1)?
        .checked_mul(32)?
        .checked_add(pairs.checked_mul(96)?)?
        .checked_add(4096)?;
    (work <= MAX_JOINT_WORK).then_some((work, bytes))
}
pub(super) fn quote(
    p: &HullMarginProblem,
    a: &Component,
    b: &Component,
) -> Option<(usize, usize, usize)> {
    let k = a.negative.len().checked_add(b.negative.len())?;
    let positives = a.positive.len().checked_add(b.positive.len())?;
    if !(2..=8).contains(&k) || positives == 0 || positives > MAX_POSITIVE {
        return None;
    }
    let masks = 1usize.checked_shl(k as u32)?;
    let width = p.input_dim.checked_add(1)?;
    let per = width
        .checked_mul(positives.checked_add(k)?.checked_add(3)?)?
        .checked_mul(4)?
        .checked_add(positives.min(4).checked_mul(width)?.checked_mul(20)?)?;
    let terms = p
        .hidden_dim
        .checked_add(p.input_dim)?
        .checked_add(positives)?
        .checked_add(k)?
        .checked_add(16)?;
    let work = masks
        .checked_mul(per)?
        .checked_mul(4)?
        .checked_add(terms.checked_mul(128)?)?;
    let bytes = masks
        .checked_mul(positives)?
        .checked_mul(32)?
        .checked_add(terms.checked_mul(256)?)?
        .checked_add(8192)?;
    (work <= MAX_JOINT_WORK).then_some((masks, work, bytes))
}
/// Whole quote is consumed BEFORE its allocation/scans; no refund on refusal.
/// Publication and old component holds remain in the same funding inequality.
pub(super) fn pay(row: &SignedMajorant, work: usize, bytes: usize, end: Instant) -> Option<()> {
    let b = &row.budget;
    if let Some(ticket) = &b.prepaid {
        return ticket.pay(row, work, bytes, end);
    }
    if !live(end.min(b.search_end)) {
        stop(b, "deadline");
        return None;
    }
    let mut c = b.state.lock().ok()?;
    let next = c.work.checked_add(work)?;
    let used = c.component_rows.joint.work_units.checked_add(work)?;
    if used > MAX_JOINT_WORK
        || next
            .checked_add(c.reserved)?
            .checked_add(c.component_reserved)?
            > MAX_WORK
    {
        c.component_rows.joint.stop_reason = "work";
        return None;
    }
    let total = b
        .admitted_bytes
        .checked_add(c.shared_bytes)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.child.cache_bytes)?
        .checked_add(c.component_rows.admitted_bytes)?
        .checked_add(bytes)?;
    if total > MAX_BYTES {
        c.component_rows.joint.stop_reason = "payload";
        return None;
    }
    let component_work = c.component_rows.work_units.checked_add(work)?;
    let component_bytes = c.component_rows.admitted_bytes.checked_add(bytes)?;
    let joint_bytes = c.component_rows.joint.admitted_bytes.checked_add(bytes)?;
    c.work = next;
    c.component_rows.work_units = component_work;
    c.component_rows.admitted_bytes = component_bytes;
    c.component_rows.joint.work_units = used;
    c.component_rows.joint.admitted_bytes = joint_bytes;
    Some(())
}
/// Generic cancellation score of signed absolute affine mass, never head IDs.
/// Pair metadata is retained only under the already-paid setup payload.
pub(super) fn choices(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<Choice>> {
    let p = cuts.problem;
    let plan = &row.plan;
    let d = p.input_dim;
    let mut features = Vec::with_capacity(plan.components.len());
    for c in &plan.components {
        let mut v = vec![0.; d + 1];
        for (&j, &units) in c.positive.iter().zip(&c.positive_units) {
            if !live(end) || !poll() {
                return None;
            }
            let mass = finite(p.c[j] * (f64::from(units) * 0.5))?;
            for i in 0..d {
                v[i] = finite(v[i] + mass * p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]))?;
            }
            v[d] = finite(v[d] + mass * p.b1[j].abs())?;
        }
        for &j in &c.negative {
            if !live(end) || !poll() {
                return None;
            }
            for i in 0..d {
                v[i] = finite(v[i] + p.c[j] * p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]))?;
            }
            v[d] = finite(v[d] + p.c[j] * p.b1[j].abs())?;
        }
        features.push(v);
    }
    let n = plan.components.len();
    let mut out = Vec::with_capacity(n.checked_mul(n)? / 2);
    for left in 0..n {
        for right in left + 1..n {
            if !live(end) || !poll() {
                return None;
            }
            let a = &plan.components[left];
            let b = &plan.components[right];
            let large = a.negative.len().checked_add(b.negative.len())? > 8;
            let quoted = if large {
                partial::quote(p, a, b)
            } else {
                quote(p, a, b)
            };
            let Some((masks, work, bytes)) = quoted else {
                continue;
            };
            let mut energy = 0.;
            for (&a, &b) in features[left].iter().zip(&features[right]) {
                let gain = finite(finite(a.abs() + b.abs())? - finite(a + b)?.abs())?.max(0.);
                energy = finite(energy + gain)?;
            }
            let score = finite(energy / (work as f64))?;
            if score > 0. {
                out.push(Choice {
                    left,
                    right,
                    partial: large,
                    masks,
                    work,
                    bytes,
                    score,
                });
            }
        }
    }
    // At most two selections: no unquoted sort or pair-count dependent replay.
    Some(out)
}
pub(super) fn pick(row: &SignedMajorant, choices: &[Choice], selected: &[usize]) -> Option<usize> {
    if let Some(ticket) = &row.budget.prepaid {
        let (allowance, room) = ticket.remaining(row)?;
        let mut best = None;
        for (i, p) in choices.iter().enumerate() {
            if p.partial || selected.contains(&i) || p.work > allowance || p.bytes > room {
                continue;
            }
            if best.is_none_or(|j: usize| p.score > choices[j].score) {
                best = Some(i);
            }
        }
        return best;
    }
    let c = row.budget.state.lock().ok()?;
    let free = MAX_WORK
        .checked_sub(c.work)?
        .checked_sub(c.reserved)?
        .checked_sub(c.component_reserved)?;
    let allowance = MAX_JOINT_WORK
        .checked_sub(c.component_rows.joint.work_units)?
        .min(free);
    let bytes = row
        .budget
        .admitted_bytes
        .checked_add(c.shared_bytes)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.child.cache_bytes)?
        .checked_add(c.component_rows.admitted_bytes)?;
    let room = MAX_BYTES.checked_sub(bytes)?;
    let mut best = None;
    for (i, p) in choices.iter().enumerate() {
        if p.partial || selected.contains(&i) || p.work > allowance || p.bytes > room {
            continue;
        }
        if best.is_none_or(|j: usize| p.score > choices[j].score) {
            best = Some(i);
        }
    }
    best
}

/// Metadata-only choice, under the original paid setup. No charged retry or
/// renewed allowance: one large quote must fit BEFORE any small-pair spending.
pub(super) fn first_partial(row: &SignedMajorant, choices: &[Choice]) -> Option<usize> {
    if let Some(ticket) = &row.budget.prepaid {
        let (allowance, room) = ticket.remaining(row)?;
        let mut best = None;
        for (i, p) in choices.iter().enumerate() {
            if !p.partial || p.work > allowance || p.bytes > room {
                continue;
            }
            if best.is_none_or(|j: usize| p.score > choices[j].score) {
                best = Some(i);
            }
        }
        return best;
    }
    let c = row.budget.state.lock().ok()?;
    let free = MAX_WORK
        .checked_sub(c.work)?
        .checked_sub(c.reserved)?
        .checked_sub(c.component_reserved)?;
    let allowance = MAX_JOINT_WORK
        .checked_sub(c.component_rows.joint.work_units)?
        .min(free);
    let used = row
        .budget
        .admitted_bytes
        .checked_add(c.shared_bytes)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.child.cache_bytes)?
        .checked_add(c.component_rows.admitted_bytes)?;
    let room = MAX_BYTES.checked_sub(used)?;
    let mut best = None;
    for (i, p) in choices.iter().enumerate() {
        if !p.partial || p.work > allowance || p.bytes > room {
            continue;
        }
        if best.is_none_or(|j: usize| p.score > choices[j].score) {
            best = Some(i);
        }
    }
    best
}
