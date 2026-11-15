// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Metadata quotes for the single preserved-mask proposal.
//! Requires the independently reviewed root payload/capacity-seal seam.
use super::*;
use std::mem::size_of;

pub(super) fn capacity(plan: &Plan) -> Option<usize> {
    payload::heap(plan)
}
fn words_bound(plan: &Plan) -> Option<usize> {
    // At most one new positive appearance and its at-most-256 alpha values.
    resources::words(plan)?.checked_add(258)
}
pub(super) fn new_capacity_bound(plan: &Plan) -> Option<usize> {
    words_bound(plan)?
        .checked_mul(
            size_of::<usize>()
                .max(size_of::<f64>())
                .max(size_of::<u64>()),
        )?
        .checked_add(plan.components.len().checked_mul(size_of::<Component>())?)?
        .checked_add(128)
}

pub(super) fn admission_with_transport(
    plan: &Plan,
    b: &Budget,
) -> Option<(usize, usize, usize, bool)> {
    let (meta, baseline, extended, words) = admission(plan, b)?;
    let state = b.state.lock().ok()?;
    let fits = b
        .admitted_bytes
        .checked_add(extended)?
        .checked_add(state.shared_bytes)?
        .checked_add(state.direction.cache_bytes)?
        .checked_add(state.child.cache_bytes)?
        .checked_add(state.component_rows.admitted_bytes)?
        <= MAX_BYTES;
    Some((meta, if fits { extended } else { baseline }, words, fits))
}

pub(super) fn admission(plan: &Plan, b: &Budget) -> Option<(usize, usize, usize, usize)> {
    let (d, h, p) = b.dimensions;
    let c = plan.components.len();
    if !(2..=64).contains(&c) || c > h || p == 0 || !eligible(plan) {
        return None;
    }
    let old = resources::words(plan)?;
    let new = words_bound(plan)?;
    let final_bytes = new_capacity_bound(plan)?;
    let old_heap = capacity(plan)?;
    if old_heap > final_bytes {
        return None;
    }
    let w = d.checked_add(1)?;
    let m = 512usize;
    // Both residual arrays, endpoint alpha, mean, all six (nonconcurrent)
    // D+1 scratch/global arrays, and both selection metadata vectors.
    let search = m
        .checked_mul(w)?
        .checked_mul(2)?
        .checked_add(m)?
        .checked_add(d)?
        .checked_add(w.checked_mul(6)?)?
        .checked_mul(size_of::<f64>())?
        .checked_add(c.checked_mul(size_of::<Option<usize>>() + size_of::<bool>())?)?
        .checked_add(4096)?;
    // The old boxed witness is shared across all phases. Nine final Plan
    // allowances cover the retained publication/refinement phase; the current
    // Source signature is allocated once outside this proposal, not cloned.
    let shared = old
        .checked_mul(size_of::<u64>())?
        .checked_add(
            c.checked_add(14)?
                .checked_mul(w)?
                .checked_add(h)?
                .checked_add(p.checked_mul(2)?)?
                .checked_add(1)?
                .checked_mul(128)?,
        )?
        .checked_add(8192)?
        .checked_add(1536)?;
    let phase = old_heap.checked_add(search)?;
    let final_phase = final_bytes.checked_mul(9)?;
    let payload = shared.checked_add(phase.max(final_phase))?;
    // The competing endpoint and two stored target directions coexist with
    // both old caches. They are dropped before an old-branch Plan is rebuilt.
    let extra = m
        .checked_add(d.checked_mul(2)?)?
        .checked_mul(size_of::<f64>())?;
    let extended = shared.checked_add(phase.checked_add(extra)?.max(final_phase))?;
    // Includes metadata-only capacity accounting, the initial boxed seal,
    // two borrowed whole-Plan seal comparisons, uniform-alpha scans, all
    // whole-positive feature scans and the complete C*C scalar tournament.
    let meta = c
        .checked_add(h.checked_mul(2)?)?
        .checked_mul(w)?
        .checked_add(c.checked_mul(c)?.checked_mul(d.checked_add(16)?)?)?
        .checked_add(h)?
        .checked_add(1)?
        .checked_mul(16)?
        .checked_add(old.checked_mul(32)?)?
        .checked_add(c.checked_mul(128)?)?
        .checked_add(4096)?;
    Some((meta, payload, extended, new))
}

pub(super) fn transported_extra(plan: &Plan, d: usize, a: usize, b: usize) -> Option<usize> {
    let m = (1usize.checked_shl(plan.components.get(a)?.negative.len() as u32)?)
        .checked_add(1usize.checked_shl(plan.components.get(b)?.negative.len() as u32)?)?;
    let w = d.checked_add(1)?;
    let k = d.min(4);
    // Endpoint search, then direct cached-row reprice. The four-face updates
    // reuse the endpoint scores; no second mask-by-dimension face scan.
    m.checked_mul(w)?
        .checked_mul(64)?
        .checked_add(m.checked_mul(k)?.checked_mul(20)?)?
        .checked_add(w.checked_mul(128)?)?
        .checked_add(
            plan.components
                .len()
                .checked_add(plan.leftover.len())?
                .checked_add(8)?
                .checked_mul(w)?
                .checked_mul(16)?,
        )?
        .checked_add(4096)
}

pub(super) fn oblique_extra(plan: &Plan, d: usize, a: usize, b: usize) -> Option<usize> {
    let m = (1usize.checked_shl(plan.components.get(a)?.negative.len() as u32)?)
        .checked_add(1usize.checked_shl(plan.components.get(b)?.negative.len() as u32)?)?;
    let w = d.checked_add(1)?;
    // One additional full stored-row reprice; active-mask bookkeeping fused
    // into the already paid face scan; <=4 full-D gradients and one trial;
    // a third complete original-order global sum. No extra checker or LP.
    m.checked_mul(w)?
        .checked_mul(16)?
        .checked_add(m.checked_mul(32)?)?
        .checked_add(w.checked_mul(512)?)?
        .checked_add(
            plan.components
                .len()
                .checked_add(plan.leftover.len())?
                .checked_add(8)?
                .checked_mul(w)?
                .checked_mul(16)?,
        )?
        .checked_add(4096)
}

fn setup(d: usize, c: &Component, fast: bool) -> Option<usize> {
    let w = d.checked_add(1)?;
    let m = 1usize.checked_shl(c.negative.len() as u32)?;
    let p = c.positive.len().checked_add(1)?; // append allowance in both roles
    let uniform_scan = m.checked_mul(p)?.checked_mul(4)?;
    let arithmetic = if fast {
        // One common affine positive sum; Gray-code negative updates with a
        // literal original-order rebase every 16 masks.
        p.checked_add(1)?
            .checked_mul(w)?
            .checked_mul(8)?
            .checked_add(m.checked_mul(w)?.checked_mul(8)?)?
            .checked_add(
                m.div_ceil(16)
                    .checked_mul(c.negative.len())?
                    .checked_mul(w)?
                    .checked_mul(8)?,
            )?
    } else {
        // Every positive and negative head is visited literally for each mask.
        m.checked_mul(p.checked_add(c.negative.len())?.checked_add(1)?)?
            .checked_mul(w)?
            .checked_mul(8)?
    };
    uniform_scan.checked_add(arithmetic)
}

pub(super) fn choice(
    plan: &Plan,
    d: usize,
    a: usize,
    b: usize,
    exchange: bool,
    fast: [bool; 2],
    old_cost: usize,
    whole_words: usize,
) -> Option<(usize, usize, usize)> {
    let ca = plan.components.get(a)?;
    let cb = plan.components.get(b)?;
    let ka = ca.negative.len();
    let kb = cb.negative.len();
    if a == b || !(1..=8).contains(&ka) || !(1..=8).contains(&kb) {
        return None;
    }
    let pa = ca.positive.len();
    let pb = cb.positive.len();
    if pa == 0 || pa > 16 || pb > 16 || !exchange && pb == 16 {
        return None;
    }
    let check = if exchange {
        old_cost
    } else {
        old_cost
            .checked_sub(resources::component_cost(d, ka, pa, 0)?)?
            .checked_sub(resources::component_cost(d, kb, pb, 0)?)?
            .checked_add(resources::component_cost(d, ka, pa, 1)?)?
            .checked_add(resources::component_cost(d, kb, pb.checked_add(1)?, 1)?)?
    };
    let m = (1usize << ka).checked_add(1usize << kb)?;
    let w = d.checked_add(1)?;
    let setup = setup(d, ca, fast[0])?.checked_add(setup(d, cb, fast[1])?)?;
    // Endpoint: seed/zero/one/single-breakpoint scores, residual and delta
    // construction. Three complete pair-mask/global line probes follow.
    let endpoint = m.checked_mul(w)?.checked_mul(48)?;
    let probes = m.checked_mul(w)?.checked_mul(16)?.checked_mul(3)?;
    // Includes mean/head directions, complete global affine setup, one final
    // Plan clone/copy, mask alpha interpolation, masks/cost/capacity scans,
    // finite-value guards and the handoff/completion bookkeeping.
    let finish = whole_words
        .checked_add(258)?
        .checked_mul(16)?
        .checked_add(
            plan.components
                .len()
                .checked_add(plan.leftover.len())?
                .checked_add(8)?
                .checked_mul(w)?
                .checked_mul(16)?,
        )?
        .checked_add(65536)?;
    let search = setup
        .checked_add(endpoint)?
        .checked_add(probes)?
        .checked_add(finish)?;
    Some((check, search, check.checked_add(search)?))
}
