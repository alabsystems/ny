// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Protect completed-mask collection and first binding from optional search.
use super::*;

pub(super) struct Quote {
    pub bytes: usize,
    pub masks: usize,
    pub collection_work: usize,
    pub bind_work: usize,
}
pub(super) fn quote(cuts: &HullRootMipCuts<'_>, plan: &Plan) -> Option<Quote> {
    let masks = plan.masks()?;
    let (d, h, components) = (
        cuts.problem.input_dim,
        cuts.problem.hidden_dim,
        plan.components.len(),
    );
    let negatives = plan
        .components
        .iter()
        .try_fold(0usize, |n, c| n.checked_add(c.negative.len()))?;
    let worker_scratch = d
        .checked_mul(16)?
        .checked_add(h)?
        .checked_add(components.checked_mul(16)?)?
        .checked_mul(bab::HULL_BAB_MAX_WORKERS)?;
    let bytes = masks
        .checked_mul(16)?
        .checked_add(negatives.checked_mul(16)?)?
        .checked_add(d.checked_mul(32)?)?
        .checked_add(h.checked_mul(64)?)?
        .checked_add(components.checked_mul(160)?)?
        .checked_add(4096)?
        .checked_add(worker_scratch)?;
    let collection_work = masks
        .checked_add(negatives)?
        .checked_add(h)?
        .checked_add(d)?
        .checked_add(components)?
        .checked_add(1)?
        .checked_mul(16)?;
    // Exact bits() length: five scalars, head map, g, and table headers/heads/values.
    let seal_words = 5usize
        .checked_add(3usize.checked_mul(h)?)?
        .checked_add(d)?
        .checked_add(3usize.checked_mul(components)?)?
        .checked_add(negatives)?
        .checked_add(masks)?;
    let bind_work = d
        .checked_mul(h)?
        .checked_add(seal_words)?
        .checked_add(h)?
        .checked_add(1)?
        .checked_mul(64)?;
    Some(Quote {
        bytes,
        masks,
        collection_work,
        bind_work,
    })
}

pub(in crate::single_hidden_hull::mip_cuts::signed) fn prepay(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
) -> Option<()> {
    if !b.portfolio_phase_enabled
        || native_publication::funding(b).funded_calls == 0
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.authority > cuts.deadline
        || !live(b.search_end)
    {
        return None;
    }
    let mut c = b.state.lock().ok()?;
    if c.replay_attempts != 0
        || c.child.collection_attempts != 0
        || !c.child.stop_reason.is_empty()
        || c.child.cache_bytes != 0
        || c.prepaid_collection_work != 0
        || c.prepaid_bind_work != 0
    {
        return None;
    }
    // The negative partition is fixed after root coupling. Shared mass/search
    // may change alpha and positive ownership, which do not enter this quote.
    // This latch is not a new Source pass or permission to retry a refusal.
    c.child.stop_reason = "prepay-refused";
    let q = quote(cuts, plan)?;
    let prepaid = q.collection_work.checked_add(q.bind_work)?;
    let work = c.work.checked_add(prepaid)?;
    let child_work = c.child.work_units.checked_add(prepaid)?;
    let bytes = b
        .admitted_bytes
        .checked_add(c.shared_bytes)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.component_rows.admitted_bytes)?
        .checked_add(q.bytes)?;
    if bytes > MAX_BYTES
        || work
            .checked_add(c.reserved)?
            .checked_add(c.component_reserved)?
            > MAX_WORK
        || !live(b.search_end)
    {
        return None;
    }
    // Irreversible debit, shared by all owners. Never mutate admitted_bytes:
    // existing root-coupling tickets authenticate that exact baseline value.
    c.work = work;
    c.child.work_units = child_work;
    c.child.cache_bytes = q.bytes;
    c.prepaid_collection_work = q.collection_work;
    c.prepaid_bind_work = q.bind_work;
    c.child.stop_reason = "prepaid";
    Some(())
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_child_reservation_tests.rs"]
mod tests;
