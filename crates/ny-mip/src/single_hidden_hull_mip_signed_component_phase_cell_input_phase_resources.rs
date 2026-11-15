// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Additive input-phase payment after immutable native and child priority.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Quote {
    pub d: usize,
    pub work: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub admitted: bool,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub masks: usize,
    pub candidates: usize,
    pub selected_candidate: Option<usize>,
    pub stop_reason: &'static str,
}
impl Default for Report {
    fn default() -> Self {
        Self {
            admitted: false,
            work_units: 0,
            admitted_bytes: 0,
            masks: 0,
            candidates: 0,
            selected_candidate: None,
            stop_reason: "not-attempted",
        }
    }
}

fn setup(d: usize, h: usize, components: usize) -> Option<usize> {
    h.checked_add(d)?
        .checked_add(components)?
        .checked_add(8)?
        .checked_mul(64)?
        .checked_add(4096)
}

/// No Source scan precedes payment. The quote uses the future P<=16 cap and
/// original Cell term cap, not a smaller seed component's positive census.
pub(in crate::single_hidden_hull::mip_cuts::signed) fn quote(
    d: usize,
    h: usize,
    k: usize,
    terms: usize,
    components: usize,
) -> Option<Quote> {
    if !(1..=32).contains(&d)
        || !(2..=8).contains(&k)
        || components == 0
        || components > h
        || terms != d.checked_add(32)?
    {
        return None;
    }
    let masks = 1usize << k;
    // M nominal affine tables plus two seed reconstructions. Each affine
    // visits <=Pcap+K original heads and D coordinates; includes box pricing.
    let affine = MAX_POSITIVE
        .checked_add(k)?
        .checked_add(2)?
        .checked_mul(d.checked_add(1)?)?
        .checked_add(k)?
        .checked_add(4)?
        .checked_mul(16)?;
    let nominal = masks
        .checked_add(2)?
        .checked_mul(affine)?
        .checked_add(32usize.checked_mul(masks)?)?
        .checked_add(256usize.checked_mul(d.checked_add(k)?.checked_add(1)?)?)?
        .checked_add(4096)?;
    // The same bounded proposal ceiling as Ray, twice for the two seeds.
    // Per-mask old Ray advice is reused from its original paid evaluation.
    let ray = 16usize
        .checked_mul(k)?
        .checked_mul(d.checked_add(1)?)?
        .checked_add(4usize.checked_mul(d)?.checked_mul(d.checked_sub(1)?)?)?
        .checked_add(128usize.checked_mul(d.checked_add(1)?)?)?
        .checked_add(128)?;
    // For each candidate/mask: shifted q, two directed box prices, one
    // original Source head correction, membership/shape checks and polling.
    let price = 128usize
        .checked_mul(d.checked_add(1)?)?
        .checked_add(16usize.checked_mul(k)?)?
        .checked_add(128)?;
    // FIVE independent supports: reference Ray plus four candidates. Final
    // actual-column Joint preparation retains its original paid allowance.
    let support = 32usize
        .checked_mul(k)?
        .checked_mul(k)?
        .checked_mul(masks)?
        .checked_add(128usize.checked_mul(k)?.checked_mul(masks)?)?
        .checked_add(64usize.checked_mul(masks)?)?
        .checked_add(128usize.checked_mul(k.checked_add(1)?.checked_pow(2)?)?)?
        .checked_add(
            256usize.checked_mul(
                terms
                    .checked_add(2usize.checked_mul(k)?)?
                    .checked_add(h)?
                    .checked_add(d)?
                    .checked_add(2)?,
            )?,
        )?
        .checked_add(4096)?;
    let compare = 128usize
        .checked_mul(d.checked_add(k)?.checked_add(1)?)?
        .checked_add(1024)?
        .checked_mul(5)?;
    // Complete-table checks, one irrevocable legacy-prefix rollback, g seal
    // and final export. No retry or repeated rollback earns another grant.
    let retention = 64usize
        .checked_mul(masks)?
        .checked_add(
            128usize.checked_mul(
                d.checked_add(k)?
                    .checked_add(h)?
                    .checked_add(components)?
                    .checked_add(16)?,
            )?,
        )?
        .checked_add(4096)?;
    let work = setup(d, h, components)?
        .checked_add(nominal)?
        .checked_add(2usize.checked_mul(ray)?)?
        .checked_add(4usize.checked_mul(masks)?.checked_mul(price)?)?
        .checked_add(5usize.checked_mul(support)?)?
        .checked_add(compare)?
        .checked_add(retention)?;
    // Three extra tables; four g vectors; one copied target; selected-g seal.
    // Primary table, row buffers and sequential numeric scratch are already
    // paid by Joint/Ray. Inline growth is charged once by Budget::new.
    let bytes = 3usize
        .checked_mul(masks)?
        .checked_add(6usize.checked_mul(d)?)?
        .checked_mul(size_of::<f64>())?;
    Some(Quote { d, work, bytes })
}

fn capture_target(cuts: &HullRootMipCuts<'_>, b: &Budget, d: usize) -> Option<Vec<f64>> {
    // Original capture is already banked. This new payment covers only the
    // bounded copy, not a second capture or a renewed authority clock.
    let account = cuts.primal.as_ref()?.lock().ok()?;
    let point = account.point(cuts.problem, b.search_end, &b.state)?;
    let p = cuts.problem;
    if point.input.len() != d || p.lower.len() != d || p.upper.len() != d {
        return None;
    }
    let mut input = Vec::with_capacity(d);
    if input.capacity() > d {
        return None;
    }
    for ((&x, &lower), &upper) in point.input.iter().zip(&p.lower).zip(&p.upper) {
        if !live(b.search_end)
            || !x.is_finite()
            || !lower.is_finite()
            || !upper.is_finite()
            || lower > upper
            || x < lower
            || x > upper
        {
            return None;
        }
        input.push(x);
    }
    live(b.search_end).then_some(input)
}

pub(in crate::single_hidden_hull::mip_cuts::signed) fn prepay(
    cuts: &HullRootMipCuts<'_>,
    b: &Budget,
) -> Option<()> {
    if !b.portfolio_phase_enabled
        || !b.phase_cube_enabled
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.dimensions.0 != cuts.problem.input_dim
        || b.dimensions.1 != cuts.problem.hidden_dim
        || b.authority > cuts.deadline
        || !live(b.search_end)
        || native_publication::funding(b).funded_calls == 0
    {
        return None;
    }
    let paid = {
        let mut c = b.state.lock().ok()?;
        let a = c.phase_cube.admission.as_ref()?;
        if !owned(a, b)
            || a.deadline != cuts.deadline
            || c.phase_cube.input_phase_attempted
            || c.phase_cube.finish_attempted
            || c.phase_cube.prepare_attempted
            || c.phase_cube.cell_attempted
            || c.replay_attempts != 0
            || c.source_replays != 0
            || c.component_rows.collection_attempts != 0
            || c.component_rows.phase.collection_attempts != 0
            || c.component_rows.phase.cube.collection_attempts != 0
            || c.child.collection_attempts != 0
            || c.child.stop_reason != "prepaid"
            || c.child.cache_bytes == 0
            || c.prepaid_collection_work == 0
            || c.prepaid_bind_work == 0
            || a.input_target.is_some()
        {
            return None;
        }
        let mut cube = a.quote;
        let mut cell = cube.cell?;
        if cell.input_phase.is_some() || cell.ray.is_none() || cell.joint.is_none() {
            return None;
        }
        c.phase_cube.input_phase_attempted = true;
        c.component_rows.phase.cube.cell.input_phase.stop_reason = "quote";
        let q = quote(
            b.dimensions.0,
            b.dimensions.1,
            cell.k,
            cell.terms,
            cube.components,
        )?;
        let setup = setup(q.d, b.dimensions.1, cube.components)?;
        let held = q.work.checked_sub(setup)?;
        let next_work = c.work.checked_add(setup)?;
        let next_reserved = c.component_reserved.checked_add(held)?;
        let total_bytes = b
            .admitted_bytes
            .checked_add(c.shared_bytes)?
            .checked_add(c.direction.cache_bytes)?
            .checked_add(c.child.cache_bytes)?
            .checked_add(c.component_rows.admitted_bytes)?
            .checked_add(q.bytes)?;
        if total_bytes > MAX_BYTES
            || next_work
                .checked_add(c.reserved)?
                .checked_add(next_reserved)?
                > MAX_WORK
            || !live(b.search_end)
        {
            c.component_rows.phase.cube.cell.input_phase.stop_reason = "work-or-payload";
            return None;
        }
        // Calculate every checked replacement before changing any ledger or
        // capability; overflow cannot leave partially installed authority.
        let component_work = c.component_rows.work_units.checked_add(setup)?;
        let component_bytes = c.component_rows.admitted_bytes.checked_add(q.bytes)?;
        let phase_work = c.component_rows.phase.work_units.checked_add(setup)?;
        let phase_bytes = c.component_rows.phase.admitted_bytes.checked_add(q.bytes)?;
        let phase_reserved = c
            .component_rows
            .phase
            .reserved_work_units
            .checked_add(held)?;
        let r = &c.component_rows.phase.cube;
        let cube_work = r.work_units.checked_add(setup)?;
        let cube_setup = r.setup_work.checked_add(setup)?;
        let cube_bytes = r.admitted_bytes.checked_add(q.bytes)?;
        let cube_reserved = r.reserved_work_units.checked_add(held)?;
        cube.work = cube.work.checked_add(q.work)?;
        cube.setup = cube.setup.checked_add(setup)?;
        cube.bytes = cube.bytes.checked_add(q.bytes)?;
        cell.input_phase = Some(q);
        cube.cell = Some(cell);

        c.work = next_work;
        c.component_reserved = next_reserved;
        c.component_rows.work_units = component_work;
        c.component_rows.reserved_work_units = next_reserved;
        c.component_rows.admitted_bytes = component_bytes;
        c.component_rows.phase.work_units = phase_work;
        c.component_rows.phase.admitted_bytes = phase_bytes;
        c.component_rows.phase.reserved_work_units = phase_reserved;
        let r = &mut c.component_rows.phase.cube;
        r.work_units = cube_work;
        r.setup_work = cube_setup;
        r.admitted_bytes = cube_bytes;
        r.reserved_work_units = cube_reserved;
        r.cell.input_phase.admitted = true;
        r.cell.input_phase.work_units = setup;
        r.cell.input_phase.admitted_bytes = q.bytes;
        r.cell.input_phase.stop_reason = "capture-refused";
        c.phase_cube.admission.as_mut()?.quote = cube;
        q
    };
    // No ledger guard survives into the captured-account lock. Paid debt
    // persists on every refusal; a missing target never activates candidates.
    let input = capture_target(cuts, b, paid.d)?;
    let mut c = b.state.lock().ok()?;
    let a = c.phase_cube.admission.as_ref()?;
    if !owned(a, b)
        || a.deadline != cuts.deadline
        || a.quote.cell.and_then(|q| q.input_phase) != Some(paid)
        || a.input_target.is_some()
        || !c.phase_cube.input_phase_attempted
        || c.phase_cube.cell_attempted
        || c.phase_cube.finish_attempted
        || c.phase_cube.prepare_attempted
        || c.replay_attempts != 0
        || c.source_replays != 0
        || !live(b.search_end)
    {
        return None;
    }
    c.phase_cube.admission.as_mut()?.input_target = Some(input);
    c.component_rows.phase.cube.cell.input_phase.stop_reason = "reserved";
    Some(())
}
