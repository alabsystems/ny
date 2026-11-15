// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Finite component planes from the SAME first complete publication replay.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SignedComponentRowsReport {
    pub attempted: bool,
    pub admitted: bool,
    pub quoted_components: usize,
    pub quoted_terms: usize,
    pub reserved_work_units: usize,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub collection_attempts: usize,
    pub collected_components: usize,
    pub prepared_rows: usize,
    pub prepared_head_terms: usize,
    pub prepared_terms: usize,
    pub appended_rows: usize,
    pub appended_terms: usize,
    pub stop_reason: &'static str,
    pub phase: SignedPhaseRowsReport,
    pub joint: SignedJointRowsReport,
    pub domination: SignedDominationReport,
}
impl Default for SignedComponentRowsReport {
    fn default() -> Self {
        Self {
            attempted: false,
            admitted: false,
            quoted_components: 0,
            quoted_terms: 0,
            reserved_work_units: 0,
            work_units: 0,
            admitted_bytes: 0,
            collection_attempts: 0,
            collected_components: 0,
            prepared_rows: 0,
            prepared_head_terms: 0,
            prepared_terms: 0,
            appended_rows: 0,
            appended_terms: 0,
            stop_reason: "not-attempted",
            phase: SignedPhaseRowsReport::default(),
            joint: SignedJointRowsReport::default(),
            domination: SignedDominationReport::default(),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub(super) struct Quote {
    pub components: usize,
    pub terms: usize,
    pub work: usize,
    pub bytes: usize,
}
fn stop(b: &Budget, reason: &'static str) {
    if let Ok(mut c) = b.state.lock() {
        c.component_rows.stop_reason = reason;
    }
}
/// Whole and half appearances together are at most twice the positive heads.
/// Negative support and component count never change during bounded refinement.
pub(super) fn quote(cuts: &HullRootMipCuts<'_>, plan: &Plan) -> Option<Quote> {
    let p = cuts.problem;
    let components = plan.components.len();
    if components == 0 || components > p.hidden_dim || p.c.len() != p.hidden_dim {
        return None;
    }
    let positive = p.c.iter().filter(|c| **c > 0.).count();
    let negative = p.c.iter().filter(|c| **c < 0.).count();
    let terms = components
        .checked_mul(p.input_dim)?
        .checked_add(positive.checked_mul(2)?)?
        .checked_add(negative)?;
    let span = terms
        .checked_add(components)?
        .checked_add(p.hidden_dim)?
        .checked_add(p.input_dim)?
        .checked_add(1)?;
    Some(Quote {
        components,
        terms,
        work: span.checked_mul(128)?,
        // Minima, row headers, BOTH prepared/IR coefficient buffers, and
        // bounded index/column validation scratch. Nominal payload, not RSS.
        bytes: terms
            .checked_mul(64)?
            .checked_add(components.checked_mul(256)?)?
            .checked_add(
                p.hidden_dim
                    .checked_add(p.input_dim)?
                    .checked_add(1)?
                    .checked_mul(64)?,
            )?
            .checked_add(4096)?,
    })
}
/// Reserve optional construction AFTER a whole baseline is checked and BEFORE
/// optional advice. This hold is separate from the two publication-pass holds.
pub(super) fn admit(cuts: &HullRootMipCuts<'_>, plan: &Plan, b: &Budget) -> Option<()> {
    let p = cuts.problem;
    if b.source != std::ptr::from_ref(p) as usize
        || b.dimensions.0 != p.input_dim
        || b.dimensions.1 != p.hidden_dim
        || p.c.len() != p.hidden_dim
        || plan.components.len() > p.hidden_dim
        || !capacity(p)
    {
        return None;
    }
    {
        let mut c = b.state.lock().ok()?;
        if c.component_rows.attempted || c.replay_attempts != 0 {
            return None;
        }
        c.component_rows.attempted = true;
        c.component_rows.stop_reason = "quote";
    }
    // Pay all metadata/count scans even if the optional larger hold refuses.
    let setup = p
        .hidden_dim
        .checked_add(p.input_dim)?
        .checked_add(plan.components.len())?
        .checked_add(1)?
        .checked_mul(32)?;
    if b.charge(setup).is_none() {
        stop(b, "setup-work");
        return None;
    }
    b.state.lock().ok()?.component_rows.work_units = setup;
    let Some(q) = quote(cuts, plan) else {
        stop(b, "shape");
        return None;
    };
    let mut c = b.state.lock().ok()?;
    c.component_rows.quoted_components = q.components;
    c.component_rows.quoted_terms = q.terms;
    let total = b
        .admitted_bytes
        .checked_add(c.shared_bytes)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.child.cache_bytes)?
        .checked_add(q.bytes)?;
    if total > MAX_BYTES {
        c.component_rows.stop_reason = "payload";
        return None;
    }
    if c.work.checked_add(c.reserved)?.checked_add(q.work)? > MAX_WORK {
        c.component_rows.stop_reason = "work";
        return None;
    }
    if !live(b.search_end) {
        c.component_rows.stop_reason = "deadline";
        return None;
    }
    c.component_reserved = q.work;
    c.component_rows.reserved_work_units = q.work;
    c.component_rows.admitted_bytes = q.bytes;
    c.component_rows.admitted = true;
    c.component_rows.stop_reason = "reserved";
    drop(c);
    phase::admit(cuts, plan, b);
    Some(())
}
/// Spend the one optional hold under the ORIGINAL publication clock. No new
/// check/replay/child-call slot, refund or quota reset is introduced.
pub(super) fn begin(b: &Budget, end: Instant) -> bool {
    let Ok(mut c) = b.state.lock() else {
        return false;
    };
    if !c.component_rows.admitted {
        return false;
    }
    if c.replay_attempts != 1 || c.source_replays != 0 || c.component_rows.collection_attempts != 0
    {
        c.component_rows.stop_reason = "publication-order";
        return false;
    }
    c.component_rows.collection_attempts += 1;
    if !live(end.min(b.authority)) {
        c.component_rows.stop_reason = "deadline";
        return false;
    }
    let work = c.component_reserved;
    let Some(next) = c.work.checked_add(work) else {
        return false;
    };
    if work == 0 || next.checked_add(c.reserved).is_none_or(|v| v > MAX_WORK) {
        c.component_rows.stop_reason = "work";
        return false;
    }
    let Some(total) = c.component_rows.work_units.checked_add(work) else {
        return false;
    };
    c.work = next;
    c.component_reserved = 0;
    c.component_rows.reserved_work_units = 0;
    c.component_rows.work_units = total;
    c.component_rows.stop_reason = "collecting";
    phase::begin(&mut c).is_some()
}
/// Revoke publication authority while preserving completed-work telemetry.
/// A subsequent preparation clears that telemetry before constructing rows.
pub(super) fn abort_prepared(b: &Budget) {
    native_publication::reset(b);
    portfolio::abort_prepared(b);
    phase::reset(b);
}
pub(super) fn reset_prepared(b: &Budget) {
    native_publication::reset(b);
    portfolio::reset_prepared(b);
    phase::reset(b);
    joint::reset(b);
    domination::reset(b);
    if let Ok(mut c) = b.state.lock() {
        c.component_rows.prepared_rows = 0;
        c.component_rows.prepared_head_terms = 0;
        c.component_rows.prepared_terms = 0;
    }
}
#[derive(Debug)]
pub(super) enum Prepared {
    Rows(Vec<RowSpec>),
    Refused,
}
fn overflow(b: &Budget) -> Option<Prepared> {
    stop(b, "doubling-overflow");
    Some(Prepared::Refused)
}
/// Private data comes only from the complete sealed first replay. The outer
/// binding has already re-encoded and compared the ENTIRE original model.
pub(super) fn prepare(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    columns: &[crate::ir::Col],
    minima: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Prepared> {
    let b = &row.budget;
    let end = end.min(b.authority).min(cuts.deadline);
    let p = cuts.problem;
    require_hull_environment().ok()?;
    if row.source != std::ptr::from_ref(p) as usize
        || minima.len() != row.plan.components.len()
        || minima.iter().any(|v| !v.is_finite())
        || columns.len() != p.hidden_dim
        || parts.input_vars.len() != p.input_dim
        || parts.num_cols != parts.problem.num_cols()
        || !live(end)
        || !poll()
    {
        return None;
    }
    let (max_components, max_terms) = {
        let c = b.state.lock().ok()?;
        if !c.component_rows.admitted
            || c.component_rows.collection_attempts != 1
            || c.component_rows.collected_components != minima.len()
        {
            return None;
        }
        (
            c.component_rows.quoted_components,
            c.component_rows.quoted_terms,
        )
    };
    if minima.len() > max_components {
        return None;
    }
    // Strict whole-model binding supplies these exact post-ReLU columns.
    // Stable-active heads may alias FREE affine columns. Scaling the WHOLE
    // row by2 avoids coefficient rounding and needs no column-sign assumption.
    for col in columns {
        if !live(end) || !poll() {
            return None;
        }
        parts.problem.cols().get(col.0)?;
    }
    let mut rows = Vec::with_capacity(minima.len());
    let (mut terms, mut heads) = (0usize, 0usize);
    for (component, &minimum) in row.plan.components.iter().zip(minima) {
        if !live(end)
            || !poll()
            || component.g.len() != p.input_dim
            || component.positive_units.len() != component.positive.len()
        {
            return None;
        }
        let cap = component
            .negative
            .len()
            .checked_add(component.positive.len())?
            .checked_add(p.input_dim)?;
        let mut coeffs = Vec::with_capacity(cap);
        for &j in &component.negative {
            let c = *p.c.get(j)?;
            if c.partial_cmp(&0.) != Some(std::cmp::Ordering::Less) || !c.is_finite() {
                return None;
            }
            let value = 2. * c;
            if !value.is_finite() {
                return overflow(b);
            }
            coeffs.push((columns.get(j)?.0, value));
            heads = heads.checked_add(1)?;
        }
        for (slot, &j) in component.positive.iter().enumerate() {
            let c = *p.c.get(j)?;
            if !c.is_finite() || c.partial_cmp(&0.) != Some(std::cmp::Ordering::Greater) {
                return None;
            }
            let value = match component.positive_units[slot] {
                1 => c,
                2 => 2. * c,
                _ => return None,
            };
            if !value.is_finite() {
                return overflow(b);
            }
            coeffs.push((columns.get(j)?.0, value));
            heads = heads.checked_add(1)?;
        }
        for (i, &g) in component.g.iter().enumerate() {
            if !live(end) || !poll() || !g.is_finite() {
                return None;
            }
            let value = -2. * g;
            if !value.is_finite() {
                return overflow(b);
            }
            if value != 0. {
                let col = parts.input_vars.get(i)?;
                parts.problem.cols().get(col.0)?;
                coeffs.push((col.0, value));
            }
        }
        terms = terms.checked_add(coeffs.len())?;
        if terms > max_terms {
            return None;
        }
        let lower = 2. * minimum;
        if !lower.is_finite() {
            return overflow(b);
        }
        rows.push(RowSpec {
            lb: lower,
            ub: f64::INFINITY,
            coeffs,
        });
    }
    require_hull_environment().ok()?;
    if !live(end) || !poll() {
        return None;
    }
    let mut c = b.state.lock().ok()?;
    c.component_rows.prepared_rows = rows.len();
    c.component_rows.prepared_head_terms = heads;
    c.component_rows.prepared_terms = terms;
    c.component_rows.stop_reason = "prepared";
    Some(Prepared::Rows(rows))
}
pub(super) fn commit(b: &Budget) {
    portfolio::commit(b);
    if let Ok(mut c) = b.state.lock() {
        joint::commit(&mut c);
        phase::cube::commit(&mut c);
        domination::commit(&mut c);
        c.component_rows.appended_rows = c.component_rows.prepared_rows;
        c.component_rows.appended_terms = c.component_rows.prepared_terms;
        let phase = &mut c.component_rows.phase;
        phase.appended_rows = phase.prepared_rows;
        phase.appended_terms = phase.prepared_terms;
        if phase.appended_rows > 0 {
            phase.stop_reason = "appended";
        }
        if c.component_rows.appended_rows > 0 {
            c.component_rows.stop_reason = "appended";
        }
    }
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_component_integration_tests.rs"]
mod integration_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_component_rows_tests.rs"]
mod tests;
