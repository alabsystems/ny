// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! All eligible original selectors, from the SAME completed winner replay.
use super::*;

pub(super) fn quote(cuts: &HullRootMipCuts<'_>, plan: &Plan) -> Option<AllQuote> {
    let p = cuts.problem;
    let components = plan.components.len();
    if components == 0 || components > p.hidden_dim {
        return None;
    }
    let negatives = p.c.iter().filter(|v| **v < 0.).count();
    if negatives < components {
        return None;
    }
    let mut visits = 0usize;
    for c in &plan.components {
        let k = c.negative.len();
        if !(1..=8).contains(&k) {
            return None;
        }
        visits = visits.checked_add(k.checked_mul(1usize.checked_shl(k as u32)?)?)?;
    }
    // Future checked owners can change positive ownership. Never quote only
    // this seed's p_i: every valid component has p_i<=16 and k_i<=8.
    let terms = negatives.checked_mul(p.input_dim.checked_add(25)?)?;
    let rows = cuts
        .len()
        .checked_add(1)?
        .checked_add(components.checked_mul(2)?)?
        .checked_add(negatives)?;
    let original_rows = p
        .hidden_dim
        .checked_add(p.input_dim)?
        .checked_add(1)?
        .checked_mul(8)?;
    let span = terms
        .checked_add(negatives.checked_mul(2)?)?
        .checked_add(p.hidden_dim)?
        .checked_add(p.input_dim)?
        .checked_add(rows)?
        .checked_add(1)?;
    Some(AllQuote {
        components,
        negatives,
        visits,
        terms,
        rows,
        original_rows,
        work: visits
            .checked_mul(32)?
            .checked_add(span.checked_mul(128)?)?
            .checked_add(original_rows.checked_mul(128)?)?
            .checked_add(4096)?,
        bytes: terms
            .checked_mul(64)?
            .checked_add(negatives.checked_mul(256)?)?
            .checked_add(negatives.checked_mul(64)?)?
            .checked_add(rows.checked_mul(256)?)?
            .checked_add(original_rows.checked_mul(128)?)?
            .checked_add(4096)?,
    })
}

/// Parent admission paid setup before this bounded metadata walk. An All
/// refusal changes no hold: the parent may take its existing One quote.
pub(super) fn admit(cuts: &HullRootMipCuts<'_>, plan: &Plan, b: &Budget, setup: usize) -> bool {
    {
        let Ok(mut c) = b.state.lock() else {
            return false;
        };
        if !b.phase_all_enabled
            || c.phase_all.admission_attempted
            || !c.component_rows.phase.attempted
            || c.component_rows.phase.work_units != setup
            || c.replay_attempts != 0
        {
            return false;
        }
        c.phase_all.admission_attempted = true;
    }
    let Some(mut q) = quote(cuts, plan) else {
        return false;
    };
    let Some(held) = q.work.checked_sub(setup) else {
        return false;
    };
    if b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.authority > cuts.deadline
        || b.dimensions
            != (
                cuts.problem.input_dim,
                cuts.problem.hidden_dim,
                cuts.problem.c.iter().filter(|v| **v > 0.).count(),
            )
    {
        return false;
    }
    // This setup debt is distinct from already banked LP activity. A refused
    // combination still selects current All below; no once latch is reset.
    let cube = cube::offer(cuts, plan, b);
    let cell = cube.and_then(|extra| {
        let protected = held.checked_add(extra.work.checked_sub(extra.setup)?)?;
        cube::cell::offer(cuts, plan, b, protected)
    });
    let Ok(mut c) = b.state.lock() else {
        return false;
    };
    let Some(base_bytes) = b
        .admitted_bytes
        .checked_add(c.shared_bytes)
        .and_then(|v| v.checked_add(c.direction.cache_bytes))
        .and_then(|v| v.checked_add(c.child.cache_bytes))
        .and_then(|v| v.checked_add(c.component_rows.admitted_bytes))
    else {
        return false;
    };
    let mut next = c.component_reserved.checked_add(held);
    let mut bytes = q.bytes;
    let mut funded_cube = cube.filter(|extra| {
        let work = extra
            .work
            .checked_sub(extra.setup)
            .and_then(|v| next?.checked_add(v))
            .and_then(|v| c.work.checked_add(c.reserved)?.checked_add(v));
        let memory = base_bytes
            .checked_add(q.bytes)
            .and_then(|v| v.checked_add(extra.bytes));
        work.is_some_and(|v| v <= MAX_WORK)
            && memory.is_some_and(|v| v <= MAX_BYTES)
            && live(b.search_end)
    });
    // Preserve the base Cube admission; only its independently funded extra
    // table/row may consume this additional first-publication hold.
    if let (Some(extra), Some(cell)) = (&mut funded_cube, cell) {
        let upgraded = (|| {
            let held = extra.work.checked_sub(extra.setup)?;
            let work = next?
                .checked_add(held)?
                .checked_add(cell.work.checked_sub(cell.setup)?)?
                .checked_add(c.work)?
                .checked_add(c.reserved)?;
            let memory = base_bytes
                .checked_add(q.bytes)?
                .checked_add(extra.bytes)?
                .checked_add(cell.bytes)?;
            if work > MAX_WORK || memory > MAX_BYTES || !live(b.search_end) {
                return None;
            }
            let mut upgraded = *extra;
            upgraded.work = extra.work.checked_add(cell.work)?;
            upgraded.setup = extra.setup.checked_add(cell.setup)?;
            upgraded.bytes = extra.bytes.checked_add(cell.bytes)?;
            upgraded.terms = extra.terms.checked_add(cell.terms)?;
            upgraded.rows = extra.rows.checked_add(1)?;
            upgraded.cell = Some(cell);
            Some(upgraded)
        })();
        if let Some(upgraded) = upgraded {
            *extra = upgraded;
        }
    }
    // The new ray cannot displace either the old Cube or its old Cell row.
    // Quote only fixed arithmetic here; all table/optimizer work follows payment.
    if let Some(extra) = &mut funded_cube {
        let upgraded = (|| {
            let mut cell = extra.cell?;
            let ray = cube::cell::ray::quote(b.dimensions.0, b.dimensions.1, cell.k, cell.terms)?;
            let work = next?
                .checked_add(extra.work.checked_sub(extra.setup)?)?
                .checked_add(ray.work)?
                .checked_add(c.work)?
                .checked_add(c.reserved)?;
            let memory = base_bytes
                .checked_add(q.bytes)?
                .checked_add(extra.bytes)?
                .checked_add(ray.bytes)?;
            if work > MAX_WORK || memory > MAX_BYTES || !live(b.search_end) {
                return None;
            }
            let mut upgraded = *extra;
            upgraded.work = extra.work.checked_add(ray.work)?;
            upgraded.bytes = extra.bytes.checked_add(ray.bytes)?;
            upgraded.terms = extra.terms.checked_add(cell.terms)?;
            upgraded.rows = extra.rows.checked_add(1)?;
            cell.ray = Some(ray);
            upgraded.cell = Some(cell);
            Some(upgraded)
        })();
        if let Some(upgraded) = upgraded {
            *extra = upgraded;
        }
    }
    // A third, separately prepaid support family retains both earlier rows.
    // The table LP is local advice, distinct from the original root proposal LP.
    if let Some(extra) = &mut funded_cube {
        let upgraded = (|| {
            let mut cell = extra.cell?;
            cell.ray?;
            let lp = cube::cell::lp::quote(b.dimensions.0, b.dimensions.1, cell.k, cell.terms)?;
            let work = next?
                .checked_add(extra.work.checked_sub(extra.setup)?)?
                .checked_add(lp.work)?
                .checked_add(c.work)?
                .checked_add(c.reserved)?;
            let memory = base_bytes
                .checked_add(q.bytes)?
                .checked_add(extra.bytes)?
                .checked_add(lp.bytes)?;
            if work > MAX_WORK || memory > MAX_BYTES || !live(b.search_end) {
                return None;
            }
            let mut upgraded = *extra;
            upgraded.work = extra.work.checked_add(lp.work)?;
            upgraded.bytes = extra.bytes.checked_add(lp.bytes)?;
            upgraded.terms = extra.terms.checked_add(cell.terms)?;
            upgraded.rows = extra.rows.checked_add(1)?;
            cell.lp = Some(lp);
            upgraded.cell = Some(cell);
            Some(upgraded)
        })();
        if let Some(upgraded) = upgraded {
            *extra = upgraded;
        }
    }
    // Joint adds a separately checked closed-cell table and keeps all old rows.
    // Ray funds the shared sequential numeric/support workspace.
    if let Some(extra) = &mut funded_cube {
        let upgraded = (|| {
            let mut cell = extra.cell?;
            cell.ray?;
            let joint =
                cube::cell::joint::quote(b.dimensions.0, b.dimensions.1, cell.k, cell.terms)?;
            let work = next?
                .checked_add(extra.work.checked_sub(extra.setup)?)?
                .checked_add(joint.work)?
                .checked_add(c.work)?
                .checked_add(c.reserved)?;
            let memory = base_bytes
                .checked_add(q.bytes)?
                .checked_add(extra.bytes)?
                .checked_add(joint.bytes)?;
            if work > MAX_WORK || memory > MAX_BYTES || !live(b.search_end) {
                return None;
            }
            let mut upgraded = *extra;
            upgraded.work = extra.work.checked_add(joint.work)?;
            upgraded.bytes = extra.bytes.checked_add(joint.bytes)?;
            upgraded.terms = extra.terms.checked_add(cell.terms)?;
            upgraded.rows = extra.rows.checked_add(1)?;
            cell.joint = Some(joint);
            upgraded.cell = Some(cell);
            Some(upgraded)
        })();
        if let Some(upgraded) = upgraded {
            *extra = upgraded;
        }
    }
    if let Some(extra) = funded_cube {
        next = extra
            .work
            .checked_sub(extra.setup)
            .and_then(|v| next?.checked_add(v));
        let Some(total_bytes) = bytes.checked_add(extra.bytes) else {
            return false;
        };
        bytes = total_bytes;
        let Some(rows) = q.rows.checked_add(extra.rows) else {
            return false;
        };
        q.rows = rows;
    }
    let total = base_bytes.checked_add(bytes);
    let funded = next.and_then(|v| c.work.checked_add(c.reserved)?.checked_add(v));
    if total.is_none_or(|v| v > MAX_BYTES)
        || funded.is_none_or(|v| v > MAX_WORK)
        || !live(b.search_end)
    {
        cube::refused(&mut c);
        return false;
    }
    let Some(admitted_bytes) = c.component_rows.admitted_bytes.checked_add(bytes) else {
        return false;
    };
    c.component_reserved = next.unwrap();
    c.component_rows.reserved_work_units = c.component_reserved;
    c.component_rows.admitted_bytes = admitted_bytes;
    if let Some(extra) = funded_cube {
        cube::activate(&mut c, cuts, b, extra);
    } else {
        cube::refused(&mut c);
    }
    let r = &mut c.component_rows.phase;
    r.admitted = true;
    r.all_selector_mode = true;
    r.original_ir_row_limit = q.original_rows;
    r.quoted_rows = q
        .negatives
        .checked_add(funded_cube.map_or(0, |x| x.rows))
        .unwrap();
    r.quoted_terms = q
        .terms
        .checked_add(funded_cube.map_or(0, |x| x.terms))
        .unwrap();
    r.admitted_bytes = bytes;
    r.reserved_work_units = held
        .checked_add(funded_cube.map_or(0, |x| x.work - x.setup))
        .unwrap();
    r.stop_reason = "all-reserved";
    c.phase_all.admission = Some(AllAdmission {
        quote: q,
        source: b.source,
        dimensions: b.dimensions,
        authority: b.authority,
        search_end: b.search_end,
        deadline: cuts.deadline,
    });
    let transfer = funded_cube.is_some();
    drop(c);
    if transfer {
        cube::transfer_activity(cuts, b);
    }
    true
}

pub(super) fn owned(a: &AllAdmission, b: &Budget) -> bool {
    a.source == b.source
        && a.dimensions == b.dimensions
        && a.authority == b.authority
        && a.search_end == b.search_end
        && a.authority <= a.deadline
}

/// Check the future owner's cheap shape BEFORE optional pair accumulation.
/// No head/alpha/source-coordinate walk and no new Source invocation.
pub(super) fn collection_shape(b: &Budget, plan: &Plan) -> bool {
    let Ok(c) = b.state.lock() else {
        return false;
    };
    let Some(a) = c.phase_all.admission.as_ref() else {
        return false;
    };
    if !owned(a, b) || plan.components.len() > a.quote.components {
        return false;
    }
    let (mut n, mut v, mut t) = (0usize, 0usize, 0usize);
    for item in &plan.components {
        let k = item.negative.len();
        let p = item.positive.len();
        if !(1..=8).contains(&k) || p > MAX_POSITIVE {
            return false;
        }
        let Some(next) = n.checked_add(k) else {
            return false;
        };
        n = next;
        let Some(next) = k.checked_mul(1usize << k).and_then(|w| v.checked_add(w)) else {
            return false;
        };
        v = next;
        let Some(next) = b
            .dimensions
            .0
            .checked_add(p)
            .and_then(|w| w.checked_add(k))
            .and_then(|w| w.checked_add(1))
            .and_then(|w| w.checked_mul(k))
            .and_then(|w| t.checked_add(w))
        else {
            return false;
        };
        t = next;
    }
    n <= a.quote.negatives && v <= a.quote.visits && t <= a.quote.terms
}

/// The old component rows must be the same original Source planes; this also
/// bounds private callers before cloning any caller-provided coefficient Vec.
pub(super) fn base_valid(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    index: usize,
    base: &RowSpec,
    minimum: f64,
) -> Option<()> {
    let c = row.plan.components.get(index)?;
    let lower = finite(2. * minimum)?;
    if base.lb != lower
        || base.ub != f64::INFINITY
        || base.coeffs.len()
            > c.negative
                .len()
                .checked_add(c.positive.len())?
                .checked_add(c.g.len())?
    {
        return None;
    }
    let mut actual = base.coeffs.iter();
    for &j in &c.negative {
        let expected = (binding.head_column(j)?.0, finite(2. * cuts.problem.c[j])?);
        if actual.next() != Some(&expected) {
            return None;
        }
    }
    for (slot, &j) in c.positive.iter().enumerate() {
        let expected = (
            binding.head_column(j)?.0,
            finite(cuts.problem.c[j] * f64::from(*c.positive_units.get(slot)?))?,
        );
        if actual.next() != Some(&expected) {
            return None;
        }
    }
    for (i, &g) in c.g.iter().enumerate() {
        let value = finite(-2. * g)?;
        if value != 0. {
            let expected = (parts.input_vars.get(i)?.0, value);
            if actual.next() != Some(&expected) {
                return None;
            }
        }
    }
    actual.next().is_none().then_some(())
}

pub(super) fn prepare(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    conditional: &Conditional,
    minima: &[f64],
    old: &[RowSpec],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    let b = &row.budget;
    if !binding.matches(cuts, parts) || !live(end) || !poll() {
        return None;
    }
    let q = {
        let mut c = b.state.lock().ok()?;
        let a = c.phase_all.admission.as_ref()?;
        if !owned(a, b)
            || a.source != row.source
            || a.deadline != cuts.deadline
            || c.phase_all.prepare_attempted
        {
            return Some(Vec::new());
        }
        let q = a.quote;
        c.phase_all.prepare_attempted = true;
        q
    };
    // The one-use context latch precedes every O(N+C) integrity/shape walk.
    if !conditional.valid(row, minima)
        || old.len() != row.plan.components.len()
        || !collection_shape(b, &row.plan)
    {
        stop(b, "all-coverage");
        return Some(Vec::new());
    }
    if parts.problem.num_rows() > q.original_rows {
        stop(b, "all-original-ir-limit");
        return Some(Vec::new());
    }
    let mut rows = Vec::with_capacity(q.negatives);
    let (mut terms, mut heads) = (0usize, 0usize);
    let native_component = native_publication::selected(row);
    let mut native_origin = None;
    for (index, component) in row.plan.components.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        if base_valid(row, cuts, parts, binding, index, &old[index], minima[index]).is_none() {
            stop(b, "all-base-row");
            return Some(Vec::new());
        }
        for (bit, &head) in component.negative.iter().enumerate() {
            if !live(end) || !poll() {
                return None;
            }
            let Some(column) = binding.column(head) else {
                continue;
            };
            let Some((delta, coefficient, lower)) =
                plane(minima[index], conditional.values[index][bit])
            else {
                stop(b, "all-overflow");
                return Some(Vec::new());
            };
            if delta <= 0. {
                continue;
            }
            let mut coeffs = Vec::with_capacity(old[index].coeffs.len().checked_add(1)?);
            coeffs.extend_from_slice(&old[index].coeffs);
            coeffs.push((column.0, coefficient));
            terms = terms.checked_add(coeffs.len())?;
            heads = heads
                .checked_add(component.positive.len())?
                .checked_add(component.negative.len())?;
            if terms > q.terms || rows.len() >= q.negatives {
                stop(b, "all-shape");
                return Some(Vec::new());
            }
            rows.push(RowSpec {
                lb: lower,
                ub: f64::INFINITY,
                coeffs,
            });
            // Exact origin at the producer; flat output positions do not
            // identify component/bit after skipped rows.
            if native_component == Some(index) && native_origin.is_none() {
                native_origin = Some((index, head, rows.len() - 1));
            }
        }
    }
    require_hull_environment().ok()?;
    if !live(end) || !poll() {
        return None;
    }
    let mut c = b.state.lock().ok()?;
    c.phase_all.pending = !rows.is_empty();
    c.phase_all.parts = std::ptr::from_ref(parts) as usize;
    c.phase_all.original_rows = parts.problem.num_rows();
    c.phase_all.prepared_rows = rows.len();
    c.phase_all.witness = Some(row.witness.clone());
    let r = &mut c.component_rows.phase;
    r.prepared_rows = rows.len();
    r.prepared_head_terms = heads;
    r.prepared_terms = terms;
    r.stop_reason = if rows.is_empty() {
        "all-no-gap"
    } else {
        "all-prepared"
    };
    drop(c);
    if let Some((component, head, position)) = native_origin {
        let mut selector_heads = [0; 8];
        selector_heads[0] = head;
        native_publication::stage(
            row,
            cuts,
            parts,
            binding,
            NativeRow {
                component,
                cube: false,
                extra: false,
                selector_heads,
                selector_count: 1,
                stored: &rows[position],
                stored_g: None,
            },
            poll,
        )?;
    } else if let Some(component) = native_component {
        // Every old row passed base_valid above, and the whole All family
        // reached its final barrier. Reuse this actually appended component
        // row only when the selected component emitted no unary lift. This
        // spends the same first-row attempt, with fewer terms; it never
        // replaces a captured row or retries a refused capture.
        native_publication::stage(
            row,
            cuts,
            parts,
            binding,
            NativeRow {
                component,
                cube: false,
                extra: false,
                selector_heads: [0; 8],
                selector_count: 0,
                stored: &old[component],
                stored_g: None,
            },
            poll,
        )?;
    }
    Some(rows)
}

/// Consume before allocation; Signed validates any plain-portfolio capability
/// next, then ONE exact reservation owns the complete no-failure commit batch.
pub(super) fn claim_ir_headers(
    row: &SignedMajorant,
    parts: &MipParts,
    additional: usize,
) -> HeaderClaim {
    let b = &row.budget;
    let mut c = b.state.lock().map_err(|_| HeaderRefused)?;
    if !c.phase_all.pending {
        return Ok(None);
    }
    let a = c.phase_all.admission.as_ref().ok_or(HeaderRefused)?;
    let witness = c.phase_all.witness.as_ref().ok_or(HeaderRefused)?;
    if !owned(a, b)
        || a.source != row.source
        || !live(a.authority)
        || !Arc::ptr_eq(witness, &row.witness)
        || c.phase_all.claimed
        || c.phase_all.parts != std::ptr::from_ref(parts) as usize
        || c.phase_all.original_rows != parts.problem.num_rows()
        || parts.problem.num_rows() > a.quote.original_rows
        || additional > a.quote.rows
        || additional
            < c.phase_all
                .prepared_rows
                .checked_add(cube::header_rows(&c).ok_or(HeaderRefused)?)
                .ok_or(HeaderRefused)?
        || c.component_rows.phase.prepared_rows
            != c.phase_all
                .prepared_rows
                .checked_add(cube::header_rows(&c).ok_or(HeaderRefused)?)
                .ok_or(HeaderRefused)?
    {
        return Err(HeaderRefused);
    }
    let grant = HeaderGrant {
        source: a.source,
        ledger: b.state.clone(),
        witness: witness.clone(),
        authority: a.authority,
        search_end: a.search_end,
        parts: c.phase_all.parts,
        original_rows: c.phase_all.original_rows,
        additional,
        total_bound: a.quote.rows,
    };
    c.phase_all.claimed = true;
    Ok(Some(grant))
}
