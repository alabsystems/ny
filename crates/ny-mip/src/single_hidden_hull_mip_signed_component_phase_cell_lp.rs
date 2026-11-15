// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! A separately prepaid table LP; complete directed repair remains authoritative.
use super::*;
use certificate::I;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_simplex.rs"]
pub(super) mod simplex;
pub(in crate::single_hidden_hull::mip_cuts::signed) const PIVOTS: usize = 32;

#[derive(Debug, Clone, Copy)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Quote {
    pub work: usize,
    pub bytes: usize,
}
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SignedTableSupportReport {
    pub admitted: bool,
    pub activity_from_root: bool,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub calls: usize,
    pub pivots: usize,
    pub pivot_limit: usize,
    pub stop_reason: &'static str,
    pub checked_masks: usize,
    pub score_lower: Option<f64>,
    pub envelope_upper: Option<f64>,
    pub gap_upper: Option<f64>,
    pub prepared_rows: usize,
    pub prepared_terms: usize,
    pub prepared_selectors: usize,
    pub appended_rows: usize,
}
#[derive(Clone, Copy)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Support {
    pub a: [f64; 8],
    pub lower: f64,
}

pub(in crate::single_hidden_hull::mip_cuts::signed) fn quote(
    d: usize,
    h: usize,
    k: usize,
    terms: usize,
) -> Option<Quote> {
    if !(1..=32).contains(&d) || !(1..=8).contains(&k) || size_of::<simplex::Simplex>() > 4096 {
        return None;
    }
    let (m, r) = (1usize << k, k + 1);
    let each = 32usize
        .checked_mul(m)?
        .checked_add(128usize.checked_mul(r * r)?)?
        .checked_add(512usize.checked_mul(r)?)?
        .checked_add(4096)?;
    // P+1 complete evaluations (including final extraction), P updates,
    // projection, initialization, exact-normalization gap, two repair walks,
    // old/new score checks, row construction and three-family deduplication.
    let work = 4096usize
        .checked_add(256usize.checked_mul(r * r)?)?
        .checked_add(128usize.checked_mul(m)?.checked_mul(r)?)?
        .checked_add((PIVOTS + 1).checked_mul(each)?)?
        .checked_add(
            256usize.checked_mul(
                terms
                    .checked_add(2 * k)?
                    .checked_add(h)?
                    .checked_add(d)?
                    .checked_add(2)?,
            )?,
        )?;
    // Two actual-size simplex slots cover construction/return overlap. Gray
    // pricing has no table scratch. The 4KiB auxiliary allowance covers three
    // 9-double vectors (216B), basic flags (256B), order/best/midpoint (192B),
    // interval marginals (128B), columns/mapping (136B), three Support copies
    // (216B), three reports, and >2KiB for scalar/interval/closure temporaries.
    // Row storage and its publication copies are charged separately below.
    if 1144usize.checked_add(3usize.checked_mul(size_of::<SignedTableSupportReport>())?)? > 2048 {
        return None;
    }
    Some(Quote {
        work,
        bytes: 2usize
            .checked_mul(size_of::<simplex::Simplex>())?
            .checked_add(4096)?
            .checked_add(64usize.checked_mul(terms)?)?,
    })
}

pub(super) fn upper(s: &simplex::Simplex, poll: &mut dyn FnMut() -> bool) -> Option<f64> {
    require_hull_environment().ok()?;
    const Q: u64 = 1 << 48;
    let mut remaining = Q;
    let (mut cost, mut marginal) = (I::zero(), [I::zero(); 8]);
    for i in 0..=s.k + 1 {
        if !poll() {
            return None;
        }
        let (mask, quanta) = if i <= s.k {
            let proposed = finite(s.mass[i].max(0.) * Q as f64)?.floor() as u64;
            (s.basis[i], proposed.min(remaining))
        } else {
            (0, remaining)
        };
        remaining = remaining.checked_sub(quanta)?;
        // Integer conservation gives sum(lambda)=1 exactly; the floating
        // simplex masses are merely proposals. Every quantum is dyadic.
        let weight = quanta as f64 * (1. / Q as f64);
        cost = cost.add(I::point(s.values[mask])?.scale(2.)?.scale(weight)?)?;
        for j in 0..s.k {
            if mask & (1 << j) != 0 {
                marginal[j] = marginal[j].add(I::point(weight)?)?;
            }
        }
    }
    if remaining != 0 {
        return None;
    }
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &s.values[..1 << s.k] {
        if !poll() {
            return None;
        }
        low = low.min(v);
        high = high.max(v);
    }
    let range = I::point(high)?.sub(I::point(low)?)?.scale(2.)?.hi;
    let mut error = I::zero();
    for j in 0..s.k {
        let delta = I::point(s.activity[j])?.sub(marginal[j])?;
        error = error.add(I::point(delta.lo.abs().max(delta.hi.abs()))?)?;
    }
    // Correct one marginal by transferring mass along that bit's cube edges.
    // Other marginals stay fixed; each transfer costs at most range*mass.
    // Therefore C + range*||s-y||_1 bounds THIS table envelope from above.
    let result = cost.add(error.scale(range)?)?.hi;
    require_hull_environment().ok()?;
    Some(result)
}

pub(super) fn choose(
    values: &[f64],
    mapped: &[bool],
    activity: &[f64],
    incumbent: Support,
    r: &mut SignedTableSupportReport,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CellAttempt<Support>> {
    r.calls = 1;
    r.pivot_limit = PIVOTS;
    r.stop_reason = "numerical-refusal";
    let mut cancelled = false;
    let mut sticky = || {
        if cancelled {
            return false;
        }
        let ok = live(end) && poll();
        cancelled |= !ok;
        ok
    };
    let result = (|| {
        let mut s = simplex::Simplex::new(values, mapped, activity, &mut sticky)?;
        let a = s.run(PIVOTS, r, &mut sticky)?;
        let solver_stop = r.stop_reason;
        r.stop_reason = "repair-refusal";
        let lower = repair(values, &a[..mapped.len()], end, &mut sticky)?;
        r.checked_masks = values.len();
        let score = support::score(lower, &a[..mapped.len()], Some(activity))?;
        let old = support::score(
            incumbent.lower,
            &incumbent.a[..mapped.len()],
            Some(activity),
        )?;
        r.stop_reason = solver_stop;
        // Best checked support at the saved activity, not necessarily a newly
        // appended row: strict interval separation below can still refuse it.
        r.score_lower = Some(score.lo.max(old.lo));
        if let Some(upper) = upper(&s, &mut sticky) {
            let gap = I::point(upper)?.sub(I::point(r.score_lower?)?)?.hi;
            if gap >= 0. {
                r.envelope_upper = Some(upper);
                r.gap_upper = Some(gap);
            }
        }
        (score.lo > old.hi).then_some(Support { a, lower })
    })();
    if !sticky() {
        r.stop_reason = "cancelled";
        return None;
    }
    if require_hull_environment().is_err() {
        r.score_lower = None;
        r.envelope_upper = None;
        r.gap_upper = None;
        r.stop_reason = "unsupported-environment";
        return None;
    }
    Some(result.map_or(CellAttempt::Refused, CellAttempt::Ready))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::single_hidden_hull::mip_cuts::signed) fn prepare(
    table: &Table,
    q: super::Quote,
    row: &SignedMajorant,
    binding: &binding::SignedPhaseBinding<'_>,
    base: &RowSpec,
    incumbent: Option<Support>,
    b: &Budget,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CellAttempt<(RowSpec, usize)>> {
    if q.lp.is_none() {
        return Some(CellAttempt::Refused);
    }
    let Some(incumbent) = incumbent else {
        let mut counts = b.state.lock().ok()?;
        report(&mut counts).cell.lp.stop_reason = "no-ray-incumbent";
        return Some(CellAttempt::Refused);
    };
    // Missing captured advice is common when some component heads are stable.
    // This canonical target is advice only. Preserve the sealed table's absent
    // activity and the earlier Ray row, and report the target's provenance.
    let midpoint = [0.5; 8];
    let activity = table.activity.as_ref().unwrap_or(&midpoint);
    let values = table.ray.as_ref()?;
    let c = row.plan.components.get(table.component)?;
    let mut columns = [None; 8];
    for (bit, &head) in c.negative.iter().enumerate() {
        columns[bit] = binding.column(head);
    }
    let mapped = columns.map(|c| c.is_some());
    let mut stats = {
        let mut counts = b.state.lock().ok()?;
        let r = &mut report(&mut counts).cell.lp;
        if r.calls != 0 {
            return Some(CellAttempt::Refused);
        }
        r.calls = 1;
        r.activity_from_root = table.activity.is_some();
        r.clone()
    };
    let result = choose(
        values,
        &mapped[..q.k],
        &activity[..q.k],
        incumbent,
        &mut stats,
        end,
        poll,
    );
    {
        let mut counts = b.state.lock().ok()?;
        report(&mut counts).cell.lp = stats;
    }
    let Some(best) = result?.into_option() else {
        return Some(CellAttempt::Refused);
    };
    let needed = base.coeffs.len().checked_add(q.k)?;
    if needed > q.terms {
        return Some(CellAttempt::Refused);
    }
    let mut coeffs = Vec::with_capacity(needed);
    if coeffs.capacity() > q.terms {
        return Some(CellAttempt::Refused);
    }
    coeffs.extend_from_slice(&base.coeffs);
    let mut used = 0;
    for bit in 0..q.k {
        if !live(end) || !poll() {
            return None;
        }
        if best.a[bit] != 0. {
            coeffs.push((columns[bit]?.0, best.a[bit]));
            used += 1;
        }
    }
    Some(CellAttempt::Ready((
        RowSpec {
            lb: best.lower,
            ub: f64::INFINITY,
            coeffs,
        },
        used,
    )))
}
