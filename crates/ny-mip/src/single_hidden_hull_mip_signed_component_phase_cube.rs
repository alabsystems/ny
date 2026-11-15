// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Finite-table supports; optional prepaid table LP, no new Source pass.
use super::*;
#[path = "single_hidden_hull_mip_signed_component_phase_cell.rs"]
pub(in crate::single_hidden_hull::mip_cuts::signed) mod cell;
pub use cell::SignedPhaseCellReport;

#[derive(Debug, Clone, PartialEq)]
pub struct SignedPhaseCubeReport {
    pub cell: SignedPhaseCellReport,
    pub attempted: bool,
    pub admitted: bool,
    pub quoted_masks: usize,
    pub quoted_rows: usize,
    pub quoted_terms: usize,
    pub capture_total_work: usize,
    pub capture_activity_work: Option<usize>,
    pub capture_activity_bytes: Option<usize>,
    pub activity_transfer_attempts: usize,
    pub transferred_activity_values: usize,
    pub activity_values: usize,
    pub setup_work: usize,
    pub work_units: usize,
    pub reserved_work_units: usize,
    pub admitted_bytes: usize,
    pub collection_attempts: usize,
    pub collected_masks: usize,
    pub completed_tables: usize,
    pub prepare_attempts: usize,
    pub activity_components: usize,
    pub projected_components: usize,
    pub prepared_rows: usize,
    pub prepared_terms: usize,
    pub prepared_selector_terms: usize,
    pub appended_rows: usize,
    pub appended_terms: usize,
    pub appended_selector_terms: usize,
    pub stop_reason: &'static str,
}
impl Default for SignedPhaseCubeReport {
    fn default() -> Self {
        Self {
            cell: SignedPhaseCellReport::default(),
            attempted: false,
            admitted: false,
            quoted_masks: 0,
            quoted_rows: 0,
            quoted_terms: 0,
            capture_total_work: 0,
            capture_activity_work: None,
            capture_activity_bytes: None,
            activity_transfer_attempts: 0,
            transferred_activity_values: 0,
            activity_values: 0,
            setup_work: 0,
            work_units: 0,
            reserved_work_units: 0,
            admitted_bytes: 0,
            collection_attempts: 0,
            collected_masks: 0,
            completed_tables: 0,
            prepare_attempts: 0,
            activity_components: 0,
            projected_components: 0,
            prepared_rows: 0,
            prepared_terms: 0,
            prepared_selector_terms: 0,
            appended_rows: 0,
            appended_terms: 0,
            appended_selector_terms: 0,
            stop_reason: "not-attempted",
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub(super) struct CubeQuote {
    pub cell: Option<cell::Quote>,
    pub rows: usize,
    pub components: usize,
    pub negatives: usize,
    pub masks: usize,
    pub visits: usize,
    pub terms: usize,
    pub setup: usize,
    pub work: usize,
    pub bytes: usize,
}
#[derive(Debug)]
struct Admission {
    quote: CubeQuote,
    source: usize,
    dimensions: (usize, usize, usize),
    clocks: (Instant, Instant),
    deadline: Instant,
    ledger: std::sync::Weak<std::sync::Mutex<Counts>>,
    activity_attempted: bool,
    activity: Option<Vec<Option<f64>>>,
    activity_values: usize,
    // Copied under the original capture clock after all input-phase debt.
    input_target: Option<Vec<f64>>,
}
#[derive(Debug, Default)]
pub(in crate::single_hidden_hull::mip_cuts) struct State {
    attempted: bool,
    admission: Option<Admission>,
    finish_attempted: bool,
    prepare_attempted: bool,
    cell_attempted: bool,
    cell_offered: bool,
    input_phase_attempted: bool,
    input_phase_prepared: bool,
    prepared_rows: usize,
    prepared_terms: usize,
    prepared_selectors: usize,
}
fn report(c: &mut Counts) -> &mut SignedPhaseCubeReport {
    &mut c.component_rows.phase.cube
}
fn stop(b: &Budget, reason: &'static str) {
    if let Ok(mut c) = b.state.lock() {
        report(&mut c).stop_reason = reason;
    }
}
fn owned(a: &Admission, b: &Budget) -> bool {
    a.source == b.source
        && a.dimensions == b.dimensions
        && a.ledger.as_ptr() == Arc::as_ptr(&b.state)
        && a.clocks == (b.authority, b.search_end)
        && a.deadline >= b.authority
}
/// Read only an existing paid Ray capability; no Source or activity walk.
pub(in crate::single_hidden_hull::mip_cuts::signed) fn paid_ray_bits(
    counts: &Counts,
    b: &Budget,
) -> Option<usize> {
    let admission = counts.phase_cube.admission.as_ref()?;
    let cell = admission.quote.cell?;
    (owned(admission, b) && cell.ray.is_some() && !counts.phase_cube.cell_attempted)
        .then_some(cell.k)
}

fn term_bound(d: usize, components: usize, negatives: usize) -> Option<usize> {
    if d == 0 || components == 0 || negatives < components {
        return None;
    }
    let old = components.checked_mul(d.checked_add(32)?)?;
    // A completed row has at most D+P+2K terms: its original head/input
    // coefficients and at most K selectors. Collection caps C and total N,
    // while every future checked owner still has P<=16 and K<=8. Positive
    // sharing may grow P after admission, so never use the seed's P census.
    let aggregate = components
        .checked_mul(d.checked_add(MAX_POSITIVE)?)?
        .checked_add(negatives.checked_mul(2)?)?;
    // The same allowance covers plane sealing: D+2P+K+4 words per component
    // fit 2*(D+16+2K), since D>=1 and K>=1. Keep the old bound when the
    // source negative census is looser than the eligible component census.
    Some(old.min(aggregate))
}
/// Pay the new metadata walk before quoting it, including on All fallback.
pub(super) fn offer(cuts: &HullRootMipCuts<'_>, plan: &Plan, b: &Budget) -> Option<CubeQuote> {
    if !b.phase_cube_enabled
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.dimensions.0 != cuts.problem.input_dim
        || b.dimensions.1 != cuts.problem.hidden_dim
        || !live(b.search_end)
    {
        return None;
    }
    let reported_capture_work = cuts.lp_primal_report().work_units;
    let capture = cuts.primal.as_ref().and_then(|a| a.lock().ok()).map(|a| {
        (
            a.work(),
            a.relaxed_activity_debt(cuts.problem, b.search_end, &b.state),
        )
    });
    {
        let mut c = b.state.lock().ok()?;
        if c.phase_cube.attempted || c.replay_attempts != 0 {
            return None;
        }
        c.phase_cube.attempted = true;
        report(&mut c).attempted = true;
    }
    let p = cuts.problem;
    let setup = p
        .hidden_dim
        .checked_add(p.input_dim)?
        .checked_add(plan.components.len())?
        .checked_add(1)?
        .checked_mul(32)?
        .checked_add(4096)?;
    b.charge(setup)?;
    {
        let mut c = b.state.lock().ok()?;
        c.component_rows.work_units = c.component_rows.work_units.checked_add(setup)?;
        c.component_rows.phase.work_units = c.component_rows.phase.work_units.checked_add(setup)?;
        let r = report(&mut c);
        r.setup_work = setup;
        r.work_units = setup;
        r.stop_reason = "quote";
        r.capture_total_work = reported_capture_work;
        if let Some((total, subset)) = capture {
            r.capture_total_work = total;
            if let Some((w, bytes)) = subset {
                r.capture_activity_work = Some(w);
                r.capture_activity_bytes = Some(bytes);
            }
        }
    }
    let q = all::quote(cuts, plan)?;
    let (mut masks, mut square) = (0usize, 0usize);
    for item in &plan.components {
        if !live(b.search_end) {
            return None;
        }
        let k = item.negative.len();
        if !(1..=8).contains(&k) {
            return None;
        }
        masks = masks.checked_add(1usize << k)?;
        square = square.checked_add(k.checked_add(1)?.checked_pow(2)?)?;
    }
    if masks > MAX_MASKS {
        return None;
    }
    let terms = term_bound(p.input_dim, q.components, q.negatives)?;
    let span = terms
        .checked_add(q.negatives.checked_mul(2)?)?
        .checked_add(p.hidden_dim)?
        .checked_add(p.input_dim)?
        .checked_add(q.components)?
        .checked_add(1)?;
    let work = q
        .visits
        .checked_mul(64)?
        .checked_add(masks.checked_mul(32)?)?
        .checked_add(square.checked_mul(64)?)?
        .checked_add(span.checked_mul(128)?)?
        .checked_add(4096)?;
    let bytes = masks
        .checked_mul(32)?
        .checked_add(terms.checked_mul(64)?)?
        .checked_add(q.negatives.checked_mul(128)?)?
        .checked_add(q.components.checked_mul(768)?)?
        .checked_add(p.hidden_dim.checked_mul(32)?)?
        .checked_add(p.input_dim.checked_add(1)?.checked_mul(64)?)?
        .checked_add(4096)?;
    (work >= setup).then_some(CubeQuote {
        cell: None,
        rows: q.components,
        components: q.components,
        negatives: q.negatives,
        masks,
        visits: q.visits,
        terms,
        setup,
        work,
        bytes,
    })
}
/// Caller holds the one ledger lock and has funded BOTH All and this addition.
pub(super) fn activate(c: &mut Counts, cuts: &HullRootMipCuts<'_>, b: &Budget, q: CubeQuote) {
    c.phase_cube.admission = Some(Admission {
        quote: q,
        source: b.source,
        dimensions: b.dimensions,
        clocks: (b.authority, b.search_end),
        deadline: cuts.deadline,
        ledger: Arc::downgrade(&b.state),
        activity_attempted: false,
        activity: None,
        activity_values: 0,
        input_target: None,
    });
    let r = report(c);
    r.admitted = true;
    r.quoted_masks = q.masks;
    r.quoted_rows = q.rows;
    r.quoted_terms = q.terms;
    r.reserved_work_units = q.work - q.setup;
    r.admitted_bytes = q.bytes;
    r.stop_reason = "reserved";
    if let Some(cell) = q.cell {
        r.cell.admitted = true;
        r.cell.admitted_bytes = cell.bytes;
        if let Some(ray) = cell.ray {
            r.cell.ray.admitted = true;
            r.cell.ray.admitted_bytes = ray.bytes;
        }
        if let Some(joint) = cell.joint {
            r.cell.joint.admitted = true;
            r.cell.joint.admitted_bytes = joint.bytes;
        }
        if let Some(lp) = cell.lp {
            r.cell.lp.admitted = true;
            r.cell.lp.admitted_bytes = lp.bytes;
            r.cell.lp.pivot_limit = cell::lp::PIVOTS;
        }
    }
}
/// Test-only advisory changes after genuine input-family payment. No receipt,
/// Source, row seal, native capability, clock, work or byte debt is fabricated.
#[cfg(test)]
pub(in crate::single_hidden_hull::mip_cuts::signed) fn configure_paid_input_for_test(
    b: &Budget,
    target: Option<Vec<f64>>,
    activity: &[(usize, f64)],
    drop_capability: bool,
) {
    assert!(target.as_ref().is_none_or(|v| v.len() == b.dimensions.0
        && v.capacity() <= b.dimensions.0
        && v.iter().all(|x| x.is_finite())));
    assert!(activity.len() <= b.dimensions.1);
    assert!(activity
        .iter()
        .all(|&(j, x)| j < b.dimensions.1 && x.is_finite() && (0. ..=1.).contains(&x)));
    assert!(native_publication::funding(b).funded_calls > 0);
    let mut counts = b.state.lock().unwrap();
    assert_eq!(counts.replay_attempts, 0);
    assert!(counts.phase_cube.input_phase_attempted);
    assert!(!counts.phase_cube.cell_attempted && !counts.phase_cube.prepare_attempted);
    assert!(counts.component_rows.phase.cube.cell.input_phase.admitted);
    let a = counts.phase_cube.admission.as_mut().unwrap();
    assert!(a.quote.cell.as_ref().unwrap().input_phase.is_some());
    let lookup = a.activity.as_ref().unwrap();
    assert_eq!(lookup.len(), b.dimensions.1);
    a.input_target = if drop_capability { None } else { target };
    let lookup = a.activity.as_mut().unwrap();
    for &(head, value) in activity {
        lookup[head] = Some(value);
    }
    if drop_capability {
        assert!(a.quote.cell.as_mut().unwrap().input_phase.take().is_some());
    }
}

/// Copy proposal advice only after its complete payload/work admission. The
/// original capture clock governs the whole transfer; the owned finite copy
/// carries no capture/proof authority. The paid H lookup is moved, not cloned.
pub(super) fn transfer_activity(cuts: &HullRootMipCuts<'_>, b: &Budget) {
    let q = {
        let Ok(mut c) = b.state.lock() else {
            return;
        };
        let Some(a) = c.phase_cube.admission.as_mut() else {
            return;
        };
        if !owned(a, b)
            || a.source != std::ptr::from_ref(cuts.problem) as usize
            || a.deadline != cuts.deadline
            || a.activity_attempted
            || !live(b.search_end)
        {
            return;
        }
        a.activity_attempted = true;
        let q = a.quote;
        report(&mut c).activity_transfer_attempts = 1;
        q
    };
    // No ledger guard survives into the account lock. Setup prepaid both this
    // H initialization/N<=H copy and the admission metadata walks.
    let copied = (|| {
        let account = cuts.primal.as_ref()?.lock().ok()?;
        let (heads, values) = account.relaxed_activity(cuts.problem, b.search_end, &b.state)?;
        if heads.len() != values.len() || heads.len() > q.negatives || !live(b.search_end) {
            return None;
        }
        let mut lookup = vec![None; cuts.problem.hidden_dim];
        if lookup.capacity() > cuts.problem.hidden_dim.checked_mul(2)? {
            return None;
        }
        for (&head, &value) in heads.iter().zip(values) {
            if !live(b.search_end) {
                return None;
            }
            let slot = lookup.get_mut(head)?;
            if slot.is_some() || !value.is_finite() {
                return None;
            }
            *slot = Some(value.clamp(0., 1.));
        }
        live(b.search_end).then_some((lookup, heads.len()))
    })();
    let Some((lookup, count)) = copied else {
        return;
    };
    let Ok(mut c) = b.state.lock() else {
        return;
    };
    let Some(a) = c.phase_cube.admission.as_mut() else {
        return;
    };
    if !owned(a, b) || !a.activity_attempted || a.activity.is_some() || !live(b.search_end) {
        return;
    }
    a.activity = Some(lookup);
    a.activity_values = count;
    report(&mut c).transferred_activity_values = count;
}
pub(super) fn refused(c: &mut Counts) {
    if c.phase_cube.attempted && c.phase_cube.admission.is_none() {
        report(c).stop_reason = "all-fallback";
    }
}
/// A separately prepaid losing-family copy of already owned root advice.
/// No captured-account access or renewed clock is permitted at this seam.
pub(in crate::single_hidden_hull::mip_cuts::signed) fn portfolio_activity(
    plan: &Plan,
    b: &Budget,
) -> Option<Vec<Option<[f64; 8]>>> {
    let c = b.state.lock().ok()?;
    let a = c.phase_cube.admission.as_ref()?;
    if !owned(a, b)
        || c.phase_cube.prepare_attempted
        || plan.components.len() != a.quote.components
        || !live(b.search_end)
    {
        return None;
    }
    let mut out = Vec::with_capacity(plan.components.len());
    if out.capacity() > 2 * plan.components.len() {
        return None;
    }
    let mut count = 0usize;
    for item in &plan.components {
        let k = item.negative.len();
        if !(1..=8).contains(&k) || !live(b.search_end) {
            return None;
        }
        count = count.checked_add(k)?;
        if count > a.quote.negatives {
            return None;
        }
        let mut values = [0.; 8];
        let mut complete = a.activity.is_some();
        for (bit, &head) in item.negative.iter().enumerate() {
            match a
                .activity
                .as_ref()
                .and_then(|v| v.get(head))
                .copied()
                .flatten()
            {
                Some(value) if value.is_finite() && (0. ..=1.).contains(&value) => {
                    values[bit] = value
                }
                _ => complete = false,
            }
        }
        out.push(complete.then_some(values));
    }
    live(b.search_end).then_some(out)
}
pub(super) fn begin(c: &mut Counts) -> Option<()> {
    if c.phase_cube.admission.is_some() {
        let r = report(c);
        if r.collection_attempts != 0 {
            return None;
        }
        r.collection_attempts = 1;
        r.work_units = r.work_units.checked_add(r.reserved_work_units)?;
        r.reserved_work_units = 0;
        r.stop_reason = "collecting";
        if let Some(cell) = c.phase_cube.admission.as_ref()?.quote.cell {
            report(c).cell.work_units = cell.work;
            if let Some(input) = cell.input_phase {
                report(c).cell.input_phase.work_units = input.work;
            }
            if let Some(ray) = cell.ray {
                report(c).cell.ray.work_units = ray.work;
            }
            if let Some(joint) = cell.joint {
                report(c).cell.joint.work_units = joint.work;
            }
            if let Some(lp) = cell.lp {
                report(c).cell.lp.work_units = lp.work;
            }
        }
    }
    Some(())
}
pub(in crate::single_hidden_hull::mip_cuts) fn collecting(b: &Budget, plan: &Plan) -> bool {
    let Ok(c) = b.state.lock() else {
        return false;
    };
    let Some(a) = c.phase_cube.admission.as_ref() else {
        return false;
    };
    let r = &c.component_rows.phase.cube;
    if !owned(a, b)
        || r.collection_attempts != 1
        || r.reserved_work_units != 0
        || c.source_replays != 0
        || c.replay_attempts != 1
        || plan.components.len() > a.quote.components
    {
        return false;
    }
    let (mut m, mut n, mut v) = (0usize, 0usize, 0usize);
    for item in &plan.components {
        let k = item.negative.len();
        if !(1..=8).contains(&k) || item.positive.len() > MAX_POSITIVE {
            return false;
        }
        let Some(mm) = m.checked_add(1usize << k) else {
            return false;
        };
        m = mm;
        let Some(nn) = n.checked_add(k) else {
            return false;
        };
        n = nn;
        let Some(vv) = k.checked_mul(1usize << k).and_then(|x| v.checked_add(x)) else {
            return false;
        };
        v = vv;
    }
    m <= a.quote.masks && n <= a.quote.negatives && v <= a.quote.visits
}
/// Alpha is deliberately absent: only the same allocated U and bit order need
/// match the completed table. The source pass checked every legal alpha row.
fn plane_bits(plan: &Plan) -> impl Iterator<Item = u64> + '_ {
    plan.components.iter().flat_map(|c| {
        [
            c.negative.len() as u64,
            c.positive.len() as u64,
            c.g.len() as u64,
            c.positive_units.len() as u64,
        ]
        .into_iter()
        .chain(c.negative.iter().map(|x| *x as u64))
        .chain(
            c.positive
                .iter()
                .zip(&c.positive_units)
                .flat_map(|(j, u)| [*j as u64, u64::from(*u)]),
        )
        .chain(c.g.iter().map(|x| x.to_bits()))
    })
}
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts) struct Collected {
    source: usize,
    ledger: Arc<std::sync::Mutex<Counts>>,
    witness: Arc<RetainedWords>,
    signature: Arc<RetainedWords>,
    clocks: (Instant, Instant),
    values: Vec<Vec<f64>>,
    planes: Vec<u64>,
    integrity: Vec<u64>,
    cell: Option<cell::Table>,
}
impl Collected {
    /// Selection advice only, after the original replay has produced this
    /// collection. The complete seal is still checked before any row license.
    pub(super) fn native_component(&self) -> Option<usize> {
        self.cell.as_ref().map(|table| table.component)
    }

    fn bits(&self) -> impl Iterator<Item = u64> + '_ {
        [
            self.source as u64,
            self.values.len() as u64,
            self.planes.len() as u64,
        ]
        .into_iter()
        .chain(self.planes.iter().copied())
        .chain(
            self.values
                .iter()
                .flat_map(|v| std::iter::once(v.len() as u64).chain(v.iter().map(|x| x.to_bits()))),
        )
        .chain(self.cell.iter().flat_map(cell::Table::bits))
    }
    /// Called only after Signed replay's complete Source, bit and FP barriers.
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn finish(
        row: &SignedMajorant,
        cuts: &HullRootMipCuts<'_>,
        values: Vec<Vec<f64>>,
        cell: Option<cell::Table>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        let b = &row.budget;
        let q = {
            let mut c = b.state.lock().ok()?;
            let a = c.phase_cube.admission.as_ref()?;
            if !owned(a, b)
                || a.source != row.source
                || a.deadline != cuts.deadline
                || c.phase_cube.finish_attempted
                || c.replay_attempts != 1
                || c.component_rows.phase.cube.collection_attempts != 1
                || c.component_rows.phase.cube.reserved_work_units != 0
                || c.source_replays != 0
                || !live(end.min(b.authority))
            {
                return None;
            }
            let q = a.quote;
            c.phase_cube.finish_attempted = true;
            q
        };
        if values.len() != row.plan.components.len()
            || values.len() > q.components
            || values.capacity() > q.components.checked_mul(2)?
        {
            return None;
        }
        let mut masks = 0usize;
        for (c, v) in row.plan.components.iter().zip(&values) {
            if !live(end)
                || !poll()
                || !(1..=8).contains(&c.negative.len())
                || c.positive.len() > MAX_POSITIVE
                || c.positive_units.len() != c.positive.len()
                || c.g.len() != cuts.problem.input_dim
                || v.len() != 1usize << c.negative.len()
            {
                return None;
            }
            masks = masks.checked_add(v.len())?;
            if masks > q.masks
                || v.capacity() > v.len().checked_mul(2)?
                || v.iter().any(|x| !x.is_finite())
            {
                return None;
            }
        }
        let plane_capacity = q.terms.checked_mul(2)?;
        let mut planes = Vec::with_capacity(plane_capacity);
        if planes.capacity() > plane_capacity {
            return None;
        }
        planes.extend(plane_bits(&row.plan));
        if planes.len() > q.terms.checked_mul(2)? {
            return None;
        }
        let cell = cell.filter(|table| {
            q.cell
                .is_some_and(|quote| table.valid(quote, &values, &row.plan))
        });
        let mut out = Self {
            source: row.source,
            ledger: b.state.clone(),
            witness: row.witness.clone(),
            signature: row.signature.clone(),
            clocks: (b.authority, b.search_end),
            values,
            planes,
            integrity: Vec::new(),
            cell,
        };
        let integrity_capacity = q
            .masks
            .checked_add(q.terms.checked_mul(2)?)?
            .checked_add(q.components)?
            .checked_add(3)?
            .checked_add(q.cell.map_or(0, |x| x.seal_words()))?;
        let mut integrity = Vec::with_capacity(integrity_capacity);
        if integrity.capacity() > integrity_capacity {
            return None;
        }
        integrity.extend(out.bits());
        out.integrity = integrity;
        require_hull_environment().ok()?;
        if !live(end.min(b.authority)) || !poll() {
            return None;
        }
        let mut c = b.state.lock().ok()?;
        let r = report(&mut c);
        r.collected_masks = masks;
        r.completed_tables = out.values.len();
        r.stop_reason = "collected";
        Some(out)
    }
    fn owned(&self, row: &SignedMajorant, q: CubeQuote) -> bool {
        self.source == row.source
            && self.source == row.budget.source
            && Arc::ptr_eq(&self.ledger, &row.budget.state)
            && Arc::ptr_eq(&self.witness, &row.witness)
            && Arc::ptr_eq(&self.signature, &row.signature)
            && self.clocks == (row.budget.authority, row.budget.search_end)
            && self.values.len() <= q.components
            && self.planes.len() <= 2 * q.terms
            && self.integrity.len()
                <= q.masks + 2 * q.terms + q.components + 3 + q.cell.map_or(0, |x| x.seal_words())
            && self.cell.as_ref().is_none_or(|table| {
                q.cell
                    .is_some_and(|quote| table.valid(quote, &self.values, &row.plan))
            })
    }
}
/// Exact real k=2 lower-envelope triangle proposal when the mixed-difference
/// sign is certified. Slopes are still advice; serialized repair stays final.
pub(super) fn two_bit_envelope(values: &[f64], activity: Option<&[f64]>) -> Option<[f64; 8]> {
    require_hull_environment().ok()?;
    if values.len() != 4 {
        return None;
    }
    let (a, b, c, d) = (values[0], values[1], values[2], values[3]);
    let mixed = certificate::I::point(a)?
        .add(certificate::I::point(d)?)?
        .sub(certificate::I::point(b)?)?
        .sub(certificate::I::point(c)?)?;
    let (x, y) = if let Some(z) = activity {
        (
            finite(*z.first()?)?.clamp(0., 1.),
            finite(*z.get(1)?)?.clamp(0., 1.),
        )
    } else {
        (0., 0.)
    };
    let (first, second) = if mixed.lo >= 0. {
        // For a clamped operand>=.5, 1-operand is exact by Sterbenz. This
        // distinguishes (1,minsubnormal) from the exact diagonal tie.
        let lower = if x >= 0.5 {
            y <= 1. - x
        } else if y >= 0.5 {
            x <= 1. - y
        } else {
            true
        };
        if lower {
            (b - a, c - a)
        } else {
            (d - c, d - b)
        }
    } else if mixed.hi <= 0. {
        if x >= y {
            (b - a, d - b)
        } else {
            (d - c, c - a)
        }
    } else {
        return None;
    };
    let mut weights = [0.; 8];
    weights[0] = finite(first)?;
    weights[1] = finite(second)?;
    Some(weights)
}
/// Production proposal dispatch. The raw staircase remains independently
/// testable as advice that may need a large complete-table intercept repair.
pub(in crate::single_hidden_hull::mip_cuts::signed) fn support_weights(
    values: &[f64],
    heads: &[usize],
    activity: Option<&[f64]>,
) -> Option<[f64; 8]> {
    if heads.len() == 2 {
        if let Some(weights) = two_bit_envelope(values, activity) {
            return Some(weights);
        }
    }
    staircase(values, heads, activity)
}
/// Stack-only simplex proposal. Validity comes from repair, not interpolation.
pub(super) fn staircase(
    values: &[f64],
    heads: &[usize],
    activity: Option<&[f64]>,
) -> Option<[f64; 8]> {
    require_hull_environment().ok()?;
    let k = heads.len();
    if !(1..=8).contains(&k) || values.len() != 1usize << k {
        return None;
    }
    let (mut order, mut distance, mut corner) = ([0usize; 8], [0.; 8], 0usize);
    for i in 0..k {
        order[i] = i;
        if let Some(z) = activity {
            let x = finite(*z.get(i)?)?.clamp(0., 1.);
            let high = x > 0.5;
            if high {
                corner |= 1usize << i;
            }
            distance[i] = if high { 1. - x } else { x };
        }
    }
    if activity.is_some() {
        for i in 1..k {
            let mut j = i;
            while j > 0
                && (distance[order[j]] > distance[order[j - 1]]
                    || (distance[order[j]] == distance[order[j - 1]]
                        && heads[order[j]] < heads[order[j - 1]]))
            {
                order.swap(j, j - 1);
                j -= 1;
            }
        }
    } // Missing advice: original bit order and all-zero corner.
    let mut weights = [0.; 8];
    let mut mask = corner;
    for &bit in &order[..k] {
        let next = mask ^ (1usize << bit);
        let d = finite(*values.get(next)? - *values.get(mask)?)?;
        weights[bit] = if mask & (1usize << bit) == 0 { d } else { -d };
        mask = next;
    }
    Some(weights)
}
/// Repair on serialized -2w coefficients, then a separate interval verification.
pub(in crate::single_hidden_hull::mip_cuts::signed) fn repair(
    values: &[f64],
    coefficient: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    require_hull_environment().ok()?;
    let k = coefficient.len();
    if !(1..=8).contains(&k)
        || values.len() != 1usize << k
        || coefficient.iter().any(|x| !x.is_finite())
    {
        return None;
    }
    let mut lower = f64::INFINITY;
    for (mask, &value) in values.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        let mut candidate = finite(mul_down(2., finite(value)?))?;
        for (bit, &a) in coefficient.iter().enumerate() {
            if mask & (1usize << bit) != 0 {
                candidate = finite(add_down(candidate, a))?;
            }
        }
        lower = lower.min(candidate);
    }
    finite(lower)?;
    for (mask, &value) in values.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        let mut interval = certificate::I::point(value)?.scale(2.)?;
        for (bit, &a) in coefficient.iter().enumerate() {
            if mask & (1usize << bit) != 0 {
                interval = interval.add(certificate::I::point(a)?)?;
            }
        }
        if lower > interval.lo {
            return None;
        }
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some(lower)
}
pub(super) fn reset(c: &mut Counts) {
    c.phase_cube.prepared_rows = 0;
    c.phase_cube.prepared_terms = 0;
    c.phase_cube.prepared_selectors = 0;
    let r = report(c);
    r.prepared_rows = 0;
    r.prepared_terms = 0;
    r.prepared_selector_terms = 0;
    r.cell.prepared_rows = 0;
    r.cell.prepared_terms = 0;
    r.cell.prepared_selectors = 0;
    r.cell.ray.prepared_rows = 0;
    r.cell.ray.prepared_terms = 0;
    r.cell.ray.prepared_selectors = 0;
    r.cell.lp.prepared_rows = 0;
    r.cell.lp.prepared_terms = 0;
    r.cell.lp.prepared_selectors = 0;
    r.cell.joint.prepared_rows = 0;
    r.cell.joint.prepared_terms = 0;
    r.cell.joint.prepared_selectors = 0;
}
pub(super) fn header_rows(c: &Counts) -> Option<usize> {
    let n = c.phase_cube.prepared_rows;
    (n == c.component_rows.phase.cube.prepared_rows
        && c.phase_cube.prepared_selectors == c.component_rows.phase.cube.prepared_selector_terms)
        .then_some(n)
}
pub(in crate::single_hidden_hull::mip_cuts) fn selector_terms(c: &Counts) -> Option<usize> {
    c.component_rows
        .phase
        .prepared_rows
        .checked_sub(c.phase_cube.prepared_rows)?
        .checked_add(c.phase_cube.prepared_selectors)
}
pub(in crate::single_hidden_hull::mip_cuts) fn commit(c: &mut Counts) {
    let rows = c.phase_cube.prepared_rows;
    let terms = c.phase_cube.prepared_terms;
    let selectors = c.phase_cube.prepared_selectors;
    let r = report(c);
    r.appended_rows = rows;
    r.appended_terms = terms;
    r.appended_selector_terms = selectors;
    r.cell.appended_rows = r.cell.prepared_rows;
    r.cell.ray.appended_rows = r.cell.ray.prepared_rows;
    r.cell.lp.appended_rows = r.cell.lp.prepared_rows;
    r.cell.joint.appended_rows = r.cell.joint.prepared_rows;
    if rows != 0 {
        r.stop_reason = "committed";
    }
}
/// Returns an atomic optional family. Every full table walk follows the once latch.
pub(super) fn prepare(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    table: &Collected,
    minima: &[f64],
    old: &[RowSpec],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    let b = &row.budget;
    let end = end.min(b.authority).min(cuts.deadline);
    if !binding.matches(cuts, parts) || !live(end.min(b.authority)) || !poll() {
        return None;
    }
    require_hull_environment().ok()?;
    let q = {
        let mut c = b.state.lock().ok()?;
        let a = c.phase_cube.admission.as_ref()?;
        if !owned(a, b)
            || a.source != row.source
            || a.deadline != cuts.deadline
            || c.phase_cube.prepare_attempted
        {
            return Some(Vec::new());
        }
        let q = a.quote;
        let a = c.phase_cube.admission.as_mut()?;
        let activity = a.activity.take();
        let activity_values = a.activity_values;
        c.phase_cube.prepare_attempted = true;
        report(&mut c).prepare_attempts += 1;
        (q, activity, activity_values)
    };
    let (q, lookup, activity_values) = q;
    if !table.owned(row, q)
        || table.values.len() != row.plan.components.len()
        || old.len() != row.plan.components.len()
        || minima.len() != old.len()
        || !table.bits().eq(table.integrity.iter().copied())
        || !plane_bits(&row.plan).eq(table.planes.iter().copied())
    {
        stop(b, "table-context");
        return Some(Vec::new());
    }
    {
        let c = b.state.lock().ok()?;
        let a = c.phase_all.admission.as_ref()?;
        if !all::owned(a, b)
            || c.phase_all.parts != std::ptr::from_ref(parts) as usize
            || c.phase_all.original_rows != parts.problem.num_rows()
            || parts.problem.num_rows() > a.quote.original_rows
            || c.phase_all
                .witness
                .as_ref()
                .is_none_or(|w| !Arc::ptr_eq(w, &row.witness))
        {
            stop_after_unlock(b, c);
            return Some(Vec::new());
        }
    }
    // This immutable proposal lookup was fully copied under search_end at
    // admission. Moving it does not reopen/renew the captured account.
    let activity_complete = lookup
        .as_ref()
        .is_some_and(|v| v.len() == cuts.problem.hidden_dim);
    let (mut rows, mut terms, mut heads, mut selectors, mut advised, mut projected) = (
        Vec::with_capacity(q.rows),
        0usize,
        0usize,
        0usize,
        0usize,
        0usize,
    );
    if rows.capacity() > q.rows {
        stop(b, "row-capacity");
        return Some(Vec::new());
    }
    let native_component = native_publication::selected(row);
    let native_extra_component = native_publication::selected_extra(row);
    let mut native_origin = None;
    let mut extra_origin = None;
    let mut cell_base_row = None;
    for (index, c) in row.plan.components.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        let k = c.negative.len();
        if !(1..=8).contains(&k) || c.positive.len() > MAX_POSITIVE {
            return Some(Vec::new());
        }
        let values = table.values.get(index)?;
        if values.len() != 1usize << k
            || values.iter().copied().fold(f64::INFINITY, f64::min) != minima[index]
            || all::base_valid(row, cuts, parts, binding, index, &old[index], minima[index])
                .is_none()
        {
            stop(b, "base-row");
            return Some(Vec::new());
        }
        if k < 2 {
            continue;
        }
        let mut activity = [0.; 8];
        let mut have = activity_complete;
        if have {
            for (bit, &head) in c.negative.iter().enumerate() {
                match lookup.as_ref().and_then(|v| v.get(head)).copied().flatten() {
                    Some(x) => activity[bit] = x,
                    None => have = false,
                }
            }
        }
        let Some(weights) = support_weights(values, &c.negative, have.then_some(&activity[..k]))
        else {
            stop(b, "proposal-numeric");
            return Some(Vec::new());
        };
        let (mut encoded, mut columns, mut used) = ([0.; 8], [None; 8], 0usize);
        for (bit, &head) in c.negative.iter().enumerate() {
            columns[bit] = binding.column(head);
            if columns[bit].is_some() {
                let Some(a) = finite(-2. * weights[bit]) else {
                    stop(b, "doubling-overflow");
                    return Some(Vec::new());
                };
                encoded[bit] = a;
                if a != 0. {
                    used += 1;
                }
            }
        }
        if used < 2 {
            continue;
        }
        let mut cancelled = false;
        let lower = repair(values, &encoded[..k], end, &mut || {
            let ok = poll();
            cancelled |= !ok;
            ok
        });
        if cancelled || !live(end) {
            return None;
        }
        let Some(lower) = lower else {
            stop(b, "repair-refused");
            return Some(Vec::new());
        };
        let needed = old[index].coeffs.len().checked_add(k)?;
        let mut coeffs = Vec::with_capacity(needed);
        if coeffs.capacity() > needed {
            stop(b, "coefficient-capacity");
            return Some(Vec::new());
        }
        coeffs.extend_from_slice(&old[index].coeffs);
        for bit in 0..k {
            if encoded[bit] != 0. {
                coeffs.push((columns[bit]?.0, encoded[bit]));
            }
        }
        terms = terms.checked_add(coeffs.len())?;
        if terms > q.terms || rows.len() >= q.components {
            stop(b, "term-bound");
            return Some(Vec::new());
        }
        heads = heads
            .checked_add(c.positive.len())?
            .checked_add(c.negative.len())?;
        selectors = selectors.checked_add(used)?;
        rows.push(RowSpec {
            lb: lower,
            ub: f64::INFINITY,
            coeffs,
        });
        if table
            .cell
            .as_ref()
            .is_some_and(|cell| cell.component == index)
        {
            cell_base_row = Some(rows.len() - 1);
        }
        if native_component == Some(index) {
            native_origin = Some((index, rows.len() - 1, used, None));
        }
        advised += usize::from(have);
        projected += usize::from(columns[..k].iter().any(|x| x.is_none()));
    }
    let mut cell_counts = [(0usize, 0usize, 0usize); 4];
    if let (Some(cell), Some(quote)) = (&table.cell, q.cell) {
        let index = cell.component;
        if all::base_valid(row, cuts, parts, binding, index, &old[index], minima[index]).is_none() {
            return Some(Vec::new());
        }
        let mut prior_rows = [None; 4];
        let mut incumbent = None;
        for (slot, family) in [
            cell::Family::Cell,
            cell::Family::Ray,
            cell::Family::Lp,
            cell::Family::Joint,
        ]
        .into_iter()
        .enumerate()
        {
            let proposal = if family == cell::Family::Lp {
                cell::lp::prepare(
                    cell,
                    quote,
                    row,
                    binding,
                    &old[index],
                    incumbent,
                    b,
                    end,
                    poll,
                )?
                .into_option()
            } else {
                cell::prepare(
                    cell,
                    quote,
                    row,
                    binding,
                    &old[index],
                    family,
                    &mut incumbent,
                    end,
                    poll,
                )?
                .into_option()
            };
            let Some((extra, used)) = proposal else {
                continue;
            };
            // A new constant or unary support is useful too. Suppress exact
            // duplicates while retaining every existing component/Cube row.
            let identical = |old: &RowSpec| {
                old.lb == extra.lb && old.ub == extra.ub && old.coeffs == extra.coeffs
            };
            if !identical(&old[index])
                && !cell_base_row
                    .and_then(|i| rows.get(i))
                    .is_some_and(identical)
                && !prior_rows
                    .iter()
                    .flatten()
                    .any(|&i| rows.get(i).is_some_and(identical))
            {
                cell_counts[slot] = (1, extra.coeffs.len(), used);
                terms = terms.checked_add(extra.coeffs.len())?;
                heads = heads
                    .checked_add(row.plan.components[index].positive.len())?
                    .checked_add(row.plan.components[index].negative.len())?;
                selectors = selectors.checked_add(used)?;
                if terms > q.terms || rows.len() >= q.rows {
                    return Some(Vec::new());
                }
                prior_rows[slot] = Some(rows.len());
                rows.push(extra);
                if native_extra_component == Some(index) {
                    // The old native unary+Cube pair stays untouched. Select
                    // only the optional THIRD slot: last actually emitted
                    // Joint > LP > Ray > Cell. This is not a dominance claim.
                    // Retain borrowed row/slope locations, never clones.
                    let g = (family == cell::Family::Joint)
                        .then(|| cell.native_input_g())
                        .flatten();
                    extra_origin = Some((index, rows.len() - 1, used, g));
                }
            }
        }
    }
    require_hull_environment().ok()?;
    if !live(end.min(b.authority)) || !poll() {
        return None;
    }
    let mut c = b.state.lock().ok()?;
    c.phase_cube.prepared_rows = rows.len();
    c.phase_cube.prepared_terms = terms;
    c.phase_cube.prepared_selectors = selectors;
    // The combined admission owns headers even if EVERY unary gap was zero.
    c.phase_all.pending |= !rows.is_empty();
    c.component_rows.phase.prepared_rows = c
        .component_rows
        .phase
        .prepared_rows
        .checked_add(rows.len())?;
    c.component_rows.phase.prepared_terms =
        c.component_rows.phase.prepared_terms.checked_add(terms)?;
    c.component_rows.phase.prepared_head_terms = c
        .component_rows
        .phase
        .prepared_head_terms
        .checked_add(heads)?;
    let r = report(&mut c);
    r.prepared_rows = rows.len();
    r.prepared_terms = terms;
    r.prepared_selector_terms = selectors;
    r.activity_values = activity_values;
    r.activity_components = advised;
    r.projected_components = projected;
    r.cell.prepared_rows = cell_counts[0].0;
    r.cell.prepared_terms = cell_counts[0].1;
    r.cell.prepared_selectors = cell_counts[0].2;
    r.cell.ray.prepared_rows = cell_counts[1].0;
    r.cell.ray.prepared_terms = cell_counts[1].1;
    r.cell.ray.prepared_selectors = cell_counts[1].2;
    r.cell.lp.prepared_rows = cell_counts[2].0;
    r.cell.lp.prepared_terms = cell_counts[2].1;
    r.cell.lp.prepared_selectors = cell_counts[2].2;
    r.cell.joint.prepared_rows = cell_counts[3].0;
    r.cell.joint.prepared_terms = cell_counts[3].1;
    r.cell.joint.prepared_selectors = cell_counts[3].2;
    r.stop_reason = if rows.is_empty() {
        "no-multibit-row"
    } else {
        "prepared"
    };
    drop(c);
    for (origin, extra) in [(native_origin, false), (extra_origin, true)] {
        let Some((component, position, selector_count, stored_g)) = origin else {
            continue;
        };
        let stored = &rows[position];
        // Every emitted family orders its selector suffix by component bits.
        // Decode only the chosen suffix once, under the original native quote.
        // A malformed optional descriptor leaves all global rows untouched.
        let selector_heads = (|| {
            if selector_count > 8 {
                return None;
            }
            let first = stored.coeffs.len().checked_sub(selector_count)?;
            let mut suffix = stored.coeffs[first..].iter().peekable();
            let mut heads = [0; 8];
            let mut count = 0;
            for &head in &row.plan.components.get(component)?.negative {
                if suffix.peek().is_some_and(|entry| {
                    binding
                        .column(head)
                        .is_some_and(|column| column.0 == entry.0)
                }) {
                    heads[count] = head;
                    count += 1;
                    suffix.next();
                }
            }
            (count == selector_count && suffix.next().is_none()).then_some(heads)
        })();
        if let Some(selector_heads) = selector_heads {
            native_publication::stage(
                row,
                cuts,
                parts,
                binding,
                NativeRow {
                    component,
                    cube: true,
                    extra,
                    selector_heads,
                    selector_count,
                    stored,
                    stored_g,
                },
                poll,
            )?;
        }
    }
    Some(rows)
}
fn stop_after_unlock(b: &Budget, c: std::sync::MutexGuard<'_, Counts>) {
    drop(c);
    stop(b, "header-context");
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_component_phase_cube_tests.rs"]
mod tests;
