// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Optional phase lifts of finite planes, produced by the same first replay.
use super::*;

/// Private producer provenance. Only this phase module and its All/Cube
/// children can mint it, after their complete-family final barrier.
/// All may retain its checked component row when no unary lift was emitted.
pub(in crate::single_hidden_hull::mip_cuts::signed) struct NativeRow<'a> {
    component: usize,
    cube: bool,
    extra: bool,
    selector_heads: [usize; 8],
    selector_count: usize,
    stored: &'a RowSpec,
    /// Borrowed only while capturing this already emitted, sealed support.
    /// None retains the original component slope. Never cloned or published.
    stored_g: Option<&'a [f64]>,
}
impl NativeRow<'_> {
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn is_extra(&self) -> bool {
        self.extra
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn view(
        &self,
    ) -> (usize, bool, &[usize], &RowSpec, Option<&[f64]>) {
        (
            self.component,
            self.cube,
            &self.selector_heads[..self.selector_count],
            self.stored,
            self.stored_g,
        )
    }
}

#[cfg(test)]
impl<'a> NativeRow<'a> {
    /// Adversarial capture controls only; no production origin constructor.
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn for_test(
        component: usize,
        cube: bool,
        selector_heads: [usize; 8],
        selector_count: usize,
        stored: &'a RowSpec,
        stored_g: Option<&'a [f64]>,
    ) -> Self {
        assert!(selector_count <= 8);
        Self {
            component,
            cube,
            extra: false,
            selector_heads,
            selector_count,
            stored,
            stored_g,
        }
    }

    pub(in crate::single_hidden_hull::mip_cuts::signed) fn for_extra_test(
        component: usize,
        selector_heads: [usize; 8],
        selector_count: usize,
        stored: &'a RowSpec,
        stored_g: Option<&'a [f64]>,
    ) -> Self {
        let mut row = Self::for_test(
            component,
            true,
            selector_heads,
            selector_count,
            stored,
            stored_g,
        );
        row.extra = true;
        row
    }
}

// The bank's existing 2048-byte token/transient allowance covers this borrowed
// producer descriptor. It adds no owned vector or persistent ledger field.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<NativeRow<'static>>() <= 128);

#[derive(Debug, Clone, PartialEq)]
pub struct SignedPhaseRowsReport {
    pub attempted: bool,
    pub all_selector_mode: bool,
    pub cube: SignedPhaseCubeReport,
    pub original_ir_row_limit: usize,
    pub reserved_ir_rows: usize,
    pub admitted: bool,
    pub quoted_rows: usize,
    pub quoted_terms: usize,
    pub work_units: usize,
    pub reserved_work_units: usize,
    pub admitted_bytes: usize,
    pub collection_attempts: usize,
    pub collected_components: usize,
    pub prepared_rows: usize,
    pub prepared_head_terms: usize,
    pub prepared_terms: usize,
    pub appended_rows: usize,
    pub appended_terms: usize,
    pub stop_reason: &'static str,
}
impl Default for SignedPhaseRowsReport {
    fn default() -> Self {
        Self {
            attempted: false,
            all_selector_mode: false,
            cube: SignedPhaseCubeReport::default(),
            original_ir_row_limit: 0,
            reserved_ir_rows: 0,
            admitted: false,
            quoted_rows: 0,
            quoted_terms: 0,
            work_units: 0,
            reserved_work_units: 0,
            admitted_bytes: 0,
            collection_attempts: 0,
            collected_components: 0,
            prepared_rows: 0,
            prepared_head_terms: 0,
            prepared_terms: 0,
            appended_rows: 0,
            appended_terms: 0,
            stop_reason: "not-attempted",
        }
    }
}
#[path = "single_hidden_hull_mip_signed_component_phase_all.rs"]
mod all;
#[path = "single_hidden_hull_mip_signed_component_phase_cube.rs"]
pub(super) mod cube;
pub use cube::{SignedPhaseCellReport, SignedPhaseCubeReport};

#[derive(Debug, Clone, Copy)]
pub(super) struct AllQuote {
    pub components: usize,
    pub negatives: usize,
    pub visits: usize,
    pub terms: usize,
    pub rows: usize,
    pub original_rows: usize,
    pub work: usize,
    pub bytes: usize,
}
#[derive(Debug)]
struct AllAdmission {
    quote: AllQuote,
    source: usize,
    dimensions: (usize, usize, usize),
    authority: Instant,
    search_end: Instant,
    deadline: Instant,
}
#[derive(Debug, Default)]
pub(super) struct AllState {
    admission_attempted: bool,
    admission: Option<AllAdmission>,
    prepare_attempted: bool,
    pending: bool,
    claimed: bool,
    parts: usize,
    original_rows: usize,
    prepared_rows: usize,
    witness: Option<Arc<RetainedWords>>,
}

/// Opaque aggregate-header authority, minted only after authenticated staging.
pub(super) struct HeaderGrant {
    source: usize,
    ledger: Arc<std::sync::Mutex<Counts>>,
    witness: Arc<RetainedWords>,
    authority: Instant,
    search_end: Instant,
    parts: usize,
    original_rows: usize,
    additional: usize,
    total_bound: usize,
}
/// The original phase-header authority could not be authenticated.
pub(super) struct HeaderRefused;
/// Ok(None) needs no phase grant; Ok(Some(_)) carries the same exact owner.
pub(super) type HeaderClaim = Result<Option<HeaderGrant>, HeaderRefused>;
impl HeaderGrant {
    pub(super) fn matches(
        &self,
        row: &SignedMajorant,
        parts: &MipParts,
        additional: usize,
    ) -> bool {
        self.context_matches(row, parts, additional) && live(self.authority)
    }
    pub(super) fn context_matches(
        &self,
        row: &SignedMajorant,
        parts: &MipParts,
        additional: usize,
    ) -> bool {
        self.source == row.source
            && self.source == row.budget.source
            && Arc::ptr_eq(&self.ledger, &row.budget.state)
            && Arc::ptr_eq(&self.witness, &row.witness)
            && self.authority == row.budget.authority
            && self.search_end == row.budget.search_end
            && self.parts == std::ptr::from_ref(parts) as usize
            && self.original_rows == parts.problem.num_rows()
            && self.additional == additional
            && additional <= self.total_bound
    }
}
pub(super) fn all_mode(b: &Budget) -> bool {
    b.state
        .lock()
        .is_ok_and(|c| c.phase_all.admission.is_some())
}
/// Inspect the existing All+Cube aggregate owner while its caller holds the ledger.
pub(super) fn portfolio_header_upgrade(c: &Counts, b: &Budget, components: usize) -> Option<usize> {
    let a = c.phase_all.admission.as_ref()?;
    if !all::owned(a, b)
        || !c.component_rows.phase.cube.admitted
        || c.phase_all.prepare_attempted
        || c.phase_all.claimed
        || a.quote.components != components
        || c.replay_attempts != 0
    {
        return None;
    }
    a.quote.rows.checked_add(components)
}
pub(super) fn apply_portfolio_header_upgrade(c: &mut Counts, rows: usize) -> Option<()> {
    c.phase_all.admission.as_mut()?.quote.rows = rows;
    Some(())
}
pub(super) fn claim_ir_headers(
    row: &SignedMajorant,
    parts: &MipParts,
    additional: usize,
) -> HeaderClaim {
    all::claim_ir_headers(row, parts, additional)
}
pub(super) fn record_ir_headers(
    row: &SignedMajorant,
    parts: &MipParts,
    additional: usize,
    grant: &HeaderGrant,
) {
    // Actual allocation completion is recorded even if its clock just expired.
    if !grant.context_matches(row, parts, additional) {
        return;
    }
    if let Ok(mut c) = row.budget.state.lock() {
        if c.phase_all.claimed && c.phase_all.pending {
            c.component_rows.phase.reserved_ir_rows = additional;
        }
    }
}
pub(super) fn collecting_plan(b: &Budget, plan: &Plan) -> bool {
    collecting(b) && (!all_mode(b) || all::collection_shape(b, plan))
}
fn stop(b: &Budget, reason: &'static str) {
    if let Ok(mut c) = b.state.lock() {
        c.component_rows.phase.stop_reason = reason;
    }
}
#[derive(Debug)]
pub(super) struct Quote {
    pub work: usize,
    pub bytes: usize,
    pub terms: usize,
}
fn setup(cuts: &HullRootMipCuts<'_>, plan: &Plan) -> Option<usize> {
    cuts.problem
        .hidden_dim
        .checked_add(cuts.problem.input_dim)?
        .checked_add(plan.components.len())?
        .checked_add(cuts.len())?
        .checked_add(2)?
        .checked_mul(32)
}
/// Called only after the existing component construction has been admitted.
/// Quote scans are prepaid even on optional refusal; old admission is preserved.
pub(super) fn quote(cuts: &HullRootMipCuts<'_>, plan: &Plan) -> Option<Quote> {
    let p = cuts.problem;
    let old = component_rows::quote(cuts, plan)?;
    let c = plan.components.len();
    // len excludes the not-yet-published signed row and optional old components.
    let r = cuts
        .len()
        .checked_add(usize::from(cuts.signed.is_none()))?
        .checked_add(c)?;
    let mut visits = 0usize;
    for item in &plan.components {
        let k = item.negative.len();
        if k == 0 || k > 8 {
            return None;
        }
        visits = visits.checked_add(k.checked_mul(1usize.checked_shl(k as u32)?)?)?;
    }
    let span = old
        .terms
        .checked_add(c.checked_mul(2)?)?
        .checked_add(p.hidden_dim)?
        .checked_add(p.input_dim)?
        .checked_add(r)?
        .checked_add(1)?;
    Some(Quote {
        work: visits
            .checked_mul(32)?
            .checked_add(span.checked_mul(128)?)?,
        terms: old.terms.checked_add(c)?,
        // Minima + integrity copy, bounded staging/IR buffers and header growth.
        // Nominal Vec payload, not allocator overhead/RSS. No new mask tables.
        bytes: old
            .terms
            .checked_add(c)?
            .checked_mul(64)?
            .checked_add(c.checked_mul(256)?)?
            .checked_add(p.c.iter().filter(|v| **v < 0.).count().checked_mul(64)?)?
            .checked_add(r.checked_mul(256)?)?
            .checked_add(4096)?,
    })
}
pub(super) fn admit(cuts: &HullRootMipCuts<'_>, plan: &Plan, b: &Budget) {
    {
        let Ok(mut c) = b.state.lock() else {
            return;
        };
        if !c.component_rows.admitted || c.component_rows.phase.attempted || c.replay_attempts != 0
        {
            return;
        }
        c.component_rows.phase.attempted = true;
        c.component_rows.phase.stop_reason = "setup";
    }
    let Some(setup) = setup(cuts, plan) else {
        stop(b, "shape");
        return;
    };
    if b.charge(setup).is_none() {
        stop(b, "setup-work");
        return;
    }
    if let Ok(mut c) = b.state.lock() {
        c.component_rows.phase.work_units = setup;
        let Some(total) = c.component_rows.work_units.checked_add(setup) else {
            return;
        };
        c.component_rows.work_units = total;
    }
    if b.phase_all_enabled && all::admit(cuts, plan, b, setup) {
        return;
    }
    let Some(q) = quote(cuts, plan) else {
        stop(b, "shape");
        return;
    };
    let Some(held) = q.work.checked_sub(setup) else {
        stop(b, "shape");
        return;
    };
    let Ok(mut c) = b.state.lock() else {
        return;
    };
    c.component_rows.phase.quoted_rows = plan.components.len();
    c.component_rows.phase.quoted_terms = q.terms;
    let total = b
        .admitted_bytes
        .checked_add(c.shared_bytes)
        .and_then(|v| v.checked_add(c.direction.cache_bytes))
        .and_then(|v| v.checked_add(c.child.cache_bytes))
        .and_then(|v| v.checked_add(c.component_rows.admitted_bytes))
        .and_then(|v| v.checked_add(q.bytes));
    if total.is_none_or(|v| v > MAX_BYTES) {
        c.component_rows.phase.stop_reason = "payload";
        return;
    }
    let next = c.component_reserved.checked_add(held);
    let funded = next.and_then(|v| c.work.checked_add(c.reserved)?.checked_add(v));
    if funded.is_none_or(|v| v > MAX_WORK) {
        c.component_rows.phase.stop_reason = "work";
        return;
    }
    if !live(b.search_end) {
        c.component_rows.phase.stop_reason = "deadline";
        return;
    }
    let Some(bytes) = c.component_rows.admitted_bytes.checked_add(q.bytes) else {
        return;
    };
    c.component_reserved = next.unwrap();
    c.component_rows.reserved_work_units = c.component_reserved;
    c.component_rows.admitted_bytes = bytes;
    c.component_rows.phase.admitted = true;
    c.component_rows.phase.admitted_bytes = q.bytes;
    c.component_rows.phase.reserved_work_units = held;
    c.component_rows.phase.stop_reason = "reserved";
}
/// The enclosing component begin has already irreversibly paid this hold.
pub(super) fn begin(c: &mut Counts) -> Option<()> {
    let report = &mut c.component_rows.phase;
    if report.admitted {
        report.work_units = report.work_units.checked_add(report.reserved_work_units)?;
        report.reserved_work_units = 0;
        report.collection_attempts += 1;
        report.stop_reason = "collecting";
    }
    cube::begin(c)?;
    Some(())
}
pub(super) fn collecting(b: &Budget) -> bool {
    b.state.lock().is_ok_and(|c| {
        c.component_rows.phase.admitted
            && c.component_rows.phase.collection_attempts == 1
            && c.component_rows.phase.reserved_work_units == 0
            && c.source_replays == 0
    })
}
/// Private producer-only payload. The integrity copy catches accidental private
/// mutation; it is NOT independent proof. Authority is the existing full replay
/// and final source/witness/G/constant/root-bit checks plus model binding.
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts) struct Conditional {
    source: usize,
    values: Vec<Vec<[f64; 2]>>,
    integrity: Vec<u64>,
    ledger: Option<Arc<std::sync::Mutex<Counts>>>,
    witness: Option<Arc<RetainedWords>>,
}
impl Conditional {
    fn bits(&self) -> impl Iterator<Item = u64> + '_ {
        [self.source as u64, self.values.len() as u64]
            .into_iter()
            .chain(self.values.iter().flat_map(|v| {
                [v.len() as u64]
                    .into_iter()
                    .chain(v.iter().flat_map(|p| p.iter().map(|x| x.to_bits())))
            }))
    }
    /// Invoked only by the completed all-mask producer, not by row consumers.
    pub(super) fn finish(cuts: &HullRootMipCuts<'_>, values: Vec<Vec<[f64; 2]>>) -> Option<Self> {
        if values
            .iter()
            .any(|v| v.is_empty() || v.len() > 8 || v.iter().flatten().any(|x| !x.is_finite()))
        {
            return None;
        }
        let count = values
            .iter()
            .try_fold(2usize, |n, v| n.checked_add(1 + 2 * v.len()))?;
        let mut out = Self {
            source: std::ptr::from_ref(cuts.problem) as usize,
            values,
            integrity: Vec::new(),
            ledger: None,
            witness: None,
        };
        let mut integrity = Vec::with_capacity(count);
        integrity.extend(out.bits());
        out.integrity = integrity;
        Some(out)
    }
    pub(super) fn bind(&mut self, b: &Budget, witness: &Arc<RetainedWords>) {
        self.ledger = Some(b.state.clone());
        self.witness = Some(witness.clone());
    }
    fn valid(&self, row: &SignedMajorant, minima: &[f64]) -> bool {
        self.ledger
            .as_ref()
            .is_some_and(|v| Arc::ptr_eq(v, &row.budget.state))
            && self
                .witness
                .as_ref()
                .is_some_and(|v| Arc::ptr_eq(v, &row.witness))
            && self.source == row.source
            && self.values.len() == row.plan.components.len()
            && minima.len() == self.values.len()
            && self.bits().eq(self.integrity.iter().copied())
            && self
                .values
                .iter()
                .zip(&row.plan.components)
                .zip(minima)
                .all(|((v, c), minimum)| {
                    v.len() == c.negative.len()
                        && v.iter().all(|p| {
                            p[0].is_finite() && p[1].is_finite() && p[0].min(p[1]) == *minimum
                        })
                })
    }
}
pub(super) fn reset(b: &Budget) {
    if let Ok(mut c) = b.state.lock() {
        let r = &mut c.component_rows.phase;
        r.prepared_rows = 0;
        r.prepared_terms = 0;
        r.prepared_head_terms = 0;
        c.phase_all.pending = false;
        c.phase_all.parts = 0;
        c.phase_all.original_rows = 0;
        c.phase_all.prepared_rows = 0;
        c.phase_all.witness = None;
        c.component_rows.phase.reserved_ir_rows = 0;
        cube::reset(&mut c);
    }
}
/// Directed algebra only; mask/source authority belongs to Conditional's producer.
pub(super) fn plane(minimum: f64, pair: [f64; 2]) -> Option<(f64, f64, f64)> {
    if pair.iter().any(|v| !v.is_finite()) || pair[0].min(pair[1]) != minimum {
        return None;
    }
    let favored = usize::from(pair[1] > pair[0]);
    let delta = finite(sub_down(pair[favored], minimum))?.max(0.);
    let coefficient = if favored == 0 {
        2. * delta
    } else {
        -2. * delta
    };
    let base = if favored == 0 {
        add_down(minimum, delta)
    } else {
        minimum
    };
    Some((delta, finite(coefficient)?, finite(2. * base)?))
}
/// Only the authenticated binding path calls this after the full first replay.
/// Any optional coverage/overflow refusal preserves ALL old component rows.
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
    if !binding.matches(cuts, parts) {
        return None;
    }
    let end = end.min(row.authority()).min(cuts.deadline);
    if all_mode(&row.budget) {
        return all::prepare(
            row,
            cuts,
            parts,
            binding,
            conditional,
            minima,
            old,
            end,
            poll,
        );
    }
    let mut rows = Vec::with_capacity(old.len());
    if !conditional.valid(row, minima) || old.len() != row.plan.components.len() {
        stop(&row.budget, "coverage");
        return Some(rows);
    }
    let mut terms = 0usize;
    let mut heads = 0usize;
    let limit = row
        .budget
        .state
        .lock()
        .ok()?
        .component_rows
        .phase
        .quoted_terms;
    for (index, component) in row.plan.components.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        let mut chosen = None;
        for (bit, &head) in component.negative.iter().enumerate() {
            let Some(column) = binding.column(head) else {
                continue;
            };
            let Some((delta, coefficient, lower)) =
                plane(minima[index], conditional.values[index][bit])
            else {
                stop(&row.budget, "overflow");
                return Some(Vec::new());
            };
            if delta > 0. && chosen.as_ref().is_none_or(|(_, old, _, _)| delta > *old) {
                chosen = Some((column, delta, coefficient, lower));
            }
        }
        let Some((column, _, coefficient, lower)) = chosen else {
            continue;
        };
        // old[index] was constructed from the same checked component/minimum.
        let mut coeffs = Vec::with_capacity(old[index].coeffs.len().checked_add(1)?);
        coeffs.extend_from_slice(&old[index].coeffs);
        coeffs.push((column.0, coefficient));
        terms = terms.checked_add(coeffs.len())?;
        heads = heads
            .checked_add(component.positive.len())?
            .checked_add(component.negative.len())?;
        if terms > limit {
            stop(&row.budget, "shape");
            return Some(Vec::new());
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
    let mut c = row.budget.state.lock().ok()?;
    let r = &mut c.component_rows.phase;
    r.prepared_rows = rows.len();
    r.prepared_head_terms = heads;
    r.prepared_terms = terms;
    r.stop_reason = if rows.is_empty() {
        "no-gap"
    } else {
        "prepared"
    };
    Some(rows)
}
impl SignedMajorant {
    pub(in crate::single_hidden_hull::mip_cuts) fn prepare_rows(
        &self,
        cuts: &HullRootMipCuts<'_>,
        parts: &MipParts,
        columns: &[crate::ir::Col],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Vec<RowSpec>> {
        self.prepare_rows_inner(cuts, parts, columns, None, end, poll)
    }
    pub(in crate::single_hidden_hull::mip_cuts) fn prepare_rows_bound(
        &self,
        cuts: &HullRootMipCuts<'_>,
        parts: &MipParts,
        columns: &[crate::ir::Col],
        binding: &binding::SignedPhaseBinding<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Vec<RowSpec>> {
        self.prepare_rows_inner(cuts, parts, columns, Some(binding), end, poll)
    }
    fn prepare_rows_inner(
        &self,
        cuts: &HullRootMipCuts<'_>,
        parts: &MipParts,
        columns: &[crate::ir::Col],
        binding: Option<&binding::SignedPhaseBinding<'_>>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Vec<RowSpec>> {
        component_rows::reset_prepared(&self.budget);
        let row = self.replay_inner(cuts, end, poll, false)?;
        if let Some(binding) = binding {
            let preferred = row
                .cube
                .as_ref()
                .and_then(cube::Collected::native_component);
            native_publication::begin(self, cuts, parts, binding, preferred, end, poll)?;
        }
        let mut coeffs = Vec::new();
        for (j, &c) in cuts.problem.c.iter().enumerate() {
            if c != 0. {
                coeffs.push((columns.get(j)?.0, c));
            }
        }
        for (i, &g) in row.g.iter().enumerate() {
            if g != 0. {
                coeffs.push((parts.input_vars.get(i)?.0, -g));
            }
        }
        let mut rows = vec![RowSpec {
            lb: row.constant,
            ub: f64::INFINITY,
            coeffs,
        }];
        if let Some(minima) = row.component_minima {
            match component_rows::prepare(self, cuts, parts, columns, &minima, end, poll)? {
                component_rows::Prepared::Rows(extra) => {
                    if let (Some(binding), Some(conditional)) = (binding, row.conditional.as_ref())
                    {
                        let lift = prepare(
                            self,
                            cuts,
                            parts,
                            binding,
                            conditional,
                            &minima,
                            &extra,
                            end,
                            poll,
                        )?;
                        let cubes = match row.cube.as_ref() {
                            Some(table) => cube::prepare(
                                self, cuts, parts, binding, table, &minima, &extra, end, poll,
                            )?,
                            None => Vec::new(),
                        };
                        rows.reserve_exact(
                            extra
                                .len()
                                .checked_add(lift.len())?
                                .checked_add(cubes.len())?,
                        );
                        rows.extend(extra);
                        rows.extend(lift);
                        rows.extend(cubes);
                    } else {
                        rows.extend(extra);
                    }
                }
                component_rows::Prepared::Refused => {}
            }
        }
        if let Some(binding) = binding {
            // These opaque recipes were produced by a separate already-paid
            // complete mass check, not by the winner's minima/phase counters.
            let portfolio = portfolio::prepare(self, cuts, parts, binding, end, poll);
            rows.reserve_exact(portfolio.len());
            rows.extend(portfolio);
            // A funded All publication owns its complete R envelope even if
            // the earlier root proposal was not admitted. It is exclusive of
            // later optional domination/union work; unfunded One stays legacy.
            if !all_mode(&self.budget) {
                rows.extend(joint::prepare(self, cuts, parts, binding, end, poll));
            }
        }
        (live(end.min(self.authority())) && poll()).then_some(rows)
    }
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_component_phase_test_fixture.rs"]
mod test_fixture;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_component_phase_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_component_phase_all_tests.rs"]
mod all_tests;
