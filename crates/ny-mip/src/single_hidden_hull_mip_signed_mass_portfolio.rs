// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Checked component outputs from the already paid final mass pass.
//! A losing scalar may retain valid planes. No alpha table is retained.
use super::*;
use std::sync::{Mutex, Weak};
#[path = "single_hidden_hull_mip_signed_mass_conditional.rs"]
pub(super) mod conditional;
#[path = "single_hidden_hull_mip_signed_mass_cube.rs"]
mod cube;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SignedMassPortfolioReport {
    pub attempted: bool,
    pub admitted: bool,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub ir_row_limit: usize,
    pub reserved_ir_rows: usize,
    pub collection_attempts: usize,
    pub completed_collections: usize,
    pub retained: bool,
    pub candidate_lower: Option<f64>,
    pub scalar_threshold: Option<f64>,
    pub export_attempts: usize,
    pub prepared_rows: usize,
    pub prepared_head_terms: usize,
    pub prepared_terms: usize,
    pub appended_rows: usize,
    pub appended_head_terms: usize,
    pub appended_terms: usize,
    pub phase_attempted: bool,
    pub phase_admitted: bool,
    pub phase_work_units: usize,
    pub phase_admitted_bytes: usize,
    pub phase_collections: usize,
    pub phase_prepared_rows: usize,
    pub phase_appended_rows: usize,
    pub phase_prepared_terms: usize,
    pub phase_appended_terms: usize,
    pub cube: SignedPhaseCubeReport,
    pub stop_reason: &'static str,
}

#[derive(Debug, Default)]
pub(super) struct State {
    report: SignedMassPortfolioReport,
    recipe: Option<Recipe>,
    pending: bool,
    headers_reserved: bool,
    prepared_row_bound: usize,
    prepared_parts: usize,
    prepared_original_rows: usize,
    prepared_original_limit: usize,
}
impl State {
    pub(super) fn report(&self) -> &SignedMassPortfolioReport {
        &self.report
    }
}

#[derive(Debug, Clone, Copy)]
struct Quote {
    components: usize,
    terms: usize,
    rows: usize,
    original_rows: usize,
    cost: usize,
    work: usize,
    bytes: usize,
}

// The permit is private, non-Clone, prepaid, and consumed by check_final.
pub(super) struct Permit {
    source: usize,
    context: usize,
    dimensions: (usize, usize, usize),
    ledger: Weak<Mutex<Counts>>,
    signature: Arc<RetainedWords>,
    authority: Instant,
    search_end: Instant,
    deadline: Instant,
    quote: Quote,
    final_paid: bool,
    conditional: Option<conditional::Quote>,
    cube: Option<cube::Quote>,
}

#[derive(Debug)]
struct Plane {
    // Entire row is doubled. A head may map to a free affine column.
    heads: Vec<(usize, f64)>,
    input: Vec<f64>,
    lower: f64,
}

#[derive(Debug)]
struct Recipe {
    source: usize,
    dimensions: (usize, usize, usize),
    ledger: Weak<Mutex<Counts>>,
    signature: Arc<RetainedWords>,
    authority: Instant,
    search_end: Instant,
    deadline: Instant,
    quote: Quote,
    planes: Vec<Plane>,
    conditional: Option<conditional::Recipe>,
    seal: Box<[u64]>,
}

fn note(b: &Budget, f: impl FnOnce(&mut SignedMassPortfolioReport)) {
    if let Ok(mut c) = b.state.lock() {
        f(&mut c.portfolio.report);
    }
}
fn stop(b: &Budget, reason: &'static str) {
    note(b, |r| r.stop_reason = reason);
}
pub(super) fn report(b: &Budget) -> SignedMassPortfolioReport {
    b.state
        .lock()
        .map(|c| c.portfolio.report.clone())
        .unwrap_or_default()
}

fn layout(plan: &Plan, b: &Budget) -> Option<(usize, usize)> {
    let (d, h, p) = b.dimensions;
    let count = plan.components.len();
    if count == 0 || count > h || plan.leftover.len() > h {
        return None;
    }
    let (mut negative, mut positive) = (0usize, 0usize);
    for c in &plan.components {
        let k = c.negative.len();
        let q = c.positive.len();
        if !(1..=8).contains(&k)
            || q > MAX_POSITIVE
            || c.positive_units.len() != q
            || c.g.len() != d
            || c.alpha.len() != (1usize << k).checked_mul(q)?
            || c.negative.iter().chain(&c.positive).any(|j| *j >= h)
            || c.positive_units.iter().any(|u| !matches!(u, 1 | 2))
        {
            return None;
        }
        negative = negative.checked_add(k)?;
        positive = positive.checked_add(q)?;
    }
    if negative > h || positive > p.checked_mul(2)? {
        return None;
    }
    let terms = count
        .checked_mul(d)?
        .checked_add(p.checked_mul(2)?)?
        .checked_add(negative)?;
    Some((count, terms))
}

fn quote(cuts: &HullRootMipCuts<'_>, plan: &Plan, b: &Budget) -> Option<Quote> {
    let (components, terms) = layout(plan, b)?;
    let (d, h, _) = b.dimensions;
    let metadata = components.checked_add(h)?.checked_add(d)?.checked_add(1)?;
    // Existing root ownership excludes append-time optional joint rows.
    // The signed global, winner components/phase and portfolio add <=1+3C.
    let rows = cuts
        .len()
        .checked_add(1)?
        .checked_add(components.checked_mul(3)?)?;
    let original_rows = h.checked_add(d)?.checked_add(1)?.checked_mul(8)?;
    Some(Quote {
        components,
        terms,
        rows,
        original_rows,
        cost: plan.cost(cuts.problem)?,
        work: terms
            .checked_add(metadata)?
            .checked_add(rows)?
            .checked_mul(256)?
            .checked_add(metadata.checked_mul(64)?)?
            .checked_add(original_rows.checked_mul(128)?)?
            .checked_add(4096)?,
        // Canonical recipe+seal, minima, mapped and IR buffers, row headers,
        // bounded index scratch and the fixed permit/report control block.
        bytes: terms
            .checked_mul(96)?
            .checked_add(components.checked_mul(384)?)?
            .checked_add(h.checked_add(d)?.checked_add(1)?.checked_mul(128)?)?
            .checked_add(rows.checked_mul(256)?)?
            .checked_add(original_rows.checked_mul(128)?)?
            .checked_add(8192)?,
    })
}

/// Called after sharing admission, before its optional advice is allocated.
/// Both work and persistent bytes are irreversible even if the pass aborts.
pub(super) fn admit(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    signature: &Arc<RetainedWords>,
    minimum_advice: usize,
) -> Option<Permit> {
    if !b.portfolio_enabled
        || b.root_coupling.is_none()
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.authority > cuts.deadline
        || !live(b.search_end)
        || plan.components.len() > b.dimensions.1
    {
        return None;
    }
    {
        let mut c = b.state.lock().ok()?;
        if c.portfolio.report.attempted
            || !c.sharing_attempted
            || c.shared_bytes == 0
            || c.replay_attempts != 0
            || c.source_replays != 0
        {
            return None;
        }
        c.portfolio.report.attempted = true;
        c.portfolio.report.stop_reason = "metadata";
    }
    let setup = plan
        .components
        .len()
        .checked_add(b.dimensions.1)?
        .checked_add(b.dimensions.0)?
        .checked_add(1)?
        .checked_mul(64)?;
    b.charge(setup)?;
    note(b, |r| r.work_units = setup);
    let q = quote(cuts, plan, b)?;
    let rest = q.work.checked_sub(setup)?;
    let mut c = b.state.lock().ok()?;
    let work = c.work.checked_add(rest)?;
    let shared = c.shared_bytes.checked_add(q.bytes)?;
    let bytes = b
        .admitted_bytes
        .checked_add(shared)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.child.cache_bytes)?
        .checked_add(c.component_rows.admitted_bytes)?;
    let protected = q.cost.checked_mul(2)?.checked_add(minimum_advice)?;
    if work
        .checked_add(c.reserved)?
        .checked_add(c.component_reserved)?
        .checked_add(protected)?
        > MAX_WORK
        || bytes > MAX_BYTES
        || !live(b.search_end)
    {
        c.portfolio.report.stop_reason = "work-payload-or-deadline";
        return None;
    }
    c.work = work;
    // Sharing has already admitted once; this additive sub-debt is never
    // overwritten or refunded and every later payload guard sees it.
    c.shared_bytes = shared;
    c.portfolio.report.admitted = true;
    c.portfolio.report.work_units = q.work;
    c.portfolio.report.admitted_bytes = q.bytes;
    c.portfolio.report.ir_row_limit = q.original_rows;
    c.portfolio.report.stop_reason = "prepaid";
    let mut permit = Permit {
        source: b.source,
        context: std::ptr::from_ref(cuts) as usize,
        dimensions: b.dimensions,
        ledger: Arc::downgrade(&b.state),
        signature: signature.clone(),
        authority: b.authority,
        search_end: b.search_end,
        deadline: cuts.deadline,
        quote: q,
        final_paid: false,
        conditional: None,
        cube: None,
    };
    drop(c);
    let _ = conditional::upgrade(cuts, plan, b, minimum_advice, &mut permit);
    let _ = cube::upgrade(cuts, plan, b, minimum_advice, &mut permit);
    Some(permit)
}

fn owned(permit: &Permit, cuts: &HullRootMipCuts<'_>, b: &Budget) -> bool {
    permit.source == b.source
        && permit.source == std::ptr::from_ref(cuts.problem) as usize
        && permit.context == std::ptr::from_ref(cuts) as usize
        && permit.dimensions == b.dimensions
        && permit.authority == b.authority
        && permit.search_end == b.search_end
        && permit.deadline == cuts.deadline
        && permit
            .ledger
            .upgrade()
            .is_some_and(|v| Arc::ptr_eq(&v, &b.state))
}

/// Wrap the existing mass final-check charge, before its advice. No new slot
/// is allocated: disabled/declined portfolios take the exact old charge.
pub(super) fn pay_final(
    cuts: &HullRootMipCuts<'_>,
    b: &Budget,
    cost: usize,
    permit: Option<&mut Permit>,
) -> Option<()> {
    if let Some(p) = permit.as_ref() {
        if !owned(p, cuts, b) || p.final_paid || p.quote.cost != cost {
            return None;
        }
    }
    b.proof(cost, false)?;
    if let Some(p) = permit {
        p.final_paid = true;
    }
    Some(())
}

impl Recipe {
    fn bits(&self) -> impl Iterator<Item = u64> + '_ {
        [
            self.source as u64,
            self.dimensions.0 as u64,
            self.dimensions.1 as u64,
            self.dimensions.2 as u64,
            self.planes.len() as u64,
            self.quote.terms as u64,
            self.quote.rows as u64,
            self.quote.original_rows as u64,
        ]
        .into_iter()
        .chain(self.planes.iter().flat_map(|p| {
            [
                p.heads.len() as u64,
                p.input.len() as u64,
                p.lower.to_bits(),
            ]
            .into_iter()
            .chain(p.heads.iter().flat_map(|(j, c)| [*j as u64, c.to_bits()]))
            .chain(p.input.iter().map(|g| g.to_bits()))
        }))
    }
    fn finish(
        permit: Permit,
        plan: &Plan,
        minima: &[f64],
        cuts: &HullRootMipCuts<'_>,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        if minima.len() != permit.quote.components || minima.iter().any(|m| !m.is_finite()) {
            return None;
        }
        let mut planes = Vec::with_capacity(minima.len());
        let mut terms = 0usize;
        for (c, &minimum) in plan.components.iter().zip(minima) {
            if !live(permit.search_end) || !poll() {
                return None;
            }
            let mut heads = Vec::with_capacity(c.negative.len().checked_add(c.positive.len())?);
            for &j in &c.negative {
                let coefficient = *cuts.problem.c.get(j)?;
                if coefficient.partial_cmp(&0.) != Some(std::cmp::Ordering::Less) {
                    return None;
                }
                heads.push((j, finite(2. * coefficient)?));
            }
            for (slot, &j) in c.positive.iter().enumerate() {
                let coefficient = *cuts.problem.c.get(j)?;
                if coefficient.partial_cmp(&0.) != Some(std::cmp::Ordering::Greater) {
                    return None;
                }
                heads.push((j, finite(coefficient * f64::from(c.positive_units[slot]))?));
            }
            let mut input = Vec::with_capacity(c.g.len());
            for &g in &c.g {
                input.push(finite(-2. * g)?);
            }
            terms = terms.checked_add(heads.len())?.checked_add(input.len())?;
            if terms > permit.quote.terms {
                return None;
            }
            planes.push(Plane {
                heads,
                input,
                lower: finite(2. * minimum)?,
            });
        }
        Some(Self {
            source: permit.source,
            dimensions: permit.dimensions,
            ledger: permit.ledger,
            signature: permit.signature,
            authority: permit.authority,
            search_end: permit.search_end,
            deadline: permit.deadline,
            quote: permit.quote,
            planes,
            conditional: None,
            seal: Box::new([]),
        })
    }
}

/// This function invokes the existing paid whole Source kernel itself. It
/// never accepts caller-supplied minima as proof, nor charges another checker.
pub(super) fn check_final(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    permit: Option<Permit>,
    scalar_threshold: f64,
    poll: &mut dyn FnMut() -> bool,
) -> Option<certificate::Checked> {
    let Some(permit) = permit else {
        return certificate::check(cuts, plan, b, b.search_end, poll);
    };
    if !owned(&permit, cuts, b)
        || !permit.final_paid
        || !live(permit.search_end)
        || !scalar_threshold.is_finite()
        || layout(plan, b)? != (permit.quote.components, permit.quote.terms)
        || plan.cost(cuts.problem)? != permit.quote.cost
    {
        return None;
    }
    {
        let mut c = b.state.lock().ok()?;
        if !c.portfolio.report.admitted
            || c.portfolio.report.collection_attempts != 0
            || c.replay_attempts != 0
            || c.source_replays != 0
            || c.component_rows.phase.collection_attempts != 0
        {
            return None;
        }
        c.portfolio.report.collection_attempts = 1;
    }
    let mut collector = conditional::collector(&permit, cuts, plan, b);
    let checked = if let Some(collector) = collector.as_mut() {
        certificate::check_portfolio(cuts, plan, b, permit.search_end, poll, collector)?
    } else {
        certificate::check_components(cuts, plan, b, permit.search_end, poll)?
    };
    note(b, |r| {
        r.completed_collections = 1;
        r.phase_collections = usize::from(collector.is_some());
        r.candidate_lower = Some(checked.root_lower);
        r.scalar_threshold = Some(scalar_threshold);
    });
    // A selected/equal scalar uses the normal publication owner. This first
    // policy retains an extra family only for a strictly losing scalar.
    if checked.root_lower >= scalar_threshold {
        if !live(b.search_end) || !poll() || require_hull_environment().is_err() {
            stop(b, "completion-refused");
            return None;
        }
        stop(b, "not-a-strict-scalar-loser");
        return Some(checked);
    }
    let mut cancelled = false;
    let mut guard = || {
        let keep = poll();
        cancelled |= !keep;
        keep
    };
    let recipe = (|| {
        let minima = checked.component_minima.as_deref()?;
        let mut recipe = Recipe::finish(permit, plan, minima, cuts, &mut guard)?;
        recipe.conditional = collector.and_then(|values| values.finish(&recipe.planes));
        let words = 8usize.checked_add(recipe.planes.iter().try_fold(0usize, |n, p| {
            n.checked_add(3)?
                .checked_add(p.heads.len().checked_mul(2)?)?
                .checked_add(p.input.len())
        })?)?;
        let mut seal = Vec::with_capacity(words);
        seal.extend(recipe.bits());
        recipe.seal = seal.into_boxed_slice();
        Some(recipe)
    })();
    if cancelled || !live(b.search_end) || !poll() || require_hull_environment().is_err() {
        stop(b, "completion-refused");
        return None;
    }
    let mut c = b.state.lock().ok()?;
    if let Some(recipe) = recipe {
        if recipe.conditional.as_ref().is_some_and(|r| r.has_cube()) {
            let r = &mut c.portfolio.report.cube;
            r.completed_tables = r.quoted_rows;
            r.collected_masks = r.quoted_masks;
            r.stop_reason = "collected";
        }
        c.portfolio.recipe = Some(recipe);
        c.portfolio.report.retained = true;
        c.portfolio.report.stop_reason = "checked-output-retained";
    } else {
        // Doubling/numeric refusal is optional; the valid scalar Checked
        // remains available after the final original-clock barrier above.
        c.portfolio.report.stop_reason = "recipe-refused";
    }
    Some(checked)
}

/// The caller has already replayed the scalar winner and authenticated the
/// complete original encoder. The token argument makes that seam explicit.
pub(super) fn prepare(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Vec<RowSpec> {
    let b = &row.budget;
    let end = end.min(b.authority).min(cuts.deadline);
    let recipe = (|| {
        if !live(end)
            || !poll()
            || !binding.matches(cuts, parts)
            || row.source != std::ptr::from_ref(cuts.problem) as usize
        {
            return None;
        }
        let mut c = b.state.lock().ok()?;
        let recipe = c.portfolio.recipe.as_ref()?;
        if c.portfolio.report.export_attempts != 0
            || recipe.source != row.source
            || recipe.dimensions != b.dimensions
            || recipe.authority != b.authority
            || recipe.search_end != b.search_end
            || recipe.deadline != cuts.deadline
            || !Arc::ptr_eq(&recipe.signature, &row.signature)
            || !recipe
                .ledger
                .upgrade()
                .is_some_and(|v| Arc::ptr_eq(&v, &b.state))
        {
            return None;
        }
        c.portfolio.report.export_attempts = 1;
        c.portfolio.recipe.take()
    })();
    let Some(recipe) = recipe else {
        return Vec::new();
    };
    let rows = (|| {
        if recipe.planes.len() != recipe.quote.components
            || recipe.planes.len() > b.dimensions.1
            || parts.input_vars.len() != b.dimensions.0
            || parts.num_cols != parts.problem.num_cols()
            || parts.problem.num_rows() > recipe.quote.original_rows
        {
            return None;
        }
        if cuts
            .len()
            .checked_add(recipe.planes.len().checked_mul(3)?)?
            > recipe.quote.rows
        {
            return None;
        }
        let mut terms = 0usize;
        for p in &recipe.planes {
            if p.heads.len() > 8 + MAX_POSITIVE || p.input.len() != b.dimensions.0 {
                return None;
            }
            terms = terms
                .checked_add(p.heads.len())?
                .checked_add(p.input.len())?;
        }
        if terms > recipe.quote.terms || !recipe.bits().eq(recipe.seal.iter().copied()) {
            return None;
        }
        let mut rows = Vec::with_capacity(recipe.planes.len());
        let mut prepared_terms = 0usize;
        let mut head_terms = 0usize;
        for p in &recipe.planes {
            if !live(end) || !poll() {
                return None;
            }
            let mut coeffs = Vec::with_capacity(p.heads.len().checked_add(p.input.len())?);
            for &(j, coefficient) in &p.heads {
                let col = binding.head_column(j)?;
                coeffs.push((col.0, finite(coefficient)?));
                head_terms = head_terms.checked_add(1)?;
            }
            for (i, &g) in p.input.iter().enumerate() {
                if !live(end) || !poll() {
                    return None;
                }
                if finite(g)? != 0. {
                    let col = *parts.input_vars.get(i)?;
                    parts.problem.cols().get(col.0)?;
                    coeffs.push((col.0, g));
                }
            }
            prepared_terms = prepared_terms.checked_add(coeffs.len())?;
            rows.push(RowSpec {
                lb: finite(p.lower)?,
                ub: f64::INFINITY,
                coeffs,
            });
        }
        let mut cancelled = false;
        let extra = recipe.conditional.as_ref().and_then(|family| {
            family.prepare(&recipe.planes, &rows, binding, end, &mut || {
                let ok = !cancelled && poll();
                cancelled |= !ok;
                ok
            })
        });
        let multibit = recipe.conditional.as_ref().and_then(|family| {
            family.prepare_cube(&recipe.planes, &rows, binding, end, &mut || {
                let ok = !cancelled && poll();
                cancelled |= !ok;
                ok
            })
        });
        if cancelled {
            return None;
        }
        let mut phase_rows = 0usize;
        let mut phase_terms = 0usize;
        if let Some((extra, extra_heads)) = extra {
            phase_rows = extra.len();
            phase_terms = extra.iter().map(|r| r.coeffs.len()).sum();
            prepared_terms = prepared_terms.checked_add(phase_terms)?;
            head_terms = head_terms.checked_add(extra_heads)?;
            rows.reserve_exact(extra.len());
            rows.extend(extra);
        }
        let mut cube_counts = (0, 0, 0, 0, 0);
        if let Some(extra) = multibit {
            cube_counts = (
                extra.rows.len(),
                extra.terms,
                extra.selectors,
                extra.advised,
                extra.projected,
            );
            prepared_terms = prepared_terms.checked_add(extra.terms)?;
            head_terms = head_terms.checked_add(extra.heads)?;
            rows.reserve_exact(extra.rows.len());
            rows.extend(extra.rows);
        }
        if !live(end) || !poll() || require_hull_environment().is_err() {
            return None;
        }
        let mut c = b.state.lock().ok()?;
        c.portfolio.report.phase_prepared_rows = phase_rows;
        c.portfolio.report.phase_prepared_terms = phase_terms;
        let r = &mut c.portfolio.report.cube;
        r.prepare_attempts = usize::from(recipe.conditional.as_ref().is_some_and(|f| f.has_cube()));
        r.prepared_rows = cube_counts.0;
        r.prepared_terms = cube_counts.1;
        r.prepared_selector_terms = cube_counts.2;
        r.activity_components = cube_counts.3;
        r.projected_components = cube_counts.4;
        c.portfolio.report.prepared_rows = rows.len();
        c.portfolio.report.prepared_head_terms = head_terms;
        c.portfolio.report.prepared_terms = prepared_terms;
        c.portfolio.pending = true;
        c.portfolio.prepared_row_bound = recipe.quote.rows;
        c.portfolio.prepared_parts = std::ptr::from_ref(parts) as usize;
        c.portfolio.prepared_original_rows = parts.problem.num_rows();
        c.portfolio.prepared_original_limit = recipe.quote.original_rows;
        c.portfolio.report.stop_reason = "prepared";
        Some(rows)
    })();
    rows.unwrap_or_else(|| {
        stop(b, "export-refused");
        Vec::new()
    })
}

/// Reserve the complete append's row headers once, only for an actually
/// staged portfolio. The original prefix and all staged headers are prepaid.
/// No failure follows this reservation before the caller's row commit loop.
pub(super) fn reserve_ir_headers(
    row: &SignedMajorant,
    parts: &mut MipParts,
    additional: usize,
) -> Option<()> {
    if claim_ir_headers(row, parts, additional)? {
        parts.problem.reserve_rows_exact(additional);
    }
    Some(())
}

/// Validate and consume only; a richer phase owner may perform the ONE reserve.
pub(super) fn claim_ir_headers(
    row: &SignedMajorant,
    parts: &MipParts,
    additional: usize,
) -> Option<bool> {
    claim_ir_headers_inner(row, parts, additional, None)
}
pub(super) fn claim_ir_headers_with_phase(
    row: &SignedMajorant,
    parts: &MipParts,
    additional: usize,
    grant: &phase::HeaderGrant,
) -> Option<bool> {
    if !grant.matches(row, parts, additional) {
        return None;
    }
    claim_ir_headers_inner(row, parts, additional, Some(grant))
}
pub(super) fn record_ir_headers_with_phase(
    row: &SignedMajorant,
    parts: &MipParts,
    additional: usize,
    grant: &phase::HeaderGrant,
) {
    if !grant.context_matches(row, parts, additional) {
        return;
    }
    if let Ok(mut c) = row.budget.state.lock() {
        if c.portfolio.pending && c.portfolio.headers_reserved {
            c.portfolio.report.reserved_ir_rows = additional;
        }
    }
}
fn claim_ir_headers_inner(
    row: &SignedMajorant,
    parts: &MipParts,
    additional: usize,
    grant: Option<&phase::HeaderGrant>,
) -> Option<bool> {
    let b = &row.budget;
    let mut c = b.state.lock().ok()?;
    if !c.portfolio.pending {
        return Some(false);
    }
    if c.portfolio.headers_reserved
        || row.source != b.source
        || c.portfolio.prepared_parts != std::ptr::from_ref(parts) as usize
        || c.portfolio.prepared_original_rows != parts.problem.num_rows()
        || !c.portfolio.report.retained
        || c.portfolio.report.prepared_rows == 0
        || parts.problem.num_rows() > c.portfolio.prepared_original_limit
        || (additional > c.portfolio.prepared_row_bound
            && !grant.is_some_and(|g| g.matches(row, parts, additional)))
    {
        return None;
    }
    // State is set before allocation; a private second call cannot renew it.
    c.portfolio.headers_reserved = true;
    if grant.is_none() {
        // Preserve the existing standalone path's counter order.
        c.portfolio.report.reserved_ir_rows = additional;
    }
    Some(true)
}

/// Drop the live binding, retaining completed preparation and consumed claims.
pub(super) fn abort_prepared(b: &Budget) {
    if let Ok(mut c) = b.state.lock() {
        c.portfolio.pending = false;
        c.portfolio.prepared_row_bound = 0;
        c.portfolio.prepared_parts = 0;
        c.portfolio.prepared_original_rows = 0;
        c.portfolio.prepared_original_limit = 0;
    }
}

pub(super) fn reset_prepared(b: &Budget) {
    if let Ok(mut c) = b.state.lock() {
        c.portfolio.report.phase_prepared_rows = 0;
        c.portfolio.report.phase_prepared_terms = 0;
        c.portfolio.report.phase_appended_rows = 0;
        c.portfolio.report.phase_appended_terms = 0;
        let r = &mut c.portfolio.report.cube;
        r.prepared_rows = 0;
        r.prepared_terms = 0;
        r.prepared_selector_terms = 0;
        r.appended_rows = 0;
        r.appended_terms = 0;
        r.appended_selector_terms = 0;
        c.portfolio.report.prepared_rows = 0;
        c.portfolio.report.prepared_head_terms = 0;
        c.portfolio.report.prepared_terms = 0;
        c.portfolio.report.appended_rows = 0;
        c.portfolio.report.appended_head_terms = 0;
        c.portfolio.report.appended_terms = 0;
        c.portfolio.pending = false;
        c.portfolio.headers_reserved = false;
        c.portfolio.prepared_row_bound = 0;
        c.portfolio.prepared_parts = 0;
        c.portfolio.prepared_original_rows = 0;
        c.portfolio.prepared_original_limit = 0;
        c.portfolio.report.reserved_ir_rows = 0;
    }
}
pub(super) fn commit(b: &Budget) {
    if let Ok(mut c) = b.state.lock() {
        if c.portfolio.pending {
            c.portfolio.report.phase_appended_rows = c.portfolio.report.phase_prepared_rows;
            c.portfolio.report.phase_appended_terms = c.portfolio.report.phase_prepared_terms;
            let r = &mut c.portfolio.report.cube;
            r.appended_rows = r.prepared_rows;
            r.appended_terms = r.prepared_terms;
            r.appended_selector_terms = r.prepared_selector_terms;
            if r.appended_rows != 0 {
                r.stop_reason = "committed";
            }
            c.portfolio.report.appended_rows = c.portfolio.report.prepared_rows;
            c.portfolio.report.appended_head_terms = c.portfolio.report.prepared_head_terms;
            c.portfolio.report.appended_terms = c.portfolio.report.prepared_terms;
            c.portfolio.pending = false;
            c.portfolio.report.stop_reason = "appended";
        }
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_mass_portfolio_default_tests.rs"]
mod default_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_mass_portfolio_tests.rs"]
mod tests;
