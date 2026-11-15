// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Capture the original two licensed rows and one separately prepaid extra.
//! Admission is minted only by the real additive bank debit before collection.
use super::super::binding::SignedPhaseBinding;
use super::*;
use std::sync::{Arc, Mutex, Weak};
#[path = "single_hidden_hull_mip_signed_native_bank.rs"]
pub(super) mod bank;
#[path = "single_hidden_hull_mip_signed_native_numeric.rs"]
mod numeric;
pub(super) fn reserve_during_construction(
    cuts: &HullRootMipCuts<'_>,
    b: &mut Budget,
    stage: NativeConstruction,
) -> Option<()> {
    bank::reserve(cuts, b, stage)
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SignedNativeRowsReport {
    /// Prepaid optional attempts, including refused maps/numerics.
    pub attempted: usize,
    /// Complete two/three-row batches committed to existing child LP models.
    pub committed: usize,
    /// Finite interval candidates accepted at the original callback barrier.
    /// This is not a count of winning bounds, closed nodes or solver successes.
    pub priced: usize,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SignedNativeFundingReport {
    pub funded_calls: usize,
    pub setup_work_units: usize,
    pub shared_work_units: usize,
    pub per_call_work_units: usize,
    /// Irreversible setup plus funded calls; unused credits are not refunded.
    pub prepaid_work_units: usize,
    /// Nominal peak-payload admission, not measured RSS.
    pub admitted_bytes: usize,
}
pub(super) fn stats(b: &Budget) -> SignedNativeRowsReport {
    bank::stats(b)
}
pub(super) fn funding(b: &Budget) -> SignedNativeFundingReport {
    bank::funding(b)
}
pub(super) fn fund(cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<()> {
    bank::fund(cuts, b)
}
pub(super) fn upgrade(cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<()> {
    bank::upgrade(cuts, b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Atom {
    Input(usize),
    Head(usize),
    Selector(usize),
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Term {
    atom: Atom,
    coefficient_bits: u64,
    original_column: usize,
}
impl Term {
    pub(super) fn semantic(&self) -> (Atom, f64) {
        (self.atom, f64::from_bits(self.coefficient_bits))
    }
}
#[derive(Debug)]
pub(super) struct Plane {
    lower_bits: u64,
    terms: Vec<Term>,
}
impl Plane {
    pub(super) fn lower(&self) -> f64 {
        f64::from_bits(self.lower_bits)
    }
    pub(super) fn terms(&self) -> &[Term] {
        &self.terms
    }
}

/// Immutable limits minted by the bank's real once-debited Budget admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Caps {
    row_capacity: usize,
    components: usize,
    positive: usize,
    negative: usize,
    semantic_row: usize,
    semantic_pair: usize,
    expanded_pair: usize,
    raw_expansion_pair: usize,
    batch_rows: usize,
}
#[derive(Debug)]
struct Admission {
    source: usize,
    ledger: Weak<Mutex<Counts>>,
    dimensions: (usize, usize, usize),
    authority: Instant,
    search_end: Instant,
    deadline: Instant,
    caps: Caps,
    bank: Arc<bank::Bank>,
}
#[derive(Clone, Debug)]
struct Context {
    source: usize,
    ledger: Weak<Mutex<Counts>>,
    signature: Arc<RetainedWords>,
    witness: Arc<RetainedWords>,
    authority: Instant,
    search_end: Instant,
    deadline: Instant,
    end: Instant,
    parts: usize,
    original_rows: usize,
    original_columns: usize,
    component: usize,
    extra_component: Option<usize>,
    caps: Caps,
    bank: Arc<bank::Bank>,
}

// The bank's already charged 2048-byte token allowance includes transient
// capture bookkeeping. Bound two contexts/cap copies and two borrowed producer
// descriptors, one selector array, plus 512 bytes for bounded iterators/indices.
// This is a nominal payload census, not a claim about compiler stack/RSS.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(
    2 * size_of::<Context>()
        + 2 * size_of::<Caps>()
        + 2 * size_of::<phase::NativeRow<'static>>()
        + size_of::<[usize; 8]>()
        + 512
        <= 2048
);

impl Context {
    fn owner(&self, row: &SignedMajorant) -> bool {
        self.source == row.source
            && self.source == row.budget.source
            && self.ledger.ptr_eq(&Arc::downgrade(&row.budget.state))
            && Arc::ptr_eq(&self.signature, &row.signature)
            && Arc::ptr_eq(&self.witness, &row.witness)
            && self.authority == row.budget.authority
            && self.search_end == row.budget.search_end
            && self.end <= self.authority
            && self.authority <= self.deadline
            && self.bank.matches_caps(self.caps)
    }
    fn original(&self, parts: &MipParts) -> bool {
        self.parts == std::ptr::from_ref(parts) as usize
            && self.original_rows == parts.problem.num_rows()
            && self.original_columns == parts.problem.num_cols()
            && parts.num_cols == self.original_columns
    }
}
#[derive(Debug)]
struct Payload {
    context: Context,
    unary: Option<Plane>,
    cube: Option<Plane>,
    extra: Option<Plane>,
    unary_attempted: bool,
    cube_attempted: bool,
    extra_attempted: bool,
    armed_rows: Option<usize>,
}
/// The only constructor is publish(), after the complete successful IR append.
/// Its Arc was allocated as private pending storage BEFORE the commit barrier.
#[derive(Clone, Debug)]
pub(super) struct Published {
    payload: Arc<Payload>,
}
/// Scoped consumer capability: prevents Source address reuse after the once
/// signature check. Counts may store Published, but it cannot call the pricer.
#[derive(Clone)]
pub(in crate::single_hidden_hull) struct BoundPublished<'p> {
    source: &'p HullMarginProblem,
    published: Published,
}
impl BoundPublished<'_> {
    pub(in crate::single_hidden_hull) fn end(&self) -> Instant {
        self.published.end()
    }
}
/// Advisory availability only; the full once bind below still authenticates
/// Source/signature/ledger and original clocks before returning a capability.
pub(super) fn publication_end(row: &SignedMajorant) -> Option<Instant> {
    let c = row.budget.state.lock().ok()?;
    let published = c.native_rows.published.as_ref()?;
    published
        .payload
        .context
        .owner(row)
        .then_some(published.end())
}
impl Published {
    pub(super) fn planes(&self) -> Option<[&Plane; 2]> {
        Some([self.payload.unary.as_ref()?, self.payload.cube.as_ref()?])
    }
    pub(super) fn end(&self) -> Instant {
        self.payload.context.end
    }
    #[cfg(test)]
    pub(super) fn component(&self) -> usize {
        self.payload.context.component
    }
    pub(super) fn extra_plane(&self) -> Option<&Plane> {
        self.payload.extra.as_ref()
    }
    #[cfg(test)]
    pub(super) fn extra_component(&self) -> Option<usize> {
        self.payload.extra.as_ref()?;
        self.payload.context.extra_component
    }
}
#[derive(Debug, Default)]
pub(super) struct State {
    reserve_attempted: bool,
    setup_work_units: usize,
    reservation: Option<bank::Quote>,
    fund_attempted: bool,
    upgrade_attempted: bool,
    admission: Option<Admission>,
    begun: bool,
    pending: Option<Arc<Payload>>,
    published: Option<Published>,
    bound: bool,
}

/// No admission means constant-time refusal, before scans, polls or allocations.
/// Called only AFTER the first completed exact Source replay and inside the
/// borrowed full-original-model binding. The precharge must cover this walk.
pub(super) fn begin(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &SignedPhaseBinding<'_>,
    preferred_component: Option<usize>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let (caps, bank) = {
        let mut counts = row.budget.state.lock().ok()?;
        let state = &counts.native_rows;
        let Some(a) = state.admission.as_ref() else {
            return Some(());
        };
        if state.begun {
            return Some(());
        }
        if a.source != row.source
            || a.source != std::ptr::from_ref(cuts.problem) as usize
            || a.dimensions != row.budget.dimensions
            || !a.ledger.ptr_eq(&Arc::downgrade(&row.budget.state))
            || a.authority != row.budget.authority
            || a.search_end != row.budget.search_end
            || a.deadline != cuts.deadline
            || a.authority > a.deadline
            || counts.replay_attempts != 1
            || counts.source_replays != 1
            || counts.component_rows.collection_attempts != 1
            || !binding.matches(cuts, parts)
        {
            return Some(());
        }
        let caps = a.caps;
        let bank = a.bank.clone();
        counts.native_rows.begun = true; // before any optional walk; never reset
        (caps, bank)
    };
    let end = end.min(row.budget.authority).min(cuts.deadline);
    require_hull_environment().ok()?;
    if !live(end) || !poll() {
        return None;
    }
    if row.plan.components.len() > caps.components {
        return Some(());
    }
    // The original pair ALWAYS keeps the first structural component. A paid
    // third slot may select the completed Cell component in the same bounded
    // scan. Refusing the upgrade cannot change the original pair's policy.
    let mut selected = None;
    let mut extra_component = None;
    let preferred_component = (caps.row_capacity == 3)
        .then_some(preferred_component)
        .flatten();
    for (index, c) in row.plan.components.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        let (d, p, k) = (cuts.problem.input_dim, c.positive.len(), c.negative.len());
        if !(2..=8).contains(&k) || k > caps.negative || p > caps.positive {
            continue;
        }
        let Some(base) = d.checked_add(p).and_then(|v| v.checked_add(k)) else {
            return Some(());
        };
        let Some(a) = base.checked_mul(2).and_then(|v| v.checked_add(k + 1)) else {
            return Some(());
        };
        let Some(width) = d.checked_add(1) else {
            return Some(());
        };
        let Some(negative_expansion) = k.checked_mul(width) else {
            return Some(());
        };
        let Some(e) = negative_expansion
            .checked_add(d)
            .and_then(|v| v.checked_add(p))
            .and_then(|v| v.checked_mul(2))
        else {
            return Some(());
        };
        let Some(u) = p
            .checked_mul(d)
            .and_then(|v| v.checked_add(d))
            .and_then(|v| v.checked_add(negative_expansion))
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(k + 1))
        else {
            return Some(());
        };
        if base.checked_add(k).is_none_or(|v| v > caps.semantic_row)
            || a > caps.semantic_pair
            || e > caps.expanded_pair
            || u > caps.raw_expansion_pair
        {
            continue;
        }
        if selected.is_none() {
            selected = Some(index);
        }
        if preferred_component == Some(index) {
            extra_component = Some(index);
            break;
        }
        if preferred_component.is_none() {
            break;
        }
    }
    let Some(component) = selected else {
        return Some(());
    };
    let payload = Arc::new(Payload {
        context: Context {
            source: row.source,
            ledger: Arc::downgrade(&row.budget.state),
            signature: row.signature.clone(),
            witness: row.witness.clone(),
            authority: row.budget.authority,
            search_end: row.budget.search_end,
            deadline: cuts.deadline,
            end,
            parts: std::ptr::from_ref(parts) as usize,
            original_rows: parts.problem.num_rows(),
            original_columns: parts.problem.num_cols(),
            component,
            extra_component,
            caps,
            bank,
        },
        unary: None,
        cube: None,
        extra: None,
        unary_attempted: false,
        cube_attempted: false,
        extra_attempted: false,
        armed_rows: None,
    });
    if !live(end) || !poll() {
        return None;
    }
    row.budget.state.lock().ok()?.native_rows.pending = Some(payload);
    Some(())
}
pub(super) fn selected(row: &SignedMajorant) -> Option<usize> {
    let counts = row.budget.state.lock().ok()?;
    let pending = counts.native_rows.pending.as_ref()?;
    pending
        .context
        .owner(row)
        .then_some(pending.context.component)
}
pub(super) fn selected_extra(row: &SignedMajorant) -> Option<usize> {
    let counts = row.budget.state.lock().ok()?;
    let pending = counts.native_rows.pending.as_ref()?;
    pending
        .context
        .owner(row)
        .then_some(pending.context.extra_component)
        .flatten()
}
fn refuse(row: &SignedMajorant) {
    if let Ok(mut counts) = row.budget.state.lock() {
        counts.native_rows.pending = None;
    }
}
fn push(
    terms: &mut Vec<Term>,
    atom: Atom,
    column: usize,
    actual: &(usize, f64),
    cap: usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    if column != actual.0 || !actual.1.is_finite() || terms.len() >= cap {
        return None;
    }
    for old in terms.iter() {
        if !live(end) || !poll() {
            return None;
        }
        // Conservative first version: reject all ambiguous original aliases.
        if old.atom == atom || old.original_column == column {
            return None;
        }
    }
    terms.push(Term {
        atom,
        original_column: column,
        coefficient_bits: actual.1.to_bits(),
    });
    Some(())
}
fn capture(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &SignedPhaseBinding<'_>,
    origin: &phase::NativeRow<'_>,
    context: &Context,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Plane> {
    let (index, cube, selectors, stored, stored_g) = origin.view();
    let expected_component = if origin.is_extra() {
        if context.caps.row_capacity != 3 {
            return None;
        }
        context.extra_component?
    } else {
        context.component
    };
    if index != expected_component
        || !context.owner(row)
        || !context.original(parts)
        || !binding.matches(cuts, parts)
        || context.source != std::ptr::from_ref(cuts.problem) as usize
        || context.deadline != cuts.deadline
        || stored.ub != f64::INFINITY
        || !stored.lb.is_finite()
        || stored.coeffs.len() > context.caps.semantic_row
        || (!cube && selectors.len() > 1)
        || selectors.len() > 8
        || (!cube && stored_g.is_some())
    {
        return None;
    }
    let c = row.plan.components.get(index)?;
    let g = stored_g.unwrap_or(&c.g);
    if g.len() != cuts.problem.input_dim || g.len() != c.g.len() {
        return None;
    }
    let mut terms = Vec::with_capacity(stored.coeffs.len());
    if terms.capacity() > stored.coeffs.len() {
        return None;
    }
    let mut actual = stored.coeffs.iter();
    for &head in c.negative.iter().chain(&c.positive) {
        if !live(context.end) || !poll() {
            return None;
        }
        push(
            &mut terms,
            Atom::Head(head),
            binding.head_column(head)?.0,
            actual.next()?,
            context.caps.semantic_row,
            context.end,
            poll,
        )?;
    }
    for (i, &g) in g.iter().enumerate() {
        if !live(context.end) || !poll() {
            return None;
        }
        let expected = finite(-2. * g)?;
        if expected != 0. {
            let entry = actual.next()?;
            if entry.1.to_bits() != expected.to_bits() {
                return None;
            }
            push(
                &mut terms,
                Atom::Input(i),
                parts.input_vars.get(i)?.0,
                entry,
                context.caps.semantic_row,
                context.end,
                poll,
            )?;
        }
    }
    for &head in selectors {
        if !live(context.end) || !poll() {
            return None;
        }
        if !c.negative.contains(&head) {
            return None;
        }
        let entry = actual.next()?;
        if entry.1 == 0. {
            return None;
        }
        push(
            &mut terms,
            Atom::Selector(head),
            binding.column(head)?.0,
            entry,
            context.caps.semantic_row,
            context.end,
            poll,
        )?;
    }
    if actual.next().is_some() {
        return None;
    }
    // Explicit bounded insertion sort, not an unquoted library sort. Preserve
    // stored bits, never merge/re-round coefficients. At most A(A-1)/2 swaps.
    for i in 1..terms.len() {
        let mut j = i;
        while j != 0 && terms[j].atom < terms[j - 1].atom {
            if !live(context.end) || !poll() {
                return None;
            }
            terms.swap(j, j - 1);
            j -= 1;
        }
    }
    Some(Plane {
        lower_bits: stored.lb.to_bits(),
        terms,
    })
}
/// NativeRow cannot be constructed by this module: its private fields live
/// in phase and are filled only at complete producer-family exits.
/// The non-Cube slot holds a unary lift or All's checked component fallback.
/// An optional third-row numeric/alias refusal also preserves the old pair;
/// a poll cancellation remains an outer transaction cancellation.
pub(super) fn stage(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &SignedPhaseBinding<'_>,
    origin: phase::NativeRow<'_>,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let cube = origin.view().1;
    let extra = origin.is_extra();
    let context = {
        let mut counts = row.budget.state.lock().ok()?;
        let Some(pending) = counts.native_rows.pending.as_mut().and_then(Arc::get_mut) else {
            return Some(());
        };
        let expected = if extra {
            (pending.context.caps.row_capacity == 3)
                .then_some(pending.context.extra_component)
                .flatten()
        } else {
            Some(pending.context.component)
        };
        if expected != Some(origin.view().0) {
            return Some(());
        }
        // One attempt per producer kind before any optional coefficient walk.
        let attempted = if extra {
            &mut pending.extra_attempted
        } else if cube {
            &mut pending.cube_attempted
        } else {
            &mut pending.unary_attempted
        };
        if *attempted {
            return Some(());
        }
        *attempted = true;
        pending.context.clone()
    };
    let mut cancelled = false;
    let captured = capture(row, cuts, parts, binding, &origin, &context, &mut || {
        let ok = poll();
        cancelled |= !ok;
        ok
    });
    if cancelled || !live(context.end) {
        refuse(row);
        return None;
    }
    let Some(plane) = captured else {
        if !extra {
            refuse(row);
        }
        return Some(());
    };
    let mut counts = row.budget.state.lock().ok()?;
    let Some(pending) = counts.native_rows.pending.as_mut().and_then(Arc::get_mut) else {
        return Some(());
    };
    if !pending.context.owner(row) || !pending.context.original(parts) {
        return Some(());
    }
    let total = [&pending.unary, &pending.cube, &pending.extra]
        .into_iter()
        .flatten()
        .try_fold(plane.terms.len(), |n, other| {
            n.checked_add(other.terms.len())
        });
    if total.is_none_or(|n| n > context.caps.semantic_pair) {
        if !extra {
            counts.native_rows.pending = None;
        }
        return Some(());
    }
    let mut stopped = false;
    let mut aliases = false;
    'planes: for other in [&pending.unary, &pending.cube, &pending.extra]
        .into_iter()
        .flatten()
    {
        for term in &plane.terms {
            for old in &other.terms {
                // No caller callback while the ledger mutex is held. The
                // bounded loop checks the original clock; poll follows unlock.
                if !live(context.end) {
                    stopped = true;
                    break 'planes;
                }
                aliases |= old.original_column == term.original_column && old.atom != term.atom;
            }
        }
        if aliases {
            break;
        }
    }
    // Drop the borrowed plane iterator before changing the pending owner.
    if stopped {
        counts.native_rows.pending = None;
        return None;
    }
    if aliases {
        if !extra {
            counts.native_rows.pending = None;
        }
        return Some(());
    }
    let slot = if extra {
        &mut pending.extra
    } else if cube {
        &mut pending.cube
    } else {
        &mut pending.unary
    };
    if slot.is_some() {
        return Some(());
    }
    *slot = Some(plane);
    drop(counts);
    if !live(context.end) || !poll() {
        refuse(row);
        return None;
    }
    Some(())
}
/// O(1), before the old header claim and first IR append. No optional refusal
/// can fail the old row batch; it only drops this prospective transport.
pub(super) fn arm(row: &SignedMajorant, parts: &MipParts, additional: usize) {
    let Ok(mut counts) = row.budget.state.lock() else {
        return;
    };
    let Some(p) = counts.native_rows.pending.as_mut().and_then(Arc::get_mut) else {
        return;
    };
    if !p.context.owner(row)
        || !p.context.original(parts)
        || !live(p.context.end)
        || additional > p.context.caps.batch_rows
        || p.armed_rows.is_some()
        || p.unary.is_none()
        || p.cube.is_none()
    {
        counts.native_rows.pending = None;
        return;
    }
    p.armed_rows = Some(additional);
}
/// No allocation, clock renewal or fallible return after successful IR commit.
/// Expiry during append does not erase that fact; bind() below still requires
/// the SAME original live end before native work can consume the token.
pub(super) fn publish(row: &SignedMajorant, parts: &MipParts, additional: usize) {
    let Ok(mut counts) = row.budget.state.lock() else {
        return;
    };
    let Some(payload) = counts.native_rows.pending.take() else {
        return;
    };
    let c = &payload.context;
    if !c.owner(row)
        || c.parts != std::ptr::from_ref(parts) as usize
        || c.original_columns != parts.problem.num_cols()
        || parts.num_cols != c.original_columns
        || c.original_rows.checked_add(additional) != Some(parts.problem.num_rows())
        || payload.armed_rows != Some(additional)
        || payload.unary.is_none()
        || payload.cube.is_none()
        || counts.native_rows.published.is_some()
    {
        return;
    }
    counts.native_rows.published = Some(Published { payload });
}
/// Called by EVERY existing reset/abort. Never refund or reopen the once latch;
/// an older genuinely committed token remains valid across a later refusal.
pub(super) fn reset(b: &Budget) {
    if let Ok(mut counts) = b.state.lock() {
        counts.native_rows.pending = None;
    }
}
/// Once-only native handoff. This exact signature walk is prepaid by the real
/// admission. The returned Arc is shared by workers, not cloned coefficient Vecs.
pub(super) fn bind<'p>(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'p>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<BoundPublished<'p>> {
    let published = {
        let mut counts = row.budget.state.lock().ok()?;
        if counts.native_rows.bound {
            return None;
        }
        let published = counts.native_rows.published.as_ref()?.clone();
        counts.native_rows.bound = true;
        published
    };
    let c = &published.payload.context;
    if !c.owner(row)
        || c.source != std::ptr::from_ref(cuts.problem) as usize
        || c.deadline != cuts.deadline
        || end > c.end
        || !live(end)
        || c.ledger.upgrade().is_none()
    {
        return None;
    }
    let mut actual = signature(cuts);
    for expected in c.signature.iter().copied() {
        if !live(end) || !poll() || actual.next() != Some(expected) {
            return None;
        }
    }
    if actual.next().is_some() {
        return None;
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some(BoundPublished {
        source: cuts.problem,
        published,
    })
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_native_routing_tests.rs"]
mod routing_tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_native_publication_tests.rs"]
mod tests;
