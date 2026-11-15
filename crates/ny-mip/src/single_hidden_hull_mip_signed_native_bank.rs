// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Additive admission with the original clock and bounded LP allowance.
//! Routing overlay threads the scoped consumer through the existing public BAB path.
use super::numeric::{Enclosure, Node, PriceRun, SourceFrame};
use super::*;
use crate::single_hidden_hull::primal_advice::RootLpCounts;
use crate::single_hidden_hull::{bab::HullSplit, HullLp};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

const SETUP: usize = 4096;
const UPGRADE_SETUP: usize = 1024;
const COMPONENTS: usize = 64;
const POSITIVE: usize = 16;
const NEGATIVE: usize = 8;
const ROWS: usize = 2;
const MAX_CALLS: usize = 2;
// One process-global optional numeric/LP staging slot, not one slot per worker.
// No spinning, waiting, work refund, or clock reset on a busy slot.
static SLOT: AtomicBool = AtomicBool::new(false);
struct Slot;
impl Slot {
    fn claim() -> Option<Self> {
        SLOT.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()?;
        Some(Self)
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        SLOT.store(false, Ordering::Release);
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Quote {
    caps: Caps,
    root: RootLpCounts,
    bytes: usize,
    shared_work: usize,
    call_work: usize,
}
#[derive(Debug)]
pub(super) struct Bank {
    root: RootLpCounts,
    quote: Quote,
    calls: usize,
    spent: AtomicUsize,
    committed: AtomicUsize,
    priced: AtomicUsize,
    enclosure: OnceLock<Option<Enclosure>>,
}
impl Bank {
    pub(super) fn matches_caps(&self, caps: Caps) -> bool {
        self.quote.caps == caps
    }
}
fn add(xs: &[usize]) -> Option<usize> {
    xs.iter().try_fold(0usize, |a, b| a.checked_add(*b))
}
fn mul(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)
}
fn ceil_log2(n: usize) -> Option<usize> {
    let n = n.checked_sub(1)?;
    Some((usize::BITS - n.leading_zeros()) as usize)
}
fn dense(q: usize, r: usize, z: usize, e: usize, added: usize) -> Option<bool> {
    let rows = r.checked_add(added)?;
    let cells = rows.checked_mul(q)?;
    Some(cells <= (1 << 20) && z.checked_add(e)?.checked_mul(2)? >= cells)
}

/// O(1) metadata only, called AFTER SETUP has been irreversibly debited.
/// Bytes use structural root upper bounds, not late actual size estimates.
/// A split node must fit the actual root receipt before any optional allocation;
/// its elastic columns/rows/nonzeros are INCLUDED in that comparison.
fn quote(cuts: &HullRootMipCuts<'_>, b: &Budget, root: RootLpCounts, rows: usize) -> Option<Quote> {
    if !(2..=3).contains(&rows) {
        return None;
    }
    let (d, h, p) = b.dimensions;
    if !(1..=62).contains(&d) || !(1..=512).contains(&h) || p > h {
        return None;
    }
    let hd = d.checked_mul(h)?;
    if hd > 65_536 {
        return None;
    }
    let n = h.checked_sub(p)?; // includes zero coefficients conservatively
    let t = n.checked_mul(d)?;
    let qcap = add(&[d, p, mul(n, d.checked_add(1)?)?])?;
    let rcap = p.checked_add(t)?;
    let zcap = mul(p, d.checked_add(1)?)?.checked_add(mul(3, t)?)?;
    if root.columns() < d || root.columns() > qcap || root.rows() > rcap || root.nonzeros() > zcap {
        return None;
    }
    let base = add(&[d, POSITIVE, NEGATIVE])?;
    let e1 = add(&[d, POSITIVE, mul(NEGATIVE, d.checked_add(1)?)?])?;
    let caps = Caps {
        row_capacity: rows,
        components: COMPONENTS,
        positive: POSITIVE,
        negative: NEGATIVE,
        semantic_row: base.checked_add(NEGATIVE)?,
        semantic_pair: mul(rows, base)?.checked_add(mul(rows - 1, NEGATIVE)?.checked_add(1)?)?,
        expanded_pair: mul(rows, e1)?,
        raw_expansion_pair: mul(rows, add(&[d, mul(POSITIVE, d)?, mul(NEGATIVE, d + 1)?])?)?
            .checked_add(mul(rows - 1, NEGATIVE)?.checked_add(1)?)?,
        batch_rows: add(&[
            cuts.len(),
            1,
            mul(if b.portfolio_phase_enabled { 5 } else { 3 }, COMPONENTS)?,
            // Independently admitted Cell, Ray, table LP and Joint rows may survive.
            4 * usize::from(b.portfolio_phase_enabled),
            h,
        ])?,
    };
    let (a, e, u) = (
        caps.semantic_pair,
        caps.expanded_pair,
        caps.raw_expansion_pair,
    );
    // Runtime size checks prevent the borrowed-pool accounting becoming an ABI
    // assumption. There is NO 24*A prototype Plane pool in this implementation.
    if size_of::<Term>() > 32
        || size_of::<Plane>() > 32
        || size_of::<Enclosure>() > 128
        || align_of::<Bank>() > align_of::<usize>()
        || align_of::<Payload>() > align_of::<usize>()
    {
        return None;
    }
    let shared = add(&[mul(28, h)?, mul(4, t)?, 512])?;
    let token = add(&[
        size_of::<Payload>(),
        mul(2, size_of::<usize>())?,
        mul(a, size_of::<Term>())?,
        size_of::<Bank>(),
        mul(2, size_of::<usize>())?,
        2048,
    ])?;
    let pricer = add(&[mul(16, qcap)?, mul(52, h)?, 1024])?;
    let rc = rcap.checked_add(rows)?;
    let headers = rc.max(4).checked_next_power_of_two()?;
    let slot = add(&[
        pricer,
        mul(64, e)?,
        mul(64, headers)?,
        mul(16, zcap.checked_add(e)?)?,
        mul(20, add(&[qcap, rcap, rows])?)?,
        mul(128, rows)?,
        4096,
    ])?;
    let bytes = add(&[shared, token, slot])?;
    // Actual root limits tighten WORK only; all bytes remain structurally paid.
    // Full signature length is computable from existing immutable slice lengths.
    let source = cuts.problem;
    let sig = add(&[
        11,
        source.w1.len(),
        source.b1.len(),
        source.c.len(),
        source.lower.len(),
        source.upper.len(),
        cuts.multipliers.alpha.len(),
        cuts.multipliers.g.len(),
    ])?;
    let token_work = add(&[
        mul(64, mul(a, a)?)?,
        mul(
            256,
            add(&[sig, COMPONENTS, d, POSITIVE, NEGATIVE * NEGATIVE, a, 1])?,
        )?,
        4096,
    ])?;
    let shared_work = add(&[
        mul(128, add(&[mul(3, hd)?, t, h, d, a, 1])?)?,
        4096,
        token_work,
    ])?;
    let q = root.columns();
    let r = root.rows();
    let z = root.nonzeros();
    let split_work = h.checked_mul(d.checked_add(1)?)?;
    let construct = add(&[
        mul(
            64,
            add(&[hd, z, t, h, split_work, a, u, mul(3 * rows, q)?, e, 1])?,
        )?,
        mul(128, mul(rows, mul(e1, ceil_log2(e1.checked_add(1)?)?)?)?)?,
        4096,
    ])?;
    let price = add(&[mul(128, add(&[hd, t, a, split_work, q, h, 1])?)?, 1024])?;
    // One actual-node row-length census, constant guards, and two sign prices.
    let call_work = add(&[construct, mul(2, price)?, mul(64, r.checked_add(1)?)?, 1024])?;
    Some(Quote {
        caps,
        root,
        bytes,
        shared_work,
        call_work,
    })
}

/// Called only by the new private opt-in entry, immediately after Budget::new.
/// No Budget clone/Ticket/component/Plan exists yet. The new inline State header
/// is separately paid in Budget::new even when this extension is refused.
pub(super) fn reserve(
    cuts: &HullRootMipCuts<'_>,
    b: &mut Budget,
    stage: NativeConstruction,
) -> Option<()> {
    if !stage.matches(b)
        || b.prepaid.is_some()
        || b.root_coupling.is_some()
        || !live(b.search_end)
        || b.source != std::ptr::from_ref(cuts.problem) as usize
    {
        return None;
    }
    {
        let mut c = b.state.lock().ok()?;
        if c.native_rows.reserve_attempted
            || c.replay_attempts != 0
            || c.shared_bytes != 0
            || c.component_rows.admitted_bytes != 0
            || c.reserved != 0
            || c.component_reserved != 0
        {
            return None;
        }
        c.native_rows.reserve_attempted = true;
        let next = c.work.checked_add(SETUP)?;
        if next > MAX_WORK || !live(b.search_end) {
            return None;
        }
        c.work = next; // never refunded, including missing-receipt refusal
        c.native_rows.setup_work_units = SETUP;
    }
    let root =
        cuts.primal
            .as_ref()?
            .lock()
            .ok()?
            .root_lp_counts(cuts.problem, b.search_end, &b.state)?;
    let q = quote(cuts, b, root, ROWS)?;
    let bytes = b.admitted_bytes.checked_add(q.bytes)?;
    if bytes > MAX_BYTES || !live(b.search_end) {
        return None;
    }
    b.admitted_bytes = bytes; // visible to every later Ticket and Budget clone
    b.state.lock().ok()?.native_rows.reservation = Some(q);
    Some(())
}

/// Called once after Cube admission and before any mass/refinement or owner
/// clone. Holds remain intact. Conditional portfolios fund one root call;
/// compatibility entries prefer two prepaid calls, then one. Otherwise
/// retain the already paid memory/setup and decline without changing old rows.
pub(super) fn fund(cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<()> {
    if !live(b.search_end) || b.source != std::ptr::from_ref(cuts.problem) as usize {
        return None;
    }
    let mut c = b.state.lock().ok()?;
    let q = c.native_rows.reservation?;
    if c.native_rows.fund_attempted {
        return None;
    }
    c.native_rows.fund_attempted = true;
    if !c.component_rows.phase.cube.admitted || c.replay_attempts != 0 {
        return None;
    }
    let available = MAX_WORK
        .checked_sub(c.work)?
        .checked_sub(c.reserved)?
        .checked_sub(c.component_reserved)?;
    let limit = if b.portfolio_phase_enabled {
        1
    } else {
        MAX_CALLS
    };
    let calls = (1..=limit).rev().find(|n| {
        q.call_work
            .checked_mul(*n)
            .and_then(|v| v.checked_add(q.shared_work))
            .is_some_and(|v| v <= available)
    })?;
    let total = q.shared_work.checked_add(q.call_work.checked_mul(calls)?)?;
    if !live(b.search_end) {
        return None;
    }
    c.work = c.work.checked_add(total)?; // complete debit before Arc/cache/row work
    let bank = Arc::new(Bank {
        root: q.root,
        quote: q,
        calls,
        spent: AtomicUsize::new(0),
        committed: AtomicUsize::new(0),
        priced: AtomicUsize::new(0),
        enclosure: OnceLock::new(),
    });
    c.native_rows.admission = Some(Admission {
        source: b.source,
        ledger: Arc::downgrade(&b.state),
        dimensions: b.dimensions,
        authority: b.authority,
        search_end: b.search_end,
        deadline: cuts.deadline,
        caps: q.caps,
        bank,
    });
    Some(())
}

/// Optional capacity for ONE additional licensed row in the same funded call.
/// Every old native/Source/child hold has priority. The base byte identity must
/// remain unchanged: an earlier root-coupling Ticket may already pin it.
pub(super) fn upgrade(cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<()> {
    let mut c = b.state.lock().ok()?;
    if c.native_rows.upgrade_attempted {
        return None;
    }
    c.native_rows.upgrade_attempted = true;
    if !b.portfolio_phase_enabled
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.authority > cuts.deadline
        || !live(b.search_end)
        || c.replay_attempts != 0
        || c.source_replays != 0
        || c.native_rows.begun
        || !c.component_rows.phase.cube.cell.admitted
    {
        return None;
    }
    let old = c.native_rows.reservation?;
    let a = c.native_rows.admission.as_ref()?;
    if old.caps.row_capacity != 2
        || a.bank.calls != 1
        || a.source != b.source
        || !a.ledger.ptr_eq(&Arc::downgrade(&b.state))
        || a.authority != b.authority
        || a.search_end != b.search_end
        || a.deadline != cuts.deadline
    {
        return None;
    }
    let setup_work = c.work.checked_add(UPGRADE_SETUP)?;
    let setup_total = c.native_rows.setup_work_units.checked_add(UPGRADE_SETUP)?;
    if setup_work
        .checked_add(c.reserved)?
        .checked_add(c.component_reserved)?
        > MAX_WORK
        || !live(b.search_end)
    {
        return None;
    }
    // Pay bounded quote/admission arithmetic even if later payload refuses.
    c.work = setup_work;
    c.native_rows.setup_work_units = setup_total;
    let q = quote(cuts, b, old.root, 3)?;
    let work = q
        .shared_work
        .checked_sub(old.shared_work)?
        .checked_add(q.call_work.checked_sub(old.call_work)?)?;
    let bytes = q.bytes.checked_sub(old.bytes)?;
    let next_work = c.work.checked_add(work)?;
    let component_bytes = c.component_rows.admitted_bytes.checked_add(bytes)?;
    if next_work
        .checked_add(c.reserved)?
        .checked_add(c.component_reserved)?
        > MAX_WORK
        || b.admitted_bytes
            .checked_add(c.shared_bytes)?
            .checked_add(c.direction.cache_bytes)?
            .checked_add(c.child.cache_bytes)?
            .checked_add(component_bytes)?
            > MAX_BYTES
        || !live(b.search_end)
    {
        return None;
    }
    let a = c.native_rows.admission.as_mut()?;
    let bank = Arc::get_mut(&mut a.bank)?;
    if bank.spent.load(Ordering::Relaxed) != 0
        || bank.committed.load(Ordering::Relaxed) != 0
        || bank.priced.load(Ordering::Relaxed) != 0
        || bank.enclosure.get().is_some()
        || !live(b.search_end)
    {
        return None;
    }
    // All fallible checks precede this irreversible debit/capacity commit.
    // No allocation, Source traversal, scoring, clock renewal or call grant.
    bank.quote = q;
    a.caps = q.caps;
    c.work = next_work;
    c.component_rows.admitted_bytes = component_bytes;
    c.native_rows.reservation = Some(q);
    Some(())
}

#[cfg(test)]
pub(super) fn upgrade_quote(cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<(usize, usize)> {
    let c = b.state.lock().ok()?;
    let old = c.native_rows.reservation?;
    if old.caps.row_capacity != 2 || c.native_rows.admission.as_ref()?.bank.calls != 1 {
        return None;
    }
    let new = quote(cuts, b, old.root, 3)?;
    Some((
        UPGRADE_SETUP
            .checked_add(new.shared_work.checked_sub(old.shared_work)?)?
            .checked_add(new.call_work.checked_sub(old.call_work)?)?,
        new.bytes.checked_sub(old.bytes)?,
    ))
}

/// Declined is possible only before mutation. Committed(None) means the advice
/// batch was appended but the original deadline then expired: do NOT run an
/// unaugmented retry. No status variant itself is a bound or an UNSAT proof.
pub(in crate::single_hidden_hull) enum Visit<R> {
    Declined,
    Stopped,
    Committed(Option<R>),
}

pub(super) fn stats(b: &Budget) -> SignedNativeRowsReport {
    let Ok(c) = b.state.lock() else {
        return SignedNativeRowsReport::default();
    };
    let Some(a) = c.native_rows.admission.as_ref() else {
        return SignedNativeRowsReport::default();
    };
    SignedNativeRowsReport {
        attempted: a.bank.spent.load(Ordering::Relaxed),
        committed: a.bank.committed.load(Ordering::Relaxed),
        priced: a.bank.priced.load(Ordering::Relaxed),
    }
}
pub(super) fn funding(b: &Budget) -> SignedNativeFundingReport {
    let Ok(c) = b.state.lock() else {
        return SignedNativeFundingReport::default();
    };
    let s = &c.native_rows;
    let calls = s.admission.as_ref().map_or(0, |a| a.bank.calls);
    let (shared, per_call, bytes) = s
        .reservation
        .map_or((0, 0, 0), |q| (q.shared_work, q.call_work, q.bytes));
    SignedNativeFundingReport {
        funded_calls: calls,
        setup_work_units: s.setup_work_units,
        shared_work_units: shared,
        per_call_work_units: per_call,
        prepaid_work_units: s.setup_work_units
            + if calls == 0 {
                0
            } else {
                shared + calls * per_call
            },
        admitted_bytes: bytes,
    }
}
impl BoundPublished<'_> {
    /// Concrete scoped consumer. No extra LP: run is the caller's ONE existing
    /// proposer call. Old and new certificates are evaluated independently on
    /// the SAME full augmented dual vector; neither decoder can suppress the
    /// other with an early ?. The bank/slot/scratch live through that call.
    pub(in crate::single_hidden_hull) fn with_node<R>(
        &self,
        p: &HullMarginProblem,
        splits: &[HullSplit],
        lp: &mut HullLp,
        split_rows: &[Option<usize>],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
        old: impl FnMut(&HullLp, &[f64]) -> Option<f64>,
        run: impl FnOnce(&HullLp, &mut dyn FnMut(&[f64]) -> Option<f64>) -> R,
    ) -> Visit<R> {
        let end = end.min(self.published.payload.context.end);
        if !std::ptr::eq(self.source, p) {
            return Visit::Declined;
        }
        let p = self.source;
        let mut stopped = false;
        let result = self.with_node_inner(
            p,
            splits,
            lp,
            split_rows,
            end,
            &mut || {
                let ok = !stopped && live(end) && poll();
                stopped |= !ok;
                ok
            },
            old,
            run,
        );
        if matches!(&result, Visit::Declined) && (stopped || !live(end)) {
            Visit::Stopped
        } else {
            result
        }
    }
    fn with_node_inner<R>(
        &self,
        p: &HullMarginProblem,
        splits: &[HullSplit],
        lp: &mut HullLp,
        split_rows: &[Option<usize>],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
        mut old: impl FnMut(&HullLp, &[f64]) -> Option<f64>,
        run: impl FnOnce(&HullLp, &mut dyn FnMut(&[f64]) -> Option<f64>) -> R,
    ) -> Visit<R> {
        let c = &self.published.payload.context;
        let end = end.min(c.end);
        if c.source != std::ptr::from_ref(p) as usize
            || !c.bank.matches_caps(c.caps)
            || !live(end)
            || splits.len() > p.hidden_dim
        {
            return Visit::Declined;
        }
        let Some(_ledger) = c.ledger.upgrade() else {
            return Visit::Declined;
        };
        let Some(_slot) = Slot::claim() else {
            return Visit::Declined;
        };
        let bank = &c.bank;
        if bank
            .spent
            .try_update(Ordering::AcqRel, Ordering::Relaxed, |n| {
                (n < bank.calls).then(|| n + 1)
            })
            .is_err()
        {
            return Visit::Declined;
        }
        // The credit is spent even if any subsequent numeric/map check refuses.
        // Includes actual split-elastic variables; exceeding root size declines.
        if lp.model.num_cols() > bank.root.columns()
            || lp.model.num_rows() > bank.root.rows()
            || lp.model.num_cols() < p.input_dim
        {
            return Visit::Declined;
        }
        let mut z = 0usize;
        for i in 0..lp.model.num_rows() {
            if !live(end) || !poll() {
                return Visit::Declined;
            }
            let Some(row) = lp.model.row_at(i) else {
                return Visit::Declined;
            };
            let Some(next) = z.checked_add(lp.model.row(row).0.len()) else {
                return Visit::Declined;
            };
            z = next;
            if z > bank.root.nonzeros() {
                return Visit::Declined;
            }
        }
        // Each canonical row has at most Q unique columns. Tighten only this
        // already-paid actual-node gate, never the early structural byte quote.
        let mut extra = self.published.extra_plane();
        let mut added = 2 + usize::from(extra.is_some());
        if added > c.caps.row_capacity {
            return Visit::Declined;
        }
        let expanded_for = |n: usize| -> Option<usize> {
            let per_row = c.caps.expanded_pair.checked_div(c.caps.row_capacity)?;
            Some(
                per_row
                    .checked_mul(n)?
                    .min(lp.model.num_cols().checked_mul(n)?),
            )
        };
        let Some(mut expanded) = expanded_for(added) else {
            return Visit::Declined;
        };
        if dense(lp.model.num_cols(), lp.model.num_rows(), z, expanded, added) != Some(false) {
            let Some(old_expanded) = expanded_for(2) else {
                return Visit::Declined;
            };
            if added != 3
                || dense(lp.model.num_cols(), lp.model.num_rows(), z, old_expanded, 2)
                    != Some(false)
            {
                return Visit::Declined;
            }
            extra = None;
            added = 2;
            expanded = old_expanded;
        }
        let data = bank.enclosure.get_or_init(|| Enclosure::new(p, end, poll));
        let Some(data) = data.as_ref() else {
            return Visit::Declined;
        };
        let Some(planes) = self.published.planes() else {
            return Visit::Declined;
        };
        let Some(frame) = SourceFrame::new_with_extra(p, planes, extra, data, end, poll) else {
            return Visit::Declined;
        };
        let Some(node) = Node::new(&frame, splits, poll) else {
            return Visit::Declined;
        };
        let Some(mut price) = PriceRun::new(node) else {
            return Visit::Declined;
        };
        let Some(batch) = price.nominal_batch(lp, split_rows, expanded, poll) else {
            return Visit::Declined;
        };
        // A numeric-only optional-plane refusal may have reduced the batch.
        // Apply the original two-row density gate to that exact retained pair.
        if batch.row_count() != added {
            let Some(old_expanded) = expanded_for(batch.row_count()) else {
                return Visit::Declined;
            };
            if batch.row_count() != 2
                || dense(lp.model.num_cols(), lp.model.num_rows(), z, old_expanded, 2)
                    != Some(false)
            {
                return Visit::Declined;
            }
        }
        if batch.append(lp, poll).is_none() {
            return Visit::Declined;
        }
        // Completion bookkeeping stays truthful even if the clock just expired.
        bank.committed.fetch_add(1, Ordering::Relaxed);
        if !live(end) || !poll() {
            return Visit::Committed(None);
        }
        let lp: &HullLp = lp;
        let mut cancelled = false;
        let mut certify = |y: &[f64]| {
            if cancelled {
                return None;
            }
            if !live(end) || !poll() {
                cancelled = true;
                return None;
            }
            let previous = old(lp, y).filter(|x| x.is_finite());
            let augmented = price
                .lower(y, &mut || {
                    let ok = poll();
                    cancelled |= !ok;
                    ok
                })
                .filter(|x| x.is_finite());
            if cancelled || !live(end) {
                cancelled = true;
                return None;
            }
            if augmented.is_some() {
                bank.priced.fetch_add(1, Ordering::Relaxed);
            }
            match (previous, augmented) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            }
        };
        Visit::Committed(Some(run(lp, &mut certify)))
    }
}
