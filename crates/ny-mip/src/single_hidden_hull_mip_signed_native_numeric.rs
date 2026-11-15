// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Native adapter: borrowed published planes, shared owned enclosures.
//! The parent bank owns funding and the one global live slot.
use super::super::certificate::I;
use super::{Atom, Plane};
use crate::single_hidden_hull::bab::HullSplit;
use crate::single_hidden_hull::{require_hull_environment, Col, HullLp, HullMarginProblem};
use std::time::Instant;
const UNMAPPED: u32 = 0;
const ZERO: u32 = 1;
const LIVE: u32 = 2;
const POS: u32 = 3;
const NEG: u32 = 4;
const SPLIT: u32 = 8;
const NONE: u32 = u32::MAX;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Bound {
    lower: f64,
    upper: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RootHead {
    kind: u32,
    start: u32,
    len: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct NodeHead {
    kind: u32,
    column: u32,
    row: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct SplitSlot {
    column: u32,
    row: u32,
}
#[derive(Debug)]
pub(super) struct Enclosure {
    source: usize,
    dimensions: (usize, usize),
    bounds: Vec<Bound>,
    heads: Vec<RootHead>,
    inputs: Vec<u32>,
}
pub(super) struct SourceFrame<'p> {
    p: &'p HullMarginProblem,
    planes: [&'p Plane; 3],
    plane_count: usize,
    bounds: &'p [Bound],
    heads: &'p [RootHead],
    inputs: &'p [u32],
    end: Instant,
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn alive(end: Instant, poll: &mut dyn FnMut() -> bool) -> bool {
    Instant::now() < end && poll()
}
fn alloc<T: Clone>(count: usize, value: T) -> Option<Vec<T>> {
    let out = vec![value; count];
    (out.capacity() <= count).then_some(out)
}
impl Enclosure {
    pub(super) fn new(
        p: &HullMarginProblem,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        require_hull_environment().ok()?;
        if !alive(end, poll) {
            return None;
        }
        let h = p.hidden_dim;
        let d = p.input_dim;
        if h == 0
            || d == 0
            || p.c.len() != h
            || p.b1.len() != h
            || p.w1.len() != h.checked_mul(d)?
            || p.lower.len() != d
            || p.upper.len() != d
        {
            return None;
        }
        // Count/allocate once, before filling. This whole O(HD) construction
        // needs a fresh prepaid shared fee; it is not another mask replay.
        let mut term_count = 0usize;
        for j in 0..h {
            if p.c[j] < 0. {
                for i in 0..d {
                    if !alive(end, poll) {
                        return None;
                    }
                    let a = p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]);
                    if a > 0. && a.is_finite() {
                        term_count = term_count.checked_add(1)?;
                    }
                }
            }
        }
        let mut out = Self {
            source: std::ptr::from_ref(p) as usize,
            dimensions: (d, h),
            bounds: alloc(
                h,
                Bound {
                    lower: 0.,
                    upper: 0.,
                },
            )?,
            heads: alloc(
                h,
                RootHead {
                    kind: UNMAPPED,
                    start: 0,
                    len: 0,
                },
            )?,
            inputs: alloc(term_count, 0)?,
        };
        let mut next = 0usize;
        for j in 0..h {
            let mut z = I::point(p.b1[j])?;
            for i in 0..d {
                if !alive(end, poll) {
                    return None;
                }
                finite(p.lower[i])?;
                finite(p.upper[i])?;
                if p.lower[i] > p.upper[i] {
                    return None;
                }
                z = z.add(
                    I {
                        lo: p.lower[i],
                        hi: p.upper[i],
                    }
                    .scale(p.w_row(j)[i])?,
                )?;
            }
            out.bounds[j] = Bound {
                lower: z.lo,
                upper: z.hi,
            };
            let c = finite(p.c[j])?;
            let kind = if c == 0. {
                UNMAPPED
            } else if z.hi <= 0. {
                ZERO
            } else if z.lo >= 0. {
                LIVE
            } else if c > 0. {
                POS
            } else {
                NEG
            };
            let start = next;
            if kind == NEG {
                finite(I::point(z.hi)?.sub(I::point(z.lo)?)?.hi)?;
                for i in 0..d {
                    if !alive(end, poll) {
                        return None;
                    }
                    let a = p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]);
                    if a > 0. && a.is_finite() {
                        *out.inputs.get_mut(next)? = i.try_into().ok()?;
                        next += 1;
                    }
                }
                if next == start {
                    return None;
                }
            }
            out.heads[j] = RootHead {
                kind,
                start: start.try_into().ok()?,
                len: (next - start).try_into().ok()?,
            };
        }
        // Reserve included all negative coordinates, so unused stable capacity
        // stays charged; no shrink/refund or second allocation is required.
        require_hull_environment().ok()?;
        alive(end, poll).then_some(out)
    }
}
impl<'p> SourceFrame<'p> {
    #[cfg(test)]
    pub(super) fn new(
        p: &'p HullMarginProblem,
        planes: [&'p Plane; 2],
        data: &'p Enclosure,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        Self::new_with_extra(p, planes, None, data, end, poll)
    }
    pub(super) fn new_with_extra(
        p: &'p HullMarginProblem,
        planes: [&'p Plane; 2],
        extra: Option<&'p Plane>,
        data: &'p Enclosure,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        require_hull_environment().ok()?;
        let (d, h) = (p.input_dim, p.hidden_dim);
        if data.source != std::ptr::from_ref(p) as usize
            || data.dimensions != (d, h)
            || !alive(end, poll)
        {
            return None;
        }
        let mut out = Self {
            p,
            planes: [planes[0], planes[1], extra.unwrap_or(planes[0])],
            plane_count: 2 + usize::from(extra.is_some()),
            bounds: &data.bounds,
            heads: &data.heads,
            inputs: &data.inputs,
            end,
        };
        for slot in 0..out.plane_count {
            let mut stopped = false;
            let checked = out.check_plane(slot, &mut || {
                let ok = poll();
                stopped |= !ok;
                ok
            });
            if stopped || !alive(end, poll) {
                return None;
            }
            if checked.is_none() {
                // An unrepresentable EXTRA plane cannot suppress the original
                // pair. Cancellation/deadline remains sticky and never retries.
                if slot != 2 {
                    return None;
                }
                out.plane_count = 2;
                break;
            }
        }
        require_hull_environment().ok()?;
        alive(end, poll).then_some(out)
    }
    fn check_plane(&self, slot: usize, poll: &mut dyn FnMut() -> bool) -> Option<()> {
        let (d, h) = (self.p.input_dim, self.p.hidden_dim);
        let plane = self.planes.get(slot)?;
        finite(plane.lower())?;
        let mut previous = None;
        for term in plane.terms() {
            if !alive(self.end, poll) {
                return None;
            }
            let (atom, a) = term.semantic();
            let order = match atom {
                Atom::Input(i) => i,
                Atom::Head(j) => d.checked_add(j)?,
                Atom::Selector(j) => d.checked_add(h)?.checked_add(j)?,
            };
            if previous.is_some_and(|old| order <= old) {
                return None;
            }
            previous = Some(order);
            finite(a)?;
            match atom {
                Atom::Input(i) if i < d => {}
                Atom::Head(j) if j < h && self.heads[j].kind != UNMAPPED => {}
                Atom::Selector(j) if j < h && matches!(self.heads[j].kind, ZERO | LIVE | NEG) => {}
                _ => return None,
            }
        }
        Some(())
    }
}
pub(super) struct Node<'s, 'p> {
    source: &'s SourceFrame<'p>,
    splits: &'s [HullSplit],
    heads: Vec<NodeHead>,
    slots: Vec<SplitSlot>,
    columns: usize,
    base_rows: usize,
    plane_count: usize,
}
impl<'s, 'p> Node<'s, 'p> {
    pub(super) fn new(
        source: &'s SourceFrame<'p>,
        splits: &'s [HullSplit],
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        require_hull_environment().ok()?;
        if splits.len() > source.p.hidden_dim || !alive(source.end, poll) {
            return None;
        }
        let mut heads = alloc(
            source.heads.len(),
            NodeHead {
                kind: UNMAPPED,
                column: NONE,
                row: NONE,
            },
        )?;
        for (j, r) in source.heads.iter().enumerate() {
            heads[j].kind = r.kind;
        }
        for s in splits {
            if !alive(source.end, poll) {
                return None;
            }
            let h = heads.get_mut(usize::try_from(s.neuron).ok()?)?;
            if h.kind & SPLIT != 0 {
                return None;
            }
            h.kind = if h.kind == UNMAPPED {
                UNMAPPED | SPLIT
            } else if s.active {
                LIVE | SPLIT
            } else {
                ZERO | SPLIT
            };
        }
        let (mut columns, mut rows) = (source.p.input_dim, 0usize);
        for (j, h) in heads.iter_mut().enumerate() {
            if !alive(source.end, poll) {
                return None;
            }
            match h.kind & 7 {
                POS => {
                    h.column = columns.try_into().ok()?;
                    h.row = rows.try_into().ok()?;
                    columns = columns.checked_add(1)?;
                    rows = rows.checked_add(1)?;
                }
                NEG => {
                    h.column = columns.try_into().ok()?;
                    h.row = rows.try_into().ok()?;
                    let n = source.heads[j].len as usize;
                    columns = columns.checked_add(1 + n)?;
                    rows = rows.checked_add(n)?;
                }
                _ => {}
            }
        }
        let mut slots = alloc(
            splits.len(),
            SplitSlot {
                column: NONE,
                row: NONE,
            },
        )?;
        for (k, s) in splits.iter().enumerate() {
            let j = usize::try_from(s.neuron).ok()?;
            let norm: f64 = source.p.w_row(j).iter().map(|x| x.abs()).sum();
            finite(norm)?;
            if norm <= 0. {
                continue;
            }
            slots[k] = SplitSlot {
                column: columns.try_into().ok()?,
                row: rows.try_into().ok()?,
            };
            columns = columns.checked_add(1)?;
            rows = rows.checked_add(1)?;
        }
        alive(source.end, poll).then_some(Self {
            source,
            splits,
            heads,
            slots,
            columns,
            base_rows: rows,
            plane_count: source.plane_count,
        })
    }
    fn row_count(&self) -> Option<usize> {
        self.base_rows.checked_add(self.plane_count)
    }
    fn term(&self, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        let r = *self.source.heads.get(j)?;
        if k >= r.len as usize {
            return None;
        }
        let i = *self.source.inputs.get(r.start as usize + k)? as usize;
        let h = *self.heads.get(j)?;
        Some((i, h.column as usize + 1 + k, h.row as usize + k))
    }
    // Read the existing constructor's maps and actual columns before injection.
    // Any nominal/certified role mismatch declines ONLY optional augmentation.
    fn matches_lp(
        &self,
        lp: &HullLp,
        split_rows: &[Option<usize>],
        poll: &mut dyn FnMut() -> bool,
    ) -> bool {
        if lp.model.num_cols() != self.columns
            || lp.model.num_rows() != self.base_rows
            || lp.pos_rows.len() != self.heads.len()
            || lp.neg_rows.len() != self.heads.len()
            || split_rows.len() != self.slots.len()
        {
            return false;
        }
        for (j, h) in self.heads.iter().enumerate() {
            if !alive(self.source.end, poll) {
                return false;
            }
            match h.kind & 7 {
                POS => {
                    if lp.pos_rows[j] != Some(h.row as usize) || !lp.neg_rows[j].is_empty() {
                        return false;
                    }
                    let Some(r) = lp.model.row_at(h.row as usize) else {
                        return false;
                    };
                    if lp.model.row(r).0.last() != Some(&(h.column, 1.)) {
                        return false;
                    }
                }
                NEG => {
                    let n = self.source.heads[j].len as usize;
                    if lp.pos_rows[j].is_some() || lp.neg_rows[j].len() != n {
                        return false;
                    }
                    for k in 0..n {
                        let Some((i, c, r)) = self.term(j, k) else {
                            return false;
                        };
                        if lp.neg_rows[j][k] != (i, r) {
                            return false;
                        }
                        let Some(row) = lp.model.row_at(r) else {
                            return false;
                        };
                        let a = lp.model.row(row).0;
                        if a.len() != 3
                            || a[0].0 != i as u32
                            || a[1].0 != h.column
                            || a[2] != (c as u32, 1.)
                        {
                            return false;
                        }
                    }
                }
                _ => {
                    if lp.pos_rows[j].is_some() || !lp.neg_rows[j].is_empty() {
                        return false;
                    }
                }
            }
        }
        for (k, slot) in self.slots.iter().enumerate() {
            if slot.row == NONE {
                if split_rows[k].is_some() {
                    return false;
                }
            } else {
                if split_rows[k] != Some(slot.row as usize) {
                    return false;
                }
                let Some(row) = lp.model.row_at(slot.row as usize) else {
                    return false;
                };
                if lp.model.row(row).0.last() != Some(&(slot.column, 1.)) {
                    return false;
                }
            }
        }
        alive(self.source.end, poll)
    }
    // Nominal proposer advice only. Each canonical head expands once. The sink
    // MUST merge/finite-check duplicate columns before Model::add_row. The paid
    // PriceRun::nominal_batch below does that with its existing16Q scratch.
    fn nominal_plane(
        &self,
        slot: usize,
        sink: &mut dyn FnMut(usize, f64) -> Option<()>,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        let p = self.source.p;
        let plane = self
            .source
            .planes
            .get(slot)
            .filter(|_| slot < self.plane_count)?;
        let mut lower = plane.lower();
        for term in plane.terms() {
            let (atom, a) = term.semantic();
            if !alive(self.source.end, poll) {
                return None;
            }
            match atom {
                Atom::Input(i) => sink(i, a)?,
                Atom::Head(j) => match self.heads[j].kind & 7 {
                    ZERO => {}
                    LIVE => {
                        lower = finite(lower - a * p.b1[j])?;
                        for i in 0..p.input_dim {
                            if !alive(self.source.end, poll) {
                                return None;
                            }
                            sink(i, finite(a * p.w_row(j)[i])?)?;
                        }
                    }
                    POS => sink(self.heads[j].column as usize, a)?,
                    NEG => {
                        let u = self.source.bounds[j].upper;
                        lower = finite(lower - a * u)?;
                        sink(self.heads[j].column as usize, finite(-a * u)?)?;
                        for k in 0..self.source.heads[j].len as usize {
                            if !alive(self.source.end, poll) {
                                return None;
                            }
                            sink(self.term(j, k)?.1, -a)?;
                        }
                    }
                    _ => return None,
                },
                Atom::Selector(j) => match self.heads[j].kind & 7 {
                    ZERO => {}
                    LIVE => lower = finite(lower - a)?,
                    NEG => {
                        lower = finite(lower - a)?;
                        sink(self.heads[j].column as usize, -a)?;
                    }
                    _ => return None,
                },
            }
        }
        alive(self.source.end, poll).then_some(lower)
    }
}
// All rows are canonicalized before the first Model mutation. A numeric-only
// third-row refusal retains the original pair; base/cancellation failure drops
// the batch. This is proposer advice, NEVER a row license.
struct AdviceRow {
    lower: f64,
    terms: Vec<(Col, f64)>,
}
pub(super) struct AdviceBatch {
    rows: Vec<AdviceRow>,
    columns: usize,
    base_rows: usize,
    end: Instant,
}
impl AdviceBatch {
    pub(super) fn row_count(&self) -> usize {
        self.rows.len()
    }
    pub(super) fn append(self, lp: &mut HullLp, poll: &mut dyn FnMut() -> bool) -> Option<()> {
        require_hull_environment().ok()?;
        if lp.model.num_cols() != self.columns
            || lp.model.num_rows() != self.base_rows
            || self.base_rows.checked_add(self.rows.len())? > u32::MAX as usize
            || !alive(self.end, poll)
        {
            return None;
        }
        // No fallible emission/poll remains after insertion starts. The finite,
        // unique, in-range vectors eliminate add_row's duplicate-overflow panic.
        // Caller must recheck this SAME original end before starting LpSession;
        // an expired commit is not permission to start/renew a solve.
        for row in self.rows {
            lp.model.add_row(row.lower, f64::INFINITY, &row.terms);
        }
        Some(())
    }
}
// Owns one optional node's paid scratch only. No allocation during either price.
pub(super) struct PriceRun<'s, 'p> {
    node: Node<'s, 'p>,
    residual: Vec<I>,
    heads: Vec<I>,
    selectors: Vec<I>,
    attempts: u8,
    cancelled: bool,
    advice_started: bool,
}
// Bounded fixed descriptors fit the existing 1024-byte pricer allowance;
// variable Vec payloads are separately paid by 16Q, 52H and E row terms.
// This is nominal live payload, not compiler stack/RSS accounting.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(
    2 * size_of::<SourceFrame<'static>>()
        + size_of::<Node<'static, 'static>>()
        + size_of::<PriceRun<'static, 'static>>()
        + size_of::<AdviceBatch>()
        + 3 * size_of::<AdviceRow>()
        + 256
        <= 1024
);
impl<'s, 'p> PriceRun<'s, 'p> {
    pub(super) fn new(node: Node<'s, 'p>) -> Option<Self> {
        let n = node.columns;
        let h = node.heads.len();
        Some(Self {
            node,
            residual: alloc(n, I::zero())?,
            heads: alloc(h, I::zero())?,
            selectors: alloc(h, I::zero())?,
            attempts: 0,
            cancelled: false,
            advice_started: false,
        })
    }
    fn alive(&mut self, poll: &mut dyn FnMut() -> bool) -> bool {
        if self.cancelled {
            return false;
        }
        if !alive(self.node.source.end, poll) {
            self.cancelled = true;
            return false;
        }
        true
    }
    // Before the LP, reuse residual.lo as a nominal dense accumulator. Never
    // retain a raw expansion vector: U visits are paid, only UNIQUE E survives.
    pub(super) fn nominal_batch(
        &mut self,
        lp: &HullLp,
        split_rows: &[Option<usize>],
        expanded_cap: usize,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<AdviceBatch> {
        if self.advice_started || self.attempts != 0 || !self.alive(poll) {
            return None;
        }
        self.advice_started = true;
        let mut stopped = false;
        let matched = self.node.matches_lp(lp, split_rows, &mut || {
            let ok = poll();
            stopped |= !ok;
            ok
        });
        if stopped {
            self.cancelled = true;
        }
        if !matched || self.node.row_count()? > u32::MAX as usize {
            return None;
        }
        let k = self.node.plane_count;
        let mut rows = Vec::with_capacity(k);
        if rows.capacity() > k {
            return None;
        }
        let mut total = 0usize;
        for slot in 0..k {
            let remaining = expanded_cap.checked_sub(total)?;
            let row = self.nominal_row(slot, lp, remaining, poll);
            let Some(row) = row else {
                if slot != 2 || !self.alive(poll) {
                    return None;
                }
                // No row reconstruction or second allocation of the old pair.
                // All three-row charges remain paid when this proposal refuses.
                self.node.plane_count = 2;
                break;
            };
            total = total.checked_add(row.terms.len())?;
            rows.push(row);
        }
        require_hull_environment().ok()?;
        self.alive(poll).then_some(AdviceBatch {
            rows,
            columns: self.node.columns,
            base_rows: self.node.base_rows,
            end: self.node.source.end,
        })
    }
    fn nominal_row(
        &mut self,
        slot: usize,
        lp: &HullLp,
        remaining: usize,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<AdviceRow> {
        for c in 0..self.residual.len() {
            if !self.alive(poll) {
                return None;
            }
            self.residual[c] = I::zero();
        }
        let mut stopped = false;
        let lower = {
            let residual = &mut self.residual;
            self.node.nominal_plane(
                slot,
                &mut |c, a| {
                    let x = residual.get_mut(c)?;
                    x.lo = finite(x.lo + a)?;
                    Some(())
                },
                &mut || {
                    let ok = poll();
                    stopped |= !ok;
                    ok
                },
            )
        };
        if stopped {
            self.cancelled = true;
        }
        let lower = lower?;
        let mut count = 0usize;
        for c in 0..self.residual.len() {
            if !self.alive(poll) {
                return None;
            }
            if self.residual[c].lo != 0. {
                count = count.checked_add(1)?;
            }
        }
        if count > remaining {
            return None;
        }
        let mut terms = Vec::with_capacity(count);
        if terms.capacity() > count {
            return None;
        }
        for c in 0..self.residual.len() {
            if !self.alive(poll) {
                return None;
            }
            let a = self.residual[c].lo;
            if a != 0. {
                terms.push((lp.model.col_at(c)?, a));
            }
        }
        Some(AdviceRow { lower, terms })
    }
    fn add(&mut self, c: usize, a: I) -> Option<()> {
        let b = self.residual.get(c)?.add(a)?;
        *self.residual.get_mut(c)? = b;
        Some(())
    }
    pub(super) fn lower(&mut self, y: &[f64], poll: &mut dyn FnMut() -> bool) -> Option<f64> {
        require_hull_environment().ok()?;
        if self.cancelled || self.attempts >= 2 || y.len() != self.node.row_count()? {
            return None;
        }
        self.attempts += 1;
        if !self.alive(poll) {
            return None;
        }
        self.residual.fill(I::zero());
        self.selectors.fill(I::zero());
        let source = self.node.source;
        let p = source.p;
        let mut constant = I::point(p.d)?;
        for j in 0..p.hidden_dim {
            if !self.alive(poll) {
                return None;
            }
            self.heads[j] = I::point(p.c[j])?;
        }
        // Preaccumulate the paid two/three planes, then expand each head ONCE.
        // The two sign-pricing attempts remain two; row count is independent.
        for k in 0..self.node.plane_count {
            if !self.alive(poll) {
                return None;
            }
            let eta = positive(y[self.node.base_rows + k])?;
            if eta == 0. {
                continue;
            }
            let plane = &source.planes[k];
            constant = constant.add(I::point(plane.lower())?.scale(eta)?)?;
            for term in plane.terms() {
                let (atom, a) = term.semantic();
                if !self.alive(poll) {
                    return None;
                }
                let a = I::point(a)?.scale(-eta)?;
                match atom {
                    Atom::Input(i) => self.add(i, a)?,
                    Atom::Head(j) => self.heads[j] = self.heads[j].add(a)?,
                    Atom::Selector(j) => self.selectors[j] = self.selectors[j].add(a)?,
                }
            }
        }
        for j in 0..p.hidden_dim {
            if !self.alive(poll) {
                return None;
            }
            let h = self.node.heads[j];
            let a = self.heads[j];
            let selector = self.selectors[j];
            match h.kind & 7 {
                UNMAPPED => {
                    if a.lo != 0. || a.hi != 0. || selector.lo != 0. || selector.hi != 0. {
                        return None;
                    }
                }
                ZERO => {}
                LIVE => {
                    constant = constant.add(a.scale(p.b1[j])?)?.add(selector)?;
                    for i in 0..p.input_dim {
                        if !self.alive(poll) {
                            return None;
                        }
                        self.add(i, a.scale(p.w_row(j)[i])?)?;
                    }
                }
                POS => {
                    if selector.lo != 0. || selector.hi != 0. {
                        return None;
                    }
                    self.add(h.column as usize, a)?;
                    let eta = positive(y[h.row as usize])?;
                    constant = constant.add(I::point(p.b1[j])?.scale(eta)?)?;
                    self.add(h.column as usize, I::point(-eta)?)?;
                    for i in 0..p.input_dim {
                        if !self.alive(poll) {
                            return None;
                        }
                        self.add(i, I::point(p.w_row(j)[i])?.scale(eta)?)?;
                    }
                }
                NEG => {
                    let u = source.bounds[j].upper;
                    constant = constant.add(a.scale(u)?)?.add(selector)?;
                    self.add(h.column as usize, a.scale(-u)?.sub(selector)?)?;
                    for k in 0..source.heads[j].len as usize {
                        if !self.alive(poll) {
                            return None;
                        }
                        let (i, t, r) = self.node.term(j, k)?;
                        let eta = positive(y[r])?;
                        self.add(t, a.scale(-1.)?.sub(I::point(eta)?)?)?;
                        let w = p.w_row(j)[i];
                        let corner = if w > 0. { p.upper[i] } else { p.lower[i] };
                        constant = constant.add(I::point(w)?.scale(corner)?.scale(eta)?)?;
                        let width = I::point(p.upper[i])?.sub(I::point(p.lower[i])?)?;
                        self.add(h.column as usize, width.scale(w.abs())?.scale(-eta)?)?;
                        self.add(i, I::point(w)?.scale(-eta)?)?;
                    }
                }
                _ => return None,
            }
        }
        for k in 0..self.node.slots.len() {
            if !self.alive(poll) {
                return None;
            }
            let slot = self.node.slots[k];
            if slot.row == NONE {
                continue;
            }
            let split = self.node.splits[k];
            let j = usize::try_from(split.neuron).ok()?;
            let sigma = if split.active { 1. } else { -1. };
            let eta = positive(y[slot.row as usize])?;
            constant = constant.add(I::point(p.b1[j])?.scale(-sigma)?.scale(eta)?)?;
            for i in 0..p.input_dim {
                if !self.alive(poll) {
                    return None;
                }
                self.add(i, I::point(p.w_row(j)[i])?.scale(-sigma)?.scale(eta)?)?;
            }
        }
        // Stream boxes: no per-worker Q-box array or R-key array.
        let mut lower = constant.lo;
        for i in 0..p.input_dim {
            if !self.alive(poll) {
                return None;
            }
            lower = price(lower, self.residual[i], p.lower[i], p.upper[i])?;
        }
        for j in 0..p.hidden_dim {
            if !self.alive(poll) {
                return None;
            }
            let h = self.node.heads[j];
            let b = source.bounds[j];
            match h.kind & 7 {
                POS => lower = price(lower, self.residual[h.column as usize], 0., b.upper.max(0.))?,
                NEG => {
                    lower = price(lower, self.residual[h.column as usize], 0., 1.)?;
                    let range = I::point(b.upper)?.sub(I::point(b.lower)?)?.hi;
                    for k in 0..source.heads[j].len as usize {
                        if !self.alive(poll) {
                            return None;
                        }
                        lower = price(lower, self.residual[self.node.term(j, k)?.1], 0., range)?;
                    }
                }
                _ => {}
            }
        }
        // Split-elastic residuals have exact graph value0 and contribute0.
        require_hull_environment().ok()?;
        self.alive(poll).then_some(lower)
    }
}
fn positive(x: f64) -> Option<f64> {
    if x.is_nan() || x <= 0. {
        Some(0.)
    } else {
        finite(x)
    }
}
fn price(lower: f64, a: I, l: f64, u: f64) -> Option<f64> {
    finite((lower + a.minimum(l, u)?).next_down())
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_native_numeric_tests.rs"]
mod tests;
