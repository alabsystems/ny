// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Same-LP float advice has no proof authority; every capture is prepaid.
use super::HullMarginProblem;
use ay_milp::Col;
use std::any::Any;
use std::sync::Arc;
use std::time::Instant;

pub(super) const CAP: usize = 664_696;
// Up to32 dense quote probes,33 old quote probes, and bounded bookkeeping.
// Paid PRE-LP even if capture fails or no later dense quote fits.
pub(super) const SELECTION_WORK: usize = 8192;
const UNSET: usize = usize::MAX;
#[derive(Debug, Clone, Copy)]
pub(super) struct RoleCounts {
    pub positive: usize,
    pub negative: usize,
}
#[derive(Debug, Clone, Copy)]
enum Head {
    Unmapped,
    Zero,
    Positive(usize),
    Negative {
        mu: usize,
        upper: f64,
        start: usize,
        len: usize,
    },
}
#[derive(Debug)]
pub(super) struct Quote {
    source: usize,
    dimensions: (usize, usize),
    work: usize,
    peak_bytes: usize,
    max_cols: usize,
    max_terms: usize,
    counts: RoleCounts,
    activity: bool,
    root_counts: bool,
}
/// Counts come from the existing role-classification walk. Its extra counter
/// updates, all descriptor writes, AY prefix validation/copy, and NY decoding
/// are included here. No new LP or full-source geometry walk is authorized.
pub(super) fn quote(p: &HullMarginProblem, c: RoleCounts) -> Option<Quote> {
    let (d, h) = (p.input_dim, p.hidden_dim);
    if d == 0
        || d >= 63
        || h == 0
        || h > 512
        || d.checked_mul(h)? > 65_536
        || p.w1.len() != d.checked_mul(h)?
        || p.b1.len() != h
        || p.c.len() != h
        || p.lower.len() != d
        || p.upper.len() != d
        || c.positive.checked_add(c.negative)? > h
    {
        return None;
    }
    let max_terms = c.negative.checked_mul(d)?;
    let max_cols = d
        .checked_add(c.positive)?
        .checked_add(c.negative.checked_mul(d + 1)?)?;
    let descriptors = d.checked_add(h)?.checked_add(max_terms)?;
    let work = h
        .checked_add(1)?
        .checked_mul(32)?
        .checked_add(descriptors.checked_mul(16)?)?
        .checked_add(max_cols.checked_mul(64)?)?
        .checked_add(SELECTION_WORK)?;
    // Exact-length boxed payloads, with conservative simultaneous staging.
    // This is nominal payload, not allocator bookkeeping or solver RSS.
    let maps = d
        .checked_add(max_terms)?
        .checked_mul(size_of::<usize>())?
        .checked_add(h.checked_mul(size_of::<Head>())?)?;
    let peak_bytes = maps
        .checked_mul(2)?
        .checked_add(max_cols.checked_mul(16)?)?
        .checked_add(d.checked_add(h)?.checked_mul(32)?)?
        .checked_add(4096)?;
    (work <= CAP && peak_bytes <= (8 << 20)).then_some(Quote {
        source: std::ptr::from_ref(p) as usize,
        dimensions: (d, h),
        work,
        peak_bytes,
        max_cols,
        max_terms,
        counts: c,
        activity: false,
        root_counts: false,
    })
}
/// Optional SAME-call relaxed activity. Old callers retain the exact old quote.
/// The extra channel pays before its allocation; failure never refunds capture.
pub(super) fn quote_with_activity(p: &HullMarginProblem, c: RoleCounts) -> Option<Quote> {
    let mut q = quote(p, c)?;
    let work = c.negative.checked_add(1)?.checked_mul(32)?;
    let bytes = c.negative.checked_mul(16)?.checked_add(256)?;
    q.work = q.work.checked_add(work)?;
    q.peak_bytes = q.peak_bytes.checked_add(bytes)?;
    q.activity = true;
    (q.work <= CAP && q.peak_bytes <= (8 << 20)).then_some(q)
}
/// Optional exact ORIGINAL root-model size receipt. The fee is admitted before
/// Capture/model construction; it does not grant native work or row authority.
/// If the activity extension cannot fit, retain the original capture selection.
pub(super) fn quote_with_native_counts(p: &HullMarginProblem, c: RoleCounts) -> Option<Quote> {
    let mut q = quote_with_activity(p, c).or_else(|| quote(p, c))?;
    let (work, bytes) = root_lp_counts_fee(p)?;
    q.work = q.work.checked_add(work)?;
    q.peak_bytes = q.peak_bytes.checked_add(bytes)?;
    q.root_counts = true;
    (q.work <= CAP && q.peak_bytes <= (8 << 20)).then_some(q)
}
fn root_lp_counts_fee(p: &HullMarginProblem) -> Option<(usize, usize)> {
    // Each original positive has one row, each negative at most D rows.
    // D >= 1 makes R <= H*D, including every role combination.
    let rows = p.input_dim.checked_mul(p.hidden_dim)?;
    // Inline copies in Attempt/Captured plus their debt/flag/padding envelope.
    // Temporary census locals remain inside the old fixed stack allowance.
    let metadata = size_of::<Option<RootLpCounts>>()
        .checked_mul(2)?
        .checked_add(size_of::<usize>().checked_mul(4)?)?
        .checked_add(24)?;
    (metadata <= 256).then_some((rows.checked_add(1)?.checked_mul(64)?.checked_add(128)?, 256))
}
/// Opaque structural receipt from the original same-call root HullLp. These
/// exact counts are eligibility caps only: no row license, work, or LP grant.
#[derive(Debug, Clone, Copy)]
pub(super) struct RootLpCounts {
    source: usize,
    dimensions: (usize, usize),
    end: Instant,
    columns: usize,
    rows: usize,
    nonzeros: usize,
}
impl RootLpCounts {
    /// Trusted caller: Attempt, while its original HullLp is still borrowed,
    /// after the optional quote was paid and before the SAME existing solve.
    /// No coefficient walk or allocation: only public row slice lengths.
    pub(super) fn from_root_model(
        source: &HullMarginProblem,
        model: &ay_milp::Model,
        end: Instant,
    ) -> Option<Self> {
        let (d, h) = (source.input_dim, source.hidden_dim);
        if d == 0 || d >= 63 || h == 0 || h > 512 || !live(end) {
            return None;
        }
        let hd = h.checked_mul(d)?;
        if hd > 65_536 {
            return None;
        }
        let columns = model.num_cols();
        let rows = model.num_rows();
        let max_columns = d.checked_add(h.checked_mul(d.checked_add(1)?)?)?;
        let max_nonzeros = hd.checked_mul(3)?.checked_add(h)?;
        if columns < d || columns > max_columns || rows > hd {
            return None;
        }
        let mut nonzeros = 0usize;
        for index in 0..rows {
            if !live(end) {
                return None;
            }
            let row = model.row_at(index)?;
            nonzeros = nonzeros.checked_add(model.row(row).0.len())?;
            if nonzeros > max_nonzeros {
                return None;
            }
        }
        if !live(end) {
            return None;
        }
        Some(Self {
            source: std::ptr::from_ref(source) as usize,
            dimensions: (d, h),
            end,
            columns,
            rows,
            nonzeros,
        })
    }
    pub(super) fn columns(&self) -> usize {
        self.columns
    }
    pub(super) fn rows(&self) -> usize {
        self.rows
    }
    pub(super) fn nonzeros(&self) -> usize {
        self.nonzeros
    }
}
// Fixed inline envelope, including conservative simultaneous move/return
// copies. Attempt adds its report/Arc/scalar allowance and compile-time checks
// the combined OLD/refused/new metadata against the unchanged 4096 allowance.
pub(super) const FIXED_METADATA_BYTES: usize = size_of::<Capture<'static>>()
    + 2 * size_of::<Captured<'static>>()
    + 4 * size_of::<Quote>()
    + 2 * size_of::<RootLpCounts>()
    + 4 * size_of::<Point>();
impl Quote {
    /// Save/report this paid debt BEFORE moving the quote into a constructor
    /// that may refuse. A declined constructor never erases spent work.
    pub(super) fn debt(&self) -> (usize, usize) {
        (self.work, self.peak_bytes)
    }
    pub(super) fn root_lp_counts_debt(&self) -> (usize, usize) {
        if self.root_counts {
            // Accepted shape/fee arithmetic was checked by the quote.
            (64 * (self.dimensions.0 * self.dimensions.1 + 1) + 128, 256)
        } else {
            (0, 0)
        }
    }
    pub(super) fn activity_debt(&self) -> (usize, usize) {
        if self.activity {
            // The accepted quote bounded N by H <= 512 and checked both fees.
            (
                32 * (self.counts.negative + 1),
                16 * self.counts.negative + 256,
            )
        } else {
            (0, 0)
        }
    }
}
/// Move-only prepaid escrow. Creating it consumes the entire capture quote;
/// refusal, missing primal, or later failure never refunds the debt.
#[derive(Debug)]
pub(super) struct Capture<'a> {
    source: &'a HullMarginProblem,
    end: Instant,
    quote: Quote,
    input: Box<[usize]>,
    heads: Box<[Head]>,
    terms: Box<[usize]>,
    used_terms: usize,
    counts: RoleCounts,
}
impl<'a> Capture<'a> {
    pub(super) fn new(source: &'a HullMarginProblem, quote: Quote, end: Instant) -> Option<Self> {
        if !live(end)
            || quote.work > CAP
            || quote.source != std::ptr::from_ref(source) as usize
            || quote.dimensions != (source.input_dim, source.hidden_dim)
        {
            return None;
        }
        Some(Self {
            source,
            end,
            input: vec![UNSET; source.input_dim].into_boxed_slice(),
            heads: vec![Head::Unmapped; source.hidden_dim].into_boxed_slice(),
            terms: vec![UNSET; quote.max_terms].into_boxed_slice(),
            used_terms: 0,
            counts: RoleCounts {
                positive: 0,
                negative: 0,
            },
            quote,
        })
    }
    pub(super) fn matches_source(&self, source: &HullMarginProblem) -> bool {
        std::ptr::eq(self.source, source) && live(self.end)
    }
    pub(super) fn input(&mut self, index: usize, col: Col) -> Option<()> {
        let entry = self.input.get_mut(index)?;
        if *entry != UNSET || col.index() >= self.quote.max_cols {
            return None;
        }
        *entry = col.index();
        Some(())
    }
    fn head(&mut self, j: usize, value: Head) -> Option<()> {
        let entry = self.heads.get_mut(j)?;
        if !matches!(entry, Head::Unmapped) {
            return None;
        }
        *entry = value;
        Some(())
    }
    pub(super) fn zero(&mut self, j: usize) -> Option<()> {
        self.head(j, Head::Zero)
    }
    pub(super) fn positive(&mut self, j: usize, col: Col) -> Option<()> {
        if self.counts.positive >= self.quote.counts.positive || col.index() >= self.quote.max_cols
        {
            return None;
        }
        self.head(j, Head::Positive(col.index()))?;
        self.counts.positive += 1;
        Some(())
    }
    pub(super) fn negative(&mut self, j: usize, mu: Col, upper: f64) -> Option<()> {
        if self.counts.negative >= self.quote.counts.negative
            || !upper.is_finite()
            || mu.index() >= self.quote.max_cols
        {
            return None;
        }
        self.head(
            j,
            Head::Negative {
                mu: mu.index(),
                upper,
                start: self.used_terms,
                len: 0,
            },
        )?;
        self.counts.negative += 1;
        Some(())
    }
    pub(super) fn term(&mut self, j: usize, col: Col) -> Option<()> {
        let Head::Negative { start, len, .. } = self.heads.get_mut(j)? else {
            return None;
        };
        if start.checked_add(*len)? != self.used_terms || col.index() >= self.quote.max_cols {
            return None;
        }
        *self.terms.get_mut(self.used_terms)? = col.index();
        *len += 1;
        self.used_terms += 1;
        Some(())
    }
    /// Check BEFORE AY's new original-prefix allocation, not after copying.
    pub(super) fn accepts_columns(&self, model_cols: usize) -> bool {
        model_cols <= self.quote.max_cols && model_cols >= self.source.input_dim && live(self.end)
    }
    /// `primal` must be MOVED from the SAME AY float_advice_for_expr return.
    /// Outer solve absence or missing primal only drops advice; old duals remain.
    pub(super) fn finish(
        self,
        model_cols: usize,
        primal: Option<Box<[f64]>>,
        poll: &mut dyn FnMut() -> bool,
    ) -> Captured<'a> {
        let decoded = primal.and_then(|raw| self.decode(model_cols, &raw, poll));
        let (point, activity) = match decoded {
            Some((point, activity)) => (Some(point), activity),
            None => (None, None),
        };
        Captured {
            source: self.source,
            end: self.end,
            work: self.quote.work,
            peak_bytes: self.quote.peak_bytes,
            point,
            activity,
            activity_debt: self.quote.activity_debt(),
            root_counts: None,
            root_counts_debt: self.quote.root_lp_counts_debt(),
            banked: false,
            bank_end: None,
            owner: None,
        }
    }
    fn decode(
        &self,
        model_cols: usize,
        raw: &[f64],
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<(Point, Option<Box<[f64]>>)> {
        if !self.accepts_columns(model_cols)
            || raw.len() != model_cols
            || self.counts.positive != self.quote.counts.positive
            || self.counts.negative != self.quote.counts.negative
        {
            return None;
        }
        let mut input = vec![0.; self.source.input_dim].into_boxed_slice();
        for (i, &col) in self.input.iter().enumerate() {
            if !live(self.end) || !poll() {
                return None;
            }
            input[i] = finite(*raw.get(col)?)?;
        }
        let mut values = vec![0.; self.heads.len()].into_boxed_slice();
        let mut negative = Vec::with_capacity(self.quote.counts.negative);
        let mut positive = Vec::with_capacity(self.quote.counts.positive);
        // Exact-length extra channel; the raw primal is still live here.
        // Its simultaneous staging/retention is paid by the optional quote.
        let mut activity = self
            .quote
            .activity
            .then(|| vec![0.; self.quote.counts.negative].into_boxed_slice());
        for (j, head) in self.heads.iter().enumerate() {
            if !live(self.end) || !poll() {
                return None;
            }
            values[j] = match *head {
                Head::Unmapped => return None,
                Head::Zero => 0.,
                Head::Positive(col) => {
                    positive.push(j);
                    *raw.get(col)?
                }
                Head::Negative {
                    mu,
                    upper,
                    start,
                    len,
                } => {
                    let mut value = finite(upper * (1. - finite(*raw.get(mu)?)?))?;
                    for &col in self.terms.get(start..start.checked_add(len)?)? {
                        if !live(self.end) || !poll() {
                            return None;
                        }
                        value = finite(value - *raw.get(col)?)?;
                    }
                    negative.push(j);
                    if let Some(activity) = activity.as_mut() {
                        // mu belongs to the ideal LP; 1-mu is relaxed advice,
                        // not an original model's binary-column primal.
                        let z = finite(1. - finite(*raw.get(mu)?)?)?;
                        *activity.get_mut(negative.len() - 1)? = z.clamp(0., 1.);
                    }
                    value
                }
            };
            finite(values[j])?;
        }
        Some((
            Point {
                input,
                values,
                negative: negative.into_boxed_slice(),
                positive: positive.into_boxed_slice(),
            },
            activity,
        ))
    }
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn live(end: Instant) -> bool {
    Instant::now() < end && ny_core::instance_budget::deadline().is_none_or(|d| Instant::now() < d)
}
#[derive(Debug)]
pub(super) struct Point {
    // Stable/dead/inert values are placeholders and never enter these lists.
    pub input: Box<[f64]>,
    pub values: Box<[f64]>,
    pub negative: Box<[usize]>,
    pub positive: Box<[usize]>,
}
pub(super) struct Captured<'a> {
    source: &'a HullMarginProblem,
    end: Instant,
    work: usize,
    peak_bytes: usize,
    point: Option<Point>,
    // Parallel to Point.negative, kept outside Point to preserve old literals.
    // Empty optional metadata fits the old fixed 4096-byte metadata allowance;
    // the new quote separately pays all activity heap payload and decode work.
    activity: Option<Box<[f64]>>,
    // Subset of work/peak_bytes already paid, retained even without a point.
    activity_debt: (usize, usize),
    root_counts: Option<RootLpCounts>,
    root_counts_debt: (usize, usize),
    banked: bool,
    bank_end: Option<Instant>,
    owner: Option<Arc<dyn Any + Send + Sync>>,
}
impl std::fmt::Debug for Captured<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturedPrimalAdvice")
            .field("work", &self.work)
            .field("peak_bytes", &self.peak_bytes)
            .field("banked", &self.banked)
            .field("has_point", &self.point.is_some())
            .finish()
    }
}
impl<'a> Captured<'a> {
    pub(super) fn empty(
        source: &'a HullMarginProblem,
        end: Instant,
        work: usize,
        peak_bytes: usize,
    ) -> Self {
        Self {
            source,
            end,
            work,
            peak_bytes,
            point: None,
            activity: None,
            activity_debt: (0, 0),
            root_counts: None,
            root_counts_debt: (0, 0),
            banked: false,
            bank_end: None,
            owner: None,
        }
    }
    /// Preserve an accepted quote's optional debt when construction declines.
    pub(super) fn empty_with_activity(
        source: &'a HullMarginProblem,
        end: Instant,
        work: usize,
        peak_bytes: usize,
        activity_debt: (usize, usize),
    ) -> Self {
        let mut out = Self::empty(source, end, work, peak_bytes);
        out.activity_debt = activity_debt;
        out
    }
    /// Attempt attaches only its paid, complete original-model census. No
    /// primal is required. This does not change total debt or banking authority.
    pub(super) fn with_root_lp_counts(
        mut self,
        counts: Option<RootLpCounts>,
        debt: (usize, usize),
    ) -> Self {
        if debt != (0, 0)
            && Some(debt) == root_lp_counts_fee(self.source)
            && debt.0 <= self.work
            && debt.1 <= self.peak_bytes
        {
            self.root_counts_debt = debt;
            self.root_counts = counts.filter(|c| {
                c.source == std::ptr::from_ref(self.source) as usize
                    && c.dimensions == (self.source.input_dim, self.source.hidden_dim)
                    && c.end <= self.end
            });
        }
        self
    }
    pub(super) fn has_point(&self) -> bool {
        self.point.is_some()
    }
    pub(super) fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }
    /// Called exactly once by Budget::new BEFORE signed search/reserves. Caller
    /// atomically banks this debt and peak payload in the original ledger.
    /// This method itself does not authorize spending or a new checker slot.
    pub(super) fn bank_once<T: Any + Send + Sync>(
        &mut self,
        source: &HullMarginProblem,
        end: Instant,
        ledger: &Arc<T>,
        debit: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Option<()> {
        if self.banked || !std::ptr::eq(self.source, source) || end > self.end {
            return None;
        }
        self.banked = true;
        self.bank_end = Some(end);
        if !debit(self.work, self.peak_bytes) {
            return None;
        }
        self.owner = Some(ledger.clone());
        Some(())
    }
    pub(super) fn debt_for<T: Any + Send + Sync>(
        &self,
        source: &HullMarginProblem,
        end: Instant,
        ledger: &Arc<T>,
    ) -> Option<usize> {
        (self.banked && std::ptr::eq(self.source, source) && end <= self.end && live(end))
            .then_some(())?;
        if end > self.bank_end? {
            return None;
        }
        let owner = self.owner.as_ref()?;
        if Arc::as_ptr(owner).cast::<()>() != Arc::as_ptr(ledger).cast::<()>() {
            return None;
        }
        Some(self.work)
    }
    pub(super) fn point<T: Any + Send + Sync>(
        &self,
        source: &HullMarginProblem,
        end: Instant,
        ledger: &Arc<T>,
    ) -> Option<&Point> {
        self.debt_for(source, end, ledger)?;
        self.point.as_ref()
    }
    /// Already banked optional subset, not another reserve or spending grant.
    /// Missing/invalid raw advice retains this receipt without claiming values.
    pub(super) fn relaxed_activity_debt<T: Any + Send + Sync>(
        &self,
        source: &HullMarginProblem,
        end: Instant,
        ledger: &Arc<T>,
    ) -> Option<(usize, usize)> {
        self.debt_for(source, end, ledger)?;
        Some(self.activity_debt)
    }
    /// Advisory relaxed ideal-LP activities from the SAME raw return. No proof
    /// or selector-column authority is carried by these finite clamped values.
    pub(super) fn relaxed_activity<T: Any + Send + Sync>(
        &self,
        source: &HullMarginProblem,
        end: Instant,
        ledger: &Arc<T>,
    ) -> Option<(&[usize], &[f64])> {
        self.debt_for(source, end, ledger)?;
        let point = self.point.as_ref()?;
        let activity = self.activity.as_ref()?;
        (activity.len() == point.negative.len())
            .then_some((point.negative.as_ref(), activity.as_ref()))
    }
    /// Copy under the ORIGINAL bank/search clock. A later native admission may
    /// own this structural advice, but this accessor never renews capture life.
    pub(super) fn root_lp_counts<T: Any + Send + Sync>(
        &self,
        source: &HullMarginProblem,
        end: Instant,
        ledger: &Arc<T>,
    ) -> Option<RootLpCounts> {
        self.debt_for(source, end, ledger)?;
        let counts = self.root_counts?;
        let fee = root_lp_counts_fee(self.source)?;
        (self.root_counts_debt == fee
            && fee.0 <= self.work
            && fee.1 <= self.peak_bytes
            && counts.source == std::ptr::from_ref(self.source) as usize
            && counts.dimensions == (self.source.input_dim, self.source.hidden_dim)
            && end <= counts.end)
            .then_some(counts)
    }
    pub(super) fn work(&self) -> usize {
        self.work
    }
}

#[cfg(test)]
mod native_debt_tests {
    use super::super::{build_hull_lp, classify, preactivation_bounds};
    use super::*;
    use std::time::Duration;

    #[test]
    fn native_root_counts_require_the_retained_exact_paid_census_debt() {
        let p = HullMarginProblem::new(
            vec![1.],
            vec![0.],
            1,
            &[-1.],
            &[0.],
            1,
            &[(0, 1.)],
            0.,
            vec![-1.],
            vec![1.],
        )
        .unwrap();
        let end = Instant::now() + Duration::from_secs(30);
        let bounds = preactivation_bounds(&p);
        let roles = classify(&p, &bounds);
        let lp = build_hull_lp(&p, &roles, &bounds).unwrap();
        let counts = RootLpCounts::from_root_model(&p, &lp.model, end).unwrap();
        let quote = quote_with_native_counts(
            &p,
            RoleCounts {
                positive: 0,
                negative: 1,
            },
        )
        .unwrap();
        let debt = quote.debt();
        let fee = quote.root_lp_counts_debt();
        let mut captured = Capture::new(&p, quote, end)
            .unwrap()
            .finish(lp.model.num_cols(), None, &mut || true)
            .with_root_lp_counts(Some(counts), fee);
        let ledger = Arc::new(());
        let mut debits = Vec::new();
        captured
            .bank_once(&p, end, &ledger, &mut |work, bytes| {
                debits.push((work, bytes));
                true
            })
            .unwrap();
        assert_eq!(debits, [debt]);
        assert!(captured.root_lp_counts(&p, end, &ledger).is_some());
        for forged in [
            (0, 0),
            (fee.0 - 1, fee.1),
            (fee.0, fee.1 - 1),
            (usize::MAX, fee.1),
        ] {
            captured.root_counts_debt = forged;
            assert!(captured.root_lp_counts(&p, end, &ledger).is_none());
            assert_eq!(captured.debt_for(&p, end, &ledger), Some(debt.0));
            assert_eq!((captured.work, captured.peak_bytes), debt);
            assert_eq!(debits, [debt]);
        }
        captured.root_counts_debt = fee;
        let retained = captured.root_lp_counts(&p, end, &ledger).unwrap();
        assert_eq!(
            (retained.columns(), retained.rows(), retained.nonzeros()),
            (counts.columns(), counts.rows(), counts.nonzeros()),
        );
        assert!(captured
            .bank_once(&p, end, &ledger, &mut |_, _| true)
            .is_none());
        assert_eq!(debits, [debt]);
    }
}
