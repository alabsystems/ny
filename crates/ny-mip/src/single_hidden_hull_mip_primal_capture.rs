// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Attempt-owned receipt survives every Option-returning root refusal.
use super::super::primal_advice::{self, Capture, Captured, RoleCounts, RootLpCounts};
use super::*;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HullLpPrimalReport {
    pub attempted: bool,
    pub admitted: bool,
    pub same_call: bool,
    pub columns: usize,
    pub positive_heads: usize,
    pub negative_heads: usize,
    pub work_units: usize,
    pub peak_bytes: usize,
    pub decoded: bool,
    pub bank_attempts: usize,
    pub banked: bool,
    pub stop_reason: &'static str,
}

pub(super) type Report = Arc<Mutex<HullLpPrimalReport>>;

pub(super) struct Attempt<'a> {
    source: &'a HullMarginProblem,
    end: Instant,
    pub report: Report,
    pub capture: Option<Capture<'a>>,
    account: Option<Captured<'a>>,
    finished: bool,
    taken: bool,
    columns: usize,
    census_started: bool,
    paid_work: usize,
    paid_peak: usize,
    paid_activity_debt: (usize, usize),
    activity_requested: bool,
    native_counts_requested: bool,
    native_counts_started: bool,
    native_counts: Option<RootLpCounts>,
    paid_native_counts_debt: (usize, usize),
}

// No new old-path fee: prove the complete fixed inline/receipt/report
// envelope still fits its existing 4096 allowance. Two Mutex/report copies
// plus two conservative 32-byte Arc controls and 512 bytes scalar scratch.
const _: () = assert!(
    size_of::<Attempt<'static>>()
        + primal_advice::FIXED_METADATA_BYTES
        + 2 * size_of::<Mutex<HullLpPrimalReport>>()
        + 64
        + 512
        <= 4096
);

impl<'a> Attempt<'a> {
    pub(super) fn new(source: &'a HullMarginProblem, end: Instant) -> Self {
        let paid_work = 32 * (source.hidden_dim + 1);
        let paid_peak = 4096;
        Self {
            source,
            end,
            // The public attempt preflights capacity; pay fixed receipt and
            // role metadata BEFORE the first optional heap allocation.
            report: Arc::new(Mutex::new(HullLpPrimalReport {
                attempted: true,
                work_units: paid_work,
                peak_bytes: paid_peak,
                stop_reason: "metadata-paid",
                ..HullLpPrimalReport::default()
            })),
            capture: None,
            account: None,
            finished: false,
            taken: false,
            columns: 0,
            census_started: false,
            paid_work,
            paid_peak,
            paid_activity_debt: (0, 0),
            activity_requested: false,
            native_counts_requested: false,
            native_counts_started: false,
            native_counts: None,
            paid_native_counts_debt: (0, 0),
        }
    }
    /// Latch the optional activity channel BEFORE census, construction and LP.
    /// It cannot recover advice discarded by an already finished old attempt.
    pub(super) fn new_with_activity(source: &'a HullMarginProblem, end: Instant) -> Self {
        let mut out = Self::new(source, end);
        out.activity_requested = true;
        out
    }
    /// Optional exact root Q/R/Z, with activity selected by the existing
    /// preference/fallback. Its extra fee is admitted before model construction.
    pub(super) fn new_with_native_counts(source: &'a HullMarginProblem, end: Instant) -> Self {
        let mut out = Self::new_with_activity(source, end);
        out.native_counts_requested = true;
        out
    }
    /// The first debit occurs BEFORE the first new role-census operation.
    pub(super) fn admit(&mut self, roles: &[Role]) {
        let Ok(mut report) = self.report.lock() else {
            return;
        };
        if self.census_started {
            return;
        }
        self.census_started = true;
        report.stop_reason = "metadata-deadline-or-shape";
        let p = self.source;
        if !live(self.end) || !capacity(p) || roles.len() != p.hidden_dim {
            return;
        }
        let Some(prefix) = p.hidden_dim.checked_add(1).and_then(|v| v.checked_mul(32)) else {
            return;
        };
        if prefix > primal_advice::CAP {
            return;
        }
        if report.work_units != prefix {
            return;
        }
        let mut counts = RoleCounts {
            positive: 0,
            negative: 0,
        };
        for role in roles {
            if !live(self.end) {
                report.stop_reason = "metadata-deadline";
                return;
            }
            match role {
                Role::PosUnstable => counts.positive += 1,
                Role::NegUnstable => counts.negative += 1,
                _ => {}
            }
        }
        report.positive_heads = counts.positive;
        report.negative_heads = counts.negative;
        report.stop_reason = "capture-quote";
        let quote = if self.native_counts_requested {
            primal_advice::quote_with_native_counts(p, counts).or_else(|| {
                primal_advice::quote_with_activity(p, counts)
                    .or_else(|| primal_advice::quote(p, counts))
            })
        } else if self.activity_requested {
            primal_advice::quote_with_activity(p, counts)
                .or_else(|| primal_advice::quote(p, counts))
        } else {
            primal_advice::quote(p, counts)
        };
        let Some(quote) = quote else {
            return;
        };
        let (work, bytes) = quote.debt();
        if work < prefix {
            return;
        }
        // Store the FULL irreversible debt before a constructor can decline.
        self.paid_work = work;
        self.paid_peak = bytes;
        self.paid_activity_debt = quote.activity_debt();
        self.paid_native_counts_debt = quote.root_lp_counts_debt();
        report.work_units = self.paid_work;
        report.peak_bytes = self.paid_peak;
        report.admitted = true;
        self.capture = Capture::new(p, quote, self.end);
        report.stop_reason = if self.capture.is_some() {
            "capture-paid"
        } else {
            "capture-constructor"
        };
    }
    /// Called only with the original root HullLp, before its existing solve.
    /// The quote is already paid. Set once BEFORE the O(R) header walk; partial
    /// cancellation/refusal retains debt and cannot retry on another model.
    fn record_root_lp_counts(&mut self, model: &Model, deadline: Instant) {
        if self.finished || self.native_counts_started || self.paid_native_counts_debt == (0, 0) {
            return;
        }
        self.native_counts_started = true;
        self.native_counts =
            RootLpCounts::from_root_model(self.source, model, self.end.min(deadline));
    }
    pub(super) fn enable_same_call(&mut self, columns: usize) -> bool {
        self.columns = columns;
        let enabled = self
            .capture
            .as_ref()
            .is_some_and(|c| c.accepts_columns(columns));
        if let Ok(mut r) = self.report.lock() {
            r.columns = columns;
            r.same_call = enabled;
        }
        enabled
    }
    pub(super) fn finish(&mut self, primal: Option<Box<[f64]>>, poll: &mut dyn FnMut() -> bool) {
        if self.finished {
            return;
        }
        self.finished = true;
        let account = if let Some(capture) = self.capture.take() {
            capture.finish(self.columns, primal, poll)
        } else {
            Captured::empty_with_activity(
                self.source,
                self.end,
                self.paid_work,
                self.paid_peak,
                self.paid_activity_debt,
            )
        }
        .with_root_lp_counts(self.native_counts.take(), self.paid_native_counts_debt);
        if let Ok(mut r) = self.report.lock() {
            r.decoded = account.has_point();
            r.stop_reason = if r.decoded {
                "decoded"
            } else if r.same_call {
                "missing-or-declined-primal"
            } else {
                r.stop_reason
            };
        }
        self.account = Some(account);
    }
    pub(super) fn take_account(&mut self) -> Option<Mutex<Captured<'a>>> {
        if self.taken {
            return None;
        }
        let end = self.end;
        self.finish(None, &mut || live(end));
        self.taken = true;
        self.account.take().map(Mutex::new)
    }
    pub(super) fn snapshot(&self) -> HullLpPrimalReport {
        self.report.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

impl HullRootMipCuts<'_> {
    #[must_use]
    pub fn lp_primal_report(&self) -> HullLpPrimalReport {
        self.primal_report
            .as_ref()
            .and_then(|r| r.lock().ok().map(|r| r.clone()))
            .unwrap_or_default()
    }
}

pub(super) fn propose_root_mip_cuts_inner<'a>(
    problem: &'a HullMarginProblem,
    deadline: Instant,
    mut advice: Option<&mut Attempt<'a>>,
) -> Option<HullRootMipCuts<'a>> {
    let deadline = ny_core::instance_budget::deadline().map_or(deadline, |d| d.min(deadline));
    if !capacity(problem) || !live(deadline) {
        return None;
    }
    require_hull_environment().ok()?;
    let bounds = preactivation_bounds(problem);
    let roles = classify(problem, &bounds);
    if !problem.c.iter().any(|c| *c < 0.0) {
        return None;
    }
    if let Some(a) = advice.as_deref_mut() {
        a.admit(&roles);
    }
    let lp = build_hull_lp_with_capture(
        problem,
        &roles,
        &bounds,
        advice.as_deref_mut().and_then(|a| a.capture.as_mut()),
    )?;
    if !live(deadline) {
        return None;
    }
    if let Some(a) = advice.as_deref_mut() {
        a.record_root_lp_counts(&lp.model, deadline);
    }
    let with_advice = advice
        .as_deref_mut()
        .is_some_and(|a| a.enable_same_call(lp.model.num_cols()));
    let Some((outcome, primal, proposal_lp_calls)) =
        one_lp(problem, &roles, &lp, deadline, with_advice)
    else {
        if let Some(a) = advice.as_deref_mut() {
            a.finish(None, &mut || live(deadline));
        }
        return None;
    };
    if let Some(a) = advice.as_deref_mut() {
        a.finish(primal, &mut || live(deadline));
    }
    if !outcome.duals_available || !live(deadline) {
        return None;
    }
    let lower = outcome.closed_form?;
    let m = multipliers_from_duals(problem, &roles, &lp, outcome.best_duals.as_deref()?)?;
    let mut cuts = prepare(problem, m, lower, deadline, &mut || live(deadline))?;
    cuts.proposal_lp_calls = proposal_lp_calls;
    if let Some(a) = advice {
        cuts.primal = a.take_account();
        cuts.primal_report = Some(Arc::clone(&a.report));
    }
    cuts.float_lp_value = outcome
        .float_dual_value
        .map(|v| v + lp.constant)
        .filter(|v| v.is_finite());
    // Optional separation consumes the existing stage allowance, never a new
    // LP or a new deadline. prepare() itself stays the individual-only seam.
    joint::discover_compact(&mut cuts, deadline, &mut || live(deadline));
    if !live(deadline) {
        return None;
    }
    Some(cuts)
}

fn one_lp(
    problem: &HullMarginProblem,
    roles: &[Role],
    lp: &HullLp,
    deadline: Instant,
    with_advice: bool,
) -> Option<(
    crate::lp_propose_certify::ProposeCertifyOutcome,
    Option<Box<[f64]>>,
    usize,
)> {
    let remaining = deadline.checked_duration_since(Instant::now())?;
    if remaining.is_zero() {
        return None;
    }
    let opts = SolveOpts::new()
        .with_time_limit(remaining)
        .with_deadline(deadline);
    let mut certify = |y: &[f64]| {
        let m = multipliers_from_duals(problem, roles, lp, y)?;
        certify_hull_lower_bound_polled(problem, &m, &mut || live(deadline))
    };
    let mut proposal_lp_calls = 0;
    let outcome = guarded_hull_proposal(|| {
        proposal_lp_calls += 1;
        if with_advice {
            crate::lp_propose_certify::propose_then_certify_min_with_advice(
                &lp.model,
                &lp.objective,
                &opts,
                ReplayPolicy::Skip,
                &mut certify,
            )
        } else {
            propose_then_certify_min(
                &lp.model,
                &lp.objective,
                &opts,
                ReplayPolicy::Skip,
                &mut certify,
            )
            .map(|outcome| (outcome, None))
        }
    });
    let (outcome, primal) = outcome.ok().and_then(|r| r.ok())?;
    Some((outcome, primal, proposal_lp_calls))
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_primal_capture_tests.rs"]
mod tests;
