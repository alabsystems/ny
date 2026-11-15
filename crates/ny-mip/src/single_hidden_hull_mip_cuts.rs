// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Transfer completed hull-LP proposals into independently checked MIP rows.
//! No solver status, float objective, or bundle itself grants a verdict.

use super::*;
use crate::ir::{Row, RowSpec};
use crate::MipParts;
use ny_core::Bound;
#[path = "single_hidden_hull_mip_words.rs"]
mod words;
use words::RetainedWords;

pub const ROOT_MIP_CUT_MAX_INPUTS: usize = 128;
pub const ROOT_MIP_CUT_MAX_HEADS: usize = 512;
pub const ROOT_MIP_CUT_MAX_CELLS: usize = 65_536;

/// Non-authoritative description of the exact encoder inputs. The appender
/// independently reconstructs and compares every row and column before use.
pub struct HullMipEncoding<'a> {
    pub weights: &'a [Vec<f64>],
    pub biases: &'a [Vec<f64>],
    pub dimensions: &'a [usize],
    pub hidden_bounds: &'a [Vec<Bound>],
    pub first_output: usize,
    pub second_output: usize,
    pub reverse: bool,
    pub unsafe_row: Row,
}

#[derive(Debug)]
struct Majorant {
    head: usize,
    g: Vec<f64>,
    upper: f64,
}

/// Opaque, immutable ownership of one completed LP vector and its exact Source.
/// This deliberately does not use HullRootReport's possibly newer fallback
/// bound: that legacy report does not retain fallback-attaining multipliers.
#[derive(Debug)]
pub struct HullRootMipCuts<'a> {
    problem: &'a HullMarginProblem,
    multipliers: HullMultipliers,
    checked_lower: f64,
    planes: Vec<Majorant>,
    joint: Vec<joint::JointMajorant>,
    joint_attempted: usize,
    joint_completed: usize,
    joint_searched: bool,
    compact_mode: bool,
    compact: Option<compact::Aggregate>,
    group_budget: Option<group::Budget>,
    global_sdp: Option<sdp::GlobalMajorant>,
    signed: Option<signed::SignedMajorant>,
    signed_report: Option<SignedRootReport>,
    signed_child_cache: std::sync::OnceLock<std::sync::Arc<SignedChildCache>>,
    float_lp_value: Option<f64>,
    proposal_lp_calls: usize,
    deadline: Instant,
    primal: Option<std::sync::Mutex<primal_advice::Captured<'a>>>,
    primal_report: Option<primal_capture::Report>,
    group_allocation_mode: bool,
    branch_tau_mode: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HullMipCutReport {
    pub cuts: usize,
    pub individual_cuts: usize,
    pub joint_cuts: usize,
    pub joint_pairs_attempted: usize,
    pub joint_pairs_completed: usize,
    /// Rows actually exported by compact objective aggregation (zero or one).
    pub compact_cuts: usize,
    pub compact_pairs: usize,
    /// Completed groups in the one selected objective aggregate.
    pub compact_groups: usize,
    pub group_attempts: usize,
    pub group_search_masks: usize,
    pub group_replay_masks: usize,
    pub group_work: usize,
    pub group_peak_bytes: usize,
    /// Independent full-q replay, never the LP-attained scalar.
    pub compact_root_lower: Option<f64>,
    pub input_terms: usize,
    /// Certificate for the completed exported vector, never a float LP bound.
    pub checked_hull_lower: f64,
    /// Telemetry only, never sufficient to publish Verified.
    pub float_lp_value: Option<f64>,
    pub proposal_lp_calls: usize,
    /// One separately replayed global row; counters are captured at append.
    pub global_sdp: Option<GlobalSdpReport>,
    /// Attempt data survives even when no new signed row was selected.
    pub signed: Option<SignedRootReport>,
}

fn live(deadline: Instant) -> bool {
    Instant::now() < deadline
        && ny_core::instance_budget::deadline().is_none_or(|d| Instant::now() < d)
}

fn capacity(problem: &HullMarginProblem) -> bool {
    problem.input_dim <= ROOT_MIP_CUT_MAX_INPUTS
        && problem.hidden_dim <= ROOT_MIP_CUT_MAX_HEADS
        && problem
            .hidden_dim
            .checked_mul(problem.input_dim)
            .is_some_and(|n| n <= ROOT_MIP_CUT_MAX_CELLS)
}

/// One LP only, under a caller-owned absolute deadline. Both dual signs pass
/// the existing Source checker; only its actual attaining vector is retained.
#[must_use]
pub fn propose_root_mip_cuts(
    problem: &HullMarginProblem,
    deadline: Instant,
) -> Option<HullRootMipCuts<'_>> {
    propose_root_mip_cuts_inner(problem, deadline, None)
}

/// Optional complete group separation under the SAME original cut allowance.
/// The caller predetermines time left for the existing short-tree fallback.
#[must_use]
pub fn propose_root_mip_cuts_with_groups(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> Option<HullRootMipCuts<'_>> {
    let mut cuts = propose_root_mip_cuts(problem, deadline)?;
    group::refine(&mut cuts, deadline, continuation_reserve);
    live(cuts.deadline).then_some(cuts)
}

fn prepare<'a>(
    problem: &'a HullMarginProblem,
    multipliers: HullMultipliers,
    completed_lower: f64,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<HullRootMipCuts<'a>> {
    if !capacity(problem) || !live(deadline) || !poll() {
        return None;
    }
    require_hull_environment().ok()?;
    let checked_lower = certify_hull_lower_bound_polled(problem, &multipliers, poll)?;
    if checked_lower.to_bits() != completed_lower.to_bits() {
        return None;
    }
    let mut planes = Vec::new();
    for (head, coefficient) in problem.c.iter().enumerate() {
        if !live(deadline) || !poll() {
            return None;
        }
        if *coefficient >= 0.0 {
            continue;
        }
        let g = multipliers.g[head * problem.input_dim..(head + 1) * problem.input_dim].to_vec();
        let upper = majorant_offset_upper(
            problem.w_row(head),
            problem.b1[head],
            &g,
            &problem.lower,
            &problem.upper,
        )?;
        planes.push(Majorant { head, g, upper });
    }
    if planes.is_empty() || !live(deadline) || !poll() {
        return None;
    }
    require_hull_environment().ok()?;
    Some(HullRootMipCuts {
        problem,
        multipliers,
        checked_lower,
        planes,
        joint: Vec::new(),
        joint_attempted: 0,
        joint_completed: 0,
        joint_searched: false,
        compact_mode: false,
        compact: None,
        group_budget: None,
        global_sdp: None,
        signed: None,
        signed_report: None,
        signed_child_cache: std::sync::OnceLock::new(),
        float_lp_value: None,
        proposal_lp_calls: 0,
        deadline,
        primal: None,
        primal_report: None,
        group_allocation_mode: false,
        branch_tau_mode: false,
    })
}

/// Opaque result of this bundle's independently checked complete phase tree.
/// The public report is telemetry; only this private-owned object's live
/// verified() gate can be consumed by the original-instance caller.
#[derive(Debug)]
pub struct HullMipContinuation {
    report: bab::HullBabReport,
    signed_child: Option<SignedChildReport>,
    deadline: Instant,
}

impl HullMipContinuation {
    pub fn report(&self) -> &bab::HullBabReport {
        &self.report
    }
    pub fn verified(&self) -> bool {
        live(self.deadline) && require_hull_environment().is_ok() && self.report.verified()
    }
    pub fn signed_child_report(&self) -> Option<&SignedChildReport> {
        self.signed_child.as_ref()
    }
}

/// Opaque positive whole-root aggregate replay. It is a real-arithmetic
/// hull proof, not a typed OriginalReplay/Source E1 or AY UNSAT receipt.
#[derive(Debug)]
pub struct HullAggregateRootProof {
    lower: f64,
    deadline: Instant,
    route: &'static str,
}
impl HullAggregateRootProof {
    pub fn lower(&self) -> f64 {
        self.lower
    }
    pub fn route(&self) -> &'static str {
        self.route
    }
    pub fn verified(&self) -> bool {
        self.lower.is_finite()
            && self.lower > 0.0
            && live(self.deadline)
            && require_hull_environment().is_ok()
    }
}

impl HullRootMipCuts<'_> {
    /// Reuse this immutable original root only. A later caller end cannot
    /// renew its stored stage clock, and no conditional child row is exported.
    /// Original model/encoding binding is still required before publication.
    #[must_use]
    pub fn continue_hull_bab(
        &self,
        end: Instant,
        options: bab::HullBabOptions,
    ) -> Option<HullMipContinuation> {
        let end = end.min(self.deadline);
        let end = ny_core::instance_budget::deadline().map_or(end, |d| d.min(end));
        if !live(end) {
            return None;
        }
        // Preserve the attaining vector/individual-plane seal as well as the
        // freshly replayed scalar. This checks every original negative head.
        let negative: Vec<_> = self
            .problem
            .c
            .iter()
            .enumerate()
            .filter_map(|(j, c)| (*c < 0.0).then_some(j))
            .collect();
        if negative.len() != self.planes.len() {
            return None;
        }
        for (&head, plane) in negative.iter().zip(&self.planes) {
            let start = head.checked_mul(self.problem.input_dim)?;
            let end = start.checked_add(self.problem.input_dim)?;
            let g = self.multipliers.g.get(start..end)?;
            if plane.head != head
                || g.len() != plane.g.len()
                || g.iter()
                    .zip(&plane.g)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                || plane.g.iter().any(|v| !v.is_finite())
            {
                return None;
            }
        }
        signed::child::continue_native(self, end, options)
    }

    /// Same root validation and original continuation clock; additionally use
    /// the once-bound committed native row bank when its original grant exists.
    #[must_use]
    pub fn continue_hull_bab_with_native_rows(
        &self,
        end: Instant,
        options: bab::HullBabOptions,
    ) -> Option<HullMipContinuation> {
        let end = end.min(self.deadline);
        let end = ny_core::instance_budget::deadline().map_or(end, |d| d.min(end));
        if !live(end) {
            return None;
        }
        // Preserve the attaining vector/individual-plane seal as well as the
        // freshly replayed scalar. This checks every original negative head.
        let negative: Vec<_> = self
            .problem
            .c
            .iter()
            .enumerate()
            .filter_map(|(j, c)| (*c < 0.0).then_some(j))
            .collect();
        if negative.len() != self.planes.len() {
            return None;
        }
        for (&head, plane) in negative.iter().zip(&self.planes) {
            let start = head.checked_mul(self.problem.input_dim)?;
            let end = start.checked_add(self.problem.input_dim)?;
            let g = self.multipliers.g.get(start..end)?;
            if plane.head != head
                || g.len() != plane.g.len()
                || g.iter()
                    .zip(&plane.g)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                || plane.g.iter().any(|v| !v.is_finite())
            {
                return None;
            }
        }
        signed::child::continue_native_with_rows(self, end, options)
    }

    fn group_counts(&self) -> (usize, usize, usize, usize, usize) {
        let Some(b) = self.group_budget.as_ref() else {
            return (0, 0, 0, 0, 0);
        };
        b.state.lock().map_or((0, 0, 0, 0, 0), |s| {
            (s.attempts, s.masks, s.replay_masks, s.work, s.peak_bytes)
        })
    }

    fn joint_row_count(&self) -> usize {
        if self.compact_mode {
            usize::from(self.compact.is_some())
        } else {
            self.joint.len()
        }
    }

    pub fn checked_lower(&self) -> f64 {
        self.checked_lower
    }
    pub fn proposal_lp_calls(&self) -> usize {
        self.proposal_lp_calls
    }
    /// Retained mandatory planes/aggregate rows, before optional component export.
    /// The actual committed count is append_to's report.cuts, which also includes
    /// completed finite component rows; len is not a future publication promise.
    pub fn len(&self) -> usize {
        self.planes.len()
            + self.joint_row_count()
            + usize::from(self.global_sdp.is_some())
            + usize::from(self.signed.is_some())
    }
    pub fn is_empty(&self) -> bool {
        self.planes.is_empty() && self.joint_row_count() == 0
    }

    /// Prepare the ENTIRE checked row set before the first append. A refusal
    /// leaves the IR, binary metadata and marker unchanged. After commit these
    /// valid rows remain valid even if the caller's clock subsequently expires;
    /// ordinary AY/publication still uses that same original absolute clock.
    #[must_use]
    pub fn append_to(
        &self,
        parts: &mut MipParts,
        encoding: &HullMipEncoding<'_>,
        deadline: Instant,
    ) -> Option<HullMipCutReport> {
        self.append_with_poll(parts, encoding, deadline, &mut || live(deadline))
    }

    fn append_with_poll(
        &self,
        parts: &mut MipParts,
        encoding: &HullMipEncoding<'_>,
        deadline: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<HullMipCutReport> {
        // Optional families may refuse without aborting the append. A caller's
        // cancellation is different: once observed, every later barrier fails.
        let mut cancelled = false;
        let result = self.append_with_poll_inner(parts, encoding, deadline, &mut || {
            if !cancelled {
                cancelled = !poll();
            }
            !cancelled
        });
        if result.is_none() {
            if let Some(signed) = &self.signed {
                signed.abort_component_rows();
            }
        }
        result
    }

    // Every refusal precedes IR mutation. The outer transaction revokes
    // staged rows/header context without refunding work or reopening latches.
    fn append_with_poll_inner(
        &self,
        parts: &mut MipParts,
        encoding: &HullMipEncoding<'_>,
        deadline: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<HullMipCutReport> {
        let deadline = deadline.min(self.deadline);
        let deadline = ny_core::instance_budget::deadline().map_or(deadline, |d| d.min(deadline));
        if !live(deadline) || !poll() {
            return None;
        }
        require_hull_environment().ok()?;
        let checked = certify_hull_lower_bound_polled(self.problem, &self.multipliers, poll)?;
        if checked.to_bits() != self.checked_lower.to_bits() {
            return None;
        }
        let rows = binding::prepare_rows(self, parts, encoding, deadline, poll)?;
        if !live(deadline) || !poll() {
            return None;
        }
        require_hull_environment().ok()?;
        let input_terms = binding::input_terms(self, &rows)?;
        let compact_root_lower = self
            .compact
            .as_ref()
            .and_then(|row| compact::root_lower(self, row, deadline, poll));
        if !live(deadline) || !poll() {
            return None;
        }
        require_hull_environment().ok()?;
        let count = rows.len();
        if let Some(signed) = &self.signed {
            signed.arm_native_rows(parts, count);
            signed.reserve_portfolio_headers(parts, count)?;
        }
        for row in rows {
            parts.problem.add_row(
                row.lb,
                row.ub,
                row.coeffs.into_iter().map(|(i, c)| (crate::ir::Col(i), c)),
            );
        }
        if let Some(signed) = &self.signed {
            signed.commit_component_rows();
            signed.publish_native_rows(parts, count);
        }
        let group_counts = self.group_counts();
        Some(HullMipCutReport {
            cuts: count,
            individual_cuts: self.planes.len(),
            joint_cuts: self.joint_row_count(),
            joint_pairs_attempted: self.joint_attempted,
            joint_pairs_completed: self.joint_completed,
            compact_cuts: usize::from(self.compact_mode && self.compact.is_some()),
            compact_pairs: self.compact.as_ref().map_or(0, |row| row.pairs.len()),
            compact_root_lower,
            compact_groups: self.compact.as_ref().map_or(0, |r| r.groups.len()),
            group_attempts: group_counts.0,
            group_search_masks: group_counts.1,
            group_replay_masks: group_counts.2,
            group_work: group_counts.3,
            group_peak_bytes: group_counts.4,
            input_terms,
            checked_hull_lower: checked,
            float_lp_value: self.float_lp_value,
            proposal_lp_calls: self.proposal_lp_calls,
            global_sdp: self.global_sdp.as_ref().map(sdp::GlobalMajorant::stats),
            signed: self
                .signed
                .as_ref()
                .map(signed::SignedMajorant::stats)
                .or_else(|| self.signed_report.clone()),
        })
    }
}

#[path = "single_hidden_hull_mip_cut_binding.rs"]
mod binding;

#[path = "single_hidden_hull_mip_joint.rs"]
mod joint;

#[cfg(test)]
#[path = "single_hidden_hull_mip_cut_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_continuation_tests.rs"]
mod continuation_tests;

#[path = "single_hidden_hull_mip_compact.rs"]
mod compact;

#[path = "single_hidden_hull_mip_group.rs"]
mod group;

#[path = "single_hidden_hull_mip_sdp.rs"]
mod sdp;
pub use sdp::GlobalSdpReport;
#[path = "single_hidden_hull_mip_sdp_admission.rs"]
mod sdp_admission;
pub use sdp_admission::propose_root_mip_cuts_with_global_sdp;

#[path = "single_hidden_hull_mip_signed.rs"]
mod signed;
pub use signed::child::SignedChildReport;
pub(in crate::single_hidden_hull) use signed::child::{
    BoundCache as SignedChildBound, Cache as SignedChildCache,
};
pub(super) use signed::{NativeRows as SignedNativeRows, NativeVisit};
pub use signed::{
    SignedDirectionReport, SignedMassPortfolioReport, SignedPhaseCellReport, SignedPhaseCubeReport,
    SignedRootCouplingReport, SignedRootReport,
};
pub use signed::{SignedNativeFundingReport, SignedNativeRowsReport};
#[path = "single_hidden_hull_mip_signed_admission.rs"]
mod signed_admission;
pub use signed_admission::{
    propose_root_mip_cuts_with_conditional_mass_portfolio_attempt,
    propose_root_mip_cuts_with_mass_portfolio_all_phases_attempt,
    propose_root_mip_cuts_with_mass_portfolio_attempt,
    propose_root_mip_cuts_with_mass_portfolio_phase_cube_attempt,
    propose_root_mip_cuts_with_mass_portfolio_phase_cube_native_attempt,
    propose_root_mip_cuts_with_prepaid_joint_attempt,
    propose_root_mip_cuts_with_root_coupling_attempt,
    propose_root_mip_cuts_with_signed_branch_tau_attempt,
    propose_root_mip_cuts_with_signed_group_allocation_attempt,
    propose_root_mip_cuts_with_signed_refinement,
    propose_root_mip_cuts_with_signed_refinement_attempt, HullRootMipCutAttempt,
};

#[path = "single_hidden_hull_mip_primal_capture.rs"]
mod primal_capture;
use primal_capture::propose_root_mip_cuts_inner;
pub use primal_capture::HullLpPrimalReport;
