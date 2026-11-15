// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Signed support retains the original authenticated model's publication owner.
use super::*;

/// One original root LP and original stage. All previous global rows survive.
#[must_use]
pub fn propose_root_mip_cuts_with_signed_refinement(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> Option<HullRootMipCuts<'_>> {
    propose_root_mip_cuts_with_signed_refinement_attempt(problem, deadline, continuation_reserve)
        .cuts
}

/// Telemetry from the one original LP attempt survives even when no cut owner
/// can be returned. It never grants proof, pass credit, or a renewed attempt.
pub struct HullRootMipCutAttempt<'a> {
    pub cuts: Option<HullRootMipCuts<'a>>,
    pub primal: HullLpPrimalReport,
}

#[must_use]
pub fn propose_root_mip_cuts_with_signed_refinement_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_policy(problem, deadline, continuation_reserve, false)
}
fn signed_attempt_policy(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
    prepaid: bool,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_root_policy(problem, deadline, continuation_reserve, prepaid, false)
}
fn signed_attempt_root_policy(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
    prepaid: bool,
    root: bool,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_portfolio_policy(
        problem,
        deadline,
        continuation_reserve,
        prepaid,
        root,
        false,
    )
}
fn signed_attempt_portfolio_policy(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
    prepaid: bool,
    root: bool,
    portfolio: bool,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_all_policy(
        problem,
        deadline,
        continuation_reserve,
        prepaid,
        root,
        portfolio,
        false,
    )
}
fn signed_attempt_all_policy(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
    prepaid: bool,
    root: bool,
    portfolio: bool,
    all_phases: bool,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_cube_policy(
        problem,
        deadline,
        continuation_reserve,
        prepaid,
        root,
        portfolio,
        all_phases,
        false,
    )
}
fn signed_attempt_cube_policy(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
    prepaid: bool,
    root: bool,
    portfolio: bool,
    all_phases: bool,
    cube: bool,
) -> HullRootMipCutAttempt<'_> {
    let end = ny_core::instance_budget::deadline().map_or(deadline, |d| d.min(deadline));
    if !capacity(problem) || !live(end) {
        return HullRootMipCutAttempt {
            cuts: None,
            primal: HullLpPrimalReport {
                stop_reason: "preflight",
                ..HullLpPrimalReport::default()
            },
        };
    }
    let mut attempt = if cube {
        primal_capture::Attempt::new_with_activity(problem, end)
    } else {
        primal_capture::Attempt::new(problem, end)
    };
    let cuts = signed_attempt_inner(
        problem,
        end,
        continuation_reserve,
        &mut attempt,
        prepaid,
        root,
        portfolio,
        all_phases,
        cube,
    );
    attempt.finish(None, &mut || live(end));
    HullRootMipCutAttempt {
        cuts,
        primal: attempt.snapshot(),
    }
}

/// The same single original LP/capture/proof owner. Only optional rows select
/// the funded canonical-first K2 successor during first publication.
#[must_use]
pub fn propose_root_mip_cuts_with_signed_group_allocation_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    let mut attempt = propose_root_mip_cuts_with_signed_refinement_attempt(
        problem,
        deadline,
        continuation_reserve,
    );
    if let Some(cuts) = attempt.cuts.as_mut() {
        cuts.group_allocation_mode = true;
    }
    attempt
}

/// Same original LP/capture/proof owner. Optional dense rows choose ONE
/// seed-useful gradient tail or independently optimized four-mask tau tail.
#[must_use]
pub fn propose_root_mip_cuts_with_signed_branch_tau_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    let mut attempt = propose_root_mip_cuts_with_signed_group_allocation_attempt(
        problem,
        deadline,
        continuation_reserve,
    );
    if let Some(cuts) = attempt.cuts.as_mut() {
        cuts.branch_tau_mode = true;
    }
    attempt
}

/// A new allocation policy: prepay ONE existing full/partial component union
/// before signed refinement. Original LP, checker/replay caps and clocks remain.
/// A refused escrow selects the legacy branch-tau schedule before optional work;
/// an admitted escrow is exclusive even if its later one-row attempt refuses.
#[must_use]
pub fn propose_root_mip_cuts_with_prepaid_joint_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_policy(problem, deadline, continuation_reserve, true)
}
/// One original LP; prepay one globally checked root-plan proposal before refinement.
#[must_use]
pub fn propose_root_mip_cuts_with_root_coupling_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_root_policy(problem, deadline, continuation_reserve, false, true)
}
/// Default successor: retain checked component recipes from a globally losing
/// half-mass Plan. Uses the same original LP, checker ledger and publication.
#[must_use]
pub fn propose_root_mip_cuts_with_mass_portfolio_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_portfolio_policy(problem, deadline, continuation_reserve, false, true, true)
}
/// Same original LP and plain mass portfolio; richer winner phase publication
/// is selected and prepaid before optional advice.
#[must_use]
pub fn propose_root_mip_cuts_with_mass_portfolio_all_phases_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_all_policy(
        problem,
        deadline,
        continuation_reserve,
        false,
        true,
        true,
        true,
    )
}
/// Same original LP and All/loser portfolio; one finite phase-cube support
/// per component, independently repaired against the same completed table.
#[must_use]
pub fn propose_root_mip_cuts_with_mass_portfolio_phase_cube_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    signed_attempt_cube_policy(
        problem,
        deadline,
        continuation_reserve,
        false,
        true,
        true,
        true,
        true,
    )
}
/// Next default policy: same original LP, prepaid source-licensed native rows.
/// Current Cube and all older entry points keep their original capture route.
#[must_use]
pub fn propose_root_mip_cuts_with_mass_portfolio_phase_cube_native_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    phase_cube_native_policy(
        problem,
        deadline,
        continuation_reserve,
        signed::NativePolicy::Rows,
    )
}
/// Checked conditional rows from an alternative mass allocation, using its
/// existing final all-mask pass and the original publication transaction.
#[must_use]
pub fn propose_root_mip_cuts_with_conditional_mass_portfolio_attempt(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> HullRootMipCutAttempt<'_> {
    phase_cube_native_policy(
        problem,
        deadline,
        continuation_reserve,
        signed::NativePolicy::Conditional,
    )
}
fn phase_cube_native_policy(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
    native: signed::NativePolicy,
) -> HullRootMipCutAttempt<'_> {
    let end = ny_core::instance_budget::deadline().map_or(deadline, |d| d.min(deadline));
    if !capacity(problem) || !live(end) {
        return HullRootMipCutAttempt {
            cuts: None,
            primal: HullLpPrimalReport {
                stop_reason: "preflight",
                ..HullLpPrimalReport::default()
            },
        };
    }
    let mut attempt = primal_capture::Attempt::new_with_native_counts(problem, end);
    let cuts = signed_attempt_inner_native(
        problem,
        end,
        continuation_reserve,
        &mut attempt,
        false,
        true,
        true,
        true,
        true,
        native,
    );
    attempt.finish(None, &mut || live(end));
    HullRootMipCutAttempt {
        cuts,
        primal: attempt.snapshot(),
    }
}
fn signed_attempt_inner<'a>(
    problem: &'a HullMarginProblem,
    end: Instant,
    continuation_reserve: Duration,
    attempt: &mut primal_capture::Attempt<'a>,
    prepaid: bool,
    root: bool,
    portfolio: bool,
    all_phases: bool,
    cube: bool,
) -> Option<HullRootMipCuts<'a>> {
    signed_attempt_inner_native(
        problem,
        end,
        continuation_reserve,
        attempt,
        prepaid,
        root,
        portfolio,
        all_phases,
        cube,
        signed::NativePolicy::Disabled,
    )
}
fn signed_attempt_inner_native<'a>(
    problem: &'a HullMarginProblem,
    end: Instant,
    continuation_reserve: Duration,
    attempt: &mut primal_capture::Attempt<'a>,
    prepaid: bool,
    root: bool,
    portfolio: bool,
    all_phases: bool,
    cube: bool,
    native: signed::NativePolicy,
) -> Option<HullRootMipCuts<'a>> {
    let mut cuts = propose_root_mip_cuts_inner(problem, end, Some(attempt))?;
    group::refine(&mut cuts, end, continuation_reserve);
    cuts.global_sdp = sdp::propose(&cuts, cuts.deadline, continuation_reserve);
    if !live(cuts.deadline) {
        return None;
    }
    if prepaid || root {
        cuts.group_allocation_mode = true;
        cuts.branch_tau_mode = true;
    }
    let (row, report) = if native == signed::NativePolicy::Conditional {
        signed::propose_conditional_portfolio(&cuts, cuts.deadline, continuation_reserve)
    } else if native == signed::NativePolicy::Rows {
        signed::propose_phase_cube_with_native_rows(&cuts, cuts.deadline, continuation_reserve)
    } else if cube {
        signed::propose_mass_portfolio_phase_cube(&cuts, cuts.deadline, continuation_reserve)
    } else if all_phases {
        signed::propose_mass_portfolio_all_phases(&cuts, cuts.deadline, continuation_reserve)
    } else if portfolio {
        signed::propose_mass_portfolio(&cuts, cuts.deadline, continuation_reserve)
    } else if root {
        signed::propose_root_coupling(&cuts, cuts.deadline, continuation_reserve)
    } else if prepaid {
        signed::propose_prepaid(&cuts, cuts.deadline, continuation_reserve)
    } else {
        signed::propose(&cuts, cuts.deadline, continuation_reserve)
    };
    cuts.signed = row;
    cuts.signed_report = Some(report);
    live(cuts.deadline).then_some(cuts)
}

impl HullRootMipCuts<'_> {
    pub fn native_funding_report(&self) -> SignedNativeFundingReport {
        self.signed
            .as_ref()
            .map_or_else(Default::default, |row| row.native_funding())
    }
    pub fn native_rows_report(&self) -> SignedNativeRowsReport {
        self.signed
            .as_ref()
            .map_or_else(Default::default, |row| row.native_stats())
    }
    /// Separate portfolio ledger; scalar-owner report shapes remain unchanged.
    pub fn mass_portfolio_report(&self) -> SignedMassPortfolioReport {
        self.signed
            .as_ref()
            .map_or_else(Default::default, |r| r.portfolio_stats())
    }
    pub(super) fn signed_phase_terms(&self) -> usize {
        self.signed
            .as_ref()
            .map_or(0, |r| r.component_phase_terms())
    }
    pub(super) fn signed_head_terms(&self) -> usize {
        if self.signed.is_some() {
            self.problem.c.iter().filter(|c| **c != 0.).count()
                + self.signed.as_ref().map_or(0, |r| r.component_head_terms())
        } else {
            0
        }
    }

    /// Second independently charged whole-q replay. Whole original binding
    /// must precede consumption by the existing opaque native publisher.
    #[must_use]
    pub fn signed_root_proof(&self, end: Instant) -> Option<HullAggregateRootProof> {
        let row = self.signed.as_ref()?;
        let end = end.min(self.deadline).min(row.authority());
        let end = ny_core::instance_budget::deadline().map_or(end, |d| d.min(end));
        if !live(end) {
            return None;
        }
        require_hull_environment().ok()?;
        let checked = row.replay_with_cache(self, end, &mut || live(end))?;
        // Only the completed second pass can fill this immutable child channel.
        // A nonpositive or later refused root proof never erases the cache.
        if live(end) && require_hull_environment().is_ok() {
            if let Some(cache) = checked.cache {
                let _ = self.signed_child_cache.set(cache);
            }
        }
        let proof = HullAggregateRootProof {
            lower: checked.root_lower,
            deadline: end,
            route: "signed-coordinate",
        };
        proof.verified().then_some(proof)
    }

    /// Refused, nonpositive or weaker signed support never replaces an old proof.
    #[must_use]
    pub fn best_signed_root_proof(&self, end: Instant) -> Option<HullAggregateRootProof> {
        let old = self.best_root_proof(end);
        let signed = self.signed_root_proof(end);
        match (old, signed) {
            (Some(a), Some(b)) => Some(if b.lower() > a.lower() { b } else { a }),
            (a, b) => a.or(b),
        }
    }
}
