// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independently checked mixed-head whole-q lower geometry; proposals are not proof.
use super::*;
use std::sync::Arc;
#[path = "single_hidden_hull_mip_signed_budget.rs"]
mod budget;
use budget::*;
#[path = "single_hidden_hull_mip_signed_geometry.rs"]
mod geometry;
use geometry::{Component, Plan};
#[path = "single_hidden_hull_mip_signed_batch.rs"]
mod batch;
#[path = "single_hidden_hull_mip_signed_certificate.rs"]
mod certificate;
#[path = "single_hidden_hull_mip_signed_component_rows.rs"]
mod component_rows;
#[path = "single_hidden_hull_mip_signed_corner.rs"]
mod corners;
#[path = "single_hidden_hull_mip_signed_native_publication.rs"]
mod native_publication;
pub use component_rows::SignedComponentRowsReport;
pub(in crate::single_hidden_hull) use native_publication::{
    bank::Visit as NativeVisit, BoundPublished as NativeRows,
};
pub use native_publication::{SignedNativeFundingReport, SignedNativeRowsReport};
#[path = "single_hidden_hull_mip_signed_component_phase.rs"]
pub(super) mod phase;
pub use phase::{SignedPhaseCellReport, SignedPhaseCubeReport, SignedPhaseRowsReport};
#[path = "single_hidden_hull_mip_signed_joint.rs"]
mod joint;
pub use joint::SignedJointRowsReport;
#[path = "single_hidden_hull_mip_signed_domination.rs"]
mod domination;
#[path = "single_hidden_hull_mip_signed_prepaid_joint.rs"]
mod prepaid;
#[path = "single_hidden_hull_mip_signed_root_coupling.rs"]
mod root_coupling;
pub use domination::SignedDominationReport;
pub use root_coupling::SignedRootCouplingReport;
#[path = "single_hidden_hull_mip_signed_child.rs"]
pub(in crate::single_hidden_hull) mod child;
#[path = "single_hidden_hull_mip_signed_direction.rs"]
mod direction;
#[path = "single_hidden_hull_mip_signed_mass.rs"]
mod mass;
#[path = "single_hidden_hull_mip_signed_mass_portfolio.rs"]
mod portfolio;
pub use portfolio::SignedMassPortfolioReport;
#[path = "single_hidden_hull_mip_signed_refine.rs"]
mod refine;
#[path = "single_hidden_hull_mip_signed_search.rs"]
mod search;
pub use direction::SignedDirectionReport;
pub use mass::SignedMassReport;
use refine::{empty_report, report};

#[derive(Debug, Clone, PartialEq)]
pub struct SignedRootReport {
    pub attempted: bool,
    pub admitted: bool,
    pub components: usize,
    pub refined_components: usize,
    pub assigned_positive_heads: usize,
    pub positive_appearances: usize,
    pub split_positive_heads: usize,
    pub mass: SignedMassReport,
    pub component_rows: SignedComponentRowsReport,
    pub negative_heads: usize,
    pub coordinate_proposals: usize,
    pub completed_masks: usize,
    pub checked_masks: usize,
    pub source_replays: usize,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub lower: Option<f64>,
    pub direction: SignedDirectionReport,
    pub root_coupling: SignedRootCouplingReport,
    pub stop_reason: &'static str,
}
#[derive(Debug)]
pub(super) struct SignedMajorant {
    source: usize,
    signature: Arc<RetainedWords>,
    witness: Arc<RetainedWords>,
    plan: Plan,
    g: Vec<f64>,
    constant: f64,
    lower: f64,
    cost: usize,
    budget: Budget,
}
#[derive(Debug)]
pub(super) struct Replay {
    pub g: Vec<f64>,
    pub constant: f64,
    pub root_lower: f64,
    pub cache: Option<Arc<child::Cache>>,
    pub component_minima: Option<Vec<f64>>,
    pub conditional: Option<phase::Conditional>,
    pub cube: Option<phase::cube::Collected>,
}
fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}
fn shape(cuts: &HullRootMipCuts<'_>) -> Option<()> {
    let p = cuts.problem;
    let cells = p.input_dim.checked_mul(p.hidden_dim)?;
    if !capacity(p)
        || p.input_dim == 0
        || p.hidden_dim == 0
        || p.w1.len() != cells
        || p.b1.len() != p.hidden_dim
        || p.c.len() != p.hidden_dim
        || p.lower.len() != p.input_dim
        || p.upper.len() != p.input_dim
        || cuts.multipliers.alpha.len() != p.hidden_dim
        || cuts.multipliers.g.len() != cells
        || !cuts.checked_lower.is_finite()
        || p.w1
            .iter()
            .chain(&p.b1)
            .chain(&p.c)
            .chain([&p.d])
            .chain(&p.lower)
            .chain(&p.upper)
            .chain(&cuts.multipliers.alpha)
            .chain(&cuts.multipliers.g)
            .any(|v| !v.is_finite())
        || p.lower.iter().zip(&p.upper).any(|(l, u)| l > u)
    {
        return None;
    }
    Some(())
}
fn signature<'a>(cuts: &'a HullRootMipCuts<'_>) -> impl Iterator<Item = u64> + 'a {
    let p = cuts.problem;
    [
        p.input_dim as u64,
        p.hidden_dim as u64,
        p.w1.len() as u64,
        p.b1.len() as u64,
        p.c.len() as u64,
        p.lower.len() as u64,
        p.upper.len() as u64,
        cuts.multipliers.alpha.len() as u64,
        cuts.multipliers.g.len() as u64,
        cuts.checked_lower.to_bits(),
    ]
    .into_iter()
    .chain(
        p.w1.iter()
            .chain(&p.b1)
            .chain(&p.c)
            .chain([&p.d])
            .chain(&p.lower)
            .chain(&p.upper)
            .chain(&cuts.multipliers.alpha)
            .chain(&cuts.multipliers.g)
            .map(|v| v.to_bits()),
    )
}
fn witness(plan: &Plan) -> Vec<u64> {
    let mut out = vec![plan.components.len() as u64, plan.leftover.len() as u64];
    out.extend(plan.leftover.iter().map(|v| *v as u64));
    for c in &plan.components {
        out.extend([
            c.negative.len() as u64,
            c.positive.len() as u64,
            c.positive_units.len() as u64,
            c.g.len() as u64,
            c.alpha.len() as u64,
        ]);
        out.extend(c.negative.iter().chain(&c.positive).map(|v| *v as u64));
        out.extend(c.positive_units.iter().map(|u| *u as u64));
        out.extend(c.g.iter().chain(&c.alpha).map(|v| v.to_bits()));
    }
    out
}
impl SignedMajorant {
    pub(super) fn g(&self) -> &[f64] {
        &self.g
    }
    pub(super) fn proposed_lower(&self) -> f64 {
        self.lower
    }
    pub(super) fn authority(&self) -> Instant {
        self.budget.authority
    }
    pub(super) fn stats(&self) -> SignedRootReport {
        report(
            &self.budget,
            Some(&self.plan),
            Some(self.proposed_lower()),
            true,
        )
    }
    pub(super) fn child_stats(&self) -> SignedChildReport {
        child::stats(&self.budget)
    }
    pub(super) fn portfolio_stats(&self) -> SignedMassPortfolioReport {
        portfolio::report(&self.budget)
    }
    pub(super) fn portfolio_admitted(&self) -> bool {
        self.budget
            .state
            .lock()
            .is_ok_and(|c| c.portfolio.report().admitted)
    }
    pub(super) fn component_head_terms(&self) -> usize {
        self.budget.state.lock().map_or(0, |c| {
            c.portfolio.report().prepared_head_terms
                + c.component_rows.prepared_head_terms
                + c.component_rows.phase.prepared_head_terms
                + c.component_rows.joint.prepared_head_terms
                + c.component_rows.domination.prepared_head_terms
        })
    }
    pub(super) fn phase_all_mode(&self) -> bool {
        phase::all_mode(&self.budget)
    }
    pub(super) fn phase_admitted(&self) -> bool {
        self.budget
            .state
            .lock()
            .is_ok_and(|c| c.component_rows.phase.admitted)
    }
    pub(super) fn component_phase_terms(&self) -> usize {
        self.budget.state.lock().map_or(0, |c| {
            phase::cube::selector_terms(&c)
                .and_then(|n| n.checked_add(c.portfolio.report().phase_prepared_rows))
                .and_then(|n| n.checked_add(c.portfolio.report().cube.prepared_selector_terms))
                .unwrap_or(usize::MAX)
        })
    }
    pub(super) fn reserve_portfolio_headers(
        &self,
        parts: &mut MipParts,
        additional: usize,
    ) -> Option<()> {
        if let Some(grant) = phase::claim_ir_headers(self, parts, additional).ok()? {
            // Both capabilities validate before a single aggregate allocation.
            portfolio::claim_ir_headers_with_phase(self, parts, additional, &grant)?;
            parts.problem.reserve_rows_exact(additional);
            phase::record_ir_headers(self, parts, additional, &grant);
            portfolio::record_ir_headers_with_phase(self, parts, additional, &grant);
            Some(())
        } else {
            portfolio::reserve_ir_headers(self, parts, additional)
        }
    }
    pub(super) fn commit_component_rows(&self) {
        component_rows::commit(&self.budget);
    }
    pub(super) fn arm_native_rows(&self, parts: &MipParts, additional: usize) {
        native_publication::arm(self, parts, additional);
    }
    pub(super) fn native_stats(&self) -> SignedNativeRowsReport {
        native_publication::stats(&self.budget)
    }
    pub(super) fn native_funding(&self) -> SignedNativeFundingReport {
        native_publication::funding(&self.budget)
    }
    pub(super) fn bind_native_rows<'p>(
        &self,
        cuts: &HullRootMipCuts<'p>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<NativeRows<'p>> {
        let end = end.min(native_publication::publication_end(self)?);
        if !live(end) {
            return None;
        }
        native_publication::bind(self, cuts, end, poll)
    }
    pub(super) fn publish_native_rows(&self, parts: &MipParts, additional: usize) {
        native_publication::publish(self, parts, additional);
    }
    pub(super) fn abort_component_rows(&self) {
        component_rows::abort_prepared(&self.budget);
    }
    pub(super) fn replay_with_cache(
        &self,
        cuts: &HullRootMipCuts<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Replay> {
        self.replay_inner(cuts, end, poll, true)
    }
    pub(super) fn replay_inner(
        &self,
        cuts: &HullRootMipCuts<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
        collect: bool,
    ) -> Option<Replay> {
        let end = end.min(self.budget.authority).min(cuts.deadline);
        let mut cancelled = false;
        let mut guard = || {
            let ok = !cancelled && live(end) && poll();
            cancelled |= !ok;
            ok
        };
        let poll: &mut dyn FnMut() -> bool = &mut guard;
        if !live(end) || !poll() || self.source != std::ptr::from_ref(cuts.problem) as usize {
            return None;
        }
        self.budget.proof(self.cost, true)?;
        shape(cuts)?;
        if !signature(cuts).eq(self.signature.iter().copied())
            || witness(&self.plan).as_slice() != self.witness.as_slice()
        {
            return None;
        }
        // Component minima belong to the first complete publication pass,
        // including scalar-only callers. Child-cache collection stays second.
        let (first, second) = {
            let counts = self.budget.state.lock().ok()?;
            (
                counts.replay_attempts == 1 && counts.source_replays == 0,
                counts.replay_attempts == 2 && counts.source_replays == 1,
            )
        };
        let permit = (collect && second)
            .then(|| child::admit(cuts, &self.plan, &self.budget, end))
            .flatten();
        let components = first && component_rows::begin(&self.budget, end);
        let mut corner = corners::begin(
            cuts,
            &self.plan,
            &self.budget,
            &self.witness,
            first,
            second && permit.is_some(),
        );
        let mut c = certificate::check_replay(
            cuts,
            &self.plan,
            &self.budget,
            end,
            poll,
            if components {
                phase::cube::collecting(&self.budget, &self.plan)
            } else {
                permit.is_some()
            },
            components,
            corner.as_mut(),
        )?;
        if c.constant.to_bits() != self.constant.to_bits()
            || c.root_lower.to_bits() != self.lower.to_bits()
            || c.g
                .iter()
                .map(|v| v.to_bits())
                .ne(self.g().iter().map(|v| v.to_bits()))
        {
            return None;
        }
        require_hull_environment().ok()?;
        if !live(end) || !poll() {
            return None;
        }
        let mut corner_done = match corner {
            Some(frame) => frame.finish(&c, cuts.problem.d, end, poll)?,
            None => None,
        };
        let mut cube_cancelled = false;
        let cube = if components && phase::cube::collecting(&self.budget, &self.plan) {
            c.tables.take().and_then(|values| {
                phase::cube::Collected::finish(self, cuts, values, c.cell.take(), end, &mut || {
                    let ok = poll();
                    cube_cancelled |= !ok;
                    ok
                })
            })
        } else {
            None
        };
        if cube_cancelled || !live(end) {
            return None;
        }
        let cache = permit.and_then(|permit| {
            let tables = c.tables.take()?;
            child::Cache::finish(
                cuts,
                &self.plan,
                &c,
                tables,
                self.signature.clone(),
                &self.budget,
                permit,
                if second { corner_done.take() } else { None },
                end,
                poll,
            )
        });
        require_hull_environment().ok()?;
        if !live(end) || !poll() {
            return None;
        }
        if let Some(value) = c.conditional.as_mut() {
            // Bind opaque geometry ONLY after every existing replay bit/clock check.
            value.bind(&self.budget, &self.witness);
        }
        {
            let mut counts = self.budget.state.lock().ok()?;
            match corner_done.take() {
                Some(corners::Finished::Axes(axes, advice)) => {
                    corners::publish_axes(&mut counts, axes, advice, &self.witness)
                }
                Some(corners::Finished::Cover(cover)) => {
                    corners::publish_cover(&mut counts, cover, &self.witness)
                }
                _ => {}
            }
            counts.source_replays += 1;
            if let Some(minima) = &c.component_minima {
                counts.component_rows.collected_components = minima.len();
                counts.component_rows.stop_reason = "collected";
                if counts.component_rows.phase.admitted {
                    if c.conditional.is_some() {
                        counts.component_rows.phase.collected_components = minima.len();
                        counts.component_rows.phase.stop_reason = "collected";
                    } else {
                        counts.component_rows.phase.stop_reason = "coverage";
                    }
                }
            }
        }
        Some(Replay {
            g: c.g,
            constant: c.constant,
            root_lower: c.root_lower,
            cache,
            component_minima: c.component_minima,
            conditional: c.conditional,
            cube,
        })
    }
}
fn row(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    checked: certificate::Checked,
    cost: usize,
    b: &Budget,
    seal: &Arc<RetainedWords>,
) -> SignedMajorant {
    SignedMajorant {
        source: std::ptr::from_ref(cuts.problem) as usize,
        signature: seal.clone(),
        witness: Arc::new(witness(plan).into()),
        plan: plan.clone(),
        g: checked.g,
        constant: checked.constant,
        lower: checked.root_lower,
        cost,
        budget: b.clone(),
    }
}
/// No-row paths retain attempted-work telemetry. Old planes, tuple, scalar and
/// previously checked whole-q candidates are never mutated by this search.
pub(super) fn propose(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy(cuts, authority, reserve, prepaid::Policy::Legacy)
}
pub(super) fn propose_prepaid(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy(cuts, authority, reserve, prepaid::Policy::OneJoint)
}
pub(super) fn propose_root_coupling(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy(cuts, authority, reserve, prepaid::Policy::RootCoupling)
}
pub(super) fn propose_mass_portfolio(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy(cuts, authority, reserve, prepaid::Policy::RootPortfolio)
}
pub(super) fn propose_mass_portfolio_all_phases(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy(
        cuts,
        authority,
        reserve,
        prepaid::Policy::RootPortfolioAllPhases,
    )
}
pub(super) fn propose_mass_portfolio_phase_cube(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy(
        cuts,
        authority,
        reserve,
        prepaid::Policy::RootPortfolioPhaseCube,
    )
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativePolicy {
    Disabled,
    Rows,
    Conditional,
}

fn propose_policy(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
    policy: prepaid::Policy,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy_native(cuts, authority, reserve, policy, NativePolicy::Disabled)
}
/// New PRIVATE opt-in entry. The original caller must first request the typed
/// count receipt before its SAME root LP; missing receipt declines only reuse.
/// No public route calls this draft entry yet.
pub(super) fn propose_phase_cube_with_native_rows(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy_native(
        cuts,
        authority,
        reserve,
        prepaid::Policy::RootPortfolioPhaseCube,
        NativePolicy::Rows,
    )
}
pub(super) fn propose_conditional_portfolio(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_policy_native(
        cuts,
        authority,
        reserve,
        prepaid::Policy::RootPortfolioPhaseCube,
        NativePolicy::Conditional,
    )
}
fn propose_policy_native(
    cuts: &HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
    policy: prepaid::Policy,
    native: NativePolicy,
) -> (Option<SignedMajorant>, SignedRootReport) {
    if require_hull_environment().is_err() || shape(cuts).is_none() {
        return (None, empty_report("shape"));
    }
    let budget = match native {
        NativePolicy::Disabled => Budget::new(cuts, authority, reserve),
        NativePolicy::Rows => Budget::new_with_native_rows(cuts, authority, reserve),
        NativePolicy::Conditional => {
            Budget::new_with_portfolio_conditions(cuts, authority, reserve)
        }
    };
    let Some(mut b) = budget else {
        return (None, empty_report("admission"));
    };
    b.portfolio_enabled = matches!(
        policy,
        prepaid::Policy::RootPortfolio
            | prepaid::Policy::RootPortfolioAllPhases
            | prepaid::Policy::RootPortfolioPhaseCube
    );
    b.phase_all_enabled = matches!(
        policy,
        prepaid::Policy::RootPortfolioAllPhases | prepaid::Policy::RootPortfolioPhaseCube
    );
    b.phase_cube_enabled = policy == prepaid::Policy::RootPortfolioPhaseCube;
    if policy == prepaid::Policy::Legacy {
        propose_with_budget(cuts, &b, &mut || true)
    } else {
        propose_with_policy(cuts, &mut b, policy, &mut || true)
    }
}
fn propose_with_budget(
    cuts: &HullRootMipCuts<'_>,
    b: &Budget,
    poll: &mut dyn FnMut() -> bool,
) -> (Option<SignedMajorant>, SignedRootReport) {
    propose_with_policy(cuts, &mut b.clone(), prepaid::Policy::Legacy, poll)
}
fn propose_with_policy(
    cuts: &HullRootMipCuts<'_>,
    b: &mut Budget,
    policy: prepaid::Policy,
    poll: &mut dyn FnMut() -> bool,
) -> (Option<SignedMajorant>, SignedRootReport) {
    if shape(cuts).is_none()
        || b.authority > cuts.deadline
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.dimensions
            != (
                cuts.problem.input_dim,
                cuts.problem.hidden_dim,
                cuts.problem.c.iter().filter(|c| **c > 0.).count(),
            )
    {
        b.stop("source");
        return (None, report(b, None, None, false));
    }
    let p = cuts.problem;
    if b.charge(64 * (p.input_dim * p.hidden_dim + p.hidden_dim + 1))
        .is_none()
    {
        return (None, report(b, None, None, false));
    }
    let mut floor = cuts.checked_lower;
    if let Some(old) = &cuts.compact {
        if let Some(v) = compact::root_lower(cuts, old, b.search_end, poll) {
            floor = floor.max(v);
        }
    }
    if let Some(old) = &cuts.global_sdp {
        floor = floor.max(old.proposed_lower());
    }
    if floor > 0. {
        b.stop("old_positive");
        return (None, report(b, None, Some(floor), false));
    }
    let Some(mut plan) = geometry::plan(cuts, b) else {
        b.stop("geometry");
        return (None, report(b, None, None, false));
    };
    let Some(mut cost) = plan.cost(p) else {
        b.stop("cost");
        return (None, report(b, Some(&plan), None, false));
    };
    if b.reserve(cost).is_none() || b.proof(cost, false).is_none() {
        b.stop("reserve");
        return (None, report(b, Some(&plan), None, false));
    }
    let Some(mut initial) = certificate::check(cuts, &plan, b, b.search_end, poll) else {
        b.stop("baseline");
        return (None, report(b, Some(&plan), None, false));
    };
    let root_funded =
        matches!(
            policy,
            prepaid::Policy::RootCoupling
                | prepaid::Policy::RootPortfolio
                | prepaid::Policy::RootPortfolioAllPhases
                | prepaid::Policy::RootPortfolioPhaseCube
        ) && root_coupling::run(cuts, &mut plan, &mut initial, &mut cost, floor, b, poll);
    if root_funded {
        let _ = b.trim_unspent_reservation(cost);
    }
    let seal = Arc::new(signature(cuts).collect());
    let mut current = initial.root_lower;
    let _ = component_rows::admit(cuts, &plan, b);
    // Complete native debit BEFORE the first owner clone or mass/refinement.
    // Missing/refused reservation is an O(1) no-op for every old entry.
    let _ = native_publication::fund(cuts, b);
    let funded =
        root_funded || (policy == prepaid::Policy::OneJoint && prepaid::admit(cuts, &plan, b));
    if current <= 0. && (current > floor || funded) {
        let _ = child::prepay(cuts, &plan, b);
    }
    // Keep the old native pair and ordinary child holds, then prioritize an
    // optional third Cell-family row and shared-input root/child floors.
    // This changes optional allocation policy, not the original quota/clock.
    let _ = native_publication::upgrade(cuts, b);
    // Prefer a paid root-only endpoint cover. Admission refusal retains the
    // old conditional-corner admission; no completed proof is presumed here.
    if corners::prepay_adaptive(cuts, &plan, b).is_none() {
        let _ = corners::prepay(cuts, &plan, b);
    }
    let _ = phase::cube::cell::input_phase_resources::prepay(cuts, b);
    // Optional root-only family preserves every preceding paid capability.
    let _ = corners::root_ray::prepay(cuts, &plan, b);
    // A checked non-improving baseline can own optional rows in this new policy.
    // Existing old rows/root proof survive; no positive proof is inferred here.
    let mut best = (current > floor || funded).then(|| row(cuts, &plan, initial, cost, b, &seal));
    let selection_floor = if funded { current } else { floor };
    let shared_first =
        b.portfolio_phase_enabled || mass::pressured(cuts, &plan, cost, b).unwrap_or(true);
    if shared_first {
        mass::refine(
            cuts,
            &mut plan,
            &mut current,
            &mut best,
            selection_floor,
            b,
            &seal,
            poll,
        );
        if let Some(selected_cost) = plan.cost(p) {
            let retained_cost = if b.prepaid.is_some() || b.root_coupling.is_some() {
                best.as_ref()
                    .map_or(selected_cost, |owner| selected_cost.max(owner.cost))
            } else {
                selected_cost
            };
            let _ = b.trim_unspent_reservation(retained_cost);
        }
    }
    if let Some(selected_cost) = plan.cost(p) {
        refine::refine(
            cuts,
            &mut plan,
            &mut current,
            &mut best,
            selection_floor,
            selected_cost,
            b,
            &seal,
            poll,
        );
    }
    if !shared_first {
        mass::refine(
            cuts,
            &mut plan,
            &mut current,
            &mut best,
            selection_floor,
            b,
            &seal,
            poll,
        );
    }
    // No alternative with a larger publication cost survives final selection.
    if let Some(selected_cost) = plan.cost(p) {
        let retained_cost = if b.prepaid.is_some() || b.root_coupling.is_some() {
            best.as_ref()
                .map_or(selected_cost, |owner| selected_cost.max(owner.cost))
        } else {
            selected_cost
        };
        let _ = b.trim_unspent_reservation(retained_cost);
    }
    let stats = report(b, Some(&plan), Some(current), best.is_some());
    if !live(b.authority) || require_hull_environment().is_err() {
        return (
            None,
            SignedRootReport {
                admitted: false,
                stop_reason: "expired",
                ..stats
            },
        );
    }
    (best, stats)
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_retention_tests.rs"]
mod retention_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_tests.rs"]
mod tests;
