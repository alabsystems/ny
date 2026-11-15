// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-v3-bankable: the multi-objective COVER CERTIFICATION system of the
//! disjunctive graph multi-objective BaB (the lane vit_2023 takes).
//!
//! # Why this exists
//!
//! vit_2023 is the largest open board gap, and ny's vit `unsat` stays
//! UNBANKED because its lane had nothing between the search and publication:
//!
//! * **completeness was trusted.** The multi-objective drivers decide
//!   `Verified` when the queue drains with no unresolved flag and publish
//!   through the counted `BetaCrownVerdict` legacy door. Nothing checked that
//!   the closed leaves tile the root, per objective row; a child dropped by a
//!   bookkeeping path left no trace (the `#violdrop` survivor loss was one).
//! * **emptiness was an error return.** An `InfeasibleDomain` child or domain
//!   closed its region with no number attached (the 2026-08-30 alarm's
//!   double-`Infeasible` fold; M2c: "Err-as-close is forbidden").
//! * **reuse was trusted.** The two defects just found on this route — the lA
//!   warm-start seed that is not a bound on a DAG (`#warm-dag`) and the batched
//!   `#2904` objective intersection — both produced child bounds TIGHTER than
//!   the truth from state a child inherited or a core intersected, and both
//!   were invisible to every guard (a sample fence cannot see a 0.06 lift).
//!
//! # What it is
//!
//! Typed key `bab.mo_cover_certify` (`BetaCrownConfig::mo_cover_certify`,
//! governed by `NY_MO_COVER_CERTIFY`), default `off` = byte-identical.
//!
//! 1. **`ledger`** arms a per-ROW path frame (`certify::graph::arm_rows`) for
//!    one disjunctive driver call. Every close the lane makes is recorded with
//!    the split literals of the region it closes:
//!    * the root's rows (`note_root`), or a root-terminal close on its recorded
//!      objective LOWERS (`note_root_lowers`);
//!    * every child a shared-executor wave returns, before the queue fold
//!      (`note_wave`): each row its bound pair proves (the lane's own closure
//!      rule, re-applied by `certify`, not read from the `verified` flag);
//!    * a re-bounded NoUnstable parent (`note_domain_bounds`);
//!    * every emptiness close — a construction-time empty half-space, an
//!      infeasible child, an infeasible NoUnstable domain — ONLY with the
//!      certified SLACK the lane's bounds give it (`note_empty`): `-u` for an
//!      active literal over a certified `u < 0`, `l` for an inactive one over
//!      `l > 0`, `l - u` for an inverted interval. No slack, no evidence: the
//!      frame gives up (UNBUILT), it never accepts an error as a close;
//!    * everything the frame cannot express (the sequential lane, a Graph-MIP
//!      leaf close, a kFSB commit with an omitted side, cuts, conflict
//!      clauses, clipping) leaves it UNBUILT: the verdict keeps the legacy
//!      door, never refused on a ledger limitation.
//!
//!    At the driver's exit a `Verified` is settled (`settle`): every row must
//!    be covered by its certified leaves (the overlap-tolerant graph gate);
//!    a gap WITHDRAWS the verdict (`[cover-gate] REFUSED scope=invocation
//!    lane=graph_mo`); a proof deposits one cover per row, so the instance
//!    door (`certify::authorize_graph_instance`, reached through ny-cli's
//!    `graph_instance_unsat`) mints an `UnsatToken` for a vit property — one
//!    single-atom clause per row — instead of the legacy door.
//!
//! 2. **`replay`** adds CERTIFY-ONCE: every witness region of the proved cover
//!    below the root (the leaves that actually closed a branch, from
//!    `CoverLedger::prove_with_witnesses`) is re-bounded from the ROOT
//!    snapshot with none of the lane's reusable state:
//!    * the region is rebuilt by `with_constraint` from the root, so its
//!      intermediate bounds are the ROOT's certified boxes re-propagated under
//!      the region's literals — never a parent's cached map;
//!    * no lA seed is passed (so the `#warm-dag` class cannot enter), no parent
//!      floor is applied (`tighten_child_bounds_with_parent`), no kFSB receipt,
//!      graph-search hint, dense-spec GPU core or bounded executor is used: the
//!      per-child CPU constrained pass (`propagate_multi_objective_with_beta_and_cache`)
//!      with no engine;
//!    * the lane's final β for that leaf is a HINT only (projected onto
//!      `β >= 0`; any such β is sound by weak duality), and when the hinted
//!      pass does not close, the lane's own analytical β optimizer runs once
//!      from it;
//!    * a region is confirmed only by a replay bound PAIR that passes the
//!      lane's own closure rule for every row it closes (finite, `lower <=
//!      upper`, the LOWER strictly past its threshold — review(vit-v3): the
//!      lower alone accepted a NaN/infinite upper or a crossed interval the
//!      lane refuses), or by an emptiness the replay itself certifies (a
//!      root-bound slack or a slack-bearing `InfeasibleDomain`).
//!
//!    `Verified` stands only when the cover is proved AND every witness below
//!    the root re-certifies; otherwise it is withdrawn. Depth-0 witnesses rest
//!    on the ROOT pipeline (counted `root_anchored`), which is what the
//!    containment fence's root checks (`bab.mo_containment_fence`) and the
//!    optional ORT adjudication fence.
//!
//! # Soundness argument
//!
//! Both modes are DOWNGRADE-ONLY: they can turn `Verified` into `Unknown`, and
//! they deposit covers only for a `Verified` the lane already decided, so they
//! can never create, strengthen or reorder a verdict. A deposited cover adds
//! completeness evidence the legacy door never had: the instance door mints
//! only when every clause is entailed by a proved per-row cover on a root box
//! containing the declared box. The arithmetic of every leaf is the lane's own
//! (as for the single-objective ledger); the replay re-derives each non-root
//! leaf's bound with the same certified evaluator but WITHOUT any state the
//! search carried between domains, which is exactly the class of the two
//! defects this route just shipped.
//!
//! # Observables
//!
//! * `[mo-cover] armed=1 mode=<ledger|replay> source=<...> rows=<K> lane=graph_mo`
//!   once per armed call (`armed=0 ... reason=` when the call cannot arm);
//! * `[cover-gate] graph-frame lane=graph_mo outcome=<proved|unbuilt> ...` /
//!   `[cover-gate] REFUSED scope=invocation lane=graph_mo row=<j> ...`;
//! * `[mo-cover-replay] regions= root_anchored= replayed= confirmed= hinted=
//!   optimized= empty= unconfirmed= deadline_hit= secs= outcome=` (replay);
//! * ny-cli: `[cover-gate] authorized scope=instance lane=graph ...` when the
//!   deposited covers entail the property.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::config::{BetaCrownConfig, MoCoverCertify};
use crate::beta_crown::domain::{
    GraphCrownContext, MultiObjectiveGraphBabDomain, MultiObjectiveTargets,
};
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::beta_crown::state::GraphBetaState;
use crate::certify::graph::{
    GraphRowsScope, GraphRowsSummary, GraphRowsVerdict, ProvedRowsCover, RegionLiteral, RowsRegion,
};
use crate::GraphNetwork;

use super::super::domain_results::MultiObjectiveGraphDomainResult;
use super::super::BetaCrownVerifier;

/// The lane label every multi-objective cover carries (`[cover-gate] … lane=`).
pub(crate) const MO_COVER_LANE: &str = "graph_mo";

/// Most β hints one armed call keeps (one per recorded region; a vit tree is
/// tiny). Past it new hints are dropped: the replay then starts that region
/// from the default β and may have to optimize.
const MO_COVER_BETA_HINT_CAP: usize = 1 << 15;

/// Most `[mo-cover-replay] unconfirmed` detail lines per call.
const MO_COVER_UNCONFIRMED_LINES: usize = 16;

/// The policy resolved for one driver call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MoCoverPolicy {
    /// The mode in force.
    pub(crate) mode: MoCoverCertify,
    /// Where it came from (`config` on the yaml route, `default` when dark).
    pub(crate) source: &'static str,
}

/// Resolve `bab.mo_cover_certify` through its governed declaration
/// (`NY_MO_COVER_CERTIFY` overrides the typed value, as every governed key
/// does). A resolution error keeps today's behaviour (`off`) and says so.
pub(crate) fn resolve_mo_cover_policy(config: &BetaCrownConfig) -> MoCoverPolicy {
    policy_from_resolution(
        config,
        ny_levers::read_over_config(
            &ny_levers::decls::mo_cover::MO_COVER_CERTIFY,
            configured_value(config),
        ),
    )
}

/// The typed value handed to the governed resolver: `None` when `off` (so an
/// absent key and an explicit `off` both resolve from the declaration).
fn configured_value(config: &BetaCrownConfig) -> Option<ny_levers::LeverValue> {
    config
        .mo_cover_certify
        .is_armed()
        .then(|| ny_levers::LeverValue::Text(config.mo_cover_certify.label().to_string()))
}

fn policy_from_resolution(
    config: &BetaCrownConfig,
    resolution: Result<ny_levers::Resolved, ny_levers::ResolveError>,
) -> MoCoverPolicy {
    match resolution {
        Ok(resolved) => {
            let mode = match &resolved.value {
                ny_levers::LeverValue::Text(text) => MoCoverCertify::parse(text),
                _ => None,
            }
            .unwrap_or(MoCoverCertify::Off);
            MoCoverPolicy {
                mode,
                source: resolved.source.as_str(),
            }
        }
        Err(error) => {
            eprintln!(
                "[mo-cover] armed=0 mode={} reason=resolve_error({error:?})",
                config.mo_cover_certify.label()
            );
            MoCoverPolicy {
                mode: MoCoverCertify::Off,
                source: "resolve_error",
            }
        }
    }
}

/// The canonical key of a region: its ReLU literals as sorted
/// `(node, neuron, active)`. `None` when the history carries a GenBaB or
/// normalization split (such a region is not replayable by `with_constraint`).
fn history_key(history: &GraphSplitHistory) -> Option<Vec<(String, usize, bool)>> {
    if !history.genbab_constraints.is_empty() || history.has_norm_inv_rms_constraints() {
        return None;
    }
    let mut key: Vec<(String, usize, bool)> = history
        .constraints
        .iter()
        .map(|c| (c.node_name.clone(), c.neuron_idx, c.is_active))
        .collect();
    key.sort();
    key.dedup();
    Some(key)
}

fn region_key(region: &RowsRegion) -> Option<Vec<(String, usize, bool)>> {
    let mut key = Vec::with_capacity(region.literals.len());
    for literal in &region.literals {
        match literal {
            RegionLiteral::Relu {
                node,
                neuron,
                active,
            } => key.push((node.clone(), *neuron, *active)),
            RegionLiteral::Other { .. } => return None,
        }
    }
    key.sort();
    key.dedup();
    Some(key)
}

/// What the replay starts from: the ROOT domain exactly as the lane queued
/// it (certified root boxes, root α), on the graph the lane evaluated.
pub(crate) struct MoReplayBase {
    graph: GraphNetwork,
    root: MultiObjectiveGraphBabDomain,
}

/// Per-call run state (verifier-scoped, like the certify frame it pairs with).
struct MoCoverRun {
    base: Option<MoReplayBase>,
    beta_hints: HashMap<Vec<(String, usize, bool)>, GraphBetaState>,
    beta_hint_cap_hit: bool,
}

/// The verifier-scoped state of an armed call. Idle (one relaxed atomic load
/// per recording site) unless a driver armed it.
#[derive(Default)]
pub(crate) struct MoCoverState {
    armed: AtomicBool,
    replay: AtomicBool,
    leaf_oracle_closes: AtomicUsize,
    run: Mutex<Option<MoCoverRun>>,
    /// The last settled call's replay report (tests read it; never a verdict).
    #[cfg(test)]
    last_replay: Mutex<Option<MoReplayReport>>,
}

impl MoCoverState {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<MoCoverRun>> {
        self.run
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn begin(&self, replay: bool) {
        *self.lock() = Some(MoCoverRun {
            base: None,
            beta_hints: HashMap::new(),
            beta_hint_cap_hit: false,
        });
        self.leaf_oracle_closes.store(0, Ordering::Relaxed);
        self.replay.store(replay, Ordering::Release);
        self.armed.store(true, Ordering::Release);
    }

    fn end(&self) -> Option<MoCoverRun> {
        self.armed.store(false, Ordering::Release);
        self.replay.store(false, Ordering::Release);
        self.lock().take()
    }

    /// Whether a call armed this state (lets a site skip preparing a record).
    pub(crate) fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    fn replays(&self) -> bool {
        self.replay.load(Ordering::Acquire)
    }
}

/// What a certify-once replay found.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct MoReplayReport {
    /// Witness regions of the proved cover.
    pub(crate) regions: usize,
    /// Depth-0 witnesses (resting on the root pipeline, not replayed).
    pub(crate) root_anchored: usize,
    /// Regions replayed.
    pub(crate) replayed: usize,
    /// Of `replayed`, confirmed.
    pub(crate) confirmed: usize,
    /// Of `confirmed`, confirmed by a hinted single pass.
    pub(crate) hinted: usize,
    /// Of `confirmed`, confirmed only after the analytical β optimizer.
    pub(crate) optimized: usize,
    /// Of `confirmed`, confirmed as an emptiness the replay certified.
    pub(crate) empty: usize,
    /// Unconfirmed regions, with why.
    pub(crate) unconfirmed: Vec<String>,
    /// Whether the replay deadline cut any region.
    pub(crate) deadline_hit: bool,
    /// Wall seconds.
    pub(crate) secs: f64,
}

impl MoReplayReport {
    /// Every witness below the root re-certified.
    pub(crate) fn all_confirmed(&self) -> bool {
        self.unconfirmed.is_empty() && self.confirmed == self.replayed
    }
}

/// One region's replay outcome.
#[derive(Debug, Clone, PartialEq)]
enum RegionOutcome {
    RootAnchored,
    Confirmed { hinted: bool, optimized: bool },
    ConfirmedEmpty,
    Unconfirmed(String),
    Deadline,
}

/// Rows `rows` confirmed by `bounds` (the replay's own objective bounds for
/// exactly those rows, in order): each bound PAIR passes the lane's own
/// lower-bound closure rule (`BetaCrownConfig::domain_is_verified_for_mode`,
/// the rule the ledger re-applies through `graph_bound_evidence`): finite
/// lower AND upper, `lower <= upper`, and the lower strictly past its
/// threshold.
///
/// review(vit-v3): the replay used to read the LOWER alone, so it could
/// confirm a pair the lane itself refuses as a close — a NaN or infinite
/// upper (the #2993 propagation-corruption signal) or a crossed interval. A
/// re-certification must never be weaker than the rule it re-certifies; the
/// stricter reading can only withdraw more.
fn rows_confirmed(bounds: &[(f32, f32)], thresholds: &[f32]) -> Result<(), String> {
    if bounds.len() != thresholds.len() {
        return Err(format!(
            "row_count bounds={} thresholds={}",
            bounds.len(),
            thresholds.len()
        ));
    }
    for (i, (&(lower, upper), &threshold)) in bounds.iter().zip(thresholds).enumerate() {
        if !BetaCrownConfig::domain_is_verified_for_mode(false, lower, upper, threshold) {
            return Err(format!(
                "row_slot={i} replay_lower={lower:e} replay_upper={upper:e} threshold={threshold:e}"
            ));
        }
    }
    Ok(())
}

/// The emptiness slack an `InfeasibleDomain` message carries, when its
/// producer wrote one (`propagation::emptiness_slack` pins the formats).
fn infeasibility_slack(error: &ny_core::NyError) -> Option<f64> {
    if !error.is_infeasible_domain() {
        return None;
    }
    super::propagation::emptiness_slack::slack_from_message(&error.to_string())
}

impl BetaCrownVerifier {
    // ---------------------------------------------------------------------
    // Recording sites (each one relaxed atomic load when dark)
    // ---------------------------------------------------------------------

    /// A domain whose objective bound pairs are its own `objective_bounds`:
    /// every row they prove is recorded for the domain's region. Under
    /// `replay` the domain's β is kept as that region's hint.
    pub(crate) fn mo_cover_note_domain(&self, domain: &MultiObjectiveGraphBabDomain) {
        if !self.mo_cover.is_armed() {
            return;
        }
        self.graph_cover
            .note_rows_bounds(&domain.history, &domain.objective_bounds);
        if self.mo_cover.replays() {
            if let Some(key) = history_key(&domain.history) {
                let mut guard = self.mo_cover.lock();
                if let Some(run) = guard.as_mut() {
                    if run.beta_hints.len() < MO_COVER_BETA_HINT_CAP
                        || run.beta_hints.contains_key(&key)
                    {
                        run.beta_hints.insert(key, domain.beta_state.clone());
                    } else {
                        run.beta_hint_cap_hit = true;
                    }
                }
            }
        }
    }

    /// A domain (history `history`) re-bounded to `bounds` by the processor
    /// (the NoUnstable final pass): every row they prove is recorded.
    pub(crate) fn mo_cover_note_bounds(&self, history: &GraphSplitHistory, bounds: &[(f32, f32)]) {
        if self.mo_cover.is_armed() {
            self.graph_cover.note_rows_bounds(history, bounds);
        }
    }

    /// Every child a shared-executor wave returns, before the queue fold.
    pub(in crate::beta_crown::engine) fn mo_cover_note_wave(
        &self,
        results: &[MultiObjectiveGraphDomainResult],
    ) {
        if !self.mo_cover.is_armed() {
            return;
        }
        for result in results {
            match result {
                MultiObjectiveGraphDomainResult::Children(children)
                | MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(children) => {
                    for (child, _) in children {
                        self.mo_cover_note_domain(child);
                    }
                }
                MultiObjectiveGraphDomainResult::AlreadyVerified
                | MultiObjectiveGraphDomainResult::Violation
                | MultiObjectiveGraphDomainResult::NoUnstable { .. }
                | MultiObjectiveGraphDomainResult::PropagationFailure
                | MultiObjectiveGraphDomainResult::DeadlineExpired
                | MultiObjectiveGraphDomainResult::InfeasibleEmpty
                | MultiObjectiveGraphDomainResult::InfeasibleCoverRefused => {}
            }
        }
    }

    /// A region the lane's certified bounds show EMPTY: the domain `history`,
    /// narrowed by `extra` for a half-space the lane never built, with the
    /// certified `slack` (`None` = the frame gives up).
    pub(crate) fn mo_cover_note_empty(
        &self,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
        slack: Option<f64>,
    ) {
        if self.mo_cover.is_armed() {
            self.graph_cover.note_rows_empty(history, extra, slack);
        }
    }

    /// An emptiness the lane reported as an `InfeasibleDomain` error on the
    /// domain `history` (its slack is read from the producer's message).
    pub(crate) fn mo_cover_note_infeasible(
        &self,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
        error_text: &str,
    ) {
        if self.mo_cover.is_armed() {
            let slack = super::propagation::emptiness_slack::slack_from_message(error_text);
            self.graph_cover.note_rows_empty(history, extra, slack);
        }
    }

    /// A close the frame cannot express: UNBUILT (legacy door; never refused).
    pub(crate) fn mo_cover_unbuildable(&self, reason: &'static str) {
        if self.mo_cover.is_armed() {
            self.graph_cover.note_unbuildable(reason);
        }
    }

    /// A Graph-MIP leaf-oracle close (counted by the queue fold's lifecycle;
    /// the frame gives up on it after the fold).
    pub(crate) fn mo_cover_note_leaf_oracle_closes(&self, closes: usize) {
        if closes > 0 && self.mo_cover.is_armed() {
            self.mo_cover
                .leaf_oracle_closes
                .fetch_add(closes, Ordering::Relaxed);
            self.graph_cover.note_unbuildable("graph_mip_leaf_close");
        }
    }

    /// The root as the lane queues it: its rows are recorded at the root
    /// region, and under `replay` the root and the graph are kept as the
    /// replay base.
    pub(crate) fn mo_cover_note_root(
        &self,
        graph: &GraphNetwork,
        root: &MultiObjectiveGraphBabDomain,
    ) {
        if !self.mo_cover.is_armed() {
            return;
        }
        self.graph_cover
            .note_rows_bounds(&root.history, &root.objective_bounds);
        if self.mo_cover.replays() {
            if let Some(run) = self.mo_cover.lock().as_mut() {
                run.base = Some(MoReplayBase {
                    graph: graph.clone(),
                    root: root.clone(),
                });
            }
        }
    }

    /// A root-terminal `Verified`: the objective LOWERS the closing root site
    /// recorded (`None` = not recorded: the frame gives up).
    pub(crate) fn mo_cover_note_root_lowers(&self, lowers: Option<&[f32]>) {
        if !self.mo_cover.is_armed() {
            return;
        }
        match lowers {
            Some(lowers) => self
                .graph_cover
                .note_rows_lowers(&GraphSplitHistory::new(), lowers),
            None => self.graph_cover.note_unbuildable("root_lowers_unrecorded"),
        }
    }

    // ---------------------------------------------------------------------
    // Arming and settlement
    // ---------------------------------------------------------------------

    /// Arm the per-row frame for one DISJUNCTIVE driver call. `None` (with an
    /// `armed=0` line) when the call cannot arm: conjunctive semantics (a
    /// region closes when ANY row does — a different aggregation), or a nested
    /// driver call on a verifier that already carries a frame.
    pub(crate) fn mo_cover_arm<'a>(
        &'a self,
        policy: MoCoverPolicy,
        graph: &GraphNetwork,
        input: &crate::BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        conjunctive: bool,
    ) -> Option<GraphRowsScope<'a>> {
        if !policy.mode.is_armed() {
            return None;
        }
        if conjunctive {
            eprintln!(
                "[mo-cover] armed=0 mode={} source={} reason=conjunctive",
                policy.mode.label(),
                policy.source
            );
            return None;
        }
        if self.mo_cover.is_armed() {
            eprintln!(
                "[mo-cover] armed=0 mode={} source={} reason=nested_driver",
                policy.mode.label(),
                policy.source
            );
            // The outer frame cannot tell this call's closes from its own.
            self.graph_cover.note_unbuildable("nested_driver");
            return None;
        }
        let scope = self.graph_cover.arm_rows(
            MO_COVER_LANE,
            graph.node_order.iter().map(String::as_str),
            input,
            objectives,
            thresholds,
        );
        if !scope.owns_frame() {
            eprintln!(
                "[mo-cover] armed=0 mode={} source={} reason=nested_frame",
                policy.mode.label(),
                policy.source
            );
            return None;
        }
        self.mo_cover.begin(policy.mode.replays());
        eprintln!(
            "[mo-cover] armed=1 mode={} source={} rows={} lane={MO_COVER_LANE}",
            policy.mode.label(),
            policy.source,
            objectives.len()
        );
        Some(scope)
    }

    /// End an armed call that returned an error (no verdict to settle).
    pub(crate) fn mo_cover_abort(&self, scope: GraphRowsScope<'_>) {
        drop(self.mo_cover.end());
        drop(scope);
    }

    /// Settle an armed call's frame against its result (see the module docs).
    /// The cover check never runs past `deadline` (else the verifier's own)
    /// nor past the ledger's wall cap; the replay never runs past the verifier's own
    /// deadline (else `deadline`).
    pub(crate) fn mo_cover_settle(
        &self,
        policy: MoCoverPolicy,
        scope: GraphRowsScope<'_>,
        mut result: BetaCrownResult,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        deadline: Option<Instant>,
    ) -> BetaCrownResult {
        let run = self.mo_cover.end();
        let verified = matches!(result.result, BabVerificationStatus::Verified);
        let deadline = deadline.or(self.config.alpha_config.deadline);
        let withdraw = |result: &mut BetaCrownResult, reason: String| {
            result.result = BabVerificationStatus::Unknown { reason };
        };
        match scope.finish(verified, deadline) {
            GraphRowsVerdict::NotArmed => {}
            GraphRowsVerdict::Unbuilt(reason) => {
                eprintln!(
                    "[cover-gate] graph-frame lane={MO_COVER_LANE} outcome=unbuilt reason={reason} \
                     mode={}{}",
                    policy.mode.label(),
                    if policy.mode.replays() {
                        " verdict=withdrawn (replay requires a proved cover)"
                    } else {
                        ""
                    }
                );
                if policy.mode.replays() {
                    withdraw(
                        &mut result,
                        format!(
                            "mo-cover replay: the per-row cover could not be built ({reason}); \
                             nothing to certify once"
                        ),
                    );
                }
            }
            GraphRowsVerdict::Refused { row, refusal } => {
                eprintln!(
                    "[cover-gate] REFUSED scope=invocation lane={MO_COVER_LANE} row={row} \
                     reason={refusal} (the lane's certified leaves do not cover its root: \
                     Verified withdrawn)"
                );
                withdraw(
                    &mut result,
                    format!("graph cover gate refused ({MO_COVER_LANE}) row {row}: {refusal}"),
                );
            }
            GraphRowsVerdict::Proved(proved) => {
                let summary = proved.summary();
                if policy.mode.replays() {
                    // The replay runs after the search DECIDED: it may use the
                    // verifier's own deadline (the instance's internal budget),
                    // not only the caller's BaB slice, whose post-BaB phases a
                    // `Verified` does not need.
                    let replay_deadline = self.config.alpha_config.deadline.or(deadline);
                    let report = self.mo_cover_replay(
                        run.as_ref().and_then(|run| run.base.as_ref()),
                        run.as_ref().map(|run| &run.beta_hints),
                        &proved,
                        objectives,
                        thresholds,
                        replay_deadline,
                    );
                    emit_replay_line(&report, run.as_ref().is_some_and(|r| r.beta_hint_cap_hit));
                    #[cfg(test)]
                    {
                        *self
                            .mo_cover
                            .last_replay
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            Some(report.clone());
                    }
                    if report.all_confirmed() {
                        let summary = proved.deposit();
                        emit_proved_line(&summary, "replay_confirmed");
                    } else {
                        eprintln!(
                            "[cover-gate] graph-frame lane={MO_COVER_LANE} outcome=proved \
                             replay=unconfirmed verdict=withdrawn regions={} unconfirmed={}",
                            summary.regions,
                            report.unconfirmed.len().max(1)
                        );
                        withdraw(
                            &mut result,
                            format!(
                                "mo-cover replay: {} of {} witness region(s) below the root did \
                                 not re-certify without reused lane state (first: {})",
                                report.replayed.saturating_sub(report.confirmed),
                                report.replayed,
                                report
                                    .unconfirmed
                                    .first()
                                    .map_or("none recorded", String::as_str)
                            ),
                        );
                    }
                } else {
                    let summary = proved.deposit();
                    emit_proved_line(&summary, "ledger");
                }
            }
        }
        result
    }

    // ---------------------------------------------------------------------
    // Certify-once replay
    // ---------------------------------------------------------------------

    fn mo_cover_replay(
        &self,
        base: Option<&MoReplayBase>,
        hints: Option<&HashMap<Vec<(String, usize, bool)>, GraphBetaState>>,
        proved: &ProvedRowsCover,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        deadline: Option<Instant>,
    ) -> MoReplayReport {
        let started = Instant::now();
        let regions = proved.regions();
        let mut report = MoReplayReport {
            regions: regions.len(),
            ..MoReplayReport::default()
        };
        let needs_base = regions.iter().any(|region| !region.literals.is_empty());
        let Some(base) = base.filter(|_| needs_base) else {
            if needs_base {
                report.replayed = regions.iter().filter(|r| !r.literals.is_empty()).count();
                report
                    .unconfirmed
                    .push("no replay base (the root never reached the queue)".to_string());
            }
            report.root_anchored = regions.iter().filter(|r| r.literals.is_empty()).count();
            report.secs = started.elapsed().as_secs_f64();
            return report;
        };
        // The replay polls the verifier's effective deadline; bound it by the
        // call's own boundary for the whole replay.
        let _scope = self.complete_clip_deadline_overrides.scoped(deadline);
        let outcomes: Vec<RegionOutcome> = regions
            .par_iter()
            .map(|region| {
                self.mo_cover_replay_region(base, hints, region, objectives, thresholds, deadline)
            })
            .collect();
        for (index, (region, outcome)) in regions.iter().zip(outcomes).enumerate() {
            match outcome {
                RegionOutcome::RootAnchored => report.root_anchored += 1,
                RegionOutcome::Confirmed { hinted, optimized } => {
                    report.replayed += 1;
                    report.confirmed += 1;
                    report.hinted += usize::from(hinted);
                    report.optimized += usize::from(optimized);
                }
                RegionOutcome::ConfirmedEmpty => {
                    report.replayed += 1;
                    report.confirmed += 1;
                    report.empty += 1;
                }
                RegionOutcome::Unconfirmed(why) => {
                    report.replayed += 1;
                    report.unconfirmed.push(format!(
                        "region={index} depth={} rows={:?} {why}",
                        region.literals.len(),
                        region.rows
                    ));
                }
                RegionOutcome::Deadline => {
                    report.replayed += 1;
                    report.deadline_hit = true;
                    report.unconfirmed.push(format!(
                        "region={index} depth={} rows={:?} deadline",
                        region.literals.len(),
                        region.rows
                    ));
                }
            }
        }
        report.secs = started.elapsed().as_secs_f64();
        report
    }

    fn mo_cover_replay_region(
        &self,
        base: &MoReplayBase,
        hints: Option<&HashMap<Vec<(String, usize, bool)>, GraphBetaState>>,
        region: &RowsRegion,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        deadline: Option<Instant>,
    ) -> RegionOutcome {
        if region.literals.is_empty() {
            return RegionOutcome::RootAnchored;
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return RegionOutcome::Deadline;
        }
        let Some(key) = region_key(region) else {
            return RegionOutcome::Unconfirmed("genbab_or_foreign_literal".to_string());
        };
        if region.rows.is_empty() || region.rows.iter().any(|&row| row >= objectives.len()) {
            return RegionOutcome::Unconfirmed("row_index".to_string());
        }
        let rows: Vec<Vec<f32>> = region.rows.iter().map(|&r| objectives[r].clone()).collect();
        let row_thresholds: Vec<f32> = region.rows.iter().map(|&r| thresholds[r]).collect();
        let prepared = match self.mo_cover_replay_prepare(base, &key, thresholds) {
            Ok(prepared) => prepared,
            Err(outcome) => return outcome,
        };
        let hint = hints.and_then(|hints| hints.get(&key));
        let hinted = hint.is_some();
        let mut beta = replay_beta(&prepared, hint);
        let first_miss =
            match self.mo_cover_replay_pass(base, &prepared, &beta, &rows, &row_thresholds) {
                Ok(bounds) => match rows_confirmed(&bounds, &row_thresholds) {
                    Ok(()) => {
                        return RegionOutcome::Confirmed {
                            hinted,
                            optimized: false,
                        }
                    }
                    Err(why) => why,
                },
                Err(error) if error.is_deadline_exceeded() => return RegionOutcome::Deadline,
                Err(error) => {
                    if infeasibility_slack(&error).is_some_and(|s| s.is_finite() && s > 0.0) {
                        return RegionOutcome::ConfirmedEmpty;
                    }
                    format!("single_pass {error}")
                }
            };
        // The lane's own analytical β optimizer, once, from the hint.
        if self.config.beta_iterations == 0 {
            return RegionOutcome::Unconfirmed(format!("{first_miss} (no beta optimizer)"));
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return RegionOutcome::Deadline;
        }
        let open = vec![false; rows.len()];
        let targets = MultiObjectiveTargets::new(&rows, &row_thresholds, &open);
        let seeds = vec![None; rows.len()];
        let context = prepared.context();
        match self.optimize_graph_beta_analytical_multi_objective_with_cache(
            &base.graph,
            prepared.domain.input_bounds.as_ref(),
            &context,
            &mut beta,
            &targets,
            false,
            true,
            &seeds,
            false,
        ) {
            Ok((bounds, _, _)) => match rows_confirmed(&bounds, &row_thresholds) {
                Ok(()) => RegionOutcome::Confirmed {
                    hinted,
                    optimized: true,
                },
                Err(why) => RegionOutcome::Unconfirmed(format!("{first_miss}; optimized {why}")),
            },
            Err(error) if error.is_deadline_exceeded() => RegionOutcome::Deadline,
            Err(error) => {
                if infeasibility_slack(&error).is_some_and(|s| s.is_finite() && s > 0.0) {
                    RegionOutcome::ConfirmedEmpty
                } else {
                    RegionOutcome::Unconfirmed(format!("{first_miss}; optimized {error}"))
                }
            }
        }
    }

    /// Rebuild the region `key` from the ROOT, literal by literal: each
    /// literal's feasibility is judged against the ROOT's certified boxes, and
    /// the region's intermediate bounds are the root's (re-propagated under
    /// the literals by the pass). `Err` carries the region's outcome when the
    /// rebuild alone decides it (an empty region the root boxes certify, or a
    /// construction the lane itself would refuse).
    fn mo_cover_replay_prepare(
        &self,
        base: &MoReplayBase,
        key: &[(String, usize, bool)],
        thresholds: &[f32],
    ) -> Result<ReplayRegion, RegionOutcome> {
        let mut domain: Option<MultiObjectiveGraphBabDomain> = None;
        for (node, neuron, active) in key {
            let constraint = GraphNeuronConstraint {
                node_name: node.clone(),
                neuron_idx: *neuron,
                is_active: *active,
                score: 0.0,
            };
            let parent = domain.as_ref().unwrap_or(&base.root);
            if parent
                .split_emptiness_slack(&base.graph, &constraint)
                .is_some_and(|slack| slack.is_finite() && slack > 0.0)
            {
                // The ROOT boxes (valid on every sub-region) already put this
                // literal's half-space outside the region: it is empty.
                return Err(RegionOutcome::ConfirmedEmpty);
            }
            // `thresholds` are the call's (one per root row): `with_constraint`
            // only recomputes the queue priority from them.
            match parent.with_constraint(&base.graph, constraint, false, thresholds) {
                Ok(Some(child)) => domain = Some(child),
                Ok(None) => {
                    return Err(RegionOutcome::Unconfirmed(format!(
                        "construct_empty_without_slack node={node} neuron={neuron}"
                    )))
                }
                Err(error) => return Err(RegionOutcome::Unconfirmed(format!("construct {error}"))),
            }
        }
        domain
            .map(|domain| ReplayRegion { domain })
            .ok_or(RegionOutcome::RootAnchored)
    }

    /// One certified pass over `prepared` for `rows` (the replay's own
    /// objective bounds, in order), with `beta` and no seed, no engine.
    fn mo_cover_replay_pass(
        &self,
        base: &MoReplayBase,
        prepared: &ReplayRegion,
        beta: &GraphBetaState,
        rows: &[Vec<f32>],
        row_thresholds: &[f32],
    ) -> ny_core::Result<Vec<(f32, f32)>> {
        let open = vec![false; rows.len()];
        let targets = MultiObjectiveTargets::new(rows, row_thresholds, &open);
        let seeds = vec![None; rows.len()];
        let context = prepared.context();
        self.propagate_multi_objective_with_beta_and_cache(
            &base.graph,
            prepared.domain.input_bounds.as_ref(),
            &context,
            beta,
            &targets,
            &seeds,
            false,
        )
        .map(|(bounds, _, _)| bounds)
    }
}

/// A region rebuilt from the replay base.
struct ReplayRegion {
    domain: MultiObjectiveGraphBabDomain,
}

impl ReplayRegion {
    /// The pass context: the region's split history, NO cut pool, the ROOT's
    /// node bounds (the rebuilt domain inherits them verbatim), NO engine,
    /// the root-born α.
    fn context(&self) -> GraphCrownContext<'_> {
        GraphCrownContext::new_with_node_bounds_map(
            &self.domain.history,
            None,
            Some(&self.domain.node_bounds),
            None,
        )
        .with_alpha(&self.domain.alpha_state)
    }
}

/// The replay's β: the lane's final β for exactly this region (matched per
/// constraint), else the root-born default; every entry projected onto
/// `β >= 0` (any such β is sound by weak duality).
fn replay_beta(prepared: &ReplayRegion, hint: Option<&GraphBetaState>) -> GraphBetaState {
    let mut beta = hint
        .and_then(|hint| {
            GraphBetaState::from_history_with_warmup(
                &prepared.domain.history,
                hint,
                GraphBetaState::DEFAULT_BETA_INIT,
            )
            .ok()
        })
        .unwrap_or_else(|| prepared.domain.beta_state.clone());
    for entry in &mut beta.entries {
        let value = entry.value;
        entry.set_value(value);
    }
    beta
}

fn emit_proved_line(summary: &GraphRowsSummary, via: &str) {
    eprintln!(
        "[cover-gate] graph-frame lane={MO_COVER_LANE} outcome=proved via={via} rows={} \
         leaves={} min_row_leaves={} empty_leaves={} max_depth={} min_margin={:e} records={} \
         empty_records={} regions={}",
        summary.rows,
        summary.leaves,
        summary.min_row_leaves,
        summary.empty_leaves,
        summary.max_depth,
        summary.min_margin,
        summary.records,
        summary.empty_records,
        summary.regions,
    );
}

fn emit_replay_line(report: &MoReplayReport, hint_cap_hit: bool) {
    eprintln!(
        "[mo-cover-replay] regions={} root_anchored={} replayed={} confirmed={} hinted={} \
         optimized={} empty={} unconfirmed={} deadline_hit={} hint_cap_hit={} secs={:.3} \
         outcome={}",
        report.regions,
        report.root_anchored,
        report.replayed,
        report.confirmed,
        report.hinted,
        report.optimized,
        report.empty,
        report.unconfirmed.len(),
        u8::from(report.deadline_hit),
        u8::from(hint_cap_hit),
        report.secs,
        if report.all_confirmed() {
            "confirmed"
        } else {
            "withdrawn"
        }
    );
    for line in report.unconfirmed.iter().take(MO_COVER_UNCONFIRMED_LINES) {
        eprintln!("[mo-cover-replay] unconfirmed {line}");
    }
}

#[cfg(test)]
#[path = "mo_cover_tests.rs"]
mod tests;
