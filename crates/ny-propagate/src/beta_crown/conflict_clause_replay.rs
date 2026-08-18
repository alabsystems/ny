// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Replay-verified graph-clause generalization contract.
//!
//! A verified leaf with literal set `L` may only be generalized to a strict
//! subset `C ⊂ L` after a sound verifier has replayed the *larger* region
//! described by `C` and certified that whole region.  Merely deleting a
//! literal from a verified leaf is unsound.
//!
//! * the deterministic planner only proposes one-literal deletions from a
//!   well-formed, pure ReLU-at-zero history;
//! * every proposal is bound to collision-free graph scope plus bit-exact
//!   objective, root, source-history, and candidate-history identities;
//! * the private trusted replay boundary runs deadline-threaded constrained
//!   CROWN over the exact candidate region before it can mint
//!   [`ReplayVerifiedGraphClause`];
//! * `GraphClauseStore` accepts that non-cloneable token by value through its
//!   new insertion seam and validates it against the store's immutable run
//!   binding. A raw proposal or caller-supplied run identity cannot be
//!   inserted through that API.
//!
//! Runtime integration is doubly dark: ordinary graph conflict-clause learning
//! must already be enabled and `NY_BAB_CLAUSE_REPLAY=1` must be set exactly.
//! The replay lane has no verdict shortcut: it can only add a replay-certified
//! clause to the existing per-run store. The current production caller is
//! lower-bound-only; upper-bound verification refuses all graph-clause
//! authority before either ordinary recording or replay is constructed.
//!
//! BICCOS-Q Stage 1 is a third, subordinate exact gate:
//! `NY_BICCOS_Q_STAGE1_REPLAY=1`. It is read only after the parent replay gate
//! has constructed this runtime. Stage 1 may change proposal order, never proof
//! authority: β/gradient values rank an opaque strict-subset history in
//! `biccos_q_stage0`, then disappear. This module replays that exact candidate,
//! captures the same run/source/candidate identities, seals the same
//! non-cloneable token, and inserts only through
//! `GraphClauseStore::insert_replay_verified`.
//!
//! The NeuralSAT-inspired BCP observer is a separate fourth, subordinate gate:
//! `NY_BICCOS_BCP_SHADOW=1`. It follows the learned-clause polarity in
//! NeuralSAT `4fb45f8:src/heuristic/util.py`, but stops after returning one
//! canonical exact implied literal. Unlike NeuralSAT's
//! `update_hidden_bounds_histories`, this shadow never writes a history or
//! bound. Both ordinary verified-close and replay-generalized sources must
//! match this runtime's exact graph/root/objective/property fingerprint, and
//! the reported provenance keeps those two authority paths distinct.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use super::branching::{GraphNeuronConstraint, GraphSplitHistory};
use super::domain::{GraphCrownContext, NodeBoundsView};
use super::engine::BetaCrownVerifier;
use super::state::GraphBetaState;
use crate::beta_crown::bab_cuts::CutFoldScope;
use crate::GraphNetwork;

const RUNTIME_ATTEMPT_CAP: usize = 16;
const RUNTIME_TOTAL_BUDGET: Duration = Duration::from_secs(2);
const RUNTIME_ATTEMPT_BUDGET: Duration = Duration::from_millis(250);
const RUNTIME_ENCLOSING_RESERVE: Duration = Duration::from_secs(5);
const RUNTIME_LITERAL_COUNT_CAP: usize = 4_096;
const RUNTIME_IDENTITY_BYTE_CAP: usize = 8 * 1024 * 1024;
const RUNTIME_VERIFY_UPPER_BOUND: bool = false;
const BICCOS_Q_STAGE1_REPLAY_ENV: &str = "NY_BICCOS_Q_STAGE1_REPLAY";
const BICCOS_BCP_SHADOW_ENV: &str = "NY_BICCOS_BCP_SHADOW";

fn runtime_gate_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

fn runtime_gate_enabled_from_env() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_RUNTIME_GATE_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    runtime_gate_enabled(std::env::var("NY_BAB_CLAUSE_REPLAY").ok().as_deref())
}

fn biccos_q_stage1_gate_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Read only after the parent replay gate has admitted runtime construction.
fn biccos_q_stage1_gate_enabled_from_env() -> bool {
    #[cfg(test)]
    {
        update_test_runtime_observations(|observations| {
            observations.stage1_gate_reads = observations.stage1_gate_reads.saturating_add(1);
        });
        if let Some(enabled) = TEST_BICCOS_Q_STAGE1_GATE_OVERRIDE.with(std::cell::Cell::get) {
            return enabled;
        }
    }
    biccos_q_stage1_gate_enabled(std::env::var(BICCOS_Q_STAGE1_REPLAY_ENV).ok().as_deref())
}

fn biccos_bcp_shadow_gate_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Read only after the parent replay gate has admitted runtime construction.
fn biccos_bcp_shadow_gate_enabled_from_env() -> bool {
    #[cfg(test)]
    {
        update_test_runtime_observations(|observations| {
            observations.bcp_shadow_gate_reads =
                observations.bcp_shadow_gate_reads.saturating_add(1);
        });
        if let Some(enabled) = TEST_BICCOS_BCP_SHADOW_GATE_OVERRIDE.with(std::cell::Cell::get) {
            return enabled;
        }
    }
    biccos_bcp_shadow_gate_enabled(std::env::var(BICCOS_BCP_SHADOW_ENV).ok().as_deref())
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GraphClauseReplayTestObservations {
    pub(crate) from_env_calls: usize,
    pub(crate) source_offers: usize,
    pub(crate) proof_attempts: usize,
    pub(crate) stage1_gate_reads: usize,
    pub(crate) bcp_shadow_gate_reads: usize,
    pub(crate) stage1_source_offers: usize,
    pub(crate) stage1_provenance_refusals: usize,
    pub(crate) stage1_proposals: usize,
    pub(crate) stage1_proof_attempts: usize,
    pub(crate) stage1_accepts: usize,
}

#[cfg(test)]
thread_local! {
    static TEST_RUNTIME_GATE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static TEST_BICCOS_Q_STAGE1_GATE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static TEST_BICCOS_BCP_SHADOW_GATE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static TEST_RUNTIME_OBSERVATIONS:
        std::cell::Cell<GraphClauseReplayTestObservations> =
        const { std::cell::Cell::new(GraphClauseReplayTestObservations {
            from_env_calls: 0,
            source_offers: 0,
            proof_attempts: 0,
            stage1_gate_reads: 0,
            bcp_shadow_gate_reads: 0,
            stage1_source_offers: 0,
            stage1_provenance_refusals: 0,
            stage1_proposals: 0,
            stage1_proof_attempts: 0,
            stage1_accepts: 0,
        }) };
}

#[cfg(test)]
pub(crate) fn set_test_runtime_gate_override(enabled: Option<bool>) {
    TEST_RUNTIME_GATE_OVERRIDE.with(|value| value.set(enabled));
}

#[cfg(test)]
pub(crate) fn set_test_biccos_q_stage1_gate_override(enabled: Option<bool>) {
    TEST_BICCOS_Q_STAGE1_GATE_OVERRIDE.with(|value| value.set(enabled));
}

#[cfg(test)]
fn set_test_biccos_bcp_shadow_gate_override(enabled: Option<bool>) {
    TEST_BICCOS_BCP_SHADOW_GATE_OVERRIDE.with(|value| value.set(enabled));
}

#[cfg(test)]
pub(crate) fn reset_test_runtime_observations() {
    TEST_RUNTIME_OBSERVATIONS.with(|observations| {
        observations.set(GraphClauseReplayTestObservations::default());
    });
}

#[cfg(test)]
pub(crate) fn test_runtime_observations() -> GraphClauseReplayTestObservations {
    TEST_RUNTIME_OBSERVATIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn update_test_runtime_observations(update: impl FnOnce(&mut GraphClauseReplayTestObservations)) {
    TEST_RUNTIME_OBSERVATIONS.with(|observations| {
        let mut current = observations.get();
        update(&mut current);
        observations.set(current);
    });
}

/// Resource and wall-clock limits for one deterministic planning session.
///
/// `attempt_cap` bounds emitted replay proposals. `literal_count_cap` bounds
/// both validation work and the canonical constraint copy. `identity_byte_cap`
/// bounds the combined exact objective/root/source/candidate identities held
/// by any proposal.
#[derive(Debug, Clone, Copy)]
pub(super) struct GraphClauseReplayLimits {
    pub(super) attempt_cap: usize,
    pub(super) literal_count_cap: usize,
    pub(super) identity_byte_cap: usize,
    pub(super) deadline: Instant,
}

/// Fail-closed reasons returned before or during deterministic planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum GraphClauseReplayRefusal {
    AttemptCapZero,
    LiteralCountCapTooSmall,
    IdentityByteCapZero,
    DeadlineExpired,
    ImpureHistory,
    TooFewLiterals,
    LiteralCountExceeded {
        count: usize,
        cap: usize,
    },
    DuplicateLiteral {
        node_name: String,
        neuron_idx: usize,
    },
    OppositePhases {
        node_name: String,
        neuron_idx: usize,
    },
    EmptyObjectiveIdentity,
    EmptyRootIdentity,
    InvalidRunSemantics,
    IdentityBudgetExceeded,
    HistoryIdentityUnavailable,
    CandidateNotStrictSubset,
}

/// Trusted production origin attached to a Stage-1 source offer.
///
/// Only an ordinary `Children` result from the shared multi-objective executor
/// is allowlisted. The legacy violated-sibling-drop variant remains a valid
/// source for established one-deletion replay, but Stage 1 conservatively
/// refuses it because its enclosing result carries abandoned-region state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BiccosQStage1SourceProvenance {
    SharedMultiObjectiveChildren,
    SharedMultiObjectiveChildrenWithViolatedSiblingDrop,
}

impl BiccosQStage1SourceProvenance {
    fn is_allowlisted(self) -> bool {
        matches!(self, Self::SharedMultiObjectiveChildren)
    }
}

/// Collision-free identity of the immutable graph, objective semantics, and
/// root box for one graph-verification run.
///
/// "Fingerprint" here means exact owned bytes, not a fixed-width hash.  The
/// future caller must carry the exact graph scope and canonically encode every
/// objective row, threshold, fixed lower-bound sense/mode, and every root-box
/// dimension/endpoint bit. Since the bytes are copied into `Arc<[u8]>`, later
/// caller mutation cannot retarget a proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GraphClauseReplayRunFingerprint {
    graph_scope: CutFoldScope,
    objective_identity: Arc<[u8]>,
    root_identity: Arc<[u8]>,
}

impl GraphClauseReplayRunFingerprint {
    pub(super) fn from_exact_identities(
        graph_scope: CutFoldScope,
        objective_identity: &[u8],
        root_identity: &[u8],
        identity_byte_cap: usize,
    ) -> Result<Self, GraphClauseReplayRefusal> {
        if identity_byte_cap == 0 {
            return Err(GraphClauseReplayRefusal::IdentityByteCapZero);
        }
        if objective_identity.is_empty() {
            return Err(GraphClauseReplayRefusal::EmptyObjectiveIdentity);
        }
        if root_identity.is_empty() {
            return Err(GraphClauseReplayRefusal::EmptyRootIdentity);
        }
        let total = objective_identity
            .len()
            .checked_add(root_identity.len())
            .ok_or(GraphClauseReplayRefusal::IdentityBudgetExceeded)?;
        if total > identity_byte_cap {
            return Err(GraphClauseReplayRefusal::IdentityBudgetExceeded);
        }
        Ok(Self {
            graph_scope,
            objective_identity: Arc::from(objective_identity),
            root_identity: Arc::from(root_identity),
        })
    }

    fn byte_len(&self) -> usize {
        // Construction already checked this sum.
        self.objective_identity.len() + self.root_identity.len()
    }
}

/// Exact identity observed at the trusted replay boundary.
///
/// Its constructor derives both history identities from the histories the
/// executor says it actually replayed.  The private sealing function compares
/// every component separately, so stale work cannot acquire authority.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphClauseReplayFingerprint {
    run: GraphClauseReplayRunFingerprint,
    source_history_identity: Arc<[u8]>,
    candidate_history_identity: Arc<[u8]>,
}

impl GraphClauseReplayFingerprint {
    fn capture(
        run: &GraphClauseReplayRunFingerprint,
        source_history: &GraphSplitHistory,
        candidate_history: &GraphSplitHistory,
        identity_byte_cap: usize,
    ) -> Result<Self, GraphClauseReplayRefusal> {
        let source = source_history
            .exact_provenance_identity()
            .ok_or(GraphClauseReplayRefusal::HistoryIdentityUnavailable)?;
        let candidate = candidate_history
            .exact_provenance_identity()
            .ok_or(GraphClauseReplayRefusal::HistoryIdentityUnavailable)?;
        let total = run
            .byte_len()
            .checked_add(source.len())
            .and_then(|n| n.checked_add(candidate.len()))
            .ok_or(GraphClauseReplayRefusal::IdentityBudgetExceeded)?;
        if total > identity_byte_cap {
            return Err(GraphClauseReplayRefusal::IdentityBudgetExceeded);
        }
        Ok(Self {
            run: run.clone(),
            source_history_identity: Arc::from(source),
            candidate_history_identity: Arc::from(candidate),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GraphClauseReplayLiteral {
    pub(super) node_name: String,
    pub(super) neuron_idx: usize,
    pub(super) is_active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphClauseReplayProposalKind {
    DeterministicOneDeletion,
    BiccosQStage1Ranked,
}

/// One deterministic strict-subset replay request.
///
/// This type is deliberately distinct from `ReplayVerifiedGraphClause`.
/// Owning a proposal conveys no store-insertion authority.
#[derive(Debug)]
pub(super) struct GraphClauseReplayProposal {
    attempt_ordinal: usize,
    kind: GraphClauseReplayProposalKind,
    removed_literal: GraphClauseReplayLiteral,
    removed_literal_count: usize,
    source_literal_count: usize,
    candidate_history: GraphSplitHistory,
    binding: GraphClauseReplayFingerprint,
    identity_byte_cap: usize,
    deadline: Instant,
}

impl GraphClauseReplayProposal {
    pub(super) fn attempt_ordinal(&self) -> usize {
        self.attempt_ordinal
    }

    pub(super) fn removed_literal(&self) -> &GraphClauseReplayLiteral {
        &self.removed_literal
    }

    fn removed_literal_count(&self) -> usize {
        self.removed_literal_count
    }

    fn kind(&self) -> GraphClauseReplayProposalKind {
        self.kind
    }

    pub(super) fn candidate_history(&self) -> &GraphSplitHistory {
        &self.candidate_history
    }

    /// Private trusted replay boundary.
    ///
    /// The future executor may call this only after its sound, complete-for-
    /// this-attempt proof path has certified the entire candidate region.  It
    /// must capture `observed` from the objective/root/source/candidate values
    /// it actually used.  Keeping this function private forces that executor
    /// to live in this module instead of exposing a raw "verified=true" mint.
    fn seal_replay_verified(
        self,
        observed: GraphClauseReplayFingerprint,
        completed_at: Instant,
    ) -> Result<ReplayVerifiedGraphClause, GraphClauseReplaySealRefusal> {
        if completed_at >= self.deadline {
            return Err(GraphClauseReplaySealRefusal::DeadlineExpired);
        }
        if observed.run.graph_scope != self.binding.run.graph_scope {
            return Err(GraphClauseReplaySealRefusal::GraphMismatch);
        }
        if observed.run.objective_identity != self.binding.run.objective_identity {
            return Err(GraphClauseReplaySealRefusal::ObjectiveMismatch);
        }
        if observed.run.root_identity != self.binding.run.root_identity {
            return Err(GraphClauseReplaySealRefusal::RootMismatch);
        }
        if observed.source_history_identity != self.binding.source_history_identity {
            return Err(GraphClauseReplaySealRefusal::SourceHistoryMismatch);
        }
        if observed.candidate_history_identity != self.binding.candidate_history_identity {
            return Err(GraphClauseReplaySealRefusal::CandidateHistoryMismatch);
        }
        if self.candidate_history.constraints.is_empty()
            || self.candidate_history.constraints.len() >= self.source_literal_count
            || !self.candidate_history.is_pure_relu_at_zero()
        {
            return Err(GraphClauseReplaySealRefusal::MalformedCandidate);
        }
        Ok(ReplayVerifiedGraphClause {
            candidate_history: self.candidate_history,
            binding: self.binding,
            identity_byte_cap: self.identity_byte_cap,
            deadline: self.deadline,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphClauseReplaySealRefusal {
    DeadlineExpired,
    GraphMismatch,
    ObjectiveMismatch,
    RootMismatch,
    SourceHistoryMismatch,
    CandidateHistoryMismatch,
    MalformedCandidate,
}

/// Non-cloneable, single-use authority that a candidate strict subset was
/// replay-certified under its exact bound objective/root/history identities.
///
/// Fields and constructor are private.  The graph clause store consumes this
/// token by value, so safe Rust cannot insert it twice.
#[derive(Debug)]
pub(super) struct ReplayVerifiedGraphClause {
    candidate_history: GraphSplitHistory,
    binding: GraphClauseReplayFingerprint,
    identity_byte_cap: usize,
    deadline: Instant,
}

impl ReplayVerifiedGraphClause {
    /// Consume the token after re-binding it to the store owner's current run
    /// and source leaf.  This is a final stale-token and structural check at
    /// the insertion boundary.
    pub(super) fn into_history_for(
        self,
        current_run: &GraphClauseReplayRunFingerprint,
        current_source_history: &GraphSplitHistory,
    ) -> Option<(GraphSplitHistory, Instant)> {
        if Instant::now() >= self.deadline || &self.binding.run != current_run {
            return None;
        }
        let current_source_identity = current_source_history.exact_provenance_identity()?;
        if current_source_identity.as_slice() != self.binding.source_history_identity.as_ref() {
            return None;
        }
        let candidate_identity = self.candidate_history.exact_provenance_identity()?;
        if candidate_identity.as_slice() != self.binding.candidate_history_identity.as_ref() {
            return None;
        }
        let total = current_run
            .byte_len()
            .checked_add(current_source_identity.len())?
            .checked_add(candidate_identity.len())?;
        if total > self.identity_byte_cap
            || !is_strict_literal_subset(&self.candidate_history, current_source_history)
            || Instant::now() >= self.deadline
        {
            return None;
        }
        Some((self.candidate_history, self.deadline))
    }
}

/// Stateful, allocation-bounded proposal generator.
#[derive(Debug)]
pub(super) struct GraphClauseReplayPlanner {
    run: GraphClauseReplayRunFingerprint,
    source_history_identity: Arc<[u8]>,
    canonical_constraints: Box<[GraphNeuronConstraint]>,
    removal_order: Box<[usize]>,
    attempts_issued: usize,
    limits: GraphClauseReplayLimits,
}

impl GraphClauseReplayPlanner {
    pub(super) fn new(
        source_history: &GraphSplitHistory,
        graph_scope: CutFoldScope,
        objective_identity: &[u8],
        root_identity: &[u8],
        limits: GraphClauseReplayLimits,
        now: Instant,
    ) -> Result<Self, GraphClauseReplayRefusal> {
        if limits.attempt_cap == 0 {
            return Err(GraphClauseReplayRefusal::AttemptCapZero);
        }
        if limits.literal_count_cap < 2 {
            return Err(GraphClauseReplayRefusal::LiteralCountCapTooSmall);
        }
        if limits.identity_byte_cap == 0 {
            return Err(GraphClauseReplayRefusal::IdentityByteCapZero);
        }
        if now >= limits.deadline {
            return Err(GraphClauseReplayRefusal::DeadlineExpired);
        }
        if !source_history.is_pure_relu_at_zero() {
            return Err(GraphClauseReplayRefusal::ImpureHistory);
        }
        let literal_count = source_history.constraints.len();
        if literal_count < 2 {
            return Err(GraphClauseReplayRefusal::TooFewLiterals);
        }
        if literal_count > limits.literal_count_cap {
            return Err(GraphClauseReplayRefusal::LiteralCountExceeded {
                count: literal_count,
                cap: limits.literal_count_cap,
            });
        }

        validate_unique_literals(source_history)?;
        let run = GraphClauseReplayRunFingerprint::from_exact_identities(
            graph_scope,
            objective_identity,
            root_identity,
            limits.identity_byte_cap,
        )?;
        let source_identity = source_history
            .exact_provenance_identity()
            .ok_or(GraphClauseReplayRefusal::HistoryIdentityUnavailable)?;

        // A candidate formed by deleting one constraint is no larger than the
        // source identity.  Reserve/check the worst exact identity footprint
        // up front so `next_proposal` cannot fail after partially issuing a
        // deterministic plan.
        let worst_total = run
            .byte_len()
            .checked_add(source_identity.len())
            .and_then(|n| n.checked_add(source_identity.len()))
            .ok_or(GraphClauseReplayRefusal::IdentityBudgetExceeded)?;
        if worst_total > limits.identity_byte_cap {
            return Err(GraphClauseReplayRefusal::IdentityBudgetExceeded);
        }

        let mut canonical_constraints = source_history.constraints.clone();
        canonical_constraints.sort_unstable_by(|a, b| {
            (a.node_name(), a.neuron_idx(), a.is_active()).cmp(&(
                b.node_name(),
                b.neuron_idx(),
                b.is_active(),
            ))
        });
        let mut removal_order: Vec<usize> = (0..canonical_constraints.len()).collect();
        // Low-impact literals are attempted first. Exact score ties are broken
        // by semantic literal identity, never source insertion order.
        removal_order.sort_unstable_by(|&a, &b| {
            let a = &canonical_constraints[a];
            let b = &canonical_constraints[b];
            a.score().total_cmp(&b.score()).then_with(|| {
                (a.node_name(), a.neuron_idx(), a.is_active()).cmp(&(
                    b.node_name(),
                    b.neuron_idx(),
                    b.is_active(),
                ))
            })
        });

        Ok(Self {
            run,
            source_history_identity: Arc::from(source_identity),
            canonical_constraints: canonical_constraints.into_boxed_slice(),
            removal_order: removal_order.into_boxed_slice(),
            attempts_issued: 0,
            limits,
        })
    }

    pub(super) fn run_fingerprint(&self) -> &GraphClauseReplayRunFingerprint {
        &self.run
    }

    /// Emit the next strict-subset attempt, or `None` after the attempt cap or
    /// the finite one-deletion schedule is exhausted.
    pub(super) fn next_proposal(
        &mut self,
        now: Instant,
    ) -> Result<Option<GraphClauseReplayProposal>, GraphClauseReplayRefusal> {
        if now >= self.limits.deadline {
            return Err(GraphClauseReplayRefusal::DeadlineExpired);
        }
        if self.attempts_issued >= self.limits.attempt_cap
            || self.attempts_issued >= self.removal_order.len()
        {
            return Ok(None);
        }

        let removed_index = self.removal_order[self.attempts_issued];
        let removed = &self.canonical_constraints[removed_index];
        let removed_literal = GraphClauseReplayLiteral {
            node_name: removed.node_name().to_string(),
            neuron_idx: removed.neuron_idx(),
            is_active: removed.is_active(),
        };
        let mut candidate_history = GraphSplitHistory::new();
        for (index, constraint) in self.canonical_constraints.iter().enumerate() {
            if index != removed_index {
                candidate_history.add_constraint(constraint.clone());
            }
        }
        debug_assert_eq!(
            candidate_history.constraints.len() + 1,
            self.canonical_constraints.len()
        );

        let candidate_identity = candidate_history
            .exact_provenance_identity()
            .ok_or(GraphClauseReplayRefusal::HistoryIdentityUnavailable)?;
        let total = self
            .run
            .byte_len()
            .checked_add(self.source_history_identity.len())
            .and_then(|n| n.checked_add(candidate_identity.len()))
            .ok_or(GraphClauseReplayRefusal::IdentityBudgetExceeded)?;
        if total > self.limits.identity_byte_cap {
            return Err(GraphClauseReplayRefusal::IdentityBudgetExceeded);
        }
        let binding = GraphClauseReplayFingerprint {
            run: self.run.clone(),
            source_history_identity: Arc::clone(&self.source_history_identity),
            candidate_history_identity: Arc::from(candidate_identity),
        };
        let proposal = GraphClauseReplayProposal {
            attempt_ordinal: self.attempts_issued,
            kind: GraphClauseReplayProposalKind::DeterministicOneDeletion,
            removed_literal,
            removed_literal_count: 1,
            source_literal_count: self.canonical_constraints.len(),
            candidate_history,
            binding,
            identity_byte_cap: self.limits.identity_byte_cap,
            deadline: self.limits.deadline,
        };
        self.attempts_issued += 1;
        Ok(Some(proposal))
    }
}

/// Bind one opaque Stage-0 ranked history to the exact replay proposal
/// contract. This is not an authority mint: the returned proposal must still
/// cross `replay_and_publish_proposal`, `seal_replay_verified`, and the bound
/// store insertion seam.
fn bind_biccos_q_stage1_proposal(
    source_history: &GraphSplitHistory,
    ranked: super::biccos_q_stage0::BiccosQStage1RankedCandidate,
    run: &GraphClauseReplayRunFingerprint,
    identity_byte_cap: usize,
    deadline: Instant,
    now: Instant,
) -> Result<GraphClauseReplayProposal, GraphClauseReplayRefusal> {
    if now >= deadline {
        return Err(GraphClauseReplayRefusal::DeadlineExpired);
    }
    let candidate_history = ranked.into_history();
    if !is_strict_literal_subset(&candidate_history, source_history) {
        return Err(GraphClauseReplayRefusal::CandidateNotStrictSubset);
    }

    let source_identity = source_history
        .exact_provenance_identity()
        .ok_or(GraphClauseReplayRefusal::HistoryIdentityUnavailable)?;
    let candidate_identity = candidate_history
        .exact_provenance_identity()
        .ok_or(GraphClauseReplayRefusal::HistoryIdentityUnavailable)?;
    let total = run
        .byte_len()
        .checked_add(source_identity.len())
        .and_then(|n| n.checked_add(candidate_identity.len()))
        .ok_or(GraphClauseReplayRefusal::IdentityBudgetExceeded)?;
    if total > identity_byte_cap {
        return Err(GraphClauseReplayRefusal::IdentityBudgetExceeded);
    }

    let mut removed: Vec<_> = source_history
        .constraints
        .iter()
        .filter(|constraint| {
            candidate_history.is_constrained(constraint.node_name(), constraint.neuron_idx())
                != Some(constraint.is_active())
        })
        .collect();
    removed.sort_unstable_by(|a, b| {
        (a.node_name(), a.neuron_idx(), a.is_active()).cmp(&(
            b.node_name(),
            b.neuron_idx(),
            b.is_active(),
        ))
    });
    let representative = removed
        .first()
        .ok_or(GraphClauseReplayRefusal::CandidateNotStrictSubset)?;
    let proposal = GraphClauseReplayProposal {
        attempt_ordinal: 0,
        kind: GraphClauseReplayProposalKind::BiccosQStage1Ranked,
        removed_literal: GraphClauseReplayLiteral {
            node_name: representative.node_name().to_string(),
            neuron_idx: representative.neuron_idx(),
            is_active: representative.is_active(),
        },
        removed_literal_count: removed.len(),
        source_literal_count: source_history.constraints.len(),
        candidate_history,
        binding: GraphClauseReplayFingerprint {
            run: run.clone(),
            source_history_identity: Arc::from(source_identity),
            candidate_history_identity: Arc::from(candidate_identity),
        },
        identity_byte_cap,
        deadline,
    };
    Ok(proposal)
}

/// Per-run, default-off replay executor.
///
/// The executor owns immutable copies of every verdict-adjacent input used by
/// a replay: the root box, root node enclosures, objective rows, thresholds,
/// lower-bound aggregation mode, and graph scope. A caller can offer only a
/// source history and the existing bound store; it cannot substitute the
/// candidate, graph, root, or objective after planning.
pub(super) struct GraphClauseReplayRuntime {
    run: GraphClauseReplayRunFingerprint,
    root_input: Arc<BoundedTensor>,
    root_node_bounds: HashMap<String, Arc<BoundedTensor>>,
    objectives: Box<[Vec<f32>]>,
    thresholds: Box<[f32]>,
    conjunctive: bool,
    biccos_q_stage1_enabled: bool,
    biccos_bcp_shadow_enabled: bool,
    authority_cutoff: Instant,
    attempts: usize,
    accepted_shortenings: usize,
    literals_removed: usize,
    refusals: usize,
    deadline_refusals: usize,
    planner_refusals: usize,
    proof_misses: usize,
    insertion_refusals: usize,
    elapsed: Duration,
}

impl GraphClauseReplayRuntime {
    /// Construct the runtime only for the exact explicit pilot gate.
    ///
    /// The owning graph loop additionally requires the existing graph-clause
    /// store to be enabled before calling this function.
    pub(super) fn from_env<'a>(
        graph: &GraphNetwork,
        root_input: &Arc<BoundedTensor>,
        root_node_bounds: impl Into<NodeBoundsView<'a>>,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        conjunctive: bool,
        enclosing_deadline: Instant,
    ) -> Result<Option<Self>, GraphClauseReplayRefusal> {
        let setup_started = Instant::now();
        #[cfg(test)]
        update_test_runtime_observations(|observations| {
            observations.from_env_calls = observations.from_env_calls.saturating_add(1);
        });
        if !runtime_gate_enabled_from_env() {
            return Ok(None);
        }
        // Do not even construct the read-only compatibility view before the
        // exact parent gate above. In particular, gate-off remains zero-clone,
        // zero-scan, and does not inspect subordinate gates.
        let root_node_bounds = root_node_bounds.into();
        // Subordinate gate: never inspect Stage 1 unless the parent replay gate
        // admitted this runtime.
        let biccos_q_stage1_enabled = biccos_q_stage1_gate_enabled_from_env();
        // The BCP observer is also subordinate to the exact replay identity:
        // without a bound store there is no graph/root/objective/property scope
        // in which an implication may be reported.
        let biccos_bcp_shadow_enabled = biccos_bcp_shadow_gate_enabled_from_env();
        let runtime = Self::new_started(
            graph,
            root_input,
            root_node_bounds,
            objectives,
            thresholds,
            conjunctive,
            enclosing_deadline,
            setup_started,
            biccos_q_stage1_enabled,
            biccos_bcp_shadow_enabled,
        )?;
        if biccos_q_stage1_enabled {
            tracing::info!(
                "BICCOS-Q Stage-1 replay armed \
                 (NY_BICCOS_Q_STAGE1_REPLAY=1; ranked proposal only, exact replay required)"
            );
        }
        if biccos_bcp_shadow_enabled {
            tracing::info!(
                "BICCOS BCP shadow armed \
                 (NY_BICCOS_BCP_SHADOW=1; exact literal telemetry only)"
            );
        }
        Ok(Some(runtime))
    }

    #[cfg(test)]
    fn new(
        graph: &GraphNetwork,
        root_input: &Arc<BoundedTensor>,
        root_node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        conjunctive: bool,
        enclosing_deadline: Instant,
    ) -> Result<Self, GraphClauseReplayRefusal> {
        Self::new_started(
            graph,
            root_input,
            root_node_bounds.into(),
            objectives,
            thresholds,
            conjunctive,
            enclosing_deadline,
            Instant::now(),
            false,
            false,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn new_with_biccos_q_stage1(
        graph: &GraphNetwork,
        root_input: &Arc<BoundedTensor>,
        root_node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        conjunctive: bool,
        enclosing_deadline: Instant,
    ) -> Result<Self, GraphClauseReplayRefusal> {
        Self::new_started(
            graph,
            root_input,
            root_node_bounds.into(),
            objectives,
            thresholds,
            conjunctive,
            enclosing_deadline,
            Instant::now(),
            true,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_started(
        graph: &GraphNetwork,
        root_input: &Arc<BoundedTensor>,
        root_node_bounds: NodeBoundsView<'_>,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        conjunctive: bool,
        enclosing_deadline: Instant,
        setup_started: Instant,
        biccos_q_stage1_enabled: bool,
        biccos_bcp_shadow_enabled: bool,
    ) -> Result<Self, GraphClauseReplayRefusal> {
        let result = (|| {
            let authority_cutoff = enclosing_deadline
                .checked_sub(RUNTIME_ENCLOSING_RESERVE)
                .ok_or(GraphClauseReplayRefusal::DeadlineExpired)?;
            let total_cutoff = setup_started
                .checked_add(RUNTIME_TOTAL_BUDGET)
                .ok_or(GraphClauseReplayRefusal::DeadlineExpired)?;
            let setup_cutoff = authority_cutoff.min(total_cutoff);
            if setup_started >= setup_cutoff || Instant::now() >= setup_cutoff {
                return Err(GraphClauseReplayRefusal::DeadlineExpired);
            }
            if objectives.is_empty() {
                return Err(GraphClauseReplayRefusal::EmptyObjectiveIdentity);
            }
            if objectives.len() != thresholds.len()
                || objectives
                    .iter()
                    .any(|row| row.is_empty() || row.iter().any(|value| !value.is_finite()))
                || thresholds.iter().any(|value| !value.is_finite())
            {
                return Err(GraphClauseReplayRefusal::InvalidRunSemantics);
            }

            let objective_identity =
                encode_objective_identity(objectives, thresholds, conjunctive)?;
            let root_identity = encode_root_identity(root_input)?;
            let run = GraphClauseReplayRunFingerprint::from_exact_identities(
                graph.cut_fold_scope(),
                &objective_identity,
                &root_identity,
                RUNTIME_IDENTITY_BYTE_CAP,
            )?;
            let root_input = Arc::clone(root_input);
            let root_node_bounds = root_node_bounds.to_shared_hash_map();
            let objectives = objectives.to_vec().into_boxed_slice();
            let thresholds = thresholds.to_vec().into_boxed_slice();
            let setup_finished = Instant::now();
            let setup_elapsed = setup_finished.saturating_duration_since(setup_started);
            if setup_finished >= setup_cutoff {
                return Err(GraphClauseReplayRefusal::DeadlineExpired);
            }

            Ok(Self {
                run,
                root_input,
                root_node_bounds,
                objectives,
                thresholds,
                conjunctive,
                biccos_q_stage1_enabled,
                biccos_bcp_shadow_enabled,
                authority_cutoff,
                attempts: 0,
                accepted_shortenings: 0,
                literals_removed: 0,
                refusals: 0,
                deadline_refusals: 0,
                planner_refusals: 0,
                proof_misses: 0,
                insertion_refusals: 0,
                elapsed: setup_elapsed,
            })
        })();
        if let Err(refusal) = &result {
            tracing::info!(
                ?refusal,
                elapsed_ms = setup_started.elapsed().as_millis(),
                "Graph clause replay setup refused"
            );
        }
        result
    }

    pub(super) fn run_fingerprint(&self) -> &GraphClauseReplayRunFingerprint {
        &self.run
    }

    fn capture_current_run(
        &self,
    ) -> Result<GraphClauseReplayRunFingerprint, GraphClauseReplayRefusal> {
        let objective_identity =
            encode_objective_identity(&self.objectives, &self.thresholds, self.conjunctive)?;
        let root_identity = encode_root_identity(&self.root_input)?;
        GraphClauseReplayRunFingerprint::from_exact_identities(
            self.run.graph_scope,
            &objective_identity,
            &root_identity,
            RUNTIME_IDENTITY_BYTE_CAP,
        )
    }

    /// Observe one canonical exact unit implication under the runtime's
    /// immutable graph/root/objective/property identity.
    ///
    /// The store performs a second exact fingerprint comparison. The returned
    /// owned diagnostic carries no phase-publication or solver-state handle.
    pub(super) fn bcp_shadow_first_implication(
        &self,
        store: &super::conflict_clauses_graph::GraphClauseStore,
        history: &GraphSplitHistory,
    ) -> Option<super::conflict_clauses_graph::GraphClauseBcpShadowImplication> {
        if !self.biccos_bcp_shadow_enabled {
            return None;
        }
        store.bcp_shadow_first_implication(true, &self.run, history)
    }

    /// Offer one Stage-0 ranked multi-deletion candidate to the existing exact
    /// replay/token/store boundary.
    ///
    /// `false` means no Stage-1 clause was inserted. In every such path this
    /// method leaves the clause store untouched. The caller may subsequently
    /// invoke established one-deletion replay, but no raw β/gradient-ranked
    /// candidate can reach that store.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_biccos_q_stage1_verified_close(
        &mut self,
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        engine: Option<&dyn GemmEngine>,
        source_history: &GraphSplitHistory,
        beta_state: &GraphBetaState,
        provenance: BiccosQStage1SourceProvenance,
        store: &mut super::conflict_clauses_graph::GraphClauseStore,
    ) -> bool {
        if !self.biccos_q_stage1_enabled {
            return false;
        }
        #[cfg(test)]
        update_test_runtime_observations(|observations| {
            observations.stage1_source_offers = observations.stage1_source_offers.saturating_add(1);
        });

        if !provenance.is_allowlisted() {
            #[cfg(test)]
            update_test_runtime_observations(|observations| {
                observations.stage1_provenance_refusals =
                    observations.stage1_provenance_refusals.saturating_add(1);
            });
            self.refusals = self.refusals.saturating_add(1);
            return false;
        }
        if self.attempts >= RUNTIME_ATTEMPT_CAP
            || graph.cut_fold_scope() != self.run.graph_scope
            || verifier.config.verify_upper_bound != RUNTIME_VERIFY_UPPER_BOUND
            || source_history.constraints.len() < 2
            || source_history.constraints.len() > RUNTIME_LITERAL_COUNT_CAP
            || !source_history.is_pure_relu_at_zero()
        {
            self.refusals = self.refusals.saturating_add(1);
            return false;
        }

        let planning_started = Instant::now();
        if self.elapsed >= RUNTIME_TOTAL_BUDGET || planning_started >= self.authority_cutoff {
            self.record_deadline_refusal();
            return false;
        }
        let Some(remaining_total) = RUNTIME_TOTAL_BUDGET.checked_sub(self.elapsed) else {
            self.record_deadline_refusal();
            return false;
        };
        let Some(total_cutoff) = planning_started.checked_add(remaining_total) else {
            self.record_deadline_refusal();
            return false;
        };
        let planning_deadline = self.authority_cutoff.min(total_cutoff);
        if planning_started >= planning_deadline {
            self.record_deadline_refusal();
            return false;
        }

        // This opaque value contains only the retained semantic history. β and
        // gradient values are ranking hints and cannot enter replay CROWN,
        // fingerprinting, token sealing, or store insertion.
        let Some(ranked) =
            super::biccos_q_stage0::biccos_q_stage1_ranked_candidate(source_history, beta_state)
        else {
            self.planner_refusals = self.planner_refusals.saturating_add(1);
            self.refusals = self.refusals.saturating_add(1);
            self.add_elapsed(planning_started.elapsed());
            return false;
        };
        let observed_run = match self.capture_current_run() {
            Ok(run) if run == self.run => run,
            _ => {
                self.planner_refusals = self.planner_refusals.saturating_add(1);
                self.refusals = self.refusals.saturating_add(1);
                self.add_elapsed(planning_started.elapsed());
                return false;
            }
        };
        let mut proposal = match bind_biccos_q_stage1_proposal(
            source_history,
            ranked,
            &observed_run,
            RUNTIME_IDENTITY_BYTE_CAP,
            planning_deadline,
            Instant::now(),
        ) {
            Ok(proposal) => proposal,
            Err(GraphClauseReplayRefusal::DeadlineExpired) => {
                self.record_deadline_refusal();
                self.add_elapsed(planning_started.elapsed());
                return false;
            }
            Err(_) => {
                self.planner_refusals = self.planner_refusals.saturating_add(1);
                self.refusals = self.refusals.saturating_add(1);
                self.add_elapsed(planning_started.elapsed());
                return false;
            }
        };
        self.add_elapsed(planning_started.elapsed());

        let attempt_started = Instant::now();
        let Some(remaining_total) = RUNTIME_TOTAL_BUDGET.checked_sub(self.elapsed) else {
            self.record_deadline_refusal();
            return false;
        };
        let Some(local_cutoff) =
            attempt_started.checked_add(RUNTIME_ATTEMPT_BUDGET.min(remaining_total))
        else {
            self.record_deadline_refusal();
            return false;
        };
        let attempt_deadline = local_cutoff.min(self.authority_cutoff);
        if attempt_started >= attempt_deadline {
            self.record_deadline_refusal();
            return false;
        }
        proposal.deadline = attempt_deadline;

        self.attempts = self.attempts.saturating_add(1);
        #[cfg(test)]
        update_test_runtime_observations(|observations| {
            observations.proof_attempts = observations.proof_attempts.saturating_add(1);
            observations.stage1_proposals = observations.stage1_proposals.saturating_add(1);
            observations.stage1_proof_attempts =
                observations.stage1_proof_attempts.saturating_add(1);
        });
        tracing::debug!(
            attempt = self.attempts,
            source_literals = source_history.constraints.len(),
            candidate_literals = proposal.candidate_history().constraints.len(),
            removed_literals = proposal.removed_literal_count(),
            "Graph clause replay: proving BICCOS-Q Stage-1 ranked candidate"
        );

        let outcome = self.replay_and_publish_proposal(
            verifier,
            graph,
            engine,
            source_history,
            store,
            proposal,
            attempt_deadline,
        );
        self.add_elapsed(attempt_started.elapsed());
        let accepted = matches!(outcome, ReplayProposalOutcome::Accepted);
        #[cfg(test)]
        if accepted {
            update_test_runtime_observations(|observations| {
                observations.stage1_accepts = observations.stage1_accepts.saturating_add(1);
            });
        }
        accepted
    }

    /// Offer one genuinely verified source close to the bounded replay lane.
    ///
    /// At most one accepted shortening is published for a source. Failed
    /// planning, proof, sealing, insertion, or deadline checks leave the
    /// ordinary source close and queue behavior unchanged.
    pub(super) fn try_generalize_verified_close(
        &mut self,
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        engine: Option<&dyn GemmEngine>,
        source_history: &GraphSplitHistory,
        store: &mut super::conflict_clauses_graph::GraphClauseStore,
    ) {
        #[cfg(test)]
        update_test_runtime_observations(|observations| {
            observations.source_offers = observations.source_offers.saturating_add(1);
        });
        if self.attempts >= RUNTIME_ATTEMPT_CAP
            || graph.cut_fold_scope() != self.run.graph_scope
            || verifier.config.verify_upper_bound != RUNTIME_VERIFY_UPPER_BOUND
            || source_history.constraints.len() < 3
            || !source_history.is_pure_relu_at_zero()
        {
            self.refusals = self.refusals.saturating_add(1);
            return;
        }

        let planning_started = Instant::now();
        if self.elapsed >= RUNTIME_TOTAL_BUDGET || planning_started >= self.authority_cutoff {
            self.record_deadline_refusal();
            return;
        }
        let Some(remaining_total) = RUNTIME_TOTAL_BUDGET.checked_sub(self.elapsed) else {
            self.record_deadline_refusal();
            return;
        };
        let Some(total_cutoff) = planning_started.checked_add(remaining_total) else {
            self.record_deadline_refusal();
            return;
        };
        let planning_deadline = self.authority_cutoff.min(total_cutoff);
        if planning_started >= planning_deadline {
            self.record_deadline_refusal();
            return;
        }

        let limits = GraphClauseReplayLimits {
            attempt_cap: (RUNTIME_ATTEMPT_CAP - self.attempts)
                .min(source_history.constraints.len()),
            literal_count_cap: RUNTIME_LITERAL_COUNT_CAP,
            identity_byte_cap: RUNTIME_IDENTITY_BYTE_CAP,
            deadline: planning_deadline,
        };
        let objective_identity =
            match encode_objective_identity(&self.objectives, &self.thresholds, self.conjunctive) {
                Ok(identity) => identity,
                Err(_) => {
                    self.planner_refusals = self.planner_refusals.saturating_add(1);
                    self.refusals = self.refusals.saturating_add(1);
                    self.add_elapsed(planning_started.elapsed());
                    return;
                }
            };
        let root_identity = match encode_root_identity(&self.root_input) {
            Ok(identity) => identity,
            Err(_) => {
                self.planner_refusals = self.planner_refusals.saturating_add(1);
                self.refusals = self.refusals.saturating_add(1);
                self.add_elapsed(planning_started.elapsed());
                return;
            }
        };
        let planning_now = Instant::now();
        let mut planner = match GraphClauseReplayPlanner::new(
            source_history,
            self.run.graph_scope,
            &objective_identity,
            &root_identity,
            limits,
            planning_now,
        ) {
            Ok(planner) if planner.run_fingerprint() == &self.run => planner,
            Err(GraphClauseReplayRefusal::DeadlineExpired) => {
                self.record_deadline_refusal();
                self.add_elapsed(planning_started.elapsed());
                return;
            }
            Ok(_) | Err(_) => {
                self.planner_refusals = self.planner_refusals.saturating_add(1);
                self.refusals = self.refusals.saturating_add(1);
                self.add_elapsed(planning_started.elapsed());
                return;
            }
        };
        self.add_elapsed(planning_started.elapsed());

        while self.attempts < RUNTIME_ATTEMPT_CAP && self.elapsed < RUNTIME_TOTAL_BUDGET {
            let attempt_started = Instant::now();
            let Some(remaining_total) = RUNTIME_TOTAL_BUDGET.checked_sub(self.elapsed) else {
                self.record_deadline_refusal();
                break;
            };
            let attempt_budget = RUNTIME_ATTEMPT_BUDGET.min(remaining_total);
            let Some(local_cutoff) = attempt_started.checked_add(attempt_budget) else {
                self.record_deadline_refusal();
                break;
            };
            let attempt_deadline = local_cutoff.min(self.authority_cutoff);
            if attempt_started >= attempt_deadline {
                self.record_deadline_refusal();
                break;
            }

            let proposal = match planner.next_proposal(attempt_started) {
                Ok(Some(mut proposal)) => {
                    // Planning has one total-session deadline. Narrow the
                    // private token boundary to this attempt's 250 ms cap.
                    proposal.deadline = attempt_deadline;
                    proposal
                }
                Ok(None) => break,
                Err(GraphClauseReplayRefusal::DeadlineExpired) => {
                    self.record_deadline_refusal();
                    self.add_elapsed(attempt_started.elapsed());
                    break;
                }
                Err(_) => {
                    self.planner_refusals = self.planner_refusals.saturating_add(1);
                    self.refusals = self.refusals.saturating_add(1);
                    self.add_elapsed(attempt_started.elapsed());
                    break;
                }
            };
            self.attempts = self.attempts.saturating_add(1);
            #[cfg(test)]
            update_test_runtime_observations(|observations| {
                observations.proof_attempts = observations.proof_attempts.saturating_add(1);
            });
            debug_assert_eq!(
                proposal.kind(),
                GraphClauseReplayProposalKind::DeterministicOneDeletion
            );
            tracing::debug!(
                attempt = self.attempts,
                ordinal = proposal.attempt_ordinal(),
                removed_node = proposal.removed_literal().node_name,
                removed_neuron = proposal.removed_literal().neuron_idx,
                removed_active = proposal.removed_literal().is_active,
                "Graph clause replay: proving deterministic one-deletion candidate"
            );

            let outcome = self.replay_and_publish_proposal(
                verifier,
                graph,
                engine,
                source_history,
                store,
                proposal,
                attempt_deadline,
            );
            self.add_elapsed(attempt_started.elapsed());
            if matches!(
                outcome,
                ReplayProposalOutcome::Accepted | ReplayProposalOutcome::DeadlineExpired
            ) || Instant::now() >= attempt_deadline
            {
                break;
            }
        }
    }

    /// The sole production proof-to-token-to-store boundary for both proposal
    /// policies. A proposal kind can influence only candidate selection and
    /// telemetry; every authority-bearing step below is shared verbatim.
    fn replay_and_publish_proposal(
        &mut self,
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        engine: Option<&dyn GemmEngine>,
        source_history: &GraphSplitHistory,
        store: &mut super::conflict_clauses_graph::GraphClauseStore,
        proposal: GraphClauseReplayProposal,
        attempt_deadline: Instant,
    ) -> ReplayProposalOutcome {
        let candidate_literal_count = proposal.candidate_history().constraints.len();
        match self.replay_candidate(
            verifier,
            graph,
            engine,
            proposal.candidate_history(),
            attempt_deadline,
        ) {
            ReplayProofOutcome::Verified => {
                let observed = self.capture_current_run().and_then(|observed_run| {
                    GraphClauseReplayFingerprint::capture(
                        &observed_run,
                        source_history,
                        proposal.candidate_history(),
                        RUNTIME_IDENTITY_BYTE_CAP,
                    )
                });
                let Ok(observed) = observed else {
                    self.refusals = self.refusals.saturating_add(1);
                    return ReplayProposalOutcome::Rejected;
                };
                match proposal.seal_replay_verified(observed, Instant::now()) {
                    Ok(token) => {
                        if store.insert_replay_verified(token, source_history) {
                            self.accepted_shortenings = self.accepted_shortenings.saturating_add(1);
                            self.literals_removed = self.literals_removed.saturating_add(
                                source_history
                                    .constraints
                                    .len()
                                    .saturating_sub(candidate_literal_count),
                            );
                            ReplayProposalOutcome::Accepted
                        } else {
                            self.insertion_refusals = self.insertion_refusals.saturating_add(1);
                            self.refusals = self.refusals.saturating_add(1);
                            ReplayProposalOutcome::Rejected
                        }
                    }
                    Err(GraphClauseReplaySealRefusal::DeadlineExpired) => {
                        self.record_deadline_refusal();
                        ReplayProposalOutcome::DeadlineExpired
                    }
                    Err(_) => {
                        self.refusals = self.refusals.saturating_add(1);
                        ReplayProposalOutcome::Rejected
                    }
                }
            }
            ReplayProofOutcome::NotVerified => {
                self.proof_misses = self.proof_misses.saturating_add(1);
                ReplayProposalOutcome::Rejected
            }
            ReplayProofOutcome::DeadlineExpired => {
                self.record_deadline_refusal();
                ReplayProposalOutcome::DeadlineExpired
            }
            ReplayProofOutcome::Refused => {
                self.refusals = self.refusals.saturating_add(1);
                ReplayProposalOutcome::Rejected
            }
        }
    }

    fn replay_candidate(
        &self,
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        engine: Option<&dyn GemmEngine>,
        candidate_history: &GraphSplitHistory,
        deadline: Instant,
    ) -> ReplayProofOutcome {
        if Instant::now() >= deadline {
            return ReplayProofOutcome::DeadlineExpired;
        }
        if graph.cut_fold_scope() != self.run.graph_scope
            || !candidate_history.is_pure_relu_at_zero()
            || verifier.config.verify_upper_bound != RUNTIME_VERIFY_UPPER_BOUND
        {
            return ReplayProofOutcome::Refused;
        }
        let beta_state = match GraphBetaState::from_history(candidate_history) {
            Ok(state) => state,
            Err(_) => return ReplayProofOutcome::Refused,
        };
        let context = GraphCrownContext::new(
            candidate_history,
            None,
            Some(&self.root_node_bounds),
            engine,
        );
        // The same authoritative deadline reaches constrained forward bounds,
        // backward CROWN, and deadline-aware GEMM. Optional Complete Clip work
        // is suppressed because a replay candidate is discarded on any miss.
        let _deadline_scope = verifier
            .complete_clip_deadline_overrides
            .scoped(Some(deadline));
        let _clip_suppression = verifier
            .complete_clip_deadline_overrides
            .suppress_complete_clip_scoped();

        for (objective, threshold) in self.objectives.iter().zip(self.thresholds.iter()) {
            if Instant::now() >= deadline {
                return ReplayProofOutcome::DeadlineExpired;
            }
            let bound = verifier.propagate_crown_with_graph_constraints(
                graph,
                self.root_input.as_ref(),
                &context,
                Some(&beta_state),
                Some(objective),
            );
            let (lower, upper) = match bound {
                Ok((output, _)) if output.len() == 1 => {
                    let Some(&lower) = output.lower().iter().next() else {
                        return ReplayProofOutcome::Refused;
                    };
                    let Some(&upper) = output.upper().iter().next() else {
                        return ReplayProofOutcome::Refused;
                    };
                    (lower, upper)
                }
                Ok(_) => return ReplayProofOutcome::Refused,
                Err(error) if error.is_infeasible_domain() => {
                    return ReplayProofOutcome::Verified;
                }
                Err(error) if error.is_deadline_exceeded() => {
                    return ReplayProofOutcome::DeadlineExpired;
                }
                Err(_) => return ReplayProofOutcome::Refused,
            };
            let row_verified = super::config::BetaCrownConfig::domain_is_verified_for_mode(
                RUNTIME_VERIFY_UPPER_BOUND,
                lower,
                upper,
                *threshold,
            );
            if self.conjunctive && row_verified {
                return ReplayProofOutcome::Verified;
            }
            if !self.conjunctive && !row_verified {
                return ReplayProofOutcome::NotVerified;
            }
        }

        if self.conjunctive {
            ReplayProofOutcome::NotVerified
        } else {
            ReplayProofOutcome::Verified
        }
    }

    fn record_deadline_refusal(&mut self) {
        self.deadline_refusals = self.deadline_refusals.saturating_add(1);
        self.refusals = self.refusals.saturating_add(1);
    }

    fn add_elapsed(&mut self, elapsed: Duration) {
        self.elapsed = self.elapsed.saturating_add(elapsed);
    }
}

impl Drop for GraphClauseReplayRuntime {
    fn drop(&mut self) {
        tracing::info!(
            attempts = self.attempts,
            accepted_shortenings = self.accepted_shortenings,
            literals_removed = self.literals_removed,
            elapsed_ms = self.elapsed.as_millis(),
            refusals = self.refusals,
            deadline_refusals = self.deadline_refusals,
            planner_refusals = self.planner_refusals,
            proof_misses = self.proof_misses,
            insertion_refusals = self.insertion_refusals,
            "Graph clause replay telemetry"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayProofOutcome {
    Verified,
    NotVerified,
    DeadlineExpired,
    Refused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayProposalOutcome {
    Accepted,
    Rejected,
    DeadlineExpired,
}

fn append_identity_bytes(
    identity: &mut Vec<u8>,
    bytes: &[u8],
) -> Result<(), GraphClauseReplayRefusal> {
    let new_len = identity
        .len()
        .checked_add(bytes.len())
        .ok_or(GraphClauseReplayRefusal::IdentityBudgetExceeded)?;
    if new_len > RUNTIME_IDENTITY_BYTE_CAP {
        return Err(GraphClauseReplayRefusal::IdentityBudgetExceeded);
    }
    identity.extend_from_slice(bytes);
    Ok(())
}

fn append_identity_usize(
    identity: &mut Vec<u8>,
    value: usize,
) -> Result<(), GraphClauseReplayRefusal> {
    append_identity_bytes(identity, &value.to_le_bytes())
}

fn append_identity_array(
    identity: &mut Vec<u8>,
    array: &ndarray::ArrayD<f32>,
) -> Result<(), GraphClauseReplayRefusal> {
    append_identity_usize(identity, array.ndim())?;
    for &dimension in array.shape() {
        append_identity_usize(identity, dimension)?;
    }
    append_identity_usize(identity, array.len())?;
    for &value in array {
        append_identity_bytes(identity, &value.to_bits().to_le_bytes())?;
    }
    Ok(())
}

fn encode_objective_identity(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    conjunctive: bool,
) -> Result<Vec<u8>, GraphClauseReplayRefusal> {
    if objectives.is_empty() || objectives.len() != thresholds.len() {
        return Err(GraphClauseReplayRefusal::InvalidRunSemantics);
    }
    let mut identity = Vec::new();
    append_identity_bytes(&mut identity, b"NY_GRAPH_CLAUSE_REPLAY_OBJECTIVE_V1\0")?;
    append_identity_bytes(
        &mut identity,
        &[u8::from(conjunctive), u8::from(RUNTIME_VERIFY_UPPER_BOUND)],
    )?;
    append_identity_usize(&mut identity, objectives.len())?;
    for (objective, threshold) in objectives.iter().zip(thresholds.iter()) {
        if objective.is_empty()
            || objective.iter().any(|value| !value.is_finite())
            || !threshold.is_finite()
        {
            return Err(GraphClauseReplayRefusal::InvalidRunSemantics);
        }
        append_identity_usize(&mut identity, objective.len())?;
        for &value in objective {
            append_identity_bytes(&mut identity, &value.to_bits().to_le_bytes())?;
        }
        append_identity_bytes(&mut identity, &threshold.to_bits().to_le_bytes())?;
    }
    Ok(identity)
}

fn encode_root_identity(input: &BoundedTensor) -> Result<Vec<u8>, GraphClauseReplayRefusal> {
    let mut identity = Vec::new();
    append_identity_bytes(&mut identity, b"NY_GRAPH_CLAUSE_REPLAY_ROOT_V1\0")?;
    append_identity_array(&mut identity, input.lower())?;
    append_identity_array(&mut identity, input.upper())?;
    match input.l2_constraint() {
        Some(l2) => {
            append_identity_bytes(&mut identity, &[1])?;
            append_identity_usize(&mut identity, l2.axis())?;
            append_identity_array(&mut identity, l2.center())?;
            append_identity_array(&mut identity, l2.radius())?;
        }
        None => append_identity_bytes(&mut identity, &[0])?,
    }
    Ok(identity)
}

fn validate_unique_literals(history: &GraphSplitHistory) -> Result<(), GraphClauseReplayRefusal> {
    let mut phases: HashMap<(&str, usize), bool> =
        HashMap::with_capacity(history.constraints.len());
    for constraint in &history.constraints {
        let key = (constraint.node_name(), constraint.neuron_idx());
        if let Some(previous_phase) = phases.insert(key, constraint.is_active()) {
            let refusal = if previous_phase == constraint.is_active() {
                GraphClauseReplayRefusal::DuplicateLiteral {
                    node_name: constraint.node_name().to_string(),
                    neuron_idx: constraint.neuron_idx(),
                }
            } else {
                GraphClauseReplayRefusal::OppositePhases {
                    node_name: constraint.node_name().to_string(),
                    neuron_idx: constraint.neuron_idx(),
                }
            };
            return Err(refusal);
        }
    }
    Ok(())
}

fn is_strict_literal_subset(candidate: &GraphSplitHistory, source: &GraphSplitHistory) -> bool {
    if !candidate.is_pure_relu_at_zero()
        || !source.is_pure_relu_at_zero()
        || candidate.constraints.is_empty()
        || candidate.constraints.len() >= source.constraints.len()
        || validate_unique_literals(candidate).is_err()
        || validate_unique_literals(source).is_err()
    {
        return false;
    }
    candidate.constraints.iter().all(|constraint| {
        source.is_constrained(constraint.node_name(), constraint.neuron_idx())
            == Some(constraint.is_active())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::branching::{GenBabConstraint, NormInvRmsConstraint};
    use crate::beta_crown::config::BetaCrownConfig;
    use crate::beta_crown::conflict_clauses_graph::{
        reset_test_store_mutations, test_store_mutations, GraphClauseBcpShadowImplication,
        GraphClauseBcpShadowProvenance, GraphClauseStore,
    };
    use crate::beta_crown::state::GraphBetaEntry;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, arr2};
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    type Lit<'a> = (&'a str, usize, bool, f32);

    struct CountingNodeBoundsView<'a> {
        source: &'a HashMap<String, Arc<BoundedTensor>>,
        conversions: &'a Cell<usize>,
    }

    impl<'a> From<CountingNodeBoundsView<'a>> for NodeBoundsView<'a> {
        fn from(value: CountingNodeBoundsView<'a>) -> Self {
            value.conversions.set(value.conversions.get() + 1);
            value.source.into()
        }
    }

    fn history_of(lits: &[Lit<'_>]) -> GraphSplitHistory {
        let mut history = GraphSplitHistory::new();
        for &(node, index, phase, score) in lits {
            history.add_constraint(
                GraphNeuronConstraint::new(node.to_string(), index, phase, score)
                    .expect("finite test score"),
            );
        }
        history
    }

    fn limits(now: Instant) -> GraphClauseReplayLimits {
        GraphClauseReplayLimits {
            attempt_cap: 8,
            literal_count_cap: 32,
            identity_byte_cap: 64 * 1024,
            deadline: now + Duration::from_secs(1),
        }
    }

    fn source() -> GraphSplitHistory {
        history_of(&[
            ("relu_b", 2, true, 0.25),
            ("relu_a", 4, false, 0.25),
            ("relu_c", 1, true, 0.05),
        ])
    }

    fn test_planner(
        source_history: &GraphSplitHistory,
        objective_identity: &[u8],
        root_identity: &[u8],
        limits: GraphClauseReplayLimits,
        now: Instant,
    ) -> Result<GraphClauseReplayPlanner, GraphClauseReplayRefusal> {
        GraphClauseReplayPlanner::new(
            source_history,
            CutFoldScope::fresh(),
            objective_identity,
            root_identity,
            limits,
            now,
        )
    }

    fn make_proposal(
        now: Instant,
    ) -> (
        GraphClauseReplayRunFingerprint,
        GraphSplitHistory,
        GraphClauseReplayProposal,
    ) {
        let source = source();
        let mut planner = test_planner(&source, b"objective-v1", b"root-v1", limits(now), now)
            .expect("valid plan");
        let run = planner.run_fingerprint().clone();
        let proposal = planner
            .next_proposal(now)
            .expect("within deadline")
            .expect("one proposal");
        (run, source, proposal)
    }

    fn mint_verified(
        run: &GraphClauseReplayRunFingerprint,
        source: &GraphSplitHistory,
        proposal: GraphClauseReplayProposal,
        now: Instant,
    ) -> ReplayVerifiedGraphClause {
        let observed = GraphClauseReplayFingerprint::capture(
            run,
            source,
            proposal.candidate_history(),
            64 * 1024,
        )
        .expect("capture");
        proposal
            .seal_replay_verified(observed, now)
            .expect("matching certified replay")
    }

    #[test]
    fn planner_refuses_impure_histories() {
        let now = Instant::now();
        let mut genbab = source();
        genbab.add_genbab_constraint(
            GenBabConstraint::new("gelu".to_string(), 0, 0.2, true, 1.0).expect("constraint"),
        );
        assert_eq!(
            test_planner(&genbab, b"objective", b"root", limits(now), now).unwrap_err(),
            GraphClauseReplayRefusal::ImpureHistory
        );

        let mut norm = source();
        norm.add_norm_inv_rms_constraint(
            NormInvRmsConstraint::new("norm".to_string(), 0, 0.5, 2.0, 1.0).expect("constraint"),
        );
        assert_eq!(
            test_planner(&norm, b"objective", b"root", limits(now), now).unwrap_err(),
            GraphClauseReplayRefusal::ImpureHistory
        );
    }

    #[test]
    fn planner_refuses_zero_and_one_literal_histories() {
        let now = Instant::now();
        for history in [
            GraphSplitHistory::new(),
            history_of(&[("relu", 0, true, 1.0)]),
        ] {
            assert_eq!(
                test_planner(&history, b"objective", b"root", limits(now), now).unwrap_err(),
                GraphClauseReplayRefusal::TooFewLiterals
            );
        }
    }

    #[test]
    fn planner_refuses_duplicate_and_opposite_phase_literals() {
        let now = Instant::now();
        let duplicate = history_of(&[
            ("relu", 0, true, 1.0),
            ("other", 0, false, 1.0),
            ("relu", 0, true, 2.0),
        ]);
        assert_eq!(
            test_planner(&duplicate, b"objective", b"root", limits(now), now).unwrap_err(),
            GraphClauseReplayRefusal::DuplicateLiteral {
                node_name: "relu".to_string(),
                neuron_idx: 0,
            }
        );

        let opposite = history_of(&[
            ("relu", 0, true, 1.0),
            ("other", 0, false, 1.0),
            ("relu", 0, false, 2.0),
        ]);
        assert_eq!(
            test_planner(&opposite, b"objective", b"root", limits(now), now).unwrap_err(),
            GraphClauseReplayRefusal::OppositePhases {
                node_name: "relu".to_string(),
                neuron_idx: 0,
            }
        );
    }

    #[test]
    fn planner_enforces_attempt_count_identity_and_deadline_caps() {
        let now = Instant::now();
        let source = source();

        let mut zero_attempts = limits(now);
        zero_attempts.attempt_cap = 0;
        assert_eq!(
            test_planner(&source, b"objective", b"root", zero_attempts, now).unwrap_err(),
            GraphClauseReplayRefusal::AttemptCapZero
        );

        let mut count_limited = limits(now);
        count_limited.literal_count_cap = 2;
        assert_eq!(
            test_planner(&source, b"objective", b"root", count_limited, now).unwrap_err(),
            GraphClauseReplayRefusal::LiteralCountExceeded { count: 3, cap: 2 }
        );

        let mut identity_limited = limits(now);
        identity_limited.identity_byte_cap = 8;
        assert_eq!(
            test_planner(&source, b"objective", b"root", identity_limited, now).unwrap_err(),
            GraphClauseReplayRefusal::IdentityBudgetExceeded
        );

        let mut expired = limits(now);
        expired.deadline = now;
        assert_eq!(
            test_planner(&source, b"objective", b"root", expired, now).unwrap_err(),
            GraphClauseReplayRefusal::DeadlineExpired
        );

        let mut one_attempt = limits(now);
        one_attempt.attempt_cap = 1;
        let mut planner = test_planner(&source, b"objective", b"root", one_attempt, now)
            .expect("valid capped planner");
        assert!(planner.next_proposal(now).unwrap().is_some());
        assert!(planner.next_proposal(now).unwrap().is_none());
        assert_eq!(
            planner.next_proposal(one_attempt.deadline).unwrap_err(),
            GraphClauseReplayRefusal::DeadlineExpired
        );
    }

    #[test]
    fn proposals_are_strict_nonempty_subsets_and_ranking_is_deterministic() {
        let now = Instant::now();
        let source = source();
        let mut first =
            test_planner(&source, b"objective", b"root", limits(now), now).expect("planner");
        // Same semantic literals in a different source order. Removal ranking
        // and canonical candidate histories must still agree.
        let reordered = history_of(&[
            ("relu_c", 1, true, 0.05),
            ("relu_a", 4, false, 0.25),
            ("relu_b", 2, true, 0.25),
        ]);
        let mut second =
            test_planner(&reordered, b"objective", b"root", limits(now), now).expect("planner");

        let mut removed = Vec::new();
        while let Some(a) = first.next_proposal(now).expect("deadline") {
            let b = second
                .next_proposal(now)
                .expect("deadline")
                .expect("same schedule");
            assert_eq!(a.attempt_ordinal(), removed.len());
            assert_eq!(a.removed_literal(), b.removed_literal());
            assert_eq!(
                a.candidate_history().exact_provenance_identity(),
                b.candidate_history().exact_provenance_identity()
            );
            assert!(is_strict_literal_subset(a.candidate_history(), &source));
            assert_eq!(a.candidate_history().constraints.len(), 2);
            removed.push(a.removed_literal().node_name.clone());
        }
        assert!(second.next_proposal(now).unwrap().is_none());
        assert_eq!(
            removed,
            ["relu_c", "relu_a", "relu_b"],
            "lowest score first; exact ties use semantic literal order"
        );
    }

    #[test]
    fn replay_seal_rejects_mismatched_objective_root_and_histories() {
        let now = Instant::now();

        let (_run, source, proposal) = make_proposal(now);
        let wrong_run = GraphClauseReplayRunFingerprint::from_exact_identities(
            CutFoldScope::fresh(),
            b"objective-v1",
            b"root-v1",
            64 * 1024,
        )
        .unwrap();
        let observed = GraphClauseReplayFingerprint::capture(
            &wrong_run,
            &source,
            proposal.candidate_history(),
            64 * 1024,
        )
        .unwrap();
        assert_eq!(
            proposal.seal_replay_verified(observed, now).unwrap_err(),
            GraphClauseReplaySealRefusal::GraphMismatch
        );

        let (run, source, proposal) = make_proposal(now);
        let wrong_run = GraphClauseReplayRunFingerprint::from_exact_identities(
            run.graph_scope,
            b"other-objective",
            b"root-v1",
            64 * 1024,
        )
        .unwrap();
        let observed = GraphClauseReplayFingerprint::capture(
            &wrong_run,
            &source,
            proposal.candidate_history(),
            64 * 1024,
        )
        .unwrap();
        assert_eq!(
            proposal.seal_replay_verified(observed, now).unwrap_err(),
            GraphClauseReplaySealRefusal::ObjectiveMismatch
        );

        let (run_again, source, proposal) = make_proposal(now);
        let wrong_run = GraphClauseReplayRunFingerprint::from_exact_identities(
            run_again.graph_scope,
            b"objective-v1",
            b"other-root",
            64 * 1024,
        )
        .unwrap();
        let observed = GraphClauseReplayFingerprint::capture(
            &wrong_run,
            &source,
            proposal.candidate_history(),
            64 * 1024,
        )
        .unwrap();
        assert_eq!(
            proposal.seal_replay_verified(observed, now).unwrap_err(),
            GraphClauseReplaySealRefusal::RootMismatch
        );

        let (run, _source, proposal) = make_proposal(now);
        let stale_source = history_of(&[
            ("relu_b", 2, true, 0.25),
            ("relu_a", 4, false, 0.25),
            ("relu_c", 1, true, 0.05),
            ("relu_d", 0, false, 1.0),
        ]);
        let observed = GraphClauseReplayFingerprint::capture(
            &run,
            &stale_source,
            proposal.candidate_history(),
            64 * 1024,
        )
        .unwrap();
        assert_eq!(
            proposal.seal_replay_verified(observed, now).unwrap_err(),
            GraphClauseReplaySealRefusal::SourceHistoryMismatch
        );

        let (run, source, proposal) = make_proposal(now);
        let different_candidate =
            history_of(&[("relu_a", 4, false, 0.25), ("relu_c", 1, true, 0.05)]);
        let observed =
            GraphClauseReplayFingerprint::capture(&run, &source, &different_candidate, 64 * 1024)
                .unwrap();
        assert_eq!(
            proposal.seal_replay_verified(observed, now).unwrap_err(),
            GraphClauseReplaySealRefusal::CandidateHistoryMismatch
        );
    }

    #[test]
    fn replay_seal_rejects_completion_at_deadline() {
        let now = Instant::now();
        let (run, source, proposal) = make_proposal(now);
        let observed = GraphClauseReplayFingerprint::capture(
            &run,
            &source,
            proposal.candidate_history(),
            64 * 1024,
        )
        .unwrap();
        assert_eq!(
            proposal
                .seal_replay_verified(observed, limits(now).deadline)
                .unwrap_err(),
            GraphClauseReplaySealRefusal::DeadlineExpired
        );
    }

    #[test]
    fn bound_store_refuses_token_after_private_publication_deadline() {
        let now = Instant::now();
        let (run, source, mut proposal) = make_proposal(now);
        let observed = GraphClauseReplayFingerprint::capture(
            &run,
            &source,
            proposal.candidate_history(),
            64 * 1024,
        )
        .expect("capture");
        // Simulate proof completion immediately before the cutoff, then offer
        // the resulting private token to the store after that cutoff.
        proposal.deadline = now;
        let token = proposal
            .seal_replay_verified(
                observed,
                now.checked_sub(Duration::from_nanos(1))
                    .expect("test process has advanced beyond its first nanosecond"),
            )
            .expect("proof completed before the private deadline");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(true, 16, run);

        assert!(!store.insert_replay_verified(token, &source));
        assert_eq!(
            store.len(),
            0,
            "expired replay authority must not publish a clause"
        );
    }

    #[test]
    fn unbound_store_refuses_replay_verified_token() {
        let now = Instant::now();
        let (run, source, proposal) = make_proposal(now);
        let token = mint_verified(&run, &source, proposal, now);
        let mut store = GraphClauseStore::with_capacity(true, 16);
        assert!(
            !store.insert_replay_verified(token, &source),
            "every ordinary constructor must leave replay insertion unbound"
        );
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn matching_bound_store_consumes_verified_token_once() {
        let now = Instant::now();
        let (run, source, proposal) = make_proposal(now);
        let token = mint_verified(&run, &source, proposal, now);
        let mut single_use_slot = Some(token);
        let mut store = GraphClauseStore::with_capacity_and_replay_run(true, 16, run);
        assert!(store.insert_replay_verified(
            single_use_slot
                .take()
                .expect("token available exactly once"),
            &source,
        ));
        assert!(
            single_use_slot.take().is_none(),
            "the non-Clone token was moved into its sole insertion"
        );
        assert_eq!(store.len(), 1);
        assert!(
            store.should_prune(&source),
            "the replay-certified strict subset covers its source leaf"
        );
    }

    #[test]
    fn bcp_shadow_preserves_replay_generalized_provenance_and_scope() {
        let now = Instant::now();
        let (run, source, proposal) = make_proposal(now);
        let candidate = proposal.candidate_history().clone();
        assert_eq!(candidate.constraints.len(), 2);
        let token = mint_verified(&run, &source, proposal, now);
        let mut store = GraphClauseStore::with_capacity_and_replay_run(true, 16, run.clone());
        assert!(store.insert_replay_verified(token, &source));

        let retained = &candidate.constraints[0];
        let missing = &candidate.constraints[1];
        let partial = history_of(&[(
            retained.node_name(),
            retained.neuron_idx(),
            retained.is_active(),
            retained.score(),
        )]);
        let expected = GraphClauseBcpShadowImplication {
            node_name: missing.node_name().to_string(),
            neuron_idx: missing.neuron_idx(),
            is_active: !missing.is_active(),
            provenance: GraphClauseBcpShadowProvenance::ReplayVerifiedGeneralized,
            source_clause_len: candidate.constraints.len(),
        };

        reset_test_store_mutations();
        assert_eq!(
            store.bcp_shadow_first_implication(true, &run, &partial),
            Some(expected),
            "a replay token must remain distinguishable from an ordinary full close"
        );
        assert_eq!(
            test_store_mutations(),
            0,
            "reading a replay-generalized implication cannot mutate the store"
        );

        let foreign_property = GraphClauseReplayRunFingerprint::from_exact_identities(
            run.graph_scope,
            b"foreign-objective-threshold-mode",
            b"root-v1",
            64 * 1024,
        )
        .expect("foreign exact fingerprint");
        assert!(
            store
                .bcp_shadow_first_implication(true, &foreign_property, &partial)
                .is_none(),
            "replay-generalized clauses must cross the same exact scope gate"
        );
    }

    #[test]
    fn bound_store_refuses_stale_source_history() {
        let now = Instant::now();
        let (run, source, proposal) = make_proposal(now);
        let token = mint_verified(&run, &source, proposal, now);
        let stale_source = source.with_constraint(
            GraphNeuronConstraint::new("later".to_string(), 0, true, 1.0).unwrap(),
        );
        let mut stale_store = GraphClauseStore::with_capacity_and_replay_run(true, 16, run);
        assert!(!stale_store.insert_replay_verified(token, &stale_source));
        assert_eq!(stale_store.len(), 0);
    }

    #[test]
    fn store_owned_run_refuses_foreign_token_and_never_rebinds() {
        let now = Instant::now();
        let (run_a, source_a, proposal_a) = make_proposal(now);
        let token_a = mint_verified(&run_a, &source_a, proposal_a, now);

        let source_b = source();
        let mut planner_b = GraphClauseReplayPlanner::new(
            &source_b,
            run_a.graph_scope,
            b"objective-v1",
            b"root-B",
            limits(now),
            now,
        )
        .expect("run B planner");
        let run_b = planner_b.run_fingerprint().clone();
        let proposal_b = planner_b
            .next_proposal(now)
            .expect("within deadline")
            .expect("run B proposal");
        let token_b = mint_verified(&run_b, &source_b, proposal_b, now);

        // The caller still owns run A, but insertion has no run argument it
        // could use to bless token A. Store B consults only its private run B.
        let _caller_owned_run_a = run_a;
        let mut store_b = GraphClauseStore::with_capacity_and_replay_run(true, 16, run_b);
        assert!(!store_b.insert_replay_verified(token_a, &source_a));
        assert_eq!(store_b.len(), 0);

        // A refused foreign token cannot mutate/rebind the store. Its original
        // run-B token remains admissible under the same immutable binding.
        assert!(store_b.insert_replay_verified(token_b, &source_b));
        assert_eq!(store_b.len(), 1);
    }

    #[test]
    fn bound_store_rejects_cross_graph_token_without_mutation() {
        let now = Instant::now();
        let source = source();
        let graph_a = CutFoldScope::fresh();
        let graph_b = CutFoldScope::fresh();

        let mut planner_a = GraphClauseReplayPlanner::new(
            &source,
            graph_a,
            b"objective-v1",
            b"root-v1",
            limits(now),
            now,
        )
        .expect("graph A planner");
        let run_a = planner_a.run_fingerprint().clone();
        let proposal_a = planner_a
            .next_proposal(now)
            .expect("within deadline")
            .expect("graph A proposal");
        let token_a = mint_verified(&run_a, &source, proposal_a, now);

        let mut planner_b = GraphClauseReplayPlanner::new(
            &source,
            graph_b,
            b"objective-v1",
            b"root-v1",
            limits(now),
            now,
        )
        .expect("graph B planner");
        let run_b = planner_b.run_fingerprint().clone();
        let proposal_b = planner_b
            .next_proposal(now)
            .expect("within deadline")
            .expect("graph B proposal");
        let token_b = mint_verified(&run_b, &source, proposal_b, now);
        let mut store_b = GraphClauseStore::with_capacity_and_replay_run(true, 16, run_b);

        assert!(
            !store_b.insert_replay_verified(token_a, &source),
            "bit-identical root/objective bytes from another graph cannot acquire authority"
        );
        assert_eq!(
            store_b.len(),
            0,
            "cross-graph refusal must not mutate the bound store"
        );
        assert!(
            store_b.insert_replay_verified(token_b, &source),
            "cross-graph refusal must not rebind or poison graph B's store"
        );
        assert_eq!(store_b.len(), 1);
    }

    fn replay_test_graph() -> (
        GraphNetwork,
        Arc<BoundedTensor>,
        HashMap<String, Arc<BoundedTensor>>,
    ) {
        let input_linear =
            LinearLayer::new(ndarray::Array2::<f32>::eye(3), None).expect("identity linear");
        let output_linear =
            LinearLayer::new(arr2(&[[-1.0, -1.0, 0.0]]), None).expect("valid output linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(input_linear),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(output_linear),
            vec!["relu1".to_string()],
        ));
        graph.set_output("linear2");

        let input = Arc::new(
            BoundedTensor::new(
                arr1(&[-1.0, -1.0, -1.0]).into_dyn(),
                arr1(&[1.0, 1.0, 1.0]).into_dyn(),
            )
            .expect("valid root box"),
        );
        let node_bounds = graph
            .collect_node_bounds(input.as_ref())
            .expect("sound root bounds")
            .into_iter()
            .map(|(name, bounds)| (name, Arc::new(bounds)))
            .collect();
        (graph, input, node_bounds)
    }

    fn replay_test_source() -> GraphSplitHistory {
        history_of(&[
            // The zero-score irrelevant literal is deterministically removed
            // first. Keeping the two inactive output-support literals proves
            // y=0 > -0.5 on the larger candidate region.
            ("relu1", 2, false, 0.0),
            ("relu1", 0, false, 1.0),
            ("relu1", 1, false, 2.0),
        ])
    }

    fn replay_test_stage1_beta() -> GraphBetaState {
        let mut entries = Vec::new();
        for (node, neuron, phase, value, grad) in [
            ("relu1", 0, false, 0.5, 0.0),
            ("relu1", 1, false, 0.0, 0.8),
            ("relu1", 2, false, 0.0, 0.1),
        ] {
            let mut entry = GraphBetaEntry::new(
                node.to_string(),
                neuron,
                0.0,
                value,
                if phase { 1.0 } else { -1.0 },
            )
            .expect("valid test beta");
            entry.grad = grad;
            entries.push(entry);
        }
        GraphBetaState::from_entries(entries)
    }

    fn replay_test_multi_delete_graph() -> (
        GraphNetwork,
        Arc<BoundedTensor>,
        HashMap<String, Arc<BoundedTensor>>,
    ) {
        let input_linear =
            LinearLayer::new(ndarray::Array2::<f32>::eye(6), None).expect("identity linear");
        let output_linear = LinearLayer::new(arr2(&[[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]]), None)
            .expect("valid output linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(input_linear),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(output_linear),
            vec!["relu1".to_string()],
        ));
        graph.set_output("linear2");

        let input = Arc::new(
            BoundedTensor::new(
                arr1(&[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]).into_dyn(),
                arr1(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).into_dyn(),
            )
            .expect("valid root box"),
        );
        let node_bounds = graph
            .collect_node_bounds(input.as_ref())
            .expect("sound root bounds")
            .into_iter()
            .map(|(name, bounds)| (name, Arc::new(bounds)))
            .collect();
        (graph, input, node_bounds)
    }

    fn replay_test_multi_delete_source() -> GraphSplitHistory {
        history_of(&[
            ("relu1", 0, false, 0.0),
            ("relu1", 1, false, 1.0),
            ("relu1", 2, false, 2.0),
            ("relu1", 3, false, 3.0),
            ("relu1", 4, false, 4.0),
            ("relu1", 5, false, 5.0),
        ])
    }

    fn replay_test_multi_delete_beta() -> GraphBetaState {
        let mut entries = Vec::new();
        for (neuron, value, grad) in [
            (0, 0.5, 0.0),
            (1, 0.0, 0.9),
            (2, 0.0, 0.8),
            (3, 0.0, 0.3),
            (4, 0.0, 0.2),
            (5, 0.0, 0.1),
        ] {
            let mut entry = GraphBetaEntry::new("relu1".to_string(), neuron, 0.0, value, -1.0)
                .expect("valid test beta");
            entry.grad = grad;
            entries.push(entry);
        }
        GraphBetaState::from_entries(entries)
    }

    #[test]
    fn runtime_gate_requires_exact_one() {
        assert!(!runtime_gate_enabled(None));
        assert!(!runtime_gate_enabled(Some("0")));
        assert!(!runtime_gate_enabled(Some("true")));
        assert!(runtime_gate_enabled(Some("1")));
    }

    #[test]
    fn biccos_q_stage1_gate_requires_exact_one() {
        assert!(!biccos_q_stage1_gate_enabled(None));
        assert!(!biccos_q_stage1_gate_enabled(Some("")));
        assert!(!biccos_q_stage1_gate_enabled(Some("0")));
        assert!(!biccos_q_stage1_gate_enabled(Some("true")));
        assert!(!biccos_q_stage1_gate_enabled(Some(" 1")));
        assert!(biccos_q_stage1_gate_enabled(Some("1")));
    }

    #[test]
    fn biccos_bcp_shadow_gate_requires_exact_one() {
        assert!(!biccos_bcp_shadow_gate_enabled(None));
        assert!(!biccos_bcp_shadow_gate_enabled(Some("")));
        assert!(!biccos_bcp_shadow_gate_enabled(Some("0")));
        assert!(!biccos_bcp_shadow_gate_enabled(Some("true")));
        assert!(!biccos_bcp_shadow_gate_enabled(Some(" 1")));
        assert!(biccos_bcp_shadow_gate_enabled(Some("1")));
    }

    #[test]
    fn biccos_q_stage1_gate_is_read_only_beneath_parent_replay_gate() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![-0.5];
        let deadline = Instant::now() + Duration::from_secs(10);

        set_test_runtime_gate_override(Some(false));
        set_test_biccos_q_stage1_gate_override(Some(true));
        set_test_biccos_bcp_shadow_gate_override(Some(true));
        reset_test_runtime_observations();
        let view_conversions = Cell::new(0);
        let strong_counts_before: Vec<_> = node_bounds.values().map(Arc::strong_count).collect();
        let disabled = GraphClauseReplayRuntime::from_env(
            &graph,
            &input,
            CountingNodeBoundsView {
                source: &node_bounds,
                conversions: &view_conversions,
            },
            &objectives,
            &thresholds,
            false,
            deadline,
        )
        .expect("parent gate-off is a clean no-op");
        assert!(disabled.is_none());
        assert_eq!(view_conversions.get(), 0, "gate-off must not invoke Into");
        assert_eq!(
            node_bounds
                .values()
                .map(Arc::strong_count)
                .collect::<Vec<_>>(),
            strong_counts_before,
            "gate-off must not clone any root node-bound Arc"
        );
        assert_eq!(test_runtime_observations().from_env_calls, 1);
        assert_eq!(
            test_runtime_observations().stage1_gate_reads,
            0,
            "the subordinate gate must not be inspected outside its parent"
        );
        assert_eq!(
            test_runtime_observations().bcp_shadow_gate_reads,
            0,
            "the BCP shadow gate must not be inspected outside its parent"
        );

        set_test_runtime_gate_override(Some(true));
        reset_test_runtime_observations();
        let enabled = GraphClauseReplayRuntime::from_env(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            deadline,
        )
        .expect("valid replay runtime");
        assert!(enabled.is_some());
        assert_eq!(test_runtime_observations().stage1_gate_reads, 1);
        assert_eq!(test_runtime_observations().bcp_shadow_gate_reads, 1);

        set_test_runtime_gate_override(None);
        set_test_biccos_q_stage1_gate_override(None);
        set_test_biccos_bcp_shadow_gate_override(None);
    }

    #[test]
    fn runtime_construction_refuses_expired_and_inside_reserve_deadlines() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![-0.5];
        let now = Instant::now();

        for enclosing_deadline in [now, now + Duration::from_secs(4)] {
            let result = GraphClauseReplayRuntime::new(
                &graph,
                &input,
                &node_bounds,
                &objectives,
                &thresholds,
                false,
                enclosing_deadline,
            );
            assert!(matches!(
                result,
                Err(GraphClauseReplayRefusal::DeadlineExpired)
            ));
        }
    }

    #[test]
    fn runtime_setup_elapsed_is_debited_from_total_budget() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![-0.5];
        let setup_debit = Duration::from_millis(100);
        let setup_started = Instant::now()
            .checked_sub(setup_debit)
            .expect("test process has run for at least 100 ms");
        let runtime = GraphClauseReplayRuntime::new_started(
            &graph,
            &input,
            (&node_bounds).into(),
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
            setup_started,
            false,
            false,
        )
        .expect("setup remains within both budgets");

        assert!(
            runtime.elapsed >= setup_debit,
            "identity encoding and immutable clones must start with the setup debit"
        );
        assert!(runtime.elapsed < RUNTIME_TOTAL_BUDGET);
    }

    #[test]
    fn stage1_default_dark_refuses_before_proposal_or_store_mutation() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![-0.5];
        let mut runtime = GraphClauseReplayRuntime::new(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid default-dark runtime");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(
            true,
            16,
            runtime.run_fingerprint().clone(),
        );
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            use_alpha_crown: false,
            use_crown_ibp: false,
            beta_iterations: 0,
            timeout: Duration::from_secs(10),
            ..Default::default()
        });
        let source = replay_test_source();
        let beta = replay_test_stage1_beta();
        reset_test_runtime_observations();
        reset_test_store_mutations();

        let accepted = runtime.try_biccos_q_stage1_verified_close(
            &verifier,
            &graph,
            None,
            &source,
            &beta,
            BiccosQStage1SourceProvenance::SharedMultiObjectiveChildren,
            &mut store,
        );

        assert!(!accepted);
        assert_eq!(runtime.attempts, 0);
        assert_eq!(test_runtime_observations().stage1_source_offers, 0);
        assert_eq!(test_runtime_observations().stage1_proof_attempts, 0);
        assert_eq!(store.len(), 0);
        assert_eq!(test_store_mutations(), 0);
    }

    #[test]
    fn stage1_provenance_allowlist_rejects_legacy_drop_without_mutation() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![-0.5];
        let mut runtime = GraphClauseReplayRuntime::new_with_biccos_q_stage1(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid Stage-1 runtime");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(
            true,
            16,
            runtime.run_fingerprint().clone(),
        );
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            use_alpha_crown: false,
            use_crown_ibp: false,
            beta_iterations: 0,
            timeout: Duration::from_secs(10),
            ..Default::default()
        });
        let source = replay_test_source();
        let beta = replay_test_stage1_beta();
        reset_test_runtime_observations();
        reset_test_store_mutations();

        let accepted = runtime.try_biccos_q_stage1_verified_close(
            &verifier,
            &graph,
            None,
            &source,
            &beta,
            BiccosQStage1SourceProvenance::SharedMultiObjectiveChildrenWithViolatedSiblingDrop,
            &mut store,
        );

        assert!(!accepted);
        assert_eq!(runtime.attempts, 0);
        assert_eq!(test_runtime_observations().stage1_source_offers, 1);
        assert_eq!(test_runtime_observations().stage1_provenance_refusals, 1);
        assert_eq!(test_runtime_observations().stage1_proof_attempts, 0);
        assert_eq!(store.len(), 0);
        assert_eq!(test_store_mutations(), 0);
    }

    #[test]
    fn stage1_proof_miss_cannot_insert_and_ranking_inputs_are_immutable() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        // The ranked candidate fixes relu1 neurons 0/1 inactive, so y=0 cannot
        // prove the strict lower-bound predicate y > 0.5.
        let thresholds = vec![0.5];
        let mut runtime = GraphClauseReplayRuntime::new_with_biccos_q_stage1(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid Stage-1 runtime");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(
            true,
            16,
            runtime.run_fingerprint().clone(),
        );
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            use_alpha_crown: false,
            use_crown_ibp: false,
            beta_iterations: 0,
            timeout: Duration::from_secs(10),
            ..Default::default()
        });
        let source = replay_test_source();
        let beta = replay_test_stage1_beta();
        let source_identity = source.exact_provenance_identity();
        let beta_snapshot: Vec<_> = beta
            .entries
            .iter()
            .map(|entry| (entry.value().to_bits(), entry.grad().to_bits()))
            .collect();
        reset_test_runtime_observations();
        reset_test_store_mutations();

        let accepted = runtime.try_biccos_q_stage1_verified_close(
            &verifier,
            &graph,
            None,
            &source,
            &beta,
            BiccosQStage1SourceProvenance::SharedMultiObjectiveChildren,
            &mut store,
        );

        assert!(!accepted);
        assert_eq!(runtime.attempts, 1);
        assert_eq!(runtime.accepted_shortenings, 0);
        assert_eq!(runtime.proof_misses, 1);
        assert_eq!(test_runtime_observations().stage1_proposals, 1);
        assert_eq!(test_runtime_observations().stage1_proof_attempts, 1);
        assert_eq!(test_runtime_observations().stage1_accepts, 0);
        assert_eq!(store.len(), 0);
        assert_eq!(test_store_mutations(), 0);
        assert_eq!(source.exact_provenance_identity(), source_identity);
        assert_eq!(
            beta.entries
                .iter()
                .map(|entry| (entry.value().to_bits(), entry.grad().to_bits()))
                .collect::<Vec<_>>(),
            beta_snapshot,
            "β/gradient state is ranking-only and must remain immutable"
        );
    }

    #[test]
    fn stage1_mints_only_after_exact_candidate_replay() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![-0.5];
        let mut runtime = GraphClauseReplayRuntime::new_with_biccos_q_stage1(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid Stage-1 runtime");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(
            true,
            16,
            runtime.run_fingerprint().clone(),
        );
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            use_alpha_crown: false,
            use_crown_ibp: false,
            beta_iterations: 0,
            timeout: Duration::from_secs(10),
            ..Default::default()
        });
        let source = replay_test_source();
        let beta = replay_test_stage1_beta();
        reset_test_runtime_observations();
        reset_test_store_mutations();

        let accepted = runtime.try_biccos_q_stage1_verified_close(
            &verifier,
            &graph,
            None,
            &source,
            &beta,
            BiccosQStage1SourceProvenance::SharedMultiObjectiveChildren,
            &mut store,
        );

        assert!(accepted);
        assert_eq!(runtime.attempts, 1);
        assert_eq!(runtime.accepted_shortenings, 1);
        assert_eq!(runtime.literals_removed, 1);
        assert_eq!(test_runtime_observations().stage1_proposals, 1);
        assert_eq!(test_runtime_observations().stage1_proof_attempts, 1);
        assert_eq!(test_runtime_observations().stage1_accepts, 1);
        assert_eq!(store.len(), 1);
        assert_eq!(test_store_mutations(), 1);
        assert!(store.should_prune(&source));
        assert_eq!(store.replay_pruned_count(), 1);
    }

    #[test]
    fn stage1_accepts_genuine_multi_deletion_and_cross_prunes_distinct_superset() {
        let (graph, input, node_bounds) = replay_test_multi_delete_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![-0.5];
        let mut runtime = GraphClauseReplayRuntime::new_with_biccos_q_stage1(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid Stage-1 runtime");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(
            true,
            16,
            runtime.run_fingerprint().clone(),
        );
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            use_alpha_crown: false,
            use_crown_ibp: false,
            beta_iterations: 0,
            timeout: Duration::from_secs(10),
            ..Default::default()
        });
        let source = replay_test_multi_delete_source();
        let beta = replay_test_multi_delete_beta();
        reset_test_runtime_observations();
        reset_test_store_mutations();

        let accepted = runtime.try_biccos_q_stage1_verified_close(
            &verifier,
            &graph,
            None,
            &source,
            &beta,
            BiccosQStage1SourceProvenance::SharedMultiObjectiveChildren,
            &mut store,
        );

        assert!(accepted);
        assert_eq!(runtime.attempts, 1);
        assert_eq!(runtime.accepted_shortenings, 1);
        assert_eq!(
            runtime.literals_removed, 3,
            "the 6-literal source must replay-certify the ranked 3-literal candidate"
        );
        assert_eq!(test_runtime_observations().stage1_proposals, 1);
        assert_eq!(test_runtime_observations().stage1_proof_attempts, 1);
        assert_eq!(test_runtime_observations().stage1_accepts, 1);
        assert_eq!(store.len(), 1);
        assert_eq!(test_store_mutations(), 1);

        let distinct_superset = history_of(&[
            ("relu1", 0, false, 10.0),
            ("relu1", 1, false, 11.0),
            ("relu1", 2, false, 12.0),
            // Opposite to the source's dropped neuron-5 literal: this domain is
            // a different branch, so a hit proves genuine cross-pruning by the
            // replayed 3-literal clause rather than source-clause equality.
            ("relu1", 5, true, 13.0),
        ]);
        assert!(store.should_prune(&distinct_superset));
        assert_eq!(store.replay_pruned_count(), 1);

        let opposite_retained_phase = history_of(&[
            ("relu1", 0, true, 20.0),
            ("relu1", 1, false, 21.0),
            ("relu1", 2, false, 22.0),
            ("relu1", 5, true, 23.0),
        ]);
        assert!(
            !store.should_prune(&opposite_retained_phase),
            "an opposite phase for a retained literal must not cross-prune"
        );
    }

    #[test]
    fn runtime_mints_only_after_real_candidate_crown_proof() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![-0.5];
        let mut runtime = GraphClauseReplayRuntime::new(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid runtime");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(
            true,
            16,
            runtime.run_fingerprint().clone(),
        );
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            use_alpha_crown: false,
            use_crown_ibp: false,
            beta_iterations: 0,
            timeout: Duration::from_secs(10),
            ..Default::default()
        });
        let source = replay_test_source();

        runtime.try_generalize_verified_close(&verifier, &graph, None, &source, &mut store);

        assert_eq!(runtime.attempts, 1);
        assert_eq!(runtime.accepted_shortenings, 1);
        assert_eq!(runtime.literals_removed, 1);
        assert_eq!(store.len(), 1);
        assert!(store.should_prune(&source));
        assert_eq!(
            store.replay_pruned_count(),
            1,
            "the inserted strict subset must be identifiable as replay-only"
        );
    }

    #[test]
    fn runtime_proof_miss_cannot_insert_or_prune() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        // No one-deletion candidate can establish y > 0.5 because y <= 0.
        let thresholds = vec![0.5];
        let mut runtime = GraphClauseReplayRuntime::new(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid runtime");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(
            true,
            16,
            runtime.run_fingerprint().clone(),
        );
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            use_alpha_crown: false,
            use_crown_ibp: false,
            beta_iterations: 0,
            timeout: Duration::from_secs(10),
            ..Default::default()
        });
        let source = replay_test_source();

        runtime.try_generalize_verified_close(&verifier, &graph, None, &source, &mut store);

        assert_eq!(runtime.attempts, 3);
        assert_eq!(runtime.accepted_shortenings, 0);
        assert_eq!(runtime.proof_misses, 3);
        assert_eq!(store.len(), 0);
        assert!(!store.should_prune(&source));
    }

    #[test]
    fn runtime_refuses_upper_bound_verifier_before_attempt_or_insertion() {
        let (graph, input, node_bounds) = replay_test_graph();
        let objectives = vec![vec![1.0]];
        let thresholds = vec![0.5];
        let mut runtime = GraphClauseReplayRuntime::new(
            &graph,
            &input,
            &node_bounds,
            &objectives,
            &thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid lower-only runtime");
        let mut store = GraphClauseStore::with_capacity_and_replay_run(
            true,
            16,
            runtime.run_fingerprint().clone(),
        );
        let upper_verifier = BetaCrownVerifier::new(BetaCrownConfig {
            verify_upper_bound: true,
            use_alpha_crown: false,
            use_crown_ibp: false,
            beta_iterations: 0,
            timeout: Duration::from_secs(10),
            ..Default::default()
        });
        let source = replay_test_source();

        runtime.try_generalize_verified_close(&upper_verifier, &graph, None, &source, &mut store);

        assert_eq!(
            runtime.attempts, 0,
            "upper-bound mode must refuse before replay solve"
        );
        assert_eq!(runtime.accepted_shortenings, 0);
        assert_eq!(store.len(), 0);
        assert!(!store.should_prune(&source));
    }
}
