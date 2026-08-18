// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BICCOS-Q Stage-0 shadow telemetry and shared deterministic ranking policy.
//!
//! This module measures, but never publishes, a prospective BICCOS clause
//! reduction policy at the graph multi-objective verified-child seam.  It is
//! deliberately incapable of changing verification:
//!
//! - the only production inputs are immutable split histories and β state;
//! - prospective clauses stay in this private observer and are never exposed
//!   to `GraphClauseStore`, `GraphCutPool`, or a domain queue;
//! - BCP is simulated against immutable histories and returns counters only;
//! - the exact `NY_BICCOS_Q_STAGE0=1` gate is default off.
//!
//! The prospective policy retains every positive-β literal, then keeps the
//! highest-|gradient| zero-β literals until at least half of the source clause
//! remains.  This is measurement, not clause authority: deleting even a
//! zero-β literal still requires independent replay before it can affect BaB.
//!
//! Stage 1 may consume the opaque [`BiccosQStage1RankedCandidate`] produced by
//! this ranking policy, but that value carries no proof authority. β values and
//! gradients disappear at this boundary: only the retained semantic history is
//! handed to the trusted graph-clause replay module, which must independently
//! certify the entire enlarged region before it can mint an insertion token.

use std::collections::VecDeque;

use super::branching::GraphSplitHistory;
use super::state::GraphBetaState;

const MAX_SOURCE_LITERALS: usize = 4_096;
const MAX_SHADOW_CLAUSES: usize = 512;

fn gate_enabled(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn gate_enabled_from_env() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_GATE_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    gate_enabled(std::env::var("NY_BICCOS_Q_STAGE0").ok().as_deref())
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BiccosQStage0TestObservations {
    pub(crate) from_env_calls: usize,
    pub(crate) wave_observations: usize,
    pub(crate) source_offers: usize,
    pub(crate) plans_emitted: usize,
    pub(crate) prospective_unit_hits: usize,
    pub(crate) prospective_conflict_hits: usize,
    pub(crate) replay_unit_hits: usize,
    pub(crate) replay_conflict_hits: usize,
}

#[cfg(test)]
thread_local! {
    static TEST_GATE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static TEST_OBSERVATIONS: std::cell::Cell<BiccosQStage0TestObservations> =
        const { std::cell::Cell::new(BiccosQStage0TestObservations {
            from_env_calls: 0,
            wave_observations: 0,
            source_offers: 0,
            plans_emitted: 0,
            prospective_unit_hits: 0,
            prospective_conflict_hits: 0,
            replay_unit_hits: 0,
            replay_conflict_hits: 0,
        }) };
}

#[cfg(test)]
pub(crate) fn set_test_gate_override(enabled: Option<bool>) {
    TEST_GATE_OVERRIDE.with(|value| value.set(enabled));
}

#[cfg(test)]
pub(crate) fn reset_test_observations() {
    TEST_OBSERVATIONS.with(|observations| {
        observations.set(BiccosQStage0TestObservations::default());
    });
}

#[cfg(test)]
pub(crate) fn test_observations() -> BiccosQStage0TestObservations {
    TEST_OBSERVATIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn update_test_observations(update: impl FnOnce(&mut BiccosQStage0TestObservations)) {
    TEST_OBSERVATIONS.with(|observations| {
        let mut current = observations.get();
        update(&mut current);
        observations.set(current);
    });
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ShadowLiteral {
    node_name: String,
    neuron_idx: usize,
    is_active: bool,
}

impl ShadowLiteral {
    fn assignment_in(&self, history: &GraphSplitHistory) -> Option<bool> {
        history.is_constrained(&self.node_name, self.neuron_idx)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProspectiveClause {
    literals: Box<[ShadowLiteral]>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BcpHits {
    unit: usize,
    conflict: usize,
}

impl BcpHits {
    fn saturating_add_assign(&mut self, other: Self) {
        self.unit = self.unit.saturating_add(other.unit);
        self.conflict = self.conflict.saturating_add(other.conflict);
    }
}

impl ProspectiveClause {
    fn bcp_hits(&self, history: &GraphSplitHistory) -> BcpHits {
        if self.literals.is_empty() || !history.is_pure_relu_at_zero() {
            return BcpHits::default();
        }

        let mut unassigned = 0usize;
        for literal in &self.literals {
            match literal.assignment_in(history) {
                Some(phase) if phase == literal.is_active => {}
                // The forbidden conjunction is already false, so this clause
                // has neither a unit nor a conflict implication here.
                Some(_) => return BcpHits::default(),
                None => unassigned = unassigned.saturating_add(1),
            }
        }
        match unassigned {
            0 => BcpHits {
                unit: 0,
                conflict: 1,
            },
            1 => BcpHits {
                unit: 1,
                conflict: 0,
            },
            _ => BcpHits::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ZeroBetaGradientStats {
    literals: usize,
    missing_entries: usize,
    non_finite: usize,
    minimum: Option<f32>,
    median: Option<f32>,
    maximum: Option<f32>,
    retained_minimum: Option<f32>,
    dropped_maximum: Option<f32>,
}

#[derive(Debug)]
struct RankedLiteral {
    literal: ShadowLiteral,
    source_constraint: super::branching::GraphNeuronConstraint,
    beta_positive: bool,
    abs_gradient: Option<f32>,
}

#[derive(Debug)]
struct ProspectivePlan {
    clause: ProspectiveClause,
    candidate_history: GraphSplitHistory,
    positive_beta_literals: usize,
    zero_beta_stats: ZeroBetaGradientStats,
}

/// Opaque, deterministic Stage-0 ranking result offered to Stage 1.
///
/// Constructing this value proves only that the candidate came from the exact
/// positive-β / high-|gradient| policy below and is a non-empty strict subset
/// of a pure ReLU-at-zero source. It does not prove that the candidate region is
/// verified and deliberately exposes no constructor outside this module.
#[derive(Debug)]
pub(super) struct BiccosQStage1RankedCandidate {
    history: GraphSplitHistory,
}

impl BiccosQStage1RankedCandidate {
    #[cfg(test)]
    pub(super) fn history(&self) -> &GraphSplitHistory {
        &self.history
    }

    pub(super) fn into_history(self) -> GraphSplitHistory {
        self.history
    }
}

fn finite_min(values: impl Iterator<Item = f32>) -> Option<f32> {
    values.reduce(f32::min)
}

fn finite_max(values: impl Iterator<Item = f32>) -> Option<f32> {
    values.reduce(f32::max)
}

fn prospective_half_clause(
    history: &GraphSplitHistory,
    beta_state: &GraphBetaState,
) -> Option<ProspectivePlan> {
    let source_len = history.constraints.len();
    if source_len == 0 || source_len > MAX_SOURCE_LITERALS || !history.is_pure_relu_at_zero() {
        return None;
    }

    let mut ranked = Vec::with_capacity(source_len);
    let mut missing_entries = 0usize;
    let mut non_finite = 0usize;
    for constraint in &history.constraints {
        let sign = if constraint.is_active() { 1.0 } else { -1.0 };
        let entry = beta_state.entry_for_constraint(
            constraint.node_name(),
            constraint.neuron_idx(),
            0.0,
            sign,
        );
        let (beta_positive, abs_gradient) = match entry {
            Some(entry) => {
                let abs_gradient = entry.grad().abs();
                if abs_gradient.is_finite() {
                    (entry.value() > 0.0, Some(abs_gradient))
                } else {
                    non_finite = non_finite.saturating_add(1);
                    (entry.value() > 0.0, None)
                }
            }
            None => {
                // Missing β entries are zero-β/zero-gradient for this shadow
                // ranking. Record them separately so telemetry exposes the
                // fallback instead of silently presenting it as measured.
                missing_entries = missing_entries.saturating_add(1);
                (false, Some(0.0))
            }
        };
        ranked.push(RankedLiteral {
            literal: ShadowLiteral {
                node_name: constraint.node_name().to_string(),
                neuron_idx: constraint.neuron_idx(),
                is_active: constraint.is_active(),
            },
            source_constraint: constraint.clone(),
            beta_positive,
            abs_gradient,
        });
    }

    // Duplicate or opposite-phase assignments violate the path invariant and
    // make the literal ranking ambiguous. Refuse the shadow proposal.
    let mut semantic_order: Vec<_> = ranked.iter().map(|entry| &entry.literal).collect();
    semantic_order.sort_unstable();
    if semantic_order.windows(2).any(|pair| {
        pair[0].node_name == pair[1].node_name && pair[0].neuron_idx == pair[1].neuron_idx
    }) {
        return None;
    }

    let positive_beta_literals = ranked.iter().filter(|entry| entry.beta_positive).count();
    let target_len = source_len.saturating_add(1) / 2;
    let zero_to_keep = target_len
        .saturating_sub(positive_beta_literals)
        .min(source_len.saturating_sub(positive_beta_literals));

    let mut zero_ranked: Vec<_> = ranked
        .iter()
        .enumerate()
        .filter(|(_, entry)| !entry.beta_positive)
        .collect();
    zero_ranked.sort_unstable_by(|(_, a), (_, b)| {
        let a_gradient = a.abs_gradient.unwrap_or(f32::NEG_INFINITY);
        let b_gradient = b.abs_gradient.unwrap_or(f32::NEG_INFINITY);
        b_gradient
            .total_cmp(&a_gradient)
            .then_with(|| a.literal.cmp(&b.literal))
    });

    let mut keep = vec![false; source_len];
    for (index, entry) in ranked.iter().enumerate() {
        if entry.beta_positive {
            keep[index] = true;
        }
    }
    for (index, _) in zero_ranked.iter().take(zero_to_keep) {
        keep[*index] = true;
    }

    let mut clause_literals: Vec<_> = ranked
        .iter()
        .zip(&keep)
        .filter(|(_, keep)| **keep)
        .map(|(entry, _)| entry.literal.clone())
        .collect();
    clause_literals.sort_unstable();
    if clause_literals.is_empty() {
        return None;
    }

    // Preserve the exact retained source constraints, including their advisory
    // scores, while canonicalizing semantic order. The replay fingerprint
    // includes those score bits, so insertion-order-independent ranking must
    // also produce an insertion-order-independent candidate identity.
    let mut candidate_constraints: Vec<_> = ranked
        .iter()
        .zip(&keep)
        .filter(|(_, keep)| **keep)
        .map(|(entry, _)| entry.source_constraint.clone())
        .collect();
    candidate_constraints.sort_unstable_by(|a, b| {
        (a.node_name(), a.neuron_idx(), a.is_active()).cmp(&(
            b.node_name(),
            b.neuron_idx(),
            b.is_active(),
        ))
    });
    let mut candidate_history = GraphSplitHistory::new();
    for constraint in candidate_constraints {
        candidate_history.add_constraint(constraint);
    }

    let mut finite_zero_gradients: Vec<f32> = zero_ranked
        .iter()
        .filter_map(|(_, entry)| entry.abs_gradient)
        .collect();
    finite_zero_gradients.sort_unstable_by(f32::total_cmp);
    let median = finite_zero_gradients
        .get(finite_zero_gradients.len().saturating_sub(1) / 2)
        .copied();
    let retained_minimum = finite_min(
        ranked
            .iter()
            .zip(&keep)
            .filter(|(entry, keep)| !entry.beta_positive && **keep)
            .filter_map(|(entry, _)| entry.abs_gradient),
    );
    let dropped_maximum = finite_max(
        ranked
            .iter()
            .zip(&keep)
            .filter(|(entry, keep)| !entry.beta_positive && !**keep)
            .filter_map(|(entry, _)| entry.abs_gradient),
    );

    Some(ProspectivePlan {
        clause: ProspectiveClause {
            literals: clause_literals.into_boxed_slice(),
        },
        candidate_history,
        positive_beta_literals,
        zero_beta_stats: ZeroBetaGradientStats {
            literals: zero_ranked.len(),
            missing_entries,
            non_finite,
            minimum: finite_zero_gradients.first().copied(),
            median,
            maximum: finite_zero_gradients.last().copied(),
            retained_minimum,
            dropped_maximum,
        },
    })
}

/// Return the deterministic Stage-0 ranked history only when it is a strict,
/// non-empty subset suitable for an independent Stage-1 replay attempt.
///
/// All-positive β can legitimately make the prospective Stage-0 clause equal
/// to its source. That remains useful telemetry but is not a generalization and
/// therefore cannot cross this boundary.
pub(super) fn biccos_q_stage1_ranked_candidate(
    source_history: &GraphSplitHistory,
    beta_state: &GraphBetaState,
) -> Option<BiccosQStage1RankedCandidate> {
    let plan = prospective_half_clause(source_history, beta_state)?;
    let candidate = plan.candidate_history;
    if candidate.constraints.is_empty()
        || candidate.constraints.len() >= source_history.constraints.len()
        || !candidate.is_pure_relu_at_zero()
    {
        return None;
    }
    Some(BiccosQStage1RankedCandidate { history: candidate })
}

/// Per-run shadow observer. It owns only prospective clauses and counters.
///
/// No method receives a mutable queue, cut store, bound, or lifecycle state,
/// and no prospective clause is exposed outside this module.
pub(crate) struct BiccosQStage0Telemetry {
    clauses: VecDeque<ProspectiveClause>,
    sources: usize,
    wave_histories: usize,
    prospective_hits: BcpHits,
    replay_hits: BcpHits,
}

impl BiccosQStage0Telemetry {
    pub(crate) fn from_env() -> Option<Self> {
        #[cfg(test)]
        update_test_observations(|observations| {
            observations.from_env_calls = observations.from_env_calls.saturating_add(1);
        });
        gate_enabled_from_env().then(|| {
            tracing::info!(
                "BICCOS-Q Stage-0 telemetry armed (NY_BICCOS_Q_STAGE0=1; \
                 shadow-only, no cut/queue authority)"
            );
            Self {
                clauses: VecDeque::new(),
                sources: 0,
                wave_histories: 0,
                prospective_hits: BcpHits::default(),
                replay_hits: BcpHits::default(),
            }
        })
    }

    /// Replay all previously observed prospective clauses against this child
    /// wave. These are Boolean counters only; no implied assignment is applied.
    pub(crate) fn observe_wave(&mut self, histories: &[&GraphSplitHistory]) {
        #[cfg(test)]
        update_test_observations(|observations| {
            observations.wave_observations = observations.wave_observations.saturating_add(1);
        });
        let mut hits = BcpHits::default();
        for history in histories {
            for clause in &self.clauses {
                hits.saturating_add_assign(clause.bcp_hits(history));
            }
        }
        self.wave_histories = self.wave_histories.saturating_add(histories.len());
        self.replay_hits.saturating_add_assign(hits);
        #[cfg(test)]
        update_test_observations(|observations| {
            observations.replay_unit_hits = observations.replay_unit_hits.saturating_add(hits.unit);
            observations.replay_conflict_hits = observations
                .replay_conflict_hits
                .saturating_add(hits.conflict);
        });
        tracing::info!(
            wave_histories = histories.len(),
            prospective_clause_pool = self.clauses.len(),
            prospective_bcp_unit_hits = hits.unit,
            prospective_bcp_conflict_hits = hits.conflict,
            "BICCOS-Q Stage-0 shadow BCP replay"
        );
    }

    /// Plan and log one prospective 50%-length clause from a genuinely
    /// verified child. `wave_histories` is immutable and is used only to count
    /// cross-child BCP implications; the source itself is excluded.
    pub(crate) fn observe_verified_close(
        &mut self,
        source_history: &GraphSplitHistory,
        beta_state: &GraphBetaState,
        wave_histories: &[&GraphSplitHistory],
    ) {
        self.sources = self.sources.saturating_add(1);
        #[cfg(test)]
        update_test_observations(|observations| {
            observations.source_offers = observations.source_offers.saturating_add(1);
        });

        let source_depth = source_history.depth();
        let clause_len = source_history.constraints.len();
        let Some(plan) = prospective_half_clause(source_history, beta_state) else {
            tracing::info!(
                source_depth,
                clause_len,
                eligible = false,
                "BICCOS-Q Stage-0 source refused"
            );
            return;
        };

        let mut hits = BcpHits::default();
        for history in wave_histories {
            if std::ptr::eq(*history, source_history) {
                continue;
            }
            hits.saturating_add_assign(plan.clause.bcp_hits(history));
        }
        self.prospective_hits.saturating_add_assign(hits);
        #[cfg(test)]
        update_test_observations(|observations| {
            observations.plans_emitted = observations.plans_emitted.saturating_add(1);
            observations.prospective_unit_hits =
                observations.prospective_unit_hits.saturating_add(hits.unit);
            observations.prospective_conflict_hits = observations
                .prospective_conflict_hits
                .saturating_add(hits.conflict);
        });

        let stats = plan.zero_beta_stats;
        tracing::info!(
            source_depth,
            clause_len,
            positive_beta_literals = plan.positive_beta_literals,
            zero_beta_literals = stats.literals,
            zero_beta_missing_entries = stats.missing_entries,
            zero_beta_non_finite_gradients = stats.non_finite,
            zero_beta_abs_grad_min = ?stats.minimum,
            zero_beta_abs_grad_median = ?stats.median,
            zero_beta_abs_grad_max = ?stats.maximum,
            zero_beta_retained_abs_grad_min = ?stats.retained_minimum,
            zero_beta_dropped_abs_grad_max = ?stats.dropped_maximum,
            prospective_reduced_len = plan.clause.literals.len(),
            prospective_bcp_unit_hits = hits.unit,
            prospective_bcp_conflict_hits = hits.conflict,
            "BICCOS-Q Stage-0 prospective 50% clause"
        );

        // Private FIFO shadow corpus. It is never replay-certified and never
        // crosses into a solver store. Dedup only keeps counters interpretable.
        if !self.clauses.iter().any(|clause| clause == &plan.clause) {
            while self.clauses.len() >= MAX_SHADOW_CLAUSES {
                self.clauses.pop_front();
            }
            self.clauses.push_back(plan.clause);
        }
    }
}

impl Drop for BiccosQStage0Telemetry {
    fn drop(&mut self) {
        tracing::info!(
            sources = self.sources,
            wave_histories = self.wave_histories,
            retained_shadow_clauses = self.clauses.len(),
            prospective_bcp_unit_hits = self.prospective_hits.unit,
            prospective_bcp_conflict_hits = self.prospective_hits.conflict,
            replay_bcp_unit_hits = self.replay_hits.unit,
            replay_bcp_conflict_hits = self.replay_hits.conflict,
            "BICCOS-Q Stage-0 telemetry summary"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::beta_crown::state::GraphBetaEntry;

    type Lit<'a> = (&'a str, usize, bool, f32, f32);

    fn fixture(literals: &[Lit<'_>]) -> (GraphSplitHistory, GraphBetaState) {
        let mut history = GraphSplitHistory::new();
        let mut entries = Vec::new();
        for &(node, neuron, phase, beta, grad) in literals {
            history.add_constraint(
                GraphNeuronConstraint::new(node.to_string(), neuron, phase, 1.0)
                    .expect("finite fixture"),
            );
            let mut entry = GraphBetaEntry::new(
                node.to_string(),
                neuron,
                0.0,
                beta,
                if phase { 1.0 } else { -1.0 },
            )
            .expect("valid β entry");
            entry.grad = grad;
            entries.push(entry);
        }
        (history, GraphBetaState::from_entries(entries))
    }

    #[test]
    fn gate_is_exact_and_default_off() {
        assert!(!gate_enabled(None));
        assert!(!gate_enabled(Some("")));
        assert!(!gate_enabled(Some("0")));
        assert!(gate_enabled(Some("1")));
        assert!(!gate_enabled(Some("true")));
        assert!(!gate_enabled(Some(" 1")));
    }

    #[test]
    fn half_plan_keeps_positive_beta_then_top_zero_beta_gradients() {
        let (history, beta) = fixture(&[
            ("relu_b", 0, true, 0.0, 0.2),
            ("relu_a", 0, false, 0.5, 0.01),
            ("relu_d", 0, false, 0.0, -0.8),
            ("relu_c", 0, true, 0.0, 0.4),
            ("relu_e", 0, true, 0.0, 0.1),
        ]);
        let plan = prospective_half_clause(&history, &beta).expect("eligible source");

        assert_eq!(plan.positive_beta_literals, 1);
        assert_eq!(plan.clause.literals.len(), 3, "ceil(5/2) literals retained");
        assert!(plan
            .clause
            .literals
            .iter()
            .any(|literal| literal.node_name == "relu_a"));
        assert!(plan
            .clause
            .literals
            .iter()
            .any(|literal| literal.node_name == "relu_d"));
        assert!(plan
            .clause
            .literals
            .iter()
            .any(|literal| literal.node_name == "relu_c"));
        assert_eq!(plan.zero_beta_stats.literals, 4);
        assert_eq!(plan.zero_beta_stats.minimum, Some(0.1));
        assert_eq!(plan.zero_beta_stats.median, Some(0.2));
        assert_eq!(plan.zero_beta_stats.maximum, Some(0.8));
        assert_eq!(plan.zero_beta_stats.retained_minimum, Some(0.4));
        assert_eq!(plan.zero_beta_stats.dropped_maximum, Some(0.2));
    }

    #[test]
    fn ranking_ties_break_by_semantic_literal_identity() {
        let (history, beta) = fixture(&[
            ("relu_z", 2, true, 0.0, 1.0),
            ("relu_a", 3, false, 0.0, -1.0),
            ("relu_m", 1, true, 0.0, 1.0),
            ("relu_b", 7, false, 0.0, 1.0),
        ]);
        let a = prospective_half_clause(&history, &beta).expect("eligible source");

        let (reordered_history, reordered_beta) = fixture(&[
            ("relu_b", 7, false, 0.0, 1.0),
            ("relu_m", 1, true, 0.0, 1.0),
            ("relu_a", 3, false, 0.0, -1.0),
            ("relu_z", 2, true, 0.0, 1.0),
        ]);
        let b =
            prospective_half_clause(&reordered_history, &reordered_beta).expect("eligible source");

        assert_eq!(a.clause, b.clause);
        assert_eq!(
            a.clause
                .literals
                .iter()
                .map(|literal| literal.node_name.as_str())
                .collect::<Vec<_>>(),
            ["relu_a", "relu_b"]
        );
        assert_eq!(
            a.candidate_history.exact_provenance_identity(),
            b.candidate_history.exact_provenance_identity(),
            "semantic tie-breaking must canonicalize the exact replay candidate too"
        );
    }

    #[test]
    fn stage1_candidate_is_strict_and_all_positive_beta_is_rejected() {
        let (source, beta) = fixture(&[
            ("relu_c", 2, true, 0.0, 0.1),
            ("relu_a", 0, false, 0.5, 0.0),
            ("relu_b", 1, false, 0.0, 0.8),
        ]);
        let candidate =
            biccos_q_stage1_ranked_candidate(&source, &beta).expect("strict ranked candidate");
        assert_eq!(candidate.history().constraints.len(), 2);
        assert!(candidate.history().is_pure_relu_at_zero());
        assert_eq!(
            candidate
                .history()
                .constraints
                .iter()
                .map(|constraint| constraint.node_name())
                .collect::<Vec<_>>(),
            ["relu_a", "relu_b"]
        );

        let (all_positive_source, all_positive_beta) = fixture(&[
            ("relu_a", 0, false, 0.5, 0.0),
            ("relu_b", 1, false, 0.5, 0.0),
            ("relu_c", 2, true, 0.5, 0.0),
        ]);
        assert!(
            biccos_q_stage1_ranked_candidate(&all_positive_source, &all_positive_beta).is_none(),
            "a full-length ranked clause is not a strict-subset replay proposal"
        );
    }

    #[test]
    fn prospective_bcp_counts_unit_conflict_and_satisfied_cases() {
        let clause = ProspectiveClause {
            literals: vec![
                ShadowLiteral {
                    node_name: "a".into(),
                    neuron_idx: 0,
                    is_active: true,
                },
                ShadowLiteral {
                    node_name: "b".into(),
                    neuron_idx: 1,
                    is_active: false,
                },
            ]
            .into_boxed_slice(),
        };
        let (unit, _) = fixture(&[("a", 0, true, 0.0, 0.0)]);
        let (conflict, _) = fixture(&[("a", 0, true, 0.0, 0.0), ("b", 1, false, 0.0, 0.0)]);
        let (satisfied, _) = fixture(&[("a", 0, true, 0.0, 0.0), ("b", 1, true, 0.0, 0.0)]);
        let (open, _) = fixture(&[]);

        assert_eq!(
            clause.bcp_hits(&unit),
            BcpHits {
                unit: 1,
                conflict: 0
            }
        );
        assert_eq!(
            clause.bcp_hits(&conflict),
            BcpHits {
                unit: 0,
                conflict: 1
            }
        );
        assert_eq!(clause.bcp_hits(&satisfied), BcpHits::default());
        assert_eq!(clause.bcp_hits(&open), BcpHits::default());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn guarded_observer_never_mutates_source_history_or_beta_state() {
        let (source, beta) = fixture(&[
            ("a", 0, true, 0.25, 0.1),
            ("b", 1, false, 0.0, 0.8),
            ("c", 2, true, 0.0, 0.4),
        ]);
        let (sibling, _) = fixture(&[("a", 0, true, 0.0, 0.0), ("b", 1, false, 0.0, 0.0)]);
        let history_identity = source
            .exact_provenance_identity()
            .expect("fixture identity");
        let beta_snapshot: Vec<_> = beta
            .entries
            .iter()
            .map(|entry| (entry.value().to_bits(), entry.grad().to_bits()))
            .collect();

        let mut observer = BiccosQStage0Telemetry {
            clauses: VecDeque::new(),
            sources: 0,
            wave_histories: 0,
            prospective_hits: BcpHits::default(),
            replay_hits: BcpHits::default(),
        };
        observer.observe_verified_close(&source, &beta, &[&source, &sibling]);
        observer.observe_wave(&[&source, &sibling]);

        assert_eq!(
            source.exact_provenance_identity().as_deref(),
            Some(history_identity.as_slice())
        );
        assert_eq!(
            beta.entries
                .iter()
                .map(|entry| (entry.value().to_bits(), entry.grad().to_bits()))
                .collect::<Vec<_>>(),
            beta_snapshot
        );
        assert_eq!(observer.sources, 1);
        assert_eq!(observer.clauses.len(), 1);
    }
}
