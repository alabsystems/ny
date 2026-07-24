// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conflict-clause learning for the GRAPH BaB engine (win-plan arc C, v2).
//!
//! Graph-engine port of `conflict_clauses` (v1, sequential engine). When the
//! graph BaB loop closes a domain as verified — its sound bound proves the
//! property on the domain's whole region, or the domain is proven infeasible
//! (empty region, #2926 "trivially verified") — its activation literal set
//! L(D) = {(relu node, neuron, phase)} is recorded as a conflict clause. A
//! later domain D' in the SAME run whose literal set is a superset of a
//! recorded clause covers a subregion of the already-certified region (more
//! fixations = smaller region: pure intersection of fixed half-spaces
//! `z >= 0` / `z <= 0` at split point 0 over one shared root box), so D' may
//! be closed as verified WITHOUT computing bounds.
//!
//! # Soundness argument (state of the art precisely, per the v2 design)
//!
//! Region semantics. For a domain whose ENTIRE history is pure ReLU-at-0
//! literals, its region is EXACTLY `region(L) = root_box ∩ {halfspace(l) : l ∈
//! L}` — including the case where a literal's pre-activation IS the network
//! input (the `input_bounds` tightening in `with_constraint` is the
//! deterministic image of that literal's half-space on the root box, not
//! private state). Therefore `L ⊆ L'  ⇒  region(L') ⊆ region(L)`, and a
//! certificate for region(L) covers region(L').
//!
//! PURITY GUARD (the blocker the design named). `GraphSplitHistory` can also
//! carry GenBaB arbitrary-split-point constraints and norm `inv_rms` clamps,
//! whose regions depend on private split points / windows that the literal
//! vocabulary cannot express — the region-inclusion argument does NOT hold for
//! them. Both entry points therefore fail closed via
//! `GraphSplitHistory::is_pure_relu_at_zero()` (O(1), with a `split_count`
//! catch-all for any split kind this code does not know about): an impure
//! history is neither recorded nor prune-checked.
//!
//! Run scope. The store lives inside ONE verification run (`verify_impl`
//! equivalent): one graph, one root input box, one objective semantics
//! (single-objective: one objective vector + threshold + sense;
//! multi-objective: one (objectives, thresholds, conjunctive) tuple). Nothing
//! crosses runs.
//!
//! Multi-objective closes. A clause may only be recorded from a domain closed
//! verified under the SAME objective semantics it would later prune for. In
//! the multi-objective CONJUNCTIVE lane a domain closes verified when ANY
//! objective clears — that single cleared objective proves the conjunction
//! impossible on the domain's ENTIRE region, i.e. the REGION is safe for the
//! whole conjunction check. Region-inclusion then remains valid for pruning:
//! a later same-run domain with a superset literal set covers a subregion of
//! that region, hence is also safe for the whole conjunction check and may be
//! closed verified. (Disjunctive lane: close = ALL objectives cleared on the
//! region; same argument a fortiori.)
//!
//! Infeasible closes. `#2926` infeasible = empty region = trivially verified:
//! emptiness is a property of the region itself (derived from the pure literal
//! set over the root box), and every superset literal set yields a subset of
//! the empty region — also empty, also trivially verified. Recording from
//! these closes is therefore sound under exactly the same inclusion argument.
//!
//! Forgetting is free. Subsumption-based insert and FIFO eviction only ever
//! FORGET clauses — strictly less pruning, never more — so the cap costs no
//! soundness.
//!
//! Gate: same env as v1, `NY_BAB_CLAUSE_LEARN=1` (default OFF => the graph
//! loops are byte-identical to baseline: the disabled store no-ops both entry
//! points). Cap: `NY_BAB_CLAUSE_CAP` (default 10_000).

use std::collections::VecDeque;

use tracing::trace;

use super::branching::GraphSplitHistory;
use super::conflict_clauses::gate_enabled;

/// One graph activation literal: (relu node name, neuron_idx, phase).
/// `true` = fixed active (pre-activation >= 0), `false` = inactive (<= 0).
/// Split point is the constant 0 — a direct image of `GraphNeuronConstraint`
/// minus the score.
pub(crate) type GraphActLit = (String, usize, bool);

/// A conflict clause: the literal set of a verified-closed pure-ReLU domain,
/// stored sorted by (node_name, neuron_idx). A set, not a sequence — split
/// ORDER is irrelevant to the region, so canonical ordering makes subset
/// tests a linear sorted merge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphConflictClause {
    lits: Box<[GraphActLit]>,
}

impl GraphConflictClause {
    /// Build a clause from a PURE ReLU-at-0 split history (the caller must
    /// have checked `is_pure_relu_at_zero`; debug-asserted here). Returns
    /// `None` for an empty history — the empty clause would prune everything,
    /// and a verified root ends the run anyway, so it is never useful and
    /// never sound to record.
    fn from_pure_history(history: &GraphSplitHistory) -> Option<Self> {
        debug_assert!(
            history.is_pure_relu_at_zero(),
            "from_pure_history requires a purity-checked history"
        );
        if history.constraints.is_empty() {
            return None;
        }
        let mut lits: Vec<GraphActLit> = history
            .constraints
            .iter()
            .map(|c| (c.node_name.clone(), c.neuron_idx, c.is_active))
            .collect();
        lits.sort_unstable_by(|a, b| (a.0.as_str(), a.1).cmp(&(b.0.as_str(), b.1)));
        // Engine invariant: a neuron is split at most once per path
        // (`is_any_constrained` guard), so (node, neuron) keys are unique.
        // Dedup defensively anyway; exact duplicates collapse harmlessly.
        lits.dedup();
        Some(Self {
            lits: lits.into_boxed_slice(),
        })
    }

    /// True iff `self`'s literal set is a subset of `other`'s.
    /// Both sides are sorted by (node, neuron); linear merge.
    fn is_subset_of(&self, other: &GraphConflictClause) -> bool {
        if self.lits.len() > other.lits.len() {
            return false;
        }
        let mut oi = 0usize;
        'outer: for (node, idx, phase) in self.lits.iter() {
            while oi < other.lits.len() {
                let (onode, oidx, ophase) = &other.lits[oi];
                match (onode.as_str(), *oidx).cmp(&(node.as_str(), *idx)) {
                    std::cmp::Ordering::Less => oi += 1,
                    std::cmp::Ordering::Equal => {
                        if ophase != phase {
                            // Same neuron, opposite phase: not a subset.
                            return false;
                        }
                        oi += 1;
                        continue 'outer;
                    }
                    std::cmp::Ordering::Greater => return false,
                }
            }
            return false;
        }
        true
    }

    /// True iff every literal of this clause is fixed with the SAME phase in
    /// `history` — i.e., clause ⊆ L(D'). Uses the O(1) per-literal
    /// `GraphSplitHistory::is_constrained` lookup; phase must match exactly.
    fn is_satisfied_by(&self, history: &GraphSplitHistory) -> bool {
        if self.lits.len() > history.constraints.len() {
            return false;
        }
        self.lits
            .iter()
            .all(|(node, idx, phase)| history.is_constrained(node, *idx) == Some(*phase))
    }
}

const DEFAULT_CLAUSE_CAP: usize = 10_000;

/// Per-run graph conflict clause store with subsumption and a FIFO cap.
///
/// Both entry points fail closed: a disabled store, or any domain whose
/// history is not pure ReLU-at-0, no-ops (`record_verified_close`) / returns
/// false (`should_prune`).
#[derive(Debug)]
pub(crate) struct GraphClauseStore {
    enabled: bool,
    cap: usize,
    clauses: VecDeque<GraphConflictClause>,
    /// Domains closed as verified via clause subsumption WITHOUT a bound
    /// computation (observability; logged by the owning loop).
    pruned: usize,
}

impl GraphClauseStore {
    /// A permanently disabled store (the gate-off state).
    pub(crate) fn disabled() -> Self {
        Self {
            enabled: false,
            cap: 0,
            clauses: VecDeque::new(),
            pruned: 0,
        }
    }

    /// Explicit constructor (tests).
    pub(crate) fn with_capacity(enabled: bool, cap: usize) -> Self {
        Self {
            enabled,
            cap: cap.max(1),
            clauses: VecDeque::new(),
            pruned: 0,
        }
    }

    /// Build the per-run store from the shared env gate
    /// (`NY_BAB_CLAUSE_LEARN=1`; default OFF => disabled => byte-identical
    /// baseline). Cap from `NY_BAB_CLAUSE_CAP` (default 10_000).
    pub(crate) fn from_env() -> Self {
        if !gate_enabled() {
            return Self::disabled();
        }
        let cap = std::env::var("NY_BAB_CLAUSE_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&c| c > 0)
            .unwrap_or(DEFAULT_CLAUSE_CAP);
        Self::with_capacity(true, cap)
    }

    /// Whether clause learning is active for this run.
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Domains clause-pruned so far (observability).
    pub(crate) fn pruned_count(&self) -> usize {
        self.pruned
    }

    /// Number of stored clauses (test observability).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.clauses.len()
    }

    /// Record the literal set of a domain the engine just closed as VERIFIED
    /// (sound bound proves the property on its whole region, or the region is
    /// proven empty — #2926 infeasible = trivially verified; see module doc).
    ///
    /// PURITY GUARD (fail closed): no-op unless the domain's ENTIRE history is
    /// pure ReLU-at-0 literals — any GenBaB / norm / unknown split entry means
    /// the literal set does not determine the region and the certificate would
    /// over-claim. Also no-ops when disabled and for empty histories.
    pub(crate) fn record_verified_close(&mut self, history: &GraphSplitHistory) {
        if !self.enabled || !history.is_pure_relu_at_zero() {
            return;
        }
        let Some(clause) = GraphConflictClause::from_pure_history(history) else {
            return;
        };
        // (a) If an existing clause is a subset of the new one, the new clause
        // is redundant: anything it would prune, the existing clause already
        // prunes.
        if self.clauses.iter().any(|c0| c0.is_subset_of(&clause)) {
            return;
        }
        // (b) Remove existing clauses that are supersets of the new one — they
        // prune strictly less than the new clause.
        self.clauses.retain(|c0| !clause.is_subset_of(c0));
        // FIFO cap: evict oldest. Eviction only loses pruning power, never
        // soundness.
        while self.clauses.len() >= self.cap {
            self.clauses.pop_front();
        }
        self.clauses.push_back(clause);
    }

    /// True iff a popped domain may be closed as verified without computing
    /// bounds: some recorded clause C satisfies C ⊆ L(domain), hence the
    /// domain's region is a subset of an already-certified region (same run,
    /// same root box, same objective semantics by construction — see module
    /// doc). Increments the pruned counter on a hit.
    ///
    /// PURITY GUARD (fail closed): returns false unless the domain's ENTIRE
    /// history is pure ReLU-at-0 literals — an impure history's region is
    /// narrowed by constraints the literal set does not capture on the RECORD
    /// side only; on the PRUNE side the extra constraints would only shrink
    /// the region further, but we fail closed anyway rather than reason about
    /// mixed semantics. Also false for disabled/empty stores.
    pub(crate) fn should_prune(&mut self, history: &GraphSplitHistory) -> bool {
        if !self.enabled || self.clauses.is_empty() || !history.is_pure_relu_at_zero() {
            return false;
        }
        let hit = self.clauses.iter().any(|c| c.is_satisfied_by(history));
        if hit {
            self.pruned += 1;
            trace!(
                depth = history.depth(),
                "Graph domain clause-pruned: subsumed by recorded conflict clause"
            );
        }
        hit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::branching::{
        GenBabConstraint, GraphNeuronConstraint, NormInvRmsConstraint,
    };
    use crate::beta_crown::conflict_clauses::set_test_gate_override;
    use std::collections::HashSet;

    type Lit<'a> = (&'a str, usize, bool);

    fn history_of(lits: &[Lit<'_>]) -> GraphSplitHistory {
        let mut h = GraphSplitHistory::new();
        for &(node, idx, phase) in lits {
            h.add_constraint(GraphNeuronConstraint {
                node_name: node.to_string(),
                neuron_idx: idx,
                is_active: phase,
                score: 1.0,
            });
        }
        h
    }

    fn clause_of(lits: &[Lit<'_>]) -> GraphConflictClause {
        GraphConflictClause::from_pure_history(&history_of(lits)).expect("non-empty clause")
    }

    fn genbab_constraint() -> GenBabConstraint {
        GenBabConstraint::new("gelu1".to_string(), 0, 0.25, true, 1.0)
            .expect("valid GenBaB constraint")
    }

    fn norm_constraint() -> NormInvRmsConstraint {
        NormInvRmsConstraint::new("rms1".to_string(), 0, 0.5, 2.0, 1.0)
            .expect("valid norm inv_rms constraint")
    }

    // ── ORACLE 1: purity guard ─────────────────────────────────────
    // A history containing one GenBaB (or norm inv_rms, or unknown-kind)
    // constraint must NEVER record NOR prune, even when its ReLU literal
    // subset would match. Discriminates against an implementation that reads
    // only `history.constraints`.

    #[test]
    fn purity_guard_genbab_history_never_records_nor_prunes() {
        let mut store = GraphClauseStore::with_capacity(true, 100);

        // Record side: verified close of a mixed ReLU+GenBaB history must not
        // produce a clause (the GenBaB split narrowed the region beyond what
        // the ReLU literals say — the clause would over-claim).
        let mut mixed = history_of(&[("relu1", 0, true), ("relu1", 1, false)]);
        mixed.add_genbab_constraint(genbab_constraint());
        assert!(!mixed.is_pure_relu_at_zero());
        store.record_verified_close(&mixed);
        assert_eq!(store.len(), 0, "GenBaB history must never be recorded");

        // Prune side: with a legitimately recorded pure clause, a mixed
        // history whose ReLU literals are a superset must NOT be pruned.
        store.record_verified_close(&history_of(&[("relu1", 0, true)]));
        assert_eq!(store.len(), 1);
        let mut superset_mixed = history_of(&[("relu1", 0, true), ("relu1", 1, false)]);
        superset_mixed.add_genbab_constraint(genbab_constraint());
        assert!(
            !store.should_prune(&superset_mixed),
            "GenBaB history must fail closed at the prune check"
        );
        // Discriminator: the same ReLU literal set WITHOUT the GenBaB entry
        // IS pruned — proving the guard (not the literals) blocked it above.
        assert!(store.should_prune(&history_of(&[("relu1", 0, true), ("relu1", 1, false)])));
        assert_eq!(store.pruned_count(), 1);
    }

    #[test]
    fn purity_guard_norm_inv_rms_history_never_records_nor_prunes() {
        let mut store = GraphClauseStore::with_capacity(true, 100);

        let mut mixed = history_of(&[("relu1", 0, true)]);
        mixed.add_norm_inv_rms_constraint(norm_constraint());
        assert!(!mixed.is_pure_relu_at_zero());
        store.record_verified_close(&mixed);
        assert_eq!(
            store.len(),
            0,
            "norm inv_rms history must never be recorded"
        );

        store.record_verified_close(&history_of(&[("relu1", 0, true)]));
        let mut superset_mixed = history_of(&[("relu1", 0, true), ("relu2", 3, false)]);
        superset_mixed.add_norm_inv_rms_constraint(norm_constraint());
        assert!(
            !store.should_prune(&superset_mixed),
            "norm inv_rms history must fail closed at the prune check"
        );
    }

    /// The `split_count == constraints.len()` catch-all: a history touched by
    /// a split kind this module does not know about (simulated by bumping
    /// `split_count` without a matching ReLU constraint) fails closed on both
    /// entry points.
    #[test]
    fn purity_guard_unknown_split_kind_fails_closed_via_split_count() {
        let mut store = GraphClauseStore::with_capacity(true, 100);

        let mut unknown = history_of(&[("relu1", 0, true), ("relu1", 1, true)]);
        unknown.split_count += 1; // a future non-ReLU split kind
        assert!(!unknown.is_pure_relu_at_zero());
        store.record_verified_close(&unknown);
        assert_eq!(store.len(), 0, "unknown split kind must never be recorded");

        store.record_verified_close(&history_of(&[("relu1", 0, true)]));
        let mut superset_unknown = history_of(&[("relu1", 0, true), ("relu1", 1, true)]);
        superset_unknown.split_count += 1;
        assert!(
            !store.should_prune(&superset_unknown),
            "unknown split kind must fail closed at the prune check"
        );
    }

    // ── ORACLE 2: record + prune round trip on a pure-ReLU tree ────

    #[test]
    fn record_prune_round_trip_prunes_superset_not_subset_not_phase_flip() {
        let mut store = GraphClauseStore::with_capacity(true, 100);
        // A verified close at depth 2 on a synthetic pure-ReLU path.
        store.record_verified_close(&history_of(&[("relu1", 0, true), ("relu1", 1, true)]));
        assert_eq!(store.len(), 1);

        // Superset of the clause: smaller region, covered — prune.
        assert!(store.should_prune(&history_of(&[
            ("relu1", 0, true),
            ("relu1", 1, true),
            ("relu2", 0, false),
        ])));
        // Subset of the clause: LARGER region, NOT covered — must not prune
        // (kills the reversed-subsumption bug).
        assert!(!store.should_prune(&history_of(&[("relu1", 0, true)])));
        // Phase flip on one literal: disjoint region — must not prune
        // (kills the phase-ignoring bug).
        assert!(!store.should_prune(&history_of(&[
            ("relu1", 0, true),
            ("relu1", 1, false),
            ("relu2", 0, false),
        ])));
        // Same neuron INDEX on a different NODE: different literal — must not
        // prune (kills a node-name-ignoring bug the sequential engine cannot
        // have).
        assert!(!store.should_prune(&history_of(&[
            ("relu1", 0, true),
            ("relu2", 1, true),
            ("relu2", 0, false),
        ])));
        assert_eq!(store.pruned_count(), 1);
    }

    // ── ORACLE 3 (store level): gate off => inert ──────────────────
    // (Loop-level A/B verdict equivalence lives in
    // engine/tests/graph_clause_learning.rs.)

    #[test]
    fn from_env_defaults_off_and_disabled_store_is_inert() {
        set_test_gate_override(Some(false));
        let mut store = GraphClauseStore::from_env();
        assert!(!store.is_enabled(), "gate off must yield a disabled store");
        store.record_verified_close(&history_of(&[("relu1", 0, true)]));
        assert_eq!(store.len(), 0, "gate off: nothing recorded");
        assert!(
            !store.should_prune(&history_of(&[("relu1", 0, true), ("relu1", 1, true)])),
            "gate off: nothing pruned"
        );
        assert_eq!(store.pruned_count(), 0);

        set_test_gate_override(Some(true));
        assert!(GraphClauseStore::from_env().is_enabled());
        set_test_gate_override(None);
    }

    // ── subset test vs brute force (node-name literal space) ───────

    /// Deterministic xorshift64* RNG — no external dep.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn subset_test_matches_brute_force_on_random_sets() {
        const NODES: [&str; 2] = ["relu_a", "relu_b"];
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..2000 {
            let mut gen_lits = |max_len: u64| {
                let len = rng.below(max_len + 1) as usize;
                let mut seen = HashSet::new();
                let mut lits: Vec<(String, usize, bool)> = Vec::new();
                for _ in 0..len {
                    let node = NODES[rng.below(2) as usize];
                    let idx = rng.below(4) as usize;
                    if seen.insert((node, idx)) {
                        lits.push((node.to_string(), idx, rng.below(2) == 0));
                    }
                }
                lits
            };
            let a = gen_lits(5);
            let b = gen_lits(5);
            if a.is_empty() || b.is_empty() {
                continue;
            }
            let a_refs: Vec<Lit<'_>> = a.iter().map(|(n, i, p)| (n.as_str(), *i, *p)).collect();
            let b_refs: Vec<Lit<'_>> = b.iter().map(|(n, i, p)| (n.as_str(), *i, *p)).collect();
            let ca = clause_of(&a_refs);
            let cb = clause_of(&b_refs);
            let sa: HashSet<&(String, usize, bool)> = a.iter().collect();
            let sb: HashSet<&(String, usize, bool)> = b.iter().collect();
            assert_eq!(
                ca.is_subset_of(&cb),
                sa.is_subset(&sb),
                "sorted-merge subset test disagrees with brute force: a={a:?} b={b:?}"
            );
            // is_satisfied_by must agree with the set-level subset relation.
            assert_eq!(
                ca.is_satisfied_by(&history_of(&b_refs)),
                sa.is_subset(&sb),
                "is_satisfied_by disagrees with brute force: a={a:?} b={b:?}"
            );
        }
    }

    // ── insert subsumption + FIFO cap + empty history ──────────────

    #[test]
    fn insert_refuses_subsumed_and_removes_supersets() {
        let mut store = GraphClauseStore::with_capacity(true, 100);
        store.record_verified_close(&history_of(&[("relu1", 0, true), ("relu1", 1, false)]));
        store.record_verified_close(&history_of(&[("relu1", 0, true), ("relu1", 2, true)]));
        assert_eq!(store.len(), 2);
        // Superset of an existing clause: redundant, refused.
        store.record_verified_close(&history_of(&[
            ("relu1", 0, true),
            ("relu1", 1, false),
            ("relu1", 3, true),
        ]));
        assert_eq!(store.len(), 2, "superset clause must be refused");
        // New clause subsumes both existing ones: they are dropped.
        store.record_verified_close(&history_of(&[("relu1", 0, true)]));
        assert_eq!(store.len(), 1, "existing supersets must be removed");
        assert!(store.should_prune(&history_of(&[("relu1", 0, true), ("relu1", 1, false)])));
    }

    #[test]
    fn fifo_cap_evicts_oldest_first() {
        let mut store = GraphClauseStore::with_capacity(true, 2);
        store.record_verified_close(&history_of(&[("relu1", 0, true)]));
        store.record_verified_close(&history_of(&[("relu1", 1, true)]));
        store.record_verified_close(&history_of(&[("relu1", 2, true)]));
        assert_eq!(store.len(), 2, "cap must bound the store");
        assert!(
            !store.should_prune(&history_of(&[("relu1", 0, true), ("relu2", 0, true)])),
            "oldest clause must have been evicted"
        );
        assert!(store.should_prune(&history_of(&[("relu1", 1, true), ("relu2", 0, true)])));
    }

    #[test]
    fn empty_history_is_never_recorded_and_prunes_nothing() {
        let mut store = GraphClauseStore::with_capacity(true, 100);
        store.record_verified_close(&GraphSplitHistory::new());
        assert_eq!(store.len(), 0, "empty clause must be refused");
        assert!(!store.should_prune(&history_of(&[("relu1", 0, true)])));
    }
}
