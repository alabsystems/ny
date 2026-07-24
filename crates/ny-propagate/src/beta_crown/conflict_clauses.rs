// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conflict-clause learning for sequential ReLU-split BaB (win-plan arc C, v1).
//!
//! When a domain is closed as verified (its sound bound proves the property on
//! that subregion), its activation literal set L(D) — the ReLU split path,
//! `SplitHistory.constraints` — is recorded as a conflict clause. Any later
//! domain D' in the SAME `verify_impl` run whose literal set is a superset of a
//! recorded clause covers a subregion of the already-certified region (more
//! fixations = smaller region, pure set intersection over fixed half-spaces
//! `z_{l,n}(x) >= 0` / `<= 0` at split point 0), so it may be closed as
//! verified WITHOUT computing bounds. No dual/proof analysis is trusted; the
//! only input is the engine's existing "domain verified" event.
//!
//! Soundness scope enforced BY CONSTRUCTION:
//! - The store lives inside one `verify_impl` call: one network, one root box,
//!   one threshold, one objective sense. Nothing crosses runs.
//! - Input-split domains are excluded fail-closed at BOTH entry points (their
//!   region depends on a private input sub-box not captured by the literals):
//!   any domain with `input_split_count > 0 || input_bounds.is_some()` is
//!   neither recorded nor prune-checked, and the store is disabled outright
//!   under the InputSplit branching heuristic.
//! - FIFO eviction and subsumption-based insert only ever FORGET clauses —
//!   strictly less pruning, never more — so the cap costs no soundness.
//!
//! Gated by `NY_BAB_CLAUSE_LEARN=1` (default OFF => byte-identical baseline,
//! kfsb measure-before-enable discipline). Clause cap: `NY_BAB_CLAUSE_CAP`
//! (default 10_000).
//!
//! EXPLICITLY OUT OF SCOPE in v1 (see design doc): beta-based clause
//! minimization (UNSOUND without re-verification — the relaxation changes with
//! a fixation even at beta=0), cross-run sharing, and recording from
//! infeasible-child closes. Graph-engine histories are handled by the v2 port
//! in `conflict_clauses_graph` behind a purity guard (pure ReLU-at-0 literal
//! paths only; mixed GenBaB/norm constraints fail closed there).

use std::collections::VecDeque;

use super::branching::SplitHistory;
use super::domain::BabDomain;

/// One activation literal: (relu layer_idx, neuron_idx, phase).
/// `is_active == true` means the neuron is fixed to the active phase
/// (pre-activation >= 0); `false` means inactive (<= 0). Split point is the
/// constant 0 — a direct image of `NeuronConstraint` minus the score.
pub(crate) type ActLit = (usize, usize, bool);

/// A conflict clause: the literal set of a verified (refuted) domain, stored
/// sorted by (layer_idx, neuron_idx). A set, not a sequence — split ORDER is
/// irrelevant to the region, so canonical ordering makes subset tests a linear
/// sorted merge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConflictClause {
    lits: Box<[ActLit]>,
}

impl ConflictClause {
    /// Build a clause from a split history. Returns `None` for an empty
    /// history (the empty clause would prune everything; a verified root ends
    /// the run anyway, so it is never useful and never sound to record).
    fn from_history(history: &SplitHistory) -> Option<Self> {
        if history.constraints.is_empty() {
            return None;
        }
        let mut lits: Vec<ActLit> = history
            .constraints
            .iter()
            .map(|c| (c.layer_idx(), c.neuron_idx(), c.is_active()))
            .collect();
        lits.sort_unstable_by_key(|&(l, n, _)| (l, n));
        // Engine invariant: a neuron is split at most once per path
        // (`is_constrained` guard), so (layer, neuron) keys are unique. Dedup
        // defensively anyway; exact duplicates collapse harmlessly.
        lits.dedup();
        Some(Self {
            lits: lits.into_boxed_slice(),
        })
    }

    /// True iff `self`'s literal set is a subset of `other`'s.
    /// Both sides are sorted by (layer, neuron); linear merge.
    fn is_subset_of(&self, other: &ConflictClause) -> bool {
        if self.lits.len() > other.lits.len() {
            return false;
        }
        let mut oi = 0usize;
        'outer: for &(l, n, p) in self.lits.iter() {
            while oi < other.lits.len() {
                let (ol, on, op) = other.lits[oi];
                match (ol, on).cmp(&(l, n)) {
                    std::cmp::Ordering::Less => oi += 1,
                    std::cmp::Ordering::Equal => {
                        if op != p {
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
    /// `SplitHistory::is_constrained` lookup; phase must match exactly.
    fn is_satisfied_by(&self, history: &SplitHistory) -> bool {
        if self.lits.len() > history.constraints.len() {
            return false;
        }
        self.lits
            .iter()
            .all(|&(l, n, p)| history.is_constrained(l, n) == Some(p))
    }
}

/// Returns true iff the clause-learning env gate is set (`NY_BAB_CLAUSE_LEARN=1`).
/// Default OFF: unset (or any other value) leaves the baseline byte-identical.
/// Shared with the graph-engine port (`conflict_clauses_graph`) so both lanes
/// ride one gate (and one test override).
pub(crate) fn gate_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_GATE_OVERRIDE.with(std::cell::Cell::get) {
        return v;
    }
    std::env::var("NY_BAB_CLAUSE_LEARN").as_deref() == Ok("1")
}

#[cfg(test)]
thread_local! {
    /// Test-only, thread-local gate override so parallel tests never leak the
    /// gate to each other via process-global env vars. `verify_impl` reads the
    /// gate on the calling thread, so setting this on the test thread is
    /// sufficient.
    static TEST_GATE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only: override the `NY_BAB_CLAUSE_LEARN` gate for the current thread.
/// Pass `None` to restore env-based behavior.
#[cfg(test)]
pub(crate) fn set_test_gate_override(v: Option<bool>) {
    TEST_GATE_OVERRIDE.with(|o| o.set(v));
}

const DEFAULT_CLAUSE_CAP: usize = 10_000;

/// Per-BaB-run conflict clause store with subsumption and a FIFO cap.
///
/// Both entry points fail closed: a disabled store, or any domain carrying
/// input-split evidence, no-ops (`record_verified_domain`) / returns false
/// (`should_prune`).
#[derive(Debug)]
pub(crate) struct ClauseStore {
    enabled: bool,
    cap: usize,
    clauses: VecDeque<ConflictClause>,
}

impl ClauseStore {
    /// A permanently disabled store (the default in `BabLoopState`; also the
    /// state when the env gate is unset or the heuristic is InputSplit).
    pub(crate) fn disabled() -> Self {
        Self {
            enabled: false,
            cap: 0,
            clauses: VecDeque::new(),
        }
    }

    /// Explicit constructor (tests and `from_env`).
    pub(crate) fn with_capacity(enabled: bool, cap: usize) -> Self {
        Self {
            enabled,
            cap: cap.max(1),
            clauses: VecDeque::new(),
        }
    }

    /// Build the per-run store from the env gate. `heuristic_allows` must be
    /// false for the InputSplit branching heuristic (input-split trees are out
    /// of scope; the per-domain guard below backstops mixed trees regardless).
    pub(crate) fn from_env(heuristic_allows: bool) -> Self {
        let enabled = heuristic_allows && gate_enabled();
        if !enabled {
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

    /// Number of stored clauses (test observability).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.clauses.len()
    }

    /// Per-domain fail-closed guard: input-split domains carry a private input
    /// sub-box (`input_bounds`) not captured by the activation literals, so
    /// their certificates over-claim the full-box region (record side) and
    /// their regions are not covered by full-box certificates (prune side).
    fn domain_excluded(domain: &BabDomain) -> bool {
        domain.input_split_count() > 0 || domain.input_bounds_arc().is_some()
    }

    /// Record the literal set of a domain that the engine just closed as
    /// VERIFIED (its sound bound proves the property on its whole region).
    /// No-op when disabled, for input-split domains, and for empty histories.
    pub(crate) fn record_verified_domain(&mut self, domain: &BabDomain) {
        if !self.enabled || Self::domain_excluded(domain) {
            return;
        }
        let Some(clause) = ConflictClause::from_history(domain.history()) else {
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

    /// True iff `domain` may be closed as verified without computing bounds:
    /// some recorded clause C satisfies C ⊆ L(domain), hence the domain's
    /// region is a subset of an already-certified region (same run, same root
    /// box, same threshold by construction). Fails closed for disabled stores
    /// and input-split domains.
    pub(crate) fn should_prune(&self, domain: &BabDomain) -> bool {
        if !self.enabled || self.clauses.is_empty() || Self::domain_excluded(domain) {
            return false;
        }
        let history = domain.history();
        self.clauses.iter().any(|c| c.is_satisfied_by(history))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::branching::NeuronConstraint;
    use crate::beta_crown::{BetaState, DomainAlphaState};
    use crate::{BoundedTensor, IntermediateLinearBounds};
    use ndarray::{ArrayD, IxDyn};
    use std::collections::HashSet;
    use std::sync::Arc;

    fn history_of(lits: &[ActLit]) -> SplitHistory {
        let mut h = SplitHistory::new();
        for &(l, n, p) in lits {
            h.add_constraint(NeuronConstraint::new(l, n, p, 1.0).unwrap());
        }
        h
    }

    fn clause_of(lits: &[ActLit]) -> ConflictClause {
        ConflictClause::from_history(&history_of(lits)).expect("non-empty clause")
    }

    fn layer_bounds_1() -> Vec<Arc<BoundedTensor>> {
        let t = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0f32]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0f32]).unwrap(),
        )
        .unwrap();
        vec![Arc::new(t)]
    }

    /// Pure ReLU-split domain with the given literal set.
    fn domain_of(lits: &[ActLit]) -> BabDomain {
        BabDomain::child(
            history_of(lits),
            0.0,
            1.0,
            layer_bounds_1(),
            None,
            DomainAlphaState::empty(),
            BetaState::empty(),
            None,
            0,
            IntermediateLinearBounds::empty(),
        )
        .unwrap()
    }

    /// Input-split-tainted domain with the given literal set.
    fn input_split_domain_of(lits: &[ActLit], via_count: bool) -> BabDomain {
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0f32]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0f32]).unwrap(),
        )
        .unwrap();
        BabDomain::child(
            history_of(lits),
            0.0,
            1.0,
            layer_bounds_1(),
            None,
            DomainAlphaState::empty(),
            BetaState::empty(),
            if via_count {
                None
            } else {
                Some(Arc::new(input))
            },
            if via_count { 1 } else { 0 },
            IntermediateLinearBounds::empty(),
        )
        .unwrap()
    }

    // ── subset test vs brute force ─────────────────────────────────

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

    fn random_lits(rng: &mut Rng, max_len: u64) -> Vec<ActLit> {
        // Small key space (2 layers x 4 neurons) to make subset relations and
        // phase collisions actually happen.
        let len = rng.below(max_len + 1) as usize;
        let mut seen = HashSet::new();
        let mut lits = Vec::new();
        for _ in 0..len {
            let l = rng.below(2) as usize;
            let n = rng.below(4) as usize;
            if seen.insert((l, n)) {
                lits.push((l, n, rng.below(2) == 0));
            }
        }
        lits
    }

    #[test]
    fn subset_test_matches_brute_force_on_random_sets() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..2000 {
            let a = random_lits(&mut rng, 5);
            let b = random_lits(&mut rng, 5);
            if a.is_empty() || b.is_empty() {
                continue;
            }
            let ca = clause_of(&a);
            let cb = clause_of(&b);
            let sa: HashSet<ActLit> = a.iter().copied().collect();
            let sb: HashSet<ActLit> = b.iter().copied().collect();
            assert_eq!(
                ca.is_subset_of(&cb),
                sa.is_subset(&sb),
                "sorted-merge subset test disagrees with brute force: a={a:?} b={b:?}"
            );
            // is_satisfied_by must agree with the set-level subset relation.
            assert_eq!(
                ca.is_satisfied_by(&history_of(&b)),
                sa.is_subset(&sb),
                "is_satisfied_by disagrees with brute force: a={a:?} b={b:?}"
            );
        }
    }

    #[test]
    fn subset_test_is_phase_sensitive() {
        let c = clause_of(&[(0, 0, true)]);
        let opposite = clause_of(&[(0, 0, false), (0, 1, true)]);
        assert!(
            !c.is_subset_of(&opposite),
            "same neuron with opposite phase must not count as subset"
        );
        assert!(!c.is_satisfied_by(&history_of(&[(0, 0, false), (0, 1, true)])));
    }

    // ── insert subsumption ─────────────────────────────────────────

    #[test]
    fn insert_refuses_clause_subsumed_by_existing() {
        let mut store = ClauseStore::with_capacity(true, 100);
        store.record_verified_domain(&domain_of(&[(0, 0, true)]));
        assert_eq!(store.len(), 1);
        // Superset of an existing clause: redundant, refused.
        store.record_verified_domain(&domain_of(&[(0, 0, true), (0, 1, false)]));
        assert_eq!(store.len(), 1, "superset clause must be refused");
        // Exact duplicate: also refused.
        store.record_verified_domain(&domain_of(&[(0, 0, true)]));
        assert_eq!(store.len(), 1, "duplicate clause must be refused");
    }

    #[test]
    fn insert_removes_existing_supersets() {
        let mut store = ClauseStore::with_capacity(true, 100);
        store.record_verified_domain(&domain_of(&[(0, 0, true), (0, 1, false)]));
        store.record_verified_domain(&domain_of(&[(0, 0, true), (0, 2, true)]));
        assert_eq!(store.len(), 2);
        // New clause subsumes both existing ones: they are dropped.
        store.record_verified_domain(&domain_of(&[(0, 0, true)]));
        assert_eq!(
            store.len(),
            1,
            "existing supersets must be removed on insert of a subsuming clause"
        );
        // The surviving clause is the short one: it prunes what the old ones did.
        assert!(store.should_prune(&domain_of(&[(0, 0, true), (0, 1, false)])));
        assert!(store.should_prune(&domain_of(&[(0, 0, true), (0, 2, true)])));
    }

    // ── FIFO cap ───────────────────────────────────────────────────

    #[test]
    fn fifo_cap_evicts_oldest_first() {
        let mut store = ClauseStore::with_capacity(true, 2);
        // Three mutually non-subsuming clauses (disjoint neurons).
        store.record_verified_domain(&domain_of(&[(0, 0, true)]));
        store.record_verified_domain(&domain_of(&[(0, 1, true)]));
        store.record_verified_domain(&domain_of(&[(0, 2, true)]));
        assert_eq!(store.len(), 2, "cap must bound the store");
        assert!(
            !store.should_prune(&domain_of(&[(0, 0, true), (1, 0, true)])),
            "oldest clause must have been evicted"
        );
        assert!(store.should_prune(&domain_of(&[(0, 1, true), (1, 0, true)])));
        assert!(store.should_prune(&domain_of(&[(0, 2, true), (1, 0, true)])));
    }

    // ── empty clause / root ────────────────────────────────────────

    #[test]
    fn empty_history_is_never_recorded() {
        let mut store = ClauseStore::with_capacity(true, 100);
        store.record_verified_domain(&domain_of(&[]));
        assert_eq!(store.len(), 0, "empty clause must be refused");
        assert!(
            !store.should_prune(&domain_of(&[(0, 0, true)])),
            "empty store must prune nothing"
        );
    }

    // ── disabled store ─────────────────────────────────────────────

    #[test]
    fn disabled_store_is_inert() {
        let mut store = ClauseStore::disabled();
        store.record_verified_domain(&domain_of(&[(0, 0, true)]));
        assert_eq!(store.len(), 0);
        assert!(!store.should_prune(&domain_of(&[(0, 0, true), (0, 1, true)])));
    }

    #[test]
    fn from_env_defaults_off() {
        set_test_gate_override(Some(false));
        assert!(!ClauseStore::from_env(true).is_enabled());
        set_test_gate_override(Some(true));
        assert!(ClauseStore::from_env(true).is_enabled());
        // InputSplit heuristic disables the store even with the gate on.
        assert!(!ClauseStore::from_env(false).is_enabled());
        set_test_gate_override(None);
    }

    // ── prune semantics ────────────────────────────────────────────

    #[test]
    fn prunes_superset_not_subset_not_phase_flip() {
        let mut store = ClauseStore::with_capacity(true, 100);
        store.record_verified_domain(&domain_of(&[(0, 0, true), (0, 1, true)]));

        // Superset of the clause: smaller region, covered — prune.
        assert!(store.should_prune(&domain_of(&[(0, 0, true), (0, 1, true), (1, 0, false)])));
        // Subset of the clause: LARGER region, NOT covered — must not prune
        // (kills the reversed-subsumption bug).
        assert!(!store.should_prune(&domain_of(&[(0, 0, true)])));
        // Phase flip on one literal: disjoint region — must not prune
        // (kills the phase-ignoring bug).
        assert!(!store.should_prune(&domain_of(&[(0, 0, true), (0, 1, false), (1, 0, false)])));
    }

    // ── input-split firewall ───────────────────────────────────────

    #[test]
    fn input_split_domains_fail_closed_both_ways() {
        let mut store = ClauseStore::with_capacity(true, 100);

        // Record side: an input-split verified domain must not produce a clause
        // (its certificate covers only its private sub-box).
        store.record_verified_domain(&input_split_domain_of(&[(0, 0, true)], true));
        store.record_verified_domain(&input_split_domain_of(&[(0, 1, true)], false));
        assert_eq!(store.len(), 0, "input-split domains must never be recorded");

        // Prune side: even with a legitimately recorded clause, an input-split
        // domain must not be prune-checked (its region depends on its sub-box).
        store.record_verified_domain(&domain_of(&[(0, 0, true)]));
        assert_eq!(store.len(), 1);
        assert!(
            !store.should_prune(&input_split_domain_of(&[(0, 0, true), (0, 1, true)], true)),
            "input_split_count > 0 must fail closed"
        );
        assert!(
            !store.should_prune(&input_split_domain_of(&[(0, 0, true), (0, 1, true)], false)),
            "input_bounds.is_some() must fail closed"
        );
    }
}
