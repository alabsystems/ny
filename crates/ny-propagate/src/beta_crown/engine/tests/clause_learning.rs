// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Discriminating oracles for BaB conflict-clause learning (win-plan arc C, v1).
//!
//! The prune rule is pure region inclusion: a recorded clause C (literal set of
//! a verified domain) may close a later domain D' iff C ⊆ L(D') — more
//! fixations = smaller region, already certified. These tests drive the REAL
//! record site (`prefilter_domain_batch` / `process_batch_children` verified
//! closes) and the REAL prune site (`prefilter_domain_batch` pop check) with
//! hand-built domains that re-encounter a subsumed assignment, and kill the
//! classic wrong implementations:
//!   - phase-ignoring subset test (matches (layer, neuron) but not phase),
//!   - reversed subsumption direction (pruning a SUBSET, i.e. a larger region),
//!   - input-split leak (recording/pruning domains whose region depends on a
//!     private input sub-box).
//!   - end-to-end gate ON vs OFF verdict equivalence on a real verify() run.

use super::prelude::*;
use crate::beta_crown::conflict_clauses::{set_test_gate_override, ClauseStore};
use crate::beta_crown::engine::verify_phases::BabLoopState;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type Lit = (usize, usize, bool);

fn history_of(lits: &[Lit]) -> SplitHistory {
    let mut h = SplitHistory::new();
    for &(l, n, p) in lits {
        h.add_constraint(NeuronConstraint::new(l, n, p, 1.0).unwrap());
    }
    h
}

fn layer_bounds_1() -> Vec<Arc<BoundedTensor>> {
    let t = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0f32]).unwrap(),
    )
    .unwrap();
    vec![Arc::new(t)]
}

/// Pure ReLU-split domain with the given literal set and output bounds.
fn domain_of(lits: &[Lit], lb: f32, ub: f32) -> BabDomain {
    BabDomain::child(
        history_of(lits),
        lb,
        ub,
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

/// Same, but carrying input-split evidence (`input_bounds` set).
fn input_split_domain_of(lits: &[Lit], lb: f32, ub: f32) -> BabDomain {
    let input =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();
    BabDomain::child(
        history_of(lits),
        lb,
        ub,
        layer_bounds_1(),
        None,
        DomainAlphaState::empty(),
        BetaState::empty(),
        Some(Arc::new(input)),
        0,
        IntermediateLinearBounds::empty(),
    )
    .unwrap()
}

struct Harness {
    verifier: BetaCrownVerifier,
    network: Network,
    input: BoundedTensor,
    cut_pool: CutPool,
}

impl Harness {
    fn new() -> Self {
        // enable_cuts stays default-false so the verified-close cut branch is
        // inert and the network/input/base_layer_bounds args are never touched.
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let network = simple_network();
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();
        Self {
            verifier,
            network,
            input,
            cut_pool: CutPool::new(0),
        }
    }

    /// Drive the real prefilter (record site + prune site) with one batch.
    fn prefilter(
        &mut self,
        batch: Vec<BabDomain>,
        threshold: f32,
        state: &mut BabLoopState,
    ) -> Vec<BabDomain> {
        let outcome = self
            .verifier
            .prefilter_domain_batch(
                batch,
                threshold,
                state,
                &mut self.cut_pool,
                &self.network,
                &self.input,
                &[],
                None,
                Instant::now(),
            )
            .expect("prefilter must succeed");
        assert!(outcome.violation.is_none(), "no violation expected");
        outcome.domains_to_process
    }
}

const THRESHOLD: f32 = 5.0;
// lb > THRESHOLD => verified close (default verify_upper_bound = false).
const VERIFIED: (f32, f32) = (10.0, 20.0);
// lb <= THRESHOLD <= ub => neither verified nor violated: must be processed.
const UNDECIDED: (f32, f32) = (0.0, 10.0);

// ---------------------------------------------------------------------------
// Discriminating oracle: record on refute, prune exactly the supersets
// ---------------------------------------------------------------------------

#[test]
fn test_records_verified_domain_then_prunes_only_true_supersets() {
    let mut h = Harness::new();
    let mut state = BabLoopState::new(false);
    state.clause_store = ClauseStore::with_capacity(true, 100);

    // Batch 1: domain with L(D) = {(0,0,+), (0,1,+)} closes verified => its
    // literal set is recorded as a conflict clause.
    let clause = [(0, 0, true), (0, 1, true)];
    let to_process = h.prefilter(
        vec![domain_of(&clause, VERIFIED.0, VERIFIED.1)],
        THRESHOLD,
        &mut state,
    );
    assert!(to_process.is_empty());
    assert_eq!(state.domains_verified, 1);
    assert_eq!(
        state.clause_store.len(),
        1,
        "verified close must record L(D)"
    );
    assert_eq!(state.domains_clause_pruned, 0);

    // Batch 2: a later re-encounter of the subsumed assignment. Only the true
    // superset may be pruned.
    let superset = [(0, 0, true), (0, 1, true), (1, 0, false)];
    let phase_flip = [(0, 0, true), (0, 1, false), (1, 0, false)];
    let subset = [(0, 0, true)];
    let to_process = h.prefilter(
        vec![
            domain_of(&superset, UNDECIDED.0, UNDECIDED.1),
            domain_of(&phase_flip, UNDECIDED.0, UNDECIDED.1),
            domain_of(&subset, UNDECIDED.0, UNDECIDED.1),
        ],
        THRESHOLD,
        &mut state,
    );

    assert_eq!(
        state.domains_clause_pruned, 1,
        "exactly the superset domain must be clause-pruned without bound work"
    );
    assert_eq!(
        state.domains_verified, 2,
        "the pruned domain counts as verified (it IS proven safe)"
    );
    assert_eq!(to_process.len(), 2, "phase-flip and subset must survive");
    for d in &to_process {
        let hist = d.history();
        let is_superset = clause
            .iter()
            .all(|&(l, n, p)| hist.is_constrained(l, n) == Some(p));
        assert!(
            !is_superset,
            "a surviving domain must NOT contain the recorded clause: {:?}",
            hist.constraints
        );
    }
}

#[test]
fn test_child_verified_close_records_clause() {
    let mut h = Harness::new();
    let mut state = BabLoopState::new(false);
    state.clause_store = ClauseStore::with_capacity(true, 100);
    let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();

    let verified_child = domain_of(&[(0, 0, false)], VERIFIED.0, VERIFIED.1);
    let undecided_child = domain_of(&[(0, 1, true)], UNDECIDED.0, UNDECIDED.1);
    let verified_children = h
        .verifier
        .process_batch_children(
            vec![verified_child, undecided_child],
            THRESHOLD,
            &mut queue,
            &mut state,
            &mut h.cut_pool,
            &h.network,
            &h.input,
            &[],
            None,
        )
        .expect("process_batch_children must succeed");

    assert_eq!(verified_children, 1);
    assert_eq!(queue.len(), 1, "undecided child must be requeued");
    assert_eq!(
        state.clause_store.len(),
        1,
        "child verified-close must record its literal set"
    );

    // The recorded child clause prunes a later superset popped from the queue.
    let to_process = h.prefilter(
        vec![domain_of(
            &[(0, 0, false), (1, 1, true)],
            UNDECIDED.0,
            UNDECIDED.1,
        )],
        THRESHOLD,
        &mut state,
    );
    assert!(to_process.is_empty());
    assert_eq!(state.domains_clause_pruned, 1);
}

// ---------------------------------------------------------------------------
// Gate OFF => inert (byte-identical baseline)
// ---------------------------------------------------------------------------

#[test]
fn test_disabled_store_never_records_nor_prunes() {
    let mut h = Harness::new();
    let mut state = BabLoopState::new(false);
    // BabLoopState::new leaves the store disabled — exactly the gate-off path.

    let clause = [(0, 0, true), (0, 1, true)];
    let to_process = h.prefilter(
        vec![domain_of(&clause, VERIFIED.0, VERIFIED.1)],
        THRESHOLD,
        &mut state,
    );
    assert!(to_process.is_empty());
    assert_eq!(state.clause_store.len(), 0, "gate off: nothing recorded");

    let superset = [(0, 0, true), (0, 1, true), (1, 0, false)];
    let to_process = h.prefilter(
        vec![domain_of(&superset, UNDECIDED.0, UNDECIDED.1)],
        THRESHOLD,
        &mut state,
    );
    assert_eq!(state.domains_clause_pruned, 0, "gate off: nothing pruned");
    assert_eq!(
        to_process.len(),
        1,
        "gate off: the superset domain must be processed normally"
    );
}

// ---------------------------------------------------------------------------
// Input-split firewall (fail closed at both entry points)
// ---------------------------------------------------------------------------

#[test]
fn test_input_split_domains_neither_recorded_nor_pruned() {
    let mut h = Harness::new();
    let mut state = BabLoopState::new(false);
    state.clause_store = ClauseStore::with_capacity(true, 100);

    // Record side: input-split verified domain must not produce a clause (its
    // certificate covers only its private input sub-box).
    let clause = [(0, 0, true), (0, 1, true)];
    let to_process = h.prefilter(
        vec![input_split_domain_of(&clause, VERIFIED.0, VERIFIED.1)],
        THRESHOLD,
        &mut state,
    );
    assert!(to_process.is_empty());
    assert_eq!(state.domains_verified, 1);
    assert_eq!(
        state.clause_store.len(),
        0,
        "input-split verified domain must NOT be recorded (over-claims full box)"
    );

    // Prune side: even with a legitimately recorded (pure) clause, an
    // input-split domain must not be prune-checked.
    let to_process = h.prefilter(
        vec![domain_of(&clause, VERIFIED.0, VERIFIED.1)],
        THRESHOLD,
        &mut state,
    );
    assert!(to_process.is_empty());
    assert_eq!(state.clause_store.len(), 1);

    let superset = [(0, 0, true), (0, 1, true), (1, 0, false)];
    let to_process = h.prefilter(
        vec![input_split_domain_of(&superset, UNDECIDED.0, UNDECIDED.1)],
        THRESHOLD,
        &mut state,
    );
    assert_eq!(
        state.domains_clause_pruned, 0,
        "input-split domain must fail closed at the prune check"
    );
    assert_eq!(to_process.len(), 1);
}

// ---------------------------------------------------------------------------
// End-to-end: gate ON vs OFF must produce the IDENTICAL final verdict
// ---------------------------------------------------------------------------

#[test]
fn test_verify_gate_on_vs_off_identical_verdicts() {
    // simple_network: out = relu(x - y) + relu(y - x) over [-1, 1]^2,
    // true range [0, 2]. Thresholds cover Verified, PotentialViolation, and
    // whatever the engine decides in between — the assertion is pure A/B
    // verdict equality, not a fixed expected verdict (parallel batches may
    // reorder closes, so only verdicts are comparable).
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let config = || BetaCrownConfig {
        timeout: Duration::from_secs(30),
        max_domains: 1000,
        ..Default::default()
    };

    for threshold in [-0.5f32, 0.25, 3.0] {
        set_test_gate_override(Some(false));
        let off = BetaCrownVerifier::new(config())
            .verify(&network, &input, threshold)
            .expect("gate-off verify must succeed");

        set_test_gate_override(Some(true));
        let on = BetaCrownVerifier::new(config())
            .verify(&network, &input, threshold)
            .expect("gate-on verify must succeed");
        set_test_gate_override(None);

        assert_eq!(
            std::mem::discriminant(&off.result),
            std::mem::discriminant(&on.result),
            "clause learning changed the verdict at threshold {threshold}: \
             OFF={:?} ON={:?}",
            off.result,
            on.result
        );
        // A sound pruner can only close domains already proven safe: it must
        // never manufacture a Verified that the OFF run's certificates don't
        // imply, and never lose one either (checked above); spot-check the two
        // decided variants exactly.
        if matches!(
            off.result,
            BabVerificationStatus::Verified | BabVerificationStatus::PotentialViolation
        ) {
            assert_eq!(off.result, on.result);
        }
    }
}
