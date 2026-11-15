// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pins for the untrusted dual proposal boundary and its certified tree cover.

use std::sync::atomic::{AtomicUsize, Ordering};

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use ny_core::search::{
    BackendCaps, DualMultipliers, DualSearchOutcome, DualSearchRequest, Refusal, SearchBackend,
    SearchPrecision, Untrusted,
};

use super::*;
use crate::margin_row::zono_state::{FrozenDualBuildStats, ReluSite};

const EVALUATORS: [Evaluator; 3] = [
    Evaluator::Node,
    Evaluator::Block { far_first: false },
    Evaluator::Block { far_first: true },
];

fn caps() -> BackendCaps {
    BackendCaps::Cpu {
        threads: 1,
        lanes: 1,
        precision: SearchPrecision::F64,
    }
}

/// Exact representation of the network margin
/// `beta + (x + y)/4 + relu(x) + relu(y) + relu(-x-y)`, on [-1,1]^2,
/// with beta = c_obj - 1. The positive instance has beta approximately .05.
/// Since relu(x)+relu(y) >= relu(x+y), its margin is >= beta everywhere.
fn toy_problem(c_obj: f64) -> FrozenDualProblem {
    let state = FrozenDualState {
        t: 0,
        class: 1,
        n_gens: 5,
        n_in_gens: 2,
        c_obj,
        d: vec![0.25, 0.25, 0.25, 0.25, 0.5],
        r_obj: 0.0,
        a: vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0,
        ],
        c_in: vec![0.0; 3],
        r: vec![0.0; 3],
        lam: vec![0.5; 3],
        mu: vec![0.25, 0.25, 0.5],
        lo: vec![-1.0, -1.0, -2.0],
        hi: vec![1.0, 1.0, 2.0],
        enc: vec![2, 3, 4],
        layer: vec![0; 3],
        site: vec![ReluSite::Trunk(0); 3],
        neuron: vec![0, 1, 2],
        stats: FrozenDualBuildStats::default(),
    };
    FrozenDualProblem::new(&state, &[0, 1, 2]).expect("exact toy problem")
}

/// Analytic optimum after the first two splits: these multipliers cancel
/// both input reduced costs, leaving only the third ReLU's own symbol.
struct ToyOptimizer;

impl SearchBackend for ToyOptimizer {
    fn name(&self) -> &'static str {
        "analytic_toy"
    }

    fn caps(&self) -> BackendCaps {
        caps()
    }

    fn refine_dual(&self, request: &DualSearchRequest<'_>) -> Result<DualSearchOutcome, Refusal> {
        let mut l0 = vec![0.0; request.rows.len()];
        for (value, row) in l0.iter_mut().zip(request.rows).take(2) {
            *value = 0.5 - 0.25 * row.sign;
        }
        Ok(DualSearchOutcome {
            multipliers: Untrusted::new(DualMultipliers {
                l0,
                l1: vec![0.0; request.rows.len()],
            }),
            steps: 1,
            // Deliberately meaningless: only the certified evaluator decides.
            gain: Untrusted::new(f64::INFINITY),
            stop: "closed",
            support: Default::default(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    Zero,
    Negative,
    Nan,
    PositiveInfinity,
    NegativeInfinity,
    Short,
    Long,
    Refusal,
    Panic,
}

struct FaultBackend {
    fault: Fault,
    calls: AtomicUsize,
    requested_steps: AtomicUsize,
}

impl FaultBackend {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            calls: AtomicUsize::new(0),
            requested_steps: AtomicUsize::new(0),
        }
    }
}

impl SearchBackend for FaultBackend {
    fn name(&self) -> &'static str {
        "hostile_dual"
    }

    fn caps(&self) -> BackendCaps {
        caps()
    }

    fn refine_dual(&self, request: &DualSearchRequest<'_>) -> Result<DualSearchOutcome, Refusal> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.requested_steps
            .fetch_max(request.steps, Ordering::Relaxed);
        let depth = request.rows.len();
        let (value, len) = match self.fault {
            Fault::Zero => (0.0, depth),
            Fault::Negative => (-100.0, depth),
            Fault::Nan => (f64::NAN, depth),
            Fault::PositiveInfinity => (f64::INFINITY, depth),
            Fault::NegativeInfinity => (f64::NEG_INFINITY, depth),
            Fault::Short => (0.0, depth - 1),
            Fault::Long => (0.0, depth + 1),
            Fault::Refusal => return Err(Refusal::Invalid("injected refusal".into())),
            Fault::Panic => panic!("injected dual backend panic"),
        };
        Ok(DualSearchOutcome {
            multipliers: Untrusted::new(DualMultipliers {
                l0: vec![value; len],
                l1: vec![value; len],
            }),
            steps: usize::MAX,
            gain: Untrusted::new(f64::INFINITY),
            stop: "unsat",
            support: Default::default(),
        })
    }
}

fn options(evaluator: Evaluator, refine_steps: usize) -> SearchOptions {
    SearchOptions {
        node_cap: 256,
        record_prunes: 32,
        evaluator,
        refine_steps,
        crosscheck_every: 1,
        ..SearchOptions::default()
    }
}

/// Compare behavior and certificates, excluding elapsed time and new telemetry.
fn same_search(a: &SearchReport, b: &SearchReport) {
    assert_eq!(a.verdict, b.verdict);
    assert_eq!(a.root_certified.to_bits(), b.root_certified.to_bits());
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.depth, b.depth);
    assert_eq!(a.certified_prunes, b.certified_prunes);
    assert_eq!(a.cert_min_slack.to_bits(), b.cert_min_slack.to_bits());
    assert_eq!(a.far_used, b.far_used);
    assert_eq!(a.peak_frontier, b.peak_frontier);
    assert_eq!(a.worst_open.to_bits(), b.worst_open.to_bits());
    assert_eq!(a.levels, b.levels);
    assert_eq!(a.pruned, b.pruned);
    assert_eq!(a.slack_hist, b.slack_hist);
    assert_eq!(a.crosschecks, b.crosschecks);
    assert_eq!(a.crosscheck_failures, b.crosscheck_failures);
    assert_eq!(a.evaluator, b.evaluator);
    assert_eq!(a.far_first, b.far_first);
}

fn q(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite certificate data")
}

/// Independent exact evaluation of weak duality from the stored floats.
fn exact_value(prob: &FrozenDualProblem, node: &PrunedNode) -> BigRational {
    let mut rc: Vec<_> = prob.d().iter().copied().map(q).collect();
    for i in 0..node.sides.len() {
        rc[prob.split_row(i).enc] = BigRational::zero();
    }
    let mut constant = q(prob.c0());
    let mut lag = BigRational::zero();
    for (i, &side) in node.sides.iter().enumerate() {
        let row = prob.split_row(i);
        let si = usize::from(side == Side::On);
        let l0 = q(node.l0[i]);
        let l1 = q(node.l1[i]);
        let multiplier = q(row.rho[si]) + q(side.sign()) * (&l0 - &l1);
        for (value, &coefficient) in rc.iter_mut().zip(row.a) {
            *value += &multiplier * q(coefficient);
        }
        constant += q(row.c0s[si]);
        lag += l0 * q(row.b0[si]) + l1 * q(row.b1[si]);
    }
    constant - rc.into_iter().map(|v| v.abs()).sum::<BigRational>() - lag
}

#[test]
fn breakpoint_disabled_or_absent_backend_preserves_baseline() {
    for evaluator in EVALUATORS {
        for prob in [toy_problem(1.05), FrozenDualProblem::synthetic(7, 4, 37)] {
            let base = search(&prob, &options(evaluator, 0));
            let backend = FaultBackend::new(Fault::Panic);
            let disabled =
                search_with_refinement_backend(&prob, &options(evaluator, 0), Some(&backend));
            let absent = search_with_refinement_backend(&prob, &options(evaluator, 16), None);
            same_search(&base, &disabled);
            same_search(&base, &absent);
            assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
            assert_eq!((disabled.refine_calls, absent.refine_calls), (0, 0));
        }
    }
}

#[test]
fn breakpoint_hostile_hints_cannot_close_zero_margin_or_change_cover() {
    let prob = FrozenDualProblem::synthetic_degenerate(2, 2, 0);
    for evaluator in EVALUATORS {
        let base = search(&prob, &options(evaluator, 0));
        assert_eq!(base.certified_prunes, 0);
        assert_eq!(
            base.nodes, 6,
            "negative control must traverse the full tree"
        );
        for fault in [
            Fault::Zero,
            Fault::Negative,
            Fault::Nan,
            Fault::PositiveInfinity,
            Fault::NegativeInfinity,
            Fault::Short,
            Fault::Long,
            Fault::Refusal,
            Fault::Panic,
        ] {
            let backend = FaultBackend::new(fault);
            let rep =
                search_with_refinement_backend(&prob, &options(evaluator, 16), Some(&backend));
            assert!(!rep.closed(), "{evaluator:?}: {fault:?} forged a closure");
            same_search(&base, &rep);
            assert_eq!(rep.refine_calls, rep.nodes, "backend was never exercised");
            assert_eq!(
                rep.refine_calls,
                backend.calls.load(Ordering::Relaxed) as u64
            );
            assert_eq!((rep.refine_accepted, rep.refine_prunes), (0, 0));
            if matches!(fault, Fault::Zero | Fault::Negative) {
                assert_eq!(
                    rep.refine_refused, 0,
                    "finite negative values must clamp to zero"
                );
            } else {
                assert_eq!(rep.refine_refused, rep.refine_calls);
            }
            assert!(!ledger_closes(&rep.levels));
        }
    }
}

#[test]
fn breakpoint_positive_search_metadata_cannot_prune_a_negative_certificate() {
    // Zero rows prevent every multiplier from improving this margin; the raw
    // value is positive, while the rounding charge makes the certificate negative.
    let n_in = 24;
    let rows = 2;
    let n = n_in + rows;
    let mut d = vec![0.0; n];
    for (i, value) in d.iter_mut().take(n_in).enumerate() {
        *value = if i % 2 == 0 { 5e7 } else { -5e7 };
    }
    let state = FrozenDualState {
        t: 0,
        class: 1,
        n_gens: n,
        n_in_gens: n_in,
        c_obj: d.iter().map(|v| v.abs()).sum::<f64>() + 5e-6,
        d,
        r_obj: 0.0,
        a: vec![0.0; rows * n],
        c_in: vec![0.0; rows],
        r: vec![0.0; rows],
        lam: vec![0.5; rows],
        mu: vec![1.0; rows],
        lo: vec![-1.0; rows],
        hi: vec![1.0; rows],
        enc: (n_in..n).collect(),
        layer: vec![0; rows],
        site: vec![ReluSite::Trunk(0); rows],
        neuron: (0..rows).collect(),
        stats: FrozenDualBuildStats::default(),
    };
    let prob = FrozenDualProblem::new(&state, &[0, 1]).expect("raw-positive fixture");
    let nb = prob.eval_node(
        &[Side::Off],
        &[0.0],
        &[0.0],
        &mut NodeScratch::default(),
        None,
    );
    assert!(
        nb.value > 0.0 && nb.certified < 0.0,
        "rounding mutation pin is void"
    );
    let backend = FaultBackend::new(Fault::Zero);
    for evaluator in EVALUATORS {
        let base = search(&prob, &options(evaluator, 0));
        let rep = search_with_refinement_backend(&prob, &options(evaluator, 16), Some(&backend));
        same_search(&base, &rep);
        assert!(!rep.closed());
        assert_eq!(rep.certified_prunes, 0);
        assert_eq!(rep.refine_calls, 6);
        assert!(
            rep.refine_steps <= rep.refine_calls * 16,
            "backend step telemetry overflowed"
        );
    }
}

#[test]
fn breakpoint_proposal_improves_toy_and_every_new_prune_replays_exactly() {
    let prob = toy_problem(1.05);
    for evaluator in EVALUATORS {
        let base = search(&prob, &options(evaluator, 0));
        assert_eq!(
            base.verdict,
            SearchVerdict::Open(OpenReason::SplitsExhausted)
        );
        assert_eq!((base.nodes, base.certified_prunes), (14, 0));
        let rep =
            search_with_refinement_backend(&prob, &options(evaluator, 16), Some(&ToyOptimizer));
        assert!(
            rep.closed(),
            "{evaluator:?}: analytic candidates did not close the toy"
        );
        assert_eq!((rep.nodes, rep.certified_prunes), (6, 4));
        assert_eq!(rep.refine_prunes, 4);
        assert!(
            ledger_closes(&rep.levels),
            "every root child must be covered"
        );
        assert_eq!(
            rep.pruned.len(),
            4,
            "exact re-verification population is void"
        );
        for node in &rep.pruned {
            assert_eq!(node.candidate, Candidate::Breakpoint);
            assert_eq!(node.sides.len(), 2);
            assert!(node.l1.iter().all(|&value| value == 0.0));
            for (i, side) in node.sides.iter().enumerate() {
                assert_eq!(node.l0[i], 0.5 - 0.25 * side.sign());
            }
            let exact = exact_value(&prob, node);
            assert!(exact.is_positive());
            assert!(
                q(node.certified) <= exact,
                "proposal certificate exceeds exact weak duality"
            );
            let replay = prob.eval_node(
                &node.sides,
                &node.l0,
                &node.l1,
                &mut NodeScratch::default(),
                None,
            );
            assert_eq!(replay.certified.to_bits(), node.certified.to_bits());
        }
    }
}

#[test]
fn breakpoint_proposals_never_close_the_toy_with_a_real_counterexample() {
    let prob = toy_problem(0.95);
    // At x=y=0 all three ReLUs vanish, so the exact margin is -.05.
    assert!(
        (q(0.95) - q(1.0)).is_negative(),
        "counterexample fixture is void"
    );
    for evaluator in EVALUATORS {
        let rep =
            search_with_refinement_backend(&prob, &options(evaluator, 16), Some(&ToyOptimizer));
        assert!(
            !rep.closed(),
            "{evaluator:?}: backend forged UNSAT for a witnessed SAT margin"
        );
        assert!(
            rep.refine_calls > 0,
            "negative control never exercised the backend"
        );
        assert!(!ledger_closes(&rep.levels));
        for node in &rep.pruned {
            assert!(q(node.certified) <= exact_value(&prob, node));
        }
    }
}

#[test]
fn breakpoint_backend_cannot_bypass_search_budgets() {
    let prob = toy_problem(1.05);
    for evaluator in EVALUATORS {
        for (reason, opts) in [
            (
                OpenReason::NodeCap,
                SearchOptions {
                    node_cap: 0,
                    ..options(evaluator, 16)
                },
            ),
            (
                OpenReason::Memory,
                SearchOptions {
                    max_frontier_bytes: 0,
                    ..options(evaluator, 16)
                },
            ),
            (
                OpenReason::Deadline,
                SearchOptions {
                    deadline: Some(Instant::now()),
                    ..options(evaluator, 16)
                },
            ),
        ] {
            let rep = search_with_refinement_backend(&prob, &opts, Some(&ToyOptimizer));
            assert_eq!(rep.verdict, SearchVerdict::Open(reason));
            assert_eq!((rep.nodes, rep.refine_calls), (0, 0));
            assert!(!ledger_closes(&rep.levels));
        }
        // The first level is evaluated but neither child closes; a candidate
        // must not fabricate the depth-two children denied by this node cap.
        let rep = search_with_refinement_backend(
            &prob,
            &SearchOptions {
                node_cap: 2,
                ..options(evaluator, 16)
            },
            Some(&ToyOptimizer),
        );
        assert_eq!(rep.verdict, SearchVerdict::Open(OpenReason::NodeCap));
        assert_eq!(rep.nodes, 2);
        assert_eq!(
            rep.levels,
            vec![LevelTrace {
                depth: 1,
                frontier: 2,
                pruned: 0
            }]
        );
        assert!(!ledger_closes(&rep.levels));
    }
}

#[test]
fn breakpoint_ingest_caps_backend_steps_and_untrusted_step_metadata() {
    let prob = FrozenDualProblem::synthetic_degenerate(2, 2, 0);
    let backend = FaultBackend::new(Fault::Zero);
    let rep = search_with_refinement_backend(
        &prob,
        &options(Evaluator::Node, usize::MAX),
        Some(&backend),
    );
    assert_eq!(backend.requested_steps.load(Ordering::Relaxed), 32);
    assert_eq!(rep.refine_calls, 6);
    assert_eq!(rep.refine_steps, 6 * 32);
    assert!(!rep.closed());
    assert_eq!((rep.refine_accepted, rep.refine_prunes), (0, 0));
}
