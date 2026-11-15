// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! The real CPU backend crosses the same hint boundary as injected test backends.
//! This target lives in ny-cli to preserve ny-propagate's dependency boundary.

use ny_bbe::CpuF32Backend;
use ny_propagate::margin_row::zono_bab::{
    ledger_closes, search, search_with_refinement_backend, Candidate, Evaluator, OpenReason,
    SearchOptions, SearchVerdict,
};
use ny_propagate::margin_row::zono_state::{
    FrozenDualBuildStats, FrozenDualProblem, FrozenDualState, NodeScratch, ReluSite,
};
use ny_propagate::margin_row::{RootGates, RoundMode, TwinNet, TwinOpSpec, TwinSpec};

const EVALUATORS: [Evaluator; 3] = [
    Evaluator::Node,
    Evaluator::Block { far_first: false },
    Evaluator::Block { far_first: true },
];

/// Margin `beta + (x+y)/4 + relu(x) + relu(y) + relu(-x-y)` on [-1,1]^2.
/// Its exact minimum is beta = c_obj-1, attained at the origin. The state
/// uses exact dyadic slopes, generators and coefficients; no build error.
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

fn options(evaluator: Evaluator, refine_steps: usize) -> SearchOptions {
    SearchOptions {
        evaluator,
        refine_steps,
        node_cap: 256,
        record_prunes: 32,
        crosscheck_every: 1,
        ..SearchOptions::default()
    }
}

#[test]
fn cpu_breakpoint_engine_closes_an_objective_the_baseline_leaves_open() {
    let prob = toy_problem(1.05);
    let backend = CpuF32Backend::with_threads(1);
    assert!(
        prob.root_bound() < 0.0,
        "the search fixture must be open at its root"
    );
    for evaluator in EVALUATORS {
        let baseline =
            search_with_refinement_backend(&prob, &options(evaluator, 0), Some(&backend));
        assert_eq!(
            baseline.verdict,
            SearchVerdict::Open(OpenReason::SplitsExhausted)
        );
        assert_eq!((baseline.nodes, baseline.refine_calls), (14, 0));
        let refined =
            search_with_refinement_backend(&prob, &options(evaluator, 16), Some(&backend));
        eprintln!(
            "[breakpoint-dual-toy] evaluator={evaluator:?} baseline_nodes={} refined_nodes={} \
             accepted={} new_prunes={} verdict={:?}",
            baseline.nodes,
            refined.nodes,
            refined.refine_accepted,
            refined.refine_prunes,
            refined.verdict,
        );
        assert!(
            refined.closed(),
            "{evaluator:?}: real CPU backend did not close the toy"
        );
        assert!(
            refined.nodes < baseline.nodes,
            "real backend must improve tree work, not timing"
        );
        assert!(
            refined.refine_prunes > 0,
            "the improvement must use the new engine"
        );
        assert_eq!(refined.crosscheck_failures, 0);
        assert!(ledger_closes(&refined.levels));
        assert_eq!(refined.pruned.len() as u64, refined.certified_prunes);
        let mut breakpoint_prunes = 0;
        for node in &refined.pruned {
            let bound = prob.eval_node(
                &node.sides,
                &node.l0,
                &node.l1,
                &mut NodeScratch::default(),
                None,
            );
            assert!(
                bound.prunes(),
                "a reported prune does not replay through the certifier"
            );
            assert_eq!(bound.certified.to_bits(), node.certified.to_bits());
            breakpoint_prunes += u64::from(node.candidate == Candidate::Breakpoint);
            #[cfg(feature = "mip")]
            assert!(exact_value(&prob, node) >= q(node.certified));
        }
        assert_eq!(breakpoint_prunes, refined.refine_prunes);
    }
}

#[test]
fn cpu_breakpoint_engine_keeps_a_witnessed_sat_objective_open() {
    let prob = toy_problem(0.95);
    let backend = CpuF32Backend::with_threads(1);
    // At the origin all ReLUs vanish: the stored network's margin is
    // 0.95 - 1 = -0.05 < 0.
    for evaluator in EVALUATORS {
        let report = search_with_refinement_backend(&prob, &options(evaluator, 16), Some(&backend));
        assert!(
            !report.closed(),
            "{evaluator:?}: real engine closed a counterexample"
        );
        assert!(
            report.refine_calls > 0,
            "negative control did not exercise the engine"
        );
        assert!(!ledger_closes(&report.levels));
        for node in &report.pruned {
            let bound = prob.eval_node(
                &node.sides,
                &node.l0,
                &node.l1,
                &mut NodeScratch::default(),
                None,
            );
            assert!(bound.prunes());
            assert_eq!(bound.certified.to_bits(), node.certified.to_bits());
            #[cfg(feature = "mip")]
            assert!(exact_value(&prob, node) >= q(node.certified));
        }
    }
}

/// The Off/Off node has dual
/// c-|2u-v|-|1-u|-|1-v|-.8(u+v). Starting at zero, either individual
/// coordinate move worsens it. The coupled ray (u,v)=(eta,2eta) reaches
/// (.5,1), where the exact value is c-1.7.
fn coupled_breakpoint_problem(c_obj: f64) -> FrozenDualProblem {
    let state = FrozenDualState {
        t: 0,
        class: 1,
        n_gens: 5,
        n_in_gens: 3,
        c_obj,
        d: vec![0.0, 1.0, 1.0, 0.0, 0.0],
        r_obj: 0.0,
        a: vec![2.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0],
        c_in: vec![-0.8; 2],
        r: vec![0.0; 2],
        lam: vec![0.5; 2],
        mu: vec![1.0; 2],
        lo: vec![
            ny_propagate::zono_core::rounding::next_down(-3.8),
            ny_propagate::zono_core::rounding::next_down(-2.8),
        ],
        hi: vec![
            ny_propagate::zono_core::rounding::next_up(2.2),
            ny_propagate::zono_core::rounding::next_up(1.2),
        ],
        enc: vec![3, 4],
        layer: vec![0; 2],
        site: vec![ReluSite::Trunk(0); 2],
        neuron: vec![0, 1],
        stats: FrozenDualBuildStats::default(),
    };
    FrozenDualProblem::new(&state, &[0, 1]).expect("coupled dual problem")
}

#[test]
fn cpu_coupled_breakpoint_rescue_crosses_the_certified_boundary() {
    use ny_core::search::{DualSearchRequest, DualSearchRow, SearchBackend};
    use ny_propagate::margin_row::zono_bab::PrunedNode;
    use ny_propagate::margin_row::zono_state::Side;

    let backend = CpuF32Backend::with_threads(1);
    let sides = [Side::Off, Side::Off];
    let zeros = [0.0; 2];
    for c_obj in [1.9, 1.6] {
        let problem = coupled_breakpoint_problem(c_obj);
        let mut scratch = NodeScratch::default();
        let initial = problem.eval_node(&sides, &zeros, &zeros, &mut scratch, None);
        assert!(!initial.prunes());
        // Actual split data are supplied by the same problem the certifier
        // evaluates; no independent reconstruction of cuts reaches the backend.
        let rows: Vec<_> = sides
            .iter()
            .enumerate()
            .map(|(i, &side)| {
                let row = problem.split_row(i);
                let si = usize::from(side == Side::On);
                DualSearchRow {
                    a: row.a,
                    sign: side.sign(),
                    b0: row.b0[si],
                    b1: row.b1[si],
                    vacuous: row.vacuous[si],
                }
            })
            .collect();
        let outcome = backend
            .refine_dual(&DualSearchRequest {
                rc: scratch.rc(),
                rows: &rows,
                l0: &zeros,
                l1: &zeros,
                steps: 4,
                deadline: None,
                mem_cap_bytes: 16 << 20,
            })
            .expect("real CPU coupled search");
        assert!(
            outcome.steps > 0,
            "the coordinate stall needs a coupled accepted step"
        );
        let mut proposal = outcome.multipliers.into_hint().into_ingest();
        assert_eq!(proposal.l0.len(), sides.len());
        assert_eq!(proposal.l1.len(), sides.len());
        for (i, row) in rows.iter().enumerate() {
            assert!(proposal.l0[i].is_finite() && proposal.l1[i].is_finite());
            proposal.l0[i] = proposal.l0[i].max(0.0);
            proposal.l1[i] = if row.vacuous {
                0.0
            } else {
                proposal.l1[i].max(0.0)
            };
        }
        let certified = problem.eval_node(
            &sides,
            &proposal.l0,
            &proposal.l1,
            &mut NodeScratch::default(),
            None,
        );
        assert!(certified.certified > initial.certified);
        if c_obj == 1.9 {
            assert!(
                certified.prunes(),
                "coupled raw gain did not survive certification"
            );
            assert!(certified.certified > 0.19);
            let node = PrunedNode {
                sides: sides.to_vec(),
                l0: proposal.l0,
                l1: proposal.l1,
                certified: certified.certified,
                candidate: Candidate::Breakpoint,
            };
            #[cfg(feature = "mip")]
            assert!(exact_value(&problem, &node) >= q(node.certified));
            let replay = problem.eval_node(
                &node.sides,
                &node.l0,
                &node.l1,
                &mut NodeScratch::default(),
                None,
            );
            assert!(replay.prunes());
            assert_eq!(replay.certified.to_bits(), node.certified.to_bits());
        } else {
            // x=(-.1,-1,-.7) lies in both Off halfspaces and has margin -.1.
            let witness = [-0.1, -1.0, -0.7, 0.0, 0.0];
            for row in &rows {
                let lhs = row.a.iter().zip(witness).map(|(&a, x)| a * x).sum::<f64>();
                assert!(lhs <= row.b0);
            }
            assert!(c_obj + witness[1] + witness[2] < 0.0);
            assert!(!certified.prunes(), "coupled search closed the unsafe node");
            assert!(certified.certified < -0.09);
        }
        // Both full objectives have a box counterexample at x1=x2=-1.
        // Closing the positive Off/Off node must not close the uncovered tree.
        assert!(c_obj - 2.0 < 0.0);
        for evaluator in EVALUATORS {
            let report =
                search_with_refinement_backend(&problem, &options(evaluator, 4), Some(&backend));
            assert!(!report.closed());
            assert!(!ledger_closes(&report.levels));
            assert!(report.refine_calls > 0);
            for node in &report.pruned {
                let replay = problem.eval_node(
                    &node.sides,
                    &node.l0,
                    &node.l1,
                    &mut NodeScratch::default(),
                    None,
                );
                assert!(replay.prunes());
                assert_eq!(replay.certified.to_bits(), node.certified.to_bits());
                #[cfg(feature = "mip")]
                assert!(exact_value(&problem, node) >= q(node.certified));
            }
            eprintln!(
                "[breakpoint-dual-coupled] c={c_obj} evaluator={evaluator:?} \
                nodes={} calls={} accepted={} new_prunes={} verdict={:?}",
                report.nodes,
                report.refine_calls,
                report.refine_accepted,
                report.refine_prunes,
                report.verdict
            );
        }
    }
}

#[test]
fn production_search_uses_the_cli_registered_cpu_backend() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let constructions = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&constructions);
    ny_propagate::margin_row::set_root_search_backend_factory(move || {
        count.fetch_add(1, Ordering::Relaxed);
        Some(Arc::new(CpuF32Backend::new()) as Arc<dyn ny_core::search::SearchBackend>)
    });
    let problem = toy_problem(1.05);
    let dormant = search(&problem, &options(Evaluator::Node, 0));
    assert!(!dormant.closed());
    assert_eq!(constructions.load(Ordering::Relaxed), 0);
    let opts = options(Evaluator::Node, 4);
    // The control explicitly bypasses process registration to stay isolated.
    let baseline = search_with_refinement_backend(&problem, &opts, None);
    assert!(!baseline.closed());
    for _ in 0..2 {
        let report = search(&problem, &opts);
        assert!(
            report.closed(),
            "production search failed to obtain the registered backend"
        );
        assert!(report.nodes < baseline.nodes);
        assert!(report.refine_calls > 0 && report.refine_prunes > 0);
        assert!(ledger_closes(&report.levels));
    }
    assert_eq!(constructions.load(Ordering::Relaxed), 1);
    automatic_tier_uses_registered_backend_and_preserves_open_classes();
    automatic_tier_streams_a_large_state_without_shortening_the_legacy_depth();
    assert_eq!(constructions.load(Ordering::Relaxed), 1);
}

/// A valid compiled network, including the required trunk ReLU. Its stable
/// trunk computes (x+2,y+2); the tail recovers x, y, -(x+y) and the positive
/// trunk coordinates. Output 0 is
/// f=.05+(x+y)/4+relu(x)+relu(y)+relu(-x-y), whose minimum is .05.
/// Outputs 1,2,3 are .5*f+.1, 0 and -.05, respectively.
fn automatic_route_network() -> TwinNet {
    automatic_route_network_with_extra_rows(0)
}

/// Extra x preactivations are real unstable ReLU rows with zero logit weight.
/// They widen the state without changing the margin or its known minimum.
fn automatic_route_network_with_extra_rows(extra: usize) -> TwinNet {
    let head = 5 + extra;
    let mut weight = vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0];
    let mut bias = vec![-2.0, -2.0, 4.0, 0.0, 0.0];
    for _ in 0..extra {
        weight.extend_from_slice(&[1.0, 0.0]);
        bias.push(-2.0);
    }
    let mut output_weight = Vec::with_capacity(4 * head);
    for row in [
        [1.0, 1.0, 1.0, 0.25, 0.25],
        [0.5, 0.5, 0.5, 0.125, 0.125],
        [0.0; 5],
        [0.0; 5],
    ] {
        output_weight.extend_from_slice(&row);
        output_weight.resize(output_weight.len() + extra, 0.0);
    }
    TwinNet::compile(&TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::ChannelAffine {
                input: 0,
                scale: vec![1.0, 1.0],
                shift: vec![2.0, 2.0],
                scale_rel_err: 0.0,
                shift_err: vec![0.0, 0.0],
                shape: (2, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Gemm {
                input: 2,
                weight,
                bias,
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (head, 2),
            },
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Gemm {
                input: 4,
                weight: output_weight,
                bias: vec![-0.95, -0.375, 0.0, -0.05],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (4, head),
            },
        ],
    })
    .expect("valid compiled automatic-route fixture")
}

/// Called after registration in the single factory-owning test. Keeping these
/// checks together prevents independent tests racing the process OnceLock.
fn automatic_tier_uses_registered_backend_and_preserves_open_classes() {
    use ny_propagate::margin_row::bab::root_eval;
    use ny_propagate::margin_row::engine::BackwardEngine;
    use ny_propagate::margin_row::zono_bab::{run_tier, TierPolicy};
    use ny_propagate::margin_row::zono_state::dd_forward;

    let net = automatic_route_network();
    let mut root = RootGates::build_prec(
        &net,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        RoundMode::Outward,
        None,
        false,
    )
    .expect("certified fixture root");
    root.head_alpha = Some(vec![0.5; 5]);
    let adv = [1, 2, 3];
    let evaluated =
        root_eval(&BackwardEngine::new(&net, &root), &net, 0, &adv).expect("production root pass");
    assert_eq!(
        evaluated.tree_classes, adv,
        "all three margins must reach the tier"
    );
    let mut policy = TierPolicy::clamped(256, 0.0);
    policy.refine_steps = 4;
    assert!(policy.automatic());
    let report = run_tier(
        &net,
        &root,
        0,
        &adv,
        &evaluated.dj,
        &evaluated.tree_classes,
        &evaluated.root_ybox(),
        None,
        &policy,
    );
    assert!(report.automatic);
    assert_eq!(report.automatic_skip, "none");
    assert_eq!(report.objectives, 3);
    assert_eq!(report.automatic_skipped, 1);
    assert_eq!(report.refused, 0);
    assert_eq!(report.classes.len(), 2);
    assert_eq!(report.budget_secs, 4.0);
    assert_eq!(
        report.closed,
        [3],
        "only the selected positive objective closes"
    );
    assert_eq!(report.crosscheck_failures, 0);

    for class in &report.classes {
        let run = class.result.as_ref().expect("automatic class built");
        assert_eq!(run.unstable, 3, "the fixture must exercise retained rows");
        assert_eq!(
            run.depth_cap,
            run.unstable
                .min(ny_propagate::margin_row::zono_bab::FROZEN_DUAL_MAX_DEPTH),
            "the production compact route must preserve the legacy depth limit"
        );
        assert_eq!(run.build_columns, 1 + run.unstable);
        assert_eq!(run.state_gens, 5);
        assert!(
            !run.streamed_rows,
            "small states retain the legacy full-build route"
        );
        assert!(run.state_bytes <= run.build_estimate_bytes);
    }

    // The real network gives a counterexample to class 1 at the origin.
    let forward = dd_forward(&net, &[0.0, 0.0]).expect("exact-network witness");
    let outputs = &forward[net.ops.len()];
    assert!((outputs[0] - outputs[1]).to_f64() < 0.0);
    let unsafe_run = &report
        .classes
        .iter()
        .find(|r| r.class == 1)
        .expect("unsafe class selected by the production ranking")
        .result
        .as_ref()
        .expect("unsafe class searched")
        .search;
    assert!(!unsafe_run.closed());
    assert!(!ledger_closes(&unsafe_run.levels));
    assert!(
        unsafe_run.refine_calls > 0,
        "automatic refinement did not engage"
    );
    let positive = &report
        .classes
        .iter()
        .find(|r| r.class == 3)
        .expect("positive class selected")
        .result
        .as_ref()
        .expect("positive class searched")
        .search;
    assert!(positive.closed());
    assert!(ledger_closes(&positive.levels));
    for run in [unsafe_run, positive] {
        assert!(run.nodes <= 256);
        assert_eq!(run.levels.first().expect("searched level").frontier, 2);
        assert_eq!(
            run.levels.iter().map(|level| level.frontier).sum::<u64>(),
            run.nodes,
            "every evaluated node appears in the level ledger"
        );
        for level in &run.levels {
            assert!(level.pruned <= level.frontier);
        }
        for pair in run.levels.windows(2) {
            assert_eq!(
                pair[1].frontier,
                2 * (pair[0].frontier - pair[0].pruned),
                "all open nodes contributed both children to the next level"
            );
        }
    }
    let remaining: Vec<_> = evaluated
        .tree_classes
        .iter()
        .copied()
        .filter(|class| !report.closed.contains(class))
        .collect();
    assert_eq!(
        remaining,
        [1, 2],
        "unsafe and skipped objectives remain in the caller tree"
    );
    eprintln!(
        "[breakpoint-dual-automatic] skipped={} closed={:?} unsafe_nodes={} \
         unsafe_calls={} unsafe_accepts={} positive_nodes={} positive_prunes={} budget={}",
        report.automatic_skipped,
        report.closed,
        unsafe_run.nodes,
        unsafe_run.refine_calls,
        unsafe_run.refine_accepted,
        positive.nodes,
        positive.refine_prunes,
        report.budget_secs,
    );
}

fn automatic_tier_streams_a_large_state_without_shortening_the_legacy_depth() {
    use ny_propagate::margin_row::bab::root_eval;
    use ny_propagate::margin_row::engine::BackwardEngine;
    use ny_propagate::margin_row::zono_bab::{run_tier, TierPolicy, FROZEN_DUAL_MAX_DEPTH};

    let net = automatic_route_network_with_extra_rows(597);
    let mut root = RootGates::build_prec(
        &net,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        RoundMode::Outward,
        None,
        false,
    )
    .expect("wide compiled fixture root");
    root.head_alpha = Some(vec![0.5; net.n_y]);
    let adv = [3];
    let evaluated = root_eval(&BackwardEngine::new(&net, &root), &net, 0, &adv)
        .expect("wide fixture production root pass");
    assert_eq!(
        evaluated.tree_classes, adv,
        "wide margin must reach the tier"
    );
    let mut policy = TierPolicy::clamped(256, 0.0);
    policy.refine_steps = 4;
    let report = run_tier(
        &net,
        &root,
        0,
        &adv,
        &evaluated.dj,
        &evaluated.tree_classes,
        &evaluated.root_ybox(),
        None,
        &policy,
    );
    assert!(report.automatic);
    assert_eq!(report.budget_secs, 4.0);
    assert_eq!(report.automatic_skip, "none");
    assert_eq!(report.automatic_skipped, 0);
    assert_eq!(report.refused, 0);
    assert_eq!(report.objectives, 1);
    assert_eq!(report.closed, [3]);
    assert_eq!(report.classes.len(), 1);
    let run = report.classes[0].result.as_ref().expect("wide state built");
    assert!(
        run.streamed_rows,
        "the production route did not use the compact builder"
    );
    assert_eq!(run.unstable, 600);
    assert_eq!(run.state_gens, 602);
    assert_eq!(
        run.build_columns, 601,
        "discarded rows still contribute seed columns"
    );
    assert_eq!(run.depth_cap, FROZEN_DUAL_MAX_DEPTH);
    assert_eq!(run.depth_cap, 512, "streaming must retain depths beyond 32");
    let full_dense_bytes = (run.unstable * run.state_gens * size_of::<f64>()) as u64;
    assert!(
        run.state_bytes < full_dense_bytes,
        "compact state including metadata must be smaller than the legacy dense rows alone"
    );
    assert!(run.state_bytes <= run.build_estimate_bytes);
    assert!(run.search.closed());
    assert!(ledger_closes(&run.search.levels));
    assert_eq!(run.search.crosscheck_failures, 0);
    eprintln!(
        "[frozen-dual-streamed-route] unstable={} retained={} gens={} columns={} \
        state_bytes={} full_dense_bytes={} nodes={} verdict={:?}",
        run.unstable,
        run.depth_cap,
        run.state_gens,
        run.build_columns,
        run.state_bytes,
        full_dense_bytes,
        run.search.nodes,
        run.search.verdict
    );
}

#[cfg(feature = "mip")]
fn q(value: f64) -> num_rational::BigRational {
    num_rational::BigRational::from_float(value).expect("finite certificate data")
}

/// The optional MIP feature already supplies exact rational dependencies.
/// The normal target always replays the certificate through the trusted CPU evaluator.
#[cfg(feature = "mip")]
fn exact_value(
    prob: &FrozenDualProblem,
    node: &ny_propagate::margin_row::zono_bab::PrunedNode,
) -> num_rational::BigRational {
    use num_traits::{Signed, Zero};
    use ny_propagate::margin_row::zono_state::Side;

    let mut rc: Vec<_> = prob.d().iter().copied().map(q).collect();
    for i in 0..node.sides.len() {
        rc[prob.split_row(i).enc] = num_rational::BigRational::zero();
    }
    let mut constant = q(prob.c0());
    let mut lag = num_rational::BigRational::zero();
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
    constant
        - rc.into_iter()
            .map(|v| v.abs())
            .sum::<num_rational::BigRational>()
        - lag
}
