// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

use super::super::{
    Candidate, Evaluator, FrozenDualSearch, FrozenDualStateMode, PrunedNode, SearchOptions,
    SearchVerdict, TierPolicy,
};
use super::*;
use crate::margin_row::net::TwinNet;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::zono_state::{
    class_margin, dd_forward, static_order, FrameInput, FrozenDualOptions, FrozenDualTrunk,
    LayerInput, TrunkRelax,
};

/// Genuine compiled network, with biased correlated trunk rows followed by
/// an unstable head. Every requested margin is negative at x=0. The large
/// negative bias keeps the old finite far probes open on the impossible
/// phase pair too; it is a scheduling/arrival control, not a speed benchmark.
fn scalar_conv(weight: Vec<f64>, bias: Vec<f64>) -> TwinOpSpec {
    let width = weight.len();
    TwinOpSpec::Conv {
        input: 0,
        weight,
        bias,
        bias_err: vec![0.0; width],
        weight_rel_err: 0.0,
        kernel: (width, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 1),
        oshape: (width, 1, 1),
    }
}

fn compiled() -> (TwinNet, FrozenDualFrame) {
    let gemm = |input, weight, bias, shape| TwinOpSpec::Gemm {
        input,
        weight,
        bias,
        shape,
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
    };
    let net = TwinNet::compile(&TwinSpec {
        n_in: 1,
        ops: vec![
            scalar_conv(vec![1.0, 1.0], vec![0.25, -0.25]),
            TwinOpSpec::Relu { input: 1 },
            gemm(2, vec![1.0, -1.0, -1.0, 1.0], vec![-0.125, 0.125], (2, 2)),
            TwinOpSpec::Relu { input: 3 },
            gemm(
                4,
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 100_000_000.0, 100_000_001.0],
                (3, 2),
            ),
        ],
    })
    .expect("compiled original network");
    let frame = FrozenDualFrame::new(
        &net,
        FrameInput {
            lo: vec![-1.0],
            hi: vec![1.0],
            trunk: vec![LayerInput {
                l: vec![-0.75, -1.25],
                u: vec![1.25, 0.75],
                alpha: vec![0.5; 2],
                s: vec![0.625, 0.375],
            }],
            head: LayerInput {
                l: vec![-0.125, -0.375],
                u: vec![0.375, 0.125],
                alpha: vec![0.5; 2],
                s: vec![0.75, 0.25],
            },
        },
        0,
    )
    .expect("mathematically valid affine/ReLU boxes");
    (net, frame)
}

fn state(net: &TwinNet, frame: &FrozenDualFrame, class: usize) -> FrozenDualState {
    let own = class_margin(net, frame, 1, None).unwrap();
    let trunk = FrozenDualTrunk::build(
        net,
        frame,
        TrunkRelax::anchor(frame, &own),
        &FrozenDualOptions::default(),
    )
    .unwrap();
    let tail = trunk.relax().tail(net, frame, class, None).unwrap();
    trunk.state(frame, class, &tail).unwrap()
}

fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
fn exact_value(prob: &FrozenDualProblem, node: &PrunedNode) -> Q {
    let mut rc: Vec<_> = prob.d().iter().copied().map(q).collect();
    for i in 0..node.sides.len() {
        rc[prob.split_row(i).enc] = Q::zero();
    }
    let (mut constant, mut lag) = (q(prob.c0()), Q::zero());
    for (i, &side) in node.sides.iter().enumerate() {
        let row = prob.split_row(i);
        let si = usize::from(side == Side::On);
        let (l0, l1) = (q(node.l0[i]), q(node.l1[i]));
        let mult = q(row.rho[si]) + q(side.sign()) * (&l0 - &l1);
        for (acc, &a) in rc.iter_mut().zip(row.a) {
            *acc += &mult * q(a);
        }
        constant += q(row.c0s[si]);
        lag += l0 * q(row.b0[si]) + l1 * q(row.b1[si]);
    }
    constant - rc.into_iter().map(|v| v.abs()).sum::<Q>() - lag
}

fn replay(prob: &FrozenDualProblem, report: &super::super::SearchReport) -> usize {
    let mut count = 0;
    for node in &report.pruned {
        let checked = prob.eval_node(
            &node.sides,
            &node.l0,
            &node.l1,
            &mut NodeScratch::default(),
            None,
        );
        assert_eq!(checked.certified.to_bits(), node.certified.to_bits());
        assert!(node.certified > 0.0 && q(node.certified) <= exact_value(prob, node));
        if node.candidate == Candidate::Relation {
            assert!(node.l0.iter().filter(|&&x| x > 0.0).count() >= 2);
            count += 1;
        }
    }
    count
}

#[test]
fn fd_relations_all_four_phases_radii_and_boundary_ties_are_checked() {
    let (net, frame) = compiled();
    let mut s = state(&net, &frame, 1);
    let found = check_pair(&s, [0, 1], [0.5, 0.5], None).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].0, [Side::Off, Side::On]);
    assert!(found[0].1 > 0.249999);
    // The other three children have concrete input witnesses; all four
    // combinations were checked, and only the strict conflict is retained.
    for (x, phases) in [
        (-0.75, [Side::Off, Side::Off]),
        (0.0, [Side::On, Side::Off]),
        (0.75, [Side::On, Side::On]),
    ] {
        for (i, phase) in phases.into_iter().enumerate() {
            let z = s.c_in[i] + s.row(i)[0] * x;
            assert!(if phase == Side::Off {
                z <= 0.0
            } else {
                z >= 0.0
            });
        }
    }
    s.r[0] = 0.25;
    s.r[1] = 0.25;
    assert_eq!((check_pair(&s, [0, 1], [0.5, 0.5], None).unwrap()).len(), 0);
    s.r[0] = 0.0;
    s.r[1] = 0.0;
    s.c_in[0] = 0.0;
    s.c_in[1] = -0.0;
    assert!(
        check_pair(&s, [0, 1], [0.5, 0.5], None).unwrap().is_empty(),
        "z=0 lies in both children"
    );
    s.r[0] = -f64::from_bits(1);
    assert!(check_pair(&s, [0, 1], [0.5, 0.5], None).is_none());
}

#[test]
fn fd_relations_unequal_and_opposite_rows_have_outward_dyadic_checks() {
    let (net, frame) = compiled();
    let mut s = state(&net, &frame, 1);
    s.a[s.n_gens] = 3.0;
    let found = check_pair(&s, [0, 1], [0.75, 0.25], None).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].0, [Side::Off, Side::On]);
    // Exact dyadic combination: coefficients cancel, RHS is -1/4.
    assert!(q(found[0].1) <= q(0.25));
    s.a[s.n_gens] = -1.0;
    s.c_in[1] = 0.25;
    let found = check_pair(&s, [0, 1], [0.5, 0.5], None).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].0, [Side::Off, Side::Off]);
    assert!(q(found[0].1) <= q(0.25));
    s.a[0] = 0.1;
    s.a[s.n_gens] = 0.3;
    s.c_in[1] = -0.25;
    let found = check_pair(&s, [0, 1], [0.75, 0.25], None).unwrap();
    let exact = q(0.25) - (q(0.75) * q(0.1) - q(0.25) * q(0.3)).abs();
    assert_eq!(found.len(), 1);
    assert!(
        q(found[0].1) <= exact,
        "decimal float coefficient cancellation is checked as exact dyadics"
    );
    s.a[0] = f64::INFINITY;
    assert!(check_pair(&s, [0, 1], [0.5, 0.5], None).is_none());
}

#[test]
fn fd_relations_transfer_requires_complete_frame_common_chart_and_original_rows() {
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let order = static_order(&s);
    let mut cache = Cache::default();
    let first = cache.bind(&frame, &s, &order, None, MAX_BYTES).unwrap();
    assert!(first.active());
    let second = state(&net, &frame, 2);
    let reused = cache
        .bind(&frame, &second, &order, None, MAX_BYTES)
        .unwrap();
    assert!(
        reused.active() && reused.stats.replayed > 0,
        "target change preserves checked original chart"
    );
    for mutate in [0, 1, 2, 3, 4] {
        let mut changed = s.clone();
        match mutate {
            0 => changed.lam[0] = 0.125,
            1 => changed.mu[0] *= 2.0,
            2 => changed.a[0] += 0.125,
            3 => changed.enc[0] = changed.enc[1],
            _ => changed.r[0] += 0.125,
        }
        let hints = cache
            .bind(&frame, &changed, &order, None, MAX_BYTES)
            .unwrap();
        assert!(!hints.active());
        assert!(hints.stats.refused > 0);
    }
    let mut changed = frame.clone();
    changed.mid[0] = 0.125;
    let hints = cache.bind(&changed, &s, &order, None, MAX_BYTES).unwrap();
    assert!(!hints.active() && hints.stats.refused > 0);
    // A class-specific tail change is never imported as common geometry.
    let mut changed = s.clone();
    changed.lam[2] = 0.125;
    assert!(cache
        .bind(&frame, &changed, &order, None, MAX_BYTES)
        .unwrap()
        .active());
}

#[test]
fn fd_relations_literal_mismatch_and_whole_level_cap_decline_without_partial_work() {
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let order = vec![0, 1, 2, 3];
    let hints = Cache::default()
        .bind(&frame, &s, &order, None, MAX_BYTES)
        .unwrap();
    let prob = FrozenDualProblem::new(&s, &order).unwrap();
    let mut stats = RelationStats::default();
    let level = hints
        .admit_level(prob.n(), 2, &[0, 0, 0, 1, 1, 0, 1, 1], None, &mut stats)
        .unwrap();
    for sides in [
        [Side::On, Side::Off],
        [Side::Off, Side::Off],
        [Side::On, Side::On],
    ] {
        assert!(level
            .propose(
                &prob,
                &sides,
                &[0.0; 2],
                &[0.0; 2],
                -1.0,
                None,
                &mut NodeScratch::default(),
                &mut stats
            )
            .is_none());
    }
    assert_eq!(stats.used, 0);
    hints
        .allowance
        .as_ref()
        .unwrap()
        .0
        .store(1, Ordering::Relaxed);
    let before = stats.work_reserved;
    assert!(hints
        .admit_level(prob.n(), 2, &[0, 0, 0, 1, 1, 0, 1, 1], None, &mut stats)
        .is_none());
    assert_eq!(stats.work_reserved, before);
    assert!(stats.refused > 0);
    let mut poisoned = frame.clone();
    poisoned.mag.push(vec![f64::NAN; TERMS as usize + 1]);
    let mut cache = Cache::default();
    let refused = cache.bind(&poisoned, &s, &order, None, MAX_BYTES).unwrap();
    assert!(!refused.active());
    assert!(refused.stats.scope_census > TERMS && refused.stats.scope_over_cap == 1);
    assert!(
        cache.scope.is_none() && cache.allowance.is_none(),
        "length-only preflight must precede scanning poisoned coefficients"
    );
    let refused = cache.bind(&frame, &s, &order, None, 1).unwrap();
    assert!(!refused.active() && cache.allowance.is_none());
    let refused = cache
        .bind(
            &frame,
            &s,
            &order,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
            MAX_BYTES,
        )
        .unwrap();
    assert!(!refused.active() && cache.scope.is_none());
}

#[test]
fn fd_relations_record_actual_multipliers_and_replay_in_exact_rationals_all_evaluators() {
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let order = vec![0, 1, 2, 3];
    let prob = FrozenDualProblem::new(&s, &order).unwrap();
    for evaluator in [
        Evaluator::Node,
        Evaluator::Block { far_first: false },
        Evaluator::Block { far_first: true },
    ] {
        let hints = Cache::default()
            .bind(&frame, &s, &order, None, MAX_BYTES)
            .unwrap();
        let opts = SearchOptions {
            evaluator,
            node_cap: 1024,
            record_prunes: 1024,
            crosscheck_every: 1,
            ..SearchOptions::default()
        };
        let baseline = super::super::search_with_refinement_backend(&prob, &opts, None);
        let actual = super::super::search_with_relations(&prob, &opts, Some(&hints));
        assert_ne!(
            actual.verdict,
            SearchVerdict::Closed,
            "original network is witnessed SAT"
        );
        assert!(actual.relations.used > 0 && actual.relations.eval_improved > 0);
        assert!(
            replay(&prob, &actual) > 0,
            "a Relation prune must retain its actual multiplier vectors"
        );
        assert_eq!(actual.crosscheck_failures, 0);
        eprintln!(
            "[fd-relation-control] eval={} baseline_nodes={} relation_nodes={} receipt={:?}",
            evaluator.token(),
            baseline.nodes,
            actual.nodes,
            actual.relations
        );
    }
}

#[test]
fn fd_relations_default_shared_tier_arrives_and_keeps_every_witnessed_class_open() {
    let (net, frame) = compiled();
    let logits = dd_forward(&net, &[0.0]).unwrap();
    for class in [1, 2] {
        assert!((logits.last().unwrap()[0] - logits.last().unwrap()[class]).to_f64() < 0.0);
    }
    let policy = TierPolicy::clamped(1024, 30.0)
        .with_state(FrozenDualStateMode::Shared)
        .with_crosscheck_every(1);
    let report = super::super::run_tier_on_frame(
        &net,
        frame,
        &[1, 2],
        &[-1.0, -1.0],
        &[1, 2],
        None,
        &policy,
    );
    assert_eq!(report.classes.len(), 2);
    assert_eq!(report.classes_built, 2);
    assert_eq!((report.closed).len(), 0);
    let mut used = 0;
    let mut replayed = 0;
    for class in &report.classes {
        let run = class.result.as_ref().unwrap();
        assert_eq!(run.state, FrozenDualStateMode::Shared);
        assert_ne!(run.search.verdict, SearchVerdict::Closed);
        used += run.search.relations.used;
        replayed += run.search.relations.replayed;
        assert_eq!(run.search.crosscheck_failures, 0);
    }
    assert!(
        used > 0 && replayed >= 2,
        "real caller must consume and replay shared relation proposals"
    );
}

#[test]
fn fd_relations_successful_wide_survivors_compose_with_original_certification() {
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let order = vec![0, 1, 2, 3];
    let prob = FrozenDualProblem::new(&s, &order).unwrap();
    let hints = Cache::default()
        .bind(&frame, &s, &order, None, MAX_BYTES)
        .unwrap();
    let mut tier =
        super::super::wide::WideTier::with_backend_factory(FrozenDualSearch::CpuF32, |prob| {
            vec![Arc::new(super::super::wide::ReferenceBackend::new(prob))]
        });
    let opts = SearchOptions {
        node_cap: 1024,
        record_prunes: 1024,
        crosscheck_every: 1,
        ..SearchOptions::default()
    };
    let actual = tier.search_class_with_relations(&prob, &opts, Some(&hints));
    assert_eq!(actual.probe, "ok");
    assert_ne!(actual.verdict, SearchVerdict::Closed);
    assert!(actual.relations.used > 0 && actual.relations.eval_improved > 0);
    assert!(replay(&prob, &actual) > 0);
    assert_eq!(actual.crosscheck_failures, 0);
}

#[test]
fn fd_relations_optional_scratch_declines_without_refusing_the_old_level() {
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let order = vec![0, 1, 2, 3];
    let prob = FrozenDualProblem::new(&s, &order).unwrap();
    let hints = Cache::default()
        .bind(&frame, &s, &order, None, MAX_BYTES)
        .unwrap();
    let opts = SearchOptions {
        node_cap: 6,
        max_frontier_bytes: super::super::level_bytes(2, 1, prob.n())
            .max(super::super::level_bytes(4, 2, prob.n())),
        ..SearchOptions::default()
    };
    let baseline = super::super::search_with_refinement_backend(&prob, &opts, None);
    let actual = super::super::search_with_relations(&prob, &opts, Some(&hints));
    assert_eq!(
        (actual.verdict, actual.nodes, &actual.levels),
        (baseline.verdict, baseline.nodes, &baseline.levels)
    );
    assert_eq!(actual.relations.used, 0);
    assert!(actual.relations.refused > 0);
}

#[test]
fn fd_relations_complete_level_admission_is_thread_count_independent() {
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let order = vec![0, 1, 2, 3];
    let prob = FrozenDualProblem::new(&s, &order).unwrap();
    let mut reports = Vec::new();
    for threads in [1, 4] {
        let hints = Cache::default()
            .bind(&frame, &s, &order, None, MAX_BYTES)
            .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let opts = SearchOptions {
            node_cap: 1024,
            record_prunes: 1024,
            crosscheck_every: 1,
            ..SearchOptions::default()
        };
        reports
            .push(pool.install(|| super::super::search_with_relations(&prob, &opts, Some(&hints))));
    }
    let (a, b) = (&reports[0], &reports[1]);
    assert_eq!(
        (a.verdict, a.nodes, &a.levels, &a.pruned, a.relations),
        (b.verdict, b.nodes, &b.levels, &b.pruned, b.relations)
    );
    assert!(a.relations.used > 0);
}

#[test]
fn fd_relations_nonmatching_nodes_do_not_copy_unreserved_incumbents() {
    let depth = 128;
    let count = 2048;
    let hints = ClassHints {
        conflicts: vec![Conflict {
            rows: [0, 1],
            hashes: [[0; 32]; 2],
            phases: [Side::Off, Side::On],
            weights: [0.5; 2],
            contradiction: 0.25,
        }],
        allowance: Some(Arc::new(Allowance(AtomicU64::new(TERMS)))),
        stats: RelationStats::default(),
    };
    // This is solely a resource dispatch fixture, never a geometric proof.
    // Only the last node matches the admitted conflict. Unconditional copies
    // would visit more elements than the entire tier allowance.
    let mut sides = vec![0; depth * count];
    sides[(count - 1) * depth + 1] = 1;
    let mut stats = RelationStats::default();
    let level = hints
        .admit_level(depth + 1, depth, &sides, None, &mut stats)
        .unwrap();
    let calls = std::cell::Cell::new(0usize);
    let visits = std::cell::Cell::new(0usize);
    let coefficients = vec![0.0; depth];
    for node in sides.chunks_exact(depth) {
        let copied = super::super::relation_incumbent(Some(level), node, || {
            calls.set(calls.get() + 1);
            visits.set(visits.get() + 2 * coefficients.len());
            (coefficients.clone(), coefficients.clone())
        });
        assert_eq!(copied.is_some(), node[1] == 1);
    }
    assert_eq!(calls.get(), 1);
    assert_eq!(visits.get(), 2 * depth);
    assert!(2 * depth * count > TERMS as usize);
    assert!(stats.work_reserved < TERMS && stats.refused == 0);
}

#[test]
fn fd_relations_missing_rows_are_charged_before_every_cached_lookup() {
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let mut cache = Cache::default();
    assert!(cache
        .bind(&frame, &s, &[0, 1, 2, 3], None, MAX_BYTES)
        .unwrap()
        .active());
    // Duplicate checked entries only to stress lookup accounting. Every entry
    // is absent from this rank prefix; none reaches geometry or a proposal.
    cache.conflicts = vec![cache.conflicts[0].clone(); 16];
    let order = [2, 3];
    let census = scope_words(&frame, &s) + order.len() as u64 * 16;
    let lookup = order.len() as u64 * 2 + 8;
    let all = cache.bind(&frame, &s, &order, None, MAX_BYTES).unwrap();
    assert!(!all.active());
    assert_eq!(all.stats.replayed, 0);
    assert_eq!(all.stats.work_reserved, census + 16 * lookup);
    cache
        .allowance
        .as_ref()
        .unwrap()
        .0
        .store(census + 3 * lookup, Ordering::Relaxed);
    let capped = cache.bind(&frame, &s, &order, None, MAX_BYTES).unwrap();
    assert!(!capped.active() && capped.stats.refused == 1);
    assert_eq!(capped.stats.replayed, 0);
    assert_eq!(capped.stats.work_reserved, census + 3 * lookup);
    assert_eq!(
        cache.allowance.as_ref().unwrap().0.load(Ordering::Relaxed),
        0
    );
}

#[test]
fn fd_relations_inactive_receipt_releases_cache_and_preserves_tight_old_frontier() {
    let (net, frame) = compiled();
    let mut s = state(&net, &frame, 1);
    // A looser ORIGINAL preactivation enclosure eliminates strict conflicts
    // while preserving containment. The optional bind still scans/discovers.
    s.r[0] = 0.5;
    s.r[1] = 0.5;
    let policy = TierPolicy::clamped(2, 30.0);
    let n = s.n_gens;
    let prob_bytes = (s.num_unstable() as u64 + 2) * n as u64 * 8;
    let frontier = super::super::level_bytes(2, 1, n);
    let resident = policy.state_byte_cap() - prob_bytes - frontier;
    let deadline = Instant::now() + Duration::from_secs(30);
    let baseline = super::super::search_state(
        s.clone(),
        &frame,
        0.0,
        deadline,
        &policy,
        resident,
        FrozenDualStateMode::Shared,
        None,
        None,
        None,
        frame.n_unstable,
        None,
        super::super::FdTable::Full,
    )
    .unwrap();
    let mut cache = Cache::default();
    let actual = super::super::search_state(
        s.clone(),
        &frame,
        0.0,
        deadline,
        &policy,
        resident,
        FrozenDualStateMode::Shared,
        None,
        None,
        None,
        frame.n_unstable,
        Some((&mut cache, &frame)),
        super::super::FdTable::Full,
    )
    .unwrap();
    assert_eq!(actual.depth_cap, baseline.depth_cap);
    assert_eq!(
        (
            actual.search.verdict,
            actual.search.nodes,
            &actual.search.levels
        ),
        (
            baseline.search.verdict,
            baseline.search.nodes,
            &baseline.search.levels
        ),
    );
    assert_eq!(
        actual.search.nodes, 2,
        "the old admitted level must be searched"
    );
    assert_eq!(actual.search.relations.used, 0);
    assert!(
        actual.search.relations.work_reserved > 0,
        "inactive preparation must retain its actual inspection receipt"
    );
    assert!(cache.retired && cache.bytes() == 0 && cache.allowance.is_none());
    let later = cache
        .bind(&frame, &s, &[0, 1, 2, 3], None, MAX_BYTES)
        .unwrap();
    assert!(!later.active() && later.stats.refused == 1);
    assert_eq!(
        later.stats.work_reserved, 0,
        "retirement cannot reset the allowance"
    );
    assert!(cache.allowance.is_none() && cache.bytes() == 0);
}

#[test]
fn fd_relations_default_per_class_tier_arrives_and_keeps_every_witnessed_class_open() {
    let (net, frame) = compiled();
    let logits = dd_forward(&net, &[0.0]).unwrap();
    for class in [1, 2] {
        assert!((logits.last().unwrap()[0] - logits.last().unwrap()[class]).to_f64() < 0.0);
    }
    // No state selector: this is the ordinary declaration-default caller.
    let policy = TierPolicy::clamped(1024, 30.0).with_crosscheck_every(1);
    assert_eq!(policy.state, FrozenDualStateMode::PerClass);
    let report = super::super::run_tier_on_frame(
        &net,
        frame,
        &[1, 2],
        &[-1.0, -1.0],
        &[1, 2],
        None,
        &policy,
    );
    assert_eq!(report.classes.len(), 2);
    assert_eq!(report.classes_built, 2);
    assert_eq!((report.closed).len(), 0);
    let mut used = 0;
    let mut replayed = 0;
    for class in &report.classes {
        let run = class.result.as_ref().unwrap();
        assert_eq!(run.state, FrozenDualStateMode::PerClass);
        assert!(!run.streamed_rows);
        assert_ne!(run.search.verdict, SearchVerdict::Closed);
        assert!(!super::super::ledger_closes(&run.search.levels));
        used += run.search.relations.used;
        replayed += run.search.relations.replayed;
        assert_eq!(run.search.crosscheck_failures, 0);
        assert_eq!(run.search.crosschecks, run.search.certified_prunes);
    }
    assert!(
        used > 0 && replayed > 0,
        "ordinary caller must evaluate checked proposals on its original chart"
    );
}

#[test]
fn fd_relations_default_streamed_chart_skips_without_changing_the_state_selector() {
    let width = super::super::FROZEN_DUAL_MAX_DEPTH + 1;
    let net = TwinNet::compile(&TwinSpec {
        n_in: 1,
        ops: vec![
            scalar_conv(vec![1.0], vec![2.0]),
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Gemm {
                input: 2,
                weight: vec![1.0; width],
                bias: vec![-2.0; width],
                shape: (width, 1),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
            },
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Gemm {
                input: 4,
                weight: vec![0.0; 2 * width],
                bias: vec![0.0, 1.0],
                shape: (2, width),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
            },
        ],
    })
    .unwrap();
    let frame = FrozenDualFrame::new(
        &net,
        FrameInput {
            lo: vec![-1.0],
            hi: vec![1.0],
            trunk: vec![LayerInput {
                l: vec![1.0],
                u: vec![3.0],
                alpha: vec![1.0],
                s: vec![0.0],
            }],
            head: LayerInput {
                l: vec![-1.0; width],
                u: vec![1.0; width],
                alpha: vec![0.5; width],
                s: vec![0.5; width],
            },
        },
        0,
    )
    .unwrap();
    let original = dd_forward(&net, &[0.0]).unwrap();
    assert!((original.last().unwrap()[0] - original.last().unwrap()[1]).to_f64() < 0.0);
    let policy = TierPolicy::clamped(16, 30.0).with_crosscheck_every(1);
    let report = super::super::run_tier_on_frame(&net, frame, &[1], &[-1.0], &[1], None, &policy);
    assert_eq!((report.closed).len(), 0);
    let run = report.classes[0].result.as_ref().unwrap();
    assert_eq!(run.state, FrozenDualStateMode::PerClass);
    assert!(run.streamed_rows);
    assert_eq!(run.unstable, width);
    assert_eq!(run.depth_cap, super::super::FROZEN_DUAL_MAX_DEPTH);
    assert_eq!(run.build_columns, 1 + width);
    assert_eq!(
        run.search.relations,
        RelationStats::default(),
        "incomplete chart cannot silently enter full-chart relation replay"
    );
    assert!(run.search.nodes > 0 && !run.search.closed());
    assert_eq!(run.search.crosscheck_failures, 0);
}

/// Independent affine/ReLU fragment in the existing frozen-row carrier:
/// z0=x0, z1=100*x0-1/4, x0 in [-1,1]. All remaining generators have zero
/// coefficients. This deliberately replaces fixture row geometry; it is not
/// a certificate of the fixture's unchanged original network.
fn high_ratio_rows() -> FrozenDualState {
    let (net, frame) = compiled();
    let mut s = state(&net, &frame, 1);
    for row in [0, 1] {
        let start = row * s.n_gens;
        s.a[start..start + s.n_gens].fill(0.0);
        s.r[row] = 0.0;
    }
    s.a[0] = 1.0;
    s.a[s.n_gens] = 100.0;
    s.c_in[0] = 0.0;
    s.c_in[1] = -0.25;
    s
}

#[test]
fn fd_relations_weight_refinement_reaches_unclamped_original_rational_conflict() {
    let s = high_ratio_rows();
    for w in [0.9375, 0.5] {
        assert_eq!(
            (check_pair(&s, [0, 1], [w, 1.0 - w], None).unwrap()).len(),
            0
        );
    }
    let w = weights::propose(s.row(0), s.row(1), [0.0, -0.25], [0.0; 2], s.n_gens, None).unwrap();
    assert!(w > 0.9375 && w < 1.0);
    let found = check_pair(&s, [0, 1], [w, 1.0 - w], None).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].0, [Side::Off, Side::On]);
    let exact = q(1.0 - w) * q(0.25) - (q(w) - q(1.0 - w) * q(100.0)).abs();
    assert!(q(found[0].1) > Q::zero() && q(found[0].1) <= exact);
    // Three remaining phase combinations have concrete rational witnesses.
    for (x, phases) in [
        (-0.5, [Side::Off, Side::Off]),
        (1.0 / 1024.0, [Side::On, Side::Off]),
        (0.5, [Side::On, Side::On]),
    ] {
        for (z, side) in [q(x), q(100.0) * q(x) - q(0.25)].into_iter().zip(phases) {
            assert!(if side == Side::Off {
                z <= Q::zero()
            } else {
                z >= Q::zero()
            });
        }
    }
}

#[test]
fn fd_relations_weight_refinement_unsampled_coordinate_never_supplies_authority() {
    let mut s = high_ratio_rows();
    s.n_gens = 64;
    s.a = vec![0.0; s.num_unstable() * s.n_gens];
    s.a[0] = 1.0;
    s.a[64] = 100.0;
    s.a[65] = 1.0;
    // Evenly selected sample positions are 0,2,4,... . The additional
    // original x1 coefficient is absent in the proposal and required in replay.
    let a: Vec<_> = (0..SAMPLE).map(|i| s.row(0)[i * 2]).collect();
    let b: Vec<_> = (0..SAMPLE).map(|i| s.row(1)[i * 2]).collect();
    let w = weights::propose(&a, &b, [0.0, -0.25], [0.0; 2], 64, None).unwrap();
    assert_eq!(
        (check_pair(&s, [0, 1], [w, 1.0 - w], None).unwrap()).len(),
        0
    );
    // Every phase has a rational original witness; sample negativity must not
    // eliminate any of these four children.
    for sign0 in [-1.0, 1.0] {
        for sign1 in [-1.0, 1.0] {
            let x0 = q(sign0 / 1024.0);
            let z1 = q(100.0) * &x0 + q(sign1) - q(0.25);
            assert_eq!(x0 > Q::zero(), sign0 > 0.0);
            assert_eq!(z1 > Q::zero(), sign1 > 0.0);
        }
    }
}

#[test]
fn fd_relations_weight_refinement_optional_credit_denial_keeps_old_discovery() {
    let s = high_ratio_rows();
    let allowance = Allowance(AtomicU64::new(4096));
    let mut stats = RelationStats::default();
    let mut out = Vec::new();
    discover(&s, &[0, 1], &allowance, None, &mut stats, &mut out);
    assert!(out.is_empty());
    assert_eq!(stats.weight_refinements, 0);
    assert_eq!(stats.weight_replays, 0);
    assert!(stats.refused > 0 && stats.work_reserved > 0 && stats.work_reserved < 4096);
    let remaining = allowance.0.load(Ordering::Relaxed);
    discover(&s, &[0, 1], &allowance, None, &mut stats, &mut out);
    assert!(allowance.0.load(Ordering::Relaxed) <= remaining);
    assert!(stats.work_reserved <= 4096);
    assert!(out.is_empty());
}

#[test]
fn fd_relations_weight_refinement_default_discovery_keeps_caps_and_original_binding() {
    let s = high_ratio_rows();
    let allowance = Allowance(AtomicU64::new(TERMS));
    let mut stats = RelationStats::default();
    let mut out = Vec::new();
    discover(&s, &[0, 1], &allowance, None, &mut stats, &mut out);
    assert_eq!(stats.weight_refinements, 1);
    assert_eq!(stats.weight_replays, 1);
    assert_eq!(out.len(), 1);
    let c = &out[0];
    assert_eq!(c.rows, [0, 1]);
    assert_eq!(
        c.hashes,
        [
            row_hash(&s, 0, None).unwrap(),
            row_hash(&s, 1, None).unwrap()
        ]
    );
    let replay = check_pair(&s, c.rows, c.weights, None).unwrap();
    assert_eq!(replay, vec![(c.phases, c.contradiction)]);
    assert!(stats.work_reserved <= TERMS);
    assert!(out.capacity() * size_of::<Conflict>() < MAX_BYTES as usize);
}

#[test]
fn fd_relations_weight_refinement_deadline_poison_radius_and_ties_refuse() {
    let s = high_ratio_rows();
    let expired = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    assert!(weights::propose(
        s.row(0),
        s.row(1),
        [0.0, -0.25],
        [0.0; 2],
        s.n_gens,
        expired
    )
    .is_none());
    for (center, radius) in [
        ([0.0, 0.0], [0.0; 2]),
        ([f64::NAN, -0.25], [0.0; 2]),
        ([0.0, -0.25], [-0.1, 0.0]),
        ([0.0, -0.25], [0.0, 0.5]),
    ] {
        assert!(weights::propose(s.row(0), s.row(1), center, radius, s.n_gens, None).is_none());
    }
    assert!(weights::propose(&[1.0], &[f64::INFINITY], [0.0, -0.25], [0.0; 2], 1, None).is_none());
}

#[test]
fn fd_relations_weight_refinement_preserves_all_old_pairs_before_optional_credit() {
    let mut s = high_ratio_rows();
    s.n_gens = 100;
    s.a = vec![0.0; s.num_unstable() * 100];
    for row in 0..3 {
        s.a[row * 100 + row] = 1.0;
        s.c_in[row] = 0.0;
        s.r[row] = 0.0;
    }
    let allowance = Allowance(AtomicU64::new(9000));
    let mut stats = RelationStats::default();
    let mut out = Vec::new();
    discover(&s, &[0, 1, 2], &allowance, None, &mut stats, &mut out);
    assert_eq!(
        stats.attempted, 3,
        "all old pair checks fit before optional work"
    );
    let old_work = 3 * 32 + 3 * 3 * 32 * 8 + 3 * (6 * 100 + 256);
    assert_eq!(stats.work_reserved, old_work);
    assert_eq!(stats.weight_refinements, 0);
    assert_eq!(stats.weight_replays, 0);
    assert!(out.is_empty() && stats.refused == 3);
}

#[test]
fn fd_relations_weight_refinement_existing_conflicts_keep_original_binding_credit() {
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let allowance = Allowance(AtomicU64::new(TERMS));
    let mut stats = RelationStats::default();
    let mut out = Vec::new();
    discover(&s, &[0, 1], &allowance, None, &mut stats, &mut out);
    assert!(!out.is_empty());
    assert_eq!(stats.weight_refinements, 0);
    assert_eq!(stats.weight_replays, 0);
    for c in &out {
        assert!(check_pair(&s, c.rows, c.weights, None)
            .unwrap()
            .contains(&(c.phases, c.contradiction)));
    }
    let n = s.n_gens;
    let samples = SAMPLE.min(n);
    assert_eq!(
        stats.work_reserved,
        (2 * samples + 4 * samples * 8 + 6 * n + 256) as u64
    );
}

/// eng14 land (B2.3 x #frozen-dual-relations): a relation-certified prune
/// enters the class's B2.3 proof trace exactly as the tier certified it. The
/// relation only PROPOSES nonnegative sign-cut multipliers; the unchanged
/// `FrozenDualProblem::eval_node` certifies, and the trace must carry THAT
/// dual point (the proposal's, not the incumbent's) under the unchanged state
/// identity, so the checker re-derives the same computation. Every recorded
/// prune streams one `fd_prune` node whose `hint.fd` re-evaluates bit for bit
/// to its margin, on every evaluator; VOID unless a Relation prune arises.
#[test]
fn fd_relations_b2_trace_streams_relation_prunes_with_their_certified_dual_points() {
    use crate::certify::{NodeClose, RecordingSink};
    let (net, frame) = compiled();
    let s = state(&net, &frame, 1);
    let order = vec![0, 1, 2, 3];
    let prob = FrozenDualProblem::new(&s, &order).unwrap();
    for evaluator in [
        Evaluator::Node,
        Evaluator::Block { far_first: false },
        Evaluator::Block { far_first: true },
    ] {
        let hints = Cache::default()
            .bind(&frame, &s, &order, None, MAX_BYTES)
            .unwrap();
        let opts = SearchOptions {
            evaluator,
            node_cap: 1024,
            record_prunes: 1024,
            crosscheck_every: 1,
            ..SearchOptions::default()
        };
        let mut actual = super::super::search_with_relations(&prob, &opts, Some(&hints));
        let relation_prunes = replay(&prob, &actual);
        assert!(relation_prunes > 0, "VOID: no Relation prune arose");
        let recorded = actual.pruned.clone();
        assert_eq!(recorded.len() as u64, actual.certified_prunes);
        let trace = super::super::fd_proof_trace_for_test(
            &s,
            &frame,
            &order,
            &prob,
            &mut actual,
            super::super::FdTable::Full,
        );
        assert!(actual.pruned.is_empty(), "the trace MOVES the prunes");
        assert_eq!(trace.class(), s.class);
        let mut sink = RecordingSink::default();
        trace.stream_into(&[], &mut sink);
        assert_eq!(sink.fd_states.len(), 1);
        let st = &sink.fd_states[0];
        assert_eq!(st.prunes, recorded.len() as u64);
        assert_eq!(st.unstable, s.num_unstable());
        assert_eq!(st.order.len(), order.len());
        assert_eq!(sink.nodes.len(), recorded.len());
        let mut relation_nodes = 0;
        for (node, rec) in sink.nodes.iter().zip(&recorded) {
            assert_eq!(node.close, NodeClose::Bound);
            assert_eq!(node.src, "fd_prune");
            assert_eq!(node.dis.as_deref(), Some(&[s.class][..]));
            assert_eq!(node.lits.len(), rec.sides.len());
            for (lit, (i, side)) in node.lits.iter().zip(rec.sides.iter().enumerate()) {
                assert_eq!(lit.site, st.order[i]);
                assert_eq!(lit.positive, *side == Side::On);
            }
            let fd = node
                .fd
                .as_ref()
                .expect("every fd_prune carries its dual point");
            assert_eq!(fd.state, st.id);
            assert_eq!((&fd.l0, &fd.l1), (&rec.l0, &rec.l1));
            let again = prob.eval_node(
                &rec.sides,
                &fd.l0,
                &fd.l1,
                &mut NodeScratch::default(),
                None,
            );
            assert_eq!(again.certified.to_bits(), node.margin.to_bits());
            assert!(node.margin > 0.0);
            relation_nodes += usize::from(rec.candidate == Candidate::Relation);
        }
        assert_eq!(relation_nodes, relation_prunes);
    }
}
