// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::margin_row::zono_bab::{
    progressive_class_for_test, ClassReport, ClassRun, OpenReason, SearchVerdict, TierPolicy,
};
use crate::margin_row::zono_state::build::progressive::{estimate_bytes, ProgressiveClass};
use std::time::{Duration, Instant};

fn bits(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

fn options(deadline: Instant) -> FrozenDualOptions {
    FrozenDualOptions {
        deadline: Some(deadline),
        max_state_bytes: 256 << 20,
        chunk_cols: 4,
    }
}

fn duplicate_head(margin: f64) -> Fx {
    duplicate_heads(2, margin)
}

fn duplicate_heads(width: usize, margin: f64) -> Fx {
    assert!(width > 0 && width.is_multiple_of(2));
    let mut weights: Vec<f64> = (0..width)
        .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    weights.resize(2 * width, 0.0);
    let spec = TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::ChannelAffine {
                input: 0,
                scale: vec![1.0],
                shift: vec![2.0],
                scale_rel_err: 0.0,
                shift_err: vec![0.0],
                shape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Gemm {
                input: 2,
                weight: vec![1.0; width],
                bias: vec![-2.0; width],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (width, 1),
            },
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Gemm {
                input: 4,
                weight: weights,
                bias: vec![margin, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, width),
            },
        ],
    };
    let mut f = fx("duplicate-head", spec, 1.0, 123, false);
    f.t = 0;
    f.adv = vec![1];
    f
}

#[test]
fn progressive_complete_objective_global_generators_and_original_rows_match() {
    let mut checked = 0;
    for f in fixtures() {
        let frame = frame_of(&f);
        let class = f.adv[0];
        let full = state_of(&f, &frame, class);
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut p = ProgressiveClass::new(&f.net, &frame, class, &options(deadline), 7).unwrap();
        assert_eq!(p.state().num_unstable(), 0);
        loop {
            let s = p.state();
            assert_eq!(s.n_gens, full.n_gens);
            assert_eq!(s.n_in_gens, full.n_in_gens);
            bits(&s.d, &full.d);
            bits(&[s.c_obj, s.r_obj], &[full.c_obj, full.r_obj]);
            assert_eq!(p.original_rows().len(), s.num_unstable());
            for (i, &original) in p.original_rows().iter().enumerate() {
                bits(s.row(i), full.row(original));
                bits(
                    &[s.c_in[i], s.r[i], s.lam[i], s.mu[i], s.lo[i], s.hi[i]],
                    &[
                        full.c_in[original],
                        full.r[original],
                        full.lam[original],
                        full.mu[original],
                        full.lo[original],
                        full.hi[original],
                    ],
                );
                assert_eq!(
                    (s.enc[i], s.layer[i], s.neuron[i], s.site[i]),
                    (
                        full.enc[original],
                        full.layer[original],
                        full.neuron[original],
                        full.site[original]
                    )
                );
                checked += 1;
            }
            let mut table = p.compact_relax();
            assert_eq!(table.len(), frame.n_unstable);
            assert_eq!(table.verify(&f.net, &frame, s, None), Ok(()));
            if !p.advance().unwrap() {
                break;
            }
        }
        assert!(p.state().num_unstable() <= 7);
    }
    assert!(checked > 0, "no original rows compared");
}

/// The global generator vector is reconstructed from ALL original source
/// activations and the complete class relaxation, including omitted rows.
/// A compact state's own point checker cannot reconstruct those omissions.
fn exact_prefix(
    f: &Fx,
    frame: &FrozenDualFrame,
    full: &FrozenDualState,
    prefix: &FrozenDualState,
    x: &[f64],
    pert: Pert,
) {
    let values = exact_forward(&f.spec, x, pert);
    let mut e = vec![Q::zero(); full.n_gens];
    for (gi, &dim) in frame.in_cols.iter().enumerate() {
        e[gi] = (q(x[dim]) - q(frame.mid[dim])) / q(frame.rad[dim]);
    }
    for k in 0..full.num_unstable() {
        let layer = &frame.layers[full.layer[k]];
        let z = &values[layer.input][full.neuron[k]];
        let relu = if z.is_positive() {
            z.clone()
        } else {
            Q::zero()
        };
        e[full.enc[k]] = (relu - q(full.lam[k]) * z - q(full.mu[k])) / q(full.mu[k]);
    }
    assert!(e.iter().all(|v| v.abs() <= qi(1)));
    let margin = &values[f.spec.ops.len()][prefix.t] - &values[f.spec.ops.len()][prefix.class];
    let predicted = q(prefix.c_obj) + prefix.d.iter().zip(&e).map(|(a, e)| q(*a) * e).sum::<Q>();
    assert!(
        (&margin - predicted).abs() <= q(prefix.r_obj),
        "{} margin",
        f.name
    );
    let mut sides = Vec::new();
    for i in 0..prefix.num_unstable() {
        let layer = &frame.layers[prefix.layer[i]];
        let z = &values[layer.input][prefix.neuron[i]];
        let predicted = q(prefix.c_in[i])
            + prefix
                .row(i)
                .iter()
                .zip(&e)
                .map(|(a, e)| q(*a) * e)
                .sum::<Q>();
        assert!((z - predicted).abs() <= q(prefix.r[i]), "{} row{i}", f.name);
        sides.push(if z.is_positive() { Side::On } else { Side::Off });
    }
    if !sides.is_empty() {
        let prob = FrozenDualProblem::new(prefix, &(0..sides.len()).collect::<Vec<_>>()).unwrap();
        let bound = prob.eval_node(
            &sides,
            &vec![0.25; sides.len()],
            &vec![0.75; sides.len()],
            &mut NodeScratch::default(),
            None,
        );
        if bound.certified.is_finite() {
            assert!(q(bound.certified) <= margin, "{} node", f.name);
        } else {
            assert_eq!(bound.certified, f64::NEG_INFINITY);
        }
    }
}

#[test]
fn progressive_prefix_contains_exact_original_fold_ball_points_and_nodes() {
    let mut points = 0;
    for f in fixtures() {
        let frame = frame_of(&f);
        let full = state_of(&f, &frame, f.adv[0]);
        let mut p = ProgressiveClass::new(
            &f.net,
            &frame,
            f.adv[0],
            &options(Instant::now() + Duration::from_secs(60)),
            3,
        )
        .unwrap();
        loop {
            for pert in [Pert::Stored, Pert::Up, Pert::Mixed(7)] {
                exact_prefix(&f, &frame, &full, p.state(), &f.lo, pert);
                points += 1;
            }
            if !p.advance().unwrap() {
                break;
            }
        }
    }
    assert!(points >= 30, "no omitted-generator/fold-ball controls");
}

#[test]
fn progressive_batch_refusal_is_atomic_and_does_not_replace_the_chart() {
    let f = fixture0();
    let frame = frame_of(&f);
    let mut p = ProgressiveClass::new(
        &f.net,
        &frame,
        f.adv[0],
        &options(Instant::now() + Duration::from_secs(30)),
        8,
    )
    .unwrap();
    assert!(p.advance().unwrap());
    let before = p.state().clone();
    let ids = p.original_rows().to_vec();
    assert!(matches!(
        p.cancel_before_commit_for_test(),
        Err(ZonoRefusal::Deadline)
    ));
    assert_eq!(p.state(), &before);
    assert_eq!(p.original_rows(), ids);
    let mut table = p.compact_relax();
    assert_eq!(table.verify(&f.net, &frame, p.state(), None), Ok(()));
    assert!(p.advance().unwrap());
    assert!(p.state().num_unstable() > before.num_unstable());
    let mut ids = p.original_rows().to_vec();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), p.state().num_unstable());
}

#[test]
fn progressive_admission_deadline_and_resident_snapshot_memory_are_bounded() {
    let f = fixture0();
    let frame = frame_of(&f);
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut opts = options(deadline);
    opts.max_state_bytes = 0;
    assert!(matches!(
        ProgressiveClass::new(&f.net, &frame, f.adv[0], &opts, 8),
        Err(ZonoRefusal::Memory { .. })
    ));
    opts = options(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    assert!(matches!(
        ProgressiveClass::new(&f.net, &frame, f.adv[0], &opts, 8),
        Err(ZonoRefusal::Deadline)
    ));
    opts = options(deadline);
    let estimate = estimate_bytes(&frame, &opts, 8);
    assert!(estimate >= estimate_bytes(&frame, &opts, 1));
    let mut p = ProgressiveClass::new(&f.net, &frame, f.adv[0], &opts, 8).unwrap();
    loop {
        assert!(estimate >= p.resident_bytes().saturating_add(2 * p.state().bytes()));
        if !p.advance().unwrap() {
            break;
        }
    }
}

struct Capture;
impl Drop for Capture {
    fn drop(&mut self) {
        crate::certify::force_hint_capture_for_test(None);
    }
}
fn capture() -> Capture {
    crate::certify::force_hint_capture_for_test(Some(true));
    Capture
}

fn run(f: &Fx, nodes: u64) -> ClassRun {
    let frame = frame_of(f);
    let deadline = Instant::now() + Duration::from_secs(30);
    progressive_class_for_test(
        &f.net,
        &frame,
        1,
        &options(deadline),
        deadline,
        &TierPolicy::clamped(nodes, 0.0),
    )
    .unwrap()
}

#[test]
fn progressive_owner_closes_a_source_relation_with_a_complete_final_cover() {
    let _capture = capture();
    let f = duplicate_head(0.2);
    let result = run(&f, 6);
    assert!(result.search.closed());
    assert!(crate::margin_row::zono_bab::ledger_closes(
        &result.search.levels
    ));
    let stats = result.progressive.as_ref().unwrap();
    assert!(stats.stages >= 2);
    assert_eq!(stats.rows_built, 2);
    assert!(stats.nodes_charged <= 6);
    assert_eq!(stats.nodes_charged, result.search.nodes);
    assert_eq!(stats.unobserved_node_allowance, 0);
    assert!(result.proof.as_ref().unwrap().complete());
    // Replay every streamed FINAL-stage prune against a freshly rebuilt
    // full original state, with its order recovered from absolute source
    // sites. No previous chart's prunes may be present in this trace.
    let frame = frame_of(&f);
    let full = state_of(&f, &frame, 1);
    let mut sink = crate::certify::RecordingSink::default();
    (**result.proof.as_ref().unwrap())
        .clone()
        .stream_into(&[], &mut sink);
    assert_eq!(sink.fd_states.len(), 1);
    let record = &sink.fd_states[0];
    assert_eq!(record.relaxed, frame.n_unstable);
    assert_eq!(sink.nodes.len() as u64, result.search.certified_prunes);
    assert_ne!((sink.nodes).len(), 0);
    let order: Vec<usize> = record
        .order
        .iter()
        .map(|site| {
            (0..full.num_unstable())
                .find(|&k| {
                    frame.layers[full.layer[k]].op == site.op as usize
                        && full.neuron[k] == site.index as usize
                })
                .unwrap()
        })
        .collect();
    let problem = FrozenDualProblem::new(&full, &order).unwrap();
    for node in &sink.nodes {
        let hint = node.fd.as_ref().unwrap();
        let sides: Vec<Side> = node
            .lits
            .iter()
            .enumerate()
            .map(|(i, lit)| {
                assert_eq!(lit.site, record.order[i]);
                if lit.positive {
                    Side::On
                } else {
                    Side::Off
                }
            })
            .collect();
        let bound = problem.eval_node(
            &sides,
            &hint.l0,
            &hint.l1,
            &mut NodeScratch::default(),
            None,
        );
        assert!(bound.prunes());
        assert_eq!(bound.certified.to_bits(), node.margin.to_bits());
    }
    let report = ClassReport {
        class: 1,
        lane_root: -1.0,
        result: Ok(result),
    };
    assert!(crate::certify::margin_row::frozen_dual_class(&report).is_some());
}

#[test]
fn progressive_owner_root_close_keeps_all_omitted_relaxation_entries() {
    let _capture = capture();
    let f = duplicate_head(2.0);
    let frame = frame_of(&f);
    assert!(frame.n_unstable > 0);
    let result = run(&f, 128);
    assert!(result.search.closed() && result.search.root_certified > 0.0);
    assert_eq!((result.search.levels).len(), 0);
    let stats = result.progressive.as_ref().unwrap();
    assert_eq!(stats.rows_built, 0);
    assert_eq!(stats.attempted_columns, 1);
    assert_eq!(stats.nodes_charged, 0);
    let proof = result.proof.as_ref().unwrap();
    assert!(proof.complete());
    assert_eq!(proof.relaxed(), frame.n_unstable);
}

#[test]
fn progressive_owner_preserves_an_exact_unsafe_witness_and_one_node_allowance() {
    let _capture = capture();
    let f = duplicate_head(-0.2);
    let values = exact_forward(&f.spec, &f.lo, Pert::Stored);
    assert!(values[f.spec.ops.len()][0] < values[f.spec.ops.len()][1]);
    let mut result = run(&f, 5);
    assert!(!result.search.closed());
    let stats = result.progressive.as_ref().unwrap();
    assert!(stats.stages >= 2);
    assert!(stats.nodes_charged + stats.unobserved_node_allowance <= 5);
    assert_eq!(
        result.search.verdict,
        SearchVerdict::Open(OpenReason::NodeCap)
    );
    assert!(!result.proof.as_ref().unwrap().complete());
    // Telemetry cannot substitute for a complete original final-search cover.
    result.progressive.as_mut().unwrap().stop = "closed";
    let report = ClassReport {
        class: 1,
        lane_root: 1.0,
        result: Ok(result),
    };
    assert!(crate::certify::margin_row::frozen_dual_class(&report).is_none());
}

#[test]
fn progressive_checked_frontier_is_the_same_single_bfs_as_eager_original_rows() {
    use crate::margin_row::zono_bab::{search, search_growing_for_test, SearchOptions};
    let mut extended = 0;
    for f in fixtures() {
        let frame = frame_of(&f);
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut p = ProgressiveClass::new(&f.net, &frame, f.adv[0], &options(deadline), 7).unwrap();
        if !p.advance().unwrap() {
            continue;
        }
        let first = FrozenDualProblem::new(
            p.state(),
            &(0..p.state().num_unstable()).collect::<Vec<_>>(),
        )
        .unwrap();
        while p.advance().unwrap() {}
        let final_problem = FrozenDualProblem::new(
            p.state(),
            &(0..p.state().num_unstable()).collect::<Vec<_>>(),
        )
        .unwrap();
        let opts = SearchOptions {
            node_cap: 128,
            record_prunes: 128,
            ..SearchOptions::default()
        };
        let eager = search(&final_problem, &opts);
        let mut next = Some(final_problem);
        let mut offered = 0;
        let (_, growing) = search_growing_for_test(first, &opts, &mut |_| {
            offered += 1;
            Ok(next.take())
        });
        assert_eq!(growing.verdict, eager.verdict);
        assert_eq!(growing.nodes, eager.nodes);
        assert_eq!(growing.levels, eager.levels);
        assert_eq!(growing.certified_prunes, eager.certified_prunes);
        assert_eq!(
            growing.root_certified.to_bits(),
            eager.root_certified.to_bits()
        );
        assert_eq!(
            growing.cert_min_slack.to_bits(),
            eager.cert_min_slack.to_bits()
        );
        if offered > 0 {
            extended += 1;
        }
    }
    assert!(
        extended > 0,
        "no frontier actually crossed an append boundary"
    );
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep the original fixture owner intact while mutating an independent candidate."
)]
fn progressive_changed_problem_cannot_resume_or_close_an_original_unsafe_class() {
    use crate::margin_row::zono_bab::{search_growing_for_test, SearchOptions};
    let f = duplicate_head(-0.2);
    let frame = frame_of(&f);
    let full = state_of(&f, &frame, 1);
    let first = FrozenDualProblem::new(&full, &[0]).unwrap();
    let mut forged = full.clone();
    forged.c_obj += 4.0;
    let bad = FrozenDualProblem::new(&forged, &[0, 1]).unwrap();
    assert!(bad.root_bound() > 0.0);
    let mut proposed = Some(bad);
    let mut calls = 0;
    let (unchanged, result) = search_growing_for_test(
        first,
        &SearchOptions {
            node_cap: 6,
            ..SearchOptions::default()
        },
        &mut |_| {
            calls += 1;
            Ok(proposed.take())
        },
    );
    assert_eq!(calls, 1);
    assert_eq!(unchanged.max_depth(), 1);
    assert_eq!(result.verdict, SearchVerdict::Open(OpenReason::Refused));
    assert_eq!(result.nodes, 2);
    assert!(!result.closed());
}

#[test]
fn progressive_optional_proof_records_do_not_refuse_a_large_certified_root() {
    let _capture = capture();
    let f = duplicate_heads(600, 1000.0);
    let frame = frame_of(&f);
    assert_eq!(frame.n_unstable, 600);
    let result = run(&f, 8192);
    assert!(result.search.closed() && result.search.root_certified > 0.0);
    assert_eq!(result.search.nodes, 0);
    assert_eq!((result.search.levels).len(), 0);
    let stats = result.progressive.as_ref().unwrap();
    assert_eq!(stats.rows_built, 0);
    let proof = result.proof.as_ref().unwrap();
    assert!(proof.complete());
    assert_eq!(proof.relaxed(), 600);
    let report = ClassReport {
        class: 1,
        lane_root: -1.0,
        result: Ok(result),
    };
    assert!(crate::certify::margin_row::frozen_dual_class(&report).is_some());
}

#[test]
fn lazy_canonical_preparation_matches_original_ranked_state_and_fixed_quantum() {
    let mut checked = 0;
    for f in fixtures() {
        let frame = frame_of(&f);
        let deadline = Instant::now() + Duration::from_secs(60);
        let opts = options(deadline);
        let class = f.adv[0];
        let mut p = ProgressiveClass::new(&f.net, &frame, class, &opts, 7).unwrap();
        if !p.advance().unwrap() {
            continue;
        }
        let canonical = FrozenDualState::build_ranked(&f.net, &frame, class, &opts, 7).unwrap();
        assert!(p.canonical_ranked().unwrap());
        assert_eq!(p.original_rows(), canonical.original_rows);
        let s = p.state();
        let t = &canonical.state;
        bits(&s.d, &t.d);
        bits(&[s.c_obj, s.r_obj], &[t.c_obj, t.r_obj]);
        bits(&s.a, &t.a);
        bits(&s.c_in, &t.c_in);
        bits(&s.r, &t.r);
        bits(&s.lam, &t.lam);
        bits(&s.mu, &t.mu);
        assert_eq!(s.enc, t.enc);
        assert_eq!(s.layer, t.layer);
        assert_eq!(s.neuron, t.neuron);
        let mut table = p.compact_relax();
        assert_eq!(table.verify(&f.net, &frame, s, None), Ok(()));
        assert!(p.columns_complete());
        assert!(!p.canonical_ranked().unwrap());
        checked += 1;
    }
    assert!(checked > 0);
}

#[test]
fn lazy_canonical_publication_is_atomic_on_late_refusal() {
    let f = fixture0();
    let frame = frame_of(&f);
    let mut p = ProgressiveClass::new(
        &f.net,
        &frame,
        f.adv[0],
        &options(Instant::now() + Duration::from_secs(60)),
        7,
    )
    .unwrap();
    assert!(p.advance().unwrap());
    let before = p.state().clone();
    let ids = p.original_rows().to_vec();
    assert!(matches!(
        p.cancel_canonical_before_commit_for_test(),
        Err(ZonoRefusal::Deadline)
    ));
    assert_eq!(p.state(), &before);
    assert_eq!(p.original_rows(), ids);
    assert!(p.canonical_ranked().unwrap());
    let mut table = p.compact_relax();
    assert_eq!(table.verify(&f.net, &frame, p.state(), None), Ok(()));
}

/// A large but already-consistent nonnegative ReLU dominates raw |d|. The
/// canonical halfspace scorer instead chooses the correlated duplicate pair.
fn misleading_magnitude_head() -> Fx {
    let spec = TwinSpec {
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
                weight: vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                bias: vec![-2.75, -2.0, -2.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (3, 2),
            },
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Gemm {
                input: 4,
                weight: vec![1000.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                bias: vec![0.2, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 3),
            },
        ],
    };
    let mut f = fx("misleading-positive-head", spec, 1.0, 123, false);
    f.t = 0;
    f.adv = vec![1];
    f
}

#[test]
fn lazy_canonical_restart_discards_old_cover_and_keeps_one_node_allowance() {
    let _capture = capture();
    let f = misleading_magnitude_head();
    let frame = frame_of(&f);
    assert_eq!(frame.n_unstable, 3);
    let opts = options(Instant::now() + Duration::from_secs(60));
    let mut p = ProgressiveClass::new(&f.net, &frame, 1, &opts, 512).unwrap();
    assert!(p.advance().unwrap());
    assert_eq!(p.original_rows(), [0]);
    let canonical = FrozenDualState::build_ranked(&f.net, &frame, 1, &opts, 512).unwrap();
    assert_ne!(
        canonical.original_rows[0], 0,
        "the canonical chart must really change the first site"
    );
    // The heuristic search is incomplete; derive its actual completed
    // canonical trajectory instead of assuming it closes this source in six
    // nodes. The lazy plan must reproduce that trajectory after paying two
    // extra cheap-plan nodes, with no renewal of the original allowance.
    use crate::margin_row::zono_bab::{search, SearchOptions};
    let problem = FrozenDualProblem::new(
        &canonical.state,
        &(0..canonical.state.num_unstable()).collect::<Vec<_>>(),
    )
    .unwrap();
    let eager = search(
        &problem,
        &SearchOptions {
            node_cap: 64,
            record_prunes: 64,
            ..SearchOptions::default()
        },
    );
    assert!(eager.nodes > 0 && eager.verdict != SearchVerdict::Open(OpenReason::NodeCap));
    let allowance = eager.nodes + 2;
    let result = run(&f, allowance);
    assert_eq!(result.search.verdict, eager.verdict);
    assert_eq!(result.search.levels, eager.levels);
    assert_eq!(result.search.nodes, eager.nodes);
    assert_eq!(result.search.certified_prunes, eager.certified_prunes);
    let stats = result.progressive.as_ref().unwrap();
    assert_eq!(stats.restarts, 1);
    assert_eq!(stats.nodes_charged, allowance);
    assert!(stats.columns_complete);
    assert_eq!(result.search.levels[0].frontier, 2);
    let proof = result.proof.as_ref().unwrap();
    assert_eq!(proof.complete(), eager.closed());
    if eager.closed() {
        let mut sink = crate::certify::RecordingSink::default();
        (**proof).clone().stream_into(&[], &mut sink);
        assert_eq!(sink.fd_states.len(), 1);
        assert_ne!((sink.nodes).len(), 0);
        assert_eq!(sink.fd_states[0].prunes, result.search.certified_prunes);
        // Every final path comes from the canonical chart, never a cheap
        // plan's cover. Their first site is the canonical first site.
        for node in &sink.nodes {
            assert_eq!(node.lits[0].site, sink.fd_states[0].order[0]);
        }
    }
    let tight = run(&f, allowance - 1);
    assert!(!tight.search.closed());
    assert_eq!(
        tight.search.verdict,
        SearchVerdict::Open(OpenReason::NodeCap)
    );
    let stats = tight.progressive.as_ref().unwrap();
    assert_eq!(stats.restarts, 1);
    assert!(stats.nodes_charged < allowance);
    assert!(!tight.proof.as_ref().unwrap().complete());
}
