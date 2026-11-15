// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.2 lane proof tapes: the capture's arming rule, the translation into
//! `ny-proof-bundle/v2` nodes, and the REAL drivers (graph ReLU split, graph
//! input split, multi-objective, lowdim) — each must deposit a tape exactly
//! when the capture is armed and return BIT-IDENTICAL results either way (the
//! Competition profile arms nothing).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::{arr1, Array1, Array2};

use super::graph::CAPTURE_TEST_LOCK;
use super::lane_proof::{
    arm_lane_proof_capture_for_test, lane_tape_armed, reset_lane_proof_capture, stream_lane_proofs,
    take_lane_proof_capture, LaneProofCapture, LaneProofInstance, TapeClose, TapeRecorder,
};
use super::sink::{force_hint_capture_for_test, NodeClose, RecordingSink};
use super::{GraphInstanceProperty, GraphObjective};
use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::{BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier};
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

fn lock() -> std::sync::MutexGuard<'static, ()> {
    CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `fc0 (2 -> 2) -> relu0 -> fc1 (2 -> outs)`.
fn mlp(w0: [[f32; 2]; 2], w1: &[Vec<f32>], b1: &[f32]) -> GraphNetwork {
    let mut g = GraphNetwork::new();
    let l0 = LinearLayer::new(
        Array2::from_shape_vec((2, 2), w0.iter().flatten().copied().collect()).expect("shape"),
        Some(Array1::from(vec![0.0f32, 0.0])),
    )
    .expect("finite");
    g.add_node(GraphNode::from_input("fc0".to_string(), Layer::Linear(l0)));
    g.add_node(GraphNode::new(
        "relu0".to_string(),
        Layer::ReLU(ReLULayer),
        vec!["fc0".into()],
    ));
    let l1 = LinearLayer::new(
        Array2::from_shape_vec((w1.len(), 2), w1.iter().flatten().copied().collect())
            .expect("shape"),
        Some(Array1::from(b1.to_vec())),
    )
    .expect("finite");
    g.add_node(GraphNode::new(
        "fc1".to_string(),
        Layer::Linear(l1),
        vec!["relu0".into()],
    ));
    g.set_output("fc1".to_string());
    g
}

/// `y = 1 − relu(x0 + x1) − relu(x0 − x1)` on `[−1, 1]²`: true minimum `−1`,
/// root CROWN `−2`, so a threshold of `−1.25` needs a split tree.
fn crossing() -> GraphNetwork {
    mlp([[1.0, 1.0], [1.0, -1.0]], &[vec![-1.0, -1.0]], &[1.0])
}

/// Two outputs: `y0` as [`crossing`], `y1 = 1 + relu(x0 + x1)` (minimum 1).
fn crossing2() -> GraphNetwork {
    mlp(
        [[1.0, 1.0], [1.0, -1.0]],
        &[vec![-1.0, -1.0], vec![1.0, 0.0]],
        &[1.0, 1.0],
    )
}

fn unit_box() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite box")
}

const THR: f32 = -1.25;

fn instance(rows: &[(&[f32], f32)]) -> LaneProofInstance {
    let clauses = rows
        .iter()
        .map(|&(r, t)| vec![GraphObjective::new(r, t, false).expect("finite")])
        .collect();
    let property =
        GraphInstanceProperty::new(vec![-1.0, -1.0], vec![1.0, 1.0], clauses).expect("well formed");
    let mut sites = HashMap::new();
    sites.insert("relu0".to_string(), "z".to_string());
    LaneProofInstance::new(&property, sites)
}

/// Take the process capture, keeping only THIS test's tapes (those carrying
/// the objective `(row, threshold)`): the capture is process-global, so lanes
/// that other tests of this binary run while a test has it armed deposit
/// into it too (review(b2-lanes)).
fn take_own(row: &[f32], threshold: f32) -> LaneProofCapture {
    let mut capture = take_lane_proof_capture();
    capture.retain_objectives(&[(row, threshold)]);
    capture
}

fn stream(capture: &LaneProofCapture, inst: &LaneProofInstance) -> RecordingSink {
    let mut sink = RecordingSink::default();
    stream_lane_proofs(capture, inst, &mut sink);
    sink
}

fn split(node: &str, i: usize, active: bool) -> GraphNeuronConstraint {
    GraphNeuronConstraint::new(node.to_string(), i, active, 1.0).expect("finite score")
}

fn history(splits: &[(usize, bool)]) -> GraphSplitHistory {
    let mut h = GraphSplitHistory::new();
    for &(i, a) in splits {
        h.add_constraint(split("relu0", i, a));
    }
    h
}

/// The capture arms ONLY under hint capture (the entry point's Proof arm):
/// under Competition no lane can build a tape.
#[test]
fn lane_capture_arms_only_under_hint_capture() {
    let _g = lock();
    force_hint_capture_for_test(Some(false));
    reset_lane_proof_capture();
    assert!(!lane_tape_armed(), "Competition must arm nothing");
    let input = unit_box();
    assert!(
        TapeRecorder::for_graph("t", &crossing(), &input, &[(&[1.0], THR, false)]).is_none(),
        "no recorder without an armed capture"
    );
    assert!(take_lane_proof_capture().is_empty());
    force_hint_capture_for_test(Some(true));
    reset_lane_proof_capture();
    assert!(lane_tape_armed());
    let _ = take_lane_proof_capture();
    assert!(!lane_tape_armed(), "take disarms");
    force_hint_capture_for_test(None);
}

/// Translation: split literals become ABSOLUTE site literals (the ONNX
/// tensor the entry point resolved, the neuron index, the side) with the
/// lane's β; each leaf discharges exactly the clauses its objective ENTAILS;
/// an emptiness leaf claims every clause; an untranslatable leaf is an opaque
/// claim; a COMPLETE tape needs no fallback, an incomplete one gets one
/// opaque claim per clause.
#[test]
fn lane_tape_translation_names_absolute_sites_and_entailed_clauses() {
    let _g = lock();
    arm_lane_proof_capture_for_test();
    let input = unit_box();
    let graph = crossing();
    let tape = TapeRecorder::for_graph("relu_split", &graph, &input, &[(&[1.0], THR, false)])
        .expect("armed");
    let beta = crate::beta_crown::state::GraphBetaState::from_entries(vec![
        crate::beta_crown::state::GraphBetaEntry::new("relu0".into(), 1, 0.0, 0.75, 1.0)
            .expect("entry"),
    ]);
    tape.note_splits(
        &history(&[(0, true), (1, true)]),
        None,
        Some(&beta),
        TapeClose::Bound,
        vec![vec![0]],
        0.25,
        false,
    );
    tape.note_splits(
        &history(&[(0, true)]),
        Some(&split("relu0", 1, false)),
        None,
        TapeClose::Empty,
        vec![vec![0]],
        f64::NAN,
        false,
    );
    tape.note_splits(
        &history(&[(0, false)]),
        Some(&split("not_a_node", 0, true)),
        None,
        TapeClose::Bound,
        vec![vec![0]],
        0.5,
        false,
    );
    tape.deposit(true);
    let capture = take_own(&[1.0], THR);
    assert_eq!(capture.len(), 1);
    // Clause 0 is this objective; clause 1 (a different row) is not entailed.
    let inst = instance(&[(&[1.0], THR), (&[1.0], 5.0)]);
    let sink = stream(&capture, &inst);
    assert_eq!(sink.sites.len(), 1);
    assert_eq!(sink.sites[0].tensors.len(), 1);
    assert_eq!(sink.sites[0].tensors[0].name, "z");
    assert_eq!(
        sink.sites[0].tensors[0].size, 2,
        "the Relu's width, from the IBP pass"
    );
    let n0 = &sink.nodes[0];
    assert_eq!(n0.close, NodeClose::Bound);
    assert_eq!(n0.dis.as_deref(), Some(&[0usize][..]));
    let lits: Vec<(u32, u32, bool)> = n0
        .lits
        .iter()
        .map(|l| (l.site.op, l.site.index, l.positive))
        .collect();
    assert_eq!(lits, [(0, 0, true), (0, 1, true)]);
    assert_eq!(n0.beta, [0.0, 0.75], "β aligned with the literals");
    assert_eq!(n0.src, "relu_split");
    let n1 = &sink.nodes[1];
    assert_eq!(n1.close, NodeClose::Empty);
    assert_eq!(n1.dis, None, "an emptiness leaf claims every clause");
    assert_eq!(n1.lits.len(), 2);
    let n2 = &sink.nodes[2];
    assert_eq!(
        n2.close,
        NodeClose::Opaque,
        "an unknown node is an opaque claim"
    );
    assert_eq!(n2.dis.as_deref(), Some(&[0usize][..]));
    // Clause 1: no complete tape discharges it -> one opaque fallback.
    assert_eq!(sink.nodes.len(), 4, "{:?}", sink.nodes);
    assert_eq!(sink.nodes[3].close, NodeClose::Opaque);
    assert_eq!(sink.nodes[3].dis.as_deref(), Some(&[1usize][..]));
    assert_eq!(sink.nodes[3].src, "lane_claim");
    // The same tape deposited INCOMPLETE: clause 0 gets a fallback too.
    arm_lane_proof_capture_for_test();
    let tape = TapeRecorder::for_graph("relu_split", &graph, &input, &[(&[1.0], THR, false)])
        .expect("armed");
    tape.note_splits(
        &history(&[]),
        None,
        None,
        TapeClose::Bound,
        vec![vec![0]],
        0.1,
        false,
    );
    tape.deposit(false);
    let sink = stream(&take_own(&[1.0], THR), &instance(&[(&[1.0], THR)]));
    assert_eq!(sink.nodes.len(), 2);
    assert_eq!(sink.nodes[1].src, "lane_claim");
}

/// Box leaves become input-threshold literals at the faces STRICTLY inside
/// the declared box (a face on the boundary is dropped, so the leaf extends
/// to the checker's own region boundary); a leaf outside the declared box is
/// dropped; a clip records its excluded slabs; a lowdim-style ANY group
/// discharges a clause only when every objective of the group entails it.
#[test]
fn lane_tape_boxes_become_input_threshold_literals() {
    let _g = lock();
    arm_lane_proof_capture_for_test();
    let tape = TapeRecorder::for_boxes(
        "lowdim",
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &[vec![1.0], vec![2.0]],
        &[f64::from(THR), 0.0],
    )
    .expect("armed");
    tape.note_box(
        &[-1.0, -1.0],
        &[0.0, 1.0],
        TapeClose::Bound,
        vec![vec![0]],
        0.5,
    );
    tape.note_box(
        &[0.0, -1.0],
        &[1.0, 1.0],
        TapeClose::Clip,
        vec![vec![0, 1]],
        0.5,
    );
    tape.note_box(
        &[2.0, -1.0],
        &[3.0, 1.0],
        TapeClose::Bound,
        vec![vec![0]],
        0.5,
    );
    tape.note_clip(
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &[-0.5, -1.0],
        &[1.0, 0.5],
        &[vec![0]],
    );
    tape.deposit(true);
    // Clause 0 = objective 0; objective 1 ([2]) entails no clause.
    let sink = stream(&take_own(&[1.0], THR), &instance(&[(&[1.0], THR)]));
    let xl = |n: &super::sink::NodeRecord| -> Vec<(u32, f64, bool)> {
        n.xlits
            .iter()
            .map(|l| (l.index, l.threshold, l.positive))
            .collect()
    };
    assert_eq!(xl(&sink.nodes[0]), [(0, 0.0, false)]);
    assert_eq!((sink.nodes[0].lits).len(), 0);
    // The ANY group {0, 1}: objective 1 entails nothing, so the group proves
    // no clause and the leaf is not streamed; the outside leaf neither. The
    // clip's two slabs follow.
    assert_eq!(sink.nodes.len(), 3 + 1, "{:?}", sink.nodes);
    assert_eq!(xl(&sink.nodes[1]), [(0, -0.5, false)]);
    assert_eq!(xl(&sink.nodes[2]), [(0, -0.5, true), (1, 0.5, true)]);
    // The tape's leaves do not all discharge clause 0: fallback.
    assert_eq!(sink.nodes[3].src, "lane_claim");
    assert_eq!((sink.sites[0].tensors).len(), 0);
}

fn same(a: &crate::beta_crown::BetaCrownResult, b: &crate::beta_crown::BetaCrownResult) -> bool {
    format!("{:?}", a.result) == format!("{:?}", b.result)
        && a.domains_explored == b.domains_explored
        && a.domains_verified == b.domains_verified
        && a.max_depth_reached == b.max_depth_reached
        && a.cuts_generated == b.cuts_generated
        && format!("{:?}", a.output_bounds) == format!("{:?}", b.output_bounds)
}

fn config() -> BetaCrownConfig {
    BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 2000,
        max_depth: 24,
        batch_size: 1,
        timeout: Duration::from_secs(20),
        ..Default::default()
    }
}

/// Run `f` disarmed (the Competition profile) and armed; the results must be
/// bit-identical, the disarmed capture empty, the armed one hold a tape that
/// covers the instance (no fallback) with more than one leaf.
fn identical_and_taped(
    lane: &str,
    inst: &LaneProofInstance,
    own: (&[f32], f32),
    min_nodes: u64,
    f: impl Fn() -> crate::beta_crown::BetaCrownResult,
) {
    let _g = lock();
    force_hint_capture_for_test(Some(false));
    reset_lane_proof_capture();
    let off = f();
    let off_capture = take_lane_proof_capture();
    force_hint_capture_for_test(None);
    assert!(
        off_capture.is_empty(),
        "{lane}: Competition deposited a tape"
    );
    arm_lane_proof_capture_for_test();
    let on = f();
    let capture = take_own(own.0, own.1);
    assert!(
        same(&off, &on),
        "{lane}: the tape changed the verdict path: {off:?} vs {on:?}"
    );
    assert!(
        matches!(on.result, BabVerificationStatus::Verified),
        "{lane}: the fixture must verify: {on:?}"
    );
    assert!(!capture.is_empty(), "{lane}: no tape deposited");
    let mut sink = RecordingSink::default();
    let summary = stream_lane_proofs(&capture, inst, &mut sink);
    assert!(
        summary.nodes >= min_nodes,
        "{lane}: {summary:?} {:?}",
        sink.nodes
    );
    eprintln!("[lane-proof-test] lane={lane} {summary:?}");
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A value in `[-1, 1)` on a 1/1024 grid (exact in binary32).
    fn weight(&mut self) -> f32 {
        let k = (self.next() % 2048) as i32 - 1024;
        k as f32 / 1024.0
    }
}

/// A random MLP `2 -> widths... -> 1` on the 1/1024 grid (the graph ledger
/// family's generator), its graph (`fc<i>` / `relu<i>`) and a threshold
/// strictly below its minimum over `[-1, 1]^2` for each `slack`.
fn random_mlp(rng: &mut Rng, widths: &[usize]) -> (GraphNetwork, Vec<(Vec<Vec<f64>>, Vec<f64>)>) {
    let mut g = GraphNetwork::new();
    let mut layers = Vec::new();
    let mut fan_in = 2usize;
    let dims: Vec<usize> = widths.iter().copied().chain(std::iter::once(1)).collect();
    let mut prev: Option<String> = None;
    for (i, &out) in dims.iter().enumerate() {
        let w: Vec<Vec<f32>> = (0..out)
            .map(|_| (0..fan_in).map(|_| rng.weight()).collect())
            .collect();
        let b: Vec<f32> = (0..out).map(|_| rng.weight() * 0.5).collect();
        let lin = LinearLayer::new(
            Array2::from_shape_vec((out, fan_in), w.iter().flatten().copied().collect())
                .expect("shape"),
            Some(Array1::from(b.clone())),
        )
        .expect("finite");
        let name = format!("fc{i}");
        match &prev {
            None => g.add_node(GraphNode::from_input(name.clone(), Layer::Linear(lin))),
            Some(p) => g.add_node(GraphNode::new(
                name.clone(),
                Layer::Linear(lin),
                vec![p.clone()],
            )),
        }
        prev = Some(name.clone());
        if i + 1 < dims.len() {
            let relu = format!("relu{i}");
            g.add_node(GraphNode::new(
                relu.clone(),
                Layer::ReLU(ReLULayer),
                vec![name],
            ));
            prev = Some(relu);
        }
        layers.push((
            w.iter()
                .map(|r| r.iter().map(|&v| f64::from(v)).collect())
                .collect(),
            b.iter().map(|&v| f64::from(v)).collect(),
        ));
        fan_in = out;
    }
    g.set_output(prev.expect("a layer"));
    (g, layers)
}

fn true_threshold(layers: &[(Vec<Vec<f64>>, Vec<f64>)], slack: f64) -> f32 {
    let eval = |x: [f64; 2]| {
        let mut h = x.to_vec();
        for (i, (w, b)) in layers.iter().enumerate() {
            let mut next: Vec<f64> = w
                .iter()
                .zip(b)
                .map(|(r, c)| r.iter().zip(&h).map(|(a, v)| a * v).sum::<f64>() + c)
                .collect();
            if i + 1 < layers.len() {
                next.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            h = next;
        }
        h[0]
    };
    let n = 100;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..=n {
        for j in 0..=n {
            let v = eval([
                -1.0 + 2.0 * f64::from(i) / f64::from(n),
                -1.0 + 2.0 * f64::from(j) / f64::from(n),
            ]);
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let lip: f64 = layers
        .iter()
        .map(|(w, _)| {
            w.iter()
                .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0, f64::max)
        })
        .product();
    (lo - lip / f64::from(n) - slack * (hi - lo) - 1e-3) as f32
}

/// The graph ReLU-split driver on the graph ledger family: bit-identical with
/// and without the tape on EVERY run (verified or not); every verified run
/// deposits a COMPLETE tape (the ledger proved it) that discharges the clause
/// by itself, and at least one of them is a split tree (split literals).
#[test]
fn relu_split_driver_tape_is_complete_and_bit_identical() {
    let input = unit_box();
    let mut rng = Rng(0x2F69_3A1C_7B45_9E0D);
    let mut trees = 0usize;
    for net in 0..8 {
        let widths: &[usize] = if net % 2 == 0 { &[4] } else { &[3, 3] };
        let (graph, layers) = random_mlp(&mut rng, widths);
        for slack in [0.02, 0.1, 0.3] {
            let thr = true_threshold(&layers, slack);
            let run = || {
                BetaCrownVerifier::new(config())
                    .verify_graph_relu_split(&graph, &input, &[1.0], thr)
                    .expect("runs")
            };
            let _g = lock();
            force_hint_capture_for_test(Some(false));
            reset_lane_proof_capture();
            let off = run();
            assert!(
                take_lane_proof_capture().is_empty(),
                "net={net}: Competition taped"
            );
            force_hint_capture_for_test(None);
            arm_lane_proof_capture_for_test();
            let on = run();
            let capture = take_own(&[1.0], thr);
            assert!(
                same(&off, &on),
                "net={net} slack={slack}: {off:?} vs {on:?}"
            );
            if !matches!(on.result, BabVerificationStatus::Verified) {
                assert!(capture.is_empty(), "net={net}: an unverified run deposited");
                continue;
            }
            let mut sites = HashMap::new();
            sites.insert("relu0".to_string(), "z0".to_string());
            sites.insert("relu1".to_string(), "z1".to_string());
            let property = GraphInstanceProperty::new(
                vec![-1.0, -1.0],
                vec![1.0, 1.0],
                vec![vec![
                    GraphObjective::new(&[1.0], thr, false).expect("finite")
                ]],
            )
            .expect("well formed");
            let inst = LaneProofInstance::new(&property, sites);
            let mut sink = RecordingSink::default();
            let s = stream_lane_proofs(&capture, &inst, &mut sink);
            assert_eq!(s.fallback_clauses, 0, "net={net}: {s:?} {:?}", sink.nodes);
            assert!(sink.nodes.iter().all(|n| n.close != NodeClose::Opaque));
            if sink.nodes.iter().any(|n| !n.lits.is_empty()) {
                trees += 1;
                eprintln!("[lane-proof-test] relu_split net={net} slack={slack} {s:?}");
            }
        }
    }
    assert!(
        trees >= 1,
        "no verified split tree in the family: the tape pin is VOID"
    );
}

/// The graph input-split driver: box leaves (and clip slabs), complete.
#[test]
fn input_split_driver_tape_is_complete_and_bit_identical() {
    let graph = crossing();
    let input = unit_box();
    let inst = instance(&[(&[1.0], THR)]);
    let cfg = BetaCrownConfig {
        max_domains: 4000,
        max_depth: 40,
        ..config()
    };
    identical_and_taped("input_split", &inst, (&[1.0], THR), 2, || {
        BetaCrownVerifier::new(cfg.clone())
            .verify_graph_input_split(&graph, &input, &[1.0], THR)
            .expect("runs")
    });
    let _g = lock();
    arm_lane_proof_capture_for_test();
    let _ = BetaCrownVerifier::new(cfg).verify_graph_input_split(&graph, &input, &[1.0], THR);
    let mut sink = RecordingSink::default();
    let s = stream_lane_proofs(&take_own(&[1.0], THR), &inst, &mut sink);
    assert_eq!(s.fallback_clauses, 0, "{s:?}");
    assert!(sink.nodes.iter().any(|n| !n.xlits.is_empty()), "box leaves");
}

/// The multi-objective (disjunctive) driver: every domain it closes records
/// itself; each leaf discharges the clauses its verified objectives entail.
#[test]
fn multi_objective_driver_tape_is_bit_identical_and_discharges_both_clauses() {
    let graph = crossing2();
    let input = unit_box();
    let objectives = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
    let thresholds = [THR, 0.5];
    let inst = instance(&[(&[1.0, 0.0], THR), (&[0.0, 1.0], 0.5)]);
    identical_and_taped("multi_objective", &inst, (&[1.0, 0.0], THR), 2, || {
        BetaCrownVerifier::new(config())
            .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
            .expect("runs")
    });
    let _g = lock();
    arm_lane_proof_capture_for_test();
    let _ = BetaCrownVerifier::new(config()).verify_graph_relu_split_multi_objective(
        &graph,
        &input,
        &objectives,
        &thresholds,
    );
    let mut sink = RecordingSink::default();
    let _ = stream_lane_proofs(&take_own(&[1.0, 0.0], THR), &inst, &mut sink);
    let both = sink
        .nodes
        .iter()
        .filter(|n| n.close == NodeClose::Bound && n.dis.as_deref() == Some(&[0usize, 1][..]))
        .count();
    assert!(
        both >= 1,
        "a disjunctive close certifies both objectives: {:?}",
        sink.nodes
    );
}

/// The lowdim tree: box leaves and slabs over the FULL input, deposited
/// complete when the queue drains; bit-identical with and without the tape.
#[test]
fn lowdim_tape_is_complete_and_bit_identical() {
    let graph = crossing();
    let input = unit_box();
    let cfg = crate::lowdim::LowDimConfig::default();
    let run = || {
        crate::lowdim::prove_conjunction_infeasible(
            &graph,
            &input,
            &[vec![1.0]],
            &[THR],
            Instant::now() + Duration::from_secs(20),
            &cfg,
        )
    };
    let key = |o: &crate::lowdim::LowDimOutcome| match o {
        crate::lowdim::LowDimOutcome::Verified(s) => {
            format!(
                "V {} {} {} {}",
                s.domains, s.closed, s.max_depth, s.clip_closed
            )
        }
        other => format!("{other:?}"),
    };
    let _g = lock();
    force_hint_capture_for_test(Some(false));
    reset_lane_proof_capture();
    let off = run();
    assert!(take_lane_proof_capture().is_empty());
    force_hint_capture_for_test(None);
    arm_lane_proof_capture_for_test();
    let on = run();
    let capture = take_own(&[1.0], THR);
    assert_eq!(key(&off), key(&on), "the tape changed the lowdim tree");
    assert!(
        matches!(on, crate::lowdim::LowDimOutcome::Verified(_)),
        "{on:?}"
    );
    let inst = instance(&[(&[1.0], THR)]);
    let mut sink = RecordingSink::default();
    let s = stream_lane_proofs(&capture, &inst, &mut sink);
    assert_eq!(s.fallback_clauses, 0, "{s:?} {:?}", sink.nodes);
    assert!(s.nodes >= 1, "{s:?}");
}

/// review(b2-lanes): a ZERO-WIDTH box leaf strictly inside the declared box
/// (a clip that shrank a face onto the other, then closed or clipped again)
/// streams nothing. Written as its faces it would carry BOTH `x_d >= t` and
/// `x_d <= t`, which the checker refuses as `both_sides` — a `malformed`
/// verdict on the WHOLE bundle — and it has zero measure, so the checker's
/// box cover never accepts it anyway. The positive-measure tiling still
/// discharges the clause by itself (no fallback claim).
#[test]
fn lane_tape_zero_width_box_leaves_stream_no_both_sided_literals() {
    let _g = lock();
    arm_lane_proof_capture_for_test();
    let tape = TapeRecorder::for_boxes(
        "lowdim",
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &[vec![1.0]],
        &[f64::from(THR)],
    )
    .expect("armed");
    tape.note_box(
        &[-1.0, -1.0],
        &[1.0, 0.25],
        TapeClose::Bound,
        vec![vec![0]],
        0.5,
    );
    tape.note_box(
        &[-1.0, 0.25],
        &[1.0, 1.0],
        TapeClose::Bound,
        vec![vec![0]],
        0.5,
    );
    // A leaf on the interior face `x1 = 0.25`, and a clip of a box of zero
    // width in `x1` (its slab keeps the zero width).
    tape.note_box(
        &[-1.0, 0.25],
        &[1.0, 0.25],
        TapeClose::Bound,
        vec![vec![0]],
        0.5,
    );
    tape.note_clip(
        &[-0.5, 0.25],
        &[0.5, 0.25],
        &[-0.25, 0.25],
        &[0.5, 0.25],
        &[vec![0]],
    );
    tape.deposit(true);
    let sink = stream(&take_own(&[1.0], THR), &instance(&[(&[1.0], THR)]));
    for n in &sink.nodes {
        for a in &n.xlits {
            assert!(
                !n.xlits.iter().any(|b| a.index == b.index
                    && a.threshold == b.threshold
                    && a.positive != b.positive),
                "both sides of one input threshold (checker: both_sides): {n:?}"
            );
        }
    }
    assert_eq!(
        sink.nodes.len(),
        2,
        "only the two positive-measure leaves: {:?}",
        sink.nodes
    );
    assert!(
        sink.nodes.iter().all(|n| n.src != "lane_claim"),
        "the complete tiling needs no fallback: {:?}",
        sink.nodes
    );
}

/// review(b2-lanes): the capture is process-global, so a test's armed window
/// also collects the tapes of lanes OTHER tests of this binary run; a
/// FOREIGN tape whose objective entails this test's clause would otherwise be
/// streamed as this test's (extra nodes, a foreign site size that malforms
/// the `sites` record, or "an unverified run deposited"). `take_own` keeps
/// exactly the tapes carrying this test's objective.
#[test]
fn take_own_keeps_only_this_tests_tapes() {
    let _g = lock();
    arm_lane_proof_capture_for_test();
    for (row, thr) in [
        (&[1.0f32][..], THR),
        (&[1.0f32][..], 0.0),
        (&[2.0f32][..], THR),
    ] {
        let tape = TapeRecorder::for_boxes(
            "lowdim",
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &[row.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>()],
            &[f64::from(thr)],
        )
        .expect("armed");
        tape.note_box(
            &[-1.0, -1.0],
            &[1.0, 1.0],
            TapeClose::Bound,
            vec![vec![0]],
            0.5,
        );
        tape.deposit(true);
    }
    let own = take_own(&[1.0], THR);
    assert_eq!(own.len(), 1, "only this test's tape: {own:?}");
    // The foreign `[1] > 0` tape ENTAILS this clause (`0 >= THR`): unfiltered
    // it would stream as a second discharging leaf.
    let sink = stream(&own, &instance(&[(&[1.0], THR)]));
    assert_eq!(sink.nodes.len(), 1, "{:?}", sink.nodes);
}
