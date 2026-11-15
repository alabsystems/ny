//! Tests for `#lowdim-prover`.
//!
//! The load-bearing ones are the SOUNDNESS tests: a published lower bound must
//! never exceed a value the network actually attains. They check it three
//! ways — against dense samples of each box, against a degenerate (point) box,
//! and end to end: BaB must refuse to verify a threshold that a sampled point
//! already violates.

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::*;
use crate::layers::{
    AddConstantLayer, AddLayer, ConcatLayer, FlattenLayer, LinearLayer, MulConstantLayer,
    ReLULayer, SliceLayer, SubConstantLayer, SubLayer, TanhLayer,
};
use crate::network::GraphNode;

/// Deterministic LCG in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn linear(rng: &mut Lcg, n_out: usize, n_in: usize, scale: f64) -> Layer {
    let w = Array2::from_shape_fn((n_out, n_in), |_| (rng.next() * scale) as f32);
    let b = Array1::from_shape_fn(n_out, |_| (rng.next() * 0.3) as f32);
    Layer::Linear(LinearLayer::new(w, Some(b)).expect("linear"))
}

fn fixed_linear(w: Vec<Vec<f32>>) -> Layer {
    let (r, c) = (w.len(), w[0].len());
    let flat: Vec<f32> = w.into_iter().flatten().collect();
    Layer::Linear(
        LinearLayer::new(Array2::from_shape_vec((r, c), flat).expect("shape"), None)
            .expect("linear"),
    )
}

/// A linearizenn-shaped graph: a ReLU trunk on a `d`-dim input, a 2-output
/// Gemm, and an affine head that re-reads input dims `1..d` through a Slice,
/// concatenates `a = h·w1` with `b = x[1..d]·w2` and mixes them with
/// `[[1,-1],[-1,1]]` (so `y0 = a - b`, `y1 = -y0`).
fn toy_graph(seed: u64, d: usize, width: usize) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    let inp = NETWORK_INPUT.to_string();
    g.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, width, d, 1.5),
        vec![inp.clone()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        linear(&mut rng, width, width, 0.6),
        vec!["r1".into()],
    ));
    g.add_node(GraphNode::new(
        "r2",
        Layer::ReLU(ReLULayer::new()),
        vec!["l2".into()],
    ));
    g.add_node(GraphNode::new(
        "l3",
        linear(&mut rng, 2, width, 0.6),
        vec!["r2".into()],
    ));
    g.add_node(GraphNode::new(
        "a",
        fixed_linear(vec![vec![-0.74, -0.44]]),
        vec!["l3".into()],
    ));
    g.add_node(GraphNode::new(
        "sl",
        Layer::Slice(SliceLayer::new(1, 1, d)),
        vec![inp],
    ));
    let w2: Vec<f32> = (1..d).map(|i| -1.5 - i as f32).collect();
    g.add_node(GraphNode::new(
        "b",
        fixed_linear(vec![w2]),
        vec!["sl".into()],
    ));
    g.add_node(GraphNode::new(
        "cat",
        Layer::Concat(ConcatLayer::new(1)),
        vec!["a".into(), "b".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        fixed_linear(vec![vec![1.0, -1.0], vec![-1.0, 1.0]]),
        vec!["cat".into()],
    ));
    g.set_output("out");
    g
}

fn boxed(lo: &[f64], hi: &[f64]) -> BoundedTensor {
    let d = lo.len();
    let l = ArrayD::from_shape_vec(IxDyn(&[1, d]), lo.iter().map(|&v| v as f32).collect()).unwrap();
    let h = ArrayD::from_shape_vec(IxDyn(&[1, d]), hi.iter().map(|&v| v as f32).collect()).unwrap();
    BoundedTensor::new(l, h).unwrap()
}

/// Root box with f32-representable endpoints (so the f64 program box and
/// the f32 BoundedTensor coincide exactly).
fn root_box(d: usize) -> (Vec<f64>, Vec<f64>) {
    let lo: Vec<f64> = (0..d)
        .map(|i| f64::from(-0.75f32 + 0.125 * i as f32))
        .collect();
    let hi: Vec<f64> = (0..d)
        .map(|i| f64::from(0.5f32 + 0.25 * i as f32))
        .collect();
    (lo, hi)
}

fn sample(rng: &mut Lcg, lo: &[f64], hi: &[f64]) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(l, h)| l + (h - l) * 0.5 * (rng.next() + 1.0))
        .collect()
}

#[test]
fn lowered_program_matches_the_graph_at_points() {
    let (d, width) = (3, 12);
    let g = toy_graph(7, d, width);
    let (lo, hi) = root_box(d);
    let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
    let mut rng = Lcg(99);
    for _ in 0..40 {
        // f32-representable point so graph IBP and the program see the same x.
        let x: Vec<f64> = sample(&mut rng, &lo, &hi)
            .into_iter()
            .map(|v| f64::from(v as f32))
            .collect();
        let y = g.propagate_ibp(&boxed(&x, &x)).expect("ibp");
        for j in 0..2 {
            let mut o = vec![0.0; 2];
            o[j] = 1.0;
            let p = point_objective(&prog, &x, &o);
            let (gl, gh) = (
                f64::from(y.lower()[IxDyn(&[0, j])]),
                f64::from(y.upper()[IxDyn(&[0, j])]),
            );
            assert!(
                p >= gl - 1e-3 && p <= gh + 1e-3,
                "program y{j}={p} outside graph IBP point enclosure [{gl}, {gh}]"
            );
        }
    }
}

#[test]
fn published_bound_never_exceeds_a_sampled_value() {
    let (d, width) = (3, 16);
    for seed in [1u64, 2, 3, 11] {
        let g = toy_graph(seed, d, width);
        let (lo, hi) = root_box(d);
        let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
        let mut rng = Lcg(seed * 31 + 5);
        // 24 random sub-boxes of assorted sizes, bounded in ONE batch with an
        // odd chunk size so chunk boundaries are exercised.
        let mut blo = Vec::new();
        let mut bhi = Vec::new();
        for i in 0..24 {
            let c = sample(&mut rng, &lo, &hi);
            let r = [0.4, 0.1, 0.01, 1e-4][i % 4];
            for k in 0..d {
                blo.push((c[k] - r).max(lo[k]));
                bhi.push((c[k] + r).min(hi[k]));
            }
        }
        let o = vec![rng.next(), rng.next()];
        let res = bound_batch(&prog, &blo, &bhi, 24, &o, 7);
        assert_eq!(res.len(), 24);
        for (b, (lb, _)) in res.iter().enumerate() {
            assert!(lb.is_finite(), "seed {seed} box {b}: non-finite bound");
            let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
            let mut min_seen = f64::INFINITY;
            for _ in 0..400 {
                let x = sample(&mut rng, l, h);
                min_seen = min_seen.min(point_objective(&prog, &x, &o));
            }
            for corner in 0..(1 << d) {
                let x: Vec<f64> = (0..d)
                    .map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] })
                    .collect();
                min_seen = min_seen.min(point_objective(&prog, &x, &o));
            }
            assert!(
                *lb <= min_seen + 1e-9,
                "seed {seed} box {b}: bound {lb} exceeds attained value {min_seen}"
            );
        }
    }
}

#[test]
fn a_point_box_is_bounded_tightly_and_from_below() {
    let (d, width) = (3, 16);
    let g = toy_graph(5, d, width);
    let (lo, hi) = root_box(d);
    let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
    let mut rng = Lcg(17);
    for _ in 0..20 {
        let x = sample(&mut rng, &lo, &hi);
        let o = vec![rng.next(), rng.next()];
        let v = point_objective(&prog, &x, &o);
        let lb = bound_batch(&prog, &x, &x, 1, &o, 1)[0].0;
        assert!(lb <= v, "point bound {lb} above value {v}");
        assert!(
            v - lb < 1e-9 * (1.0 + v.abs()),
            "point bound {lb} not tight to {v}"
        );
    }
}

#[test]
fn bab_proves_a_true_margin_and_never_a_false_one() {
    let (d, width) = (3, 12);
    let g = toy_graph(3, d, width);
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&g, &input).expect("lowers");
    let o32 = [1.0f32, 0.0];
    let o = [1.0f64, 0.0];
    let mut rng = Lcg(1234);
    let mut min_seen = f64::INFINITY;
    for _ in 0..20_000 {
        let x = sample(&mut rng, &lo, &hi);
        min_seen = min_seen.min(point_objective(&prog, &x, &o));
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };

    // A threshold a sampled point ALREADY violates must never verify.
    let false_thr = (min_seen + 0.05) as f32;
    match prove_objective_lower_bound(&g, &input, &o32, false_thr, deadline, &cfg) {
        LowDimOutcome::Verified(s) => {
            panic!("SOUNDNESS: verified lower > {false_thr} but a point attains {min_seen} ({s:?})")
        }
        LowDimOutcome::Unproved(_, stop) => assert_eq!(stop, LowDimStop::Candidate),
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }

    // A threshold with real margin below the true minimum must verify.
    let true_thr = (min_seen - 0.25) as f32;
    match prove_objective_lower_bound(&g, &input, &o32, true_thr, deadline, &cfg) {
        LowDimOutcome::Verified(s) => assert!(s.closed >= 1 && s.domains >= s.closed),
        other => panic!("expected a proof of lower > {true_thr} (min seen {min_seen}): {other:?}"),
    }
}

#[test]
fn declines_outside_the_fragment() {
    let mut rng = Lcg(3);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, 4, 3, 1.0),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "t",
        Layer::Tanh(TanhLayer::new()),
        vec!["l1".into()],
    ));
    g.set_output("t");
    let (lo, hi) = root_box(3);
    let out = prove_objective_lower_bound(
        &g,
        &boxed(&lo, &hi),
        &[1.0, 0.0, 0.0, 0.0],
        -10.0,
        Instant::now() + Duration::from_secs(5),
        &LowDimConfig::default(),
    );
    assert!(
        matches!(&out, LowDimOutcome::Declined(r) if r.contains("Tanh")),
        "{out:?}"
    );
}

#[test]
fn declines_a_wide_input() {
    let mut rng = Lcg(3);
    let d = MAX_INPUT_DIM + 1;
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, 2, d, 1.0),
        vec![NETWORK_INPUT.into()],
    ));
    g.set_output("l1");
    let lo = vec![0.0; d];
    let hi = vec![1.0; d];
    let out = prove_objective_lower_bound(
        &g,
        &boxed(&lo, &hi),
        &[1.0, 0.0],
        -10.0,
        Instant::now() + Duration::from_secs(5),
        &LowDimConfig::default(),
    );
    assert!(matches!(out, LowDimOutcome::Declined(_)), "{out:?}");
}

/// Every lowering the linearizenn-shaped graph does NOT reach: Add, Sub,
/// AddConstant (vector and scalar), SubConstant (both directions),
/// MulConstant with mixed signs, and Flatten as a flat identity.
fn ops_graph(seed: u64) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    let inp = NETWORK_INPUT.to_string();
    let k4 = |v: [f32; 4]| ArrayD::from_shape_vec(IxDyn(&[4]), v.to_vec()).unwrap();
    g.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, 8, 3, 1.5),
        vec![inp.clone()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new("p", linear(&mut rng, 4, 3, 1.0), vec![inp]));
    g.add_node(GraphNode::new(
        "q",
        linear(&mut rng, 4, 8, 0.8),
        vec!["r1".into()],
    ));
    g.add_node(GraphNode::new(
        "s1",
        Layer::Add(AddLayer),
        vec!["p".into(), "q".into()],
    ));
    g.add_node(GraphNode::new(
        "s2",
        Layer::Sub(SubLayer),
        vec!["q".into(), "p".into()],
    ));
    g.add_node(GraphNode::new(
        "s3",
        Layer::AddConstant(AddConstantLayer::new(k4([0.5, -1.0, 0.25, 2.0]))),
        vec!["s1".into()],
    ));
    g.add_node(GraphNode::new(
        "s4",
        Layer::SubConstant(SubConstantLayer::new(k4([1.0, 0.5, -0.5, 0.0]))),
        vec!["s2".into()],
    ));
    g.add_node(GraphNode::new(
        "s5",
        Layer::SubConstant(SubConstantLayer::new_reverse(k4([0.3, 0.0, -2.0, 1.5]))),
        vec!["s3".into()],
    ));
    g.add_node(GraphNode::new(
        "s6",
        Layer::MulConstant(MulConstantLayer::new(k4([2.0, -1.5, 0.5, -0.25]))),
        vec!["s4".into()],
    ));
    g.add_node(GraphNode::new(
        "s7",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.75f32]).unwrap(),
        )),
        vec!["s6".into()],
    ));
    g.add_node(GraphNode::new(
        "s8",
        Layer::Add(AddLayer),
        vec!["s5".into(), "s7".into()],
    ));
    g.add_node(GraphNode::new(
        "r2",
        Layer::ReLU(ReLULayer::new()),
        vec!["s8".into()],
    ));
    g.add_node(GraphNode::new(
        "f",
        Layer::Flatten(FlattenLayer::new(1)),
        vec!["r2".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        linear(&mut rng, 2, 4, 1.0),
        vec!["f".into()],
    ));
    g.set_output("out");
    g
}

#[test]
fn every_admitted_op_lowers_to_the_graphs_own_semantics() {
    let d = 3;
    let g = ops_graph(21);
    let (lo, hi) = root_box(d);
    let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
    let mut rng = Lcg(77);
    for _ in 0..60 {
        let x: Vec<f64> = sample(&mut rng, &lo, &hi)
            .into_iter()
            .map(|v| f64::from(v as f32))
            .collect();
        let y = g.propagate_ibp(&boxed(&x, &x)).expect("ibp");
        for j in 0..2 {
            let mut o = vec![0.0; 2];
            o[j] = 1.0;
            let p = point_objective(&prog, &x, &o);
            let (gl, gh) = (
                f64::from(y.lower()[IxDyn(&[0, j])]),
                f64::from(y.upper()[IxDyn(&[0, j])]),
            );
            assert!(
                p >= gl - 1e-3 && p <= gh + 1e-3,
                "program y{j}={p} outside graph IBP point enclosure [{gl}, {gh}]"
            );
        }
    }
    // and the bound over that lowering is sound against samples
    let mut blo = Vec::new();
    let mut bhi = Vec::new();
    for i in 0..16 {
        let c = sample(&mut rng, &lo, &hi);
        let r = [0.5, 0.05, 0.005, 1e-5][i % 4];
        for k in 0..d {
            blo.push((c[k] - r).max(lo[k]));
            bhi.push((c[k] + r).min(hi[k]));
        }
    }
    let o = vec![rng.next(), rng.next()];
    for (b, (lb, _)) in bound_batch(&prog, &blo, &bhi, 16, &o, 5).iter().enumerate() {
        let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
        let mut min_seen = f64::INFINITY;
        for _ in 0..400 {
            min_seen = min_seen.min(point_objective(&prog, &sample(&mut rng, l, h), &o));
        }
        assert!(
            *lb <= min_seen + 1e-9,
            "box {b}: bound {lb} exceeds attained {min_seen}"
        );
    }
}

#[test]
fn a_degenerate_config_never_verifies_a_violated_threshold() {
    // chunk 0 used to bound NOTHING while the driver had already dequeued the
    // domains, so the next wave saw an empty queue and returned Verified.
    let (d, width) = (3, 12);
    let g = toy_graph(3, d, width);
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&g, &input).expect("lowers");
    let o = [1.0f64, 0.0];
    let mut rng = Lcg(4321);
    let mut min_seen = f64::INFINITY;
    for _ in 0..5_000 {
        min_seen = min_seen.min(point_objective(&prog, &sample(&mut rng, &lo, &hi), &o));
    }
    let cfg = LowDimConfig {
        batch: 0,
        chunk: 0,
        ..LowDimConfig::default()
    };
    let out = prove_objective_lower_bound(
        &g,
        &input,
        &[1.0, 0.0],
        (min_seen + 0.05) as f32,
        Instant::now() + Duration::from_secs(30),
        &cfg,
    );
    assert!(
        !matches!(out, LowDimOutcome::Verified(_)),
        "SOUNDNESS: degenerate config verified a violated threshold: {out:?}"
    );
}

#[test]
fn relu_upper_poisons_instead_of_asserting_a_flat_chord() {
    let (s, t) = relu_upper(-1.5e308, 1.5e308);
    assert!(
        s.is_infinite() && t.is_infinite(),
        "overflowed width must poison, got ({s}, {t})"
    );
    let (s, t) = relu_upper(-1.0, 3.0);
    assert!(
        s >= 0.75 && t >= 0.75,
        "chord slope/intercept must round UP: ({s}, {t})"
    );
}

/// Conjunctions (ACAS-style `Y_j <= Y_0 for all j`): a leaf closes when ANY
/// conjunct is proved impossible, and a conjunction some point satisfies must
/// never verify. In `toy_graph`, y1 == -y0.
#[test]
fn a_conjunction_closes_on_any_impossible_conjunct_and_never_when_a_point_satisfies_all() {
    let (d, width) = (3, 12);
    let g = toy_graph(3, d, width);
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&g, &input).expect("lowers");
    let (e0, e1) = (vec![1.0f64, 0.0], vec![0.0f64, 1.0]);
    let mut rng = Lcg(2468);
    let mut min0 = f64::INFINITY;
    let mut witness = None;
    for i in 0..20_000 {
        let x = sample(&mut rng, &lo, &hi);
        let y0 = point_objective(&prog, &x, &e0);
        min0 = min0.min(y0);
        if i == 777 {
            witness = Some((y0, point_objective(&prog, &x, &e1)));
        }
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    let objs = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];

    // (a) conjunct 0 is impossible everywhere (y0 > t0 with margin); conjunct
    // 1 (y1 > 1e6) is hopeless. The conjunction is infeasible via conjunct 0.
    let t = [(min0 - 0.25) as f32, 1.0e6f32];
    match prove_conjunction_infeasible(&g, &input, &objs, &t, deadline, &cfg) {
        LowDimOutcome::Verified(s) => assert!(s.closed >= 1),
        other => panic!("expected a proof via the impossible conjunct: {other:?}"),
    }

    // (b) a sampled point satisfies BOTH conjuncts, so the conjunction is
    // feasible and must never verify.
    let (w0, w1) = witness.expect("witness sampled");
    let t = [(w0 + 0.05) as f32, (w1 + 0.05) as f32];
    match prove_conjunction_infeasible(&g, &input, &objs, &t, deadline, &cfg) {
        LowDimOutcome::Verified(s) => panic!(
            "SOUNDNESS: verified an infeasible conjunction that the point ({w0}, {w1}) satisfies ({s:?})"
        ),
        LowDimOutcome::Unproved(_, stop) => assert_eq!(stop, LowDimStop::Candidate),
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }

    // (c) every per-objective bound from the SHARED forward pass is sound.
    let mut blo = Vec::new();
    let mut bhi = Vec::new();
    for i in 0..12 {
        let c = sample(&mut rng, &lo, &hi);
        let r = [0.3, 0.03, 0.003][i % 3];
        for k in 0..d {
            blo.push((c[k] - r).max(lo[k]));
            bhi.push((c[k] + r).min(hi[k]));
        }
    }
    let objs64 = vec![e0, e1, vec![rng.next(), rng.next()]];
    for (b, (lbs, _)) in bound_batch_multi(&prog, &blo, &bhi, 12, &objs64, 5)
        .iter()
        .enumerate()
    {
        let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
        for (i, o) in objs64.iter().enumerate() {
            let mut min_seen = f64::INFINITY;
            for _ in 0..300 {
                min_seen = min_seen.min(point_objective(&prog, &sample(&mut rng, l, h), o));
            }
            assert!(
                lbs[i] <= min_seen + 1e-9,
                "box {b} objective {i}: bound {} exceeds attained {min_seen}",
                lbs[i]
            );
        }
    }
}

/// The Sequential lane hands the lane `GraphNetwork::from_sequential(net)`
/// and the contract `network(x)[0] > threshold`: a true margin must verify
/// and a threshold a sampled point already violates must not.
#[test]
fn a_sequential_network_through_from_sequential_proves_its_first_output() {
    let mut rng = Lcg(55);
    let d = 2;
    let mut net = crate::network::Network::new();
    net.add_layer(linear(&mut rng, 16, d, 1.5));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(linear(&mut rng, 16, 16, 0.7));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(linear(&mut rng, 3, 16, 0.7));
    let graph = GraphNetwork::from_sequential(&net).expect("converts");
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&graph, &input).expect("lowers");
    let e0 = vec![1.0, 0.0, 0.0];
    let mut min_seen = f64::INFINITY;
    for _ in 0..20_000 {
        min_seen = min_seen.min(point_objective(&prog, &sample(&mut rng, &lo, &hi), &e0));
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    match prove_first_output_exceeds(&graph, &input, (min_seen - 0.25) as f32, deadline, &cfg) {
        LowDimOutcome::Verified(_) => {}
        other => panic!("expected a proof of output[0] > min - 0.25: {other:?}"),
    }
    match prove_first_output_exceeds(&graph, &input, (min_seen + 0.05) as f32, deadline, &cfg) {
        LowDimOutcome::Verified(s) => panic!("SOUNDNESS: verified a violated threshold ({s:?})"),
        LowDimOutcome::Unproved(_, stop) => assert_eq!(stop, LowDimStop::Candidate),
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
}

/// A Candidate stop hands back the leaf centre it scored as violating: inside
/// the input box and violating on the lane's own lowering, in BOTH modes. A
/// proof leaves the out-parameter untouched.
#[test]
fn a_candidate_stop_hands_back_an_in_box_violating_point_in_both_modes() {
    let mut rng = Lcg(55);
    let d = 2;
    let mut net = crate::network::Network::new();
    net.add_layer(linear(&mut rng, 16, d, 1.5));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(linear(&mut rng, 16, 16, 0.7));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(linear(&mut rng, 3, 16, 0.7));
    let graph = GraphNetwork::from_sequential(&net).expect("converts");
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&graph, &input).expect("lowers");
    let e0 = vec![1.0, 0.0, 0.0];
    let (mut min_seen, mut max_seen) = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..20_000 {
        let v = point_objective(&prog, &sample(&mut rng, &lo, &hi), &e0);
        min_seen = min_seen.min(v);
        max_seen = max_seen.max(v);
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    let in_box = |x: &[f64]| {
        x.len() == d
            && x.iter()
                .zip(&lo)
                .zip(&hi)
                .all(|((v, l), h)| l <= v && v <= h)
    };

    // lower mode, violated threshold: y0 <= t somewhere
    let t = (min_seen + 0.05) as f32;
    let mut w = None;
    match prove_first_output_witnessed(&graph, &input, t, false, deadline, &cfg, &mut w) {
        LowDimOutcome::Unproved(_, LowDimStop::Candidate) => {}
        other => panic!("expected a Candidate stop: {other:?}"),
    }
    let x = w.expect("a Candidate stop must hand back its point");
    assert!(in_box(&x), "witness {x:?} outside the box");
    assert!(point_objective(&prog, &x, &e0) <= f64::from(t));

    // upper mode, violated threshold: y0 >= t somewhere
    let t = (max_seen - 0.05) as f32;
    let mut w = None;
    match prove_first_output_witnessed(&graph, &input, t, true, deadline, &cfg, &mut w) {
        LowDimOutcome::Unproved(_, LowDimStop::Candidate) => {}
        other => panic!("expected a Candidate stop: {other:?}"),
    }
    let x = w.expect("a Candidate stop must hand back its point");
    assert!(in_box(&x), "witness {x:?} outside the box");
    assert!(point_objective(&prog, &x, &e0) >= f64::from(t));

    // a proof never writes a witness
    let mut w = None;
    match prove_first_output_witnessed(
        &graph,
        &input,
        (min_seen - 0.25) as f32,
        false,
        deadline,
        &cfg,
        &mut w,
    ) {
        LowDimOutcome::Verified(_) => assert!(w.is_none()),
        other => panic!("expected a proof: {other:?}"),
    }
}

/// Upper-bound mode is the SAME predicate through an exact negation:
/// `y0 < t` everywhere. A true margin above the sampled maximum verifies; a
/// threshold below a sampled value never does.
#[test]
fn upper_mode_proves_output_below_threshold_and_never_a_violated_one() {
    let mut rng = Lcg(77);
    let d = 2;
    let mut net = crate::network::Network::new();
    net.add_layer(linear(&mut rng, 16, d, 1.5));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(linear(&mut rng, 2, 16, 0.7));
    let graph = GraphNetwork::from_sequential(&net).expect("converts");
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&graph, &input).expect("lowers");
    let e0 = vec![1.0, 0.0];
    let mut max_seen = f64::NEG_INFINITY;
    for _ in 0..20_000 {
        max_seen = max_seen.max(point_objective(&prog, &sample(&mut rng, &lo, &hi), &e0));
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    match prove_first_output(
        &graph,
        &input,
        (max_seen + 0.25) as f32,
        true,
        deadline,
        &cfg,
    ) {
        LowDimOutcome::Verified(_) => {}
        other => panic!("expected a proof of y0 < max + 0.25: {other:?}"),
    }
    match prove_first_output(
        &graph,
        &input,
        (max_seen - 0.05) as f32,
        true,
        deadline,
        &cfg,
    ) {
        LowDimOutcome::Verified(s) => {
            panic!("SOUNDNESS: verified y0 < t below an attained value ({s:?})")
        }
        LowDimOutcome::Unproved(_, stop) => assert_eq!(stop, LowDimStop::Candidate),
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
}

/// The same-LHS conjunction reduction appends `Reshape[1,1,k] -> MaxPool2d((1,k))
/// -> Reshape[1]` and asks for `max_j z_j > t`. Lower mode decomposes it into
/// "ANY branch closes"; upper mode (`max < t`) goes through the generic
/// MaxGlobal relaxation. Both must prove a true margin and refuse a violated one.
#[test]
fn a_trailing_global_max_proves_in_both_modes_and_never_a_violated_threshold() {
    use crate::layers::{MaxPool2dLayer, ReshapeLayer};
    let mut rng = Lcg(808);
    let d = 2;
    let k = 3;
    let mut net = crate::network::Network::new();
    net.add_layer(linear(&mut rng, 12, d, 1.5));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(linear(&mut rng, k, 12, 0.8));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, k as i64])));
    net.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (1, k),
        (1, k),
        (0, 0),
    )));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1])));
    let graph = GraphNetwork::from_sequential(&net).expect("converts");
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&graph, &input).expect("lowers the global max");
    let one = vec![1.0];
    let (mut min_seen, mut max_seen) = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..20_000 {
        let v = point_objective(&prog, &sample(&mut rng, &lo, &hi), &one);
        min_seen = min_seen.min(v);
        max_seen = max_seen.max(v);
    }
    // the generic MaxGlobal relaxation is sound on its own
    let mut blo = Vec::new();
    let mut bhi = Vec::new();
    for i in 0..12 {
        let c = sample(&mut rng, &lo, &hi);
        let r = [0.3, 0.03, 0.003][i % 3];
        for j in 0..d {
            blo.push((c[j] - r).max(lo[j]));
            bhi.push((c[j] + r).min(hi[j]));
        }
    }
    for (b, (lb, _)) in bound_batch(&prog, &blo, &bhi, 12, &one, 5)
        .iter()
        .enumerate()
    {
        let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
        let mut m = f64::INFINITY;
        for _ in 0..300 {
            m = m.min(point_objective(&prog, &sample(&mut rng, l, h), &one));
        }
        assert!(
            *lb <= m + 1e-9,
            "box {b}: max bound {lb} exceeds attained {m}"
        );
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    // lower mode: max > t
    match prove_first_output(
        &graph,
        &input,
        (min_seen - 0.25) as f32,
        false,
        deadline,
        &cfg,
    ) {
        LowDimOutcome::Verified(_) => {}
        other => panic!("expected max > min - 0.25 to verify: {other:?}"),
    }
    match prove_first_output(
        &graph,
        &input,
        (min_seen + 0.05) as f32,
        false,
        deadline,
        &cfg,
    ) {
        LowDimOutcome::Verified(s) => {
            panic!("SOUNDNESS: verified max > t above an attained value ({s:?})")
        }
        LowDimOutcome::Unproved(_, stop) => assert_eq!(stop, LowDimStop::Candidate),
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
    // upper mode: max < t
    match prove_first_output(
        &graph,
        &input,
        (max_seen + 0.25) as f32,
        true,
        deadline,
        &cfg,
    ) {
        LowDimOutcome::Verified(_) => {}
        other => panic!("expected max < max + 0.25 to verify: {other:?}"),
    }
    match prove_first_output(
        &graph,
        &input,
        (max_seen - 0.05) as f32,
        true,
        deadline,
        &cfg,
    ) {
        LowDimOutcome::Verified(s) => {
            panic!("SOUNDNESS: verified max < t below an attained value ({s:?})")
        }
        LowDimOutcome::Unproved(_, stop) => assert_eq!(stop, LowDimStop::Candidate),
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
}

/// Backward-CROWN refinement of narrow layers must (a) actually tighten some
/// intermediate bound on a deep narrow trunk (engagement), (b) only ever
/// INTERSECT (refined bounds inside the forward ones), and (c) keep the published
/// objective bound sound against samples.
#[test]
fn refinement_tightens_intermediates_within_the_forward_bounds_and_stays_sound() {
    let mut rng = Lcg(4242);
    let d = 3;
    let mut net = crate::network::Network::new();
    net.add_layer(linear(&mut rng, 10, d, 1.5));
    for _ in 0..4 {
        net.add_layer(Layer::ReLU(ReLULayer::new()));
        net.add_layer(linear(&mut rng, 10, 10, 0.9));
    }
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(linear(&mut rng, 2, 10, 0.9));
    let graph = GraphNetwork::from_sequential(&net).expect("converts");
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let plain = lower_graph(&graph, &input).expect("lowers");
    let mut refined = plain.clone();
    refined.refine_max = 64;
    let ch = Chunk::new(1, d, &lo, &hi);
    let (one_p, one_r) = ([&plain], [&refined]);
    let (vp, _) = forward(Fam::single(&one_p), &ch);
    let (vr, _) = forward(Fam::single(&one_r), &ch);
    let mut tightened = 0;
    for idx in 0..plain.ops.len() {
        let (a, b) = (vp[idx].as_ref().unwrap(), vr[idx].as_ref().unwrap());
        if !plain.feeds_relu[idx] {
            continue;
        }
        for i in 0..plain.sizes[idx] {
            assert!(
                b.lo[i] >= a.lo[i] - 1e-12 || !b.lo[i].is_finite(),
                "refined lower below forward at node {idx}"
            );
            assert!(
                b.hi[i] <= a.hi[i] + 1e-12,
                "refined upper above forward at node {idx}"
            );
            if b.lo[i] > a.lo[i] + 1e-9 || b.hi[i] < a.hi[i] - 1e-9 {
                tightened += 1;
            }
        }
    }
    assert!(
        tightened > 0,
        "refinement engaged on no neuron of a 5-deep narrow trunk"
    );
    // soundness of the published bound with refinement on
    let o = vec![rng.next(), rng.next()];
    let mut blo = Vec::new();
    let mut bhi = Vec::new();
    for i in 0..12 {
        let c = sample(&mut rng, &lo, &hi);
        let r = [0.5, 0.05, 0.005][i % 3];
        for k in 0..d {
            blo.push((c[k] - r).max(lo[k]));
            bhi.push((c[k] + r).min(hi[k]));
        }
    }
    for (b, (lb, _)) in bound_batch(&refined, &blo, &bhi, 12, &o, 5)
        .iter()
        .enumerate()
    {
        let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
        let mut m = f64::INFINITY;
        for _ in 0..400 {
            m = m.min(point_objective(&refined, &sample(&mut rng, l, h), &o));
        }
        assert!(
            *lb <= m + 1e-9,
            "box {b}: refined bound {lb} exceeds attained {m}"
        );
    }
}

/// #router-r6 review fix: `rule.trial_split_restart` DISCARDS the tree. Whatever
/// the open queue held — many boxes, deep, with parent margins — the restart
/// leaves EXACTLY the root box at depth 0 with no parent margin: no stale box
/// may be bounded in the root's place, and the queue may never be left empty
/// (an empty queue reads as a proof at the next wave top).
#[test]
fn router_r6_restart_queue_is_exactly_the_root_box() {
    let lo0 = vec![-4.0, -1.0, 0.5];
    let hi0 = vec![8.0, 2.0, 0.75];
    let d = lo0.len();
    let mut q_lo: Vec<f64> = Vec::new();
    let mut q_hi: Vec<f64> = Vec::new();
    let mut q_depth: Vec<u32> = Vec::new();
    let mut q_parent: Vec<f64> = Vec::new();
    for k in 0..37u32 {
        let f = f64::from(k);
        q_lo.extend([-4.0 + f * 0.1, -1.0, 0.5]);
        q_hi.extend([-3.0 + f * 0.1, 0.5, 0.6]);
        q_depth.push(k % 9 + 1);
        q_parent.push(-f);
    }
    restart_queue_from_root(
        &mut q_lo,
        &mut q_hi,
        &mut q_depth,
        &mut q_parent,
        &lo0,
        &hi0,
    );
    assert_eq!(q_lo, lo0, "the open queue's only box is the root (lo)");
    assert_eq!(q_hi, hi0, "the open queue's only box is the root (hi)");
    assert_eq!(q_depth, vec![0], "one open domain, at depth 0");
    assert_eq!(q_parent.len(), 1);
    assert!(q_parent[0].is_nan(), "a root has no parent margin");
    assert_eq!(q_lo.len(), d * q_depth.len(), "queue arrays stay aligned");
    // Idempotent from an EMPTY queue too: the root is always there to close.
    let (mut a, mut b, mut c, mut e) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    restart_queue_from_root(&mut a, &mut b, &mut c, &mut e, &lo0, &hi0);
    assert_eq!((a, b, c), (lo0, hi0, vec![0u32]));
}
