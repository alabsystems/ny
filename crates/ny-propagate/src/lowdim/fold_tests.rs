// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `#lowdim-fold`. The fixture is an mscn-shaped set network:
//! row-wise MLP branches over slices of a `[rows, cols]` input, a masked mean
//! pool (`MulBinary` by a mask column, `ReduceSum` over rows, `Div` by the mask
//! count), a `Concat`, a head MLP and a `Sigmoid`; the dual variant subtracts
//! two sigmoid copies (copy A fully fixed). Soundness tests check the lane
//! against dense samples of the true output.

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::super::{bound_batch_multi, candidate_margin, sigmoid_down, sigmoid_sum_lower, Combine};
use super::*;
use crate::layers::{
    AddLayer, ConcatLayer, DivLayer, LinearLayer, MulBinaryLayer, ReLULayer, ReduceSumLayer,
    SigmoidLayer, SliceLayer, SubLayer,
};
use crate::network::GraphNode;

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

fn lin(rng: &mut Lcg, o: usize, i: usize, scale: f64) -> Layer {
    let w = Array2::from_shape_fn((o, i), |_| (rng.next() * scale) as f32);
    let b = Array1::from_shape_fn(o, |_| (rng.next() * 0.2) as f32);
    Layer::Linear(LinearLayer::new(w, Some(b)).expect("linear"))
}

/// One mscn-shaped copy reading input rows `[r0, r0 + 4)` of a `[_, 5]` input
/// (cols 0..4 features, col 4 mask); returns its logit node name.
fn copy(g: &mut GraphNetwork, rng: &mut Lcg, p: &str, r0: usize) -> String {
    let n = |s: &str| format!("{p}{s}");
    let add = |g: &mut GraphNetwork, name: String, layer: Layer, ins: Vec<String>| {
        g.add_node(GraphNode::new(name, layer, ins));
    };
    // branch A: rows r0..r0+3
    add(
        g,
        n("sA"),
        Layer::Slice(SliceLayer::new(0, r0, r0 + 3)),
        vec![NETWORK_INPUT.into()],
    );
    add(
        g,
        n("fA"),
        Layer::Slice(SliceLayer::new(1, 0, 4)),
        vec![n("sA")],
    );
    add(
        g,
        n("mA"),
        Layer::Slice(SliceLayer::new(1, 4, 5)),
        vec![n("sA")],
    );
    add(g, n("l1A"), lin(rng, 6, 4, 1.0), vec![n("fA")]);
    add(g, n("r1A"), Layer::ReLU(ReLULayer), vec![n("l1A")]);
    add(g, n("l2A"), lin(rng, 6, 6, 0.8), vec![n("r1A")]);
    add(g, n("r2A"), Layer::ReLU(ReLULayer), vec![n("l2A")]);
    add(
        g,
        n("mulA"),
        Layer::MulBinary(MulBinaryLayer),
        vec![n("r2A"), n("mA")],
    );
    add(
        g,
        n("sumA"),
        Layer::ReduceSum(ReduceSumLayer::new(vec![0], false)),
        vec![n("mulA")],
    );
    add(
        g,
        n("cntA"),
        Layer::ReduceSum(ReduceSumLayer::new(vec![0], false)),
        vec![n("mA")],
    );
    add(
        g,
        n("divA"),
        Layer::Div(DivLayer),
        vec![n("sumA"), n("cntA")],
    );
    // branch B: row r0+3
    add(
        g,
        n("sB"),
        Layer::Slice(SliceLayer::new(0, r0 + 3, r0 + 4)),
        vec![NETWORK_INPUT.into()],
    );
    add(
        g,
        n("fB"),
        Layer::Slice(SliceLayer::new(1, 0, 4)),
        vec![n("sB")],
    );
    add(
        g,
        n("mB"),
        Layer::Slice(SliceLayer::new(1, 4, 5)),
        vec![n("sB")],
    );
    add(g, n("l1B"), lin(rng, 6, 4, 1.0), vec![n("fB")]);
    add(g, n("r1B"), Layer::ReLU(ReLULayer), vec![n("l1B")]);
    add(
        g,
        n("mulB"),
        Layer::MulBinary(MulBinaryLayer),
        vec![n("r1B"), n("mB")],
    );
    add(
        g,
        n("sumB"),
        Layer::ReduceSum(ReduceSumLayer::new(vec![0], false)),
        vec![n("mulB")],
    );
    add(
        g,
        n("cntB"),
        Layer::ReduceSum(ReduceSumLayer::new(vec![0], false)),
        vec![n("mB")],
    );
    add(
        g,
        n("divB"),
        Layer::Div(DivLayer),
        vec![n("sumB"), n("cntB")],
    );
    add(
        g,
        n("cat"),
        Layer::Concat(ConcatLayer::new(0)),
        vec![n("divA"), n("divB")],
    );
    add(g, n("o1"), lin(rng, 5, 12, 0.7), vec![n("cat")]);
    add(g, n("or"), Layer::ReLU(ReLULayer), vec![n("o1")]);
    add(g, n("o2"), lin(rng, 1, 5, 1.0), vec![n("or")]);
    n("o2")
}

/// Single model: one copy + Sigmoid. `rows` = 4.
fn single() -> GraphNetwork {
    let mut rng = Lcg(11);
    let mut g = GraphNetwork::new();
    let z = copy(&mut g, &mut rng, "", 0);
    g.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec![z],
    ));
    g.set_output("sig");
    g
}

/// Dual: sigmoid(copy A on rows 0..4) - sigmoid(copy B on rows 4..8), same
/// weights (built from the same seed).
fn dual() -> GraphNetwork {
    dual_of(11, 0, 11, 4)
}

/// `sigmoid(copy A) - sigmoid(copy B)`: copy A's weights from `Lcg(seed_a)`,
/// reading input rows `[r0_a, r0_a + 4)`; copy B's from `Lcg(seed_b)`, rows
/// `[r0_b, r0_b + 4)`. `dual()` is `dual_of(11, 0, 11, 4)`.
fn dual_of(seed_a: u64, r0_a: usize, seed_b: u64, r0_b: usize) -> GraphNetwork {
    let mut g = GraphNetwork::new();
    let za = copy(&mut g, &mut Lcg(seed_a), "a_", r0_a);
    let zb = copy(&mut g, &mut Lcg(seed_b), "b_", r0_b);
    g.add_node(GraphNode::new(
        "siga",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec![za],
    ));
    g.add_node(GraphNode::new(
        "sigb",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec![zb],
    ));
    g.add_node(GraphNode::new(
        "y",
        Layer::Sub(SubLayer),
        vec!["siga".into(), "sigb".into()],
    ));
    g.set_output("y");
    g
}

/// A `[rows, 5]` box: fixed features, masks `[1, 1, 0, 1]` per 4-row copy, and
/// the listed free coordinates `(row, col, lo, hi)`.
fn boxed(rows: usize, free: &[(usize, usize, f32, f32)]) -> BoundedTensor {
    let mut rng = Lcg(3);
    let mut lo = ArrayD::from_elem(IxDyn(&[rows, 5]), 0.0f32);
    for r in 0..rows {
        for c in 0..4 {
            lo[[r, c]] = rng.next() as f32;
        }
        lo[[r, 4]] = if r % 4 == 2 { 0.0 } else { 1.0 };
    }
    let mut hi = lo.clone();
    for &(r, c, l, h) in free {
        lo[[r, c]] = l;
        hi[[r, c]] = h;
    }
    BoundedTensor::new(lo, hi).expect("box")
}

fn graph_at(g: &GraphNetwork, x: &ArrayD<f32>) -> f64 {
    let y = g
        .propagate_concrete_point(&BoundedTensor::concrete(x.clone()).unwrap(), None, None)
        .unwrap()
        .center();
    f64::from(*y.iter().next().unwrap())
}

/// Every point of a dense grid over the free coordinates: (flat x, value).
fn grid(g: &GraphNetwork, input: &BoundedTensor, steps: usize) -> Vec<(ArrayD<f32>, f64)> {
    let (lo, hi) = (input.lower(), input.upper());
    let free: Vec<usize> = (0..lo.len())
        .filter(|&i| hi.iter().nth(i) > lo.iter().nth(i))
        .collect();
    let total = steps.pow(free.len() as u32);
    (0..total)
        .map(|mut k| {
            let mut x = lo.clone();
            for &i in &free {
                let t = (k % steps) as f32 / (steps - 1) as f32;
                k /= steps;
                let (l, h) = (*lo.iter().nth(i).unwrap(), *hi.iter().nth(i).unwrap());
                *x.iter_mut().nth(i).unwrap() = l + (h - l) * t;
            }
            let y = graph_at(g, &x);
            (x, y)
        })
        .collect()
}

#[test]
fn the_folded_program_is_the_graph_at_points() {
    for (g, rows, free) in [
        (
            single(),
            4,
            vec![(1, 2, -0.5f32, 0.5f32), (3, 0, -1.0, 0.25)],
        ),
        (dual(), 8, vec![(5, 1, -0.75, 0.75)]),
    ] {
        let input = boxed(rows, &free);
        let cache = FoldCache::new();
        let f = fold_program(&g, &input, &cache).expect("folds");
        assert_eq!(f.free.len(), free.len());
        for (x, y) in grid(&g, &input, 7) {
            let xf: Vec<f64> = f
                .free
                .iter()
                .map(|&i| f64::from(*x.iter().nth(i).unwrap()))
                .collect();
            let mine = folded_point_output(&f, &xf);
            assert!(
                (mine - y).abs() <= 1e-5 * (1.0 + y.abs()),
                "folded {mine} vs graph {y}"
            );
        }
    }
}

/// The lane proves `s * Y > t` just below the sampled extremum and NEVER at a
/// threshold a sampled point already violates — in both directions, for the
/// single sigmoid and for the `const - sigmoid` dual.
#[test]
fn folded_lane_proves_true_bounds_and_never_false_ones() {
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    for (g, rows, free) in [
        (single(), 4, vec![(1, 2, -0.5f32, 0.5f32)]),
        (dual(), 8, vec![(5, 1, -0.75, 0.75), (7, 3, -0.3, 0.3)]),
    ] {
        let input = boxed(rows, &free);
        let pts = grid(&g, &input, 41);
        let ymin = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let ymax = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let spread = (ymax - ymin).max(1e-3);
        let cache = FoldCache::new();
        let run = |s: f64, t: f64| {
            prove_clause_folded(
                &g,
                &input,
                &[FoldConjunct { s, t }],
                Instant::now() + Duration::from_secs(60),
                &cfg,
                &cache,
            )
        };
        // true margins verify
        assert!(
            matches!(run(1.0, ymin - 0.25 * spread), LowDimOutcome::Verified(_)),
            "Y > ymin - margin must verify"
        );
        assert!(
            matches!(
                run(-1.0, -(ymax + 0.25 * spread)),
                LowDimOutcome::Verified(_)
            ),
            "Y < ymax + margin must verify"
        );
        // violated thresholds never verify
        if let LowDimOutcome::Verified(s) = run(1.0, ymin + 1e-4 * spread) {
            panic!("SOUNDNESS: verified Y > t with a sampled Y below t ({s:?})")
        }
        if let LowDimOutcome::Verified(s) = run(-1.0, -(ymax - 1e-4 * spread)) {
            panic!("SOUNDNESS: verified Y < t with a sampled Y above t ({s:?})")
        }
    }
}

#[test]
fn the_logit_transform_is_conservative() {
    let form = OutForm::Sig {
        sgn: 1.0,
        off_lo: 0.0,
        off_hi: 0.0,
    };
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    // Y > 0.3  <=  z > U with sigmoid(U) >= 0.3
    match target(&form, FoldConjunct { s: 1.0, t: 0.3 }) {
        Target::Lane(c, u) => {
            assert_eq!(c, 1.0);
            assert!(sig(u) >= 0.3);
        }
        _ => panic!("expected a lane target"),
    }
    // Y < 0.7  <=  -z > -L with sigmoid(L) <= 0.7
    match target(&form, FoldConjunct { s: -1.0, t: -0.7 }) {
        Target::Lane(c, neg_l) => {
            assert_eq!(c, -1.0);
            assert!(sig(-neg_l) <= 0.7);
        }
        _ => panic!("expected a lane target"),
    }
    assert!(matches!(
        target(&form, FoldConjunct { s: 1.0, t: -0.1 }),
        Target::Always
    ));
    assert!(matches!(
        target(&form, FoldConjunct { s: 1.0, t: 1.0 }),
        Target::Never
    ));
    // dual: Y = [0.4, 0.41] - sigmoid(z); Y > -0.2  <=  sigmoid(z) < 0.6 - eps
    let dual = OutForm::Sig {
        sgn: -1.0,
        off_lo: 0.4,
        off_hi: 0.41,
    };
    match target(&dual, FoldConjunct { s: 1.0, t: -0.2 }) {
        Target::Lane(c, neg_l) => {
            assert_eq!(c, -1.0);
            assert!(sig(-neg_l) <= 0.6);
        }
        _ => panic!("expected a lane target"),
    }
}

#[test]
fn a_product_of_two_free_values_is_declined() {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "a",
        Layer::Slice(SliceLayer::new(0, 0, 1)),
    ));
    g.add_node(GraphNode::from_input(
        "b",
        Layer::Slice(SliceLayer::new(0, 1, 2)),
    ));
    g.add_node(GraphNode::new(
        "p",
        Layer::MulBinary(MulBinaryLayer),
        vec!["a".into(), "b".into()],
    ));
    g.set_output("p");
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();
    let err = fold_program(&g, &input, &FoldCache::new())
        .err()
        .expect("declines");
    assert!(err.contains("two free"), "{err}");
}

/// A FAMILY run (many clauses of one structure bounded together) agrees with
/// running each clause alone, and never verifies a threshold a sampled point
/// violates.
#[test]
fn a_family_run_agrees_with_one_clause_at_a_time_and_never_verifies_a_false_bound() {
    let g = single();
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    let cache = FoldCache::new();
    let mut inputs = Vec::new();
    let mut conjuncts = Vec::new();
    let mut truth = Vec::new();
    for &(l, h) in &[(-0.5f32, 0.5f32), (-0.25, 0.75), (0.0, 0.3), (-1.0, -0.2)] {
        let input = boxed(4, &[(1, 2, l, h)]);
        let pts = grid(&g, &input, 41);
        let ymin = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let ymax = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let spread = (ymax - ymin).max(1e-3);
        for (t, holds) in [(ymin - 0.25 * spread, true), (ymin + 1e-4 * spread, false)] {
            inputs.push(input.clone());
            conjuncts.push(vec![FoldConjunct { s: 1.0, t }]);
            truth.push(holds);
        }
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let family = prove_clauses_folded(&g, &inputs, &conjuncts, deadline, &cfg, &cache);
    assert_eq!(family.len(), inputs.len());
    for (i, r) in family.iter().enumerate() {
        let alone = prove_clause_folded(&g, &inputs[i], &conjuncts[i], deadline, &cfg, &cache);
        let (fv, av) = (
            matches!(r, LowDimOutcome::Verified(_)),
            matches!(alone, LowDimOutcome::Verified(_)),
        );
        assert_eq!(fv, av, "clause {i}: family {r:?} vs alone {alone:?}");
        if !truth[i] {
            assert!(
                !fv,
                "SOUNDNESS: clause {i} verified a sampled-violated threshold"
            );
        } else {
            assert!(fv, "clause {i}: a true margin must verify ({r:?})");
        }
    }
}

// ------------------------------------------------------ #lowdim-fold-sigdiff

fn sig(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// `(W, b)` of a `Linear` node, in f64.
type Affine = (Vec<Vec<f64>>, Vec<f64>);

fn lin_f64(g: &GraphNetwork, name: &str) -> Affine {
    let node = g.node(name).unwrap_or_else(|| panic!("no node `{name}`"));
    let Layer::Linear(l) = node.layer() else {
        panic!("`{name}` is not Linear");
    };
    let w = l.weight();
    let rows: Vec<Vec<f64>> = (0..w.nrows())
        .map(|i| (0..w.ncols()).map(|j| f64::from(w[(i, j)])).collect())
        .collect();
    let b: Vec<f64> = match l.bias() {
        Some(b) => b.iter().map(|&v| f64::from(v)).collect(),
        None => vec![0.0; w.nrows()],
    };
    (rows, b)
}

fn affine(wb: &Affine, x: &[f64], relu: bool) -> Vec<f64> {
    wb.0.iter()
        .zip(&wb.1)
        .map(|(row, b)| {
            let v = b + row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>();
            if relu {
                v.max(0.0)
            } else {
                v
            }
        })
        .collect()
}

/// An independent plain-f64 evaluation of one `copy` (prefix `p`, input rows
/// `[r0, r0 + 4)`), written from the fixture's definition, not from the fold.
struct CopyRef {
    r0: usize,
    l1a: Affine,
    l2a: Affine,
    l1b: Affine,
    o1: Affine,
    o2: Affine,
}

impl CopyRef {
    fn new(g: &GraphNetwork, p: &str, r0: usize) -> Self {
        let w = |s: &str| lin_f64(g, &format!("{p}{s}"));
        Self {
            r0,
            l1a: w("l1A"),
            l2a: w("l2A"),
            l1b: w("l1B"),
            o1: w("o1"),
            o2: w("o2"),
        }
    }

    /// The copy's logit at the flat `[rows, 5]` input `x`.
    fn logit(&self, x: &[f64]) -> f64 {
        let row = |r: usize| (x[r * 5..r * 5 + 4].to_vec(), x[r * 5 + 4]);
        let (mut sum_a, mut cnt_a) = (vec![0.0f64; 6], 0.0f64);
        for r in self.r0..self.r0 + 3 {
            let (f, m) = row(r);
            let h = affine(&self.l2a, &affine(&self.l1a, &f, true), true);
            for (s, v) in sum_a.iter_mut().zip(&h) {
                *s += v * m;
            }
            cnt_a += m;
        }
        let (f, m) = row(self.r0 + 3);
        let hb = affine(&self.l1b, &f, true);
        let mut cat: Vec<f64> = sum_a.iter().map(|s| s / cnt_a).collect();
        cat.extend(hb.iter().map(|v| (v * m) / m));
        affine(&self.o2, &affine(&self.o1, &cat, true), false)[0]
    }
}

/// Reference for [`dual_of`]: `(zA, zB)` and `Y = sigmoid(zA) - sigmoid(zB)`.
struct DualRef {
    a: CopyRef,
    b: CopyRef,
}

impl DualRef {
    fn new(g: &GraphNetwork, r0_a: usize, r0_b: usize) -> Self {
        Self {
            a: CopyRef::new(g, "a_", r0_a),
            b: CopyRef::new(g, "b_", r0_b),
        }
    }

    fn z(&self, x: &[f64]) -> (f64, f64) {
        (self.a.logit(x), self.b.logit(x))
    }

    fn y(&self, x: &[f64]) -> f64 {
        let (za, zb) = self.z(x);
        sig(za) - sig(zb)
    }
}

/// (graph, rows, r0_a, r0_b, free coordinates `(row, col, lo, hi)`).
type SigDiffFixture = (
    GraphNetwork,
    usize,
    usize,
    usize,
    Vec<(usize, usize, f32, f32)>,
);

/// Two-sigmoid fixtures: `dual()` with free coordinates in BOTH copies
/// (disjoint rows, so zA and zB are independent), and two DIFFERENT copies
/// reading the SAME rows (zA and zB correlated through shared coordinates).
fn sigdiff_fixtures() -> Vec<SigDiffFixture> {
    vec![
        (
            dual(),
            8,
            0,
            4,
            vec![
                (1, 2, -0.5, 0.5),
                (3, 0, -1.0, 0.25),
                (5, 1, -0.75, 0.75),
                (7, 3, -0.3, 0.3),
            ],
        ),
        (dual(), 8, 0, 4, vec![(0, 1, -0.6, 0.4), (4, 2, -0.8, 0.8)]),
        (
            dual_of(11, 0, 12, 0),
            4,
            0,
            0,
            vec![(0, 1, -0.6, 0.6), (1, 3, -0.4, 0.8), (3, 2, -0.9, 0.1)],
        ),
    ]
}

/// Flat indices of the free coordinates, ascending (the fold's lane order).
fn free_of(input: &BoundedTensor) -> Vec<usize> {
    input
        .lower()
        .iter()
        .zip(input.upper().iter())
        .enumerate()
        .filter(|(_, (l, h))| h > l)
        .map(|(i, _)| i)
        .collect()
}

/// The box's bounds on the free coordinates `free`.
fn free_box(input: &BoundedTensor, free: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let lo: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    (
        free.iter().map(|&i| lo[i]).collect(),
        free.iter().map(|&i| hi[i]).collect(),
    )
}

/// The flat input with the free coordinates set to `p` (fixed ones from the box).
fn with_free(input: &BoundedTensor, free: &[usize], p: &[f64]) -> Vec<f64> {
    let mut x: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    for (&i, &v) in free.iter().zip(p) {
        x[i] = v;
    }
    x
}

/// `p` rounded to f32 (inside `[lo, hi]`), so the f32 graph and the f64
/// reference evaluate the SAME point.
fn to_f32_point(p: &[f64], lo: &[f64], hi: &[f64]) -> Vec<f64> {
    p.iter()
        .zip(lo.iter().zip(hi))
        .map(|(&v, (&l, &h))| f64::from(v as f32).clamp(l, h))
        .collect()
}

fn graph_at_flat(g: &GraphNetwork, rows: usize, x: &[f64]) -> f64 {
    let a = ArrayD::from_shape_vec(IxDyn(&[rows, 5]), x.iter().map(|&v| v as f32).collect())
        .expect("shape");
    graph_at(g, &a)
}

/// Points of the box `[lo, hi]`: a grid with the fewest steps per axis giving
/// at least `min_grid` points (its end steps are exactly `lo` and `hi`, so
/// every corner is included), plus `extra` random points.
fn box_points(
    lo: &[f64],
    hi: &[f64],
    min_grid: usize,
    extra: usize,
    rng: &mut Lcg,
) -> Vec<Vec<f64>> {
    let d = lo.len();
    let mut steps = 2usize;
    while steps.pow(d as u32) < min_grid {
        steps += 1;
    }
    let mut pts = Vec::new();
    for mut k in 0..steps.pow(d as u32) {
        let p: Vec<f64> = (0..d)
            .map(|j| {
                let i = k % steps;
                k /= steps;
                if i == 0 {
                    lo[j]
                } else if i == steps - 1 {
                    hi[j]
                } else {
                    (lo[j] + (hi[j] - lo[j]) * i as f64 / (steps - 1) as f64).clamp(lo[j], hi[j])
                }
            })
            .collect();
        pts.push(p);
    }
    for _ in 0..extra {
        pts.push(
            (0..d)
                .map(|j| (lo[j] + (hi[j] - lo[j]) * 0.5 * (rng.next() + 1.0)).clamp(lo[j], hi[j]))
                .collect(),
        );
    }
    pts
}

/// Distinct free coordinates: one or two from copy A's unmasked rows and one
/// or two from copy B's (the same pool when the copies share rows), each over
/// a random sub-interval of about [-1, 1].
fn random_free(rng: &mut Lcg, r0_a: usize, r0_b: usize) -> Vec<(usize, usize, f32, f32)> {
    let pool = |r0: usize| -> Vec<(usize, usize)> {
        [r0, r0 + 1, r0 + 3]
            .iter()
            .flat_map(|&r| (0..4).map(move |c| (r, c)))
            .collect()
    };
    let pick = |rng: &mut Lcg, pool: &[(usize, usize)], taken: &[(usize, usize)]| loop {
        let k = (f64::midpoint(rng.next(), 1.0) * pool.len() as f64) as usize;
        let rc = pool[k.min(pool.len() - 1)];
        if !taken.contains(&rc) {
            break rc;
        }
    };
    let n_a = 1 + usize::from(rng.next() > 0.0);
    let n_b = 1 + usize::from(rng.next() > 0.0);
    let (pa, pb) = (pool(r0_a), pool(r0_b));
    let mut taken: Vec<(usize, usize)> = Vec::new();
    for _ in 0..n_a {
        let rc = pick(&mut *rng, &pa[..], &taken[..]);
        taken.push(rc);
    }
    for _ in 0..n_b {
        let rc = pick(&mut *rng, &pb[..], &taken[..]);
        taken.push(rc);
    }
    taken
        .into_iter()
        .map(|(r, c)| {
            let centre = 0.7 * rng.next();
            let half = 0.15 + 0.25 * (rng.next() + 1.0);
            (r, c, (centre - half) as f32, (centre + half) as f32)
        })
        .collect()
}

/// `#lowdim-fold-sigdiff` ADMISSION: a dual whose BOTH copies read free
/// coordinates folds to the two-sigmoid form (no decline) with the lane seed
/// `[zA, zB]` in that order; the folded output agrees with an independent f64
/// evaluation of the fixture to 1e-9 relative at every corner, a grid and
/// random points; the f64 reference agrees with ny's own f32 graph forward;
/// and the clause passes the admission cross-check as the sigmoid-sum kind.
#[test]
fn a_dual_with_both_copies_free_folds_to_the_two_sigmoid_form_and_matches_the_graph() {
    let mut rng = Lcg(71);
    for (g, rows, r0a, r0b, free) in sigdiff_fixtures() {
        let dref = DualRef::new(&g, r0a, r0b);
        let input = boxed(rows, &free);
        let cache = FoldCache::new();
        let f = match fold_program(&g, &input, &cache) {
            Ok(f) => f,
            Err(e) => panic!("the two-sigmoid dual must fold, declined: {e}"),
        };
        match f.out {
            OutForm::SigDiff {
                a_sgn,
                b_sgn,
                c_lo,
                c_hi,
            } => {
                assert_eq!((a_sgn, b_sgn), (1.0, -1.0));
                assert!(
                    c_lo <= 0.0 && c_hi >= 0.0 && c_hi - c_lo < 1e-12,
                    "offset [{c_lo}, {c_hi}]"
                );
            }
            ref other => panic!("expected the two-sigmoid output form, got {other:?}"),
        }
        assert_eq!(f.prog.sizes[f.prog.seed], 2, "the lane seed is [zA, zB]");
        let fi = free_of(&input);
        assert_eq!(f.free, fi);
        let (lo, hi) = free_box(&input, &fi);
        let pts = box_points(&lo, &hi, 200, 100, &mut rng);
        assert!(pts.len() >= 300);
        for p in &pts {
            let (za, zb) = dref.z(&with_free(&input, &fi, p));
            let want = sig(za) - sig(zb);
            let mine = folded_point_output(&f, p);
            assert!(
                (mine - want).abs() <= 1e-9 * (1.0 + want.abs()),
                "folded {mine} vs f64 reference {want} at {p:?}"
            );
            let seed = super::super::point_values(&f.prog, p);
            assert!(
                (seed[0] - za).abs() <= 1e-9 * (1.0 + za.abs())
                    && (seed[1] - zb).abs() <= 1e-9 * (1.0 + zb.abs()),
                "seed {seed:?} vs (zA, zB) = ({za}, {zb})"
            );
            // The reference is the graph's function: ny's f32 forward agrees.
            let p32 = to_f32_point(p, &lo, &hi);
            let x32 = with_free(&input, &fi, &p32);
            let (y64, y32) = (dref.y(&x32), graph_at_flat(&g, rows, &x32));
            assert!(
                (y64 - y32).abs() <= 1e-5 * (1.0 + y32.abs()),
                "f64 reference {y64} vs graph {y32}"
            );
        }
        match prepare_clause(&g, &input, &[FoldConjunct { s: 1.0, t: 0.0 }], &cache) {
            Ok(c) => assert!(
                matches!(c.combine, Combine::SigmoidSum { .. }),
                "{:?}",
                c.combine
            ),
            Err(o) => panic!("the two-sigmoid clause must pass admission: {o:?}"),
        }
    }
}

/// SOUNDNESS of the combined per-domain bound: on random sub-boxes, the lane's
/// rigorous lower bounds of `zA` and `-zB`, and the outward-rounded sigmoid sum
/// built from them, never exceed the f64 reference at any sampled point
/// (corners included) by more than 1e-12 relative. Non-vacuity: some bound
/// comes within 0.2 of its sampled minimum (VOID otherwise).
#[test]
fn sigdiff_domain_bounds_never_exceed_sampled_values_on_sub_boxes() {
    let mut rng = Lcg(5);
    let mut tight = 0usize;
    for (g, rows, r0a, r0b, free) in sigdiff_fixtures() {
        let dref = DualRef::new(&g, r0a, r0b);
        let input = boxed(rows, &free);
        let cache = FoldCache::new();
        let clause = match prepare_clause(&g, &input, &[FoldConjunct { s: 1.0, t: 0.0 }], &cache) {
            Ok(c) => c,
            Err(o) => panic!("the two-sigmoid clause must be admitted: {o:?}"),
        };
        assert!(
            matches!(clause.combine, Combine::SigmoidSum { .. }),
            "{:?}",
            clause.combine
        );
        // s = +1 on `sigmoid(zA) - sigmoid(zB)`: the objectives are zA and -zB.
        assert_eq!(clause.objs, vec![vec![1.0, 0.0], vec![0.0, -1.0]]);
        let fi = free_of(&input);
        let d = fi.len();
        let nbox = 48;
        let (mut los, mut his) = (Vec::new(), Vec::new());
        for _ in 0..nbox {
            for k in 0..d {
                let (l, h) = (clause.lo[k], clause.hi[k]);
                let u = |r: &mut Lcg| (l + (h - l) * 0.5 * (r.next() + 1.0)).clamp(l, h);
                let (a, b) = (u(&mut rng), u(&mut rng));
                los.push(a.min(b));
                his.push(a.max(b));
            }
        }
        let bounds = bound_batch_multi(&clause.prog, &los, &his, nbox, &clause.objs, 8);
        assert_eq!(bounds.len(), nbox);
        for (bi, (lbs, _)) in bounds.iter().enumerate() {
            let (la, lb) = (lbs[0], lbs[1]);
            assert!(
                la.is_finite() && lb.is_finite(),
                "non-finite bounds {la} {lb}"
            );
            let comb = sigmoid_sum_lower(la, lb);
            let (bl, bh) = (&los[bi * d..(bi + 1) * d], &his[bi * d..(bi + 1) * d]);
            let mut smin = f64::INFINITY;
            for p in box_points(bl, bh, 32, 16, &mut rng) {
                let (za, zb) = dref.z(&with_free(&input, &fi, &p));
                let tol = |v: f64| 1e-12 * (1.0 + v.abs());
                assert!(la <= za + tol(za), "SOUNDNESS: lb(zA) = {la} > zA = {za}");
                assert!(
                    lb <= -zb + tol(zb),
                    "SOUNDNESS: lb(-zB) = {lb} > -zB = {}",
                    -zb
                );
                let truth = sig(za) + sig(-zb);
                assert!(
                    comb <= truth + tol(truth),
                    "SOUNDNESS: combined bound {comb} > sampled {truth}"
                );
                smin = smin.min(truth);
            }
            if smin - comb < 0.2 {
                tight += 1;
            }
        }
    }
    assert!(
        tight > 0,
        "VOID: no sub-box bound came within 0.2 of its sampled minimum"
    );
}

/// SOUNDNESS end to end: over random boxes with free coordinates in BOTH
/// copies (disjoint rows, and two different copies sharing rows), the lane
/// never verifies `s * Y > t` when a dense sample of the box (>= 200 grid
/// points including every corner, plus random points) has `s * Y <= t`, in
/// both directions; and it DOES verify true margins (VOID if none verify).
#[test]
fn the_sigdiff_lane_never_verifies_a_sampled_violation_and_verifies_true_margins() {
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    let mut rng = Lcg(2027);
    let (mut true_runs, mut verified_true, mut violated_runs) = (0usize, 0usize, 0usize);
    let shapes: [(GraphNetwork, usize, usize, usize); 2] =
        [(dual(), 8, 0, 4), (dual_of(11, 0, 12, 0), 4, 0, 0)];
    for (g, rows, r0a, r0b) in shapes {
        let dref = DualRef::new(&g, r0a, r0b);
        for _ in 0..5 {
            let free = random_free(&mut rng, r0a, r0b);
            let input = boxed(rows, &free);
            let cache = FoldCache::new();
            match fold_program(&g, &input, &cache) {
                Ok(f) => assert!(matches!(f.out, OutForm::SigDiff { .. }), "{:?}", f.out),
                Err(e) => panic!("box {free:?} declined: {e}"),
            }
            let fi = free_of(&input);
            let (lo, hi) = free_box(&input, &fi);
            let pts = box_points(&lo, &hi, 200, 64, &mut rng);
            assert!(pts.len() >= 264);
            let ys: Vec<f64> = pts
                .iter()
                .map(|p| {
                    let x = with_free(&input, &fi, &to_f32_point(p, &lo, &hi));
                    let (y, yg) = (dref.y(&x), graph_at_flat(&g, rows, &x));
                    assert!(
                        (y - yg).abs() <= 1e-5 * (1.0 + yg.abs()),
                        "reference {y} vs graph {yg}"
                    );
                    y
                })
                .collect();
            for s in [1.0f64, -1.0] {
                let sv: Vec<f64> = ys.iter().map(|y| s * y).collect();
                let vmin = sv.iter().copied().fold(f64::INFINITY, f64::min);
                let vmax = sv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let spread = (vmax - vmin).max(1e-3);
                let ts = [
                    vmin - 0.25 * spread,
                    vmin - 0.02 * spread,
                    vmin + 3.0 * candidate_margin(vmin),
                    vmin + 0.1 * spread,
                    f64::midpoint(vmin, vmax),
                ];
                for t in ts {
                    let r = prove_clause_folded(
                        &g,
                        &input,
                        &[FoldConjunct { s, t }],
                        Instant::now() + Duration::from_secs(60),
                        &cfg,
                        &cache,
                    );
                    let verified = matches!(r, LowDimOutcome::Verified(_));
                    if sv.iter().any(|&v| v <= t) {
                        violated_runs += 1;
                        assert!(
                            !verified,
                            "SOUNDNESS: verified s*Y > t (s={s}, t={t}) with a sampled \
s*Y = {vmin} <= t, box {free:?}"
                        );
                    } else {
                        true_runs += 1;
                        // Engagement counts only proofs the LANE made (bounded
                        // at least one domain), not an output-range shortcut.
                        if let LowDimOutcome::Verified(st) = &r {
                            verified_true += usize::from(st.domains > 0);
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "[sigdiff-test] true_runs={true_runs} verified_true={verified_true} \
violated_runs={violated_runs}"
    );
    assert!(violated_runs > 0, "VOID: no violated threshold was tried");
    assert!(
        verified_true > 0,
        "VOID: the lane verified no true threshold ({true_runs} tried)"
    );
}

/// A FAMILY run mixing the two-sigmoid kind with the existing `const - sigmoid`
/// kind (copy A fully fixed): every clause's verdict matches its one-at-a-time
/// run, no sampled-violated threshold verifies, and both kinds verify some true
/// margins (VOID otherwise).
#[test]
fn a_mixed_family_of_sigdiff_and_const_sigmoid_clauses_agrees_with_one_at_a_time() {
    let g = dual();
    let dref = DualRef::new(&g, 0, 4);
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    let cache = FoldCache::new();
    let mut rng = Lcg(404);
    let boxes: Vec<Vec<(usize, usize, f32, f32)>> = vec![
        // two-sigmoid kind (rows 0..4 are copy A), one structure
        vec![(1, 2, -0.5, 0.5), (5, 1, -0.75, 0.75)],
        vec![(1, 2, -0.2, 0.9), (5, 1, -0.9, 0.1)],
        // `const - sigmoid` kind (copy A fixed), one structure
        vec![(5, 1, -0.75, 0.75), (7, 3, -0.3, 0.3)],
        vec![(5, 1, 0.0, 0.6), (7, 3, -0.8, -0.1)],
    ];
    let (mut inputs, mut conjuncts) = (Vec::new(), Vec::new());
    let (mut violated, mut two_sig) = (Vec::new(), Vec::new());
    for free in &boxes {
        let input = boxed(8, free);
        let is_two = free.iter().any(|&(r, ..)| r < 4);
        match fold_program(&g, &input, &cache).map(|f| f.out) {
            Ok(OutForm::SigDiff { .. }) => assert!(is_two),
            Ok(OutForm::Sig { .. }) => assert!(!is_two),
            other => panic!("unexpected fold {other:?}"),
        }
        let fi = free_of(&input);
        let (lo, hi) = free_box(&input, &fi);
        let ys: Vec<f64> = box_points(&lo, &hi, 225, 32, &mut rng)
            .iter()
            .map(|p| dref.y(&with_free(&input, &fi, p)))
            .collect();
        for s in [1.0f64, -1.0] {
            let sv: Vec<f64> = ys.iter().map(|y| s * y).collect();
            let vmin = sv.iter().copied().fold(f64::INFINITY, f64::min);
            let vmax = sv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let spread = (vmax - vmin).max(1e-3);
            for t in [vmin - 0.25 * spread, vmin + 0.1 * spread] {
                inputs.push(input.clone());
                conjuncts.push(vec![FoldConjunct { s, t }]);
                violated.push(sv.iter().any(|&v| v <= t));
                two_sig.push(is_two);
            }
        }
    }
    let deadline = Instant::now() + Duration::from_secs(120);
    let family = prove_clauses_folded(&g, &inputs, &conjuncts, deadline, &cfg, &cache);
    assert_eq!(family.len(), inputs.len());
    let (mut v_two, mut v_one) = (0usize, 0usize);
    for (i, r) in family.iter().enumerate() {
        let alone = prove_clause_folded(&g, &inputs[i], &conjuncts[i], deadline, &cfg, &cache);
        let (fv, av) = (
            matches!(r, LowDimOutcome::Verified(_)),
            matches!(alone, LowDimOutcome::Verified(_)),
        );
        assert_eq!(fv, av, "clause {i}: family {r:?} vs alone {alone:?}");
        if violated[i] {
            assert!(
                !fv,
                "SOUNDNESS: clause {i} verified a sampled-violated threshold"
            );
        } else if let LowDimOutcome::Verified(st) = r {
            // Only proofs the lane made (at least one domain bounded).
            if st.domains > 0 {
                if two_sig[i] {
                    v_two += 1;
                } else {
                    v_one += 1;
                }
            }
        }
    }
    assert!(
        v_two > 0,
        "VOID: no two-sigmoid clause verified in the family run"
    );
    assert!(
        v_one > 0,
        "VOID: no const-sigmoid clause verified in the family run"
    );
}

/// The two-sigmoid TARGET is conservative for every sign combination: when
/// exact pre-activation values would close it, `s * Y > t` holds for every
/// constant in the interval; `Always` holds everywhere; the candidate `shift`
/// reproduces `s * Y` at the constant's midpoint. And the directed sigmoid
/// sits below the f64 value by its designed slack, never above.
#[test]
fn the_sigdiff_target_is_conservative_for_every_sign_combination() {
    let mut rng = Lcg(99);
    let zs: Vec<f64> = (0..=24).map(|i| -6.0 + 0.5 * f64::from(i)).collect();
    let (mut lanes, mut closes) = (0usize, 0usize);
    for s in [1.0f64, -1.0] {
        for a_sgn in [1.0f64, -1.0] {
            for b_sgn in [1.0f64, -1.0] {
                for _ in 0..60 {
                    let c_lo = 0.3 * rng.next();
                    let c_hi = c_lo + 0.05 * (rng.next() + 1.0);
                    let t = 1.5 * rng.next();
                    let form = OutForm::SigDiff {
                        a_sgn,
                        b_sgn,
                        c_lo,
                        c_hi,
                    };
                    let tg = target(&form, FoldConjunct { s, t });
                    if matches!(tg, Target::SigSum { .. }) {
                        lanes += 1;
                    }
                    for &za in &zs {
                        for &zb in &zs {
                            let worst = [c_lo, c_hi]
                                .iter()
                                .map(|&c| s * (a_sgn * sig(za) + b_sgn * sig(zb) + c))
                                .fold(f64::INFINITY, f64::min);
                            match tg {
                                Target::Always => assert!(
                                    worst > t - 1e-12,
                                    "Always, but s*Y = {worst} <= t = {t}"
                                ),
                                Target::Never => {}
                                Target::Lane(..) => {
                                    panic!("a two-sigmoid output never targets one objective")
                                }
                                Target::SigSum { wa, wb, thr, shift } => {
                                    if sigmoid_sum_lower(wa * za, wb * zb) > thr {
                                        closes += 1;
                                        assert!(
                                            worst > t - 1e-12,
                                            "closed, but s*Y = {worst} <= t = {t} \
(s={s} a={a_sgn} b={b_sgn} zA={za} zB={zb})"
                                        );
                                    }
                                    let mid = s
                                        * (a_sgn * sig(za)
                                            + b_sgn * sig(zb)
                                            + f64::midpoint(c_lo, c_hi));
                                    assert!(
                                        (sig(wa * za) + sig(wb * zb) + shift - mid).abs() < 1e-12,
                                        "candidate shift"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        lanes > 0 && closes > 0,
        "VOID: lanes={lanes} closes={closes}"
    );
    // The directed sigmoid.
    for i in 0..=4000 {
        let x = -800.0 + 0.4 * f64::from(i);
        let (v, lo) = (sig(x), sigmoid_down(x));
        assert!((0.0..=1.0).contains(&lo), "sigmoid_down({x}) = {lo}");
        assert!(lo <= v, "sigmoid_down({x}) = {lo} > {v}");
        if v > 1e-280 {
            assert!(lo <= v * (1.0 - 1e-13), "no slack at {x}");
            assert!(lo >= v * (1.0 - 1e-11), "too loose at {x}");
        }
    }
    for x in [
        0.0, -0.0, 1e-300, -1e-300, 709.0, -709.0, 745.0, -745.0, 1e308, -1e308,
    ] {
        assert!(sigmoid_down(x) <= sig(x), "sigmoid_down({x})");
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(sigmoid_sum_lower(bad, 0.0), f64::NEG_INFINITY);
        assert_eq!(sigmoid_sum_lower(0.0, bad), f64::NEG_INFINITY);
    }
    let half_sum = sigmoid_sum_lower(0.0, 0.0);
    assert!(half_sum < 1.0 && half_sum > 1.0 - 1e-11, "{half_sum}");
}

/// Two sigmoids of two raw inputs, `Y = sigmoid(x0) - sigmoid(x1)` on
/// `[-1, 1]^2`: the exact minimum of `s * Y` is `sigmoid(-1) - sigmoid(1)` for
/// both signs (at the corner `(-1, 1)`, resp. `(1, -1)`). The lane proves a
/// threshold 1e-4 below it and refuses one 1e-5 above it. A sigmoid plus a free
/// value, and three sigmoids, still decline.
#[test]
fn two_raw_sigmoids_prove_to_the_exact_corner_minimum_and_other_mixes_decline() {
    let sig_of = |g: &mut GraphNetwork, i: usize| {
        g.add_node(GraphNode::from_input(
            format!("x{i}"),
            Layer::Slice(SliceLayer::new(0, i, i + 1)),
        ));
        g.add_node(GraphNode::new(
            format!("s{i}"),
            Layer::Sigmoid(SigmoidLayer::new()),
            vec![format!("x{i}")],
        ));
    };
    let unit_box = |n: usize| {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[n]), -1.0f32),
            ArrayD::from_elem(IxDyn(&[n]), 1.0f32),
        )
        .expect("box")
    };
    let mut g = GraphNetwork::new();
    sig_of(&mut g, 0);
    sig_of(&mut g, 1);
    g.add_node(GraphNode::new(
        "y",
        Layer::Sub(SubLayer),
        vec!["s0".into(), "s1".into()],
    ));
    g.set_output("y");
    let input = unit_box(2);
    let cache = FoldCache::new();
    match fold_program(&g, &input, &cache) {
        Ok(f) => assert!(matches!(f.out, OutForm::SigDiff { .. }), "{:?}", f.out),
        Err(e) => panic!("two sigmoids must fold: {e}"),
    }
    let cfg = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    let run = |s: f64, t: f64| {
        prove_clause_folded(
            &g,
            &input,
            &[FoldConjunct { s, t }],
            Instant::now() + Duration::from_secs(60),
            &cfg,
            &cache,
        )
    };
    let ymin = sig(-1.0) - sig(1.0);
    for s in [1.0f64, -1.0] {
        let r = run(s, ymin - 1e-4);
        assert!(
            matches!(r, LowDimOutcome::Verified(st) if st.domains > 0),
            "s={s}: s*Y > min - 1e-4 must verify by the lane ({r:?})"
        );
        let r = run(s, ymin + 1e-5);
        assert!(
            !matches!(r, LowDimOutcome::Verified(_)),
            "SOUNDNESS: s={s}: s*Y > min + 1e-5 is violated at a corner ({r:?})"
        );
    }
    // sigmoid(x0) + x1: a sigmoid combined with a free value.
    let mut g = GraphNetwork::new();
    sig_of(&mut g, 0);
    g.add_node(GraphNode::from_input(
        "x1",
        Layer::Slice(SliceLayer::new(0, 1, 2)),
    ));
    g.add_node(GraphNode::new(
        "y",
        Layer::Add(AddLayer),
        vec!["s0".into(), "x1".into()],
    ));
    g.set_output("y");
    let err = fold_program(&g, &unit_box(2), &FoldCache::new())
        .err()
        .expect("declines");
    assert!(
        err.contains("a sigmoid combined with a free value"),
        "{err}"
    );
    // (sigmoid(x0) - sigmoid(x1)) + sigmoid(x2): three sigmoids.
    let mut g = GraphNetwork::new();
    for i in 0..3 {
        sig_of(&mut g, i);
    }
    g.add_node(GraphNode::new(
        "d",
        Layer::Sub(SubLayer),
        vec!["s0".into(), "s1".into()],
    ));
    g.add_node(GraphNode::new(
        "y",
        Layer::Add(AddLayer),
        vec!["d".into(), "s2".into()],
    ));
    g.set_output("y");
    let err = fold_program(&g, &unit_box(3), &FoldCache::new())
        .err()
        .expect("declines");
    assert!(err.contains("more than two sigmoids"), "{err}");
}

// ------------------------------------------------ #lowdim-bundle (4) shared

/// #lowdim-bundle (4): the SHARED worklist (every family on one round-robin
/// worklist, no per-clause domain cap), with and without the relaxed clip,
/// proves exactly the clauses the sequential family loop proves on clauses
/// from TWO structure families, and never verifies a threshold a sampled
/// point violates. Unarmed is the default.
#[test]
fn a_shared_worklist_agrees_with_the_family_loop_and_never_verifies_a_false_bound() {
    assert!(!LowDimConfig::default().shared_worklist);
    let g = single();
    let base = LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    };
    let cache = FoldCache::new();
    let mut inputs = Vec::new();
    let mut conjuncts = Vec::new();
    let mut truth = Vec::new();
    // Two free coordinates in different rows: two program structures.
    for &(r, c) in &[(1usize, 2usize), (3, 1)] {
        for &(l, h) in &[(-0.5f32, 0.5f32), (-0.25, 0.75), (0.0, 0.3)] {
            let input = boxed(4, &[(r, c, l, h)]);
            let pts = grid(&g, &input, 41);
            let ymin = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
            let ymax = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            let spread = (ymax - ymin).max(1e-3);
            for (s, t, holds) in [
                (1.0, ymin - 0.25 * spread, true),
                (1.0, ymin + 1e-4 * spread, false),
                (-1.0, -(ymax + 0.25 * spread), true),
                (-1.0, -(ymax - 1e-4 * spread), false),
            ] {
                inputs.push(input.clone());
                conjuncts.push(vec![FoldConjunct { s, t }]);
                truth.push(holds);
            }
        }
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let sequential = prove_clauses_folded(&g, &inputs, &conjuncts, deadline, &base, &cache);
    assert_eq!(sequential.len(), inputs.len());
    for clip in [false, true] {
        let cfg = LowDimConfig {
            shared_worklist: true,
            clip,
            ..base.clone()
        };
        let shared = prove_clauses_folded(&g, &inputs, &conjuncts, deadline, &cfg, &cache);
        assert_eq!(shared.len(), inputs.len());
        for (i, (s, q)) in shared.iter().zip(&sequential).enumerate() {
            let (sv, qv) = (
                matches!(s, LowDimOutcome::Verified(_)),
                matches!(q, LowDimOutcome::Verified(_)),
            );
            if truth[i] {
                assert!(
                    sv,
                    "clip={clip} clause {i}: a true margin must verify ({s:?})"
                );
            } else {
                assert!(
                    !sv,
                    "SOUNDNESS: clip={clip} clause {i} verified a sampled-violated threshold"
                );
            }
            assert_eq!(
                sv, qv,
                "clip={clip} clause {i}: shared {s:?} vs sequential {q:?}"
            );
        }
    }
}
