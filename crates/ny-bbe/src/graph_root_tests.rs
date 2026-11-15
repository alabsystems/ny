// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! V2 (#vit-root-alpha) search-engine tests. Every oracle is INDEPENDENT of
//! the engine: interval arithmetic written from scratch for the reference
//! boxes, the net's own f64 point forward for truth, and central differences
//! for the gradient.
//!
//! * `graph_search_f64_bounds_are_valid_relaxations_at_random_parameters` —
//!   in f64 the engine's own relaxation (ReLU slopes, interpolated McCormick
//!   planes of both products and of the MulBinary, frozen Exp / Reciprocal /
//!   Clip lines, frozen softmax bounds) lower-bounds every sampled margin at
//!   RANDOM parameters, and the linearised forward reproduces the backward
//!   bound (a sign, plane-selection or index bug fails here).
//! * `graph_search_gradient_matches_central_differences` — the envelope
//!   gradient of every family (ReLU, product planes `[0]`/`[2]`, MulBinary
//!   `[0]`/`[1]`) against central differences away from kinks.
//! * `graph_search_f32_tracks_the_f64_reference`.
//! * `graph_search_never_returns_a_worse_score_and_never_moves_unsearched_planes`.
//! * `graph_search_improves_the_initial_bound_on_some_attention_block`.
//! * `graph_search_is_thread_count_independent` (R6).
//! * `graph_search_stops_on_rows_already_verified`.
//! * `graph_search_refuses_malformed_requests` (public backend API).
//! * `graph_mccormick_planes_match_the_textbook_corners` (KAT).
//! * `graph_unary_lines_enclose_their_functions` (KAT).

use ny_core::search::{
    GraphAlphaHint, GraphRootRequest, GraphSearchNet, GraphSearchOp, Refusal, SearchBackend,
    SparseTerm, TensorBox, UnaryKind,
};

use crate::graph_root::{
    eval_pass, eval_row, mccormick, planes, run_graph_root, score, search, unary_lines, Alpha,
    GConfig, GProgram,
};
use crate::CpuF32Backend;

// ---------------------------------------------------------------------------
//  Fixtures
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.uniform(-scale, scale)).collect()
    }
}

/// Tokens, model width and classes of the attention block.
const T: usize = 3;
const D: usize = 2;
const CLASSES: usize = 4;

fn term(out: usize, slot: usize, index: usize, weight: f64) -> SparseTerm {
    SparseTerm {
        out: out as u32,
        slot: slot as u32,
        index: index as u32,
        weight,
    }
}

fn dense(input: usize, rows: usize, n_in: usize, n_out: usize, rng: &mut Lcg) -> GraphSearchOp {
    GraphSearchOp::Dense {
        input,
        rows,
        n_in,
        n_out,
        weight: rng.vec(n_out * n_in, 0.8),
        bias: rng.vec(n_out, 0.2),
    }
}

/// A random attention block: per-token q/k/v projections, `q·kᵀ` (scale 0.7),
/// a softmax (monolithic with frozen constant bounds, or — `fixed_index` — the
/// lane's fixed-index decomposition: exp, row sum, reciprocal, broadcast,
/// MulBinary, clip), `attn·v`, a per-token output projection, a ReLU, mean
/// pooling over tokens and a classifier head.
fn attention_net(seed: u64, fixed_index: bool) -> GraphSearchNet {
    let mut rng = Lcg(seed);
    let mut ops = vec![
        dense(0, T, D, D, &mut rng),
        dense(0, T, D, D, &mut rng),
        dense(0, T, D, D, &mut rng),
        GraphSearchOp::Bilinear {
            a: 1,
            b: 2,
            batch: 1,
            m: T,
            n: T,
            k: D,
            transpose_b: true,
            scale: 0.7,
        },
    ];
    let attn = if fixed_index {
        ops.push(GraphSearchOp::Unary {
            input: 4,
            kind: UnaryKind::Exp,
        });
        let mut sum = Vec::new();
        for g in 0..T {
            for i in 0..T {
                sum.push(term(g, 0, g * T + i, 1.0));
            }
        }
        ops.push(GraphSearchOp::Sparse {
            inputs: vec![5],
            n_out: T,
            terms: sum,
            bias: vec![0.0; T],
        });
        ops.push(GraphSearchOp::Unary {
            input: 6,
            kind: UnaryKind::Reciprocal,
        });
        let mut bcast = Vec::new();
        for g in 0..T {
            for i in 0..T {
                bcast.push(term(g * T + i, 0, g, 1.0));
            }
        }
        ops.push(GraphSearchOp::Sparse {
            inputs: vec![7],
            n_out: T * T,
            terms: bcast,
            bias: vec![0.0; T * T],
        });
        ops.push(GraphSearchOp::MulBinary { a: 5, b: 8 });
        ops.push(GraphSearchOp::Unary {
            input: 9,
            kind: UnaryKind::Clip { lo: 0.0, hi: 1.0 },
        });
        10
    } else {
        ops.push(GraphSearchOp::Softmax {
            input: 4,
            groups: T,
            len: T,
            lower_a: vec![0.0; T * T * T],
            lower_b: vec![0.0; T * T],
            upper_a: vec![0.0; T * T * T],
            upper_b: vec![1.0; T * T],
        });
        5
    };
    let mixed = ops.len() + 1;
    ops.push(GraphSearchOp::Bilinear {
        a: attn,
        b: 3,
        batch: 1,
        m: T,
        n: D,
        k: T,
        transpose_b: false,
        scale: 1.0,
    });
    let projected = ops.len() + 1;
    ops.push(dense(mixed, T, D, 3, &mut rng));
    let activated = ops.len() + 1;
    ops.push(GraphSearchOp::Relu { input: projected });
    let pooled = ops.len() + 1;
    let mut mean = Vec::new();
    for r in 0..T {
        for o in 0..3 {
            mean.push(term(o, 0, r * 3 + o, 1.0 / T as f64));
        }
    }
    ops.push(GraphSearchOp::Sparse {
        inputs: vec![activated],
        n_out: 3,
        terms: mean,
        bias: vec![0.0; 3],
    });
    ops.push(GraphSearchOp::Dense {
        input: pooled,
        rows: 1,
        n_in: 3,
        n_out: CLASSES,
        weight: rng.vec(CLASSES * 3, 1.0),
        bias: rng.vec(CLASSES, 0.2),
    });
    GraphSearchNet { n_in: T * D, ops }
}

fn widen(l: f64, u: f64) -> (f64, f64) {
    (l - 1e-12 * (1.0 + l.abs()), u + 1e-12 * (1.0 + u.abs()))
}

type Boxes = Vec<(Vec<f64>, Vec<f64>)>;

/// Interval propagation written from scratch (outward-widened by 1e-12):
/// TRUE enclosures of every tensor over the input box.
fn interval_forward(net: &GraphSearchNet, lo: &[f64], hi: &[f64]) -> Boxes {
    let mut bx: Boxes = vec![(lo.to_vec(), hi.to_vec())];
    for op in &net.ops {
        let (l, u): (Vec<f64>, Vec<f64>) = match op {
            GraphSearchOp::Sparse {
                inputs,
                terms,
                bias,
                ..
            } => {
                let (mut l, mut u) = (bias.clone(), bias.clone());
                for t in terms {
                    let src = &bx[inputs[t.slot as usize]];
                    let i = t.index as usize;
                    let (p, q) = (t.weight * src.0[i], t.weight * src.1[i]);
                    l[t.out as usize] += p.min(q);
                    u[t.out as usize] += p.max(q);
                }
                (l, u)
            }
            GraphSearchOp::Dense {
                input,
                rows,
                n_in,
                n_out,
                weight,
                bias,
            } => {
                let src = &bx[*input];
                let (mut l, mut u) = (vec![0.0; rows * n_out], vec![0.0; rows * n_out]);
                for r in 0..*rows {
                    for o in 0..*n_out {
                        let (mut a, mut b) = (bias[o], bias[o]);
                        for i in 0..*n_in {
                            let w = weight[o * n_in + i];
                            let (p, q) = (w * src.0[r * n_in + i], w * src.1[r * n_in + i]);
                            a += p.min(q);
                            b += p.max(q);
                        }
                        l[r * n_out + o] = a;
                        u[r * n_out + o] = b;
                    }
                }
                (l, u)
            }
            GraphSearchOp::Relu { input } => (
                bx[*input].0.iter().map(|v| v.max(0.0)).collect(),
                bx[*input].1.iter().map(|v| v.max(0.0)).collect(),
            ),
            GraphSearchOp::Bilinear {
                a,
                b,
                batch,
                m,
                n,
                k,
                transpose_b,
                scale,
            } => {
                let (xa, xb) = (&bx[*a], &bx[*b]);
                let size = batch * m * n;
                let (mut l, mut u) = (vec![0.0; size], vec![0.0; size]);
                for bb in 0..*batch {
                    for i in 0..*m {
                        for j in 0..*n {
                            let (mut pl, mut pu) = (0.0, 0.0);
                            for q in 0..*k {
                                let ai = (bb * m + i) * k + q;
                                let bi = if *transpose_b {
                                    (bb * n + j) * k + q
                                } else {
                                    (bb * k + q) * n + j
                                };
                                let corners = [
                                    xa.0[ai] * xb.0[bi],
                                    xa.0[ai] * xb.1[bi],
                                    xa.1[ai] * xb.0[bi],
                                    xa.1[ai] * xb.1[bi],
                                ];
                                pl += corners.iter().copied().fold(f64::INFINITY, f64::min);
                                pu += corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            }
                            let c = (bb * m + i) * n + j;
                            let (p, q) = (scale * pl, scale * pu);
                            l[c] = p.min(q);
                            u[c] = p.max(q);
                        }
                    }
                }
                (l, u)
            }
            GraphSearchOp::MulBinary { a, b } => {
                let (xa, xb) = (&bx[*a], &bx[*b]);
                let mut l = Vec::with_capacity(xa.0.len());
                let mut u = Vec::with_capacity(xa.0.len());
                for j in 0..xa.0.len() {
                    let corners = [
                        xa.0[j] * xb.0[j],
                        xa.0[j] * xb.1[j],
                        xa.1[j] * xb.0[j],
                        xa.1[j] * xb.1[j],
                    ];
                    l.push(corners.iter().copied().fold(f64::INFINITY, f64::min));
                    u.push(corners.iter().copied().fold(f64::NEG_INFINITY, f64::max));
                }
                (l, u)
            }
            GraphSearchOp::Unary { input, kind } => {
                let src = &bx[*input];
                match kind {
                    UnaryKind::Reciprocal => {
                        assert!(
                            src.0.iter().all(|&v| v > 0.0),
                            "fixture reciprocal box must be positive"
                        );
                        (
                            src.1.iter().map(|v| 1.0 / v).collect(),
                            src.0.iter().map(|v| 1.0 / v).collect(),
                        )
                    }
                    _ => (
                        src.0.iter().map(|&v| kind.eval(v)).collect(),
                        src.1.iter().map(|&v| kind.eval(v)).collect(),
                    ),
                }
            }
            GraphSearchOp::Softmax {
                input, groups, len, ..
            } => {
                let src = &bx[*input];
                let (mut l, mut u) = (vec![0.0; groups * len], vec![0.0; groups * len]);
                for g in 0..*groups {
                    for i in 0..*len {
                        let (mut den_l, mut den_u) = (0.0, 0.0);
                        for j in 0..*len {
                            if j != i {
                                den_l += src.1[g * len + j].exp();
                                den_u += src.0[g * len + j].exp();
                            }
                        }
                        let (el, eu) = (src.0[g * len + i].exp(), src.1[g * len + i].exp());
                        l[g * len + i] = el / (el + den_l);
                        u[g * len + i] = eu / (eu + den_u);
                    }
                }
                (l, u)
            }
            GraphSearchOp::Conv(_) => panic!("no convolution in these fixtures"),
        };
        bx.push(l.iter().zip(&u).map(|(&a, &b)| widen(a, b)).unzip());
    }
    bx
}

/// The net, its input box and the TRUE reference boxes; a monolithic softmax
/// gets its (valid) frozen bounds from the interval enclosure of its output.
fn fixture(seed: u64, fixed_index: bool) -> (GraphSearchNet, Vec<f64>, Vec<f64>, Boxes) {
    let mut net = attention_net(seed, fixed_index);
    let mut rng = Lcg(seed ^ 0x5eed);
    let centre = rng.vec(T * D, 1.0);
    let lo: Vec<f64> = centre.iter().map(|c| c - 0.3).collect();
    let hi: Vec<f64> = centre.iter().map(|c| c + 0.3).collect();
    let boxes = interval_forward(&net, &lo, &hi);
    for (k, op) in net.ops.iter_mut().enumerate() {
        if let GraphSearchOp::Softmax {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            ..
        } = op
        {
            lower_a.iter_mut().for_each(|v| *v = 0.0);
            upper_a.iter_mut().for_each(|v| *v = 0.0);
            lower_b.clone_from(&boxes[k + 1].0);
            upper_b.clone_from(&boxes[k + 1].1);
        }
    }
    (net, lo, hi, boxes)
}

/// Margin rows `y_0 - y_j` (j = 1..CLASSES) with a common threshold.
fn request(
    lo: &[f64],
    hi: &[f64],
    boxes: &Boxes,
    threshold: f64,
    init: GraphAlphaHint,
) -> GraphRootRequest {
    let rows: Vec<Vec<f64>> = (1..CLASSES)
        .map(|j| {
            let mut r = vec![0.0; CLASSES];
            r[0] = 1.0;
            r[j] = -1.0;
            r
        })
        .collect();
    GraphRootRequest {
        lo: lo.to_vec(),
        hi: hi.to_vec(),
        boxes: boxes
            .iter()
            .enumerate()
            .map(|(t, (l, u))| TensorBox {
                tensor: t,
                l: l.clone(),
                u: u.clone(),
            })
            .collect(),
        thresholds: vec![threshold; rows.len()],
        rows,
        init,
        iters: 30,
        lr: 0.1,
        deadline: None,
        threads: 1,
        mem_cap_bytes: 1 << 30,
    }
}

/// Random starting parameters in `[lo, hi]` for every site.
fn random_init(net: &GraphSearchNet, seed: u64, lo: f64, hi: f64) -> GraphAlphaHint {
    let sizes = net.tensor_sizes().expect("valid fixture");
    let mut rng = Lcg(seed);
    let mut draw = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.uniform(lo, hi) as f32).collect() };
    let mut hint = GraphAlphaHint::default();
    for (op_index, op) in net.ops.iter().enumerate() {
        match op {
            GraphSearchOp::Relu { input } => hint.relu.push((op_index, draw(sizes[*input]))),
            GraphSearchOp::Bilinear { m, n, k, .. } => {
                hint.bilinear.push((op_index, draw(4 * m * n * k)));
            }
            GraphSearchOp::MulBinary { a, .. } => {
                hint.mul_binary.push((op_index, draw(2 * sizes[*a])));
            }
            _ => {}
        }
    }
    hint
}

fn margin_at(net: &GraphSearchNet, row: &[f64], x: &[f64]) -> f64 {
    let vals = net.forward_f64(x).expect("forward");
    row.iter()
        .zip(&vals[net.output()])
        .map(|(c, y)| c * y)
        .sum()
}

/// Every corner of the input box plus `extra` interior points.
fn samples(lo: &[f64], hi: &[f64], extra: usize, seed: u64) -> Vec<Vec<f64>> {
    let n = lo.len();
    let mut out: Vec<Vec<f64>> = (0..(1usize << n))
        .map(|mask| {
            (0..n)
                .map(|i| if mask >> i & 1 == 1 { hi[i] } else { lo[i] })
                .collect()
        })
        .collect();
    let mut rng = Lcg(seed);
    for _ in 0..extra {
        out.push((0..n).map(|i| rng.uniform(lo[i], hi[i])).collect());
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn hint_bits(h: &GraphAlphaHint) -> Vec<(usize, Vec<u32>)> {
    h.relu
        .iter()
        .chain(&h.bilinear)
        .chain(&h.mul_binary)
        .map(|(op, v)| (*op, bits(v)))
        .collect()
}

// ---------------------------------------------------------------------------
//  Validity, linearisation and gradient
// ---------------------------------------------------------------------------

#[test]
fn graph_search_f64_bounds_are_valid_relaxations_at_random_parameters() {
    let mut checked = 0usize;
    for fixed_index in [false, true] {
        for seed in 1..=4u64 {
            let (net, lo, hi, boxes) = fixture(seed, fixed_index);
            let init = random_init(&net, seed * 31, 0.0, 1.0);
            let req = request(&lo, &hi, &boxes, 0.0, init);
            let prog = GProgram::<f64>::lower(&net, &req).expect("lowers");
            assert!(!prog.relu.is_empty() && prog.bil.len() == 2);
            assert_eq!(prog.mul.len(), usize::from(fixed_index));
            let alpha = prog.init_alpha();
            let pl = planes(&prog, &alpha);
            let points = samples(&lo, &hi, 120, seed + 7);
            for (r, row) in req.rows.iter().enumerate() {
                let out = eval_row(&prog, &prog.root_view(), &alpha, &pl, r);
                assert!(out.bound.is_finite(), "seed {seed} row {r}: finite bound");
                let tol = 1e-9 * (1.0 + out.bound.abs());
                assert!(
                    (out.lin_value - out.bound).abs() <= tol,
                    "seed {seed} fixed_index {fixed_index} row {r}: the linearised forward \
                     {} must reproduce the backward bound {}",
                    out.lin_value,
                    out.bound
                );
                for x in &points {
                    let value = margin_at(&net, row, x);
                    assert!(
                        out.bound <= value + 1e-9 * (1.0 + value.abs()),
                        "seed {seed} fixed_index {fixed_index} row {r}: the search's relaxation \
                         bound {} exceeds the network's margin {value} at {x:?}",
                        out.bound
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 1000,
        "the validity sweep ran ({checked} comparisons)"
    );
}

#[test]
fn graph_search_gradient_matches_central_differences() {
    let h = 1e-6;
    let mut checked = [0usize; 3];
    for fixed_index in [false, true] {
        for seed in 1..=3u64 {
            let (net, lo, hi, boxes) = fixture(seed + 10, fixed_index);
            // Interior parameters so a +-h probe never meets the clamp.
            let init = random_init(&net, seed * 17, 0.2, 0.8);
            let req = request(&lo, &hi, &boxes, 0.0, init);
            let prog = GProgram::<f64>::lower(&net, &req).expect("lowers");
            let alpha = prog.init_alpha();
            let bound_at = |a: &Alpha<f64>, r: usize| {
                eval_row(&prog, &prog.root_view(), a, &planes(&prog, a), r).bound
            };
            for r in 0..req.rows.len() {
                let out = eval_row(&prog, &prog.root_view(), &alpha, &planes(&prog, &alpha), r);
                let grad = out.grad.expect("a finite row carries a gradient");
                let b0 = out.bound;
                // (family, site, element) of the first few nonzero entries.
                let mut probes: Vec<(usize, usize, usize)> = Vec::new();
                for (family, sites) in [(0usize, &grad.relu), (1, &grad.bil), (2, &grad.mul)] {
                    let mut taken = 0usize;
                    for (s, g) in sites.iter().enumerate() {
                        for (e, &v) in g.iter().enumerate() {
                            if v.abs() > 1e-7 && taken < 6 {
                                probes.push((family, s, e));
                                taken += 1;
                            }
                        }
                    }
                }
                for (family, s, e) in probes {
                    let analytic = match family {
                        0 => grad.relu[s][e],
                        1 => grad.bil[s][e],
                        _ => grad.mul[s][e],
                    };
                    let shifted = |delta: f64| {
                        let mut a = alpha.clone();
                        let slot = match family {
                            0 => &mut a.relu[s][e],
                            1 => &mut a.bil[s][e],
                            _ => &mut a.mul[s][e],
                        };
                        *slot += delta;
                        bound_at(&a, r)
                    };
                    let (bp, bm) = (shifted(h), shifted(-h));
                    let (fwd, bwd) = ((bp - b0) / h, (b0 - bm) / h);
                    if (fwd - bwd).abs() > 1e-4 * (1.0 + fwd.abs()) {
                        // A kink (a backward coefficient changed sign): the
                        // envelope rule holds only between kinks.
                        continue;
                    }
                    let central = (bp - bm) / (2.0 * h);
                    assert!(
                        (central - analytic).abs() <= 1e-5 * (1.0 + analytic.abs()),
                        "seed {seed} fixed_index {fixed_index} row {r} family {family} site {s} \
                         element {e}: analytic {analytic} vs central difference {central}"
                    );
                    checked[family] += 1;
                }
            }
        }
    }
    assert!(
        checked.iter().all(|&c| c > 0),
        "every family must be checked at least once (relu, bilinear, mul): {checked:?}"
    );
}

#[test]
fn graph_search_f32_tracks_the_f64_reference() {
    for fixed_index in [false, true] {
        for seed in 1..=3u64 {
            let (net, lo, hi, boxes) = fixture(seed + 20, fixed_index);
            let req = request(&lo, &hi, &boxes, 0.0, random_init(&net, seed, 0.0, 1.0));
            let p32 = GProgram::<f32>::lower(&net, &req).expect("f32 lowers");
            let p64 = GProgram::<f64>::lower(&net, &req).expect("f64 lowers");
            let a32 = p32.init_alpha();
            let a64 = p64.init_alpha();
            let b32 = eval_pass(&p32, &p32.root_view(), &a32, &planes(&p32, &a32), 1).bounds;
            let b64 = eval_pass(&p64, &p64.root_view(), &a64, &planes(&p64, &a64), 1).bounds;
            for (r, (x, y)) in b32.iter().zip(&b64).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-3 * (1.0 + y.abs()),
                    "seed {seed} fixed_index {fixed_index} row {r}: f32 {x} vs f64 {y}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  The search
// ---------------------------------------------------------------------------

fn config(threads: usize) -> GConfig {
    GConfig {
        iters: 30,
        lr: 0.1,
        deadline: None,
        threads,
        mem_cap: 1 << 30,
    }
}

#[test]
fn graph_search_never_returns_a_worse_score_and_never_moves_unsearched_planes() {
    for fixed_index in [false, true] {
        for seed in 1..=3u64 {
            let (net, lo, hi, boxes) = fixture(seed + 30, fixed_index);
            // Thresholds far above reach: every row stays unverified.
            let init = random_init(&net, seed, 0.0, 1.0);
            let req = request(&lo, &hi, &boxes, 50.0, init.clone());
            let prog = GProgram::<f32>::lower(&net, &req).expect("lowers");
            let result = search(&prog, &config(2)).expect("search runs");
            assert!(
                result.best_score.hinge >= result.init.hinge,
                "seed {seed}: the returned hinge {} is worse than the start {}",
                result.best_score.hinge,
                result.init.hinge
            );
            assert!(result.evals >= 1 && result.iters <= 30);
            let hint = &result.hint;
            assert_eq!(hint.relu.len(), prog.relu.len());
            assert_eq!(hint.bilinear.len(), prog.bil.len());
            assert_eq!(hint.mul_binary.len(), prog.mul.len());
            for (_, v) in hint
                .relu
                .iter()
                .chain(&hint.bilinear)
                .chain(&hint.mul_binary)
            {
                assert!(
                    v.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
                    "every hinted parameter is finite and in [0, 1]"
                );
            }
            // Stable ReLU neurons carry their exact limit.
            for (site, (op, v)) in prog.relu.iter().zip(&hint.relu) {
                assert_eq!(site.op, *op);
                let (l, u) = &boxes[site.input];
                for (j, &x) in v.iter().enumerate() {
                    if l[j] >= 0.0 {
                        assert_eq!(x.to_bits(), 1.0f32.to_bits(), "active neuron {j}");
                    } else if u[j] <= 0.0 {
                        assert_eq!(x.to_bits(), 0.0f32.to_bits(), "inactive neuron {j}");
                    }
                }
            }
            // The upper-direction planes [1] and [3] no margin row reads are
            // returned exactly as they came in.
            for (site, (op, v)) in prog.bil.iter().zip(&hint.bilinear) {
                let (_, given) = init
                    .bilinear
                    .iter()
                    .find(|(o, _)| o == op)
                    .expect("the fixture hinted every product");
                let block = site.block();
                for plane in [1usize, 3] {
                    let range = plane * block..(plane + 1) * block;
                    assert_eq!(
                        bits(&v[range.clone()]),
                        bits(&given[range]),
                        "seed {seed}: plane {plane} must never move"
                    );
                }
            }
        }
    }
}

#[test]
fn graph_search_improves_the_initial_bound_on_some_attention_block() {
    let mut improved = 0usize;
    for fixed_index in [false, true] {
        for seed in 1..=4u64 {
            let (net, lo, hi, boxes) = fixture(seed + 40, fixed_index);
            let req = request(&lo, &hi, &boxes, 50.0, GraphAlphaHint::default());
            let prog = GProgram::<f32>::lower(&net, &req).expect("lowers");
            let result = search(&prog, &config(3)).expect("search runs");
            if result.best_score.hinge > result.init.hinge {
                improved += 1;
            }
        }
    }
    assert!(
        improved > 0,
        "the search must raise the objective on at least one of eight random blocks"
    );
}

#[test]
fn graph_search_is_thread_count_independent() {
    for fixed_index in [false, true] {
        let (net, lo, hi, boxes) = fixture(50, fixed_index);
        let req = request(&lo, &hi, &boxes, 50.0, random_init(&net, 5, 0.0, 1.0));
        let mut reference: Option<(Vec<(usize, Vec<u32>)>, String)> = None;
        for threads in [1usize, 2, 3, 7] {
            let outcome = run_graph_root::<f32>(&net, &req, threads).expect("search runs");
            let stats = &outcome.stats;
            let fingerprint = format!(
                "iters={} evals={} stop={} init={} best={} verified={}/{}",
                stats.iters,
                stats.evals,
                stats.stop,
                stats.init_slack,
                stats.best_slack,
                stats.verified_init,
                stats.verified_best
            );
            let hint = outcome.alpha.into_hint().into_ingest();
            let hb = hint_bits(&hint);
            let expected = reference.get_or_insert_with(|| (hb.clone(), fingerprint.clone()));
            assert_eq!(
                &hb, &expected.0,
                "threads={threads}: the hint must not depend on it"
            );
            assert_eq!(&fingerprint, &expected.1, "threads={threads}");
        }
    }
}

#[test]
fn graph_search_stops_on_rows_already_verified() {
    let (net, lo, hi, boxes) = fixture(60, false);
    let req = request(&lo, &hi, &boxes, -1e6, GraphAlphaHint::default());
    let prog = GProgram::<f32>::lower(&net, &req).expect("lowers");
    let result = search(&prog, &config(1)).expect("search runs");
    assert_eq!(result.stop, "verified");
    assert_eq!(result.iters, 0);
    assert_eq!(result.best, prog.init_alpha());
    let s = score(&[1.0, f64::NAN], &[0.0, 0.0]);
    assert_eq!(s.verified, 1, "a NaN slack never counts as verified");
    assert_eq!(s.min_slack, f64::NEG_INFINITY);
}

#[test]
fn graph_search_refuses_malformed_requests() {
    let backend = CpuF32Backend::with_threads(1);
    let (net, lo, hi, boxes) = fixture(70, true);
    let good = request(&lo, &hi, &boxes, 0.0, GraphAlphaHint::default());
    assert!(backend.graph_root_alpha(&net, &good).is_ok());

    let mut missing = good.clone();
    let relu_input = net
        .ops
        .iter()
        .find_map(|op| match op {
            GraphSearchOp::Relu { input } => Some(*input),
            _ => None,
        })
        .expect("the fixture has a ReLU");
    missing.boxes.retain(|b| b.tensor != relu_input);
    assert!(matches!(
        backend.graph_root_alpha(&net, &missing),
        Err(Refusal::Invalid(_))
    ));

    let mut wide = good.clone();
    wide.rows[0].push(1.0);
    assert!(matches!(
        backend.graph_root_alpha(&net, &wide),
        Err(Refusal::Invalid(_))
    ));

    let mut nan = good.clone();
    nan.thresholds[1] = f64::NAN;
    assert!(matches!(
        backend.graph_root_alpha(&net, &nan),
        Err(Refusal::Invalid(_))
    ));

    let mut empty = good.clone();
    empty.rows.clear();
    empty.thresholds.clear();
    assert!(matches!(
        backend.graph_root_alpha(&net, &empty),
        Err(Refusal::Nothing(_))
    ));

    // The reciprocal's operand box reaches zero: its lines are undefined.
    let mut nonpositive = good.clone();
    let recip_input = net
        .ops
        .iter()
        .find_map(|op| match op {
            GraphSearchOp::Unary {
                input,
                kind: UnaryKind::Reciprocal,
            } => Some(*input),
            _ => None,
        })
        .expect("the fixed-index fixture has a reciprocal");
    for b in &mut nonpositive.boxes {
        if b.tensor == recip_input {
            b.l[0] = -1.0;
        }
    }
    assert!(matches!(
        backend.graph_root_alpha(&net, &nonpositive),
        Err(Refusal::Nothing(_))
    ));

    let mut tiny = good.clone();
    tiny.mem_cap_bytes = 16;
    assert!(matches!(
        backend.graph_root_alpha(&net, &tiny),
        Err(Refusal::Memory { .. })
    ));

    let mut late = good.clone();
    late.deadline = Some(std::time::Instant::now());
    assert!(matches!(
        backend.graph_root_alpha(&net, &late),
        Err(Refusal::Deadline)
    ));

    let mut open_box = good;
    open_box.hi[0] = f64::INFINITY;
    assert!(matches!(
        backend.graph_root_alpha(&net, &open_box),
        Err(Refusal::Invalid(_))
    ));
}

// ---------------------------------------------------------------------------
//  Known answers
// ---------------------------------------------------------------------------

#[test]
fn graph_mccormick_planes_match_the_textbook_corners() {
    let (xl, xu, yl, yu) = (-1.0, 2.0, 0.5, 3.0);
    // r = 1: L1 = yl x + xl y - xl yl, U1 = yu x + xl y - xl yu.
    assert_eq!(
        mccormick(xl, xu, yl, yu, 1.0, 1.0),
        [0.5, -1.0, 0.5, 3.0, -1.0, 3.0]
    );
    // r = 0: L2 = yu x + xu y - xu yu, U2 = yl x + xu y - xu yl.
    assert_eq!(
        mccormick(xl, xu, yl, yu, 0.0, 0.0),
        [3.0, 2.0, -6.0, 0.5, 2.0, -1.0]
    );
    let mut rng = Lcg(3);
    for r in [0.0, 0.3, 0.75, 1.0] {
        let p = mccormick(xl, xu, yl, yu, r, 1.0 - r);
        for _ in 0..200 {
            let (x, y) = (rng.uniform(xl, xu), rng.uniform(yl, yu));
            let z = x * y;
            assert!(
                p[0] * x + p[1] * y + p[2] <= z + 1e-12,
                "lower plane at r={r}"
            );
            assert!(
                p[3] * x + p[4] * y + p[5] >= z - 1e-12,
                "upper plane at r={r}"
            );
        }
    }
}

#[test]
fn graph_unary_lines_enclose_their_functions() {
    let kinds = [
        UnaryKind::Exp,
        UnaryKind::Reciprocal,
        UnaryKind::Clip { lo: 0.0, hi: 1.0 },
        UnaryKind::Clip { lo: -0.5, hi: 0.25 },
    ];
    let boxes = [
        (0.1, 0.1),
        (0.2, 3.0),
        (-2.0, 1.5),
        (-0.4, 0.1),
        (0.5, 0.9),
        (1.5, 4.0),
    ];
    let mut rng = Lcg(11);
    for kind in kinds {
        for (l, u) in boxes {
            if matches!(kind, UnaryKind::Reciprocal) && l <= 0.0 {
                assert!(unary_lines(kind, l, u).is_none(), "1/x needs l > 0");
                continue;
            }
            let (lower, upper) = unary_lines(kind, l, u).expect("finite lines");
            for q in 0..=100 {
                let x = if q == 100 {
                    rng.uniform(l, u)
                } else {
                    l + (u - l) * f64::from(q) / 99.0
                };
                let x = x.min(u);
                let f = kind.eval(x);
                let tol = 1e-12 * (1.0 + f.abs());
                assert!(
                    lower[0] * x + lower[1] <= f + tol,
                    "{kind:?} [{l}, {u}] at {x}: lower line above f"
                );
                assert!(
                    upper[0] * x + upper[1] >= f - tol,
                    "{kind:?} [{l}, {u}] at {x}: upper line below f"
                );
            }
        }
    }
    assert!(unary_lines(UnaryKind::Exp, f64::NEG_INFINITY, 0.0).is_none());
    assert!(
        unary_lines(UnaryKind::Exp, 0.0, 1000.0).is_none(),
        "overflow refuses"
    );
}
