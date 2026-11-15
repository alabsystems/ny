// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M7 (#graph-lane-search) search-engine tests. Every oracle is INDEPENDENT of
//! the engine: interval arithmetic written from scratch for the reference
//! boxes, the net's own f64 point forward for truth (rejection-sampled into
//! each domain's split literals), central differences for the gradients, and
//! a hand-derived known-answer net for the multiplier's sign.
//!
//! * `graph_bab_kat_split_multiplier_bound_and_gradient` — the Lagrangian KAT:
//!   `min_x (1 − β) x` over `[−1, 1]` on the domain `x ≥ 0`: bound
//!   `−|1 − β|`, gradient `+1` below `β = 1` (a flipped sign reads `−1 − β`).
//! * `graph_bab_f64_bounds_contain_sampled_truth_on_split_conv_and_residual_domains`
//!   — containment: at RANDOM slopes and multipliers, in f64, every searched
//!   bound lower-bounds the true margin at every sampled point of the domain,
//!   on MLP, conv and residual graphs; the linearised forward reproduces the
//!   bound.
//! * `graph_bab_mutation_pins_catch_a_wrong_piece_and_a_flipped_multiplier`.
//! * `graph_bab_gradient_matches_central_differences` (slopes and multipliers).
//! * `graph_bab_empty_domain_reproduces_the_root_search_bit_for_bit` (V2 KAT).
//! * `graph_bab_is_thread_count_independent` (R6).
//! * `graph_bab_domain_results_do_not_depend_on_the_batch` (R6).
//! * `graph_bab_multipliers_are_nonnegative_and_split_neurons_are_pinned`.
//! * `graph_bab_never_returns_a_worse_score`.
//! * `graph_bab_multipliers_tighten_a_split_domain`.
//! * `graph_bab_f32_tracks_the_f64_reference`.
//! * `graph_bab_refuses_malformed_requests_and_isolates_malformed_domains`.
//! * `graph_bab_cpu_backend_serves_the_batch_through_the_trait`.

use std::time::{Duration, Instant};

use ny_core::search::{
    GraphAlphaHint, GraphBabRequest, GraphDomainRequest, GraphDomainSplit, GraphRootRequest,
    GraphSearchNet, GraphSearchOp, Refusal, SearchBackend, SearchConv, SparseTerm, TensorBox,
    GATHER_PAD,
};

use crate::graph_bab::{
    domain_pass, lower_batch, run_graph_bab, search_domain, setup_domain, BabConfig, DomainRefusal,
    DomainSetup,
};
use crate::graph_root::{
    eval_row, improves, planes, search, Alpha, GConfig, GProgram, View, ACTIVE, INACTIVE, UNSTABLE,
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

    fn below(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n - 1)
    }
}

fn dense(input: usize, n_in: usize, n_out: usize, rng: &mut Lcg, scale: f64) -> GraphSearchOp {
    GraphSearchOp::Dense {
        input,
        rows: 1,
        n_in,
        n_out,
        weight: rng.vec(n_out * n_in, scale),
        bias: rng.vec(n_out, 0.3),
    }
}

/// A 3×3, stride-1, pad-1 convolution over a `[cin][h][w]` tensor.
fn conv3x3(
    input: usize,
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
    rng: &mut Lcg,
) -> GraphSearchOp {
    let (k, p) = (cin * 9, h * w);
    let mut gather = vec![GATHER_PAD; k * p];
    for oy in 0..h {
        for ox in 0..w {
            let sp = oy * w + ox;
            for ci in 0..cin {
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = oy as isize + ky as isize - 1;
                        let ix = ox as isize + kx as isize - 1;
                        if iy >= 0 && (iy as usize) < h && ix >= 0 && (ix as usize) < w {
                            gather[(ci * 9 + ky * 3 + kx) * p + sp] =
                                (ci * h * w + iy as usize * w + ix as usize) as u32;
                        }
                    }
                }
            }
        }
    }
    GraphSearchOp::Conv(Box::new(SearchConv {
        input,
        cout,
        k,
        p,
        gather,
        weight: rng.vec(cout * k, 0.6),
        bias: rng.vec(cout, 0.2),
    }))
}

/// `out = a + b` elementwise (a residual add as an exact sparse map).
fn add(a: usize, b: usize, n: usize) -> GraphSearchOp {
    let mut terms = Vec::with_capacity(2 * n);
    for i in 0..n {
        for slot in 0..2u32 {
            terms.push(SparseTerm {
                out: i as u32,
                slot,
                index: i as u32,
                weight: 1.0,
            });
        }
    }
    GraphSearchOp::Sparse {
        inputs: vec![a, b],
        n_out: n,
        terms,
        bias: vec![0.0; n],
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// 4 → 6 → relu → 5 → relu → 3.
    Mlp,
    /// [1][4][4] → conv(2) → relu → dense 3.
    Conv,
    /// [1][4][4] → conv(2) → relu → conv(2) → (+ relu1) → relu → dense 3.
    Residual,
}

const CLASSES: usize = 3;

fn net(shape: Shape, seed: u64) -> GraphSearchNet {
    let mut rng = Lcg(seed);
    match shape {
        Shape::Mlp => GraphSearchNet {
            n_in: 4,
            ops: vec![
                dense(0, 4, 6, &mut rng, 0.9),
                GraphSearchOp::Relu { input: 1 },
                dense(2, 6, 5, &mut rng, 0.8),
                GraphSearchOp::Relu { input: 3 },
                dense(4, 5, CLASSES, &mut rng, 0.8),
            ],
        },
        Shape::Conv => GraphSearchNet {
            n_in: 16,
            ops: vec![
                conv3x3(0, 1, 2, 4, 4, &mut rng),
                GraphSearchOp::Relu { input: 1 },
                dense(2, 32, CLASSES, &mut rng, 0.4),
            ],
        },
        Shape::Residual => GraphSearchNet {
            n_in: 16,
            ops: vec![
                conv3x3(0, 1, 2, 4, 4, &mut rng),
                GraphSearchOp::Relu { input: 1 },
                conv3x3(2, 2, 2, 4, 4, &mut rng),
                add(3, 2, 32),
                GraphSearchOp::Relu { input: 4 },
                dense(5, 32, CLASSES, &mut rng, 0.4),
            ],
        },
    }
}

/// Outward-widened interval.
fn widen(l: f64, u: f64) -> (f64, f64) {
    (l - 1e-9 * (1.0 + l.abs()), u + 1e-9 * (1.0 + u.abs()))
}

/// Sound interval forward (every tensor's box), written from scratch.
fn interval_forward(net: &GraphSearchNet, lo: &[f64], hi: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut boxes = vec![(lo.to_vec(), hi.to_vec())];
    for op in &net.ops {
        let out = match op {
            GraphSearchOp::Dense {
                input,
                n_in,
                n_out,
                weight,
                bias,
                ..
            } => {
                let (l, u) = &boxes[*input];
                let mut ol = vec![0.0; *n_out];
                let mut ou = vec![0.0; *n_out];
                for o in 0..*n_out {
                    let (mut a, mut b) = (bias[o], bias[o]);
                    for i in 0..*n_in {
                        let w = weight[o * n_in + i];
                        if w >= 0.0 {
                            a += w * l[i];
                            b += w * u[i];
                        } else {
                            a += w * u[i];
                            b += w * l[i];
                        }
                    }
                    (ol[o], ou[o]) = widen(a, b);
                }
                (ol, ou)
            }
            GraphSearchOp::Conv(c) => {
                let (l, u) = &boxes[c.input];
                let mut ol = vec![0.0; c.cout * c.p];
                let mut ou = vec![0.0; c.cout * c.p];
                for o in 0..c.cout {
                    for sp in 0..c.p {
                        let (mut a, mut b) = (c.bias[o], c.bias[o]);
                        for t in 0..c.k {
                            let g = c.gather[t * c.p + sp];
                            if g == GATHER_PAD {
                                continue;
                            }
                            let w = c.weight[o * c.k + t];
                            let g = g as usize;
                            if w >= 0.0 {
                                a += w * l[g];
                                b += w * u[g];
                            } else {
                                a += w * u[g];
                                b += w * l[g];
                            }
                        }
                        (ol[o * c.p + sp], ou[o * c.p + sp]) = widen(a, b);
                    }
                }
                (ol, ou)
            }
            GraphSearchOp::Sparse {
                inputs,
                n_out,
                terms,
                bias,
            } => {
                let mut ol = bias.clone();
                let mut ou = bias.clone();
                for t in terms {
                    let (l, u) = &boxes[inputs[t.slot as usize]];
                    let i = t.index as usize;
                    let o = t.out as usize;
                    if t.weight >= 0.0 {
                        ol[o] += t.weight * l[i];
                        ou[o] += t.weight * u[i];
                    } else {
                        ol[o] += t.weight * u[i];
                        ou[o] += t.weight * l[i];
                    }
                }
                for o in 0..*n_out {
                    (ol[o], ou[o]) = widen(ol[o], ou[o]);
                }
                (ol, ou)
            }
            GraphSearchOp::Relu { input } => {
                let (l, u) = &boxes[*input];
                (
                    l.iter().map(|v| v.max(0.0)).collect(),
                    u.iter().map(|v| v.max(0.0)).collect(),
                )
            }
            _ => unreachable!("fixture nets use Dense / Conv / Sparse / Relu only"),
        };
        boxes.push(out);
    }
    boxes
}

/// The batch request of `net` over the box `centre ± eps`: every tensor's
/// interval box as a reference box, the `target − other` margin rows.
fn batch(net: &GraphSearchNet, seed: u64, eps: f64) -> GraphBabRequest {
    let mut rng = Lcg(seed ^ 0x5eed);
    let centre = rng.vec(net.n_in, 1.0);
    let lo: Vec<f64> = centre.iter().map(|c| c - eps).collect();
    let hi: Vec<f64> = centre.iter().map(|c| c + eps).collect();
    let boxes = interval_forward(net, &lo, &hi);
    let target = rng.below(CLASSES);
    let rows: Vec<Vec<f64>> = (0..CLASSES)
        .filter(|&c| c != target)
        .map(|c| {
            let mut row = vec![0.0; CLASSES];
            row[target] = 1.0;
            row[c] = -1.0;
            row
        })
        .collect();
    let thresholds = vec![0.0; rows.len()];
    GraphBabRequest {
        lo,
        hi,
        boxes: boxes
            .into_iter()
            .enumerate()
            .map(|(tensor, (l, u))| TensorBox { tensor, l, u })
            .collect(),
        rows,
        thresholds,
        domains: vec![GraphDomainRequest::default()],
        iters: 20,
        lr_alpha: 0.1,
        lr_beta: 0.05,
        deadline: None,
        threads: 1,
        mem_cap_bytes: 1 << 30,
    }
}

/// `(relu op, input tensor)` of every ReLU of `net`.
fn relu_ops(net: &GraphSearchNet) -> Vec<(usize, usize)> {
    net.ops
        .iter()
        .enumerate()
        .filter_map(|(k, op)| match op {
            GraphSearchOp::Relu { input } => Some((k, *input)),
            _ => None,
        })
        .collect()
}

/// A domain of `n_splits` random literals on neurons unstable in the batch
/// boxes, with its own clamped pre-activation boxes (split sides one-signed).
fn random_domain(
    net: &GraphSearchNet,
    req: &GraphBabRequest,
    n_splits: usize,
    rng: &mut Lcg,
) -> GraphDomainRequest {
    let mut candidates = Vec::new();
    for (op, input) in relu_ops(net) {
        let b = &req.boxes[input];
        for j in 0..b.l.len() {
            if b.l[j] < 0.0 && b.u[j] > 0.0 {
                candidates.push((op, input, j));
            }
        }
    }
    let mut dom = GraphDomainRequest::default();
    let mut own: Vec<TensorBox> = Vec::new();
    for _ in 0..n_splits.min(candidates.len()) {
        let pick = candidates.swap_remove(rng.below(candidates.len()));
        let active = rng.unit() < 0.5;
        dom.splits.push(GraphDomainSplit {
            op: pick.0,
            neuron: pick.2 as u32,
            active,
        });
        let slot = match own.iter().position(|b| b.tensor == pick.1) {
            Some(slot) => slot,
            None => {
                own.push(req.boxes[pick.1].clone());
                own.len() - 1
            }
        };
        if active {
            own[slot].l[pick.2] = 0.0;
        } else {
            own[slot].u[pick.2] = 0.0;
        }
    }
    dom.boxes = own;
    dom
}

/// The margin of `row` at the point `x` (the net's own f64 forward).
fn margin_at(net: &GraphSearchNet, row: &[f64], x: &[f64]) -> (f64, Vec<Vec<f64>>) {
    let vals = net.forward_f64(x).expect("forward");
    let m = row
        .iter()
        .zip(&vals[net.output()])
        .map(|(c, y)| c * y)
        .sum();
    (m, vals)
}

/// Sample points of the input box: corners of random sign patterns and
/// uniform interior points.
fn samples(lo: &[f64], hi: &[f64], n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|s| {
            lo.iter()
                .zip(hi)
                .map(|(&l, &u)| {
                    if s % 3 == 0 {
                        if rng.unit() < 0.5 {
                            l
                        } else {
                            u
                        }
                    } else {
                        rng.uniform(l, u)
                    }
                })
                .collect()
        })
        .collect()
}

/// `true` when `vals` satisfies every split literal of `dom` (`input` of
/// each split's ReLU from `net`).
fn in_domain(net: &GraphSearchNet, dom: &GraphDomainRequest, vals: &[Vec<f64>]) -> bool {
    dom.splits.iter().all(|s| {
        let GraphSearchOp::Relu { input } = &net.ops[s.op] else {
            return false;
        };
        let z = vals[*input][s.neuron as usize];
        if s.active {
            z >= 0.0
        } else {
            z <= 0.0
        }
    })
}

fn random_params<R: crate::real::Real>(
    prog: &GProgram<R>,
    dom: &DomainSetup<R>,
    rng: &mut Lcg,
) -> (Alpha<R>, Vec<R>) {
    let base = prog.init_alpha();
    let relu = dom
        .relu
        .iter()
        .map(|s| {
            s.status
                .iter()
                .map(|_| R::from_f64(rng.uniform(0.0, 1.0)))
                .collect()
        })
        .collect();
    let beta = dom
        .lits
        .iter()
        .map(|_| R::from_f64(rng.uniform(0.0, 2.0)))
        .collect();
    (
        Alpha {
            relu,
            bil: base.bil,
            mul: base.mul,
        },
        beta,
    )
}

fn view_of<'a, R: crate::real::Real>(
    dom: &'a DomainSetup<R>,
    inject: &'a [(usize, u32, R)],
) -> View<'a, R> {
    View {
        relu: &dom.relu,
        lo: &dom.lo,
        hi: &dom.hi,
        inject,
    }
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

const SHAPES: [Shape; 3] = [Shape::Mlp, Shape::Conv, Shape::Residual];

// ---------------------------------------------------------------------------
//  Known answers and soundness of the search's own relaxation
// ---------------------------------------------------------------------------

/// `x ∈ [−1, 1]`, `z = x`, `y = relu(z)`, output `z` (the ReLU feeds nothing
/// the row reads). On the domain `z ≥ 0` the Lagrangian bound is
/// `min_x (1 − β) x = −|1 − β|`: `−1` at `β = 0`, `0` at `β = 1`.
fn kat_net() -> (GraphSearchNet, GraphBabRequest) {
    let net = GraphSearchNet {
        n_in: 1,
        ops: vec![
            GraphSearchOp::Dense {
                input: 0,
                rows: 1,
                n_in: 1,
                n_out: 1,
                weight: vec![1.0],
                bias: vec![0.0],
            },
            GraphSearchOp::Relu { input: 1 },
            GraphSearchOp::Sparse {
                inputs: vec![1, 2],
                n_out: 1,
                terms: vec![SparseTerm {
                    out: 0,
                    slot: 0,
                    index: 0,
                    weight: 1.0,
                }],
                bias: vec![0.0],
            },
        ],
    };
    let req = GraphBabRequest {
        lo: vec![-1.0],
        hi: vec![1.0],
        boxes: vec![TensorBox {
            tensor: 1,
            l: vec![-1.0],
            u: vec![1.0],
        }],
        rows: vec![vec![1.0]],
        thresholds: vec![0.5],
        domains: vec![GraphDomainRequest {
            splits: vec![GraphDomainSplit {
                op: 1,
                neuron: 0,
                active: true,
            }],
            ..GraphDomainRequest::default()
        }],
        iters: 60,
        lr_alpha: 0.1,
        lr_beta: 0.1,
        deadline: None,
        threads: 1,
        mem_cap_bytes: 1 << 24,
    };
    (net, req)
}

#[test]
fn graph_bab_kat_split_multiplier_bound_and_gradient() {
    let (net, req) = kat_net();
    let (prog, bb) = lower_batch::<f64>(&net, &req).expect("lower");
    let dom = setup_domain(&prog, &req, &bb, &req.domains[0]).expect("setup");
    assert_eq!(dom.lits, vec![(1, 0, 1.0)], "one active literal on z");
    assert_eq!(
        dom.relu[0].status,
        vec![ACTIVE],
        "the split neuron is pinned active"
    );
    let alpha = prog.init_alpha();
    for (beta, bound, gbeta) in [
        (0.0, -1.0, 1.0),
        (0.5, -0.5, 1.0),
        (1.0, 0.0, f64::NAN),
        (1.5, -0.5, -1.0),
    ] {
        let pass = domain_pass(&prog, &dom, &alpha, &[beta]);
        assert!(
            (pass.bounds[0] - bound).abs() < 1e-12,
            "beta {beta}: bound {} != {bound}",
            pass.bounds[0]
        );
        if gbeta.is_finite() {
            assert!(
                (pass.gbeta[0] - gbeta).abs() < 1e-12,
                "beta {beta}: d/dbeta {} != {gbeta}",
                pass.gbeta[0]
            );
        }
    }
    // The search climbs to the optimum `beta = 1` (bound 0 > -1).
    let cfg = BabConfig {
        iters: 200,
        lr_alpha: 0.1,
        lr_beta: 0.1,
        deadline: None,
    };
    let found = search_domain(&prog, &dom, &cfg);
    assert!(
        found.best_score.min_slack > -0.5 - 0.1,
        "best {:?}",
        found.best_score
    );
    assert!(
        found.hint.beta[0] > 0.8 && found.hint.beta[0] < 1.2,
        "beta {:?}",
        found.hint.beta
    );
}

#[test]
fn graph_bab_f64_bounds_contain_sampled_truth_on_split_conv_and_residual_domains() {
    let mut checked_points = 0usize;
    let mut checked_bounds = 0usize;
    for (si, shape) in SHAPES.into_iter().enumerate() {
        for (seed, eps) in (0..8u64).zip([0.35, 0.1, 0.35, 0.15, 0.35, 0.1, 0.25, 0.05]) {
            let net = net(shape, 100 * si as u64 + seed);
            let req = batch(&net, seed, eps);
            let (prog, bb) = lower_batch::<f64>(&net, &req).expect("lower");
            let mut rng = Lcg(seed ^ 0xd0);
            for n_splits in [0usize, 1, 2, 3] {
                let dreq = random_domain(&net, &req, n_splits, &mut rng);
                let dom = setup_domain(&prog, &req, &bb, &dreq).expect("setup");
                for draw in 0..4 {
                    let (mut alpha, mut beta) = random_params(&prog, &dom, &mut rng);
                    if draw == 0 {
                        // The starting point: adaptive slopes, zero multipliers.
                        alpha.relu = dom.relu.iter().map(|s| s.init.clone()).collect();
                        beta.iter_mut().for_each(|b| *b = 0.0);
                    }
                    let inject = crate::graph_bab::injections(&dom, &beta);
                    let view = view_of(&dom, &inject);
                    let pl = planes(&prog, &alpha);
                    let bounds: Vec<f64> = (0..prog.rows.len())
                        .map(|r| {
                            let out = eval_row(&prog, &view, &alpha, &pl, r);
                            assert!(
                                (out.lin_value - out.bound).abs() <= 1e-9 * (1.0 + out.bound.abs()),
                                "{shape:?} seed {seed}: linearised {} vs bound {}",
                                out.lin_value,
                                out.bound
                            );
                            out.bound
                        })
                        .collect();
                    checked_bounds += bounds.len();
                    for x in samples(&req.lo, &req.hi, 600, seed ^ 0xabc) {
                        let (_, vals) = margin_at(&net, &req.rows[0], &x);
                        if !in_domain(&net, &dreq, &vals) {
                            continue;
                        }
                        checked_points += 1;
                        for (r, row) in req.rows.iter().enumerate() {
                            let (m, _) = margin_at(&net, row, &x);
                            assert!(
                                bounds[r] <= m + 1e-9 * (1.0 + m.abs()),
                                "{shape:?} seed {seed} splits {n_splits} row {r}: bound {} > \
                                 true margin {m}",
                                bounds[r]
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        checked_points > 2_000,
        "too few in-domain samples: {checked_points}"
    );
    assert!(checked_bounds > 100, "too few bounds: {checked_bounds}");
}

#[test]
fn graph_bab_mutation_pins_catch_a_wrong_piece_and_a_flipped_multiplier() {
    // (a) Pinning an ACTIVE split neuron to the inactive piece (y = 0 where
    // z ≥ 0) is unsound on the domain: some sampled point must violate.
    let mut wrong_piece_caught = 0usize;
    for seed in 0..12u64 {
        let net = net(Shape::Mlp, 700 + seed);
        let req = batch(&net, seed, 0.5);
        let (prog, bb) = lower_batch::<f64>(&net, &req).expect("lower");
        let mut rng = Lcg(seed);
        let mut dreq = random_domain(&net, &req, 1, &mut rng);
        let Some(split) = dreq.splits.first_mut() else {
            continue;
        };
        split.active = true;
        dreq.boxes.clear();
        let mut dom = setup_domain(&prog, &req, &bb, &dreq).expect("setup");
        let site = dom
            .relu
            .iter()
            .position(|s| s.op == dreq.splits[0].op)
            .expect("site");
        dom.relu[site].status[dreq.splits[0].neuron as usize] = INACTIVE; // MUTATION
        let alpha = prog.init_alpha();
        let alpha = Alpha {
            relu: dom.relu.iter().map(|s| s.init.clone()).collect(),
            ..alpha
        };
        let pass = domain_pass(&prog, &dom, &alpha, &[0.0]);
        'points: for x in samples(&req.lo, &req.hi, 600, seed) {
            let (_, vals) = margin_at(&net, &req.rows[0], &x);
            if !in_domain(&net, &dreq, &vals) {
                continue;
            }
            for (r, row) in req.rows.iter().enumerate() {
                let (m, _) = margin_at(&net, row, &x);
                if pass.bounds[r] > m + 1e-6 {
                    wrong_piece_caught += 1;
                    break 'points;
                }
            }
        }
    }
    assert!(
        wrong_piece_caught > 0,
        "the containment oracle must catch a wrong piece"
    );
    // (b) The KAT catches a flipped multiplier sign: `+β s z` reads -1-β.
    let (net, req) = kat_net();
    let (prog, bb) = lower_batch::<f64>(&net, &req).expect("lower");
    let dom = setup_domain(&prog, &req, &bb, &req.domains[0]).expect("setup");
    let alpha = prog.init_alpha();
    let flipped = [(1usize, 0u32, 1.0f64)]; // MUTATION: +β s with β = 1
    let view = view_of(&dom, &flipped);
    let out = eval_row(&prog, &view, &alpha, &planes(&prog, &alpha), 0);
    assert!(
        (out.bound - -2.0).abs() < 1e-12,
        "flipped sign reads {}",
        out.bound
    );
    let right = domain_pass(&prog, &dom, &alpha, &[1.0]);
    assert!(
        right.bounds[0] > out.bound + 1.0,
        "the correct sign is the tighter one"
    );
}

#[test]
fn graph_bab_gradient_matches_central_differences() {
    let mut checked_alpha = 0usize;
    let mut checked_beta = 0usize;
    for (si, shape) in SHAPES.into_iter().enumerate() {
        for seed in 0..4u64 {
            let net = net(shape, 300 + 10 * si as u64 + seed);
            let mut req = batch(&net, seed, 0.3);
            // Every row counts in the loss.
            req.thresholds = vec![1e9; req.rows.len()];
            let (prog, bb) = lower_batch::<f64>(&net, &req).expect("lower");
            let mut rng = Lcg(seed ^ 0x9d);
            let dreq = random_domain(&net, &req, 2, &mut rng);
            let dom = setup_domain(&prog, &req, &bb, &dreq).expect("setup");
            let (alpha, beta) = random_params(&prog, &dom, &mut rng);
            let beta: Vec<f64> = beta.iter().map(|b| 0.2 + b).collect();
            let pass = domain_pass(&prog, &dom, &alpha, &beta);
            let loss = |a: &Alpha<f64>, b: &[f64]| -> f64 {
                domain_pass(&prog, &dom, a, b).bounds.iter().sum()
            };
            let h = 1e-6;
            let close = |fd: f64, an: f64| (fd - an).abs() <= 1e-5 * (1.0 + fd.abs());
            for (k, s) in dom.relu.iter().enumerate() {
                for j in 0..s.status.len() {
                    if s.status[j] != UNSTABLE {
                        continue;
                    }
                    let (mut up, mut dn) = (alpha.clone(), alpha.clone());
                    up.relu[k][j] += h;
                    dn.relu[k][j] -= h;
                    let (fu, f0, fd) = (loss(&up, &beta), loss(&alpha, &beta), loss(&dn, &beta));
                    let (right, left) = ((fu - f0) / h, (f0 - fd) / h);
                    if !close(right, left) {
                        continue; // a kink (a backward sign flips inside ±h)
                    }
                    let central = (fu - fd) / (2.0 * h);
                    assert!(
                        close(central, pass.grad.relu[k][j]),
                        "{shape:?} seed {seed} alpha[{k}][{j}]: fd {central} vs {}",
                        pass.grad.relu[k][j]
                    );
                    checked_alpha += 1;
                }
            }
            for i in 0..beta.len() {
                let (mut up, mut dn) = (beta.clone(), beta.clone());
                up[i] += h;
                dn[i] -= h;
                let (fu, f0, fd) = (loss(&alpha, &up), loss(&alpha, &beta), loss(&alpha, &dn));
                let (right, left) = ((fu - f0) / h, (f0 - fd) / h);
                if !close(right, left) {
                    continue;
                }
                let central = (fu - fd) / (2.0 * h);
                assert!(
                    close(central, pass.gbeta[i]),
                    "{shape:?} seed {seed} beta[{i}]: fd {central} vs {}",
                    pass.gbeta[i]
                );
                checked_beta += 1;
            }
        }
    }
    assert!(
        checked_alpha > 40,
        "too few slope gradients checked: {checked_alpha}"
    );
    assert!(
        checked_beta > 8,
        "too few multiplier gradients checked: {checked_beta}"
    );
}

// ---------------------------------------------------------------------------
//  The search: V2 equivalence, determinism, keep-best, tightening
// ---------------------------------------------------------------------------

#[test]
fn graph_bab_empty_domain_reproduces_the_root_search_bit_for_bit() {
    for (si, shape) in SHAPES.into_iter().enumerate() {
        for seed in 0..4u64 {
            let net = net(shape, 500 + 10 * si as u64 + seed);
            let req = batch(&net, seed, 0.3);
            let root = GraphRootRequest {
                lo: req.lo.clone(),
                hi: req.hi.clone(),
                boxes: req.boxes.clone(),
                rows: req.rows.clone(),
                thresholds: req.thresholds.clone(),
                init: GraphAlphaHint::default(),
                iters: req.iters,
                lr: req.lr_alpha,
                deadline: None,
                threads: 1,
                mem_cap_bytes: req.mem_cap_bytes,
            };
            let prog = GProgram::<f32>::lower(&net, &root).expect("lower");
            for threads in [1usize, 2, 3] {
                let v2 = search(
                    &prog,
                    &GConfig {
                        iters: req.iters,
                        lr: req.lr_alpha,
                        deadline: None,
                        threads,
                        mem_cap: req.mem_cap_bytes,
                    },
                )
                .expect("v2 search");
                let out = run_graph_bab::<f32>(&net, &req, threads).expect("m7 batch");
                let hints = out.hints.into_hint().into_ingest();
                let m7 = hints[0].as_ref().expect("the unsplit domain is searched");
                assert_eq!(
                    hint_bits(&m7.alpha),
                    hint_bits(&v2.hint),
                    "{shape:?} seed {seed} threads {threads}: the unsplit domain is the V2 search"
                );
                assert_eq!(m7.beta, [] as [f32; 0]);
                let stats = out.stats.per_domain[0].as_ref().expect("stats");
                assert_eq!(
                    (stats.iters, stats.evals, stats.stop),
                    (v2.iters, v2.evals, v2.stop)
                );
                assert_eq!(
                    format!("{}", stats.best_slack),
                    format!("{}", v2.best_score.min_slack)
                );
            }
        }
    }
}

fn batch_of_domains(net: &GraphSearchNet, seed: u64, n: usize) -> GraphBabRequest {
    let mut req = batch(net, seed, 0.35);
    let mut rng = Lcg(seed ^ 0xba7c);
    req.domains = (0..n)
        .map(|k| {
            let mut d = random_domain(net, &req, k % 4, &mut rng);
            if k % 3 == 1 {
                d.rows = vec![0];
            }
            if k % 5 == 2 {
                d.init_beta = vec![0.3; d.splits.len()];
            }
            d
        })
        .collect();
    req
}

/// A domain's hint and stats as comparable bits.
type DomainBits = (Vec<(usize, Vec<u32>)>, Vec<u32>, usize, usize, &'static str);

fn outcome_bits(out: ny_core::search::GraphBabOutcome) -> Vec<Option<DomainBits>> {
    let stats = out.stats.per_domain;
    out.hints
        .into_hint()
        .into_ingest()
        .into_iter()
        .zip(stats)
        .map(|(h, s)| {
            h.map(|h| {
                let s = s.expect("a hint has stats");
                (hint_bits(&h.alpha), bits(&h.beta), s.iters, s.evals, s.stop)
            })
        })
        .collect()
}

#[test]
fn graph_bab_is_thread_count_independent() {
    for (si, shape) in SHAPES.into_iter().enumerate() {
        let net = net(shape, 900 + si as u64);
        let req = batch_of_domains(&net, si as u64, 9);
        let reference = outcome_bits(run_graph_bab::<f32>(&net, &req, 1).expect("t1"));
        assert!(
            reference.iter().all(Option::is_some),
            "every domain is searched"
        );
        for threads in [2usize, 3, 8] {
            let got = outcome_bits(run_graph_bab::<f32>(&net, &req, threads).expect("tn"));
            assert_eq!(
                got, reference,
                "{shape:?}: threads {threads} changed a hint"
            );
        }
    }
}

#[test]
fn graph_bab_domain_results_do_not_depend_on_the_batch() {
    let net = net(Shape::Residual, 77);
    let req = batch_of_domains(&net, 5, 7);
    let together = outcome_bits(run_graph_bab::<f32>(&net, &req, 3).expect("batch"));
    for k in 0..req.domains.len() {
        let mut alone = req.clone();
        alone.domains = vec![req.domains[k].clone()];
        let single = outcome_bits(run_graph_bab::<f32>(&net, &alone, 1).expect("alone"));
        assert_eq!(single[0], together[k], "domain {k} depends on its batch");
    }
}

#[test]
fn graph_bab_multipliers_are_nonnegative_and_split_neurons_are_pinned() {
    for (si, shape) in SHAPES.into_iter().enumerate() {
        let net = net(shape, 1_100 + si as u64);
        let req = batch_of_domains(&net, 11 + si as u64, 8);
        let (prog, _) = lower_batch::<f32>(&net, &req).expect("lower");
        let hints = run_graph_bab::<f32>(&net, &req, 2)
            .expect("batch")
            .hints
            .into_hint()
            .into_ingest();
        for (dreq, hint) in req.domains.iter().zip(&hints) {
            let hint = hint.as_ref().expect("searched");
            assert_eq!(hint.beta.len(), dreq.splits.len());
            assert!(
                hint.beta.iter().all(|b| b.is_finite() && *b >= 0.0),
                "{:?}",
                hint.beta
            );
            assert_eq!(hint.alpha.relu.len(), prog.relu.len());
            for ((op, slopes), site) in hint.alpha.relu.iter().zip(&prog.relu) {
                assert_eq!(*op, site.op);
                assert_eq!(slopes.len(), site.status.len());
                assert!(slopes
                    .iter()
                    .all(|a| a.is_finite() && (0.0..=1.0).contains(a)));
            }
            for split in &dreq.splits {
                let (_, slopes) = hint
                    .alpha
                    .relu
                    .iter()
                    .find(|(op, _)| *op == split.op)
                    .expect("split site in the hint");
                let want = if split.active { 1.0 } else { 0.0 };
                assert_eq!(
                    slopes[split.neuron as usize], want,
                    "split neuron exact limit"
                );
            }
        }
    }
}

#[test]
fn graph_bab_never_returns_a_worse_score() {
    for (si, shape) in SHAPES.into_iter().enumerate() {
        let net = net(shape, 1_300 + si as u64);
        let req = batch_of_domains(&net, 21 + si as u64, 6);
        let (prog, bb) = lower_batch::<f64>(&net, &req).expect("lower");
        let cfg = BabConfig {
            iters: 30,
            lr_alpha: 0.2,
            lr_beta: 0.1,
            deadline: None,
        };
        for dreq in &req.domains {
            let dom = setup_domain(&prog, &req, &bb, dreq).expect("setup");
            let found = search_domain(&prog, &dom, &cfg);
            assert!(
                !improves(found.init, found.best_score),
                "{shape:?}: returned {:?} is worse than the start {:?}",
                found.best_score,
                found.init
            );
            // The returned parameters reproduce the returned score.
            let again = domain_pass(&prog, &dom, &found.best_alpha, &found.best_beta);
            let thr: Vec<f64> = dom.rows.iter().map(|&r| prog.thr[r]).collect();
            assert_eq!(
                crate::graph_root::score(&again.bounds, &thr),
                found.best_score
            );
        }
    }
}

#[test]
fn graph_bab_multipliers_tighten_a_split_domain() {
    // On split domains the multipliers must be worth something somewhere:
    // the best bound with β searched beats the best bound with β held at 0 on
    // at least one domain, and never loses (β = 0 is the starting point).
    let mut strictly_better = 0usize;
    for seed in 0..10u64 {
        let net = net(Shape::Mlp, 1_500 + seed);
        let req = batch(&net, seed, 0.6);
        let (prog, bb) = lower_batch::<f64>(&net, &req).expect("lower");
        let mut rng = Lcg(seed ^ 0x77);
        let dreq = random_domain(&net, &req, 3, &mut rng);
        if dreq.splits.is_empty() {
            continue;
        }
        let dom = setup_domain(&prog, &req, &bb, &dreq).expect("setup");
        let with_beta = search_domain(
            &prog,
            &dom,
            &BabConfig {
                iters: 40,
                lr_alpha: 0.1,
                lr_beta: 0.2,
                deadline: None,
            },
        );
        let without = search_domain(
            &prog,
            &dom,
            &BabConfig {
                iters: 40,
                lr_alpha: 0.1,
                lr_beta: 1e-300,
                deadline: None,
            },
        );
        if improves(with_beta.best_score, without.best_score) {
            strictly_better += 1;
        }
    }
    assert!(
        strictly_better > 0,
        "the multipliers never tightened a split domain"
    );
}

#[test]
fn graph_bab_f32_tracks_the_f64_reference() {
    for (si, shape) in SHAPES.into_iter().enumerate() {
        for seed in 0..3u64 {
            let net = net(shape, 1_700 + 10 * si as u64 + seed);
            let req = batch_of_domains(&net, seed, 4);
            let (p32, b32) = lower_batch::<f32>(&net, &req).expect("lower f32");
            let (p64, b64) = lower_batch::<f64>(&net, &req).expect("lower f64");
            let mut rng = Lcg(seed);
            for dreq in &req.domains {
                let d32 = setup_domain(&p32, &req, &b32, dreq).expect("setup f32");
                let d64 = setup_domain(&p64, &req, &b64, dreq).expect("setup f64");
                let (a64, beta64) = random_params(&p64, &d64, &mut rng);
                let a32 = Alpha {
                    relu: a64
                        .relu
                        .iter()
                        .map(|v| v.iter().map(|&x| x as f32).collect())
                        .collect(),
                    bil: Vec::new(),
                    mul: Vec::new(),
                };
                let beta32: Vec<f32> = beta64.iter().map(|&b| b as f32).collect();
                let lo = domain_pass(&p32, &d32, &a32, &beta32).bounds;
                let hi = domain_pass(&p64, &d64, &a64, &beta64).bounds;
                for (a, b) in lo.iter().zip(&hi) {
                    assert!(
                        (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                        "{shape:?}: f32 {a} vs f64 {b}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Refusals and the public backend
// ---------------------------------------------------------------------------

#[test]
fn graph_bab_refuses_malformed_requests_and_isolates_malformed_domains() {
    let net = net(Shape::Mlp, 2_000);
    let good = batch_of_domains(&net, 3, 3);
    let (prog, bb) = lower_batch::<f64>(&net, &good).expect("lower");

    // Whole-request refusals.
    let mut r = good.clone();
    r.domains.clear();
    assert!(matches!(
        run_graph_bab::<f32>(&net, &r, 1),
        Err(Refusal::Nothing(_))
    ));
    let mut r = good.clone();
    r.lo[0] = f64::NAN;
    assert!(matches!(
        run_graph_bab::<f32>(&net, &r, 1),
        Err(Refusal::Invalid(_))
    ));
    let mut r = good.clone();
    r.boxes[1].l.pop();
    assert!(matches!(
        run_graph_bab::<f32>(&net, &r, 1),
        Err(Refusal::Invalid(_))
    ));
    let mut r = good.clone();
    r.thresholds.pop();
    assert!(matches!(
        run_graph_bab::<f32>(&net, &r, 1),
        Err(Refusal::Invalid(_))
    ));
    let mut r = good.clone();
    r.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    assert!(matches!(
        run_graph_bab::<f32>(&net, &r, 1),
        Err(Refusal::Deadline)
    ));
    let mut r = good.clone();
    r.mem_cap_bytes = 64;
    assert!(matches!(
        run_graph_bab::<f32>(&net, &r, 4),
        Err(Refusal::Memory { .. })
    ));

    // Per-domain refusals: named reasons, and the batch's other domains keep
    // their hints.
    let relus = relu_ops(&net);
    let (op0, in0) = relus[0];
    let width = good.boxes[in0].l.len();
    let lit = |op: usize, neuron: usize, active: bool| GraphDomainSplit {
        op,
        neuron: neuron as u32,
        active,
    };
    let bad: Vec<(GraphDomainRequest, DomainRefusal)> = vec![
        (
            GraphDomainRequest {
                splits: vec![lit(0, 0, true)],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::SplitNotRelu,
        ),
        (
            GraphDomainRequest {
                splits: vec![lit(op0, width, true)],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::SplitNeuronRange,
        ),
        (
            GraphDomainRequest {
                splits: vec![lit(op0, 0, true), lit(op0, 0, false)],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::SplitContradiction,
        ),
        (
            GraphDomainRequest {
                boxes: vec![TensorBox {
                    tensor: in0,
                    l: vec![1.0; width],
                    u: vec![0.0; width],
                }],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::BadBox,
        ),
        (
            GraphDomainRequest {
                boxes: vec![good.boxes[in0].clone(), good.boxes[in0].clone()],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::BadBox,
        ),
        (
            GraphDomainRequest {
                lo: vec![0.0],
                hi: vec![1.0],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::BadInputBox,
        ),
        (
            GraphDomainRequest {
                rows: vec![0, 0],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::BadRows,
        ),
        (
            GraphDomainRequest {
                rows: vec![good.rows.len()],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::BadRows,
        ),
        (
            GraphDomainRequest {
                boxes: vec![TensorBox {
                    tensor: in0,
                    l: vec![f64::NEG_INFINITY; width],
                    u: vec![1.0; width],
                }],
                ..GraphDomainRequest::default()
            },
            DomainRefusal::NonFiniteUnstable,
        ),
    ];
    for (dreq, why) in &bad {
        assert_eq!(
            setup_domain(&prog, &good, &bb, dreq).err(),
            Some(*why),
            "domain {dreq:?}"
        );
    }
    let mut mixed = good.clone();
    mixed.domains = good
        .domains
        .iter()
        .cloned()
        .chain(bad.iter().map(|(d, _)| d.clone()))
        .collect();
    let out = run_graph_bab::<f32>(&net, &mixed, 3).expect("mixed batch");
    assert_eq!(out.stats.domains, mixed.domains.len());
    assert_eq!(out.stats.searched, good.domains.len());
    assert_eq!(out.stats.refused, bad.len());
    assert_eq!(out.stats.late, 0);
    let hints = out.hints.into_hint().into_ingest();
    assert!(hints[..good.domains.len()].iter().all(Option::is_some));
    assert!(hints[good.domains.len()..].iter().all(Option::is_none));
    // The good domains' hints are exactly their standalone ones.
    let alone = outcome_bits(run_graph_bab::<f32>(&net, &good, 1).expect("good"));
    let again = outcome_bits(run_graph_bab::<f32>(&net, &mixed, 1).expect("mixed"));
    assert_eq!(&again[..good.domains.len()], &alone[..]);
}

#[test]
fn graph_bab_cpu_backend_serves_the_batch_through_the_trait() {
    let net = net(Shape::Conv, 2_100);
    let mut req = batch_of_domains(&net, 7, 5);
    req.threads = 0; // the backend's own worker count applies
    let backend: &dyn SearchBackend = &CpuF32Backend::with_threads(2);
    let out = backend.graph_bab_batch(&net, &req).expect("served");
    assert_eq!(out.stats.domains, 5);
    assert_eq!(out.stats.threads, 2);
    assert_eq!(out.stats.relu_sites, 1);
    let direct = outcome_bits(run_graph_bab::<f32>(&net, &req, 1).expect("direct"));
    assert_eq!(
        outcome_bits(out),
        direct,
        "the trait serves the engine unchanged"
    );
}
