// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M5a search tests. Every oracle here is INDEPENDENT of the engine: a dense
//! f64 CROWN written from scratch over the [`SearchNet`] (no lanes, no touched
//! sets, no lowering), interval propagation for reference boxes, and sampled
//! f64 forward evaluation for truth.
//!
//! * `f32_search_root_tracks_an_independent_f64_reference` — the f32 search's
//!   reported per-class bounds equal the dense f64 CROWN of the SAME slopes and
//!   boxes to f32 accuracy, and its root tracks the f64 instantiation of the
//!   search.
//! * `f64_search_bounds_and_boxes_enclose_sampled_truth` — the search's own
//!   math is a valid relaxation: in f64 every bound it reports lower-bounds
//!   every sampled margin and every box contains every sampled pre-activation
//!   (a sign or line-selection bug fails here).
//! * `crown_init_equals_the_reference_crown_with_adaptive_slopes`.
//! * `objective_and_intermediate_gradients_match_central_differences`.
//! * `search_result_is_independent_of_the_thread_count` (R6).
//! * `gradient_passes_stream_in_bounded_chunks_bit_identically` — the memory
//!   contract: chunked streaming is bit-identical and holds one chunk.
//! * `hint_is_well_formed_with_exact_limits_on_stable_neurons` (public API).
//! * `per_row_and_shared_slopes_both_search`.
//! * `spec_prune_and_crown_verified_exit`.
//! * `invalid_requests_and_budgets_refuse`.
//! * `lowering_transposes_the_gather_and_records_the_sites`.

use std::time::{Duration, Instant};

use ny_core::search::{
    Refusal, ReluBox, RootSearchRequest, SearchBackend, SearchConv, SearchNet, SearchOp, GATHER_PAD,
};

use crate::cpu::run_root;
use crate::joint_root::{eval_rows_for_test, search, Config, SearchResult};
use crate::program::{Op, Program, NONE};
use crate::real::Real;
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

/// Gather table `[k][p]` of a zero-padded conv (the margin-row formula).
fn conv_gather(
    cin: usize,
    (ih, iw): (usize, usize),
    (kh, kw): (usize, usize),
    stride: usize,
    pad: usize,
) -> (Vec<u32>, usize) {
    let oh = (ih + 2 * pad - kh) / stride + 1;
    let ow = (iw + 2 * pad - kw) / stride + 1;
    let (k, p) = (cin * kh * kw, oh * ow);
    let mut gather = vec![GATHER_PAD; k * p];
    for oy in 0..oh {
        for ox in 0..ow {
            let sp = oy * ow + ox;
            for c in 0..cin {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy * stride + ky) as isize - pad as isize;
                        let ix = (ox * stride + kx) as isize - pad as isize;
                        if iy >= 0 && (iy as usize) < ih && ix >= 0 && (ix as usize) < iw {
                            gather[(c * kh * kw + ky * kw + kx) * p + sp] =
                                (c * ih * iw + iy as usize * iw + ix as usize) as u32;
                        }
                    }
                }
            }
        }
    }
    (gather, p)
}

/// A residual conv net of the margin-row twin-wall shape: 2x6x6 input, a
/// stride-2 conv, a residual block, two trunk ReLUs, a 5-wide head ReLU and 4
/// classes.
fn conv_net(seed: u64, scale: f64) -> SearchNet {
    let mut rng = Lcg(seed);
    let (g1, p1) = conv_gather(2, (6, 6), (3, 3), 2, 1);
    let (g2, p2) = conv_gather(4, (3, 3), (3, 3), 1, 1);
    SearchNet {
        n_in: 72,
        ops: vec![
            SearchOp::Conv(Box::new(SearchConv {
                input: 0,
                cout: 4,
                k: 18,
                p: p1,
                gather: g1,
                weight: rng.vec(4 * 18, scale),
                bias: rng.vec(4, scale),
            })),
            SearchOp::Relu {
                input: 1,
                head: false,
            },
            SearchOp::Conv(Box::new(SearchConv {
                input: 2,
                cout: 4,
                k: 36,
                p: p2,
                gather: g2,
                weight: rng.vec(4 * 36, scale),
                bias: rng.vec(4, scale),
            })),
            SearchOp::Add { lhs: 3, rhs: 2 },
            SearchOp::Relu {
                input: 4,
                head: false,
            },
            SearchOp::Flatten { input: 5 },
            SearchOp::Gemm {
                input: 6,
                n_out: 5,
                n_in: 36,
                weight: rng.vec(5 * 36, scale),
                bias: rng.vec(5, scale),
            },
            SearchOp::Relu {
                input: 7,
                head: true,
            },
            SearchOp::Gemm {
                input: 8,
                n_out: 4,
                n_in: 5,
                weight: rng.vec(4 * 5, 1.0),
                bias: rng.vec(4, 0.5),
            },
        ],
    }
}

/// A dense net with a per-neuron affine (a folded BN) between two ReLUs.
fn mlp_net(seed: u64, scale: f64) -> SearchNet {
    let mut rng = Lcg(seed);
    SearchNet {
        n_in: 6,
        ops: vec![
            SearchOp::Gemm {
                input: 0,
                n_out: 10,
                n_in: 6,
                weight: rng.vec(60, scale),
                bias: rng.vec(10, scale),
            },
            SearchOp::Relu {
                input: 1,
                head: false,
            },
            SearchOp::Affine {
                input: 2,
                scale: (0..10).map(|_| rng.uniform(0.5, 1.5)).collect(),
                shift: rng.vec(10, 0.2),
            },
            SearchOp::Gemm {
                input: 3,
                n_out: 10,
                n_in: 10,
                weight: rng.vec(100, scale),
                bias: rng.vec(10, scale),
            },
            SearchOp::Relu {
                input: 4,
                head: false,
            },
            SearchOp::Gemm {
                input: 5,
                n_out: 6,
                n_in: 10,
                weight: rng.vec(60, scale),
                bias: rng.vec(6, scale),
            },
            SearchOp::Relu {
                input: 6,
                head: true,
            },
            SearchOp::Gemm {
                input: 7,
                n_out: 4,
                n_in: 6,
                weight: rng.vec(24, 1.0),
                bias: rng.vec(4, 0.5),
            },
        ],
    }
}

/// Interval propagation (plain f64): the reference box of every ReLU input.
fn ibp_boxes(net: &SearchNet, lo: &[f64], hi: &[f64]) -> Vec<ReluBox> {
    let mut l: Vec<Vec<f64>> = vec![lo.to_vec()];
    let mut u: Vec<Vec<f64>> = vec![hi.to_vec()];
    let mut boxes = Vec::new();
    let span = |w: f64, a: f64, b: f64| {
        if w >= 0.0 {
            (w * a, w * b)
        } else {
            (w * b, w * a)
        }
    };
    for (k, op) in net.ops.iter().enumerate() {
        let (nl, nu): (Vec<f64>, Vec<f64>) = match op {
            SearchOp::Conv(c) => {
                let (mut nl, mut nu) = (vec![0.0; c.cout * c.p], vec![0.0; c.cout * c.p]);
                for o in 0..c.cout {
                    for sp in 0..c.p {
                        let (mut a, mut b) = (c.bias[o], c.bias[o]);
                        for t in 0..c.k {
                            let g = c.gather[t * c.p + sp];
                            if g != GATHER_PAD {
                                let g = g as usize;
                                let (x, y) =
                                    span(c.weight[o * c.k + t], l[c.input][g], u[c.input][g]);
                                a += x;
                                b += y;
                            }
                        }
                        nl[o * c.p + sp] = a;
                        nu[o * c.p + sp] = b;
                    }
                }
                (nl, nu)
            }
            SearchOp::Affine {
                input,
                scale,
                shift,
            } => (0..scale.len())
                .map(|j| {
                    let (x, y) = span(scale[j], l[*input][j], u[*input][j]);
                    (x + shift[j], y + shift[j])
                })
                .unzip(),
            SearchOp::Relu { input, .. } => {
                boxes.push(ReluBox {
                    op: k,
                    l: l[*input].clone(),
                    u: u[*input].clone(),
                });
                (
                    l[*input].iter().map(|v| v.max(0.0)).collect(),
                    u[*input].iter().map(|v| v.max(0.0)).collect(),
                )
            }
            SearchOp::Add { lhs, rhs } => (
                l[*lhs].iter().zip(&l[*rhs]).map(|(a, b)| a + b).collect(),
                u[*lhs].iter().zip(&u[*rhs]).map(|(a, b)| a + b).collect(),
            ),
            SearchOp::Flatten { input } => (l[*input].clone(), u[*input].clone()),
            SearchOp::Gemm {
                input,
                n_out,
                n_in,
                weight,
                bias,
            } => (0..*n_out)
                .map(|o| {
                    let (mut a, mut b) = (bias[o], bias[o]);
                    for i in 0..*n_in {
                        let (x, y) = span(weight[o * n_in + i], l[*input][i], u[*input][i]);
                        a += x;
                        b += y;
                    }
                    (a, b)
                })
                .unzip(),
        };
        l.push(nl);
        u.push(nu);
    }
    boxes
}

/// Dense f64 CROWN, written independently of the engine: seed `seed` on
/// tensor `start`, walk ops `k < limit` backwards with the given boxes and
/// FULL-WIDTH lower slopes per ReLU site, concretise over `[lo, hi]`.
fn reference_crown(
    net: &SearchNet,
    lo: &[f64],
    hi: &[f64],
    boxes: &[(Vec<f64>, Vec<f64>)],
    slopes: &[Vec<f64>],
    start: usize,
    limit: usize,
    seed: &[(usize, f64)],
) -> f64 {
    let sizes = net.tensor_sizes().expect("valid net");
    let mut coef: Vec<Vec<f64>> = sizes.iter().map(|&n| vec![0.0; n]).collect();
    for &(i, c) in seed {
        coef[start][i] += c;
    }
    let mut site_of = vec![usize::MAX; net.ops.len()];
    let mut next_site = 0usize;
    for (k, op) in net.ops.iter().enumerate() {
        if matches!(op, SearchOp::Relu { .. }) {
            site_of[k] = next_site;
            next_site += 1;
        }
    }
    let mut bias = 0.0;
    for k in (0..limit).rev() {
        let v = std::mem::take(&mut coef[k + 1]);
        match &net.ops[k] {
            SearchOp::Conv(c) => {
                for o in 0..c.cout {
                    for sp in 0..c.p {
                        let vv = v[o * c.p + sp];
                        bias += c.bias[o] * vv;
                        for t in 0..c.k {
                            let g = c.gather[t * c.p + sp];
                            if g != GATHER_PAD {
                                coef[c.input][g as usize] += c.weight[o * c.k + t] * vv;
                            }
                        }
                    }
                }
            }
            SearchOp::Affine {
                input,
                scale,
                shift,
            } => {
                for j in 0..v.len() {
                    coef[*input][j] += scale[j] * v[j];
                    bias += shift[j] * v[j];
                }
            }
            SearchOp::Relu { input, .. } => {
                let s = site_of[k];
                let (l, u) = (&boxes[s].0, &boxes[s].1);
                for j in 0..v.len() {
                    if l[j] >= 0.0 {
                        coef[*input][j] += v[j];
                    } else if u[j] <= 0.0 {
                    } else if v[j] >= 0.0 {
                        coef[*input][j] += slopes[s][j] * v[j];
                    } else {
                        let chord = u[j] / (u[j] - l[j]);
                        coef[*input][j] += chord * v[j];
                        bias += -(chord * l[j]) * v[j];
                    }
                }
            }
            SearchOp::Add { lhs, rhs } => {
                for j in 0..v.len() {
                    coef[*lhs][j] += v[j];
                    coef[*rhs][j] += v[j];
                }
            }
            SearchOp::Flatten { input } => {
                for j in 0..v.len() {
                    coef[*input][j] += v[j];
                }
            }
            SearchOp::Gemm {
                input,
                n_out,
                n_in,
                weight,
                bias: b,
            } => {
                for o in 0..*n_out {
                    bias += b[o] * v[o];
                    for i in 0..*n_in {
                        coef[*input][i] += weight[o * n_in + i] * v[o];
                    }
                }
            }
        }
    }
    let mut lb = bias;
    for i in 0..lo.len() {
        let c = coef[0][i];
        lb += c * f64::midpoint(lo[i], hi[i]) - c.abs() * (0.5 * (hi[i] - lo[i]));
    }
    lb
}

/// A request for the property "the class winning at the centre stays the
/// winner" over `centre +- half`, with interval reference boxes.
fn request(net: &SearchNet, seed: u64, half: f64) -> RootSearchRequest {
    let mut rng = Lcg(seed ^ 0x5eed);
    let centre: Vec<f64> = (0..net.n_in).map(|_| rng.uniform(-0.5, 0.5)).collect();
    let lo: Vec<f64> = centre.iter().map(|c| c - half).collect();
    let hi: Vec<f64> = centre.iter().map(|c| c + half).collect();
    let vals = net.forward_f64(&centre).expect("forward");
    let y = vals.last().expect("output");
    let target = (0..y.len())
        .max_by(|&a, &b| y[a].total_cmp(&y[b]))
        .expect("classes");
    RootSearchRequest {
        ref_boxes: ibp_boxes(net, &lo, &hi),
        lo,
        hi,
        target,
        adv: (0..y.len()).filter(|&j| j != target).collect(),
        iters: 20,
        lr: 0.25,
        deadline: None,
        threads: 2,
        mem_cap_bytes: 1 << 30,
    }
}

fn config(req: &RootSearchRequest, threads: usize, force_shared: bool) -> Config {
    Config {
        iters: req.iters,
        lr: req.lr,
        deadline: req.deadline,
        threads,
        mem_cap: req.mem_cap_bytes,
        force_shared,
        chunk_blocks: None,
    }
}

fn run<R: Real>(net: &SearchNet, req: &RootSearchRequest, threads: usize) -> SearchResult<R> {
    let prog = Program::<R>::lower(net).expect("lower");
    search(&prog, req, &config(req, threads, false)).expect("search")
}

fn nets() -> Vec<(SearchNet, u64, f64)> {
    let mut out = Vec::new();
    for seed in 0..4u64 {
        out.push((
            conv_net(0xC0_0000 + seed, 0.45 + 0.1 * seed as f64),
            seed,
            0.3,
        ));
        out.push((
            mlp_net(0x3E_0000 + seed, 0.8 + 0.1 * seed as f64),
            seed,
            0.4,
        ));
    }
    out
}

/// The objective's full-width slopes per site from a result's universe layout.
fn full_slopes<R: Real>(res: &SearchResult<R>, boxes: &[(Vec<f64>, Vec<f64>)]) -> Vec<Vec<f64>> {
    let mut off = 0usize;
    let mut slopes = Vec::with_capacity(boxes.len());
    for (s, (l, u)) in boxes.iter().enumerate() {
        let mut v: Vec<f64> = (0..l.len())
            .map(|j| if u[j] >= -l[j] { 1.0 } else { 0.0 })
            .collect();
        for (p, &j) in res.universes[s].iter().enumerate() {
            v[j as usize] = res.root_alpha[off + p].to_f64();
        }
        off += res.universes[s].len();
        slopes.push(v);
    }
    slopes
}

fn boxes_f64<R: Real>(boxes: &[(Vec<R>, Vec<R>)]) -> Vec<(Vec<f64>, Vec<f64>)> {
    boxes
        .iter()
        .map(|(l, u)| {
            (
                l.iter().map(|v| v.to_f64()).collect(),
                u.iter().map(|v| v.to_f64()).collect(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
//  Tracking: f32 search vs independent f64 references
// ---------------------------------------------------------------------------

#[test]
fn f32_search_root_tracks_an_independent_f64_reference() {
    let mut compared = 0usize;
    let mut optimised = 0usize;
    for (net, seed, half) in nets() {
        let req = request(&net, seed, half);
        let res32 = run::<f32>(&net, &req, 2);
        let res64 = run::<f64>(&net, &req, 2);
        let boxes = boxes_f64(&res32.root_boxes);
        let slopes = full_slopes(&res32, &boxes);
        for (k, &j) in req.adv.iter().enumerate() {
            if !res32.surviving[k] || !res32.spec_lb[k].is_finite() {
                continue;
            }
            let reference = reference_crown(
                &net,
                &req.lo,
                &req.hi,
                &boxes,
                &slopes,
                net.ops.len(),
                net.ops.len(),
                &[(req.target, 1.0), (j, -1.0)],
            );
            let got = res32.spec_lb[k];
            assert!(
                (got - reference).abs() <= 1e-4 * (1.0 + reference.abs()),
                "seed {seed} class {j}: f32 search reported {got}, the f64 reference CROWN of the \
                 same slopes and boxes is {reference}"
            );
            compared += 1;
        }
        optimised += usize::from(res32.iters > 0);
        // The f32 instantiation tracks the f64 instantiation of the search.
        assert!(
            (res32.root_lb - res64.root_lb).abs() <= 1e-2 * (1.0 + res64.root_lb.abs()),
            "seed {seed}: f32 root {} vs f64 root {}",
            res32.root_lb,
            res64.root_lb
        );
        assert!(
            res32.root_lb >= res32.crown_lb,
            "keep-best never loses the init"
        );
    }
    assert!(compared >= 8, "only {compared} classes compared: VOID");
    assert!(optimised >= 4, "only {optimised} runs optimised: VOID");
}

#[test]
fn f64_search_bounds_and_boxes_enclose_sampled_truth() {
    let mut tightened = 0usize;
    for (net, seed, half) in nets() {
        let req = request(&net, seed, half);
        let res = run::<f64>(&net, &req, 3);
        tightened += res.tightened;
        let mut rng = Lcg(seed ^ 0xABCD);
        let relu_inputs: Vec<usize> = net
            .ops
            .iter()
            .filter_map(|op| match op {
                SearchOp::Relu { input, .. } => Some(*input),
                _ => None,
            })
            .collect();
        for sample in 0..300 {
            let x: Vec<f64> = (0..net.n_in)
                .map(|i| match sample {
                    0 => req.lo[i],
                    1 => req.hi[i],
                    s if s % 2 == 0 => {
                        if rng.unit() < 0.5 {
                            req.lo[i]
                        } else {
                            req.hi[i]
                        }
                    }
                    _ => rng.uniform(req.lo[i], req.hi[i]),
                })
                .collect();
            let vals = net.forward_f64(&x).expect("forward");
            let y = vals.last().expect("output");
            for (k, &j) in req.adv.iter().enumerate() {
                let margin = y[req.target] - y[j];
                assert!(
                    res.spec_lb[k] <= margin + 1e-9,
                    "seed {seed} class {j}: search bound {} above the sampled margin {margin}",
                    res.spec_lb[k]
                );
            }
            for (s, &t) in relu_inputs.iter().enumerate() {
                let (l, u) = &res.root_boxes[s];
                for (jj, &z) in vals[t].iter().enumerate() {
                    assert!(
                        l[jj] - 1e-9 <= z && z <= u[jj] + 1e-9,
                        "seed {seed} site {s} neuron {jj}: {z} outside [{}, {}]",
                        l[jj],
                        u[jj]
                    );
                }
            }
        }
    }
    assert!(
        tightened > 0,
        "the search never tightened a reference box: VOID"
    );
}

#[test]
fn crown_init_equals_the_reference_crown_with_adaptive_slopes() {
    for (net, seed, half) in nets() {
        let mut req = request(&net, seed, half);
        req.iters = 0;
        let res = run::<f64>(&net, &req, 1);
        assert_eq!(res.iters, 0);
        let boxes = boxes_f64(&res.root_boxes);
        let slopes: Vec<Vec<f64>> = boxes
            .iter()
            .map(|(l, u)| {
                (0..l.len())
                    .map(|j| if u[j] >= -l[j] { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        for (k, &j) in req.adv.iter().enumerate() {
            let reference = reference_crown(
                &net,
                &req.lo,
                &req.hi,
                &boxes,
                &slopes,
                net.ops.len(),
                net.ops.len(),
                &[(req.target, 1.0), (j, -1.0)],
            );
            assert!(
                (res.spec_lb[k] - reference).abs() <= 1e-9 * (1.0 + reference.abs()),
                "seed {seed} class {j}: init {} vs reference {reference}",
                res.spec_lb[k]
            );
        }
        assert!((res.crown_lb - res.root_lb).abs() <= 0.0);
    }
}

// ---------------------------------------------------------------------------
//  The gradient
// ---------------------------------------------------------------------------

#[test]
fn objective_and_intermediate_gradients_match_central_differences() {
    let (mut checked, mut matched, mut nonzero) = (0usize, 0usize, 0usize);
    for (net, seed, half) in nets() {
        let req = request(&net, seed, half);
        let res = run::<f64>(&net, &req, 1);
        let prog = Program::<f64>::lower(&net).expect("lower");
        let mut rng = Lcg(seed ^ 0x6A6A);
        // (start node, rows, stride) for the objective and the last trunk site.
        let mut cases: Vec<(Option<usize>, Vec<(u32, u32, bool)>, usize)> = Vec::new();
        let total: usize = res.universes.iter().map(Vec::len).sum();
        cases.push((
            None,
            vec![(req.target as u32, req.adv[0] as u32, false)],
            total,
        ));
        let last_trunk = prog
            .sites
            .iter()
            .rposition(|s| !s.head)
            .expect("trunk site");
        let before: usize = res.universes[..last_trunk].iter().map(Vec::len).sum();
        let (l, u) = &res.root_boxes[last_trunk];
        let rows: Vec<(u32, u32, bool)> = (0..l.len())
            .filter(|&j| l[j] < 0.0 && u[j] > 0.0)
            .take(3)
            .flat_map(|j| [(j as u32, NONE, false), (j as u32, NONE, true)])
            .collect();
        if !rows.is_empty() {
            cases.push((Some(last_trunk), rows, before));
        }
        for (start, rows, stride) in cases {
            if stride == 0 {
                continue;
            }
            let alpha: Vec<f64> = (0..stride).map(|_| rng.uniform(0.15, 0.85)).collect();
            let base = eval_rows_for_test(
                &prog,
                &req.lo,
                &req.hi,
                &res.root_boxes,
                &res.universes,
                start,
                &rows,
                &alpha,
            );
            let h = 1e-6;
            for i in (0..stride).step_by(stride.div_ceil(12).max(1)) {
                let mut plus = alpha.clone();
                plus[i] += h;
                let mut minus = alpha.clone();
                minus[i] -= h;
                let up = eval_rows_for_test(
                    &prog,
                    &req.lo,
                    &req.hi,
                    &res.root_boxes,
                    &res.universes,
                    start,
                    &rows,
                    &plus,
                );
                let down = eval_rows_for_test(
                    &prog,
                    &req.lo,
                    &req.hi,
                    &res.root_boxes,
                    &res.universes,
                    start,
                    &rows,
                    &minus,
                );
                for r in 0..rows.len() {
                    let fd = (up[r].0 - down[r].0) / (2.0 * h);
                    let g = base[r].1[i];
                    checked += 1;
                    nonzero += usize::from(g.abs() > 1e-9);
                    if (fd - g).abs() <= 1e-6 * (1.0 + g.abs()) {
                        matched += 1;
                    }
                }
            }
        }
    }
    assert!(
        checked >= 100,
        "only {checked} gradient coordinates checked: VOID"
    );
    assert!(
        nonzero >= 40,
        "only {nonzero} of {checked} checked gradient coordinates were nonzero: VOID"
    );
    // Piecewise-linear objective: a central difference straddling a line
    // switch is legitimately off. Allow a sliver, nothing more.
    assert!(
        matched * 100 >= checked * 97,
        "gradient matched central differences on {matched}/{checked} coordinates"
    );
}

// ---------------------------------------------------------------------------
//  Determinism, the hint, slope sharing, pruning, refusals, lowering
// ---------------------------------------------------------------------------

#[test]
fn search_result_is_independent_of_the_thread_count() {
    for (net, seed, half) in nets().into_iter().take(4) {
        let req = request(&net, seed, half);
        let one = run::<f32>(&net, &req, 1);
        for threads in [2usize, 5] {
            let many = run::<f32>(&net, &req, threads);
            assert_eq!(one.root_lb.to_bits(), many.root_lb.to_bits(), "seed {seed}");
            assert_eq!(one.iters, many.iters, "seed {seed}");
            for (a, b) in one.spec_lb.iter().zip(&many.spec_lb) {
                assert_eq!(a.to_bits(), b.to_bits(), "seed {seed}");
            }
            for ((op_a, a), (op_b, b)) in one.hint_trunk.iter().zip(&many.hint_trunk) {
                assert_eq!(op_a, op_b);
                assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
            }
            assert!(one
                .hint_head
                .iter()
                .zip(&many.hint_head)
                .all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }
}

/// Every bit the search reports (bounds, stop, slopes, boxes, hint).
fn result_bits<R: Real>(res: &SearchResult<R>) -> Vec<u64> {
    let mut bits = vec![
        res.root_lb.to_bits(),
        res.crown_lb.to_bits(),
        res.iters as u64,
        res.passes as u64,
        res.stop.len() as u64,
        u64::from(res.share_rows),
    ];
    bits.extend(res.spec_lb.iter().map(|v| v.to_bits()));
    bits.extend(res.root_alpha.iter().map(|v| v.to_f64().to_bits()));
    for (l, u) in &res.root_boxes {
        bits.extend(l.iter().chain(u).map(|v| v.to_f64().to_bits()));
    }
    for (op, slopes) in &res.hint_trunk {
        bits.push(*op as u64);
        bits.extend(slopes.iter().map(|v| u64::from(v.to_bits())));
    }
    bits.extend(res.hint_head.iter().map(|v| u64::from(v.to_bits())));
    bits
}

/// THE MEMORY CONTRACT of the gradient passes. A pass streams its blocks in
/// chunks and absorbs each chunk's sparse gradients in block order, so (a)
/// the result is bit-identical whatever the chunk size and the thread count,
/// and (b) the gradient bytes held at once are one chunk's: with one block per
/// chunk at most one worst-case block (an entry per owned slope), and by
/// default within the transient share of the admission (or the workers' own
/// blocks, which their workspace estimate already covers). Before this, every
/// pass held `rows * stride` dense values whatever the admission said, which
/// on a tinyimagenet-sized start node is tens of GB.
#[test]
fn gradient_passes_stream_in_bounded_chunks_bit_identically() {
    let mut streamed = 0usize;
    for (net, seed, half) in nets().into_iter().take(6) {
        let req = request(&net, seed, half);
        let prog = Program::<f32>::lower(&net).expect("lower");
        let entry = size_of::<(u32, [f32; crate::joint_root::LANES])>() as u64;
        for shared in [false, true] {
            let whole = search(&prog, &req, &config(&req, 2, shared)).expect("default chunks");
            let slopes: u64 = whole.universes.iter().map(|u| u.len() as u64).sum();
            let one_block = slopes * entry;
            assert!(
                whole.peak_grad_bytes <= whole.transient_cap.max(2 * one_block),
                "seed {seed} shared={shared}: held {} B of gradients, admitted {} B",
                whole.peak_grad_bytes,
                whole.transient_cap
            );
            for (threads, chunk) in [(1usize, 1usize), (3, 1), (2, 2), (5, 3)] {
                let mut cfg = config(&req, threads, shared);
                cfg.chunk_blocks = Some(chunk);
                let small = search(&prog, &req, &cfg).expect("streamed");
                assert_eq!(
                    result_bits(&small),
                    result_bits(&whole),
                    "seed {seed} shared={shared} threads={threads} chunk={chunk}: streaming \
                     changed the search"
                );
                assert!(
                    small.peak_grad_bytes <= chunk as u64 * one_block,
                    "seed {seed}: {chunk}-block chunks held {} B > {} B",
                    small.peak_grad_bytes,
                    chunk as u64 * one_block
                );
                if chunk == 1 && small.peak_grad_bytes < whole.peak_grad_bytes {
                    streamed += 1;
                }
            }
        }
    }
    assert!(
        streamed > 0,
        "no pass ever had more than one block, so streaming was never exercised: VOID"
    );
}

#[test]
fn hint_is_well_formed_with_exact_limits_on_stable_neurons() {
    let backend = CpuF32Backend::with_threads(2);
    assert_eq!(backend.name(), "cpu_f32");
    for (net, seed, half) in nets() {
        let req = request(&net, seed, half);
        let outcome = backend.root_joint_alpha(&net, &req).expect("search");
        let stats = &outcome.stats;
        assert_eq!(stats.specs, req.adv.len());
        assert!(stats.iters <= req.iters);
        assert!(
            [
                "crown_verified",
                "verified",
                "patience",
                "iters",
                "deadline"
            ]
            .contains(&stats.stop),
            "unknown stop {}",
            stats.stop
        );
        let hint = outcome.alpha.into_hint().into_ingest();
        let relus: Vec<(usize, usize, bool)> = net
            .ops
            .iter()
            .enumerate()
            .filter_map(|(k, op)| match op {
                SearchOp::Relu { input, head } => Some((k, *input, *head)),
                _ => None,
            })
            .collect();
        let sizes = net.tensor_sizes().expect("sizes");
        let trunk: Vec<&(usize, usize, bool)> = relus.iter().filter(|r| !r.2).collect();
        assert_eq!(hint.trunk.len(), trunk.len());
        for ((op, slopes), &&(k, input, _)) in hint.trunk.iter().zip(&trunk) {
            assert_eq!(*op, k);
            assert_eq!(slopes.len(), sizes[input]);
            assert!(slopes
                .iter()
                .all(|a| a.is_finite() && (0.0..=1.0).contains(a)));
        }
        let head = relus.iter().find(|r| r.2).expect("head relu");
        assert_eq!(hint.head.len(), sizes[head.1]);
        assert!(hint
            .head
            .iter()
            .all(|a| a.is_finite() && (0.0..=1.0).contains(a)));
        // Stable-in-the-reference neurons carry their exact limit.
        for ((op, slopes), rb) in hint
            .trunk
            .iter()
            .zip(req.ref_boxes.iter().filter(|b| b.op != head.0))
        {
            assert_eq!(*op, rb.op);
            for j in 0..slopes.len() {
                if rb.l[j] >= 0.0 {
                    assert_eq!(slopes[j], 1.0, "op {op} neuron {j} is active");
                } else if rb.u[j] <= 0.0 {
                    assert_eq!(slopes[j], 0.0, "op {op} neuron {j} is inactive");
                }
            }
        }
    }
}

#[test]
fn per_row_and_shared_slopes_both_search() {
    for (net, seed, half) in nets().into_iter().take(4) {
        let req = request(&net, seed, half);
        let prog = Program::<f32>::lower(&net).expect("lower");
        let per_row = search(&prog, &req, &config(&req, 2, false)).expect("per-row");
        let shared = search(&prog, &req, &config(&req, 2, true)).expect("shared");
        assert!(
            !per_row.share_rows,
            "seed {seed}: 1 GiB fits per-row slopes"
        );
        assert!(
            shared.share_rows || shared.rows == 0,
            "seed {seed}: forced sharing"
        );
        for res in [&per_row, &shared] {
            assert!(res.root_lb.is_finite() && res.root_lb >= res.crown_lb);
        }
        // Same init, so the same CROWN bound.
        assert_eq!(per_row.crown_lb.to_bits(), shared.crown_lb.to_bits());
    }
}

#[test]
fn spec_prune_and_crown_verified_exit() {
    let mut net = mlp_net(0x3E_0042, 0.6);
    let n_ops = net.ops.len();
    let req0 = request(&net, 7, 0.05);
    // A decisive target: every class is proved by the CROWN init.
    if let SearchOp::Gemm { bias, .. } = &mut net.ops[n_ops - 1] {
        bias[req0.target] += 100.0;
    }
    let res = run::<f32>(&net, &req0, 2);
    assert_eq!(res.stop, "crown_verified");
    assert_eq!(res.iters, 0);
    assert_eq!(res.pruned, res.specs);
    assert!(res.root_lb > 0.0);
    // Now make exactly ONE class easy: it is pruned, the others optimised.
    let mut net2 = mlp_net(0x3E_0042, 0.6);
    let req = request(&net2, 7, 0.4);
    let easy = req.adv[0];
    if let SearchOp::Gemm { bias, .. } = &mut net2.ops[n_ops - 1] {
        bias[easy] -= 100.0;
    }
    let res = run::<f32>(&net2, &req, 2);
    assert!(!res.surviving[0], "the easy class must be pruned");
    assert!(res.pruned >= 1);
    assert!(res.spec_lb[0] > 0.0);
}

#[test]
fn invalid_requests_and_budgets_refuse() {
    let net = conv_net(0xC0_0001, 0.5);
    let good = request(&net, 1, 0.3);
    let prog = Program::<f32>::lower(&net).expect("lower");
    let refuse = |req: &RootSearchRequest| search(&prog, req, &config(req, 1, false)).err();

    let mut r = good.clone();
    r.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("the monotonic clock is past its first millisecond"),
    );
    assert_eq!(refuse(&r), Some(Refusal::Deadline));

    let mut r = good.clone();
    r.adv.clear();
    assert!(matches!(refuse(&r), Some(Refusal::Nothing(_))));

    let mut r = good.clone();
    r.target = 99;
    assert!(matches!(refuse(&r), Some(Refusal::Invalid(_))));

    let mut r = good.clone();
    r.adv.push(r.target);
    assert!(matches!(refuse(&r), Some(Refusal::Invalid(_))));

    let mut r = good.clone();
    r.lo[0] = r.hi[0] + 1.0;
    assert!(matches!(refuse(&r), Some(Refusal::Invalid(_))));

    let mut r = good.clone();
    r.ref_boxes[0].l.pop();
    assert!(matches!(refuse(&r), Some(Refusal::Invalid(_))));

    let mut r = good.clone();
    r.ref_boxes[0].op = 0;
    assert!(matches!(refuse(&r), Some(Refusal::Invalid(_))));

    let mut r = good.clone();
    r.mem_cap_bytes = 16;
    assert!(matches!(refuse(&r), Some(Refusal::Memory { .. })));

    // A malformed net refuses at lowering, through the public backend.
    let mut bad = net.clone();
    if let SearchOp::Conv(c) = &mut bad.ops[0] {
        c.gather[0] = 10_000;
    }
    assert!(matches!(
        CpuF32Backend::new().root_joint_alpha(&bad, &good),
        Err(Refusal::Invalid(_))
    ));

    // A deadline that lands mid-search never panics: it refuses (no complete
    // init) or returns its best so far with `stop = deadline`.
    let mut r = good;
    r.iters = 10_000;
    r.deadline = Some(Instant::now() + Duration::from_millis(2));
    match run_root::<f32>(&net, &r, 1, false) {
        Ok(outcome) => assert!([
            "deadline",
            "patience",
            "verified",
            "iters",
            "crown_verified"
        ]
        .contains(&outcome.stats.stop)),
        Err(refusal) => assert_eq!(refusal, Refusal::Deadline),
    }
}

#[test]
fn lowering_transposes_the_gather_and_records_the_sites() {
    let net = conv_net(0xC0_0002, 0.5);
    let prog = Program::<f32>::lower(&net).expect("lower");
    assert_eq!(prog.sites.len(), 3);
    assert_eq!(
        prog.sites
            .iter()
            .map(|s| (s.op, s.input, s.n, s.head))
            .collect::<Vec<_>>(),
        vec![(1, 1, 36, false), (4, 4, 36, false), (7, 7, 5, true)]
    );
    let SearchOp::Conv(c) = &net.ops[0] else {
        unreachable!("conv first")
    };
    let Op::Conv {
        gather_pk, k, p, ..
    } = &prog.ops[0]
    else {
        unreachable!("conv first")
    };
    for t in 0..*k {
        for sp in 0..*p {
            let g = c.gather[t * c.p + sp];
            let want = if g == GATHER_PAD { NONE } else { g };
            assert_eq!(gather_pk[sp * k + t], want);
        }
    }
    assert_eq!(prog.spatial[1], 9, "conv output spatial size");
    assert_eq!(prog.tsize, net.tensor_sizes().expect("sizes"));
}
