// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M6 wave search tests. The oracles are INDEPENDENT of the kernel: hand
//! derived known answers on one-neuron nets, exhaustive vertex enumeration of
//! linear nets, interval propagation for truthful boxes, sampled f64 forward
//! evaluation for truth and central differences for the gradients.
//!
//! Known-answer tests (KATs) of the kernels:
//! * `wave_kat_lower_slope_bound_and_gradient` — `relu(x)` on `[-1, 1]`: the
//!   lower line `alpha x` gives `-alpha`, `d/d alpha = -1`;
//! * `wave_kat_split_multiplier_bound_and_gradient` — the same neuron pinned
//!   active / inactive with a multiplier: `-(1 - beta)` / `-beta` and
//!   `d/d beta = +1` / `-1` (the SIGN of the Lagrangian term on both sides);
//! * `wave_kat_upper_chord_intercept_and_branch_score` — `-relu(x)` on
//!   `[-1, 3]`: the chord bound `-3` and the BaBSR score `3/4`;
//! * `wave_kat_conv_net_bound_is_the_exact_linear_minimum` — a conv net whose
//!   boxes make it linear: the bound is the exact minimum over all 512 input
//!   vertices.
//!
//! Search math and protocol:
//! * `wave_f64_bounds_enclose_sampled_truth_on_split_domains`;
//! * `wave_gradients_match_central_differences`;
//! * `wave_adam_keeps_the_best_iterate`;
//! * `wave_search_is_thread_count_and_batch_independent` (R6);
//! * `wave_refuses_malformed_foreign_oversized_and_late_requests`;
//! * `wave_warm_hints_are_sanitized_at_the_backend_ingest`;
//! * `wave_f32_tracks_the_f64_reference`;
//! * `wave_branch_scores_rank_only_unsplit_unstable_neurons`;
//! * `wave_early_stop_leaves_the_optimisation`;
//! * `wave_work_bytes_cover_the_workspace`.

use std::time::{Duration, Instant};

use ny_core::search::{
    BranchOut, DeviceLease, Refusal, ReluBox, SearchBackend, SearchConv, SearchNet, SearchOp,
    WaveBatch, WaveHandle, WaveOut, WaveProblem, WaveSearchCfg, WaveSite, WaveSplit, GATHER_PAD,
    WAVE_NONE,
};

use crate::wave::{self, MAX_BRANCH_K};
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

    fn below(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n.saturating_sub(1))
    }

    fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.uniform(-scale, scale)).collect()
    }
}

const LEASE: DeviceLease = DeviceLease { bytes: 1 << 30 };

/// `relu(z)` or `-relu(z)` of `z = x` on a one-input net, objective
/// `y_0 - y_1` with `y = [sign * relu(z), 0]`.
fn one_relu(sign: f64, lo: f64, hi: f64) -> WaveProblem {
    WaveProblem {
        net: SearchNet {
            n_in: 1,
            ops: vec![
                SearchOp::Gemm {
                    input: 0,
                    n_out: 1,
                    n_in: 1,
                    weight: vec![1.0],
                    bias: vec![0.0],
                },
                SearchOp::Relu {
                    input: 1,
                    head: false,
                },
                SearchOp::Gemm {
                    input: 2,
                    n_out: 2,
                    n_in: 1,
                    weight: vec![sign, 0.0],
                    bias: vec![0.0, 0.0],
                },
            ],
        },
        boxes: vec![ReluBox {
            op: 1,
            l: vec![lo],
            u: vec![hi],
        }],
        universe: vec![WaveSite { op: 1, neuron: 0 }],
        alpha0: vec![1.0],
        lo: vec![lo],
        hi: vec![hi],
        target: 0,
        classes: vec![1],
    }
}

fn gemm_ibp(w: &[f64], b: &[f64], n_in: usize, lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n_out = b.len();
    let (mut ol, mut oh) = (b.to_vec(), b.to_vec());
    for o in 0..n_out {
        for i in 0..n_in {
            let (p, q) = (w[o * n_in + i] * lo[i], w[o * n_in + i] * hi[i]);
            ol[o] += p.min(q);
            oh[o] += p.max(q);
        }
    }
    // Widen by a relative 1e-9: the boxes must contain the f64 forward values
    // the sampled-truth oracle computes, round-off included.
    for o in 0..n_out {
        let pad = 1e-9 * (1.0 + ol[o].abs().max(oh[o].abs()));
        ol[o] -= pad;
        oh[o] += pad;
    }
    (ol, oh)
}

/// A random two-ReLU MLP `x -> Gemm -> ReLU -> Gemm (y) -> ReLU (head) ->
/// Gemm`, its truthful interval boxes and its universe (every neuron unstable
/// in its box).
fn mlp(seed: u64, n_in: usize, h: usize, n_y: usize, n_out: usize) -> WaveProblem {
    let mut rng = Lcg(seed);
    let (w1, b1) = (rng.vec(h * n_in, 0.8), rng.vec(h, 0.3));
    let (w2, b2) = (rng.vec(n_y * h, 0.6), rng.vec(n_y, 0.3));
    let (w3, b3) = (rng.vec(n_out * n_y, 0.8), rng.vec(n_out, 0.3));
    let lo: Vec<f64> = (0..n_in).map(|_| rng.uniform(-0.6, 0.0)).collect();
    let hi: Vec<f64> = lo.iter().map(|&l| l + rng.uniform(0.3, 1.0)).collect();
    let (l1, u1) = gemm_ibp(&w1, &b1, n_in, &lo, &hi);
    let (r1l, r1u): (Vec<f64>, Vec<f64>) = l1
        .iter()
        .zip(&u1)
        .map(|(&l, &u)| (l.max(0.0), u.max(0.0)))
        .unzip();
    let (l2, u2) = gemm_ibp(&w2, &b2, h, &r1l, &r1u);
    let mut universe = Vec::new();
    for (j, (&l, &u)) in l1.iter().zip(&u1).enumerate() {
        if l < 0.0 && u > 0.0 {
            universe.push(WaveSite {
                op: 1,
                neuron: j as u32,
            });
        }
    }
    for (j, (&l, &u)) in l2.iter().zip(&u2).enumerate() {
        if l < 0.0 && u > 0.0 {
            universe.push(WaveSite {
                op: 3,
                neuron: j as u32,
            });
        }
    }
    let alpha0: Vec<f64> = (0..universe.len())
        .map(|p| if p % 3 == 0 { f64::NAN } else { 0.5 })
        .collect();
    WaveProblem {
        net: SearchNet {
            n_in,
            ops: vec![
                SearchOp::Gemm {
                    input: 0,
                    n_out: h,
                    n_in,
                    weight: w1,
                    bias: b1,
                },
                SearchOp::Relu {
                    input: 1,
                    head: false,
                },
                SearchOp::Gemm {
                    input: 2,
                    n_out: n_y,
                    n_in: h,
                    weight: w2,
                    bias: b2,
                },
                SearchOp::Relu {
                    input: 3,
                    head: true,
                },
                SearchOp::Gemm {
                    input: 4,
                    n_out,
                    n_in: n_y,
                    weight: w3,
                    bias: b3,
                },
            ],
        },
        boxes: vec![
            ReluBox {
                op: 1,
                l: l1,
                u: u1,
            },
            ReluBox {
                op: 3,
                l: l2,
                u: u2,
            },
        ],
        universe,
        alpha0,
        lo,
        hi,
        target: 0,
        classes: (1..n_out).collect(),
    }
}

/// The first [`mlp`] from `seed` on whose universe has at least `min`
/// positions (fixtures that need splits must not depend on one seed's luck).
fn mlp_with_universe(
    seed: u64,
    (n_in, h, n_y, n_out): (usize, usize, usize, usize),
    min: usize,
) -> WaveProblem {
    (seed..seed + 64)
        .map(|s| mlp(s, n_in, h, n_y, n_out))
        .find(|p| p.universe.len() >= min)
        .expect("no seed in 64 gives the universe the fixture needs")
}

/// One wave's owned inputs.
struct Wave {
    rows: Vec<u32>,
    offsets: Vec<u32>,
    splits: Vec<WaveSplit>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    stride: usize,
}

impl Wave {
    fn new(stride: usize) -> Self {
        Self {
            rows: Vec::new(),
            offsets: vec![0],
            splits: Vec::new(),
            beta: Vec::new(),
            alpha: Vec::new(),
            stride,
        }
    }

    fn push(&mut self, row: u32, lits: &[(u32, bool, f32)]) {
        self.rows.push(row);
        for &(pos, active, b) in lits {
            self.splits.push(WaveSplit { pos, active });
            self.beta.push(b);
        }
        self.offsets.push(self.splits.len() as u32);
    }

    fn batch(&self) -> WaveBatch<'_> {
        WaveBatch {
            count: self.rows.len(),
            rows: &self.rows,
            offsets: &self.offsets,
            splits: &self.splits,
            beta: &self.beta,
            alpha: &self.alpha,
            stride: self.stride,
            deadline: None,
        }
    }
}

/// A random wave of `count` domains over `p`'s universe (random rows, 0..=3
/// random literals with random warm multipliers).
fn random_wave(p: &WaveProblem, count: usize, seed: u64) -> Wave {
    let mut rng = Lcg(seed);
    let stride = p.universe.len();
    let mut w = Wave::new(stride);
    for _ in 0..count {
        let row = rng.below(p.classes.len()) as u32;
        let mut lits: Vec<(u32, bool, f32)> = Vec::new();
        let n = rng.below(4).min(stride);
        while lits.len() < n {
            let pos = rng.below(stride) as u32;
            if lits.iter().all(|l| l.0 != pos) {
                lits.push((pos, rng.unit() < 0.5, rng.uniform(0.0, 0.4) as f32));
            }
        }
        w.push(row, &lits);
    }
    w
}

fn f64_search(p: &WaveProblem, w: &Wave, cfg: &WaveSearchCfg, threads: usize) -> WaveOut {
    let h = wave::upload::<f64>("ref", p, &LEASE, threads).expect("upload");
    let mut out = WaveOut::default();
    wave::backward_batch::<f64>(&h, &w.batch(), cfg, &mut out).expect("search");
    out
}

/// Forward every tensor at `x` (f64 reference).
fn forward(p: &WaveProblem, x: &[f64]) -> Vec<Vec<f64>> {
    p.net.forward_f64(x).expect("forward")
}

// ---------------------------------------------------------------------------
//  Known-answer tests of the kernels
// ---------------------------------------------------------------------------

#[test]
fn wave_kat_lower_slope_bound_and_gradient() {
    let p = one_relu(1.0, -1.0, 1.0);
    for (name, h) in [
        (
            "f64",
            wave::upload::<f64>("ref", &p, &LEASE, 1).expect("upload"),
        ),
        (
            "f32",
            wave::upload::<f32>("cpu", &p, &LEASE, 1).expect("upload"),
        ),
    ] {
        let probe = |a: f64| {
            if name == "f64" {
                wave::probe::<f64>(&h, 0, &[], &[a], &[]).expect("probe")
            } else {
                wave::probe::<f32>(&h, 0, &[], &[a], &[]).expect("probe")
            }
        };
        // min over [-1, 1] of alpha * x is -alpha, at x* = -1.
        for (a, lb) in [(1.0, -1.0), (0.25, -0.25), (0.0, 0.0)] {
            let (got, scale, ga, gb, lam) = probe(a);
            assert_eq!(got, lb, "{name}: bound at alpha = {a}");
            assert_eq!(gb, [] as [f64; 0]);
            assert_eq!(lam, vec![1.0], "{name}: the objective reads relu(z) once");
            if a > 0.0 {
                assert_eq!(ga, vec![-1.0], "{name}: d lb / d alpha = Lambda z(x*)");
                assert_eq!(scale, a, "{name}: S = |alpha| (|mid| + rad)");
            } else {
                // No coefficient reaches the input: x* is the midpoint.
                assert_eq!(ga, vec![0.0], "{name}: z(mid) = 0");
            }
        }
    }
}

#[test]
fn wave_kat_split_multiplier_bound_and_gradient() {
    let p = one_relu(1.0, -1.0, 1.0);
    let h = wave::upload::<f64>("ref", &p, &LEASE, 1).expect("upload");
    let on = [WaveSplit {
        pos: 0,
        active: true,
    }];
    let off = [WaveSplit {
        pos: 0,
        active: false,
    }];
    // ACTIVE pin: relu = z, Lagrangian z - beta z, min over [-1, 1] is
    // -(1 - beta) for beta in [0, 1]; d/d beta = -s z(x*) = +1 at x* = -1.
    for (b, lb, g) in [(0.0, -1.0, 1.0), (0.5, -0.5, 1.0)] {
        let (got, _, ga, gb, _) = wave::probe::<f64>(&h, 0, &on, &[1.0], &[b]).expect("probe");
        assert_eq!(got, lb, "active pin at beta = {b}");
        assert_eq!(gb, vec![g], "active pin gradient at beta = {b}");
        assert_eq!(ga, vec![0.0], "a pinned neuron's slope is not read");
    }
    // beta = 1 cancels the objective: the bound is the exact 0.
    let (got, ..) = wave::probe::<f64>(&h, 0, &on, &[1.0], &[1.0]).expect("probe");
    assert_eq!(got, 0.0);
    // INACTIVE pin: relu = 0, Lagrangian 0 + beta z, min is -beta at x* = -1;
    // d/d beta = -s z(x*) = z(x*) = -1.
    for (b, lb) in [(0.0, 0.0), (0.5, -0.5)] {
        let (got, _, _, gb, _) = wave::probe::<f64>(&h, 0, &off, &[1.0], &[b]).expect("probe");
        assert_eq!(got, lb, "inactive pin at beta = {b}");
        if b > 0.0 {
            assert_eq!(gb, vec![-1.0], "inactive pin gradient");
        }
    }
    // The MUTANT that flips the Lagrangian sign would report -(1 + beta) on
    // the active side: the KAT above can see it (0.5 -> -0.5, not -1.5).
}

#[test]
fn wave_kat_upper_chord_intercept_and_branch_score() {
    // -relu(x) on [-1, 3]: the chord s = 3/4, c = 3/4 gives -(0.75 x + 0.75),
    // minimal at x = 3: -3 (the exact minimum).
    let p = one_relu(-1.0, -1.0, 3.0);
    for h in [
        wave::upload::<f64>("ref", &p, &LEASE, 1).expect("upload"),
        wave::upload::<f32>("cpu", &p, &LEASE, 1).expect("upload"),
    ] {
        let (lb, scale, ..) = if h.backend() == "ref" {
            wave::probe::<f64>(&h, 0, &[], &[1.0], &[]).expect("probe")
        } else {
            wave::probe::<f32>(&h, 0, &[], &[1.0], &[]).expect("probe")
        };
        assert_eq!(lb, -3.0, "{}: chord bound", h.backend());
        // S = |c v| + |coef| (|mid| + rad) = 0.75 + 0.75 * 3.
        assert_eq!(scale, 3.0, "{}: absolute-term sum", h.backend());
    }
    let backend = CpuF32Backend::with_threads(1);
    let h = backend.upload_wave(&p, &LEASE).expect("upload");
    let mut w = Wave::new(1);
    w.push(0, &[]);
    let mut br = BranchOut::default();
    backend
        .branch_scores(&h, &w.batch(), 2, &mut br)
        .expect("scores")
        .into_hint()
        .into_ingest();
    let view = br.hint().into_ingest();
    assert!(view.well_formed());
    assert_eq!(view.pos(0), &[0, WAVE_NONE][..]);
    // |Lambda| * (-u l / (u - l)) = 1 * 3/4.
    assert_eq!(view.score(0)[0], 0.75);
}

#[test]
fn wave_kat_conv_net_bound_is_the_exact_linear_minimum() {
    // x in [0, 1]^9 (1 x 3 x 3) -> 2x2 conv, 2 channels, bias 5 -> ReLU whose
    // certified box is ACTIVE (every pre-activation >= 5 - 4 * 0.9) ->
    // Flatten -> Gemm 8 -> 3. The net is linear on the box, so the bound of
    // y_0 - y_2 is its exact minimum, attained at an input vertex.
    let mut rng = Lcg(77);
    let (k, p) = (4usize, 4usize);
    let mut gather = vec![GATHER_PAD; k * p];
    for oy in 0..2 {
        for ox in 0..2 {
            let sp = oy * 2 + ox;
            for ky in 0..2 {
                for kx in 0..2 {
                    gather[(ky * 2 + kx) * p + sp] = ((oy + ky) * 3 + (ox + kx)) as u32;
                }
            }
        }
    }
    let weight = rng.vec(2 * k, 0.9);
    let net = SearchNet {
        n_in: 9,
        ops: vec![
            SearchOp::Conv(Box::new(SearchConv {
                input: 0,
                cout: 2,
                k,
                p,
                gather,
                weight,
                bias: vec![5.0, 5.0],
            })),
            SearchOp::Relu {
                input: 1,
                head: false,
            },
            SearchOp::Flatten { input: 2 },
            SearchOp::Gemm {
                input: 3,
                n_out: 3,
                n_in: 8,
                weight: rng.vec(3 * 8, 1.0),
                bias: rng.vec(3, 0.5),
            },
        ],
    };
    let prob = WaveProblem {
        net,
        boxes: vec![ReluBox {
            op: 1,
            l: vec![1.0; 8],
            u: vec![9.0; 8],
        }],
        universe: Vec::new(),
        alpha0: Vec::new(),
        lo: vec![0.0; 9],
        hi: vec![1.0; 9],
        target: 0,
        classes: vec![2],
    };
    let mut exact = f64::INFINITY;
    for mask in 0u32..512 {
        let x: Vec<f64> = (0..9).map(|i| f64::from((mask >> i) & 1)).collect();
        let v = forward(&prob, &x);
        let y = v.last().expect("output");
        exact = exact.min(y[0] - y[2]);
    }
    let h = wave::upload::<f64>("ref", &prob, &LEASE, 1).expect("upload");
    let (lb, ..) = wave::probe::<f64>(&h, 0, &[], &[], &[]).expect("probe");
    assert!(
        (lb - exact).abs() <= 1e-12 * (1.0 + exact.abs()),
        "linear conv net: bound {lb} vs exact vertex minimum {exact}"
    );
}

// ---------------------------------------------------------------------------
//  Search math and protocol
// ---------------------------------------------------------------------------

#[test]
fn wave_f64_bounds_enclose_sampled_truth_on_split_domains() {
    let mut checked = 0usize;
    for seed in 0..12u64 {
        let p = mlp(seed, 4, 6, 5, 4);
        if p.universe.is_empty() {
            continue;
        }
        let w = random_wave(&p, 16, 1000 + seed);
        let cfg = WaveSearchCfg {
            iters: 6,
            stop_above: f64::INFINITY,
            ..WaveSearchCfg::default()
        };
        let out = f64_search(&p, &w, &cfg, 2);
        let view = out.hint().into_ingest();
        let mut rng = Lcg(seed * 31 + 7);
        for d in 0..w.rows.len() {
            let lb = f64::from(view.lb(d));
            assert!(lb.is_finite(), "seed {seed} domain {d}: no bound");
            let lits = &w.splits[w.offsets[d] as usize..w.offsets[d + 1] as usize];
            let class = p.classes[w.rows[d] as usize];
            for _ in 0..200 {
                let x: Vec<f64> =
                    p.lo.iter()
                        .zip(&p.hi)
                        .map(|(&l, &h)| rng.uniform(l, h))
                        .collect();
                let v = forward(&p, &x);
                let inside = lits.iter().all(|s| {
                    let site = p.universe[s.pos as usize];
                    // The ReLU at op k reads tensor k (its input).
                    let z = v[site.op as usize][site.neuron as usize];
                    if s.active {
                        z >= 0.0
                    } else {
                        z <= 0.0
                    }
                });
                if !inside {
                    continue;
                }
                let y = v.last().expect("output");
                let margin = y[p.target] - y[class];
                assert!(
                    margin >= lb - 1e-7 * (1.0 + lb.abs()),
                    "seed {seed} domain {d}: search bound {lb} above a sampled margin {margin}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 1000, "only {checked} sampled points: VOID");
}

#[test]
fn wave_gradients_match_central_differences() {
    let mut compared = 0usize;
    for seed in 0..8u64 {
        let p = mlp(seed + 40, 3, 5, 4, 3);
        let stride = p.universe.len();
        if stride == 0 {
            continue;
        }
        let h = wave::upload::<f64>("ref", &p, &LEASE, 1).expect("upload");
        let w = random_wave(&p, 4, seed + 90);
        for d in 0..w.rows.len() {
            let lits = &w.splits[w.offsets[d] as usize..w.offsets[d + 1] as usize];
            let alpha: Vec<f64> = (0..stride).map(|q| 0.2 + 0.6 * (q % 2) as f64).collect();
            let beta: Vec<f64> = (0..lits.len()).map(|q| 0.1 + 0.05 * q as f64).collect();
            let row = w.rows[d] as usize;
            let (_, _, ga, gb, _) =
                wave::probe::<f64>(&h, row, lits, &alpha, &beta).expect("probe");
            let eps = 1e-7;
            for q in 0..stride {
                let (mut ap, mut am) = (alpha.clone(), alpha.clone());
                ap[q] += eps;
                am[q] -= eps;
                let fp = wave::probe::<f64>(&h, row, lits, &ap, &beta).expect("p").0;
                let fm = wave::probe::<f64>(&h, row, lits, &am, &beta).expect("p").0;
                let fd = (fp - fm) / (2.0 * eps);
                assert!(
                    (fd - ga[q]).abs() <= 1e-5 * (1.0 + ga[q].abs()),
                    "seed {seed} domain {d} slope {q}: analytic {} vs central {fd}",
                    ga[q]
                );
                compared += 1;
            }
            for q in 0..lits.len() {
                let (mut bp, mut bm) = (beta.clone(), beta.clone());
                bp[q] += eps;
                bm[q] -= eps;
                let fp = wave::probe::<f64>(&h, row, lits, &alpha, &bp).expect("p").0;
                let fm = wave::probe::<f64>(&h, row, lits, &alpha, &bm).expect("p").0;
                let fd = (fp - fm) / (2.0 * eps);
                assert!(
                    (fd - gb[q]).abs() <= 1e-5 * (1.0 + gb[q].abs()),
                    "seed {seed} domain {d} multiplier {q}: analytic {} vs central {fd}",
                    gb[q]
                );
                compared += 1;
            }
        }
    }
    assert!(
        compared > 50,
        "only {compared} gradient entries compared: VOID"
    );
}

#[test]
fn wave_adam_keeps_the_best_iterate() {
    let mut improved = 0usize;
    for seed in 0..6u64 {
        let p = mlp(seed + 200, 4, 6, 5, 4);
        if p.universe.is_empty() {
            continue;
        }
        let w = random_wave(&p, 12, seed + 300);
        let backend = CpuF32Backend::with_threads(2);
        let h = backend.upload_wave(&p, &LEASE).expect("upload");
        let warm = WaveSearchCfg {
            iters: 0,
            stop_above: f64::INFINITY,
            ..WaveSearchCfg::default()
        };
        let asc = WaveSearchCfg { iters: 10, ..warm };
        let (mut o0, mut o1) = (WaveOut::default(), WaveOut::default());
        let _ = backend
            .backward_batch(&h, &w.batch(), &warm, &mut o0)
            .expect("warm");
        let _ = backend
            .backward_batch(&h, &w.batch(), &asc, &mut o1)
            .expect("ascent");
        let (v0, v1) = (o0.hint().into_ingest(), o1.hint().into_ingest());
        for d in 0..w.rows.len() {
            assert!(
                v1.lb(d) >= v0.lb(d),
                "seed {seed} domain {d}: ascent {} below its warm start {}",
                v1.lb(d),
                v0.lb(d)
            );
            improved += usize::from(v1.lb(d) > v0.lb(d));
            // The reported bound is the bound AT the reported point.
            let lits = &w.splits[w.offsets[d] as usize..w.offsets[d + 1] as usize];
            let alpha: Vec<f64> = v1.alpha(d).iter().map(|&a| f64::from(a)).collect();
            let beta: Vec<f64> = v1.beta()[w.offsets[d] as usize..w.offsets[d + 1] as usize]
                .iter()
                .map(|&b| f64::from(b))
                .collect();
            assert!(alpha.iter().all(|a| (0.0..=1.0).contains(a)));
            assert!(beta.iter().all(|&b| b >= 0.0));
            let (at, ..) =
                wave::probe::<f32>(&h, w.rows[d] as usize, lits, &alpha, &beta).expect("probe");
            assert_eq!(
                (at as f32).to_bits(),
                v1.lb(d).to_bits(),
                "seed {seed} domain {d}: reported bound is not the bound at the reported point"
            );
        }
    }
    assert!(improved > 0, "the ascent never improved a bound: VOID");
}

#[test]
fn wave_search_is_thread_count_and_batch_independent() {
    let p = mlp_with_universe(501, (4, 7, 5, 4), 3);
    let w = random_wave(&p, 24, 502);
    let cfg = WaveSearchCfg {
        iters: 5,
        ..WaveSearchCfg::default()
    };
    let bits = |o: &WaveOut, d: usize, w: &Wave| -> Vec<u32> {
        let v = o.hint().into_ingest();
        let mut out = vec![v.lb(d).to_bits(), v.scale(d).to_bits(), v.evals(d)];
        out.extend(v.alpha(d).iter().map(|a| a.to_bits()));
        out.extend(
            v.beta()[w.offsets[d] as usize..w.offsets[d + 1] as usize]
                .iter()
                .map(|b| b.to_bits()),
        );
        out
    };
    let mut reference: Option<Vec<Vec<u32>>> = None;
    for threads in [1usize, 3, 8] {
        let backend = CpuF32Backend::with_threads(threads);
        let h = backend.upload_wave(&p, &LEASE).expect("upload");
        let mut out = WaveOut::default();
        let _ = backend
            .backward_batch(&h, &w.batch(), &cfg, &mut out)
            .expect("search");
        let all: Vec<Vec<u32>> = (0..w.rows.len()).map(|d| bits(&out, d, &w)).collect();
        match &reference {
            None => reference = Some(all),
            Some(r) => assert_eq!(r, &all, "threads = {threads} changed a domain's output"),
        }
    }
    // Batch independence: every domain searched ALONE gives the same bits.
    let reference = reference.expect("reference");
    let backend = CpuF32Backend::with_threads(2);
    let h = backend.upload_wave(&p, &LEASE).expect("upload");
    for d in [0usize, 5, 23] {
        let lits = &w.splits[w.offsets[d] as usize..w.offsets[d + 1] as usize];
        let beta = &w.beta[w.offsets[d] as usize..w.offsets[d + 1] as usize];
        let mut solo = Wave::new(w.stride);
        let owned: Vec<(u32, bool, f32)> = lits
            .iter()
            .zip(beta)
            .map(|(s, &b)| (s.pos, s.active, b))
            .collect();
        solo.push(w.rows[d], &owned);
        let mut out = WaveOut::default();
        let _ = backend
            .backward_batch(&h, &solo.batch(), &cfg, &mut out)
            .expect("solo");
        assert_eq!(
            bits(&out, 0, &solo),
            reference[d],
            "domain {d} searched alone differs from its wave"
        );
    }
}

#[test]
fn wave_refuses_malformed_foreign_oversized_and_late_requests() {
    let p = one_relu(1.0, -1.0, 1.0);
    let backend = CpuF32Backend::with_threads(1);
    // A lease below the copy refuses with the exact need.
    match backend.upload_wave(&p, &DeviceLease { bytes: 16 }) {
        Err(Refusal::Memory { need, cap }) => assert!(need > cap && cap == 16),
        other => panic!("tiny lease must refuse on memory, got {other:?}"),
    }
    // An invalid problem refuses.
    let mut bad = p.clone();
    bad.classes = vec![0];
    assert!(matches!(
        backend.upload_wave(&bad, &LEASE),
        Err(Refusal::Invalid(_))
    ));
    // A net without a ReLU has nothing to relax.
    let linear = WaveProblem {
        net: SearchNet {
            n_in: 1,
            ops: vec![SearchOp::Gemm {
                input: 0,
                n_out: 2,
                n_in: 1,
                weight: vec![1.0, 2.0],
                bias: vec![0.0, 0.0],
            }],
        },
        boxes: Vec::new(),
        universe: Vec::new(),
        alpha0: Vec::new(),
        ..p.clone()
    };
    assert!(matches!(
        backend.upload_wave(&linear, &LEASE),
        Err(Refusal::Nothing(_))
    ));
    let h = backend.upload_wave(&p, &LEASE).expect("upload");
    let mut out = WaveOut::default();
    let cfg = WaveSearchCfg::default();
    // A foreign handle refuses.
    let foreign = WaveHandle::new("cpu_f32", 1, 7u64);
    let mut w = Wave::new(1);
    w.push(0, &[]);
    assert!(matches!(
        backend.backward_batch(&foreign, &w.batch(), &cfg, &mut out),
        Err(Refusal::Invalid(_))
    ));
    // The f64 reference upload is foreign to the f32 kernels too.
    let h64 = wave::upload::<f64>("cpu_f32", &p, &LEASE, 1).expect("upload");
    assert!(matches!(
        backend.backward_batch(&h64, &w.batch(), &cfg, &mut out),
        Err(Refusal::Invalid(_))
    ));
    // A malformed batch (stride, row, offsets) refuses.
    let mut wrong = Wave::new(2);
    wrong.push(0, &[]);
    assert!(backend
        .backward_batch(&h, &wrong.batch(), &cfg, &mut out)
        .is_err());
    let mut row = Wave::new(1);
    row.push(3, &[]);
    assert!(backend
        .backward_batch(&h, &row.batch(), &cfg, &mut out)
        .is_err());
    // A passed deadline refuses with Deadline.
    let late = WaveBatch {
        deadline: Some(
            Instant::now()
                .checked_sub(Duration::from_millis(5))
                .unwrap(),
        ),
        ..w.batch()
    };
    assert!(matches!(
        backend.backward_batch(&h, &late, &cfg, &mut out),
        Err(Refusal::Deadline)
    ));
    let mut br = BranchOut::default();
    assert!(matches!(
        backend.branch_scores(&h, &late, 1, &mut br),
        Err(Refusal::Deadline)
    ));
    // k outside 1..=MAX_BRANCH_K refuses.
    for k in [0usize, MAX_BRANCH_K + 1] {
        assert!(matches!(
            backend.branch_scores(&h, &w.batch(), k, &mut br),
            Err(Refusal::Invalid(_))
        ));
    }
    // An empty wave is a no-op, not a refusal.
    let empty = Wave::new(1);
    assert!(backend
        .backward_batch(&h, &empty.batch(), &cfg, &mut out)
        .is_ok());
}

#[test]
fn wave_warm_hints_are_sanitized_at_the_backend_ingest() {
    let p = mlp_with_universe(611, (4, 6, 5, 3), 2);
    let stride = p.universe.len();
    let cfg = WaveSearchCfg {
        iters: 3,
        ..WaveSearchCfg::default()
    };
    // Garbage warm state ...
    let mut dirty = Wave::new(stride);
    dirty.push(0, &[(0, true, f32::NAN), (1, false, -3.0)]);
    dirty.push(1, &[(1, true, f32::INFINITY)]);
    dirty.alpha = (0..2 * stride)
        .map(|q| match q % 4 {
            0 => f32::NAN,
            1 => 7.0,
            2 => -1.0,
            _ => 0.25,
        })
        .collect();
    // ... equals its sanitized form: multipliers 0, slopes clamped, NaN -> the
    // upload's own warm slope.
    let h = wave::upload::<f64>("ref", &p, &LEASE, 1).expect("upload");
    let mut clean = Wave::new(stride);
    clean.push(0, &[(0, true, 0.0), (1, false, 0.0)]);
    clean.push(1, &[(1, true, 0.0)]);
    clean.alpha = dirty
        .alpha
        .iter()
        .enumerate()
        .map(|(q, &a)| {
            if a.is_nan() {
                let pos = q % stride;
                let a0 = p.alpha0[pos];
                if a0.is_finite() {
                    a0.clamp(0.0, 1.0) as f32
                } else {
                    f32::NAN // the upload's adaptive slope, as in dirty
                }
            } else {
                a.clamp(0.0, 1.0)
            }
        })
        .collect();
    let mut o_dirty = WaveOut::default();
    let mut o_clean = WaveOut::default();
    wave::backward_batch::<f64>(&h, &dirty.batch(), &cfg, &mut o_dirty).expect("dirty");
    wave::backward_batch::<f64>(&h, &clean.batch(), &cfg, &mut o_clean).expect("clean");
    let (a, b) = (o_dirty.hint().into_ingest(), o_clean.hint().into_ingest());
    for d in 0..2 {
        assert_eq!(a.lb(d).to_bits(), b.lb(d).to_bits(), "domain {d}");
        assert!(a.alpha(d).iter().all(|v| (0.0..=1.0).contains(v)));
    }
    assert!(a.beta().iter().all(|&v| v.is_finite() && v >= 0.0));
}

#[test]
fn wave_f32_tracks_the_f64_reference() {
    for seed in 0..6u64 {
        let p = mlp(seed + 700, 5, 8, 6, 4);
        if p.universe.is_empty() {
            continue;
        }
        let w = random_wave(&p, 10, seed + 800);
        let cfg = WaveSearchCfg {
            iters: 0,
            stop_above: f64::INFINITY,
            ..WaveSearchCfg::default()
        };
        let o64 = f64_search(&p, &w, &cfg, 1);
        let h32 = wave::upload::<f32>("cpu", &p, &LEASE, 1).expect("upload");
        let mut o32 = WaveOut::default();
        wave::backward_batch::<f32>(&h32, &w.batch(), &cfg, &mut o32).expect("f32");
        let (a, b) = (o64.hint().into_ingest(), o32.hint().into_ingest());
        for d in 0..w.rows.len() {
            let (x, y, s) = (
                f64::from(a.lb(d)),
                f64::from(b.lb(d)),
                f64::from(a.scale(d)),
            );
            assert!(
                (x - y).abs() <= 1e-5 * (1.0 + s),
                "seed {seed} domain {d}: f32 {y} vs f64 {x} (S = {s})"
            );
        }
    }
}

#[test]
fn wave_branch_scores_rank_only_unsplit_unstable_neurons() {
    let p = mlp_with_universe(901, (4, 8, 6, 4), 4);
    let stride = p.universe.len();
    let backend = CpuF32Backend::with_threads(2);
    let h = backend.upload_wave(&p, &LEASE).expect("upload");
    let mut w = Wave::new(stride);
    w.push(0, &[]);
    w.push(1, &[(0, true, 0.1), (2, false, 0.0)]);
    let k = 4;
    let mut br = BranchOut::default();
    backend
        .branch_scores(&h, &w.batch(), k, &mut br)
        .expect("scores")
        .into_hint()
        .into_ingest();
    let view = br.hint().into_ingest();
    assert!(view.well_formed());
    for d in 0..2 {
        let pos = view.pos(d);
        let score = view.score(d);
        let real: Vec<u32> = pos.iter().copied().filter(|&q| q != WAVE_NONE).collect();
        assert!(!real.is_empty(), "domain {d}: no candidate");
        let mut sorted = real.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), real.len(), "domain {d}: repeated candidate");
        assert!(real.iter().all(|&q| (q as usize) < stride));
        for c in 1..real.len() {
            assert!(
                score[c - 1] > score[c] || (score[c - 1] == score[c] && pos[c - 1] < pos[c]),
                "domain {d}: candidates out of order"
            );
        }
        assert!(score[..real.len()].iter().all(|&s| s >= 0.0));
    }
    // The split domain never proposes its own literals.
    assert!(view.pos(1).iter().all(|&q| q != 0 && q != 2));
}

#[test]
fn wave_early_stop_leaves_the_optimisation() {
    let p = one_relu(1.0, -1.0, 1.0);
    let backend = CpuF32Backend::with_threads(1);
    let h = backend.upload_wave(&p, &LEASE).expect("upload");
    let mut w = Wave::new(1);
    // alpha 0 already gives the exact bound 0.
    w.alpha = vec![0.0];
    w.push(0, &[]);
    let cfg = WaveSearchCfg {
        iters: 8,
        stop_above: -0.5,
        ..WaveSearchCfg::default()
    };
    let mut out = WaveOut::default();
    let stats = backend
        .backward_batch(&h, &w.batch(), &cfg, &mut out)
        .expect("search")
        .into_hint()
        .into_ingest();
    let view = out.hint().into_ingest();
    assert_eq!(view.evals(0), 1, "a domain above stop_above leaves at once");
    assert_eq!(stats.early_stopped, 1);
    assert_eq!(view.lb(0), 0.0);
    // Without the early stop the same domain runs every evaluation.
    let full = WaveSearchCfg {
        stop_above: f64::INFINITY,
        ..cfg
    };
    let _ = backend
        .backward_batch(&h, &w.batch(), &full, &mut out)
        .expect("search");
    assert_eq!(out.hint().into_ingest().evals(0), 9);
}

#[test]
fn wave_work_bytes_cover_the_workspace() {
    for (seed, dims) in [(1u64, (4, 6, 5, 4)), (2, (9, 16, 7, 5))] {
        let p = mlp(seed, dims.0, dims.1, dims.2, dims.3);
        let h = wave::upload::<f32>("cpu", &p, &LEASE, 3).expect("upload");
        let up = h.payload::<wave::WaveUpload<f32>>().expect("own payload");
        let held = wave::workspace_capacity_bytes::<f32>(&h, 0).expect("capacity");
        assert!(
            up.work_bytes() >= held,
            "work_bytes {} below the {held} bytes a worker holds",
            up.work_bytes()
        );
        assert!(up.bytes() > 0);
    }
}
