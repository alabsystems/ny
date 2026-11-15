// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#wave-beta-bab` (ENGINE_PROGRAM M6), DEVICE side: the KNOWN-ANSWER tests of
//! ny-cuda's wave kernels against the CPU f32 wave search of ny-bbe, which is
//! their REFERENCE (ny-cli is the only crate that sees both backends; ny-cuda
//! may not depend on ny-bbe: `crate_dag_ratchet`), and `margin_row.tree_mode:
//! wave_cuda` through the registered device factory.
//!
//! The contract is BIT IDENTITY of every output the lane reads — per domain
//! the best search bound, its absolute-term sum, the bound evaluations, the
//! multipliers and the slopes at the best iterate; per domain the BaBSR
//! candidates and scores — plus the untrusted counters (passes, early stops).
//! Two canonicalisations only: a NaN is a NaN, and the sign of a zero is not
//! an output.
//!
//! * `cuda_wave_matches_cpu_f32_bit_for_bit_on_synthetic_waves` — dense and
//!   convolutional nets (padding, stride, residual add, affine, flatten, head
//!   ReLU), waves of 1..200 domains with 0..8 literals, empty and warm slopes,
//!   dirty hints (NaN / inf / negative / out-of-range), bound-only, early-stop
//!   and full ascents, default and invalid learning rates; branch candidates
//!   at k = 1, 3, 16; engagement counted (finite bounds, stepped domains, early
//!   stops, moved slopes) so a vacuous comparison fails;
//! * `cuda_wave_matches_cpu_f32_on_cifar_shaped_waves` — a 3x32x32 input;
//! * `cuda_wave_is_launch_and_host_thread_independent` — the CPU answer does not
//!   move with its thread count (it is the same answer), nor the device's with
//!   its slab size (a lease admitting 1, 2 or 5 domains per slab);
//! * `cuda_wave_kat_comparator_catches_a_one_ulp_mutation` — the MUTATION PIN
//!   of the comparator itself (device-free: it runs on every host);
//! * `registered_cuda_backend_reaches_the_armed_wave_tree` — `wave_cuda` with
//!   the factory registered as `main.rs` registers it: the receipt names
//!   `backend=cuda device_refused=none`, and the device searched;
//! * `real_cuda_wave_is_the_cpu_f32_wave_on_tiny_nets` — the device search is
//!   the CPU search, so the whole tree (verdict, expansions, domains, closes,
//!   stop) is the `wave` tree's;
//! * `real_cuda_wave_never_closes_a_sat_instance` — the SAT moat on the device.
//!
//! GATING. On a host that advertises a CUDA device every device test must
//! pass; on any other host each prints `SKIP=VOID` and PROVES NOTHING (it is
//! not a pass).

use std::sync::Arc;

use ny_core::search::{
    BranchOut, DeviceLease, ReluBox, SearchBackend, SearchConv, SearchNet, SearchOp, WaveBatch,
    WaveHandle, WaveOut, WaveProblem, WaveSearchCfg, WaveSite, WaveSplit, GATHER_PAD,
};
use ny_propagate::margin_row::bab::wave::{TreeMode, WavePolicy};
use ny_propagate::margin_row::{
    BabConfig, MarginRowBab, MarginRowOutcome, RootGates, RoundMode, TwinNet,
};

use super::wave_bab_tests::{
    register_real_backend, summary, targets, tiny_spec, tree_instances, wave_cfg,
};

const BIG: DeviceLease = DeviceLease { bytes: 1 << 34 };

/// A device the tests must use, or `None` after printing `SKIP=VOID`.
fn wave_device(test: &str) -> Option<ny_cuda::search::CudaSearchBackend> {
    let library = ny_cuda::CudaGemmEngine::runtime_driver_library_present();
    let devices = if library {
        ny_cuda::CudaGemmEngine::runtime_device_count()
    } else {
        None
    };
    if !(library && devices.is_some_and(|c| c > 0)) {
        eprintln!(
            "[cuda-wave-kat] {test}: SKIP=VOID (libcuda={library}, devices={devices:?}) — no \
             CUDA device on this host; this run proved NOTHING about the device wave search"
        );
        return None;
    }
    let backend = ny_cuda::search::CudaSearchBackend::new();
    // The device is advertised: a refusal to serve the wave search is a
    // failure, not a skip.
    if let Err(why) = backend.upload_wave(&mlp(1, 3, 6, 4, 3), &BIG) {
        panic!(
            "{test}: this host advertises a CUDA device ({devices:?}) but the device wave search \
             refused it: {why} (unavailable: {:?})",
            backend.wave_unavailable()
        );
    }
    Some(backend)
}

// ---------------------------------------------------------------------------
//  Synthetic problems (interval boxes: every universe neuron unstable)
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

    fn vec(&mut self, n: usize, s: f64) -> Vec<f64> {
        (0..n).map(|_| self.uniform(-s, s)).collect()
    }
}

/// Interval propagation of every tensor (f64, widened by a relative 1e-9).
fn ibp(net: &SearchNet, lo: &[f64], hi: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut t: Vec<(Vec<f64>, Vec<f64>)> = vec![(lo.to_vec(), hi.to_vec())];
    for op in &net.ops {
        let (ol, oh): (Vec<f64>, Vec<f64>) = match op {
            SearchOp::Conv(c) => {
                let (sl, sh) = &t[c.input];
                let mut ol = vec![0.0; c.cout * c.p];
                let mut oh = vec![0.0; c.cout * c.p];
                for o in 0..c.cout {
                    for sp in 0..c.p {
                        let (mut a, mut b) = (c.bias[o], c.bias[o]);
                        for tap in 0..c.k {
                            let g = c.gather[tap * c.p + sp];
                            if g == GATHER_PAD {
                                continue;
                            }
                            let w = c.weight[o * c.k + tap];
                            let (x, y) = (w * sl[g as usize], w * sh[g as usize]);
                            a += x.min(y);
                            b += x.max(y);
                        }
                        ol[o * c.p + sp] = a;
                        oh[o * c.p + sp] = b;
                    }
                }
                (ol, oh)
            }
            SearchOp::Affine {
                input,
                scale,
                shift,
            } => {
                let (sl, sh) = &t[*input];
                (0..sl.len())
                    .map(|j| {
                        let (x, y) = (scale[j] * sl[j], scale[j] * sh[j]);
                        (x.min(y) + shift[j], x.max(y) + shift[j])
                    })
                    .unzip()
            }
            SearchOp::Relu { input, .. } => {
                let (sl, sh) = &t[*input];
                (
                    sl.iter().map(|v| v.max(0.0)).collect(),
                    sh.iter().map(|v| v.max(0.0)).collect(),
                )
            }
            SearchOp::Add { lhs, rhs } => {
                let (a, b) = (&t[*lhs], &t[*rhs]);
                (
                    a.0.iter().zip(&b.0).map(|(x, y)| x + y).collect(),
                    a.1.iter().zip(&b.1).map(|(x, y)| x + y).collect(),
                )
            }
            SearchOp::Flatten { input } => t[*input].clone(),
            SearchOp::Gemm {
                input,
                n_out,
                n_in,
                weight,
                bias,
            } => {
                let (sl, sh) = &t[*input];
                let (mut ol, mut oh) = (bias.clone(), bias.clone());
                for o in 0..*n_out {
                    for i in 0..*n_in {
                        let w = weight[o * n_in + i];
                        let (x, y) = (w * sl[i], w * sh[i]);
                        ol[o] += x.min(y);
                        oh[o] += x.max(y);
                    }
                }
                (ol, oh)
            }
        };
        let pad = |l: f64, h: f64| 1e-9 * (1.0 + l.abs().max(h.abs()));
        let widened = (
            ol.iter().zip(&oh).map(|(&l, &h)| l - pad(l, h)).collect(),
            oh.iter().zip(&ol).map(|(&h, &l)| h + pad(l, h)).collect(),
        );
        t.push(widened);
    }
    t
}

/// Boxes of every ReLU, a `keep` fraction of the unstable neurons as the
/// universe, warm slopes with NaN / out-of-range entries.
fn finish(
    net: SearchNet,
    lo: Vec<f64>,
    hi: Vec<f64>,
    n_out: usize,
    keep: f64,
    rng: &mut Lcg,
) -> WaveProblem {
    let t = ibp(&net, &lo, &hi);
    let (mut boxes, mut universe) = (Vec::new(), Vec::new());
    for (k, op) in net.ops.iter().enumerate() {
        if let SearchOp::Relu { input, .. } = op {
            let (l, u) = t[*input].clone();
            for j in 0..l.len() {
                if l[j] < 0.0 && u[j] > 0.0 && rng.unit() < keep {
                    universe.push(WaveSite {
                        op: k as u32,
                        neuron: j as u32,
                    });
                }
            }
            boxes.push(ReluBox { op: k, l, u });
        }
    }
    let alpha0 = (0..universe.len())
        .map(|p| match p % 7 {
            0 => f64::NAN,
            1 => 1.7,
            2 => -0.3,
            _ => rng.unit(),
        })
        .collect();
    WaveProblem {
        net,
        boxes,
        universe,
        alpha0,
        lo,
        hi,
        target: 0,
        classes: (1..n_out).collect(),
    }
}

fn mlp(seed: u64, n_in: usize, h: usize, n_y: usize, n_out: usize) -> WaveProblem {
    let mut rng = Lcg(seed);
    let net = SearchNet {
        n_in,
        ops: vec![
            SearchOp::Gemm {
                input: 0,
                n_out: h,
                n_in,
                weight: rng.vec(h * n_in, 0.8),
                bias: rng.vec(h, 0.3),
            },
            SearchOp::Relu {
                input: 1,
                head: false,
            },
            SearchOp::Gemm {
                input: 2,
                n_out: n_y,
                n_in: h,
                weight: rng.vec(n_y * h, 0.6),
                bias: rng.vec(n_y, 0.3),
            },
            SearchOp::Relu {
                input: 3,
                head: true,
            },
            SearchOp::Gemm {
                input: 4,
                n_out,
                n_in: n_y,
                weight: rng.vec(n_out * n_y, 0.8),
                bias: rng.vec(n_out, 0.3),
            },
        ],
    };
    let lo: Vec<f64> = (0..n_in).map(|_| rng.uniform(-0.6, 0.0)).collect();
    let hi: Vec<f64> = lo.iter().map(|&l| l + rng.uniform(0.3, 1.0)).collect();
    finish(net, lo, hi, n_out, 1.0, &mut rng)
}

/// The `[k][p]` gather of a `kh x kw` conv over `cin x h x w`.
fn gather(
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    stride: usize,
    pad: usize,
) -> (Vec<u32>, usize, usize) {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kh) / stride + 1;
    let (k, p) = (cin * kh * kh, oh * ow);
    let mut g = vec![GATHER_PAD; k * p];
    for c in 0..cin {
        for ky in 0..kh {
            for kx in 0..kh {
                let tap = (c * kh + ky) * kh + kx;
                for oy in 0..oh {
                    for ox in 0..ow {
                        let iy = (oy * stride + ky) as isize - pad as isize;
                        let ix = (ox * stride + kx) as isize - pad as isize;
                        if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                            g[tap * p + oy * ow + ox] =
                                (c * h * w + iy as usize * w + ix as usize) as u32;
                        }
                    }
                }
            }
        }
    }
    (g, oh, ow)
}

/// `cin x hw x hw` -> conv(c1, stride 2, pad 1) -> ReLU -> conv(c1, pad 1) ->
/// residual add -> ReLU -> affine -> flatten -> dense(ny) -> head ReLU ->
/// dense(n_out).
#[allow(clippy::too_many_arguments)]
fn conv_net(
    seed: u64,
    cin: usize,
    hw: usize,
    c1: usize,
    ny: usize,
    n_out: usize,
    scale: f64,
    eps: f64,
    keep: f64,
) -> WaveProblem {
    let mut rng = Lcg(seed);
    let (g1, oh, ow) = gather(cin, hw, hw, 3, 2, 1);
    let (g2, _, _) = gather(c1, oh, ow, 3, 1, 1);
    let n1 = c1 * oh * ow;
    let (k1, k2) = (cin * 9, c1 * 9);
    let net = SearchNet {
        n_in: cin * hw * hw,
        ops: vec![
            SearchOp::Conv(Box::new(SearchConv {
                input: 0,
                cout: c1,
                k: k1,
                p: oh * ow,
                gather: g1,
                weight: rng.vec(c1 * k1, scale),
                bias: rng.vec(c1, scale),
            })),
            SearchOp::Relu {
                input: 1,
                head: false,
            },
            SearchOp::Conv(Box::new(SearchConv {
                input: 2,
                cout: c1,
                k: k2,
                p: oh * ow,
                gather: g2,
                weight: rng.vec(c1 * k2, scale),
                bias: rng.vec(c1, scale),
            })),
            SearchOp::Add { lhs: 3, rhs: 2 },
            SearchOp::Relu {
                input: 4,
                head: false,
            },
            SearchOp::Affine {
                input: 5,
                scale: (0..n1).map(|_| rng.uniform(0.5, 1.5)).collect(),
                shift: rng.vec(n1, 0.1),
            },
            SearchOp::Flatten { input: 6 },
            SearchOp::Gemm {
                input: 7,
                n_out: ny,
                n_in: n1,
                weight: rng.vec(ny * n1, scale),
                bias: rng.vec(ny, scale),
            },
            SearchOp::Relu {
                input: 8,
                head: true,
            },
            SearchOp::Gemm {
                input: 9,
                n_out,
                n_in: ny,
                weight: rng.vec(n_out * ny, scale),
                bias: rng.vec(n_out, scale),
            },
        ],
    };
    let lo: Vec<f64> = (0..cin * hw * hw)
        .map(|_| rng.uniform(0.0, 1.0 - eps))
        .collect();
    let hi: Vec<f64> = lo.iter().map(|&l| l + eps).collect();
    finish(net, lo, hi, n_out, keep, &mut rng)
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

/// `count` domains of up to `max_lits` literals, optionally with warm slopes,
/// optionally dirty.
fn random_wave(
    p: &WaveProblem,
    count: usize,
    max_lits: usize,
    with_alpha: bool,
    dirty: bool,
    seed: u64,
) -> Wave {
    let mut rng = Lcg(seed);
    let stride = p.universe.len();
    let mut w = Wave {
        rows: Vec::new(),
        offsets: vec![0],
        splits: Vec::new(),
        beta: Vec::new(),
        alpha: Vec::new(),
        stride,
    };
    for _ in 0..count {
        w.rows.push(rng.below(p.classes.len()) as u32);
        let n = rng.below(max_lits + 1).min(stride);
        let mut used: Vec<u32> = Vec::new();
        while used.len() < n {
            let pos = rng.below(stride) as u32;
            if used.contains(&pos) {
                continue;
            }
            used.push(pos);
            w.splits.push(WaveSplit {
                pos,
                active: rng.unit() < 0.5,
            });
            w.beta.push(if dirty && rng.unit() < 0.2 {
                [f32::NAN, f32::INFINITY, -1.0, -0.0][rng.below(4)]
            } else {
                rng.uniform(0.0, 0.5) as f32
            });
        }
        w.offsets.push(w.splits.len() as u32);
        if with_alpha {
            for _ in 0..stride {
                w.alpha.push(if dirty && rng.unit() < 0.15 {
                    [f32::NAN, 1.5, -0.25, f32::NEG_INFINITY][rng.below(4)]
                } else {
                    rng.unit() as f32
                });
            }
        }
    }
    w
}

// ---------------------------------------------------------------------------
//  The comparator
// ---------------------------------------------------------------------------

fn canon(x: f32) -> u32 {
    if x.is_nan() {
        0x7FC0_0000
    } else if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}

/// Every output of a wave the lane reads, bit for bit (canonicalised).
fn digest(out: &WaveOut) -> Vec<u32> {
    let v = out.hint().into_ingest();
    assert!(v.well_formed(), "malformed wave output");
    let mut d = Vec::new();
    for k in 0..v.count() {
        d.extend([canon(v.lb(k)), canon(v.scale(k)), v.evals(k)]);
        d.extend(v.alpha(k).iter().map(|&a| canon(a)));
    }
    d.extend(v.beta().iter().map(|&b| canon(b)));
    d
}

fn branch_digest(out: &BranchOut) -> Vec<(u32, u32)> {
    let v = out.hint().into_ingest();
    assert!(v.well_formed(), "malformed branch output");
    (0..v.count())
        .flat_map(|d| {
            v.pos(d)
                .iter()
                .zip(v.score(d))
                .map(|(&p, &s)| (p, canon(s)))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn search(
    backend: &dyn SearchBackend,
    h: &WaveHandle,
    b: &WaveBatch<'_>,
    cfg: &WaveSearchCfg,
) -> (WaveOut, (u64, u64)) {
    let mut out = WaveOut::default();
    let stats = backend
        .backward_batch(h, b, cfg, &mut out)
        .unwrap_or_else(|r| panic!("{} backward_batch refused: {r}", backend.name()))
        .into_hint()
        .into_ingest();
    (out, (stats.passes, stats.early_stopped))
}

fn branch(backend: &dyn SearchBackend, h: &WaveHandle, b: &WaveBatch<'_>, k: usize) -> BranchOut {
    let mut out = BranchOut::default();
    let _ = backend
        .branch_scores(h, b, k, &mut out)
        .unwrap_or_else(|r| panic!("{} branch_scores refused: {r}", backend.name()));
    out
}

/// Engagement of a comparison set (a vacuous comparison is VOID).
#[derive(Default, Debug)]
struct Engaged {
    domains: usize,
    finite: usize,
    stepped: usize,
    early: u64,
    moved: usize,
}

impl Engaged {
    fn count(&mut self, out: &WaveOut, w: &Wave, early: u64) {
        let v = out.hint().into_ingest();
        for d in 0..v.count() {
            self.domains += 1;
            self.finite += usize::from(v.lb(d).is_finite());
            self.stepped += usize::from(v.evals(d) > 1);
            if !w.alpha.is_empty()
                && v.alpha(d)
                    .iter()
                    .zip(&w.alpha[d * w.stride..(d + 1) * w.stride])
                    .any(|(o, i)| i.is_finite() && (0.0..=1.0).contains(i) && o != i)
            {
                self.moved += 1;
            }
        }
        self.early += early;
    }

    fn assert_engaged(&self) {
        assert!(
            self.domains > 0
                && self.finite > 0
                && self.stepped > 0
                && self.early > 0
                && self.moved > 0,
            "VOID: a comparison set never engaged the ascent: {self:?}"
        );
    }
}

/// The CPU reference and the device on `p` over a matrix of waves and
/// configurations; every output must agree bit for bit.
fn compare(
    device: &ny_cuda::search::CudaSearchBackend,
    name: &str,
    p: &WaveProblem,
    counts: &[(usize, usize)],
    engaged: &mut Engaged,
) -> usize {
    let cpu = ny_bbe::CpuF32Backend::with_threads(3);
    let hc = cpu.upload_wave(p, &BIG).expect("cpu upload");
    let hd = device.upload_wave(p, &BIG).expect("device upload");
    let mut checked = 0usize;
    let mut case = 0u64;
    for &(count, lits) in counts {
        for with_alpha in [false, true] {
            case += 1;
            let w = random_wave(
                p,
                count,
                lits,
                with_alpha,
                case.is_multiple_of(2),
                case * 7919 + count as u64,
            );
            let b = w.batch();
            for cfg in [
                WaveSearchCfg {
                    iters: 0,
                    stop_above: f64::INFINITY,
                    ..WaveSearchCfg::default()
                },
                WaveSearchCfg {
                    iters: 3,
                    stop_above: 0.0,
                    ..WaveSearchCfg::default()
                },
                WaveSearchCfg::default(),
                WaveSearchCfg {
                    iters: 8,
                    stop_above: -1e30,
                    lr_alpha: f64::NAN,
                    lr_beta: -1.0,
                    threads: 2,
                },
            ] {
                let (want, want_counts) = search(&cpu, &hc, &b, &cfg);
                let (got, got_counts) = search(device, &hd, &b, &cfg);
                assert_eq!(
                    digest(&want),
                    digest(&got),
                    "{name}: n={count} alpha={with_alpha} cfg={cfg:?}: the device differs from the CPU \
                     reference"
                );
                assert_eq!(want_counts, got_counts, "{name}: passes / early stops");
                engaged.count(&got, &w, got_counts.1);
                checked += 1;
            }
            for k in [1usize, 3, 16] {
                assert_eq!(
                    branch_digest(&branch(&cpu, &hc, &b, k)),
                    branch_digest(&branch(device, &hd, &b, k)),
                    "{name}: n={count} k={k}: branch candidates differ"
                );
                checked += 1;
            }
        }
    }
    checked
}

// ---------------------------------------------------------------------------
//  Kernel KATs
// ---------------------------------------------------------------------------

#[test]
fn cuda_wave_matches_cpu_f32_bit_for_bit_on_synthetic_waves() {
    let Some(device) = wave_device("cuda_wave_matches_cpu_f32_bit_for_bit_on_synthetic_waves")
    else {
        return;
    };
    let mut engaged = Engaged::default();
    let mut checked = 0usize;
    for (name, p) in [
        ("mlp-small", mlp(3, 5, 12, 6, 4)),
        ("mlp-mid", mlp(11, 40, 64, 20, 10)),
        ("conv-tiny", conv_net(7, 2, 6, 4, 5, 4, 0.35, 0.04, 1.0)),
        ("conv-mid", conv_net(19, 3, 12, 8, 16, 10, 0.25, 0.02, 0.6)),
        ("conv-wide", conv_net(23, 3, 16, 16, 32, 10, 0.2, 0.01, 0.3)),
    ] {
        assert!(!p.universe.is_empty(), "{name}: VOID universe");
        checked += compare(
            &device,
            name,
            &p,
            &[(1, 0), (7, 3), (64, 5), (200, 8)],
            &mut engaged,
        );
    }
    engaged.assert_engaged();
    assert!(checked >= 200, "VOID: {checked}");
    eprintln!("[cuda-wave-kat] synthetic: {checked} comparisons, {engaged:?}");
}

#[test]
fn cuda_wave_matches_cpu_f32_on_cifar_shaped_waves() {
    let Some(device) = wave_device("cuda_wave_matches_cpu_f32_on_cifar_shaped_waves") else {
        return;
    };
    let p = conv_net(29, 3, 32, 32, 64, 10, 0.12, 0.004, 0.05);
    let mut engaged = Engaged::default();
    let checked = compare(&device, "conv-cifar", &p, &[(33, 6)], &mut engaged);
    engaged.assert_engaged();
    assert!(checked > 0);
}

#[test]
fn cuda_wave_is_launch_and_host_thread_independent() {
    let Some(device) = wave_device("cuda_wave_is_launch_and_host_thread_independent") else {
        return;
    };
    let p = conv_net(31, 2, 8, 6, 12, 6, 0.3, 0.03, 1.0);
    let w = random_wave(&p, 41, 6, true, true, 77);
    let cfg = WaveSearchCfg::default();
    let base = {
        let cpu = ny_bbe::CpuF32Backend::with_threads(1);
        let h = cpu.upload_wave(&p, &BIG).expect("upload");
        digest(&search(&cpu, &h, &w.batch(), &cfg).0)
    };
    for threads in [2usize, 7] {
        let cpu = ny_bbe::CpuF32Backend::with_threads(threads);
        let h = cpu.upload_wave(&p, &BIG).expect("upload");
        assert_eq!(
            base,
            digest(&search(&cpu, &h, &w.batch(), &cfg).0),
            "cpu threads {threads}"
        );
    }
    let h = device.upload_wave(&p, &BIG).expect("upload");
    assert_eq!(
        base,
        digest(&search(&device, &h, &w.batch(), &cfg).0),
        "ample lease"
    );
    // Leases that admit only a few domains per slab: the smallest lease the
    // backend accepts, then a little more.
    let mut lease = 1u64 << 12;
    let mut small_leases = Vec::new();
    while small_leases.len() < 3 && lease < 1 << 34 {
        if let Ok(h) = device.upload_wave(&p, &DeviceLease { bytes: lease }) {
            small_leases.push(lease);
            assert_eq!(
                base,
                digest(&search(&device, &h, &w.batch(), &cfg).0),
                "a {lease}-byte lease (a few domains per slab) changed the search"
            );
        }
        lease = lease * 3 / 2;
    }
    assert_eq!(small_leases.len(), 3, "VOID: no small lease was admitted");
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Check that cloning preserves the digest before independently mutating cloned outputs."
)]
fn cuda_wave_kat_comparator_catches_a_one_ulp_mutation() {
    // Device-free: the reference against itself, then with one output nudged
    // by one ulp in each field the comparator reads.
    let p = mlp(5, 6, 10, 5, 4);
    let w = random_wave(&p, 9, 3, true, false, 3);
    let cpu = ny_bbe::CpuF32Backend::with_threads(2);
    let h = cpu.upload_wave(&p, &BIG).expect("upload");
    let (out, _) = search(&cpu, &h, &w.batch(), &WaveSearchCfg::default());
    let base = digest(&out);
    assert_eq!(base, digest(&out.clone()));
    // One ulp (a NaN, which the comparator canonicalises, becomes a value).
    let bump = |x: f32| {
        if x.is_nan() {
            0.5
        } else {
            f32::from_bits(x.to_bits() ^ 1)
        }
    };
    {
        let v = out.hint().into_ingest();
        assert!(
            v.stride() >= 2 && !v.beta().is_empty() && v.lb(4).is_finite(),
            "VOID: the fixture has no slope, multiplier or finite bound to mutate"
        );
    }
    let mut caught = 0usize;
    for field in 0..4 {
        let mut m = out.clone();
        {
            let wr = m.writer();
            match field {
                0 => wr.lb[4] = bump(wr.lb[4]),
                1 => wr.scale[4] = bump(wr.scale[4]),
                2 => wr.alpha[wr.stride + 1] = bump(wr.alpha[wr.stride + 1]),
                _ => wr.beta[0] = bump(wr.beta[0]),
            }
        }
        caught += usize::from(digest(&m) != base);
    }
    assert_eq!(caught, 4, "the comparator missed a one-ulp mutation");
}

// ---------------------------------------------------------------------------
//  The lane: `margin_row.tree_mode: wave_cuda` through the registered factory
// ---------------------------------------------------------------------------

/// Register the device wave search exactly as `main.rs` does (and the CPU
/// search it falls through to).
fn register_device_backend() {
    register_real_backend();
    ny_propagate::margin_row::set_wave_cuda_search_backend_factory(|| {
        Some(Arc::new(ny_cuda::search::CudaSearchBackend::new()) as Arc<dyn SearchBackend>)
    });
}

fn cuda_cfg(cert_threads: usize) -> BabConfig {
    let base = wave_cfg(cert_threads);
    BabConfig {
        wave: WavePolicy {
            mode: TreeMode::WaveCuda,
            ..base.wave
        },
        ..base
    }
}

fn tree_key(out: &MarginRowOutcome) -> (bool, usize, usize, usize, String) {
    match out {
        MarginRowOutcome::Unsat(s, _) => (
            true,
            s.expansions,
            s.domains_created,
            s.closed,
            s.stop.clone(),
        ),
        MarginRowOutcome::Unknown { stats, reason } => {
            stats
                .as_ref()
                .map_or((false, 0, 0, 0, reason.clone()), |s| {
                    (
                        false,
                        s.expansions,
                        s.domains_created,
                        s.closed,
                        s.stop.clone(),
                    )
                })
        }
    }
}

#[test]
fn registered_cuda_backend_reaches_the_armed_wave_tree() {
    if wave_device("registered_cuda_backend_reaches_the_armed_wave_tree").is_none() {
        return;
    }
    register_device_backend();
    let (net, root, t, adv) = tree_instances(1).swap_remove(0);
    let _ = summary();
    let _ = MarginRowBab::run(&net, &root, t, &adv, cuda_cfg(2));
    let line = summary();
    assert!(
        line.contains("[bbe] phase=wave armed=1 backend=cuda "),
        "the armed wave_cuda tree did not search on the device (VOID): {line:?}"
    );
    assert!(
        line.contains("mode=wave_cuda device_refused=none device_switch=none")
            && line.contains("wave-beta-bab/v1"),
        "{line}"
    );
    assert!(!line.contains("fallback=legacy"), "{line}");
    assert!(
        !line.contains(" searched=0 "),
        "the device never searched: {line}"
    );
}

#[test]
fn real_cuda_wave_is_the_cpu_f32_wave_on_tiny_nets() {
    if wave_device("real_cuda_wave_is_the_cpu_f32_wave_on_tiny_nets").is_none() {
        return;
    }
    register_device_backend();
    let instances = tree_instances(4);
    let mut proved = 0usize;
    for (net, root, t, adv) in &instances {
        let on_cpu = MarginRowBab::run(net, root, *t, adv, wave_cfg(2));
        let cpu_line = summary();
        assert!(cpu_line.contains("backend=cpu_f32"), "{cpu_line}");
        let on_device = MarginRowBab::run(net, root, *t, adv, cuda_cfg(2));
        let dev_line = summary();
        assert!(dev_line.contains("backend=cuda "), "{dev_line}");
        assert_eq!(
            tree_key(&on_cpu),
            tree_key(&on_device),
            "the device search is the CPU search: the trees must be the same"
        );
        if let MarginRowOutcome::Unsat(stats, token) = on_device {
            assert!(token.publishable());
            assert_eq!(stats.ledger_ok, Some(true));
            proved += 1;
        }
    }
    assert!(
        proved > 0,
        "the device wave proved none of {} legacy-provable instances: positive control failed",
        instances.len()
    );
}

#[test]
fn real_cuda_wave_never_closes_a_sat_instance() {
    if wave_device("real_cuda_wave_never_closes_a_sat_instance").is_none() {
        return;
    }
    register_device_backend();
    let (mut checked, mut engaged) = (0usize, 0usize);
    for seed in [29u64, 31, 37, 41] {
        let Ok(net) = TwinNet::compile(&tiny_spec(seed, 0.35)) else {
            continue;
        };
        let (lo, hi) = (vec![0.01; 72], vec![0.05; 72]);
        let (t, worst, _) = targets(&net, &lo, &hi);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let _ = summary();
        let out = MarginRowBab::run(
            &net,
            &root,
            worst,
            &[t],
            BabConfig {
                max_expansions: 400,
                ..cuda_cfg(2)
            },
        );
        assert!(
            !matches!(out, MarginRowOutcome::Unsat(..)),
            "MOAT VIOLATION: the device wave closed a witnessed SAT instance (seed {seed})"
        );
        let line = summary();
        engaged += usize::from(line.contains("backend=cuda ") && !line.contains("fallback=legacy"));
        checked += 1;
    }
    assert!(checked >= 2, "only {checked} SAT fixtures compiled: VOID");
    assert!(
        engaged > 0,
        "no SAT fixture reached the device wave engine: VOID"
    );
}
