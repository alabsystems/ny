// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#vgg-gpu-crown`: the CERTIFIED coefficient products, on the GPU or the CPU.
//!
//! Every CROWN backward step through an affine op needs `mu = W^T lambda` for a
//! chunk of rows. This module returns the computed `mu` together with a
//! per-entry bound `err >= |mu - (W^T lambda)_exact|` for the STORED binary64
//! `lambda`; the caller charges `sum err * |x|max` against the box of the value
//! `mu` multiplies (see the module docs of `gpu_patches_crown`).
//!
//! # The certificate (a-posteriori `gamma * |A||B|`)
//!
//! For any evaluation order of a length-`k` dot product (FMA, blocked, split
//! over tiles of the OUTPUT axes — never the contraction),
//! `|fl(sum a_i b_i) - sum a_i b_i| <= gamma_k * sum |a_i b_i|` (Higham,
//! Thm 3.1 / sec 3.1) plus an absolute underflow term. Absent conv taps are
//! EXACT zeros (`0 * w = +-0`, `fl(s + 0) = s`), so they add no rounding.
//! `S = |A||B|` is computed by a second GEMM (f32 on the GPU, over operands
//! rounded UP and never subnormal; f64 on the CPU), and the certified bound is
//!
//! ```text
//!   CPU  (f64 signed, f64 S):  err = gamma64(k)   / (1 - gamma64(k)) * (S + a1) + a2
//!   GPU  (f64 signed, f32 S):  err = gamma64(k)   / (1 - gamma32(k)) * (S + a1) + a2
//!   GPU  (f32 signed, f32 S):  err = gamma32(k+1) / (1 - gamma32(k)) * (S + a1) + a2
//! ```
//!
//! where the `1 / (1 - gamma(k))` factor turns the nearest-rounded `S` into an
//! upper bound of the exact `sum |a||b|`, `a1` covers flush-to-zero losses in
//! the `S` GEMM (`4 (k+1) 2^-126`, or `(k+2) 2^-1074` on the CPU), and `a2`
//! covers underflow in the signed product (plus, on the f32 route, the
//! rounding of `lambda` to f32 — hence `gamma32(k+1)` — and a device that
//! treats subnormal operands as zero). A layer with a nonzero subnormal f32
//! weight refuses the signed f32 route. Everything is evaluated with upward
//! rounding (`next_up`, `slack16`).
//!
//! # Refusal is a bit-identical CPU fallback
//!
//! The GPU route returns a result only WHOLE: every tile's two GEMMs must
//! succeed with the right length and all-finite entries. Any refusal —
//! missing / non-CUDA engine, CPU-only scope, a non-finite or out-of-f32-range
//! operand, an engine error, the pass deadline, a malformed output, a tile the
//! memory cap cannot hold — runs the CPU product, which is deterministic (one
//! task per output entry, fixed order), so the published bound for that chunk
//! is exactly what the dark CPU route publishes. Every refusal is counted.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use ny_core::{f32_to_f64_exact, GemmEngine, NyError};
use rayon::prelude::*;

use super::plan::{abs_up_f32, abs_up_weight_f32, ConvSpec, LinearSpec};
use super::GpuPatchesPrecision;
use crate::margin_row::rounding::{next_down, next_up, slack16, F32_MIN_POSITIVE, UNIT, UNIT_F32};

/// Smallest positive binary64 subnormal, `2^-1074`.
pub(crate) const ETA_F64: f64 = f64::from_bits(1);

/// Requested cap on one non-interruptible engine dispatch (MACs); passed
/// through `GemmEngine::deadline_dispatch_macs_hint`. The engine splits only
/// output axes, so the value never touches the certificate.
pub(crate) const DISPATCH_MACS: usize = 1 << 26;

/// Largest single engine call (MACs). Every call is one non-interruptible
/// library GEMM, so this bounds the pass's cancellation latency: 2^33 MACs is
/// ~21 ms of GB10 `Dgemm` (0.41 TF/s) and well under a millisecond of `Sgemm`.
pub(crate) const PLAIN_MAX_MACS: u128 = 1 << 33;

/// Global-route floor: below this many MACs one engine call's fixed cost
/// dominates, so the CPU product runs (counted as `small`). Performance only.
pub(crate) const MIN_GPU_MACS: u64 = 1 << 22;

/// Output-column block of the CPU linear product (parallel granularity).
const LIN_BLOCK: usize = 1024;

/// `n * unit / (1 - n * unit)` rounded UP, or `None` once `n * unit > 0.01`
/// (a contraction that long gets no certificate and refuses).
pub(crate) fn gamma_up(n: u64, unit: f64) -> Option<f64> {
    if n >= 1 << 53 {
        return None;
    }
    let nu = (n as f64) * unit; // exact: `unit` is a power of two
    if nu > 0.01 || !unit.is_finite() {
        return None;
    }
    let den = next_down(1.0 - nu);
    Some(next_up(next_up(nu) / den))
}

/// Per-product error model: `err(S) = slack16(factor * (S + a1)) + a2`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ErrModel {
    factor: f64,
    a1: f64,
    a2: f64,
}

impl ErrModel {
    /// Certified bound for one entry whose `|A||B|` sum was computed as `s`.
    #[inline]
    pub(crate) fn err(&self, s: f64) -> f64 {
        slack16(self.factor * (s + self.a1)) + self.a2
    }

    /// f64 signed product, f64 `S` (the CPU route).
    pub(crate) fn cpu(k: usize) -> Option<Self> {
        let g = gamma_up(k as u64, UNIT)?;
        let a = ETA_F64 * (k as f64 + 2.0);
        Some(Self {
            factor: next_up(g / next_down(1.0 - g)),
            a1: a,
            a2: a,
        })
    }

    /// f64 signed product on the device, f32 `S`.
    pub(crate) fn gpu_f64(k: usize) -> Option<Self> {
        let g64 = gamma_up(k as u64, UNIT)?;
        let g32 = gamma_up(k as u64, UNIT_F32)?;
        Some(Self {
            factor: next_up(g64 / next_down(1.0 - g32)),
            a1: slack16(4.0 * (k as f64 + 1.0) * F32_MIN_POSITIVE),
            a2: ETA_F64 * (k as f64 + 2.0),
        })
    }

    /// f32 signed product (operand `lambda` rounded to nearest f32), f32 `S`.
    pub(crate) fn gpu_f32(k: usize, wmax: f64) -> Option<Self> {
        let g1 = gamma_up(k as u64 + 1, UNIT_F32)?;
        let g32 = gamma_up(k as u64, UNIT_F32)?;
        if !wmax.is_finite() {
            return None;
        }
        Some(Self {
            factor: next_up(g1 / next_down(1.0 - g32)),
            a1: slack16(4.0 * (k as f64 + 1.0) * F32_MIN_POSITIVE),
            a2: slack16(((k as f64) * (wmax.max(1.0) + 4.0) + 4.0) * F32_MIN_POSITIVE),
        })
    }
}

/// Where the products run.
#[derive(Clone, Default)]
pub(crate) enum Route {
    /// CPU only (the dark reference; bit-identical to every refusal).
    #[default]
    Cpu,
    /// The process-global sound f64 engine slot (cuBLAS on a CUDA build),
    /// admitted per product: CUDA provenance, MAC floor, not in a CPU-only
    /// scope, already materialised.
    Global,
    /// Test / probe injection: THIS engine for every product (no floor, no
    /// provenance check). Same certificate, same fallback.
    #[cfg_attr(not(test), allow(dead_code))]
    Engine(Arc<dyn GemmEngine>),
}

impl Route {
    /// Stop-line tag.
    pub(crate) fn tag(&self) -> &'static str {
        match self {
            Route::Cpu => "cpu",
            Route::Global => "global",
            Route::Engine(_) => "engine",
        }
    }

    /// Does this route ever try an engine?
    pub(crate) fn is_gpu(&self) -> bool {
        !matches!(self, Route::Cpu)
    }
}

/// Why a product ran on the CPU instead (every one is counted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    Small,
    CpuOnly,
    NoEngine,
    Backend,
    Shape,
    NonFiniteIn,
    Subnormal,
    Tile,
    Engine,
    Deadline,
    BadOutput,
}

pub(crate) const REFUSALS: usize = 11;

impl Refusal {
    pub(crate) const ALL: [Refusal; REFUSALS] = [
        Refusal::Small,
        Refusal::CpuOnly,
        Refusal::NoEngine,
        Refusal::Backend,
        Refusal::Shape,
        Refusal::NonFiniteIn,
        Refusal::Subnormal,
        Refusal::Tile,
        Refusal::Engine,
        Refusal::Deadline,
        Refusal::BadOutput,
    ];

    pub(crate) fn index(self) -> usize {
        match self {
            Refusal::Small => 0,
            Refusal::CpuOnly => 1,
            Refusal::NoEngine => 2,
            Refusal::Backend => 3,
            Refusal::Shape => 4,
            Refusal::NonFiniteIn => 5,
            Refusal::Subnormal => 6,
            Refusal::Tile => 7,
            Refusal::Engine => 8,
            Refusal::Deadline => 9,
            Refusal::BadOutput => 10,
        }
    }

    pub(crate) fn tag(self) -> &'static str {
        match self {
            Refusal::Small => "small",
            Refusal::CpuOnly => "cpu_only",
            Refusal::NoEngine => "no_engine",
            Refusal::Backend => "backend",
            Refusal::Shape => "shape",
            Refusal::NonFiniteIn => "nonfinite_in",
            Refusal::Subnormal => "subnormal",
            Refusal::Tile => "tile",
            Refusal::Engine => "engine_err",
            Refusal::Deadline => "deadline",
            Refusal::BadOutput => "bad_output",
        }
    }
}

/// Per-pass counters (atomics: products fan out over Rayon).
#[derive(Default)]
pub(crate) struct Stats {
    /// Products served whole by the engine.
    pub(crate) gpu_passes: AtomicU64,
    /// Engine GEMM calls (two per tile).
    pub(crate) gpu_calls: AtomicU64,
    /// Wall time inside engine products.
    pub(crate) gpu_ns: AtomicU64,
    /// Products computed on the CPU because the route is CPU-only.
    pub(crate) cpu_products: AtomicU64,
    /// Products computed on the CPU after a GPU refusal.
    pub(crate) cpu_fallback: AtomicU64,
    pub(crate) refused: [AtomicU64; REFUSALS],
}

impl Stats {
    fn refuse(&self, why: Refusal) {
        self.cpu_fallback.fetch_add(1, Relaxed);
        self.refused[why.index()].fetch_add(1, Relaxed);
    }
}

/// Everything a product needs besides its operands.
pub(crate) struct Ctx<'a> {
    pub(crate) route: &'a Route,
    pub(crate) stats: &'a Stats,
    pub(crate) precision: GpuPatchesPrecision,
    /// Byte cap on one engine call's operands + result (GPU-memory admission;
    /// on unified memory these bytes are ALSO charged to the host budget).
    pub(crate) tile_bytes: usize,
    pub(crate) deadline: Instant,
}

/// Why a pass step stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    /// The pass deadline expired.
    Expired,
    /// A fail-closed refusal of the whole pass.
    Refused(&'static str),
}

/// A certified product: `mu` and `err >= |mu - exact|`, same layout.
pub(crate) struct Certified {
    pub(crate) mu: Vec<f64>,
    pub(crate) err: Vec<f64>,
}

/// Conv backward geometry for one chunk.
///
/// `lambda` is over the conv OUTPUT (value `k+1`, `cout` channels): per row a
/// `(cout, qh, qw)` window at origin `(oy, ox)`. `mu` is over the conv INPUT
/// (value `k`, `cin` channels): a `(cin, qh2, qw2)` window at `(oy2, ox2)`.
pub(crate) struct ConvGeom<'a> {
    pub(crate) rows: usize,
    pub(crate) qh: usize,
    pub(crate) qw: usize,
    pub(crate) oy: &'a [usize],
    pub(crate) ox: &'a [usize],
    pub(crate) qh2: usize,
    pub(crate) qw2: usize,
    pub(crate) oy2: &'a [usize],
    pub(crate) ox2: &'a [usize],
}

impl ConvGeom<'_> {
    fn valid(&self, spec: &ConvSpec, lam: &[f64]) -> bool {
        self.oy.len() == self.rows
            && self.ox.len() == self.rows
            && self.oy2.len() == self.rows
            && self.ox2.len() == self.rows
            && self.qh > 0
            && self.qw > 0
            && self.qh2 > 0
            && self.qw2 > 0
            && lam.len() == self.rows * spec.cout * self.qh * self.qw
    }

    /// `lambda` window coordinate feeding `mu` window row `y2` through tap
    /// `ky` for row `r`, if any (`Y2 = s*Y - p + ky`).
    #[inline]
    fn src_y(&self, spec: &ConvSpec, r: usize, y2: usize, ky: usize) -> Option<usize> {
        src_coord(self.oy2[r] + y2, spec.ph, ky, spec.sh, self.oy[r], self.qh)
    }

    #[inline]
    fn src_x(&self, spec: &ConvSpec, r: usize, x2: usize, kx: usize) -> Option<usize> {
        src_coord(self.ox2[r] + x2, spec.pw, kx, spec.sw, self.ox[r], self.qw)
    }
}

/// Solve `abs2 = s*abs - p + k` for `abs`, then shift into the window
/// `[origin, origin + q)`.
#[inline]
fn src_coord(abs2: usize, p: usize, k: usize, s: usize, origin: usize, q: usize) -> Option<usize> {
    let num = abs2 as isize + p as isize - k as isize;
    if num < 0 || num % s as isize != 0 {
        return None;
    }
    let local = num / s as isize - origin as isize;
    (local >= 0 && local < q as isize).then_some(local as usize)
}

/// Run `attempt` on the route's engine, or say why not (`None` = CPU route).
fn gpu_attempt<R>(
    ctx: &Ctx<'_>,
    macs: u64,
    attempt: impl FnOnce(&dyn GemmEngine) -> Result<R, Refusal>,
) -> Option<Result<R, Refusal>> {
    match ctx.route {
        Route::Cpu => None,
        Route::Engine(engine) => Some(attempt(engine.as_ref())),
        Route::Global => {
            if macs < MIN_GPU_MACS {
                return Some(Err(Refusal::Small));
            }
            if crate::sound_f64_gemm::cpu_only_f64_active() {
                return Some(Err(Refusal::CpuOnly));
            }
            let ran = crate::sound_f64_gemm::with_preinitialized_engine(|engine| {
                if !backend_admitted(engine.backend_provenance()) {
                    return Err(Refusal::Backend);
                }
                attempt(engine)
            });
            Some(ran.unwrap_or_else(|| {
                crate::sound_f64_gemm::start_background_initialization();
                Err(Refusal::NoEngine)
            }))
        }
    }
}

/// The global route admits only a CUDA engine (a CPU floor engine in the same
/// slot would just compete with the host pool).
pub(crate) fn backend_admitted(backend: &str) -> bool {
    backend.starts_with("cuda")
}

fn classify(e: &NyError) -> Refusal {
    match e {
        NyError::DeadlineExceeded(_) => Refusal::Deadline,
        _ => Refusal::Engine,
    }
}

/// Is one call small enough for an engine's plain (non-deadline) GEMM?
fn plain_call_admitted(m: usize, k: usize, n: usize) -> bool {
    (m as u128) * (k as u128) * (n as u128) <= PLAIN_MAX_MACS
}

/// One RN-f32 GEMM. The engine's plain `gemm_f32` runs the whole (tile-
/// bounded) call as ONE library GEMM; an engine without it is asked for its
/// deadline-bounded method instead. Either way the result is validated whole.
#[allow(clippy::too_many_arguments)]
fn f32_gemm(
    engine: &dyn GemmEngine,
    stats: &Stats,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    deadline: Instant,
    cap: usize,
) -> Result<Vec<f32>, Refusal> {
    if !plain_call_admitted(m, k, n) {
        return Err(Refusal::Tile);
    }
    if Instant::now() >= deadline {
        return Err(Refusal::Deadline);
    }
    stats.gpu_calls.fetch_add(1, Relaxed);
    let c = match engine.gemm_f32(m, k, n, a, b) {
        Ok(c) => c,
        Err(NyError::UnsupportedOp(_)) => engine
            .gemm_f32_with_deadline(m, k, n, a, b, deadline, cap)
            .map_err(|e| classify(&e))?,
        Err(e) => return Err(classify(&e)),
    };
    if c.len() != m * n || c.iter().any(|v| !v.is_finite()) {
        return Err(Refusal::BadOutput);
    }
    Ok(c)
}

/// One IEEE-f64 GEMM, same policy as [`f32_gemm`]: the plain `gemm_f64`
/// (cuBLAS `Dgemm` as ONE call — the deadline method's 2^24-MAC dispatches
/// would re-read the right operand once per output row), else the
/// deadline-bounded method.
#[allow(clippy::too_many_arguments)]
fn f64_gemm(
    engine: &dyn GemmEngine,
    stats: &Stats,
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    deadline: Instant,
    cap: usize,
) -> Result<Vec<f64>, Refusal> {
    if !plain_call_admitted(m, k, n) {
        return Err(Refusal::Tile);
    }
    if Instant::now() >= deadline {
        return Err(Refusal::Deadline);
    }
    stats.gpu_calls.fetch_add(1, Relaxed);
    let c = match engine.gemm_f64(m, k, n, a, b) {
        Ok(c) => c,
        Err(NyError::UnsupportedOp(_)) => engine
            .gemm_f64_with_deadline(m, k, n, a, b, deadline, cap)
            .map_err(|e| classify(&e))?,
        Err(e) => return Err(classify(&e)),
    };
    if c.len() != m * n || c.iter().any(|v| !v.is_finite()) {
        return Err(Refusal::BadOutput);
    }
    Ok(c)
}

/// Every stored `lambda` finite and inside f32 range (the `S` GEMM is f32 on
/// every GPU route).
fn lam_admissible(lam: &[f64]) -> bool {
    let limit = f64::from(f32::MAX) / 2.0;
    lam.iter().all(|v| v.is_finite() && v.abs() < limit)
}

fn elapsed_ns(t0: Instant) -> u64 {
    u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Dispatch one certified product through the route with CPU fallback.
fn route_product(
    ctx: &Ctx<'_>,
    macs: u64,
    gpu: impl FnOnce(&dyn GemmEngine) -> Result<Certified, Refusal>,
    cpu: impl FnOnce() -> Result<Certified, Stop>,
) -> Result<Certified, Stop> {
    let t0 = Instant::now();
    match gpu_attempt(ctx, macs, gpu) {
        None => {
            ctx.stats.cpu_products.fetch_add(1, Relaxed);
            cpu()
        }
        Some(Ok(done)) => {
            ctx.stats.gpu_passes.fetch_add(1, Relaxed);
            ctx.stats.gpu_ns.fetch_add(elapsed_ns(t0), Relaxed);
            Ok(done)
        }
        Some(Err(why)) => {
            ctx.stats.refuse(why);
            cpu()
        }
    }
}

// ---------------------------------------------------------------------------
//  Conv backward
// ---------------------------------------------------------------------------

/// Certified conv backward `mu = W^T lambda` for one chunk.
pub(crate) fn conv_product(
    ctx: &Ctx<'_>,
    spec: &ConvSpec,
    g: &ConvGeom<'_>,
    lam: &[f64],
) -> Result<Certified, Stop> {
    if !g.valid(spec, lam) {
        return Err(Stop::Refused("conv geometry"));
    }
    let macs = [spec.cin, spec.contraction(), g.rows, g.qh2, g.qw2]
        .iter()
        .fold(1u64, |acc, &x| acc.saturating_mul(x as u64));
    route_product(
        ctx,
        macs,
        |engine| conv_gpu(engine, ctx, spec, g, lam),
        || conv_cpu(ctx, spec, g, lam),
    )
}

/// The CPU reference: one task per `(row, input channel)` plane, every entry
/// summed in the fixed `(c, ky, kx)` order — deterministic for any thread
/// count, which is what makes every refusal bit-identical to the dark route.
pub(crate) fn conv_cpu(
    ctx: &Ctx<'_>,
    spec: &ConvSpec,
    g: &ConvGeom<'_>,
    lam: &[f64],
) -> Result<Certified, Stop> {
    let k = spec.contraction();
    let model = ErrModel::cpu(k).ok_or(Stop::Refused("conv contraction too long"))?;
    let in_len = spec.cout * g.qh * g.qw;
    let plane2 = g.qh2 * g.qw2;
    let mut mu = vec![0.0_f64; g.rows * spec.cin * plane2];
    let mut err = vec![0.0_f64; g.rows * spec.cin * plane2];
    let expired = AtomicBool::new(false);
    let deadline = ctx.deadline;
    mu.par_chunks_mut(plane2)
        .zip(err.par_chunks_mut(plane2))
        .enumerate()
        .for_each(|(task, (mu_p, s_p))| {
            if expired.load(Relaxed) {
                return;
            }
            if Instant::now() >= deadline {
                expired.store(true, Relaxed);
                return;
            }
            let r = task / spec.cin;
            let ci = task % spec.cin;
            let lam_r = &lam[r * in_len..(r + 1) * in_len];
            for c in 0..spec.cout {
                let lam_c = &lam_r[c * g.qh * g.qw..(c + 1) * g.qh * g.qw];
                for ky in 0..spec.kh {
                    for kx in 0..spec.kw {
                        let w = spec.weight(c, ci, ky, kx);
                        if w == 0.0 {
                            continue;
                        }
                        let wa = w.abs();
                        for y2 in 0..g.qh2 {
                            let Some(y) = g.src_y(spec, r, y2, ky) else {
                                continue;
                            };
                            let src = &lam_c[y * g.qw..(y + 1) * g.qw];
                            let mu_row = &mut mu_p[y2 * g.qw2..(y2 + 1) * g.qw2];
                            let s_row = &mut s_p[y2 * g.qw2..(y2 + 1) * g.qw2];
                            for x2 in 0..g.qw2 {
                                let Some(x) = g.src_x(spec, r, x2, kx) else {
                                    continue;
                                };
                                let l = src[x];
                                mu_row[x2] += l * w;
                                s_row[x2] += l.abs() * wa;
                            }
                        }
                    }
                }
            }
            for s in s_p.iter_mut() {
                *s = model.err(*s);
            }
        });
    if expired.load(Relaxed) {
        return Err(Stop::Expired);
    }
    Ok(Certified { mu, err })
}

/// Gather the GEMM right operand for `mu` lines `l0..l1` (a line is one
/// `(row, y2)` pair of `qw2` columns): `B[(t*cout + c), col] = lambda` at the
/// back tap `t = (ky, kx)`, else EXACT zero.
fn gather_conv<T: Copy + Default + Send + Sync>(
    spec: &ConvSpec,
    g: &ConvGeom<'_>,
    lam: &[f64],
    l0: usize,
    l1: usize,
    map: impl Fn(f64) -> T + Sync,
) -> Vec<T> {
    let n = (l1 - l0) * g.qw2;
    let in_len = spec.cout * g.qh * g.qw;
    let mut b = vec![T::default(); spec.contraction() * n];
    b.par_chunks_mut(n).enumerate().for_each(|(row, brow)| {
        let t = row / spec.cout;
        let c = row % spec.cout;
        let (ky, kx) = (t / spec.kw, t % spec.kw);
        for line in l0..l1 {
            let r = line / g.qh2;
            let y2 = line % g.qh2;
            let Some(y) = g.src_y(spec, r, y2, ky) else {
                continue;
            };
            let base = r * in_len + (c * g.qh + y) * g.qw;
            let src = &lam[base..base + g.qw];
            let dst = &mut brow[(line - l0) * g.qw2..(line - l0 + 1) * g.qw2];
            for (x2, slot) in dst.iter_mut().enumerate() {
                if let Some(x) = g.src_x(spec, r, x2, kx) {
                    *slot = map(src[x]);
                }
            }
        }
    });
    b
}

fn conv_gpu(
    engine: &dyn GemmEngine,
    ctx: &Ctx<'_>,
    spec: &ConvSpec,
    g: &ConvGeom<'_>,
    lam: &[f64],
) -> Result<Certified, Refusal> {
    let use_f32 = ctx.precision == GpuPatchesPrecision::F32;
    if !lam_admissible(lam) {
        return Err(Refusal::NonFiniteIn);
    }
    if use_f32 && spec.subnormal_f32 {
        return Err(Refusal::Subnormal);
    }
    let k = spec.contraction();
    let model = if use_f32 {
        ErrModel::gpu_f32(k, spec.wmax)
    } else {
        ErrModel::gpu_f64(k)
    }
    .ok_or(Refusal::Shape)?;
    let cap = engine.deadline_dispatch_macs_hint(DISPATCH_MACS).max(1);
    if k > cap {
        return Err(Refusal::Shape);
    }
    let cin = spec.cin;
    let plane2 = g.qh2 * g.qw2;
    let lines = g.rows * g.qh2;
    // Device + host bytes per call: A (signed <= 8 B + |A| 4 B), B (<= 8 + 4),
    // C (<= 8 + 4) per column.
    let fixed = cin * k * 12;
    let per_line = (k + cin) * g.qw2 * 12;
    let per_tile = ctx.tile_bytes.saturating_sub(fixed) / per_line.max(1);
    let macs_per_line = (cin as u128) * (k as u128) * (g.qw2 as u128);
    let mac_lines = usize::try_from(PLAIN_MAX_MACS / macs_per_line.max(1)).unwrap_or(usize::MAX);
    let lines_per_tile = per_tile.min(mac_lines).min(lines);
    if lines_per_tile == 0 {
        return Err(Refusal::Tile);
    }
    let a_abs = spec.gemm_a32_abs();
    let mut mu = vec![0.0_f64; g.rows * cin * plane2];
    let mut s = vec![0.0_f64; g.rows * cin * plane2];
    let mut l0 = 0usize;
    while l0 < lines {
        if Instant::now() >= ctx.deadline {
            return Err(Refusal::Deadline);
        }
        let l1 = (l0 + lines_per_tile).min(lines);
        let n = (l1 - l0) * g.qw2;
        let b_abs = gather_conv(spec, g, lam, l0, l1, abs_up_f32);
        let c_abs = f32_gemm(
            engine,
            ctx.stats,
            cin,
            k,
            n,
            a_abs,
            &b_abs,
            ctx.deadline,
            cap,
        )?;
        drop(b_abs);
        let c_signed: Vec<f64> = if use_f32 {
            let b32 = gather_conv(spec, g, lam, l0, l1, |v| v as f32);
            let c = f32_gemm(
                engine,
                ctx.stats,
                cin,
                k,
                n,
                spec.gemm_a32(),
                &b32,
                ctx.deadline,
                cap,
            )?;
            c.iter().map(|&v| f32_to_f64_exact(v)).collect()
        } else {
            let b64 = gather_conv(spec, g, lam, l0, l1, |v| v);
            f64_gemm(
                engine,
                ctx.stats,
                cin,
                k,
                n,
                spec.gemm_a64(),
                &b64,
                ctx.deadline,
                cap,
            )?
        };
        // Scatter `C (cin x lines*qw2)` into the rows this tile touches, one
        // Rayon task per row (disjoint `mu` / `s` slices).
        let row_len = cin * plane2;
        let (r0, r1) = (l0 / g.qh2, (l1 - 1) / g.qh2 + 1);
        mu[r0 * row_len..r1 * row_len]
            .par_chunks_mut(row_len)
            .zip(s[r0 * row_len..r1 * row_len].par_chunks_mut(row_len))
            .enumerate()
            .for_each(|(dr, (mu_r, s_r))| {
                let r = r0 + dr;
                let y_lo = l0.saturating_sub(r * g.qh2).min(g.qh2);
                let y_hi = (l1 - r * g.qh2).min(g.qh2);
                for ci in 0..cin {
                    for y2 in y_lo..y_hi {
                        let src = ci * n + (r * g.qh2 + y2 - l0) * g.qw2;
                        let dst = ci * plane2 + y2 * g.qw2;
                        mu_r[dst..dst + g.qw2].copy_from_slice(&c_signed[src..src + g.qw2]);
                        for (slot, &v) in s_r[dst..dst + g.qw2]
                            .iter_mut()
                            .zip(&c_abs[src..src + g.qw2])
                        {
                            *slot = f32_to_f64_exact(v);
                        }
                    }
                }
            });
        l0 = l1;
    }
    // The certificate IN PLACE over `S`: a separate `err` vector would hold
    // `mu`, `S` and `err` (three output-sized vectors) at once, one more than
    // the chunk admission (`backward::row_bytes`) charges per row.
    s.par_iter_mut().for_each(|v| *v = model.err(*v));
    if s.iter().any(|v| !v.is_finite()) {
        return Err(Refusal::BadOutput);
    }
    Ok(Certified { mu, err: s })
}

// ---------------------------------------------------------------------------
//  Linear backward
// ---------------------------------------------------------------------------

/// Certified linear backward `mu (rows x nin) = lambda (rows x nout) . W`.
pub(crate) fn linear_product(
    ctx: &Ctx<'_>,
    spec: &LinearSpec<'_>,
    rows: usize,
    lam: &[f64],
) -> Result<Certified, Stop> {
    if lam.len() != rows * spec.nout {
        return Err(Stop::Refused("linear geometry"));
    }
    let macs = [rows, spec.nout, spec.nin]
        .iter()
        .fold(1u64, |acc, &x| acc.saturating_mul(x as u64));
    route_product(
        ctx,
        macs,
        |engine| linear_gpu(engine, ctx, spec, rows, lam),
        || linear_cpu(ctx, spec, rows, lam),
    )
}

/// The CPU reference: blocks of output columns in parallel, every entry
/// summed over `o` ascending — deterministic for any thread count.
pub(crate) fn linear_cpu(
    ctx: &Ctx<'_>,
    spec: &LinearSpec<'_>,
    rows: usize,
    lam: &[f64],
) -> Result<Certified, Stop> {
    let model = ErrModel::cpu(spec.nout).ok_or(Stop::Refused("linear contraction too long"))?;
    let (nout, nin) = (spec.nout, spec.nin);
    let mut mu = vec![0.0_f64; rows * nin];
    let mut err = vec![0.0_f64; rows * nin];
    let expired = AtomicBool::new(false);
    let deadline = ctx.deadline;
    mu.par_chunks_mut(nin)
        .zip(err.par_chunks_mut(nin))
        .enumerate()
        .for_each(|(r, (mu_r, s_r))| {
            let lam_r = &lam[r * nout..(r + 1) * nout];
            mu_r.par_chunks_mut(LIN_BLOCK)
                .zip(s_r.par_chunks_mut(LIN_BLOCK))
                .enumerate()
                .for_each(|(blk, (mb, sb))| {
                    if expired.load(Relaxed) {
                        return;
                    }
                    if Instant::now() >= deadline {
                        expired.store(true, Relaxed);
                        return;
                    }
                    let j0 = blk * LIN_BLOCK;
                    for (o, &l) in lam_r.iter().enumerate() {
                        if l == 0.0 {
                            continue;
                        }
                        let la = l.abs();
                        let wrow = &spec.w[o * nin + j0..o * nin + j0 + mb.len()];
                        for ((m, s), &wf) in mb.iter_mut().zip(sb.iter_mut()).zip(wrow) {
                            let wv = f32_to_f64_exact(wf);
                            *m += l * wv;
                            *s += la * wv.abs();
                        }
                    }
                    for s in sb.iter_mut() {
                        *s = model.err(*s);
                    }
                });
        });
    if expired.load(Relaxed) {
        return Err(Stop::Expired);
    }
    Ok(Certified { mu, err })
}

fn linear_gpu(
    engine: &dyn GemmEngine,
    ctx: &Ctx<'_>,
    spec: &LinearSpec<'_>,
    rows: usize,
    lam: &[f64],
) -> Result<Certified, Refusal> {
    let use_f32 = ctx.precision == GpuPatchesPrecision::F32;
    if !lam_admissible(lam) {
        return Err(Refusal::NonFiniteIn);
    }
    if use_f32 && spec.subnormal_f32 {
        return Err(Refusal::Subnormal);
    }
    let (nout, nin) = (spec.nout, spec.nin);
    let model = if use_f32 {
        ErrModel::gpu_f32(nout, spec.wmax)
    } else {
        ErrModel::gpu_f64(nout)
    }
    .ok_or(Refusal::Shape)?;
    let cap = engine.deadline_dispatch_macs_hint(DISPATCH_MACS).max(1);
    if nout > cap {
        return Err(Refusal::Shape);
    }
    // Per call: A (rows x nout, <= 8 + 4 B), B (nout x n, <= 8 + 4 B),
    // C (rows x n, <= 8 + 4 B).
    let fixed = rows * nout * 12;
    let per_col = (nout + rows) * 12;
    let per_tile = ctx.tile_bytes.saturating_sub(fixed) / per_col.max(1);
    let macs_per_col = (rows as u128) * (nout as u128);
    let mac_cols = usize::try_from(PLAIN_MAX_MACS / macs_per_col.max(1)).unwrap_or(usize::MAX);
    let cols = per_tile.min(mac_cols).min(nin);
    if cols == 0 {
        return Err(Refusal::Tile);
    }
    let a_abs: Vec<f32> = lam.par_iter().map(|&v| abs_up_f32(v)).collect();
    let a32: Vec<f32> = if use_f32 {
        lam.par_iter().map(|&v| v as f32).collect()
    } else {
        Vec::new()
    };
    let mut mu = vec![0.0_f64; rows * nin];
    let mut s = vec![0.0_f64; rows * nin];
    let mut j0 = 0usize;
    while j0 < nin {
        if Instant::now() >= ctx.deadline {
            return Err(Refusal::Deadline);
        }
        let j1 = (j0 + cols).min(nin);
        let n = j1 - j0;
        let mut b_abs = vec![0.0_f32; nout * n];
        b_abs.par_chunks_mut(n).enumerate().for_each(|(o, brow)| {
            for (slot, &wf) in brow.iter_mut().zip(&spec.w[o * nin + j0..o * nin + j1]) {
                *slot = abs_up_weight_f32(wf);
            }
        });
        let c_abs = f32_gemm(
            engine,
            ctx.stats,
            rows,
            nout,
            n,
            &a_abs,
            &b_abs,
            ctx.deadline,
            cap,
        )?;
        drop(b_abs);
        let c_signed: Vec<f64> = if use_f32 {
            let mut b32 = vec![0.0_f32; nout * n];
            b32.par_chunks_mut(n).enumerate().for_each(|(o, brow)| {
                brow.copy_from_slice(&spec.w[o * nin + j0..o * nin + j1]);
            });
            let c = f32_gemm(
                engine,
                ctx.stats,
                rows,
                nout,
                n,
                &a32,
                &b32,
                ctx.deadline,
                cap,
            )?;
            c.iter().map(|&v| f32_to_f64_exact(v)).collect()
        } else {
            let mut b64 = vec![0.0_f64; nout * n];
            b64.par_chunks_mut(n).enumerate().for_each(|(o, brow)| {
                for (slot, &wf) in brow.iter_mut().zip(&spec.w[o * nin + j0..o * nin + j1]) {
                    *slot = f32_to_f64_exact(wf);
                }
            });
            f64_gemm(
                engine,
                ctx.stats,
                rows,
                nout,
                n,
                lam,
                &b64,
                ctx.deadline,
                cap,
            )?
        };
        mu.par_chunks_mut(nin)
            .zip(s.par_chunks_mut(nin))
            .enumerate()
            .for_each(|(r, (mu_r, s_r))| {
                mu_r[j0..j1].copy_from_slice(&c_signed[r * n..(r + 1) * n]);
                for (slot, &v) in s_r[j0..j1].iter_mut().zip(&c_abs[r * n..(r + 1) * n]) {
                    *slot = f32_to_f64_exact(v);
                }
            });
        j0 = j1;
    }
    // In place, as in `conv_gpu` (the admission charges `mu + err` only).
    s.par_iter_mut().for_each(|v| *v = model.err(*v));
    if s.iter().any(|v| !v.is_finite()) {
        return Err(Refusal::BadOutput);
    }
    Ok(Certified { mu, err: s })
}
