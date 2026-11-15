// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M6 — output-only beta-CROWN over a WAVE of margin-row sub-domains,
//! UNTRUSTED, on the CPU (ENGINE_PROGRAM §2 M6; `margin_row.tree_mode: wave`).
//!
//! # What it computes
//!
//! A wave is a batch of sub-domains of one margin-row tree. Domain `k` is the
//! input box intersected with its split literals (`z_j >= 0` or `z_j <= 0` for
//! universe neurons `j`) and asks for a lower bound of ONE objective row
//! `y_target - y_class` on the net's last tensor. Per domain, abc's tree
//! bounding recipe (`study_abc.md` §1.3-1.5):
//!
//! * the lane's CERTIFIED pre-activation boxes are FROZEN for the whole tree
//!   (no per-domain intermediate refresh: abc `interm_transfer: False`);
//! * the domain's splits PIN their neurons exactly (active: identity,
//!   inactive: zero) and each carries a Lagrange multiplier `beta_k >= 0` for
//!   its constraint `s_k z_k >= 0` (weak duality: `min f >= min f - sum beta_k
//!   s_k z_k` over the relaxation), added to `z_k`'s backward coefficient right
//!   after the ReLU transform, exactly where the lane's certified engine adds
//!   it (`margin_row::beta::set_terms`);
//! * every other unstable neuron of the universe carries its own lower slope
//!   `alpha_j in [0, 1]` (the upper chord is the box's);
//! * `iters` projected Adam ASCENT steps on `(alpha, beta)` from the
//!   concretisation vertex `x*` of one CROWN backward pass and the linearised
//!   forward walk at `x*`: `d lb / d alpha_j = Lambda_j z_j(x*)` (for the
//!   neurons whose lower line the pass chose) and `d lb / d beta_k = -s_k
//!   z_k(x*)` — the same frozen-sign linearisation the lane's own beta ascent
//!   and the M5a root search use; keep-best over the evaluated iterates; a
//!   domain whose bound clears `stop_above` leaves early.
//!
//! [`branch_scores`] runs ONE backward pass at the batch's slopes and
//! multipliers and ranks each domain's unsplit universe neurons by the BaBSR
//! intercept score `|Lambda_j| * (-u_j l_j / (u_j - l_j))` (the relaxation
//! area the split removes, weighted by how much the objective reads the
//! neuron), best first, ties to the lower universe position.
//!
//! # Trust
//!
//! UNTRUSTED end to end: f32 in production (f64 in the in-crate reference
//! instantiation), round to nearest, no rounding charged, the boxes rounded to
//! the search scalar. Only HINTS leave: the best bound (the lane's claim
//! threshold compares it, and a claim is then certified by the lane's own f64
//! Outward evaluator or refuted and split), the multipliers and slopes (clamped
//! again at the lane's ingest: any `beta >= 0` and any `alpha in [0, 1]` is a
//! valid relaxation) and branch candidates (any split of a region covers it).
//! A bug here costs time, never a verdict.
//!
//! # Cost shape
//!
//! Dense per-domain backward and forward walks over the program (zero
//! coefficients skipped), one domain per worker at a time; domains are claimed
//! from an atomic counter and scattered back by index, so every output depends
//! only on its own domain — not on the thread count, the claim order or which
//! domains share the wave (R6; `wave_search_is_thread_count_and_batch_independent`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use ny_core::search::{
    wave_adam_bias, wave_ingest_rates, wave_lr_decay, wave_warm_alpha, wave_warm_beta, BranchOut,
    DeviceLease, Refusal, SearchOp, WaveBatch, WaveHandle, WaveOut, WaveProblem, WaveSearchCfg,
    WaveSplit, WaveStats, WAVE_ADAM_B1 as ADAM_B1, WAVE_ADAM_B2 as ADAM_B2,
    WAVE_ADAM_EPS as ADAM_EPS, WAVE_MAX_BRANCH_K, WAVE_NONE,
};

use crate::program::{Op, Program, NONE};
use crate::real::Real;

// Adam's constants, its bias corrections, the per-iteration learning-rate
// decay (abc's BaB `lr_decay` 0.98), the learning-rate ingest and the warm-hint
// ingest are `ny_core::search`'s (`WAVE_ADAM_*`, `wave_adam_bias`,
// `wave_lr_decay`, `wave_ingest_rates`, `wave_warm_alpha` / `wave_warm_beta`):
// one copy for this reference and the device search that reproduces it.

/// Candidates a [`branch_scores`] call may return per domain.
pub(crate) const MAX_BRANCH_K: usize = WAVE_MAX_BRANCH_K;

// `BranchOut` pads with `WAVE_NONE`; the program's "no position" is `NONE`.
const _: () = assert!(WAVE_NONE == NONE);

/// A ReLU site's frozen box, upper chords, adaptive slope and universe map.
pub(crate) struct SiteState<R> {
    l: Vec<R>,
    u: Vec<R>,
    slope: Vec<R>,
    icpt: Vec<R>,
    adaptive: Vec<R>,
    /// Universe position per neuron, [`NONE`] outside the universe.
    pos: Vec<u32>,
}

impl<R: Real> SiteState<R> {
    fn new(l: &[f64], u: &[f64]) -> Self {
        let n = l.len();
        let mut s = Self {
            l: l.iter().map(|&v| R::from_f64(v)).collect(),
            u: u.iter().map(|&v| R::from_f64(v)).collect(),
            slope: vec![R::ZERO; n],
            icpt: vec![R::ZERO; n],
            adaptive: vec![R::ZERO; n],
            pos: vec![NONE; n],
        };
        for j in 0..n {
            let (l, u) = (s.l[j], s.u[j]);
            let (sl, ic) = if !(l < R::ZERO && u > R::ZERO) {
                (R::ZERO, R::ZERO)
            } else if l.is_finite() && u.is_finite() {
                let sl = u / (u - l);
                (sl, -(sl * l))
            } else if u.is_finite() {
                (R::ZERO, u)
            } else {
                (R::ZERO, R::INFINITY)
            };
            s.slope[j] = sl;
            s.icpt[j] = ic;
            s.adaptive[j] = if u >= -l { R::ONE } else { R::ZERO };
        }
        s
    }

    #[inline]
    fn unstable(&self, j: usize) -> bool {
        self.l[j] < R::ZERO && self.u[j] > R::ZERO
    }
}

/// A lowered wave problem (the payload of a [`WaveHandle`]).
pub(crate) struct WaveUpload<R> {
    pub(crate) prog: Program<R>,
    sites: Vec<SiteState<R>>,
    /// Per universe position: the program site and the neuron.
    uni_site: Vec<u32>,
    uni_neuron: Vec<u32>,
    alpha0: Vec<R>,
    mid: Vec<R>,
    rad: Vec<R>,
    target: usize,
    classes: Vec<usize>,
    pub(crate) threads: usize,
}

impl<R: Real> WaveUpload<R> {
    /// Lower and validate a problem.
    pub(crate) fn lower(problem: &WaveProblem, threads: usize) -> Result<Self, Refusal> {
        problem.validate().map_err(Refusal::Invalid)?;
        let prog = Program::<R>::lower(&problem.net).map_err(Refusal::Invalid)?;
        let mut site_of_op = vec![NONE; problem.net.ops.len()];
        for (s, site) in prog.sites.iter().enumerate() {
            site_of_op[site.op] = s as u32;
        }
        let mut sites = Vec::with_capacity(prog.sites.len());
        for site in &prog.sites {
            let b = problem
                .boxes
                .iter()
                .find(|b| b.op == site.op)
                .ok_or_else(|| Refusal::Invalid(format!("no box for ReLU op {}", site.op)))?;
            sites.push(SiteState::<R>::new(&b.l, &b.u));
        }
        let mut uni_site = Vec::with_capacity(problem.universe.len());
        let mut uni_neuron = Vec::with_capacity(problem.universe.len());
        for (p, site) in problem.universe.iter().enumerate() {
            let s = site_of_op
                .get(site.op as usize)
                .copied()
                .filter(|&s| s != NONE)
                .ok_or_else(|| {
                    Refusal::Invalid(format!("universe op {} is not a ReLU", site.op))
                })?;
            sites[s as usize].pos[site.neuron as usize] = p as u32;
            uni_site.push(s);
            uni_neuron.push(site.neuron);
        }
        let alpha0 = problem
            .universe
            .iter()
            .zip(&problem.alpha0)
            .zip(uni_site.iter().zip(&uni_neuron))
            .map(|((_, &a), (&s, &j))| {
                if a.is_finite() {
                    R::from_f64(a.clamp(0.0, 1.0))
                } else {
                    sites[s as usize].adaptive[j as usize]
                }
            })
            .collect();
        let mid = problem
            .lo
            .iter()
            .zip(&problem.hi)
            .map(|(&l, &h)| R::from_f64(f64::midpoint(l, h)))
            .collect();
        let rad = problem
            .lo
            .iter()
            .zip(&problem.hi)
            .map(|(&l, &h)| R::from_f64((h - l) * 0.5))
            .collect();
        Ok(Self {
            prog,
            sites,
            uni_site,
            uni_neuron,
            alpha0,
            mid,
            rad,
            target: problem.target,
            classes: problem.classes.clone(),
            threads: threads.max(1),
        })
    }

    /// Universe positions (the slope stride).
    pub(crate) fn stride(&self) -> usize {
        self.uni_site.len()
    }

    /// Resident bytes of the upload.
    pub(crate) fn bytes(&self) -> u64 {
        let r = R::BYTES as u64;
        let prog: u64 = self
            .prog
            .ops
            .iter()
            .map(|op| match op {
                Op::Conv {
                    gather_pk, w, b, ..
                } => (gather_pk.len() as u64) * 4 + ((w.len() + b.len()) as u64) * r,
                Op::Affine { scale, shift, .. } => ((scale.len() + shift.len()) as u64) * r,
                Op::Gemm { w, b, .. } => ((w.len() + b.len()) as u64) * r,
                Op::Relu { .. } | Op::Add { .. } | Op::Copy { .. } => 0,
            })
            .fold(0u64, u64::saturating_add);
        let site_n: u64 = self.sites.iter().map(|s| s.l.len() as u64).sum();
        prog.saturating_add(site_n.saturating_mul(5 * r + 4))
            .saturating_add((self.stride() as u64).saturating_mul(8 + r))
    }

    /// One worker's working set.
    pub(crate) fn work_bytes(&self) -> u64 {
        let r = R::BYTES as u64;
        let tensors: u64 = self.prog.tsize.iter().map(|&n| n as u64).sum();
        let site_n: u64 = self.sites.iter().map(|s| s.l.len() as u64).sum();
        let stride = self.stride() as u64;
        tensors
            .saturating_mul(2 * r)
            .saturating_add(site_n.saturating_mul(r + 5))
            // alpha, best alpha and the gradient in R; two Adam moments in f64.
            .saturating_add(stride.saturating_mul(3 * r + 16))
    }
}

/// One worker's reusable buffers.
struct Ws<R> {
    /// Backward coefficients per tensor (all zero between passes).
    coef: Vec<Vec<R>>,
    /// Linearised forward values per tensor.
    val: Vec<Vec<R>>,
    /// Backward coefficient at every ReLU output, per site.
    lam: Vec<Vec<R>>,
    /// Pin per site neuron: `1` active, `-1` inactive, `0` free.
    pin: Vec<Vec<i8>>,
    /// Split index per site neuron, [`NONE`] when unsplit.
    split_at: Vec<Vec<u32>>,
    g_alpha: Vec<R>,
    g_beta: Vec<R>,
}

impl<R: Real> Ws<R> {
    fn new(up: &WaveUpload<R>) -> Self {
        Self {
            coef: up.prog.tsize.iter().map(|&n| vec![R::ZERO; n]).collect(),
            val: up.prog.tsize.iter().map(|&n| vec![R::ZERO; n]).collect(),
            lam: up.sites.iter().map(|s| vec![R::ZERO; s.l.len()]).collect(),
            pin: up.sites.iter().map(|s| vec![0i8; s.l.len()]).collect(),
            split_at: up.sites.iter().map(|s| vec![NONE; s.l.len()]).collect(),
            g_alpha: vec![R::ZERO; up.stride()],
            g_beta: Vec::new(),
        }
    }

    fn pin_splits(&mut self, up: &WaveUpload<R>, splits: &[WaveSplit]) {
        for (k, sp) in splits.iter().enumerate() {
            let p = sp.pos as usize;
            let (s, j) = (up.uni_site[p] as usize, up.uni_neuron[p] as usize);
            self.pin[s][j] = if sp.active { 1 } else { -1 };
            self.split_at[s][j] = k as u32;
        }
    }

    fn unpin_splits(&mut self, up: &WaveUpload<R>, splits: &[WaveSplit]) {
        for sp in splits {
            let p = sp.pos as usize;
            let (s, j) = (up.uni_site[p] as usize, up.uni_neuron[p] as usize);
            self.pin[s][j] = 0;
            self.split_at[s][j] = NONE;
        }
    }
}

/// What one pass computes besides the bound.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    /// The bound only (branch scores read `lam` afterwards).
    Bound,
    /// The bound and the slope / multiplier gradients.
    Grad,
}

/// One CROWN backward pass of objective row `row` for the pinned domain in
/// `ws`, at slopes `alpha` and multipliers `beta` (aligned with `splits`), then
/// optionally the linearised forward walk that yields the gradients. Returns
/// the bound and its absolute-term sum.
fn pass<R: Real>(
    up: &WaveUpload<R>,
    ws: &mut Ws<R>,
    row: usize,
    splits: &[WaveSplit],
    alpha: &[R],
    beta: &[R],
    want: Want,
) -> (R, R) {
    let prog = &up.prog;
    let Ws {
        coef,
        val,
        lam,
        pin,
        split_at,
        g_alpha,
        g_beta,
    } = ws;
    for buf in coef.iter_mut() {
        buf.fill(R::ZERO);
    }
    let out_t = prog.out_tensor();
    coef[out_t][up.target] += R::ONE;
    coef[out_t][up.classes[row]] -= R::ONE;
    let mut bias = R::ZERO;
    let mut bias_abs = R::ZERO;

    // ---- backward: op outputs -> op inputs, in reverse op order ------------
    for k in (0..prog.ops.len()).rev() {
        let out = k + 1;
        let obuf = std::mem::take(&mut coef[out]);
        match &prog.ops[k] {
            Op::Conv {
                input,
                k: taps,
                p,
                gather_pk,
                w,
                b,
            } => {
                let (a, taps, p) = (*input, *taps, *p);
                let dst = &mut coef[a];
                for (idx, &v) in obuf.iter().enumerate() {
                    if v == R::ZERO {
                        continue;
                    }
                    let (o, sp) = (idx / p, idx % p);
                    let t = b[o] * v;
                    bias += t;
                    bias_abs += t.abs();
                    let g_row = &gather_pk[sp * taps..(sp + 1) * taps];
                    let w_row = &w[o * taps..(o + 1) * taps];
                    for (&g, &wt) in g_row.iter().zip(w_row) {
                        if g == NONE || wt == R::ZERO {
                            continue;
                        }
                        dst[g as usize] += wt * v;
                    }
                }
            }
            Op::Affine {
                input,
                scale,
                shift,
            } => {
                let dst = &mut coef[*input];
                for (j, &v) in obuf.iter().enumerate() {
                    if v == R::ZERO {
                        continue;
                    }
                    let t = shift[j] * v;
                    bias += t;
                    bias_abs += t.abs();
                    dst[j] += scale[j] * v;
                }
            }
            Op::Relu { input, site } => {
                let (a, s) = (*input, *site);
                let sb = &up.sites[s];
                let lam_s = &mut lam[s];
                let dst = &mut coef[a];
                for (j, &v) in obuf.iter().enumerate() {
                    lam_s[j] = v;
                    let pj = pin[s][j];
                    let mut c = R::ZERO;
                    if pj > 0 || (pj == 0 && sb.l[j] >= R::ZERO) {
                        c = v;
                    } else if pj < 0 || sb.u[j] <= R::ZERO {
                        // Inactive: the output is exactly zero.
                    } else if v >= R::ZERO {
                        let p = sb.pos[j];
                        let a_j = if p == NONE {
                            sb.adaptive[j]
                        } else {
                            alpha[p as usize]
                        };
                        c = a_j * v;
                    } else {
                        c = sb.slope[j] * v;
                        let t = sb.icpt[j] * v;
                        bias += t;
                        bias_abs += t.abs();
                    }
                    let ks = split_at[s][j];
                    if ks != NONE {
                        // The Lagrangian term `- beta_k s_k z_k`.
                        let ks = ks as usize;
                        if splits[ks].active {
                            c -= beta[ks];
                        } else {
                            c += beta[ks];
                        }
                    }
                    if c != R::ZERO {
                        dst[j] += c;
                    }
                }
            }
            Op::Add { lhs, rhs } => {
                for (j, &v) in obuf.iter().enumerate() {
                    if v == R::ZERO {
                        continue;
                    }
                    coef[*lhs][j] += v;
                    coef[*rhs][j] += v;
                }
            }
            Op::Copy { input } => {
                let dst = &mut coef[*input];
                for (j, &v) in obuf.iter().enumerate() {
                    if v != R::ZERO {
                        dst[j] += v;
                    }
                }
            }
            Op::Gemm { input, n_in, w, b } => {
                let (a, n_in) = (*input, *n_in);
                let dst = &mut coef[a];
                for (o, &v) in obuf.iter().enumerate() {
                    if v == R::ZERO {
                        continue;
                    }
                    let t = b[o] * v;
                    bias += t;
                    bias_abs += t.abs();
                    let w_row = &w[o * n_in..(o + 1) * n_in];
                    for (i, &wi) in w_row.iter().enumerate() {
                        if wi != R::ZERO {
                            dst[i] += wi * v;
                        }
                    }
                }
            }
        }
        coef[out] = obuf;
    }

    // ---- concretise over the input box; x* for the forward walk ------------
    let mut lb = bias;
    let mut scale = bias_abs;
    {
        let c0 = &coef[0];
        let x = &mut val[0];
        for i in 0..prog.n_in {
            let (c, m, r) = (c0[i], up.mid[i], up.rad[i]);
            lb += c * m - c.abs() * r;
            scale += c.abs() * (m.abs() + r);
            x[i] = if c > R::ZERO {
                m - r
            } else if c < R::ZERO {
                m + r
            } else {
                m
            };
        }
    }
    if want == Want::Bound {
        return (lb, scale);
    }

    // ---- forward: the linearised network at x*, harvesting the gradients --
    g_alpha.fill(R::ZERO);
    g_beta.clear();
    g_beta.resize(splits.len(), R::ZERO);
    for k in 0..prog.ops.len() {
        let out = k + 1;
        let mut ob = std::mem::take(&mut val[out]);
        match &prog.ops[k] {
            Op::Conv {
                input,
                k: taps,
                p,
                gather_pk,
                w,
                b,
            } => {
                let (a, taps, p) = (*input, *taps, *p);
                let src = &val[a];
                for (idx, cell) in ob.iter_mut().enumerate() {
                    let (o, sp) = (idx / p, idx % p);
                    let mut acc = b[o];
                    let g_row = &gather_pk[sp * taps..(sp + 1) * taps];
                    let w_row = &w[o * taps..(o + 1) * taps];
                    for (&g, &wt) in g_row.iter().zip(w_row) {
                        if g != NONE && wt != R::ZERO {
                            acc += wt * src[g as usize];
                        }
                    }
                    *cell = acc;
                }
            }
            Op::Affine {
                input,
                scale: sc,
                shift,
            } => {
                let src = &val[*input];
                for (j, cell) in ob.iter_mut().enumerate() {
                    *cell = sc[j] * src[j] + shift[j];
                }
            }
            Op::Relu { input, site } => {
                let (a, s) = (*input, *site);
                let sb = &up.sites[s];
                let src = &val[a];
                for (j, cell) in ob.iter_mut().enumerate() {
                    let z = src[j];
                    let pj = pin[s][j];
                    let lj = lam[s][j];
                    *cell = if pj > 0 || (pj == 0 && sb.l[j] >= R::ZERO) {
                        z
                    } else if pj < 0 || sb.u[j] <= R::ZERO {
                        R::ZERO
                    } else if lj >= R::ZERO {
                        let p = sb.pos[j];
                        if p == NONE {
                            sb.adaptive[j] * z
                        } else {
                            let p = p as usize;
                            g_alpha[p] += lj * z;
                            alpha[p] * z
                        }
                    } else {
                        sb.slope[j] * z + sb.icpt[j]
                    };
                    let ks = split_at[s][j];
                    if ks != NONE {
                        let ks = ks as usize;
                        g_beta[ks] = if splits[ks].active { -z } else { z };
                    }
                }
            }
            Op::Add { lhs, rhs } => {
                let (x, y) = (&val[*lhs], &val[*rhs]);
                for (j, cell) in ob.iter_mut().enumerate() {
                    *cell = x[j] + y[j];
                }
            }
            Op::Copy { input } => {
                ob.copy_from_slice(&val[*input]);
            }
            Op::Gemm { input, n_in, w, b } => {
                let (a, n_in) = (*input, *n_in);
                let src = &val[a];
                for (o, cell) in ob.iter_mut().enumerate() {
                    let mut acc = b[o];
                    let w_row = &w[o * n_in..(o + 1) * n_in];
                    for (i, &wi) in w_row.iter().enumerate() {
                        if wi != R::ZERO {
                            acc += wi * src[i];
                        }
                    }
                    *cell = acc;
                }
            }
        }
        val[out] = ob;
    }
    (lb, scale)
}

/// One domain's search result.
struct DomOut {
    lb: f32,
    scale: f32,
    evals: u32,
    early: bool,
    beta: Vec<f32>,
    alpha: Vec<f32>,
}

fn past(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() > d)
}

/// The ingest of a warm multiplier (`ny_core::search::wave_warm_beta`):
/// finite and positive, else 0.
#[inline]
fn warm_beta<R: Real>(b: f32) -> R {
    R::from_f64(wave_warm_beta(b))
}

/// The ingest of a warm slope (`ny_core::search::wave_warm_alpha`): finite ->
/// clamped into `[0, 1]`, else `fall`.
#[inline]
fn warm_alpha<R: Real>(a: f32, fall: R) -> R {
    wave_warm_alpha(a).map_or(fall, R::from_f64)
}

/// Adam moments of one domain's ascent.
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    /// One projected ascent step (`t` counts from 1), then `project`.
    fn step<R: Real>(
        &mut self,
        x: &mut [R],
        g: &[R],
        lr: f64,
        t: i32,
        project: impl Fn(f64) -> f64,
    ) {
        let (c1, c2) = wave_adam_bias(t);
        for i in 0..x.len() {
            let gi = g[i].to_f64();
            if !gi.is_finite() {
                continue;
            }
            self.m[i] = ADAM_B1 * self.m[i] + (1.0 - ADAM_B1) * gi;
            self.v[i] = ADAM_B2 * self.v[i] + (1.0 - ADAM_B2) * gi * gi;
            if self.m[i] == 0.0 {
                continue;
            }
            let mh = self.m[i] / c1;
            let vh = self.v[i] / c2;
            let next = project(x[i].to_f64() + lr * mh / (vh.sqrt() + ADAM_EPS));
            if next.is_finite() {
                x[i] = R::from_f64(next);
            }
        }
    }
}

/// Search one domain (see the module docs). `None` when the deadline passed.
#[allow(clippy::too_many_arguments)]
fn run_domain<R: Real>(
    up: &WaveUpload<R>,
    ws: &mut Ws<R>,
    row: usize,
    splits: &[WaveSplit],
    beta0: &[f32],
    alpha0: Option<&[f32]>,
    cfg: &WaveSearchCfg,
    deadline: Option<Instant>,
) -> Option<DomOut> {
    let stride = up.stride();
    let mut alpha: Vec<R> = match alpha0 {
        Some(a) => a
            .iter()
            .zip(&up.alpha0)
            .map(|(&v, &fall)| warm_alpha(v, fall))
            .collect(),
        None => up.alpha0.clone(),
    };
    let mut beta: Vec<R> = beta0.iter().map(|&b| warm_beta(b)).collect();
    let mut best_alpha = alpha.clone();
    let mut best_beta = beta.clone();
    let mut best_lb = f64::NAN;
    let mut best_scale = f64::NAN;
    let mut adam_a = Adam::new(stride);
    let mut adam_b = Adam::new(splits.len());
    let (lr_a, lr_b) = wave_ingest_rates(cfg.lr_alpha, cfg.lr_beta);
    ws.pin_splits(up, splits);
    let mut evals = 0u32;
    let mut early = false;
    let mut late = false;
    for it in 0..=cfg.iters {
        if past(deadline) {
            late = true;
            break;
        }
        let want = if it < cfg.iters {
            Want::Grad
        } else {
            Want::Bound
        };
        let (lb, scale) = pass(up, ws, row, splits, &alpha, &beta, want);
        evals = evals.saturating_add(1);
        let (lb, scale) = (lb.to_f64(), scale.to_f64());
        if lb.is_finite() && (best_lb.is_nan() || lb > best_lb) {
            best_lb = lb;
            best_scale = scale;
            best_alpha.copy_from_slice(&alpha);
            best_beta.copy_from_slice(&beta);
        }
        if !lb.is_finite() {
            // An infinite chord or a NaN: no usable gradient from here.
            break;
        }
        if lb > cfg.stop_above {
            early = true;
            break;
        }
        if want == Want::Bound {
            break;
        }
        let t = i32::try_from(it + 1).unwrap_or(i32::MAX);
        let decay = wave_lr_decay(it);
        adam_a.step(&mut alpha, &ws.g_alpha, lr_a * decay, t, |x| {
            x.clamp(0.0, 1.0)
        });
        adam_b.step(&mut beta, &ws.g_beta, lr_b * decay, t, |x| x.max(0.0));
    }
    ws.unpin_splits(up, splits);
    if late {
        return None;
    }
    Some(DomOut {
        lb: best_lb as f32,
        scale: best_scale as f32,
        evals,
        early,
        beta: best_beta.iter().map(|b| b.to_f32()).collect(),
        alpha: best_alpha.iter().map(|a| a.to_f32()).collect(),
    })
}

/// Branch candidates of one domain: its unsplit universe neurons unstable in
/// the frozen box, by descending intercept score, ties to the lower position.
fn score_domain<R: Real>(
    up: &WaveUpload<R>,
    ws: &mut Ws<R>,
    row: usize,
    splits: &[WaveSplit],
    beta: &[R],
    alpha: &[R],
    k: usize,
) -> Vec<(u32, f32)> {
    ws.pin_splits(up, splits);
    let _ = pass(up, ws, row, splits, alpha, beta, Want::Bound);
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for p in 0..up.stride() {
        let (s, j) = (up.uni_site[p] as usize, up.uni_neuron[p] as usize);
        if ws.pin[s][j] != 0 || !up.sites[s].unstable(j) {
            continue;
        }
        let (l, u) = (up.sites[s].l[j].to_f64(), up.sites[s].u[j].to_f64());
        let lam = ws.lam[s][j].to_f64().abs();
        let area = if l.is_finite() && u.is_finite() {
            -u * l / (u - l)
        } else {
            1.0
        };
        let mut score = (lam * area) as f32;
        if !score.is_finite() {
            score = f32::MAX;
        }
        // Insert keeping `top` sorted: score descending, position ascending.
        let at = top
            .iter()
            .position(|&(q, sc)| score > sc || (score == sc && (p as u32) < q))
            .unwrap_or(top.len());
        if at < k {
            top.insert(at, (p as u32, score));
            top.truncate(k);
        }
    }
    ws.unpin_splits(up, splits);
    top
}

/// Run `work(domain, ws)` for every domain on `threads` workers (domains
/// claimed from an atomic counter), returning the per-domain results in
/// domain order. A worker panic or a result of `None` (the deadline) refuses.
fn run_domains<R: Real, T: Send>(
    up: &WaveUpload<R>,
    count: usize,
    threads: usize,
    deadline: Option<Instant>,
    work: &(dyn Fn(usize, &mut Ws<R>) -> Option<T> + Sync),
) -> Result<Vec<T>, Refusal> {
    let workers = threads.max(1).min(count.max(1));
    let next = AtomicUsize::new(0);
    let cut = AtomicBool::new(false);
    let run = || -> Vec<(usize, T)> {
        let mut ws = Ws::new(up);
        let mut local = Vec::new();
        loop {
            if cut.load(Ordering::Relaxed) {
                break;
            }
            if past(deadline) {
                cut.store(true, Ordering::Relaxed);
                break;
            }
            let d = next.fetch_add(1, Ordering::Relaxed);
            if d >= count {
                break;
            }
            match work(d, &mut ws) {
                Some(out) => local.push((d, out)),
                None => {
                    cut.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
        local
    };
    let parts: Vec<Vec<(usize, T)>> = if workers == 1 {
        vec![run()]
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers).map(|_| scope.spawn(&run)).collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| Refusal::Invalid("wave search worker panicked".into()))
                })
                .collect::<Result<Vec<_>, Refusal>>()
        })?
    };
    if cut.load(Ordering::Relaxed) {
        return Err(Refusal::Deadline);
    }
    let mut slots: Vec<Option<T>> = (0..count).map(|_| None).collect();
    for (d, out) in parts.into_iter().flatten() {
        slots[d] = Some(out);
    }
    slots
        .into_iter()
        .map(|s| s.ok_or_else(|| Refusal::Invalid("wave search lost a domain".into())))
        .collect()
}

/// M6 upload: lower the problem, refusing when the copy plus every worker's
/// working set exceeds the lease.
pub(crate) fn upload<R: Real>(
    backend: &'static str,
    problem: &WaveProblem,
    lease: &DeviceLease,
    threads: usize,
) -> Result<WaveHandle, Refusal> {
    if !problem
        .net
        .ops
        .iter()
        .any(|op| matches!(op, SearchOp::Relu { .. }))
    {
        return Err(Refusal::Nothing("no ReLU to relax"));
    }
    let up = WaveUpload::<R>::lower(problem, threads)?;
    let need = up
        .bytes()
        .saturating_add((up.threads as u64).saturating_mul(up.work_bytes()));
    if need > lease.bytes {
        return Err(Refusal::Memory {
            need,
            cap: lease.bytes,
        });
    }
    let stride = up.stride();
    Ok(WaveHandle::new(backend, stride, up))
}

fn upload_of<'h, R: Real>(
    handle: &'h WaveHandle,
    batch: &WaveBatch<'_>,
) -> Result<&'h WaveUpload<R>, Refusal> {
    let up = handle
        .payload::<WaveUpload<R>>()
        .ok_or_else(|| Refusal::Invalid(format!("foreign wave handle ({})", handle.backend())))?;
    batch
        .validate(up.stride(), up.classes.len())
        .map_err(Refusal::Invalid)?;
    if past(batch.deadline) {
        return Err(Refusal::Deadline);
    }
    Ok(up)
}

fn domain_lits(batch: &WaveBatch<'_>, d: usize) -> std::ops::Range<usize> {
    batch.offsets[d] as usize..batch.offsets[d + 1] as usize
}

/// M6 `backward_batch` on the upload in `handle`.
pub(crate) fn backward_batch<R: Real>(
    handle: &WaveHandle,
    batch: &WaveBatch<'_>,
    cfg: &WaveSearchCfg,
    out: &mut WaveOut,
) -> Result<WaveStats, Refusal> {
    let t0 = Instant::now();
    let up = upload_of::<R>(handle, batch)?;
    let stride = up.stride();
    out.prepare(batch.count, stride, batch.splits.len());
    if batch.count == 0 {
        return Ok(WaveStats::default());
    }
    let threads = if cfg.threads > 0 {
        cfg.threads
    } else {
        up.threads
    };
    let work = |d: usize, ws: &mut Ws<R>| -> Option<DomOut> {
        let lits = domain_lits(batch, d);
        let alpha0 = (!batch.alpha.is_empty()).then(|| &batch.alpha[d * stride..(d + 1) * stride]);
        run_domain(
            up,
            ws,
            batch.rows[d] as usize,
            &batch.splits[lits.clone()],
            &batch.beta[lits],
            alpha0,
            cfg,
            batch.deadline,
        )
    };
    let results = run_domains(up, batch.count, threads, batch.deadline, &work)?;
    let w = out.writer();
    let mut stats = WaveStats {
        domains: batch.count as u64,
        threads: threads.max(1).min(batch.count),
        ..WaveStats::default()
    };
    for (d, r) in results.into_iter().enumerate() {
        w.lb[d] = r.lb;
        w.scale[d] = r.scale;
        w.evals[d] = r.evals;
        stats.passes += u64::from(r.evals);
        stats.early_stopped += u64::from(r.early);
        let lits = domain_lits(batch, d);
        w.beta[lits].copy_from_slice(&r.beta);
        w.alpha[d * stride..(d + 1) * stride].copy_from_slice(&r.alpha);
    }
    stats.secs = t0.elapsed().as_secs_f64();
    Ok(stats)
}

/// M6 `branch_scores` on the upload in `handle`.
pub(crate) fn branch_scores<R: Real>(
    handle: &WaveHandle,
    batch: &WaveBatch<'_>,
    k: usize,
    out: &mut BranchOut,
) -> Result<(), Refusal> {
    let up = upload_of::<R>(handle, batch)?;
    if k == 0 || k > MAX_BRANCH_K {
        return Err(Refusal::Invalid(format!(
            "branch_scores: k = {k} outside 1..={MAX_BRANCH_K}"
        )));
    }
    let stride = up.stride();
    out.prepare(batch.count, k);
    if batch.count == 0 {
        return Ok(());
    }
    let work = |d: usize, ws: &mut Ws<R>| -> Option<Vec<(u32, f32)>> {
        let lits = domain_lits(batch, d);
        let beta: Vec<R> = batch.beta[lits.clone()]
            .iter()
            .map(|&b| warm_beta(b))
            .collect();
        let alpha: Vec<R> = if batch.alpha.is_empty() {
            up.alpha0.clone()
        } else {
            batch.alpha[d * stride..(d + 1) * stride]
                .iter()
                .zip(&up.alpha0)
                .map(|(&a, &fall)| warm_alpha(a, fall))
                .collect()
        };
        Some(score_domain(
            up,
            ws,
            batch.rows[d] as usize,
            &batch.splits[lits],
            &beta,
            &alpha,
            k,
        ))
    };
    let results = run_domains(up, batch.count, up.threads, batch.deadline, &work)?;
    let w = out.writer();
    for (d, top) in results.into_iter().enumerate() {
        for (c, (p, s)) in top.into_iter().enumerate() {
            w.pos[d * k + c] = p;
            w.score[d * k + c] = s;
        }
    }
    Ok(())
}

/// TEST ONLY: the bound, slope gradient and multiplier gradient of one domain
/// at the given point (the kernels' known-answer tests read these directly).
#[cfg(test)]
pub(crate) fn probe<R: Real>(
    handle: &WaveHandle,
    row: usize,
    splits: &[WaveSplit],
    alpha: &[f64],
    beta: &[f64],
) -> Option<(f64, f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let up = handle.payload::<WaveUpload<R>>()?;
    let mut ws = Ws::new(up);
    let a: Vec<R> = alpha.iter().map(|&v| R::from_f64(v)).collect();
    let b: Vec<R> = beta.iter().map(|&v| R::from_f64(v)).collect();
    ws.pin_splits(up, splits);
    let (lb, scale) = pass(up, &mut ws, row, splits, &a, &b, Want::Grad);
    let lam: Vec<f64> = up
        .uni_site
        .iter()
        .zip(&up.uni_neuron)
        .map(|(&s, &j)| ws.lam[s as usize][j as usize].to_f64())
        .collect();
    let ga = ws.g_alpha.iter().map(|g| g.to_f64()).collect();
    let gb = ws.g_beta.iter().map(|g| g.to_f64()).collect();
    ws.unpin_splits(up, splits);
    Some((lb.to_f64(), scale.to_f64(), ga, gb, lam))
}

/// TEST ONLY: bytes one worker's buffers actually hold for the upload in
/// `handle` (pins the `work_bytes` admission estimate from below).
#[cfg(test)]
pub(crate) fn workspace_capacity_bytes<R: Real>(handle: &WaveHandle, splits: usize) -> Option<u64> {
    let up = handle.payload::<WaveUpload<R>>()?;
    let mut ws = Ws::new(up);
    ws.g_beta.resize(splits, R::ZERO);
    let r = R::BYTES as u64;
    let vecs = |v: &Vec<Vec<R>>| v.iter().map(|b| b.capacity() as u64 * r).sum::<u64>();
    let pins: u64 = ws.pin.iter().map(|b| b.capacity() as u64).sum();
    let at: u64 = ws.split_at.iter().map(|b| b.capacity() as u64 * 4).sum();
    // The per-domain ascent state `run_domain` holds beside the workspace:
    // alpha and its best copy in R, two f64 Adam moments.
    let ascent = (up.stride() as u64) * (2 * r + 16);
    Some(
        vecs(&ws.coef)
            + vecs(&ws.val)
            + vecs(&ws.lam)
            + pins
            + at
            + (ws.g_alpha.capacity() as u64) * r
            + ascent,
    )
}
