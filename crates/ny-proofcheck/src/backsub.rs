// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched DAG back-substitution with α/β/chord relaxations and a-priori
//! Higham charges — the checker's own sound lower bound on a linear objective
//! over a node region (ENGINE_PROGRAM §1.7 B2; brief §1, §4.3).
//!
//! # The invariant
//!
//! For every row, with `x` ranging over the node region `R_n`:
//!
//! ```text
//!   objective(x)  >=  Σ_T Λ̂_T · v_T(x)  +  κ  −  ε
//! ```
//!
//! where `v_T(x)` is the exact value of tensor `T`, `Λ̂_T` the computed
//! coefficient row of every tensor still "live" in the sweep, `κ` a
//! down-rounded constant and `ε` an up-rounded charge. It holds at the seed
//! (`Λ̂_seed = seed`, `κ = κ₀`, `ε = 0`) and every step preserves it:
//!
//! * **accumulation.** A tensor's row is a sum of `n` rounded terms, so the
//!   computed row differs from the exact sum of its terms by at most
//!   `γ_n Σ|terms| + n·2^-1074` per element; multiplied by `M_T >= |v_T|` and
//!   summed that is [`crate::arith::charge`], added to `ε` once the row is
//!   final (right before its producer is processed).
//! * **Conv / Gemm** `v_T = W v_x + b`: `Λ̂_T·v_T = (Λ̂_T W)·v_x + Λ̂_T·b`
//!   exactly; the product's terms are charged through `Q = |W| M_x`
//!   (`Σ_i M_x,i Σ_j |Λ̂_T,j||W_ji| = Σ_j |Λ̂_T,j| Q_j`), the bias dot is
//!   rounded down into `κ`.
//! * **BatchNormalization** (and the elementwise [`Op::Scale`] of a
//!   Sub / Mul / Div / Neg by constants) `v_T = s·v_x + sh`, `s ∈ [s_lo, s_hi]`,
//!   `sh ∈ [sh_lo, sh_hi]` (the real parameters are irrational): the term
//!   `Λ̂_T s_mid` enters the input row; `|Λ̂_T|·s_rad·M_x` is charged to `ε`;
//!   `Λ̂_T·sh >= Σ min(Λ̂ sh_lo, Λ̂ sh_hi)` enters `κ`.
//! * **Add / AddConst / Identity**: exact identities (the constant's dot is
//!   rounded down into `κ`).
//! * **Relu** `h = relu(z)`, bounds `[l, u] ⊇ z(R_n)`: stable-off → `0`;
//!   stable-on → `z`; unstable with `λ >= 0` → `λ h >= (λα) z` for any
//!   `α ∈ [0, 1]` (pointwise on all of ℝ); with `λ < 0` → `λ h >= λ s z +
//!   λ t` for the chord of [`crate::arith::relu_chord`] (valid on `[l, u]`),
//!   `λ t` rounded down into `κ`. `l > u` certifies `R_n = ∅`.
//! * **β.** On `R_n` every literal holds, `σ z_s >= 0`, so for `β >= 0`
//!   `objective >= objective − β σ z_s`: the coefficient of `z_s` gains
//!   `−β` for a `z >= 0` literal and `+β` for `z <= 0` (the caller builds
//!   the `coef`; the margin-row lane's `-s_j·β_j` convention is the same).
//!
//! At the input the row is concretized over the node box with every product
//! and sum rounded down: `L = κ − ε + Σ_i min(Λ̂_i lo_i, Λ̂_i hi_i)`. A
//! non-finite `L` is reported as NaN, so `L > 0` can never accept it.
//!
//! Rows are independent and each is summed sequentially in a fixed index
//! order, so a row's bits do not depend on the chunking or the thread count.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::arith::{
    add_down, add_up, charge, concretize_down, dot_abs_up, dot_down, down, relu_chord, up,
};
use crate::ibp::Magnitudes;
use crate::lower::{Conv, Gemm, Net, Op};

/// Hint slopes for unstable neurons: site tensor -> neuron -> α.
pub type AlphaMap = HashMap<usize, HashMap<usize, f64>>;

/// Certified bounds of site tensors over the region being bounded.
pub trait SiteView: Sync {
    /// Lower and upper ends of site tensor `t`.
    fn site(&self, t: usize) -> (&[f64], &[f64]);
}

/// Objective rows, seeded at one tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Seeds {
    /// The tensor the rows are expressed over.
    pub tensor: usize,
    /// Sparse rows `(element, coefficient)`.
    pub rows: Vec<Vec<(usize, f64)>>,
    /// Per row, a LOWER bound on the row's constant term.
    pub kappa0: Vec<f64>,
    /// Upper bound on the number of rounded terms per seed element.
    pub nterms: usize,
}

/// A β term: `coef` is added to the coefficient of `(tensor, index)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaTerm {
    /// Site tensor.
    pub tensor: usize,
    /// Neuron.
    pub index: usize,
    /// Coefficient shift (`-β` for `z >= 0`, `+β` for `z <= 0`).
    pub coef: f64,
}

/// One back-substitution job.
pub struct Ctx<'a> {
    /// The network.
    pub net: &'a Net,
    /// Region magnitudes (valid over a superset of the node region).
    pub mags: &'a Magnitudes,
    /// Site bounds over the node region.
    pub view: &'a dyn SiteView,
    /// Hint α (`None` = adaptive).
    pub alpha: Option<&'a AlphaMap>,
    /// β terms.
    pub betas: &'a [BetaTerm],
    /// Node box, lower ends.
    pub box_lo: &'a [f64],
    /// Node box, upper ends.
    pub box_hi: &'a [f64],
    /// Rows per chunk (memory governor).
    pub chunk_rows: usize,
    /// Keep each row's final linear form.
    pub want_lin: bool,
}

/// One row's result.
#[derive(Debug, Clone, PartialEq)]
pub struct RowOut {
    /// Certified lower bound (NaN when nothing could be certified).
    pub lb: f64,
    /// The row met an empty certified interval: the region is empty.
    pub empty: bool,
    /// The final linear form over the input and its certified constant
    /// (`objective(x) >= lin·x + c` on the region), when requested — and
    /// never for a row whose relaxation failed (`lb` NaN for that reason).
    pub lin: Option<(Vec<f64>, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Relax {
    Off,
    On,
    Unst {
        alpha: f64,
        s: f64,
        t: f64,
    },
    /// Unstable but the chord cannot be built (non-finite bounds): only
    /// `λ >= 0` rows can pass through it.
    LowerOnly {
        alpha: f64,
    },
    Empty,
}

fn adaptive_alpha(l: f64, u: f64) -> f64 {
    if u >= -l {
        1.0
    } else {
        0.0
    }
}

fn relaxations(ctx: &Ctx<'_>, upto: usize) -> Vec<Vec<Relax>> {
    let mut out = vec![Vec::new(); ctx.net.nodes.len()];
    for k in 0..=upto {
        let Op::Relu { x } = ctx.net.nodes[k].op else {
            continue;
        };
        let (lo, hi) = ctx.view.site(x);
        let hint = ctx.alpha.and_then(|a| a.get(&x));
        out[k] = lo
            .iter()
            .zip(hi)
            .enumerate()
            .map(|(i, (&l, &u))| {
                if l.is_nan() || u.is_nan() || l > u {
                    return if l > u {
                        Relax::Empty
                    } else {
                        Relax::LowerOnly { alpha: 0.0 }
                    };
                }
                if l >= 0.0 {
                    return Relax::On;
                }
                if u <= 0.0 {
                    return Relax::Off;
                }
                let alpha = hint
                    .and_then(|h| h.get(&i))
                    .copied()
                    .filter(|a| !a.is_nan())
                    .map_or_else(|| adaptive_alpha(l, u), |a| a.clamp(0.0, 1.0));
                match relu_chord(l, u) {
                    Some((s, t)) => Relax::Unst { alpha, s, t },
                    None => Relax::LowerOnly { alpha },
                }
            })
            .collect();
    }
    out
}

/// A tensor's accumulating rows for one chunk.
struct Acc {
    data: Vec<f64>,
    s: Vec<f64>,
    nterms: usize,
}

impl Acc {
    fn new(rows: usize, n: usize) -> Self {
        Self {
            data: vec![0.0; rows * n],
            s: vec![0.0; rows],
            nterms: 0,
        }
    }
}

fn acc_entry(lam: &mut HashMap<usize, Acc>, t: usize, rows: usize, n: usize) -> &mut Acc {
    lam.entry(t).or_insert_with(|| Acc::new(rows, n))
}

/// `out += lam · W_conv` (the transposed convolution of one row).
pub(crate) fn conv_back(c: &Conv, lam: &[f64], out: &mut [f64]) {
    let (ih, iw) = c.ihw;
    let (oh, ow) = c.ohw;
    let (sh, sw) = c.stride;
    let (pt, pl) = (c.pad.0 as isize, c.pad.1 as isize);
    for co in 0..c.co {
        for y in 0..oh {
            for x in 0..ow {
                let g = lam[(co * oh + y) * ow + x];
                if g == 0.0 {
                    continue;
                }
                for ci in 0..c.ci {
                    for ky in 0..c.kh {
                        let iy = (y * sh) as isize + ky as isize - pt;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        let wbase = ((co * c.ci + ci) * c.kh + ky) * c.kw;
                        let obase = (ci * ih + iy as usize) * iw;
                        for kx in 0..c.kw {
                            let ix = (x * sw) as isize + kx as isize - pl;
                            if ix < 0 || ix >= iw as isize {
                                continue;
                            }
                            out[obase + ix as usize] += g * c.w[wbase + kx];
                        }
                    }
                }
            }
        }
    }
}

fn conv_bias_down(c: &Conv, lam: &[f64]) -> f64 {
    let hw = c.ohw.0 * c.ohw.1;
    let mut s = 0.0f64;
    for co in 0..c.co {
        let b = c.b[co];
        for g in &lam[co * hw..(co + 1) * hw] {
            s = down(s + down(*g * b));
        }
    }
    s
}

/// `out += lam · W` (one row through a dense layer).
pub(crate) fn gemm_back(g: &Gemm, lam: &[f64], out: &mut [f64]) {
    for (j, &l) in lam.iter().enumerate() {
        if l == 0.0 {
            continue;
        }
        let row = &g.w[j * g.n_in..(j + 1) * g.n_in];
        for (o, w) in out.iter_mut().zip(row) {
            *o += l * *w;
        }
    }
}

/// Run one job over all seed rows (chunked, chunks in parallel).
#[must_use]
pub fn backsub(ctx: &Ctx<'_>, seeds: &Seeds) -> Vec<RowOut> {
    let n_rows = seeds.rows.len();
    if n_rows == 0 {
        return Vec::new();
    }
    let upto = ctx.net.topo(seeds.tensor);
    let relax = match upto {
        Some(k) => relaxations(ctx, k),
        None => vec![Vec::new(); ctx.net.nodes.len()],
    };
    // A certified empty interval anywhere in the cone certifies R_n = ∅.
    if relax.iter().flatten().any(|r| matches!(r, Relax::Empty)) {
        return (0..n_rows)
            .map(|_| RowOut {
                lb: f64::NAN,
                empty: true,
                lin: None,
            })
            .collect();
    }
    let mut beta_by: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for b in ctx.betas {
        let ok_topo = match (ctx.net.topo(b.tensor), upto) {
            // Only sites strictly upstream of the seed are processed.
            (Some(bt), Some(st)) => bt < st,
            (None, _) => true,
            (Some(_), None) => false,
        };
        let coef = if b.coef.is_finite() { b.coef } else { 0.0 };
        if ok_topo && coef != 0.0 && b.index < ctx.net.tensors[b.tensor].size {
            beta_by.entry(b.tensor).or_default().push((b.index, coef));
        }
    }
    let chunk = ctx.chunk_rows.max(1);
    let starts: Vec<usize> = (0..n_rows).step_by(chunk).collect();
    let parts: Vec<Vec<RowOut>> = starts
        .par_iter()
        .map(|&s| {
            let e = (s + chunk).min(n_rows);
            run_chunk(ctx, seeds, s, e, upto, &relax, &beta_by)
        })
        .collect();
    parts.into_iter().flatten().collect()
}

fn add_betas(
    ctx: &Ctx<'_>,
    t: usize,
    beta_by: &HashMap<usize, Vec<(usize, f64)>>,
    lam: &mut HashMap<usize, Acc>,
    rows: usize,
) {
    let Some(bs) = beta_by.get(&t) else {
        return;
    };
    let n = ctx.net.tensors[t].size;
    let m = &ctx.mags.m[t];
    let acc = acc_entry(lam, t, rows, n);
    let mut mag = 0.0f64;
    for &(i, coef) in bs {
        mag = add_up(mag, up(coef.abs() * m[i]));
    }
    for r in 0..rows {
        for &(i, coef) in bs {
            acc.data[r * n + i] += coef;
        }
        acc.s[r] = add_up(acc.s[r], mag);
    }
    acc.nterms += bs.len();
}

#[allow(clippy::too_many_lines)]
fn run_chunk(
    ctx: &Ctx<'_>,
    seeds: &Seeds,
    s: usize,
    e: usize,
    upto: Option<usize>,
    relax: &[Vec<Relax>],
    beta_by: &HashMap<usize, Vec<(usize, f64)>>,
) -> Vec<RowOut> {
    let net = ctx.net;
    let rows = e - s;
    let mut kappa: Vec<f64> = seeds.kappa0[s..e].to_vec();
    let mut eps = vec![0.0f64; rows];
    let mut empty = vec![false; rows];
    let mut failed = vec![false; rows];
    let mut lam: HashMap<usize, Acc> = HashMap::new();
    {
        let t = seeds.tensor;
        let n = net.tensors[t].size;
        let m = &ctx.mags.m[t];
        let acc = acc_entry(&mut lam, t, rows, n);
        for (r, row) in seeds.rows[s..e].iter().enumerate() {
            let mut mag = 0.0f64;
            for &(i, v) in row {
                if i < n {
                    acc.data[r * n + i] += v;
                    mag = add_up(mag, up(v.abs() * m[i]));
                } else {
                    failed[r] = true;
                }
            }
            acc.s[r] = mag;
        }
        acc.nterms = seeds.nterms.max(1);
    }
    if let Some(start) = upto {
        for k in (0..=start).rev() {
            let node = &net.nodes[k];
            let t = node.out;
            add_betas(ctx, t, beta_by, &mut lam, rows);
            let Some(acc) = lam.remove(&t) else {
                continue;
            };
            let n_t = net.tensors[t].size;
            for r in 0..rows {
                eps[r] = add_up(eps[r], charge(acc.nterms, acc.s[r], ctx.mags.msum[t]));
            }
            match &node.op {
                Op::Conv(c) => {
                    let n_x = net.tensors[c.x].size;
                    let q = &ctx.mags.q[k];
                    let fanin = ctx.mags.fanin[k];
                    let dst = acc_entry(&mut lam, c.x, rows, n_x);
                    for r in 0..rows {
                        let row = &acc.data[r * n_t..(r + 1) * n_t];
                        conv_back(c, row, &mut dst.data[r * n_x..(r + 1) * n_x]);
                        dst.s[r] = add_up(dst.s[r], dot_abs_up(row, q));
                        kappa[r] = add_down(kappa[r], conv_bias_down(c, row));
                    }
                    dst.nterms += fanin;
                }
                Op::Gemm(g) => {
                    let n_x = net.tensors[g.x].size;
                    let q = &ctx.mags.q[k];
                    let fanin = ctx.mags.fanin[k];
                    let dst = acc_entry(&mut lam, g.x, rows, n_x);
                    for r in 0..rows {
                        let row = &acc.data[r * n_t..(r + 1) * n_t];
                        gemm_back(g, row, &mut dst.data[r * n_x..(r + 1) * n_x]);
                        dst.s[r] = add_up(dst.s[r], dot_abs_up(row, q));
                        kappa[r] = add_down(kappa[r], dot_down(row, &g.b));
                    }
                    dst.nterms += fanin;
                }
                Op::Bn(b) | Op::Scale(b) => {
                    let m = &ctx.mags.m[b.x];
                    let dst = acc_entry(&mut lam, b.x, rows, n_t);
                    for r in 0..rows {
                        let row = &acc.data[r * n_t..(r + 1) * n_t];
                        let out = &mut dst.data[r * n_t..(r + 1) * n_t];
                        let mut mag = 0.0f64;
                        let mut rem = 0.0f64;
                        let mut kap = 0.0f64;
                        for i in 0..n_t {
                            let c = i / b.hw;
                            let l = row[i];
                            let v = l * b.s_mid[c];
                            out[i] += v;
                            mag = add_up(mag, up(v.abs() * m[i]));
                            rem = add_up(rem, up(up(l.abs() * b.s_rad[c]) * m[i]));
                            let sh = if l >= 0.0 { b.sh_lo[c] } else { b.sh_hi[c] };
                            kap = add_down(kap, down(l * sh));
                        }
                        dst.s[r] = add_up(dst.s[r], mag);
                        eps[r] = add_up(eps[r], rem);
                        kappa[r] = add_down(kappa[r], kap);
                    }
                    dst.nterms += 1;
                }
                Op::Select(sel) => {
                    // `v_T[k] = v_{xs[s]}[i]` exactly: each coefficient is
                    // scattered back unchanged (one exact term per routed
                    // element); an input element read by `f` outputs receives
                    // up to `f` terms, so its accumulation is charged with
                    // the source's largest fan-in.
                    for (x, fan) in sel.sources(&net.tensors) {
                        let n_x = net.tensors[x].size;
                        let m = &ctx.mags.m[x];
                        let dst = acc_entry(&mut lam, x, rows, n_x);
                        for r in 0..rows {
                            let row = &acc.data[r * n_t..(r + 1) * n_t];
                            let out = &mut dst.data[r * n_x..(r + 1) * n_x];
                            let mut mag = 0.0f64;
                            for (k, &(s, i)) in sel.src.iter().enumerate() {
                                if sel.xs[s] != x {
                                    continue;
                                }
                                let v = row[k];
                                out[i] += v;
                                mag = add_up(mag, up(v.abs() * m[i]));
                            }
                            dst.s[r] = add_up(dst.s[r], mag);
                        }
                        dst.nterms += fan;
                    }
                }
                Op::Add { a, b } => {
                    for &x in &[*a, *b] {
                        let m = &ctx.mags.m[x];
                        let dst = acc_entry(&mut lam, x, rows, n_t);
                        for r in 0..rows {
                            let row = &acc.data[r * n_t..(r + 1) * n_t];
                            let out = &mut dst.data[r * n_t..(r + 1) * n_t];
                            for (o, v) in out.iter_mut().zip(row) {
                                *o += *v;
                            }
                            dst.s[r] = add_up(dst.s[r], dot_abs_up(row, m));
                        }
                        dst.nterms += 1;
                    }
                }
                Op::AddConst { x, c } => {
                    let m = &ctx.mags.m[*x];
                    let dst = acc_entry(&mut lam, *x, rows, n_t);
                    for r in 0..rows {
                        let row = &acc.data[r * n_t..(r + 1) * n_t];
                        let out = &mut dst.data[r * n_t..(r + 1) * n_t];
                        for (o, v) in out.iter_mut().zip(row) {
                            *o += *v;
                        }
                        dst.s[r] = add_up(dst.s[r], dot_abs_up(row, m));
                        kappa[r] = add_down(kappa[r], dot_down(row, c));
                    }
                    dst.nterms += 1;
                }
                Op::Identity { x } => {
                    let m = &ctx.mags.m[*x];
                    let dst = acc_entry(&mut lam, *x, rows, n_t);
                    for r in 0..rows {
                        let row = &acc.data[r * n_t..(r + 1) * n_t];
                        let out = &mut dst.data[r * n_t..(r + 1) * n_t];
                        for (o, v) in out.iter_mut().zip(row) {
                            *o += *v;
                        }
                        dst.s[r] = add_up(dst.s[r], dot_abs_up(row, m));
                    }
                    dst.nterms += 1;
                }
                Op::Relu { x } => {
                    let m = &ctx.mags.m[*x];
                    let rel = &relax[k];
                    let dst = acc_entry(&mut lam, *x, rows, n_t);
                    for r in 0..rows {
                        let row = &acc.data[r * n_t..(r + 1) * n_t];
                        let out = &mut dst.data[r * n_t..(r + 1) * n_t];
                        let mut mag = 0.0f64;
                        let mut kap = 0.0f64;
                        for i in 0..n_t {
                            let l = row[i];
                            if l == 0.0 {
                                if matches!(rel[i], Relax::Empty) {
                                    empty[r] = true;
                                }
                                continue;
                            }
                            let v = match rel[i] {
                                Relax::Off => continue,
                                Relax::On => l,
                                Relax::Unst { alpha, s, t } => {
                                    if l >= 0.0 {
                                        l * alpha
                                    } else {
                                        kap = add_down(kap, down(l * t));
                                        l * s
                                    }
                                }
                                Relax::LowerOnly { alpha } => {
                                    if l >= 0.0 {
                                        l * alpha
                                    } else {
                                        failed[r] = true;
                                        0.0
                                    }
                                }
                                Relax::Empty => {
                                    empty[r] = true;
                                    continue;
                                }
                            };
                            out[i] += v;
                            mag = add_up(mag, up(v.abs() * m[i]));
                        }
                        dst.s[r] = add_up(dst.s[r], mag);
                        kappa[r] = add_down(kappa[r], kap);
                    }
                    dst.nterms += 1;
                }
            }
        }
    }
    // The input tensor.
    let t_in = net.input;
    add_betas(ctx, t_in, beta_by, &mut lam, rows);
    let n_in = net.tensors[t_in].size;
    let acc = lam.remove(&t_in).unwrap_or_else(|| Acc::new(rows, n_in));
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let e_r = add_up(
            eps[r],
            charge(acc.nterms.max(1), acc.s[r], ctx.mags.msum[t_in]),
        );
        let row = &acc.data[r * n_in..(r + 1) * n_in];
        let konst = add_down(kappa[r], -e_r);
        let conc = concretize_down(row, ctx.box_lo, ctx.box_hi);
        let mut lb = add_down(konst, conc);
        if failed[r] || !lb.is_finite() || !konst.is_finite() || !e_r.is_finite() {
            lb = f64::NAN;
        }
        out.push(RowOut {
            lb,
            empty: empty[r] && !failed[r],
            // review(b2-lanes): a FAILED row (a `LowerOnly` Relu met a
            // negative λ and its term was dropped) certifies nothing, so its
            // form is not `objective(x) >= lin·x + c` either. B2.2's clip
            // consumes forms as certified half-spaces: hand it none here.
            lin: (ctx.want_lin && !failed[r]).then(|| {
                (
                    row.to_vec(),
                    if konst.is_finite() { konst } else { f64::NAN },
                )
            }),
        });
    }
    out
}

/// A lower bound on the linear form `lin·x + c` at the point `x` (every
/// product and sum rounded down).
#[must_use]
pub fn eval_lin_down(lin: &[f64], c: f64, x: &[f64]) -> f64 {
    add_down(dot_down(lin, x), c)
}
