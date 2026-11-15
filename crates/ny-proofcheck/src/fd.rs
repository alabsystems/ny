// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.3: the checker's own FROZEN-DUAL prune evaluation — the `fd` rung of
//! the discharge ladder (ENGINE_PROGRAM §1.7 B2, `fd-check`).
//!
//! The producer's frozen-dual tier (`zono_state` + `zono_bab`) closes a class
//! by a level-synchronous BaB whose every leaf is certified by a Lagrangian
//! bound `g(L)` over a frozen alpha-zonotope state. Its bundle carries, per
//! prune, the split literals (absolute Relu sites in branch order), the class
//! it discharges and the dual point `(L0, L1)` it was certified at, plus one
//! `hint.fd_state` / `hint.fd_lambda` record per state (identity, split
//! order, slopes). NOTHING of the producer's state is trusted: this module
//! rebuilds its OWN state from the checker's lowered network and root bounds
//! and evaluates `g` at the hinted point with its own directed rounding. By
//! weak duality every `L >= 0` gives a valid bound, so a wrong hint (slopes,
//! multipliers, sites) can only fail to confirm.
//!
//! # The state (exact real arithmetic)
//!
//! Over a region box `B` with root bounds `[l, u]` on every Relu input
//! (`crate::check`'s root bounds, valid on all of `B`), the generator vector
//! `e(x)` has:
//!
//! * one INPUT symbol per non-degenerate input dimension,
//!   `e_i = (x_i - mid_i) / rad_i`, with `mid - rad <= lo` and
//!   `mid + rad >= hi` exactly (so `e_i ∈ [-1, 1]`);
//! * one RELAXATION symbol per unstable neuron (`l < 0 < u`),
//!   `e_k = (relu(z_k) - λ_k z_k - μ_k) / μ_k` for a slope `λ_k ∈ [0, 1]`
//!   (the hint's, else the chord `u / (u - l)`) and
//!   `μ_k >= max((1 - λ_k) u_k, -λ_k l_k) / 2` (rounded up), so
//!   `e_k ∈ [-1, 1]` because `relu(z) - λ z ∈ [0, 2μ]` on `[l, u]`.
//!
//! The slope table (`#fd-compact-checkable`). A producer's #breakpoint-dual
//! RANKED compact state keeps rows only for its retained split candidates,
//! but every unstable neuron keeps its relaxation symbol at the class slope,
//! so its `hint.fd_lambda` table carries the slope of EVERY unstable neuron
//! and `hint.fd_state.relaxed` counts them. The checker consumes the whole
//! table (a neuron the table does not name falls back to the chord; the
//! report counts those as `chord_lambda`), and a state whose table does not
//! hold exactly its declared `relaxed` entries is refused
//! (`fd_table_count`, `crate::check`): a declared table's hole is never
//! filled silently. None of this is trusted for a bound: whatever slopes the
//! table holds, the relaxation below is an identity on `B`.
//!
//! Substituting `relu(z) = λ z + μ + μ e` (unstable), `relu(z) = z` (stable
//! on) and `0` (stable off) is an IDENTITY on `B`, so every tensor is exactly
//! affine in `e(x)`. A generator-space back-substitution ([`forms`]) computes,
//! for an objective `C y - d` and for any Relu input `z_s`,
//!
//! ```text
//!   v(x) = K + g . e(x) + δ(x),   K ∈ [c_lo, c_hi],   |δ(x)| <= err
//! ```
//!
//! with `g` the stored floats. Every backward step is the checker's own
//! back-substitution (`crate::backsub`'s accumulation charges, the BN
//! interval parameters, interval bias dots), with the relaxation identity in
//! place of the α/chord inequality: `l h = (l λ) z + l μ + (l μ) e` — the
//! product `l λ` is charged like any accumulated term, `l μ` enters `K` as an
//! outward interval, and the generator coefficient `fl(l μ)` carries its own
//! rounding into `err` (`|e| <= 1`). At the input `Λ x = Λ·mid + (Λ ∘ rad)·e`
//! the same way. The result is TWO-sided, which the rows need.
//!
//! # The prune bound (the producer's Lagrangian, re-derived)
//!
//! A node fixes the side of the Relu inputs `z_i` of its literals (`Off`:
//! `z <= 0`, sign `s = +1`; `On`: `z >= 0`, `s = -1`). With the objective
//! form `(c_obj = c_lo, d = g, r_obj = err)` and each row
//! `z_i = c_i + a_i . e + η_i`, `|η_i| <= r_i` (`c_i = c_lo`,
//! `r_i = err + (c_hi - c_lo)`):
//!
//! * a row whose neuron owns a symbol `γ` satisfies, on the node,
//!   `e_γ = (slope_i z_i - μ_i) / μ_i` with `slope = -λ` (Off) or `1 - λ`
//!   (On), so `d_γ e_γ = ρ_i z_i - d_γ`, `ρ_i = d_γ slope_i / μ_i` — an
//!   exact identity, evaluated here as an OUTWARD interval for `ρ_i`;
//! * the cuts `s z_i <= 0` (`s a_i . e <= -s c_i + r_i = b0_i`) and, for a
//!   symbol row, `e_γ <= 1` (`-s a_i . e <= q_i + s c_i + r_i`, with
//!   `q = 2μ/λ` Off, `2μ/(1-λ)` On, when the denominator is positive),
//!   capped by the always-valid `Σ_j |a_ij|` (`b1_i`);
//! * for multipliers `m0, m1 >= 0` (the hint's, clamped; NaN is `0`) and the
//!   margin-only lower bound `obj >= c_obj - r_obj + d . e`,
//!
//! ```text
//!   obj >= c_obj - r_obj - Σ d_γ + Σ ρ_i c_i - Σ |ρ_i| r_i
//!          - Σ_j |rc_j| - Σ_i (m0_i b0_i + m1_i b1_i),
//!   rc  =  d_base + Σ_i (ρ_i + s_i (m0_i - m1_i)) a_i
//! ```
//!
//! (`d_base` is `d` with every substituted symbol column zeroed). Every term
//! is evaluated as an interval with one-ulp outward steps after EACH
//! correctly rounded operation ([`crate::arith`]): `ρ`, the multipliers and
//! `rc_j` are intervals, `|rc_j|` takes the larger end, and the result is
//! the lower end. A discharge is confirmed only on `g > 0` (strict).
//!
//! Rows of stable neurons (in the checker's box) substitute nothing
//! (`ρ = 0`) and keep only their sign cut; input-threshold literals are
//! ignored (a bound over a larger region is a bound over the node). A node
//! naming one symbol twice refuses.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::arith::{add_down, add_up, charge, dot_abs_up, down, mul_up, up, Iv};
use crate::backsub::{conv_back, gemm_back};
use crate::bundle::sha256_hex;
use crate::ibp::{Bounds, Magnitudes};
use crate::lower::{Conv, Net, Op};

/// Hint slopes of one FD state: site tensor -> neuron -> λ (clamped to
/// `[0, 1]` by the parser; untrusted).
pub type LambdaMap = HashMap<usize, HashMap<usize, f64>>;

/// How one Relu neuron is relaxed in the checker's frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Relu {
    /// Stable inactive on the region (`u <= 0`): `relu(z) = 0`.
    Off,
    /// Stable active on the region (`l >= 0`): `relu(z) = z`.
    On,
    /// Unstable: owns generator column `col`, slope `lam ∈ [0, 1]` and
    /// certified `mu > 0`.
    Sym {
        /// Generator column.
        col: usize,
        /// Slope.
        lam: f64,
        /// Certified half-height.
        mu: f64,
    },
    /// Non-finite bounds: no relaxation (any row through it fails).
    Bad,
}

/// The checker's generator layout over one region.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// Generator count.
    pub n_gens: usize,
    /// Input symbols.
    pub n_in_gens: usize,
    /// Per input element: its symbol column (`None`: a degenerate dimension).
    pub in_col: Vec<Option<usize>>,
    /// Input centres.
    pub mid: Vec<f64>,
    /// Input radii (`0` on a degenerate dimension).
    pub rad: Vec<f64>,
    /// Per tensor: the relaxation of every neuron (empty for non-sites).
    pub relax: Vec<Vec<Relu>>,
    /// Hint slopes actually used (neurons unstable here with a hint).
    pub hinted: usize,
    /// `#fd-compact-checkable`: neurons unstable here that NO hint slope
    /// names, relaxed at the chord default instead. A producer's complete
    /// table (a full state's rows, or a ranked compact state's class
    /// relaxation) leaves this at the count of neurons the PRODUCER saw
    /// stable but the checker's own (looser or different) root bounds do
    /// not: those are the only symbols whose relaxation the two sides chose
    /// differently. Diagnostic only; nothing is confirmed by it.
    pub chord: usize,
    /// `#fd-compact-checkable`: hint entries that relax nothing here (the
    /// neuron is stable or unbounded in the checker's box, or the entry is
    /// not finite). Diagnostic only.
    pub hint_unused: usize,
}

/// `μ >= max((1 - λ) u, -λ l) / 2` with upward rounding at every step.
#[must_use]
pub fn mu_up(lam: f64, l: f64, u: f64) -> f64 {
    let v1 = mul_up(add_up(1.0, -lam), u);
    let v2 = mul_up(-lam, l);
    up(0.5 * v1.max(v2))
}

/// The chord slope `u / (u - l)` clamped to `[0, 1]` (any value in `[0, 1]`
/// is sound; the chord is the default when no hint names the neuron).
fn chord(l: f64, u: f64) -> f64 {
    let s = u / (u - l);
    if s.is_nan() {
        1.0
    } else {
        s.clamp(0.0, 1.0)
    }
}

impl Frame {
    /// The frame of the region box `[box_lo, box_hi]` with root bounds
    /// `bounds` (valid on the whole box) and hint slopes `lambda`.
    ///
    /// # Errors
    /// A non-finite or crossed input box.
    pub fn new(
        net: &Net,
        bounds: &Bounds,
        box_lo: &[f64],
        box_hi: &[f64],
        lambda: Option<&LambdaMap>,
    ) -> Result<Self, String> {
        let n_in = net.n_in();
        if box_lo.len() != n_in || box_hi.len() != n_in {
            return Err("fd frame: box width".into());
        }
        let mut in_col = Vec::with_capacity(n_in);
        let mut mid = Vec::with_capacity(n_in);
        let mut rad = Vec::with_capacity(n_in);
        let mut col = 0usize;
        for (&lo, &hi) in box_lo.iter().zip(box_hi) {
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err("fd frame: non-finite or crossed input box".into());
            }
            if lo == hi {
                in_col.push(None);
                mid.push(lo);
                rad.push(0.0);
                continue;
            }
            let m = (0.5 * lo + 0.5 * hi).clamp(lo, hi);
            // `add_up(a, b) >= a + b` exactly, so `m - r <= lo` and
            // `m + r >= hi` hold in real arithmetic.
            let r = add_up(hi, -m).max(add_up(m, -lo));
            if !r.is_finite() || r <= 0.0 {
                return Err("fd frame: input radius".into());
            }
            in_col.push(Some(col));
            col += 1;
            mid.push(m);
            rad.push(r);
        }
        let n_in_gens = col;
        let mut relax = vec![Vec::new(); net.tensors.len()];
        let mut hinted = 0usize;
        let mut chord_used = 0usize;
        for t in 0..net.tensors.len() {
            if !net.is_site(t) {
                continue;
            }
            let (lo, hi) = (&bounds.lo[t], &bounds.hi[t]);
            if lo.len() != net.tensors[t].size || hi.len() != lo.len() {
                return Err("fd frame: root bounds missing for a site".into());
            }
            let hint = lambda.and_then(|m| m.get(&t));
            relax[t] = lo
                .iter()
                .zip(hi)
                .enumerate()
                .map(|(i, (&l, &u))| {
                    if !l.is_finite() || !u.is_finite() || l > u {
                        return Relu::Bad;
                    }
                    if l >= 0.0 {
                        return Relu::On;
                    }
                    if u <= 0.0 {
                        return Relu::Off;
                    }
                    let lam = match hint.and_then(|h| h.get(&i)).copied() {
                        Some(v) if v.is_finite() => {
                            hinted += 1;
                            v.clamp(0.0, 1.0)
                        }
                        _ => {
                            chord_used += 1;
                            chord(l, u)
                        }
                    };
                    let mu = mu_up(lam, l, u);
                    if !(mu.is_finite() && mu > 0.0) {
                        return Relu::Bad;
                    }
                    let r = Relu::Sym { col, lam, mu };
                    col += 1;
                    r
                })
                .collect();
        }
        // Every hint entry names one neuron; the ones not counted as used
        // relaxed nothing here (a `Bad` neuron's hint included).
        let entries: usize = lambda.map_or(0, |m| m.values().map(HashMap::len).sum());
        Ok(Self {
            n_gens: col,
            n_in_gens,
            in_col,
            mid,
            rad,
            relax,
            hinted,
            chord: chord_used,
            hint_unused: entries.saturating_sub(hinted),
        })
    }

    /// Relaxation of neuron `i` of site tensor `t` (`Bad` when out of range).
    #[must_use]
    pub fn relu(&self, t: usize, i: usize) -> Relu {
        self.relax
            .get(t)
            .and_then(|r| r.get(i))
            .copied()
            .unwrap_or(Relu::Bad)
    }

    /// sha256 of the layout and every relaxation (the report's identity of
    /// the checker's OWN state; it shares nothing with the producer's hash).
    #[must_use]
    pub fn digest(&self) -> String {
        let mut b: Vec<u8> = Vec::new();
        let mut put = |v: u64| b.extend_from_slice(&v.to_le_bytes());
        put(self.n_gens as u64);
        put(self.n_in_gens as u64);
        for (c, (m, r)) in self.in_col.iter().zip(self.mid.iter().zip(&self.rad)) {
            put(c.map_or(u64::MAX, |c| c as u64));
            put(m.to_bits());
            put(r.to_bits());
        }
        for (t, rs) in self.relax.iter().enumerate() {
            if rs.is_empty() {
                continue;
            }
            put(t as u64);
            for r in rs {
                match *r {
                    Relu::Off => put(0),
                    Relu::On => put(1),
                    Relu::Bad => put(2),
                    Relu::Sym { col, lam, mu } => {
                        put(3);
                        put(col as u64);
                        put(lam.to_bits());
                        put(mu.to_bits());
                    }
                }
            }
        }
        sha256_hex(&b)
    }
}

/// An affine form over the generators: `v(x) = K + g . e(x) + δ(x)`,
/// `K ∈ [c_lo, c_hi]`, `|δ| <= err`, for every `x` in the region.
#[derive(Debug, Clone, PartialEq)]
pub struct Form {
    /// Lower end of the constant.
    pub c_lo: f64,
    /// Upper end of the constant.
    pub c_hi: f64,
    /// Generator coefficients (`n_gens`).
    pub g: Vec<f64>,
    /// Remainder bound.
    pub err: f64,
    /// False when a step could not be certified (a `Bad` neuron, a
    /// non-finite value): the form certifies nothing.
    pub ok: bool,
}

/// One seed: `Σ coeffs · v_tensor + k0`.
#[derive(Debug, Clone, PartialEq)]
pub struct Seed {
    /// The tensor the seed is expressed over.
    pub tensor: usize,
    /// Sparse coefficients.
    pub coeffs: Vec<(usize, f64)>,
    /// Constant (an exact enclosure).
    pub k0: Iv,
}

struct Acc {
    data: Vec<f64>,
    s: f64,
    nterms: usize,
}

fn acc(lam: &mut HashMap<usize, Acc>, t: usize, n: usize) -> &mut Acc {
    lam.entry(t).or_insert_with(|| Acc {
        data: vec![0.0; n],
        s: 0.0,
        nterms: 0,
    })
}

/// An outward enclosure of `Σ row_i b_i`.
fn dot_iv(row: &[f64], b: &[f64]) -> Iv {
    let mut lo = 0.0f64;
    let mut hi = 0.0f64;
    for (x, y) in row.iter().zip(b) {
        let p = *x * *y;
        lo = down(lo + down(p));
        hi = up(hi + up(p));
    }
    Iv::new(lo, hi)
}

/// An outward enclosure of the bias term of a Conv's backward row.
fn conv_bias_iv(c: &Conv, row: &[f64]) -> Iv {
    let hw = c.ohw.0 * c.ohw.1;
    let mut lo = 0.0f64;
    let mut hi = 0.0f64;
    for co in 0..c.co {
        let b = c.b[co];
        for g in &row[co * hw..(co + 1) * hw] {
            let p = *g * b;
            lo = down(lo + down(p));
            hi = up(hi + up(p));
        }
    }
    Iv::new(lo, hi)
}

/// The rounding of one stored generator coefficient `fl(p)` of an exact
/// product `p`: `down(fl) <= p <= up(fl)`, so `|p - fl| <= up(fl) - down(fl)`.
fn coef_err(g: f64) -> f64 {
    up(up(g) - down(g))
}

/// One seed's generator-space form (see the module docs for the invariant).
#[allow(clippy::too_many_lines)]
fn form_of(net: &Net, mags: &Magnitudes, frame: &Frame, seed: &Seed) -> Form {
    let n_gens = frame.n_gens;
    let mut g = vec![0.0f64; n_gens];
    let mut set = vec![false; n_gens];
    let mut k = seed.k0;
    let mut err = 0.0f64;
    let mut ok = seed.k0.is_finite();
    let mut lam: HashMap<usize, Acc> = HashMap::new();
    let t0 = seed.tensor;
    if t0 >= net.tensors.len() {
        return Form {
            c_lo: f64::NAN,
            c_hi: f64::NAN,
            g,
            err: f64::NAN,
            ok: false,
        };
    }
    {
        let n0 = net.tensors[t0].size;
        let m = &mags.m[t0];
        let a = acc(&mut lam, t0, n0);
        for &(i, v) in &seed.coeffs {
            if i >= n0 || !v.is_finite() {
                ok = false;
                continue;
            }
            a.data[i] += v;
            a.s = add_up(a.s, up(v.abs() * m[i]));
        }
        a.nterms = 1;
    }
    let mut put_g = |col: usize, v: f64, err: &mut f64, ok: &mut bool| {
        if col >= n_gens || set[col] {
            // A symbol is introduced by exactly one neuron, met once.
            *ok = false;
            return;
        }
        set[col] = true;
        g[col] = v;
        *err = add_up(*err, coef_err(v));
    };
    if let Some(start) = net.topo(t0) {
        for kk in (0..=start).rev() {
            let node = &net.nodes[kk];
            let t = node.out;
            let Some(a) = lam.remove(&t) else {
                continue;
            };
            err = add_up(err, charge(a.nterms.max(1), a.s, mags.msum[t]));
            let n_t = net.tensors[t].size;
            match &node.op {
                Op::Conv(c) => {
                    let dst = acc(&mut lam, c.x, net.tensors[c.x].size);
                    conv_back(c, &a.data, &mut dst.data);
                    dst.s = add_up(dst.s, dot_abs_up(&a.data, &mags.q[kk]));
                    dst.nterms += mags.fanin[kk];
                    k += conv_bias_iv(c, &a.data);
                }
                Op::Gemm(gm) => {
                    let dst = acc(&mut lam, gm.x, net.tensors[gm.x].size);
                    gemm_back(gm, &a.data, &mut dst.data);
                    dst.s = add_up(dst.s, dot_abs_up(&a.data, &mags.q[kk]));
                    dst.nterms += mags.fanin[kk];
                    k += dot_iv(&a.data, &gm.b);
                }
                Op::Bn(b) | Op::Scale(b) => {
                    let m = &mags.m[b.x];
                    let dst = acc(&mut lam, b.x, n_t);
                    let mut mag = 0.0f64;
                    let mut rem = 0.0f64;
                    let mut kap = Iv::point(0.0);
                    for i in 0..n_t {
                        let c = i / b.hw;
                        let l = a.data[i];
                        if l == 0.0 {
                            continue;
                        }
                        let v = l * b.s_mid[c];
                        dst.data[i] += v;
                        mag = add_up(mag, up(v.abs() * m[i]));
                        rem = add_up(rem, up(up(l.abs() * b.s_rad[c]) * m[i]));
                        let (p, q) = (l * b.sh_lo[c], l * b.sh_hi[c]);
                        kap += Iv::new(down(p.min(q)), up(p.max(q)));
                    }
                    dst.s = add_up(dst.s, mag);
                    dst.nterms += 1;
                    err = add_up(err, rem);
                    k += kap;
                }
                Op::Select(sel) => {
                    // Exact routing: the coefficient of output element `k`
                    // moves unchanged to its source element (the same
                    // charges as `crate::backsub`).
                    for (x, fan) in sel.sources(&net.tensors) {
                        let m = &mags.m[x];
                        let dst = acc(&mut lam, x, net.tensors[x].size);
                        let mut mag = 0.0f64;
                        for (kk2, &(s, i)) in sel.src.iter().enumerate() {
                            if sel.xs[s] != x {
                                continue;
                            }
                            let v = a.data[kk2];
                            dst.data[i] += v;
                            mag = add_up(mag, up(v.abs() * m[i]));
                        }
                        dst.s = add_up(dst.s, mag);
                        dst.nterms += fan;
                    }
                }
                Op::Add { a: x, b: y } => {
                    for &src in &[*x, *y] {
                        let m = &mags.m[src];
                        let dst = acc(&mut lam, src, n_t);
                        for (o, v) in dst.data.iter_mut().zip(&a.data) {
                            *o += *v;
                        }
                        dst.s = add_up(dst.s, dot_abs_up(&a.data, m));
                        dst.nterms += 1;
                    }
                }
                Op::AddConst { x, c } => {
                    let m = &mags.m[*x];
                    let dst = acc(&mut lam, *x, n_t);
                    for (o, v) in dst.data.iter_mut().zip(&a.data) {
                        *o += *v;
                    }
                    dst.s = add_up(dst.s, dot_abs_up(&a.data, m));
                    dst.nterms += 1;
                    k += dot_iv(&a.data, c);
                }
                Op::Identity { x } => {
                    let m = &mags.m[*x];
                    let dst = acc(&mut lam, *x, n_t);
                    for (o, v) in dst.data.iter_mut().zip(&a.data) {
                        *o += *v;
                    }
                    dst.s = add_up(dst.s, dot_abs_up(&a.data, m));
                    dst.nterms += 1;
                }
                Op::Relu { x } => {
                    let m = &mags.m[*x];
                    let rel = &frame.relax[*x];
                    let dst = acc(&mut lam, *x, n_t);
                    let mut mag = 0.0f64;
                    for i in 0..n_t {
                        let l = a.data[i];
                        if l == 0.0 {
                            continue;
                        }
                        match rel.get(i).copied().unwrap_or(Relu::Bad) {
                            Relu::Off => {}
                            Relu::On => {
                                dst.data[i] += l;
                                mag = add_up(mag, up(l.abs() * m[i]));
                            }
                            Relu::Sym { col, lam: sl, mu } => {
                                // l h = (l λ) z + l μ + (l μ) e, exactly.
                                let v = l * sl;
                                dst.data[i] += v;
                                mag = add_up(mag, up(v.abs() * m[i]));
                                k += Iv::point(mu).scale(l);
                                put_g(col, l * mu, &mut err, &mut ok);
                            }
                            Relu::Bad => ok = false,
                        }
                    }
                    dst.s = add_up(dst.s, mag);
                    dst.nterms += 1;
                }
            }
        }
    }
    // The input: Λ x = Λ·mid + (Λ ∘ rad)·e_in (degenerate dims are exact).
    let t_in = net.input;
    if let Some(a) = lam.remove(&t_in) {
        err = add_up(err, charge(a.nterms.max(1), a.s, mags.msum[t_in]));
        for (i, &l) in a.data.iter().enumerate() {
            if l == 0.0 {
                continue;
            }
            k += Iv::point(frame.mid[i]).scale(l);
            if let Some(col) = frame.in_col[i] {
                put_g(col, l * frame.rad[i], &mut err, &mut ok);
            }
        }
    }
    if !lam.is_empty() {
        // A live row the sweep never reached (a tensor with no producer
        // that is not the input): nothing certifies it.
        ok = false;
    }
    if !(k.lo.is_finite() && k.hi.is_finite() && err.is_finite())
        || g.iter().any(|v| !v.is_finite())
    {
        ok = false;
    }
    Form {
        c_lo: k.lo,
        c_hi: k.hi,
        g,
        err,
        ok,
    }
}

/// The forms of `seeds` (parallel across seeds; each is summed in a fixed
/// order, so the result does not depend on the thread count).
#[must_use]
pub fn forms(net: &Net, mags: &Magnitudes, frame: &Frame, seeds: &[Seed]) -> Vec<Form> {
    seeds
        .par_iter()
        .map(|s| form_of(net, mags, frame, s))
        .collect()
}

/// The seed of Relu input `z_{t,i}`.
#[must_use]
pub fn site_seed(t: usize, i: usize) -> Seed {
    Seed {
        tensor: t,
        coeffs: vec![(i, 1.0)],
        k0: Iv::point(0.0),
    }
}

/// One literal row of a node: the row's form, the neuron's relaxation and
/// its side (`on`: `z >= 0`).
#[derive(Debug, Clone, Copy)]
pub struct RowRef<'a> {
    /// The row `z = K + a . e + δ`.
    pub form: &'a Form,
    /// The neuron's relaxation in the frame.
    pub relax: Relu,
    /// `z >= 0` (else `z <= 0`).
    pub on: bool,
}

/// A hint multiplier as a dual point: NaN is `0`, a negative value clamps
/// to `0`, an infinite one refuses.
fn dual(v: f64) -> Option<f64> {
    if v.is_nan() {
        Some(0.0)
    } else if v.is_infinite() {
        None
    } else {
        Some(v.max(0.0))
    }
}

/// The certified prune bound `g(L)` of the objective form `margin` over the
/// node whose literal rows are `rows`, at the dual point `(l0, l1)` aligned
/// with `rows` (see the module docs). NaN when nothing can be certified
/// (a failed form, a length mismatch, a symbol named twice, an infinite
/// multiplier or a non-finite result); confirm only on `> 0`.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn node_bound(margin: &Form, rows: &[RowRef<'_>], l0: &[f64], l1: &[f64]) -> f64 {
    let nan = f64::NAN;
    if !margin.ok || l0.len() != rows.len() || l1.len() != rows.len() {
        return nan;
    }
    let n = margin.g.len();
    let mut subst: Vec<usize> = Vec::new();
    let mut mult: Vec<Iv> = Vec::with_capacity(rows.len());
    // c_obj - r_obj, then every substituted constant.
    let mut klo = add_down(margin.c_lo, -margin.err);
    let mut lag = 0.0f64;
    for (i, row) in rows.iter().enumerate() {
        let f = row.form;
        if !f.ok || f.g.len() != n {
            return nan;
        }
        let (Some(m0), Some(m1)) = (dual(l0[i]), dual(l1[i])) else {
            return nan;
        };
        let s = if row.on { -1.0 } else { 1.0 };
        let c = f.c_lo;
        let r = add_up(f.err, add_up(f.c_hi, -f.c_lo));
        let mut row_abs = 0.0f64;
        for v in &f.g {
            row_abs = add_up(row_abs, v.abs());
        }
        let (rho, q) = match row.relax {
            Relu::Sym { col, lam, mu } => {
                if col >= n || subst.contains(&col) {
                    return nan;
                }
                subst.push(col);
                let dg = margin.g[col];
                let slope = if row.on {
                    let one_m = 1.0 - lam;
                    Iv::new(down(one_m), up(one_m))
                } else {
                    Iv::point(-lam)
                };
                let num = slope.scale(dg);
                let rho = Iv::new(down(num.lo / mu), up(num.hi / mu));
                // - d_γ (exact negation).
                klo = add_down(klo, -dg);
                let q = if row.on {
                    let den = down(1.0 - lam);
                    (den > 0.0).then(|| up(up(2.0 * mu) / den))
                } else {
                    (lam > 0.0).then(|| up(up(2.0 * mu) / lam))
                };
                (rho, q)
            }
            Relu::On | Relu::Off => (Iv::point(0.0), None),
            Relu::Bad => return nan,
        };
        // + ρ c - |ρ| r
        klo = add_down(klo, rho.scale(c).lo);
        klo = add_down(klo, -mul_up(rho.lo.abs().max(rho.hi.abs()), r));
        // Cuts: s a.e <= b0 (the sign cut), -s a.e <= b1 (the far cut).
        let b0 = add_up(-s * c, r);
        let b1 = match q {
            Some(q) => add_up(add_up(q, s * c), r).min(row_abs),
            None => row_abs,
        };
        lag = add_up(lag, add_up(mul_up(m0, b0), mul_up(m1, b1)));
        mult.push(rho + (Iv::point(m0) - Iv::point(m1)).scale(s));
    }
    let mut lo = margin.g.clone();
    let mut hi = margin.g.clone();
    for &col in &subst {
        lo[col] = 0.0;
        hi[col] = 0.0;
    }
    for (row, m) in rows.iter().zip(&mult) {
        for (j, &a) in row.form.g.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let t = m.scale(a);
            lo[j] = down(lo[j] + t.lo);
            hi[j] = up(hi[j] + t.hi);
        }
    }
    let mut sumabs = 0.0f64;
    for (l, h) in lo.iter().zip(&hi) {
        sumabs = add_up(sumabs, l.abs().max(h.abs()));
    }
    let g = add_down(add_down(klo, -sumabs), -lag);
    if g.is_finite() {
        g
    } else {
        nan
    }
}

/// The generator values at a point, as outward intervals, from the point's
/// rigorous enclosure `b` (`crate::ibp::eval_point`): input symbols from
/// `x`, relaxation symbols from their Relu input's enclosure.
#[must_use]
pub fn symbols_at(net: &Net, frame: &Frame, x: &[f64], b: &Bounds) -> Vec<Iv> {
    let mut e = vec![Iv::new(f64::NAN, f64::NAN); frame.n_gens];
    for (i, c) in frame.in_col.iter().enumerate() {
        if let Some(col) = *c {
            let num = Iv::point(x[i]) - Iv::point(frame.mid[i]);
            let r = frame.rad[i];
            e[col] = Iv::new(down(num.lo / r), up(num.hi / r));
        }
    }
    for t in 0..net.tensors.len() {
        for (i, rel) in frame.relax[t].iter().enumerate() {
            let Relu::Sym { col, lam, mu } = *rel else {
                continue;
            };
            let (zl, zh) = (b.lo[t][i], b.hi[t][i]);
            // f(z) = relu(z) - λ z, piecewise linear: its range over
            // [zl, zh] is the hull of the ends and 0 (when 0 is inside).
            let f = |v: f64| -> Iv {
                if v <= 0.0 {
                    Iv::point(v).scale(-lam)
                } else {
                    let one_m = Iv::new(down(1.0 - lam), up(1.0 - lam));
                    one_m * Iv::point(v)
                }
            };
            let mut h = f(zl);
            let fh = f(zh);
            h = Iv::new(h.lo.min(fh.lo), h.hi.max(fh.hi));
            if zl < 0.0 && zh > 0.0 {
                h = Iv::new(h.lo.min(0.0), h.hi.max(0.0));
            }
            let num = h - Iv::point(mu);
            e[col] = Iv::new(down(num.lo / mu), up(num.hi / mu));
        }
    }
    e
}

/// The form's value at a point as an outward interval (`K + g . e ± err`).
#[must_use]
pub fn form_at(form: &Form, e: &[Iv]) -> Iv {
    let mut v = Iv::new(form.c_lo, form.c_hi);
    for (gj, ej) in form.g.iter().zip(e) {
        if *gj != 0.0 {
            v += ej.scale(*gj);
        }
    }
    Iv::new(down(v.lo - form.err), up(v.hi + form.err))
}
