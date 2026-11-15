// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! vit_2023 transformers for the DAG zonotope program (`#vit-zono-root`,
//! VIBECHECK plan I1).
//!
//! They extend the shared certified core (`zono_core`); nothing here forks it.
//! Every transformer keeps the [`ZonoState`] invariant (`state.rs`): if the
//! TRUE value of each input lies in its represented set for the current
//! symbol assignment, the true output lies in the output's represented set
//! for the SAME assignment of every pre-existing symbol and SOME value of each
//! fresh symbol. Arithmetic is plain binary64 round-to-nearest; every rounding
//! error is charged into the remainder with an explicit Higham `gamma_n`
//! envelope (valid for any summation order, blocked GEMM and FMA alike) and
//! every endpoint is stepped outward.
//!
//! * [`bilinear_matmul`] — `C = s * A @ B` (or `A @ B^T`) of two LIVE
//!   zonotopes (vit's `Q @ K^T` and `attn @ V`, lowered by ny to
//!   `BilinearCrown`). First-order terms are exact per symbol and combined BY
//!   SYMBOL ID (never by position — Vibecheck's U8 defect); the second-order
//!   term `DeltaA @ DeltaB` is bounded by `|s| * rad(A) @ rad(B)` and carried
//!   by ONE fresh symbol per output element (DeepT's certified remainder
//!   generator, no `e^2` credit).
//! * [`exp_band`], [`reciprocal_band`], [`clip_band`] — rigorous affine bands
//!   (`lambda x + [mu_lo, mu_hi]`) for [`super::elementwise::apply_band`]: the
//!   convex chord/tangent parallelogram for `exp` (certified exponential
//!   enclosures only, no libm accuracy assumption) and for `1/x` on `x > 0`
//!   (with wave-6's `[l, +inf)` relaxation as its degenerate slope-0 case), and
//!   the exact piecewise-linear band of `Clip`.
//! * [`group_sum`] — `ReduceSum`/`ReduceMean` (sparse, Higham-charged).
//! * [`softmax_box`] — a certified box of a DIRECT `Softmax` from the
//!   zonotope's pairwise differences `x_b - x_a` (the shared symbols cancel
//!   exactly), then [`box_state`] (one fresh symbol per element).
//!
//! Test-only fault switches ([`fault`]) exist solely so the pins can prove the
//! runtime self-check (`vit_root.rs`) catches a transformer that encloses the
//! wrong function and a compiler that models the wrong one.

use std::collections::HashMap;
use std::time::Instant;

use faer::MatRef;
use rayon::prelude::*;

use super::elementwise::{append_fresh, drop_zero_rows, Band};
use super::ids::{SymbolAllocator, SymbolId};
use super::kernels::{check_deadline, gemm_into};
use super::rounding::{gamma, next_down, next_up, TINY, UNIT};
use super::state::ZonoState;
use super::ZonoRefusal;
use crate::layers::misc::reciprocal::reciprocal_unbounded_upper_relaxation;
use crate::network::graph_ibp_f64_cell::exp_certified_total;

#[cfg(test)]
fn matmul_ignores_scale() -> bool {
    fault::get() == fault::VitFault::MatMulIgnoreScale
}

#[cfg(not(test))]
#[inline]
const fn matmul_ignores_scale() -> bool {
    false
}

/// The compile-side fault (test only): the DAG compiler reads it when it
/// builds a `BilinearCrown`/`MatMul` step. Always `false` in production.
#[cfg(test)]
pub(super) fn compile_drops_scale() -> bool {
    fault::get() == fault::VitFault::CompileDropScale
}

/// Always `false` in production.
#[cfg(not(test))]
#[inline]
pub(super) const fn compile_drops_scale() -> bool {
    false
}

/// Rows per chunk so that one scratch buffer stays near 1 Mi elements (8 MiB).
fn chunk_rows(per_row_elems: usize) -> usize {
    ((1usize << 20) / per_row_elems.max(1)).max(1)
}

// ---------------------------------------------------------------------------
// Live x live MatMul
// ---------------------------------------------------------------------------

/// Shape of `C[b] = s * A[b] @ B[b]` (or `A[b] @ B[b]^T`), all row-major:
/// `A` is `batch x m x k`, `B` is `batch x k x n` (`batch x n x k` when
/// `transpose_b`), `C` is `batch x m x n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatMulDims {
    /// Product of the leading (batch) axes.
    pub batch: usize,
    /// Rows of each `A` matrix.
    pub m: usize,
    /// Contraction length.
    pub k: usize,
    /// Columns of each output matrix.
    pub n: usize,
    /// `B` is stored transposed (`n x k` per batch).
    pub transpose_b: bool,
    /// Exact binary64 image of the layer's `f32` scale (`None` = 1).
    pub scale: Option<f64>,
}

impl MatMulDims {
    /// Validated dimensions (every product checked for overflow).
    pub fn new(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        transpose_b: bool,
        scale: Option<f64>,
    ) -> Result<Self, ZonoRefusal> {
        let dims = Self {
            batch,
            m,
            k,
            n,
            transpose_b,
            scale,
        };
        for (what, len) in [
            ("A", batch.checked_mul(m).and_then(|v| v.checked_mul(k))),
            ("B", batch.checked_mul(k).and_then(|v| v.checked_mul(n))),
            ("C", batch.checked_mul(m).and_then(|v| v.checked_mul(n))),
        ] {
            if len.is_none() {
                return Err(ZonoRefusal::Shape(format!(
                    "matmul {what} size overflows ({batch} x {m} x {k} x {n})"
                )));
            }
        }
        if let Some(s) = scale {
            if !s.is_finite() {
                return Err(ZonoRefusal::NonFinite("matmul scale"));
            }
        }
        Ok(dims)
    }

    /// Flat length of `A`.
    #[must_use]
    pub fn a_len(&self) -> usize {
        self.batch * self.m * self.k
    }

    /// Flat length of `B`.
    #[must_use]
    pub fn b_len(&self) -> usize {
        self.batch * self.k * self.n
    }

    /// Flat length of `C`.
    #[must_use]
    pub fn out_len(&self) -> usize {
        self.batch * self.m * self.n
    }

    /// Flat index of the logical element `B[bt][l][j]` (row `l` of the
    /// contraction, column `j` of the output).
    #[inline]
    #[must_use]
    pub fn b_index(&self, bt: usize, l: usize, j: usize) -> usize {
        if self.transpose_b {
            (bt * self.n + j) * self.k + l
        } else {
            (bt * self.k + l) * self.n + j
        }
    }

    /// Fill `out` (row-major `k x n`) with batch `bt` of a `B`-shaped vector.
    fn b_block(&self, values: &[f64], bt: usize, out: &mut [f64]) {
        for l in 0..self.k {
            for j in 0..self.n {
                out[l * self.n + j] = values[self.b_index(bt, l, j)];
            }
        }
    }
}

/// Certified UPPER bound on `left @ right` for NON-NEGATIVE operands
/// (`left` is `A`-shaped, `right` is `B`-shaped), per output element.
///
/// The binary64 sum of `k` non-negative products is below the exact value by
/// at most a relative `gamma_k`; `1 / (1 - gamma_k) <= 1 + gamma_{k+2}` for
/// every realistic `k`, and the final product is rounded up one ulp.
/// Gradual underflow (absolute `<= k * 2^-1074`) is covered by the `TINY`
/// floor every caller adds to the remainder of the SAME element.
fn nonneg_matmul_upper(dims: &MatMulDims, left: &[f64], right: &[f64]) -> Vec<f64> {
    let MatMulDims { batch, m, k, n, .. } = *dims;
    let factor = 1.0 + gamma(k + 2);
    let mut out = vec![0.0f64; dims.out_len()];
    for bt in 0..batch {
        for i in 0..m {
            let row = &left[(bt * m + i) * k..(bt * m + i + 1) * k];
            for j in 0..n {
                let mut acc = 0.0f64;
                for (l, &x) in row.iter().enumerate() {
                    acc += x * right[dims.b_index(bt, l, j)];
                }
                out[(bt * m + i) * n + j] = next_up(acc * factor);
            }
        }
    }
    out
}

/// `C = s * A @ B` (or `A @ B^T`) of two live zonotopes, aligned BY SYMBOL ID.
///
/// SOUND. Write `A = cA + dA`, `dA = sum_t GA_t e_t + rhoA` (likewise `B`).
/// Then `A B = cA cB + sum_t (cA GB_t + GA_t cB) e_t + (cA rhoB + rhoA cB) +
/// dA dB` for the SAME assignment `e` of every symbol (ID alignment). So:
///
/// * the centre is `s cA cB` and the row of symbol `t` is
///   `s (cA GB_t + GA_t cB)` (a symbol one side lacks has a zero row there);
/// * `|(dA dB)_ij| <= sum_l radA_il radB_lj` with `rad = sum|G| + r`, carried
///   by ONE fresh symbol per output element with radius `|s| UB(radA @ radB)`;
/// * `|s (cA rhoB + rhoA cB)| <= |s| (|cA| @ rB + rA @ |cB|)` goes into the
///   remainder;
/// * rounding: the centre is within `|s| gamma_{k+1} |cA| @ |cB|` and each
///   row, computed as two length-`k` sums, one addition and one scaling,
///   within `|s| gamma_{k+3} (|cA| @ |GB_t| + |GA_t| @ |cB|)`; summed over
///   every symbol that is at most `|s| gamma_{k+3} (|cA| @ absB + absA @
///   |cB|) <= 2 |s| gamma_{k+3} mA @ mB` (`m = |c| + abs + r` dominates both
///   `|c|` and `abs`). The remainder charges `4 gamma_{k+4} |s| UB(mA @ mB)`,
///   which dominates the centre and the rows together.
///
/// Returns the state and the number of fresh symbols.
pub fn bilinear_matmul(
    a: &ZonoState,
    b: &ZonoState,
    dims: &MatMulDims,
    alloc: &mut SymbolAllocator,
    deadline: Option<Instant>,
) -> Result<(ZonoState, usize), ZonoRefusal> {
    let MatMulDims { batch, m, k, n, .. } = *dims;
    if a.len() != dims.a_len() || b.len() != dims.b_len() {
        return Err(ZonoRefusal::Shape(format!(
            "matmul operands of {} and {} values for {batch} x ({m} x {k}) @ ({k} x {n})",
            a.len(),
            b.len()
        )));
    }
    let abs_a = a.abs_generator_sum();
    let abs_b = b.abs_generator_sum();
    a.ensure_finite(&abs_a)?;
    b.ensure_finite(&abs_b)?;
    let n_out = dims.out_len();
    let scale = if matmul_ignores_scale() {
        1.0
    } else {
        dims.scale.unwrap_or(1.0)
    };
    let scale_abs = scale.abs();

    let ca = a.center();
    let cb = b.center();
    let ra = a.remainder();
    let rb = b.remainder();
    let ma = a.magnitude(&abs_a);
    let mb = b.magnitude(&abs_b);
    let rad_a: Vec<f64> = abs_a
        .iter()
        .zip(ra)
        .map(|(&g, &r)| next_up(g + r))
        .collect();
    let rad_b: Vec<f64> = abs_b
        .iter()
        .zip(rb)
        .map(|(&g, &r)| next_up(g + r))
        .collect();
    let ca_abs: Vec<f64> = ca.iter().map(|v| v.abs()).collect();
    let cb_abs: Vec<f64> = cb.iter().map(|v| v.abs()).collect();

    // Centre.
    let mut center = vec![0.0f64; n_out];
    let mut cb_block = vec![0.0f64; k * n];
    for bt in 0..batch {
        dims.b_block(cb, bt, &mut cb_block);
        for i in 0..m {
            let row = &ca[(bt * m + i) * k..(bt * m + i + 1) * k];
            for j in 0..n {
                let mut acc = 0.0f64;
                for (l, &x) in row.iter().enumerate() {
                    acc += x * cb_block[l * n + j];
                }
                center[(bt * m + i) * n + j] = acc * scale;
            }
        }
    }

    // Symbol union BY ID: every symbol of A in A's order, then the symbols
    // only B knows in B's order.
    let a_row_of: HashMap<SymbolId, usize> = a
        .ids()
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, row))
        .collect();
    let mut ids: Vec<SymbolId> = a.ids().to_vec();
    let mut b_out_row = Vec::with_capacity(b.num_generators());
    for &id in b.ids() {
        match a_row_of.get(&id) {
            Some(&row) => b_out_row.push(row),
            None => {
                b_out_row.push(ids.len());
                ids.push(id);
            }
        }
    }
    let k_out = ids.len();
    let total = k_out
        .checked_mul(n_out)
        .ok_or_else(|| ZonoRefusal::Shape("matmul generator size overflows".to_string()))?;
    let mut gens = vec![0.0f64; total];

    if n_out > 0 {
        // A side: row t of the output starts as GA_t @ cB (batch by batch;
        // the rows of a chunk of symbols are stacked into one GEMM).
        let ka = a.num_generators();
        let mk = m * k;
        let mn = m * n;
        if ka > 0 && mk > 0 {
            let chunk = chunk_rows(m * k.max(n)).min(ka);
            let mut lhs = vec![0.0f64; chunk * mk];
            let mut out = vec![0.0f64; chunk * mn];
            for bt in 0..batch {
                dims.b_block(cb, bt, &mut cb_block);
                let rhs = MatRef::from_row_major_slice(&cb_block, k, n);
                for t0 in (0..ka).step_by(chunk) {
                    check_deadline(deadline)?;
                    let rows = chunk.min(ka - t0);
                    for tl in 0..rows {
                        let src = &a.generator_row(t0 + tl)[bt * mk..(bt + 1) * mk];
                        lhs[tl * mk..(tl + 1) * mk].copy_from_slice(src);
                    }
                    let lhs_view = MatRef::from_row_major_slice(&lhs[..rows * mk], rows * m, k);
                    gemm_into(&mut out[..rows * mn], rows * m, n, lhs_view, rhs);
                    for tl in 0..rows {
                        let base = (t0 + tl) * n_out + bt * mn;
                        gens[base..base + mn].copy_from_slice(&out[tl * mn..(tl + 1) * mn]);
                    }
                }
            }
        }
        // B side: add cA @ GB_u into the output row of u's symbol (the column
        // blocks of a chunk of symbols are laid side by side in one GEMM).
        let kb = b.num_generators();
        if kb > 0 && k > 0 {
            let chunk = chunk_rows(k.max(m) * n).min(kb);
            let mut rhs_buf = vec![0.0f64; k * chunk * n];
            let mut out = vec![0.0f64; m * chunk * n];
            for bt in 0..batch {
                let lhs = MatRef::from_row_major_slice(&ca[bt * mk..(bt + 1) * mk], m, k);
                for u0 in (0..kb).step_by(chunk) {
                    check_deadline(deadline)?;
                    let cols = chunk.min(kb - u0);
                    let width = cols * n;
                    for ul in 0..cols {
                        let g = b.generator_row(u0 + ul);
                        for l in 0..k {
                            for j in 0..n {
                                rhs_buf[l * width + ul * n + j] = g[dims.b_index(bt, l, j)];
                            }
                        }
                    }
                    let rhs = MatRef::from_row_major_slice(&rhs_buf[..k * width], k, width);
                    gemm_into(&mut out[..m * width], m, width, lhs, rhs);
                    for ul in 0..cols {
                        let base = b_out_row[u0 + ul] * n_out + bt * mn;
                        for i in 0..m {
                            let src = &out[i * width + ul * n..i * width + (ul + 1) * n];
                            let dst = &mut gens[base + i * n..base + (i + 1) * n];
                            for (d, s) in dst.iter_mut().zip(src) {
                                *d += *s;
                            }
                        }
                    }
                }
            }
        }
        if scale != 1.0 {
            for value in &mut gens {
                *value *= scale;
            }
        }
    }

    // Charges (see the SOUND note).
    let mm = nonneg_matmul_upper(dims, &ma, &mb);
    let rad2 = nonneg_matmul_upper(dims, &rad_a, &rad_b);
    let cross_a = nonneg_matmul_upper(dims, &ca_abs, rb);
    let cross_b = nonneg_matmul_upper(dims, ra, &cb_abs);
    let g_round = 4.0 * gamma(k + 4);
    let mut rem = Vec::with_capacity(n_out);
    let mut second = Vec::with_capacity(n_out);
    for j in 0..n_out {
        second.push(next_up(scale_abs * rad2[j]));
        let cross = next_up(scale_abs * next_up(cross_a[j] + cross_b[j]));
        let charge = next_up(g_round * next_up(scale_abs * mm[j]));
        rem.push(next_up((cross + charge) * (1.0 + 4.0 * UNIT) + TINY));
    }
    let state = ZonoState::from_parts_trusted(n_out, center, gens, ids, rem);
    let abs_out = state.abs_generator_sum();
    state.ensure_finite(&abs_out)?;
    let (state, fresh) = append_fresh(state, &second, alloc)?;
    Ok((drop_zero_rows(state), fresh))
}

/// Plain binary64 `s * A @ B` (the sampling oracle; not a certificate).
#[must_use]
pub fn matmul_eval(dims: &MatMulDims, a: &[f64], b: &[f64]) -> Vec<f64> {
    let MatMulDims { batch, m, k, n, .. } = *dims;
    let scale = dims.scale.unwrap_or(1.0);
    let mut out = vec![0.0f64; dims.out_len()];
    for bt in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += a[(bt * m + i) * k + l] * b[dims.b_index(bt, l, j)];
                }
                out[(bt * m + i) * n + j] = acc * scale;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Bands: Exp, Reciprocal, Clip
// ---------------------------------------------------------------------------

/// A certified `t` with `exp(t) >= v` (so `t >= ln v`), or `None`.
///
/// The libm logarithm only PROPOSES the point; the claim rests on the
/// certified exponential enclosure (`exp_certified_total`, no libm accuracy
/// assumption): its LOWER end at `t` reaching `v` proves `e^t >= v`.
fn ln_upper_bound(v: f64) -> Option<f64> {
    if !(v > 0.0 && v.is_finite()) {
        return None;
    }
    let t0 = v.ln();
    if !t0.is_finite() {
        return None;
    }
    let mut pad = 8.0 * f64::EPSILON * t0.abs().max(1.0);
    for _ in 0..12 {
        let t = t0 + pad;
        if t.is_finite() && exp_certified_total(t).0 >= v {
            return Some(t);
        }
        pad *= 16.0;
    }
    None
}

/// Rigorous affine band of `exp` over `[l, u]` per element.
///
/// SOUND. `g(x) = e^x - lambda x` is convex for every slope `lambda`, so
/// (a) its maximum over `[l, u]` sits at an endpoint: `mu_hi` is the larger of
/// `next_up(E_hi(t) - next_down(lambda t))`, `t in {l, u}`, where `E_hi` is the
/// certified upper end of `e^t`; (b) for `lambda > 0` its global minimum is
/// `lambda (1 - ln lambda)`, bounded below by `lambda - UB(lambda t_hi)` with
/// a certified `t_hi >= ln lambda` ([`ln_upper_bound`]), rounded down. The
/// slope is the chord of the libm values (any finite positive slope is sound;
/// the offsets are computed for exactly this binary64 slope). Where the chord
/// is unusable (a point interval, a non-finite slope, no certified logarithm)
/// the band degrades to the constant `[E_lo(l), E_hi(u)]` (slope 0), valid
/// because `exp` is increasing. An infinite `E_hi(u)` REFUSES.
pub fn exp_band(lo: &[f64], hi: &[f64]) -> Result<Band, ZonoRefusal> {
    if lo.len() != hi.len() {
        return Err(ZonoRefusal::Shape("exp band bounds".to_string()));
    }
    let n = lo.len();
    let mut band = Band {
        lambda: Vec::with_capacity(n),
        mu_lo: Vec::with_capacity(n),
        mu_hi: Vec::with_capacity(n),
    };
    for (&l, &u) in lo.iter().zip(hi) {
        if !(l.is_finite() && u.is_finite()) || l > u {
            return Err(ZonoRefusal::NonFinite("exp band interval"));
        }
        let (el_lo, el_hi) = exp_certified_total(l);
        let (_, eu_hi) = exp_certified_total(u);
        if !(el_hi.is_finite() && eu_hi.is_finite()) {
            return Err(ZonoRefusal::NonFinite("exp band overflow"));
        }
        let chord = (|| {
            if u <= l {
                return None;
            }
            let lam = (u.exp() - l.exp()) / (u - l);
            if !(lam.is_finite() && lam > 0.0) {
                return None;
            }
            let up_l = next_up(el_hi - next_down(lam * l));
            let up_u = next_up(eu_hi - next_down(lam * u));
            let mu_hi = up_l.max(up_u);
            let t_hi = ln_upper_bound(lam)?;
            let mu_lo = next_down(lam - next_up(lam * t_hi));
            (mu_lo.is_finite() && mu_hi.is_finite() && mu_lo <= mu_hi)
                .then_some((lam, mu_lo, mu_hi))
        })();
        match chord {
            Some((lam, mu_lo, mu_hi)) => {
                band.lambda.push(lam);
                band.mu_lo.push(mu_lo);
                band.mu_hi.push(mu_hi);
            }
            None => {
                band.lambda.push(0.0);
                band.mu_lo.push(el_lo);
                band.mu_hi.push(eu_hi);
            }
        }
    }
    Ok(band)
}

/// Certified range of `exp` over each `[l, u]` (increasing, so
/// `[E_lo(l), E_hi(u)]`).
#[must_use]
pub fn exp_range(lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    lo.iter()
        .zip(hi)
        .map(|(&l, &u)| (exp_certified_total(l).0, exp_certified_total(u).1))
        .unzip()
}

/// Rigorous affine band of `1/x` over `[l, u]`, `0 < l <= u`, per element.
///
/// SOUND. `g(x) = 1/x - lambda x` is convex on `x > 0` (`g'' = 2/x^3`), so
/// (a) its maximum over `[l, u]` sits at an endpoint:
/// `mu_hi = max_t next_up(next_up(1/t) - next_down(lambda t))`; (b) for
/// `lambda < 0` its minimum over `x > 0` is `2 sqrt(-lambda)` (AM-GM:
/// `1/x + v x >= 2 sqrt(v)`), bounded below by
/// `next_down(2 next_down(sqrt(-lambda)))` (IEEE `sqrt` is correctly rounded;
/// the doubling is exact). The slope is the chord `-1/(l u)`. Where it is not
/// a finite negative number (an `l u` that overflows — the `u -> +inf` regime —
/// or a point interval) the band has slope 0: its upper offset is wave-6's
/// `[l, +inf)` relaxation (`reciprocal_unbounded_upper_relaxation`, the
/// supremum `1/l` rounded up), its lower offset the larger of wave-6's `0` and
/// `next_down(1/u)`; both are valid because `1/x` is positive and decreasing.
/// `l <= 0` (the interval touches the pole) REFUSES.
pub fn reciprocal_band(lo: &[f64], hi: &[f64]) -> Result<Band, ZonoRefusal> {
    if lo.len() != hi.len() {
        return Err(ZonoRefusal::Shape("reciprocal band bounds".to_string()));
    }
    let n = lo.len();
    let mut band = Band {
        lambda: Vec::with_capacity(n),
        mu_lo: Vec::with_capacity(n),
        mu_hi: Vec::with_capacity(n),
    };
    for (&l, &u) in lo.iter().zip(hi) {
        if !(l > 0.0 && l.is_finite() && u.is_finite()) || l > u {
            return Err(ZonoRefusal::NonFinite(
                "reciprocal band needs 0 < l <= u < inf",
            ));
        }
        let lam = -1.0 / (l * u);
        let chord = (lam < 0.0 && lam.is_finite() && u > l).then(|| {
            let up_l = next_up(next_up(1.0 / l) - next_down(lam * l));
            let up_u = next_up(next_up(1.0 / u) - next_down(lam * u));
            let mu_hi = up_l.max(up_u);
            let mu_lo = next_down(2.0 * next_down((-lam).sqrt()));
            (mu_lo, mu_hi)
        });
        match chord {
            Some((mu_lo, mu_hi)) if mu_lo.is_finite() && mu_hi.is_finite() && mu_lo <= mu_hi => {
                band.lambda.push(lam);
                band.mu_lo.push(mu_lo);
                band.mu_hi.push(mu_hi);
            }
            _ => {
                // wave-6's [l, +inf) relaxation: the f32 image of `l` rounded
                // DOWN is still a valid (smaller) positive lower end, so its
                // intercept `>= 1/l32 >= 1/l`. Its lower line `0` is raised to
                // `next_down(1/u)`, the finite `u`'s own (decreasing) floor.
                let l32 = ny_core::f64_to_f32_down(l);
                let relaxation = reciprocal_unbounded_upper_relaxation(l32);
                let mu_lo = f64::from(relaxation.lower_intercept).max(next_down(1.0 / u).max(0.0));
                let mu_hi = f64::from(relaxation.upper_intercept);
                if relaxation.lower_slope != 0.0
                    || relaxation.upper_slope != 0.0
                    || !(mu_lo.is_finite() && mu_hi.is_finite())
                    || mu_lo > mu_hi
                {
                    return Err(ZonoRefusal::NonFinite("reciprocal unbounded band"));
                }
                band.lambda.push(0.0);
                band.mu_lo.push(mu_lo);
                band.mu_hi.push(mu_hi);
            }
        }
    }
    Ok(band)
}

/// Certified range of `1/x` over each `[l, u]`, `0 < l` (decreasing, so
/// `[next_down(1/u), next_up(1/l)]`; IEEE division is correctly rounded).
#[must_use]
pub fn reciprocal_range(lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    lo.iter()
        .zip(hi)
        .map(|(&l, &u)| {
            if l > 0.0 && l.is_finite() && u.is_finite() && l <= u {
                (next_down(1.0 / u).max(0.0), next_up(1.0 / l))
            } else {
                (f64::NEG_INFINITY, f64::INFINITY)
            }
        })
        .unzip()
}

/// `clip(x) = min(max(x, lo), hi)`.
#[inline]
#[must_use]
pub fn clip_value(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Exact affine band of `Clip(cmin, cmax)` over `[l, u]` per element.
///
/// SOUND. `g(t) = clip(t) - lambda t` is continuous and affine between the
/// breakpoints `cmin`, `cmax`, so its extrema over `[l, u]` sit on
/// `{l, u} ∪ ({cmin, cmax} ∩ (l, u))`. `clip` of a binary64 is exact; `lambda
/// t` is bracketed one ulp each way and the difference rounded outward. The
/// slope is the chord (any finite value is sound).
pub fn clip_band(lo: &[f64], hi: &[f64], cmin: f64, cmax: f64) -> Result<Band, ZonoRefusal> {
    if lo.len() != hi.len() || cmin.is_nan() || cmax.is_nan() || cmin > cmax {
        return Err(ZonoRefusal::Shape("clip band inputs".to_string()));
    }
    let n = lo.len();
    let mut band = Band {
        lambda: Vec::with_capacity(n),
        mu_lo: Vec::with_capacity(n),
        mu_hi: Vec::with_capacity(n),
    };
    for (&l, &u) in lo.iter().zip(hi) {
        if !(l.is_finite() && u.is_finite()) || l > u {
            return Err(ZonoRefusal::NonFinite("clip band interval"));
        }
        let (fl, fu) = (clip_value(l, cmin, cmax), clip_value(u, cmin, cmax));
        let mut lam = if u > l { (fu - fl) / (u - l) } else { 0.0 };
        if !lam.is_finite() {
            lam = 0.0;
        }
        let mut g_hi = f64::NEG_INFINITY;
        let mut g_lo = f64::INFINITY;
        let mut visit = |t: f64| {
            let f = clip_value(t, cmin, cmax);
            let lt = lam * t;
            g_hi = g_hi.max(next_up(f - next_down(lt)));
            g_lo = g_lo.min(next_down(f - next_up(lt)));
        };
        visit(l);
        visit(u);
        for bp in [cmin, cmax] {
            if bp > l && bp < u {
                visit(bp);
            }
        }
        if !(g_lo.is_finite() && g_hi.is_finite()) || g_lo > g_hi {
            return Err(ZonoRefusal::NonFinite("clip band offsets"));
        }
        band.lambda.push(lam);
        band.mu_lo.push(g_lo);
        band.mu_hi.push(g_hi);
    }
    Ok(band)
}

/// Exact range of `Clip` over each `[l, u]` (monotone).
#[must_use]
pub fn clip_range(lo: &[f64], hi: &[f64], cmin: f64, cmax: f64) -> (Vec<f64>, Vec<f64>) {
    lo.iter()
        .zip(hi)
        .map(|(&l, &u)| (clip_value(l, cmin, cmax), clip_value(u, cmin, cmax)))
        .unzip()
}

// ---------------------------------------------------------------------------
// ReduceSum / ReduceMean
// ---------------------------------------------------------------------------

/// The reduction map of `ReduceSum`/`ReduceMean(axes, keepdims)` over
/// `in_shape`: `(out_of, n_out, count)` where input element `i` feeds output
/// element `out_of[i]` and every output sums exactly `count` inputs.
///
/// Axes resolve as ny's layer resolves them (empty = every axis, negative
/// axes count from the end, a duplicate refuses). The resulting shape must be
/// `out_shape` (ny's own IBP shape) or the node refuses.
pub fn reduce_group_map(
    in_shape: &[usize],
    axes: &[i64],
    keepdims: bool,
    out_shape: &[usize],
) -> Result<(Vec<usize>, usize, usize), ZonoRefusal> {
    let ndim = in_shape.len();
    let resolved: Vec<usize> = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.iter()
            .map(|&axis| {
                ny_core::resolve_axis(axis, ndim, "reduce")
                    .map_err(|e| ZonoRefusal::UnsupportedGraph(format!("reduce axis: {e}")))
            })
            .collect::<Result<_, _>>()?
    };
    let mut reduced = vec![false; ndim];
    for &axis in &resolved {
        if reduced[axis] {
            return Err(ZonoRefusal::UnsupportedGraph(format!(
                "reduce: duplicate axis {axis}"
            )));
        }
        reduced[axis] = true;
    }
    let mut expected = Vec::with_capacity(ndim);
    let mut count = 1usize;
    for (axis, &dim) in in_shape.iter().enumerate() {
        if reduced[axis] {
            count = count
                .checked_mul(dim)
                .ok_or_else(|| ZonoRefusal::Shape("reduce count overflows".to_string()))?;
            if keepdims {
                expected.push(1);
            }
        } else {
            expected.push(dim);
        }
    }
    if expected != out_shape {
        return Err(ZonoRefusal::Shape(format!(
            "reduce {in_shape:?} over {resolved:?} (keepdims={keepdims}) gives {expected:?}, \
             IBP shape {out_shape:?}"
        )));
    }
    if count == 0 {
        return Err(ZonoRefusal::Shape("reduce over an empty axis".to_string()));
    }
    let n_in = in_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ZonoRefusal::Shape("reduce input overflows".to_string()))?;
    // Output strides over the kept axes (row-major).
    let mut out_stride = vec![0usize; ndim];
    let mut stride = 1usize;
    for axis in (0..ndim).rev() {
        if !reduced[axis] {
            out_stride[axis] = stride;
            stride *= in_shape[axis];
        }
    }
    let n_out = stride;
    let mut out_of = Vec::with_capacity(n_in);
    for flat in 0..n_in {
        let mut rest = flat;
        let mut out = 0usize;
        for axis in (0..ndim).rev() {
            let coord = rest % in_shape[axis];
            rest /= in_shape[axis];
            if !reduced[axis] {
                out += coord * out_stride[axis];
            }
        }
        out_of.push(out);
    }
    Ok((out_of, n_out, count))
}

/// `y_j = sum_{i : out_of[i] = j} x_i` (sparse; every group has at most `fan`
/// members).
///
/// SOUND (the affine charge of `apply_affine` with `|A|` a 0/1 matrix): the
/// computed centre and every row are sums of at most `fan` terms, within
/// `gamma_fan` of the exact sums of `|c|` and `|G_t|`; summed over rows that is
/// at most `gamma_fan sum_i m_i` with `m >= |c| + sum|G| + r`. The remainders
/// add exactly (`|sum rho_i| <= sum r_i`). The new remainder
/// `(sum r + gamma_{fan+2} sum m)(1 + gamma_{fan+5})`, rounded up, dominates
/// all of it (the trailing factor covers the binary64 sums of `r` and `m`).
pub fn group_sum(
    x: &ZonoState,
    abs_g: &[f64],
    out_of: &[usize],
    n_out: usize,
    fan: usize,
) -> Result<ZonoState, ZonoRefusal> {
    let n = x.len();
    if out_of.len() != n || abs_g.len() != n {
        return Err(ZonoRefusal::Shape(format!(
            "group sum map of {} for {n} values",
            out_of.len()
        )));
    }
    if out_of.iter().any(|&j| j >= n_out) {
        return Err(ZonoRefusal::Internal(
            "group sum map points past its output".to_string(),
        ));
    }
    let m = x.magnitude(abs_g);
    let mut center = vec![0.0f64; n_out];
    let mut rsum = vec![0.0f64; n_out];
    let mut msum = vec![0.0f64; n_out];
    for (i, &j) in out_of.iter().enumerate() {
        center[j] += x.center()[i];
        rsum[j] += x.remainder()[i];
        msum[j] += m[i];
    }
    let k = x.num_generators();
    let total = k
        .checked_mul(n_out)
        .ok_or_else(|| ZonoRefusal::Shape("group sum generator size overflows".to_string()))?;
    let mut gens = vec![0.0f64; total];
    if n > 0 && n_out > 0 {
        for (t, row) in x.generators().chunks_exact(n).enumerate() {
            let dst = &mut gens[t * n_out..(t + 1) * n_out];
            for (i, &j) in out_of.iter().enumerate() {
                dst[j] += row[i];
            }
        }
    }
    let g_round = gamma(fan + 2);
    let g_eval = 1.0 + gamma(fan + 5);
    let rem: Vec<f64> = (0..n_out)
        .map(|j| next_up((rsum[j] + g_round * msum[j]) * g_eval + TINY))
        .collect();
    Ok(drop_zero_rows(ZonoState::from_parts_trusted(
        n_out,
        center,
        gens,
        x.ids().to_vec(),
        rem,
    )))
}

// ---------------------------------------------------------------------------
// Direct Softmax
// ---------------------------------------------------------------------------

/// The softmax lanes of `shape` along `axis`: the flat indices of every
/// 1-D fiber the softmax normalises, in row-major order.
pub fn softmax_lanes(shape: &[usize], axis: usize) -> Result<Vec<Vec<usize>>, ZonoRefusal> {
    if axis >= shape.len() {
        return Err(ZonoRefusal::Shape(format!(
            "softmax axis {axis} for shape {shape:?}"
        )));
    }
    let outer: usize = shape[..axis].iter().product();
    let n = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    if n == 0 {
        return Err(ZonoRefusal::Shape("softmax over an empty axis".to_string()));
    }
    let mut lanes = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            lanes.push((0..n).map(|j| (o * n + j) * inner + i).collect());
        }
    }
    Ok(lanes)
}

/// Certified box of `softmax` over every lane from a zonotope of its input.
///
/// `softmax_a(x) = 1 / (1 + sum_{b != a} exp(x_b - x_a))` is increasing in
/// every difference's negation, so with `U_ab >= x_b - x_a` and `L_ab <= x_b -
/// x_a` over the whole input set, `p_a in [1/(1 + sum_b E_hi(U_ab)),
/// 1/(1 + sum_b E_lo(L_ab))]`. The differences come from the ZONOTOPE:
/// `x_b - x_a = (c_b - c_a) + sum_t (G_tb - G_ta) e_t + (rho_b - rho_a)`, so
/// `U_ab = next_up(c_b - c_a) + UB(sum_t |G_tb - G_ta|) + r_a + r_b` (each
/// `|fl(G_tb - G_ta)|` is within a relative `u`, the sum of `K` of them within
/// `gamma_K`: the factor `1 + gamma_{K+4}` covers both), every addition
/// rounded up; `L_ab = -U_ba`. A shared symbol CANCELS exactly here — the
/// query-row term of vit's `Q K^T` — which is what makes this tighter than the
/// independent-box bound `hi_b - lo_a`, itself also valid (the tighter of the
/// two is kept). Every exponential is certified (`exp_certified_total`), the
/// denominators are accumulated outward (and the lower one floored at 1, its
/// exact value's floor), the reciprocals are rounded outward and the result is
/// clipped to `[0, 1]`, the range of every softmax coordinate.
pub fn softmax_box(
    x: &ZonoState,
    abs_g: &[f64],
    lo: &[f64],
    hi: &[f64],
    lanes: &[Vec<usize>],
    deadline: Option<Instant>,
) -> Result<(Vec<f64>, Vec<f64>), ZonoRefusal> {
    let len = x.len();
    if lo.len() != len || hi.len() != len || abs_g.len() != len {
        return Err(ZonoRefusal::Shape("softmax box bounds".to_string()));
    }
    x.ensure_finite(abs_g)?;
    if lanes.iter().flatten().any(|&idx| idx >= len) {
        return Err(ZonoRefusal::Internal(
            "softmax lane past its input".to_string(),
        ));
    }
    let kk = x.num_generators();
    let factor = 1.0 + gamma(kk + 4);
    let lane_boxes: Vec<(Vec<f64>, Vec<f64>)> = lanes
        .par_iter()
        .map(|lane| -> Result<(Vec<f64>, Vec<f64>), ZonoRefusal> {
            check_deadline(deadline)?;
            let n = lane.len();
            let mut buf = vec![0.0f64; kk * n];
            for t in 0..kk {
                let row = x.generator_row(t);
                for (a, &idx) in lane.iter().enumerate() {
                    buf[t * n + a] = row[idx];
                }
            }
            let mut diff = vec![0.0f64; n * n];
            for t in 0..kk {
                let r = &buf[t * n..(t + 1) * n];
                for a in 0..n {
                    let va = r[a];
                    for b in (a + 1)..n {
                        diff[a * n + b] += (va - r[b]).abs();
                    }
                }
            }
            let c = x.center();
            let rem = x.remainder();
            // upper[a * n + b] >= x_b - x_a.
            let mut upper = vec![0.0f64; n * n];
            for a in 0..n {
                for b in 0..n {
                    if a == b {
                        continue;
                    }
                    let (p, q) = if a < b { (a, b) } else { (b, a) };
                    let dsum = next_up(diff[p * n + q] * factor);
                    let (ia, ib) = (lane[a], lane[b]);
                    let centre_diff = next_up(c[ib] - c[ia]);
                    let spread = next_up(dsum + next_up(rem[ia] + rem[ib]));
                    let zono = next_up(centre_diff + spread);
                    let boxed = next_up(hi[ib] - lo[ia]);
                    let chosen = if boxed.is_nan() {
                        zono
                    } else {
                        zono.min(boxed)
                    };
                    if chosen.is_nan() {
                        return Err(ZonoRefusal::NonFinite("softmax difference bound"));
                    }
                    upper[a * n + b] = chosen;
                }
            }
            let mut out_lo = Vec::with_capacity(n);
            let mut out_hi = Vec::with_capacity(n);
            for a in 0..n {
                let mut den_up = 1.0f64;
                let mut den_down = 1.0f64;
                for b in 0..n {
                    if a == b {
                        continue;
                    }
                    den_up = next_up(den_up + exp_certified_total(upper[a * n + b]).1);
                    // x_b - x_a >= -(x_a - x_b upper bound); negation is exact.
                    den_down = next_down(den_down + exp_certified_total(-upper[b * n + a]).0);
                }
                if den_down.is_nan() || den_down < 1.0 {
                    den_down = 1.0;
                }
                let p_lo = if den_up.is_nan() {
                    0.0
                } else {
                    next_down(1.0 / den_up).max(0.0)
                };
                let p_hi = next_up(1.0 / den_down).min(1.0);
                if !(p_lo.is_finite() && p_hi.is_finite()) || p_lo > p_hi {
                    return Err(ZonoRefusal::NonFinite("softmax box"));
                }
                out_lo.push(p_lo);
                out_hi.push(p_hi);
            }
            Ok((out_lo, out_hi))
        })
        .collect::<Result<_, _>>()?;
    let mut out_lo = vec![f64::NAN; len];
    let mut out_hi = vec![f64::NAN; len];
    for (lane, (l, h)) in lanes.iter().zip(&lane_boxes) {
        for (a, &idx) in lane.iter().enumerate() {
            out_lo[idx] = l[a];
            out_hi[idx] = h[a];
        }
    }
    if out_lo.iter().chain(&out_hi).any(|v| v.is_nan()) {
        return Err(ZonoRefusal::Internal(
            "softmax lanes do not cover the tensor".to_string(),
        ));
    }
    Ok((out_lo, out_hi))
}

/// Plain binary64 softmax over the lanes (the sampling oracle).
#[must_use]
pub fn softmax_eval(lanes: &[Vec<usize>], x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; x.len()];
    for lane in lanes {
        let max = lane.iter().map(|&i| x[i]).fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = lane.iter().map(|&i| (x[i] - max).exp()).sum();
        for &i in lane {
            out[i] = (x[i] - max).exp() / sum;
        }
    }
    out
}

/// A zonotope of the box `[lo, hi]` with ONE fresh symbol per non-degenerate
/// element (centre `mid`, generator `rad >= max(hi - mid, mid - lo)`, both
/// differences rounded up), so every downstream consumer of the same element
/// keeps its correlation. Returns the state and the fresh count.
pub fn box_state(
    lo: &[f64],
    hi: &[f64],
    alloc: &mut SymbolAllocator,
) -> Result<(ZonoState, usize), ZonoRefusal> {
    let n = lo.len();
    if hi.len() != n {
        return Err(ZonoRefusal::Shape("box state bounds".to_string()));
    }
    let mut center = Vec::with_capacity(n);
    let mut radii = Vec::with_capacity(n);
    for (&l, &h) in lo.iter().zip(hi) {
        if !(l.is_finite() && h.is_finite()) || l > h {
            return Err(ZonoRefusal::NonFinite("box state interval"));
        }
        if l == h {
            center.push(l);
            radii.push(0.0);
        } else {
            let mid = 0.5 * l + 0.5 * h;
            center.push(mid);
            radii.push(next_up(next_up(h - mid).max(next_up(mid - l))));
        }
    }
    let state = ZonoState::from_parts_trusted(n, center, Vec::new(), Vec::new(), vec![0.0; n]);
    append_fresh(state, &radii, alloc)
}

/// Test-only fault injection for the `#vit-zono-root` pins.
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::Cell;

    /// The defects the pins must catch.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum VitFault {
        /// Production behaviour.
        None,
        /// [`super::bilinear_matmul`] silently drops the layer's `scale`: the
        /// transformer encloses a DIFFERENT function than the program's own
        /// point semantics. The sampled containment half of the runtime
        /// self-check must refuse.
        MatMulIgnoreScale,
        /// The DAG compiler drops a `BilinearCrown` scale, so the program
        /// models a different function than ny's layer. The semantic half of
        /// the runtime self-check (program vs ny's pointwise forward) must
        /// refuse.
        CompileDropScale,
    }

    thread_local! {
        static FAULT: Cell<VitFault> = const { Cell::new(VitFault::None) };
    }

    /// Arm (or, with `VitFault::None`, disarm) a fault on this thread.
    pub(crate) fn set(fault: VitFault) {
        FAULT.with(|cell| cell.set(fault));
    }

    pub(crate) fn get() -> VitFault {
        FAULT.with(Cell::get)
    }
}
