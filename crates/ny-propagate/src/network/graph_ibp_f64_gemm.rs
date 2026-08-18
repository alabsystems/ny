// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fast SOUND interval matrix product: Rump midpoint-radius form on
//! probed-thread f64 GEMMs (#f64-blas-gemm).
//!
//! # Why
//!
//! The f64 graph cell ([`super::graph_ibp_f64_cell`]) bounds nn4sys
//! mscn_2048d clause boxes through 2048-wide Linear/MatMul layers. Its scalar
//! interval matmul is a per-element corner-product triple loop (~28ms/node);
//! the box-refinement screen evaluates thousands of leaves, so the cell
//! matmul dominates once the CROWN-cost gate sheds CROWN. This module
//! replaces the O(m·n·k) scalar interval loop with 3-4 plain f64 GEMMs plus
//! O(m·n + m·k + k·n) elementwise work, and remains a SOUND enclosure. The
//! GEMMs deliberately do **not** use `ndarray::dot`: this workspace enables
//! threaded `matrixmultiply` and Apple Accelerate BLAS, whose private worker
//! threads expose no way to prove their FTZ/DAZ state. [`probed_matmul`]
//! instead partitions output rows over Rayon and probes the active binary64
//! environment on every worker immediately before its left-to-right dot
//! products. Any failed probe declines the kernel and callers use the scalar
//! path on their already-probed thread. It fires only for FAT left operands
//! (see [`FAST_GEMM_MIN_ROWS`]); thin batches stay on the cell's unrolled
//! scalar loop, which reads the f32 weight exactly once.
//!
//! # The math (Rump 1999, midpoint-radius interval GEMM)
//!
//! S. M. Rump, "Fast and parallel interval arithmetic", BIT 39(3), 1999.
//! Represent interval matrices by midpoint and radius: `[A] = <Am, Ar>`
//! means `{ A : |A - Am| <= Ar elementwise }`, radius `Ar >= 0`.
//! For any `A ∈ <Am, Ar>`, `B ∈ <Bm, Br>` write `A = Am + Ea`, `B = Bm + Eb`
//! with `|Ea| <= Ar`, `|Eb| <= Br`. Then, in EXACT real arithmetic,
//!
//! ```text
//! A·B = Am·Bm + Am·Eb + Ea·Bm + Ea·Eb
//! |A·B - Am·Bm| <= |Am|·Br + Ar·|Bm| + Ar·Br  =  |Am|·Br + Ar·(|Bm| + Br)
//! ```
//!
//! so the true interval product is enclosed by `<Am·Bm, |Am|·Br +
//! Ar·(|Bm|+Br)>` (this over-approximates the exact interval hull by at most
//! a factor 1.5 in radius — Rump 1999, Thm 2.2 — which the tightness test
//! checks against the scalar corner-product path).
//!
//! # Float soundness argument
//!
//! Everything above was real arithmetic; in f64 each piece rounds. Let
//! `u = 2^-53` and `γ_n = n·u/(1-n·u)` (Higham, *Accuracy and Stability of
//! Numerical Algorithms*, sec. 3.5): a length-k dot product computed in f64
//! round-to-nearest — in ANY summation order, with or without FMA — satisfies
//! `|fl(x·y) - x·y| <= γ_k · Σ|x_i||y_i|`. This covers conventional (blocked,
//! vectorized or threaded-by-output-partition) GEMMs. [`probed_matmul`] uses
//! the simple left-to-right fold explicitly, so it introduces no hidden
//! reduction order, Strassen-style rounded combinations, or unqualified
//! backend workers.
//!
//! Computed quantities (2-4 GEMMs; `Q`/`S` are skipped when the respective
//! radius is exactly zero, e.g. the constant weight side of Linear):
//!
//! ```text
//! Cm = fl(Am @ Bm)                 |Cm - (Am·Bm)|  <= γ_k · P*   per entry
//! P  = fl(|Am| @ |Bm|)             P* := (|Am|·|Bm|),  P >= P*(1-γ_k)
//! Q  = fl(|Am| @ Br)               Q* := (|Am|·Br),    Q >= Q*(1-γ_k)
//! S  = fl(Ar @ Bs)                 Bs := next_up(fl(|Bm|+Br)) >= |Bm|+Br,
//!                                  S* := (Ar·Bs) >= Ar·(|Bm|+Br),
//!                                  S >= S*(1-γ_k)
//! ```
//!
//! ## Rank-1 magnitude bound for POINT-B products (#rank1-radius)
//!
//! `P` feeds ONLY the `γ'` rounding term of the radius, so any per-entry
//! OVER-estimate of `P*` keeps the enclosure (it can only inflate the
//! radius). For a point right operand (`Br == 0` — the constant Linear
//! weight, the dominant mscn shape) the `P` GEMM is replaced by the rank-1
//! upper bound
//!
//! ```text
//! P̂_ij = up1( rs_i · cmax_j ),   rs_i   := directed-up Σ_l |Am|_il  >= Σ_l |Am|_il,
//!                                cmax_j := max_l |Bm|_lj             (exact),
//! P̂_ij >= rs_i·cmax_j >= (Σ_l |Am|_il)·(max_l |Bm|_lj) >= Σ_l |Am|_il·|Bm|_lj = P*_ij
//! ```
//!
//! (the middle inequality is elementwise `|Bm|_lj <= cmax_j` with
//! nonnegative weights `|Am|_il`; `rs_i` nudges every partial sum one ulp
//! toward +inf so no nearest rounding can land below the real sum; the final
//! `up1` covers the single product rounding). `P̂ >= P*` is a STRICTLY
//! stronger guarantee than the exact GEMM's `P >= P*(1-γ_k)`, so the radius
//! argument below holds a fortiori, at O(m·k + k·n + m·n) cost instead of an
//! O(m·n·k) GEMM. Tightness cost: `P̂` exceeds `P*` by at most the
//! column-spread of `|Bm|` (bounded by `k·max|Am|·max|Bm| / P*` relatively) —
//! but the whole `γ'·P̂` term stays ~`k·2^-53` RELATIVE to the operand
//! magnitudes (~1e-12 at k = 2048): invisible against the 1e-5-scale mscn
//! margins and against the S radius term on non-degenerate boxes. The a1
//! containment gate (fast ⊇ scalar) is unaffected: `P̂ >= P* >= P/(1+γ_k)`,
//! so the radius the containment argument needs shrinks by at most
//! `γ'·γ_k·P*` — third-order, far inside the ~γ_k headroom the γ' = γ_{4k+16}
//! choice leaves.
//!
//! The required real radius per entry is `R* = Q* + Ar·(|Bm|+Br) + γ_k·P*`
//! (midpoint-radius product radius plus the rounding of `Cm`). We return
//!
//! ```text
//! base = fl(Q + S)                       >= (Q+S)(1-u)
//! t    = fl(P + base)                    >= (P+Q+S)(1-u)^2
//! e    = up2( fl(γ' · t) )               >= γ'·(P+Q+S)·(1-u)^2,  γ' = γ_{4k+16}
//! η    = (12k+32)·2^-1074                (subnormal-operation guard, below)
//! r    = up2( fl(base + e + η) )         >= base + e + η
//! lo   = next_down( fl(Cm - r) )         <= Cm - r
//! hi   = next_up  ( fl(Cm + r) )         >= Cm + r
//! ```
//!
//! where `upN` steps N ulps toward +inf (each `next_up` of a nearest-rounded
//! value reaches or passes the real value, since nearest rounding is within
//! 0.5 ulp). Soundness needs `base + e >= R*`. Using `X* <= X/(1-γ_k)`:
//!
//! ```text
//! R*       <= (Q+S)/(1-γ_k) + γ_k·P/(1-γ_k)
//! base + e >= (Q+S)(1-u) + γ'(1-u)^2·(P+Q+S)
//! ```
//!
//! so it suffices that `γ'(1-u)^2 - u >= γ_k/(1-γ_k)` for both the `(Q+S)`
//! and `P` coefficients — already satisfied by `γ' = γ_{2k}`. We use the
//! LARGER `γ' = γ_{4k+16} ~ 4·γ_k` so that the fast interval additionally
//! CONTAINS the scalar corner-product path's interval per entry (soundness
//! gate a1): the scalar path's endpoints sit at most `~2·γ_k·Σmax|corner|
//! <= 2·γ_k·(P+Q+S)`-ish below/above the real hull (its own fl summation
//! plus its `γ_{k+2}` widening), the fast midpoint `Cm` is within `γ_k·P*`
//! of the real midpoint, and the mid-rad radius is >= the hull radius — so
//! `γ' >= ~3·γ_k` + margin makes containment hold with ~γ_k to spare. The
//! extra width is still only ~k·2^-53 RELATIVE (~4e-13 at k = 2048:
//! irrelevant against the 1e-5-scale mscn margins, and the tightness test
//! bounds it globally at < 2x scalar width on mscn-shaped boxes).
//!
//! The `η` term: a purely relative γ model is insufficient whenever an exact
//! elementary result lies in the subnormal range. Each of the <= 4 GEMMs
//! performs `k` multiplies and `k` reduction adds per entry (<= `8k`
//! operations total). We charge a full minimum subnormal for every one, then
//! reserve another `4k+32` operations for magnitude, midpoint/radius, and
//! final radius assembly. Thus the ABSOLUTE guard
//! `η = (12k+32)·2^-1074` covers product underflow, cancellation/addition
//! underflow, and any under-reporting of `P`, `Q`, or `S`; directed `next_up`
//! steps still cover their local operations independently. The floor is
//! ~1e-319 at `k = 2048`, invisible at normal magnitudes.
//!
//! Hence `[lo, hi] ⊇ Cm ± (R* + underflow) ⊇ [A]·[B]` — the true interval
//! product — for every entry. The midpoint/radius input split itself is made
//! outward-safe in [`split_mid_rad`] (radius widened 2 ulps past
//! `max(m-lo, hi-m)`).
//!
//! Any non-finite intermediate (overflow, NaN input) makes the kernel return
//! `None`; callers MUST fall back to the scalar path — the fast path never
//! guesses.
//!
//! # Kill-switch and shape gates
//!
//! `NY_F64_BLAS=0` disables the fast path everywhere (callers check
//! [`fast_gemm_enabled`]); shapes with `m·n·k <=` [`FAST_GEMM_MIN_VOLUME`]
//! or fewer than [`FAST_GEMM_MIN_ROWS`] left-side rows keep the scalar path
//! because the kernel's multiple weight-sized f64 passes are not worthwhile
//! for thin batches (see [`FAST_GEMM_MIN_ROWS`]); those shapes are served by
//! the unrolled scalar Linear loop in the cell instead. The environment
//! variable retains its historical name even though certified arithmetic no
//! longer dispatches to an opaque BLAS backend.

use ndarray::{Array2, ArrayView2, Zip};
use rayon::prelude::*;

use super::graph_ibp_f64_cell::{
    gamma_n, operation_underflow_floor, require_f64_interval_proof_environment, widen_up_n,
};

/// Minimum `m·n·k` volume for the fast path: below this the midpoint/radius
/// conversion + 3-4 GEMM dispatch overhead beats the scalar loop's locality.
pub(super) const FAST_GEMM_MIN_VOLUME: usize = 32_768;

/// Minimum LEFT-side row count `m` for the fast path. The Rump form makes
/// multiple weight-sized f64 passes, so thin per-clause-box shapes
/// (`m ∈ {1,2,3,6}`) stay on the one-pass scalar interval loop. At `m >= 16`
/// the rows also expose enough independent work for [`probed_matmul`] to use
/// the qualified Rayon pool.
pub(super) const FAST_GEMM_MIN_ROWS: usize = 16;

/// Batteries-included default-ON; `NY_F64_BLAS=0` is the kill-switch that
/// restores the scalar interval matmul everywhere.
pub(super) fn fast_gemm_enabled() -> bool {
    !std::env::var("NY_F64_BLAS").is_ok_and(|v| v == "0")
}

/// Matrix product whose arithmetic runs only on threads that just passed the
/// required binary64 round-to-nearest/gradual-underflow probe.
///
/// The workspace's `ndarray::dot` may dispatch to private
/// `matrixmultiply-threading` or Accelerate workers. Their FP control state
/// cannot be inspected, so proof arithmetic cannot use that route. This
/// implementation gives each output row to Rayon; every executing worker
/// probes immediately before its left-to-right length-`k` folds. `None`
/// means a shape/size error or an unsafe worker environment.
fn probed_matmul(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return None;
    }
    let a_standard = a.as_standard_layout();
    let b_standard = b.as_standard_layout();
    let a = a_standard.as_slice()?;
    let b = b_standard.as_slice()?;
    let len = m.checked_mul(n)?;
    if len == 0 {
        return Array2::from_shape_vec((m, n), Vec::new()).ok();
    }

    let mut output = vec![0.0_f64; len];
    output
        .par_chunks_mut(n)
        .enumerate()
        .try_for_each(|(i, row)| -> Option<()> {
            require_f64_interval_proof_environment().ok()?;
            let a_row = &a[i * k..(i + 1) * k];
            for (l, &a_value) in a_row.iter().enumerate() {
                let b_row = &b[l * n..(l + 1) * n];
                for (value, &b_value) in row.iter_mut().zip(b_row) {
                    *value += a_value * b_value;
                }
            }
            Some(())
        })?;
    Array2::from_shape_vec((m, n), output).ok()
}

/// Outward-safe midpoint/radius split of `[lo, hi]`:
/// `[lo, hi] ⊆ [mid - rad, mid + rad]` is GUARANTEED for the returned float
/// matrices. Returns `(mid, rad, is_point)`; `None` if any entry is
/// non-finite (NaN/inf input, or an inverted pair) — caller falls back.
///
/// - `mid = fl(0.5·lo + 0.5·hi)` (the halvings are exact except subnormals;
///   cannot overflow for finite inputs since `|mid| <= max(|lo|, |hi|)`).
/// - `rad = up2( max(fl(mid - lo), fl(hi - mid)) )`: each subtraction is one
///   nearest rounding, so one `next_up` reaches the real value; two add
///   margin. For an exactly-point entry (`lo == hi == mid`) the radius is
///   EXACTLY 0.0 so point operands (e.g. Linear weights) are detected and
///   their radius GEMMs skipped.
fn split_mid_rad(
    lo: &ArrayView2<'_, f64>,
    hi: &ArrayView2<'_, f64>,
) -> Option<(Array2<f64>, Array2<f64>, bool)> {
    let mid = Zip::from(lo)
        .and(hi)
        .map_collect(|&l, &h| 0.5 * l + 0.5 * h);
    let mut is_point = true;
    let rad = Zip::from(&mid).and(lo).and(hi).map_collect(|&m, &l, &h| {
        if l == h && m == l {
            0.0
        } else {
            is_point = false;
            widen_up_n((m - l).max(h - m), 2)
        }
    });
    let ok = mid.iter().all(|v| v.is_finite()) && rad.iter().all(|v| v.is_finite() && *v >= 0.0);
    if !ok {
        return None;
    }
    Some((mid, rad, is_point))
}

/// Directed-UPWARD row sums of `|A|`: `rs[i] >= Σ_j |A[i,j]|` in REAL
/// arithmetic. Every partial sum is nudged one ulp toward +inf after its
/// nearest-rounded add (which is within 0.5 ulp of the real sum), so by
/// induction no partial sum ever falls below the real one; `|v|` is exact.
/// `None` on a non-finite (overflowed) sum — caller falls back.
fn abs_rowsums_up(a: &Array2<f64>) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(a.nrows());
    for row in a.rows() {
        let mut s = 0.0f64;
        for &v in row {
            s = (s + v.abs()).next_up();
        }
        if !s.is_finite() {
            return None;
        }
        out.push(s);
    }
    Some(out)
}

/// Exact per-column maxima of a nonnegative matrix (`max` of finite f64 is
/// exact — no rounding). Used as the `cmax` factor of the rank-1 magnitude
/// bound; precomputed once per Linear weight by the batch cache.
pub(super) fn abs_colmax(b_abs: &ArrayView2<'_, f64>) -> Vec<f64> {
    let mut out = vec![0.0f64; b_abs.ncols()];
    for row in b_abs.rows() {
        for (o, &v) in out.iter_mut().zip(row) {
            if v > *o {
                *o = v;
            }
        }
    }
    out
}

/// Rump midpoint-radius interval GEMM: sound `[lo, hi]` enclosure of
/// `[a_lo, a_hi] @ [b_lo, b_hi]` for `[m,k] @ [k,n]` (see module docs for
/// the full soundness argument). Returns `None` (caller must fall back to
/// the scalar path) on shape mismatch or any non-finite intermediate.
pub(super) fn rump_interval_matmul(
    a_lo: ArrayView2<'_, f64>,
    a_hi: ArrayView2<'_, f64>,
    b_lo: ArrayView2<'_, f64>,
    b_hi: ArrayView2<'_, f64>,
) -> Option<(Array2<f64>, Array2<f64>)> {
    require_f64_interval_proof_environment().ok()?;
    let (m, k) = a_lo.dim();
    let (kb, n) = b_lo.dim();
    if a_hi.dim() != (m, k) || b_hi.dim() != (kb, n) || k != kb {
        return None;
    }
    let (am, ar, a_point) = split_mid_rad(&a_lo, &a_hi)?;
    let (bm, br, b_point) = split_mid_rad(&b_lo, &b_hi)?;
    let bm_abs = bm.mapv(f64::abs);

    if b_point {
        // Point B (Br == 0): Q skipped, Bs = |Bm| exactly, and the P GEMM is
        // replaced by the rank-1 magnitude bound (#rank1-radius, module
        // docs). Bit-identical to `rump_interval_matmul_point_b` on the same
        // operands: same rowsums, same (exact) column maxima, same GEMMs.
        let rs = abs_rowsums_up(&am)?;
        let cmax = abs_colmax(&bm_abs.view());
        let s = if a_point {
            None
        } else {
            Some(probed_matmul(ar.view(), bm_abs.view())?)
        };
        let cm = probed_matmul(am.view(), bm.view())?;
        return rump_combine(&cm, |i, j| (rs[i] * cmax[j]).next_up(), None, s, k);
    }

    let am_abs = am.mapv(f64::abs);
    // GEMM: Q = fl(|Am| @ Br).
    let q = Some(probed_matmul(am_abs.view(), br.view())?);
    // GEMM (skipped for a point A): S = fl(Ar @ Bs), Bs >= |Bm| + Br real.
    let s = if a_point {
        None
    } else if b_point {
        // Br == 0 exactly: |Bm| + Br = |Bm|, no rounding.
        Some(probed_matmul(ar.view(), bm_abs.view())?)
    } else {
        let bs = Zip::from(&bm_abs)
            .and(&br)
            .map_collect(|&mb, &rb| (mb + rb).next_up());
        Some(probed_matmul(ar.view(), bs.view())?)
    };
    // GEMM: midpoint product Cm = fl(Am @ Bm).
    let cm = probed_matmul(am.view(), bm.view())?;
    // GEMM: P = fl(|Am| @ |Bm|) — magnitude base of the Higham term.
    let p = probed_matmul(am_abs.view(), bm_abs.view())?;
    let p_s = p.as_slice()?;
    rump_combine(&cm, |i, j| p_s[i * n + j], q, s, k)
}

/// Rump interval GEMM against a PREPARED POINT right operand `Bm` (radius
/// exactly zero; `bm_abs = |Bm|` and its column maxima `bm_colmax`
/// precomputed by the caller — the cached f64 Linear weight,
/// #f64-batch-boxes). BIT-IDENTICAL to [`rump_interval_matmul`] with
/// `b_lo = b_hi = Bm` for any `Bm` whose entries halve exactly (every
/// f32-sourced f64 does: f32 subnormals become normal f64, so
/// `fl(0.5·b + 0.5·b) = b` and the split's radius is exactly 0): that call
/// computes `bm == Bm`, `bm_abs == |Bm|`, `b_point == true`, and takes the
/// identical rank-1 point-B branch (`max` is exact and order-free, so the
/// precomputed column maxima equal the on-the-fly ones). Skipping the
/// per-call weight split removes the O(k·n) conversion/split traffic that
/// dominated thin-m calls.
pub(super) fn rump_interval_matmul_point_b(
    a_lo: ArrayView2<'_, f64>,
    a_hi: ArrayView2<'_, f64>,
    bm: ArrayView2<'_, f64>,
    bm_abs: ArrayView2<'_, f64>,
    bm_colmax: &[f64],
) -> Option<(Array2<f64>, Array2<f64>)> {
    require_f64_interval_proof_environment().ok()?;
    let (m, k) = a_lo.dim();
    let (kb, n) = bm.dim();
    if a_hi.dim() != (m, k) || bm_abs.dim() != (kb, n) || k != kb || bm_colmax.len() != n {
        return None;
    }
    let (am, ar, a_point) = split_mid_rad(&a_lo, &a_hi)?;
    let rs = abs_rowsums_up(&am)?;
    // Q skipped (Br == 0); S = fl(Ar @ |Bm|) unless A is also a point.
    let s = if a_point {
        None
    } else {
        Some(probed_matmul(ar.view(), bm_abs)?)
    };
    let cm = probed_matmul(am.view(), bm)?;
    rump_combine(&cm, |i, j| (rs[i] * bm_colmax[j]).next_up(), None, s, k)
}

/// Shared tail of the Rump GEMM entries: the per-entry radius combination
/// (module docs). `mag(i, j)` is the Higham magnitude base for entry
/// `(i, j)` and must satisfy `mag >= (|Am|·|Bm|)_ij · (1-γ_k)(1-u)` in real
/// arithmetic — the exact `fl(|Am| @ |Bm|)` GEMM does (Higham lower bound),
/// and the rank-1 point-B bound `up1(rs_i·cmax_j) >= (|Am|·|Bm|)_ij` does a
/// fortiori. `q`/`s` are the optional radius GEMM results (skipped when the
/// respective operand is a point).
fn rump_combine(
    cm: &Array2<f64>,
    mag: impl Fn(usize, usize) -> f64,
    q: Option<Array2<f64>>,
    s: Option<Array2<f64>>,
    k: usize,
) -> Option<(Array2<f64>, Array2<f64>)> {
    // γ' = γ_{4k+16}: covers the γ_k dot-product bound of every GEMM, all
    // O(1) elementwise roundings, AND the scalar reference path's own fl
    // slack so the fast interval contains the scalar one (module docs).
    let relative_guard_terms = k.checked_mul(4)?.checked_add(16)?;
    let gamma = gamma_n(relative_guard_terms).ok()?;
    let (m, n) = cm.dim();

    let cm_s = cm.as_slice()?;
    let q_s = q.as_ref().and_then(|a| a.as_slice());
    let s_s = s.as_ref().and_then(|a| a.as_slice());
    if q.is_some() != q_s.is_some() || s.is_some() != s_s.is_some() {
        return None; // owned probed-matmul results are contiguous; defensive only
    }

    // Every one of the <=4 dots performs k multiplies and k reduction adds.
    // The extra 4k+32 charge covers magnitude and radius assembly. Keep this
    // separate from the tighter relative γ count: the absolute floor costs
    // nothing at normal magnitudes and must cover operations, not just terms.
    let underflow_operations = k.checked_mul(12)?.checked_add(32)?;
    let eta = operation_underflow_floor(underflow_operations).ok()?;

    let mut lo = vec![0.0f64; m * n];
    let mut hi = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let c = cm_s[idx];
            let base = q_s.map_or(0.0, |v| v[idx]) + s_s.map_or(0.0, |v| v[idx]);
            let t = mag(i, j) + base;
            let e = widen_up_n(gamma * t, 2); // >= γ'·t real
            let r = widen_up_n(base + e + eta, 2); // >= base + e + η real
            if !(c.is_finite() && r.is_finite() && r >= 0.0) {
                return None; // overflow anywhere -> scalar fallback
            }
            lo[idx] = (c - r).next_down();
            hi[idx] = (c + r).next_up();
        }
    }
    Some((
        Array2::from_shape_vec((m, n), lo).ok()?,
        Array2::from_shape_vec((m, n), hi).ok()?,
    ))
}

// ---------------------------------------------------------------------------
// Soundness gates: enclosure vs the scalar path AND vs sampled true products,
// tightness vs the scalar path. If ANY of these fails, the fast path must be
// gated OFF — never ship a non-enclosing kernel.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::super::graph_ibp_f64_cell::{eval_matmul_scalar, Interval64};
    use super::*;
    use crate::layers::MatMulLayer;
    use ndarray::ArrayD;

    /// Deterministic xorshift stream — no extra dev-dep, reproducible seeds.
    struct Rng(u64);
    impl Rng {
        fn next_unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_unit() * (hi - lo + 1) as f64) as usize
        }
    }

    #[test]
    fn probed_matmul_uses_the_documented_left_to_right_fold() {
        let a =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -4.0, 5.0, -6.0]).expect("shape");
        let b =
            Array2::from_shape_vec((3, 2), vec![7.0, 8.0, -9.0, 10.0, 11.0, -12.0]).expect("shape");
        let product = probed_matmul(a.view(), b.view()).expect("qualified worker product");
        assert_eq!(
            product,
            Array2::from_shape_vec((2, 2), vec![22.0, -8.0, -139.0, 90.0]).unwrap()
        );
    }

    /// Random interval matrix: mixed signs, magnitude scale `scale`,
    /// `point_frac` of entries degenerate (lo == hi).
    fn random_interval(
        rng: &mut Rng,
        rows: usize,
        cols: usize,
        scale: f64,
        point_frac: f64,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut lo = Array2::zeros((rows, cols));
        let mut hi = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let c = (rng.next_unit() * 2.0 - 1.0) * scale;
                let w = if rng.next_unit() < point_frac {
                    0.0
                } else {
                    rng.next_unit() * 0.5 * scale
                };
                lo[[i, j]] = c - w;
                hi[[i, j]] = c + w;
            }
        }
        (lo, hi)
    }

    fn scalar_reference(
        a_lo: &Array2<f64>,
        a_hi: &Array2<f64>,
        b_lo: &Array2<f64>,
        b_hi: &Array2<f64>,
    ) -> Interval64 {
        let mk = |a: &Array2<f64>| ArrayD::from(a.clone().into_dyn());
        let a = Interval64 {
            lower: mk(a_lo),
            upper: mk(a_hi),
        };
        let b = Interval64 {
            lower: mk(b_lo),
            upper: mk(b_hi),
        };
        eval_matmul_scalar(&MatMulLayer::new(false, None), &a, &b).expect("scalar matmul")
    }

    /// Uniform sample inside the interval matrix.
    fn sample_inside(rng: &mut Rng, lo: &Array2<f64>, hi: &Array2<f64>) -> Array2<f64> {
        let mut out = lo.clone();
        for (o, (&l, &h)) in out.iter_mut().zip(lo.iter().zip(hi.iter())) {
            *o = l + (h - l) * rng.next_unit();
        }
        out
    }

    /// ENCLOSURE GATE (a): across many seeds/shapes/magnitudes, the fast
    /// product must CONTAIN the scalar path's interval AND every sampled
    /// true product.
    #[test]
    fn rump_encloses_scalar_and_sampled_products() {
        // (seed, scale, point_frac); dims drawn 1..=300 from the seed.
        let configs: &[(u64, f64, f64)] = &[
            (0x9E3779B97F4A7C15, 1.0, 0.0),
            (0xDEADBEEFCAFEF00D, 1.0, 0.5),
            (0x0123456789ABCDEF, 1e-8, 0.2),
            (0xFEDCBA9876543210, 1e8, 0.0),
            (0x1111111111111111, 1e-150, 0.3),
            (0x2222222222222222, 1e120, 0.1),
            (0x3333333333333333, 1.0, 1.0), // both operands point matrices
            (0x4444444444444444, 1e-300, 0.0), // subnormal-adjacent tiny
            (0x5555555555555555, 3.7, 0.9),
            (0x6666666666666666, 42.0, 0.0),
            (0x7777777777777777, 1e4, 0.5),
            (0x8888888888888888, 1e-40, 0.0),
            (0xAAAAAAAAAAAAAAAA, 1.0, 0.0),
            (0xBBBBBBBBBBBBBBBB, 7e2, 0.25),
            (0xCCCCCCCCCCCCCCCC, 1e60, 0.0),
            (0xDDDDDDDDDDDDDDDD, 1e-3, 0.0),
        ];
        let mut total_samples = 0usize;
        for &(seed, scale, point_frac) in configs {
            let mut rng = Rng(seed);
            let (m, k, n) = (rng.range(1, 300), rng.range(1, 300), rng.range(1, 300));
            let (a_lo, a_hi) = random_interval(&mut rng, m, k, scale, point_frac);
            let (b_lo, b_hi) = random_interval(&mut rng, k, n, scale, point_frac);
            let (f_lo, f_hi) =
                rump_interval_matmul(a_lo.view(), a_hi.view(), b_lo.view(), b_hi.view())
                    .expect("fast kernel declined finite input");
            // (a1) fast ⊇ scalar.
            let scalar = scalar_reference(&a_lo, &a_hi, &b_lo, &b_hi);
            for i in 0..m {
                for j in 0..n {
                    let (sl, sh) = (scalar.lower[[i, j]], scalar.upper[[i, j]]);
                    let (fl_, fh) = (f_lo[[i, j]], f_hi[[i, j]]);
                    assert!(
                        fl_ <= sl && fh >= sh,
                        "seed {seed:#x} ({m}x{k}x{n}, scale {scale:e}): fast \
                         [{fl_}, {fh}] does not contain scalar [{sl}, {sh}] at [{i},{j}]"
                    );
                }
            }
            // (a2) fast ⊇ sampled true products (fl f64 matmul of samples;
            // its rounding is ~γ_k·|A||B|, far inside the kernel's γ_{4k+16}
            // slack).
            let n_samples = 32; // 16 configs x 32 = 512 sampled true products
            for _ in 0..n_samples {
                let a_s = sample_inside(&mut rng, &a_lo, &a_hi);
                let b_s = sample_inside(&mut rng, &b_lo, &b_hi);
                let prod = a_s.dot(&b_s);
                for i in 0..m {
                    for j in 0..n {
                        let v = prod[[i, j]];
                        assert!(
                            f_lo[[i, j]] <= v && v <= f_hi[[i, j]],
                            "seed {seed:#x} ({m}x{k}x{n}): sample {v} escapes \
                             [{}, {}] at [{i},{j}]",
                            f_lo[[i, j]],
                            f_hi[[i, j]]
                        );
                    }
                }
                total_samples += 1;
            }
        }
        assert!(
            total_samples >= 500,
            "sample budget regressed: {total_samples}"
        );
    }

    /// ENCLOSURE GATE (a), degenerate dims: 1x1x1 and thin/flat shapes.
    #[test]
    fn rump_encloses_degenerate_dims() {
        let mut rng = Rng(0x517CC1B727220A95);
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (1, 300, 1),
            (300, 1, 300),
            (1, 1, 300),
            (2, 3, 1),
        ] {
            let (a_lo, a_hi) = random_interval(&mut rng, m, k, 5.0, 0.3);
            let (b_lo, b_hi) = random_interval(&mut rng, k, n, 5.0, 0.3);
            let (f_lo, f_hi) =
                rump_interval_matmul(a_lo.view(), a_hi.view(), b_lo.view(), b_hi.view())
                    .expect("fast kernel declined finite input");
            let scalar = scalar_reference(&a_lo, &a_hi, &b_lo, &b_hi);
            for i in 0..m {
                for j in 0..n {
                    assert!(f_lo[[i, j]] <= scalar.lower[[i, j]]);
                    assert!(f_hi[[i, j]] >= scalar.upper[[i, j]]);
                }
            }
            for _ in 0..100 {
                let a_s = sample_inside(&mut rng, &a_lo, &a_hi);
                let b_s = sample_inside(&mut rng, &b_lo, &b_hi);
                let prod = a_s.dot(&b_s);
                for i in 0..m {
                    for j in 0..n {
                        assert!(f_lo[[i, j]] <= prod[[i, j]] && prod[[i, j]] <= f_hi[[i, j]]);
                    }
                }
            }
        }
    }

    /// TIGHTNESS GATE (b): on typical mscn shapes (k = 2048, near-point and
    /// moderately wide boxes) the fast width stays within 2x of scalar.
    #[test]
    fn rump_tightness_within_2x_of_scalar_mscn_shapes() {
        for &(seed, width) in &[(0xABCDEF0123456789u64, 1e-9f64), (0x13579BDF2468ACE0, 0.1)] {
            let mut rng = Rng(seed);
            let (m, k, n) = (24, 2048, 32);
            let mut a_lo = Array2::zeros((m, k));
            let mut a_hi = Array2::zeros((m, k));
            for i in 0..m {
                for j in 0..k {
                    let c = rng.next_unit() * 2.0 - 1.0;
                    a_lo[[i, j]] = c - width * rng.next_unit();
                    a_hi[[i, j]] = c + width * rng.next_unit();
                }
            }
            // Point B — the Linear-weight case (mscn's dominant op).
            let mut b = Array2::zeros((k, n));
            for v in b.iter_mut() {
                *v = rng.next_unit() * 2.0 - 1.0;
            }
            let (f_lo, f_hi) = rump_interval_matmul(a_lo.view(), a_hi.view(), b.view(), b.view())
                .expect("fast kernel declined");
            let scalar = scalar_reference(&a_lo, &a_hi, &b, &b);
            for i in 0..m {
                for j in 0..n {
                    let wf = f_hi[[i, j]] - f_lo[[i, j]];
                    let ws = scalar.upper[[i, j]] - scalar.lower[[i, j]];
                    assert!(
                        wf <= 2.0 * ws + 1e-290,
                        "width {wf} > 2x scalar {ws} at [{i},{j}] (width class {width})"
                    );
                }
            }
        }
    }

    /// Live x live tightness: within 2x of scalar (mid-rad hull excess is
    /// bounded by 1.5x — Rump 1999 Thm 2.2 — plus the γ' slack).
    #[test]
    fn rump_tightness_within_2x_of_scalar_live_operands() {
        let mut rng = Rng(0x0F1E2D3C4B5A6978);
        let (m, k, n) = (16, 512, 16);
        let (a_lo, a_hi) = random_interval(&mut rng, m, k, 1.0, 0.0);
        let (b_lo, b_hi) = random_interval(&mut rng, k, n, 1.0, 0.0);
        let (f_lo, f_hi) = rump_interval_matmul(a_lo.view(), a_hi.view(), b_lo.view(), b_hi.view())
            .expect("fast kernel declined");
        let scalar = scalar_reference(&a_lo, &a_hi, &b_lo, &b_hi);
        for i in 0..m {
            for j in 0..n {
                let wf = f_hi[[i, j]] - f_lo[[i, j]];
                let ws = scalar.upper[[i, j]] - scalar.lower[[i, j]];
                assert!(wf <= 2.0 * ws + 1e-290, "width {wf} > 2x scalar {ws}");
            }
        }
    }

    /// Point x point stays gamma-tight in the rank-1 magnitude measure
    /// (#rank1-radius): the radius is the `γ'·P̂` term only, with
    /// `P̂_ij ~ rowsum|A|_i · colmax|B|_j` — still `~k·2^-53` RELATIVE to the
    /// operand magnitudes (the intentional inflation vs the exact `|A|·|B|`
    /// is bounded by the column spread of `|B|`, and is invisible against
    /// the 1e-5-scale mscn margins this kernel serves).
    #[test]
    fn rump_point_times_point_is_relative_gamma_tight() {
        let mut rng = Rng(0x243F6A8885A308D3);
        let (m, k, n) = (8, 256, 8);
        let (a, _) = random_interval(&mut rng, m, k, 1.0, 1.0);
        let (b, _) = random_interval(&mut rng, k, n, 1.0, 1.0);
        let (f_lo, f_hi) = rump_interval_matmul(a.view(), a.view(), b.view(), b.view())
            .expect("fast kernel declined");
        let gamma = gamma_n(4 * k + 16).unwrap();
        let rs = abs_rowsums_up(&a).unwrap();
        let cmax = abs_colmax(&b.mapv(f64::abs).view());
        for i in 0..m {
            for j in 0..n {
                let w = f_hi[[i, j]] - f_lo[[i, j]];
                assert!(
                    w <= 4.0 * gamma * rs[i] * cmax[j] + 1e-290,
                    "point-point width {w} not gamma-tight in the rank-1 measure"
                );
                // Absolute sanity: never worse than the trivial k·max·max
                // magnitude bound.
                assert!(w <= 4.0 * gamma * (k as f64) + 1e-290);
            }
        }
    }

    /// #rank1-radius soundness gate: the rank-1 magnitude bound
    /// `up1(rs_i · cmax_j)` DOMINATES the true product `(|A|·|B|)_ij` on
    /// 10k+ random entries, including denormal and large-dynamic-range
    /// magnitudes. The reference is a directed-DOWNWARD sum (every partial
    /// sum nudged one ulp toward -inf), which is <= the real product, so
    /// `P̂ >= down_sum` is implied by the claimed `P̂ >= P*` and the test can
    /// never spuriously fail while catching any index/transpose/rounding bug.
    #[test]
    fn rank1_magnitude_bound_dominates_directed_lower_product() {
        let mut rng = Rng(0x5DEE_CE66_D202_6717);
        let mut checked = 0usize;
        for round in 0..60 {
            let (m, k, n) = (rng.range(1, 24), rng.range(1, 96), rng.range(1, 24));
            let mut a = Array2::<f64>::zeros((m, k));
            let mut b = Array2::<f64>::zeros((k, n));
            for v in a.iter_mut().chain(b.iter_mut()) {
                // Log-uniform magnitudes across ~600 orders of magnitude:
                // hits subnormals (~1e-320) and near-overflow (~1e280).
                let exp = rng.next_unit() * 600.0 - 320.0;
                let sign = if rng.next_unit() < 0.5 { -1.0 } else { 1.0 };
                *v = sign * 10f64.powf(exp);
            }
            let Some(rs) = abs_rowsums_up(&a) else {
                continue; // overflowed rowsum: kernel would fall back
            };
            let cmax = abs_colmax(&b.mapv(f64::abs).view());
            for i in 0..m {
                for j in 0..n {
                    let p_hat = (rs[i] * cmax[j]).next_up();
                    // Directed-down reference: <= Σ_l |a_il||b_lj| real.
                    let mut down = 0.0f64;
                    for l in 0..k {
                        down = (down + (a[[i, l]] * b[[l, j]]).abs().next_down()).next_down();
                    }
                    assert!(
                        p_hat >= down,
                        "round {round}: rank-1 bound {p_hat} < directed-down product {down} \
                         at [{i},{j}] ({m}x{k}x{n})"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked >= 10_000, "coverage regressed: {checked} entries");
    }

    /// #rank1-radius EXACT gold oracle (adversarial-verifier gate): the
    /// rank-1 magnitude bound `up1(rs_i · cmax_j)` dominates the TRUE
    /// rational product `Σ_l |a_il|·|b_lj|` computed in exact BigRational
    /// arithmetic (same dev-dep precedent as the #vnncomp-aw-soundness gold
    /// oracle). This is strictly stronger than the directed-down float
    /// reference above, which was measured to MISS marginal (~k/2-ulp)
    /// under-estimation injections (e.g. plain nearest row summation) that
    /// this oracle falsifies outright. Regimes: denormals, mixed 1e±150
    /// dynamic range, negative zeros, near-1 ulp-adversarial ramps, and a
    /// k = 2048 production-width row.
    #[test]
    fn rank1_magnitude_bound_dominates_exact_rational() {
        use num_rational::BigRational;
        let exact = |v: f64| BigRational::from_float(v).expect("finite");
        let mut rng = Rng(0x000E_84C7_2026_0717);
        let mut checked = 0usize;
        let gen_val = |mode: u32, rng: &mut Rng| -> f64 {
            let sign = if rng.next_unit() < 0.5 { -1.0 } else { 1.0 };
            match mode {
                // subnormals (ulp-uniform)
                0 => sign * f64::from_bits((rng.next_unit() * 4.5e15) as u64 + 1),
                // mixed magnitudes 1e-150..1e150
                1 => sign * 10f64.powf(rng.next_unit() * 300.0 - 150.0),
                // negative zero / exact zero
                2 => {
                    if rng.next_unit() < 0.5 {
                        -0.0
                    } else {
                        0.0
                    }
                }
                // near-1 ulp-adversarial ramp (worst nearest-rounding sums)
                _ => sign * (1.0 + rng.next_unit() * 2f64.powi(-30)),
            }
        };
        for round in 0..40 {
            // Rounds 0-3 are the FORCED falsifier regime for sub-ulp
            // accumulation bugs: k = 2048 near-1 ramps against an all-ones
            // column (a plain-nearest row sum drifts ~k/2 ulps below the
            // real sum here and up1 cannot recover it — measured 47%
            // falsification against that injection; random modes miss it).
            let forced = round < 4;
            let k = if forced { 2048 } else { 1 + (rng.range(1, 96)) };
            let mode_a = if forced {
                3
            } else {
                (rng.next_unit() * 4.0) as u32
            };
            let mode_b = (rng.next_unit() * 4.0) as u32;
            let row: Vec<f64> = (0..k).map(|_| gen_val(mode_a, &mut rng)).collect();
            let col: Vec<f64> = if forced {
                vec![1.0; k]
            } else {
                (0..k).map(|_| gen_val(mode_b, &mut rng)).collect()
            };
            let a = Array2::from_shape_vec((1, k), row.clone()).unwrap();
            let Some(rs) = abs_rowsums_up(&a) else {
                continue; // overflowed rowsum: kernel falls back
            };
            let b = Array2::from_shape_vec((k, 1), col.clone()).unwrap();
            let cmax = abs_colmax(&b.mapv(f64::abs).view());
            let p_hat = (rs[0] * cmax[0]).next_up();
            if !p_hat.is_finite() {
                continue; // kernel falls back on an infinite radius
            }
            let mut p_star = BigRational::from_float(0.0).unwrap();
            for (av, bv) in row.iter().zip(col.iter()) {
                p_star += exact(av.abs()) * exact(bv.abs());
            }
            assert!(
                exact(p_hat) >= p_star,
                "round {round} (k={k}, modes {mode_a}/{mode_b}): rank-1 bound {p_hat:e} \
                 is EXACTLY below the true rational product"
            );
            checked += 1;
        }
        assert!(checked >= 30, "coverage regressed: {checked} rounds");
    }

    /// The prepared point-B entry is BIT-IDENTICAL to the generic entry's
    /// point-B branch on the same operands (the batch cache contract).
    #[test]
    fn point_b_entry_bit_identical_to_generic() {
        let mut rng = Rng(0x0BAD_F00D_2026_0717);
        for &(m, k, n) in &[(3usize, 64usize, 48usize), (24, 96, 40), (1, 7, 5)] {
            let (a_lo, a_hi) = random_interval(&mut rng, m, k, 2.0, 0.3);
            // f32-sourced weight: halves exactly, so the generic split's
            // radius is exactly zero (the cache contract's precondition).
            let mut b = Array2::<f64>::zeros((k, n));
            for v in b.iter_mut() {
                *v = f64::from(((rng.next_unit() * 2.0 - 1.0) * 3.0) as f32);
            }
            let b_abs = b.mapv(f64::abs);
            let cmax = abs_colmax(&b_abs.view());
            let generic = rump_interval_matmul(a_lo.view(), a_hi.view(), b.view(), b.view())
                .expect("generic declined");
            let prepared = rump_interval_matmul_point_b(
                a_lo.view(),
                a_hi.view(),
                b.view(),
                b_abs.view(),
                &cmax,
            )
            .expect("point_b declined");
            let bits = |a: &Array2<f64>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&generic.0), bits(&prepared.0), "{m}x{k}x{n} lower");
            assert_eq!(bits(&generic.1), bits(&prepared.1), "{m}x{k}x{n} upper");
        }
    }

    /// Non-finite inputs decline (caller falls back to scalar).
    #[test]
    fn rump_declines_non_finite_and_overflow() {
        let inf = Array2::from_elem((8, 8), f64::INFINITY);
        let one = Array2::from_elem((8, 8), 1.0);
        assert!(rump_interval_matmul(one.view(), inf.view(), one.view(), one.view()).is_none());
        let nan = Array2::from_elem((8, 8), f64::NAN);
        assert!(rump_interval_matmul(nan.view(), nan.view(), one.view(), one.view()).is_none());
        // Finite inputs whose product overflows: decline, don't emit NaN.
        let big = Array2::from_elem((8, 8), 1e300);
        assert!(rump_interval_matmul(big.view(), big.view(), big.view(), big.view()).is_none());
    }

    /// Kill-switch: NY_F64_BLAS=0 gates the fast path off. (Serialized +
    /// restored via the blessed env choke point; the fast/scalar paths agree
    /// on enclosure either way.)
    #[test]
    fn fast_gemm_kill_switch() {
        ny_test_utils::env::with_env_edits(|env| {
            env.set("NY_F64_BLAS", "0");
            assert!(!fast_gemm_enabled());
            env.remove("NY_F64_BLAS");
            assert!(fast_gemm_enabled());
        });
    }

    /// Split guarantees `[lo, hi] ⊆ [mid - rad, mid + rad]`, radius exactly
    /// zero for point entries.
    #[test]
    fn split_mid_rad_is_outward_and_point_exact() {
        let mut rng = Rng(0x452821E638D01377);
        for _ in 0..200 {
            let scale = 10f64.powf((rng.next_unit() * 60.0) - 30.0);
            let l = (rng.next_unit() * 2.0 - 1.0) * scale;
            let h = l + rng.next_unit() * scale;
            let lo = Array2::from_elem((1, 1), l);
            let hi = Array2::from_elem((1, 1), h);
            let (mid, rad, _) = split_mid_rad(&lo.view(), &hi.view()).unwrap();
            let (m, r) = (mid[[0, 0]], rad[[0, 0]]);
            assert!(
                m - r <= l && h <= m + r,
                "split not outward: [{l}, {h}] vs {m}±{r}"
            );
        }
        let p = Array2::from_elem((2, 2), 0.1);
        let (mid, rad, is_point) = split_mid_rad(&p.view(), &p.view()).unwrap();
        assert!(is_point);
        assert_eq!(mid[[0, 0]], 0.1);
        assert_eq!(rad[[0, 0]], 0.0);
        // Subnormal point entries also stay outward (halving rounds there).
        let s = Array2::from_elem((1, 1), 3e-323);
        let (mid, rad, _) = split_mid_rad(&s.view(), &s.view()).unwrap();
        assert!(mid[[0, 0]] - rad[[0, 0]] <= 3e-323 && 3e-323 <= mid[[0, 0]] + rad[[0, 0]]);
    }
}
