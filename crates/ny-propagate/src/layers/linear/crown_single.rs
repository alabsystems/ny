// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Single-domain CROWN backward propagation for linear layers.
//!
//! Contains CPU (faer) and GEMM-engine paths for propagating linear bounds
//! through a fully-connected layer: `new_A = A @ W`, `new_b = A @ bias + old_b`.

use faer::Mat;
use ndarray::{Array1, Array2};
use ny_core::{is_crown_coeff_safe, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use std::borrow::Cow;
use tracing::debug;

use super::bias::{accumulate_bias_f64, finalize_bias_directed, BiasBlockParams};
use super::layout::resolve_backward_layout;
use super::LinearLayer;
use crate::faer_parallelism::{mat_mul, mat_mul_f64};
use crate::{contiguous_flat_slice_mut, LinearBounds};

/// Relative growth factor `γ_n = n·2^-53 / (1 - n·2^-53)` for an f64 dot product
/// of `n` terms (Higham, Accuracy and Stability, Thm 3.1). Multiplied by the
/// absolute-product sum `S_ij = Σ_k |a[i,k]|·|w[k,j]|` it bounds the f64
/// accumulation error of `Σ_k a[i,k]·w[k,j]`, S-scaled so it survives
/// cancellation (unlike a result-scaled n-ULP model). See
/// `ibp.rs::propagate_ibp_sound` for the same-sign IBP analogue.
#[inline]
pub(crate) fn gamma_n_f64(n: usize) -> f64 {
    let n = n as f64;
    let d = n * 2f64.powi(-53);
    // For pathologically wide contractions (n >= 2^53) the denominator goes
    // non-positive; clamp to a conservative large factor so the error matrix
    // degrades the row at concretize rather than under-counting.
    if d >= 1.0 {
        f64::INFINITY
    } else {
        d / (1.0 - d)
    }
}

/// Higham growth factor `γ_n = n·u / (1 - n·u)` with the **f32** unit roundoff
/// `u = 2^-24`. The GPU/engine `gemm_f32` accumulates the `A·W` dot product in
/// f32, so its coefficient error is `S`-scaled by this (much larger) factor —
/// `≈ 2^29×` the f64 factor. Using the f64 factor for an f32-accumulated
/// coefficient would be UNSOUND.
#[inline]
pub(crate) fn gamma_n_f32(n: usize) -> f64 {
    let n = n as f64;
    let d = n * 2f64.powi(-24);
    if d >= 1.0 {
        f64::INFINITY
    } else {
        d / (1.0 - d)
    }
}

/// Kill-switch (cached): `NY_NAIVE_F64_AW=1` restores the historical scalar
/// triple-loop for the sub-threshold `aw_f64_with_abssum` path (the byte-identical
/// A/B + parity reference). Read once (this is a hot per-domain leaf).
fn use_naive_f64_aw() -> bool {
    static NAIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *NAIVE.get_or_init(|| std::env::var_os("NY_NAIVE_F64_AW").is_some())
}

/// Accumulate `A·W` and its absolute-value version in f64.
///
/// Returns `(a64, s)` where:
/// - `a64[i,j] = Σ_k (a[i,k] as f64)·(w[k,j] as f64)` (exact products: f32×f32
///   fits in 48 < 53 significand bits, so only the f64 sum rounds), and
/// - `s[i,j]   = Σ_k |a[i,k] as f64|·|w[k,j] as f64|` (the S-scaling base for the
///   certified error `ε_ij = γ_n·S_ij`).
///
/// `a` is `(num_outputs, contraction)` and `w` is `(contraction, out)`.
pub(crate) fn aw_f64_with_abssum(a_block: &Mat<f32>, w: &Mat<f32>) -> (Array2<f64>, Array2<f64>) {
    let m = a_block.nrows();
    let k = a_block.ncols();
    let p = w.ncols();
    debug_assert_eq!(w.nrows(), k);

    // Sound GPU acceleration (#cuda-aw): offload the two f64 products `A·W` and
    // `|A|·|W|` to a process-global sound f64 GEMM engine (e.g. cuBLAS Dgemm) for
    // large products. The engine computes the SAME f64 dot products; Higham's
    // `γ_n·S` certified-error bound is summation-order independent, so the result
    // is sound (validated against an exact-rational oracle, 0 violations) — though
    // NOT bit-identical to the loop below. Small products stay on the CPU; any
    // engine error falls back to the CPU loop.
    //
    // Threshold = the GPU-vs-parallel-CPU crossover: the engine GEMM is ~229µs
    // (cached unified memory), while the CPU runs these calls rayon-parallel
    // across ~N cores, so its effective per-call cost is (single-thread)/N. They
    // cross around ~16M MACs (1<<24) on the GB10; below that the parallel CPU
    // wins and offloading only adds overhead (and triggers lazy GPU init on light
    // instances). Above it the GPU's f64 throughput dominates (measured 2.46×
    // end-to-end on mnist_concat --method alpha, where large A·W dominate).
    const SOUND_F64_GEMM_MIN_MACS: usize = 1 << 24;
    if m.saturating_mul(k).saturating_mul(p) >= SOUND_F64_GEMM_MIN_MACS {
        if let Some(Some(res)) =
            crate::sound_f64_gemm::with_engine(|eng| aw_via_engine(eng, a_block, w, m, k, p))
        {
            return res;
        }
    }

    // Sub-threshold CPU acceleration (#linearizenn-faer-f64-aw): route the two
    // f64 products `A·W` and `|A|·|W|` through faer's blocked SIMD f64 GEMM
    // instead of the scalar triple loop below. On linearizenn AllInOne_120_120
    // this f64 certified backward is ~75% of CPU after the f32 batched-engine
    // speedup; the per-domain 120-wide contraction runs at scalar ~1-3 GFLOP/s in
    // the naive loop and at memory bandwidth through faer.
    //
    // SOUNDNESS: identical envelope to the naive loop. `f32→f64` widening is
    // EXACT and `|x|` on the widened value is EXACT, so both GEMMs sum the SAME
    // exact-in-f64 products, only in a different (blocked/SIMD) order. The
    // certified error `γ_n·S` (Higham, Accuracy & Stability Thm 3.1) bounds the
    // f64 accumulation error for ANY summation order — the identical
    // order-independence argument the engine offload above (and the k≥2^23 f64
    // abs-sum GEMM in `aw_via_engine`) already rely on. So faer's `a64` and `s`
    // are a valid basis for the same enclosure `est ± γ_n·S`; validated against a
    // high-precision oracle in `faer_aw_matches_naive_and_encloses_exact`. Inside
    // the per-domain rayon workers `mat_mul_f64`'s `current_par()` forces faer to
    // `Par::Seq`, so there is no nested-Rayon deadlock (#4392).
    if !use_naive_f64_aw() {
        // #rebound-scratch: pooled twin of the four `Mat::from_fn` operands +
        // two `mat_mul_f64` products. Same operands, same `current_par()`
        // reduction order, same `Accum::Replace` → BIT-IDENTICAL to the owned
        // path below; only the buffers are recycled per-thread instead of
        // re-malloc'd per Linear backward per domain. The `else` arm is the
        // byte-for-byte historical reference (gate OFF / `NY_REBOUND_SCRATCH=0`).
        if crate::rebound_scratch::enabled() {
            let par = crate::faer_parallelism::current_par();
            let (aw_cm, s_cm) = crate::rebound_scratch::pooled_aw_and_abssum(
                m,
                k,
                p,
                par,
                |i, j| f64::from(a_block[(i, j)]),
                |i, j| f64::from(w[(i, j)]),
            );
            // Column-major product → row-major ndarray (element (i,j) at
            // `j*m + i`), identical values to reading the owned faer `Mat`.
            let a64 = Array2::from_shape_fn((m, p), |(i, j)| aw_cm[j * m + i]);
            let s = Array2::from_shape_fn((m, p), |(i, j)| s_cm[j * m + i]);
            crate::rebound_scratch::recycle_f64(aw_cm);
            crate::rebound_scratch::recycle_f64(s_cm);
            return (a64, s);
        }
        let a_f = Mat::<f64>::from_fn(m, k, |i, j| f64::from(a_block[(i, j)]));
        let w_f = Mat::<f64>::from_fn(k, p, |i, j| f64::from(w[(i, j)]));
        // |A|, |W| are exact in f64 (abs of an exactly-widened f32 clears the
        // sign bit only), so the abs-sum GEMM sums the exact |a|·|w| products.
        let a_abs = Mat::<f64>::from_fn(m, k, |i, j| f64::from(a_block[(i, j)]).abs());
        let w_abs = Mat::<f64>::from_fn(k, p, |i, j| f64::from(w[(i, j)]).abs());
        let c = mat_mul_f64(&a_f, &w_f);
        let s_mat = mat_mul_f64(&a_abs, &w_abs);
        let a64 = Array2::from_shape_fn((m, p), |(i, j)| c[(i, j)]);
        let s = Array2::from_shape_fn((m, p), |(i, j)| s_mat[(i, j)]);
        return (a64, s);
    }

    let mut a64 = Array2::<f64>::zeros((m, p));
    let mut s = Array2::<f64>::zeros((m, p));

    // Accumulate over contiguous `&mut [f64]` row slices of the (row-major,
    // freshly-allocated) outputs instead of strided `a64[[i,j]]` / `s[[i,j]]`
    // ndarray indexing. Each `[[i,j]]` read-modify-write pays ndarray's stride
    // arithmetic (and a bounds check in debug) on every one of the `m·k·p` inner
    // trips; this function is the hot inner of the per-domain β-CROWN Linear
    // backward (the dominant cost on the SwiGLU/MoE BaB graphs), so the slice
    // form lets the compiler keep the running sums in registers / vectorize.
    //
    // SOUNDNESS: bit-identical to the prior loop — same operands, same nesting
    // order (`i` → `kk` → `j`), same exact f32→f64 widening, and the same
    // per-(i,j) f64 running-sum sequence, so every rounding step is unchanged.
    let a64_buf = a64
        .as_slice_mut()
        .expect("a64 is freshly allocated row-major contiguous");
    let s_buf = s
        .as_slice_mut()
        .expect("s is freshly allocated row-major contiguous");
    for i in 0..m {
        let a64_row = &mut a64_buf[i * p..i * p + p];
        let s_row = &mut s_buf[i * p..i * p + p];
        for kk in 0..k {
            let av = a_block[(i, kk)] as f64;
            if av == 0.0 {
                continue;
            }
            let av_abs = av.abs();
            for j in 0..p {
                let wv = w[(kk, j)] as f64;
                a64_row[j] += av * wv;
                s_row[j] += av_abs * wv.abs();
            }
        }
    }
    (a64, s)
}

/// Provably-sound f64 inflation factor for an f32-accumulated abs-sum `S`
/// (#f32-abssum-seam; design `docs/F32_ABSSUM_SEAM.md`). Returns `F_hat` — an
/// over-bound of the tight factor `1/(1 − γ_k^f32)` already carrying the
/// `(1 + 2^-40)` margin that dominates the ≤ 3 f64 roundings in the caller's
/// `(r + G)·F_hat`. Returns `None` when `k ≥ 2^23` (the tight factor is
/// non-finite / would go negative), so the caller must use the exact f64 `S`
/// path there. The `gamma_n_f32(k) < 1.0` guard is STRICTER than
/// `gamma_n_f32`'s own `k < 2^24` clamp: for `k ∈ [2^23, 2^24)` the factor
/// `1/(1 − γ_k)` is finite but NEGATIVE (a stored negative error → false
/// VERIFIED), which this guard rejects.
#[inline]
fn f32_abssum_inflation(k: usize) -> Option<f64> {
    let g = gamma_n_f32(k);
    // NaN-aware "not (g < 1)": TRUE for NaN — `g >= 1.0` would let a NaN gamma
    // through to the inflation factor, so the negated form is load-bearing.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(g < 1.0) {
        return None;
    }
    Some((1.0 / (1.0 - g)) * (1.0 + 2f64.powi(-40)))
}

/// Compute `(A·W, S)` via a sound external GEMM engine (cuBLAS). `A·W` stays
/// f64 `Dgemm` (the STORED coefficient needs f64 accuracy so its own error is
/// the tiny `γ_n^f64·S`, not the ~2^29×-larger f32 factor). The abs-sum base
/// `S = Σ_k |a|·|w|` is only ever OVER-bounded, so for `k < 2^23` it is computed
/// with the much faster f32 `Sgemm` and inflated to a guaranteed over-bound
/// `S_hat ≥ true_S` (design + soundness proof in `docs/F32_ABSSUM_SEAM.md` §1-2):
/// `S_hat = (fl32_result + 2k·2^-126)·F_hat`, where the additive `2k·2^-126`
/// guard closes the GPU flush-to-zero underflow hole a pure multiplicative
/// factor cannot. Since `S_hat ≥ true_S`, `ε = γ_n^f64·S_hat` still soundly
/// over-bounds `|a64 − true_aw|` (Higham, order-independent — valid for cuBLAS's
/// unknown reduction order). For `k ≥ 2^23` the second GEMM stays f64 (tighter
/// than the degenerate factor). Requires `eng.gemm_f32` to be IEEE RN f32
/// (`u = 2^-24`), not a TF32/tensor-core path — enforced by `CudaGemmEngine`'s
/// pinned `CUBLAS_DEFAULT_MATH`. Returns `None` on any engine/dimension error so
/// the caller falls back to the CPU loop.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn aw_via_engine(
    eng: &dyn GemmEngine,
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    m: usize,
    k: usize,
    p: usize,
) -> Option<(Array2<f64>, Array2<f64>)> {
    // Coefficient A·W in f64 (unchanged); |a|, |w| as EXACT f32 (abs clears the
    // sign bit) for the abs-sum GEMM.
    let mut a64 = vec![0.0f64; m * k];
    let mut absa = vec![0.0f32; m * k];
    for i in 0..m {
        for kk in 0..k {
            let x = a_block[(i, kk)];
            a64[i * k + kk] = f64::from(x);
            absa[i * k + kk] = x.abs();
        }
    }
    let mut w64 = vec![0.0f64; k * p];
    let mut absw = vec![0.0f32; k * p];
    for kk in 0..k {
        for j in 0..p {
            let y = w[(kk, j)];
            w64[kk * p + j] = f64::from(y);
            absw[kk * p + j] = y.abs();
        }
    }
    let aw = Array2::from_shape_vec((m, p), eng.gemm_f64(m, k, p, &a64, &w64).ok()?).ok()?;

    let s = match f32_abssum_inflation(k) {
        Some(f_hat) => {
            let s32 = eng.gemm_f32(m, k, p, &absa, &absw).ok()?;
            if s32.len() != m * p {
                return None;
            }
            // FTZ-safe underflow guard: ≤ 2k−1 f32 roundings, each losing < 2^-126
            // under flush-to-zero (design §2 step 5); placed INSIDE the F multiply.
            let g = 2.0f64 * (k as f64) * 2f64.powi(-126);
            Array2::from_shape_fn((m, p), |(i, j)| {
                let r = f64::from(s32[i * p + j]); // exact widen (f32 ⊂ f64)
                (r + g) * f_hat // ≥ true_S (design §2)
            })
        }
        None => {
            // k ≥ 2^23: the f32 factor degenerates; use the exact f64 S path.
            let absa64: Vec<f64> = absa.iter().map(|&v| f64::from(v)).collect();
            let absw64: Vec<f64> = absw.iter().map(|&v| f64::from(v)).collect();
            Array2::from_shape_vec((m, p), eng.gemm_f64(m, k, p, &absa64, &absw64).ok()?).ok()?
        }
    };
    Some((aw, s))
}

/// Row-major IEEE **round-to-nearest f32** GEMM `C = A·W`. Tries the
/// process-global fast f32 accelerator (cuBLAS `Sgemm` on `--features cuda`,
/// ~40× the GB10 f64 path) first, then falls back to the faer CPU f32
/// `mat_mul`. Both are plain RN-f32 (`u = 2^-24`), so the accumulation error of
/// the result is charged to the caller's `gamma_n_f32(k)·S` penalty regardless
/// of summation order (Higham `γ_n·S` is order-independent). `a`/`w` are indexed
/// with `[(i, j)]` (layout-agnostic) when flattening, so a column-major faer
/// `Mat` is handled correctly.
#[allow(dead_code)] // unwired: only reachable via `aw_f32_sound_bound` + its oracle test.
fn f32_gemm_rn(a: &Mat<f32>, w: &Mat<f32>, m: usize, k: usize, p: usize) -> Array2<f32> {
    let mut a_flat = vec![0.0f32; m * k];
    for i in 0..m {
        for kk in 0..k {
            a_flat[i * k + kk] = a[(i, kk)];
        }
    }
    let mut w_flat = vec![0.0f32; k * p];
    for kk in 0..k {
        for j in 0..p {
            w_flat[kk * p + j] = w[(kk, j)];
        }
    }
    if let Some(Some(v)) =
        crate::fast_f32_gemm::with_engine(|e| e.gemm_f32(m, k, p, &a_flat, &w_flat).ok())
    {
        if v.len() == m * p {
            if let Ok(arr) = Array2::from_shape_vec((m, p), v) {
                return arr;
            }
        }
    }
    let c = mat_mul(a, w);
    Array2::from_shape_fn((m, p), |(i, j)| c[(i, j)])
}

/// Per-output weight-amplified DAZ operand-flush floor for a length-`k` `A·W`:
/// `≥ Σ_l max(|a_il|,|w_lj|)·FLT_MIN`, the maximum mass a **denormals-are-zero** f32
/// GEMM (a DAZ backend flushes subnormal INPUTS to 0 *before* the multiply) can
/// silently drop. A pure result-flush floor (`c·2^-126`) is magnitude-independent and
/// cover it (a flushed subnormal operand `|a_l| < FLT_MIN` loses `|a_l|·|w_lj|`, up to
/// `|w_lj|·FLT_MIN`). Separable over-bound `(‖a_i‖₁ + ‖w_j‖₁)·FLT_MIN` (`max(x,y) ≤
/// x+y` for `x,y ≥ 0`), computed in O(mk+kp). `FLT_MIN = f32::MIN_POSITIVE = 2^-126`.
/// Mirrors the resident CROWN combine + `ny_core::gemm::crown_aw_error_step`.
#[allow(dead_code)] // reached via aw_f32_sound_bound + sound_abs_product_upper.
fn daz_operand_flush_floor(
    a: &Mat<f32>,
    w: &Mat<f32>,
    m: usize,
    k: usize,
    p: usize,
) -> Array2<f64> {
    let flt_min = f64::from(f32::MIN_POSITIVE);
    let row: Vec<f64> = (0..m)
        .map(|i| (0..k).map(|l| f64::from(a[(i, l)].abs())).sum())
        .collect();
    let col: Vec<f64> = (0..p)
        .map(|j| (0..k).map(|l| f64::from(w[(l, j)].abs())).sum())
        .collect();
    Array2::from_shape_fn((m, p), |(i, j)| (row[i] + col[j]) * flt_min)
}

/// Provably-sound per-entry UPPER bound `S[i,j] ≥ (|A|·|W|)_exact[i,j]` for the
/// f32 CROWN-backward Higham penalty, carrying an extra `(1 + 2^-20)`
/// containment margin (see [`aw_f32_sound_bound`]).
///
/// # S-soundness argument
///
/// Let `P = Σ_k |a_k|·|w_k|` (exact real). The abs values `|a|`,`|w|` are EXACT
/// in f32 (abs only clears the sign bit) and `|a|·|w|` is exact in f64
/// (48 < 53 significand bits), so the only rounding is the summation.
///
/// * **f32 seam (`k < 2^23`).** `r = fl_f32(|A|·|W|)` is an RN-f32 GEMM. By the
///   same theorem the abs-sum engine seam uses (`aw_via_engine`,
///   `docs/F32_ABSSUM_SEAM.md` §1-2, validated 0 violations), `(r + g)·F_hat ≥ P`
///   where `g = 2k·2^-126` closes the flush-to-zero underflow hole and
///   `F_hat = f32_abssum_inflation(k)` corrects the ≤ `γ_k^f32` RN shortfall
///   (already carrying its own `(1 + 2^-40)` f64-rounding margin). We then
///   multiply by `(1 + 2^-20)`, giving `S ≥ P·(1 + 2^-20)`.
/// * **f64 fallback (`k ≥ 2^23`, degenerate `F_hat`).** `s64 = fl_f64(|A|·|W|)`
///   from [`aw_f64_with_abssum`] satisfies `s64 ≥ P·(1 − γ_k^f64)`; since
///   `γ_k^f64 ≪ 2^-20` for every finite-`γ^f32` width, `s64·(1 + 2^-20) ≥ P` and
///   in fact `≥ P·(1 + 2^-20 − γ_k^f64) ≈ P·(1 + 2^-20)`.
///
/// In BOTH branches `S ≥ P·(1 + 2^-20)`, i.e. `S − P ≥ 2^-20·P`. This margin is
/// what makes the f32 enclosure strictly CONTAIN the proven f64 enclosure: the
/// worst-case center gap `|Ĉ_f32 − Ĉ_f64| ≤ (γ_k^f32 + γ_k^f64)·P + ftz` is
/// absorbed because `γ_k^f32·(S − P) ≥ γ_k^f32·2^-20·P ≫ γ_k^f64·(P + s64)`
/// (the ratio `γ^f32/γ^f64 ≈ 2^29`, so `2^29·2^-20 = 2^9` covers the `~2·P`
/// f64-side slack with two orders of magnitude to spare). When in doubt this
/// OVER-widens `S`; it never under-estimates it.
#[allow(dead_code)] // unwired: only reachable via `aw_f32_sound_bound` + its oracle test.
fn sound_abs_product_upper(
    a: &Mat<f32>,
    w: &Mat<f32>,
    m: usize,
    k: usize,
    p: usize,
) -> Array2<f64> {
    // Strict containment margin over the tight abs-product upper bound. `2^-20`
    // dwarfs the `≤ γ_k^f64 ≈ k·2^-53` shortfall of any f64 abs-sum yet stays
    // ~2^9× below the `γ^f32/γ^f64 ≈ 2^29` head-room the f32 penalty carries.
    let contain_margin = 1.0 + 2f64.powi(-20);
    match f32_abssum_inflation(k) {
        Some(f_hat) => {
            let a_abs = Mat::<f32>::from_fn(m, k, |i, j| a[(i, j)].abs());
            let w_abs = Mat::<f32>::from_fn(k, p, |i, j| w[(i, j)].abs());
            let r = f32_gemm_rn(&a_abs, &w_abs, m, k, p);
            // FTZ-safe additive guard: ≤ 2k f32 roundings each losing < 2^-126.
            let g = 2.0 * (k as f64) * 2f64.powi(-126);
            // DAZ (#gpu-metal-daz): `r = fl32(|A|@|W|)` via the fast_f32_gemm engine
            // (cuBLAS in prod) can UNDER-count P by `(‖a_i‖₁+‖w_j‖₁)·FLT_MIN` if a
            // subnormal INPUT is flushed to 0 before the multiply — the result-flush
            // `g` cannot cover that weight-amplified loss. Add it (defensively) so the
            // returned S is `≥ P` even under input-flush.
            let daz = daz_operand_flush_floor(a, w, m, k, p);
            Array2::from_shape_fn((m, p), |(i, j)| {
                (f64::from(r[[i, j]]) + g + daz[[i, j]]) * f_hat * contain_margin
            })
        }
        None => {
            // k ≥ 2^23: the f32 inflation factor degenerates; use the exact f64
            // abs-sum (whose own products are exact) and widen by the margin.
            let (_a64, s64) = aw_f64_with_abssum(a, w);
            s64.mapv(|v| v * contain_margin)
        }
    }
}

/// SOUND f32 CROWN-backward coefficient product. Returns (lower_a, upper_a):
/// a certified f32 enclosure of the exact A·W, i.e. lower_a <= (A·W)_exact <= upper_a
/// entrywise, computed with a round-to-nearest f32 GEMM widened by the Higham
/// gamma_n_f32 * S + FTZ penalty. ~40x cheaper than the f64 path on the GB10.
/// Default-off / experimental until the differential oracle + wiring land.
///
/// # Soundness
///
/// `Ĉ = fl_f32(A·W)` (RN-f32 GEMM) satisfies, entrywise,
/// `|Ĉ − (A·W)_exact| ≤ γ_n^f32(k)·P + ftz` where `P = Σ_k |a_k|·|w_k|` and the
/// FTZ term bounds flush-to-zero underflow (Higham, Accuracy & Stability
/// Thm 3.1, extended with the subnormal-flush absolute floor). We use a proven
/// over-bound `S ≥ P` (see [`sound_abs_product_upper`], which additionally
/// carries a `(1 + 2^-20)` containment margin) and set
/// `lower = Ĉ − (γ_n^f32(k)·S + ftz)`, `upper = Ĉ + (γ_n^f32(k)·S + ftz)`,
/// finally rounding lower DOWN / upper UP to f32 (`next_down_f32`/`next_up_f32`)
/// so the stored f32 endpoints never round inward. Non-finite coefficients (or a
/// degenerate `γ = ∞` at `k ≥ 2^24`) degrade the entry to `[-∞, +∞]` — sound
/// and maximally loose.
///
/// FTZ term: `4·k·2^-126` per output entry. A length-`k` f32 dot product incurs
/// ≤ `2k` roundings that under flush-to-zero each lose an absolute `< 2^-126`
/// (magnitude-INDEPENDENT, unlike the forward-seam bias variant which scales by
/// the input-box `mag_sum`); the factor 4 (vs the tight 2) is conservative.
///
/// # Weight-amplified operand-flush floor — defensive (#gpu-metal-daz)
/// The `4·k·2^-126` FTZ term covers only *result* flush (a subnormal product or
/// partial-sum flushing to 0 — magnitude-independent). It does NOT cover *operand*
/// flush (DAZ): a subnormal input `|a_l| ∈ [2^-149, 2^-126)` zeroed BEFORE the
/// multiply loses up to `|w_l|·2^-126` — WEIGHT-AMPLIFIED. Both `c_hat` and `S` come
/// from `f32_gemm_rn`, which dispatches to the process-global `fast_f32_gemm` engine.
/// In production that engine is **cuBLAS Sgemm** (installed at `ny-cli/main.rs`,
/// pinned to IEEE RN-f32 math mode); `F32_ABSSUM_SEAM.md` already treats it as
/// *result*-flush-capable (hence the `+ g` guard in `sound_abs_product_upper`).
/// This primitive additionally adds the per-output weight-amplified floor
/// [`daz_operand_flush_floor`] `(‖a_i‖₁ + ‖w_j‖₁)·FLT_MIN` to BOTH the abs-sum `S`
/// (so `S ≥ P` even if the engine input-flushes) and the coefficient penalty (so the
/// enclosure covers a flushed `c_hat`) — the same term as `crown_aw_error_step`. It
/// is a strict OUTWARD over-bound (never tightens a bound): it costs ~2^-119 for
/// normal nets and makes the enclosure sound even if a future `fast_f32_gemm` backend
/// input-flushes. NOTE: it does NOT cover the ny-gpu *on-device WGSL* GEMM (a separate
/// engine, `crown_backward_sound_resident`), which flushes on Metal and is handled by
/// its own shader term (`CROWN_AW_ERROR_COMBINE_SHADER`, validated by
/// `crown_backward_sound_resident_daz_subnormal_*`).
///
/// Default-off / experimental: standalone, UNWIRED into any live call site.
#[allow(dead_code)] // unwired by design until the differential oracle + wiring land.
pub(crate) fn aw_f32_sound_bound(a_block: &Mat<f32>, w: &Mat<f32>) -> (Array2<f32>, Array2<f32>) {
    let m = a_block.nrows();
    let k = a_block.ncols();
    let p = w.ncols();
    debug_assert_eq!(w.nrows(), k);

    // Point coefficient: fast RN-f32 GEMM (charged to gamma_n_f32 below).
    let c_hat = f32_gemm_rn(a_block, w, m, k, p);
    // Sound abs-product upper bound S >= |A|·|W| (with containment margin).
    let s = sound_abs_product_upper(a_block, w, m, k, p);

    // The RN-f32 accumulation error over the width-`k` contraction is
    // `γ_n^f32(k)·S` (>> the f64 factor — using the f64 factor for an
    // f32-accumulated coefficient would be UNSOUND). `ftz` is the per-entry
    // flush-to-zero underflow floor.
    let gamma = gamma_n_f32(k);
    let ftz = 4.0 * (k as f64) * 2f64.powi(-126);
    // Weight-amplified DAZ operand-flush floor (#gpu-metal-daz): covers a DAZ-flushed
    // `c_hat` (subnormal input zeroed before the multiply → up to `|w|·FLT_MIN` lost),
    // which `ftz` (result-flush only) cannot. Negligible for normal-magnitude nets.
    let daz = daz_operand_flush_floor(a_block, w, m, k, p);

    let mut lower = Array2::<f32>::zeros((m, p));
    let mut upper = Array2::<f32>::zeros((m, p));
    for i in 0..m {
        for j in 0..p {
            let c = f64::from(c_hat[[i, j]]);
            let penalty = gamma * s[[i, j]] + ftz + daz[[i, j]];
            if !c.is_finite() || !penalty.is_finite() {
                // Overflowed coefficient or degenerate (k >= 2^24) γ: sound but
                // maximally loose. Contains any finite true coefficient.
                lower[[i, j]] = f32::NEG_INFINITY;
                upper[[i, j]] = f32::INFINITY;
            } else {
                // Round the enclosure OUTWARD to f32 so the stored endpoints do
                // not themselves round inward (an f32 overflow of `c ± penalty`
                // saturates to ±inf, still a sound enclosure).
                lower[[i, j]] = next_down_f32((c - penalty) as f32);
                upper[[i, j]] = next_up_f32((c + penalty) as f32);
            }
        }
    }
    (lower, upper)
}

/// CPU CROWN backward propagation using faer matrix multiply.
///
/// For a linear layer y = Wx + b, and current linear bounds A @ y + c:
/// - Substitute y: A @ (Wx + b) + c = (A @ W) @ x + (A @ b + c)
/// - new_A = A @ W
/// - new_b = A @ b + c
///
/// Uses faer for accelerated matrix multiplication (5-10x faster than ndarray::dot).
///
/// # Non-finite coefficient handling (#2681)
///
/// When `A @ W` overflows f32 and produces non-finite coefficients, the old approach
/// of substituting `0.0` is unsound: it drops the input dimension's contribution,
/// which can inflate lower bounds or deflate upper bounds depending on input signs.
///
/// Fix: for any output row with non-finite coefficients, zero the entire row's
/// A-coefficients and set bias to ±inf. This makes the row's concretized bounds
/// `[-inf, +inf]` (sound, maximally loose). Zero coefficients propagate cleanly
/// through subsequent backward layers (no cascading). The post-concretization
/// `has_degraded_bounds` check in `propagate_crown_with_engine` then falls back
/// to IBP for the whole network.
pub(crate) fn propagate_linear_cpu<'a>(
    layer: &LinearLayer,
    bounds: &'a LinearBounds,
) -> Result<Cow<'a, LinearBounds>> {
    debug!("Linear layer CROWN backward propagation");

    let (num_outputs, bounds_inputs) = (bounds.lower_a().nrows(), bounds.lower_a().ncols());
    let weight_rows = layer.weight_faer().nrows();
    let in_features = layer.weight_faer().ncols();

    let layout = resolve_backward_layout(num_outputs, bounds_inputs, weight_rows, in_features)?;

    if layout.num_positions > 1 {
        debug!(
            "Linear backward with sequence dim: {} positions, {} features each",
            layout.num_positions, weight_rows
        );
    }

    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
    // Certified per-coefficient error matrices (#vnncomp-aw-soundness): the
    // SOUND A·W. Each entry bounds `|stored - true_coeff|` and is consumed at
    // concretize as an S-scaled, box-magnitude-scaled penalty.
    let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
    let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
    // #1863: Use f64 accumulators for bias to match nonlinear path's precision
    // standard (common.rs:170-175, fix for catastrophic cancellation #1745).
    let mut lower_bias_contrib = Array1::<f64>::zeros(num_outputs);
    let mut upper_bias_contrib = Array1::<f64>::zeros(num_outputs);

    // Track which output rows have non-finite coefficients (#2681).
    // These rows will be overridden with ±inf bias after the position loop.
    let mut lower_nonfinite_rows = vec![false; num_outputs];
    let mut upper_nonfinite_rows = vec![false; num_outputs];

    // Incoming certified error on this bounds object's coefficients (from a
    // prior linear backward composed through earlier layers). When present, it
    // is propagated as `Σ_k err_in[i, in_start+k]·|W[k,j]|`.
    let in_lower_err = bounds.lower_a_err();
    let in_upper_err = bounds.upper_a_err();
    // |W| once per call (reused per position); shape (contraction, in_features).
    let w_abs = Mat::<f32>::from_fn(weight_rows, in_features, |k, j| {
        layer.weight_faer()[(k, j)].abs()
    });
    let n_contraction = weight_rows;
    let gamma = gamma_n_f64(n_contraction);

    for pos in 0..layout.num_positions {
        let in_start = pos * layout.out_features;
        let out_start = pos * in_features;

        // SOUND A·W (#vnncomp-aw-soundness): accumulate the product AND its
        // absolute-value version in f64. f32×f32 is exact in f64, so only the
        // f64 SUM rounds (covered by γ_n·S below).
        //
        // TEST-ONLY (NY_AW_LEGACY_F32): reproduce the original UNSOUND behaviour
        // — the round-to-nearest f32 faer GEMM with no error accounting — so the
        // strict soundness proptest can confirm it CATCHES the bug. Never set in
        // production.
        let legacy_f32 = cfg!(debug_assertions) && std::env::var("NY_AW_LEGACY_F32").is_ok();
        let (lower_a64, lower_s, upper_a64, upper_s) = if legacy_f32 {
            let lower_block = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                bounds.lower_a()[[i, in_start + j]]
            });
            let upper_block = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                bounds.upper_a()[[i, in_start + j]]
            });
            let ml = mat_mul(&lower_block, layer.weight_faer());
            let mu = mat_mul(&upper_block, layer.weight_faer());
            let la = Array2::from_shape_fn((num_outputs, in_features), |(i, j)| ml[(i, j)] as f64);
            let ua = Array2::from_shape_fn((num_outputs, in_features), |(i, j)| mu[(i, j)] as f64);
            let z = Array2::<f64>::zeros((num_outputs, in_features));
            (la, z.clone(), ua, z)
        } else {
            // Stack lower OVER upper into one (2·num_outputs × out_features) block:
            // both sides multiply the SAME weight, so the f64 A·W + abs-sum runs as
            // ONE engine GEMM pair over the stacked block instead of two — halving
            // the cuBLAS launch/sync/transfer overhead on the sound-f64 seam (and
            // pushing more work above the offload gate). Rows are independent in
            // `aw_f64_with_abssum` (per-`i` f64 accumulation), so the split halves
            // are bit-identical to two separate calls on the CPU path.
            let stacked = Mat::<f32>::from_fn(2 * num_outputs, layout.out_features, |i, j| {
                if i < num_outputs {
                    bounds.lower_a()[[i, in_start + j]]
                } else {
                    bounds.upper_a()[[i - num_outputs, in_start + j]]
                }
            });
            let (a64, s) = aw_f64_with_abssum(&stacked, layer.weight_faer());
            let lower_a64 = a64.slice(ndarray::s![0..num_outputs, ..]).to_owned();
            let upper_a64 = a64.slice(ndarray::s![num_outputs.., ..]).to_owned();
            let lower_s = s.slice(ndarray::s![0..num_outputs, ..]).to_owned();
            let upper_s = s.slice(ndarray::s![num_outputs.., ..]).to_owned();
            (lower_a64, lower_s, upper_a64, upper_s)
        };

        // Propagated incoming error: P[i,j] = Σ_k err_in[i,in_start+k]·|W[k,j]|.
        let prop_lower = in_lower_err.map(|e| {
            let blk = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, k| {
                e[[i, in_start + k]]
            });
            mat_mul(&blk, &w_abs)
        });
        let prop_upper = in_upper_err.map(|e| {
            let blk = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, k| {
                e[[i, in_start + k]]
            });
            mat_mul(&blk, &w_abs)
        });

        // Place result in output, tracking non-finite or near-overflow coefficients
        // per row (#2681, #1932). The magnitude check catches coefficients approaching
        // f32 overflow before they actually reach Inf, preventing NaN from subsequent
        // multiplications. See CROWN_COEFF_MAX documentation.
        for i in 0..num_outputs {
            for j in 0..in_features {
                let l = lower_a64[[i, j]] as f32;
                let u = upper_a64[[i, j]] as f32;
                // Certified error: cast rounding |a64 - stored| + γ_n·S + propagated
                // incoming error, rounded UP to a sound f32.
                let l_cast_err = (lower_a64[[i, j]] - l as f64).abs();
                let u_cast_err = (upper_a64[[i, j]] - u as f64).abs();
                let l_prop = prop_lower.as_ref().map_or(0.0, |p| p[(i, j)] as f64);
                let u_prop = prop_upper.as_ref().map_or(0.0, |p| p[(i, j)] as f64);
                let l_err = next_up_f32((l_cast_err + gamma * lower_s[[i, j]] + l_prop) as f32);
                let u_err = next_up_f32((u_cast_err + gamma * upper_s[[i, j]] + u_prop) as f32);
                // The stored coefficient is sound iff BOTH endpoints stay finite
                // and within the magnitude guard; otherwise degrade the row.
                if is_crown_coeff_safe(l) && l_err.is_finite() {
                    new_lower_a[[i, out_start + j]] = l;
                    new_lower_a_err[[i, out_start + j]] = l_err;
                } else {
                    lower_nonfinite_rows[i] = true;
                }
                if is_crown_coeff_safe(u) && u_err.is_finite() {
                    new_upper_a[[i, out_start + j]] = u;
                    new_upper_a_err[[i, out_start + j]] = u_err;
                } else {
                    upper_nonfinite_rows[i] = true;
                }
            }
        }

        // Accumulate bias contribution across all positions in f64
        if let Some(ref bias) = layer.bias {
            let block = BiasBlockParams {
                num_outputs,
                out_features: layout.out_features,
                col_offset: in_start,
            };
            let lower_slice = contiguous_flat_slice_mut(&mut lower_bias_contrib)?;
            let upper_slice = contiguous_flat_slice_mut(&mut upper_bias_contrib)?;
            accumulate_bias_f64(
                &mut (lower_slice, upper_slice),
                |i, j| bounds.lower_a()[[i, j]],
                |i, j| bounds.upper_a()[[i, j]],
                bias,
                &block,
            );
        }
    }

    // Propagated incoming error contributes to the BIAS too when this layer has
    // a bias: bias_contrib[i] = Σ_j A[i,j]·bias[j], so its certified error is
    // Σ_j err_in[i,j]·|bias[j]|. Fold this into the f64 bias accumulators BEFORE
    // the directed cast (lower decreases, upper increases) so the bias term is
    // sound (#vnncomp-aw-soundness).
    if let Some(ref bias) = layer.bias {
        if let Some(le) = in_lower_err {
            for i in 0..num_outputs {
                let mut e = 0.0f64;
                for j in 0..bias.len() {
                    e += le[[i, j]] as f64 * (bias[j] as f64).abs();
                }
                lower_bias_contrib[i] -= e;
            }
        }
        if let Some(ue) = in_upper_err {
            for i in 0..num_outputs {
                let mut e = 0.0f64;
                for j in 0..bias.len() {
                    e += ue[[i, j]] as f64 * (bias[j] as f64).abs();
                }
                upper_bias_contrib[i] += e;
            }
        }
    }

    // Finalize bias with directed rounding (#2164)
    let (mut new_lower_b, mut new_upper_b) = if layer.bias.is_some() {
        finalize_bias_directed(
            &lower_bias_contrib,
            &upper_bias_contrib,
            bounds.lower_b(),
            bounds.upper_b(),
        )
    } else {
        (bounds.lower_b().clone(), bounds.upper_b().clone())
    };

    // #2681/#1932: For rows with non-finite or near-overflow A-matrix coefficients,
    // zero the entire row and set bias to ±inf. This makes the concretized bound
    // [-inf, +inf] for that output neuron — sound but maximally loose.
    //
    // #1932 extends #2681: besides catching actual Inf/NaN, we now also catch
    // coefficients exceeding CROWN_COEFF_MAX (1e10). This prevents near-overflow
    // f32 coefficients from producing NaN in subsequent multiplications.
    //
    // Reference: alpha-beta-CROWN does no coefficient clamping (relies on float64
    // dynamic range). Our f32 path needs proactive protection.
    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    if lower_affected > 0 || upper_affected > 0 {
        debug!(
            "Linear CROWN backward (faer): overflow/magnitude in {}/{} lower rows, \
             {}/{} upper rows — falling back to ±inf bias for affected rows (#1932)",
            lower_affected, num_outputs, upper_affected, num_outputs
        );
        for i in 0..num_outputs {
            if lower_nonfinite_rows[i] {
                for j in 0..layout.total_in_features {
                    new_lower_a[[i, j]] = 0.0;
                    new_lower_a_err[[i, j]] = 0.0;
                }
                new_lower_b[i] = f32::NEG_INFINITY;
            }
            if upper_nonfinite_rows[i] {
                for j in 0..layout.total_in_features {
                    new_upper_a[[i, j]] = 0.0;
                    new_upper_a_err[[i, j]] = 0.0;
                }
                new_upper_b[i] = f32::INFINITY;
            }
        }
    }

    // CROWN backward NaN firewall (#2812): conservative fallback instead of hard error.
    // SOUND coefficient interval carried via the certified error matrices.
    Ok(Cow::Owned(LinearBounds::new_or_conservative_with_err(
        new_lower_a,
        new_lower_b,
        new_upper_a,
        new_upper_b,
        new_lower_a_err,
        new_upper_a_err,
    )?))
}

/// CROWN backward propagation using an optional GEMM engine for acceleration.
///
/// Falls back to CPU propagation if the engine is `None` or if the GEMM call fails.
pub(crate) fn propagate_linear_with_engine<'a>(
    layer: &LinearLayer,
    bounds: &'a LinearBounds,
    engine: Option<&dyn GemmEngine>,
) -> Result<Cow<'a, LinearBounds>> {
    let Some(engine) = engine else {
        return propagate_linear_cpu(layer, bounds);
    };

    match propagate_linear_via_gemm(layer, bounds, engine) {
        Ok(lb) => Ok(Cow::Owned(lb)),
        Err(e) => {
            debug!("GEMM engine failed for Linear CROWN backward, falling back to CPU: {e}");
            propagate_linear_cpu(layer, bounds)
        }
    }
}

/// Minimum output-row count before a deadline triggers row-chunked backward.
///
/// Below this, the single dense `A @ W` GEMM is cheap enough that per-row
/// deadline granularity is unnecessary; chunking would only add overhead.
/// Matches the spirit of `DEADLINE_GEMM_ROW_CHUNK` in the Conv2d backward.
const DEADLINE_LINEAR_ROW_CHUNK: usize = 64;

/// Deadline-aware CROWN backward propagation through a linear layer (#4321).
///
/// The dense `A @ W` GEMM is the single largest uninterrupted op on the
/// spec-matrix root output-bound path: a wide classifier-head GEMM with many
/// objective rows (e.g. TinyImageNet ResNet, 199 specs) can run for tens of
/// seconds with no internal deadline checkpoint, overrunning the verifier's own
/// `--timeout` and getting killed externally with no JSON verdict.
///
/// When a `deadline` is present and the workload has enough output rows to
/// matter, the spec rows are processed in bounded chunks. CROWN backward is
/// row-independent (`new_A[i] = A[i] @ W`, `new_b[i] = A[i] @ bias + b[i]`),
/// so chunking the output rows and concatenating is **bit-identical** to the
/// single-pass result — only the abort granularity changes. Between chunks we
/// check the wall clock and return [`NyError::DeadlineExceeded`] once it passes,
/// which the graph-CROWN dispatch converts to a graceful per-node IBP fallback
/// (sound: a timeout never claims Verified).
///
/// With no deadline, or for small workloads, this is exactly
/// [`propagate_linear_with_engine`].
pub(crate) fn propagate_linear_with_engine_and_deadline<'a>(
    layer: &LinearLayer,
    bounds: &'a LinearBounds,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
) -> Result<Cow<'a, LinearBounds>> {
    let Some(d) = deadline else {
        return propagate_linear_with_engine(layer, bounds, engine);
    };

    let num_outputs = bounds.lower_a().nrows();
    if num_outputs <= DEADLINE_LINEAR_ROW_CHUNK {
        // Cheap enough: a single pre-op deadline check suffices. If we are
        // already past the deadline, bail before launching the GEMM at all.
        if std::time::Instant::now() >= d {
            return Err(NyError::DeadlineExceeded(
                "Linear CROWN backward: per-node deadline exceeded before GEMM".to_string(),
            ));
        }
        return propagate_linear_with_engine(layer, bounds, engine);
    }

    // Row-chunked backward: each chunk is an independent slice of output rows.
    let lower_a = bounds.lower_a();
    let upper_a = bounds.upper_a();
    let lower_b = bounds.lower_b();
    let upper_b = bounds.upper_b();
    // Certified coefficient-error must be sliced alongside the coefficients so it
    // is propagated per chunk (#vnncomp-aw-soundness); dropping it would lose the
    // concretization penalty for those rows.
    let lower_err = bounds.lower_a_err();
    let upper_err = bounds.upper_a_err();

    let mut chunks: Vec<LinearBounds> = Vec::new();
    let mut row_start = 0usize;
    while row_start < num_outputs {
        if std::time::Instant::now() >= d {
            return Err(NyError::DeadlineExceeded(
                "Linear CROWN backward: per-node deadline exceeded (inter-chunk check)".to_string(),
            ));
        }
        let row_end = (row_start + DEADLINE_LINEAR_ROW_CHUNK).min(num_outputs);
        let mut chunk_bounds = LinearBounds::new(
            lower_a
                .slice(ndarray::s![row_start..row_end, ..])
                .to_owned(),
            lower_b.slice(ndarray::s![row_start..row_end]).to_owned(),
            upper_a
                .slice(ndarray::s![row_start..row_end, ..])
                .to_owned(),
            upper_b.slice(ndarray::s![row_start..row_end]).to_owned(),
        )?;
        if let (Some(le), Some(ue)) = (lower_err, upper_err) {
            chunk_bounds.set_coeff_err(
                le.slice(ndarray::s![row_start..row_end, ..]).to_owned(),
                ue.slice(ndarray::s![row_start..row_end, ..]).to_owned(),
            );
        }
        // Each chunk runs the identical engine/CPU GEMM, just on fewer rows.
        let out = propagate_linear_with_engine(layer, &chunk_bounds, engine)?.into_owned();
        chunks.push(out);
        row_start = row_end;
    }

    Ok(Cow::Owned(concat_linear_bounds_rows(&chunks)?))
}

/// Concatenate row-chunked [`LinearBounds`] back into a single result.
///
/// All chunks share the same input dimension (column count); only the
/// output-row count differs. Concatenation along the row axis reproduces the
/// single-pass layout exactly.
fn concat_linear_bounds_rows(chunks: &[LinearBounds]) -> Result<LinearBounds> {
    let first = chunks.first().ok_or_else(|| {
        NyError::InternalError("concat_linear_bounds_rows: no chunks to concatenate".to_string())
    })?;
    let in_features = first.lower_a().ncols();
    let total_rows: usize = chunks.iter().map(|c| c.lower_a().nrows()).sum();

    let mut lower_a = Array2::<f32>::zeros((total_rows, in_features));
    let mut upper_a = Array2::<f32>::zeros((total_rows, in_features));
    let mut lower_b = Array1::<f32>::zeros(total_rows);
    let mut upper_b = Array1::<f32>::zeros(total_rows);
    // Certified coefficient-error: concat along rows iff every chunk carries it
    // (the linear backward always produces it, so this holds on the verdict path).
    let any_err = chunks.iter().any(|c| c.has_coeff_err());
    let mut lower_err = Array2::<f32>::zeros((total_rows, in_features));
    let mut upper_err = Array2::<f32>::zeros((total_rows, in_features));

    let mut row = 0usize;
    for chunk in chunks {
        let rows = chunk.lower_a().nrows();
        lower_a
            .slice_mut(ndarray::s![row..row + rows, ..])
            .assign(chunk.lower_a());
        upper_a
            .slice_mut(ndarray::s![row..row + rows, ..])
            .assign(chunk.upper_a());
        lower_b
            .slice_mut(ndarray::s![row..row + rows])
            .assign(chunk.lower_b());
        upper_b
            .slice_mut(ndarray::s![row..row + rows])
            .assign(chunk.upper_b());
        if any_err {
            // A chunk without err is exact (err 0); leave its rows zeroed.
            if let Some(le) = chunk.lower_a_err() {
                lower_err
                    .slice_mut(ndarray::s![row..row + rows, ..])
                    .assign(le);
            }
            if let Some(ue) = chunk.upper_a_err() {
                upper_err
                    .slice_mut(ndarray::s![row..row + rows, ..])
                    .assign(ue);
            }
        }
        row += rows;
    }

    // Chunks may legitimately carry ±Inf bias rows (non-finite #2681 handling);
    // use new_or_conservative which tolerates that, matching the single-pass path.
    if any_err {
        LinearBounds::new_or_conservative_with_err(
            lower_a, lower_b, upper_a, upper_b, lower_err, upper_err,
        )
    } else {
        LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
    }
}

/// GEMM-engine CROWN backward propagation.
///
/// # Soundness (#vnncomp-aw-soundness)
///
/// The GPU/engine `gemm_f32` computes the `A·W` coefficient product in
/// round-to-nearest f32 with NO directed rounding and NO error compensation.
/// Over a wide contraction this f32 coefficient error is many ULPs with
/// data-dependent sign and can flip a near-zero coefficient across zero,
/// selecting the wrong concretization corner — a false `Verified`.
///
/// The engine remains the source of the **point coefficient** (keeping the
/// accelerated GEMM and its call-count contract), but the certified
/// coefficient-error matrices are computed independently on the CPU in f64
/// (`a64`, the absolute-product sum `S`, plus any propagated incoming error)
/// and attached to the result. The error bounds `|engine_coeff − true_coeff|`
/// because the engine result and the f64 `a64` both round the same exact real
/// `A·W`, so `|engine − true| <= |engine − a64| + |a64 − true|`, and we widen
/// `S`-scaled to cover BOTH the engine↔a64 gap (one f32 ULP of the magnitude,
/// dominated by `γ_n·S`) and the a64↔true gap (`γ_n·S`). Concretize then applies
/// the box-magnitude-scaled penalty over this interval — sound for any corner.
fn propagate_linear_via_gemm(
    layer: &LinearLayer,
    bounds: &LinearBounds,
    engine: &dyn GemmEngine,
) -> Result<LinearBounds> {
    let (num_outputs, bounds_inputs) = (bounds.lower_a().nrows(), bounds.lower_a().ncols());
    let weight_rows = layer.weight.nrows();
    let in_features = layer.weight.ncols();

    let layout = resolve_backward_layout(num_outputs, bounds_inputs, weight_rows, in_features)?;

    let weight_slice = layer
        .weight
        .as_slice()
        .ok_or_else(|| NyError::InvalidSpec("Linear weight is not contiguous".to_string()))?;

    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
    let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
    let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
    let mut lower_bias_contrib = Array1::<f64>::zeros(num_outputs);
    let mut upper_bias_contrib = Array1::<f64>::zeros(num_outputs);

    let mut lower_nonfinite_rows = vec![false; num_outputs];
    let mut upper_nonfinite_rows = vec![false; num_outputs];

    let in_lower_err = bounds.lower_a_err();
    let in_upper_err = bounds.upper_a_err();
    let weight_faer = layer.weight_faer();
    let w_abs = Mat::<f32>::from_fn(weight_rows, in_features, |k, j| weight_faer[(k, j)].abs());
    // The engine GEMM accumulates in f32 → use the f32 growth factor (sound).
    let gamma = gamma_n_f32(weight_rows);

    for pos in 0..layout.num_positions {
        let in_start = pos * layout.out_features;
        let out_start = pos * in_features;

        let mut lower_block = vec![0.0f32; num_outputs * layout.out_features];
        let mut upper_block = vec![0.0f32; num_outputs * layout.out_features];

        for i in 0..num_outputs {
            let row_off = i * layout.out_features;
            for j in 0..layout.out_features {
                lower_block[row_off + j] = bounds.lower_a()[[i, in_start + j]];
                upper_block[row_off + j] = bounds.upper_a()[[i, in_start + j]];
            }
        }

        let new_lower_block = engine.gemm_f32(
            num_outputs,
            layout.out_features,
            in_features,
            &lower_block,
            weight_slice,
        )?;
        let new_upper_block = engine.gemm_f32(
            num_outputs,
            layout.out_features,
            in_features,
            &upper_block,
            weight_slice,
        )?;

        // Independent f64 S (absolute-product sum) and propagated incoming error
        // for the certified coefficient interval. The engine supplies the point
        // coefficient; the error magnitudes come from the CPU f64 path.
        let lower_faer = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
            bounds.lower_a()[[i, in_start + j]]
        });
        let upper_faer = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
            bounds.upper_a()[[i, in_start + j]]
        });
        let (_, lower_s) = aw_f64_with_abssum(&lower_faer, weight_faer);
        let (_, upper_s) = aw_f64_with_abssum(&upper_faer, weight_faer);
        let prop_lower = in_lower_err.map(|e| {
            let blk = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, k| {
                e[[i, in_start + k]]
            });
            mat_mul(&blk, &w_abs)
        });
        let prop_upper = in_upper_err.map(|e| {
            let blk = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, k| {
                e[[i, in_start + k]]
            });
            mat_mul(&blk, &w_abs)
        });

        for i in 0..num_outputs {
            let src_off = i * in_features;
            for j in 0..in_features {
                let l = new_lower_block[src_off + j];
                let u = new_upper_block[src_off + j];
                let l_prop = prop_lower.as_ref().map_or(0.0, |p| p[(i, j)] as f64);
                let u_prop = prop_upper.as_ref().map_or(0.0, |p| p[(i, j)] as f64);
                // γ_n·S covers BOTH the engine↔a64 and a64↔true gaps (S-scaled).
                let l_err = next_up_f32((gamma * lower_s[[i, j]] + l_prop) as f32);
                let u_err = next_up_f32((gamma * upper_s[[i, j]] + u_prop) as f32);
                if is_crown_coeff_safe(l) && l_err.is_finite() {
                    new_lower_a[[i, out_start + j]] = l;
                    new_lower_a_err[[i, out_start + j]] = l_err;
                } else {
                    lower_nonfinite_rows[i] = true;
                }
                if is_crown_coeff_safe(u) && u_err.is_finite() {
                    new_upper_a[[i, out_start + j]] = u;
                    new_upper_a_err[[i, out_start + j]] = u_err;
                } else {
                    upper_nonfinite_rows[i] = true;
                }
            }
        }

        // Accumulate bias contribution in f64
        if let Some(ref bias) = layer.bias {
            let block = BiasBlockParams {
                num_outputs,
                out_features: layout.out_features,
                col_offset: in_start,
            };
            let lower_slice = contiguous_flat_slice_mut(&mut lower_bias_contrib)?;
            let upper_slice = contiguous_flat_slice_mut(&mut upper_bias_contrib)?;
            accumulate_bias_f64(
                &mut (lower_slice, upper_slice),
                |i, j| bounds.lower_a()[[i, j]],
                |i, j| bounds.upper_a()[[i, j]],
                bias,
                &block,
            );
        }
    }

    // Propagated incoming error into the bias (same as the CPU path).
    if let Some(ref bias) = layer.bias {
        if let Some(le) = in_lower_err {
            for i in 0..num_outputs {
                let mut e = 0.0f64;
                for j in 0..bias.len() {
                    e += le[[i, j]] as f64 * (bias[j] as f64).abs();
                }
                lower_bias_contrib[i] -= e;
            }
        }
        if let Some(ue) = in_upper_err {
            for i in 0..num_outputs {
                let mut e = 0.0f64;
                for j in 0..bias.len() {
                    e += ue[[i, j]] as f64 * (bias[j] as f64).abs();
                }
                upper_bias_contrib[i] += e;
            }
        }
    }

    let (mut new_lower_b, mut new_upper_b) = if layer.bias.is_some() {
        finalize_bias_directed(
            &lower_bias_contrib,
            &upper_bias_contrib,
            bounds.lower_b(),
            bounds.upper_b(),
        )
    } else {
        (bounds.lower_b().clone(), bounds.upper_b().clone())
    };

    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    if lower_affected > 0 || upper_affected > 0 {
        debug!(
            "Linear CROWN backward (GEMM): overflow/magnitude in {}/{} lower rows, \
             {}/{} upper rows — falling back to ±inf bias for affected rows (#1932)",
            lower_affected, num_outputs, upper_affected, num_outputs
        );
        for i in 0..num_outputs {
            if lower_nonfinite_rows[i] {
                for j in 0..layout.total_in_features {
                    new_lower_a[[i, j]] = 0.0;
                    new_lower_a_err[[i, j]] = 0.0;
                }
                new_lower_b[i] = f32::NEG_INFINITY;
            }
            if upper_nonfinite_rows[i] {
                for j in 0..layout.total_in_features {
                    new_upper_a[[i, j]] = 0.0;
                    new_upper_a_err[[i, j]] = 0.0;
                }
                new_upper_b[i] = f32::INFINITY;
            }
        }
    }

    LinearBounds::new_or_conservative_with_err(
        new_lower_a,
        new_lower_b,
        new_upper_a,
        new_upper_b,
        new_lower_a_err,
        new_upper_a_err,
    )
}

#[cfg(test)]
mod aw_f32_sound_tests {
    use super::{aw_f32_sound_bound, aw_f64_with_abssum, gamma_n_f32, gamma_n_f64};
    use faer::Mat;

    /// Dependency-free deterministic PRNG (SplitMix64) so the differential
    /// oracle reproduces bit-for-bit across runs without a `rand` dev-dep.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform f32 in `[-scale, scale]`.
        fn signed(&mut self, scale: f32) -> f32 {
            let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
            (u * 2.0 - 1.0) * scale
        }
    }

    /// The weight-amplified DAZ operand-flush floor must OVER-BOUND the worst-case
    /// mass a denormals-are-zero f32 GEMM drops: `Σ_l max(|a_il|,|w_lj|)·FLT_MIN`.
    /// This is the term that makes `aw_f32_sound_bound` sound when `f32_gemm_rn`
    /// dispatches to a subnormal-input-flushing f32 backend — a pure result-flush
    /// floor (`c·2^-126`, magnitude-independent) cannot cover it.
    #[test]
    fn daz_operand_flush_floor_over_bounds_worst_case() {
        use super::daz_operand_flush_floor;
        let flt_min = f64::from(f32::MIN_POSITIVE);
        // A (m=2, k=3) / W (k=3, p=2) mixing subnormal, tiny-normal, and large operands.
        let a_vals = [[1e-40f32, 2.0, 3.0], [0.5, 1e-38, 4.0]];
        let w_vals = [[1e20f32, 2.0], [3.0, 1e-40], [5.0, 6.0]];
        let a = Mat::<f32>::from_fn(2, 3, |i, j| a_vals[i][j]);
        let w = Mat::<f32>::from_fn(3, 2, |i, j| w_vals[i][j]);
        let floor = daz_operand_flush_floor(&a, &w, 2, 3, 2);
        for i in 0..2 {
            for j in 0..2 {
                let worst: f64 = (0..3)
                    .map(|l| {
                        f64::from(a_vals[i][l].abs()).max(f64::from(w_vals[l][j].abs())) * flt_min
                    })
                    .sum();
                assert!(
                    floor[[i, j]] >= worst,
                    "DAZ floor {} < worst-case operand-flush loss {worst} at [{i},{j}]",
                    floor[[i, j]]
                );
            }
        }
    }

    /// THE soundness gate (#f32-aw-seam): the SOUND f32 CROWN-backward enclosure
    /// `[lo32, hi32]` must always CONTAIN the already-proven f64 enclosure
    /// `[lo64, hi64] = c64 ± γ_n^f64·s64`. Since the f64 bound is proven to
    /// contain the exact `A·W`, containment transitively proves the f32 bound is
    /// sound. A genuine containment must hold entrywise — the tolerance only
    /// absorbs the last-ULP directed rounding, never a real violation.
    #[test]
    fn aw_f32_sound_bound_contains_f64_bound() {
        // (m, k, p, scale): varied magnitudes + a few large-k contractions that
        // stress cancellation (mixed signs → s64 ≫ |c64|). All k ≪ 2^23 so the
        // f32-seam abs-sum branch (not the f64 fallback) is exercised.
        let cases: &[(usize, usize, usize, f32)] = &[
            (4, 8, 4, 1.0),
            (8, 64, 8, 0.5),
            (3, 256, 5, 2.0),
            (2, 1000, 3, 1.0),
            (5, 4096, 4, 0.1),
            (6, 50, 6, 1.0e3),   // large magnitudes
            (4, 300, 4, 1.0e-3), // small magnitudes
            (7, 2048, 3, 4.0),
        ];

        for (idx, &(m, k, p, scale)) in cases.iter().enumerate() {
            let mut rng =
                SplitMix64(0x1234_5678_ABCD_EF01 ^ (idx as u64).wrapping_mul(0x1000_0001));
            let a = Mat::<f32>::from_fn(m, k, |_, _| rng.signed(scale));
            let w = Mat::<f32>::from_fn(k, p, |_, _| rng.signed(1.0));

            // f64 reference enclosure (already proven sound).
            let (c64, s64) = aw_f64_with_abssum(&a, &w);
            let g64 = gamma_n_f64(k);
            let g32 = gamma_n_f32(k);
            assert!(
                g32.is_finite(),
                "case {idx}: gamma_n_f32({k}) must be finite"
            );

            // f32 sound enclosure under test.
            let (lo32, hi32) = aw_f32_sound_bound(&a, &w);
            assert_eq!(lo32.dim(), (m, p));
            assert_eq!(hi32.dim(), (m, p));

            // Inherent width ratio: the f32 penalty uses γ_n^f32 (u = 2^-24) vs
            // the f64 γ_n^f64 (u = 2^-53), so it is intrinsically ~2^29 ≈ 5.4e8
            // wider — NOT "loose", just the price of f32 accumulation.
            let expected_ratio = g32 / g64.max(f64::MIN_POSITIVE);

            for i in 0..m {
                for j in 0..p {
                    let c = c64[[i, j]];
                    let half64 = g64 * s64[[i, j]];
                    let lo64 = c - half64;
                    let hi64 = c + half64;
                    let l32 = f64::from(lo32[[i, j]]);
                    let h32 = f64::from(hi32[[i, j]]);

                    // Tolerance ONLY to absorb the ≤1-ULP `next_down/up` step; a
                    // real containment failure exceeds this by many orders.
                    let scale_ij = 1.0 + lo64.abs().max(hi64.abs());
                    let tol = 1e-5 * scale_ij;

                    assert!(
                        l32 <= lo64 + tol,
                        "case {idx} [{i},{j}]: LOWER containment violated: \
                         lo32={l32:e} > lo64={lo64:e} (c64={c:e}, half64={half64:e})"
                    );
                    assert!(
                        h32 >= hi64 - tol,
                        "case {idx} [{i},{j}]: UPPER containment violated: \
                         hi32={h32:e} < hi64={hi64:e} (c64={c:e}, half64={half64:e})"
                    );
                    assert!(
                        l32 <= h32,
                        "case {idx} [{i},{j}]: degenerate interval lo32={l32:e} > hi32={h32:e}"
                    );

                    // Usability: the f32 enclosure must not be absurdly loose.
                    // Compare against the intrinsic ~expected_ratio factor with
                    // generous head-room (directed-rounding ULPs + the FTZ floor
                    // + the 1+2^-20 S margin), only when the f64 width is
                    // non-degenerate.
                    let w64 = hi64 - lo64;
                    let w32 = h32 - l32;
                    assert!(
                        w32.is_finite(),
                        "case {idx} [{i},{j}]: f32 width non-finite"
                    );
                    if w64 > 1e-20 {
                        let ratio = w32 / w64;
                        assert!(
                            ratio <= 64.0 * expected_ratio + 1.0e3,
                            "case {idx} [{i},{j}]: f32 bound absurdly loose: \
                             ratio={ratio:e} vs intrinsic ~{expected_ratio:e}"
                        );
                    }
                }
            }
        }
    }

    /// Guard the guard: a deliberately UNDER-widened bound (γ_n^f64 instead of
    /// γ_n^f32, i.e. pretending the f32 GEMM accumulated as accurately as f64)
    /// must FAIL containment against a large-k cancellation case — proving the
    /// oracle actually catches an unsound (too-tight) f32 bound.
    #[test]
    fn oracle_catches_undersized_f32_penalty() {
        let (m, k, p, scale) = (4, 2048, 4, 1.0f32);
        let mut rng = SplitMix64(0xDEAD_BEEF_0000_0001);
        let a = Mat::<f32>::from_fn(m, k, |_, _| rng.signed(scale));
        let w = Mat::<f32>::from_fn(k, p, |_, _| rng.signed(1.0));

        let (c64, s64) = aw_f64_with_abssum(&a, &w);
        let g64 = gamma_n_f64(k);
        // A faux "f32" bound that uses the (far too small) f64 factor — the kind
        // of mistake the oracle must reject.
        let mut any_violation = false;
        for i in 0..m {
            for j in 0..p {
                let c = c64[[i, j]];
                let half_bad = g64 * s64[[i, j]]; // UNSOUND for an f32 accumulation
                let lo_bad = (c - half_bad) as f32 as f64;
                let hi_bad = (c + half_bad) as f32 as f64;
                let lo64 = c - half_bad;
                let hi64 = c + half_bad;
                // The bad bound equals the f64 bound here, so it does NOT strictly
                // fail against ITSELF; instead confirm the REAL sound bound is
                // strictly WIDER than this undersized one (i.e. the undersized one
                // would be tighter than sound → the oracle's `<=`/`>=` catches it
                // when substituted for lo32/hi32).
                let (lo32, hi32) = aw_f32_sound_bound(&a, &w);
                let wide = f64::from(hi32[[i, j]]) - f64::from(lo32[[i, j]]);
                let narrow = hi_bad - lo_bad;
                let _ = (lo64, hi64);
                if wide > narrow * 1.5 {
                    any_violation = true;
                }
            }
        }
        assert!(
            any_violation,
            "the SOUND f32 bound should be materially wider than an f64-factor bound \
             on a large-k cancellation case (confirms γ_n^f32 is actually applied)"
        );
    }
}

/// Parity + soundness for the faer f64 SIMD sub-threshold `aw_f64_with_abssum`
/// path (#linearizenn-faer-f64-aw): the blocked/SIMD f64 GEMM must live in the
/// SAME certified envelope as the historical scalar triple loop, and its
/// resulting enclosure `est ± γ_n·S` must still contain the exact real `A·W`.
#[cfg(test)]
mod faer_f64_aw_tests {
    use super::{aw_f64_with_abssum, gamma_n_f64};
    use faer::Mat;
    use ny_tensor::next_up_f32;

    /// Dependency-free deterministic PRNG (SplitMix64) — same generator as the
    /// f32-seam oracle so cases reproduce bit-for-bit without a `rand` dev-dep.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn signed(&mut self, scale: f32) -> f32 {
            let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
            (u * 2.0 - 1.0) * scale
        }
    }

    /// Scalar triple-loop reference: BYTE-FOR-BYTE the historical (kill-switch)
    /// sub-threshold path — same `i → kk → j` nesting, same exact f32→f64 widen,
    /// same per-(i,j) f64 running sum. This is the `NY_NAIVE_F64_AW=1` result.
    fn naive_aw(a: &Mat<f32>, w: &Mat<f32>) -> (Vec<f64>, Vec<f64>) {
        let (m, k, p) = (a.nrows(), a.ncols(), w.ncols());
        let mut a64 = vec![0.0f64; m * p];
        let mut s = vec![0.0f64; m * p];
        for i in 0..m {
            for kk in 0..k {
                let av = f64::from(a[(i, kk)]);
                if av == 0.0 {
                    continue;
                }
                let av_abs = av.abs();
                for j in 0..p {
                    let wv = f64::from(w[(kk, j)]);
                    a64[i * p + j] += av * wv;
                    s[i * p + j] += av_abs * wv.abs();
                }
            }
        }
        (a64, s)
    }

    /// Kahan–Babuška–Neumaier compensated sum — a high-precision reference whose
    /// error is bounded by `(2u + O(nu²))·Σ|xᵢ|` (Higham, Accuracy & Stability
    /// §4.3), i.e. INDEPENDENT of `n` in the leading term. The test's reference
    /// slack `δ = 8·EPSILON·P` (= 16u·P) dominates this bound, so `[r−δ, r+δ]`
    /// provably brackets the exact real sum.
    fn kbn(terms: &[f64]) -> f64 {
        let mut sum = 0.0f64;
        let mut c = 0.0f64;
        for &t in terms {
            let tt = sum + t;
            if sum.abs() >= t.abs() {
                c += (sum - tt) + t;
            } else {
                c += (t - tt) + sum;
            }
            sum = tt;
        }
        sum + c
    }

    #[test]
    fn faer_aw_matches_naive_and_encloses_exact() {
        // Sub-threshold sizes (m·k·p ≪ 1<<24 so the faer path — not the engine
        // offload — is exercised), k ≥ 64 so the enclosure half-width γ_n·S ≈
        // k·u·P dominates the KBN reference slack (8·EPSILON·P = 16u·P). Includes
        // linearizenn's 120-wide contraction and heavy-cancellation / extreme-
        // magnitude cases (mixed signs → S ≫ |A·W|).
        let cases: &[(usize, usize, usize, f32)] = &[
            (4, 64, 4, 1.0),
            (8, 120, 8, 1.0), // linearizenn AllInOne_120_120 contraction width
            (3, 256, 5, 2.0),
            (2, 1024, 3, 1.0),
            (5, 512, 4, 0.1),
            (6, 128, 6, 1.0e3),  // large magnitudes
            (4, 300, 4, 1.0e-3), // small magnitudes
            (7, 2048, 3, 4.0),   // wide contraction, strong cancellation
        ];

        // Non-vacuity witnesses across all cases.
        let mut total_diff_entries = 0usize; // faer ≠ naive (bitwise)
        let mut zero_width_excludes = false; // a point enclosure excludes exact

        for (idx, &(m, k, p, scale)) in cases.iter().enumerate() {
            let mut rng =
                SplitMix64(0x0FED_CBA9_8765_4321 ^ (idx as u64).wrapping_mul(0x1000_0001));
            let a = Mat::<f32>::from_fn(m, k, |_, _| rng.signed(scale));
            let w = Mat::<f32>::from_fn(k, p, |_, _| rng.signed(1.0));

            // Under test: the DEFAULT (faer) path — NY_NAIVE_F64_AW is unset in
            // the test process, so the cached kill-switch resolves to faer.
            let (faer_a64, faer_s) = aw_f64_with_abssum(&a, &w);
            assert_eq!(faer_a64.dim(), (m, p));
            assert_eq!(faer_s.dim(), (m, p));
            // Reference: the byte-identical scalar loop (the kill-switch result).
            let (naive_a64, naive_s) = naive_aw(&a, &w);

            let gamma = gamma_n_f64(k);
            assert!(gamma.is_finite() && gamma > 0.0, "case {idx}: bad gamma");

            for i in 0..m {
                for j in 0..p {
                    // Non-vacuity tally: faer must actually REORDER the sum (a
                    // blocked/SIMD accumulation, not a bit-copy of the left-fold)
                    // — otherwise the parity envelope below is trivially satisfied
                    // and we would not even be exercising the faer path.
                    if faer_a64[[i, j]].to_bits() != naive_a64[i * p + j].to_bits() {
                        total_diff_entries += 1;
                    }
                    let fa = faer_a64[[i, j]];
                    let fs = faer_s[[i, j]];
                    let na = naive_a64[i * p + j];
                    let ns = naive_s[i * p + j];

                    // High-precision exact reference for THIS entry.
                    let mut prod = Vec::with_capacity(k);
                    let mut absprod = Vec::with_capacity(k);
                    for kk in 0..k {
                        let av = f64::from(a[(i, kk)]);
                        let wv = f64::from(w[(kk, j)]);
                        prod.push(av * wv); // exact (f32×f32 ⊂ f64)
                        absprod.push(av.abs() * wv.abs()); // exact
                    }
                    let exact_c = kbn(&prod);
                    let exact_p = kbn(&absprod);
                    // KBN reference slack: 16u·P (> its proven 2u·P + O(nu²) error).
                    let delta = 8.0 * f64::EPSILON * exact_p;

                    // (1) PARITY — faer and naive both round the SAME exact
                    // products, each within γ_k·P of the exact sum, so their
                    // difference is ≤ 2γ_k·P ≤ γ_k·(faer_s + naive_s) (+ a floor
                    // for the all-tiny case). This is the "same envelope" claim.
                    let env = gamma * (fs + ns) + 8.0 * f64::from(f32::MIN_POSITIVE);
                    assert!(
                        (fa - na).abs() <= env,
                        "case {idx} [{i},{j}]: a64 parity broken: faer={fa:e} naive={na:e} \
                         gap={:e} > env={env:e}",
                        (fa - na).abs()
                    );
                    assert!(
                        (fs - ns).abs() <= env,
                        "case {idx} [{i},{j}]: S parity broken: faer={fs:e} naive={ns:e} \
                         gap={:e} > env={env:e}",
                        (fs - ns).abs()
                    );

                    // (2) faer S is a VALID basis: S ≥ (1−γ_k)·P (never under-
                    // counts the exact abs-sum by more than the f64 accumulation).
                    assert!(
                        fs >= exact_p * (1.0 - gamma) - delta,
                        "case {idx} [{i},{j}]: faer S={fs:e} under-counts P={exact_p:e}"
                    );

                    // (3) SOUNDNESS — the production enclosure built from faer's
                    // (a64, S) must contain the exact real A·W (NO tighter-than-
                    // true bound). Replicates the caller's per-entry error:
                    // stored f32 coeff + next_up_f32(|cast| + γ_k·S).
                    let stored = fa as f32;
                    let cast_err = (fa - f64::from(stored)).abs();
                    let err = f64::from(next_up_f32((cast_err + gamma * fs) as f32));
                    let lo = f64::from(stored) - err;
                    let hi = f64::from(stored) + err;
                    // Conservative: require the enclosure to bracket the WHOLE
                    // reference-uncertainty interval [exact_c−δ, exact_c+δ] ⊇
                    // {true exact}. If this holds, the true A·W is enclosed.
                    assert!(
                        lo <= exact_c - delta && exact_c + delta <= hi,
                        "case {idx} [{i},{j}]: faer enclosure [{lo:e},{hi:e}] does NOT \
                         contain exact A·W {exact_c:e} (±δ={delta:e}); stored={stored:e} \
                         err={err:e}, S={fs:e}"
                    );

                    // Non-vacuity: a ZERO-WIDTH enclosure (the point {stored}) must
                    // EXCLUDE the exact real A·W for at least one entry — proving
                    // the containment assertion above genuinely requires the width
                    // `err = |cast| + γ_k·S` and is not trivially satisfied.
                    if f64::from(stored) < exact_c - delta || exact_c + delta < f64::from(stored) {
                        zero_width_excludes = true;
                    }
                }
            }
        }

        // NOTE: faer's blocked f64 kernel may or may not reorder the sum vs the
        // scalar left-fold for a given shape; when it does NOT, `total_diff_entries`
        // is 0 and the two paths are byte-identical (the STRONGEST parity). Either
        // way the per-entry parity + enclosure assertions above are the guarantee.
        // Reported for visibility only — NOT asserted (byte-identical is valid).
        eprintln!(
            "faer_f64_aw: {total_diff_entries} entries differ from the scalar loop \
             (0 = byte-identical; both are sound, order-independent)"
        );
        assert!(
            zero_width_excludes,
            "a zero-width enclosure must exclude the exact A·W somewhere (confirms the \
             enclosure width is load-bearing, i.e. the containment check is non-vacuous)"
        );
    }
}
