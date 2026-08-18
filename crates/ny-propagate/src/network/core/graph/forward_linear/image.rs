// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified forward-linear compositions for image-class conv DAGs
//! (#vnncomp-image-forward-linear).
//!
//! Fills the forward-substitution (DeepPoly-style) op surface for
//! Conv2d / ConvTranspose2d / BatchNorm / binary Add / ReLU / typed-cGAN Tanh /
//! Linear / shape
//! ops so 17-conv ResNets and cGAN-style generator chains get O(L)
//! finite intermediate bounds instead of exploding plain-IBP intervals (which
//! drive the CROWN backward NaN firewall and a vacuous -inf root bound).
//!
//! # Soundness contract (#vnncomp-aw-soundness)
//!
//! Every composition here is on the production verdict path, so every
//! floating-point rounding is certified:
//!
//! * Exact affine maps (Conv2d, Linear) are composed with the upstream
//!   [`LinearBounds`] via the **center–radius identity** in **f64**:
//!   `C⁺U_l + C⁻U_u = C·U_c − |C|·U_r` and `C⁺U_u + C⁻U_l = C·U_c + |C|·U_r`
//!   with `U_c = (U_l+U_u)/2`, `U_r = (U_u−U_l)/2` (exact in f64 for f32
//!   inputs; the identity is algebraic and needs no sign assumption on `U_r`).
//!   The f64 GEMM accumulation error is bounded by the Higham factor
//!   `γ_{K+4}·S` with `S = |C|(|U_c|+|U_r|)` (order-independent, so any IEEE
//!   f64 GEMM backend is admissible — see `sound_f64_gemm`).
//! * The final f64→f32 round-to-nearest coefficient cast gap is **measured
//!   per entry** (`|stored_f32 − value_f64|`, exact because f32→f64 widening
//!   is exact).
//! * Both error sources are discharged immediately through the existing
//!   certified coefficient-error channel semantics: the per-row penalty
//!   `Σ_j err_ij·max(|x_l_j|,|x_u_j|)` — exactly what
//!   `LinearBounds::fold_coeff_err_into_bias` / `concretize_sound` would
//!   apply — is folded OUTWARD into the bias (lower decreases, upper
//!   increases) with directed rounding. Folding eagerly at each op is
//!   algebraically identical to carrying the error matrices and discharging
//!   at concretization (the downstream `C⁺/C⁻` bias split multiplies a
//!   symmetric ±p widening by exactly `|C|`, the same transform
//!   `coeff_err_carrier` propagation applies), but needs no O(N·n) error
//!   matrices.
//! * All bias arithmetic runs in f64 and commits with directed f32 rounding
//!   (`next_down_f32` / `next_up_f32`); the 1-ULP directed step dominates the
//!   ~1e-12-relative residual f64 rounding by >4 orders of magnitude.
//! * Non-finite coefficients degrade the affected row to `A=0, b=±inf`
//!   (sound, maximally loose) via `detect_and_fix_nonfinite_rows`; NaN biases
//!   are mapped to ∓inf. The `LinearBounds::new_or_conservative` NaN firewall
//!   stays as the backstop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use ndarray::{s, Array1, Array2, ArrayD, ArrayView2};
use ny_core::{is_crown_coeff_safe, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use rayon::prelude::*;

use crate::bounds::{safe_mul_for_bounds_f64, LinearBounds};
use crate::layers::activations::relu::relu_linear_relaxation;
use crate::layers::convolution::crown_helpers::detect_and_fix_nonfinite_rows;
use crate::layers::convolution::{Conv2dLayer, ConvTranspose2dLayer};
use crate::layers::linear::crown_single_gamma_n_f32 as gamma_n_f32;
use crate::layers::linear::crown_single_gamma_n_f64 as gamma_n_f64;
use crate::layers::trigonometric::tanh_linear_relaxation;
use crate::layers::BatchNormLayer;

/// Blanket multiplicative inflation applied to every accumulated f64 penalty
/// before it is folded into the bias. Covers all second-order f64 roundings in
/// the penalty accumulation itself (relative error ≤ γ_n^f64 ≈ 5e-13 for the
/// widest sums here) by >5 orders of magnitude while staying negligible
/// relative to the penalty.
const PENALTY_INFLATE: f64 = 1.0 + 1e-7;

/// Maximum inner-loop work between ConvTranspose deadline polls. The exact
/// count is deliberately small enough to bound cancellation latency while
/// avoiding a clock read for every multiply-add.
pub(super) const CONV_TRANSPOSE_DEADLINE_POLL_WORK: usize = 4096;

fn check_conv_transpose_deadline(deadline: Option<Instant>, context: &str) -> Result<()> {
    if deadline.is_some_and(|value| Instant::now() >= value) {
        Err(NyError::DeadlineExceeded(format!(
            "forward-linear ConvTranspose2d: deadline exceeded during {context}"
        )))
    } else {
        Ok(())
    }
}

#[inline]
fn poll_conv_transpose_deadline(
    work: &mut usize,
    cancelled: &AtomicBool,
    deadline: Option<Instant>,
) -> bool {
    let Some(deadline) = deadline else {
        return false;
    };
    if cancelled.load(Ordering::Relaxed) {
        return true;
    }
    *work = work.saturating_add(1);
    if work.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) && Instant::now() >= deadline {
        cancelled.store(true, Ordering::Relaxed);
        true
    } else {
        false
    }
}

/// Finish a ConvTranspose composition without reopening an uninterruptible
/// full-matrix validation tail.
///
/// The generic poller makes the bounded-work contract deterministic in tests.
/// Production passes [`check_conv_transpose_deadline`]. Every coefficient is
/// either observed CROWN-safe or its complete row is zeroed with a conservative
/// infinite bias, exactly like [`detect_and_fix_nonfinite_rows`]. Bias NaNs are
/// repaired at the same row granularity. The resulting parts therefore satisfy
/// [`LinearBounds::from_prevalidated_parts`] without that constructor rescanning
/// `O(num_outputs * num_inputs)` values after the final deadline poll.
pub(super) fn finish_conv_transpose_bounds_with_poll<F>(
    mut lower_a: Array2<f32>,
    mut lower_b: Array1<f32>,
    mut upper_a: Array2<f32>,
    mut upper_b: Array1<f32>,
    conv_in_size: usize,
    layer_name: &str,
    mut poll: F,
) -> Result<LinearBounds>
where
    F: FnMut(&str) -> Result<()>,
{
    debug_assert_eq!(lower_a.ncols(), conv_in_size);
    debug_assert_eq!(upper_a.ncols(), conv_in_size);
    debug_assert_eq!(lower_a.nrows(), upper_a.nrows());
    debug_assert_eq!(lower_a.nrows(), lower_b.len());
    debug_assert_eq!(upper_a.nrows(), upper_b.len());

    let mut work = 0usize;
    let mut lower_affected = 0usize;
    let mut upper_affected = 0usize;
    for row_idx in 0..lower_a.nrows() {
        let mut lower_unsafe = lower_b[row_idx].is_nan();
        if !lower_unsafe {
            for value in lower_a.row(row_idx) {
                if work.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
                    poll("lower tail validation")?;
                }
                work = work.saturating_add(1);
                if !is_crown_coeff_safe(*value) {
                    lower_unsafe = true;
                    break;
                }
            }
        }
        if lower_unsafe {
            for value in lower_a.row_mut(row_idx) {
                if work.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
                    poll("lower tail repair")?;
                }
                work = work.saturating_add(1);
                *value = 0.0;
            }
            lower_b[row_idx] = f32::NEG_INFINITY;
            lower_affected += 1;
        }

        let mut upper_unsafe = upper_b[row_idx].is_nan();
        if !upper_unsafe {
            for value in upper_a.row(row_idx) {
                if work.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
                    poll("upper tail validation")?;
                }
                work = work.saturating_add(1);
                if !is_crown_coeff_safe(*value) {
                    upper_unsafe = true;
                    break;
                }
            }
        }
        if upper_unsafe {
            for value in upper_a.row_mut(row_idx) {
                if work.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
                    poll("upper tail repair")?;
                }
                work = work.saturating_add(1);
                *value = 0.0;
            }
            upper_b[row_idx] = f32::INFINITY;
            upper_affected += 1;
        }
    }
    poll("tail construction")?;

    if lower_affected > 0 || upper_affected > 0 {
        tracing::debug!(
            "{layer_name}: unsafe A/b state in {lower_affected}/{} lower rows, \
             {upper_affected}/{} upper rows — falling back to ±inf bias for affected rows",
            lower_a.nrows(),
            upper_a.nrows()
        );
    }
    LinearBounds::from_prevalidated_parts(lower_a, lower_b, upper_a, upper_b)
}

/// Opt-in (`NY_FORWARD_LINEAR_F32=1`, default OFF) sound f32 fast path for the
/// forward-linear composition's big *value* GEMMs (`A·W`, `|A|·|W|`). The dense
/// f64 conv-forward composition is the measured #1 cost of BaB-bound conv-ResNet
/// instances (cifar100_resnet_medium), and f64 cannot use the wgpu GPU (no f64)
/// so it stalls on the weak GB10 cuBLAS Dgemm (0.41 TF/s). Routing the value
/// GEMMs through the fast RN-f32 path (cuBLAS Sgemm, ~40× on the GB10) is SOUND
/// because the larger f32 accumulation error is bounded by the Higham factor
/// `γ_{K+4}^f32·S` (vs the f64 `γ_{K+4}^f64·S`) plus an FTZ underflow guard —
/// the SAME `S`-scaled certified-error channel the f64 path already discharges
/// into the bias (see [`compose_conv2d_forward`]). PRECISION-NEGATIVE (looser
/// intermediates → risk of regressing categories ny currently verifies), so it
/// stays default-OFF pending broad verdict-parity validation. The S-BASE GEMMs
/// (`v_abs` → `s_coeff`/`s_bias`) stay f64 so `S` is never under-estimated.
#[inline]
fn forward_linear_f32_gemm_enabled() -> bool {
    // Uncached env read (matches `forward_linear_reference_enabled`); the
    // forward composition runs O(alpha-iters·layers) times per instance with
    // huge GEMMs, so a per-call env probe is negligible.
    matches!(
        std::env::var("NY_FORWARD_LINEAR_F32").ok().as_deref(),
        Some("1")
    )
}

/// FTZ-safe underflow addend for the f32 value GEMMs, discharged into the bias.
/// Under flush-to-zero a length-`k` f32 dot product can lose ≤ `2k` roundings of
/// `< 2^-126` each (design mirror of the f32-abs-sum seam, `crown_single.rs`);
/// there are two value GEMMs (`A·W` center and `|A|·|W|` radius), so `4k·2^-126`
/// bounds the per-coefficient underflow error, and `·Σ_j mag_j` discharges it
/// across the input columns exactly as the `γ·S` penalty is discharged.
#[inline]
fn forward_f32_ftz_bias(contraction: usize, mag_sum: f64) -> f64 {
    4.0 * (contraction as f64) * 2f64.powi(-126) * mag_sum
}

/// Row-major RN-f32 GEMM `C = A @ B` for the forward-linear value seam. `a`/`b`
/// hold f32-representable-or-wider f64 values (conv im2col of the upstream
/// center/radius, and the f32 kernel widened to f64); both are rounded to f32,
/// multiplied by a plain IEEE round-to-nearest f32 GEMM (the coefficient error
/// of the resulting f32 accumulation is charged to the caller's `γ_n^f32·S`
/// penalty). Tries the process-global fast f32 accelerator (cuBLAS `Sgemm` on
/// `--features cuda`) first, then the per-call engine's `gemm_f32` (so tests can
/// inject an engine without the process-global `OnceLock`). Returns `None` (→
/// caller falls back to the certified f64 path) on any unavailable/failed
/// engine or dimension mismatch.
fn forward_value_gemm_f32(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    engine: Option<&dyn GemmEngine>,
) -> Option<Vec<f64>> {
    let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let r32 = crate::fast_f32_gemm::with_engine(|e| e.gemm_f32(m, k, n, &a32, &b32).ok())
        .flatten()
        .or_else(|| engine.and_then(|e| e.gemm_f32(m, k, n, &a32, &b32).ok()))?;
    if r32.len() != m * n {
        return None;
    }
    Some(r32.into_iter().map(f64::from).collect())
}

/// Maximum MACs in one non-interruptible accelerator dispatch while the
/// verifier deadline is authoritative. This is large enough to clear the
/// measured CUDA-vs-CPU crossover, but forbids the historical unbounded
/// full-product launch. Implementations may impose a smaller hard cap.
const DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS: usize = 1 << 24;

/// Try one engine's explicit deadline-bounded IEEE-f64 contract.
///
/// Deadline errors are terminal. Unsupported, failed, malformed, or non-finite
/// results decline to the existing pollable CPU implementation; no partial
/// accelerator result can feed a verdict.
fn certified_f64_gemm_deadline_try_engine(
    engine: &dyn GemmEngine,
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    deadline: Instant,
) -> Result<Option<Vec<f64>>> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "forward-linear image bounds: deadline exceeded before bounded f64 GEMM".into(),
        ));
    }
    let expected_output_len = m
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec("forward-linear GEMM output overflow".into()))?;
    match engine.gemm_f64_with_deadline(
        m,
        k,
        n,
        a,
        b,
        deadline,
        DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS,
    ) {
        Ok(result)
            if result.len() == expected_output_len && result.iter().all(|x| x.is_finite()) =>
        {
            if Instant::now() >= deadline {
                Err(NyError::DeadlineExceeded(
                    "forward-linear image bounds: bounded f64 GEMM completed after deadline".into(),
                ))
            } else {
                Ok(Some(result))
            }
        }
        Ok(_) => {
            if Instant::now() >= deadline {
                Err(NyError::DeadlineExceeded(
                    "forward-linear image bounds: malformed bounded f64 GEMM crossed deadline"
                        .into(),
                ))
            } else {
                Ok(None)
            }
        }
        Err(error) if error.is_deadline_exceeded() => Err(error),
        Err(_) => {
            if Instant::now() >= deadline {
                Err(NyError::DeadlineExceeded(
                    "forward-linear image bounds: failed bounded f64 GEMM crossed deadline".into(),
                ))
            } else {
                Ok(None)
            }
        }
    }
}

/// Row-major f64 GEMM `C = A @ B` with a certified-soundness contract: the
/// backend must compute plain IEEE round-to-nearest **f64** dot products (any
/// summation order — the Higham `γ_n·S` bound used by callers is
/// order-independent). Tries the process-global sound f64 accelerator
/// (e.g. cuBLAS Dgemm), then the per-call engine's `gemm_f64`, then faer CPU.
fn certified_f64_gemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    engine: Option<&dyn GemmEngine>,
    allow_f32: bool,
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    if let Some(deadline) = deadline {
        // A finite deadline may use only the engine's explicit bounded-dispatch
        // contract. The global accessor never waits on OnceLock/factory
        // initialization on this thread; unsupported/unavailable engines retain
        // the 6f49a660 pollable f64 CPU fallback below.
        if let Some(attempt) = crate::sound_f64_gemm::with_engine_deadline(deadline, |engine| {
            certified_f64_gemm_deadline_try_engine(engine, m, k, n, a, b, deadline)
        })? {
            if let Some(result) = attempt? {
                return Ok(result);
            }
        }
        if let Some(engine) = engine {
            if let Some(result) =
                certified_f64_gemm_deadline_try_engine(engine, m, k, n, a, b, deadline)?
            {
                return Ok(result);
            }
        }
        // #fwd-linear-deadline-f32: honor the sound f32 value-GEMM seam HERE too.
        // This branch used to `return` straight to the f64 CPU path, so a finite
        // deadline — which every competition run carries — made the f32 seam
        // below unreachable no matter how it was configured. The seam's error is
        // charged by the CALLER as `gamma_{K+4}^f32·S` + FTZ, computed from the
        // same `use_f32` gate that produced `allow_f32` here, so that accounting
        // already covers this path; running f64 instead was merely more accurate
        // than the penalty assumed (sound, just slower).
        if allow_f32 {
            // #fl-value-gpu-tier: the process-global deadline-bounded wgpu f32
            // engine first (chunked submissions, host-side polls, size
            // threshold, all-finite validation — all inside the engine). The
            // charge is order-independent, so the same accounting covers it.
            // Any refusal (size, memory, deadline, non-finite, not installed)
            // falls through to the tiled CPU f32 tier.
            if let Some(result) = certified_f32_gemm_deadline_gpu(m, k, n, a, b, deadline)? {
                return Ok(result);
            }
            if let Some(result) = certified_f32_gemm_deadline_cpu(m, k, n, a, b, deadline)? {
                return Ok(result);
            }
        }
        return certified_f64_gemm_deadline_cpu(m, k, n, a, b, deadline);
    }
    // Sound f32 fast path (opt-in, VALUE GEMMs only — never the S base). The
    // caller resolves the seam gate and charges the larger `γ_{K+4}^f32·S` + FTZ
    // error to the bias, so any RN-f32 summation order is admissible; falls
    // through to the certified f64 path when the seam is off for this GEMM or no
    // f32 engine is available.
    if allow_f32 {
        if let Some(res) = forward_value_gemm_f32(m, k, n, a, b, engine) {
            return Ok(res);
        }
    }
    if let Some(Some(res)) =
        crate::sound_f64_gemm::with_engine(|eng| eng.gemm_f64(m, k, n, a, b).ok())
    {
        if res.len() == m * n {
            return Ok(res);
        }
    }
    if let Some(eng) = engine {
        if let Ok(res) = eng.gemm_f64(m, k, n, a, b) {
            if res.len() == m * n {
                return Ok(res);
            }
        }
    }
    let am = faer::Mat::<f64>::from_fn(m, k, |i, j| a[i * k + j]);
    let bm = faer::Mat::<f64>::from_fn(k, n, |i, j| b[i * n + j]);
    let mut dst = faer::Mat::<f64>::zeros(m, n);
    faer::linalg::matmul::matmul(
        &mut dst,
        faer::Accum::Replace,
        &am,
        &bm,
        1.0,
        crate::faer_parallelism::current_par(),
    );
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = dst[(i, j)];
        }
    }
    Ok(out)
}

/// Maximum MACs per non-interruptible wgpu submission for the FL value tier
/// (#fl-value-gpu-tier). 2^28 ≈ 268M MACs: tens of milliseconds even at the
/// low end of m7's measured wgpu GEMM range (1bb88165), so host-side polls
/// between submissions bound deadline-cancellation latency well inside any
/// verifier budget, while each submission stays large enough to amortize
/// dispatch + readback (the 0.38x small-shape regime). Larger than the CUDA
/// f64 cap (2^24) because the wgpu round-trip overhead per submission is
/// higher and the f32 chunks are half the bytes.
const DEADLINE_F32_WGPU_MAX_DISPATCH_MACS: usize = 1 << 28;

/// Deadline-authoritative f32 value GEMM on the process-global FL-value wgpu
/// engine (#fl-value-gpu-tier).
///
/// Consulted BEFORE [`certified_f32_gemm_deadline_cpu`] when the seam is on.
/// The engine owns the size threshold (measured GPU/CPU crossover), the
/// chunked row-block submission with host-side deadline polls, and the
/// all-finite result validation; this wrapper owns the f64↔f32 narrowing /
/// widening (identical to the CPU tier's) and the never-publish-after-deadline
/// check. Returns `Ok(None)` — the caller falls to the next tier — on ANY
/// engine refusal: not installed, below the size threshold, memory, deadline,
/// or non-finite. A deadline refusal falls through safely because every lower
/// tier re-checks the same authoritative deadline before doing work.
fn certified_f32_gemm_deadline_gpu(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    deadline: Instant,
) -> Result<Option<Vec<f64>>> {
    if !crate::fl_value_gemm::is_installed() {
        return Ok(None);
    }
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "forward-linear image bounds: deadline exceeded before FL-value GPU GEMM".into(),
        ));
    }
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec("forward-linear GEMM output overflow".into()))?;
    // The SIZE THRESHOLD (measured GPU/CPU crossover) lives in the engine,
    // which owns the measurement citation; a sub-crossover shape costs this
    // wrapper only the O(mk+kn) narrowing below before the typed refusal
    // falls through — negligible against the O(mkn) product itself.
    // Same narrowing contract as the CPU tier: a non-finite narrowing means
    // f32 cannot represent this problem — decline to the next tier.
    let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    if a32.iter().any(|v| !v.is_finite()) || b32.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    let attempt = crate::fl_value_gemm::with_engine_deadline(deadline, |engine| {
        engine.gemm_f32_with_deadline(
            m,
            k,
            n,
            &a32,
            &b32,
            deadline,
            DEADLINE_F32_WGPU_MAX_DISPATCH_MACS,
        )
    })?;
    match attempt {
        Some(Ok(r32)) if r32.len() == output_len && r32.iter().all(|v| v.is_finite()) => {
            if Instant::now() >= deadline {
                Err(NyError::DeadlineExceeded(
                    "forward-linear image bounds: FL-value GPU GEMM completed after deadline"
                        .into(),
                ))
            } else {
                crate::fl_value_gemm::record_gpu_tier_hit();
                Ok(Some(r32.into_iter().map(f64::from).collect()))
            }
        }
        // Malformed / non-finite / refused (size, memory, deadline): fall
        // through. No partial engine result can feed a bound, and expired
        // deadlines are re-detected by the next tier's first poll.
        Some(_) | None => Ok(None),
    }
}

/// Deadline-authoritative f32 value GEMM (#fwd-linear-deadline-f32).
///
/// Same tiling contract as [`certified_f64_gemm_deadline_cpu`] — the contraction
/// dimension is never split, so every output coefficient is still one ordinary
/// length-`k` dot product — but the products run in f32. Used ONLY for the value
/// GEMMs the caller has already gated with `use_f32`; the caller charges their
/// error as `gamma_{K+4}^f32·S` plus the FTZ addend, which is
/// summation-order independent, so any RN-f32 order here is admissible.
///
/// Returns `Ok(None)` if a non-finite product appears (f32 overflow on operands
/// an f64 product would have held), so the caller falls back to the f64 path
/// rather than propagating an inf into a bound.
fn certified_f32_gemm_deadline_cpu(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    deadline: Instant,
) -> Result<Option<Vec<f64>>> {
    const MAX_TILE_MACS: usize = 1 << 26;

    let check = || {
        if Instant::now() >= deadline {
            Err(NyError::DeadlineExceeded(
                "forward-linear image bounds: deadline exceeded during f32 value GEMM".into(),
            ))
        } else {
            Ok(())
        }
    };

    check()?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec("forward-linear GEMM output overflow".into()))?;
    if m == 0 || n == 0 {
        return Ok(Some(vec![0.0f64; output_len]));
    }
    // One linear narrowing pass over each operand; a non-finite narrowing means
    // f32 cannot represent this problem, so decline to the f64 path.
    let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    if a32.iter().any(|v| !v.is_finite()) || b32.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    check()?;
    let mut out32 = vec![0.0f32; output_len];

    let mut i0 = 0usize;
    while i0 < m {
        check()?;
        let (rows, max_cols) = certified_f64_gemm_deadline_tile_shape(m - i0, k, n, MAX_TILE_MACS);
        let a_tile = faer::MatRef::from_row_major_slice(&a32[i0 * k..(i0 + rows) * k], rows, k);
        let mut j0 = 0usize;
        while j0 < n {
            check()?;
            let cols = max_cols.min(n - j0);
            let b_tile = faer::MatRef::from_row_major_slice_with_stride(&b32[j0..], k, cols, n);
            // See the note in `certified_f64_gemm_deadline_cpu`: faer 0.24's
            // `from_row_major_slice_with_stride_mut` yields a COLUMN-major view.
            let dst = faer::MatMut::from_column_major_slice_with_stride_mut(
                &mut out32[i0 * n + j0..],
                cols,
                rows,
                n,
            )
            .transpose_mut();
            faer::linalg::matmul::matmul(
                dst,
                faer::Accum::Replace,
                a_tile,
                b_tile,
                1.0f32,
                crate::faer_parallelism::current_par(),
            );
            check()?;
            j0 += cols;
        }
        i0 += rows;
    }
    check()?;
    if out32.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    Ok(Some(out32.into_iter().map(f64::from).collect()))
}

/// Deadline-authoritative f64 GEMM.
///
/// Generic and process-global GEMM engines have no cancellation contract, and
/// one full faer product can be arbitrarily larger than the remaining verifier
/// budget. Finite-deadline work therefore stays on CPU and is tiled so every
/// opaque faer call contains at most a bounded number of MACs. The contraction
/// dimension is not split, preserving one ordinary length-`k` dot product per
/// output coefficient and the caller's existing `gamma_(k+4)` certificate.
fn certified_f64_gemm_deadline_cpu(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    deadline: Instant,
) -> Result<Vec<f64>> {
    // Bound on MACs per opaque faer call. Raised from 1<<22 (~4.2M, ~1ms —
    // too small for faer's internal parallelism to pay off, so the per-tile
    // overhead dominated) to 1<<26 (~67M, tens of ms), which still keeps each
    // uninterruptible call far inside any verifier budget.
    // Tunable for the shape experiment (`NY_FWDLIN_TILE_MACS`, log2). The cap
    // bounds ONE opaque faer call; too small a cap makes every tile a skinny
    // GEMM (at k=1152 a 1<<26 cap allows only ~58 rows), which wastes most of
    // the machine's f64 throughput.
    let max_tile_macs: usize = std::env::var("NY_FWDLIN_TILE_MACS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|b| (20..=34).contains(b))
        .map_or(1usize << 26, |b| 1usize << b);
    #[allow(non_snake_case)]
    let MAX_TILE_MACS = max_tile_macs;

    let check = || {
        if Instant::now() >= deadline {
            Err(NyError::DeadlineExceeded(
                "forward-linear image bounds: deadline exceeded during certified f64 GEMM".into(),
            ))
        } else {
            Ok(())
        }
    };

    check()?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec("forward-linear GEMM output overflow".into()))?;
    let mut out = vec![0.0f64; output_len];
    check()?;
    if m == 0 || n == 0 {
        return Ok(out);
    }

    // A single very long dot product cannot be bounded by tiling only output
    // rows/columns. Use its historical scalar reduction order with explicit
    // polls; products are already f64 and the same Higham enclosure applies.
    if k > MAX_TILE_MACS {
        let mut operations = 0usize;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    if operations.is_multiple_of(4096) {
                        check()?;
                    }
                    operations = operations.wrapping_add(1);
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        check()?;
        return Ok(out);
    }

    // ZERO-COPY TILES (#fwd-linear-tile-copies).
    //
    // This loop used to materialize every tile: `Mat::from_fn` for the lhs, one
    // for the rhs, a `Mat::zeros` destination, and a scalar `dst[(i,j)]`
    // copy-out. faer stores column-major, so each `from_fn` read row-major
    // source memory while writing column-major — a strided, cache-hostile copy —
    // and the copy-out ran the same pattern in reverse. All four are SERIAL,
    // while only the matmul between them is parallel. With the old 4.2M-MAC cap
    // an individual faer call was ~1ms, far too small for its internal
    // parallelism to pay off, so the serial copies dominated: measured on
    // CIFAR100_resnet_medium, the forward-linear collection ran 177s pinned to a
    // SINGLE thread with every other core parked in rayon's idle path
    // (`wait_until_cold`/`cthread_yield` were 80% of all samples).
    //
    // Row-major views over `a`, `b` and `out` remove all four copies, and the
    // larger cap gives faer a tile worth parallelizing while keeping each opaque
    // call well inside the deadline-responsiveness contract this function exists
    // to provide (a 64M-MAC f64 tile is tens of milliseconds).
    //
    // SOUND: unchanged arithmetic. The contraction dimension is still never
    // split, so every output coefficient is still ONE ordinary length-`k` dot
    // product and the caller's `gamma_(k+4)` certificate — which is summation
    // -ORDER independent (Higham `gamma_k·S`) — covers this exactly as before.
    // Tiling only partitions which output coefficients are computed together.
    // One-shot probe: is faer actually allowed to use the machine here?
    if crate::phase_telemetry::phase_telemetry_enabled() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let par = crate::faer_parallelism::current_par();
            eprintln!(
                "[phase] fwdlin-gemm-par par={par:?} rayon_threads={} m={m} k={k} n={n}",
                rayon::current_num_threads()
            );
        });
    }
    let mut i0 = 0usize;
    while i0 < m {
        check()?;
        let (rows, max_cols) = certified_f64_gemm_deadline_tile_shape(m - i0, k, n, MAX_TILE_MACS);
        // Rows i0..i0+rows of the row-major m x k lhs are contiguous.
        let a_tile = faer::MatRef::from_row_major_slice(&a[i0 * k..(i0 + rows) * k], rows, k);
        let mut j0 = 0usize;
        while j0 < n {
            check()?;
            let cols = max_cols.min(n - j0);
            debug_assert!(
                rows.saturating_mul(k).saturating_mul(cols) <= MAX_TILE_MACS,
                "deadline GEMM tile exceeded its MAC cap"
            );
            // Columns j0..j0+cols of the row-major k x n rhs: same buffer,
            // row stride `n`, base offset `j0`.
            let b_tile = faer::MatRef::from_row_major_slice_with_stride(&b[j0..], k, cols, n);
            // Write straight into the output tile — row stride `n`, base
            // offset `i0*n + j0` — instead of into a scratch `Mat` + copy.
            // NOTE: do NOT use `MatMut::from_row_major_slice_with_stride_mut`
            // here. In faer 0.24 that constructor forwards to
            // `from_raw_parts_mut(ptr, nrows, ncols, 1, row_stride)` — byte for
            // byte what `from_column_major_slice_with_stride_mut` does — so it
            // yields a COLUMN-major view despite the name, and the product lands
            // transposed. (The immutable `from_row_major_slice_with_stride` is
            // fine: it composes column-major + `transpose()`.) Compose it the
            // same way here, explicitly.
            let dst = faer::MatMut::from_column_major_slice_with_stride_mut(
                &mut out[i0 * n + j0..],
                cols,
                rows,
                n,
            )
            .transpose_mut();
            faer::linalg::matmul::matmul(
                dst,
                faer::Accum::Replace,
                a_tile,
                b_tile,
                1.0,
                crate::faer_parallelism::current_par(),
            );
            check()?;
            j0 += cols;
        }
        i0 += rows;
    }
    check()?;
    Ok(out)
}

/// Choose a non-empty output tile whose opaque contraction stays within the
/// caller's MAC cap. The row bound matters when `k` is large: clamping only
/// columns still permits `rows * k` to exceed the cap for a one-column tile.
fn certified_f64_gemm_deadline_tile_shape(
    remaining_rows: usize,
    k: usize,
    remaining_cols: usize,
    max_tile_macs: usize,
) -> (usize, usize) {
    debug_assert!(remaining_rows > 0);
    debug_assert!(remaining_cols > 0);
    debug_assert!(max_tile_macs > 0);
    debug_assert!(k <= max_tile_macs);

    let rows_for_one_col = (max_tile_macs / k.max(1)).max(1);
    let rows = remaining_rows.min(64).min(rows_for_one_col).max(1);
    let cols_for_rows = (max_tile_macs / rows.saturating_mul(k).max(1)).max(1);
    let cols = remaining_cols.min(cols_for_rows).max(1);
    (rows, cols)
}

/// Commit an f64 bias value to f32 with directed rounding after folding the
/// certified penalty outward. NaN degrades to the conservative infinity.
fn commit_lower_bias(value: f64, penalty: f64) -> f32 {
    let v = value - penalty * PENALTY_INFLATE;
    if v.is_nan() {
        f32::NEG_INFINITY
    } else {
        next_down_f32(v as f32)
    }
}

fn commit_upper_bias(value: f64, penalty: f64) -> f32 {
    let v = value + penalty * PENALTY_INFLATE;
    if v.is_nan() {
        f32::INFINITY
    } else {
        next_up_f32(v as f32)
    }
}

/// Cast one composed f64 coefficient pair to f32 (round-to-nearest) and
/// accumulate the measured cast gap, weighted by the input-box magnitude, into
/// the per-row penalties. The f32→f64 widening of the stored value is exact,
/// so `gap = |stored − value|` is the true cast error.
#[inline]
fn cast_coeff_with_gap(value: f64, mag: f64, penalty: &mut f64) -> f32 {
    let stored = value as f32;
    if stored.is_finite() {
        *penalty += (stored as f64 - value).abs() * mag;
    }
    stored
}

/// Contiguous row-major views of the upstream coefficient matrices. Fails
/// closed (caller falls back to IBP) when a matrix is not standard-layout —
/// `LinearBounds` are constructed row-major everywhere, so this is defensive.
fn upstream_slices<'a>(
    upstream: &'a LinearBounds,
    node_name: &str,
) -> Result<(&'a [f32], &'a [f32])> {
    match (upstream.lower_a().as_slice(), upstream.upper_a().as_slice()) {
        (Some(l), Some(u)) => Ok((l, u)),
        _ => Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' upstream coefficients are not \
             standard-layout"
        ))),
    }
}

/// Geometry of a 2D convolution derived from the layer + the predecessor shape.
/// Shared with the forward-map alpha optimizer (`alpha_opt`, #w4-root-alpha-opt).
pub(super) struct ConvGeometry {
    pub(super) in_c: usize,
    pub(super) in_h: usize,
    pub(super) in_w: usize,
    pub(super) out_c: usize,
    pub(super) out_h: usize,
    pub(super) out_w: usize,
    pub(super) kh: usize,
    pub(super) kw: usize,
    pub(super) stride: (usize, usize),
    pub(super) padding: (usize, usize),
    pub(super) dilation: (usize, usize),
    /// Contraction width per output: in_c * kh * kw.
    pub(super) contraction: usize,
}

impl ConvGeometry {
    pub(super) fn conv_in_size(&self) -> usize {
        self.in_c * self.in_h * self.in_w
    }
    pub(super) fn conv_out_size(&self) -> usize {
        self.out_c * self.out_h * self.out_w
    }
    pub(super) fn spatial(&self) -> usize {
        self.out_h * self.out_w
    }
}

pub(super) fn resolve_conv_geometry(
    node_name: &str,
    layer: &Conv2dLayer,
    pred_shape: &[usize],
    upstream_outputs: usize,
    output_dim: usize,
) -> Result<ConvGeometry> {
    if layer.groups != 1 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' has groups={} (only groups=1 supported)",
            layer.groups
        )));
    }
    let kshape = layer.kernel.shape();
    if kshape.len() != 4 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' kernel must be 4-D, got {kshape:?}"
        )));
    }
    let (out_c, in_c, kh, kw) = (kshape[0], kshape[1], kshape[2], kshape[3]);

    // Predecessor shape: strip leading batch-1 dims down to (C, H, W).
    let mut dims = pred_shape;
    while dims.len() > 3 && dims[0] == 1 {
        dims = &dims[1..];
    }
    if dims.len() != 3 || dims[0] != in_c {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' expects (C={in_c}, H, W) input, got {pred_shape:?}"
        )));
    }
    let (in_h, in_w) = (dims[1], dims[2]);
    if in_c * in_h * in_w != upstream_outputs {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_c * in_h * in_w],
            got: vec![upstream_outputs],
        });
    }

    let (sh, sw) = layer.stride;
    let (ph, pw) = layer.padding;
    let (dh, dw) = layer.dilation;
    if sh == 0 || sw == 0 || dh == 0 || dw == 0 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' has zero stride/dilation"
        )));
    }
    let eff_kh = dh * (kh - 1) + 1;
    let eff_kw = dw * (kw - 1) + 1;
    let padded_h = in_h + 2 * ph;
    let padded_w = in_w + 2 * pw;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' effective kernel exceeds padded input"
        )));
    }
    let out_h = (padded_h - eff_kh) / sh + 1;
    let out_w = (padded_w - eff_kw) / sw + 1;
    if out_c * out_h * out_w != output_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c * out_h * out_w],
            got: vec![output_dim],
        });
    }
    Ok(ConvGeometry {
        in_c,
        in_h,
        in_w,
        out_c,
        out_h,
        out_w,
        kh,
        kw,
        stride: (sh, sw),
        padding: (ph, pw),
        dilation: (dh, dw),
        contraction: in_c * kh * kw,
    })
}

/// Apply the forward convolution to each ROW of `rows` (shape
/// `(n_obj, conv_in_size)`, f64) via im2col + certified f64 GEMM. `kernel_col`
/// is the reshaped kernel `(contraction, out_c)` (from [`kernel_col_f64`]).
/// Output is `(n_obj, conv_out_size)` in `(oc, oh, ow)` C-order per row —
/// the same contraction as `conv2d_forward_batched_gemm`, in f64.
fn conv_apply_rows_f64(
    rows: ArrayView2<'_, f64>,
    kernel_col: &[f64],
    geo: &ConvGeometry,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    allow_f32: bool,
) -> Result<Array2<f64>> {
    let n_obj = rows.nrows();
    let spatial = geo.spatial();
    let total_rows = n_obj * spatial;
    let k = geo.contraction;
    let (sh, sw) = geo.stride;
    let (ph, pw) = geo.padding;
    let (dh, dw) = geo.dilation;
    let kernel_spatial = geo.kh * geo.kw;
    let input_spatial = geo.in_h * geo.in_w;

    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded(
            "forward-linear image bounds: deadline exceeded before conv im2col".into(),
        ));
    }

    // im2col gather: (n_obj*out_h*out_w, contraction), row-major; one
    // contiguous block per objective, so the gather parallelizes cleanly.
    let mut im2col = vec![0.0f64; total_rows * k];
    im2col
        .par_chunks_mut(spatial * k)
        .enumerate()
        .for_each(|(obj, block)| {
            let row_view = rows.row(obj);
            // Rows of a row-major (possibly row-sliced) matrix are contiguous.
            let row = row_view.to_slice().expect("contiguous objective row");
            for oh in 0..geo.out_h {
                for ow in 0..geo.out_w {
                    let base = (oh * geo.out_w + ow) * k;
                    for col in 0..k {
                        let ic = col / kernel_spatial;
                        let rem = col % kernel_spatial;
                        let ki = rem / geo.kw;
                        let kj = rem % geo.kw;
                        let ih = (oh * sh + ki * dh) as isize - ph as isize;
                        let iw = (ow * sw + kj * dw) as isize - pw as isize;
                        if ih >= 0 && ih < geo.in_h as isize && iw >= 0 && iw < geo.in_w as isize {
                            block[base + col] =
                                row[ic * input_spatial + ih as usize * geo.in_w + iw as usize];
                        }
                    }
                }
            }
        });

    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded(
            "forward-linear image bounds: deadline exceeded before conv GEMM".into(),
        ));
    }

    let gemm = certified_f64_gemm(
        total_rows, k, geo.out_c, &im2col, kernel_col, engine, allow_f32, deadline,
    )?;

    // Scatter to (n_obj, out_c*out_h*out_w) with (oc, oh, ow) C-order per row.
    let mut out = Array2::<f64>::zeros((n_obj, geo.conv_out_size()));
    let conv_out = geo.conv_out_size();
    out.as_slice_mut()
        .expect("freshly allocated row-major")
        .par_chunks_mut(conv_out)
        .enumerate()
        .for_each(|(obj, row_out)| {
            for oh in 0..geo.out_h {
                for ow in 0..geo.out_w {
                    let gemm_row = obj * spatial + oh * geo.out_w + ow;
                    for oc in 0..geo.out_c {
                        row_out[oc * spatial + oh * geo.out_w + ow] =
                            gemm[gemm_row * geo.out_c + oc];
                    }
                }
            }
        });
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded(
            "forward-linear image bounds: deadline exceeded after conv scatter".into(),
        ));
    }
    Ok(out)
}

/// Reshape the conv kernel `(out_c, in_c, kh, kw)` to a column matrix
/// `(contraction, out_c)` in f64, optionally taking absolute values.
fn kernel_col_f64(layer: &Conv2dLayer, geo: &ConvGeometry, absolute: bool) -> Vec<f64> {
    let kernel_spatial = geo.kh * geo.kw;
    let mut w = vec![0.0f64; geo.contraction * geo.out_c];
    for oc in 0..geo.out_c {
        for col in 0..geo.contraction {
            let ic = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / geo.kw;
            let kj = rem % geo.kw;
            let v = layer.kernel[[oc, ic, ki, kj]] as f64;
            w[col * geo.out_c + oc] = if absolute { v.abs() } else { v };
        }
    }
    w
}

/// Compose the upstream forward-linear bounds through a Conv2d node with
/// certified rounding (see module docs). O(input_dim) forward conv passes via
/// im2col + f64 GEMM, chunked over the network-input columns.
#[allow(clippy::too_many_arguments)]
pub(super) fn compose_conv2d_forward(
    node_name: &str,
    layer: &Conv2dLayer,
    upstream: &LinearBounds,
    pred_shape: &[usize],
    output_dim: usize,
    input_mag: &[f64],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    // Seam gate for the value GEMMs; `None` reads `NY_FORWARD_LINEAR_F32` (the
    // production default), tests pass `Some(_)` to force the path race-free.
    use_f32_override: Option<bool>,
) -> Result<LinearBounds> {
    let use_f32 = use_f32_override.unwrap_or_else(forward_linear_f32_gemm_enabled);
    let geo = resolve_conv_geometry(
        node_name,
        layer,
        pred_shape,
        upstream.num_outputs(),
        output_dim,
    )?;
    let n = upstream.num_inputs();
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let conv_in = geo.conv_in_size();
    let conv_out = geo.conv_out_size();

    // #fwdlin-timing (NY_PHASE_TELEMETRY=1): per-node breakdown of this
    // composition's wall clock. Costs one branch when telemetry is off.
    //
    // Added because profiling kept attributing the whole pre-alpha bootstrap
    // phase to this function, which was misleading. Measured on
    // CIFAR100_resnet_medium: all 19 conv nodes compose in 19.6s TOTAL (each
    // called exactly once -- the DAG collector is cached), and that time is
    // essentially pure GEMM (e.g. Conv_11 m=8192 k=1152 n=3072: gemm=1.87s of
    // total=1.94s, ~62 GFLOP/s f64, already near what this CPU can do). The
    // bootstrap phase is 130.4s, so ~110s is the REST of
    // `collect_forward_linear_state_dag` -- carrying a dense
    // (out_dim x input_dim) linear map through every node -- not the conv
    // composition. Keep this probe: it is the difference between optimizing
    // the GEMM again and finding the real cost.
    let tprobe = crate::phase_telemetry::phase_telemetry_enabled();
    let t_start = Instant::now();
    let mut t_kernel = std::time::Duration::ZERO;
    let mut t_ws = std::time::Duration::ZERO;
    let mut t_rows = std::time::Duration::ZERO;
    let mut t_gemm = std::time::Duration::ZERO;
    let mut t_cast = std::time::Duration::ZERO;
    let tk = Instant::now();
    let w_col = kernel_col_f64(layer, &geo, false);
    let wabs_col = kernel_col_f64(layer, &geo, true);
    t_kernel += tk.elapsed();

    let mut new_lower_a = Array2::<f32>::zeros((conv_out, n));
    let mut new_upper_a = Array2::<f32>::zeros((conv_out, n));
    let mut penalty_l = vec![0.0f64; conv_out];
    let mut penalty_u = vec![0.0f64; conv_out];

    // Raw row-major slices: the composition loops below are the hot path and
    // ndarray's checked `[[i, j]]` indexing dominates them in dev profiles.
    // Fail closed (caller falls back to IBP) on a non-standard layout.
    let (ul, uu) = upstream_slices(upstream, node_name)?;

    // Mag-weighted column absolute sums for the γ·S penalty, computed in one
    // parallel pass over features (contiguous upstream rows):
    // w_s[k] = Σ_j (|U_c[k,j]| + |U_r[k,j]|) · mag_j.
    let tws = Instant::now();
    let w_s: Vec<f64> = (0..conv_in)
        .into_par_iter()
        .map(|k_feat| {
            let row_l = &ul[k_feat * n..k_feat * n + n];
            let row_u = &uu[k_feat * n..k_feat * n + n];
            let mut acc = 0.0f64;
            for j in 0..n {
                let l = row_l[j] as f64;
                let u = row_u[j] as f64;
                // Bit-identical to `(l + u) * 0.5`: finite f32-cast operands stay on
                // f64::midpoint's non-overflow `(a + b) * 0.5` path.
                let c = f64::midpoint(l, u);
                let r = (u - l) * 0.5;
                acc += (c.abs() + r.abs()) * input_mag[j];
            }
            acc
        })
        .collect();
    t_ws += tws.elapsed();

    // Chunk the network-input columns so the transient f64 im2col stays
    // bounded (~256 MB): rows_per_chunk * spatial * contraction * 8 bytes.
    let spatial = geo.spatial();
    let budget_rows = (256usize << 20) / 8 / geo.contraction.max(1);
    let chunk_cols = (budget_rows / spatial.max(1)).clamp(16, 1024).min(n.max(1));

    let mut rows_c = Array2::<f64>::zeros((chunk_cols, conv_in));
    let mut rows_r = Array2::<f64>::zeros((chunk_cols, conv_in));

    let mut j0 = 0usize;
    while j0 < n {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "forward-linear image bounds: deadline exceeded inside conv node '{node_name}'"
            )));
        }
        let cb = chunk_cols.min(n - j0);
        // Build center/radius rows for this chunk (exact in f64: f32 widening
        // is exact and (l+u), (u−l), /2 are exact f64 ops on f32 inputs).
        // Parallel over chunk rows (each objective row jj is contiguous).
        let trows = Instant::now();
        {
            let rows_c_flat = rows_c.as_slice_mut().expect("row-major rows_c");
            let rows_r_flat = rows_r.as_slice_mut().expect("row-major rows_r");
            rows_c_flat[..cb * conv_in]
                .par_chunks_mut(conv_in)
                .zip(rows_r_flat[..cb * conv_in].par_chunks_mut(conv_in))
                .enumerate()
                .for_each(|(jj, (crow, rrow))| {
                    let j = j0 + jj;
                    for k_feat in 0..conv_in {
                        let l = ul[k_feat * n + j] as f64;
                        let u = uu[k_feat * n + j] as f64;
                        // Bit-identical (f32-cast operands, f64::midpoint fast path).
                        crow[k_feat] = f64::midpoint(l, u);
                        rrow[k_feat] = (u - l) * 0.5;
                    }
                });
        }
        t_rows += trows.elapsed();
        let tg = Instant::now();
        // Value GEMMs (center/radius coefficients): routed to the sound f32 fast
        // path iff the seam is on — their error is charged to the `γ^f32·S`
        // penalty below.
        let g_center = conv_apply_rows_f64(
            rows_c.slice(s![..cb, ..]),
            &w_col,
            &geo,
            engine,
            deadline,
            use_f32,
        )?;
        let g_radius = conv_apply_rows_f64(
            rows_r.slice(s![..cb, ..]),
            &wabs_col,
            &geo,
            engine,
            deadline,
            use_f32,
        )?;

        t_gemm += tg.elapsed();
        let tc = Instant::now();
        // Cast + measured-gap accumulation, parallel over output rows p
        // (each thread owns row p of both coefficient matrices and its
        // penalty slots).
        {
            let gc = g_center.as_slice().expect("row-major g_center");
            let gr = g_radius.as_slice().expect("row-major g_radius");
            let la_flat = new_lower_a.as_slice_mut().expect("row-major lower_a");
            let ua_flat = new_upper_a.as_slice_mut().expect("row-major upper_a");
            la_flat
                .par_chunks_mut(n)
                .zip(ua_flat.par_chunks_mut(n))
                .zip(penalty_l.par_iter_mut().zip(penalty_u.par_iter_mut()))
                .enumerate()
                .for_each(|(p, ((lrow, urow), (pl, pu)))| {
                    for jj in 0..cb {
                        let j = j0 + jj;
                        let mag = input_mag[j];
                        let c = gc[jj * conv_out + p];
                        let r = gr[jj * conv_out + p];
                        lrow[j] = cast_coeff_with_gap(c - r, mag, pl);
                        urow[j] = cast_coeff_with_gap(c + r, mag, pu);
                    }
                });
        }
        t_cast += tc.elapsed();
        j0 += cb;
    }
    let t_loop_end = t_start.elapsed();
    if tprobe {
        let tot = t_loop_end.as_secs_f64();
        let tail = tot - (t_kernel + t_ws + t_rows + t_gemm + t_cast).as_secs_f64();
        eprintln!(
            "[phase] fwdlin-node {node_name} total={tot:.2}s kernel={:.2} ws={:.2} rows={:.2} \
gemm={:.2} cast={:.2} tail={tail:.2} LOOP-ONLY (m={} k={} n={} chunks={})",
            t_kernel.as_secs_f64(),
            t_ws.as_secs_f64(),
            t_rows.as_secs_f64(),
            t_gemm.as_secs_f64(),
            t_cast.as_secs_f64(),
            conv_out,
            geo.contraction,
            n,
            n.div_ceil(chunk_cols.max(1)),
        );
    }

    // γ·S penalty (coefficient accumulation error) + bias terms via
    // single-vector conv passes.
    let up_lb = upstream.lower_b();
    let up_ub = upstream.upper_b();
    let mut small = Array2::<f64>::zeros((4, conv_in));
    for k_feat in 0..conv_in {
        let l = up_lb[k_feat] as f64;
        let u = up_ub[k_feat] as f64;
        // Bit-identical (f32-cast operands, f64::midpoint fast path).
        let c = f64::midpoint(l, u);
        let r = (u - l) * 0.5;
        small[[0, k_feat]] = c; // u_c  (through W)
        small[[1, k_feat]] = r; // u_r  (through |W|)
        small[[2, k_feat]] = c.abs() + r.abs(); // bias S base (through |W|)
        small[[3, k_feat]] = w_s[k_feat]; // coefficient S base (through |W|)
    }
    // S-BASE GEMMs stay f64: `v_abs` produces `s_bias`/`s_coeff` (the certified
    // error base), which must never be under-estimated, and the bias values
    // (`v_center`/`v_abs` row 0) are tiny so f64 costs nothing here.
    let v_center = conv_apply_rows_f64(
        small.slice(s![0..1, ..]),
        &w_col,
        &geo,
        engine,
        deadline,
        false,
    )?;
    let v_abs = conv_apply_rows_f64(
        small.slice(s![1..4, ..]),
        &wabs_col,
        &geo,
        engine,
        deadline,
        false,
    )?;

    // The value GEMMs (`g_center`/`g_radius`) ran in f32 iff the seam is on, so
    // the coefficient accumulation error is `γ^f32` (much larger) plus an FTZ
    // underflow addend discharged across the input columns. `γ^f32 ≥ γ^f64`
    // conservatively bounds the (f64) bias-GEMM error too, so a single factor is
    // sound for both `s_coeff` (f32 value GEMM) and `s_bias` (f64 bias GEMM).
    let gamma = if use_f32 {
        gamma_n_f32(geo.contraction + 4)
    } else {
        gamma_n_f64(geo.contraction + 4)
    };
    let ftz = if use_f32 {
        let mag_sum: f64 = input_mag.iter().sum();
        forward_f32_ftz_bias(geo.contraction, mag_sum)
    } else {
        0.0
    };
    let mut new_lower_b = Array1::<f32>::zeros(conv_out);
    let mut new_upper_b = Array1::<f32>::zeros(conv_out);
    for p in 0..conv_out {
        let oc = p / spatial;
        let conv_bias = layer.bias.as_ref().map_or(0.0f64, |b| b[oc] as f64);
        let vc = v_center[[0, p]];
        let vr = v_abs[[0, p]];
        let s_bias = v_abs[[1, p]];
        let s_coeff = v_abs[[2, p]];
        let gamma_pen = gamma * (s_coeff + s_bias) + ftz;
        new_lower_b[p] = commit_lower_bias(vc - vr + conv_bias, penalty_l[p] + gamma_pen);
        new_upper_b[p] = commit_upper_bias(vc + vr + conv_bias, penalty_u[p] + gamma_pen);
    }

    let t_bias_end = t_start.elapsed();
    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        "forward-linear Conv2d",
    );
    let t_nonfinite_end = t_start.elapsed();
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return Err(NyError::DeadlineExceeded(format!(
            "forward-linear image bounds: deadline exceeded before returning conv node '{node_name}'"
        )));
    }
    let out = LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b);
    if tprobe {
        let done = t_start.elapsed();
        // The four stamps are successive `t_start.elapsed()` samples, so each
        // difference is already non-negative; `saturating_sub` reports the
        // same spans and keeps this diagnostic print panic-free.
        eprintln!(
            "[phase] fwdlin-tail {node_name} loop={:.2}s vSbias={:.2}s nonfinite={:.2}s \
new_or_cons={:.2}s TOTAL={:.2}s",
            t_loop_end.as_secs_f64(),
            t_bias_end.saturating_sub(t_loop_end).as_secs_f64(),
            t_nonfinite_end.saturating_sub(t_bias_end).as_secs_f64(),
            done.saturating_sub(t_nonfinite_end).as_secs_f64(),
            done.as_secs_f64(),
        );
    }
    out
}

/// Geometry of a transposed 2-D convolution derived from the layer + the
/// predecessor shape. `ConvTranspose2dLayer` has no `groups` field, so the
/// scatter below is always the dense groups=1 map.
pub(super) struct ConvTransposeGeometry {
    pub(super) in_c: usize,
    pub(super) in_h: usize,
    pub(super) in_w: usize,
    pub(super) out_c: usize,
    pub(super) out_h: usize,
    pub(super) out_w: usize,
    pub(super) kh: usize,
    pub(super) kw: usize,
    pub(super) stride: (usize, usize),
    pub(super) padding: (usize, usize),
    pub(super) dilation: (usize, usize),
}

impl ConvTransposeGeometry {
    #[inline]
    pub(super) fn in_size(&self) -> usize {
        self.in_c * self.in_h * self.in_w
    }
    #[inline]
    pub(super) fn out_size(&self) -> usize {
        self.out_c * self.out_h * self.out_w
    }
    /// Maximum number of products accumulated into ONE output cell: for a fixed
    /// output `(oc, oh, ow)` and a fixed `(ic, kh, kw)` there is at most one
    /// `(ih, iw)` with `ih·sh + kh·dh − ph = oh` and `iw·sw + kw·dw − pw = ow`,
    /// so the fan-in never exceeds `in_c·kh·kw` (the same count the certified
    /// interval kernel charges as `macs`).
    #[inline]
    pub(super) fn fan_in(&self) -> usize {
        self.in_c.saturating_mul(self.kh).saturating_mul(self.kw)
    }
}

pub(super) fn resolve_conv_transpose_geometry(
    node_name: &str,
    layer: &ConvTranspose2dLayer,
    pred_shape: &[usize],
    upstream_outputs: usize,
    output_dim: usize,
) -> Result<ConvTransposeGeometry> {
    if pred_shape.len() != 3 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' expects squeezed [C,H,W], got {pred_shape:?}"
        )));
    }
    let (in_c, in_h, in_w) = (pred_shape[0], pred_shape[1], pred_shape[2]);
    let expected_in_c = layer.try_in_channels()?;
    if in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![in_c],
        });
    }
    let (kh, kw) = layer.try_kernel_size()?;
    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    let out_c = layer.try_out_channels()?;
    let geo = ConvTransposeGeometry {
        in_c,
        in_h,
        in_w,
        out_c,
        out_h,
        out_w,
        kh,
        kw,
        stride: layer.stride,
        padding: layer.padding,
        dilation: layer.dilation,
    };
    if geo.dilation.0 == 0 || geo.dilation.1 == 0 {
        return Err(NyError::InvalidSpec(format!(
            "forward-linear ConvTranspose2d '{node_name}': dilation must be >= 1"
        )));
    }
    if geo.in_size() != upstream_outputs || geo.out_size() != output_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![upstream_outputs, output_dim],
            got: vec![geo.in_size(), geo.out_size()],
        });
    }
    Ok(geo)
}

/// Apply the dense transposed-conv linear operator to a BATCH of flattened
/// `[C,H,W]` rows in IEEE binary64:
/// `out[r, (oc,oh,ow)] = Σ_{ic,ih,iw,kh,kw} W[ic,oc,kh,kw] · rows[r, (ic,ih,iw)]`
/// with `oh = ih·sh + kh·dh − ph`, `ow = iw·sw + kw·dw − pw` (out-of-range
/// positions dropped, exactly as `conv2d_transpose_forward`). `absolute` takes
/// `|W|`. The layer bias is NOT applied here.
///
/// The f32 kernel widens to f64 EXACTLY, so every product is a product of two
/// binary64 values and the accumulation error of each output cell is bounded by
/// the Higham factor `γ_{fan_in}` times the sum of product magnitudes — which
/// the caller obtains from a companion `|W|` pass and discharges into the bias.
/// Because the bound is order-independent, the parallel row split below is
/// admissible.
fn conv_transpose_apply_rows_f64(
    rows: &[f64],
    batch: usize,
    kernel: &ArrayD<f32>,
    absolute: bool,
    geo: &ConvTransposeGeometry,
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    check_conv_transpose_deadline(deadline, "kernel application admission")?;
    let in_size = geo.in_size();
    let out_size = geo.out_size();
    if rows.len() != batch.saturating_mul(in_size) {
        return Err(NyError::ShapeMismatch {
            expected: vec![batch * in_size],
            got: vec![rows.len()],
        });
    }
    let total = batch.checked_mul(out_size).ok_or_else(|| {
        NyError::InvalidSpec("forward-linear ConvTranspose2d output overflows usize".into())
    })?;
    // Kernel repacked to (ic, kh, kw, oc) so the innermost `oc` loop is
    // contiguous in both the kernel and the output plane stride.
    let mut w = vec![0.0f64; geo.in_c * geo.kh * geo.kw * geo.out_c];
    let mut pack_work = 0usize;
    for ic in 0..geo.in_c {
        for ki in 0..geo.kh {
            for kj in 0..geo.kw {
                let base = ((ic * geo.kh + ki) * geo.kw + kj) * geo.out_c;
                for oc in 0..geo.out_c {
                    if pack_work.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
                        check_conv_transpose_deadline(deadline, "kernel packing")?;
                    }
                    pack_work = pack_work.saturating_add(1);
                    let v = f64::from(kernel[[ic, oc, ki, kj]]);
                    w[base + oc] = if absolute { v.abs() } else { v };
                }
            }
        }
    }

    let (sh, sw) = geo.stride;
    let (dh, dw) = geo.dilation;
    let (ph, pw) = geo.padding;
    let spatial = geo.out_h * geo.out_w;
    // DEGENERATE OUTPUT GRID: `ConvTranspose2dLayer::output_size` deliberately
    // permits a zero-size spatial dim (it errors only when `expanded < 2*pad`, so
    // equality yields 0), and `par_chunks_mut(0)` panics with "chunk_size must not
    // be zero". Refuse by returning the empty composition instead of aborting the
    // process: the pre-f64 packed-coefficient route returned `Ok` here, and this
    // path must degrade rather than unwind — a panic gives no bound at all, where
    // an empty one still lets the caller fall back.
    if out_size == 0 || total == 0 {
        check_conv_transpose_deadline(deadline, "empty kernel application")?;
        return Ok(vec![0.0f64; total]);
    }
    let mut out = vec![0.0f64; total];
    let cancelled = AtomicBool::new(false);
    out.par_chunks_mut(out_size)
        .enumerate()
        .for_each(|(r, out_row)| {
            if let Some(deadline) = deadline {
                if cancelled.load(Ordering::Relaxed) || Instant::now() >= deadline {
                    cancelled.store(true, Ordering::Relaxed);
                    return;
                }
            }
            let mut work = 0usize;
            let row = &rows[r * in_size..r * in_size + in_size];
            for ic in 0..geo.in_c {
                for ih in 0..geo.in_h {
                    for iw in 0..geo.in_w {
                        if poll_conv_transpose_deadline(&mut work, &cancelled, deadline) {
                            return;
                        }
                        let v = row[(ic * geo.in_h + ih) * geo.in_w + iw];
                        if v == 0.0 {
                            continue;
                        }
                        for ki in 0..geo.kh {
                            if poll_conv_transpose_deadline(&mut work, &cancelled, deadline) {
                                return;
                            }
                            let oh = (ih * sh + ki * dh) as isize - ph as isize;
                            if oh < 0 || oh >= geo.out_h as isize {
                                continue;
                            }
                            for kj in 0..geo.kw {
                                if poll_conv_transpose_deadline(&mut work, &cancelled, deadline) {
                                    return;
                                }
                                let ow = (iw * sw + kj * dw) as isize - pw as isize;
                                if ow < 0 || ow >= geo.out_w as isize {
                                    continue;
                                }
                                let cell = oh as usize * geo.out_w + ow as usize;
                                let base = ((ic * geo.kh + ki) * geo.kw + kj) * geo.out_c;
                                for oc in 0..geo.out_c {
                                    if poll_conv_transpose_deadline(&mut work, &cancelled, deadline)
                                    {
                                        return;
                                    }
                                    out_row[oc * spatial + cell] += w[base + oc] * v;
                                }
                            }
                        }
                    }
                }
            }
        });
    if deadline.is_some() && cancelled.load(Ordering::Relaxed) {
        return Err(NyError::DeadlineExceeded(
            "forward-linear ConvTranspose2d: deadline exceeded during kernel scatter".into(),
        ));
    }
    check_conv_transpose_deadline(deadline, "kernel application completion")?;
    Ok(out)
}

/// Compose the upstream forward-linear bounds through a ConvTranspose2d node
/// `y = convT(h) + b` with certified rounding (see module docs).
///
/// The composition is the SAME certified center–radius identity the Conv2d and
/// Gemm paths use, evaluated in IEEE binary64 by
/// [`conv_transpose_apply_rows_f64`]:
///
/// * `A_c = (A_l + A_u)/2`, `A_r = (A_u − A_l)/2` (exact in f64 on f32 inputs),
/// * `W⁺A_l + W⁻A_u = W·A_c − |W|·A_r` and `W⁺A_u + W⁻A_l = W·A_c + |W|·A_r`
///   (algebraic; needs no sign assumption on `A_r`),
/// * the f64 accumulation error of both value passes is bounded by
///   `γ_{fan_in+4}^f64 · S` with `S = |W|·(|A_c| + |A_r|)` from a third `|W|`
///   pass, discharged into the bias as `Σ_j γ·S[p,j]·max(|x_l_j|,|x_u_j|)`,
/// * the final f64→f32 coefficient cast gap is MEASURED per entry and
///   discharged through the same channel.
///
/// # Why this does not reuse the certified interval ConvTranspose kernel
///
/// It used to (`propagate_ibp_sound_with_engine` on packed coefficient
/// columns), and that cost the whole point of the pass. That kernel is f32, so
/// its Higham widening is `γ_{K+2}^f32 ≈ 1.2e-4` for cGAN's `K = 2048` — NINE
/// orders of magnitude coarser than `γ^f64 ≈ 2.3e-13` here — and the widening
/// is proportional to `S = Σ|W||A|`, an IBP-like quantity that does not shrink
/// under cancellation. MEASURED on `cGAN_imgSz32_nCh_1` prop_1: the first
/// ConvTranspose came out 2.7 % wider than its exact affine range, and by the
/// output the compounded loss was 15×. It also fails OPEN to `[-inf, +inf]` for
/// the whole node on any binary32-subnormal source operand (its
/// DAZ-independence guard) — and the forward-linear ReLU composition
/// manufactures exactly such operands, because a stable-inactive neuron commits
/// its exact `0` intercept through `next_down_f32(0.0)`/`next_up_f32(0.0)`,
/// which by construction returns `∓1.4e-45`. On cGAN that made EVERY
/// ConvTranspose downstream of the first ReLU return the universal interval,
/// collapsing the entire forward-linear map back to plain IBP (root
/// `[-1131.14, 486.41]` versus the pure-IBP `[-1131.19, 486.42]`).
///
/// The f64 path has neither problem: a binary32 subnormal widens to a NORMAL
/// binary64, so no DAZ guard applies and no fail-open is needed.
#[allow(clippy::too_many_arguments)]
pub(super) fn compose_conv_transpose2d_forward(
    node_name: &str,
    layer: &ConvTranspose2dLayer,
    upstream: &LinearBounds,
    pred_shape: &[usize],
    output_dim: usize,
    input_mag: &[f64],
    _engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<LinearBounds> {
    check_conv_transpose_deadline(deadline, "composition admission")?;
    let geo = resolve_conv_transpose_geometry(
        node_name,
        layer,
        pred_shape,
        upstream.num_outputs(),
        output_dim,
    )?;
    let n = upstream.num_inputs();
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    for (index, value) in upstream
        .lower_b()
        .iter()
        .chain(upstream.upper_b().iter())
        .enumerate()
    {
        if index.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
            check_conv_transpose_deadline(deadline, "finite-bias scan")?;
        }
        if !value.is_finite() {
            check_conv_transpose_deadline(deadline, "conservative composition return")?;
            return Ok(LinearBounds::conservative(output_dim, n));
        }
    }
    let in_size = geo.in_size();
    let (ul, uu) = upstream_slices(upstream, node_name)?;

    // Higham factor for a fan-in-many f64 multiply-accumulate, +4 for the
    // bias add and the center/radius combination (mirrors the Conv2d seam).
    let gamma = gamma_n_f64(geo.fan_in().saturating_add(4));
    let spatial = geo.out_h * geo.out_w;

    let mut new_lower_a = Array2::<f32>::zeros((output_dim, n));
    let mut new_upper_a = Array2::<f32>::zeros((output_dim, n));
    let mut penalty_l = vec![0.0f64; output_dim];
    let mut penalty_u = vec![0.0f64; output_dim];

    // Chunk the network-input columns so the transient f64 row state stays
    // bounded (~256 MB): each column costs 3 rows of `in_size` f64 (center
    // through `W`, radius and the `S` base through `|W|`).
    let budget_rows = (256usize << 20) / 8 / in_size.max(1);
    let chunk_cols = (budget_rows / 3).clamp(1, 1024).min(n.max(1));
    let mut j0 = 0usize;
    while j0 < n {
        check_conv_transpose_deadline(deadline, "coefficient chunk admission")?;
        let cb = chunk_cols.min(n - j0);
        // Row batches, one flattened [C,H,W] map per row.
        //   signed : [A_c columns (cb)]                        -> W
        //   abs    : [A_r columns (cb), |A_c|+|A_r| (cb)]      -> |W|
        let mut signed_rows = vec![0.0f64; cb * in_size];
        let mut abs_rows = vec![0.0f64; 2 * cb * in_size];
        let mut pack_work = 0usize;
        for jj in 0..cb {
            let j = j0 + jj;
            for k in 0..in_size {
                if pack_work.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
                    check_conv_transpose_deadline(deadline, "coefficient packing")?;
                }
                pack_work = pack_work.saturating_add(1);
                let l = f64::from(ul[k * n + j]);
                let u = f64::from(uu[k * n + j]);
                // Exact in f64 for f32 operands (`f64::midpoint` non-overflow path).
                let c = f64::midpoint(l, u);
                let r = (u - l) * 0.5;
                signed_rows[jj * in_size + k] = c;
                abs_rows[jj * in_size + k] = r;
                abs_rows[(cb + jj) * in_size + k] = c.abs() + r.abs();
            }
        }
        let signed =
            conv_transpose_apply_rows_f64(&signed_rows, cb, &layer.kernel, false, &geo, deadline)?;
        let absolute =
            conv_transpose_apply_rows_f64(&abs_rows, 2 * cb, &layer.kernel, true, &geo, deadline)?;

        let la_flat = new_lower_a.as_slice_mut().expect("row-major lower_a");
        let ua_flat = new_upper_a.as_slice_mut().expect("row-major upper_a");
        let cancelled = AtomicBool::new(false);
        la_flat
            .par_chunks_mut(n)
            .zip(ua_flat.par_chunks_mut(n))
            .zip(penalty_l.par_iter_mut().zip(penalty_u.par_iter_mut()))
            .enumerate()
            .for_each(|(p, ((lrow, urow), (pl, pu)))| {
                if let Some(deadline) = deadline {
                    if cancelled.load(Ordering::Relaxed) || Instant::now() >= deadline {
                        cancelled.store(true, Ordering::Relaxed);
                        return;
                    }
                }
                let mut work = 0usize;
                for jj in 0..cb {
                    if poll_conv_transpose_deadline(&mut work, &cancelled, deadline) {
                        return;
                    }
                    let j = j0 + jj;
                    let mag = input_mag[j];
                    let c = signed[jj * output_dim + p];
                    let r = absolute[jj * output_dim + p];
                    let s = absolute[(cb + jj) * output_dim + p];
                    lrow[j] = cast_coeff_with_gap(c - r, mag, pl);
                    urow[j] = cast_coeff_with_gap(c + r, mag, pu);
                    let accum = safe_mul_for_bounds_f64(gamma * s, mag);
                    *pl += accum;
                    *pu += accum;
                }
            });
        if deadline.is_some() && cancelled.load(Ordering::Relaxed) {
            return Err(NyError::DeadlineExceeded(
                "forward-linear ConvTranspose2d: deadline exceeded during coefficient commit"
                    .into(),
            ));
        }
        check_conv_transpose_deadline(deadline, "coefficient chunk completion")?;
        j0 += cb;
    }

    // Bias: the same identity on `b_c`/`b_r`, plus the layer bias.
    let mut bias_signed = vec![0.0f64; in_size];
    let mut bias_abs = vec![0.0f64; 2 * in_size];
    {
        let up_lb = upstream.lower_b();
        let up_ub = upstream.upper_b();
        for k in 0..in_size {
            if k.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
                check_conv_transpose_deadline(deadline, "bias packing")?;
            }
            let l = f64::from(up_lb[k]);
            let u = f64::from(up_ub[k]);
            let c = f64::midpoint(l, u);
            let r = (u - l) * 0.5;
            bias_signed[k] = c;
            bias_abs[k] = r;
            bias_abs[in_size + k] = c.abs() + r.abs();
        }
    }
    let bias_center =
        conv_transpose_apply_rows_f64(&bias_signed, 1, &layer.kernel, false, &geo, deadline)?;
    let bias_radius =
        conv_transpose_apply_rows_f64(&bias_abs, 2, &layer.kernel, true, &geo, deadline)?;

    let mut new_lower_b = Array1::<f32>::zeros(output_dim);
    let mut new_upper_b = Array1::<f32>::zeros(output_dim);
    for p in 0..output_dim {
        if p.is_multiple_of(CONV_TRANSPOSE_DEADLINE_POLL_WORK) {
            check_conv_transpose_deadline(deadline, "bias commit")?;
        }
        let oc = p / spatial;
        let layer_bias = layer.bias.as_ref().map_or(0.0f64, |b| f64::from(b[oc]));
        let c = bias_center[p];
        let r = bias_radius[p];
        let s = bias_radius[output_dim + p];
        let gamma_pen = gamma * s;
        new_lower_b[p] = commit_lower_bias(c - r + layer_bias, penalty_l[p] + gamma_pen);
        new_upper_b[p] = commit_upper_bias(c + r + layer_bias, penalty_u[p] + gamma_pen);
    }

    let bounds = finish_conv_transpose_bounds_with_poll(
        new_lower_a,
        new_lower_b,
        new_upper_a,
        new_upper_b,
        n,
        &format!("forward-linear ConvTranspose2d '{node_name}'"),
        |context| check_conv_transpose_deadline(deadline, context),
    )?;
    check_conv_transpose_deadline(deadline, "composition completion")?;
    Ok(bounds)
}

/// Compose a shape-aware inference BatchNorm through the forward affine map.
/// The nominal per-channel scale is an exact diagonal affine composition; its
/// f64->f32 coefficient cast gap is discharged through `input_mag`.  The
/// existing BatchNorm precompute-error bounds are expanded with the SAME
/// channel-layout decoder used by IBP/CROWN, then folded as
/// `scale_err*max(|pre_l|,|pre_u|) + bias_err` into each output bias.
pub(super) fn compose_batch_norm_forward(
    node_name: &str,
    layer: &BatchNormLayer,
    upstream: &LinearBounds,
    pre_activation: &BoundedTensor,
    output_dim: usize,
    input_mag: &[f64],
) -> Result<LinearBounds> {
    let n = upstream.num_inputs();
    if upstream.num_outputs() != output_dim || pre_activation.len() != output_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![upstream.num_outputs()],
            got: vec![output_dim.max(pre_activation.len())],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let (scale, bias, scale_err, bias_err) =
        layer.expanded_affine_parameters(pre_activation.shape(), output_dim)?;
    if scale_err
        .iter()
        .chain(bias_err.iter())
        .any(|&err| err.is_nan() || err < 0.0)
    {
        return Err(NyError::NumericalInstability(format!(
            "forward-linear BatchNorm '{node_name}' has NaN or negative certified error metadata"
        )));
    }
    let pre_l: Vec<f32> = pre_activation.lower().iter().copied().collect();
    let pre_u: Vec<f32> = pre_activation.upper().iter().copied().collect();

    let mut new_lower_a = Array2::<f32>::zeros((output_dim, n));
    let mut new_upper_a = Array2::<f32>::zeros((output_dim, n));
    let mut new_lower_b = Array1::<f32>::zeros(output_dim);
    let mut new_upper_b = Array1::<f32>::zeros(output_dim);
    for p in 0..output_dim {
        let s = scale[p] as f64;
        let b = bias[p] as f64;
        let (src_lower_a, src_lower_b, src_upper_a, src_upper_b) = if s >= 0.0 {
            (
                upstream.lower_a(),
                upstream.lower_b()[p],
                upstream.upper_a(),
                upstream.upper_b()[p],
            )
        } else {
            (
                upstream.upper_a(),
                upstream.upper_b()[p],
                upstream.lower_a(),
                upstream.lower_b()[p],
            )
        };

        let mut penalty_l = 0.0f64;
        let mut penalty_u = 0.0f64;
        for j in 0..n {
            let lower_exact = safe_mul_for_bounds_f64(s, src_lower_a[[p, j]] as f64);
            let upper_exact = safe_mul_for_bounds_f64(s, src_upper_a[[p, j]] as f64);
            new_lower_a[[p, j]] = cast_coeff_with_gap(lower_exact, input_mag[j], &mut penalty_l);
            new_upper_a[[p, j]] = cast_coeff_with_gap(upper_exact, input_mag[j], &mut penalty_u);
        }

        // The true (unrounded-precompute) BN affine differs from the stored
        // scale/bias by this constant over the certified pre-activation box.
        let xmag = (pre_l[p] as f64).abs().max((pre_u[p] as f64).abs());
        let parameter_margin =
            safe_mul_for_bounds_f64(xmag, scale_err[p] as f64) + bias_err[p] as f64;
        penalty_l += parameter_margin;
        penalty_u += parameter_margin;

        let lower_base = safe_mul_for_bounds_f64(s, src_lower_b as f64) + b;
        let upper_base = safe_mul_for_bounds_f64(s, src_upper_b as f64) + b;
        new_lower_b[p] = commit_lower_bias(lower_base, penalty_l);
        new_upper_b[p] = commit_upper_bias(upper_base, penalty_u);
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear BatchNorm '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Compose the upstream forward-linear bounds through a dense affine layer
/// `y = W h + b` (Linear/Gemm) with certified rounding (see module docs).
#[allow(clippy::too_many_arguments)]
pub(super) fn compose_dense_affine_forward(
    node_name: &str,
    weight: &Array2<f32>,
    bias: Option<&Array1<f32>>,
    upstream: &LinearBounds,
    input_mag: &[f64],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    // Seam gate for the value GEMMs; `None` reads `NY_FORWARD_LINEAR_F32`.
    use_f32_override: Option<bool>,
) -> Result<LinearBounds> {
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return Err(NyError::DeadlineExceeded(format!(
            "forward-linear Linear '{node_name}': deadline exceeded before composition"
        )));
    }
    let use_f32 = use_f32_override.unwrap_or_else(forward_linear_f32_gemm_enabled);
    let m = weight.nrows();
    let k = weight.ncols();
    let n = upstream.num_inputs();
    if upstream.num_outputs() != k {
        return Err(NyError::ShapeMismatch {
            expected: vec![k],
            got: vec![upstream.num_outputs()],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }

    let up_l = upstream.lower_a();
    let up_u = upstream.upper_a();
    let up_lb = upstream.lower_b();
    let up_ub = upstream.upper_b();

    // Center/radius of the upstream coefficients (exact in f64) + the
    // mag-weighted column S base for the γ penalty.
    let mut uc = vec![0.0f64; k * n];
    let mut ur = vec![0.0f64; k * n];
    let mut w_s = vec![0.0f64; k];
    for kk in 0..k {
        for j in 0..n {
            if (kk.saturating_mul(n).saturating_add(j)).is_multiple_of(4096)
                && deadline.is_some_and(|value| Instant::now() >= value)
            {
                return Err(NyError::DeadlineExceeded(format!(
                    "forward-linear Linear '{node_name}': deadline exceeded during operand construction"
                )));
            }
            let l = up_l[[kk, j]] as f64;
            let u = up_u[[kk, j]] as f64;
            // Bit-identical (f32-cast operands, f64::midpoint fast path).
            let c = f64::midpoint(l, u);
            let r = (u - l) * 0.5;
            uc[kk * n + j] = c;
            ur[kk * n + j] = r;
            w_s[kk] += (c.abs() + r.abs()) * input_mag[j];
        }
    }
    let w64: Vec<f64> = weight.iter().map(|&v| v as f64).collect();
    let wabs64: Vec<f64> = weight.iter().map(|&v| (v as f64).abs()).collect();

    // Value GEMMs (center/radius): routed to the sound f32 fast path iff on.
    let g_center = certified_f64_gemm(m, k, n, &w64, &uc, engine, use_f32, deadline)?;
    let g_radius = certified_f64_gemm(m, k, n, &wabs64, &ur, engine, use_f32, deadline)?;

    let mut new_lower_a = Array2::<f32>::zeros((m, n));
    let mut new_upper_a = Array2::<f32>::zeros((m, n));
    let mut penalty_l = vec![0.0f64; m];
    let mut penalty_u = vec![0.0f64; m];
    for i in 0..m {
        for j in 0..n {
            if (i.saturating_mul(n).saturating_add(j)).is_multiple_of(4096)
                && deadline.is_some_and(|value| Instant::now() >= value)
            {
                return Err(NyError::DeadlineExceeded(format!(
                    "forward-linear Linear '{node_name}': deadline exceeded during coefficient commit"
                )));
            }
            let c = g_center[i * n + j];
            let r = g_radius[i * n + j];
            new_lower_a[[i, j]] = cast_coeff_with_gap(c - r, input_mag[j], &mut penalty_l[i]);
            new_upper_a[[i, j]] = cast_coeff_with_gap(c + r, input_mag[j], &mut penalty_u[i]);
        }
    }

    // Value GEMMs (`g_center`/`g_radius`) ran in f32 iff the seam is on; the
    // bias terms below are exact f64 CPU sums, so only the coefficient error
    // grows to `γ^f32`. `γ^f32 ≥ γ^f64` conservatively covers the bias `s_bias`.
    let gamma = if use_f32 {
        gamma_n_f32(k + 4)
    } else {
        gamma_n_f64(k + 4)
    };
    let ftz = if use_f32 {
        let mag_sum: f64 = input_mag.iter().sum();
        forward_f32_ftz_bias(k, mag_sum)
    } else {
        0.0
    };
    let mut new_lower_b = Array1::<f32>::zeros(m);
    let mut new_upper_b = Array1::<f32>::zeros(m);
    for i in 0..m {
        if deadline.is_some_and(|value| Instant::now() >= value) {
            return Err(NyError::DeadlineExceeded(format!(
                "forward-linear Linear '{node_name}': deadline exceeded during bias composition"
            )));
        }
        let mut vc = 0.0f64; // W  @ u_c
        let mut vr = 0.0f64; // |W| @ u_r
        let mut s_bias = 0.0f64; // |W| @ (|u_c|+|u_r|)
        let mut s_coeff = 0.0f64; // |W| @ w_s
        for kk in 0..k {
            let w = weight[[i, kk]] as f64;
            let wa = w.abs();
            let l = up_lb[kk] as f64;
            let u = up_ub[kk] as f64;
            // Bit-identical (f32-cast operands, f64::midpoint fast path).
            let c = f64::midpoint(l, u);
            let r = (u - l) * 0.5;
            vc += w * c;
            vr += wa * r;
            s_bias += wa * (c.abs() + r.abs());
            s_coeff += wa * w_s[kk];
        }
        let b = bias.map_or(0.0f64, |b| b[i] as f64);
        let gamma_pen = gamma * (s_coeff + s_bias) + ftz;
        new_lower_b[i] = commit_lower_bias(vc - vr + b, penalty_l[i] + gamma_pen);
        new_upper_b[i] = commit_upper_bias(vc + vr + b, penalty_u[i] + gamma_pen);
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear Linear '{node_name}'"),
    );
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return Err(NyError::DeadlineExceeded(format!(
            "forward-linear Linear '{node_name}': deadline exceeded before return"
        )));
    }
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Compose the upstream forward-linear bounds through a ReLU node using the
/// per-neuron diagonal relaxation (no dense identity materialization — the
/// generic identity-trick path is O(N²) memory and infeasible at image scale).
///
/// Uses the production `relu_linear_relaxation` (matched to the reference;
/// sound chord upper with bumped intercept). Slope·coefficient products are
/// exact in f64 (f32×f32 fits in 53 bits); only the measured f32 cast gap is
/// discharged. Rows with non-finite relaxation intercepts (NaN/±inf
/// pre-activation) degrade to `A=0, b=±inf`.
///
/// # Optimized lower slopes (#w4-root-alpha)
///
/// `alpha_lower`, when present, supplies per-neuron lower slopes (e.g. the
/// alpha-CROWN warmup's optimized values). For a CROSSING neuron
/// (finite `l < 0 < u`) the lower relaxation `y >= α·x + 0` is sound for ANY
/// `α ∈ [0, 1]` with intercept exactly 0, independent of `l`/`u`:
/// on `x ∈ [l, 0]`, `ReLU(x) = 0 >= α·x` (α >= 0, x <= 0); on `x ∈ [0, u]`,
/// `ReLU(x) = x >= α·x` (α <= 1, x >= 0). The adaptive rule is the
/// `α ∈ {0, 1}` special case, so feeding the adaptive value reproduces the
/// legacy composition bit-for-bit. Stable neurons keep their exact
/// identity/zero relaxation (α ignored); NaN α falls back to the adaptive
/// rule; values outside [0, 1] are clamped. The UPPER relaxation is always
/// the sound chord — never touched by α.
pub(super) fn compose_relu_diag_forward(
    node_name: &str,
    upstream: &LinearBounds,
    pre_activation: &BoundedTensor,
    input_mag: &[f64],
    alpha_lower: Option<&[f32]>,
) -> Result<LinearBounds> {
    let m = upstream.num_outputs();
    let n = upstream.num_inputs();
    let pre_flat = pre_activation.flatten();
    if pre_flat.len() != m {
        return Err(NyError::ShapeMismatch {
            expected: vec![m],
            got: vec![pre_flat.len()],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let pre_l: Vec<f32> = pre_flat.lower().iter().copied().collect();
    let pre_u: Vec<f32> = pre_flat.upper().iter().copied().collect();

    let up_lb = upstream.lower_b();
    let up_ub = upstream.upper_b();
    let (ul, uu) = upstream_slices(upstream, node_name)?;

    let mut new_lower_a = Array2::<f32>::zeros((m, n));
    let mut new_upper_a = Array2::<f32>::zeros((m, n));
    let mut new_lower_b = Array1::<f32>::zeros(m);
    let mut new_upper_b = Array1::<f32>::zeros(m);

    {
        let la_flat = new_lower_a.as_slice_mut().expect("row-major lower_a");
        let ua_flat = new_upper_a.as_slice_mut().expect("row-major upper_a");
        let lb_flat = new_lower_b.as_slice_mut().expect("contiguous lower_b");
        let ub_flat = new_upper_b.as_slice_mut().expect("contiguous upper_b");
        la_flat
            .par_chunks_mut(n)
            .zip(ua_flat.par_chunks_mut(n))
            .zip(lb_flat.par_iter_mut().zip(ub_flat.par_iter_mut()))
            .enumerate()
            .for_each(|(i, ((lrow, urow), (lb, ub)))| {
                let relax = relu_linear_relaxation(pre_l[i], pre_u[i]);

                // Lower side: y_i >= d·h_i + c with d = lower_slope >= 0 for
                // ReLU; written sign-generally (d < 0 selects the upstream
                // UPPER side) so the helper stays sound for future monotone
                // activations.
                //
                // #w4-root-alpha: a caller-supplied α replaces the adaptive
                // lower slope ONLY for finite crossing neurons, where
                // `y >= α·x` is sound with intercept 0 for any α ∈ [0, 1]
                // (see fn docs). All other cases keep the proven adaptive
                // relaxation (including its NaN/±inf fallbacks).
                let crossing = pre_l[i] < 0.0
                    && pre_u[i] > 0.0
                    && pre_l[i].is_finite()
                    && pre_u[i].is_finite();
                let optimized = alpha_lower.and_then(|alpha| {
                    let a = alpha[i];
                    (crossing && a.is_finite()).then(|| f64::from(a.clamp(0.0, 1.0)))
                });
                let (d, c) = match optimized {
                    Some(a) => (a, 0.0f64),
                    None => (relax.lower_slope as f64, relax.lower_intercept as f64),
                };
                if !d.is_finite() || !c.is_finite() {
                    *lb = f32::NEG_INFINITY;
                } else if d == 0.0 {
                    *lb = commit_lower_bias(c, 0.0);
                } else {
                    let (src_a, src_b) = if d >= 0.0 {
                        (&ul[i * n..i * n + n], up_lb[i] as f64)
                    } else {
                        (&uu[i * n..i * n + n], up_ub[i] as f64)
                    };
                    let mut pen = 0.0f64;
                    for j in 0..n {
                        // f32×f32 in f64 is exact; only the cast gap is discharged.
                        lrow[j] =
                            cast_coeff_with_gap(d * (src_a[j] as f64), input_mag[j], &mut pen);
                    }
                    *lb = commit_lower_bias(d * src_b + c, pen);
                }

                // Upper side: y_i <= d·h_i + c.
                let d = relax.upper_slope as f64;
                let c = relax.upper_intercept as f64;
                if !d.is_finite() || !c.is_finite() {
                    for v in urow.iter_mut() {
                        *v = 0.0;
                    }
                    *ub = f32::INFINITY;
                } else if d == 0.0 {
                    *ub = commit_upper_bias(c, 0.0);
                } else {
                    let (src_a, src_b) = if d >= 0.0 {
                        (&uu[i * n..i * n + n], up_ub[i] as f64)
                    } else {
                        (&ul[i * n..i * n + n], up_lb[i] as f64)
                    };
                    let mut pen = 0.0f64;
                    for j in 0..n {
                        urow[j] =
                            cast_coeff_with_gap(d * (src_a[j] as f64), input_mag[j], &mut pen);
                    }
                    *ub = commit_upper_bias(d * src_b + c, pen);
                }
            });
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear ReLU '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Compose the upstream forward-linear bounds through a Tanh node using its
/// certified per-neuron diagonal relaxation. This is the image-scale analogue
/// of the generic forward-linear identity composition, without its O(N^2)
/// identity materialization. Coefficient products are formed in f64 and every
/// f32 cast gap is discharged into the outward-rounded bias.
pub(super) fn compose_tanh_diag_forward(
    node_name: &str,
    upstream: &LinearBounds,
    pre_activation: &BoundedTensor,
    input_mag: &[f64],
) -> Result<LinearBounds> {
    let m = upstream.num_outputs();
    let n = upstream.num_inputs();
    let pre_flat = pre_activation.flatten();
    if pre_flat.len() != m {
        return Err(NyError::ShapeMismatch {
            expected: vec![m],
            got: vec![pre_flat.len()],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let pre_l: Vec<f32> = pre_flat.lower().iter().copied().collect();
    let pre_u: Vec<f32> = pre_flat.upper().iter().copied().collect();

    let up_lb = upstream.lower_b();
    let up_ub = upstream.upper_b();
    let (ul, uu) = upstream_slices(upstream, node_name)?;

    let mut new_lower_a = Array2::<f32>::zeros((m, n));
    let mut new_upper_a = Array2::<f32>::zeros((m, n));
    let mut new_lower_b = Array1::<f32>::zeros(m);
    let mut new_upper_b = Array1::<f32>::zeros(m);

    {
        let la_flat = new_lower_a.as_slice_mut().expect("row-major lower_a");
        let ua_flat = new_upper_a.as_slice_mut().expect("row-major upper_a");
        let lb_flat = new_lower_b.as_slice_mut().expect("contiguous lower_b");
        let ub_flat = new_upper_b.as_slice_mut().expect("contiguous upper_b");
        la_flat
            .par_chunks_mut(n)
            .zip(ua_flat.par_chunks_mut(n))
            .zip(lb_flat.par_iter_mut().zip(ub_flat.par_iter_mut()))
            .enumerate()
            .for_each(|(i, ((lrow, urow), (lb, ub)))| {
                let relax = tanh_linear_relaxation(pre_l[i], pre_u[i]);

                // Lower side: y_i >= d*h_i+c. Tanh's finite relaxation slopes
                // are nonnegative, but retain sign-general substitution for
                // conservative non-finite/future relaxation fallbacks.
                let d = relax.lower_slope as f64;
                let c = relax.lower_intercept as f64;
                if !d.is_finite() || !c.is_finite() {
                    *lb = f32::NEG_INFINITY;
                } else if d == 0.0 {
                    *lb = commit_lower_bias(c, 0.0);
                } else {
                    let (src_a, src_b) = if d >= 0.0 {
                        (&ul[i * n..i * n + n], up_lb[i] as f64)
                    } else {
                        (&uu[i * n..i * n + n], up_ub[i] as f64)
                    };
                    let mut pen = 0.0f64;
                    for j in 0..n {
                        lrow[j] =
                            cast_coeff_with_gap(d * (src_a[j] as f64), input_mag[j], &mut pen);
                    }
                    *lb = commit_lower_bias(d * src_b + c, pen);
                }

                // Upper side: y_i <= d*h_i+c.
                let d = relax.upper_slope as f64;
                let c = relax.upper_intercept as f64;
                if !d.is_finite() || !c.is_finite() {
                    for value in urow.iter_mut() {
                        *value = 0.0;
                    }
                    *ub = f32::INFINITY;
                } else if d == 0.0 {
                    *ub = commit_upper_bias(c, 0.0);
                } else {
                    let (src_a, src_b) = if d >= 0.0 {
                        (&uu[i * n..i * n + n], up_ub[i] as f64)
                    } else {
                        (&ul[i * n..i * n + n], up_lb[i] as f64)
                    };
                    let mut pen = 0.0f64;
                    for j in 0..n {
                        urow[j] =
                            cast_coeff_with_gap(d * (src_a[j] as f64), input_mag[j], &mut pen);
                    }
                    *ub = commit_upper_bias(d * src_b + c, pen);
                }
            });
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear Tanh '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Compose a binary residual Add: `y = h_a + h_b`. The Jacobian is the
/// identity toward both parents (both weights +1 ≥ 0), so lower composes with
/// lowers and upper with uppers. Coefficient sums are exact in f64; the
/// measured f32 cast gap is discharged into the bias.
pub(super) fn compose_add_forward(
    node_name: &str,
    a: &LinearBounds,
    b: &LinearBounds,
    input_mag: &[f64],
) -> Result<LinearBounds> {
    let m = a.num_outputs();
    let n = a.num_inputs();
    if b.num_outputs() != m || b.num_inputs() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![b.num_outputs(), b.num_inputs()],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let (alb, aub) = (a.lower_b(), a.upper_b());
    let (blb, bub) = (b.lower_b(), b.upper_b());
    let (als, aus) = upstream_slices(a, node_name)?;
    let (bls, bus) = upstream_slices(b, node_name)?;

    let mut new_lower_a = Array2::<f32>::zeros((m, n));
    let mut new_upper_a = Array2::<f32>::zeros((m, n));
    let mut new_lower_b = Array1::<f32>::zeros(m);
    let mut new_upper_b = Array1::<f32>::zeros(m);
    {
        let la_flat = new_lower_a.as_slice_mut().expect("row-major lower_a");
        let ua_flat = new_upper_a.as_slice_mut().expect("row-major upper_a");
        let lb_flat = new_lower_b.as_slice_mut().expect("contiguous lower_b");
        let ub_flat = new_upper_b.as_slice_mut().expect("contiguous upper_b");
        la_flat
            .par_chunks_mut(n)
            .zip(ua_flat.par_chunks_mut(n))
            .zip(lb_flat.par_iter_mut().zip(ub_flat.par_iter_mut()))
            .enumerate()
            .for_each(|(i, ((lrow, urow), (lb, ub)))| {
                let arow_l = &als[i * n..i * n + n];
                let arow_u = &aus[i * n..i * n + n];
                let brow_l = &bls[i * n..i * n + n];
                let brow_u = &bus[i * n..i * n + n];
                let mut pen_l = 0.0f64;
                let mut pen_u = 0.0f64;
                for j in 0..n {
                    // f32+f32 in f64 is exact; only the f32 cast gap is discharged.
                    let lo = arow_l[j] as f64 + brow_l[j] as f64;
                    let hi = arow_u[j] as f64 + brow_u[j] as f64;
                    lrow[j] = cast_coeff_with_gap(lo, input_mag[j], &mut pen_l);
                    urow[j] = cast_coeff_with_gap(hi, input_mag[j], &mut pen_u);
                }
                *lb = commit_lower_bias(alb[i] as f64 + blb[i] as f64, pen_l);
                *ub = commit_upper_bias(aub[i] as f64 + bub[i] as f64, pen_u);
            });
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear Add '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ny_core::NaiveCpuGemmEngine;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

    struct RejectOpaqueGemm;

    impl GemmEngine for RejectOpaqueGemm {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("finite-deadline forward-linear work entered opaque f32 GEMM")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            panic!("finite-deadline forward-linear work entered opaque f64 GEMM")
        }
    }

    #[derive(Default)]
    struct RecordingDeadlineF64 {
        calls: Mutex<Vec<(usize, usize, usize, usize)>>,
    }

    impl GemmEngine for RecordingDeadlineF64 {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("deadline-bounded f64 routing entered ordinary f32 GEMM")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            panic!("deadline-bounded f64 routing entered ordinary f64 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f64],
            b: &[f64],
            _deadline: Instant,
            max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            self.calls
                .lock()
                .expect("recording lock")
                .push((m, k, n, max_dispatch_macs));
            NaiveCpuGemmEngine.gemm_f64(m, k, n, a, b)
        }
    }

    struct TerminalDeadlineF64;

    impl GemmEngine for TerminalDeadlineF64 {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("deadline terminal test entered ordinary f32 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            Err(NyError::DeadlineExceeded(
                "injected terminal deadline".into(),
            ))
        }
    }

    struct MalformedDeadlineF64;

    impl GemmEngine for MalformedDeadlineF64 {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("malformed deadline test entered ordinary f32 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            Ok(vec![f64::NAN])
        }
    }

    struct SleepPastDeadlineF64 {
        calls: AtomicUsize,
    }

    impl GemmEngine for SleepPastDeadlineF64 {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("post-expiry test entered ordinary f32 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f64],
            _b: &[f64],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(30));
            Ok(vec![0.0; m * n])
        }
    }

    fn exact_fixture() -> (Array2<f32>, LinearBounds, Vec<f64>) {
        let weight = arr2(&[[2.0f32, -1.0], [1.0, 3.0]]);
        let upstream = LinearBounds::new(
            arr2(&[[1.0f32, 0.0], [0.0, 1.0]]),
            arr1(&[0.0f32, 0.0]),
            arr2(&[[1.0f32, 0.0], [0.0, 1.0]]),
            arr1(&[0.0f32, 0.0]),
        )
        .expect("valid exact affine fixture");
        (weight, upstream, vec![1.0, 1.0])
    }

    #[test]
    fn finite_deadline_dense_composition_never_enters_opaque_engine() {
        let (weight, upstream, input_mag) = exact_fixture();
        let bounded = compose_dense_affine_forward(
            "deadline-test",
            &weight,
            None,
            &upstream,
            &input_mag,
            Some(&RejectOpaqueGemm),
            Some(Instant::now() + Duration::from_mins(1)),
            Some(false),
        )
        .expect("pollable CPU composition");
        let baseline = compose_dense_affine_forward(
            "baseline",
            &weight,
            None,
            &upstream,
            &input_mag,
            None,
            None,
            Some(false),
        )
        .expect("historical unbounded composition");

        assert_eq!(bounded.lower_a(), baseline.lower_a());
        assert_eq!(bounded.upper_a(), baseline.upper_a());
        assert_eq!(bounded.lower_b(), baseline.lower_b());
        assert_eq!(bounded.upper_b(), baseline.upper_b());
    }

    #[test]
    fn finite_deadline_selects_explicit_bounded_f64_contract() {
        let engine = RecordingDeadlineF64::default();
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [2.0, -1.0, 0.5, 3.0, -2.0, 4.0];
        let deadline = Instant::now() + Duration::from_mins(1);
        let accelerated = certified_f64_gemm(2, 3, 2, &a, &b, Some(&engine), false, Some(deadline))
            .expect("explicit bounded f64 engine");
        let expected = NaiveCpuGemmEngine
            .gemm_f64(2, 3, 2, &a, &b)
            .expect("CPU oracle");

        assert_eq!(accelerated, expected);
        assert_eq!(
            *engine.calls.lock().expect("recording lock"),
            vec![(2, 3, 2, DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS)]
        );
    }

    #[test]
    fn deadline_error_from_bounded_engine_is_terminal() {
        let deadline = Instant::now() + Duration::from_mins(1);
        let error = certified_f64_gemm(
            1,
            1,
            1,
            &[2.0],
            &[3.0],
            Some(&TerminalDeadlineF64),
            false,
            Some(deadline),
        )
        .expect_err("deadline engine error must propagate");
        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
    }

    #[test]
    fn malformed_bounded_engine_result_falls_back_to_pollable_cpu() {
        let deadline = Instant::now() + Duration::from_mins(1);
        let result = certified_f64_gemm(
            1,
            2,
            1,
            &[2.0, 3.0],
            &[4.0, 5.0],
            Some(&MalformedDeadlineF64),
            false,
            Some(deadline),
        )
        .expect("malformed engine result must use CPU fallback");
        assert_eq!(result, vec![23.0]);
    }

    #[test]
    fn bounded_engine_result_completed_after_deadline_is_never_published() {
        let engine = SleepPastDeadlineF64 {
            calls: AtomicUsize::new(0),
        };
        let deadline = Instant::now() + Duration::from_millis(10);
        let error =
            certified_f64_gemm_deadline_try_engine(&engine, 1, 1, 1, &[2.0], &[3.0], deadline)
                .expect_err("post-deadline result must be discarded");

        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn expired_deadline_refuses_before_explicit_engine_launch() {
        let engine = SleepPastDeadlineF64 {
            calls: AtomicUsize::new(0),
        };
        let deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before now");
        let error =
            certified_f64_gemm_deadline_try_engine(&engine, 1, 1, 1, &[2.0], &[3.0], deadline)
                .expect_err("expired deadline must refuse");

        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn expired_dense_composition_refuses_before_engine_launch() {
        let (weight, upstream, input_mag) = exact_fixture();
        let error = compose_dense_affine_forward(
            "expired-test",
            &weight,
            None,
            &upstream,
            &input_mag,
            Some(&RejectOpaqueGemm),
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("one millisecond fits before the current instant"),
            ),
            Some(false),
        )
        .expect_err("expired composition must refuse");

        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
    }

    // --- #fl-value-gpu-tier seam integration ---------------------------------

    /// Switchable stand-in for the wgpu `FlValueGemmDevice`: same trait
    /// surface (ONLY `gemm_f32_with_deadline` is live; ordinary GEMM panics so
    /// a misroute is loud), RN-f32 values via the naive CPU reference. Modes:
    /// 0 = typed refusal (size/memory-class), 1 = serve, 2 = non-finite poison.
    struct SwitchableFlValueMock;

    static FL_MOCK_MODE: AtomicUsize = AtomicUsize::new(0);
    static FL_MOCK_CALLS: AtomicUsize = AtomicUsize::new(0);

    impl GemmEngine for SwitchableFlValueMock {
        fn backend_provenance(&self) -> &'static str {
            "fl-value-mock"
        }

        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("FL-value channel must never route ordinary f32 GEMM")
        }

        fn gemm_f32_with_deadline(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f32],
            b: &[f32],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f32>> {
            FL_MOCK_CALLS.fetch_add(1, Ordering::SeqCst);
            match FL_MOCK_MODE.load(Ordering::SeqCst) {
                0 => Err(NyError::UnsupportedConfiguration(
                    "mock: below measured crossover".into(),
                )),
                1 => NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b),
                _ => Ok(vec![f32::NAN; m * n]),
            }
        }
    }

    /// One test owns the process-global FL-value registry (first-install-wins
    /// `OnceLock`), exercising every tier outcome deterministically:
    ///
    /// 1. serve  → the GPU tier is taken BEFORE the CPU f32 tier (telemetry
    ///    hit counter increments) and the published product equals the CPU
    ///    f32 tier's product (exact fixture ⇒ bit-equal; in general both are
    ///    RN-f32 orders covered by the same `γ^f32·S`+FTZ charge);
    /// 2. refuse → falls through to the CPU f32 tier, same result, no hit;
    /// 3. poison → non-finite engine result is never published, falls
    ///    through, no hit;
    /// 4. flag-off (allow_f32=false) → the registry is never consulted and
    ///    the composition is bit-identical to the engine-free baseline; and
    /// 5. the dense composition seam (`compose_dense_affine_forward`) yields
    ///    identical bounds through the GPU tier and the CPU f32 tier.
    #[test]
    fn fl_value_gpu_tier_seam_integration() {
        crate::fl_value_gemm::set_fl_value_gemm_engine(std::sync::Arc::new(SwitchableFlValueMock));
        let deadline = Instant::now() + Duration::from_mins(10);
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [2.0f64, -1.0, 0.5, 3.0, -2.0, 4.0];
        let cpu_f32 = certified_f32_gemm_deadline_cpu(2, 3, 2, &a, &b, deadline)
            .expect("CPU f32 tier")
            .expect("finite CPU f32 product");

        // (1) serve: GPU tier taken, hit recorded, product matches CPU f32.
        FL_MOCK_MODE.store(1, Ordering::SeqCst);
        let hits_before = crate::fl_value_gemm::telemetry_snapshot().hits;
        let calls_before = FL_MOCK_CALLS.load(Ordering::SeqCst);
        let served = certified_f64_gemm(2, 3, 2, &a, &b, None, true, Some(deadline))
            .expect("served GPU tier");
        assert_eq!(served, cpu_f32, "GPU tier product must match CPU f32 tier");
        assert_eq!(
            FL_MOCK_CALLS.load(Ordering::SeqCst),
            calls_before + 1,
            "engine must be consulted exactly once"
        );
        let snapshot = crate::fl_value_gemm::telemetry_snapshot();
        assert_eq!(
            snapshot.hits,
            hits_before + 1,
            "published GPU-tier result must record a telemetry hit"
        );
        assert_eq!(snapshot.backend, Some("fl-value-mock"));

        // (2) typed refusal: falls through to the CPU f32 tier, no hit.
        FL_MOCK_MODE.store(0, Ordering::SeqCst);
        let hits = crate::fl_value_gemm::telemetry_snapshot().hits;
        let refused = certified_f64_gemm(2, 3, 2, &a, &b, None, true, Some(deadline))
            .expect("refusal must fall through");
        assert_eq!(refused, cpu_f32);
        assert_eq!(crate::fl_value_gemm::telemetry_snapshot().hits, hits);

        // (3) non-finite poison: never published, falls through, no hit.
        FL_MOCK_MODE.store(2, Ordering::SeqCst);
        let hits = crate::fl_value_gemm::telemetry_snapshot().hits;
        let poisoned = certified_f64_gemm(2, 3, 2, &a, &b, None, true, Some(deadline))
            .expect("poison must fall through");
        assert_eq!(poisoned, cpu_f32);
        assert_eq!(crate::fl_value_gemm::telemetry_snapshot().hits, hits);

        // (4) flag-off bit-parity: allow_f32=false never consults the
        // registry and matches the engine-free f64 composition exactly.
        FL_MOCK_MODE.store(1, Ordering::SeqCst);
        let calls = FL_MOCK_CALLS.load(Ordering::SeqCst);
        let f64_deadline = certified_f64_gemm(2, 3, 2, &a, &b, None, false, Some(deadline))
            .expect("flag-off f64 path");
        let f64_baseline =
            certified_f64_gemm(2, 3, 2, &a, &b, None, false, None).expect("engine-free baseline");
        assert_eq!(f64_deadline, f64_baseline);
        assert_eq!(
            FL_MOCK_CALLS.load(Ordering::SeqCst),
            calls,
            "flag-off dispatch must never consult the FL-value registry"
        );

        // (5) composition-level bound parity through the seam: identical
        // charge (same `use_f32`), value products equal on the exact fixture
        // ⇒ identical LinearBounds through GPU-tier vs CPU-f32-tier routing.
        let (weight, upstream, input_mag) = exact_fixture();
        FL_MOCK_MODE.store(1, Ordering::SeqCst);
        let via_gpu = compose_dense_affine_forward(
            "fl-gpu-tier",
            &weight,
            None,
            &upstream,
            &input_mag,
            None,
            Some(deadline),
            Some(true),
        )
        .expect("composition through the GPU tier");
        FL_MOCK_MODE.store(0, Ordering::SeqCst);
        let via_cpu = compose_dense_affine_forward(
            "fl-cpu-tier",
            &weight,
            None,
            &upstream,
            &input_mag,
            None,
            Some(deadline),
            Some(true),
        )
        .expect("composition through the CPU f32 tier");
        assert_eq!(via_gpu.lower_a(), via_cpu.lower_a());
        assert_eq!(via_gpu.upper_a(), via_cpu.upper_a());
        assert_eq!(via_gpu.lower_b(), via_cpu.lower_b());
        assert_eq!(via_gpu.upper_b(), via_cpu.upper_b());
    }

    #[test]
    fn finite_deadline_gemm_tile_caps_rows_for_long_contractions() {
        const CAP: usize = 1 << 22;

        let (rows, cols) = certified_f64_gemm_deadline_tile_shape(64, CAP / 2, 128, CAP);
        assert_eq!((rows, cols), (2, 1));
        assert!(rows * (CAP / 2) * cols <= CAP);

        let (rows, cols) = certified_f64_gemm_deadline_tile_shape(1_000, 1_024, 1_000, CAP);
        assert!(rows <= 64);
        assert!(rows * 1_024 * cols <= CAP);
    }
}
