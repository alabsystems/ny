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
use std::time::Instant;
use tracing::debug;

use super::bias::{
    accumulate_bias_f64, add_coeff_err_bias_product_up, add_f64_down, add_f64_up, f32_to_f64_exact,
    finalize_bias_directed, nonnegative_f32_error_or_infinity, publish_error_up_normal,
    BiasBlockParams,
};
use super::layout::resolve_backward_layout;
use super::LinearLayer;
use crate::faer_parallelism::{mat_mul, mat_mul_f64};
use crate::{contiguous_flat_slice_mut, LinearBounds};

/// Measured CPU/GPU crossover for verdict-grade f64 `A·W`.
///
/// Keep deadline-bearing calls behind the same size gate as unbounded calls:
/// cold accelerator admission and host-image construction are not worthwhile
/// for smaller products, whose existing pollable CPU reduction is faster.
const SOUND_F64_GEMM_MIN_MACS: usize = 1 << 24;

/// The shared verdict-path constant above and the engine-facing default in
/// `ny-core` describe the SAME historical policy. Pin them together so neither
/// can drift: `SoundF64GemmAdmission::CONSTANT_FLOOR` must reproduce
/// `deadline_f64_accelerator_eligible` exactly for every engine that does not
/// override its declaration.
const _: () = assert!(SOUND_F64_GEMM_MIN_MACS == ny_core::SOUND_F64_GEMM_DEFAULT_MIN_MACS);

/// Engine-independent hard floor for the gated engine-aware admission path.
///
/// No engine declaration may open admission below this, whatever it claims. The
/// measured faer crossover is bracketed `512 < x <= 1,024` MACs, so 512 sits
/// strictly below every measured win and exists purely so a malformed or
/// over-eager declaration cannot route trivially small products through an
/// accelerator.
const ENGINE_AWARE_ABSOLUTE_MIN_MACS: usize = 512;

/// Environment gate for the engine-aware admission floor
/// (#b4-engine-aware-macs-floor).
///
/// UNSET (the default) ⇒ [`deadline_f64_accelerator_eligible`] is the whole
/// policy, exactly as before: one CUDA-tuned constant, no engine consulted, no
/// extra work beyond one relaxed atomic load. `NY_ENGINE_AWARE_MACS_FLOOR=1`
/// (also `true`/`yes`/`on`) additionally lets an ALREADY-MATERIALIZED engine
/// declare its own, measured crossover for sub-threshold shapes.
///
/// WHY GATED. `SOUND_F64_GEMM_MIN_MACS` is a shared verdict-path constant four
/// arcs depend on; this ships the mechanism and the measurement without
/// unilaterally moving the default.
const ENGINE_AWARE_MACS_FLOOR_ENV: &str = "NY_ENGINE_AWARE_MACS_FLOOR";

/// Tri-state cache for [`ENGINE_AWARE_MACS_FLOOR_ENV`]: `0` off, `1` on, `2`
/// unread.
static ENGINE_AWARE_MACS_FLOOR_GATE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(2);

#[cfg(test)]
thread_local! {
    /// Per-test-thread override; `None` keeps the process-global cache. Tests
    /// never mutate the production gate.
    static ENGINE_AWARE_MACS_FLOOR_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Force the gate for the current test thread only; `None` restores the
/// production cache. Never touches the process-global atomic.
#[cfg(test)]
fn set_engine_aware_macs_floor_for_test(value: Option<bool>) {
    ENGINE_AWARE_MACS_FLOOR_OVERRIDE.with(|cell| cell.set(value));
}

/// Whether the engine-aware admission floor is armed. Fails closed: anything
/// other than an explicit affirmative value leaves the historical policy.
#[inline]
fn engine_aware_macs_floor_armed() -> bool {
    #[cfg(test)]
    {
        if let Some(forced) = ENGINE_AWARE_MACS_FLOOR_OVERRIDE.with(std::cell::Cell::get) {
            return forced;
        }
    }
    use std::sync::atomic::Ordering;
    match ENGINE_AWARE_MACS_FLOOR_GATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let armed = std::env::var(ENGINE_AWARE_MACS_FLOOR_ENV).is_ok_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            });
            ENGINE_AWARE_MACS_FLOOR_GATE.store(u8::from(armed), Ordering::Relaxed);
            armed
        }
    }
}

/// Maximum MACs in one non-interruptible accelerator dispatch while a verifier
/// deadline is authoritative. The engine may impose a smaller cap, but it must
/// tile only the output axes and retain the complete `k` contraction.
const DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS: usize = 1 << 24;

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
    aw_f64_with_abssum_unbounded(a_block, w)
}

/// Domain-batched twin of [`aw_f64_with_abssum`] (#iso-batched-rebound): the
/// per-domain shortcut rebound's dominant leaf is this f64 `A·W` + `|A|·|W|`
/// pair, issued once per Linear layer PER DOMAIN as a small latency-bound GEMM.
/// This twin stacks `n_domains` per-domain coefficient blocks — which all share
/// the SAME layer weight `w` — into one tall `a_stacked` (`(n_domains*m) × k`,
/// domain `d`'s block at rows `d*m .. (d+1)*m`) and issues ONE faer f64 GEMM
/// pair over the stack, turning `n_domains` tiny GEMMs into one throughput-bound
/// GEMM.
///
/// PARITY (BIT-IDENTICAL by construction): faer's blocked f64 GEMM computes each
/// output element `(i,j)` by the SAME `k`-reduction sequence regardless of the
/// row count `M` (the batch axis only adds rows `i`, never touches the `k`
/// contraction or the `(i,j)` accumulation order). So block `d` of the stacked
/// product equals `aw_f64_with_abssum(&A_d, w)`'s faer arm output bit-for-bit —
/// the same operands, the same `current_par()` policy, the same `Accum::Replace`.
/// Empirically verified: 0 / 1.87M element mismatches across shapes/batches,
/// under both `Par::Seq` and `Par::rayon` (see
/// `aw_batched_matches_per_domain_bit_for_bit`). The certified error `γ_n·S`
/// (Higham, summation-order independent) is unchanged, so the stacked `s` is a
/// valid basis for the identical enclosure `a64 ± γ_n·S`.
///
/// Returns `(a64, s)` stacked the same way as the input rows. The rebound-scratch
/// pool is intentionally NOT used here (the whole point is one large product, not
/// per-domain recycled buffers); the pooled per-domain arm is itself proven
/// bit-identical to this owned faer arm (#rebound-scratch), so a caller that runs
/// the reference with rebound-scratch ON still matches this twin.
#[cfg_attr(not(test), allow(dead_code))] // primitive for the batched-forward SoA driver — the merged-GEMM leaf of the per-node CROWN-IBP forward that the iso shortcut rebound spends ~85% of its time in; SoA-loop wiring is the follow-on.
pub(crate) fn aw_batched_f64_with_abssum(
    a_stacked: &Mat<f32>,
    w: &Mat<f32>,
) -> (Array2<f64>, Array2<f64>) {
    let mk = a_stacked.nrows(); // n_domains * m (per-domain rows stacked)
    let k = a_stacked.ncols();
    let p = w.ncols();
    debug_assert_eq!(w.nrows(), k);

    // Faer owned arm, verbatim to `aw_f64_with_abssum`'s faer branch but over the
    // full stacked row set. `f32→f64` widening and `|·|` are EXACT, so both GEMMs
    // sum the SAME exact-in-f64 products; per-block bit-identity is the row-count
    // independence of faer's per-element reduction (verified). `mat_mul_f64` uses
    // `current_par()`, matching the reference's parallelism policy at every call
    // site (driver thread or, under a NestedFaerParGuard, a rayon worker).
    let a_f = Mat::<f64>::from_fn(mk, k, |i, j| f64::from(a_stacked[(i, j)]));
    let w_f = Mat::<f64>::from_fn(k, p, |i, j| f64::from(w[(i, j)]));
    let a_abs = Mat::<f64>::from_fn(mk, k, |i, j| f64::from(a_stacked[(i, j)]).abs());
    let w_abs = Mat::<f64>::from_fn(k, p, |i, j| f64::from(w[(i, j)]).abs());
    let c = mat_mul_f64(&a_f, &w_f);
    let s_mat = mat_mul_f64(&a_abs, &w_abs);
    let a64 = Array2::from_shape_fn((mk, p), |(i, j)| c[(i, j)]);
    let s = Array2::from_shape_fn((mk, p), |(i, j)| s_mat[(i, j)]);
    (a64, s)
}

/// Deadline-aware twin of [`aw_f64_with_abssum`].
///
/// A sufficiently large deadline-scoped call may use the process-global sound
/// f64 engine, but only through
/// [`GemmEngine::gemm_f64_with_deadline`]. Both `A·W` and `|A|·|W|` stay f64,
/// retain the full `k` contraction, and are independently validated before
/// either result is published. Unsupported, ordinary, malformed, or non-finite
/// engine outcomes fall back to the pollable chunked-faer CPU reduction
/// (scalar for over-quantum rows; see
/// [`aw_f64_with_abssum_cpu_deadline`]); an engine
/// [`NyError::DeadlineExceeded`] or any post-deadline completion is terminal.
/// The ordinary `gemm_f64`/`gemm_f32` methods are never called under finite
/// deadline authority. With `None`, preserve the existing acceleration policy
/// and arithmetic unchanged.
pub(crate) fn aw_f64_with_abssum_and_deadline(
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    deadline: Option<Instant>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let Some(deadline) = deadline else {
        return Ok(aw_f64_with_abssum_unbounded(a_block, w));
    };

    // A comprehensive host sweep installs this call-local authority on every
    // worker in its private Rayon pool. It must never consult (or lazily start)
    // the process-global CUDA/WGPU slot: use the already-audited pollable faer
    // route directly. Ordinary callers do not install the guard and retain the
    // historical global admission byte-for-byte.
    if crate::sound_f64_gemm::cpu_only_f64_active() {
        return aw_f64_with_abssum_cpu_deadline(a_block, w, deadline);
    }

    let m = a_block.nrows();
    let k = a_block.ncols();
    let p = w.ncols();
    if w.nrows() != k {
        return Err(NyError::ShapeMismatch {
            expected: vec![k, p],
            got: vec![w.nrows(), p],
        });
    }

    if deadline_f64_accelerator_eligible(m, k, p) {
        // Deadline-safe admission never waits on OnceLock/factory construction
        // on this verifier thread. Once admitted, the closure uses only the
        // engine's explicit bounded method; its ordinary methods are not a
        // fallback. If no engine is ready, retain the pollable CPU path below.
        //
        // The inner `deadline_f64_engine_admits` is UNCONDITIONALLY true when
        // the engine-aware gate is unset, so this branch is byte-identical to
        // its historical form. Armed, it additionally lets an engine decline a
        // large product its own measurements say it loses on (`k == 1` and
        // `m == 1` at 16.7 M MACs are 0.13× and 0.77× on faer).
        if let Some(result) = crate::sound_f64_gemm::with_engine_deadline(deadline, |engine| {
            if deadline_f64_engine_admits(engine, m, k, p) {
                aw_f64_with_abssum_deadline_via_engine_or_cpu(engine, a_block, w, deadline)
            } else {
                aw_f64_with_abssum_cpu_deadline(a_block, w, deadline)
            }
        })? {
            return result;
        }
    } else if engine_aware_macs_floor_armed() && engine_aware_admission_candidate(m, k, p) {
        // GATED, sub-threshold engine-aware admission (#b4-engine-aware-macs-floor).
        //
        // The constant above is the GPU launch-latency crossover; a CPU-resident
        // engine's is ~16,000× lower, and on THIS path the fall-through it must
        // beat is a single-threaded pollable scalar triple loop, not faer. So a
        // large band where the engine wins 3×–17× is gated out by a number that
        // was never about this engine. `deadline_f64_engine_admits` asks the
        // engine for its own measured crossover instead.
        //
        // FAIL CLOSED, and deliberately weaker than the branch above: this
        // consults ONLY an already-materialized engine. It never enters the
        // factory, never waits, and cannot mark initialization abandoned — so it
        // can neither stall this verifier thread nor disable the accelerator for
        // the products that DO cross the constant floor. Materialization is only
        // kicked off in the background; until it completes these calls take the
        // historical CPU path.
        crate::sound_f64_gemm::start_background_initialization();
        let admitted = crate::sound_f64_gemm::with_preinitialized_engine(|engine| {
            deadline_f64_engine_admits(engine, m, k, p).then(|| {
                aw_f64_with_abssum_deadline_via_engine_or_cpu(engine, a_block, w, deadline)
            })
        })
        .flatten();
        if let Some(result) = admitted {
            return result;
        }
    }

    aw_f64_with_abssum_cpu_deadline(a_block, w, deadline)
}

#[inline]
fn deadline_f64_accelerator_eligible(m: usize, k: usize, p: usize) -> bool {
    m.saturating_mul(k).saturating_mul(p) >= SOUND_F64_GEMM_MIN_MACS
}

/// Cheap, engine-FREE pre-check for the gated engine-aware path.
///
/// Runs before any engine is consulted so the gate-off cost is a single relaxed
/// atomic load and the gate-on cost for an obviously-tiny product is three
/// comparisons. It is a necessary condition only; the engine's own declaration
/// still decides.
#[inline]
fn engine_aware_admission_candidate(m: usize, k: usize, p: usize) -> bool {
    m.saturating_mul(k).saturating_mul(p) >= ENGINE_AWARE_ABSOLUTE_MIN_MACS
}

/// The full deadline-path admission predicate, engine included.
///
/// Gate OFF ⇒ identically [`deadline_f64_accelerator_eligible`]: the engine's
/// declaration is not even read, and every engine — including cuBLAS — behaves
/// exactly as before.
///
/// Gate ON ⇒ the engine's own declaration is authoritative in BOTH directions,
/// subject to the engine-independent hard floor:
///   * it may OPEN products below the historical constant (faer's measured
///     crossover is ~16,000× lower than the GPU-tuned one); and
///   * it may DECLINE products above it. That direction is not cosmetic: at
///     `4096x1x4096` — 16,777,216 MACs, i.e. admitted by the constant today —
///     the measured engine result is 0.13×, a 7.6× SLOWDOWN, and the same holds
///     for every `k == 1` and `m == 1` shape at that size.
///
/// An engine that does not override its declaration gets
/// [`ny_core::SoundF64GemmAdmission::CONSTANT_FLOOR`], which reproduces the
/// constant pointwise, so arming the gate changes nothing for it — in either
/// direction.
#[inline]
fn deadline_f64_engine_admits(engine: &dyn GemmEngine, m: usize, k: usize, p: usize) -> bool {
    if !engine_aware_macs_floor_armed() {
        return deadline_f64_accelerator_eligible(m, k, p);
    }
    engine_aware_admission_candidate(m, k, p)
        && engine
            .sound_f64_deadline_admission()
            .sanitized()
            .admits(m, k, p)
}

/// Historical unbounded implementation, including its optional global f64 GEMM.
fn aw_f64_with_abssum_unbounded(a_block: &Mat<f32>, w: &Mat<f32>) -> (Array2<f64>, Array2<f64>) {
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
                |i, j| f32_to_f64_exact(a_block[(i, j)]),
                |i, j| f32_to_f64_exact(w[(i, j)]),
            );
            // Column-major product → row-major ndarray (element (i,j) at
            // `j*m + i`), identical values to reading the owned faer `Mat`.
            let a64 = Array2::from_shape_fn((m, p), |(i, j)| aw_cm[j * m + i]);
            let s = Array2::from_shape_fn((m, p), |(i, j)| s_cm[j * m + i]);
            crate::rebound_scratch::recycle_f64(aw_cm);
            crate::rebound_scratch::recycle_f64(s_cm);
            return (a64, s);
        }
        let a_f = Mat::<f64>::from_fn(m, k, |i, j| f32_to_f64_exact(a_block[(i, j)]));
        let w_f = Mat::<f64>::from_fn(k, p, |i, j| f32_to_f64_exact(w[(i, j)]));
        // |A|, |W| are exact in f64 (abs of an exactly-widened f32 clears the
        // sign bit only), so the abs-sum GEMM sums the exact |a|·|w| products.
        let a_abs = Mat::<f64>::from_fn(m, k, |i, j| f32_to_f64_exact(a_block[(i, j)]).abs());
        let w_abs = Mat::<f64>::from_fn(k, p, |i, j| f32_to_f64_exact(w[(i, j)]).abs());
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
            let av = f32_to_f64_exact(a_block[(i, kk)]);
            if av == 0.0 {
                continue;
            }
            let av_abs = av.abs();
            for j in 0..p {
                let wv = f32_to_f64_exact(w[(kk, j)]);
                a64_row[j] += av * wv;
                s_row[j] += av_abs * wv.abs();
            }
        }
    }
    (a64, s)
}

/// Try the explicit deadline-bounded f64 engine and retain the existing
/// pollable CPU implementation as the only fallback.
///
/// Kept separate from process-global admission so unit tests can inject fake
/// engines and prove that ordinary GEMM methods remain unreachable.
fn aw_f64_with_abssum_deadline_via_engine_or_cpu(
    engine: &dyn GemmEngine,
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    deadline: Instant,
) -> Result<(Array2<f64>, Array2<f64>)> {
    if let Some(result) = aw_via_engine_deadline(engine, a_block, w, deadline)? {
        return Ok(result);
    }
    aw_f64_with_abssum_cpu_deadline(a_block, w, deadline)
}

/// Poll the Linear-CROWN f64 deadline with a phase-specific diagnostic.
#[inline]
fn check_aw_deadline(deadline: Instant, phase: &'static str) -> Result<()> {
    if Instant::now() >= deadline {
        Err(NyError::DeadlineExceeded(format!(
            "Linear CROWN backward: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

/// Run one full-`k` IEEE-f64 product through an engine's explicit bounded
/// contract and validate it before publication.
///
/// Deadline errors are terminal. Any other engine error, a malformed length, or
/// a non-finite coefficient declines to the caller's pollable CPU fallback,
/// provided the deadline is still live.
fn deadline_f64_gemm_try_engine(
    engine: &dyn GemmEngine,
    m: usize,
    k: usize,
    p: usize,
    a: &[f64],
    w: &[f64],
    deadline: Instant,
    require_nonnegative: bool,
) -> Result<Option<Vec<f64>>> {
    const VALIDATION_POLL_ELEMENTS: usize = 1 << 12;

    check_aw_deadline(deadline, "before bounded f64 GEMM")?;
    let expected_len = m.checked_mul(p).ok_or_else(|| {
        NyError::InvalidSpec("Linear CROWN bounded f64 GEMM output size overflow".into())
    })?;

    match engine.gemm_f64_with_deadline(
        m,
        k,
        p,
        a,
        w,
        deadline,
        DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS,
    ) {
        Ok(result) if result.len() == expected_len => {
            for chunk in result.chunks(VALIDATION_POLL_ELEMENTS) {
                if chunk.iter().any(|value| {
                    !value.is_finite() || (require_nonnegative && value.is_sign_negative())
                }) {
                    check_aw_deadline(deadline, "while rejecting invalid bounded f64 GEMM")?;
                    return Ok(None);
                }
                check_aw_deadline(deadline, "while validating bounded f64 GEMM")?;
            }
            check_aw_deadline(deadline, "after bounded f64 GEMM")?;
            Ok(Some(result))
        }
        Ok(_) => {
            check_aw_deadline(deadline, "while rejecting malformed bounded f64 GEMM")?;
            Ok(None)
        }
        Err(error) if error.is_deadline_exceeded() => Err(error),
        Err(_) => {
            check_aw_deadline(deadline, "after failed bounded f64 GEMM")?;
            Ok(None)
        }
    }
}

/// Compute `(A·W, |A|·|W|)` with two explicit deadline-bounded IEEE-f64
/// products.
///
/// The same exactly widened row-major operands are used for the first product
/// and then changed in place to their exact absolute values for the second.
/// Each engine call receives the original, complete `k` contraction. Thus the
/// caller's existing summation-order-independent `γ_k·S` certificate applies
/// unchanged even when the engine tiles the output `m`/`p` axes.
///
/// Returns `Ok(None)` for safe CPU-fallback outcomes. No partial result from
/// either product is observable.
fn aw_via_engine_deadline(
    engine: &dyn GemmEngine,
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    deadline: Instant,
) -> Result<Option<(Array2<f64>, Array2<f64>)>> {
    const CONVERSION_POLL_ELEMENTS: usize = 1 << 12;

    check_aw_deadline(deadline, "before bounded f64 A·W preparation")?;
    let m = a_block.nrows();
    let k = a_block.ncols();
    let p = w.ncols();
    if w.nrows() != k {
        return Err(NyError::ShapeMismatch {
            expected: vec![k, p],
            got: vec![w.nrows(), p],
        });
    }
    let a_len = m.checked_mul(k).ok_or_else(|| {
        NyError::InvalidSpec("Linear CROWN bounded f64 left operand size overflow".into())
    })?;
    let w_len = k.checked_mul(p).ok_or_else(|| {
        NyError::InvalidSpec("Linear CROWN bounded f64 right operand size overflow".into())
    })?;

    let mut a64 = Vec::with_capacity(a_len);
    for i in 0..m {
        for kk in 0..k {
            let value = a_block[(i, kk)];
            if !value.is_finite() {
                check_aw_deadline(deadline, "while rejecting non-finite f64 GEMM input")?;
                return Ok(None);
            }
            a64.push(f32_to_f64_exact(value));
            if a64.len().is_multiple_of(CONVERSION_POLL_ELEMENTS) {
                check_aw_deadline(deadline, "while preparing bounded f64 GEMM input")?;
            }
        }
    }

    let mut w64 = Vec::with_capacity(w_len);
    for kk in 0..k {
        for j in 0..p {
            let value = w[(kk, j)];
            if !value.is_finite() {
                check_aw_deadline(deadline, "while rejecting non-finite f64 GEMM input")?;
                return Ok(None);
            }
            w64.push(f32_to_f64_exact(value));
            if w64.len().is_multiple_of(CONVERSION_POLL_ELEMENTS) {
                check_aw_deadline(deadline, "while preparing bounded f64 GEMM input")?;
            }
        }
    }
    check_aw_deadline(deadline, "after bounded f64 GEMM input preparation")?;

    let Some(aw) = deadline_f64_gemm_try_engine(engine, m, k, p, &a64, &w64, deadline, false)?
    else {
        return Ok(None);
    };

    // f32→f64 widening and f64 abs are exact. Mutating only after the first
    // synchronous contract returns keeps peak host memory to two input images.
    for chunk in a64.chunks_mut(CONVERSION_POLL_ELEMENTS) {
        for value in chunk {
            *value = value.abs();
        }
        check_aw_deadline(deadline, "while preparing bounded f64 absolute input")?;
    }
    for chunk in w64.chunks_mut(CONVERSION_POLL_ELEMENTS) {
        for value in chunk {
            *value = value.abs();
        }
        check_aw_deadline(deadline, "while preparing bounded f64 absolute input")?;
    }

    let Some(abs_sum) = deadline_f64_gemm_try_engine(engine, m, k, p, &a64, &w64, deadline, true)?
    else {
        return Ok(None);
    };
    check_aw_deadline(deadline, "before publishing bounded f64 A·W")?;

    let aw = Array2::from_shape_vec((m, p), aw).map_err(|_| {
        NyError::InternalError("validated Linear CROWN A·W shape became malformed".into())
    })?;
    let abs_sum = Array2::from_shape_vec((m, p), abs_sum).map_err(|_| {
        NyError::InternalError("validated Linear CROWN |A|·|W| shape became malformed".into())
    })?;
    check_aw_deadline(deadline, "after publishing bounded f64 A·W")?;
    Ok(Some((aw, abs_sum)))
}

/// Quantum bounding one uninterrupted faer dispatch on the deadline-scoped
/// CPU `A·W` path: the same non-pollable-stretch budget the bounded engine
/// contract grants a single accelerator dispatch
/// ([`DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS`]). One chunk is
/// `rows_per_chunk·k·p ≤ 2·`this many MACs of GEMM work (`A·W` plus
/// `|A|·|W|`), a few milliseconds on faer — polls run between chunks and
/// between the two products of each chunk.
const DEADLINE_AW_FAER_CHUNK_MACS: usize = 1 << 24;
// Review defect 6: the doc above ASSERTS these are the same budget — bind it,
// so a future edit to either constant fails the build instead of the contract.
const _: () = assert!(
    DEADLINE_AW_FAER_CHUNK_MACS == DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS,
    "the deadline A·W chunk quantum must equal the bounded-dispatch MAC budget"
);

/// Byte ceiling on the f64 scratch one chunked deadline `A·W` may materialise
/// (review defect 5). The chunk predicate bounds MACs, which does NOT bound
/// MEMORY: at `k·p ≈ 2^24` the widened `W` + `|W|` pair alone is 268 MB of
/// f64, and this box's 121 GiB is SHARED between CPU and GPU — a documented
/// global-OOM vector. Above this ceiling the scalar loop (which allocates
/// nothing beyond its outputs) remains the pollable form.
const DEADLINE_AW_FAER_MAX_SCRATCH_BYTES: usize = 64 << 20;

/// faer 0.24 aligns owned plain-number matrix allocations to 64 bytes and
/// rounds the row capacity up to that alignment. `Mat<f64>` therefore owns a
/// multiple of eight rows even when its logical matrix has fewer.
const FAER_F64_ROW_CAPACITY_GRANULARITY: usize = 64 / size_of::<f64>();
const _: () = assert!(64 % size_of::<f64>() == 0);

fn faer_f64_padded_row_capacity(rows: usize) -> Option<usize> {
    let remainder = rows % FAER_F64_ROW_CAPACITY_GRANULARITY;
    if remainder == 0 {
        Some(rows)
    } else {
        rows.checked_add(FAER_F64_ROW_CAPACITY_GRANULARITY - remainder)
    }
}

/// Exact user-owned faer allocation footprint at the peak of one chunk.
///
/// `w_f` and `w_abs` are `k×p`; `a_f` and `a_abs` are `rows×k`; and the
/// two products are `rows×p`. All six matrices are live when the second
/// product is materialised, and every owned faer matrix uses its padded row
/// capacity rather than its logical row count.
fn deadline_aw_faer_owned_scratch_bytes(k: usize, p: usize, rows: usize) -> Option<usize> {
    let padded_k = faer_f64_padded_row_capacity(k)?;
    let padded_rows = faer_f64_padded_row_capacity(rows)?;

    let one_weight_elements = padded_k.checked_mul(p)?;
    let one_chunk_pair_elements = padded_rows.checked_mul(k.checked_add(p)?)?;
    one_weight_elements
        .checked_add(one_chunk_pair_elements)?
        .checked_mul(2)?
        .checked_mul(size_of::<f64>())
}

/// Admit a deadline-scoped faer chunk only when both widened weights and at
/// least one complete row of operands/results fit under the scratch ceiling.
///
/// Returning `None` is important when integer division yields zero rows: a
/// full contraction cannot be split across calls, so forcing that zero back
/// to one would exceed [`DEADLINE_AW_FAER_MAX_SCRATCH_BYTES`].
fn deadline_aw_faer_rows_per_chunk(k: usize, p: usize, chunk_macs: usize) -> Option<usize> {
    let row_macs = k.checked_mul(p)?;
    if row_macs == 0 || row_macs > chunk_macs {
        return None;
    }

    let rows_by_macs = chunk_macs / row_macs;
    if deadline_aw_faer_owned_scratch_bytes(k, p, 1)? > DEADLINE_AW_FAER_MAX_SCRATCH_BYTES {
        return None;
    }

    // The scratch predicate is monotone but advances in eight-row plateaus.
    // First accept the MAC-limited maximum when possible; otherwise binary
    // search between the known-good one-row chunk and that known-bad maximum.
    if deadline_aw_faer_owned_scratch_bytes(k, p, rows_by_macs)
        .is_some_and(|bytes| bytes <= DEADLINE_AW_FAER_MAX_SCRATCH_BYTES)
    {
        return Some(rows_by_macs);
    }

    let mut accepted = 1usize;
    let mut rejected = rows_by_macs;
    while rejected - accepted > 1 {
        let candidate = accepted.midpoint(rejected);
        if deadline_aw_faer_owned_scratch_bytes(k, p, candidate)
            .is_some_and(|bytes| bytes <= DEADLINE_AW_FAER_MAX_SCRATCH_BYTES)
        {
            accepted = candidate;
        } else {
            rejected = candidate;
        }
    }
    Some(accepted)
}

/// Pollable CPU-only `A·W` + `|A|·|W|` for deadline-scoped replay work.
///
/// # Rounding discipline + certificate (#cgan-row7-h4, b90a9fbf mirror)
///
/// 6f49a660 replaced the deadline arm's faer GEMMs with a scalar per-MAC
/// triple loop (the audit contract: no unpollable engine/kernel entry under a
/// deadline). The ROUNDING discipline of that loop was already the tight
/// accumulate-then-charge form — plain round-to-nearest f64 dual accumulators
/// (`a64`, `S`) with the CALLER charging `γ_k^{f64}·S` (`gamma_n_f64` over the
/// full contraction) — so no per-operation outward stepping ever existed here;
/// what 6f49a660 cost this lane was THROUGHPUT, not tightness (the sibling
/// ConvTranspose lanes measured a ~60× scalar tax, b90a9fbf). This restores
/// row-chunked faer `mat_mul_f64` GEMMs under the b90a9fbf argument:
///
/// * f32→f64 widening and f64 `abs` are EXACT, so both GEMMs sum the SAME
///   exact-in-f64 products as the scalar loop, only in faer's blocked order;
/// * the caller's certificate `γ_k·S` (Higham, Accuracy & Stability Thm 3.1)
///   is summation-order-INDEPENDENT — exactly the certificate the no-deadline
///   faer twin (#linearizenn-faer-f64-aw), the unbounded engine offload
///   (`aw_via_engine`), and the bounded engine seam (`aw_via_engine_deadline`)
///   already rely on;
/// * each chunk keeps every row's FULL `k` contraction inside one
///   `mat_mul_f64` call (chunking splits only the output row axis), so no
///   extra cross-call partial-sum roundings are introduced.
///
/// When the whole product fits one chunk (every sub-threshold Linear CROWN
/// shape in practice), the operands and the single `mat_mul_f64` pair are
/// IDENTICAL to the no-deadline faer twin's, so the deadline arm is
/// BIT-IDENTICAL to it — the same convergence property b90a9fbf pinned for
/// the ConvTranspose lanes. Multi-chunk results may differ bitwise from the
/// single-call twin (faer may block differently per shape); the enclosure is
/// certified either way by order-independence.
///
/// The audit contract stays closed: `mat_mul_f64` is the plain faer CPU
/// kernel (never the process-global engine; `current_par()` forces `Par::Seq`
/// inside rayon domain workers, #4392), a single dispatch is bounded by
/// [`DEADLINE_AW_FAER_CHUNK_MACS`], and the deadline is polled between chunks
/// and between the two GEMMs of a chunk. Shapes whose single-row dispatch
/// `k·p` exceeds the quantum — where chunking cannot bound the stretch —
/// fall back to the historical per-MAC scalar loop
/// ([`aw_f64_with_abssum_cpu_deadline_scalar`]), as does the
/// `NY_NAIVE_F64_AW` parity kill-switch.
fn aw_f64_with_abssum_cpu_deadline(
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    deadline: Instant,
) -> Result<(Array2<f64>, Array2<f64>)> {
    aw_f64_with_abssum_cpu_deadline_with_chunk_macs(
        a_block,
        w,
        deadline,
        DEADLINE_AW_FAER_CHUNK_MACS,
    )
}

/// Testable core of [`aw_f64_with_abssum_cpu_deadline`]. Production always
/// passes [`DEADLINE_AW_FAER_CHUNK_MACS`]; tests pass a tiny quantum to force
/// one row per chunk and pin the BETWEEN-chunk typed-deadline abort (the
/// `ops_gemm.rs` between-blocks pattern).
fn aw_f64_with_abssum_cpu_deadline_with_chunk_macs(
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    deadline: Instant,
    chunk_macs: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    check_aw_deadline(deadline, "during certified f64 A·W")?;
    let m = a_block.nrows();
    let k = a_block.ncols();
    let p = w.ncols();
    if w.nrows() != k {
        return Err(NyError::ShapeMismatch {
            expected: vec![k, p],
            got: vec![w.nrows(), p],
        });
    }

    // A single row's dispatch is `k·p` MACs and must stay whole (full-`k`
    // contraction per call); if even that exceeds the quantum, chunking cannot
    // bound the non-pollable stretch and the scalar per-MAC loop remains the
    // only pollable form. Degenerate shapes and the parity kill-switch take
    // the same historical path.
    let Some(rows_per_chunk) = deadline_aw_faer_rows_per_chunk(k, p, chunk_macs) else {
        return aw_f64_with_abssum_cpu_deadline_scalar(a_block, w, deadline);
    };
    if use_naive_f64_aw() {
        return aw_f64_with_abssum_cpu_deadline_scalar(a_block, w, deadline);
    }

    // Widen W (and |W|) once — exact conversions, O(k·p) ≤ one quantum of
    // work per matrix, with a poll after each.
    let w_f = Mat::<f64>::from_fn(k, p, |i, j| f32_to_f64_exact(w[(i, j)]));
    check_aw_deadline(deadline, "during certified f64 A·W")?;
    let w_abs = Mat::<f64>::from_fn(k, p, |i, j| f32_to_f64_exact(w[(i, j)]).abs());
    check_aw_deadline(deadline, "during certified f64 A·W")?;

    let mut a64 = Array2::<f64>::zeros((m, p));
    let mut s = Array2::<f64>::zeros((m, p));
    // Large zero-fills may themselves consume the remaining budget.
    check_aw_deadline(deadline, "during certified f64 A·W")?;

    let mut row_start = 0usize;
    while row_start < m {
        let row_end = row_start.saturating_add(rows_per_chunk).min(m);
        let rows = row_end - row_start;
        check_aw_deadline(deadline, "during certified f64 A·W")?;
        let a_f = Mat::<f64>::from_fn(rows, k, |i, j| {
            f32_to_f64_exact(a_block[(row_start + i, j)])
        });
        let a_abs = Mat::<f64>::from_fn(rows, k, |i, j| {
            f32_to_f64_exact(a_block[(row_start + i, j)]).abs()
        });
        let c = mat_mul_f64(&a_f, &w_f);
        check_aw_deadline(deadline, "during certified f64 A·W")?;
        let s_mat = mat_mul_f64(&a_abs, &w_abs);
        check_aw_deadline(deadline, "during certified f64 A·W")?;
        for i in 0..rows {
            for j in 0..p {
                a64[[row_start + i, j]] = c[(i, j)];
                s[[row_start + i, j]] = s_mat[(i, j)];
            }
        }
        row_start = row_end;
    }
    check_aw_deadline(deadline, "during certified f64 A·W")?;
    Ok((a64, s))
}

/// Historical scalar per-MAC pollable loop, byte-identical to the 6f49a660
/// form — retained as the over-quantum / degenerate-shape / kill-switch
/// fallback of [`aw_f64_with_abssum_cpu_deadline`] and as the bitwise parity
/// reference against the naive reduction.
///
/// The loop order is the historical `i → kk → j` order. Splitting the innermost
/// `j` traversal only inserts deadline checks and therefore leaves every
/// per-entry f64 reduction sequence unchanged. Each product is exact after
/// f32→f64 widening, and the caller's existing `γ_k·S` enclosure remains valid.
fn aw_f64_with_abssum_cpu_deadline_scalar(
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    deadline: Instant,
) -> Result<(Array2<f64>, Array2<f64>)> {
    const DEADLINE_POLL_MACS: usize = 1 << 12;
    const DEADLINE_POLL_SKIPPED_K: usize = 1 << 12;

    check_aw_deadline(deadline, "during certified f64 A·W")?;
    let m = a_block.nrows();
    let k = a_block.ncols();
    let p = w.ncols();
    if w.nrows() != k {
        return Err(NyError::ShapeMismatch {
            expected: vec![k, p],
            got: vec![w.nrows(), p],
        });
    }

    let mut a64 = Array2::<f64>::zeros((m, p));
    let mut s = Array2::<f64>::zeros((m, p));
    // Large zero-fills may themselves consume the remaining budget.
    check_aw_deadline(deadline, "during certified f64 A·W")?;

    let a64_buf = a64
        .as_slice_mut()
        .expect("a64 is freshly allocated row-major contiguous");
    let s_buf = s
        .as_slice_mut()
        .expect("s is freshly allocated row-major contiguous");
    let mut skipped_k_since_poll = 0usize;
    for i in 0..m {
        check_aw_deadline(deadline, "during certified f64 A·W")?;
        let a64_row = &mut a64_buf[i * p..i * p + p];
        let s_row = &mut s_buf[i * p..i * p + p];
        for kk in 0..k {
            let av = f32_to_f64_exact(a_block[(i, kk)]);
            if av == 0.0 || p == 0 {
                skipped_k_since_poll += 1;
                if skipped_k_since_poll >= DEADLINE_POLL_SKIPPED_K {
                    check_aw_deadline(deadline, "during certified f64 A·W")?;
                    skipped_k_since_poll = 0;
                }
                continue;
            }

            let av_abs = av.abs();
            let mut j_start = 0usize;
            while j_start < p {
                let j_end = (j_start + DEADLINE_POLL_MACS).min(p);
                for j in j_start..j_end {
                    let wv = f32_to_f64_exact(w[(kk, j)]);
                    a64_row[j] += av * wv;
                    s_row[j] += av_abs * wv.abs();
                }
                check_aw_deadline(deadline, "during certified f64 A·W")?;
                j_start = j_end;
            }
        }
    }
    check_aw_deadline(deadline, "during certified f64 A·W")?;
    Ok((a64, s))
}

/// Propagate one non-negative incoming coefficient-error matrix through
/// `|W|` without entering an opaque GEMM under finite deadline authority.
///
/// Every product of two f32 values is exact after widening to f64. The
/// deadline arm publishes the same historical per-add outward-stepped bound
/// as the unbounded arm; its only semantic addition is bounded polling and a
/// typed deadline abort. A previously authored dual-accumulator candidate was
/// withdrawn because the final f32 publication made its f64 tightening inert
/// (see [`incoming_error_product`]).
fn incoming_error_product_deadline(
    error: &Array2<f32>,
    column_offset: usize,
    contraction: usize,
    w_abs: &Mat<f32>,
    deadline: Instant,
) -> Result<Array2<f32>> {
    incoming_error_product(error, column_offset, contraction, w_abs, Some(deadline))
}

/// Deadline poll cadence for the incoming-error composition: at most this many
/// MACs run between two deadline observations.
const INCOMING_ERROR_POLL_MACS: usize = 1 << 12;

/// Test-only multiplicative outward slack for the withdrawn dual-accumulator
/// candidate's error term
/// `err = γ_{n+1}·(acc·abs_inflate)` — the exact mirror of the Conv2d IBP
/// kernel's `ERR_PRODUCT_SLACK_F64` (#cgan-conv-ibp-magnitude-floor,
/// `ops_ibp_fwd.rs`). Counting every round-to-nearest f64 operation that could
/// make the computed `err` under-shoot its exact real-arithmetic value: the 3
/// multiplications of the `err` expression, plus the subtract+divide inside
/// `γ = (n+1)·u/(1−(n+1)·u)` (`(n+1)·u` itself is exact: an integer < 2^53
/// scaled by a power of two), plus the subtract+divide inside
/// `abs_inflate = 1/(1−γ)` — ≤ 7 roundings.
///
/// CORRECTED 2026-08-12 (review defect 2): the naive "(1−u)^7 under-shoot, the
/// residue is O(u²)" reading is WRONG IN FORM. γ's own ≤2-rounding error is
/// AMPLIFIED when it passes through `abs_inflate = 1/(1−γ)`; the resulting
/// deficit is `≈ 2u·d/(1−2d)` with `d = (n+1)·2^-53` — not `O(u²)`, and
/// unbounded as `d → ½`. What actually carries the bound is the γ-INDEX
/// MARGIN: `γ_{n+1}` is charged where the fold performs at most `n−1`
/// roundings, a two-index spare of `≈ 4u/(1−2d)`, and `1 + 3d − 4d² > 0` for
/// every `d ∈ [0, ½)` — so the margin dominates the amplified deficit at every
/// reachable width. The `1 + 4·EPSILON` factor is belt-and-braces on top.
/// DO NOT "simplify" `contraction + 1` to `contraction` on the strength of the
/// old text: that removes the margin the proof rests on.
const INCOMING_ERROR_SLACK_F64: f64 = 1.0 + 4.0 * f64::EPSILON;

/// Test-only `(γ_{n+1}^{f64}, abs_inflate)` factors for the withdrawn
/// dual-accumulator candidate over a width-`n` incoming-error contraction.
/// `abs_inflate = 1/(1−γ)` covers the accumulator's own round-to-nearest
/// deficit (`acc ≥ T·(1−γ)` ⇒ `T ≤ acc·abs_inflate`); `+inf` when γ saturates
/// (`(n+1)·u64 ≥ 1`), which makes the candidate conservatively unusable.
#[inline]
pub(crate) fn incoming_error_dual_factors(contraction: usize) -> (f64, f64) {
    let gamma = gamma_n_f64(contraction.saturating_add(1));
    let abs_inflate = if gamma < 1.0 {
        1.0 / (1.0 - gamma)
    } else {
        f64::INFINITY
    };
    (gamma, abs_inflate)
}

/// Test-only dual-accumulator candidate upper bound on the exact non-negative
/// sum `T` whose round-to-nearest f64 fold produced `acc`:
///
/// ```text
/// acc + γ_{n+1}·(acc·abs_inflate)·INCOMING_ERROR_SLACK_F64  ≥  T
/// ```
///
/// Proof sketch (the fc5e569c derivation, specialized to same-sign terms):
/// every term is an exact-in-f64 product of two f32 values and non-negative,
/// so `Σ|t_k| = T` and Higham's summation bound (Accuracy and Stability,
/// Thm 3.1) gives `|acc − T| ≤ γ_n·T`, hence `T ≤ acc/(1−γ_n) =
/// acc·abs_inflate = acc + γ_n·(acc·abs_inflate)`. Charging index `n+1` and
/// the closed-form slack covers the ≤ 7 roundings of the factor computation
/// itself. Degenerate inputs stay conservative: `acc = +inf` ⇒ `+inf`;
/// `acc = 0` with a saturated `abs_inflate` yields NaN, which makes the
/// candidate unusable.
#[inline]
#[cfg(test)]
pub(crate) fn incoming_error_dual_upper(acc: f64, gamma: f64, abs_inflate: f64) -> f64 {
    // PRECONDITION (review defect 1): `acc` must be a round-to-nearest fold of
    // EXACT f32xf32 products, so it is either 0.0 or a NORMAL f64. On a
    // SUBNORMAL `acc` the `gamma * (acc * abs_inflate)` charge underflows and
    // the result collapses to `acc` itself — measured up to 1.2e-4 BELOW the
    // true sum at n = 2^40. Unreachable from `incoming_error_product`
    // (min |f32xf32| = 2^-298, a normal f64), but this helper is pub(crate),
    // so fail CLOSED rather than trust a future caller's term provenance.
    if acc != 0.0 && acc.abs() < f64::MIN_POSITIVE {
        return f64::INFINITY;
    }
    acc + gamma * (acc * abs_inflate) * INCOMING_ERROR_SLACK_F64
}

/// Compose an incoming certified error matrix through `|W|`:
/// `P[i,j] = Σ_k err[i, offset+k]·|W[k,j]|`, published as a certified f32
/// UPPER bound per entry (the pre-6f49a660 round-to-nearest f32 faer
/// `mat_mul` could round inward, which is why this is a self-certifying
/// scalar loop and not a GEMM).
///
/// # Rounding discipline (#cgan-row7-h4)
///
/// One pass over the terms maintains `stepped`, the historical per-add outward
/// fold (`next_up_nonnegative_f64` after every round-to-nearest addition).
/// This is an inductive upper bound on the exact sum `T`. Both finite-deadline
/// and unbounded calls publish that same accumulator, preserving byte parity;
/// a finite deadline only inserts the polling cadence below.
///
/// The test-only `incoming_error_dual_upper` helper documents and checks a
/// withdrawn alternative that charges one round-to-nearest accumulator at the
/// end. It is deliberately not wired into this production function: its f64
/// improvement is erased by `publish_error_up_normal` at practical widths.
///
/// The scalar loop is retained deliberately: the per-term sanitization
/// (`nonnegative_f32_error_or_infinity` poisoning, and the `err·0 → 0`
/// domination rule for a poisoned `+inf` error against a zero weight) has no
/// GEMM equivalent (`inf·0 = NaN` inside an opaque kernel), and the audit
/// contract from 6f49a660 (no unpollable engine entry under a deadline)
/// requires the [`INCOMING_ERROR_POLL_MACS`] cadence. The rounding discipline
/// itself remains the historical stepped form.
///
/// Casting the final stepped value to f32 could round inward;
/// `publish_error_up_normal` prevents that with a directed up-cast plus an
/// explicit `next_up_f32`.
/// Temporary probe accumulators for the incoming-error composition share
/// (read by the collection summary when `NY_DUMP_NODE_BOUNDS=1`).
pub(crate) static INCOMING_ERR_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static INCOMING_ERR_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// GEMM lane for the incoming-error composition (#err-compose-gemm).
///
/// Same contract, same publication, ~10 s -> ~0.05 s on the biasfield walks:
/// the scalar stepped triple loop below does GEMM-scale work at scalar speed
/// (measured 10.478 s of a 63.1 s collection on `cifar_bias_field_46`;
/// REGRESSION_FC_UNSAT_LOST_2026-08-14.md "walk-cost decomposition").
///
/// SOUNDNESS. Inputs are sanitized per element by the SAME
/// `nonnegative_f32_error_or_infinity` the scalar arm uses. The f64 tile
/// matmul's round-to-nearest sum `acc` is covered by the certified
/// dual-accumulator bound `acc·(1 + γ_{k+1}·abs_inflate·SLACK) ≥ T` — the
/// #cgan-row7-h4 candidate, validated against an exact-rational oracle and
/// retained above precisely for this reuse. Publication goes through the same
/// `publish_error_up_normal`. The withdrawn-candidate analysis proves the
/// stepped-vs-dual delta (≤ 4.5e-13 relative at k = 4096) is INERT through
/// that publication (2^-24-scale widening), so this lane publishes the SAME
/// f32 values as the scalar arm — pinned bit-exactly by
/// `incoming_error_gemm_publishes_identical_values`.
///
/// FALLBACKS (return `Ok(None)` => caller runs the scalar arm):
/// - any ±inf/NaN-poisoned input element: the scalar arm's `err·0 -> 0`
///   zero-dominates-infinity semantics cannot be reproduced by a plain GEMM
///   (0·inf = NaN there);
/// - a saturated dual factor (`abs_inflate` non-finite at extreme k).
///
/// DEADLINE. Checked at entry, between row tiles (the `gemm_f64_with_deadline`
/// cadence: tile cost ≤ INCOMING_ERROR_GEMM_TILE_MACS), and before
/// publication, with the same typed message as the scalar arm.
const INCOMING_ERROR_GEMM_TILE_MACS: usize = 1 << 24;

fn incoming_error_product_gemm(
    error: &Array2<f32>,
    column_offset: usize,
    contraction: usize,
    w_abs: &Mat<f32>,
    deadline: Option<Instant>,
) -> Result<Option<Array2<f32>>> {
    use faer::linalg::matmul::matmul;
    let check = || {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            Err(NyError::DeadlineExceeded(
                "Linear CROWN backward: deadline exceeded during incoming-error composition"
                    .to_string(),
            ))
        } else {
            Ok(())
        }
    };
    check()?;
    let rows = error.nrows();
    let out_cols = w_abs.ncols();
    let k = contraction;
    if rows == 0 || out_cols == 0 || k == 0 {
        return Ok(Some(Array2::<f32>::zeros((rows, out_cols))));
    }
    let (gamma, abs_inflate) = incoming_error_dual_factors(k);
    if !gamma.is_finite() || !abs_inflate.is_finite() {
        return Ok(None);
    }
    // Sanitize with the scalar arm's exact element map; refuse the lane on any
    // non-finite element (zero-dominates-infinity is the scalar arm's job).
    let mut a = vec![0.0f64; rows * k];
    let mut all_zero = true;
    for i in 0..rows {
        for kk in 0..k {
            let v = nonnegative_f32_error_or_infinity(error[[i, column_offset + kk]]);
            if !v.is_finite() {
                return Ok(None);
            }
            if v != 0.0 {
                all_zero = false;
            }
            a[i * k + kk] = v;
        }
    }
    if all_zero {
        // Exact zero product: publish the scalar arm's zero verbatim.
        return Ok(Some(Array2::<f32>::from_elem(
            (rows, out_cols),
            publish_error_up_normal(0.0),
        )));
    }
    check()?;
    let mut b = vec![0.0f64; k * out_cols];
    for kk in 0..k {
        for j in 0..out_cols {
            let v = nonnegative_f32_error_or_infinity(w_abs[(kk, j)]);
            if !v.is_finite() {
                return Ok(None);
            }
            b[kk * out_cols + j] = v;
        }
    }
    check()?;
    let b_mat = faer::MatRef::from_row_major_slice(&b, k, out_cols);
    let dual_scale = 1.0 + gamma * abs_inflate * INCOMING_ERROR_SLACK_F64;
    let mut product = Array2::<f32>::zeros((rows, out_cols));
    let rows_per = (INCOMING_ERROR_GEMM_TILE_MACS / k.saturating_mul(out_cols).max(1)).max(1);
    let mut i0 = 0usize;
    while i0 < rows {
        check()?;
        let tile_rows = rows_per.min(rows - i0);
        let a_blk =
            faer::MatRef::from_row_major_slice(&a[i0 * k..(i0 + tile_rows) * k], tile_rows, k);
        let mut dst = Mat::<f64>::zeros(tile_rows, out_cols);
        matmul(
            &mut dst,
            faer::Accum::Replace,
            a_blk,
            b_mat,
            1.0,
            crate::faer_parallelism::current_par(),
        );
        for i in 0..tile_rows {
            for j in 0..out_cols {
                // Non-negative inputs: a negative RN sum is impossible; clamp
                // defensively so the dual bound never shrinks below zero.
                let acc = dst[(i, j)].max(0.0);
                product[[i0 + i, j]] = publish_error_up_normal(acc * dual_scale);
            }
        }
        i0 += tile_rows;
    }
    check()?;
    Ok(Some(product))
}

pub(crate) fn incoming_error_product(
    error: &Array2<f32>,
    column_offset: usize,
    contraction: usize,
    w_abs: &Mat<f32>,
    deadline: Option<Instant>,
) -> Result<Array2<f32>> {
    let probe_start = Instant::now();
    // #err-compose-gemm: production takes the GEMM lane; `Ok(None)` (a poisoned
    // element or saturated dual factor) falls back to the scalar stepped arm,
    // and tests that pass an explicit poll quantum pin the scalar arm directly.
    let result =
        match incoming_error_product_gemm(error, column_offset, contraction, w_abs, deadline) {
            Ok(Some(product)) => Ok(product),
            Ok(None) => incoming_error_product_with_poll_quantum(
                error,
                column_offset,
                contraction,
                w_abs,
                deadline,
                INCOMING_ERROR_POLL_MACS,
            ),
            Err(error) => Err(error),
        };
    INCOMING_ERR_NANOS.fetch_add(
        u64::try_from(probe_start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
    INCOMING_ERR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    result
}

/// Testable core of [`incoming_error_product`]. Production always passes
/// [`INCOMING_ERROR_POLL_MACS`]; tests pass `poll_quantum = 1` to pin the
/// mid-loop typed-deadline abort without waiting on a full-size quantum
/// (the `ops_gemm.rs` between-blocks pattern).
fn incoming_error_product_with_poll_quantum(
    error: &Array2<f32>,
    column_offset: usize,
    contraction: usize,
    w_abs: &Mat<f32>,
    deadline: Option<Instant>,
    poll_quantum: usize,
) -> Result<Array2<f32>> {
    let poll_quantum = poll_quantum.max(1);

    let check = || {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            Err(NyError::DeadlineExceeded(
                "Linear CROWN backward: deadline exceeded during incoming-error composition"
                    .to_string(),
            ))
        } else {
            Ok(())
        }
    };

    check()?;
    let rows = error.nrows();
    let output_columns = w_abs.ncols();
    if error.ncols() < column_offset.saturating_add(contraction) || w_abs.nrows() != contraction {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, column_offset.saturating_add(contraction)],
            got: vec![error.nrows(), error.ncols()],
        });
    }
    // #cgan-row7-h4, WITHDRAWN 2026-08-12 after adversarial review: a
    // `min(stepped, dual)` form was authored here to remove the per-add
    // next_up inflation (~n·u64 relative). It was CERTIFIED (validated
    // against an exact-rational oracle) but provably INERT: this value is
    // published through `publish_error_up_normal`, whose directed cast plus
    // unconditional `next_up_f32` widen by 2^-24..2^-23 — five orders ABOVE
    // the gap the dual arm closes (4.5e-13 relative at n=4096; the crossover
    // needs n > 2^29), and it cannot compound across layers because every
    // layer re-publishes through the same f32 step. Certified-arithmetic
    // surface that cannot move a verdict is a bad trade under the moat rule,
    // so the historical stepped fold stands. The tightening lever that WOULD
    // pay lives at the f32 publication step / the `l_cast_err + γ·S + l_prop`
    // aggregation, not here. (`incoming_error_dual_*` are retained as
    // test-only, reviewed oracle helpers that such a lever would reuse.)
    let mut product = Array2::<f32>::zeros((rows, output_columns));
    check()?;
    let mut operations = 0usize;
    for i in 0..rows {
        for j in 0..output_columns {
            let mut stepped = 0.0f64;
            for kk in 0..contraction {
                if operations.is_multiple_of(poll_quantum) {
                    check()?;
                }
                operations = operations.wrapping_add(1);
                let error_value = nonnegative_f32_error_or_infinity(error[[i, column_offset + kk]]);
                let weight_abs = nonnegative_f32_error_or_infinity(w_abs[(kk, j)]);
                let term = if error_value == 0.0 || weight_abs == 0.0 {
                    0.0
                } else {
                    error_value * weight_abs
                };
                // Certified coefficient errors and |W| are non-negative. A
                // negative or NaN term violates that invariant; +inf is the
                // conservative error enclosure. The stepped arm takes one
                // successor after every RN addition, which dominates its exact
                // real sum without any contraction-width assumption.
                if term == 0.0 {
                    continue;
                }
                if term > 0.0 {
                    stepped = next_up_nonnegative_f64(stepped + term);
                } else {
                    stepped = f64::INFINITY;
                }
            }
            product[[i, j]] = publish_error_up_normal(stepped);
        }
    }
    check()?;
    Ok(product)
}

#[inline]
fn next_up_nonnegative_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > 0x7ff0_0000_0000_0000 || bits >> 63 != 0 {
        return f64::INFINITY;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if magnitude == 0x7ff0_0000_0000_0000 {
        return value;
    }
    f64::from_bits(bits + 1)
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
/// NUMERICAL REPAIR (2026-08-08, #f32-abssum-inflation-conditioning). The old
/// body evaluated `1.0 / (1.0 - g)` with `g = γ_k^f32`. As `k → 2^23`, `g → 1`
/// and `1 - g` is CATASTROPHICALLY CANCELLING, so the quotient could land BELOW
/// the exact `1/(1 - γ_k)` — i.e. the checked-in claim "`F_hat` over-bounds the
/// tight factor" was FALSE at large `k`. Measured at `k = 8_388_582`: the old
/// form gave `161319.8846141918` against an exact `161319.8846153846`, an
/// under-round of `7.39e-12` relative.
///
/// (That defect never produced an unsound `S`: the universal factor
/// `(1 - 2^-24)^-k` is under `1.649` in that regime, so the returned value still
/// dominated the real requirement by a wide margin. The PROOF was wrong, not the
/// bound — and a proof this seam depends on must not be wrong.)
///
/// The repair evaluates the algebraically identical but well-conditioned form.
/// With `u = 2^-24` and `d = k·u`, `γ_k = d/(1-d)`, so
/// `1 - γ_k = (1 - 2d)/(1 - d)` and therefore
/// `1/(1 - γ_k) = (1 - d)/(1 - 2d)` — no subtraction of nearly-equal quantities.
/// The result is then pushed OUTWARD one ulp before the `(1 + 2^-40)` margin, so
/// the return value dominates the exact factor by construction rather than by
/// the accident of a rounding direction.
///
/// Credit: the conditioning defect was found by an independent Codex CLI review.
#[inline]
fn f32_abssum_inflation(k: usize) -> Option<f64> {
    let g = gamma_n_f32(k);
    // NaN-aware "not (g < 1)": TRUE for NaN — `g >= 1.0` would let a NaN gamma
    // through to the inflation factor, so the negated form is load-bearing.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(g < 1.0) {
        return None;
    }
    let d = (k as f64) * 2f64.powi(-24);
    // `g < 1.0` already implies `d < 0.5`, hence `1 - 2d > 0`; assert the
    // load-bearing consequence rather than trusting the implication silently.
    let denom = 1.0 - 2.0 * d;
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }
    let tight = (1.0 - d) / denom;
    if !tight.is_finite() || tight < 1.0 {
        return None;
    }
    Some(next_up_nonnegative_f64(tight) * (1.0 + 2f64.powi(-40)))
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
            a64[i * k + kk] = f32_to_f64_exact(x);
            absa[i * k + kk] = x.abs();
        }
    }
    let mut w64 = vec![0.0f64; k * p];
    let mut absw = vec![0.0f32; k * p];
    for kk in 0..k {
        for j in 0..p {
            let y = w[(kk, j)];
            w64[kk * p + j] = f32_to_f64_exact(y);
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
            let daz = daz_operand_flush_floor(a_block, w, m, k, p);
            Array2::from_shape_fn((m, p), |(i, j)| {
                let r = f32_to_f64_exact(s32[i * p + j]);
                (r + g + daz[[i, j]]) * f_hat
            })
        }
        None => {
            // k ≥ 2^23: the f32 factor degenerates; use the f64 S path.
            //
            // HONEST SCOPE NOTE (2026-08-08, from an independent Codex review).
            // This branch is a RAW round-to-nearest f64 GEMM, and RN summation is
            // NOT guaranteed to be an elementwise UPPER bound on `Σ|a||w|` — it
            // can round down. So the blanket claim "S is always an over-bound",
            // which appears in several places in this tree and in the commit
            // history, is FALSE for this branch. The `γ_n^f64·S` term the caller
            // adds absorbs it in practice (the shortfall is ≤ γ_k^f64·S, which is
            // exactly what that term charges), so this is a PROOF-STATEMENT
            // defect rather than a demonstrated unsoundness — and it is
            // PRE-EXISTING: the historical second-f64-GEMM fall-through at
            // `aw_f64_with_abssum_unbounded` has the identical property. It is
            // recorded here rather than silently inherited. The f32 branch above
            // does not share it: its inflation + FTZ + DAZ machinery makes the
            // over-bound explicit and provable.
            let absa64: Vec<f64> = absa.iter().map(|&v| f32_to_f64_exact(v)).collect();
            let absw64: Vec<f64> = absw.iter().map(|&v| f32_to_f64_exact(v)).collect();
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
        .map(|i| (0..k).map(|l| f32_to_f64_exact(a[(i, l)]).abs()).sum())
        .collect();
    let col: Vec<f64> = (0..p)
        .map(|j| (0..k).map(|l| f32_to_f64_exact(w[(l, j)]).abs()).sum())
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
                (f32_to_f64_exact(r[[i, j]]) + g + daz[[i, j]]) * f_hat * contain_margin
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
/// # S2 — the EFT-compensated arm (`NY_EFT_ERR=1`, DARK by default)
///
/// The `γ_n^f32·S` charge above is A-PRIORI: it is the worst case over all sign
/// patterns, and in the CROWN regime (mixed-sign coefficients, `Σ|a·w|` ≫ the
/// running sum) it over-states the ACTUAL rounding error of the executed fold by
/// orders of magnitude. Under the gate this function additionally computes an
/// A-POSTERIORI enclosure whose radius is the exactly-measured EFT residual of
/// its own fold, and INTERSECTS the two — `max` on the lower endpoints, `min` on
/// the upper, which is the design doc's `max(lb_higham, lb_eft)` verbatim.
/// See [`aw_eft_sound_bound`] for why intersection (rather than replacement) is
/// what makes the downgrade-only property structural.
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
            let c = f32_to_f64_exact(c_hat[[i, j]]);
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

    // S2: intersect with the a-posteriori enclosure. Dark by default; when the
    // arm refuses (gate off, EFT preconditions unmet, size overflow, non-finite
    // fold) NOTHING below runs and the incumbent stands byte-identically.
    if eft_err_channel_enabled() {
        if let Some((eft_lower, eft_upper)) = aw_eft_sound_bound(a_block, w, m, k, p) {
            intersect_enclosure_downgrade_only(&mut lower, &mut upper, &eft_lower, &eft_upper);
        }
    }
    (lower, upper)
}

/// #eft-err process gate (`NY_EFT_ERR=1`), DARK by default.
///
/// Deliberately UNCACHED and named/valued identically to the GPU twin's gate
/// (`ny-gpu`'s `crown_backward_sound_resident::eft_err_env_enabled`) so one
/// switch arms the whole S2 channel and scoped-env tests can flip it. Read once
/// per GEMM, never per element.
#[allow(dead_code)] // reached via `aw_f32_sound_bound`, itself unwired by design.
fn eft_err_channel_enabled() -> bool {
    std::env::var("NY_EFT_ERR").ok().as_deref() == Some("1")
}

/// A-POSTERIORI certified enclosure of `A·W`: the plain left-to-right f32 fold's
/// value, widened by the EFT-MEASURED rounding error of *that* fold rather than
/// by the a-priori `γ_n^f32·S` worst case.
///
/// # Soundness
///
/// The channel owns BOTH halves of every entry. `eft_dot_f32_downgrade_only`
/// returns the value of the fold it just executed together with
/// `min(γ_{k+1}·Σ|a·w|, Σ|e_prod| + Σ|e_sum|)` for that same fold — the residual
/// sum is an EXACT measurement (TwoProdFMA + Knuth TwoSum telescope to an
/// identity), and the `min` is `ny_core::eft`'s single downgrade-only
/// chokepoint. Computing the radius for a value produced by a DIFFERENT
/// reduction order would be unsound, which is precisely why this function does
/// not certify `f32_gemm_rn`'s output: the accelerator's summation order is
/// unknown, and only an order-independent bound like `γ_n·S` may be applied to
/// it. Here the order is ours.
///
/// The `ftz` and `daz` floors are carried over UNCHANGED from the a-priori arm.
/// They bound subnormal result-flush and operand-flush, which the EFT identity
/// (a theorem about gradual underflow) does not cover; `ny_core::eft`'s
/// self-check verifies this target does not flush, so on it they are pure
/// widening.
///
/// Returns `None` — leaving the caller's a-priori enclosure untouched — when the
/// EFT preconditions do not hold on this target, an index overflows, or any fold
/// goes non-finite. `None` is never "zero error".
#[allow(dead_code)] // reached via `aw_f32_sound_bound`, itself unwired by design.
fn aw_eft_sound_bound(
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    m: usize,
    k: usize,
    p: usize,
) -> Option<(Array2<f32>, Array2<f32>)> {
    // Fail-closed on the target's EFT preconditions (fused FMA, RN, no FTZ).
    if !ny_core::eft::eft_available() {
        return None;
    }

    // A row-major, W column-major, so each dot reads two contiguous slices and
    // `[(i, j)]` layout-agnostic indexing is paid once instead of per fold.
    let mut a_rows = vec![0.0f32; m.checked_mul(k)?];
    for i in 0..m {
        for kk in 0..k {
            a_rows[i * k + kk] = a_block[(i, kk)];
        }
    }
    let mut w_cols = vec![0.0f32; k.checked_mul(p)?];
    for j in 0..p {
        for kk in 0..k {
            w_cols[j * k + kk] = w[(kk, j)];
        }
    }

    // Same floors, same expression, as the a-priori arm above.
    let ftz = 4.0 * (k as f64) * 2f64.powi(-126);
    let daz = daz_operand_flush_floor(a_block, w, m, k, p);

    let mut lower = Array2::<f32>::zeros((m, p));
    let mut upper = Array2::<f32>::zeros((m, p));
    for i in 0..m {
        let a_row = &a_rows[i * k..i * k + k];
        for j in 0..p {
            let w_col = &w_cols[j * k..j * k + k];
            let certified = ny_core::eft::eft_dot_f32_downgrade_only(a_row, w_col)?;
            let c = f32_to_f64_exact(certified.value);
            let penalty = f32_to_f64_exact(certified.err) + ftz + daz[[i, j]];
            if !c.is_finite() || !penalty.is_finite() {
                lower[[i, j]] = f32::NEG_INFINITY;
                upper[[i, j]] = f32::INFINITY;
            } else {
                lower[[i, j]] = next_down_f32((c - penalty) as f32);
                upper[[i, j]] = next_up_f32((c + penalty) as f32);
            }
        }
    }
    Some((lower, upper))
}

/// Intersect a candidate certified enclosure into an incumbent one, IN PLACE.
///
/// This is the structural form of `max(lb_higham, lb_eft)`: both arguments are
/// sound enclosures of the SAME exact `A·W` entry, so their intersection is
/// also one, and it is by construction never WIDER than the incumbent. An entry
/// only moves when it strictly improves; a NaN candidate endpoint fails every
/// comparison and is discarded; and a candidate that would INVERT the interval
/// (which can only happen if one of the two arms is unsound) is rejected
/// wholesale for that entry, leaving the a-priori channel in charge.
///
/// The function cannot WIDEN a bound: it never writes a lower endpoint smaller
/// than the incumbent's, nor an upper endpoint larger. Note carefully what that
/// does and does not buy.
///
/// It buys termination and monotonicity — the interval only ever shrinks, so no
/// sequence of applications can drift outward.
///
/// It does NOT make the result independent of the candidate. This is an
/// INTERSECTION, and intersecting narrows; a narrowed interval is a valid
/// enclosure only if the candidate was itself a valid enclosure of the same
/// quantity. A candidate that wrongly excludes part of the true range yields a
/// bound that is too tight — a FALSE PROOF, the one direction that matters. The
/// `l <= u` guard below rejects only a candidate that inverts the interval
/// outright, i.e. one that is grossly and detectably disjoint; it cannot detect
/// a candidate that is merely slightly wrong on one side.
///
/// So the caller's obligation is exactly the one this comment used to disclaim:
/// the candidate must be a proven enclosure before it is passed here. That is
/// why this function is unwired by design — see the `dead_code` note below.
#[allow(dead_code)] // reached via `aw_f32_sound_bound`, itself unwired by design.
fn intersect_enclosure_downgrade_only(
    lower: &mut Array2<f32>,
    upper: &mut Array2<f32>,
    candidate_lower: &Array2<f32>,
    candidate_upper: &Array2<f32>,
) {
    if candidate_lower.dim() != lower.dim() || candidate_upper.dim() != upper.dim() {
        return;
    }
    for i in 0..lower.nrows() {
        for j in 0..lower.ncols() {
            let mut l = lower[[i, j]];
            let mut u = upper[[i, j]];
            if candidate_lower[[i, j]] > l {
                l = candidate_lower[[i, j]];
            }
            if candidate_upper[[i, j]] < u {
                u = candidate_upper[[i, j]];
            }
            // Disjoint arms mean one of them is unsound; keep the proven one.
            if l <= u {
                lower[[i, j]] = l;
                upper[[i, j]] = u;
            }
        }
    }
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
    propagate_linear_cpu_with_deadline(layer, bounds, None, true)
}

fn propagate_linear_cpu_with_deadline<'a>(
    layer: &LinearLayer,
    bounds: &'a LinearBounds,
    deadline: Option<Instant>,
    allow_global_f64_engine: bool,
) -> Result<Cow<'a, LinearBounds>> {
    debug!("Linear layer CROWN backward propagation");

    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded(
            "Linear CROWN backward: deadline exceeded before CPU propagation".to_string(),
        ));
    }

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
        let legacy_f32 = allow_global_f64_engine
            && cfg!(debug_assertions)
            && std::env::var("NY_AW_LEGACY_F32").is_ok();
        let (stacked_a64, stacked_s) = if legacy_f32 {
            let lower_block = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                bounds.lower_a()[[i, in_start + j]]
            });
            let upper_block = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                bounds.upper_a()[[i, in_start + j]]
            });
            let ml = mat_mul(&lower_block, layer.weight_faer());
            let mu = mat_mul(&upper_block, layer.weight_faer());
            let a64 = Array2::from_shape_fn((2 * num_outputs, in_features), |(i, j)| {
                if i < num_outputs {
                    f32_to_f64_exact(ml[(i, j)])
                } else {
                    f32_to_f64_exact(mu[(i - num_outputs, j)])
                }
            });
            let s = Array2::<f64>::zeros((2 * num_outputs, in_features));
            (a64, s)
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
            if allow_global_f64_engine {
                aw_f64_with_abssum_and_deadline(&stacked, layer.weight_faer(), deadline)?
            } else {
                let deadline = deadline.ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "bounded Linear CROWN requires a finite host deadline".into(),
                    )
                })?;
                aw_f64_with_abssum_cpu_deadline(&stacked, layer.weight_faer(), deadline)?
            }
        };

        // Propagated incoming error: P[i,j] = Σ_k err_in[i,in_start+k]·|W[k,j]|.
        let prop_lower = match (in_lower_err, deadline) {
            (Some(error), Some(limit)) => Some(incoming_error_product_deadline(
                error,
                in_start,
                layout.out_features,
                &w_abs,
                limit,
            )?),
            (Some(error), None) => Some(incoming_error_product(
                error,
                in_start,
                layout.out_features,
                &w_abs,
                None,
            )?),
            (None, _) => None,
        };
        let prop_upper = match (in_upper_err, deadline) {
            (Some(error), Some(limit)) => Some(incoming_error_product_deadline(
                error,
                in_start,
                layout.out_features,
                &w_abs,
                limit,
            )?),
            (Some(error), None) => Some(incoming_error_product(
                error,
                in_start,
                layout.out_features,
                &w_abs,
                None,
            )?),
            (None, _) => None,
        };

        // Place result in output, tracking non-finite or near-overflow coefficients
        // per row (#2681, #1932). The magnitude check catches coefficients approaching
        // f32 overflow before they actually reach Inf, preventing NaN from subsequent
        // multiplications. See CROWN_COEFF_MAX documentation.
        for i in 0..num_outputs {
            let upper_i = num_outputs + i;
            for j in 0..in_features {
                let l = stacked_a64[[i, j]] as f32;
                let u = stacked_a64[[upper_i, j]] as f32;
                // Certified error: cast rounding |a64 - stored| + γ_n·S + propagated
                // incoming error, rounded UP to a sound f32.
                let l_cast_err = (stacked_a64[[i, j]] - f32_to_f64_exact(l)).abs();
                let u_cast_err = (stacked_a64[[upper_i, j]] - f32_to_f64_exact(u)).abs();
                let l_prop = prop_lower
                    .as_ref()
                    .map_or(0.0, |p| f32_to_f64_exact(p[[i, j]]));
                let u_prop = prop_upper
                    .as_ref()
                    .map_or(0.0, |p| f32_to_f64_exact(p[[i, j]]));
                let l_err =
                    publish_error_up_normal(l_cast_err + gamma * stacked_s[[i, j]] + l_prop);
                let u_err =
                    publish_error_up_normal(u_cast_err + gamma * stacked_s[[upper_i, j]] + u_prop);
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
                    e = add_coeff_err_bias_product_up(e, le[[i, j]], bias[j]);
                }
                lower_bias_contrib[i] = add_f64_down(lower_bias_contrib[i], -e);
            }
        }
        if let Some(ue) = in_upper_err {
            for i in 0..num_outputs {
                let mut e = 0.0f64;
                for j in 0..bias.len() {
                    e = add_coeff_err_bias_product_up(e, ue[[i, j]], bias[j]);
                }
                upper_bias_contrib[i] = add_f64_up(upper_bias_contrib[i], e);
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
    propagate_linear_with_engine_and_f64_deadline(layer, bounds, engine, None)
}

/// Shared engine/CPU dispatch carrying the certified-f64 offload policy.
///
/// This helper is retained for the unbounded public entry and tests of the
/// engine fallback. Finite-deadline authority bypasses it entirely because the
/// generic `GemmEngine` API has no cancellation capability.
fn propagate_linear_with_engine_and_f64_deadline<'a>(
    layer: &LinearLayer,
    bounds: &'a LinearBounds,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Cow<'a, LinearBounds>> {
    let Some(engine) = engine else {
        return propagate_linear_cpu_with_deadline(layer, bounds, deadline, true);
    };
    if engine.forbids_unbounded_cpu_fallback() {
        return Err(NyError::UnsupportedOp(
            "bounded Linear CROWN requires the explicit deadline-aware entry".into(),
        ));
    }

    match propagate_linear_via_gemm(layer, bounds, engine, deadline) {
        Ok(lb) => Ok(Cow::Owned(lb)),
        Err(e) if e.is_deadline_exceeded() => Err(e),
        Err(e) => {
            debug!("GEMM engine failed for Linear CROWN backward, falling back to CPU: {e}");
            propagate_linear_cpu_with_deadline(layer, bounds, deadline, true)
        }
    }
}

/// Deadline-aware CROWN backward propagation through a linear layer (#4321).
///
/// The dense `A @ W` GEMM is the single largest uninterrupted op on the
/// spec-matrix root output-bound path: a wide classifier-head GEMM with many
/// objective rows (e.g. TinyImageNet ResNet, 199 specs) can run for tens of
/// seconds with no internal deadline checkpoint, overrunning the verifier's own
/// `--timeout` and getting killed externally with no JSON verdict.
///
/// A finite deadline never enters the caller's generic f32 engine or the
/// process-global engine's ordinary `gemm_f64` method: neither API advertises
/// bounded dispatch. The certified f64 coefficient path remains CPU-pollable,
/// but sufficiently large `A @ W` products may use the process-global engine's
/// explicit [`GemmEngine::gemm_f64_with_deadline`] contract, which retains the
/// full contraction and bounds each output-tile dispatch. Incoming coefficient
/// errors are still composed on the separately pollable, outward-rounded CPU
/// path. With no deadline this is exactly [`propagate_linear_with_engine`].
pub(crate) fn propagate_linear_with_engine_and_deadline<'a>(
    layer: &LinearLayer,
    bounds: &'a LinearBounds,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Cow<'a, LinearBounds>> {
    let bounded_marker = engine.is_some_and(|engine| engine.forbids_unbounded_cpu_fallback());
    let bounded_host_engine = bounded_marker
        && engine.is_some_and(|engine| engine.provides_deadline_pollable_host_gemm());
    if bounded_marker && !bounded_host_engine {
        return Err(NyError::UnsupportedOp(
            "bounded Linear CROWN requires a pollable capped host engine".into(),
        ));
    }
    let Some(d) = deadline else {
        if bounded_host_engine {
            return Err(NyError::UnsupportedOp(
                "bounded Linear CROWN requires an explicit finite deadline".into(),
            ));
        }
        return propagate_linear_with_engine(layer, bounds, engine);
    };
    if Instant::now() >= d {
        return Err(NyError::DeadlineExceeded(
            "Linear CROWN backward: deadline exceeded before pollable CPU propagation".to_string(),
        ));
    }
    let result = propagate_linear_cpu_with_deadline(layer, bounds, Some(d), !bounded_host_engine)?;
    if Instant::now() >= d {
        return Err(NyError::DeadlineExceeded(
            "Linear CROWN backward: deadline exceeded before returning pollable CPU result"
                .to_string(),
        ));
    }
    Ok(result)
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
    deadline: Option<Instant>,
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
        let (lower_reference, lower_s) =
            aw_f64_with_abssum_and_deadline(&lower_faer, weight_faer, deadline)?;
        let (upper_reference, upper_s) =
            aw_f64_with_abssum_and_deadline(&upper_faer, weight_faer, deadline)?;
        let prop_lower = match in_lower_err {
            Some(error) => Some(incoming_error_product(
                error,
                in_start,
                layout.out_features,
                &w_abs,
                deadline,
            )?),
            None => None,
        };
        let prop_upper = match in_upper_err {
            Some(error) => Some(incoming_error_product(
                error,
                in_start,
                layout.out_features,
                &w_abs,
                deadline,
            )?),
            None => None,
        };

        for i in 0..num_outputs {
            let src_off = i * in_features;
            for j in 0..in_features {
                let l = new_lower_block[src_off + j];
                let u = new_upper_block[src_off + j];
                let l_prop = prop_lower
                    .as_ref()
                    .map_or(0.0, |p| f32_to_f64_exact(p[(i, j)]));
                let u_prop = prop_upper
                    .as_ref()
                    .map_or(0.0, |p| f32_to_f64_exact(p[(i, j)]));
                // Measure the opaque engine's result against the bit-exact f64
                // reference. This closes DAZ as well as ordinary accumulation
                // error; a relative γ·S term alone cannot cover an input
                // subnormal that the engine flushes before a large multiply.
                let l_gap = (f32_to_f64_exact(l) - lower_reference[[i, j]]).abs();
                let u_gap = (f32_to_f64_exact(u) - upper_reference[[i, j]]).abs();
                let l_err = publish_error_up_normal(l_gap + gamma * lower_s[[i, j]] + l_prop);
                let u_err = publish_error_up_normal(u_gap + gamma * upper_s[[i, j]] + u_prop);
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
                    e = add_coeff_err_bias_product_up(e, le[[i, j]], bias[j]);
                }
                lower_bias_contrib[i] = add_f64_down(lower_bias_contrib[i], -e);
            }
        }
        if let Some(ue) = in_upper_err {
            for i in 0..num_outputs {
                let mut e = 0.0f64;
                for j in 0..bias.len() {
                    e = add_coeff_err_bias_product_up(e, ue[[i, j]], bias[j]);
                }
                upper_bias_contrib[i] = add_f64_up(upper_bias_contrib[i], e);
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
    use super::{
        aw_batched_f64_with_abssum, aw_f32_sound_bound, aw_f64_with_abssum, gamma_n_f32,
        gamma_n_f64,
    };
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
    /// THE moat for #iso-batched-rebound: every per-domain block of the stacked
    /// [`aw_batched_f64_with_abssum`] product — BOTH the coefficient matrix `a64`
    /// AND the certified-error base `s` — must be BIT-IDENTICAL (raw f64 bits) to
    /// the per-domain [`aw_f64_with_abssum`] reference the shortcut rebound runs
    /// today. This is the faer row-count-independence property the whole batched
    /// forward relies on for its byte-identical parity gate. Varied shapes cover
    /// the iso backward regime: tiny input GEMMs (`p=5`), square intermediates,
    /// and mixed signs (so `s ≫ |a64|`, exercising cancellation).
    #[test]
    fn aw_batched_matches_per_domain_bit_for_bit() {
        use ndarray::Array2;
        // (m, k, p): m = target/spec rows, k = layer width, p = in-features.
        let shapes: &[(usize, usize, usize)] = &[
            (5, 50, 5),
            (50, 50, 50),
            (7, 100, 6),
            (13, 64, 5),
            (50, 5, 5),
            (1, 128, 5),
        ];
        let batches: &[usize] = &[1, 4, 33, 128];
        for &(m, k, p) in shapes {
            for &n_domains in batches {
                let mut rng =
                    SplitMix64(0xA5A5_1234 ^ ((m * 131 + k * 17 + p * 7 + n_domains) as u64));
                // Shared weight (all domains share the layer weight).
                let w = Mat::<f32>::from_fn(k, p, |_, _| rng.signed(4.0));
                // Per-domain coefficient blocks, stacked row-major.
                let a_stacked = Mat::<f32>::from_fn(n_domains * m, k, |_, _| rng.signed(4.0));

                let (a64_b, s_b) = aw_batched_f64_with_abssum(&a_stacked, &w);
                assert_eq!(a64_b.dim(), (n_domains * m, p));
                assert_eq!(s_b.dim(), (n_domains * m, p));

                for d in 0..n_domains {
                    let a_d = Mat::<f32>::from_fn(m, k, |i, j| a_stacked[(d * m + i, j)]);
                    let (a64_ref, s_ref): (Array2<f64>, Array2<f64>) = aw_f64_with_abssum(&a_d, &w);
                    for i in 0..m {
                        for j in 0..p {
                            assert_eq!(
                                a64_b[[d * m + i, j]].to_bits(),
                                a64_ref[[i, j]].to_bits(),
                                "a64 mismatch shape=({m},{k},{p}) batch={n_domains} dom={d} [{i},{j}]"
                            );
                            assert_eq!(
                                s_b[[d * m + i, j]].to_bits(),
                                s_ref[[i, j]].to_bits(),
                                "s mismatch shape=({m},{k},{p}) batch={n_domains} dom={d} [{i},{j}]"
                            );
                        }
                    }
                }
            }
        }
    }

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

    // -----------------------------------------------------------------------
    // S2 — the EFT-compensated arm (docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md)
    // -----------------------------------------------------------------------

    use super::{aw_eft_sound_bound, intersect_enclosure_downgrade_only};
    use ndarray::arr2;

    /// The f64 reference enclosure `[c64 − γ64·S, c64 + γ64·S]` provably
    /// contains the exact `A·W`, so a sound enclosure that is orders WIDER —
    /// the a-posteriori radius runs ~1e7× the f64 half-width on these shapes —
    /// must contain it too.
    ///
    /// SCOPE, stated honestly: this tolerance exceeds `2·half`, so what the
    /// assertions below actually decide is "the enclosure contains the exact
    /// `A·W`", not "it contains the f64 interval with room to spare". That is
    /// the containment property that matters, and it is the same idiom
    /// `aw_f32_sound_bound_contains_f64_bound` above uses. The *tight*
    /// statement — `|Σ a·w − value| ≤ err` decided in exact rationals, with
    /// mutation testing to prove the oracle has teeth — lives in
    /// `ny-core/tests/eft_certified_error_exact_rational_oracle.rs`, which
    /// governs the radius this function's arm publishes.
    fn containment_tol(c64: f64, half: f64) -> f64 {
        1.0e-9 * (1.0 + c64.abs() + half)
    }

    /// The intersection is the whole downgrade-only argument. It must never
    /// widen an endpoint, must ignore a NaN candidate, and must reject —
    /// wholesale, per entry — a candidate that would invert the interval (which
    /// can only happen if one arm is unsound; the proven one keeps the entry).
    #[test]
    fn intersection_never_widens_and_rejects_broken_candidates() {
        let base_l = arr2(&[[-1.0f32, -1.0, -1.0, -1.0]]);
        let base_u = arr2(&[[1.0f32, 1.0, 1.0, 1.0]]);

        // col 0: candidate strictly tighter        -> adopted
        // col 1: candidate strictly WIDER          -> incumbent kept
        // col 2: candidate NaN                     -> incumbent kept
        // col 3: candidate disjoint (would invert) -> incumbent kept
        let cand_l = arr2(&[[-0.5f32, -9.0, f32::NAN, 5.0]]);
        let cand_u = arr2(&[[0.5f32, 9.0, f32::NAN, 6.0]]);

        let mut lower = base_l.clone();
        let mut upper = base_u.clone();
        intersect_enclosure_downgrade_only(&mut lower, &mut upper, &cand_l, &cand_u);

        assert_eq!((lower[[0, 0]], upper[[0, 0]]), (-0.5, 0.5), "tighter arm");
        assert_eq!((lower[[0, 1]], upper[[0, 1]]), (-1.0, 1.0), "wider arm");
        assert_eq!((lower[[0, 2]], upper[[0, 2]]), (-1.0, 1.0), "NaN arm");
        assert_eq!((lower[[0, 3]], upper[[0, 3]]), (-1.0, 1.0), "disjoint arm");

        // Universal statement: no endpoint ever moved outward.
        for j in 0..4 {
            assert!(lower[[0, j]] >= base_l[[0, j]], "lower widened at {j}");
            assert!(upper[[0, j]] <= base_u[[0, j]], "upper widened at {j}");
        }
    }

    /// A shape-mismatched candidate is a refusal, not a panic and not a partial
    /// application.
    #[test]
    fn intersection_ignores_a_shape_mismatched_candidate() {
        let mut lower = arr2(&[[-1.0f32, -1.0]]);
        let mut upper = arr2(&[[1.0f32, 1.0]]);
        let cand_l = arr2(&[[0.0f32]]);
        let cand_u = arr2(&[[0.0f32]]);
        intersect_enclosure_downgrade_only(&mut lower, &mut upper, &cand_l, &cand_u);
        assert_eq!(lower, arr2(&[[-1.0f32, -1.0]]));
        assert_eq!(upper, arr2(&[[1.0f32, 1.0]]));
    }

    /// The a-posteriori arm must itself be a SOUND enclosure: it has to contain
    /// the f64-computed `A·W` (whose own error is `γ_n^f64·S`, subtracted here)
    /// on the cancellation-heavy regime S2 targets.
    #[test]
    fn eft_arm_encloses_the_f64_reference_product() {
        let cases: &[(usize, usize, usize, f32)] = &[
            (4, 8, 4, 1.0),
            (3, 256, 5, 2.0),
            (2, 1000, 3, 1.0),
            (4, 300, 4, 1.0e-3),
            (6, 50, 6, 1.0e3),
        ];
        for (idx, &(m, k, p, scale)) in cases.iter().enumerate() {
            let mut rng = SplitMix64(0x0EF7_0000_0000_0001_u64.wrapping_add(idx as u64));
            let a = Mat::<f32>::from_fn(m, k, |_, _| rng.signed(scale));
            let w = Mat::<f32>::from_fn(k, p, |_, _| rng.signed(1.0));

            let (lo, hi) = aw_eft_sound_bound(&a, &w, m, k, p)
                .expect("the EFT preconditions hold on this target");
            let (c64, s64) = aw_f64_with_abssum(&a, &w);
            let g64 = gamma_n_f64(k);

            for i in 0..m {
                for j in 0..p {
                    // The exact A·W lies in [c64 − γ64·S, c64 + γ64·S].
                    let half = g64 * s64[[i, j]];
                    let tol = containment_tol(c64[[i, j]], half);
                    let exact_lo = c64[[i, j]] - half;
                    let exact_hi = c64[[i, j]] + half;
                    assert!(
                        f64::from(lo[[i, j]]) <= exact_lo + tol,
                        "case {idx} [{i},{j}]: EFT lower {} does not enclose {exact_lo:e}",
                        lo[[i, j]]
                    );
                    assert!(
                        f64::from(hi[[i, j]]) >= exact_hi - tol,
                        "case {idx} [{i},{j}]: EFT upper {} does not enclose {exact_hi:e}",
                        hi[[i, j]]
                    );
                    assert!(lo[[i, j]] <= hi[[i, j]], "case {idx} [{i},{j}]: inverted");
                }
            }
        }
    }

    /// End-to-end on the shipped entry point: the gate is DARK by default, and
    /// arming it can only ever narrow the published enclosure — never widen it,
    /// never break containment of the f64-proven interval.
    #[test]
    fn eft_gate_is_dark_by_default_and_only_ever_narrows() {
        let _lock = ny_test_utils::env::lock_env();
        let (m, k, p) = (4usize, 512usize, 4usize);
        let mut rng = SplitMix64(0xC0FF_EE00_0000_0001);
        let a = Mat::<f32>::from_fn(m, k, |_, _| rng.signed(1.0));
        let w = Mat::<f32>::from_fn(k, p, |_, _| rng.signed(1.0));

        let (dark_lo, dark_hi) = {
            let _off = ny_test_utils::env::ScopedEnvVar::unset("NY_EFT_ERR");
            aw_f32_sound_bound(&a, &w)
        };
        let (armed_lo, armed_hi) = {
            let _on = ny_test_utils::env::ScopedEnvVar::set("NY_EFT_ERR", "1");
            aw_f32_sound_bound(&a, &w)
        };

        let (c64, s64) = aw_f64_with_abssum(&a, &w);
        let g64 = gamma_n_f64(k);
        let mut narrowed = 0usize;
        for i in 0..m {
            for j in 0..p {
                // (a) downgrade-only: never wider than dark.
                assert!(
                    armed_lo[[i, j]] >= dark_lo[[i, j]] && armed_hi[[i, j]] <= dark_hi[[i, j]],
                    "[{i},{j}]: arming the gate WIDENED the enclosure"
                );
                // (b) still sound: contains the f64-proven interval.
                let half = g64 * s64[[i, j]];
                let tol = containment_tol(c64[[i, j]], half);
                assert!(
                    f64::from(armed_lo[[i, j]]) <= c64[[i, j]] - half + tol
                        && f64::from(armed_hi[[i, j]]) >= c64[[i, j]] + half - tol,
                    "[{i},{j}]: armed enclosure lost containment of the f64 bound"
                );
                assert!(
                    armed_lo[[i, j]] <= armed_hi[[i, j]],
                    "[{i},{j}]: armed enclosure inverted"
                );
                if armed_hi[[i, j]] - armed_lo[[i, j]] < dark_hi[[i, j]] - dark_lo[[i, j]] {
                    narrowed += 1;
                }
            }
        }
        // (c) anti-vacuity: on the mixed-sign k=512 regime the a-posteriori arm
        // must actually be the binding one somewhere, or the channel is inert
        // and this test proves nothing.
        assert!(
            narrowed > 0,
            "the EFT arm never bound: the channel is inert on its own target regime"
        );
    }

    /// The refusal path: a fold that overflows f32 must make the a-posteriori
    /// arm decline entirely rather than publish a finite radius around a
    /// non-finite value. `aw_f32_sound_bound` then stands on its a-priori arm.
    #[test]
    fn eft_arm_refuses_a_non_finite_fold() {
        let a = Mat::<f32>::from_fn(1, 2, |_, _| f32::MAX);
        let w = Mat::<f32>::from_fn(2, 1, |_, _| f32::MAX);
        assert!(
            aw_eft_sound_bound(&a, &w, 1, 2, 1).is_none(),
            "an overflowing fold must fail closed, not publish a radius"
        );
    }
}

/// Parity + soundness for the faer f64 SIMD sub-threshold `aw_f64_with_abssum`
/// path (#linearizenn-faer-f64-aw): the blocked/SIMD f64 GEMM must live in the
/// SAME certified envelope as the historical scalar triple loop, and its
/// resulting enclosure `est ± γ_n·S` must still contain the exact real `A·W`.
#[cfg(test)]
mod faer_f64_aw_tests {
    use super::{
        aw_f64_with_abssum, aw_f64_with_abssum_and_deadline,
        aw_f64_with_abssum_cpu_deadline_scalar, aw_f64_with_abssum_deadline_via_engine_or_cpu,
        aw_via_engine_deadline, deadline_f64_accelerator_eligible, f32_to_f64_exact, gamma_n_f64,
        incoming_error_product, incoming_error_product_deadline, incoming_error_product_gemm,
        incoming_error_product_with_poll_quantum, next_up_nonnegative_f64,
        DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS, INCOMING_ERROR_POLL_MACS,
        SOUND_F64_GEMM_MIN_MACS,
    };
    use faer::Mat;
    use ndarray::{arr2, Array2};
    use ny_core::{GemmEngine, NyError, Result};
    use ny_tensor::next_up_f32;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

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

    #[derive(Clone, Debug, PartialEq)]
    struct BoundedCall {
        m: usize,
        k: usize,
        p: usize,
        max_dispatch_macs: usize,
        a: Vec<f64>,
        w: Vec<f64>,
    }

    fn exact_engine_product(m: usize, k: usize, p: usize, a: &[f64], w: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; m * p];
        for i in 0..m {
            for kk in 0..k {
                for j in 0..p {
                    output[i * p + j] += a[i * k + kk] * w[kk * p + j];
                }
            }
        }
        output
    }

    #[derive(Default)]
    struct RecordingDeadlineEngine {
        calls: Mutex<Vec<BoundedCall>>,
    }

    impl GemmEngine for RecordingDeadlineEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("finite-deadline Linear CROWN entered ordinary f32 GEMM")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            panic!("finite-deadline Linear CROWN entered ordinary f64 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            k: usize,
            p: usize,
            a: &[f64],
            w: &[f64],
            _deadline: Instant,
            max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            self.calls
                .lock()
                .expect("recording lock")
                .push(BoundedCall {
                    m,
                    k,
                    p,
                    max_dispatch_macs,
                    a: a.to_vec(),
                    w: w.to_vec(),
                });
            Ok(exact_engine_product(m, k, p, a, w))
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum ScriptedOutcome {
        UnsupportedFirst,
        OrdinaryFailureFirst,
        MalformedSecond,
        NonfiniteSecond,
        NegativeAbsSumSecond,
        DeadlineSecond,
    }

    struct ScriptedDeadlineEngine {
        outcome: ScriptedOutcome,
        calls: AtomicUsize,
    }

    impl ScriptedDeadlineEngine {
        fn new(outcome: ScriptedOutcome) -> Self {
            Self {
                outcome,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl GemmEngine for ScriptedDeadlineEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("deadline fallback entered ordinary f32 GEMM")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            panic!("deadline fallback entered ordinary f64 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            k: usize,
            p: usize,
            a: &[f64],
            w: &[f64],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            match (self.outcome, call) {
                (ScriptedOutcome::UnsupportedFirst, 0) => Err(NyError::UnsupportedOp(
                    "injected bounded-f64 unsupported".into(),
                )),
                (ScriptedOutcome::OrdinaryFailureFirst, 0) => Err(NyError::NumericalInstability(
                    "injected ordinary engine failure".into(),
                )),
                (ScriptedOutcome::MalformedSecond, 1) => {
                    Ok(vec![0.0; m.saturating_mul(p).saturating_sub(1)])
                }
                (ScriptedOutcome::NonfiniteSecond, 1) => Ok(vec![f64::NAN; m * p]),
                (ScriptedOutcome::NegativeAbsSumSecond, 1) => Ok(vec![-1.0; m * p]),
                (ScriptedOutcome::DeadlineSecond, 1) => Err(NyError::DeadlineExceeded(
                    "injected terminal second-product deadline".into(),
                )),
                _ => Ok(exact_engine_product(m, k, p, a, w)),
            }
        }
    }

    struct SleepPastDeadlineEngine {
        calls: AtomicUsize,
    }

    impl GemmEngine for SleepPastDeadlineEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("post-deadline test entered ordinary f32 GEMM")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            panic!("post-deadline test entered ordinary f64 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            _k: usize,
            p: usize,
            _a: &[f64],
            _b: &[f64],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(30));
            Ok(vec![0.0; m * p])
        }
    }

    fn deadline_engine_fixture() -> (Mat<f32>, Mat<f32>) {
        let a_vals = [[1.0f32, -2.0, 3.5], [-0.5, 8.0, -16.0]];
        let w_vals = [[2.0f32, -3.0], [-4.0, 5.0], [6.0, -7.0]];
        (
            Mat::<f32>::from_fn(2, 3, |i, j| a_vals[i][j]),
            Mat::<f32>::from_fn(3, 2, |i, j| w_vals[i][j]),
        )
    }

    #[test]
    fn deadline_f64_aw_policy_offloads_only_large_products() {
        assert!(!deadline_f64_accelerator_eligible(
            1,
            1,
            SOUND_F64_GEMM_MIN_MACS - 1
        ));
        assert!(deadline_f64_accelerator_eligible(
            1,
            1,
            SOUND_F64_GEMM_MIN_MACS
        ));
    }

    #[test]
    fn deadline_engine_uses_two_full_k_bounded_f64_products() {
        let (a, w) = deadline_engine_fixture();
        let engine = RecordingDeadlineEngine::default();
        let (actual_a, actual_s) =
            aw_via_engine_deadline(&engine, &a, &w, Instant::now() + Duration::from_mins(1))
                .expect("live bounded engine call")
                .expect("recording engine should be accepted");
        let (expected_a, expected_s) = naive_aw(&a, &w);

        assert_eq!(
            actual_a.as_slice().expect("contiguous A·W"),
            expected_a.as_slice()
        );
        assert_eq!(
            actual_s.as_slice().expect("contiguous abs sum"),
            expected_s.as_slice()
        );

        let calls = engine.calls.lock().expect("recording lock");
        assert_eq!(calls.len(), 2);
        for call in calls.iter() {
            assert_eq!(
                (call.m, call.k, call.p, call.max_dispatch_macs),
                (2, 3, 2, DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS)
            );
        }
        assert_eq!(calls[0].a, vec![1.0, -2.0, 3.5, -0.5, 8.0, -16.0]);
        assert_eq!(calls[0].w, vec![2.0, -3.0, -4.0, 5.0, 6.0, -7.0]);
        assert_eq!(calls[1].a, vec![1.0, 2.0, 3.5, 0.5, 8.0, 16.0]);
        assert_eq!(calls[1].w, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn bounded_engine_ordinary_and_malformed_failures_use_pollable_cpu() {
        let (a, w) = deadline_engine_fixture();
        // The pollable CPU fallback is single-chunk here, hence bit-identical
        // to the no-deadline faer twin (see
        // `deadline_f64_aw_cpu_path_is_bit_identical_to_the_unbounded_faer_twin`).
        // Any ordinary engine method reached instead of it panics via the fake.
        let (twin_a, twin_s) = aw_f64_with_abssum(&a, &w);
        for outcome in [
            ScriptedOutcome::UnsupportedFirst,
            ScriptedOutcome::OrdinaryFailureFirst,
            ScriptedOutcome::MalformedSecond,
            ScriptedOutcome::NonfiniteSecond,
            ScriptedOutcome::NegativeAbsSumSecond,
        ] {
            let engine = ScriptedDeadlineEngine::new(outcome);
            let (actual_a, actual_s) = aw_f64_with_abssum_deadline_via_engine_or_cpu(
                &engine,
                &a,
                &w,
                Instant::now() + Duration::from_mins(1),
            )
            .unwrap_or_else(|error| panic!("{outcome:?} should fall back to CPU: {error}"));

            assert_eq!(
                actual_a.as_slice().expect("contiguous A·W"),
                twin_a.as_slice().expect("contiguous twin A·W"),
                "{outcome:?} changed A·W"
            );
            assert_eq!(
                actual_s.as_slice().expect("contiguous abs sum"),
                twin_s.as_slice().expect("contiguous twin abs sum"),
                "{outcome:?} changed abs sum"
            );
        }
    }

    #[test]
    fn bounded_engine_deadline_is_terminal_even_after_first_product() {
        let (a, w) = deadline_engine_fixture();
        let engine = ScriptedDeadlineEngine::new(ScriptedOutcome::DeadlineSecond);
        let error = aw_f64_with_abssum_deadline_via_engine_or_cpu(
            &engine,
            &a,
            &w,
            Instant::now() + Duration::from_mins(1),
        )
        .expect_err("second-product deadline must be terminal");

        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert_eq!(engine.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn bounded_engine_post_deadline_result_is_never_published() {
        let (a, w) = deadline_engine_fixture();
        let engine = SleepPastDeadlineEngine {
            calls: AtomicUsize::new(0),
        };
        let error =
            aw_via_engine_deadline(&engine, &a, &w, Instant::now() + Duration::from_millis(10))
                .expect_err("post-deadline result must be rejected");

        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn expired_or_nonfinite_input_never_launches_bounded_engine() {
        let (a, w) = deadline_engine_fixture();
        let engine = RecordingDeadlineEngine::default();
        let error = aw_via_engine_deadline(
            &engine,
            &a,
            &w,
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("one millisecond fits before now"),
        )
        .expect_err("expired deadline must refuse");
        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert!(engine.calls.lock().expect("recording lock").is_empty());

        let nonfinite = Mat::<f32>::from_fn(2, 3, |i, j| {
            if i == 0 && j == 1 {
                f32::NAN
            } else {
                a[(i, j)]
            }
        });
        assert!(aw_via_engine_deadline(
            &engine,
            &nonfinite,
            &w,
            Instant::now() + Duration::from_mins(1)
        )
        .expect("non-finite input should safely decline")
        .is_none());
        assert!(engine.calls.lock().expect("recording lock").is_empty());
    }

    /// The single-chunk deadline arm must be BIT-IDENTICAL to the no-deadline
    /// faer twin (the b90a9fbf convergence property: same exactly-widened
    /// operands, same single `mat_mul_f64` pair), and the retained scalar core
    /// keeps its byte-for-byte pin against the historical naive reduction.
    #[test]
    fn deadline_f64_aw_cpu_path_is_bit_identical_to_the_unbounded_faer_twin() {
        let a_vals = [
            [1.0f32, -2.0, 3.5, -4.25, 0.0],
            [-0.5, 8.0, -16.0, 0.25, 2.0],
        ];
        let w_vals = [
            [2.0f32, -3.0, 0.5],
            [-4.0, 5.0, 0.25],
            [6.0, -7.0, -0.5],
            [-8.0, 9.0, 0.75],
            [10.0, -11.0, -1.0],
        ];
        let a = Mat::<f32>::from_fn(2, 5, |i, j| a_vals[i][j]);
        let w = Mat::<f32>::from_fn(5, 3, |i, j| w_vals[i][j]);

        let (actual_a, actual_s) =
            aw_f64_with_abssum_and_deadline(&a, &w, Some(Instant::now() + Duration::from_mins(1)))
                .expect("live deadline should complete");
        let (twin_a, twin_s) = aw_f64_with_abssum(&a, &w);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(
                    actual_a[[i, j]].to_bits(),
                    twin_a[[i, j]].to_bits(),
                    "deadline arm diverged from the no-deadline faer twin at [{i},{j}]"
                );
                assert_eq!(
                    actual_s[[i, j]].to_bits(),
                    twin_s[[i, j]].to_bits(),
                    "deadline |A|·|W| diverged from the no-deadline faer twin at [{i},{j}]"
                );
            }
        }

        // The scalar fallback core is the byte-for-byte 6f49a660 loop.
        let (scalar_a, scalar_s) =
            aw_f64_with_abssum_cpu_deadline_scalar(&a, &w, Instant::now() + Duration::from_mins(1))
                .expect("live scalar core");
        let (expected_a, expected_s) = naive_aw(&a, &w);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(
                    scalar_a[[i, j]].to_bits(),
                    expected_a[i * 3 + j].to_bits(),
                    "scalar core A·W reduction order changed at [{i},{j}]"
                );
                assert_eq!(
                    scalar_s[[i, j]].to_bits(),
                    expected_s[i * 3 + j].to_bits(),
                    "scalar core |A|·|W| reduction order changed at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn deadline_f64_aw_expired_fails_closed() {
        let a = Mat::<f32>::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
        let w = Mat::<f32>::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f32);
        let error = aw_f64_with_abssum_and_deadline(
            &a,
            &w,
            Some(
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("one second fits before the current instant"),
            ),
        )
        .expect_err("expired deadline must refuse certified f64 work");
        assert!(
            matches!(error, NyError::DeadlineExceeded(_)),
            "expected DeadlineExceeded, got {error:?}"
        );
    }

    #[test]
    fn incoming_error_accumulator_rounds_every_addition_upward() {
        // 2^-54 is below half an f64 ULP at 1.0, so round-to-nearest drops it.
        // The per-add successor must nevertheless enclose the exact real sum.
        let term = 2f64.powi(-54);
        assert_eq!(1.0 + term, 1.0);
        let enclosed = next_up_nonnegative_f64(1.0 + term);
        assert!(enclosed > 1.0 + term);
        assert_eq!(enclosed, f64::from_bits(1.0f64.to_bits() + 1));
        assert_eq!(next_up_nonnegative_f64(f64::NAN), f64::INFINITY);
        assert_eq!(
            next_up_nonnegative_f64(f64::from_bits(7)).to_bits(),
            8,
            "subnormal classification must not depend on DAZ"
        );
        assert_eq!(
            next_up_nonnegative_f64(-f64::from_bits(1)),
            f64::INFINITY,
            "invalid negative errors must fail closed"
        );
    }

    #[test]
    fn incoming_error_deadline_product_is_outward_and_pollable() {
        let tiny = 2f32.powi(-27);
        let error = arr2(&[[99.0f32, 1.0, tiny, tiny]]);
        let weights = Mat::<f32>::from_fn(3, 2, |k, j| {
            if j == 1 {
                0.0
            } else if k == 0 {
                1.0
            } else {
                tiny
            }
        });
        let result = incoming_error_product_deadline(
            &error,
            1,
            3,
            &weights,
            Instant::now() + Duration::from_mins(1),
        )
        .expect("live finite-deadline incoming-error product");

        assert!(
            f64::from(result[[0, 0]]) >= 1.0 + f64::EPSILON,
            "the outward result must cover terms lost by an ordinary f64 left fold"
        );
        assert_eq!(result[[0, 1]], 0.0);

        let error = incoming_error_product_deadline(
            &error,
            1,
            3,
            &weights,
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("one millisecond fits before the current instant"),
        )
        .expect_err("expired incoming-error composition must refuse");
        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
    }

    /// #err-compose-gemm equivalence pin: the GEMM lane must publish EXACTLY
    /// the f32 values the scalar stepped arm publishes, across zeros,
    /// subnormals, and normal magnitudes (the inertness claim of the withdrawn
    /// #cgan-row7-h4 analysis, now load-bearing). Deterministic LCG inputs.
    #[test]
    fn incoming_error_gemm_publishes_identical_values() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next_f32 = |zero_every: u64| -> f32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (state >> 40) as f32 / (1u64 << 24) as f32;
            if state.is_multiple_of(zero_every) {
                0.0
            } else if state.is_multiple_of(7) {
                f32::from_bits(((state >> 8) as u32) & 0x007f_ffff) // subnormal
            } else {
                r * 3.0
            }
        };
        for &(rows, k, cols) in &[
            (1usize, 1usize, 1usize),
            (3, 5, 4),
            (8, 33, 7),
            (2, 4096, 3),
        ] {
            let error = Array2::from_shape_fn((rows, k + 2), |_| next_f32(5));
            let mut w_abs = Mat::<f32>::zeros(k, cols);
            for kk in 0..k {
                for j in 0..cols {
                    w_abs[(kk, j)] = next_f32(4);
                }
            }
            let gemm = incoming_error_product_gemm(&error, 1, k, &w_abs, None)
                .expect("gemm lane must not error without a deadline")
                .expect("finite inputs must take the gemm lane");
            let scalar = incoming_error_product_with_poll_quantum(
                &error,
                1,
                k,
                &w_abs,
                None,
                INCOMING_ERROR_POLL_MACS,
            )
            .expect("scalar arm");
            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(
                        gemm[[i, j]].to_bits(),
                        scalar[[i, j]].to_bits(),
                        "published divergence at ({i},{j}) rows={rows} k={k} cols={cols}:                          gemm={} scalar={}",
                        gemm[[i, j]],
                        scalar[[i, j]],
                    );
                }
            }
        }
    }

    /// A poisoned (infinite) element must refuse the GEMM lane so the scalar
    /// arm's zero-dominates-infinity semantics stay authoritative.
    #[test]
    fn incoming_error_gemm_refuses_poisoned_inputs_to_the_scalar_arm() {
        let mut error = Array2::from_elem((2, 4), 0.5f32);
        error[[1, 2]] = f32::INFINITY;
        let mut w_abs = Mat::<f32>::zeros(3, 2);
        w_abs[(0, 0)] = 1.0;
        let lane = incoming_error_product_gemm(&error, 1, 3, &w_abs, None).expect("no deadline");
        assert!(
            lane.is_none(),
            "poisoned input must fall back to the scalar arm"
        );
        // And the full entry point still answers (via the scalar arm).
        let full = incoming_error_product(&error, 1, 3, &w_abs, None).expect("entry point");
        assert_eq!(full.nrows(), 2);
    }

    /// An expired deadline refuses the GEMM lane with the same typed message
    /// the scalar arm uses.
    #[test]
    fn incoming_error_gemm_refuses_expired_deadline() {
        let error = Array2::from_elem((4, 6), 0.25f32);
        let w_abs = Mat::<f32>::zeros(5, 4);
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(5))
            .expect("clock is past the epoch");
        let err = incoming_error_product_gemm(&error, 1, 5, &w_abs, Some(expired))
            .expect_err("expired deadline must refuse");
        assert!(matches!(err, NyError::DeadlineExceeded(_)));
    }

    // WAS DORMANT — this function carried no `#[test]` and had NEVER RUN. A
    // duplicated `#[test]` on the function above was masking it from clippy's
    // dead-code pass; removing that duplicate surfaced it. (Found twice
    // independently on 2026-08-18.)
    //
    // It is ENABLED because the open question has now been answered by running
    // it: it PASSES. The concern against enabling it was that its outcome was
    // unknown and a guess could turn the gate red everywhere — that concern was
    // right, and it is settled by measurement rather than by assumption.
    // Measured on ny-propagate --lib, faer_f64_aw_tests 15/15.
    //
    // Worth keeping live: it asserts an incoming-error product REJECTS a
    // negative subnormal weight (poisoning to +inf) and PRESERVES a positive
    // one — a sign/subnormal guard in the certified error path, which is
    // exactly the class of check that silently rotting is worst for.
    #[test]
    fn incoming_error_product_rejects_negative_subnormal_and_preserves_positive_one() {
        let weights = Mat::<f32>::from_fn(1, 1, |_i, _j| 2.0_f32.powi(120));

        let invalid = arr2(&[[f32::from_bits(0x8000_0001)]]);
        let poisoned =
            incoming_error_product(&invalid, 0, 1, &weights, None).expect("valid product geometry");
        assert_eq!(poisoned[[0, 0]], f32::INFINITY);

        let positive = arr2(&[[f32::from_bits(1)]]);
        let preserved = incoming_error_product(&positive, 0, 1, &weights, None)
            .expect("valid product geometry");
        assert!(
            f32_to_f64_exact(preserved[[0, 0]]) >= 2.0_f64.powi(-29),
            "positive subnormal error was lost: {}",
            preserved[[0, 0]]
        );

        let coefficient = Mat::<f32>::from_fn(1, 1, |_i, _j| f32::from_bits(1));
        let (product, absolute_sum) = aw_f64_with_abssum(&coefficient, &weights);
        assert_eq!(product[[0, 0]], 2.0_f64.powi(-29));
        assert_eq!(absolute_sum[[0, 0]], 2.0_f64.powi(-29));
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

#[cfg(test)]
mod f32_abssum_inflation_conditioning_tests {
    use super::{f32_abssum_inflation, gamma_n_f32};
    use num_rational::BigRational;
    use num_traits::One;

    /// EXACT `1/(1 - γ_k^f32)` as a rational, with no float in the oracle.
    /// `γ_k = d/(1-d)`, `d = k·2^-24`, so `1/(1-γ_k) = (1-d)/(1-2d)`.
    fn exact_tight_factor(k: usize) -> BigRational {
        let u = BigRational::new(1.into(), (1u64 << 24).into());
        let d = BigRational::from_integer(k.into()) * u;
        let one = BigRational::one();
        (one.clone() - d.clone()) / (one - BigRational::from_integer(2.into()) * d)
    }

    /// The OLD body — kept verbatim so the regression it caused stays visible.
    fn old_form(k: usize) -> f64 {
        (1.0 / (1.0 - gamma_n_f32(k))) * (1.0 + 2f64.powi(-40))
    }

    /// #f32-abssum-inflation-conditioning: the shipped factor must DOMINATE the
    /// exact tight factor at every admissible k, including the ill-conditioned
    /// region just under 2^23 where the old `1/(1-γ)` form cancelled.
    #[test]
    fn inflation_dominates_the_exact_factor_including_near_two_pow_23() {
        let mut checked = 0usize;
        for k in [
            1usize,
            2,
            7,
            47,
            1024,
            65_536,
            1_000_000,
            8_000_000,
            8_300_000,
            8_388_000,
            8_388_582,
            8_388_600,
            (1 << 23) - 1,
        ] {
            let Some(f_hat) = f32_abssum_inflation(k) else {
                continue;
            };
            let exact = exact_tight_factor(k);
            let got = BigRational::from_float(f_hat).expect("finite factor");
            assert!(
                got >= exact,
                "k={k}: shipped F_hat {f_hat} is BELOW the exact tight factor — \
                 an under-bound here under-charges S and can publish a wrongly tight bound"
            );
            checked += 1;
        }
        assert!(
            checked >= 10,
            "coverage: only {checked} admissible k values"
        );
    }

    /// The defect this repair fixes was real: the old form lands BELOW the exact
    /// factor at k = 8_388_582 (measured under-round 7.39e-12 relative). Pinning
    /// it stops anyone "simplifying" the stable form back to the cancelling one.
    #[test]
    fn the_old_cancelling_form_was_measurably_below_the_exact_factor() {
        let k = 8_388_582usize;
        let exact = exact_tight_factor(k);
        let old = BigRational::from_float(old_form(k)).expect("finite");
        assert!(
            old < exact,
            "the old 1/(1-gamma) form is expected to UNDER-round here; if this \
             now passes, the conditioning premise changed and the repair needs review"
        );
        let new =
            BigRational::from_float(f32_abssum_inflation(k).expect("admissible")).expect("finite");
        assert!(
            new >= exact,
            "the repaired form must dominate where the old one did not"
        );
    }

    /// The k >= 2^23 guard still refuses rather than returning a negative or
    /// non-finite factor (a negative stored error would be a false VERIFIED).
    #[test]
    fn guard_refuses_at_and_above_two_pow_23() {
        for k in [1usize << 23, (1 << 23) + 1, 1 << 24, usize::MAX / 2] {
            assert!(
                f32_abssum_inflation(k).is_none(),
                "k={k} must be refused so the caller takes the exact f64 abs-sum path"
            );
        }
    }
}

/// Engine-aware sound-f64 admission floor (#b4-engine-aware-macs-floor).
///
/// Three obligations, in order of importance:
///   1. GATE OFF ⇒ byte-identical admission for EVERY engine (the shared
///      `SOUND_F64_GEMM_MIN_MACS` policy is untouched, four arcs depend on it);
///   2. GATE ON ⇒ the faer override is what decides, including its measured
///      pathological declines, and an engine WITHOUT an override still gets the
///      historical constant;
///   3. the band the gate newly opens is still a valid enclosure — checked
///      against an EXACT rational oracle, not by eyeballing widths.
#[cfg(test)]
mod engine_aware_macs_floor_tests {
    use super::{
        aw_f64_with_abssum_cpu_deadline_scalar, aw_f64_with_abssum_deadline_via_engine_or_cpu,
        aw_via_engine_deadline, deadline_f64_accelerator_eligible, deadline_f64_engine_admits,
        engine_aware_macs_floor_armed, gamma_n_f64, set_engine_aware_macs_floor_for_test,
        ENGINE_AWARE_ABSOLUTE_MIN_MACS, SOUND_F64_GEMM_MIN_MACS,
    };
    use crate::faer_parallelism::FaerCpuGemmEngine;
    use faer::Mat;
    use num_rational::BigRational;
    use num_traits::Signed;
    use ny_core::{GemmEngine, NyError, Result, SoundF64GemmAdmission};
    use ny_tensor::next_up_f32;
    use std::time::{Duration, Instant};

    /// An engine that overrides NOTHING relevant — the stand-in for cuBLAS and
    /// every other existing backend. Its admission must stay the constant.
    struct UndeclaredEngine;
    impl GemmEngine for UndeclaredEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("test engine".into()))
        }
    }

    /// Counts bounded-f64 dispatches while delegating to the real faer engine —
    /// the non-vacuity witness that the engine path was actually taken.
    #[derive(Default)]
    struct CountingFaerEngine {
        bounded_f64_calls: std::sync::atomic::AtomicUsize,
    }
    impl GemmEngine for CountingFaerEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            FaerCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }
        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f64],
            b: &[f64],
            deadline: Instant,
            max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            self.bounded_f64_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            FaerCpuGemmEngine.gemm_f64_with_deadline(m, k, n, a, b, deadline, max_dispatch_macs)
        }
        fn sound_f64_deadline_admission(&self) -> SoundF64GemmAdmission {
            FaerCpuGemmEngine.sound_f64_deadline_admission()
        }
    }

    /// A deliberately over-eager declaration: it asks for everything. The
    /// engine-independent hard floor must still refuse trivially small products.
    struct GreedyEngine;
    impl GemmEngine for GreedyEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("test engine".into()))
        }
        fn sound_f64_deadline_admission(&self) -> SoundF64GemmAdmission {
            SoundF64GemmAdmission {
                min_macs: 0,
                min_rows: 0,
                min_contraction: 0,
                min_columns: 0,
                small_contraction_below: 0,
                small_contraction_max_output: usize::MAX,
            }
        }
    }

    /// Every shape in the B4 crossover + guard tables, plus the historical
    /// boundary. `(m, k, p)`.
    const SHAPE_GRID: &[(usize, usize, usize)] = &[
        // sub-crossover / thin
        (1, 16, 16),
        (2, 16, 16),
        (9, 8, 8),
        (4, 16, 16),
        (18, 8, 8),
        (2, 32, 32),
        (1, 64, 64),
        // measured engine wins, all currently gated out
        (9, 16, 16),
        (9, 32, 32),
        (18, 32, 32),
        (9, 64, 64),
        (18, 64, 64),
        (9, 128, 128),
        (18, 128, 128),
        (9, 256, 256),
        (64, 256, 256),
        (128, 256, 256),
        (200, 256, 256),
        (63, 512, 512),
        // the historical floor itself
        (64, 512, 512),
        (1, 4096, 4096),
        (512, 256, 256),
        (2048, 512, 512),
        // measured pathologies
        (4, 1, 4),
        (9, 1, 9),
        (64, 1, 64),
        (4096, 1, 4096),
        (64, 2, 64),
        (256, 2, 256),
        (1024, 2, 1024),
        (131072, 2, 64),
        (256, 4, 256),
        (64, 4, 64),
        (1024, 4, 1024),
        (2048, 4, 2048),
        (1, 512, 512),
        (1, 1048576, 16),
        // huge-k, NOT pathological on faer
        (4, 4194304, 1),
        (16, 1048576, 1),
        (4096, 4096, 1),
    ];

    fn constant_policy(m: usize, k: usize, p: usize) -> bool {
        m.saturating_mul(k).saturating_mul(p) >= SOUND_F64_GEMM_MIN_MACS
    }

    /// OBLIGATION 1. With the gate unset, the engine is irrelevant: the faer
    /// engine, an undeclared engine, and a greedy engine all reproduce the
    /// historical constant on every shape.
    #[test]
    fn gate_off_admission_is_byte_identical_to_the_shared_constant() {
        set_engine_aware_macs_floor_for_test(Some(false));
        assert!(!engine_aware_macs_floor_armed());
        for &(m, k, p) in SHAPE_GRID {
            let want = constant_policy(m, k, p);
            assert_eq!(
                deadline_f64_accelerator_eligible(m, k, p),
                want,
                "{m}x{k}x{p}: the shared constant predicate itself moved"
            );
            for (name, engine) in [
                ("faer", &FaerCpuGemmEngine as &dyn GemmEngine),
                ("undeclared", &UndeclaredEngine as &dyn GemmEngine),
                ("greedy", &GreedyEngine as &dyn GemmEngine),
            ] {
                assert_eq!(
                    deadline_f64_engine_admits(engine, m, k, p),
                    want,
                    "gate OFF, engine={name}, {m}x{k}x{p}: admission diverged from the constant"
                );
            }
        }
        set_engine_aware_macs_floor_for_test(None);
    }

    /// OBLIGATION 1 (end to end). Gate off, a sub-threshold product must be
    /// BIT-IDENTICAL to the pollable CPU reduction — i.e. no engine was even
    /// consulted, whatever is installed process-globally.
    #[test]
    fn gate_off_subthreshold_result_is_bit_identical_to_the_cpu_reduction() {
        set_engine_aware_macs_floor_for_test(Some(false));
        let (a, w) = random_operands(9, 64, 64, 0x51D2_A77E);
        let deadline = Instant::now() + Duration::from_mins(1);
        let (gated_a, gated_s) = super::aw_f64_with_abssum_and_deadline(&a, &w, Some(deadline))
            .expect("sub-threshold deadline product");
        let (cpu_a, cpu_s) =
            // Review defect 7: compare against the SCALAR core, not the faer
            // path. Once the deadline arm became faer-backed, a faer-backed
            // engine being consulted produced IDENTICAL bits and this oracle
            // silently stopped being able to detect it; the scalar fallback is
            // retained precisely to keep this check independent.
            aw_f64_with_abssum_cpu_deadline_scalar(&a, &w, deadline).expect("cpu reduction");
        for (lhs, rhs, label) in [(&gated_a, &cpu_a, "A·W"), (&gated_s, &cpu_s, "S")] {
            for (i, (x, y)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "gate OFF {label}[{i}]: {x:e} != {y:e} — an engine was consulted"
                );
            }
        }
        set_engine_aware_macs_floor_for_test(None);
    }

    /// OBLIGATION 2a. Armed, the FAER declaration decides — and it opens
    /// exactly the measured-win band while declining every measured pathology.
    #[test]
    fn armed_gate_uses_the_measured_faer_declaration() {
        set_engine_aware_macs_floor_for_test(Some(true));
        let faer = &FaerCpuGemmEngine as &dyn GemmEngine;

        // Newly opened: measured engine WINS that the 1<<24 constant gated out.
        for &(m, k, p, speedup) in &[
            (4usize, 16usize, 16usize, 1.29f64),
            (9, 16, 16, 1.83),
            (9, 32, 32, 2.71),
            (18, 32, 32, 3.70),
            (9, 64, 64, 3.17),
            (18, 128, 128, 4.96),
            (64, 256, 256, 7.13),
            (128, 256, 256, 12.72),
            (200, 256, 256, 17.18),
            (63, 512, 512, 16.49),
            (256, 4, 256, 1.789),
            (64, 4, 64, 1.704),
            (4, 4194304, 1, 7.450),
            (16, 1048576, 1, 9.386),
        ] {
            assert!(
                !constant_policy(m, k, p) || m * k * p >= SOUND_F64_GEMM_MIN_MACS,
                "{m}x{k}x{p}: grid bookkeeping"
            );
            assert!(
                deadline_f64_engine_admits(faer, m, k, p),
                "{m}x{k}x{p} (measured {speedup}x engine win) must be admitted when armed"
            );
        }

        // Declined: every measured LOSS, whatever its MAC count.
        for &(m, k, p, speedup, why) in &[
            (4usize, 1usize, 4usize, 0.001f64, "k==1 catastrophic"),
            (9, 1, 9, 0.003, "k==1 catastrophic"),
            (64, 1, 64, 0.027, "k==1 catastrophic"),
            (1024, 1, 1024, 0.236, "k==1 catastrophic"),
            (64, 2, 64, 0.780, "k==2"),
            (256, 2, 256, 0.656, "k==2"),
            (1024, 2, 1024, 0.437, "k==2"),
            (131072, 2, 64, 0.428, "k==2, large output"),
            (1024, 4, 1024, 0.812, "small k, large output"),
            (2048, 4, 2048, 0.599, "small k, large output"),
            (1, 32, 32, 0.535, "m==1"),
            (1, 512, 512, 0.555, "m==1"),
            (1, 1048576, 16, 0.630, "m==1"),
            (2, 16, 16, 0.79, "thin m"),
            (1, 64, 64, 0.55, "m==1"),
            (9, 8, 8, 1.71, "below the 1024-MAC crossover bracket"),
        ] {
            assert!(
                !deadline_f64_engine_admits(faer, m, k, p),
                "{m}x{k}x{p} ({why}, measured {speedup}x) must stay on the CPU path"
            );
        }

        // Large products the engine WINS stay admitted.
        for &(m, k, p, speedup) in &[
            (64usize, 512usize, 512usize, 15.36f64),
            (512, 256, 256, 20.07),
            (2048, 512, 512, 22.43),
            (4096, 4096, 1, 2.865),
        ] {
            assert!(constant_policy(m, k, p), "{m}x{k}x{p}: grid bookkeeping");
            assert!(
                deadline_f64_engine_admits(faer, m, k, p),
                "{m}x{k}x{p} (measured {speedup}x) must remain admitted"
            );
        }

        // THE OTHER DIRECTION. These are AT OR ABOVE the shared constant — so
        // they are dispatched to the engine TODAY — and every one of them is a
        // measured LOSS. Armed, the declaration declines them; unset, they are
        // untouched (asserted below).
        for &(m, k, p, speedup, why) in &[
            (4096usize, 1usize, 4096usize, 0.132f64, "k==1 at 16.7M MACs"),
            (65536, 1, 256, 0.182, "k==1 at 16.7M MACs"),
            (4194304, 1, 4, 0.426, "k==1, unbounded-arm analogue"),
            (131072, 2, 64, 0.428, "k==2 at 16.7M MACs"),
            (2048, 4, 2048, 0.599, "small k, large output, 16.7M MACs"),
            (1, 4096, 4096, 0.765, "m==1 at 16.7M MACs"),
            (1, 262144, 64, 0.712, "m==1 at 16.7M MACs"),
            (1, 1048576, 16, 0.630, "m==1 at 16.7M MACs"),
            (1, 8192, 8192, 0.83, "m==1 at 67M MACs"),
        ] {
            assert!(
                constant_policy(m, k, p),
                "{m}x{k}x{p}: this case only means something if the constant admits it"
            );
            assert!(
                !deadline_f64_engine_admits(faer, m, k, p),
                "{m}x{k}x{p} ({why}, measured {speedup}x) must be declined when armed"
            );
            set_engine_aware_macs_floor_for_test(Some(false));
            assert!(
                deadline_f64_engine_admits(faer, m, k, p),
                "{m}x{k}x{p} must stay admitted with the gate unset — the default is preserved"
            );
            set_engine_aware_macs_floor_for_test(Some(true));
        }
        set_engine_aware_macs_floor_for_test(None);
    }

    /// OBLIGATION 2b. Armed, an engine that does not override its declaration
    /// (cuBLAS and friends) keeps the 1<<24 constant exactly.
    #[test]
    fn armed_gate_keeps_the_constant_for_engines_without_a_declaration() {
        set_engine_aware_macs_floor_for_test(Some(true));
        let undeclared = &UndeclaredEngine as &dyn GemmEngine;
        assert_eq!(
            UndeclaredEngine.sound_f64_deadline_admission(),
            SoundF64GemmAdmission::CONSTANT_FLOOR
        );
        for &(m, k, p) in SHAPE_GRID {
            assert_eq!(
                deadline_f64_engine_admits(undeclared, m, k, p),
                constant_policy(m, k, p),
                "armed, undeclared engine, {m}x{k}x{p}: the default declaration must be the constant"
            );
        }
        set_engine_aware_macs_floor_for_test(None);
    }

    /// OBLIGATION 2c (fail closed). No declaration, however greedy, may open
    /// admission below the engine-independent hard floor.
    #[test]
    fn no_declaration_opens_below_the_engine_independent_hard_floor() {
        set_engine_aware_macs_floor_for_test(Some(true));
        let greedy = &GreedyEngine as &dyn GemmEngine;
        for &(m, k, p) in &[
            (1usize, 1usize, 1usize),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 7),
            (1, 511, 1),
        ] {
            assert!(
                m * k * p < ENGINE_AWARE_ABSOLUTE_MIN_MACS,
                "{m}x{k}x{p}: grid bookkeeping"
            );
            assert!(
                !deadline_f64_engine_admits(greedy, m, k, p),
                "{m}x{k}x{p} is below the hard floor and must be refused despite the declaration"
            );
        }
        // A zeroed declaration must not admit a degenerate operand either.
        assert!(!deadline_f64_engine_admits(greedy, 0, 4096, 4096));
        assert!(!deadline_f64_engine_admits(greedy, 4096, 0, 4096));
        set_engine_aware_macs_floor_for_test(None);
    }

    /// The `ny-core` default declaration and the shared `ny-propagate` constant
    /// are the same policy, pointwise.
    #[test]
    fn constant_floor_declaration_reproduces_the_shared_constant() {
        assert_eq!(
            SOUND_F64_GEMM_MIN_MACS,
            ny_core::SOUND_F64_GEMM_DEFAULT_MIN_MACS
        );
        for &(m, k, p) in SHAPE_GRID {
            assert_eq!(
                SoundF64GemmAdmission::CONSTANT_FLOOR.admits(m, k, p),
                constant_policy(m, k, p),
                "{m}x{k}x{p}"
            );
        }
    }

    fn random_operands(m: usize, k: usize, p: usize, seed: u64) -> (Mat<f32>, Mat<f32>) {
        let mut state = seed | 1;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let z = z ^ (z >> 31);
            let u = (z >> 40) as f32 / (1u64 << 24) as f32;
            (u * 2.0 - 1.0) * 1.5
        };
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..k * p).map(|_| next()).collect();
        (
            Mat::<f32>::from_fn(m, k, |i, j| a[i * k + j]),
            Mat::<f32>::from_fn(k, p, |i, j| w[i * p + j]),
        )
    }

    /// EXACT rational reference for one output entry: `f32 × f32` is exact in
    /// f64 (24+24 = 48 < 53 significand bits), so each product converts to a
    /// BigRational with no loss and the sum is the true real value.
    fn exact_entry(a: &Mat<f32>, w: &Mat<f32>, i: usize, j: usize) -> (BigRational, BigRational) {
        let k = a.ncols();
        let mut sum = BigRational::from_integer(0.into());
        let mut abs_sum = BigRational::from_integer(0.into());
        for kk in 0..k {
            let product = f64::from(a[(i, kk)]) * f64::from(w[(kk, j)]);
            let exact = BigRational::from_float(product).expect("finite exact product");
            abs_sum += exact.clone().abs();
            sum += exact;
        }
        (sum, abs_sum)
    }

    /// OBLIGATION 3 — THE MOAT. For shapes the armed gate NEWLY admits, the
    /// engine path's published enclosure must contain the exact real `A·W`
    /// (exact-rational oracle), its `S` must remain a valid certificate basis,
    /// and its half-width must not collapse relative to the CPU path's.
    #[test]
    fn newly_admitted_band_encloses_the_exact_product() {
        set_engine_aware_macs_floor_for_test(Some(true));
        let counting = CountingFaerEngine::default();
        let faer = &counting as &dyn GemmEngine;
        let deadline = Instant::now() + Duration::from_mins(2);
        let mut checked = 0usize;
        let mut reordered = 0usize;

        for (case, &(m, k, p)) in [
            (4usize, 16usize, 16usize),
            (9, 64, 64),
            (18, 128, 128),
            (256, 4, 256),
        ]
        .iter()
        .enumerate()
        {
            assert!(
                !constant_policy(m, k, p),
                "{m}x{k}x{p} must be BELOW the historical constant for this test to mean anything"
            );
            assert!(
                deadline_f64_engine_admits(faer, m, k, p),
                "{m}x{k}x{p} must be newly admitted when armed"
            );

            let (a, w) = random_operands(m, k, p, 0x2026_0810 ^ (case as u64) << 17);
            let (engine_aw, engine_s) = aw_via_engine_deadline(faer, &a, &w, deadline)
                .expect("bounded engine call")
                .expect("faer engine accepts this shape");
            let (cpu_aw, cpu_s) =
                // Review defect 7: compare against the SCALAR core, not the faer
            // path. Once the deadline arm became faer-backed, a faer-backed
            // engine being consulted produced IDENTICAL bits and this oracle
            // silently stopped being able to detect it; the scalar fallback is
            // retained precisely to keep this check independent.
            aw_f64_with_abssum_cpu_deadline_scalar(&a, &w, deadline).expect("cpu reduction");
            // What production would actually run for this shape when armed.
            let (routed_aw, routed_s) =
                aw_f64_with_abssum_deadline_via_engine_or_cpu(faer, &a, &w, deadline)
                    .expect("routed product");

            let gamma = gamma_n_f64(k);
            assert!(gamma.is_finite() && gamma > 0.0);
            let tiny = 8.0 * f64::from(f32::MIN_POSITIVE);

            for i in 0..m {
                for j in 0..p {
                    let ea = engine_aw[[i, j]];
                    let es = engine_s[[i, j]];
                    assert_eq!(ea.to_bits(), routed_aw[[i, j]].to_bits());
                    assert_eq!(es.to_bits(), routed_s[[i, j]].to_bits());
                    if ea.to_bits() != cpu_aw[[i, j]].to_bits() {
                        reordered += 1;
                    }

                    let (exact_c, exact_p) = exact_entry(&a, &w, i, j);
                    let exact_c_f = rational_to_f64(&exact_c);
                    let exact_p_f = rational_to_f64(&exact_p);

                    // (1) the certificate the caller charges actually covers the
                    // engine's accumulation error against the EXACT sum.
                    assert!(
                        (ea - exact_c_f).abs() <= gamma * es + tiny,
                        "case {case} [{i},{j}]: engine A·W={ea:e} exact={exact_c_f:e} \
                         err={:e} > gamma*S={:e}",
                        (ea - exact_c_f).abs(),
                        gamma * es
                    );
                    // (2) S is a valid basis — it never under-counts the exact
                    // abs-sum by more than its own f64 accumulation error.
                    assert!(
                        es >= exact_p_f * (1.0 - gamma) - tiny,
                        "case {case} [{i},{j}]: engine S={es:e} under-counts P={exact_p_f:e}"
                    );
                    // (3) the PUBLISHED enclosure contains the exact real value.
                    let stored = ea as f32;
                    let cast_err = (ea - f64::from(stored)).abs();
                    let err = f64::from(next_up_f32((cast_err + gamma * es) as f32));
                    let lo = f64::from(stored) - err;
                    let hi = f64::from(stored) + err;
                    assert!(
                        lo <= exact_c_f && exact_c_f <= hi,
                        "case {case} [{i},{j}]: published [{lo:e},{hi:e}] excludes exact \
                         {exact_c_f:e}"
                    );
                    // (4) NO NARROWING. Bit-equal widths are unsatisfiable for a
                    // reordering accelerator, so the provable statement is that
                    // the engine's half-width cannot fall outside the shared
                    // certificate envelope of the CPU path's.
                    let cpu_es = cpu_s[[i, j]];
                    assert!(
                        (es - cpu_es).abs() <= gamma * (es + cpu_es) + tiny,
                        "case {case} [{i},{j}]: engine S={es:e} collapsed vs cpu S={cpu_es:e}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
        // NON-VACUITY: 4 shapes × 2 products (A·W and |A|·|W|) × 2 routes
        // exercised (`aw_via_engine_deadline` directly, then the production
        // router) — the engine really ran, this is not a silent CPU fallback.
        let dispatches = counting
            .bounded_f64_calls
            .load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            dispatches, 16,
            "expected 16 bounded f64 engine dispatches, got {dispatches}"
        );
        // How often faer's blocked accumulation differs bitwise from the scalar
        // loop. NOT asserted non-zero: at these contraction widths faer's
        // micro-kernel can accumulate in the same k order, so bit-equality is a
        // legitimate (and stronger) outcome. Reported for the record.
        println!(
            "engine-vs-cpu bitwise differences: {reordered} of {checked} entries \
             across {dispatches} engine dispatches"
        );
        set_engine_aware_macs_floor_for_test(None);
    }

    fn rational_to_f64(value: &BigRational) -> f64 {
        use num_traits::ToPrimitive;
        value.to_f64().expect("exact reference fits f64 range")
    }
}

/// Deadline-scoped rounding-discipline tests. They pin:
///   (a) the production stepped enclosure and finite/unbounded BIT-PARITY;
///   (b) the withdrawn dual candidate's standalone certificate and its
///       quantization to the same production f32 value at n = 4096;
///   (c) the chunked-faer `A·W` certificate and scratch admission boundary;
///   (d) the mid-loop typed-deadline aborts.
#[cfg(test)]
mod deadline_rounding_discipline_tests {
    use super::{
        aw_f64_with_abssum, aw_f64_with_abssum_and_deadline,
        aw_f64_with_abssum_cpu_deadline_with_chunk_macs, deadline_aw_faer_owned_scratch_bytes,
        deadline_aw_faer_rows_per_chunk, f32_to_f64_exact, faer_f64_padded_row_capacity,
        gamma_n_f64, incoming_error_dual_factors, incoming_error_dual_upper,
        incoming_error_product, incoming_error_product_with_poll_quantum, next_up_nonnegative_f64,
        nonnegative_f32_error_or_infinity, publish_error_up_normal, DEADLINE_AW_FAER_CHUNK_MACS,
        DEADLINE_AW_FAER_MAX_SCRATCH_BYTES,
    };
    use faer::Mat;
    use ndarray::Array2;
    use num_rational::BigRational;
    use num_traits::{Signed, ToPrimitive};
    use ny_core::NyError;
    use ny_tensor::next_up_f32;
    use std::time::{Duration, Instant};

    /// Dependency-free deterministic PRNG (SplitMix64) — the same generator as
    /// the sibling oracle modules so cases reproduce bit-for-bit.
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
        /// Non-negative f32 with a MIXED exponent in `[2^-40, 2^31)` — the
        /// adversarial band for the incoming-error terms — with occasional
        /// exact zeros so the skip rule is exercised.
        fn nonneg_band(&mut self) -> f32 {
            if self.next_u64().is_multiple_of(11) {
                return 0.0;
            }
            let exponent = (self.next_u64() % 71) as i32 - 40;
            let mantissa = 1.0 + (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            mantissa * 2f32.powi(exponent)
        }
    }

    fn rational(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite exact value")
    }

    /// Exact real `Σ_k err[i, offset+k]·|W[k,j]|` with the production
    /// sanitization semantics (`x·0 → 0`); every f32×f32 product is exact in
    /// f64, so each term converts to a rational losslessly.
    fn exact_incoming_entry(
        error: &Array2<f32>,
        offset: usize,
        contraction: usize,
        w_abs: &Mat<f32>,
        i: usize,
        j: usize,
    ) -> BigRational {
        let mut sum = BigRational::from_integer(0.into());
        for kk in 0..contraction {
            let e = f32_to_f64_exact(error[[i, offset + kk]]);
            let w = f32_to_f64_exact(w_abs[(kk, j)]);
            if e == 0.0 || w == 0.0 {
                continue;
            }
            sum += rational(e * w);
        }
        sum
    }

    /// Byte-for-byte replica of the historical per-add stepped fold + publish
    /// for one entry — the reference for both finite and unbounded production
    /// calls.
    fn historical_stepped_entry(
        error: &Array2<f32>,
        offset: usize,
        contraction: usize,
        w_abs: &Mat<f32>,
        i: usize,
        j: usize,
    ) -> f32 {
        let mut sum = 0.0f64;
        for kk in 0..contraction {
            let error_value = nonnegative_f32_error_or_infinity(error[[i, offset + kk]]);
            let weight_abs = nonnegative_f32_error_or_infinity(w_abs[(kk, j)]);
            let term = if error_value == 0.0 || weight_abs == 0.0 {
                0.0
            } else {
                error_value * weight_abs
            };
            sum = if term == 0.0 {
                sum
            } else if term > 0.0 {
                next_up_nonnegative_f64(sum + term)
            } else {
                f64::INFINITY
            };
        }
        publish_error_up_normal(sum)
    }

    fn incoming_fixture(n: usize, rows: usize, cols: usize, seed: u64) -> (Array2<f32>, Mat<f32>) {
        let mut rng = SplitMix64(seed);
        // column_offset = 1 is exercised: column 0 is a decoy the product
        // must ignore.
        let error = Array2::from_shape_fn((rows, n + 1), |_| rng.nonneg_band());
        let w_abs = Mat::<f32>::from_fn(n, cols, |_, _| rng.nonneg_band());
        (error, w_abs)
    }

    /// (a): on mixed-exponent adversarial bands at n ∈ {3, 64, 4096}, both
    /// deadline modes must ENCLOSE the exact real sum and reproduce the
    /// historical per-add stepped publication bit-for-bit.
    #[test]
    fn incoming_error_stepped_bound_encloses_exact_and_matches_historical() {
        for &(n, seed) in &[
            (3usize, 0x0C6A_0001u64),
            (64, 0x0C6A_0002),
            (4096, 0x0C6A_0003),
        ] {
            let (rows, cols) = (3usize, 2usize);
            let (error, w_abs) = incoming_fixture(n, rows, cols, seed);
            let live = Instant::now() + Duration::from_mins(2);
            let deadline_bound = incoming_error_product(&error, 1, n, &w_abs, Some(live))
                .expect("live deadline-arm product");
            let unbounded =
                incoming_error_product(&error, 1, n, &w_abs, None).expect("unbounded product");

            for i in 0..rows {
                for j in 0..cols {
                    let exact = exact_incoming_entry(&error, 1, n, &w_abs, i, j);
                    for (label, published) in [
                        ("deadline", deadline_bound[[i, j]]),
                        ("unbounded", unbounded[[i, j]]),
                    ] {
                        assert!(
                            published.is_finite() && published >= 0.0,
                            "n={n} [{i},{j}]: {label} bound not a finite error: {published}"
                        );
                        assert!(
                            rational(f64::from(published)) >= exact,
                            "n={n} [{i},{j}]: {label} bound {published:e} EXCLUDES the exact \
                             sum {:e} — a tighter-than-truth error term",
                            rational_to_f64(&exact)
                        );
                    }
                    let old = historical_stepped_entry(&error, 1, n, &w_abs, i, j);
                    assert_eq!(
                        deadline_bound[[i, j]].to_bits(),
                        old.to_bits(),
                        "n={n} [{i},{j}]: deadline call drifted from stepped production"
                    );
                    assert_eq!(
                        unbounded[[i, j]].to_bits(),
                        old.to_bits(),
                        "n={n} [{i},{j}]: unbounded call drifted from stepped production"
                    );
                }
            }
        }
    }

    /// (b, test-only candidate): one exactly-representable large term followed by
    /// 4095 terms each just below half an ulp of the running sum. Round to
    /// nearest drops every small term, so the dual arm's `γ_{n+1}` charge is
    /// nearly TIGHT against the dropped mass, while the historical fold still
    /// pays a FULL ulp step per add — its excess over the exact sum is ~n·u
    /// against the dual arm's ~1–2·u (measured ratio ~2·10³, asserted ≥ 10).
    /// f32 publication quantizes both to the same value at this width, so the
    /// excess is compared on the pre-publish f64 bounds, using the test-only
    /// candidate helpers on a fold replicated term-for-term. The final check
    /// pins why this candidate remains withdrawn: both f64 bounds publish to
    /// the same f32, while production explicitly publishes `stepped`.
    #[test]
    fn test_only_dual_candidate_is_tighter_but_f32_publication_is_inert() {
        let n = 4096usize;
        let big_err = 1.0f32;
        let big_w = 1.0f32;
        let tiny_err = 1.0 - 2f32.powi(-23); // exactly representable, just below 1
        let tiny_w = 2f32.powi(-53); // normal f32; product exact in f64

        // Production term order: the big term first, then the tiny band.
        let mut terms = Vec::with_capacity(n);
        terms.push(f32_to_f64_exact(big_err) * f32_to_f64_exact(big_w));
        for _ in 1..n {
            terms.push(f32_to_f64_exact(tiny_err) * f32_to_f64_exact(tiny_w));
        }

        let mut exact = BigRational::from_integer(0.into());
        let mut stepped = 0.0f64;
        let mut acc = 0.0f64;
        for &term in &terms {
            exact += rational(term);
            stepped = next_up_nonnegative_f64(stepped + term);
            acc += term;
        }
        let (gamma, abs_inflate) = incoming_error_dual_factors(n);
        let dual = incoming_error_dual_upper(acc, gamma, abs_inflate);

        // Both bounds enclose the exact sum...
        assert!(rational(stepped) >= exact, "stepped fold lost enclosure");
        assert!(rational(dual) >= exact, "dual bound lost enclosure");
        // ...and the dual arm's excess is at least 10x smaller.
        let stepped_excess = rational(stepped) - exact.clone();
        let dual_excess = rational(dual) - exact;
        assert!(
            dual_excess.is_positive(),
            "dual bound must strictly enclose (its excess prices the charge)"
        );
        let ratio = rational_to_f64(&(stepped_excess / dual_excess));
        assert!(
            ratio >= 10.0,
            "excess ratio {ratio:.1} < 10x — the test-only dual candidate \
             lost its expected f64 improvement"
        );

        // Production remains the stepped fold under a finite deadline.
        let error =
            Array2::from_shape_fn((1, n), |(_, kk)| if kk == 0 { big_err } else { tiny_err });
        let w_abs = Mat::<f32>::from_fn(n, 1, |kk, _| if kk == 0 { big_w } else { tiny_w });
        let live = Instant::now() + Duration::from_mins(2);
        let published = incoming_error_product(&error, 0, n, &w_abs, Some(live))
            .expect("live deadline-arm product");
        let stepped_published = publish_error_up_normal(stepped);
        assert_eq!(
            published[[0, 0]].to_bits(),
            stepped_published.to_bits(),
            "finite-deadline production must publish the stepped fold"
        );
        assert_eq!(
            publish_error_up_normal(dual).to_bits(),
            stepped_published.to_bits(),
            "the test-only f64 improvement should remain inert after f32 publication"
        );
    }

    /// The accounting model must remain coupled to faer's actual owned-matrix
    /// layout. This samples both sides of its eight-row padding boundaries and
    /// counts all six matrices simultaneously live at peak.
    #[test]
    fn aw_deadline_faer_scratch_estimator_matches_owned_layout() {
        fn allocated_bytes(matrix: &Mat<f64>) -> usize {
            usize::try_from(matrix.col_stride())
                .expect("owned faer column stride is nonnegative")
                .checked_mul(matrix.ncols())
                .and_then(|elements| elements.checked_mul(size_of::<f64>()))
                .expect("small test matrix allocation size fits usize")
        }

        for &(k, p, rows) in &[(1, 3, 1), (7, 5, 8), (8, 2, 9), (17, 4, 15)] {
            let w_f = Mat::<f64>::zeros(k, p);
            let w_abs = Mat::<f64>::zeros(k, p);
            let a_f = Mat::<f64>::zeros(rows, k);
            let a_abs = Mat::<f64>::zeros(rows, k);
            let product = Mat::<f64>::zeros(rows, p);
            let abs_product = Mat::<f64>::zeros(rows, p);

            assert_eq!(
                usize::try_from(w_f.col_stride()).expect("nonnegative stride"),
                faer_f64_padded_row_capacity(k).expect("small row count")
            );
            assert_eq!(
                usize::try_from(a_f.col_stride()).expect("nonnegative stride"),
                faer_f64_padded_row_capacity(rows).expect("small row count")
            );

            let actual_bytes = [&w_f, &w_abs, &a_f, &a_abs, &product, &abs_product]
                .into_iter()
                .map(allocated_bytes)
                .try_fold(0usize, usize::checked_add)
                .expect("small aggregate allocation fits usize");
            assert_eq!(
                deadline_aw_faer_owned_scratch_bytes(k, p, rows),
                Some(actual_bytes),
                "scratch estimator diverged at k={k}, p={p}, rows={rows}"
            );
        }
    }

    /// (c, scratch boundary): one logical row still allocates eight faer rows
    /// for every owned matrix. At k=1, p=2^18−1 is 128 bytes below the
    /// 64 MiB ceiling and permits the full eight-row padding plateau; adding
    /// one output column is 128 bytes over and must decline faer entirely.
    #[test]
    fn aw_deadline_faer_admission_requires_one_complete_row_of_scratch() {
        let last_fitting_p = (1usize << 18) - 1;
        assert_eq!(
            deadline_aw_faer_owned_scratch_bytes(1, last_fitting_p, 1),
            Some(DEADLINE_AW_FAER_MAX_SCRATCH_BYTES - 128)
        );
        assert_eq!(
            deadline_aw_faer_owned_scratch_bytes(1, last_fitting_p + 1, 1),
            Some(DEADLINE_AW_FAER_MAX_SCRATCH_BYTES + 128)
        );
        assert_eq!(
            deadline_aw_faer_rows_per_chunk(1, last_fitting_p, DEADLINE_AW_FAER_CHUNK_MACS),
            Some(8),
            "all eight logical rows in the first padded block fit"
        );
        assert_eq!(
            deadline_aw_faer_rows_per_chunk(1, last_fitting_p + 1, DEADLINE_AW_FAER_CHUNK_MACS),
            None,
            "faer must decline when the remaining scratch cannot hold one complete row"
        );
        assert_eq!(
            deadline_aw_faer_owned_scratch_bytes(usize::MAX, 1, 1),
            None,
            "row-capacity rounding overflow must fail closed"
        );
        assert_eq!(
            deadline_aw_faer_rows_per_chunk(usize::MAX, 2, usize::MAX),
            None,
            "MAC-count overflow must fail closed"
        );
    }

    /// (c): the deadline=None arms are byte-identical to their historical
    /// forms — the incoming-error unbounded lane reproduces the per-add
    /// stepped publish bit-for-bit (including the poisoned-negative → +inf
    /// path), and the unbounded `A·W` entry is the untouched faer twin.
    #[test]
    fn deadline_none_arms_are_bit_identical_to_the_historical_forms() {
        let n = 64usize;
        let (rows, cols) = (4usize, 3usize);
        let (mut error, _) = incoming_fixture(n, rows, cols, 0x0C6A_00C0);
        // Strictly positive weights so the poisoned term below cannot be
        // silenced by the `err·0 → 0` domination rule.
        let mut rng = SplitMix64(0x0C6A_00C2);
        let w_abs = Mat::<f32>::from_fn(n, cols, |_, _| {
            let v = rng.nonneg_band();
            if v == 0.0 {
                1.0
            } else {
                v
            }
        });
        // Poison one entry with a negative payload: the invariant-violation
        // lane must still publish +inf identically.
        error[[2, 17]] = f32::from_bits(0x8000_0001);

        let got = incoming_error_product(&error, 1, n, &w_abs, None).expect("unbounded product");
        for i in 0..rows {
            for j in 0..cols {
                let want = historical_stepped_entry(&error, 1, n, &w_abs, i, j);
                assert_eq!(
                    got[[i, j]].to_bits(),
                    want.to_bits(),
                    "[{i},{j}]: unbounded incoming-error lane drifted from the \
                     historical stepped publish"
                );
            }
        }
        assert_eq!(got[[2, 0]], f32::INFINITY, "poisoned row must stay +inf");

        let mut rng = SplitMix64(0x0C6A_00C1);
        let a = Mat::<f32>::from_fn(4, 96, |_, _| rng.signed(2.0));
        let w = Mat::<f32>::from_fn(96, 5, |_, _| rng.signed(1.0));
        let (da, ds) = aw_f64_with_abssum_and_deadline(&a, &w, None).expect("unbounded A·W");
        let (ua, us) = aw_f64_with_abssum(&a, &w);
        for i in 0..4 {
            for j in 0..5 {
                assert_eq!(da[[i, j]].to_bits(), ua[[i, j]].to_bits(), "A·W [{i},{j}]");
                assert_eq!(ds[[i, j]].to_bits(), us[[i, j]].to_bits(), "S [{i},{j}]");
            }
        }
    }

    /// (a, `A·W`): the chunked-faer deadline path must keep the caller's
    /// order-independent `γ_k·S` certificate valid ACROSS CHUNK SEAMS
    /// (quantum forced to two rows per chunk) on cancellation-heavy,
    /// mixed-magnitude operands at k ∈ {3, 64, 4096}, against an
    /// exact-rational oracle — and stay inside the shared certificate
    /// envelope of the no-deadline faer twin.
    #[test]
    fn aw_deadline_chunked_path_encloses_the_exact_product_across_chunk_seams() {
        for &(k, scale, seed) in &[
            (3usize, 1.0f32, 0x0C6A_A001u64),
            (64, 1.0e3, 0x0C6A_A002),
            (4096, 1.0e-3, 0x0C6A_A003),
        ] {
            let (m, p) = (5usize, 3usize);
            let mut rng = SplitMix64(seed);
            let a = Mat::<f32>::from_fn(m, k, |_, _| rng.signed(scale));
            let w = Mat::<f32>::from_fn(k, p, |_, _| rng.signed(1.0));
            let deadline = Instant::now() + Duration::from_mins(2);
            // Two rows per chunk → three chunks over m = 5: seams exercised.
            let chunk_macs = k * p * 2;
            let (a64, s) =
                aw_f64_with_abssum_cpu_deadline_with_chunk_macs(&a, &w, deadline, chunk_macs)
                    .expect("live chunked product");
            let (twin_a, twin_s) = aw_f64_with_abssum(&a, &w);

            let gamma = gamma_n_f64(k);
            assert!(gamma.is_finite() && gamma > 0.0, "k={k}: bad gamma");
            let tiny = 8.0 * f64::from(f32::MIN_POSITIVE);

            for i in 0..m {
                for j in 0..p {
                    let mut exact_c = BigRational::from_integer(0.into());
                    let mut exact_p = BigRational::from_integer(0.into());
                    for kk in 0..k {
                        let term = f32_to_f64_exact(a[(i, kk)]) * f32_to_f64_exact(w[(kk, j)]);
                        exact_p += rational(term).abs();
                        exact_c += rational(term);
                    }
                    let exact_c_f = rational_to_f64(&exact_c);
                    let exact_p_f = rational_to_f64(&exact_p);

                    // (1) the caller's γ_k·S charge covers the chunked
                    // accumulation against the EXACT sum.
                    assert!(
                        (a64[[i, j]] - exact_c_f).abs() <= gamma * s[[i, j]] + tiny,
                        "k={k} [{i},{j}]: chunked A·W={:e} exact={exact_c_f:e} escapes γ·S={:e}",
                        a64[[i, j]],
                        gamma * s[[i, j]]
                    );
                    // (2) S stays a valid certificate basis.
                    assert!(
                        s[[i, j]] >= exact_p_f * (1.0 - gamma) - tiny,
                        "k={k} [{i},{j}]: chunked S={:e} under-counts P={exact_p_f:e}",
                        s[[i, j]]
                    );
                    // (3) the enclosure production publishes contains exact.
                    let stored = a64[[i, j]] as f32;
                    let cast_err = (a64[[i, j]] - f64::from(stored)).abs();
                    let err = f64::from(next_up_f32((cast_err + gamma * s[[i, j]]) as f32));
                    assert!(
                        f64::from(stored) - err <= exact_c_f
                            && exact_c_f <= f64::from(stored) + err,
                        "k={k} [{i},{j}]: published enclosure excludes the exact A·W"
                    );
                    // (4) shared-envelope agreement with the no-deadline twin.
                    assert!(
                        (a64[[i, j]] - twin_a[[i, j]]).abs()
                            <= gamma * (s[[i, j]] + twin_s[[i, j]]) + tiny,
                        "k={k} [{i},{j}]: chunked A·W left the twin's certificate envelope"
                    );
                }
            }
        }
    }

    /// (d, incoming): with the poll quantum forced to one MAC, a deadline that
    /// expires while the composition loop is in flight must surface as the
    /// TYPED deadline error at a mid-loop poll — never a partial result, never
    /// a panic. Determinism: entry is verified live, and the workload is
    /// ~1M polled MACs (an `Instant::now()` each) — orders of magnitude more
    /// than the 1 ms budget on any host.
    #[ntest::timeout(60_000)]
    #[test]
    fn incoming_error_deadline_poll_fires_mid_loop_with_typed_error() {
        let (rows, n) = (256usize, 4096usize);
        let error = Array2::<f32>::from_elem((rows, n), 1.0);
        let w_abs = Mat::<f32>::from_fn(n, 1, |_, _| 1.0);
        let deadline = Instant::now() + Duration::from_millis(1);
        assert!(
            Instant::now() < deadline,
            "deadline must be live at entry so the abort is mid-run"
        );
        let abort =
            incoming_error_product_with_poll_quantum(&error, 0, n, &w_abs, Some(deadline), 1)
                .expect_err("expiry across ~1M forced polls must abort");
        assert!(
            matches!(abort, NyError::DeadlineExceeded(_)),
            "expected the typed DeadlineExceeded, got {abort:?}"
        );
    }

    /// (d, `A·W`): with the chunk quantum forced to one row per chunk, expiry
    /// while the chunk loop is in flight must surface as the TYPED deadline
    /// error at a between-chunk poll (the `ops_gemm.rs` between-blocks
    /// pattern). 16384 chunks, each allocating two faer matrices and running
    /// two GEMMs — orders of magnitude more than the 1 ms budget.
    #[ntest::timeout(60_000)]
    #[test]
    fn aw_deadline_chunk_loop_aborts_between_chunks_with_typed_error() {
        let (m, k, p) = (16_384usize, 8usize, 8usize);
        let a = Mat::<f32>::from_fn(m, k, |i, j| (((i * 37 + j * 13) % 29) as f32 - 14.0) / 7.0);
        let w = Mat::<f32>::from_fn(k, p, |i, j| (((i * 17 + j * 11) % 23) as f32 - 11.0) / 5.0);
        let deadline = Instant::now() + Duration::from_millis(1);
        assert!(
            Instant::now() < deadline,
            "deadline must be live at entry so the abort is mid-run"
        );
        // k·p = 64 == chunk quantum → exactly one row per chunk.
        let abort = aw_f64_with_abssum_cpu_deadline_with_chunk_macs(&a, &w, deadline, 64)
            .expect_err("expiry across 16384 forced chunks must abort");
        assert!(
            matches!(abort, NyError::DeadlineExceeded(_)),
            "expected the typed DeadlineExceeded, got {abort:?}"
        );
    }

    fn rational_to_f64(value: &BigRational) -> f64 {
        value.to_f64().expect("exact reference fits f64 range")
    }
}
