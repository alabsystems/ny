// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::mem::size_of;

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};

thread_local! {
    static RAYON_TASK_DEPTH: Cell<u32> = const { Cell::new(0) };
    /// Depth of active [`NestedFaerParGuard`] scopes on this thread. When > 0,
    /// [`current_par`] permits faer to use the configured global (Rayon)
    /// parallelism even while running on a Rayon worker, instead of forcing
    /// `Par::Seq`. Set ONLY around the per-domain input-split CROWN collection
    /// (#tll-nested-collect-par), whose wide f64 `A·W` / `|A|·|W|` GEMMs would
    /// otherwise run single-threaded on ONE Rayon worker while the other cores
    /// sit idle in the shallow phase of the BaB tree (few concurrent domains).
    static NESTED_FAER_PAR_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Process-global enable for the nested-faer-parallel input-split collection
/// (#tll-nested-collect-par). Default ON (batteries-included); set
/// `NY_INPUT_SPLIT_NESTED_PAR=0` to restore the historical `Par::Seq`-inside-
/// Rayon behavior byte-for-byte (the A/B + parity reference). Read once.
fn nested_faer_par_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("NY_INPUT_SPLIT_NESTED_PAR").ok().as_deref(),
            Some("0")
        )
    })
}

fn nested_faer_par_active() -> bool {
    NESTED_FAER_PAR_DEPTH.with(|d| d.get() > 0)
}

/// RAII guard that permits nested faer (Rayon) parallelism for GEMMs issued in
/// its scope — even on a Rayon worker thread. Deadlock-free: faer's explicit
/// `Par::Rayon` path drives a work-stealing `rayon` scope (NOT the `&Mat * &Mat`
/// operator that #4392 avoided), so nesting it inside an outer `par_iter` worker
/// participates in work-stealing rather than blocking. Self-balancing: when the
/// BaB batch already saturates every core, the nested scope finds no idle worker
/// to steal and runs effectively sequentially (no oversubscription); when cores
/// are idle (shallow tree, 1–few domains) the wide collection GEMM fans out
/// across them. SOUNDNESS: this changes ONLY the GEMM summation order (blocked
/// SIMD across threads); the certified `γ_n·S` coefficient-error envelope is
/// summation-order independent (the identical Higham argument that already
/// routes these products through cuBLAS/faer), so every bound stays a valid
/// enclosure. A no-op when [`nested_faer_par_enabled`] is false.
#[must_use = "hold the guard for the full nested-parallel collection scope"]
pub(crate) struct NestedFaerParGuard {
    active: bool,
}

impl NestedFaerParGuard {
    pub(crate) fn new() -> Self {
        let active = nested_faer_par_enabled();
        if active {
            NESTED_FAER_PAR_DEPTH.with(|d| d.set(d.get().saturating_add(1)));
        }
        Self { active }
    }
}

impl Drop for NestedFaerParGuard {
    fn drop(&mut self) {
        if self.active {
            NESTED_FAER_PAR_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        }
    }
}

/// Matrix multiply with explicit faer parallelism control.
///
/// faer 0.21.9's `&Mat * &Mat` operator uses `get_global_parallelism()`
/// (`src/operator/operator_impl/matref.rs`, `src/lib.rs:1077-1097`). Inside a
/// Rayon worker, that schedules nested Rayon work and can deadlock when the
/// outer task is already running under `par_iter()` (#4392). Callers should use
/// `mat_mul()` instead of the operator so Rayon-scoped work degrades to
/// `Par::Seq` while sequential callers preserve the configured global policy.
pub(crate) fn mat_mul_with_par(a: &Mat<f32>, b: &Mat<f32>, par: Par) -> Mat<f32> {
    let mut dst = Mat::<f32>::zeros(a.nrows(), b.ncols());
    matmul(&mut dst, Accum::Replace, a, b, 1.0, par);
    dst
}

pub(crate) fn mat_mul(a: &Mat<f32>, b: &Mat<f32>) -> Mat<f32> {
    mat_mul_with_par(a, b, current_par())
}

/// f64 twin of [`mat_mul`] for the sound conv coefficient recompute
/// (#vnncomp-aw-soundness). Same Rayon-aware parallelism policy. The certified
/// conv error `cast + γ_n^f64·S + prop` is summation-order independent, so
/// faer's blocked accumulation order is covered by the same certificate as the
/// scalar loop it replaces.
pub(crate) fn mat_mul_f64(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let mut dst = Mat::<f64>::zeros(a.nrows(), b.ncols());
    matmul(&mut dst, Accum::Replace, a, b, 1.0, current_par());
    dst
}

/// f64 matrix multiply over row-major slices without repacking them into owned
/// faer matrices first.
///
/// This is the wide-root Conv adapter: its im2col tile and weights already live
/// in row-major buffers. faer may pack those strided views internally for its
/// blocked kernel, but NY avoids an additional full-size user-space copy. The
/// arithmetic contract is identical to [`mat_mul_f64`]: IEEE round-to-nearest
/// f64 products and sums in an implementation-defined order, covered by the
/// caller's summation-order-independent Higham envelope.
pub(crate) fn mat_mul_f64_row_major(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
) -> Mat<f64> {
    assert_eq!(a.len(), m * k, "row-major lhs shape mismatch");
    assert_eq!(b.len(), k * n, "row-major rhs shape mismatch");
    let a = MatRef::from_row_major_slice(a, m, k);
    let b = MatRef::from_row_major_slice(b, k, n);
    let mut dst = Mat::<f64>::zeros(m, n);
    matmul(&mut dst, Accum::Replace, a, b, 1.0, current_par());
    dst
}

/// faer-backed CPU [`GemmEngine`](ny_core::GemmEngine) (#cgan-batched-stack).
///
/// `NaiveCpuGemmEngine::gemm_f32` is a single-threaded scalar triple loop
/// (~1-3 GFLOP/s); on the batched dense-spec rebound the stacked conv
/// backward GEMMs (e.g. 25088x512x64 on cgan's ConvTranspose_10) dominate the
/// whole BaB batch at that rate. This engine routes `gemm_f32`/`gemm_f64`
/// through faer's blocked SIMD matmul with the crate's Rayon-aware
/// parallelism policy.
///
/// CONTRACT COMPLIANCE: `GemmEngine::gemm_f32` requires "plain IEEE
/// round-to-nearest f32 arithmetic (any summation order)" — faer's blocked
/// accumulation is exactly RN-f32 with a reordered sum, which the
/// verdict-feeding callers' order-independent certified error bounds already
/// cover (the same Higham argument used to route these products through
/// cuBLAS). All other `GemmEngine` methods keep their trait defaults
/// (fallbacks to `gemm_f32` / CPU paths).
pub(crate) struct FaerCpuGemmEngine;

#[derive(Clone, Copy)]
struct CheckedGemmLengths {
    lhs: usize,
    rhs: usize,
    output: usize,
    output_bytes: usize,
}

fn checked_gemm_lengths(
    m: usize,
    k: usize,
    n: usize,
    element_size: usize,
    site: &'static str,
) -> ny_core::Result<CheckedGemmLengths> {
    let lhs = m
        .checked_mul(k)
        .ok_or_else(|| ny_core::NyError::InvalidSpec(format!("{site}: m*k overflows usize")))?;
    let rhs = k
        .checked_mul(n)
        .ok_or_else(|| ny_core::NyError::InvalidSpec(format!("{site}: k*n overflows usize")))?;
    let output = m
        .checked_mul(n)
        .ok_or_else(|| ny_core::NyError::InvalidSpec(format!("{site}: m*n overflows usize")))?;
    let output_bytes = output.checked_mul(element_size).ok_or_else(|| {
        ny_core::NyError::InvalidSpec(format!("{site}: output byte count overflows usize"))
    })?;
    Ok(CheckedGemmLengths {
        lhs,
        rhs,
        output,
        output_bytes,
    })
}

fn allocate_gemm_output<T: Copy>(
    shape: CheckedGemmLengths,
    zero: T,
    site: &'static str,
) -> ny_core::Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(shape.output)
        .map_err(|_| ny_core::NyError::CpuMemoryExceeded {
            required_bytes: shape.output_bytes,
            budget_bytes: usize::MAX,
            site,
        })?;
    output.resize(shape.output, zero);
    Ok(output)
}

impl ny_core::GemmEngine for FaerCpuGemmEngine {
    fn backend_provenance(&self) -> &'static str {
        "faer-cpu"
    }

    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        const SITE: &str = "FaerCpuGemmEngine::gemm_f32";
        let shape = checked_gemm_lengths(m, k, n, size_of::<f32>(), SITE)?;
        if a.len() != shape.lhs || b.len() != shape.rhs {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "{SITE}: a.len()={} (want {m}*{k}={}) b.len()={} (want {k}*{n}={})",
                a.len(),
                shape.lhs,
                b.len(),
                shape.rhs,
            )));
        }
        if m == 0 || k == 0 || n == 0 {
            return allocate_gemm_output(shape, 0.0f32, SITE);
        }
        // Borrow the caller's row-major buffers directly instead of repacking
        // them into owned column-major `Mat`s first.
        //
        // `Mat::from_fn` fills COLUMN-major, so `from_fn(m, k, |i, j| a[i*k+j])`
        // walked a row-major source with stride `k` — one cache line per
        // element — and the unpack loop below it did the same again on the
        // result. Sample-profiling the patches compose seam attributed **2,531
        // of its 4,791 samples (52.8%)** to this repack, ~15.9% of the whole
        // main thread. faer may pack the strided views internally for its
        // blocked kernel, but that is its own cache-aware packing, not a
        // full-size user-space copy.
        //
        // This mirrors [`mat_mul_f64_row_major`] exactly. The arithmetic
        // contract is unchanged: `GemmEngine::gemm_f32` requires plain IEEE
        // round-to-nearest f32 in ANY summation order, and faer's blocked
        // accumulation was already relied upon under that clause — the callers'
        // certified error is the summation-order-independent Higham envelope,
        // and it is computed from the INCOMING coefficients, never from this
        // GEMM's output.
        let a_mat = MatRef::from_row_major_slice(a, m, k);
        let b_mat = MatRef::from_row_major_slice(b, k, n);
        let mut c = Mat::<f32>::zeros(m, n);
        matmul(&mut c, Accum::Replace, a_mat, b_mat, 1.0, current_par());
        let mut out = allocate_gemm_output(shape, 0.0f32, SITE)?;
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = c[(i, j)];
            }
        }
        Ok(out)
    }

    /// Deadline-bounded f64 GEMM (#cpu-sound-f64-deadline).
    ///
    /// WHY THIS EXISTS. Without it the trait default returns `UnsupportedOp`,
    /// and — because that default is contractually forbidden from delegating to
    /// the unbounded `gemm_f64` — EVERY timeout-bearing run bypassed the engine
    /// entirely. Measured before this landed: 82 dispatches with no deadline set,
    /// **0 with one**. A VNN-COMP run always carries a deadline, so the sound f64
    /// engine was structurally unreachable exactly where it matters.
    ///
    /// THE SOUNDNESS-CRITICAL RULE, and the reason the tiling looks the way it
    /// does: split ONLY the output `m`/`n` axes, NEVER the contraction `k`. Every
    /// output coefficient therefore remains ONE ordinary length-`k` RN-f64 dot
    /// product, which is what keeps the caller's summation-order-independent
    /// `gamma_k * S` certificate valid. Splitting `k` would turn one dot product
    /// into a sum of partial dots with a DIFFERENT error structure than the
    /// certificate charges. Because `k` is indivisible, a contraction larger than
    /// the whole dispatch cap cannot be made bounded at all, and this refuses
    /// rather than silently exceeding the cap — the same choice the CUDA engine
    /// makes (`ny-cuda/src/lib.rs`, "contraction {k} exceeds dispatch cap").
    ///
    /// Polls before every tile and after the final tile, and returns
    /// `DeadlineExceeded` rather than publishing a late result.
    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        deadline: std::time::Instant,
        max_dispatch_macs: usize,
    ) -> ny_core::Result<Vec<f64>> {
        const SITE: &str = "FaerCpuGemmEngine::gemm_f64_with_deadline";
        let past = |site: &str| {
            ny_core::NyError::DeadlineExceeded(format!("{SITE}: deadline exceeded {site}"))
        };
        if std::time::Instant::now() >= deadline {
            return Err(past("before launch"));
        }
        let shape = checked_gemm_lengths(m, k, n, size_of::<f64>(), SITE)?;
        if a.len() != shape.lhs || b.len() != shape.rhs {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "{SITE}: a.len()={} (want {m}*{k}={}) b.len()={} (want {k}*{n}={})",
                a.len(),
                shape.lhs,
                b.len(),
                shape.rhs,
            )));
        }
        let mut out = allocate_gemm_output(shape, 0.0f64, SITE)?;
        if m == 0 || k == 0 || n == 0 {
            return Ok(out);
        }
        if max_dispatch_macs == 0 {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "{SITE}: max_dispatch_macs must be non-zero"
            )));
        }
        // k is indivisible (see the doc comment): one output coefficient is one
        // length-k dot, so the smallest possible tile already costs k MACs.
        if k > max_dispatch_macs {
            return Err(ny_core::NyError::UnsupportedOp(format!(
                "{SITE}: contraction k={k} exceeds dispatch cap {max_dispatch_macs}; \
                 splitting k would invalidate the caller's gamma_k*S certificate"
            )));
        }
        // Widest column tile whose single-row cost fits the cap, then as many
        // rows as still fit. Both are >= 1 because k <= max_dispatch_macs.
        let cols_per = match k.checked_mul(n) {
            Some(row_macs) if row_macs <= max_dispatch_macs => n,
            _ => (max_dispatch_macs / k).max(1),
        };
        let tile_row_macs = k.saturating_mul(cols_per).max(1);
        let rows_per = (max_dispatch_macs / tile_row_macs).max(1);

        let b_mat = MatRef::from_row_major_slice(b, k, n);
        let mut i0 = 0usize;
        while i0 < m {
            let rows = rows_per.min(m - i0);
            let a_blk = MatRef::from_row_major_slice(&a[i0 * k..(i0 + rows) * k], rows, k);
            let mut j0 = 0usize;
            while j0 < n {
                if std::time::Instant::now() >= deadline {
                    return Err(past("between tiles"));
                }
                let cols = cols_per.min(n - j0);
                let b_tile = b_mat.submatrix(0, j0, k, cols);
                let mut dst = Mat::<f64>::zeros(rows, cols);
                matmul(&mut dst, Accum::Replace, a_blk, b_tile, 1.0, current_par());
                for i in 0..rows {
                    for j in 0..cols {
                        out[(i0 + i) * n + j0 + j] = dst[(i, j)];
                    }
                }
                j0 += cols;
            }
            i0 += rows;
        }
        if std::time::Instant::now() >= deadline {
            return Err(past("after final tile"));
        }
        Ok(out)
    }

    /// MEASURED crossover for this CPU engine's deadline-bounded f64 seam
    /// (#b4-engine-aware-macs-floor).
    ///
    /// WHY IT DIFFERS FROM THE DEFAULT BY ~16,000×. The shared constant
    /// [`ny_core::SOUND_F64_GEMM_DEFAULT_MIN_MACS`] (`1<<24`) is the *GPU*
    /// crossover: a cuBLAS dispatch costs ~0.4 ms of launch latency. This engine
    /// runs in-process at ~1 µs of call overhead, and — on the deadline path —
    /// competes not against faer but against the pollable CPU fall-through
    /// (`aw_f64_with_abssum_cpu_deadline`; a single-threaded scalar triple
    /// loop when these measurements were taken, chunked faer `mat_mul_f64`
    /// since the b90a9fbf-mirror landing — the measured band below therefore
    /// OVERSTATES the engine's edge and is due a re-measure). The crossover sits
    /// three orders of magnitude lower and the whole 1e3..1.7e7 MAC band, where
    /// 3×–17× lives, was being gated out.
    ///
    /// MEASUREMENTS (this host, best-of-N µs/call, A = pollable scalar loop,
    /// B = this engine via `aw_via_engine_deadline`; A/B > 1 means engine wins):
    ///   4x16x16 (1,024 MACs) 1.29×;  9x32x32 3.71×;  18x128x128 4.96×;
    ///   64x256x256 7.13×;  128x256x256 12.7×;  200x256x256 17.2×;
    ///   512x256x256 20.1×;  2048x512x512 22.4×.
    /// The crossover is bracketed 512 < x <= 1,024 MACs for `m >= 4`.
    ///
    /// THE THREE DECLINES ARE ALSO MEASURED, not guessed:
    ///   * `k == 1` is catastrophic — 0.001×–0.24× (a ~65–70 µs/call fixed cost
    ///     inside faer's `matmul` that exists at `k == 1` and nowhere else;
    ///     `9x1x9` spends 131.8 of its 137 µs in two bare `matmul` calls on
    ///     zero-copy `MatRef`s, while `9x2x9` spends 0.2 µs). `min_contraction`
    ///     of 4 excludes it with margin, since `k == 2` also loses everywhere
    ///     measured (0.43×–0.82×).
    ///   * `m == 1` loses at EVERY scale, 1024 MACs through 16.7 M
    ///     (0.478×–0.775×); `m == 2` is break-even and `m >= 4` wins from
    ///     1.29×, so `min_rows` is 4.
    ///   * a small `k` feeding a large output is output-detranspose bound:
    ///     `k = 4` WINS at `256x4x256` (1.79×) and `64x4x64` (1.70×) but LOSES
    ///     at `1024x4x1024` (0.81×) and `2048x4x2048` (0.63×). Hence
    ///     `k < 16 => m*p <= 65,536`.
    ///
    /// NOT declined: huge `k` is fine here (`4x4194304x1` = 7.45×,
    /// `16x1048576x1` = 9.39×) — the Accelerate tall-skinny blowup does not
    /// reproduce on faer.
    ///
    /// This declaration is consulted ONLY when `NY_ENGINE_AWARE_MACS_FLOOR` is
    /// armed; unset, the caller uses the historical constant unchanged.
    fn sound_f64_deadline_admission(&self) -> ny_core::SoundF64GemmAdmission {
        ny_core::SoundF64GemmAdmission {
            min_macs: 1 << 10,
            min_rows: 4,
            min_contraction: 4,
            min_columns: 1,
            small_contraction_below: 16,
            small_contraction_max_output: 1 << 16,
        }
    }

    fn gemm_f64(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
    ) -> ny_core::Result<Vec<f64>> {
        const SITE: &str = "FaerCpuGemmEngine::gemm_f64";
        let shape = checked_gemm_lengths(m, k, n, size_of::<f64>(), SITE)?;
        if a.len() != shape.lhs || b.len() != shape.rhs {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "{SITE}: a.len()={} (want {m}*{k}={}) b.len()={} (want {k}*{n}={})",
                a.len(),
                shape.lhs,
                b.len(),
                shape.rhs,
            )));
        }
        if m == 0 || k == 0 || n == 0 {
            return allocate_gemm_output(shape, 0.0f64, SITE);
        }
        // ZERO-COPY OPERANDS (#b4-gemm-f64-repack). This used to rebuild BOTH
        // operands with `Mat::<f64>::from_fn(..)` — a full out-of-place
        // transposition — on top of the marshalling the caller had already done,
        // while the sibling `gemm_f32` and `gemm_f64_with_deadline` were already
        // zero-copy via `MatRef::from_row_major_slice`. Measured cost of that
        // asymmetry: the repack was 9.3% of the engine path at 100x100x2048 and
        // 53.4% at 1x4096x4096, and it is the main reason the UNBOUNDED engine
        // path never beat its fall-through in 72 measurements — an implementation
        // artifact, not a property of the seam.
        //
        // SOUNDNESS: identical arithmetic — the same `faer::matmul` under the same
        // `current_par()` policy. Only faer's internal blocking may differ, which
        // changes summation ORDER, and the caller's certified `gamma_n*S` envelope
        // is summation-order independent (the argument already banked for cuBLAS
        // and for faer's own threaded reduction). Bounds stay valid enclosures;
        // last ULPs may move.
        let c = mat_mul_f64_row_major(a, m, k, b, n);
        let mut out = allocate_gemm_output(shape, 0.0f64, SITE)?;
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = c[(i, j)];
            }
        }
        Ok(out)
    }
}

/// Env kill switch for [`install_cpu_gemm_engine_if_absent`]. Exact `0`
/// restores the historical "no process-global f32 engine unless wgpu or CUDA
/// installed one" behaviour byte-for-byte.
const CPU_GEMM_ENGINE_ENV: &str = "NY_CPU_GEMM_ENGINE";

/// Install [`FaerCpuGemmEngine`] as the process-global fast-f32 engine when
/// nothing better has been registered (#cpu-gemm-engine).
///
/// WHY. Until this existed, the process-global f32 engine was installed ONLY by
/// the wgpu backend (`ny-gpu/src/backend.rs:440`) or by a CUDA build
/// (`ny-cli/src/main.rs:323`). On any CPU-only run — no GPU, no `--features
/// cuda`, or CUDA present but unusable — `fast_f32_gemm::is_installed()` stayed
/// false and every engine-gated fast path silently chose its scalar fallback.
/// `FaerCpuGemmEngine` already existed and was already audited for the RN-f32
/// contract, but had exactly ONE caller
/// (`beta_crown/engine/graph/input_split/shared_specs.rs:244`); the rest of the
/// tree could not see it. The gap was reachability, not capability: NY owned a
/// blocked SIMD CPU GEMM and still ran scalar loops on most hosts.
///
/// This matters most where an engine gates a REPRESENTATION choice rather than
/// just speed — the conv Patches backward's batched-GEMM seam requires
/// `engine.is_some() || fast_f32_gemm::is_installed()`, so on a CPU-only host it
/// was unreachable by construction and composition fell to a per-position
/// scalar `conv2d_transpose` loop.
///
/// ORDERING. First installation wins (the factory is `OnceLock`-guarded), so
/// call this AFTER any accelerator has had its chance to register. The CPU
/// engine is a floor, never a preemption.
///
/// SOUNDNESS. `FaerCpuGemmEngine` satisfies the `gemm_f32` contract (plain IEEE
/// round-to-nearest f32, any summation order) — see its own doc comment. Every
/// verdict-feeding caller certifies engine results with a summation-order
/// INDEPENDENT envelope, which is the same argument that licenses cuBLAS on
/// this seam. Installing it can change a bound's last ULP, never its validity.
pub fn install_cpu_gemm_engine_if_absent() {
    if matches!(
        std::env::var(CPU_GEMM_ENGINE_ENV).ok().as_deref(),
        Some("0")
    ) {
        return;
    }
    if crate::fast_f32_gemm::is_installed() {
        return;
    }
    crate::fast_f32_gemm::set_fast_f32_gemm_engine(std::sync::Arc::new(FaerCpuGemmEngine));
}

/// Env kill switch for [`install_cpu_sound_f64_gemm_engine_if_absent`]. Exact
/// `0` restores the historical "no process-global f64 engine unless CUDA
/// installed one" behaviour.
const CPU_SOUND_F64_ENGINE_ENV: &str = "NY_CPU_SOUND_F64_ENGINE";

/// Install [`FaerCpuGemmEngine`] as the process-global **sound f64** GEMM engine
/// when nothing better has been registered (#cpu-sound-f64-engine).
///
/// WHY, and it is NOT the obvious reason. Until this existed, `sound_f64_gemm`
/// was populated ONLY by a CUDA build (cuBLAS `Dgemm`). On every CPU-only host
/// `gemm_f64` returned `Err`, and `aw_f64_with_abssum_unbounded`
/// (`layers/linear/crown_single.rs:149`) took its fall-through path — which
/// computes the abs-sum `S` with a **second full f64 GEMM**. With any engine
/// installed the same function instead routes through `aw_via_engine`, which
/// builds `S` from the **cheap f32 seam** (`crown_single.rs:703`).
///
/// So the win is a *saved second f64 GEMM*, not a faster first one. That
/// distinction was measured, not assumed
/// (`docs/ACCELERATE_SEAM_REFUTED_2026-08-07.md`, `tests/audit_attribution.rs`):
/// a three-arm experiment — engine absent / faer engine / Accelerate engine —
/// isolated it as **FAER/OFF = 1.024x**, while the vendor BLAS on top measured
/// **ACC/FAER = 0.955x**, i.e. negative. The whole reproducible gain belongs to
/// *having an engine at all*, and faer captures it with no FFI, no `unsafe` and
/// no vendor dependency.
///
/// ORDERING. First installation wins (`OnceLock`-guarded factory), so call this
/// AFTER any accelerator has had its chance to register. This is a floor, never
/// a preemption — a CUDA build still gets cuBLAS.
///
/// SOUNDNESS. Identical to the f32 installer's argument and to the one already
/// banked for cuBLAS on this exact seam: the `γ_n·S` certified-error bound is
/// summation-order INDEPENDENT, so any conventional inner product satisfies the
/// envelope the certificate needs. Measured on this seam: published bounds are
/// bit-identical to the engine-absent arm (`tests/audit_attribution.rs` reports
/// `bounds: FAER==OFF true` on all six configurations), and enclosure was
/// verified against an exact-rational (`BigRational`) oracle with 0 violations.
/// Installing it can move a bound's last ULP, never its validity.
pub fn install_cpu_sound_f64_gemm_engine_if_absent() {
    if matches!(
        std::env::var(CPU_SOUND_F64_ENGINE_ENV).ok().as_deref(),
        Some("0")
    ) {
        return;
    }
    if crate::sound_f64_gemm::is_installed() {
        return;
    }
    crate::sound_f64_gemm::set_sound_f64_gemm_engine(std::sync::Arc::new(FaerCpuGemmEngine));
}

pub(crate) fn current_par() -> Par {
    if is_inside_rayon_task() {
        // #tll-nested-collect-par: inside a Rayon worker we normally force
        // `Par::Seq` to avoid the #4392 operator deadlock and cross-domain
        // oversubscription. But when a `NestedFaerParGuard` is active (the
        // per-domain input-split CROWN collection), permit the configured
        // global parallelism: faer's work-stealing scope self-balances (fills
        // idle cores in the shallow BaB phase, stays effectively sequential
        // when the batch already saturates the machine).
        if nested_faer_par_active() {
            faer::get_global_parallelism()
        } else {
            Par::Seq
        }
    } else {
        faer::get_global_parallelism()
    }
}

fn is_inside_rayon_task() -> bool {
    rayon::current_thread_index().is_some() || RAYON_TASK_DEPTH.with(|depth| depth.get() > 0)
}

/// RAII guard for work launched from Rayon closures that can reach faer matmul.
///
/// Besides forcing faer to `Par::Seq` (see [`current_par`]), this guard also
/// disables the L2/Cauchy–Schwarz tightening lever for its scope. These guards
/// are constructed inside the CROWN / beta-CROWN rayon worker closures, where the
/// driver thread's [`crate::l2_lever_gate::L2LeverGuard`] is NOT inherited (a
/// fresh worker reads the gate's default = ON). Disabling here keeps the lever
/// inert for any IBP forward pass that runs on a CROWN-spawned worker, matching
/// the driver-thread behavior. Always sound (the lever only ever tightens).
#[must_use = "hold the guard for the full Rayon task scope"]
pub(crate) struct RayonTaskGuard {
    _l2: crate::l2_lever_gate::L2LeverGuard,
}

impl RayonTaskGuard {
    pub(crate) fn new() -> Self {
        RAYON_TASK_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Self {
            _l2: crate::l2_lever_gate::L2LeverGuard::disabled(),
        }
    }
}

impl Drop for RayonTaskGuard {
    fn drop(&mut self) {
        // `_l2` (the L2 lever guard) is dropped first, restoring the lever, then
        // we decrement the faer rayon depth.
        RAYON_TASK_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[cfg(test)]
mod tests {
    use faer::{mat, Par};
    use ny_core::GemmEngine;
    use rayon::prelude::*;

    use super::{current_par, mat_mul, mat_mul_f64_row_major, FaerCpuGemmEngine, RayonTaskGuard};

    // -----------------------------------------------------------------------
    // #fl-f32-cpu-seam (CONVWALL_PANEL_VERDICT_2026-08-01, Lane A step 2):
    // `FaerCpuGemmEngine` is the registry floor that lights the forward-linear
    // f32 value-GEMM seam (`NY_FORWARD_LINEAR_F32`) on CPU/Metal hosts. These
    // tests pin the engine's RN-f32 contract and its narrow refusal surface.
    // -----------------------------------------------------------------------

    /// Deterministic wide-magnitude mixed-sign stream (mirror of the
    /// `tests_image.rs` seam stream): exponents span [-15, 14] with random
    /// signs, so cancellation pushes the f32 accumulation error toward the
    /// Higham `γ_k^f32` worst case instead of averaging it away.
    fn seam_stream(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = ((s >> 40) % 30) as i32 - 15;
            let mant = ((s >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32;
            let sign = if (s >> 3) & 1 == 1 { -1.0 } else { 1.0 };
            sign * (1.0 + mant) * 2f32.powi(e)
        }
    }

    /// Engine correctness against an exact-widened f64 reference: every entry
    /// of `gemm_f32` must sit inside the order-independent f32 ULP envelope
    /// `|fl(A·B) − A·B| ≤ γ_k^f32 · S`, `S = Σ_k |a|·|b|`, `γ_k = k·u/(1−k·u)`,
    /// `u = 2⁻²⁴`. This is the exact certificate every verdict-feeding
    /// registry consumer charges (the FL seam charges the wider `γ_{K+4}`), so
    /// passing it here is what licenses installing this engine process-wide.
    /// Shapes include the census contractions k=27, 1152, 2304.
    #[test]
    fn faer_f32_engine_matches_f64_reference_within_f32_ulp_envelope() {
        const U: f64 = 5.960_464_477_539_063e-8; // 2^-24 exactly
        let mut next = seam_stream(0xF1_F32);
        for &(m, k, n) in &[
            (3usize, 27usize, 5usize),
            (4, 256, 7),
            (2, 1152, 9),
            (3, 2304, 4),
        ] {
            let a32: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let b32: Vec<f32> = (0..k * n).map(|_| next()).collect();
            let r32 = FaerCpuGemmEngine
                .gemm_f32(m, k, n, &a32, &b32)
                .expect("engine gemm_f32");
            assert_eq!(r32.len(), m * n);
            let ku = (k as f64) * U;
            let gamma = ku / (1.0 - ku);
            for i in 0..m {
                for j in 0..n {
                    let mut dot = 0.0f64; // exact-widened f64 reference
                    let mut s = 0.0f64; // S = Σ|a||b| (nonnegative)
                    for kk in 0..k {
                        let av = f64::from(a32[i * k + kk]);
                        let bv = f64::from(b32[kk * n + j]);
                        dot += av * bv;
                        s += av.abs() * bv.abs();
                    }
                    let err = (f64::from(r32[i * n + j]) - dot).abs();
                    let bound = gamma * s * (1.0 + 1e-6); // slack for the f64 ref's own rounding
                    assert!(
                        err <= bound,
                        "gemm_f32 outside the f32 ULP envelope: err={err} > γ_k·S={bound} \
                         (m={m}, k={k}, n={n}, i={i}, j={j})"
                    );
                }
            }
        }
    }

    /// #cpu-sound-f64-deadline: the bounded f64 GEMM must (a) agree EXACTLY with
    /// the unbounded one — tiling the OUTPUT axes cannot change any coefficient,
    /// because each stays one whole length-k dot — (b) refuse when k alone
    /// exceeds the cap rather than silently splitting the contraction, and (c)
    /// return DeadlineExceeded rather than a late result.
    #[test]
    fn bounded_f64_gemm_matches_unbounded_and_honours_its_contract() {
        use ny_core::{GemmEngine, NyError};
        let far = std::time::Instant::now() + std::time::Duration::from_mins(10);

        // (a) Exact agreement across cap sizes that force 1, several, and all
        // tiles — including caps that force COLUMN tiling as well as row tiling.
        for (m, k, n) in [(1usize, 1usize, 1usize), (4, 3, 5), (7, 8, 6), (16, 4, 9)] {
            let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.5 - 3.0).collect();
            let b: Vec<f64> = (0..k * n).map(|i| 1.0 / ((i + 1) as f64)).collect();
            let reference = FaerCpuGemmEngine
                .gemm_f64(m, k, n, &a, &b)
                .expect("unbounded reference");
            for cap in [k, k + 1, k * 2, k * n, k * n * m, usize::MAX / 4] {
                let got = FaerCpuGemmEngine
                    .gemm_f64_with_deadline(m, k, n, &a, &b, far, cap)
                    .unwrap_or_else(|e| panic!("({m},{k},{n}) cap={cap}: {e}"));
                assert_eq!(got.len(), reference.len(), "({m},{k},{n}) cap={cap}: shape");
                for (idx, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        r.to_bits(),
                        "({m},{k},{n}) cap={cap} idx={idx}: tiling the OUTPUT axes \
                         must be bit-identical to the untiled product"
                    );
                }
            }
        }

        // (b) k indivisible: a cap below k must REFUSE, not split the contraction.
        let a = vec![1.0f64; 2 * 9];
        let b = vec![1.0f64; 9 * 2];
        assert!(
            matches!(
                FaerCpuGemmEngine.gemm_f64_with_deadline(2, 9, 2, &a, &b, far, 8),
                Err(NyError::UnsupportedOp(_))
            ),
            "cap < k must refuse; splitting k would invalidate gamma_k*S"
        );

        // (c) An already-expired deadline yields DeadlineExceeded, never a value.
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("1s before now is inside the process uptime");
        assert!(
            matches!(
                FaerCpuGemmEngine.gemm_f64_with_deadline(2, 9, 2, &a, &b, past, 1 << 20),
                Err(NyError::DeadlineExceeded(_))
            ),
            "an expired deadline must never publish a result"
        );

        // Zero-extent still respects the deadline contract and returns empty.
        assert_eq!(
            FaerCpuGemmEngine
                .gemm_f64_with_deadline(0, 4, 4, &[], &[0.0; 16], far, 1 << 20)
                .expect("zero-extent is legal")
                .len(),
            0
        );
    }

    /// #cpu-sound-f64-engine: the f64 floor installs, is idempotent, and — the
    /// load-bearing part — is a FLOOR, never a preemption. Codex review flagged
    /// that nothing pinned either property.
    ///
    /// The ordering claim is structural, not timing-dependent: `is_installed()`
    /// tests FACTORY registration (`sound_f64_gemm.rs:105`), not engine
    /// materialization, so a CUDA build's lazily-registered cuBLAS factory is
    /// already visible here and this installer returns without touching it. This
    /// test pins the mechanism that makes that true — a pre-registered factory is
    /// not displaced.
    #[test]
    fn cpu_sound_f64_floor_installs_idempotently_and_never_preempts() {
        // WHY THIS ASSERTS SO LITTLE — two failed attempts are recorded here so
        // the next person does not repeat them.
        //
        // The sound-f64 factory is a process-global `OnceLock`, and the test
        // binary runs tests in PARALLEL. So:
        //   attempt 1 registered a sentinel factory and asserted the sentinel was
        //     the one materialized. Passed alone, FAILED in-suite: whichever test
        //     registers first wins and `set_sound_f64_gemm_factory` is silently a
        //     no-op for everyone after, so the assertion tested test ORDERING.
        //   attempt 2 branched on `is_installed()` and, when false, forced
        //     materialization and asserted the engine computed. Also FAILED
        //     in-suite: the check and the installation are not atomic, so a
        //     concurrent test can win the race in between and materialize ITS
        //     factory — which may legitimately yield `None`.
        //
        // What is left is what is actually order-independent and actually true:
        // the installer never panics, and after it runs SOME factory is
        // installed and stays installed. The never-preempt property is
        // guaranteed by construction rather than by this test —
        // `set_sound_f64_gemm_factory` is `OnceLock::set`, which cannot
        // overwrite an existing value — and the floor engine's own arithmetic is
        // covered directly by `bounded_f64_gemm_matches_unbounded_and_honours_its_contract`
        // and `faer_engine_*`, neither of which touches global state.
        super::install_cpu_sound_f64_gemm_engine_if_absent();
        assert!(
            crate::sound_f64_gemm::is_installed(),
            "after the floor runs, some factory must be installed"
        );
        super::install_cpu_sound_f64_gemm_engine_if_absent();
        assert!(
            crate::sound_f64_gemm::is_installed(),
            "the installer must be idempotent and must never remove a factory"
        );
    }

    /// Narrow-engine refusal surface: accelerator-only methods must fail with
    /// TYPED errors (`NyError::UnsupportedOp`), never panic or silently
    /// succeed, and the optional-capability probes must stay `None` — so no
    /// registry consumer can mistake the CPU floor for a fused-conv accelerator
    /// (the `GradientSteeringDevice` / `AttackSteeringDevice` narrow-engine
    /// precedent).
    ///
    /// SCOPE CHANGE (2026-08-09, #cpu-sound-f64-deadline). This test previously
    /// also asserted that `gemm_f64_with_deadline` refuses. It no longer does,
    /// and that is DELIBERATE: the CPU floor now implements the bounded-dispatch
    /// contract (tiles the output axes only, never the contraction `k`; polls
    /// before every tile; returns `DeadlineExceeded` rather than a late result),
    /// so it is a legitimate deadline-bounded engine. Its contract is pinned by
    /// `bounded_f64_gemm_matches_unbounded_and_honours_its_contract`, including
    /// the refusal that DOES remain — a cap smaller than `k`, which cannot be
    /// satisfied without splitting the contraction.
    ///
    /// Why the change was necessary rather than cosmetic: while this method
    /// returned `UnsupportedOp`, and because the trait default is contractually
    /// forbidden from delegating to the unbounded `gemm_f64`, EVERY
    /// timeout-bearing run bypassed the sound f64 engine entirely — 82 dispatches
    /// with no deadline set, 0 with one. Every VNN-COMP run carries a deadline.
    #[test]
    fn faer_engine_refuses_accelerator_only_methods_with_typed_errors() {
        use ny_core::{ConvTranspose2dParams, NyError};

        let deadline = std::time::Instant::now() + std::time::Duration::from_mins(1);
        assert!(
            FaerCpuGemmEngine
                .gemm_f64_with_deadline(1, 1, 1, &[1.0], &[1.0], deadline, 1)
                .is_ok(),
            "the CPU floor is now a deadline-bounded engine by design (#cpu-sound-f64-deadline)"
        );
        let params = ConvTranspose2dParams {
            num_specs: 1,
            out_channels: 1,
            in_channels: 1,
            out_h: 1,
            out_w: 1,
            in_h: 1,
            in_w: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        assert!(
            matches!(
                FaerCpuGemmEngine.conv_transpose_2d(&[1.0], &[1.0], &params),
                Err(NyError::UnsupportedOp(_))
            ),
            "fused conv_transpose_2d must refuse with a typed UnsupportedOp"
        );
        assert!(FaerCpuGemmEngine.as_gpu_crown_backward().is_none());
        assert!(FaerCpuGemmEngine.as_gpu_ibp_forward().is_none());
        assert!(FaerCpuGemmEngine.as_gpu_ibp_forward_ext().is_none());
        assert!(FaerCpuGemmEngine.as_gpu_dag_ibp_forward_ext().is_none());
        assert_eq!(FaerCpuGemmEngine.backend_provenance(), "faer-cpu");
    }

    /// The non-CUDA startup registration (`ny-cli/src/main.rs` calls this
    /// exact function after every accelerator has had its chance) installs the
    /// registry floor and stays idempotent. Process-global `OnceLock`: this
    /// asserts installation, not identity — the fresh-process FL seam
    /// integration test (`tests/fl_f32_cpu_engine_seam.rs`) covers the rest.
    #[test]
    fn install_cpu_gemm_engine_if_absent_installs_and_is_idempotent() {
        super::install_cpu_gemm_engine_if_absent();
        assert!(crate::fast_f32_gemm::is_installed());
        super::install_cpu_gemm_engine_if_absent();
        assert!(crate::fast_f32_gemm::is_installed());
    }

    #[test]
    fn faer_gemm_rejects_dimension_overflow_without_panicking() {
        let huge = 1usize << (usize::BITS - 1);
        assert!(FaerCpuGemmEngine
            .gemm_f32(huge, huge, huge, &[], &[])
            .is_err());
        assert!(FaerCpuGemmEngine
            .gemm_f64(huge, huge, huge, &[], &[])
            .is_err());
    }

    #[test]
    fn faer_gemm_preserves_empty_contraction_shape_without_entering_faer() {
        assert_eq!(
            FaerCpuGemmEngine
                .gemm_f32(usize::MAX, 0, 0, &[], &[])
                .expect("empty GEMM"),
            Vec::<f32>::new()
        );
        assert_eq!(
            FaerCpuGemmEngine
                .gemm_f64(usize::MAX, 0, 0, &[], &[])
                .expect("empty GEMM"),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn test_mat_mul_matches_operator_4392() {
        let lhs = mat![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let rhs = mat![[7.0_f32, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let expected = &lhs * &rhs;
        let actual = mat_mul(&lhs, &rhs);

        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                assert!(
                    (expected[(row, col)] - actual[(row, col)]).abs() <= 1e-6,
                    "mismatch at ({row}, {col}): expected {}, got {}",
                    expected[(row, col)],
                    actual[(row, col)]
                );
            }
        }
    }

    #[test]
    fn test_f64_row_major_views_match_scalar_product() {
        let lhs = [1.0_f64, -2.0, 3.0, 4.0, 5.0, -6.0];
        let rhs = [7.0_f64, 8.0, -9.0, 10.0, 11.0, -12.0];
        let actual = mat_mul_f64_row_major(&lhs, 2, 3, &rhs, 2);
        for i in 0..2 {
            for j in 0..2 {
                let expected = (0..3).map(|t| lhs[i * 3 + t] * rhs[t * 2 + j]).sum::<f64>();
                assert_eq!(actual[(i, j)], expected, "mismatch at ({i}, {j})");
            }
        }
    }

    #[test]
    fn test_rayon_task_guard_is_nestable_4392() {
        let before = matches!(current_par(), Par::Seq);

        {
            let _guard = RayonTaskGuard::new();
            assert!(matches!(current_par(), Par::Seq));

            {
                let _nested = RayonTaskGuard::new();
                assert!(matches!(current_par(), Par::Seq));
            }

            assert!(matches!(current_par(), Par::Seq));
        }

        assert_eq!(matches!(current_par(), Par::Seq), before);
    }

    #[test]
    fn test_current_par_is_seq_inside_rayon_worker_4392() {
        let all_seq = (0..8usize)
            .into_par_iter()
            .map(|_| matches!(current_par(), Par::Seq))
            .reduce(|| true, |acc, is_seq| acc && is_seq);

        assert!(
            all_seq,
            "current_par must force Par::Seq inside Rayon workers"
        );
    }

    // #tll-nested-collect-par: with a `NestedFaerParGuard` active on a Rayon
    // worker, `current_par` must permit the global (Rayon) parallelism instead
    // of `Par::Seq`, and nesting faer's `Par::Rayon` matmul inside an outer
    // `par_iter` worker must COMPLETE (no #4392-style deadlock) with correct
    // results. Runs many outer domains × wide inner GEMMs to exercise
    // work-stealing under contention.
    #[test]
    fn test_nested_faer_par_guard_no_deadlock_and_correct() {
        use super::{mat_mul_f64, NestedFaerParGuard};
        use faer::Mat;

        let results: Vec<f64> = (0..64usize)
            .into_par_iter()
            .map(|seed| {
                let _g = NestedFaerParGuard::new();
                // Inside a Rayon worker with the guard, current_par must NOT be Seq
                // (global parallelism is Rayon by default in this build).
                assert!(
                    !matches!(current_par(), Par::Seq),
                    "guard must lift Par::Seq inside a Rayon worker"
                );
                // A moderately wide f64 GEMM routed through the nested-parallel
                // path. If faer's nested Rayon scope deadlocked, this hangs.
                let m = 96;
                let k = 128;
                let n = 96;
                let a = Mat::<f64>::from_fn(m, k, |i, j| ((i + j + seed) % 7) as f64);
                let b = Mat::<f64>::from_fn(k, n, |i, j| ((i * j + seed) % 5) as f64);
                let c = mat_mul_f64(&a, &b);
                c[(0, 0)]
            })
            .collect();
        // After the parallel section, the guard is dropped: depth back to 0.
        assert_eq!(results.len(), 64);
        // Cross-check one product against a scalar reference (seed 0).
        let m = 96;
        let k = 128;
        let n = 96;
        let a = Mat::<f64>::from_fn(m, k, |i, j| ((i + j) % 7) as f64);
        let b = Mat::<f64>::from_fn(k, n, |i, j| ((i * j) % 5) as f64);
        let mut expect = 0.0f64;
        for kk in 0..k {
            expect += a[(0, kk)] * b[(kk, 0)];
        }
        assert!(
            (results[0] - expect).abs() < 1e-9,
            "nested-parallel GEMM result {} != reference {}",
            results[0],
            expect
        );
    }
}
