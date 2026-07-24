// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;

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

impl ny_core::GemmEngine for FaerCpuGemmEngine {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "FaerCpuGemmEngine::gemm_f32: a.len()={} (want {m}*{k}) b.len()={} (want {k}*{n})",
                a.len(),
                b.len(),
            )));
        }
        let a_mat = Mat::<f32>::from_fn(m, k, |i, j| a[i * k + j]);
        let b_mat = Mat::<f32>::from_fn(k, n, |i, j| b[i * n + j]);
        let c = mat_mul(&a_mat, &b_mat);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = c[(i, j)];
            }
        }
        Ok(out)
    }

    fn gemm_f64(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
    ) -> ny_core::Result<Vec<f64>> {
        if a.len() != m * k || b.len() != k * n {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "FaerCpuGemmEngine::gemm_f64: a.len()={} (want {m}*{k}) b.len()={} (want {k}*{n})",
                a.len(),
                b.len(),
            )));
        }
        let a_mat = Mat::<f64>::from_fn(m, k, |i, j| a[i * k + j]);
        let b_mat = Mat::<f64>::from_fn(k, n, |i, j| b[i * n + j]);
        let c = mat_mul_f64(&a_mat, &b_mat);
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = c[(i, j)];
            }
        }
        Ok(out)
    }
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
    use rayon::prelude::*;

    use super::{current_par, mat_mul, mat_mul_f64_row_major, RayonTaskGuard};

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
