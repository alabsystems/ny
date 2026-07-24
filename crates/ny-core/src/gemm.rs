// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{NyError, Result};

// Trust contract attribute. Under tRustc contract verification (`--cfg trust_verify`)
// `#[ensures]` is the first-class builtin from `core::contracts`; under stable rustc
// it is the no-op NY-owned `trust` compatibility crate. Mirrors ny-cert's dual
// import so the same `#[ensures(...)]` source verifies under trustc and compiles
// unchanged under rustc.
#[cfg(trust_verify)]
use core::contracts::{ensures, requires};
#[cfg(not(trust_verify))]
use trust::{ensures, requires};

#[path = "gemm_gpu_dag_ibp.rs"]
mod gpu_dag_ibp;
#[path = "gemm_gpu_ibp.rs"]
mod gpu_ibp;

pub use gpu_dag_ibp::{
    GpuDagIbpForwardExt, GpuDagIbpModelPlan, GpuDagIbpOp, GpuDagIbpPlanDesc, NETWORK_INPUT_IDX,
};
pub use gpu_ibp::{GpuIbpForward, GpuIbpForwardExt, GpuIbpLayer, GpuIbpModelPlan, GpuIbpResult};

/// Conservative fallback bound for NaN/Inf sanitization in bound propagation.
///
/// When interval arithmetic produces non-finite endpoints (NaN or Inf), callers
/// may repair those endpoints with `±FALLBACK_BOUND` while preserving finite
/// endpoints as-is. This avoids silently narrowing valid finite intervals.
///
/// Used by both CPU and GPU paths. GPU WGSL shaders embed this as a literal;
/// the contract test `test_fallback_bound_consistent` verifies the values match.
pub const FALLBACK_BOUND: f32 = 1e10;

/// Maximum absolute value for CROWN backward A-matrix coefficients (#1932).
///
/// When |A[i,j]| exceeds this threshold after a backward propagation step,
/// the entire row is degraded to zero coefficients with ±inf bias — the same
/// sound treatment as actual Inf overflow (#2681), but triggered proactively
/// before coefficients reach f32::INFINITY.
///
/// Without this, coefficients growing via A_new = A @ W can silently reach
/// magnitudes near f32::MAX (~3.4e38) where subsequent multiplications produce
/// Inf or NaN. The #2681 handler only catches actual Inf, missing the "near
/// overflow" regime where a coefficient like 1e35 * 1e5 = 1e40 > f32::MAX.
///
/// Set to match FALLBACK_BOUND for consistency with IBP overflow repair.
/// Reference: alpha-beta-CROWN does no coefficient clamping (relies on float64
/// dynamic range). Our f32 path needs proactive protection.
pub const CROWN_COEFF_MAX: f32 = 1e10;

/// Check whether a CROWN A-matrix coefficient is within safe bounds.
///
/// Returns `true` if the value is finite and its absolute value does not
/// exceed [`CROWN_COEFF_MAX`]. Used by CROWN backward paths to detect
/// near-overflow coefficients before they cascade (#1932).
#[inline]
#[must_use]
pub fn is_crown_coeff_safe(value: f32) -> bool {
    value.is_finite() && value.abs() <= CROWN_COEFF_MAX
}

/// f64 variant of [`is_crown_coeff_safe`] for normalization CROWN backward
/// paths that accumulate coefficients in double precision (#3228).
///
/// Uses the same [`CROWN_COEFF_MAX`] threshold (cast to f64). Values exceeding
/// this would overflow when converted to f32 bounds downstream.
#[inline]
#[must_use]
pub fn is_crown_coeff_safe_f64(value: f64) -> bool {
    value.is_finite() && value.abs() <= f64::from(CROWN_COEFF_MAX)
}

/// Parameters for fused GPU conv_transpose_2d (GEMM + col2im).
///
/// Describes a single-group Conv2d backward: the caller loops over groups and
/// passes per-group slices. All spatial dimensions refer to the forward conv's
/// input/output (backward reverses the direction).
///
/// Reference: designs/2026-03-15-issue-3813-fused-gpu-conv2d-backward.md
#[derive(Debug, Clone, Copy)]
pub struct ConvTranspose2dParams {
    /// Number of specification/objective rows (S).
    pub num_specs: usize,
    /// Output channels per group (OC) — the forward conv's out_channels / groups.
    pub out_channels: usize,
    /// Input channels per group (IC) — the forward conv's in_channels / groups.
    pub in_channels: usize,
    /// Grad spatial height (OH) — forward conv's output height.
    pub out_h: usize,
    /// Grad spatial width (OW) — forward conv's output width.
    pub out_w: usize,
    /// Input spatial height (IH) — forward conv's input height.
    pub in_h: usize,
    /// Input spatial width (IW) — forward conv's input width.
    pub in_w: usize,
    /// Kernel height (KH).
    pub kernel_h: usize,
    /// Kernel width (KW).
    pub kernel_w: usize,
    /// Stride height (SH).
    pub stride_h: usize,
    /// Stride width (SW).
    pub stride_w: usize,
    /// Padding height (PH).
    pub pad_h: usize,
    /// Padding width (PW).
    pub pad_w: usize,
}

/// Minimal GEMM interface for accelerating CROWN/α-CROWN linear backprop.
///
/// Computes `C = A @ B` for f32 row-major matrices:
/// - `A`: shape (m, k)
/// - `B`: shape (k, n)
/// - `C`: shape (m, n)
///
/// Implementations may run on CPU, GPU, or remote accelerators. Callers must be
/// prepared to fall back to a local implementation if this returns an error.
///
/// The trait requires `Sync + Send` to allow use in rayon parallel contexts
/// (e.g., parallel domain processing in BaB).
pub trait GemmEngine: Sync + Send {
    /// Compute `C = A @ B` for row-major f32 matrices.
    ///
    /// `a` has shape (m, k), `b` has shape (k, n). Returns `C` as a flat
    /// row-major `Vec<f32>` of length `m * n`.
    ///
    /// PRECISION CONTRACT: plain IEEE round-to-nearest f32 arithmetic (any
    /// summation order). Verdict-feeding callers certify results with
    /// order-independent error bounds that assume exactly this.
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    /// Compute `C = A @ B` for row-major f32 matrices with NO precision
    /// contract beyond "approximately f32": implementations MAY use
    /// reduced-precision tensor-core paths (TF32 / BF16-split accumulation).
    ///
    /// ONLY for soundness-free consumers — adversarial attack / counterexample
    /// search (candidates are re-checked concretely) and heuristic scoring.
    /// NEVER for verdict-feeding bound arithmetic: the certified error bounds
    /// on those paths assume IEEE RN-f32 (`gemm_f32`).
    ///
    /// Default: falls back to the exact [`GemmEngine::gemm_f32`].
    fn gemm_f32_fast(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        self.gemm_f32(m, k, n, a, b)
    }

    /// Compute `C = A @ B` for row-major f64 matrices.
    ///
    /// Used by the f64 propagation path (`double_fp: true`) for VNN-COMP
    /// soundnessbench/sat_relu. GPU implementations may return `Err` since
    /// f64 GPU performance is poor on consumer hardware and these benchmarks
    /// use small networks.
    ///
    /// Default implementation returns `Err(Unsupported)`.
    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        let _ = (m, k, n, a, b);
        Err(NyError::UnsupportedOp(
            "f64 GEMM not supported by this engine".into(),
        ))
    }

    /// Compute two independent, same-shape IEEE-f64 products that share one
    /// immutable right-hand operand.
    ///
    /// ConvTranspose CROWN recomputes lower and upper coefficient matrices with
    /// the same exact-widened kernel. An accelerator may retain that RHS, queue
    /// both products on one ordered stream, and synchronize once. Each result
    /// retains the ordinary [`gemm_f64`](Self::gemm_f64) precision contract;
    /// pairing changes scheduling only, never the algebra or allowed arithmetic.
    ///
    /// The default deliberately performs two ordinary calls in lower/upper
    /// order. Engines without a transactional override therefore retain their
    /// prior allocation, failure, synchronization, and numerical behavior.
    fn gemm_f64_pair_shared_rhs(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: [&[f64]; 2],
        b: &[f64],
    ) -> Result<[Vec<f64>; 2]> {
        Ok([
            self.gemm_f64(m, k, n, a[0], b)?,
            self.gemm_f64(m, k, n, a[1], b)?,
        ])
    }

    /// Compute three independent, same-shape IEEE-f64 matrix products.
    ///
    /// This scheduling seam exists for sound CROWN's `(center, magnitude,
    /// propagated_error)` products. The products are algebraically independent:
    /// an accelerator may queue all three on one ordered stream and synchronize
    /// once, but it must preserve the exact [`gemm_f64`](Self::gemm_f64)
    /// precision contract for every member.
    ///
    /// The default deliberately performs three ordinary calls in array order.
    /// Engines without a transactional override therefore retain their prior
    /// allocation, failure, synchronization, and numerical behavior.
    fn gemm_f64_triplet(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: [&[f64]; 3],
        b: [&[f64]; 3],
    ) -> Result<[Vec<f64>; 3]> {
        Ok([
            self.gemm_f64(m, k, n, a[0], b[0])?,
            self.gemm_f64(m, k, n, a[1], b[1])?,
            self.gemm_f64(m, k, n, a[2], b[2])?,
        ])
    }

    /// Fused conv_transpose_2d: GEMM + col2im in a single dispatch.
    ///
    /// Computes the Conv2d CROWN backward for one group:
    ///   1. GEMM: `(S*OH*OW, OC) × (OC, IC*KH*KW)` → `(S*OH*OW, IC*KH*KW)`
    ///   2. col2im: scatter GEMM output → `(S, IC*IH*IW)` using stride/padding
    ///
    /// `a_reshaped` is `(S*OH*OW, OC)` row-major — already extracted per-group.
    /// `weight_col` is `(OC, IC*KH*KW)` row-major.
    /// Returns `(S, IC*IH*IW)` row-major (flat length = S * IC * IH * IW).
    ///
    /// GPU implementations fuse both steps into GPU-resident passes with no host
    /// roundtrip between GEMM and col2im — eliminating the CPU col2im bottleneck.
    ///
    /// Default: returns `Err(Unsupported)`. Callers fall back to `gemm_f32` +
    /// CPU col2im when this method is not available.
    ///
    /// Reference: designs/2026-03-15-issue-3813-fused-gpu-conv2d-backward.md
    /// Part of #3813.
    fn conv_transpose_2d(
        &self,
        _a_reshaped: &[f32],
        _weight_col: &[f32],
        _params: &ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        Err(NyError::UnsupportedOp(
            "conv_transpose_2d not supported by this engine".into(),
        ))
    }

    /// Fused conv_transpose_2d for a `(lower_a, upper_a)` pair sharing one weight.
    ///
    /// Both A-matrices share the *same* weight column `Arc<[f32]>` (the Conv2d
    /// kernel reshaped per group), so a GPU engine can keep that weight matrix
    /// **resident** across the two calls and reuse its device buffers/plan,
    /// keyed by the weight `Arc`'s pointer identity. It may also stack the two
    /// inputs into a single dispatch (`2*S` rows). Both are pure-performance
    /// optimizations: the result is bit-identical (modulo f32 GEMM reassociation,
    /// which is unaffected here since the reduction axis `OC` is unchanged) to
    /// calling [`conv_transpose_2d`](Self::conv_transpose_2d) twice.
    ///
    /// Returns `(lower_result, upper_result)`, each `(S, IC*IH*IW)` row-major.
    ///
    /// Default: forwards to two [`conv_transpose_2d`](Self::conv_transpose_2d)
    /// calls (no residency), so engines that do not override this keep their
    /// existing behavior exactly. Part of the conv_transpose plan-cache work.
    fn conv_transpose_2d_pair_cached(
        &self,
        a_lower: &[f32],
        a_upper: &[f32],
        weight_col: &Arc<[f32]>,
        params: &ConvTranspose2dParams,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let lower = self.conv_transpose_2d(a_lower, weight_col, params)?;
        let upper = self.conv_transpose_2d(a_upper, weight_col, params)?;
        Ok((lower, upper))
    }

    /// Optional GPU CROWN backward accelerator (#3397).
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        None
    }

    /// Optional GPU-resident IBP forward accelerator (#4081).
    fn as_gpu_ibp_forward(&self) -> Option<&dyn GpuIbpForward> {
        None
    }

    /// Optional cached GPU-resident IBP planner (#4268).
    fn as_gpu_ibp_forward_ext(&self) -> Option<&dyn GpuIbpForwardExt> {
        None
    }

    /// Optional cached graph-DAG GPU-resident IBP planner (#4276, #4318).
    fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn GpuDagIbpForwardExt> {
        None
    }

    /// Sound interval matrix product: a guaranteed enclosure of every real
    /// product `A @ B` with `A ∈ [a_lo, a_hi]` and `B ∈ [b_lo, b_hi]`
    /// (elementwise). Returns `(c_lo, c_hi)`, each row-major `(m, n)`, such that
    /// `c_lo ≤ A@B ≤ c_hi` elementwise for ALL such `A`, `B` — accounting for
    /// every floating-point rounding error introduced along the way.
    ///
    /// This is the sound building block for running CROWN coefficient
    /// propagation on a GPU. GPU shading languages (WGSL/MSL) expose only f32
    /// round-to-nearest with no directed-rounding modes, so the usual
    /// "accumulate in f64, round the final cast outward" trick is unavailable on
    /// device. Instead this uses **Rump's midpoint–radius interval matmul**: the
    /// result midpoint and three nonnegative radius products are evaluated with
    /// ordinary round-to-nearest `gemm_f32` (hence on whatever backend this
    /// engine provides — GPU for a device engine), then a closed-form bound on
    /// the f32 dot-product rounding error,
    ///   `γ_k = k·u / (1 − k·u)`,  `u = 2⁻²⁴`  (f32 unit roundoff),
    /// is added and the `± radius` is committed to f32 with **outward** directed
    /// rounding done here on the host in f64. The enclosure is valid under any
    /// IEEE-754 round-to-nearest GEMM, which is exactly what GPUs guarantee.
    ///
    /// Default implementation is backend-agnostic (built on [`gemm_f32`]); GPU
    /// engines need not override it to benefit. Soundness is independent of the
    /// reduction order `gemm_f32` happens to use.
    ///
    /// [`gemm_f32`]: Self::gemm_f32
    fn gemm_interval_sound(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a_lo: &[f32],
        a_hi: &[f32],
        b_lo: &[f32],
        b_hi: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if a_lo.len() != m * k || a_hi.len() != m * k {
            return Err(NyError::shape_mismatch(vec![m, k], vec![a_lo.len()]));
        }
        if b_lo.len() != k * n || b_hi.len() != k * n {
            return Err(NyError::shape_mismatch(vec![k, n], vec![b_lo.len()]));
        }
        if m == 0 || k == 0 || n == 0 {
            return Ok((vec![], vec![]));
        }

        // f32 unit roundoff (2^-24) and the running-error growth factor γ_k for a
        // length-k dot product: |fl(x·y) − x·y| ≤ γ_k · (|x|·|y|)  +  additive
        // underflow.  The γ_k term is the *relative* (normalized) error model;
        // the smallest positive f32 subnormal η = 2⁻¹⁴⁹ bounds the *additive*
        // per-op error in the subnormal range (where products flush toward 0 and
        // the relative model alone is unsound). Every dot of length k performs
        // ≤ 2k roundings, each ≤ η/2, so an additive term `c·k·η` covers all
        // underflow in p, abs_p, r1, r2, r3 together (c = 8 leaves margin).
        const U: f64 = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24 exactly
        const ETA: f64 = f64::from_bits(0x36A0_0000_0000_0000); // 2^-149 (min f32 subnormal)
        let ku = (k as f64) * U;
        // k·u < 1 for any realistic k (k·u = 1 at k = 2^24 ≈ 16.7M); guard anyway.
        let gamma_k = if ku < 0.5 { ku / (1.0 - ku) } else { 2.0 * ku };
        let underflow_add = 8.0 * (k as f64) * ETA;

        // Midpoint (signed, nearest-f32) and radius (nonnegative, rounded OUTWARD
        // to f32). The radius is taken DIRECTLY as max(ma − l, h − ma) so that
        // `[ma − rad, ma + rad] ⊇ [l, h]` holds by construction for ALL f32
        // inputs — including widely-separated exponents where `(l+h)*0.5` and
        // `(h−l)*0.5` are not exact in f64 (the previous trad+offset form could
        // under-cover there). `f32→f64` is exact; the subtractions can lose ≤ 1
        // f64 ulp, covered by a relative bump, and `+ ETA` keeps the radius
        // strictly positive so a point interval still carries the floor needed
        // for the matmul's subnormal rounding.
        let build_mid_rad = |lo: &[f32], hi: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let len = lo.len();
            let mut mid = vec![0.0f32; len];
            let mut rad = vec![0.0f32; len];
            for i in 0..len {
                let l = f64::from(lo[i]);
                let h = f64::from(hi[i]);
                let m32 = f64::midpoint(l, h) as f32; // nearest; need not be the true midpoint
                let mf = f64::from(m32);
                // rad ≥ (mf − l) ⇒ ma − rad ≤ l;  rad ≥ (h − mf) ⇒ ma + rad ≥ h.
                let half = (mf - l).max(h - mf).max(0.0);
                mid[i] = m32;
                rad[i] = round_f32_up(half * (1.0 + 1e-12) + ETA);
            }
            (mid, rad)
        };
        let (ma, ra) = build_mid_rad(a_lo, a_hi);
        let (mb, rb) = build_mid_rad(b_lo, b_hi);
        let abs_ma: Vec<f32> = ma.iter().map(|v| v.abs()).collect();
        let abs_mb: Vec<f32> = mb.iter().map(|v| v.abs()).collect();

        // Round-to-nearest matmuls (run on this engine's backend).
        let p = self.gemm_f32(m, k, n, &ma, &mb)?; // signed midpoint product
        let abs_p = self.gemm_f32(m, k, n, &abs_ma, &abs_mb)?; // ≥ 0, for γ_k bound
        let r1 = self.gemm_f32(m, k, n, &abs_ma, &rb)?; // |ma|·rb
        let r2 = self.gemm_f32(m, k, n, &ra, &abs_mb)?; // ra·|mb|
        let r3 = self.gemm_f32(m, k, n, &ra, &rb)?; // ra·rb

        // For A = ma+δa (|δa| ≤ ra) and B = mb+δb (|δb| ≤ rb):
        //   |A·B − ma·mb|       ≤ r1 + r2 + r3        (interval spread, reals)
        //   |ma·mb − fl(ma·mb)| ≤ γ_k · |ma|·|mb|     (f32 dot rounding, normalized)
        // Each fl(·) matmul under-reports the real value by at most a (1−γ_k)
        // factor, so multiply by (1+2γ_k) to recover a real upper bound, add the
        // additive underflow term, then a hair of f64 slack for the host
        // combination itself. If any matmul overflowed to ±inf (so the f32
        // product left representable range), the only sound bound is the trivial
        // [−∞, +∞]; downstream this triggers the usual IBP fallback.
        let real_factor = 1.0 + 2.0 * gamma_k;
        let host_slack = 1.0 + 1e-10;
        let mut c_lo = vec![0.0f32; m * n];
        let mut c_hi = vec![0.0f32; m * n];
        for i in 0..(m * n) {
            let spread = f64::from(r1[i]) + f64::from(r2[i]) + f64::from(r3[i]);
            let round_err = gamma_k * f64::from(abs_p[i]);
            let radius = (spread + round_err) * real_factor * host_slack + underflow_add;
            let pi = f64::from(p[i]);
            if !pi.is_finite() || !radius.is_finite() {
                c_lo[i] = f32::NEG_INFINITY;
                c_hi[i] = f32::INFINITY;
                continue;
            }
            c_lo[i] = round_f32_down(pi - radius);
            c_hi[i] = round_f32_up(pi + radius);
        }
        Ok((c_lo, c_hi))
    }

    /// Sound coefficient-error propagation for ONE linear CROWN-backward step.
    ///
    /// Given a coefficient matrix `a` (`m×k`) with a nonnegative incoming error
    /// bound `a_err` (`m×k`, so the exact coefficient lies in
    /// `[a − a_err, a + a_err]`) and a weight `w` (`k×n`), returns
    /// `(a_new, a_err_new)` where `a_new = fl(a @ w)` and `a_err_new` bounds
    /// `|a_new − a_exact@w|` for EVERY `a_exact ∈ [a − a_err, a + a_err]`:
    ///   `a_err_new = round_up( γ_k·(|a|@|w|) + (a_err @ |w|) ) + additive`.
    /// The `γ_k·(|a|@|w|)` term bounds the f32 GEMM's own rounding; `a_err@|w|`
    /// propagates the incoming coefficient error; `additive = 8·k·η` covers
    /// subnormal underflow.
    ///
    /// This is the on-device mirror of the CPU `crown_single` `γ_n·S` certified
    /// error — the per-layer core of a sound GPU-resident CROWN backward
    /// (task #15). Built on `gemm_f32` (three products on the same backend), so a
    /// GPU engine runs it on device. Soundness is independent of reduction order.
    fn crown_aw_error_step(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        a_err: &[f32],
        w: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if a.len() != m * k || a_err.len() != m * k {
            return Err(NyError::shape_mismatch(vec![m, k], vec![a.len()]));
        }
        if w.len() != k * n {
            return Err(NyError::shape_mismatch(vec![k, n], vec![w.len()]));
        }
        if m == 0 || k == 0 || n == 0 {
            return Ok((vec![], vec![]));
        }
        const U: f64 = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24
        let ku = (k as f64) * U;
        let gamma_k = if ku < 0.5 { ku / (1.0 - ku) } else { 2.0 * ku };
        // Base (weight-INDEPENDENT) FTZ floor: ≥ 8k·2^-126, a NORMAL f32 that
        // survives Metal flush-to-zero and covers subnormal *result* flushes. This
        // replaces the prior `8k·2^-149` (ETA) floor, which under-counted FTZ result
        // loss (up to 2^-126, not 2^-149) on flush-to-zero hardware.
        let base_additive = f64::from(ftz_safe_underflow_floor(
            u32::try_from(k).unwrap_or(u32::MAX),
        ));

        let abs_a: Vec<f32> = a.iter().map(|v| v.abs()).collect();
        let abs_w: Vec<f32> = w.iter().map(|v| v.abs()).collect();
        let a_new = self.gemm_f32(m, k, n, a, w)?;
        let s = self.gemm_f32(m, k, n, &abs_a, &abs_w)?; // |a| @ |w|  (≥ 0)
        let prop = self.gemm_f32(m, k, n, a_err, &abs_w)?; // a_err @ |w|  (≥ 0)

        // Weight-AMPLIFIED FTZ floor (#gpu-metal-daz; this fn's `# Scope` note +
        // docs/SOUND_GPU_IBP_PLAN.md §0). `A·W` is a weight-amplified reduction, so a
        // subnormal operand DAZ-zeroed by Metal before the multiply loses up to
        // `|other|·FLT_MIN` — which `base_additive` alone cannot cover. Per output
        // (i,j) the exact worst case is `Σ_l max(|a_il|,|w_lj|)·FLT_MIN ≤
        // flushacc[i,j]·FLT_MIN`, `flushacc[i,j] = 1 + Σ_l max(|a_il|,|w_lj|,1)`. We
        // use the separable over-bound `flushacc[i,j] ≤ 1 + k + ‖a_i‖₁ + ‖w_j‖₁`
        // (`max(x,y,1) ≤ x+y+1`), computed in O(mk+kn) not O(mnk). Mirrors the
        // already-sound IBP MatMul shader (crates/ny-gpu shaders.rs `flushacc`).
        let row_abs_a: Vec<f64> = (0..m)
            .map(|i| (0..k).map(|c| f64::from(a[i * k + c].abs())).sum())
            .collect();
        let col_abs_w: Vec<f64> = (0..n)
            .map(|j| (0..k).map(|c| f64::from(w[c * n + j].abs())).sum())
            .collect();

        let host_slack = 1.0 + 1e-10;
        let mut a_err_new = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                let flushacc = 1.0 + (k as f64) + row_abs_a[i] + col_abs_w[j];
                let flush = amplified_ftz_floor(base_additive, flushacc, host_slack);
                // round the error UP so [a_new − err, a_new + err] never under-covers.
                let e = (gamma_k * f64::from(s[idx]) + f64::from(prop[idx])) * host_slack + flush;
                a_err_new[idx] = round_f32_up(e);
            }
        }
        Ok((a_new, a_err_new))
    }
}

/// Sound activation-backward coefficient + error propagation (elementwise),
/// increment 3 of the sound GPU-resident CROWN backward (task #15).
///
/// For each `(output_row, neuron)` the relaxation composes the incoming
/// coefficient with the per-neuron slope, sign-routed exactly as the CPU path:
///   lower bound: `a ≥ 0 → a·lower_slope`,  `a < 0 → a·upper_slope`
///   upper bound: `a ≥ 0 → a·upper_slope`,  `a < 0 → a·lower_slope`
/// and the certified coefficient error becomes
///   `new_err = round_up( in_err·(|lower_slope| + |upper_slope|) + gap ) + additive`,
/// where `gap = |a·slope − fl(a·slope)|` is the f32 multiply rounding and the
/// `slope_sum` factor covers a possible sign-flip of `a` under its error
/// selecting the OTHER envelope slope. Mirrors `crown_dense.rs` (validated there
/// at 0/6M trials); this is the engine-independent form the GPU activation shader
/// will inline. Returns `(new_lower_a, new_upper_a, new_lower_err, new_upper_err)`,
/// each `num_outputs × num_neurons` row-major.
#[allow(clippy::too_many_arguments)]
pub fn crown_activation_error_step(
    num_outputs: usize,
    num_neurons: usize,
    lower_a: &[f32],
    upper_a: &[f32],
    lower_a_err: &[f32],
    upper_a_err: &[f32],
    lower_slope: &[f32],
    upper_slope: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = num_outputs * num_neurons;
    if lower_a.len() != n || upper_a.len() != n || lower_a_err.len() != n || upper_a_err.len() != n
    {
        return Err(NyError::shape_mismatch(
            vec![num_outputs, num_neurons],
            vec![lower_a.len()],
        ));
    }
    if lower_slope.len() != num_neurons || upper_slope.len() != num_neurons {
        return Err(NyError::shape_mismatch(
            vec![num_neurons],
            vec![lower_slope.len()],
        ));
    }
    const ETA: f64 = f64::from_bits(0x36A0_0000_0000_0000); // 2^-149
    let additive = 8.0 * ETA;

    let mut new_lower_a = vec![0.0f32; n];
    let mut new_upper_a = vec![0.0f32; n];
    let mut new_lower_err = vec![0.0f32; n];
    let mut new_upper_err = vec![0.0f32; n];
    for j in 0..num_outputs {
        for i in 0..num_neurons {
            let idx = j * num_neurons + i;
            let ls = lower_slope[i];
            let us = upper_slope[i];
            let slope_sum = (f64::from(ls).abs()) + (f64::from(us).abs());

            let la = lower_a[idx];
            let lsel = if la >= 0.0 { ls } else { us };
            let lcoeff = la * lsel;
            new_lower_a[idx] = lcoeff;
            let lgap = (f64::from(la) * f64::from(lsel) - f64::from(lcoeff)).abs();
            new_lower_err[idx] =
                round_f32_up(f64::from(lower_a_err[idx]) * slope_sum + lgap + additive);

            let ua = upper_a[idx];
            let usel = if ua >= 0.0 { us } else { ls };
            let ucoeff = ua * usel;
            new_upper_a[idx] = ucoeff;
            let ugap = (f64::from(ua) * f64::from(usel) - f64::from(ucoeff)).abs();
            new_upper_err[idx] =
                round_f32_up(f64::from(upper_a_err[idx]) * slope_sum + ugap + additive);
        }
    }
    Ok((new_lower_a, new_upper_a, new_lower_err, new_upper_err))
}

/// Round an `f64` value DOWN to the nearest `f32` (toward −∞).
///
/// Soundness helper for [`GemmEngine::gemm_interval_sound`]: a lower bound must
/// never round up. Bit-manipulation `next_*` so it is correct at the MSRV
/// (predating stable `f32::next_down`). A *finite* `x` above `f32::MAX` clamps
/// to `f32::MAX` (the largest finite f32 ≤ x), NOT `+∞` — returning `+∞` for a
/// lower bound would be unsound.
fn round_f32_down(x: f64) -> f32 {
    if x.is_nan() {
        return f32::NEG_INFINITY; // most conservative lower bound
    }
    let near = x as f32; // nearest, may be ±∞ if x is out of f32 range
    if near == f32::INFINITY {
        // x is finite and > f32::MAX (or x == +∞): largest f32 ≤ x is f32::MAX.
        return if x.is_finite() {
            f32::MAX
        } else {
            f32::INFINITY
        };
    }
    if near == f32::NEG_INFINITY {
        return f32::NEG_INFINITY; // no finite f32 ≤ x; −∞ is the sound floor
    }
    if f64::from(near) <= x {
        near
    } else {
        next_down_f32(near)
    }
}

/// Round an `f64` value UP to the nearest `f32` (toward +∞).
///
/// A *finite* `x` below `−f32::MAX` clamps to `−f32::MAX`, NOT `−∞`.
fn round_f32_up(x: f64) -> f32 {
    if x.is_nan() {
        return f32::INFINITY; // most conservative upper bound
    }
    let near = x as f32;
    if near == f32::NEG_INFINITY {
        // x is finite and < −f32::MAX (or x == −∞): smallest f32 ≥ x is −f32::MAX.
        return if x.is_finite() {
            f32::MIN
        } else {
            f32::NEG_INFINITY
        };
    }
    if near == f32::INFINITY {
        return f32::INFINITY;
    }
    if f64::from(near) >= x {
        near
    } else {
        next_up_f32(near)
    }
}

/// IEEE-754 successor of a finite `f32` (toward +∞).
fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1); // smallest positive subnormal
    }
    let bits = x.to_bits();
    f32::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

/// FTZ-safe additive underflow floor for a sound f32 error term evaluated on a GPU
/// that may **flush subnormals to zero** (Metal MSL defaults to flush-to-zero;
/// Vulkan keeps IEEE subnormals). The returned additive is added to a certified
/// error value before an OUTWARD round; for the bound to stay sound the additive
/// must:
///   1. **survive flush-to-zero** — be a NORMAL f32 (`>= f32::MIN_POSITIVE = 2^-126`),
///      not a subnormal the GPU silently zeroes; AND
///   2. **upper-bound the error FTZ can lose** — at most `flush_points · FLT_MIN`,
///      since each of the (`<= flush_points`) subnormal-producing operations can
///      drop a value `< FLT_MIN` to `0`.
///
/// `8 · flush_points · FLT_MIN` (clamped `>= FLT_MIN`) satisfies both. Since
/// `FLT_MIN (2^-126) >= ETA (2^-149)`, it also DOMINATES the prior subnormal
/// `8·ETA`-based floor used by the wgpu resident CROWN, so it stays a valid
/// over-bound on non-FTZ hardware (Vulkan) as well — sound on BOTH backends. The
/// `8` is the small safety factor the resident CROWN already used.
///
/// # Scope: weight-INDEPENDENT floor (sufficient only for coefficient-≤1 paths)
/// This bounds the flush loss as `<= flush_points · FLT_MIN`, which is correct for
/// ELEMENTWISE / activation floors (the transform coefficient is `<= 1`). It is
/// NOT sufficient for a WEIGHT-AMPLIFIED reduction (`fl(W·x)` where a subnormal `x`
/// flushed to 0 by Metal FTZ is then scaled by a large `|W|`): there the loss is up
/// to `|W|·FLT_MIN`, which can exceed any weight-independent floor. Reduction paths
/// (Linear/Conv/MatMul, and the CROWN backward's `add_b` abs-sum) need the on-device
/// amplified floor `flushacc·slack·F32_MIN_NORMAL` derived in
/// `docs/SOUND_GPU_IBP_PLAN.md` §0. This function is the correct base term (the `+
/// ftz_safe_underflow_floor(k)` addend) of that amplified floor.
///
/// Smallest positive NORMAL `f32` as an IEEE-754 bit pattern (`f32::MIN_POSITIVE`,
/// i.e. `2^-126`). A positive `f32` is normal (survives flush-to-zero) iff its bits
/// are `>= this && < 0x7F80_0000` (infinity).
const F32_MIN_NORMAL_BITS: u32 = 0x0080_0000;

/// The IEEE-754 **bit pattern** of an FTZ-safe underflow floor over-bounding a
/// reduction of `flush_points` f32 terms. Returns a NORMAL f32's bits, encoded as
/// `FLT_MIN` scaled up by `2^exp_steps` where `2^exp_steps >= 8·flush_points`, so
/// the value is `>= 8·flush_points·FLT_MIN >= flush_points·FLT_MIN` (the max FTZ
/// flush loss) and `>= FLT_MIN` (normal ⇒ FTZ-safe).
///
/// # Soundness contract (Trust)
/// The FTZ-survival lemma is stated as a machine-checkable `#[ensures]` on the u32
/// BIT PATTERN (integer, the solver's supported domain — f32 arithmetic and
/// `to_bits` are not): `exp_steps` is clamped to `<= 200`, so the exponent add
/// neither overflows u32 nor reaches the infinity exponent, and the result is
/// `>= F32_MIN_NORMAL_BITS` because `exp_steps << 23 >= 0`. tRustc attempts this
/// obligation on every `targo trust` build; the current native solver returns
/// `unknown` for the dynamic shift (like most of ny-core's numeric obligations —
/// the toolchain's shift/intrinsic support is still maturing), NOT a disproof. The
/// property is proven by construction (above) and pinned by the unit tests; L0
/// memory/overflow safety of this function IS trustc-verified. When the solver
/// gains dynamic-shift support the L1 obligation discharges with no code change.
#[ensures(|r: &u32| *r >= F32_MIN_NORMAL_BITS)]
#[must_use]
fn ftz_safe_underflow_floor_bits(flush_points: u32) -> u32 {
    let fp = flush_points.max(1);
    // exp_steps >= ceil(log2(8·fp)): the u64 bit-length of 8·fp. `2^exp_steps` then
    // strictly exceeds 8·fp, so FLT_MIN·2^exp_steps > 8·fp·FLT_MIN. Clamp to 200 so
    // FLT_MIN's exponent (1) + exp_steps stays < 254 (finite normal) and the u32
    // add cannot overflow. (`leading_zeros` may be opaque to the solver, but the
    // `.min(200)` bound is all the contract needs.)
    let eight_fp = u64::from(fp).saturating_mul(8);
    let exp_steps = (64 - eight_fp.leading_zeros()).min(200);
    F32_MIN_NORMAL_BITS + (exp_steps << 23)
}

/// FTZ-safe additive underflow floor as an `f32` (thin wrapper over
/// [`ftz_safe_underflow_floor_bits`], whose bit-pattern contract is Trust-verified).
/// The returned value is a positive NORMAL `f32` (`>= f32::MIN_POSITIVE`), so it
/// survives Metal's flush-to-zero, and over-bounds `flush_points · FLT_MIN`.
#[must_use]
pub fn ftz_safe_underflow_floor(flush_points: u32) -> f32 {
    f32::from_bits(ftz_safe_underflow_floor_bits(flush_points))
}

/// Weight-AMPLIFIED FTZ operand-flush floor for one weight-amplified reduction
/// output entry (Linear/Conv/MatMul `fl(A·W)`), the term the weight-independent
/// [`ftz_safe_underflow_floor`] `base` cannot supply (see its `# Scope` note and
/// `docs/SOUND_GPU_IBP_PLAN.md` §0).
///
/// A subnormal operand `|a| ∈ [2^-149, 2^-126)` that Metal DAZ-zeroes *before* the
/// multiply loses the whole product `|a|·|w|` (up to `|w|·FLT_MIN`), so per output
/// `(i,j)` the worst-case operand-flush loss is `Σ_l max(|a_il|,|w_lj|)·FLT_MIN`.
/// The caller passes `flushacc ≥ 1 + Σ_l max(|a_il|,|w_lj|,1)` (an over-count), so
/// `flushacc·slack·FLT_MIN` over-bounds that loss and the certified error stays
/// OUTWARD. `FLT_MIN = f32::MIN_POSITIVE = 2^-126` keeps every added quantum a
/// NORMAL f32 (it survives flush-to-zero itself).
///
/// # Soundness contract (Trust)
/// The result is never below `base` (the added term is nonnegative), so composing
/// this with the already-verified base floor can only *widen* the error — it can
/// never tighten a bound into a false `Verified`. The full enclosure property
/// (`[a_new − err, a_new + err] ⊇ a_exact·w` under DAZ) is pinned by the
/// zero-tolerance exact-rational oracle test
/// `gemm_tests::crown_aw_error_step_daz_operand_flush_stays_outward`.
#[requires(slack >= 0.0)]
#[ensures(|f: &f64| *f >= base)]
#[must_use]
fn amplified_ftz_floor(base: f64, flushacc: f64, slack: f64) -> f64 {
    // `.max(0.0)` on both factors makes the added term unconditionally ≥ 0, so the
    // `#[ensures(*f >= base)]` obligation holds for every input (defensive; callers
    // always pass flushacc, slack ≥ 0).
    base + flushacc.max(0.0) * slack.max(0.0) * f64::from(f32::MIN_POSITIVE)
}

/// IEEE-754 predecessor of a finite `f32` (toward −∞).
fn next_down_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f32::from_bits(1); // smallest negative subnormal
    }
    let bits = x.to_bits();
    f32::from_bits(if x > 0.0 { bits - 1 } else { bits + 1 })
}

/// Per-layer data for GPU-accelerated CROWN backward pass.
///
/// Describes one layer in the backward propagation sequence. Linear layers
/// contribute weight matrix multiplication; activation layers contribute
/// element-wise slope/intercept relaxation.
///
/// Reference: designs/2026-03-06-gpu-crown-backward.md
#[derive(Clone)]
pub enum GpuCrownLayer {
    /// Linear: A_new = A @ weight, bias_new += A_old @ layer_bias
    Linear {
        /// Weight matrix (out_features × in_features) row-major.
        /// Uses `Arc<[f32]>` so static weights are shared across CROWN calls
        /// without per-call cloning (#3397 plan cache Step 1).
        weight: Arc<[f32]>,
        /// Layer bias (out_features,), None if no bias
        bias: Option<Arc<[f32]>>,
        out_features: usize,
        in_features: usize,
    },
    /// Activation: element-wise relaxation with per-neuron slopes/intercepts.
    ///
    /// Positive A coefficients use lower_slope (for lower bound) / upper_slope (for upper).
    /// Negative A coefficients use upper_slope (for lower bound) / lower_slope (for upper).
    /// Reference: compose.rs compose_lower/compose_upper
    ///
    /// Activation data remains `Vec<f32>` because slopes/intercepts are dynamic —
    /// they depend on the current pre-activation bounds (which change per BaB split).
    Activation {
        lower_slope: Vec<f32>,
        upper_slope: Vec<f32>,
        lower_intercept: Vec<f32>,
        upper_intercept: Vec<f32>,
        num_neurons: usize,
    },
    /// Conv2d: transposed convolution backward for CROWN.
    ///
    /// The backward pass computes A_new = conv_transpose(A, W), decomposed into:
    /// 1. Reshape A from (S, OC*OH*OW) to (S*OH*OW, OC)
    /// 2. GEMM: (S*OH*OW, OC) × (OC, IC*KH*KW) → (S*OH*OW, IC*KH*KW)
    /// 3. col2im gather: (S*OH*OW, IC*KH*KW) → (S, IC*IH*IW)
    ///
    /// Reference: alpha-beta-CROWN auto_LiRPA/operators/convolution.py:bound_backward
    /// Reference: designs/2026-03-06-conv-crown-backward-gemm.md
    Conv2d {
        /// Kernel reshaped to W_col: (out_c, in_c * kh * kw) row-major.
        /// Uses `Arc<[f32]>` for zero-copy sharing (#3397).
        weight_col: Arc<[f32]>,
        /// Optional per-channel bias expanded to (out_c * oh * ow)
        bias_expanded: Option<Arc<[f32]>>,
        out_channels: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        /// Output spatial dimensions of the conv layer (grad_h, grad_w for backward)
        out_h: usize,
        out_w: usize,
        /// Input spatial dimensions (result spatial after backward)
        in_h: usize,
        in_w: usize,
    },
    /// ReLU dual-alpha activation: exact per-neuron alpha_lower/alpha_upper parity (#4313).
    ///
    /// Unlike `Activation` (which uses symmetric 2-slope lower/upper semantics),
    /// this variant routes three independent affine branches based on coefficient
    /// sign, matching the CPU/reference dual-alpha rule exactly:
    ///
    /// - lower bound, a >= 0: `a * lower_pos_slope` (alpha_lower, through origin)
    /// - lower bound, a < 0:  `a * cross_slope`, bias += `a * cross_intercept`
    /// - upper bound, a >= 0: `a * cross_slope`, bias += `a * cross_intercept`
    /// - upper bound, a < 0:  `a * upper_neg_slope` (alpha_upper, through origin)
    ///
    /// Packed layout: `[lower_pos_slope | cross_slope | upper_neg_slope | cross_intercept]`,
    /// same 4 × num_neurons footprint as `Activation`.
    ///
    /// Reference: auto_LiRPA/operators/relu.py:641-652 (alpha_lower/alpha_upper)
    ActivationReluDualAlpha {
        /// Optimized lower-bound slope for positive A coefficients (alpha_lower).
        lower_pos_slope: Vec<f32>,
        /// Chord slope u/(u-l), shared by lower-neg and upper-pos paths.
        cross_slope: Vec<f32>,
        /// Optimized upper-bound slope for negative A coefficients (alpha_upper).
        upper_neg_slope: Vec<f32>,
        /// Chord intercept -l*u/(u-l), shared by lower-neg and upper-pos paths.
        cross_intercept: Vec<f32>,
        num_neurons: usize,
    },
    /// MaxPool2d: sparse winner routing or IBP fallback for CROWN backward.
    ///
    /// For each output position, extraction computes one of:
    /// - a definite winner input flat-index when `lower(winner) >= max upper(other)`
    /// - `u32::MAX` to signal IBP fallback using the precomputed window bounds
    ///
    /// The GPU kernel zeroes the destination A-matrix, scatters routed
    /// coefficients into their unique input position, and accumulates IBP
    /// fallback bias contributions per spec row.
    ///
    /// Reference: alpha-beta-CROWN auto_LiRPA/operators/pooling.py:78-337
    MaxPool2d {
        /// Per output-position winner input flat-index, or `u32::MAX` for IBP fallback.
        routing: Vec<u32>,
        /// Per output-position lower fallback bound (`max(lower(window))`).
        ibp_lower: Vec<f32>,
        /// Per output-position upper fallback bound (`max(upper(window))`).
        ibp_upper: Vec<f32>,
        /// Flattened input dimension (channels * in_h * in_w, or batch * channels * in_h * in_w).
        input_dim: usize,
        /// Flattened output dimension (channels * out_h * out_w, or batch * channels * out_h * out_w).
        output_dim: usize,
    },
}

/// A residual/skip-connected network decomposed into **backward-order** segments
/// for the sound GPU-resident CROWN backward (the cifar100/tinyimagenet win path).
///
/// Each segment carries owned layer sub-chains (also in backward order,
/// output→input). The backend folds the coefficient frontier through each segment
/// in order, forking at residual blocks and merging the skip stream soundly,
/// carrying the certified rounding error ACROSS block boundaries so stacked blocks
/// compose without dropping error. This mirrors the in-tree `ResnetSegment`
/// (`ny-gpu`) but lives at the trait boundary so `ny-propagate` can build it from a
/// graph decomposition without depending on `ny-gpu` internals.
///
/// Soundness contract for the *decomposition* (the caller's responsibility): a
/// `Residual`/`ResidualProj` is valid only when the merge is an exact element-wise
/// `Add` and both branches are pure functions of the block input `z` (i.e. every
/// branch node's only data dependency traces back to `z`). Then `out = F(z) + z`
/// (resp. `F(z) + P(z)`) holds exactly, and independently relaxing each branch and
/// summing is always a sound over-approximation.
#[derive(Clone)]
pub enum GpuResnetSegment {
    /// A plain sequential sub-chain of layers (backward order, output→input).
    Chain(Vec<GpuCrownLayer>),
    /// An identity-skip residual block `out = F(z) + z`; the vec is `F`'s sub-chain
    /// (backward order), which must map the block dimension back to itself.
    Residual(Vec<GpuCrownLayer>),
    /// A projection residual block `out = F(z) + P(z)` (e.g. a 1×1-conv skip at a
    /// stage transition): `(F_branch, P_branch)`. Both branches map the block input
    /// dimension to the block output dimension; the backend computes
    /// `A_in = backward_F(A) + backward_P(A)` (with the incoming bias counted once).
    ResidualProj(Vec<GpuCrownLayer>, Vec<GpuCrownLayer>),
}

/// Seed state for a GPU CROWN backward suffix that starts mid-network.
///
/// Unlike `crown_backward_gpu(...)`, which starts from a fresh symmetric
/// specification matrix with zero bias, seeded backward begins from an existing
/// asymmetric linear relaxation:
///
/// - `lower_a`, `upper_a`: shape `(num_specs, current_dim)` row-major
/// - `lower_b`, `upper_b`: shape `(num_specs,)`
///
/// This lets graph constrained backward hand its live `LinearBounds` frontier
/// into the existing GPU CROWN suffix without re-running the already-computed
/// prefix on CPU. Part of #3813.
#[derive(Clone, Debug)]
pub struct GpuCrownSeed {
    pub lower_a: Arc<[f32]>,
    pub upper_a: Arc<[f32]>,
    pub lower_b: Arc<[f32]>,
    pub upper_b: Arc<[f32]>,
    pub num_specs: usize,
    pub current_dim: usize,
}

/// Result from GPU CROWN backward pass: concretized lower and upper bounds.
pub struct GpuCrownResult {
    /// Lower bounds per specification row
    pub lower_bounds: Vec<f32>,
    /// Upper bounds per specification row
    pub upper_bounds: Vec<f32>,
}

/// One BaB subdomain's per-domain operands for a BATCHED sound resnet CROWN
/// backward (#batched-bab). All domains in a batch share the SAME network — the
/// `segments`' Linear/Conv2d weights are the same `Arc<[f32]>` across domains
/// (`Arc::ptr_eq`), only the `Activation` relaxation slopes/intercepts differ —
/// so a batched backward runs one shared-weight GEMM over the stacked
/// `n_domains × num_specs` spec rows. Per-domain: the segments' Activation
/// relaxation, `beta_signed`, `frontier_abs`, `node_abs`, and the input box. The
/// spec seed is shared and passed once to the batched call.
pub struct GpuResnetBatchedDomainRef<'a> {
    /// This domain's segment list: shared weights + this domain's baked ReLU slopes.
    pub segments: &'a [GpuResnetSegment],
    /// This domain's input-box lower/upper (the final concretize box).
    pub input_lower: &'a [f32],
    pub input_upper: &'a [f32],
    /// Per-ReLU signed beta (β·sign) in fold order, one slice per `Activation`.
    pub beta_signed: &'a [Vec<f32>],
    /// Per-segment frontier (input-side) abs-max bounds for error concretization.
    pub frontier_abs: &'a [Vec<f32>],
    /// Per-ReLU pre-node abs-max bounds in fold order (finer error concretization).
    pub node_abs: &'a [Vec<f32>],
}

/// #clip-interm-resnet-batched: the DOWNLOADED input-relative coefficient frontier of a
/// BATCHED sound resnet CROWN backward, BEFORE the per-coefficient error has been folded
/// outward — the object the batched intermediate-domain clip needs (one seeded backward
/// for the WHOLE domain frontier, instead of a serial per-child backward).
///
/// All arrays are row-major over the final coefficient dim `dim` (= the network input dim
/// for an identity seed folded to `NETWORK_INPUT`). There are `num_specs = n_domains *
/// num_specs_per_dom` stacked rows; row `s` belongs to domain `s / num_specs_per_dom`.
/// Row `s` of domain `d` is the input-relative affine form of one seeded pre-activation
/// neuron: `lower_a[s]·x + (lower_b[s] − lower_b_err[s]) ≤ z(x) ≤ upper_a[s]·x +
/// (upper_b[s] + upper_b_err[s])`, MODULO the still-live per-coefficient certified error
/// `lower_err[s]`/`upper_err[s]`. Consumers MUST discharge that per-coefficient error
/// OUTWARD into the bias over their own input box before using a row as an enclosure
/// (a raw-coefficient enclosure is UNSOUND — dropping the certified error can yield a
/// too-tight bound → false UNSAT). Any row whose outward penalty is non-finite must be
/// refused (keep the inherited bound). Non-empty only on the explicit coeff-capture
/// batched entry; otherwise the arrays are empty.
pub struct GpuResidentCoeffBatched {
    /// Lower input-relative coefficients, `num_specs × dim` row-major.
    pub lower_a: Vec<f32>,
    /// Upper input-relative coefficients, `num_specs × dim` row-major.
    pub upper_a: Vec<f32>,
    /// Certified per-coefficient error on `lower_a`, `num_specs × dim` (folded via
    /// per-ReLU concretization on the capture pass; residual must be folded by the
    /// consumer).
    pub lower_err: Vec<f32>,
    /// Certified per-coefficient error on `upper_a`, `num_specs × dim`.
    pub upper_err: Vec<f32>,
    /// Lower bias center, `num_specs`.
    pub lower_b: Vec<f32>,
    /// Upper bias center, `num_specs`.
    pub upper_b: Vec<f32>,
    /// Certified lower bias error (subtract to widen the lower bound down), `num_specs`.
    pub lower_b_err: Vec<f32>,
    /// Certified upper bias error (add to widen the upper bound up), `num_specs`.
    pub upper_b_err: Vec<f32>,
    /// Final coefficient dim (the network input dim for an input-relative seed).
    pub dim: usize,
    /// Total stacked rows `= n_domains * num_specs_per_dom`.
    pub num_specs: usize,
    /// Per-domain spec-row count (domain `d` = rows `[d*num_specs_per_dom, ..)`).
    pub num_specs_per_dom: usize,
}

/// Result from a gradient-capturing GPU CROWN resnet backward: the sound concretized
/// bounds plus each unstable ReLU's analytic alpha gradient (one `Vec<f32>` per ReLU
/// in fold order). Gradients are NON-soundness-critical — they only steer alpha
/// (any alpha ∈ [0,1] is a sound relaxation) — so capturing them never affects the
/// verdict bound. Used by the GPU-resident warmup alpha optimization.
pub struct GpuCrownGradResult {
    /// Lower bounds per specification row (identical to the non-grad path).
    pub lower_bounds: Vec<f32>,
    /// Upper bounds per specification row.
    pub upper_bounds: Vec<f32>,
    /// Per-ReLU analytic alpha gradients (fold order). `relu_grads[r][i]` is neuron
    /// `i`'s gradient for ReLU `r`.
    pub relu_grads: Vec<Vec<f32>>,
}

/// Result from a beta-gradient-capturing GPU CROWN resnet backward: the sound
/// concretized bounds (with the β-CROWN split dual folded, identical to
/// `crown_backward_gpu_resnet_sound_beta`) plus, per requested ReLU, the LOWER
/// A-coefficient values at the requested (split) neuron columns — the analytic
/// β-gradient inputs. `beta_gather[r]` is row-major `num_specs × idx_r.len()`:
/// `beta_gather[r][s*n_idx + i] = A_lower[s, idx_r[i]]` captured at ReLU `r`'s
/// output (before the ReLU relaxation is applied), matching the CPU capture
/// point (`capture_constrained_relu_intermediate` → `a_at_relu`). The CPU
/// analytic rule then gives `∂lb_s/∂β_k = −sign_k · A_lower[s, k]` for the
/// critical spec row `s`. Gather values are NON-soundness-critical — they only
/// steer β, and any β ≥ 0 yields a valid Lagrangian-dual bound.
pub struct GpuCrownBetaGradResult {
    /// Lower bounds per specification row (identical to the beta path).
    pub lower_bounds: Vec<f32>,
    /// Upper bounds per specification row.
    pub upper_bounds: Vec<f32>,
    /// Per-ReLU gathered lower A-values (fold order, one entry per ReLU;
    /// empty `Vec` for ReLUs with an empty index list).
    pub beta_gather: Vec<Vec<f32>>,
}

/// Result from one trajectory-capturing wide sound resnet call.  The four
/// channels correspond to the SAME domain batch and relaxation/dual state:
/// verdict-safe concretized bounds, non-soundness-critical alpha gradients and
/// beta gathers, and the input-relative affine frontier. Keeping them together
/// avoids a second caller-visible backward just to recover coefficients.
pub struct GpuCrownTrajectoryResult {
    /// One sound concretized result per domain, in domain-major order.
    pub bounds: Vec<GpuCrownResult>,
    /// Per-ReLU analytic alpha gradients, domain-stacked within each ReLU.
    pub alpha_grads: Vec<Vec<f32>>,
    /// Per-ReLU gathered lower-A values, row-major over all domain/spec rows.
    pub beta_gather: Vec<Vec<f32>>,
    /// Input-relative coefficient frontier for all domain/spec rows.
    pub coeff: GpuResidentCoeffBatched,
}

/// GPU-accelerated CROWN backward pass that keeps A-matrices on device.
///
/// Unlike [`GemmEngine`] (per-operation upload/download), this trait keeps all
/// intermediate A-matrix state on GPU and only reads back the final concretized
/// bounds. This eliminates N-1 roundtrips for an N-layer network.
///
/// Reference: alpha-beta-CROWN keeps PyTorch tensors on GPU from the initial
/// C matrix through to concretization. Source: designs/2026-03-06-gpu-crown-backward.md
pub trait GpuCrownBackward: Sync + Send {
    /// Run complete CROWN backward pass on GPU.
    ///
    /// - `layers`: Layer descriptors in backward order (output-to-input)
    /// - `spec`: Initial specification matrix C (num_specs × output_dim) row-major
    /// - `num_specs`: Number of specification rows
    /// - `input_lower`: Input lower bounds for concretization
    /// - `input_upper`: Input upper bounds for concretization
    ///
    /// Returns concretized lower and upper bounds, one per spec row.
    fn crown_backward_gpu(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult>;

    /// Run GPU CROWN backward from an arbitrary asymmetric seed state.
    ///
    /// This is the graph-constrained counterpart to `crown_backward_gpu(...)`:
    /// instead of starting from a fresh identity/spec matrix, callers provide
    /// the current lower/upper A-matrices and bias terms for the live suffix.
    ///
    /// Default: unsupported. Engines may fall back to CPU suffix propagation.
    fn crown_backward_gpu_seeded(
        &self,
        _layers: &[GpuCrownLayer],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "seeded GPU CROWN backward not supported by this engine".into(),
        ))
    }

    /// SOUND GPU-resident CROWN backward: same contract as
    /// [`crown_backward_gpu`](Self::crown_backward_gpu), but every coefficient,
    /// its certified f32 rounding error, the bias, and the final concretization
    /// are carried with directed/over-bounded error so the returned bounds are a
    /// SOUND enclosure — usable to decide a verdict even under the soundness gate.
    ///
    /// The coefficient GEMMs/activation/conv stay GPU-resident across layers (only
    /// the final coefficients download once), so it is both sound AND fast.
    ///
    /// Default: unsupported, so non-sound engines fall back to the proven CPU path.
    fn crown_backward_gpu_sound(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "sound GPU CROWN backward not supported by this engine".into(),
        ))
    }

    /// Whether this engine provides a sound GPU-resident CROWN backward
    /// (`crown_backward_gpu_sound`). Lets callers route verdict-deciding bounds
    /// onto the sound GPU path under the soundness gate instead of the CPU
    /// fallback. Default `false`.
    fn provides_sound_gpu_crown(&self) -> bool {
        false
    }

    /// SOUND seeded GPU-resident CROWN backward: the soundness counterpart of
    /// [`crown_backward_gpu_seeded`](Self::crown_backward_gpu_seeded), used by the
    /// graph alpha-CROWN suffix path. The frontier coefficient/bias in `seed` is
    /// treated as exact (matching the CPU sound suffix path, which carries no
    /// coefficient-error frontier) and only the suffix's own f32 rounding is
    /// tracked with directed/over-bounded error — so the returned bounds are a
    /// sound enclosure, decided GPU-resident.
    ///
    /// Default: unsupported, so non-sound engines fall back to the CPU path.
    fn crown_backward_gpu_seeded_sound(
        &self,
        _layers: &[GpuCrownLayer],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "seeded sound GPU CROWN backward not supported by this engine".into(),
        ))
    }

    /// SOUND seeded GPU-resident CROWN backward over a RESNET decomposed into
    /// backward-order [`GpuResnetSegment`]s (plain chains + identity/projection
    /// residual blocks). Same soundness contract as
    /// [`crown_backward_gpu_seeded_sound`](Self::crown_backward_gpu_seeded_sound):
    /// the `seed` frontier coefficient/bias is treated as exact and only the
    /// suffix's own f32 rounding is over-bounded, so the returned bounds are a sound
    /// enclosure usable to decide a verdict under the soundness gate. The certified
    /// error is carried ACROSS segment/residual-block boundaries so stacked blocks
    /// compose soundly.
    ///
    /// This is the resnet counterpart of the unary-chain seeded sound backward: it
    /// lets the verdict-deciding alpha-CROWN suffix on cifar100/tinyimagenet ResNets
    /// stay GPU-resident (no host coefficient round-trip) instead of bailing to the
    /// slow CPU dense path on the residual `Add` nodes.
    ///
    /// `frontier_abs` is the per-segment frontier-node abs-max bounds (`max(|l|,|u|)` per
    /// dim, SAME order as `segments`). Gated on `NY_RESNET_ERR_CONCRETIZE=1`, the backend
    /// uses it to concretize the accumulated coefficient error into the (non-amplifying)
    /// bias error at each segment boundary — capping the #unsat-keystone L1 error blow-up
    /// on the MAIN bound, mirroring what `_grad`/`_beta` already do. Empty (or gate off) ⇒
    /// byte-identical to the pre-concretization path.
    ///
    /// `node_abs` is the per-ReLU PRE-activation abs-max bounds (`max(|pre_l|,|pre_u|)` per
    /// dim) in FOLD order (each branch's `Activation`s output→input, F before P) — the
    /// finer per-ReLU error-concretization frontier. It drives the AUTO-FALLBACK: when the
    /// un-concretized MAIN bound explodes (non-finite or astronomically wide), the backend
    /// re-runs with the per-ReLU fine concretization (strictly ≥ as tight as the per-segment
    /// fold) and returns the element-wise intersection of the sound results. Empty ⇒ the
    /// fallback degrades to the per-segment `frontier_abs` path (or, with both empty, the
    /// pre-concretization path) — so the verdict default for non-exploding nets is unchanged.
    ///
    /// Default: unsupported, so non-sound engines fall back to the proven CPU path.
    fn crown_backward_gpu_resnet_sound(
        &self,
        _segments: &[GpuResnetSegment],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "resnet sound GPU CROWN backward not supported by this engine".into(),
        ))
    }

    /// Gradient-capturing variant of [`crown_backward_gpu_resnet_sound`]: returns the
    /// SAME sound concretized bounds, plus each unstable ReLU's analytic alpha
    /// gradient captured from the on-device PRE-transform lower coefficient
    /// (`grad[i] = pre_lower[i]·Σ_j max(A_lower[j,i], 0)`). `relu_pre_lower` are the
    /// masked pre-activation lower bounds per ReLU in FOLD order (each branch's
    /// `Activation` layers in order, F-branch before P-branch for a projection block;
    /// 0 entries for stable neurons). This lets the cifar100/tinyimagenet resnet
    /// alpha-CROWN WARMUP optimize alpha GPU-resident instead of paying the per-
    /// iteration dense CPU coefficient round-trip that makes the warmup overrun the
    /// budget (BaB then never runs — measured: 0 domains at ≤400 s). Gradients are
    /// non-soundness-critical, so this can never affect a verdict.
    ///
    /// Default: unsupported (engines fall back to the CPU gradient path).
    fn crown_backward_gpu_resnet_sound_grad(
        &self,
        _segments: &[GpuResnetSegment],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _relu_pre_lower: &[Vec<f32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownGradResult> {
        Err(NyError::UnsupportedOp(
            "gradient-capturing resnet sound GPU CROWN backward not supported by this engine"
                .into(),
        ))
    }

    /// Beta-capable variant of [`crown_backward_gpu_resnet_sound`] (cifar100/tinyimagenet
    /// unsat keystone, step 4): returns the sound concretized bounds with the per-domain
    /// β-CROWN split-constraint Lagrangian dual folded into the POST-slope coefficient
    /// (lower −= β·sign, upper += β·sign per split neuron). `beta_signed` is the per-ReLU
    /// `β·sign` (β≥0; 0 for non-split neurons) in FOLD order (each branch's `Activation`
    /// layers in order, F-branch before P-branch). This is the BaB per-domain bound on the
    /// GPU instead of the ~60 s/domain CPU dense backward. Because a β-CROWN bound is a
    /// valid Lagrangian dual for ANY β≥0, this is SOUND regardless of the β values; the
    /// extra f32 add is over-bounded outward in the certified error.
    ///
    /// Default: unsupported (engines fall back to the CPU beta-CROWN per-domain path).
    fn crown_backward_gpu_resnet_sound_beta(
        &self,
        _segments: &[GpuResnetSegment],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _beta_signed: &[Vec<f32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "beta resnet sound GPU CROWN backward not supported by this engine".into(),
        ))
    }

    /// Guard-only serial re-fold of
    /// [`GpuCrownBackward::crown_backward_gpu_resnet_sound_beta`].
    ///
    /// Wide proof-forest callers use this as an independent, single-domain
    /// numerical oracle before accepting a batched result. Implementations may
    /// bypass performance-only dispatch gates here (for example, a minimum GPU
    /// work-size threshold), but MUST preserve the full sound arithmetic and
    /// validation contract of the ordinary serial entry. The returned bound is
    /// compared only; it is never substituted directly into a verdict.
    ///
    /// The default delegates to the ordinary serial entry, so existing engines
    /// retain byte-identical behavior. Backends with a performance gate can
    /// override this method to force the same sound serial kernel for the small
    /// guard sample.
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_beta_refold_oracle(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownResult> {
        self.crown_backward_gpu_resnet_sound_beta(
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
            frontier_abs,
            node_abs,
        )
    }

    /// BATCHED (multi-domain) form of [`crown_backward_gpu_resnet_sound_beta`]
    /// (#batched-bab): compute the sound β-folded bounds for MANY BaB subdomains
    /// that share the SAME network (weights/topology) but differ in relaxation
    /// slopes, β, error-frontier bounds, and input box, from ONE shared spec
    /// `seed`. Returns one `GpuCrownResult` per domain, in `domains` order.
    ///
    /// The domain axis is a pure batch dimension (CROWN backward has no
    /// cross-spec-row reduction), so all domains can share one wide GEMM.
    /// Increment 1 (the reference stacker) dispatches the existing per-domain
    /// kernel per block — byte-identical to N serial
    /// [`crown_backward_gpu_resnet_sound_beta`] calls — to establish the API +
    /// homogeneity gate + differential oracle; a later increment replaces the
    /// dispatch with a single wide GPU pass. Default: unsupported (callers fall
    /// back to the per-domain serial/rayon loop). Engines that support it MUST
    /// return `Err` (→ serial fallback) on a heterogeneous or non-finite batch —
    /// never a wrong (tighter) bound.
    fn crown_backward_gpu_resnet_sound_beta_batched(
        &self,
        _domains: &[GpuResnetBatchedDomainRef<'_>],
        _seed: &GpuCrownSeed,
    ) -> Result<Vec<GpuCrownResult>> {
        Err(NyError::UnsupportedOp(
            "batched beta resnet sound GPU CROWN backward not supported by this engine".into(),
        ))
    }

    /// #clip-interm-resnet-batched: the coeff-CAPTURING sibling of
    /// [`crown_backward_gpu_resnet_sound_beta_batched`]. Runs the SAME single wide
    /// resident backward over all `n_domains` subdomains (one GPU pass for the whole
    /// frontier) and returns BOTH the concretized per-domain bounds AND the downloaded
    /// input-relative coefficient frontier ([`GpuResidentCoeffBatched`]) — captured from
    /// a force-fine (per-ReLU error-concretized) pass so the per-coefficient error is
    /// already largely folded into the bias error. The coeff frontier lets the batched
    /// intermediate-domain clip do its constrained concretization per child WITHOUT a
    /// per-child seeded backward (the throughput lever). NON-default: only the dark
    /// `NY_CLIP_INTERM_RESNET` clip lane calls this. Default: unsupported (caller keeps
    /// the frozen intermediates — sound, no tightening).
    fn crown_backward_gpu_resnet_sound_beta_batched_coeff(
        &self,
        _domains: &[GpuResnetBatchedDomainRef<'_>],
        _seed: &GpuCrownSeed,
    ) -> Result<(Vec<GpuCrownResult>, GpuResidentCoeffBatched)> {
        Err(NyError::UnsupportedOp(
            "coeff-capturing batched beta resnet sound GPU CROWN backward not supported by this \
             engine"
                .into(),
        ))
    }

    /// #batched-bab part A (wide β-opt): the GRADIENT-capturing wide batched backward.
    /// Runs all `n_domains` subdomains in ONE wide resident pass (like
    /// [`crown_backward_gpu_resnet_sound_beta_batched`]) AND gathers, per ReLU, the
    /// pre-transform LOWER A-coefficient values at the caller-supplied UNION of every
    /// domain's split-neuron columns — the inputs to the per-domain analytic β gradient
    /// `∂lb_row/∂β_k = −sign_k·A_lower[row, k]`. `union_gather_idx` is per-ReLU in fold
    /// order (one entry per `Activation`; empty ⇒ nothing gathered for that ReLU). The
    /// SAME column list applies to every wide row `s ∈ [0, N)`; row `s` belongs to domain
    /// `s / num_specs_per_dom`, so `gathers[r][s*U_r+i] = A_lower[wide-row s,
    /// union_gather_idx[r][i]]` — each domain reads its own columns' A-values from its own
    /// rows. Bounds are identical to the non-gather batched path (gather reads the
    /// coefficient stream only).
    ///
    /// #w4 wide α+β ascent: `relu_pre_lower` additionally requests per-domain ALPHA
    /// gradients — per domain, per ReLU (fold order), the pre-activation lower bounds
    /// with stable neurons masked to 0. Non-empty ⇒ the returned `alpha_grads[r]` is
    /// `n_domains*nn_r` with domain d's block at `d*nn_r`, holding the analytic
    /// `∂lb/∂α_i = pre_lower[d·nn+i] · Σ_{rows of d} max(A_lower[row, i], 0)` reduced
    /// over ONLY that domain's spec-row block. Empty ⇒ no capture (empty vec), bounds
    /// byte-for-byte unchanged. Non-soundness-critical (steers β/α; any β ≥ 0 is a
    /// valid dual, any α ∈ [0,1] a valid lower relaxation slope). Default: unsupported
    /// (caller falls back to the per-domain serial ascent).
    fn crown_backward_gpu_resnet_sound_beta_batched_grad(
        &self,
        _domains: &[GpuResnetBatchedDomainRef<'_>],
        _seed: &GpuCrownSeed,
        _union_gather_idx: &[&[u32]],
        _relu_pre_lower: &[&[Vec<f32>]],
    ) -> Result<(Vec<GpuCrownResult>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        Err(NyError::UnsupportedOp(
            "batched-grad beta resnet sound GPU CROWN backward not supported by this engine".into(),
        ))
    }

    /// Trajectory-capturing sibling of
    /// [`crown_backward_gpu_resnet_sound_beta_batched_grad`](Self::crown_backward_gpu_resnet_sound_beta_batched_grad).
    /// Returns the sound bounds, alpha gradients, beta gathers, and downloaded
    /// input-relative coefficient frontier from ONE logical wide call.  A backend
    /// may internally device-safe-subchunk the domain axis or run its established
    /// sound error-concretization tightening pass.
    ///
    /// Default: unsupported.  Callers can retain the preceding sound bound and
    /// skip trajectory refinement without affecting soundness.
    fn crown_backward_gpu_resnet_sound_beta_batched_trajectory(
        &self,
        _domains: &[GpuResnetBatchedDomainRef<'_>],
        _seed: &GpuCrownSeed,
        _union_gather_idx: &[&[u32]],
        _relu_pre_lower: &[&[Vec<f32>]],
    ) -> Result<GpuCrownTrajectoryResult> {
        Err(NyError::UnsupportedOp(
            "trajectory-capturing batched beta resnet sound GPU CROWN backward not supported by \
             this engine"
                .into(),
        ))
    }

    /// TRUE joint α-gradient, computed ON-DEVICE (task #39, the cifar100/tinyimagenet
    /// throughput lever; `docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md` §3). Computes
    /// `∂(lower_bound)/∂α` for every ReLU neuron of ONE BaB sub-domain by the
    /// coefficient-channel forward fold + hand-derived reverse-mode adjoint of
    /// `ny_core::joint_alpha_grad` (the FD-proven CPU oracle), entirely on device —
    /// so the correct joint gradient no longer pays the per-domain CPU re-fold.
    /// Returns one `Vec<f32>` (length `num_neurons`) per `Activation` in FOLD order,
    /// identical in shape/order/semantics to the CPU oracle
    /// `ny_core::joint_alpha_grad::joint_alpha_gradient`.
    ///
    /// `seed_lower_a` is the shared spec seed (`num_specs × output_dim` row-major);
    /// the per-domain α is baked into the segments' `Activation` `lower_slope`;
    /// `input_lower/upper` is this domain's input box. NON-soundness-critical (steers
    /// α∈[0,1]; the verdict is always the sound fold). Default: unsupported (caller
    /// falls back to the CPU oracle — still the correct gradient, never unsound).
    fn crown_joint_alpha_gradient_resident(
        &self,
        _segments: &[GpuResnetSegment],
        _seed_lower_a: &[f32],
        _num_specs: usize,
        _output_dim: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        Err(NyError::UnsupportedOp(
            "on-device joint alpha gradient not supported by this engine".into(),
        ))
    }

    /// Beta-GRADIENT variant of [`crown_backward_gpu_resnet_sound_beta`]
    /// (#w4-split-tightening): same sound β-folded bounds, plus each requested
    /// ReLU's LOWER A-coefficient values gathered at the requested (split)
    /// neuron columns — the inputs to the CPU analytic β-gradient rule
    /// `∂lb_row/∂β_k = −sign_k · A_lower[row, k]`
    /// (`GraphBetaState::compute_gradients_for_spec_row`). `beta_gather_idx`
    /// is per-ReLU in the SAME fold order as `beta_signed` (one entry per
    /// `Activation`; empty list ⇒ nothing gathered for that ReLU). The gather
    /// reads the pre-transform lower coefficient buffer only (no bound-buffer
    /// writes), so the returned bounds are identical to the non-gather beta
    /// path. Gathered values are non-soundness-critical (they only steer β;
    /// any β ≥ 0 is a valid dual).
    ///
    /// Default: unsupported (callers fall back to single-shot beta bounds
    /// without per-domain β optimization).
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_beta_grad(
        &self,
        _segments: &[GpuResnetSegment],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _beta_signed: &[Vec<f32>],
        _beta_gather_idx: &[Vec<u32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownBetaGradResult> {
        Err(NyError::UnsupportedOp(
            "beta-gradient resnet sound GPU CROWN backward not supported by this engine".into(),
        ))
    }

    /// #batched-vjp: EXACT point-Jacobian VJP for K attack restarts in ONE wide
    /// GPU pass — `grads[k] = spec_row_k · W_L · D_{L-1,k} ··· D_{1,k} · W_1`,
    /// the exact gradient of `spec_row_k · f(x_k)` for a piecewise-linear net
    /// whose ReLU masks at restart point `x_k` are `D_{i,k}`.
    ///
    /// - `layers_backward`: the SHARED backward-order (output→input) layer
    ///   template — `Linear`/`Conv2d` weights (shared `Arc`s across the batch)
    ///   plus `Activation` entries. Fold-away ops (Flatten/Reshape) are absent.
    /// - `mask_positions`: indices into `layers_backward` of the `Activation`
    ///   entries that are per-restart ReLU MASK slots (backward/fold order).
    ///   Non-listed `Activation` entries are static affine ops (constant
    ///   arithmetic) shared by every restart.
    /// - `masks`: `masks[k][r]` is restart `k`'s 0/1 mask (`pre_act > 0`) for
    ///   mask slot `r` (`len == num_neurons` of that slot). The engine bakes it
    ///   as `lower_slope == upper_slope == mask`, zero intercepts — the sign
    ///   routing is then irrelevant, so the folded input-level LOWER
    ///   coefficient row IS the exact f32 point gradient.
    /// - `spec_rows`: `K × output_dim` row-major, restart `k`'s cotangent row
    ///   (rows MAY differ per restart — e.g. per-point joint-margin rows).
    ///
    /// Returns `K` gradient vectors, each `input_dim` long. ATTACK-ONLY: the
    /// gradients steer PGD; every counterexample is concretely re-validated, so
    /// this can never affect a verdict. Engines MUST return `Err` on any
    /// shape/assembly/GPU failure (caller falls back to the sequential exact
    /// gradient), never a silently wrong gradient batch.
    ///
    /// Default: unsupported.
    fn crown_point_vjp_batched(
        &self,
        _layers_backward: &[GpuCrownLayer],
        _mask_positions: &[usize],
        _masks: &[Vec<Vec<f32>>],
        _spec_rows: &[f32],
        _output_dim: usize,
        _input_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        Err(NyError::UnsupportedOp(
            "batched point-VJP not supported by this engine".into(),
        ))
    }

    /// #batched-vjp-resnet: the RESNET-DAG sibling of [`Self::crown_point_vjp_batched`]
    /// — exact point-Jacobian VJP for K attack restarts in ONE wide GPU pass over a
    /// backward-order (output→input) [`GpuResnetSegment`] template (chains +
    /// identity/projection residual blocks). At a concrete point the residual merge's
    /// reverse rule is the plain fan-in ADD, which is exactly what the resident fold's
    /// `Residual`/`ResidualProj` handling computes (`A_in = backward_F(A) + A` /
    /// `backward_F(A) + backward_P(A)`), so with per-restart 0/1 mask slopes the
    /// folded input-level LOWER coefficient rows ARE the exact per-restart gradients.
    ///
    /// - `segments_backward`: the SHARED backward-order segment template. Weights are
    ///   `Arc`-shared across the batch; fold-away ops (Flatten/Reshape) are absent.
    /// - `mask_flat_positions`: per-restart ReLU MASK slot positions as indices into
    ///   the FLATTENED layer traversal of `segments_backward` — for each segment in
    ///   order: `Chain` layers in stored order; `Residual` F-branch layers;
    ///   `ResidualProj` F-branch then P-branch layers. Non-listed `Activation`
    ///   entries are static affine ops shared by every restart.
    /// - `masks`: `masks[k][r]` is restart `k`'s 0/1 mask for slot `r` (aligned with
    ///   `mask_flat_positions` order).
    /// - `spec_rows`: `K × output_dim` row-major per-restart cotangent rows.
    ///
    /// Returns `K` gradient vectors, each `input_dim` long. ATTACK-ONLY (identical
    /// contract to the chain entry): engines MUST return `Err` on any failure —
    /// never a silently wrong gradient batch.
    ///
    /// Default: unsupported.
    fn crown_point_vjp_batched_resnet(
        &self,
        _segments_backward: &[GpuResnetSegment],
        _mask_flat_positions: &[usize],
        _masks: &[Vec<Vec<f32>>],
        _spec_rows: &[f32],
        _output_dim: usize,
        _input_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        Err(NyError::UnsupportedOp(
            "batched resnet point-VJP not supported by this engine".into(),
        ))
    }

    /// Cooperative-cancellation deadline for long multi-dispatch CROWN backward
    /// calls (#w4-refresh-deadline). A single GPU dispatch cannot be interrupted
    /// mid-flight, but a wide spec-batched backward (e.g. a 14400-spec per-target
    /// refresh split into dozens of batches) and a deep sound resident layer walk
    /// CAN stop *between* units of work. Engines that honor this check the stored
    /// deadline between spec batches / layer folds and return
    /// `NyError::DeadlineExceeded`, which every CROWN caller already treats as a
    /// sound fallback (reference/IBP bounds). Callers scope the deadline around a
    /// bounded region and MUST clear it (set `None`) afterwards.
    ///
    /// Default: no-op (engines without cooperative cancellation run to completion,
    /// the pre-existing behavior).
    fn set_crown_backward_deadline(&self, _deadline: Option<std::time::Instant>) {}
}

#[path = "gemm_naive.rs"]
mod naive;
pub use naive::NaiveCpuGemmEngine;

#[cfg(test)]
#[path = "gemm_tests.rs"]
mod tests;
