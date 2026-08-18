// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #fl-value-gpu-tier: deadline-capable WGPU f32 GEMM for the forward-linear
//! VALUE seam only.
//!
//! Third narrow-engine sibling of [`crate::attack_steering`] (b030e2a8) and
//! [`crate::gradient_steering`]: a quarantine-preserving wrapper around
//! [`WgpuDevice`] that exposes exactly ONE method —
//! [`GemmEngine::gemm_f32_with_deadline`] — for the forward-linear image
//! composition's big value GEMMs (`A·W` center / `|A|·|W|` radius,
//! `image.rs::certified_f64_gemm` deadline branch, `allow_f32` gated by
//! `NY_FORWARD_LINEAR_F32`).
//!
//! # Why this consumer is sound with an f32 GPU product
//!
//! The call site charges the value GEMMs' full accumulation error to the bias
//! as `gamma_{K+4}^f32 · S` plus an FTZ underflow addend
//! (`image.rs:forward_f32_ftz_bias`) — a Higham bound that is
//! **summation-order independent** and whose FTZ addend exists precisely FOR
//! Metal-class flush-to-zero backends. The S-base GEMMs (the certified error
//! base) stay f64 on CPU and never route here. Any plain RN-f32 GEMM backend
//! is therefore admissible for the VALUES; measured GPU/CPU divergence on the
//! fused wgpu ops is 9.5e-7 (m7, 1bb88165), far inside the charged penalty.
//! PROPOSAL-GRADE VALUES ONLY in the sense that the certificate lives at the
//! call site: this engine must never be reachable from a call site that does
//! not charge the f32 penalty — hence the steering-precedent quarantine below.
//!
//! # Quarantine (steering precedent)
//!
//! Every other `GemmEngine` surface typed-refuses: `gemm_f32` refuses
//! explicitly (which also poisons the `gemm_f32_fast` / `gemm_interval_sound`
//! / `crown_aw_error_step` trait defaults built on it), `gemm_f64*` and
//! `conv_transpose_2d*` keep their refusing defaults, and every `as_gpu_*`
//! accessor stays `None`. A misrouted handle is unusable as an ordinary
//! engine, and `WgpuDevice`'s own verdict quarantine
//! (`provides_sound_gpu_crown() == false`, quarantined `GemmEngine` impl) is
//! never weakened by constructing one.
//!
//! # Deadline / chunking contract (lane D granularity)
//!
//! Full products are split along the output row axis `m` only (the
//! contraction `k` is never split), one blocking WGPU submission per chunk,
//! with a host-side deadline poll before every submission and after the last.
//! Chunk rows are bounded by BOTH the caller's `max_dispatch_macs` cap and
//! [`FL_VALUE_CHUNK_BUDGET_ELEMS`] (half the 128 MB wgpu binding budget), so
//! no submission can approach the budget refusal m7 measured the best shapes
//! against. At expiry the remaining chunks are abandoned and a typed
//! [`NyError::DeadlineExceeded`] is returned — never a partial result.
//!
//! This capability view shares the process-global ordinary WGPU context with
//! attack and alpha-gradient steering. It does not construct or retain a second
//! adapter/device/pipeline set, and sharing does not widen any trait surface.

use ny_core::{GemmEngine, NyError, Result};
use std::sync::Arc;

use crate::attack_steering::shared_proposal_wgpu_device;
use crate::wgpu_device::{WgpuDevice, MAX_BINDING_ELEMS};

/// Minimum total MACs (`m*k*n`) below which this engine refuses and the caller
/// keeps the tiled CPU f32 path.
///
/// MEASURED basis (m7, commit 1bb88165, this box, M5 Max): fused wgpu conv
/// ops span 0.38x vs CPU at small shapes (GPU LOSES — dispatch + readback
/// overhead dominates) up to 15.52x at OC64 IC64 16x16, still climbing at the
/// 128 MB buffer-budget refusal. The win region is GMAC-class totals; the
/// constant sits at the LOWER edge of that class (0.5 GMAC) deliberately so
/// the census-representative FL rate-probe fixture (contraction 1152, out_c
/// 16 → ~0.537 GMAC per value GEMM, `forward_linear.rs`
/// `forward_linear_calibration_fixture`) exercises this tier: the measured
/// admission rate must price the SAME dispatch chain the real 559-GMAC build
/// uses (whose dominant per-GEMM chunks are 4+ GMAC). If the GPU loses at the
/// probe shape on some host, the probe honestly reports the slower rate and
/// the gate refuses — fail-honest either way.
pub const FL_VALUE_GEMM_MIN_MACS: u128 = 500_000_000;

/// Per-submission element budget for the chunked row-block path: HALF of the
/// derived wgpu binding capacity ([`MAX_BINDING_ELEMS`] = 128 MB binding limit
/// / 1.2 buffer-pool growth / 4 bytes), applied independently to the A-chunk
/// (`rows*k`) and the output chunk (`rows*n`). Staying at half the budget
/// keeps every submission well clear of the binding refusal (and of the pool's
/// growth-factor slack) instead of riding its edge.
pub const FL_VALUE_CHUNK_BUDGET_ELEMS: usize = MAX_BINDING_ELEMS / 2;

impl FlValueGemmDevice {
    /// The chunk budget from the LIVE device limit (#hard-caps, 2026-08-06).
    ///
    /// [`FL_VALUE_CHUNK_BUDGET_ELEMS`] derives from a hard-coded 128 MiB, which
    /// is wgpu's *default* binding size — not the device's. `WgpuDevice::new`
    /// requests the adapter's real limits, and on an Apple M4 Pro that is
    /// 4095 MiB, so the constant under-estimates by 32x and this tier chunked
    /// far more finely than the hardware required.
    ///
    /// Still HALF the live capacity, preserving the constant's stated intent —
    /// stay clear of the binding refusal and the pool's growth-factor slack
    /// rather than riding the edge. Never below the constant, so this can only
    /// raise the ceiling relative to today.
    ///
    /// Throughput only: this tier exposes a VALUE GEMM and never a certified
    /// bound, so a wider chunk changes how work is submitted, never what is
    /// proved.
    fn chunk_budget_elems(&self) -> usize {
        (self.device.max_binding_elems_live() / 2).max(FL_VALUE_CHUNK_BUDGET_ELEMS)
    }
}

/// FL-value WGPU engine: exposes ONLY the deadline-bounded f32 GEMM. See the
/// module docs for the soundness argument and the routing contract.
/// Constructing one NEVER weakens the proof-adapter quarantine.
pub struct FlValueGemmDevice {
    device: Arc<WgpuDevice>,
}

impl FlValueGemmDevice {
    /// Create the FL-value accelerator on the WGPU adapter.
    ///
    /// Fails (and the caller keeps the tiled CPU f32/f64 fallbacks) when no
    /// adapter is available; failure must never fail the verification run.
    pub fn new_wgpu() -> Result<Self> {
        Ok(Self {
            device: shared_proposal_wgpu_device()?,
        })
    }
}

impl GemmEngine for FlValueGemmDevice {
    fn backend_provenance(&self) -> &'static str {
        // Truthful identity (#backend-provenance): distinguishes the FL value
        // handle from the CPU engine, the quarantined seam, and both steering
        // channels.
        "wgpu-fl-value-f32"
    }

    /// The FL value channel is NOT a general GEMM engine. Refusing here keeps
    /// the handle unusable at any engine-shaped call site it might be
    /// misrouted to (and poisons every trait default built on `gemm_f32`);
    /// its single consumer only ever calls `gemm_f32_with_deadline`.
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        Err(NyError::UnsupportedOp(
            "wgpu-fl-value-f32 exposes only the deadline-bounded f32 value GEMM; \
             it is not a general GEMM engine"
                .into(),
        ))
    }

    /// Live: the forward-linear f32 value seam drives its deadline-bearing
    /// value GEMMs through this method. See the module docs for the chunking
    /// and refusal contract.
    fn gemm_f32_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        deadline: std::time::Instant,
        max_dispatch_macs: usize,
    ) -> Result<Vec<f32>> {
        let overflow = |product: &str| {
            NyError::InvalidSpec(format!(
                "wgpu-fl-value-f32: {product} overflows usize for shape {m}x{k}x{n}"
            ))
        };
        let lhs_len = m.checked_mul(k).ok_or_else(|| overflow("m*k"))?;
        let rhs_len = k.checked_mul(n).ok_or_else(|| overflow("k*n"))?;
        let out_len = m.checked_mul(n).ok_or_else(|| overflow("m*n"))?;
        if a.len() != lhs_len {
            return Err(NyError::shape_mismatch(vec![m, k], vec![a.len()]));
        }
        if b.len() != rhs_len {
            return Err(NyError::shape_mismatch(vec![k, n], vec![b.len()]));
        }

        // SIZE THRESHOLD (measured crossover, see FL_VALUE_GEMM_MIN_MACS).
        let total_macs = (m as u128) * (k as u128) * (n as u128);
        if total_macs < FL_VALUE_GEMM_MIN_MACS {
            return Err(NyError::UnsupportedConfiguration(format!(
                "wgpu-fl-value-f32: {total_macs} MACs is below the measured GPU/CPU \
                 crossover ({FL_VALUE_GEMM_MIN_MACS}); the tiled CPU f32 path is faster \
                 at this shape"
            )));
        }
        // total_macs >= 1 GMAC implies m, k, n >= 1 from here on.

        // B (k x n) is shared by every chunk and cannot be split without
        // splitting the contraction; refuse when it alone exceeds the budget.
        let chunk_budget = self.chunk_budget_elems();
        if rhs_len > chunk_budget {
            return Err(NyError::GpuMemoryExceeded {
                required_bytes: rhs_len.saturating_mul(size_of::<f32>()),
                budget_bytes: chunk_budget.saturating_mul(size_of::<f32>()),
            });
        }
        // One row of output is one non-interruptible k*n-MAC dispatch minimum;
        // if that already exceeds the caller's cap, the bounded-dispatch
        // contract cannot be honored without splitting k (forbidden — it would
        // break the caller's order-independent gamma certificate's "one dot
        // product per coefficient" shape). Typed-refuse.
        let row_macs = k.saturating_mul(n);
        if row_macs > max_dispatch_macs {
            return Err(NyError::UnsupportedConfiguration(format!(
                "wgpu-fl-value-f32: one output row is {row_macs} MACs, above the \
                 {max_dispatch_macs}-MAC dispatch cap; cannot chunk without splitting k"
            )));
        }

        // CHUNKED row-block submission: rows per blocking submission bounded by
        // (a) the dispatch-MAC cap and (b) half the wgpu binding budget for
        // both the A-chunk and the output chunk.
        let chunk_rows = (max_dispatch_macs / row_macs)
            .min(chunk_budget / k)
            .min(chunk_budget / n)
            .clamp(1, m);
        if chunk_rows < m {
            tracing::debug!(
                m,
                k,
                n,
                chunk_rows,
                chunks = m.div_ceil(chunk_rows),
                "wgpu-fl-value-f32: chunked row-block submission engaged (#fl-value-gpu-tier)"
            );
        }

        let check = |context: &str| {
            if std::time::Instant::now() >= deadline {
                Err(NyError::DeadlineExceeded(format!(
                    "wgpu-fl-value-f32: deadline exceeded {context}; abandoning remaining chunks"
                )))
            } else {
                Ok(())
            }
        };

        let mut out = Vec::new();
        out.try_reserve_exact(out_len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: out_len.saturating_mul(size_of::<f32>()),
                budget_bytes: usize::MAX,
                site: "ny-gpu::fl_value_gemm/output",
            })?;
        let mut row0 = 0usize;
        while row0 < m {
            // Host-side poll BEFORE every submission: at expiry the remaining
            // chunks are abandoned and no partial result can be published.
            check("before chunk submission")?;
            let rows = chunk_rows.min(m - row0);
            let chunk = self
                .device
                .gemm_f32(rows, k, n, &a[row0 * k..(row0 + rows) * k], b)?;
            if chunk.len() != rows * n {
                return Err(NyError::InternalError(format!(
                    "wgpu-fl-value-f32: chunk returned {} elements, expected {}",
                    chunk.len(),
                    rows * n
                )));
            }
            out.extend_from_slice(&chunk);
            row0 += rows;
        }
        // Poll after the final synchronization: a result completed after the
        // deadline is never published.
        check("after final chunk")?;

        if out.len() != out_len {
            return Err(NyError::InternalError(format!(
                "wgpu-fl-value-f32: assembled {} elements, expected {out_len}",
                out.len()
            )));
        }
        // Result validation: poison is a typed refusal, never a value — the
        // caller's gamma/FTZ charge covers rounding, not overflow. TWO poison
        // forms exist because the WGSL GEMM shader's `nan_safe_clamp`
        // (shaders.rs, #2258/#2708) preserves NaN but saturates overflow Inf
        // to the FINITE sentinel ±FALLBACK_BOUND (±1e10): an all-finite scan
        // alone would PUBLISH a clamped overflow, silently losing magnitude
        // the certificate does not cover. Any |v| >= FALLBACK_BOUND is
        // indistinguishable from the sentinel and refuses fail-closed (the
        // CPU tiers recompute the true value; upstream forward-linear
        // coefficients are CROWN-safe < 1e10, so legitimate products at the
        // sentinel magnitude are degenerate rows the caller degrades anyway).
        if out
            .iter()
            .any(|v| !v.is_finite() || v.abs() >= ny_core::FALLBACK_BOUND)
        {
            return Err(NyError::NumericalInstability(
                "wgpu-fl-value-f32: non-finite or overflow-sentinel (|v| >= 1e10) value in \
                 GPU GEMM result"
                    .into(),
            ));
        }
        Ok(out)
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Deterministic wide-magnitude f32 stream (mirrors the seam oracles).
    fn stream(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = ((s >> 40) % 8) as i32 - 4;
            let mant = ((s >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32;
            let sign = if (s >> 3) & 1 == 1 { -1.0 } else { 1.0 };
            sign * (1.0 + mant) * 2f32.powi(e)
        }
    }

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_mins(10)
    }

    /// Census-class shape >= the 1-GMAC threshold: k=1152 (128ch 3x3), small
    /// enough to keep the test fast. m*k*n = 2048*1152*512 = 1.21 GMAC.
    const CENSUS: (usize, usize, usize) = (2048, 1152, 512);

    /// Sound-enclosure correctness at a census shape, including a chunked
    /// configuration: |gpu - f64_ref| <= gamma_k^f32 * S per coefficient (the
    /// exact criterion the call-site charge assumes; bit-equality with a CPU
    /// f32 reference is NOT required — RN-f32 order may differ, measured
    /// divergence 9.5e-7 on the fused ops).
    #[test]
    fn fl_value_gemm_matches_f64_reference_within_gamma_incl_chunked() {
        let device = FlValueGemmDevice::new_wgpu().expect("adapter probed available");
        let (m, k, n) = CENSUS;
        let mut next = stream(0xF1F1);
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| next()).collect();

        // gamma_k^f32 for k+4 (the call-site factor), u = 2^-24.
        let u = 2f64.powi(-24);
        let ku = ((k + 4) as f64) * u;
        let gamma = ku / (1.0 - ku);

        // f64 reference via faer-free scalar path on a row subsample (full
        // m*n*k f64 reference would dominate the test); S computed alongside.
        let check_rows = [0usize, 1, m / 2, m - 1];

        for (label, max_dispatch) in [
            ("single-submission", usize::MAX),
            // Forces ~8 chunks at this shape: cap = m*k*n/8 MACs.
            ("chunked", m * k * n / 8),
        ] {
            let out = device
                .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), max_dispatch)
                .expect("census-shape GEMM must succeed");
            assert_eq!(out.len(), m * n, "{label}: output shape");
            for &i in &check_rows {
                for j in [0usize, n / 2, n - 1] {
                    let mut refv = 0.0f64;
                    let mut s = 0.0f64;
                    for kk in 0..k {
                        let x = f64::from(a[i * k + kk]);
                        let y = f64::from(b[kk * n + j]);
                        refv += x * y;
                        s += x.abs() * y.abs();
                    }
                    let err = (f64::from(out[i * n + j]) - refv).abs();
                    let bound = gamma * s * (1.0 + 1e-6);
                    assert!(
                        err <= bound,
                        "{label}: UNSOUND at ({i},{j}): err {err} > gamma*S {bound}"
                    );
                }
            }
        }
    }

    /// Below the measured crossover the engine must typed-refuse (the caller
    /// keeps the tiled CPU f32 path).
    #[test]
    fn fl_value_gemm_refuses_below_measured_crossover() {
        let device = FlValueGemmDevice::new_wgpu().expect("adapter probed available");
        let (m, k, n) = (64usize, 64usize, 64usize); // 262144 MACs << 1 GMAC
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let err = device
            .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
            .expect_err("sub-crossover shape must refuse");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(_)),
            "unexpected refusal type: {err}"
        );
    }

    /// Deadline expiry mid-sequence: typed refusal, never a partial result.
    /// The already-expired case is fully deterministic; the mid-sequence case
    /// uses a chunked configuration with a deadline far shorter than the
    /// measured multi-chunk duration.
    #[test]
    fn fl_value_gemm_deadline_expiry_is_typed_refusal_no_partial() {
        let device = FlValueGemmDevice::new_wgpu().expect("adapter probed available");
        let (m, k, n) = CENSUS;
        let mut next = stream(0xDEAD);
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| next()).collect();

        // Expired before the first submission: deterministic refusal.
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("1ms before now is inside the process uptime");
        let err = device
            .gemm_f32_with_deadline(m, k, n, &a, &b, expired, usize::MAX)
            .expect_err("expired deadline must refuse before any submission");
        assert!(err.is_deadline_exceeded(), "unexpected error: {err}");

        // Mid-sequence: many chunks, deadline shorter than the full product
        // (measured single-chunk latency is far above 1 ms on every adapter
        // this engine targets). Either the poll between submissions or the
        // final poll must convert expiry into the typed refusal.
        let err = device
            .gemm_f32_with_deadline(
                m,
                k,
                n,
                &a,
                &b,
                Instant::now() + Duration::from_millis(1),
                m * k * n / 64,
            )
            .expect_err("mid-sequence expiry must abandon and refuse");
        assert!(err.is_deadline_exceeded(), "unexpected error: {err}");
    }

    /// Poisoned products must be typed refusals, never values. The WGSL GEMM
    /// shader saturates overflow Inf to the FINITE ±1e10 sentinel
    /// (`nan_safe_clamp`, #2258) and preserves NaN (#2708) — so this pins the
    /// sentinel-magnitude scan (an all-finite check alone would unsoundly
    /// publish clamped overflow) AND the NaN path.
    #[test]
    fn fl_value_gemm_nonfinite_or_sentinel_result_is_typed_refusal() {
        let device = FlValueGemmDevice::new_wgpu().expect("adapter probed available");
        let (m, k, n) = (1024usize, 1152usize, 1024usize); // 1.21 GMAC

        // Overflow → shader clamps to the finite +1e10 sentinel → refusal.
        let mut a = vec![1.0f32; m * k];
        let b = vec![f32::MAX; k * n];
        a[0] = f32::MAX; // row 0 accumulates k * MAX*MAX -> inf -> sentinel
        let err = device
            .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
            .expect_err("overflow-sentinel product must refuse");
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "unexpected refusal type: {err}"
        );

        // NaN (Inf - Inf inside one dot product) is preserved by the shader
        // and must also refuse.
        let mut a = vec![1.0f32; m * k];
        a[0] = f32::MAX;
        a[1] = -f32::MAX;
        let mut b = vec![1.0f32; k * n];
        for j in 0..n {
            b[j] = f32::MAX; // row 0 of B
            b[n + j] = f32::MAX; // row 1 of B
        }
        let err = device
            .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
            .expect_err("NaN product must refuse");
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "unexpected refusal type: {err}"
        );
    }

    /// Steering-precedent quarantine pin: every other engine surface refuses,
    /// no verdict-bearing accessor is exposed, and provenance is distinct.
    #[test]
    fn fl_value_gemm_quarantine_and_provenance() {
        let device = FlValueGemmDevice::new_wgpu().expect("adapter probed available");
        let engine: &dyn GemmEngine = &device;
        assert_eq!(engine.backend_provenance(), "wgpu-fl-value-f32");
        assert!(engine.gemm_f32(1, 1, 1, &[1.0], &[1.0]).is_err());
        assert!(engine.gemm_f32_fast(1, 1, 1, &[1.0], &[1.0]).is_err());
        assert!(engine.gemm_f64(1, 1, 1, &[1.0], &[1.0]).is_err());
        assert!(engine
            .gemm_f64_with_deadline(1, 1, 1, &[1.0], &[1.0], far_deadline(), 1 << 20)
            .is_err());
        assert!(engine
            .gemm_interval_sound(1, 1, 1, &[1.0], &[1.0], &[1.0], &[1.0])
            .is_err());
        assert!(engine
            .crown_aw_error_step(1, 1, 1, &[1.0], &[0.0], &[1.0])
            .is_err());
        assert!(engine.as_gpu_crown_backward().is_none());
        assert!(engine.as_gpu_ibp_forward().is_none());
        assert!(engine.as_gpu_ibp_forward_ext().is_none());
        assert!(engine.as_gpu_dag_ibp_forward_ext().is_none());
    }
}
