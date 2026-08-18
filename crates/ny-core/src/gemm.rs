// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, mem::size_of, sync::Arc};

use crate::{checked_dim_product, NyError, Result};

// Trust contract attribute. Under tRustc contract verification (`--cfg trust_verify`)
// `#[ensures]` is the first-class builtin from `core::contracts`; under stable rustc
// it is the no-op NY-owned `trust` compatibility crate. Mirrors ny-cert's dual
// import so the same `#[ensures(...)]` source verifies under trustc and compiles
// unchanged under rustc.
#[cfg(trust_verify)]
use core::contracts::{ensures, requires};
#[cfg(not(trust_verify))]
use trust::{ensures, requires};

#[path = "gemm_gpu_bab_bound.rs"]
mod gpu_bab_bound;
#[path = "gemm_gpu_dag_ibp.rs"]
mod gpu_dag_ibp;
#[path = "gemm_gpu_ibp.rs"]
mod gpu_ibp;

pub use gpu_bab_bound::{
    certify_gpu_bab_bound_static_schedule, gpu_bab_bound_static_payload_identity_v1,
    GpuBabBoundAcceptedOpen, GpuBabBoundAcceptedResidentDomain,
    GpuBabBoundAcceptedResidentMaintenance, GpuBabBoundAcceptedResidentWave,
    GpuBabBoundAcceptedWave, GpuBabBoundArenaRange, GpuBabBoundBackendCloseDisposition,
    GpuBabBoundBackendCloseReceipt, GpuBabBoundBackendDomainOutcome,
    GpuBabBoundBackendDomainOutcomeKind, GpuBabBoundBackendFailureKind,
    GpuBabBoundBackendIssuerIdentity, GpuBabBoundBackendOpen, GpuBabBoundBackendOpenFailureKind,
    GpuBabBoundBackendOpenPreparation, GpuBabBoundBackendOpenReceipt,
    GpuBabBoundBackendPrepareDisposition, GpuBabBoundBackendRegistration,
    GpuBabBoundBackendResidentMaintenanceDisposition,
    GpuBabBoundBackendResidentMaintenancePrepareDisposition,
    GpuBabBoundBackendResidentMaintenanceReceipt, GpuBabBoundBackendResidentPrepareDisposition,
    GpuBabBoundBackendResidentWaveDisposition, GpuBabBoundBackendResidentWaveReceipt,
    GpuBabBoundBackendRow, GpuBabBoundBackendScheduleDisposition,
    GpuBabBoundBackendScheduleEvidence, GpuBabBoundBackendScheduleIdentity,
    GpuBabBoundBackendSession, GpuBabBoundBackendWaveDisposition, GpuBabBoundBackendWaveReceipt,
    GpuBabBoundDomainArena, GpuBabBoundDomainTranscript, GpuBabBoundF32Tensor,
    GpuBabBoundF32TensorRole, GpuBabBoundGraphPlan, GpuBabBoundMemoryReceipt,
    GpuBabBoundNumericalTcb, GpuBabBoundOperandView, GpuBabBoundOwnedSlice, GpuBabBoundParentGroup,
    GpuBabBoundPhaseCloseDisposition, GpuBabBoundPhaseDecline, GpuBabBoundPhaseDescriptor,
    GpuBabBoundPhaseLease, GpuBabBoundPhaseOpen, GpuBabBoundPhaseOpenFailure,
    GpuBabBoundPhasePolicy, GpuBabBoundPhaseTranscript, GpuBabBoundPreparedResidentGroup,
    GpuBabBoundPreparedResidentMaintenance, GpuBabBoundPreparedResidentWave,
    GpuBabBoundPreparedWave, GpuBabBoundProposedResidentDomain, GpuBabBoundProviderFailure,
    GpuBabBoundProviderFailureKind, GpuBabBoundResidentConstruction,
    GpuBabBoundResidentDomainPolicy, GpuBabBoundResidentF32Family,
    GpuBabBoundResidentFamilyTransfer, GpuBabBoundResidentHostAudit,
    GpuBabBoundResidentMaintenanceCapability, GpuBabBoundResidentMaintenanceDisposition,
    GpuBabBoundResidentMaintenanceFailure, GpuBabBoundResidentMaintenanceMemoryReceipt,
    GpuBabBoundResidentMaintenancePreparation, GpuBabBoundResidentMaintenanceRequest,
    GpuBabBoundResidentMemoryReceipt, GpuBabBoundResidentParentGroup,
    GpuBabBoundResidentParentSource, GpuBabBoundResidentSlotRef, GpuBabBoundResidentSlotTranscript,
    GpuBabBoundResidentSourceAudit, GpuBabBoundResidentSourceClass,
    GpuBabBoundResidentSourcePresence, GpuBabBoundResidentTransferReceipt,
    GpuBabBoundResidentWaveCapability, GpuBabBoundResidentWaveDecline,
    GpuBabBoundResidentWaveDisposition, GpuBabBoundResidentWaveFailure,
    GpuBabBoundResidentWavePreparation, GpuBabBoundResidentWaveRequest,
    GpuBabBoundScheduleCertificate, GpuBabBoundScheduleCertification,
    GpuBabBoundScheduleTcbInvocation, GpuBabBoundSessionTerminal, GpuBabBoundSplitHistoryArena,
    GpuBabBoundSplitHistoryLiteral, GpuBabBoundSplitHistoryPhase, GpuBabBoundSplitHistoryView,
    GpuBabBoundStaticPayloadSource, GpuBabBoundStaticScheduleRequest,
    GpuBabBoundStaticTransferReceipt, GpuBabBoundSubchunk, GpuBabBoundTcbInvocation,
    GpuBabBoundTerminalFailureKind, GpuBabBoundTerminalTranscript, GpuBabBoundTransferReceipt,
    GpuBabBoundU32Tensor, GpuBabBoundU32TensorRole, GpuBabBoundValidatedDomainOutcome,
    GpuBabBoundValidatedDomainOutcomeKind, GpuBabBoundValidatedPhaseClose,
    GpuBabBoundValidatedResidentDomainOutcomeRef, GpuBabBoundValidatedResidentDomainOutcomes,
    GpuBabBoundValidatedResidentRowRef, GpuBabBoundValidatedResidentRows,
    GpuBabBoundValidatedResidentWaveReceipt, GpuBabBoundValidatedRow,
    GpuBabBoundValidatedWaveReceipt, GpuBabBoundWaveCapability, GpuBabBoundWaveDecline,
    GpuBabBoundWaveDisposition, GpuBabBoundWaveFailure, GpuBabBoundWavePreparation,
    GpuBabBoundWaveRequest, ValidatedGpuBabBoundResidentMaintenanceResult,
    ValidatedGpuBabBoundResidentWaveResult, ValidatedGpuBabBoundWaveResult,
    GPU_BAB_BOUND_MAX_APPEND_SPLITS, GPU_BAB_BOUND_MAX_ARENA_VALUES,
    GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE, GPU_BAB_BOUND_MAX_DOMAINS, GPU_BAB_BOUND_MAX_OBJECTIVES,
    GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES, GPU_BAB_BOUND_MAX_RESIDENT_DOMAIN_SLOTS,
    GPU_BAB_BOUND_MAX_RETAINED_V2_CORE_HOST_CHARGED_BYTES, GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS,
    GPU_BAB_BOUND_MAX_SUBMITS_PER_WAVE, GPU_BAB_BOUND_OWNED_SLICE_FIXED_CHARGED_BYTES,
    GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS,
};
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
/// Returns `true` if the value is finite and its absolute value is strictly
/// below [`CROWN_COEFF_MAX`]. The threshold itself is the finite overflow
/// sentinel and therefore cannot represent a trustworthy coefficient. Used by
/// CROWN backward paths to detect
/// near-overflow coefficients before they cascade (#1932).
#[inline]
#[must_use]
pub fn is_crown_coeff_safe(value: f32) -> bool {
    value.is_finite() && value.abs() < CROWN_COEFF_MAX
}

/// f64 variant of [`is_crown_coeff_safe`] for normalization CROWN backward
/// paths that accumulate coefficients in double precision (#3228).
///
/// Uses the same [`CROWN_COEFF_MAX`] threshold (cast to f64). Values exceeding
/// this would overflow when converted to f32 bounds downstream.
#[inline]
#[must_use]
pub fn is_crown_coeff_safe_f64(value: f64) -> bool {
    value.is_finite() && value.abs() < f64::from(CROWN_COEFF_MAX)
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

/// The historical, engine-INDEPENDENT MAC floor for admitting a verdict-grade
/// sound f64 `A·W` to a GEMM engine.
///
/// This mirrors `ny-propagate`'s `SOUND_F64_GEMM_MIN_MACS` (a shared verdict-path
/// constant several arcs depend on); the two are pinned equal by a `const`
/// assertion at that definition. It exists here only so
/// [`SoundF64GemmAdmission::CONSTANT_FLOOR`] — the default every engine gets —
/// reproduces the historical policy exactly.
///
/// Its documented rationale is the **GPU** crossover: a cuBLAS `Dgemm` dispatch
/// costs ~0.4 ms of launch latency, so below ~16 M MACs the rayon-parallel CPU
/// path wins. A CPU-resident engine has ~1 µs of call overhead and no such
/// crossover, which is what [`GemmEngine::sound_f64_deadline_admission`] exists
/// to let it say.
pub const SOUND_F64_GEMM_DEFAULT_MIN_MACS: usize = 1 << 24;

/// An engine's own declaration of which `A·W` shapes it is worth dispatching to
/// on the **deadline-bounded** sound f64 seam (`gemm_f64_with_deadline`).
///
/// # This is a PERFORMANCE declaration, never a soundness one
///
/// Admission does not change what is computed or what is certified. Both the
/// engine path and the pollable CPU fallback evaluate the same full-`k` IEEE-f64
/// dot products, and the caller's `γ_k·S` certificate (Higham Thm 3.1) is
/// summation-order independent, so it covers either result. A wrong declaration
/// costs time; it cannot produce an invalid enclosure or narrow a published
/// bound. Every engine result is still independently validated (shape, finite,
/// sign) before publication.
///
/// # Shape-aware on purpose
///
/// A single MAC count is not sufficient. Measured on faer/CPU, dispatching is
/// catastrophic at `k == 1` (up to ~1000× slower than the scalar fallback),
/// consistently bad at `m == 1`, and bad for a small contraction feeding a very
/// large output. A floor tuned for GPU launch latency hides all three because it
/// gates out the entire region where they live. The fields below let an engine
/// decline exactly those regions while opening the band where it wins.
///
/// # Fail closed
///
/// The default is [`Self::CONSTANT_FLOOR`], which is *exactly*
/// `m*k*p >= SOUND_F64_GEMM_DEFAULT_MIN_MACS` — so an engine that does not
/// override [`GemmEngine::sound_f64_deadline_admission`] (including every GPU
/// engine) keeps byte-identical admission behaviour.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoundF64GemmAdmission {
    /// Minimum `m*k*p` product. The engine's measured crossover.
    pub min_macs: usize,
    /// Minimum number of `A` rows (`m`).
    pub min_rows: usize,
    /// Minimum contraction width (`k`).
    pub min_contraction: usize,
    /// Minimum number of output columns (`p`).
    pub min_columns: usize,
    /// A contraction strictly below this width counts as "small" and is
    /// additionally limited by [`Self::small_contraction_max_output`]. Use `0`
    /// to disable the rule (no `k` is below `0`).
    pub small_contraction_below: usize,
    /// For a small contraction, the largest `m*p` output that may still be
    /// dispatched. Use [`usize::MAX`] to disable.
    pub small_contraction_max_output: usize,
}

impl SoundF64GemmAdmission {
    /// The historical single-constant policy: admit iff
    /// `m*k*p >= SOUND_F64_GEMM_DEFAULT_MIN_MACS`, with no shape rules.
    ///
    /// This is the trait default, so engines that do not override
    /// [`GemmEngine::sound_f64_deadline_admission`] are unaffected.
    pub const CONSTANT_FLOOR: Self = Self {
        min_macs: SOUND_F64_GEMM_DEFAULT_MIN_MACS,
        min_rows: 1,
        min_contraction: 1,
        min_columns: 1,
        small_contraction_below: 0,
        small_contraction_max_output: usize::MAX,
    };

    /// Clamp a declaration into a usable form before it is consulted.
    ///
    /// A zero minimum would admit a degenerate (empty) operand, so each is
    /// raised to `1`. This is defensive: it makes a malformed declaration
    /// harmless rather than trusted.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            min_macs: self.min_macs.max(1),
            min_rows: self.min_rows.max(1),
            min_contraction: self.min_contraction.max(1),
            min_columns: self.min_columns.max(1),
            small_contraction_below: self.small_contraction_below,
            small_contraction_max_output: self.small_contraction_max_output,
        }
    }

    /// Whether this declaration admits an `(m, k, p)` product.
    ///
    /// For [`Self::CONSTANT_FLOOR`] this is exactly the historical
    /// `m.saturating_mul(k).saturating_mul(p) >= SOUND_F64_GEMM_DEFAULT_MIN_MACS`.
    #[must_use]
    pub fn admits(&self, m: usize, k: usize, p: usize) -> bool {
        if m < self.min_rows || k < self.min_contraction || p < self.min_columns {
            return false;
        }
        if m.saturating_mul(k).saturating_mul(p) < self.min_macs {
            return false;
        }
        if k < self.small_contraction_below
            && m.saturating_mul(p) > self.small_contraction_max_output
        {
            return false;
        }
        true
    }
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
    /// Stable, truthful backend identity for runtime telemetry.
    ///
    /// Implementations that do not override this must remain explicitly
    /// unidentified; callers may never infer CUDA/WGPU merely because an
    /// engine object or accelerator-shaped entrypoint exists.
    fn backend_provenance(&self) -> &'static str {
        "unspecified-custom-gemm-engine"
    }

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

    /// Compute a full row-major IEEE-f64 product under an authoritative
    /// wall-clock deadline.
    ///
    /// This is an explicit opt-in contract, deliberately separate from
    /// [`gemm_f64`](Self::gemm_f64). Implementations MUST:
    ///
    /// - poll `deadline` before every resource wait and accelerator launch;
    /// - split only the output `m`/`n` axes, never the contraction `k`;
    /// - cap each non-interruptible dispatch at `max_dispatch_macs`;
    /// - poll between dispatches and after the final synchronization; and
    /// - return [`NyError::DeadlineExceeded`] instead of publishing a result
    ///   after the deadline.
    ///
    /// Keeping the full contraction in each output tile preserves callers'
    /// summation-order-independent `gamma_k * S` certificate. Engines without
    /// a proven bounded-dispatch implementation return `UnsupportedOp`; this
    /// default MUST NOT delegate to the ordinary, potentially unbounded
    /// `gemm_f64`.
    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        deadline: std::time::Instant,
        max_dispatch_macs: usize,
    ) -> Result<Vec<f64>> {
        let _ = (m, k, n, a, b, deadline, max_dispatch_macs);
        Err(NyError::UnsupportedOp(
            "deadline-bounded f64 GEMM not supported by this engine".into(),
        ))
    }

    /// Compute a full row-major IEEE round-to-nearest **f32** product under an
    /// authoritative wall-clock deadline (#fl-value-gpu-tier).
    ///
    /// The f32 sibling of [`gemm_f64_with_deadline`](Self::gemm_f64_with_deadline),
    /// with the identical bounded-dispatch contract. Implementations MUST:
    ///
    /// - poll `deadline` before every resource wait and accelerator launch;
    /// - split only the output `m` axis (never the contraction `k`), so every
    ///   output coefficient remains ONE ordinary length-`k` RN-f32 dot product
    ///   and the caller's summation-order-independent `gamma_{k}^f32 * S`
    ///   certificate (plus FTZ addend) stays valid;
    /// - cap each non-interruptible dispatch at `max_dispatch_macs`;
    /// - poll between dispatches and after the final synchronization;
    /// - validate the result (shape, all-finite) and return a typed error —
    ///   never a value — on any failure; and
    /// - return [`NyError::DeadlineExceeded`] instead of publishing a result
    ///   after the deadline.
    ///
    /// PRECISION CONTRACT: plain IEEE round-to-nearest f32 arithmetic (any
    /// summation order); no TF32 / BF16-split / fast-math. Engines without a
    /// proven bounded-dispatch implementation return `UnsupportedOp`; this
    /// default MUST NOT delegate to the ordinary, potentially unbounded
    /// [`gemm_f32`](Self::gemm_f32) — only engines that explicitly implement
    /// this method participate in deadline-bearing dispatch.
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
        let _ = (m, k, n, a, b, deadline, max_dispatch_macs);
        Err(NyError::UnsupportedOp(
            "deadline-bounded f32 GEMM not supported by this engine".into(),
        ))
    }

    /// This engine's own crossover for the deadline-bounded sound f64 `A·W`
    /// seam — which shapes are worth routing to
    /// [`gemm_f64_with_deadline`](Self::gemm_f64_with_deadline) instead of the
    /// caller's pollable CPU reduction.
    ///
    /// The default is [`SoundF64GemmAdmission::CONSTANT_FLOOR`], i.e. the
    /// historical `m*k*p >= SOUND_F64_GEMM_DEFAULT_MIN_MACS` policy, so an
    /// engine that does not override this keeps byte-identical behaviour.
    /// Override it only with a MEASURED crossover for this backend.
    ///
    /// Callers may consult this only behind their own opt-in gate, and must
    /// treat "no engine available to ask" as "use the historical constant"
    /// (fail closed). Soundness never depends on the answer — see
    /// [`SoundF64GemmAdmission`].
    fn sound_f64_deadline_admission(&self) -> SoundF64GemmAdmission {
        SoundF64GemmAdmission::CONSTANT_FLOOR
    }

    /// Poll a call-local cooperative CROWN deadline between host-side fold
    /// units.
    ///
    /// The default is deliberately inert. A deadline-bearing adapter can
    /// override this without changing the scheduling of ordinary engine calls.
    fn poll_crown_backward_deadline(&self) -> Result<()> {
        Ok(())
    }

    /// Whether a structured host-memory refusal from this engine must be
    /// propagated instead of entering a caller's ordinary local CPU fallback.
    ///
    /// Most accelerators return `false`: their `CpuMemoryExceeded` result may
    /// describe only an optional staging route, and the established local
    /// fallback remains valid. A bounded host facade returns `true` because
    /// that fallback would allocate the very buffer the facade was introduced
    /// to cap, defeating its resource-authority contract.
    fn forbids_unbounded_cpu_fallback(&self) -> bool {
        false
    }

    /// Whether ordinary host GEMM methods are cooperatively bounded by the
    /// same authority exposed through [`Self::poll_crown_backward_deadline`].
    ///
    /// This is a narrow capability for local bounded facades. Accelerators and
    /// ordinary CPU engines must keep the default `false`; callers may invoke
    /// ordinary GEMM under finite authority only when this returns `true`.
    fn provides_deadline_pollable_host_gemm(&self) -> bool {
        false
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

    /// Whether this engine implements the complete, deadline-safe execution
    /// surface required when an already-materialized process engine is handed
    /// to post-root multi-objective graph BaB.
    ///
    /// This is deliberately stricter than exposing [`GpuCrownBackward`]. The
    /// shared executor and its sequential fallbacks may also issue ordinary
    /// verdict-grade [`Self::gemm_f32`] calls. Implementations returning `true`
    /// must therefore guarantee that every supported operation on that surface,
    /// including generic GEMM resource waits and accelerator synchronization,
    /// observes the active graph-BaB deadline. A partial CROWN adapter or an
    /// engine whose generic GEMM can block past that deadline must remain
    /// ineligible. Admission additionally requires the GPU CROWN soundness and
    /// cooperative-deadline capabilities.
    ///
    /// Default `false` keeps every existing engine out of this process-global
    /// handoff unless its concrete implementation has been reviewed for the
    /// complete surface.
    fn supports_deadline_safe_post_root_multi_objective_bab(&self) -> bool {
        false
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
    /// IEEE-754 round-to-nearest GEMM, including backends that flush subnormal
    /// operands/results to zero. Contractions for which `k·u >= 1/2` are
    /// rejected because the closed-form recovery factor is no longer available.
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
        let a_len = checked_dim_product(&[m, k], "gemm_interval_sound lhs")?;
        let b_len = checked_dim_product(&[k, n], "gemm_interval_sound rhs")?;
        let output_len = checked_dim_product(&[m, n], "gemm_interval_sound output")?;
        if a_lo.len() != a_len || a_hi.len() != a_len {
            return Err(NyError::shape_mismatch(vec![m, k], vec![a_lo.len()]));
        }
        if b_lo.len() != b_len || b_hi.len() != b_len {
            return Err(NyError::shape_mismatch(vec![k, n], vec![b_lo.len()]));
        }
        for (label, lo, hi) in [("left", a_lo, a_hi), ("right", b_lo, b_hi)] {
            for (index, (&lower, &upper)) in lo.iter().zip(hi).enumerate() {
                if !lower.is_finite() || !upper.is_finite() || lower > upper {
                    return Err(NyError::NumericalInstability(format!(
                        "gemm_interval_sound {label} interval {index} must have finite ordered endpoints; got [{lower}, {upper}]"
                    )));
                }
            }
        }
        if m == 0 || n == 0 {
            return Ok((vec![], vec![]));
        }
        // The empty contraction is the additive identity, with the ordinary
        // `(m, n)` output shape. Avoid dispatching a zero-width GEMM because not
        // every accelerator backend accepts it.
        if k == 0 {
            return Ok((vec![0.0; output_len], vec![0.0; output_len]));
        }

        // f32 unit roundoff (2^-24) and the running-error growth factor γ_k for a
        // length-k dot product: |fl(x·y) − x·y| ≤ γ_k · (|x|·|y|)  +  additive
        // underflow. The γ_k term is the relative (normalized) error model. The
        // separate amplified floors below cover both gradual underflow and
        // GPU-style DAZ/FTZ, including a subnormal operand multiplied by a large
        // normal operand.
        const ETA: f64 = f64::from_bits(0x36A0_0000_0000_0000); // 2^-149 (min f32 subnormal)
        let gamma_k = sound_dot_gamma(k)?;
        let base_additive = f64::from(ftz_safe_underflow_floor(
            u32::try_from(k).unwrap_or(u32::MAX),
        ));

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
        let row_abs_ma: Vec<f64> = (0..m)
            .map(|i| (0..k).map(|c| f64::from(ma[i * k + c].abs())).sum())
            .collect();
        let row_ra: Vec<f64> = (0..m)
            .map(|i| (0..k).map(|c| f64::from(ra[i * k + c])).sum())
            .collect();
        let col_abs_mb: Vec<f64> = (0..n)
            .map(|j| (0..k).map(|c| f64::from(mb[c * n + j].abs())).sum())
            .collect();
        let col_rb: Vec<f64> = (0..n)
            .map(|j| (0..k).map(|c| f64::from(rb[c * n + j])).sum())
            .collect();

        // Round-to-nearest matmuls (run on this engine's backend).
        let p = self.gemm_f32(m, k, n, &ma, &mb)?; // signed midpoint product
        validate_backend_output_len("gemm_interval_sound midpoint", output_len, p.len())?;
        let abs_p = self.gemm_f32(m, k, n, &abs_ma, &abs_mb)?; // ≥ 0, for γ_k bound
        validate_backend_output_len(
            "gemm_interval_sound absolute midpoint",
            output_len,
            abs_p.len(),
        )?;
        let r1 = self.gemm_f32(m, k, n, &abs_ma, &rb)?; // |ma|·rb
        validate_backend_output_len("gemm_interval_sound radius r1", output_len, r1.len())?;
        let r2 = self.gemm_f32(m, k, n, &ra, &abs_mb)?; // ra·|mb|
        validate_backend_output_len("gemm_interval_sound radius r2", output_len, r2.len())?;
        let r3 = self.gemm_f32(m, k, n, &ra, &rb)?; // ra·rb
        validate_backend_output_len("gemm_interval_sound radius r3", output_len, r3.len())?;

        // For A = ma+δa (|δa| ≤ ra) and B = mb+δb (|δb| ≤ rb):
        //   |A·B − ma·mb|       ≤ r1 + r2 + r3        (interval spread, reals)
        //   |ma·mb − fl(ma·mb)| ≤ γ_k · |ma|·|mb|     (f32 dot rounding, normalized)
        // Each fl(·) matmul under-reports the real value by at most a (1−γ_k)
        // factor, so multiply by 1/(1−γ_k) to recover a real upper bound, add the
        // additive underflow term, then a hair of f64 slack for the host
        // combination itself. If any matmul overflowed to ±inf (so the f32
        // product left representable range), the only sound bound is the trivial
        // [−∞, +∞]; downstream this triggers the usual IBP fallback.
        let real_factor = 1.0 / (1.0 - gamma_k);
        let host_slack = 1.0 + 1e-10;
        let mut c_lo = vec![0.0f32; output_len];
        let mut c_hi = vec![0.0f32; output_len];
        for row in 0..m {
            for col in 0..n {
                let i = row * n + col;
                // A DAZ/FTZ backend may erase a subnormal operand before it is
                // multiplied by a large normal one. For X@Y, the lost magnitude
                // is bounded by
                //   (1 + k + ||X_row||_1 + ||Y_col||_1) * FLT_MIN.
                // Apply that bound separately to every backend GEMM. This is the
                // same separable amplified floor used by crown_aw_error_step.
                let flush = |row_sum: f64, col_sum: f64| {
                    amplified_ftz_floor(
                        base_additive,
                        1.0 + (k as f64) + row_sum + col_sum,
                        host_slack,
                    )
                };
                let midpoint_flush = flush(row_abs_ma[row], col_abs_mb[col]);
                let r1_flush = flush(row_abs_ma[row], col_rb[col]);
                let r2_flush = flush(row_ra[row], col_abs_mb[col]);
                let r3_flush = flush(row_ra[row], col_rb[col]);
                let spread = f64::from(r1[i])
                    + f64::from(r2[i])
                    + f64::from(r3[i])
                    + r1_flush
                    + r2_flush
                    + r3_flush;
                // abs_p can itself under-report under DAZ/FTZ. The outer
                // real_factor recovers its normalized rounding error; the
                // separate midpoint_flush covers the signed midpoint product.
                let round_err = gamma_k * (f64::from(abs_p[i]) + midpoint_flush);
                let radius = (spread + round_err) * real_factor * host_slack + midpoint_flush;
                let pi = f64::from(p[i]);
                if !pi.is_finite() || !radius.is_finite() {
                    c_lo[i] = f32::NEG_INFINITY;
                    c_hi[i] = f32::INFINITY;
                    continue;
                }
                c_lo[i] = round_f32_down(pi - radius);
                c_hi[i] = round_f32_up(pi + radius);
            }
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
    ///   `a_err_new = round_up( γ_k·(|a|@|w|) + (a_err @ |w|) ) + additive`,
    /// with every nonnegative backend product recovered by `1/(1−γ_k)`.
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
        let a_len = checked_dim_product(&[m, k], "crown_aw_error_step lhs")?;
        let w_len = checked_dim_product(&[k, n], "crown_aw_error_step rhs")?;
        let output_len = checked_dim_product(&[m, n], "crown_aw_error_step output")?;
        if a.len() != a_len || a_err.len() != a_len {
            return Err(NyError::shape_mismatch(vec![m, k], vec![a.len()]));
        }
        if w.len() != w_len {
            return Err(NyError::shape_mismatch(vec![k, n], vec![w.len()]));
        }
        if let Some((index, value)) = a
            .iter()
            .chain(w)
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(NyError::NumericalInstability(format!(
                "crown_aw_error_step coefficient/weight {index} must be finite; got {value}"
            )));
        }
        if let Some((index, value)) = a_err
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || *value < 0.0)
        {
            return Err(NyError::NumericalInstability(format!(
                "crown_aw_error_step incoming error {index} must be finite and nonnegative; got {value}"
            )));
        }
        if m == 0 || n == 0 {
            return Ok((vec![], vec![]));
        }
        // An empty contraction is exactly zero and introduces no arithmetic
        // error. Preserve the `(m, n)` result shape without a backend dispatch.
        if k == 0 {
            return Ok((vec![0.0; output_len], vec![0.0; output_len]));
        }
        let gamma_k = sound_dot_gamma(k)?;
        let recovery = 1.0 / (1.0 - gamma_k);
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
        validate_backend_output_len("crown_aw_error_step coefficients", output_len, a_new.len())?;
        let s = self.gemm_f32(m, k, n, &abs_a, &abs_w)?; // |a| @ |w|  (≥ 0)
        validate_backend_output_len("crown_aw_error_step magnitude", output_len, s.len())?;
        let prop = self.gemm_f32(m, k, n, a_err, &abs_w)?; // a_err @ |w|  (≥ 0)
        validate_backend_output_len(
            "crown_aw_error_step propagated error",
            output_len,
            prop.len(),
        )?;

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
        let row_a_err: Vec<f64> = (0..m)
            .map(|i| (0..k).map(|c| f64::from(a_err[i * k + c].abs())).sum())
            .collect();
        let col_abs_w: Vec<f64> = (0..n)
            .map(|j| (0..k).map(|c| f64::from(w[c * n + j].abs())).sum())
            .collect();

        let host_slack = 1.0 + 1e-10;
        let mut a_err_new = vec![0.0f32; output_len];
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                let aw_flush = amplified_ftz_floor(
                    base_additive,
                    1.0 + (k as f64) + row_abs_a[i] + col_abs_w[j],
                    host_slack,
                );
                let prop_flush = amplified_ftz_floor(
                    base_additive,
                    1.0 + (k as f64) + row_a_err[i] + col_abs_w[j],
                    host_slack,
                );
                // round the error UP so [a_new − err, a_new + err] never under-covers.
                // Both nonnegative backend products can under-report by their
                // normalized γ_k error and by DAZ/FTZ. Recover those products
                // before using them, then separately add the signed A@W
                // midpoint's DAZ/FTZ error.
                let e =
                    (gamma_k * (f64::from(s[idx]) + aw_flush) + f64::from(prop[idx]) + prop_flush)
                        * recovery
                        * host_slack
                        + aw_flush;
                a_err_new[idx] = round_f32_up(e);
            }
        }
        Ok((a_new, a_err_new))
    }
}

fn validate_backend_output_len(operation: &str, expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(NyError::InternalError(format!(
            "{operation}: GEMM backend returned {actual} elements, expected {expected}"
        )));
    }
    Ok(())
}

/// Return the standard f32 dot-product growth factor while the callers'
/// `1 / (1 - γ_k)` recovery factor remains finite and positive. Since
/// `γ_k = k·u / (1 - k·u)`, this requires `k·u < 1/2`.
fn sound_dot_gamma(k: usize) -> Result<f64> {
    const U: f64 = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24 exactly
    let ku = (k as f64) * U;
    if !ku.is_finite() || ku >= 0.5 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "sound f32 GEMM roundoff certificate requires contraction dimension k with k*u < 1/2; got k={k}"
        )));
    }
    Ok(ku / (1.0 - ku))
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
    let n = checked_dim_product(
        &[num_outputs, num_neurons],
        "crown_activation_error_step coefficients",
    )?;
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
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > f32::INFINITY.to_bits() || bits == f32::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f32::from_bits(1); // smallest positive subnormal
    }
    f32::from_bits(if bits & 0x8000_0000 == 0 {
        bits + 1
    } else {
        bits - 1
    })
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
// `move` is REQUIRED, not stylistic: this closure captures the parameter `base`, and
// `core::contracts::ensures` (contracts.rs:21) requires the closure to be `'static`.
// Borrowing `base` fails borrowck under real verification with
//   error[E0597]: `base` does not live long enough ... requires that `base` is
//   borrowed for `'static`
// while the stable-rustc no-op facade accepts it, so `cargo build` cannot see the
// break — it surfaces only under `targo trust check`. `f64: Copy`, so moving is free
// and changes nothing at runtime. The sibling contract at :951 compiles without
// `move` only because it references a constant rather than a parameter.
#[ensures(move |f: &f64| *f >= base)]
#[must_use]
fn amplified_ftz_floor(base: f64, flushacc: f64, slack: f64) -> f64 {
    // `.max(0.0)` on both factors makes the added term unconditionally ≥ 0, so the
    // `#[ensures(*f >= base)]` obligation holds for every input (defensive; callers
    // always pass flushacc, slack ≥ 0).
    base + flushacc.max(0.0) * slack.max(0.0) * f64::from(f32::MIN_POSITIVE)
}

/// IEEE-754 predecessor of a finite `f32` (toward −∞).
fn next_down_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > f32::INFINITY.to_bits() || bits == f32::NEG_INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return -f32::from_bits(1); // smallest negative subnormal
    }
    f32::from_bits(if bits & 0x8000_0000 == 0 {
        bits - 1
    } else {
        bits + 1
    })
}

/// Certified per-layer weight error carried INTO the GPU walk.
///
/// Both terms come from a BN-fold (or any pre-fold) the CALLER performed, and
/// neither is representable in the existing exact-weight contract. The sound
/// resident walk otherwise assumes the supplied `weight` / `weight_col` IS the
/// layer's exact real weight and charges only the f32 rounding of its own
/// arithmetic; when the caller folded (say) a BatchNorm into the convolution,
/// the shipped f32 weights are merely an APPROXIMATION of the exact real fold
/// and that discrepancy must be charged too, or the published bound is not an
/// enclosure.
///
/// # Caller's obligation (the definition the kernels charge against)
///
/// For the layer's exact real weight `w*` and bias `b*`, and the supplied
/// `w` / `b`, this type asserts — for EVERY entry —
///
/// * `|w*ᵢⱼ − wᵢⱼ| ≤ weight_rel_err · |wᵢⱼ|`  (RELATIVE, elementwise), and
/// * `|b*ᵢ  − bᵢ|  ≤ bias_abs_err`             (ABSOLUTE, max over outputs).
///
/// The denominator in the weight contract is the SUPPLIED weight `|w|`.
/// Callers whose upstream certificate is relative to an exact or pre-downcast
/// weight must convert that radius before constructing this type; the same
/// numeric radius cannot in general be copied across denominators. For an
/// upstream ball `|w - w*| <= R |w*|` with `R < 1`, the corresponding radius
/// here is at least `R / (1 - R)`.
///
/// # Non-breaking by construction
///
/// [`Default`] is the all-zero value, which IS today's exact-weight contract:
/// a walk whose every layer carries the default charges nothing extra and is
/// byte-identical to the pre-`cert_err` build. That is pinned as a test, not
/// merely asserted — see
/// `zero_cert_err_charges_are_byte_identical` in `ny-gpu`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CertifiedWeightError {
    /// RELATIVE error of the supplied weights vs the exact real fold.
    pub weight_rel_err: f32,
    /// ABSOLUTE per-output bias error bound (max over outputs).
    pub bias_abs_err: f32,
}

/// Next representable f64 above a finite non-negative value (outward step for
/// certified accumulations). `+inf`/NaN pass through; 0.0 steps to the least
/// positive subnormal, which is still an upper bound on any exact 0 result.
#[inline]
fn next_up_f64_positive(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    f64::from_bits(x.to_bits() + 1)
}

impl CertifiedWeightError {
    /// The exact-weight value: both terms zero. Identical to [`Default`], named
    /// so construction sites read as a deliberate assertion of exactness.
    pub const EXACT: Self = Self {
        weight_rel_err: 0.0,
        bias_abs_err: 0.0,
    };

    /// Whether this layer declares NO weight/bias error, i.e. the legacy
    /// exact-weight contract. `-0.0` counts as zero (`-0.0 == 0.0`).
    ///
    /// Every consumer that cannot charge the error must branch on this and
    /// REFUSE (never silently ignore) — see
    /// [`refuse_uncharged_certified_weight_error`].
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.weight_rel_err == 0.0 && self.bias_abs_err == 0.0
    }

    /// Whether the declaration is usable at all: both terms finite and `>= 0`.
    ///
    /// A negative or non-finite declaration is meaningless as a radius, so
    /// charging it could TIGHTEN — callers must refuse instead.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.weight_rel_err.is_finite()
            && self.bias_abs_err.is_finite()
            && self.weight_rel_err >= 0.0
            && self.bias_abs_err >= 0.0
    }

    /// The CHARGED relative coefficient factor for a layer whose own f32
    /// reduction already carries Higham's `gamma`.
    ///
    /// # Derivation (this is the whole soundness argument)
    ///
    /// The walk ships `A_new = fl(A @ W)` with a certified radius `e_new` that
    /// must enclose the EXACT `A* @ W*`, where `A*` is the exact predecessor
    /// coefficient (`|A* − A| ≤ e`) and `W*` the exact real fold
    /// (`|W* − W| ≤ w·|W|` elementwise, `w = weight_rel_err`).
    ///
    /// ```text
    /// |A*@W* − A@W|
    ///     = |Σ (A±e)(W ± w|W|) − Σ A·W|
    ///     ≤ Σ ( e·|W| + w·(|A| + e)·|W| )
    ///     = ( (1+w)·e + w·|A| ) @ |W|
    /// |fl(A@W) − A@W| ≤ gamma · (|A| @ |W|)          (Higham, length-k reduction)
    /// ⇒ |A*@W* − fl(A@W)| ≤ ( (gamma + w)·|A| + (1+w)·e ) @ |W|
    /// ```
    ///
    /// `g = gamma + w + gamma·w` satisfies `g ≥ gamma + w`, so charging `g` in
    /// place of `gamma` covers the coefficient term, and the extra `gamma·w`
    /// cross term absorbs the rounding of forming `g` itself. The propagated
    /// error term needs its own `(1 + w)` factor, which the kernel host applies
    /// to the combine's `slack` (see `cert_charged_slack` in `ny-gpu`); the two
    /// substitutions together dominate the bound above term by term.
    ///
    /// This is the composition the CPU margin-row lane uses
    /// (`margin_row/engine.rs`, the `TwinOp::Conv` arm, where the same `g` meets
    /// `comb = e + g·(|A| + e)`), so a GPU seam charged this way is directly
    /// comparable to — and never looser in KIND than — the lane it replaces.
    ///
    /// # Direction
    ///
    /// The arithmetic is done in f64 (exact for the two f32 inputs and their
    /// product; only the two adds round) and the f32 narrowing rounds UP, so
    /// the result is always `>= ` the real `gamma + w + gamma·w`. A non-finite
    /// outcome saturates to `+∞` rather than wrapping to a finite under-charge;
    /// callers must treat `+∞` as a refusal, since no finite radius follows.
    #[must_use]
    pub fn charged_gamma(&self, gamma: f32) -> f32 {
        if !self.is_valid() || !gamma.is_finite() || gamma < 0.0 {
            return f32::INFINITY;
        }
        // Review defect 1: the exact-weight case must return `gamma` UNCHANGED
        // (the byte-identity contract), and every intermediate of the charged
        // case must round OUTWARD. `g + w` rounds to NEAREST, so for
        // `w << ulp(g)/2` the sum is exactly `g`, which is f32-representable —
        // `f64_to_f32_up` then does not bump and the shipped value violates the
        // required `>= gamma + w`. Step each accumulation up explicitly.
        if self.is_exact() {
            return gamma;
        }
        let g = crate::floating_point::f32_to_f64_exact(gamma);
        let w = crate::floating_point::f32_to_f64_exact(self.weight_rel_err);
        let sum = next_up_f64_positive(next_up_f64_positive(g + w) + g * w);
        let charged = crate::floating_point::f64_to_f32_up(sum);
        if charged.is_finite() {
            charged
        } else {
            f32::INFINITY
        }
    }
}

/// Fail-closed guard for every consumer of [`GpuCrownLayer`] that does NOT
/// implement the [`CertifiedWeightError`] charge.
///
/// Adding `cert_err` to the layer descriptors is non-breaking for CONSTRUCTION
/// (the default is the old exact-weight contract), but it is emphatically NOT
/// non-breaking for CONSUMPTION: a sound backward that ignores a nonzero
/// `cert_err` publishes a radius that omits a real term, i.e. a bound that can
/// sit BELOW the truth. Every sound route that has not been taught the charge
/// therefore calls this at entry and refuses.
///
/// Advisory-only consumers (α/β gradients, point VJPs, attack forwards) carry
/// no verdict authority and are exempt; they steer search, and any steering is
/// sound.
pub fn refuse_uncharged_certified_weight_error(
    layers: &[GpuCrownLayer],
    route: &str,
) -> Result<()> {
    for (index, layer) in layers.iter().enumerate() {
        let cert_err = match layer {
            GpuCrownLayer::Linear { cert_err, .. } | GpuCrownLayer::Conv2d { cert_err, .. } => {
                cert_err
            }
            _ => continue,
        };
        if !cert_err.is_exact() {
            return Err(NyError::UnsupportedOp(format!(
                "{route}: layer {index} declares a certified weight error \
                 (weight_rel_err={:e}, bias_abs_err={:e}) that this route does \
                 not charge; refusing rather than publishing a radius that \
                 omits it (fail-closed)",
                cert_err.weight_rel_err, cert_err.bias_abs_err
            )));
        }
    }
    Ok(())
}

/// [`refuse_uncharged_certified_weight_error`] over a resnet decomposition.
pub fn refuse_uncharged_certified_weight_error_segments(
    segments: &[GpuResnetSegment],
    route: &str,
) -> Result<()> {
    for segment in segments {
        match segment {
            GpuResnetSegment::Chain(layers) | GpuResnetSegment::Residual(layers) => {
                refuse_uncharged_certified_weight_error(layers, route)?;
            }
            GpuResnetSegment::ResidualProj(f_branch, p_branch) => {
                refuse_uncharged_certified_weight_error(f_branch, route)?;
                refuse_uncharged_certified_weight_error(p_branch, route)?;
            }
        }
    }
    Ok(())
}

/// Certified affine coefficients + their error, as published by a sound GPU
/// CROWN walk. Row-major `[num_specs x dim]`.
///
/// This is the EGRESS the concretized [`GpuCrownResult`] cannot express. The
/// margin-row lane's hot step consumes COEFFICIENTS (it folds them onward
/// through `conv_apply_backward`), so a GPU entry that only ever returns
/// concretized bounds is structurally unseamable there. Publishing the frontier
/// instead lets the lane keep its own composition and merely accelerate the
/// walk that produced it.
///
/// # Soundness contract
///
/// For every spec row `s` and input coordinate `j`, the exact real coefficient
/// `a*` of the relaxation this walk represents satisfies
/// `lower_a[s,j] − lower_a_err[s,j] ≤ a* ≤ lower_a[s,j] + lower_a_err[s,j]`
/// (and likewise for the upper side and the biases). The error arrays are
/// NON-NEGATIVE radii, never signed corrections. Concretizing
/// `(lower_a ± lower_a_err, lower_b ± lower_b_err)` over the input box with
/// outward rounding must reproduce a bound no tighter than the corresponding
/// bounds entry — pinned by `coeffs_entry_concretizes_to_the_bounds_entry`.
#[derive(Debug, Clone)]
pub struct CertifiedCoeffs {
    /// Lower-bound coefficient centres, `[num_specs x dim]` row-major.
    pub lower_a: Vec<f32>,
    /// Upper-bound coefficient centres, `[num_specs x dim]` row-major.
    pub upper_a: Vec<f32>,
    /// Non-negative radii for `lower_a`, same shape.
    pub lower_a_err: Vec<f32>,
    /// Non-negative radii for `upper_a`, same shape.
    pub upper_a_err: Vec<f32>,
    /// Lower-bound bias centres, one per spec row.
    pub lower_b: Vec<f32>,
    /// Upper-bound bias centres, one per spec row.
    pub upper_b: Vec<f32>,
    /// Non-negative radii for `lower_b`, one per spec row.
    pub lower_b_err: Vec<f32>,
    /// Non-negative radii for `upper_b`, one per spec row.
    pub upper_b_err: Vec<f32>,
    /// Number of specification rows.
    pub num_specs: usize,
    /// Width of the coefficient frontier (the walk's final dimension).
    pub dim: usize,
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
        /// Certified error of `weight`/`bias` vs the exact real fold the caller
        /// performed. [`CertifiedWeightError::default()`] (all zeros) is the
        /// legacy exact-weight contract and charges nothing.
        cert_err: CertifiedWeightError,
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
        /// Certified error of `weight_col`/`bias_expanded` vs the exact real
        /// fold the caller performed. [`CertifiedWeightError::default()`] (all
        /// zeros) is the legacy exact-weight contract and charges nothing.
        cert_err: CertifiedWeightError,
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
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuCrownResult {
    /// Lower bounds per specification row
    pub lower_bounds: Vec<f32>,
    /// Upper bounds per specification row
    pub upper_bounds: Vec<f32>,
}

/// Architecture-neutral host validation ceiling for sweep graph slots.
///
/// Device-specific capacity remains dynamic; this only prevents malformed
/// requests from turning validation itself into an unbounded allocation.
pub const GPU_INTERMEDIATE_SWEEP_MAX_SLOTS: usize = 1 << 20;

/// Architecture-neutral host validation ceiling for sweep operations.
pub const GPU_INTERMEDIATE_SWEEP_MAX_OPS: usize = 1 << 20;

/// Architecture-neutral host validation ceiling for injected targets.
pub const GPU_INTERMEDIATE_SWEEP_MAX_TARGETS: usize = 1 << 16;

/// Architecture-neutral host validation ceiling for aggregate selected rows.
/// Backends still apply their live adapter/device-memory limits below this cap.
pub const GPU_INTERMEDIATE_SWEEP_MAX_ROWS: usize = 1 << 24;

/// Architecture-neutral rank ceiling for one injected target tensor.
/// Ordinary model tensors are far below this value; the cap only bounds the
/// amount of host work a malformed descriptor can force before dispatch.
pub const GPU_INTERMEDIATE_SWEEP_MAX_TARGET_RANK: usize = 64;

const GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE: usize = 4096;

/// Dense identifier for one forward-graph frontier in a canonical reverse-DAG
/// CROWN sweep.
///
/// Slots are numbered in reverse topological order: every backward edge goes
/// from a smaller slot to a larger slot, and the network input is the final
/// (largest) slot. [`GpuIntermediateSweepPlan::slot_dims`] is indexed directly
/// by this value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBackwardSlot(pub u32);

impl GpuBackwardSlot {
    /// Convert this slot to its dense host index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One forward-graph operation in a canonical GPU backward sweep.
///
/// The names describe the forward graph. Backward execution consumes the
/// coefficient frontier at `output` and contributes it to `input`, `lhs`, or
/// `rhs`. Consequently every referenced input slot must be strictly greater
/// than `output`. Multiple operations may contribute to the same later slot;
/// the backend must add those contributions before consuming that slot.
#[derive(Clone)]
pub enum GpuBackwardOp {
    /// Fold one ordinary CROWN layer from its forward output to its input.
    Unary {
        output: GpuBackwardSlot,
        input: GpuBackwardSlot,
        layer: Box<GpuCrownLayer>,
    },
    /// Copy the output frontier to the input frontier unchanged.
    Identity {
        output: GpuBackwardSlot,
        input: GpuBackwardSlot,
    },
    /// Reverse an elementwise `output = lhs + rhs` fan-out.
    Add {
        output: GpuBackwardSlot,
        lhs: GpuBackwardSlot,
        rhs: GpuBackwardSlot,
    },
    /// Reverse an elementwise `output = lhs - rhs` fan-out.
    Sub {
        output: GpuBackwardSlot,
        lhs: GpuBackwardSlot,
        rhs: GpuBackwardSlot,
    },
}

impl GpuBackwardOp {
    #[must_use]
    fn output(&self) -> GpuBackwardSlot {
        match self {
            Self::Unary { output, .. }
            | Self::Identity { output, .. }
            | Self::Add { output, .. }
            | Self::Sub { output, .. } => *output,
        }
    }

    fn inputs(&self) -> impl Iterator<Item = GpuBackwardSlot> + '_ {
        let inputs = match self {
            Self::Unary { input, .. } | Self::Identity { input, .. } => [Some(*input), None],
            Self::Add { lhs, rhs, .. } | Self::Sub { lhs, rhs, .. } => [Some(*lhs), Some(*rhs)],
        };
        inputs.into_iter().flatten()
    }
}

/// One selected intermediate target injected into a shared backward sweep.
///
/// `selected_rows` are flattened row indices in `target_shape`. They are
/// strictly increasing. `row_offset` is the first row in the transaction-wide
/// carrier; canonical injections have contiguous prefix-sum offsets. The
/// backend adds these identity rows at `slot` before folding the operation
/// whose forward output is that slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuIntermediateInjection {
    /// Stable, caller-assigned identity. IDs are unique within the plan.
    pub target_id: u64,
    /// Forward-graph frontier at which the selected identity rows are injected.
    pub slot: GpuBackwardSlot,
    /// Complete forward target shape; its checked product equals the slot dim.
    pub target_shape: Arc<[usize]>,
    /// Strictly increasing flattened indices selected from `target_shape`.
    pub selected_rows: Arc<[u32]>,
    /// Contiguous offset of this target in the shared row carrier.
    pub row_offset: usize,
}

/// Backend-neutral plan for one sound, GPU-resident, multi-depth CROWN sweep.
///
/// Operations use forward-graph names but are stored in canonical backward
/// traversal order: their consumed `output` slots are strictly increasing and
/// every produced input slot is larger. Injections are strictly ordered by
/// `(slot, target_id)`. This makes row association independent of incidental
/// caller collection order. Transcript identities are opaque caller-supplied
/// SHA-256 values: core validates exact echoes but does not recompute hashes of
/// backend-neutral descriptors.
///
/// The operation stream is pruned to the live ancestors of the selected
/// injections: every listed operation must receive injected or propagated rows,
/// and every such live frontier must eventually reach `input_slot`. At an
/// Add/Sub convergence, validation tracks reachability while the backend must
/// retain and algebraically combine every actual row contribution.
#[derive(Clone)]
pub struct GpuIntermediateSweepPlan {
    /// Caller-supplied identity of the immutable graph transcript.
    pub graph_identity_sha256: [u8; 32],
    /// Caller-supplied identity of the intermediate bound/relaxation transcript.
    pub bounds_identity_sha256: [u8; 32],
    /// Caller-supplied identity of the canonical target/injection set.
    pub target_set_identity_sha256: [u8; 32],
    /// Reverse-topological operation stream.
    pub ops_backward: Arc<[GpuBackwardOp]>,
    /// Dense slot dimensions, indexed by [`GpuBackwardSlot`].
    pub slot_dims: Arc<[usize]>,
    /// Network input slot. It must be the final/largest dense slot.
    pub input_slot: GpuBackwardSlot,
    /// Canonically ordered intermediate identity-row injections.
    pub injections: Arc<[GpuIntermediateInjection]>,
    /// Exact checked sum of all `injections[*].selected_rows.len()` values.
    pub total_rows: usize,
}

impl GpuIntermediateSweepPlan {
    /// Validate canonical topology, dimensions, target identities, and row
    /// association before a backend may accept this plan.
    pub fn validate(&self) -> Result<()> {
        self.validate_with_deadline(None)
    }

    fn validate_with_deadline(&self, deadline: Option<std::time::Instant>) -> Result<()> {
        intermediate_sweep_check_deadline(deadline)?;
        if self.slot_dims.is_empty() {
            return Err(intermediate_sweep_invalid("slot_dims must not be empty"));
        }
        if self.slot_dims.len() > GPU_INTERMEDIATE_SWEEP_MAX_SLOTS {
            return Err(intermediate_sweep_invalid(format!(
                "slot count {} exceeds host validation cap {GPU_INTERMEDIATE_SWEEP_MAX_SLOTS}",
                self.slot_dims.len()
            )));
        }
        if self.ops_backward.len() > GPU_INTERMEDIATE_SWEEP_MAX_OPS {
            return Err(intermediate_sweep_invalid(format!(
                "operation count {} exceeds host validation cap {GPU_INTERMEDIATE_SWEEP_MAX_OPS}",
                self.ops_backward.len()
            )));
        }
        if self.injections.len() > GPU_INTERMEDIATE_SWEEP_MAX_TARGETS {
            return Err(intermediate_sweep_invalid(format!(
                "target count {} exceeds host validation cap {GPU_INTERMEDIATE_SWEEP_MAX_TARGETS}",
                self.injections.len()
            )));
        }
        let final_slot = self.slot_dims.len() - 1;
        let final_slot_u32 = u32::try_from(final_slot).map_err(|_| {
            intermediate_sweep_invalid("slot count cannot be represented by GpuBackwardSlot")
        })?;
        if self.input_slot != GpuBackwardSlot(final_slot_u32) {
            return Err(intermediate_sweep_invalid(format!(
                "input slot {} must be final slot {final_slot}",
                self.input_slot.index()
            )));
        }
        for (slot, &dim) in self.slot_dims.iter().enumerate() {
            if slot.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
                intermediate_sweep_check_deadline(deadline)?;
            }
            if dim == 0 {
                return Err(intermediate_sweep_invalid(format!(
                    "slot {slot} has zero dimension"
                )));
            }
        }

        if self.injections.is_empty() {
            return Err(intermediate_sweep_invalid(
                "at least one intermediate injection is required",
            ));
        }

        let mut pending = vec![false; self.slot_dims.len()];
        let mut touched = vec![false; self.slot_dims.len()];
        let mut target_ids = HashSet::with_capacity(self.injections.len());
        let mut expected_row_offset = 0usize;
        let mut previous_injection_key = None;
        for (index, injection) in self.injections.iter().enumerate() {
            if index.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
                intermediate_sweep_check_deadline(deadline)?;
            }
            let slot = self.checked_slot(injection.slot, "injection")?;
            if injection.slot == self.input_slot {
                return Err(intermediate_sweep_invalid(format!(
                    "target {} injects at network input; direct-input identities are outside the intermediate sweep contract",
                    injection.target_id
                )));
            }
            touched[slot] = true;
            let key = (injection.slot, injection.target_id);
            if previous_injection_key.is_some_and(|previous| previous >= key) {
                return Err(intermediate_sweep_invalid(format!(
                    "injection {index} key ({}, {}) is not strictly after the previous key",
                    injection.slot.index(),
                    injection.target_id
                )));
            }
            previous_injection_key = Some(key);
            if !target_ids.insert(injection.target_id) {
                return Err(intermediate_sweep_invalid(format!(
                    "target ID {} is duplicated",
                    injection.target_id
                )));
            }

            if injection.row_offset != expected_row_offset {
                return Err(intermediate_sweep_invalid(format!(
                    "target {} row offset {} != expected contiguous offset {expected_row_offset}",
                    injection.target_id, injection.row_offset
                )));
            }
            if injection.target_shape.is_empty()
                || injection.target_shape.len() > GPU_INTERMEDIATE_SWEEP_MAX_TARGET_RANK
                || injection.target_shape.contains(&0)
            {
                return Err(intermediate_sweep_invalid(format!(
                    "target {} shape rank must be in 1..={GPU_INTERMEDIATE_SWEEP_MAX_TARGET_RANK} and contain only nonzero dimensions",
                    injection.target_id,
                )));
            }
            let target_dim = checked_dim_product(
                &injection.target_shape,
                "GPU intermediate sweep target shape",
            )?;
            if target_dim != self.slot_dims[slot] {
                return Err(intermediate_sweep_invalid(format!(
                    "target {} shape product {target_dim} != slot {slot} dimension {}",
                    injection.target_id, self.slot_dims[slot]
                )));
            }
            if injection.selected_rows.is_empty() {
                return Err(intermediate_sweep_invalid(format!(
                    "target {} selects no rows",
                    injection.target_id
                )));
            }
            let mut previous_row = None;
            for (row_index, &row) in injection.selected_rows.iter().enumerate() {
                if row_index.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
                    intermediate_sweep_check_deadline(deadline)?;
                }
                let row = row as usize;
                if row >= target_dim {
                    return Err(intermediate_sweep_invalid(format!(
                        "target {} selected row {row} is outside dimension {target_dim}",
                        injection.target_id
                    )));
                }
                if previous_row.is_some_and(|previous| previous >= row) {
                    return Err(intermediate_sweep_invalid(format!(
                        "target {} selected rows are not strictly increasing",
                        injection.target_id
                    )));
                }
                previous_row = Some(row);
            }
            expected_row_offset = expected_row_offset
                .checked_add(injection.selected_rows.len())
                .ok_or_else(|| intermediate_sweep_invalid("total selected rows overflow usize"))?;
            if expected_row_offset > GPU_INTERMEDIATE_SWEEP_MAX_ROWS {
                return Err(intermediate_sweep_invalid(format!(
                    "selected row count {expected_row_offset} exceeds host validation cap \
                     {GPU_INTERMEDIATE_SWEEP_MAX_ROWS}"
                )));
            }
            pending[slot] = true;
        }
        if expected_row_offset != self.total_rows {
            return Err(intermediate_sweep_invalid(format!(
                "declared total rows {} != checked injection total {expected_row_offset}",
                self.total_rows
            )));
        }
        self.total_rows
            .checked_mul(2)
            .and_then(|endpoints| endpoints.checked_mul(size_of::<f32>()))
            .ok_or_else(|| {
                intermediate_sweep_invalid("result endpoint byte count overflows usize")
            })?;

        let mut previous_output = None;
        for (index, op) in self.ops_backward.iter().enumerate() {
            if index.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
                intermediate_sweep_check_deadline(deadline)?;
            }
            let output = op.output();
            let output_index = self.checked_slot(output, "operation output")?;
            touched[output_index] = true;
            if previous_output.is_some_and(|previous| previous >= output) {
                return Err(intermediate_sweep_invalid(format!(
                    "operation {index} output slot {output_index} is not strictly increasing"
                )));
            }
            previous_output = Some(output);
            if !pending[output_index] {
                return Err(intermediate_sweep_invalid(format!(
                    "operation {index} output slot {output_index} has no pending injected or propagated rows"
                )));
            }

            for input in op.inputs() {
                let input_index = self.checked_slot(input, "operation input")?;
                touched[input_index] = true;
                if output >= input {
                    return Err(intermediate_sweep_invalid(format!(
                        "operation {index} edge {output_index}->{input_index} is not forward-slot increasing"
                    )));
                }
            }
            self.validate_op_dims(index, op, deadline)?;

            pending[output_index] = false;
            for input in op.inputs() {
                pending[input.index()] = true;
            }
        }

        if let Some(unused) = touched.iter().position(|used| !used) {
            return Err(intermediate_sweep_invalid(format!(
                "dense canonical slot {unused} is unused"
            )));
        }

        for (slot, &is_pending) in pending.iter().enumerate() {
            if is_pending && slot != self.input_slot.index() {
                return Err(intermediate_sweep_invalid(format!(
                    "slot {slot} retains an unfurled coefficient frontier"
                )));
            }
        }
        if !pending[self.input_slot.index()] {
            return Err(intermediate_sweep_invalid(
                "the reverse sweep does not reach input_slot",
            ));
        }
        intermediate_sweep_check_deadline(deadline)
    }

    fn checked_slot(&self, slot: GpuBackwardSlot, role: &str) -> Result<usize> {
        let index = slot.index();
        if index >= self.slot_dims.len() {
            return Err(intermediate_sweep_invalid(format!(
                "{role} slot {index} is outside {} dense slots",
                self.slot_dims.len()
            )));
        }
        Ok(index)
    }

    fn validate_op_dims(
        &self,
        index: usize,
        op: &GpuBackwardOp,
        deadline: Option<std::time::Instant>,
    ) -> Result<()> {
        let output = op.output().index();
        let output_dim = self.slot_dims[output];
        match op {
            GpuBackwardOp::Unary { input, layer, .. } => {
                let input_dim = self.slot_dims[input.index()];
                validate_intermediate_unary_layer(index, layer, output_dim, input_dim, deadline)
            }
            GpuBackwardOp::Identity { input, .. } => require_intermediate_dim_eq(
                index,
                "identity input",
                output_dim,
                self.slot_dims[input.index()],
            ),
            GpuBackwardOp::Add { lhs, rhs, .. } | GpuBackwardOp::Sub { lhs, rhs, .. } => {
                require_intermediate_dim_eq(
                    index,
                    "binary lhs",
                    output_dim,
                    self.slot_dims[lhs.index()],
                )?;
                require_intermediate_dim_eq(
                    index,
                    "binary rhs",
                    output_dim,
                    self.slot_dims[rhs.index()],
                )
            }
        }
    }
}

/// Backend-recommended host scheduling policy for one intermediate sweep.
///
/// This is a capacity hint from an already-created, retained device. It grants
/// no numerical or verdict authority: callers must still require both sound
/// capability predicates, and the backend must still preflight the exact typed
/// request before accepting it. The byte value is an accountable-buffer
/// ceiling, while the row values let a host start near the device's useful
/// operating point and bound whole-request downshifts after clean declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuIntermediateSweepResourcePolicy {
    /// Recommended ceiling for all explicitly accountable live device buffers.
    pub max_device_bytes: usize,
    /// Initial identity-row ceiling for each target in a comprehensive sweep.
    pub preferred_rows_per_target: usize,
    /// Smallest useful per-target row ceiling for an atomic retry.
    pub minimum_rows_per_target: usize,
}

impl GpuIntermediateSweepResourcePolicy {
    /// Whether all fields form a usable bounded scheduling policy.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.max_device_bytes > 0
            && self.minimum_rows_per_target > 0
            && self.minimum_rows_per_target <= self.preferred_rows_per_target
            && self.preferred_rows_per_target <= GPU_INTERMEDIATE_SWEEP_MAX_ROWS
    }
}

/// Borrowed operands and call-local resource authority for one sweep.
#[derive(Clone, Copy)]
pub struct GpuIntermediateSweepRequest<'a> {
    pub plan: &'a GpuIntermediateSweepPlan,
    /// Caller-supplied identity of the exact input endpoint bits.
    pub input_identity_sha256: [u8; 32],
    /// Finite lower input endpoints, one per `plan.input_slot` dimension.
    pub input_lower: &'a [f32],
    /// Finite upper input endpoints, one per `plan.input_slot` dimension.
    pub input_upper: &'a [f32],
    /// Absolute call-local deadline. A backend must never publish a late result.
    pub deadline: std::time::Instant,
    /// Nonzero caller-authorized ceiling for the backend's explicitly
    /// accountable numerical buffers: the request working set, queued upload
    /// staging, and retained data caches that can remain live during this
    /// transaction. Opaque driver/compiler objects and allocations made by
    /// unrelated callers through a raw device handle are outside this logical
    /// byte receipt; any pressure they create must still fail the transaction
    /// without returning a partial result.
    pub max_device_bytes: usize,
}

impl GpuIntermediateSweepRequest<'_> {
    /// Validate the complete request before any allocation, wait, or dispatch.
    ///
    /// Time is sampled internally so a caller cannot make a late request appear
    /// live by reusing a pre-dispatch timestamp.
    pub fn validate(&self) -> Result<()> {
        intermediate_sweep_check_deadline(Some(self.deadline))?;
        if self.max_device_bytes == 0 {
            return Err(intermediate_sweep_invalid(
                "max_device_bytes must be nonzero",
            ));
        }
        self.plan.validate_with_deadline(Some(self.deadline))?;
        let input_dim = self.plan.slot_dims[self.plan.input_slot.index()];
        if self.input_lower.len() != input_dim || self.input_upper.len() != input_dim {
            return Err(intermediate_sweep_invalid(format!(
                "input box lengths ({}, {}) != input slot dimension {input_dim}",
                self.input_lower.len(),
                self.input_upper.len()
            )));
        }
        for (index, (&lower, &upper)) in self
            .input_lower
            .iter()
            .zip(self.input_upper.iter())
            .enumerate()
        {
            if index.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
                intermediate_sweep_check_deadline(Some(self.deadline))?;
            }
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(intermediate_sweep_invalid(format!(
                    "input box element {index} is not a finite ordered interval"
                )));
            }
        }
        intermediate_sweep_check_deadline(Some(self.deadline))
    }
}

/// Bounds returned for exactly one injected target.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuIntermediateTargetResult {
    /// Exact echo of [`GpuIntermediateInjection::target_id`].
    pub target_id: u64,
    /// Exact echo of [`GpuIntermediateInjection::row_offset`].
    pub row_offset: usize,
    /// Exact echo of [`GpuIntermediateInjection::selected_rows`].
    pub selected_rows: Arc<[u32]>,
    /// Finite lower bounds in `selected_rows` order.
    pub lower_bounds: Vec<f32>,
    /// Finite upper bounds in `selected_rows` order.
    pub upper_bounds: Vec<f32>,
}

/// Auditable resource receipt for one completed GPU sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuIntermediateSweepReceipt {
    /// Exact echo of [`GpuIntermediateSweepPlan::graph_identity_sha256`].
    pub graph_identity_sha256: [u8; 32],
    /// Exact echo of [`GpuIntermediateSweepRequest::input_identity_sha256`].
    pub input_identity_sha256: [u8; 32],
    /// Exact echo of [`GpuIntermediateSweepPlan::bounds_identity_sha256`].
    pub bounds_identity_sha256: [u8; 32],
    /// Exact echo of [`GpuIntermediateSweepPlan::target_set_identity_sha256`].
    pub target_set_identity_sha256: [u8; 32],
    /// Number of target descriptors accepted from the request.
    pub requested_targets: usize,
    /// Number of target descriptors completed in this atomic result.
    pub completed_targets: usize,
    /// Number of selected rows accepted from the request.
    pub requested_rows: usize,
    /// Number of selected rows completed in this atomic result.
    pub completed_rows: usize,
    /// Peak simultaneously live accountable numerical-buffer bytes, never
    /// above the request cap.
    pub peak_device_bytes: usize,
    /// Number of accelerator dispatches submitted by the accepted request.
    pub dispatches: usize,
    /// Host-to-device payload bytes transferred for this request.
    pub host_to_device_bytes: usize,
    /// Device-to-host payload bytes transferred for this request.
    pub device_to_host_bytes: usize,
    /// Number of explicit device readbacks.
    pub readbacks: usize,
    /// Number of command-buffer/queue submissions.
    pub submits: usize,
    /// Number of host-visible device synchronizations.
    pub synchronizations: usize,
    /// Number of bounded scheduling waves used by those dispatches.
    pub waves: usize,
}

/// Atomic result of one sound intermediate sweep.
///
/// A backend must return every requested target in canonical injection order or
/// return `Err`; it must never return a successful prefix. Its payload is kept
/// private until consuming [`Self::validate`] returns a
/// [`ValidatedGpuIntermediateSweepResult`]. Consequently one malformed,
/// missing, late, or mis-associated target invalidates the whole transaction.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuIntermediateSweepResult {
    targets: Vec<GpuIntermediateTargetResult>,
    receipt: GpuIntermediateSweepReceipt,
}

impl GpuIntermediateSweepResult {
    /// Construct an opaque, unvalidated backend result.
    ///
    /// No target can be read through this type. A caller must consume it with
    /// [`Self::validate`] before any interval becomes publishable.
    #[must_use]
    pub fn new_unvalidated(
        targets: Vec<GpuIntermediateTargetResult>,
        receipt: GpuIntermediateSweepReceipt,
    ) -> Self {
        Self { targets, receipt }
    }

    /// Atomically validate exact request/result association, all intervals,
    /// the live deadline, and the device-memory receipt.
    pub fn validate(
        self,
        request: &GpuIntermediateSweepRequest<'_>,
    ) -> Result<ValidatedGpuIntermediateSweepResult> {
        request.validate()?;
        let expected_targets = request.plan.injections.len();
        let expected_rows = request.plan.total_rows;
        if self.receipt.graph_identity_sha256 != request.plan.graph_identity_sha256
            || self.receipt.input_identity_sha256 != request.input_identity_sha256
            || self.receipt.bounds_identity_sha256 != request.plan.bounds_identity_sha256
            || self.receipt.target_set_identity_sha256 != request.plan.target_set_identity_sha256
        {
            return Err(intermediate_sweep_invalid(
                "receipt transcript identities do not exactly echo the request",
            ));
        }
        if self.receipt.requested_targets != expected_targets
            || self.receipt.completed_targets != expected_targets
            || self.receipt.requested_rows != expected_rows
            || self.receipt.completed_rows != expected_rows
        {
            return Err(intermediate_sweep_invalid(format!(
                "receipt requested/completed counts do not exactly match {expected_targets} targets and {expected_rows} rows"
            )));
        }
        if self.receipt.peak_device_bytes == 0
            || self.receipt.peak_device_bytes > request.max_device_bytes
        {
            return Err(intermediate_sweep_invalid(format!(
                "receipt peak device bytes {} must be in 1..={}",
                self.receipt.peak_device_bytes, request.max_device_bytes
            )));
        }
        let minimum_readback_bytes = expected_rows
            .checked_mul(2)
            .and_then(|endpoints| endpoints.checked_mul(size_of::<f32>()))
            .ok_or_else(|| intermediate_sweep_invalid("result readback bytes overflow usize"))?;
        if self.receipt.dispatches == 0
            || self.receipt.waves == 0
            || self.receipt.readbacks == 0
            || self.receipt.submits == 0
            || self.receipt.synchronizations == 0
            || self.receipt.device_to_host_bytes < minimum_readback_bytes
        {
            return Err(intermediate_sweep_invalid(format!(
                "successful receipt must record nonzero GPU work and at least {minimum_readback_bytes} result readback bytes"
            )));
        }
        if self.targets.len() != expected_targets {
            return Err(intermediate_sweep_invalid(format!(
                "result target count {} != requested {}",
                self.targets.len(),
                request.plan.injections.len()
            )));
        }

        let mut validated_rows = 0usize;
        for (index, (target, injection)) in self
            .targets
            .iter()
            .zip(request.plan.injections.iter())
            .enumerate()
        {
            if index.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
                intermediate_sweep_check_deadline(Some(request.deadline))?;
            }
            if target.target_id != injection.target_id
                || target.row_offset != injection.row_offset
                || target.selected_rows != injection.selected_rows
            {
                return Err(intermediate_sweep_invalid(format!(
                    "result target {index} does not exactly echo its requested identity, offset, and rows"
                )));
            }
            let row_count = injection.selected_rows.len();
            if target.lower_bounds.len() != row_count || target.upper_bounds.len() != row_count {
                return Err(intermediate_sweep_invalid(format!(
                    "result target {} bound lengths ({}, {}) != selected row count {row_count}",
                    target.target_id,
                    target.lower_bounds.len(),
                    target.upper_bounds.len()
                )));
            }
            for (row, (&lower, &upper)) in target
                .lower_bounds
                .iter()
                .zip(target.upper_bounds.iter())
                .enumerate()
            {
                if row.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
                    intermediate_sweep_check_deadline(Some(request.deadline))?;
                }
                if !lower.is_finite() || !upper.is_finite() || lower > upper {
                    return Err(intermediate_sweep_invalid(format!(
                        "result target {} row {row} is not a finite ordered interval",
                        target.target_id
                    )));
                }
            }
            validated_rows = validated_rows.checked_add(row_count).ok_or_else(|| {
                intermediate_sweep_invalid("validated result rows overflow usize")
            })?;
        }
        if validated_rows != request.plan.total_rows {
            return Err(intermediate_sweep_invalid(format!(
                "validated result rows {validated_rows} != planned {}",
                request.plan.total_rows
            )));
        }
        intermediate_sweep_check_deadline(Some(request.deadline))?;
        Ok(ValidatedGpuIntermediateSweepResult {
            targets: self.targets,
            receipt: self.receipt,
        })
    }
}

/// Completely validated, deadline-live result whose targets may be published.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedGpuIntermediateSweepResult {
    targets: Vec<GpuIntermediateTargetResult>,
    receipt: GpuIntermediateSweepReceipt,
}

impl ValidatedGpuIntermediateSweepResult {
    /// Canonically ordered, completely validated target bounds.
    #[must_use]
    pub fn targets(&self) -> &[GpuIntermediateTargetResult] {
        &self.targets
    }

    /// Validated resource and association receipt.
    #[must_use]
    pub fn receipt(&self) -> &GpuIntermediateSweepReceipt {
        &self.receipt
    }

    /// Consume the validated wrapper into its atomic payload and receipt.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<GpuIntermediateTargetResult>,
        GpuIntermediateSweepReceipt,
    ) {
        (self.targets, self.receipt)
    }
}

fn intermediate_sweep_invalid(message: impl Into<String>) -> NyError {
    NyError::InvalidSpec(format!("GPU intermediate CROWN sweep: {}", message.into()))
}

fn intermediate_sweep_check_deadline(deadline: Option<std::time::Instant>) -> Result<()> {
    if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
        return Err(NyError::DeadlineExceeded(
            "GPU intermediate sweep deadline expired during validation".into(),
        ));
    }
    Ok(())
}

fn intermediate_sweep_all_finite(
    values: &[f32],
    deadline: Option<std::time::Instant>,
) -> Result<bool> {
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
            intermediate_sweep_check_deadline(deadline)?;
        }
        if !value.is_finite() {
            return Ok(false);
        }
    }
    intermediate_sweep_check_deadline(deadline)?;
    Ok(true)
}

fn require_intermediate_dim_eq(
    op_index: usize,
    role: &str,
    expected: usize,
    actual: usize,
) -> Result<()> {
    if actual != expected {
        return Err(intermediate_sweep_invalid(format!(
            "operation {op_index} {role} dimension {actual} != output dimension {expected}"
        )));
    }
    Ok(())
}

fn validate_intermediate_unary_layer(
    op_index: usize,
    layer: &GpuCrownLayer,
    output_dim: usize,
    input_dim: usize,
    deadline: Option<std::time::Instant>,
) -> Result<()> {
    let context = |message: String| {
        intermediate_sweep_invalid(format!("operation {op_index} unary layer: {message}"))
    };
    match layer {
        GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            cert_err,
        } => {
            if *out_features != output_dim || *in_features != input_dim {
                return Err(context(format!(
                    "linear dimensions ({out_features}, {in_features}) != slots ({output_dim}, {input_dim})"
                )));
            }
            let weight_len = checked_dim_product(
                &[*out_features, *in_features],
                "GPU intermediate sweep linear weight",
            )?;
            if weight.len() != weight_len
                || bias
                    .as_ref()
                    .is_some_and(|values| values.len() != *out_features)
                || !intermediate_sweep_all_finite(weight, deadline)?
                || match bias.as_ref() {
                    Some(values) => !intermediate_sweep_all_finite(values, deadline)?,
                    None => false,
                }
                || !cert_err.is_valid()
            {
                return Err(context("linear payload is malformed or non-finite".into()));
            }
        }
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        } => {
            if *num_neurons != output_dim || input_dim != output_dim {
                return Err(context(format!(
                    "activation dimension {num_neurons} != slots ({output_dim}, {input_dim})"
                )));
            }
            for values in [lower_slope, upper_slope, lower_intercept, upper_intercept] {
                if values.len() != *num_neurons || !intermediate_sweep_all_finite(values, deadline)?
                {
                    return Err(context(
                        "activation payload is malformed or non-finite".into(),
                    ));
                }
            }
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            out_h,
            out_w,
            in_h,
            in_w,
            cert_err,
            pad_h,
            pad_w,
        } => {
            if *kernel_h == 0 || *kernel_w == 0 || *stride_h == 0 || *stride_w == 0 {
                return Err(context(
                    "conv kernel dimensions and strides must be nonzero".into(),
                ));
            }
            let expected_output = checked_dim_product(
                &[*out_channels, *out_h, *out_w],
                "GPU intermediate sweep conv output",
            )?;
            let expected_input = checked_dim_product(
                &[*in_channels, *in_h, *in_w],
                "GPU intermediate sweep conv input",
            )?;
            let weight_len = checked_dim_product(
                &[*out_channels, *in_channels, *kernel_h, *kernel_w],
                "GPU intermediate sweep conv weight",
            )?;
            let expected_out_h = intermediate_sweep_conv_output_extent(
                *in_h, *kernel_h, *stride_h, *pad_h, "height", op_index,
            )?;
            let expected_out_w = intermediate_sweep_conv_output_extent(
                *in_w, *kernel_w, *stride_w, *pad_w, "width", op_index,
            )?;
            if expected_output != output_dim
                || expected_input != input_dim
                || (*out_h, *out_w) != (expected_out_h, expected_out_w)
                || weight_col.len() != weight_len
                || bias_expanded
                    .as_ref()
                    .is_some_and(|values| values.len() != output_dim)
                || !intermediate_sweep_all_finite(weight_col, deadline)?
                || match bias_expanded.as_ref() {
                    Some(values) => !intermediate_sweep_all_finite(values, deadline)?,
                    None => false,
                }
                || !cert_err.is_valid()
            {
                return Err(context("conv payload is malformed or non-finite".into()));
            }
        }
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            num_neurons,
        } => {
            if *num_neurons != output_dim || input_dim != output_dim {
                return Err(context(format!(
                    "dual-alpha dimension {num_neurons} != slots ({output_dim}, {input_dim})"
                )));
            }
            for values in [
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
            ] {
                if values.len() != *num_neurons || !intermediate_sweep_all_finite(values, deadline)?
                {
                    return Err(context(
                        "dual-alpha payload is malformed or non-finite".into(),
                    ));
                }
            }
        }
        GpuCrownLayer::MaxPool2d {
            routing,
            ibp_lower,
            ibp_upper,
            input_dim: layer_input_dim,
            output_dim: layer_output_dim,
        } => {
            if *layer_output_dim != output_dim
                || *layer_input_dim != input_dim
                || routing.len() != output_dim
                || ibp_lower.len() != output_dim
                || ibp_upper.len() != output_dim
            {
                return Err(context("max-pool dimensions are malformed".into()));
            }
            for (row, ((&route, &lower), &upper)) in routing
                .iter()
                .zip(ibp_lower.iter())
                .zip(ibp_upper.iter())
                .enumerate()
            {
                if row.is_multiple_of(GPU_INTERMEDIATE_SWEEP_VALIDATION_POLL_STRIDE) {
                    intermediate_sweep_check_deadline(deadline)?;
                }
                let valid_route = route == u32::MAX || (route as usize) < input_dim;
                if !valid_route || !lower.is_finite() || !upper.is_finite() || lower > upper {
                    return Err(context(format!("max-pool row {row} is malformed")));
                }
            }
        }
    }
    intermediate_sweep_check_deadline(deadline)?;
    Ok(())
}

fn intermediate_sweep_conv_output_extent(
    input: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    axis: &str,
    op_index: usize,
) -> Result<usize> {
    let double_pad = pad.checked_mul(2).ok_or_else(|| {
        intermediate_sweep_invalid(format!(
            "operation {op_index} conv padded {axis} extent overflows"
        ))
    })?;
    let padded = input.checked_add(double_pad).ok_or_else(|| {
        intermediate_sweep_invalid(format!(
            "operation {op_index} conv padded {axis} extent overflows"
        ))
    })?;
    let available = padded.checked_sub(kernel).ok_or_else(|| {
        intermediate_sweep_invalid(format!(
            "operation {op_index} conv {axis} kernel {kernel} exceeds padded input {padded}"
        ))
    })?;
    available
        .checked_div(stride)
        .and_then(|steps| steps.checked_add(1))
        .ok_or_else(|| {
            intermediate_sweep_invalid(format!(
                "operation {op_index} conv output {axis} extent overflows"
            ))
        })
}

/// Maximum number of independently demanded Patches targets described by one
/// observation-only resident-root plan.
///
/// M0 deliberately stops at the planning boundary: this cap bounds host
/// validation and the future device transaction's proof surface. It is not a
/// bound on ordinary CROWN collection, which never consults this type unless
/// the exact-dark experiment is armed.
pub const GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS: usize = 8;

/// Maximum aggregate target-row count admitted to one M0 resident-root plan.
pub const GPU_RESIDENT_PATCHES_ROOT_MAX_ROWS: usize = 131_072;

/// Hard ceiling for the call-local device-workspace cap carried by M0 plans.
pub const GPU_RESIDENT_PATCHES_ROOT_MAX_DEVICE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// One demanded, Patches-eligible graph target in a future cross-target
/// resident CUDA transaction.
///
/// This is metadata only. In particular it carries neither coefficients nor
/// candidate bounds, so an M0 backend cannot manufacture verifier authority
/// through this API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuResidentPatchesRootTargetPlan {
    /// Stable rank in the bounded transaction (zero-based).
    pub rank: usize,
    /// Exact graph node name.
    pub node_name: Arc<str>,
    /// Forward target shape. Its checked product must equal `target_rows`.
    pub target_shape: Arc<[usize]>,
    /// Number of identity/spec rows demanded at this target.
    pub target_rows: usize,
    /// Flat input width of the deepest eligible convolution ancestor.
    pub conv_input_cols: usize,
    /// Bytes in the dense lower/upper backward pair this plan intends to avoid.
    pub dense_pair_bytes: usize,
    /// Bytes in the target's lower/upper f32 interval endpoints.
    pub bound_endpoint_bytes: usize,
}

/// Observation-only plan for a bounded multi-target implicit-Patches root
/// transaction.
///
/// The three SHA-256 values bind the exact process-local graph transcript,
/// input endpoint bits, and selected-target bound endpoint bits. They are
/// telemetry identities, not proof authority: M0 does not return coefficients,
/// bounds, or verdict state. A future authoritative implementation must accept
/// a separate, certificate-bearing request and independently outward-enclose
/// every returned target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuResidentPatchesRootPlan {
    pub graph_identity_sha256: [u8; 32],
    pub input_identity_sha256: [u8; 32],
    pub bounds_identity_sha256: [u8; 32],
    pub targets: Arc<[GpuResidentPatchesRootTargetPlan]>,
    /// Absolute call-local deadline. Backends must refuse a late observation.
    pub deadline: std::time::Instant,
    /// Upper bound reserved for a future device workspace. M0 allocates zero.
    pub max_device_bytes: usize,
}

impl GpuResidentPatchesRootPlan {
    /// Validate all duplicated geometry before an observer may accept the plan.
    ///
    /// This is intentionally stricter than telemetry needs so the same typed
    /// seam can be extended by the next kernel increment without retrofitting
    /// basic overflow, ordering, or deadline checks.
    pub fn validate(&self, now: std::time::Instant) -> Result<()> {
        if now >= self.deadline {
            return Err(NyError::DeadlineExceeded(
                "resident Patches root plan deadline expired".into(),
            ));
        }
        if self.targets.is_empty() || self.targets.len() > GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS {
            return Err(NyError::InvalidSpec(format!(
                "resident Patches root plan requires 1..={} targets; got {}",
                GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS,
                self.targets.len()
            )));
        }
        if self.max_device_bytes == 0
            || self.max_device_bytes > GPU_RESIDENT_PATCHES_ROOT_MAX_DEVICE_BYTES
        {
            return Err(NyError::InvalidSpec(format!(
                "resident Patches root plan device cap must be in 1..={}; got {}",
                GPU_RESIDENT_PATCHES_ROOT_MAX_DEVICE_BYTES, self.max_device_bytes
            )));
        }

        let mut total_rows = 0usize;
        for (expected_rank, target) in self.targets.iter().enumerate() {
            if target.rank != expected_rank {
                return Err(NyError::InvalidSpec(format!(
                    "resident Patches root target rank {} != expected {}",
                    target.rank, expected_rank
                )));
            }
            if target.node_name.is_empty()
                || target.target_shape.is_empty()
                || target.target_rows == 0
                || target.conv_input_cols == 0
            {
                return Err(NyError::InvalidSpec(format!(
                    "resident Patches root target {expected_rank} has empty identity/geometry"
                )));
            }
            if self.targets[..expected_rank]
                .iter()
                .any(|prior| prior.node_name == target.node_name)
            {
                return Err(NyError::InvalidSpec(format!(
                    "resident Patches root target '{}' is duplicated",
                    target.node_name
                )));
            }
            let shape_rows = target
                .target_shape
                .iter()
                .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "resident Patches root target '{}' shape overflows",
                        target.node_name
                    ))
                })?;
            if shape_rows != target.target_rows {
                return Err(NyError::InvalidSpec(format!(
                    "resident Patches root target '{}' shape product {} != rows {}",
                    target.node_name, shape_rows, target.target_rows
                )));
            }
            let expected_dense_pair_bytes = target
                .target_rows
                .checked_mul(target.conv_input_cols)
                .and_then(|elements| elements.checked_mul(2 * size_of::<f32>()))
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "resident Patches root target '{}' dense pair overflows",
                        target.node_name
                    ))
                })?;
            if expected_dense_pair_bytes != target.dense_pair_bytes {
                return Err(NyError::InvalidSpec(format!(
                    "resident Patches root target '{}' dense bytes {} != expected {}",
                    target.node_name, target.dense_pair_bytes, expected_dense_pair_bytes
                )));
            }
            let expected_endpoint_bytes = target
                .target_rows
                .checked_mul(2 * size_of::<f32>())
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "resident Patches root target '{}' endpoint bytes overflow",
                        target.node_name
                    ))
                })?;
            if expected_endpoint_bytes != target.bound_endpoint_bytes {
                return Err(NyError::InvalidSpec(format!(
                    "resident Patches root target '{}' endpoint bytes {} != expected {}",
                    target.node_name, target.bound_endpoint_bytes, expected_endpoint_bytes
                )));
            }
            total_rows = total_rows.checked_add(target.target_rows).ok_or_else(|| {
                NyError::InvalidSpec("resident Patches root total rows overflow".into())
            })?;
        }
        if total_rows > GPU_RESIDENT_PATCHES_ROOT_MAX_ROWS {
            return Err(NyError::InvalidSpec(format!(
                "resident Patches root total rows {} exceeds {}",
                total_rows, GPU_RESIDENT_PATCHES_ROOT_MAX_ROWS
            )));
        }
        Ok(())
    }

    #[must_use]
    pub fn total_rows(&self) -> usize {
        self.targets.iter().fold(0usize, |total, target| {
            total.saturating_add(target.target_rows)
        })
    }

    #[must_use]
    pub fn dense_pair_bytes_avoided(&self) -> usize {
        self.targets.iter().fold(0usize, |total, target| {
            total.saturating_add(target.dense_pair_bytes)
        })
    }
}

/// Backend acknowledgement of an M0 resident-Patches plan.
///
/// Every work/publication counter is required to remain zero in M0. The only
/// positive fields describe validated metadata, so there is structurally no
/// data path from this return type to a verifier bound or verdict.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuResidentPatchesRootObservation {
    pub backend_ready: bool,
    pub accepted_targets: usize,
    pub accepted_rows: usize,
    pub device_allocations: usize,
    pub cuda_dispatches: usize,
    pub bound_values_published: usize,
    pub verdict_mutations: usize,
}

impl GpuResidentPatchesRootObservation {
    #[must_use]
    pub fn is_zero_authority(self) -> bool {
        self.device_allocations == 0
            && self.cuda_dispatches == 0
            && self.bound_values_published == 0
            && self.verdict_mutations == 0
    }
}

/// Maximum specification-row count for the call-local, deadline-bounded
/// small-batch ResNet sound-CROWN contract.
///
/// This is deliberately a small, fixed proof surface. Backends may advertise a
/// lower capacity (including zero), but must never accept a wider request
/// through [`GpuCrownBackward::crown_backward_gpu_resnet_sound_bounded_rows_with_deadline`].
pub const DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS: usize = 8;

/// One BaB subdomain's per-domain operands for a BATCHED sound resnet CROWN
/// backward (#batched-bab). All domains in a batch share the SAME network — the
/// `segments`' Linear/Conv2d weights are the same `Arc<[f32]>` across domains
/// (`Arc::ptr_eq`), only the `Activation` relaxation slopes/intercepts differ —
/// so a batched backward runs one shared-weight GEMM over the stacked
/// `n_domains × num_specs` spec rows. Per-domain: the segments' Activation
/// relaxation, `beta_signed`, `frontier_abs`, `node_abs`, and the input box. The
/// spec seed is shared and passed once to the batched call.
///
/// Bounds-producing entries may consume all of those fields. The
/// [`GpuCrownBackward::crown_backward_gpu_resnet_sound_batched_coeffs`]
/// coefficient egress is different: its box and abs-max fields exist only for
/// signature parity and MUST be ignored so the returned [`CertifiedCoeffs`]
/// remain box-independent.
#[derive(Clone, Copy)]
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

/// Hyperparameters for one fused resident β-only projected-Adam ascent.
///
/// The optimizer uses AMSGrad-style second moments: [`GpuBetaAdamState::v_max`]
/// is the elementwise running maximum of the second moment. Every evaluated β
/// remains projected to `β >= 0`; optimizer arithmetic only chooses which sound
/// β-CROWN bounds to evaluate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBetaAdamConfig {
    /// Maximum number of resident β-CROWN evaluations, including the initial state.
    pub iterations: usize,
    /// Projected-Adam ascent learning rate for β.
    pub beta_lr: f32,
    /// First-moment decay coefficient.
    pub beta1: f32,
    /// Second-moment decay coefficient.
    pub beta2: f32,
    /// Positive denominator stabilizer.
    pub epsilon: f32,
    /// Stop after an evaluated iteration when the maximum absolute gradient is smaller.
    pub tolerance: f32,
}

impl Default for GpuBetaAdamConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            beta_lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            tolerance: 1e-5,
        }
    }
}

/// Location of one sparse β parameter in the resident batched layout.
///
/// A mapped parameter must satisfy
/// `union_gather_idx[relu_index][union_position] == neuron_index`. Backends
/// reject out-of-range indices or a mismatched union entry instead of silently
/// gathering a different neuron's gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBetaAdamMapping {
    /// ReLU index in backward/fold order.
    pub relu_index: u32,
    /// Neuron column within that ReLU's β table.
    pub neuron_index: u32,
    /// Column position within `union_gather_idx[relu_index]`.
    pub union_position: u32,
}

/// One caller-aligned sparse β parameter and its incoming Adam state.
///
/// Evaluation zero starts from the domain's
/// [`GpuResnetBatchedDomainRef::beta_signed`] tables. Mapped parameters then
/// overlay those tables in caller order as `sign * value`; duplicate
/// `(relu_index, neuron_index)` targets therefore use the last input entry.
/// `mapping == None` neither changes the tables nor participates in Adam, and
/// its returned state remains byte-for-byte unchanged. Every state/result
/// vector preserves the exact input parameter order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuBetaAdamParam {
    /// Resident ReLU/neuron/gather location, or `None` for an unmapped parameter.
    pub mapping: Option<GpuBetaAdamMapping>,
    /// Split-constraint sign (`+1` active, `-1` inactive).
    pub sign: f32,
    /// Current nonnegative β value.
    pub value: f32,
    /// Incoming accumulated gradient.
    pub grad: f32,
    /// Incoming Adam first moment.
    pub m: f32,
    /// Incoming Adam second moment.
    pub v: f32,
    /// Incoming AMSGrad maximum second moment.
    pub v_max: f32,
}

/// One domain's borrowed inputs for fused resident β-only Adam.
///
/// The existing CROWN carrier supplies the domain-specific relaxation, initial
/// signed-β tables, error frontiers, and input box. `params` is a sparse,
/// caller-ordered list; returned state uses exactly the same order.
#[derive(Clone, Copy)]
pub struct GpuBetaAdamDomainRef<'a> {
    /// Resident CROWN operands for this domain.
    pub crown: GpuResnetBatchedDomainRef<'a>,
    /// Sparse β parameters and their incoming optimizer state.
    pub params: &'a [GpuBetaAdamParam],
    /// Already-verified specification rows; length must equal `seed.num_specs`.
    pub row_verified: &'a [bool],
    /// Whether to update mapped parameters; `false` requests one frozen evaluation.
    pub optimize: bool,
}

/// Optimizer snapshot for one sparse β parameter.
///
/// Immutable caller metadata (`mapping` and `sign`) is intentionally omitted:
/// the vector containing this state remains aligned with the input `params`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuBetaAdamState {
    /// Nonnegative projected β value.
    pub value: f32,
    /// Gradient carried by the selected snapshot.
    pub grad: f32,
    /// Adam first moment.
    pub m: f32,
    /// Adam second moment.
    pub v: f32,
    /// AMSGrad maximum second moment.
    pub v_max: f32,
}

/// Fused resident β-only Adam output for one domain.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuBetaAdamDomainResult {
    /// Sound evaluation-zero bounds after mapped parameters overlay `crown.beta_signed`.
    pub initial_bounds: GpuCrownResult,
    /// Elementwise-best sound bounds across all completed evaluations:
    /// lower bounds use `max`, while upper bounds use `min`.
    pub best_bounds: GpuCrownResult,
    /// One evaluated parameter snapshot selected for caller warm-start.
    ///
    /// Selection maximizes the minimum `lower[s] - thresholds[s]` across
    /// unverified rows; strict ties retain the earliest snapshot. This single
    /// snapshot need not produce every element of `best_bounds`, whose rows may
    /// come from different evaluations. With no unverified row, no mapped
    /// parameter, or `optimize == false`, it is the unchanged incoming snapshot.
    pub best_state: Vec<GpuBetaAdamState>,
    /// Number of completed resident β-CROWN bound evaluations, including evaluation zero.
    pub iterations_run: usize,
}

/// Results from one fused resident β-only Adam call.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuBetaAdamResult {
    /// Per-domain outputs in exactly the same order as the input domains.
    pub domains: Vec<GpuBetaAdamDomainResult>,
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
    /// Release model/wave-specific resident CROWN and point-VJP working state.
    ///
    /// Long-lived attack engines may be reused across independently loaded
    /// graphs. Callers invoke this at that ownership boundary so cached
    /// coefficient buffers/plans from the old graph cannot overlap the next
    /// model's allocation peak. Backends without resident caches keep the
    /// default no-op; this hook carries no computation or verdict authority.
    fn clear_crown_working_set(&self) -> Result<()> {
        Ok(())
    }

    /// Whether this already-created backend can validate an observation-only
    /// multi-target implicit-Patches root plan.
    ///
    /// This is not a kernel capability. It must not initialize an accelerator,
    /// allocate device state, or imply verdict authority.
    fn provides_resident_patches_root_observer(&self) -> bool {
        false
    }

    /// Validate and acknowledge an M0 resident-Patches root plan.
    ///
    /// Implementations must perform no accelerator allocation/dispatch and
    /// return a zero-authority observation. The default refuses, preserving all
    /// existing backends byte-for-byte.
    fn observe_resident_patches_root_plan(
        &self,
        _plan: &GpuResidentPatchesRootPlan,
    ) -> Result<GpuResidentPatchesRootObservation> {
        Err(NyError::UnsupportedOp(
            "resident Patches root plan observer not supported by this engine".into(),
        ))
    }

    /// Whether this backend implements the sound, GPU-resident, multi-depth
    /// intermediate sweep contract.
    ///
    /// This narrow capability does not independently confer sound-CROWN
    /// authority. A caller must also require [`Self::provides_sound_gpu_crown`]
    /// before requesting verdict-feeding bounds. Implementations should leave
    /// this `false` unless they validate the entire typed request, honor its
    /// call-local deadline and memory cap, and can return every target with the
    /// exact association required by [`GpuIntermediateSweepResult`].
    fn provides_sound_intermediate_sweep(&self) -> bool {
        false
    }

    /// Recommend a bounded scheduling policy for comprehensive sweeps.
    ///
    /// Implementations derive this from live device class and granted limits,
    /// never vendor/model-name matching. The query must be cheap and read-only:
    /// it may not initialize an accelerator, allocate buffers, dispatch work,
    /// or grant proof authority. `None` means the host must not enter the
    /// comprehensive automatic route for this backend.
    fn intermediate_sweep_resource_policy(&self) -> Option<GpuIntermediateSweepResourcePolicy> {
        None
    }

    /// Run one sound GPU-resident reverse-DAG sweep with identity rows injected
    /// at multiple forward depths.
    ///
    /// An accepting backend MUST validate `request` before its first resource
    /// wait, allocation, or dispatch; poll the absolute deadline between every
    /// bounded work unit; keep its explicitly accountable numerical-buffer
    /// peak at or below `max_device_bytes`; charge every certified arithmetic
    /// error required by [`GpuCrownLayer`]; and validate all target intervals
    /// and receipt metadata before returning.
    ///
    /// Success is atomic: `Some(result)` contains exactly one target per plan
    /// injection, in canonical order, and never a completed prefix. Once a
    /// request is accepted, any deadline, allocation, device, numerical, or
    /// association failure returns `Err`, with no partial result. `Ok(None)` is
    /// a pre-dispatch decline only. Callers MUST run
    /// [`GpuIntermediateSweepResult::validate`] on `Some` before
    /// intersecting or publishing any bound.
    ///
    /// The default declines without granting authority or changing existing
    /// backends.
    fn crown_backward_gpu_sound_intermediate_sweep(
        &self,
        _request: &GpuIntermediateSweepRequest<'_>,
    ) -> Result<Option<GpuIntermediateSweepResult>> {
        Ok(None)
    }

    /// Whether this backend implements the typed, sound phase-resident BaB
    /// bound contract.
    ///
    /// This is an explicit opt-in and does not independently grant verdict
    /// authority: a caller must also require [`Self::provides_sound_gpu_crown`].
    /// Existing backends retain the default `false` and cannot be entered.
    fn provides_sound_gpu_bab_bound_phase(&self) -> bool {
        false
    }

    /// Enter the explicitly reviewed resident-BaB numerical TCB seam.
    ///
    /// `None` is the default-closed behavior. Implementations returning `Some`
    /// are an explicit source-reviewed expansion of the numerical TCB; the
    /// returned adapter still cannot act until core supplies a private-field
    /// [`GpuBabBoundTcbInvocation`] after descriptor and soundness-gate checks.
    /// The adapter must expose one stable registration for its exact qualified
    /// device epoch; replacing it is explicit device requalification and TCB
    /// expansion, never an in-session poison reset.
    /// This is a narrow provider seam, not generic sound-CROWN selection:
    /// callers must deliberately select an exact-device WGPU provider whose
    /// separate BaB kernel/selfcheck gate is default closed. CUDA preference or
    /// another backend with `None` here must not be treated as phase support.
    fn gpu_bab_bound_numerical_tcb(&self) -> Option<&dyn GpuBabBoundNumericalTcb> {
        None
    }

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

    /// COEFFICIENT egress for the seeded sound GPU-resident CROWN backward.
    ///
    /// Same arguments, same authority gate, and the same soundness contract as
    /// [`crown_backward_gpu_seeded_sound`](Self::crown_backward_gpu_seeded_sound)
    /// — but the walk's certified affine frontier is published INSTEAD of being
    /// concretized on device. Implementations MUST NOT concretize: the whole
    /// point is that a coefficient-consuming caller (the margin-row lane's
    /// `conv_apply_backward`) can fold the frontier onward itself, which a
    /// concretized `GpuCrownResult` makes impossible.
    ///
    /// `input_lower`/`input_upper` are accepted only for signature parity with the
    /// bounds entry and MUST be deliberately unused. [`CertifiedCoeffs`] certifies
    /// each coefficient independently of a box; folding a radius against this box
    /// and moving it into bias would publish a domain-bound functional envelope,
    /// not the coefficient enclosure this method promises.
    ///
    /// Returning `Ok(None)` means "this backend declines" (the default, so no
    /// existing implementor breaks, and so a backend without
    /// [`provides_sound_gpu_crown`](Self::provides_sound_gpu_crown) authority
    /// can never publish coefficients a verdict would trust). `Err` is reserved
    /// for a real failure of an accepted request.
    ///
    /// Concretizing the returned frontier over the same input box with outward
    /// rounding must give a bound no tighter than the bounds entry's result.
    fn crown_backward_gpu_seeded_sound_coeffs(
        &self,
        _layers: &[GpuCrownLayer],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<Option<CertifiedCoeffs>> {
        Ok(None)
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

    /// COEFFICIENT egress for the RESNET sound GPU-resident CROWN backward.
    ///
    /// The residual twin of
    /// [`crown_backward_gpu_seeded_sound_coeffs`](Self::crown_backward_gpu_seeded_sound_coeffs):
    /// same arguments as
    /// [`crown_backward_gpu_resnet_sound`](Self::crown_backward_gpu_resnet_sound),
    /// same authority gate, same soundness contract — but the COMPOSED
    /// certified frontier (composed ACROSS every segment: chains folded,
    /// identity skips added, projection branches merged) is published instead
    /// of being concretized on device. Implementations MUST publish the
    /// composed frontier the bounds entry would concretize, never a single
    /// segment's intermediate one, and MUST NOT concretize it.
    ///
    /// This is the entry the margin-row twin-wall lane needs: EVERY
    /// cifar100/tinyimagenet net it must accelerate is a resnet, so the flat
    /// chain egress refuses them all at the first residual `Add`.
    ///
    /// `input_lower`/`input_upper` and `frontier_abs`/`node_abs` are accepted for
    /// signature parity with the bounds entry and MUST be deliberately unused.
    /// Folding a coefficient radius against any of those domain-specific
    /// magnitudes and moving it into the bias would only certify the resulting
    /// functional on that domain; it would no longer satisfy
    /// [`CertifiedCoeffs`]' coefficient-wise, box-independent contract. The
    /// caller of this entry chooses the eventual concretization.
    ///
    /// Returning `Ok(None)` means "this backend declines" (the default, so no
    /// existing implementor breaks, and so a backend without
    /// [`provides_sound_gpu_crown`](Self::provides_sound_gpu_crown) authority
    /// can never publish coefficients a verdict would trust). `Err` is reserved
    /// for a real failure of an accepted request.
    ///
    /// Concretizing the returned frontier over the same input box with outward
    /// rounding must give a bound no tighter than the resnet bounds entry's
    /// result.
    fn crown_backward_gpu_resnet_sound_coeffs(
        &self,
        _segments: &[GpuResnetSegment],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<Option<CertifiedCoeffs>> {
        Ok(None)
    }

    /// BATCHED (multi-domain) COEFFICIENT egress for the RESNET sound walk
    /// (#margin-row-gpu-batch).
    ///
    /// This is to [`crown_backward_gpu_resnet_sound_coeffs`](Self::crown_backward_gpu_resnet_sound_coeffs)
    /// exactly what
    /// [`crown_backward_gpu_resnet_sound_beta_batched`](Self::crown_backward_gpu_resnet_sound_beta_batched)
    /// is to the single-domain bounds entry: `N` BaB subdomains that share the
    /// SAME network (weights, topology, `CertifiedWeightError` charges) and the
    /// SAME spec `seed`, differing in their per-domain relaxation values
    /// (`Activation` slopes/intercepts), are folded in ONE wide resident pass,
    /// and each domain's COMPOSED certified frontier is published
    /// un-concretized. Domain descriptors also carry box and abs-max fields for
    /// signature parity with the bounds entries; the coefficient egress MUST
    /// ignore those fields.
    ///
    /// This entry exists because the margin-row twin-wall lane processes BaB
    /// domains one at a time: a per-pass seam accelerates one domain's backward,
    /// but the deciding cifar100/tinyimagenet pools need the whole popped
    /// frontier folded in a single dispatch.
    ///
    /// # Contract
    ///
    /// * Returns EXACTLY `domains.len()` entries, in `domains` order — result
    ///   slot `d` is the frontier of `domains[d]`, computed from
    ///   `domains[d]`'s own segments and no other domain's.
    ///   A backend that cannot guarantee that association MUST decline.
    /// * Every returned [`CertifiedCoeffs`] has `num_specs == seed.num_specs`
    ///   (the SHARED per-domain spec-row count; the wide row stack is an
    ///   implementation detail that must not leak into the payload).
    /// * Same authority gate, same fail-closed firewall and the same
    ///   "MUST NOT concretize" rule as the single-domain coefficient egress.
    /// * `input_lower`/`input_upper` and the abs-max frontiers on each domain are
    ///   accepted for signature parity with the bounds entries and are
    ///   deliberately unused: the caller chooses the concretization. In
    ///   particular, an implementation must not fold coefficient radii against
    ///   those domain-specific values before publishing [`CertifiedCoeffs`].
    /// * `Ok(None)` = "this backend declines" (the default, so no existing
    ///   implementor breaks, and so a backend without
    ///   [`provides_sound_gpu_crown`](Self::provides_sound_gpu_crown) authority
    ///   can never publish coefficients a verdict would trust). Declining is
    ///   also the correct answer for a heterogeneous batch or an unbatchable
    ///   layer kind.
    /// * [`NyError::GpuBatchCapacityExceeded`] is reserved for a device-safe
    ///   width/capacity refusal detected BEFORE any dispatch. A caller may
    ///   narrow and retry only that variant. Every other `Err` is a terminal
    ///   failure of an ACCEPTED request and must not be retried in a different
    ///   arithmetic shape.
    /// * Concretizing slot `d`'s frontier over `domains[d]`'s box with outward
    ///   rounding must give a bound no tighter than what the single-domain
    ///   coefficient egress would publish for that same domain.
    fn crown_backward_gpu_resnet_sound_batched_coeffs(
        &self,
        _domains: &[GpuResnetBatchedDomainRef<'_>],
        _seed: &GpuCrownSeed,
    ) -> Result<Option<Vec<CertifiedCoeffs>>> {
        Ok(None)
    }

    /// Whether this backend exposes the dedicated, deadline-bounded, exactly
    /// one-row ResNet sound backward below.
    ///
    /// This capability is intentionally separate from
    /// [`Self::honors_crown_backward_deadline`]. Enabling the one-row research
    /// call must not alter any pre-existing CROWN route or install mutable
    /// backend-global deadline state.
    fn provides_deadline_bounded_single_row_resnet_sound(&self) -> bool {
        false
    }

    /// Run one sound ResNet CROWN row under an explicit call-local deadline.
    ///
    /// Implementations MUST refuse unless `seed.num_specs == 1`, poll the
    /// deadline during host-side folds, bound every accelerator dispatch, and
    /// return [`NyError::DeadlineExceeded`] instead of publishing a late result.
    /// The default is unsupported.
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_single_row_with_deadline(
        &self,
        _segments: &[GpuResnetSegment],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
        _deadline: std::time::Instant,
    ) -> Result<GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "deadline-bounded single-row resnet sound GPU CROWN backward not supported by this engine"
                .into(),
        ))
    }

    /// Maximum row count accepted by the call-local, deadline-bounded ResNet
    /// sound-CROWN entry below.
    ///
    /// The default preserves compatibility with existing single-row backends:
    /// an implementation that already advertises the dedicated one-row API
    /// automatically reports capacity one. Other backends report zero. A
    /// backend overriding this method must return at most
    /// [`DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS`].
    fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
        usize::from(self.provides_deadline_bounded_single_row_resnet_sound())
    }

    /// Run a small batch of sound ResNet CROWN rows under an explicit call-local
    /// deadline.
    ///
    /// Calls with `seed.num_specs == 1` delegate to the existing single-row
    /// method and inherit its validation and result contract exactly. For calls
    /// with `2..=self.deadline_bounded_resnet_sound_max_rows()`,
    /// implementations advertising capacity greater than one MUST:
    ///
    /// - accept only `2..=self.deadline_bounded_resnet_sound_max_rows()` rows,
    ///   never exceeding [`DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS`];
    /// - reject malformed or non-finite seeds, network data, and input boxes;
    /// - poll the deadline during host-side validation and folds;
    /// - bound every accelerator dispatch;
    /// - return exactly one finite, ordered interval per row; and
    /// - return [`NyError::DeadlineExceeded`] instead of publishing a late
    ///   result.
    ///
    /// The default preserves that exact K=1 delegation and refuses wider
    /// requests.
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        deadline: std::time::Instant,
    ) -> Result<GpuCrownResult> {
        if seed.num_specs == 1 {
            return self.crown_backward_gpu_resnet_sound_single_row_with_deadline(
                segments,
                seed,
                input_lower,
                input_upper,
                frontier_abs,
                node_abs,
                deadline,
            );
        }
        Err(NyError::UnsupportedOp(
            "deadline-bounded multi-row resnet sound GPU CROWN backward not supported by this engine"
                .into(),
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
    /// layers in order, F-branch before P-branch). An empty **outer** slice is the
    /// canonical representation of “no beta constraints”. A nonempty outer slice means
    /// beta is present and each inner slice must have the corresponding activation's
    /// full neuron count; `N` empty inner slices are therefore malformed, not another
    /// spelling of absence. This is the BaB per-domain bound on the
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

    /// Maximum row count accepted by the call-local, deadline-bounded
    /// beta-ResNet sound-CROWN entry below.
    ///
    /// This capability is deliberately independent from both
    /// [`Self::honors_crown_backward_deadline`] and
    /// [`Self::deadline_bounded_resnet_sound_max_rows`]. A backend must
    /// explicitly attest that the beta table is validated and folded inside
    /// the same bounded call. The default is zero.
    fn deadline_bounded_resnet_sound_beta_max_rows(&self) -> usize {
        0
    }

    /// Run `2..=capacity` beta-CROWN specification rows under one explicit,
    /// call-local deadline.
    ///
    /// Implementations MUST reject malformed/non-finite seeds, network data,
    /// input boxes, and beta tables before publication; poll the deadline
    /// during validation and host-side folds; bound every accelerator
    /// dispatch; and return exactly one finite ordered interval per row.
    ///
    /// This small-row surface lets a caller stream a larger specification
    /// matrix through independently bounded transactions. A caller may publish
    /// the combined result only after every transaction succeeds.
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline(
        &self,
        _segments: &[GpuResnetSegment],
        _seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _beta_signed: &[Vec<f32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
        _deadline: std::time::Instant,
    ) -> Result<GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "deadline-bounded beta resnet sound GPU CROWN backward not supported by this engine"
                .into(),
        ))
    }

    /// Observation-only, call-local Cut-CROWN sibling of
    /// [`Self::crown_backward_gpu_resnet_sound_beta`].
    ///
    /// The default-disabled branch calls the existing resident method directly
    /// and returns before inspecting `carrier`, preserving the historical call
    /// and result bits. `Shadow` is never verdict authority: every disposition
    /// returns that same baseline, and a completed cut evaluation may appear
    /// only as telemetry in [`crate::ResidentCutShadowOutcome`].
    ///
    /// The default implementation deliberately has no resident cut kernel.  It
    /// validates enough of a requested carrier to distinguish rejection from an
    /// unavailable backend, then reports `BackendUnavailable` without producing
    /// an observation. A backend override is valid only when it atomically
    /// validates and applies lower post/pre/bias channels with both source and
    /// resident-mutation errors charged outward.
    ///
    /// Callers that need to select a backend specifically for this explicit
    /// deadline-bearing method must first require
    /// [`Self::provides_resident_cut_shadow`]. This capability is deliberately
    /// narrower than [`Self::honors_crown_backward_deadline`].
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_beta_cut_shadow(
        &self,
        policy: crate::ResidentCutShadowPolicy,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        carrier: Option<&crate::ResidentLowerCutCarrier>,
        binding_row: usize,
        deadline: std::time::Instant,
    ) -> Result<crate::ResidentCutShadowOutcome> {
        // Load-bearing off parity: no carrier, deadline, or shadow-specific
        // shape is inspected before the unchanged resident method runs.
        if policy == crate::ResidentCutShadowPolicy::Disabled {
            let baseline = self.crown_backward_gpu_resnet_sound_beta(
                segments,
                seed,
                input_lower,
                input_upper,
                beta_signed,
                frontier_abs,
                node_abs,
            )?;
            return Ok(crate::ResidentCutShadowOutcome::disabled(baseline));
        }

        // Shadow remains observation-only even on refusal: obtain and preserve
        // the exact baseline before touching any carrier field.
        let baseline = self.crown_backward_gpu_resnet_sound_beta(
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
            frontier_abs,
            node_abs,
        )?;
        let mut activation_widths = Vec::new();
        let mut visit = |layers: &[GpuCrownLayer]| {
            for layer in layers {
                match layer {
                    GpuCrownLayer::Activation { num_neurons, .. }
                    | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => {
                        activation_widths.push(*num_neurons);
                    }
                    _ => {}
                }
            }
        };
        for segment in segments {
            match segment {
                GpuResnetSegment::Chain(layers) | GpuResnetSegment::Residual(layers) => {
                    visit(layers);
                }
                GpuResnetSegment::ResidualProj(function, projection) => {
                    visit(function);
                    visit(projection);
                }
            }
        }
        let accepted = binding_row < seed.num_specs
            && carrier.is_some_and(|candidate| {
                candidate.has_nonzero_multiplier()
                    && activation_widths
                        .get(candidate.target_activation())
                        .is_some_and(|&target_width| {
                            candidate
                                .validate_for_call(
                                    activation_widths.len(),
                                    target_width,
                                    seed.num_specs,
                                    deadline,
                                )
                                .is_ok()
                        })
            });
        Ok(if accepted {
            crate::ResidentCutShadowOutcome::backend_unavailable(baseline)
        } else {
            crate::ResidentCutShadowOutcome::rejected(baseline)
        })
    }

    /// Whether this backend implements the observation-only, explicit-deadline
    /// resident Cut-CROWN method above.
    ///
    /// This does not grant verdict authority and does not claim that the
    /// backend's other CROWN methods honor deadlines. The default is false.
    fn provides_resident_cut_shadow(&self) -> bool {
        false
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

    /// Fused resident β-only projected-Adam ascent for a homogeneous domain batch.
    ///
    /// This is the optimizer-resident sibling of
    /// [`crown_backward_gpu_resnet_sound_beta_batched_grad`](Self::crown_backward_gpu_resnet_sound_beta_batched_grad):
    /// a backend may keep the wide coefficient frontier, sparse β values, gradients,
    /// and Adam moments resident for the complete one-shot loop. `thresholds` has
    /// shape `seed.num_specs`; each domain's `row_verified` has the same shape.
    /// `union_gather_idx` has one neuron-column slice per ReLU in fold order, and
    /// every mapped parameter's `union_position` indexes its ReLU's slice and
    /// must resolve to that mapping's `neuron_index`.
    ///
    /// Each returned bound is produced by the sound resident β-CROWN fold. Adam
    /// gradients and moments are non-soundness-critical: they only select later
    /// nonnegative β values. Implementations must preserve domain and sparse-param
    /// order. Evaluation zero starts from each domain's `crown.beta_signed` and
    /// overlays mapped params in caller order as `sign * value`; duplicate targets
    /// are last-entry-wins. Unmapped params are returned byte-for-byte unchanged.
    ///
    /// Implementations must return `Err` unless `config.iterations >= 1`;
    /// learning rate and tolerance are finite and nonnegative; `beta1` and `beta2`
    /// are finite and in `[0, 1]`; and epsilon is finite and positive. Parameter
    /// values, gradients, and moments must be finite, with `value`, `v`, and
    /// `v_max` nonnegative; mapped signs must be exactly `-1` or `1`. All mapping
    /// indices, threshold/verification/result shapes, and union identities must
    /// match, and completed bounds must be finite. Frozen domains (no unverified
    /// row, no mapped param, or `optimize == false`) receive exactly one evaluation.
    /// Thus `iterations_run` counts bound evaluations and at most
    /// `config.iterations - 1` Adam transitions are themselves evaluated.
    ///
    /// Default: unsupported. Callers retain the existing host-orchestrated wide
    /// β-Adam loop or serial fallback.
    fn crown_backward_gpu_resnet_sound_beta_batched_adam(
        &self,
        _domains: &[GpuBetaAdamDomainRef<'_>],
        _seed: &GpuCrownSeed,
        _thresholds: &[f32],
        _union_gather_idx: &[&[u32]],
        _config: GpuBetaAdamConfig,
    ) -> Result<GpuBetaAdamResult> {
        Err(NyError::UnsupportedOp(
            "resident batched beta-Adam resnet sound GPU CROWN backward not supported by this \
             engine"
                .into(),
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

    /// Whether this engine implements the call-local, cooperatively cancellable
    /// joint-α adjoint below.
    ///
    /// This capability is intentionally method-specific.  Advertising the
    /// broader [`Self::honors_crown_backward_deadline`] contract is not enough:
    /// an engine may poll ordinary sound CROWN folds while leaving the separate
    /// joint-adjoint scheduler unbounded.  Deadline-scored callers must require
    /// this exact capability before entering the method below.
    fn provides_deadline_bounded_joint_alpha_gradient_resident(&self) -> bool {
        false
    }

    /// Call-local, cooperatively cancellable twin of
    /// [`Self::crown_joint_alpha_gradient_resident`].
    ///
    /// Implementations that advertise the capability above MUST poll
    /// `deadline` before accelerator launches/resource waits and between
    /// bounded host-side forward, adjoint, and download work units.  Once the
    /// deadline expires they must return [`NyError::DeadlineExceeded`] without
    /// executing the remaining tail or publishing a late gradient.
    ///
    /// The default deliberately refuses instead of delegating to the ordinary,
    /// potentially unbounded joint method.
    #[allow(clippy::too_many_arguments)]
    fn crown_joint_alpha_gradient_resident_with_deadline(
        &self,
        _segments: &[GpuResnetSegment],
        _seed_lower_a: &[f32],
        _num_specs: usize,
        _output_dim: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _deadline: std::time::Instant,
    ) -> Result<Vec<Vec<f32>>> {
        Err(NyError::UnsupportedOp(
            "deadline-bounded on-device joint alpha gradient not supported by this engine".into(),
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

    /// Whether [`Self::set_crown_backward_deadline`] is implemented and long
    /// resident CROWN calls poll it between bounded work units. Deadline-scored
    /// optional lanes must refuse backends that leave this at the default.
    fn honors_crown_backward_deadline(&self) -> bool {
        false
    }
}

#[path = "gemm_naive.rs"]
mod naive;
pub use naive::NaiveCpuGemmEngine;

#[cfg(test)]
mod beta_adam_api_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn beta_adam_config_defaults_match_the_host_optimizer() {
        let config = GpuBetaAdamConfig::default();

        assert_eq!(
            config,
            GpuBetaAdamConfig {
                iterations: 3,
                beta_lr: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                tolerance: 1e-5,
            }
        );
        assert!(format!("{config:?}").contains("GpuBetaAdamConfig"));
        let copied = config;
        assert_eq!(copied, config);
    }

    #[test]
    fn beta_adam_carriers_clone_and_preserve_shapes() {
        let param = GpuBetaAdamParam {
            mapping: Some(GpuBetaAdamMapping {
                relu_index: 2,
                neuron_index: 7,
                union_position: 3,
            }),
            sign: -1.0,
            value: 0.25,
            grad: -0.5,
            m: 0.75,
            v: 1.25,
            v_max: 1.5,
        };
        let state = GpuBetaAdamState {
            value: param.value,
            grad: param.grad,
            m: param.m,
            v: param.v,
            v_max: param.v_max,
        };
        let result = GpuBetaAdamResult {
            domains: vec![GpuBetaAdamDomainResult {
                initial_bounds: GpuCrownResult {
                    lower_bounds: vec![1.0],
                    upper_bounds: vec![2.0],
                },
                best_bounds: GpuCrownResult {
                    lower_bounds: vec![1.5],
                    upper_bounds: vec![1.75],
                },
                best_state: vec![state],
                iterations_run: 2,
            }],
        };

        let cloned = result.clone();
        assert_eq!(cloned, result);
        assert!(format!("{param:?}").contains("union_position: 3"));
        assert!(format!("{result:?}").contains("iterations_run: 2"));
    }

    #[test]
    fn beta_adam_domain_ref_is_a_borrowed_copy_carrier() {
        let crown = GpuResnetBatchedDomainRef {
            segments: &[],
            input_lower: &[],
            input_upper: &[],
            beta_signed: &[],
            frontier_abs: &[],
            node_abs: &[],
        };
        let params = [GpuBetaAdamParam::default()];
        let row_verified = [false, true];
        let domain = GpuBetaAdamDomainRef {
            crown,
            params: &params,
            row_verified: &row_verified,
            optimize: true,
        };

        let copied = domain;
        assert_eq!(copied.params.len(), 1);
        assert_eq!(copied.row_verified, [false, true]);
        assert!(copied.optimize);
        assert!(std::ptr::eq(copied.params.as_ptr(), params.as_ptr()));
    }

    #[derive(Default)]
    struct NoBetaAdamOverride {
        ordinary_calls: AtomicUsize,
    }

    impl GpuCrownBackward for NoBetaAdamOverride {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            self.ordinary_calls.fetch_add(1, Ordering::Relaxed);
            Ok(GpuCrownResult::default())
        }
    }

    #[test]
    fn beta_adam_default_is_unsupported_without_fallback_side_effects() {
        let engine = NoBetaAdamOverride::default();
        let seed = GpuCrownSeed {
            lower_a: Arc::from([]),
            upper_a: Arc::from([]),
            lower_b: Arc::from([]),
            upper_b: Arc::from([]),
            num_specs: 0,
            current_dim: 0,
        };

        let error = engine
            .crown_backward_gpu_resnet_sound_beta_batched_adam(
                &[],
                &seed,
                &[],
                &[],
                GpuBetaAdamConfig::default(),
            )
            .expect_err("an engine with no resident beta-Adam override must decline");

        match error {
            NyError::UnsupportedOp(message) => {
                assert!(message.contains("beta-Adam"));
            }
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(engine.ordinary_calls.load(Ordering::Relaxed), 0);
    }
}

#[cfg(test)]
mod deadline_bounded_rows_api_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    use super::*;

    #[derive(Default)]
    struct LegacySingleRowBackend {
        single_row_calls: AtomicUsize,
    }

    impl GpuCrownBackward for LegacySingleRowBackend {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            unreachable!("the bounded-row default must not enter the ordinary API")
        }

        fn provides_deadline_bounded_single_row_resnet_sound(&self) -> bool {
            true
        }

        fn crown_backward_gpu_resnet_sound_single_row_with_deadline(
            &self,
            _segments: &[GpuResnetSegment],
            seed: &GpuCrownSeed,
            _input_lower: &[f32],
            _input_upper: &[f32],
            _frontier_abs: &[Vec<f32>],
            _node_abs: &[Vec<f32>],
            _deadline: Instant,
        ) -> Result<GpuCrownResult> {
            assert_eq!(seed.num_specs, 1);
            self.single_row_calls.fetch_add(1, Ordering::SeqCst);
            Ok(GpuCrownResult {
                lower_bounds: vec![-1.0],
                upper_bounds: vec![1.0],
            })
        }
    }

    fn seed(rows: usize) -> GpuCrownSeed {
        GpuCrownSeed {
            lower_a: Arc::from(vec![1.0; rows]),
            upper_a: Arc::from(vec![1.0; rows]),
            lower_b: Arc::from(vec![0.0; rows]),
            upper_b: Arc::from(vec![0.0; rows]),
            num_specs: rows,
            current_dim: 1,
        }
    }

    #[test]
    fn bounded_rows_default_preserves_legacy_single_row_and_refuses_wider_calls() {
        let engine = LegacySingleRowBackend::default();
        assert_eq!(DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS, 8);
        assert_eq!(engine.deadline_bounded_resnet_sound_max_rows(), 1);
        assert_eq!(
            engine.deadline_bounded_resnet_sound_beta_max_rows(),
            0,
            "legacy single-row support must not imply the beta capability"
        );

        let result = engine
            .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                &[],
                &seed(1),
                &[],
                &[],
                &[],
                &[],
                Instant::now(),
            )
            .expect("K=1 must delegate to the exact legacy entry");
        assert_eq!(
            result,
            GpuCrownResult {
                lower_bounds: vec![-1.0],
                upper_bounds: vec![1.0],
            }
        );
        assert_eq!(engine.single_row_calls.load(Ordering::SeqCst), 1);

        let error = engine
            .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                &[],
                &seed(2),
                &[],
                &[],
                &[],
                &[],
                Instant::now(),
            )
            .expect_err("the compatibility default must refuse K>1");
        assert!(matches!(error, NyError::UnsupportedOp(_)));
        assert_eq!(
            engine.single_row_calls.load(Ordering::SeqCst),
            1,
            "a wider refusal must not call the legacy one-row entry"
        );
        assert!(matches!(
            engine.crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline(
                &[],
                &seed(2),
                &[],
                &[],
                &[],
                &[],
                &[],
                Instant::now(),
            ),
            Err(NyError::UnsupportedOp(_))
        ));
        assert_eq!(
            engine.single_row_calls.load(Ordering::SeqCst),
            1,
            "the independent beta default must not enter a legacy one-row route"
        );
    }

    #[test]
    fn bounded_rows_default_capacity_is_zero_without_single_row_support() {
        struct UnsupportedBackend;

        impl GpuCrownBackward for UnsupportedBackend {
            fn crown_backward_gpu(
                &self,
                _layers: &[GpuCrownLayer],
                _spec: &[f32],
                _num_specs: usize,
                _input_lower: &[f32],
                _input_upper: &[f32],
            ) -> Result<GpuCrownResult> {
                unreachable!("the bounded-row default must not enter the ordinary API")
            }
        }

        let engine = UnsupportedBackend;
        assert_eq!(engine.deadline_bounded_resnet_sound_max_rows(), 0);
        assert_eq!(engine.deadline_bounded_resnet_sound_beta_max_rows(), 0);
        assert!(matches!(
            engine.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                &[],
                &seed(1),
                &[],
                &[],
                &[],
                &[],
                Instant::now(),
            ),
            Err(NyError::UnsupportedOp(_))
        ));
    }
}

#[cfg(test)]
mod resident_patches_root_plan_tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn target() -> GpuResidentPatchesRootTargetPlan {
        GpuResidentPatchesRootTargetPlan {
            rank: 0,
            node_name: Arc::from("conv1"),
            target_shape: Arc::from([2, 3]),
            target_rows: 6,
            conv_input_cols: 4,
            dense_pair_bytes: 6 * 4 * 2 * size_of::<f32>(),
            bound_endpoint_bytes: 6 * 2 * size_of::<f32>(),
        }
    }

    fn plan() -> GpuResidentPatchesRootPlan {
        GpuResidentPatchesRootPlan {
            graph_identity_sha256: [1; 32],
            input_identity_sha256: [2; 32],
            bounds_identity_sha256: [3; 32],
            targets: Arc::from([target()]),
            deadline: Instant::now() + Duration::from_secs(1),
            max_device_bytes: 1024,
        }
    }

    #[test]
    fn resident_patches_root_plan_validates_duplicated_geometry_and_caps() {
        let valid = plan();
        valid.validate(Instant::now()).unwrap();
        assert_eq!(valid.total_rows(), 6);
        assert_eq!(valid.dense_pair_bytes_avoided(), 192);

        let mut bad_shape = valid.clone();
        Arc::make_mut(&mut bad_shape.targets)[0].target_shape = Arc::from([7]);
        assert!(bad_shape.validate(Instant::now()).is_err());

        let mut bad_device_cap = valid.clone();
        bad_device_cap.max_device_bytes = GPU_RESIDENT_PATCHES_ROOT_MAX_DEVICE_BYTES + 1;
        assert!(bad_device_cap.validate(Instant::now()).is_err());

        let mut duplicate = target();
        duplicate.rank = 1;
        let mut duplicate_plan = valid.clone();
        duplicate_plan.targets = Arc::from([target(), duplicate]);
        assert!(duplicate_plan.validate(Instant::now()).is_err());

        let mut expired = valid;
        expired.deadline = Instant::now();
        assert!(expired.validate(Instant::now()).is_err());
    }

    #[test]
    fn resident_patches_root_observation_rejects_every_authority_channel() {
        let empty = GpuResidentPatchesRootObservation::default();
        assert!(empty.is_zero_authority());

        for observation in [
            GpuResidentPatchesRootObservation {
                device_allocations: 1,
                ..empty
            },
            GpuResidentPatchesRootObservation {
                cuda_dispatches: 1,
                ..empty
            },
            GpuResidentPatchesRootObservation {
                bound_values_published: 1,
                ..empty
            },
            GpuResidentPatchesRootObservation {
                verdict_mutations: 1,
                ..empty
            },
        ] {
            assert!(!observation.is_zero_authority());
        }
    }
}

#[cfg(test)]
mod intermediate_sweep_contract_tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn linear() -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from([1.0, 0.0, 0.0, 1.0]),
            bias: Some(Arc::from([0.0, 0.0])),
            out_features: 2,
            in_features: 2,
            cert_err: CertifiedWeightError::EXACT,
        }
    }

    fn plan() -> GpuIntermediateSweepPlan {
        GpuIntermediateSweepPlan {
            graph_identity_sha256: [1; 32],
            bounds_identity_sha256: [2; 32],
            target_set_identity_sha256: [3; 32],
            ops_backward: Arc::from([
                GpuBackwardOp::Add {
                    output: GpuBackwardSlot(0),
                    lhs: GpuBackwardSlot(1),
                    rhs: GpuBackwardSlot(2),
                },
                GpuBackwardOp::Sub {
                    output: GpuBackwardSlot(1),
                    lhs: GpuBackwardSlot(2),
                    rhs: GpuBackwardSlot(3),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(2),
                    input: GpuBackwardSlot(3),
                    layer: Box::new(linear()),
                },
            ]),
            slot_dims: Arc::from([2, 2, 2, 2]),
            input_slot: GpuBackwardSlot(3),
            injections: Arc::from([
                GpuIntermediateInjection {
                    target_id: 10,
                    slot: GpuBackwardSlot(0),
                    target_shape: Arc::from([2]),
                    selected_rows: Arc::from([0]),
                    row_offset: 0,
                },
                GpuIntermediateInjection {
                    target_id: 20,
                    slot: GpuBackwardSlot(1),
                    target_shape: Arc::from([2]),
                    selected_rows: Arc::from([1]),
                    row_offset: 1,
                },
            ]),
            total_rows: 2,
        }
    }

    fn request(plan: &GpuIntermediateSweepPlan) -> GpuIntermediateSweepRequest<'_> {
        GpuIntermediateSweepRequest {
            plan,
            input_identity_sha256: [4; 32],
            input_lower: &[-1.0, 0.0],
            input_upper: &[1.0, 2.0],
            deadline: Instant::now() + Duration::from_mins(1),
            max_device_bytes: 4096,
        }
    }

    fn result() -> GpuIntermediateSweepResult {
        GpuIntermediateSweepResult::new_unvalidated(
            vec![
                GpuIntermediateTargetResult {
                    target_id: 10,
                    row_offset: 0,
                    selected_rows: Arc::from([0]),
                    lower_bounds: vec![-0.5],
                    upper_bounds: vec![0.5],
                },
                GpuIntermediateTargetResult {
                    target_id: 20,
                    row_offset: 1,
                    selected_rows: Arc::from([1]),
                    lower_bounds: vec![-1.0],
                    upper_bounds: vec![2.0],
                },
            ],
            GpuIntermediateSweepReceipt {
                graph_identity_sha256: [1; 32],
                input_identity_sha256: [4; 32],
                bounds_identity_sha256: [2; 32],
                target_set_identity_sha256: [3; 32],
                requested_targets: 2,
                completed_targets: 2,
                requested_rows: 2,
                completed_rows: 2,
                peak_device_bytes: 1024,
                dispatches: 3,
                host_to_device_bytes: 64,
                device_to_host_bytes: 16,
                readbacks: 1,
                submits: 1,
                synchronizations: 1,
                waves: 1,
            },
        )
    }

    #[test]
    fn validates_residual_fan_out_convergence_and_atomic_result() {
        let plan = plan();
        let request = request(&plan);
        plan.validate().unwrap();
        request.validate().unwrap();
        result().validate(&request).unwrap();
    }

    #[test]
    fn rejects_noncanonical_injections_and_offsets() {
        let mut duplicate_id = plan();
        Arc::make_mut(&mut duplicate_id.injections)[1].target_id = 10;
        assert!(duplicate_id.validate().is_err());

        let mut wrong_order = plan();
        Arc::make_mut(&mut wrong_order.injections).swap(0, 1);
        assert!(wrong_order.validate().is_err());

        let mut duplicate_row = plan();
        let injections = Arc::make_mut(&mut duplicate_row.injections);
        injections[0].selected_rows = Arc::from([0, 0]);
        injections[1].row_offset = 2;
        duplicate_row.total_rows = 3;
        assert!(duplicate_row.validate().is_err());

        let mut wrong_offset = plan();
        Arc::make_mut(&mut wrong_offset.injections)[1].row_offset = 0;
        assert!(wrong_offset.validate().is_err());

        let mut out_of_range = plan();
        Arc::make_mut(&mut out_of_range.injections)[0].selected_rows = Arc::from([2]);
        assert!(out_of_range.validate().is_err());

        let mut direct_input = plan();
        let direct = &mut Arc::make_mut(&mut direct_input.injections)[1];
        direct.slot = direct_input.input_slot;
        direct.target_shape = Arc::from([2]);
        assert!(direct_input.validate().is_err());

        let mut excessive_rank = plan();
        Arc::make_mut(&mut excessive_rank.injections)[0].target_shape =
            vec![1; GPU_INTERMEDIATE_SWEEP_MAX_TARGET_RANK + 1].into();
        assert!(excessive_rank.validate().is_err());
    }

    #[test]
    fn rejects_noncanonical_topology_dimensions_and_layer_payloads() {
        let mut backward_edge = plan();
        Arc::make_mut(&mut backward_edge.ops_backward)[0] = GpuBackwardOp::Add {
            output: GpuBackwardSlot(0),
            lhs: GpuBackwardSlot(0),
            rhs: GpuBackwardSlot(2),
        };
        assert!(backward_edge.validate().is_err());

        let mut wrong_order = plan();
        Arc::make_mut(&mut wrong_order.ops_backward).swap(0, 1);
        assert!(wrong_order.validate().is_err());

        let mut wrong_dim = plan();
        Arc::make_mut(&mut wrong_dim.slot_dims)[2] = 3;
        assert!(wrong_dim.validate().is_err());

        let mut non_finite_weight = plan();
        Arc::make_mut(&mut non_finite_weight.ops_backward)[2] = GpuBackwardOp::Unary {
            output: GpuBackwardSlot(2),
            input: GpuBackwardSlot(3),
            layer: Box::new(GpuCrownLayer::Linear {
                weight: Arc::from([f32::NAN, 0.0, 0.0, 1.0]),
                bias: None,
                out_features: 2,
                in_features: 2,
                cert_err: CertifiedWeightError::EXACT,
            }),
        };
        assert!(non_finite_weight.validate().is_err());

        let malformed_conv_geometry = GpuIntermediateSweepPlan {
            graph_identity_sha256: [1; 32],
            bounds_identity_sha256: [2; 32],
            target_set_identity_sha256: [3; 32],
            ops_backward: Arc::from([GpuBackwardOp::Unary {
                output: GpuBackwardSlot(0),
                input: GpuBackwardSlot(1),
                layer: Box::new(GpuCrownLayer::Conv2d {
                    weight_col: Arc::from([1.0; 9]),
                    bias_expanded: None,
                    out_channels: 1,
                    in_channels: 1,
                    kernel_h: 3,
                    kernel_w: 3,
                    stride_h: 2,
                    stride_w: 2,
                    pad_h: 1,
                    pad_w: 1,
                    out_h: 2,
                    out_w: 4,
                    in_h: 5,
                    in_w: 7,
                    cert_err: CertifiedWeightError::EXACT,
                }),
            }]),
            slot_dims: Arc::from([8, 35]),
            input_slot: GpuBackwardSlot(1),
            injections: Arc::from([GpuIntermediateInjection {
                target_id: 1,
                slot: GpuBackwardSlot(0),
                target_shape: Arc::from([8]),
                selected_rows: Arc::from([0]),
                row_offset: 0,
            }]),
            total_rows: 1,
        };
        assert!(
            malformed_conv_geometry.validate().is_err(),
            "the typed plan must reject internally self-consistent slot dimensions when the conv output formula is wrong"
        );
    }

    #[test]
    fn request_rejects_expiry_zero_cap_and_malformed_input_box() {
        let plan = plan();

        let mut expired = request(&plan);
        expired.deadline = Instant::now();
        assert!(matches!(
            expired.validate(),
            Err(NyError::DeadlineExceeded(_))
        ));

        let mut zero_cap = request(&plan);
        zero_cap.max_device_bytes = 0;
        assert!(zero_cap.validate().is_err());

        let mut malformed_box = request(&plan);
        malformed_box.input_lower = &[2.0, f32::NAN];
        assert!(malformed_box.validate().is_err());
    }

    #[test]
    fn result_validation_rejects_partial_misassociated_or_unbounded_output() {
        let plan = plan();
        let request = request(&plan);

        let mut partial = result();
        partial.targets.pop();
        assert!(partial.validate(&request).is_err());

        let mut misassociated = result();
        misassociated.targets[0].selected_rows = Arc::from([1]);
        assert!(misassociated.validate(&request).is_err());

        let mut non_finite = result();
        non_finite.targets[0].lower_bounds[0] = f32::NEG_INFINITY;
        assert!(non_finite.validate(&request).is_err());

        let mut over_cap = result();
        over_cap.receipt.peak_device_bytes = request.max_device_bytes + 1;
        assert!(over_cap.validate(&request).is_err());

        let mut wrong_identity = result();
        wrong_identity.receipt.bounds_identity_sha256 = [9; 32];
        assert!(wrong_identity.validate(&request).is_err());

        let mut incomplete_receipt = result();
        incomplete_receipt.receipt.completed_rows -= 1;
        assert!(incomplete_receipt.validate(&request).is_err());

        let mut zero_work_receipt = result();
        zero_work_receipt.receipt.peak_device_bytes = 0;
        zero_work_receipt.receipt.dispatches = 0;
        zero_work_receipt.receipt.device_to_host_bytes = 0;
        zero_work_receipt.receipt.readbacks = 0;
        zero_work_receipt.receipt.submits = 0;
        zero_work_receipt.receipt.synchronizations = 0;
        zero_work_receipt.receipt.waves = 0;
        assert!(zero_work_receipt.validate(&request).is_err());
    }

    struct DecliningBackend;

    impl GpuCrownBackward for DecliningBackend {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("test backend".into()))
        }
    }

    #[test]
    fn legacy_backends_decline_without_gaining_sound_authority() {
        let backend = DecliningBackend;
        let plan = plan();
        let request = request(&plan);

        assert!(!backend.provides_sound_intermediate_sweep());
        assert!(!backend.provides_sound_gpu_crown());
        assert!(backend
            .crown_backward_gpu_sound_intermediate_sweep(&request)
            .unwrap()
            .is_none());
    }
}

#[cfg(test)]
mod certified_weight_error_tests {
    use super::*;

    fn linear(cert_err: CertifiedWeightError) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from(vec![1.0f32]),
            bias: None,
            out_features: 1,
            in_features: 1,
            cert_err,
        }
    }

    fn conv(cert_err: CertifiedWeightError) -> GpuCrownLayer {
        GpuCrownLayer::Conv2d {
            weight_col: Arc::from(vec![1.0f32]),
            bias_expanded: None,
            out_channels: 1,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 1,
            out_w: 1,
            in_h: 1,
            in_w: 1,
            cert_err,
        }
    }

    fn activation() -> GpuCrownLayer {
        GpuCrownLayer::Activation {
            lower_slope: vec![1.0],
            upper_slope: vec![1.0],
            lower_intercept: vec![0.0],
            upper_intercept: vec![0.0],
            num_neurons: 1,
        }
    }

    /// The default is the legacy exact-weight contract, and `charged_gamma`
    /// is then the bit-identity on `gamma`.
    #[test]
    fn default_is_the_exact_weight_contract() {
        let exact = CertifiedWeightError::default();
        assert_eq!(exact, CertifiedWeightError::EXACT);
        assert!(exact.is_exact());
        assert!(exact.is_valid());
        for gamma in [0.0f32, 1e-30, 1e-7, 1e-3, 0.5] {
            assert_eq!(
                exact.charged_gamma(gamma).to_bits(),
                gamma.to_bits(),
                "gamma={gamma:e} must survive the exact-weight charge unchanged"
            );
        }
    }

    /// `g = gamma + w + gamma*w`, rounded OUTWARD — never below the real value.
    #[test]
    fn charged_gamma_dominates_the_real_composition() {
        for gamma in [0.0f32, 1e-12, 1e-7, 1e-3] {
            for w in [1e-12f32, 1e-7, 1e-3, 0.5] {
                let cert_err = CertifiedWeightError {
                    weight_rel_err: w,
                    bias_abs_err: 0.0,
                };
                let charged = f64::from(cert_err.charged_gamma(gamma));
                let (g64, w64) = (f64::from(gamma), f64::from(w));
                assert!(
                    charged >= g64 + w64 + g64 * w64,
                    "gamma={gamma:e} w={w:e}: charged {charged:e} is BELOW the \
                     exact composition {:e} — that direction is a false proof",
                    g64 + w64 + g64 * w64
                );
            }
        }
    }

    /// An unusable declaration saturates to `+inf` so the caller must refuse;
    /// a finite substitute would be a silent under-charge.
    #[test]
    fn invalid_declarations_saturate_to_infinity() {
        for bad in [
            CertifiedWeightError {
                weight_rel_err: -1e-9,
                bias_abs_err: 0.0,
            },
            CertifiedWeightError {
                weight_rel_err: 0.0,
                bias_abs_err: -1e-9,
            },
            CertifiedWeightError {
                weight_rel_err: f32::NAN,
                bias_abs_err: 0.0,
            },
            CertifiedWeightError {
                weight_rel_err: f32::INFINITY,
                bias_abs_err: 0.0,
            },
            CertifiedWeightError {
                weight_rel_err: 0.0,
                bias_abs_err: f32::INFINITY,
            },
        ] {
            assert!(!bad.is_valid(), "{bad:?}");
            assert_eq!(bad.charged_gamma(1e-7), f32::INFINITY, "{bad:?}");
        }
        // A huge but valid declaration still overflows outward, not around.
        let huge = CertifiedWeightError {
            weight_rel_err: f32::MAX,
            bias_abs_err: 0.0,
        };
        assert!(huge.is_valid());
        assert_eq!(huge.charged_gamma(f32::MAX), f32::INFINITY);
    }

    /// The fail-closed guard admits the exact-weight default everywhere and
    /// refuses ANY nonzero declaration, in either variant, at any depth of a
    /// resnet decomposition.
    #[test]
    fn uncharged_guard_refuses_every_nonzero_declaration() {
        let exact = CertifiedWeightError::default();
        let weight_only = CertifiedWeightError {
            weight_rel_err: 1e-6,
            bias_abs_err: 0.0,
        };
        let bias_only = CertifiedWeightError {
            weight_rel_err: 0.0,
            bias_abs_err: 1e-6,
        };

        let clean = vec![linear(exact), activation(), conv(exact)];
        refuse_uncharged_certified_weight_error(&clean, "test").unwrap();

        for dirty in [weight_only, bias_only] {
            assert!(matches!(
                refuse_uncharged_certified_weight_error(&[linear(dirty)], "test"),
                Err(NyError::UnsupportedOp(_))
            ));
            assert!(matches!(
                refuse_uncharged_certified_weight_error(&[activation(), conv(dirty)], "test"),
                Err(NyError::UnsupportedOp(_))
            ));
        }

        let clean_segments = vec![
            GpuResnetSegment::Chain(vec![linear(exact)]),
            GpuResnetSegment::Residual(vec![conv(exact)]),
            GpuResnetSegment::ResidualProj(vec![linear(exact)], vec![conv(exact)]),
        ];
        refuse_uncharged_certified_weight_error_segments(&clean_segments, "test").unwrap();

        for dirty_segments in [
            vec![GpuResnetSegment::Chain(vec![linear(weight_only)])],
            vec![GpuResnetSegment::Residual(vec![conv(bias_only)])],
            vec![GpuResnetSegment::ResidualProj(
                vec![linear(exact)],
                vec![conv(weight_only)],
            )],
        ] {
            assert!(matches!(
                refuse_uncharged_certified_weight_error_segments(&dirty_segments, "test"),
                Err(NyError::UnsupportedOp(_))
            ));
        }
    }
}

#[cfg(test)]
mod certified_coeffs_default_entry_tests {
    use super::*;

    struct DecliningBackend;

    impl GpuCrownBackward for DecliningBackend {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("test backend".into()))
        }
    }

    /// The coefficient egress must be OPT-IN: an implementor that has not been
    /// taught to publish a frontier declines, and declining is `Ok(None)` — a
    /// silent, non-erroring "no authority here" — so adding the method breaks
    /// nobody and grants nobody new authority.
    #[test]
    fn default_coeffs_entry_declines_without_erroring() {
        let backend = DecliningBackend;
        let seed = GpuCrownSeed {
            lower_a: Arc::from(vec![1.0f32]),
            upper_a: Arc::from(vec![1.0f32]),
            lower_b: Arc::from(vec![0.0f32]),
            upper_b: Arc::from(vec![0.0f32]),
            num_specs: 1,
            current_dim: 1,
        };
        let declined = backend
            .crown_backward_gpu_seeded_sound_coeffs(&[], &seed, &[0.0], &[1.0])
            .expect("declining must not be an error");
        assert!(
            declined.is_none(),
            "a backend without the coefficient egress must publish nothing"
        );
        assert!(
            !backend.provides_sound_gpu_crown(),
            "and it carries no sound-CROWN authority either"
        );
    }
}

#[cfg(test)]
#[path = "gemm_tests.rs"]
mod tests;
