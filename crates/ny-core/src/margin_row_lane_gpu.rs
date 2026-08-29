// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Device seam TRAIT for the margin-row certified GPU backward lane
//! (#margin-row-gpu-eft, `docs/GPU_CERTIFIED_LANE_BACKWARD_DESIGN_2026-08-19.md`).
//!
//! ny-propagate owns the lane (kernels, admission, verdict-authority chain,
//! certified-error accounting) but depends on ny-gpu only as a
//! dev-dependency, so the device executor is injected through this trait —
//! the same registration pattern as [`crate::GemmEngine`] /
//! `GpuCrownBackward`: the trait lives here, ny-gpu implements it
//! (`MarginRowLaneDevice`), ny-cli registers the factory with
//! `ny_propagate::margin_row::set_margin_row_lane_gpu_factory`.
//!
//! # Contract (design §6, requirements R2–R4 — the implementor's obligations)
//!
//! * **R2 (drain-safe transaction)**: [`MarginRowLaneGpu::run_lane_transaction`]
//!   encodes every pass into ONE command encoder, submits EXACTLY once, and
//!   then drains UNCONDITIONALLY — no fallible operation, no deadline check,
//!   no early return between queue submission and the completed drain. On any
//!   device error the implementation still polls the queue to idle before
//!   returning `Err`.
//! * **R3 (no shared-fate state)**: the implementation holds its OWN device
//!   handle (a dedicated `wgpu::Device`/`Queue`, never the process's sweep
//!   device), installs no deadline on any shared backend object, and never
//!   touches the intermediate sweep's serialization mutex or memory ledger.
//! * **R4 (non-blocking refusal)**: the trait itself performs no blocking
//!   admission; exclusivity is the CALLER's lane-local token. A transaction
//!   error is terminal for the call, never repaired in-lane.
//!
//! The caller (ny-propagate `margin_row::gpu_backward`) treats every `Err` as
//! a fail-closed refusal: the unchanged CPU pass runs bit-for-bit, and the
//! lane's channel latch decides whether to keep asking.

/// One storage binding of a lane transaction, in WGSL binding order starting
/// at `@binding(1)` (binding 0 is always the uniform params buffer).
pub enum LaneBinding<'a> {
    /// Read-only storage, uploaded from these bytes.
    Read(&'a [u8]),
    /// Read-write storage of this byte length, zero-initialized on device,
    /// copied to a staging buffer after the dispatch and returned to the host
    /// (in declaration order) by `run_lane_transaction`.
    Output(u64),
}

/// One arena buffer of a lane PROGRAM (the multi-step transaction shape).
pub enum LaneArenaBuffer<'a> {
    /// Storage buffer uploaded from these bytes. May be bound read-only or
    /// read-write by any step.
    Data(&'a [u8]),
    /// Zero-initialized storage buffer of this byte length.
    Zeroed(u64),
    /// A RESIDENT storage buffer previously uploaded through
    /// [`MarginRowLaneGpu::upload_resident`] (B1 residency cache): the bytes
    /// live on the device for the executor's lifetime and are NOT re-uploaded
    /// per pass. Fail-closed: an id the executor does not recognize is an
    /// `Err` for the whole program, never a silent substitution.
    Resident(LaneResidentId),
}

/// Opaque handle to a device-resident lane buffer (B1 plan-residency cache).
///
/// Minted ONLY by [`MarginRowLaneGpu::upload_resident`]; valid for the
/// lifetime of the executor that minted it (the lane's executor is
/// once-per-process, so ids never dangle in production). The inner index is
/// deliberately private — the caller can only pass a handle back, not forge
/// one from arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneResidentId(u64);

impl LaneResidentId {
    /// Construct from the executor's registry index (implementors only).
    #[must_use]
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The registry index (implementors only).
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Host-observable timing of one lane program (R9 telemetry; NEVER verdict
/// input). All fields are wall-clock milliseconds measured by the executor.
#[derive(Debug, Default, Clone)]
pub struct LaneProgramTiming {
    /// Prepare phase: buffer creation + `write_buffer` uploads + bind groups
    /// (everything before the submit).
    pub upload_ms: f64,
    /// Submit -> completed unconditional drain.
    pub dispatch_ms: f64,
    /// Per-step wall times, filled ONLY in profile mode
    /// ([`LaneProgramSpec::profile`]): one step per submit, each drained to
    /// idle before the next. Order matches [`LaneProgramSpec::steps`].
    pub per_step_ms: Option<Vec<f64>>,
}

/// One step binding: an arena buffer index plus its access mode in this step.
///
/// Steps sharing a pipeline `label` MUST present the same `read_only` shape in
/// the same binding order (the compiled bind-group layout is cached by label).
#[derive(Clone, Copy)]
pub struct LaneStepBinding {
    /// Index into [`LaneProgramSpec::buffers`].
    pub buffer: usize,
    /// Bound as read-only storage (`var<storage, read>`) when true.
    pub read_only: bool,
}

/// One compute dispatch of a lane program: a single-entry WGSL module run once
/// over `workgroups`, with binding 0 = `uniforms` and bindings 1.. = `bindings`
/// in order. Same `label` contract as [`LaneTransactionSpec`]: byte-identical
/// `shader_wgsl`/`entry`/binding shape per label (pipelines are cached by it).
pub struct LaneStepSpec<'a> {
    /// Stable pipeline-cache key AND wgpu debug label.
    pub label: &'a str,
    /// Entry point name inside `shader_wgsl`.
    pub entry: &'a str,
    /// Complete WGSL source (shared primitive library concatenated by the
    /// caller; compiled through the device's resolved shader-loading seam).
    pub shader_wgsl: &'a str,
    /// Uniform buffer contents for `@binding(0)` (non-empty 16-byte multiple).
    pub uniforms: &'a [u8],
    /// Storage bindings 1.., in order.
    pub bindings: &'a [LaneStepBinding],
    /// Dispatch size in workgroups.
    pub workgroups: [u32; 3],
}

/// One drain-safe multi-step transaction: a shared buffer arena and an ordered
/// list of dispatches over it, encoded into ONE command encoder (one compute
/// pass per step, so each step's writes are visible to the next), submitted
/// EXACTLY once, drained unconditionally, then `readbacks` returned in order.
///
/// This is the L-lane pass shape of design §6: the whole backward walk —
/// head gemm, per-layer gate transforms, bias reductions, conv gathers,
/// residual merges — is one submit and one unconditional drain, never a
/// fallible operation in between.
pub struct LaneProgramSpec<'a> {
    /// Debug label for the encoder/error scope.
    pub label: &'a str,
    /// The buffer arena, indexed by [`LaneStepBinding::buffer`].
    pub buffers: &'a [LaneArenaBuffer<'a>],
    /// Ordered dispatches.
    pub steps: &'a [LaneStepSpec<'a>],
    /// Arena indices to copy to staging after the last step and return, in
    /// this order.
    pub readbacks: &'a [usize],
    /// PROFILE-ONLY mode (`NY_MARGIN_ROW_GPU_PROF=1`): the executor MAY run
    /// one step per submit (each submit still unconditionally drained — the
    /// R2 shape holds per submit) and fill
    /// [`LaneProgramTiming::per_step_ms`]. The CALLER guarantees a profiled
    /// program computes nothing verdict-bearing (the lane arms it only on
    /// the M2 shadow path, which discards by construction). `false` is the
    /// one-submit production shape, bit-identical results either way (same
    /// dispatches over the same arena, barriers between steps in both forms).
    pub profile: bool,
}

/// One drain-safe compute transaction: a single-entry WGSL module dispatched
/// once over `workgroups`, with binding 0 = `uniforms` and bindings 1.. =
/// `bindings` in order.
pub struct LaneTransactionSpec<'a> {
    /// Stable pipeline-cache key AND wgpu debug label. Two specs sharing a
    /// label MUST carry byte-identical `shader_wgsl`/`entry` (the implementor
    /// may cache compiled pipelines by this key).
    pub label: &'a str,
    /// Entry point name inside `shader_wgsl`.
    pub entry: &'a str,
    /// Complete WGSL source (the caller concatenates its shared primitive
    /// library with the kernel; the implementor compiles it through its
    /// device's resolved shader-loading seam, DenormPreserve included).
    pub shader_wgsl: &'a str,
    /// Uniform buffer contents for `@binding(0)` (std140-clean, caller's job).
    pub uniforms: &'a [u8],
    /// Storage bindings 1.., in order.
    pub bindings: &'a [LaneBinding<'a>],
    /// Dispatch size in workgroups.
    pub workgroups: [u32; 3],
}

/// The margin-row lane's injected device executor. See the module docs for
/// the R2–R4 obligations an implementation carries.
pub trait MarginRowLaneGpu: Send + Sync {
    /// The device's RESOLVED DenormPreserve word-channel state
    /// (`DenormPreservePolicy::Auto` resolves to `passthrough_supported`, i.e.
    /// silently OFF on an unsupported adapter). The lane's admission REFUSES
    /// unless this is `true` (design §4.1, adversarial-review item 3).
    fn denorm_preserve_enabled(&self) -> bool;

    /// Adapter name + backend, for telemetry only. Never verdict input.
    fn adapter_summary(&self) -> String;

    /// Run one drain-safe transaction and return every [`LaneBinding::Output`]
    /// buffer's bytes, in declaration order. `Err(reason)` means the value is
    /// unusable AND the queue was already drained/polled to idle; the caller
    /// fails closed (CPU pass) and decides whether to latch its channel dead.
    fn run_lane_transaction(&self, spec: &LaneTransactionSpec<'_>) -> Result<Vec<Vec<u8>>, String>;

    /// Run one drain-safe multi-step PROGRAM (the L-lane pass shape): every
    /// step encoded into ONE command encoder in order, ONE `queue.submit`,
    /// then an UNCONDITIONAL drain; the [`LaneProgramSpec::readbacks`] buffers
    /// are returned in order. Same failure contract as
    /// [`Self::run_lane_transaction`]: `Err` means unusable AND the queue was
    /// already driven to idle — the caller fails closed to the CPU pass.
    fn run_lane_program(&self, spec: &LaneProgramSpec<'_>) -> Result<Vec<Vec<u8>>, String>;

    /// [`Self::run_lane_program`] plus host wall-clock timing (R9 telemetry,
    /// never verdict input). The default implementation runs the program
    /// untimed and leaves `timing` zeroed, so implementors without a timing
    /// path stay correct; the ny-gpu executor overrides it with real
    /// upload/dispatch (and, in profile mode, per-step) measurements.
    fn run_lane_program_timed(
        &self,
        spec: &LaneProgramSpec<'_>,
        timing: &mut LaneProgramTiming,
    ) -> Result<Vec<Vec<u8>>, String> {
        *timing = LaneProgramTiming::default();
        self.run_lane_program(spec)
    }

    /// Upload bytes ONCE as a device-resident storage buffer and return its
    /// handle for [`LaneArenaBuffer::Resident`] bindings (B1 residency cache:
    /// pass-invariant weight/bias planes stop being re-uploaded per pass).
    /// The buffer lives until the OWNER releases it
    /// ([`Self::release_resident`]) or the executor is dropped. Default:
    /// unsupported — the caller MUST fall back to [`LaneArenaBuffer::Data`] on
    /// `Err` (fail-closed, semantics-neutral: residency only changes WHERE the
    /// same bytes come from).
    fn upload_resident(&self, bytes: &[u8]) -> Result<LaneResidentId, String> {
        let _ = bytes;
        Err("resident lane buffers unsupported by this executor".to_string())
    }

    /// Release a resident buffer minted by [`Self::upload_resident`]
    /// (FIX D3, impl-v9).
    ///
    /// OWNERSHIP: exactly one caller-side owner holds each id (the lane's
    /// `PlanStatic` plane) and releases it when that owner is dropped, so the
    /// registry is bounded by the LIVE plan statics rather than by every plan
    /// static the process ever built. Before this existed the registry was
    /// append-only, and because impl-v8 gave each `MarginRowBab` its own plan
    /// cache, every BaB's first pass re-uploaded the entire weight/bias plane
    /// set PERMANENTLY.
    ///
    /// Idempotent and total: releasing an unknown or already-released id is a
    /// no-op, never an error and never a panic. An implementation MUST NOT
    /// reuse a released id for a later upload — a stale handle must stay
    /// unknown (fail-closed at bind time) rather than silently alias fresh
    /// bytes. Releasing changes no pass VALUE: while a plane is bound its
    /// contents are identical either way.
    fn release_resident(&self, id: LaneResidentId) {
        let _ = id;
    }

    /// Bytes currently LIVE in the resident-plane registry (FIX D3
    /// observable; telemetry only, never verdict input). `0` when the
    /// executor has no residency support.
    fn resident_bytes(&self) -> u64 {
        0
    }
}
