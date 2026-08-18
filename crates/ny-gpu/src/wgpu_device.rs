// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated bound propagation using wgpu.
//!
//! This module provides GPU acceleration for core bound propagation operations
//! using wgpu (WebGPU) for cross-platform compute shader support.
//!
//! ## Supported Backends
//!
//! - **Metal** (macOS/iOS) - Primary target for Apple Silicon
//! - **Vulkan** (Linux/Windows/Android)
//! - **DX12** (Windows)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ny_gpu::WgpuDevice;
//!
//! let device = WgpuDevice::new().unwrap();
//! // Use device for GPU-accelerated bound propagation
//! ```

#[path = "wgpu_device/buffers.rs"]
mod buffers;
#[path = "wgpu_device/conv_pipelines.rs"]
mod conv_pipelines;
#[path = "wgpu_device/crown_pipelines.rs"]
mod crown_pipelines;
#[path = "wgpu_device/crown_pipelines_concretize.rs"]
mod crown_pipelines_concretize;
#[path = "wgpu_device/device.rs"]
mod device;
#[path = "wgpu_device/error_scope.rs"]
mod error_scope;
#[path = "wgpu_device/ops/mod.rs"]
mod ops;
#[path = "wgpu_device/shader_loading.rs"]
pub(crate) mod shader_loading;
/// #batched-bab wide-lane publication counter, surfaced for the CLI's dark
/// `[wide-lane]` readout (observability only).
pub use ops::wide_resnet_batched_taken_count;
/// Buffer-budget constant shared with the FL value tier (#fl-value-gpu-tier).
pub(crate) use ops::MAX_BINDING_ELEMS;
#[path = "wgpu_device/params.rs"]
mod params;
#[path = "wgpu_device/pipelines.rs"]
mod pipelines;
#[path = "wgpu_device/shaders.rs"]
mod shaders;
#[path = "wgpu_device/shaders_intermediate_sweep.rs"]
mod shaders_intermediate_sweep;
#[path = "wgpu_device/shaders_taint.rs"]
mod shaders_taint;
#[path = "wgpu_device/sound_consts.rs"]
mod sound_consts;
#[path = "wgpu_device/utils.rs"]
mod utils;

#[cfg(test)]
#[path = "wgpu_device/contract_tests.rs"]
mod contract_tests;
/// `#daz-flush-cover-v2` soundness artifact: exact-rational oracle deciding
/// whether subnormal flushing can be CHARGED by the shipped `flush` floor.
#[cfg(test)]
#[path = "wgpu_device/flush_charge_oracle.rs"]
mod flush_charge_oracle;
#[cfg(all(test, feature = "gpu-tests"))]
#[path = "wgpu_device/test_support.rs"]
pub(crate) mod test_support;

/// #flush-charge Lane B: end-to-end ENCLOSURE-PARITY acceptance harness on the
/// TEST-SCOPED charged device (real adapter; the last OPEN row on the charged
/// gate's audit ledger). Compiled only under `gpu-tests`.
#[cfg(all(test, feature = "gpu-tests"))]
#[path = "wgpu_device/flush_charge_acceptance_gpu_tests.rs"]
mod flush_charge_acceptance_gpu_tests;

pub use self::device::AdapterProbe;

use self::buffers::BufferPool;
use self::error_scope::UncapturedErrorState;
use self::ops::bab_bound_authority::WgpuBabBoundProvider;
pub use self::ops::bab_bound_authority::{
    WgpuBabBoundQualificationError, WgpuBabBoundVerdictRequest,
};
use self::ops::conv_transpose_plan::{ConvTransposePlanKey, PreparedConvTransposePlan};
use self::ops::crown_host_profile::CrownHostTimingProfileState;
pub use self::ops::crown_host_profile::{
    CrownHostPhaseTiming, CrownHostPhaseTimingSummary, CrownHostTimingProfile,
};
pub use self::ops::crown_memory_estimate::{
    estimate_crown_backward_peak_bytes, gpu_memory_budget_bytes,
};
use self::ops::crown_plan::PreparedCrownPlan;
use self::ops::crown_plan_key::CrownPlanKey;
use self::ops::crown_timestamps::CrownTimestampProfileState;
pub use self::ops::crown_timestamps::{
    CrownGpuPassTiming, CrownGpuPassTimingSummary, CrownGpuTimingProfile,
};
use self::ops::point_vjp_resident::PointVjpResidentPlans;
use self::ops::resident_weights::{ResidentWeightEntry, ResidentWeightKey};
pub use self::ops::sound_authority::{
    FlushChargePolicy, WgpuChargedVerdictRequest, WgpuVerdictAuthority,
    WgpuVerdictQualificationError, WgpuVerdictReport, WgpuVerdictRequest, WgpuVerdictRung,
    WgpuVerdictRungOutcome,
};
pub(crate) use self::ops::sound_authority::{
    PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED, PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED,
};
// Certified Cut-CROWN C2 resident-lane dark hook (`NY_CUT_FOLD_RESIDENT`):
// registry surface for experiment harnesses (default OFF; empty ⇒ zero-cost).
pub use self::ops::cut_fold_resident::{
    clear_resident_cut_fold, reset_resident_cut_fold_applied_count,
    resident_cut_fold_applied_count, resident_cut_fold_capture_enabled, resident_cut_fold_enabled,
    set_resident_cut_fold, take_resident_cut_fold_capture, ResidentCutFold, ResidentCutFoldCapture,
};
use ny_core::{NyError, Result};

/// The always-built compute pipelines for the sound GPU-resident CROWN backward,
/// cached on [`WgpuDevice`] so they are compiled once and reused across every
/// segment/sub-chain instead of rebuilt per call. Each is the
/// `(ComputePipeline, BindGroupLayout)` pair returned by `create_simple_pipeline`.
/// These hold no numerical/bound data — caching them changes nothing about the
/// computed bounds, only removes redundant shader compilation.
pub(crate) struct ResidentBackwardPipelines {
    pub abs: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub combine: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub bias: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub act: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub act_bias: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// #eft-err (dark, `NY_EFT_ERR=1` + `verify_eft_primitives`): the EFT twin
    /// GEMM (qualified core-multiply/add value sequence plus measured or
    /// conservatively charged FMA residuals) and the min-combine that tightens
    /// the Higham certified error with that bound. Built unconditionally
    /// (pipeline creation carries no numerical
    /// state); DISPATCHED only under the gate ⇒ gate off is byte-identical.
    pub eft_twin: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub eft_min_combine: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub eft_col2im: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// #seg-resident: the on-device lane-pair merge for residual skips /
    /// projection merges (`RESIDENT_SEG_MERGE_SHADER`).
    pub seg_merge: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// Conv fold pipelines + the α-gradient capture pipeline. These are
    /// shader-constant (no shape specialization) but were historically created
    /// FRESH ON EVERY FOLD CALL — thousands of identical driver-side pipeline
    /// compiles per BaB run, and the NVIDIA driver segfaulted inside
    /// `create_compute_pipeline` under the seg-resident call rate. Cached here
    /// once; value-identical (same WGSL, same bind layout).
    pub conv_reshape: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub conv_col2im: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub conv_err: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub alpha_grad: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// #u4 taint twins: the out-of-band `u32` taint-word channel of
    /// TAINT_GUARD_AUDIT.md §4. The production gate is AUTO/default-on when
    /// these twins are available; `NY_GPU_TAINT_WORDS=0` explicitly opts out,
    /// while `=1` requires them and turns unavailability into a typed refusal.
    /// `None` when the device's
    /// granted `max_storage_buffers_per_shader_stage` cannot host the widest
    /// twin (11 bindings — e.g. the wgpu default of 8 under
    /// `NY_GPU_BIG_BINDINGS=0`): building them anyway would raise a BGL
    /// validation error inside the shared `OnceCell` cache and poison EVERY
    /// later walk. DISPATCHED only when the resolved word gate is armed; AUTO
    /// stays unavailable on such a device and explicit `=1` refuses at walk
    /// entry. The twins recompute the base values
    /// byte-for-byte (drift-pinned by
    /// `sentinel_taint_selfcheck::gpu_tests::random_wide_twin_drift_pin`), so
    /// swapping a base dispatch for its twin changes no value bits.
    pub gemm_taint: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub gemm_small_k_taint: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub conv_reshape_taint: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub conv_col2im_taint: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub act_taint: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub combine_taint: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub eft_min_combine_taint: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// #u4 on-device word helper: the AUTO/default route uses the per-spec-row
    /// word fold (`TAINT_ROW_OR_SHADER`, atomicOr with an optional
    /// clean multiplicative-partner annihilation conjunct). Conv uses exact-
    /// value reshape/GEMM/col2im twins; the dormant G13 value→word shader is
    /// retained only as a reference and is not used as a lossy substitute for
    /// internal transport. The row fold replaced
    /// the walk's per-layer blocking word readbacks (the measured 2.3–3.1× tax).
    /// Unlike the 11-binding twins they need at most 3 storage bindings —
    /// under every granted limit — so they are built UNCONDITIONALLY (no
    /// Option or BGL-validation poisoning risk). The row fold dispatches only
    /// when the resolved word gate is armed, which still requires the twins.
    pub taint_row_or: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

impl WgpuDevice {
    /// Submit + periodic NON-BLOCKING maintain: every 64th submit through this
    /// helper runs `device.poll(PollType::Poll)` so completed submissions are
    /// reclaimed even on long stretches with no blocking readback. The NVIDIA
    /// 580.159 Vulkan ICD on the GB10 segfaults (NULL jump inside
    /// vkQueueSubmit) after sustained high-rate submits with no maintenance —
    /// reproduced under the seg-resident + no-merge fold rate, where keep-mode
    /// skips the per-fold readback (and its implicit `poll(Wait)`) entirely.
    /// Value-neutral: `Poll` is non-blocking resource reclamation; results and
    /// submission order are unchanged.
    pub(crate) fn submit_ticked(&self, cmd: wgpu::CommandBuffer) {
        use std::sync::atomic::Ordering;
        self.queue.submit(Some(cmd));
        ops::intermediate_sweep::note_submits(1);
        let n = self.submit_tick.fetch_add(1, Ordering::Relaxed);
        if n % 64 == 63 {
            let _ = self.device.poll(wgpu::PollType::Poll);
        }
    }
}

/// The ON-DEVICE joint α-gradient adjoint pipelines (design doc §3), cached on
/// [`WgpuDevice`] so the coefficient-channel forward fold + reverse-mode adjoint
/// shaders compile once and are reused across every per-domain gradient call
/// (`crown_joint_alpha_gradient_resident`). These hold no numerical/bound data and
/// are NON-soundness-critical (they only steer α∈[0,1]); caching only removes
/// redundant shader compilation from the per-domain hot path.
pub(crate) struct JointAdjointPipelines {
    pub xi_seed: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub relu_fwd: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub conv_t_fwd: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub add: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub rowvec_add: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub relu_harvest: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub relu_prop: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub conv_adj: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

/// Lazily-built candidate certified GPU-resident IBP pipelines.
///
/// These are retained for diagnostics and future requalification, but WGPU
/// currently has no verdict authority. They hold no numerical data; each entry
/// is the `(ComputePipeline, BindGroupLayout)` pair from `create_simple_pipeline`.
pub(crate) struct IbpSoundPipelines {
    /// §3.1 keystone: `LINEAR_IBP_SOUND` — 7 storage (in_l, in_u, wp, wn, bias, out_l, out_u).
    pub linear: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// §3.7: `RELU_IBP_SOUND` — 2 storage (lower, upper), in-place.
    pub relu: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// §3.2: `CONV2D_IBP_SOUND` — 7 storage (in_l, in_u, wp, wn, bias, out_l, out_u).
    pub conv2d: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// §3.3: `MATMUL_IBP_SOUND` — 6 storage (a_l, a_u, b_l, b_u, out_l, out_u).
    pub matmul: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// §3.4: `AVGPOOL_IBP_SOUND` — 4 storage (in_l, in_u, out_l, out_u).
    pub avgpool: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// §3.5: `ADD_IBP_SOUND` — 6 storage (a_l, a_u, b_l, b_u, out_l, out_u).
    pub add: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// §3.6: `TRANSPOSE_IBP_SOUND` — 4 storage (in_l, in_u, out_l, out_u).
    pub transpose: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// §3.8: `SCALE_IBP_SOUND` — 4 storage (in_l, in_u, out_l, out_u).
    pub scale: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// T1.2: `MAXPOOL_CROWN_SOUND` coefficient gather — 6 storage (lower_a, upper_a,
    /// window_meta RO; new_lower_a, new_upper_a, err_comb RW).
    pub maxpool_crown: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

/// GPU device for accelerated bound propagation via wgpu.
///
/// This struct manages the wgpu device, queue, and compute pipelines for
/// running bound propagation operations on the GPU.
///
/// ## Buffer Reuse
///
/// The device maintains a buffer pool to avoid per-call allocation overhead.
/// Buffers are reused when their size is sufficient for the current operation.
/// This significantly reduces the overhead for repeated operations with similar
/// sizes (common in neural network verification).
///
/// ## Chained Operations
///
/// For attention computation, the device supports chained operations that keep
/// intermediate results on the GPU, avoiding costly host roundtrips:
/// - `attention_ibp()`: Q @ K^T -> scale -> softmax -> probs @ V in a single call
pub struct WgpuDevice {
    adapter_info: wgpu::AdapterInfo,
    /// Capability-driven shader loading profile selected before any module was
    /// created on this device. Verdict authority still comes only from the
    /// complete live ladder stored in `verdict_report`.
    denorm_preserve_policy: shader_loading::DenormPreservePolicy,
    denorm_preserve_enabled: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Linear IBP pipeline
    linear_ibp_pipeline: wgpu::ComputePipeline,
    linear_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // MatMul IBP pipeline
    matmul_ibp_pipeline: wgpu::ComputePipeline,
    matmul_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // ReLU IBP pipeline (in-place max(x,0) for resident forward, #4081)
    relu_ibp_pipeline: wgpu::ComputePipeline,
    relu_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // Conv2d IBP pipeline (resident forward, #4275)
    conv2d_ibp_pipeline: wgpu::ComputePipeline,
    conv2d_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // Softmax IBP pipeline (two passes: reduce + apply)
    softmax_reduce_pipeline: wgpu::ComputePipeline,
    softmax_reduce_bind_group_layout: wgpu::BindGroupLayout,
    softmax_apply_pipeline: wgpu::ComputePipeline,
    softmax_apply_bind_group_layout: wgpu::BindGroupLayout,
    // Transpose IBP pipeline (for fused attention)
    transpose_ibp_pipeline: wgpu::ComputePipeline,
    transpose_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // Scale IBP pipeline (for fused attention)
    scale_ibp_pipeline: wgpu::ComputePipeline,
    scale_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // GEMM pipelines (CROWN linear backward)
    gemm_f32_pipeline: wgpu::ComputePipeline,
    gemm_f32_bind_group_layout: wgpu::BindGroupLayout,
    // Small-K GEMM: flat kernel for K ≤ 64, no shared memory/barriers (#3599)
    gemm_f32_small_k_pipeline: wgpu::ComputePipeline,
    // CROWN backward pipelines (persistent GPU A-matrices, #3397)
    crown_activation_backward_pipeline: wgpu::ComputePipeline,
    crown_activation_backward_bind_group_layout: wgpu::BindGroupLayout,
    // ReLU dual-alpha activation backward (#4313) — same bind-group layout as standard activation
    crown_activation_relu_dual_alpha_pipeline: wgpu::ComputePipeline,
    crown_maxpool2d_backward_pipeline: wgpu::ComputePipeline,
    crown_maxpool2d_backward_bind_group_layout: wgpu::BindGroupLayout,
    crown_bias_accumulate_pipeline: wgpu::ComputePipeline,
    crown_bias_accumulate_bind_group_layout: wgpu::BindGroupLayout,
    crown_concretize_pipeline: wgpu::ComputePipeline,
    crown_concretize_bind_group_layout: wgpu::BindGroupLayout,
    /// Lazily initialized for ordinary callers, but eagerly materialized by
    /// verdict qualification so an accepted authoritative sweep performs no
    /// synchronous shader compilation.
    sound_concretize_pipeline: std::sync::OnceLock<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// Full-IEEE Add/Sub intermediate-sweep pipelines. The full verdict
    /// constructor materializes these before publishing authority; charged
    /// adapters never do because their DAG route deliberately declines.
    intermediate_sweep_dag_pipelines:
        std::sync::OnceLock<shaders_intermediate_sweep::SweepDagPipelines>,
    // Conv2d CROWN backward pipelines (#3397)
    conv_reshape_pipeline: wgpu::ComputePipeline,
    conv_reshape_bind_group_layout: wgpu::BindGroupLayout,
    conv_col2im_pipeline: wgpu::ComputePipeline,
    conv_col2im_bind_group_layout: wgpu::BindGroupLayout,
    // Add IBP pipeline (element-wise interval addition for residual, #4319)
    add_ibp_pipeline: wgpu::ComputePipeline,
    add_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // AvgPool IBP pipeline (windowed/global average pooling, #4320)
    avgpool_ibp_pipeline: wgpu::ComputePipeline,
    avgpool_ibp_bind_group_layout: wgpu::BindGroupLayout,
    /// Buffer pool for reuse across calls
    /// Uses Mutex for Sync, allowing use in rayon parallel contexts
    buffer_pool: std::sync::Mutex<BufferPool>,
    /// Lazily-built, reused-forever compute pipelines for the sound GPU-resident
    /// CROWN backward (`crown_backward_sound_resident`). These are pure compiled
    /// shader programs (no numerical/bound data); building them once instead of
    /// per-segment removes redundant shader-module + pipeline compilation from the
    /// deep-resnet hot path without touching any FP math. Filled under the
    /// `gpu_serialize` lock on first use. See `crown_backward_sound_resident.rs`.
    resident_pipelines: std::sync::OnceLock<ResidentBackwardPipelines>,
    /// Lazily-built strided coefficient-gather pipeline used when a CROWN caller
    /// requests enough `(row, column)` values that emitting one copy command per
    /// value would overwhelm command encoding. Kept separate from
    /// `resident_pipelines` so bound-only calls never pay to compile it.
    resident_gather_pipeline: std::sync::OnceLock<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// Lazily-built, reused-forever ON-DEVICE joint α-gradient adjoint pipelines
    /// (design doc §3). Built under the `gpu_serialize` lock on first use; hold no
    /// numerical data (non-soundness-critical gradient path). See
    /// `crown_backward_sound_resident.rs::crown_joint_alpha_gradient_resident`.
    joint_adjoint_pipelines: std::sync::OnceLock<JointAdjointPipelines>,
    /// Lazily-built, reused-forever SOUND GPU-resident IBP forward pipelines
    /// (`docs/SOUND_GPU_IBP_PLAN.md` §3). Built under the `gpu_serialize` lock on
    /// first verdict use; the FAST speed pipelines above stay untouched. See
    /// `ops/ibp_forward_sound.rs`.
    ibp_sound_pipelines: std::sync::OnceLock<IbpSoundPipelines>,
    /// Cached static GPU CROWN plans keyed by topology + static parameter data.
    crown_plan_cache: std::sync::Mutex<
        std::collections::HashMap<CrownPlanKey, std::sync::Arc<PreparedCrownPlan>>,
    >,
    /// Cached fused conv_transpose_2d plans keyed by topology + weight `Arc`
    /// identity. Keeps the weight column matrix GPU-resident and reuses buffers
    /// across the two (`lower_a`/`upper_a`) calls per Conv2d layer (#perf
    /// conv_transpose dispatch wall). Cleared between models alongside
    /// `crown_plan_cache` via `clear_crown_working_set`.
    conv_transpose_plan_cache: std::sync::Mutex<
        std::collections::HashMap<ConvTransposePlanKey, std::sync::Arc<PreparedConvTransposePlan>>,
    >,
    /// GPU-resident constant-weight buffers for the sound resident CROWN folds
    /// (weight residency: constant weights are uploaded once per model instead
    /// of per domain per call).
    /// Keyed by weight `Arc` identity + length + derived form (Raw/Abs/Wᵀ); each
    /// entry retains a KEEP-ALIVE clone of the weight `Arc` so a freed
    /// allocation's address can never be recycled into a false hit (see
    /// `ops/resident_weights.rs`). Cleared between models alongside
    /// `crown_plan_cache` via `clear_crown_working_set`.
    resident_weight_buffers:
        std::sync::Mutex<std::collections::HashMap<ResidentWeightKey, ResidentWeightEntry>>,
    /// Transactional live-byte reservations for accepted typed intermediate
    /// sweeps. A request reserves before its first dispatch and releases by
    /// RAII on every exit, so concurrent callers cannot each pass an isolated
    /// cap preflight against the same retained device.
    intermediate_sweep_reserved_bytes: std::sync::Mutex<usize>,
    /// Count of resident-weight uploads (cache misses); test introspection for
    /// the no-re-upload guarantee.
    resident_weight_uploads: std::sync::atomic::AtomicUsize,
    /// Device-resident wide point-VJP templates (#vjp-resident, attack-only):
    /// per-(template identity, K) buffers + pre-written uniforms + static
    /// slabs, so each attack step uploads only the K mask slabs + spec rows.
    /// Keyed by weight `Arc` identity with keep-alive guards (see
    /// `ops/point_vjp_resident.rs`); cleared between models alongside
    /// `crown_plan_cache` via `clear_crown_working_set`.
    point_vjp_resident_plans: PointVjpResidentPlans,
    /// Count of resident VJP template builds (cache misses; test introspection).
    point_vjp_resident_builds: std::sync::atomic::AtomicUsize,
    /// Optional diagnostic host-side timing profile captured from the last CROWN run.
    crown_host_timing_profile: std::sync::Mutex<CrownHostTimingProfileState>,
    /// Optional diagnostic timestamp profile captured from the last CROWN run.
    crown_timestamp_profile: std::sync::Mutex<CrownTimestampProfileState>,
    /// Non-panicking backstop for wgpu errors that escape thread-local error
    /// scopes. Installed as `on_uncaptured_error`; records errors so in-flight
    /// GPU ops can fail cleanly and fall back to CPU instead of aborting (#live
    /// wgpu validation panic). See `error_scope.rs`.
    uncaptured_errors: std::sync::Arc<UncapturedErrorState>,
    /// Serializes top-level GPU submit+readback operations.
    ///
    /// The device shares mutable state across calls: the pooled buffers, the
    /// cached CROWN plans (`Arc<PreparedCrownPlan>` whose working buffers are
    /// reused), and the thread-local error scopes. When BaB drives CROWN from
    /// multiple Rayon worker threads, two concurrent calls would submit to and
    /// `map_async` the *same* shared staging buffer — hitting wgpu's
    /// `assert_eq!(mapped_range, 0..0, "Buffer is already mapped")` abort
    /// (`wgpu-28.0.0/src/api/buffer.rs:572`) and racing on the same compute
    /// buffers (a soundness hazard). Holding this lock for the duration of each
    /// op makes GPU submission single-threaded, which is correct and matches the
    /// sequential GPU usage the buffer pool already assumes (#3813). See
    /// `error_scope.rs::run_gpu_checked`.
    gpu_serialize: std::sync::Mutex<()>,
    /// Cooperative-cancellation deadline for multi-dispatch CROWN backward calls
    /// (#w4-refresh-deadline). Set/cleared via the `GpuCrownBackward` trait's
    /// `set_crown_backward_deadline`; checked between spec batches
    /// (`crown_backward_gpu_batched_seeded`) and between sound resident layer
    /// folds, where stopping is safe (an expired check returns
    /// `DeadlineExceeded`, and every CROWN caller falls back soundly).
    crown_backward_deadline: std::sync::Mutex<Option<std::time::Instant>>,
    /// Cached one-time IEEE-754 f32 diagnostic (`ops/f32_selfcheck.rs`).
    ///
    /// A passing result proves only this rung. Raw CROWN authority additionally
    /// requires the reviewed source gate, an explicit request, and every other
    /// ladder rung. Populated lazily; the probe runs at most once per device.
    f32_selfcheck: std::sync::OnceLock<bool>,
    /// Cached result of the one-time per-adapter EFT-primitive self-check
    /// (`verify_eft_primitives`, see `ops/eft_selfcheck.rs`): whether fma
    /// TwoProduct and the fma-barrier TwoSum execute bit-exactly, authorizing
    /// the EFT compensated certified-error channel. `false`/failure only
    /// refuses that optional tightening (Higham charge ships unchanged) —
    /// never the sound path itself.
    eft_selfcheck: std::sync::OnceLock<bool>,
    /// Cached result of the one-time per-adapter GRADUAL-UNDERFLOW self-check
    /// (`verify_gradual_underflow`, see `ops/subnormal_selfcheck.rs`): whether
    /// this adapter honours subnormal operands (no DAZ) and subnormal results
    /// (no FTZ). This is a PRECONDITION of the EFT residual identity — under
    /// flush-to-zero every subnormal `ep`/`es` residual is silently dropped and
    /// the measured radius UNDER-counts. Fail-closed: uninitialized ⇒ `false`.
    subnormal_selfcheck: std::sync::OnceLock<bool>,
    /// Cached result of the one-time per-adapter `#u4` SENTINEL-TAINT
    /// STICKINESS self-check (`verify_sentinel_taint_sticky`, see
    /// `ops/sentinel_taint_selfcheck.rs`): does the finite ±FALLBACK_BOUND
    /// overflow sentinel survive nonzero fused resident CROWN ops, while clean
    /// exact-zero partners annihilate it? The armed production predicate consults the
    /// out-of-band word because the magnitude-only controls can be laundered by
    /// a small weight or activation slope. This discharges U4; the independently
    /// reviewed B0 source gate is now open, but authority still requires the
    /// immutable successful per-device report. Fail-closed: uninitialized ⇒ `false`.
    sentinel_taint_selfcheck: std::sync::OnceLock<bool>,
    /// #flush-charge §H: cached result of the one-time per-adapter RUNG-5
    /// SUBNORMAL-MULTIPLIER probe (`verify_subnormal_mult_taint`, see the §H
    /// section of `ops/sentinel_taint_selfcheck.rs`): is the taint-word
    /// annihilation domain of this adapter's DAZ compares confined to the
    /// STRICTLY-SUBNORMAL multipliers the charged walk guard refuses (or is
    /// the word path structurally immune)? Any word loss at/above the
    /// `2^-126` normal boundary, or beside a non-flushed value, is HAZARDOUS
    /// and closes charged-flush authority. Consulted only by
    /// `charged_flush_authority_cached` (primed eagerly by the charged
    /// constructor); no uncharged rung reads it. Fail-closed: uninitialized ⇒
    /// `false`.
    subnormal_mult_taint_selfcheck: std::sync::OnceLock<bool>,
    /// Cached result of the one-time pinned RESIDENT CUT-APPLY selfcheck
    /// (`ops/cut_shadow_resident.rs::verify_resident_cut_apply`): whether the
    /// audited cut-apply kernel reproduces its host transcription bit-exactly
    /// and encloses the exact channel demand on the pinned all-normal fixture.
    /// Gates ONLY the observation-only resident Cut-CROWN shadow capability
    /// (`provides_resident_cut_shadow`); no verdict rung reads it. Prewarmed by
    /// both verdict constructors. Fail-closed: uninitialized ⇒ probe runs
    /// lazily; any GPU error ⇒ `false`.
    resident_cut_selfcheck: std::sync::OnceLock<bool>,
    /// Successful typed verdict-qualification report for this exact device.
    /// `None` for every device built through [`WgpuDevice::new`]. The field is
    /// written only after the explicit constructor has eagerly passed all five
    /// live rungs — or, for the charged-flush constructor, its complete
    /// admission predicate — then remains immutable for the device lifetime.
    verdict_report: Option<WgpuVerdictReport>,
    /// Private, default-closed retained-BaB provider foundation. Its optional
    /// core registration is authority-free; only a private qualification token
    /// bound to this exact registration epoch could expose the numerical TCB.
    bab_bound_provider: WgpuBabBoundProvider,
    /// #flush-charge: the armed charge policy for a device built through
    /// [`WgpuDevice::new_for_verdict_flush_charged`]. `None` for every other
    /// device (including fully qualified ones — the two authority states are
    /// mutually exclusive by construction). Immutable for the device lifetime;
    /// read only through `charged_flush_authority_cached`, which composes the
    /// forced-fail hooks and the loading-path contract on every call.
    charged_policy: Option<FlushChargePolicy>,
    /// Submit counter for [`Self::submit_ticked`]'s periodic maintain poll.
    submit_tick: std::sync::atomic::AtomicU64,
}

impl WgpuDevice {
    /// Clear the reusable GPU CROWN working set for long-lived runners (#3515).
    ///
    /// This drops only the CROWN-specific pooled buffers and cached staging
    /// plans, preserving the shared device and compiled pipelines.
    pub fn clear_crown_working_set(&self) -> Result<()> {
        // Use the same lock order as an accepted intermediate sweep: first the
        // whole-device transaction lock, then its reservation ledger. Holding
        // both guards through every cache clear prevents a reserve-after-check
        // race from dropping buffers underneath an in-flight sweep.
        let _gpu_transaction = self
            .gpu_serialize
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let reservation = self
            .intermediate_sweep_reserved_bytes
            .lock()
            .map_err(|err| {
                NyError::InternalError(format!(
                    "intermediate sweep reservation lock poisoned: {err}"
                ))
            })?;
        if *reservation != 0 {
            return Err(NyError::UnsupportedOp(
                "cannot clear the CROWN working set while an intermediate sweep is reserved".into(),
            ));
        }
        {
            let mut pool = self.buffer_pool.lock().map_err(|err| {
                NyError::InternalError(format!("crown working-set lock poisoned: {err}"))
            })?;
            pool.release_crown_buffers();
        }
        self.clear_crown_plan_cache()?;
        self.clear_conv_transpose_plan_cache()?;
        self.clear_point_vjp_resident_plans()?;
        self.clear_resident_weight_buffers()
    }

    /// Return whether this device was created with timestamp-query support.
    #[must_use]
    pub fn supports_timestamp_queries(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// Enable or disable GPU timestamp profiling for subsequent CROWN runs.
    pub fn set_crown_timestamp_profiling(&self, enabled: bool) -> Result<()> {
        if enabled && !self.supports_timestamp_queries() {
            return Err(NyError::UnsupportedConfiguration(
                "wgpu timestamp queries are not enabled on this device".into(),
            ));
        }

        let mut state = self.crown_timestamp_profile.lock().map_err(|err| {
            NyError::InternalError(format!("crown timestamp profile lock poisoned: {err}"))
        })?;
        state.set_enabled(enabled);
        Ok(())
    }

    /// Enable or disable host-side timing profiling for subsequent CROWN runs.
    pub fn set_crown_host_timing_profiling(&self, enabled: bool) -> Result<()> {
        let mut state = self.crown_host_timing_profile.lock().map_err(|err| {
            NyError::InternalError(format!("crown host timing profile lock poisoned: {err}"))
        })?;
        state.set_enabled(enabled);
        Ok(())
    }

    /// Take the host-side timing profile captured by the last profiled CROWN run.
    pub fn take_last_crown_host_timing_profile(&self) -> Result<Option<CrownHostTimingProfile>> {
        let mut state = self.crown_host_timing_profile.lock().map_err(|err| {
            NyError::InternalError(format!("crown host timing profile lock poisoned: {err}"))
        })?;
        Ok(state.take_profile())
    }

    /// Take the timestamp profile captured by the last profiled CROWN run.
    pub fn take_last_crown_timestamp_profile(&self) -> Result<Option<CrownGpuTimingProfile>> {
        let mut state = self.crown_timestamp_profile.lock().map_err(|err| {
            NyError::InternalError(format!("crown timestamp profile lock poisoned: {err}"))
        })?;
        Ok(state.take_profile())
    }

    pub(crate) fn crown_host_timing_profiling_enabled(&self) -> Result<bool> {
        let state = self.crown_host_timing_profile.lock().map_err(|err| {
            NyError::InternalError(format!("crown host timing profile lock poisoned: {err}"))
        })?;
        Ok(state.enabled)
    }

    pub(crate) fn crown_timestamp_profiling_enabled(&self) -> Result<bool> {
        let state = self.crown_timestamp_profile.lock().map_err(|err| {
            NyError::InternalError(format!("crown timestamp profile lock poisoned: {err}"))
        })?;
        Ok(state.enabled)
    }

    pub(crate) fn store_crown_host_timing_profile(
        &self,
        profile: Option<CrownHostTimingProfile>,
    ) -> Result<()> {
        let mut state = self.crown_host_timing_profile.lock().map_err(|err| {
            NyError::InternalError(format!("crown host timing profile lock poisoned: {err}"))
        })?;
        state.store_profile(profile);
        Ok(())
    }

    pub(crate) fn store_crown_timestamp_profile(
        &self,
        profile: Option<CrownGpuTimingProfile>,
    ) -> Result<()> {
        let mut state = self.crown_timestamp_profile.lock().map_err(|err| {
            NyError::InternalError(format!("crown timestamp profile lock poisoned: {err}"))
        })?;
        state.store_profile(profile);
        Ok(())
    }

    /// Store the cooperative CROWN backward deadline (#w4-refresh-deadline).
    /// A poisoned lock is treated as "no deadline" — cancellation is a perf
    /// courtesy, never a correctness requirement.
    pub(crate) fn store_crown_backward_deadline(&self, deadline: Option<std::time::Instant>) {
        if let Ok(mut slot) = self.crown_backward_deadline.lock() {
            *slot = deadline;
        }
    }

    /// Whether the cooperative CROWN backward deadline has passed. `false` when
    /// unset or on a poisoned lock (fail-open: work runs to completion, the
    /// pre-existing behavior).
    ///
    /// Two independent sources compose here (earliest wins by OR):
    /// - the backend-global slot written through `set_crown_backward_deadline`
    ///   (exclusively owned by ny-propagate's deadline-lease registry while a
    ///   lease is live); and
    /// - the thread-local CALL-LOCAL deadline armed by
    ///   [`CallLocalCrownDeadlineScope`] for the deadline-bounded ResNet trait
    ///   entries. Keeping the call-local deadline out of the shared slot means
    ///   these entries can honor an explicit per-call deadline WITHOUT touching
    ///   the lease-owned backend slot (whose foreign writes the lease registry
    ///   cannot detect or compose).
    pub(crate) fn crown_backward_deadline_expired(&self) -> bool {
        let now = std::time::Instant::now();
        let backend_expired = self
            .crown_backward_deadline
            .lock()
            .ok()
            .and_then(|slot| *slot)
            .is_some_and(|deadline| now >= deadline);
        backend_expired
            || CALL_LOCAL_CROWN_DEADLINE
                .with(|slot| slot.get())
                .is_some_and(|deadline| now >= deadline)
    }

    /// Effective call-local deadline for deadline-aware GPU serialization.
    /// Kept separate from the backend-global lease-owned deadline.
    pub(crate) fn call_local_crown_deadline(&self) -> Option<std::time::Instant> {
        CALL_LOCAL_CROWN_DEADLINE.with(|slot| slot.get())
    }
}

thread_local! {
    /// Call-local CROWN deadline for the deadline-bounded ResNet sound entries
    /// (`crown_backward_gpu_resnet_sound_{single_row,bounded_rows}_with_deadline`).
    ///
    /// Thread-local by design: the resident sound fold polls
    /// `crown_backward_deadline_expired` between layers ON THE CALLING THREAD,
    /// so a scope armed here bounds exactly the call that armed it and can
    /// never race a concurrent lease on the shared backend slot.
    static CALL_LOCAL_CROWN_DEADLINE: std::cell::Cell<Option<std::time::Instant>> =
        const { std::cell::Cell::new(None) };
}

/// RAII scope arming the thread-local call-local CROWN deadline.
///
/// Nested scopes compose by retaining the EARLIEST active deadline; drop
/// restores the previous value exactly (LIFO — matching scope nesting).
#[must_use = "the call-local CROWN deadline is armed only while this scope is alive"]
pub(crate) struct CallLocalCrownDeadlineScope {
    previous: Option<std::time::Instant>,
}

impl CallLocalCrownDeadlineScope {
    pub(crate) fn arm(deadline: std::time::Instant) -> Self {
        let previous = CALL_LOCAL_CROWN_DEADLINE.with(|slot| {
            let previous = slot.get();
            let effective = previous.map_or(deadline, |active| active.min(deadline));
            slot.set(Some(effective));
            previous
        });
        Self { previous }
    }
}

impl Drop for CallLocalCrownDeadlineScope {
    fn drop(&mut self) {
        CALL_LOCAL_CROWN_DEADLINE.with(|slot| slot.set(self.previous));
    }
}
