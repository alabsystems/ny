// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::buffers::BufferPool;
use super::ops::crown_host_profile::CrownHostTimingProfileState;
use super::ops::crown_timestamps::CrownTimestampProfileState;
use super::WgpuDevice;
use ny_core::{NyError, Result};
use tracing::info;

/// What the cheap adapter probe saw, as plain strings (#backend-detect).
///
/// `backend` is wgpu's backend debug name (`Metal`, `Vulkan`, `Dx12`, `Gl`),
/// `device_type` its device class (`IntegratedGpu`, `DiscreteGpu`, ...).
#[derive(Debug, Clone)]
pub struct AdapterProbe {
    pub backend: String,
    pub name: String,
    pub device_type: String,
}

/// All pipeline + bind group layout pairs, extracted for function size (#3397).
pub(super) struct PipelineSet {
    pub(super) linear_ibp: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) matmul_ibp: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) relu_ibp: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) conv2d_ibp: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) softmax_reduce: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) softmax_apply: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) transpose_ibp: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) scale_ibp: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) gemm_f32: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) gemm_f32_small_k: wgpu::ComputePipeline,
    pub(super) crown_activation_backward: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) crown_activation_relu_dual_alpha: wgpu::ComputePipeline,
    pub(super) crown_maxpool2d_backward: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) crown_bias_accumulate: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) crown_concretize: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) conv_reshape: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) conv_col2im: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) add_ibp: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(super) avgpool_ibp: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

impl WgpuDevice {
    /// Create a new GPU device.
    ///
    /// This initializes wgpu, selects the best available GPU backend,
    /// and compiles the compute shaders.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async(None))
    }

    /// #flush-charge admission-config: construct a device with a PROGRAMMATIC
    /// DenormPreserve policy override. `None` is env resolution — the existing
    /// behavior, byte-identical to [`WgpuDevice::new`]. The only defined
    /// override is [`shader_loading::DenormPreservePolicy::ForcedDisabled`]
    /// (the charged-flush constructors): every shader module on the returned
    /// device is created through the plain-WGSL (flushing) path regardless of
    /// ambient env, except that an explicit `NY_GPU_DENORM_PRESERVE=1` pin
    /// refuses with a typed error (env wins, repo-wide precedence rule).
    pub(crate) fn new_with_denorm_preserve_override(
        denorm_override: Option<super::shader_loading::DenormPreservePolicy>,
    ) -> Result<Self> {
        pollster::block_on(Self::new_async(denorm_override))
    }

    /// Per-device DenormPreserve loading-path contract: composes the
    /// process-wide sticky passthrough-fallback poison with THIS device's
    /// resolved loading profile (`denorm_preserve_enabled`). A device that
    /// never requested passthrough creates plain-WGSL modules only and is
    /// structurally immune to the poison.
    pub(crate) fn denorm_preserve_contract_intact(&self) -> bool {
        super::shader_loading::denorm_preserve_contract_intact_for(self.denorm_preserve_enabled)
    }

    /// Get a reference to the underlying wgpu device.
    ///
    /// Capability boundary: buffers allocated directly through this raw handle
    /// are not owned by [`WgpuDevice`], cannot participate in its serialized
    /// intermediate-sweep reservation ledger, and are therefore outside that
    /// backend's `peak_device_bytes` receipt. Prefer typed methods whenever a
    /// call needs wrapper-enforced memory authority. Removing or wrapping this
    /// legacy escape hatch is follow-up work.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the underlying wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    async fn new_async(
        denorm_override: Option<super::shader_loading::DenormPreservePolicy>,
    ) -> Result<Self> {
        let (adapter, adapter_info) = Self::request_adapter().await?;
        info!("wgpu adapter: {}", Self::format_adapter_info(&adapter_info));
        // Resolve the shader-loading profile from the adapter's live capability,
        // never its name. AUTO enables DenormPreserve only when passthrough is
        // advertised; forced `1` refuses here if it cannot possibly work. The
        // complete verdict ladder still measures the exact created device and
        // is the only route to authority. A programmatic ForcedDisabled
        // override (#flush-charge admission-config) instead forces the
        // plain-WGSL path for every module on this device, refusing only when
        // the user explicitly pinned `NY_GPU_DENORM_PRESERVE=1` (env wins).
        let adapter_features = adapter.features();
        let passthrough_supported = adapter_features.contains(wgpu::Features::PASSTHROUGH_SHADERS);
        let (denorm_preserve_policy, denorm_preserve_enabled) = match denorm_override {
            None => super::shader_loading::resolve_denorm_preserve(passthrough_supported)?,
            Some(super::shader_loading::DenormPreservePolicy::ForcedDisabled) => {
                super::shader_loading::resolve_denorm_preserve_forced_disabled()?
            }
            Some(other) => {
                return Err(NyError::InternalError(format!(
                    "unsupported programmatic DenormPreserve override {other:?}: \
                     only ForcedDisabled is defined"
                )))
            }
        };
        let mut required_features = adapter_features & wgpu::Features::TIMESTAMP_QUERY;
        if denorm_preserve_enabled {
            required_features |= wgpu::Features::PASSTHROUGH_SHADERS;
        }
        let timestamp_queries_enabled = required_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        // #big-bindings (dark, `NY_GPU_BIG_BINDINGS=1`, default OFF = wgpu defaults,
        // byte-identical): request the ADAPTER's actual buffer limits instead of
        // wgpu's conservative 128 MiB / 256 MiB defaults. On Apple-silicon Metal the
        // adapter supports multi-GiB storage bindings; the default limit is what
        // makes the WIDE batched resnet backward fail validation (and fall to the
        // serial per-domain stacker) for any batch with
        // `N_rows × widest_im2col_dim × 4 > 128 MiB` — e.g. the LAYERS=2
        // interm-refine cascade at real BaB batch sizes (measured: 32 dom × 69 rows
        // × 73728-dim unrolled conv = 162 MB binding → validation error → 13.5 s
        // serial pass). Buffer-fit gates elsewhere read `device.limits()` live and
        // adapt; hardcoded 128 MiB splitters stay conservative (sound, cost-only).
        // Any allocation that still exceeds the device errs → the existing
        // serial/CPU fallbacks (the 0-wrong moat is untouched).
        // DEFAULT ON since the wide-VJP PGD waves (K=64 × ~884k-dim im2col rows =
        // 226 MB bindings) depend on it; `NY_GPU_BIG_BINDINGS=0` restores the wgpu
        // defaults. Capacity gates read `device.limits()` live, and any allocation
        // that still exceeds the device errs into the serial/CPU fallbacks.
        let required_limits = if std::env::var("NY_GPU_BIG_BINDINGS").ok().as_deref() != Some("0") {
            let a = adapter.limits();
            info!(
                max_storage_buffer_binding_size = a.max_storage_buffer_binding_size,
                max_buffer_size = a.max_buffer_size,
                "NY_GPU_BIG_BINDINGS=1: requesting adapter buffer limits"
            );
            wgpu::Limits {
                max_storage_buffer_binding_size: a.max_storage_buffer_binding_size,
                max_buffer_size: a.max_buffer_size,
                // #u4 taint channel: the activation taint twin binds 11 storage
                // buffers (7 value/error + 4 taint words); the wgpu default is
                // 8 per stage, and exceeding it fails the bind group ASYNC —
                // the dispatch silently no-ops and readbacks return zeros,
                // which is exactly how the first taint-probe run "measured"
                // taint words that never reached the shader. Take the
                // adapter's real limit (clamped for sanity).
                max_storage_buffers_per_shader_stage: a
                    .max_storage_buffers_per_shader_stage
                    .min(16),
                ..wgpu::Limits::default()
            }
        } else {
            wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ny-gpu device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| {
                NyError::UnsupportedConfiguration(format!(
                    "Failed to create device for adapter {}: {e}",
                    Self::format_adapter_info(&adapter_info),
                ))
            })?;
        // Publish only ENV-RESOLVED selections: two env devices with
        // conflicting resolutions still refuse (unchanged). The ForcedDisabled
        // device neither installs nor consults the process selection — its
        // loading path is threaded per-device into every module creation, so
        // it coexists with an env passthrough device without mixing paths.
        if denorm_override.is_none() {
            super::shader_loading::install_denorm_preserve_selection(denorm_preserve_enabled)?;
        }
        info!(timestamp_queries_enabled, "wgpu optional features");
        // Install a non-panicking uncaptured-error backstop BEFORE compiling
        // pipelines or running any GPU work, so that no stray validation/
        // internal error can reach wgpu's default handler and abort the process.
        let uncaptured_errors = Self::install_uncaptured_error_handler(&device);
        let p = Self::init_pipelines(&device, denorm_preserve_enabled).await?;
        let constructed = Self {
            adapter_info,
            denorm_preserve_policy,
            denorm_preserve_enabled,
            device,
            queue,
            linear_ibp_pipeline: p.linear_ibp.0,
            linear_ibp_bind_group_layout: p.linear_ibp.1,
            matmul_ibp_pipeline: p.matmul_ibp.0,
            matmul_ibp_bind_group_layout: p.matmul_ibp.1,
            relu_ibp_pipeline: p.relu_ibp.0,
            relu_ibp_bind_group_layout: p.relu_ibp.1,
            conv2d_ibp_pipeline: p.conv2d_ibp.0,
            conv2d_ibp_bind_group_layout: p.conv2d_ibp.1,
            softmax_reduce_pipeline: p.softmax_reduce.0,
            softmax_reduce_bind_group_layout: p.softmax_reduce.1,
            softmax_apply_pipeline: p.softmax_apply.0,
            softmax_apply_bind_group_layout: p.softmax_apply.1,
            transpose_ibp_pipeline: p.transpose_ibp.0,
            transpose_ibp_bind_group_layout: p.transpose_ibp.1,
            scale_ibp_pipeline: p.scale_ibp.0,
            scale_ibp_bind_group_layout: p.scale_ibp.1,
            gemm_f32_pipeline: p.gemm_f32.0,
            gemm_f32_bind_group_layout: p.gemm_f32.1,
            gemm_f32_small_k_pipeline: p.gemm_f32_small_k,
            crown_activation_backward_pipeline: p.crown_activation_backward.0,
            crown_activation_backward_bind_group_layout: p.crown_activation_backward.1,
            crown_activation_relu_dual_alpha_pipeline: p.crown_activation_relu_dual_alpha,
            crown_maxpool2d_backward_pipeline: p.crown_maxpool2d_backward.0,
            crown_maxpool2d_backward_bind_group_layout: p.crown_maxpool2d_backward.1,
            crown_bias_accumulate_pipeline: p.crown_bias_accumulate.0,
            crown_bias_accumulate_bind_group_layout: p.crown_bias_accumulate.1,
            crown_concretize_pipeline: p.crown_concretize.0,
            crown_concretize_bind_group_layout: p.crown_concretize.1,
            sound_concretize_pipeline: std::sync::OnceLock::new(),
            intermediate_sweep_dag_pipelines: std::sync::OnceLock::new(),
            conv_reshape_pipeline: p.conv_reshape.0,
            conv_reshape_bind_group_layout: p.conv_reshape.1,
            conv_col2im_pipeline: p.conv_col2im.0,
            conv_col2im_bind_group_layout: p.conv_col2im.1,
            add_ibp_pipeline: p.add_ibp.0,
            add_ibp_bind_group_layout: p.add_ibp.1,
            avgpool_ibp_pipeline: p.avgpool_ibp.0,
            avgpool_ibp_bind_group_layout: p.avgpool_ibp.1,
            buffer_pool: std::sync::Mutex::new(BufferPool::default()),
            resident_pipelines: std::sync::OnceLock::new(),
            resident_gather_pipeline: std::sync::OnceLock::new(),
            joint_adjoint_pipelines: std::sync::OnceLock::new(),
            ibp_sound_pipelines: std::sync::OnceLock::new(),
            crown_plan_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            conv_transpose_plan_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            resident_weight_buffers: std::sync::Mutex::new(std::collections::HashMap::new()),
            intermediate_sweep_reserved_bytes: std::sync::Mutex::new(0),
            resident_weight_uploads: std::sync::atomic::AtomicUsize::new(0),
            point_vjp_resident_plans: std::sync::Mutex::new(std::collections::HashMap::new()),
            point_vjp_resident_builds: std::sync::atomic::AtomicUsize::new(0),
            crown_host_timing_profile: std::sync::Mutex::new(CrownHostTimingProfileState::default()),
            crown_timestamp_profile: std::sync::Mutex::new(CrownTimestampProfileState::default()),
            uncaptured_errors,
            gpu_serialize: std::sync::Mutex::new(()),
            crown_backward_deadline: std::sync::Mutex::new(None),
            f32_selfcheck: std::sync::OnceLock::new(),
            eft_selfcheck: std::sync::OnceLock::new(),
            subnormal_selfcheck: std::sync::OnceLock::new(),
            sentinel_taint_selfcheck: std::sync::OnceLock::new(),
            subnormal_mult_taint_selfcheck: std::sync::OnceLock::new(),
            resident_cut_selfcheck: std::sync::OnceLock::new(),
            verdict_report: None,
            bab_bound_provider: super::ops::bab_bound_authority::WgpuBabBoundProvider::new(),
            charged_policy: None,
            submit_tick: std::sync::atomic::AtomicU64::new(0),
        };
        // #eft-err DEADLOCK GUARD: the EFT self-check dispatches GPU work, so it
        // must NEVER be first-initialized from inside a fold that already holds
        // the GPU-checked section (self-deadlock: the check's own guarded
        // dispatch waits on the enclosing lock forever). When the channel is
        // requested, run the check EAGERLY here — device creation holds no GPU
        // locks — so every in-fold read hits the cache. The in-fold gate uses
        // the never-initializing cached read (`eft_primitives_cached`),
        // fail-closed to the Higham channel if somehow still uninitialized.
        if std::env::var("NY_EFT_ERR").ok().as_deref() == Some("1") {
            let _ = constructed.verify_eft_primitives();
        }
        Ok(constructed)
    }

    async fn request_adapter() -> Result<(wgpu::Adapter, wgpu::AdapterInfo)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(e) => {
                let available_adapters: Vec<wgpu::AdapterInfo> = instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .await
                    .into_iter()
                    .map(|adapter| adapter.get_info())
                    .collect();
                return Err(NyError::UnsupportedConfiguration(format!(
                    "No GPU adapter found (power_preference: {:?}, backends: {:?}, available: [{}]): {}",
                    wgpu::PowerPreference::HighPerformance,
                    wgpu::Backends::all(),
                    Self::format_adapter_list(&available_adapters),
                    e
                )));
            }
        };
        let info = adapter.get_info();
        Ok((adapter, info))
    }

    async fn init_pipelines(device: &wgpu::Device, denorm_preserve: bool) -> Result<PipelineSet> {
        macro_rules! scoped {
            ($name:literal, $method:ident) => {
                Self::create_pipeline_scoped(device, $name, || {
                    Self::$method(device, denorm_preserve)
                })
                .await?
            };
        }

        // gemm_f32 first: its bind group layout is shared with gemm_f32_small_k.
        let gemm_f32 = scoped!("gemm_f32", create_gemm_f32_pipeline);
        let gemm_f32_small_k = Self::create_pipeline_scoped(device, "gemm_f32_small_k", || {
            Self::create_gemm_f32_small_k_pipeline(device, denorm_preserve, &gemm_f32.1)
        })
        .await?;

        Ok(PipelineSet {
            linear_ibp: scoped!("linear_ibp", create_linear_ibp_pipeline),
            matmul_ibp: scoped!("matmul_ibp", create_matmul_ibp_pipeline),
            relu_ibp: scoped!("relu_ibp", create_relu_ibp_pipeline),
            conv2d_ibp: scoped!("conv2d_ibp", create_conv2d_ibp_pipeline),
            softmax_reduce: scoped!("softmax_reduce", create_softmax_reduce_pipeline),
            softmax_apply: scoped!("softmax_apply", create_softmax_apply_pipeline),
            transpose_ibp: scoped!("transpose_ibp", create_transpose_ibp_pipeline),
            scale_ibp: scoped!("scale_ibp", create_scale_ibp_pipeline),
            gemm_f32,
            gemm_f32_small_k,
            crown_activation_backward: scoped!(
                "crown_act_bwd",
                create_crown_activation_backward_pipeline
            ),
            crown_activation_relu_dual_alpha: scoped!(
                "crown_act_dual_alpha",
                create_crown_activation_relu_dual_alpha_pipeline
            ),
            crown_maxpool2d_backward: scoped!(
                "crown_maxpool2d_bwd",
                create_crown_maxpool2d_backward_pipeline
            ),
            crown_bias_accumulate: scoped!("crown_bias_acc", create_crown_bias_accumulate_pipeline),
            crown_concretize: scoped!("crown_concretize", create_crown_concretize_pipeline),
            conv_reshape: scoped!("conv_reshape", create_conv_reshape_pipeline),
            conv_col2im: scoped!("conv_col2im", create_conv_col2im_pipeline),
            add_ibp: scoped!("add_ibp", create_add_ibp_pipeline),
            avgpool_ibp: scoped!("avgpool_ibp", create_avgpool_ibp_pipeline),
        })
    }

    /// Get information about the GPU device.
    pub fn info(&self) -> String {
        format!(
            "wgpu device: {}; denorm_preserve={} policy={}",
            Self::format_adapter_info(&self.adapter_info),
            self.denorm_preserve_enabled,
            self.denorm_preserve_policy.name(),
        )
    }

    /// Stable diagnostic name for the capability/override shader policy.
    #[must_use]
    pub fn denorm_preserve_policy_name(&self) -> &'static str {
        self.denorm_preserve_policy.name()
    }

    /// Whether this device creates shaders through the DenormPreserve seam.
    #[must_use]
    pub fn denorm_preserve_enabled(&self) -> bool {
        self.denorm_preserve_enabled
    }

    fn format_adapter_info(info: &wgpu::AdapterInfo) -> String {
        format!(
            "{} (backend: {:?}, device: {}, vendor: 0x{:x}, driver: {}, driver_info: {})",
            info.name, info.backend, info.device, info.vendor, info.driver, info.driver_info
        )
    }

    /// Probe the highest-preference GPU adapter WITHOUT building a device or
    /// compiling any shader (#backend-detect).
    ///
    /// `WgpuDevice::new` pays full pipeline compilation; host backend
    /// *detection* must not, because it runs at the top of every scored
    /// instance. This is the adapter half of `request_adapter` alone —
    /// a few milliseconds — reduced to plain strings so callers outside this
    /// crate need no wgpu types.
    pub fn probe_adapter() -> Option<AdapterProbe> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok()?;
            let info = adapter.get_info();
            // llvmpipe/WARP-class adapters are CPU rasterizers wearing a GPU
            // API; reporting one as a GPU would repeat the exact measurement
            // confusion this probe exists to prevent.
            if info.device_type == wgpu::DeviceType::Cpu {
                return None;
            }
            Some(AdapterProbe {
                backend: format!("{:?}", info.backend),
                name: info.name,
                device_type: format!("{:?}", info.device_type),
            })
        })
    }

    fn format_adapter_list(adapters: &[wgpu::AdapterInfo]) -> String {
        if adapters.is_empty() {
            return "none".to_string();
        }
        adapters
            .iter()
            .map(Self::format_adapter_info)
            .collect::<Vec<_>>()
            .join("; ")
    }

    async fn create_pipeline_scoped<T>(
        device: &wgpu::Device,
        label: &str,
        create: impl FnOnce() -> T,
    ) -> Result<T> {
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let pipeline = create();
        if let Some(err) = oom_scope.pop().await {
            return Err(NyError::UnsupportedConfiguration(format!(
                "wgpu out-of-memory while creating {label} pipeline: {err}"
            )));
        }
        if let Some(err) = internal_scope.pop().await {
            return Err(NyError::UnsupportedConfiguration(format!(
                "wgpu internal error while creating {label} pipeline: {err}"
            )));
        }
        if let Some(err) = validation_scope.pop().await {
            return Err(NyError::UnsupportedConfiguration(format!(
                "wgpu validation error while creating {label} pipeline: {err}"
            )));
        }
        Ok(pipeline)
    }
}
