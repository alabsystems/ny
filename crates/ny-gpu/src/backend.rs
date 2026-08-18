// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::accelerated::{AcceleratedBoundPropagation, AcceleratedDevice};
#[cfg(feature = "wgpu")]
use crate::wgpu_device::{
    WgpuBabBoundQualificationError, WgpuBabBoundVerdictRequest, WgpuDevice,
    WgpuVerdictQualificationError, WgpuVerdictReport, WgpuVerdictRequest,
    PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED,
};
#[cfg(feature = "wgpu")]
use crate::WgpuChargedVerdictRequest;
use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

#[cfg(feature = "wgpu")]
fn wgpu_proof_quarantine(operation: &str) -> NyError {
    NyError::UnsupportedConfiguration(format!(
        "WGPU {operation} is quarantined from the public ComputeDevice proof adapter; \
         only CROWN backward is exposed on an explicitly qualified proof device"
    ))
}

/// Backend selection for compute operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU with Rayon parallelization (default, always available)
    #[default]
    Cpu,
    /// wgpu GPU compute (cross-platform: Metal, Vulkan, DX12)
    Wgpu,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu => write!(f, "cpu"),
            Backend::Wgpu => write!(f, "wgpu"),
        }
    }
}

impl std::str::FromStr for Backend {
    type Err = NyError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Backend::Cpu),
            "wgpu" | "gpu" => Ok(Backend::Wgpu),
            _ => Err(NyError::InvalidSpec(format!(
                "Unknown backend: {}. Valid options: cpu, wgpu",
                s
            ))),
        }
    }
}

/// Unified compute device that dispatches to available backends.
///
/// This enum allows runtime backend selection while providing a common interface.
/// Use `ComputeDevice::new(backend)` to create a device with the specified backend.
pub enum ComputeDevice {
    /// CPU with Rayon parallelization
    Cpu(AcceleratedDevice),
    /// wgpu GPU compute (boxed to avoid large enum size)
    #[cfg(feature = "wgpu")]
    Wgpu(Box<WgpuDevice>),
}

/// Whether the wgpu GPU backend is compiled into this build.
///
/// This is a cheap, compile-time availability signal for backend auto-selection:
/// it reports whether `ComputeDevice::new(Backend::Wgpu)` *could* succeed (the
/// `wgpu` feature is present), without paying the cost of actually initializing
/// an adapter and compiling pipelines. Runtime device-init failures (no adapter,
/// unsupported hardware) are still handled by callers falling back to CPU when
/// `ComputeDevice::new` returns an error, so a `true` here never forces GPU use
/// nor changes a verdict — it only gates the *default* preference toward GPU.
#[must_use]
pub const fn wgpu_backend_compiled() -> bool {
    cfg!(feature = "wgpu")
}

/// Whether this build contains the reviewed WGPU proof-qualification path.
///
/// This is a build/source capability only, not authority for any device.
/// [`ComputeDevice::new(Backend::Wgpu)`] remains unarmed. A caller must consume
/// [`WgpuVerdictRequest`] through [`ComputeDevice::new_for_proof`], and the
/// exact returned device gains CROWN authority only after all five live rungs
/// pass. Every other WGPU proof route remains quarantined operation by operation.
///
/// CORRECTED 2026-08-04: this used to end "and, on this host, because WGSL has
/// no f64". WGSL indeed has no f64, but that is NOT a reason to withhold
/// authority — the sound-resident lane certifies its error in pure f32 (Higham
/// `γ_k·S` with outward-rounded host uniforms), and the EFT/double-single
/// channel supplies an f64-grade compensated residual without f64. Its two
/// primitives measured bit-exact on an Apple M5 Max/Metal adapter on 2026-08-04
/// (509/509 and 307/307 lanes, 0 ULP). Bit-exact primitives are necessary and
/// NOT sufficient; see `docs/METAL_EFT_VIABLE_2026-08-04.md`.
///
#[must_use]
pub const fn wgpu_proof_authority() -> bool {
    #[cfg(feature = "wgpu")]
    {
        PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED
    }
    #[cfg(not(feature = "wgpu"))]
    {
        false
    }
}

/// #flush-charge: whether this build's CHARGED-flush WGPU verdict gate is open.
///
/// Read-only exposure of `PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED`
/// (`ops/sound_authority.rs`) so backend reporting can narrate the charged
/// route's state without constructing a device. Like [`wgpu_proof_authority`],
/// this is a build/source capability only and grants nothing: charged authority
/// still requires the explicit typed request
/// ([`crate::WgpuChargedVerdictRequest`] through
/// [`ComputeDevice::new_for_proof_flush_charged`]) AND the complete live
/// pure-flush admission ladder on the exact returned device. `false` here means
/// the charged constructor refuses unconditionally.
#[must_use]
pub const fn wgpu_charged_proof_authority() -> bool {
    #[cfg(feature = "wgpu")]
    {
        crate::wgpu_device::PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED
    }
    #[cfg(not(feature = "wgpu"))]
    {
        false
    }
}

/// Read-only hardware identity returned by the WGPU adapter enumeration probe.
///
/// The probe does not create a device, compile a pipeline, or grant proof
/// authority.  Keeping the formatted identity alongside the hardware/software
/// classification lets default-dark routing diagnostics bind measurements to
/// the actual Metal/Vulkan/DX12 adapter even when info-level logs are disabled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuAdapterProvenance {
    /// Whether the selected adapter is hardware rather than a CPU rasterizer.
    pub hardware_available: bool,
    /// Stable human-readable adapter identity or the fail-closed probe reason.
    pub description: String,
}

#[cfg(feature = "wgpu")]
fn adapter_device_type_is_hardware(device_type: wgpu::DeviceType) -> bool {
    device_type != wgpu::DeviceType::Cpu
}

/// Enumerate the preferred WGPU adapter without constructing a device.
#[cfg(feature = "wgpu")]
#[must_use]
pub fn wgpu_adapter_provenance() -> WgpuAdapterProvenance {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    match adapter {
        Ok(adapter) => {
            let info = adapter.get_info();
            WgpuAdapterProvenance {
                hardware_available: adapter_device_type_is_hardware(info.device_type),
                description: format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend),
            }
        }
        Err(error) => WgpuAdapterProvenance {
            hardware_available: false,
            description: format!("no adapter ({error})"),
        },
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod adapter_capability_tests {
    use super::adapter_device_type_is_hardware;

    #[test]
    fn cpu_adapter_is_hermetically_classified_unavailable() {
        assert!(!adapter_device_type_is_hardware(wgpu::DeviceType::Cpu));
    }

    #[test]
    fn gpu_adapter_types_are_hermetically_classified_available() {
        for device_type in [
            wgpu::DeviceType::IntegratedGpu,
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::VirtualGpu,
        ] {
            assert!(adapter_device_type_is_hardware(device_type));
        }
    }
}

/// No-wgpu builds retain explicit provenance for the unavailable adapter.
#[cfg(not(feature = "wgpu"))]
#[must_use]
pub fn wgpu_adapter_provenance() -> WgpuAdapterProvenance {
    WgpuAdapterProvenance {
        hardware_available: false,
        description: "wgpu backend not compiled".to_string(),
    }
}

/// Runtime probe: is a real (hardware) GPU adapter available to wgpu?
///
/// Complements [`wgpu_backend_compiled`] as the AUTO-backend capability hint
/// when the `GPU_AVAILABLE` env var is unset, so the hint cannot be lost across
/// the prepare/run script process boundary (#vnncomp-gpu-routing). Costs one
/// adapter enumeration (no device or pipeline init). Software rasterizers
/// (llvmpipe/SwiftShader report `DeviceType::Cpu`) return `false` — preferring
/// wgpu on those would be a slowdown, not an acceleration. Like the env hint,
/// this only feeds the default backend *preference*; it never forces GPU use
/// nor changes a verdict.
#[must_use]
pub fn wgpu_adapter_available() -> bool {
    let probe = wgpu_adapter_provenance();
    tracing::info!(
        "GPU adapter probe: {} => hardware GPU {}",
        probe.description,
        if probe.hardware_available {
            "available"
        } else {
            "unavailable"
        }
    );
    probe.hardware_available
}

/// Process-shared CPU-variant [`ComputeDevice`], for handing an engine to
/// engine-optional consumers (e.g. the PGD attacker) on CPU-routed instances.
///
/// Why hand out a CPU "engine" at all: engine presence switches the PGD
/// attacker into its batched-restarts mode, and this device's `gemm_f32`
/// offloads batched GEMMs at or above the MACs gate to the process-global
/// [`ny_propagate::fast_f32_gemm`] accelerator (cuBLAS when installed) while
/// erring below it — so callers' per-layer CPU fallbacks keep byte-identical
/// numerics for everything that does not qualify for the GPU.
#[must_use]
pub fn shared_cpu_engine() -> &'static ComputeDevice {
    static CPU_ENGINE: std::sync::OnceLock<ComputeDevice> = std::sync::OnceLock::new();
    CPU_ENGINE.get_or_init(|| ComputeDevice::Cpu(AcceleratedDevice::new()))
}

impl ComputeDevice {
    /// Create a new compute device with the specified backend.
    ///
    /// Returns an error if the requested backend is not available.
    pub fn new(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Cpu => Ok(ComputeDevice::Cpu(AcceleratedDevice::new())),
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => {
                let device = WgpuDevice::new()?;
                Ok(ComputeDevice::Wgpu(Box::new(device)))
            }
            #[cfg(not(feature = "wgpu"))]
            Backend::Wgpu => Err(NyError::InvalidSpec(
                "wgpu backend not available. Rebuild with `--features wgpu`".to_string(),
            )),
        }
    }

    /// Create one explicitly qualified WGPU proof device.
    ///
    /// This constructor creates exactly one WGPU context, eagerly evaluates all
    /// five authority rungs on it, and returns that same device only when the
    /// complete report passes. The typed error preserves both the report and
    /// any underlying NY device/probe error. Only the CROWN-backward trait seam
    /// is opened; GEMM, convolution, IBP, and DAG accessors stay quarantined.
    #[cfg(feature = "wgpu")]
    pub fn new_for_proof(
        request: WgpuVerdictRequest,
    ) -> std::result::Result<Self, WgpuVerdictQualificationError> {
        WgpuDevice::new_for_verdict(request).map(|device| Self::Wgpu(Box::new(device)))
    }

    /// Create one explicitly retained-BaB-qualified WGPU proof device.
    ///
    /// This distinct router consumes [`WgpuBabBoundVerdictRequest`]; ordinary,
    /// full-verdict, and charged constructors cannot drift into retained-BaB
    /// authority. Production currently refuses in a static preflight before
    /// device creation because all retained-BaB implementation gates are dark.
    #[cfg(feature = "wgpu")]
    pub fn new_for_proof_bab_bound(
        request: WgpuBabBoundVerdictRequest,
    ) -> std::result::Result<Self, WgpuBabBoundQualificationError> {
        WgpuDevice::new_for_verdict_bab_bound(request).map(|device| Self::Wgpu(Box::new(device)))
    }

    /// #flush-charge: create one explicitly CHARGED-flush qualified WGPU proof
    /// device (see `WgpuDevice::new_for_verdict_flush_charged`).
    ///
    /// A separate typed request keeps every existing `new_for_proof` call site
    /// out of charged mode. Admits only when the reviewed charged source gate
    /// is open (it is, since the 2026-08-13 review) AND the adapter measures
    /// PURE-FLUSH with rungs 1/4/5 passing and rung 3 failing; refuses typed,
    /// with the complete report, otherwise. Only the
    /// CROWN-backward trait seam opens; GEMM, convolution, IBP, and DAG
    /// accessors stay quarantined exactly as for the fully qualified device.
    #[cfg(feature = "wgpu")]
    pub fn new_for_proof_flush_charged(
        request: WgpuChargedVerdictRequest,
    ) -> std::result::Result<Self, WgpuVerdictQualificationError> {
        WgpuDevice::new_for_verdict_flush_charged(request)
            .map(|device| Self::Wgpu(Box::new(device)))
    }

    /// #flush-charge TEST-SCOPED wrapper over
    /// `WgpuDevice::test_only_new_flush_charged_for_acceptance_evidence`.
    /// Compiled out of every production build (`cfg(any(test, feature =
    /// "gpu-tests"))`; the `gpu-tests` feature exists only for real-adapter
    /// test invocations and no production crate enables it), so no production
    /// caller can name or reach it. See the device-level constructor's docs
    /// for the acceptance-evidence contract; production admission still goes
    /// only through `new_for_proof_flush_charged` and its full live ladder.
    #[cfg(all(feature = "wgpu", any(test, feature = "gpu-tests")))]
    pub fn test_only_new_for_proof_flush_charged_acceptance_evidence(
    ) -> std::result::Result<Self, WgpuVerdictQualificationError> {
        WgpuDevice::test_only_new_flush_charged_for_acceptance_evidence()
            .map(|device| Self::Wgpu(Box::new(device)))
    }

    /// Successful WGPU verdict report attached to this exact proof device.
    /// Returns `None` for CPU and ordinary/unqualified WGPU devices.
    #[cfg(feature = "wgpu")]
    #[must_use]
    pub fn wgpu_verdict_report(&self) -> Option<&WgpuVerdictReport> {
        match self {
            Self::Cpu(_) => None,
            Self::Wgpu(device) => device.verdict_report(),
        }
    }

    /// Get the backend type of this device.
    pub fn backend(&self) -> Backend {
        match self {
            ComputeDevice::Cpu(_) => Backend::Cpu,
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Backend::Wgpu,
        }
    }

    /// Release reusable GPU CROWN buffers between long-lived runs (#3515).
    pub fn clear_crown_working_set(&self) -> Result<()> {
        match self {
            ComputeDevice::Cpu(_) => Ok(()),
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(device) => device.clear_crown_working_set(),
        }
    }

    /// Full attention IBP: softmax((Q @ K^T) * scale) @ V
    ///
    /// Input shapes: Q, K, V with shape [batch, heads, seq, dim]
    /// Output shape: [batch, heads, seq, dim]
    pub fn attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.attention_ibp(q, k, v, scale),
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("attention IBP")),
        }
    }

    /// Causal attention IBP for decoder-only models (LLaMA, GPT).
    ///
    /// Position i can only attend to positions j where j <= i.
    ///
    /// Input shapes: Q, K, V with shape [batch, heads, seq, dim]
    /// Output shape: [batch, heads, seq, dim]
    pub fn causal_attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.causal_attention_ibp(q, k, v, scale),
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("causal-attention IBP")),
        }
    }

    /// Cross-attention IBP for encoder-decoder models (Whisper).
    ///
    /// Q (queries) from decoder: [batch, heads, seq_dec, dim]
    /// K, V from encoder: [batch, heads, seq_enc, dim]
    /// Output: [batch, heads, seq_dec, dim]
    pub fn cross_attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.cross_attention_ibp(q, k, v, scale),
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("cross-attention IBP")),
        }
    }
}

/// Minimum `m·k·n` MACs for the Cpu variant to offload `gemm_f32` to the
/// process-global accelerator ([`ny_propagate::fast_f32_gemm`], e.g. cuBLAS).
/// Below this the 20-core rayon faer paths the caller falls back to win
/// (measured crossover vs the ~17–21 µs warmed cuBLAS per-call floor); mirrors
/// the rationale of `SOUND_F64_GEMM_MIN_MACS` on the f64 seam.
const CPU_FAST_F32_GEMM_MIN_MACS: usize = 1 << 24;

impl GemmEngine for ComputeDevice {
    fn backend_provenance(&self) -> &'static str {
        match self {
            ComputeDevice::Cpu(_) => "compute-device-cpu",
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(device) if device.sound_gpu_authority_cached() => {
                "wgpu-qualified-crown"
            }
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(device) if device.charged_flush_authority_cached().is_some() => {
                "wgpu-qualified-crown-flush-charged"
            }
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => "wgpu-quarantined",
        }
    }

    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        match self {
            ComputeDevice::Cpu(_) => {
                // Large engine-routed GEMMs beat the CPU fallback on cuBLAS;
                // below the gate keep the existing Err so callers use their
                // (rayon-parallel) CPU implementations.
                if _m * _k * _n >= CPU_FAST_F32_GEMM_MIN_MACS {
                    if let Some(Ok(c)) =
                        ny_propagate::fast_f32_gemm::with_engine(|e| e.gemm_f32(_m, _k, _n, _a, _b))
                    {
                        return Ok(c);
                    }
                }
                Err(NyError::UnsupportedConfiguration(
                    "GEMM acceleration requested but backend is CPU".to_string(),
                ))
            }
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("GEMM")),
        }
    }

    fn gemm_f32_fast(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        // Same routing/gates as gemm_f32 (fast-vs-exact never changes WHERE a
        // GEMM runs, only which precision the accelerator uses); the wgpu
        // shader has no reduced-precision variant, so its fallback is exact.
        match self {
            ComputeDevice::Cpu(_) => {
                if _m * _k * _n >= CPU_FAST_F32_GEMM_MIN_MACS {
                    if let Some(Ok(c)) = ny_propagate::fast_f32_gemm::with_engine(|e| {
                        e.gemm_f32_fast(_m, _k, _n, _a, _b)
                    }) {
                        return Ok(c);
                    }
                }
                Err(NyError::UnsupportedConfiguration(
                    "GEMM acceleration requested but backend is CPU".to_string(),
                ))
            }
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("fast GEMM")),
        }
    }

    fn conv_transpose_2d(
        &self,
        _a_reshaped: &[f32],
        _weight_col: &[f32],
        _params: &ny_core::ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        match self {
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("transpose convolution")),
            _ => Err(NyError::UnsupportedOp(
                "conv_transpose_2d not supported by this backend".into(),
            )),
        }
    }

    fn conv_transpose_2d_pair_cached(
        &self,
        _a_lower: &[f32],
        _a_upper: &[f32],
        _weight_col: &std::sync::Arc<[f32]>,
        _params: &ny_core::ConvTranspose2dParams,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        match self {
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => {
                Err(wgpu_proof_quarantine("cached transpose-convolution pair"))
            }
            _ => {
                let lower = self.conv_transpose_2d(_a_lower, _weight_col, _params)?;
                let upper = self.conv_transpose_2d(_a_upper, _weight_col, _params)?;
                Ok((lower, upper))
            }
        }
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn ny_core::GpuCrownBackward> {
        match self {
            ComputeDevice::Cpu(_) => None,
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(device) => device.as_gpu_crown_backward(),
        }
    }

    fn as_gpu_ibp_forward(&self) -> Option<&dyn ny_core::GpuIbpForward> {
        None
    }

    fn as_gpu_ibp_forward_ext(&self) -> Option<&dyn ny_core::GpuIbpForwardExt> {
        None
    }

    fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn ny_core::GpuDagIbpForwardExt> {
        None
    }
}

impl AcceleratedBoundPropagation for ComputeDevice {
    fn linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.linear_ibp(input, weight, bias),
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("linear IBP")),
        }
    }

    fn matmul_ibp(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.matmul_ibp(input_a, input_b),
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("matrix-multiply IBP")),
        }
    }

    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.crown_per_position_parallel(graph, input),
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(_) => Err(wgpu_proof_quarantine("parallel CROWN")),
        }
    }
}

#[cfg(test)]
mod fast_f32_seam_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Correct naive RN-f32 GEMM that counts invocations. Must be numerically
    /// correct: the process-global engine stays installed for the whole test
    /// process, so any later caller must still get valid results.
    struct CountingGemm {
        hits: AtomicUsize,
    }

    impl GemmEngine for CountingGemm {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            let mut c = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for p in 0..k {
                        s += a[i * k + p] * b[p * n + j];
                    }
                    c[i * n + j] = s;
                }
            }
            Ok(c)
        }
    }

    /// Cpu-variant routing: >= the MACs gate goes to the installed accelerator,
    /// below it keeps the historical Err (callers use their own CPU paths).
    /// The assertion body runs in an exact-test child because the
    /// `fast_f32_gemm` registry is process-wide and immutable after first use.
    #[test]
    fn cpu_variant_offloads_large_f32_gemm_and_rejects_small() {
        const CHILD_MARKER: &str = "NY_GPU_FAST_F32_SEAM_CHILD";
        const TEST_NAME: &str =
            "backend::fast_f32_seam_tests::cpu_variant_offloads_large_f32_gemm_and_rejects_small";

        if std::env::var_os(CHILD_MARKER).as_deref() != Some(std::ffi::OsStr::new("1")) {
            let output = std::process::Command::new(
                std::env::current_exe().expect("locate ny-gpu unit-test executable"),
            )
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_MARKER, "1")
            .output()
            .expect("spawn isolated fast-f32 seam test");
            assert!(
                output.status.success(),
                "isolated fast-f32 seam test failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }

        let engine = Arc::new(CountingGemm {
            hits: AtomicUsize::new(0),
        });
        ny_propagate::fast_f32_gemm::set_fast_f32_gemm_engine(engine.clone());

        let device = ComputeDevice::new(Backend::Cpu).expect("cpu device");

        // Small: below the gate the accelerator must NOT be consulted.
        let small = device.gemm_f32(2, 2, 2, &[1.0; 4], &[1.0; 4]);
        assert!(
            small.is_err(),
            "sub-gate Cpu gemm_f32 must keep returning Err"
        );
        assert_eq!(engine.hits.load(Ordering::SeqCst), 0);

        // Large: 256^3 = 2^24 MACs = exactly the gate; must route to the engine.
        assert!(
            ny_propagate::fast_f32_gemm::with_engine(|_| ()).is_some(),
            "isolated test must install the counting fast-f32 engine"
        );
        let dim = 256;
        let a = vec![0.5f32; dim * dim];
        let b = vec![2.0f32; dim * dim];
        let c = device
            .gemm_f32(dim, dim, dim, &a, &b)
            .expect("super-gate Cpu gemm_f32 must offload");
        assert_eq!(engine.hits.load(Ordering::SeqCst), 1);
        // 0.5 * 2.0 summed over k=256 => 256.0 exactly in f32.
        assert!(
            c.iter().all(|&v| v == 256.0),
            "offloaded result must be the real product"
        );
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod bab_bound_api_routing_tests {
    use super::*;

    #[test]
    fn distinct_bab_router_refuses_before_gpu_initialization_while_default_dark() {
        let result = ComputeDevice::new_for_proof_bab_bound(WgpuBabBoundVerdictRequest::new());
        let error = match result {
            Ok(_) => panic!("default-dark retained-BaB router unexpectedly created a device"),
            Err(error) => error,
        };
        assert!(error.verdict_report().is_none());
        assert!(error.to_string().contains("source gate is closed"));
    }

    #[test]
    fn explicit_request_and_router_signatures_are_not_generic_verdict_routes() {
        let _device_constructor: fn(
            WgpuBabBoundVerdictRequest,
        ) -> std::result::Result<
            WgpuDevice,
            WgpuBabBoundQualificationError,
        > = WgpuDevice::new_for_verdict_bab_bound;
        let _router: fn(
            WgpuBabBoundVerdictRequest,
        )
            -> std::result::Result<ComputeDevice, WgpuBabBoundQualificationError> =
            ComputeDevice::new_for_proof_bab_bound;
        let _ordinary: fn(Backend) -> Result<ComputeDevice> = ComputeDevice::new;
        let _full: fn(
            WgpuVerdictRequest,
        )
            -> std::result::Result<ComputeDevice, WgpuVerdictQualificationError> =
            ComputeDevice::new_for_proof;
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod dag_ibp_routing_tests {
    use super::*;

    /// Ordinary construction is never proof authority, even on a conformant
    /// adapter and regardless of ambient process state.
    #[test]
    fn ordinary_compute_device_wgpu_quarantines_all_gpu_resident_accessors() {
        let device = ComputeDevice::new(Backend::Wgpu)
            .expect("live WGPU quarantine test requires a usable device");
        assert_eq!(device.backend_provenance(), "wgpu-quarantined");
        assert!(device.wgpu_verdict_report().is_none());
        assert!(device.as_gpu_dag_ibp_forward_ext().is_none());
        assert!(device.as_gpu_ibp_forward().is_none());
        assert!(device.as_gpu_ibp_forward_ext().is_none());
        assert!(device.as_gpu_crown_backward().is_none());
        match &device {
            ComputeDevice::Wgpu(device) => {
                assert!(
                    !ny_core::GpuCrownBackward::provides_sound_gpu_bab_bound_phase(device.as_ref())
                );
                assert!(
                    ny_core::GpuCrownBackward::gpu_bab_bound_numerical_tcb(device.as_ref())
                        .is_none()
                );
            }
            ComputeDevice::Cpu(_) => unreachable!("requested WGPU backend"),
        }
    }

    /// A qualified proof device opens exactly the CROWN accessor. All broader
    /// engine and resident-forward surfaces remain closed.
    #[test]
    fn qualified_compute_device_opens_only_crown_backward() {
        let device = ComputeDevice::new_for_proof(WgpuVerdictRequest::new())
            .expect("gpu-tests proof routing requires a conformant WGPU adapter");
        let report = device
            .wgpu_verdict_report()
            .expect("qualified ComputeDevice must retain its report");
        assert!(report.qualified());
        assert_eq!(device.backend_provenance(), "wgpu-qualified-crown");
        let crown = device
            .as_gpu_crown_backward()
            .expect("qualified WGPU exposes ordinary sound CROWN");
        assert!(!crown.provides_sound_gpu_bab_bound_phase());
        assert!(crown.gpu_bab_bound_numerical_tcb().is_none());
        assert!(device.as_gpu_dag_ibp_forward_ext().is_none());
        assert!(device.as_gpu_ibp_forward().is_none());
        assert!(device.as_gpu_ibp_forward_ext().is_none());
        assert!(device.gemm_f32(1, 1, 1, &[1.0], &[1.0]).is_err());
    }
}
