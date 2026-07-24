// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::accelerated::{AcceleratedBoundPropagation, AcceleratedDevice};
#[cfg(feature = "wgpu")]
use crate::wgpu_device::WgpuDevice;
use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

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
#[cfg(feature = "wgpu")]
#[must_use]
pub fn wgpu_adapter_available() -> bool {
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
            let hardware = info.device_type != wgpu::DeviceType::Cpu;
            tracing::info!(
                "GPU adapter probe: {} ({:?}, {:?}) => hardware GPU {}",
                info.name,
                info.device_type,
                info.backend,
                if hardware {
                    "available"
                } else {
                    "unavailable (software adapter)"
                }
            );
            hardware
        }
        Err(e) => {
            tracing::info!("GPU adapter probe: no adapter ({e}); using CPU");
            false
        }
    }
}

/// No-wgpu builds have no adapter to probe.
#[cfg(not(feature = "wgpu"))]
#[must_use]
pub fn wgpu_adapter_available() -> bool {
    false
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
            ComputeDevice::Wgpu(d) => d.attention_ibp(q, k, v, scale),
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
            ComputeDevice::Wgpu(d) => d.causal_attention_ibp(q, k, v, scale),
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
            ComputeDevice::Wgpu(d) => d.cross_attention_ibp(q, k, v, scale),
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
            ComputeDevice::Wgpu(d) => {
                // No MACs gate here: this substitutes cuBLAS for the wgpu WGSL
                // shader (never for CPU — the caller already chose the engine
                // path), and cuBLAS is measured faster than the shader at every
                // shape (floors ~17 µs vs ~63 µs; 2–3.4× at hotspot sizes).
                // Numerics stay IEEE RN-f32 (the accelerator's contract), which
                // the verdict-feeding IBP call sites certify with
                // order-independent ULP widening. Err falls back to the shader.
                if let Some(Ok(c)) =
                    ny_propagate::fast_f32_gemm::with_engine(|e| e.gemm_f32(_m, _k, _n, _a, _b))
                {
                    return Ok(c);
                }
                d.gemm_f32(_m, _k, _n, _a, _b)
            }
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
            ComputeDevice::Wgpu(d) => {
                if let Some(Ok(c)) = ny_propagate::fast_f32_gemm::with_engine(|e| {
                    e.gemm_f32_fast(_m, _k, _n, _a, _b)
                }) {
                    return Ok(c);
                }
                d.gemm_f32(_m, _k, _n, _a, _b)
            }
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
            ComputeDevice::Wgpu(d) => d.conv_transpose_2d(_a_reshaped, _weight_col, _params),
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
            // Route to the WgpuDevice fused plan cache (weight uploaded once, lower
            // and upper fused into ONE 2*S dispatch). Without this forward the enum
            // hit the trait default (two separate dispatches) — #4276 T1.0.
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(d) => {
                d.conv_transpose_2d_pair_cached(_a_lower, _a_upper, _weight_col, _params)
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
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(d) => d.as_gpu_crown_backward(),
            _ => None,
        }
    }

    fn as_gpu_ibp_forward(&self) -> Option<&dyn ny_core::GpuIbpForward> {
        match self {
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(d) => d.as_gpu_ibp_forward(),
            _ => None,
        }
    }

    fn as_gpu_ibp_forward_ext(&self) -> Option<&dyn ny_core::GpuIbpForwardExt> {
        match self {
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(d) => d.as_gpu_ibp_forward_ext(),
            _ => None,
        }
    }

    fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn ny_core::GpuDagIbpForwardExt> {
        match self {
            // Forward the cached graph-DAG GPU-resident IBP planner. Without this the
            // enum returned the trait default `None`, silently CPU-falling the whole
            // GPU-resident DAG IBP forward path (#4276, #4318 T1.0).
            #[cfg(feature = "wgpu")]
            ComputeDevice::Wgpu(d) => d.as_gpu_dag_ibp_forward_ext(),
            _ => None,
        }
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
            ComputeDevice::Wgpu(d) => d.linear_ibp(input, weight, bias),
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
            ComputeDevice::Wgpu(d) => d.matmul_ibp(input_a, input_b),
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
            ComputeDevice::Wgpu(d) => d.crown_per_position_parallel(graph, input),
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
    /// One test covers both directions because the fast_f32_gemm global is
    /// process-wide and install order across tests is not guaranteed.
    #[test]
    fn cpu_variant_offloads_large_f32_gemm_and_rejects_small() {
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
        // (Skip gracefully if another test materialized the global as None
        // before our install — process-global OnceLock, order-dependent.)
        if ny_propagate::fast_f32_gemm::with_engine(|_| ()).is_none() {
            eprintln!("skipping: fast_f32_gemm global already materialized empty");
            return;
        }
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

#[cfg(all(test, feature = "gpu-tests"))]
mod dag_ibp_routing_tests {
    use super::*;

    /// The `ComputeDevice` enum must FORWARD the optional GPU-resident accessors to
    /// its wgpu backend. Before #4276/#4318 T1.0 it dropped
    /// `as_gpu_dag_ibp_forward_ext` (→ trait default `None`) and
    /// `conv_transpose_2d_pair_cached` (→ default two-dispatch), so a GPU-resident
    /// DAG IBP forward silently ran on the CPU even with a GPU present. Assert the
    /// wgpu variant now reports the planner; the sibling ext accessors are checked as
    /// a routing sanity net against a future re-drop.
    #[test]
    fn compute_device_wgpu_forwards_gpu_resident_accessors() {
        let device = match ComputeDevice::new(Backend::Wgpu) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no wgpu device available ({e})");
                return;
            }
        };
        assert!(
            device.as_gpu_dag_ibp_forward_ext().is_some(),
            "ComputeDevice::Wgpu must forward the graph-DAG GPU-resident IBP planner \
             (was None pre-T1.0 → silent CPU fallback)"
        );
        assert!(device.as_gpu_ibp_forward().is_some());
        assert!(device.as_gpu_ibp_forward_ext().is_some());
        assert!(device.as_gpu_crown_backward().is_some());
    }
}
