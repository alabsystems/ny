// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// Inherent kernels remain exercised by crate-internal `gpu-tests`; the public
// `GemmEngine` implementation below deliberately fails closed.
#![allow(dead_code)]

use ny_core::{ConvTranspose2dParams, GemmEngine, GpuCrownBackward, NyError, Result};

use super::super::WgpuDevice;
use super::gpu_checked_u32;
use crate::wgpu_device::params::GemmParams;

/// wgpu default `max_storage_buffer_binding_size` (128 MB).
///
/// Each bind group entry must reference a buffer range within this limit.
/// With the BufferPool 1.2× growth factor, the effective max element count
/// per buffer is `WGPU_MAX_BINDING_BYTES / 1.2 / sizeof(f32)`.
///
/// Reference: `crown_backward.rs` uses the same constant for spec batching.
/// `pub(crate)`: the FL value tier (`crate::fl_value_gemm`) derives its
/// row-chunk budget from this same constant (#fl-value-gpu-tier).
pub(crate) const WGPU_MAX_BINDING_BYTES: usize = 134_217_728;

/// Maximum f32 elements per buffer that will fit within the binding limit
/// after the BufferPool 1.2× growth factor.
///
/// `128 MB / 1.2 / 4 = 128 MB × 5/6 / 4 = 27,962,026` f32 elements.
/// (1/1.2 = 5/6 exactly, so this is precise integer arithmetic.)
/// `pub(crate)`: shared with `crate::fl_value_gemm` (#fl-value-gpu-tier).
pub(crate) const MAX_BINDING_ELEMS: usize = WGPU_MAX_BINDING_BYTES * 5 / 6 / size_of::<f32>();

impl WgpuDevice {
    /// Max f32 elements per storage binding, from the LIVE device limits
    /// (#hard-caps, 2026-08-06).
    ///
    /// [`MAX_BINDING_ELEMS`] is derived from a hard-coded 128 MiB, which is
    /// wgpu's *default* `max_storage_buffer_binding_size`. But `WgpuDevice::new`
    /// requests the ADAPTER's real limits by default (`NY_GPU_BIG_BINDINGS`,
    /// which is on unless explicitly set to `0`), and `device.rs` states the
    /// intent outright: "Buffer-fit gates elsewhere read `device.limits()` live."
    /// This gate did not, so it refused work the device could do.
    ///
    /// Measured on an Apple M4 Pro, 2026-08-06: the granted binding size is
    /// **4095 MiB** and `max_buffer_size` is 13639 MiB, against the constant's
    /// 128 MiB — a 32x under-estimate. The conv-CROWN speedup sweep hit exactly
    /// that wall: 15.52x at the largest shape that ran, then a refusal at the
    /// next size up, on the very path all four scoreboard deficits share.
    ///
    /// Falls back to the constant when the granted limit is smaller (a device
    /// with tighter limits than wgpu's default), so this can only ever RAISE the
    /// ceiling relative to today, never lower it.
    pub(crate) fn max_binding_elems_live(&self) -> usize {
        let granted = self.device.limits().max_storage_buffer_binding_size as usize;
        let live = granted * 5 / 6 / size_of::<f32>();
        live.max(MAX_BINDING_ELEMS)
    }

    /// Byte twin of [`Self::max_binding_elems_live`], for diagnostics.
    pub(crate) fn max_binding_bytes_live(&self) -> usize {
        (self.device.limits().max_storage_buffer_binding_size as usize).max(WGPU_MAX_BINDING_BYTES)
    }
}

/// Output tile dimension for the GEMM shaders (= workgroup_size per dimension).
///
/// Both the tiled and small-K shaders use workgroup_size(16, 16) = 256 threads.
/// The tiled shader covers TILE_DIM × TILE_DIM outputs per workgroup.
/// The small-K shader covers TILE_DIM × (TILE_DIM × SMALL_K_ROWS_PER_THREAD).
pub(super) const GEMM_TILE_DIM: usize = 16;

/// K dimension threshold for small-K GEMM shader eligibility.
///
/// The small-K shader (no shared memory, no barriers, ROWS_PER_THREAD=4) is
/// only used as a dispatch-limit fallback when the tiled shader's workgroup Y
/// would exceed 65535 (i.e., M > 65535 × TILE_DIM = 1,048,560).
///
/// Benchmarks at commit f249911c9 showed the tiled shader is always faster on
/// Apple Silicon (unified memory makes shared-memory reuse cheap, barriers
/// nearly free). The small-K shader regressed metaroom by 23% (34.3s vs 28.0s)
/// when used unconditionally. It remains necessary for soundnessbench Conv2
/// (M=1,572,864) where tiled dispatch Y = 98,304 > 65,535.
///
/// Reference: benchmark data at gpu_crown_backward_timing_smallk_*.csv.
pub(super) const GEMM_SMALL_K_THRESHOLD: u32 = 64;

/// Rows per thread in the small-K GEMM shader.
///
/// Must match `ROWS_PER_THREAD` constant in `GEMM_F32_SMALL_K_SHADER`.
pub(super) const SMALL_K_ROWS_PER_THREAD: usize = 4;

/// Maximum M rows per dispatch of the TILED shader: wgpu requires each
/// dispatch dimension to be ≤ 65535 and its Y = ceil(M/16), so
/// 65535 × 16 = 1,048,560.
const MAX_M_FOR_TILED_DISPATCH: usize = 65535 * GEMM_TILE_DIM;

/// Maximum M rows per dispatch of the SMALL-K shader: Y = ceil(M/64), so
/// 65535 × 64 = 4,194,240.
const MAX_M_FOR_SMALL_K_DISPATCH: usize = 65535 * GEMM_TILE_DIM * SMALL_K_ROWS_PER_THREAD;

/// Maximum M rows ONE dispatch can cover for contraction length `k`: the grid
/// limit of the shader [`select_gemm_dispatch`] can pick for that `k`.
///
/// Only `k ≤ GEMM_SMALL_K_THRESHOLD` can reach the small-K shader; above it
/// the tiled shader is the only choice, so a batch of more than 1,048,560 rows
/// is refused by `gemm_f32_single`'s defensive dispatch check. Capping every
/// `k` at the small-K limit used to be harmless only because the static
/// 128 MiB binding split (`MAX_BINDING_ELEMS / k`, at most 430,185 rows for
/// k ≥ 65) always bound first; once the split reads the live binding limit
/// (#hard-caps) it no longer does, so the cap must follow the shader.
const fn max_m_for_dispatch(k: usize) -> usize {
    if k <= GEMM_SMALL_K_THRESHOLD as usize {
        MAX_M_FOR_SMALL_K_DISPATCH
    } else {
        MAX_M_FOR_TILED_DISPATCH
    }
}

/// Rows per M-batch for `WgpuDevice::gemm_f32`: bounded by the A binding
/// (rows × k), the output binding (rows × n), and the dispatch grid of the
/// shader chosen for `k`.
///
/// `max_elems` must be the SAME budget the caller's B gate applied. With
/// `m, k, n > 0` and `k·n ≤ max_elems` (the gate), `k ≤ max_elems` and
/// `n ≤ max_elems`, so the result is ≥ 1. A split on a smaller budget than the
/// gate (for example the static `MAX_BINDING_ELEMS` under a live gate) gives 0
/// rows for k or n above it, and `gemm_f32_batched` would never advance.
///
/// Pure, so the batch/dispatch agreement is pinned without a GPU.
fn gemm_batch_rows(m: usize, k: usize, n: usize, max_elems: usize) -> usize {
    let max_m_for_a = max_elems.checked_div(k).unwrap_or(m);
    let max_m_for_out = max_elems.checked_div(n).unwrap_or(m);
    max_m_for_a
        .min(max_m_for_out)
        .min(max_m_for_dispatch(k))
        .min(m)
}

#[derive(Clone, Copy)]
struct CheckedGemmLengths {
    lhs: usize,
    rhs: usize,
    output: usize,
    rhs_bytes: usize,
    output_bytes: usize,
}

fn checked_gemm_lengths(m: usize, k: usize, n: usize) -> Result<CheckedGemmLengths> {
    let overflow = |product: &str| {
        NyError::InvalidSpec(format!(
            "wgpu gemm_f32: {product} overflows usize for shape {m}x{k}x{n}"
        ))
    };
    let lhs = m.checked_mul(k).ok_or_else(|| overflow("m*k"))?;
    let rhs = k.checked_mul(n).ok_or_else(|| overflow("k*n"))?;
    let output = m.checked_mul(n).ok_or_else(|| overflow("m*n"))?;
    let rhs_bytes = rhs
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| overflow("k*n*sizeof(f32)"))?;
    let output_bytes = output
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| overflow("m*n*sizeof(f32)"))?;
    Ok(CheckedGemmLengths {
        lhs,
        rhs,
        output,
        rhs_bytes,
        output_bytes,
    })
}

fn allocate_host_gemm_output(shape: CheckedGemmLengths) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(shape.output)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: shape.output_bytes,
            budget_bytes: usize::MAX,
            site: "ny-gpu::wgpu::gemm_f32/output",
        })?;
    output.resize(shape.output, 0.0);
    Ok(output)
}

/// GEMM dispatch parameters computed from matrix dimensions.
///
/// Pure function output — no GPU state needed. Used by both the standalone
/// `gemm_f32_single()` path and the CROWN backward encoder to ensure the
/// same shader-selection rule applies everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GemmDispatch {
    /// Whether the small-K shader should be used instead of the tiled shader.
    pub use_small_k: bool,
    /// Workgroup count in X (covers N columns): `ceil(N / TILE_DIM)`.
    pub wg_x: u32,
    /// Workgroup count in Y (covers M rows): depends on shader selection.
    pub wg_y: u32,
    /// Effective M tile size (TILE_DIM for tiled, TILE_DIM*ROWS_PER_THREAD for small-K).
    pub m_tile: usize,
}

/// Select GEMM shader and dispatch dimensions based on matrix shape.
///
/// Prefers the tiled shader (shared-memory B reuse is faster on Apple Silicon).
/// Falls back to the small-K shader only when K ≤ `GEMM_SMALL_K_THRESHOLD`
/// AND the tiled shader's workgroup Y would exceed the 65535 wgpu dispatch limit.
///
/// This is the single source of truth for dispatch selection — both
/// `gemm_f32_single()` and `gemm_pipeline_and_dispatch()` call this.
///
/// Reference: benchmark evidence at commits f249911c9, ff03f9d16.
pub(super) fn select_gemm_dispatch(m: u32, k: u32, n: u32) -> GemmDispatch {
    let wg_x = n.div_ceil(GEMM_TILE_DIM as u32);
    let tiled_wg_y = m.div_ceil(GEMM_TILE_DIM as u32);

    if k <= GEMM_SMALL_K_THRESHOLD && tiled_wg_y > 65535 {
        let m_tile = (GEMM_TILE_DIM * SMALL_K_ROWS_PER_THREAD) as u32;
        GemmDispatch {
            use_small_k: true,
            wg_x,
            wg_y: m.div_ceil(m_tile),
            m_tile: GEMM_TILE_DIM * SMALL_K_ROWS_PER_THREAD,
        }
    } else {
        GemmDispatch {
            use_small_k: false,
            wg_x,
            wg_y: tiled_wg_y,
            m_tile: GEMM_TILE_DIM,
        }
    }
}

impl WgpuDevice {
    pub(crate) fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        let shape = checked_gemm_lengths(m, k, n)?;
        if a.len() != shape.lhs {
            return Err(NyError::shape_mismatch(vec![m, k], vec![a.len()]));
        }
        if b.len() != shape.rhs {
            return Err(NyError::shape_mismatch(vec![k, n], vec![b.len()]));
        }
        if m == 0 || k == 0 || n == 0 {
            return allocate_host_gemm_output(shape);
        }

        // Check if any buffer exceeds the binding limit after 1.2× growth.
        // B matrix (k×n) cannot be split, so if it alone exceeds the limit,
        // return an error to let the caller fall back to CPU.
        //
        // #hard-caps: the LIVE device limit, not wgpu's 128 MiB default. The
        // sweep that moved the conv_transpose gate (d153bf0a4), crown_backward
        // (664af6bf) and the FL value tier's chunk budget (16034c8fc) to
        // `max_binding_elems_live` missed this gate, so the FL tier admitted B
        // up to half the live capacity and this gate then refused everything
        // past 112 MB (M5 Max: 4 GiB - 4 B granted, 1.79 GB admitted).
        let max_elems = self.max_binding_elems_live();
        if shape.rhs > max_elems {
            return Err(NyError::GpuMemoryExceeded {
                required_bytes: shape.rhs_bytes,
                budget_bytes: self.max_binding_bytes_live(),
            });
        }

        // Max M rows per batch: limited by the A buffer (batch_m × k), the
        // output buffer (batch_m × n), and the dispatch grid of the shader
        // selected for k. The gate above and both splits read the SAME
        // `max_elems`, so batch_m ≥ 1 (see `gemm_batch_rows`).
        //
        // With the live limit, one large-A or large-output call can leave each
        // `gemm_*` pool slot holding up to the granted binding size (about
        // 4 GiB on Apple M-series; host RAM on unified memory) instead of at most
        // 128 MiB, as conv_transpose's live gate already allows. The
        // #gpu-pool-highwater rule reallocates a slot at the next request
        // that is much smaller.
        let batch_m = gemm_batch_rows(m, k, n, max_elems);
        if batch_m == 0 {
            // Unreachable while the gate and the splits share `max_elems`;
            // refuse rather than let `gemm_f32_batched` loop without advancing.
            return Err(NyError::InternalError(format!(
                "wgpu gemm_f32: zero-row batch for {m}x{k}x{n} (binding budget {max_elems} elements)"
            )));
        }

        // Wrap the GPU work in an error scope so a wgpu validation/internal/OOM
        // error returns Err (caller falls back to CPU) instead of aborting the
        // process via wgpu's panicking uncaptured-error handler (#live bug).
        self.run_gpu_checked("gemm_f32", || {
            if batch_m < m {
                self.gemm_f32_batched(m, k, n, a, b, batch_m)
            } else {
                self.gemm_f32_single(m, k, n, a, b)
            }
        })
    }

    /// Run a single GEMM dispatch (no M-batching).
    ///
    /// Uses the tiled shader by default. Falls back to the small-K shader only
    /// when K ≤ threshold AND the tiled shader would exceed the 65535 dispatch
    /// limit. See `GEMM_SMALL_K_THRESHOLD` doc for benchmark rationale.
    ///
    /// Caller must ensure M ≤ `max_m_for_dispatch(k)` and N dispatch fits in 65535.
    fn gemm_f32_single(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        let m_u32 = gpu_checked_u32(m, "gemm m")?;
        let k_u32 = gpu_checked_u32(k, "gemm k")?;
        let n_u32 = gpu_checked_u32(n, "gemm n")?;

        // Single source of truth for shader selection.
        let dispatch = select_gemm_dispatch(m_u32, k_u32, n_u32);

        // Defensive check: wgpu dispatch dimensions must be ≤ 65535.
        // M is capped by the caller via batch_m; N has no batching path,
        // so reject here with a clear error rather than a wgpu panic.
        if dispatch.wg_y > 65535 || dispatch.wg_x > 65535 {
            return Err(NyError::InternalError(format!(
                "wgpu GEMM dispatch exceeds 65535 limit: M={m} (wg_y={}), N={n} (wg_x={}), small_k={}",
                dispatch.wg_y,
                dispatch.wg_x,
                dispatch.use_small_k,
            )));
        }

        let shape = checked_gemm_lengths(m, k, n)?;
        let out_elems = shape.output;
        let params = GemmParams {
            m: m_u32,
            k: k_u32,
            n: n_u32,
            _padding: 0,
        };

        let mut pool = self.buffer_pool.lock().map_err(|e| {
            NyError::InternalError(format!("wgpu gemm_f32: buffer pool lock poisoned: {e}"))
        })?;

        if pool.gemm_params_buffer.is_none() {
            pool.gemm_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gemm_params_buffer"),
                size: size_of::<GemmParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let params_buffer = pool
            .gemm_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu gemm_f32: params buffer not created".into())
            })?
            .clone();

        let a_buffer = self.get_or_create_storage_buffer(
            &mut pool.gemm_a_buffer,
            a.len(),
            "gemm_a_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let b_buffer = self.get_or_create_storage_buffer(
            &mut pool.gemm_b_buffer,
            b.len(),
            "gemm_b_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let out_buffer = self.get_or_create_storage_buffer(
            &mut pool.gemm_out_buffer,
            out_elems,
            "gemm_out_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = self.get_or_create_storage_buffer(
            &mut pool.gemm_staging_buffer,
            out_elems,
            "gemm_staging_buffer",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Keep the shared buffer-pool lock alive until readback completes.
        // Otherwise concurrent Rayon GEMM calls can recycle `gemm_staging_buffer`
        // while a previous `map_async`/`unmap` cycle is still in flight, which
        // triggers wgpu's "Buffer... is still mapped" validation panic.
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        self.queue
            .write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm_f32_bind_group"),
            layout: &self.gemm_f32_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm_f32_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm_f32_pass"),
                timestamp_writes: None,
            });

            // Select pipeline and dispatch from shared selector.
            let pipeline = if dispatch.use_small_k {
                &self.gemm_f32_small_k_pipeline
            } else {
                &self.gemm_f32_pipeline
            };
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch.wg_x, dispatch.wg_y, 1);
        }

        let out_bytes = u64::try_from(shape.output_bytes).map_err(|_| {
            NyError::InvalidSpec(format!(
                "wgpu gemm_f32: output byte count {} exceeds u64",
                shape.output_bytes
            ))
        })?;
        encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        Self::read_buffer(&self.device, &staging, out_elems)
    }

    /// Split GEMM along M dimension when A or output buffers exceed binding limits.
    ///
    /// `C = A @ B` decomposes as `C[i..i+batch] = A[i..i+batch] @ B` because
    /// rows of A are independent. B is shared across all batches.
    ///
    /// Reference: soundnessbench Conv2d activations produce A-matrices of ~724MB,
    /// exceeding wgpu's 128MB per-binding limit. This M-batching avoids the panic.
    fn gemm_f32_batched(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        batch_m: usize,
    ) -> Result<Vec<f32>> {
        let shape = checked_gemm_lengths(m, k, n)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(shape.output)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: shape.output_bytes,
                budget_bytes: usize::MAX,
                site: "ny-gpu::wgpu::gemm_f32_batched/output",
            })?;
        let mut row_offset = 0;

        while row_offset < m {
            let chunk_m = batch_m.min(m - row_offset);
            let a_start = row_offset * k;
            let a_end = a_start + chunk_m * k;
            let chunk_result = self.gemm_f32_single(chunk_m, k, n, &a[a_start..a_end], b)?;
            result.extend_from_slice(&chunk_result);
            row_offset += chunk_m;
        }

        Ok(result)
    }
}

/// Crate-private adapter for numerical diagnostics that deliberately exercise
/// the raw WGPU GEMM kernel.
///
/// `WgpuDevice`'s public [`GemmEngine`] implementation must stay quarantined:
/// exposing raw GEMM there would let an ordinary device enter verdict-bearing
/// generic routes.  A few in-crate arithmetic oracles still need the trait's
/// backend-agnostic certified helpers, so they use this narrowly scoped
/// adapter instead.  It exposes no GPU proof capability and cannot escape the
/// private `wgpu_device::ops` module.
pub(super) struct WgpuDiagnosticGemm<'a> {
    device: &'a WgpuDevice,
}

impl<'a> WgpuDiagnosticGemm<'a> {
    pub(super) const fn new(device: &'a WgpuDevice) -> Self {
        Self { device }
    }
}

impl GemmEngine for WgpuDiagnosticGemm<'_> {
    fn backend_provenance(&self) -> &'static str {
        "wgpu-private-diagnostic"
    }

    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.device.gemm_f32(m, k, n, a, b)
    }
}

/// Quarantined proof-engine adapter for WGPU.
///
/// Crate-internal inherent kernels remain available to GPU diagnostic tests.
/// The public `GemmEngine` seam fails closed so a caller cannot
/// inject `WgpuDevice` directly and bypass `ComputeDevice`'s verdict quarantine.
impl GemmEngine for WgpuDevice {
    fn backend_provenance(&self) -> &'static str {
        if self.sound_gpu_authority_cached() {
            "wgpu-qualified-crown"
        } else if self.charged_flush_authority_cached().is_some() {
            // #flush-charge: every ledger row records the charged mode
            // distinctly from full qualification.
            "wgpu-qualified-crown-flush-charged"
        } else {
            "wgpu-quarantined"
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
        Err(NyError::UnsupportedConfiguration(
            "WGPU GEMM is quarantined from verdict-bearing GemmEngine routes".to_string(),
        ))
    }

    fn conv_transpose_2d(
        &self,
        _a_reshaped: &[f32],
        _weight_col: &[f32],
        _params: &ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        Err(NyError::UnsupportedConfiguration(
            "WGPU transpose convolution is quarantined from verdict-bearing GemmEngine routes"
                .to_string(),
        ))
    }

    fn conv_transpose_2d_pair_cached(
        &self,
        _a_lower: &[f32],
        _a_upper: &[f32],
        _weight_col: &std::sync::Arc<[f32]>,
        _params: &ConvTranspose2dParams,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Err(NyError::UnsupportedConfiguration(
            "WGPU cached transpose convolution is quarantined from verdict-bearing GemmEngine \
             routes"
                .to_string(),
        ))
    }

    /// The SECOND authority seam (the router only ever reaches
    /// `provides_sound_gpu_crown` through this accessor, so both must move
    /// together or qualification is inconsistent — pinned by the authority
    /// routing tests.
    ///
    /// Decided by the same reviewed source gate, exact request, and cached live
    /// ladder as `provides_sound_gpu_crown`, never by a backend name alone.
    /// Failed or uninitialized qualification remains `None`. The public
    /// `ComputeDevice`/CLI proof router consumes this exact accessor; its raw
    /// GEMM, IBP, convolution, and DAG accessors remain closed.
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        if self.sound_gpu_authority_cached() || self.charged_flush_authority_cached().is_some() {
            Some(self)
        } else {
            None
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

#[cfg(test)]
mod tests {
    use super::*;

    // Boundary: M = 65535 * 16 = 1,048,560 is the exact tiled dispatch limit.
    // tiled_wg_y = ceil(1_048_560 / 16) = 65535, which is NOT > 65535 → tiled.
    const M_EXACT_BOUNDARY: u32 = 65535 * 16; // 1,048,560

    #[test]
    fn checked_gemm_lengths_reject_overflow_without_gpu_state() {
        let huge = 1usize << (usize::BITS - 1);
        assert!(checked_gemm_lengths(huge, huge, huge).is_err());
        assert!(checked_gemm_lengths(usize::MAX / 2, 1, 1).is_err());
    }

    #[test]
    fn checked_gemm_lengths_preserve_empty_shape() {
        let shape = checked_gemm_lengths(usize::MAX, 0, 0).expect("empty shape is representable");
        assert_eq!(shape.lhs, 0);
        assert_eq!(shape.rhs, 0);
        assert_eq!(shape.output, 0);
    }

    #[test]
    fn test_select_gemm_dispatch_small_m_uses_tiled() {
        let d = select_gemm_dispatch(100, 32, 100);
        assert!(!d.use_small_k, "small M should use tiled shader");
        assert_eq!(d.m_tile, GEMM_TILE_DIM);
        assert_eq!(d.wg_y, 100_u32.div_ceil(16));
        assert_eq!(d.wg_x, 100_u32.div_ceil(16));
    }

    #[test]
    fn test_select_gemm_dispatch_boundary_exact_uses_tiled() {
        // M = 1,048,560: tiled_wg_y = 65535 (exactly at limit, not over)
        let d = select_gemm_dispatch(M_EXACT_BOUNDARY, 32, 288);
        assert!(!d.use_small_k, "M at exact boundary should use tiled");
        assert_eq!(d.wg_y, 65535);
        assert_eq!(d.m_tile, GEMM_TILE_DIM);
    }

    #[test]
    fn test_select_gemm_dispatch_boundary_plus_one_uses_small_k() {
        // M = 1,048,561: tiled_wg_y = 65536 (exceeds limit)
        let d = select_gemm_dispatch(M_EXACT_BOUNDARY + 1, 32, 288);
        assert!(
            d.use_small_k,
            "M past boundary with small K should use small-K"
        );
        assert_eq!(d.m_tile, GEMM_TILE_DIM * SMALL_K_ROWS_PER_THREAD);
        // Small-K wg_y = ceil(1_048_561 / 64) = 16384
        assert!(d.wg_y <= 65535, "small-K dispatch must fit in 65535");
    }

    #[test]
    fn test_select_gemm_dispatch_boundary_plus_one_large_k_uses_tiled() {
        // M = 1,048,561 but K = 65 (above threshold): forced tiled even though
        // tiled_wg_y > 65535. This is the caller's responsibility to M-batch.
        let d = select_gemm_dispatch(M_EXACT_BOUNDARY + 1, 65, 288);
        assert!(!d.use_small_k, "K above threshold must stay tiled");
        assert_eq!(d.wg_y, 65536);
        assert_eq!(d.m_tile, GEMM_TILE_DIM);
    }

    #[test]
    fn test_select_gemm_dispatch_k_threshold_boundary() {
        // K = 64 (at threshold) with large M → small-K
        let d = select_gemm_dispatch(M_EXACT_BOUNDARY + 1, 64, 216);
        assert!(
            d.use_small_k,
            "K=64 at threshold with large M should use small-K"
        );

        // K = 65 (above threshold) with large M → tiled
        let d = select_gemm_dispatch(M_EXACT_BOUNDARY + 1, 65, 216);
        assert!(!d.use_small_k, "K=65 above threshold should use tiled");
    }

    #[test]
    fn test_select_gemm_dispatch_soundnessbench_shape() {
        // soundnessbench Conv2: M=1,572,864, K=24, N=216
        let d = select_gemm_dispatch(1_572_864, 24, 216);
        assert!(d.use_small_k, "soundnessbench shape should use small-K");
        assert!(d.wg_y <= 65535, "dispatch must fit in 65535 limit");
        assert_eq!(d.wg_y, 1_572_864_u32.div_ceil(64));
    }

    #[test]
    fn test_select_gemm_dispatch_metaroom_shape() {
        // metaroom: M=35,840, K=32, N=288 (below dispatch limit → tiled)
        let d = select_gemm_dispatch(35_840, 32, 288);
        assert!(!d.use_small_k, "metaroom shape should use tiled");
        assert_eq!(d.wg_y, 35_840_u32.div_ceil(16));
        assert!(d.wg_y <= 65535);
    }

    /// Largest binding budget any adapter can grant: wgpu's
    /// `max_storage_buffer_binding_size` is a `u32` byte count, after the
    /// pool's 1.2× growth: 894,784,853 elements. An M5 Max grants 4 GiB - 4 B
    /// (live budget 894,784,852 elements, measured).
    const LARGEST_LIVE_BUDGET: usize = (u32::MAX as usize) * 5 / 6 / size_of::<f32>();

    /// #hard-caps: every M-batch `gemm_f32` hands one dispatch must fit the
    /// grid of the shader `select_gemm_dispatch` picks for it, under the static
    /// budget AND the largest live one. Capping all k at the small-K limit
    /// handed k ≥ 65 batches of up to 4,194,240 rows to the tiled shader
    /// (limit 1,048,560): m = 1.1M, k = 65 came back as "dispatch exceeds 65535"
    /// and fell to the CPU once the split read the live limit.
    #[test]
    fn gemm_batch_rows_fit_the_selected_shader_dispatch() {
        for max_elems in [MAX_BINDING_ELEMS, LARGEST_LIVE_BUDGET] {
            for k in [1usize, 2, 24, 63, 64, 65, 66, 128, 1152] {
                for n in [1usize, 2, 8, 288] {
                    for m in [
                        1usize,
                        M_EXACT_BOUNDARY as usize,
                        M_EXACT_BOUNDARY as usize + 1,
                        1_100_000,
                        1_953_125,
                        MAX_M_FOR_SMALL_K_DISPATCH,
                        MAX_M_FOR_SMALL_K_DISPATCH + 1,
                        50_000_000,
                    ] {
                        let rows = gemm_batch_rows(m, k, n, max_elems);
                        let at = format!("m={m} k={k} n={n} budget={max_elems}: rows={rows}");
                        assert!(rows >= 1 && rows <= m, "{at}");
                        assert!(rows * k <= max_elems && rows * n <= max_elems, "{at}");
                        let rows_u32 = u32::try_from(rows).expect("batch rows fit u32");
                        let d = select_gemm_dispatch(rows_u32, k as u32, n as u32);
                        assert!(
                            d.wg_y <= 65535,
                            "{at}: wg_y={} small_k={}",
                            d.wg_y,
                            d.use_small_k
                        );
                    }
                }
            }
        }
    }

    /// The splits read the budget the B gate read. A k (or n) past wgpu's
    /// DEFAULT 128 MiB wall but inside a live budget must still get rows: a
    /// split on the static `MAX_BINDING_ELEMS` under a live gate gives 0 here,
    /// and `gemm_f32_batched` never advances.
    #[test]
    fn gemm_batch_rows_split_on_the_gate_budget() {
        let past_static = MAX_BINDING_ELEMS + 1;
        assert_eq!(gemm_batch_rows(3, past_static, 1, LARGEST_LIVE_BUDGET), 3);
        assert_eq!(gemm_batch_rows(3, 1, past_static, LARGEST_LIVE_BUDGET), 3);
        assert_eq!(
            gemm_batch_rows(usize::MAX, past_static, 1, LARGEST_LIVE_BUDGET),
            LARGEST_LIVE_BUDGET / past_static
        );
        // At the gate's own edge (k·n == budget) one row still fits, on the
        // static budget and on the live one.
        for budget in [MAX_BINDING_ELEMS, LARGEST_LIVE_BUDGET] {
            assert_eq!(gemm_batch_rows(5, budget, 1, budget), 1);
            assert_eq!(gemm_batch_rows(5, 1, budget, budget), 1);
        }
    }

    #[test]
    fn max_m_for_dispatch_follows_the_shader_for_k() {
        assert_eq!(max_m_for_dispatch(1), MAX_M_FOR_SMALL_K_DISPATCH);
        assert_eq!(
            max_m_for_dispatch(GEMM_SMALL_K_THRESHOLD as usize),
            MAX_M_FOR_SMALL_K_DISPATCH
        );
        assert_eq!(
            max_m_for_dispatch(GEMM_SMALL_K_THRESHOLD as usize + 1),
            M_EXACT_BOUNDARY as usize
        );
        // The tiled limit is exactly the boundary the selector tests pin.
        assert_eq!(MAX_M_FOR_TILED_DISPATCH, M_EXACT_BOUNDARY as usize);
        let d = select_gemm_dispatch(MAX_M_FOR_SMALL_K_DISPATCH as u32, 64, 1);
        assert!(d.use_small_k && d.wg_y == 65535);
    }

    #[test]
    fn test_select_gemm_dispatch_consistent_wg_x() {
        // wg_x should always be ceil(N / TILE_DIM) regardless of shader choice
        for n in [1_u32, 16, 17, 100, 288, 1024] {
            let expected = n.div_ceil(GEMM_TILE_DIM as u32);
            let d_tiled = select_gemm_dispatch(100, 32, n);
            assert_eq!(d_tiled.wg_x, expected, "wg_x mismatch for N={n}");
        }
    }
}
