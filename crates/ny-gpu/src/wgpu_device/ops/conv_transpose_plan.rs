// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// Retained for crate-internal `gpu-tests` diagnostics while WGPU is quarantined
// from the public verdict-bearing engine seam.
#![allow(dead_code)]

//! GPU-resident plan cache for the fused conv_transpose_2d op (#perf dispatch wall).
//!
//! The fused `conv_transpose_2d` (GEMM + col2im, see `conv_transpose.rs`) is
//! invoked twice per Conv2d CROWN-backward layer — once for `lower_a`, once for
//! `upper_a` — with the *same* weight column matrix. The original code
//! re-allocated ~7 buffers, **re-uploaded the static weight column**, and did a
//! blocking readback on every call, with no plan reuse. For deep convolutional
//! networks this per-call dispatch overhead dominates.
//!
//! This module adds a plan cache keyed by **topology + weight content**. A
//! cached [`PreparedConvTransposePlan`] keeps the weight column matrix
//! **GPU-resident** (uploaded once, never re-uploaded) and reuses all device
//! buffers and bind groups across calls. The dynamic `a_reshaped` input is the
//! only thing re-uploaded per call (it genuinely changes per call).
//!
//! ## Bit-identical / soundness
//!
//! This is a **pure performance** change: it alters *how buffers are reused*,
//! never *what is computed*. The shaders, dispatch dimensions, GEMM
//! associativity (along the unchanged `OC` reduction axis), and col2im scatter
//! are identical to the non-cached path. Reusing a resident weight buffer is
//! bit-identical to re-uploading it exactly when the resident bytes ARE the bytes
//! this call would have uploaded — which is what the key plus its verification
//! establish:
//!
//! * The key mixes topology, `total_rows`, the weight length, and a hash of the
//!   weight **bytes**. It deliberately does NOT key on `Arc` pointer identity the
//!   way [`crown_plan_key`](super::crown_plan_key) does. That mirror only holds
//!   for `Arc`s the plan is guaranteed to outlive; this op's callers mint a
//!   fresh, short-lived `Arc` per call (`ops_transpose_gemm.rs` rebuilds each
//!   group's weight column and drops it at the end of the iteration), so an
//!   address is free before the next key is computed and the allocator may hand
//!   it straight back. A pointer hit here would therefore be an alias — a
//!   different layer's or group's weights at a recycled address — not an
//!   identity. Content keying also makes the hit a real one: a fresh `Arc` per
//!   call could never legitimately match a pointer-keyed entry.
//! * The hash is a bucket, never the authority. Every hit re-checks the incoming
//!   weights against `weight_col` — the retained `Arc<[f32]>` the resident buffer
//!   was uploaded from — bit for bit, so a colliding key rebuilds and replaces
//!   rather than serving the wrong weights. Bit equality, not `f32` `PartialEq`:
//!   equal bytes ⇒ an identical upload ⇒ an identical result, while `0.0` vs
//!   `-0.0` and differing NaN payloads simply key apart (conservative, never
//!   wrong).
//!
//! ## Fused lower/upper pair
//!
//! Because the GEMM is purely row-wise — `(rows, OC) × (OC, IC*KH*KW)` — and
//! col2im scatters each spec row independently, stacking the `lower_a` rows
//! above the `upper_a` rows into a single `2*S*OH*OW`-row input produces results
//! bit-identical to two separate `S*OH*OW`-row calls (every output element is a
//! function only of its own row's reduction over `OC`, which is unchanged). The
//! [`conv_transpose_2d_pair_cached`](WgpuDevice::conv_transpose_2d_pair_cached)
//! entry point exploits this to halve command submissions and readbacks.
//!
//! ## Cache-clear invariant (mirrors `clear_crown_plan_cache`)
//!
//! The cache is cleared between models via
//! [`clear_conv_transpose_plan_cache`](WgpuDevice::clear_conv_transpose_plan_cache),
//! itself invoked from `clear_crown_working_set`, alongside the CROWN plan
//! cache. With content keying this bounds the memory footprint (it frees the
//! previous model's resident weight buffers and retained weight `Arc`s); it is
//! not what keeps a hit correct — the bit-for-bit weight check above does that.

use std::collections::hash_map::{DefaultHasher, Entry};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ny_core::{ConvTranspose2dParams, NyError, Result};

use super::super::WgpuDevice;
use super::gemm::{select_gemm_dispatch, MAX_BINDING_ELEMS, WGPU_MAX_BINDING_BYTES};
use super::gpu_checked_u32;
use crate::wgpu_device::params::{ConvCol2imParams, GemmParams};

/// Cache key: conv topology + weight content + row count.
///
/// `weight_hash`/`weight_len` capture the weight column by CONTENT — a 64-bit
/// hash of its raw bytes — because the callers mint a fresh, short-lived `Arc`
/// per call, so only content can legitimately match across calls (module doc).
/// The hash is a bucket, never the authority: every hit is re-verified
/// bit-for-bit against the plan's retained [`PreparedConvTransposePlan::weight_col`]
/// before the resident buffer is reused. `total_rows` (= `num_specs * out_h *
/// out_w`, already doubled for the fused pair path) is part of the key because
/// it fixes the GEMM `M` dimension and therefore all derived buffer sizes and
/// dispatch dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ConvTransposePlanKey {
    topology_hash: u64,
    weight_hash: u64,
    weight_len: usize,
    total_rows: usize,
}

/// Bit-exact weight equality: equal bytes ⇒ the resident upload is exactly the
/// upload this call would have performed. `f32` `PartialEq` would be wrong in
/// both directions for that purpose (`-0.0 == 0.0`, `NaN != NaN`); distinct bit
/// patterns simply rebuild (conservative, never wrong).
fn weight_bytes_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && bytemuck::cast_slice::<f32, u8>(a) == bytemuck::cast_slice::<f32, u8>(b)
}

fn conv_transpose_plan_key(
    weight_col: &Arc<[f32]>,
    params: &ConvTranspose2dParams,
    total_rows: usize,
) -> ConvTransposePlanKey {
    let mut topology = DefaultHasher::new();
    params.out_channels.hash(&mut topology);
    params.in_channels.hash(&mut topology);
    params.out_h.hash(&mut topology);
    params.out_w.hash(&mut topology);
    params.in_h.hash(&mut topology);
    params.in_w.hash(&mut topology);
    params.kernel_h.hash(&mut topology);
    params.kernel_w.hash(&mut topology);
    params.stride_h.hash(&mut topology);
    params.stride_w.hash(&mut topology);
    params.pad_h.hash(&mut topology);
    params.pad_w.hash(&mut topology);

    // Weight content: hash the raw bytes. An address would be meaningless here —
    // the caller's `Arc` is dead before the next key is computed, so a recycled
    // allocation could alias different weights at the same address (module doc).
    let mut weight = DefaultHasher::new();
    bytemuck::cast_slice::<f32, u8>(&weight_col[..]).hash(&mut weight);

    ConvTransposePlanKey {
        topology_hash: topology.finish(),
        weight_hash: weight.finish(),
        weight_len: weight_col.len(),
        total_rows,
    }
}

/// A cached conv_transpose_2d plan: GPU-resident weight + reusable buffers.
///
/// The weight buffer `w_buf` is uploaded **once** at plan-build time and never
/// re-uploaded. All other buffers are sized for the topology and reused; only
/// the dynamic `a_reshaped` contents and the (constant) param buffers are
/// written per call.
pub(crate) struct PreparedConvTransposePlan {
    /// The exact weights the resident `w_buf` was uploaded from. Every cache hit
    /// is re-verified bit-for-bit against this before the plan is reused, so a
    /// 64-bit content-hash collision can never serve another tensor's weights;
    /// retaining the `Arc` also documents whose bytes the GPU buffer holds.
    weight_col: Arc<[f32]>,
    // These four buffers are referenced only through the pre-built bind groups
    // (which hold their own wgpu references). They are retained as fields to own
    // the GPU allocations for the plan's lifetime and document the resident set;
    // they are intentionally not read back through the field after `build`.
    /// GPU-resident weight column matrix — uploaded once, reused across calls.
    #[allow(dead_code)]
    w_buf: wgpu::Buffer,
    /// GEMM uniform params (constant for a given key; written once at build).
    #[allow(dead_code)]
    gemm_params_buf: wgpu::Buffer,
    /// col2im uniform params (constant for a given key; written once at build).
    #[allow(dead_code)]
    col2im_params_buf: wgpu::Buffer,
    /// Dynamic A input `(total_rows, OC)`; contents re-uploaded each call.
    a_buf: wgpu::Buffer,
    /// GEMM output / col2im input `(total_rows, IC*KH*KW)`; GPU-internal.
    #[allow(dead_code)]
    gemm_out_buf: wgpu::Buffer,
    /// col2im destination `(S, IC*IH*IW)`; GPU-internal, copied to staging.
    dst_buf: wgpu::Buffer,
    /// Host-readable staging buffer for the final result.
    staging_buf: wgpu::Buffer,
    /// Pre-built bind groups (stable: all referenced buffers are owned here).
    gemm_bind_group: wgpu::BindGroup,
    col2im_bind_group: wgpu::BindGroup,
    /// Cached dispatch geometry.
    dispatch_wg_x: u32,
    dispatch_wg_y: u32,
    use_small_k: bool,
    col2im_workgroups: u32,
    /// Element counts used for uploads / readback bounds-checking.
    out_elems: usize,
    a_len: usize,
}

impl PreparedConvTransposePlan {
    fn retained_device_bytes(&self) -> Result<usize> {
        let mut total = 0usize;
        for (label, buffer) in [
            ("weight", &self.w_buf),
            ("gemm_params", &self.gemm_params_buf),
            ("col2im_params", &self.col2im_params_buf),
            ("input", &self.a_buf),
            ("gemm_output", &self.gemm_out_buf),
            ("destination", &self.dst_buf),
            ("staging", &self.staging_buf),
        ] {
            let bytes = usize::try_from(buffer.size()).map_err(|_| {
                NyError::InternalError(format!(
                    "conv_transpose plan buffer `{label}` does not fit in usize"
                ))
            })?;
            total = total.checked_add(bytes).ok_or_else(|| {
                NyError::InternalError("conv_transpose plan byte count overflow".into())
            })?;
        }
        Ok(total)
    }
}

impl WgpuDevice {
    /// Checked retained bytes across every cached fused conv-transpose plan.
    pub(crate) fn conv_transpose_plan_cache_bytes(&self) -> Result<usize> {
        let cache = self.conv_transpose_plan_cache.lock().map_err(|err| {
            NyError::InternalError(format!("conv_transpose plan cache lock poisoned: {err}"))
        })?;
        cache.values().try_fold(0usize, |total, plan| {
            total
                .checked_add(plan.retained_device_bytes()?)
                .ok_or_else(|| {
                    NyError::InternalError("conv_transpose plan cache byte count overflow".into())
                })
        })
    }

    /// Clear the conv_transpose plan cache, freeing GPU-resident weight buffers
    /// and the retained weight `Arc`s.
    ///
    /// Mirrors [`clear_crown_plan_cache`](Self::clear_crown_plan_cache); called
    /// from `clear_crown_working_set` between models to bound the resident
    /// footprint (hits are content-verified, so this is memory hygiene, not a
    /// correctness precondition).
    pub(crate) fn clear_conv_transpose_plan_cache(&self) -> Result<()> {
        let mut cache = self.conv_transpose_plan_cache.lock().map_err(|err| {
            NyError::InternalError(format!("conv_transpose plan cache lock poisoned: {err}"))
        })?;
        cache.clear();
        Ok(())
    }

    /// Number of cached conv_transpose plans (test-only introspection).
    // Callers live in `ops/tests.rs` under `cfg(test)` + the `gpu-tests` feature,
    // so compile this exactly there (any(..) left it dead in non-test builds).
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn conv_transpose_plan_cache_len(&self) -> usize {
        self.conv_transpose_plan_cache
            .lock()
            .map(|c| c.len())
            .unwrap_or(0)
    }

    /// Look up or build a cached plan for `weight_col` + topology + `total_rows`.
    ///
    /// On a cache hit the weight buffer is **not** re-uploaded; a hit is served
    /// only after the incoming weights compare bit-for-bit equal to the plan's
    /// retained `weight_col`, so a content-hash collision rebuilds instead of
    /// running against the wrong resident weights. Returns the plan and `true`
    /// if this call built a plan (cache miss or collision), `false` on a
    /// verified hit. The boolean is consumed by tests to assert resident-buffer
    /// reuse.
    fn get_or_prepare_conv_transpose_plan(
        &self,
        weight_col: &Arc<[f32]>,
        params: &ConvTranspose2dParams,
        total_rows: usize,
    ) -> Result<(Arc<PreparedConvTransposePlan>, bool)> {
        let key = conv_transpose_plan_key(weight_col, params, total_rows);
        {
            let cache = self.conv_transpose_plan_cache.lock().map_err(|err| {
                NyError::InternalError(format!("conv_transpose plan cache lock poisoned: {err}"))
            })?;
            if let Some(plan) = cache.get(&key) {
                if weight_bytes_equal(&plan.weight_col, weight_col) {
                    return Ok((plan.clone(), false));
                }
                // Distinct weights collided on the 64-bit content hash: fall
                // through and rebuild; the stale entry is replaced below.
            }
        }

        let plan = Arc::new(PreparedConvTransposePlan::build(
            self, weight_col, params, total_rows,
        )?);
        let mut cache = self.conv_transpose_plan_cache.lock().map_err(|err| {
            NyError::InternalError(format!("conv_transpose plan cache lock poisoned: {err}"))
        })?;
        // A concurrent miss may have inserted first; keep the canonical entry if
        // it carries these exact weight bytes (dropping our duplicate), replace
        // it otherwise (hash collision). Report `true` (this call built a plan)
        // so callers never assume an upload was skipped on a miss. `false` is
        // returned only on the verified fast-path hit above.
        let entry = match cache.entry(key) {
            Entry::Occupied(mut occupied) => {
                if weight_bytes_equal(&occupied.get().weight_col, weight_col) {
                    occupied.get().clone()
                } else {
                    occupied.insert(plan.clone());
                    plan
                }
            }
            Entry::Vacant(vacant) => vacant.insert(plan).clone(),
        };
        Ok((entry, true))
    }

    /// Cached, GPU-resident fused conv_transpose_2d for a `(lower, upper)` pair.
    ///
    /// Stacks `a_lower` over `a_upper` into a single `2*S*OH*OW`-row GEMM +
    /// col2im dispatch (halving submissions/readbacks), keeps the weight column
    /// GPU-resident across calls, and reuses all buffers. Returns
    /// `(lower_result, upper_result)`, each `(S, IC*IH*IW)` row-major — bit
    /// identical to two separate [`conv_transpose_2d`](Self::conv_transpose_2d)
    /// calls.
    pub(crate) fn conv_transpose_2d_pair_cached(
        &self,
        a_lower: &[f32],
        a_upper: &[f32],
        weight_col: &Arc<[f32]>,
        params: &ConvTranspose2dParams,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.run_gpu_checked("conv_transpose_2d_pair_cached", || {
            let s = params.num_specs;
            let oc = params.out_channels;
            let ic = params.in_channels;
            let (oh, ow) = (params.out_h, params.out_w);
            let (ih, iw) = (params.in_h, params.in_w);
            let (kh, kw) = (params.kernel_h, params.kernel_w);
            let spatial = oh * ow;
            let single_rows = s * spatial;
            let kernel_cols = ic * kh * kw;
            let flat_input_dim = ic * ih * iw;
            let single_out_elems = s * flat_input_dim;

            // Degenerate shapes: match the non-cached fast path exactly.
            if single_rows == 0 || oc == 0 || kernel_cols == 0 {
                return Ok((
                    vec![0.0f32; single_out_elems],
                    vec![0.0f32; single_out_elems],
                ));
            }

            // Build the stacked input: lower rows then upper rows. Doubling the
            // GEMM M dimension is exact and bit-identical per row.
            let mut a_pair = Vec::with_capacity(a_lower.len() + a_upper.len());
            a_pair.extend_from_slice(a_lower);
            a_pair.extend_from_slice(a_upper);

            // Fuse by doubling num_specs: the col2im shader treats specs
            // independently, so 2*S specs over the stacked input produces the
            // lower results in specs [0, S) and upper results in specs [S, 2S).
            let pair_params = ConvTranspose2dParams {
                num_specs: 2 * s,
                ..*params
            };
            let total_rows = 2 * single_rows;

            let (plan, _built) =
                self.get_or_prepare_conv_transpose_plan(weight_col, &pair_params, total_rows)?;
            let full = plan.execute(self, &a_pair)?;

            // Split: first `single_out_elems` are lower, next are upper.
            debug_assert_eq!(full.len(), 2 * single_out_elems);
            let upper = full[single_out_elems..].to_vec();
            let mut lower = full;
            lower.truncate(single_out_elems);
            Ok((lower, upper))
        })
    }
}

impl PreparedConvTransposePlan {
    /// Build a plan: allocate buffers, upload the weight **once**, pre-build
    /// bind groups, and cache dispatch geometry. `total_rows` already accounts
    /// for any lower/upper fusion (the caller passes the GEMM `M`).
    fn build(
        device: &WgpuDevice,
        weight_col: &Arc<[f32]>,
        params: &ConvTranspose2dParams,
        total_rows: usize,
    ) -> Result<Self> {
        let s = params.num_specs;
        let oc = params.out_channels;
        let ic = params.in_channels;
        let (oh, ow) = (params.out_h, params.out_w);
        let (ih, iw) = (params.in_h, params.in_w);
        let (kh, kw) = (params.kernel_h, params.kernel_w);
        let kernel_cols = ic * kh * kw;
        let flat_input_dim = ic * ih * iw;
        let out_elems = s * flat_input_dim;

        // GEMM: (total_rows, OC) × (OC, IC*KH*KW) → (total_rows, IC*KH*KW)
        let gemm_m = total_rows;
        let gemm_k = oc;
        let gemm_n = kernel_cols;
        let gemm_out_elems = gemm_m * gemm_n;
        let a_len = gemm_m * gemm_k;

        if weight_col.len() != gemm_k * gemm_n {
            return Err(NyError::InvalidSpec(format!(
                "conv_transpose_2d_cached: weight_col.len()={} != OC*IC*KH*KW={}",
                weight_col.len(),
                gemm_k * gemm_n
            )));
        }

        let m_u32 = gpu_checked_u32(gemm_m, "conv_t2d_plan gemm_m")?;
        let k_u32 = gpu_checked_u32(gemm_k, "conv_t2d_plan gemm_k")?;
        let n_u32 = gpu_checked_u32(gemm_n, "conv_t2d_plan gemm_n")?;

        let dispatch = select_gemm_dispatch(m_u32, k_u32, n_u32);
        if dispatch.wg_y > 65535 || dispatch.wg_x > 65535 {
            return Err(NyError::InternalError(format!(
                "conv_transpose_2d_cached GEMM dispatch exceeds 65535: M={gemm_m}, N={gemm_n}",
            )));
        }

        if gemm_k * gemm_n > MAX_BINDING_ELEMS
            || gemm_m * gemm_k > MAX_BINDING_ELEMS
            || gemm_out_elems > MAX_BINDING_ELEMS
            || out_elems > MAX_BINDING_ELEMS
        {
            return Err(NyError::GpuMemoryExceeded {
                required_bytes: gemm_out_elems.max(out_elems) * size_of::<f32>(),
                budget_bytes: WGPU_MAX_BINDING_BYTES,
            });
        }

        let out_workgroups = out_elems.div_ceil(256);
        let col2im_workgroups = gpu_checked_u32(out_workgroups, "conv_t2d_plan col2im_workgroups")?;
        if col2im_workgroups > 65_535 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "conv_transpose_2d_cached col2im dispatch exceeds 65535: out_elems={out_elems}"
            )));
        }

        let gemm_params = GemmParams {
            m: m_u32,
            k: k_u32,
            n: n_u32,
            _padding: 0,
        };
        let col2im_params = ConvCol2imParams {
            num_specs: gpu_checked_u32(s, "conv_t2d_plan num_specs")?,
            flat_input_dim: gpu_checked_u32(flat_input_dim, "conv_t2d_plan flat_input_dim")?,
            out_h: gpu_checked_u32(oh, "conv_t2d_plan out_h")?,
            out_w: gpu_checked_u32(ow, "conv_t2d_plan out_w")?,
            in_channels: gpu_checked_u32(ic, "conv_t2d_plan in_channels")?,
            in_h: gpu_checked_u32(ih, "conv_t2d_plan in_h")?,
            in_w: gpu_checked_u32(iw, "conv_t2d_plan in_w")?,
            kernel_h: gpu_checked_u32(kh, "conv_t2d_plan kernel_h")?,
            kernel_w: gpu_checked_u32(kw, "conv_t2d_plan kernel_w")?,
            stride_h: gpu_checked_u32(params.stride_h, "conv_t2d_plan stride_h")?,
            stride_w: gpu_checked_u32(params.stride_w, "conv_t2d_plan stride_w")?,
            pad_h: gpu_checked_u32(params.pad_h, "conv_t2d_plan pad_h")?,
            pad_w: gpu_checked_u32(params.pad_w, "conv_t2d_plan pad_w")?,
            kernel_cols: gpu_checked_u32(kernel_cols, "conv_t2d_plan kernel_cols")?,
            _padding2: [0; 2],
        };

        let dev = &device.device;

        let gemm_params_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_plan_gemm_params"),
            size: size_of::<GemmParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let a_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_plan_a"),
            size: (a_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Weight buffer: uploaded once below and kept resident.
        let w_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_plan_w_resident"),
            size: (weight_col.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gemm_out_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_plan_gemm_out"),
            size: (gemm_out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let col2im_params_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_plan_col2im_params"),
            size: size_of::<ConvCol2imParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dst_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_plan_dst"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_plan_staging"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload the constant params and the resident weight ONCE.
        device
            .queue
            .write_buffer(&gemm_params_buf, 0, bytemuck::cast_slice(&[gemm_params]));
        device.queue.write_buffer(
            &col2im_params_buf,
            0,
            bytemuck::cast_slice(&[col2im_params]),
        );
        device
            .queue
            .write_buffer(&w_buf, 0, bytemuck::cast_slice(&weight_col[..]));

        let gemm_bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_t2d_plan_gemm_bg"),
            layout: &device.gemm_f32_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gemm_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gemm_out_buf.as_entire_binding(),
                },
            ],
        });
        let col2im_bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_t2d_plan_col2im_bg"),
            layout: &device.conv_col2im_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: col2im_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gemm_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            weight_col: Arc::clone(weight_col),
            w_buf,
            gemm_params_buf,
            col2im_params_buf,
            a_buf,
            gemm_out_buf,
            dst_buf,
            staging_buf,
            gemm_bind_group,
            col2im_bind_group,
            dispatch_wg_x: dispatch.wg_x,
            dispatch_wg_y: dispatch.wg_y,
            use_small_k: dispatch.use_small_k,
            col2im_workgroups,
            out_elems,
            a_len,
        })
    }

    /// Run the cached plan for a fresh `a_reshaped`: upload A (only), encode the
    /// GEMM + col2im passes, submit, and read back the result.
    ///
    /// The weight buffer is **not** touched here — it stays GPU-resident from
    /// `build`. `gemm_out_buf` and `dst_buf` are fully overwritten by the
    /// shaders each run, so reusing them across calls cannot leak stale data:
    /// the GEMM shader writes every `(row, col)` of `gemm_out`, and the col2im
    /// shader does an **unconditional** `dst[thread_id] = sum` for every output
    /// element (it gathers, not scatters — `shaders.rs::CONV_COL2IM_SHADER`),
    /// so no prior-call value survives. This is why the non-cached path's
    /// reliance on a fresh zero-filled `dst` is unnecessary here.
    fn execute(&self, device: &WgpuDevice, a_reshaped: &[f32]) -> Result<Vec<f32>> {
        if a_reshaped.len() != self.a_len {
            return Err(NyError::InternalError(format!(
                "conv_transpose_2d_cached: a_reshaped.len()={} != expected M*K={}",
                a_reshaped.len(),
                self.a_len
            )));
        }

        // Re-upload ONLY the dynamic A input; weight stays resident.
        device
            .queue
            .write_buffer(&self.a_buf, 0, bytemuck::cast_slice(a_reshaped));

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv_t2d_plan_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_t2d_plan_gemm_pass"),
                timestamp_writes: None,
            });
            let pipeline = if self.use_small_k {
                &device.gemm_f32_small_k_pipeline
            } else {
                &device.gemm_f32_pipeline
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.gemm_bind_group, &[]);
            pass.dispatch_workgroups(self.dispatch_wg_x, self.dispatch_wg_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_t2d_plan_col2im_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&device.conv_col2im_pipeline);
            pass.set_bind_group(0, &self.col2im_bind_group, &[]);
            pass.dispatch_workgroups(self.col2im_workgroups, 1, 1);
        }

        let out_bytes = (self.out_elems * size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.dst_buf, 0, &self.staging_buf, 0, out_bytes);

        device.queue.submit(std::iter::once(encoder.finish()));
        WgpuDevice::read_buffer(&device.device, &self.staging_buf, self.out_elems)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> ConvTranspose2dParams {
        ConvTranspose2dParams {
            num_specs: 2,
            out_channels: 3,
            in_channels: 2,
            out_h: 4,
            out_w: 4,
            in_h: 5,
            in_w: 5,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        }
    }

    /// Same weight `Arc` + same topology + same row count ⇒ same key (true hit).
    #[test]
    fn same_arc_same_key() {
        let w: Arc<[f32]> = Arc::from(vec![0.1f32; 3 * 2 * 3 * 3]);
        let a = conv_transpose_plan_key(&w, &params(), 32);
        let b = conv_transpose_plan_key(&Arc::clone(&w), &params(), 32);
        assert_eq!(a, b, "same weight Arc must hash to the same plan key");
    }

    /// Distinct `Arc` allocations with identical contents ⇒ SAME key: the
    /// production caller mints a fresh `Arc` per call, so re-minted identical
    /// weights are the cache's entire legitimate hit population.
    #[test]
    fn different_arc_same_contents_same_key() {
        let contents = vec![0.1f32; 3 * 2 * 3 * 3];
        let wa: Arc<[f32]> = Arc::from(contents.clone());
        let wb: Arc<[f32]> = Arc::from(contents);
        assert_ne!(
            Arc::as_ptr(&wa).cast::<f32>() as usize,
            Arc::as_ptr(&wb).cast::<f32>() as usize,
            "precondition: distinct allocations"
        );
        let a = conv_transpose_plan_key(&wa, &params(), 32);
        let b = conv_transpose_plan_key(&wb, &params(), 32);
        assert_eq!(
            a, b,
            "identical weight contents must key to the same plan regardless of allocation"
        );
    }

    /// Different contents at the same length/topology/rows ⇒ different key. The
    /// first `Arc` is dropped before the second is minted, so the allocator may
    /// recycle its address — the drop-then-realloc pattern the grouped-conv
    /// caller produces every iteration. Content keying is indifferent to the
    /// address; a pointer key would collide here and serve stale weights.
    #[test]
    fn recycled_allocation_different_contents_different_key() {
        let key_a = {
            let wa: Arc<[f32]> = Arc::from(vec![0.1f32; 3 * 2 * 3 * 3]);
            conv_transpose_plan_key(&wa, &params(), 32)
        };
        let wb: Arc<[f32]> = Arc::from(vec![0.2f32; 3 * 2 * 3 * 3]);
        let key_b = conv_transpose_plan_key(&wb, &params(), 32);
        assert_ne!(
            key_a, key_b,
            "different weight contents must never share a plan key"
        );
    }

    /// Bit-exact weight comparison: `NaN` bytes compare equal to themselves and
    /// `0.0` vs `-0.0` compare unequal — upload identity, not `f32` semantics.
    #[test]
    fn weight_bytes_equal_is_bitwise() {
        assert!(weight_bytes_equal(
            &[f32::NAN, 1.0, -0.0],
            &[f32::NAN, 1.0, -0.0]
        ));
        assert!(!weight_bytes_equal(&[0.0], &[-0.0]));
        assert!(!weight_bytes_equal(&[1.0, 2.0], &[1.0]));
    }

    /// Row count (GEMM M) participates in the key — fused (2*S) vs single (S)
    /// must not collide, since buffer sizes/dispatch differ.
    #[test]
    fn total_rows_affects_key() {
        let w: Arc<[f32]> = Arc::from(vec![0.1f32; 3 * 2 * 3 * 3]);
        let single = conv_transpose_plan_key(&w, &params(), 32);
        let doubled = conv_transpose_plan_key(&w, &params(), 64);
        assert_ne!(single, doubled, "total_rows must affect the key");
    }

    /// Topology changes (e.g. stride) must change the key.
    #[test]
    fn topology_affects_key() {
        let w: Arc<[f32]> = Arc::from(vec![0.1f32; 3 * 2 * 3 * 3]);
        let base = conv_transpose_plan_key(&w, &params(), 32);
        let mut p2 = params();
        p2.stride_h = 2;
        let other = conv_transpose_plan_key(&w, &p2, 32);
        assert_ne!(base, other, "stride change must affect the key");
    }
}
