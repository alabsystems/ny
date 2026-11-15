// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fused GPU conv_transpose_2d: GEMM + col2im in a single command submission.
//!
//! Chains the existing GEMM shader and col2im shader with no host roundtrip
//! between them — the GEMM output stays GPU-resident as the col2im input.
//!
//! Eliminates the CPU col2im bottleneck that causes RSPLITTER biasfield
//! timeouts.

use ny_core::{ConvTranspose2dParams, NyError, Result};

use super::super::WgpuDevice;
use super::gpu_checked_u32;
use crate::wgpu_device::params::{ConvCol2imParams, GemmParams};

use super::gemm::select_gemm_dispatch;

/// Largest element count a fused conv_transpose binding may hold: the GEMM and
/// col2im shaders index every binding with WGSL `u32` arithmetic, and each
/// index they compute is below its binding's element count.
///
/// wgpu 29 reports `max_storage_buffer_binding_size` as a `u64`, but every
/// backend it ships clamps it to at most `u32::MAX` bytes (Metal grants
/// 4 GiB - 4 B on Apple M-series), so the live element budget stays below this
/// today. The gate enforces it rather than rely on that, so a future adapter
/// granting a larger binding gets a typed refusal, never a wrapped index.
const CONV_TRANSPOSE_U32_INDEXABLE_ELEMS: usize = u32::MAX as usize;

/// #hard-caps: THE storage-binding fit gate for the fused conv_transpose_2d
/// dispatch (GEMM `(m, k) x (k, n)` followed by col2im into `out_elems`).
///
/// Shared by the per-call path ([`WgpuDevice::conv_transpose_2d`]) and the
/// cached, weight-resident pair plan (`conv_transpose_plan.rs`), so the two can
/// never admit different shapes. They did: the per-call gate moved to the live
/// device limit in d153bf0a4 while the plan kept wgpu's DEFAULT 128 MiB, so on
/// an adapter granting more (an M5 Max grants 4 GiB - 4 B) the plan refused
/// every fused pair whose bindings were past 112 MB that two per-call
/// dispatches of the same work ran.
///
/// Checks the four storage bindings: the weight column `k x n`, the input
/// `m x k`, the GEMM output `m x n`, and the col2im destination `out_elems`.
/// Each must be at most `max_elems` (and u32-indexable). Products are checked,
/// so an overflowing shape is a refusal, not a wrapped product that passes.
/// The refusal names the LARGEST binding's bytes and `budget_bytes`, capped at
/// the u32-indexable bytes when that is the tighter bound.
///
/// Pure, so the edges are pinned without a GPU.
fn check_conv_transpose_bindings_fit(
    gemm_m: usize,
    gemm_k: usize,
    gemm_n: usize,
    out_elems: usize,
    max_elems: usize,
    budget_bytes: usize,
) -> Result<()> {
    let cap = max_elems.min(CONV_TRANSPOSE_U32_INDEXABLE_ELEMS);
    let bindings = [
        gemm_k.checked_mul(gemm_n),
        gemm_m.checked_mul(gemm_k),
        gemm_m.checked_mul(gemm_n),
        Some(out_elems),
    ];
    if bindings
        .iter()
        .all(|elems| elems.is_some_and(|elems| elems <= cap))
    {
        return Ok(());
    }
    let largest = bindings
        .iter()
        .map(|elems| elems.unwrap_or(usize::MAX))
        .max()
        .unwrap_or(usize::MAX);
    let budget_bytes = if cap < max_elems {
        budget_bytes.min(cap.saturating_mul(size_of::<f32>()))
    } else {
        budget_bytes
    };
    Err(NyError::GpuMemoryExceeded {
        required_bytes: largest.saturating_mul(size_of::<f32>()),
        budget_bytes,
    })
}

impl WgpuDevice {
    /// [`check_conv_transpose_bindings_fit`] against this device's LIVE binding
    /// limit ([`Self::max_binding_elems_live`], floored at wgpu's default, so it
    /// can only raise the ceiling). Both fused conv_transpose paths call this,
    /// so the budget's source cannot drift between them either.
    pub(super) fn check_conv_transpose_bindings_live(
        &self,
        gemm_m: usize,
        gemm_k: usize,
        gemm_n: usize,
        out_elems: usize,
    ) -> Result<()> {
        check_conv_transpose_bindings_fit(
            gemm_m,
            gemm_k,
            gemm_n,
            out_elems,
            self.max_binding_elems_live(),
            self.max_binding_bytes_live(),
        )
    }
}

fn checked_col2im_workgroups(out_elems: usize) -> Result<u32> {
    let workgroups = out_elems.div_ceil(256);
    let workgroups_u32 = gpu_checked_u32(workgroups, "conv_t2d col2im_workgroups")?;
    if workgroups_u32 > 65_535 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "conv_transpose_2d col2im dispatch exceeds 65535: out_elems={out_elems}, workgroups={workgroups}"
        )));
    }
    Ok(workgroups_u32)
}

impl WgpuDevice {
    /// Fused conv_transpose_2d: GEMM + col2im on GPU in a single submission.
    ///
    /// Chains the existing GEMM shader and col2im shader with no host roundtrip
    /// between them — the GEMM output stays GPU-resident as the col2im input.
    pub(crate) fn conv_transpose_2d(
        &self,
        a_reshaped: &[f32],
        weight_col: &[f32],
        params: &ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        // Wrap GPU work in an error scope: a wgpu error returns Err (caller
        // falls back to CPU) instead of aborting via wgpu's panicking handler.
        self.run_gpu_checked("conv_transpose_2d", || {
            self.conv_transpose_2d_inner(a_reshaped, weight_col, params)
        })
    }

    fn conv_transpose_2d_inner(
        &self,
        a_reshaped: &[f32],
        weight_col: &[f32],
        params: &ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        let s = params.num_specs;
        let oc = params.out_channels;
        let ic = params.in_channels;
        let (oh, ow) = (params.out_h, params.out_w);
        let (ih, iw) = (params.in_h, params.in_w);
        let (kh, kw) = (params.kernel_h, params.kernel_w);
        let spatial = oh * ow;
        let total_rows = s * spatial;
        let kernel_cols = ic * kh * kw;
        let flat_input_dim = ic * ih * iw;
        let out_elems = s * flat_input_dim;

        if total_rows == 0 || oc == 0 || kernel_cols == 0 {
            return Ok(vec![0.0f32; out_elems]);
        }

        // --- Phase 1: GEMM (S*OH*OW, OC) × (OC, IC*KH*KW) → (S*OH*OW, IC*KH*KW) ---
        let gemm_m = total_rows;
        let gemm_k = oc;
        let gemm_n = kernel_cols;
        let gemm_out_elems = gemm_m * gemm_n;

        let m_u32 = gpu_checked_u32(gemm_m, "conv_t2d gemm_m")?;
        let k_u32 = gpu_checked_u32(gemm_k, "conv_t2d gemm_k")?;
        let n_u32 = gpu_checked_u32(gemm_n, "conv_t2d gemm_n")?;

        let dispatch = select_gemm_dispatch(m_u32, k_u32, n_u32);
        if dispatch.wg_y > 65535 || dispatch.wg_x > 65535 {
            return Err(NyError::InternalError(format!(
                "conv_transpose_2d GEMM dispatch exceeds 65535: M={gemm_m}, N={gemm_n}",
            )));
        }

        // Check buffer limits: if any single buffer exceeds the binding limit,
        // fall back to error (caller uses CPU path).
        // #hard-caps: compare against the LIVE device limit, not wgpu's 128 MiB
        // default. `WgpuDevice::new` requests the adapter's real limits, which on
        // an Apple M4 Pro is 4095 MiB — 32x the constant this used to use. The
        // old gate refused shapes the device could run, on the conv backward path
        // that all four scoreboard deficits share. The cached pair plan applies
        // this same gate (`check_conv_transpose_bindings_live`).
        self.check_conv_transpose_bindings_live(gemm_m, gemm_k, gemm_n, out_elems)?;

        let gemm_params = GemmParams {
            m: m_u32,
            k: k_u32,
            n: n_u32,
            _padding: 0,
        };

        // --- Phase 2: col2im params ---
        let col2im_params = ConvCol2imParams {
            num_specs: gpu_checked_u32(s, "conv_t2d num_specs")?,
            flat_input_dim: gpu_checked_u32(flat_input_dim, "conv_t2d flat_input_dim")?,
            out_h: gpu_checked_u32(oh, "conv_t2d out_h")?,
            out_w: gpu_checked_u32(ow, "conv_t2d out_w")?,
            in_channels: gpu_checked_u32(ic, "conv_t2d in_channels")?,
            in_h: gpu_checked_u32(ih, "conv_t2d in_h")?,
            in_w: gpu_checked_u32(iw, "conv_t2d in_w")?,
            kernel_h: gpu_checked_u32(kh, "conv_t2d kernel_h")?,
            kernel_w: gpu_checked_u32(kw, "conv_t2d kernel_w")?,
            stride_h: gpu_checked_u32(params.stride_h, "conv_t2d stride_h")?,
            stride_w: gpu_checked_u32(params.stride_w, "conv_t2d stride_w")?,
            pad_h: gpu_checked_u32(params.pad_h, "conv_t2d pad_h")?,
            pad_w: gpu_checked_u32(params.pad_w, "conv_t2d pad_w")?,
            kernel_cols: gpu_checked_u32(kernel_cols, "conv_t2d kernel_cols")?,
            _padding2: [0; 2],
        };

        // --- Phase 3: Allocate buffers ---
        let gemm_params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_gemm_params"),
            size: size_of::<GemmParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let a_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_a"),
            size: (a_reshaped.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let w_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_w"),
            size: (weight_col.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // GEMM output = col2im input. Needs STORAGE for both shader passes.
        let gemm_out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_gemm_out"),
            size: (gemm_out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let col2im_params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_col2im_params"),
            size: size_of::<ConvCol2imParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dst_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_dst"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_t2d_staging"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Phase 4: Upload data ---
        self.queue
            .write_buffer(&gemm_params_buf, 0, bytemuck::cast_slice(&[gemm_params]));
        self.queue
            .write_buffer(&a_buf, 0, bytemuck::cast_slice(a_reshaped));
        self.queue
            .write_buffer(&w_buf, 0, bytemuck::cast_slice(weight_col));
        self.queue.write_buffer(
            &col2im_params_buf,
            0,
            bytemuck::cast_slice(&[col2im_params]),
        );

        // --- Phase 5: Encode GEMM pass + col2im pass in one command buffer ---
        let gemm_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_t2d_gemm_bg"),
            layout: &self.gemm_f32_bind_group_layout,
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

        let col2im_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_t2d_col2im_bg"),
            layout: &self.conv_col2im_bind_group_layout,
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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv_t2d_encoder"),
            });

        // Pass 1: GEMM
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_t2d_gemm_pass"),
                timestamp_writes: None,
            });
            let pipeline = if dispatch.use_small_k {
                &self.gemm_f32_small_k_pipeline
            } else {
                &self.gemm_f32_pipeline
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &gemm_bind_group, &[]);
            pass.dispatch_workgroups(dispatch.wg_x, dispatch.wg_y, 1);
        }

        // Pass 2: col2im — dst buffer is zero-initialized by wgpu (storage buffers
        // are zero-filled on creation when not mapped_at_creation).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_t2d_col2im_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.conv_col2im_pipeline);
            pass.set_bind_group(0, &col2im_bind_group, &[]);
            let col2im_workgroups = checked_col2im_workgroups(out_elems)?;
            pass.dispatch_workgroups(col2im_workgroups, 1, 1);
        }

        // Copy result to staging for readback
        let out_bytes = (out_elems * size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&dst_buf, 0, &staging_buf, 0, out_bytes);

        // --- Phase 6: Submit and readback ---
        self.queue.submit(std::iter::once(encoder.finish()));
        Self::read_buffer(&self.device, &staging_buf, out_elems)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_conv_transpose_bindings_fit, checked_col2im_workgroups,
        CONV_TRANSPOSE_U32_INDEXABLE_ELEMS,
    };
    use crate::wgpu_device::ops::gemm::{MAX_BINDING_ELEMS, WGPU_MAX_BINDING_BYTES};
    use ny_core::{NyError, Result};

    /// Largest live element budget wgpu 29 can produce: a `u32::MAX`-byte
    /// binding after the pool's 1.2x growth (an M5 Max grants 4 GiB - 4 B,
    /// one element less).
    const LARGEST_LIVE_ELEMS: usize = (u32::MAX as usize) * 5 / 6 / size_of::<f32>();
    const LARGEST_LIVE_BYTES: usize = u32::MAX as usize;

    /// `(required_bytes, budget_bytes)` of a `GpuMemoryExceeded` refusal.
    fn refused(result: Result<()>, at: &str) -> (usize, usize) {
        match result {
            Err(NyError::GpuMemoryExceeded {
                required_bytes,
                budget_bytes,
            }) => (required_bytes, budget_bytes),
            Err(other) => panic!("{at}: expected GpuMemoryExceeded, got {other}"),
            Ok(()) => panic!("{at}: expected a refusal, got Ok"),
        }
    }

    /// #hard-caps: every one of the four bindings is admitted AT the budget and
    /// refused one element past it, on the static budget and the largest live
    /// one; a refusal names the largest binding and the caller's budget.
    #[test]
    fn conv_transpose_bindings_fit_edges_per_binding() {
        for (max, bytes) in [
            (MAX_BINDING_ELEMS, WGPU_MAX_BINDING_BYTES),
            (LARGEST_LIVE_ELEMS, LARGEST_LIVE_BYTES),
        ] {
            // 2 * half is max + 1 or max + 2, and half itself fits.
            let half = max / 2 + 1;
            // (m, k, n, out_elems) exactly at the budget: admitted.
            for (m, k, n, out) in [
                (1, max, 1, 1),
                (max, 1, 1, 1),
                (1, 1, max, 1),
                (1, 1, 1, max),
            ] {
                check_conv_transpose_bindings_fit(m, k, n, out, max, bytes).unwrap_or_else(|e| {
                    panic!("({m},{k},{n},{out}) at budget {max} must fit: {e}")
                });
            }
            // One past, and each binding as the SOLE offender:
            // (m, k, n, out_elems, largest binding).
            for (m, k, n, out, largest) in [
                (1, max + 1, 1, 1, max + 1), // weight and input
                (max + 1, 1, 1, 1, max + 1), // input and GEMM output
                (1, 1, max + 1, 1, max + 1), // weight and GEMM output
                (1, 1, 1, max + 1, max + 1), // col2im destination
                (1, 2, half, 1, 2 * half),   // weight k*n only
                (half, 2, 1, 1, 2 * half),   // input m*k only
                (2, 1, half, 1, 2 * half),   // GEMM output m*n only
            ] {
                let at = format!("({m},{k},{n},{out}) budget {max}");
                let (required, budget) = refused(
                    check_conv_transpose_bindings_fit(m, k, n, out, max, bytes),
                    &at,
                );
                assert_eq!(required, largest * 4, "{at}: names the largest binding");
                assert_eq!(budget, bytes, "{at}: names the caller's budget");
            }
        }
    }

    /// The defect in pure form: the gpu-tests pin's fused pair (GEMM output
    /// 98304 x 400 = 157.3 MB) is refused by wgpu's DEFAULT binding budget and
    /// admitted by a live one. The cached plan used to apply the former while
    /// the per-call path applied the latter.
    #[test]
    fn conv_transpose_bindings_fit_static_wall_vs_live_budget() {
        let (m, k, n, out) = (98_304, 2, 400, 96 * 16 * 36 * 36);
        let (required, budget) = refused(
            check_conv_transpose_bindings_fit(
                m,
                k,
                n,
                out,
                MAX_BINDING_ELEMS,
                WGPU_MAX_BINDING_BYTES,
            ),
            "static budget",
        );
        assert_eq!((required, budget), (m * n * 4, WGPU_MAX_BINDING_BYTES));
        check_conv_transpose_bindings_fit(m, k, n, out, LARGEST_LIVE_ELEMS, LARGEST_LIVE_BYTES)
            .expect("the live budget admits the fused pair");
    }

    /// Overflowing products refuse (they never wrap into a passing value) and
    /// the reported bytes saturate instead of panicking.
    ///
    /// The first three shapes wrap to values that would still be refused, so
    /// they alone cannot tell a checked product from a wrapping one. The last
    /// two wrap to values every budget admits: `half_word` cubed wraps each
    /// product to exactly 0, and `3 * (usize::MAX / 3 + 1)` is
    /// `usize::MAX + 3`, which wraps to 2. Both callers bound each dimension
    /// with `gpu_checked_u32` before this gate, so production never reaches
    /// such a product; the gate refuses it anyway.
    #[test]
    fn conv_transpose_bindings_fit_refuses_overflowing_products() {
        let half_word = 1usize << (usize::BITS / 2);
        let wraps_to_two = usize::MAX / 3 + 1;
        assert_eq!(half_word.wrapping_mul(half_word), 0);
        assert_eq!(3usize.wrapping_mul(wraps_to_two), 2);
        for (m, k, n) in [
            (usize::MAX, 2, 1),
            (1, usize::MAX, 2),
            (2, 1, usize::MAX),
            (half_word, half_word, half_word),
            (3, wraps_to_two, 3),
        ] {
            let at = format!("({m},{k},{n}) overflow");
            let (required, budget) = refused(
                check_conv_transpose_bindings_fit(m, k, n, 1, usize::MAX, usize::MAX),
                &at,
            );
            assert_eq!(required, usize::MAX, "{at}");
            assert_eq!(budget, CONV_TRANSPOSE_U32_INDEXABLE_ELEMS * 4, "{at}");
        }
    }

    /// The shaders index in `u32`: a budget past the u32-indexable element
    /// count (no wgpu 29 backend grants one) still refuses a binding past it,
    /// naming the u32-indexable bytes as the budget.
    #[test]
    fn conv_transpose_bindings_fit_caps_at_u32_indexable() {
        let cap = CONV_TRANSPOSE_U32_INDEXABLE_ELEMS;
        assert!(
            LARGEST_LIVE_ELEMS < cap,
            "wgpu 29's largest budget is below the cap"
        );
        check_conv_transpose_bindings_fit(1, 1, 1, cap, usize::MAX, usize::MAX)
            .expect("a u32-indexable binding fits an unbounded budget");
        let (required, budget) = refused(
            check_conv_transpose_bindings_fit(1, 1, 1, cap + 1, usize::MAX, usize::MAX),
            "past u32",
        );
        assert_eq!((required, budget), ((cap + 1) * 4, cap * 4));
    }

    /// #hard-caps: what the adapter ACTUALLY grants, vs the hard-coded
    /// `WGPU_MAX_BINDING_BYTES` the buffer-fit gate compares against.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn adapter_binding_limits_cover_the_hardcoded_cap() {
        let _g = crate::wgpu_device::test_support::gpu_test_serial_guard();
        let device = crate::wgpu_device::test_support::require_device();
        let l = device.device.limits();
        let routing_cap = u64::try_from(WGPU_MAX_BINDING_BYTES)
            .expect("WGPU routing cap must fit the API's u64 byte unit");
        assert!(
            l.max_storage_buffer_binding_size >= routing_cap,
            "constructor granted {} binding bytes, below the routing cap {}",
            l.max_storage_buffer_binding_size,
            routing_cap,
        );
    }

    /// Deterministic small-shape correctness fixture. Unlike the measurement,
    /// compiled by the explicit `gpu-tests` hardware suite.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn wgpu_conv_transpose_matches_naive_cpu_reference() {
        use ny_core::{ConvTranspose2dParams, GemmEngine, NaiveCpuGemmEngine};
        let _g = crate::wgpu_device::test_support::gpu_test_serial_guard();
        let device = crate::wgpu_device::test_support::require_device();
        // Mirror the small conv graph case: conv kernel [1,1,2,2], stride 1, pad 0,
        // input 4x4 -> conv out 3x3. CROWN backward conv_transpose maps grad (3,3)
        // back to input (4,4). num_specs = number of objectives.
        let s = 9usize;
        let oc = 1usize;
        let ic = 1usize;
        let (grad_h, grad_w) = (3usize, 3usize);
        let (in_h, in_w) = (4usize, 4usize);
        let (kh, kw) = (2usize, 2usize);
        let params = ConvTranspose2dParams {
            num_specs: s,
            out_channels: oc,
            in_channels: ic,
            out_h: grad_h,
            out_w: grad_w,
            in_h,
            in_w,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let kernel_cols = ic * kh * kw;
        let total_rows = s * grad_h * grad_w;
        // deterministic-ish A and W
        let a: Vec<f32> = (0..total_rows * oc)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.31)
            .collect();
        let w: Vec<f32> = (0..oc * kernel_cols)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.5)
            .collect();

        let gpu = device.conv_transpose_2d(&a, &w, &params).unwrap();
        let cpu = NaiveCpuGemmEngine
            .conv_transpose_2d(&a, &w, &params)
            .unwrap();
        let mut maxdiff = 0.0f32;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = (g - c).abs();
            if d > maxdiff {
                maxdiff = d;
            }
            if d > 1e-4 {
                eprintln!("idx {i}: gpu={g} cpu={c} diff={d}");
            }
        }
        eprintln!("conv_transpose maxdiff = {maxdiff}");
        assert!(
            maxdiff < 1e-3,
            "conv_transpose maxdiff too large: {maxdiff}"
        );
    }

    #[test]
    fn test_checked_col2im_workgroups_accepts_wgpu_boundary_4404() {
        assert_eq!(checked_col2im_workgroups(65_535 * 256).unwrap(), 65_535);
    }

    #[test]
    fn test_checked_col2im_workgroups_rejects_boundary_plus_one_4404() {
        let err = checked_col2im_workgroups(65_535 * 256 + 1).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("conv_transpose_2d col2im dispatch exceeds 65535"),
            "expected explicit dispatch-limit error, got: {message}"
        );
    }
}
