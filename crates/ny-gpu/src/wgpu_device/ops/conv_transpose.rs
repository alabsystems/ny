// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fused GPU conv_transpose_2d: GEMM + col2im in a single command submission.
//!
//! Chains the existing GEMM shader and col2im shader with no host roundtrip
//! between them — the GEMM output stays GPU-resident as the col2im input.
//!
//! Part of #3813: eliminates the CPU col2im bottleneck that causes RSPLITTER
//! biasfield timeouts.
//!
//! Reference: designs/2026-03-15-issue-3813-fused-gpu-conv2d-backward.md

use ny_core::{ConvTranspose2dParams, NyError, Result};

use super::super::WgpuDevice;
use super::gpu_checked_u32;
use crate::wgpu_device::params::{ConvCol2imParams, GemmParams};

use super::gemm::{select_gemm_dispatch, MAX_BINDING_ELEMS, WGPU_MAX_BINDING_BYTES};

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

        // Check buffer limits: if any single buffer exceeds wgpu binding limit,
        // fall back to error (caller uses CPU path).
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
    use super::checked_col2im_workgroups;

    /// The GPU fused `conv_transpose_2d` (GEMM + col2im) must be numerically
    /// equivalent to the `NaiveCpuGemmEngine` reference for the small-conv
    /// CROWN-backward shape. This pins down that the GPU conv op itself is NOT
    /// the source of any GPU-vs-CPU CROWN bound gap (that gap was an alpha
    /// optimization issue in the graph backward suffix, not this op). Skips
    /// cleanly when no GPU is available.
    #[test]
    fn wgpu_conv_transpose_matches_naive_cpu_reference() {
        use ny_core::{ConvTranspose2dParams, GemmEngine, NaiveCpuGemmEngine};
        let device = match crate::WgpuDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no gpu; skipping wgpu_conv_transpose_matches_naive_cpu_reference");
                return;
            }
        };
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
