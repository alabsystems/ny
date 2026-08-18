// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SOUND (verdict-legal) standalone single-op GPU IBP dispatch for the GRAPH-only
//! kinds (`docs/SOUND_GPU_IBP_PLAN.md` §3.3–§3.8): MatMul, AvgPool, Add, Transpose,
//! Scale.
//!
//! These ops are NOT part of the sequential dense chain (`GpuIbpLayer` carries only
//! Linear/Conv2d/ReLU/View), so — unlike the chain driver in `ibp_forward_sound.rs`
//! — they are exposed as standalone helpers. Each certified sound shader is
//! oracle-tested here on Vulkan (`ops/tests.rs`). Two of them ARE on a verdict
//! path: since T1.0 landed, the Add and AveragePool sound shaders are compiled
//! (via the shared `ibp_sound_pipelines()`) into the graph-DAG sound plan
//! (`ibp_graph_forward_plan_sound.rs`), which the default-on sound gate returns
//! as the authoritative bound on any adapter passing `verify_ieee_f32_model()`.
//! MatMul/Transpose/Scale have no `GpuDagIbpOp` kind and remain oracle-tested
//! only. The helper FUNCTIONS below are test-only entry points either way — the
//! DAG plan builds its own bind groups and never calls them.
//!
//! Every helper returns a CERTIFIED enclosure: the reduction/scale rounding is
//! over-bounded by the same directed widening + NORMAL-range FTZ-safe floors + §0
//! amplified-flush term as the keystone, the outward store is `center ∓ positive
//! radius`, and any wgpu error → `Err` (the shared `run_gpu_checked` wrapper). All
//! arithmetic in the WGSL body is f32 (Metal-legal); every f64 (γ_k, slack, the
//! `|s|`-scaled Scale floor) is host-side, rounded OUTWARD to an f32 uniform.

use ny_core::{ftz_safe_underflow_floor, Result};

use crate::wgpu_device::sound_consts::{combine_slack_f32, gamma_k_f32, up_f32};

use super::super::params::{
    AddIbpParams, AvgPoolIbpSoundParams, MatmulIbpSoundParams, ScaleIbpSoundParams,
    TransposeIbpParams,
};
use super::super::WgpuDevice;
use super::gpu_checked_u32;
use super::ibp_forward::create_buffer;

impl WgpuDevice {
    /// Generic single-op sound dispatch. Binding 0 = the uniform params (raw bytes),
    /// bindings `1..=inputs.len()` = the read-only input storage buffers in order,
    /// then `output_lens.len()` read-write output buffers. Uploads the inputs,
    /// dispatches `ceil(dispatch_elems/64)` workgroups, and reads back each output.
    ///
    /// The bind-group binding order + read/write flags MUST match the pipeline built
    /// in `ibp_sound_pipelines()` for the given `pipe`. Wrapped in `run_gpu_checked`
    /// so a wgpu validation error becomes `Err` (never a value from a failed op).
    fn run_ibp_sound_op(
        &self,
        pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
        label: &'static str,
        params_bytes: &[u8],
        inputs: &[&[f32]],
        output_lens: &[usize],
        dispatch_elems: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let f32_size = size_of::<f32>() as u64;
        self.run_gpu_checked(label, || {
            let params_buf = create_buffer(
                &self.device,
                "ibp_snd_op_params",
                params_bytes.len() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            self.queue.write_buffer(&params_buf, 0, params_bytes);

            // RO inputs then RW outputs, in binding order (1..).
            let mut storage: Vec<wgpu::Buffer> =
                Vec::with_capacity(inputs.len() + output_lens.len());
            for data in inputs {
                let buf = create_buffer(
                    &self.device,
                    "ibp_snd_op_in",
                    (data.len().max(1) as u64) * f32_size,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                );
                self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
                storage.push(buf);
            }
            let out_start = storage.len();
            for &len in output_lens {
                storage.push(create_buffer(
                    &self.device,
                    "ibp_snd_op_out",
                    (len.max(1) as u64) * f32_size,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                ));
            }

            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }];
            for (i, b) in storage.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: (i + 1) as u32,
                    resource: b.as_entire_binding(),
                });
            }
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipe.1,
                entries: &entries,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipe.0);
                pass.set_bind_group(0, &bind_group, &[]);
                let wg = gpu_checked_u32(dispatch_elems, "sound op dispatch")?.div_ceil(64);
                pass.dispatch_workgroups(wg.max(1), 1, 1);
            }

            let mut stagings = Vec::with_capacity(output_lens.len());
            for (oi, &len) in output_lens.iter().enumerate() {
                let staging = create_buffer(
                    &self.device,
                    "ibp_snd_op_staging",
                    (len.max(1) as u64) * f32_size,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                );
                encoder.copy_buffer_to_buffer(
                    &storage[out_start + oi],
                    0,
                    &staging,
                    0,
                    (len as u64) * f32_size,
                );
                stagings.push(staging);
            }

            self.queue.submit(std::iter::once(encoder.finish()));

            let read_specs: Vec<(&wgpu::Buffer, usize)> = stagings
                .iter()
                .zip(output_lens.iter())
                .map(|(s, &l)| (s, l))
                .collect();
            WgpuDevice::read_buffers_batched(&self.device, &read_specs)
        })
    }

    /// SOUND MatMul IBP (§3.3): interval `A @ B` over `batch` independent `m×k · k×n`
    /// products. `a_*`/`b_*` are flat row-major bound pairs. `k = contraction + 3`
    /// (the +3 covers the per-corner product rounding on top of the length-k sum).
    /// Returns `(lower, upper)` flat `batch·m·n`.
    // Test-only oracle entry (see module doc): called only from the gpu-tests harness.
    #[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
    pub(crate) fn matmul_ibp_sound(
        &self,
        a_lower: &[f32],
        a_upper: &[f32],
        b_lower: &[f32],
        b_upper: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let kk = k.checked_add(3).ok_or_else(|| {
            ny_core::NyError::InvalidSpec("matmul sound reduction length overflow".into())
        })?;
        let params = MatmulIbpSoundParams {
            batch_size: gpu_checked_u32(batch, "matmul sound batch")?,
            m: gpu_checked_u32(m, "matmul sound m")?,
            k: gpu_checked_u32(k, "matmul sound k")?,
            n: gpu_checked_u32(n, "matmul sound n")?,
            gamma_k: gamma_k_f32(kk)?,
            slack: combine_slack_f32(kk)?,
            additive: ftz_safe_underflow_floor(gpu_checked_u32(kk, "matmul sound k+3")?),
            _pad: 0,
        };
        let total = batch
            .checked_mul(m)
            .and_then(|v| v.checked_mul(n))
            .ok_or_else(|| ny_core::NyError::InvalidSpec("matmul sound output overflow".into()))?;
        let pipe = &self.ibp_sound_pipelines().matmul;
        let mut out = self.run_ibp_sound_op(
            pipe,
            "matmul_ibp_sound",
            bytemuck::bytes_of(&params),
            &[a_lower, a_upper, b_lower, b_upper],
            &[total, total],
            total,
        )?;
        let hi = out.pop().expect("2 outputs");
        let lo = out.pop().expect("2 outputs");
        Ok((lo, hi))
    }

    /// SOUND element-wise Add IBP (§3.5): `[a] + [b]`. All four slices are the same
    /// length `n`. Coefficient 1, single RN add per endpoint ⇒ the elementwise widen
    /// suffices. Returns `(lower, upper)` length `n`.
    // Test-only oracle entry (see module doc): called only from the gpu-tests harness.
    #[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
    pub(crate) fn add_ibp_sound(
        &self,
        a_lower: &[f32],
        a_upper: &[f32],
        b_lower: &[f32],
        b_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = a_lower.len();
        let params = AddIbpParams {
            num_elements: gpu_checked_u32(n, "add sound n")?,
            _padding: [0; 3],
        };
        let pipe = &self.ibp_sound_pipelines().add;
        let mut out = self.run_ibp_sound_op(
            pipe,
            "add_ibp_sound",
            bytemuck::bytes_of(&params),
            &[a_lower, a_upper, b_lower, b_upper],
            &[n, n],
            n,
        )?;
        let hi = out.pop().expect("2 outputs");
        let lo = out.pop().expect("2 outputs");
        Ok((lo, hi))
    }

    /// SOUND Transpose IBP (§3.6): swap the last two dims of `[batch, rows, cols]` →
    /// `[batch, cols, rows]`. The permutation is EXACT, so a 1-ULP widen is applied
    /// for CPU parity (S2). Returns `(lower, upper)` flat `batch·rows·cols`.
    // Test-only oracle entry (see module doc): called only from the gpu-tests harness.
    #[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
    pub(crate) fn transpose_ibp_sound(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let params = TransposeIbpParams {
            batch_size: gpu_checked_u32(batch, "transpose sound batch")?,
            rows: gpu_checked_u32(rows, "transpose sound rows")?,
            cols: gpu_checked_u32(cols, "transpose sound cols")?,
            _padding: 0,
        };
        let total = batch
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(cols))
            .ok_or_else(|| {
                ny_core::NyError::InvalidSpec("transpose sound output overflow".into())
            })?;
        let pipe = &self.ibp_sound_pipelines().transpose;
        let mut out = self.run_ibp_sound_op(
            pipe,
            "transpose_ibp_sound",
            bytemuck::bytes_of(&params),
            &[input_lower, input_upper],
            &[total, total],
            total,
        )?;
        let hi = out.pop().expect("2 outputs");
        let lo = out.pop().expect("2 outputs");
        Ok((lo, hi))
    }

    /// SOUND Scale IBP (§3.8): element-wise `x · scale`. Uses the HOST-computed
    /// `|s|`-amplified floor `scale_floor ≥ |s|·FLT_MIN` (the fixed `ADDITIVE1` is
    /// UNSOUND for `|s|>16`). Returns `(lower, upper)` the same length as the input.
    // Test-only oracle entry (see module doc): called only from the gpu-tests harness.
    #[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
    pub(crate) fn scale_ibp_sound(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        scale: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = input_lower.len();
        // floor = ftz_safe_underflow_floor(1) = 2^-122 (NORMAL). scale_floor =
        // up(|s|·floor + floor) ≥ |s|·FLT_MIN AND ≥ floor.
        let floor = ftz_safe_underflow_floor(1);
        let scale_floor = up_f32(f64::from(scale.abs()) * f64::from(floor) + f64::from(floor));
        let params = ScaleIbpSoundParams {
            total_elements: gpu_checked_u32(n, "scale sound n")?,
            scale,
            scale_floor,
            zero_ulp_floor: floor,
        };
        let pipe = &self.ibp_sound_pipelines().scale;
        let mut out = self.run_ibp_sound_op(
            pipe,
            "scale_ibp_sound",
            bytemuck::bytes_of(&params),
            &[input_lower, input_upper],
            &[n, n],
            n,
        )?;
        let hi = out.pop().expect("2 outputs");
        let lo = out.pop().expect("2 outputs");
        Ok((lo, hi))
    }

    /// SOUND AvgPool IBP (§3.4): windowed average pool over `[channels, input_h,
    /// input_w]` (batch handled by the caller flattening channels). Coefficient
    /// `1/D ≤ 1` ⇒ no §0 amplifier. `k = kernel_h·kernel_w + 3`. Returns
    /// `(lower, upper)` flat `channels·output_h·output_w`.
    // Test-only oracle entry (see module doc): called only from the gpu-tests harness.
    #[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn avgpool_ibp_sound(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        channels: usize,
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        count_include_pad: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let num_elements = channels
            .checked_mul(output_h)
            .and_then(|v| v.checked_mul(output_w))
            .ok_or_else(|| ny_core::NyError::InvalidSpec("avgpool sound output overflow".into()))?;
        let kk = kernel_h
            .checked_mul(kernel_w)
            .and_then(|v| v.checked_add(3))
            .ok_or_else(|| ny_core::NyError::InvalidSpec("avgpool sound k overflow".into()))?;
        let params = AvgPoolIbpSoundParams {
            num_elements: gpu_checked_u32(num_elements, "avgpool sound elems")?,
            channels: gpu_checked_u32(channels, "avgpool sound channels")?,
            input_h: gpu_checked_u32(input_h, "avgpool sound in_h")?,
            input_w: gpu_checked_u32(input_w, "avgpool sound in_w")?,
            output_h: gpu_checked_u32(output_h, "avgpool sound out_h")?,
            output_w: gpu_checked_u32(output_w, "avgpool sound out_w")?,
            kernel_h: gpu_checked_u32(kernel_h, "avgpool sound k_h")?,
            kernel_w: gpu_checked_u32(kernel_w, "avgpool sound k_w")?,
            stride_h: gpu_checked_u32(stride_h, "avgpool sound s_h")?,
            stride_w: gpu_checked_u32(stride_w, "avgpool sound s_w")?,
            pad_h: gpu_checked_u32(pad_h, "avgpool sound p_h")?,
            pad_w: gpu_checked_u32(pad_w, "avgpool sound p_w")?,
            count_include_pad: u32::from(count_include_pad),
            gamma_k: gamma_k_f32(kk)?,
            slack: combine_slack_f32(kk)?,
            additive: ftz_safe_underflow_floor(gpu_checked_u32(kk, "avgpool sound k")?),
        };
        let pipe = &self.ibp_sound_pipelines().avgpool;
        let mut out = self.run_ibp_sound_op(
            pipe,
            "avgpool_ibp_sound",
            bytemuck::bytes_of(&params),
            &[input_lower, input_upper],
            &[num_elements, num_elements],
            num_elements,
        )?;
        let hi = out.pop().expect("2 outputs");
        let lo = out.pop().expect("2 outputs");
        Ok((lo, hi))
    }
}
