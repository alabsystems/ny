// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound bind-group builders for graph-DAG GPU-resident IBP (T1.0).
//!
//! Extracted from `ibp_graph_forward_plan_sound.rs` for file-size compliance. Each
//! builder assembles the bind group for ONE sound op against the matching sound
//! pipeline layout, in the exact binding order the sound shader declares (see
//! `wgpu_device/shaders.rs`). The weight `wp`/`wn` split is NaN-propagating so a NaN
//! weight makes both non-finite ⇒ the shader's `is_non_finite` product guard fires ⇒
//! `[-FALLBACK, +FALLBACK]` (a sound superset), matching the sequential sound path.

use ny_core::{
    ftz_safe_underflow_floor, nan_propagating_max_zero, nan_propagating_min_zero, Result,
};

use crate::wgpu_device::params::{
    AddIbpParams, AvgPoolIbpSoundParams, Conv2dIbpSoundParams, LinearIbpSoundParams, ReluIbpParams,
};
use crate::wgpu_device::sound_consts::{combine_slack_f32, gamma_k_f32};
use crate::wgpu_device::WgpuDevice;

use super::gpu_checked_u32;
use super::ibp_forward::create_buffer;

impl WgpuDevice {
    /// Sound Linear bind group (§3.1): bindings 0=params, 1=in_l, 2=in_u, 3=wp,
    /// 4=wn, 5=bias, 6=out_l, 7=out_u — 8 buffers (Metal's limit). `k = in+3`,
    /// `n_ulps = 2·(in+2)`, the `3γS + 4Nu|·| + flush` radius covers GPU-vs-CPU
    /// center drift + the CPU double N-ULP widen (S2).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_sound_dag_linear_bg(
        &self,
        layout: &wgpu::BindGroupLayout,
        src_l: &wgpu::Buffer,
        src_u: &wgpu::Buffer,
        dst_l: &wgpu::Buffer,
        dst_u: &wgpu::Buffer,
        weight: &[f32],
        bias: Option<&[f32]>,
        out_features: usize,
        in_features: usize,
        batch_size: usize,
    ) -> Result<wgpu::BindGroup> {
        let f32_size = size_of::<f32>() as u64;
        let weight_pos: Vec<f32> = weight
            .iter()
            .map(|&w| nan_propagating_max_zero(w))
            .collect();
        let weight_neg: Vec<f32> = weight
            .iter()
            .map(|&w| nan_propagating_min_zero(w))
            .collect();
        let bias_data: Vec<f32> = bias.map_or_else(|| vec![0.0; out_features], |b| b.to_vec());

        let k = in_features + 3;
        let k_u32 = gpu_checked_u32(k, "sound dag lin k")?;
        let params = LinearIbpSoundParams {
            batch_size: gpu_checked_u32(batch_size, "sound dag lin batch")?,
            in_features: gpu_checked_u32(in_features, "sound dag lin in")?,
            out_features: gpu_checked_u32(out_features, "sound dag lin out")?,
            n_ulps: gpu_checked_u32(
                2usize.saturating_mul(in_features.saturating_add(2)),
                "sound dag lin n_ulps",
            )?,
            gamma_k: gamma_k_f32(k),
            slack: combine_slack_f32(k),
            additive: ftz_safe_underflow_floor(k_u32),
            _pad: 0,
        };
        let params_buf = self.upload_uniform("snd_dag_lin_p", bytemuck::bytes_of(&params));

        let ws = weight.len() as u64 * f32_size;
        let wp_buf = self.upload_storage("snd_dag_lin_wp", ws, bytemuck::cast_slice(&weight_pos));
        let wn_buf = self.upload_storage("snd_dag_lin_wn", ws, bytemuck::cast_slice(&weight_neg));
        let bias_buf = self.upload_storage(
            "snd_dag_lin_b",
            (out_features as u64) * f32_size,
            bytemuck::cast_slice(&bias_data),
        );

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snd_dag_lin_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wn_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dst_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dst_u.as_entire_binding(),
                },
            ],
        }))
    }

    /// Sound Conv2d bind group (§3.2): same 8-buffer shape as sound Linear.
    /// `k = in_channels·kh·kw + 3` over the full window (padding taps over-counted ⇒
    /// sound but looser at the border). Caller has already rejected groups!=1.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_sound_dag_conv2d_bg(
        &self,
        layout: &wgpu::BindGroupLayout,
        src_l: &wgpu::Buffer,
        src_u: &wgpu::Buffer,
        dst_l: &wgpu::Buffer,
        dst_u: &wgpu::Buffer,
        weight: &[f32],
        bias: Option<&[f32]>,
        out_channels: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        input_h: usize,
        input_w: usize,
        batch_size: usize,
        out_h: usize,
        out_w: usize,
    ) -> Result<wgpu::BindGroup> {
        let f32_size = size_of::<f32>() as u64;
        let weight_pos: Vec<f32> = weight
            .iter()
            .map(|&w| nan_propagating_max_zero(w))
            .collect();
        let weight_neg: Vec<f32> = weight
            .iter()
            .map(|&w| nan_propagating_min_zero(w))
            .collect();
        let bias_data: Vec<f32> = bias.map_or_else(|| vec![0.0; out_channels], |b| b.to_vec());

        let macs = in_channels
            .checked_mul(kernel_h)
            .and_then(|v| v.checked_mul(kernel_w))
            .ok_or_else(|| ny_core::NyError::InvalidSpec("sound dag conv MAC overflow".into()))?;
        let k = macs + 3;
        let k_u32 = gpu_checked_u32(k, "sound dag conv k")?;
        let params = Conv2dIbpSoundParams {
            batch_size: gpu_checked_u32(batch_size, "sound dag conv batch")?,
            in_channels: gpu_checked_u32(in_channels, "sound dag conv in_c")?,
            out_channels: gpu_checked_u32(out_channels, "sound dag conv out_c")?,
            input_h: gpu_checked_u32(input_h, "sound dag conv in_h")?,
            input_w: gpu_checked_u32(input_w, "sound dag conv in_w")?,
            out_h: gpu_checked_u32(out_h, "sound dag conv out_h")?,
            out_w: gpu_checked_u32(out_w, "sound dag conv out_w")?,
            kernel_h: gpu_checked_u32(kernel_h, "sound dag conv k_h")?,
            kernel_w: gpu_checked_u32(kernel_w, "sound dag conv k_w")?,
            stride_h: gpu_checked_u32(stride_h, "sound dag conv s_h")?,
            stride_w: gpu_checked_u32(stride_w, "sound dag conv s_w")?,
            pad_h: gpu_checked_u32(pad_h, "sound dag conv p_h")?,
            pad_w: gpu_checked_u32(pad_w, "sound dag conv p_w")?,
            groups: 1,
            n_ulps: gpu_checked_u32(
                2usize.saturating_mul(macs.saturating_add(2)),
                "sound dag conv n_ulps",
            )?,
            gamma_k: gamma_k_f32(k),
            slack: combine_slack_f32(k),
            additive: ftz_safe_underflow_floor(k_u32),
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = self.upload_uniform("snd_dag_conv_p", bytemuck::bytes_of(&params));

        let ws = weight.len() as u64 * f32_size;
        let wp_buf = self.upload_storage("snd_dag_conv_wp", ws, bytemuck::cast_slice(&weight_pos));
        let wn_buf = self.upload_storage("snd_dag_conv_wn", ws, bytemuck::cast_slice(&weight_neg));
        let bias_buf = self.upload_storage(
            "snd_dag_conv_b",
            (out_channels as u64) * f32_size,
            bytemuck::cast_slice(&bias_data),
        );

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snd_dag_conv_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wn_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dst_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dst_u.as_entire_binding(),
                },
            ],
        }))
    }

    /// Sound ReLU bind group (§3.7): in-place, bindings 0=params, 1=lower, 2=upper.
    pub(super) fn build_sound_dag_relu_bg(
        &self,
        layout: &wgpu::BindGroupLayout,
        params: &ReluIbpParams,
        lower: &wgpu::Buffer,
        upper: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let params_buf = self.upload_uniform("snd_dag_relu_p", bytemuck::bytes_of(params));
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snd_dag_relu_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lower.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: upper.as_entire_binding(),
                },
            ],
        })
    }

    /// Sound Add bind group (§3.5): bindings 0=params, 1=a_l, 2=a_u, 3=b_l, 4=b_u,
    /// 5=out_l, 6=out_u — 6 storage (two predecessors, residual connection).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_sound_dag_add_bg(
        &self,
        layout: &wgpu::BindGroupLayout,
        params: &AddIbpParams,
        a_l: &wgpu::Buffer,
        a_u: &wgpu::Buffer,
        b_l: &wgpu::Buffer,
        b_u: &wgpu::Buffer,
        out_l: &wgpu::Buffer,
        out_u: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let params_buf = self.upload_uniform("snd_dag_add_p", bytemuck::bytes_of(params));
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snd_dag_add_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_u.as_entire_binding(),
                },
            ],
        })
    }

    /// Sound AvgPool bind group (§3.4): bindings 0=params, 1=in_l, 2=in_u, 3=out_l,
    /// 4=out_u — 4 storage. Coefficient 1/D ≤ 1 ⇒ no §0 amplifier.
    pub(super) fn build_sound_dag_avgpool_bg(
        &self,
        layout: &wgpu::BindGroupLayout,
        params: &AvgPoolIbpSoundParams,
        in_l: &wgpu::Buffer,
        in_u: &wgpu::Buffer,
        out_l: &wgpu::Buffer,
        out_u: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let params_buf = self.upload_uniform("snd_dag_avgpool_p", bytemuck::bytes_of(params));
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snd_dag_avgpool_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: in_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_u.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a UNIFORM buffer and upload `bytes` (params blocks).
    fn upload_uniform(&self, label: &'static str, bytes: &[u8]) -> wgpu::Buffer {
        let buf = create_buffer(
            &self.device,
            label,
            bytes.len() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        self.queue.write_buffer(&buf, 0, bytes);
        buf
    }

    /// Create a read-only STORAGE buffer of `bytes` capacity and upload `data`.
    fn upload_storage(&self, label: &'static str, bytes: u64, data: &[u8]) -> wgpu::Buffer {
        let buf = create_buffer(
            &self.device,
            label,
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue.write_buffer(&buf, 0, data);
        buf
    }
}
