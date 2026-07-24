// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bind group builders for graph-DAG GPU-resident IBP (#4319).
//!
//! Extracted from `ibp_graph_forward_plan_build.rs` for file-size compliance.

use ny_core::{nan_propagating_max_zero, nan_propagating_min_zero, Result};

use super::gpu_checked_u32;
use super::ibp_forward::create_buffer;
use crate::wgpu_device::params::{Conv2dIbpParams, LinearIbpParams};
use crate::wgpu_device::WgpuDevice;

impl WgpuDevice {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_dag_linear_bind_group(
        &self,
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

        let params = LinearIbpParams {
            batch_size: gpu_checked_u32(batch_size, "dag lin batch")?,
            in_features: gpu_checked_u32(in_features, "dag lin in")?,
            out_features: gpu_checked_u32(out_features, "dag lin out")?,
            _padding: 0,
        };
        let params_buf = create_buffer(
            &self.device,
            "dag_lin_p",
            size_of::<LinearIbpParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

        let ws = weight.len() as u64 * f32_size;
        let wp_buf = create_buffer(
            &self.device,
            "dag_lin_wp",
            ws,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&wp_buf, 0, bytemuck::cast_slice(&weight_pos));
        let wn_buf = create_buffer(
            &self.device,
            "dag_lin_wn",
            ws,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&wn_buf, 0, bytemuck::cast_slice(&weight_neg));
        let bias_buf = create_buffer(
            &self.device,
            "dag_lin_b",
            (out_features as u64) * f32_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&bias_buf, 0, bytemuck::cast_slice(&bias_data));

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dag_lin_bg"),
            layout: &self.linear_ibp_bind_group_layout,
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_dag_conv2d_bind_group(
        &self,
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
        groups: usize,
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

        let params = Conv2dIbpParams {
            batch_size: gpu_checked_u32(batch_size, "dag conv batch")?,
            in_channels: gpu_checked_u32(in_channels, "dag conv in_c")?,
            out_channels: gpu_checked_u32(out_channels, "dag conv out_c")?,
            input_h: gpu_checked_u32(input_h, "dag conv in_h")?,
            input_w: gpu_checked_u32(input_w, "dag conv in_w")?,
            out_h: gpu_checked_u32(out_h, "dag conv out_h")?,
            out_w: gpu_checked_u32(out_w, "dag conv out_w")?,
            kernel_h: gpu_checked_u32(kernel_h, "dag conv kh")?,
            kernel_w: gpu_checked_u32(kernel_w, "dag conv kw")?,
            stride_h: gpu_checked_u32(stride_h, "dag conv sh")?,
            stride_w: gpu_checked_u32(stride_w, "dag conv sw")?,
            pad_h: gpu_checked_u32(pad_h, "dag conv ph")?,
            pad_w: gpu_checked_u32(pad_w, "dag conv pw")?,
            groups: gpu_checked_u32(groups, "dag conv groups")?,
            _padding: [0; 2],
        };
        let params_buf = create_buffer(
            &self.device,
            "dag_conv_p",
            size_of::<Conv2dIbpParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

        let ws = weight.len() as u64 * f32_size;
        let wp_buf = create_buffer(
            &self.device,
            "dag_conv_wp",
            ws,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&wp_buf, 0, bytemuck::cast_slice(&weight_pos));
        let wn_buf = create_buffer(
            &self.device,
            "dag_conv_wn",
            ws,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&wn_buf, 0, bytemuck::cast_slice(&weight_neg));
        let bias_buf = create_buffer(
            &self.device,
            "dag_conv_b",
            (out_channels as u64) * f32_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&bias_buf, 0, bytemuck::cast_slice(&bias_data));

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dag_conv_bg"),
            layout: &self.conv2d_ibp_bind_group_layout,
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
}
