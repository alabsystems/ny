// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Plan construction for graph-DAG GPU-resident IBP (#4319).
//!
//! Builds execution steps and GPU buffers from a `GpuDagIbpPlanDesc`.

use ny_core::{checked_shape_product, GpuDagIbpOp, GpuDagIbpPlanDesc, NyError, Result};

use super::ibp_forward::create_buffer;
use super::ibp_forward_plan::shape_product_or_err;
use super::{
    gpu_checked_u32,
    ibp_graph_forward_plan::{
        compute_op_elems, compute_output_shape, resolve_bufs, resolve_elems, BufIdx, DagStep,
        OpBufPair, PipelineKind, WgpuDagIbpModelPlan,
    },
};
use crate::wgpu_device::params::{AddIbpParams, AvgPoolIbpParams, ReluIbpParams};
use crate::wgpu_device::WgpuDevice;

impl WgpuDevice {
    pub(super) fn prepare_dag_model_plan_internal(
        &self,
        plan: &GpuDagIbpPlanDesc,
    ) -> Result<WgpuDagIbpModelPlan> {
        let input_elements = shape_product_or_err(&plan.input_shape, "dag input")?;
        let f32_size = size_of::<f32>() as u64;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let op_elems = compute_op_elems(&plan.ops, input_elements)?;

        // Allocate input buffers.
        let input_buf_bytes = (input_elements as u64) * f32_size;
        let input_lower = create_buffer(&self.device, "dag_ibp_in_l", input_buf_bytes, usage_rw);
        let input_upper = create_buffer(&self.device, "dag_ibp_in_u", input_buf_bytes, usage_rw);

        // Allocate per-op output buffer pairs.
        let op_bufs: Vec<OpBufPair> = op_elems
            .iter()
            .map(|&elems| {
                let bytes = (elems as u64) * f32_size;
                OpBufPair {
                    lower: create_buffer(&self.device, "dag_op_l", bytes, usage_rw),
                    upper: create_buffer(&self.device, "dag_op_u", bytes, usage_rw),
                }
            })
            .collect();

        // Build execution steps.
        let mut steps: Vec<DagStep> = Vec::with_capacity(plan.ops.len() * 2);

        for (op_idx, op) in plan.ops.iter().enumerate() {
            match op {
                GpuDagIbpOp::Linear {
                    weight,
                    bias,
                    out_features,
                    in_features,
                    input_idx,
                } => {
                    let (src_l, src_u) =
                        resolve_bufs(*input_idx, &input_lower, &input_upper, &op_bufs);
                    let in_elems = resolve_elems(*input_idx, input_elements, &op_elems);
                    let batch_size = in_elems / *in_features;

                    let bg = self.build_dag_linear_bind_group(
                        src_l,
                        src_u,
                        &op_bufs[op_idx].lower,
                        &op_bufs[op_idx].upper,
                        weight,
                        bias.as_deref(),
                        *out_features,
                        *in_features,
                        batch_size,
                    )?;
                    let wg = gpu_checked_u32(op_elems[op_idx], "dag lin dispatch")?.div_ceil(64);
                    steps.push(DagStep::Compute {
                        pipeline_kind: PipelineKind::Linear,
                        bind_group: bg,
                        workgroup_count: wg,
                    });
                }

                GpuDagIbpOp::Conv2d {
                    weight,
                    bias,
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    groups,
                    input_h,
                    input_w,
                    input_idx,
                } => {
                    let (src_l, src_u) =
                        resolve_bufs(*input_idx, &input_lower, &input_upper, &op_bufs);
                    let in_elems = resolve_elems(*input_idx, input_elements, &op_elems);
                    let input_plane = checked_shape_product(&[*in_channels, *input_h, *input_w])
                        .ok_or_else(|| NyError::InvalidSpec("dag Conv2d input overflow".into()))?;
                    let batch_size = in_elems.checked_div(input_plane).unwrap_or(1);
                    let padded_h = input_h + 2 * pad_h;
                    let padded_w = input_w + 2 * pad_w;
                    let out_h = (padded_h - kernel_h) / stride_h + 1;
                    let out_w = (padded_w - kernel_w) / stride_w + 1;

                    let bg = self.build_dag_conv2d_bind_group(
                        src_l,
                        src_u,
                        &op_bufs[op_idx].lower,
                        &op_bufs[op_idx].upper,
                        weight,
                        bias.as_deref(),
                        *out_channels,
                        *in_channels,
                        *kernel_h,
                        *kernel_w,
                        *stride_h,
                        *stride_w,
                        *pad_h,
                        *pad_w,
                        *groups,
                        *input_h,
                        *input_w,
                        batch_size,
                        out_h,
                        out_w,
                    )?;
                    let wg = gpu_checked_u32(op_elems[op_idx], "dag conv dispatch")?.div_ceil(64);
                    steps.push(DagStep::Compute {
                        pipeline_kind: PipelineKind::Conv2d,
                        bind_group: bg,
                        workgroup_count: wg,
                    });
                }

                GpuDagIbpOp::ReLU {
                    num_elements,
                    input_idx,
                } => {
                    let bytes = (*num_elements as u64) * f32_size;
                    steps.push(DagStep::Copy {
                        src_lower_idx: BufIdx::from_plan_idx(*input_idx),
                        src_upper_idx: BufIdx::from_plan_idx(*input_idx),
                        dst_op_idx: op_idx,
                        bytes,
                    });

                    let params = ReluIbpParams {
                        num_elements: gpu_checked_u32(*num_elements, "dag relu")?,
                        _padding: [0; 3],
                    };
                    let params_buf = create_buffer(
                        &self.device,
                        "dag_relu_p",
                        size_of::<ReluIbpParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("dag_relu_bg"),
                        layout: &self.relu_ibp_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: op_bufs[op_idx].lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: op_bufs[op_idx].upper.as_entire_binding(),
                            },
                        ],
                    });
                    let wg = gpu_checked_u32(*num_elements, "dag relu dispatch")?.div_ceil(64);
                    steps.push(DagStep::Compute {
                        pipeline_kind: PipelineKind::ReLU,
                        bind_group: bg,
                        workgroup_count: wg,
                    });
                }

                GpuDagIbpOp::Add {
                    num_elements,
                    input_a_idx,
                    input_b_idx,
                } => {
                    let (src_a_l, src_a_u) =
                        resolve_bufs(*input_a_idx, &input_lower, &input_upper, &op_bufs);
                    let (src_b_l, src_b_u) =
                        resolve_bufs(*input_b_idx, &input_lower, &input_upper, &op_bufs);

                    let params = AddIbpParams {
                        num_elements: gpu_checked_u32(*num_elements, "dag add")?,
                        _padding: [0; 3],
                    };
                    let params_buf = create_buffer(
                        &self.device,
                        "dag_add_p",
                        size_of::<AddIbpParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("dag_add_bg"),
                        layout: &self.add_ibp_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: src_a_l.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: src_a_u.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: src_b_l.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: src_b_u.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: op_bufs[op_idx].lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: op_bufs[op_idx].upper.as_entire_binding(),
                            },
                        ],
                    });
                    let wg = gpu_checked_u32(*num_elements, "dag add dispatch")?.div_ceil(64);
                    steps.push(DagStep::Compute {
                        pipeline_kind: PipelineKind::Add,
                        bind_group: bg,
                        workgroup_count: wg,
                    });
                }

                GpuDagIbpOp::View { input_idx, .. } => {
                    let bytes = (op_elems[op_idx] as u64) * f32_size;
                    steps.push(DagStep::Copy {
                        src_lower_idx: BufIdx::from_plan_idx(*input_idx),
                        src_upper_idx: BufIdx::from_plan_idx(*input_idx),
                        dst_op_idx: op_idx,
                        bytes,
                    });
                }

                GpuDagIbpOp::AveragePool {
                    channels,
                    input_h,
                    input_w,
                    output_h,
                    output_w,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    count_include_pad,
                    num_elements,
                    input_idx,
                    ..
                } => {
                    let (src_l, src_u) =
                        resolve_bufs(*input_idx, &input_lower, &input_upper, &op_bufs);

                    let params = AvgPoolIbpParams {
                        num_elements: gpu_checked_u32(*num_elements, "dag avgpool")?,
                        channels: gpu_checked_u32(*channels, "dag avgpool ch")?,
                        input_h: gpu_checked_u32(*input_h, "dag avgpool ih")?,
                        input_w: gpu_checked_u32(*input_w, "dag avgpool iw")?,
                        output_h: gpu_checked_u32(*output_h, "dag avgpool oh")?,
                        output_w: gpu_checked_u32(*output_w, "dag avgpool ow")?,
                        kernel_h: gpu_checked_u32(*kernel_h, "dag avgpool kh")?,
                        kernel_w: gpu_checked_u32(*kernel_w, "dag avgpool kw")?,
                        stride_h: gpu_checked_u32(*stride_h, "dag avgpool sh")?,
                        stride_w: gpu_checked_u32(*stride_w, "dag avgpool sw")?,
                        pad_h: gpu_checked_u32(*pad_h, "dag avgpool ph")?,
                        pad_w: gpu_checked_u32(*pad_w, "dag avgpool pw")?,
                        count_include_pad: if *count_include_pad { 1 } else { 0 },
                        _padding: [0; 3],
                    };
                    let params_buf = create_buffer(
                        &self.device,
                        "dag_avgpool_p",
                        size_of::<AvgPoolIbpParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("dag_avgpool_bg"),
                        layout: &self.avgpool_ibp_bind_group_layout,
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
                                resource: op_bufs[op_idx].lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: op_bufs[op_idx].upper.as_entire_binding(),
                            },
                        ],
                    });
                    let wg = gpu_checked_u32(*num_elements, "dag avgpool dispatch")?.div_ceil(64);
                    steps.push(DagStep::Compute {
                        pipeline_kind: PipelineKind::AvgPool,
                        bind_group: bg,
                        workgroup_count: wg,
                    });
                }
            }
        }

        let output_elements = op_elems[plan.output_op_idx];
        let output_bytes = (output_elements as u64) * f32_size;
        let staging_lower = create_buffer(
            &self.device,
            "dag_stg_l",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = create_buffer(
            &self.device,
            "dag_stg_u",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let output_shape = compute_output_shape(plan)?;

        Ok(WgpuDagIbpModelPlan {
            device: self.device.clone(),
            queue: self.queue.clone(),
            linear_pipeline: self.linear_ibp_pipeline.clone(),
            conv2d_pipeline: self.conv2d_ibp_pipeline.clone(),
            relu_pipeline: self.relu_ibp_pipeline.clone(),
            add_pipeline: self.add_ibp_pipeline.clone(),
            avgpool_pipeline: self.avgpool_ibp_pipeline.clone(),
            input_shape: plan.input_shape.clone(),
            input_elements,
            output_elements,
            output_shape,
            output_op_idx: plan.output_op_idx,
            input_lower,
            input_upper,
            op_bufs,
            staging_lower,
            staging_upper,
            steps,
        })
    }
}
