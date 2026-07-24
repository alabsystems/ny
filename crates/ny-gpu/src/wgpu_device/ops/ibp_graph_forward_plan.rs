// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cached-plan support for graph-DAG GPU-resident IBP forward (#4319).
//!
//! Unlike the sequential plan (`ibp_forward_plan.rs`) which ping-pongs between
//! two buffer pairs, the DAG plan allocates one buffer pair per op so that
//! residual Add can read from two arbitrary predecessors.
//!
//! Split across three sibling files:
//! - `ibp_graph_forward_plan.rs` — types, trait impl, shape helpers
//! - `ibp_graph_forward_plan_build.rs` — plan construction
//! - `ibp_graph_forward_plan_bind.rs` — bind group builders

use ny_core::{
    checked_shape_product, GpuDagIbpModelPlan, GpuDagIbpOp, GpuDagIbpPlanDesc, GpuIbpResult,
    NyError, Result, NETWORK_INPUT_IDX,
};

use super::super::WgpuDevice;
use super::ibp_forward_plan::shape_product_or_err;
use super::sanitize_readback;

/// One execution step in the cached DAG plan.
pub(super) enum DagStep {
    /// Copy src buffer pair to dst (for View and pre-ReLU).
    Copy {
        src_lower_idx: BufIdx,
        src_upper_idx: BufIdx,
        dst_op_idx: usize,
        bytes: u64,
    },
    /// Compute dispatch (src→dst via bind group).
    Compute {
        pipeline_kind: PipelineKind,
        bind_group: wgpu::BindGroup,
        workgroup_count: u32,
    },
}

#[derive(Clone, Copy)]
pub(super) enum PipelineKind {
    Linear,
    Conv2d,
    ReLU,
    Add,
    AvgPool,
}

/// Buffer index: either the network input pair or an op's output pair.
#[derive(Clone, Copy)]
pub(super) enum BufIdx {
    Input,
    Op(usize),
}

impl BufIdx {
    pub(super) fn from_plan_idx(idx: usize) -> Self {
        if idx == NETWORK_INPUT_IDX {
            BufIdx::Input
        } else {
            BufIdx::Op(idx)
        }
    }
}

pub(super) struct OpBufPair {
    pub(super) lower: wgpu::Buffer,
    pub(super) upper: wgpu::Buffer,
}

pub(super) struct WgpuDagIbpModelPlan {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) linear_pipeline: wgpu::ComputePipeline,
    pub(super) conv2d_pipeline: wgpu::ComputePipeline,
    pub(super) relu_pipeline: wgpu::ComputePipeline,
    pub(super) add_pipeline: wgpu::ComputePipeline,
    pub(super) avgpool_pipeline: wgpu::ComputePipeline,
    pub(super) input_shape: Vec<usize>,
    pub(super) input_elements: usize,
    pub(super) output_elements: usize,
    pub(super) output_shape: Vec<usize>,
    pub(super) output_op_idx: usize,
    pub(super) input_lower: wgpu::Buffer,
    pub(super) input_upper: wgpu::Buffer,
    pub(super) op_bufs: Vec<OpBufPair>,
    pub(super) staging_lower: wgpu::Buffer,
    pub(super) staging_upper: wgpu::Buffer,
    pub(super) steps: Vec<DagStep>,
}

impl GpuDagIbpModelPlan for WgpuDagIbpModelPlan {
    fn dag_ibp_forward_cached(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        // Wrap GPU work in an error scope: a wgpu error returns Err (caller
        // falls back to CPU) instead of aborting via wgpu's panicking handler.
        super::super::error_scope::run_gpu_checked_on_device(
            &self.device,
            "dag_ibp_forward_cached",
            || self.dag_ibp_forward_cached_inner(input_lower, input_upper, input_shape),
        )
    }
}

impl WgpuDagIbpModelPlan {
    fn dag_ibp_forward_cached_inner(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        if input_shape != self.input_shape {
            return Err(NyError::InvalidSpec(format!(
                "dag_ibp: shape mismatch, expected {:?}, got {:?}",
                self.input_shape, input_shape
            )));
        }
        if input_lower.len() != self.input_elements || input_upper.len() != self.input_elements {
            return Err(NyError::InvalidSpec(format!(
                "dag_ibp: length mismatch, expected {}, got lower={} upper={}",
                self.input_elements,
                input_lower.len(),
                input_upper.len()
            )));
        }

        self.queue
            .write_buffer(&self.input_lower, 0, bytemuck::cast_slice(input_lower));
        self.queue
            .write_buffer(&self.input_upper, 0, bytemuck::cast_slice(input_upper));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dag_ibp_encoder"),
            });

        for step in &self.steps {
            match step {
                DagStep::Copy {
                    src_lower_idx,
                    src_upper_idx,
                    dst_op_idx,
                    bytes,
                } => {
                    let sl = match *src_lower_idx {
                        BufIdx::Input => &self.input_lower,
                        BufIdx::Op(i) => &self.op_bufs[i].lower,
                    };
                    let su = match *src_upper_idx {
                        BufIdx::Input => &self.input_upper,
                        BufIdx::Op(j) => &self.op_bufs[j].upper,
                    };
                    encoder.copy_buffer_to_buffer(
                        sl,
                        0,
                        &self.op_bufs[*dst_op_idx].lower,
                        0,
                        *bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        su,
                        0,
                        &self.op_bufs[*dst_op_idx].upper,
                        0,
                        *bytes,
                    );
                }
                DagStep::Compute {
                    pipeline_kind,
                    bind_group,
                    workgroup_count,
                } => {
                    let pipeline = match pipeline_kind {
                        PipelineKind::Linear => &self.linear_pipeline,
                        PipelineKind::Conv2d => &self.conv2d_pipeline,
                        PipelineKind::ReLU => &self.relu_pipeline,
                        PipelineKind::Add => &self.add_pipeline,
                        PipelineKind::AvgPool => &self.avgpool_pipeline,
                    };
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("dag_ibp_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(*workgroup_count, 1, 1);
                }
            }
        }

        let output_bytes = (self.output_elements as u64) * size_of::<f32>() as u64;
        encoder.copy_buffer_to_buffer(
            &self.op_bufs[self.output_op_idx].lower,
            0,
            &self.staging_lower,
            0,
            output_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.op_bufs[self.output_op_idx].upper,
            0,
            &self.staging_upper,
            0,
            output_bytes,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let mut result_lower =
            WgpuDevice::read_buffer(&self.device, &self.staging_lower, self.output_elements)?;
        let mut result_upper =
            WgpuDevice::read_buffer(&self.device, &self.staging_upper, self.output_elements)?;

        sanitize_readback(&mut result_lower, &mut result_upper);

        Ok(GpuIbpResult {
            lower_bounds: result_lower,
            upper_bounds: result_upper,
            output_shape: self.output_shape.clone(),
        })
    }
}

// --- Shape and element helpers ---

pub(super) fn resolve_bufs<'a>(
    idx: usize,
    input_l: &'a wgpu::Buffer,
    input_u: &'a wgpu::Buffer,
    op_bufs: &'a [OpBufPair],
) -> (&'a wgpu::Buffer, &'a wgpu::Buffer) {
    if idx == NETWORK_INPUT_IDX {
        (input_l, input_u)
    } else {
        (&op_bufs[idx].lower, &op_bufs[idx].upper)
    }
}

pub(super) fn resolve_elems(idx: usize, input_elements: usize, op_elems: &[usize]) -> usize {
    if idx == NETWORK_INPUT_IDX {
        input_elements
    } else {
        op_elems[idx]
    }
}

pub(super) fn compute_op_elems(ops: &[GpuDagIbpOp], input_elements: usize) -> Result<Vec<usize>> {
    let mut elems = Vec::with_capacity(ops.len());
    for op in ops {
        let primary = match op {
            GpuDagIbpOp::Linear { input_idx, .. }
            | GpuDagIbpOp::Conv2d { input_idx, .. }
            | GpuDagIbpOp::ReLU { input_idx, .. }
            | GpuDagIbpOp::View { input_idx, .. }
            | GpuDagIbpOp::AveragePool { input_idx, .. } => *input_idx,
            GpuDagIbpOp::Add { input_a_idx, .. } => *input_a_idx,
        };
        let in_e = resolve_elems(primary, input_elements, &elems);
        let out_e = match op {
            GpuDagIbpOp::Linear {
                out_features,
                in_features,
                ..
            } => {
                if *in_features == 0 || !in_e.is_multiple_of(*in_features) {
                    return Err(NyError::shape_mismatch(vec![*in_features], vec![in_e]));
                }
                (in_e / *in_features)
                    .checked_mul(*out_features)
                    .ok_or_else(|| NyError::InvalidSpec("dag linear overflow".into()))?
            }
            GpuDagIbpOp::Conv2d {
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                input_h,
                input_w,
                ..
            } => {
                let plane = checked_shape_product(&[*in_channels, *input_h, *input_w])
                    .ok_or_else(|| NyError::InvalidSpec("dag conv input overflow".into()))?;
                let batch = in_e.checked_div(plane).unwrap_or(1);
                let oh = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
                let ow = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
                checked_shape_product(&[batch, *out_channels, oh, ow])
                    .ok_or_else(|| NyError::InvalidSpec("dag conv output overflow".into()))?
            }
            GpuDagIbpOp::ReLU { num_elements, .. }
            | GpuDagIbpOp::Add { num_elements, .. }
            | GpuDagIbpOp::AveragePool { num_elements, .. } => *num_elements,
            GpuDagIbpOp::View { output_shape, .. } => {
                shape_product_or_err(output_shape, "dag view")?
            }
        };
        elems.push(out_e);
    }
    Ok(elems)
}

pub(super) fn compute_output_shape(plan: &GpuDagIbpPlanDesc) -> Result<Vec<usize>> {
    let input_shape = &plan.input_shape;
    let mut op_shapes: Vec<Vec<usize>> = Vec::with_capacity(plan.ops.len());
    for op in &plan.ops {
        let shape = match op {
            GpuDagIbpOp::Linear {
                out_features,
                in_features,
                input_idx,
                ..
            } => {
                let s = resolve_shape(*input_idx, input_shape, &op_shapes);
                let mut v = s.to_vec();
                if let Some(last) = v.last_mut() {
                    if *last != *in_features {
                        return Err(NyError::shape_mismatch(vec![*in_features], vec![*last]));
                    }
                    *last = *out_features;
                }
                v
            }
            GpuDagIbpOp::Conv2d {
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                input_h,
                input_w,
                input_idx,
                ..
            } => {
                let s = resolve_shape(*input_idx, input_shape, &op_shapes);
                let oh = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
                let ow = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
                match s.len() {
                    3 => vec![*out_channels, oh, ow],
                    4 => vec![s[0], *out_channels, oh, ow],
                    _ => return Err(NyError::InvalidSpec("dag Conv2d shape".into())),
                }
            }
            GpuDagIbpOp::ReLU { input_idx, .. } => {
                resolve_shape(*input_idx, input_shape, &op_shapes).to_vec()
            }
            GpuDagIbpOp::Add { input_a_idx, .. } => {
                resolve_shape(*input_a_idx, input_shape, &op_shapes).to_vec()
            }
            GpuDagIbpOp::View { output_shape, .. } => output_shape.to_vec(),
            GpuDagIbpOp::AveragePool {
                channels,
                output_h,
                output_w,
                input_idx,
                ..
            } => {
                let s = resolve_shape(*input_idx, input_shape, &op_shapes);
                match s.len() {
                    3 => vec![*channels, *output_h, *output_w],
                    4 => vec![s[0], *channels, *output_h, *output_w],
                    _ => return Err(NyError::InvalidSpec("dag AvgPool shape".into())),
                }
            }
        };
        op_shapes.push(shape);
    }
    Ok(op_shapes[plan.output_op_idx].clone())
}

fn resolve_shape<'a>(
    idx: usize,
    input_shape: &'a [usize],
    op_shapes: &'a [Vec<usize>],
) -> &'a [usize] {
    if idx == NETWORK_INPUT_IDX {
        input_shape
    } else {
        &op_shapes[idx]
    }
}
