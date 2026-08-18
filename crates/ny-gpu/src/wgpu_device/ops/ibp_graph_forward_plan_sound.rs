// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SOUND (verdict-legal) plan construction for graph-DAG GPU-resident IBP
//! (`docs/SOUND_GPU_IBP_PLAN.md` T1.0 — the DAG sibling of the sequential
//! `ibp_forward_sound.rs`).
//!
//! # Why this is a thin sibling of the FAST builder
//! The sound DAG is the **fast DAG topology + sound pipelines + sound params**.
//! There is NO new buffer kind and NO cross-node error buffer: each sound op is a
//! self-contained interval widening that reads only its predecessors' `[lo, hi]`
//! buffers and writes `[lo − r_lo, hi + r_hi]`. By induction over the topological
//! order, if every predecessor buffer already encloses the truth then so does the
//! output (compositional interval arithmetic — an op need not see exact reals, only
//! enclosing endpoints). So this builder reuses the fast [`WgpuDagIbpModelPlan`]
//! struct and its already-tested execution VERBATIM, and only swaps, per op:
//!   * the pipeline → `self.ibp_sound_pipelines().<kind>`,
//!   * the params → the sound param struct carrying `gamma_k`/`slack`/`additive`
//!     (`n_ulps` for the amplified reductions),
//!   * the bind-group layout → the matching sound pipeline layout.
//!
//! # Soundness of each swapped op (all Metal FTZ-safe — every additive floor is
//! NORMAL-range; the §0 amplified-flush term rides in `flushacc` inside the shader)
//!   * **Linear/Conv2d** — the §3.1 keystone reduction: `k = reduction + 3`,
//!     `n_ulps = 2·(reduction + 2)`, `3·γ_k·S + 4·N·u·|endpoint| + flush` radius.
//!     8 buffers (1 uniform + 7 storage) = exactly Metal's limit. Grouped Conv2d is
//!     REJECTED host-side (the shader's FALLBACK path is not sized for it).
//!   * **ReLU** — exact + monotone; `widen_lo`/`widen_hi` (coefficient ≤ 1). The
//!     `widen_hi` additive floor `ADDITIVE1 = 2⁻¹²²` dominates any FTZ-flushed
//!     subnormal upper endpoint (≤ 2⁻¹²⁶), so a flushed `max(0, subnormal)` stays
//!     enclosed. In-place, 2 storage.
//!   * **Add** — coefficient 1, single RN add per endpoint; `widen_lo`/`widen_hi`.
//!     6 storage (two predecessors).
//!   * **AvgPool** — coefficient `1/D ≤ 1` ⇒ no §0 amplifier, only the normal-range
//!     floor; `k = kernel_h·kernel_w + 3`. Batch is handled implicitly by the flat
//!     `num_elements` dispatch (the shader's `channels` param is inert). 4 storage.
//!   * **View** — element-preserving Copy (no dispatch, no widening — exact).
//!
//! Any wgpu error (including a first-use sound-shader compile failure) becomes `Err`
//! via the `run_gpu_checked` wrapper, so a verdict is never decided by a failed op —
//! the caller falls back to the proven-sound CPU graph loop.

use ny_core::{
    checked_shape_product, ftz_safe_underflow_floor, GpuDagIbpOp, GpuDagIbpPlanDesc, NyError,
    Result,
};

use crate::wgpu_device::params::{AddIbpParams, AvgPoolIbpSoundParams, ReluIbpParams};
use crate::wgpu_device::sound_consts::{combine_slack_f32, gamma_k_f32};
use crate::wgpu_device::WgpuDevice;

use super::ibp_forward::create_buffer;
use super::ibp_forward_plan::shape_product_or_err;
use super::{
    gpu_checked_u32,
    ibp_graph_forward_plan::{
        compute_op_elems, compute_output_shape, resolve_bufs, resolve_elems, BufIdx, DagStep,
        OpBufPair, PipelineKind, WgpuDagIbpModelPlan,
    },
};

impl WgpuDevice {
    /// Build a SOUND graph-DAG resident-IBP plan (T1.0). Same topology and readback
    /// as [`WgpuDevice::prepare_dag_model_plan_internal`], but every op dispatches a
    /// certified sound shader. Wrapped in `run_gpu_checked` so the one-time sound
    /// shader compilation (and every bind-group validation) can only yield `Err`
    /// (→ CPU sound fallback), never a process abort.
    pub(super) fn prepare_sound_dag_model_plan_internal(
        &self,
        plan: &GpuDagIbpPlanDesc,
    ) -> Result<WgpuDagIbpModelPlan> {
        self.run_gpu_checked("prepare_sound_dag_model_plan", || {
            self.prepare_sound_dag_model_plan_encode(plan)
        })
    }

    fn prepare_sound_dag_model_plan_encode(
        &self,
        plan: &GpuDagIbpPlanDesc,
    ) -> Result<WgpuDagIbpModelPlan> {
        let input_elements = shape_product_or_err(&plan.input_shape, "sound dag input")?;
        let f32_size = size_of::<f32>() as u64;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let op_elems = compute_op_elems(&plan.ops, input_elements)?;

        // The sound pipelines are lazily compiled here (under the run_gpu_checked
        // error scope) and reused forever. Each is (ComputePipeline, BindGroupLayout).
        let pipes = self.ibp_sound_pipelines();

        // Input + per-op output buffer pairs (identical allocation to the fast path).
        let input_buf_bytes = (input_elements as u64) * f32_size;
        let input_lower = create_buffer(&self.device, "snd_dag_in_l", input_buf_bytes, usage_rw);
        let input_upper = create_buffer(&self.device, "snd_dag_in_u", input_buf_bytes, usage_rw);

        let op_bufs: Vec<OpBufPair> = op_elems
            .iter()
            .map(|&elems| {
                let bytes = (elems as u64) * f32_size;
                OpBufPair {
                    lower: create_buffer(&self.device, "snd_dag_op_l", bytes, usage_rw),
                    upper: create_buffer(&self.device, "snd_dag_op_u", bytes, usage_rw),
                }
            })
            .collect();

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
                    if *in_features == 0 || !in_elems.is_multiple_of(*in_features) {
                        return Err(NyError::shape_mismatch(vec![*in_features], vec![in_elems]));
                    }
                    let batch_size = in_elems / *in_features;

                    let bg = self.build_sound_dag_linear_bg(
                        &pipes.linear.1,
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
                    let wg =
                        gpu_checked_u32(op_elems[op_idx], "sound dag lin dispatch")?.div_ceil(64);
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
                    if *groups != 1 {
                        // The sound conv shader emits a maximal FALLBACK superset for
                        // groups!=1, but the host sizing below assumes groups=1;
                        // reject so the caller takes the proven-sound CPU loop.
                        return Err(NyError::UnsupportedOp(
                            "sound DAG IBP: grouped Conv2d not certified; CPU sound fallback"
                                .into(),
                        ));
                    }
                    let (src_l, src_u) =
                        resolve_bufs(*input_idx, &input_lower, &input_upper, &op_bufs);
                    let in_elems = resolve_elems(*input_idx, input_elements, &op_elems);
                    let input_plane = checked_shape_product(&[*in_channels, *input_h, *input_w])
                        .ok_or_else(|| {
                            NyError::InvalidSpec("sound dag Conv2d input overflow".into())
                        })?;
                    let batch_size = in_elems.checked_div(input_plane).unwrap_or(1);
                    let padded_h = input_h + 2 * pad_h;
                    let padded_w = input_w + 2 * pad_w;
                    let out_h = (padded_h - kernel_h) / stride_h + 1;
                    let out_w = (padded_w - kernel_w) / stride_w + 1;

                    let bg = self.build_sound_dag_conv2d_bg(
                        &pipes.conv2d.1,
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
                        *input_h,
                        *input_w,
                        batch_size,
                        out_h,
                        out_w,
                    )?;
                    let wg =
                        gpu_checked_u32(op_elems[op_idx], "sound dag conv dispatch")?.div_ceil(64);
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
                    // Copy predecessor bounds into this op's pair, then rewrite them
                    // in place with the sound ReLU (exact + monotone + widen).
                    let bytes = (*num_elements as u64) * f32_size;
                    steps.push(DagStep::Copy {
                        src_lower_idx: BufIdx::from_plan_idx(*input_idx),
                        src_upper_idx: BufIdx::from_plan_idx(*input_idx),
                        dst_op_idx: op_idx,
                        bytes,
                    });

                    let params = ReluIbpParams {
                        num_elements: gpu_checked_u32(*num_elements, "sound dag relu")?,
                        _padding: [0; 3],
                    };
                    let bg = self.build_sound_dag_relu_bg(
                        &pipes.relu.1,
                        &params,
                        &op_bufs[op_idx].lower,
                        &op_bufs[op_idx].upper,
                    );
                    let wg =
                        gpu_checked_u32(*num_elements, "sound dag relu dispatch")?.div_ceil(64);
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
                        num_elements: gpu_checked_u32(*num_elements, "sound dag add")?,
                        _padding: [0; 3],
                    };
                    let bg = self.build_sound_dag_add_bg(
                        &pipes.add.1,
                        &params,
                        src_a_l,
                        src_a_u,
                        src_b_l,
                        src_b_u,
                        &op_bufs[op_idx].lower,
                        &op_bufs[op_idx].upper,
                    );
                    let wg = gpu_checked_u32(*num_elements, "sound dag add dispatch")?.div_ceil(64);
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

                    // k = kernel window + 3; coefficient 1/D ≤ 1 ⇒ normal-range floor
                    // (no §0 amplifier). Batch is folded into the flat num_elements
                    // dispatch — the shader's `channels` field is inert for indexing.
                    let kk = kernel_h
                        .checked_mul(*kernel_w)
                        .and_then(|v| v.checked_add(3))
                        .ok_or_else(|| {
                            NyError::InvalidSpec("sound dag avgpool k overflow".into())
                        })?;
                    let kk_u32 = gpu_checked_u32(kk, "sound dag avgpool k")?;
                    let params = AvgPoolIbpSoundParams {
                        num_elements: gpu_checked_u32(*num_elements, "sound dag avgpool elems")?,
                        channels: gpu_checked_u32(*channels, "sound dag avgpool ch")?,
                        input_h: gpu_checked_u32(*input_h, "sound dag avgpool ih")?,
                        input_w: gpu_checked_u32(*input_w, "sound dag avgpool iw")?,
                        output_h: gpu_checked_u32(*output_h, "sound dag avgpool oh")?,
                        output_w: gpu_checked_u32(*output_w, "sound dag avgpool ow")?,
                        kernel_h: gpu_checked_u32(*kernel_h, "sound dag avgpool kh")?,
                        kernel_w: gpu_checked_u32(*kernel_w, "sound dag avgpool kw")?,
                        stride_h: gpu_checked_u32(*stride_h, "sound dag avgpool sh")?,
                        stride_w: gpu_checked_u32(*stride_w, "sound dag avgpool sw")?,
                        pad_h: gpu_checked_u32(*pad_h, "sound dag avgpool ph")?,
                        pad_w: gpu_checked_u32(*pad_w, "sound dag avgpool pw")?,
                        count_include_pad: u32::from(*count_include_pad),
                        gamma_k: gamma_k_f32(kk)?,
                        slack: combine_slack_f32(kk)?,
                        additive: ftz_safe_underflow_floor(kk_u32),
                    };
                    let bg = self.build_sound_dag_avgpool_bg(
                        &pipes.avgpool.1,
                        &params,
                        src_l,
                        src_u,
                        &op_bufs[op_idx].lower,
                        &op_bufs[op_idx].upper,
                    );
                    let wg =
                        gpu_checked_u32(*num_elements, "sound dag avgpool dispatch")?.div_ceil(64);
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
            "snd_dag_stg_l",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = create_buffer(
            &self.device,
            "snd_dag_stg_u",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let output_shape = compute_output_shape(plan)?;

        Ok(WgpuDagIbpModelPlan {
            device: self.device.clone(),
            queue: self.queue.clone(),
            // SOUND pipelines in the fast plan's pipeline slots: a `PipelineKind::X`
            // step now dispatches the certified sound shader for X.
            linear_pipeline: pipes.linear.0.clone(),
            conv2d_pipeline: pipes.conv2d.0.clone(),
            relu_pipeline: pipes.relu.0.clone(),
            add_pipeline: pipes.add.0.clone(),
            avgpool_pipeline: pipes.avgpool.0.clone(),
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
