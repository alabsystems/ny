// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched command encoding for GPU CROWN backward pass (#3397).
//!
//! Encodes all dispatch steps into a single `CommandEncoder`, copying
//! per-dispatch data from the staging buffer via `copy_buffer_to_buffer`
//! before each compute pass.

use ny_core::{NyError, Result};

use super::super::WgpuDevice;
use super::crown_backward_types::DispatchStep;
use super::crown_plan::PreparedCrownPlan;
use super::crown_plan_working::StepBindGroups;
use super::crown_timestamps::CrownTimestampProfiler;
use super::gemm::select_gemm_dispatch;

fn step_compute_pass_count(step: &DispatchStep) -> u32 {
    match step {
        DispatchStep::ActivationBackward { .. }
        | DispatchStep::MaxPool2dBackward { .. }
        | DispatchStep::BiasAccumulate { .. }
        | DispatchStep::Concretize { .. } => 1,
        DispatchStep::GemmCrownLinear { .. }
        | DispatchStep::GemmCrownConv { .. }
        | DispatchStep::ConvReshapeLowerUpper { .. }
        | DispatchStep::ConvCol2imLowerUpper { .. } => 2,
    }
}

pub(super) fn count_encoded_compute_passes(steps: &[DispatchStep]) -> u32 {
    steps.iter().map(step_compute_pass_count).sum()
}

impl WgpuDevice {
    /// Encode all dispatch steps using pre-built bind groups (#3397 Step 4).
    ///
    /// Unlike `encode_crown_steps`, this method uses bind groups that were
    /// created at plan build time and stored in `plan.step_bind_groups`.
    /// Only staging buffer copies and compute pass encoding remain per-call.
    pub(super) fn encode_crown_steps_cached(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &PreparedCrownPlan,
        mut profiler: Option<&mut CrownTimestampProfiler>,
    ) -> Result<()> {
        let working = &plan.working;
        for (step, bg) in plan.steps.iter().zip(plan.step_bind_groups.iter()) {
            match (step, bg) {
                (
                    DispatchStep::ActivationBackward {
                        params_off,
                        params_size,
                        slopes_off,
                        slopes_size,
                        num_specs_u32,
                        dual_alpha,
                        ..
                    },
                    StepBindGroups::Activation { bind_group },
                ) => {
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *params_off,
                        &working.params_buf,
                        0,
                        *params_size,
                    );
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *slopes_off,
                        &working.slopes_buf,
                        0,
                        *slopes_size,
                    );
                    // Select pipeline: dual-alpha uses sign-specific routing (#4313).
                    let pipeline = if *dual_alpha {
                        &self.crown_activation_relu_dual_alpha_pipeline
                    } else {
                        &self.crown_activation_backward_pipeline
                    };
                    Self::encode_compute(
                        encoder,
                        "crown_act",
                        pipeline,
                        bind_group,
                        (*num_specs_u32, 1, 1),
                        profiler.as_deref_mut(),
                    )?;
                }
                (
                    DispatchStep::MaxPool2dBackward {
                        params_off,
                        params_size,
                        routing_off,
                        routing_size,
                        bounds_off,
                        bounds_size,
                        num_specs_u32,
                        ..
                    },
                    StepBindGroups::MaxPool2d { bind_group },
                ) => {
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *params_off,
                        &working.params_buf,
                        0,
                        *params_size,
                    );
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *routing_off,
                        &working.weight_buf,
                        0,
                        *routing_size,
                    );
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *bounds_off,
                        &working.slopes_buf,
                        0,
                        *bounds_size,
                    );
                    Self::encode_compute(
                        encoder,
                        "crown_maxpool2d",
                        &self.crown_maxpool2d_backward_pipeline,
                        bind_group,
                        (*num_specs_u32, 1, 1),
                        profiler.as_deref_mut(),
                    )?;
                }
                (
                    DispatchStep::BiasAccumulate {
                        params_off,
                        params_size,
                        bias_off,
                        bias_size,
                        num_specs_u32,
                        ..
                    },
                    StepBindGroups::BiasAccumulate { bind_group },
                ) => {
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *params_off,
                        &working.params_buf,
                        0,
                        *params_size,
                    );
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *bias_off,
                        &working.layer_bias_buf,
                        0,
                        *bias_size,
                    );
                    Self::encode_compute(
                        encoder,
                        "crown_bias",
                        &self.crown_bias_accumulate_pipeline,
                        bind_group,
                        (*num_specs_u32, 1, 1),
                        profiler.as_deref_mut(),
                    )?;
                }
                (
                    DispatchStep::GemmCrownLinear {
                        params_off,
                        params_size,
                        weight_off,
                        weight_size,
                        gemm_params,
                        ..
                    },
                    StepBindGroups::GemmLinear { lower, upper },
                ) => {
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *params_off,
                        &working.params_buf,
                        0,
                        *params_size,
                    );
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *weight_off,
                        &working.weight_buf,
                        0,
                        *weight_size,
                    );
                    // Select pipeline based on K dimension (#3599).
                    let (pipeline, wg_x, wg_y) = self.gemm_pipeline_and_dispatch(gemm_params);
                    Self::encode_compute(
                        encoder,
                        "crown_gemm_lower",
                        pipeline,
                        lower,
                        (wg_x, wg_y, 1),
                        profiler.as_deref_mut(),
                    )?;
                    Self::encode_compute(
                        encoder,
                        "crown_gemm_upper",
                        pipeline,
                        upper,
                        (wg_x, wg_y, 1),
                        profiler.as_deref_mut(),
                    )?;
                }
                (
                    DispatchStep::GemmCrownConv {
                        params_off,
                        params_size,
                        weight_off,
                        weight_size,
                        gemm_params,
                    },
                    StepBindGroups::GemmConv { lower, upper },
                ) => {
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *params_off,
                        &working.params_buf,
                        0,
                        *params_size,
                    );
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *weight_off,
                        &working.weight_buf,
                        0,
                        *weight_size,
                    );
                    // Select pipeline based on K dimension (#3599).
                    let (pipeline, wg_x, wg_y) = self.gemm_pipeline_and_dispatch(gemm_params);
                    Self::encode_compute(
                        encoder,
                        "crown_gemm_lower",
                        pipeline,
                        lower,
                        (wg_x, wg_y, 1),
                        profiler.as_deref_mut(),
                    )?;
                    Self::encode_compute(
                        encoder,
                        "crown_gemm_upper",
                        pipeline,
                        upper,
                        (wg_x, wg_y, 1),
                        profiler.as_deref_mut(),
                    )?;
                }
                (
                    DispatchStep::ConvReshapeLowerUpper {
                        params_off,
                        params_size,
                        workgroups,
                        ..
                    },
                    StepBindGroups::ConvReshape { lower, upper },
                ) => {
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *params_off,
                        &working.params_buf,
                        0,
                        *params_size,
                    );
                    Self::encode_compute(
                        encoder,
                        "conv_reshape_lower",
                        &self.conv_reshape_pipeline,
                        lower,
                        (*workgroups, 1, 1),
                        profiler.as_deref_mut(),
                    )?;
                    Self::encode_compute(
                        encoder,
                        "conv_reshape_upper",
                        &self.conv_reshape_pipeline,
                        upper,
                        (*workgroups, 1, 1),
                        profiler.as_deref_mut(),
                    )?;
                }
                (
                    DispatchStep::ConvCol2imLowerUpper {
                        params_off,
                        params_size,
                        workgroups,
                        ..
                    },
                    StepBindGroups::ConvCol2im { lower, upper },
                ) => {
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *params_off,
                        &working.params_buf,
                        0,
                        *params_size,
                    );
                    Self::encode_compute(
                        encoder,
                        "conv_col2im_lower",
                        &self.conv_col2im_pipeline,
                        lower,
                        (*workgroups, 1, 1),
                        profiler.as_deref_mut(),
                    )?;
                    Self::encode_compute(
                        encoder,
                        "conv_col2im_upper",
                        &self.conv_col2im_pipeline,
                        upper,
                        (*workgroups, 1, 1),
                        profiler.as_deref_mut(),
                    )?;
                }
                (
                    DispatchStep::Concretize {
                        params_off,
                        params_size,
                        num_specs_u32,
                        ..
                    },
                    StepBindGroups::Concretize { bind_group },
                ) => {
                    encoder.copy_buffer_to_buffer(
                        &plan.staging_buf,
                        *params_off,
                        &working.params_buf,
                        0,
                        *params_size,
                    );
                    Self::encode_compute(
                        encoder,
                        "crown_conc",
                        &self.crown_concretize_pipeline,
                        bind_group,
                        (*num_specs_u32, 1, 1),
                        profiler.as_deref_mut(),
                    )?;
                }
                _ => {
                    return Err(NyError::InternalError(
                        "crown plan: step/bind_group type mismatch".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Select GEMM pipeline and compute dispatch dimensions based on K (#3599).
    ///
    /// Delegates to `select_gemm_dispatch()` — single source of truth for the
    /// tiled-vs-small-K decision. Maps the pure dispatch result to a pipeline ref.
    ///
    /// Returns `(pipeline, wg_x, wg_y)`.
    fn gemm_pipeline_and_dispatch(
        &self,
        gemm_params: &crate::wgpu_device::params::GemmParams,
    ) -> (&wgpu::ComputePipeline, u32, u32) {
        let dispatch = select_gemm_dispatch(gemm_params.m, gemm_params.k, gemm_params.n);
        let pipeline = if dispatch.use_small_k {
            &self.gemm_f32_small_k_pipeline
        } else {
            &self.gemm_f32_pipeline
        };
        (pipeline, dispatch.wg_x, dispatch.wg_y)
    }
}
