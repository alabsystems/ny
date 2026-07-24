// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cached static staging plans for GPU CROWN backward (#3397).

use std::sync::Arc;

use ny_core::{GpuCrownLayer, NyError, Result};
use wgpu::util::DeviceExt;

use super::super::WgpuDevice;
use super::crown_backward_types::{
    build_dispatch_plan, conv2d_buffer_sizes, layer_input_dim, DispatchStep,
};
use super::crown_plan_key::crown_plan_key;
use super::gpu_checked_u32;

#[derive(Debug, Clone, Copy)]
pub(super) enum DynamicUpload {
    Activation {
        offset_bytes: u64,
        num_neurons: usize,
    },
    MaxPool2d {
        routing_offset_bytes: u64,
        bounds_offset_bytes: u64,
        output_dim: usize,
    },
}

pub(crate) struct PreparedCrownPlan {
    pub(super) steps: Vec<DispatchStep>,
    pub(super) staging_buf: wgpu::Buffer,
    pub(super) dynamic_uploads: Vec<DynamicUpload>,
    /// Dedicated working buffers owned by this plan (#3397 Step 4).
    pub(super) working: super::crown_plan_working::CrownWorkingBuffers,
    /// Pre-built bind groups for each dispatch step (#3397 Step 4).
    pub(super) step_bind_groups: Vec<super::crown_plan_working::StepBindGroups>,
    /// Keep-alive clones of every static weight `Arc` this plan was built from
    /// (#perf-plan-cache CORRECTNESS FIX). The cache key hashes weights by
    /// **pointer identity** (`crown_plan_key::hash_arc_identity`), which is only
    /// sound while those allocations are alive: once a model's weight `Arc`s are
    /// dropped, the allocator can recycle the same address for a *different*
    /// model's weights, colliding with the stale key and silently serving this
    /// plan — with the OLD weights baked into `staging_buf` — for the new model
    /// (observed: `test_crown_backward_dual_alpha_crossing` returned bounds from
    /// the previous test's weights, GPU=-3.3 vs CPU=-0.325). Holding the `Arc`s
    /// here makes address recycling impossible while the key is live, so a
    /// pointer-identity hit is always a true content hit.
    /// `clear_crown_plan_cache()` (called between models by the VNN-COMP runner)
    /// releases them.
    #[allow(dead_code)] // held for its Drop/lifetime semantics only
    static_weight_arcs: Vec<Arc<[f32]>>,
}

impl WgpuDevice {
    pub(super) fn get_or_prepare_crown_plan(
        &self,
        layers: &[GpuCrownLayer],
        num_specs: usize,
        first_dim: usize,
    ) -> Result<Arc<PreparedCrownPlan>> {
        let key = crown_plan_key(layers, num_specs, first_dim);
        {
            let cache = self.crown_plan_cache.lock().map_err(|err| {
                NyError::InternalError(format!("crown plan cache lock poisoned: {err}"))
            })?;
            if let Some(plan) = cache.get(&key) {
                return Ok(plan.clone());
            }
        }

        let plan = Arc::new(PreparedCrownPlan::build(
            self, layers, num_specs, first_dim,
        )?);
        let mut cache = self.crown_plan_cache.lock().map_err(|err| {
            NyError::InternalError(format!("crown plan cache lock poisoned: {err}"))
        })?;
        Ok(cache.entry(key).or_insert_with(|| plan.clone()).clone())
    }

    /// Clear the CROWN plan cache, freeing cached staging buffers (#3515).
    ///
    /// Call between models in the VNN-COMP runner alongside
    /// `BufferPool::release_crown_buffers()` to prevent cross-model memory
    /// accumulation.
    // Used by gpu-tests feature gate (test_crown_resource_release_between_models)
    // and Phase 2 VNN-COMP runner integration.
    #[cfg_attr(not(feature = "gpu-tests"), allow(dead_code))]
    pub(crate) fn clear_crown_plan_cache(&self) -> Result<()> {
        let mut cache = self.crown_plan_cache.lock().map_err(|err| {
            NyError::InternalError(format!("crown plan cache lock poisoned: {err}"))
        })?;
        cache.clear();
        Ok(())
    }

    pub(super) fn refresh_crown_plan_dynamic_layers(
        &self,
        plan: &PreparedCrownPlan,
        layers: &[GpuCrownLayer],
    ) -> Result<()> {
        let dynamic_layers: Vec<_> = layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer,
                    GpuCrownLayer::Activation { .. }
                        | GpuCrownLayer::ActivationReluDualAlpha { .. }
                        | GpuCrownLayer::MaxPool2d { .. }
                )
            })
            .collect();

        if dynamic_layers.len() != plan.dynamic_uploads.len() {
            return Err(NyError::InternalError(format!(
                "crown plan dynamic mismatch: plan has {} uploads, runtime has {} dynamic layers",
                plan.dynamic_uploads.len(),
                dynamic_layers.len()
            )));
        }

        for (upload, layer) in plan.dynamic_uploads.iter().zip(dynamic_layers) {
            match (upload, layer) {
                (
                    DynamicUpload::Activation {
                        offset_bytes,
                        num_neurons,
                    },
                    GpuCrownLayer::Activation {
                        lower_slope,
                        upper_slope,
                        lower_intercept,
                        upper_intercept,
                        ..
                    },
                ) => {
                    if lower_slope.len() != *num_neurons
                        || upper_slope.len() != *num_neurons
                        || lower_intercept.len() != *num_neurons
                        || upper_intercept.len() != *num_neurons
                    {
                        return Err(NyError::InternalError(format!(
                            "crown plan activation size mismatch: expected {} neurons, got lower={} upper={} lower_b={} upper_b={}",
                            num_neurons,
                            lower_slope.len(),
                            upper_slope.len(),
                            lower_intercept.len(),
                            upper_intercept.len()
                        )));
                    }

                    let part_bytes = (*num_neurons * size_of::<f32>()) as u64;
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *offset_bytes,
                        bytemuck::cast_slice(lower_slope),
                    );
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *offset_bytes + part_bytes,
                        bytemuck::cast_slice(upper_slope),
                    );
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *offset_bytes + part_bytes * 2,
                        bytemuck::cast_slice(lower_intercept),
                    );
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *offset_bytes + part_bytes * 3,
                        bytemuck::cast_slice(upper_intercept),
                    );
                }
                (
                    DynamicUpload::MaxPool2d {
                        routing_offset_bytes,
                        bounds_offset_bytes,
                        output_dim,
                    },
                    GpuCrownLayer::MaxPool2d {
                        routing,
                        ibp_lower,
                        ibp_upper,
                        ..
                    },
                ) => {
                    if routing.len() != *output_dim
                        || ibp_lower.len() != *output_dim
                        || ibp_upper.len() != *output_dim
                    {
                        return Err(NyError::InternalError(format!(
                            "crown plan maxpool size mismatch: expected {} outputs, got routing={} lower={} upper={}",
                            output_dim,
                            routing.len(),
                            ibp_lower.len(),
                            ibp_upper.len()
                        )));
                    }

                    let part_bytes = (*output_dim * size_of::<f32>()) as u64;
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *routing_offset_bytes,
                        bytemuck::cast_slice(routing),
                    );
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *bounds_offset_bytes,
                        bytemuck::cast_slice(ibp_lower),
                    );
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *bounds_offset_bytes + part_bytes,
                        bytemuck::cast_slice(ibp_upper),
                    );
                }
                (
                    DynamicUpload::Activation {
                        offset_bytes,
                        num_neurons,
                    },
                    GpuCrownLayer::ActivationReluDualAlpha {
                        lower_pos_slope,
                        cross_slope,
                        upper_neg_slope,
                        cross_intercept,
                        ..
                    },
                ) => {
                    if lower_pos_slope.len() != *num_neurons
                        || cross_slope.len() != *num_neurons
                        || upper_neg_slope.len() != *num_neurons
                        || cross_intercept.len() != *num_neurons
                    {
                        return Err(NyError::InternalError(format!(
                            "crown plan dual-alpha activation size mismatch: expected {} neurons, got lps={} cs={} uns={} ci={}",
                            num_neurons,
                            lower_pos_slope.len(),
                            cross_slope.len(),
                            upper_neg_slope.len(),
                            cross_intercept.len()
                        )));
                    }

                    let part_bytes = (*num_neurons * size_of::<f32>()) as u64;
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *offset_bytes,
                        bytemuck::cast_slice(lower_pos_slope),
                    );
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *offset_bytes + part_bytes,
                        bytemuck::cast_slice(cross_slope),
                    );
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *offset_bytes + part_bytes * 2,
                        bytemuck::cast_slice(upper_neg_slope),
                    );
                    self.queue.write_buffer(
                        &plan.staging_buf,
                        *offset_bytes + part_bytes * 3,
                        bytemuck::cast_slice(cross_intercept),
                    );
                }
                _ => {
                    return Err(NyError::InternalError(
                        "crown plan dynamic upload/runtime layer mismatch".into(),
                    ));
                }
            }
        }

        Ok(())
    }
}

impl PreparedCrownPlan {
    fn build(
        device: &WgpuDevice,
        layers: &[GpuCrownLayer],
        num_specs: usize,
        first_dim: usize,
    ) -> Result<Self> {
        let num_specs_u32 = gpu_checked_u32(num_specs, "num_specs")?;
        let (steps, staging, _final_ping, final_dim) =
            build_dispatch_plan(layers, num_specs, num_specs_u32, first_dim)?;

        let staging_buf = device
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("crown_staging_plan"),
                contents: staging.as_bytes(),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            });

        let max_dim = layers
            .iter()
            .map(|layer| layer_input_dim(layer).unwrap_or(0))
            .chain(std::iter::once(first_dim))
            .max()
            .unwrap_or(first_dim);
        let max_weight_elems = layers
            .iter()
            .filter_map(|layer| match layer {
                GpuCrownLayer::Linear {
                    out_features,
                    in_features,
                    ..
                } => Some(out_features * in_features),
                GpuCrownLayer::MaxPool2d { output_dim, .. } => Some(*output_dim),
                GpuCrownLayer::Conv2d {
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    ..
                } => Some(out_channels * in_channels * kernel_h * kernel_w),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let max_bias_elems = layers
            .iter()
            .filter_map(|layer| match layer {
                GpuCrownLayer::Linear {
                    out_features, bias, ..
                } => bias.as_ref().map(|_| *out_features),
                GpuCrownLayer::Conv2d {
                    bias_expanded,
                    out_channels,
                    out_h,
                    out_w,
                    ..
                } => bias_expanded.as_ref().map(|_| out_channels * out_h * out_w),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let max_activation_elems = layers
            .iter()
            .filter_map(|layer| match layer {
                GpuCrownLayer::Activation { num_neurons, .. }
                | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => {
                    Some(*num_neurons * 4)
                }
                GpuCrownLayer::MaxPool2d { output_dim, .. } => Some(*output_dim * 2),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let (max_conv_reshaped, max_conv_gemm_out) = conv2d_buffer_sizes(layers, num_specs);
        let dynamic_uploads = steps
            .iter()
            .filter_map(|step| match step {
                DispatchStep::ActivationBackward {
                    slopes_off,
                    slopes_size,
                    ..
                } => Some(DynamicUpload::Activation {
                    offset_bytes: *slopes_off,
                    num_neurons: (*slopes_size as usize) / (4 * size_of::<f32>()),
                }),
                DispatchStep::MaxPool2dBackward {
                    routing_off,
                    bounds_off,
                    bounds_size,
                    ..
                } => Some(DynamicUpload::MaxPool2d {
                    routing_offset_bytes: *routing_off,
                    bounds_offset_bytes: *bounds_off,
                    output_dim: (*bounds_size as usize) / (2 * size_of::<f32>()),
                }),
                _ => None,
            })
            .collect();

        // Step 4 (#3397): Create dedicated working buffers and pre-build
        // all bind groups at plan creation time, eliminating per-call
        // pool allocation and bind group creation overhead.
        let input_dim = final_dim;
        let working = super::crown_plan_working::CrownWorkingBuffers::new(
            device.device(),
            num_specs,
            max_dim,
            max_weight_elems,
            max_bias_elems,
            max_activation_elems,
            max_conv_reshaped,
            max_conv_gemm_out,
            input_dim,
        );
        let step_bind_groups =
            super::crown_plan_working::build_step_bind_groups(device, &steps, &working, layers)?;

        Ok(Self {
            steps,
            staging_buf,
            dynamic_uploads,
            working,
            step_bind_groups,
            static_weight_arcs: collect_static_weight_arcs(layers),
        })
    }
}

/// Clone every static weight `Arc` referenced by the plan-cache key's
/// pointer-identity hash (`Linear::weight`/`bias`, `Conv2d::weight_col`/
/// `bias_expanded`) so the cached plan keeps those allocations alive. See
/// `PreparedCrownPlan::static_weight_arcs`.
fn collect_static_weight_arcs(layers: &[GpuCrownLayer]) -> Vec<Arc<[f32]>> {
    let mut arcs = Vec::new();
    for layer in layers {
        match layer {
            GpuCrownLayer::Linear { weight, bias, .. } => {
                arcs.push(weight.clone());
                if let Some(bias) = bias {
                    arcs.push(bias.clone());
                }
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                ..
            } => {
                arcs.push(weight_col.clone());
                if let Some(bias_expanded) = bias_expanded {
                    arcs.push(bias_expanded.clone());
                }
            }
            GpuCrownLayer::Activation { .. }
            | GpuCrownLayer::ActivationReluDualAlpha { .. }
            | GpuCrownLayer::MaxPool2d { .. } => {}
        }
    }
    arcs
}
