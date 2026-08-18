// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dedicated working buffers and pre-built bind groups for GPU CROWN backward
//! plans (#3397 Step 4).
//!
//! By owning the working buffers inside the cached plan and pre-building all
//! bind groups at plan creation time, we eliminate ~20-30
//! `device.create_bind_group()` calls and the buffer pool lock per CROWN
//! backward invocation.

use ny_core::{GpuCrownLayer, NyError, Result};

use super::super::WgpuDevice;
use super::crown_backward_types::{DispatchStep, MAX_PARAMS_SIZE};

/// Pre-built bind groups for one dispatch step.
///
/// Created once at plan build time and reused across all backward pass
/// invocations. Buffer contents change via `queue.write_buffer` /
/// `copy_buffer_to_buffer`, which does not invalidate bind groups.
pub(super) enum StepBindGroups {
    Activation {
        bind_group: wgpu::BindGroup,
    },
    MaxPool2d {
        bind_group: wgpu::BindGroup,
    },
    BiasAccumulate {
        bind_group: wgpu::BindGroup,
    },
    GemmLinear {
        lower: wgpu::BindGroup,
        upper: wgpu::BindGroup,
    },
    GemmConv {
        lower: wgpu::BindGroup,
        upper: wgpu::BindGroup,
    },
    ConvReshape {
        lower: wgpu::BindGroup,
        upper: wgpu::BindGroup,
    },
    ConvCol2im {
        lower: wgpu::BindGroup,
        upper: wgpu::BindGroup,
    },
    Concretize {
        bind_group: wgpu::BindGroup,
    },
}

/// Dedicated working buffers owned by a cached CROWN plan.
///
/// Unlike the shared `BufferPool` path, these buffers are exact-sized for the
/// specific plan (no 1.2× growth factor) and stable across invocations, so
/// bind groups built against them remain valid indefinitely.
pub(super) struct CrownWorkingBuffers {
    pub(super) params_buf: wgpu::Buffer,
    pub(super) a_lower_0: wgpu::Buffer,
    pub(super) a_upper_0: wgpu::Buffer,
    pub(super) a_lower_1: wgpu::Buffer,
    pub(super) a_upper_1: wgpu::Buffer,
    pub(super) bias_lower: wgpu::Buffer,
    pub(super) bias_upper: wgpu::Buffer,
    pub(super) slopes_buf: wgpu::Buffer,
    pub(super) weight_buf: wgpu::Buffer,
    pub(super) layer_bias_buf: wgpu::Buffer,
    pub(super) conv_reshaped_lower: Option<wgpu::Buffer>,
    pub(super) conv_reshaped_upper: Option<wgpu::Buffer>,
    pub(super) conv_gemm_lower: Option<wgpu::Buffer>,
    pub(super) conv_gemm_upper: Option<wgpu::Buffer>,
    pub(super) inp_lower_buf: wgpu::Buffer,
    pub(super) inp_upper_buf: wgpu::Buffer,
    pub(super) out_lower: wgpu::Buffer,
    pub(super) out_upper: wgpu::Buffer,
    pub(super) readback_lower: wgpu::Buffer,
    pub(super) readback_upper: wgpu::Buffer,
}

impl CrownWorkingBuffers {
    /// Exact logical bytes owned by this cached plan's working buffers.
    pub(super) fn retained_device_bytes(&self) -> Result<usize> {
        fn add(total: &mut usize, buffer: &wgpu::Buffer, label: &str) -> Result<()> {
            let bytes = usize::try_from(buffer.size()).map_err(|_| {
                NyError::InternalError(format!("CROWN plan buffer `{label}` does not fit in usize"))
            })?;
            *total = total.checked_add(bytes).ok_or_else(|| {
                NyError::InternalError("CROWN plan working-buffer byte count overflow".into())
            })?;
            Ok(())
        }

        let mut total = 0usize;
        for (label, buffer) in [
            ("params", &self.params_buf),
            ("a_lower_0", &self.a_lower_0),
            ("a_upper_0", &self.a_upper_0),
            ("a_lower_1", &self.a_lower_1),
            ("a_upper_1", &self.a_upper_1),
            ("bias_lower", &self.bias_lower),
            ("bias_upper", &self.bias_upper),
            ("slopes", &self.slopes_buf),
            ("weight", &self.weight_buf),
            ("layer_bias", &self.layer_bias_buf),
            ("input_lower", &self.inp_lower_buf),
            ("input_upper", &self.inp_upper_buf),
            ("output_lower", &self.out_lower),
            ("output_upper", &self.out_upper),
            ("readback_lower", &self.readback_lower),
            ("readback_upper", &self.readback_upper),
        ] {
            add(&mut total, buffer, label)?;
        }
        for (label, buffer) in [
            ("conv_reshaped_lower", self.conv_reshaped_lower.as_ref()),
            ("conv_reshaped_upper", self.conv_reshaped_upper.as_ref()),
            ("conv_gemm_lower", self.conv_gemm_lower.as_ref()),
            ("conv_gemm_upper", self.conv_gemm_upper.as_ref()),
        ] {
            if let Some(buffer) = buffer {
                add(&mut total, buffer, label)?;
            }
        }
        Ok(total)
    }

    /// Allocate all working buffers for the given plan dimensions.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &wgpu::Device,
        num_specs: usize,
        max_dim: usize,
        max_weight_elems: usize,
        max_bias_elems: usize,
        max_activation_elems: usize,
        max_conv_reshaped: usize,
        max_conv_gemm_out: usize,
        input_dim: usize,
    ) -> Self {
        let storage_copy = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let a_elems = num_specs * max_dim;

        Self {
            params_buf: create_buf(
                device,
                "crown_plan_params",
                MAX_PARAMS_SIZE,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            a_lower_0: create_f32_buf(device, "crown_plan_a_l0", a_elems, storage_copy),
            a_upper_0: create_f32_buf(device, "crown_plan_a_u0", a_elems, storage_copy),
            a_lower_1: create_f32_buf(device, "crown_plan_a_l1", a_elems, storage_copy),
            a_upper_1: create_f32_buf(device, "crown_plan_a_u1", a_elems, storage_copy),
            bias_lower: create_f32_buf(device, "crown_plan_bias_l", num_specs, storage_copy),
            bias_upper: create_f32_buf(device, "crown_plan_bias_u", num_specs, storage_copy),
            slopes_buf: create_f32_buf(
                device,
                "crown_plan_slopes",
                max_activation_elems.max(1),
                storage_dst,
            ),
            weight_buf: create_f32_buf(
                device,
                "crown_plan_weight",
                max_weight_elems.max(1),
                storage_dst,
            ),
            layer_bias_buf: create_f32_buf(
                device,
                "crown_plan_layer_bias",
                max_bias_elems.max(1),
                storage_dst,
            ),
            conv_reshaped_lower: (max_conv_reshaped > 0).then(|| {
                create_f32_buf(device, "crown_plan_conv_rl", max_conv_reshaped, storage_dst)
            }),
            conv_reshaped_upper: (max_conv_reshaped > 0).then(|| {
                create_f32_buf(device, "crown_plan_conv_ru", max_conv_reshaped, storage_dst)
            }),
            conv_gemm_lower: (max_conv_gemm_out > 0).then(|| {
                create_f32_buf(device, "crown_plan_conv_gl", max_conv_gemm_out, storage_dst)
            }),
            conv_gemm_upper: (max_conv_gemm_out > 0).then(|| {
                create_f32_buf(device, "crown_plan_conv_gu", max_conv_gemm_out, storage_dst)
            }),
            inp_lower_buf: create_f32_buf(device, "crown_plan_inp_l", input_dim, storage_dst),
            inp_upper_buf: create_f32_buf(device, "crown_plan_inp_u", input_dim, storage_dst),
            out_lower: create_f32_buf(device, "crown_plan_out_l", num_specs, storage_copy),
            out_upper: create_f32_buf(device, "crown_plan_out_u", num_specs, storage_copy),
            readback_lower: create_f32_buf(
                device,
                "crown_plan_rb_l",
                num_specs,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            readback_upper: create_f32_buf(
                device,
                "crown_plan_rb_u",
                num_specs,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }
}

/// Build pre-built bind groups for every dispatch step.
///
/// The bind groups reference the plan's dedicated working buffers, which are
/// stable across invocations (never resized). Data is loaded into these buffers
/// via `queue.write_buffer` and `copy_buffer_to_buffer` staging copies.
pub(super) fn build_step_bind_groups(
    device: &WgpuDevice,
    steps: &[DispatchStep],
    working: &CrownWorkingBuffers,
    layers: &[GpuCrownLayer],
) -> Result<Vec<StepBindGroups>> {
    let has_conv = layers
        .iter()
        .any(|l| matches!(l, GpuCrownLayer::Conv2d { .. }));
    let wgpu_dev = device.device();

    /// Extract a Conv2d buffer or return `InternalError`.
    fn conv_buf<'a>(buf: &'a Option<wgpu::Buffer>, name: &str) -> Result<&'a wgpu::Buffer> {
        buf.as_ref().ok_or_else(|| {
            NyError::InternalError(format!(
                "invariant: conv {name} allocated when Conv2d layers present"
            ))
        })
    }

    steps
        .iter()
        .map(|step| match step {
            DispatchStep::ActivationBackward { ping, .. } => {
                let (src_l, src_u) = a_ping(working, *ping);
                let (dst_l, dst_u) = a_ping(working, 1 - ping);
                Ok(StepBindGroups::Activation {
                    bind_group: wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("crown_act_bg_cached"),
                        layout: &device.crown_activation_backward_bind_group_layout,
                        entries: &[
                            entry(0, &working.params_buf),
                            entry(1, src_l),
                            entry(2, src_u),
                            entry(3, &working.slopes_buf),
                            entry(4, dst_l),
                            entry(5, dst_u),
                            entry(6, &working.bias_lower),
                            entry(7, &working.bias_upper),
                        ],
                    }),
                })
            }
            DispatchStep::MaxPool2dBackward { ping, .. } => {
                let (src_l, src_u) = a_ping(working, *ping);
                let (dst_l, dst_u) = a_ping(working, 1 - ping);
                Ok(StepBindGroups::MaxPool2d {
                    bind_group: wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("crown_maxpool2d_bg_cached"),
                        layout: &device.crown_maxpool2d_backward_bind_group_layout,
                        entries: &[
                            entry(0, &working.params_buf),
                            entry(1, src_l),
                            entry(2, src_u),
                            entry(3, &working.weight_buf),
                            entry(4, &working.slopes_buf),
                            entry(5, dst_l),
                            entry(6, dst_u),
                            entry(7, &working.bias_lower),
                            entry(8, &working.bias_upper),
                        ],
                    }),
                })
            }
            DispatchStep::BiasAccumulate { ping, .. } => {
                let (src_l, src_u) = a_ping(working, *ping);
                Ok(StepBindGroups::BiasAccumulate {
                    bind_group: wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("crown_bias_bg_cached"),
                        layout: &device.crown_bias_accumulate_bind_group_layout,
                        entries: &[
                            entry(0, &working.params_buf),
                            entry(1, src_l),
                            entry(2, src_u),
                            entry(3, &working.layer_bias_buf),
                            entry(4, &working.bias_lower),
                            entry(5, &working.bias_upper),
                        ],
                    }),
                })
            }
            DispatchStep::GemmCrownLinear { ping, .. } => {
                let (src_l, src_u) = a_ping(working, *ping);
                let (dst_l, dst_u) = a_ping(working, 1 - ping);
                Ok(StepBindGroups::GemmLinear {
                    lower: gemm_bind_group(
                        wgpu_dev,
                        &device.gemm_f32_bind_group_layout,
                        "crown_gemm_lower_cached",
                        &working.params_buf,
                        src_l,
                        &working.weight_buf,
                        dst_l,
                    ),
                    upper: gemm_bind_group(
                        wgpu_dev,
                        &device.gemm_f32_bind_group_layout,
                        "crown_gemm_upper_cached",
                        &working.params_buf,
                        src_u,
                        &working.weight_buf,
                        dst_u,
                    ),
                })
            }
            DispatchStep::GemmCrownConv { .. } => {
                if !has_conv {
                    return Err(NyError::InternalError(
                        "GemmCrownConv step requires conv buffers".into(),
                    ));
                }
                let rl = conv_buf(&working.conv_reshaped_lower, "reshaped_lower")?;
                let ru = conv_buf(&working.conv_reshaped_upper, "reshaped_upper")?;
                let gl = conv_buf(&working.conv_gemm_lower, "gemm_lower")?;
                let gu = conv_buf(&working.conv_gemm_upper, "gemm_upper")?;
                Ok(StepBindGroups::GemmConv {
                    lower: gemm_bind_group(
                        wgpu_dev,
                        &device.gemm_f32_bind_group_layout,
                        "crown_conv_gemm_lower_cached",
                        &working.params_buf,
                        rl,
                        &working.weight_buf,
                        gl,
                    ),
                    upper: gemm_bind_group(
                        wgpu_dev,
                        &device.gemm_f32_bind_group_layout,
                        "crown_conv_gemm_upper_cached",
                        &working.params_buf,
                        ru,
                        &working.weight_buf,
                        gu,
                    ),
                })
            }
            DispatchStep::ConvReshapeLowerUpper { ping, .. } => {
                let (src_l, src_u) = a_ping(working, *ping);
                let rl = conv_buf(&working.conv_reshaped_lower, "reshaped_lower")?;
                let ru = conv_buf(&working.conv_reshaped_upper, "reshaped_upper")?;
                Ok(StepBindGroups::ConvReshape {
                    lower: wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("conv_reshape_lower_cached"),
                        layout: &device.conv_reshape_bind_group_layout,
                        entries: &[entry(0, &working.params_buf), entry(1, src_l), entry(2, rl)],
                    }),
                    upper: wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("conv_reshape_upper_cached"),
                        layout: &device.conv_reshape_bind_group_layout,
                        entries: &[entry(0, &working.params_buf), entry(1, src_u), entry(2, ru)],
                    }),
                })
            }
            DispatchStep::ConvCol2imLowerUpper { ping, .. } => {
                let (dst_l, dst_u) = a_ping(working, 1 - ping);
                let gl = conv_buf(&working.conv_gemm_lower, "gemm_lower")?;
                let gu = conv_buf(&working.conv_gemm_upper, "gemm_upper")?;
                Ok(StepBindGroups::ConvCol2im {
                    lower: wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("conv_col2im_lower_cached"),
                        layout: &device.conv_col2im_bind_group_layout,
                        entries: &[entry(0, &working.params_buf), entry(1, gl), entry(2, dst_l)],
                    }),
                    upper: wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("conv_col2im_upper_cached"),
                        layout: &device.conv_col2im_bind_group_layout,
                        entries: &[entry(0, &working.params_buf), entry(1, gu), entry(2, dst_u)],
                    }),
                })
            }
            DispatchStep::Concretize { ping, .. } => {
                let (src_l, src_u) = a_ping(working, *ping);
                Ok(StepBindGroups::Concretize {
                    bind_group: wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("crown_conc_bg_cached"),
                        layout: &device.crown_concretize_bind_group_layout,
                        entries: &[
                            entry(0, &working.params_buf),
                            entry(1, src_l),
                            entry(2, src_u),
                            entry(3, &working.inp_lower_buf),
                            entry(4, &working.inp_upper_buf),
                            entry(5, &working.bias_lower),
                            entry(6, &working.bias_upper),
                            entry(7, &working.out_lower),
                            entry(8, &working.out_upper),
                        ],
                    }),
                })
            }
        })
        .collect()
}

// --- Helpers ---

fn create_f32_buf(
    device: &wgpu::Device,
    label: &str,
    count: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = (count * size_of::<f32>()) as u64;
    create_buf(device, label, bytes.max(4), usage)
}

fn create_buf(
    device: &wgpu::Device,
    label: &str,
    size_bytes: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes.max(4),
        usage,
        mapped_at_creation: false,
    })
}

fn a_ping(w: &CrownWorkingBuffers, ping: usize) -> (&wgpu::Buffer, &wgpu::Buffer) {
    if ping == 0 {
        (&w.a_lower_0, &w.a_upper_0)
    } else {
        (&w.a_lower_1, &w.a_upper_1)
    }
}

fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn gemm_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    params: &wgpu::Buffer,
    src: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    dst: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            entry(0, params),
            entry(1, src),
            entry(2, weight),
            entry(3, dst),
        ],
    })
}
