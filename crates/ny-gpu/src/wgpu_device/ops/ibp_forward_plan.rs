// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cached-plan support for GPU-resident IBP forward (#4268).

use ny_core::{
    checked_shape_product, nan_propagating_max_zero, nan_propagating_min_zero, GpuIbpLayer,
    GpuIbpModelPlan, GpuIbpResult, NyError, Result,
};

use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;
use super::{gpu_checked_u32, sanitize_readback};
use crate::wgpu_device::params::{Conv2dIbpParams, LinearIbpParams, ReluIbpParams};

enum PreparedIbpStep {
    Linear {
        bind_group: wgpu::BindGroup,
        workgroup_count: u32,
    },
    Conv2d {
        bind_group: wgpu::BindGroup,
        workgroup_count: u32,
    },
    ReLU {
        bind_group: wgpu::BindGroup,
        workgroup_count: u32,
    },
}

pub(super) struct WgpuIbpModelPlan {
    device: wgpu::Device,
    queue: wgpu::Queue,
    linear_pipeline: wgpu::ComputePipeline,
    conv2d_pipeline: wgpu::ComputePipeline,
    relu_pipeline: wgpu::ComputePipeline,
    input_shape: Vec<usize>,
    input_elements: usize,
    output_shape: Vec<usize>,
    output_elements: usize,
    final_use_b: bool,
    buf_lower_a: wgpu::Buffer,
    buf_upper_a: wgpu::Buffer,
    buf_lower_b: wgpu::Buffer,
    buf_upper_b: wgpu::Buffer,
    staging_lower: wgpu::Buffer,
    staging_upper: wgpu::Buffer,
    steps: Vec<PreparedIbpStep>,
}

pub(super) fn shape_product_or_err(shape: &[usize], context: &str) -> Result<usize> {
    checked_shape_product(shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ibp_forward_gpu: {context} shape product overflows usize: {shape:?}"
        ))
    })
}

fn next_conv2d_dim(
    current_dim: usize,
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
) -> Result<usize> {
    if groups != 1 {
        return Err(NyError::UnsupportedConfiguration(
            "resident Conv2d IBP currently supports groups=1 only".into(),
        ));
    }
    let input_plane = checked_shape_product(&[in_channels, input_h, input_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ibp_forward_gpu: Conv2d input dims overflow for {in_channels}x{input_h}x{input_w}"
        ))
    })?;
    if input_plane == 0 || !current_dim.is_multiple_of(input_plane) {
        return Err(NyError::shape_mismatch(
            vec![input_plane],
            vec![current_dim],
        ));
    }
    let batch_size = current_dim / input_plane;
    let (out_h, out_w) = conv2d_output_size(
        input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
    )?;
    checked_shape_product(&[batch_size, out_channels, out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ibp_forward_gpu: Conv2d output overflow for batch={batch_size}, \
             out_channels={out_channels}, out_h={out_h}, out_w={out_w}"
        ))
    })
}

fn next_resident_dim(current_dim: usize, layer: &GpuIbpLayer) -> Result<usize> {
    match layer {
        GpuIbpLayer::Linear {
            out_features,
            in_features,
            ..
        } => {
            if *in_features == 0 || !current_dim.is_multiple_of(*in_features) {
                return Err(NyError::shape_mismatch(
                    vec![*in_features],
                    vec![current_dim],
                ));
            }
            (current_dim / *in_features)
                .checked_mul(*out_features)
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "ibp_forward_gpu: linear output overflow for current_dim={current_dim}, \
                         in_features={in_features}, out_features={out_features}"
                    ))
                })
        }
        GpuIbpLayer::ReLU { num_elements } => {
            if *num_elements != current_dim {
                return Err(NyError::shape_mismatch(
                    vec![current_dim],
                    vec![*num_elements],
                ));
            }
            Ok(current_dim)
        }
        GpuIbpLayer::Conv2d {
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
            ..
        } => next_conv2d_dim(
            current_dim,
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
        ),
        GpuIbpLayer::View { output_shape } => {
            let view_dim = shape_product_or_err(output_shape, "view output")?;
            if view_dim != current_dim {
                return Err(NyError::shape_mismatch(vec![current_dim], vec![view_dim]));
            }
            Ok(current_dim)
        }
    }
}

#[allow(clippy::too_many_arguments)] // Conv2d spatial params are naturally 8 scalars
fn conv2d_output_size(
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<(usize, usize)> {
    let padded_h = input_h
        .checked_add(
            pad_h
                .checked_mul(2)
                .ok_or_else(|| NyError::InvalidSpec("Conv2d padded height overflow".into()))?,
        )
        .ok_or_else(|| NyError::InvalidSpec("Conv2d padded height overflow".into()))?;
    let padded_w = input_w
        .checked_add(
            pad_w
                .checked_mul(2)
                .ok_or_else(|| NyError::InvalidSpec("Conv2d padded width overflow".into()))?,
        )
        .ok_or_else(|| NyError::InvalidSpec("Conv2d padded width overflow".into()))?;
    if padded_h < kernel_h || padded_w < kernel_w {
        return Err(NyError::InvalidSpec(format!(
            "Conv2d kernel larger than padded input: input=({input_h},{input_w}), padding=({pad_h},{pad_w}), kernel=({kernel_h},{kernel_w})"
        )));
    }
    Ok((
        (padded_h - kernel_h) / stride_h + 1,
        (padded_w - kernel_w) / stride_w + 1,
    ))
}

pub(super) fn max_resident_buffer_elems(
    layers: &[GpuIbpLayer],
    input_elements: usize,
) -> Result<usize> {
    let mut current_dim = input_elements;
    let mut max_dim = input_elements;
    for layer in layers {
        current_dim = next_resident_dim(current_dim, layer)?;
        max_dim = max_dim.max(current_dim);
    }
    Ok(max_dim)
}

impl WgpuDevice {
    pub(super) fn prepare_model_plan_internal(
        &self,
        layers: &[GpuIbpLayer],
        input_shape: &[usize],
    ) -> Result<WgpuIbpModelPlan> {
        let input_elements = shape_product_or_err(input_shape, "input")?;
        let max_dim = max_resident_buffer_elems(layers, input_elements)?;
        let f32_size = size_of::<f32>() as u64;
        let max_buf_bytes = (max_dim as u64) * f32_size;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let buf_lower_a = create_buffer(&self.device, "ibp_fwd_lower_a", max_buf_bytes, usage_rw);
        let buf_upper_a = create_buffer(&self.device, "ibp_fwd_upper_a", max_buf_bytes, usage_rw);
        let buf_lower_b = create_buffer(&self.device, "ibp_fwd_lower_b", max_buf_bytes, usage_rw);
        let buf_upper_b = create_buffer(&self.device, "ibp_fwd_upper_b", max_buf_bytes, usage_rw);

        let mut steps = Vec::with_capacity(layers.len());
        let mut use_b = false;
        let mut cur_dim = input_elements;
        let mut cur_shape = input_shape.to_vec();

        for layer in layers {
            match layer {
                GpuIbpLayer::Linear {
                    weight,
                    bias,
                    out_features,
                    in_features,
                } => {
                    let expected_weight_len =
                        in_features.checked_mul(*out_features).ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "ibp_forward_gpu: linear weight shape overflow for in_features={}, \
                                 out_features={}",
                                in_features, out_features
                            ))
                        })?;
                    if weight.len() != expected_weight_len {
                        return Err(NyError::shape_mismatch(
                            vec![expected_weight_len],
                            vec![weight.len()],
                        ));
                    }
                    if let Some(bias) = bias {
                        if bias.len() != *out_features {
                            return Err(NyError::shape_mismatch(
                                vec![*out_features],
                                vec![bias.len()],
                            ));
                        }
                    }

                    let (src_lower, src_upper, dst_lower, dst_upper) = if use_b {
                        (&buf_lower_b, &buf_upper_b, &buf_lower_a, &buf_upper_a)
                    } else {
                        (&buf_lower_a, &buf_upper_a, &buf_lower_b, &buf_upper_b)
                    };

                    let weight_pos: Vec<f32> = weight
                        .iter()
                        .map(|&w| nan_propagating_max_zero(w))
                        .collect();
                    let weight_neg: Vec<f32> = weight
                        .iter()
                        .map(|&w| nan_propagating_min_zero(w))
                        .collect();
                    let bias_data: Vec<f32> = match bias {
                        Some(bias) => bias.to_vec(),
                        None => vec![0.0; *out_features],
                    };

                    if *in_features == 0 || cur_dim % *in_features != 0 {
                        return Err(NyError::shape_mismatch(vec![*in_features], vec![cur_dim]));
                    }
                    let batch_size = cur_dim / *in_features;
                    let params = LinearIbpParams {
                        batch_size: gpu_checked_u32(batch_size, "ibp_fwd linear batch")?,
                        in_features: gpu_checked_u32(*in_features, "ibp_fwd linear in")?,
                        out_features: gpu_checked_u32(*out_features, "ibp_fwd linear out")?,
                        _padding: 0,
                    };

                    let params_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_linear_params",
                        size_of::<LinearIbpParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let weight_size = expected_weight_len as u64 * f32_size;
                    let wp_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_wp",
                        weight_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&wp_buf, 0, bytemuck::cast_slice(&weight_pos));

                    let wn_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_wn",
                        weight_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&wn_buf, 0, bytemuck::cast_slice(&weight_neg));

                    let bias_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_bias",
                        (*out_features as u64) * f32_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&bias_buf, 0, bytemuck::cast_slice(&bias_data));

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ibp_fwd_linear_bg"),
                        layout: &self.linear_ibp_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: src_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: src_upper.as_entire_binding(),
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
                                resource: dst_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: dst_upper.as_entire_binding(),
                            },
                        ],
                    });

                    let output_elems = batch_size.checked_mul(*out_features).ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "ibp_forward_gpu: linear output overflow for batch_size={batch_size}, \
                             out_features={out_features}"
                        ))
                    })?;
                    let workgroup_count =
                        gpu_checked_u32(output_elems, "ibp_fwd linear dispatch")?.div_ceil(64);
                    steps.push(PreparedIbpStep::Linear {
                        bind_group,
                        workgroup_count,
                    });

                    use_b = !use_b;
                    cur_dim = output_elems;

                    let Some(last_dim) = cur_shape.last_mut() else {
                        return Err(NyError::InvalidSpec(
                            "ibp_forward_gpu: linear layer requires at least 1D shape".into(),
                        ));
                    };
                    if *last_dim != *in_features {
                        return Err(NyError::shape_mismatch(vec![*in_features], vec![*last_dim]));
                    }
                    *last_dim = *out_features;
                }
                GpuIbpLayer::Conv2d {
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
                } => {
                    if *groups != 1 {
                        return Err(NyError::UnsupportedConfiguration(
                            "resident Conv2d IBP currently supports groups=1 only".into(),
                        ));
                    }

                    let (batch_size, shape_is_batched) = match cur_shape.as_slice() {
                        [channels, height, width] => {
                            if *channels != *in_channels
                                || *height != *input_h
                                || *width != *input_w
                            {
                                return Err(NyError::shape_mismatch(
                                    vec![*in_channels, *input_h, *input_w],
                                    cur_shape.clone(),
                                ));
                            }
                            (1, false)
                        }
                        [batch, channels, height, width] => {
                            if *channels != *in_channels
                                || *height != *input_h
                                || *width != *input_w
                            {
                                return Err(NyError::shape_mismatch(
                                    vec![*batch, *in_channels, *input_h, *input_w],
                                    cur_shape.clone(),
                                ));
                            }
                            (*batch, true)
                        }
                        _ => {
                            return Err(NyError::InvalidSpec(format!(
                                "ibp_forward_gpu: Conv2d requires CHW or NCHW shape, got {:?}",
                                cur_shape
                            )));
                        }
                    };

                    let input_plane =
                        checked_shape_product(&[*in_channels, *input_h, *input_w]).ok_or_else(
                            || {
                                NyError::InvalidSpec(format!(
                                    "ibp_forward_gpu: Conv2d input dims overflow for {in_channels}x{input_h}x{input_w}"
                                ))
                            },
                        )?;
                    let expected_cur_dim = batch_size.checked_mul(input_plane).ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "ibp_forward_gpu: Conv2d batch/input overflow for batch={batch_size}, input_plane={input_plane}"
                        ))
                    })?;
                    if expected_cur_dim != cur_dim {
                        return Err(NyError::shape_mismatch(
                            vec![expected_cur_dim],
                            vec![cur_dim],
                        ));
                    }

                    let expected_weight_len = checked_shape_product(&[
                        *out_channels,
                        *in_channels,
                        *kernel_h,
                        *kernel_w,
                    ])
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "ibp_forward_gpu: Conv2d weight dims overflow for out_channels={out_channels}, in_channels={in_channels}, kernel={kernel_h}x{kernel_w}"
                        ))
                    })?;
                    if weight.len() != expected_weight_len {
                        return Err(NyError::shape_mismatch(
                            vec![expected_weight_len],
                            vec![weight.len()],
                        ));
                    }
                    if let Some(bias) = bias {
                        if bias.len() != *out_channels {
                            return Err(NyError::shape_mismatch(
                                vec![*out_channels],
                                vec![bias.len()],
                            ));
                        }
                    }

                    let (out_h, out_w) = conv2d_output_size(
                        *input_h, *input_w, *kernel_h, *kernel_w, *stride_h, *stride_w, *pad_h,
                        *pad_w,
                    )?;
                    let output_elems =
                        checked_shape_product(&[batch_size, *out_channels, out_h, out_w])
                            .ok_or_else(|| {
                                NyError::InvalidSpec(format!(
                                    "ibp_forward_gpu: Conv2d output overflow for batch={batch_size}, out_channels={out_channels}, out_h={out_h}, out_w={out_w}"
                                ))
                            })?;

                    let (src_lower, src_upper, dst_lower, dst_upper) = if use_b {
                        (&buf_lower_b, &buf_upper_b, &buf_lower_a, &buf_upper_a)
                    } else {
                        (&buf_lower_a, &buf_upper_a, &buf_lower_b, &buf_upper_b)
                    };

                    let weight_pos: Vec<f32> = weight
                        .iter()
                        .map(|&w| nan_propagating_max_zero(w))
                        .collect();
                    let weight_neg: Vec<f32> = weight
                        .iter()
                        .map(|&w| nan_propagating_min_zero(w))
                        .collect();
                    let bias_data: Vec<f32> = match bias {
                        Some(bias) => bias.to_vec(),
                        None => vec![0.0; *out_channels],
                    };

                    let params = Conv2dIbpParams {
                        batch_size: gpu_checked_u32(batch_size, "ibp_fwd conv batch")?,
                        in_channels: gpu_checked_u32(*in_channels, "ibp_fwd conv in_channels")?,
                        out_channels: gpu_checked_u32(*out_channels, "ibp_fwd conv out_channels")?,
                        input_h: gpu_checked_u32(*input_h, "ibp_fwd conv input_h")?,
                        input_w: gpu_checked_u32(*input_w, "ibp_fwd conv input_w")?,
                        out_h: gpu_checked_u32(out_h, "ibp_fwd conv out_h")?,
                        out_w: gpu_checked_u32(out_w, "ibp_fwd conv out_w")?,
                        kernel_h: gpu_checked_u32(*kernel_h, "ibp_fwd conv kernel_h")?,
                        kernel_w: gpu_checked_u32(*kernel_w, "ibp_fwd conv kernel_w")?,
                        stride_h: gpu_checked_u32(*stride_h, "ibp_fwd conv stride_h")?,
                        stride_w: gpu_checked_u32(*stride_w, "ibp_fwd conv stride_w")?,
                        pad_h: gpu_checked_u32(*pad_h, "ibp_fwd conv pad_h")?,
                        pad_w: gpu_checked_u32(*pad_w, "ibp_fwd conv pad_w")?,
                        groups: gpu_checked_u32(*groups, "ibp_fwd conv groups")?,
                        _padding: [0; 2],
                    };

                    let params_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_conv_params",
                        size_of::<Conv2dIbpParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let weight_size = expected_weight_len as u64 * f32_size;
                    let wp_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_conv_wp",
                        weight_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&wp_buf, 0, bytemuck::cast_slice(&weight_pos));

                    let wn_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_conv_wn",
                        weight_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&wn_buf, 0, bytemuck::cast_slice(&weight_neg));

                    let bias_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_conv_bias",
                        (*out_channels as u64) * f32_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&bias_buf, 0, bytemuck::cast_slice(&bias_data));

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ibp_fwd_conv_bg"),
                        layout: &self.conv2d_ibp_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: src_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: src_upper.as_entire_binding(),
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
                                resource: dst_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: dst_upper.as_entire_binding(),
                            },
                        ],
                    });

                    let workgroup_count =
                        gpu_checked_u32(output_elems, "ibp_fwd conv dispatch")?.div_ceil(64);
                    steps.push(PreparedIbpStep::Conv2d {
                        bind_group,
                        workgroup_count,
                    });

                    use_b = !use_b;
                    cur_dim = output_elems;
                    cur_shape = if shape_is_batched {
                        vec![batch_size, *out_channels, out_h, out_w]
                    } else {
                        vec![*out_channels, out_h, out_w]
                    };
                }
                GpuIbpLayer::ReLU { num_elements } => {
                    if *num_elements != cur_dim {
                        return Err(NyError::shape_mismatch(vec![cur_dim], vec![*num_elements]));
                    }

                    let (cur_lower, cur_upper) = if use_b {
                        (&buf_lower_b, &buf_upper_b)
                    } else {
                        (&buf_lower_a, &buf_upper_a)
                    };

                    let params = ReluIbpParams {
                        num_elements: gpu_checked_u32(*num_elements, "ibp_fwd relu elems")?,
                        _padding: [0; 3],
                    };
                    let params_buf = create_buffer(
                        &self.device,
                        "ibp_fwd_relu_params",
                        size_of::<ReluIbpParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ibp_fwd_relu_bg"),
                        layout: &self.relu_ibp_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: cur_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: cur_upper.as_entire_binding(),
                            },
                        ],
                    });

                    let workgroup_count =
                        gpu_checked_u32(*num_elements, "ibp_fwd relu dispatch")?.div_ceil(64);
                    steps.push(PreparedIbpStep::ReLU {
                        bind_group,
                        workgroup_count,
                    });
                }
                GpuIbpLayer::View { output_shape } => {
                    let output_dim = shape_product_or_err(output_shape, "view output")?;
                    if output_dim != cur_dim {
                        return Err(NyError::shape_mismatch(vec![cur_dim], vec![output_dim]));
                    }
                    cur_shape = output_shape.to_vec();
                }
            }
        }

        let output_bytes = (cur_dim as u64) * f32_size;
        let staging_lower = create_buffer(
            &self.device,
            "ibp_fwd_staging_lower",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = create_buffer(
            &self.device,
            "ibp_fwd_staging_upper",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        Ok(WgpuIbpModelPlan {
            device: self.device.clone(),
            queue: self.queue.clone(),
            linear_pipeline: self.linear_ibp_pipeline.clone(),
            conv2d_pipeline: self.conv2d_ibp_pipeline.clone(),
            relu_pipeline: self.relu_ibp_pipeline.clone(),
            input_shape: input_shape.to_vec(),
            input_elements,
            output_shape: cur_shape,
            output_elements: cur_dim,
            final_use_b: use_b,
            buf_lower_a,
            buf_upper_a,
            buf_lower_b,
            buf_upper_b,
            staging_lower,
            staging_upper,
            steps,
        })
    }
}

impl GpuIbpModelPlan for WgpuIbpModelPlan {
    fn ibp_forward_cached(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        // Wrap GPU work in an error scope: a wgpu validation/internal/OOM error
        // returns Err (caller falls back to per-layer CPU) instead of aborting
        // via wgpu's panicking uncaptured-error handler (#live bug).
        super::super::error_scope::run_gpu_checked_on_device(
            &self.device,
            "ibp_forward_cached",
            || self.ibp_forward_cached_inner(input_lower, input_upper, input_shape),
        )
    }
}

impl WgpuIbpModelPlan {
    fn ibp_forward_cached_inner(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        if input_shape != self.input_shape {
            return Err(NyError::InvalidSpec(format!(
                "ibp_forward_cached: input shape mismatch, expected {:?}, got {:?}",
                self.input_shape, input_shape
            )));
        }
        if input_lower.len() != self.input_elements || input_upper.len() != self.input_elements {
            return Err(NyError::InvalidSpec(format!(
                "ibp_forward_cached: input length mismatch, expected {}, got lower={} upper={}",
                self.input_elements,
                input_lower.len(),
                input_upper.len()
            )));
        }

        self.queue
            .write_buffer(&self.buf_lower_a, 0, bytemuck::cast_slice(input_lower));
        self.queue
            .write_buffer(&self.buf_upper_a, 0, bytemuck::cast_slice(input_upper));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ibp_forward_encoder"),
            });

        for step in &self.steps {
            match step {
                PreparedIbpStep::Linear {
                    bind_group,
                    workgroup_count,
                } => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ibp_fwd_linear_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.linear_pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(*workgroup_count, 1, 1);
                }
                PreparedIbpStep::Conv2d {
                    bind_group,
                    workgroup_count,
                } => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ibp_fwd_conv_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.conv2d_pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(*workgroup_count, 1, 1);
                }
                PreparedIbpStep::ReLU {
                    bind_group,
                    workgroup_count,
                } => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ibp_fwd_relu_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.relu_pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(*workgroup_count, 1, 1);
                }
            }
        }

        let (final_lower, final_upper) = if self.final_use_b {
            (&self.buf_lower_b, &self.buf_upper_b)
        } else {
            (&self.buf_lower_a, &self.buf_upper_a)
        };
        let output_bytes = (self.output_elements as u64) * size_of::<f32>() as u64;
        encoder.copy_buffer_to_buffer(final_lower, 0, &self.staging_lower, 0, output_bytes);
        encoder.copy_buffer_to_buffer(final_upper, 0, &self.staging_upper, 0, output_bytes);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Both staging buffers were filled by the SINGLE submit above, so they are
        // ready after one poll. Map them together with ONE blocking
        // `device.poll(Wait)` instead of two sequential polls (one per
        // `read_buffer`). Bit-identical: each vec is the same
        // `get_mapped_range()[..output_elements].to_vec()` of the same buffer.
        let mut batched = WgpuDevice::read_buffers_batched(
            &self.device,
            &[
                (&self.staging_lower, self.output_elements),
                (&self.staging_upper, self.output_elements),
            ],
        )?;
        let mut result_upper = batched.pop().expect("2 readbacks");
        let mut result_lower = batched.pop().expect("2 readbacks");

        sanitize_readback(&mut result_lower, &mut result_upper);

        Ok(GpuIbpResult {
            lower_bounds: result_lower,
            upper_bounds: result_upper,
            output_shape: self.output_shape.clone(),
        })
    }
}
