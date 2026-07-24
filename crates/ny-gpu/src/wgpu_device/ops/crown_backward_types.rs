// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Types and helpers for the batched GPU CROWN backward pass (#3397).
//!
//! Contains `DispatchStep`, `StagingBuilder`, the dispatch plan builder,
//! and dimension query helpers.

use ny_core::{GpuCrownLayer, Result};

use super::gpu_checked_u32;
use crate::wgpu_device::params::{
    ConvCol2imParams, ConvReshapeParams, CrownActivationParams, CrownBiasAccumParams,
    CrownConcretizeParams, CrownMaxPool2dParams, GemmParams,
};

/// Maximum size of any CROWN params struct in bytes.
/// ConvCol2imParams is the largest at 64 bytes (16 × u32).
pub(super) const MAX_PARAMS_SIZE: u64 = 64;

/// Staging buffer builder: accumulates dispatch data as contiguous bytes.
pub(super) struct StagingBuilder {
    data: Vec<u8>,
}

impl StagingBuilder {
    pub(super) fn new() -> Self {
        Self {
            data: Vec::with_capacity(4096),
        }
    }

    /// Append a params struct. Returns byte offset in the staging buffer.
    pub(super) fn push_params<T: bytemuck::Pod>(&mut self, params: &T) -> u64 {
        let offset = self.data.len() as u64;
        self.data.extend_from_slice(bytemuck::bytes_of(params));
        offset
    }

    /// Append f32 slice data. Returns byte offset in the staging buffer.
    pub(super) fn push_f32(&mut self, data: &[f32]) -> u64 {
        let offset = self.data.len() as u64;
        self.data.extend_from_slice(bytemuck::cast_slice(data));
        offset
    }

    /// Append u32 slice data. Returns byte offset in the staging buffer.
    pub(super) fn push_u32(&mut self, data: &[u32]) -> u64 {
        let offset = self.data.len() as u64;
        self.data.extend_from_slice(bytemuck::cast_slice(data));
        offset
    }

    pub(super) fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Describes one dispatch step in the batched backward pass.
pub(super) enum DispatchStep {
    ActivationBackward {
        params_off: u64,
        params_size: u64,
        slopes_off: u64,
        slopes_size: u64,
        num_specs_u32: u32,
        ping: usize,
        /// When true, dispatch the ReLU dual-alpha pipeline instead of
        /// the standard activation pipeline (#4313).
        dual_alpha: bool,
    },
    BiasAccumulate {
        params_off: u64,
        params_size: u64,
        bias_off: u64,
        bias_size: u64,
        num_specs_u32: u32,
        ping: usize,
    },
    MaxPool2dBackward {
        params_off: u64,
        params_size: u64,
        routing_off: u64,
        routing_size: u64,
        bounds_off: u64,
        bounds_size: u64,
        num_specs_u32: u32,
        ping: usize,
    },
    /// Linear GEMM: reads from a_bufs[ping], writes to a_bufs[1-ping]
    GemmCrownLinear {
        params_off: u64,
        params_size: u64,
        weight_off: u64,
        weight_size: u64,
        gemm_params: GemmParams,
        ping: usize,
    },
    /// Conv GEMM: reads from conv.reshaped_*, writes to conv.gemm_*
    GemmCrownConv {
        params_off: u64,
        params_size: u64,
        weight_off: u64,
        weight_size: u64,
        gemm_params: GemmParams,
    },
    ConvReshapeLowerUpper {
        params_off: u64,
        params_size: u64,
        workgroups: u32,
        ping: usize,
    },
    ConvCol2imLowerUpper {
        params_off: u64,
        params_size: u64,
        workgroups: u32,
        ping: usize,
    },
    Concretize {
        params_off: u64,
        params_size: u64,
        num_specs_u32: u32,
        ping: usize,
    },
}

/// Build the staging buffer and dispatch plan for the backward pass.
///
/// Returns `(steps, staging, final_ping, final_dim)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_dispatch_plan(
    layers: &[GpuCrownLayer],
    num_specs: usize,
    num_specs_u32: u32,
    first_dim: usize,
) -> Result<(Vec<DispatchStep>, StagingBuilder, usize, usize)> {
    let mut staging = StagingBuilder::new();
    let mut steps: Vec<DispatchStep> = Vec::new();
    let mut cur_dim = first_dim;
    let mut ping = 0usize;

    for layer in layers {
        match layer {
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                let params = CrownActivationParams {
                    num_specs: num_specs_u32,
                    num_neurons: gpu_checked_u32(*num_neurons, "act num_neurons")?,
                    _padding: [0; 2],
                };
                let params_off = staging.push_params(&params);
                let params_size = size_of::<CrownActivationParams>() as u64;
                let mut slopes_data = Vec::with_capacity(*num_neurons * 4);
                slopes_data.extend_from_slice(lower_slope);
                slopes_data.extend_from_slice(upper_slope);
                slopes_data.extend_from_slice(lower_intercept);
                slopes_data.extend_from_slice(upper_intercept);
                let slopes_off = staging.push_f32(&slopes_data);
                let slopes_size = (slopes_data.len() * size_of::<f32>()) as u64;
                steps.push(DispatchStep::ActivationBackward {
                    params_off,
                    params_size,
                    slopes_off,
                    slopes_size,
                    num_specs_u32,
                    ping,
                    dual_alpha: false,
                });
                ping = 1 - ping;
            }
            GpuCrownLayer::ActivationReluDualAlpha {
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
                num_neurons,
            } => {
                let params = CrownActivationParams {
                    num_specs: num_specs_u32,
                    num_neurons: gpu_checked_u32(*num_neurons, "act dual_alpha num_neurons")?,
                    _padding: [0; 2],
                };
                let params_off = staging.push_params(&params);
                let params_size = size_of::<CrownActivationParams>() as u64;
                // Packed layout: [lower_pos_slope | cross_slope | upper_neg_slope | cross_intercept]
                let mut slopes_data = Vec::with_capacity(*num_neurons * 4);
                slopes_data.extend_from_slice(lower_pos_slope);
                slopes_data.extend_from_slice(cross_slope);
                slopes_data.extend_from_slice(upper_neg_slope);
                slopes_data.extend_from_slice(cross_intercept);
                let slopes_off = staging.push_f32(&slopes_data);
                let slopes_size = (slopes_data.len() * size_of::<f32>()) as u64;
                steps.push(DispatchStep::ActivationBackward {
                    params_off,
                    params_size,
                    slopes_off,
                    slopes_size,
                    num_specs_u32,
                    ping,
                    dual_alpha: true,
                });
                ping = 1 - ping;
            }
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
            } => {
                if let Some(layer_bias) = bias {
                    let bias_params = CrownBiasAccumParams {
                        num_specs: num_specs_u32,
                        num_features: gpu_checked_u32(*out_features, "bias num_features")?,
                        _padding: [0; 2],
                    };
                    let bp_off = staging.push_params(&bias_params);
                    let bp_size = size_of::<CrownBiasAccumParams>() as u64;
                    let bias_off = staging.push_f32(layer_bias);
                    let bias_size = (layer_bias.len() * size_of::<f32>()) as u64;
                    steps.push(DispatchStep::BiasAccumulate {
                        params_off: bp_off,
                        params_size: bp_size,
                        bias_off,
                        bias_size,
                        num_specs_u32,
                        ping,
                    });
                }
                let gemm_params = GemmParams {
                    m: num_specs_u32,
                    k: gpu_checked_u32(*out_features, "gemm k")?,
                    n: gpu_checked_u32(*in_features, "gemm n")?,
                    _padding: 0,
                };
                let gp_off = staging.push_params(&gemm_params);
                let gp_size = size_of::<GemmParams>() as u64;
                let weight_off = staging.push_f32(weight);
                let weight_size = (weight.len() * size_of::<f32>()) as u64;
                steps.push(DispatchStep::GemmCrownLinear {
                    params_off: gp_off,
                    params_size: gp_size,
                    weight_off,
                    weight_size,
                    gemm_params,
                    ping,
                });
                ping = 1 - ping;
                cur_dim = *in_features;
            }
            GpuCrownLayer::MaxPool2d {
                routing,
                ibp_lower,
                ibp_upper,
                input_dim,
                output_dim,
            } => {
                let params = CrownMaxPool2dParams {
                    num_specs: num_specs_u32,
                    input_dim: gpu_checked_u32(*input_dim, "maxpool input_dim")?,
                    output_dim: gpu_checked_u32(*output_dim, "maxpool output_dim")?,
                    _padding: 0,
                };
                let params_off = staging.push_params(&params);
                let params_size = size_of::<CrownMaxPool2dParams>() as u64;
                let routing_off = staging.push_u32(routing);
                let routing_size = (routing.len() * size_of::<u32>()) as u64;

                let mut bounds = Vec::with_capacity(*output_dim * 2);
                bounds.extend_from_slice(ibp_lower);
                bounds.extend_from_slice(ibp_upper);
                let bounds_off = staging.push_f32(&bounds);
                let bounds_size = (bounds.len() * size_of::<f32>()) as u64;

                steps.push(DispatchStep::MaxPool2dBackward {
                    params_off,
                    params_size,
                    routing_off,
                    routing_size,
                    bounds_off,
                    bounds_size,
                    num_specs_u32,
                    ping,
                });
                ping = 1 - ping;
                cur_dim = *input_dim;
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                out_h,
                out_w,
                in_h,
                in_w,
            } => {
                let spatial = out_h * out_w;
                let total_spatial = num_specs * spatial;
                let kernel_cols = in_channels * kernel_h * kernel_w;
                let flat_input_dim = in_channels * in_h * in_w;

                if let Some(expanded_bias) = bias_expanded {
                    let bias_params = CrownBiasAccumParams {
                        num_specs: num_specs_u32,
                        num_features: gpu_checked_u32(
                            out_channels * spatial,
                            "conv bias num_features",
                        )?,
                        _padding: [0; 2],
                    };
                    let bp_off = staging.push_params(&bias_params);
                    let bp_size = size_of::<CrownBiasAccumParams>() as u64;
                    let bias_off = staging.push_f32(expanded_bias);
                    let bias_size = (expanded_bias.len() * size_of::<f32>()) as u64;
                    steps.push(DispatchStep::BiasAccumulate {
                        params_off: bp_off,
                        params_size: bp_size,
                        bias_off,
                        bias_size,
                        num_specs_u32,
                        ping,
                    });
                }

                let reshape_params = ConvReshapeParams {
                    num_specs: num_specs_u32,
                    out_channels: gpu_checked_u32(*out_channels, "conv out_c")?,
                    spatial: gpu_checked_u32(spatial, "conv spatial")?,
                    _padding: 0,
                };
                let rp_off = staging.push_params(&reshape_params);
                let rp_size = size_of::<ConvReshapeParams>() as u64;
                let total = num_specs * spatial * out_channels;
                let workgroups = gpu_checked_u32(total.div_ceil(256), "conv_reshape wg")?;
                steps.push(DispatchStep::ConvReshapeLowerUpper {
                    params_off: rp_off,
                    params_size: rp_size,
                    workgroups,
                    ping,
                });

                let gemm_params = GemmParams {
                    m: gpu_checked_u32(total_spatial, "conv gemm m")?,
                    k: gpu_checked_u32(*out_channels, "conv gemm k")?,
                    n: gpu_checked_u32(kernel_cols, "conv gemm n")?,
                    _padding: 0,
                };
                let gp_off = staging.push_params(&gemm_params);
                let gp_size = size_of::<GemmParams>() as u64;
                let weight_off = staging.push_f32(weight_col);
                let weight_size = (weight_col.len() * size_of::<f32>()) as u64;
                steps.push(DispatchStep::GemmCrownConv {
                    params_off: gp_off,
                    params_size: gp_size,
                    weight_off,
                    weight_size,
                    gemm_params,
                });

                let col2im_params = ConvCol2imParams {
                    num_specs: num_specs_u32,
                    flat_input_dim: gpu_checked_u32(flat_input_dim, "conv flat_dim")?,
                    out_h: gpu_checked_u32(*out_h, "conv out_h")?,
                    out_w: gpu_checked_u32(*out_w, "conv out_w")?,
                    in_channels: gpu_checked_u32(*in_channels, "conv in_c")?,
                    in_h: gpu_checked_u32(*in_h, "conv in_h")?,
                    in_w: gpu_checked_u32(*in_w, "conv in_w")?,
                    kernel_h: gpu_checked_u32(*kernel_h, "conv kh")?,
                    kernel_w: gpu_checked_u32(*kernel_w, "conv kw")?,
                    stride_h: gpu_checked_u32(*stride_h, "conv sh")?,
                    stride_w: gpu_checked_u32(*stride_w, "conv sw")?,
                    pad_h: gpu_checked_u32(*pad_h, "conv ph")?,
                    pad_w: gpu_checked_u32(*pad_w, "conv pw")?,
                    kernel_cols: gpu_checked_u32(kernel_cols, "conv kcols")?,
                    _padding2: [0; 2],
                };
                let total_c2i = num_specs_u32 * col2im_params.flat_input_dim;
                let c2i_wg = gpu_checked_u32((total_c2i as usize).div_ceil(256), "conv_col2im wg")?;
                let cp_off = staging.push_params(&col2im_params);
                let cp_size = size_of::<ConvCol2imParams>() as u64;
                steps.push(DispatchStep::ConvCol2imLowerUpper {
                    params_off: cp_off,
                    params_size: cp_size,
                    workgroups: c2i_wg,
                    ping,
                });

                ping = 1 - ping;
                cur_dim = flat_input_dim;
            }
        }
    }

    // Concretize step
    let conc_params = CrownConcretizeParams {
        num_specs: num_specs_u32,
        input_dim: gpu_checked_u32(cur_dim, "conc input_dim")?,
        _padding: [0; 2],
    };
    let conc_off = staging.push_params(&conc_params);
    let conc_size = size_of::<CrownConcretizeParams>() as u64;
    steps.push(DispatchStep::Concretize {
        params_off: conc_off,
        params_size: conc_size,
        num_specs_u32,
        ping,
    });

    Ok((steps, staging, ping, cur_dim))
}

/// Compute max Conv2d reshape and GEMM output buffer sizes across all layers.
pub(super) fn conv2d_buffer_sizes(layers: &[GpuCrownLayer], num_specs: usize) -> (usize, usize) {
    let max_reshaped = layers
        .iter()
        .filter_map(|l| match l {
            GpuCrownLayer::Conv2d {
                out_channels,
                out_h,
                out_w,
                ..
            } => Some(num_specs * out_h * out_w * out_channels),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    let max_gemm_out = layers
        .iter()
        .filter_map(|l| match l {
            GpuCrownLayer::Conv2d {
                in_channels,
                kernel_h,
                kernel_w,
                out_h,
                out_w,
                ..
            } => Some(num_specs * out_h * out_w * in_channels * kernel_h * kernel_w),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    (max_reshaped, max_gemm_out)
}

pub(super) fn layer_output_dim(layer: &GpuCrownLayer) -> Result<usize> {
    match layer {
        GpuCrownLayer::Linear { out_features, .. } => Ok(*out_features),
        GpuCrownLayer::Activation { num_neurons, .. }
        | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => Ok(*num_neurons),
        GpuCrownLayer::MaxPool2d { output_dim, .. } => Ok(*output_dim),
        GpuCrownLayer::Conv2d {
            out_channels,
            out_h,
            out_w,
            ..
        } => Ok(out_channels * out_h * out_w),
    }
}

pub(super) fn layer_input_dim(layer: &GpuCrownLayer) -> Result<usize> {
    match layer {
        GpuCrownLayer::Linear { in_features, .. } => Ok(*in_features),
        GpuCrownLayer::Activation { num_neurons, .. }
        | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => Ok(*num_neurons),
        GpuCrownLayer::MaxPool2d { input_dim, .. } => Ok(*input_dim),
        GpuCrownLayer::Conv2d {
            in_channels,
            in_h,
            in_w,
            ..
        } => Ok(in_channels * in_h * in_w),
    }
}
