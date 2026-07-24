// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::nan_propagating_max;
use ny_core::GpuCrownLayer;
use ny_tensor::BoundedTensor;

const MAXPOOL_IBP_FALLBACK: u32 = u32::MAX;

/// Build a `GpuCrownLayer::MaxPool2d` from dynamic winner routing bounds.
///
/// This mirrors the CPU `MaxPool2dLayer::propagate_linear_with_bounds` routing
/// logic for the non-overlapping (`stride == kernel_size`) case that the GPU
/// kernel can implement without atomics. For each pooling window we either:
/// - route through a definite winner input position, or
/// - record IBP fallback bounds for sign-aware bias accumulation.
///
/// Reference: alpha-beta-CROWN `BoundMaxPool.bound_backward`
/// (`auto_LiRPA/operators/pooling.py:78-337`)
pub(super) fn extract_maxpool_gpu_layer(
    layer: &crate::layers::pooling::max::MaxPool2dLayer,
    pre: &BoundedTensor,
) -> Option<GpuCrownLayer> {
    let (kh, kw) = layer.kernel_size;
    let (sh, sw) = layer.stride;
    if sh != kh || sw != kw {
        return None;
    }

    let shape = pre.shape();
    let (batch, channels, in_h, in_w) = match shape.len() {
        3 => (1, shape[0], shape[1], shape[2]),
        4 => (shape[0], shape[1], shape[2], shape[3]),
        _ => return None,
    };
    let (out_h, out_w) = layer.output_size(in_h, in_w).ok()?;
    let (ph, pw) = layer.padding;

    let input_dim = batch
        .checked_mul(channels)?
        .checked_mul(in_h)?
        .checked_mul(in_w)?;
    let output_dim = batch
        .checked_mul(channels)?
        .checked_mul(out_h)?
        .checked_mul(out_w)?;

    let pre_lower = pre.lower().as_slice()?;
    let pre_upper = pre.upper().as_slice()?;
    let mut routing = Vec::with_capacity(output_dim);
    let mut ibp_lower = Vec::with_capacity(output_dim);
    let mut ibp_upper = Vec::with_capacity(output_dim);

    let input_index = |batch_idx: usize, channel_idx: usize, h: usize, w: usize| -> usize {
        batch_idx * channels * in_h * in_w + channel_idx * in_h * in_w + h * in_w + w
    };

    for batch_idx in 0..batch {
        for channel_idx in 0..channels {
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let start_y = out_y * sh;
                    let start_x = out_x * sw;
                    let mut max_lower = f32::NEG_INFINITY;
                    let mut max_upper = f32::NEG_INFINITY;
                    let mut max_lower_idx = MAXPOOL_IBP_FALLBACK;
                    let mut second_max_upper = f32::NEG_INFINITY;

                    for kernel_y in 0..kh {
                        for kernel_x in 0..kw {
                            let in_y = (start_y + kernel_y) as isize - ph as isize;
                            let in_x = (start_x + kernel_x) as isize - pw as isize;
                            if in_y < 0
                                || in_y >= in_h as isize
                                || in_x < 0
                                || in_x >= in_w as isize
                            {
                                continue;
                            }

                            let flat_idx =
                                input_index(batch_idx, channel_idx, in_y as usize, in_x as usize);
                            let lower = pre_lower[flat_idx];
                            let upper = pre_upper[flat_idx];
                            if max_lower_idx == MAXPOOL_IBP_FALLBACK || lower > max_lower {
                                max_lower = lower;
                                max_lower_idx = flat_idx as u32;
                            }
                            max_upper = nan_propagating_max(max_upper, upper);
                        }
                    }

                    if max_lower_idx == MAXPOOL_IBP_FALLBACK {
                        return None;
                    }

                    for kernel_y in 0..kh {
                        for kernel_x in 0..kw {
                            let in_y = (start_y + kernel_y) as isize - ph as isize;
                            let in_x = (start_x + kernel_x) as isize - pw as isize;
                            if in_y < 0
                                || in_y >= in_h as isize
                                || in_x < 0
                                || in_x >= in_w as isize
                            {
                                continue;
                            }

                            let flat_idx =
                                input_index(batch_idx, channel_idx, in_y as usize, in_x as usize);
                            if flat_idx as u32 != max_lower_idx {
                                second_max_upper =
                                    nan_propagating_max(second_max_upper, pre_upper[flat_idx]);
                            }
                        }
                    }

                    routing.push(if max_lower >= second_max_upper {
                        max_lower_idx
                    } else {
                        MAXPOOL_IBP_FALLBACK
                    });
                    ibp_lower.push(max_lower);
                    ibp_upper.push(max_upper);
                }
            }
        }
    }

    Some(GpuCrownLayer::MaxPool2d {
        routing,
        ibp_lower,
        ibp_upper,
        input_dim,
        output_dim,
    })
}
