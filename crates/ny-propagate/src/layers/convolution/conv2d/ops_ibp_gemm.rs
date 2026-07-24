// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::Axis;
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};

use super::{conv2d_forward_batched_gemm, Conv2dLayer};
use crate::bounds::{nan_propagating_max_zero, nan_propagating_min_zero};

pub(crate) fn propagate_ibp_via_gemm(
    layer: &Conv2dLayer,
    input: &BoundedTensor,
    engine: &dyn GemmEngine,
) -> Result<BoundedTensor> {
    let in_c = layer.in_channels();
    let (batch, input_h, input_w, squeeze_batch) = match input.lower().ndim() {
        3 => {
            if input.lower().shape()[0] != in_c {
                return Err(NyError::ShapeMismatch {
                    expected: vec![in_c],
                    got: vec![input.lower().shape()[0]],
                });
            }
            (1, input.lower().shape()[1], input.lower().shape()[2], true)
        }
        4 => {
            if input.lower().shape()[1] != in_c {
                return Err(NyError::ShapeMismatch {
                    expected: vec![0, in_c, 0, 0],
                    got: input.lower().shape().to_vec(),
                });
            }
            (
                input.lower().shape()[0],
                input.lower().shape()[2],
                input.lower().shape()[3],
                false,
            )
        }
        _ => {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0, 0],
                got: input.lower().shape().to_vec(),
            });
        }
    };

    let out_c = layer.out_channels();
    let (out_h, out_w) = layer.output_size(input_h, input_w)?;
    let kernel_pos = layer.kernel.mapv(nan_propagating_max_zero);
    let kernel_neg = layer.kernel.mapv(nan_propagating_min_zero);
    let flat_dim = checked_shape_product(&[in_c, input_h, input_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Conv2d IBP: flat input dims overflow: {in_c} * {input_h} * {input_w}"
        ))
    })?;

    let lower_flat = input
        .lower()
        .view()
        .into_shape_with_order((batch, flat_dim))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch, flat_dim],
            got: input.lower().shape().to_vec(),
        })?
        .to_owned();
    let upper_flat = input
        .upper()
        .view()
        .into_shape_with_order((batch, flat_dim))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch, flat_dim],
            got: input.upper().shape().to_vec(),
        })?
        .to_owned();

    let lower_from_pos = conv2d_forward_batched_gemm(
        &lower_flat,
        &kernel_pos,
        layer.stride,
        layer.padding,
        layer.dilation,
        (input_h, input_w),
        Some(engine),
    )?;
    let lower_from_neg = conv2d_forward_batched_gemm(
        &upper_flat,
        &kernel_neg,
        layer.stride,
        layer.padding,
        layer.dilation,
        (input_h, input_w),
        Some(engine),
    )?;
    let upper_from_pos = conv2d_forward_batched_gemm(
        &upper_flat,
        &kernel_pos,
        layer.stride,
        layer.padding,
        layer.dilation,
        (input_h, input_w),
        Some(engine),
    )?;
    let upper_from_neg = conv2d_forward_batched_gemm(
        &lower_flat,
        &kernel_neg,
        layer.stride,
        layer.padding,
        layer.dilation,
        (input_h, input_w),
        Some(engine),
    )?;

    let lower_rows = lower_from_pos + lower_from_neg;
    let upper_rows = upper_from_pos + upper_from_neg;
    let spatial = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Conv2d IBP: output dims overflow: {out_c} * {out_h} * {out_w}"
        ))
    })?;
    let mut lower_y = if squeeze_batch {
        lower_rows
            .index_axis(Axis(0), 0)
            .to_owned()
            .into_shape_with_order((out_c, out_h, out_w))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![out_c, out_h, out_w],
                got: vec![spatial],
            })?
            .into_dyn()
    } else {
        lower_rows
            .into_shape_with_order((batch, out_c, out_h, out_w))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![batch, out_c, out_h, out_w],
                got: vec![batch, spatial],
            })?
            .into_dyn()
    };
    let mut upper_y = if squeeze_batch {
        upper_rows
            .index_axis(Axis(0), 0)
            .to_owned()
            .into_shape_with_order((out_c, out_h, out_w))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![out_c, out_h, out_w],
                got: vec![spatial],
            })?
            .into_dyn()
    } else {
        upper_rows
            .into_shape_with_order((batch, out_c, out_h, out_w))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![batch, out_c, out_h, out_w],
                got: vec![batch, spatial],
            })?
            .into_dyn()
    };

    if let Some(bias) = &layer.bias {
        if squeeze_batch {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        lower_y[[oc, oh, ow]] += bias[oc];
                        upper_y[[oc, oh, ow]] += bias[oc];
                    }
                }
            }
        } else {
            for b in 0..batch {
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            lower_y[[b, oc, oh, ow]] += bias[oc];
                            upper_y[[b, oc, oh, ow]] += bias[oc];
                        }
                    }
                }
            }
        }
    }

    BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)
}
