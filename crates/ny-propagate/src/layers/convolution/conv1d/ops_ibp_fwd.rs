// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pollable finite-deadline Conv1d interval forwards.
//!
//! The historical Conv1d IBP paths either enter an opaque [`ny_core::GemmEngine`]
//! or execute a complete convolution before the graph can inspect its deadline.
//! Deadline-scored verifier work uses the direct contractions here instead. They
//! compute the same sign-split interval image while polling between bounded
//! scalar-work quanta.

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::time::Instant;

/// Maximum candidate multiply/add operations admitted between deadline polls.
const DEADLINE_CPU_POLL_OPS: usize = 4_096;

#[derive(Debug)]
pub(crate) struct Conv1dIbpForward {
    pub(crate) lower: ArrayD<f32>,
    pub(crate) upper: ArrayD<f32>,
    pub(crate) out_len: usize,
}

fn deadline_error(layer: &str) -> NyError {
    NyError::DeadlineExceeded(format!(
        "{layer} IBP forward: deadline exceeded during pollable CPU contraction"
    ))
}

fn check_deadline(deadline: Instant, layer: &str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(deadline_error(layer));
    }
    Ok(())
}

fn zeroed_output(len: usize, deadline: Instant, layer: &str, name: &str) -> Result<Vec<f32>> {
    check_deadline(deadline, layer)?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NyError::InvalidSpec(format!(
            "{layer} IBP {name} allocation failed for {len} elements: {error}"
        ))
    })?;
    check_deadline(deadline, layer)?;
    while values.len() < len {
        let chunk = (len - values.len()).min(DEADLINE_CPU_POLL_OPS);
        values.extend(std::iter::repeat_n(0.0, chunk));
        check_deadline(deadline, layer)?;
    }
    Ok(values)
}

/// Finite-deadline grouped Conv1d interval forward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_ibp_forward_with_deadline(
    input_lower: ArrayViewD<'_, f32>,
    input_upper: ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    deadline: Instant,
) -> Result<Conv1dIbpForward> {
    const LAYER: &str = "Conv1d";
    check_deadline(deadline, LAYER)?;
    if input_lower.ndim() != 2 || input_upper.ndim() != 2 {
        return Err(NyError::ShapeMismatch {
            expected: vec![2],
            got: vec![input_lower.ndim().max(input_upper.ndim())],
        });
    }
    if input_lower.shape() != input_upper.shape() {
        return Err(NyError::ShapeMismatch {
            expected: input_lower.shape().to_vec(),
            got: input_upper.shape().to_vec(),
        });
    }
    if kernel.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }
    if stride == 0 {
        return Err(NyError::InvalidSpec(
            "Conv1d IBP forward: stride must be >= 1".to_string(),
        ));
    }
    if dilation == 0 {
        return Err(NyError::InvalidSpec(
            "Conv1d IBP forward: dilation must be >= 1".to_string(),
        ));
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "Conv1d IBP forward: groups must be >= 1".to_string(),
        ));
    }

    let in_c = input_lower.shape()[0];
    let in_len = input_lower.shape()[1];
    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kernel_len = kernel.shape()[2];
    if kernel_len == 0 {
        return Err(NyError::InvalidSpec(
            "Conv1d IBP forward: kernel length must be >= 1".to_string(),
        ));
    }
    let expected_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec("Conv1d IBP forward: grouped input channels overflow".to_string())
    })?;
    if in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![in_c],
        });
    }
    if !out_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "Conv1d IBP forward: out_channels {out_c} not divisible by groups {groups}"
        )));
    }

    let effective_kernel = kernel_len
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dilation))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("Conv1d IBP forward: effective kernel overflow".to_string())
        })?;
    let padded = padding
        .checked_mul(2)
        .and_then(|pad| in_len.checked_add(pad))
        .ok_or_else(|| {
            NyError::InvalidSpec("Conv1d IBP forward: padded length overflow".to_string())
        })?;
    if padded < effective_kernel {
        return Err(NyError::InvalidSpec(format!(
            "Conv1d IBP forward: effective kernel {effective_kernel} larger than padded input \
             {padded}"
        )));
    }
    let out_len = (padded - effective_kernel) / stride + 1;
    let output_size = checked_shape_product(&[out_c, out_len]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Conv1d IBP forward: output dimensions overflow: {out_c} * {out_len}"
        ))
    })?;
    let mut lower = zeroed_output(output_size, deadline, LAYER, "lower")?;
    let mut upper = zeroed_output(output_size, deadline, LAYER, "upper")?;

    let out_c_per_group = out_c / groups;
    let mut operations = 0usize;
    let mut output_positions = 0usize;
    for oc in 0..out_c {
        check_deadline(deadline, LAYER)?;
        let group = oc / out_c_per_group;
        let ic_start = group * in_c_per_group;
        for ol in 0..out_len {
            output_positions += 1;
            if output_positions == DEADLINE_CPU_POLL_OPS {
                check_deadline(deadline, LAYER)?;
                output_positions = 0;
            }
            let mut lower_sum = 0.0f32;
            let mut upper_sum = 0.0f32;
            for ic_local in 0..in_c_per_group {
                let ic = ic_start + ic_local;
                for ki in 0..kernel_len {
                    operations += 1;
                    if operations == DEADLINE_CPU_POLL_OPS {
                        check_deadline(deadline, LAYER)?;
                        operations = 0;
                    }
                    let input_index = ol
                        .checked_mul(stride)
                        .and_then(|base| ki.checked_mul(dilation)?.checked_add(base))
                        .and_then(|padded_index| padded_index.checked_sub(padding))
                        .filter(|&index| index < in_len);
                    let Some(input_index) = input_index else {
                        continue;
                    };
                    let weight = kernel[[oc, ic_local, ki]];
                    let input_lo = input_lower[[ic, input_index]];
                    let input_up = input_upper[[ic, input_index]];
                    if weight >= 0.0 {
                        lower_sum += input_lo * weight;
                        upper_sum += input_up * weight;
                    } else {
                        lower_sum += input_up * weight;
                        upper_sum += input_lo * weight;
                    }
                }
            }
            let output_index = oc * out_len + ol;
            lower[output_index] = lower_sum;
            upper[output_index] = upper_sum;
        }
    }
    check_deadline(deadline, LAYER)?;
    let lower = ArrayD::from_shape_vec(IxDyn(&[out_c, out_len]), lower)
        .map_err(|error| NyError::InternalError(format!("Conv1d IBP lower reshape: {error}")))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&[out_c, out_len]), upper)
        .map_err(|error| NyError::InternalError(format!("Conv1d IBP upper reshape: {error}")))?;
    check_deadline(deadline, LAYER)?;
    Ok(Conv1dIbpForward {
        lower,
        upper,
        out_len,
    })
}

/// Finite-deadline grouped ConvTranspose1d interval forward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_transpose_ibp_forward_with_deadline(
    input_lower: ArrayViewD<'_, f32>,
    input_upper: ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    deadline: Instant,
) -> Result<Conv1dIbpForward> {
    const LAYER: &str = "ConvTranspose1d";
    check_deadline(deadline, LAYER)?;
    if input_lower.ndim() != 2 || input_upper.ndim() != 2 {
        return Err(NyError::ShapeMismatch {
            expected: vec![2],
            got: vec![input_lower.ndim().max(input_upper.ndim())],
        });
    }
    if input_lower.shape() != input_upper.shape() {
        return Err(NyError::ShapeMismatch {
            expected: input_lower.shape().to_vec(),
            got: input_upper.shape().to_vec(),
        });
    }
    if kernel.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }
    if stride == 0 {
        return Err(NyError::InvalidSpec(
            "ConvTranspose1d IBP forward: stride must be >= 1".to_string(),
        ));
    }
    if dilation == 0 {
        return Err(NyError::InvalidSpec(
            "ConvTranspose1d IBP forward: dilation must be >= 1".to_string(),
        ));
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "ConvTranspose1d IBP forward: groups must be >= 1".to_string(),
        ));
    }

    let in_c = input_lower.shape()[0];
    let in_len = input_lower.shape()[1];
    let kernel_in_c = kernel.shape()[0];
    let out_c_per_group = kernel.shape()[1];
    let kernel_len = kernel.shape()[2];
    if in_c != kernel_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![kernel_in_c],
            got: vec![in_c],
        });
    }
    if kernel_len == 0 {
        return Err(NyError::InvalidSpec(
            "ConvTranspose1d IBP forward: kernel length must be >= 1".to_string(),
        ));
    }
    if !in_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "ConvTranspose1d IBP forward: in_channels {in_c} not divisible by groups {groups}"
        )));
    }
    let out_c = out_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec(
            "ConvTranspose1d IBP forward: grouped output channels overflow".to_string(),
        )
    })?;
    let effective_kernel = kernel_len
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dilation))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose1d IBP forward: effective kernel overflow".to_string(),
            )
        })?;
    let expanded = in_len
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(stride))
        .and_then(|extent| extent.checked_add(effective_kernel))
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose1d IBP forward: expanded length overflow".to_string(),
            )
        })?;
    let double_padding = padding.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec("ConvTranspose1d IBP forward: padding overflow".to_string())
    })?;
    if expanded < double_padding {
        return Err(NyError::InvalidSpec(format!(
            "ConvTranspose1d IBP forward: expanded length {expanded} smaller than double \
             padding {double_padding}"
        )));
    }
    let out_len = expanded - double_padding;
    let output_size = checked_shape_product(&[out_c, out_len]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose1d IBP forward: output dimensions overflow: {out_c} * {out_len}"
        ))
    })?;
    let mut lower = zeroed_output(output_size, deadline, LAYER, "lower")?;
    let mut upper = zeroed_output(output_size, deadline, LAYER, "upper")?;

    let in_c_per_group = in_c / groups;
    let mut operations = 0usize;
    let mut input_positions = 0usize;
    for ic in 0..in_c {
        check_deadline(deadline, LAYER)?;
        let group = ic / in_c_per_group;
        let oc_start = group * out_c_per_group;
        for input_index in 0..in_len {
            input_positions += 1;
            if input_positions == DEADLINE_CPU_POLL_OPS {
                check_deadline(deadline, LAYER)?;
                input_positions = 0;
            }
            let input_lo = input_lower[[ic, input_index]];
            let input_up = input_upper[[ic, input_index]];
            for oc_local in 0..out_c_per_group {
                let oc = oc_start + oc_local;
                for ki in 0..kernel_len {
                    operations += 1;
                    if operations == DEADLINE_CPU_POLL_OPS {
                        check_deadline(deadline, LAYER)?;
                        operations = 0;
                    }
                    let output_index = input_index
                        .checked_mul(stride)
                        .and_then(|base| ki.checked_mul(dilation)?.checked_add(base))
                        .and_then(|padded_index| padded_index.checked_sub(padding))
                        .filter(|&index| index < out_len);
                    let Some(output_position) = output_index else {
                        continue;
                    };
                    let weight = kernel[[ic, oc_local, ki]];
                    let flat_output = oc * out_len + output_position;
                    if weight >= 0.0 {
                        lower[flat_output] += input_lo * weight;
                        upper[flat_output] += input_up * weight;
                    } else {
                        lower[flat_output] += input_up * weight;
                        upper[flat_output] += input_lo * weight;
                    }
                }
            }
        }
    }
    check_deadline(deadline, LAYER)?;
    let lower = ArrayD::from_shape_vec(IxDyn(&[out_c, out_len]), lower).map_err(|error| {
        NyError::InternalError(format!("ConvTranspose1d IBP lower reshape: {error}"))
    })?;
    let upper = ArrayD::from_shape_vec(IxDyn(&[out_c, out_len]), upper).map_err(|error| {
        NyError::InternalError(format!("ConvTranspose1d IBP upper reshape: {error}"))
    })?;
    check_deadline(deadline, LAYER)?;
    Ok(Conv1dIbpForward {
        lower,
        upper,
        out_len,
    })
}
