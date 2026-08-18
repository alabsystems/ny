// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! FTZ/DAZ-independent certified Conv1d interval forwards.
//!
//! Every finite binary32 operand is decoded from its integer bit pattern into
//! an exactly equal *normal* binary64 value. A product of two binary32 values
//! has at most 48 significant bits and exponent range `[-298, 256]`, so it is
//! represented exactly by binary64 and cannot be affected by binary64 FTZ/DAZ.
//! Running sums are stepped outward after every binary64 addition. The final
//! directed conversion deliberately emits no binary32 subnormal endpoint:
//! values in the subnormal range are enclosed by zero and
//! `±f32::MIN_POSITIVE`. Consequently neither a flushed binary32 product/result
//! nor a DAZ-sensitive binary32-to-binary64 conversion can tighten the proof.

use ndarray::{Array1, ArrayD, ArrayViewD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use std::time::Instant;

const DEADLINE_CPU_POLL_OPS: usize = 4_096;
const F64_FRACTION_BITS: u32 = 52;
const F64_EXPONENT_BIAS: i32 = 1_023;

#[derive(Debug)]
pub(crate) struct Conv1dCertifiedForward {
    pub(crate) lower: ArrayD<f32>,
    pub(crate) upper: ArrayD<f32>,
}

fn deadline_error(layer: &str, stage: &str) -> NyError {
    NyError::DeadlineExceeded(format!(
        "{layer} certified IBP forward: deadline exceeded {stage}"
    ))
}

#[inline]
fn check_deadline(deadline: Option<Instant>, layer: &str, stage: &str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(deadline_error(layer, stage));
    }
    Ok(())
}

#[inline]
fn poll_operation(operations: &mut usize, deadline: Option<Instant>, layer: &str) -> Result<()> {
    *operations += 1;
    if *operations == DEADLINE_CPU_POLL_OPS {
        check_deadline(deadline, layer, "during directed binary64 accumulation")?;
        *operations = 0;
    }
    Ok(())
}

fn reserve_output<T>(
    len: usize,
    deadline: Option<Instant>,
    layer: &str,
    name: &str,
) -> Result<Vec<T>> {
    check_deadline(deadline, layer, "before output allocation")?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NyError::InvalidSpec(format!(
            "{layer} certified IBP {name} allocation failed for {len} elements: {error}"
        ))
    })?;
    check_deadline(deadline, layer, "after output allocation")?;
    Ok(values)
}

fn zeroed_f64_output(
    len: usize,
    deadline: Option<Instant>,
    layer: &str,
    name: &str,
) -> Result<Vec<f64>> {
    let mut values = reserve_output(len, deadline, layer, name)?;
    while values.len() < len {
        check_deadline(deadline, layer, "during chunked output initialization")?;
        let chunk = (len - values.len()).min(DEADLINE_CPU_POLL_OPS);
        values.extend(std::iter::repeat_n(0.0, chunk));
    }
    check_deadline(deadline, layer, "after chunked output initialization")?;
    Ok(values)
}

/// Decode a binary32 bit pattern into the exactly equal binary64 bit pattern.
///
/// In particular, a binary32 subnormal becomes a *normal* binary64 number.
/// This avoids a hardware `f32 -> f64` conversion whose source could be treated
/// as zero when the host has DAZ enabled.
#[inline]
fn f32_to_f64_exact(value: f32) -> f64 {
    let bits = value.to_bits();
    let sign = u64::from(bits >> 31) << 63;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;

    match (exponent, fraction) {
        (0, 0) => f64::from_bits(sign),
        (0, _) => {
            let leading = fraction.ilog2();
            let unbiased_exponent = leading as i32 - 149;
            let exponent64 = (unbiased_exponent + F64_EXPONENT_BIAS) as u64;
            let leading_bit = 1_u32 << leading;
            let fraction64 = u64::from(fraction - leading_bit) << (F64_FRACTION_BITS - leading);
            f64::from_bits(sign | (exponent64 << F64_FRACTION_BITS) | fraction64)
        }
        (0xff, 0) => f64::from_bits(sign | (0x7ff_u64 << F64_FRACTION_BITS)),
        (0xff, _) => f64::NAN,
        _ => {
            let unbiased_exponent = exponent as i32 - 127;
            let exponent64 = (unbiased_exponent + F64_EXPONENT_BIAS) as u64;
            let fraction64 = u64::from(fraction) << (F64_FRACTION_BITS - 23);
            f64::from_bits(sign | (exponent64 << F64_FRACTION_BITS) | fraction64)
        }
    }
}

/// One binary64 step toward negative infinity without publishing a binary64
/// subnormal that DAZ could erase in the next addition.
#[inline]
fn next_down_f64_no_subnormal(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() {
        return f64::NEG_INFINITY;
    }
    if bits == f64::NEG_INFINITY.to_bits() {
        return f64::NEG_INFINITY;
    }
    if bits == f64::INFINITY.to_bits() {
        return f64::MAX;
    }
    if magnitude == 0 {
        return -f64::MIN_POSITIVE;
    }
    if magnitude < f64::MIN_POSITIVE.to_bits() {
        return if bits & 0x8000_0000_0000_0000 != 0 {
            -f64::MIN_POSITIVE
        } else {
            0.0
        };
    }

    let stepped = if bits & 0x8000_0000_0000_0000 == 0 {
        bits - 1
    } else {
        bits + 1
    };
    let result = f64::from_bits(stepped);
    let result_magnitude = stepped & 0x7fff_ffff_ffff_ffff;
    if result_magnitude != 0 && result_magnitude < f64::MIN_POSITIVE.to_bits() {
        if stepped & 0x8000_0000_0000_0000 != 0 {
            -f64::MIN_POSITIVE
        } else {
            0.0
        }
    } else {
        result
    }
}

#[inline]
fn next_up_f64_no_subnormal(value: f64) -> f64 {
    -next_down_f64_no_subnormal(-value)
}

#[inline]
fn add_down_f64(left: f64, right: f64) -> f64 {
    let rounded = left + right;
    if rounded.is_nan() {
        f64::NEG_INFINITY
    } else {
        next_down_f64_no_subnormal(rounded)
    }
}

#[inline]
fn add_up_f64(left: f64, right: f64) -> f64 {
    let rounded = left + right;
    if rounded.is_nan() {
        f64::INFINITY
    } else {
        next_up_f64_no_subnormal(rounded)
    }
}

/// Directed binary64-to-binary32 lower conversion that never relies on a
/// binary32 subnormal result surviving FTZ.
#[inline]
fn f64_to_f32_down_no_subnormal(value: f64) -> f32 {
    if value.is_nan() {
        return f32::NEG_INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }

    let min_normal = f64::from_bits((F64_EXPONENT_BIAS as u64 - 126) << F64_FRACTION_BITS);
    if value.abs() < min_normal {
        return if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        };
    }

    let nearest = value as f32;
    if nearest == f32::INFINITY {
        return if value.is_finite() {
            f32::MAX
        } else {
            f32::INFINITY
        };
    }
    if nearest == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if f32_to_f64_exact(nearest) <= value {
        nearest
    } else {
        next_down_f32(nearest)
    }
}

/// Directed binary64-to-binary32 upper conversion that never relies on a
/// binary32 subnormal result surviving FTZ.
#[inline]
fn f64_to_f32_up_no_subnormal(value: f64) -> f32 {
    if value.is_nan() {
        return f32::INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }

    let min_normal = f64::from_bits((F64_EXPONENT_BIAS as u64 - 126) << F64_FRACTION_BITS);
    if value.abs() < min_normal {
        return if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        };
    }

    let nearest = value as f32;
    if nearest == f32::NEG_INFINITY {
        return if value.is_finite() {
            f32::MIN
        } else {
            f32::NEG_INFINITY
        };
    }
    if nearest == f32::INFINITY {
        return f32::INFINITY;
    }
    if f32_to_f64_exact(nearest) >= value {
        nearest
    } else {
        next_up_f32(nearest)
    }
}

#[inline]
fn input_value(input: &ArrayViewD<'_, f32>, batch: usize, channel: usize, position: usize) -> f32 {
    if input.ndim() == 2 {
        input[[channel, position]]
    } else {
        input[[batch, channel, position]]
    }
}

#[inline]
fn finite_weighted_interval(weight: f32, input_lower: f32, input_upper: f32) -> Option<(f64, f64)> {
    if input_lower > input_upper {
        return None;
    }
    let weight_magnitude = weight.to_bits() & !0x8000_0000;
    if weight_magnitude == 0 {
        return Some((0.0, 0.0));
    }
    if !weight.is_finite() || !input_lower.is_finite() || !input_upper.is_finite() {
        return None;
    }

    let weight64 = f32_to_f64_exact(weight);
    let lower64 = f32_to_f64_exact(input_lower);
    let upper64 = f32_to_f64_exact(input_upper);
    if weight.is_sign_positive() {
        Some((lower64 * weight64, upper64 * weight64))
    } else {
        Some((upper64 * weight64, lower64 * weight64))
    }
}

fn checked_conv1d_output_length(
    input_len: usize,
    kernel_len: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    layer: &str,
) -> Result<usize> {
    if stride == 0 || dilation == 0 || kernel_len == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{layer} certified IBP requires nonzero stride, dilation, and kernel length"
        )));
    }
    let effective_kernel = kernel_len
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dilation))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!("{layer} certified IBP effective kernel overflow"))
        })?;
    let padded = padding
        .checked_mul(2)
        .and_then(|pad| input_len.checked_add(pad))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!("{layer} certified IBP padded length overflow"))
        })?;
    if padded < effective_kernel {
        return Err(NyError::InvalidSpec(format!(
            "{layer} certified IBP effective kernel {effective_kernel} exceeds padded input \
             {padded}"
        )));
    }
    Ok((padded - effective_kernel) / stride + 1)
}

fn checked_convtranspose1d_output_length(
    input_len: usize,
    kernel_len: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    layer: &str,
) -> Result<usize> {
    if stride == 0 || dilation == 0 || kernel_len == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{layer} certified IBP requires nonzero stride, dilation, and kernel length"
        )));
    }
    let effective_kernel = kernel_len
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dilation))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!("{layer} certified IBP effective kernel overflow"))
        })?;
    let expanded = input_len
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(stride))
        .and_then(|extent| extent.checked_add(effective_kernel))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!("{layer} certified IBP expanded length overflow"))
        })?;
    let double_padding = padding
        .checked_mul(2)
        .ok_or_else(|| NyError::InvalidSpec(format!("{layer} certified IBP padding overflow")))?;
    if expanded < double_padding {
        return Err(NyError::InvalidSpec(format!(
            "{layer} certified IBP expanded length {expanded} is smaller than double padding \
             {double_padding}"
        )));
    }
    Ok(expanded - double_padding)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_ibp_certified_forward(
    input_lower: ArrayViewD<'_, f32>,
    input_upper: ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    deadline: Option<Instant>,
) -> Result<Conv1dCertifiedForward> {
    const LAYER: &str = "Conv1d";
    check_deadline(deadline, LAYER, "before entry")?;
    if !matches!(input_lower.ndim(), 2 | 3) || input_lower.ndim() != input_upper.ndim() {
        return Err(NyError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![input_lower.ndim(), input_upper.ndim()],
        });
    }
    if input_lower.shape() != input_upper.shape() {
        return Err(NyError::ShapeMismatch {
            expected: input_lower.shape().to_vec(),
            got: input_upper.shape().to_vec(),
        });
    }
    if kernel.ndim() != 3 || groups == 0 {
        return Err(NyError::InvalidSpec(
            "Conv1d certified IBP requires a rank-3 kernel and nonzero groups".to_string(),
        ));
    }

    let (batch, in_c, input_len) = if input_lower.ndim() == 2 {
        (1, input_lower.shape()[0], input_lower.shape()[1])
    } else {
        (
            input_lower.shape()[0],
            input_lower.shape()[1],
            input_lower.shape()[2],
        )
    };
    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kernel_len = kernel.shape()[2];
    let expected_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec("Conv1d certified IBP grouped channels overflow".to_string())
    })?;
    if in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![in_c],
        });
    }
    if !out_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "Conv1d certified IBP out_channels {out_c} not divisible by groups {groups}"
        )));
    }
    if bias.is_some_and(|values| values.len() != out_c) {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c],
            got: vec![bias.map_or(0, Array1::len)],
        });
    }

    let out_len =
        checked_conv1d_output_length(input_len, kernel_len, stride, padding, dilation, LAYER)?;
    let output_size = checked_shape_product(&[batch, out_c, out_len]).ok_or_else(|| {
        NyError::InvalidSpec("Conv1d certified IBP output dimensions overflow".to_string())
    })?;
    let mut lower = reserve_output(output_size, deadline, LAYER, "lower")?;
    let mut upper = reserve_output(output_size, deadline, LAYER, "upper")?;

    let out_c_per_group = out_c / groups;
    let mut operations = 0usize;
    let mut output_positions = 0usize;
    for batch_index in 0..batch {
        check_deadline(deadline, LAYER, "before a batch item")?;
        for output_channel in 0..out_c {
            let group = output_channel / out_c_per_group;
            let input_channel_start = group * in_c_per_group;
            for output_position in 0..out_len {
                output_positions += 1;
                if output_positions == DEADLINE_CPU_POLL_OPS {
                    check_deadline(deadline, LAYER, "during output publication")?;
                    output_positions = 0;
                }
                let mut lower_sum = 0.0_f64;
                let mut upper_sum = 0.0_f64;
                let mut conservative = false;
                'terms: for input_channel_local in 0..in_c_per_group {
                    let input_channel = input_channel_start + input_channel_local;
                    for kernel_position in 0..kernel_len {
                        poll_operation(&mut operations, deadline, LAYER)?;
                        let input_position = output_position
                            .checked_mul(stride)
                            .and_then(|base| {
                                kernel_position.checked_mul(dilation)?.checked_add(base)
                            })
                            .and_then(|padded_position| padded_position.checked_sub(padding))
                            .filter(|&position| position < input_len);
                        let Some(input_position) = input_position else {
                            continue;
                        };
                        let term = finite_weighted_interval(
                            kernel[[output_channel, input_channel_local, kernel_position]],
                            input_value(&input_lower, batch_index, input_channel, input_position),
                            input_value(&input_upper, batch_index, input_channel, input_position),
                        );
                        let Some((term_lower, term_upper)) = term else {
                            conservative = true;
                            break 'terms;
                        };
                        lower_sum = add_down_f64(lower_sum, term_lower);
                        upper_sum = add_up_f64(upper_sum, term_upper);
                    }
                }

                if !conservative {
                    if let Some(bias) = bias {
                        let bias_value = bias[output_channel];
                        if bias_value.is_finite() {
                            let bias64 = f32_to_f64_exact(bias_value);
                            lower_sum = add_down_f64(lower_sum, bias64);
                            upper_sum = add_up_f64(upper_sum, bias64);
                        } else {
                            conservative = true;
                        }
                    }
                }

                if conservative || lower_sum.is_nan() || upper_sum.is_nan() {
                    lower.push(f32::NEG_INFINITY);
                    upper.push(f32::INFINITY);
                } else {
                    let lower_value = f64_to_f32_down_no_subnormal(lower_sum);
                    let upper_value = f64_to_f32_up_no_subnormal(upper_sum);
                    if lower_value <= upper_value {
                        lower.push(lower_value);
                        upper.push(upper_value);
                    } else {
                        lower.push(f32::NEG_INFINITY);
                        upper.push(f32::INFINITY);
                    }
                }
            }
        }
    }

    let output_shape = if input_lower.ndim() == 2 {
        vec![out_c, out_len]
    } else {
        vec![batch, out_c, out_len]
    };
    check_deadline(deadline, LAYER, "before reshaping certified bounds")?;
    let lower = ArrayD::from_shape_vec(IxDyn(&output_shape), lower).map_err(|error| {
        NyError::InternalError(format!("Conv1d certified IBP lower reshape: {error}"))
    })?;
    let upper = ArrayD::from_shape_vec(IxDyn(&output_shape), upper).map_err(|error| {
        NyError::InternalError(format!("Conv1d certified IBP upper reshape: {error}"))
    })?;
    check_deadline(deadline, LAYER, "before publishing certified bounds")?;
    Ok(Conv1dCertifiedForward { lower, upper })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_transpose_ibp_certified_forward(
    input_lower: ArrayViewD<'_, f32>,
    input_upper: ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    deadline: Option<Instant>,
) -> Result<Conv1dCertifiedForward> {
    const LAYER: &str = "ConvTranspose1d";
    check_deadline(deadline, LAYER, "before entry")?;
    if !matches!(input_lower.ndim(), 2 | 3) || input_lower.ndim() != input_upper.ndim() {
        return Err(NyError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![input_lower.ndim(), input_upper.ndim()],
        });
    }
    if input_lower.shape() != input_upper.shape() {
        return Err(NyError::ShapeMismatch {
            expected: input_lower.shape().to_vec(),
            got: input_upper.shape().to_vec(),
        });
    }
    if kernel.ndim() != 3 || groups == 0 {
        return Err(NyError::InvalidSpec(
            "ConvTranspose1d certified IBP requires a rank-3 kernel and nonzero groups".to_string(),
        ));
    }

    let (batch, in_c, input_len) = if input_lower.ndim() == 2 {
        (1, input_lower.shape()[0], input_lower.shape()[1])
    } else {
        (
            input_lower.shape()[0],
            input_lower.shape()[1],
            input_lower.shape()[2],
        )
    };
    let kernel_in_c = kernel.shape()[0];
    let out_c_per_group = kernel.shape()[1];
    let kernel_len = kernel.shape()[2];
    if in_c != kernel_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![kernel_in_c],
            got: vec![in_c],
        });
    }
    if !in_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "ConvTranspose1d certified IBP in_channels {in_c} not divisible by groups {groups}"
        )));
    }
    let out_c = out_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec(
            "ConvTranspose1d certified IBP grouped output channels overflow".to_string(),
        )
    })?;
    if bias.is_some_and(|values| values.len() != out_c) {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c],
            got: vec![bias.map_or(0, Array1::len)],
        });
    }

    let out_len = checked_convtranspose1d_output_length(
        input_len, kernel_len, stride, padding, dilation, LAYER,
    )?;
    let output_size = checked_shape_product(&[batch, out_c, out_len]).ok_or_else(|| {
        NyError::InvalidSpec("ConvTranspose1d certified IBP output dimensions overflow".to_string())
    })?;
    let mut lower = zeroed_f64_output(output_size, deadline, LAYER, "lower")?;
    let mut upper = zeroed_f64_output(output_size, deadline, LAYER, "upper")?;

    let in_c_per_group = in_c / groups;
    let mut operations = 0usize;
    let mut input_positions = 0usize;
    for batch_index in 0..batch {
        check_deadline(deadline, LAYER, "before a batch item")?;
        for input_channel in 0..in_c {
            let group = input_channel / in_c_per_group;
            let output_channel_start = group * out_c_per_group;
            for input_position in 0..input_len {
                input_positions += 1;
                if input_positions == DEADLINE_CPU_POLL_OPS {
                    check_deadline(deadline, LAYER, "during input traversal")?;
                    input_positions = 0;
                }
                let input_lower_value =
                    input_value(&input_lower, batch_index, input_channel, input_position);
                let input_upper_value =
                    input_value(&input_upper, batch_index, input_channel, input_position);
                for output_channel_local in 0..out_c_per_group {
                    let output_channel = output_channel_start + output_channel_local;
                    for kernel_position in 0..kernel_len {
                        poll_operation(&mut operations, deadline, LAYER)?;
                        let output_position = input_position
                            .checked_mul(stride)
                            .and_then(|base| {
                                kernel_position.checked_mul(dilation)?.checked_add(base)
                            })
                            .and_then(|padded_position| padded_position.checked_sub(padding))
                            .filter(|&position| position < out_len);
                        let Some(output_position) = output_position else {
                            continue;
                        };
                        let flat_output =
                            (batch_index * out_c + output_channel) * out_len + output_position;
                        let term = finite_weighted_interval(
                            kernel[[input_channel, output_channel_local, kernel_position]],
                            input_lower_value,
                            input_upper_value,
                        );
                        let Some((term_lower, term_upper)) = term else {
                            lower[flat_output] = f64::NEG_INFINITY;
                            upper[flat_output] = f64::INFINITY;
                            continue;
                        };
                        lower[flat_output] = add_down_f64(lower[flat_output], term_lower);
                        upper[flat_output] = add_up_f64(upper[flat_output], term_upper);
                    }
                }
            }
        }
    }

    let mut lower_f32 = reserve_output(output_size, deadline, LAYER, "published lower")?;
    let mut upper_f32 = reserve_output(output_size, deadline, LAYER, "published upper")?;
    for flat_output in 0..output_size {
        if flat_output.is_multiple_of(DEADLINE_CPU_POLL_OPS) {
            check_deadline(deadline, LAYER, "during bias and directed publication")?;
        }
        let output_channel = (flat_output / out_len) % out_c;
        let mut lower_value = lower[flat_output];
        let mut upper_value = upper[flat_output];
        if let Some(bias) = bias {
            let bias_value = bias[output_channel];
            if bias_value.is_finite() {
                let bias64 = f32_to_f64_exact(bias_value);
                lower_value = add_down_f64(lower_value, bias64);
                upper_value = add_up_f64(upper_value, bias64);
            } else {
                lower_value = f64::NEG_INFINITY;
                upper_value = f64::INFINITY;
            }
        }
        let published_lower = f64_to_f32_down_no_subnormal(lower_value);
        let published_upper = f64_to_f32_up_no_subnormal(upper_value);
        if published_lower <= published_upper {
            lower_f32.push(published_lower);
            upper_f32.push(published_upper);
        } else {
            lower_f32.push(f32::NEG_INFINITY);
            upper_f32.push(f32::INFINITY);
        }
    }

    let output_shape = if input_lower.ndim() == 2 {
        vec![out_c, out_len]
    } else {
        vec![batch, out_c, out_len]
    };
    check_deadline(deadline, LAYER, "before reshaping certified bounds")?;
    let lower = ArrayD::from_shape_vec(IxDyn(&output_shape), lower_f32).map_err(|error| {
        NyError::InternalError(format!(
            "ConvTranspose1d certified IBP lower reshape: {error}"
        ))
    })?;
    let upper = ArrayD::from_shape_vec(IxDyn(&output_shape), upper_f32).map_err(|error| {
        NyError::InternalError(format!(
            "ConvTranspose1d certified IBP upper reshape: {error}"
        ))
    })?;
    check_deadline(deadline, LAYER, "before publishing certified bounds")?;
    Ok(Conv1dCertifiedForward { lower, upper })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_decode_promotes_binary32_subnormals_to_normal_binary64() {
        let cases = [
            (0x0000_0001_u32, 874_u64 << 52),
            (0x0000_0002, 875_u64 << 52),
            (0x0040_0000, 896_u64 << 52),
            (0x007f_ffff, (896_u64 << 52) | (0x003f_ffff_u64 << 30)),
            (0x8000_0001, (1_u64 << 63) | (874_u64 << 52)),
        ];
        for (binary32_bits, expected_binary64_bits) in cases {
            let promoted = f32_to_f64_exact(f32::from_bits(binary32_bits));
            assert_eq!(promoted.to_bits(), expected_binary64_bits);
            assert!(promoted.abs() >= f64::MIN_POSITIVE);
        }

        for normal in [0.0_f32, -0.0, 1.0, -2.5, f32::MIN_POSITIVE, f32::MAX] {
            assert_eq!(f32_to_f64_exact(normal), normal as f64);
        }
    }

    #[test]
    fn directed_binary64_steps_cover_zero_and_infinities() {
        assert_eq!(next_down_f64_no_subnormal(0.0), -f64::MIN_POSITIVE);
        assert_eq!(next_up_f64_no_subnormal(0.0), f64::MIN_POSITIVE);
        assert_eq!(next_down_f64_no_subnormal(f64::INFINITY), f64::MAX);
        assert_eq!(next_up_f64_no_subnormal(f64::NEG_INFINITY), -f64::MAX);
        assert!(add_down_f64(1.0, -1.0) < 0.0);
        assert!(add_up_f64(1.0, -1.0) > 0.0);
    }

    #[test]
    fn directed_publication_uses_normal_floor_for_binary32_subnormals() {
        let tiny_positive = f64::from_bits((F64_EXPONENT_BIAS as u64 - 150) << 52);
        assert_eq!(f64_to_f32_down_no_subnormal(tiny_positive), 0.0);
        assert_eq!(f64_to_f32_up_no_subnormal(tiny_positive), f32::MIN_POSITIVE);
        assert_eq!(
            f64_to_f32_down_no_subnormal(-tiny_positive),
            -f32::MIN_POSITIVE
        );
        assert_eq!(f64_to_f32_up_no_subnormal(-tiny_positive), 0.0);
    }
}
