// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! FTZ/DAZ-independent certified Conv3d interval forward.
//!
//! This is the rank-3-spatial generalisation of the Conv1d certified kernel in
//! `conv1d/ops_ibp_cert.rs` and of the `deadline` arm of Conv2d's
//! `propagate_ibp_sound_with_engine_and_deadline`. It is deliberately a
//! SELF-CONTAINED copy of those directed-rounding primitives: this module owns
//! the Conv3d family and must not reach into another family's private
//! numerics.
//!
//! # Soundness argument
//!
//! Every finite binary32 operand is decoded from its integer bit pattern into
//! an exactly equal *normal* binary64 value ([`f32_to_f64_exact`]). A product
//! of two binary32 values has at most 48 significant bits and lives in exponent
//! range `[-298, 256]`, so it is represented EXACTLY by binary64 and cannot be
//! perturbed by binary64 FTZ/DAZ. Therefore the only operations that round are
//! the running additions, and each of those is stepped OUTWARD immediately
//! after it is performed ([`add_down_f64`] / [`add_up_f64`]), so
//!
//! * `lower_sum <= Σ_k min(w_k·l_k, w_k·u_k)` and
//! * `upper_sum >= Σ_k max(w_k·l_k, w_k·u_k)`
//!
//! hold exactly at every step of the fold, for every fold order. Because each
//! window term is the exact interval image of `w_k · [l_k, u_k]` (the endpoints
//! are swapped when `w_k < 0`) and the convolution window is a SUM of
//! independent terms, the accumulated interval encloses the true real-arithmetic
//! convolution output. The final binary64 -> binary32 conversion is likewise
//! directed outward and deliberately emits no binary32 subnormal endpoint:
//! values in the subnormal range are enclosed by zero and `±f32::MIN_POSITIVE`,
//! so neither a flushed binary32 result nor a DAZ-sensitive binary32-to-binary64
//! conversion downstream can tighten the proof.
//!
//! Anything the enclosure cannot certify — a non-finite weight, a non-finite or
//! inverted input endpoint, a non-finite bias, a NaN accumulator, or an
//! inverted published pair — degrades that output cell to `[-inf, +inf]`.
//! FAIL-CLOSED: the box is vacuous, never wrong.

use ndarray::{Array1, ArrayD, ArrayViewD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use std::time::Instant;

const DEADLINE_CPU_POLL_OPS: usize = 4_096;
const F64_FRACTION_BITS: u32 = 52;
const F64_EXPONENT_BIAS: i32 = 1_023;

const LAYER: &str = "Conv3d";

/// Certified interval image of one Conv3d forward pass.
#[derive(Debug)]
pub(crate) struct Conv3dCertifiedForward {
    pub(crate) lower: ArrayD<f32>,
    pub(crate) upper: ArrayD<f32>,
}

fn deadline_error(stage: &str) -> NyError {
    NyError::DeadlineExceeded(format!(
        "{LAYER} certified IBP forward: deadline exceeded {stage}"
    ))
}

#[inline]
fn check_deadline(deadline: Option<Instant>, stage: &str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(deadline_error(stage));
    }
    Ok(())
}

#[inline]
fn poll_operation(operations: &mut usize, deadline: Option<Instant>) -> Result<()> {
    *operations += 1;
    if *operations == DEADLINE_CPU_POLL_OPS {
        check_deadline(deadline, "during directed binary64 accumulation")?;
        *operations = 0;
    }
    Ok(())
}

fn reserve_output<T>(len: usize, deadline: Option<Instant>, name: &str) -> Result<Vec<T>> {
    check_deadline(deadline, "before output allocation")?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NyError::InvalidSpec(format!(
            "{LAYER} certified IBP {name} allocation failed for {len} elements: {error}"
        ))
    })?;
    check_deadline(deadline, "after output allocation")?;
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

/// Exact interval image of `weight * [input_lower, input_upper]` in binary64.
///
/// `None` means "this term cannot be certified" — the caller must degrade the
/// whole output cell to `[-inf, +inf]`. A `+0.0`/`-0.0` weight annihilates a
/// non-finite input, which is why the zero test precedes the finiteness test
/// (matching the Conv1d kernel).
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

/// `floor((input + 2*padding - (dilation*(kernel-1) + 1)) / stride) + 1`, with
/// every intermediate checked. Shared by [`super::types::Conv3dLayer::output_size`]
/// so the layer geometry and this kernel can never disagree.
pub(crate) fn checked_conv3d_output_extent(
    input_extent: usize,
    kernel_extent: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    axis: &str,
) -> Result<usize> {
    if stride == 0 || dilation == 0 || kernel_extent == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{LAYER} {axis} extent requires nonzero stride, dilation, and kernel extent"
        )));
    }
    let effective_kernel = kernel_extent
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dilation))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!("{LAYER} {axis} effective kernel extent overflow"))
        })?;
    let padded = padding
        .checked_mul(2)
        .and_then(|pad| input_extent.checked_add(pad))
        .ok_or_else(|| NyError::InvalidSpec(format!("{LAYER} {axis} padded extent overflow")))?;
    if padded < effective_kernel {
        return Err(NyError::InvalidSpec(format!(
            "{LAYER} effective kernel {effective_kernel} exceeds padded input {padded} on the \
             {axis} axis (input={input_extent}, padding={padding}, kernel={kernel_extent}, \
             dilation={dilation})"
        )));
    }
    Ok((padded - effective_kernel) / stride + 1)
}

/// Volumetric input geometry recovered from a rank-4 or rank-5 bound tensor.
#[derive(Clone, Copy)]
struct InputGeometry {
    batch: usize,
    in_c: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    batched: bool,
}

impl InputGeometry {
    #[inline]
    fn value(
        &self,
        input: &ArrayViewD<'_, f32>,
        n: usize,
        c: usize,
        d: usize,
        h: usize,
        w: usize,
    ) -> f32 {
        if self.batched {
            input[[n, c, d, h, w]]
        } else {
            input[[c, d, h, w]]
        }
    }
}

fn input_geometry(lower: &ArrayViewD<'_, f32>) -> Result<InputGeometry> {
    match lower.ndim() {
        4 => Ok(InputGeometry {
            batch: 1,
            in_c: lower.shape()[0],
            in_d: lower.shape()[1],
            in_h: lower.shape()[2],
            in_w: lower.shape()[3],
            batched: false,
        }),
        5 => Ok(InputGeometry {
            batch: lower.shape()[0],
            in_c: lower.shape()[1],
            in_d: lower.shape()[2],
            in_h: lower.shape()[3],
            in_w: lower.shape()[4],
            batched: true,
        }),
        _ => Err(NyError::ShapeMismatch {
            expected: vec![4, 5],
            got: vec![lower.ndim()],
        }),
    }
}

/// Certified Conv3d interval forward over a `(C, D, H, W)` or `(N, C, D, H, W)`
/// bound tensor.
///
/// See the module doc comment for the enclosure argument. `deadline: Some(_)`
/// makes the kernel pollable: every `DEADLINE_CPU_POLL_OPS` multiply-accumulates
/// and every 4096 published cells re-check the limit and return a typed
/// [`NyError::DeadlineExceeded`]. The kernel is engine-free by construction, so
/// a finite deadline never disappears into an opaque GEMM.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv3d_ibp_certified_forward(
    input_lower: ArrayViewD<'_, f32>,
    input_upper: ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    deadline: Option<Instant>,
) -> Result<Conv3dCertifiedForward> {
    check_deadline(deadline, "before entry")?;
    if input_lower.ndim() != input_upper.ndim() {
        return Err(NyError::ShapeMismatch {
            expected: vec![input_lower.ndim()],
            got: vec![input_upper.ndim()],
        });
    }
    if input_lower.shape() != input_upper.shape() {
        return Err(NyError::ShapeMismatch {
            expected: input_lower.shape().to_vec(),
            got: input_upper.shape().to_vec(),
        });
    }
    if kernel.ndim() != 5 || groups == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{LAYER} certified IBP requires a rank-5 kernel and nonzero groups, got kernel {:?} \
             and groups {groups}",
            kernel.shape()
        )));
    }
    if stride.contains(&0) || dilation.contains(&0) {
        return Err(NyError::InvalidSpec(format!(
            "{LAYER} certified IBP requires stride and dilation >= 1 on every axis, got \
             stride={stride:?}, dilation={dilation:?}"
        )));
    }

    let geometry = input_geometry(&input_lower)?;
    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let [kernel_d, kernel_h, kernel_w] = [kernel.shape()[2], kernel.shape()[3], kernel.shape()[4]];
    let expected_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec(format!("{LAYER} certified IBP grouped channels overflow"))
    })?;
    if geometry.in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![geometry.in_c],
        });
    }
    if !out_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "{LAYER} certified IBP out_channels {out_c} not divisible by groups {groups}"
        )));
    }
    if bias.is_some_and(|values| values.len() != out_c) {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c],
            got: vec![bias.map_or(0, Array1::len)],
        });
    }

    let out_d = checked_conv3d_output_extent(
        geometry.in_d,
        kernel_d,
        stride[0],
        padding[0],
        dilation[0],
        "depth",
    )?;
    let out_h = checked_conv3d_output_extent(
        geometry.in_h,
        kernel_h,
        stride[1],
        padding[1],
        dilation[1],
        "height",
    )?;
    let out_w = checked_conv3d_output_extent(
        geometry.in_w,
        kernel_w,
        stride[2],
        padding[2],
        dilation[2],
        "width",
    )?;
    let output_size = checked_shape_product(&[geometry.batch, out_c, out_d, out_h, out_w])
        .ok_or_else(|| {
            NyError::InvalidSpec(format!("{LAYER} certified IBP output dimensions overflow"))
        })?;
    let mut lower = reserve_output(output_size, deadline, "lower")?;
    let mut upper = reserve_output(output_size, deadline, "upper")?;

    let out_c_per_group = out_c / groups;
    let mut operations = 0usize;
    let mut published = 0usize;
    for batch_index in 0..geometry.batch {
        check_deadline(deadline, "before a batch item")?;
        for output_channel in 0..out_c {
            let group = output_channel / out_c_per_group;
            let input_channel_start = group * in_c_per_group;
            for output_depth in 0..out_d {
                for output_row in 0..out_h {
                    for output_column in 0..out_w {
                        published += 1;
                        if published == DEADLINE_CPU_POLL_OPS {
                            check_deadline(deadline, "during output publication")?;
                            published = 0;
                        }
                        let mut lower_sum = 0.0_f64;
                        let mut upper_sum = 0.0_f64;
                        let mut conservative = false;
                        'terms: for input_channel_local in 0..in_c_per_group {
                            let input_channel = input_channel_start + input_channel_local;
                            for kd in 0..kernel_d {
                                let Some(input_depth) = tap_position(
                                    output_depth,
                                    kd,
                                    stride[0],
                                    dilation[0],
                                    padding[0],
                                    geometry.in_d,
                                ) else {
                                    continue;
                                };
                                for kh in 0..kernel_h {
                                    let Some(input_row) = tap_position(
                                        output_row,
                                        kh,
                                        stride[1],
                                        dilation[1],
                                        padding[1],
                                        geometry.in_h,
                                    ) else {
                                        continue;
                                    };
                                    for kw in 0..kernel_w {
                                        poll_operation(&mut operations, deadline)?;
                                        let Some(input_column) = tap_position(
                                            output_column,
                                            kw,
                                            stride[2],
                                            dilation[2],
                                            padding[2],
                                            geometry.in_w,
                                        ) else {
                                            continue;
                                        };
                                        let term = finite_weighted_interval(
                                            kernel
                                                [[output_channel, input_channel_local, kd, kh, kw]],
                                            geometry.value(
                                                &input_lower,
                                                batch_index,
                                                input_channel,
                                                input_depth,
                                                input_row,
                                                input_column,
                                            ),
                                            geometry.value(
                                                &input_upper,
                                                batch_index,
                                                input_channel,
                                                input_depth,
                                                input_row,
                                                input_column,
                                            ),
                                        );
                                        let Some((term_lower, term_upper)) = term else {
                                            conservative = true;
                                            break 'terms;
                                        };
                                        lower_sum = add_down_f64(lower_sum, term_lower);
                                        upper_sum = add_up_f64(upper_sum, term_upper);
                                    }
                                }
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
        }
    }

    let output_shape = if geometry.batched {
        vec![geometry.batch, out_c, out_d, out_h, out_w]
    } else {
        vec![out_c, out_d, out_h, out_w]
    };
    check_deadline(deadline, "before reshaping certified bounds")?;
    let lower = ArrayD::from_shape_vec(IxDyn(&output_shape), lower).map_err(|error| {
        NyError::InternalError(format!("{LAYER} certified IBP lower reshape: {error}"))
    })?;
    let upper = ArrayD::from_shape_vec(IxDyn(&output_shape), upper).map_err(|error| {
        NyError::InternalError(format!("{LAYER} certified IBP upper reshape: {error}"))
    })?;
    check_deadline(deadline, "before publishing certified bounds")?;
    Ok(Conv3dCertifiedForward { lower, upper })
}

/// Input index touched by kernel tap `k` of output position `o`, or `None` when
/// the tap lands in the zero padding (or outside the input entirely).
///
/// All arithmetic is checked: a padded position that would underflow `usize`
/// is a padding tap, and a position at or beyond `input_extent` is outside.
#[inline]
fn tap_position(
    output_position: usize,
    kernel_position: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    input_extent: usize,
) -> Option<usize> {
    output_position
        .checked_mul(stride)
        .and_then(|base| kernel_position.checked_mul(dilation)?.checked_add(base))
        .and_then(|padded_position| padded_position.checked_sub(padding))
        .filter(|&position| position < input_extent)
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

    /// A padding tap and an out-of-range tap must both be skipped, and a real
    /// tap must land on `o*stride + k*dilation - padding`.
    #[test]
    fn tap_positions_skip_padding_and_overrun() {
        assert_eq!(tap_position(0, 0, 1, 1, 1, 4), None);
        assert_eq!(tap_position(0, 1, 1, 1, 1, 4), Some(0));
        assert_eq!(tap_position(2, 2, 2, 3, 0, 4), None);
        assert_eq!(tap_position(1, 2, 2, 1, 0, 5), Some(4));
    }
}
