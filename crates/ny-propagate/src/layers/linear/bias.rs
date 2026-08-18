// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bias accumulation and directed-rounding finalization for linear CROWN.
//!
//! Centralizes the f64 bias accumulation pattern shared by CPU CROWN,
//! GEMM CROWN, and batched CROWN backward paths.
//!
//! # Precision rationale
//!
//! Bias accumulation uses f64 to prevent catastrophic cancellation when
//! large positive and negative contributions nearly cancel. After
//! accumulation, the result is cast back to f32 with directed rounding. Values
//! in the binary32 subnormal range are published at the adjacent normal/zero
//! endpoint so the certificate does not depend on host FTZ state.
//!
//! References:
//! - Issue #1863: f64 accumulation requirement
//! - Issue #2164: directed-rounding finalization
//! - `common.rs:170-175`: nonlinear path's equivalent precision standard

use ndarray::Array1;
use ny_core::dd::{next_down_f64, next_up_f64};
use ny_tensor::{next_down_f32, next_up_f32};

const F64_FRACTION_BITS: u32 = 52;
const F64_EXPONENT_BIAS: i32 = 1023;

/// Decode a binary32 bit pattern without presenting a subnormal binary32
/// operand to a hardware conversion instruction.
///
/// On hosts with DAZ enabled, an ordinary `f32 as f64` conversion may observe a
/// subnormal source as signed zero. A binary32 subnormal is a normal binary64
/// value, so constructing the binary64 bits directly keeps all later bias
/// arithmetic outside the binary64 FTZ/DAZ range.
#[inline]
pub(crate) fn f32_to_f64_exact(value: f32) -> f64 {
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

#[inline]
fn binary32_min_normal_as_f64() -> f64 {
    f64::from_bits(((F64_EXPONENT_BIAS - 126) as u64) << F64_FRACTION_BITS)
}

/// Directed binary64-to-binary32 lower conversion that never publishes a
/// subnormal binary32 endpoint.
#[inline]
fn f64_to_f32_down_no_subnormal(value: f64) -> f32 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }

    let min_normal = binary32_min_normal_as_f64();
    if value.abs() < min_normal {
        return if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        };
    }

    let nearest = value as f32;
    if nearest == f32::INFINITY {
        return f32::MAX;
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

/// Directed binary64-to-binary32 upper conversion that never publishes a
/// subnormal binary32 endpoint.
#[inline]
fn f64_to_f32_up_no_subnormal(value: f64) -> f32 {
    if value.is_nan() || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }

    let min_normal = binary32_min_normal_as_f64();
    if value.abs() < min_normal {
        return if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        };
    }

    let nearest = value as f32;
    if nearest == f32::NEG_INFINITY {
        return f32::MIN;
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

/// Publish a nonnegative error carrier upward without a flushable subnormal.
#[inline]
pub(crate) fn publish_error_up_normal(value: f64) -> f32 {
    if value.is_nan() || value < 0.0 || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }
    next_up_f32(f64_to_f32_up_no_subnormal(value))
}

/// Decode nonnegative binary32 error metadata, poisoning invalid entries.
#[inline]
pub(crate) fn nonnegative_f32_error_or_infinity(value: f32) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude >> 23;
    if exponent == 0xff || (bits >> 31 != 0 && magnitude != 0) {
        f64::INFINITY
    } else {
        f32_to_f64_exact(value)
    }
}

/// Add one exact-real term to a binary64 lower-bound accumulator.
///
/// Merely casting the final binary64 sum outward to binary32 is insufficient
/// when cancellation has already discarded a small residual in binary64.
#[inline]
pub(crate) fn add_f64_down(acc: f64, term: f64) -> f64 {
    if acc.is_nan() || term.is_nan() {
        return f64::NEG_INFINITY;
    }
    if term == 0.0 {
        return acc;
    }
    let sum = acc + term;
    if sum.is_nan() {
        f64::NEG_INFINITY
    } else {
        next_down_f64(sum)
    }
}

/// Upper-bound counterpart of [`add_f64_down`].
#[inline]
pub(crate) fn add_f64_up(acc: f64, term: f64) -> f64 {
    if acc.is_nan() || term.is_nan() {
        return f64::INFINITY;
    }
    if term == 0.0 {
        return acc;
    }
    let sum = acc + term;
    if sum.is_nan() {
        f64::INFINITY
    } else {
        next_up_f64(sum)
    }
}

/// Add `coeff_err * |bias|` to a nonnegative upper-bound accumulator.
///
/// Coefficient errors are untrusted certificate metadata. A negative or
/// non-finite entry therefore poisons every nonzero use to `+inf`; a zero bias
/// has exactly zero contribution and is short-circuited before `0 * inf`.
#[inline]
pub(crate) fn add_coeff_err_bias_product_up(acc: f64, coeff_err: f32, bias: f32) -> f64 {
    let bias_abs = f32_to_f64_exact(bias).abs();
    if bias_abs == 0.0 {
        return acc;
    }
    let err = nonnegative_f32_error_or_infinity(coeff_err);
    add_f64_up(acc, err * bias_abs)
}

/// Parameters for one position block of bias accumulation.
pub(crate) struct BiasBlockParams {
    /// Number of output rows in the A matrix.
    pub num_outputs: usize,
    /// Number of features in this position block.
    pub out_features: usize,
    /// Column offset into the A matrix for this position.
    pub col_offset: usize,
}

/// Accumulate the bias contribution for one position block of the A matrix.
///
/// For each output row `i`, computes:
///   `accum.0[i] += sum_j A_lower[i, offset+j] * bias[j]`  (lower)
///   `accum.1[i] += sum_j A_upper[i, offset+j] * bias[j]`  (upper)
///
/// All arithmetic is in f64 to prevent catastrophic cancellation (#1863).
pub(crate) fn accumulate_bias_f64(
    accum: &mut (&mut [f64], &mut [f64]),
    lower_a_val: impl Fn(usize, usize) -> f32,
    upper_a_val: impl Fn(usize, usize) -> f32,
    bias: &Array1<f32>,
    block: &BiasBlockParams,
) {
    for i in 0..block.num_outputs {
        for j in 0..block.out_features {
            let col = block.col_offset + j;
            let bias_value = f32_to_f64_exact(bias[j]);
            let lower_term = f32_to_f64_exact(lower_a_val(i, col)) * bias_value;
            let upper_term = f32_to_f64_exact(upper_a_val(i, col)) * bias_value;
            accum.0[i] = add_f64_down(accum.0[i], lower_term);
            accum.1[i] = add_f64_up(accum.1[i], upper_term);
        }
    }
}

/// Finalize bias accumulators with directed rounding.
///
/// Adds the original bias vector (promoted to f64) to the accumulated
/// bias contributions, then casts back to f32 with directed rounding. The
/// publication step never emits a binary32 subnormal endpoint.
///
/// Callers bypass this function when the layer has no bias, avoiding needless
/// widening of an otherwise unchanged bias vector.
pub(crate) fn finalize_bias_directed(
    lower_accum: &Array1<f64>,
    upper_accum: &Array1<f64>,
    old_lower_b: &Array1<f32>,
    old_upper_b: &Array1<f32>,
) -> (Array1<f32>, Array1<f32>) {
    assert_eq!(lower_accum.len(), old_lower_b.len());
    assert_eq!(upper_accum.len(), old_upper_b.len());
    (
        Array1::from_shape_fn(old_lower_b.len(), |i| {
            f64_to_f32_down_no_subnormal(add_f64_down(
                f32_to_f64_exact(old_lower_b[i]),
                lower_accum[i],
            ))
        }),
        Array1::from_shape_fn(old_upper_b.len(), |i| {
            f64_to_f32_up_no_subnormal(add_f64_up(f32_to_f64_exact(old_upper_b[i]), upper_accum[i]))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_accumulate_bias_basic() {
        // A = [[1, 0], [0, 1]], bias = [2, 3]
        // lower_accum[0] += 1*2 + 0*3 = 2
        // lower_accum[1] += 0*2 + 1*3 = 3
        let mut lower_accum = [0.0_f64; 2];
        let mut upper_accum = [0.0_f64; 2];
        let bias = array![2.0_f32, 3.0];

        let a = [[1.0_f32, 0.0], [0.0_f32, 1.0]];
        let block = BiasBlockParams {
            num_outputs: 2,
            out_features: 2,
            col_offset: 0,
        };
        accumulate_bias_f64(
            &mut (&mut lower_accum[..], &mut upper_accum[..]),
            |i, j| a[i][j],
            |i, j| a[i][j],
            &bias,
            &block,
        );

        assert!((lower_accum[0] - 2.0).abs() < 1e-10);
        assert!((lower_accum[1] - 3.0).abs() < 1e-10);
        assert!((upper_accum[0] - 2.0).abs() < 1e-10);
        assert!((upper_accum[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_finalize_directed_rounding_widens() {
        // With non-zero bias contribution, directed rounding should widen bounds
        let lower_accum = array![1.0_f64];
        let upper_accum = array![1.0_f64];
        let old_lower_b = array![0.0_f32];
        let old_upper_b = array![0.0_f32];

        let (lb, ub) =
            finalize_bias_directed(&lower_accum, &upper_accum, &old_lower_b, &old_upper_b);

        // next_down_f32(1.0) < 1.0, next_up_f32(1.0) > 1.0
        assert!(lb[0] <= 1.0_f32, "lower should be <= 1.0, got {}", lb[0]);
        assert!(ub[0] >= 1.0_f32, "upper should be >= 1.0, got {}", ub[0]);
    }

    #[test]
    fn test_finalize_no_bias_exact() {
        // An exactly representable unchanged endpoint needs no widening.
        let lower_accum = array![0.0_f64];
        let upper_accum = array![0.0_f64];
        let old_lower_b = array![5.0_f32];
        let old_upper_b = array![5.0_f32];

        let (lb, ub) =
            finalize_bias_directed(&lower_accum, &upper_accum, &old_lower_b, &old_upper_b);

        assert!(lb[0] <= 5.0_f32, "lower must be <= 5.0, got {}", lb[0]);
        assert!(ub[0] >= 5.0_f32, "upper must be >= 5.0, got {}", ub[0]);
        assert_eq!(lb[0], 5.0);
        assert_eq!(ub[0], 5.0);
    }

    #[test]
    fn cancellation_residual_is_enclosed() {
        // In plain binary64 accumulation the middle term is discarded:
        // 2^32 + 2^-32 - 2^32 == 0. The exact-real result is 2^-32.
        let huge = 2.0_f32.powi(32);
        let tiny = 2.0_f32.powi(-32);
        let a = [[huge, tiny, -huge]];
        let bias = array![1.0_f32, 1.0, 1.0];
        let mut lower_accum = [0.0_f64; 1];
        let mut upper_accum = [0.0_f64; 1];
        accumulate_bias_f64(
            &mut (&mut lower_accum, &mut upper_accum),
            |i, j| a[i][j],
            |i, j| a[i][j],
            &bias,
            &BiasBlockParams {
                num_outputs: 1,
                out_features: 3,
                col_offset: 0,
            },
        );

        let (lower, upper) = finalize_bias_directed(
            &array![lower_accum[0]],
            &array![upper_accum[0]],
            &array![0.0_f32],
            &array![0.0_f32],
        );
        assert!(lower[0] <= tiny, "lower {} excluded {tiny}", lower[0]);
        assert!(upper[0] >= tiny, "upper {} excluded {tiny}", upper[0]);
    }

    /// Model a hardware conversion with DAZ enabled. The production code must
    /// not use this result; it exists to demonstrate that the regression inputs
    /// really do defeat an ordinary arithmetic conversion on a DAZ backend.
    fn simulated_daz_f32_to_f64(value: f32) -> f64 {
        let bits = value.to_bits();
        let exponent = (bits >> 23) & 0xff;
        let fraction = bits & 0x7f_ffff;
        if exponent == 0 && fraction != 0 {
            f64::from_bits(u64::from(bits >> 31) << 63)
        } else {
            f64::from(value)
        }
    }

    #[test]
    fn subnormal_coefficient_times_large_bias_survives_simulated_daz() {
        let coefficient = f32::from_bits(1); // 2^-149
        let large_bias = 2.0_f32.powi(120);
        let exact = 2.0_f64.powi(-29);
        assert_eq!(
            simulated_daz_f32_to_f64(coefficient) * f64::from(large_bias),
            0.0,
            "the regression must exercise a DAZ-sensitive source"
        );

        let mut lower_accum = [0.0_f64];
        let mut upper_accum = [0.0_f64];
        accumulate_bias_f64(
            &mut (&mut lower_accum, &mut upper_accum),
            |_i, _j| coefficient,
            |_i, _j| coefficient,
            &array![large_bias],
            &BiasBlockParams {
                num_outputs: 1,
                out_features: 1,
                col_offset: 0,
            },
        );
        let (lower, upper) = finalize_bias_directed(
            &array![lower_accum[0]],
            &array![upper_accum[0]],
            &array![0.0],
            &array![0.0],
        );

        assert!(
            f32_to_f64_exact(lower[0]) <= exact,
            "lower {} excludes exact {exact}",
            lower[0]
        );
        assert!(
            f32_to_f64_exact(upper[0]) >= exact,
            "upper {} excludes exact {exact}",
            upper[0]
        );
        assert!(lower[0].is_finite() && upper[0].is_finite());
    }

    #[test]
    fn subnormal_coefficient_error_penalty_survives_simulated_daz() {
        let coefficient_error = f32::from_bits(1); // 2^-149
        let large_bias = 2.0_f32.powi(120);
        let exact_penalty = 2.0_f64.powi(-29);
        assert_eq!(
            simulated_daz_f32_to_f64(coefficient_error) * f64::from(large_bias),
            0.0,
            "the regression must exercise a DAZ-sensitive error source"
        );

        let penalty = add_coeff_err_bias_product_up(0.0, coefficient_error, large_bias);
        let lower_accum = add_f64_down(0.0, -penalty);
        let upper_accum = add_f64_up(0.0, penalty);
        let (lower, upper) = finalize_bias_directed(
            &array![lower_accum],
            &array![upper_accum],
            &array![0.0],
            &array![0.0],
        );

        assert!(
            f32_to_f64_exact(lower[0]) <= -exact_penalty,
            "lower {} omitted -coeff_err*|bias|={}",
            lower[0],
            -exact_penalty
        );
        assert!(
            f32_to_f64_exact(upper[0]) >= exact_penalty,
            "upper {} omitted coeff_err*|bias|={exact_penalty}",
            upper[0]
        );
    }

    #[test]
    fn negative_subnormal_coefficient_error_poison_is_daz_independent() {
        let invalid_error = f32::from_bits(0x8000_0001);
        assert_eq!(
            simulated_daz_f32_to_f64(invalid_error),
            -0.0,
            "a comparison-based sanitizer could mistake this for valid -0 under DAZ"
        );
        assert_eq!(
            add_coeff_err_bias_product_up(0.0, invalid_error, 1.0),
            f64::INFINITY,
            "an illegal negative error must fail closed"
        );
        assert_eq!(
            add_coeff_err_bias_product_up(0.0, invalid_error, 0.0),
            0.0,
            "an exactly zero bias has no error contribution"
        );
    }

    #[test]
    fn publication_never_emits_subnormal_endpoints() {
        let coefficient = f32::from_bits(1);
        let mut lower_accum = [0.0_f64];
        let mut upper_accum = [0.0_f64];
        accumulate_bias_f64(
            &mut (&mut lower_accum, &mut upper_accum),
            |_i, _j| coefficient,
            |_i, _j| coefficient,
            &array![1.0],
            &BiasBlockParams {
                num_outputs: 1,
                out_features: 1,
                col_offset: 0,
            },
        );
        let (lower, upper) = finalize_bias_directed(
            &array![lower_accum[0]],
            &array![upper_accum[0]],
            &array![0.0],
            &array![0.0],
        );

        let is_subnormal = |value: f32| {
            let magnitude = value.to_bits() & 0x7fff_ffff;
            magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits()
        };
        assert!(!is_subnormal(lower[0]), "lower={}", lower[0]);
        assert!(!is_subnormal(upper[0]), "upper={}", upper[0]);
        let exact = f32_to_f64_exact(coefficient);
        assert!(f32_to_f64_exact(lower[0]) <= exact);
        assert!(f32_to_f64_exact(upper[0]) >= exact);
    }
}
