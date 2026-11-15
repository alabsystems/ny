// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Closed binary32 head contracts, representation admission and error enclosure.
//!
//! These contracts describe a specified head evaluation on finite binary32
//! hidden inputs. They do not qualify an unknown BLAS, Torch or device kernel.
//! The coefficients remain their original binary64 values; admission checks
//! their exact binary32 representation using integer bits, without a host cast.
use super::numeric::{up_add, up_mul};
use super::{DecoderRefusal, HeadArithmetic, HiddenBox};
use crate::zono_core::rounding::{next_down, next_up};
use serde::{Deserialize, Serialize};

/// All modes use round-to-nearest, ties-to-even, gradual binary32 subnormals,
/// and refuse possible overflow. No mode permits FTZ, DAZ, TF32, quantization,
/// soft caps, omitted terms or logit processors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Binary32Reduction {
    /// Start with the original bias; in ascending coordinate order round the
    /// product to binary32, then separately round its addition to the accumulator.
    SequentialSeparate,
    /// Round every original product separately; use products and the original
    /// bias as leaves in a contiguous recursive midpoint-split addition tree.
    BalancedPairwiseSeparate,
    /// Start with the original bias; in ascending order correctly round each
    /// fused `weight[k] * hidden[k] + accumulator` only once.
    SequentialFma,
    /// A read-once reduction of every original product and the bias exactly
    /// once. Internal nodes add two disjoint partial sums or fuse an original
    /// product with a disjoint partial sum. Every operation is correctly rounded.
    /// This family includes the three concrete modes above. It is an explicit
    /// grammar, not authority for an arbitrary implementation of a dot product.
    ReadOnceTreeFamily,
}

impl Binary32Reduction {
    pub fn contract_name(self) -> &'static str {
        match self {
            Self::SequentialSeparate => "binary32-sequential-separate/v1",
            Self::BalancedPairwiseSeparate => "binary32-balanced-pairwise-separate/v1",
            Self::SequentialFma => "binary32-sequential-fma/v1",
            Self::ReadOnceTreeFamily => "binary32-read-once-tree-family/v1",
        }
    }

    pub(super) fn contract_bytes(self) -> &'static [u8] {
        match self {
            Self::SequentialSeparate => b"finite-binary32-coefficients-and-inputs;bias-first;ascending;separate-multiply-add;rn-even;gradual-underflow;overflow-refused/v1",
            Self::BalancedPairwiseSeparate => b"finite-binary32-coefficients-and-inputs;rounded-product-leaves-then-bias;contiguous-midpoint-split-add-tree;rn-even;gradual-underflow;overflow-refused/v1",
            Self::SequentialFma => b"finite-binary32-coefficients-and-inputs;bias-first;ascending;correctly-rounded-fma;rn-even;gradual-underflow;overflow-refused/v1",
            Self::ReadOnceTreeFamily => b"finite-binary32-coefficients-and-inputs;each-original-product-and-bias-exactly-once;disjoint-add-or-original-product-fma-tree;rn-even;gradual-underflow;overflow-refused/v1",
        }
    }
}

pub(super) const BINARY32_INPUT_ASSUMPTIONS: &[u8] = b"caller-independently-enclosed-hidden-state;hidden-inputs-are-finite-binary32-values-in-box;no-logit-processors/v1";

// Exact binary64 encodings of 2^-149 and the largest finite binary32 value.
pub(super) const ETA32: f64 = f64::from_bits(874u64 << 52);
pub(super) const MAX32: f64 = f64::from_bits((1150u64 << 52) | (0x7f_ffffu64 << 29));
const U32: f64 = f64::from_bits(999u64 << 52); // 2^-24

/// Exact widening by bits, including binary32 subnormals on a DAZ host.
pub(super) fn binary32_value(bits: u32) -> f64 {
    let sign = ((bits as u64) & 0x8000_0000) << 32;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    if exponent == 0 {
        if fraction == 0 {
            return f64::from_bits(sign);
        }
        let leading = fraction.ilog2();
        let exp64 = (leading as i32 - 149 + 1023) as u64;
        let mantissa = ((fraction as u64) << (52 - leading)) & ((1u64 << 52) - 1);
        return f64::from_bits(sign | (exp64 << 52) | mantissa);
    }
    f64::from_bits(sign | (((exponent as u64) + 896) << 52) | ((fraction as u64) << 29))
}

/// Returns an exact finite binary32 encoding, with no narrowing conversion.
pub(super) fn binary32_bits(value: f64) -> Option<u32> {
    if !value.is_finite() || value.abs() > MAX32 {
        return None;
    }
    let sign = ((value.to_bits() >> 32) & 0x8000_0000) as u32;
    if value == 0.0 {
        return Some(sign);
    }
    let raw_exp = ((value.to_bits() >> 52) & 0x7ff) as i32;
    let exponent = raw_exp - 1023;
    if raw_exp == 0 || exponent < -149 {
        return None;
    }
    let sig = (1u64 << 52) | (value.to_bits() & ((1u64 << 52) - 1));
    if exponent >= -126 {
        if sig & ((1u64 << 29) - 1) != 0 {
            return None;
        }
        Some(sign | (((exponent + 127) as u32) << 23) | ((sig >> 29) as u32 & 0x7f_ffff))
    } else {
        let shift = (-exponent - 97) as u32;
        if sig & ((1u64 << shift) - 1) != 0 {
            return None;
        }
        Some(sign | (sig >> shift) as u32)
    }
}

/// Positive finite magnitude rounded toward zero to a binary32 bit pattern;
/// the boolean indicates that the original magnitude was between grid points.
fn floor_magnitude(value: f64) -> (u32, bool) {
    if value == 0.0 {
        return (0, false);
    }
    if value < ETA32 {
        return (0, true);
    }
    let exponent = ((value.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let sig = (1u64 << 52) | (value.to_bits() & ((1u64 << 52) - 1));
    let shift = if exponent >= -126 {
        29
    } else {
        (-exponent - 97) as u32
    };
    let remainder = sig & ((1u64 << shift) - 1) != 0;
    let bits = if exponent >= -126 {
        (((exponent + 127) as u32) << 23) | ((sig >> shift) as u32 & 0x7f_ffff)
    } else {
        (sig >> shift) as u32
    };
    (bits, remainder)
}

fn finite_ceiling(value: f64) -> Option<f64> {
    if value > MAX32 {
        return None;
    }
    if value <= -MAX32 {
        return Some(-MAX32);
    }
    let (bits, remainder) = floor_magnitude(value.abs());
    if value < 0.0 {
        Some(binary32_value(bits | 0x8000_0000))
    } else {
        Some(binary32_value(bits + u32::from(remainder)))
    }
}

/// The enclosure is conditional on finite binary32 inputs, not every real
/// point in this box. Refuse a coordinate with no input in that domain so an
/// unrepresentable singleton cannot yield a vacuous certificate.
pub(super) fn validate_input_domain(
    arithmetic: HeadArithmetic,
    hidden: HiddenBox<'_>,
) -> Result<(), DecoderRefusal> {
    if !matches!(arithmetic, HeadArithmetic::Binary32 { .. }) {
        return Ok(());
    }
    if hidden.lo.len() != hidden.hi.len() {
        return Err(DecoderRefusal::Shape("binary32 hidden box"));
    }
    for (&lo, &hi) in hidden.lo.iter().zip(hidden.hi) {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(DecoderRefusal::NonFinite("binary32 hidden endpoints"));
        }
        if !finite_ceiling(lo).is_some_and(|value| value <= hi) {
            return Err(DecoderRefusal::Shape(
                "hidden box has no finite binary32 input",
            ));
        }
    }
    Ok(())
}

/// Uniform read-once-family enclosure. At most `2*dim+1` rounded operations
/// contribute relative error `u` and absolute underflow error at most `eta`.
/// Repeated substitution along any reduction path gives
/// `gamma_N*A + N*eta/(1-N*u)`, with A the original absolute term sum.
/// FMA and balanced trees share this deliberately conservative bound; tighter
/// operation/depth bounds require a separately checked refinement.
pub(super) fn binary32_error(absolute: f64, dim: usize) -> Result<f64, DecoderRefusal> {
    let count = dim
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(DecoderRefusal::Resource("binary32 operation count"))?;
    // This also ensures the count's conversion to binary64 is exact.
    if count >= 1 << 23 {
        return Err(DecoderRefusal::Resource("binary32 error theorem dimension"));
    }
    if !absolute.is_finite() || absolute < 0.0 {
        return Err(DecoderRefusal::NonFinite("binary32 absolute term sum"));
    }
    let nu = up_mul(count as f64, U32)?;
    let denominator = next_down(1.0 - nu);
    let gamma = next_up(nu / denominator);
    let underflow = next_up(up_mul(count as f64, ETA32)? / denominator);
    let error = up_add(up_mul(gamma, absolute)?, underflow)?;
    // Bounds every product and every partial sum, including cancellation cases.
    if up_add(absolute, error)? > MAX32 {
        return Err(DecoderRefusal::NonFinite("possible binary32 head overflow"));
    }
    Ok(error)
}

#[cfg(test)]
mod tests;
