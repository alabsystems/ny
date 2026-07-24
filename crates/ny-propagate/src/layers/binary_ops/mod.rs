// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Binary Ops layers for bound propagation.

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

/// Maximum safe magnitude for McCormick plane inputs.
///
/// McCormick computes products like lx*ly, ux*uy. If |bound| > sqrt(f32::MAX)
/// ≈ 1.84e19, these products overflow to infinity and cause NaN in subsequent
/// arithmetic.
pub(crate) const MCCORMICK_MAX_MAGNITUDE: f32 = 1.84e19;

/// Check whether any element in `bounds` is infinite, NaN, or exceeds the
/// McCormick overflow threshold.
fn has_bad_mccormick_values(bounds: &BoundedTensor) -> bool {
    bounds
        .lower()
        .iter()
        .chain(bounds.upper().iter())
        .any(|&v| v.is_infinite() || v.is_nan() || v.abs() > MCCORMICK_MAX_MAGNITUDE)
}

/// Validate that both input bound tensors are safe for McCormick CROWN.
///
/// Returns `Ok(())` if all elements are finite and within the overflow
/// threshold. Returns an error with `context` in the message otherwise.
pub(crate) fn validate_mccormick_inputs(
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    context: &str,
) -> Result<()> {
    if has_bad_mccormick_values(input_a_bounds) || has_bad_mccormick_values(input_b_bounds) {
        return Err(NyError::UnsupportedOp(format!(
            "{context} McCormick CROWN requires bounded inputs; input bounds are infinite, NaN, or exceed overflow threshold"
        )));
    }
    Ok(())
}

mod add;
mod atan2;
mod atan2_relax;
mod bilinear;
mod compare_tensor;
mod concat;
mod div;
mod elementwise;
mod matmul;
mod max;
mod min;
mod minmax_relax;
mod mul;
mod sub;

pub use add::AddLayer;
pub use atan2::Atan2Layer;
pub use bilinear::BilinearCrownLayer;
pub use compare_tensor::CompareTensorLayer;
pub use concat::ConcatLayer;
pub use div::DivLayer;
pub use matmul::{MatMulIbpMode, MatMulLayer};
pub use max::MaxBinaryLayer;
pub use min::MinBinaryLayer;
pub use mul::MulBinaryLayer;
pub use sub::SubLayer;
