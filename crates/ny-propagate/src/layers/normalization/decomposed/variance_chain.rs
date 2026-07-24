// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared reciprocal -> sqrt -> mean -> square accumulation for normalization.
//!
//! Part of #3911.

use ny_tensor::{next_down_f32, next_up_f32};

use crate::layers::activations::LinearRelaxation;
use crate::layers::arithmetic::pow2_linear_relaxation;
use crate::layers::common::compose::{compose_lower, compose_upper};

/// Accumulate the shared variance-chain contribution into caller-owned row state.
///
/// The helper keeps the existing rounding contract:
/// - the auxiliary coefficient is cast to `f32` exactly once before reciprocal
///   and sqrt composition
/// - square-path coefficients are rounded when each composition is merged into
///   the caller-owned f64 per-element accumulator
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the helper threads the reciprocal/sqrt relaxations plus caller-owned accumulators explicitly so the variance-chain merge points remain auditable"
)]
pub(crate) fn accumulate_variance_chain(
    lower_aux_coeff: f64,
    upper_aux_coeff: f64,
    recip_relax: &LinearRelaxation,
    sqrt_relax: &LinearRelaxation,
    primary_lower: &[f32],
    primary_upper: &[f32],
    norm_size: usize,
    eps: f32,
    lower_primary_accum: &mut [f64],
    upper_primary_accum: &mut [f64],
    lower_bias: &mut f64,
    upper_bias: &mut f64,
) -> (bool, bool) {
    debug_assert_eq!(primary_lower.len(), norm_size);
    debug_assert_eq!(primary_upper.len(), norm_size);
    debug_assert_eq!(lower_primary_accum.len(), norm_size);
    debug_assert_eq!(upper_primary_accum.len(), norm_size);

    let lower_aux_coeff_f32 = next_down_f32(lower_aux_coeff as f32);
    let upper_aux_coeff_f32 = next_up_f32(upper_aux_coeff as f32);

    let recip_l = compose_lower(lower_aux_coeff_f32, recip_relax);
    let recip_u = compose_upper(upper_aux_coeff_f32, recip_relax);
    let mut lower_nonfinite = recip_l.nonfinite;
    let mut upper_nonfinite = recip_u.nonfinite;
    *lower_bias += recip_l.intercept_contrib;
    *upper_bias += recip_u.intercept_contrib;

    let sqrt_l = compose_lower(recip_l.new_coeff, sqrt_relax);
    let sqrt_u = compose_upper(recip_u.new_coeff, sqrt_relax);
    lower_nonfinite |= sqrt_l.nonfinite;
    upper_nonfinite |= sqrt_u.nonfinite;
    *lower_bias += sqrt_l.intercept_contrib;
    *upper_bias += sqrt_u.intercept_contrib;

    // The post-sqrt coefficient multiplies s = mean(x^2) + eps, but the square
    // path below only re-expresses the mean(x^2) term (sqrt_*.new_coeff / nf fed
    // into pow2). The constant +eps offset of s must be accumulated into the
    // bias here, or the bound is too tight by sqrt_*.new_coeff * eps. Computed in
    // f64 from the already directed-rounded f32 `new_coeff`; finalize_decomposed_
    // norm_bounds casts the f64 biases with next_down_f32 (lower) / next_up_f32
    // (upper), matching the intercept_contrib rounding contract above.
    *lower_bias += sqrt_l.new_coeff as f64 * eps as f64;
    *upper_bias += sqrt_u.new_coeff as f64 * eps as f64;

    let nf = norm_size as f32;
    let lower_square_coeff = next_down_f32(sqrt_l.new_coeff / nf);
    let upper_square_coeff = next_up_f32(sqrt_u.new_coeff / nf);

    for i in 0..norm_size {
        let sq_relax = pow2_linear_relaxation(primary_lower[i], primary_upper[i]);

        let sq_l = compose_lower(lower_square_coeff, &sq_relax);
        lower_nonfinite |= sq_l.nonfinite;
        *lower_bias += sq_l.intercept_contrib;
        lower_primary_accum[i] += sq_l.new_coeff as f64;

        let sq_u = compose_upper(upper_square_coeff, &sq_relax);
        upper_nonfinite |= sq_u.nonfinite;
        *upper_bias += sq_u.intercept_contrib;
        upper_primary_accum[i] += sq_u.new_coeff as f64;
    }

    (lower_nonfinite, upper_nonfinite)
}
