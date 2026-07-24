// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared McCormick bilinear accumulation for decomposed normalization CROWN.
//!
//! Part of #3911.

use ny_tensor::{next_down_f32, next_up_f32};

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the helper mirrors one McCormick plane-selection decision with the primary and auxiliary interval endpoints kept explicit for auditability"
)]
fn select_lower_plane(
    weight: f32,
    l1_val: f32,
    l2_val: f32,
    u1_val: f32,
    u2_val: f32,
    primary_lower: f32,
    primary_upper: f32,
    aux_lower: f32,
    aux_upper: f32,
) -> (f32, f32, f32) {
    if weight > 0.0 {
        if l1_val >= l2_val {
            (aux_lower, primary_lower, -(primary_lower * aux_lower))
        } else {
            (aux_upper, primary_upper, -(primary_upper * aux_upper))
        }
    } else if weight < 0.0 {
        if u1_val <= u2_val {
            (aux_upper, primary_lower, -(primary_lower * aux_upper))
        } else {
            (aux_lower, primary_upper, -(primary_upper * aux_lower))
        }
    } else {
        (0.0, 0.0, 0.0)
    }
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the helper mirrors one McCormick plane-selection decision with the primary and auxiliary interval endpoints kept explicit for auditability"
)]
fn select_upper_plane(
    weight: f32,
    l1_val: f32,
    l2_val: f32,
    u1_val: f32,
    u2_val: f32,
    primary_lower: f32,
    primary_upper: f32,
    aux_lower: f32,
    aux_upper: f32,
) -> (f32, f32, f32) {
    if weight > 0.0 {
        if u1_val <= u2_val {
            (aux_upper, primary_lower, -(primary_lower * aux_upper))
        } else {
            (aux_lower, primary_upper, -(primary_upper * aux_lower))
        }
    } else if weight < 0.0 {
        if l1_val >= l2_val {
            (aux_lower, primary_lower, -(primary_lower * aux_lower))
        } else {
            (aux_upper, primary_upper, -(primary_upper * aux_upper))
        }
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Accumulate one McCormick-relaxed bilinear term into caller-owned row state.
///
/// The helper preserves the existing rounding boundary by rounding the
/// primary-variable contribution exactly when it is merged into the f64
/// accumulator, while keeping auxiliary-coefficient and bias accumulation in
/// f64 for the caller's later cast/composition step.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the shared bilinear accumulator keeps the caller-owned row slots explicit so the decomposition math and directed-rounding merge points stay visible"
)]
pub(crate) fn accumulate_mccormick_bilinear_term(
    lower_weight: f32,
    upper_weight: f32,
    primary_lower: f32,
    primary_upper: f32,
    aux_lower: f32,
    aux_upper: f32,
    lower_primary_slot: &mut f64,
    upper_primary_slot: &mut f64,
    lower_aux_accum: &mut f64,
    upper_aux_accum: &mut f64,
    lower_bias: &mut f64,
    upper_bias: &mut f64,
) -> (bool, bool) {
    if !primary_lower.is_finite()
        || !primary_upper.is_finite()
        || !aux_lower.is_finite()
        || !aux_upper.is_finite()
    {
        return (true, true);
    }

    let primary_mid = primary_lower * 0.5 + primary_upper * 0.5;
    let aux_mid = aux_lower * 0.5 + aux_upper * 0.5;

    let l1_val = aux_lower * primary_mid + primary_lower * aux_mid - primary_lower * aux_lower;
    let l2_val = aux_upper * primary_mid + primary_upper * aux_mid - primary_upper * aux_upper;
    let u1_val = aux_upper * primary_mid + primary_lower * aux_mid - primary_lower * aux_upper;
    let u2_val = aux_lower * primary_mid + primary_upper * aux_mid - primary_upper * aux_lower;

    let (lower_primary_coeff, lower_aux_coeff, lower_const) = select_lower_plane(
        lower_weight,
        l1_val,
        l2_val,
        u1_val,
        u2_val,
        primary_lower,
        primary_upper,
        aux_lower,
        aux_upper,
    );
    let lower_product = lower_weight * lower_primary_coeff;
    let lower_nonfinite = !lower_product.is_finite();
    if !lower_nonfinite {
        *lower_primary_slot += next_down_f32(lower_product) as f64;
    }
    *lower_aux_accum += lower_weight as f64 * lower_aux_coeff as f64;
    *lower_bias += lower_weight as f64 * lower_const as f64;

    let (upper_primary_coeff, upper_aux_coeff, upper_const) = select_upper_plane(
        upper_weight,
        l1_val,
        l2_val,
        u1_val,
        u2_val,
        primary_lower,
        primary_upper,
        aux_lower,
        aux_upper,
    );
    let upper_product = upper_weight * upper_primary_coeff;
    let upper_nonfinite = !upper_product.is_finite();
    if !upper_nonfinite {
        *upper_primary_slot += next_up_f32(upper_product) as f64;
    }
    *upper_aux_accum += upper_weight as f64 * upper_aux_coeff as f64;
    *upper_bias += upper_weight as f64 * upper_const as f64;

    (lower_nonfinite, upper_nonfinite)
}
