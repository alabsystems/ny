// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Owned, deadline-aware 7D Patches alpha-ReLU backward.
//!
//! This is a deliberately narrow companion to `crown_patches_alpha`: callers
//! must first prepare and validate an explicit-row carrier while it is still
//! borrowed, then transfer sole ownership for the in-place transform.  After
//! ownership transfer the only recoverable failure is deadline expiry; callers
//! must propagate that error upward rather than expose the partially transformed
//! carrier to a Dense or historical Patches fallback.

use std::time::Instant;

use ndarray::Array1;
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::compose;
use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::layers::activations::{relu_crossing_upper_chord, LinearRelaxation};

const EXPLICIT_ROW_ALPHA_DEADLINE_POLL_COORDS: usize = 4_096;
const PADDING_TAP: u32 = u32::MAX;

/// Fully validated immutable state for one owned in-place alpha-ReLU transform.
///
/// Constructing this plan performs every fallible shape, layout, alpha, carried
/// error, and coordinate-overflow check before the caller transfers ownership
/// of the Patches carrier.
pub(crate) struct PreparedAlphaPatchesReluInPlace {
    deadline: Instant,
    row_count: usize,
    row_volume: usize,
    tap_inputs: Vec<u32>,
    relaxations: Vec<LinearRelaxation>,
    max_slope_sum: f64,
    intercept_sum: f64,
    gamma_bar: f64,
}

#[inline]
fn check_alpha_patches_deadline(deadline: Instant, phase: &str) -> Result<()> {
    if Instant::now() >= deadline {
        Err(NyError::DeadlineExceeded(format!(
            "alpha-ReLU Patches in-place backward: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

#[inline(always)]
fn poll_coordinate<P>(coordinates_since_poll: &mut usize, poll: &P) -> Result<()>
where
    P: Fn() -> Result<()>,
{
    *coordinates_since_poll += 1;
    if *coordinates_since_poll >= EXPLICIT_ROW_ALPHA_DEADLINE_POLL_COORDS {
        poll()?;
        *coordinates_since_poll = 0;
    }
    Ok(())
}

fn alpha_relaxations(
    pre_lower: &[f32],
    pre_upper: &[f32],
    alpha: &Array1<f32>,
    deadline: Instant,
) -> Result<Vec<LinearRelaxation>> {
    let mut relaxations = Vec::with_capacity(pre_lower.len());
    let mut coordinates_since_poll = 0usize;
    let relaxation_poll =
        || check_alpha_patches_deadline(deadline, "during relaxation preparation");
    for (i, (&l, &u)) in pre_lower.iter().zip(pre_upper.iter()).enumerate() {
        poll_coordinate(&mut coordinates_since_poll, &relaxation_poll)?;
        let relaxation = if l.is_nan() || u.is_nan() {
            LinearRelaxation::new(0.0, 0.0, 0.0, f32::INFINITY)
        } else if l >= 0.0 {
            LinearRelaxation::identity()
        } else if u <= 0.0 {
            LinearRelaxation::zero()
        } else if l.is_infinite() && u.is_infinite() {
            LinearRelaxation::new(alpha[i], 0.0, 0.0, f32::INFINITY)
        } else if u.is_infinite() {
            LinearRelaxation::new(alpha[i], 0.0, 1.0, -l)
        } else if l.is_infinite() {
            LinearRelaxation::new(alpha[i], 0.0, 0.0, u)
        } else {
            let (lambda, lambda_intercept) = relu_crossing_upper_chord(l, u, None);
            LinearRelaxation::new(alpha[i], 0.0, lambda, lambda_intercept)
        };
        relaxations.push(relaxation);
    }
    relaxation_poll()?;
    Ok(relaxations)
}

/// Prevalidate and prepare the exact explicit-row carrier admitted by the
/// default-dark owned alpha-ReLU route.
pub(crate) fn prepare_crown_relu_backward_patches_with_alpha_in_place(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    alpha: &Array1<f32>,
    deadline: Instant,
) -> Result<PreparedAlphaPatchesReluInPlace> {
    check_alpha_patches_deadline(deadline, "before preparation")?;
    super::non_finite_domain_guard("ReLU-alpha-patches", pre_activation)?;
    bounds.lower_a.validate_common_geometry(&bounds.upper_a)?;
    let affine_geometry = bounds
        .lower_a
        .geometry
        .require_affine("alpha-ReLU Patches in-place")?;

    if bounds.lower_a.identity
        || bounds.upper_a.identity
        || bounds.lower_a.unstable_idx.is_some()
        || bounds.upper_a.unstable_idx.is_some()
    {
        return Err(NyError::UnsupportedConfiguration(
            "alpha-ReLU Patches in-place requires materialized dense explicit rows".into(),
        ));
    }

    let lower = bounds.lower_a.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("Non-identity lower PatchesData has no patches tensor".into())
    })?;
    let upper = bounds.upper_a.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("Non-identity upper PatchesData has no patches tensor".into())
    })?;
    if lower.ndim() != 7 {
        return Err(NyError::ShapeMismatch {
            expected: vec![7],
            got: vec![lower.ndim()],
        });
    }
    if upper.shape() != lower.shape() {
        return Err(NyError::ShapeMismatch {
            expected: lower.shape().to_vec(),
            got: upper.shape().to_vec(),
        });
    }
    if lower.as_slice().is_none() || upper.as_slice().is_none() {
        return Err(NyError::UnsupportedConfiguration(
            "alpha-ReLU Patches in-place requires contiguous explicit rows".into(),
        ));
    }

    let (out_c, out_h, out_w) = bounds.lower_a.output_shape;
    let (in_c, in_h, in_w) = bounds.lower_a.input_shape;
    let shape = lower.shape();
    let expected_shape = [
        bounds.row_count,
        out_c,
        out_h,
        out_w,
        in_c,
        shape[5],
        shape[6],
    ];
    if shape != expected_shape.as_slice() {
        return Err(NyError::ShapeMismatch {
            expected: expected_shape.to_vec(),
            got: shape.to_vec(),
        });
    }
    if bounds.lower_b.len() != bounds.row_count || bounds.upper_b.len() != bounds.row_count {
        return Err(NyError::ShapeMismatch {
            expected: vec![bounds.row_count, bounds.row_count],
            got: vec![bounds.lower_b.len(), bounds.upper_b.len()],
        });
    }
    if bounds.lower_b.as_slice().is_none() || bounds.upper_b.as_slice().is_none() {
        return Err(NyError::UnsupportedConfiguration(
            "alpha-ReLU Patches in-place requires contiguous bias rows".into(),
        ));
    }
    for err in [
        bounds.lower_a.coeff_err.as_ref(),
        bounds.upper_a.coeff_err.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if err.len() != bounds.row_count {
            return Err(NyError::ShapeMismatch {
                expected: vec![bounds.row_count],
                got: vec![err.len()],
            });
        }
    }

    let num_input_neurons = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
        NyError::InvalidSpec("alpha-ReLU Patches in-place input-neuron count overflow".into())
    })?;
    if num_input_neurons > u32::MAX as usize {
        return Err(NyError::UnsupportedConfiguration(format!(
            "alpha-ReLU Patches in-place input-neuron count {num_input_neurons} exceeds the compact tap-map limit"
        )));
    }
    let pre_flat = pre_activation.flatten();
    let pre_lower = pre_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.lower().shape().to_vec(),
        })?;
    let pre_upper = pre_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.upper().shape().to_vec(),
        })?;
    if pre_lower.len() != num_input_neurons || pre_upper.len() != num_input_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_input_neurons, num_input_neurons],
            got: vec![pre_lower.len(), pre_upper.len()],
        });
    }
    if alpha.len() != num_input_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_input_neurons],
            got: vec![alpha.len()],
        });
    }
    let mut alpha_coordinates_since_poll = 0usize;
    let alpha_validation_poll =
        || check_alpha_patches_deadline(deadline, "during alpha validation");
    for (index, &value) in alpha.iter().enumerate() {
        poll_coordinate(&mut alpha_coordinates_since_poll, &alpha_validation_poll)?;
        if !(0.0..=1.0).contains(&value) {
            return Err(NyError::InvalidSpec(format!(
                "alpha-ReLU Patches in-place requires finite alpha in [0,1], got alpha[{index}]={value}"
            )));
        }
    }
    alpha_validation_poll()?;

    let row_volume = checked_shape_product(&shape[1..]).ok_or_else(|| {
        NyError::InvalidSpec("alpha-ReLU Patches in-place row-volume overflow".into())
    })?;
    if row_volume >= (1usize << 28) {
        return Err(NyError::UnsupportedConfiguration(format!(
            "alpha-ReLU Patches in-place row volume {row_volume} breaches the n < 2^28 certificate bound"
        )));
    }
    let expected_len = bounds.row_count.checked_mul(row_volume).ok_or_else(|| {
        NyError::InvalidSpec("alpha-ReLU Patches in-place coefficient count overflow".into())
    })?;
    if lower.len() != expected_len || upper.len() != expected_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_len, expected_len],
            got: vec![lower.len(), upper.len()],
        });
    }

    let pre_lower_slice = pre_lower
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_lower array".into()))?;
    let pre_upper_slice = pre_upper
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_upper array".into()))?;
    let relaxations = alpha_relaxations(pre_lower_slice, pre_upper_slice, alpha, deadline)?;

    // This map is row-invariant. Building it before ownership transfer makes
    // every coordinate calculation and overflow check fallible here, while the
    // hot transform becomes one flat serial pass per row.
    let (sh, sw) = affine_geometry.stride();
    let (pad_left, _pad_right, pad_top, _pad_bottom) = affine_geometry.padding();
    let kh = shape[5];
    let kw = shape[6];
    let mut tap_inputs = Vec::with_capacity(row_volume);
    let mut max_slope_sum = 0.0f64;
    let mut intercept_sum = 0.0f64;
    let mut coordinates_since_poll = 0usize;
    let preparation_poll =
        || check_alpha_patches_deadline(deadline, "during coordinate preparation");
    for _oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            poll_coordinate(&mut coordinates_since_poll, &preparation_poll)?;
                            let ih_base = oh
                                .checked_mul(sh)
                                .and_then(|value| value.checked_add(ki))
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "alpha-ReLU Patches in-place height coordinate overflow"
                                            .into(),
                                    )
                                })?;
                            let iw_base = ow
                                .checked_mul(sw)
                                .and_then(|value| value.checked_add(kj))
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "alpha-ReLU Patches in-place width coordinate overflow"
                                            .into(),
                                    )
                                })?;
                            if ih_base < pad_top || iw_base < pad_left {
                                tap_inputs.push(PADDING_TAP);
                                continue;
                            }
                            let ih = ih_base - pad_top;
                            let iw = iw_base - pad_left;
                            if ih >= in_h || iw >= in_w {
                                tap_inputs.push(PADDING_TAP);
                                continue;
                            }
                            let input_flat = ic
                                .checked_mul(in_h)
                                .and_then(|value| value.checked_mul(in_w))
                                .and_then(|value| {
                                    ih.checked_mul(in_w)
                                        .and_then(|offset| value.checked_add(offset))
                                })
                                .and_then(|value| value.checked_add(iw))
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "alpha-ReLU Patches in-place input index overflow".into(),
                                    )
                                })?;
                            if input_flat >= relaxations.len() {
                                return Err(NyError::ShapeMismatch {
                                    expected: vec![relaxations.len()],
                                    got: vec![input_flat.saturating_add(1)],
                                });
                            }
                            tap_inputs.push(u32::try_from(input_flat).map_err(|_| {
                                NyError::InvalidSpec(format!(
                                    "alpha-ReLU Patches in-place input index {input_flat} exceeds the compact tap-map limit"
                                ))
                            })?);
                            let relax = &relaxations[input_flat];
                            let slope_sum = f64::from(relax.lower_slope).abs()
                                + f64::from(relax.upper_slope).abs();
                            if slope_sum > max_slope_sum {
                                max_slope_sum = slope_sum;
                            }
                            intercept_sum += f64::from(relax.lower_intercept).abs()
                                + f64::from(relax.upper_intercept).abs();
                        }
                    }
                }
            }
        }
    }
    if tap_inputs.len() != row_volume {
        return Err(NyError::ShapeMismatch {
            expected: vec![row_volume],
            got: vec![tap_inputs.len()],
        });
    }
    check_alpha_patches_deadline(deadline, "after preparation")?;

    let gamma_bar = crate::layers::linear::crown_single_gamma_n_f64(
        row_volume.saturating_mul(8).saturating_add(16),
    );
    Ok(PreparedAlphaPatchesReluInPlace {
        deadline,
        row_count: bounds.row_count,
        row_volume,
        tap_inputs,
        relaxations,
        max_slope_sum,
        intercept_sum,
        gamma_bar,
    })
}

fn zero_row_with_poll<P>(row: &mut [f32], poll: &P) -> Result<()>
where
    P: Fn() -> Result<()>,
{
    let mut coordinates_since_poll = 0usize;
    for value in row {
        poll_coordinate(&mut coordinates_since_poll, poll)?;
        *value = 0.0;
    }
    poll()
}

fn crown_relu_backward_patches_with_alpha_in_place_impl<P>(
    mut bounds: Box<PatchesLinearBounds>,
    prepared: PreparedAlphaPatchesReluInPlace,
    pre_activation: &BoundedTensor,
    explicit_eager_7d_policy: Option<bool>,
    poll: &P,
) -> Result<CrownBounds>
where
    P: Fn() -> Result<()>,
{
    poll()?;
    debug_assert_eq!(bounds.row_count, prepared.row_count);
    let PatchesLinearBounds {
        row_count: _,
        lower_a,
        lower_b,
        upper_a,
        upper_b,
    } = bounds.as_mut();
    let old_lower_err = lower_a.coeff_err.take();
    let old_upper_err = upper_a.coeff_err.take();
    let mut new_lower_err = Array1::<f32>::zeros(prepared.row_count);
    let mut new_upper_err = Array1::<f32>::zeros(prepared.row_count);

    let lower_patches = lower_a
        .patches
        .as_mut()
        .expect("prepared alpha-ReLU lower patches must remain materialized")
        .as_slice_mut()
        .expect("prepared alpha-ReLU lower patches must remain contiguous");
    let upper_patches = upper_a
        .patches
        .as_mut()
        .expect("prepared alpha-ReLU upper patches must remain materialized")
        .as_slice_mut()
        .expect("prepared alpha-ReLU upper patches must remain contiguous");
    let lower_bias = lower_b
        .as_slice_mut()
        .expect("prepared alpha-ReLU lower bias must remain contiguous");
    let upper_bias = upper_b
        .as_slice_mut()
        .expect("prepared alpha-ReLU upper bias must remain contiguous");

    let sanitize = |value: f32| -> f64 {
        if value.is_finite() && value >= 0.0 {
            f64::from(value)
        } else {
            f64::INFINITY
        }
    };
    let mut lower_affected = 0usize;
    let mut upper_affected = 0usize;

    for row in 0..prepared.row_count {
        poll()?;
        let start = row * prepared.row_volume;
        let end = start + prepared.row_volume;
        let lower_row = &mut lower_patches[start..end];
        let upper_row = &mut upper_patches[start..end];

        let old_lower_bias = lower_bias[row];
        let old_upper_bias = upper_bias[row];
        let old_lower_error = old_lower_err.as_ref().map_or(0.0, |err| sanitize(err[row]));
        let old_upper_error = old_upper_err.as_ref().map_or(0.0, |err| sanitize(err[row]));

        let mut lower_bias_f64 = f64::from(old_lower_bias);
        let mut upper_bias_f64 = f64::from(old_upper_bias);
        let mut lower_nonfinite = false;
        let mut upper_nonfinite = false;
        let mut max_lower_gap = 0.0f64;
        let mut max_upper_gap = 0.0f64;
        let mut abs_lower_sum = f64::from(old_lower_bias).abs();
        let mut abs_upper_sum = f64::from(old_upper_bias).abs();
        let mut coordinates_since_poll = 0usize;

        for (tap, &input_flat) in prepared.tap_inputs.iter().enumerate() {
            poll_coordinate(&mut coordinates_since_poll, poll)?;
            if input_flat == PADDING_TAP {
                // Historical out-of-place output is zero-initialized, including
                // the sign bit. In-place padding must therefore write +0.0.
                lower_row[tap] = 0.0;
                upper_row[tap] = 0.0;
                continue;
            }
            let relax = &prepared.relaxations[input_flat as usize];

            let old_lower = lower_row[tap];
            let lower_result = compose::compose_lower(old_lower, relax);
            lower_bias_f64 += lower_result.intercept_contrib;
            lower_nonfinite |= lower_result.nonfinite;
            if old_lower != 0.0 {
                let (slope_used, intercept_used) = if old_lower > 0.0 {
                    (
                        f64::from(relax.lower_slope),
                        f64::from(relax.lower_intercept),
                    )
                } else {
                    (
                        f64::from(relax.upper_slope),
                        f64::from(relax.upper_intercept),
                    )
                };
                let stored = f64::from(lower_result.new_coeff);
                let gap = (f64::from(old_lower) * slope_used - stored).abs();
                if gap > max_lower_gap {
                    max_lower_gap = gap;
                }
                abs_lower_sum += (f64::from(old_lower) * intercept_used).abs();
            }
            lower_row[tap] = lower_result.new_coeff;

            let old_upper = upper_row[tap];
            let upper_result = compose::compose_upper(old_upper, relax);
            upper_bias_f64 += upper_result.intercept_contrib;
            upper_nonfinite |= upper_result.nonfinite;
            if old_upper != 0.0 {
                let (slope_used, intercept_used) = if old_upper > 0.0 {
                    (
                        f64::from(relax.upper_slope),
                        f64::from(relax.upper_intercept),
                    )
                } else {
                    (
                        f64::from(relax.lower_slope),
                        f64::from(relax.lower_intercept),
                    )
                };
                let stored = f64::from(upper_result.new_coeff);
                let gap = (f64::from(old_upper) * slope_used - stored).abs();
                if gap > max_upper_gap {
                    max_upper_gap = gap;
                }
                abs_upper_sum += (f64::from(old_upper) * intercept_used).abs();
            }
            upper_row[tap] = upper_result.new_coeff;
        }

        let lower_discharge = prepared.gamma_bar * abs_lower_sum
            + if old_lower_error != 0.0 {
                old_lower_error * (prepared.intercept_sum * (1.0 + prepared.gamma_bar))
            } else {
                0.0
            };
        if lower_discharge.is_finite() {
            lower_bias_f64 -= lower_discharge;
        } else {
            lower_bias_f64 = f64::NEG_INFINITY;
        }
        let upper_discharge = prepared.gamma_bar * abs_upper_sum
            + if old_upper_error != 0.0 {
                old_upper_error * (prepared.intercept_sum * (1.0 + prepared.gamma_bar))
            } else {
                0.0
            };
        if upper_discharge.is_finite() {
            upper_bias_f64 += upper_discharge;
        } else {
            upper_bias_f64 = f64::INFINITY;
        }

        let lower_term = if old_lower_error != 0.0 {
            old_lower_error * prepared.max_slope_sum
        } else {
            0.0
        };
        let upper_term = if old_upper_error != 0.0 {
            old_upper_error * prepared.max_slope_sum
        } else {
            0.0
        };
        let lower_error = lower_term + max_lower_gap;
        let upper_error = upper_term + max_upper_gap;
        new_lower_err[row] = if lower_nonfinite {
            0.0
        } else if !lower_error.is_finite() {
            f32::INFINITY
        } else {
            next_up_f32(lower_error as f32)
        };
        new_upper_err[row] = if upper_nonfinite {
            0.0
        } else if !upper_error.is_finite() {
            f32::INFINITY
        } else {
            next_up_f32(upper_error as f32)
        };
        lower_bias[row] = next_down_f32(lower_bias_f64 as f32);
        upper_bias[row] = next_up_f32(upper_bias_f64 as f32);

        if lower_nonfinite {
            zero_row_with_poll(lower_row, poll)?;
            lower_bias[row] = f32::NEG_INFINITY;
            lower_affected += 1;
        }
        if upper_nonfinite {
            zero_row_with_poll(upper_row, poll)?;
            upper_bias[row] = f32::INFINITY;
            upper_affected += 1;
        }
        poll()?;
    }

    compose::log_nonfinite_fallback(
        "Patches ReLU alpha",
        lower_affected,
        upper_affected,
        prepared.row_count,
    );
    lower_a.coeff_err = Some(new_lower_err);
    upper_a.coeff_err = Some(new_upper_err);
    poll()?;

    // The owned path bypasses `crown_patches_alpha`'s publication seam, so it
    // must perform the same eager post-activation discharge itself. Production
    // uses the exact runtime gates. Tests inject an explicit policy to avoid
    // process-global `OnceLock` environment state and to pin both compositions.
    match explicit_eager_7d_policy {
        Some(true) => {
            bounds.fold_coeff_err_over_box_eager_with_policy(pre_activation, true);
        }
        Some(false) => {}
        None if crate::bounds::patches::eager_err_enabled() => {
            bounds.fold_coeff_err_over_box_eager(pre_activation);
        }
        None => {}
    }
    poll()?;
    Ok(CrownBounds::Patches(bounds))
}

/// Execute a prepared transform with the same absolute deadline used during
/// preparation. The owned carrier is dropped on expiry and is never returned
/// partially transformed.
pub(crate) fn crown_relu_backward_patches_with_alpha_in_place(
    bounds: Box<PatchesLinearBounds>,
    prepared: PreparedAlphaPatchesReluInPlace,
    pre_activation: &BoundedTensor,
) -> Result<CrownBounds> {
    let deadline = prepared.deadline;
    crown_relu_backward_patches_with_alpha_in_place_impl(
        bounds,
        prepared,
        pre_activation,
        None,
        &|| check_alpha_patches_deadline(deadline, "during owned transform"),
    )
}

#[cfg(test)]
pub(crate) fn crown_relu_backward_patches_with_alpha_in_place_with_poll_for_test<P>(
    bounds: Box<PatchesLinearBounds>,
    prepared: PreparedAlphaPatchesReluInPlace,
    pre_activation: &BoundedTensor,
    allow_eager_7d: bool,
    poll: P,
) -> Result<CrownBounds>
where
    P: Fn() -> Result<()>,
{
    crown_relu_backward_patches_with_alpha_in_place_impl(
        bounds,
        prepared,
        pre_activation,
        Some(allow_eager_7d),
        &poll,
    )
}
