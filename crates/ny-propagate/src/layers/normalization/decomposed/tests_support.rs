// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::BatchedLinearBounds;
use ndarray::{Array1, Array2};
use proptest::prelude::*;

pub(super) fn eval_bilinear_affine(
    primary_coeff: f64,
    aux_coeff: f64,
    bias: f64,
    primary: f32,
    aux: f32,
) -> f64 {
    primary_coeff * f64::from(primary) + aux_coeff * f64::from(aux) + bias
}

pub(super) fn interval_samples(lower: f32, upper: f32) -> [f32; 5] {
    let width = upper - lower;
    [
        lower,
        lower + width * 0.25,
        lower + width * 0.5,
        lower + width * 0.75,
        upper,
    ]
}

#[allow(clippy::too_many_arguments)]
pub(super) fn expected_lower_plane(
    weight: f32,
    primary_lower: f32,
    primary_upper: f32,
    aux_lower: f32,
    aux_upper: f32,
) -> (f32, f32, f32) {
    let primary_mid = primary_lower * 0.5 + primary_upper * 0.5;
    let aux_mid = aux_lower * 0.5 + aux_upper * 0.5;
    let l1_val = aux_lower * primary_mid + primary_lower * aux_mid - primary_lower * aux_lower;
    let l2_val = aux_upper * primary_mid + primary_upper * aux_mid - primary_upper * aux_upper;
    let u1_val = aux_upper * primary_mid + primary_lower * aux_mid - primary_lower * aux_upper;
    let u2_val = aux_lower * primary_mid + primary_upper * aux_mid - primary_upper * aux_lower;

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

#[allow(clippy::too_many_arguments)]
pub(super) fn expected_upper_plane(
    weight: f32,
    primary_lower: f32,
    primary_upper: f32,
    aux_lower: f32,
    aux_upper: f32,
) -> (f32, f32, f32) {
    let primary_mid = primary_lower * 0.5 + primary_upper * 0.5;
    let aux_mid = aux_lower * 0.5 + aux_upper * 0.5;
    let l1_val = aux_lower * primary_mid + primary_lower * aux_mid - primary_lower * aux_lower;
    let l2_val = aux_upper * primary_mid + primary_upper * aux_mid - primary_upper * aux_upper;
    let u1_val = aux_upper * primary_mid + primary_lower * aux_mid - primary_lower * aux_upper;
    let u2_val = aux_lower * primary_mid + primary_upper * aux_mid - primary_upper * aux_lower;

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

pub(super) fn mean_square_bounds(primary_lower: &[f32], primary_upper: &[f32]) -> (f32, f32) {
    let norm_size = primary_lower.len() as f32;
    let lower = primary_lower
        .iter()
        .zip(primary_upper.iter())
        .map(|(&l, &u)| {
            if l <= 0.0 && u >= 0.0 {
                0.0
            } else {
                l.abs().min(u.abs()).powi(2)
            }
        })
        .sum::<f32>()
        / norm_size;
    let upper = primary_lower
        .iter()
        .zip(primary_upper.iter())
        .map(|(&l, &u)| l.abs().max(u.abs()).powi(2))
        .sum::<f32>()
        / norm_size;
    (lower, upper)
}

pub(super) fn eval_variance_affine(primary_coeffs: &[f64], bias: f64, primary: &[f32]) -> f64 {
    primary_coeffs
        .iter()
        .zip(primary.iter())
        .map(|(&coeff, &value)| coeff * f64::from(value))
        .sum::<f64>()
        + bias
}

pub(super) fn true_variance_chain_value(aux_coeff: f64, primary: &[f32]) -> f64 {
    let mean_sq = primary
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        / primary.len() as f64;
    aux_coeff / mean_sq.sqrt()
}

pub(super) fn interpolate(lower: f32, upper: f32, t: f32) -> f32 {
    lower + (upper - lower) * t
}

pub(super) fn constant_batched_bounds(
    lower_a: Array2<f32>,
    lower_b: Array1<f32>,
    upper_a: Array2<f32>,
    upper_b: Array1<f32>,
    input_dim: usize,
) -> BatchedLinearBounds {
    let output_dim = lower_b.len();
    BatchedLinearBounds::new(
        lower_a.into_dyn(),
        lower_b.into_dyn(),
        upper_a.into_dyn(),
        upper_b.into_dyn(),
        vec![input_dim],
        vec![output_dim],
    )
    .expect("test bounds should be valid")
}

pub(super) fn ordered_interval() -> impl Strategy<Value = (f32, f32)> {
    (-3.0f32..2.0, 0.05f32..1.5).prop_map(|(start, width)| (start, start + width))
}

pub(super) fn positive_interval() -> impl Strategy<Value = (f32, f32)> {
    (0.25f32..2.0, 0.01f32..0.75).prop_map(|(start, width)| (start, start + width))
}
