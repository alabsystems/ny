// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SPSA gradient accumulation helpers for monotone, sqrt, and reciprocal activations.
//!
//! Extracted from `spsa.rs` to keep file sizes under 500 lines.

use crate::bounds::alpha_reciprocal::ReciprocalGradients;
use crate::bounds::{MonotoneSShapedGradients, SqrtGradients};
use ndarray::Array1;

pub(super) fn accumulate_sqrt_gradients(
    gradients: &mut SqrtGradients,
    perturbation: &SqrtGradients,
    diff: f32,
    eps: f32,
) {
    for i in 0..gradients.lower_path.len() {
        if perturbation.lower_path[i].abs() > 0.5 {
            gradients.lower_path[i] += diff / (2.0 * eps * perturbation.lower_path[i]);
        }
    }
    for i in 0..gradients.upper_path.len() {
        if perturbation.upper_path[i].abs() > 0.5 {
            gradients.upper_path[i] += diff / (2.0 * eps * perturbation.upper_path[i]);
        }
    }
}

pub(super) fn accumulate_monotone_gradients(
    gradients: &mut MonotoneSShapedGradients,
    perturbations: &MonotoneSShapedGradients,
    diff: f32,
    eps: f32,
) {
    accumulate_monotone_gradient_group(
        &mut gradients.tp_pos.lower_path,
        &perturbations.tp_pos.lower_path,
        diff,
        eps,
    );
    accumulate_monotone_gradient_group(
        &mut gradients.tp_pos.upper_path,
        &perturbations.tp_pos.upper_path,
        diff,
        eps,
    );
    accumulate_monotone_gradient_group(
        &mut gradients.tp_neg.lower_path,
        &perturbations.tp_neg.lower_path,
        diff,
        eps,
    );
    accumulate_monotone_gradient_group(
        &mut gradients.tp_neg.upper_path,
        &perturbations.tp_neg.upper_path,
        diff,
        eps,
    );
    accumulate_monotone_gradient_group(
        &mut gradients.tp_both_lower.lower_path,
        &perturbations.tp_both_lower.lower_path,
        diff,
        eps,
    );
    accumulate_monotone_gradient_group(
        &mut gradients.tp_both_lower.upper_path,
        &perturbations.tp_both_lower.upper_path,
        diff,
        eps,
    );
    accumulate_monotone_gradient_group(
        &mut gradients.tp_both_upper.lower_path,
        &perturbations.tp_both_upper.lower_path,
        diff,
        eps,
    );
    accumulate_monotone_gradient_group(
        &mut gradients.tp_both_upper.upper_path,
        &perturbations.tp_both_upper.upper_path,
        diff,
        eps,
    );
}

pub(super) fn accumulate_reciprocal_gradients(
    gradients: &mut ReciprocalGradients,
    perturbation: &ReciprocalGradients,
    diff: f32,
    eps: f32,
) {
    for i in 0..gradients.lower_path.len() {
        if perturbation.lower_path[i].abs() > 0.5 {
            gradients.lower_path[i] += diff / (2.0 * eps * perturbation.lower_path[i]);
        }
    }
    for i in 0..gradients.upper_path.len() {
        if perturbation.upper_path[i].abs() > 0.5 {
            gradients.upper_path[i] += diff / (2.0 * eps * perturbation.upper_path[i]);
        }
    }
}

fn accumulate_monotone_gradient_group(
    gradients: &mut Array1<f32>,
    perturbations: &Array1<f32>,
    diff: f32,
    eps: f32,
) {
    for i in 0..gradients.len() {
        if perturbations[i].abs() > 0.5 {
            gradients[i] += diff / (2.0 * eps * perturbations[i]);
        }
    }
}
