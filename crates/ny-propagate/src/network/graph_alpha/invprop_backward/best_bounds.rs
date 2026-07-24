// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use crate::bounds::LinearBounds;
use ndarray::Zip;
use ny_tensor::BoundedTensor;
use tracing::trace;

use super::super::propagate_helpers::clamp_inverted_best_bounds;

/// Take element-wise best (tighter) linear bounds from two computations.
///
/// For linear bounds (A, b matrices):
/// - For lower A: take the A that produces higher lower bounds
/// - For upper A: take the A that produces lower upper bounds
///
/// Since we can't easily determine which A is better without concretizing,
/// this function takes the A from the bound with better b values.
#[cfg(test)]
pub(super) fn take_best_linear_bounds(
    bounds_with_oc: &LinearBounds,
    bounds_without_oc: &LinearBounds,
) -> LinearBounds {
    let oc_lower_sum: f32 = bounds_with_oc.lower_b.iter().sum();
    let no_oc_lower_sum: f32 = bounds_without_oc.lower_b.iter().sum();
    let oc_upper_sum: f32 = bounds_with_oc.upper_b.iter().sum();
    let no_oc_upper_sum: f32 = bounds_without_oc.upper_b.iter().sum();

    let use_oc_lower = oc_lower_sum >= no_oc_lower_sum;
    let use_oc_upper = oc_upper_sum <= no_oc_upper_sum;

    LinearBounds {
        lower_a: if use_oc_lower {
            bounds_with_oc.lower_a.clone()
        } else {
            bounds_without_oc.lower_a.clone()
        },
        lower_b: if use_oc_lower {
            bounds_with_oc.lower_b.clone()
        } else {
            bounds_without_oc.lower_b.clone()
        },
        upper_a: if use_oc_upper {
            bounds_with_oc.upper_a.clone()
        } else {
            bounds_without_oc.upper_a.clone()
        },
        upper_b: if use_oc_upper {
            bounds_with_oc.upper_b.clone()
        } else {
            bounds_without_oc.upper_b.clone()
        },
        lower_a_err: None,
        upper_a_err: None,
    }
}

/// Take element-wise best (tighter) concrete bounds from two computations.
///
/// Used for `best_of_oc_and_no_oc` mode:
/// - best_lower[i] = max(bounds_oc.lower()[i], bounds_no_oc.lower()[i])
/// - best_upper[i] = min(bounds_oc.upper()[i], bounds_no_oc.upper()[i])
///
/// This mirrors the element-wise merge used by auto_LiRPA.
/// Source: Verified-Intelligence, "auto_LiRPA output_constraints.py",
/// filepath: auto_LiRPA/output_constraints.py:116-132,
/// https://github.com/Verified-Intelligence/auto_LiRPA/blob/9d100ec070868440b48d34e2f1dd21b97aab9172/auto_LiRPA/output_constraints.py#L116-L132
pub(crate) fn take_best_bounds(
    bounds_with_oc: &BoundedTensor,
    bounds_without_oc: &BoundedTensor,
) -> BoundedTensor {
    if bounds_with_oc.lower().shape() != bounds_without_oc.lower().shape()
        || bounds_with_oc.upper().shape() != bounds_without_oc.upper().shape()
    {
        trace!(
            "INVPROP best_of: shape mismatch, returning bounds_with_oc. \
             with_oc lower/upper: {:?}/{:?}, without_oc lower/upper: {:?}/{:?}",
            bounds_with_oc.lower().shape(),
            bounds_with_oc.upper().shape(),
            bounds_without_oc.lower().shape(),
            bounds_without_oc.upper().shape()
        );
        return bounds_with_oc.clone();
    }

    let mut best_lower = bounds_with_oc.lower().clone();
    let mut best_upper = bounds_with_oc.upper().clone();

    Zip::from(&mut best_lower)
        .and(bounds_without_oc.lower())
        .for_each(|best, &other| {
            if other > *best || best.is_nan() {
                *best = other;
            }
        });

    Zip::from(&mut best_upper)
        .and(bounds_without_oc.upper())
        .for_each(|best, &other| {
            if other < *best || best.is_nan() {
                *best = other;
            }
        });

    clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, "invprop-take-best-bounds");
    BoundedTensor::new_allow_infinite(best_lower, best_upper).unwrap_or_else(|e| {
        tracing::warn!(
            error = %e,
            "take_best_bounds: bounds invalid after inversion widening, falling back to OC bounds"
        );
        bounds_with_oc.clone()
    })
}
