// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-aware ReLU GPU CROWN layer extraction (#4312, #4313).
//!
//! Builds the legacy `GpuCrownLayer::Activation` descriptor for same-slope
//! alpha cases and upgrades to `GpuCrownLayer::ActivationReluDualAlpha` only
//! when the optimized lower and upper alpha paths diverge.

use crate::layers::activations::relu::{relu_linear_relaxation, RELU_RELAX_MIN_WIDTH};
use crate::layers::activations::relu_crossing_upper_chord;
use ny_core::GpuCrownLayer;

/// Build a GPU activation layer for an alpha-parameterized ReLU.
///
/// For stable neurons (l >= 0 or u <= 0), uses exact relaxation (identity or zero).
/// For unstable neurons with active alpha, the four-slice ABI preserves exact
/// dual-alpha semantics:
/// - `lower_pos_slope = alpha_lower[i]` (lower bound, positive A)
/// - `cross_slope = chord(l, u)` (lower-neg and upper-pos A paths)
/// - `upper_neg_slope = alpha_upper[i]` (upper bound, negative A)
/// - `cross_intercept = chord_intercept(l, u)` (bias for chord paths)
///
/// Reference: auto_LiRPA/operators/relu.py:641-652
/// Reference: designs/2026-03-21-issue-4313-relu-dual-alpha-four-slice-abi.md
pub(crate) fn extract_relu_gpu_layer_with_alpha(
    pre_l: &[f32],
    pre_u: &[f32],
    alpha_lower: &[f32],
    alpha_upper: &[f32],
    unstable_mask: &[bool],
) -> GpuCrownLayer {
    let num_neurons = pre_l.len();
    let mut lower_slope = Vec::with_capacity(num_neurons);
    let mut upper_slope = Vec::with_capacity(num_neurons);
    let mut lower_intercept = Vec::with_capacity(num_neurons);
    let mut upper_intercept = Vec::with_capacity(num_neurons);
    let mut lower_pos_slope = Vec::with_capacity(num_neurons);
    let mut cross_slope = Vec::with_capacity(num_neurons);
    let mut upper_neg_slope = Vec::with_capacity(num_neurons);
    let mut cross_intercept = Vec::with_capacity(num_neurons);
    let mut needs_dual_alpha = false;

    for i in 0..num_neurons {
        let l = pre_l[i];
        let u = pre_u[i];

        if l.is_nan() || u.is_nan() {
            let r = relu_linear_relaxation(l, u);
            lower_slope.push(r.lower_slope);
            upper_slope.push(r.upper_slope);
            lower_intercept.push(r.lower_intercept);
            upper_intercept.push(r.upper_intercept);
            lower_pos_slope.push(r.lower_slope);
            cross_slope.push(r.upper_slope);
            upper_neg_slope.push(r.lower_slope);
            cross_intercept.push(r.upper_intercept);
        } else if l >= 0.0 {
            lower_slope.push(1.0);
            upper_slope.push(1.0);
            lower_intercept.push(0.0);
            upper_intercept.push(0.0);
            lower_pos_slope.push(1.0);
            cross_slope.push(1.0);
            upper_neg_slope.push(1.0);
            cross_intercept.push(0.0);
        } else if u <= 0.0 {
            lower_slope.push(0.0);
            upper_slope.push(0.0);
            lower_intercept.push(0.0);
            upper_intercept.push(0.0);
            lower_pos_slope.push(0.0);
            cross_slope.push(0.0);
            upper_neg_slope.push(0.0);
            cross_intercept.push(0.0);
        } else if unstable_mask.get(i).copied().unwrap_or(false) {
            let lower_alpha = alpha_lower.get(i).copied().unwrap_or(0.0);
            let upper_alpha = alpha_upper.get(i).copied().unwrap_or(lower_alpha);
            let (chord_s, chord_i) = relu_crossing_upper_chord(l, u, Some(RELU_RELAX_MIN_WIDTH));
            lower_slope.push(lower_alpha);
            upper_slope.push(chord_s);
            lower_intercept.push(0.0);
            upper_intercept.push(chord_i);
            lower_pos_slope.push(lower_alpha);
            cross_slope.push(chord_s);
            upper_neg_slope.push(upper_alpha);
            cross_intercept.push(chord_i);
            needs_dual_alpha |= lower_alpha != upper_alpha;
        } else {
            let r = relu_linear_relaxation(l, u);
            lower_slope.push(r.lower_slope);
            upper_slope.push(r.upper_slope);
            lower_intercept.push(r.lower_intercept);
            upper_intercept.push(r.upper_intercept);
            lower_pos_slope.push(r.lower_slope);
            cross_slope.push(r.upper_slope);
            upper_neg_slope.push(r.lower_slope);
            cross_intercept.push(r.upper_intercept);
        }
    }

    if needs_dual_alpha {
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            num_neurons,
        }
    } else {
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        }
    }
}
