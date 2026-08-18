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

/// Whole-row affine interpretation shared by the legacy GPU descriptor and
/// the retained-BaB v1 composer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuReluAffineVariant {
    Ordinary,
    DualAlpha,
}

/// Compute one executed four-f32 ReLU affine cell without allocating.
///
/// `Ordinary` returns lower-slope, upper-slope, lower-intercept, and
/// upper-intercept. `DualAlpha` returns lower-positive slope, crossing slope,
/// upper-negative slope, and crossing intercept. Selecting the variant is a
/// separate whole-row prepass; callers must never change it while filling.
pub(crate) fn gpu_relu_affine_cell(
    lower: f32,
    upper: f32,
    lower_alpha: f32,
    upper_alpha: f32,
    alpha_is_active: bool,
    variant: GpuReluAffineVariant,
) -> [f32; 4] {
    if lower.is_nan() || upper.is_nan() {
        let r = relu_linear_relaxation(lower, upper);
        return match variant {
            GpuReluAffineVariant::Ordinary => [
                r.lower_slope,
                r.upper_slope,
                r.lower_intercept,
                r.upper_intercept,
            ],
            GpuReluAffineVariant::DualAlpha => [
                r.lower_slope,
                r.upper_slope,
                r.lower_slope,
                r.upper_intercept,
            ],
        };
    }
    if lower >= 0.0 {
        return match variant {
            GpuReluAffineVariant::Ordinary => [1.0, 1.0, 0.0, 0.0],
            GpuReluAffineVariant::DualAlpha => [1.0, 1.0, 1.0, 0.0],
        };
    }
    if upper <= 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    if alpha_is_active {
        let (cross_slope, cross_intercept) =
            relu_crossing_upper_chord(lower, upper, Some(RELU_RELAX_MIN_WIDTH));
        return match variant {
            GpuReluAffineVariant::Ordinary => [lower_alpha, cross_slope, 0.0, cross_intercept],
            GpuReluAffineVariant::DualAlpha => {
                [lower_alpha, cross_slope, upper_alpha, cross_intercept]
            }
        };
    }
    let r = relu_linear_relaxation(lower, upper);
    match variant {
        GpuReluAffineVariant::Ordinary => [
            r.lower_slope,
            r.upper_slope,
            r.lower_intercept,
            r.upper_intercept,
        ],
        GpuReluAffineVariant::DualAlpha => [
            r.lower_slope,
            r.upper_slope,
            r.lower_slope,
            r.upper_intercept,
        ],
    }
}

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
    debug_assert_eq!(
        pre_u.len(),
        num_neurons,
        "ReLU lower/upper endpoint rows must have the same width"
    );
    // Seal the whole-row variant before allocating/filling any section. A late
    // mismatch can therefore never reinterpret an already-written ordinary
    // row as dual-alpha.
    let needs_dual_alpha = (0..num_neurons).any(|index| {
        let lower_bound = pre_l[index];
        let upper_bound = pre_u[index];
        let lower = alpha_lower.get(index).copied().unwrap_or(0.0);
        let upper = alpha_upper.get(index).copied().unwrap_or(lower);
        !lower_bound.is_nan()
            && !upper_bound.is_nan()
            && lower_bound < 0.0
            && upper_bound > 0.0
            && unstable_mask.get(index).copied().unwrap_or(false)
            && lower != upper
    });
    let variant = if needs_dual_alpha {
        GpuReluAffineVariant::DualAlpha
    } else {
        GpuReluAffineVariant::Ordinary
    };
    let mut section_0 = Vec::with_capacity(num_neurons);
    let mut section_1 = Vec::with_capacity(num_neurons);
    let mut section_2 = Vec::with_capacity(num_neurons);
    let mut section_3 = Vec::with_capacity(num_neurons);
    for i in 0..num_neurons {
        let l = pre_l[i];
        let u = pre_u[i];
        let lower_alpha = alpha_lower.get(i).copied().unwrap_or(0.0);
        let upper_alpha = alpha_upper.get(i).copied().unwrap_or(lower_alpha);
        let cell = gpu_relu_affine_cell(
            l,
            u,
            lower_alpha,
            upper_alpha,
            unstable_mask.get(i).copied().unwrap_or(false),
            variant,
        );
        section_0.push(cell[0]);
        section_1.push(cell[1]);
        section_2.push(cell[2]);
        section_3.push(cell[3]);
    }

    if needs_dual_alpha {
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: section_0,
            cross_slope: section_1,
            upper_neg_slope: section_2,
            cross_intercept: section_3,
            num_neurons,
        }
    } else {
        GpuCrownLayer::Activation {
            lower_slope: section_0,
            upper_slope: section_1,
            lower_intercept: section_2,
            upper_intercept: section_3,
            num_neurons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frozen copy of the pre-refactor implementation. It intentionally does
    // not call `gpu_relu_affine_cell`, so the comparison detects a shared
    // helper mapping regression rather than merely exercising it twice.
    fn legacy_reference(
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

        for index in 0..num_neurons {
            let lower = pre_l[index];
            let upper = pre_u[index];
            if lower.is_nan() || upper.is_nan() {
                let relaxation = relu_linear_relaxation(lower, upper);
                lower_slope.push(relaxation.lower_slope);
                upper_slope.push(relaxation.upper_slope);
                lower_intercept.push(relaxation.lower_intercept);
                upper_intercept.push(relaxation.upper_intercept);
                lower_pos_slope.push(relaxation.lower_slope);
                cross_slope.push(relaxation.upper_slope);
                upper_neg_slope.push(relaxation.lower_slope);
                cross_intercept.push(relaxation.upper_intercept);
            } else if lower >= 0.0 {
                lower_slope.push(1.0);
                upper_slope.push(1.0);
                lower_intercept.push(0.0);
                upper_intercept.push(0.0);
                lower_pos_slope.push(1.0);
                cross_slope.push(1.0);
                upper_neg_slope.push(1.0);
                cross_intercept.push(0.0);
            } else if upper <= 0.0 {
                lower_slope.push(0.0);
                upper_slope.push(0.0);
                lower_intercept.push(0.0);
                upper_intercept.push(0.0);
                lower_pos_slope.push(0.0);
                cross_slope.push(0.0);
                upper_neg_slope.push(0.0);
                cross_intercept.push(0.0);
            } else if unstable_mask.get(index).copied().unwrap_or(false) {
                let lower_alpha = alpha_lower.get(index).copied().unwrap_or(0.0);
                let upper_alpha = alpha_upper.get(index).copied().unwrap_or(lower_alpha);
                let (chord_slope, chord_intercept_value) =
                    relu_crossing_upper_chord(lower, upper, Some(RELU_RELAX_MIN_WIDTH));
                lower_slope.push(lower_alpha);
                upper_slope.push(chord_slope);
                lower_intercept.push(0.0);
                upper_intercept.push(chord_intercept_value);
                lower_pos_slope.push(lower_alpha);
                cross_slope.push(chord_slope);
                upper_neg_slope.push(upper_alpha);
                cross_intercept.push(chord_intercept_value);
                needs_dual_alpha |= lower_alpha != upper_alpha;
            } else {
                let relaxation = relu_linear_relaxation(lower, upper);
                lower_slope.push(relaxation.lower_slope);
                upper_slope.push(relaxation.upper_slope);
                lower_intercept.push(relaxation.lower_intercept);
                upper_intercept.push(relaxation.upper_intercept);
                lower_pos_slope.push(relaxation.lower_slope);
                cross_slope.push(relaxation.upper_slope);
                upper_neg_slope.push(relaxation.lower_slope);
                cross_intercept.push(relaxation.upper_intercept);
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

    fn descriptor_bits(layer: GpuCrownLayer) -> (bool, [Vec<u32>; 4]) {
        let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect();
        match layer {
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                ..
            } => (
                false,
                [
                    bits(&lower_slope),
                    bits(&upper_slope),
                    bits(&lower_intercept),
                    bits(&upper_intercept),
                ],
            ),
            GpuCrownLayer::ActivationReluDualAlpha {
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
                ..
            } => (
                true,
                [
                    bits(&lower_pos_slope),
                    bits(&cross_slope),
                    bits(&upper_neg_slope),
                    bits(&cross_intercept),
                ],
            ),
            _ => panic!("unexpected descriptor in ReLU parity test"),
        }
    }

    #[test]
    fn scalar_refactor_is_bit_exact_to_legacy_matrix() {
        let half_width = RELU_RELAX_MIN_WIDTH / 2.0;
        let cases: &[(&[f32], &[f32], &[f32], &[f32], &[bool])] = &[
            // NaN/generic, stable positive/negative, and a late divergent
            // crossing force every earlier cell into the dual interpretation.
            (
                &[f32::NAN, -0.0, -2.0, -1.0],
                &[1.0, 0.0, -0.0, 2.0],
                &[0.2, -0.0, 0.7, 0.25],
                &[0.2, 0.0, 0.7, 0.75],
                &[false, false, false, true],
            ),
            // Missing/short alpha and mask slices retain their historical
            // defaulting behavior.
            (&[-2.0, -3.0], &[3.0, 2.0], &[], &[], &[true]),
            // A missing upper alpha inherits the present lower alpha rather
            // than falling back independently.
            (&[-2.0], &[3.0], &[0.375], &[], &[true]),
            // Numeric +0 == -0 must not select the dual ABI.
            (&[-1.0], &[1.0], &[0.0], &[-0.0], &[true]),
            // Exact guard width and its adjacent representable values.
            (
                &[
                    -half_width,
                    -f32::from_bits(half_width.to_bits() + 1),
                    -f32::from_bits(half_width.to_bits().saturating_sub(1)),
                ],
                &[
                    half_width,
                    f32::from_bits(half_width.to_bits() + 1),
                    f32::from_bits(half_width.to_bits().saturating_sub(1)),
                ],
                &[0.0, 0.25, 0.5],
                &[0.0, 0.75, 0.5],
                &[true, true, true],
            ),
        ];

        for &(pre_l, pre_u, alpha_lower, alpha_upper, unstable_mask) in cases {
            let expected = descriptor_bits(legacy_reference(
                pre_l,
                pre_u,
                alpha_lower,
                alpha_upper,
                unstable_mask,
            ));
            let actual = descriptor_bits(extract_relu_gpu_layer_with_alpha(
                pre_l,
                pre_u,
                alpha_lower,
                alpha_upper,
                unstable_mask,
            ));
            assert_eq!(actual, expected);
        }
    }
}
