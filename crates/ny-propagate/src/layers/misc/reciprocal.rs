// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reciprocal layer (y = 1/x) with tangent-line CROWN relaxation.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use tracing::debug;

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    compose::{compose_lower, compose_upper, log_nonfinite_fallback, precompute_relaxations},
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

/// Reciprocal layer: y = 1/x.
///
/// Requires x != 0. For interval bounds containing zero, returns conservative [-inf, inf].
#[derive(Debug, Clone, Default)]
pub struct ReciprocalLayer;

impl ReciprocalLayer {
    /// Create a new Reciprocal layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for ReciprocalLayer {
    /// IBP for Reciprocal: y = 1/x
    ///
    /// Reciprocal is monotonically decreasing for x > 0 and x < 0.
    /// For x in [lb, ub] where lb > 0: y in [1/ub, 1/lb]
    /// For x in [lb, ub] where ub < 0: y in [1/ub, 1/lb]
    /// For intervals containing zero: y in [-inf, inf] (conservative)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
        let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));

        for (idx, &lb) in input.lower().indexed_iter() {
            let ub = input.upper()[idx.clone()];

            let (r_lb, r_ub) = if lb > 0.0 {
                // Entire interval is positive: 1/x is decreasing, so flip bounds.
                // Directed rounding: compute in f64, round lower down, upper up. (#3243)
                // IEEE 754 division is correctly rounded (max 0.5 ULP), but f64→f32
                // cast may round the wrong direction.
                (
                    next_down_f32((1.0_f64 / ub as f64) as f32),
                    next_up_f32((1.0_f64 / lb as f64) as f32),
                )
            } else if ub < 0.0 {
                // Entire interval is negative: 1/x is decreasing, so flip bounds.
                // Same directed rounding pattern. (#3243)
                (
                    next_down_f32((1.0_f64 / ub as f64) as f32),
                    next_up_f32((1.0_f64 / lb as f64) as f32),
                )
            } else {
                // Interval contains zero: reciprocal is undefined at 0
                // Return conservative bounds
                (f32::NEG_INFINITY, f32::INFINITY)
            };

            out_lower[idx.clone()] = r_lb;
            out_upper[idx] = r_ub;
        }

        // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #3030).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// CROWN requires pre-activation bounds for nonlinear Reciprocal.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Reciprocal is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        ReciprocalLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Linear relaxation for reciprocal on interval [l, u].
/// Reciprocal f(x) = 1/x is:
/// - Monotonically decreasing for both x > 0 and x < 0
/// - Convex for x > 0 (f''(x) = 2/x³ > 0)
/// - Concave for x < 0 (f''(x) = 2/x³ < 0)
///
/// For convex regions (x > 0): secant is upper bound, tangent is lower bound.
/// For concave regions (x < 0): secant is lower bound, tangent is upper bound.
// Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
#[allow(clippy::manual_midpoint)]
pub(crate) fn reciprocal_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    reciprocal_linear_relaxation_with_alpha(l, u, 0.5 * (l + u))
}

/// Alpha-aware linear relaxation for reciprocal on interval [l, u].
///
/// For the positive domain (convex): the tangent at `mid` is the lower bound
/// (optimizable), and the secant/chord is the upper bound (fixed).
/// For the negative domain (concave): the tangent at `mid` is the upper bound
/// (optimizable), and the secant/chord is the lower bound (fixed).
///
/// `mid` is the optimizable tangent point (alpha parameter). When
/// `mid = 0.5 * (l + u)` this reproduces the fixed midpoint tangent
/// from `reciprocal_linear_relaxation`.
///
/// Reference: alpha-beta-CROWN `operators/convex_concave.py:66-196`
pub(crate) fn reciprocal_linear_relaxation_with_alpha(
    l: f32,
    u: f32,
    mid: f32,
) -> LinearRelaxation {
    // Guard: NaN bounds → return conservative intercepts so downstream
    // propagation widens to ±inf instead of silently absorbing NaN.
    // Pattern: every other *_linear_relaxation function has this guard.
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::nan_fallback();
    }

    #[inline]
    fn round_line_to_f32(
        slope: f64,
        intercept: f64,
        l: f64,
        u: f64,
        interval_positive: bool,
        is_lower: bool,
    ) -> (f32, f32) {
        let rounded_slope = if is_lower {
            if interval_positive {
                // x >= 0: smaller slope lowers the lower line.
                next_down_f32(slope as f32)
            } else {
                // x <= 0: larger slope lowers the lower line.
                next_up_f32(slope as f32)
            }
        } else if interval_positive {
            // x >= 0: larger slope raises the upper line.
            next_up_f32(slope as f32)
        } else {
            // x <= 0: smaller slope raises the upper line.
            next_down_f32(slope as f32)
        };

        // After slope rounding, choose the tightest intercept that still keeps
        // the rounded line conservative for both interval endpoints.
        //
        // If L(x)=s*x+b is exact and R(x)=s_r*x+b_r is rounded, then:
        // R(x) <= L(x) (lower)  iff b_r <= (s - s_r)*x + b
        // R(x) >= L(x) (upper)  iff b_r >= (s - s_r)*x + b
        // The RHS is linear in x, so endpoint extrema are sufficient.
        let rounded_slope_f64 = f64::from(rounded_slope);
        let intercept_at_l = (slope - rounded_slope_f64) * l + intercept;
        let intercept_at_u = (slope - rounded_slope_f64) * u + intercept;
        let rounded_intercept = if is_lower {
            next_down_f32(intercept_at_l.min(intercept_at_u) as f32)
        } else {
            next_up_f32(intercept_at_l.max(intercept_at_u) as f32)
        };

        (rounded_slope, rounded_intercept)
    }

    // Handle intervals that cross zero - reciprocal is undefined.
    if l <= 0.0 && u >= 0.0 {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    // Use f64 intermediates to prevent catastrophic cancellation.
    // Reciprocal slopes grow as 1/x² near zero (e.g., l=0.01 → slope=-10000),
    // making intercept subtraction (1/l - slope*l) prone to cancellation in f32.
    // Same pattern as Exp fix (#1745).
    let l64 = l as f64;
    let u64 = u as f64;

    // Degenerate case: zero-width interval (l == u in f64).
    // Use tangent-at-l for both bounds since there's only one point.
    if l64 == u64 {
        let slope = -1.0 / (l64 * l64);
        let intercept = 1.0 / l64 - slope * l64;
        let interval_positive = l > 0.0;
        let (lower_slope, lower_intercept) =
            round_line_to_f32(slope, intercept, l64, u64, interval_positive, true);
        let (upper_slope, upper_intercept) =
            round_line_to_f32(slope, intercept, l64, u64, interval_positive, false);
        return LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept);
    }

    let recip_l = 1.0 / l64;

    // Secant (chord) through endpoints: slope = (1/u - 1/l) / (u - l) = -1/(l*u).
    let secant_slope = -1.0 / (l64 * u64);
    let secant_intercept = recip_l - secant_slope * l64;

    // Tangent at optimizable point d (clamped to [l, u]).
    let d = (mid as f64).clamp(l64, u64);
    let tangent_slope = -1.0 / (d * d);
    let tangent_intercept = 2.0 / d;

    if l > 0.0 {
        // Convex: secant is upper, tangent is lower.
        let (lower_slope, lower_intercept) =
            round_line_to_f32(tangent_slope, tangent_intercept, l64, u64, true, true);
        let (upper_slope, upper_intercept) =
            round_line_to_f32(secant_slope, secant_intercept, l64, u64, true, false);
        LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
    } else {
        // Concave: secant is lower, tangent is upper.
        let (lower_slope, lower_intercept) =
            round_line_to_f32(secant_slope, secant_intercept, l64, u64, false, true);
        let (upper_slope, upper_intercept) =
            round_line_to_f32(tangent_slope, tangent_intercept, l64, u64, false, false);
        LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
    }
}

impl ReciprocalLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Reciprocal layer CROWN backward propagation with pre-activation bounds");
        if pre_activation
            .lower()
            .iter()
            .zip(pre_activation.upper().iter())
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || (l <= 0.0 && u >= 0.0))
        {
            return Err(NyError::InvalidSpec(
                "Reciprocal CROWN requires finite pre-activation bounds that do not cross zero"
                    .to_string(),
            ));
        }
        crown_elementwise_backward(bounds, pre_activation, reciprocal_linear_relaxation)
    }

    /// Alpha-parameterized CROWN backward propagation.
    ///
    /// `alpha` contains the tangent point for each neuron on the lower-path
    /// (for positive domain: tangent is the lower bound).
    /// `alpha_upper` optionally provides separate tangent points for the upper path.
    /// Mirrors `SqrtLayer::propagate_linear_with_alpha` structure.
    pub(crate) fn propagate_linear_with_alpha(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
    ) -> Result<LinearBounds> {
        debug!("Reciprocal layer alpha-CROWN backward propagation");
        if pre_activation
            .lower()
            .iter()
            .zip(pre_activation.upper().iter())
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || (l <= 0.0 && u >= 0.0))
        {
            return Err(NyError::InvalidSpec(
                "Reciprocal alpha-CROWN requires finite pre-activation bounds that do not cross zero"
                    .to_string(),
            ));
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
        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }
        if alpha.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![alpha.len()],
            });
        }
        if let Some(alpha_upper) = alpha_upper {
            if alpha_upper.len() != num_neurons {
                return Err(NyError::ShapeMismatch {
                    expected: vec![num_neurons],
                    got: vec![alpha_upper.len()],
                });
            }
        }

        let pre_lower_cow = crate::contiguous_flat_slice(&pre_lower);
        let pre_upper_cow = crate::contiguous_flat_slice(&pre_upper);
        let pre_lower_slice = pre_lower_cow.as_ref();
        let pre_upper_slice = pre_upper_cow.as_ref();
        let lower_path_relaxations =
            precompute_relaxations(pre_lower_slice, pre_upper_slice, &|l, u, idx| {
                reciprocal_linear_relaxation_with_alpha(l, u, alpha[idx])
            });
        let upper_path_relaxations =
            precompute_relaxations(pre_lower_slice, pre_upper_slice, &|l, u, idx| {
                reciprocal_linear_relaxation_with_alpha(
                    l,
                    u,
                    alpha_upper.map_or(alpha[idx], |upper| upper[idx]),
                )
            });

        let num_outputs = bounds.num_outputs();
        let mut new_lower_a = ndarray::Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_a = ndarray::Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);
        let mut lower_nonfinite_rows = vec![false; num_outputs];
        let mut upper_nonfinite_rows = vec![false; num_outputs];

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let lr = compose_lower(bounds.lower_a()[[j, i]], &lower_path_relaxations[i]);
                new_lower_a[[j, i]] = lr.new_coeff;
                new_lower_b_f64[j] += lr.intercept_contrib;
                lower_nonfinite_rows[j] |= lr.nonfinite;

                let ur = compose_upper(bounds.upper_a()[[j, i]], &upper_path_relaxations[i]);
                new_upper_a[[j, i]] = ur.new_coeff;
                new_upper_b_f64[j] += ur.intercept_contrib;
                upper_nonfinite_rows[j] |= ur.nonfinite;
            }
        }

        let lower_affected = lower_nonfinite_rows.iter().filter(|&&row| row).count();
        let upper_affected = upper_nonfinite_rows.iter().filter(|&&row| row).count();
        log_nonfinite_fallback(
            "Reciprocal-alpha",
            lower_affected,
            upper_affected,
            num_outputs,
        );

        let mut new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
        let mut new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));
        for j in 0..num_outputs {
            if lower_nonfinite_rows[j] {
                for i in 0..num_neurons {
                    new_lower_a[[j, i]] = 0.0;
                }
                new_lower_b[j] = f32::NEG_INFINITY;
            }
            if upper_nonfinite_rows[j] {
                for i in 0..num_neurons {
                    new_upper_a[[j, i]] = 0.0;
                }
                new_upper_b[j] = f32::INFINITY;
            }
        }

        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("Reciprocal layer batched CROWN backward propagation");
        if pre_activation
            .lower()
            .iter()
            .zip(pre_activation.upper().iter())
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || (l <= 0.0 && u >= 0.0))
        {
            return Err(NyError::InvalidSpec(
                "Reciprocal batched CROWN requires finite pre-activation bounds that do not cross zero"
                    .to_string(),
            ));
        }
        crown_elementwise_backward_batched(bounds, pre_activation, reciprocal_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        if pre_activation
            .lower()
            .iter()
            .zip(pre_activation.upper().iter())
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || (l <= 0.0 && u >= 0.0))
        {
            return Err(NyError::InvalidSpec(
                "Reciprocal Patches CROWN requires finite pre-activation bounds that do not cross zero"
                    .to_string(),
            ));
        }
        crown_elementwise_backward_patches(bounds, pre_activation, reciprocal_linear_relaxation)
    }
}
