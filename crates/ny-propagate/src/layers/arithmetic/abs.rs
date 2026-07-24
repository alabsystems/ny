// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Element-wise absolute value layer: y = |x|.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use crate::bounds::nan_propagating_max;
use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

/// Element-wise absolute value: y = |x|.
#[derive(Debug, Clone)]
pub struct AbsLayer;

impl AbsLayer {
    /// Create a new Abs layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for AbsLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundPropagation for AbsLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Guard: reject non-finite inputs. Previously the only IBP without this
        // guard — NaN inputs fell through comparisons to the crossing branch
        // where (-l).max(u) silently swallowed NaN (IEEE 754-2008 §5.3.1). (#3316)
        if input.lower().iter().any(|x| !x.is_finite())
            || input.upper().iter().any(|x| !x.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "Abs IBP: non-finite input bounds".to_string(),
            ));
        }
        // For x ∈ [l, u], |x| bounds:
        // - if l >= 0: [l, u]
        // - if u <= 0: [-u, -l]
        // - if l < 0 < u: [0, max(-l, u)]
        let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
        let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));

        for (idx, &l) in input.lower().indexed_iter() {
            let u = input.upper()[idx.clone()];
            if l >= 0.0 {
                out_lower[idx.clone()] = l;
                out_upper[idx] = u;
            } else if u <= 0.0 {
                out_lower[idx.clone()] = -u;
                out_upper[idx] = -l;
            } else {
                out_lower[idx.clone()] = 0.0;
                // NaN-propagating: .max() swallows NaN (IEEE 754-2008). (#3316)
                out_upper[idx] = nan_propagating_max(-l, u);
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Abs is nonlinear — requires pre-activation bounds for sound relaxation.
        Err(NyError::UnsupportedOp(
            "Abs is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        AbsLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Linear relaxation for |x| on interval [l, u].
///
/// Abs is a V-shaped piecewise linear function with a kink at x=0:
/// - For x >= 0: |x| = x  (slope = 1)
/// - For x <= 0: |x| = -x (slope = -1)
///
/// Returns a [`LinearRelaxation`] with lower/upper slope and intercept.
///
/// For crossing intervals (l < 0 < u):
/// - Upper bound: chord from (l, -l) to (u, u).
///   slope = (u + l)/(u - l), intercept = u * (1 - slope)
/// - Lower bound: heuristic tangent — identity (slope=1) if u > -l, else
///   negation (slope=-1). Intercept is always 0 since both tangent lines
///   pass through the origin.
///
/// Reference: alpha-beta-CROWN `BoundAbs.bound_relax`
pub fn abs_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }
    if l >= 0.0 {
        // Always positive: identity |x| = x
        LinearRelaxation::new(1.0, 0.0, 1.0, 0.0)
    } else if u <= 0.0 {
        // Always negative: negation |x| = -x
        LinearRelaxation::new(-1.0, 0.0, -1.0, 0.0)
    } else {
        // Crossing: l < 0 < u
        // Upper bound: chord from (l, -l) to (u, u)
        // slope = (u - (-l)) / (u - l) = (u + l) / (u - l)
        //
        // Use f64 intermediates to prevent f32 rounding from violating soundness.
        // Kani proof abs_crown_upper_bound_sound found counterexample at
        // l ≈ -1.31e-33, u ≈ 4.27e-34 where f32 chord undershoots |x|.
        //
        // Strategy: compute slope in f64, cast to f32. Then compute the minimum
        // intercept needed in f64 (using the f32 slope promoted to f64), and
        // round the intercept UP to the next f32 to guarantee soundness.
        //
        // For |x| (piecewise linear), the worst-case points are:
        //   x=l: need slope*l + intercept >= -l  →  intercept >= -l - slope*l
        //   x=u: need slope*u + intercept >= u   →  intercept >= u - slope*u
        // Computing in f64 with the f32 slope gives the exact required intercept.
        // Rounding up to f32 ensures the f32 evaluation can only exceed the target.
        let l64 = l as f64;
        let u64 = u as f64;
        let slope = ((u64 + l64) / (u64 - l64)) as f32;
        let slope_f64 = slope as f64;
        let needed_at_l = -l64 - slope_f64 * l64;
        let needed_at_u = u64 - slope_f64 * u64;
        let needed = needed_at_l.max(needed_at_u);
        // Closed-form safety margin for f32 evaluation rounding.
        //
        // We use `eps = f32::EPSILON` (2^-23) as a conservative roundoff unit.
        // This over-approximates per-op roundoff (`u = 2^-24`) and keeps the
        // Kani proof arithmetic simple. For endpoint evaluation of
        // `slope*x + intercept`, we conservatively bound:
        //   - multiply + add roundoff, and
        //   - f64→f32 intercept cast roundoff.
        // A `4 * eps * max_endpoint` margin gives a robust safety factor.
        //
        // This replaces an iterative 8-step ULP repair loop that was correct but
        // made Kani/CBMC proofs intractable (600K+ variable SAT formulas).
        // Reference: #1784 for mathematical justification.
        let eps_f64: f64 = f64::from(f32::EPSILON);
        let max_endpoint = (-l64).max(u64);
        let margin = 4.0 * eps_f64 * max_endpoint;
        let intercept = (needed + margin) as f32;
        let intercept = if intercept.is_finite() {
            intercept
        } else {
            needed as f32
        };
        // Lower bound heuristic: identity if positive side dominates, else negation.
        // Both tangent lines pass through the origin, so intercept = 0.
        let lower_slope = if u > -l { 1.0 } else { -1.0 };
        LinearRelaxation::new(lower_slope, 0.0, slope, intercept)
    }
}

impl AbsLayer {
    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// Abs is a V-shaped piecewise linear function with a kink at x=0:
    /// - For x >= 0: |x| = x (slope = 1)
    /// - For x <= 0: |x| = -x (slope = -1)
    ///
    /// For crossing neurons (l < 0 < u):
    /// - Upper bound: chord from (l, -l) to (u, u), slope = (u + l)/(u - l), intercept computed to pass through endpoints
    /// - Lower bound: use either identity (slope=1) or negation (slope=-1) based on heuristic
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Abs", pre_activation)?;
        debug!("Abs layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, abs_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Abs", pre_activation)?;
        debug!("Abs layer batched CROWN backward propagation");
        crown_elementwise_backward_batched(bounds, pre_activation, abs_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Abs", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, abs_linear_relaxation)
    }
}
