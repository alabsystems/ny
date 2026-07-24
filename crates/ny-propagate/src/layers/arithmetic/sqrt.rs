// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Element-wise square root layer: y = sqrt(x).

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use tracing::debug;

use crate::bounds::nan_propagating_max_zero;
use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    compose::{compose_lower, compose_upper, log_nonfinite_fallback, precompute_relaxations},
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

const SQRT_ALPHA_MIN_MID: f32 = 1e-6;

/// Element-wise square root: y = sqrt(x).
///
/// Requires x >= 0. Returns error if input bounds include negative values,
/// as sqrt is undefined for x < 0. CROWN propagation also rejects negative
/// pre-activation bounds.
#[derive(Debug, Clone)]
pub struct SqrtLayer;

impl SqrtLayer {
    /// Create a new Sqrt layer.
    pub fn new() -> Self {
        Self
    }

    /// Lenient IBP propagation for soundness scans.
    ///
    /// This clamps negative inputs to zero so downstream bounds can be computed
    /// during heuristic detection. It is NOT sound for verification and should
    /// only be used by soundness provenance scans.
    pub(crate) fn propagate_ibp_lenient(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Directed rounding: compute in f64 then round lower bound down, upper bound up.
        // IEEE 754 sqrt is correctly rounded (max 0.5 ULP), but f64→f32 cast may round
        // the wrong direction. next_down_f32/next_up_f32 ensures soundness. (#3243)
        let out_lower = input
            .lower()
            .mapv(|v| next_down_f32((nan_propagating_max_zero(v) as f64).sqrt() as f32));
        let out_upper = input
            .upper()
            .mapv(|v| next_up_f32((nan_propagating_max_zero(v) as f64).sqrt() as f32));
        // NaN in output indicates NaN in input (data corruption, not overflow).
        // Must check before repair which would silently swallow NaN (#2635).
        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "Sqrt IBP lenient: NaN in bounds (from NaN input)".into(),
            ));
        }
        // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #3030).
        // NaN check above ensures only Inf reaches Conservative repair.
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}

impl Default for SqrtLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundPropagation for SqrtLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // sqrt is monotonically increasing for x >= 0.
        // For negative inputs, clamp to 0 and let the soundness provenance scanner
        // detect and flag the result as heuristic (SqrtNegativeDomain).
        // This matches the design in #424: gate negative sqrt in sound mode by
        // marking as heuristic, not by returning an error.
        // Directed rounding: compute in f64 then round lower bound down, upper bound up.
        // IEEE 754 sqrt is correctly rounded (max 0.5 ULP), but f64→f32 cast may round
        // the wrong direction. next_down_f32/next_up_f32 ensures soundness. (#3243)
        let out_lower = input
            .lower()
            .mapv(|v| next_down_f32((nan_propagating_max_zero(v) as f64).sqrt() as f32));
        let out_upper = input
            .upper()
            .mapv(|v| next_up_f32((nan_propagating_max_zero(v) as f64).sqrt() as f32));
        // NaN in output indicates NaN in input (data corruption, not overflow).
        // Must check before repair which would silently swallow NaN (#2635).
        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "Sqrt IBP: NaN in bounds (from NaN input)".into(),
            ));
        }
        // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #3030).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Sqrt requires pre-activation bounds to build a sound linear relaxation.
        Err(NyError::InvalidSpec(
            "Sqrt CROWN propagation requires pre-activation bounds; use propagate_linear_with_bounds or IBP."
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
        SqrtLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Linear relaxation for sqrt on interval [l, u].
/// sqrt(x) is concave and monotonically increasing for x >= 0.
/// For concave functions: chord is lower bound, tangent is upper bound.
///
/// The upper tangent point is the chord-parallel (minimal-gap) point
/// t* = ((sqrt(l)+sqrt(u))/2)^2: the unique point where the tangent slope
/// equals the chord slope, giving the minimal-area / min-max-gap tangent.
/// Since sqrt is concave, the tangent at ANY point is a global upper bound, so
/// this is sound and strictly tighter than the prior tangent-at-u. t* lies in
/// [l, u] (the with-alpha body re-clamps it into the valid domain regardless).
pub fn sqrt_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    sqrt_linear_relaxation_with_alpha(
        l,
        u,
        f32::midpoint((l.max(0.0)).sqrt(), u.max(0.0).sqrt()).powi(2),
    )
}

/// Alpha-aware linear relaxation for sqrt on interval [l, u].
///
/// Uses the chord as the lower relaxation and a tangent at `mid` as the upper
/// relaxation. `mid = u` reproduces the fixed-slope path used by
/// `sqrt_linear_relaxation`, matching alpha-beta-CROWN's optimizable BoundSqrt
/// tangent point.
pub(crate) fn sqrt_linear_relaxation_with_alpha(l: f32, u: f32, mid: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l > u {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    let original_l = l;

    // Clamp to valid domain
    let l = l.max(0.0);
    let u = u.max(0.0);

    // Handle degenerate cases
    if u <= 0.0 {
        return LinearRelaxation::new(0.0, 0.0, 0.0, 0.0);
    }
    if (u - l).abs() < 1e-8 {
        let lower = l.sqrt();
        let upper = u.sqrt();
        return LinearRelaxation::new(0.0, next_down_f32(lower), 0.0, next_up_f32(upper));
    }
    if u < 1e-12 {
        let lower = l.sqrt();
        let upper = u.sqrt();
        return LinearRelaxation::new(0.0, next_down_f32(lower), 0.0, next_up_f32(upper));
    }

    // Use f64 intermediates to prevent catastrophic cancellation.
    // The chord slope (sqrt(u) - sqrt(l)) / (u - l) cancels when l ≈ u.
    // Same pattern as Exp fix (#1745).
    let l64 = l as f64;
    let u64 = u as f64;

    let sqrt_l = l64.sqrt();
    let sqrt_u = u64.sqrt();

    // Chord slope connecting (l, sqrt(l)) to (u, sqrt(u))
    // For concave sqrt, chord is a LOWER bound
    let chord_slope = (sqrt_u - sqrt_l) / (u64 - l64);
    let chord_intercept = sqrt_l - chord_slope * l64;

    // Directed rounding: compensate for f64→f32 truncation in both slope and
    // intercept. Same pattern as exp_linear_relaxation.
    let max_abs_x = l.abs().max(u.abs()) as f64;

    // For concave sqrt: chord is a lower bound, tangent is a global upper bound.
    let lower_slope = chord_slope as f32;
    let lower_slope_err =
        next_up_f32(((chord_slope - lower_slope as f64).abs() * max_abs_x) as f32);
    let lower_intercept = next_down_f32((chord_intercept as f32) - lower_slope_err);

    let mid = if u > 0.0 {
        mid.clamp(l.max(SQRT_ALPHA_MIN_MID.min(u)), u)
    } else {
        0.0
    };
    let mid64 = mid as f64;
    let sqrt_mid = mid64.sqrt();

    let tangent_slope_f64 = 0.5 / sqrt_mid;
    let tangent_intercept_f64 = sqrt_mid - tangent_slope_f64 * mid64;
    let tangent_slope = tangent_slope_f64 as f32;
    let tangent_slope_err =
        next_up_f32(((tangent_slope_f64 - tangent_slope as f64).abs() * max_abs_x) as f32);
    let tangent_intercept = next_up_f32((tangent_intercept_f64 as f32) + tangent_slope_err);
    let upper_slope = tangent_slope;
    let min_intercept = if original_l < 0.0 {
        // Ensure upper bound stays >= 0 for x <= 0.
        next_up_f32(-tangent_slope * original_l)
    } else {
        tangent_intercept
    };
    let upper_intercept = tangent_intercept.max(min_intercept);

    LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl SqrtLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Sqrt", pre_activation)?;
        debug!("Sqrt layer CROWN backward propagation with pre-activation bounds");
        self.ensure_nonnegative_bounds(pre_activation)?;
        crown_elementwise_backward(bounds, pre_activation, sqrt_linear_relaxation)
    }

    pub(crate) fn propagate_linear_with_alpha(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Sqrt-alpha", pre_activation)?;
        self.ensure_nonnegative_bounds(pre_activation)?;

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

        let pre_lower_slice = pre_lower
            .as_slice()
            .ok_or_else(|| NyError::InternalError("Non-contiguous sqrt pre_lower".into()))?;
        let pre_upper_slice = pre_upper
            .as_slice()
            .ok_or_else(|| NyError::InternalError("Non-contiguous sqrt pre_upper".into()))?;
        let lower_path_relaxations =
            precompute_relaxations(pre_lower_slice, pre_upper_slice, &|l, u, idx| {
                sqrt_linear_relaxation_with_alpha(l, u, alpha[idx])
            });
        let upper_path_relaxations =
            precompute_relaxations(pre_lower_slice, pre_upper_slice, &|l, u, idx| {
                sqrt_linear_relaxation_with_alpha(
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
        log_nonfinite_fallback("Sqrt-alpha", lower_affected, upper_affected, num_outputs);

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

    pub(crate) fn ensure_nonnegative_bounds(&self, pre_activation: &BoundedTensor) -> Result<()> {
        if pre_activation.lower().iter().any(|v| *v < 0.0) {
            // Negative lower bounds are an artifact of IBP over-approximation.
            // sqrt(x) is defined for x >= 0, so the true domain is [max(l,0), u].
            // The linear relaxation (sqrt_linear_relaxation_with_alpha) already
            // clamps l to 0 and adjusts the upper intercept for negative
            // original_l (lines 143, 200-206), so CROWN backward is sound here.
            //
            // Previously this returned UnsupportedConfiguration (#3499), which
            // caused the entire graph CROWN backward to fall back to IBP. After
            // #4112 widened intermediate bounds past multi-input nodes, the Sqrt
            // pre-activation can legitimately include negative values even when
            // the model is correct. Proceeding with CROWN preserves tighter
            // bounds. Fix: #4113.
            debug!(
                "Sqrt CROWN backward: pre-activation lower bounds include negative values; \
                 relaxation clamps to [0, u] (domain refinement, proceeding with CROWN)"
            );
        }
        Ok(())
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Sqrt", pre_activation)?;
        debug!("Sqrt layer batched CROWN backward propagation");
        self.ensure_nonnegative_bounds(pre_activation)?;
        crown_elementwise_backward_batched(bounds, pre_activation, sqrt_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Sqrt", pre_activation)?;
        self.ensure_nonnegative_bounds(pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, sqrt_linear_relaxation)
    }
}

#[cfg(test)]
mod tests {
    use super::{sqrt_linear_relaxation, sqrt_linear_relaxation_with_alpha, SqrtLayer};
    use crate::layers::common::BoundPropagation;
    use crate::LinearBounds;
    use ndarray::{arr1, array};
    use ny_tensor::BoundedTensor;

    #[test]
    fn sqrt_ibp_nan_input_returns_error() {
        let layer = SqrtLayer::new();
        let input =
            BoundedTensor::new_unchecked(array![f32::NAN].into_dyn(), array![1.0_f32].into_dyn())
                .expect("shape-only constructor should accept NaN");

        assert!(
            layer.propagate_ibp(&input).is_err(),
            "NaN input must not be absorbed to finite sqrt IBP bounds"
        );
    }

    #[test]
    fn sqrt_ibp_lenient_nan_input_returns_error() {
        let layer = SqrtLayer::new();
        let input =
            BoundedTensor::new_unchecked(array![f32::NAN].into_dyn(), array![1.0_f32].into_dyn())
                .expect("shape-only constructor should accept NaN");

        assert!(
            layer.propagate_ibp_lenient(&input).is_err(),
            "Lenient path must still reject NaN-derived bounds instead of forcing zero"
        );
    }

    /// NaN in upper bound only must also be caught — exercises the
    /// `out_upper.iter().any(|v| v.is_nan())` branch of the guard.
    #[test]
    fn sqrt_ibp_nan_upper_only_returns_error() {
        let layer = SqrtLayer::new();
        let input =
            BoundedTensor::new_unchecked(array![1.0_f32].into_dyn(), array![f32::NAN].into_dyn())
                .expect("shape-only constructor should accept NaN");

        assert!(
            layer.propagate_ibp(&input).is_err(),
            "NaN in upper bound must not be absorbed to finite sqrt IBP bounds"
        );
    }

    /// The default relaxation now uses the chord-parallel (minimal-gap) tangent
    /// point t* = ((sqrt(l)+sqrt(u))/2)^2, not the loose tangent-at-u. Verify the
    /// default reproduces the with-alpha path when fed that same t*, and that the
    /// lower chord is identical regardless of the upper tangent choice.
    #[test]
    fn sqrt_default_uses_chord_parallel_tangent() {
        let (l, u) = (0.25_f32, 4.0_f32);
        let t_star = f32::midpoint((l.max(0.0)).sqrt(), u.max(0.0).sqrt()).powi(2);

        let default = sqrt_linear_relaxation(l, u);
        let alpha_at_tstar = sqrt_linear_relaxation_with_alpha(l, u, t_star);

        assert_eq!(default.lower_slope, alpha_at_tstar.lower_slope);
        assert_eq!(default.lower_intercept, alpha_at_tstar.lower_intercept);
        assert_eq!(default.upper_slope, alpha_at_tstar.upper_slope);
        assert_eq!(default.upper_intercept, alpha_at_tstar.upper_intercept);

        // Lower chord is independent of the tangent point.
        let alpha_at_u = sqrt_linear_relaxation_with_alpha(l, u, u);
        assert_eq!(default.lower_slope, alpha_at_u.lower_slope);
        assert_eq!(default.lower_intercept, alpha_at_u.lower_intercept);
    }

    /// SOUNDNESS + TIGHTNESS: for a concave function the tangent at ANY point is a
    /// global upper bound, so the chord-parallel tangent is sound; it must also be
    /// strictly tighter (smaller max gap to sqrt) than the old tangent-at-u upper
    /// over a wide interval.
    #[test]
    fn sqrt_chord_parallel_upper_is_sound_and_tighter_than_tangent_at_u() {
        let (l, u) = (0.01_f32, 100.0_f32);
        let t_star = f32::midpoint((l.max(0.0)).sqrt(), u.max(0.0).sqrt()).powi(2);

        let new = sqrt_linear_relaxation(l, u);
        let old = sqrt_linear_relaxation_with_alpha(l, u, u);

        let mut new_max_gap = f32::NEG_INFINITY;
        let mut old_max_gap = f32::NEG_INFINITY;
        let n = 2000;
        for i in 0..=n {
            let x = l + (u - l) * (i as f32 / n as f32);
            let fx = x.sqrt();
            let new_upper = new.upper_slope * x + new.upper_intercept;
            let old_upper = old.upper_slope * x + old.upper_intercept;

            // SOUND: both uppers must enclose sqrt at every sample (zero slack
            // beyond the directed-rounding margin already baked into the relaxation).
            assert!(
                new_upper >= fx,
                "new upper bound crosses sqrt at x={x}: {new_upper} < {fx}"
            );
            assert!(
                old_upper >= fx,
                "old upper bound crosses sqrt at x={x}: {old_upper} < {fx}"
            );

            new_max_gap = new_max_gap.max(new_upper - fx);
            old_max_gap = old_max_gap.max(old_upper - fx);
        }

        // TIGHTER: the chord-parallel tangent minimizes the max gap.
        assert!(
            new_max_gap < old_max_gap,
            "chord-parallel upper (max gap {new_max_gap}, t*={t_star}) should be \
             strictly tighter than tangent-at-u (max gap {old_max_gap})"
        );
    }

    #[test]
    fn sqrt_alpha_backward_changes_upper_path_3773() {
        let layer = SqrtLayer::new();
        let pre = BoundedTensor::new(arr1(&[0.25_f32]).into_dyn(), arr1(&[4.0]).into_dyn())
            .expect("sqrt bounds should construct");
        let incoming = LinearBounds::identity(1);

        let fixed = layer
            .propagate_linear_with_bounds(&incoming, &pre)
            .expect("fixed-slope sqrt backward should succeed");
        let alpha = layer
            .propagate_linear_with_alpha(&incoming, &pre, &arr1(&[0.5]), Some(&arr1(&[1.0])))
            .expect("alpha sqrt backward should succeed");

        assert_eq!(
            fixed.lower_a[[0, 0]],
            alpha.lower_a[[0, 0]],
            "sqrt alpha keeps the lower chord slope unchanged"
        );
        assert_eq!(
            fixed.lower_b[0], alpha.lower_b[0],
            "sqrt alpha keeps the lower chord intercept unchanged"
        );
        assert!(
            (fixed.upper_a[[0, 0]] - alpha.upper_a[[0, 0]]).abs() > 1e-6
                || (fixed.upper_b[0] - alpha.upper_b[0]).abs() > 1e-6,
            "sqrt alpha should move the upper tangent away from the endpoint tangent"
        );
    }

    #[test]
    fn sqrt_alpha_backward_rejects_mismatched_alpha_len_3773() {
        let layer = SqrtLayer::new();
        let pre = BoundedTensor::new(arr1(&[0.25_f32]).into_dyn(), arr1(&[4.0]).into_dyn())
            .expect("sqrt bounds should construct");
        let incoming = LinearBounds::identity(1);

        let err = layer
            .propagate_linear_with_alpha(&incoming, &pre, &arr1(&[0.5, 0.75]), None)
            .expect_err("mismatched sqrt alpha length should fail");
        assert!(
            matches!(err, ny_core::NyError::ShapeMismatch { .. }),
            "expected shape mismatch, got {err:?}"
        );
    }
}
