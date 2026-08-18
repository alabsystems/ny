// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Error-function activation layer for bound propagation.
//!
//! ONNX `Erf` is an element-wise, monotone S-shaped function.  Its second
//! derivative is positive on the negative half-line and negative on the
//! positive half-line, so it has exactly the same convexity split used by the
//! shared Tanh/Sigmoid relaxation machinery.

use ny_core::{f64_to_f32_down, f64_to_f32_up, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use std::sync::OnceLock;
use tracing::debug;

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, ibp_bound_interval_parallel, non_finite_domain_guard,
    BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

use super::shared::{s_shaped_linear_relaxation, SShapedPrecomputeTables, S_SHAPED_RELAX_EPS};

/// Element-wise Gaussian error function: `y = erf(x)`.
///
/// `erf` is monotone increasing, odd, and has range `(-1, 1)` on finite
/// inputs.  This layer exists primarily so an ONNX decomposition such as
/// `x * (1 + Erf(x / sqrt(2))) / 2` can be represented without replacing its
/// source constants with a canonical fused GELU implementation.
#[derive(Debug, Clone, Default)]
pub struct ErfLayer;

impl ErfLayer {
    /// Create an error-function layer.
    pub fn new() -> Self {
        Self
    }
}

/// f64 evaluation used to construct the analytic relaxations.
pub(super) fn erf_f64(x: f64) -> f64 {
    libm::erf(x)
}

/// `d erf(x) / dx = 2/sqrt(pi) * exp(-x^2)`.
pub(super) fn erf_d_f64(x: f64) -> f64 {
    // Kept as a verbatim literal rather than `f64::consts::FRAC_2_SQRT_PI`:
    // the tables and envelopes below are built from this exact bit pattern and
    // this crate rounds deliberately, so a constant substitution is not a
    // value-preserving edit.
    #[allow(clippy::approx_constant)]
    const TWO_OVER_SQRT_PI: f64 = 1.128_379_167_095_512_6;
    TWO_OVER_SQRT_PI * (-x * x).exp()
}

/// Monotone interval image with outward rounding.
///
/// `libm::erf` is used in f64, then the conversion is directed and widened by
/// one further f32 ULP to cover the pointwise libm faithful-rounding policy
/// used by NY's other transcendental layers.  Finally, the mathematical range
/// is used as an exact clamp.
fn erf_bound_interval(l: f32, u: f32) -> (f32, f32) {
    let lower = next_down_f32(f64_to_f32_down(erf_f64(f64::from(l)))).max(-1.0);
    let upper = next_up_f32(f64_to_f32_up(erf_f64(f64::from(u)))).min(1.0);
    (lower, upper)
}

fn erf_tables() -> &'static SShapedPrecomputeTables {
    static TABLES: OnceLock<SShapedPrecomputeTables> = OnceLock::new();
    TABLES.get_or_init(|| SShapedPrecomputeTables::new(erf_f64, erf_d_f64))
}

fn erf_constant_relaxation() -> LinearRelaxation {
    LinearRelaxation::constant(-1.0 - S_SHAPED_RELAX_EPS, 1.0 + S_SHAPED_RELAX_EPS)
}

/// Sound affine lower and upper envelopes for `erf` on `[l, u]`.
pub(crate) fn erf_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    s_shaped_linear_relaxation(
        l,
        u,
        erf_f64,
        erf_d_f64,
        erf_tables(),
        erf_constant_relaxation,
    )
}

impl BoundPropagation for ErfLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        ibp_bound_interval_parallel(input, erf_bound_interval)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Erf is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        ErfLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl ErfLayer {
    /// Dense CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Erf", pre_activation)?;
        debug!("Erf layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, erf_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Erf", pre_activation)?;
        debug!("Erf layer batched CROWN backward propagation");
        crown_elementwise_backward_batched(bounds, pre_activation, erf_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Erf", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, erf_linear_relaxation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_crown_backward_sound, assert_relaxation_sound};
    use ndarray::arr1;

    #[test]
    fn ibp_encloses_erf_and_respects_exact_range() {
        let input = BoundedTensor::new(
            arr1(&[-f32::MAX, -3.0, -0.25, 0.0, 2.0]).into_dyn(),
            arr1(&[-2.0, 0.5, 0.75, 3.0, f32::MAX]).into_dyn(),
        )
        .unwrap();
        let output = ErfLayer::new().propagate_ibp(&input).unwrap();

        for i in 0..input.len() {
            let exact_lower = erf_f64(f64::from(input.lower()[i]));
            let exact_upper = erf_f64(f64::from(input.upper()[i]));
            assert!(f64::from(output.lower()[i]) <= exact_lower);
            assert!(f64::from(output.upper()[i]) >= exact_upper);
            assert!(output.lower()[i] >= -1.0);
            assert!(output.upper()[i] <= 1.0);
        }
    }

    #[test]
    fn concrete_interval_center_tracks_onnx_float_erf() {
        let values = arr1(&[-8.0_f32, -2.0, -0.25, 0.0, 0.25, 2.0, 8.0]).into_dyn();
        let input = BoundedTensor::concrete(values.clone()).unwrap();
        let output = ErfLayer::new().propagate_ibp(&input).unwrap();
        let center = output.center();
        for (actual, &x) in center.iter().zip(values.iter()) {
            let expected = libm::erff(x);
            assert!(
                (*actual - expected).abs() <= 2.0 * f32::EPSILON,
                "Erf point evaluation at {x}: center={actual}, ONNX-f32 policy={expected}"
            );
        }
    }

    #[test]
    fn affine_relaxations_are_sound_across_all_convexity_cases() {
        for (l, u) in [
            (-8.0, -2.0),
            (-2.0, -0.125),
            (-3.0, 0.25),
            (-0.25, 3.0),
            (-3.0, 3.0),
            (0.125, 2.0),
            (2.0, 8.0),
        ] {
            let relaxation = erf_linear_relaxation(l, u);
            assert_relaxation_sound(l, u, relaxation, libm::erff, 2.0e-6, "erf");
        }
    }

    #[test]
    fn dense_crown_backward_is_sound_for_positive_and_negative_coefficients() {
        let intervals = [(-3.0, 3.0), (-6.0, -2.0), (-1.0, 1.0), (1.0, 6.0)];
        assert_crown_backward_sound(&ErfLayer::new(), &intervals, libm::erff);
    }
}
