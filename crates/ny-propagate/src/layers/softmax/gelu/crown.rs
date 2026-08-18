// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scalar and batched CROWN backward propagation through GELU.

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::heuristic_relax::adaptive_gelu_linear_relaxation;
use super::sound_relax::{
    gelu_sound_linear_relaxation, gelu_sound_linear_relaxation_with_alpha,
    gelu_tanh_sound_linear_relaxation,
};
use super::{GELULayer, GeluApproximation};
use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_batched_indexed, crown_elementwise_backward_patches,
    non_finite_domain_guard,
};
use crate::{BatchedLinearBounds, LinearBounds};

impl GELULayer {
    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// Similar to ReLU's propagate_linear_with_bounds, computes linear relaxation
    /// of GELU based on the input interval [l, u] and transforms the linear bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("GELU", pre_activation)?;
        debug!("GELU layer CROWN backward propagation with pre-activation bounds");
        let approx = self.approximation;
        let mode = self.relaxation_mode;
        let sound = self.is_sound();
        crown_elementwise_backward(bounds, pre_activation, move |l, u| {
            let (ls, li, us, ui) = if sound {
                match approx {
                    GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                    GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
                }
            } else {
                adaptive_gelu_linear_relaxation(l, u, approx, mode)
            };
            LinearRelaxation::new(ls, li, us, ui)
        })
    }

    /// Batched CROWN backward propagation through GELU with pre-activation bounds.
    ///
    /// Same as `propagate_linear_with_bounds` but operates on N-D batched bounds,
    /// preserving batch structure [...batch, dim].
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("GELU", pre_activation)?;
        debug!("GELU layer batched CROWN backward propagation");
        let approx = self.approximation;
        let mode = self.relaxation_mode;
        let sound = self.is_sound();
        crown_elementwise_backward_batched(bounds, pre_activation, move |l, u| {
            let (ls, li, us, ui) = if sound {
                match approx {
                    GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                    GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
                }
            } else {
                adaptive_gelu_linear_relaxation(l, u, approx, mode)
            };
            LinearRelaxation::new(ls, li, us, ui)
        })
    }

    /// Batched CROWN backward with per-neuron alpha parameters for alpha-CROWN optimization.
    ///
    /// Like `propagate_linear_batched_with_bounds` but uses `alpha` to parameterize
    /// the lower bound tangent point for GELU neurons in the convex region. This
    /// enables per-block alpha-CROWN optimization of GELU slopes.
    ///
    /// `alphas` has length `in_dim` — one alpha per neuron. Alpha values outside
    /// [0, 1] are clamped. Only applies to the Erf GELU sound path; for other
    /// configurations falls back to the standard relaxation.
    ///
    /// Part of #3221 Phase 4: alpha-CROWN per block.
    pub fn propagate_linear_batched_with_bounds_and_alpha(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
        alphas: &Array1<f32>,
    ) -> Result<BatchedLinearBounds> {
        // Only meaningful for sound Erf mode. Otherwise fall back.
        if !self.is_sound() || self.approximation != GeluApproximation::Erf {
            return self.propagate_linear_batched_with_bounds(bounds, pre_activation);
        }

        non_finite_domain_guard("GELU", pre_activation)?;
        debug!("GELU alpha-CROWN batched backward propagation");
        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }
        let in_dim = a_shape[a_shape.len() - 1];
        if alphas.len() != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![alphas.len()],
            });
        }
        crown_elementwise_backward_batched_indexed(bounds, pre_activation, |l, u, i| {
            let (ls, li, us, ui) = gelu_sound_linear_relaxation_with_alpha(l, u, alphas[i]);
            LinearRelaxation::new(ls, li, us, ui)
        })
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("GELU", pre_activation)?;
        let approx = self.approximation;
        let mode = self.relaxation_mode;
        let sound = self.is_sound();
        let relax_fn = move |l: f32, u: f32| -> LinearRelaxation {
            let (ls, li, us, ui) = if sound {
                match approx {
                    GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                    GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
                }
            } else {
                adaptive_gelu_linear_relaxation(l, u, approx, mode)
            };
            LinearRelaxation::new(ls, li, us, ui)
        };
        crown_elementwise_backward_patches(bounds, pre_activation, relax_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};

    use super::super::eval::gelu_eval;

    /// Build identity LinearBounds for n neurons: lower_a = upper_a = I, lower_b = upper_b = 0.
    fn identity_bounds(n: usize) -> LinearBounds {
        LinearBounds::new(
            Array2::eye(n),
            Array1::zeros(n),
            Array2::eye(n),
            Array1::zeros(n),
        )
        .unwrap()
    }

    // =========================================================================
    // propagate_linear_with_bounds — soundness
    // =========================================================================

    /// CROWN backward through GELU with identity bounds should produce valid linear relaxation.
    /// For each neuron i: lower_b[i] <= GELU(x_i) <= upper_b[i] + upper_a[i,i]*x_i.
    #[test]
    fn test_crown_identity_bounds_soundness() {
        let n = 4;
        let pre_lower = arr1(&[-2.0, -1.0, 0.0, 1.0]).into_dyn();
        let pre_upper = arr1(&[-0.5, 0.5, 1.0, 2.0]).into_dyn();
        let pre_act = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

        for sound in [true, false] {
            let layer = if sound {
                GELULayer::sound(GeluApproximation::Erf)
            } else {
                GELULayer::adaptive(GeluApproximation::Erf)
            };

            let bounds = identity_bounds(n);
            let result = layer
                .propagate_linear_with_bounds(&bounds, &pre_act)
                .unwrap();

            // Check soundness: for 11 sample points per neuron
            for i in 0..n {
                let l = pre_lower[i];
                let u = pre_upper[i];
                for k in 0..=10 {
                    let t = k as f32 / 10.0;
                    let x = l + (u - l) * t;
                    let gx = gelu_eval(x, GeluApproximation::Erf);
                    // With identity A, output[i] = A[i,:]*relaxation = relaxation(x_i)
                    let lower = result.lower_a[[i, i]] * x + result.lower_b[i];
                    let upper = result.upper_a[[i, i]] * x + result.upper_b[i];
                    assert!(
                        lower <= gx + 1e-4,
                        "sound={sound} neuron {i} x={x}: lower={lower} > GELU={gx}"
                    );
                    assert!(
                        upper >= gx - 1e-4,
                        "sound={sound} neuron {i} x={x}: upper={upper} < GELU={gx}"
                    );
                }
            }
        }
    }

    /// CROWN backward with non-identity coefficient matrix.
    #[test]
    fn test_crown_non_identity_coefficients() {
        let pre_lower = arr1(&[-1.0, 0.0]).into_dyn();
        let pre_upper = arr1(&[1.0, 2.0]).into_dyn();
        let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

        let layer = GELULayer::sound(GeluApproximation::Erf);

        // Output = W * GELU(x) where W = [[1, 2], [-1, 0.5]]
        let bounds = LinearBounds::new(
            ndarray::array![[1.0, 2.0], [-1.0, 0.5]],
            Array1::zeros(2),
            ndarray::array![[1.0, 2.0], [-1.0, 0.5]],
            Array1::zeros(2),
        )
        .unwrap();

        let result = layer
            .propagate_linear_with_bounds(&bounds, &pre_act)
            .unwrap();

        // Basic shape check
        assert_eq!(result.lower_a.shape(), &[2, 2]);
        assert_eq!(result.lower_b.len(), 2);
        assert_eq!(result.upper_a.shape(), &[2, 2]);
        assert_eq!(result.upper_b.len(), 2);

        // Bounds should not contain NaN
        assert!(
            !result.lower_a.iter().any(|x| x.is_nan()),
            "lower_a contains NaN"
        );
        assert!(
            !result.upper_a.iter().any(|x| x.is_nan()),
            "upper_a contains NaN"
        );
        assert!(
            !result.lower_b.iter().any(|x| x.is_nan()),
            "lower_b contains NaN"
        );
        assert!(
            !result.upper_b.iter().any(|x| x.is_nan()),
            "upper_b contains NaN"
        );
    }

    /// Shape mismatch: pre-activation size != bounds num_inputs should error.
    #[test]
    fn test_crown_shape_mismatch() {
        let pre_lower = arr1(&[-1.0, 0.0, 1.0]).into_dyn();
        let pre_upper = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

        let layer = GELULayer::sound(GeluApproximation::Erf);
        // Bounds for 2 neurons but pre_activation has 3
        let bounds = identity_bounds(2);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre_act);
        assert!(result.is_err(), "Should error on shape mismatch");
    }

    /// Tanh approximation should also produce sound results.
    #[test]
    fn test_crown_tanh_approx_soundness() {
        let n = 3;
        let pre_lower = arr1(&[-1.5, 0.0, 0.5]).into_dyn();
        let pre_upper = arr1(&[-0.5, 1.0, 2.0]).into_dyn();
        let pre_act = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

        let layer = GELULayer::sound(GeluApproximation::Tanh);
        let bounds = identity_bounds(n);
        let result = layer
            .propagate_linear_with_bounds(&bounds, &pre_act)
            .unwrap();

        for i in 0..n {
            let l = pre_lower[i];
            let u = pre_upper[i];
            for k in 0..=10 {
                let t = k as f32 / 10.0;
                let x = l + (u - l) * t;
                let gx = gelu_eval(x, GeluApproximation::Tanh);
                let lower = result.lower_a[[i, i]] * x + result.lower_b[i];
                let upper = result.upper_a[[i, i]] * x + result.upper_b[i];
                assert!(
                    lower <= gx + 1e-4,
                    "Tanh neuron {i} x={x}: lower={lower} > GELU={gx}"
                );
                assert!(
                    upper >= gx - 1e-4,
                    "Tanh neuron {i} x={x}: upper={upper} < GELU={gx}"
                );
            }
        }
    }

    // =========================================================================
    // propagate_linear_batched_with_bounds
    // =========================================================================

    /// Batched CROWN: 2D bounds (no batch dims) should work like non-batched.
    #[test]
    fn test_batched_crown_2d() {
        let n = 3;
        let pre_lower = arr1(&[-1.0, 0.0, 0.5]).into_dyn();
        let pre_upper = arr1(&[1.0, 1.0, 2.0]).into_dyn();
        let pre_act = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

        let layer = GELULayer::sound(GeluApproximation::Erf);

        // 2D batched bounds = [out_dim, in_dim]
        let eye = Array2::<f32>::eye(n);
        let zeros = Array1::<f32>::zeros(n);
        let batched = BatchedLinearBounds::from_parts_unchecked(
            eye.clone().into_dyn(),
            zeros.clone().into_dyn(),
            eye.into_dyn(),
            zeros.into_dyn(),
            vec![n],
            vec![n],
        );

        let result = layer
            .propagate_linear_batched_with_bounds(&batched, &pre_act)
            .unwrap();

        // Output shape should match
        assert_eq!(result.lower_a.shape(), &[n, n]);
        assert_eq!(result.lower_b.shape(), &[n]);

        // Soundness check for each neuron
        for i in 0..n {
            let l = pre_lower[i];
            let u = pre_upper[i];
            for k in 0..=10 {
                let t = k as f32 / 10.0;
                let x = l + (u - l) * t;
                let gx = gelu_eval(x, GeluApproximation::Erf);
                let lower = result.lower_a[[i, i]] * x + result.lower_b[i];
                let upper = result.upper_a[[i, i]] * x + result.upper_b[i];
                assert!(
                    lower <= gx + 1e-4,
                    "Batched 2D neuron {i} x={x}: lower={lower} > GELU={gx}"
                );
                assert!(
                    upper >= gx - 1e-4,
                    "Batched 2D neuron {i} x={x}: upper={upper} < GELU={gx}"
                );
            }
        }
    }

    /// Batched CROWN dimension check: <2 dims should error.
    #[test]
    fn test_batched_crown_too_few_dims() {
        let pre_lower = arr1(&[0.0]).into_dyn();
        let pre_upper = arr1(&[1.0]).into_dyn();
        let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

        let layer = GELULayer::sound(GeluApproximation::Erf);

        // 1D bounds should fail
        let batched = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&[3])),
            ArrayD::zeros(IxDyn(&[3])),
            ArrayD::zeros(IxDyn(&[3])),
            ArrayD::zeros(IxDyn(&[3])),
            vec![1],
            vec![3],
        );

        let result = layer.propagate_linear_batched_with_bounds(&batched, &pre_act);
        assert!(result.is_err(), "Should error on <2 dim bounds");
    }

    /// Batched CROWN: pre-activation/bounds dimension mismatch should error.
    #[test]
    fn test_batched_crown_dim_mismatch() {
        let pre_lower = arr1(&[0.0, 1.0]).into_dyn();
        let pre_upper = arr1(&[1.0, 2.0]).into_dyn();
        let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

        let layer = GELULayer::sound(GeluApproximation::Erf);

        // Bounds with in_dim=3 but pre_activation has in_dim=2
        let batched = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&[2, 3])),
            ArrayD::zeros(IxDyn(&[2])),
            ArrayD::zeros(IxDyn(&[2, 3])),
            ArrayD::zeros(IxDyn(&[2])),
            vec![3],
            vec![2],
        );

        let result = layer.propagate_linear_batched_with_bounds(&batched, &pre_act);
        assert!(result.is_err(), "Should error on dim mismatch");
    }
}
