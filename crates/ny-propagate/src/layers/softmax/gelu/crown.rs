// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scalar and batched CROWN backward propagation through GELU.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use super::heuristic_relax::adaptive_gelu_linear_relaxation;
use super::sound_relax::{
    gelu_sound_linear_relaxation, gelu_sound_linear_relaxation_with_alpha,
    gelu_tanh_sound_linear_relaxation,
};
use super::{GELULayer, GeluApproximation};
use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{crown_elementwise_backward_patches, non_finite_domain_guard};
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

        let num_outputs = bounds.num_outputs();

        // Compute relaxation parameters for each neuron
        let mut lower_slopes = Array1::<f32>::zeros(num_neurons);
        let mut lower_intercepts = Array1::<f32>::zeros(num_neurons);
        let mut upper_slopes = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercepts = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];
            let (ls, li, us, ui) = if self.is_sound() {
                match self.approximation {
                    GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                    GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
                }
            } else {
                adaptive_gelu_linear_relaxation(l, u, self.approximation, self.relaxation_mode)
            };
            lower_slopes[i] = ls;
            lower_intercepts[i] = li;
            upper_slopes[i] = us;
            upper_intercepts[i] = ui;
        }

        // Backward propagation through GELU
        // Same pattern as ReLU: choose relaxation based on coefficient sign.
        // Bias accumulation uses f64 to prevent catastrophic cancellation (#1745),
        // with directed rounding on final f32 cast (#1992, #2164).
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a()[[j, i]];
                let ua = bounds.upper_a()[[j, i]];

                // For lower bound: use lower relaxation when coeff is positive, upper when negative.
                // Guard: skip zero coefficients to avoid IEEE 754 NaN from 0.0 * ±inf (#1736).
                // Directed rounding on slope products (#2786): next_down_f32 for lower_a,
                // next_up_f32 for upper_a. Moves bounds away from true value (sound).
                if la > 0.0 {
                    new_lower_a[[j, i]] = next_down_f32(la * lower_slopes[i]);
                    new_lower_b_f64[j] += la as f64 * lower_intercepts[i] as f64;
                } else if la < 0.0 {
                    new_lower_a[[j, i]] = next_down_f32(la * upper_slopes[i]);
                    new_lower_b_f64[j] += la as f64 * upper_intercepts[i] as f64;
                }

                // For upper bound: use upper relaxation when coeff is positive, lower when negative.
                if ua > 0.0 {
                    new_upper_a[[j, i]] = next_up_f32(ua * upper_slopes[i]);
                    new_upper_b_f64[j] += ua as f64 * upper_intercepts[i] as f64;
                } else if ua < 0.0 {
                    new_upper_a[[j, i]] = next_up_f32(ua * lower_slopes[i]);
                    new_upper_b_f64[j] += ua as f64 * lower_intercepts[i] as f64;
                }
            }
        }

        LinearBounds::new_or_conservative(
            new_lower_a,
            new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
            new_upper_a,
            new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
        )
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

        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let in_dim = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "GELU batched CROWN: batch dims product overflows usize: {:?}",
                batch_dims,
            ))
        })?;
        let total_batch = total_batch.max(1);

        let pre_in_dim = *pre_shape.last().unwrap_or(&0);
        if pre_in_dim != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![pre_in_dim],
            });
        }

        // Reshape pre-activation to [batch, in_dim]
        let pre_lower_flat = pre_activation
            .lower()
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape pre_lower".to_string()))?;
        let pre_upper_flat = pre_activation
            .upper()
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape pre_upper".to_string()))?;

        // Reshape bounds
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        // Output arrays
        // Bias accumulation uses f64 to prevent catastrophic cancellation (#1745, #2164).
        let mut new_lower_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_upper_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_lower_b_f64 = Array2::<f64>::zeros((total_batch, out_dim));
        let mut new_upper_b_f64 = Array2::<f64>::zeros((total_batch, out_dim));

        // Copy initial bias
        for b in 0..total_batch {
            for j in 0..out_dim {
                new_lower_b_f64[[b, j]] = lower_b_2d[[b, j]] as f64;
                new_upper_b_f64[[b, j]] = upper_b_2d[[b, j]] as f64;
            }
        }

        // Process each batch position
        for b in 0..total_batch {
            for i in 0..in_dim {
                let l = pre_lower_flat[[b, i]];
                let u = pre_upper_flat[[b, i]];
                let (lower_slope, lower_intercept, upper_slope, upper_intercept) = if self
                    .is_sound()
                {
                    match self.approximation {
                        GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                        GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
                    }
                } else {
                    adaptive_gelu_linear_relaxation(l, u, self.approximation, self.relaxation_mode)
                };

                for j in 0..out_dim {
                    let la = lower_a_3d[[b, j, i]];
                    let ua = upper_a_3d[[b, j, i]];
                    let row_idx = b * out_dim + j;

                    // For lower bound: skip zero coefficients to avoid 0.0 * ±inf = NaN (#1736).
                    // Directed rounding on slope products (#2786): next_down_f32 for lower_a,
                    // next_up_f32 for upper_a. Moves bounds away from true value (sound).
                    if la > 0.0 {
                        new_lower_a[[row_idx, i]] = next_down_f32(la * lower_slope);
                        new_lower_b_f64[[b, j]] += la as f64 * lower_intercept as f64;
                    } else if la < 0.0 {
                        new_lower_a[[row_idx, i]] = next_down_f32(la * upper_slope);
                        new_lower_b_f64[[b, j]] += la as f64 * upper_intercept as f64;
                    }

                    // For upper bound: skip zero coefficients to avoid 0.0 * ±inf = NaN (#1736).
                    if ua > 0.0 {
                        new_upper_a[[row_idx, i]] = next_up_f32(ua * upper_slope);
                        new_upper_b_f64[[b, j]] += ua as f64 * upper_intercept as f64;
                    } else if ua < 0.0 {
                        new_upper_a[[row_idx, i]] = next_up_f32(ua * lower_slope);
                        new_upper_b_f64[[b, j]] += ua as f64 * lower_intercept as f64;
                    }
                }
            }
        }

        // Cast bias back to f32 with directed rounding (#1992, #2164)
        let new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
        let new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));

        // Reshape back
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .cloned()
            .chain([out_dim, in_dim])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();
        // Validated construction: finite coeff × finite GELU slopes, NaN firewall (#3033).
        BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
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

        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let in_dim = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "GELU alpha batched CROWN: batch dims overflow: {:?}",
                batch_dims,
            ))
        })?;
        let total_batch = total_batch.max(1);

        let pre_in_dim = *pre_shape.last().unwrap_or(&0);
        if pre_in_dim != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![pre_in_dim],
            });
        }

        if alphas.len() != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![alphas.len()],
            });
        }

        let pre_lower_flat = pre_activation
            .lower()
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape pre_lower".to_string()))?;
        let pre_upper_flat = pre_activation
            .upper()
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape pre_upper".to_string()))?;

        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        let mut new_lower_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_upper_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_lower_b_f64 = Array2::<f64>::zeros((total_batch, out_dim));
        let mut new_upper_b_f64 = Array2::<f64>::zeros((total_batch, out_dim));

        for b in 0..total_batch {
            for j in 0..out_dim {
                new_lower_b_f64[[b, j]] = lower_b_2d[[b, j]] as f64;
                new_upper_b_f64[[b, j]] = upper_b_2d[[b, j]] as f64;
            }
        }

        for b in 0..total_batch {
            for i in 0..in_dim {
                let l = pre_lower_flat[[b, i]];
                let u = pre_upper_flat[[b, i]];
                // Use alpha-parameterized relaxation for the lower bound.
                let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                    gelu_sound_linear_relaxation_with_alpha(l, u, alphas[i]);

                for j in 0..out_dim {
                    let la = lower_a_3d[[b, j, i]];
                    let ua = upper_a_3d[[b, j, i]];
                    let row_idx = b * out_dim + j;

                    if la > 0.0 {
                        new_lower_a[[row_idx, i]] = next_down_f32(la * lower_slope);
                        new_lower_b_f64[[b, j]] += la as f64 * lower_intercept as f64;
                    } else if la < 0.0 {
                        new_lower_a[[row_idx, i]] = next_down_f32(la * upper_slope);
                        new_lower_b_f64[[b, j]] += la as f64 * upper_intercept as f64;
                    }

                    if ua > 0.0 {
                        new_upper_a[[row_idx, i]] = next_up_f32(ua * upper_slope);
                        new_upper_b_f64[[b, j]] += ua as f64 * upper_intercept as f64;
                    } else if ua < 0.0 {
                        new_upper_a[[row_idx, i]] = next_up_f32(ua * lower_slope);
                        new_upper_b_f64[[b, j]] += ua as f64 * lower_intercept as f64;
                    }
                }
            }
        }

        let new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
        let new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));

        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .cloned()
            .chain([out_dim, in_dim])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();
        BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
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
    use ndarray::{arr1, Array1, Array2};

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
