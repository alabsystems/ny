// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::super::matmul::parse_matmul_dims;
use super::super::validate_mccormick_inputs;
use super::{BilinearCrownLayer, BilinearRelaxation};

impl BilinearCrownLayer {
    /// CROWN backward propagation with McCormick composition.
    ///
    /// This is the key method that enables N-D CROWN through attention.
    /// It uses `BilinearRelaxation::compose_backward_broadcast` to properly compose
    /// downstream bounds with McCormick envelope bounds.
    ///
    /// # Arguments
    /// * `downstream_bounds` - Linear bounds from layers after this matmul
    /// * `input_a_bounds` - Concrete bounds on Q (query)
    /// * `input_b_bounds` - Concrete bounds on K (key)
    ///
    /// # Returns
    /// Two sets of linear bounds: one for Q input, one for K input.
    pub fn propagate_linear_batched_binary(
        &self,
        downstream_bounds: &crate::BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(crate::BatchedLinearBounds, crate::BatchedLinearBounds)> {
        debug!("BilinearCrown CROWN backward with broadcast composition");

        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;
        if dims.batch_dims.contains(&0) {
            return Err(NyError::InvalidSpec(format!(
                "BilinearCrown: zero-valued batch dimension in {:?}",
                dims.batch_dims
            )));
        }

        validate_mccormick_inputs(input_a_bounds, input_b_bounds, "BilinearCrown")?;
        let z_size = dims.c_size_per_batch()?;
        let scale = self.scale.unwrap_or(1.0);

        // Validate scale: negative scale would flip lower/upper bounds incorrectly
        if scale < 0.0 {
            return Err(NyError::UnsupportedOp(
                "BilinearCrown does not support negative scale (would require bound swapping)"
                    .to_string(),
            ));
        }

        // N-D compose path (#286): when downstream has N-D structure (last dim = n,
        // not m*n) AND interval coefficients (from composition through >=1 layer),
        // compose along the n dimension only. K contribution is eagerly concretized
        // and folded into Q bias. This is tighter than partial CROWN when the
        // downstream has non-trivial interval structure.
        //
        // Skip for identity (point-valued) downstream: McCormick linearization at
        // the output node produces bounds inherently WIDER than IBP because the
        // McCormick lower plane is below the actual product curve. Partial CROWN
        // fallback (= IBP) gives tighter results for the output node case.
        let ds_last_dim = *downstream_bounds.lower_a().shape().last().unwrap_or(&0);
        let ds_has_interval_structure = downstream_bounds.lower_a() != downstream_bounds.upper_a();
        if ds_last_dim == dims.n && ds_last_dim != z_size && ds_has_interval_structure {
            debug!(
                "BilinearCrown using N-D one-sided compose (ds_last={}, n={}, m={}, k={})",
                ds_last_dim, dims.n, dims.m, dims.k
            );
            return self.propagate_nd_one_sided(
                downstream_bounds,
                input_a_bounds,
                input_b_bounds,
                dims.m,
                dims.n,
                dims.k,
                scale,
                &dims.batch_dims,
            );
        }

        // Per-batch broadcast McCormick backward (#286, Approach A):
        // Use per-batch-position element-wise [batch, m, n, k] coefficients composed
        // with downstream bounds via broadcast einsum, avoiding:
        //   1. Dense O(seq^4) identity matrix materialization
        //   2. Batch-reduced global intervals (conservative widening)
        //
        // For batch_size>1, produces tighter bounds via per-position intervals.
        //
        // Reference: auto_LiRPA operators/linear.py `propagate_A_xy` and
        // operators/bivariate.py `bound_backward_both_perturbed`
        // Design: designs/2026-03-04-286-attention-bilinear-alternative.md Approach A
        debug!(
            "BilinearCrown per-batch broadcast backward (m={}, n={}, k={})",
            dims.m, dims.n, dims.k
        );
        let relaxation = BilinearRelaxation::from_bounds(
            input_a_bounds,
            input_b_bounds,
            self.transpose_b,
            self.scale,
        )?;
        relaxation.compose_backward_broadcast(downstream_bounds)
    }

    /// CROWN backward propagation with optimizable alpha parameters.
    ///
    /// This variant accepts alpha parameters for McCormick plane interpolation,
    /// enabling tighter bounds through alpha-CROWN optimization.
    ///
    /// # Alpha Parameters
    ///
    /// The `alphas` array has shape [4, m, n, k] with direction-dependent face selection:
    /// - alphas[[0, i, j, l]] = r_l for positive downstream A
    /// - alphas[[1, i, j, l]] = r_l for negative downstream A
    /// - alphas[[2, i, j, l]] = r_u for positive downstream A
    /// - alphas[[3, i, j, l]] = r_u for negative downstream A
    ///
    /// Currently, slices [0] (r_l) and [2] (r_u) are used for the McCormick relaxation
    /// (positive-direction face selection). Full direction-dependent composition using
    /// all 4 slices is planned as a follow-up.
    ///
    /// Each r ∈ [0, 1] interpolates between two valid McCormick planes:
    /// - r=0: Uses L2/U2 plane (tight at upper corner)
    /// - r=1: Uses L1/U1 plane (tight at lower corner)
    ///
    /// Default initialization is r=1.0 (matching auto_LiRPA's torch.ones()).
    ///
    /// # Reference
    /// auto_LiRPA/operators/bivariate.py:MulHelper.get_relaxation
    pub fn propagate_linear_batched_binary_with_alpha(
        &self,
        downstream_bounds: &crate::BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: Option<&ndarray::Array4<f32>>,
    ) -> Result<(crate::BatchedLinearBounds, crate::BatchedLinearBounds)> {
        // If no alphas provided, use the existing fixed-selection method
        let alphas = match alphas {
            Some(a) => a,
            None => {
                return self.propagate_linear_batched_binary(
                    downstream_bounds,
                    input_a_bounds,
                    input_b_bounds,
                )
            }
        };

        debug!("BilinearCrown alpha-CROWN backward with interpolated McCormick");

        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;
        if dims.batch_dims.contains(&0) {
            return Err(NyError::InvalidSpec(format!(
                "BilinearCrown: zero-valued batch dimension in {:?}",
                dims.batch_dims
            )));
        }

        // Validate alpha shape: [4, m, n, k]
        let alpha_shape = alphas.shape();
        if alpha_shape != [4, dims.m, dims.n, dims.k] {
            return Err(NyError::ShapeMismatch {
                expected: vec![4, dims.m, dims.n, dims.k],
                got: alpha_shape.to_vec(),
            });
        }

        // Direction-dependent alpha: use all 4 slices from the alpha tensor.
        //
        // alpha[0] = r_l for lower-bound direction (pos(A_L) picks lower face)
        // alpha[1] = r_l for upper-bound direction (neg(A_U) picks lower face)
        // alpha[2] = r_u for lower-bound direction (neg(A_L) picks upper face)
        // alpha[3] = r_u for upper-bound direction (pos(A_U) picks upper face)
        //
        // We build two relaxation instances: one for lower-bound direction (using
        // alpha[0] and alpha[2]) and one for upper-bound direction (using alpha[1]
        // and alpha[3]). The bidirectional compose uses the appropriate relaxation
        // for each direction, giving the optimizer 4 independent degrees of freedom
        // per McCormick element.
        //
        // Reference: auto_LiRPA bivariate.py:306-315 (sign-dependent dispatch)
        // Design: designs/2026-03-04-286-attention-bilinear-alternative.md Approach B
        let mut rl_ru_lower = ndarray::Array4::zeros((2, dims.m, dims.n, dims.k));
        rl_ru_lower
            .slice_mut(ndarray::s![0, .., .., ..])
            .assign(&alphas.slice(ndarray::s![0, .., .., ..]));
        rl_ru_lower
            .slice_mut(ndarray::s![1, .., .., ..])
            .assign(&alphas.slice(ndarray::s![2, .., .., ..]));

        let mut rl_ru_upper = ndarray::Array4::zeros((2, dims.m, dims.n, dims.k));
        rl_ru_upper
            .slice_mut(ndarray::s![0, .., .., ..])
            .assign(&alphas.slice(ndarray::s![1, .., .., ..]));
        rl_ru_upper
            .slice_mut(ndarray::s![1, .., .., ..])
            .assign(&alphas.slice(ndarray::s![3, .., .., ..]));

        validate_mccormick_inputs(input_a_bounds, input_b_bounds, "BilinearCrown")?;

        // Per-batch broadcast McCormick backward with direction-dependent alpha (#286, Approach B):
        // Lower-bound direction uses relax_lower (from alpha[0], alpha[2]).
        // Upper-bound direction uses relax_upper (from alpha[1], alpha[3]).
        //
        // Reference: auto_LiRPA operators/bivariate.py `MulHelper.get_relaxation`
        // Design: designs/2026-03-04-286-attention-bilinear-alternative.md Approach A+B
        debug!(
            "BilinearCrown per-batch direction-dependent alpha broadcast backward (m={}, n={}, k={})",
            dims.m, dims.n, dims.k
        );
        let relax_lower = BilinearRelaxation::from_bounds_with_alpha(
            input_a_bounds,
            input_b_bounds,
            self.transpose_b,
            self.scale,
            &rl_ru_lower,
        )?;
        let relax_upper = BilinearRelaxation::from_bounds_with_alpha(
            input_a_bounds,
            input_b_bounds,
            self.transpose_b,
            self.scale,
            &rl_ru_upper,
        )?;
        relax_lower.compose_backward_broadcast_bidirectional(&relax_upper, downstream_bounds)
    }
}
