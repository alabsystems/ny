// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Relaxed clipping (Clip-and-Verify) utilities.

use ndarray::{Array2, ArrayD};
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::relaxed_clip::relaxed_clip;
use crate::{GraphNetwork, Network};

use super::BetaCrownVerifier;
use super::RelaxedClipOutcome;

impl BetaCrownVerifier {
    /// Apply relaxed clipping (Clip-and-Verify) to tighten input bounds.
    ///
    /// This uses CROWN linear constraints to tighten input bounds further
    /// after a dimension split. The algorithm computes the linear coefficients
    /// from CROWN backward propagation and uses them to shrink the input domain.
    ///
    /// # Arguments
    ///
    /// * `network` - The neural network
    /// * `input_bounds` - Initial input bounds (after split)
    /// * `original_shape` - Original shape for reshaping results
    /// * `threshold` - Verification threshold (used to build clipping constraints)
    ///
    /// # Returns
    ///
    /// Tightened input bounds (may be unchanged if clipping had no effect).
    ///
    /// # References
    ///
    /// - `designs/2026-01-28-clip-and-verify-algorithms.md`
    /// - `alpha-beta-CROWN/complete_verifier/input_split/clip.py`
    pub(in crate::beta_crown::engine) fn apply_relaxed_clipping(
        &self,
        network: &Network,
        input_bounds: BoundedTensor,
        original_shape: &[usize],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<RelaxedClipOutcome> {
        use ndarray::{Array1, Array2, Array3};

        // Get CROWN linear bounds at input layer (#3598: thread engine for GPU acceleration)
        let (_output_bounds, mut linear_bounds) =
            network.propagate_crown_with_linear_and_engine(&input_bounds, engine)?;
        Self::discharge_coeff_err_for_clip(&mut linear_bounds, &input_bounds);

        // Select bounds based on verification direction.
        // For upper-bound verification (output <= threshold), we clip using the constraint
        // uA·x + ub >= threshold, which we rewrite as (-uA)·x + (-ub) <= -threshold.
        let (coeffs, biases, threshold_value): (Array2<f32>, Array1<f32>, f32) =
            if self.config.verify_upper_bound {
                (
                    linear_bounds.upper_a().mapv(|v| -v),
                    linear_bounds.upper_b().mapv(|v| -v),
                    -threshold,
                )
            } else {
                (
                    linear_bounds.lower_a().clone(),
                    linear_bounds.lower_b().clone(),
                    threshold,
                )
            };
        let is_lower = true;

        // Convert to relaxed_clip expected format:
        // - x_l, x_u: (batch=1, x_dim)
        // - lA: (batch=1, n_spec, x_dim)
        // - lbias: (batch=1, n_spec)
        // - thresholds: (batch=1, n_spec)

        let flat = input_bounds.flatten();
        let x_dim = flat.lower().len();
        let n_spec = coeffs.nrows();

        // Flatten input bounds to (1, x_dim)
        let x_l: Array2<f32> = flat
            .lower()
            .to_owned()
            .into_shape_clone((1, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape x_l: {}", e)))?;
        let x_u: Array2<f32> = flat
            .upper()
            .to_owned()
            .into_shape_clone((1, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape x_u: {}", e)))?;

        // Convert lower_a from (n_spec, x_dim) to (1, n_spec, x_dim)
        let l_a_flat = coeffs;
        let l_a: Array3<f32> = l_a_flat
            .into_shape_clone((1, n_spec, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lA: {}", e)))?;

        // Convert lower_b from (n_spec,) to (1, n_spec)
        let lbias: Array2<f32> = biases
            .into_shape_clone((1, n_spec))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lbias: {}", e)))?;

        // Thresholds come from the verification objective
        let thresholds: Array2<f32> = Array2::from_elem((1, n_spec), threshold_value);

        let dm_lb_pre = Self::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, is_lower);
        if Self::any_verified(&dm_lb_pre, &thresholds) {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds,
                verified: true,
            });
        }

        // Apply relaxed clipping
        let (new_l, new_u) = relaxed_clip(
            &x_l.into_dyn(),
            &x_u.into_dyn(),
            &l_a.clone().into_dyn(),
            &lbias.clone().into_dyn(),
            &thresholds.clone().into_dyn(),
            self.config.relaxed_clip_iterations,
            is_lower, // always lower bound form after normalization
        )?;

        let dm_lb_post = Self::concretize_dm_lb_from_dyn(&new_l, &new_u, &l_a, &lbias, is_lower);
        let verified = Self::any_verified(&dm_lb_post, &thresholds);

        // Reshape results back to original shape
        let new_lower = new_l
            .into_shape_clone(ndarray::IxDyn(original_shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clipped lower: {}", e)))?;
        let new_upper = new_u
            .into_shape_clone(ndarray::IxDyn(original_shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clipped upper: {}", e)))?;

        Ok(RelaxedClipOutcome {
            bounds: BoundedTensor::new(new_lower, new_upper)?,
            verified,
        })
    }

    /// Clip using pre-computed linear bounds from a multi-row CROWN pass.
    ///
    /// This avoids a separate CROWN pass per objective by reusing linear bounds
    /// that were already computed for the full spec matrix. Extracts the row at
    /// `obj_idx` from the multi-row `LinearBounds` and runs the clipping algorithm.
    ///
    /// Sound: CROWN linear bounds computed on a domain D are valid lower/upper
    /// bounds for any subset D' ⊆ D. When the input has been progressively
    /// tightened by prior clipping passes, the pre-computed coefficients remain
    /// valid but may be slightly less tight than a fresh CROWN pass.
    ///
    /// Reference: alpha-beta-CROWN input_split/clip.py:174-227
    pub(in crate::beta_crown::engine) fn clip_with_precomputed_linear(
        &self,
        input_bounds: &BoundedTensor,
        original_shape: &[usize],
        linear_bounds: &crate::bounds::LinearBounds,
        obj_idx: usize,
        threshold: f32,
    ) -> Result<RelaxedClipOutcome> {
        use ndarray::{Array2, Array3};

        let n_rows = linear_bounds.lower_a().nrows();
        if obj_idx >= n_rows {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds.clone(),
                verified: false,
            });
        }

        // The pre-computed coefficients may still carry their certified error
        // envelope; discharge it over this child's box before the row drives
        // the verified pre-check and the clip constraint.
        let folded_bounds;
        let linear_bounds = if linear_bounds.has_coeff_err() {
            let mut folded = linear_bounds.clone();
            Self::discharge_coeff_err_for_clip(&mut folded, input_bounds);
            folded_bounds = folded;
            &folded_bounds
        } else {
            linear_bounds
        };

        let (coeffs, biases, threshold_value) = if self.config.verify_upper_bound {
            (
                linear_bounds.upper_a().row(obj_idx).mapv(|v| -v),
                ndarray::Array1::from_elem(1, -linear_bounds.upper_b()[obj_idx]),
                -threshold,
            )
        } else {
            (
                linear_bounds.lower_a().row(obj_idx).to_owned(),
                ndarray::Array1::from_elem(1, linear_bounds.lower_b()[obj_idx]),
                threshold,
            )
        };
        let is_lower = true;

        let flat = input_bounds.flatten();
        let x_dim = flat.lower().len();

        let x_l: Array2<f32> = flat
            .lower()
            .to_owned()
            .into_shape_clone((1, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape x_l: {}", e)))?;
        let x_u: Array2<f32> = flat
            .upper()
            .to_owned()
            .into_shape_clone((1, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape x_u: {}", e)))?;

        let l_a: Array3<f32> = coeffs
            .into_shape_clone((1, 1, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lA: {}", e)))?;
        let lbias: Array2<f32> = biases
            .into_shape_clone((1, 1))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lbias: {}", e)))?;
        let thresholds: Array2<f32> = Array2::from_elem((1, 1), threshold_value);

        let dm_lb_pre = Self::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, is_lower);
        if Self::any_verified(&dm_lb_pre, &thresholds) {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds.clone(),
                verified: true,
            });
        }

        let (new_l, new_u) = relaxed_clip(
            &x_l.into_dyn(),
            &x_u.into_dyn(),
            &l_a.clone().into_dyn(),
            &lbias.clone().into_dyn(),
            &thresholds.clone().into_dyn(),
            self.config.relaxed_clip_iterations,
            is_lower,
        )?;

        let dm_lb_post = Self::concretize_dm_lb_from_dyn(&new_l, &new_u, &l_a, &lbias, is_lower);
        let verified = Self::any_verified(&dm_lb_post, &thresholds);

        let new_lower = new_l
            .into_shape_clone(ndarray::IxDyn(original_shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clipped lower: {}", e)))?;
        let new_upper = new_u
            .into_shape_clone(ndarray::IxDyn(original_shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clipped upper: {}", e)))?;

        Ok(RelaxedClipOutcome {
            bounds: BoundedTensor::new(new_lower, new_upper)?,
            verified,
        })
    }

    /// GraphNetwork variant of relaxed clipping using linearized spec bounds.
    pub(in crate::beta_crown::engine) fn apply_relaxed_clipping_graph(
        &self,
        graph: &GraphNetwork,
        input_bounds: &BoundedTensor,
        original_shape: &[usize],
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<RelaxedClipOutcome> {
        use ndarray::{Array2, Array3};

        if objective.is_empty() {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds.clone(),
                verified: false,
            });
        }

        let spec_matrix = Array2::from_shape_vec((1, objective.len()), objective.to_vec())
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("spec matrix: {}", e)))?;

        let (_spec_bounds, linear_opt) = graph.propagate_crown_with_specs_and_engine_with_linear(
            input_bounds,
            &spec_matrix,
            engine,
        )?;

        let mut linear_bounds = match linear_opt {
            Some(bounds) => bounds,
            None => {
                return Ok(RelaxedClipOutcome {
                    bounds: input_bounds.clone(),
                    verified: false,
                })
            }
        };
        Self::discharge_coeff_err_for_clip(&mut linear_bounds, input_bounds);

        let (coeffs, biases, threshold_value) = if self.config.verify_upper_bound {
            (
                linear_bounds.upper_a().mapv(|v| -v),
                linear_bounds.upper_b().mapv(|v| -v),
                -threshold,
            )
        } else {
            (
                linear_bounds.lower_a().clone(),
                linear_bounds.lower_b().clone(),
                threshold,
            )
        };
        let is_lower = true;

        let flat = input_bounds.flatten();
        let x_dim = flat.lower().len();
        let n_spec = coeffs.nrows();

        let x_l: Array2<f32> = flat
            .lower()
            .to_owned()
            .into_shape_clone((1, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape x_l: {}", e)))?;
        let x_u: Array2<f32> = flat
            .upper()
            .to_owned()
            .into_shape_clone((1, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape x_u: {}", e)))?;

        let l_a: Array3<f32> = coeffs
            .into_shape_clone((1, n_spec, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lA: {}", e)))?;
        let lbias: Array2<f32> = biases
            .into_shape_clone((1, n_spec))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lbias: {}", e)))?;
        let thresholds: Array2<f32> = Array2::from_elem((1, n_spec), threshold_value);

        let dm_lb_pre = Self::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, is_lower);
        if Self::any_verified(&dm_lb_pre, &thresholds) {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds.clone(),
                verified: true,
            });
        }

        let (new_l, new_u) = relaxed_clip(
            &x_l.into_dyn(),
            &x_u.into_dyn(),
            &l_a.clone().into_dyn(),
            &lbias.clone().into_dyn(),
            &thresholds.clone().into_dyn(),
            self.config.relaxed_clip_iterations,
            is_lower,
        )?;

        let dm_lb_post = Self::concretize_dm_lb_from_dyn(&new_l, &new_u, &l_a, &lbias, is_lower);
        let verified = Self::any_verified(&dm_lb_post, &thresholds);

        let new_lower = new_l
            .into_shape_clone(ndarray::IxDyn(original_shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clipped lower: {}", e)))?;
        let new_upper = new_u
            .into_shape_clone(ndarray::IxDyn(original_shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clipped upper: {}", e)))?;

        Ok(RelaxedClipOutcome {
            bounds: BoundedTensor::new(new_lower, new_upper)?,
            verified,
        })
    }

    /// Discharge any certified coefficient-error envelope carried by CROWN
    /// linear bounds into the bias over the given input box
    /// (#vnncomp-aw-soundness). The stored f32 coefficients are certified only
    /// to within `lower_a_err`/`upper_a_err`, so feeding them raw into the
    /// clip constraint or the `any_verified` pre-check could claim a margin
    /// the true coefficients do not entail. Rows whose penalty is non-finite
    /// degrade to a ±inf bias, which can never verify and never clips.
    pub(in crate::beta_crown::engine) fn discharge_coeff_err_for_clip(
        linear_bounds: &mut crate::bounds::LinearBounds,
        input_bounds: &BoundedTensor,
    ) {
        if !linear_bounds.has_coeff_err() {
            return;
        }
        let flat = input_bounds.flatten();
        match (flat.lower().as_slice(), flat.upper().as_slice()) {
            (Some(in_l), Some(in_u)) => linear_bounds.fold_coeff_err_into_bias(in_l, in_u),
            // `flatten()` yields standard-layout 1-D arrays, so this arm is
            // unreachable; degrade rather than assume.
            _ => linear_bounds.discharge_coeff_err_to_conservative(),
        }
    }

    pub(in crate::beta_crown::engine) fn concretize_dm_lb(
        x_l: &Array2<f32>,
        x_u: &Array2<f32>,
        l_a: &ndarray::Array3<f32>,
        lbias: &Array2<f32>,
        is_lower: bool,
    ) -> Array2<f32> {
        let x_hat = (x_l + x_u) / 2.0;
        let eps = (x_u - x_l) / 2.0;
        crate::relaxed_clip::concretize_bounds(
            &x_hat.into_dyn(),
            &eps.into_dyn(),
            &l_a.clone().into_dyn(),
            &lbias.clone().into_dyn(),
            is_lower,
        )
    }

    pub(in crate::beta_crown::engine) fn concretize_dm_lb_from_dyn(
        x_l: &ArrayD<f32>,
        x_u: &ArrayD<f32>,
        l_a: &ndarray::Array3<f32>,
        lbias: &Array2<f32>,
        is_lower: bool,
    ) -> Array2<f32> {
        let x_hat = (x_u + x_l) / 2.0;
        let eps = (x_u - x_l) / 2.0;
        crate::relaxed_clip::concretize_bounds(
            &x_hat,
            &eps,
            &l_a.clone().into_dyn(),
            &lbias.clone().into_dyn(),
            is_lower,
        )
    }

    pub(in crate::beta_crown::engine) fn any_verified(
        dm_lb: &Array2<f32>,
        thresholds: &Array2<f32>,
    ) -> bool {
        if dm_lb.shape() != thresholds.shape() {
            return false;
        }
        let batch = dm_lb.shape()[0];
        let n_spec = dm_lb.shape()[1];
        for b in 0..batch {
            let mut any = false;
            for s in 0..n_spec {
                if dm_lb[[b, s]] > thresholds[[b, s]] {
                    any = true;
                    break;
                }
            }
            if any {
                return true;
            }
        }
        false
    }
}
