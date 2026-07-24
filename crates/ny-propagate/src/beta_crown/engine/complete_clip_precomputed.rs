// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Complete clipping with pre-computed CROWN linear bounds.
//!
//! Multi-objective conjunctive input-split path: reuses CROWN linear bounds
//! from the multi-row spec pass to avoid redundant CROWN passes per child.
//! Extends `clip_with_precomputed_linear` (relaxed_clip.rs) with the
//! Lagrangian LP step from `complete_clip_engine.rs`.
//!
//! Reference: alpha-beta-CROWN `input_split/clip.py:174-227`

use ndarray::{Array2, Array3};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::complete_clip::complete_clip;
use crate::complete_clip::filter::sort_out_constraints;
use crate::relaxed_clip::relaxed_clip;

use super::complete_clip_engine::{add_certified_lower_bias, construct_constraints};
use super::{BetaCrownVerifier, RelaxedClipOutcome};

impl BetaCrownVerifier {
    /// Complete clipping variant that reuses all pre-computed CROWN rows from
    /// the multi-objective spec pass in one shared clip step.
    ///
    /// This mirrors the alpha-beta-CROWN input-split flow: relaxed clipping runs
    /// once on the full `(n_spec, x_dim)` bundle, and the LP branch then sees
    /// all cross-spec constraints together instead of an unreachable `n_spec=1`
    /// per-objective loop.
    ///
    /// Reference: alpha-beta-CROWN `input_split/clip.py:174-227`
    pub(in crate::beta_crown::engine) fn complete_clip_with_precomputed_specs(
        &self,
        input_bounds: &BoundedTensor,
        original_shape: &[usize],
        linear_bounds: &crate::bounds::LinearBounds,
        thresholds: &[f32],
    ) -> Result<RelaxedClipOutcome> {
        let n_rows = linear_bounds.lower_a().nrows();
        if n_rows == 0 || thresholds.is_empty() {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds.clone(),
                verified: false,
            });
        }
        if n_rows != thresholds.len() {
            return Err(ny_core::NyError::shape_mismatch(
                vec![n_rows],
                vec![thresholds.len()],
            ));
        }

        // The pre-computed coefficients may still carry their certified error
        // envelope; discharge it over this child's box before the rows drive
        // the verified pre-check and the clip constraints.
        let folded_bounds;
        let linear_bounds = if linear_bounds.has_coeff_err() {
            let mut folded = linear_bounds.clone();
            Self::discharge_coeff_err_for_clip(&mut folded, input_bounds);
            folded_bounds = folded;
            &folded_bounds
        } else {
            linear_bounds
        };

        let (coeffs, biases, threshold_values) = if self.config.verify_upper_bound {
            (
                linear_bounds.upper_a().mapv(|v| -v),
                linear_bounds.upper_b().mapv(|v| -v),
                thresholds.iter().map(|&threshold| -threshold).collect(),
            )
        } else {
            (
                linear_bounds.lower_a().clone(),
                linear_bounds.lower_b().clone(),
                thresholds.to_vec(),
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
            .into_shape_clone((1, n_rows, x_dim))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lA: {}", e)))?;
        let lbias: Array2<f32> = biases
            .into_shape_clone((1, n_rows))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lbias: {}", e)))?;
        let thresholds = Array2::from_shape_vec((1, n_rows), threshold_values)
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape thresholds: {}", e)))?;

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
        if Self::any_verified(&dm_lb_post, &thresholds) {
            let new_lower = new_l
                .into_shape_clone(ndarray::IxDyn(original_shape))
                .map_err(|e| {
                    ny_core::NyError::InvalidSpec(format!("reshape clipped lower: {}", e))
                })?;
            let new_upper = new_u
                .into_shape_clone(ndarray::IxDyn(original_shape))
                .map_err(|e| {
                    ny_core::NyError::InvalidSpec(format!("reshape clipped upper: {}", e))
                })?;
            return Ok(RelaxedClipOutcome {
                bounds: BoundedTensor::new(new_lower, new_upper)?,
                verified: true,
            });
        }

        let verified = if n_rows > 1 {
            let clip_x_l = new_l
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clip_x_l: {}", e)))?;
            let clip_x_u = new_u
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clip_x_u: {}", e)))?;
            let (constr_a, constr_b) = construct_constraints(&l_a, &lbias, &thresholds)?;
            let filter_result = sort_out_constraints(&constr_a, &constr_b, &clip_x_l, &clip_x_u);

            if filter_result.any_infeasible() {
                true
            } else if filter_result.all_fully_covered() {
                false
            } else {
                match complete_clip(
                    &clip_x_l.into_dyn(),
                    &clip_x_u.into_dyn(),
                    &l_a.into_dyn(),
                    &constr_a.into_dyn(),
                    &constr_b.into_dyn(),
                    -1.0_f32,
                    true,
                    1,
                ) {
                    Ok(constrained_lb) => {
                        match constrained_lb.into_dimensionality::<ndarray::Ix2>() {
                            Ok(constrained_lb_2d) => {
                                let output_lb =
                                    add_certified_lower_bias(&constrained_lb_2d, &lbias)?;
                                Self::any_verified(&output_lb, &thresholds)
                            }
                            Err(_) => false,
                        }
                    }
                    Err(ny_core::NyError::InfeasibleDomain(_)) => true,
                    Err(_) => false,
                }
            }
        } else {
            false
        };

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
}
