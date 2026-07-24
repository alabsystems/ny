// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective relaxed clipping helpers.
//!
//! Two variants:
//! - `clip_multi_objective_with_precomputed_linear`: returns a conjunctive
//!   `RelaxedClipOutcome` (collapses per-row bounds via `any_verified`). Used by
//!   conjunctive callers and existing tests.
//! - `clip_multi_objective_grouped_safe`: returns a `MultiSpecRelaxedClipOutcome`
//!   with per-row lower bounds and an infeasibility flag. Grouped/disjunctive
//!   callers use these for their own OR-within-clause reduction (see `#3740`).

use ndarray::{Array2, Array3};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::relaxed_clip::relaxed_clip_with_infeasible_mask;

#[cfg(test)]
use super::RelaxedClipOutcome;
use super::{BetaCrownVerifier, MultiSpecRelaxedClipOutcome};

/// Intermediate tensors shared by both conjunctive and grouped-safe clip paths.
struct MultiSpecClipSetup {
    x_l: Array2<f32>,
    x_u: Array2<f32>,
    l_a: Array3<f32>,
    lbias: Array2<f32>,
    thresholds_arr: Array2<f32>,
    n_rows: usize,
}

impl BetaCrownVerifier {
    /// Prepare coefficient tensors for multi-spec clipping.
    fn prepare_multi_spec_clip(
        &self,
        input_bounds: &BoundedTensor,
        linear_bounds: &crate::bounds::LinearBounds,
        thresholds: &[f32],
    ) -> Result<MultiSpecClipSetup> {
        let n_rows = linear_bounds.lower_a().nrows();
        if n_rows != thresholds.len() {
            return Err(ny_core::NyError::shape_mismatch(
                vec![n_rows],
                vec![thresholds.len()],
            ));
        }

        let (coeffs, biases, threshold_values) = if self.config.verify_upper_bound {
            (
                linear_bounds.upper_a().mapv(|v| -v),
                linear_bounds.upper_b().mapv(|v| -v),
                thresholds.iter().map(|&t| -t).collect::<Vec<_>>(),
            )
        } else {
            (
                linear_bounds.lower_a().clone(),
                linear_bounds.lower_b().clone(),
                thresholds.to_vec(),
            )
        };

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
        let thresholds_arr = Array2::from_shape_vec((1, n_rows), threshold_values)
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape thresholds: {}", e)))?;

        Ok(MultiSpecClipSetup {
            x_l,
            x_u,
            l_a,
            lbias,
            thresholds_arr,
            n_rows,
        })
    }

    /// Grouped-safe multi-objective clipping that returns per-row lower bounds.
    ///
    /// Unlike `clip_multi_objective_with_precomputed_linear`, this does NOT collapse
    /// per-row bounds into a single `verified` boolean. The caller (grouped/disjunctive
    /// process_batch) performs its own OR-within-clause and AND-across-clauses reduction.
    ///
    /// Reference: `alpha-beta-CROWN/complete_verifier/input_split/clip.py:clip_domains`
    /// Design: `designs/2026-03-22-issue-4367-grouped-joint-multispec-relaxed-clip.md`
    pub(in crate::beta_crown::engine) fn clip_multi_objective_grouped_safe(
        &self,
        input_bounds: &BoundedTensor,
        original_shape: &[usize],
        linear_bounds: &crate::bounds::LinearBounds,
        thresholds: &[f32],
    ) -> Result<MultiSpecRelaxedClipOutcome> {
        let n_rows = linear_bounds.lower_a().nrows();
        if n_rows == 0 || thresholds.is_empty() {
            return Ok(MultiSpecRelaxedClipOutcome {
                bounds: input_bounds.clone(),
                infeasible_after_clip: false,
                postclip_lower_bounds: vec![],
            });
        }

        let setup = self.prepare_multi_spec_clip(input_bounds, linear_bounds, thresholds)?;
        let is_lower = true;

        // Check pre-clip bounds but return per-row values instead of collapsing.
        let dm_lb_pre =
            Self::concretize_dm_lb(&setup.x_l, &setup.x_u, &setup.l_a, &setup.lbias, is_lower);
        let pre_lower_bounds: Vec<f32> = (0..setup.n_rows).map(|s| dm_lb_pre[[0, s]]).collect();

        // If any row is already verified pre-clip, we still run clipping to tighten
        // the box. The caller decides how to use the per-row lower bounds.

        let (new_l, new_u, verified_by_clip) = relaxed_clip_with_infeasible_mask(
            &setup.x_l.into_dyn(),
            &setup.x_u.into_dyn(),
            &setup.l_a.clone().into_dyn(),
            &setup.lbias.clone().into_dyn(),
            &setup.thresholds_arr.clone().into_dyn(),
            self.config.relaxed_clip_iterations,
            is_lower,
        )?;

        let infeasible = verified_by_clip.into_iter().any(|v| v);
        if infeasible {
            // Child box is empty — per-row lower bounds are meaningless.
            // Return the pre-clip lower bounds; caller checks infeasible_after_clip first.
            return Ok(MultiSpecRelaxedClipOutcome {
                bounds: input_bounds.clone(),
                infeasible_after_clip: true,
                postclip_lower_bounds: pre_lower_bounds,
            });
        }

        let dm_lb_post =
            Self::concretize_dm_lb_from_dyn(&new_l, &new_u, &setup.l_a, &setup.lbias, is_lower);
        let postclip_lower_bounds: Vec<f32> =
            (0..setup.n_rows).map(|s| dm_lb_post[[0, s]]).collect();

        let new_lower = new_l
            .into_shape_clone(ndarray::IxDyn(original_shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clipped lower: {}", e)))?;
        let new_upper = new_u
            .into_shape_clone(ndarray::IxDyn(original_shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clipped upper: {}", e)))?;

        Ok(MultiSpecRelaxedClipOutcome {
            bounds: BoundedTensor::new(new_lower, new_upper)?,
            infeasible_after_clip: false,
            postclip_lower_bounds,
        })
    }
}

#[cfg(test)]
impl BetaCrownVerifier {
    /// Multi-objective variant of pre-computed clipping that uses all spec rows together.
    ///
    /// Returns a conjunctive `RelaxedClipOutcome` where `verified` is true if ANY
    /// spec row is verified. Used by existing tests only — production grouped path
    /// uses `clip_multi_objective_grouped_safe` instead.
    ///
    /// Reference: `alpha-beta-CROWN/complete_verifier/input_split/clip.py:clip_domains`
    pub(in crate::beta_crown::engine) fn clip_multi_objective_with_precomputed_linear(
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

        let setup = self.prepare_multi_spec_clip(input_bounds, linear_bounds, thresholds)?;
        let is_lower = true;

        let dm_lb_pre =
            Self::concretize_dm_lb(&setup.x_l, &setup.x_u, &setup.l_a, &setup.lbias, is_lower);
        if Self::any_verified(&dm_lb_pre, &setup.thresholds_arr) {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds.clone(),
                verified: true,
            });
        }

        let (new_l, new_u, verified_by_clip) = relaxed_clip_with_infeasible_mask(
            &setup.x_l.into_dyn(),
            &setup.x_u.into_dyn(),
            &setup.l_a.clone().into_dyn(),
            &setup.lbias.clone().into_dyn(),
            &setup.thresholds_arr.clone().into_dyn(),
            self.config.relaxed_clip_iterations,
            is_lower,
        )?;
        if verified_by_clip.into_iter().any(|v| v) {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds.clone(),
                verified: true,
            });
        }

        let dm_lb_post =
            Self::concretize_dm_lb_from_dyn(&new_l, &new_u, &setup.l_a, &setup.lbias, is_lower);
        let verified = Self::any_verified(&dm_lb_post, &setup.thresholds_arr);

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
