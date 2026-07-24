// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Complete clipping (Clip-and-Verify) engine integration.
//!
//! Uses the Lagrangian dual coordinate ascent solver from `complete_clip` to
//! achieve LP-optimal output bound tightening. Cross-specification constraints
//! enable tighter bounds than relaxed clipping's per-dimension closed form.
//!
//! Reference: alpha-beta-CROWN `clip_type: complete`
//! (`auto_LiRPA/concretize_func.py:constraints_solving`)

use ndarray::{Array1, Array2, Array3};
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::trace;

use crate::clip_interm_domain::{add_f32_down, sub_f32_down};
use crate::complete_clip::complete_clip;
use crate::complete_clip::filter::sort_out_constraints;
use crate::relaxed_clip::relaxed_clip;
use crate::{GraphNetwork, Network};

use super::BetaCrownVerifier;
use super::RelaxedClipOutcome;

/// Construct constraints in standard form from CROWN linear bounds.
///
/// Given CROWN bounds `lA @ x + lbias` and verification thresholds, produces
/// constraints `constr_A @ x + constr_b <= 0` expressing that any feasible
/// input must violate the specification (otherwise the domain is verified).
///
/// # Arguments
///
/// * `l_a` - CROWN coefficients, shape: `(batch, n_spec, x_dim)`
/// * `lbias` - CROWN bias, shape: `(batch, n_spec)`
/// * `thresholds` - Verification thresholds, shape: `(batch, n_spec)`
///
/// # Returns
///
/// `(constr_a, constr_b)` in standard form: `constr_a @ x + constr_b <= 0`
///
/// Reference: `auto_LiRPA/concretize_func.py:construct_constraints` (line 50)
pub(in crate::beta_crown::engine) fn construct_constraints(
    l_a: &Array3<f32>,
    lbias: &Array2<f32>,
    thresholds: &Array2<f32>,
) -> Result<(Array3<f32>, Array2<f32>)> {
    if lbias.shape() != thresholds.shape() {
        return Err(ny_core::NyError::shape_mismatch(
            lbias.shape().to_vec(),
            thresholds.shape().to_vec(),
        ));
    }

    // Standard form: lA @ x + (lbias - threshold) <= 0
    // This encodes: for the spec to be violated, lA @ x + lbias <= threshold.
    // Negating: -(lA @ x + lbias) >= -threshold => lA @ x + lbias - threshold <= 0
    let constr_a = l_a.clone();
    let mut constr_b = Array2::zeros(lbias.raw_dim());
    for ((result, &bias), &threshold) in constr_b.iter_mut().zip(lbias).zip(thresholds) {
        *result = sub_f32_down(bias, threshold).ok_or_else(|| {
            ny_core::NyError::InvalidSpec(format!(
                "complete-clip constraints require finite bias and threshold, got {bias} and {threshold}"
            ))
        })?;
    }
    Ok((constr_a, constr_b))
}

/// Add a certified objective lower bound to its CROWN lower bias without
/// rounding the resulting lower endpoint upward.
pub(in crate::beta_crown::engine) fn add_certified_lower_bias(
    constrained_lb: &Array2<f32>,
    lbias: &Array2<f32>,
) -> Result<Array2<f32>> {
    if constrained_lb.shape() != lbias.shape() {
        return Err(ny_core::NyError::shape_mismatch(
            constrained_lb.shape().to_vec(),
            lbias.shape().to_vec(),
        ));
    }

    let mut output_lb = Array2::zeros(constrained_lb.raw_dim());
    for ((result, &lower), &bias) in output_lb.iter_mut().zip(constrained_lb).zip(lbias) {
        *result = add_f32_down(lower, bias).ok_or_else(|| {
            ny_core::NyError::InvalidSpec(format!(
                "complete-clip bias merge requires finite operands, got {lower} and {bias}"
            ))
        })?;
    }
    Ok(output_lb)
}

impl BetaCrownVerifier {
    /// Apply complete clipping to tighten output bounds via constrained concretization.
    ///
    /// This first applies relaxed clipping (input box tightening), then uses the
    /// Lagrangian dual solver to compute LP-optimal output bounds subject to
    /// cross-specification constraints.
    ///
    /// # Algorithm
    ///
    /// 1. Get CROWN linear bounds at input layer
    /// 2. Apply relaxed clipping to tighten input box
    /// 3. Construct constraints from spec CROWN bounds
    /// 4. Use `complete_clip()` with each spec as objective and ALL specs as constraints
    /// 5. If any constrained output bound exceeds threshold → verified
    ///
    /// Reference: alpha-beta-CROWN `clip_type: complete`
    pub(in crate::beta_crown::engine) fn apply_complete_clipping(
        &self,
        network: &Network,
        input_bounds: BoundedTensor,
        original_shape: &[usize],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<RelaxedClipOutcome> {
        // Get CROWN linear bounds at input layer (#3598: thread engine for GPU acceleration)
        let (_output_bounds, mut linear_bounds) =
            network.propagate_crown_with_linear_and_engine(&input_bounds, engine)?;
        Self::discharge_coeff_err_for_clip(&mut linear_bounds, &input_bounds);

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

        // Check if already verified before clipping
        let dm_lb_pre = Self::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, is_lower);
        if Self::any_verified(&dm_lb_pre, &thresholds) {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds,
                verified: true,
            });
        }

        // Step 1: Apply relaxed clipping to tighten input box
        let (new_l, new_u) = relaxed_clip(
            &x_l.into_dyn(),
            &x_u.into_dyn(),
            &l_a.clone().into_dyn(),
            &lbias.clone().into_dyn(),
            &thresholds.clone().into_dyn(),
            self.config.relaxed_clip_iterations,
            is_lower,
        )?;

        // Check if relaxed clipping alone verified
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

        // Step 2: Build constraints from spec CROWN bounds and use complete_clip
        // for LP-optimal output bound concretization.
        // Only beneficial with multiple specs (cross-constraint tightening).
        let verified = if n_spec > 1 {
            // Reshape tightened bounds for complete_clip: (1, x_dim)
            let clip_x_l = new_l
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clip_x_l: {}", e)))?;
            let clip_x_u = new_u
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clip_x_u: {}", e)))?;

            let (constr_a, constr_b) = construct_constraints(&l_a, &lbias, &thresholds)?;

            // Pre-filter constraints: detect infeasible or fully-covered cases
            // before running the expensive LP solver.
            // Reference: auto_LiRPA/concretize_func.py:_sort_out_constraints
            let filter_result = sort_out_constraints(&constr_a, &constr_b, &clip_x_l, &clip_x_u);

            if filter_result.any_infeasible() {
                // Infeasible constraints → domain is empty → verified
                trace!("complete_clip: infeasible constraints detected, domain verified");
                true
            } else if filter_result.all_fully_covered() {
                // All constraints always satisfied within the box → LP gives
                // no benefit over unconstrained Holder.
                trace!(
                    "complete_clip: all {} constraints fully covered, skipping LP",
                    n_spec
                );
                false
            } else {
                // Active constraints exist — run the LP solver.
                trace!(
                    "complete_clip: {}/{} active constraints, running LP",
                    filter_result.active_constraint_indices.len(),
                    n_spec
                );
                let objective = l_a;
                let sign = -1.0_f32; // lower bound (minimize)

                match complete_clip(
                    &clip_x_l.into_dyn(),
                    &clip_x_u.into_dyn(),
                    &objective.into_dyn(),
                    &constr_a.into_dyn(),
                    &constr_b.into_dyn(),
                    sign,
                    true, // rearrange constraints
                    1,    // single iteration for input-split (called per child)
                ) {
                    Ok(constrained_lb) => {
                        // constrained_lb shape: (1, n_spec) — LP-optimal lower bounds
                        // Add bias to get final output lower bounds
                        match constrained_lb.into_dimensionality::<ndarray::Ix2>() {
                            Ok(constrained_lb_2d) => {
                                let output_lb =
                                    add_certified_lower_bias(&constrained_lb_2d, &lbias)?;
                                Self::any_verified(&output_lb, &thresholds)
                            }
                            Err(_) => {
                                // Shape error is an internal bug, not infeasibility.
                                // Conservative: treat as not verified (sound).
                                false
                            }
                        }
                    }
                    Err(ny_core::NyError::InfeasibleDomain(_)) => {
                        // Infeasible constraints → domain already ruled out → verified
                        true
                    }
                    Err(_) => {
                        // Other errors (shape, config) are not infeasibility.
                        // Conservative: treat as not verified (sound).
                        false
                    }
                }
            }
        } else {
            false
        };

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

    /// GraphNetwork variant of complete clipping using linearized spec bounds.
    pub(in crate::beta_crown::engine) fn apply_complete_clipping_graph(
        &self,
        graph: &GraphNetwork,
        input_bounds: &BoundedTensor,
        original_shape: &[usize],
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<RelaxedClipOutcome> {
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

        // Check pre-verification
        let dm_lb_pre = Self::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, is_lower);
        if Self::any_verified(&dm_lb_pre, &thresholds) {
            return Ok(RelaxedClipOutcome {
                bounds: input_bounds.clone(),
                verified: true,
            });
        }

        // Step 1: Relaxed clipping
        let (new_l, new_u) = relaxed_clip(
            &x_l.into_dyn(),
            &x_u.into_dyn(),
            &l_a.clone().into_dyn(),
            &lbias.clone().into_dyn(),
            &thresholds.clone().into_dyn(),
            self.config.relaxed_clip_iterations,
            is_lower,
        )?;

        // Check if relaxed alone verified
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

        // Step 2: Complete clipping for multi-spec cross-constraint tightening
        let verified = if n_spec > 1 {
            let clip_x_l = new_l
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clip_x_l: {}", e)))?;
            let clip_x_u = new_u
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape clip_x_u: {}", e)))?;

            let (constr_a, constr_b) = construct_constraints(&l_a, &lbias, &thresholds)?;

            // Pre-filter constraints: detect infeasible or fully-covered cases
            // before running the expensive LP solver.
            // Reference: auto_LiRPA/concretize_func.py:_sort_out_constraints
            let filter_result = sort_out_constraints(&constr_a, &constr_b, &clip_x_l, &clip_x_u);

            if filter_result.any_infeasible() {
                trace!("complete_clip_graph: infeasible constraints, domain verified");
                true
            } else if filter_result.all_fully_covered() {
                trace!(
                    "complete_clip_graph: all {} constraints fully covered, skipping LP",
                    n_spec
                );
                false
            } else {
                trace!(
                    "complete_clip_graph: {}/{} active constraints, running LP",
                    filter_result.active_constraint_indices.len(),
                    n_spec
                );
                match complete_clip(
                    &clip_x_l.into_dyn(),
                    &clip_x_u.into_dyn(),
                    &l_a.into_dyn(),
                    &constr_a.into_dyn(),
                    &constr_b.into_dyn(),
                    -1.0,
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
                            Err(_) => {
                                // Shape error is an internal bug, not infeasibility.
                                // Conservative: treat as not verified (sound).
                                false
                            }
                        }
                    }
                    Err(ny_core::NyError::InfeasibleDomain(_)) => {
                        // Infeasible constraints → domain already ruled out → verified
                        true
                    }
                    Err(_) => {
                        // Other errors (shape, config) are not infeasibility.
                        // Conservative: treat as not verified (sound).
                        false
                    }
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
