// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Split history conversion methods for [`ArenaConstraintStore`].
//!
//! Converts BaB split histories (ReLU, GenBaB, CROWN-based) into
//! arena-stored linear constraints.
//!
//! # Sources
//!
//! - Design doc: `designs/2026-01-29-genbab-split-to-linear.md`

use super::arena::ArenaConstraintStore;
use super::types::{ConstraintOrigin, ConstraintSense};
use crate::beta_crown::branching::{GraphSplitHistory, NeuronSplit, SplitHistory};
use crate::LinearBounds;
use ny_core::Result;

impl ArenaConstraintStore {
    /// Add a ReLU split constraint.
    ///
    /// - **Active split** (`is_active = true`): z >= 0, encoded as -z <= 0
    /// - **Inactive split** (`is_active = false`): z <= 0, encoded as z <= 0
    ///
    /// # Arguments
    /// * `var_idx` - Index of the pre-activation variable
    /// * `is_active` - True for active (z >= 0), false for inactive (z <= 0)
    pub fn add_relu_split(&mut self, var_idx: u32, is_active: bool) -> Result<()> {
        if is_active {
            // z >= 0 → -z <= 0
            // SAFETY: Single-element arrays always satisfy constraints (len == 1 <= u16::MAX)
            self.add_constraint(
                &[var_idx],
                &[-1.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )?;
        } else {
            // z <= 0
            // SAFETY: Single-element arrays always satisfy constraints
            self.add_constraint(
                &[var_idx],
                &[1.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )?;
        }
        Ok(())
    }

    /// Add a GenBaB split constraint with arbitrary split point.
    ///
    /// - **Lower branch** (`is_upper = false`): z <= split_point
    /// - **Upper branch** (`is_upper = true`): z >= split_point, encoded as -z <= -split_point
    ///
    /// # Arguments
    /// * `var_idx` - Index of the pre-activation variable
    /// * `split_point` - The branching point value
    /// * `is_upper` - True for upper branch (z >= s), false for lower (z <= s)
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if `split_point` is NaN or Inf (#2259).
    pub fn add_genbab_split(
        &mut self,
        var_idx: u32,
        split_point: f32,
        is_upper: bool,
    ) -> Result<()> {
        // NaN/Inf validation (#2259): -NaN is NaN, and -Inf produces a
        // meaningless bias. Catch at the entry point before negation.
        if !split_point.is_finite() {
            return Err(ny_core::NyError::NumericalInstability(format!(
                "GenBaB split_point is {} for variable {} — would produce \
                 meaningless constraint (#2259)",
                split_point, var_idx
            )));
        }
        if is_upper {
            // z >= split_point → -z <= -split_point
            // SAFETY: Single-element arrays always satisfy constraints
            self.add_constraint(
                &[var_idx],
                &[-1.0],
                -split_point,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )?;
        } else {
            // z <= split_point
            // SAFETY: Single-element arrays always satisfy constraints
            self.add_constraint(
                &[var_idx],
                &[1.0],
                split_point,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )?;
        }
        Ok(())
    }

    /// Convert a `SplitHistory` to constraints.
    ///
    /// Requires a mapping from (layer_idx, neuron_idx) to global variable index.
    /// Returns the number of constraints added.
    ///
    /// # Arguments
    /// * `history` - The split history to convert
    /// * `var_index_fn` - Function to map (layer_idx, neuron_idx) to global variable index
    pub fn add_from_split_history<F>(
        &mut self,
        history: &SplitHistory,
        var_index_fn: F,
    ) -> Result<usize>
    where
        F: Fn(usize, usize) -> u32,
    {
        let count = history.constraints.len();
        for constraint in &history.constraints {
            let var_idx = var_index_fn(constraint.layer_idx, constraint.neuron_idx);
            self.add_relu_split(var_idx, constraint.is_active)?;
        }
        Ok(count)
    }

    /// Convert a `GraphSplitHistory` to constraints.
    ///
    /// Requires a mapping from (node_name, neuron_idx) to global variable index.
    /// Returns the number of constraints added.
    ///
    /// # Arguments
    /// * `history` - The graph split history to convert
    /// * `var_index_fn` - Function to map (node_name, neuron_idx) to global variable index
    pub fn add_from_graph_split_history<F>(
        &mut self,
        history: &GraphSplitHistory,
        var_index_fn: F,
    ) -> Result<usize>
    where
        F: Fn(&str, usize) -> u32,
    {
        let mut count = 0;

        // Add ReLU constraints
        for constraint in &history.constraints {
            let var_idx = var_index_fn(&constraint.node_name, constraint.neuron_idx);
            self.add_relu_split(var_idx, constraint.is_active)?;
            count += 1;
        }

        // Add GenBaB constraints
        for constraint in &history.genbab_constraints {
            let var_idx = var_index_fn(&constraint.node_name, constraint.neuron_idx);
            self.add_genbab_split(var_idx, constraint.split_point, constraint.is_upper_branch)?;
            count += 1;
        }

        Ok(count)
    }

    /// Add input-space constraints from a BaB split using CROWN linear bounds.
    ///
    /// Converts pre-activation split constraints (e.g., `z >= s`) to input-space
    /// constraints using CROWN bound propagation. This enables Clip-and-Verify
    /// to constrain inputs based on BaB branching decisions.
    ///
    /// # Conversion Rules
    ///
    /// | Split Constraint | CROWN Bound Used | Input-Space Constraint |
    /// |------------------|------------------|------------------------|
    /// | z >= s | Lower bound (lA, lb) | `-lA·x + (s - lb) ≤ 0` |
    /// | z <= s | Upper bound (uA, ub) | `uA·x + (ub - s) ≤ 0` |
    ///
    /// # Arguments
    /// * `split` - The neuron split constraint
    /// * `crown_bounds` - CROWN linear bounds for the target neuron (single row extracted)
    /// * `neuron_row` - Row index in the CROWN bounds matrix for this neuron
    ///
    /// # Returns
    /// Number of constraints added (0, 1, or 2 depending on split bounds).
    ///
    /// # Errors
    /// Returns `NyError::InvalidSpec` if crown bounds have more than u16::MAX terms.
    ///
    /// # Design Reference
    /// See `designs/2026-01-29-genbab-split-to-linear.md` for derivation and soundness proof.
    pub fn add_split_with_crown_bounds(
        &mut self,
        split: &NeuronSplit,
        crown_bounds: &LinearBounds,
        neuron_row: usize,
    ) -> Result<usize> {
        const EPSILON: f32 = 1e-10;
        let mut count = 0;

        // Guard: CROWN coefficient row length must fit in u32 for arena index storage.
        // Saturating `i as u32` on rows wider than u32::MAX would silently produce
        // wrong constraint indices. (#2911)
        let ncols = crown_bounds.lower_a().ncols();
        if ncols > u32::MAX as usize {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "CROWN bounds have {} columns, exceeds u32::MAX for constraint indices",
                ncols,
            )));
        }

        // Lower bound constraint: z >= s → -lA·x + (s - lb) ≤ 0
        if let Some(s) = split.lower_bound {
            let la_row = crown_bounds.lower_a().row(neuron_row);
            let lb = crown_bounds.lower_b()[neuron_row];

            // NaN/Inf validation (#2259): CROWN bounds can contain NaN from
            // upstream propagation bugs (e.g., Exp/Log near boundaries). Catch
            // before computing bias to give a diagnostic error.
            if !lb.is_finite() {
                return Err(ny_core::NyError::NumericalInstability(format!(
                    "CROWN lower_b[{}] is {} — cannot form valid constraint (#2259)",
                    neuron_row, lb
                )));
            }
            if !s.is_finite() {
                return Err(ny_core::NyError::NumericalInstability(format!(
                    "split lower_bound is {} — cannot form valid constraint (#2259)",
                    s
                )));
            }

            let bias = s - lb;

            // Convert to sparse representation: negate coefficients, filter zeros
            let (indices, coeffs): (Vec<u32>, Vec<f32>) = la_row
                .iter()
                .enumerate()
                .filter(|(_, &v)| v.abs() > EPSILON)
                .map(|(i, &v)| (i as u32, -v)) // Negate: -lA
                .unzip();

            if !indices.is_empty() {
                self.add_constraint(
                    &indices,
                    &coeffs,
                    bias,
                    ConstraintSense::Le,
                    ConstraintOrigin::Split,
                )?;
                count += 1;
            } else if bias > 0.0 {
                // All coefficients are near-zero, so the constraint reduces to
                // `(s - lb) ≤ 0`. If bias > 0 this is infeasible: the domain is
                // provably empty (#2260). Return a special error so the caller
                // can prune this domain from the BaB tree.
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "CROWN lower bound constraint infeasible: all coefficients \
                     below EPSILON but bias={:.6} > 0 (s={:.6}, lb={:.6})",
                    bias, s, lb
                )));
            }
            // else: bias <= 0, constraint is trivially satisfied, safe to drop.
        }

        // Upper bound constraint: z <= s → uA·x + (ub - s) ≤ 0
        if let Some(s) = split.upper_bound {
            let ua_row = crown_bounds.upper_a().row(neuron_row);
            let ub = crown_bounds.upper_b()[neuron_row];

            // NaN/Inf validation (#2259): same as lower bound path above.
            if !ub.is_finite() {
                return Err(ny_core::NyError::NumericalInstability(format!(
                    "CROWN upper_b[{}] is {} — cannot form valid constraint (#2259)",
                    neuron_row, ub
                )));
            }
            if !s.is_finite() {
                return Err(ny_core::NyError::NumericalInstability(format!(
                    "split upper_bound is {} — cannot form valid constraint (#2259)",
                    s
                )));
            }

            let bias = ub - s;

            // Convert to sparse representation: use coefficients directly, filter zeros
            let (indices, coeffs): (Vec<u32>, Vec<f32>) = ua_row
                .iter()
                .enumerate()
                .filter(|(_, &v)| v.abs() > EPSILON)
                .map(|(i, &v)| (i as u32, v)) // Use directly: uA
                .unzip();

            if !indices.is_empty() {
                self.add_constraint(
                    &indices,
                    &coeffs,
                    bias,
                    ConstraintSense::Le,
                    ConstraintOrigin::Split,
                )?;
                count += 1;
            } else if bias > 0.0 {
                // Same infeasibility check for upper bound.
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "CROWN upper bound constraint infeasible: all coefficients \
                     below EPSILON but bias={:.6} > 0 (ub={:.6}, s={:.6})",
                    bias, ub, s
                )));
            }
            // else: bias <= 0, constraint is trivially satisfied, safe to drop.
        }

        Ok(count)
    }
}
