// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Output specification constraints in matrix form for INVPROP.

use ndarray::{Array1, Array2};
use ny_core::NyError;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Output specification constraints in matrix form: `A*y <= rhs`
///
/// Represents the output constraint as a system of linear inequalities.
/// For verification, we check if the network output satisfies these constraints
/// for all inputs in the perturbation region.
///
/// # Constraint Mapping
///
/// VNN-LIB constraints map to this form as follows:
/// - `Y_i <= Y_j` -> row has `+1` at `i`, `-1` at `j`, `rhs=0`
/// - `Y_i >= Y_j` -> row has `-1` at `i`, `+1` at `j`, `rhs=0`
/// - `Y_i <= c` -> row has `+1` at `i`, `rhs=c`
/// - `Y_i >= c` -> row has `-1` at `i`, `rhs=-c`
///
/// Strict inequalities (`<`, `>`) are treated as non-strict for soundness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConstraints {
    /// Constraint matrix A with shape `[num_constraints, output_dim]`.
    /// Each row represents one linear inequality.
    pub a_matrix: Array2<f32>,

    /// Right-hand side thresholds with shape `[num_constraints]`.
    /// Constraint i is satisfied when `(A*y)[i] <= rhs[i]`.
    pub rhs: Array1<f32>,

    /// Whether constraints form a conjunction (AND) or disjunction (OR).
    ///
    /// - `true` (conjunction): ALL constraints must be satisfied
    /// - `false` (disjunction): AT LEAST ONE constraint must be satisfied
    ///
    /// Note: Initial INVPROP implementation only supports conjunction.
    /// Disjunction requires additional handling (tracked in #360).
    pub is_conjunction: bool,

    /// Optional clause grouping for disjunctive properties.
    ///
    /// When present and `is_conjunction == false`, each entry lists the
    /// constraint indices that must all be satisfied for that clause.
    /// This enables representing OR-of-ANDs without flattening to a single OR.
    #[serde(default)]
    pub clause_indices: Option<Vec<Vec<usize>>>,
}

impl OutputConstraints {
    /// Create new output constraints from matrix form.
    ///
    /// # Arguments
    /// * `a_matrix` - Constraint matrix `[num_constraints, output_dim]`
    /// * `rhs` - Right-hand side thresholds `[num_constraints]`
    /// * `is_conjunction` - Whether constraints are AND'd (true) or OR'd (false)
    ///
    /// # Errors
    /// Returns `NyError::InvalidSpec` if `a_matrix.nrows() != rhs.len()`.
    pub fn new(
        a_matrix: Array2<f32>,
        rhs: Array1<f32>,
        is_conjunction: bool,
    ) -> Result<Self, NyError> {
        if a_matrix.nrows() != rhs.len() {
            return Err(NyError::InvalidSpec(format!(
                "OutputConstraints::new row/rhs mismatch: a_matrix has {} rows, rhs has {} entries",
                a_matrix.nrows(),
                rhs.len()
            )));
        }
        Ok(Self {
            a_matrix,
            rhs,
            is_conjunction,
            clause_indices: None,
        })
    }

    fn validate_target(output_dim: usize, target: usize, context: &str) -> Result<(), NyError> {
        if target >= output_dim {
            return Err(NyError::InvalidSpec(format!(
                "OutputConstraints::{context} target index {} out of bounds for output_dim {}",
                target, output_dim
            )));
        }
        Ok(())
    }

    /// Create constraints for a simple threshold: `output[target] >= threshold`
    ///
    /// This represents verifying that a specific output neuron exceeds a threshold.
    /// Converted to `A*y <= rhs` form as: `-output[target] <= -threshold`
    pub fn ge_threshold(output_dim: usize, target: usize, threshold: f32) -> Result<Self, NyError> {
        Self::validate_target(output_dim, target, "ge_threshold")?;
        let mut a_matrix = Array2::zeros((1, output_dim));
        a_matrix[[0, target]] = -1.0;
        let rhs = Array1::from_elem(1, -threshold);
        Self::new(a_matrix, rhs, true)
    }

    /// Create constraints for: `output[target] <= threshold`
    ///
    /// Converted to `A*y <= rhs` form as: `output[target] <= threshold`
    pub fn le_threshold(output_dim: usize, target: usize, threshold: f32) -> Result<Self, NyError> {
        Self::validate_target(output_dim, target, "le_threshold")?;
        let mut a_matrix = Array2::zeros((1, output_dim));
        a_matrix[[0, target]] = 1.0;
        let rhs = Array1::from_elem(1, threshold);
        Self::new(a_matrix, rhs, true)
    }

    /// Create constraints for argmax verification: `output[target] >= output[other]` for all other.
    ///
    /// This verifies that `target` has the highest (or tied highest) output value.
    /// Converted to `A*y <= rhs` form as: `output[other] - output[target] <= 0` for all other.
    ///
    /// Note: Uses non-strict inequality (`>=`) so ties with target are considered satisfied.
    /// For strict argmax (no ties), use a small positive `rhs` via custom constraints.
    pub fn argmax(output_dim: usize, target: usize) -> Result<Self, NyError> {
        Self::validate_target(output_dim, target, "argmax")?;
        let num_constraints = output_dim - 1;
        let mut a_matrix = Array2::zeros((num_constraints, output_dim));
        let mut row = 0;
        for other in 0..output_dim {
            if other != target {
                a_matrix[[row, other]] = 1.0;
                a_matrix[[row, target]] = -1.0;
                row += 1;
            }
        }
        let rhs = Array1::zeros(num_constraints);
        Self::new(a_matrix, rhs, true)
    }

    /// Number of constraints.
    #[must_use]
    pub fn num_constraints(&self) -> usize {
        self.a_matrix.nrows()
    }

    /// Output dimension.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.a_matrix.ncols()
    }

    /// Check if a concrete output satisfies the constraints.
    ///
    /// # Arguments
    /// * `output` - Concrete output vector of shape `[output_dim]`
    ///
    /// # Returns
    /// `true` if constraints are satisfied (considering conjunction/disjunction)
    #[must_use]
    pub fn is_satisfied(&self, output: &Array1<f32>) -> bool {
        if self.num_constraints() == 0 {
            return false;
        }
        if output.len() != self.output_dim() {
            warn!(
                "OutputConstraints::is_satisfied dimension mismatch: output_dim={}, got={}",
                self.output_dim(),
                output.len()
            );
            return false;
        }
        let constraint_values = self.a_matrix.dot(output);
        if self.is_conjunction {
            // All constraints must be satisfied: A*y <= rhs
            constraint_values
                .iter()
                .zip(self.rhs.iter())
                .all(|(val, &rhs)| *val <= rhs)
        } else if let Some(clauses) = self.clause_indices.as_ref() {
            // Disjunction over clauses, each clause is a conjunction.
            let num_constraints = self.num_constraints();
            if clauses.iter().flatten().any(|&idx| idx >= num_constraints) {
                warn!(
                    "OutputConstraints::is_satisfied found clause index out of bounds: max={}, num_constraints={}",
                    clauses
                        .iter()
                        .flatten()
                        .max()
                        .cloned()
                        .unwrap_or(0),
                    num_constraints
                );
                return false;
            }
            clauses.iter().any(|clause| {
                clause.iter().all(|&idx| {
                    constraint_values
                        .get(idx)
                        .zip(self.rhs.get(idx))
                        .map(|(val, rhs)| *val <= *rhs)
                        .unwrap_or(false)
                })
            })
        } else {
            // At least one constraint must be satisfied
            constraint_values
                .iter()
                .zip(self.rhs.iter())
                .any(|(val, &rhs)| *val <= rhs)
        }
    }
}
