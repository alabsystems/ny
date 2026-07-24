// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constraint pre-filtering for Complete Clipping.
//!
//! Classifies constraints into infeasible, fully-covered, or active before
//! running the LP solver. This avoids unnecessary coordinate ascent work
//! on redundant constraints and enables early detection of infeasible domains.
//!
//! Reference: `auto_LiRPA/concretize_func.py:_sort_out_constraints` (line 79)

use ndarray::{Array2, Array3};

/// Result of constraint pre-filtering.
///
/// Per-batch classification of constraints relative to the input box.
#[derive(Debug, Clone)]
pub struct ConstraintFilterResult {
    /// Per-batch infeasibility flag.
    ///
    /// `infeasible[b] == true` means at least one constraint has `min(A·x + b) > 0`
    /// over the box, so no input in the box satisfies ALL constraints.
    /// The domain is empty → can be marked as verified.
    pub infeasible: Vec<bool>,
    /// Per-batch fully-covered flag.
    ///
    /// `fully_covered[b] == true` means `max(A·x + b) ≤ 0` for ALL constraints,
    /// so the constraints are always satisfied. LP would not tighten bounds.
    pub fully_covered: Vec<bool>,
    /// Indices of constraints that are active (neither infeasible nor fully covered)
    /// for at least one batch element.
    ///
    /// When this is empty, either all batches are infeasible or fully covered.
    pub active_constraint_indices: Vec<usize>,
}

impl ConstraintFilterResult {
    /// Returns true if any batch element has an infeasible constraint set.
    pub fn any_infeasible(&self) -> bool {
        self.infeasible.iter().any(|&v| v)
    }

    /// Returns true if ALL batch elements are fully covered by constraints.
    pub fn all_fully_covered(&self) -> bool {
        self.fully_covered.iter().all(|&v| v)
    }
}

/// Pre-filter constraints by feasibility and coverage relative to the input box.
///
/// For each constraint `A_k·x + b_k ≤ 0`:
/// - Compute `min_val = A_k·x_opt + b_k` where `x_opt` minimizes `A_k·x` over `[x_l, x_u]`
/// - Compute `max_val = A_k·x_opt + b_k` where `x_opt` maximizes `A_k·x` over `[x_l, x_u]`
///
/// Classification:
/// - `min_val > 0` for ANY constraint → infeasible (no x in box satisfies all constraints)
/// - `max_val ≤ 0` for ALL constraints → fully covered (constraints always satisfied)
/// - Otherwise → active (constraint may or may not be binding)
///
/// # Arguments
///
/// * `a_matrix` - Constraint coefficients, shape: `(batch, n_constraints, x_dim)`
/// * `b_vector` - Constraint offsets, shape: `(batch, n_constraints)`
/// * `x_l` - Lower bounds, shape: `(batch, x_dim)`
/// * `x_u` - Upper bounds, shape: `(batch, x_dim)`
///
/// # Reference
///
/// `auto_LiRPA/concretize_func.py:_sort_out_constraints` (line 79)
pub(crate) fn sort_out_constraints(
    a_matrix: &Array3<f32>,
    b_vector: &Array2<f32>,
    x_l: &Array2<f32>,
    x_u: &Array2<f32>,
) -> ConstraintFilterResult {
    let batch = a_matrix.shape()[0];
    let n_constraints = a_matrix.shape()[1];
    let x_dim = a_matrix.shape()[2];

    let mut infeasible = vec![false; batch];
    let mut fully_covered = vec![true; batch];
    // Track which constraints are active in at least one batch
    let mut constraint_active = vec![false; n_constraints];

    for b_idx in 0..batch {
        for k in 0..n_constraints {
            // f64 corner evaluation: f32→f64 widening is exact and the product
            // of two widened f32 values fits in an f64 significand, so the only
            // round-off is the (x_dim+1)-term summation, certified below.
            let b_val = b_vector[[b_idx, k]] as f64;
            let mut min_val = b_val;
            let mut max_val = b_val;
            let mut abs_sum = b_val.abs();

            for x in 0..x_dim {
                let a_val = a_matrix[[b_idx, k, x]] as f64;
                let (t_min, t_max) = if a_val >= 0.0 {
                    (
                        a_val * x_l[[b_idx, x]] as f64,
                        a_val * x_u[[b_idx, x]] as f64,
                    )
                } else {
                    (
                        a_val * x_u[[b_idx, x]] as f64,
                        a_val * x_l[[b_idx, x]] as f64,
                    )
                };
                min_val += t_min;
                max_val += t_max;
                abs_sum += t_min.abs().max(t_max.abs());
            }

            // Certified summation round-off (Higham gamma_n with headroom):
            // the verdict-bearing tests below must hold for the EXACT corner
            // values, so the round-off has to scale with the accumulated
            // magnitude, not sit at a fixed absolute constant.
            let err = (x_dim as f64 + 1.0) * f64::EPSILON * abs_sum;

            // Infeasible: min(A·x + b) > 0 → no x in box can satisfy this
            // constraint. `min_val - err` is a certified lower bound on the
            // exact minimum; NaN/∞-poisoned rows compare false (feasible).
            if min_val - err > 0.0 {
                infeasible[b_idx] = true;
                // Once infeasible, no need to check other constraints for this batch
                break;
            }

            // Not fully covered if max(A·x + b) can exceed 0 for any
            // constraint; `max_val + err` is a certified upper bound on the
            // exact maximum, erring toward keeping the constraint active.
            if max_val + err > 0.0 {
                fully_covered[b_idx] = false;
                constraint_active[k] = true;
            }
        }
    }

    let active_constraint_indices: Vec<usize> = constraint_active
        .iter()
        .enumerate()
        .filter(|(_, &active)| active)
        .map(|(idx, _)| idx)
        .collect();

    ConstraintFilterResult {
        infeasible,
        fully_covered,
        active_constraint_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2, Array3};

    #[test]
    fn test_fully_covered_constraints() {
        // Constraint: x1 + x2 ≤ 3.0, with box [0, 1] × [0, 1]
        // max(x1 + x2) = 2.0 ≤ 3.0 → fully covered
        let a = Array3::from_shape_vec((1, 1, 2), vec![1.0, 1.0]).unwrap();
        let b = array![[-3.0]]; // A·x + b ≤ 0 → x1 + x2 - 3.0 ≤ 0
        let x_l = array![[0.0, 0.0]];
        let x_u = array![[1.0, 1.0]];

        let result = sort_out_constraints(&a, &b, &x_l, &x_u);
        assert!(!result.any_infeasible());
        assert!(result.all_fully_covered());
        assert!(result.active_constraint_indices.is_empty());
    }

    #[test]
    fn test_infeasible_constraint() {
        // Constraint: x1 + x2 ≤ -1.0, with box [0, 1] × [0, 1]
        // min(x1 + x2) = 0.0 > -1.0... wait, let me think.
        // Standard form: A·x + b ≤ 0
        // A = [1, 1], b = 1.0 → x1 + x2 + 1.0 ≤ 0 → x1 + x2 ≤ -1.0
        // min(x1 + x2 + 1.0) = 0 + 0 + 1.0 = 1.0 > 0 → infeasible
        let a = Array3::from_shape_vec((1, 1, 2), vec![1.0, 1.0]).unwrap();
        let b = array![[1.0]];
        let x_l = array![[0.0, 0.0]];
        let x_u = array![[1.0, 1.0]];

        let result = sort_out_constraints(&a, &b, &x_l, &x_u);
        assert!(result.any_infeasible());
        assert!(result.infeasible[0]);
    }

    #[test]
    fn test_active_constraint() {
        // Constraint: x1 + x2 ≤ 1.5 → A·x + b ≤ 0 with A=[1,1], b=-1.5
        // Box: [0, 1] × [0, 1]
        // min(x1 + x2 - 1.5) = -1.5 ≤ 0 (feasible)
        // max(x1 + x2 - 1.5) = 0.5 > 0 (not fully covered)
        // → active constraint
        let a = Array3::from_shape_vec((1, 1, 2), vec![1.0, 1.0]).unwrap();
        let b = array![[-1.5]];
        let x_l = array![[0.0, 0.0]];
        let x_u = array![[1.0, 1.0]];

        let result = sort_out_constraints(&a, &b, &x_l, &x_u);
        assert!(!result.any_infeasible());
        assert!(!result.all_fully_covered());
        assert_eq!(result.active_constraint_indices, vec![0]);
    }

    #[test]
    fn test_mixed_constraints() {
        // Two constraints on box [0, 1] × [0, 1]:
        // k=0: x1 + x2 ≤ 3.0 → fully covered (max = 2.0 ≤ 3.0)
        // k=1: x1 + x2 ≤ 1.5 → active (max = 2.0 > 1.5, min = 0.0 ≤ 1.5)
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let b = array![[-3.0, -1.5]];
        let x_l = array![[0.0, 0.0]];
        let x_u = array![[1.0, 1.0]];

        let result = sort_out_constraints(&a, &b, &x_l, &x_u);
        assert!(!result.any_infeasible());
        assert!(!result.all_fully_covered());
        // Only constraint 1 is active
        assert_eq!(result.active_constraint_indices, vec![1]);
    }

    #[test]
    fn test_boundary_feasible_large_magnitude_not_infeasible() {
        // Corner terms of magnitude 2^24 that cancel exactly to a box-minimum
        // of 0 (feasible: the corner satisfies A·x + b <= 0). f32
        // round-to-nearest accumulation drifts to +0.75 on these terms — far
        // above any fixed absolute tolerance — so the infeasibility verdict
        // must come from a certified lower bound on the exact minimum.
        let a =
            Array3::from_shape_vec((1, 1, 4), vec![16777216.0, 1.5, -16777216.0, -1.25]).unwrap();
        let b = array![[-0.25]];
        let x_l = array![[1.0, 1.0, 1.0, 1.0]];
        let x_u = array![[1.0, 1.0, 1.0, 1.0]];

        let result = sort_out_constraints(&a, &b, &x_l, &x_u);
        assert!(
            !result.any_infeasible(),
            "exact box-minimum is 0: constraint is feasible"
        );
    }

    #[test]
    fn test_multi_batch() {
        // Batch of 2 with different box sizes:
        // Batch 0: box [0, 0.5] × [0, 0.5], constraint x1+x2 ≤ 1.5
        //   max(x1+x2-1.5) = -0.5 ≤ 0 → fully covered
        // Batch 1: box [0, 1.0] × [0, 1.0], constraint x1+x2 ≤ 1.5
        //   max(x1+x2-1.5) = 0.5 > 0 → active
        let a = Array3::from_shape_vec((2, 1, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let b = Array2::from_shape_vec((2, 1), vec![-1.5, -1.5]).unwrap();
        let x_l = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let x_u = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 1.0, 1.0]).unwrap();

        let result = sort_out_constraints(&a, &b, &x_l, &x_u);
        assert!(!result.any_infeasible());
        assert!(!result.all_fully_covered());
        assert!(result.fully_covered[0]); // Batch 0 is fully covered
        assert!(!result.fully_covered[1]); // Batch 1 is not
        assert_eq!(result.active_constraint_indices, vec![0]);
    }
}
