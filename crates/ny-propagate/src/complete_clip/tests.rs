// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for Complete Clipping algorithm.

use super::dual::{solve_dual_variable, solve_dual_variable_with_deadline_check};
use super::objective::{
    compute_eps_term_with_deadline_check, update_objective_with_deadline_check,
};
use super::rearrange::{
    compute_constraint_slack_with_deadline_check, rearrange_constraints,
    rearrange_constraints_with_deadline_check,
};
use super::validate::ensure_constraints_feasible_with_deadline_check;
use super::*;
use ndarray::array;
use ndarray::{Array1, Ix2, Ix3};

fn enumerate_feasible_vertices_2d(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
) -> Vec<[f32; 2]> {
    let x_l = x_l
        .view()
        .into_dimensionality::<Ix2>()
        .expect("x_l must be 2D");
    let x_u = x_u
        .view()
        .into_dimensionality::<Ix2>()
        .expect("x_u must be 2D");
    let a_matrix = a_matrix
        .view()
        .into_dimensionality::<Ix3>()
        .expect("a_matrix must be 3D");
    let b_vector = b_vector
        .view()
        .into_dimensionality::<Ix2>()
        .expect("b_vector must be 2D");

    let x1_l = x_l[[0, 0]];
    let x2_l = x_l[[0, 1]];
    let x1_u = x_u[[0, 0]];
    let x2_u = x_u[[0, 1]];

    let n_constraints = a_matrix.shape()[1];
    let mut constraints = Vec::with_capacity(n_constraints + 4);
    for k in 0..n_constraints {
        constraints.push([a_matrix[[0, k, 0]], a_matrix[[0, k, 1]], b_vector[[0, k]]]);
    }

    // Box constraints: x1 <= x1_u, x1 >= x1_l, x2 <= x2_u, x2 >= x2_l
    constraints.push([1.0, 0.0, -x1_u]);
    constraints.push([-1.0, 0.0, x1_l]);
    constraints.push([0.0, 1.0, -x2_u]);
    constraints.push([0.0, -1.0, x2_l]);

    let mut candidates = Vec::new();
    let eps = 1e-6_f32;

    // Include box corners explicitly for stability.
    candidates.push([x1_l, x2_l]);
    candidates.push([x1_l, x2_u]);
    candidates.push([x1_u, x2_l]);
    candidates.push([x1_u, x2_u]);

    for i in 0..constraints.len() {
        for j in (i + 1)..constraints.len() {
            let [a1, a2, b1] = constraints[i];
            let [c1, c2, b2] = constraints[j];
            let det = a1 * c2 - a2 * c1;
            if det.abs() <= eps {
                continue;
            }
            let x1 = (-b1 * c2 + b2 * a2) / det;
            let x2 = (-a1 * b2 + c1 * b1) / det;
            candidates.push([x1, x2]);
        }
    }

    let mut feasible = Vec::new();
    'candidate: for [x1, x2] in candidates {
        for [a1, a2, b] in &constraints {
            if a1 * x1 + a2 * x2 + b > eps {
                continue 'candidate;
            }
        }
        feasible.push([x1, x2]);
    }

    feasible
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clip_simple() {
    // Simple test: 1 batch, 1 output dim, 2D input
    // Minimize x1 subject to x1 + x2 <= 0.5, x in [0, 1]^2
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn(); // min x1
    let a_matrix = array![[[1.0, 1.0]]].into_dyn(); // x1 + x2
    let b_vector = array![[-0.5]].into_dyn(); // <= 0.5

    let result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect("complete_clip failed");

    // Without constraint: min x1 in [0,1] = 0
    // With constraint x1 + x2 <= 0.5: still min is 0 (at x1=0, x2<=0.5)
    assert_eq!(result.shape(), &[1, 1]);
    let min_val = result[[0, 0]];
    assert!(
        (-1e-6..=0.1).contains(&min_val),
        "Expected min near 0, got {}",
        min_val
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clip_maximize() {
    // Maximize x1 subject to x1 + x2 <= 1.5, x in [0, 1]^2
    // NOTE: Complete clipping returns dual BOUNDS, not optimal values.
    // The dual bound may be loose when constraints don't tighten the box.
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn(); // max x1
    let a_matrix = array![[[1.0, 1.0]]].into_dyn(); // x1 + x2
    let b_vector = array![[-1.5]].into_dyn(); // <= 1.5

    let result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, 1.0, false, 1)
        .expect("complete_clip failed");

    // Dual bound computation:
    // - True optimal max x1 = 1.0 (at x1=1, x2=0)
    // - Dual bound is valid but may be loose: 1.5 (upper bound)
    // The algorithm correctly computes a valid upper bound
    assert_eq!(result.shape(), &[1, 1]);
    let max_val = result[[0, 0]];
    // Dual bound for max x1 with x1+x2<=1.5 on [0,1]^2 is at most 1.5
    assert!(
        (1.0..=1.6).contains(&max_val),
        "Expected dual upper bound in [1.0, 1.6], got {}",
        max_val
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clip_binding_constraint() {
    // Minimize x1 subject to x1 >= 0.3, x in [0, 1]
    // Constraint: -x1 + 0.3 <= 0 (i.e., x1 >= 0.3)
    let x_l = array![[0.0]].into_dyn();
    let x_u = array![[1.0]].into_dyn();
    let objective = array![[[1.0]]].into_dyn(); // min x1
    let a_matrix = array![[[-1.0]]].into_dyn(); // -x1
    let b_vector = array![[0.3]].into_dyn(); // + 0.3 <= 0

    let result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect("complete_clip failed");

    // min x1 with x1 >= 0.3: LP should give exactly 0.3 (1e-3 for solver precision)
    let min_val = result[[0, 0]];
    assert!(
        (min_val - 0.3).abs() < 1e-3,
        "Expected min near 0.3, got {}",
        min_val
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_objective_broadcast_2d_matches_3d() {
    let x_l = array![[0.0, 0.0], [0.5, 0.5]].into_dyn();
    let x_u = array![[1.0, 1.0], [1.0, 1.0]].into_dyn();
    let objective_2d = array![[1.0, 1.0]].into_dyn();
    let objective_3d = array![[[1.0, 1.0]], [[1.0, 1.0]]].into_dyn();
    let a_matrix = array![[[0.0, 0.0]], [[0.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0], [0.0]].into_dyn();

    let result_2d = complete_clip(
        &x_l,
        &x_u,
        &objective_2d,
        &a_matrix,
        &b_vector,
        -1.0,
        false,
        1,
    )
    .expect("complete_clip failed for 2D objective");
    let result_3d = complete_clip(
        &x_l,
        &x_u,
        &objective_3d,
        &a_matrix,
        &b_vector,
        -1.0,
        false,
        1,
    )
    .expect("complete_clip failed for 3D objective");

    assert_eq!(result_2d.shape(), result_3d.shape());
    for b in 0..result_2d.shape()[0] {
        for h in 0..result_2d.shape()[1] {
            let diff = (result_2d[[b, h]] - result_3d[[b, h]]).abs();
            assert!(
                diff < 1e-6,
                "Expected broadcasted objective to match: batch={}, h_dim={}, diff={}",
                b,
                h,
                diff
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_rearrange_constraints() {
    // Two constraints: one closer to centroid, one further
    let a_matrix = array![[[1.0, 0.0], [0.0, 1.0]]].into_dyn(); // x1 and x2
    let b_vector = array![[-0.1, -0.9]].into_dyn(); // different offsets
    let x0 = array![[0.5, 0.5]].into_dyn(); // centroid at (0.5, 0.5)

    let (a_rearranged, b_rearranged) =
        rearrange_constraints(&a_matrix, &b_vector, &x0).expect("rearrange failed");

    // Constraint 1: x1 <= 0.1, distance at centroid = 0.5 - 0.1 = 0.4
    // Constraint 2: x2 <= 0.9, distance at centroid = 0.5 - 0.9 = -0.4
    // Sorted descending: constraint 1 first (0.4), then constraint 2 (-0.4)
    assert_eq!(a_rearranged.shape(), &[1, 2, 2]);
    assert_eq!(b_rearranged, b_vector);
}

#[ntest::timeout(10000)]
#[test]
fn test_solve_dual_variable_gradient_negative() {
    // Case where gradient is negative at β=0, so optimal β=0
    let constr_a = array![[0.1, 0.1]]; // small constraint coefficients, shape (1, 2)
    let objective_a = array![[[1.0, 1.0]]]; // large objective, shape (1, 1, 2)
    let constr_d = array![10.0]; // large slack, shape (1,) - 1D array
    let epsilon = array![[0.5, 0.5]]; // shape (1, 2)

    let beta = solve_dual_variable(&constr_a, &objective_a, &constr_d, &epsilon, None)
        .expect("solve_dual_variable failed");

    // With large positive d (constraint far from binding), gradient at β=0 is negative
    // So optimal β should be 0
    assert_eq!(beta.shape(), &[1, 1]);
    assert!(
        beta[[0, 0]] < 0.1,
        "Expected β near 0, got {}",
        beta[[0, 0]]
    );
}

// =========================================================================
// CPU tests for monotonicity and feasibility (#348)
// =========================================================================

/// Test monotonicity: tighter constraints should yield tighter (or equal) bounds.
///
/// For a minimization problem, as constraints become more restrictive,
/// the computed lower bound should monotonically increase (cannot decrease).
#[ntest::timeout(10000)]
#[test]
fn test_monotonicity_tighter_constraints_tighter_bounds() {
    // Minimize x1 + x2 over [0,1]^2 with constraint: x1 + x2 <= threshold
    // As threshold decreases, the feasible region shrinks, so min value increases.
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 1.0]]].into_dyn(); // min (x1 + x2)
    let a_matrix = array![[[1.0, 1.0]]].into_dyn(); // x1 + x2

    // Test with progressively tighter thresholds
    let thresholds = [2.0, 1.5, 1.0, 0.8, 0.5, 0.3];
    let mut prev_bound = f32::NEG_INFINITY;

    for threshold in thresholds {
        let b_vector = array![[-threshold]].into_dyn(); // x1 + x2 <= threshold

        let result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
            .expect("complete_clip failed");

        let bound = result[[0, 0]];

        // Monotonicity: bound should not decrease as threshold gets tighter
        assert!(
            bound >= prev_bound - 1e-5,
            "Monotonicity violated: threshold={}, bound={}, prev_bound={}",
            threshold,
            bound,
            prev_bound
        );

        prev_bound = bound;
    }
}

/// Test monotonicity for maximization: tighter constraints yield lower upper bounds.
#[ntest::timeout(10000)]
#[test]
fn test_monotonicity_maximize() {
    // Maximize x1 + x2 over [0,1]^2 with constraint: x1 + x2 <= threshold
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 1.0]]].into_dyn(); // max (x1 + x2)
    let a_matrix = array![[[1.0, 1.0]]].into_dyn();

    // As threshold decreases, max possible value decreases
    let thresholds = [2.0, 1.5, 1.0, 0.8, 0.5];
    let mut prev_bound = f32::INFINITY;

    for threshold in thresholds {
        let b_vector = array![[-threshold]].into_dyn();

        let result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, 1.0, false, 1)
            .expect("complete_clip failed");

        let bound = result[[0, 0]];

        // Monotonicity: upper bound should not increase as threshold gets tighter
        assert!(
            bound <= prev_bound + 1e-5,
            "Monotonicity violated: threshold={}, bound={}, prev_bound={}",
            threshold,
            bound,
            prev_bound
        );

        prev_bound = bound;
    }
}

/// Test feasibility: computed bounds should be valid (lower <= upper).
///
/// For a feasible problem, min objective <= max objective.
#[ntest::timeout(10000)]
#[test]
fn test_feasibility_bounds_ordering() {
    // Use several small random-like test cases
    let test_cases = [
        // (x_l, x_u, objective, a_matrix, b_vector)
        (
            array![[0.0, 0.0]],
            array![[1.0, 1.0]],
            array![[[1.0, 0.5]]],
            array![[[0.5, 0.5]]],
            array![[-0.7]], // x1/2 + x2/2 <= 0.7
        ),
        (
            array![[-1.0, -1.0]],
            array![[1.0, 1.0]],
            array![[[1.0, -1.0]]],
            array![[[1.0, 1.0]]],
            array![[-0.5]], // x1 + x2 <= 0.5
        ),
        (
            array![[0.0, 0.0, 0.0]],
            array![[1.0, 1.0, 1.0]],
            array![[[0.3, 0.5, 0.2]]],
            array![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            array![[-0.6, -0.7]], // x1 <= 0.6, x2 <= 0.7
        ),
    ];

    for (i, (x_l, x_u, obj, a_mat, b_vec)) in test_cases.iter().enumerate() {
        let lower = complete_clip(
            &x_l.clone().into_dyn(),
            &x_u.clone().into_dyn(),
            &obj.clone().into_dyn(),
            &a_mat.clone().into_dyn(),
            &b_vec.clone().into_dyn(),
            -1.0,
            false,
            1,
        )
        .expect("complete_clip (min) failed");

        let upper = complete_clip(
            &x_l.clone().into_dyn(),
            &x_u.clone().into_dyn(),
            &obj.clone().into_dyn(),
            &a_mat.clone().into_dyn(),
            &b_vec.clone().into_dyn(),
            1.0,
            false,
            1,
        )
        .expect("complete_clip (max) failed");

        let lower_val = lower[[0, 0]];
        let upper_val = upper[[0, 0]];

        assert!(
            lower_val <= upper_val + 1e-5,
            "Test case {}: lower bound {} > upper bound {} (invalid)",
            i,
            lower_val,
            upper_val
        );
    }
}

/// Test with multiple interacting constraints.
#[ntest::timeout(10000)]
#[test]
fn test_multiple_constraints() {
    // Minimize x1 over [0,1]^2 with:
    // - x1 + x2 <= 1.0
    // - x1 - x2 <= 0.5
    // - -x1 + x2 <= 0.5
    // Feasible region is a quadrilateral
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn(); // min x1
    let a_matrix = array![[[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]]].into_dyn();
    let b_vector = array![[-1.0, -0.5, -0.5]].into_dyn();

    let result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect("complete_clip failed");

    // Also check max
    let result_max = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, 1.0, false, 1)
        .expect("complete_clip (max) failed");

    let lower_bound = result[[0, 0]];
    let upper_bound = result_max[[0, 0]];

    let vertices = enumerate_feasible_vertices_2d(&x_l, &x_u, &a_matrix, &b_vector);
    assert!(
        !vertices.is_empty(),
        "Expected feasible region for multiple constraints"
    );

    let mut actual_min = f32::INFINITY;
    let mut actual_max = f32::NEG_INFINITY;
    let obj_c1 = objective[[0, 0, 0]];
    let obj_c2 = objective[[0, 0, 1]];
    for [x1, x2] in vertices {
        let obj_val = obj_c1 * x1 + obj_c2 * x2;
        actual_min = actual_min.min(obj_val);
        actual_max = actual_max.max(obj_val);
    }

    // Soundness: lower <= true min, upper >= true max
    assert!(
        lower_bound <= actual_min + 1e-4,
        "Lower bound {} > actual min {} (unsound)",
        lower_bound,
        actual_min
    );
    assert!(
        upper_bound >= actual_max - 1e-4,
        "Upper bound {} < actual max {} (unsound)",
        upper_bound,
        actual_max
    );
}

/// Test batch processing: multiple domains processed together.
#[ntest::timeout(10000)]
#[test]
fn test_batch_processing() {
    // Two batches with different domains
    let x_l = array![[0.0, 0.0], [-0.5, -0.5]].into_dyn();
    let x_u = array![[1.0, 1.0], [0.5, 0.5]].into_dyn();
    let objective = array![[[1.0, 1.0]], [[1.0, 1.0]]].into_dyn(); // sum for each batch
    let a_matrix = array![[[1.0, 0.0]], [[1.0, 0.0]]].into_dyn(); // x1 <= threshold
    let b_vector = array![[-0.5], [-0.0]].into_dyn(); // batch 0: x1<=0.5, batch 1: x1<=0

    let result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect("complete_clip failed");

    assert_eq!(result.shape(), &[2, 1]);

    // Batch 0: min(x1+x2) with x1<=0.5, x1,x2 in [0,1] -> min is 0 (x1=0, x2=0)
    let bound0 = result[[0, 0]];
    assert!(
        (-1e-6..=0.1).contains(&bound0),
        "Batch 0: expected min near 0, got {}",
        bound0
    );

    // Batch 1: min(x1+x2) with x1<=0, x1,x2 in [-0.5, 0.5] -> min is -1.0 (x1=-0.5, x2=-0.5)
    // But constraint x1<=0 doesn't affect min since x1=-0.5 satisfies it
    let bound1 = result[[1, 0]];
    assert!(
        (-1.1..=-0.9).contains(&bound1),
        "Batch 1: expected min near -1.0, got {}",
        bound1
    );
}

/// Test soundness: bound respects the actual achievable values.
///
/// The computed lower bound should be <= any feasible point's objective value.
#[ntest::timeout(10000)]
#[test]
fn test_soundness_against_grid_samples() {
    // Minimize x1 + 0.5*x2 over [0,1]^2 with x1 + x2 <= 1.2
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.5]]].into_dyn();
    let a_matrix = array![[[1.0, 1.0]]].into_dyn();
    let b_vector = array![[-1.2]].into_dyn();

    let lower_result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect("complete_clip failed");
    let computed_lower = lower_result[[0, 0]];

    let upper_result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, 1.0, false, 1)
        .expect("complete_clip (max) failed");
    let computed_upper = upper_result[[0, 0]];

    // Sample a grid of feasible points and check that computed bounds are valid
    let grid_size = 20;
    let mut actual_min = f32::INFINITY;
    let mut actual_max = f32::NEG_INFINITY;

    for i in 0..=grid_size {
        for j in 0..=grid_size {
            let x1 = i as f32 / grid_size as f32;
            let x2 = j as f32 / grid_size as f32;

            // Check constraint: x1 + x2 <= 1.2
            if x1 + x2 <= 1.2 + 1e-6 {
                let obj_val = x1 + 0.5 * x2;
                actual_min = actual_min.min(obj_val);
                actual_max = actual_max.max(obj_val);
            }
        }
    }

    // Soundness: computed lower <= actual min, computed upper >= actual max
    assert!(
        computed_lower <= actual_min + 1e-4,
        "Lower bound {} > actual min {} (unsound)",
        computed_lower,
        actual_min
    );
    assert!(
        computed_upper >= actual_max - 1e-4,
        "Upper bound {} < actual max {} (unsound)",
        computed_upper,
        actual_max
    );
}

/// Test with box-only constraints (no additional linear constraints).
#[ntest::timeout(10000)]
#[test]
fn test_box_only_no_constraints() {
    // min x1 + x2 over [0.2, 0.8]^2 with trivial constraint (always satisfied)
    let x_l = array![[0.2, 0.2]].into_dyn();
    let x_u = array![[0.8, 0.8]].into_dyn();
    let objective = array![[[1.0, 1.0]]].into_dyn();
    // Trivial constraint: 0*x + 0 <= 0 (always true)
    let a_matrix = array![[[0.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let lower = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect("complete_clip failed");
    let upper = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, 1.0, false, 1)
        .expect("complete_clip (max) failed");

    // min = 0.2 + 0.2 = 0.4, max = 0.8 + 0.8 = 1.6
    let lower_val = lower[[0, 0]];
    let upper_val = upper[[0, 0]];

    // Box-only: exact computation, 1e-3 for LP solver precision
    assert!(
        (lower_val - 0.4).abs() < 1e-3,
        "Expected lower ~0.4, got {}",
        lower_val
    );
    assert!(
        (upper_val - 1.6).abs() < 1e-3,
        "Expected upper ~1.6, got {}",
        upper_val
    );
}

/// Test infeasible constraint detection returns an error.
#[ntest::timeout(10000)]
#[test]
fn test_infeasible_constraint_returns_error() {
    // x in [0,1], constraint x1 <= -1 is infeasible.
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[1.0]].into_dyn(); // x1 + 1 <= 0 -> x1 <= -1

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("expected infeasible constraint to error");

    assert!(
        err.to_string().contains("infeasible constraint"),
        "unexpected error message: {}",
        err
    );
}

/// Test invalid num_iterations returns InvalidSpec.
#[ntest::timeout(10000)]
#[test]
fn test_invalid_num_iterations() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 0)
        .expect_err("expected invalid num_iterations to error");

    assert!(
        err.to_string().contains("num_iterations must be >= 1"),
        "unexpected error message: {}",
        err
    );
}

/// Test inverted input bounds return InvalidSpec before dual optimization.
#[ntest::timeout(10000)]
#[test]
fn test_inverted_input_bounds_return_invalid_spec() {
    let x_l = array![[1.0, 0.0]].into_dyn();
    let x_u = array![[0.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("inverted input bounds must be rejected");

    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec for inverted bounds, got {err:?}"
    );
    assert!(
        err.to_string().contains("complete_clip: x_l > x_u"),
        "unexpected error message: {err}"
    );
}

/// Test NaN dual results fail closed to the box-only baseline.
#[ntest::timeout(10000)]
#[test]
fn test_nan_constraint_bias_falls_back_to_box_baseline() {
    let x_l = array![[0.0]].into_dyn();
    let x_u = array![[1.0]].into_dyn();
    let objective = array![[[1.0]]].into_dyn();
    let a_matrix = array![[[1.0]]].into_dyn();
    let b_vector = array![[f32::NAN]].into_dyn();

    let result = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect("NaN constraint bias should fail closed to the baseline");

    assert!(
        result[[0, 0]].is_finite(),
        "baseline clamp must clear NaN results, got {}",
        result[[0, 0]]
    );
    assert!(
        result[[0, 0]].abs() < 1e-6,
        "box-only baseline for min x over [0,1] should be 0, got {}",
        result[[0, 0]]
    );
}

/// Test a_matrix shape mismatch returns InvalidSpec.
#[ntest::timeout(10000)]
#[test]
fn test_a_matrix_shape_mismatch() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]], [[1.0, 0.0]]].into_dyn(); // batch=2 mismatch
    let b_vector = array![[0.0], [0.0]].into_dyn();

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("expected a_matrix shape mismatch to error");

    assert!(
        err.to_string().contains("a_matrix shape"),
        "unexpected error message: {}",
        err
    );
}

/// Test objective with unsupported rank returns InvalidSpec.
#[ntest::timeout(10000)]
#[test]
fn test_objective_rank_invalid() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![1.0, 0.0].into_dyn(); // 1D is unsupported
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("expected objective rank error");

    assert!(
        err.to_string().contains("objective must be 2D or 3D"),
        "unexpected error message: {}",
        err
    );
}

/// Test invalid x_l rank returns InvalidSpec.
#[ntest::timeout(10000)]
#[test]
fn test_x_l_rank_invalid() {
    let x_l = array![0.0, 0.0].into_dyn(); // 1D invalid
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("expected x_l rank error");

    assert!(
        err.to_string().contains("x_l must be 2D"),
        "unexpected error message: {}",
        err
    );
}

/// Test x_u shape mismatch returns InvalidSpec.
#[ntest::timeout(10000)]
#[test]
fn test_x_u_shape_mismatch() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0, 1.0]].into_dyn(); // wrong x_dim
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("expected x_u shape mismatch error");

    assert!(
        err.to_string().contains("x_u shape"),
        "unexpected error message: {}",
        err
    );
}

/// Test b_vector rank invalid returns InvalidSpec.
#[ntest::timeout(10000)]
#[test]
fn test_b_vector_rank_invalid() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![0.0].into_dyn(); // 1D invalid

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("expected b_vector rank error");

    assert!(
        err.to_string().contains("b_vector must be 2D"),
        "unexpected error message: {}",
        err
    );
}

/// Test b_vector shape mismatch returns InvalidSpec.
#[ntest::timeout(10000)]
#[test]
fn test_b_vector_shape_mismatch() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0, 0.0]].into_dyn(); // n_constraints mismatch

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("expected b_vector shape mismatch error");

    assert!(
        err.to_string().contains("b_vector shape"),
        "unexpected error message: {}",
        err
    );
}

/// Test objective x_dim mismatch returns InvalidSpec.
#[ntest::timeout(10000)]
#[test]
fn test_objective_x_dim_mismatch() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0, 0.5]]].into_dyn(); // x_dim=3 mismatch
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let err = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect_err("expected objective x_dim mismatch error");

    assert!(
        err.to_string().contains("objective shape"),
        "unexpected error message: {}",
        err
    );
}

/// Test higher-dimensional input (4D).
#[ntest::timeout(10000)]
#[test]
fn test_higher_dimension() {
    // 4D input space
    let x_l = array![[0.0, 0.0, 0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0, 1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.5, 0.25, 0.125]]].into_dyn();
    // Constraint: sum(x) <= 2.0
    let a_matrix = array![[[1.0, 1.0, 1.0, 1.0]]].into_dyn();
    let b_vector = array![[-2.0]].into_dyn();

    let lower = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
        .expect("complete_clip failed");
    let upper = complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, 1.0, false, 1)
        .expect("complete_clip (max) failed");

    // min at x=(0,0,0,0) = 0
    // max at x=(1,1,1,1) subject to constraint sum<=2: with sum(x)<=2, max obj is bounded
    let lower_val = lower[[0, 0]];
    let upper_val = upper[[0, 0]];

    assert!(lower_val >= -0.1, "Expected lower >= 0, got {}", lower_val);
    assert!(
        lower_val <= upper_val + 1e-5,
        "Bound ordering violated: {} > {}",
        lower_val,
        upper_val
    );
}

/// Test rearrangement with multiple constraints improves or maintains bound.
#[ntest::timeout(10000)]
#[test]
fn test_rearrange_effect() {
    // With and without rearrangement should both produce valid bounds
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.5]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]].into_dyn();
    let b_vector = array![[-0.7, -0.6, -1.0]].into_dyn();

    let result_no_rearrange =
        complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
            .expect("without rearrange");

    let result_with_rearrange =
        complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, true, 1)
            .expect("with rearrange");

    // Both should produce valid bounds (similar values, rearrangement is heuristic)
    let bound_no = result_no_rearrange[[0, 0]];
    let bound_yes = result_with_rearrange[[0, 0]];

    // Both should be non-negative (feasible min is at least 0)
    assert!(
        bound_no >= -0.1,
        "No rearrange: expected non-negative, got {}",
        bound_no
    );
    assert!(
        bound_yes >= -0.1,
        "With rearrange: expected non-negative, got {}",
        bound_yes
    );
}

#[ntest::timeout(10000)]
#[test]
fn certified_rearrangement_keeps_witness_rows_aligned() {
    let x_l = array![[0.0f32]].into_dyn();
    let x_u = array![[1.0f32]].into_dyn();
    let objective = array![[[1.0f32]]].into_dyn();
    // 0.5 <= x <= 0.9. Rearrangement swaps/advises proposal order, and the
    // certificate must consume beta with the exact same A/b permutation.
    let a = array![[[-1.0f32], [1.0f32]]].into_dyn();
    let b = array![[0.5f32, -0.9f32]].into_dyn();
    let lower =
        complete_clip_certified_with_deadline(&x_l, &x_u, &objective, &a, &b, -1.0, true, 2, None)
            .expect("certified rearranged lower");
    assert!(lower[[0, 0]] <= 0.5 && lower[[0, 0]] > 0.499_9);
}

#[ntest::timeout(10000)]
#[test]
fn certified_clip_checks_deadline_and_budget_before_dense_work() {
    let cap = validate_clip_work_budget(1, 20, 8, 1_000_000)
        .expect_err("synthetic dense proposal must exceed hard cap");
    assert!(matches!(cap, NyError::CpuMemoryExceeded { .. }));
    let overflow = validate_clip_work_budget(usize::MAX, 2, 2, 2)
        .expect_err("shape product overflow must be fallible");
    assert!(matches!(overflow, NyError::InvalidSpec(_)));
    let op_cap = validate_clip_work_budget(1, 500, 500, 10_000)
        .expect_err("non-materialized H*N*X work must also have a hard cap");
    assert!(op_cap.to_string().contains("arithmetic budget exceeded"));
    let hidden_sort = validate_clip_iteration_budget(1, 10, 4, 1_000_000, 4)
        .expect_err("million-dimensional turning-point sorts must count logarithmic work");
    assert!(hidden_sort
        .to_string()
        .contains("sort-aware iteration budget exceeded"));
    // Intended TinyImageNet-shaped selective clipping remains inside the hard
    // envelope: 8 domains, 20 objectives, 4 constraints, 3x64x64 input.
    validate_clip_iteration_budget(8, 20, 4, 3 * 64 * 64, 1)
        .expect("Tiny selective sort cost should fit the certified cap");
    validate_clip_iteration_budget(8, 20, 4, 13_354, 1)
        .expect("last level-14 Tiny-shaped x_dim below the cap must fit");
    let boundary = validate_clip_iteration_budget(8, 20, 4, 13_355, 1)
        .expect_err("the next x_dim must cross the exact sort-aware cap");
    assert!(boundary.to_string().contains("total=500011456"));

    let x_l = array![[0.0f32]].into_dyn();
    let x_u = array![[1.0f32]].into_dyn();
    let objective = array![[[1.0f32]]].into_dyn();
    let a = array![[[-1.0f32]]].into_dyn();
    let b = array![[0.5f32]].into_dyn();
    let iter_err = complete_clip(
        &x_l,
        &x_u,
        &objective,
        &a,
        &b,
        -1.0,
        false,
        COMPLETE_CLIP_MAX_ITERS + 1,
    )
    .expect_err("iteration budget must fail closed");
    assert!(iter_err.to_string().contains("exceeds certified clip cap"));
    let err = complete_clip_certified_with_deadline(
        &x_l,
        &x_u,
        &objective,
        &a,
        &b,
        -1.0,
        false,
        1,
        Some(Instant::now()),
    )
    .expect_err("expired deadline must refuse before proposal allocation");
    assert!(matches!(err, NyError::DeadlineExceeded(_)));
}

#[ntest::timeout(10000)]
#[test]
fn dual_deadline_is_polled_inside_turning_point_sort() {
    let dim = 4096usize;
    let constr_a = Array2::from_elem((1, dim), 1.0f32);
    // A descending finite sequence forces a real comparison sort. The injected
    // callback expires on the first comparator poll, after preprocessing and
    // the explicit pre-sort check have all succeeded.
    let objective = Array3::from_shape_fn((1, 1, dim), |(_, _, x)| x as f32);
    let constr_d = Array1::zeros(1);
    let epsilon = Array2::from_elem((1, dim), 1.0f32);
    let mut polls = 0usize;
    let mut expires_during_sort = || {
        polls += 1;
        polls >= 14
    };
    let err = solve_dual_variable_with_deadline_check(
        &constr_a,
        &objective,
        &constr_d,
        &epsilon,
        &mut expires_during_sort,
    )
    .expect_err("in-sort expiry must interrupt the proposal deterministically");
    assert!(matches!(err, NyError::DeadlineExceeded(_)));
    assert!(err.to_string().contains("turning-point sort"));
    assert_eq!(polls, 14, "fixture must expire at the comparator poll");
}

#[ntest::timeout(10000)]
#[test]
fn x1_many_constraints_poll_feasibility_and_slack() {
    let n = 32_768usize;
    let a = Array3::zeros((1, n, 1)).into_dyn();
    let b = Array2::zeros((1, n)).into_dyn();
    let lo = array![[-1.0f32]].into_dyn();
    let hi = array![[1.0f32]].into_dyn();
    let mut polls = 0usize;
    let mut expire = || {
        polls += 1;
        polls >= 8
    };
    let err = ensure_constraints_feasible_with_deadline_check(&a, &b, &lo, &hi, &mut expire)
        .expect_err("X=1 must still poll across the constraint dimension");
    assert!(matches!(err, NyError::DeadlineExceeded(_)));
    assert!(err.to_string().contains("feasibility"));

    let a = a.into_dimensionality::<Ix3>().unwrap();
    let b = b.into_dimensionality::<Ix2>().unwrap();
    let x0 = Array2::zeros((1, 1));
    let mut polls = 0usize;
    let mut expire = || {
        polls += 1;
        polls >= 8
    };
    let err =
        compute_constraint_slack_with_deadline_check(a.view(), x0.view(), b.view(), &mut expire)
            .expect_err("X=1 slack fold must still poll across rows");
    assert!(matches!(err, NyError::DeadlineExceeded(_)));
    assert!(err.to_string().contains("slack"));
}

#[ntest::timeout(10000)]
#[test]
fn rearrangement_and_objective_dense_phases_fail_closed_on_expiry() {
    let n = 4096usize;
    let a = Array3::from_shape_fn((1, n, 1), |(_, k, _)| (k + 1) as f32).into_dyn();
    let b = Array2::zeros((1, n)).into_dyn();
    let x0 = Array2::from_elem((1, 1), 0.5f32);
    let mut polls = 0usize;
    let mut expire = || {
        polls += 1;
        polls >= 12
    };
    let err = rearrange_constraints_with_deadline_check(&a, &b, x0.view(), &mut expire)
        .expect_err("rearrangement must refuse an in-phase expiry");
    assert!(matches!(err, NyError::DeadlineExceeded(_)));

    let dim = 4096usize;
    let objective = Array3::from_elem((1, 1, dim), 1.0f32);
    let beta = Array2::from_elem((1, 1), 0.5f32);
    let constraint = Array2::from_elem((1, dim), 1.0f32);
    let mut polls = 0usize;
    let mut expire = || {
        polls += 1;
        polls >= 3
    };
    let err = update_objective_with_deadline_check(
        &objective,
        beta.view(),
        constraint.view(),
        1.0,
        &mut expire,
    )
    .expect_err("objective update must refuse an in-fold expiry");
    assert!(err.to_string().contains("objective-update"));

    let eps = Array2::from_elem((1, dim), 1.0f32);
    let mut polls = 0usize;
    let mut expire = || {
        polls += 1;
        polls >= 3
    };
    let err = compute_eps_term_with_deadline_check(&objective, eps.view(), &mut expire)
        .expect_err("epsilon fold must refuse an in-fold expiry");
    assert!(err.to_string().contains("epsilon-term"));
}
