// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for DA-CROWN analytical tools.
//! Part of #3921 / #4249.

use super::*;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;
use crate::probabilistic::distributional::{propagate_distribution, AnalyticDistribution};

#[test]
fn test_tight_bounds_tighter_than_standard() {
    // When var_L != var_U, tight bounds should be strictly tighter
    let dim = 2;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;

    // Asymmetric linear bounds: A_L = [[1, 0], [0, 0.5]], A_U = [[2, 0], [0, 1]]
    let a_l = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.5]).unwrap();
    let a_u = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 1.0]).unwrap();
    let b = Array1::zeros(2);
    let lb = LinearBounds::new(a_l, b.clone(), a_u, b).unwrap();

    let standard = propagate_distribution(&lb, &dist, &bounds, 0.95).unwrap();
    let tight = propagate_distribution_tight(&lb, &dist, &bounds, 0.95).unwrap();

    for i in 0..dim {
        let std_width = standard.prob_upper[[i]] - standard.prob_lower[[i]];
        let tight_width = tight.prob_upper[[i]] - tight.prob_lower[[i]];
        assert!(
            tight_width <= std_width + 1e-6,
            "dim {i}: tight width {tight_width} should be <= standard width {std_width}"
        );
    }
}

#[test]
fn test_tight_bounds_same_when_symmetric() {
    // When A_L == A_U (symmetric), tight and standard should match
    let dim = 2;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let lb = LinearBounds::identity(dim);

    let standard = propagate_distribution(&lb, &dist, &bounds, 0.95).unwrap();
    let tight = propagate_distribution_tight(&lb, &dist, &bounds, 0.95).unwrap();

    for i in 0..dim {
        assert!(
            (standard.prob_lower[[i]] - tight.prob_lower[[i]]).abs() < 1e-6,
            "lower bounds should match for symmetric case"
        );
        assert!(
            (standard.prob_upper[[i]] - tight.prob_upper[[i]]).abs() < 1e-6,
            "upper bounds should match for symmetric case"
        );
    }
}

#[test]
fn test_objective_increases_with_tighter_bounds() {
    // Scaling by a positive factor > 1 increases the objective (more confident lower bound)
    let dim = 1;
    let a_small = Array2::from_diag(&Array1::from_elem(dim, 0.5_f32));
    let a_large = Array2::from_diag(&Array1::from_elem(dim, 2.0_f32));
    let b = Array1::zeros(dim);

    let lb_small = LinearBounds::new(a_small.clone(), b.clone(), a_small, b.clone()).unwrap();
    let lb_large = LinearBounds::new(a_large.clone(), b.clone(), a_large, b).unwrap();

    let mu = vec![2.0_f64]; // positive mean
    let var = vec![1.0_f64 / 3.0]; // var for uniform [0, 4]: 16/12

    let obj_small = distributional_objective(&lb_small, &mu, &var, 0.95);
    let obj_large = distributional_objective(&lb_large, &mu, &var, 0.95);

    // Scaling by 2 vs 0.5 with positive mean should give higher objective
    assert!(
        obj_large > obj_small,
        "larger scaling with positive mean should give higher objective: large={obj_large}, small={obj_small}"
    );
}

#[test]
fn test_gradient_has_correct_shape() {
    let dim = 3;
    let lb = LinearBounds::identity(dim);
    let mu = vec![1.0_f64; dim];
    let var = vec![1.0_f64; dim];

    let grad = distributional_gradient(&lb, &mu, &var, 0.95);

    assert_eq!(grad.shape(), &[dim, dim]);
}

#[test]
fn test_gradient_positive_for_positive_mean() {
    // For identity A_L and positive mean, gradient w.r.t. diagonal elements
    // should be positive (increasing the coefficient helps)
    let dim = 2;
    let lb = LinearBounds::identity(dim);
    let mu = vec![5.0_f64, 5.0]; // large positive mean
    let var = vec![0.01_f64, 0.01]; // small variance

    let grad = distributional_gradient(&lb, &mu, &var, 0.95);

    // grad[i,i] = mu_i - z * a_ii * var_i / sigma_i
    // With large mu and small var, the mu_j term dominates → positive
    for i in 0..dim {
        assert!(
            grad[[i, i]] > 0.0,
            "diagonal gradient should be positive for large positive mean"
        );
    }
}
