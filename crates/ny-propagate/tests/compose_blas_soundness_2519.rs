// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BLAS compose concrete soundness tests for #2519.
//!
//! Verifies that the composed bounds (BLAS path) enclose the true two-layer
//! evaluation at concrete input points. The BLAS compose uses pos/neg coefficient
//! split per alpha-beta-CROWN backward_bound.py.

use ndarray::{ArrayD, IxDyn};
use ny_propagate::BatchedLinearBounds;

/// Create a BatchedLinearBounds with batch_size=1 from flat coefficient/bias arrays.
fn bounds_1d(
    la: &[f32],
    lb: &[f32],
    ua: &[f32],
    ub: &[f32],
    rows: usize,
    cols: usize,
) -> BatchedLinearBounds {
    BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[rows, cols]), la.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[rows]), lb.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[rows, cols]), ua.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[rows]), ub.to_vec()).unwrap(),
        vec![cols],
        vec![rows],
    )
    .unwrap()
}

#[test]
fn test_compose_exact_bounds_matches_analytical_2519() {
    // For exact (point) bounds where lower == upper, compose should yield:
    //   composed_a = a2 * a1
    //   composed_b = a2 * b1 + b2
    let a1 = bounds_1d(
        &[2.0, -1.0, 0.5, 3.0], // la = [[2,-1],[0.5,3]]
        &[1.0, -0.5],           // lb
        &[2.0, -1.0, 0.5, 3.0], // ua == la (exact)
        &[1.0, -0.5],           // ub == lb
        2,
        2,
    );
    let a2 = bounds_1d(
        &[-1.0, 2.0, 0.0, 1.0], // la = [[-1,2],[0,1]]
        &[0.5, -1.0],           // lb
        &[-1.0, 2.0, 0.0, 1.0], // ua == la
        &[0.5, -1.0],           // ub == lb
        2,
        2,
    );
    let result = a1.compose(&a2).unwrap();

    // Analytical: composed_a = a2 * a1
    //   [-1  2] * [2  -1]  = [-1*2+2*0.5  -1*(-1)+2*3] = [-1   7]
    //   [ 0  1]   [0.5 3]    [0*2+1*0.5    0*(-1)+1*3]   [0.5  3]
    let expected_a = [-1.0_f32, 7.0, 0.5, 3.0];
    // composed_b = a2 * b1 + b2
    //   [-1  2] * [1, -0.5]^T + [0.5, -1]^T = [-1-1+0.5, 0-0.5-1] = [-1.5, -1.5]
    let expected_b = [-1.5_f32, -1.5];

    for (i, (&la, &e)) in result.lower_a().iter().zip(expected_a.iter()).enumerate() {
        assert!(la <= e + 1e-5, "lower_a[{i}]={la} > expected {e}");
    }
    for (i, (&ua, &e)) in result.upper_a().iter().zip(expected_a.iter()).enumerate() {
        assert!(ua >= e - 1e-5, "upper_a[{i}]={ua} < expected {e}");
    }
    for (i, (&lb, &e)) in result.lower_b().iter().zip(expected_b.iter()).enumerate() {
        assert!(lb <= e + 1e-5, "lower_b[{i}]={lb} > expected {e}");
    }
    for (i, (&ub, &e)) in result.upper_b().iter().zip(expected_b.iter()).enumerate() {
        assert!(ub >= e - 1e-5, "upper_b[{i}]={ub} < expected {e}");
    }
}

#[test]
fn test_compose_interval_bounds_enclose_reference_2519() {
    // With interval bounds, verify the composed result encloses the
    // reference two-layer evaluation at concrete x = [1.0, 0.5].
    let b1 = bounds_1d(
        &[1.0, -2.0, 3.0, -1.0],
        &[0.5, 0.0],
        &[2.0, -1.0, 4.0, 0.0],
        &[1.0, 0.5],
        2,
        2,
    );
    let b2 = bounds_1d(
        &[-0.5, 1.0, 0.5, -0.5],
        &[0.1, -0.1],
        &[0.5, 2.0, 1.0, 0.5],
        &[0.2, 0.1],
        2,
        2,
    );
    let composed = b1.compose(&b2).unwrap();
    let x = [1.0_f64, 0.5];

    // Inner bounds at x: y_l[k] = sum_j b1_la[k,j]*x[j] + b1_lb[k]
    let b1_la = [[1.0_f64, -2.0], [3.0, -1.0]];
    let b1_ua = [[2.0_f64, -1.0], [4.0, 0.0]];
    let b1_lb = [0.5_f64, 0.0];
    let b1_ub = [1.0_f64, 0.5];
    let mut y_l = [0.0_f64; 2];
    let mut y_u = [0.0_f64; 2];
    for k in 0..2 {
        y_l[k] = b1_lb[k];
        y_u[k] = b1_ub[k];
        for j in 0..2 {
            y_l[k] += b1_la[k][j] * x[j];
            y_u[k] += b1_ua[k][j] * x[j];
        }
    }

    // Outer at y interval using pos/neg split (the correct CROWN semantic).
    let b2_la = [[-0.5_f64, 1.0], [0.5, -0.5]];
    let b2_ua = [[0.5_f64, 2.0], [1.0, 0.5]];
    let b2_lb = [0.1_f64, -0.1];
    let b2_ub = [0.2_f64, 0.1];
    for i in 0..2 {
        let mut z_l = b2_lb[i];
        let mut z_u = b2_ub[i];
        for k in 0..2 {
            if b2_la[i][k] >= 0.0 {
                z_l += b2_la[i][k] * y_l[k];
            } else {
                z_l += b2_la[i][k] * y_u[k];
            }
            if b2_ua[i][k] >= 0.0 {
                z_u += b2_ua[i][k] * y_u[k];
            } else {
                z_u += b2_ua[i][k] * y_l[k];
            }
        }
        // Evaluate composed bounds at x
        let mut comp_l = composed.lower_b()[[i]] as f64;
        let mut comp_u = composed.upper_b()[[i]] as f64;
        for j in 0..2 {
            comp_l += composed.lower_a()[[i, j]] as f64 * x[j];
            comp_u += composed.upper_a()[[i, j]] as f64 * x[j];
        }
        assert!(
            comp_l <= z_l + 1e-4,
            "composed lower {comp_l} > reference lower {z_l} at output {i}"
        );
        assert!(
            comp_u >= z_u - 1e-4,
            "composed upper {comp_u} < reference upper {z_u} at output {i}"
        );
    }
}
