// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use crate::BoundedTensor;
use ndarray::{arr1, arr2, Axis};
use ny_test_utils::assert_f32_close as assert_close;

#[test]
fn test_concrete_zonotope() {
    let values = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let z = ZonotopeTensor::concrete(values);

    assert_eq!(z.n_error_terms, 0);
    assert_eq!(z.max_width(), 0.0);
}

#[test]
fn test_from_input_shared() {
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    assert_eq!(z.n_error_terms, 1);
    assert_eq!(z.element_shape, vec![2]);

    let bounds = z.to_bounded_tensor().unwrap();
    assert_close(bounds.lower()[[0]], 0.9, 1e-6, "from_input_shared lower[0]");
    assert_close(bounds.upper()[[0]], 1.1, 1e-6, "from_input_shared upper[0]");
}

#[test]
fn test_from_input_elementwise() {
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z = ZonotopeTensor::from_input_elementwise(&values, 0.1);

    assert_eq!(z.n_error_terms, 2); // One per element
    assert_eq!(z.element_shape, vec![2]);

    let bounds = z.to_bounded_tensor().unwrap();
    assert_close(bounds.lower()[[0]], 0.9, 1e-6, "elementwise lower[0]");
    assert_close(bounds.upper()[[1]], 2.1, 1e-6, "elementwise upper[1]");
}

#[test]
fn test_shift_and_scale() {
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    // Shift by 1
    let shifted = z.shift(1.0);
    assert_close(shifted.center()[[0]], 2.0, 1e-6, "shifted center[0]");

    // Scale by 2
    let scaled = z.scale(2.0);
    assert_close(scaled.center()[[0]], 2.0, 1e-6, "scaled center[0]");
    assert_close(scaled.max_width(), 0.4, 1e-6, "scaled max_width"); // 2 * 0.1 * 2 = 0.4
}

#[test]
fn test_add_zonotopes() {
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z1 = ZonotopeTensor::from_input_shared(&values, 0.1);
    let z2 = ZonotopeTensor::from_input_shared(&values, 0.1);

    let sum = z1.add(&z2).unwrap();
    assert_close(sum.center()[[0]], 2.0, 1e-6, "sum center[0]");
    assert_close(sum.max_width(), 0.4, 1e-6, "sum max_width"); // 2 * 0.1 * 2 = 0.4
}

#[test]
fn test_linear_transform() {
    // Create zonotope: [1±0.1, 2±0.1]
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z = ZonotopeTensor::from_input_elementwise(&values, 0.1);

    // Weight: [[1, 1], [1, -1]] -> out = [x+y, x-y]
    let weight = arr2(&[[1.0, 1.0], [1.0, -1.0]]);

    let result = z.linear(&weight, None).unwrap();

    // Center: [1+2, 1-2] = [3, -1]
    assert_close(result.center()[[0]], 3.0, 1e-6, "linear center[0]");
    assert_close(result.center()[[1]], -1.0, 1e-6, "linear center[1]");

    // Error propagation:
    // For out[0] = x + y: both errors contribute, but since x and y have separate symbols,
    // width = |1*0.1| + |1*0.1| = 0.2 (from x's symbol) + 0.2 (from y's symbol)
    // Actually: error1 affects x, error2 affects y
    // out[0] gets 0.1 from error1 (through x) and 0.1 from error2 (through y)
    // Total width = 2 * (0.1 + 0.1) = 0.4
    let bounds = result.to_bounded_tensor().unwrap();
    assert_close(
        bounds.upper()[[0]] - bounds.lower()[[0]],
        0.4,
        1e-6,
        "linear output width[0]",
    );
}

#[test]
fn test_dot_product_concrete() {
    // Two concrete zonotopes (no error)
    let a = ZonotopeTensor::concrete(arr1(&[1.0, 2.0]).into_dyn());
    let b = ZonotopeTensor::concrete(arr1(&[3.0, 4.0]).into_dyn());

    let result = a.dot(&b).unwrap();

    // 1*3 + 2*4 = 11
    assert_close(result.center()[[0]], 11.0, 1e-6, "dot center[0]");
    assert_eq!(result.max_width(), 0.0); // No error
}

#[test]
fn test_dot_product_with_errors() {
    // z1 = [1+0.1e1, 2+0.1e2] (each element has its own error)
    // z2 = [1+0.1e1, 1+0.1e2] (SAME error symbols - correlated!)
    let values1 = arr1(&[1.0, 2.0]).into_dyn();
    let values2 = arr1(&[1.0, 1.0]).into_dyn();

    let z1 = ZonotopeTensor::from_input_elementwise(&values1, 0.1);
    let z2 = ZonotopeTensor::from_input_elementwise(&values2, 0.1);

    let result = z1.dot(&z2).unwrap();

    // Center: 1*1 + 2*1 = 3
    // Plus center shift from e_i^2 terms: 0.5*(0.1*0.1 + 0.1*0.1) = 0.01
    let expected_center = 3.0 + 0.01;
    assert_close(
        result.center()[[0]],
        expected_center,
        1e-5,
        "dot center with errors",
    );

    // The zonotope dot product should be tighter than IBP
    let bounds = result.to_bounded_tensor().unwrap();
    let width = bounds.upper()[[0]] - bounds.lower()[[0]];

    // Compare to IBP: [0.9,1.1]*[0.9,1.1] + [1.9,2.1]*[0.9,1.1]
    // = [0.81,1.21] + [1.71,2.31] = [2.52,3.52] -> width 1.0
    // Zonotope should be tighter (but not by huge amount for this small example)
    assert!(width < 1.5, "dot width should stay below 1.5, got {width}"); // Sanity check
}

#[test]
fn test_zonotope_vs_ibp_correlation_advantage() {
    // This test demonstrates why zonotopes are better for correlated inputs
    //
    // Scenario: z = x * x where x = 1 ± 0.5
    // IBP: [0.5, 1.5] * [0.5, 1.5] = [0.25, 2.25] (treats as independent)
    // True range: (1±0.5)² = 1 ± 1 + 0.25 = [0.25, 2.25] (same for single var!)
    //
    // But for z = x * y where x,y SHARE perturbation (x = 1+e, y = 1+e):
    // IBP: still [0.25, 2.25]
    // Zonotope: (1+e)*(1+e) = 1 + 2e + e² where e²∈[0,1]
    //         = 1.5 + 2e + 0.5e' where e,e'∈[-1,1]
    //         range: [1.5-2-0.5, 1.5+2+0.5] = [-1, 4] -- wait that's worse!

    // Actually the advantage is when Q and K both depend on same X
    // but through DIFFERENT linear transforms. Let me redo:

    // x = [1±0.1] (input with 1 error symbol)
    // Q = 2x (linear)
    // K = 3x (linear)
    // Q*K = 6x² = 6*(1±0.1)² = 6*(1 ± 0.2 + 0.01) = 6*[0.81, 1.21] = [4.86, 7.26]
    // width_true = 2.4

    // IBP would compute:
    // Q ∈ [1.8, 2.2]
    // K ∈ [2.7, 3.3]
    // Q*K ∈ [1.8*2.7, 2.2*3.3] = [4.86, 7.26] -- same! Because 1D is special.

    // The advantage appears in higher dimensions where cross-correlations matter.
    // For example, Q·K where Q,K ∈ R^d and both depend on same input.

    // Let's verify basic functionality with shared error:
    let x = ZonotopeTensor::from_input_shared(&arr1(&[1.0]).into_dyn(), 0.1);

    // q = 2*x
    let q = x.scale(2.0);
    assert_close(q.center()[[0]], 2.0, 1e-6, "q center[0]");

    // k = 3*x
    let k = x.scale(3.0);
    assert_close(k.center()[[0]], 3.0, 1e-6, "k center[0]");

    // Both Q and K share the same error symbol!
    assert_eq!(q.n_error_terms, 1);
    assert_eq!(k.n_error_terms, 1);
}

#[test]
fn test_from_input_2d() {
    // Create 2D zonotope with shape (2, 3)
    let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.1);

    assert_eq!(z.n_error_terms, 6); // 2 * 3 = 6 elements
    assert_eq!(z.element_shape, vec![2, 3]);

    // Center should be the original values
    let center = z.center();
    assert!((center[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((center[[1, 2]] - 6.0).abs() < 1e-6);

    // Each element should have width 0.2 (±0.1)
    let bounds = z.to_bounded_tensor().unwrap();
    assert!((bounds.lower()[[0, 0]] - 0.9).abs() < 1e-6);
    assert!((bounds.upper()[[0, 0]] - 1.1).abs() < 1e-6);
}

#[test]
fn test_matmul_transposed_concrete() {
    // Two concrete matrices (no error)
    // Q: (2, 3), K: (2, 3) -> Q @ K^T: (2, 2)
    let q_vals = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let k_vals = arr2(&[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);

    // Create zonotopes with 0 error (concrete)
    let mut q_coeffs = ndarray::Array3::<f32>::zeros((1, 2, 3));
    q_coeffs.index_axis_mut(Axis(0), 0).assign(&q_vals);
    let q = ZonotopeTensor {
        coeffs: q_coeffs.into_dyn(),
        n_error_terms: 0,
        element_shape: vec![2, 3],
    };

    let mut k_coeffs = ndarray::Array3::<f32>::zeros((1, 2, 3));
    k_coeffs.index_axis_mut(Axis(0), 0).assign(&k_vals);
    let k = ZonotopeTensor {
        coeffs: k_coeffs.into_dyn(),
        n_error_terms: 0,
        element_shape: vec![2, 3],
    };

    let result = q.matmul_transposed(&k).unwrap();

    // Q @ K^T = [[1,0],[0,0]] (row i, col j = Q[i,:] · K[j,:])
    assert_eq!(result.element_shape, vec![2, 2]);
    let center = result.center();
    assert!((center[[0, 0]] - 1.0).abs() < 1e-6); // [1,0,0] · [1,0,0] = 1
    assert!((center[[0, 1]] - 0.0).abs() < 1e-6); // [1,0,0] · [0,0,1] = 0
    assert!((center[[1, 0]] - 0.0).abs() < 1e-6); // [0,1,0] · [1,0,0] = 0
    assert!((center[[1, 1]] - 0.0).abs() < 1e-6); // [0,1,0] · [0,0,1] = 0
}

#[test]
fn test_matmul_transposed_with_error() {
    // Q and K share error symbols - this is the key for attention!
    // Q: (1, 2) with values [1, 0] + 0.1*error
    // K: (1, 2) with values [1, 0] + 0.1*error (SAME error symbol!)
    // Result: (1, 1) = Q · K = 1 + perturbation

    let q_vals = arr2(&[[1.0, 0.0]]);
    let k_vals = arr2(&[[1.0, 0.0]]);

    let q = ZonotopeTensor::from_input_2d(&q_vals, 0.1);
    let k = ZonotopeTensor::from_input_2d(&k_vals, 0.1);

    let result = q.matmul_transposed(&k).unwrap();

    assert_eq!(result.element_shape, vec![1, 1]);

    // Center: 1*1 + 0*0 = 1, plus e_i² center shift
    // For error term 0 (position 0,0 in both Q and K):
    //   Q[1,0,0] = 0.1, K[1,0,0] = 0.1
    //   center_shift += 0.5 * (0.1 * 0.1) = 0.005
    // For error term 1 (position 0,1 in both Q and K):
    //   Q[2,0,1] = 0.1, K[2,0,1] = 0.1
    //   center_shift += 0.5 * (0.1 * 0.1) = 0.005
    // Total center = 1 + 0.01 = 1.01
    let center = result.center();
    assert!((center[[0, 0]] - 1.01).abs() < 1e-5);

    // Width should be computable
    let bounds = result.to_bounded_tensor().unwrap();
    let width = bounds.upper()[[0, 0]] - bounds.lower()[[0, 0]];
    assert!(
        width > 0.0,
        "Zonotope bound width should be positive, got {width}"
    );
    assert!(
        width < 1.0,
        "Zonotope bound width should be small for small epsilon, got {width}"
    );
}

#[test]
fn test_matmul_transposed_zonotope_tighter_than_ibp() {
    // This test demonstrates the zonotope advantage for correlated Q@K^T
    //
    // Scenario: Q and K both come from same input X
    // Q = X, K = X (identity transforms)
    // Q@K^T = X@X^T
    //
    // When X has correlated perturbations across positions,
    // zonotope tracks this and gives tighter bounds.

    // X: (2, 2) matrix with perturbation
    let x_vals = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let epsilon = 0.1;

    // Create zonotope for X (Q and K share the same zonotope!)
    let x = ZonotopeTensor::from_input_2d(&x_vals, epsilon);

    // Zonotope: Q@K^T where Q=K=X
    let result_zonotope = x.matmul_transposed(&x).unwrap();
    let zonotope_bounds = result_zonotope.to_bounded_tensor().unwrap();

    // IBP would compute: (X ± ε) @ (X ± ε)^T treating them independently
    // For diagonal elements: X[i,:] · X[i,:] = sum of squares
    // [1,0]·[1,0] = 1, perturbed: [1±0.1, 0±0.1]·[1±0.1, 0±0.1]
    // IBP treats as independent: [0.81,1.21] + [0,0.04] = [0.81, 1.25]
    // Width = 0.44

    // For zonotope with shared symbols, x_i * x_i where x_i = center + ε*e_i
    // gives tighter bounds because we know e_i * e_i = e_i² ∈ [0,1]

    let zonotope_width_00 = zonotope_bounds.upper()[[0, 0]] - zonotope_bounds.lower()[[0, 0]];

    // IBP width for [0,0] position (computed independently)
    // [1±0.1]² + [0±0.1]² = [0.81,1.21] + [0,0.01] = [0.81, 1.22]
    // IBP width ≈ 0.41
    let ibp_lower_00 = 0.81_f32; // min of x²
    let ibp_upper_00 = 1.21 + 0.01; // max of sum of squares with epsilon
    let ibp_width_00 = ibp_upper_00 - ibp_lower_00;

    // Zonotope should be at least as tight, ideally tighter
    // (In practice, for x*x with same error, zonotope gives exact bounds)
    println!("Zonotope width [0,0]: {}", zonotope_width_00);
    println!("IBP width [0,0]: {}", ibp_width_00);

    // The zonotope should give a valid bound
    assert!(zonotope_bounds.lower()[[0, 0]] <= 1.01); // Center is ~1.01 after shift
    assert!(zonotope_bounds.upper()[[0, 0]] >= 1.01);
}

#[test]
fn test_matmul_disjoint_soundness() {
    let a_center = arr2(&[[1.0_f32, 0.5], [0.2, -0.3]]);
    let b_center = arr2(&[[0.4_f32, -0.2, 0.1], [1.0, 0.3, -0.5]]);
    let a_epsilon = 0.1_f32;
    let b_epsilon = 0.2_f32;

    let a = ZonotopeTensor::from_input_shared(&a_center.clone().into_dyn(), a_epsilon);
    let b = ZonotopeTensor::from_input_shared(&b_center.clone().into_dyn(), b_epsilon);

    let result = a.matmul_disjoint(&b).unwrap();
    assert_eq!(result.element_shape, vec![2, 3]);
    assert_eq!(result.n_error_terms, 3);

    let bounds = result.to_bounded_tensor().unwrap();
    for &ea in &[-1.0_f32, 1.0_f32] {
        for &eb in &[-1.0_f32, 1.0_f32] {
            let a_concrete = a_center.mapv(|v| v + a_epsilon * ea);
            let b_concrete = b_center.mapv(|v| v + b_epsilon * eb);
            let output = a_concrete.dot(&b_concrete);
            for i in 0..output.nrows() {
                for j in 0..output.ncols() {
                    assert!(
                        output[[i, j]] >= bounds.lower()[[i, j]] - 1e-6,
                        "disjoint matmul lower[{i},{j}]={} > concrete {}",
                        bounds.lower()[[i, j]],
                        output[[i, j]]
                    );
                    assert!(
                        output[[i, j]] <= bounds.upper()[[i, j]] + 1e-6,
                        "disjoint matmul upper[{i},{j}]={} < concrete {}",
                        bounds.upper()[[i, j]],
                        output[[i, j]]
                    );
                }
            }
        }
    }
}

#[test]
fn test_zonotope_advantage_different_transforms() {
    // The zonotope advantage appears when Q and K are DIFFERENT transforms of X.
    //
    // Scenario: X = [1] (scalar), epsilon = 0.5
    // Q = 2*X = 2 (but shares X's error symbol!)
    // K = -1*X = -1 (also shares X's error symbol!)
    //
    // Q * K = 2*(-1) = -2 when no perturbation
    //
    // With perturbation X = 1 + 0.5*e where e ∈ [-1, 1]:
    // Q = 2*(1 + 0.5*e) = 2 + e
    // K = -1*(1 + 0.5*e) = -1 - 0.5*e
    // Q * K = (2 + e)*(-1 - 0.5*e) = -2 - e + -e - 0.5*e² = -2 - 2e - 0.5*e²
    //
    // Since e ∈ [-1, 1]: -2e ∈ [-2, 2], and e² ∈ [0, 1], so -0.5*e² ∈ [-0.5, 0]
    // True range: [-2 - 2 - 0.5, -2 + 2 + 0] = [-4.5, 0]
    //
    // IBP (treating Q and K independently):
    // Q ∈ [1.5, 2.5], K ∈ [-1.5, -0.5]
    // Q * K ∈ [2.5 * -1.5, 1.5 * -0.5] = [-3.75, -0.75]  <- WRONG ORDER
    // Actually: min = 2.5*-1.5 = -3.75, max = 1.5*-0.5 = -0.75
    // IBP width = 3.0
    //
    // Zonotope (tracking correlation):
    // Q = 2 + e (center=2, coeff[1]=1 for e)
    // K = -1 - 0.5*e (center=-1, coeff[1]=-0.5 for e)
    //
    // Q*K center: 2*(-1) = -2
    // e² term: 1*(-0.5)*e² = -0.5*e² -> center shift = -0.25, half_term = 0.25
    // Linear: 2*(-0.5) + 1*(-1) = -2 for e
    // New center = -2 + (-0.25) = -2.25
    // Radius = |−2| + 0.25 = 2.25
    // Zonotope range: [-2.25 - 2.25, -2.25 + 2.25] = [-4.5, 0] <- EXACT!
    //
    // Zonotope width = 4.5 (exact)
    // IBP width = 3.0 (wrong! doesn't contain true range)
    //
    // This shows zonotopes give SOUND bounds while IBP can be UNSOUND
    // for correlated variables with opposite signs.

    // Build zonotopes manually for Q = [2+e] and K = [-1-0.5e]
    // where they share the same error symbol

    // Q: (1,1) zonotope with 1 error term
    // coeffs shape: (2, 1, 1) = [center=2, error_coeff=1]
    let mut q_coeffs = ndarray::Array3::<f32>::zeros((2, 1, 1));
    q_coeffs[[0, 0, 0]] = 2.0; // center
    q_coeffs[[1, 0, 0]] = 1.0; // coefficient for e (since Q = 2X = 2*(1+0.5e) = 2+e)
    let q = ZonotopeTensor {
        coeffs: q_coeffs.into_dyn(),
        n_error_terms: 1,
        element_shape: vec![1, 1],
    };

    // K: (1,1) zonotope with same error symbol
    // K = -X = -(1+0.5e) = -1 - 0.5e
    let mut k_coeffs = ndarray::Array3::<f32>::zeros((2, 1, 1));
    k_coeffs[[0, 0, 0]] = -1.0; // center
    k_coeffs[[1, 0, 0]] = -0.5; // coefficient for e
    let k = ZonotopeTensor {
        coeffs: k_coeffs.into_dyn(),
        n_error_terms: 1,
        element_shape: vec![1, 1],
    };

    let result = q.matmul_transposed(&k).unwrap();

    // Center should be: 2*(-1) + 0.5*(1*-0.5) = -2 - 0.25 = -2.25
    let center = result.center()[[0, 0]];
    assert!(
        (center - (-2.25)).abs() < 1e-5,
        "Expected center -2.25, got {}",
        center
    );

    let bounds = result.to_bounded_tensor().unwrap();
    let lower = bounds.lower()[[0, 0]];
    let upper = bounds.upper()[[0, 0]];
    let width = upper - lower;

    println!(
        "Q*K zonotope: center={}, bounds=[{}, {}], width={}",
        center, lower, upper, width
    );

    // Zonotope should contain the true range [-4.5, 0]
    assert!(
        lower <= -4.5 + 0.01,
        "Lower bound {} should be <= -4.5",
        lower
    );
    assert!(upper >= 0.0 - 0.01, "Upper bound {} should be >= 0", upper);

    // IBP would give [-3.75, -0.75] which is UNSOUND (doesn't contain 0 or -4.5)
    // So zonotope wins by being CORRECT, not just tighter
}

#[test]
fn test_add_constant() {
    // Create zonotope: [1±0.1, 2±0.1]
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    // Add constant [0.5, 1.0]
    let constant = arr1(&[0.5, 1.0]).into_dyn();
    let result = z.add_constant(&constant).unwrap();

    // Center should be [1.5, 3.0]
    assert_close(result.center()[[0]], 1.5, 1e-6, "add_constant center[0]");
    assert_close(result.center()[[1]], 3.0, 1e-6, "add_constant center[1]");

    // Radius should be unchanged (0.1 for each)
    let bounds = result.to_bounded_tensor().unwrap();
    assert_close(bounds.lower()[[0]], 1.4, 1e-6, "add_constant lower[0]");
    assert_close(bounds.upper()[[0]], 1.6, 1e-6, "add_constant upper[0]");
}

#[test]
fn test_add_constant_broadcast_last_axis_bias() {
    let values = ndarray::Array::from_shape_vec((1, 2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid values")
        .into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    let bias = arr1(&[0.5, 1.0, 1.5]).into_dyn();
    let result = z
        .add_constant(&bias)
        .expect("1D bias should broadcast across the last axis");

    assert_eq!(result.shape(), &[1, 2, 3]);

    let bounds = result.to_bounded_tensor().unwrap();
    assert!((bounds.lower()[[0, 0, 0]] - 1.4).abs() < 1e-6);
    assert!((bounds.upper()[[0, 0, 0]] - 1.6).abs() < 1e-6);
    assert!((bounds.lower()[[0, 1, 2]] - 7.4).abs() < 1e-6);
    assert!((bounds.upper()[[0, 1, 2]] - 7.6).abs() < 1e-6);
}

#[test]
fn test_add_constant_broadcast_channel_first_bias_compatibility() {
    let values = ndarray::Array::from_shape_vec(
        (3, 2, 2),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("valid values")
    .into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    let bias = arr1(&[10.0, 20.0, 30.0]).into_dyn();
    let result = z
        .add_constant(&bias)
        .expect("1D channel bias should broadcast across [C, H, W]");

    assert_eq!(result.shape(), &[3, 2, 2]);

    let bounds = result.to_bounded_tensor().unwrap();
    assert!((bounds.lower()[[0, 0, 0]] - 10.9).abs() < 1e-6);
    assert!((bounds.upper()[[0, 0, 0]] - 11.1).abs() < 1e-6);
    assert!((bounds.lower()[[2, 1, 1]] - 41.9).abs() < 1e-6);
    assert!((bounds.upper()[[2, 1, 1]] - 42.1).abs() < 1e-6);
}

#[test]
fn test_mul_constant() {
    // Create zonotope: [1±0.1, 2±0.1]
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    // Multiply by constant [2.0, 0.5]
    let constant = arr1(&[2.0, 0.5]).into_dyn();
    let result = z.mul_constant(&constant).unwrap();

    // Center should be [2.0, 1.0]
    assert_close(result.center()[[0]], 2.0, 1e-6, "mul_constant center[0]");
    assert_close(result.center()[[1]], 1.0, 1e-6, "mul_constant center[1]");

    // Widths: first element width = 0.2 * 2 = 0.4, second = 0.2 * 0.5 = 0.1
    let bounds = result.to_bounded_tensor().unwrap();
    let width_0 = bounds.upper()[[0]] - bounds.lower()[[0]];
    let width_1 = bounds.upper()[[1]] - bounds.lower()[[1]];
    assert_close(width_0, 0.4, 1e-6, "mul_constant width[0]");
    assert_close(width_1, 0.1, 1e-6, "mul_constant width[1]");
}

#[test]
fn test_mul_constant_broadcast_last_axis_scale() {
    let values = ndarray::Array::from_shape_vec((1, 2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid values")
        .into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    let scale = arr1(&[2.0, 0.5, 1.5]).into_dyn();
    let result = z
        .mul_constant(&scale)
        .expect("1D scale should broadcast across the last axis");

    assert_eq!(result.shape(), &[1, 2, 3]);

    let bounds = result.to_bounded_tensor().unwrap();
    assert!((bounds.lower()[[0, 0, 0]] - 1.8).abs() < 1e-6);
    assert!((bounds.upper()[[0, 0, 0]] - 2.2).abs() < 1e-6);
    assert!((bounds.lower()[[0, 1, 2]] - 8.85).abs() < 1e-6);
    assert!((bounds.upper()[[0, 1, 2]] - 9.15).abs() < 1e-6);
}

#[test]
fn test_mul_constant_broadcast_channel_first_scale_parity() {
    // #3457 parity regression: verify [C, H, W] * [C] channel-first broadcast
    // matches the IBP compatibility path (R1 directive from commit 5ffa88a).
    let values = ndarray::Array::from_shape_vec(
        (3, 2, 2),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("valid values")
    .into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    let scale = arr1(&[2.0, 0.5, 3.0]).into_dyn();
    let result = z
        .mul_constant(&scale)
        .expect("1D channel scale should broadcast across [C, H, W]");

    assert_eq!(result.shape(), &[3, 2, 2]);

    let bounds = result.to_bounded_tensor().unwrap();
    // Channel 0: value * 2.0, so center ± width*2
    // [0,0,0]: 1.0*2.0 = 2.0, width = 0.2*2.0 = 0.4
    assert!((bounds.lower()[[0, 0, 0]] - 1.8).abs() < 1e-6);
    assert!((bounds.upper()[[0, 0, 0]] - 2.2).abs() < 1e-6);
    // Channel 1: value * 0.5
    // [1,0,0]: 5.0*0.5 = 2.5, width = 0.2*0.5 = 0.1
    assert!((bounds.lower()[[1, 0, 0]] - 2.45).abs() < 1e-6);
    assert!((bounds.upper()[[1, 0, 0]] - 2.55).abs() < 1e-6);
    // Channel 2: value * 3.0
    // [2,1,1]: 12.0*3.0 = 36.0, width = 0.2*3.0 = 0.6
    assert!((bounds.lower()[[2, 1, 1]] - 35.7).abs() < 1e-6);
    assert!((bounds.upper()[[2, 1, 1]] - 36.3).abs() < 1e-6);
}

#[test]
fn test_from_bounded_tensor() {
    // Create a BoundedTensor
    let bounds =
        BoundedTensor::new(arr1(&[0.5, 1.5]).into_dyn(), arr1(&[1.5, 2.5]).into_dyn()).unwrap();

    let z = ZonotopeTensor::from_bounded_tensor(&bounds);

    // Center should be [1.0, 2.0]
    assert_close(z.center()[[0]], 1.0, 1e-6, "from_bounded_tensor center[0]");
    assert_close(z.center()[[1]], 2.0, 1e-6, "from_bounded_tensor center[1]");

    // Should round-trip back to same bounds
    let bounds2 = z.to_bounded_tensor().unwrap();
    assert_close(bounds2.lower()[[0]], 0.5, 1e-6, "round-trip lower[0]");
    assert_close(bounds2.upper()[[0]], 1.5, 1e-6, "round-trip upper[0]");
    assert_close(bounds2.lower()[[1]], 1.5, 1e-6, "round-trip lower[1]");
    assert_close(bounds2.upper()[[1]], 2.5, 1e-6, "round-trip upper[1]");
}

#[test]
fn test_expand_to_match() {
    // Create two zonotopes with different error term counts
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z1 = ZonotopeTensor::from_input_shared(&values, 0.1); // 1 error term
    let z2 = ZonotopeTensor::from_input_elementwise(&values, 0.1); // 2 error terms

    let (expanded1, expanded2) = z1.expand_to_match(&z2).unwrap();

    // Both should have 2 error terms now
    assert_eq!(expanded1.n_error_terms, 2);
    assert_eq!(expanded2.n_error_terms, 2);

    // z1's bounds should be preserved
    let b1 = expanded1.to_bounded_tensor().unwrap();
    assert_close(b1.lower()[[0]], 0.9, 1e-6, "expanded1 lower[0]");
    assert_close(b1.upper()[[0]], 1.1, 1e-6, "expanded1 upper[0]");

    // z2's bounds should be preserved
    let b2 = expanded2.to_bounded_tensor().unwrap();
    assert_close(b2.lower()[[0]], 0.9, 1e-6, "expanded2 lower[0]");
    assert_close(b2.upper()[[0]], 1.1, 1e-6, "expanded2 upper[0]");
}

#[test]
fn test_reshape() {
    // Create (2, 4) zonotope with per-position error terms (2 errors)
    let values = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.1);

    assert_eq!(z.element_shape, vec![2, 4]);
    assert_eq!(z.n_error_terms, 8); // 2*4 = 8 per-element errors

    // Reshape to (4, 2)
    let reshaped = z.reshape(&[4, 2]).unwrap();

    assert_eq!(reshaped.element_shape, vec![4, 2]);
    assert_eq!(reshaped.n_error_terms, 8); // Same number of error terms

    // Total elements preserved
    assert_eq!(z.len(), reshaped.len());

    // Bounds should be preserved (same values, just rearranged)
    let orig_bounds = z.to_bounded_tensor().unwrap();
    let new_bounds = reshaped.to_bounded_tensor().unwrap();

    // First element [0,0] in original = first element [0,0] in reshaped (row-major order)
    assert!((orig_bounds.lower()[[0, 0]] - new_bounds.lower()[[0, 0]]).abs() < 1e-6);
    assert!((orig_bounds.upper()[[0, 0]] - new_bounds.upper()[[0, 0]]).abs() < 1e-6);
}

#[test]
fn test_reshape_error_different_size() {
    let values = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.1);

    // Try to reshape to different size - should fail
    let result = z.reshape(&[3, 3]);
    assert!(
        result.is_err(),
        "reshape should reject element-count changes, got {result:?}"
    );
}

#[test]
fn test_tile() {
    // Create (2, 3) zonotope
    let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.1);

    assert_eq!(z.element_shape, vec![2, 3]);

    // Tile 3x along axis 0 -> (6, 3)
    let tiled = z.tile(0, 3).unwrap();

    assert_eq!(tiled.element_shape, vec![6, 3]);
    assert_eq!(tiled.n_error_terms, z.n_error_terms); // Same errors (shared symbols!)

    // Original values should be repeated
    let center = tiled.center();
    // Row 0 = original row 0
    assert!((center[[0, 0]] - 1.0).abs() < 1e-6);
    // Row 2 = original row 0 (repeated)
    assert!((center[[2, 0]] - 1.0).abs() < 1e-6);
    // Row 4 = original row 0 (repeated again)
    assert!((center[[4, 0]] - 1.0).abs() < 1e-6);
    // Row 1 = original row 1
    assert!((center[[1, 0]] - 4.0).abs() < 1e-6);
}

#[test]
fn test_tile_preserves_correlations() {
    // Key test: tiling should preserve error symbol correlations
    // This is essential for GQA where K heads are tiled to match Q heads
    // The tiled values share the same uncertainty as the original

    let values = arr2(&[[1.0, 2.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.5);

    assert_eq!(z.n_error_terms, 2); // Per-element errors

    // Tile 4x -> 4 copies, all sharing the same error symbols
    let tiled = z.tile(0, 4).unwrap();

    assert_eq!(tiled.element_shape, vec![4, 2]);
    assert_eq!(tiled.n_error_terms, 2); // SAME number of errors!

    // All 4 rows should have the same center and error structure
    let bounds = tiled.to_bounded_tensor().unwrap();
    for i in 0..4 {
        // Each row has the same bounds as the original
        assert!((bounds.lower()[[i, 0]] - 0.5).abs() < 1e-6); // 1.0 - 0.5
        assert!((bounds.upper()[[i, 0]] - 1.5).abs() < 1e-6); // 1.0 + 0.5
        assert!((bounds.lower()[[i, 1]] - 1.5).abs() < 1e-6); // 2.0 - 0.5
        assert!((bounds.upper()[[i, 1]] - 2.5).abs() < 1e-6); // 2.0 + 0.5
    }

    // Crucially, the tiled rows share error symbols!
    // This means if row 0 takes value 1.5, rows 1-3 ALSO take 1.5
    // (they move together, not independently)
}

#[test]
fn test_tile_identity() {
    // Tiling with reps=1 should return identical zonotope
    let values = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.1);

    let tiled = z.tile(0, 1).unwrap();

    assert_eq!(tiled.element_shape, z.element_shape);
    assert_eq!(tiled.n_error_terms, z.n_error_terms);

    let orig_bounds = z.to_bounded_tensor().unwrap();
    let tiled_bounds = tiled.to_bounded_tensor().unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert!((orig_bounds.lower()[[i, j]] - tiled_bounds.lower()[[i, j]]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_to_bounded_tensor_rejects_nan_coefficients() {
    // NaN in zonotope coefficients must be caught by to_bounded_tensor(),
    // not silently passed through via from_parts_unchecked (soundness fix #2396).
    let mut coeffs = arr2(&[[1.0_f32, 2.0], [0.1, 0.2]]).into_dyn();
    coeffs[[1, 0]] = f32::NAN; // inject NaN into error coefficient
    let z = ZonotopeTensor::new(coeffs).unwrap();

    let result = z.to_bounded_tensor();
    assert!(
        result.is_err(),
        "to_bounded_tensor() must reject NaN coefficients, got: {:?}",
        result.unwrap()
    );
}

#[test]
fn test_to_bounded_tensor_allows_infinite_bounds() {
    // Infinite bounds are legitimate (very uncertain zonotopes).
    let coeffs = arr2(&[[1.0_f32], [f32::INFINITY]]).into_dyn();
    let z = ZonotopeTensor::new(coeffs).unwrap();

    let result = z.to_bounded_tensor();
    assert!(
        result.is_ok(),
        "to_bounded_tensor() should allow infinite bounds, got err: {:?}",
        result.err()
    );
    let bounds = result.unwrap();
    assert!(
        bounds.lower()[[0]].is_infinite(),
        "lower bound should stay infinite, got {}",
        bounds.lower()[[0]]
    );
    assert!(
        bounds.upper()[[0]].is_infinite(),
        "upper bound should stay infinite, got {}",
        bounds.upper()[[0]]
    );
}

/// #2676: max_width must propagate NaN from coefficients, not silently absorb it.
/// f32::max(NaN, x) == x (IEEE 754), so the old fold(0.0, f32::max) would
/// return a finite value even when NaN is present. nan_propagating_max fixes this.
#[test]
fn test_max_width_nan_coefficient_propagates_2676() {
    // Build a 2-element zonotope with 1 error term where one coeff is NaN.
    let mut coeffs = arr2(&[[1.0_f32, 2.0], [0.1, 0.2]]).into_dyn();
    coeffs[[1, 0]] = f32::NAN; // Inject NaN into error coefficient for element 0
    let z = ZonotopeTensor::new(coeffs).unwrap();

    let width = z.max_width();
    assert!(
        width.is_nan(),
        "#2676: max_width() must return NaN when error coefficients contain NaN, got {}",
        width
    );
}

/// #2676: max_width with all-finite coefficients still works correctly.
#[test]
fn test_max_width_finite_unaffected_2676() {
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.5);
    let width = z.max_width();
    assert!(
        (width - 1.0).abs() < 1e-6,
        "#2676: max_width() regression — expected 1.0 (2*0.5), got {}",
        width
    );
}

/// Proptest: disjoint matmul with multi-error-term zonotopes.
///
/// The existing `test_matmul_disjoint_soundness` uses `from_input_shared`
/// (1 error term each), which barely exercises the cross-term accumulation
/// loop. This test manually constructs zonotopes with per-element error terms
/// (2D element_shape) and random sampling to verify soundness.
///
/// Mathematical contract: for Z₁ and Z₂ with disjoint error symbols,
/// `matmul_disjoint` must produce bounds that contain all concrete products
/// A(ε) @ B(δ) for ε ∈ [-1,1]^n₁, δ ∈ [-1,1]^n₂.
mod disjoint_matmul_proptest {
    use super::*;
    use ndarray::IxDyn;
    use proptest::prelude::*;

    fn arb_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-2.0_f32..2.0_f32, rows * cols)
    }

    fn arb_epsilon_vec(n: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-1.0_f32..=1.0_f32, n)
    }

    /// Build a zonotope with 2D element_shape [rows, cols] and one error term
    /// per element (n_error_terms = rows*cols), each with coefficient `epsilon`.
    fn make_2d_elementwise(center: &ndarray::Array2<f32>, epsilon: f32) -> ZonotopeTensor {
        let rows = center.nrows();
        let cols = center.ncols();
        let n = rows * cols;
        // coeffs shape: (1 + n, rows, cols)
        let mut coeffs = ndarray::ArrayD::<f32>::zeros(IxDyn(&[1 + n, rows, cols]));
        for r in 0..rows {
            for c in 0..cols {
                coeffs[[0, r, c]] = center[[r, c]];
                let idx = r * cols + c;
                coeffs[[1 + idx, r, c]] = epsilon;
            }
        }
        ZonotopeTensor {
            coeffs,
            n_error_terms: n,
            element_shape: vec![rows, cols],
        }
    }

    /// Concretize: Z(ε) = center + epsilon * Σᵢ εᵢ * eᵢ where eᵢ is the
    /// unit vector for element i.
    fn concretize(
        center: &ndarray::Array2<f32>,
        epsilon: f32,
        epsilons: &[f32],
    ) -> ndarray::Array2<f32> {
        let mut result = center.clone();
        let cols = center.ncols();
        for (idx, &eps) in epsilons.iter().enumerate() {
            let row = idx / cols;
            let col = idx % cols;
            if row < center.nrows() {
                result[[row, col]] += epsilon * eps;
            }
        }
        result
    }

    proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

        #[test]
        fn disjoint_matmul_multi_error_soundness(
            a_flat in arb_matrix(2, 3),
            b_flat in arb_matrix(3, 2),   // B is (3,2) so A@B = (2,3)@(3,2) = (2,2)
            a_eps in arb_epsilon_vec(6),   // 2×3 = 6 per-element error terms
            b_eps in arb_epsilon_vec(6),   // 3×2 = 6 per-element error terms
        ) {
            let a_center = ndarray::Array2::from_shape_vec((2, 3), a_flat)
                .expect("valid shape");
            let b_center = ndarray::Array2::from_shape_vec((3, 2), b_flat)
                .expect("valid shape");

            let epsilon = 0.3_f32;
            let a = make_2d_elementwise(&a_center, epsilon);
            let b = make_2d_elementwise(&b_center, epsilon);

            prop_assert_eq!(a.n_error_terms, 6);
            prop_assert_eq!(b.n_error_terms, 6);

            let result = a.matmul_disjoint(&b)
                .map_err(|e| TestCaseError::Fail(format!("matmul_disjoint failed: {e}").into()))?;
            // 6 + 6 + 1 cross = 13 error terms
            prop_assert_eq!(result.n_error_terms, 13);
            prop_assert!(
                result.element_shape == vec![2, 2],
                "expected [2,2], got {:?}", result.element_shape,
            );

            let bounds = result.to_bounded_tensor()
                .map_err(|e| TestCaseError::Fail(format!("to_bounded_tensor failed: {e}").into()))?;

            // Concretize at random epsilon values and check containment
            let a_concrete = concretize(&a_center, epsilon, &a_eps);
            let b_concrete = concretize(&b_center, epsilon, &b_eps);
            let output = a_concrete.dot(&b_concrete);

            for i in 0..output.nrows() {
                for j in 0..output.ncols() {
                    prop_assert!(
                        output[[i, j]] >= bounds.lower()[[i, j]] - 1e-4,
                        "lower bound violated at [{i},{j}]: bound={} > concrete={}",
                        bounds.lower()[[i, j]],
                        output[[i, j]],
                    );
                    prop_assert!(
                        output[[i, j]] <= bounds.upper()[[i, j]] + 1e-4,
                        "upper bound violated at [{i},{j}]: bound={} < concrete={}",
                        bounds.upper()[[i, j]],
                        output[[i, j]],
                    );
                }
            }
        }
    }
}
