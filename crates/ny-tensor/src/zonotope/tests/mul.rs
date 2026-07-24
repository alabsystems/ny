// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr1, arr2};

#[test]
fn test_mul_elementwise_concrete() {
    // Test element-wise multiplication with concrete zonotopes
    let v1 = arr1(&[2.0_f32, 3.0]);
    let v2 = arr1(&[4.0_f32, 5.0]);
    let z1 = ZonotopeTensor::concrete(v1.into_dyn());
    let z2 = ZonotopeTensor::concrete(v2.into_dyn());

    let result = z1.mul_elementwise(&z2).unwrap();
    let center = result.center();

    // Expected: [2*4, 3*5] = [8, 15]
    assert!(
        (center[0] - 8.0).abs() < 1e-6,
        "center[0] should be 8.0, got {}",
        center[0]
    );
    assert!(
        (center[1] - 15.0).abs() < 1e-6,
        "center[1] should be 15.0, got {}",
        center[1]
    );
}

#[test]
fn test_mul_elementwise_with_shared_errors() {
    // Test that mul_elementwise exploits shared error symbols
    // z1 = 1 + 0.1*e1, z2 = 2 + 0.2*e1 (same error symbol)
    // z1*z2 = (1 + 0.1*e1)*(2 + 0.2*e1)
    //       = 2 + 0.2*e1 + 0.2*e1 + 0.02*e1²
    //       = 2 + 0.4*e1 + 0.02*e1²
    // e1² ∈ [0,1], so center shift = 0.5*0.02 = 0.01
    // Center = 2 + 0.01 = 2.01
    // Linear coeff = 0.4
    // Quadratic error = 0.5*|0.02| = 0.01

    // Create zonotopes with same error symbol
    let mut coeffs1 = ndarray::Array2::<f32>::zeros((2, 1));
    coeffs1[[0, 0]] = 1.0; // center
    coeffs1[[1, 0]] = 0.1; // error coeff

    let mut coeffs2 = ndarray::Array2::<f32>::zeros((2, 1));
    coeffs2[[0, 0]] = 2.0;
    coeffs2[[1, 0]] = 0.2;

    let z1 = ZonotopeTensor {
        coeffs: coeffs1.into_dyn(),
        n_error_terms: 1,
        element_shape: vec![1],
    };
    let z2 = ZonotopeTensor {
        coeffs: coeffs2.into_dyn(),
        n_error_terms: 1,
        element_shape: vec![1],
    };

    let result = z1.mul_elementwise(&z2).unwrap();

    // Check center (with quadratic shift)
    let center = result.center();
    assert!(
        (center[0] - 2.01).abs() < 1e-5,
        "Expected center 2.01, got {}",
        center[0]
    );

    // Check linear error coefficient
    // c1*a2 + c2*a1 = 1*0.2 + 2*0.1 = 0.4
    assert!(
        (result.coeffs[[1, 0]] - 0.4).abs() < 1e-5,
        "Expected linear coeff 0.4, got {}",
        result.coeffs[[1, 0]]
    );

    // Check quadratic error term
    // 0.5 * |0.1*0.2| = 0.01
    assert!(
        (result.coeffs[[2, 0]] - 0.01).abs() < 1e-5,
        "Expected quadratic error 0.01, got {}",
        result.coeffs[[2, 0]]
    );
}

#[test]
fn test_mul_elementwise_soundness() {
    // Test that zonotope multiplication produces SOUND (over-approximate) bounds
    //
    // Note: For a SINGLE shared error symbol with SAME-SIGN coefficients,
    // zonotope multiplication may be LOOSER than IBP because the e² term
    // adds conservative error. This is sound but not tight.
    //
    // Zonotope multiplication shines when:
    // 1. Coefficients have opposite signs (e_i can't be at both extremes simultaneously)
    // 2. There are many error symbols where cross-terms cancel
    //
    // For SwiGLU, the benefit comes from preserving correlations through the
    // entire FFN (Linear -> SiLU -> MulBinary -> Linear), where zonotope
    // tracks dependencies that IBP loses at each operation.

    // Test soundness: verify that true product values fall within zonotope bounds
    let mut coeffs1 = ndarray::Array2::<f32>::zeros((2, 1));
    coeffs1[[0, 0]] = 1.0; // center
    coeffs1[[1, 0]] = 0.5; // error coeff

    let mut coeffs2 = ndarray::Array2::<f32>::zeros((2, 1));
    coeffs2[[0, 0]] = 2.0;
    coeffs2[[1, 0]] = 0.3;

    let z1 = ZonotopeTensor {
        coeffs: coeffs1.into_dyn(),
        n_error_terms: 1,
        element_shape: vec![1],
    };
    let z2 = ZonotopeTensor {
        coeffs: coeffs2.into_dyn(),
        n_error_terms: 1,
        element_shape: vec![1],
    };

    let zono_result = z1.mul_elementwise(&z2).unwrap();
    let zono_bounds = zono_result.to_bounded_tensor().unwrap();

    // Check soundness: true product at various e values should be within bounds
    for e in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
        let x1 = 1.0 + 0.5 * e;
        let x2 = 2.0 + 0.3 * e;
        let product = x1 * x2;
        assert!(
            zono_bounds.lower()[0] <= product && product <= zono_bounds.upper()[0],
            "Product {} at e={} not in zonotope bounds [{}, {}]",
            product,
            e,
            zono_bounds.lower()[0],
            zono_bounds.upper()[0]
        );
    }

    // Verify zonotope bounds are finite and reasonable
    assert!(
        zono_bounds.lower()[0].is_finite(),
        "lower bound should be finite, got {}",
        zono_bounds.lower()[0]
    );
    assert!(
        zono_bounds.upper()[0].is_finite(),
        "upper bound should be finite, got {}",
        zono_bounds.upper()[0]
    );
    assert!(
        zono_bounds.lower()[0] < zono_bounds.upper()[0],
        "lower {} should be < upper {}",
        zono_bounds.lower()[0],
        zono_bounds.upper()[0]
    );
}

#[test]
fn test_mul_elementwise_opposite_signs_tighter() {
    // Test that zonotope multiplication IS tighter than IBP when
    // coefficients have opposite signs (anti-correlated)
    //
    // z1 = 1 + 0.5*e, z2 = 2 - 0.5*e (opposite sign coefficients)
    // When e=1: z1=1.5, z2=1.5, product=2.25
    // When e=-1: z1=0.5, z2=2.5, product=1.25
    // When e=0: z1=1.0, z2=2.0, product=2.0
    //
    // IBP: z1 ∈ [0.5, 1.5], z2 ∈ [1.5, 2.5]
    // IBP corners: [0.75, 1.25, 2.25, 3.75], range [0.75, 3.75], width 3.0
    //
    // But true range is [1.25, 2.25] (anti-correlation constrains extremes)
    // Zonotope should capture some of this anti-correlation benefit

    let mut coeffs1 = ndarray::Array2::<f32>::zeros((2, 1));
    coeffs1[[0, 0]] = 1.0;
    coeffs1[[1, 0]] = 0.5; // positive coefficient

    let mut coeffs2 = ndarray::Array2::<f32>::zeros((2, 1));
    coeffs2[[0, 0]] = 2.0;
    coeffs2[[1, 0]] = -0.5; // negative coefficient (anti-correlated)

    let z1 = ZonotopeTensor {
        coeffs: coeffs1.into_dyn(),
        n_error_terms: 1,
        element_shape: vec![1],
    };
    let z2 = ZonotopeTensor {
        coeffs: coeffs2.into_dyn(),
        n_error_terms: 1,
        element_shape: vec![1],
    };

    let zono_result = z1.mul_elementwise(&z2).unwrap();
    let zono_bounds = zono_result.to_bounded_tensor().unwrap();
    let zono_width = zono_bounds.upper()[0] - zono_bounds.lower()[0];

    // IBP width
    let ibp_width = 3.75 - 0.75; // = 3.0

    // Zonotope should be much tighter due to anti-correlation
    // The linear terms c1*a2 + c2*a1 = 1*(-0.5) + 2*0.5 = 0.5
    // Combined with quadratic terms, should give tighter bounds
    assert!(
        zono_width < ibp_width,
        "Zonotope width {} should be < IBP width {} for anti-correlated inputs",
        zono_width,
        ibp_width
    );
}

#[test]
fn test_mul_elementwise_2d() {
    // Test 2D element-wise multiplication (for transformer sequence data)
    let v1 = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    let v2 = arr2(&[[5.0_f32, 6.0], [7.0, 8.0]]);
    let z1 = ZonotopeTensor::concrete(v1.into_dyn());
    let z2 = ZonotopeTensor::concrete(v2.into_dyn());

    let result = z1.mul_elementwise(&z2).unwrap();
    let center = result.center();

    // Expected: [[1*5, 2*6], [3*7, 4*8]] = [[5, 12], [21, 32]]
    let expected = [[5.0, 12.0], [21.0, 32.0]];
    for r in 0..2 {
        for c in 0..2 {
            assert!(
                (center[[r, c]] - expected[r][c]).abs() < 1e-6,
                "center[{r},{c}] should be {}, got {}",
                expected[r][c],
                center[[r, c]]
            );
        }
    }
}
