// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

#[test]
fn test_from_input_per_position_rejects_1d() {
    // Kills: replace < with == in line 210 (shape.len() < 2)
    // Kills: replace < with > in line 210
    // Kills: replace < with <= in line 210
    let values_1d = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let result = ZonotopeTensor::from_input_per_position(&values_1d, 0.1);
    assert!(
        result.is_err(),
        "1D input should fail (requires >= 2 dimensions)"
    );
}

#[test]
fn test_from_input_per_position_rejects_4d() {
    let values_4d = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4, 5]));
    let result = ZonotopeTensor::from_input_per_position(&values_4d, 0.1);
    assert!(result.is_err(), "4D input should be rejected");
}

#[test]
fn test_from_input_per_position_accepts_2d() {
    // Verify 2D input works - complements the rejection test
    let values_2d = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(); // shape [2, 2]
    let result = ZonotopeTensor::from_input_per_position(&values_2d, 0.1);
    assert!(result.is_ok(), "2D input should succeed");

    let z = result.unwrap();
    // seq_len = shape[-2] = 2, so n_error_terms = 2
    assert_eq!(z.n_error_terms, 2, "2D: n_error_terms should be seq_len");
}

#[test]
fn test_from_input_per_position_seq_len_calculation() {
    // Kills: replace - with + in line 216 (shape[shape.len() - 2])
    // Kills: replace - with / in line 216
    // For 2D [seq_len=3, embed=4], seq_len should be 3
    let values = ArrayD::<f32>::zeros(IxDyn(&[3, 4])); // seq=3, embed=4
    let z = ZonotopeTensor::from_input_per_position(&values, 0.1).unwrap();

    // If mutation changed - to +, would try shape[2+2] = shape[4] -> panic or wrong
    // If mutation changed - to /, would try shape[2/2] = shape[1] = 4 (wrong)
    assert_eq!(
        z.n_error_terms, 3,
        "seq_len should be 3 (second-to-last dim)"
    );
}

#[test]
fn test_from_input_per_position_2d_error_assignment() {
    // Kills: delete match arm 2 in line 220/239
    // Kills: replace + with - in line 243 (1 + pos)
    // Kills: replace + with * in line 243
    let values = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn(); // [3, 2]
    let z = ZonotopeTensor::from_input_per_position(&values, 0.5).unwrap();

    // For 2D, coeffs[1+pos, pos, :] = epsilon
    // coeffs shape: [1+3, 3, 2] = [4, 3, 2]
    assert_eq!(z.coeffs.shape(), &[4, 3, 2]);

    // Check error terms are at correct positions
    // Error term 0 (index 1) affects position 0
    assert!(
        (z.coeffs[[1, 0, 0]] - 0.5).abs() < 1e-6,
        "err0 should affect pos0"
    );
    assert!(
        (z.coeffs[[1, 0, 1]] - 0.5).abs() < 1e-6,
        "err0 should affect pos0"
    );
    assert_eq!(z.coeffs[[1, 1, 0]], 0.0, "err0 should NOT affect pos1");

    // Error term 1 (index 2) affects position 1
    assert_eq!(z.coeffs[[2, 0, 0]], 0.0, "err1 should NOT affect pos0");
    assert!(
        (z.coeffs[[2, 1, 0]] - 0.5).abs() < 1e-6,
        "err1 should affect pos1"
    );

    // Error term 2 (index 3) affects position 2
    assert_eq!(z.coeffs[[3, 1, 0]], 0.0, "err2 should NOT affect pos1");
    assert!(
        (z.coeffs[[3, 2, 0]] - 0.5).abs() < 1e-6,
        "err2 should affect pos2"
    );
}

#[test]
fn test_from_input_per_position_3d_n_error_terms() {
    // Kills: delete match arm 3 in lines 221, 247
    // Kills: replace * with + in line 221 (shape[0] * seq_len)
    // Kills: replace * with / in line 221
    // For 3D [batch=2, seq=3, embed=4], n_error_terms = batch * seq = 6
    let values = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4]));
    let z = ZonotopeTensor::from_input_per_position(&values, 0.1).unwrap();

    // If * was +, would get 2 + 3 = 5 (wrong)
    // If * was /, would get 2 / 3 = 0 (wrong)
    assert_eq!(
        z.n_error_terms, 6,
        "3D: n_error_terms should be batch * seq = 6"
    );
}

#[test]
fn test_from_input_per_position_3d_error_assignment() {
    // Kills mutations in line 253: 1 + b * seq_len + pos
    // Tests: replace + with -, replace * with +, etc.
    let values = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4])); // batch=2, seq=3, embed=4
    let z = ZonotopeTensor::from_input_per_position(&values, 0.5).unwrap();

    // coeffs shape: [1+6, 2, 3, 4] = [7, 2, 3, 4]
    assert_eq!(z.coeffs.shape(), &[7, 2, 3, 4]);

    // For (b=0, pos=0): err index = 1 + 0*3 + 0 = 1
    assert!(
        (z.coeffs[[1, 0, 0, 0]] - 0.5).abs() < 1e-6,
        "b0p0 should use err1"
    );
    assert_eq!(z.coeffs[[1, 0, 1, 0]], 0.0, "err1 should NOT affect b0p1");
    assert_eq!(z.coeffs[[1, 1, 0, 0]], 0.0, "err1 should NOT affect b1p0");

    // For (b=0, pos=2): err index = 1 + 0*3 + 2 = 3
    assert!(
        (z.coeffs[[3, 0, 2, 0]] - 0.5).abs() < 1e-6,
        "b0p2 should use err3"
    );

    // For (b=1, pos=0): err index = 1 + 1*3 + 0 = 4
    assert!(
        (z.coeffs[[4, 1, 0, 0]] - 0.5).abs() < 1e-6,
        "b1p0 should use err4"
    );
    assert_eq!(z.coeffs[[4, 0, 0, 0]], 0.0, "err4 should NOT affect b0p0");

    // For (b=1, pos=2): err index = 1 + 1*3 + 2 = 6
    assert!(
        (z.coeffs[[6, 1, 2, 0]] - 0.5).abs() < 1e-6,
        "b1p2 should use err6"
    );
}
