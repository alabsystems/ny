// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the 3 input-validation error returns in `softmax_affine_causal`:
//! ndim < 2, axis != last, seq_q > seq_k.
//! Coverage gap identified in #2519.

use super::super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;

/// Minimal zonotope that only needs valid element_shape metadata.
/// Coeffs shape doesn't match element_shape — validation fires before coeff access.
fn metadata_only_zonotope(element_shape: &[usize]) -> ZonotopeTensor {
    ZonotopeTensor {
        coeffs: ArrayD::zeros(IxDyn(&[1, 1, 1, 2])),
        n_error_terms: 0,
        element_shape: element_shape.to_vec(),
    }
}

#[test]
fn test_softmax_affine_causal_rejects_1d_input_2519() {
    // element_shape = [4] → ndim = 1 < 2 → error
    let z = metadata_only_zonotope(&[4]);
    let err = z
        .softmax_affine_causal(-1)
        .expect_err("1D input should fail: requires at least 2 dimensions");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("at least 2 dimensions")),
        "expected ndim<2 error, got: {err:?}"
    );
}

#[test]
fn test_softmax_affine_causal_rejects_non_last_axis_2519() {
    // element_shape = [3, 4] → ndim = 2, axis=0 → axis_usize=0 != ndim-1=1 → error
    let z = metadata_only_zonotope(&[3, 4]);
    let err = z
        .softmax_affine_causal(0)
        .expect_err("non-last axis should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("only supports last-axis")),
        "expected axis!=last error, got: {err:?}"
    );
}

#[test]
fn test_softmax_affine_causal_rejects_seq_q_gt_seq_k_2519() {
    // element_shape = [5, 3] → seq_q=5, seq_k=3, seq_q > seq_k → error
    let z = metadata_only_zonotope(&[5, 3]);
    let err = z
        .softmax_affine_causal(-1)
        .expect_err("seq_q > seq_k should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("seq_q (5) <= seq_k (3)")),
        "expected seq_q>seq_k error, got: {err:?}"
    );
}

#[test]
fn test_softmax_affine_causal_accepts_square_causal_2519() {
    // element_shape = [3, 3] → seq_q=3 == seq_k=3 → should NOT hit seq_q>seq_k error.
    // May fail later (e.g., coeff reshape) but must not fail on validation guards.
    let z = metadata_only_zonotope(&[3, 3]);
    let result = z.softmax_affine_causal(-1);
    // If it errors, it should NOT be one of the 3 validation errors.
    if let Err(ref e) = result {
        let msg = format!("{e:?}");
        assert!(
            !msg.contains("at least 2 dimensions"),
            "should not hit ndim guard for 2D input"
        );
        assert!(
            !msg.contains("only supports last-axis"),
            "should not hit axis guard for axis=-1"
        );
        assert!(
            !msg.contains("seq_q"),
            "should not hit seq_q guard for square input"
        );
    }
}
