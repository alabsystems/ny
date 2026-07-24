// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;

fn metadata_only_zonotope(element_shape: &[usize]) -> ZonotopeTensor {
    ZonotopeTensor {
        coeffs: ArrayD::zeros(IxDyn(&[1, 1, 1, 2])),
        n_error_terms: 0,
        element_shape: element_shape.to_vec(),
    }
}

#[test]
fn test_softmax_affine_causal_prefix_product_overflow_returns_error_3012() {
    let zonotope = metadata_only_zonotope(&[usize::MAX, 2, 1, 2]);

    let err = zonotope
        .softmax_affine_causal(-1)
        .expect_err("overflowing prefix product should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("prefix shape product overflows")),
        "expected prefix-product overflow error, got: {err:?}"
    );
}

#[test]
fn test_softmax_affine_causal_n_attn_rows_overflow_returns_error_3012() {
    let zonotope = metadata_only_zonotope(&[(usize::MAX / 2) + 1, 2, 2]);

    let err = zonotope
        .softmax_affine_causal(-1)
        .expect_err("overflowing attention-row count should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("n_attn_rows overflows")),
        "expected n_attn_rows overflow error, got: {err:?}"
    );
}

#[test]
fn test_softmax_affine_causal_n_new_error_terms_overflow_returns_error_3012() {
    let zonotope = metadata_only_zonotope(&[(usize::MAX / 2) + 1, 1, 3]);

    let err = zonotope
        .softmax_affine_causal(-1)
        .expect_err("overflowing new-error-term count should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("n_new_error_terms overflows")),
        "expected n_new_error_terms overflow error, got: {err:?}"
    );
}

#[test]
fn test_softmax_affine_causal_n_rows_out_overflow_returns_error_3012() {
    let zonotope = metadata_only_zonotope(&[usize::MAX, 1, 1]);

    let err = zonotope
        .softmax_affine_causal(-1)
        .expect_err("overflowing output-row count should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("n_rows_out overflows")),
        "expected n_rows_out overflow error, got: {err:?}"
    );
}

#[test]
fn test_softmax_affine_causal_metadata_only_fixture_hits_guard_before_coeff_access_3012() {
    let zonotope = metadata_only_zonotope(&[usize::MAX, 1, 2]);

    let err = zonotope
        .softmax_affine_causal(-1)
        .expect_err("metadata-only zonotope should fail on overflow before reshape");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("n_new_error_terms overflows")),
        "expected overflow guard before coeff reshape, got: {err:?}"
    );
}
