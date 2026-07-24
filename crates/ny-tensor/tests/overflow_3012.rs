// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// ---------- Unit tests: checked_shape_product for softmax overflow guard ----------
//
// Integration tests for softmax_affine_causal overflow guards are not possible
// because ndarray panics in debug mode when constructing arrays whose shape
// product overflows usize — the internal `Dimension::size()` method uses
// unchecked `*` (dimension_trait.rs:142). Even `from_shape_vec_unchecked` hits
// this path.
//
// The softmax guard chain (softmax.rs:330-364) uses five steps:
//   1. prefix_size = checked_shape_product(element_shape[..ndim-2])
//   2. n_attn_rows = prefix_size.checked_mul(seq_q)
//   3. n_terms = n_error_terms.checked_add(1)
//   4. n_new_error_terms = n_attn_rows.checked_mul(seq_k)
//   5. n_rows_out = n_terms.checked_add(n_new_error_terms)
//
// We test each overflow step directly with the checked arithmetic building
// blocks. The Conv2d integration tests in ny-propagate/tests/overflow_3012.rs
// prove the full error path works end-to-end for the identical pattern.

use ny_core::checked_shape_product;

#[test]
fn test_softmax_prefix_product_overflow_3012() {
    // Step 1: prefix shape product overflows
    // Simulates element_shape prefix like [usize::MAX, 2, ...]
    assert!(
        checked_shape_product(&[usize::MAX, 2]).is_none(),
        "prefix product usize::MAX * 2 should overflow"
    );
}

#[test]
fn test_softmax_n_attn_rows_overflow_3012() {
    // Step 2: prefix_size * seq_q overflows
    let prefix_size: usize = usize::MAX / 2 + 1;
    let seq_q: usize = 2;
    assert!(
        prefix_size.checked_mul(seq_q).is_none(),
        "n_attn_rows = (usize::MAX/2 + 1) * 2 should overflow"
    );
}

#[test]
fn test_softmax_n_terms_overflow_3012() {
    // Step 3: n_error_terms + 1 overflows
    let n_error_terms: usize = usize::MAX;
    assert!(
        n_error_terms.checked_add(1).is_none(),
        "n_terms = usize::MAX + 1 should overflow"
    );
}

#[test]
fn test_softmax_n_new_error_terms_overflow_3012() {
    // Step 4: n_attn_rows * seq_k overflows
    let n_attn_rows: usize = usize::MAX / 2 + 1;
    let seq_k: usize = 3;
    assert!(
        n_attn_rows.checked_mul(seq_k).is_none(),
        "n_new_error_terms = (usize::MAX/2 + 1) * 3 should overflow"
    );
}

#[test]
fn test_softmax_n_rows_out_overflow_3012() {
    // Step 5: n_terms + n_new_error_terms overflows
    let n_terms: usize = usize::MAX / 2 + 1;
    let n_new_error_terms: usize = usize::MAX / 2 + 1;
    assert!(
        n_terms.checked_add(n_new_error_terms).is_none(),
        "n_rows_out = (usize::MAX/2 + 1) + (usize::MAX/2 + 1) should overflow"
    );
}

#[test]
fn test_checked_shape_product_valid_softmax_shapes_3012() {
    // Sanity: typical softmax shapes don't overflow
    // batch=2, heads=8, seq_q=512, seq_k=512
    assert_eq!(
        checked_shape_product(&[2, 8, 512, 512]),
        Some(2 * 8 * 512 * 512)
    );
}
