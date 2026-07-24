// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Compute the monotonicity gap for centroid bounds.
///
/// For each consecutive pair of query positions within each attention head,
/// gap(t) = centroid_upper(t-1) - centroid_lower(t). A non-positive gap means
/// monotonicity is certified; a positive gap is a potential violation.
pub(super) fn centroid_monotonicity_gaps(
    lower: &[f32],
    upper: &[f32],
    query_seq_len: usize,
) -> Vec<f32> {
    assert_eq!(
        lower.len(),
        upper.len(),
        "centroid lower/upper length mismatch"
    );
    assert!(
        query_seq_len >= 2 && lower.len().is_multiple_of(query_seq_len),
        "centroid rows={} should decompose into query_seq_len={query_seq_len}",
        lower.len()
    );

    let mut gaps = Vec::with_capacity(lower.len() - lower.len() / query_seq_len);
    for (lower_chunk, upper_chunk) in lower.chunks(query_seq_len).zip(upper.chunks(query_seq_len)) {
        for t in 1..query_seq_len {
            gaps.push(upper_chunk[t - 1] - lower_chunk[t]);
        }
    }
    gaps
}

/// Compute centroid interval bounds from softmax probability bounds.
///
/// centroid(t) = Σ_j j * A[t, j]
/// Because positions j ≥ 0, interval arithmetic is straightforward:
///   centroid_lower(t) = Σ_j j * probs_lower[t, j]
///   centroid_upper(t) = Σ_j j * probs_upper[t, j]
///
/// Reference: crates/ny-propagate/src/tests/attention_monotonicity.rs:53-86
fn compute_centroid_bounds(
    probs_lower: &[f32],
    probs_upper: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut lower = Vec::with_capacity(rows);
    let mut upper = Vec::with_capacity(rows);
    for t in 0..rows {
        let mut c_lo = 0.0f32;
        let mut c_hi = 0.0f32;
        for j in 0..cols {
            let w = j as f32;
            c_lo += w * probs_lower[t * cols + j];
            c_hi += w * probs_upper[t * cols + j];
        }
        lower.push(c_lo);
        upper.push(c_hi);
    }
    (lower, upper)
}

pub(super) fn centroid_bounds_from_softmax(
    softmax_bounds: &BoundedTensor,
    label: &str,
) -> (Vec<f32>, Vec<f32>, usize) {
    let lower_flat = softmax_bounds
        .lower()
        .as_slice()
        .expect("softmax lower bounds should be contiguous for centroid computation");
    let upper_flat = softmax_bounds
        .upper()
        .as_slice()
        .expect("softmax upper bounds should be contiguous for centroid computation");

    let softmax_shape = softmax_bounds.lower().shape();
    assert!(
        softmax_shape.len() >= 2,
        "{label}: softmax bounds should have at least query/key axes, got {:?}",
        softmax_shape
    );
    let query_seq_len = softmax_shape[softmax_shape.len() - 2];
    let key_seq_len = *softmax_shape
        .last()
        .expect("softmax bounds should have at least one dim");
    let total_elements = lower_flat.len();
    let num_rows = total_elements / key_seq_len;
    assert!(
        num_rows > 0 && total_elements.is_multiple_of(key_seq_len),
        "{label}: softmax total={total_elements} not divisible by key_seq_len={key_seq_len}"
    );
    assert!(
        num_rows.is_multiple_of(query_seq_len),
        "{label}: centroid rows={num_rows} should align with query_seq_len={query_seq_len}"
    );

    let (centroid_lower, centroid_upper) =
        compute_centroid_bounds(lower_flat, upper_flat, num_rows, key_seq_len);

    for (idx, (&lo, &hi)) in centroid_lower.iter().zip(centroid_upper.iter()).enumerate() {
        assert!(
            lo.is_finite(),
            "{label}: centroid lower[{idx}] not finite: {lo}"
        );
        assert!(
            hi.is_finite(),
            "{label}: centroid upper[{idx}] not finite: {hi}"
        );
        assert!(
            lo <= hi,
            "{label}: centroid inverted at row {idx}: {lo} > {hi}"
        );
    }

    (centroid_lower, centroid_upper, query_seq_len)
}

pub(super) fn assert_centroid_bounds_no_looser(
    label: &str,
    lower: &[f32],
    upper: &[f32],
    reference_lower: &[f32],
    reference_upper: &[f32],
    tol: f32,
) {
    assert_eq!(
        lower.len(),
        reference_lower.len(),
        "{label}: lower length mismatch: {} vs {}",
        lower.len(),
        reference_lower.len()
    );
    assert_eq!(
        upper.len(),
        reference_upper.len(),
        "{label}: upper length mismatch: {} vs {}",
        upper.len(),
        reference_upper.len()
    );

    for (idx, (((&lo, &hi), &ref_lo), &ref_hi)) in lower
        .iter()
        .zip(upper.iter())
        .zip(reference_lower.iter())
        .zip(reference_upper.iter())
        .enumerate()
    {
        assert!(
            lo >= ref_lo - tol,
            "{label}: lower[{idx}] is looser than reference: got {lo}, reference {ref_lo}"
        );
        assert!(
            hi <= ref_hi + tol,
            "{label}: upper[{idx}] is looser than reference: got {hi}, reference {ref_hi}"
        );
    }
}

// ---------------------------------------------------------------------------
// Shared centroid monotonicity statistics (#3497)
//
// Used by monotonicity.rs and crown_ibp_tightening.rs to compute and compare
// centroid-based monotonicity metrics from softmax probability bounds.
// ---------------------------------------------------------------------------

pub(super) struct CentroidMonotonicityStats {
    pub(super) centroid_lower: Vec<f32>,
    pub(super) centroid_upper: Vec<f32>,
    pub(super) query_seq_len: usize,
    pub(super) violations: usize,
    pub(super) max_gap: f32,
    pub(super) avg_width: f32,
}

pub(super) fn centroid_monotonicity_stats(
    softmax_bounds: &BoundedTensor,
    label: &str,
) -> CentroidMonotonicityStats {
    let (centroid_lower, centroid_upper, query_seq_len) =
        centroid_bounds_from_softmax(softmax_bounds, label);
    let gaps = centroid_monotonicity_gaps(&centroid_lower, &centroid_upper, query_seq_len);
    let violations = gaps.iter().filter(|&&g| g > 1e-4).count();
    let max_gap = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let avg_width = centroid_lower
        .iter()
        .zip(&centroid_upper)
        .map(|(lo, hi)| hi - lo)
        .sum::<f32>()
        / centroid_lower.len() as f32;
    CentroidMonotonicityStats {
        centroid_lower,
        centroid_upper,
        query_seq_len,
        violations,
        max_gap,
        avg_width,
    }
}
