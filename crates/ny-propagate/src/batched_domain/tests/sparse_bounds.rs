// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{array, ArrayD, IxDyn};
use ny_tensor::PooledArray;
use std::collections::HashMap;

#[ntest::timeout(10000)]
#[test]
fn test_batched_domains_sparse_to_dense_indices() {
    let mut builder = BatchedDomainsBuilder::new_with_options(
        vec!["relu0".to_string()],
        BatchedDomainOptions {
            enable_interm_transfer: true,
        },
    );

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            array![-1.0, -0.2, -0.1, -0.3].into_dyn(),
            array![1.0, -0.1, 0.2, 0.0].into_dyn(),
        ),
    );

    builder.add_domain(
        &layer_bounds,
        array![0.0].into_dyn(),
        array![1.0].into_dyn(),
        -1.0,
        1.0,
        0,
        vec![],
    );

    let batched = builder.build().unwrap();
    let sparse = batched.sparse_to_dense_indices("relu0").unwrap();
    assert_eq!(sparse, vec![0, 2]);
    assert_eq!(batched.unstable_count("relu0"), Some(2));
    assert_eq!(batched.is_neuron_unstable("relu0", 0), Some(true));
    assert_eq!(batched.is_neuron_unstable("relu0", 1), Some(false));
    assert_eq!(batched.is_neuron_unstable("relu0", 2), Some(true));
    assert_eq!(batched.is_neuron_unstable("relu0", 3), Some(false));
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_domains_sparse_to_dense_indices_multi_dim() {
    let mut builder = BatchedDomainsBuilder::new_with_options(
        vec!["relu0".to_string()],
        BatchedDomainOptions {
            enable_interm_transfer: true,
        },
    );

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            array![[-1.0, 1.0], [-0.5, -0.1]].into_dyn(),
            array![[1.0, 2.0], [-0.2, 0.2]].into_dyn(),
        ),
    );

    builder.add_domain(
        &layer_bounds,
        array![0.0].into_dyn(),
        array![1.0].into_dyn(),
        -1.0,
        1.0,
        0,
        vec![],
    );

    let batched = builder.build().unwrap();
    let sparse = batched.sparse_to_dense_indices("relu0").unwrap();
    assert_eq!(sparse, vec![0, 3]);
    assert_eq!(batched.unstable_count("relu0"), Some(2));
    assert_eq!(batched.is_neuron_unstable("relu0", 0), Some(true));
    assert_eq!(batched.is_neuron_unstable("relu0", 1), Some(false));
    assert_eq!(batched.is_neuron_unstable("relu0", 2), Some(false));
    assert_eq!(batched.is_neuron_unstable("relu0", 3), Some(true));
}

#[ntest::timeout(10000)]
#[test]
fn test_sparse_bounds_from_batched_domains() {
    let mut builder = BatchedDomainsBuilder::new_with_options(
        vec!["relu0".to_string()],
        BatchedDomainOptions {
            enable_interm_transfer: true,
        },
    );

    let mut layer_bounds0 = HashMap::new();
    layer_bounds0.insert(
        "relu0".to_string(),
        (
            array![-1.0, -0.2, -0.1].into_dyn(),
            array![1.0, -0.1, 0.2].into_dyn(),
        ),
    );
    builder.add_domain(
        &layer_bounds0,
        array![0.0].into_dyn(),
        array![1.0].into_dyn(),
        -1.0,
        1.0,
        0,
        vec![],
    );

    let mut layer_bounds1 = HashMap::new();
    layer_bounds1.insert(
        "relu0".to_string(),
        (
            array![-2.0, -0.5, -0.4].into_dyn(),
            array![2.0, -0.2, 0.3].into_dyn(),
        ),
    );
    builder.add_domain(
        &layer_bounds1,
        array![0.1].into_dyn(),
        array![0.9].into_dyn(),
        -0.5,
        0.5,
        1,
        vec![],
    );

    let mut batched = builder.build().unwrap();
    let sparse = SparseIntermediateBounds::from_batched_domains(&batched)
        .expect("sparse bounds should be available when unstable masks exist");

    let (sparse_lower, sparse_upper) = sparse.layer_bounds("relu0").unwrap();
    assert_eq!(sparse_lower.shape(), &[2, 2]);
    assert_eq!(sparse_upper.shape(), &[2, 2]);
    assert_eq!(sparse_lower[[0, 0]], -1.0);
    assert_eq!(sparse_lower[[0, 1]], -0.1);
    assert_eq!(sparse_upper[[0, 0]], 1.0);
    assert_eq!(sparse_upper[[0, 1]], 0.2);
    assert_eq!(sparse_lower[[1, 0]], -2.0);
    assert_eq!(sparse_lower[[1, 1]], -0.4);
    assert_eq!(sparse_upper[[1, 0]], 2.0);
    assert_eq!(sparse_upper[[1, 1]], 0.3);

    let updated = sparse.merge_into(&mut batched).unwrap();
    assert_eq!(updated, 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_sparse_bounds_missing_masks_returns_none() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (array![-1.0, 0.0].into_dyn(), array![1.0, 2.0].into_dyn()),
    );

    builder.add_domain(
        &layer_bounds,
        array![0.0].into_dyn(),
        array![1.0].into_dyn(),
        -1.0,
        1.0,
        0,
        vec![],
    );

    let batched = builder.build().unwrap();
    assert!(SparseIntermediateBounds::from_batched_domains(&batched).is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_sparse_bounds_merge_into_tightens_bounds() {
    let mut builder = BatchedDomainsBuilder::new_with_options(
        vec!["relu0".to_string()],
        BatchedDomainOptions {
            enable_interm_transfer: true,
        },
    );

    // Initial bounds: neurons 0 and 2 are unstable
    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            array![-1.0, -0.2, -0.1].into_dyn(), // lb for neurons 0, 1, 2
            array![1.0, -0.1, 0.2].into_dyn(),   // ub for neurons 0, 1, 2
        ),
    );
    builder.add_domain(
        &layer_bounds,
        array![0.0].into_dyn(),
        array![1.0].into_dyn(),
        -1.0,
        1.0,
        0,
        vec![],
    );

    let mut batched = builder.build().unwrap();

    // Create sparse bounds with tighter values for unstable neurons (0 and 2)
    let sparse = SparseIntermediateBounds::from_batched_domains(&batched)
        .expect("sparse bounds should be available");

    // Modify batched to have looser bounds, then merge tighter sparse bounds back
    // The full_lower for unstable neuron 0 is at index 0, neuron 2 is at index 2
    if let Some(full_lower) = batched.layer_lowers_mut().get_mut("relu0") {
        let arr = full_lower.as_array_mut();
        // Make neuron 0's lower bound looser: -1.0 -> -2.0
        arr[[0, 0]] = -2.0;
        // Make neuron 2's lower bound looser: -0.1 -> -0.5
        arr[[0, 2]] = -0.5;
    }
    if let Some(full_upper) = batched.layer_uppers_mut().get_mut("relu0") {
        let arr = full_upper.as_array_mut();
        // Make neuron 0's upper bound looser: 1.0 -> 2.0
        arr[[0, 0]] = 2.0;
        // Make neuron 2's upper bound looser: 0.2 -> 0.5
        arr[[0, 2]] = 0.5;
    }

    // Now merge the original tighter sparse bounds back
    let updated = sparse.merge_into(&mut batched).unwrap();

    // Should have updated 4 values: lower and upper for neurons 0 and 2
    assert_eq!(updated, 4);

    // Verify the bounds are now tighter (original values restored)
    let final_lower = batched.layer_lowers().get("relu0").unwrap().as_array();
    let final_upper = batched.layer_uppers().get("relu0").unwrap().as_array();
    assert_eq!(final_lower[[0, 0]], -1.0); // Was -2.0, now -1.0 (tighter)
    assert_eq!(final_lower[[0, 2]], -0.1); // Was -0.5, now -0.1 (tighter)
    assert_eq!(final_upper[[0, 0]], 1.0); // Was 2.0, now 1.0 (tighter)
    assert_eq!(final_upper[[0, 2]], 0.2); // Was 0.5, now 0.2 (tighter)
}

#[ntest::timeout(10000)]
#[test]
fn test_sparse_bounds_merge_into_error_without_masks() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (array![-1.0, 0.0].into_dyn(), array![1.0, 2.0].into_dyn()),
    );

    builder.add_domain(
        &layer_bounds,
        array![0.0].into_dyn(),
        array![1.0].into_dyn(),
        -1.0,
        1.0,
        0,
        vec![],
    );

    let mut batched = builder.build().unwrap();

    // Create empty sparse bounds
    let sparse = SparseIntermediateBounds::new();

    // merge_into should return an error because unstable_masks is not populated
    let result = sparse.merge_into(&mut batched);
    assert!(result.is_err());
}

/// Regression test for #2084/#2091: merge_into must not silently skip the
/// bounds merge when layer bounds are non-contiguous.
///
/// The original bug was that `as_slice_mut()` returned `None` for a
/// non-contiguous (e.g. transposed) layer-bounds array, which silently dropped
/// the entire tightening pass. The current production path normalizes the
/// layout in-place via `contiguous_flat_slice_mut` and then merges correctly,
/// so the merge succeeds (returning `Ok`) and the previously non-contiguous
/// array is left in standard (C-contiguous) layout.
#[ntest::timeout(10000)]
#[test]
fn test_sparse_bounds_merge_into_noncontiguous_normalizes_layout() {
    let mut builder = BatchedDomainsBuilder::new_with_options(
        vec!["relu0".to_string()],
        BatchedDomainOptions {
            enable_interm_transfer: true,
        },
    );

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            array![-1.0, -0.2, -0.1].into_dyn(),
            array![1.0, -0.1, 0.2].into_dyn(),
        ),
    );
    builder.add_domain(
        &layer_bounds,
        array![0.0].into_dyn(),
        array![1.0].into_dyn(),
        -1.0,
        1.0,
        0,
        vec![],
    );

    let mut batched = builder.build().unwrap();
    let sparse = SparseIntermediateBounds::from_batched_domains(&batched)
        .expect("sparse bounds should be available");

    // Replace lower bounds with a non-standard-layout array.
    // Create a [2, 3] array and reverse axes to get [3, 2] with non-C-contiguous strides.
    // Then the as_slice_mut() guard returns None, triggering InternalError.
    let base =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, -0.2, -0.1, -2.0, -0.5, -0.4]).unwrap();
    let noncontiguous = base.reversed_axes(); // shape [3, 2] with column-major strides
    assert!(
        !noncontiguous.is_standard_layout(),
        "reversed_axes on [2,3] must produce non-standard layout"
    );
    batched
        .layer_lowers_mut()
        .insert("relu0".to_string(), PooledArray::from_array(noncontiguous));

    // merge_into must NOT silently skip the merge: it normalizes the
    // non-contiguous array in place and returns Ok rather than dropping the
    // tightening pass (the original #2084/#2091 bug).
    sparse
        .merge_into(&mut batched)
        .expect("merge_into should normalize non-contiguous bounds and succeed");

    // After the merge the layer-bounds array must be back in standard layout,
    // confirming the in-place normalization happened (rather than the merge
    // being silently skipped on a non-contiguous buffer).
    assert!(
        batched
            .layer_lowers_mut()
            .get("relu0")
            .expect("relu0 bounds must be present")
            .as_array()
            .is_standard_layout(),
        "merge_into must normalize non-contiguous bounds to standard layout"
    );
}
