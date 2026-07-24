// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::utils::{slice_batch_dim, stack_arrays_pooled};
use super::{super::*, Counterexample, DomainResult};
use crate::beta_crown::GraphBabDomain;
use ndarray::{array, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::PooledArray;
use std::collections::HashMap;

// ============================================================================
// Input bounds access tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_batched_domains_input_bounds_at() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (array![-1.0, 0.0].into_dyn(), array![1.0, 2.0].into_dyn()),
    );

    builder.add_domain(
        &layer_bounds,
        array![0.0, 0.5].into_dyn(),
        array![1.0, 1.5].into_dyn(),
        -1.0,
        0.5,
        0,
        vec![],
    );
    builder.add_domain(
        &layer_bounds,
        array![2.0, 2.5].into_dyn(),
        array![3.0, 3.5].into_dyn(),
        -0.5,
        1.0,
        1,
        vec![],
    );

    let batched = builder.build().unwrap();

    let first = batched.input_bounds_at(0).unwrap();
    assert_eq!(first.lower(), array![0.0, 0.5].into_dyn());
    assert_eq!(first.upper(), array![1.0, 1.5].into_dyn());

    let second = batched.input_bounds_at(1).unwrap();
    assert_eq!(second.lower(), array![2.0, 2.5].into_dyn());
    assert_eq!(second.upper(), array![3.0, 3.5].into_dyn());
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_domains_input_bounds_at_out_of_range() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (array![-1.0, 0.0].into_dyn(), array![1.0, 2.0].into_dyn()),
    );

    builder.add_domain(
        &layer_bounds,
        array![0.0, 0.5].into_dyn(),
        array![1.0, 1.5].into_dyn(),
        -1.0,
        0.5,
        0,
        vec![],
    );

    let batched = builder.build().unwrap();
    let err = batched.input_bounds_at(1).unwrap_err();
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_domains_input_bounds_at_batch_mismatch() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (array![-1.0, 0.0].into_dyn(), array![1.0, 2.0].into_dyn()),
    );

    for i in 0..2 {
        builder.add_domain(
            &layer_bounds,
            array![i as f32, i as f32 + 0.5].into_dyn(),
            array![i as f32 + 1.0, i as f32 + 1.5].into_dyn(),
            -1.0 + i as f32,
            0.5 + i as f32,
            i,
            vec![],
        );
    }

    let mut batched = builder.build().unwrap();
    batched.input_lowers = PooledArray::from_array(array![0.0].into_dyn());

    let err = batched.input_bounds_at(0).unwrap_err();
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

// ============================================================================
// Static bounds and unstable mask tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_batched_domains_static_bounds_none_when_disabled() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (array![-1.0, 0.0].into_dyn(), array![1.0, 2.0].into_dyn()),
    );

    builder.add_domain(
        &layer_bounds,
        array![0.0, 0.5].into_dyn(),
        array![1.0, 1.5].into_dyn(),
        -1.0,
        0.5,
        0,
        vec![],
    );

    let batched = builder.build().unwrap();
    assert!(batched.static_layer_lowers().is_none());
    assert!(batched.static_layer_uppers().is_none());
    assert!(batched.unstable_masks().is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_domains_interm_transfer_static_bounds() {
    let mut builder = BatchedDomainsBuilder::new_with_options(
        vec!["relu0".to_string()],
        BatchedDomainOptions {
            enable_interm_transfer: true,
        },
    );

    // Domain 0
    let mut layer_bounds0 = HashMap::new();
    layer_bounds0.insert(
        "relu0".to_string(),
        (array![-1.0, 0.2].into_dyn(), array![1.0, 0.8].into_dyn()),
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

    // Domain 1
    let mut layer_bounds1 = HashMap::new();
    layer_bounds1.insert(
        "relu0".to_string(),
        (array![-0.5, -0.1].into_dyn(), array![0.5, 0.1].into_dyn()),
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

    let batched = builder.build().unwrap();
    let static_lowers = batched.static_layer_lowers().unwrap();
    let static_uppers = batched.static_layer_uppers().unwrap();

    let relu0_lower = static_lowers.get("relu0").unwrap().as_array();
    let relu0_upper = static_uppers.get("relu0").unwrap().as_array();
    assert_eq!(relu0_lower.shape(), &[2]);
    assert_eq!(relu0_upper.shape(), &[2]);
    assert_eq!(relu0_lower.as_slice().unwrap(), &[-1.0, 0.2]);
    assert_eq!(relu0_upper.as_slice().unwrap(), &[1.0, 0.8]);

    let mask = batched.unstable_masks().unwrap();
    let relu0_mask = mask.get("relu0").unwrap();
    assert_eq!(relu0_mask.as_slice().unwrap(), &[true, false]);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_domains_unstable_mask_strict_boundary() {
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
            array![-1.0, 0.0, -0.1].into_dyn(),
            array![0.0, 1.0, 0.2].into_dyn(),
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
    let mask = batched.unstable_masks().unwrap();
    let relu0_mask = mask.get("relu0").unwrap();
    assert_eq!(relu0_mask.as_slice().unwrap(), &[false, false, true]);
}

// ============================================================================
// Utility function tests (stack_arrays, slice_batch_dim)
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_stack_arrays() {
    let arrays = vec![
        array![1.0, 2.0, 3.0].into_dyn(),
        array![4.0, 5.0, 6.0].into_dyn(),
    ];

    let stacked = stack_arrays_pooled(&arrays).unwrap();

    let stacked = stacked.as_array();
    assert_eq!(stacked.shape(), &[2, 3]);
    assert_eq!(stacked[[0, 0]], 1.0);
    assert_eq!(stacked[[0, 2]], 3.0);
    assert_eq!(stacked[[1, 0]], 4.0);
    assert_eq!(stacked[[1, 2]], 6.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_stack_arrays_empty() {
    let arrays: Vec<ArrayD<f32>> = vec![];
    assert!(stack_arrays_pooled(&arrays).is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_domain_result_enum() {
    // Verify enum variant discrimination works correctly
    let verified = DomainResult::Verified;
    let falsified = DomainResult::Falsified(Box::new(Counterexample {
        input: array![0.5].into_dyn(),
        output: array![1.0].into_dyn(),
    }));
    let cont = DomainResult::Continue;

    assert!(matches!(verified, DomainResult::Verified));
    match falsified {
        DomainResult::Falsified(counterexample) => {
            assert_eq!(counterexample.input, array![0.5].into_dyn());
            assert_eq!(counterexample.output, array![1.0].into_dyn());
        }
        _ => panic!("Expected DomainResult::Falsified"),
    }
    assert!(matches!(cont, DomainResult::Continue));
}

#[ntest::timeout(10000)]
#[test]
fn test_slice_batch_dim() {
    // Test slicing a [3, 4] array along batch dimension
    let arr = ArrayD::from_shape_vec(
        IxDyn(&[3, 4]),
        vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
            9.0, 10.0, 11.0, 12.0, // batch 2
        ],
    )
    .unwrap();

    // Slice batch 0
    let slice0 = slice_batch_dim(&arr, 0).unwrap();
    assert_eq!(slice0.shape(), &[4]);
    assert_eq!(slice0.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    // Slice batch 1
    let slice1 = slice_batch_dim(&arr, 1).unwrap();
    assert_eq!(slice1.shape(), &[4]);
    assert_eq!(slice1.as_slice().unwrap(), &[5.0, 6.0, 7.0, 8.0]);

    // Slice batch 2
    let slice2 = slice_batch_dim(&arr, 2).unwrap();
    assert_eq!(slice2.shape(), &[4]);
    assert_eq!(slice2.as_slice().unwrap(), &[9.0, 10.0, 11.0, 12.0]);

    // Out of bounds
    assert!(slice_batch_dim(&arr, 3).is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_slice_batch_dim_3d() {
    // Test slicing a [2, 3, 2] array along batch dimension
    let arr = ArrayD::from_shape_vec(
        IxDyn(&[2, 3, 2]),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0: [3, 2]
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1: [3, 2]
        ],
    )
    .unwrap();

    let slice0 = slice_batch_dim(&arr, 0).unwrap();
    assert_eq!(slice0.shape(), &[3, 2]);
    assert_eq!(slice0.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let slice1 = slice_batch_dim(&arr, 1).unwrap();
    assert_eq!(slice1.shape(), &[3, 2]);
    assert_eq!(
        slice1.as_slice().unwrap(),
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_slice_batch_dim_non_standard_layout() {
    // Permute axes to produce non-standard layout while keeping a batch dimension.
    let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .permuted_axes(IxDyn(&[1, 0]));
    assert!(
        arr.as_slice().is_none(),
        "permuted layout should not expose a contiguous slice"
    );

    let slice0 = slice_batch_dim(&arr, 0).unwrap();
    assert_eq!(slice0.shape(), &[2]);
    assert_eq!(slice0.as_slice().unwrap(), &[1.0, 4.0]);

    let slice1 = slice_batch_dim(&arr, 1).unwrap();
    assert_eq!(slice1.shape(), &[2]);
    assert_eq!(slice1.as_slice().unwrap(), &[2.0, 5.0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_slice_batch_dim_scalar_input() {
    let scalar = ArrayD::from_elem(IxDyn(&[]), 42.0);
    assert!(slice_batch_dim(&scalar, 0).is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_slice_batch_dim_batch_only() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let slice1 = slice_batch_dim(&arr, 1).unwrap();
    assert!(slice1.shape().is_empty());
    assert_eq!(slice1.iter().cloned().collect::<Vec<_>>(), vec![20.0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_stack_arrays_non_standard_layout() {
    let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .permuted_axes(IxDyn(&[1, 0]));
    assert!(
        arr1.as_slice().is_none(),
        "permuted layout should not expose a contiguous slice"
    );
    let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .permuted_axes(IxDyn(&[1, 0]));
    assert!(
        arr2.as_slice().is_none(),
        "permuted layout should not expose a contiguous slice"
    );

    let stacked = stack_arrays_pooled(&[arr1, arr2]).unwrap();
    let stacked = stacked.as_array();

    assert_eq!(stacked.shape(), &[2, 2, 2]);
    assert_eq!(stacked[[0, 0, 0]], 1.0);
    assert_eq!(stacked[[0, 0, 1]], 3.0);
    assert_eq!(stacked[[0, 1, 0]], 2.0);
    assert_eq!(stacked[[0, 1, 1]], 4.0);
    assert_eq!(stacked[[1, 0, 0]], 5.0);
    assert_eq!(stacked[[1, 0, 1]], 7.0);
    assert_eq!(stacked[[1, 1, 0]], 6.0);
    assert_eq!(stacked[[1, 1, 1]], 8.0);
}

// ============================================================================
// Domain conversion and update extraction tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_from_graph_domains_empty() {
    let domains: Vec<&GraphBabDomain> = vec![];
    let layer_names = vec!["relu0".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    assert!(batched.is_empty());
    assert_eq!(batched.len(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_from_graph_domains_single() {
    use crate::beta_crown::{GraphBetaState, GraphSplitHistory};
    use ny_tensor::BoundedTensor;
    use std::sync::Arc;

    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "relu0".to_string(),
        Arc::new(
            BoundedTensor::new(
                array![-1.0, 0.0, 1.0].into_dyn(),
                array![1.0, 2.0, 3.0].into_dyn(),
            )
            .unwrap(),
        ),
    );

    let domain = GraphBabDomain {
        history: GraphSplitHistory::new(),
        node_bounds,
        lower_bound: -0.5,
        upper_bound: 0.8,
        depth: 0,
        priority: 0.5,
        input_bounds: Arc::new(
            BoundedTensor::new(array![0.0, 0.5].into_dyn(), array![1.0, 1.5].into_dyn()).unwrap(),
        ),
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let domains = vec![&domain];
    let layer_names = vec!["relu0".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    assert_eq!(batched.len(), 1);
    assert_eq!(batched.lower_bounds(), &[-0.5]);
    assert_eq!(batched.upper_bounds(), &[0.8]);

    let relu0_lower = batched.layer_lowers().get("relu0").unwrap().as_array();
    assert_eq!(relu0_lower.shape(), &[1, 3]);
}

#[ntest::timeout(10000)]
#[test]
fn test_extract_updates() {
    let mut builder = BatchedDomainsBuilder::new(vec![]);

    // Add 2 domains
    for i in 0..2 {
        builder.add_domain(
            &HashMap::new(),
            array![0.0].into_dyn(),
            array![1.0].into_dyn(),
            0.0,
            1.0,
            i,
            vec![],
        );
    }

    let batched = builder.build().unwrap();

    // Extract updates with new bounds
    let new_lower = vec![-0.5, -0.3];
    let new_upper = vec![0.8, 0.9];
    let updates = batched.extract_updates(&new_lower, &new_upper).unwrap();

    assert_eq!(updates.len(), 2);
    assert_eq!(updates[0].domain_idx, 0);
    assert_eq!(updates[0].new_lower_bound, -0.5);
    assert_eq!(updates[0].new_upper_bound, 0.8);
    assert_eq!(updates[1].domain_idx, 1);
    assert_eq!(updates[1].new_lower_bound, -0.3);
    assert_eq!(updates[1].new_upper_bound, 0.9);
}

#[ntest::timeout(10000)]
#[test]
fn test_from_graph_domains_with_constraints() {
    use crate::beta_crown::{GraphBetaState, GraphNeuronConstraint, GraphSplitHistory};
    use ny_tensor::BoundedTensor;
    use std::sync::Arc;

    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 5,
        is_active: true,
        score: 0.5,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 3,
        is_active: false,
        score: 0.3,
    });

    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "relu0".to_string(),
        Arc::new(BoundedTensor::new(array![-1.0].into_dyn(), array![1.0].into_dyn()).unwrap()),
    );

    let domain = GraphBabDomain {
        history,
        node_bounds,
        lower_bound: -0.3,
        upper_bound: 0.7,
        depth: 2,
        priority: 0.5,
        input_bounds: Arc::new(
            BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap(),
        ),
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let domains = vec![&domain];
    let layer_names = vec!["relu0".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    assert_eq!(batched.len(), 1);
    assert_eq!(batched.constraints().len(), 1);
    assert_eq!(batched.constraints()[0].len(), 2);
    assert_eq!(
        batched.constraints()[0][0],
        ("relu0".to_string(), 5, true, None)
    );
    assert_eq!(
        batched.constraints()[0][1],
        ("relu1".to_string(), 3, false, None)
    );
}

// ============================================================================
// Extract updates with layer bounds tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_extract_updates_with_layer_bounds() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);
    // Add 2 domains with layer bounds
    for i in 0..2 {
        let mut layer_bounds = HashMap::new();
        layer_bounds.insert(
            "relu0".to_string(),
            (
                array![i as f32, i as f32 + 0.1].into_dyn(),
                array![i as f32 + 1.0, i as f32 + 1.1].into_dyn(),
            ),
        );
        builder.add_domain(
            &layer_bounds,
            array![0.0].into_dyn(),
            array![1.0].into_dyn(),
            0.0,
            1.0,
            i,
            vec![],
        );
    }

    let batched = builder.build().unwrap();

    // Create updated layer bounds from "GPU"
    let mut new_layer_lowers = HashMap::new();
    new_layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                -0.5, -0.4, // domain 0 new lower bounds
                -0.3, -0.2, // domain 1 new lower bounds
            ],
        )
        .unwrap(),
    );

    let mut new_layer_uppers = HashMap::new();
    new_layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                0.5, 0.6, // domain 0 new upper bounds
                0.7, 0.8, // domain 1 new upper bounds
            ],
        )
        .unwrap(),
    );

    let new_lower = vec![-1.0, -0.9];
    let new_upper = vec![1.0, 0.9];

    let updates = batched
        .extract_updates_with_layer_bounds(
            &new_lower,
            &new_upper,
            Some(&new_layer_lowers),
            Some(&new_layer_uppers),
        )
        .unwrap();
    assert_eq!(updates.len(), 2);
    // Check domain 0
    assert_eq!(updates[0].domain_idx, 0);
    assert_eq!(updates[0].new_lower_bound, -1.0);
    assert_eq!(updates[0].new_upper_bound, 1.0);
    let (d0_lower, d0_upper) = updates[0].new_layer_bounds.get("relu0").unwrap();
    assert_eq!(d0_lower.shape(), &[2]);
    assert_eq!(d0_lower.as_slice().unwrap(), &[-0.5, -0.4]);
    assert_eq!(d0_upper.as_slice().unwrap(), &[0.5, 0.6]);

    // Check domain 1
    assert_eq!(updates[1].domain_idx, 1);
    assert_eq!(updates[1].new_lower_bound, -0.9);
    assert_eq!(updates[1].new_upper_bound, 0.9);
    let (d1_lower, d1_upper) = updates[1].new_layer_bounds.get("relu0").unwrap();
    assert_eq!(d1_lower.shape(), &[2]);
    assert_eq!(d1_lower.as_slice().unwrap(), &[-0.3, -0.2]);
    assert_eq!(d1_upper.as_slice().unwrap(), &[0.7, 0.8]);
}

#[ntest::timeout(10000)]
#[test]
fn test_extract_updates_multi_layer_multi_dimension() {
    // Multi-layer + multi-dimension extraction: 3 layers, 3 domains, varying shapes.
    use ndarray::IxDyn;
    let layer_names = vec![
        "relu0".to_string(),
        "relu1".to_string(),
        "relu2".to_string(),
    ];
    let mut builder = BatchedDomainsBuilder::new(layer_names);

    // Define layer shapes (note: "shape" refers to tensor dimensions):
    // relu0: 1D shape [4] (dense layer with 4 neurons)
    // relu1: 3D shape [2, 2, 3] (conv output: H=2, W=2, C=3 = 12 elements)
    // relu2: 1D shape [2] (final dense with 2 neurons)

    // Add 3 domains with different layer bounds
    for domain_idx in 0..3 {
        let mut layer_bounds = HashMap::new();

        // relu0: [4] - 1D dense layer
        let base = domain_idx as f32;
        layer_bounds.insert(
            "relu0".to_string(),
            (
                array![base, base + 0.1, base + 0.2, base + 0.3].into_dyn(),
                array![base + 1.0, base + 1.1, base + 1.2, base + 1.3].into_dyn(),
            ),
        );

        // relu1: [2, 2, 3] - conv layer with shape (H=2, W=2, C=3)
        let relu1_lower: Vec<f32> = (0..12).map(|i| base + i as f32 * 0.01).collect();
        let relu1_upper: Vec<f32> = (0..12).map(|i| base + 1.0 + i as f32 * 0.01).collect();
        layer_bounds.insert(
            "relu1".to_string(),
            (
                ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), relu1_lower).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), relu1_upper).unwrap(),
            ),
        );

        // relu2: [2] - 1D final dense
        layer_bounds.insert(
            "relu2".to_string(),
            (
                array![base + 0.5, base + 0.6].into_dyn(),
                array![base + 1.5, base + 1.6].into_dyn(),
            ),
        );

        builder.add_domain(
            &layer_bounds,
            array![0.0].into_dyn(),
            array![1.0].into_dyn(),
            -(domain_idx as f32),
            domain_idx as f32,
            domain_idx,
            vec![],
        );
    }

    let batched = builder.build().unwrap();
    assert_eq!(batched.len(), 3);

    // Simulate GPU processing: create new batched layer bounds
    // Shape: [batch, *layer_shape]

    // relu0: [3, 4]
    let mut new_layer_lowers = HashMap::new();
    let relu0_new_lower: Vec<f32> = (0..12).map(|i| -0.5 + i as f32 * 0.1).collect();
    new_layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3, 4]), relu0_new_lower).unwrap(),
    );

    // relu1: [3, 2, 2, 3] - batch + (H, W, C)
    let relu1_new_lower: Vec<f32> = (0..36).map(|i| -1.0 + i as f32 * 0.05).collect();
    new_layer_lowers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 2, 3]), relu1_new_lower).unwrap(),
    );

    // relu2: [3, 2]
    let relu2_new_lower: Vec<f32> = vec![-0.2, -0.1, -0.3, -0.2, -0.4, -0.3];
    new_layer_lowers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), relu2_new_lower).unwrap(),
    );

    // Corresponding upper bounds
    let mut new_layer_uppers = HashMap::new();
    let relu0_new_upper: Vec<f32> = (0..12).map(|i| 0.5 + i as f32 * 0.1).collect();
    new_layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3, 4]), relu0_new_upper).unwrap(),
    );

    let relu1_new_upper: Vec<f32> = (0..36).map(|i| 1.0 + i as f32 * 0.05).collect();
    new_layer_uppers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 2, 3]), relu1_new_upper).unwrap(),
    );

    let relu2_new_upper: Vec<f32> = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    new_layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), relu2_new_upper).unwrap(),
    );

    // New objective bounds per domain
    let new_lower = vec![-2.0, -1.5, -1.0];
    let new_upper = vec![2.0, 1.5, 1.0];
    let updates = batched
        .extract_updates_with_layer_bounds(
            &new_lower,
            &new_upper,
            Some(&new_layer_lowers),
            Some(&new_layer_uppers),
        )
        .unwrap();

    assert_eq!(updates.len(), 3);

    // Verify each domain's extracted bounds
    for (idx, update) in updates.iter().enumerate() {
        assert_eq!(update.domain_idx, idx);
        assert_eq!(update.new_lower_bound, new_lower[idx]);
        assert_eq!(update.new_upper_bound, new_upper[idx]);

        // Verify all 3 layers extracted
        assert_eq!(
            update.new_layer_bounds.len(),
            3,
            "Domain {} should have 3 layer bounds",
            idx
        );

        // Verify relu0 shape: [4]
        let (relu0_l, relu0_u) = update.new_layer_bounds.get("relu0").unwrap();
        assert_eq!(
            relu0_l.shape(),
            &[4],
            "Domain {} relu0 lower should be [4]",
            idx
        );
        assert_eq!(
            relu0_u.shape(),
            &[4],
            "Domain {} relu0 upper should be [4]",
            idx
        );

        // Verify relu1 shape: [2, 2, 3] (3D shape preserved)
        let (relu1_l, relu1_u) = update.new_layer_bounds.get("relu1").unwrap();
        assert_eq!(
            relu1_l.shape(),
            &[2, 2, 3],
            "Domain {} relu1 lower should be [2,2,3]",
            idx
        );
        assert_eq!(
            relu1_u.shape(),
            &[2, 2, 3],
            "Domain {} relu1 upper should be [2,2,3]",
            idx
        );

        // Verify relu2 shape: [2]
        let (relu2_l, relu2_u) = update.new_layer_bounds.get("relu2").unwrap();
        assert_eq!(
            relu2_l.shape(),
            &[2],
            "Domain {} relu2 lower should be [2]",
            idx
        );
        assert_eq!(
            relu2_u.shape(),
            &[2],
            "Domain {} relu2 upper should be [2]",
            idx
        );
    }

    // Verify correct slicing for domain 1 (middle domain)
    let d1 = &updates[1];

    // relu0 domain 1: indices 4..8 of flattened [3, 4] array
    let (d1_relu0_l, d1_relu0_u) = d1.new_layer_bounds.get("relu0").unwrap();
    assert!(
        (d1_relu0_l[[0]] - (-0.5 + 4.0 * 0.1)).abs() < 1e-6,
        "relu0 lower[0] should be {} but got {}",
        -0.5 + 4.0 * 0.1,
        d1_relu0_l[[0]]
    );
    assert!(
        (d1_relu0_u[[0]] - (0.5 + 4.0 * 0.1)).abs() < 1e-6,
        "relu0 upper[0] should be {} but got {}",
        0.5 + 4.0 * 0.1,
        d1_relu0_u[[0]]
    );

    // relu1 domain 1: indices 12..24 of flattened [3, 2, 2, 3] array
    let (d1_relu1_l, d1_relu1_u) = d1.new_layer_bounds.get("relu1").unwrap();
    assert!(
        (d1_relu1_l[[0, 0, 0]] - (-1.0 + 12.0 * 0.05)).abs() < 1e-6,
        "relu1 lower[0,0,0] mismatch"
    );
    assert!(
        (d1_relu1_u[[0, 0, 0]] - (1.0 + 12.0 * 0.05)).abs() < 1e-6,
        "relu1 upper[0,0,0] mismatch"
    );

    // relu2 domain 1: indices 2..4 of [3, 2] array
    let (d1_relu2_l, d1_relu2_u) = d1.new_layer_bounds.get("relu2").unwrap();
    assert!(
        (d1_relu2_l[[0]] - (-0.3)).abs() < 1e-6,
        "relu2 lower[0] should be -0.3"
    );
    assert!(
        (d1_relu2_u[[0]] - 0.4).abs() < 1e-6,
        "relu2 upper[0] should be 0.4"
    );
}

#[cfg(feature = "benchmarks")] // #2249: wall-clock timing is flaky under load
#[ntest::timeout(10000)]
#[test]
fn test_extract_updates_scales_linearly() {
    // Performance regression test: verify layer-bound extraction scales O(n) not O(n^2).
    //
    // If extraction is O(n), doubling batch size should roughly double time.
    // If extraction is O(n^2), doubling batch size would quadruple time.
    //
    // We test with batch sizes 100, 500, 1000, 2000 and verify the ratio stays reasonable.
    use ndarray::IxDyn;
    use std::time::Instant;

    let layer_names = vec!["relu0".to_string(), "relu1".to_string()];

    // Helper to build batched domains and measure extraction time
    fn measure_extraction_time(batch_size: usize, layer_names: &[String]) -> std::time::Duration {
        let mut builder = BatchedDomainsBuilder::new(layer_names.to_vec());
        for i in 0..batch_size {
            let mut layer_bounds = HashMap::new();
            let base = i as f32;
            let relu0_lower: Vec<f32> = (0..16).map(|j| base + j as f32 * 0.001).collect();
            let relu0_upper: Vec<f32> = (0..16).map(|j| base + 1.0 + j as f32 * 0.001).collect();
            layer_bounds.insert(
                "relu0".to_string(),
                (
                    ArrayD::from_shape_vec(IxDyn(&[16]), relu0_lower).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[16]), relu0_upper).unwrap(),
                ),
            );

            // relu1: [8] neurons
            let relu1_lower: Vec<f32> = (0..8).map(|j| base + j as f32 * 0.002).collect();
            let relu1_upper: Vec<f32> = (0..8).map(|j| base + 1.0 + j as f32 * 0.002).collect();
            layer_bounds.insert(
                "relu1".to_string(),
                (
                    ArrayD::from_shape_vec(IxDyn(&[8]), relu1_lower).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[8]), relu1_upper).unwrap(),
                ),
            );

            builder.add_domain(
                &layer_bounds,
                array![0.0].into_dyn(),
                array![1.0].into_dyn(),
                -(i as f32),
                i as f32,
                i,
                vec![],
            );
        }

        let batched = builder.build().unwrap();

        // Create "GPU" output layer bounds
        let mut new_layer_lowers = HashMap::new();
        let mut new_layer_uppers = HashMap::new();

        let relu0_data: Vec<f32> = (0..(batch_size * 16))
            .map(|i| -0.5 + i as f32 * 0.0001)
            .collect();
        new_layer_lowers.insert(
            "relu0".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[batch_size, 16]), relu0_data.clone()).unwrap(),
        );
        new_layer_uppers.insert(
            "relu0".to_string(),
            ArrayD::from_shape_vec(
                IxDyn(&[batch_size, 16]),
                relu0_data.iter().map(|x| x + 1.0).collect(),
            )
            .unwrap(),
        );

        let relu1_data: Vec<f32> = (0..(batch_size * 8))
            .map(|i| -0.3 + i as f32 * 0.0002)
            .collect();
        new_layer_lowers.insert(
            "relu1".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[batch_size, 8]), relu1_data.clone()).unwrap(),
        );
        new_layer_uppers.insert(
            "relu1".to_string(),
            ArrayD::from_shape_vec(
                IxDyn(&[batch_size, 8]),
                relu1_data.iter().map(|x| x + 1.0).collect(),
            )
            .unwrap(),
        );

        let new_lower: Vec<f32> = (0..batch_size).map(|i| -(i as f32)).collect();
        let new_upper: Vec<f32> = (0..batch_size).map(|i| i as f32).collect();

        // Measure extraction time
        let start = Instant::now();
        let _updates = batched
            .extract_updates_with_layer_bounds(
                &new_lower,
                &new_upper,
                Some(&new_layer_lowers),
                Some(&new_layer_uppers),
            )
            .unwrap();
        start.elapsed()
    }

    // Warmup run to stabilize measurements (avoid cold-start overhead)
    let _ = measure_extraction_time(50, &layer_names);

    // Run multiple iterations and take median to reduce variance
    const NUM_ITERATIONS: usize = 5;

    let mut times_100 = Vec::with_capacity(NUM_ITERATIONS);
    let mut times_500 = Vec::with_capacity(NUM_ITERATIONS);
    let mut times_1000 = Vec::with_capacity(NUM_ITERATIONS);
    let mut times_2000 = Vec::with_capacity(NUM_ITERATIONS);

    for _ in 0..NUM_ITERATIONS {
        times_100.push(measure_extraction_time(100, &layer_names));
        times_500.push(measure_extraction_time(500, &layer_names));
        times_1000.push(measure_extraction_time(1000, &layer_names));
        times_2000.push(measure_extraction_time(2000, &layer_names));
    }

    // Sort and take median
    times_100.sort();
    times_500.sort();
    times_1000.sort();
    times_2000.sort();

    let time_100 = times_100[NUM_ITERATIONS / 2];
    let time_500 = times_500[NUM_ITERATIONS / 2];
    let time_1000 = times_1000[NUM_ITERATIONS / 2];
    let time_2000 = times_2000[NUM_ITERATIONS / 2];

    // Calculate ratios
    // For O(n): time_500 / time_100 ≈ 5, time_1000 / time_500 ≈ 2, time_2000 / time_1000 ≈ 2
    // For O(n^2): time_500 / time_100 ≈ 25, time_1000 / time_500 ≈ 4, time_2000 / time_1000 ≈ 4

    let ratio_500_100 = time_500.as_nanos() as f64 / time_100.as_nanos().max(1) as f64;
    let ratio_1000_500 = time_1000.as_nanos() as f64 / time_500.as_nanos().max(1) as f64;
    let ratio_2000_1000 = time_2000.as_nanos() as f64 / time_1000.as_nanos().max(1) as f64;

    // Log results for debugging
    eprintln!("Batch 100:  {:?} (median of {})", time_100, NUM_ITERATIONS);
    eprintln!("Batch 500:  {:?} (median of {})", time_500, NUM_ITERATIONS);
    eprintln!("Batch 1000: {:?} (median of {})", time_1000, NUM_ITERATIONS);
    eprintln!("Batch 2000: {:?} (median of {})", time_2000, NUM_ITERATIONS);
    eprintln!("Ratio 500/100:    {:.2}", ratio_500_100);
    eprintln!("Ratio 1000/500:   {:.2}", ratio_1000_500);
    eprintln!("Ratio 2000/1000:  {:.2}", ratio_2000_1000);

    // For linear scaling:
    // - 500/100 (5x domains) should give roughly 5x time, max ~10x with overhead
    // - 1000/500 (2x domains) should give roughly 2x time, max ~4.5x with overhead
    // - 2000/1000 (2x domains) should give roughly 2x time, max ~3.5x with overhead
    //
    // For quadratic scaling:
    // - 500/100 would give 25x (fails 15.0 threshold)
    // - 1000/500 would give 4x (borderline with 4.5x threshold)
    // - 2000/1000 would give 4x (clear failure with 3.5x threshold - primary detection)
    //
    // Using 5 iterations with median reduces variance and allows tighter thresholds.
    // The 2000/1000 ratio is the primary O(n^2) detector per #164:
    // - O(n) gives ~2x
    // - O(n^2) gives 4x
    // - Threshold of 3.5x provides clear separation
    assert!(
        ratio_500_100 < 15.0,
        "Scaling from 100->500 domains should be linear (≤15x), got {:.1}x (O(n^2) would be ~25x)",
        ratio_500_100
    );
    assert!(
        ratio_1000_500 < 4.5,
        "Scaling from 500->1000 domains should be linear (≤4.5x), got {:.1}x (O(n^2) would be ~4x)",
        ratio_1000_500
    );
    // Primary O(n^2) detector: 2000/1000 ratio (Part of #164)
    // O(n): 2x, O(n^2): 4x, threshold: 3.5x (12.5% margin from quadratic)
    assert!(
        ratio_2000_1000 < 3.5,
        "Scaling from 1000->2000 domains should be linear (≤3.5x), got {:.1}x (O(n^2) would give 4x)",
        ratio_2000_1000
    );
}
