// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::array;
use ny_tensor::TensorPool;
use std::collections::HashMap;

#[ntest::timeout(5000)]
#[test]
fn test_batched_domains_builder_empty() {
    let builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);
    let batched = builder.build().unwrap();

    assert!(batched.is_empty());
    assert_eq!(batched.len(), 0);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_domains_builder_single() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            array![-1.0, 0.0, 1.0].into_dyn(),
            array![1.0, 2.0, 3.0].into_dyn(),
        ),
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

    assert_eq!(batched.len(), 1);
    assert_eq!(batched.batch_size(), 1);
    assert_eq!(batched.lower_bounds(), &[-1.0]);
    assert_eq!(batched.upper_bounds(), &[0.5]);
    assert_eq!(batched.depths(), &[0]);

    // Check layer bounds shape: [1, 3] (batch=1, hidden=3)
    let relu0_lower = batched.layer_lowers().get("relu0").unwrap().as_array();
    assert_eq!(relu0_lower.shape(), &[1, 3]);

    // Check input bounds shape: [1, 2] (batch=1, input=2)
    assert_eq!(batched.input_lowers().as_array().shape(), &[1, 2]);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_domains_builder_multiple() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    // Add 3 domains
    for i in 0..3 {
        let mut layer_bounds = HashMap::new();
        layer_bounds.insert(
            "relu0".to_string(),
            (
                array![-1.0 + i as f32, 0.0, 1.0].into_dyn(),
                array![1.0 + i as f32, 2.0, 3.0].into_dyn(),
            ),
        );

        builder.add_domain(
            &layer_bounds,
            array![0.0, 0.5].into_dyn(),
            array![1.0, 1.5].into_dyn(),
            -1.0 + i as f32,
            0.5 + i as f32,
            i,
            if i > 0 {
                vec![("relu0".to_string(), i, i % 2 == 0, None)]
            } else {
                vec![]
            },
        );
    }

    let batched = builder.build().unwrap();

    assert_eq!(batched.len(), 3);
    assert_eq!(batched.batch_size(), 3);
    assert_eq!(batched.lower_bounds(), &[-1.0, 0.0, 1.0]);
    assert_eq!(batched.upper_bounds(), &[0.5, 1.5, 2.5]);
    assert_eq!(batched.depths(), &[0, 1, 2]);

    // Check layer bounds shape: [3, 3] (batch=3, hidden=3)
    let relu0_lower = batched.layer_lowers().get("relu0").unwrap().as_array();
    assert_eq!(relu0_lower.shape(), &[3, 3]);

    // Check input bounds shape: [3, 2] (batch=3, input=2)
    assert_eq!(batched.input_lowers().as_array().shape(), &[3, 2]);

    // Check constraints (4-tuple format: name, idx, is_active, split_point)
    assert!(batched.constraints()[0].is_empty());
    assert_eq!(
        batched.constraints()[1],
        vec![("relu0".to_string(), 1, false, None)]
    );
    assert_eq!(
        batched.constraints()[2],
        vec![("relu0".to_string(), 2, true, None)]
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_domains_pool_returns_buffers() {
    TensorPool::clear();
    TensorPool::reset_stats();

    {
        let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);
        let mut layer_bounds = HashMap::new();
        layer_bounds.insert(
            "relu0".to_string(),
            (
                array![-1.0, 0.0, 1.0].into_dyn(),
                array![1.0, 2.0, 3.0].into_dyn(),
            ),
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

        let _batched = builder.build().unwrap();
    }

    let stats = TensorPool::stats();
    assert!(stats.returns > 0, "expected pooled buffers to be returned");
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_domains_builder_missing_layer_bounds_errors() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    builder.add_domain(
        &HashMap::new(),
        array![0.0].into_dyn(),
        array![1.0].into_dyn(),
        0.0,
        1.0,
        0,
        vec![],
    );

    let result = builder.build();
    assert!(result.is_err());
}

// --- Builder rejects non-finite layer/input tensors (#3115) ---

#[ntest::timeout(5000)]
#[test]
fn test_batched_domains_builder_rejects_nan_layer_bounds_3115() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            array![f32::NAN, 0.0, 1.0].into_dyn(), // NaN in layer lower
            array![1.0, 2.0, 3.0].into_dyn(),
        ),
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

    let err = builder
        .build()
        .expect_err("NaN layer bounds must be rejected at build time");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_domains_builder_rejects_inf_input_bounds_3115() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            array![-1.0, 0.0, 1.0].into_dyn(),
            array![1.0, 2.0, 3.0].into_dyn(),
        ),
    );

    builder.add_domain(
        &layer_bounds,
        array![0.0, f32::INFINITY].into_dyn(), // Inf in input lower
        array![1.0, 1.5].into_dyn(),
        -1.0,
        0.5,
        0,
        vec![],
    );

    let err = builder
        .build()
        .expect_err("Inf input bounds must be rejected at build time");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}
