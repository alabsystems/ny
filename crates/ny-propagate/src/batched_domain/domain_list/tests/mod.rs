// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `DomainList` pick/add operations, alpha state persistence,
//! permutation utilities, and non-contiguous tensor handling.

use super::super::builder::BatchedDomainsBuilder;
use super::super::options::BatchedDomainOptions;
use super::super::types::BatchedDomains;
use super::*;

use ndarray::{ArrayD, IxDyn};
use ny_tensor::TreeTraversal;
use std::collections::HashMap;
use std::sync::Arc;

mod alpha_state;
mod grouped;
mod non_contiguous;
mod override_validation;
mod permutation;
mod pick_add;

fn create_test_config() -> DomainListConfig {
    let mut layer_shapes = HashMap::new();
    layer_shapes.insert("relu1".to_string(), vec![2]);
    layer_shapes.insert("relu2".to_string(), vec![2]);
    DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: vec!["relu1".to_string(), "relu2".to_string()],
        layer_shapes,
        input_shape: vec![4],
        initial_capacity: 16,
        max_queue_size: 0,
    }
}
