// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::batched_domain::{DomainMetadata, ProcessedDomains};

/// Create a ProcessedDomains structure containing just the root domain.
///
/// This is used to initialize DomainList with the root domain before BaB starts.
///
/// # Arguments
/// * `node_bounds` - Initial bounds for each node in the graph
/// * `input` - Input bounds specification
/// * `lower_bound` - Root domain's objective lower bound
/// * `upper_bound` - Root domain's objective upper bound
/// * `layer_names` - Ordered list of layer names for deterministic iteration
pub fn create_root_processed_domain(
    node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    lower_bound: f32,
    upper_bound: f32,
    layer_names: &[String],
) -> Result<ProcessedDomains> {
    let mut layer_lowers: HashMap<String, ndarray::ArrayD<f32>> = HashMap::new();
    let mut layer_uppers: HashMap<String, ndarray::ArrayD<f32>> = HashMap::new();

    for name in layer_names {
        if let Some(bounds) = node_bounds.get(name) {
            let lower = bounds.lower().clone().insert_axis(ndarray::Axis(0));
            let upper = bounds.upper().clone().insert_axis(ndarray::Axis(0));
            layer_lowers.insert(name.clone(), lower);
            layer_uppers.insert(name.clone(), upper);
        }
    }

    let input_lowers = input.lower().clone().insert_axis(ndarray::Axis(0));
    let input_uppers = input.upper().clone().insert_axis(ndarray::Axis(0));
    let metadata = vec![DomainMetadata::root(lower_bound, upper_bound)?];

    Ok(ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers,
        input_uppers,
        global_lbs: vec![lower_bound],
        global_ubs: vec![upper_bound],
        metadata,
        keep_mask: vec![true],
    })
}
