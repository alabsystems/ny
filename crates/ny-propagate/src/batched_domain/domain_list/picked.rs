// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Picked domains for GPU processing.
//!
//! Contains `PickedDomains` and its branch-selection methods
//! (`find_unstable_neurons_batched`, `select_branch_batched`).

use super::DomainMetadata;
use ndarray::ArrayD;
use ny_core::{checked_dim_product, NyError, Result};
use std::collections::HashMap;

use crate::contiguous_flat_slice;

/// Domains picked out for GPU processing.
#[derive(Debug)]
pub struct PickedDomains {
    /// Number of domains in this batch.
    pub batch_size: usize,
    /// Per-layer lower bounds: [batch, *shape].
    pub layer_lowers: HashMap<String, ArrayD<f32>>,
    /// Per-layer upper bounds: [batch, *shape].
    pub layer_uppers: HashMap<String, ArrayD<f32>>,
    /// Input lower bounds: [batch, *input_shape].
    pub input_lowers: ArrayD<f32>,
    /// Input upper bounds: [batch, *input_shape].
    pub input_uppers: ArrayD<f32>,
    /// Global lower bounds: [batch].
    pub global_lbs: Vec<f32>,
    /// Global upper bounds: [batch].
    pub global_ubs: Vec<f32>,
    /// Domain metadata (constraints, depths).
    pub metadata: Vec<DomainMetadata>,
}

impl PickedDomains {
    /// Find unstable ReLU neurons for all domains in the batch.
    ///
    /// Returns a Vec of Vec of (node_name, neuron_idx) pairs, one per domain.
    /// A neuron is unstable if: lower < 0 AND upper > 0 AND not already constrained.
    ///
    /// # Arguments
    /// * `relu_pre_map` - Map from ReLU node name to pre-activation layer name.
    ///   For example: `{"relu0": "linear0", "relu1": "linear1"}`.
    ///   Only ReLU nodes present in this map will be checked.
    ///
    /// # Returns
    /// Vec of length batch_size, each containing (node_name, neuron_idx) pairs
    /// for unstable neurons in that domain.
    ///
    /// # Example
    /// ```text
    /// let relu_pre_map = [("relu0".into(), "linear0".into())].into_iter().collect();
    /// let unstable = picked.find_unstable_neurons_batched(&relu_pre_map)?;
    /// // unstable[0] = [("relu0", 2), ("relu0", 5)] - domain 0 has neurons 2,5 unstable
    /// // unstable[1] = [("relu0", 1)] - domain 1 has neuron 1 unstable
    /// ```
    pub fn find_unstable_neurons_batched(
        &self,
        relu_pre_map: &HashMap<String, String>,
    ) -> Result<Vec<Vec<(String, usize)>>> {
        let batch_size = self.batch_size;
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Build constraint maps for each domain: (node_name, neuron_idx) -> true
        let constraint_sets: Vec<std::collections::HashSet<(String, usize)>> = self
            .metadata
            .iter()
            .map(|m| {
                m.constraints
                    .iter()
                    .map(|(node, idx, _, _)| (node.clone(), *idx))
                    .collect()
            })
            .collect();

        let mut result: Vec<Vec<(String, usize)>> = vec![Vec::new(); batch_size];

        for (relu_name, pre_name) in relu_pre_map {
            // Get pre-activation bounds for this layer
            let Some(pre_lower) = self.layer_lowers.get(pre_name) else {
                continue;
            };
            let Some(pre_upper) = self.layer_uppers.get(pre_name) else {
                continue;
            };

            // Bounds shape: [batch, *layer_shape]
            let shape = pre_lower.shape();
            if shape.is_empty() || shape[0] != batch_size {
                continue;
            }

            // Flatten the layer dimensions: [batch, num_neurons]
            let num_neurons: usize =
                checked_dim_product(&shape[1..], "PickedDomains neuron dimensions")?;

            let lower_slice = contiguous_flat_slice(pre_lower);
            let upper_slice = contiguous_flat_slice(pre_upper);

            for neuron_idx in 0..num_neurons {
                for batch_idx in 0..batch_size {
                    // Index into flattened array: batch_idx * num_neurons + neuron_idx
                    let idx = batch_idx * num_neurons + neuron_idx;

                    let l = *checked_bound_lookup(
                        lower_slice.as_ref(),
                        idx,
                        "find_unstable_neurons_batched",
                        pre_name,
                    )?;
                    let u = *checked_bound_lookup(
                        upper_slice.as_ref(),
                        idx,
                        "find_unstable_neurons_batched",
                        pre_name,
                    )?;

                    // Neuron is unstable if l < 0 && u > 0.
                    // NaN bounds are conservatively treated as unstable — both
                    // NaN < 0 and NaN > 0 return false, so without this guard
                    // NaN-bounded neurons would be classified as stable (#2246).
                    if (l < 0.0 && u > 0.0) || l.is_nan() || u.is_nan() {
                        // Check if already constrained
                        let key = (relu_name.clone(), neuron_idx);
                        if !constraint_sets[batch_idx].contains(&key) {
                            result[batch_idx].push(key);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Select branching decisions for all domains using intercept scoring.
    ///
    /// For each domain, selects the unstable neuron with the highest intercept score:
    /// `intercept = (-lower * upper) / (upper - lower)`
    ///
    /// # Arguments
    /// * `unstable_per_domain` - Output from `find_unstable_neurons_batched`
    /// * `relu_pre_map` - Map from ReLU node name to pre-activation layer name
    ///
    /// # Returns
    /// Vec of (node_name, neuron_idx, score) for each domain.
    /// Returns None for domains with no unstable neurons.
    pub fn select_branch_batched(
        &self,
        unstable_per_domain: &[Vec<(String, usize)>],
        relu_pre_map: &HashMap<String, String>,
    ) -> Result<Vec<Option<(String, usize, f32)>>> {
        let batch_size = self.batch_size;
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let mut result: Vec<Option<(String, usize, f32)>> = vec![None; batch_size];

        for (batch_idx, unstable) in unstable_per_domain.iter().enumerate() {
            // Guard against mismatched input sizes
            if batch_idx >= batch_size {
                break;
            }
            if unstable.is_empty() {
                continue;
            }

            let mut best: Option<(String, usize, f32)> = None;

            for (relu_name, neuron_idx) in unstable {
                let Some(pre_name) = relu_pre_map.get(relu_name) else {
                    continue;
                };
                let Some(pre_lower) = self.layer_lowers.get(pre_name) else {
                    continue;
                };
                let Some(pre_upper) = self.layer_uppers.get(pre_name) else {
                    continue;
                };

                let shape = pre_lower.shape();
                if shape.is_empty() || shape[0] != batch_size {
                    continue;
                }
                let num_neurons: usize =
                    checked_dim_product(&shape[1..], "PickedDomains neuron dimensions")?;
                let idx = batch_idx * num_neurons + neuron_idx;

                let lower_slice = contiguous_flat_slice(pre_lower);
                let upper_slice = contiguous_flat_slice(pre_upper);

                let l = *checked_bound_lookup(
                    lower_slice.as_ref(),
                    idx,
                    "select_branch_batched",
                    pre_name,
                )?;
                let u = *checked_bound_lookup(
                    upper_slice.as_ref(),
                    idx,
                    "select_branch_batched",
                    pre_name,
                )?;

                // Skip neurons with NaN bounds — NaN intercept scores would
                // corrupt branch selection via map_or(true, ...) (#2246).
                if l.is_nan() || u.is_nan() {
                    continue;
                }

                // Intercept score: (-l * u) / (u - l)
                // Higher values indicate more relaxation error to eliminate
                let width = u - l;
                if width > 0.0 {
                    let intercept = (-l * u) / width;
                    if best
                        .as_ref()
                        .map_or(true, |(_, _, score)| intercept > *score)
                    {
                        best = Some((relu_name.clone(), *neuron_idx, intercept));
                    }
                }
            }

            result[batch_idx] = best;
        }

        Ok(result)
    }
}

/// Checked bound-slice lookup returning `NyError::InternalError` on out-of-range.
///
/// Replaces raw `slice[idx]` panics in `find_unstable_neurons_batched` and
/// `select_branch_batched` with a structured error that reports the function,
/// layer name, index, and slice length.
fn checked_bound_lookup<'a>(
    slice: &'a [f32],
    idx: usize,
    function: &str,
    layer: &str,
) -> Result<&'a f32> {
    slice.get(idx).ok_or_else(|| {
        NyError::InternalError(format!(
            "{function}: bound index {idx} out of range (slice len {}) for layer '{layer}'",
            slice.len(),
        ))
    })
}
