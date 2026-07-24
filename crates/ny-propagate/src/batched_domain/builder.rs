// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::options::BatchedDomainOptions;
use super::types::BatchedDomains;
use super::utils::{slice_batch_dim, stack_arrays_pooled, unstable_mask_from_bounds};
use super::ConstraintTuple;
use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::PooledArray;
use std::collections::HashMap;

/// Builder for creating BatchedDomains from individual domains.
///
/// This follows alpha-beta-CROWN's pattern of stacking tensors along
/// the batch dimension for GPU transfer.
pub struct BatchedDomainsBuilder {
    layer_names: Vec<String>,
    layer_lowers: HashMap<String, Vec<ArrayD<f32>>>,
    layer_uppers: HashMap<String, Vec<ArrayD<f32>>>,
    input_lowers: Vec<ArrayD<f32>>,
    input_uppers: Vec<ArrayD<f32>>,
    lower_bounds: Vec<f32>,
    upper_bounds: Vec<f32>,
    depths: Vec<usize>,
    constraints: Vec<Vec<ConstraintTuple>>,
    options: BatchedDomainOptions,
}

impl BatchedDomainsBuilder {
    /// Create a new builder for the given layer names (test convenience wrapper).
    #[cfg(test)]
    pub fn new(layer_names: Vec<String>) -> Self {
        Self::new_with_options(layer_names, BatchedDomainOptions::default())
    }

    /// Create a new builder with options for batched domain storage.
    pub fn new_with_options(layer_names: Vec<String>, options: BatchedDomainOptions) -> Self {
        let layer_lowers = layer_names
            .iter()
            .map(|n| (n.clone(), Vec::new()))
            .collect();
        let layer_uppers = layer_names
            .iter()
            .map(|n| (n.clone(), Vec::new()))
            .collect();

        Self {
            layer_names,
            layer_lowers,
            layer_uppers,
            input_lowers: Vec::new(),
            input_uppers: Vec::new(),
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            depths: Vec::new(),
            constraints: Vec::new(),
            options,
        }
    }

    /// Add a domain's bounds to the batch.
    ///
    /// # Arguments
    /// * `layer_bounds` - Map from layer name to (lower, upper) bounds
    /// * `input_lower` - Input lower bounds
    /// * `input_upper` - Input upper bounds
    /// * `lower_bound` - Lower bound on objective
    /// * `upper_bound` - Upper bound on objective
    /// * `depth` - Number of splits applied
    /// * `domain_constraints` - List of (node_name, neuron_idx, is_active, split_point)
    // Justification: Domain builder needs layer bounds, input bounds, objective bounds,
    // depth, constraints, alpha/beta state, and split history — all from a BaB domain.
    #[allow(clippy::too_many_arguments)]
    pub fn add_domain(
        &mut self,
        layer_bounds: &HashMap<String, (ArrayD<f32>, ArrayD<f32>)>,
        input_lower: ArrayD<f32>,
        input_upper: ArrayD<f32>,
        lower_bound: f32,
        upper_bound: f32,
        depth: usize,
        domain_constraints: Vec<ConstraintTuple>,
    ) {
        for name in &self.layer_names {
            if let Some((lower, upper)) = layer_bounds.get(name) {
                self.layer_lowers
                    .entry(name.clone())
                    .or_default()
                    .push(lower.clone());
                self.layer_uppers
                    .entry(name.clone())
                    .or_default()
                    .push(upper.clone());
            }
        }

        self.input_lowers.push(input_lower);
        self.input_uppers.push(input_upper);
        self.lower_bounds.push(lower_bound);
        self.upper_bounds.push(upper_bound);
        self.depths.push(depth);
        self.constraints.push(domain_constraints);
    }

    /// Build the batched domains by stacking tensors.
    ///
    /// This performs the actual tensor stacking along the batch dimension.
    ///
    /// # REQUIRES
    /// - All arrays added via `add_domain()` must have consistent shapes per layer
    /// - At least one domain must have been added (batch_size > 0)
    ///
    /// # ENSURES
    /// - Returns `BatchedDomains` with tensors stacked along batch dimension
    /// - `result.batch_size == number of domains added`
    /// - Each `layer_lowers[name]` has shape `[batch, *original_layer_shape]`
    /// - Constraints preserved for each domain in batch order
    pub fn build(self) -> Result<BatchedDomains> {
        let batch_size = self.lower_bounds.len();
        if batch_size == 0 {
            return Ok(BatchedDomains::with_capacity(0, &self.layer_names));
        }

        // Reject non-finite global bounds early (#2246).
        super::utils::validate_global_bounds_finite(
            &self.lower_bounds,
            &self.upper_bounds,
            "BatchedDomainsBuilder",
        )?;

        if self.upper_bounds.len() != batch_size
            || self.depths.len() != batch_size
            || self.constraints.len() != batch_size
            || self.input_lowers.len() != batch_size
            || self.input_uppers.len() != batch_size
        {
            return Err(NyError::InvalidSpec(format!(
                "BatchedDomainsBuilder length mismatch: lower_bounds={}, upper_bounds={}, depths={}, constraints={}, input_lowers={}, input_uppers={}",
                batch_size,
                self.upper_bounds.len(),
                self.depths.len(),
                self.constraints.len(),
                self.input_lowers.len(),
                self.input_uppers.len()
            )));
        }

        // Reject non-finite layer/input tensors before stacking (#3115).
        // Builder stores individual domain arrays (not batched), so validate
        // elements directly rather than using the batched helper.
        for name in &self.layer_names {
            if let Some(arrays) = self.layer_lowers.get(name) {
                for (i, arr) in arrays.iter().enumerate() {
                    for (j, &val) in arr.iter().enumerate() {
                        if !val.is_finite() {
                            return Err(NyError::NumericalInstability(format!(
                                "BatchedDomainsBuilder layer_lower[{name}] domain {i} \
                                 element {j} has non-finite value ({val})"
                            )));
                        }
                    }
                }
            }
            if let Some(arrays) = self.layer_uppers.get(name) {
                for (i, arr) in arrays.iter().enumerate() {
                    for (j, &val) in arr.iter().enumerate() {
                        if !val.is_finite() {
                            return Err(NyError::NumericalInstability(format!(
                                "BatchedDomainsBuilder layer_upper[{name}] domain {i} \
                                 element {j} has non-finite value ({val})"
                            )));
                        }
                    }
                }
            }
        }
        for (i, arr) in self.input_lowers.iter().enumerate() {
            for (j, &val) in arr.iter().enumerate() {
                if !val.is_finite() {
                    return Err(NyError::NumericalInstability(format!(
                        "BatchedDomainsBuilder input_lower domain {i} \
                         element {j} has non-finite value ({val})"
                    )));
                }
            }
        }
        for (i, arr) in self.input_uppers.iter().enumerate() {
            for (j, &val) in arr.iter().enumerate() {
                if !val.is_finite() {
                    return Err(NyError::NumericalInstability(format!(
                        "BatchedDomainsBuilder input_upper domain {i} \
                         element {j} has non-finite value ({val})"
                    )));
                }
            }
        }

        // Stack layer bounds along batch dimension
        let mut layer_lowers = HashMap::new();
        for name in &self.layer_names {
            let arrays = self.layer_lowers.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Missing lower bounds for layer '{}' in builder",
                    name
                ))
            })?;
            if arrays.len() != batch_size {
                return Err(NyError::InvalidSpec(format!(
                    "Layer '{}' lower bounds count {} does not match batch size {}",
                    name,
                    arrays.len(),
                    batch_size
                )));
            }
            layer_lowers.insert(name.clone(), stack_arrays_pooled(arrays)?);
        }

        let mut layer_uppers = HashMap::new();
        for name in &self.layer_names {
            let arrays = self.layer_uppers.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Missing upper bounds for layer '{}' in builder",
                    name
                ))
            })?;
            if arrays.len() != batch_size {
                return Err(NyError::InvalidSpec(format!(
                    "Layer '{}' upper bounds count {} does not match batch size {}",
                    name,
                    arrays.len(),
                    batch_size
                )));
            }
            layer_uppers.insert(name.clone(), stack_arrays_pooled(arrays)?);
        }

        // Stack input bounds
        let input_lowers = stack_arrays_pooled(&self.input_lowers)?;
        let input_uppers = stack_arrays_pooled(&self.input_uppers)?;

        let (static_layer_lowers, static_layer_uppers, unstable_masks) =
            if self.options.enable_interm_transfer {
                let mut static_lowers = HashMap::new();
                let mut static_uppers = HashMap::new();
                let mut masks = HashMap::new();

                for name in &self.layer_names {
                    let lowers = layer_lowers
                        .get(name)
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Missing pooled lower bounds for layer '{}' in builder",
                                name
                            ))
                        })?
                        .as_array();
                    let uppers = layer_uppers
                        .get(name)
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Missing pooled upper bounds for layer '{}' in builder",
                                name
                            ))
                        })?
                        .as_array();

                    let lower_slice = slice_batch_dim(lowers, 0).ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Failed to slice static lower bounds for layer '{}'",
                            name
                        ))
                    })?;
                    let upper_slice = slice_batch_dim(uppers, 0).ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Failed to slice static upper bounds for layer '{}'",
                            name
                        ))
                    })?;

                    let mask = unstable_mask_from_bounds(&lower_slice, &upper_slice)?;
                    static_lowers.insert(name.clone(), PooledArray::from_array(lower_slice));
                    static_uppers.insert(name.clone(), PooledArray::from_array(upper_slice));
                    masks.insert(name.clone(), mask);
                }

                (Some(static_lowers), Some(static_uppers), Some(masks))
            } else {
                (None, None, None)
            };

        Ok(BatchedDomains {
            batch_size,
            layer_lowers,
            layer_uppers,
            input_lowers,
            input_uppers,
            static_layer_lowers,
            static_layer_uppers,
            unstable_masks,
            lower_bounds: self.lower_bounds,
            upper_bounds: self.upper_bounds,
            depths: self.depths,
            constraints: self.constraints,
        })
    }
}
