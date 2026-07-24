// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sparse storage for intermediate bounds of unstable neurons only.
//!
//! This module implements memory-efficient storage for branch-and-bound domains
//! by storing only the bounds for unstable neurons (those with crossing bounds:
//! lb < 0 < ub). Stable neurons have fixed activation status across all domains
//! and don't need per-domain tracking.
//!
//! Reference: alpha-beta-CROWN's `unstable_interm_bounds` in branching_domains.py

#[cfg(test)]
use ndarray::Array2;
#[cfg(test)]
use ny_core::{checked_shape_product, NyError};
#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
use ny_tensor::PooledArray;

#[cfg(test)]
use super::types::BatchedDomains;
#[cfg(test)]
use crate::{contiguous_flat_slice, contiguous_flat_slice_mut};

/// Sparse storage for unstable intermediate bounds only.
///
/// For each layer, stores bounds only for neurons in the unstable mask.
/// This reduces memory usage significantly when most neurons are stable.
///
/// # Storage Format
/// - `bounds`: HashMap from layer name to (lower, upper) bound arrays
/// - Shape: `[batch_size, num_unstable_neurons]`
/// - Only neurons where the parent domain has crossing bounds (lb < 0 < ub)
///   are tracked in the sparse storage
///
/// # Usage Pattern
/// 1. Create from full bounds using `from_batched_domains()`
/// 2. Store in domain queue (memory efficient)
/// 3. Merge back into full bounds using `merge_into()` when picked for processing
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct SparseIntermediateBounds {
    /// Per-layer: (lower_bounds, upper_bounds) for unstable neurons only.
    /// Shape: `[batch_size, num_unstable_neurons]` for each layer.
    bounds: HashMap<String, (Array2<f32>, Array2<f32>)>,
}

#[cfg(test)]
impl SparseIntermediateBounds {
    /// Create empty sparse bounds storage.
    pub fn new() -> Self {
        Self {
            bounds: HashMap::new(),
        }
    }

    /// Create sparse bounds from full batched domain bounds.
    ///
    /// Extracts only the unstable neuron bounds based on the masks in `batched`.
    /// Returns `None` if unstable masks are not populated.
    pub fn from_batched_domains(batched: &BatchedDomains) -> Option<Self> {
        let masks = batched.unstable_masks()?;
        let layer_lowers = batched.layer_lowers();
        let layer_uppers = batched.layer_uppers();
        let batch_size = batched.batch_size();

        let mut bounds = HashMap::new();

        for layer_name in masks.keys() {
            let Some(full_lower) = layer_lowers.get(layer_name) else {
                continue;
            };
            let Some(full_upper) = layer_uppers.get(layer_name) else {
                continue;
            };

            // Get indices of unstable neurons using consistent helper
            let Some(indices) = batched.sparse_to_dense_indices(layer_name) else {
                continue;
            };

            if indices.is_empty() {
                continue;
            }

            let num_unstable = indices.len();
            let full_lower_arr = full_lower.as_array();
            let full_upper_arr = full_upper.as_array();

            // Extract sparse bounds: [batch, num_unstable]
            let mut sparse_lower = Array2::<f32>::zeros((batch_size, num_unstable));
            let mut sparse_upper = Array2::<f32>::zeros((batch_size, num_unstable));

            // Flatten for indexing (assumes bounds are [batch, ...features...])
            let feature_size = checked_shape_product(&full_lower_arr.shape()[1..])
                .expect("sparse_bounds: feature shape product overflows");
            let flat_lower = contiguous_flat_slice(full_lower_arr);
            let flat_upper = contiguous_flat_slice(full_upper_arr);

            for batch_idx in 0..batch_size {
                for (sparse_idx, &dense_idx) in indices.iter().enumerate() {
                    if dense_idx < feature_size {
                        // Index into flattened feature dimension
                        let offset = batch_idx * feature_size + dense_idx;
                        if offset < flat_lower.len() {
                            sparse_lower[[batch_idx, sparse_idx]] = flat_lower[offset];
                            sparse_upper[[batch_idx, sparse_idx]] = flat_upper[offset];
                        }
                    }
                }
            }

            bounds.insert(layer_name.clone(), (sparse_lower, sparse_upper));
        }

        Some(Self { bounds })
    }

    /// Get sparse bounds for a specific layer.
    ///
    /// Returns (lower, upper) arrays of shape `[batch_size, num_unstable]`.
    pub fn layer_bounds(&self, layer: &str) -> Option<(&Array2<f32>, &Array2<f32>)> {
        self.bounds.get(layer).map(|(l, u)| (l, u))
    }

    /// Merge sparse bounds back into full batched domain bounds.
    ///
    /// Uses `max(full_lower, sparse_lower)` and `min(full_upper, sparse_upper)`
    /// to get tighter bounds after child domain processing.
    ///
    /// # Arguments
    /// * `batched` - The batched domains to merge into (must have unstable_masks populated)
    ///
    /// # Returns
    /// * `Ok(num_updated)` - Number of bound values updated
    /// * `Err(_)` - If unstable masks are not populated
    pub fn merge_into(&self, batched: &mut BatchedDomains) -> Result<usize, NyError> {
        // Collect indices using existing sparse_to_dense_indices helper
        // This must happen before mutable borrows of layer bounds
        let indices_by_layer: HashMap<String, Vec<usize>> = {
            if batched.unstable_masks().is_none() {
                return Err(NyError::InvalidSpec(
                    "Cannot merge sparse bounds: unstable_masks not populated".to_string(),
                ));
            }

            self.bounds
                .keys()
                .filter_map(|layer_name| {
                    batched
                        .sparse_to_dense_indices(layer_name)
                        .map(|indices| (layer_name.clone(), indices))
                })
                .collect()
        };

        let mut num_updated = 0;

        for (layer_name, (sparse_lower, sparse_upper)) in &self.bounds {
            let Some(indices) = indices_by_layer.get(layer_name) else {
                continue;
            };
            if indices.is_empty() {
                continue;
            }

            let Some(full_lower) = batched.layer_lowers_mut().get_mut(layer_name) else {
                continue;
            };
            let feature_size = checked_shape_product(&full_lower.as_array().shape()[1..])
                .expect("sparse_bounds merge: feature shape product overflows");
            let batch_size = sparse_lower.shape()[0];

            // Merge lower bounds (tighten via max)
            let lower_data = get_contiguous_slice_mut(full_lower, layer_name)?;
            num_updated += scatter_merge(
                lower_data,
                sparse_lower,
                indices,
                feature_size,
                batch_size,
                true,
            );

            let Some(full_upper) = batched.layer_uppers_mut().get_mut(layer_name) else {
                continue;
            };

            // Merge upper bounds (tighten via min)
            let upper_data = get_contiguous_slice_mut(full_upper, layer_name)?;
            num_updated += scatter_merge(
                upper_data,
                sparse_upper,
                indices,
                feature_size,
                batch_size,
                false,
            );
        }

        Ok(num_updated)
    }
}

#[cfg(test)]
impl Default for SparseIntermediateBounds {
    fn default() -> Self {
        Self::new()
    }
}

/// Get a contiguous mutable slice from a PooledArray, returning an error if non-contiguous.
#[cfg(test)]
fn get_contiguous_slice_mut<'a>(
    array: &'a mut PooledArray,
    layer_name: &str,
) -> Result<&'a mut [f32], NyError> {
    contiguous_flat_slice_mut(array.as_array_mut()).map_err(|_| {
        NyError::InternalError(format!(
            "merge_sparse_to_batched: could not normalize bounds for layer '{layer_name}'"
        ))
    })
}

/// Scatter-merge sparse bounds into a flat buffer.
///
/// If `tighten_lower` is true, updates where sparse > full (lower bound tightening).
/// If false, updates where sparse < full (upper bound tightening).
/// Returns the number of values updated.
#[cfg(test)]
fn scatter_merge(
    full_data: &mut [f32],
    sparse: &Array2<f32>,
    indices: &[usize],
    feature_size: usize,
    batch_size: usize,
    tighten_lower: bool,
) -> usize {
    let mut count = 0;
    for batch_idx in 0..batch_size {
        for (sparse_idx, &dense_idx) in indices.iter().enumerate() {
            if dense_idx < feature_size {
                let offset = batch_idx * feature_size + dense_idx;
                if offset < full_data.len() {
                    let sparse_val = sparse[[batch_idx, sparse_idx]];
                    let should_update = if tighten_lower {
                        sparse_val > full_data[offset]
                    } else {
                        sparse_val < full_data[offset]
                    };
                    if should_update {
                        full_data[offset] = sparse_val;
                        count += 1;
                    }
                }
            }
        }
    }
    count
}
