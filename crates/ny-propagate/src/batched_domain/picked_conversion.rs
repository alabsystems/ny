// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conversion from `PickedDomains` to `BatchedDomains`.
//!
//! Extracted from `types.rs` to stay within the 500-line file limit.
//! Provides the GPU-friendly path: `PickedDomains` already has batched
//! arrays, so this wraps them into `PooledArray` without re-stacking.

use super::domain_list::PickedDomains;
use super::options::BatchedDomainOptions;
use super::types::BatchedDomains;
use super::utils::{slice_batch_dim, unstable_mask_from_bounds};
use super::ConstraintTuple;
use ny_tensor::PooledArray;
use std::collections::HashMap;

impl BatchedDomains {
    /// Create a BatchedDomains directly from PickedDomains with default options.
    ///
    /// Convenience wrapper around `from_picked_domains_with_options` using
    /// `BatchedDomainOptions::default()` (no intermediate transfer).
    ///
    /// # Reference
    /// Design: `designs/2026-02-03-batched-domain-pickout-gpu-transfer.md`
    /// Alpha-beta-CROWN: `complete_verifier/branching_domains.py`:270-305
    pub fn from_picked_domains(picked: PickedDomains) -> Self {
        Self::from_picked_domains_with_options(picked, BatchedDomainOptions::default())
    }

    /// Create a BatchedDomains directly from PickedDomains with options.
    ///
    /// This is the GPU-friendly conversion path: `PickedDomains` already has batched
    /// `ArrayD<f32>` arrays, so this wraps them into `PooledArray` without re-stacking.
    /// Use this for efficient pick_out -> GPU transfer instead of going through
    /// `BatchedDomainsBuilder`.
    ///
    /// When `options.enable_interm_transfer` is true, computes static intermediate
    /// bounds from the first domain in the batch (index 0) and derives unstable masks.
    /// This matches the builder's behavior in `BatchedDomainsBuilder::build()`.
    ///
    /// # Arguments
    /// * `picked` - The batched arrays from `DomainList::pick_out`
    /// * `options` - Options controlling intermediate bound transfer
    ///
    /// # Reference
    /// Design: `designs/2026-02-03-batched-domain-pickout-gpu-transfer.md`
    /// Alpha-beta-CROWN: `complete_verifier/branching_domains.py`:270-305
    /// Fix: #1655 — enable_interm_transfer was a no-op on the pick_out path
    pub fn from_picked_domains_with_options(
        picked: PickedDomains,
        options: BatchedDomainOptions,
    ) -> Self {
        let batch_size = picked.batch_size;
        if batch_size == 0 {
            let layer_names: Vec<String> = picked.layer_lowers.keys().cloned().collect();
            return Self::with_capacity(0, &layer_names);
        }

        // Wrap batched ArrayD<f32> into PooledArray without re-stacking
        let layer_lowers: HashMap<String, PooledArray> = picked
            .layer_lowers
            .into_iter()
            .map(|(name, arr)| (name, PooledArray::from_array(arr)))
            .collect();
        let layer_uppers: HashMap<String, PooledArray> = picked
            .layer_uppers
            .into_iter()
            .map(|(name, arr)| (name, PooledArray::from_array(arr)))
            .collect();
        let input_lowers = PooledArray::from_array(picked.input_lowers);
        let input_uppers = PooledArray::from_array(picked.input_uppers);

        // Copy per-domain scalars from metadata
        let lower_bounds: Vec<f32> = picked.metadata.iter().map(|m| m.lower_bound).collect();
        let upper_bounds: Vec<f32> = picked.metadata.iter().map(|m| m.upper_bound).collect();
        let depths: Vec<usize> = picked.metadata.iter().map(|m| m.depth).collect();
        let constraints: Vec<Vec<ConstraintTuple>> = picked
            .metadata
            .iter()
            .map(|m| m.constraints.clone())
            .collect();

        // Compute static bounds for interm_transfer from batch index 0.
        // Matches BatchedDomainsBuilder::build() lines 193-241: slice the first
        // domain's layer bounds as reference static bounds, then derive unstable
        // masks (lower < 0 < upper) for sparse storage.
        let (static_layer_lowers, static_layer_uppers, unstable_masks) =
            if options.enable_interm_transfer {
                compute_static_bounds(&layer_lowers, &layer_uppers)
            } else {
                (None, None, None)
            };

        Self {
            batch_size,
            layer_lowers,
            layer_uppers,
            input_lowers,
            input_uppers,
            static_layer_lowers,
            static_layer_uppers,
            unstable_masks,
            lower_bounds,
            upper_bounds,
            depths,
            constraints,
        }
    }
}

/// Static bounds tuple: (static_lowers, static_uppers, unstable_masks).
type StaticBounds = (
    Option<HashMap<String, PooledArray>>,
    Option<HashMap<String, PooledArray>>,
    Option<HashMap<String, ndarray::ArrayD<bool>>>,
);

/// Compute static intermediate bounds and unstable masks from batch index 0.
fn compute_static_bounds(
    layer_lowers: &HashMap<String, PooledArray>,
    layer_uppers: &HashMap<String, PooledArray>,
) -> StaticBounds {
    let mut static_lowers = HashMap::new();
    let mut static_uppers = HashMap::new();
    let mut masks = HashMap::new();

    for (name, pooled_lower) in layer_lowers {
        if let Some(pooled_upper) = layer_uppers.get(name) {
            let lower_arr = pooled_lower.as_array();
            let upper_arr = pooled_upper.as_array();

            if let (Some(lower_slice), Some(upper_slice)) =
                (slice_batch_dim(lower_arr, 0), slice_batch_dim(upper_arr, 0))
            {
                if let Ok(mask) = unstable_mask_from_bounds(&lower_slice, &upper_slice) {
                    static_lowers.insert(name.clone(), PooledArray::from_array(lower_slice));
                    static_uppers.insert(name.clone(), PooledArray::from_array(upper_slice));
                    masks.insert(name.clone(), mask);
                }
            }
        }
    }

    (Some(static_lowers), Some(static_uppers), Some(masks))
}
