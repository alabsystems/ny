// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Processed domains returned from GPU evaluation.
//!
//! Contains `ProcessedDomains`, the result struct for GPU-computed domain bounds.
//! Production code uses `ProcessedDomains::empty()` for empty batches;
//! test-only constructors (`from_batched_results`, `from_batched_results_with_la`)
//! are `#[cfg(test)]`.

use super::types::DomainMetadata;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

#[cfg(test)]
use super::super::types::BatchedDomains;
#[cfg(test)]
use super::types::CachedLinearBounds;
#[cfg(test)]
use std::sync::Arc;

/// Results from GPU processing to add back to the domain list.
#[derive(Debug)]
pub struct ProcessedDomains {
    /// Per-layer lower bounds after CROWN: [batch, *shape].
    pub layer_lowers: HashMap<String, ArrayD<f32>>,
    /// Per-layer upper bounds after CROWN: [batch, *shape].
    pub layer_uppers: HashMap<String, ArrayD<f32>>,
    /// Input lower bounds (possibly tightened): [batch, *input_shape].
    pub input_lowers: ArrayD<f32>,
    /// Input upper bounds (possibly tightened): [batch, *input_shape].
    pub input_uppers: ArrayD<f32>,
    /// Updated global lower bounds: [batch].
    pub global_lbs: Vec<f32>,
    /// Updated global upper bounds: [batch].
    pub global_ubs: Vec<f32>,
    /// Updated metadata (constraints, depths).
    pub metadata: Vec<DomainMetadata>,
    /// Mask of which domains to keep (not verified/infeasible).
    pub keep_mask: Vec<bool>,
}

impl ProcessedDomains {
    /// Construct an empty `ProcessedDomains` with no domains.
    ///
    /// Used when the input domain list is empty or all domains have been
    /// filtered out (verified/infeasible). Replaces repeated struct literals
    /// across `domain_conversion/processed.rs`.
    ///
    /// Part of #1860's already-landed stale-body cleanup. The current
    /// `#1860` EXECUTE design reserves Packet D for a separate frontier
    /// remeasurement, not this constructor.
    pub fn empty() -> Self {
        Self {
            layer_lowers: HashMap::new(),
            layer_uppers: HashMap::new(),
            input_lowers: ArrayD::zeros(IxDyn(&[0])),
            input_uppers: ArrayD::zeros(IxDyn(&[0])),
            global_lbs: Vec::new(),
            global_ubs: Vec::new(),
            metadata: Vec::new(),
            keep_mask: Vec::new(),
        }
    }
}

#[cfg(test)]
impl ProcessedDomains {
    /// Create a single valid domain for testing (layers: relu1[2], relu2[2], input[4]).
    ///
    /// Matches the `create_test_config()` shape used in `domain_list/tests/`.
    pub fn valid_single_domain() -> Self {
        let mut layer_lowers = HashMap::new();
        layer_lowers.insert(
            "relu1".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.1, -0.2]).unwrap(),
        );
        layer_lowers.insert(
            "relu2".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.3, -0.4]).unwrap(),
        );
        let mut layer_uppers = HashMap::new();
        layer_uppers.insert(
            "relu1".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.1, 0.2]).unwrap(),
        );
        layer_uppers.insert(
            "relu2".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.3, 0.4]).unwrap(),
        );
        Self {
            layer_lowers,
            layer_uppers,
            input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0; 4]).unwrap(),
            input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
            global_lbs: vec![-1.0],
            global_ubs: vec![1.0],
            metadata: vec![DomainMetadata {
                lower_bound: -1.0,
                upper_bound: 1.0,
                depth: 0,
                constraints: Vec::new(),
                cached_la: None,
                needs_bounding: false,
                node_bounds_override: None,
                alpha_state: None,
            }],
            keep_mask: vec![true],
        }
    }

    /// Create ProcessedDomains from BatchedDomains and GPU results.
    ///
    /// This converts the GPU-computed results back into the format expected
    /// by `DomainList::add`. Uses `BatchedDomains::extract_updates_with_layer_bounds`
    /// to slice batched arrays into per-domain updates.
    ///
    /// # Arguments
    /// * `batched` - The original BatchedDomains that was processed
    /// * `new_lower_bounds` - Updated objective lower bounds: [batch]
    /// * `new_upper_bounds` - Updated objective upper bounds: [batch]
    /// * `new_layer_lowers` - Updated layer lower bounds: layer_name -> [batch, *shape]
    /// * `new_layer_uppers` - Updated layer upper bounds: layer_name -> [batch, *shape]
    /// * `new_input_lowers` - Updated input lower bounds: [batch, *input_shape]
    /// * `new_input_uppers` - Updated input upper bounds: [batch, *input_shape]
    /// * `keep_mask` - Which domains to keep (not verified/infeasible)
    ///
    /// # Reference
    /// Design: `designs/2026-02-03-batched-domain-pickout-gpu-transfer.md`
    // Justification: Unbatching GPU results requires the batched source, per-domain output
    // bounds, per-layer bounds, updated input bounds, and keep mask — all from GPU kernel output.
    #[allow(clippy::too_many_arguments)]
    pub fn from_batched_results(
        batched: &BatchedDomains,
        new_lower_bounds: Vec<f32>,
        new_upper_bounds: Vec<f32>,
        new_layer_lowers: HashMap<String, ArrayD<f32>>,
        new_layer_uppers: HashMap<String, ArrayD<f32>>,
        new_input_lowers: ArrayD<f32>,
        new_input_uppers: ArrayD<f32>,
        keep_mask: Vec<bool>,
    ) -> ny_core::Result<Self> {
        Self::from_batched_results_with_la(
            batched,
            new_lower_bounds,
            new_upper_bounds,
            new_layer_lowers,
            new_layer_uppers,
            new_input_lowers,
            new_input_uppers,
            keep_mask,
            None,
        )
    }

    /// Create ProcessedDomains from BatchedDomains and GPU results with cached lA.
    ///
    /// This is the full version that includes cached linear bound coefficients
    /// for reuse in child domains.
    ///
    /// # Arguments
    /// * `batched` - The original BatchedDomains that was processed
    /// * `new_lower_bounds` - Updated objective lower bounds: [batch]
    /// * `new_upper_bounds` - Updated objective upper bounds: [batch]
    /// * `new_layer_lowers` - Updated layer lower bounds: layer_name -> [batch, *shape]
    /// * `new_layer_uppers` - Updated layer upper bounds: layer_name -> [batch, *shape]
    /// * `new_input_lowers` - Updated input lower bounds: [batch, *input_shape]
    /// * `new_input_uppers` - Updated input upper bounds: [batch, *input_shape]
    /// * `keep_mask` - Which domains to keep (not verified/infeasible)
    /// * `cached_la_per_domain` - Optional cached lA per domain for reuse
    ///
    /// # Reference
    /// Design: `designs/2026-02-03-batched-domain-pickout-gpu-transfer.md`
    /// Issue: #1564 (lA matrix caching)
    // Justification: Same as from_batched_results plus optional cached lA matrices per domain
    // for reuse in child domain bound computation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_batched_results_with_la(
        batched: &BatchedDomains,
        new_lower_bounds: Vec<f32>,
        new_upper_bounds: Vec<f32>,
        new_layer_lowers: HashMap<String, ArrayD<f32>>,
        new_layer_uppers: HashMap<String, ArrayD<f32>>,
        new_input_lowers: ArrayD<f32>,
        new_input_uppers: ArrayD<f32>,
        keep_mask: Vec<bool>,
        cached_la_per_domain: Option<Vec<Arc<CachedLinearBounds>>>,
    ) -> ny_core::Result<Self> {
        let batch_size = batched.batch_size();

        // Invariant: batched vectors must be at least as long as batch_size.
        // Violation means the caller passed inconsistent data — silently defaulting
        // would drop constraints or produce wrong bounds (#2226).
        assert!(
            new_lower_bounds.len() >= batch_size,
            "new_lower_bounds.len()={} < batch_size={}",
            new_lower_bounds.len(),
            batch_size,
        );
        assert!(
            new_upper_bounds.len() >= batch_size,
            "new_upper_bounds.len()={} < batch_size={}",
            new_upper_bounds.len(),
            batch_size,
        );
        assert!(
            batched.constraints().len() >= batch_size,
            "constraints.len()={} < batch_size={} — branching constraints would be silently dropped (#2226)",
            batched.constraints().len(),
            batch_size,
        );
        assert!(
            batched.depths().len() >= batch_size,
            "depths.len()={} < batch_size={} — domain depths would default to 0 (#2226)",
            batched.depths().len(),
            batch_size,
        );

        // Build updated metadata with new bounds from GPU — NaN-validated (#3125)
        let metadata: Vec<DomainMetadata> = (0..batch_size)
            .map(|i| {
                DomainMetadata::new(
                    new_lower_bounds[i],
                    new_upper_bounds[i],
                    batched.depths()[i],
                    batched.constraints()[i].clone(),
                    cached_la_per_domain
                        .as_ref()
                        .and_then(|v| v.get(i).cloned()),
                    None,
                )
            })
            .collect::<ny_core::Result<Vec<DomainMetadata>>>()?;

        Ok(Self {
            layer_lowers: new_layer_lowers,
            layer_uppers: new_layer_uppers,
            input_lowers: new_input_lowers,
            input_uppers: new_input_uppers,
            global_lbs: new_lower_bounds,
            global_ubs: new_upper_bounds,
            metadata,
            keep_mask,
        })
    }
}
