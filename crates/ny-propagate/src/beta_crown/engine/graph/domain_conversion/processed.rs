// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conversion from GraphBabDomains to ProcessedDomains for DomainList insertion.
//!
//! These functions convert child domains (after CROWN propagation and branching)
//! back into the batched `ProcessedDomains` format used by `DomainList`.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{ArrayD, Axis};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::{
    BatchedDomainOptions, BatchedDomains, CachedLinearBounds, DomainMetadata, ProcessedDomains,
};
use crate::beta_crown::GraphBabDomain;

use super::history::constraints_from_history;

/// Convert GraphBabDomains to ProcessedDomains with optional cached lA.
///
/// Like `processed_from_graph_domains` but allows passing cached linear bounds
/// (lA matrices) captured during the backward pass. These are stored in
/// `DomainMetadata::cached_la` for reuse in subsequent backward passes.
///
/// **Note:** This function routes through `BatchedDomains::from_graph_domains_with_options`,
/// which stacks per-domain tensors into batched format via the builder. For the GPU BaB
/// hot path where `enable_interm_transfer` is not needed, prefer
/// `processed_from_graph_domains_direct` which stacks tensors directly without the
/// `BatchedDomains` intermediate.
///
/// # Arguments
/// * `domains` - The child domains to convert
/// * `layer_names` - Ordered list of layer names
/// * `enable_interm_transfer` - Whether to include intermediate bounds
/// * `cached_la` - Optional cached linear bounds per domain
///
/// # Reference
/// Issue: #1564 (lA matrix caching)
pub fn processed_from_graph_domains_with_la(
    domains: &[GraphBabDomain],
    layer_names: &[String],
    enable_interm_transfer: bool,
    cached_la: Option<Vec<Arc<CachedLinearBounds>>>,
) -> Result<ProcessedDomains> {
    // For the common case where interm_transfer is not needed, use the direct path
    // which avoids the BatchedDomains intermediate and its extra clone + PooledArray
    // wrapping/unwrapping cycle.
    if !enable_interm_transfer {
        return processed_from_graph_domains_direct(domains, layer_names, cached_la);
    }

    if domains.is_empty() {
        return Ok(ProcessedDomains::empty());
    }

    let domain_refs: Vec<&GraphBabDomain> = domains.iter().collect();
    let batched = BatchedDomains::from_graph_domains_with_options(
        &domain_refs,
        layer_names,
        BatchedDomainOptions {
            enable_interm_transfer,
        },
    )?;

    let layer_lowers = batched
        .layer_lowers()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_array().clone()))
        .collect();
    let layer_uppers = batched
        .layer_uppers()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_array().clone()))
        .collect();

    let input_lowers = batched.input_lowers().as_array().clone();
    let input_uppers = batched.input_uppers().as_array().clone();

    // Structural invariant: parallel arrays must match batch_size. (#2637)
    let bs = batched.batch_size();
    debug_assert_eq!(batched.lower_bounds().len(), bs);
    debug_assert_eq!(batched.upper_bounds().len(), bs);
    debug_assert_eq!(batched.depths().len(), bs);
    debug_assert_eq!(batched.constraints().len(), bs);
    debug_assert_eq!(domains.len(), bs);
    if let Some(ref la) = cached_la {
        debug_assert!(la.len() >= bs);
    }

    let metadata: Vec<DomainMetadata> = (0..batched.batch_size())
        .map(|i| {
            DomainMetadata::new(
                batched.lower_bounds()[i],
                batched.upper_bounds()[i],
                batched.depths()[i],
                batched.constraints()[i].clone(),
                cached_la.as_ref().and_then(|v| v.get(i).cloned()),
                // Preserve alpha state from source domains. BatchedDomains strips
                // alpha during tensor stacking, so we read it directly.
                // Issue: #1845
                if domains[i].alpha_state.is_empty() {
                    None
                } else {
                    Some(domains[i].alpha_state.clone())
                },
            )
        })
        .collect::<Result<Vec<DomainMetadata>>>()?;

    let keep_mask = vec![true; batched.batch_size()];

    Ok(ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers,
        input_uppers,
        global_lbs: batched.lower_bounds().to_vec(),
        global_ubs: batched.upper_bounds().to_vec(),
        metadata,
        keep_mask,
    })
}

/// Convert GraphBabDomains to ProcessedDomains directly, bypassing BatchedDomains.
///
/// Direction 3 of #1668: This builds `ProcessedDomains` by stacking per-domain
/// tensors directly into batched `ArrayD<f32>` arrays, avoiding the
/// `BatchedDomains::from_graph_domains_with_options` → extract → clone round-trip.
///
/// The old path was:
///   GraphBabDomain[] → builder.add_domain() (clones each tensor)
///     → builder.build() (stacks into PooledArray)
///     → extract as ArrayD clones
///
/// The new path is:
///   GraphBabDomain[] → ndarray::stack (one copy per tensor, directly into ArrayD)
///
/// This eliminates one full copy pass per BaB iteration for all layer bounds,
/// input bounds, and metadata.
///
/// # Arguments
/// * `domains` - The child domains to convert
/// * `layer_names` - Ordered list of layer names
/// * `cached_la` - Optional cached linear bounds per domain
///
/// # Reference
/// Issue: #1668 (zero-copy domain flow)
/// Design: `designs/2026-02-07-gpu-bab-zero-copy-domain-flow.md` Direction 3
pub fn processed_from_graph_domains_direct(
    domains: &[GraphBabDomain],
    layer_names: &[String],
    cached_la: Option<Vec<Arc<CachedLinearBounds>>>,
) -> Result<ProcessedDomains> {
    if domains.is_empty() {
        return Ok(ProcessedDomains::empty());
    }

    let batch_size = domains.len();

    // Stack layer bounds directly: for each layer, collect views and stack along axis 0.
    let mut layer_lowers: HashMap<String, ArrayD<f32>> = HashMap::new();
    let mut layer_uppers: HashMap<String, ArrayD<f32>> = HashMap::new();

    for name in layer_names {
        // Collect views of each domain's bounds for this layer
        let lower_views: Vec<_> = domains
            .iter()
            .filter_map(|d| {
                d.node_bounds
                    .get(name)
                    .map(|b| b.lower().view().insert_axis(Axis(0)))
            })
            .collect();
        let upper_views: Vec<_> = domains
            .iter()
            .filter_map(|d| {
                d.node_bounds
                    .get(name)
                    .map(|b| b.upper().view().insert_axis(Axis(0)))
            })
            .collect();

        if lower_views.len() != batch_size || upper_views.len() != batch_size {
            return Err(NyError::InvalidSpec(format!(
                "Missing bounds for layer '{}': expected {} domains, got {} lower / {} upper",
                name,
                batch_size,
                lower_views.len(),
                upper_views.len()
            )));
        }

        let stacked_lower = ndarray::concatenate(Axis(0), &lower_views).map_err(|e| {
            NyError::InvalidSpec(format!(
                "Failed to stack lower bounds for layer '{}': {}",
                name, e
            ))
        })?;
        let stacked_upper = ndarray::concatenate(Axis(0), &upper_views).map_err(|e| {
            NyError::InvalidSpec(format!(
                "Failed to stack upper bounds for layer '{}': {}",
                name, e
            ))
        })?;

        layer_lowers.insert(name.clone(), stacked_lower);
        layer_uppers.insert(name.clone(), stacked_upper);
    }

    // Stack input bounds directly
    let input_lower_views: Vec<_> = domains
        .iter()
        .map(|d| d.input_bounds.lower().view().insert_axis(Axis(0)))
        .collect();
    let input_upper_views: Vec<_> = domains
        .iter()
        .map(|d| d.input_bounds.upper().view().insert_axis(Axis(0)))
        .collect();

    let input_lowers = ndarray::concatenate(Axis(0), &input_lower_views)
        .map_err(|e| NyError::InvalidSpec(format!("Failed to stack input lower bounds: {}", e)))?;
    let input_uppers = ndarray::concatenate(Axis(0), &input_upper_views)
        .map_err(|e| NyError::InvalidSpec(format!("Failed to stack input upper bounds: {}", e)))?;

    // Collect scalar bounds and metadata directly
    let global_lbs: Vec<f32> = domains.iter().map(|d| d.lower_bound).collect();
    let global_ubs: Vec<f32> = domains.iter().map(|d| d.upper_bound).collect();

    // Structural invariant: cached_la must cover batch_size. (#2637)
    if let Some(ref la) = cached_la {
        debug_assert!(la.len() >= batch_size);
    }

    let metadata: Vec<DomainMetadata> = domains
        .iter()
        .enumerate()
        .map(|(i, d)| {
            DomainMetadata::new(
                d.lower_bound,
                d.upper_bound,
                d.depth,
                constraints_from_history(&d.history)?,
                cached_la.as_ref().and_then(|v| v.get(i).cloned()),
                if d.alpha_state.is_empty() {
                    None
                } else {
                    Some(d.alpha_state.clone())
                },
            )
        })
        .collect::<Result<Vec<DomainMetadata>>>()?;

    let keep_mask = vec![true; batch_size];

    Ok(ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers,
        input_uppers,
        global_lbs,
        global_ubs,
        metadata,
        keep_mask,
    })
}

/// Build ProcessedDomains directly from backward pass results, skipping per-domain
/// GraphBabDomain materialization on the output side.
///
/// Direction 3 extension of #1668: After the batched backward pass produces per-domain
/// `(BoundedTensor, HashMap<String, BoundedTensor>)` results, this function stacks the
/// updated node caches directly into ProcessedDomains format without writing them back
/// into individual `GraphBabDomain.node_bounds` and then re-stacking.
///
/// This eliminates one full copy pass per BaB iteration: the old path wrote backward
/// results into `child.node_bounds` (wrapping in Arc), then `processed_from_graph_domains_direct`
/// unwrapped and stacked them. This path stacks directly from the backward output.
///
/// # Arguments
/// * `node_caches` - Per-domain backward pass results: layer_name -> BoundedTensor (updated bounds)
/// * `children` - The child GraphBabDomains used for the backward pass (for metadata: history,
///   input_bounds, depth). Node bounds in these are stale (pre-backward); the `node_caches`
///   contain the updated bounds.
/// * `lower_bounds` - Updated objective lower bound per domain
/// * `upper_bounds` - Updated objective upper bound per domain
/// * `keep_mask` - Which domains survived verification/violation/depth filtering
/// * `layer_names` - Ordered list of layer names
/// * `cached_la` - Optional captured lA per domain (only for kept domains, in kept-domain order)
///
/// # Reference
/// Issue: #1668 (zero-copy domain flow)
/// Design: `designs/2026-02-07-gpu-bab-zero-copy-domain-flow.md` Direction 3
pub fn processed_from_backward_results(
    node_caches: Vec<HashMap<String, Arc<BoundedTensor>>>,
    children: &[GraphBabDomain],
    lower_bounds: &[f32],
    upper_bounds: &[f32],
    keep_mask: &[bool],
    layer_names: &[String],
    cached_la: Option<Vec<Arc<CachedLinearBounds>>>,
) -> Result<ProcessedDomains> {
    let batch_size = node_caches.len();
    if batch_size != children.len() {
        return Err(NyError::InternalError(format!(
            "children batch mismatch: expected {batch_size}, got {} (#2136)",
            children.len()
        )));
    }
    if batch_size != lower_bounds.len() {
        return Err(NyError::InternalError(format!(
            "lower_bounds batch mismatch: expected {batch_size}, got {} (#2136)",
            lower_bounds.len()
        )));
    }
    if batch_size != upper_bounds.len() {
        return Err(NyError::InternalError(format!(
            "upper_bounds batch mismatch: expected {batch_size}, got {} (#2136)",
            upper_bounds.len()
        )));
    }
    if batch_size != keep_mask.len() {
        return Err(NyError::InternalError(format!(
            "keep_mask batch mismatch: expected {batch_size}, got {} (#2136)",
            keep_mask.len()
        )));
    }

    // Count kept domains to pre-allocate
    let kept_count: usize = keep_mask.iter().filter(|&&k| k).count();
    if kept_count == 0 {
        return Ok(ProcessedDomains::empty());
    }

    // Stack layer bounds from backward results (node_caches), only for kept domains
    let mut layer_lowers_map: HashMap<String, ArrayD<f32>> = HashMap::new();
    let mut layer_uppers_map: HashMap<String, ArrayD<f32>> = HashMap::new();

    for name in layer_names {
        let lower_views: Vec<_> = node_caches
            .iter()
            .zip(keep_mask.iter())
            .filter(|(_, &kept)| kept)
            .filter_map(|(cache, _)| {
                cache
                    .get(name)
                    .map(|b| b.lower().view().insert_axis(Axis(0)))
            })
            .collect();
        let upper_views: Vec<_> = node_caches
            .iter()
            .zip(keep_mask.iter())
            .filter(|(_, &kept)| kept)
            .filter_map(|(cache, _)| {
                cache
                    .get(name)
                    .map(|b| b.upper().view().insert_axis(Axis(0)))
            })
            .collect();

        if lower_views.len() != kept_count || upper_views.len() != kept_count {
            return Err(NyError::InvalidSpec(format!(
                "processed_from_backward_results: missing bounds for layer '{}': \
                 expected {} domains, got {} lower / {} upper",
                name,
                kept_count,
                lower_views.len(),
                upper_views.len()
            )));
        }

        let stacked_lower = ndarray::concatenate(Axis(0), &lower_views).map_err(|e| {
            NyError::InvalidSpec(format!(
                "Failed to stack backward lower bounds for layer '{}': {}",
                name, e
            ))
        })?;
        let stacked_upper = ndarray::concatenate(Axis(0), &upper_views).map_err(|e| {
            NyError::InvalidSpec(format!(
                "Failed to stack backward upper bounds for layer '{}': {}",
                name, e
            ))
        })?;

        layer_lowers_map.insert(name.clone(), stacked_lower);
        layer_uppers_map.insert(name.clone(), stacked_upper);
    }

    // Stack input bounds from original children (backward pass doesn't update these)
    let input_lower_views: Vec<_> = children
        .iter()
        .zip(keep_mask.iter())
        .filter(|(_, &kept)| kept)
        .map(|(child, _)| child.input_bounds.lower().view().insert_axis(Axis(0)))
        .collect();
    let input_upper_views: Vec<_> = children
        .iter()
        .zip(keep_mask.iter())
        .filter(|(_, &kept)| kept)
        .map(|(child, _)| child.input_bounds.upper().view().insert_axis(Axis(0)))
        .collect();

    let input_lowers = ndarray::concatenate(Axis(0), &input_lower_views).map_err(|e| {
        NyError::InvalidSpec(format!(
            "Failed to stack backward input lower bounds: {}",
            e
        ))
    })?;
    let input_uppers = ndarray::concatenate(Axis(0), &input_upper_views).map_err(|e| {
        NyError::InvalidSpec(format!(
            "Failed to stack backward input upper bounds: {}",
            e
        ))
    })?;

    // Collect scalar bounds and metadata for kept domains only
    let mut global_lbs = Vec::with_capacity(kept_count);
    let mut global_ubs = Vec::with_capacity(kept_count);
    let mut metadata = Vec::with_capacity(kept_count);
    let mut la_idx = 0;

    for (i, &kept) in keep_mask.iter().enumerate() {
        if !kept {
            continue;
        }
        global_lbs.push(lower_bounds[i]);
        global_ubs.push(upper_bounds[i]);
        metadata.push(DomainMetadata::new(
            lower_bounds[i],
            upper_bounds[i],
            children[i].depth,
            constraints_from_history(&children[i].history)?,
            cached_la.as_ref().and_then(|v| v.get(la_idx).cloned()),
            if children[i].alpha_state.is_empty() {
                None
            } else {
                Some(children[i].alpha_state.clone())
            },
        )?);
        la_idx += 1;
    }

    Ok(ProcessedDomains {
        layer_lowers: layer_lowers_map,
        layer_uppers: layer_uppers_map,
        input_lowers,
        input_uppers,
        global_lbs,
        global_ubs,
        metadata,
        keep_mask: vec![true; kept_count],
    })
}
