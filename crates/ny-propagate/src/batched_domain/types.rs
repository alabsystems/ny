// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::builder::BatchedDomainsBuilder;
use super::options::BatchedDomainOptions;
use super::utils::slice_batch_dim;
use super::ConstraintTuple;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::GraphBabDomain;
use ndarray::ArrayD;
use ndarray::Axis;
use ny_core::{NyError, Result};
use ny_tensor::{BoundedTensor, PooledArray};
use std::collections::HashMap;

/// Batched representation of multiple BaB domains for GPU processing.
///
/// Stores bounds as `[batch, ...original_shape]` tensors.
/// Designed for efficient GPU transfer and parallel CROWN computation.
///
/// Uses pooled CPU storage for large batched tensors to reduce allocation churn
/// between batches (TensorStorage-style pooling from alpha-beta-CROWN).
///
/// # Example
///
/// Doctests must import via `ny_propagate::...` (not `crate::...`).
///
/// ```rust,no_run
/// # use ny_propagate::beta_crown::{BatchedDomains, GraphBabDomain};
/// # let domains: Vec<GraphBabDomain> = Vec::new();
/// # let layer_names: Vec<String> = Vec::new();
/// // BatchedDomains packs multiple BaB domains for GPU processing:
/// let domain_refs: Vec<&GraphBabDomain> = domains.iter().collect();
/// let batched = BatchedDomains::from_graph_domains(&domain_refs, &layer_names).unwrap();
/// let (input_lowers, input_uppers) = batched.input_bounds_batched();
/// let batch_size = batched.batch_size();
/// # let _ = (input_lowers, input_uppers, batch_size);
/// ```
#[derive(Debug, Clone)]
pub struct BatchedDomains {
    /// Number of domains in this batch.
    pub(crate) batch_size: usize,

    /// Per-layer lower bounds: layer_name -> [batch, *shape].
    /// Shape varies by layer (e.g., [batch, hidden_dim] for linear layers).
    pub(crate) layer_lowers: HashMap<String, PooledArray>,

    /// Per-layer upper bounds: layer_name -> [batch, *shape].
    /// Shape varies by layer (e.g., [batch, hidden_dim] for linear layers).
    pub(crate) layer_uppers: HashMap<String, PooledArray>,

    /// Input lower bounds: [batch, *input_shape].
    pub(crate) input_lowers: PooledArray,

    /// Input upper bounds: [batch, *input_shape].
    pub(crate) input_uppers: PooledArray,

    /// Static intermediate lower bounds (no batch dim) for bound transfer.
    /// Only populated when interm_transfer is enabled.
    pub(crate) static_layer_lowers: Option<HashMap<String, PooledArray>>,

    /// Static intermediate upper bounds (no batch dim) for bound transfer.
    /// Only populated when interm_transfer is enabled.
    pub(crate) static_layer_uppers: Option<HashMap<String, PooledArray>>,

    /// Unstable masks derived from static bounds (lower < 0 < upper).
    /// Only populated when interm_transfer is enabled.
    pub(crate) unstable_masks: Option<HashMap<String, ArrayD<bool>>>,

    /// Lower bound on objective per domain: [batch].
    pub(crate) lower_bounds: Vec<f32>,

    /// Upper bound on objective per domain: [batch].
    pub(crate) upper_bounds: Vec<f32>,

    /// Depth (number of splits) per domain: [batch].
    pub(crate) depths: Vec<usize>,

    /// Constraint history per domain.
    /// Each entry is a list of (node_name, neuron_idx, is_active, split_point) tuples.
    /// Stored as CPU data since constraints are sparse and variable-length.
    /// - ReLU constraints: split_point = None
    /// - GenBaB constraints: split_point = Some(pt)
    pub(crate) constraints: Vec<Vec<ConstraintTuple>>,
}

impl BatchedDomains {
    /// Create an empty batch with pre-allocated capacity.
    pub fn with_capacity(capacity: usize, layer_names: &[String]) -> Self {
        Self {
            batch_size: 0,
            layer_lowers: layer_names
                .iter()
                .map(|n| (n.clone(), PooledArray::empty()))
                .collect(),
            layer_uppers: layer_names
                .iter()
                .map(|n| (n.clone(), PooledArray::empty()))
                .collect(),
            input_lowers: PooledArray::empty(),
            input_uppers: PooledArray::empty(),
            static_layer_lowers: None,
            static_layer_uppers: None,
            unstable_masks: None,
            lower_bounds: Vec::with_capacity(capacity),
            upper_bounds: Vec::with_capacity(capacity),
            depths: Vec::with_capacity(capacity),
            constraints: Vec::with_capacity(capacity),
        }
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    /// Return the number of domains in this batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Access per-layer lower bounds for the batch.
    pub fn layer_lowers(&self) -> &HashMap<String, PooledArray> {
        &self.layer_lowers
    }

    /// Mutable access to per-layer lower bounds for the batch.
    pub fn layer_lowers_mut(&mut self) -> &mut HashMap<String, PooledArray> {
        &mut self.layer_lowers
    }

    /// Access per-layer upper bounds for the batch.
    pub fn layer_uppers(&self) -> &HashMap<String, PooledArray> {
        &self.layer_uppers
    }

    /// Mutable access to per-layer upper bounds for the batch.
    pub fn layer_uppers_mut(&mut self) -> &mut HashMap<String, PooledArray> {
        &mut self.layer_uppers
    }

    /// Access the batched input lower bounds.
    pub fn input_lowers(&self) -> &PooledArray {
        &self.input_lowers
    }

    /// Access the batched input upper bounds.
    pub fn input_uppers(&self) -> &PooledArray {
        &self.input_uppers
    }

    /// Access the batched input bounds as a pair.
    pub fn input_bounds_batched(&self) -> (&PooledArray, &PooledArray) {
        (&self.input_lowers, &self.input_uppers)
    }

    /// Access static intermediate lower bounds, if present.
    pub fn static_layer_lowers(&self) -> Option<&HashMap<String, PooledArray>> {
        self.static_layer_lowers.as_ref()
    }

    /// Access static intermediate upper bounds, if present.
    pub fn static_layer_uppers(&self) -> Option<&HashMap<String, PooledArray>> {
        self.static_layer_uppers.as_ref()
    }

    /// Access unstable masks derived from static bounds, if present.
    pub fn unstable_masks(&self) -> Option<&HashMap<String, ArrayD<bool>>> {
        self.unstable_masks.as_ref()
    }

    /// Index mapping: unstable index -> full tensor index for a layer.
    ///
    /// Returns indices for neurons that are "unstable" (crossing bounds: lb < 0 < ub).
    /// This enables sparse storage: only unstable neurons need per-domain tracking.
    ///
    /// # Example
    /// For a layer with 5 neurons where neurons 1 and 3 are unstable:
    /// - Returns `Some(vec![1, 3])`
    /// - Sparse index 0 -> dense index 1
    /// - Sparse index 1 -> dense index 3
    pub fn sparse_to_dense_indices(&self, layer: &str) -> Option<Vec<usize>> {
        self.unstable_masks.as_ref().and_then(|masks| {
            masks.get(layer).map(|mask| {
                mask.iter()
                    .enumerate()
                    .filter_map(|(i, &is_unstable)| if is_unstable { Some(i) } else { None })
                    .collect()
            })
        })
    }

    /// Count the number of unstable neurons in a layer.
    ///
    /// Returns `None` if unstable masks are not populated.
    pub fn unstable_count(&self, layer: &str) -> Option<usize> {
        self.unstable_masks.as_ref().and_then(|masks| {
            masks
                .get(layer)
                .map(|mask| mask.iter().filter(|&&x| x).count())
        })
    }

    /// Check if a specific neuron is unstable in a layer.
    ///
    /// Returns `None` if unstable masks are not populated or the layer doesn't exist.
    pub fn is_neuron_unstable(&self, layer: &str, neuron_idx: usize) -> Option<bool> {
        self.unstable_masks.as_ref().and_then(|masks| {
            masks
                .get(layer)
                .and_then(|mask| mask.iter().nth(neuron_idx).copied())
        })
    }

    /// Access per-domain objective lower bounds.
    pub fn lower_bounds(&self) -> &[f32] {
        &self.lower_bounds
    }

    /// Access per-domain objective upper bounds.
    pub fn upper_bounds(&self) -> &[f32] {
        &self.upper_bounds
    }

    /// Access per-domain split depths.
    pub fn depths(&self) -> &[usize] {
        &self.depths
    }

    /// Access per-domain constraint histories.
    ///
    /// Each constraint is `(node_name, neuron_idx, is_active, split_point)`:
    /// - ReLU constraints: split_point = None
    /// - GenBaB constraints: split_point = Some(pt)
    pub fn constraints(&self) -> &[Vec<ConstraintTuple>] {
        &self.constraints
    }

    /// Build input bounds for a single domain from the batched input tensors.
    pub fn input_bounds_at(&self, domain_idx: usize) -> Result<BoundedTensor> {
        if domain_idx >= self.batch_size {
            return Err(NyError::InvalidSpec(format!(
                "BatchedDomains index {} out of range (batch_size {})",
                domain_idx, self.batch_size
            )));
        }

        let lowers = self.input_lowers.as_array();
        let uppers = self.input_uppers.as_array();
        let lower_batch = lowers.shape().first().copied().unwrap_or(0);
        let upper_batch = uppers.shape().first().copied().unwrap_or(0);
        if lower_batch != self.batch_size || upper_batch != self.batch_size {
            return Err(NyError::InvalidSpec(format!(
                "BatchedDomains input batch mismatch: batch_size={}, input_lowers[0]={}, input_uppers[0]={}",
                self.batch_size, lower_batch, upper_batch
            )));
        }

        let lower = self
            .input_lowers
            .as_array()
            .index_axis(Axis(0), domain_idx)
            .to_owned()
            .into_dyn();
        let upper = self
            .input_uppers
            .as_array()
            .index_axis(Axis(0), domain_idx)
            .to_owned()
            .into_dyn();

        BoundedTensor::new(lower, upper)
    }

    /// Get the number of domains in this batch.
    pub fn len(&self) -> usize {
        self.batch_size
    }

    /// Create batched domains from a slice of GraphBabDomain references.
    ///
    /// This is the primary conversion method for integrating with the BaB verifier.
    /// It extracts bounds from each domain and stacks them along the batch dimension.
    ///
    /// # Arguments
    /// * `domains` - Slice of domain references to batch together
    /// * `layer_names` - Names of layers to extract bounds for (typically ReLU layers)
    ///
    /// # Example
    ///
    /// Doctests must import via `ny_propagate::...` (not `crate::...`).
    ///
    /// ```rust,no_run
    /// # use ny_propagate::beta_crown::{BatchedDomains, GraphBabDomain};
    /// # let domains: Vec<GraphBabDomain> = Vec::new();
    /// # let layer_names: Vec<String> = Vec::new();
    /// // Convert BaB domains to batched representation:
    /// let domain_refs: Vec<&GraphBabDomain> = domains.iter().collect();
    /// let batched = BatchedDomains::from_graph_domains(&domain_refs, &layer_names).unwrap();
    /// # let _ = batched;
    /// ```
    pub fn from_graph_domains(domains: &[&GraphBabDomain], layer_names: &[String]) -> Result<Self> {
        Self::from_graph_domains_with_options(domains, layer_names, BatchedDomainOptions::default())
    }

    /// Create batched domains from GraphBabDomain references with options.
    pub fn from_graph_domains_with_options(
        domains: &[&GraphBabDomain],
        layer_names: &[String],
        options: BatchedDomainOptions,
    ) -> Result<Self> {
        if domains.is_empty() {
            return Ok(Self::with_capacity(0, layer_names));
        }

        let mut builder = BatchedDomainsBuilder::new_with_options(layer_names.to_vec(), options);

        for domain in domains {
            // #stack-double-copy: BORROW each layer's bounds rather than cloning
            // them into a temporary map the builder immediately clones again and
            // then drops. Only the builder's copy is retained, so the first clone
            // was pure waste — paid per layer per domain, on a path the kFSB
            // simulation walks for every candidate split rather than only the
            // committed one. Byte-identical output, one clone earlier.
            let mut layer_bounds: HashMap<&str, (&ArrayD<f32>, &ArrayD<f32>)> =
                HashMap::with_capacity(layer_names.len());
            for name in layer_names {
                let bounded = domain.node_bounds.get(name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Missing bounds for layer '{}' in GraphBabDomain",
                        name
                    ))
                })?;
                layer_bounds.insert(name.as_str(), (bounded.lower(), bounded.upper()));
            }

            // Extract input bounds
            let input_lower = domain.input_bounds.lower().clone();
            let input_upper = domain.input_bounds.upper().clone();

            // Extract constraints from history, interleaving ReLU and GenBaB by split order
            let constraints = serialize_constraints(&domain.history)?;

            builder.add_domain_borrowed(
                &layer_bounds,
                input_lower,
                input_upper,
                domain.lower_bound,
                domain.upper_bound,
                domain.depth,
                constraints,
            );
        }

        builder.build()
    }

    /// Extract domain updates from batched bounds after GPU processing.
    ///
    /// This creates minimal updates that can be applied back to the original domains.
    /// Only includes bounds that changed during GPU processing.
    pub fn extract_updates(
        &self,
        new_lower_bounds: &[f32],
        new_upper_bounds: &[f32],
    ) -> Result<Vec<DomainUpdate>> {
        self.extract_updates_with_layer_bounds(new_lower_bounds, new_upper_bounds, None, None)
    }

    /// Extract domain updates including updated layer bounds from GPU processing.
    /// Layer bounds are sliced from batched `[batch, *shape]` to individual `[*shape]`.
    pub fn extract_updates_with_layer_bounds(
        &self,
        new_lower_bounds: &[f32],
        new_upper_bounds: &[f32],
        new_layer_lowers: Option<&HashMap<String, ArrayD<f32>>>,
        new_layer_uppers: Option<&HashMap<String, ArrayD<f32>>>,
    ) -> Result<Vec<DomainUpdate>> {
        if new_lower_bounds.len() != self.batch_size || new_upper_bounds.len() != self.batch_size {
            return Err(NyError::InternalError(format!(
                "extract_updates: bounds len ({}/{}) != batch_size {}",
                new_lower_bounds.len(),
                new_upper_bounds.len(),
                self.batch_size,
            )));
        }
        (0..self.batch_size)
            .map(|idx| {
                let layer_bounds = match (new_layer_lowers, new_layer_uppers) {
                    (Some(lowers), Some(uppers)) => {
                        self.slice_layer_bounds_for_domain(idx, lowers, uppers)
                    }
                    _ => HashMap::new(),
                };

                Ok(DomainUpdate {
                    domain_idx: idx,
                    new_lower_bound: new_lower_bounds[idx],
                    new_upper_bound: new_upper_bounds[idx],
                    new_layer_bounds: layer_bounds,
                })
            })
            .collect()
    }

    /// Slice batched layer bounds for a single domain.
    ///
    /// Takes arrays of shape [batch, *layer_shape] and extracts [*layer_shape] for domain idx.
    fn slice_layer_bounds_for_domain(
        &self,
        domain_idx: usize,
        lowers: &HashMap<String, ArrayD<f32>>,
        uppers: &HashMap<String, ArrayD<f32>>,
    ) -> HashMap<String, (ArrayD<f32>, ArrayD<f32>)> {
        let mut result = HashMap::new();

        for (name, lower_batched) in lowers {
            if let Some(upper_batched) = uppers.get(name) {
                // Both lower and upper must have same shape
                if lower_batched.shape() != upper_batched.shape() {
                    continue;
                }

                // Extract slice for this domain: [batch, *shape] -> [*shape]
                if let (Some(lower_slice), Some(upper_slice)) = (
                    slice_batch_dim(lower_batched, domain_idx),
                    slice_batch_dim(upper_batched, domain_idx),
                ) {
                    result.insert(name.clone(), (lower_slice, upper_slice));
                }
            }
        }

        result
    }
}

/// Update to apply back to a domain after GPU processing.
#[derive(Debug, Clone)]
pub struct DomainUpdate {
    /// Index of the domain in the original batch.
    pub domain_idx: usize,

    /// New lower bound on objective.
    pub new_lower_bound: f32,

    /// New upper bound on objective.
    pub new_upper_bound: f32,

    /// Updated layer bounds (only layers that changed).
    pub new_layer_bounds: HashMap<String, (ArrayD<f32>, ArrayD<f32>)>,
}

/// Serialize history constraints into tuples, preserving split order.
/// GenBaB entries are interleaved with ReLU entries using `genbab_split_ids`.
fn serialize_constraints(history: &GraphSplitHistory) -> Result<Vec<ConstraintTuple>> {
    let mut result =
        Vec::with_capacity(history.constraints.len() + history.genbab_constraints.len());
    let mut relu_idx = 0;
    let mut genbab_idx = 0;

    for split_id in 0..history.split_count {
        // Check if current GenBaB constraint belongs to this split_id
        let next_genbab_split = history.genbab_split_ids.get(genbab_idx).copied();

        if next_genbab_split == Some(split_id) {
            // Emit all GenBaB constraints with this split_id (handles range splits)
            while history.genbab_split_ids.get(genbab_idx) == Some(&split_id) {
                let c = &history.genbab_constraints[genbab_idx];
                // SOUNDNESS (#mul-genbab): the 4-tuple ConstraintTuple has no slot
                // for `input_index`. A second-input MulBinary/BilinearCrown split
                // would be reconstructed as input 0 and clamp the wrong input node.
                // Reject lossy serialization so the batched/GPU path surfaces a
                // PropagationFailure (unresolved domain) instead of a silent
                // misroute. The CPU GenBaB path keeps the full history.
                if c.input_index().unwrap_or(0) != 0 {
                    return Err(NyError::InvalidSpec(format!(
                        "serialize_constraints: GenBaB constraint on '{}' targets \
                         input_index={:?}, which ConstraintTuple cannot represent — \
                         refusing lossy serialization (#mul-genbab)",
                        c.node_name, c.input_index,
                    )));
                }
                result.push((
                    c.node_name.clone(),
                    c.neuron_idx,
                    c.is_upper_branch,
                    Some(c.split_point),
                ));
                genbab_idx += 1;
            }
        } else {
            // This split_id is a ReLU constraint
            if relu_idx < history.constraints.len() {
                let c = &history.constraints[relu_idx];
                result.push((c.node_name.clone(), c.neuron_idx, c.is_active, None));
                relu_idx += 1;
            }
        }
    }

    // Soundness guard (#2248): reject histories where split_count under-consumes constraints.
    if relu_idx != history.constraints.len() || genbab_idx != history.genbab_constraints.len() {
        return Err(NyError::InvalidSpec(format!(
            "serialize_constraints: not all constraints consumed (split_count={}, relu={}/{}, genbab={}/{})",
            history.split_count,
            relu_idx,
            history.constraints.len(),
            genbab_idx,
            history.genbab_constraints.len(),
        )));
    }

    Ok(result)
}
