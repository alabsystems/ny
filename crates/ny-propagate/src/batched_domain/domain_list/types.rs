// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core type definitions for the domain list module.
//!
//! Contains `CachedLinearBounds`, `DomainMetadata`, and `DomainListConfig` —
//! the data structures that describe individual domains and list configuration.

use super::super::ConstraintTuple;
use super::alpha_queue::QueuedGraphAlphaState;
use crate::beta_crown::state::{
    GraphAlphaStateByteCensus, GraphAlphaStateRepresentation, GraphDomainAlphaState,
};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use ny_tensor::TreeTraversal;
use std::collections::HashMap;
use std::sync::Arc;

/// Cached linear bound coefficients (A matrices + bias) for CROWN backward pass.
///
/// Stores the full linear bounds (coefficient matrices and bias vectors) at each
/// splittable activation's input. These can be reused in child domains to warm-start
/// the backward pass at the branch point instead of recomputing from the output.
///
/// The bias terms (`lower_b`, `upper_b`) are essential for warm-start correctness:
/// `concretize` computes `lA @ x + lb`, so omitting bias gives wrong bounds.
///
/// # Reference
/// alpha-beta-CROWN: `complete_verifier/tensor_storage.py` (all_lAs storage)
/// Design: `designs/2026-02-07-gpu-bab-la-reuse-closure.md` (Direction 2b)
#[derive(Debug, Clone, Default)]
pub struct CachedLinearBounds {
    /// Lower bound coefficient matrices per node: node_name -> [output_dim, input_dim].
    pub(crate) lower_a: HashMap<String, ndarray::Array2<f32>>,
    /// Upper bound coefficient matrices per node: node_name -> [output_dim, input_dim].
    pub(crate) upper_a: HashMap<String, ndarray::Array2<f32>>,
    /// Lower bound bias vectors per node: node_name -> [output_dim].
    pub(crate) lower_b: HashMap<String, ndarray::Array1<f32>>,
    /// Upper bound bias vectors per node: node_name -> [output_dim].
    pub(crate) upper_b: HashMap<String, ndarray::Array1<f32>>,
}

impl CachedLinearBounds {
    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.lower_a.is_empty() && self.upper_a.is_empty()
    }

    /// Number of cached layers (max of lower_a and upper_a entries).
    pub fn len(&self) -> usize {
        self.lower_a.len().max(self.upper_a.len())
    }

    /// Reconstruct full `LinearBounds` at a given node from cached data.
    ///
    /// Returns `None` if the node is not cached or if A/b are mismatched.
    /// Used by backward pass warm-start to seed `node_linear_bounds` at the
    /// branch point instead of at the output node.
    pub fn linear_bounds(&self, node_name: &str) -> Option<crate::LinearBounds> {
        let la = self.lower_a.get(node_name)?;
        let ua = self.upper_a.get(node_name)?;
        let lb = self.lower_b.get(node_name)?;
        let ub = self.upper_b.get(node_name)?;
        // KEEP unchecked: cache may store Inf coefficients from gradient paths
        // (gradients.rs:217,451), which LinearBounds::new() rejects for coefficients.
        // These arrays come from previously-decomposed LinearBounds with no mutation
        // between store and retrieve; fields stay pub(crate) to block external writes.
        Some(crate::LinearBounds::from_parts_unchecked(
            la.clone(),
            lb.clone(),
            ua.clone(),
            ub.clone(),
        ))
    }

    /// Create CachedLinearBounds from a map of node names to LinearBounds.
    ///
    /// This extracts the full linear bounds (A matrices + bias vectors) from each
    /// `LinearBounds` and stores them by node name.
    ///
    /// # Arguments
    /// * `linear_bounds_map` - Map from node name to LinearBounds (as produced by
    ///   the backward pass)
    ///
    /// # Reference
    /// Issue: #1564 (lA matrix caching), #1669 (warm-start requires bias)
    pub fn from_linear_bounds_map(linear_bounds_map: HashMap<String, crate::LinearBounds>) -> Self {
        let mut lower_a = HashMap::new();
        let mut upper_a = HashMap::new();
        let mut lower_b = HashMap::new();
        let mut upper_b = HashMap::new();

        for (node_name, lb) in linear_bounds_map {
            let (la, lbias, ua, ubias) = lb.into_parts();
            lower_a.insert(node_name.clone(), la);
            upper_a.insert(node_name.clone(), ua);
            lower_b.insert(node_name.clone(), lbias);
            upper_b.insert(node_name, ubias);
        }

        Self {
            lower_a,
            upper_a,
            lower_b,
            upper_b,
        }
    }

    /// #clip-interm-resnet: replace one node's cached input-relative linear
    /// bounds, overwriting all four A/b maps.
    ///
    /// Used to substitute a FINITE backward-CROWN enclosure for the ±inf forward
    /// fallback that `compute_forward_linear_bounds` emits on deep-resnet conv /
    /// residual split pre-activations. The replacement is a valid enclosure of
    /// the SAME node over the SAME input box, so downstream constraint building
    /// and tightening stay sound.
    pub fn override_node(&mut self, node_name: &str, bounds: crate::LinearBounds) {
        // `into_parts()` yields `(lower_a, lower_b, upper_a, upper_b)`.
        let (la, lbias, ua, ubias) = bounds.into_parts();
        self.lower_a.insert(node_name.to_string(), la);
        self.upper_a.insert(node_name.to_string(), ua);
        self.lower_b.insert(node_name.to_string(), lbias);
        self.upper_b.insert(node_name.to_string(), ubias);
    }

    /// Split a multi-row cache into per-row single-objective caches.
    ///
    /// Each entry's A matrix has shape `[num_objectives, input_dim]` and bias has
    /// shape `[num_objectives]`. This function slices row `i` from every node's
    /// A/bias to produce one `CachedLinearBounds` per objective.
    ///
    /// Returns `None` if the cache is empty or any matrix has fewer rows than
    /// `num_objectives`.
    ///
    /// # Reference
    /// Design: `designs/2026-03-15-issue-3813-multi-objective-la-warm-start.md`
    /// Issue: #3813
    pub fn split_multi_row(&self, num_objectives: usize) -> Option<Vec<CachedLinearBounds>> {
        if self.is_empty() || num_objectives == 0 {
            return None;
        }

        // Validate all matrices have at least num_objectives rows.
        for a in self.lower_a.values().chain(self.upper_a.values()) {
            if a.nrows() < num_objectives {
                return None;
            }
        }
        for b in self.lower_b.values().chain(self.upper_b.values()) {
            if b.len() < num_objectives {
                return None;
            }
        }

        let mut result = Vec::with_capacity(num_objectives);
        for row_idx in 0..num_objectives {
            let mut la = HashMap::new();
            let mut ua = HashMap::new();
            let mut lb = HashMap::new();
            let mut ub = HashMap::new();

            for (name, a) in &self.lower_a {
                // Slice row row_idx: shape [1, input_dim]
                let row = a.row(row_idx).to_owned().insert_axis(ndarray::Axis(0));
                la.insert(name.clone(), row);
            }
            for (name, a) in &self.upper_a {
                let row = a.row(row_idx).to_owned().insert_axis(ndarray::Axis(0));
                ua.insert(name.clone(), row);
            }
            for (name, b) in &self.lower_b {
                let val = ndarray::arr1(&[b[row_idx]]);
                lb.insert(name.clone(), val);
            }
            for (name, b) in &self.upper_b {
                let val = ndarray::arr1(&[b[row_idx]]);
                ub.insert(name.clone(), val);
            }

            result.push(CachedLinearBounds {
                lower_a: la,
                upper_a: ua,
                lower_b: lb,
                upper_b: ub,
            });
        }

        Some(result)
    }
}

/// Metadata for a single domain (non-tensor fields).
///
/// Fields are `pub(crate)` to enforce construction through validated methods (#3125, #2982).
#[derive(Debug, Clone)]
pub struct DomainMetadata {
    /// Lower bound on objective.
    pub(crate) lower_bound: f32,
    /// Upper bound on objective.
    pub(crate) upper_bound: f32,
    /// Depth (number of splits).
    pub(crate) depth: usize,
    /// Constraint history: (node_name, neuron_idx, is_active, split_point).
    ///
    /// - ReLU constraints: split_point = None
    /// - GenBaB constraints: split_point = Some(pt)
    pub(crate) constraints: Vec<ConstraintTuple>,
    /// Cached linear bound coefficients (lA) from previous backward pass.
    ///
    /// When a domain is branched, children inherit the parent's lA and can
    /// reuse it as a starting point for the backward pass, avoiding O(n)
    /// recomputation at layers before the branch point.
    ///
    /// Wrapped in `Arc` for O(1) cloning during child domain creation (#2326).
    /// The cached bounds are read-only after creation.
    ///
    /// `None` indicates this is a new domain with no cached bounds.
    pub(crate) cached_la: Option<Arc<CachedLinearBounds>>,
    /// True when the domain was enqueued before its first CROWN/IBP bound pass.
    ///
    /// Currently used by reordered GPU input-split BaB so child domains can be
    /// queued with parent-estimated bounds and bounded when later popped.
    pub(crate) needs_bounding: bool,
    /// Child-local node-bounds override carried by complete clipping until the
    /// deferred CROWN pass consumes it.
    pub(crate) node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
    /// Optimized per-neuron alpha state from the joint beta+alpha optimization.
    ///
    /// When present, this preserves the optimized alpha values across the
    /// DomainList round-trip (pick → evaluate → store). Without this, alpha
    /// would be re-initialized from the heuristic (α = 1 if u > -l, else 0)
    /// every iteration, losing optimization gains.
    ///
    /// The adapter owns exactly one canonical representation: mutable runtime
    /// state outside the queue or packed state while resident in this
    /// graph-local DomainList. Packed state is never a cross-graph cache.
    ///
    /// `None` for non-graph-BaB domains or when alpha optimization is disabled.
    /// Issue: #1845
    pub(crate) alpha_state: Option<QueuedGraphAlphaState>,
}

impl DomainMetadata {
    // --- Validated constructors (#3125, #2982) ---

    /// Create a new DomainMetadata with NaN validation on bounds.
    ///
    /// Returns `Err(NumericalInstability)` if `lower_bound` or `upper_bound` is NaN.
    /// This prevents zombie domains from entering the BaB search.
    ///
    /// # Reference
    /// designs/2026-02-25-validated-bab-domain-types.md Phase 4
    pub(crate) fn new(
        lower_bound: f32,
        upper_bound: f32,
        depth: usize,
        constraints: Vec<ConstraintTuple>,
        cached_la: Option<Arc<CachedLinearBounds>>,
        alpha_state: Option<GraphDomainAlphaState>,
    ) -> Result<Self> {
        if lower_bound.is_nan() || upper_bound.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "DomainMetadata bounds contain NaN: lower={lower_bound}, upper={upper_bound}"
            )));
        }
        Ok(Self {
            lower_bound,
            upper_bound,
            depth,
            constraints,
            cached_la,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: alpha_state.map(QueuedGraphAlphaState::from),
        })
    }

    /// Create a root domain metadata (depth 0, no constraints/cache/alpha).
    ///
    /// Convenience constructor for the common case of creating the initial root domain.
    /// Validates bounds are not NaN.
    pub(crate) fn root(lower_bound: f32, upper_bound: f32) -> Result<Self> {
        Self::new(lower_bound, upper_bound, 0, Vec::new(), None, None)
    }

    /// Update bounds with NaN validation.
    ///
    /// Returns `Err(NumericalInstability)` if either bound is NaN.
    /// Use this instead of direct field assignment for defense-in-depth.
    pub(crate) fn update_bounds(&mut self, lower_bound: f32, upper_bound: f32) -> Result<()> {
        if lower_bound.is_nan() || upper_bound.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "DomainMetadata bounds update contains NaN: lower={lower_bound}, upper={upper_bound}"
            )));
        }
        self.lower_bound = lower_bound;
        self.upper_bound = upper_bound;
        Ok(())
    }

    // --- Accessor methods (#3125) ---

    /// Lower bound on objective.
    pub fn lower_bound(&self) -> f32 {
        self.lower_bound
    }

    /// Upper bound on objective.
    pub fn upper_bound(&self) -> f32 {
        self.upper_bound
    }

    /// Depth (number of splits).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Constraint history.
    pub fn constraints(&self) -> &[ConstraintTuple] {
        &self.constraints
    }

    /// Cached linear bound coefficients (Arc-wrapped for O(1) cloning, #2326).
    pub fn cached_la(&self) -> &Option<Arc<CachedLinearBounds>> {
        &self.cached_la
    }

    /// Whether this domain still needs its first bound pass after being picked.
    pub fn needs_bounding(&self) -> bool {
        self.needs_bounding
    }

    /// Mark whether this domain should be bounded before branching.
    pub(crate) fn set_needs_bounding(&mut self, needs_bounding: bool) {
        self.needs_bounding = needs_bounding;
    }

    /// Child-local node-bounds override to use on the deferred CROWN pass.
    pub fn node_bounds_override(&self) -> Option<&HashMap<String, BoundedTensor>> {
        self.node_bounds_override.as_deref()
    }

    /// Update the deferred child node-bounds override.
    pub(crate) fn set_node_bounds_override(
        &mut self,
        node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
    ) -> Result<()> {
        if let Some(node_bounds_override) = node_bounds_override.as_deref() {
            // alpha-beta-CROWN keeps queued domains resident in CPU storage until a later
            // pick-out/bound pass (`complete_verifier/branching_domains.py`,
            // `complete_verifier/input_split/branching_domains.py`), so the Rust
            // deferred override queue must reject non-finite values before enqueueing.
            super::super::utils::validate_node_bounds_override_finite(
                node_bounds_override,
                "DomainMetadata::set_node_bounds_override",
            )?;
        }
        self.node_bounds_override = node_bounds_override;
        Ok(())
    }

    /// Alpha state for optimization.
    pub fn alpha_state(&self) -> Option<&GraphDomainAlphaState> {
        self.alpha_state
            .as_ref()
            .and_then(QueuedGraphAlphaState::runtime)
    }

    /// Whether the metadata currently owns runtime or packed queue alpha state.
    pub fn alpha_state_representation(&self) -> Option<GraphAlphaStateRepresentation> {
        self.alpha_state
            .as_ref()
            .map(QueuedGraphAlphaState::representation)
    }

    /// Explicit owned-byte estimate for the currently stored alpha representation.
    ///
    /// Includes the `DomainMetadata` alpha adapter slot and, for packed state,
    /// the boxed packed header plus all owned vector/string allocations.
    pub fn alpha_state_byte_census(&self) -> Option<GraphAlphaStateByteCensus> {
        self.alpha_state
            .as_ref()
            .map(QueuedGraphAlphaState::byte_census)
    }

    pub(crate) fn set_alpha_state(&mut self, alpha_state: Option<GraphDomainAlphaState>) {
        self.alpha_state = alpha_state.map(QueuedGraphAlphaState::from);
    }

    pub(crate) fn require_runtime_alpha_state(&self) -> Result<Option<&GraphDomainAlphaState>> {
        match &self.alpha_state {
            None => Ok(None),
            Some(QueuedGraphAlphaState::Runtime(runtime)) => Ok(Some(runtime)),
            Some(QueuedGraphAlphaState::Packed(_)) => Err(NyError::InternalError(
                "packed graph alpha state escaped DomainList::pick_out".to_string(),
            )),
        }
    }

    pub(super) fn pack_alpha_state_for_queue(&mut self, queue_identity: u64) -> Result<()> {
        if let Some(alpha_state) = &mut self.alpha_state {
            alpha_state.pack_for_queue(queue_identity)?;
        }
        Ok(())
    }

    pub(super) fn validate_queued_alpha_state(&self, queue_identity: u64) -> Result<()> {
        if let Some(alpha_state) = &self.alpha_state {
            alpha_state.validate(queue_identity)?;
        }
        Ok(())
    }

    pub(super) fn unpack_alpha_state_after_dequeue(&mut self, queue_identity: u64) -> Result<()> {
        if let Some(alpha_state) = &mut self.alpha_state {
            alpha_state.unpack_after_dequeue(queue_identity)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn corrupt_packed_alpha_layout_for_test(&mut self) {
        if let Some(alpha_state) = &mut self.alpha_state {
            alpha_state.corrupt_packed_layout_for_test();
        }
    }
}

/// Configuration for DomainList.
#[derive(Debug, Clone)]
pub struct DomainListConfig {
    /// Tree traversal mode (DFS or BFS).
    pub traversal: TreeTraversal,
    /// Layer names for bound storage.
    pub layer_names: Vec<String>,
    /// Layer shapes (without batch dimension) keyed by layer name.
    pub layer_shapes: HashMap<String, Vec<usize>>,
    /// Input shape (without batch dimension).
    pub input_shape: Vec<usize>,
    /// Initial storage capacity.
    pub initial_capacity: usize,
    /// Maximum number of domains to store simultaneously.
    ///
    /// When the queue exceeds this limit after an `add()`, the lowest-priority
    /// domains (highest lower_bound in verify-lower mode) are evicted to prevent
    /// unbounded memory growth.
    ///
    /// Set to 0 to disable the cap (unbounded queue, original behavior).
    ///
    /// Reference: Issue #2326 Finding 1
    pub max_queue_size: usize,
}

impl Default for DomainListConfig {
    fn default() -> Self {
        Self {
            traversal: TreeTraversal::DepthFirst,
            layer_names: Vec::new(),
            layer_shapes: HashMap::new(),
            input_shape: Vec::new(),
            initial_capacity: 1024,
            max_queue_size: 0, // 0 = disabled (unbounded, original behavior)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_split_multi_row_produces_correct_per_objective_caches() {
        let mut cache = CachedLinearBounds::default();
        // 3 objectives, 2 input dims
        cache.lower_a.insert(
            "relu1".to_string(),
            arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        );
        cache.upper_a.insert(
            "relu1".to_string(),
            arr2(&[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        );
        cache
            .lower_b
            .insert("relu1".to_string(), arr1(&[10.0, 20.0, 30.0]));
        cache
            .upper_b
            .insert("relu1".to_string(), arr1(&[11.0, 21.0, 31.0]));

        let split = cache.split_multi_row(3).expect("split should succeed");
        assert_eq!(split.len(), 3);

        // Objective 0: row 0
        let obj0 = &split[0];
        assert_eq!(obj0.lower_a["relu1"], arr2(&[[1.0, 2.0]]));
        assert_eq!(obj0.upper_a["relu1"], arr2(&[[0.1, 0.2]]));
        assert_eq!(obj0.lower_b["relu1"], arr1(&[10.0]));
        assert_eq!(obj0.upper_b["relu1"], arr1(&[11.0]));

        // Objective 1: row 1
        let obj1 = &split[1];
        assert_eq!(obj1.lower_a["relu1"], arr2(&[[3.0, 4.0]]));
        assert_eq!(obj1.upper_a["relu1"], arr2(&[[0.3, 0.4]]));
        assert_eq!(obj1.lower_b["relu1"], arr1(&[20.0]));
        assert_eq!(obj1.upper_b["relu1"], arr1(&[21.0]));

        // Objective 2: row 2
        let obj2 = &split[2];
        assert_eq!(obj2.lower_a["relu1"], arr2(&[[5.0, 6.0]]));
        assert_eq!(obj2.upper_a["relu1"], arr2(&[[0.5, 0.6]]));
        assert_eq!(obj2.lower_b["relu1"], arr1(&[30.0]));
        assert_eq!(obj2.upper_b["relu1"], arr1(&[31.0]));
    }

    #[test]
    fn test_split_multi_row_returns_none_for_empty_cache() {
        assert!(CachedLinearBounds::default().split_multi_row(3).is_none());
    }

    #[test]
    fn test_split_multi_row_returns_none_when_rows_insufficient() {
        let mut cache = CachedLinearBounds::default();
        // Only 2 rows but requesting 3 objectives
        cache
            .lower_a
            .insert("relu1".to_string(), arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        cache
            .upper_a
            .insert("relu1".to_string(), arr2(&[[0.1, 0.2], [0.3, 0.4]]));
        cache
            .lower_b
            .insert("relu1".to_string(), arr1(&[10.0, 20.0]));
        cache
            .upper_b
            .insert("relu1".to_string(), arr1(&[11.0, 21.0]));

        assert!(cache.split_multi_row(3).is_none());
    }

    #[test]
    fn test_split_multi_row_handles_multiple_nodes() {
        let mut cache = CachedLinearBounds::default();
        // 2 objectives, 2 nodes
        cache
            .lower_a
            .insert("relu1".to_string(), arr2(&[[1.0], [2.0]]));
        cache
            .lower_a
            .insert("relu2".to_string(), arr2(&[[3.0, 4.0], [5.0, 6.0]]));
        cache
            .upper_a
            .insert("relu1".to_string(), arr2(&[[0.1], [0.2]]));
        cache
            .upper_a
            .insert("relu2".to_string(), arr2(&[[0.3, 0.4], [0.5, 0.6]]));
        cache
            .lower_b
            .insert("relu1".to_string(), arr1(&[10.0, 20.0]));
        cache
            .lower_b
            .insert("relu2".to_string(), arr1(&[30.0, 40.0]));
        cache
            .upper_b
            .insert("relu1".to_string(), arr1(&[11.0, 21.0]));
        cache
            .upper_b
            .insert("relu2".to_string(), arr1(&[31.0, 41.0]));

        let split = cache.split_multi_row(2).expect("split should succeed");
        assert_eq!(split.len(), 2);

        // Objective 0 has both nodes, row 0
        assert_eq!(split[0].lower_a["relu1"], arr2(&[[1.0]]));
        assert_eq!(split[0].lower_a["relu2"], arr2(&[[3.0, 4.0]]));
        assert_eq!(split[0].lower_b["relu1"], arr1(&[10.0]));
        assert_eq!(split[0].lower_b["relu2"], arr1(&[30.0]));

        // Objective 1 has both nodes, row 1
        assert_eq!(split[1].lower_a["relu1"], arr2(&[[2.0]]));
        assert_eq!(split[1].lower_a["relu2"], arr2(&[[5.0, 6.0]]));
    }
}
