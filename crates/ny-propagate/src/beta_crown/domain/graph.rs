// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph network branch-and-bound domain.
//!
//! Contains `GraphBabDomain` and graph-domain child construction methods
//! (`with_constraint`, `with_general_split`).

use std::sync::Arc;

use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::branching::{
    GenBabConstraint, GraphNeuronConstraint, GraphSplitHistory, NeuronSplit,
};
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::GraphNetwork;
use crate::Layer;
use crate::NETWORK_INPUT;

/// Domain for GraphNetwork branch-and-bound with ReLU splitting.
///
/// Fields are `pub(crate)` to enforce construction through validated methods
/// that reject NaN bounds (#3125, #2982). Use accessor methods for reads
/// and `update_bounds()` for bound/priority updates.
#[derive(Debug, Clone)]
pub struct GraphBabDomain {
    /// Split history for this domain.
    pub(crate) history: GraphSplitHistory,
    /// Pre-activation bounds for each node (before applying ReLU constraints).
    /// Uses Arc for cheap cloning during branch-and-bound splits.
    pub(crate) node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>>,
    /// Lower bound on the objective.
    pub(crate) lower_bound: f32,
    /// Upper bound on the objective.
    pub(crate) upper_bound: f32,
    /// Current depth in the B&B tree.
    pub(crate) depth: usize,
    /// Priority for queue ordering.
    pub(crate) priority: f32,
    /// Input bounds for this domain.
    pub(crate) input_bounds: Arc<BoundedTensor>,
    /// β parameters for Lagrangian optimization (initialized from history).
    pub(crate) beta_state: GraphBetaState,
    /// Per-domain α state for ReLU lower-bound slope optimization.
    ///
    /// When present, the batched backward pass uses optimized per-neuron alpha
    /// values instead of the fixed heuristic (alpha = 1 if u > -l, else 0).
    /// Enables joint α-β optimization in the graph BaB path.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `auto_LiRPA/operators/relu.py` (optimizable slopes)
    /// Issue: #1841
    pub(crate) alpha_state: GraphDomainAlphaState,
    /// Cached lA coefficients from the parent domain's backward pass.
    ///
    /// When present, these can seed the backward pass at intermediate layers
    /// instead of recomputing from scratch at the output node. Children inherit
    /// the parent's lA cache; it is invalidated on the next backward pass and
    /// replaced with freshly captured lA.
    ///
    /// Wrapped in `Arc` for O(1) cloning during child domain creation (#2326).
    /// The cached bounds are read-only after creation -- children only read them
    /// to warm-start the backward pass. When the child needs fresh bounds, the
    /// entire `Option<Arc<...>>` is replaced, not mutated in place.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/tensor_storage.py` (all_lAs)
    /// Issue: #1564, #1669, #2326
    pub(crate) cached_la: Option<Arc<CachedLinearBounds>>,
    /// #cone-delta: pre-activation node names of the constraints added since
    /// `node_bounds` was last fixpointed.
    ///
    /// INVARIANT: `delta_pre_nodes` = the pre-activation nodes of every
    /// constraint appended to `history` since `node_bounds` was last replaced
    /// by a post-bounding fixpoint of the constrained forward pass.
    /// `with_constraint`/`with_general_split` append the split's pre-activation
    /// node (resolved exactly as `build_constraint_lookups` does);
    /// `child_with_norm_history` appends nothing (norm splits never touch
    /// `node_bounds`); every bounding path that replaces `node_bounds` clears
    /// the vec. Domains whose `node_bounds` provenance is outside that cycle
    /// (root construction, GPU `from_metadata` reconstruction) carry the
    /// `delta_pre_nodes_unknown()` sentinel, which the delta-seed gate rejects.
    ///
    /// Consumed (dark, `NY_CONE_REFRESH=1`) by
    /// `compute_constrained_forward_bounds_inner` to shrink the recompute seed
    /// set from the full split history to just the delta; gate off or any
    /// fail-closed condition ⇒ the vec is ignored and the full-history seeds
    /// run, byte-identically to today.
    pub(crate) delta_pre_nodes: Vec<String>,
}

impl PartialEq for GraphBabDomain {
    fn eq(&self, other: &Self) -> bool {
        super::cmp_domain_priority(self.priority, other.priority) == std::cmp::Ordering::Equal
    }
}

impl Eq for GraphBabDomain {}

impl PartialOrd for GraphBabDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphBabDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority = pop first
        super::cmp_domain_priority(self.priority, other.priority)
    }
}

impl GraphBabDomain {
    /// Create root domain with initial bounds.
    ///
    /// Returns `Err(NumericalInstability)` if bounds are non-finite (#2982, #3125).
    pub fn root(
        node_bounds: std::collections::HashMap<String, BoundedTensor>,
        lower_bound: f32,
        upper_bound: f32,
        input: &BoundedTensor,
        verify_upper: bool,
    ) -> Result<Self> {
        if !lower_bound.is_finite() || !upper_bound.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GraphBaB root domain bounds are non-finite: lower={lower_bound}, upper={upper_bound}"
            )));
        }
        let node_bounds = node_bounds
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();
        let priority =
            BetaCrownConfig::domain_priority_for_mode(verify_upper, lower_bound, upper_bound)?;
        Ok(Self {
            history: GraphSplitHistory::new(),
            node_bounds,
            lower_bound,
            upper_bound,
            depth: 0,
            priority,
            input_bounds: Arc::new(input.clone()),
            beta_state: GraphBetaState::empty(), // Root has no constraints
            alpha_state: GraphDomainAlphaState::empty(), // Populated after initial CROWN pass
            cached_la: None,                     // Root has no cached lA
            // #cone-delta: the root map comes from the caller (IBP/CROWN root
            // pass), not from a post-bounding replacement of the constrained
            // forward — delta unknown until the first bounding pass clears it.
            delta_pre_nodes: super::delta_pre_nodes_unknown(),
        })
    }

    /// Apply a constraint to create a child domain.
    ///
    /// Returns `Ok(None)` if the constraint makes the domain infeasible.
    /// Returns `Err(...)` if an internal error occurs (e.g., shape mismatch
    /// in flatten→reshape roundtrip). This distinguishes genuine infeasibility
    /// from bugs that would silently drop domains (#2302).
    pub fn with_constraint(
        &self,
        graph: &GraphNetwork,
        constraint: GraphNeuronConstraint,
        verify_upper: bool,
    ) -> Result<Option<Self>> {
        let node_name = &constraint.node_name;
        let neuron_idx = constraint.neuron_idx;
        let is_active = constraint.is_active;

        // Constraints are on the *pre-activation* of this ReLU/Sign, i.e. its input node.
        // Both ReLU and Sign split at x=0 with identical half-space semantics (#3769).
        let relu_node = match graph.nodes.get(node_name) {
            Some(node) => node,
            None => return Ok(None),
        };
        if !matches!(relu_node.layer, Layer::ReLU(_) | Layer::Sign(_)) {
            return Ok(None);
        }
        let pre_name = match relu_node.inputs.first().map(|s| s.as_str()) {
            Some(name) => name,
            None => {
                // #2098: Return None for nodes with empty inputs instead of fabricating NETWORK_INPUT.
                tracing::warn!("ReLU node has empty inputs — cannot determine pre-activation");
                return Ok(None);
            }
        };

        let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
            self.input_bounds.as_ref()
        } else {
            match self.node_bounds.get(pre_name) {
                Some(b) => b.as_ref(),
                None => return Ok(None),
            }
        };
        let flat = pre_bounds.flatten();

        if neuron_idx >= flat.len() {
            return Ok(None);
        }

        let l = flat.lower()[[neuron_idx]];
        let u = flat.upper()[[neuron_idx]];

        // Explicit NaN/Inf guard: non-finite neuron bounds make the feasibility
        // check below unreliable (IEEE 754: NaN < 0.0 = false, so NaN passes).
        // Mirrors the NaN guard in with_general_split (#2954). (#2599)
        if !l.is_finite() || !u.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "with_constraint: non-finite neuron bounds for {node_name}[{neuron_idx}] \
                 (l={l}, u={u})"
            )));
        }

        // Feasibility check for intersection with the half-space.
        // Use strict inequalities so boundary cases (x==0) remain feasible.
        if is_active && u < 0.0 {
            return Ok(None);
        }
        if !is_active && l > 0.0 {
            return Ok(None);
        }

        // If the constrained pre-activation is the network input, tighten input bounds
        // so subsequent consumer nodes see the restricted range.
        let new_input_bounds = if pre_name == NETWORK_INPUT {
            let shape = pre_bounds.shape().to_vec();
            let mut lower_flat = flat.lower().clone();
            let mut upper_flat = flat.upper().clone();

            if is_active {
                // NaN-safe: propagate NaN instead of silently clamping to 0.0 (#2643)
                lower_flat[[neuron_idx]] = nan_propagating_max(lower_flat[[neuron_idx]], 0.0);
            } else {
                upper_flat[[neuron_idx]] = nan_propagating_min(upper_flat[[neuron_idx]], 0.0);
            }
            if lower_flat[[neuron_idx]] > upper_flat[[neuron_idx]] {
                return Ok(None);
            }

            // Shape errors in flatten→reshape roundtrip indicate an internal bug,
            // not legitimate infeasibility (#2302).
            let lower_new = lower_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "with_constraint: child domain lower shape mismatch: {e}"
                    ))
                })?;
            let upper_new = upper_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "with_constraint: child domain upper shape mismatch: {e}"
                    ))
                })?;
            Arc::new(BoundedTensor::new(lower_new, upper_new)?)
        } else {
            self.input_bounds.clone()
        };

        let new_history = self.history.with_constraint(constraint);
        let priority = BetaCrownConfig::domain_priority_for_mode(
            verify_upper,
            self.lower_bound,
            self.upper_bound,
        )?;

        // Initialize β state with warmup from parent domain.
        // This inherits optimized β values for existing constraints while
        // initializing the new constraint's β to the default value.
        // This warmup is crucial for β-CROWN convergence per α,β-CROWN paper.
        let beta_state = GraphBetaState::from_history_with_warmup(
            &new_history,
            &self.beta_state,
            GraphBetaState::DEFAULT_BETA_INIT,
        )?;

        // Initialize α state from parent with warm start.
        // Inherits optimized α for neurons that remain unstable in the child.
        let alpha_state = GraphDomainAlphaState::from_parent(
            &self.alpha_state,
            graph,
            &self.node_bounds,
            &new_history,
            &new_input_bounds,
        );

        Ok(Some(Self {
            history: new_history,
            node_bounds: self.node_bounds.clone(),
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            depth: self.depth + 1,
            priority,
            input_bounds: new_input_bounds,
            beta_state,
            alpha_state,
            cached_la: self.cached_la.clone(), // Inherit parent's lA cache
            // #cone-delta: the child inherits `node_bounds` verbatim, so its
            // delta = parent delta + this split's pre-activation node (the same
            // resolution `build_constraint_lookups` performs).
            delta_pre_nodes: {
                let mut delta = self.delta_pre_nodes.clone();
                delta.push(pre_name.to_string());
                delta
            },
        }))
    }

    /// Apply a general split (GenBaB) to create a child domain.
    ///
    /// Unlike `with_constraint` which is ReLU-specific (branching at 0), this method
    /// supports arbitrary branching points for general nonlinearities (GeLU, Sigmoid, etc.).
    ///
    /// # Arguments
    /// * `graph` - The graph network being verified
    /// * `split` - The neuron split specification with explicit lower/upper bounds
    /// * `verify_upper` - Whether verifying upper bound
    ///
    /// # Returns
    /// `Ok(Some(domain))` for valid child, `Ok(None)` if infeasible,
    /// `Err(...)` on internal error (shape mismatch etc.) (#2302).
    pub fn with_general_split(
        &self,
        graph: &GraphNetwork,
        split: NeuronSplit,
        verify_upper: bool,
    ) -> Result<Option<Self>> {
        let node_name = match split.layer.as_name() {
            Some(name) => name,
            None => return Ok(None),
        };

        // GenBaB norm branching (#norm-genbab): a norm split clamps the node's
        // internal inv_rms scalar, not any graph node's value interval. It does
        // NOT narrow input_bounds or node_bounds — the narrowing is applied
        // inside the RmsNorm CROWN backward via the recorded NormInvRmsConstraint
        // (intersected with the node's own IBP inv_rms range). The two sibling
        // children carry the lower/upper halves of the parent inv_rms range and
        // union-cover it (hence the full input box), so the verdict is sound.
        if let Some((group, inv_lo, inv_hi)) = split.norm_inv_rms_window() {
            let constraint = crate::beta_crown::branching::NormInvRmsConstraint::new(
                node_name.to_string(),
                group,
                inv_lo,
                inv_hi,
                split.score(),
            )?;
            let new_history = self.history.with_norm_inv_rms_constraint(constraint);
            return Ok(Some(self.child_with_norm_history(new_history)));
        }

        let neuron_idx = split.neuron_idx;

        // Get the node and find its input node for bounds.
        // For binary ops (BilinearCrown), input_index selects which input to split
        // (0 = Q, 1 = K). For unary activations, defaults to first input.
        let node = match graph.nodes.get(node_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let input_idx = split.input_index().unwrap_or(0);
        let pre_name = match node.inputs.get(input_idx).map(|s| s.as_str()) {
            Some(name) => name,
            None => {
                tracing::warn!(
                    node = %node_name,
                    input_idx = input_idx,
                    "Node has no input at index {input_idx} — cannot determine pre-activation"
                );
                return Ok(None);
            }
        };

        let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
            self.input_bounds.as_ref()
        } else {
            match self.node_bounds.get(pre_name) {
                Some(b) => b.as_ref(),
                None => return Ok(None),
            }
        };
        let flat = pre_bounds.flatten();

        if neuron_idx >= flat.len() {
            return Ok(None);
        }

        let current_l = flat.lower()[[neuron_idx]];
        let current_u = flat.upper()[[neuron_idx]];

        // Compute effective bounds after applying the split
        let new_lower = split.lower_bound.unwrap_or(current_l);
        let new_upper = split.upper_bound.unwrap_or(current_u);

        // Feasibility check: new bounds must intersect with current bounds.
        // Use NaN-propagating ops so NaN bounds are not silently absorbed (#2954).
        // IEEE 754: NaN.max(x) = x would hide corruption; nan_propagating_max
        // returns NaN, consistent with with_constraint (#2643).
        let effective_l = nan_propagating_max(current_l, new_lower);
        let effective_u = nan_propagating_min(current_u, new_upper);

        // Explicit NaN guard: if either effective bound is NaN, the split
        // inputs are corrupted. Reject immediately rather than letting NaN
        // flow into constraints or child BoundedTensor (#2954).
        if effective_l.is_nan() || effective_u.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "with_general_split: NaN in effective bounds for {node_name}[{neuron_idx}] \
                 (effective_l={effective_l}, effective_u={effective_u}, \
                 current_l={current_l}, current_u={current_u}, \
                 new_lower={new_lower}, new_upper={new_upper})"
            )));
        }

        if effective_l > effective_u {
            return Ok(None);
        }

        // Update input bounds if constraining network input
        let new_input_bounds = if pre_name == NETWORK_INPUT {
            let shape = pre_bounds.shape().to_vec();
            let mut lower_flat = flat.lower().clone();
            let mut upper_flat = flat.upper().clone();

            lower_flat[[neuron_idx]] = effective_l;
            upper_flat[[neuron_idx]] = effective_u;

            // Shape errors in flatten→reshape roundtrip indicate an internal bug (#2302).
            let lower_new = lower_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "with_general_split: child domain lower shape mismatch: {e}"
                    ))
                })?;
            let upper_new = upper_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "with_general_split: child domain upper shape mismatch: {e}"
                    ))
                })?;
            Arc::new(BoundedTensor::new(lower_new, upper_new)?)
        } else {
            self.input_bounds.clone()
        };

        // Determine constraint type based on split
        let new_history = if split.is_relu_split() {
            // ReLU split (at 0): use standard ReLU constraint
            let constraint = GraphNeuronConstraint {
                node_name: node_name.to_string(),
                neuron_idx,
                is_active: split.lower_bound.is_some(),
                score: split.score, // lower_bound = Some(0.0) means x >= 0
            };
            self.history.with_constraint(constraint)
        } else {
            // GenBaB split (non-zero point): use native GenBaB constraint(s).
            //
            // For binary McCormick ops (MulBinary, BilinearCrown) `input_idx`
            // selects WHICH input node the split_point clamp targets in the
            // forward pass. We thread it onto the constraint so
            // `build_constraint_lookups` resolves the correct pre-activation
            // node (`node.inputs[input_idx]`) instead of always `inputs.first()`.
            // For unary nonlinearities input_idx is 0 → unchanged behavior.
            // SOUNDNESS: `effective_l`/`effective_u` are computed from the
            // *correct* input's pre_bounds above, so the clamp value is right;
            // tagging the input index only fixes WHERE it is applied.
            match (split.lower_bound, split.upper_bound) {
                (Some(_lower), Some(_upper)) => {
                    // Range split: enforce both bounds with two constraints.
                    let upper_branch = GenBabConstraint::new(
                        node_name.to_string(),
                        neuron_idx,
                        effective_l,
                        true,
                        split.score,
                    )?
                    .with_input_index(input_idx);
                    let lower_branch = GenBabConstraint::new(
                        node_name.to_string(),
                        neuron_idx,
                        effective_u,
                        false,
                        split.score,
                    )?
                    .with_input_index(input_idx);
                    self.history
                        .with_genbab_constraints_for_split([upper_branch, lower_branch])
                }
                (Some(_point), None) => {
                    let constraint = GenBabConstraint::new(
                        node_name.to_string(),
                        neuron_idx,
                        effective_l,
                        true,
                        split.score,
                    )?
                    .with_input_index(input_idx);
                    self.history.with_genbab_constraint(constraint)
                }
                (None, Some(_point)) => {
                    let constraint = GenBabConstraint::new(
                        node_name.to_string(),
                        neuron_idx,
                        effective_u,
                        false,
                        split.score,
                    )?
                    .with_input_index(input_idx);
                    self.history.with_genbab_constraint(constraint)
                }
                (None, None) => return Ok(None),
            }
        };
        let priority = BetaCrownConfig::domain_priority_for_mode(
            verify_upper,
            self.lower_bound,
            self.upper_bound,
        )?;

        // Initialize β state with warmup from parent
        let beta_state = GraphBetaState::from_history_with_warmup(
            &new_history,
            &self.beta_state,
            GraphBetaState::DEFAULT_BETA_INIT,
        )?;

        // Initialize α state from parent with warm start
        let alpha_state = GraphDomainAlphaState::from_parent(
            &self.alpha_state,
            graph,
            &self.node_bounds,
            &new_history,
            &new_input_bounds,
        );

        Ok(Some(Self {
            history: new_history,
            node_bounds: self.node_bounds.clone(),
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            depth: self.depth + 1,
            priority,
            input_bounds: new_input_bounds,
            beta_state,
            alpha_state,
            cached_la: self.cached_la.clone(), // Inherit parent's lA cache
            // #cone-delta: both the ReLU-split and GenBaB arms above tighten
            // exactly `pre_name` in the forward pass, so it is the delta entry
            // (matches `build_constraint_lookups`' pre/pre_genbab resolution,
            // including the `input_index` selection for binary McCormick ops).
            delta_pre_nodes: {
                let mut delta = self.delta_pre_nodes.clone();
                delta.push(pre_name.to_string());
                delta
            },
        }))
    }

    /// Build a child domain for a GenBaB norm split (#norm-genbab).
    ///
    /// A norm split records a [`NormInvRmsConstraint`] in the history but does
    /// NOT narrow `input_bounds` or `node_bounds` (the constrained quantity is
    /// the RmsNorm node's internal inv_rms, applied inside its CROWN backward).
    /// The child therefore inherits the parent's input/node bounds, β state, and
    /// lA cache unchanged; only the history grows. The objective bounds are
    /// carried from the parent and refined by the child's re-propagation.
    fn child_with_norm_history(&self, new_history: GraphSplitHistory) -> Self {
        Self {
            history: new_history,
            node_bounds: self.node_bounds.clone(),
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            depth: self.depth + 1,
            priority: self.priority,
            input_bounds: self.input_bounds.clone(),
            beta_state: self.beta_state.clone(),
            alpha_state: self.alpha_state.clone(),
            cached_la: self.cached_la.clone(),
            // #cone-delta: a norm split acts only inside the RmsNorm CROWN
            // backward — it tightens no forward node bounds, so it contributes
            // nothing to the delta.
            delta_pre_nodes: self.delta_pre_nodes.clone(),
        }
    }

    /// Create 2^k children from k simultaneous ReLU split decisions.
    ///
    /// Each child gets a unique combination of active/inactive constraints
    /// for the k neurons, matching the truth-table pattern from the reference
    /// (`domain_updater.py:146-304`).
    ///
    /// For k neurons [A, B, C], creates up to 8 children:
    ///   [A<=0, B<=0, C<=0], [A>=0, B<=0, C<=0], [A<=0, B>=0, C<=0], ...
    ///
    /// Infeasible children (where constraint makes domain empty) are pruned.
    /// Returns fewer than 2^k children when constraints conflict with bounds.
    ///
    /// Reference: alpha-beta-CROWN `set_branched_bounds()` in `domain_updater.py:63-304`.
    /// Part of #2767 (multi-depth ReLU splitting).
    pub fn with_multi_constraints(
        &self,
        graph: &GraphNetwork,
        splits: &[(String, usize, f32)],
        verify_upper_bound: bool,
    ) -> Result<Vec<Self>> {
        if splits.is_empty() {
            return Ok(vec![self.clone()]);
        }

        // Cap at reasonable depth to prevent combinatorial explosion.
        // 2^10 = 1024 children max. The config's max_relu_split_depth should
        // enforce a lower cap, but guard here defensively.
        if splits.len() > 10 {
            return Err(NyError::InternalError(format!(
                "with_multi_constraints: split depth {} exceeds maximum 10",
                splits.len()
            )));
        }

        let num_children = 1usize << splits.len(); // 2^k
        let mut children = Vec::with_capacity(num_children);

        for child_idx in 0..num_children {
            let mut domain = self.clone();
            let mut feasible = true;

            for (split_pos, (node_name, neuron_idx, score)) in splits.iter().enumerate() {
                // Bit at position split_pos determines active (1) vs inactive (0)
                let is_active = (child_idx >> split_pos) & 1 == 1;
                let constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx: *neuron_idx,
                    is_active,
                    score: *score,
                };
                match domain.with_constraint(graph, constraint, verify_upper_bound)? {
                    Some(new_domain) => domain = new_domain,
                    None => {
                        feasible = false;
                        break;
                    }
                }
            }

            if feasible {
                children.push(domain);
            }
        }

        Ok(children)
    }

    // --- Accessor methods (#3125) ---

    /// Lower bound on the objective.
    pub fn lower_bound(&self) -> f32 {
        self.lower_bound
    }

    /// Upper bound on the objective.
    pub fn upper_bound(&self) -> f32 {
        self.upper_bound
    }

    /// Priority for queue ordering.
    pub fn priority(&self) -> f32 {
        self.priority
    }

    /// Current depth in the B&B tree.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Split history for this domain.
    pub fn history(&self) -> &GraphSplitHistory {
        &self.history
    }

    /// Pre-activation bounds for each node.
    pub fn node_bounds(&self) -> &std::collections::HashMap<String, Arc<BoundedTensor>> {
        &self.node_bounds
    }

    /// #cone-delta: pre-activation nodes of constraints added since
    /// `node_bounds` was last fixpointed (see the field invariant).
    pub fn delta_pre_nodes(&self) -> &[String] {
        &self.delta_pre_nodes
    }

    /// Input bounds for this domain.
    pub fn input_bounds(&self) -> &BoundedTensor {
        &self.input_bounds
    }

    /// Input bounds Arc reference.
    pub fn input_bounds_arc(&self) -> &Arc<BoundedTensor> {
        &self.input_bounds
    }

    /// β state for Lagrangian optimization.
    pub fn beta_state(&self) -> &GraphBetaState {
        &self.beta_state
    }

    /// α state for ReLU slope optimization.
    pub fn alpha_state(&self) -> &GraphDomainAlphaState {
        &self.alpha_state
    }

    /// Cached lA coefficients (Arc-wrapped for O(1) child cloning, #2326).
    pub fn cached_la(&self) -> &Option<Arc<CachedLinearBounds>> {
        &self.cached_la
    }

    /// Create a domain from DomainList metadata (GPU→CPU path).
    ///
    /// Used by `graph_domain_from_picked()` to reconstruct domains from batched
    /// DomainList storage. Returns `Err(NumericalInstability)` if bounds or
    /// priority contain NaN (#3125, #2982).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_metadata(
        history: GraphSplitHistory,
        node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>>,
        lower_bound: f32,
        upper_bound: f32,
        depth: usize,
        priority: f32,
        input_bounds: Arc<BoundedTensor>,
        beta_state: GraphBetaState,
        alpha_state: GraphDomainAlphaState,
        cached_la: Option<Arc<CachedLinearBounds>>,
    ) -> Result<Self> {
        if !lower_bound.is_finite() || !upper_bound.is_finite() || !priority.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GraphBaB domain from metadata non-finite: lower={lower_bound}, \
                 upper={upper_bound}, priority={priority}"
            )));
        }
        Ok(Self {
            history,
            node_bounds,
            lower_bound,
            upper_bound,
            depth,
            priority,
            input_bounds,
            beta_state,
            alpha_state,
            cached_la,
            // #cone-delta: bounds reconstructed from DomainList storage — their
            // fixpoint provenance is not tracked through the batched lane, so
            // the delta is unknown (fail-closed to full-history seeding).
            delta_pre_nodes: super::delta_pre_nodes_unknown(),
        })
    }

    /// Create a child domain from DomainList metadata (GPU BaB split path).
    ///
    /// Used by `branch_relu_from_picked()` for ReLU-split children. Increments
    /// depth from metadata. Returns `Err(NumericalInstability)` if bounds or
    /// priority are non-finite (#3125, #2982).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn child_from_metadata(
        history: GraphSplitHistory,
        node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>>,
        lower_bound: f32,
        upper_bound: f32,
        depth: usize,
        priority: f32,
        input_bounds: Arc<BoundedTensor>,
        beta_state: GraphBetaState,
        alpha_state: GraphDomainAlphaState,
        cached_la: Option<Arc<CachedLinearBounds>>,
    ) -> Result<Self> {
        if !lower_bound.is_finite() || !upper_bound.is_finite() || !priority.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GraphBaB child domain non-finite: lower={lower_bound}, \
                 upper={upper_bound}, priority={priority}"
            )));
        }
        Ok(Self {
            history,
            node_bounds,
            lower_bound,
            upper_bound,
            depth: depth + 1,
            priority,
            input_bounds,
            beta_state,
            alpha_state,
            cached_la,
            // #cone-delta: same fail-closed rationale as `from_metadata` — the
            // picked-batch parent map's fixpoint provenance is untracked.
            delta_pre_nodes: super::delta_pre_nodes_unknown(),
        })
    }

    /// Update bounds and priority atomically. Rejects non-finite values (#2982, #3125).
    pub(crate) fn update_bounds(&mut self, lower: f32, upper: f32, priority: f32) -> Result<()> {
        if !lower.is_finite() || !upper.is_finite() || !priority.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GraphBaB domain update non-finite: lower={lower}, upper={upper}, priority={priority}"
            )));
        }
        self.lower_bound = lower;
        self.upper_bound = upper;
        self.priority = priority;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;

    use super::*;
    use crate::beta_crown::branching::{GraphNeuronConstraint, LayerRef, NeuronSplit};
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::{GraphNetwork, GraphNode};

    /// Create a simple Linear(2→2) → ReLU → Linear(2→1) graph with a root
    /// BaB domain. Returns (graph, root_domain). The pre-activation node for
    /// "relu1" is "linear1", stored in `domain.node_bounds`.
    fn simple_graph_and_domain() -> (GraphNetwork, GraphBabDomain) {
        let w1 = ndarray::arr2(&[[1.0_f32, -1.0], [-1.0, 1.0]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();

        let w2 = ndarray::arr2(&[[1.0_f32, 1.0]]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu1".to_string()],
        ));
        graph.set_output("linear2");

        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32, -1.0]).into_dyn(),
            ndarray::arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();

        let node_bounds = graph.collect_node_bounds(&input).unwrap();
        let root = GraphBabDomain::root(node_bounds, 0.0, 4.0, &input, false).unwrap();
        (graph, root)
    }

    #[test]
    fn test_graph_bab_domain_cmp_treats_nan_priority_as_high_priority() {
        let input = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid test tensor");

        let finite =
            GraphBabDomain::root(std::collections::HashMap::new(), 0.0, 1.0, &input, false)
                .unwrap();
        let mut nan = finite.clone();
        nan.priority = f32::NAN;

        assert_eq!(nan.cmp(&finite), std::cmp::Ordering::Greater);
        assert_eq!(finite.cmp(&nan), std::cmp::Ordering::Less);

        let mut heap = BinaryHeap::new();
        heap.push(finite);
        heap.push(nan);

        let popped = heap.pop().expect("heap should contain two domains");
        assert!(popped.priority.is_nan());
    }

    /// Regression test for #2599: with_constraint must reject NaN neuron bounds.
    ///
    /// IEEE 754: NaN < 0.0 = false, so without an explicit guard, NaN bounds
    /// silently pass both feasibility checks (u < 0.0 and l > 0.0).
    #[test]
    fn test_with_constraint_rejects_nan_lower_bound() {
        let (graph, mut domain) = simple_graph_and_domain();

        // Inject NaN into lower bound of pre-activation for relu1's first neuron.
        let nan_bounds = BoundedTensor::new_unchecked(
            ndarray::arr1(&[f32::NAN, -0.5]).into_dyn(),
            ndarray::arr1(&[1.0, 1.0]).into_dyn(),
        )
        .unwrap();
        domain
            .node_bounds
            .insert("linear1".to_string(), Arc::new(nan_bounds));

        let constraint = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        };

        let result = domain.with_constraint(&graph, constraint, false);
        assert!(
            result.is_err(),
            "with_constraint should reject NaN lower bound"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("non-finite"),
            "error should mention non-finite: {err}"
        );
    }

    /// Regression test for #2599: with_constraint must reject NaN upper bound.
    #[test]
    fn test_with_constraint_rejects_nan_upper_bound() {
        let (graph, mut domain) = simple_graph_and_domain();

        let nan_bounds = BoundedTensor::new_unchecked(
            ndarray::arr1(&[-1.0, -0.5]).into_dyn(),
            ndarray::arr1(&[f32::NAN, 1.0]).into_dyn(),
        )
        .unwrap();
        domain
            .node_bounds
            .insert("linear1".to_string(), Arc::new(nan_bounds));

        let constraint = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: false,
            score: 1.0,
        };

        let result = domain.with_constraint(&graph, constraint, false);
        assert!(
            result.is_err(),
            "with_constraint should reject NaN upper bound"
        );
    }

    /// Test multi-constraint child creation (Phase 3, #2767).
    ///
    /// With 2 unstable neurons in relu1 (both cross zero), splitting depth=2
    /// should produce up to 4 children (2^2), each with a unique combination
    /// of active/inactive constraints.
    #[test]
    fn test_with_multi_constraints_creates_correct_children() {
        let (graph, domain) = simple_graph_and_domain();

        // relu1 has 2 neurons, both unstable (l < 0, u > 0) from the
        // simple graph setup: linear1 bounds are [-2, 2] for each neuron.
        let splits = vec![("relu1".to_string(), 0, 1.0), ("relu1".to_string(), 1, 0.8)];

        let children = domain
            .with_multi_constraints(&graph, &splits, false)
            .unwrap();

        // Should have up to 4 children (2^2). Some may be pruned if infeasible.
        assert!(
            !children.is_empty(),
            "should produce at least one feasible child"
        );
        assert!(
            children.len() <= 4,
            "should produce at most 2^2=4 children, got {}",
            children.len()
        );

        // Each child should have depth = domain.depth + 2 (one per constraint)
        for child in &children {
            assert_eq!(
                child.depth(),
                domain.depth() + 2,
                "child depth should be parent + 2 (one per split)"
            );
        }

        // Each child should have exactly 2 constraints in its history
        for child in &children {
            assert_eq!(
                child.history().constraints.len(),
                2,
                "each child should have 2 constraints"
            );
        }
    }

    /// Test that empty splits returns a clone of the parent.
    #[test]
    fn test_with_multi_constraints_empty_splits() {
        let (graph, domain) = simple_graph_and_domain();

        let children = domain.with_multi_constraints(&graph, &[], false).unwrap();
        assert_eq!(children.len(), 1, "empty splits should return parent clone");
        assert_eq!(children[0].depth(), domain.depth());
    }

    /// Test that single split produces exactly 2 children (same as with_constraint).
    #[test]
    fn test_with_multi_constraints_single_split() {
        let (graph, domain) = simple_graph_and_domain();

        let splits = vec![("relu1".to_string(), 0, 1.0)];

        let children = domain
            .with_multi_constraints(&graph, &splits, false)
            .unwrap();

        // Should produce 2 children: one active, one inactive
        assert_eq!(children.len(), 2, "single split should produce 2 children");

        // One child should have is_active=false (bit 0 = 0), other is_active=true (bit 0 = 1)
        let c0_active = children[0].history().constraints[0].is_active;
        let c1_active = children[1].history().constraints[0].is_active;
        assert_ne!(
            c0_active, c1_active,
            "children should have opposite constraints"
        );
    }

    /// Test that split depth > 10 is rejected.
    #[test]
    fn test_with_multi_constraints_rejects_excessive_depth() {
        let (graph, domain) = simple_graph_and_domain();

        let splits: Vec<(String, usize, f32)> =
            (0..11).map(|i| ("relu1".to_string(), i, 1.0)).collect();

        let result = domain.with_multi_constraints(&graph, &splits, false);
        assert!(result.is_err(), "depth > 10 should be rejected");
    }

    /// #cone-delta: root domains carry the delta-unknown sentinel (their map
    /// was not installed by a post-bounding replacement), `with_constraint`
    /// appends the split's PRE-ACTIVATION node, and a simulated bounding pass
    /// (map replacement + clear) restarts the delta at empty.
    #[test]
    fn test_delta_pre_nodes_tracking_through_split_and_bound_cycle() {
        let (graph, mut root) = simple_graph_and_domain();

        // Root: delta unknown (NETWORK_INPUT sentinel).
        assert_eq!(root.delta_pre_nodes(), &[NETWORK_INPUT.to_string()]);

        // Simulate a bounding pass fixpointing the map: delta restarts empty.
        root.delta_pre_nodes.clear();

        let constraint = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        };
        let child = root
            .with_constraint(&graph, constraint, false)
            .unwrap()
            .unwrap();
        // The delta entry is relu1's PRE-ACTIVATION node, never the ReLU name.
        assert_eq!(child.delta_pre_nodes(), &["linear1".to_string()]);

        // Grandchild without an intervening bounding pass accumulates.
        let constraint2 = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 1,
            is_active: false,
            score: 1.0,
        };
        let grandchild = child
            .with_constraint(&graph, constraint2, false)
            .unwrap()
            .unwrap();
        assert_eq!(
            grandchild.delta_pre_nodes(),
            &["linear1".to_string(), "linear1".to_string()]
        );
    }

    /// #cone-delta: `with_multi_constraints` (which delegates to
    /// `with_constraint` per split) appends one pre-node per constraint.
    #[test]
    fn test_delta_pre_nodes_tracking_multi_constraints() {
        let (graph, mut root) = simple_graph_and_domain();
        root.delta_pre_nodes.clear();

        let splits = vec![("relu1".to_string(), 0, 1.0), ("relu1".to_string(), 1, 0.8)];
        let children = root.with_multi_constraints(&graph, &splits, false).unwrap();
        assert!(!children.is_empty());
        for child in &children {
            assert_eq!(
                child.delta_pre_nodes(),
                &["linear1".to_string(), "linear1".to_string()],
                "each multi-split child records one pre-node per constraint"
            );
        }
    }

    /// #cone-delta: a ReLU-shaped `with_general_split` appends the
    /// pre-activation node, exactly like `with_constraint`.
    #[test]
    fn test_delta_pre_nodes_tracking_general_split() {
        let (graph, mut root) = simple_graph_and_domain();
        root.delta_pre_nodes.clear();

        let split = NeuronSplit {
            layer: LayerRef::Name("relu1".to_string()),
            neuron_idx: 0,
            lower_bound: Some(0.0),
            upper_bound: None,
            score: 1.0,
            input_index: None,
            norm_inv_rms_window: None,
        };
        let child = root.with_general_split(&graph, split, false).unwrap();
        let child = child.expect("feasible child");
        assert_eq!(child.delta_pre_nodes(), &["linear1".to_string()]);
    }

    /// Regression test for #2599: with_general_split must reject NaN current bounds.
    ///
    /// nan_propagating_max/min correctly propagate NaN through the effective
    /// bound computation, and the explicit NaN guard catches the result.
    #[test]
    fn test_with_general_split_rejects_nan_current_lower() {
        let (graph, mut domain) = simple_graph_and_domain();

        let nan_bounds = BoundedTensor::new_unchecked(
            ndarray::arr1(&[f32::NAN, -0.5]).into_dyn(),
            ndarray::arr1(&[1.0, 1.0]).into_dyn(),
        )
        .unwrap();
        domain
            .node_bounds
            .insert("linear1".to_string(), Arc::new(nan_bounds));

        let split = NeuronSplit {
            layer: LayerRef::Name("relu1".to_string()),
            neuron_idx: 0,
            lower_bound: Some(0.0),
            upper_bound: None,
            score: 1.0,
            input_index: None,
            norm_inv_rms_window: None,
        };

        let result = domain.with_general_split(&graph, split, false);
        assert!(
            result.is_err(),
            "with_general_split should reject NaN current lower bound"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("NaN"), "error should mention NaN: {err}");
    }

    /// Regression test for #2599: with_general_split must reject NaN current upper.
    #[test]
    fn test_with_general_split_rejects_nan_current_upper() {
        let (graph, mut domain) = simple_graph_and_domain();

        let nan_bounds = BoundedTensor::new_unchecked(
            ndarray::arr1(&[-1.0, -0.5]).into_dyn(),
            ndarray::arr1(&[f32::NAN, 1.0]).into_dyn(),
        )
        .unwrap();
        domain
            .node_bounds
            .insert("linear1".to_string(), Arc::new(nan_bounds));

        let split = NeuronSplit {
            layer: LayerRef::Name("relu1".to_string()),
            neuron_idx: 0,
            lower_bound: None,
            upper_bound: Some(0.5),
            score: 1.0,
            input_index: None,
            norm_inv_rms_window: None,
        };

        let result = domain.with_general_split(&graph, split, false);
        assert!(
            result.is_err(),
            "with_general_split should reject NaN current upper bound"
        );
    }
}
