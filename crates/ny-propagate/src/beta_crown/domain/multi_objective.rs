// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective graph branch-and-bound domain.
//!
//! Contains `MultiObjectiveGraphBabDomain` for multi-objective property
//! verification. Queue ordering and objective-directed branch guidance share
//! one aggregation-aware critical-margin policy.

use std::sync::Arc;

use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::GraphNetwork;
use crate::Layer;
use crate::NETWORK_INPUT;

use super::{NodeBoundsHostAllocationObservationV1, NodeBoundsMap};

/// How objective rows combine for domain-level proof progress.
///
/// This mode affects scheduling and advisory objective-row selection only.
/// Authoritative verification and violation predicates remain explicit at
/// their call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveAggregation {
    /// Every objective row must verify, so the minimum proof margin is critical.
    Disjunctive,
    /// Any objective row may verify, so the maximum proof margin is critical.
    Conjunctive,
}

impl ObjectiveAggregation {
    /// Translate the verifier's historical `conjunctive` flag into a typed mode.
    pub const fn from_conjunctive(conjunctive: bool) -> Self {
        if conjunctive {
            Self::Conjunctive
        } else {
            Self::Disjunctive
        }
    }

    #[inline]
    fn prefers_margin(self, candidate: f32, incumbent: f32) -> bool {
        match self {
            Self::Disjunctive => candidate < incumbent,
            Self::Conjunctive => candidate > incumbent,
        }
    }
}

/// Domain for multi-objective GraphNetwork branch-and-bound.
///
/// Tracks bounds for each objective and carries the aggregation mode used for
/// queue scheduling and objective-directed branch guidance.
///
/// Fields are `pub(crate)` to enforce construction through validated methods
/// that reject NaN bounds (#3125, #2982).
#[derive(Debug, Clone)]
pub struct MultiObjectiveGraphBabDomain {
    /// Split history for this domain.
    pub(crate) history: GraphSplitHistory,
    /// Pre-activation bounds for each node.
    pub(crate) node_bounds: NodeBoundsMap,
    /// Bounds (lower, upper) for each objective.
    pub(crate) objective_bounds: Vec<(f32, f32)>,
    /// Which objectives are verified in this domain.
    pub(crate) verified: Vec<bool>,
    /// Objective aggregation used by queue and advisory row selection.
    pub(crate) aggregation: ObjectiveAggregation,
    /// Bound direction used to compute row verification margins.
    pub(crate) verify_upper: bool,
    /// Current depth in the B&B tree.
    pub(crate) depth: usize,
    /// Priority for queue ordering (negative aggregation-critical proof margin).
    pub(crate) priority: f32,
    /// Input bounds for this domain.
    pub(crate) input_bounds: Arc<BoundedTensor>,
    /// β parameters for Lagrangian optimization (initialized from history).
    pub(crate) beta_state: GraphBetaState,
    /// Domain-specific α state for optimized ReLU relaxation slopes.
    ///
    /// This mirrors `GraphBabDomain::alpha_state` so multi-objective BaB can
    /// reuse root α-CROWN optimized slopes and warm-start child domains.
    pub(crate) alpha_state: GraphDomainAlphaState,
    /// Per-disjunct α states for `optimize_disjuncts_separately` mode (#4355).
    ///
    /// When populated, each entry is an α state optimized for a specific
    /// disjunct (OR clause). BaB evaluation runs separate 1-row CROWN
    /// backward passes, each using the corresponding disjunct's α.
    ///
    /// `None` when `optimize_disjuncts_separately` is disabled (shared α mode).
    ///
    /// Reference: alpha-beta-CROWN `optimize_disjuncts_separately`.
    pub(crate) per_disjunct_alphas: Option<Vec<GraphDomainAlphaState>>,
    /// Cached linear bound coefficients (lA) per objective.
    ///
    /// Each entry corresponds to one objective in `objective_bounds`. When
    /// present, the constrained backward pass can seed at the branch point
    /// instead of recomputing from the output node. Children inherit the
    /// parent's cache; it is replaced with freshly captured lA after each
    /// backward pass.
    ///
    /// Mirrors `GraphBabDomain::cached_la` but per-objective because the
    /// multi-objective path evaluates one objective at a time and each uses
    /// a different spec row.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/branching_domains.py:94-99` (all_lAs)
    /// Design: `designs/2026-03-15-issue-3813-multi-objective-la-warm-start.md`
    /// Issue: #3813
    /// The coefficient payload is immutable after publication and shared
    /// across descendants.  A CIFAR root carries one cache per objective over
    /// every graph node; deep-cloning those ndarrays for both children made
    /// frontier memory grow as objectives * nodes * children.
    pub(crate) cached_las: Vec<Option<Arc<CachedLinearBounds>>>,
    /// #cone-delta: pre-activation node names of the constraints added since
    /// `node_bounds` was last fixpointed.
    ///
    /// INVARIANT: `delta_pre_nodes` = the pre-activation nodes of every
    /// constraint appended to `history` since `node_bounds` was last replaced
    /// by a post-bounding fixpoint of the constrained forward pass.
    /// `with_constraint` appends the split's pre-activation node; every
    /// bounding path that replaces `node_bounds` clears the vec; the root
    /// carries the `delta_pre_nodes_unknown()` sentinel until the first
    /// bounding pass. Mirrors `GraphBabDomain::delta_pre_nodes` — see that
    /// field's doc for the full contract and the fail-closed gate.
    pub(crate) delta_pre_nodes: Vec<String>,
}

impl PartialEq for MultiObjectiveGraphBabDomain {
    fn eq(&self, other: &Self) -> bool {
        super::cmp_domain_priority(self.priority, other.priority) == std::cmp::Ordering::Equal
    }
}

impl Eq for MultiObjectiveGraphBabDomain {}

impl PartialOrd for MultiObjectiveGraphBabDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// #bab-dive DIAGNOSTIC (dark, `NY_BAB_DIVE=1`, default OFF ⇒ byte-identical).
/// Depth-first search order: pop the DEEPEST domain first (worst-margin
/// tiebreak at equal depth), forcing the worst frontier path down instead of
/// the default best-first breadth. SOUNDNESS: search order is advisory — it
/// changes only WHICH domain is refined next, never a bound or verdict (a
/// domain still verifies iff its own sound bound clears). Used to extend the
/// worst-child-vs-depth curve past the depth best-first reaches in budget, to
/// decide whether the cifar wall's per-subdomain LP crosses zero with depth
/// (throughput-bound) or asymptotes negative (relaxation-bound). Cached to
/// keep `cmp` (a heap hot path) branch-cheap.
fn bab_dive_enabled() -> bool {
    static DIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DIVE.get_or_init(|| std::env::var("NY_BAB_DIVE").ok().as_deref() == Some("1"))
}

impl Ord for MultiObjectiveGraphBabDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority = pop first
        if bab_dive_enabled() {
            // Deeper first; worst-margin (default priority) breaks ties.
            return self
                .depth
                .cmp(&other.depth)
                .then_with(|| super::cmp_domain_priority(self.priority, other.priority));
        }
        super::cmp_domain_priority(self.priority, other.priority)
    }
}

impl MultiObjectiveGraphBabDomain {
    /// Create a disjunctive root domain with initial bounds for all objectives.
    ///
    /// This compatibility constructor preserves the original public API.
    /// Conjunctive callers must use [`Self::root_with_aggregation`] explicitly.
    /// Returns `Err(NumericalInstability)` if any objective bounds contain NaN (#2982).
    pub fn root(
        node_bounds: std::collections::HashMap<String, BoundedTensor>,
        objective_bounds: Vec<(f32, f32)>,
        input: &BoundedTensor,
        thresholds: &[f32],
        verify_upper: bool,
    ) -> Result<Self> {
        Self::root_with_aggregation(
            node_bounds,
            objective_bounds,
            input,
            thresholds,
            verify_upper,
            ObjectiveAggregation::Disjunctive,
        )
    }

    /// Create a root domain with an explicit objective aggregation mode.
    ///
    /// Returns `Err(NumericalInstability)` if any objective bound, threshold,
    /// or derived proof margin is non-finite.
    pub fn root_with_aggregation(
        node_bounds: std::collections::HashMap<String, BoundedTensor>,
        objective_bounds: Vec<(f32, f32)>,
        input: &BoundedTensor,
        thresholds: &[f32],
        verify_upper: bool,
        aggregation: ObjectiveAggregation,
    ) -> Result<Self> {
        let node_bounds = node_bounds
            .into_iter()
            .map(|(name, bounds)| (name, Arc::new(bounds)))
            .collect();

        Self::root_with_shared_node_bounds_and_aggregation(
            node_bounds,
            objective_bounds,
            input,
            thresholds,
            verify_upper,
            aggregation,
        )
    }

    /// Create a root domain from already shared immutable node bounds.
    ///
    /// This crate-private ownership path lets graph setup and the root domain
    /// share the same tensor buffers. It performs the same validation and
    /// initialization as [`Self::root_with_aggregation`].
    pub(crate) fn root_with_shared_node_bounds_and_aggregation(
        node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>>,
        objective_bounds: Vec<(f32, f32)>,
        input: &BoundedTensor,
        thresholds: &[f32],
        verify_upper: bool,
        aggregation: ObjectiveAggregation,
    ) -> Result<Self> {
        if objective_bounds.is_empty() || objective_bounds.len() != thresholds.len() {
            return Err(NyError::InvalidSpec(format!(
                "root(): expected one threshold per non-empty objective row; bounds={}, thresholds={}",
                objective_bounds.len(),
                thresholds.len()
            )));
        }
        // Check which objectives are already verified
        let verified: Vec<bool> = objective_bounds
            .iter()
            .zip(thresholds.iter())
            .map(|((l, u), &t)| {
                BetaCrownConfig::domain_is_verified_for_mode(verify_upper, *l, *u, t)
            })
            .collect();

        let priority = Self::aggregation_priority(
            &objective_bounds,
            thresholds,
            &verified,
            verify_upper,
            aggregation,
        )?;

        let num_objectives = objective_bounds.len();
        Ok(Self {
            history: GraphSplitHistory::new(),
            node_bounds: NodeBoundsMap::from_shared_hash_map(node_bounds),
            objective_bounds,
            verified,
            aggregation,
            verify_upper,
            depth: 0,
            priority,
            input_bounds: Arc::new(input.clone()),
            beta_state: GraphBetaState::empty(), // Root has no constraints
            alpha_state: GraphDomainAlphaState::empty(),
            per_disjunct_alphas: None, // Set by caller when optimize_disjuncts_separately (#4355)
            cached_las: vec![None; num_objectives], // Root has no cached lA (#3813)
            // #cone-delta: the root map comes from the caller, not from a
            // post-bounding replacement — delta unknown until the first
            // bounding pass clears it.
            delta_pre_nodes: super::delta_pre_nodes_unknown(),
        })
    }

    /// Check if all objectives are verified (disjunctive stop criterion).
    ///
    /// Returns `false` for empty `verified` vec (defense-in-depth: empty
    /// objectives must not be treated as "all verified").
    pub fn all_verified(&self) -> bool {
        !self.verified.is_empty() && self.verified.iter().all(|&v| v)
    }

    /// Check if ANY objective is verified (conjunctive stop criterion).
    ///
    /// For conjunctive properties (AND of constraints), a subdomain is safe if
    /// at least one conjunct is proven impossible — the conjunction cannot hold.
    ///
    /// Returns `false` for empty `verified` vec (defense-in-depth).
    ///
    /// Reference: alpha-beta-CROWN `stop_criterion_batch_any` in
    /// `auto_LiRPA/utils.py:107-113`.
    pub fn any_verified(&self) -> bool {
        !self.verified.is_empty() && self.verified.iter().any(|&v| v)
    }

    /// Check if ALL objectives are conclusively violated.
    ///
    /// For conjunctive properties: all constraints might hold simultaneously →
    /// the conjunction might hold → subdomain is NOT safe.
    /// Only drop a domain when every single conjunct might still hold.
    ///
    /// Reference: alpha-beta-CROWN `multi_spec_keep_func_all` in
    /// `auto_LiRPA/utils.py:143-144`.
    pub fn all_violated(&self, thresholds: &[f32], verify_upper: bool) -> bool {
        if verify_upper != self.verify_upper {
            tracing::warn!(
                supplied_verify_upper = verify_upper,
                domain_verify_upper = self.verify_upper,
                "all_violated(): verification direction mismatch; retaining domain"
            );
            return false;
        }
        !self.objective_bounds.is_empty()
            && self.objective_bounds.len() == thresholds.len()
            && self
                .objective_bounds
                .iter()
                .zip(thresholds.iter())
                .all(|((l, u), &t)| {
                    BetaCrownConfig::domain_is_violation_for_mode(verify_upper, *l, *u, t)
                })
    }

    /// Count of verified objectives.
    pub fn verified_count(&self) -> usize {
        self.verified.iter().filter(|&&v| v).count()
    }

    /// Apply a constraint to create a child domain.
    ///
    /// Returns `Ok(None)` only when the requested half-space is infeasible.
    /// Structural inconsistencies are errors: treating a missing node/bound or
    /// invalid neuron index as an empty region could silently erase coverage
    /// from the BaB tree.
    pub fn with_constraint(
        &self,
        graph: &GraphNetwork,
        constraint: GraphNeuronConstraint,
        verify_upper: bool,
        thresholds: &[f32],
    ) -> Result<Option<Self>> {
        self.with_constraint_policy(graph, constraint, verify_upper, thresholds, true)
    }

    /// Create a child without cloning optional, potentially dense warm-start
    /// state. The bounded shared executor uses this before its first backend
    /// poll so inherited lA caches and per-disjunct alpha collections cannot
    /// multiply by both children in a wave. The mandatory shared alpha/beta
    /// state and every proof-relevant bound/history field are unchanged.
    pub(crate) fn with_constraint_without_optional_warm_starts(
        &self,
        graph: &GraphNetwork,
        constraint: GraphNeuronConstraint,
        verify_upper: bool,
        thresholds: &[f32],
    ) -> Result<Option<Self>> {
        self.with_constraint_policy(graph, constraint, verify_upper, thresholds, false)
    }

    fn with_constraint_policy(
        &self,
        graph: &GraphNetwork,
        constraint: GraphNeuronConstraint,
        verify_upper: bool,
        thresholds: &[f32],
        inherit_optional_warm_starts: bool,
    ) -> Result<Option<Self>> {
        if verify_upper != self.verify_upper {
            return Err(NyError::InternalError(format!(
                "with_constraint(): verification direction mismatch: domain={}, supplied={}",
                self.verify_upper, verify_upper
            )));
        }
        let node_name = &constraint.node_name;
        let neuron_idx = constraint.neuron_idx;
        let is_active = constraint.is_active;

        // Constraints are on the *pre-activation* of this ReLU/Sign (#3769)
        let relu_node = graph.nodes.get(node_name).ok_or_else(|| {
            NyError::InternalError(format!(
                "with_constraint (multi-obj): split node '{node_name}' is missing"
            ))
        })?;
        if !matches!(relu_node.layer, Layer::ReLU(_) | Layer::Sign(_)) {
            return Err(NyError::InternalError(format!(
                "with_constraint (multi-obj): split node '{node_name}' is not a ReLU or Sign"
            )));
        }
        let pre_name = relu_node
            .inputs
            .first()
            .map(String::as_str)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "with_constraint (multi-obj): split node '{node_name}' has no input"
                ))
            })?;

        let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
            self.input_bounds.as_ref()
        } else {
            self.node_bounds.get(pre_name).map(AsRef::as_ref).ok_or_else(|| {
                NyError::InternalError(format!(
                    "with_constraint (multi-obj): pre-activation bounds for '{pre_name}' are missing"
                ))
            })?
        };
        // #flatten-to-read-a-scalar: this used to call `pre_bounds.flatten()`,
        // which COPIES the entire tensor into two fresh `Vec<f32>`s
        // (`bounded_tensor/core/shape_ops.rs`), purely to read `len()` and two
        // elements. On a conv ResNet that is a full activation-sized allocation
        // per call, and the call is on a hot path: kFSB prepare evaluates it
        // `2k` times per domain (k=7 here) and the commit expands it
        // `2^(d+1) - 2` times per parent.
        //
        // `len()` is already O(1), and the flattened element is reachable
        // directly. `flatten` builds its vectors with `iter().copied()`, so
        // logical iteration order IS the flattened order and reading through the
        // same order is value-identical -- `as_slice()` is the O(1) path for the
        // contiguous standard layout, and the `iter().nth()` fallback keeps any
        // non-standard layout correct rather than silently wrong.
        let flat_at = |array: &ndarray::ArrayD<f32>, index: usize| -> f32 {
            array.as_slice().map_or_else(
                || array.iter().nth(index).copied().unwrap_or(f32::NAN),
                |slice| slice[index],
            )
        };

        if neuron_idx >= pre_bounds.len() {
            return Err(NyError::InternalError(format!(
                "with_constraint (multi-obj): neuron index {neuron_idx} is out of bounds for \
                 '{node_name}' (size {})",
                pre_bounds.len()
            )));
        }

        let l = flat_at(pre_bounds.lower(), neuron_idx);
        let u = flat_at(pre_bounds.upper(), neuron_idx);
        if !l.is_finite() || !u.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "with_constraint (multi-obj): non-finite neuron bounds for \
                 {node_name}[{neuron_idx}] (l={l}, u={u})"
            )));
        }
        if l > u {
            return Err(NyError::NumericalInstability(format!(
                "with_constraint (multi-obj): inverted neuron bounds for \
                 {node_name}[{neuron_idx}] (l={l}, u={u})"
            )));
        }

        // Feasibility check
        if is_active && u < 0.0 {
            return Ok(None);
        }
        if !is_active && l > 0.0 {
            return Ok(None);
        }

        // Update input bounds if constraining network input
        let new_input_bounds = if pre_name == NETWORK_INPUT {
            let shape = pre_bounds.shape().to_vec();
            // The flattened COPY is genuinely required here: this branch mutates
            // one element and reshapes the result back. It is now scoped to this
            // branch (splitting on the network input) instead of being paid by
            // every ReLU pre-activation split, which is the overwhelmingly
            // common case on a conv ResNet.
            let flat = pre_bounds.flatten();
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

            // Shape errors in flatten→reshape roundtrip indicate an internal bug (#2302).
            let lower_new = lower_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "with_constraint (multi-obj): child domain lower shape mismatch: {e}"
                    ))
                })?;
            let upper_new = upper_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "with_constraint (multi-obj): child domain upper shape mismatch: {e}"
                    ))
                })?;
            Arc::new(BoundedTensor::new(lower_new, upper_new)?)
        } else {
            self.input_bounds.clone()
        };

        let new_history = self.history.with_constraint(constraint);

        let priority = Self::aggregation_priority(
            &self.objective_bounds,
            thresholds,
            &self.verified,
            self.verify_upper,
            self.aggregation,
        )?;

        // Initialize β state with warmup from parent domain.
        // This inherits optimized β values for existing constraints while
        // initializing the new constraint's β to the default value.
        let beta_state = GraphBetaState::from_history_with_warmup(
            &new_history,
            &self.beta_state,
            GraphBetaState::DEFAULT_BETA_INIT,
        )?;
        let alpha_state = if inherit_optional_warm_starts {
            GraphDomainAlphaState::from_parent_node_bounds_map(
                &self.alpha_state,
                graph,
                &self.node_bounds,
                &new_history,
                new_input_bounds.as_ref(),
            )
        } else {
            // The bounded lane never enters analytical alpha optimization.
            // An empty state selects the independently sound heuristic ReLU
            // slopes at use time and avoids retaining two large HashMaps in
            // every frontier domain.
            GraphDomainAlphaState::empty()
        };
        // Inherit per-disjunct alphas from parent, building child states (#4355).
        let per_disjunct_alphas = inherit_optional_warm_starts
            .then(|| {
                self.per_disjunct_alphas.as_ref().map(|parent_alphas| {
                    parent_alphas
                        .iter()
                        .map(|pa| {
                            GraphDomainAlphaState::from_parent_node_bounds_map(
                                pa,
                                graph,
                                &self.node_bounds,
                                &new_history,
                                new_input_bounds.as_ref(),
                            )
                        })
                        .collect()
                })
            })
            .flatten();
        let cached_las = if inherit_optional_warm_starts {
            // Complete-Clip scores every objective row, including rows already
            // certified on this domain. A child can be a parent in the next
            // wave, so retain the full-spec vector while sharing every
            // immutable coefficient payload.
            self.cached_las.clone()
        } else {
            vec![None; self.cached_las.len()]
        };

        Ok(Some(Self {
            history: new_history,
            node_bounds: self.node_bounds.clone(),
            objective_bounds: self.objective_bounds.clone(),
            verified: self.verified.clone(),
            aggregation: self.aggregation,
            verify_upper: self.verify_upper,
            depth: self.depth + 1,
            priority,
            input_bounds: new_input_bounds,
            beta_state,
            alpha_state,
            per_disjunct_alphas,
            cached_las,
            // #cone-delta: the child inherits `node_bounds` verbatim, so its
            // delta = parent delta + this split's pre-activation node.
            delta_pre_nodes: {
                let mut delta = self.delta_pre_nodes.clone();
                delta.push(pre_name.to_string());
                delta
            },
        }))
    }

    /// Update bounds and verification status after propagation.
    ///
    /// Returns `Err(NumericalInstability)` if any objective bounds contain NaN (#2982).
    pub fn update_bounds(
        &mut self,
        new_bounds: Vec<(f32, f32)>,
        thresholds: &[f32],
        verify_upper: bool,
    ) -> Result<()> {
        if verify_upper != self.verify_upper {
            return Err(NyError::InternalError(format!(
                "update_bounds(): verification direction mismatch: domain={}, supplied={}",
                self.verify_upper, verify_upper
            )));
        }
        if new_bounds.is_empty() || new_bounds.len() != thresholds.len() {
            return Err(NyError::InvalidSpec(format!(
                "update_bounds(): expected one threshold per non-empty objective row; bounds={}, thresholds={}",
                new_bounds.len(),
                thresholds.len()
            )));
        }
        let new_verified: Vec<bool> = new_bounds
            .iter()
            .zip(thresholds.iter())
            .map(|((l, u), &t)| {
                BetaCrownConfig::domain_is_verified_for_mode(self.verify_upper, *l, *u, t)
            })
            .collect();
        let new_priority = Self::aggregation_priority(
            &new_bounds,
            thresholds,
            &new_verified,
            self.verify_upper,
            self.aggregation,
        )?;
        self.objective_bounds = new_bounds;
        self.verified = new_verified;
        self.priority = new_priority;
        Ok(())
    }

    /// Check if any objective is conclusively violated (provably cannot be verified).
    pub fn any_violated(&self, thresholds: &[f32], verify_upper: bool) -> bool {
        if verify_upper != self.verify_upper {
            tracing::warn!(
                supplied_verify_upper = verify_upper,
                domain_verify_upper = self.verify_upper,
                "any_violated(): verification direction mismatch; retaining domain"
            );
            return false;
        }
        self.objective_bounds.len() == thresholds.len()
            && self
                .objective_bounds
                .iter()
                .zip(thresholds.iter())
                .any(|((l, u), &t)| {
                    BetaCrownConfig::domain_is_violation_for_mode(verify_upper, *l, *u, t)
                })
    }

    /// Lower bound on the first objective. Debug-asserts non-empty (#3263 F2).
    pub fn lower_bound(&self) -> f32 {
        debug_assert!(!self.objective_bounds.is_empty(), "empty objectives");
        self.objective_bounds
            .first()
            .map(|(l, _)| *l)
            .unwrap_or(f32::NAN)
    }

    /// Upper bound on the first objective. Debug-asserts non-empty (#3263 F2).
    pub fn upper_bound(&self) -> f32 {
        debug_assert!(!self.objective_bounds.is_empty(), "empty objectives");
        self.objective_bounds
            .first()
            .map(|(_, u)| *u)
            .unwrap_or(f32::NAN)
    }

    /// Priority for queue ordering.
    pub fn priority(&self) -> f32 {
        self.priority
    }

    /// Historical zero-threshold conjunctive queue priority.
    ///
    /// New domains carry their aggregation mode and thresholds, so callers
    /// should use [`Self::priority`]. This wrapper remains for downstream API
    /// compatibility. It preserves the threshold-free priority rule and
    /// `INFINITY` for no unverified row, but rejects mismatched layouts rather
    /// than silently zip-truncating objective state.
    #[deprecated(
        note = "use aggregation-aware MultiObjectiveGraphBabDomain::priority on a validated domain"
    )]
    pub fn conjunctive_priority(
        objective_bounds: &[(f32, f32)],
        verified: &[bool],
        verify_upper: bool,
    ) -> Result<f32> {
        if objective_bounds.len() != verified.len() {
            return Err(NyError::InvalidSpec(format!(
                "conjunctive priority length mismatch: bounds={}, verified={}",
                objective_bounds.len(),
                verified.len()
            )));
        }
        objective_bounds
            .iter()
            .zip(verified.iter())
            .filter(|(_, &v)| !v)
            .try_fold(f32::INFINITY, |acc, ((l, u), _)| {
                let priority = BetaCrownConfig::domain_priority_for_mode(verify_upper, *l, *u)?;
                Ok(nan_propagating_min(acc, priority))
            })
    }

    /// Aggregation mode used for queue and advisory objective-row selection.
    pub fn aggregation(&self) -> ObjectiveAggregation {
        self.aggregation
    }

    /// Bound direction used for queue and advisory objective-row selection.
    pub fn verify_upper(&self) -> bool {
        self.verify_upper
    }

    /// Return the aggregation-critical unverified objective row.
    ///
    /// A row's proof margin is `lower - threshold` in lower-bound mode and
    /// `threshold - upper` in upper-bound mode. Disjunction selects the minimum
    /// margin (all rows must verify); conjunction selects the maximum (any row
    /// may verify). Equal margins retain the lowest row index deterministically.
    ///
    /// This is scheduling/advisory metadata only and is never a verdict source.
    pub(crate) fn critical_objective_index(&self, thresholds: &[f32]) -> Result<Option<usize>> {
        Ok(Self::critical_margin_and_index(
            &self.objective_bounds,
            thresholds,
            &self.verified,
            self.verify_upper,
            self.aggregation,
        )?
        .map(|(index, _)| index))
    }

    /// Compute max-heap priority from the same critical row used by branching.
    ///
    /// Higher priority means a smaller aggregation-level proof margin, i.e. the
    /// globally harder domain is processed first. Fully verified/empty row sets
    /// retain the historical `NEG_INFINITY` sentinel.
    fn aggregation_priority(
        objective_bounds: &[(f32, f32)],
        thresholds: &[f32],
        verified: &[bool],
        verify_upper: bool,
        aggregation: ObjectiveAggregation,
    ) -> Result<f32> {
        Ok(
            match Self::critical_margin_and_index(
                objective_bounds,
                thresholds,
                verified,
                verify_upper,
                aggregation,
            )? {
                Some((_, margin)) => -margin,
                None => f32::NEG_INFINITY,
            },
        )
    }

    fn critical_margin_and_index(
        objective_bounds: &[(f32, f32)],
        thresholds: &[f32],
        verified: &[bool],
        verify_upper: bool,
        aggregation: ObjectiveAggregation,
    ) -> Result<Option<(usize, f32)>> {
        if objective_bounds.len() != thresholds.len() || objective_bounds.len() != verified.len() {
            return Err(NyError::InvalidSpec(format!(
                "multi-objective critical margin length mismatch: bounds={}, thresholds={}, \
                 verified={}",
                objective_bounds.len(),
                thresholds.len(),
                verified.len()
            )));
        }

        let mut critical: Option<(usize, f32)> = None;
        for (index, (((lower, upper), &threshold), &row_verified)) in objective_bounds
            .iter()
            .zip(thresholds)
            .zip(verified)
            .enumerate()
        {
            if !lower.is_finite() || !upper.is_finite() || !threshold.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "multi-objective row {index} is non-finite: lower={lower}, \
                     upper={upper}, threshold={threshold}"
                )));
            }
            if lower > upper {
                return Err(NyError::NumericalInstability(format!(
                    "multi-objective row {index} has inverted bounds: lower={lower}, upper={upper}"
                )));
            }
            if row_verified {
                continue;
            }
            let margin = if verify_upper {
                threshold - upper
            } else {
                lower - threshold
            };
            if !margin.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "multi-objective row {index} proof margin is non-finite: \
                     lower={lower}, upper={upper}, threshold={threshold}"
                )));
            }
            if critical.is_none_or(|(_, incumbent)| aggregation.prefers_margin(margin, incumbent)) {
                critical = Some((index, margin));
            }
        }
        Ok(critical)
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
    ///
    /// This intentionally returns the opaque read-only carrier rather than the
    /// former concrete standard map. Lookup and unordered iteration are
    /// preserved; raw table mutation is not part of the corrected API.
    pub fn node_bounds(&self) -> &NodeBoundsMap {
        &self.node_bounds
    }

    /// Observe the node-bound map's narrow, non-authorizing host-allocation
    /// model. This does not account any other domain field and cannot select or
    /// open a retained runtime.
    pub fn node_bounds_host_allocation_observation_v1(
        &self,
    ) -> NodeBoundsHostAllocationObservationV1<'_> {
        self.node_bounds.host_allocation_observation_v1()
    }

    /// #cone-delta: pre-activation nodes of constraints added since
    /// `node_bounds` was last fixpointed (see the field invariant).
    pub fn delta_pre_nodes(&self) -> &[String] {
        &self.delta_pre_nodes
    }

    /// Bounds for each objective.
    pub fn objective_bounds(&self) -> &[(f32, f32)] {
        &self.objective_bounds
    }

    /// Which objectives are verified.
    pub fn verified(&self) -> &[bool] {
        &self.verified
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

    /// Mutable access to α state (for optimization and test setup).
    pub fn alpha_state_mut(&mut self) -> &mut GraphDomainAlphaState {
        &mut self.alpha_state
    }

    /// Set α state (for root initialization after α-CROWN pass).
    pub fn set_alpha_state(&mut self, state: GraphDomainAlphaState) {
        self.alpha_state = state;
    }

    /// Per-disjunct α states for `optimize_disjuncts_separately` mode (#4355).
    pub fn per_disjunct_alphas(&self) -> Option<&[GraphDomainAlphaState]> {
        self.per_disjunct_alphas.as_deref()
    }

    /// Set per-disjunct α states (for root initialization after per-disjunct α-CROWN).
    pub fn set_per_disjunct_alphas(&mut self, alphas: Vec<GraphDomainAlphaState>) {
        self.per_disjunct_alphas = Some(alphas);
    }

    /// Invalidate objective-local alpha transport after replacing the shared
    /// root alpha state with a different certified state.
    pub(crate) fn clear_per_disjunct_alphas(&mut self) {
        self.per_disjunct_alphas = None;
    }

    /// Prepare a root domain for the bounded executor's compact frontier.
    ///
    /// These fields affect relaxation choice/reuse/performance only. An empty
    /// alpha state uses the sound heuristic fallback; objective bounds, proof
    /// masks, split history, and beta state retain their full authority.
    pub(crate) fn prepare_for_bounded_executor(&mut self) {
        self.per_disjunct_alphas = None;
        self.cached_las.fill(None);
        self.alpha_state = GraphDomainAlphaState::empty();
    }

    /// Build a lightweight, equal-region shell for a parent-wide verified
    /// close. The caller must publish it only after every objective is
    /// certified: continuation state is deliberately absent, so this domain
    /// is suitable for clause/cut history recording but must never re-enter
    /// propagation or the frontier.
    ///
    /// Avoiding the derived `Clone` here is important on near-terminal CIFAR
    /// parents, where node bounds, alpha/beta state, and cached lA can occupy
    /// tens of MiB even though a verified close consumes only history.
    pub(crate) fn clone_for_verified_close(&self) -> Self {
        Self {
            history: self.history.clone(),
            node_bounds: NodeBoundsMap::new(),
            objective_bounds: self.objective_bounds.clone(),
            verified: self.verified.clone(),
            aggregation: self.aggregation,
            verify_upper: self.verify_upper,
            depth: self.depth,
            priority: self.priority,
            input_bounds: self.input_bounds.clone(),
            beta_state: GraphBetaState::empty(),
            alpha_state: GraphDomainAlphaState::empty(),
            per_disjunct_alphas: None,
            cached_las: vec![None; self.cached_las.len()],
            // A verified shell cannot propagate. If a future regression tries,
            // the unknown sentinel keeps every delta-based cache gate closed.
            delta_pre_nodes: super::delta_pre_nodes_unknown(),
        }
    }

    /// Per-objective cached linear bound coefficients.
    ///
    /// Entries are immutable, `Arc`-shared payloads. Callers that only need a
    /// borrowed [`CachedLinearBounds`] can use `Option::as_deref` on a slot (or
    /// [`Self::cached_la_for_objective`]) without cloning the coefficient
    /// ndarrays. The shared wrapper is part of this accessor's return type so a
    /// descendant/queue transport can preserve allocation identity.
    pub fn cached_las(&self) -> &[Option<Arc<CachedLinearBounds>>] {
        &self.cached_las
    }

    /// Set per-objective cached linear bounds (for root capture after
    /// spec-guided CROWN pass). Validates length matches objective count.
    ///
    /// # Errors
    /// Returns `InvalidSpec` if `cached_las.len() != self.objective_bounds.len()`.
    pub fn set_cached_las(&mut self, cached_las: Vec<Option<CachedLinearBounds>>) -> Result<()> {
        if cached_las.len() != self.objective_bounds.len() {
            return Err(NyError::InvalidSpec(format!(
                "cached_las length {} != objective_bounds length {} (#3813)",
                cached_las.len(),
                self.objective_bounds.len()
            )));
        }
        self.cached_las = cached_las
            .into_iter()
            .map(|cache| cache.map(Arc::new))
            .collect();
        Ok(())
    }

    /// Install already-shared per-objective caches after an active-row merge.
    ///
    /// Unchanged inherited rows retain their `Arc` identity; freshly captured
    /// rows are wrapped exactly once by the merge helper.
    pub(crate) fn set_shared_cached_las(
        &mut self,
        cached_las: Vec<Option<Arc<CachedLinearBounds>>>,
    ) -> Result<()> {
        if cached_las.len() != self.objective_bounds.len() {
            return Err(NyError::InvalidSpec(format!(
                "cached_las length {} != objective_bounds length {} (#3813)",
                cached_las.len(),
                self.objective_bounds.len()
            )));
        }
        self.cached_las = cached_las;
        Ok(())
    }

    /// Get the cached lA for a specific objective index.
    ///
    /// Returns `None` if the index is out of bounds or the cache is empty.
    pub fn cached_la_for_objective(&self, objective_idx: usize) -> Option<&CachedLinearBounds> {
        self.cached_las.get(objective_idx)?.as_deref()
    }

    /// Invalidate one objective's linear warm-start after publishing a bound
    /// paired with a different alpha state.
    ///
    /// Out-of-range indices are a no-op; callers already validate the
    /// objective index as part of their atomic publication.
    pub(crate) fn clear_cached_la_for_objective(&mut self, objective_idx: usize) {
        if let Some(cache) = self.cached_las.get_mut(objective_idx) {
            *cache = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a test domain with given objective bounds and verification flags.
    fn make_domain(
        objective_bounds: Vec<(f32, f32)>,
        verified: Vec<bool>,
    ) -> MultiObjectiveGraphBabDomain {
        let num_objectives = objective_bounds.len();
        MultiObjectiveGraphBabDomain {
            history: GraphSplitHistory::new(),
            node_bounds: NodeBoundsMap::new(),
            objective_bounds,
            verified,
            aggregation: ObjectiveAggregation::Disjunctive,
            verify_upper: false,
            depth: 0,
            priority: 0.0,
            input_bounds: Arc::new(
                BoundedTensor::new(
                    ndarray::arr1(&[0.0_f32]).into_dyn(),
                    ndarray::arr1(&[1.0_f32]).into_dyn(),
                )
                .expect("valid test tensor"),
            ),
            beta_state: GraphBetaState::empty(),
            alpha_state: GraphDomainAlphaState::empty(),
            per_disjunct_alphas: None,
            cached_las: vec![None; num_objectives],
            delta_pre_nodes: Vec::new(),
        }
    }

    #[test]
    fn root_with_shared_node_bounds_preserves_arc_identity() {
        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid input bounds");
        let shared = Arc::new(
            BoundedTensor::new(
                ndarray::arr1(&[-0.5_f32]).into_dyn(),
                ndarray::arr1(&[0.75_f32]).into_dyn(),
            )
            .expect("valid node bounds"),
        );
        let retained = Arc::clone(&shared);
        let node_bounds = std::collections::HashMap::from([("pre".to_string(), shared)]);

        let domain = MultiObjectiveGraphBabDomain::root_with_shared_node_bounds_and_aggregation(
            node_bounds,
            vec![(-0.25, 0.5)],
            &input,
            &[0.0],
            false,
            ObjectiveAggregation::Disjunctive,
        )
        .expect("shared root should be valid");

        assert!(Arc::ptr_eq(
            domain.node_bounds().get("pre").expect("shared pre bound"),
            &retained
        ));
    }

    #[test]
    fn root_with_shared_node_bounds_retains_root_validation() {
        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid input bounds");

        let error = MultiObjectiveGraphBabDomain::root_with_shared_node_bounds_and_aggregation(
            std::collections::HashMap::new(),
            vec![(f32::NAN, 0.5)],
            &input,
            &[0.0],
            false,
            ObjectiveAggregation::Disjunctive,
        )
        .expect_err("non-finite objective bounds must still be rejected");

        assert!(matches!(error, NyError::NumericalInstability(_)));
    }

    // --- all_verified (disjunctive stop criterion) ---

    #[test]
    fn test_all_verified_returns_false_for_empty_verified() {
        assert!(!make_domain(vec![], vec![]).all_verified());
    }

    #[test]
    fn test_all_verified_returns_true_when_all_objectives_verified() {
        assert!(make_domain(vec![(0.0, 1.0), (0.0, 1.0)], vec![true, true]).all_verified());
    }

    #[test]
    fn test_all_verified_returns_false_when_one_objective_unverified() {
        assert!(!make_domain(vec![(0.0, 1.0), (0.0, 1.0)], vec![true, false]).all_verified());
    }

    // --- any_verified (conjunctive stop criterion) ---

    #[test]
    fn test_any_verified_returns_true_when_one_objective_verified() {
        assert!(make_domain(vec![(0.0, 1.0), (0.0, 1.0)], vec![true, false]).any_verified());
    }

    #[test]
    fn test_any_verified_returns_false_when_none_verified() {
        assert!(!make_domain(vec![(0.0, 1.0), (0.0, 1.0)], vec![false, false]).any_verified());
    }

    #[test]
    fn test_any_verified_returns_false_for_empty_verified() {
        assert!(!make_domain(vec![], vec![]).any_verified());
    }

    // --- all_violated (conjunctive drop criterion) ---

    #[test]
    fn test_all_violated_returns_true_when_all_violated() {
        // upper < threshold=0 → violated for verify_upper=false
        let d = make_domain(vec![(-1.0, -0.5), (-2.0, -0.1)], vec![false, false]);
        assert!(d.all_violated(&[0.0, 0.0], false));
    }

    #[test]
    fn test_all_violated_returns_false_when_only_some_violated() {
        let d = make_domain(vec![(-1.0, -0.5), (-0.5, 0.5)], vec![false, false]);
        assert!(!d.all_violated(&[0.0, 0.0], false));
    }

    #[test]
    fn test_all_violated_returns_false_for_empty_objectives() {
        assert!(!make_domain(vec![], vec![]).all_violated(&[], false));
    }

    #[test]
    fn violation_predicates_fail_closed_on_threshold_length_mismatch() {
        let domain = make_domain(vec![(-2.0, -1.0), (-2.0, -1.0)], vec![false, false]);
        assert!(!domain.any_violated(&[0.0], false));
        assert!(!domain.all_violated(&[0.0], false));
    }

    #[test]
    fn violation_predicates_retain_domain_on_direction_mismatch() {
        // In upper-bound mode this row would be conclusively violated because
        // lower >= threshold. The domain is lower-bound-authoritative, so a
        // stale/opposite compatibility argument must never authorize a drop.
        let domain = make_domain(vec![(1.0, 2.0)], vec![false]);
        assert!(!domain.any_violated(&[0.0], true));
        assert!(!domain.all_violated(&[0.0], true));
    }

    #[test]
    fn update_bounds_direction_mismatch_is_transactional_error() {
        let mut domain = make_domain(vec![(-1.0, 1.0)], vec![false]);
        let before_bounds = domain.objective_bounds.clone();
        let before_verified = domain.verified.clone();
        let before_priority = domain.priority;

        let error = domain
            .update_bounds(vec![(1.0, 2.0)], &[0.0], true)
            .expect_err("opposite verification direction must be rejected");
        assert!(error.to_string().contains("direction mismatch"));
        assert_eq!(domain.objective_bounds, before_bounds);
        assert_eq!(domain.verified, before_verified);
        assert_eq!(domain.priority, before_priority);
    }

    #[test]
    fn update_bounds_shape_or_interval_failure_is_transactional() {
        let mut domain = make_domain(vec![(-1.0, 1.0)], vec![false]);
        let before = domain.clone();

        assert!(domain.update_bounds(vec![(1.0, 2.0)], &[], false).is_err());
        assert_eq!(domain.objective_bounds, before.objective_bounds);
        assert_eq!(domain.verified, before.verified);
        assert_eq!(domain.priority, before.priority);

        assert!(domain
            .update_bounds(vec![(1.0, 0.5)], &[0.0], false)
            .is_err());
        assert_eq!(domain.objective_bounds, before.objective_bounds);
        assert_eq!(domain.verified, before.verified);
        assert_eq!(domain.priority, before.priority);
    }

    #[allow(deprecated)]
    #[test]
    fn deprecated_conjunctive_priority_preserves_historical_behavior() {
        let bounds = [(2.0, 8.0), (-4.0, 5.0), (7.0, 9.0)];
        let verified = [false, false, true];

        assert_eq!(
            MultiObjectiveGraphBabDomain::conjunctive_priority(&bounds, &verified, false).unwrap(),
            -2.0,
            "lower mode historically minimized -lower across unverified rows"
        );
        assert_eq!(
            MultiObjectiveGraphBabDomain::conjunctive_priority(&bounds, &verified, true).unwrap(),
            5.0,
            "upper mode historically minimized upper across unverified rows"
        );
        assert_eq!(
            MultiObjectiveGraphBabDomain::conjunctive_priority(
                &bounds,
                &[true, true, true],
                false,
            )
            .unwrap(),
            f32::INFINITY,
            "no unverified row historically returned the fold identity"
        );
        assert!(
            MultiObjectiveGraphBabDomain::conjunctive_priority(&bounds, &[false], false,).is_err()
        );
    }

    // --- aggregation-aware critical margin ---

    #[test]
    fn critical_row_uses_aggregation_and_unequal_thresholds() {
        let input = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid test tensor");
        let bounds = vec![(9.0, 20.0), (-2.0, 5.0)];
        let thresholds = [10.0, 0.0];

        let disjunctive = MultiObjectiveGraphBabDomain::root_with_aggregation(
            std::collections::HashMap::new(),
            bounds.clone(),
            &input,
            &thresholds,
            false,
            ObjectiveAggregation::Disjunctive,
        )
        .unwrap();
        let conjunctive = MultiObjectiveGraphBabDomain::root_with_aggregation(
            std::collections::HashMap::new(),
            bounds,
            &input,
            &thresholds,
            false,
            ObjectiveAggregation::Conjunctive,
        )
        .unwrap();

        // Proof margins are [-1, -2]. ALL-row verification is bottlenecked by
        // row 1; ANY-row verification should aim at row 0.
        assert_eq!(
            disjunctive.critical_objective_index(&thresholds).unwrap(),
            Some(1)
        );
        assert_eq!(
            conjunctive.critical_objective_index(&thresholds).unwrap(),
            Some(0)
        );
        assert_eq!(disjunctive.priority(), 2.0);
        assert_eq!(conjunctive.priority(), 1.0);
    }

    #[test]
    fn critical_row_upper_mode_uses_threshold_relative_margin() {
        let input = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid test tensor");
        let thresholds = [10.0, 0.0];
        let domain = MultiObjectiveGraphBabDomain::root_with_aggregation(
            std::collections::HashMap::new(),
            vec![(0.0, 11.0), (-3.0, 2.0)],
            &input,
            &thresholds,
            true,
            ObjectiveAggregation::Conjunctive,
        )
        .unwrap();

        // Upper-mode proof margins are also [-1, -2], so conjunction selects
        // row 0. A raw-upper comparison would incorrectly select row 1.
        assert_eq!(
            domain.critical_objective_index(&thresholds).unwrap(),
            Some(0)
        );
        assert_eq!(domain.priority(), 1.0);
    }

    #[test]
    fn aggregation_reverses_heap_order_for_unequal_thresholds() {
        let input = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid test tensor");
        let thresholds = [10.0, 0.0];
        let bounds_a = vec![(9.0, 20.0), (-2.0, 5.0)]; // margins [-1, -2]
        let bounds_b = vec![(5.0, 20.0), (-0.5, 5.0)]; // margins [-5, -0.5]
        let make = |bounds, aggregation| {
            MultiObjectiveGraphBabDomain::root_with_aggregation(
                std::collections::HashMap::new(),
                bounds,
                &input,
                &thresholds,
                false,
                aggregation,
            )
            .unwrap()
        };

        let mut disjunctive_heap = std::collections::BinaryHeap::new();
        disjunctive_heap.push(make(bounds_a.clone(), ObjectiveAggregation::Disjunctive));
        disjunctive_heap.push(make(bounds_b.clone(), ObjectiveAggregation::Disjunctive));
        assert_eq!(
            disjunctive_heap.pop().unwrap().objective_bounds()[0].0,
            5.0,
            "disjunction must pop the domain with the smaller minimum margin"
        );

        let mut conjunctive_heap = std::collections::BinaryHeap::new();
        conjunctive_heap.push(make(bounds_a, ObjectiveAggregation::Conjunctive));
        conjunctive_heap.push(make(bounds_b, ObjectiveAggregation::Conjunctive));
        assert_eq!(
            conjunctive_heap.pop().unwrap().objective_bounds()[0].0,
            9.0,
            "conjunction must pop the domain with the smaller maximum margin"
        );
    }

    // --- Ordering ---

    #[test]
    fn test_multi_objective_domain_cmp_treats_nan_priority_as_high_priority() {
        let input = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid test tensor");

        let finite = MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(0.0, 1.0)],
            &input,
            &[0.5],
            false,
        )
        .unwrap();
        let mut nan = finite.clone();
        nan.priority = f32::NAN;

        assert_eq!(nan.cmp(&finite), std::cmp::Ordering::Greater);
        assert_eq!(finite.cmp(&nan), std::cmp::Ordering::Less);

        let mut heap = std::collections::BinaryHeap::new();
        heap.push(finite);
        heap.push(nan);
        let popped = heap.pop().expect("heap should contain two domains");
        assert!(popped.priority.is_nan());
    }

    // --- cached_las (lA warm-start, #3813) ---

    #[test]
    fn test_root_cached_las_length_matches_num_objectives() {
        let input = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid test tensor");

        let domain = MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(0.0, 1.0), (-0.5, 0.5), (0.1, 0.9)],
            &input,
            &[0.0, 0.0, 0.0],
            false,
        )
        .unwrap();

        assert_eq!(
            domain.cached_las().len(),
            3,
            "root cached_las length must match num_objectives"
        );
        assert!(
            domain.cached_las().iter().all(|c| c.is_none()),
            "root cached_las must all be None"
        );
    }

    /// Build a test graph + domain with pre-activation bounds for constraint tests.
    fn make_graph_and_domain_for_constraint_test() -> (GraphNetwork, MultiObjectiveGraphBabDomain) {
        use crate::layers::{Layer, LinearLayer, ReLULayer};
        use crate::network::GraphNode;

        let mut graph = GraphNetwork::new();
        let linear = LinearLayer::new(
            ndarray::arr2(&[[1.0_f32, -0.5], [0.25, 0.75]]),
            Some(ndarray::arr1(&[0.0_f32, 0.1])),
        )
        .expect("valid linear layer");
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear)));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        let linear2 = LinearLayer::new(
            ndarray::arr2(&[[1.0_f32, -1.0]]),
            Some(ndarray::arr1(&[0.0_f32])),
        )
        .unwrap();
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu1".to_string()],
        ));
        graph.set_output("linear2");

        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32, -0.5]).into_dyn(),
            ndarray::arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("valid input");
        let pre_act = BoundedTensor::new(
            ndarray::arr1(&[-1.5_f32, -0.275]).into_dyn(),
            ndarray::arr1(&[0.75_f32, 0.6625]).into_dyn(),
        )
        .expect("valid pre-activation bounds");
        let mut node_bounds = std::collections::HashMap::new();
        node_bounds.insert("linear1".to_string(), pre_act);

        let domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(0.0, 1.0), (-0.5, 0.5)],
            &input,
            &[0.0, 0.0],
            false,
        )
        .unwrap();
        (graph, domain)
    }

    #[test]
    fn with_constraint_arc_shares_full_spec_cache_for_next_wave() {
        let (graph, mut domain) = make_graph_and_domain_for_constraint_test();

        let mut active_cache = CachedLinearBounds::default();
        active_cache
            .lower_a
            .insert("relu1".to_string(), ndarray::arr2(&[[1.0_f32, 0.0]]));
        active_cache
            .lower_b
            .insert("relu1".to_string(), ndarray::arr1(&[0.125_f32]));
        let mut verified_cache = CachedLinearBounds::default();
        verified_cache
            .lower_a
            .insert("relu1".to_string(), ndarray::arr2(&[[9.0_f32, 8.0]]));
        domain
            .set_cached_las(vec![Some(active_cache), Some(verified_cache)])
            .expect("cache shape should match objectives");
        // Model the root census after one objective has already closed.
        // Complete-Clip still averages all rows when this child becomes a
        // parent in the next wave.
        domain.verified[1] = true;

        let parent_caches = domain
            .cached_las()
            .iter()
            .map(|cache| cache.as_ref().expect("full-spec cache should exist"))
            .collect::<Vec<_>>();
        let parent_payloads = parent_caches
            .iter()
            .map(|cache| cache.lower_a["relu1"].as_ptr())
            .collect::<Vec<_>>();
        let expected_active_bias_bits = parent_caches[0].lower_b["relu1"][0].to_bits();

        let active_constraint = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        };
        let inactive_constraint = GraphNeuronConstraint {
            is_active: false,
            ..active_constraint.clone()
        };
        let active_child = domain
            .with_constraint(&graph, active_constraint, false, &[0.0, 0.0])
            .expect("with_constraint should succeed")
            .expect("active child should be feasible");
        let inactive_child = domain
            .with_constraint(&graph, inactive_constraint, false, &[0.0, 0.0])
            .expect("with_constraint should succeed")
            .expect("inactive child should be feasible");

        for child in [&active_child, &inactive_child] {
            assert_eq!(child.cached_las().len(), 2);
            assert!(child.cached_las().iter().all(Option::is_some));
            for (idx, parent_cache) in parent_caches.iter().enumerate() {
                let child_cache = child.cached_las()[idx]
                    .as_ref()
                    .expect("full-spec cache should be inherited");
                assert!(Arc::ptr_eq(parent_cache, child_cache));
                assert_eq!(child_cache.lower_a["relu1"].as_ptr(), parent_payloads[idx]);
            }
            assert_eq!(
                child.cached_las()[0]
                    .as_ref()
                    .expect("active cache should exist")
                    .lower_b["relu1"][0]
                    .to_bits(),
                expected_active_bias_bits
            );
        }
        for cache in parent_caches {
            assert_eq!(
                Arc::strong_count(cache),
                3,
                "one immutable payload should be shared by parent and both children"
            );
        }
    }

    #[test]
    fn bounded_constraint_drops_only_optional_warm_starts() {
        let (graph, mut domain) = make_graph_and_domain_for_constraint_test();
        let mut test_cache = CachedLinearBounds::default();
        test_cache
            .lower_a
            .insert("relu1".to_string(), ndarray::arr2(&[[1.0_f32, 0.0]]));
        domain.cached_las[0] = Some(Arc::new(test_cache));
        domain.set_per_disjunct_alphas(vec![GraphDomainAlphaState::empty()]);

        let constraint = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        };
        let child = domain
            .with_constraint_without_optional_warm_starts(&graph, constraint, false, &[0.0, 0.0])
            .expect("bounded child construction should succeed")
            .expect("child should be feasible");

        assert!(child.cached_las().iter().all(Option::is_none));
        assert!(child.per_disjunct_alphas().is_none());
        assert!(child.alpha_state().is_empty());
        assert_eq!(child.objective_bounds(), domain.objective_bounds());

        domain.prepare_for_bounded_executor();
        assert!(domain.cached_las().iter().all(Option::is_none));
        assert!(domain.per_disjunct_alphas().is_none());
        assert!(domain.alpha_state().is_empty());
    }

    #[test]
    fn with_constraint_rejects_direction_mismatch() {
        let (graph, domain) = make_graph_and_domain_for_constraint_test();
        let constraint = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        };

        let error = domain
            .with_constraint(&graph, constraint, true, &[0.0, 0.0])
            .expect_err("opposite verification direction must be rejected");
        assert!(error.to_string().contains("direction mismatch"));
    }

    #[test]
    fn with_constraint_rejects_inverted_preactivation_bounds() {
        let (graph, mut domain) = make_graph_and_domain_for_constraint_test();
        let inverted = BoundedTensor::new_unchecked(
            ndarray::arr1(&[1.0_f32, -0.25]).into_dyn(),
            ndarray::arr1(&[0.5_f32, 0.75]).into_dyn(),
        )
        .unwrap();
        domain
            .node_bounds
            .insert("linear1".to_string(), Arc::new(inverted));
        let constraint = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        };

        let error = domain
            .with_constraint(&graph, constraint, false, &[0.0, 0.0])
            .expect_err("inverted preactivation bounds must be rejected");
        assert!(error.to_string().contains("inverted"));
    }

    #[test]
    fn test_set_cached_las_validates_length() {
        let mut domain = make_domain(vec![(0.0, 1.0), (-0.5, 0.5)], vec![false, false]);

        // Correct length succeeds.
        assert!(domain.set_cached_las(vec![None, None]).is_ok());

        // Wrong length fails.
        let err = domain.set_cached_las(vec![None]);
        assert!(err.is_err(), "set_cached_las with wrong length must fail");
    }

    #[test]
    fn test_cached_la_for_objective_returns_none_for_empty_cache() {
        let domain = make_domain(vec![(0.0, 1.0), (-0.5, 0.5)], vec![false, false]);

        assert!(domain.cached_la_for_objective(0).is_none());
        assert!(domain.cached_la_for_objective(1).is_none());
        assert!(domain.cached_la_for_objective(99).is_none());
    }
}
