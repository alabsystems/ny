// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective graph branch-and-bound domain.
//!
//! Contains `MultiObjectiveGraphBabDomain` for disjunctive property verification
//! where ALL constraints must be verified simultaneously.

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

/// Domain for multi-objective GraphNetwork branch-and-bound.
///
/// Used for disjunctive properties where ALL constraints must be verified simultaneously.
/// Tracks bounds for each objective and only considers a domain verified when ALL
/// objectives are verified, enabling shared computation across objectives.
///
/// Fields are `pub(crate)` to enforce construction through validated methods
/// that reject NaN bounds (#3125, #2982).
#[derive(Debug, Clone)]
pub struct MultiObjectiveGraphBabDomain {
    /// Split history for this domain.
    pub(crate) history: GraphSplitHistory,
    /// Pre-activation bounds for each node.
    pub(crate) node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>>,
    /// Bounds (lower, upper) for each objective.
    pub(crate) objective_bounds: Vec<(f32, f32)>,
    /// Which objectives are verified in this domain.
    pub(crate) verified: Vec<bool>,
    /// Current depth in the B&B tree.
    pub(crate) depth: usize,
    /// Priority for queue ordering (max gap across unverified objectives).
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
    pub(crate) cached_las: Vec<Option<CachedLinearBounds>>,
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
    /// Create root domain with initial bounds for all objectives.
    /// Returns `Err(NumericalInstability)` if any objective bounds contain NaN (#2982).
    pub fn root(
        node_bounds: std::collections::HashMap<String, BoundedTensor>,
        objective_bounds: Vec<(f32, f32)>,
        input: &BoundedTensor,
        thresholds: &[f32],
        verify_upper: bool,
    ) -> Result<Self> {
        let node_bounds = node_bounds
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();

        // Defense-in-depth: entry points validate, but assert here too (#3383).
        debug_assert_eq!(
            objective_bounds.len(),
            thresholds.len(),
            "root(): objective_bounds/thresholds length mismatch ({} vs {})",
            objective_bounds.len(),
            thresholds.len()
        );
        // Check which objectives are already verified
        let verified: Vec<bool> = objective_bounds
            .iter()
            .zip(thresholds.iter())
            .map(|((l, u), &t)| {
                BetaCrownConfig::domain_is_verified_for_mode(verify_upper, *l, *u, t)
            })
            .collect();

        // Priority: max gap across unverified objectives.
        // NaN in any objective bound is rejected (#2982).
        let priority = objective_bounds
            .iter()
            .zip(verified.iter())
            .filter(|(_, &v)| !v)
            .try_fold(f32::NEG_INFINITY, |acc, ((l, u), _)| {
                let p = BetaCrownConfig::domain_priority_for_mode(verify_upper, *l, *u)?;
                Ok(nan_propagating_max(acc, p))
            })?;

        let num_objectives = objective_bounds.len();
        Ok(Self {
            history: GraphSplitHistory::new(),
            node_bounds,
            objective_bounds,
            verified,
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
        debug_assert_eq!(
            self.objective_bounds.len(),
            thresholds.len(),
            "all_violated(): objective_bounds/thresholds length mismatch (#3383)"
        );
        !self.objective_bounds.is_empty()
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
    /// Returns `Ok(None)` if infeasible, `Err(...)` on internal error (#2302).
    pub fn with_constraint(
        &self,
        graph: &GraphNetwork,
        constraint: GraphNeuronConstraint,
        verify_upper: bool,
        _thresholds: &[f32],
    ) -> Result<Option<Self>> {
        let node_name = &constraint.node_name;
        let neuron_idx = constraint.neuron_idx;
        let is_active = constraint.is_active;

        // Constraints are on the *pre-activation* of this ReLU/Sign (#3769)
        let relu_node = match graph.nodes.get(node_name) {
            Some(node) => node,
            None => return Ok(None),
        };
        if !matches!(relu_node.layer, Layer::ReLU(_) | Layer::Sign(_)) {
            return Ok(None);
        }
        // #2098: Return None for nodes with empty inputs instead of fabricating NETWORK_INPUT.
        let pre_name = match relu_node.inputs.first().map(|s| s.as_str()) {
            Some(name) => name,
            None => {
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

        // Compute priority based on unverified objectives.
        // NaN in any objective bound is rejected (#2982).
        let priority = self
            .objective_bounds
            .iter()
            .zip(self.verified.iter())
            .filter(|(_, &v)| !v)
            .try_fold(f32::NEG_INFINITY, |acc, ((l, u), _)| {
                let p = BetaCrownConfig::domain_priority_for_mode(verify_upper, *l, *u)?;
                Ok(nan_propagating_max(acc, p))
            })?;

        // Initialize β state with warmup from parent domain.
        // This inherits optimized β values for existing constraints while
        // initializing the new constraint's β to the default value.
        let beta_state = GraphBetaState::from_history_with_warmup(
            &new_history,
            &self.beta_state,
            GraphBetaState::DEFAULT_BETA_INIT,
        )?;
        let alpha_state = GraphDomainAlphaState::from_parent(
            &self.alpha_state,
            graph,
            &self.node_bounds,
            &new_history,
            new_input_bounds.as_ref(),
        );
        // Inherit per-disjunct alphas from parent, building child states (#4355).
        let per_disjunct_alphas = self.per_disjunct_alphas.as_ref().map(|parent_alphas| {
            parent_alphas
                .iter()
                .map(|pa| {
                    GraphDomainAlphaState::from_parent(
                        pa,
                        graph,
                        &self.node_bounds,
                        &new_history,
                        new_input_bounds.as_ref(),
                    )
                })
                .collect()
        });

        Ok(Some(Self {
            history: new_history,
            node_bounds: self.node_bounds.clone(),
            objective_bounds: self.objective_bounds.clone(),
            verified: self.verified.clone(),
            depth: self.depth + 1,
            priority,
            input_bounds: new_input_bounds,
            beta_state,
            alpha_state,
            per_disjunct_alphas,
            cached_las: self.cached_las.clone(), // Inherit parent's lA cache (#3813)
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
        debug_assert_eq!(
            new_bounds.len(),
            thresholds.len(),
            "update_bounds(): new_bounds/thresholds length mismatch (#3383)"
        );
        self.objective_bounds = new_bounds;
        self.verified = self
            .objective_bounds
            .iter()
            .zip(thresholds.iter())
            .map(|((l, u), &t)| {
                BetaCrownConfig::domain_is_verified_for_mode(verify_upper, *l, *u, t)
            })
            .collect();

        // Update priority. NaN in any objective bound is rejected (#2982).
        self.priority = self
            .objective_bounds
            .iter()
            .zip(self.verified.iter())
            .filter(|(_, &v)| !v)
            .try_fold(f32::NEG_INFINITY, |acc, ((l, u), _)| {
                let p = BetaCrownConfig::domain_priority_for_mode(verify_upper, *l, *u)?;
                Ok(nan_propagating_max(acc, p))
            })?;
        Ok(())
    }

    /// Check if any objective is conclusively violated (provably cannot be verified).
    pub fn any_violated(&self, thresholds: &[f32], verify_upper: bool) -> bool {
        debug_assert_eq!(
            self.objective_bounds.len(),
            thresholds.len(),
            "any_violated(): objective_bounds/thresholds length mismatch (#3383)"
        );
        self.objective_bounds
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

    /// Compute priority for conjunctive mode: min gap across unverified objectives.
    ///
    /// The most constrained (tightest) spec determines how much work this domain
    /// needs. Uses min instead of max because for conjunctive properties, we only
    /// need ONE objective verified — so the closest-to-verified objective is the
    /// most promising direction.
    ///
    /// Note: the design doc suggests starting with max-gap to match alpha-beta-CROWN
    /// reference. We use the existing max-gap priority (same as disjunctive) for now
    /// and can experiment with min-gap later if needed.
    pub fn conjunctive_priority(
        objective_bounds: &[(f32, f32)],
        verified: &[bool],
        verify_upper: bool,
    ) -> Result<f32> {
        objective_bounds
            .iter()
            .zip(verified.iter())
            .filter(|(_, &v)| !v)
            .try_fold(f32::INFINITY, |acc, ((l, u), _)| {
                let p = BetaCrownConfig::domain_priority_for_mode(verify_upper, *l, *u)?;
                Ok(nan_propagating_min(acc, p))
            })
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

    /// Per-objective cached linear bound coefficients.
    pub fn cached_las(&self) -> &[Option<CachedLinearBounds>] {
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
        self.cached_las = cached_las;
        Ok(())
    }

    /// Get the cached lA for a specific objective index.
    ///
    /// Returns `None` if the index is out of bounds or the cache is empty.
    pub fn cached_la_for_objective(&self, objective_idx: usize) -> Option<&CachedLinearBounds> {
        self.cached_las.get(objective_idx)?.as_ref()
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
            node_bounds: std::collections::HashMap::new(),
            objective_bounds,
            verified,
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

    // --- conjunctive_priority ---

    #[test]
    fn test_conjunctive_priority_returns_min_gap() {
        let bounds = vec![(0.0, 1.0), (0.0, 5.0)];
        let verified = vec![false, false];
        let p =
            MultiObjectiveGraphBabDomain::conjunctive_priority(&bounds, &verified, false).unwrap();
        assert!(p.is_finite(), "conjunctive priority should be finite");
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
    fn test_with_constraint_preserves_cached_las() {
        let (graph, mut domain) = make_graph_and_domain_for_constraint_test();

        // Populate one objective's cache.
        let mut test_cache = CachedLinearBounds::default();
        test_cache
            .lower_a
            .insert("relu1".to_string(), ndarray::arr2(&[[1.0_f32, 0.0]]));
        domain.cached_las[0] = Some(test_cache);

        let constraint = GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        };
        let child = domain
            .with_constraint(&graph, constraint, false, &[0.0, 0.0])
            .expect("with_constraint should succeed")
            .expect("child should be feasible");

        assert_eq!(child.cached_las().len(), 2);
        assert!(child.cached_las()[0].is_some());
        assert!(child.cached_las()[1].is_none());
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
