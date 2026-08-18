// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential network branch-and-bound domain types.
//!
//! - `IntermediateLinearBounds`: Stores intermediate backward pass bounds for transfer
//! - `BabDomain`: Domain for sequential (non-graph) network BaB

use std::sync::Arc;

use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::{AlphaState, LinearBounds, Network};

/// Stores the LinearBounds at each layer during the backward pass, enabling
/// efficient bound transfer to child domains. When a domain is split on a
/// neuron at layer L, child domains can reuse the parent's intermediate
/// bounds for layers > L instead of recomputing the full backward pass.
///
/// Reference: α,β-CROWN `lAs` storage in `domain_updater.py`.
#[derive(Debug, Clone)]
pub struct IntermediateLinearBounds {
    /// LinearBounds at each layer index (0..num_layers).
    /// Index i holds the LinearBounds BEFORE processing layer i in the backward pass.
    /// This means bounds_at_layer[num_layers-1] = identity (output), and
    /// bounds_at_layer[L] = state after processing layers num_layers-1 down to L+1.
    ///
    /// CONVENTION: the coefficient columns of bounds_at_layer[i] range over
    /// layer i's OUTPUT activation (the input of layer i+1). Concretize
    /// bounds_at_layer[i] against layer_bounds[i] — layer i's output box —
    /// never against layer_bounds[i-1] or the network input, which bound
    /// layer i's INPUT (see engine/optimization/intermediate_merge.rs).
    ///
    /// Uses Arc for cheap cloning during splits.
    pub(crate) bounds_at_layer: Vec<Arc<LinearBounds>>,
    /// The layer index where the backward pass started (usually last layer).
    pub(crate) start_layer: usize,
}

impl IntermediateLinearBounds {
    /// Create empty intermediate bounds.
    pub fn empty() -> Self {
        Self {
            bounds_at_layer: Vec::new(),
            start_layer: 0,
        }
    }

    /// Check if intermediate bounds are available.
    pub fn is_empty(&self) -> bool {
        self.bounds_at_layer.is_empty()
    }

    /// Get the LinearBounds at a specific layer (if available).
    pub fn get(&self, layer_idx: usize) -> Option<&LinearBounds> {
        self.bounds_at_layer.get(layer_idx).map(|arc| arc.as_ref())
    }

    /// Get the bounds_at_layer slice (for sub-slicing during partial transfers).
    pub fn bounds_at_layer(&self) -> &[Arc<LinearBounds>] {
        &self.bounds_at_layer
    }

    /// Get the start layer index.
    pub fn start_layer(&self) -> usize {
        self.start_layer
    }
}

/// A domain in the branch-and-bound search tree.
///
/// Fields are `pub(crate)` to enforce construction through validated methods
/// that reject NaN bounds (#3125, #2982). Use accessor methods for reads
/// and `update_bounds()` for bound/priority updates.
#[derive(Debug, Clone)]
pub struct BabDomain {
    /// Split history for this domain.
    pub(crate) history: SplitHistory,
    /// Lower bound on the output.
    pub(crate) lower_bound: f32,
    /// Upper bound on the output.
    pub(crate) upper_bound: f32,
    /// Priority for queue ordering.
    ///
    /// Set by the BaB loop using `BetaCrownConfig::violation_priority()` to
    /// respect the `verify_upper_bound` flag. When `verify_upper_bound=true`,
    /// domains with higher lower bounds are prioritized (most likely to contain
    /// a counterexample). When false, domains with lower upper bounds are
    /// prioritized.
    ///
    /// Default: `lower_bound` (preserves legacy behavior for tests that don't
    /// set priority explicitly).
    ///
    /// Matches the `GraphBabDomain::priority` pattern (#2682).
    pub(crate) priority: f32,
    /// Pre-activation bounds for each layer (tightened by constraints).
    /// Uses Arc for cheap cloning during branch-and-bound splits - only modified
    /// layers need new allocations.
    pub(crate) layer_bounds: Vec<Arc<BoundedTensor>>,
    /// α state for α-CROWN (if used) - legacy field.
    pub(crate) alpha_state: Option<AlphaState>,
    /// Domain-specific α state for joint α-β optimization.
    pub(crate) domain_alpha_state: DomainAlphaState,
    /// β state for constrained neurons.
    pub(crate) beta_state: BetaState,
    /// Input bounds for this domain (used for input splitting).
    /// When None, the domain uses the original input bounds from verification.
    /// When Some, contains tightened input bounds from input space splitting.
    pub(crate) input_bounds: Option<Arc<BoundedTensor>>,
    /// Number of input splits applied to this domain.
    pub(crate) input_split_count: usize,
    /// Intermediate linear bounds from the last backward pass.
    /// Enables efficient bound transfer: child domains can reuse parent's
    /// bounds for layers after the split neuron instead of recomputing.
    /// Empty for root domain; populated after first full backward pass.
    pub(crate) intermediate_bounds: IntermediateLinearBounds,
}

impl BabDomain {
    /// Create root domain with no constraints.
    ///
    /// Returns `Err(NumericalInstability)` if either bound is NaN, preventing
    /// zombie domains from entering the BaB queue (#2982).
    pub fn root(
        layer_bounds: Vec<BoundedTensor>,
        lower_bound: f32,
        upper_bound: f32,
    ) -> Result<Self> {
        super::validate_domain_interval("BaB root domain", lower_bound, upper_bound)?;
        let layer_bounds: Vec<Arc<BoundedTensor>> =
            layer_bounds.into_iter().map(Arc::new).collect();
        Ok(Self {
            history: SplitHistory::new(),
            lower_bound,
            upper_bound,
            priority: lower_bound,
            layer_bounds,
            alpha_state: None,
            domain_alpha_state: DomainAlphaState::empty(),
            beta_state: BetaState::empty(),
            input_bounds: None,
            input_split_count: 0,
            intermediate_bounds: IntermediateLinearBounds::empty(),
        })
    }

    /// Create root domain with input bounds (for input splitting).
    ///
    /// Returns `Err(NumericalInstability)` if either bound is non-finite (#2982, #3125).
    pub fn root_with_input(
        layer_bounds: Vec<BoundedTensor>,
        lower_bound: f32,
        upper_bound: f32,
        input: &BoundedTensor,
    ) -> Result<Self> {
        super::validate_domain_interval("BaB root domain", lower_bound, upper_bound)?;
        let layer_bounds: Vec<Arc<BoundedTensor>> =
            layer_bounds.into_iter().map(Arc::new).collect();
        Ok(Self {
            history: SplitHistory::new(),
            lower_bound,
            upper_bound,
            priority: lower_bound,
            layer_bounds,
            alpha_state: None,
            domain_alpha_state: DomainAlphaState::empty(),
            beta_state: BetaState::empty(),
            input_bounds: Some(Arc::new(input.clone())),
            input_split_count: 0,
            intermediate_bounds: IntermediateLinearBounds::empty(),
        })
    }

    /// Create root domain with initialized α state for joint optimization.
    ///
    /// Returns `Err(NumericalInstability)` if either bound is non-finite (#2982, #3125).
    pub fn root_with_alpha(
        network: &Network,
        layer_bounds: Vec<BoundedTensor>,
        lower_bound: f32,
        upper_bound: f32,
    ) -> Result<Self> {
        super::validate_domain_interval("BaB root domain", lower_bound, upper_bound)?;
        let layer_bounds: Vec<Arc<BoundedTensor>> =
            layer_bounds.into_iter().map(Arc::new).collect();
        let history = SplitHistory::new();
        let domain_alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(network, &layer_bounds, &history);
        Ok(Self {
            history,
            lower_bound,
            upper_bound,
            priority: lower_bound,
            layer_bounds,
            alpha_state: None,
            domain_alpha_state,
            beta_state: BetaState::empty(),
            input_bounds: None,
            input_split_count: 0,
            intermediate_bounds: IntermediateLinearBounds::empty(),
        })
    }

    /// Create a child domain from a parent split.
    ///
    /// This is the validated constructor for all child domain creation within
    /// the BaB engine. Returns `Err(NumericalInstability)` if bounds or priority
    /// contain NaN, preventing zombie domains (#3125, #2982).
    ///
    /// Priority defaults to `lower_bound` (backward compat). The BaB loop
    /// overwrites priority via `set_priority()` before queue insertion (#2682).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn child(
        history: SplitHistory,
        lower_bound: f32,
        upper_bound: f32,
        layer_bounds: Vec<Arc<BoundedTensor>>,
        alpha_state: Option<AlphaState>,
        domain_alpha_state: DomainAlphaState,
        beta_state: BetaState,
        input_bounds: Option<Arc<BoundedTensor>>,
        input_split_count: usize,
        intermediate_bounds: IntermediateLinearBounds,
    ) -> Result<Self> {
        super::validate_domain_interval("BaB child domain", lower_bound, upper_bound)?;
        Ok(Self {
            history,
            lower_bound,
            upper_bound,
            priority: lower_bound,
            layer_bounds,
            alpha_state,
            domain_alpha_state,
            beta_state,
            input_bounds,
            input_split_count,
            intermediate_bounds,
        })
    }

    /// Depth of this domain (number of splits including input splits).
    pub fn depth(&self) -> usize {
        self.history.depth() + self.input_split_count
    }

    /// Effective input bounds for this domain.
    /// Returns domain-specific bounds if available, otherwise None (use original).
    pub fn input_bounds(&self) -> Option<&BoundedTensor> {
        self.input_bounds.as_ref().map(|arc| arc.as_ref())
    }

    // --- Accessor methods (#3125) ---

    /// Lower bound on the output.
    pub fn lower_bound(&self) -> f32 {
        self.lower_bound
    }

    /// Upper bound on the output.
    pub fn upper_bound(&self) -> f32 {
        self.upper_bound
    }

    /// Priority for queue ordering.
    pub fn priority(&self) -> f32 {
        self.priority
    }

    /// Split history for this domain.
    pub fn history(&self) -> &SplitHistory {
        &self.history
    }

    /// Pre-activation bounds for each layer.
    pub fn layer_bounds(&self) -> &[Arc<BoundedTensor>] {
        &self.layer_bounds
    }

    /// α state for α-CROWN (if used).
    pub fn alpha_state(&self) -> &Option<AlphaState> {
        &self.alpha_state
    }

    /// Domain-specific α state for joint α-β optimization.
    pub fn domain_alpha_state(&self) -> &DomainAlphaState {
        &self.domain_alpha_state
    }

    /// β state for constrained neurons.
    pub fn beta_state(&self) -> &BetaState {
        &self.beta_state
    }

    /// Input bounds arc reference.
    pub fn input_bounds_arc(&self) -> &Option<Arc<BoundedTensor>> {
        &self.input_bounds
    }

    /// Number of input splits applied to this domain.
    pub fn input_split_count(&self) -> usize {
        self.input_split_count
    }

    /// Intermediate linear bounds from the last backward pass.
    pub fn intermediate_bounds(&self) -> &IntermediateLinearBounds {
        &self.intermediate_bounds
    }
}

// For BinaryHeap: max-heap on priority field.
// Priority is set by the BaB loop via `BetaCrownConfig::violation_priority()`
// to respect the `verify_upper_bound` flag (#2682).
impl PartialEq for BabDomain {
    fn eq(&self, other: &Self) -> bool {
        super::cmp_domain_priority(self.priority, other.priority) == std::cmp::Ordering::Equal
    }
}

impl Eq for BabDomain {}

impl PartialOrd for BabDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BabDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority = pop first.
        // Priority is computed by BetaCrownConfig::violation_priority() which
        // respects verify_upper_bound (#2682).
        super::cmp_domain_priority(self.priority, other.priority)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;

    use super::*;

    #[test]
    fn test_bab_domain_root_rejects_nan_bounds() {
        let bounds = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid test tensor");

        // NaN lower_bound is rejected at construction (#2982)
        let result = BabDomain::root(vec![bounds.clone()], f32::NAN, 1.0);
        assert!(result.is_err(), "NaN lower_bound should be rejected");

        // NaN upper_bound is also rejected
        let result = BabDomain::root(vec![bounds.clone()], 0.0, f32::NAN);
        assert!(result.is_err(), "NaN upper_bound should be rejected");

        // Both NaN
        let result = BabDomain::root(vec![bounds.clone()], f32::NAN, f32::NAN);
        assert!(result.is_err(), "NaN bounds should be rejected");

        // Finite bounds are accepted
        let result = BabDomain::root(vec![bounds], 0.0, 1.0);
        assert!(result.is_ok(), "Finite bounds should be accepted");
    }

    #[test]
    fn sequential_domain_constructors_reject_inverted_objective_bounds() {
        let bounds = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .unwrap();
        assert!(BabDomain::root(vec![bounds.clone()], 1.0, 0.0).is_err());
        assert!(BabDomain::root_with_input(vec![bounds.clone()], 1.0, 0.0, &bounds,).is_err());
        assert!(BabDomain::root_with_alpha(&Network::new(), vec![bounds], 1.0, 0.0,).is_err());
        assert!(BabDomain::child(
            SplitHistory::new(),
            1.0,
            0.0,
            Vec::new(),
            None,
            DomainAlphaState::empty(),
            BetaState::empty(),
            None,
            0,
            IntermediateLinearBounds::empty(),
        )
        .is_err());
    }

    #[test]
    fn test_bab_domain_priority_overrides_lower_bound_ordering() {
        // When priority is set explicitly, it controls ordering regardless
        // of lower_bound values. This is critical for verify_upper_bound=true
        // mode where priority is computed from upper_bound (#2682).
        let bounds = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid test tensor");

        let mut domain_a = BabDomain::root(vec![bounds.clone()], 0.5, 2.0).unwrap();
        let mut domain_b = BabDomain::root(vec![bounds], 1.0, 3.0).unwrap();

        // Override priority: domain_a gets higher priority despite lower lower_bound
        domain_a.priority = 10.0;
        domain_b.priority = 5.0;

        let mut heap = BinaryHeap::new();
        heap.push(domain_b);
        heap.push(domain_a);

        let first = heap.pop().expect("heap should have domains");
        assert_eq!(first.priority, 10.0);
        assert_eq!(first.lower_bound, 0.5); // lower_bound != priority
    }
}
