// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};

/// A constraint on a single ReLU neuron in a GraphNetwork.
///
/// Unlike NeuronConstraint which uses layer indices, this uses node names
/// to identify ReLU nodes in the DAG structure.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphNeuronConstraint {
    /// Name of the ReLU node in the graph.
    pub(crate) node_name: String,
    /// Neuron index within the ReLU node's output.
    pub(crate) neuron_idx: usize,
    /// True if neuron is constrained to be active (x >= 0), false if inactive (x <= 0).
    pub(crate) is_active: bool,
    /// Influence score for this split (used by BICCOS constraint strengthening).
    /// Larger values indicate more impactful splits.
    pub(crate) score: f32,
}

impl GraphNeuronConstraint {
    /// Create a new graph neuron constraint with validated score.
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if `score` is not finite.
    pub fn new(node_name: String, neuron_idx: usize, is_active: bool, score: f32) -> Result<Self> {
        if !score.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GraphNeuronConstraint score must be finite, got {score} \
                 for node {node_name}[{neuron_idx}]"
            )));
        }
        Ok(Self {
            node_name,
            neuron_idx,
            is_active,
            score,
        })
    }

    /// Returns the node name.
    #[inline]
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Returns the neuron index within the node's output.
    #[inline]
    pub fn neuron_idx(&self) -> usize {
        self.neuron_idx
    }

    /// Returns whether the neuron is constrained to be active (x >= 0).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Returns the influence score for this split.
    #[inline]
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// A constraint on a single neuron for GenBaB (general nonlinearity branching).
///
/// Unlike `GraphNeuronConstraint` which only supports ReLU (split at 0),
/// this supports arbitrary split points for general nonlinearities (GeLU, Sigmoid, Tanh, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct GenBabConstraint {
    /// Name of the nonlinear node in the graph.
    pub(crate) node_name: String,
    /// Neuron index within the node's output.
    pub(crate) neuron_idx: usize,
    /// Split point value (branching point).
    pub(crate) split_point: f32,
    /// True if this is the upper branch (x >= split_point), false if lower (x <= split_point).
    pub(crate) is_upper_branch: bool,
    /// Influence score for this split (used by BICCOS constraint strengthening).
    /// Larger values indicate more impactful splits.
    pub(crate) score: f32,
    /// For binary McCormick ops (`MulBinary z = x·y`, `BilinearCrown z = Q@Kᵀ`),
    /// which of the node's inputs this split constrains: `Some(0)` = first input,
    /// `Some(1)` = second input. `None` (the default) means the first input — the
    /// only input for unary nonlinearities (GeLU, Sigmoid, …).
    ///
    /// SOUNDNESS: this selects WHICH pre-activation node the `split_point` clamp is
    /// applied to in the forward pass. Misrouting it (the prior hard-coded
    /// `inputs.first()` behavior) clamps the wrong input — either a hard index
    /// error (different-length inputs) or, worse, silently excluding reachable
    /// values of the wrong neuron. The split target must match the input whose
    /// interval the branching decision actually subdivided (#mul-genbab).
    pub(crate) input_index: Option<usize>,
}

impl GenBabConstraint {
    /// Create a new GenBaB constraint with validated split point.
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if `split_point` is NaN or infinite.
    /// NaN split points arise when pre-activation bounds are NaN from upstream
    /// propagation failure; allowing them through causes a panic in
    /// `GraphBetaEntry::new()` during beta state construction.
    pub fn new(
        node_name: String,
        neuron_idx: usize,
        split_point: f32,
        is_upper_branch: bool,
        score: f32,
    ) -> Result<Self> {
        if !split_point.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GenBabConstraint split_point must be finite, got {split_point} \
                 for node {node_name}[{neuron_idx}]"
            )));
        }
        if !score.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GenBabConstraint score must be finite, got {score} \
                 for node {node_name}[{neuron_idx}]"
            )));
        }
        Ok(Self {
            node_name,
            neuron_idx,
            split_point,
            is_upper_branch,
            score,
            input_index: None,
        })
    }

    /// Set which input of a binary McCormick op this split constrains.
    ///
    /// `Some(1)` selects the second input (e.g. `up` in `MulBinary(silu, up)` or
    /// `K` in `BilinearCrown(Q, K)`); `Some(0)`/`None` selects the first. For
    /// unary nonlinearities leave this unset.
    #[inline]
    pub fn with_input_index(mut self, input_index: usize) -> Self {
        self.input_index = Some(input_index);
        self
    }

    /// Returns the input index this constraint targets for binary ops.
    /// `None` or `Some(0)` = first input (the only input for unary activations).
    #[inline]
    pub fn input_index(&self) -> Option<usize> {
        self.input_index
    }

    /// Convert to a `GraphNeuronConstraint` (lossy - loses split point info).
    ///
    /// This approximates by treating upper branch (x >= split_point) as active
    /// and lower branch (x <= split_point) as inactive, regardless of split point.
    /// Used for backward compatibility with ReLU-only beta state initialization.
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if the score is not finite.
    pub fn to_graph_neuron_constraint(&self) -> Result<GraphNeuronConstraint> {
        GraphNeuronConstraint::new(
            self.node_name.clone(),
            self.neuron_idx,
            self.is_upper_branch,
            self.score,
        )
    }

    /// Create from a `GraphNeuronConstraint` (assumes split at 0).
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if the constraint's score is not finite.
    pub fn from_graph_neuron_constraint(constraint: &GraphNeuronConstraint) -> Result<Self> {
        GenBabConstraint::new(
            constraint.node_name.clone(),
            constraint.neuron_idx,
            0.0,
            constraint.is_active,
            constraint.score,
        )
    }

    /// Returns the node name for this constraint.
    #[inline]
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Returns the neuron index within the node's output.
    #[inline]
    pub fn neuron_idx(&self) -> usize {
        self.neuron_idx
    }

    /// Returns the split point value.
    #[inline]
    pub fn split_point(&self) -> f32 {
        self.split_point
    }

    /// Returns whether this is the upper branch (x >= split_point).
    #[inline]
    pub fn is_upper_branch(&self) -> bool {
        self.is_upper_branch
    }

    /// Returns the influence score for this split.
    #[inline]
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// A GenBaB norm-branching constraint: a clamp on a `Layer::RmsNorm` /
/// `Layer::LayerNorm` node's INTERNAL `inv_rms = 1/sqrt(mean(x²)+eps)` scalar
/// (#norm-genbab).
///
/// Unlike [`GenBabConstraint`], whose `split_point` clamps a graph node's
/// observable value interval, the `inv_rms` quantity is NOT a graph node — it
/// lives inside the decomposed normalization CROWN backward. So this constraint
/// is consumed only by the RmsNorm dispatch arm, which intersects the node's
/// IBP-derived `inv_rms` interval with `[lo, hi]` for the requesting child
/// subdomain (see `InvRmsOverride`).
///
/// SOUNDNESS: the two sibling children of a norm split carry the lower and upper
/// halves of the parent `inv_rms` range, which union-cover the parent range and
/// hence the full input box (every `x` has `inv_rms(x)` in the parent range).
/// Each child's narrowed relaxation is a sound over-approximation on its own
/// input subregion `{x : inv_rms(x) ∈ [lo,hi]}`, so the combined BaB verdict is
/// sound.
#[derive(Debug, Clone, PartialEq)]
pub struct NormInvRmsConstraint {
    /// Name of the normalization node whose `inv_rms` this clamps.
    pub(crate) node_name: String,
    /// Normalization group (batch row) this clamp targets. Splitting ONE group
    /// at a time is required for soundness: a window shared across all groups
    /// would create a join gap between the sibling children (see
    /// `InvRmsOverride`).
    pub(crate) group_index: usize,
    /// Lower clamp on `inv_rms` for this child.
    pub(crate) inv_rms_lo: f32,
    /// Upper clamp on `inv_rms` for this child.
    pub(crate) inv_rms_hi: f32,
    /// Influence score (for BaB bookkeeping parity with other constraints).
    pub(crate) score: f32,
}

impl NormInvRmsConstraint {
    /// Create a norm `inv_rms` constraint, rejecting non-finite / inverted
    /// windows (these arise from corrupted upstream bounds and would otherwise
    /// poison the child relaxation).
    pub fn new(
        node_name: String,
        group_index: usize,
        inv_rms_lo: f32,
        inv_rms_hi: f32,
        score: f32,
    ) -> Result<Self> {
        if !inv_rms_lo.is_finite() || !inv_rms_hi.is_finite() || !score.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "NormInvRmsConstraint requires finite values, got \
                 lo={inv_rms_lo}, hi={inv_rms_hi}, score={score} for {node_name}"
            )));
        }
        if inv_rms_lo > inv_rms_hi {
            return Err(NyError::NumericalInstability(format!(
                "NormInvRmsConstraint inverted window [{inv_rms_lo}, {inv_rms_hi}] for {node_name}"
            )));
        }
        Ok(Self {
            node_name,
            group_index,
            inv_rms_lo,
            inv_rms_hi,
            score,
        })
    }

    /// Node name this constraint targets.
    #[inline]
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Normalization group (batch row) this clamp targets.
    #[inline]
    pub fn group_index(&self) -> usize {
        self.group_index
    }

    /// `inv_rms` window `[lo, hi]` for the constrained child.
    #[inline]
    pub fn window(&self) -> (f32, f32) {
        (self.inv_rms_lo, self.inv_rms_hi)
    }
}

/// A constraint in a `GraphSplitHistory`, either ReLU or GenBaB.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphConstraint {
    /// ReLU constraint (binary split at 0).
    Relu(GraphNeuronConstraint),
    /// GenBaB constraint (arbitrary split point).
    GenBab(GenBabConstraint),
}

impl GraphConstraint {
    /// Get the node name for this constraint.
    pub fn node_name(&self) -> &str {
        match self {
            GraphConstraint::Relu(c) => &c.node_name,
            GraphConstraint::GenBab(c) => &c.node_name,
        }
    }

    /// Get the neuron index for this constraint.
    pub fn neuron_idx(&self) -> usize {
        match self {
            GraphConstraint::Relu(c) => c.neuron_idx,
            GraphConstraint::GenBab(c) => c.neuron_idx,
        }
    }

    /// Get the sign for beta optimization (+1 for active/upper, -1 for inactive/lower).
    pub fn beta_sign(&self) -> f32 {
        match self {
            GraphConstraint::Relu(c) => {
                if c.is_active {
                    1.0
                } else {
                    -1.0
                }
            }
            GraphConstraint::GenBab(c) => {
                if c.is_upper_branch {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    /// Check if this is an upper/active branch constraint.
    pub fn is_upper_branch(&self) -> bool {
        match self {
            GraphConstraint::Relu(c) => c.is_active,
            GraphConstraint::GenBab(c) => c.is_upper_branch,
        }
    }

    /// Get the split point (0.0 for ReLU, arbitrary for GenBaB).
    pub fn split_point(&self) -> f32 {
        match self {
            GraphConstraint::Relu(_) => 0.0,
            GraphConstraint::GenBab(c) => c.split_point,
        }
    }

    /// Convert to a `GraphNeuronConstraint` (lossy for GenBaB).
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if score validation fails.
    pub fn to_graph_neuron_constraint(&self) -> Result<GraphNeuronConstraint> {
        match self {
            GraphConstraint::Relu(c) => Ok(c.clone()),
            GraphConstraint::GenBab(c) => c.to_graph_neuron_constraint(),
        }
    }
}
