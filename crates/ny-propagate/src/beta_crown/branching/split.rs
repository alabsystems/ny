// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::graph_constraints::GraphNeuronConstraint;
use super::sequential::NeuronConstraint;
use ny_core::{NyError, Result};
use std::fmt;

/// Reference to a layer (supports both sequential and graph networks).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayerRef {
    /// Layer index for sequential networks.
    Index(usize),
    /// Node name for graph networks (DAG structure).
    Name(String),
}

impl fmt::Display for LayerRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerRef::Index(i) => write!(f, "layer_{}", i),
            LayerRef::Name(s) => write!(f, "{s}"),
        }
    }
}

impl LayerRef {
    /// Get the name if this is a Name variant, None otherwise.
    pub fn as_name(&self) -> Option<&str> {
        match self {
            LayerRef::Name(s) => Some(s),
            LayerRef::Index(_) => None,
        }
    }

    /// Get the index if this is an Index variant, None otherwise.
    pub fn as_index(&self) -> Option<usize> {
        match self {
            LayerRef::Index(i) => Some(*i),
            LayerRef::Name(_) => None,
        }
    }
}

/// Constraint on a single neuron for branching.
///
/// Supports both ReLU (binary at 0) and general nonlinearities (arbitrary points).
/// Unlike `NeuronConstraint`/`GraphNeuronConstraint` which use `is_active: bool`,
/// this representation uses explicit bounds to support non-zero branching points.
///
/// # Examples
///
/// ```
/// use ny_propagate::beta_crown::{LayerRef, NeuronSplit};
///
/// // ReLU split: active branch (x >= 0)
/// let relu_active = NeuronSplit::relu_active(LayerRef::Index(2), 5);
///
/// // ReLU split: inactive branch (x <= 0)
/// let relu_inactive = NeuronSplit::relu_inactive(LayerRef::Index(2), 5);
///
/// // GeLU split at midpoint -0.5: upper branch (x >= -0.5)
/// let gelu_upper = NeuronSplit::at_point(LayerRef::Name("gelu_1".into()), 3, -0.5, true)
///     .expect("finite point");
///
/// // GeLU split at midpoint -0.5: lower branch (x <= -0.5)
/// let gelu_lower = NeuronSplit::at_point(LayerRef::Name("gelu_1".into()), 3, -0.5, false)
///     .expect("finite point");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronSplit {
    /// Layer reference (index or node name).
    pub(crate) layer: LayerRef,
    /// Neuron index within the layer.
    pub(crate) neuron_idx: usize,
    /// Lower bound of this branch (None = use domain's original lower bound).
    pub(crate) lower_bound: Option<f32>,
    /// Upper bound of this branch (None = use domain's original upper bound).
    pub(crate) upper_bound: Option<f32>,
    /// Influence score for this split (used by BICCOS constraint strengthening).
    /// Larger values indicate more impactful splits.
    pub(crate) score: f32,
    /// For binary operations (e.g., BilinearCrown), which input to split.
    /// `None` or `Some(0)` = first input (default, backward compatible).
    /// `Some(1)` = second input (e.g., K for BilinearCrown's Q @ K^T).
    pub(crate) input_index: Option<usize>,
    /// GenBaB norm branching (#norm-genbab): when `Some((lo, hi))`, this split
    /// clamps a `Layer::RmsNorm` node's internal `inv_rms` to `[lo, hi]` for the
    /// child (not the node's value interval). `with_general_split` turns it into
    /// a [`NormInvRmsConstraint`]. Mutually exclusive with `lower_bound`/
    /// `upper_bound` value clamps. The `usize` is the normalization group this
    /// clamp targets (one group per split for soundness; see `InvRmsOverride`).
    pub(crate) norm_inv_rms_window: Option<(usize, f32, f32)>,
}

impl NeuronSplit {
    /// Create a new NeuronSplit with validated bounds and score.
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if:
    /// - `score` is not finite
    /// - Any bound value is not finite
    /// - Both bounds are `Some` and `lower_bound > upper_bound`
    pub fn new(
        layer: LayerRef,
        neuron_idx: usize,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        score: f32,
    ) -> Result<Self> {
        if !score.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "NeuronSplit score must be finite, got {score} for {layer:?}[{neuron_idx}]"
            )));
        }
        if let Some(lb) = lower_bound {
            if !lb.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "NeuronSplit lower_bound must be finite, got {lb} for {layer:?}[{neuron_idx}]"
                )));
            }
        }
        if let Some(ub) = upper_bound {
            if !ub.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "NeuronSplit upper_bound must be finite, got {ub} for {layer:?}[{neuron_idx}]"
                )));
            }
        }
        if let (Some(lb), Some(ub)) = (lower_bound, upper_bound) {
            if lb > ub {
                return Err(NyError::NumericalInstability(format!(
                    "NeuronSplit lower_bound ({lb}) > upper_bound ({ub}) \
                     for {layer:?}[{neuron_idx}]"
                )));
            }
        }
        Ok(Self {
            layer,
            neuron_idx,
            lower_bound,
            upper_bound,
            score,
            input_index: None,
            norm_inv_rms_window: None,
        })
    }

    /// Returns the layer reference.
    #[inline]
    pub fn layer(&self) -> &LayerRef {
        &self.layer
    }

    /// Returns the neuron index within the layer.
    #[inline]
    pub fn neuron_idx(&self) -> usize {
        self.neuron_idx
    }

    /// Returns the lower bound of this branch (None = use domain's original lower bound).
    #[inline]
    pub fn lower_bound(&self) -> Option<f32> {
        self.lower_bound
    }

    /// Returns the upper bound of this branch (None = use domain's original upper bound).
    #[inline]
    pub fn upper_bound(&self) -> Option<f32> {
        self.upper_bound
    }

    /// Returns the influence score for this split.
    #[inline]
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Returns the input index for binary operations.
    /// `None` or `Some(0)` means first input (default).
    /// `Some(1)` means second input (e.g., K for BilinearCrown).
    #[inline]
    pub fn input_index(&self) -> Option<usize> {
        self.input_index
    }

    /// Set the input index for binary operation splits.
    #[inline]
    pub fn with_input_index(mut self, input_index: usize) -> Self {
        self.input_index = Some(input_index);
        self
    }

    /// Create a GenBaB norm-branching split clamping a `Layer::RmsNorm` node's
    /// internal `inv_rms` to `[lo, hi]` for the child (#norm-genbab).
    ///
    /// `neuron_idx` is unused for norm splits (the split is on the node-level
    /// `inv_rms` scalar, shared across normalization groups). `with_general_split`
    /// converts this into a [`NormInvRmsConstraint`].
    pub fn norm_inv_rms(
        layer: LayerRef,
        group: usize,
        lo: f32,
        hi: f32,
        score: f32,
    ) -> Result<Self> {
        if !lo.is_finite() || !hi.is_finite() || !score.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "NeuronSplit::norm_inv_rms requires finite values, got \
                 lo={lo}, hi={hi}, score={score} for {layer:?}"
            )));
        }
        if lo > hi {
            return Err(NyError::NumericalInstability(format!(
                "NeuronSplit::norm_inv_rms inverted window [{lo}, {hi}] for {layer:?}"
            )));
        }
        Ok(Self {
            layer,
            neuron_idx: 0,
            lower_bound: None,
            upper_bound: None,
            score,
            input_index: None,
            norm_inv_rms_window: Some((group, lo, hi)),
        })
    }

    /// Returns the norm `inv_rms` window `(group, lo, hi)` if this is a
    /// norm-branching split.
    #[inline]
    pub fn norm_inv_rms_window(&self) -> Option<(usize, f32, f32)> {
        self.norm_inv_rms_window
    }

    /// Create a ReLU split for the active branch (x >= 0).
    pub fn relu_active(layer: LayerRef, neuron_idx: usize) -> Self {
        Self {
            layer,
            neuron_idx,
            lower_bound: Some(0.0),
            upper_bound: None,
            score: 0.0,
            input_index: None,
            norm_inv_rms_window: None,
        }
    }

    /// Create a ReLU split for the inactive branch (x <= 0).
    pub fn relu_inactive(layer: LayerRef, neuron_idx: usize) -> Self {
        Self {
            layer,
            neuron_idx,
            lower_bound: None,
            upper_bound: Some(0.0),
            score: 0.0,
            input_index: None,
            norm_inv_rms_window: None,
        }
    }

    /// Create a split at an arbitrary branching point.
    ///
    /// # Arguments
    /// * `layer` - Layer reference
    /// * `neuron_idx` - Neuron index within the layer
    /// * `point` - Branching point value (must be finite)
    /// * `upper` - If true, creates upper branch (x >= point); if false, lower branch (x <= point)
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if `point` is NaN or infinite.
    pub fn at_point(layer: LayerRef, neuron_idx: usize, point: f32, upper: bool) -> Result<Self> {
        if !point.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "NeuronSplit point must be finite, got {point} for {layer:?}[{neuron_idx}]"
            )));
        }
        if upper {
            Ok(Self {
                layer,
                neuron_idx,
                lower_bound: Some(point),
                upper_bound: None,
                score: 0.0,
                input_index: None,
                norm_inv_rms_window: None,
            })
        } else {
            Ok(Self {
                layer,
                neuron_idx,
                lower_bound: None,
                upper_bound: Some(point),
                score: 0.0,
                input_index: None,
                norm_inv_rms_window: None,
            })
        }
    }

    /// Create from an existing NeuronConstraint (for backward compatibility).
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if the constraint's score is not finite.
    pub fn from_constraint(constraint: &NeuronConstraint) -> Result<Self> {
        NeuronSplit::new(
            LayerRef::Index(constraint.layer_idx),
            constraint.neuron_idx,
            if constraint.is_active {
                Some(0.0)
            } else {
                None
            },
            if constraint.is_active {
                None
            } else {
                Some(0.0)
            },
            constraint.score,
        )
    }

    /// Create from a GraphNeuronConstraint (for backward compatibility).
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if the constraint's score is not finite.
    pub fn from_graph_constraint(constraint: &GraphNeuronConstraint) -> Result<Self> {
        NeuronSplit::new(
            LayerRef::Name(constraint.node_name.clone()),
            constraint.neuron_idx,
            if constraint.is_active {
                Some(0.0)
            } else {
                None
            },
            if constraint.is_active {
                None
            } else {
                Some(0.0)
            },
            constraint.score,
        )
    }

    /// Check if this is a ReLU-style split (branching at 0).
    pub fn is_relu_split(&self) -> bool {
        matches!(
            (self.lower_bound, self.upper_bound),
            (Some(0.0), None) | (None, Some(0.0))
        )
    }

    /// Get the branching point if this is a single-point split.
    pub fn branching_point(&self) -> Option<f32> {
        match (self.lower_bound, self.upper_bound) {
            (Some(p), None) => Some(p),
            (None, Some(p)) => Some(p),
            _ => None,
        }
    }
}
