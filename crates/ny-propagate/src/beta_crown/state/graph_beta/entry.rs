// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};

/// Graph network beta entry. Like BetaEntry but uses node_name (String) instead of layer_idx.
/// Optimizer fields are `pub(crate)` — use `set_value()`/`get_value()` for external access.
#[derive(Debug, Clone)]
pub struct GraphBetaEntry {
    /// Node name of the constrained ReLU neuron.
    pub(crate) node_name: String,
    /// Neuron index within the ReLU node's output.
    pub(crate) neuron_idx: usize,
    /// Split point for this constraint (0.0 for ReLU).
    pub(crate) split_point: f32,
    /// Current beta value (Lagrangian multiplier, must be >= 0). Use `set_value()` for writes.
    pub(crate) value: f32,
    /// Sign of the constraint: +1 for active (x >= 0), -1 for inactive (x <= 0).
    pub(crate) sign: f32,
    /// Accumulated gradient for this iteration.
    pub(crate) grad: f32,
    /// First moment estimate (mean of gradients) for Adam optimizer.
    pub(crate) m: f32,
    /// Second moment estimate (uncentered variance) for Adam optimizer.
    pub(crate) v: f32,
    /// Maximum second moment estimate for AMSGrad variant.
    pub(crate) v_max: f32,
}

impl GraphBetaEntry {
    /// Create a new GraphBetaEntry with validated sign and split_point.
    ///
    /// Sign must be +1.0 or -1.0. Split point must be finite.
    /// Value must be >= 0 and finite; invalid values are clamped to 0.0.
    /// Optimizer state (grad, m, v, v_max) is initialized to zero.
    pub fn new(
        node_name: String,
        neuron_idx: usize,
        split_point: f32,
        value: f32,
        sign: f32,
    ) -> Result<Self> {
        if sign != 1.0 && sign != -1.0 {
            return Err(NyError::InvalidSpec(format!(
                "GraphBetaEntry sign must be +1.0 or -1.0, got {sign}"
            )));
        }
        if !split_point.is_finite() {
            return Err(NyError::InvalidSpec(format!(
                "GraphBetaEntry split_point must be finite, got {split_point}"
            )));
        }
        let value = if value.is_finite() && value >= 0.0 {
            value
        } else {
            0.0
        };
        Ok(Self {
            node_name,
            neuron_idx,
            split_point,
            value,
            sign,
            grad: 0.0,
            m: 0.0,
            v: 0.0,
            v_max: 0.0,
        })
    }

    /// Get the node name of the constrained neuron.
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Get the neuron index within the node.
    pub fn neuron_idx(&self) -> usize {
        self.neuron_idx
    }

    /// Get the split point for this constraint.
    pub fn split_point(&self) -> f32 {
        self.split_point
    }

    /// Get the constraint sign (+1.0 for active, -1.0 for inactive).
    pub fn sign(&self) -> f32 {
        self.sign
    }

    /// The current β value.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Set β value. NaN/Inf/negative → reset to 0. Matches NaN guard in gradient_step.
    pub fn set_value(&mut self, value: f32) {
        self.value = if value.is_finite() && value >= 0.0 {
            value
        } else {
            0.0
        };
    }

    /// The accumulated gradient.
    pub fn grad(&self) -> f32 {
        self.grad
    }

    /// Get the signed beta contribution (value * sign).
    pub fn signed_value(&self) -> f32 {
        self.value * self.sign
    }

    /// Reset β value and all optimizer state to zero.
    pub fn reset_optimizer(&mut self) {
        self.value = 0.0;
        self.grad = 0.0;
        self.m = 0.0;
        self.v = 0.0;
        self.v_max = 0.0;
    }
}
