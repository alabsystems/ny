// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Project α to the valid ReLU relaxation range `[0, 1]`.
///
/// Any value in `[0, 1]` is a valid lower-bound slope for unstable ReLU neurons.
/// We map NaN to `0.5` to prevent invalid optimizer values from propagating through
/// child-domain warm-start.
///
/// Reference:
/// - alpha-beta-CROWN `auto_LiRPA/operators/base.py:472-473` guards bound math with
///   non-NaN assertions instead of allowing NaN propagation.
#[inline]
pub(super) fn sanitize_alpha(value: f32) -> f32 {
    let projected = value.clamp(0.0, 1.0);
    if projected.is_nan() {
        0.5
    } else {
        projected
    }
}

/// Per-neuron optimization state for α parameters.
///
/// Compacts 6 separate f32 values that were previously stored in 6 separate
/// HashMaps into a single struct, reducing from ~392 bytes/neuron to ~40 bytes
/// (single HashMap entry: 16-byte key + 24-byte value + overhead).
#[derive(Debug, Clone, Copy)]
pub struct AlphaNeuronState {
    /// α value for this neuron's ReLU relaxation slope.
    /// Use [`set_alpha`](Self::set_alpha) for writes to enforce `[0, 1]` + NaN sanitization.
    pub(crate) alpha: f32,
    /// Gradient accumulator. Use [`grad`](Self::grad) for reads.
    pub(crate) grad: f32,
    /// Velocity (momentum) for gradient updates.
    pub(crate) velocity: f32,
    /// First moment estimate (m) for Adam optimizer.
    pub(crate) adam_m: f32,
    /// Second moment estimate (v) for Adam optimizer.
    pub(crate) adam_v: f32,
    /// Maximum second moment estimate (v_max) for AMSGrad variant.
    pub(crate) adam_v_max: f32,
}

impl AlphaNeuronState {
    /// Create a new neuron state with the given alpha and zeroed optimizer state.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: sanitize_alpha(alpha),
            grad: 0.0,
            velocity: 0.0,
            adam_m: 0.0,
            adam_v: 0.0,
            adam_v_max: 0.0,
        }
    }

    /// The sanitized α value, clamped to `[0, 1]` with NaN → 0.5.
    pub fn alpha(&self) -> f32 {
        sanitize_alpha(self.alpha)
    }

    /// Set α value, clamping to `[0, 1]` and sanitizing NaN → 0.5.
    ///
    /// All production writes to `alpha` must go through this method to enforce
    /// the invariant that α ∈ \[0, 1\] (valid ReLU relaxation slope).
    pub fn set_alpha(&mut self, value: f32) {
        self.alpha = sanitize_alpha(value);
    }

    /// The accumulated gradient.
    pub fn grad(&self) -> f32 {
        self.grad
    }

    /// Velocity (momentum).
    pub fn velocity(&self) -> f32 {
        self.velocity
    }

    /// Adam first moment estimate.
    pub fn adam_m(&self) -> f32 {
        self.adam_m
    }

    /// Adam second moment estimate.
    pub fn adam_v(&self) -> f32 {
        self.adam_v
    }

    /// AMSGrad maximum second moment estimate.
    pub fn adam_v_max(&self) -> f32 {
        self.adam_v_max
    }

    /// Reset all optimizer state (grad, velocity, moments) to zero.
    pub fn reset_optimizer(&mut self) {
        self.grad = 0.0;
        self.velocity = 0.0;
        self.adam_m = 0.0;
        self.adam_v = 0.0;
        self.adam_v_max = 0.0;
    }
}
