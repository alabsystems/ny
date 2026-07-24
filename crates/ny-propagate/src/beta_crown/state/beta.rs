// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential network β state for constrained CROWN propagation.

use std::collections::HashMap;

use ny_core::{nan_propagating_max, NyError, Result};

use super::super::branching::SplitHistory;
use super::super::config::{radam_rectification_factor, AdaptiveOptConfig, LookaheadConfig};

/// A single beta parameter entry for sparse beta representation.
///
/// Optimizer fields (`value`, `grad`, `m`, `v`, `v_max`) are `pub(crate)` to prevent
/// external direct writes that bypass NaN validation. Use [`set_value`](Self::set_value)
/// for writes and [`get_value`](Self::get_value) for reads.
#[derive(Debug, Clone, Copy)]
pub struct BetaEntry {
    /// Layer index of the constrained neuron.
    pub(crate) layer_idx: usize,
    /// Neuron index within the layer.
    pub(crate) neuron_idx: usize,
    /// Current beta value (Lagrangian multiplier, must be >= 0).
    /// Use [`set_value`](Self::set_value) for writes.
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
    /// Tracks max(v) over all past iterations for stable convergence.
    pub(crate) v_max: f32,
}

impl BetaEntry {
    /// Create a new BetaEntry with validated sign.
    ///
    /// Sign must be +1.0 or -1.0. Invalid sign values are rejected with an error.
    /// Value must be >= 0 and finite; invalid values are clamped to 0.0.
    /// Optimizer state (grad, m, v, v_max) is initialized to zero.
    pub fn new(layer_idx: usize, neuron_idx: usize, value: f32, sign: f32) -> Result<Self> {
        if sign != 1.0 && sign != -1.0 {
            return Err(NyError::InvalidSpec(format!(
                "BetaEntry sign must be +1.0 or -1.0, got {sign}"
            )));
        }
        let value = if value.is_finite() && value >= 0.0 {
            value
        } else {
            tracing::warn!(
                "NaN/Inf/negative in BetaEntry::new, clamping to 0.0: layer={layer_idx}, neuron={neuron_idx}, value={value}"
            );
            0.0
        };
        Ok(Self {
            layer_idx,
            neuron_idx,
            value,
            sign,
            grad: 0.0,
            m: 0.0,
            v: 0.0,
            v_max: 0.0,
        })
    }

    /// Get the layer index of the constrained neuron.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get the neuron index within the layer.
    pub fn neuron_idx(&self) -> usize {
        self.neuron_idx
    }

    /// Get the constraint sign (+1.0 for active, -1.0 for inactive).
    pub fn sign(&self) -> f32 {
        self.sign
    }

    /// The current β value.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Set β value. Validates value >= 0 and finite; NaN/Inf/negative → reset to 0.
    ///
    /// Mirrors the NaN guard pattern in `gradient_step` / `gradient_step_adam`:
    /// invalid values are silently clamped to 0.0 rather than returning an error,
    /// because the optimizer loop calls this at high frequency and NaN recovery
    /// must be automatic (not error-propagated).
    pub fn set_value(&mut self, value: f32) {
        if !value.is_finite() || value < 0.0 {
            tracing::warn!(
                "NaN/Inf/negative in BetaEntry::set_value, clamping to 0.0: layer={}, neuron={}, value={value}",
                self.layer_idx, self.neuron_idx
            );
            self.value = 0.0;
        } else {
            self.value = value;
        }
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

/// β parameters for constrained CROWN propagation.
///
/// β values are Lagrangian multipliers in the dual formulation of split constraints.
/// The Lagrangian augmented bound is: lb = c^T * A * x + b + sum_i(β_i * sign_i * x_i)
///
/// During optimization:
/// 1. Forward pass: compute bounds with current β values
/// 2. Compute gradient: (sub)gradient of the current lower bound w.r.t. β (depends on the
///    current piecewise-linear relaxation choices and concretization point)
/// 3. Update: β = max(0, β + lr * grad)  (projected gradient ascent)
#[derive(Debug, Clone, Default)]
pub struct BetaState {
    /// Sparse beta entries for constrained neurons.
    pub(crate) entries: Vec<BetaEntry>,
    /// Slow weights for Lookahead optimizer.
    /// When Some, contains β values that are a moving average of the fast weights.
    /// Maps 1:1 with entries (slow_weights\[i\] corresponds to entries\[i\].value).
    pub(crate) slow_weights: Option<Vec<f32>>,
    /// Fast `(layer_idx, neuron_idx)` lookup for hot β-state reads.
    pub(crate) entry_index: HashMap<(usize, usize), usize>,
    /// Fast per-layer fanout for `entries_for_layer()` hot paths.
    pub(crate) layer_index: HashMap<usize, Vec<usize>>,
    /// Number of entries covered by the lookup indexes.
    pub(crate) indexed_entries: usize,
}

use super::beta_lookup::BetaEntriesForLayer;

impl BetaState {
    /// Create β state from split history.
    ///
    /// Sign is derived from constraint: +1 for active, -1 for inactive.
    /// Initial β value is 0.0 (standard initialization).
    pub fn from_history(history: &SplitHistory) -> Result<Self> {
        let entries = history
            .constraints
            .iter()
            .map(|c| {
                let sign = if c.is_active { 1.0 } else { -1.0 };
                BetaEntry::new(c.layer_idx, c.neuron_idx, 0.0, sign)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_parts(entries, None))
    }

    fn from_parts(entries: Vec<BetaEntry>, slow_weights: Option<Vec<f32>>) -> Self {
        let mut state = Self {
            entries,
            slow_weights,
            ..Self::default()
        };
        state.rebuild_lookup_indexes();
        state
    }

    fn rebuild_lookup_indexes(&mut self) {
        self.entry_index.clear();
        self.layer_index.clear();
        for (idx, entry) in self.entries.iter().enumerate() {
            self.entry_index
                .entry((entry.layer_idx, entry.neuron_idx))
                .or_insert(idx);
            self.layer_index
                .entry(entry.layer_idx)
                .or_default()
                .push(idx);
        }
        self.indexed_entries = self.entries.len();
    }

    /// Create empty beta state.
    pub fn empty() -> Self {
        Self::default()
    }

    /// β entry for a neuron, if constrained.
    pub fn entry(&self, layer_idx: usize, neuron_idx: usize) -> Option<&BetaEntry> {
        if self.indexed_entries == self.entries.len() {
            return self
                .entry_index
                .get(&(layer_idx, neuron_idx))
                .and_then(|&idx| self.entries.get(idx));
        }
        self.entries
            .iter()
            .find(|e| e.layer_idx == layer_idx && e.neuron_idx == neuron_idx)
    }

    /// Mutable β entry for a neuron, if constrained.
    pub fn entry_mut(&mut self, layer_idx: usize, neuron_idx: usize) -> Option<&mut BetaEntry> {
        if self.indexed_entries != self.entries.len() {
            self.rebuild_lookup_indexes();
        }
        if let Some(idx) = self.entry_index.get(&(layer_idx, neuron_idx)).copied() {
            return self.entries.get_mut(idx);
        }
        self.entries
            .iter_mut()
            .find(|e| e.layer_idx == layer_idx && e.neuron_idx == neuron_idx)
    }

    /// β value for a neuron, if constrained.
    pub fn beta(&self, layer_idx: usize, neuron_idx: usize) -> Option<f32> {
        self.entry(layer_idx, neuron_idx).map(|e| e.value())
    }

    /// Set β value for a neuron.
    pub fn set_beta(&mut self, layer_idx: usize, neuron_idx: usize, value: f32) {
        if let Some(entry) = self.entry_mut(layer_idx, neuron_idx) {
            entry.set_value(value);
        }
    }

    /// Reset all gradients to zero.
    pub fn zero_grad(&mut self) {
        for entry in &mut self.entries {
            entry.grad = 0.0;
        }
    }

    /// Accumulate gradient for a specific neuron.
    ///
    /// NaN guard (#3112): skip non-finite gradients to prevent NaN from entering the
    /// optimizer state. Without this, a single NaN gradient poisons m/v accumulators
    /// and wastes an Adam step before the post-step guard resets everything.
    pub fn accumulate_grad(&mut self, layer_idx: usize, neuron_idx: usize, grad: f32) {
        if !grad.is_finite() {
            tracing::warn!(
                "NaN/Inf gradient in BetaState::accumulate_grad, skipping: layer={layer_idx}, neuron={neuron_idx}, grad={grad}"
            );
            return;
        }
        if let Some(entry) = self.entry_mut(layer_idx, neuron_idx) {
            entry.grad += grad;
        }
    }

    /// Perform projected gradient ascent step.
    /// β = max(0, β + lr * grad)
    ///
    /// Returns the maximum gradient magnitude (for convergence check).
    pub fn gradient_step(&mut self, lr: f32) -> f32 {
        let mut max_grad = 0.0f32;
        for entry in &mut self.entries {
            // NaN-aware convergence tracking: f32::max silently drops NaN (#2939).
            max_grad = nan_propagating_max(max_grad, entry.grad.abs());
            // Gradient ascent (we want to maximize lower bound)
            // NaN-safe: propagate NaN instead of silently projecting to 0.0 (#2643)
            entry.value = nan_propagating_max(entry.value + lr * entry.grad, 0.0);
            // NaN guard: if grad was NaN, value is now permanently NaN.
            // Reset to zero (matches Adam path at line ~250). (#2939)
            if !entry.value.is_finite() {
                tracing::warn!(
                    "NaN/Inf in BetaState::gradient_step (SGD), resetting beta and grad to 0.0: layer={}, neuron={}",
                    entry.layer_idx, entry.neuron_idx
                );
                entry.value = 0.0;
                entry.grad = 0.0;
            }
        }
        max_grad
    }

    /// Perform Adam optimizer step for β parameters.
    ///
    /// Adam update rule:
    /// - m = β₁ * m + (1 - β₁) * grad
    /// - v = β₂ * v + (1 - β₂) * grad²
    /// - m_hat = m / (1 - β₁^t)  (bias correction)
    /// - v_hat = v / (1 - β₂^t)  (bias correction)
    /// - β = max(0, β + lr * m_hat / (√v_hat + ε))
    ///
    /// When `config.amsgrad` is true, uses AMSGrad variant:
    /// - v_max = max(v_max, v)
    /// - v_hat = v_max / (1 - β₂^t)  (use v_max instead of v)
    ///
    /// The learning rate is adjusted by:
    /// 1. The scheduler based on iteration `t`
    /// 2. Per-layer scaling based on `config.per_layer_lr_beta`
    ///
    /// Parameter `t` is 1-indexed (first iteration is t=1) for bias correction.
    ///
    /// Returns the maximum raw gradient magnitude for convergence check (#2416).
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) -> f32 {
        let mut max_grad = 0.0f32;
        let t_float = t.max(1) as f32; // Avoid division by zero
        let radam_r = if config.radam {
            radam_rectification_factor(config.beta2, t_float)
        } else {
            None
        };

        // Bias correction factors — .max(EPSILON) guards div-by-zero when beta=1.0 (#2575, #2586)
        let beta1_corr = if config.bias_correction {
            (1.0 - config.beta1.powf(t_float)).max(f32::EPSILON)
        } else {
            1.0
        };
        let beta2_corr = if config.bias_correction {
            (1.0 - config.beta2.powf(t_float)).max(f32::EPSILON)
        } else {
            1.0
        };

        // Compute base scheduled learning rate (scheduler uses 0-indexed iteration)
        let base_scheduled_lr = config.scheduler.lr(t.saturating_sub(1), config.beta_lr);

        for entry in &mut self.entries {
            // Compute per-layer LR factor
            let total_layers = if config.total_layers > 0 {
                config.total_layers
            } else {
                entry.layer_idx + 1 // Fallback: assume current layer is the deepest
            };
            let layer_factor = config
                .per_layer_lr_beta
                .factor(entry.layer_idx, total_layers);
            let scheduled_lr = base_scheduled_lr * layer_factor;

            // Gradient clipping (NaN.clamp() returns NaN — not a NaN filter, #2596)
            let grad = if config.grad_clip > 0.0 {
                entry.grad.clamp(-config.grad_clip, config.grad_clip)
            } else {
                entry.grad
            };

            // Update biased first moment estimate
            entry.m = config.beta1 * entry.m + (1.0 - config.beta1) * grad;

            // Update biased second raw moment estimate
            entry.v = config.beta2 * entry.v + (1.0 - config.beta2) * grad * grad;

            // Compute bias-corrected estimates
            let m_hat = entry.m / beta1_corr;

            // AMSGrad: use maximum of past squared gradients for stable convergence.
            // #3111: use nan_propagating_max instead of f32::max so NaN in v
            // propagates immediately rather than being silently absorbed.
            let v_for_update = if config.amsgrad {
                entry.v_max = nan_propagating_max(entry.v_max, entry.v);
                entry.v_max
            } else {
                entry.v
            };
            let v_hat = v_for_update / beta2_corr;

            // NaN-aware convergence tracking: f32::max silently drops NaN (#2939).
            // Use raw gradient (not m_hat) for convergence check. m_hat is an EMA
            // (β₁=0.9) that lags the true gradient — after a flat→steep transition,
            // m_hat stays near zero for several iterations, causing premature
            // termination. Raw gradient reflects the current landscape. (#2416)
            // GraphBetaState already uses raw gradient; this makes all states consistent.
            max_grad = nan_propagating_max(max_grad, grad.abs());

            // Adaptive update:
            // - Adam: θ = θ + lr * m_hat / (√v_hat + ε)
            // - RAdam: use SGD-with-momentum style step for early iterations; otherwise
            //   apply rectification factor r_t.
            // Gradient ascent (we want to maximize lower bound).
            let update = if config.radam {
                if let Some(r_t) = radam_r {
                    scheduled_lr * r_t * m_hat / (v_hat.sqrt() + config.epsilon)
                } else {
                    scheduled_lr * m_hat
                }
            } else {
                scheduled_lr * m_hat / (v_hat.sqrt() + config.epsilon)
            };

            // AdamW: apply decoupled weight decay directly to parameters
            // θ = θ * (1 - lr * λ) + update
            let decay_factor = if config.weight_decay > 0.0 {
                1.0 - scheduled_lr * config.weight_decay
            } else {
                1.0
            };
            // NaN-safe: propagate NaN instead of silently projecting to 0.0 (#2643)
            entry.value = nan_propagating_max(entry.value * decay_factor + update, 0.0);
            // NaN guard: NaN propagates through projection; m/v remain NaN permanently.
            // Check m (first to become NaN via grad propagation). (#2596)
            if !entry.m.is_finite() || !entry.v.is_finite() {
                tracing::warn!(
                    "NaN/Inf in BetaState::gradient_step_adam, resetting beta/m/v/v_max to 0.0: layer={}, neuron={}, m={}, v={}",
                    entry.layer_idx, entry.neuron_idx, entry.m, entry.v
                );
                entry.value = 0.0;
                entry.m = 0.0;
                entry.v = 0.0;
                entry.v_max = 0.0;
            }
        }
        max_grad
    }

    /// The signed beta contribution for A matrix modification.
    /// Returns β * sign for the specified neuron.
    pub fn signed_beta(&self, layer_idx: usize, neuron_idx: usize) -> Option<f32> {
        self.entry(layer_idx, neuron_idx).map(|e| e.signed_value())
    }

    /// Check if there are any beta entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get total number of constrained neurons.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get all entries for a specific layer.
    pub fn entries_for_layer(&self, layer_idx: usize) -> impl Iterator<Item = &BetaEntry> {
        if self.indexed_entries == self.entries.len() {
            let empty: &[usize] = &[];
            let indices = self
                .layer_index
                .get(&layer_idx)
                .map(Vec::as_slice)
                .unwrap_or(empty);
            BetaEntriesForLayer::Indexed {
                indices: indices.iter(),
                entries: &self.entries,
            }
        } else {
            BetaEntriesForLayer::Linear {
                entries: self.entries.iter(),
                layer_idx,
            }
        }
    }

    /// Initialize slow weights for Lookahead optimizer.
    ///
    /// Should be called once at the beginning of optimization when lookahead is enabled.
    /// Copies current β values as the initial slow weights.
    pub fn init_slow_weights(&mut self) {
        self.slow_weights = Some(self.entries.iter().map(|e| e.value()).collect());
    }

    /// Perform Lookahead synchronization step: slow = slow + α*(fast - slow), then fast = slow.
    /// Called every `sync_period` iterations. Requires prior `init_slow_weights()` call.
    pub fn lookahead_step(&mut self, config: &LookaheadConfig) -> Result<()> {
        let slow = self.slow_weights.as_mut().ok_or_else(|| {
            NyError::InvalidSpec(
                "BetaState::lookahead_step requires init_slow_weights()".to_string(),
            )
        })?;

        if slow.len() != self.entries.len() {
            return Err(NyError::InternalError(format!(
                "BetaState::lookahead_step: slow weights length ({}) != entries length ({})",
                slow.len(),
                self.entries.len(),
            )));
        }

        for (i, entry) in self.entries.iter_mut().enumerate() {
            let fast = entry.value;
            // slow = slow + α * (fast - slow)
            slow[i] = slow[i] + config.alpha * (fast - slow[i]);
            // fast = slow (reset fast weights to slow)
            // Apply projection to [0, ∞) since β must be non-negative
            // NaN-safe: propagate NaN instead of silently projecting to 0.0 (#2643)
            entry.value = nan_propagating_max(slow[i], 0.0);
        }
        Ok(())
    }

    /// Check if slow weights are initialized for Lookahead.
    pub fn has_slow_weights(&self) -> bool {
        self.slow_weights.is_some()
    }

    /// Current slow weights (for debugging/testing).
    pub fn slow_weights(&self) -> Option<&[f32]> {
        self.slow_weights.as_deref()
    }
}

#[cfg(test)]
#[path = "beta_tests.rs"]
mod tests;
