// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::{nan_propagating_max, NyError, Result};
use ny_tensor::BoundedTensor;

use super::neuron::{sanitize_alpha, AlphaNeuronState};
use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::config::{radam_rectification_factor, AdaptiveOptConfig, LookaheadConfig};
use crate::{Layer, Network};

fn scheduled_alpha_lr(base_scheduled_lr: f32, config: &AdaptiveOptConfig, layer_idx: usize) -> f32 {
    let total_layers = if config.total_layers > 0 {
        config.total_layers
    } else {
        layer_idx + 1 // Fallback: assume current layer is the deepest
    };
    let layer_factor = config.per_layer_lr_alpha.factor(layer_idx, total_layers);
    base_scheduled_lr * layer_factor
}

fn step_domain_alpha_neuron_adam(
    neuron: &mut AlphaNeuronState,
    config: &AdaptiveOptConfig,
    beta1_corr: f32,
    beta2_corr: f32,
    scheduled_lr: f32,
    radam_r: Option<f32>,
) -> f32 {
    let grad_raw = neuron.grad;

    // Gradient clipping
    let grad = if config.grad_clip > 0.0 {
        grad_raw.clamp(-config.grad_clip, config.grad_clip)
    } else {
        grad_raw
    };

    // Update biased first and second moment estimates.
    neuron.adam_m = config.beta1 * neuron.adam_m + (1.0 - config.beta1) * grad;
    neuron.adam_v = config.beta2 * neuron.adam_v + (1.0 - config.beta2) * grad * grad;

    let m_hat = neuron.adam_m / beta1_corr;
    let v_for_update = if config.amsgrad {
        neuron.adam_v_max = nan_propagating_max(neuron.adam_v_max, neuron.adam_v);
        neuron.adam_v_max
    } else {
        neuron.adam_v
    };
    let v_hat = v_for_update / beta2_corr;

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
    neuron.set_alpha(neuron.alpha * decay_factor + update);
    // NaN guard: set_alpha catches NaN→0.5 but m/v remain NaN,
    // permanently corrupting this neuron's optimizer. Reset all state.
    // Matches sequential alpha.rs:136-140 pattern. (#2595)
    if !neuron.adam_m.is_finite() || !neuron.adam_v.is_finite() {
        neuron.adam_m = 0.0;
        neuron.adam_v = 0.0;
        neuron.adam_v_max = 0.0;
    }

    grad.abs()
}

/// Domain-specific α state for joint α-β optimization.
///
/// Unlike `AlphaState` which uses relu_idx (position in list of ReLU layers),
/// this struct maps α values by layer_idx for direct integration with β-CROWN.
///
/// Uses a single HashMap with `AlphaNeuronState` per neuron instead of 7
/// separate HashMaps, reducing memory from ~392 bytes/neuron to ~40 bytes/neuron.
#[derive(Debug, Clone)]
pub struct DomainAlphaState {
    /// Per-neuron optimization state indexed by (layer_idx, neuron_idx).
    /// Only stores entries for unstable, unconstrained ReLU neurons.
    pub(crate) neurons: HashMap<(usize, usize), AlphaNeuronState>,
    /// Slow weights for Lookahead optimizer.
    /// When Some, contains α values that are a moving average of the fast weights.
    pub(crate) slow_alphas: Option<HashMap<(usize, usize), f32>>,
}

impl DomainAlphaState {
    /// Create empty alpha state.
    pub fn empty() -> Self {
        Self {
            neurons: HashMap::new(),
            slow_alphas: None, // Initialized lazily when lookahead is first used
        }
    }

    /// Initialize α state from layer bounds and constraints.
    ///
    /// For each ReLU layer, identifies unstable neurons (l < 0 < u) that are not
    /// constrained, and initializes α using the standard heuristic: α = 1 if u > -l, else 0.
    pub fn from_layer_bounds_and_constraints(
        network: &Network,
        layer_bounds: &[Arc<BoundedTensor>],
        history: &SplitHistory,
    ) -> Self {
        let mut state = Self::empty();

        // Build constraint lookup
        let mut constraints: HashMap<(usize, usize), bool> = HashMap::new();
        for c in &history.constraints {
            constraints.insert((c.layer_idx, c.neuron_idx), c.is_active);
        }

        // Find ReLU layers and initialize α for unstable neurons
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !matches!(layer, Layer::ReLU(_)) {
                continue;
            }

            // Get pre-activation bounds for this ReLU
            if layer_idx == 0 || layer_idx > layer_bounds.len() {
                continue;
            }
            let pre_bounds = &layer_bounds[layer_idx - 1];
            let pre_flat = pre_bounds.flatten();

            for neuron_idx in 0..pre_flat.len() {
                let l = pre_flat.lower()[[neuron_idx]];
                let u = pre_flat.upper()[[neuron_idx]];

                // Skip stable neurons
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Skip constrained neurons (they use fixed slopes 0 or 1)
                if constraints.contains_key(&(layer_idx, neuron_idx)) {
                    continue;
                }

                // Unstable and not constrained: initialize α with heuristic
                let alpha = if u > -l { 1.0 } else { 0.0 };
                state
                    .neurons
                    .insert((layer_idx, neuron_idx), AlphaNeuronState::new(alpha));
            }
        }

        state
    }

    /// α value for a neuron. Returns the heuristic default if not found.
    pub fn alpha(&self, layer_idx: usize, neuron_idx: usize) -> f32 {
        self.neurons
            .get(&(layer_idx, neuron_idx))
            .map_or(0.0, |n| n.alpha())
    }

    /// Set α value for a neuron.
    pub fn set_alpha(&mut self, layer_idx: usize, neuron_idx: usize, value: f32) {
        if let Some(n) = self.neurons.get_mut(&(layer_idx, neuron_idx)) {
            n.set_alpha(value);
        }
    }

    /// Check if a neuron has an optimizable α (present in the neurons map).
    pub fn is_unstable(&self, layer_idx: usize, neuron_idx: usize) -> bool {
        self.neurons.contains_key(&(layer_idx, neuron_idx))
    }

    /// Reset all gradients to zero.
    pub fn zero_grad(&mut self) {
        for n in self.neurons.values_mut() {
            n.grad = 0.0;
        }
    }

    /// Accumulate gradient for a specific neuron.
    ///
    /// NaN guard (#3112): skip non-finite gradients to prevent NaN from entering the
    /// optimizer state. Without this, a single NaN gradient poisons m/v accumulators
    /// and wastes an Adam step before the post-step guard resets everything.
    pub fn accumulate_grad(&mut self, layer_idx: usize, neuron_idx: usize, grad: f32) {
        if !grad.is_finite() {
            return;
        }
        if let Some(n) = self.neurons.get_mut(&(layer_idx, neuron_idx)) {
            n.grad += grad;
        }
    }

    /// Perform gradient ascent step with optional momentum.
    /// Returns the maximum gradient magnitude (for convergence check).
    pub fn gradient_step(&mut self, lr: f32, momentum: f32) -> f32 {
        let mut max_grad = 0.0f32;

        for n in self.neurons.values_mut() {
            // NaN-aware convergence tracking: f32::max silently drops NaN (#2939).
            max_grad = nan_propagating_max(max_grad, n.grad.abs());

            // Update with momentum
            n.velocity = momentum * n.velocity + lr * n.grad;

            // Gradient ascent (we want to maximize lower bound)
            n.set_alpha(n.alpha + n.velocity);
            // NaN guard: set_alpha catches NaN→0.5 but velocity remains NaN,
            // permanently corrupting this neuron's momentum. Reset velocity. (#2608)
            if !n.velocity.is_finite() {
                n.velocity = 0.0;
            }
        }
        max_grad
    }

    /// Perform Adam optimizer step for α parameters.
    ///
    /// Similar to BetaState::gradient_step_adam, but additionally clamps α to \[0, 1\].
    /// The learning rate is adjusted by:
    /// 1. The scheduler based on iteration `t`
    /// 2. Per-layer scaling based on `config.per_layer_lr_alpha`
    ///
    /// Parameter `t` is 1-indexed (first iteration is t=1) for bias correction.
    ///
    /// When `config.amsgrad` is true, uses AMSGrad variant:
    /// - v_max = max(v_max, v)
    /// - v_hat = v_max / (1 - β₂^t)  (use v_max instead of v)
    ///
    /// Returns the maximum raw gradient magnitude for convergence check (#2416).
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) -> f32 {
        let mut max_grad = 0.0f32;
        let t_float = t.max(1) as f32;
        let radam_r = if config.radam {
            radam_rectification_factor(config.beta2, t_float)
        } else {
            None
        };

        // Bias correction factors — .max(EPSILON) guards division by zero (#2315, #2556)
        let beta1_corr = if config.bias_correction {
            (1.0f32 - config.beta1.powf(t_float)).max(f32::EPSILON)
        } else {
            1.0
        };
        let beta2_corr = if config.bias_correction {
            (1.0f32 - config.beta2.powf(t_float)).max(f32::EPSILON)
        } else {
            1.0
        };

        // Compute base scheduled learning rate (scheduler uses 0-indexed iteration)
        let base_scheduled_lr = config.scheduler.lr(t.saturating_sub(1), config.alpha_lr);

        for (&(layer_idx, _neuron_idx), neuron) in &mut self.neurons {
            let scheduled_lr = scheduled_alpha_lr(base_scheduled_lr, config, layer_idx);

            // NaN-aware convergence tracking: f32::max silently drops NaN (#2939).
            // Use raw gradient (not m_hat) for convergence check. m_hat is an EMA
            // (β₁=0.9) that lags the true gradient — after a flat→steep transition,
            // m_hat stays near zero for several iterations, causing premature
            // termination. Raw gradient reflects the current landscape. (#2416)
            // GraphBetaState already uses raw gradient; this makes all states consistent.
            let grad_abs = step_domain_alpha_neuron_adam(
                neuron,
                config,
                beta1_corr,
                beta2_corr,
                scheduled_lr,
                radam_r,
            );
            max_grad = nan_propagating_max(max_grad, grad_abs);
        }
        max_grad
    }

    /// Check if there are any optimizable α values.
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Get the number of optimizable α values.
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Get read-only access to the per-neuron state map.
    pub fn neurons(&self) -> &HashMap<(usize, usize), AlphaNeuronState> {
        &self.neurons
    }

    /// Get mutable access to the per-neuron state map.
    pub fn neurons_mut(&mut self) -> &mut HashMap<(usize, usize), AlphaNeuronState> {
        &mut self.neurons
    }

    /// Initialize slow weights for Lookahead optimizer.
    ///
    /// Should be called once at the beginning of optimization when lookahead is enabled.
    /// Copies current α values as the initial slow weights.
    pub fn init_slow_weights(&mut self) {
        self.slow_alphas = Some(self.neurons.iter().map(|(&k, n)| (k, n.alpha)).collect());
    }

    /// Perform Lookahead synchronization step.
    ///
    /// This should be called after every `sync_period` iterations of the inner optimizer.
    ///
    /// Algorithm:
    /// 1. slow = slow + α * (fast - slow)  [interpolate slow toward fast]
    /// 2. fast = slow  [reset fast weights to slow]
    ///
    /// # Arguments
    /// * `config` - Lookahead configuration with interpolation coefficient α
    ///
    /// # Errors
    /// Returns `NyError::InvalidSpec` if slow weights are uninitialized
    /// (caller must invoke `init_slow_weights()` first).
    pub fn lookahead_step(&mut self, config: &LookaheadConfig) -> Result<()> {
        let slow = self.slow_alphas.as_mut().ok_or_else(|| {
            NyError::InvalidSpec(
                "DomainAlphaState::lookahead_step requires init_slow_weights()".to_string(),
            )
        })?;

        for (&key, n) in &mut self.neurons {
            let slow_val = slow.entry(key).or_insert(n.alpha);
            // slow = slow + α * (fast - slow)
            *slow_val = sanitize_alpha(*slow_val + config.alpha * (n.alpha - *slow_val));
            // fast = slow (reset fast weights to slow)
            // Apply projection to [0, 1] since α must be in [0, 1]
            n.set_alpha(*slow_val);
        }
        Ok(())
    }

    /// Check if slow weights are initialized for Lookahead.
    pub fn has_slow_weights(&self) -> bool {
        self.slow_alphas.is_some()
    }

    /// Current slow weights (for debugging/testing).
    pub fn slow_weights(&self) -> Option<&HashMap<(usize, usize), f32>> {
        self.slow_alphas.as_ref()
    }
}
