// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::core::GraphBetaState;
use crate::beta_crown::config::AdaptiveOptConfig;
use ny_core::nan_propagating_max;

impl GraphBetaState {
    /// Reset all gradients to zero.
    pub fn zero_grad(&mut self) {
        for entry in &mut self.entries {
            entry.grad = 0.0;
        }
    }

    /// Accumulate gradient for ALL entries matching a specific neuron.
    ///
    /// When GenBaB creates multiple constraints at the same neuron (different
    /// split points), the gradient must flow to all matching entries. Previously
    /// this used `get_entry_mut()` which returns only the first match — leaving
    /// subsequent entries with zero gradient and frozen beta values.
    ///
    /// Fix for #2247: iterate all entries, matching the pattern used by
    /// `get_signed_beta()` which correctly sums contributions from all entries.
    /// NaN guard (#3112): skip non-finite gradients to prevent NaN from entering the
    /// optimizer state. Without this, a single NaN gradient poisons m/v accumulators.
    pub fn accumulate_grad(&mut self, node_name: &str, neuron_idx: usize, grad: f32) {
        if !grad.is_finite() {
            return;
        }
        if !self.lookup_index_fresh() {
            self.rebuild_lookup_index();
        }
        if let Some(indices) = self
            .neuron_index
            .get(node_name)
            .and_then(|by_neuron| by_neuron.get(&neuron_idx))
            .cloned()
        {
            for idx in indices {
                self.entries[idx].grad += grad;
            }
            return;
        }
        for entry in &mut self.entries {
            if entry.node_name == node_name && entry.neuron_idx == neuron_idx {
                entry.grad += grad;
            }
        }
    }

    /// Perform projected gradient ascent step.
    /// β = max(0, β + lr * grad)
    ///
    /// Returns the maximum gradient magnitude (for convergence check).
    pub fn gradient_step(&mut self, lr: f32) -> f32 {
        let mut max_grad = 0.0f32;
        for entry in &mut self.entries {
            max_grad = nan_propagating_max(max_grad, entry.grad.abs());
            entry.value = nan_propagating_max(entry.value + lr * entry.grad, 0.0);
            if !entry.value.is_finite() {
                entry.value = 0.0;
                entry.grad = 0.0;
            }
        }
        max_grad
    }

    /// Perform Adam optimizer step for β parameters.
    ///
    /// Returns the maximum gradient magnitude (for convergence check).
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) -> f32 {
        let lr = config.beta_lr;
        let beta1 = config.beta1;
        let beta2 = config.beta2;
        let eps = config.epsilon;

        let t = t.max(1);
        let t_i32 = t.min(i32::MAX as usize) as i32;
        let bias_correction1 = (1.0 - beta1.powi(t_i32)).max(f32::EPSILON);
        let bias_correction2 = (1.0 - beta2.powi(t_i32)).max(f32::EPSILON);

        let mut max_grad = 0.0f32;
        for entry in &mut self.entries {
            max_grad = nan_propagating_max(max_grad, entry.grad.abs());
            entry.m = beta1 * entry.m + (1.0 - beta1) * entry.grad;
            entry.v = beta2 * entry.v + (1.0 - beta2) * entry.grad * entry.grad;
            entry.v_max = nan_propagating_max(entry.v_max, entry.v);

            let m_hat = entry.m / bias_correction1;
            let v_hat = entry.v_max / bias_correction2;
            entry.value = nan_propagating_max(entry.value + lr * m_hat / (v_hat.sqrt() + eps), 0.0);
            if !entry.m.is_finite() || !entry.v.is_finite() {
                entry.value = 0.0;
                entry.m = 0.0;
                entry.v = 0.0;
                entry.v_max = 0.0;
            }
        }
        max_grad
    }
}
