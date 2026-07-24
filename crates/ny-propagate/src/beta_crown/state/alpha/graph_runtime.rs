// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ny_core::nan_propagating_max;
use ny_tensor::BoundedTensor;

use super::graph_init::GraphDomainAlphaState;
use super::neuron::AlphaNeuronState;
use crate::beta_crown::config::AdaptiveOptConfig;

fn step_graph_alpha_neuron_maps_adam(
    neuron_maps: &mut HashMap<String, HashMap<usize, AlphaNeuronState>>,
    config: &AdaptiveOptConfig,
    t: usize,
) -> f32 {
    let mut max_grad = 0.0f32;
    let t_float = t.max(1) as f32;

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

    let scheduled_lr = config.scheduler.lr(t.saturating_sub(1), config.alpha_lr);

    for neuron_map in neuron_maps.values_mut() {
        for n in neuron_map.values_mut() {
            let grad = if config.grad_clip > 0.0 {
                n.grad.clamp(-config.grad_clip, config.grad_clip)
            } else {
                n.grad
            };

            n.adam_m = config.beta1 * n.adam_m + (1.0 - config.beta1) * grad;
            n.adam_v = config.beta2 * n.adam_v + (1.0 - config.beta2) * grad * grad;

            let m_hat = n.adam_m / beta1_corr;
            let v_for_update = if config.amsgrad {
                n.adam_v_max = nan_propagating_max(n.adam_v_max, n.adam_v);
                n.adam_v_max
            } else {
                n.adam_v
            };
            let v_hat = v_for_update / beta2_corr;

            max_grad = nan_propagating_max(max_grad, grad.abs());

            let update = scheduled_lr * m_hat / (v_hat.sqrt() + config.epsilon);
            n.set_alpha(n.alpha + update);
            if !n.adam_m.is_finite() || !n.adam_v.is_finite() {
                n.adam_m = 0.0;
                n.adam_v = 0.0;
                n.adam_v_max = 0.0;
            }
        }
    }

    max_grad
}

impl GraphDomainAlphaState {
    /// α value for a neuron. Returns the heuristic default (0.0) if not found.
    ///
    /// Zero heap allocations: looks up node by `&str`, then neuron by `usize`.
    pub fn alpha(&self, node_name: &str, neuron_idx: usize) -> f32 {
        self.neurons
            .get(node_name)
            .and_then(|m| m.get(&neuron_idx))
            .map_or(0.0, |n| n.alpha())
    }

    /// Upper-path α value for a neuron. Returns the heuristic default (0.0) if not found.
    pub fn alpha_upper(&self, node_name: &str, neuron_idx: usize) -> f32 {
        self.upper_neurons
            .get(node_name)
            .and_then(|m| m.get(&neuron_idx))
            .map_or(0.0, |n| n.alpha())
    }

    fn build_alpha_array_from_node_map(
        node_neurons: Option<&HashMap<usize, AlphaNeuronState>>,
        pre_activation: &BoundedTensor,
    ) -> ndarray::Array1<f32> {
        let pre_flat = pre_activation.flatten();
        let n = pre_flat.len();
        let mut alphas = ndarray::Array1::<f32>::zeros(n);

        for i in 0..n {
            let l = pre_flat.lower()[[i]];
            let u = pre_flat.upper()[[i]];

            if l >= 0.0 {
                alphas[i] = 1.0;
            } else if u <= 0.0 {
                alphas[i] = 0.0;
            } else if let Some(neuron) = node_neurons.and_then(|m| m.get(&i)) {
                alphas[i] = neuron.alpha();
            } else {
                alphas[i] = if u > -l { 1.0 } else { 0.0 };
            }
        }

        alphas
    }

    /// Build a lower-path alpha array for a specific ReLU node.
    ///
    /// Returns an `Array1<f32>` of alpha values for all neurons in the node,
    /// using optimized values from the state and the heuristic for missing neurons.
    ///
    /// Zero per-neuron String allocations: looks up the node map once, then
    /// indexes by `usize` for each neuron.
    pub fn build_alpha_array(
        &self,
        node_name: &str,
        pre_activation: &BoundedTensor,
    ) -> ndarray::Array1<f32> {
        Self::build_alpha_array_from_node_map(self.neurons.get(node_name), pre_activation)
    }

    /// Build an upper-path alpha array for a specific ReLU node.
    pub fn build_alpha_upper_array(
        &self,
        node_name: &str,
        pre_activation: &BoundedTensor,
    ) -> ndarray::Array1<f32> {
        Self::build_alpha_array_from_node_map(self.upper_neurons.get(node_name), pre_activation)
    }

    /// Reset all gradients to zero.
    pub fn zero_grad(&mut self) {
        for neuron_map in self.neurons.values_mut() {
            for n in neuron_map.values_mut() {
                n.grad = 0.0;
            }
        }
        for neuron_map in self.upper_neurons.values_mut() {
            for n in neuron_map.values_mut() {
                n.grad = 0.0;
            }
        }
    }

    /// Accumulate gradient for a specific neuron.
    ///
    /// Zero heap allocations: looks up node by `&str`, then neuron by `usize`.
    /// NaN guard (#3112): skip non-finite gradients to prevent NaN from entering the
    /// optimizer state.
    pub fn accumulate_grad(&mut self, node_name: &str, neuron_idx: usize, grad: f32) {
        if !grad.is_finite() {
            return;
        }
        if let Some(n) = self
            .neurons
            .get_mut(node_name)
            .and_then(|m| m.get_mut(&neuron_idx))
        {
            n.grad += grad;
        }
    }

    /// Accumulate upper-path gradient for a specific neuron.
    pub fn accumulate_grad_upper(&mut self, node_name: &str, neuron_idx: usize, grad: f32) {
        if !grad.is_finite() {
            return;
        }
        if let Some(n) = self
            .upper_neurons
            .get_mut(node_name)
            .and_then(|m| m.get_mut(&neuron_idx))
        {
            n.grad += grad;
        }
    }

    /// Perform Adam optimizer step for α parameters.
    ///
    /// Returns the maximum raw gradient magnitude for convergence check (#2416).
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) -> f32 {
        let lower_max = step_graph_alpha_neuron_maps_adam(&mut self.neurons, config, t);
        let upper_max = step_graph_alpha_neuron_maps_adam(&mut self.upper_neurons, config, t);
        nan_propagating_max(lower_max, upper_max)
    }

    /// Check if there are any optimizable α values.
    pub fn is_empty(&self) -> bool {
        self.neurons.values().all(|m| m.is_empty())
            && self.upper_neurons.values().all(|m| m.is_empty())
    }

    /// Get the total number of optimizable α values across all nodes.
    pub fn len(&self) -> usize {
        self.neurons.values().map(|m| m.len()).sum()
    }

    /// Get read-only access to the nested per-neuron state map.
    pub fn neurons(&self) -> &HashMap<String, HashMap<usize, AlphaNeuronState>> {
        &self.neurons
    }

    /// Get read-only access to the nested upper-path per-neuron state map.
    pub fn upper_neurons(&self) -> &HashMap<String, HashMap<usize, AlphaNeuronState>> {
        &self.upper_neurons
    }

    /// Get mutable access to the nested lower-path per-neuron state map.
    pub fn neurons_mut(&mut self) -> &mut HashMap<String, HashMap<usize, AlphaNeuronState>> {
        &mut self.neurons
    }

    /// Get mutable access to the nested upper-path per-neuron state map.
    pub fn upper_neurons_mut(&mut self) -> &mut HashMap<String, HashMap<usize, AlphaNeuronState>> {
        &mut self.upper_neurons
    }

    /// A specific neuron's state by node name and neuron index.
    ///
    /// Zero heap allocations.
    pub fn neuron(&self, node_name: &str, neuron_idx: usize) -> Option<&AlphaNeuronState> {
        self.neurons.get(node_name).and_then(|m| m.get(&neuron_idx))
    }

    /// Mutable access to a specific neuron's state.
    ///
    /// Zero heap allocations.
    pub fn neuron_mut(
        &mut self,
        node_name: &str,
        neuron_idx: usize,
    ) -> Option<&mut AlphaNeuronState> {
        self.neurons
            .get_mut(node_name)
            .and_then(|m| m.get_mut(&neuron_idx))
    }

    /// Insert a neuron state for the given node and neuron index.
    pub fn insert(&mut self, node_name: String, neuron_idx: usize, state: AlphaNeuronState) {
        self.neurons
            .entry(node_name.clone())
            .or_default()
            .insert(neuron_idx, state);
        self.upper_neurons
            .entry(node_name)
            .or_default()
            .insert(neuron_idx, state);
    }

    /// #hard-six unshared-α: copy every lower-path α into the matching
    /// upper-path slot so a persisted (Adam-stepped) state keeps
    /// `lower == upper`.
    ///
    /// The wide α ascent accumulates gradients into the LOWER map only
    /// (`accumulate_grad`), so a stepped snapshot diverges from its upper
    /// map; a child prepped from a diverged state upgrades every ReLU with
    /// a stepped neuron to `GpuCrownLayer::ActivationReluDualAlpha`
    /// (`extract_relu_gpu_layer_with_alpha`), which
    /// `crown_backward_gpu_resnet_sound_beta_batched_grad` rejects as
    /// unbatchable — silently demoting every later batch containing such a
    /// child to the serial per-domain ascent (the measured prop54 1-wide /
    /// 110-serial collapse). SOUND: the upper-path α parameterizes the α·x
    /// LOWER relaxation consumed by negative-coefficient rows of the upper
    /// bound; any value in [0,1] is admissible and `set_alpha` clamps.
    pub fn sync_upper_from_lower(&mut self) {
        for (node_name, neuron_map) in &self.neurons {
            if let Some(upper_map) = self.upper_neurons.get_mut(node_name) {
                for (neuron_idx, neuron) in neuron_map {
                    if let Some(upper) = upper_map.get_mut(neuron_idx) {
                        upper.set_alpha(neuron.alpha());
                    }
                }
            }
        }
    }
}
