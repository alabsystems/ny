// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use ny_tensor::BoundedTensor;

use super::neuron::AlphaNeuronState;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

/// Graph-domain-specific α state for joint α-β optimization in graph BaB.
///
/// Like `DomainAlphaState` but keyed by `(node_name, neuron_idx)` instead of
/// `(layer_idx, neuron_idx)`, matching the graph network's node-based addressing.
///
/// This enables per-domain alpha optimization in the graph BaB path, which was
/// previously using only the fixed heuristic (alpha = 1 if u > -l, else 0).
///
/// # Reference
/// alpha-beta-CROWN: `auto_LiRPA/operators/relu.py` — optimizable slopes
/// Issue: #1841
#[derive(Debug, Clone)]
pub struct GraphDomainAlphaState {
    /// Lower-path per-neuron optimization state indexed by node_name → neuron_idx.
    /// Nested structure avoids String allocation per lookup: the outer map
    /// is keyed by `&str` (via `HashMap::get`), the inner by `usize`.
    /// Only stores entries for unstable, unconstrained ReLU neurons.
    pub(crate) neurons: HashMap<String, HashMap<usize, AlphaNeuronState>>,
    /// Upper-path per-neuron optimization state, mirrored to `neurons`.
    pub(crate) upper_neurons: HashMap<String, HashMap<usize, AlphaNeuronState>>,
}

impl GraphDomainAlphaState {
    /// Create empty alpha state.
    pub fn empty() -> Self {
        Self {
            neurons: HashMap::new(),
            upper_neurons: HashMap::new(),
        }
    }

    /// Convert a global `GraphAlphaState` into per-neuron format for the
    /// batched backward kernel in input-split paths.
    ///
    /// In input-split verification, all sub-domains share the global α-CROWN
    /// state from the warmup pass. This converter creates a `GraphDomainAlphaState`
    /// with entries for neurons that were unstable at the root level. Sub-domain
    /// unstable neurons are always a subset of root unstable neurons (bounds
    /// get tighter after splitting), so this covers all relevant neurons.
    ///
    /// The batched backward core's `build_alpha_array` handles stable/unstable
    /// classification at runtime using per-domain pre-activation bounds, so
    /// stable neurons automatically get their correct slopes (0 or 1) regardless
    /// of the stored alpha values.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/input_split/branching_domains.py` —
    ///   "fix_interm_bounds" pattern shares root alpha across all sub-domains.
    /// Part of #4210.
    pub fn from_global_alpha_state_for_input_split(
        global: &crate::bounds::GraphAlphaState,
    ) -> Self {
        let mut neurons = HashMap::new();
        let mut upper_neurons = HashMap::new();

        for (node_name, alphas) in &global.alphas {
            let mask = global.unstable_mask.get(node_name);
            let mut node_map = HashMap::new();
            for (i, &alpha_val) in alphas.iter().enumerate() {
                let is_unstable = mask
                    .map(|m| m.get(i).copied().unwrap_or(false))
                    .unwrap_or(true); // conservative: include if no mask
                if is_unstable {
                    node_map.insert(i, AlphaNeuronState::new(alpha_val));
                }
            }
            if !node_map.is_empty() {
                neurons.insert(node_name.clone(), node_map);
            }
        }

        for (node_name, alphas_upper) in &global.alphas_upper {
            let mask = global.unstable_mask.get(node_name);
            let mut node_map = HashMap::new();
            for (i, &alpha_val) in alphas_upper.iter().enumerate() {
                let is_unstable = mask
                    .map(|m| m.get(i).copied().unwrap_or(false))
                    .unwrap_or(true);
                if is_unstable {
                    node_map.insert(i, AlphaNeuronState::new(alpha_val));
                }
            }
            if !node_map.is_empty() {
                upper_neurons.insert(node_name.clone(), node_map);
            }
        }

        Self {
            neurons,
            upper_neurons,
        }
    }

    /// Initialize α state from graph node bounds and split history.
    ///
    /// For each ReLU node, identifies unstable neurons (l < 0 < u) that are not
    /// constrained by the history, and initializes α with the standard heuristic.
    ///
    /// # Arguments
    /// * `graph` - The graph network containing ReLU nodes
    /// * `node_bounds` - Pre-activation bounds for each node
    /// * `history` - Split history with ReLU constraints
    /// * `input_bounds` - Input bounds for the network
    pub fn from_graph_bounds(
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        let mut state = Self::empty();

        for node_name in graph.node_names() {
            let node = match graph.node(node_name) {
                Some(n) => n,
                None => continue,
            };

            // Only process ReLU nodes
            if !matches!(node.layer, Layer::ReLU(_)) {
                continue;
            }

            // #2098: Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
            let pre_name = match node.inputs.first() {
                Some(s) => s.as_str(),
                None => {
                    tracing::warn!(node = %node_name, "ReLU node has empty inputs — skipping");
                    continue;
                }
            };
            let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
                input_bounds
            } else {
                match node_bounds.get(pre_name) {
                    Some(b) => b.as_ref(),
                    None => continue,
                }
            };

            let pre_flat = pre_bounds.flatten();
            for neuron_idx in 0..pre_flat.len() {
                let l = pre_flat.lower()[[neuron_idx]];
                let u = pre_flat.upper()[[neuron_idx]];

                // Skip stable neurons
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Skip constrained neurons (they use fixed slopes 0 or 1)
                if history.is_constrained(node_name, neuron_idx).is_some() {
                    continue;
                }

                // Unstable and unconstrained: initialize α with heuristic
                let alpha = if u > -l { 1.0 } else { 0.0 };
                let neuron_state = AlphaNeuronState::new(alpha);
                state
                    .neurons
                    .entry(node_name.clone())
                    .or_default()
                    .insert(neuron_idx, neuron_state);
                state
                    .upper_neurons
                    .entry(node_name.clone())
                    .or_default()
                    .insert(neuron_idx, neuron_state);
            }
        }

        state
    }

    /// Initialize from root-level optimized α-CROWN state.
    ///
    /// Transfers the gradient-optimized alpha values from `GraphAlphaState`
    /// (used during root-level α-CROWN in `collect_alpha_crown_bounds_dag`)
    /// into a sparse `GraphDomainAlphaState` for the BaB root domain.
    ///
    /// Without this, root domain alpha would be re-initialized from the
    /// `u > -l` heuristic, discarding the SPSA/Adam-optimized values.
    /// This is the primary fix for #1851 Cause 1.
    ///
    /// # Arguments
    /// * `root_alpha` - Optimized `GraphAlphaState` from `collect_alpha_crown_bounds_dag`
    /// * `graph` - Graph network for identifying ReLU nodes
    /// * `node_bounds` - Pre-activation bounds per node
    /// * `history` - Split history (empty for root domain)
    /// * `input_bounds` - Input bounds for the network
    pub fn from_root_alpha_state(
        root_alpha: &crate::bounds::GraphAlphaState,
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        // Start with heuristic initialization (handles neuron enumeration,
        // constraint filtering, and stable neuron skipping).
        let mut state = Self::from_graph_bounds(graph, node_bounds, history, input_bounds);

        // #hard-six α-inherit-expand (dark, NY_ALPHA_INHERIT_EXPAND=1):
        // when the warmup ran CHANNEL-ONLY α (`full_conv_alpha: false`, e.g.
        // the cifar100_2024 preset — 1188 shared params for ~24k unstable
        // neurons), `root_alpha.alphas[node]` has length C while `neuron_idx`
        // here is the FLAT per-neuron index into [C, H, W]. The historical
        // code below then (a) silently keeps the heuristic α for every neuron
        // with flat index ≥ C (the warmup optimum never reaches BaB) and
        // (b) mis-indexes the first C neurons with OTHER channels' α values.
        // With the gate on, channel-only arrays are spatially broadcast to
        // per-neuron before the override, so every unstable neuron seeds from
        // its own channel's warmup-optimized α. Unset ⇒ byte-identical.
        // SOUND either way: any α ∈ [0,1] yields a valid CROWN relaxation;
        // this only changes the ascent's starting point.
        let expand_gate = std::env::var("NY_ALPHA_INHERIT_EXPAND").ok().as_deref() == Some("1");

        // Override heuristic alpha with optimized values from root α-CROWN.
        // GraphAlphaState stores dense Array1 per node name; GraphDomainAlphaState
        // stores sparse node_name → neuron_idx entries.
        for (node_name, neuron_map) in &mut state.neurons {
            if let Some(alpha_arr) = root_alpha.alphas.get(node_name) {
                let expanded = (expand_gate && root_alpha.spatial_shape(node_name).is_some())
                    .then(|| root_alpha.expand_alpha(node_name, alpha_arr));
                let alpha_arr = expanded.as_ref().unwrap_or(alpha_arr);
                for (&neuron_idx, neuron_state) in neuron_map.iter_mut() {
                    if neuron_idx < alpha_arr.len() {
                        neuron_state.set_alpha(alpha_arr[neuron_idx]);
                    }
                }
            }
        }

        for (node_name, neuron_map) in &mut state.upper_neurons {
            let alpha_arr = root_alpha
                .alphas_upper
                .get(node_name)
                .or_else(|| root_alpha.alphas.get(node_name));
            if let Some(alpha_arr) = alpha_arr {
                let expanded = (expand_gate && root_alpha.spatial_shape(node_name).is_some())
                    .then(|| root_alpha.expand_alpha(node_name, alpha_arr));
                let alpha_arr = expanded.as_ref().unwrap_or(alpha_arr);
                for (&neuron_idx, neuron_state) in neuron_map.iter_mut() {
                    if neuron_idx < alpha_arr.len() {
                        neuron_state.set_alpha(alpha_arr[neuron_idx]);
                    }
                }
            }
        }

        state
    }

    /// Initialize from parent state, inheriting optimized alpha values.
    ///
    /// For neurons that were also unstable in the parent domain, copies the
    /// parent's optimized alpha (warm start). For newly unstable neurons,
    /// uses the heuristic initialization.
    pub fn from_parent(
        parent: &GraphDomainAlphaState,
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        let mut state = Self::from_graph_bounds(graph, node_bounds, history, input_bounds);

        // Warm-start: inherit parent's optimized alpha for matching neurons
        for (node_name, parent_neuron_map) in &parent.neurons {
            if let Some(child_neuron_map) = state.neurons.get_mut(node_name) {
                for (&neuron_idx, parent_neuron) in parent_neuron_map {
                    if let Some(child_neuron) = child_neuron_map.get_mut(&neuron_idx) {
                        child_neuron.set_alpha(parent_neuron.alpha);
                        // Reset optimizer state for fresh optimization in child domain
                    }
                }
            }
        }

        for (node_name, parent_neuron_map) in &parent.neurons {
            if let Some(child_neuron_map) = state.upper_neurons.get_mut(node_name) {
                for (&neuron_idx, parent_neuron) in parent_neuron_map {
                    if let Some(child_neuron) = child_neuron_map.get_mut(&neuron_idx) {
                        let upper_alpha = parent
                            .upper_neurons
                            .get(node_name)
                            .and_then(|m| m.get(&neuron_idx))
                            .map_or(parent_neuron.alpha(), AlphaNeuronState::alpha);
                        child_neuron.set_alpha(upper_alpha);
                    }
                }
            }
        }

        state
    }
}
