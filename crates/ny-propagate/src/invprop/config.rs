// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! INVPROP configuration for output constraint backward propagation.

use serde::{Deserialize, Serialize};

use crate::NETWORK_INPUT;

/// INVPROP configuration: output constraint backward propagation.
///
/// INVPROP propagates output specification constraints backward through the network
/// to tighten intermediate bounds BEFORE branch-and-bound begins. This breaks the
/// chicken-and-egg problem where bounds are too loose initially for cuts to be effective.
///
/// NOT YET EFFECTIVE: the dual variables ("gammas") that drive the tightening are
/// allocated and carried through the backward pass, but no optimization step for
/// them exists yet. They stay at their zero initialization, so enabling INVPROP
/// currently does not tighten any bounds.
///
/// Reference: Kotha et al., "Provably Computing the Preimage of Deep Neural Networks",
/// arXiv:2302.01404 (NeurIPS 2023)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvpropConfig {
    /// Whether INVPROP is enabled.
    ///
    /// When false, output constraints are not used for bound tightening.
    #[serde(default)]
    pub enabled: bool,

    /// Layer names/types to apply output constraints to.
    ///
    /// Examples:
    /// - `["BoundLinear"]` - apply to all linear layers
    /// - `["/input.7"]` - apply to specific node by name
    /// - `["all"]` - apply to all layers
    ///
    /// Follows alpha,beta-CROWN convention: node names starting with `/` are matched as
    /// prefix; type names starting with `Bound` are matched by layer type prefix.
    pub apply_output_constraints_to: Vec<String>,

    /// Tighten input bounds using output constraints.
    ///
    /// When enabled, after INVPROP optimization, update `x_L` and `x_U` from the
    /// constraint-aware linear bounds and propagate tighter bounds to subsequent phases.
    pub tighten_input_bounds: bool,

    /// Compute bounds both with and without output constraints, use best.
    ///
    /// More expensive (2x bound computation) but produces tighter results when
    /// output constraints make some bounds looser.
    pub best_of_oc_and_no_oc: bool,

    /// Layer names for direct bound optimization (disables IBP for these layers).
    ///
    /// For layers in this list, bounds are computed via direct optimization rather
    /// than interval bound propagation. More expensive but can produce tighter bounds.
    pub directly_optimize: Vec<String>,

    /// Share gammas across neurons to reduce memory.
    ///
    /// When false (default): allocate `(2, num_constraints, num_neurons)` per layer
    /// When true: allocate `(2, num_constraints, 1)` and broadcast across neurons
    ///
    /// Shared gammas reduce memory but may produce slightly looser bounds.
    pub share_gammas: bool,

    /// Also allocate per-layer intermediate-bound gammas (the general INVPROP
    /// top-down channel), in addition to the always-allocated output-seed duals.
    ///
    /// Default `false` = **output-node-only** (the shipped, adversarially-verified
    /// assume-violation channel: dualize the violation region at the output seed).
    /// Set `true` only for the research per-layer / split-lifting path once its
    /// off-seed fold is wired and oracle-gated.
    #[serde(default)]
    pub per_layer_gammas: bool,

    /// Optimize the gammas via projected Adam ascent during alpha-CROWN.
    ///
    /// Default `false`: gammas stay at their zero initialization, so the seed fold
    /// is the identity map and INVPROP is byte-identical to the baseline (inert
    /// until opted in). Set `true` to run the gamma ascent (Stage 2).
    #[serde(default)]
    pub optimize_gammas: bool,

    /// Learning rate for the gamma projected-Adam ascent (when `optimize_gammas`).
    #[serde(default = "default_gamma_lr")]
    pub gamma_lr: f32,
}

fn default_gamma_lr() -> f32 {
    0.5
}

impl InvpropConfig {
    /// Create an INVPROP config that applies to all layers.
    pub fn all_layers() -> Self {
        Self {
            enabled: true,
            apply_output_constraints_to: vec!["all".to_string()],
            ..Default::default()
        }
    }

    /// Check if a layer should have output constraints applied.
    ///
    /// Matches alpha,beta-CROWN semantics:
    /// - Names starting with `/` are matched as prefix (layer_name starts with pattern)
    /// - Names starting with `Bound` are matched by layer type prefix
    /// - `"all"` matches all layers
    ///
    /// Returns `false` if INVPROP is not enabled.
    pub fn should_apply_to(&self, layer_name: &str, layer_type: &str) -> bool {
        if !self.enabled {
            return false;
        }
        for pattern in &self.apply_output_constraints_to {
            if pattern == "all" {
                return true;
            }
            // Match by node name (starts with /)
            if pattern.starts_with('/') && layer_name.starts_with(pattern) {
                return true;
            }
            // Match by layer type (starts with Bound)
            if pattern.starts_with("Bound") && layer_type.starts_with(pattern) {
                return true;
            }
            // Also allow matching exact layer type for ny types
            if layer_type == pattern {
                return true;
            }
        }
        for pattern in &self.directly_optimize {
            if pattern.starts_with('/') {
                if layer_name.starts_with(pattern) {
                    return true;
                }
            } else if layer_name == pattern {
                return true;
            }
        }
        false
    }

    /// Whether to apply INVPROP to the input bounds for tightening.
    ///
    /// This only takes effect when `tighten_input_bounds` is true and
    /// the input is explicitly included in `apply_output_constraints_to`
    /// or `directly_optimize`.
    pub fn should_apply_to_input(&self) -> bool {
        if !self.enabled || !self.tighten_input_bounds {
            return false;
        }
        self.apply_output_constraints_to
            .iter()
            .chain(self.directly_optimize.iter())
            .any(|pattern| Self::input_pattern_matches(pattern))
    }

    fn input_pattern_matches(pattern: &str) -> bool {
        if pattern == "all" {
            return true;
        }
        if pattern == "BoundInput" || pattern == NETWORK_INPUT || pattern == "input" {
            return true;
        }
        if pattern.starts_with('/') {
            return "/input".starts_with(pattern) || "/x".starts_with(pattern);
        }
        false
    }
}
