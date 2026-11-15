// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Layer configuration methods for GraphNetwork.
//!
//! Each mutator delegates to the shared helpers in [`super::super::mode_mutators`]
//! so that per-layer mutation logic is defined exactly once.

use crate::layers::{Layer, LayerNormCrownMode, LayerNormMode};

use super::super::mode_mutators;
use super::GraphNetwork;

impl GraphNetwork {
    /// Enable the forward-linear spec-alpha surrogate for this graph.
    ///
    /// The production CLI only sets this from the typed preset key after its
    /// loader admission check has sealed the authored FLOAT initializers and
    /// required raw BatchNormalization nodes. The gate is graph-local and
    /// default off; it authorizes only the existing slope-selection heuristic.
    /// Every published bound still comes from the certified alpha-fed rebuild.
    /// Changing the policy drops every forward-linear cache identity, including
    /// a memoized optimizer refusal.
    pub fn set_forward_linear_spec_alpha(&mut self, enabled: bool) {
        if self.forward_linear_spec_alpha != enabled {
            self.forward_linear_spec_alpha = enabled;
            self.invalidate_forward_linear_cache();
        }
    }

    /// Whether this graph admits the typed forward-linear spec-alpha surrogate.
    #[must_use]
    pub fn forward_linear_spec_alpha_enabled(&self) -> bool {
        self.forward_linear_spec_alpha
    }

    /// #fl-seq-no-transpose: admit the sequential forward-linear intermediate
    /// reference on a Conv2d graph with NO ConvTranspose2d.
    ///
    /// Like `set_forward_linear_spec_alpha`, changing this drops every
    /// forward-linear cache identity — the admission decides WHICH reference
    /// bounds exist, so an entry collected under one policy must never be served
    /// under the other.
    ///
    /// Sound by the admitted clause's own argument: a forward-linear reference
    /// is an enclosure either way, and admitting it can only tighten. The
    /// ConvTranspose requirement it relaxes is scoping (avoiding shapes where
    /// the collector may fail closed), not correctness. Default off.
    pub fn set_forward_linear_sequential_without_conv_transpose(&mut self, enabled: bool) {
        if self.forward_linear_sequential_without_conv_transpose != enabled {
            self.forward_linear_sequential_without_conv_transpose = enabled;
            self.invalidate_forward_linear_cache();
        }
    }

    /// `#fl-s-shaped-untyped`: admit Tanh/Sigmoid on the ORDINARY forward-linear
    /// surface so `spec-CROWN` stops falling back fail-closed on graphs that
    /// carry them.
    ///
    /// Drops every forward-linear cache identity for the same reason as the
    /// sibling above: admission decides WHICH reference bounds exist, so an entry
    /// collected under one policy must never be served under the other.
    ///
    /// ⚠ Unlike that sibling, this is NOT sound-by-a-tightening-argument. The
    /// path it replaces is per-node CROWN-IBP and neither enclosure dominates, so
    /// the SIGN is unmeasured — it can loosen. Default off; measure before
    /// arming anything.
    pub fn set_forward_linear_s_shaped_untyped(&mut self, enabled: bool) {
        if self.forward_linear_s_shaped_untyped != enabled {
            self.forward_linear_s_shaped_untyped = enabled;
            self.invalidate_forward_linear_cache();
        }
    }

    /// Whether this graph admits S-shaped nodes on the untyped forward-linear
    /// surface (`#fl-s-shaped-untyped`).
    #[must_use]
    pub fn forward_linear_s_shaped_untyped_enabled(&self) -> bool {
        self.forward_linear_s_shaped_untyped
    }

    /// Whether this graph admits the sequential Conv2d-only forward-linear
    /// reference (no ConvTranspose2d present).
    #[must_use]
    pub fn forward_linear_sequential_without_conv_transpose_enabled(&self) -> bool {
        self.forward_linear_sequential_without_conv_transpose
    }

    /// Backward-compatible alias for the original cGAN-named admission API.
    ///
    /// This deliberately writes the same generic graph-local bit; it does not
    /// create a second gate or widen the optimizer's authority.
    pub fn set_cgan_forward_alpha_surrogate(&mut self, enabled: bool) {
        self.set_forward_linear_spec_alpha(enabled);
    }

    /// Backward-compatible alias for the original cGAN-named admission API.
    #[must_use]
    pub fn cgan_forward_alpha_surrogate_enabled(&self) -> bool {
        self.forward_linear_spec_alpha_enabled()
    }

    /// Enable or disable forward mode for all normalization nodes that share
    /// `LayerNorm`-style statistics in the graph.
    ///
    /// Forward mode uses the center point (midpoint of bounds) for mean/std
    /// computation, dramatically reducing bound explosion (up to 80x tighter
    /// bounds) but may not be perfectly sound for large perturbations.
    ///
    /// Affects `LayerNorm`, `RmsNorm`, `GroupNorm`, `InstanceNorm1d`, and
    /// `AdaIN1d`.
    ///
    /// Returns the number of normalization nodes modified.
    pub fn set_layernorm_forward_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_layernorm_forward_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            enabled,
        )
    }

    /// Create a copy of this graph with forward mode enabled for all
    /// normalization nodes that share `LayerNormCrownMode`.
    #[deprecated(
        since = "0.1.0",
        note = "use set_layernorm_forward_mode() instead — it returns the count of nodes modified"
    )]
    pub fn with_layernorm_forward_mode(mut self, enabled: bool) -> Self {
        self.set_layernorm_forward_mode(enabled);
        self
    }

    /// Set the shared normalization CROWN mode for all supported norm nodes in
    /// the graph.
    ///
    /// Affects `LayerNorm`, `RmsNorm`, `GroupNorm`, `InstanceNorm1d`, and
    /// `AdaIN1d`.
    ///
    /// - `IbpValidated` (layer default): Jacobian linearization with IBP-validated margins
    /// - `Sound`: Return error if CROWN linearization is attempted
    /// - `Cut`: Use identity relaxation (sound but loses correlations)
    /// - `Sampling`: Use heuristic sampling-based linearization (NOT provably sound)
    ///
    /// Returns the number of normalization nodes modified.
    pub fn set_layernorm_crown_mode(&mut self, mode: LayerNormCrownMode) -> usize {
        mode_mutators::set_layernorm_crown_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            mode,
        )
    }

    /// Set the normalization mode for all LayerNorm nodes in the graph.
    ///
    /// - `Standard` (default): Full LayerNorm (subtract mean, divide by std)
    /// - `MeanOnly`: DeepT-style LayerNorm (subtract mean only, no variance normalization)
    ///
    /// Returns the number of LayerNorm nodes modified.
    pub fn set_layernorm_norm_mode(&mut self, mode: LayerNormMode) -> usize {
        let changes_model = self
            .nodes
            .values()
            .any(|node| matches!(&node.layer, Layer::LayerNorm(layer) if layer.mode != mode));
        let count = mode_mutators::set_layernorm_norm_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            mode,
        );
        if changes_model {
            self.invalidate_forward_linear_cache();
            self.invalidate_exec_order_cache();
        }
        count
    }

    /// Create a copy of this graph with the specified LayerNorm normalization mode.
    #[deprecated(
        since = "0.1.0",
        note = "use set_layernorm_norm_mode() instead — it returns the count of nodes modified"
    )]
    pub fn with_layernorm_norm_mode(mut self, mode: LayerNormMode) -> Self {
        self.set_layernorm_norm_mode(mode);
        self
    }

    /// Enable or disable sound (no sampling) GELU relaxations for all GELU nodes.
    ///
    /// Returns the number of GELU nodes modified.
    pub fn set_gelu_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_gelu_sound_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            enabled,
        )
    }

    /// Create a copy of this graph with sound GELU relaxations enabled/disabled.
    #[deprecated(
        since = "0.1.0",
        note = "use set_gelu_sound_mode() instead — it returns the count of nodes modified"
    )]
    pub fn with_gelu_sound_mode(mut self, enabled: bool) -> Self {
        self.set_gelu_sound_mode(enabled);
        self
    }

    /// Enable or disable sound (no sampling) LogSoftmax relaxations for all LogSoftmax nodes.
    ///
    /// Returns the number of LogSoftmax nodes modified.
    pub fn set_logsoftmax_sound_mode(&mut self, enabled: bool) -> usize {
        self.invalidate_forward_linear_cache();
        mode_mutators::set_logsoftmax_sound_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            enabled,
        )
    }

    /// Create a copy of this graph with sound LogSoftmax relaxations enabled/disabled.
    #[deprecated(
        since = "0.1.0",
        note = "use set_logsoftmax_sound_mode() instead — it returns the count of nodes modified"
    )]
    pub fn with_logsoftmax_sound_mode(mut self, enabled: bool) -> Self {
        self.set_logsoftmax_sound_mode(enabled);
        self
    }

    /// Enable or disable sound (no sampling) Softmax relaxations for all Softmax nodes.
    ///
    /// Returns the number of Softmax nodes modified.
    pub fn set_softmax_sound_mode(&mut self, enabled: bool) -> usize {
        self.invalidate_forward_linear_cache();
        mode_mutators::set_softmax_sound_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            enabled,
        )
    }

    /// Create a copy of this graph with sound Softmax relaxations enabled/disabled.
    #[deprecated(
        since = "0.1.0",
        note = "use set_softmax_sound_mode() instead — it returns the count of nodes modified"
    )]
    pub fn with_softmax_sound_mode(mut self, enabled: bool) -> Self {
        self.set_softmax_sound_mode(enabled);
        self
    }

    /// Enable or disable sound (no sampling) CausalSoftmax relaxations for all CausalSoftmax nodes.
    ///
    /// Returns the number of CausalSoftmax nodes modified.
    pub fn set_causal_softmax_sound_mode(&mut self, enabled: bool) -> usize {
        self.invalidate_forward_linear_cache();
        mode_mutators::set_causal_softmax_sound_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            enabled,
        )
    }

    /// Create a copy of this graph with sound CausalSoftmax relaxations enabled/disabled.
    #[deprecated(
        since = "0.1.0",
        note = "use set_causal_softmax_sound_mode() instead — it returns the count of nodes modified"
    )]
    pub fn with_causal_softmax_sound_mode(mut self, enabled: bool) -> Self {
        self.set_causal_softmax_sound_mode(enabled);
        self
    }

    /// Enable or disable conservative (IBP-only) Sin relaxations for all Sin nodes.
    ///
    /// Returns the number of Sin nodes modified.
    pub fn set_sin_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_sin_sound_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            enabled,
        )
    }

    /// Create a copy of this graph with conservative Sin relaxations enabled/disabled.
    #[deprecated(
        since = "0.1.0",
        note = "use set_sin_sound_mode() instead — it returns the count of nodes modified"
    )]
    pub fn with_sin_sound_mode(mut self, enabled: bool) -> Self {
        self.set_sin_sound_mode(enabled);
        self
    }

    /// Enable or disable conservative (IBP-only) Cos relaxations for all Cos nodes.
    ///
    /// Returns the number of Cos nodes modified.
    pub fn set_cos_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_cos_sound_mode(
            self.nodes.values_mut().map(|node| &mut node.layer),
            enabled,
        )
    }

    /// Create a copy of this graph with conservative Cos relaxations enabled/disabled.
    #[deprecated(
        since = "0.1.0",
        note = "use set_cos_sound_mode() instead — it returns the count of nodes modified"
    )]
    pub fn with_cos_sound_mode(mut self, enabled: bool) -> Self {
        self.set_cos_sound_mode(enabled);
        self
    }
}
