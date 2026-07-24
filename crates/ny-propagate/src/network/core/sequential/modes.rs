// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Network configuration toggles for layer-specific modes.
//!
//! Each mutator delegates to the shared helpers in [`super::super::mode_mutators`]
//! so that per-layer mutation logic is defined exactly once.

use super::super::mode_mutators;
use super::Network;

impl Network {
    /// Enable or disable forward mode for all normalization layers that share
    /// `LayerNorm`-style statistics in the network.
    ///
    /// Affects `LayerNorm`, `RmsNorm`, `GroupNorm`, `InstanceNorm1d`, and
    /// `AdaIN1d`.
    ///
    /// Returns the number of normalization layers modified.
    pub fn set_layernorm_forward_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_layernorm_forward_mode(self.layers.iter_mut(), enabled)
    }

    /// Create a copy of this network with forward mode enabled for all
    /// normalization layers that share `LayerNormCrownMode`.
    #[deprecated(
        note = "use set_layernorm_forward_mode() instead — it returns the count of layers modified"
    )]
    pub fn with_layernorm_forward_mode(mut self, enabled: bool) -> Self {
        self.set_layernorm_forward_mode(enabled);
        self
    }

    /// Set the shared normalization CROWN mode for all supported norm layers in
    /// the network.
    ///
    /// Affects `LayerNorm`, `RmsNorm`, `GroupNorm`, `InstanceNorm1d`, and
    /// `AdaIN1d`.
    ///
    /// Returns the number of normalization layers modified.
    pub fn set_layernorm_crown_mode(&mut self, mode: crate::layers::LayerNormCrownMode) -> usize {
        mode_mutators::set_layernorm_crown_mode(self.layers.iter_mut(), mode)
    }

    /// Set the normalization mode for all LayerNorm layers in the network.
    ///
    /// - `Standard` (default): Full LayerNorm (subtract mean, divide by std)
    /// - `MeanOnly`: DeepT-style LayerNorm (subtract mean only, no variance normalization)
    ///
    /// Returns the number of LayerNorm layers modified.
    pub fn set_layernorm_norm_mode(&mut self, mode: crate::layers::LayerNormMode) -> usize {
        mode_mutators::set_layernorm_norm_mode(self.layers.iter_mut(), mode)
    }

    /// Create a copy of this network with the specified LayerNorm normalization mode.
    #[deprecated(
        note = "use set_layernorm_norm_mode() instead — it returns the count of layers modified"
    )]
    pub fn with_layernorm_norm_mode(mut self, mode: crate::layers::LayerNormMode) -> Self {
        self.set_layernorm_norm_mode(mode);
        self
    }

    /// Enable or disable sound (no sampling) GELU relaxations for all GELU layers.
    ///
    /// Returns the number of GELU layers modified.
    pub fn set_gelu_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_gelu_sound_mode(self.layers.iter_mut(), enabled)
    }

    /// Create a copy of this network with sound GELU relaxations enabled/disabled.
    #[deprecated(
        note = "use set_gelu_sound_mode() instead — it returns the count of layers modified"
    )]
    pub fn with_gelu_sound_mode(mut self, enabled: bool) -> Self {
        self.set_gelu_sound_mode(enabled);
        self
    }

    /// Enable or disable sound (no sampling) LogSoftmax relaxations for all LogSoftmax layers.
    ///
    /// Returns the number of LogSoftmax layers modified.
    pub fn set_logsoftmax_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_logsoftmax_sound_mode(self.layers.iter_mut(), enabled)
    }

    /// Create a copy of this network with sound LogSoftmax relaxations enabled/disabled.
    #[deprecated(
        note = "use set_logsoftmax_sound_mode() instead — it returns the count of layers modified"
    )]
    pub fn with_logsoftmax_sound_mode(mut self, enabled: bool) -> Self {
        self.set_logsoftmax_sound_mode(enabled);
        self
    }

    /// Enable or disable sound (no sampling) Softmax relaxations for all Softmax layers.
    ///
    /// Returns the number of Softmax layers modified.
    pub fn set_softmax_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_softmax_sound_mode(self.layers.iter_mut(), enabled)
    }

    /// Create a copy of this network with sound Softmax relaxations enabled/disabled.
    #[deprecated(
        note = "use set_softmax_sound_mode() instead — it returns the count of layers modified"
    )]
    pub fn with_softmax_sound_mode(mut self, enabled: bool) -> Self {
        self.set_softmax_sound_mode(enabled);
        self
    }

    /// Enable or disable sound (no sampling) CausalSoftmax relaxations for all CausalSoftmax layers.
    ///
    /// Returns the number of CausalSoftmax layers modified.
    pub fn set_causal_softmax_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_causal_softmax_sound_mode(self.layers.iter_mut(), enabled)
    }

    /// Create a copy of this network with sound CausalSoftmax relaxations enabled/disabled.
    #[deprecated(
        note = "use set_causal_softmax_sound_mode() instead — it returns the count of layers modified"
    )]
    pub fn with_causal_softmax_sound_mode(mut self, enabled: bool) -> Self {
        self.set_causal_softmax_sound_mode(enabled);
        self
    }

    /// Enable or disable conservative (IBP-only) Sin relaxations for all Sin layers.
    ///
    /// Returns the number of Sin layers modified.
    pub fn set_sin_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_sin_sound_mode(self.layers.iter_mut(), enabled)
    }

    /// Create a copy of this network with conservative Sin relaxations enabled/disabled.
    #[deprecated(
        note = "use set_sin_sound_mode() instead — it returns the count of layers modified"
    )]
    pub fn with_sin_sound_mode(mut self, enabled: bool) -> Self {
        self.set_sin_sound_mode(enabled);
        self
    }

    /// Enable or disable conservative (IBP-only) Cos relaxations for all Cos layers.
    ///
    /// Returns the number of Cos layers modified.
    pub fn set_cos_sound_mode(&mut self, enabled: bool) -> usize {
        mode_mutators::set_cos_sound_mode(self.layers.iter_mut(), enabled)
    }

    /// Create a copy of this network with conservative Cos relaxations enabled/disabled.
    #[deprecated(
        note = "use set_cos_sound_mode() instead — it returns the count of layers modified"
    )]
    pub fn with_cos_sound_mode(mut self, enabled: bool) -> Self {
        self.set_cos_sound_mode(enabled);
        self
    }
}
