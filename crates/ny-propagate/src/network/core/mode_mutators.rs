// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared layer-mode mutator helpers.
//!
//! Each function accepts an iterator of mutable `Layer` references and applies
//! a single mode toggle, returning the count of layers modified. Both `Network`
//! (via `self.layers.iter_mut()`) and `GraphNetwork` (via
//! `self.nodes.values_mut().map(|n| &mut n.layer)`) delegate to these helpers
//! so that per-layer mutation logic is defined exactly once.

use crate::layers::{Layer, LayerNormCrownMode, LayerNormMode};

/// Set `forward_mode` on all LayerNorm, RmsNorm, InstanceNorm1d, AdaIN1d, and GroupNorm layers.
pub(crate) fn set_layernorm_forward_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    enabled: bool,
) -> usize {
    let mut count = 0;
    for layer in layers {
        match layer {
            Layer::LayerNorm(ref mut ln) => {
                ln.forward_mode = enabled;
                count += 1;
            }
            Layer::RmsNorm(ref mut rn) => {
                rn.forward_mode = enabled;
                count += 1;
            }
            Layer::InstanceNorm1d(ref mut inst) => {
                inst.forward_mode = enabled;
                count += 1;
            }
            Layer::AdaIN1d(ref mut adain) => {
                adain.instance_norm.forward_mode = enabled;
                count += 1;
            }
            Layer::GroupNorm(ref mut gn) => {
                gn.forward_mode = enabled;
                count += 1;
            }
            _ => {}
        }
    }
    count
}

/// Set the CROWN linearization mode on all LayerNorm, RmsNorm, InstanceNorm1d, AdaIN1d, and GroupNorm layers.
pub(crate) fn set_layernorm_crown_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    mode: LayerNormCrownMode,
) -> usize {
    let mut count = 0;
    for layer in layers {
        match layer {
            Layer::LayerNorm(ref mut ln) => {
                ln.crown_mode = mode;
                count += 1;
            }
            Layer::RmsNorm(ref mut rn) => {
                rn.crown_mode = mode;
                count += 1;
            }
            Layer::InstanceNorm1d(ref mut inst) => {
                inst.crown_mode = mode;
                count += 1;
            }
            Layer::AdaIN1d(ref mut adain) => {
                adain.instance_norm.crown_mode = mode;
                count += 1;
            }
            Layer::GroupNorm(ref mut gn) => {
                gn.crown_mode = mode;
                count += 1;
            }
            _ => {}
        }
    }
    count
}

/// Set the normalization mode on all LayerNorm layers.
pub(crate) fn set_layernorm_norm_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    mode: LayerNormMode,
) -> usize {
    let mut count = 0;
    for layer in layers {
        if let Layer::LayerNorm(ref mut ln) = layer {
            ln.mode = mode;
            count += 1;
        }
    }
    count
}

/// Set `sound` mode on all GELU layers.
pub(crate) fn set_gelu_sound_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    enabled: bool,
) -> usize {
    let mut count = 0;
    for layer in layers {
        if let Layer::GELU(ref mut gelu) = layer {
            gelu.sound = enabled;
            count += 1;
        }
    }
    count
}

/// Set `sound` mode on all LogSoftmax layers.
pub(crate) fn set_logsoftmax_sound_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    enabled: bool,
) -> usize {
    let mut count = 0;
    for layer in layers {
        if let Layer::LogSoftmax(ref mut logsoftmax) = layer {
            logsoftmax.sound = enabled;
            count += 1;
        }
    }
    count
}

/// Set `sound` mode on all Softmax layers.
pub(crate) fn set_softmax_sound_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    enabled: bool,
) -> usize {
    let mut count = 0;
    for layer in layers {
        if let Layer::Softmax(ref mut softmax) = layer {
            softmax.sound = enabled;
            count += 1;
        }
    }
    count
}

/// Set `sound` mode on all CausalSoftmax layers.
pub(crate) fn set_causal_softmax_sound_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    enabled: bool,
) -> usize {
    let mut count = 0;
    for layer in layers {
        if let Layer::CausalSoftmax(ref mut softmax) = layer {
            softmax.sound = enabled;
            count += 1;
        }
    }
    count
}

/// Set `sound` mode on all Sin layers.
pub(crate) fn set_sin_sound_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    enabled: bool,
) -> usize {
    let mut count = 0;
    for layer in layers {
        if let Layer::Sin(ref mut sin) = layer {
            sin.sound = enabled;
            count += 1;
        }
    }
    count
}

/// Set `sound` mode on all Cos layers.
pub(crate) fn set_cos_sound_mode<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    enabled: bool,
) -> usize {
    let mut count = 0;
    for layer in layers {
        if let Layer::Cos(ref mut cos) = layer {
            cos.sound = enabled;
            count += 1;
        }
    }
    count
}
