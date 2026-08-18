// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential (layer-based) network representation.

pub(crate) mod crown;
mod modes;

pub(crate) use crown::extract_relu_gpu_layer_with_alpha;
pub(crate) use crown::materialize_terminal_crown_bounds_with_deadline;
pub(crate) use crown::tighten_crown_output;
pub(crate) use crown::tighten_crown_output_with_deadline;
pub(crate) use crown::tighten_crown_output_with_provenance_and_deadline;
pub(crate) use crown::try_extract_single_gpu_layer;
pub(crate) use crown::CrownStepFallback;
pub(crate) use crown::CrownStepResult;
pub(crate) use crown::{apply_bn_werr_to_host_relu, try_extract_batch_norm_conv1x1};
pub(crate) use crown::{
    crown_backward_step_patches, crown_backward_step_patches_spec_crown,
    crown_backward_step_patches_with_deadline_authority, SpecPatchesStepError,
};
pub(crate) use crown::{gpu_relu_affine_cell, GpuReluAffineVariant};

use crate::layers::Layer;
use crown::GpuCrownStaticCache;

/// A neural network represented as a sequence of layers.
pub struct Network {
    pub(crate) layers: Vec<Layer>,
    /// Cached static GPU CROWN layer data (#3397 plan cache Step 1).
    /// Populated on first GPU CROWN extraction; subsequent calls reuse
    /// Arc-shared weight data and only refresh dynamic activation entries.
    pub(crate) gpu_crown_cache: std::sync::Mutex<Option<GpuCrownStaticCache>>,
}

impl std::fmt::Debug for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Network")
            .field("layers", &self.layers)
            .field("gpu_crown_cache", &"<Mutex>")
            .finish()
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            // Don't clone the cache — it will be repopulated on first use.
            gpu_crown_cache: std::sync::Mutex::new(None),
        }
    }
}

impl Network {
    /// Drop static GPU CROWN descriptors after any model mutation.
    ///
    /// The cache owns cloned Linear/Conv parameters and topology. Returning a
    /// mutable layer view or changing the sequence without invalidating it
    /// could otherwise let a later authoritative GPU route bound the previous
    /// network. Replacing the mutex is safe under `&mut self` and also clears a
    /// prior poison state.
    fn invalidate_gpu_crown_cache(&mut self) {
        self.gpu_crown_cache = std::sync::Mutex::new(None);
    }

    /// Create an empty network.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            gpu_crown_cache: std::sync::Mutex::new(None),
        }
    }

    /// Add a layer to the network.
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
        self.invalidate_gpu_crown_cache();
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Immutable view of the layer list.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Mutable access to existing layers. Returns a slice to prevent
    /// structural changes (push/pop/clear) — use `add_layer()` for those. Any
    /// static GPU CROWN extraction is invalidated before mutable access is
    /// granted.
    pub fn layers_mut(&mut self) -> &mut [Layer] {
        self.invalidate_gpu_crown_cache();
        &mut self.layers
    }

    /// Consume the network and return its layer list.
    pub fn into_layers(self) -> Vec<Layer> {
        self.layers
    }

    pub(crate) fn has_self_attention(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(layer, Layer::SelfAttention(_)))
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}
