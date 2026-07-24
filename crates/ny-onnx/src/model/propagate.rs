// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential network conversion delegate.
//!
//! The core logic now lives in [`ny_build::build_propagate_network`] (#1752).
//! This module keeps the `OnnxModel` method surface for backwards compatibility.

use ny_build::build_propagate_network;
use ny_core::{NyError, Result};
use ny_propagate::Network as PropNetwork;

use super::{OnnxModel, PropagateNetworkOptions};

impl OnnxModel {
    /// Convert to ny-propagate Network for IBP/CROWN propagation.
    ///
    /// This extracts layers and weights and builds a `ny_propagate::Network`.
    /// Returns an error if a Reshape has a dynamic (non-constant) shape.
    pub fn to_propagate_network(&self) -> Result<PropNetwork> {
        self.to_propagate_network_with_options(PropagateNetworkOptions::default())
    }

    /// Convert to ny-propagate Network with explicit conversion options.
    ///
    /// By default this returns an error if a Reshape has a dynamic (non-constant) shape.
    /// Set `allow_dynamic_reshape` to true to explicitly skip such Reshape ops.
    pub fn to_propagate_network_with_options(
        &self,
        options: PropagateNetworkOptions,
    ) -> Result<PropNetwork> {
        let ctx = self.convert_context();
        let network = build_propagate_network(&self.network.layers, &ctx, &options)?;
        // Fail closed on constant-only models: when every layer spec was
        // skipped as a constant / shape computation, the resulting empty
        // sequential network would act as the IDENTITY function, silently
        // replacing the model's constant output with its input. Such a model
        // has no activation path for the sequential lane to represent.
        // Permissive mode (allow_dynamic_reshape) is an explicitly requested
        // best-effort lossy conversion whose callers assert on skip behavior,
        // so the guard only applies to the default strict mode.
        if !options.allow_dynamic_reshape
            && network.layers().is_empty()
            && !self.network.layers.is_empty()
        {
            return Err(NyError::ModelLoad(format!(
                "sequential conversion produced an empty network: all {} layer(s) are \
                 constant or shape computations with no runtime activation path \
                 (an empty sequential network would behave as identity, not as the \
                 model's constant output)",
                self.network.layers.len()
            )));
        }
        Ok(network)
    }
}
