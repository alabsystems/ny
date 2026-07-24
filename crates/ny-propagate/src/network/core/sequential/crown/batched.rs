// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched N-D CROWN propagation for transformer-shaped networks.
//!
//! Extracted from `crown.rs` as part of #4233 Packet D.

use crate::bounds::BatchedLinearBounds;
use crate::layers::Layer;
use crate::network::core::Network;
use crate::network::crown_memory::check_batched_identity_budget;
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::{debug, instrument};

use super::tighten::tighten_crown_output;

impl Network {
    /// Propagate bounds through the network using batched CROWN.
    ///
    /// This version preserves N-D shape structure (e.g., [batch, seq, hidden]) throughout
    /// propagation, unlike regular CROWN which flattens to 1D. This is essential for
    /// transformer verification where position-wise operations need to maintain structure.
    ///
    /// Algorithm:
    /// 1. Run IBP forward to collect pre-activation bounds
    /// 2. Initialize batched linear bounds at output: A = I, b = 0 per position
    /// 3. Propagate backward through each layer using batched operations
    /// 4. Concretize final linear bounds using input bounds
    /// 5. Intersect with IBP forward bounds to ensure output is at least as tight as IBP (#2990)
    ///
    /// Currently supports: Linear, ReLU, GELU, Softmax, LayerNorm, Conv1d/ConvTranspose1d,
    /// Conv2d/ConvTranspose2d, Flatten, Reshape, Squeeze, Unsqueeze, Transpose,
    /// AddConstant, SubConstant, MulConstant, DivConstant.
    /// If Conv2d/ConvTranspose2d are present, spatial dims are flattened into
    /// a single feature dim and LayerNorm/Softmax are not supported.
    /// Other layers fall back to regular CROWN.
    #[inline]
    #[instrument(skip(self, input), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_batched_with_engine(input, None)
    }

    /// Propagate bounds using batched CROWN with an optional GPU GEMM engine.
    ///
    /// Same as `propagate_crown_batched` but threads `engine` through Conv2d
    /// layers for GPU-accelerated GEMM dispatch (#3399).
    #[instrument(skip(self, input, engine), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }

        let has_conv2d = self
            .layers
            .iter()
            .any(|layer| matches!(layer, Layer::Conv2d(_) | Layer::ConvTranspose2d(_)));

        // Check if we can use batched CROWN. Delegates to Layer::supports_batched_crown()
        // which is the single source of truth for the allow-list (#1753).
        // Conv2d networks exclude Softmax/LayerNorm (spatial flattening incompatible).
        let can_use_batched = if has_conv2d {
            self.layers
                .iter()
                .all(|layer| layer.supports_batched_crown_with_conv2d())
        } else {
            self.layers
                .iter()
                .all(|layer| layer.supports_batched_crown())
        };

        if !can_use_batched {
            debug!("Batched CROWN: Falling back to regular CROWN (unsupported layers)");
            return self.propagate_crown_with_engine(input, engine);
        }

        // Step 1: Run IBP forward to collect bounds at each layer
        // Note: Using IBP instead of CROWN-IBP for batched mode due to shape complexities
        // with multi-dimensional tensors. The regular propagate_crown() uses CROWN-IBP.
        let layer_bounds = self.collect_ibp_bounds(input)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_shape = output_bounds.shape().to_vec();
        let batched_output_shape = if has_conv2d {
            if output_shape.len() < 3 {
                output_shape.clone()
            } else {
                let mut flat_shape = output_shape[..output_shape.len() - 3].to_vec();
                let flat_dim = checked_shape_product(&output_shape[output_shape.len() - 3..])
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Network CROWN: Conv2d output spatial dims overflow: {:?}",
                            &output_shape[output_shape.len() - 3..]
                        ))
                    })?;
                flat_shape.push(flat_dim);
                flat_shape
            }
        } else {
            output_shape.clone()
        };

        debug!(
            "Batched CROWN: Starting backward propagation with shape {:?}",
            batched_output_shape
        );

        // Step 2: Initialize batched linear bounds at output
        // A = I (identity) for each position, b = 0
        // Guard: check CPU dense budget before allocating (#3550).
        if let Err(e) =
            check_batched_identity_budget("Network::propagate_crown_batched", &batched_output_shape)
        {
            debug!("Batched CROWN: {}, falling back to regular CROWN", e);
            return self.propagate_crown_with_engine(input, engine);
        }
        let mut batched_bounds = BatchedLinearBounds::identity(&batched_output_shape)?;

        // Step 3: Propagate backward through each layer
        for (i, layer) in self.layers.iter().enumerate().rev() {
            debug!(
                "Batched CROWN: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            // Get pre-activation bounds
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            batched_bounds = match layer.propagate_crown_backward_batched(
                &batched_bounds,
                Some(pre_activation),
                engine,
            ) {
                Ok(next) => next,
                // #3131: Catch both UnsupportedOp and UnsupportedConfiguration.
                // #3795: DeadlineExceeded also falls back.
                Err(
                    NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::DeadlineExceeded(_),
                ) => {
                    // Should not reach here due to earlier check.
                    debug!("Batched CROWN: Unsupported layer, falling back to regular CROWN");
                    return self.propagate_crown_with_engine(input, engine);
                }
                Err(err) => return Err(err),
            };
        }

        // Step 4: Concretize using input bounds
        debug!("Batched CROWN: Concretizing linear bounds with input");
        let input_for_concretize = if input.shape() == batched_bounds.input_shape.as_slice() {
            Cow::Borrowed(input)
        } else {
            Cow::Owned(input.reshape(&batched_bounds.input_shape)?)
        };
        let concrete = batched_bounds.concretize_sound(input_for_concretize.as_ref())?;

        // Output should match the expected output shape
        let final_output = if concrete.shape() != output_shape.as_slice() {
            concrete.reshape(&output_shape)?
        } else {
            concrete
        };

        // Step 5+6: Degrade check + forward-bound tightening (#3043 dedup).
        tighten_crown_output(final_output, output_bounds, "Batched CROWN")
    }
}
