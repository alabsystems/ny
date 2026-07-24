// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{CheckpointedBounds, StreamingConfig};
use crate::bounds::{BatchedLinearBounds, LinearBounds};
use crate::layers::{BoundPropagation, Layer};
use crate::network::crown_memory::check_batched_identity_budget;
use crate::network::{tighten_crown_output, Network};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

/// Streaming verifier that uses gradient checkpointing for memory efficiency.
pub struct StreamingVerifier {
    config: StreamingConfig,
}

impl StreamingVerifier {
    /// Create a new streaming verifier with the given config.
    pub fn new(config: StreamingConfig) -> Self {
        Self { config }
    }

    /// Run forward IBP pass with checkpointing.
    /// Returns checkpointed bounds for use in backward pass.
    pub fn collect_checkpointed_bounds(
        &self,
        network: &Network,
        input: &BoundedTensor,
    ) -> Result<CheckpointedBounds> {
        let num_layers = network.layers.len();

        // Create checkpointed storage with appropriate storage mode
        let mut checkpointed = if self.config.use_f16_checkpoints {
            CheckpointedBounds::new_compressed(
                input.clone(),
                num_layers,
                self.config.f16_widening_epsilon,
            )
        } else {
            CheckpointedBounds::new(input.clone(), num_layers)
        };

        if num_layers == 0 {
            return Ok(checkpointed);
        }

        let mut current = input.clone();
        let interval = self.config.checkpoint_interval.max(1);

        for (i, layer) in network.layers.iter().enumerate() {
            current = layer
                .propagate_ibp(&current)
                .map_err(|e| NyError::LayerError {
                    layer_index: i,
                    layer_type: layer.layer_type().to_string(),
                    source: Box::new(e),
                })?;

            // Store checkpoint at intervals and always at the last layer
            if (i + 1) % interval == 0 || i == num_layers - 1 {
                debug!(
                    "Streaming: checkpoint at layer {} (size {}, f16={})",
                    i,
                    current.len(),
                    self.config.use_f16_checkpoints
                );
                checkpointed.add_checkpoint(i, current.clone());
            }
        }

        let storage_type = if checkpointed.is_compressed() {
            "f16"
        } else {
            "f32"
        };
        debug!(
            "Streaming: {} checkpoints, {} bytes ({})",
            checkpointed.num_checkpoints(),
            checkpointed.memory_bytes(),
            storage_type
        );

        // Log compression stats if using f16
        if let Some((f16_bytes, f32_bytes, ratio)) = checkpointed.compression_stats() {
            debug!(
                "Streaming f16: {} bytes vs {} bytes f32 ({:.1}% of original)",
                f16_bytes,
                f32_bytes,
                ratio * 100.0
            );
        }

        Ok(checkpointed)
    }

    /// Propagate CROWN with gradient checkpointing.
    ///
    /// This is memory-efficient but slower than regular CROWN due to recomputation.
    /// Memory: O(L/K * N) instead of O(L * N).
    /// Compute: O(L * K) instead of O(L).
    ///
    /// Uses `concretize_sound()` for directed rounding (#2239).
    pub fn propagate_crown_streaming(
        &self,
        network: &Network,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.streaming_crown_backward(network, input, None)
    }

    /// Propagate CROWN with gradient checkpointing and GPU engine (#3598).
    pub fn propagate_crown_streaming_with_engine(
        &self,
        network: &Network,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.streaming_crown_backward(network, input, engine)
    }

    /// Core streaming CROWN implementation with sound directed rounding.
    ///
    /// Always uses `concretize_sound()` for f64→f32 directed rounding (#2239).
    fn streaming_crown_backward(
        &self,
        network: &Network,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if network.layers.is_empty() {
            return Ok(input.clone());
        }

        // Step 1: Forward pass with checkpointing
        let checkpointed = self.collect_checkpointed_bounds(network, input)?;

        // Get output bounds from last checkpoint
        let output_bounds = checkpointed
            .last_checkpoint()?
            .ok_or_else(|| NyError::InvalidSpec("No checkpoints created".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!("CROWN streaming: backward from {} outputs", output_dim);

        // Step 2: Initialize linear bounds at output
        let mut linear_bounds = LinearBounds::identity(output_dim);

        // Step 3: Backward pass with recomputation from checkpoints
        for i in (0..network.layers.len()).rev() {
            let layer = &network.layers[i];

            debug!(
                "CROWN streaming: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            // Get pre-activation bounds (recompute from checkpoint if needed)
            let pre_activation = if i == 0 {
                input.clone()
            } else {
                checkpointed.bounds_at(i - 1, network)?
            };

            linear_bounds = Self::propagate_layer_backward_with_engine(
                layer,
                linear_bounds,
                &pre_activation,
                i,
                engine,
            )?;
        }

        // Step 4: Concretize using input bounds with directed rounding (#2239).
        // concretize_sound() guarantees no NaN/inversion (#2287).
        debug!("CROWN streaming: concretizing (sound)");
        let crown_output = linear_bounds
            .concretize_sound(input)
            .reshape(&output_shape)?;

        // Step 5+6: Degrade check + forward-bound tightening (#3043 dedup).
        tighten_crown_output(crown_output, &output_bounds, "CROWN streaming")
    }

    /// Propagate backward through a single layer with optional engine (#3598).
    ///
    /// For Linear/Conv layers, dispatches to engine-aware methods for GPU
    /// acceleration. All other layers use the standard trait method.
    fn propagate_layer_backward_with_engine(
        layer: &Layer,
        linear_bounds: LinearBounds,
        pre_activation: &BoundedTensor,
        layer_idx: usize,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<LinearBounds> {
        use std::borrow::Cow;

        let result: Result<Cow<'_, LinearBounds>> = match layer {
            // GPU-accelerated paths for GEMM-heavy layers (#3598)
            Layer::Linear(l) => l.propagate_linear_with_engine(&linear_bounds, engine),
            Layer::Conv1d(c) => {
                let mut conv = c.clone();
                let shape = pre_activation.shape();
                if shape.len() >= 2 {
                    conv.set_input_length(shape[shape.len() - 1]);
                }
                conv.propagate_linear_with_engine(&linear_bounds, engine)
            }
            Layer::ConvTranspose1d(c) => {
                let mut conv = c.clone();
                let shape = pre_activation.shape();
                if shape.len() >= 2 {
                    conv.set_input_length(shape[shape.len() - 1]);
                }
                conv.propagate_linear_with_engine(&linear_bounds, engine)
            }
            Layer::Conv2d(c) => c.propagate_linear_with_engine(&linear_bounds, engine),
            // All other layers: standard trait method (no GEMM benefit)
            _ => {
                return layer
                    .propagate_crown_backward(&linear_bounds, Some(pre_activation))
                    .map_err(|source| NyError::LayerError {
                        layer_index: layer_idx,
                        layer_type: layer.layer_type().to_string(),
                        source: Box::new(source),
                    });
            }
        };

        result
            .map(|cow| cow.into_owned())
            .map_err(|source| NyError::LayerError {
                layer_index: layer_idx,
                layer_type: layer.layer_type().to_string(),
                source: Box::new(source),
            })
    }

    /// Propagate batched CROWN with gradient checkpointing.
    ///
    /// Batched version that preserves N-D shape structure.
    /// Uses `concretize_sound()` for directed rounding (#2239).
    pub fn propagate_crown_batched_streaming(
        &self,
        network: &Network,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.batched_streaming_crown_backward(network, input, None)
    }

    /// Propagate batched CROWN with gradient checkpointing and GPU engine (#3598).
    pub fn propagate_crown_batched_streaming_with_engine(
        &self,
        network: &Network,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.batched_streaming_crown_backward(network, input, engine)
    }

    /// Core batched streaming CROWN implementation with sound directed rounding.
    ///
    /// Always uses `concretize_sound()` for f64→f32 directed rounding (#2239).
    fn batched_streaming_crown_backward(
        &self,
        network: &Network,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if network.layers.is_empty() {
            return Ok(input.clone());
        }

        // Check if we can use batched CROWN. Delegates to Layer::supports_batched_crown()
        // which is the single source of truth for the allow-list (#1753).
        // Streaming verifier doesn't support Conv/shape-transform layers (no spatial handling).
        let can_use_batched = network
            .layers
            .iter()
            .all(|layer| layer.supports_batched_crown());

        if !can_use_batched {
            debug!("Streaming batched CROWN: unsupported layers, using regular streaming CROWN");
            return self.streaming_crown_backward(network, input, engine);
        }

        // Step 1: Forward pass with checkpointing
        let checkpointed = self.collect_checkpointed_bounds(network, input)?;

        let output_bounds = checkpointed
            .last_checkpoint()?
            .ok_or_else(|| NyError::InvalidSpec("No checkpoints created".to_string()))?;
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "Batched CROWN streaming: backward with shape {:?}",
            output_shape
        );

        // Step 2: Initialize batched linear bounds
        // Guard: check CPU dense budget before allocating (#3550).
        if let Err(e) = check_batched_identity_budget(
            "StreamingVerifier::batched_streaming_crown_backward",
            &output_shape,
        ) {
            debug!(
                "Batched streaming CROWN: {}, falling back to regular streaming CROWN",
                e
            );
            return self.streaming_crown_backward(network, input, engine);
        }
        let mut batched_bounds = BatchedLinearBounds::identity(&output_shape)?;

        // Step 3: Backward pass with recomputation
        for i in (0..network.layers.len()).rev() {
            let layer = &network.layers[i];

            debug!(
                "Batched CROWN streaming: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            let pre_activation = if i == 0 {
                input.clone()
            } else {
                checkpointed.bounds_at(i - 1, network)?
            };

            batched_bounds = match layer
                // #3598: Thread GemmEngine through streaming verifier.
                .propagate_crown_backward_batched(&batched_bounds, Some(&pre_activation), engine)
            {
                Ok(next) => next,
                // Batched support can still be shape- or config-conditional at runtime.
                // Mirror the sequential batched path and fall back to regular streaming
                // CROWN instead of surfacing a hard layer error.
                Err(
                    NyError::UnsupportedOp(reason)
                    | NyError::UnsupportedConfiguration(reason)
                    | NyError::NumericalInstability(reason),
                ) => {
                    debug!(
                        "Streaming batched CROWN: layer {} ({}) unsupported/unstable ({}), falling back to regular streaming CROWN",
                        i,
                        layer.layer_type(),
                        reason
                    );
                    return self.streaming_crown_backward(network, input, engine);
                }
                Err(source) => {
                    return Err(NyError::LayerError {
                        layer_index: i,
                        layer_type: layer.layer_type().to_string(),
                        source: Box::new(source),
                    });
                }
            };
        }

        // Step 4: Concretize with directed rounding (#2239).
        // concretize_sound() guarantees no NaN/inversion (#2287).
        let crown_output = batched_bounds.concretize_sound(input)?;
        let crown_output = if crown_output.shape() != output_shape.as_slice() {
            crown_output.reshape(&output_shape)?
        } else {
            crown_output
        };

        // Step 5+6: Degrade check + forward-bound tightening (#3043 dedup).
        tighten_crown_output(crown_output, &output_bounds, "Batched CROWN streaming")
    }
}
