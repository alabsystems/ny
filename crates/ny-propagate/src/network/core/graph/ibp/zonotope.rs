// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Zonotope propagation helpers for graph networks.

use crate::layers::{Layer, LayerNormMode};

use ny_core::{NyError, Result};
use ny_tensor::{BoundedTensor, ZonotopeTensor};
use tracing::{debug, info};

use super::super::{GraphNetwork, GraphNode, NETWORK_INPUT};
use super::dispatch::{check_nan_firewall, dispatch_ibp_for_node};
use super::zonotope_matmul::propagate_disjoint_matmul;

/// How graph zonotope propagation should handle Softmax-family nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ZonotopeSoftmaxMode {
    /// Use the zonotope affine relaxation for Softmax-family nodes.
    #[default]
    Affine,
    /// Intentionally cut the zonotope pipeline at Softmax-family nodes and
    /// reuse the existing graph-level IBP fallback for that operator.
    IntervalFallback,
}

/// Operator-local controls for graph zonotope propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub struct ZonotopePropagationOptions {
    softmax_mode: ZonotopeSoftmaxMode,
}

impl ZonotopePropagationOptions {
    /// Default graph zonotope options.
    pub const fn new() -> Self {
        Self {
            softmax_mode: ZonotopeSoftmaxMode::Affine,
        }
    }

    /// Softmax-family handling mode for this propagation run.
    pub const fn softmax_mode(self) -> ZonotopeSoftmaxMode {
        self.softmax_mode
    }

    /// Override Softmax-family handling for this propagation run.
    pub const fn with_softmax_mode(mut self, softmax_mode: ZonotopeSoftmaxMode) -> Self {
        self.softmax_mode = softmax_mode;
        self
    }
}

impl Default for ZonotopePropagationOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphNetwork {
    /// Propagate bounds through the graph using zonotopes for correlation-aware attention.
    ///
    /// Zonotopes track correlations between Q and K through shared error symbols,
    /// giving tighter bounds for Q@K^T in attention than IBP.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (center values)
    /// * `epsilon` - Perturbation epsilon for input
    ///
    /// # Returns
    /// Bounds as a BoundedTensor (converted from zonotope at output).
    ///
    /// # Supported Operations
    /// - Linear layers (preserves zonotope form exactly)
    /// - AddConstant, MulConstant (preserves zonotope form exactly)
    /// - MatMul for Q@K^T patterns (uses matmul_transposed)
    /// - Other ops: falls back to IBP and converts to zonotope (loses correlations)
    ///
    /// # Limitations
    /// - Best effort for tensors with >=2 dims (..., seq, dim)
    /// - Larger inputs may require significant memory for error symbols
    #[inline]
    pub fn propagate_zonotope(&self, input: &BoundedTensor, epsilon: f32) -> Result<BoundedTensor> {
        self.propagate_zonotope_with_options(input, epsilon, ZonotopePropagationOptions::default())
    }

    /// Propagate bounds through the graph using zonotopes with operator-local
    /// fallback controls.
    #[inline]
    pub fn propagate_zonotope_with_options(
        &self,
        input: &BoundedTensor,
        _epsilon: f32,
        options: ZonotopePropagationOptions,
    ) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            // Empty graph: nothing to do.
            return Ok(input.clone());
        }

        // Get execution order
        let exec_order = self.exec_order()?;

        // Zonotope propagation works best on 2D, but we support batched sequence tensors too.
        let input_shape = input.shape();
        if input_shape.len() < 2 {
            debug!(
                "GraphNetwork zonotope: input shape {:?} has <2 dims, falling back to IBP",
                input_shape
            );
            return self.propagate_ibp(input);
        }

        // Reject non-finite input early: if a previous block produced NaN/Inf,
        // zonotope construction is not meaningful. Return a fallback-class error
        // so the caller (e.g. Whisper verifier) can recover via IBP. Part of #3548.
        if input.has_overflow() {
            return Err(NyError::UnsupportedConfiguration(
                "zonotope propagation requires finite input bounds".to_string(),
            ));
        }

        // Create input zonotope with per-position error symbols derived from the current bounds.
        // This ensures zonotope propagation remains compatible with compositional bounds where
        // per-element radii may not equal the original epsilon.
        let input_zonotope = ZonotopeTensor::from_bounded_tensor_per_position(input)?;
        debug!(
            "GraphNetwork zonotope: created input zonotope with {} error terms, shape {:?}",
            input_zonotope.n_error_terms(),
            input_zonotope.shape()
        );

        // Store zonotopes for each node's output
        let mut zonotope_cache: std::collections::HashMap<String, ZonotopeTensor> =
            std::collections::HashMap::new();

        // Also store interval bounds for fallback
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        // Process nodes in topological order
        for node_name in exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            debug!(
                "GraphNetwork zonotope: processing node {} ({})",
                node_name,
                node.layer.layer_type()
            );

            // Try to propagate as zonotope; fall back to IBP if not supported
            let result =
                self.propagate_zonotope_node(node, &input_zonotope, &zonotope_cache, options);

            match result {
                Ok(z) => {
                    debug!(
                        "GraphNetwork zonotope: node {} output zonotope with {} error terms, max_width {}",
                        node_name,
                        z.n_error_terms(),
                        z.max_width()
                    );
                    bounds_cache.insert(node_name.clone(), z.to_bounded_tensor()?);
                    zonotope_cache.insert(node_name.clone(), z);
                }
                // #3166, #3548: UnsupportedOp/UnsupportedConfiguration/NumericalInstability:
                // zonotope propagation not available or produced non-finite values for
                // this node. IBP fallback is sound — it is the conservative degradation
                // for optional tightening paths. Other errors (SoundnessRefusal, etc.)
                // must propagate (#3106).
                Err(e @ NyError::UnsupportedOp(_))
                | Err(e @ NyError::UnsupportedConfiguration(_))
                | Err(e @ NyError::NumericalInstability(_)) => {
                    debug!(
                        "GraphNetwork zonotope: node {} falling back to IBP: {}",
                        node_name, e
                    );

                    // Use unified dispatch for IBP fallback (#2405).
                    // This correctly handles Concat before is_binary().
                    let ibp_bounds = dispatch_ibp_for_node(node, node_name, &mut |name| {
                        Ok(self.bounds_ref(name, input, &bounds_cache)?.clone())
                    })?;

                    // Convert IBP bounds to zonotope (loses correlation info).
                    // For attention `softmax @ V`, this is the Packet C seam:
                    // non-transposed MatMul falls back to interval bounds, then
                    // the interval is reintroduced as a single shared-error
                    // zonotope for the remaining affine suffix.
                    let z = ZonotopeTensor::from_bounded_tensor(&ibp_bounds);
                    if matches!(&node.layer, Layer::MatMul(matmul) if !matmul.transpose_b) {
                        debug!(
                            "GraphNetwork zonotope: node {} non-transposed MatMul fallback width {} -> re-zonotized width {} with {} error term(s)",
                            node_name,
                            ibp_bounds.max_width(),
                            z.max_width(),
                            z.n_error_terms()
                        );
                    }
                    bounds_cache.insert(node_name.clone(), ibp_bounds);
                    zonotope_cache.insert(node_name.clone(), z);
                    // Derived from IBP, not zonotope - it's derived from IBP
                }
                Err(e) => return Err(e),
            }

            // NaN firewall (#2812 Slice 4, #2706).
            if let Some(output_bounds) = bounds_cache.get(node_name) {
                check_nan_firewall(
                    output_bounds,
                    "Zonotope IBP",
                    node_name,
                    node.layer.layer_type(),
                )?;
            }
        }

        // Get output node
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        // Return bounds from output zonotope
        let output = zonotope_cache.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;

        info!(
            "GraphNetwork zonotope: final output has {} error terms, max_width {}",
            output.n_error_terms(),
            output.max_width()
        );

        output.to_bounded_tensor()
    }

    /// Try to propagate a single node using zonotopes.
    ///
    /// Returns Ok(ZonotopeTensor) if the operation is supported, Err otherwise.
    pub(crate) fn propagate_zonotope_node(
        &self,
        node: &GraphNode,
        input_zonotope: &ZonotopeTensor,
        zonotope_cache: &std::collections::HashMap<String, ZonotopeTensor>,
        options: ZonotopePropagationOptions,
    ) -> Result<ZonotopeTensor> {
        // Helper to get zonotope for an input
        let get_zonotope = |name: &str| -> Result<ZonotopeTensor> {
            if name == NETWORK_INPUT {
                Ok(input_zonotope.clone())
            } else if let Some(z) = zonotope_cache.get(name) {
                Ok(z.clone())
            } else {
                Err(NyError::InvalidSpec(format!(
                    "No zonotope found for node {}",
                    name
                )))
            }
        };
        let get_unary_zonotope = || -> Result<ZonotopeTensor> {
            let input_name = node.require_unary_input()?;
            get_zonotope(input_name)
        };
        let get_binary_zonotopes = || -> Result<(ZonotopeTensor, ZonotopeTensor)> {
            let (input_a_name, input_b_name) = node.require_binary_inputs()?;
            Ok((get_zonotope(input_a_name)?, get_zonotope(input_b_name)?))
        };

        match &node.layer {
            // Linear layer: z = W @ z + b
            Layer::Linear(linear) => {
                let input_z = get_unary_zonotope()?;
                input_z.linear(&linear.weight, linear.bias.as_ref())
            }

            // AddConstant: z + c
            Layer::AddConstant(add_const) => {
                let input_z = get_unary_zonotope()?;
                input_z.add_constant(&add_const.constant)
            }

            // MulConstant: z * c
            Layer::MulConstant(mul_const) => {
                let input_z = get_unary_zonotope()?;
                input_z.mul_constant(&mul_const.constant)
            }

            // MatMul: Q @ K^T or probs @ V in attention
            Layer::MatMul(matmul) => {
                let (q_z, k_z) = get_binary_zonotopes()?;

                if !matmul.transpose_b {
                    // `softmax @ V` commonly reaches this path after the Softmax
                    // node has fallen back to interval bounds and been reintroduced
                    // as a zonotope with fresh, disjoint error symbols. Use the
                    // conservative disjoint-symbol bilinear path instead of faking
                    // shared-symbol provenance (#318 Packet C). Keep the previous
                    // interval fallback if it is narrower on this node.
                    return propagate_disjoint_matmul(matmul, &q_z, &k_z);
                }

                // Check if Q and K share error symbols
                if q_z.n_error_terms() != k_z.n_error_terms() {
                    debug!(
                        "MatMul zonotope: Q has {} errors, K has {}, need to expand",
                        q_z.n_error_terms(),
                        k_z.n_error_terms()
                    );
                    let (q_expanded, k_expanded) = q_z.expand_to_match(&k_z)?;
                    return q_expanded.matmul_transposed(&k_expanded);
                }

                // Both have same error symbols - this is the ideal case for Q@K^T
                let out = q_z.matmul_transposed(&k_z)?;
                if let Some(scale) = matmul.scale {
                    let scale_tensor = ndarray::ArrayD::from_elem(out.shape().to_vec(), scale);
                    out.mul_constant(&scale_tensor)
                } else {
                    Ok(out)
                }
            }

            // Add: z1 + z2 (element-wise)
            Layer::Add(_) => {
                let (a_z, b_z) = get_binary_zonotopes()?;

                // Expand to match if needed
                let (a_expanded, b_expanded) = a_z.expand_to_match(&b_z)?;
                a_expanded.add(&b_expanded)
            }

            // Reshape: preserves correlations perfectly (just rearranges elements)
            Layer::Reshape(reshape_layer) => {
                let input_z = get_unary_zonotope()?;

                // Compute output shape from reshape layer
                let input_shape = input_z.shape();
                let output_shape = reshape_layer.compute_output_shape(input_shape)?;

                input_z.reshape(&output_shape)
            }

            // Tile: preserves correlations (duplicated elements share error symbols)
            // Essential for GQA where K/V heads are tiled to match Q heads
            Layer::Tile(tile_layer) => {
                let input_z = get_unary_zonotope()?;

                // Compute actual axis (handle negative indexing)
                let ndim = input_z.shape().len();
                let axis = crate::layers::common::resolve_axis_i32(
                    tile_layer.axis,
                    ndim,
                    "Tile (zonotope IBP)",
                )?;

                input_z.tile(axis, tile_layer.reps)
            }

            // Transpose: preserves correlations (just permutes axes)
            Layer::Transpose(transpose_layer) => {
                let input_z = get_unary_zonotope()?;

                // Check if it's a simple swap of last two dimensions
                let ndim = transpose_layer.axes.len();
                if ndim >= 2
                    && transpose_layer.axes[ndim - 2] == ndim - 1
                    && transpose_layer.axes[ndim - 1] == ndim - 2
                {
                    input_z.transpose_last_two()
                } else {
                    Err(NyError::UnsupportedOp(format!(
                        "Zonotope transpose only supports swapping last two dims, got axes {:?}",
                        transpose_layer.axes
                    )))
                }
            }

            // LayerNorm: use affine approximation to preserve correlations
            Layer::LayerNorm(ln) => {
                let input_z = get_unary_zonotope()?;
                match ln.mode {
                    LayerNormMode::Standard => input_z.layer_norm_affine(&ln.ny, &ln.beta, ln.eps),
                    LayerNormMode::MeanOnly => {
                        input_z.layer_norm_affine_mean_only(&ln.ny, &ln.beta)
                    }
                }
            }

            // SiLU: use affine approximation to preserve correlations
            Layer::SiLU(_) => {
                let input_z = get_unary_zonotope()?;
                input_z.silu_affine()
            }
            // GELU: use affine approximation to preserve correlations
            // Uses correct GELU math (not SiLU). Fix for #2470.
            Layer::GELU(g) => {
                let input_z = get_unary_zonotope()?;
                let use_tanh = matches!(
                    g.approximation,
                    crate::layers::softmax::GeluApproximation::Tanh
                );
                input_z.gelu_affine(use_tanh)
            }

            // MulBinary: element-wise multiplication (needed for SwiGLU)
            // z1 ⊙ z2 = silu(gate) * up, exploits shared error symbols
            Layer::MulBinary(_) => {
                let (z1, z2) = get_binary_zonotopes()?;

                z1.mul_elementwise(&z2)
            }

            // Softmax: use affine approximation to preserve correlations
            // This linearizes softmax around the center and adds a conservative error term;
            // the result is still heuristic and not a proof of global soundness.
            Layer::Softmax(s) => {
                if matches!(
                    options.softmax_mode(),
                    ZonotopeSoftmaxMode::IntervalFallback
                ) {
                    // #318 Packet B: cut the zonotope pipeline at Softmax so the
                    // existing node-local IBP fallback can isolate whether the
                    // widener is Softmax itself or a later attention operator.
                    return Err(NyError::UnsupportedOp(
                        "Zonotope Softmax explicitly configured to fall back to IBP".to_string(),
                    ));
                }
                let input_z = get_unary_zonotope()?;
                match input_z.softmax_affine(s.axis) {
                    Ok(z) => Ok(z),
                    Err(NyError::InvalidSpec(msg)) if msg.contains("not yet implemented") => {
                        Err(NyError::UnsupportedConfiguration(msg))
                    }
                    Err(e) => Err(e),
                }
            }

            // CausalSoftmax: use affine approximation (same as Softmax for now)
            Layer::CausalSoftmax(cs) => {
                if matches!(
                    options.softmax_mode(),
                    ZonotopeSoftmaxMode::IntervalFallback
                ) {
                    return Err(NyError::UnsupportedOp(
                        "Zonotope CausalSoftmax explicitly configured to fall back to IBP"
                            .to_string(),
                    ));
                }
                let input_z = get_unary_zonotope()?;
                match input_z.softmax_affine_causal(cs.axis) {
                    Ok(z) => Ok(z),
                    Err(NyError::InvalidSpec(msg)) if msg.contains("not yet implemented") => {
                        Err(NyError::UnsupportedConfiguration(msg))
                    }
                    Err(e) => Err(e),
                }
            }

            // Operations that don't preserve zonotope form well
            Layer::ReLU(_)
            | Layer::Tanh(_)
            | Layer::Sigmoid(_)
            | Layer::Tan(_)
            | Layer::Arctan(_)
            | Layer::Sqrt(_)
            | Layer::PowConstant(_) => {
                // These operations break zonotope form — fall back to IBP (#3117).
                Err(NyError::UnsupportedOp(format!(
                    "Operation {} not supported for zonotope propagation",
                    node.layer.layer_type()
                )))
            }

            // Other operations — not yet implemented (#3117).
            _ => Err(NyError::UnsupportedOp(format!(
                "Operation {} not yet implemented for zonotope propagation",
                node.layer.layer_type()
            ))),
        }
    }
}
