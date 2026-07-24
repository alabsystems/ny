// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::Layer;

use ndarray::IxDyn;
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, ZonotopeTensor};
use tracing::debug;

use crate::network::core::{GraphNetwork, GraphNode};

fn required_named_input<'a>(node: &'a GraphNode, role: &str) -> Result<&'a str> {
    node.require_unary_input()
        .map_err(|_| NyError::InvalidSpec(format!("{role} node {} has no inputs", node.name)))
}

impl GraphNetwork {
    /// Try to compute tighter bounds for Q@K^T using zonotope correlation tracking.
    ///
    /// This handles both MHA (Q,K directly from Linear) and GQA (K through reshape/tile ops):
    /// - MHA: input -> q_proj -> Q, input -> k_proj -> K
    /// - GQA: input -> q_proj -> Q, input -> k_proj -> k_reshape -> k_tile -> k_reshape -> K
    ///
    /// For GQA, zonotope propagation through reshape/tile preserves correlations because
    /// the tiled K heads share the same error symbols as the original projection.
    pub(crate) fn try_attention_matmul_bounds_zonotope(
        &self,
        matmul_node: &GraphNode,
        input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        let matmul = match &matmul_node.layer {
            Layer::MatMul(m) => m,
            _ => return Ok(None),
        };
        if !matmul.transpose_b {
            return Ok(None);
        }

        let (q_node_name, k_node_name) = match matmul_node.require_binary_inputs() {
            Ok(inputs) => inputs,
            Err(_) => return Ok(None),
        };

        // Trace back Q and K to find their base Linear projections and the operations in between
        let q_path = self.trace_zonotope_path(q_node_name);
        let k_path = self.trace_zonotope_path(k_node_name);

        // Find the Linear layer in each path (should be the first element after reverse)
        let (q_linear_name, q_ops) = match q_path.split_last() {
            Some((linear_name, ops)) => (linear_name.clone(), ops.to_vec()),
            None => return Ok(None),
        };
        let (k_linear_name, k_ops) = match k_path.split_last() {
            Some((linear_name, ops)) => (linear_name.clone(), ops.to_vec()),
            None => return Ok(None),
        };

        let q_linear_node = match self.nodes.get(&q_linear_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let k_linear_node = match self.nodes.get(&k_linear_name) {
            Some(n) => n,
            None => return Ok(None),
        };

        let (q_linear, k_linear) = match (&q_linear_node.layer, &k_linear_node.layer) {
            (Layer::Linear(q), Layer::Linear(k)) => (q, k),
            _ => return Ok(None),
        };

        // Check that Q and K projections share the same base input
        let q_base = required_named_input(q_linear_node, "Linear")?;
        let k_base = required_named_input(k_linear_node, "Linear")?;

        if q_base != k_base {
            return Ok(None);
        }

        // Pre-validate that all Transpose operations in Q and K paths use supported
        // axes patterns (last-two-axes swap) BEFORE computing expensive linear projections.
        // Without this check, we waste O(n²d) zonotope matmuls only to bail out later
        // when encountering unsupported transposes (e.g. Whisper's [0,2,1,3] permutation).
        // This was the root cause of test_encoder_layer_graph_network_ibp timing out (#1662).
        for ops in [&q_ops, &k_ops] {
            for op_name in ops.iter() {
                if let Some(op_node) = self.nodes.get(op_name) {
                    if let Layer::Transpose(transpose) = &op_node.layer {
                        let ndim = transpose.axes.len();
                        if ndim < 2
                            || transpose.axes[ndim - 2] != ndim - 1
                            || transpose.axes[ndim - 1] != ndim - 2
                        {
                            debug!(
                                "Zonotope Q@K^T: unsupported transpose axes {:?} in {}, skipping",
                                transpose.axes, op_name
                            );
                            return Ok(None);
                        }
                    }
                }
            }
        }

        // Use the LayerNorm output directly (q_base) without propagating through LayerNorm.
        // Empirical testing shows that the LayerNorm affine approximation error overwhelms
        // any benefit from preserved correlations - creating fresh zonotopes from LayerNorm
        // output gives 10^6-10^9x tighter Q@K^T bounds than propagating through LayerNorm.
        //
        // Evidence (Qwen3-0.6B block-wise verification):
        // - Layer0 with LN propagation: Q@K^T width = 2.440e3
        // - Layers 1-27 without LN prop: Q@K^T width = 1e-6 to 1e-3
        let actual_base_name = q_base.to_string();

        // Get base bounds and create zonotope
        let base_bounds = self.bounds_ref(&actual_base_name, input, bounds_cache)?;
        // Require at least (..., seq, dim) so per-position zonotopes can be created.
        if base_bounds.shape().len() < 2 {
            return Ok(None);
        }
        let base_width = base_bounds.max_width();

        // When input bounds are large (from cumulative propagation), use a single
        // shared error term to prevent overflow in matmul_transposed. With per-position
        // error terms (n=seq_len), the O(n²) cross terms can overflow. A single error
        // term gives tighter bounds than IBP while avoiding overflow.
        //
        // Threshold: base_width > 1.0 suggests cumulative bound growth from previous layers.
        // For fresh input (epsilon ~ 0.001), base_width < 1.0 and we use per-position tracking.
        //
        // Spectral norm scaling: The Q and K Linear projections amplify zonotope coefficients
        // by their spectral norms. For Q@K^T (quadratic in coefficients), cross-terms explode
        // as O(spec_q * spec_k). We normalize by max spectral norm to keep coefficients ~1
        // after Linear, preventing overflow while preserving soundness.
        let max_spectral = q_linear.spectral_norm().max(k_linear.spectral_norm());
        let zonotope_scale = if base_width > 1.0 || max_spectral > 1.0 {
            (base_width / 2.0).max(1.0) * max_spectral.max(1.0)
        } else {
            1.0
        };
        let (base_z, needs_rescale) = if zonotope_scale > 1.0 {
            // Large bounds or large spectral norm: use single error term with normalization
            let normalized_bounds = BoundedTensor::new(
                base_bounds.lower().mapv(|v| v / zonotope_scale),
                base_bounds.upper().mapv(|v| v / zonotope_scale),
            )?;
            // Reshape to 2D for matmul compatibility
            let flat_shape = vec![checked_shape_product(base_bounds.shape()).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Zonotope attention: shape product overflows: {:?}",
                    base_bounds.shape()
                ))
            })?];
            let flat_lower = normalized_bounds
                .lower()
                .clone()
                .into_shape_with_order(IxDyn(&flat_shape))
                .map_err(|e| NyError::InvalidSpec(format!("reshape failed: {}", e)))?;
            let flat_upper = normalized_bounds
                .upper()
                .clone()
                .into_shape_with_order(IxDyn(&flat_shape))
                .map_err(|e| NyError::InvalidSpec(format!("reshape failed: {}", e)))?;
            let flat_bounds = BoundedTensor::new(flat_lower, flat_upper)?;

            // Create single-error-term zonotope then reshape to 2D
            let z_flat = ZonotopeTensor::from_bounded_tensor(&flat_bounds);
            let z_2d = z_flat.reshape(base_bounds.shape())?;
            (z_2d, true)
        } else {
            // Small bounds: use per-position error terms for tighter correlation tracking
            let z = ZonotopeTensor::from_bounded_tensor_per_position(base_bounds)?;
            (z, false)
        };
        debug!(
            "Zonotope Q@K^T tightening: base={} base_width={:.3e} max_spectral={:.3e} scale={:.3e} n_err={}",
            actual_base_name, base_width, max_spectral, zonotope_scale, base_z.n_error_terms()
        );

        // Apply Q projection
        let q_z = base_z.linear(&q_linear.weight, q_linear.bias.as_ref())?;

        // Apply K projection then propagate through reshape/tile operations
        let mut k_z = base_z.linear(&k_linear.weight, k_linear.bias.as_ref())?;

        // Apply operations in order (k_ops is in forward order: reshape1, tile, reshape2)
        for op_name in k_ops.iter().rev() {
            let op_node = match self.nodes.get(op_name) {
                Some(n) => n,
                None => return Ok(None),
            };

            k_z = match &op_node.layer {
                Layer::Reshape(reshape) => {
                    let output_shape = reshape.compute_output_shape(k_z.shape())?;
                    k_z.reshape(&output_shape)?
                }
                Layer::Tile(tile) => {
                    let ndim = k_z.shape().len();
                    let axis = crate::layers::common::resolve_axis_i32(
                        tile.axis,
                        ndim,
                        "Tile (zonotope K-path)",
                    )?;
                    k_z.tile(axis, tile.reps)?
                }
                Layer::Transpose(transpose) => {
                    let ndim = transpose.axes.len();
                    if ndim >= 2
                        && transpose.axes[ndim - 2] == ndim - 1
                        && transpose.axes[ndim - 1] == ndim - 2
                    {
                        k_z.transpose_last_two()?
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None), // Unsupported operation in K path
            };
        }

        // Apply Q operations if any
        let mut q_z_final = q_z;
        for op_name in q_ops.iter().rev() {
            let op_node = match self.nodes.get(op_name) {
                Some(n) => n,
                None => return Ok(None),
            };

            q_z_final = match &op_node.layer {
                Layer::Reshape(reshape) => {
                    let output_shape = reshape.compute_output_shape(q_z_final.shape())?;
                    q_z_final.reshape(&output_shape)?
                }
                Layer::Tile(tile) => {
                    let ndim = q_z_final.shape().len();
                    let axis = crate::layers::common::resolve_axis_i32(
                        tile.axis,
                        ndim,
                        "Tile (zonotope Q-path)",
                    )?;
                    q_z_final.tile(axis, tile.reps)?
                }
                Layer::Transpose(transpose) => {
                    let ndim = transpose.axes.len();
                    if ndim >= 2
                        && transpose.axes[ndim - 2] == ndim - 1
                        && transpose.axes[ndim - 1] == ndim - 2
                    {
                        q_z_final.transpose_last_two()?
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            };
        }

        // Check that Q and K zonotopes have compatible shapes for matmul_transposed
        // Q: (seq_q, dim_q), K: (seq_k, dim_k) where dim_q == dim_k
        if q_z_final.shape().len() != 2 || k_z.shape().len() != 2 {
            return Ok(None);
        }
        if q_z_final.shape()[1] != k_z.shape()[1] {
            return Ok(None);
        }

        let mut out = q_z_final.matmul_transposed(&k_z)?;
        if let Some(scale) = matmul.scale {
            let scale_tensor = ndarray::ArrayD::from_elem(out.shape().to_vec(), scale);
            out = out.mul_constant(&scale_tensor)?;
        }

        let result = out.to_bounded_tensor()?;

        // Scale back the result by zonotope_scale² (matmul is quadratic in input scale)
        // If we normalized input x by scale s, then:
        //   Q_norm = Q / s, K_norm = K / s
        //   (Q_norm @ K_norm^T) = (Q @ K^T) / s²
        // So we multiply by s² to recover the correct scale.
        // Use f64 intermediates to avoid inf * 0 = NaN when scale² overflows f32.
        // IMPORTANT: Check for NaN/Inf BEFORE creating BoundedTensor to avoid panic.
        let result = if needs_rescale {
            let scale_sq = (zonotope_scale as f64) * (zonotope_scale as f64);
            // Directed rounding: lower → next_down, upper → next_up (#2225).
            let scaled_lower = result
                .lower()
                .mapv(|v| next_down_f32((v as f64 * scale_sq) as f32));
            let scaled_upper = result
                .upper()
                .mapv(|v| next_up_f32((v as f64 * scale_sq) as f32));

            // Check for NaN/Inf before creating BoundedTensor
            let has_bad_values = scaled_lower
                .iter()
                .chain(scaled_upper.iter())
                .any(|v| v.is_nan() || v.is_infinite());

            if has_bad_values {
                // Zonotope rescale produced overflow - fall back to IBP
                debug!(
                    "Zonotope Q@K^T rescale overflow: scale²={:.3e}, falling back to IBP",
                    scale_sq
                );
                return Ok(None);
            }

            BoundedTensor::new(scaled_lower, scaled_upper)?
        } else {
            // Check original result for NaN/Inf
            let has_bad_values = result
                .lower()
                .iter()
                .chain(result.upper().iter())
                .any(|v| v.is_nan() || v.is_infinite());

            if has_bad_values {
                // Zonotope produced overflow - fall back to IBP
                debug!("Zonotope Q@K^T overflow, falling back to IBP");
                return Ok(None);
            }
            result
        };

        let result_width = result.max_width();

        debug!(
            "Zonotope Q@K^T output: width={:.3e} scale²={:.3e} shape={:?}",
            result_width,
            zonotope_scale * zonotope_scale,
            result.shape()
        );

        Ok(Some(result))
    }

    /// Trace back from a node through zonotope-preserving operations to find the source Linear.
    ///
    /// Returns the path of node names from the given node back to the Linear layer (inclusive).
    /// Operations that preserve zonotope form: Reshape, Tile, Transpose, Linear.
    fn trace_zonotope_path(&self, start_node: &str) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = start_node.to_string();

        while let Some(node) = self.nodes.get(&current) {
            path.push(current.clone());

            match &node.layer {
                Layer::Linear(_) => break, // Found the base Linear
                Layer::Reshape(_) | Layer::Tile(_) | Layer::Transpose(_) => {
                    // Continue tracing back
                    if let Some(input) = node.inputs.first() {
                        current = input.clone();
                    } else {
                        break;
                    }
                }
                _ => break, // Non-zonotope-preserving operation
            }
        }

        path
    }
}
