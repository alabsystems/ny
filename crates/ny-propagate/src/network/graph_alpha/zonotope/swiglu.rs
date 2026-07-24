// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::Layer;

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, ZonotopeTensor};
use tracing::debug;

use crate::network::core::{GraphNetwork, GraphNode};

fn required_named_input<'a>(node: &'a GraphNode, role: &str) -> Result<&'a str> {
    node.require_unary_input()
        .map_err(|_| NyError::InvalidSpec(format!("{role} node {} has no inputs", node.name)))
}

impl GraphNetwork {
    /// Try to apply zonotope tightening for SwiGLU FFN in block-wise mode.
    ///
    /// SwiGLU: output = up * silu(gate), where both up and gate come from the same base (ffn_norm).
    /// By using zonotopes, we can track correlations through the shared error symbols and
    /// get tighter bounds than IBP which treats them as independent.
    ///
    /// # Pattern
    /// ```text
    /// ffn_norm -> ffn_up (Linear) -------> up
    ///          -> ffn_gate (Linear) -> silu -> gate
    /// MulBinary(up, gate) -> swiglu
    /// ```
    pub(crate) fn try_ffn_swiglu_bounds_zonotope_block(
        &self,
        mul_node: &GraphNode,
        block_input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        let (input_a_name, input_b_name) = match mul_node.require_binary_inputs() {
            Ok(inputs) => inputs,
            Err(_) => return Ok(None),
        };

        // Identify up and gate branches
        // Pattern: MulBinary(up, silu(gate)) or MulBinary(silu(gate), up)
        let (up_name, silu_name) = {
            let node_a = self.nodes.get(input_a_name);
            let node_b = self.nodes.get(input_b_name);

            match (node_a, node_b) {
                (Some(a), Some(b)) => {
                    // Check if one is SiLU and trace back
                    let a_is_silu = matches!(&a.layer, Layer::SiLU(_));
                    let b_is_silu = matches!(&b.layer, Layer::SiLU(_));

                    if a_is_silu && !b_is_silu {
                        (input_b_name.to_string(), input_a_name.to_string())
                    } else if b_is_silu && !a_is_silu {
                        (input_a_name.to_string(), input_b_name.to_string())
                    } else {
                        // Neither or both are SiLU - not standard SwiGLU pattern
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        };

        // Get the SiLU node and trace back to gate Linear
        let silu_node = match self.nodes.get(silu_name.as_str()) {
            Some(n) => n,
            None => return Ok(None),
        };

        let gate_name = match silu_node.require_unary_input() {
            Ok(input_name) => input_name,
            Err(_) => return Ok(None),
        };

        // Gate should be a Linear layer
        let gate_node = match self.nodes.get(gate_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let gate_linear = match &gate_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Up should be a Linear layer (or trace back to one)
        let up_node = match self.nodes.get(up_name.as_str()) {
            Some(n) => n,
            None => return Ok(None),
        };
        let up_linear = match &up_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Check that gate and up share the same input (ffn_norm output)
        let gate_base = required_named_input(gate_node, "Gate")?;
        let up_base = required_named_input(up_node, "Up")?;

        if gate_base != up_base {
            // Different bases - can't exploit correlation
            debug!(
                "SwiGLU zonotope: gate_base='{}' != up_base='{}', skipping",
                gate_base, up_base
            );
            return Ok(None);
        }

        // Get base bounds (should be ffn_norm output)
        let base_bounds = self.bounds_for_block(gate_base, block_input, bounds_cache)?;
        if base_bounds.shape().len() < 2 {
            return Ok(None);
        }

        let base_width = base_bounds.max_width();

        // Compute normalization scale for the quadratic mul_elementwise step.
        // The zonotope multiplication (up * silu(gate)) is quadratic, so large
        // post-linear coefficients cause cross-term explosion.
        //
        // SOUNDNESS FIX (#2386): We no longer normalize the input to the SiLU.
        // SiLU is nonlinear: silu(x/s) ≠ silu(x)/s, so normalizing before SiLU
        // then scaling back by s² was unsound. Instead:
        //   1. Compute gate, silu, up at full scale (correct values)
        //   2. Normalize both sides by s just before mul_elementwise
        //   3. Scale back result by s² (exact: (a/s)*(b/s)*s² = a*b)
        let max_spectral = gate_linear.spectral_norm().max(up_linear.spectral_norm());
        let zonotope_scale = if base_width > 1.0 || max_spectral > 1.0 {
            (base_width / 2.0).max(1.0) * max_spectral.max(1.0)
        } else {
            1.0
        };

        // Create zonotope from UNNORMALIZED bounds — SiLU needs true values.
        let base_z = ZonotopeTensor::from_bounded_tensor_per_position(&base_bounds)?;

        debug!(
            "SwiGLU zonotope: base='{}' base_width={:.3e} max_spectral={:.1} scale={:.3e} n_err={}",
            gate_base,
            base_width,
            max_spectral,
            zonotope_scale,
            base_z.n_error_terms()
        );

        // Apply gate Linear projection (full scale — exact)
        let gate_z = base_z.linear(&gate_linear.weight, gate_linear.bias.as_ref())?;

        // Apply SiLU to gate at full scale — correct because silu is nonlinear
        let silu_z = gate_z.silu_affine()?;

        // Apply up Linear projection (full scale — exact)
        let up_z = base_z.linear(&up_linear.weight, up_linear.bias.as_ref())?;

        // Normalize both sides before the quadratic multiplication to prevent
        // cross-term overflow, then scale back by s² (exact: (a/s)*(b/s)*s² = a*b)
        let (up_z_mul, silu_z_mul) = if zonotope_scale > 1.0 {
            let inv_s = 1.0 / zonotope_scale;
            (up_z.scale(inv_s), silu_z.scale(inv_s))
        } else {
            (up_z, silu_z)
        };
        let swiglu_z = up_z_mul.mul_elementwise(&silu_z_mul)?;

        let result = swiglu_z.to_bounded_tensor()?;

        // Scale back by s² — exact because zonotope scaling is linear and
        // (a/s) * (b/s) * s² = a * b.
        let result = if zonotope_scale > 1.0 {
            let scale_sq = (zonotope_scale as f64) * (zonotope_scale as f64);
            // Directed rounding: lower → next_down, upper → next_up (#2225).
            BoundedTensor::new(
                result
                    .lower()
                    .mapv(|v| next_down_f32((v as f64 * scale_sq) as f32)),
                result
                    .upper()
                    .mapv(|v| next_up_f32((v as f64 * scale_sq) as f32)),
            )?
        } else {
            result
        };

        let result_width = result.max_width();

        debug!(
            "SwiGLU zonotope output: width={:.3e} scale²={:.3e} shape={:?}",
            result_width,
            zonotope_scale * zonotope_scale,
            result.shape()
        );

        // Validate output: if zonotope produced NaN/Inf, fall back to IBP
        let has_bad_values = result
            .lower()
            .iter()
            .chain(result.upper().iter())
            .any(|v| v.is_nan() || v.is_infinite());

        if has_bad_values {
            debug!("SwiGLU zonotope: output has NaN/Inf, falling back to IBP");
            return Ok(None);
        }

        Ok(Some(result))
    }

    /// Try to compute tighter bounds for SwiGLU (up * silu(gate)) using zonotope.
    ///
    /// Non-block version for full network propagation. Uses single-error zonotope
    /// for large bounds to prevent overflow.
    pub(crate) fn try_ffn_swiglu_bounds_zonotope(
        &self,
        mul_node: &GraphNode,
        input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        let (input_a_name, input_b_name) = match mul_node.require_binary_inputs() {
            Ok(inputs) => inputs,
            Err(_) => return Ok(None),
        };

        // Identify up and gate branches
        // Pattern: MulBinary(up, silu(gate)) or MulBinary(silu(gate), up)
        let (up_name, silu_name) = {
            let node_a = self.nodes.get(input_a_name);
            let node_b = self.nodes.get(input_b_name);

            match (node_a, node_b) {
                (Some(a), Some(b)) => {
                    let a_is_silu = matches!(&a.layer, Layer::SiLU(_));
                    let b_is_silu = matches!(&b.layer, Layer::SiLU(_));

                    if a_is_silu && !b_is_silu {
                        (input_b_name.to_string(), input_a_name.to_string())
                    } else if b_is_silu && !a_is_silu {
                        (input_a_name.to_string(), input_b_name.to_string())
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        };

        // Get the SiLU node and trace back to gate Linear
        let silu_node = match self.nodes.get(silu_name.as_str()) {
            Some(n) => n,
            None => return Ok(None),
        };

        let gate_name = match silu_node.require_unary_input() {
            Ok(input_name) => input_name,
            Err(_) => return Ok(None),
        };

        let gate_node = match self.nodes.get(gate_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let gate_linear = match &gate_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        let up_node = match self.nodes.get(up_name.as_str()) {
            Some(n) => n,
            None => return Ok(None),
        };
        let up_linear = match &up_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Check that gate and up share the same input
        let gate_base = required_named_input(gate_node, "Gate")?;
        let up_base = required_named_input(up_node, "Up")?;

        if gate_base != up_base {
            return Ok(None);
        }

        // Get base bounds
        let base_bounds = self.bounds_ref(gate_base, input, bounds_cache)?;
        if base_bounds.shape().len() < 2 {
            return Ok(None);
        }

        let base_width = base_bounds.max_width();

        // Compute normalization scale for the quadratic mul_elementwise step.
        // SOUNDNESS FIX (#2386): normalize AFTER silu, not before.
        // See try_ffn_swiglu_bounds_zonotope_block for detailed explanation.
        let max_spectral = gate_linear.spectral_norm().max(up_linear.spectral_norm());
        let zonotope_scale = if base_width > 1.0 || max_spectral > 1.0 {
            (base_width / 2.0).max(1.0) * max_spectral.max(1.0)
        } else {
            1.0
        };

        // Per-position error terms preserve correlations through up/gate projections.
        // Create from UNNORMALIZED bounds — SiLU needs true values.
        let base_z = ZonotopeTensor::from_bounded_tensor_per_position(base_bounds)?;

        debug!(
            "SwiGLU zonotope (full): base='{}' base_width={:.3e} max_spectral={:.1} scale={:.3e} n_err={}",
            gate_base, base_width, max_spectral, zonotope_scale, base_z.n_error_terms()
        );

        // Apply gate Linear projection (full scale — exact)
        let gate_z = base_z.linear(&gate_linear.weight, gate_linear.bias.as_ref())?;

        // Apply SiLU to gate at full scale — correct because silu is nonlinear
        let silu_z = gate_z.silu_affine()?;

        // Apply up Linear projection (full scale — exact)
        let up_z = base_z.linear(&up_linear.weight, up_linear.bias.as_ref())?;

        // Normalize both sides before the quadratic multiplication to prevent
        // cross-term overflow, then scale back by s² (exact: (a/s)*(b/s)*s² = a*b)
        let (up_z_mul, silu_z_mul) = if zonotope_scale > 1.0 {
            let inv_s = 1.0 / zonotope_scale;
            (up_z.scale(inv_s), silu_z.scale(inv_s))
        } else {
            (up_z, silu_z)
        };
        let swiglu_z = up_z_mul.mul_elementwise(&silu_z_mul)?;

        let result = swiglu_z.to_bounded_tensor()?;

        // Scale back by s² — exact because (a/s)*(b/s)*s² = a*b.
        let result = if zonotope_scale > 1.0 {
            let scale_sq = (zonotope_scale as f64) * (zonotope_scale as f64);
            // Directed rounding: lower → next_down, upper → next_up (#2225).
            BoundedTensor::new(
                result
                    .lower()
                    .mapv(|v| next_down_f32((v as f64 * scale_sq) as f32)),
                result
                    .upper()
                    .mapv(|v| next_up_f32((v as f64 * scale_sq) as f32)),
            )?
        } else {
            result
        };

        let result_width = result.max_width();

        debug!(
            "SwiGLU zonotope (full) output: width={:.3e} scale²={:.3e} shape={:?}",
            result_width,
            zonotope_scale * zonotope_scale,
            result.shape()
        );

        // Validate output
        let has_bad_values = result
            .lower()
            .iter()
            .chain(result.upper().iter())
            .any(|v| v.is_nan() || v.is_infinite());

        if has_bad_values {
            debug!("SwiGLU zonotope (full): output has NaN/Inf, falling back to IBP");
            return Ok(None);
        }

        Ok(Some(result))
    }

    /// Try to apply zonotope tightening for the full FFN (including down projection).
    ///
    /// When processing the ffn_down Linear node, if its input is a SwiGLU pattern,
    /// we can propagate zonotopes through the entire FFN for tighter bounds:
    /// ffn_norm -> up + (gate -> silu) -> mul -> down
    ///
    /// This extends the SwiGLU zonotope tightening to include the down projection,
    /// which previously fell back to IBP and amplified bounds ~16x per block.
    pub(crate) fn try_ffn_down_zonotope_block(
        &self,
        linear_node: &GraphNode,
        block_input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        // This node must be Linear
        let down_linear = match &linear_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Get input node (should be MulBinary for SwiGLU)
        let mul_name = match linear_node.require_unary_input() {
            Ok(input_name) => input_name,
            Err(_) => return Ok(None),
        };

        // Check if input is MulBinary
        let mul_node = match self.nodes.get(mul_name) {
            Some(n) => n,
            None => return Ok(None),
        };

        if !matches!(&mul_node.layer, Layer::MulBinary(_)) {
            return Ok(None);
        }

        // Now trace back through SwiGLU pattern: MulBinary(up, silu(gate))
        let (input_a_name, input_b_name) = match mul_node.require_binary_inputs() {
            Ok(inputs) => inputs,
            Err(_) => return Ok(None),
        };

        // Identify up and gate branches
        let (up_name, silu_name) = {
            let node_a = self.nodes.get(input_a_name);
            let node_b = self.nodes.get(input_b_name);

            match (node_a, node_b) {
                (Some(a), Some(b)) => {
                    let a_is_silu = matches!(&a.layer, Layer::SiLU(_));
                    let b_is_silu = matches!(&b.layer, Layer::SiLU(_));

                    if a_is_silu && !b_is_silu {
                        (input_b_name.to_string(), input_a_name.to_string())
                    } else if b_is_silu && !a_is_silu {
                        (input_a_name.to_string(), input_b_name.to_string())
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        };

        // Get the SiLU node and trace back to gate Linear
        let silu_node = match self.nodes.get(silu_name.as_str()) {
            Some(n) => n,
            None => return Ok(None),
        };

        let gate_name = match silu_node.require_unary_input() {
            Ok(input_name) => input_name,
            Err(_) => return Ok(None),
        };

        // Gate should be a Linear layer
        let gate_node = match self.nodes.get(gate_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let gate_linear = match &gate_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Up should be a Linear layer
        let up_node = match self.nodes.get(up_name.as_str()) {
            Some(n) => n,
            None => return Ok(None),
        };
        let up_linear = match &up_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Check that gate and up share the same input (ffn_norm output)
        let gate_base = required_named_input(gate_node, "Gate")?;
        let up_base = required_named_input(up_node, "Up")?;

        if gate_base != up_base {
            debug!(
                "FFN down zonotope: gate_base='{}' != up_base='{}', skipping",
                gate_base, up_base
            );
            return Ok(None);
        }

        // Get base bounds (ffn_norm output)
        let base_bounds = self.bounds_for_block(gate_base, block_input, bounds_cache)?;
        if base_bounds.shape().len() < 2 {
            return Ok(None);
        }

        let base_width = base_bounds.max_width();

        // Compute normalization scale for the quadratic mul_elementwise step.
        // SOUNDNESS FIX (#2386): normalize AFTER silu, not before.
        // Include down spectral norm for numerical safety (conservative).
        let max_spectral = gate_linear
            .spectral_norm()
            .max(up_linear.spectral_norm())
            .max(down_linear.spectral_norm());
        let zonotope_scale = if base_width > 1.0 || max_spectral > 1.0 {
            (base_width / 2.0).max(1.0) * max_spectral.max(1.0)
        } else {
            1.0
        };

        // Create zonotope from UNNORMALIZED bounds — SiLU needs true values.
        let base_z = ZonotopeTensor::from_bounded_tensor_per_position(&base_bounds)?;

        debug!(
            "FFN down zonotope: base='{}' base_width={:.3e} max_spectral={:.1} scale={:.3e} n_err={}",
            gate_base, base_width, max_spectral, zonotope_scale, base_z.n_error_terms()
        );

        // Apply gate Linear projection (full scale — exact)
        let gate_z = base_z.linear(&gate_linear.weight, gate_linear.bias.as_ref())?;

        // Apply SiLU to gate at full scale — correct because silu is nonlinear
        let silu_z = gate_z.silu_affine()?;

        // Apply up Linear projection (full scale — exact)
        let up_z = base_z.linear(&up_linear.weight, up_linear.bias.as_ref())?;

        // Normalize both sides before the quadratic multiplication to prevent
        // cross-term overflow, then scale back by s² (exact: (a/s)*(b/s)*s² = a*b)
        let (up_z_mul, silu_z_mul) = if zonotope_scale > 1.0 {
            let inv_s = 1.0 / zonotope_scale;
            (up_z.scale(inv_s), silu_z.scale(inv_s))
        } else {
            (up_z, silu_z)
        };
        let swiglu_z = up_z_mul.mul_elementwise(&silu_z_mul)?;

        // Scale back by s² BEFORE applying down projection.
        // We must restore true magnitude before down_linear because
        // down(z) = W*z + b — if z is at 1/s² scale, the bias b is added
        // at the wrong relative magnitude.
        let swiglu_z = if zonotope_scale > 1.0 {
            swiglu_z.scale(zonotope_scale * zonotope_scale)
        } else {
            swiglu_z
        };

        // Apply down Linear projection (the key extension!)
        // Now swiglu_z is at correct magnitude, so bias is applied correctly.
        let down_z = swiglu_z.linear(&down_linear.weight, down_linear.bias.as_ref())?;

        let result = down_z.to_bounded_tensor()?;

        let result_width = result.max_width();

        debug!(
            "FFN down zonotope output: width={:.3e} scale²={:.3e} shape={:?}",
            result_width,
            zonotope_scale * zonotope_scale,
            result.shape()
        );

        // Validate output
        let has_bad_values = result
            .lower()
            .iter()
            .chain(result.upper().iter())
            .any(|v| v.is_nan() || v.is_infinite());

        if has_bad_values {
            debug!("FFN down zonotope: output has NaN/Inf, falling back to IBP");
            return Ok(None);
        }

        Ok(Some(result))
    }
}
