// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated compositional decoder verification.

use crate::fallback_logging::warn_gpu_fallback;
use crate::GpuCompositionalDetails;
use ny_core::{NyError, Result};
use ny_gpu::{AcceleratedBoundPropagation, AcceleratedDevice, ComputeDevice};
use ny_propagate::layers::LayerNormLayer;
use ny_propagate::BoundPropagation;
use tracing::info;

use super::DecoderModel;

impl DecoderModel {
    /// GPU-accelerated compositional verification of a decoder block.
    ///
    /// Uses GPU for causal self-attention and parallel CROWN for MLP.
    /// Falls back to CPU when GPU is unavailable or for small sequences.
    ///
    /// # Arguments
    /// * `block_index` - Index of the decoder block
    /// * `input` - Input bounded tensor [batch, seq, hidden]
    /// * `gpu_device` - Optional GPU device for acceleration
    ///
    /// # Returns
    /// (output_bounds, details) with GPU usage information.
    pub fn verify_block_compositional_gpu(
        &self,
        block_index: usize,
        input: &ny_tensor::BoundedTensor,
        gpu_device: Option<&ComputeDevice>,
    ) -> Result<(ny_tensor::BoundedTensor, GpuCompositionalDetails)> {
        const GPU_ATTENTION_THRESHOLD: usize = 64;

        let shape = input.shape();
        let seq_len = if shape.len() >= 2 { shape[1] } else { 1 };

        let cpu_device = AcceleratedDevice::new();

        // Step 1: Bound causal attention subgraph
        // Use GPU if we have a device and seq >= threshold
        let (attn_delta, used_gpu_attention) = if let Some(gpu) = gpu_device {
            if seq_len >= GPU_ATTENTION_THRESHOLD {
                // Try GPU causal attention
                match self.causal_attention_ibp_gpu(block_index, input, gpu) {
                    Ok(delta) => (delta, true),
                    Err(e) => {
                        warn_gpu_fallback("GPU causal attention failed", &e);
                        let attn_graph = self.causal_attention_subgraph(block_index)?;
                        (attn_graph.propagate_ibp(input)?, false)
                    }
                }
            } else {
                // seq < threshold, use CPU
                let attn_graph = self.causal_attention_subgraph(block_index)?;
                (attn_graph.propagate_ibp(input)?, false)
            }
        } else {
            // No GPU device, use CPU
            let attn_graph = self.causal_attention_subgraph(block_index)?;
            (attn_graph.propagate_ibp(input)?, false)
        };

        // Step 2: Compose with first residual: x_attn = x + attn_delta
        let x_attn = input.add(&attn_delta)?;

        // Step 3: Bound MLP subgraph with parallel per-position CROWN
        let mlp_graph = self.mlp_subgraph(block_index)?;
        let mlp_delta = cpu_device.crown_per_position_parallel(&mlp_graph, &x_attn)?;

        // Step 4: Compose with second residual: x_out = x_attn + mlp_delta
        let x_out = x_attn.add(&mlp_delta)?;

        let details = GpuCompositionalDetails {
            attention_delta_width: attn_delta.max_width(),
            x_attn_width: x_attn.max_width(),
            mlp_delta_width: mlp_delta.max_width(),
            output_width: x_out.max_width(),
            used_gpu_attention,
            used_zonotope_attention: false, // Decoder doesn't use zonotope attention yet
            seq_len,
            normalization_row_stats: vec![],
        };

        Ok((x_out, details))
    }

    /// GPU-accelerated causal attention IBP for a decoder block.
    ///
    /// This method extracts Q, K, V projections from the attention subgraph,
    /// runs them through CPU IBP, then uses GPU hybrid (GPU matmul, CPU causal softmax).
    ///
    /// # Arguments
    /// * `block_index` - Decoder block index
    /// * `input` - Input bounded tensor [batch, seq, hidden]
    /// * `gpu` - GPU device for acceleration
    ///
    /// # Returns
    /// Attention delta bounds (output of attention before residual add)
    fn causal_attention_ibp_gpu(
        &self,
        block_index: usize,
        input: &ny_tensor::BoundedTensor,
        gpu: &ComputeDevice,
    ) -> Result<ny_tensor::BoundedTensor> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads;
        let shape = input.shape();

        if shape.len() != 3 {
            return Err(NyError::InvalidSpec(format!(
                "Expected 3D input [batch, seq, hidden], got {:?}",
                shape
            )));
        }

        let batch = shape[0];
        let seq = shape[1];

        if shape[2] != hidden_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![batch, seq, hidden_dim],
                got: shape.to_vec(),
            });
        }

        if num_heads == 0 || !hidden_dim.is_multiple_of(num_heads) {
            return Err(NyError::InvalidSpec(format!(
                "hidden_dim {} not divisible by num_heads {}",
                hidden_dim, num_heads
            )));
        }

        let head_dim = hidden_dim / num_heads;

        // Determine naming pattern
        let prefix = if self.num_blocks == 1 && !self.has_layer("/blocks.0/self_attn/q_proj/MatMul")
        {
            String::new()
        } else {
            format!("/blocks.{}", block_index)
        };

        let norm1_name = if prefix.is_empty() {
            "/norm1".to_string()
        } else {
            format!("{}/norm1", prefix)
        };

        let self_attn_prefix = if prefix.is_empty() {
            "/self_attn".to_string()
        } else {
            format!("{}/self_attn", prefix)
        };

        // Get LayerNorm weights
        let (ln_gamma, ln_beta, ln_eps) = self.get_decoder_layer_norm_weights(&norm1_name)?;

        // Get Q, K, V projection weights
        let (q_weight, q_bias) =
            self.get_decoder_linear_weights(&format!("{}/q_proj", self_attn_prefix))?;
        let (k_weight, k_bias) =
            self.get_decoder_linear_weights(&format!("{}/k_proj", self_attn_prefix))?;
        let (v_weight, v_bias) =
            self.get_decoder_linear_weights(&format!("{}/v_proj", self_attn_prefix))?;

        // Get output projection weights
        let (out_weight, out_bias) =
            self.get_decoder_linear_weights(&format!("{}/out_proj", self_attn_prefix))?;

        // Step 1: Apply LayerNorm (forward-mode for tighter bounds, matching whisper default)
        let ln_layer = LayerNormLayer::new_forward_mode(ln_gamma, ln_beta, ln_eps)?;
        let ln_output = ln_layer.propagate_ibp(input)?;

        // Step 2: Apply Q, K, V projections (CPU linear IBP)
        let cpu_device = AcceleratedDevice::new();

        let q_proj = cpu_device.linear_ibp(&ln_output, &q_weight, q_bias.as_ref())?;
        let k_proj = cpu_device.linear_ibp(&ln_output, &k_weight, k_bias.as_ref())?;
        let v_proj = cpu_device.linear_ibp(&ln_output, &v_weight, v_bias.as_ref())?;

        // Step 3: Reshape [batch, seq, hidden] -> [batch, seq, heads, dim]
        let q_4d = q_proj.reshape(&[batch, seq, num_heads, head_dim])?;
        let k_4d = k_proj.reshape(&[batch, seq, num_heads, head_dim])?;
        let v_4d = v_proj.reshape(&[batch, seq, num_heads, head_dim])?;

        // Step 4: Transpose [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        let q_bhsd = q_4d.transpose(&[0, 2, 1, 3])?;
        let k_bhsd = k_4d.transpose(&[0, 2, 1, 3])?;
        let v_bhsd = v_4d.transpose(&[0, 2, 1, 3])?;

        // Step 5: GPU causal attention (hybrid: GPU matmul, CPU causal softmax)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_output = gpu.causal_attention_ibp(&q_bhsd, &k_bhsd, &v_bhsd, scale)?;

        // Step 6: Transpose back [batch, heads, seq, dim] -> [batch, seq, heads, dim]
        let attn_bshd = attn_output.transpose(&[0, 2, 1, 3])?;

        // Step 7: Reshape [batch, seq, heads, dim] -> [batch, seq, hidden]
        let attn_flat = attn_bshd.reshape(&[batch, seq, hidden_dim])?;

        // Step 8: Apply output projection
        let attn_delta = cpu_device.linear_ibp(&attn_flat, &out_weight, out_bias.as_ref())?;

        Ok(attn_delta)
    }

    /// Verify multiple decoder blocks sequentially with GPU acceleration.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor [batch, seq, hidden]
    /// * `start_block` - First block to verify (0-indexed)
    /// * `end_block` - Last block to verify (exclusive)
    /// * `gpu_device` - Optional GPU device for acceleration
    ///
    /// # Returns
    /// (output_bounds, details) with per-block GPU usage information.
    pub fn verify_sequential_gpu(
        &self,
        input: &ny_tensor::BoundedTensor,
        start_block: usize,
        end_block: usize,
        gpu_device: Option<&ComputeDevice>,
    ) -> Result<(ny_tensor::BoundedTensor, Vec<GpuCompositionalDetails>)> {
        if start_block >= end_block {
            return Err(NyError::InvalidSpec(format!(
                "Invalid block range: start {} >= end {}",
                start_block, end_block
            )));
        }
        if end_block > self.num_blocks {
            return Err(NyError::InvalidSpec(format!(
                "Block {} out of range (max {})",
                end_block, self.num_blocks
            )));
        }

        let mut current_bounds = input.clone();
        let mut block_details = Vec::with_capacity(end_block - start_block);

        for block_idx in start_block..end_block {
            let (block_output, details) =
                self.verify_block_compositional_gpu(block_idx, &current_bounds, gpu_device)?;

            info!(
                "Decoder block {} output: max_width {:.2e}, attn_delta {:.2e}, mlp_delta {:.2e}, gpu={}",
                block_idx,
                details.output_width,
                details.attention_delta_width,
                details.mlp_delta_width,
                details.used_gpu_attention,
            );

            block_details.push(details);
            current_bounds = block_output;
        }

        Ok((current_bounds, block_details))
    }
}
