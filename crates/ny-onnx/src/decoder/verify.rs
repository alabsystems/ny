// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CPU-based compositional decoder block verification.

use ny_core::{NyError, Result};
use tracing::info;

use super::{DecoderModel, DecoderVerificationDetails};

impl DecoderModel {
    /// Compositional verification of a decoder block using IBP.
    ///
    /// Algorithm:
    /// 1. Bound causal self-attention subgraph with IBP
    /// 2. Compose with first residual: x_attn = x + attn_delta
    /// 3. Bound MLP subgraph with IBP
    /// 4. Compose with second residual: x_out = x_attn + mlp_delta
    ///
    /// # Arguments
    /// * `block_index` - Index of the decoder block
    /// * `input` - Input bounded tensor [batch, seq, hidden]
    ///
    /// # Returns
    /// (output_bounds, details) with intermediate information.
    pub fn verify_block_compositional(
        &self,
        block_index: usize,
        input: &ny_tensor::BoundedTensor,
    ) -> Result<(ny_tensor::BoundedTensor, DecoderVerificationDetails)> {
        // Step 1: Bound causal self-attention subgraph
        let attn_graph = self.causal_attention_subgraph(block_index)?;
        let attn_delta = attn_graph.propagate_ibp(input)?;

        // Step 2: Compose with first residual
        let x_attn = input.add(&attn_delta)?;

        // Step 3: Bound MLP subgraph
        let mlp_graph = self.mlp_subgraph(block_index)?;
        let mlp_delta = mlp_graph.propagate_ibp(&x_attn)?;

        // Step 4: Compose with second residual
        let x_out = x_attn.add(&mlp_delta)?;

        let details = DecoderVerificationDetails {
            attention_delta_width: attn_delta.max_width(),
            x_attn_width: x_attn.max_width(),
            mlp_delta_width: mlp_delta.max_width(),
            output_width: x_out.max_width(),
        };

        Ok((x_out, details))
    }

    /// Verify multiple decoder blocks sequentially.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor [batch, seq, hidden]
    /// * `start_block` - First block to verify (0-indexed)
    /// * `end_block` - Last block to verify (exclusive)
    ///
    /// # Returns
    /// (output_bounds, details) with per-block information.
    pub fn verify_sequential(
        &self,
        input: &ny_tensor::BoundedTensor,
        start_block: usize,
        end_block: usize,
    ) -> Result<(ny_tensor::BoundedTensor, Vec<DecoderVerificationDetails>)> {
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
                self.verify_block_compositional(block_idx, &current_bounds)?;

            info!(
                "Decoder block {} output: max_width {:.2e}, attn_delta {:.2e}, mlp_delta {:.2e}",
                block_idx,
                details.output_width,
                details.attention_delta_width,
                details.mlp_delta_width
            );

            block_details.push(details);
            current_bounds = block_output;
        }

        Ok((current_bounds, block_details))
    }
}
