// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{CausalSoftmaxLayer, SoftmaxLayer};
use ny_propagate::BoundPropagation;
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::super::super::WgpuDevice;

impl WgpuDevice {
    /// Chained attention IBP: Q @ K^T -> scale -> softmax -> probs @ V
    ///
    /// This method chains all attention operations on the GPU without intermediate
    /// host roundtrips, providing significant speedup compared to separate calls.
    ///
    /// # Arguments
    /// * `q` - Query tensor with shape [batch, heads, seq, dim]
    /// * `k` - Key tensor with shape [batch, heads, seq, dim]
    /// * `v` - Value tensor with shape [batch, heads, seq, dim]
    /// * `scale` - Scaling factor (typically 1.0 / sqrt(dim))
    ///
    /// # Returns
    /// Output tensor with shape [batch, heads, seq, dim]
    pub fn attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        let shape_q = q.shape();
        let shape_k = k.shape();
        let shape_v = v.shape();

        // Validate shapes: all should be [batch, heads, seq, dim]
        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(NyError::InvalidSpec(
                "Attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        // Verify Q, K, V shapes are compatible
        if shape_q != shape_k {
            return Err(NyError::shape_mismatch(shape_q.to_vec(), shape_k.to_vec()));
        }
        if shape_q != shape_v {
            return Err(NyError::shape_mismatch(shape_q.to_vec(), shape_v.to_vec()));
        }

        let batch = shape_q[0];
        let heads = shape_q[1];
        let seq = shape_q[2];
        let dim = shape_q[3];

        debug!(
            "WgpuDevice attention_ibp: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Step 1: Compute K^T
        // K shape: [batch, heads, seq, dim]
        // K^T shape: [batch, heads, dim, seq]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T
        // Q: [batch, heads, seq, dim] @ K^T: [batch, heads, dim, seq]
        // -> scores: [batch, heads, seq, seq]
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Softmax over last dimension (seq)
        let probs = self.softmax_ibp(&scores_scaled)?;

        // Step 5: probs @ V
        // probs: [batch, heads, seq, seq] @ V: [batch, heads, seq, dim]
        // -> output: [batch, heads, seq, dim]
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Causal attention IBP with GPU acceleration (hybrid: GPU matmul, CPU causal softmax).
    ///
    /// Computes: causal_softmax((Q @ K^T) * scale) @ V
    ///
    /// This is for decoder-only (LLaMA, GPT) and decoder blocks (Whisper decoder).
    /// Position i can only attend to positions j where j <= i.
    ///
    /// # Arguments
    /// * `q` - Query tensor with shape [batch, heads, seq, dim]
    /// * `k` - Key tensor with shape [batch, heads, seq, dim]
    /// * `v` - Value tensor with shape [batch, heads, seq, dim]
    /// * `scale` - Scaling factor (typically 1.0 / sqrt(dim))
    ///
    /// # Returns
    /// Output tensor with shape [batch, heads, seq, dim]
    ///
    /// # Implementation Note
    /// Uses GPU for matmul operations but CPU for causal softmax since the
    /// causal mask requires per-row varying normalization.
    pub fn causal_attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        let shape_q = q.shape();
        let shape_k = k.shape();
        let shape_v = v.shape();

        // Validate shapes: all should be [batch, heads, seq, dim]
        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(NyError::InvalidSpec(
                "Attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        // Verify Q, K, V shapes are compatible
        if shape_q != shape_k {
            return Err(NyError::shape_mismatch(shape_q.to_vec(), shape_k.to_vec()));
        }
        if shape_q != shape_v {
            return Err(NyError::shape_mismatch(shape_q.to_vec(), shape_v.to_vec()));
        }

        let batch = shape_q[0];
        let heads = shape_q[1];
        let seq = shape_q[2];
        let dim = shape_q[3];

        debug!(
            "WgpuDevice causal_attention_ibp: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Step 1: Compute K^T on CPU (fast transpose operation)
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using GPU matmul
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Causal softmax using CPU
        // This applies the lower-triangular mask: position i attends only to j <= i
        let probs = CausalSoftmaxLayer::new(-1).propagate_ibp(&scores_scaled)?;

        // Step 5: probs @ V using GPU matmul
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Cross-attention IBP for encoder-decoder models (e.g., Whisper) using GPU.
    ///
    /// In cross-attention:
    /// - Q (queries) comes from the decoder with shape [batch, heads, seq_dec, dim]
    /// - K, V (keys, values) come from the encoder with shape [batch, heads, seq_enc, dim]
    /// - Output has shape [batch, heads, seq_dec, dim]
    ///
    /// Unlike causal self-attention, cross-attention has NO causal mask:
    /// decoder positions can attend to ALL encoder positions.
    ///
    /// Computes: softmax((Q @ K^T) * scale) @ V
    ///
    /// Uses GPU for matmul operations, CPU for standard softmax.
    pub fn cross_attention_ibp(
        &self,
        q: &BoundedTensor, // [batch, heads, seq_dec, dim]
        k: &BoundedTensor, // [batch, heads, seq_enc, dim]
        v: &BoundedTensor, // [batch, heads, seq_enc, dim]
        scale: f32,
    ) -> Result<BoundedTensor> {
        let shape_q = q.shape();
        let shape_k = k.shape();
        let shape_v = v.shape();

        // Validate shapes: all should be 4D [batch, heads, seq, dim]
        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(NyError::InvalidSpec(
                "Cross-attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        // Verify batch and heads match
        if shape_q[0] != shape_k[0] || shape_q[0] != shape_v[0] {
            return Err(NyError::shape_mismatch(
                vec![shape_q[0]],
                vec![shape_k[0], shape_v[0]],
            ));
        }
        if shape_q[1] != shape_k[1] || shape_q[1] != shape_v[1] {
            return Err(NyError::shape_mismatch(
                vec![shape_q[1]],
                vec![shape_k[1], shape_v[1]],
            ));
        }

        // Verify K and V have same sequence length (encoder sequence)
        if shape_k[2] != shape_v[2] {
            return Err(NyError::shape_mismatch(vec![shape_k[2]], vec![shape_v[2]]));
        }

        // Verify dim matches for Q and K (needed for Q @ K^T)
        if shape_q[3] != shape_k[3] {
            return Err(NyError::shape_mismatch(vec![shape_q[3]], vec![shape_k[3]]));
        }
        if shape_k[3] != shape_v[3] {
            return Err(NyError::shape_mismatch(vec![shape_k[3]], vec![shape_v[3]]));
        }

        let batch = shape_q[0];
        let heads = shape_q[1];
        let seq_dec = shape_q[2];
        let seq_enc = shape_k[2];
        let dim = shape_q[3];

        debug!(
            "WgpuDevice cross_attention_ibp: batch={}, heads={}, seq_dec={}, seq_enc={}, dim={}, scale={}",
            batch, heads, seq_dec, seq_enc, dim, scale
        );

        // Step 1: Compute K^T on CPU (fast transpose operation)
        // K shape: [batch, heads, seq_enc, dim]
        // K^T shape: [batch, heads, dim, seq_enc]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using GPU matmul
        // Q: [batch, heads, seq_dec, dim] @ K^T: [batch, heads, dim, seq_enc]
        // -> scores: [batch, heads, seq_dec, seq_enc]
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Standard softmax (no causal mask)
        // Softmax over last dimension (seq_enc)
        let probs = SoftmaxLayer::new(-1).propagate_ibp(&scores_scaled)?;

        // Step 5: probs @ V using GPU matmul
        // probs: [batch, heads, seq_dec, seq_enc] @ V: [batch, heads, seq_enc, dim]
        // -> output: [batch, heads, seq_dec, dim]
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }
}
