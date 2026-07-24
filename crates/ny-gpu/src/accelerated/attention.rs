// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Attention IBP operations on `AcceleratedDevice`.
//!
//! Provides standard, causal, and cross-attention IBP using CPU-parallel
//! matmul and sequential softmax.

use ny_core::{NyError, Result};
use ny_propagate::{
    layers::{CausalSoftmaxLayer, SoftmaxLayer},
    BoundPropagation,
};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::AcceleratedBoundPropagation;

impl super::AcceleratedDevice {
    /// Full attention IBP using CPU (Rayon parallelized matmul, sequential softmax).
    ///
    /// Computes: softmax((Q @ K^T) * scale) @ V
    ///
    /// Input shapes: Q, K, V with shape [batch, heads, seq, dim]
    /// Output shape: [batch, heads, seq, dim]
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
            "AcceleratedDevice attention_ibp: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Step 1: Compute K^T
        // K shape: [batch, heads, seq, dim]
        // K^T shape: [batch, heads, dim, seq]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using Rayon parallel matmul
        // Q: [batch, heads, seq, dim] @ K^T: [batch, heads, dim, seq]
        // -> scores: [batch, heads, seq, seq]
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Softmax over last dimension (seq)
        let probs = SoftmaxLayer::new(-1).propagate_ibp(&scores_scaled)?;

        // Step 5: probs @ V using Rayon parallel matmul
        // probs: [batch, heads, seq, seq] @ V: [batch, heads, seq, dim]
        // -> output: [batch, heads, seq, dim]
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Causal attention IBP using CPU (Rayon parallelized matmul).
    ///
    /// Computes: causal_softmax((Q @ K^T) * scale) @ V
    ///
    /// This is for decoder-only (LLaMA, GPT) and decoder blocks (Whisper decoder).
    /// Position i can only attend to positions j where j <= i.
    ///
    /// Input shapes: Q, K, V with shape [batch, heads, seq, dim]
    /// Output shape: [batch, heads, seq, dim]
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
            "AcceleratedDevice causal_attention_ibp: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Step 1: Compute K^T
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using Rayon parallel matmul
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Causal softmax over last dimension
        // This applies the lower-triangular mask: position i attends only to j <= i
        let probs = CausalSoftmaxLayer::new(-1).propagate_ibp(&scores_scaled)?;

        // Step 5: probs @ V using Rayon parallel matmul
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Cross-attention IBP for encoder-decoder models (e.g., Whisper).
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
            "AcceleratedDevice cross_attention_ibp: batch={}, heads={}, seq_dec={}, seq_enc={}, dim={}, scale={}",
            batch, heads, seq_dec, seq_enc, dim, scale
        );

        // Step 1: Compute K^T
        // K shape: [batch, heads, seq_enc, dim]
        // K^T shape: [batch, heads, dim, seq_enc]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using Rayon parallel matmul
        // Q: [batch, heads, seq_dec, dim] @ K^T: [batch, heads, dim, seq_enc]
        // -> scores: [batch, heads, seq_dec, seq_enc]
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Standard softmax (no causal mask - decoder can attend to all encoder positions)
        // Softmax over last dimension (seq_enc)
        let probs = SoftmaxLayer::new(-1).propagate_ibp(&scores_scaled)?;

        // Step 5: probs @ V using Rayon parallel matmul
        // probs: [batch, heads, seq_dec, seq_enc] @ V: [batch, heads, seq_enc, dim]
        // -> output: [batch, heads, seq_dec, dim]
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }
}
