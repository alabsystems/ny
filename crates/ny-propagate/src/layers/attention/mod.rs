// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Self-attention layer for bound propagation.

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use super::binary_ops::MatMulLayer;
use super::common::BoundPropagation;
use super::softmax::{CausalSoftmaxLayer, SoftmaxLayer};
use crate::LinearBounds;

mod crown_ternary;

/// Attention mask behavior for self-attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    /// Standard softmax (bidirectional attention).
    Standard,
    /// Causal softmax (masked to j <= i).
    Causal,
}

/// Self-attention layer: softmax((Q @ K^T) * scale) @ V.
///
/// This layer is a ternary op that consumes Q, K, and V tensors.
#[derive(Debug, Clone)]
pub struct SelfAttentionLayer {
    /// Attention masking behavior.
    pub mask: AttentionMask,
    /// Optional scaling factor (e.g., 1/sqrt(d_k)).
    pub scale: Option<f32>,
    /// Optional causal sliding window size. Ignored for standard attention.
    pub window_size: Option<usize>,
}

impl SelfAttentionLayer {
    /// Create a new SelfAttention layer.
    pub fn new(mask: AttentionMask, scale: Option<f32>) -> Self {
        Self {
            mask,
            scale,
            window_size: None,
        }
    }

    /// Standard (bidirectional) self-attention.
    pub fn standard() -> Self {
        Self::new(AttentionMask::Standard, None)
    }

    /// Causal (masked) self-attention.
    pub fn causal() -> Self {
        Self::new(AttentionMask::Causal, None)
    }

    /// Restrict causal attention to a sliding window.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = Some(window_size);
        self
    }

    fn resolve_scale(&self, q: &BoundedTensor) -> Result<f32> {
        if let Some(scale) = self.scale {
            return Ok(scale);
        }
        let shape = q.shape();
        if shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "SelfAttention requires at least 2D input for scale inference".to_string(),
            ));
        }
        let head_dim = *shape.last().unwrap_or(&0);
        if head_dim == 0 {
            return Err(NyError::InvalidSpec(
                "SelfAttention cannot infer scale with zero head_dim".to_string(),
            ));
        }
        if head_dim > (1 << 24) {
            // f32 precision guard (#2136)
            return Err(NyError::InternalError(format!(
                "head_dim {head_dim} exceeds f32 exact integer range"
            )));
        }
        Ok(1.0 / (head_dim as f32).sqrt())
    }

    /// Propagate IBP bounds through self-attention using Q/K/V inputs.
    pub fn propagate_ibp_ternary(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let scale = self.resolve_scale(q)?;

        // Q @ K^T (scaled)
        let qk = MatMulLayer::new(true, Some(scale)).propagate_ibp_binary(q, k)?;

        // Softmax (causal or standard)
        let probs = match self.mask {
            AttentionMask::Standard => SoftmaxLayer::new(-1).propagate_ibp(&qk)?,
            AttentionMask::Causal => {
                let layer = match self.window_size {
                    Some(window_size) => CausalSoftmaxLayer::new(-1).with_window_size(window_size),
                    None => CausalSoftmaxLayer::new(-1),
                };
                layer.propagate_ibp(&qk)?
            }
        };

        // probs @ V — term-wise interval matmul, then tighten with the softmax
        // sum-to-1 ("simplex water-filling") lever. The softmax rows of `probs`
        // are row-stochastic (sum to 1, non-negative), so each output is a convex
        // combination of V's rows; exploiting that is strictly tighter than the
        // term-wise IBP (which drops the sum-to-1 constraint). Sound by
        // construction — see `softmax::simplex_v` — and a no-op when the simplex
        // structure is unavailable. transpose_b=false: V is (.., K, N). (#softmax-V-lever)
        let out_ibp = MatMulLayer::new(false, None).propagate_ibp_binary(&probs, v)?;
        Ok(crate::layers::softmax::tighten_softmax_v_ibp(
            &probs, v, &out_ibp, false,
        ))
    }
}

impl BoundPropagation for SelfAttentionLayer {
    fn propagate_ibp(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        Err(NyError::UnsupportedOp(
            "SelfAttention requires 3 inputs (Q, K, V); use propagate_ibp_ternary".to_string(),
        ))
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "SelfAttention CROWN propagation not implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    /// Helper: build a 2D BoundedTensor from flat lower/upper vecs and shape.
    fn bounded_2d(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
        let l = ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap();
        BoundedTensor::new(l, u).unwrap()
    }

    // =========================================================================
    // Constructor tests
    // =========================================================================

    /// SelfAttentionLayer::standard() creates standard mask with no scale.
    #[test]
    fn test_standard_constructor() {
        let layer = SelfAttentionLayer::standard();
        assert_eq!(layer.mask, AttentionMask::Standard);
        assert!(layer.scale.is_none());
        assert!(layer.window_size.is_none());
    }

    /// SelfAttentionLayer::causal() creates causal mask with no scale.
    #[test]
    fn test_causal_constructor() {
        let layer = SelfAttentionLayer::causal();
        assert_eq!(layer.mask, AttentionMask::Causal);
        assert!(layer.scale.is_none());
        assert!(layer.window_size.is_none());
    }

    /// Custom constructor with explicit scale.
    #[test]
    fn test_custom_constructor_with_scale() {
        let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(0.125));
        assert_eq!(layer.mask, AttentionMask::Standard);
        assert_eq!(layer.scale, Some(0.125));
        assert!(layer.window_size.is_none());
    }

    /// Sliding-window builder stores the requested window size.
    #[test]
    fn test_with_window_size_sets_window() {
        let layer = SelfAttentionLayer::causal().with_window_size(2);
        assert_eq!(layer.mask, AttentionMask::Causal);
        assert_eq!(layer.window_size, Some(2));
    }

    // =========================================================================
    // resolve_scale
    // =========================================================================

    /// resolve_scale infers 1/sqrt(head_dim) from last dimension of Q.
    #[test]
    fn test_resolve_scale_inferred() {
        let layer = SelfAttentionLayer::standard();
        // Q shape [2, 4] => head_dim = 4, scale = 1/sqrt(4) = 0.5
        let q = bounded_2d(&[2, 4], vec![0.0; 8], vec![1.0; 8]);
        let scale = layer.resolve_scale(&q).unwrap();
        assert!(
            (scale - 0.5).abs() < 1e-6,
            "Expected scale=0.5 for head_dim=4, got {scale}"
        );
    }

    /// resolve_scale returns explicit scale when set.
    #[test]
    fn test_resolve_scale_explicit() {
        let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(0.25));
        let q = bounded_2d(&[2, 4], vec![0.0; 8], vec![1.0; 8]);
        let scale = layer.resolve_scale(&q).unwrap();
        assert_eq!(scale, 0.25);
    }

    /// resolve_scale errors on 1D input (cannot infer head_dim).
    #[test]
    fn test_resolve_scale_1d_error() {
        let layer = SelfAttentionLayer::standard();
        let q = bounded_2d(&[4], vec![0.0; 4], vec![1.0; 4]);
        let err = layer.resolve_scale(&q).unwrap_err();
        assert!(
            format!("{err}").contains("at least 2D"),
            "Expected 2D error, got: {err}"
        );
    }

    // =========================================================================
    // BoundPropagation trait — expected errors
    // =========================================================================

    /// propagate_ibp (unary) should error with UnsupportedOp.
    #[test]
    fn test_propagate_ibp_unary_errors() {
        let layer = SelfAttentionLayer::standard();
        let input = bounded_2d(&[2, 2], vec![0.0; 4], vec![1.0; 4]);
        let err = layer.propagate_ibp(&input).unwrap_err();
        assert!(
            format!("{err}").contains("3 inputs"),
            "Expected ternary-required error, got: {err}"
        );
    }

    /// The single-input trait `propagate_linear` still errors: self-attention is
    /// a ternary op (Q, K, V), so its CROWN backward is the dedicated
    /// `propagate_crown_ternary` (returning three `LinearBounds`, dispatched as
    /// `BackwardDispatchResult::Nary`), NOT the single-`LinearBounds` trait
    /// method. The trait method cannot express a 3-input backward and so remains
    /// `UnsupportedOp` by design. See `crown_ternary` for the sound ternary path.
    #[test]
    fn test_propagate_linear_errors() {
        let layer = SelfAttentionLayer::standard();
        let bounds = LinearBounds::new(
            ndarray::Array2::eye(2),
            ndarray::Array1::zeros(2),
            ndarray::Array2::eye(2),
            ndarray::Array1::zeros(2),
        )
        .unwrap();
        let err = layer.propagate_linear(&bounds).unwrap_err();
        assert!(
            format!("{err}").contains("CROWN propagation not implemented"),
            "Expected CROWN unsupported error, got: {err}"
        );
    }

    /// The ternary CROWN backward IS implemented and produces sound, enclosing
    /// bounds for the fused attention. Concrete (zero-width) Q/K/V must pin the
    /// output: the three per-input `LinearBounds` + shared bias, concretized over
    /// the (degenerate) box, reproduce the exact attention output. This guards
    /// the wiring; exhaustive interior-sampling soundness lives in
    /// `crown_ternary::tests`.
    #[test]
    fn test_propagate_crown_ternary_concrete_pins_output() {
        let scale = 1.0 / 2.0_f32.sqrt();
        let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(scale));
        let qv = vec![0.1, 0.2, -0.3, 0.4];
        let kv = vec![0.2, -0.1, 0.3, 0.2];
        let vv = vec![1.0, -0.5, 0.3, 0.7];
        let q = bounded_2d(&[2, 2], qv.clone(), qv);
        let k = bounded_2d(&[2, 2], kv.clone(), kv);
        let v = bounded_2d(&[2, 2], vv.clone(), vv);

        // Identity upstream over the flattened Y (size 4).
        let node_lb = LinearBounds::new(
            ndarray::Array2::eye(4),
            ndarray::Array1::zeros(4),
            ndarray::Array2::eye(4),
            ndarray::Array1::zeros(4),
        )
        .unwrap();
        let (bounds, blo, bhi) = layer
            .propagate_crown_ternary(&node_lb, &q, &k, &v)
            .expect("ternary CROWN should succeed");
        assert_eq!(bounds.len(), 3, "Q/K/V per-input bounds");

        // True attention output.
        let true_out = layer.propagate_ibp_ternary(&q, &k, &v).unwrap();
        let to = true_out.flatten();
        let to_lo = to.lower();
        let to_lo = to_lo.as_slice().unwrap();

        // Concretize the Nary result over the (degenerate) box: lower/upper should
        // both equal the true output. With zero-width boxes, A·x is exact at the
        // single point, so bias + A·point pins the value.
        let boxes = [&q, &k, &v];
        for o in 0..4 {
            let mut lo = blo[o] as f64;
            let mut hi = bhi[o] as f64;
            for (bi, lb) in bounds.iter().enumerate() {
                let lb = lb.as_ref().unwrap();
                let bx = boxes[bi].flatten();
                let pt = bx.lower();
                let pt = pt.as_slice().unwrap();
                for j in 0..lb.num_inputs() {
                    lo += lb.lower_a()[[o, j]] as f64 * pt[j] as f64;
                    hi += lb.upper_a()[[o, j]] as f64 * pt[j] as f64;
                }
            }
            assert!(
                (lo as f32 - to_lo[o]).abs() < 2e-3 && (hi as f32 - to_lo[o]).abs() < 2e-3,
                "concrete ternary out={o}: bound=[{lo}, {hi}] true={}",
                to_lo[o]
            );
        }
    }

    // =========================================================================
    // propagate_ibp_ternary — standard attention
    // =========================================================================

    /// Standard attention: concrete Q, K, V with known values.
    /// Q = K = V = [[1, 0], [0, 1]] (identity-like), scale = 1/sqrt(2).
    /// Q @ K^T = I (when Q=K=identity), softmax(I * scale) produces attention probs,
    /// then probs @ V produces output.
    #[test]
    fn test_standard_attention_concrete() {
        let layer = SelfAttentionLayer::standard();

        // 2x2 identity-like Q, K, V (concrete: lower == upper)
        let vals = vec![1.0, 0.0, 0.0, 1.0];
        let q = bounded_2d(&[2, 2], vals.clone(), vals.clone());
        let k = bounded_2d(&[2, 2], vals.clone(), vals.clone());
        let v = bounded_2d(&[2, 2], vals.clone(), vals);

        let output = layer.propagate_ibp_ternary(&q, &k, &v).unwrap();

        // Output shape should be [2, 2] (seq=2, head_dim=2)
        assert_eq!(output.shape(), &[2, 2]);

        // Output should be valid bounds (lower <= upper)
        let lower = output.lower();
        let upper = output.upper();
        for (l, u) in lower.iter().zip(upper.iter()) {
            assert!(l <= u, "Invalid bounds: lower={l} > upper={u}");
        }

        // Since inputs are concrete (no perturbation), lower should be close to upper
        for (l, u) in lower.iter().zip(upper.iter()) {
            assert!(
                (u - l).abs() < 1e-4,
                "Concrete input should give tight bounds: l={l}, u={u}"
            );
        }
    }

    /// Sliding window size 0 yields self-only causal attention, so concrete output equals V.
    #[test]
    fn test_causal_attention_self_only_window_matches_v() {
        let layer = SelfAttentionLayer::causal().with_window_size(0);

        let q_vals = vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.5];
        let k_vals = vec![0.3, -0.2, 0.6, 0.1, -0.4, 0.8];
        let v_vals = vec![0.2, 0.4, -0.3, 0.7, 1.1, -0.9];
        let q = bounded_2d(&[3, 2], q_vals.clone(), q_vals);
        let k = bounded_2d(&[3, 2], k_vals.clone(), k_vals);
        let v = bounded_2d(&[3, 2], v_vals.clone(), v_vals.clone());

        let output = layer.propagate_ibp_ternary(&q, &k, &v).unwrap();

        for (idx, (&lo, &hi)) in output.lower().iter().zip(output.upper().iter()).enumerate() {
            assert!(
                (lo - v_vals[idx]).abs() < 1e-5,
                "lower[{idx}] should match V exactly"
            );
            assert!(
                (hi - v_vals[idx]).abs() < 1e-5,
                "upper[{idx}] should match V exactly"
            );
        }
    }

    /// Standard attention with perturbed inputs: output bounds should contain
    /// any concrete output from inputs in the perturbation range.
    #[test]
    fn test_standard_attention_perturbed_soundness() {
        let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(0.5));

        // Small 2x2 inputs with perturbation
        let q = bounded_2d(&[2, 2], vec![0.5, 0.0, 0.0, 0.5], vec![1.5, 1.0, 1.0, 1.5]);
        let k = bounded_2d(&[2, 2], vec![0.5, 0.0, 0.0, 0.5], vec![1.5, 1.0, 1.0, 1.5]);
        let v = bounded_2d(&[2, 2], vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);

        let output = layer.propagate_ibp_ternary(&q, &k, &v).unwrap();
        assert_eq!(output.shape(), &[2, 2]);

        // Bounds should be valid
        for (l, u) in output.lower().iter().zip(output.upper().iter()) {
            assert!(l <= u, "Invalid bounds: lower={l} > upper={u}");
        }

        // Attention output with softmax should have non-negative lower bounds
        // since softmax produces [0,1] probs and V is non-negative
        for l in output.lower().iter() {
            assert!(
                *l >= -1e-4,
                "With non-negative V, attention output lower bound should be >= 0, got {l}"
            );
        }
    }

    // =========================================================================
    // propagate_ibp_ternary — causal attention
    // =========================================================================

    /// Causal attention: 2x2 Q, K, V. CausalSoftmax masks upper-triangle.
    #[test]
    fn test_causal_attention_concrete() {
        let layer = SelfAttentionLayer::causal();

        // Concrete 2x2 inputs
        let vals = vec![1.0, 0.0, 0.0, 1.0];
        let q = bounded_2d(&[2, 2], vals.clone(), vals.clone());
        let k = bounded_2d(&[2, 2], vals.clone(), vals.clone());
        let v = bounded_2d(&[2, 2], vals.clone(), vals);

        let output = layer.propagate_ibp_ternary(&q, &k, &v).unwrap();
        assert_eq!(output.shape(), &[2, 2]);

        // Bounds should be valid
        for (l, u) in output.lower().iter().zip(output.upper().iter()) {
            assert!(l <= u, "Invalid bounds: lower={l} > upper={u}");
        }
    }

    /// Causal attention with perturbation.
    #[test]
    fn test_causal_attention_perturbed() {
        let layer = SelfAttentionLayer::new(AttentionMask::Causal, Some(0.5));

        let q = bounded_2d(&[2, 2], vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);
        let k = bounded_2d(&[2, 2], vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);
        let v = bounded_2d(&[2, 2], vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);

        let output = layer.propagate_ibp_ternary(&q, &k, &v).unwrap();
        assert_eq!(output.shape(), &[2, 2]);

        for (l, u) in output.lower().iter().zip(output.upper().iter()) {
            assert!(l <= u, "Invalid bounds: lower={l} > upper={u}");
        }
    }

    // =========================================================================
    // Edge cases / errors
    // =========================================================================

    /// Mismatched Q and K shapes should propagate error from MatMul.
    #[test]
    fn test_attention_shape_mismatch_qk() {
        let layer = SelfAttentionLayer::standard();
        let q = bounded_2d(&[2, 3], vec![0.0; 6], vec![1.0; 6]);
        let k = bounded_2d(&[2, 4], vec![0.0; 8], vec![1.0; 8]); // head_dim mismatch
        let v = bounded_2d(&[2, 3], vec![0.0; 6], vec![1.0; 6]);

        // Q @ K^T requires Q columns == K columns (since transpose_b=true)
        // Q is [2,3], K is [2,4], K^T is [4,2], so Q[2,3] @ K^T[4,2] fails
        let err = layer.propagate_ibp_ternary(&q, &k, &v).unwrap_err();
        assert!(
            !format!("{err}").is_empty(),
            "Expected shape mismatch error from Q @ K^T"
        );
    }

    /// 3D inputs (batched attention).
    #[test]
    fn test_standard_attention_3d() {
        let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(0.5));

        // [batch=1, seq=2, head_dim=2]
        let q = bounded_2d(&[1, 2, 2], vec![0.0; 4], vec![1.0; 4]);
        let k = bounded_2d(&[1, 2, 2], vec![0.0; 4], vec![1.0; 4]);
        let v = bounded_2d(&[1, 2, 2], vec![0.0; 4], vec![1.0; 4]);

        let output = layer.propagate_ibp_ternary(&q, &k, &v).unwrap();
        // MatMul with batch dim should produce [1, 2, 2]
        assert_eq!(output.shape(), &[1, 2, 2]);

        for (l, u) in output.lower().iter().zip(output.upper().iter()) {
            assert!(l <= u, "Invalid bounds: lower={l} > upper={u}");
        }
    }
}
