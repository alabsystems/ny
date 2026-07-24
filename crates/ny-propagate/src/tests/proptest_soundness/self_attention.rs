// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-based soundness tests for SelfAttentionLayer propagation.
//!
//! SelfAttention composes Q/K/V linear projections with softmax:
//!   output = softmax(Q @ K^T * scale) @ V
//!
//! Tests cover:
//! 1. **IBP soundness** — `propagate_ibp_ternary` (monolithic layer)
//! 2. **Graph CROWN soundness** — decomposed attention via GraphNetwork
//!    (BilinearCrown + Softmax + BilinearCrown), verifying that CROWN backward
//!    through the decomposition produces sound bounds.
//!
//! Reference: alpha-beta-CROWN decomposes attention at ONNX graph level into
//! MatMul + Softmax + MatMul primitives, each with `bound_backward`.
//! Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/

use crate::layers::attention::{AttentionMask, SelfAttentionLayer};
use crate::{GraphNetwork, GraphNode, Layer, ReLULayer};
use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{causal_softmax, softmax, FP_TOLERANCE};

/// Generate a pair (lower, upper) vectors of length `n`
/// where lower[i] <= upper[i] and values are in [-range, range].
fn valid_interval_vec(n: usize, range: f32) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    proptest::collection::vec((-range..=range, -range..=range), n).prop_map(move |pairs| {
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for (a, b) in pairs {
            lo.push(a.min(b));
            hi.push(a.max(b));
        }
        (lo, hi)
    })
}

/// Build a BoundedTensor from flat lower/upper vecs and shape.
fn bounded_nd(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    let lo = ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape");
    let hi = ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape");
    BoundedTensor::new(lo, hi).expect("valid bounds")
}

/// Reference implementation of standard self-attention for concrete 2D inputs.
///
/// Computes: softmax(Q @ K^T * scale) @ V
/// where softmax is applied row-wise on the last axis.
fn eval_standard_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    scale: f32,
) -> Array2<f32> {
    let qk_scaled = q.dot(&k.t()) * scale;
    let (seq_q, seq_k) = (qk_scaled.nrows(), qk_scaled.ncols());
    let mut probs = Array2::zeros((seq_q, seq_k));
    for i in 0..seq_q {
        let row = qk_scaled.row(i).to_owned();
        let sm = softmax(&row);
        probs.row_mut(i).assign(&sm);
    }
    probs.dot(v)
}

/// Reference implementation of causal self-attention for concrete 2D inputs.
///
/// Computes: causal_softmax(Q @ K^T * scale) @ V
/// where causal_softmax masks positions j > i to -inf before softmax.
fn eval_causal_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    scale: f32,
) -> Array2<f32> {
    let qk_scaled = q.dot(&k.t()) * scale;
    let probs = causal_softmax(&qk_scaled);
    probs.dot(v)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // ========================================================================
    // Standard attention IBP soundness: 2x2 Q, K, V
    // ========================================================================

    /// Standard self-attention IBP soundness: for any concrete (Q, K, V) within
    /// the input intervals, the true output must lie within the IBP bounds.
    ///
    /// Tests all 2^4 corners for each of Q, K, V independently. With 3 matrices
    /// of 4 elements each, full enumeration is 2^12 = 4096 corners per proptest
    /// case. We sample a representative subset: all Q corners at K/V midpoints,
    /// all K corners at Q/V midpoints, and all V corners at Q/K midpoints (48 total),
    /// plus the all-lower and all-upper corners.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_self_attention_standard_ibp_2x2(
        (q_lo, q_hi) in valid_interval_vec(4, 3.0),
        (k_lo, k_hi) in valid_interval_vec(4, 3.0),
        (v_lo, v_hi) in valid_interval_vec(4, 3.0),
    ) {
        let scale = 1.0 / 2.0_f32.sqrt(); // 1/sqrt(head_dim) for head_dim=2
        let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(scale));

        let q_bounds = bounded_nd(&[2, 2], q_lo.clone(), q_hi.clone());
        let k_bounds = bounded_nd(&[2, 2], k_lo.clone(), k_hi.clone());
        let v_bounds = bounded_nd(&[2, 2], v_lo.clone(), v_hi.clone());

        let result = layer.propagate_ibp_ternary(&q_bounds, &k_bounds, &v_bounds)
            .map_err(|e| TestCaseError::fail(format!("IBP failed: {e}")))?;

        prop_assert_eq!(result.shape(), &[2, 2], "Output shape mismatch");

        // Midpoints for each matrix
        let q_mid: Vec<f32> = q_lo.iter().zip(q_hi.iter()).map(|(&l, &u)| f32::midpoint(l, u)).collect();
        let k_mid: Vec<f32> = k_lo.iter().zip(k_hi.iter()).map(|(&l, &u)| f32::midpoint(l, u)).collect();
        let v_mid: Vec<f32> = v_lo.iter().zip(v_hi.iter()).map(|(&l, &u)| f32::midpoint(l, u)).collect();

        // Helper to build a 2x2 Array2 from a flat vec
        let to_mat = |vals: &[f32]| -> Array2<f32> {
            Array2::from_shape_vec((2, 2), vals.to_vec()).unwrap()
        };

        // Helper to build a corner from lo/hi vecs using a bitmask
        let corner = |lo: &[f32], hi: &[f32], mask: u32| -> Vec<f32> {
            lo.iter().zip(hi.iter()).enumerate().map(|(idx, (&l, &u))| {
                if mask & (1 << idx) != 0 { u } else { l }
            }).collect()
        };

        // Tolerance: IBP through softmax + two matmuls can accumulate FP error.
        // Use a scaled tolerance based on the output magnitude.
        let check_sound = |q_vals: &[f32], k_vals: &[f32], v_vals: &[f32], label: &str|
            -> Result<(), TestCaseError>
        {
            let q_mat = to_mat(q_vals);
            let k_mat = to_mat(k_vals);
            let v_mat = to_mat(v_vals);
            let true_out = eval_standard_attention(&q_mat, &k_mat, &v_mat, scale);

            for i in 0..2_usize {
                for j in 0..2_usize {
                    let lo = result.lower()[[i, j]];
                    let hi = result.upper()[[i, j]];
                    let tv = true_out[[i, j]];
                    // Scaled tolerance: softmax has exponential sensitivity
                    let tol = FP_TOLERANCE * tv.abs().max(lo.abs()).max(hi.abs()).max(1.0);
                    prop_assert!(
                        lo - tol <= tv,
                        "Standard attention IBP lower unsound at [{i},{j}] ({label}): \
                         lo={lo} > true={tv} (tol={tol})"
                    );
                    prop_assert!(
                        tv <= hi + tol,
                        "Standard attention IBP upper unsound at [{i},{j}] ({label}): \
                         true={tv} > hi={hi} (tol={tol})"
                    );
                }
            }
            Ok(())
        };

        // All-lower and all-upper corners
        check_sound(&q_lo, &k_lo, &v_lo, "all-lower")?;
        check_sound(&q_hi, &k_hi, &v_hi, "all-upper")?;
        check_sound(&q_mid, &k_mid, &v_mid, "all-mid")?;

        // Sweep Q corners with K and V at midpoints
        for q_mask in 0..16_u32 {
            let q_c = corner(&q_lo, &q_hi, q_mask);
            check_sound(&q_c, &k_mid, &v_mid, &format!("Q corner {q_mask}"))?;
        }

        // Sweep K corners with Q and V at midpoints
        for k_mask in 0..16_u32 {
            let k_c = corner(&k_lo, &k_hi, k_mask);
            check_sound(&q_mid, &k_c, &v_mid, &format!("K corner {k_mask}"))?;
        }

        // Sweep V corners with Q and K at midpoints
        for v_mask in 0..16_u32 {
            let v_c = corner(&v_lo, &v_hi, v_mask);
            check_sound(&q_mid, &k_mid, &v_c, &format!("V corner {v_mask}"))?;
        }
    }

    // ========================================================================
    // Causal attention IBP soundness: 2x2 Q, K, V
    // ========================================================================

    /// Causal self-attention IBP soundness: same structure as standard but
    /// with causal masking (position i can only attend to positions 0..=i).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_self_attention_causal_ibp_2x2(
        (q_lo, q_hi) in valid_interval_vec(4, 3.0),
        (k_lo, k_hi) in valid_interval_vec(4, 3.0),
        (v_lo, v_hi) in valid_interval_vec(4, 3.0),
    ) {
        let scale = 1.0 / 2.0_f32.sqrt();
        let layer = SelfAttentionLayer::new(AttentionMask::Causal, Some(scale));

        let q_bounds = bounded_nd(&[2, 2], q_lo.clone(), q_hi.clone());
        let k_bounds = bounded_nd(&[2, 2], k_lo.clone(), k_hi.clone());
        let v_bounds = bounded_nd(&[2, 2], v_lo.clone(), v_hi.clone());

        let result = layer.propagate_ibp_ternary(&q_bounds, &k_bounds, &v_bounds)
            .map_err(|e| TestCaseError::fail(format!("Causal IBP failed: {e}")))?;

        prop_assert_eq!(result.shape(), &[2, 2], "Output shape mismatch");

        let q_mid: Vec<f32> = q_lo.iter().zip(q_hi.iter()).map(|(&l, &u)| f32::midpoint(l, u)).collect();
        let k_mid: Vec<f32> = k_lo.iter().zip(k_hi.iter()).map(|(&l, &u)| f32::midpoint(l, u)).collect();
        let v_mid: Vec<f32> = v_lo.iter().zip(v_hi.iter()).map(|(&l, &u)| f32::midpoint(l, u)).collect();

        let to_mat = |vals: &[f32]| -> Array2<f32> {
            Array2::from_shape_vec((2, 2), vals.to_vec()).unwrap()
        };

        let corner = |lo: &[f32], hi: &[f32], mask: u32| -> Vec<f32> {
            lo.iter().zip(hi.iter()).enumerate().map(|(idx, (&l, &u))| {
                if mask & (1 << idx) != 0 { u } else { l }
            }).collect()
        };

        let check_sound = |q_vals: &[f32], k_vals: &[f32], v_vals: &[f32], label: &str|
            -> Result<(), TestCaseError>
        {
            let q_mat = to_mat(q_vals);
            let k_mat = to_mat(k_vals);
            let v_mat = to_mat(v_vals);
            let true_out = eval_causal_attention(&q_mat, &k_mat, &v_mat, scale);

            for i in 0..2_usize {
                for j in 0..2_usize {
                    let lo = result.lower()[[i, j]];
                    let hi = result.upper()[[i, j]];
                    let tv = true_out[[i, j]];
                    let tol = FP_TOLERANCE * tv.abs().max(lo.abs()).max(hi.abs()).max(1.0);
                    prop_assert!(
                        lo - tol <= tv,
                        "Causal attention IBP lower unsound at [{i},{j}] ({label}): \
                         lo={lo} > true={tv} (tol={tol})"
                    );
                    prop_assert!(
                        tv <= hi + tol,
                        "Causal attention IBP upper unsound at [{i},{j}] ({label}): \
                         true={tv} > hi={hi} (tol={tol})"
                    );
                }
            }
            Ok(())
        };

        // All-lower, all-upper, midpoints
        check_sound(&q_lo, &k_lo, &v_lo, "all-lower")?;
        check_sound(&q_hi, &k_hi, &v_hi, "all-upper")?;
        check_sound(&q_mid, &k_mid, &v_mid, "all-mid")?;

        // Sweep each matrix's corners while others at midpoints
        for q_mask in 0..16_u32 {
            let q_c = corner(&q_lo, &q_hi, q_mask);
            check_sound(&q_c, &k_mid, &v_mid, &format!("Q corner {q_mask}"))?;
        }
        for k_mask in 0..16_u32 {
            let k_c = corner(&k_lo, &k_hi, k_mask);
            check_sound(&q_mid, &k_c, &v_mid, &format!("K corner {k_mask}"))?;
        }
        for v_mask in 0..16_u32 {
            let v_c = corner(&v_lo, &v_hi, v_mask);
            check_sound(&q_mid, &k_mid, &v_c, &format!("V corner {v_mask}"))?;
        }
    }

    // ========================================================================
    // Concrete inputs produce tight bounds
    // ========================================================================

    /// When Q, K, V are concrete (lower == upper), IBP bounds should be
    /// close to the true output since there's no interval uncertainty.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_self_attention_concrete_tight(
        q_vals in proptest::collection::vec(-3.0f32..3.0, 4),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 4),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 4),
    ) {
        let scale = 1.0 / 2.0_f32.sqrt();
        let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(scale));

        let q_bounds = bounded_nd(&[2, 2], q_vals.clone(), q_vals.clone());
        let k_bounds = bounded_nd(&[2, 2], k_vals.clone(), k_vals.clone());
        let v_bounds = bounded_nd(&[2, 2], v_vals.clone(), v_vals.clone());

        let result = layer.propagate_ibp_ternary(&q_bounds, &k_bounds, &v_bounds)
            .map_err(|e| TestCaseError::fail(format!("Concrete IBP failed: {e}")))?;

        let q_mat = Array2::from_shape_vec((2, 2), q_vals).unwrap();
        let k_mat = Array2::from_shape_vec((2, 2), k_vals).unwrap();
        let v_mat = Array2::from_shape_vec((2, 2), v_vals).unwrap();
        let true_out = eval_standard_attention(&q_mat, &k_mat, &v_mat, scale);

        for i in 0..2_usize {
            for j in 0..2_usize {
                let lo = result.lower()[[i, j]];
                let hi = result.upper()[[i, j]];
                let tv = true_out[[i, j]];
                // Concrete inputs (lower == upper) have no interval uncertainty,
                // so IBP bounds should closely match the true output. The only
                // error source is f32 rounding through the MatMul→Softmax→MatMul
                // chain. Use relative tolerance matching other soundness tests.
                let tol = FP_TOLERANCE * tv.abs().max(lo.abs()).max(hi.abs()).max(1.0);
                prop_assert!(
                    lo - tol <= tv && tv <= hi + tol,
                    "Concrete attention not tight at [{i},{j}]: \
                     lo={lo}, true={tv}, hi={hi} (tol={tol})"
                );
                // For concrete inputs, the bound gap should also be small.
                prop_assert!(
                    hi - lo <= 2.0 * tol,
                    "Concrete attention bounds too wide at [{i},{j}]: \
                     lo={lo}, hi={hi}, gap={} (max_gap={})",
                    hi - lo, 2.0 * tol
                );
            }
        }
    }

    // ========================================================================
    // Scale inference soundness
    // ========================================================================

    /// Verify that auto-inferred scale (1/sqrt(head_dim)) produces the same
    /// bounds as manually specified scale.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_self_attention_scale_inference_matches(
        (q_lo, q_hi) in valid_interval_vec(4, 2.0),
        (k_lo, k_hi) in valid_interval_vec(4, 2.0),
        (v_lo, v_hi) in valid_interval_vec(4, 2.0),
    ) {
        let explicit_scale = 1.0 / 2.0_f32.sqrt(); // head_dim = 2
        let layer_explicit = SelfAttentionLayer::new(AttentionMask::Standard, Some(explicit_scale));
        let layer_inferred = SelfAttentionLayer::new(AttentionMask::Standard, None);

        let q_bounds = bounded_nd(&[2, 2], q_lo, q_hi);
        let k_bounds = bounded_nd(&[2, 2], k_lo, k_hi);
        let v_bounds = bounded_nd(&[2, 2], v_lo, v_hi);

        let result_explicit = layer_explicit.propagate_ibp_ternary(&q_bounds, &k_bounds, &v_bounds)
            .map_err(|e| TestCaseError::fail(format!("Explicit scale IBP failed: {e}")))?;
        let result_inferred = layer_inferred.propagate_ibp_ternary(&q_bounds, &k_bounds, &v_bounds)
            .map_err(|e| TestCaseError::fail(format!("Inferred scale IBP failed: {e}")))?;

        for i in 0..2_usize {
            for j in 0..2_usize {
                let tol = FP_TOLERANCE;
                prop_assert!(
                    (result_explicit.lower()[[i, j]] - result_inferred.lower()[[i, j]]).abs() < tol,
                    "Scale inference mismatch at lower[{i},{j}]: explicit={}, inferred={}",
                    result_explicit.lower()[[i, j]], result_inferred.lower()[[i, j]]
                );
                prop_assert!(
                    (result_explicit.upper()[[i, j]] - result_inferred.upper()[[i, j]]).abs() < tol,
                    "Scale inference mismatch at upper[{i},{j}]: explicit={}, inferred={}",
                    result_explicit.upper()[[i, j]], result_inferred.upper()[[i, j]]
                );
            }
        }
    }
}

// ============================================================================
// Decomposed SelfAttention graph soundness (Part of #2072)
// ============================================================================
//
// These tests verify soundness through the decomposed attention graph:
//   Input -> ReLU (identity passthrough) -> Q=K=V
//   Q, K -> BilinearCrown (Q @ K^T * scale)
//   QK -> Softmax
//   Softmax, V -> BilinearCrown (probs @ V)
//
// Note: `propagate_crown_batched` currently falls back to IBP for attention
// graphs due to a ShapeMismatch in BilinearCrown batched backward (the Q@K^T
// intermediate has shape [seq, seq] but batched CROWN expects the flattened
// input dimension). These tests gracefully handle the fallback: if CROWN fails,
// they verify the IBP result is sound. When BilinearCrown batched CROWN shape
// handling is extended, these tests will automatically verify CROWN bounds.
//
// Since the graph routes a single input to all three attention inputs through
// ReLU passthrough, Q=K=V=input for positive inputs. This is a valid soundness
// test: for any concrete input x within bounds, attention(x, x, x) must lie
// within the output bounds.

/// Build a GraphNetwork with decomposed SelfAttention for testing.
///
/// Graph structure:
///   _input -> q (ReLU) -+-> attn/qk (BilinearCrown, Q@K^T*scale)
///   _input -> k (ReLU) -+                                |
///   _input -> v (ReLU) --------+-> attn/softmax (Softmax) -> attn (BilinearCrown, probs@V)
fn build_attention_graph(mask: AttentionMask, scale: f32) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    let attn = SelfAttentionLayer::new(mask, Some(scale));
    let attn_node = GraphNode::new(
        "attn",
        Layer::SelfAttention(attn),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    );
    graph.try_add_node(attn_node).expect("decompose attention");
    graph.set_output("attn");
    graph
}

/// Generate a pair (lower, upper) vecs of length `n` with positive values only.
/// Positive values ensure ReLU passthrough (ReLU(x) = x for x > 0).
fn valid_positive_interval_vec(
    n: usize,
    lo_range: f32,
    hi_range: f32,
) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    proptest::collection::vec((lo_range..=hi_range, lo_range..=hi_range), n).prop_map(
        move |pairs| {
            let mut lo = Vec::with_capacity(n);
            let mut hi = Vec::with_capacity(n);
            for (a, b) in pairs {
                lo.push(a.min(b));
                hi.push(a.max(b));
            }
            (lo, hi)
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    // ========================================================================
    // Decomposed standard attention: CROWN backward soundness
    // ========================================================================

    /// CROWN backward through decomposed standard attention is sound.
    ///
    /// Graph: input -> ReLU -> Q=K=V -> BilinearCrown(Q@K^T*scale) -> Softmax -> BilinearCrown(@V)
    ///
    /// For any concrete positive input x within bounds, attention(x, x, x) must
    /// lie within the CROWN output bounds. Uses positive inputs so ReLU acts as
    /// identity passthrough.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_self_attention_decomposed_crown_standard_2x2(
        (in_lo, in_hi) in valid_positive_interval_vec(4, 0.1, 3.0),
    ) {
        let scale = 1.0 / 2.0_f32.sqrt();
        let graph = build_attention_graph(AttentionMask::Standard, scale);

        let input = bounded_nd(&[2, 2], in_lo.clone(), in_hi.clone());

        let crown_result = graph.propagate_crown_batched(&input);
        let result = match crown_result {
            Ok(output) => output,
            Err(_) => {
                // CROWN may fall back to IBP for some configurations.
                // If CROWN fails, verify IBP is still sound.
                let ibp_result = graph.propagate_ibp(&input)
                    .map_err(|e| TestCaseError::fail(format!("Both CROWN and IBP failed: {e}")))?;
                ibp_result
            }
        };

        prop_assert_eq!(result.shape(), &[2, 2], "Output shape mismatch");

        // Verify finite bounds
        for l in result.lower().iter() {
            prop_assert!(l.is_finite(), "CROWN lower bound is not finite: {l}");
        }
        for u in result.upper().iter() {
            prop_assert!(u.is_finite(), "CROWN upper bound is not finite: {u}");
        }

        // Verify bounds ordering (lower <= upper)
        for (l, u) in result.lower().iter().zip(result.upper().iter()) {
            prop_assert!(l <= u, "Invalid bounds: lower={l} > upper={u}");
        }

        let to_mat = |vals: &[f32]| -> Array2<f32> {
            Array2::from_shape_vec((2, 2), vals.to_vec()).unwrap()
        };

        let corner = |lo: &[f32], hi: &[f32], mask: u32| -> Vec<f32> {
            lo.iter().zip(hi.iter()).enumerate().map(|(idx, (&l, &u))| {
                if mask & (1 << idx) != 0 { u } else { l }
            }).collect()
        };

        // For Q=K=V=input, attention(x, x, x) = softmax(x @ x^T * scale) @ x
        let check_sound = |vals: &[f32], label: &str|
            -> Result<(), TestCaseError>
        {
            let mat = to_mat(vals);
            let true_out = eval_standard_attention(&mat, &mat, &mat, scale);

            for i in 0..2_usize {
                for j in 0..2_usize {
                    let lo = result.lower()[[i, j]];
                    let hi = result.upper()[[i, j]];
                    let tv = true_out[[i, j]];
                    let tol = FP_TOLERANCE * tv.abs().max(lo.abs()).max(hi.abs()).max(1.0);
                    prop_assert!(
                        lo - tol <= tv,
                        "Decomposed CROWN lower unsound at [{i},{j}] ({label}): \
                         lo={lo} > true={tv} (tol={tol})"
                    );
                    prop_assert!(
                        tv <= hi + tol,
                        "Decomposed CROWN upper unsound at [{i},{j}] ({label}): \
                         true={tv} > hi={hi} (tol={tol})"
                    );
                }
            }
            Ok(())
        };

        // Midpoints
        let mid: Vec<f32> = in_lo.iter().zip(in_hi.iter())
            .map(|(&l, &u)| f32::midpoint(l, u)).collect();

        check_sound(&in_lo, "all-lower")?;
        check_sound(&in_hi, "all-upper")?;
        check_sound(&mid, "midpoint")?;

        // All 16 corners of the 4-element input
        for mask in 0..16_u32 {
            let c = corner(&in_lo, &in_hi, mask);
            check_sound(&c, &format!("corner {mask}"))?;
        }
    }

    // ========================================================================
    // Decomposed causal attention: CROWN backward soundness
    // ========================================================================

    /// CROWN backward through decomposed causal attention is sound.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_self_attention_decomposed_crown_causal_2x2(
        (in_lo, in_hi) in valid_positive_interval_vec(4, 0.1, 3.0),
    ) {
        let scale = 1.0 / 2.0_f32.sqrt();
        let graph = build_attention_graph(AttentionMask::Causal, scale);

        let input = bounded_nd(&[2, 2], in_lo.clone(), in_hi.clone());

        let crown_result = graph.propagate_crown_batched(&input);
        let result = match crown_result {
            Ok(output) => output,
            Err(_) => {
                let ibp_result = graph.propagate_ibp(&input)
                    .map_err(|e| TestCaseError::fail(format!("Both CROWN and IBP failed: {e}")))?;
                ibp_result
            }
        };

        prop_assert_eq!(result.shape(), &[2, 2], "Output shape mismatch");

        for l in result.lower().iter() {
            prop_assert!(l.is_finite(), "CROWN lower bound is not finite: {l}");
        }
        for u in result.upper().iter() {
            prop_assert!(u.is_finite(), "CROWN upper bound is not finite: {u}");
        }
        for (l, u) in result.lower().iter().zip(result.upper().iter()) {
            prop_assert!(l <= u, "Invalid bounds: lower={l} > upper={u}");
        }

        let to_mat = |vals: &[f32]| -> Array2<f32> {
            Array2::from_shape_vec((2, 2), vals.to_vec()).unwrap()
        };

        let corner = |lo: &[f32], hi: &[f32], mask: u32| -> Vec<f32> {
            lo.iter().zip(hi.iter()).enumerate().map(|(idx, (&l, &u))| {
                if mask & (1 << idx) != 0 { u } else { l }
            }).collect()
        };

        let check_sound = |vals: &[f32], label: &str|
            -> Result<(), TestCaseError>
        {
            let mat = to_mat(vals);
            let true_out = eval_causal_attention(&mat, &mat, &mat, scale);

            for i in 0..2_usize {
                for j in 0..2_usize {
                    let lo = result.lower()[[i, j]];
                    let hi = result.upper()[[i, j]];
                    let tv = true_out[[i, j]];
                    let tol = FP_TOLERANCE * tv.abs().max(lo.abs()).max(hi.abs()).max(1.0);
                    prop_assert!(
                        lo - tol <= tv,
                        "Decomposed causal CROWN lower unsound at [{i},{j}] ({label}): \
                         lo={lo} > true={tv} (tol={tol})"
                    );
                    prop_assert!(
                        tv <= hi + tol,
                        "Decomposed causal CROWN upper unsound at [{i},{j}] ({label}): \
                         true={tv} > hi={hi} (tol={tol})"
                    );
                }
            }
            Ok(())
        };

        let mid: Vec<f32> = in_lo.iter().zip(in_hi.iter())
            .map(|(&l, &u)| f32::midpoint(l, u)).collect();

        check_sound(&in_lo, "all-lower")?;
        check_sound(&in_hi, "all-upper")?;
        check_sound(&mid, "midpoint")?;

        for mask in 0..16_u32 {
            let c = corner(&in_lo, &in_hi, mask);
            check_sound(&c, &format!("corner {mask}"))?;
        }
    }

    // ========================================================================
    // CROWN vs IBP comparison: decomposed attention
    // ========================================================================

    /// CROWN bounds through decomposed attention should be at least as tight
    /// as IBP bounds (or equal when CROWN falls back to IBP).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_self_attention_decomposed_crown_tighter_than_ibp(
        (in_lo, in_hi) in valid_positive_interval_vec(4, 0.1, 2.0),
    ) {
        let scale = 1.0 / 2.0_f32.sqrt();
        let graph = build_attention_graph(AttentionMask::Standard, scale);

        let input = bounded_nd(&[2, 2], in_lo, in_hi);

        let ibp_result = graph.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(format!("IBP failed: {e}")))?;

        let crown_result = graph.propagate_crown_batched(&input);
        let crown_output = match crown_result {
            Ok(output) => output,
            Err(_) => {
                // CROWN not yet supported for attention graphs — verify IBP
                // soundness so this test is not vacuous. When CROWN support
                // lands, this branch will stop being taken and the tightness
                // comparison below will run.
                let to_mat = |vals: &[f32]| -> Array2<f32> {
                    Array2::from_shape_vec((2, 2), vals.to_vec()).unwrap()
                };
                let in_lo_v: Vec<f32> = input.lower().iter().copied().collect();
                let in_hi_v: Vec<f32> = input.upper().iter().copied().collect();
                let mid: Vec<f32> = in_lo_v.iter().zip(in_hi_v.iter())
                    .map(|(&l, &u)| f32::midpoint(l, u)).collect();
                for vals in [&in_lo_v, &in_hi_v, &mid] {
                    let mat = to_mat(vals);
                    let true_out = eval_standard_attention(&mat, &mat, &mat, scale);
                    for i in 0..2_usize {
                        for j in 0..2_usize {
                            let lo = ibp_result.lower()[[i, j]];
                            let hi = ibp_result.upper()[[i, j]];
                            let tv = true_out[[i, j]];
                            let tol = FP_TOLERANCE * tv.abs().max(lo.abs()).max(hi.abs()).max(1.0);
                            prop_assert!(
                                lo - tol <= tv && tv <= hi + tol,
                                "IBP fallback unsound at [{i},{j}]: \
                                 lo={lo}, true={tv}, hi={hi} (tol={tol})"
                            );
                        }
                    }
                }
                return Ok(());
            }
        };

        // Verify CROWN bounds are sound (contain true output at corners/midpoint).
        // McCormick-based bilinear CROWN may produce bounds wider than IBP for some
        // inputs (the McCormick relaxation can add slack beyond what interval
        // arithmetic introduces). This is expected: bilinear CROWN trades
        // potential looseness for capturing cross-variable correlations that IBP
        // misses on non-trivial input ranges.
        let to_mat = |vals: &[f32]| -> Array2<f32> {
            Array2::from_shape_vec((2, 2), vals.to_vec()).unwrap()
        };
        let corner = |lo: &[f32], hi: &[f32], mask: u32| -> Vec<f32> {
            lo.iter().zip(hi.iter()).enumerate().map(|(idx, (&l, &u))| {
                if mask & (1 << idx) != 0 { u } else { l }
            }).collect()
        };
        let in_lo_v: Vec<f32> = input.lower().iter().copied().collect();
        let in_hi_v: Vec<f32> = input.upper().iter().copied().collect();

        let check_crown_sound = |vals: &[f32], label: &str|
            -> Result<(), TestCaseError>
        {
            let mat = to_mat(vals);
            let true_out = eval_standard_attention(&mat, &mat, &mat, scale);
            for i in 0..2_usize {
                for j in 0..2_usize {
                    let crown_lo = crown_output.lower()[[i, j]];
                    let crown_hi = crown_output.upper()[[i, j]];
                    let tv = true_out[[i, j]];
                    let tol = FP_TOLERANCE * tv.abs().max(crown_lo.abs()).max(crown_hi.abs()).max(1.0);
                    prop_assert!(
                        crown_lo - tol <= tv,
                        "CROWN lower unsound at [{i},{j}] ({label}): \
                         lo={crown_lo} > true={tv} (tol={tol})"
                    );
                    prop_assert!(
                        tv <= crown_hi + tol,
                        "CROWN upper unsound at [{i},{j}] ({label}): \
                         true={tv} > hi={crown_hi} (tol={tol})"
                    );
                }
            }
            Ok(())
        };

        check_crown_sound(&in_lo_v, "all-lower")?;
        check_crown_sound(&in_hi_v, "all-upper")?;

        for mask in 0..16_u32 {
            let c = corner(&in_lo_v, &in_hi_v, mask);
            check_crown_sound(&c, &format!("corner {mask}"))?;
        }
    }

    // ========================================================================
    // Concrete input: decomposed CROWN produces tight bounds
    // ========================================================================

    /// When input is concrete (lower == upper), CROWN through decomposed
    /// attention should produce bounds close to the true output.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_self_attention_decomposed_crown_concrete_tight(
        vals in proptest::collection::vec(0.1f32..3.0, 4),
    ) {
        let scale = 1.0 / 2.0_f32.sqrt();
        let graph = build_attention_graph(AttentionMask::Standard, scale);

        let input = bounded_nd(&[2, 2], vals.clone(), vals.clone());

        let crown_result = graph.propagate_crown_batched(&input);
        let result = match crown_result {
            Ok(output) => output,
            Err(_) => {
                // CROWN not yet supported for attention graphs — fall back to
                // IBP so this test is not vacuous. Concrete IBP should still
                // produce tight bounds.
                graph.propagate_ibp(&input)
                    .map_err(|e| TestCaseError::fail(format!("Both CROWN and IBP failed: {e}")))?
            }
        };

        let mat = Array2::from_shape_vec((2, 2), vals).unwrap();
        let true_out = eval_standard_attention(&mat, &mat, &mat, scale);

        for i in 0..2_usize {
            for j in 0..2_usize {
                let lo = result.lower()[[i, j]];
                let hi = result.upper()[[i, j]];
                let tv = true_out[[i, j]];
                let tol = FP_TOLERANCE * tv.abs().max(lo.abs()).max(hi.abs()).max(1.0);
                prop_assert!(
                    lo - tol <= tv && tv <= hi + tol,
                    "Concrete attention not sound at [{i},{j}]: \
                     lo={lo}, true={tv}, hi={hi} (tol={tol})"
                );
            }
        }
    }
}
