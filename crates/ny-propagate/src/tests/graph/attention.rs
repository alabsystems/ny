// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork attention graph builder and pattern tests.
use crate::layers::{AttentionMask, SelfAttentionLayer};
use crate::*;
use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_attention_pattern() {
    // Test bounded matmul with 2D input tensors (simulates Q @ K^T in attention)
    // Uses 2D input directly to matmul nodes without linear projection
    let mut graph = GraphNetwork::new();

    // For bounded matmul, we use ReLU first to create two 2D bounded tensors
    // then apply matmul. This tests the DAG structure.
    // input -> relu (as Q) AND input -> relu (as K) -> matmul

    // Simple pass-through via relu (positive inputs remain unchanged)
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));

    // MatMul: Q @ K^T (both are 2x2)
    let matmul = MatMulLayer::new(true, Some(1.0 / 2.0_f32.sqrt())); // scale by 1/sqrt(d)
    graph.add_node(GraphNode::binary(
        "attn_scores",
        Layer::MatMul(matmul),
        "q",
        "k",
    ));

    // Softmax on last axis
    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    graph.add_node(GraphNode::new(
        "attn_probs",
        Layer::Softmax(softmax),
        vec!["attn_scores".to_string()],
    ));
    graph.set_output("attn_probs");

    // Input: 2D tensor (2 tokens, 2 dims each) - all positive so ReLU is identity
    let input = BoundedTensor::new(
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Check that softmax outputs are valid probabilities
    for &val in output.lower().iter() {
        assert!(val >= 0.0, "Softmax output {} < 0", val);
    }
    for &val in output.upper().iter() {
        assert!(val <= 1.0, "Softmax output {} > 1", val);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_attention_graph_builder() {
    let mut builder = AttentionGraphBuilder::new();

    // Build a simple graph using the builder API
    // Test with 1D vector to avoid shape issues with Linear layers
    let weight_q = arr2(&[[1.0_f32, 0.5], [0.5, 1.0]]);
    let weight_k = arr2(&[[1.0_f32, -0.5], [-0.5, 1.0]]);

    let q = builder.add_projection("q", weight_q, None).unwrap();
    let k = builder.add_projection("k", weight_k, None).unwrap();

    // Add outputs together (both are 1D vectors after projection)
    let sum = builder.add_residual(&q, &k).unwrap();

    let graph = builder.build(&sum);

    assert_eq!(graph.num_nodes(), 3);

    // Test propagation with 1D input
    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // q = [1*1 + 0.5*1, 0.5*1 + 1*1] = [1.5, 1.5]
    // k = [1*1 + (-0.5)*1, (-0.5)*1 + 1*1] = [0.5, 0.5]
    // sum = [2.0, 2.0]
    assert!((output.lower()[[0]] - 2.0).abs() < 1e-5);
    assert!((output.lower()[[1]] - 2.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_attention_graph_builder_matmul() {
    // Test the builder with matmul using 2D inputs via ReLU passthrough
    let mut builder = AttentionGraphBuilder::new();

    // Add ReLU nodes for Q and K (acts as identity for positive inputs)
    let q = builder.add_relu("_input").unwrap();
    let k = builder.add_relu("_input").unwrap();
    let attn = builder.add_matmul(&q, &k, true, Some(0.5)).unwrap();
    let probs = builder.add_softmax(&attn, -1).unwrap();

    let graph = builder.build(&probs);

    assert_eq!(graph.num_nodes(), 4);

    // Test propagation with 2D input (positive values, so ReLU is identity)
    let input = BoundedTensor::new(
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.5, 0.5, 1.0])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.5, 0.5, 1.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Softmax outputs should be valid probabilities
    for &val in output.lower().iter() {
        assert!(val >= 0.0, "Softmax lower {} < 0", val);
    }
    for &val in output.upper().iter() {
        assert!(val <= 1.0, "Softmax upper {} > 1", val);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_attention_graph_builder_with_residual() {
    let mut builder = AttentionGraphBuilder::new();

    // Build: projection -> relu -> add(input, relu_output)
    let weight = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj = builder.add_projection("proj", weight, None).unwrap();
    let relu = builder.add_relu(&proj).unwrap();
    let residual = builder.add_residual("_input", &relu).unwrap();

    let graph = builder.build(&residual);

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // proj = input (identity), relu = proj (all positive), residual = input + relu = 2 * input
    assert!((output.lower()[[0]] - 2.0).abs() < 1e-5);
    assert!((output.lower()[[1]] - 4.0).abs() < 1e-5);
}

// =========================================================================
// SelfAttention decomposition tests (Part of #2072)
// =========================================================================

/// Helper: build a BoundedTensor from flat lower/upper vecs and shape.
fn bounded_nd(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap();
    BoundedTensor::new(l, u).unwrap()
}

/// try_add_node decomposes SelfAttention into 3 sub-nodes (BilinearCrown + Softmax + BilinearCrown).
#[test]
fn test_self_attention_decomposition_structure() {
    let mut graph = GraphNetwork::new();

    // Add Q, K, V source nodes (ReLU as identity passthrough for positive inputs)
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    // Add SelfAttention via try_add_node — should be decomposed
    let attn = SelfAttentionLayer::new(AttentionMask::Standard, Some(0.5));
    let attn_node = GraphNode::new(
        "attn",
        Layer::SelfAttention(attn),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    );
    graph.try_add_node(attn_node).unwrap();

    // Should have 6 nodes: q, k, v, attn/qk, attn/softmax, attn
    assert_eq!(graph.num_nodes(), 6);

    // Verify sub-node types
    assert!(
        matches!(
            graph.node("attn/qk").unwrap().layer(),
            Layer::BilinearCrown(_)
        ),
        "attn/qk should be BilinearCrown"
    );
    assert!(
        matches!(
            graph.node("attn/softmax").unwrap().layer(),
            Layer::Softmax(_)
        ),
        "attn/softmax should be Softmax"
    );
    assert!(
        matches!(graph.node("attn").unwrap().layer(), Layer::BilinearCrown(_)),
        "attn (output) should be BilinearCrown"
    );

    // Verify wiring: attn/qk takes (q, k)
    assert_eq!(graph.node("attn/qk").unwrap().inputs(), &["q", "k"]);
    // attn/softmax takes attn/qk
    assert_eq!(graph.node("attn/softmax").unwrap().inputs(), &["attn/qk"]);
    // attn takes (attn/softmax, v)
    assert_eq!(graph.node("attn").unwrap().inputs(), &["attn/softmax", "v"]);
}

/// Causal attention decomposition uses CausalSoftmax instead of Softmax.
#[test]
fn test_self_attention_decomposition_causal() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    let attn = SelfAttentionLayer::new(AttentionMask::Causal, Some(0.5));
    let attn_node = GraphNode::new(
        "attn",
        Layer::SelfAttention(attn),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    );
    graph.try_add_node(attn_node).unwrap();

    assert!(
        matches!(
            graph.node("attn/softmax").unwrap().layer(),
            Layer::CausalSoftmax(_)
        ),
        "Causal attention should use CausalSoftmax"
    );
}

/// SelfAttention without explicit scale errors on try_add_node.
#[test]
fn test_self_attention_decomposition_requires_scale() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    let attn = SelfAttentionLayer::standard(); // scale = None
    let attn_node = GraphNode::new(
        "attn",
        Layer::SelfAttention(attn),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    );
    let err = graph.try_add_node(attn_node).unwrap_err();
    assert!(
        format!("{err}").contains("explicit scale"),
        "Expected explicit scale error, got: {err}"
    );
}

/// IBP through decomposed SelfAttention produces valid bounds.
#[ntest::timeout(10000)]
#[test]
fn test_self_attention_decomposition_ibp() {
    let mut graph = GraphNetwork::new();

    // Q, K, V passthrough (ReLU on positive inputs = identity)
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    // SelfAttention with explicit scale (1/sqrt(2) for head_dim=2)
    let attn = SelfAttentionLayer::new(AttentionMask::Standard, Some(1.0 / 2.0_f32.sqrt()));
    let attn_node = GraphNode::new(
        "attn",
        Layer::SelfAttention(attn),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    );
    graph.try_add_node(attn_node).unwrap();
    graph.set_output("attn");

    // Concrete 2x2 input (positive so ReLU is identity)
    let input = bounded_nd(&[2, 2], vec![1.0, 0.5, 0.5, 1.0], vec![1.0, 0.5, 0.5, 1.0]);

    let output = graph.propagate_ibp(&input).unwrap();

    // Output shape should be [2, 2]
    assert_eq!(output.shape(), &[2, 2]);

    // Bounds should be valid (lower <= upper)
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "Invalid bounds: lower={l} > upper={u}");
    }
}

/// CROWN backward through decomposed SelfAttention produces valid bounds.
/// This is the key test — previously SelfAttention had NO CROWN support.
#[ntest::timeout(60000)]
#[test]
fn test_self_attention_decomposition_crown() {
    let mut graph = GraphNetwork::new();

    // Q, K, V passthrough
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    let attn = SelfAttentionLayer::new(AttentionMask::Standard, Some(0.5));
    let attn_node = GraphNode::new(
        "attn",
        Layer::SelfAttention(attn),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    );
    graph.try_add_node(attn_node).unwrap();
    graph.set_output("attn");

    // Perturbed 2x2 input
    let input = bounded_nd(&[2, 2], vec![0.5, 0.0, 0.0, 0.5], vec![1.5, 1.0, 1.0, 1.5]);

    // CROWN backward should succeed (not return UnsupportedOp)
    let output = graph
        .propagate_crown_batched(&input)
        .expect("decomposed SelfAttention CROWN must publish bounds");
    assert_eq!(output.shape(), &[2, 2]);
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "CROWN bounds invalid: lower={l} > upper={u}");
    }
    for l in output.lower().iter() {
        assert!(l.is_finite(), "CROWN lower bound is not finite: {l}");
    }
    for u in output.upper().iter() {
        assert!(u.is_finite(), "CROWN upper bound is not finite: {u}");
    }
}

/// Decomposed attention IBP should be sound: concrete input gives tight bounds.
#[ntest::timeout(10000)]
#[test]
fn test_self_attention_decomposition_ibp_soundness() {
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    let scale = 0.5_f32;
    let attn = SelfAttentionLayer::new(AttentionMask::Standard, Some(scale));
    let attn_node = GraphNode::new(
        "attn",
        Layer::SelfAttention(attn),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    );
    graph.try_add_node(attn_node).unwrap();
    graph.set_output("attn");

    // Concrete input (lower == upper)
    let vals = vec![1.0_f32, 0.0, 0.0, 1.0];
    let input = bounded_nd(&[2, 2], vals.clone(), vals);

    let output = graph.propagate_ibp(&input).unwrap();

    // For concrete input, lower should be close to upper
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(
            (u - l).abs() < 0.15,
            "Concrete input should give tight bounds: l={l}, u={u}, gap={}",
            u - l
        );
    }
}

// =========================================================================
// Performance benchmark: decomposed attention (Part of #2072)
// =========================================================================

/// Benchmark IBP performance for decomposed self-attention at multiple sizes.
///
/// CROWN backward through the decomposed attention graph currently falls back
/// to IBP due to a ShapeMismatch in batched CROWN backward for BilinearCrown.
/// The batched CROWN infrastructure expects 1D flattened linear bounds but
/// the BilinearCrown produces 2D intermediate shapes (seq_len × seq_len after
/// Q@K^T). This is a known limitation — the decomposition is correct (verified
/// sound via proptest at IBP level), and when batched CROWN BilinearCrown shape
/// handling is extended, CROWN will work through the decomposed graph automatically.
///
/// This benchmark:
/// - Verifies IBP works at multiple dimensions
/// - Documents CROWN fallback behavior
/// - Reports IBP bound width for typical attention dimensions
#[ntest::timeout(60000)]
#[test]
fn test_self_attention_decomposition_performance_benchmark() {
    use std::time::Instant;

    /// Run one benchmark configuration.
    fn benchmark_config(seq_len: usize, head_dim: usize) {
        let n = seq_len * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

        let attn = SelfAttentionLayer::new(AttentionMask::Standard, Some(scale));
        let attn_node = GraphNode::new(
            "attn",
            Layer::SelfAttention(attn),
            vec!["q".to_string(), "k".to_string(), "v".to_string()],
        );
        graph.try_add_node(attn_node).unwrap();
        graph.set_output("attn");

        let lo: Vec<f32> = (0..n).map(|i| 0.5 + 0.1 * ((i % 7) as f32)).collect();
        let hi: Vec<f32> = lo.iter().map(|&v| v + 0.5).collect();
        let input = bounded_nd(&[seq_len, head_dim], lo, hi);

        // Warmup
        let _ = graph.propagate_ibp(&input);

        let ibp_start = Instant::now();
        let ibp_output = graph.propagate_ibp(&input).unwrap();
        let ibp_elapsed = ibp_start.elapsed();

        let ibp_max_width = ibp_output
            .upper()
            .iter()
            .zip(ibp_output.lower().iter())
            .map(|(&u, &l)| u - l)
            .fold(0.0_f32, f32::max);
        let ibp_mean_width = ibp_output
            .upper()
            .iter()
            .zip(ibp_output.lower().iter())
            .map(|(&u, &l)| u - l)
            .sum::<f32>()
            / n as f32;

        eprintln!(
            "  seq={seq_len:>2}, d={head_dim:>2}: IBP {:>8.2?}  max_w={ibp_max_width:.4}  mean_w={ibp_mean_width:.4}",
            ibp_elapsed
        );

        for (&l, &u) in ibp_output.lower().iter().zip(ibp_output.upper().iter()) {
            assert!(l.is_finite() && u.is_finite() && l <= u, "IBP: [{l}, {u}]");
        }
        assert!(ibp_max_width > 0.0, "IBP non-degenerate: {ibp_max_width}");
        // CROWN backward: assert enclosure whenever it succeeds; larger
        // configurations may still fall back to IBP-only.
        let crown_result = graph.propagate_crown_batched(&input);
        match crown_result {
            Ok(crown_output) => {
                let crown_max_width = crown_output
                    .upper()
                    .iter()
                    .zip(crown_output.lower().iter())
                    .map(|(&u, &l)| u - l)
                    .fold(0.0_f32, f32::max);
                let ratio = crown_max_width / ibp_max_width.max(1e-10);
                eprintln!("         CROWN succeeded! max_w={crown_max_width:.4}  ratio={ratio:.4}");
                for (l, u) in crown_output.lower().iter().zip(crown_output.upper().iter()) {
                    assert!(l <= u, "CROWN bounds invalid: lower={l} > upper={u}");
                }
            }
            Err(_) => {
                // Benchmark-only path: a failed CROWN pass leaves the IBP
                // measurements above as the datapoint for this configuration.
            }
        }
    }

    eprintln!("\n=== SelfAttention Decomposition Benchmark ===");
    benchmark_config(2, 2);
    benchmark_config(4, 4);
    benchmark_config(8, 8);
    benchmark_config(16, 16);
    eprintln!("=== End Benchmark ===\n");
}

/// CROWN backward through the attention graph at the transformer-standard
/// `1/sqrt(d)` scale must succeed with valid, finite bounds. Complements
/// `test_self_attention_decomposition_crown`, which covers scale 0.5.
#[ntest::timeout(10000)]
#[test]
fn test_self_attention_crown_transformer_scale() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    let attn = SelfAttentionLayer::new(AttentionMask::Standard, Some(1.0 / 2.0_f32.sqrt()));
    let attn_node = GraphNode::new(
        "attn",
        Layer::SelfAttention(attn),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    );
    graph.try_add_node(attn_node).unwrap();
    graph.set_output("attn");

    let lo = ArrayD::from_elem(IxDyn(&[2, 2]), 0.5_f32);
    let hi = ArrayD::from_elem(IxDyn(&[2, 2]), 1.5_f32);
    let input = BoundedTensor::new(lo, hi).unwrap();

    let output = graph
        .propagate_crown_batched(&input)
        .expect("CROWN through decomposed self-attention must succeed");
    assert_eq!(output.shape(), &[2, 2]);
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(
            l.is_finite() && u.is_finite(),
            "CROWN bounds must be finite: [{l}, {u}]"
        );
        assert!(l <= u, "CROWN bounds invalid: lower={l} > upper={u}");
    }
}
