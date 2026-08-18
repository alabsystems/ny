// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::fixtures::*;
use super::super::*;
use ndarray::{Array2, ArrayD};
use ny_propagate::layers::{GELULayer, MatMulLayer, SoftmaxLayer};
use ny_propagate::Layer as PropLayer;

#[ntest::timeout(10000)]
#[test]
fn test_whisper_attention_core_ibp() {
    // Test IBP on the attention core with Whisper dimensions.
    // This demonstrates compositional verification of attention.
    //
    // Whisper-tiny attention dimensions:
    //   hidden_dim = 384
    //   num_heads = 6
    //   head_dim = 64
    //
    // The attention core takes Q, K, V with shape [num_heads, seq, head_dim]
    // and produces output of same shape.
    use ny_propagate::{GraphNetwork, GraphNode};
    use ny_tensor::BoundedTensor;

    // Whisper-tiny dimensions
    let num_heads = 6;
    let seq_len = 4; // Small for testing
    let head_dim = 64;

    // Build attention core graph
    // Input is shared as Q, K, V (in practice they come from different projections)
    let mut graph = GraphNetwork::new();

    // Pass through GELU to create bounded Q, K, V from input
    graph.add_node(GraphNode::from_input(
        "q",
        PropLayer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        PropLayer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        PropLayer::GELU(GELULayer::default()),
    ));

    // Attention scores: Q @ K^T with scaling
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores = MatMulLayer::new(true, Some(scale)); // transpose_b=true for K^T
    graph.add_node(GraphNode::binary(
        "scores",
        PropLayer::MatMul(scores),
        "q",
        "k",
    ));

    // Softmax on attention scores
    let softmax = SoftmaxLayer::new(-1);
    graph.add_node(GraphNode::new(
        "probs",
        PropLayer::Softmax(softmax),
        vec!["scores".to_string()],
    ));

    // Output: attention_probs @ V
    let out_matmul = MatMulLayer::new(false, None);
    graph.add_node(GraphNode::binary(
        "out",
        PropLayer::MatMul(out_matmul),
        "probs",
        "v",
    ));
    graph.set_output("out");

    // Create input with Whisper attention shape [num_heads, seq, head_dim]
    let input_shape = vec![num_heads, seq_len, head_dim];
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&input_shape), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.1).expect("valid test input");

    println!("\n=== Testing Whisper Attention Core IBP ===");
    println!("Input shape (Q, K, V): {:?}", input_shape);
    println!(
        "Num heads: {}, Seq len: {}, Head dim: {}",
        num_heads, seq_len, head_dim
    );

    match graph.propagate_ibp(&input) {
        Ok(output) => {
            println!("SUCCESS: Attention core IBP completed");
            println!("Output shape: {:?}", output.shape());
            println!("Max width: {:.6}", output.max_width());

            // Output should be [num_heads, seq, head_dim]
            assert_eq!(output.shape(), &[num_heads, seq_len, head_dim]);

            // Bounds should be sound
            let sound = output
                .lower()
                .iter()
                .zip(output.upper().iter())
                .all(|(l, u)| l <= u);
            assert!(sound, "Bounds must be sound");

            // Verify output is in reasonable range (GELU output combined with attention)
            let max_upper = output
                .upper()
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_lower = output.lower().iter().cloned().fold(f32::INFINITY, f32::min);
            println!("Output range: [{:.4}, {:.4}]", min_lower, max_upper);
        }
        Err(e) => {
            panic!("Whisper attention core IBP failed: {:?}", e);
        }
    }

    // Also test CROWN for tighter bounds
    println!("\n--- CROWN bounds ---");
    match graph.propagate_crown(&input) {
        Ok(output) => {
            println!("CROWN output shape: {:?}", output.shape());
            println!("CROWN max width: {:.6}", output.max_width());

            // Soundness check: lower <= upper for all elements. Part of #1721.
            let unsound_count = output
                .lower()
                .iter()
                .zip(output.upper().iter())
                .filter(|(l, u)| l > u)
                .count();
            assert_eq!(
                unsound_count, 0,
                "CROWN bounds unsound: {unsound_count} elements have lower > upper"
            );

            // No NaN/Inf in CROWN output
            let has_nan_inf = output
                .lower()
                .iter()
                .chain(output.upper().iter())
                .any(|v| v.is_nan() || v.is_infinite());
            assert!(
                !has_nan_inf,
                "CROWN output contains NaN/Inf — bounds are vacuous"
            );
        }
        Err(e) => {
            // CROWN may not support all ops in the attention subgraph yet.
            // Assert the failure is from an unsupported op, not a soundness bug.
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("not supported")
                    || msg.contains("Unsupported")
                    || msg.contains("CROWN propagation requires"),
                "CROWN failed with unexpected error (not an unsupported-op error): {e}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_whisper_full_attention_subgraph_ibp() {
    // Test IBP on full attention subgraph including projections and shape transforms.
    // This demonstrates compositional verification of the complete attention mechanism.
    //
    // Full attention path:
    // 1. Input [seq, hidden]
    // 2. Q/K/V projections: Linear [seq, hidden] → [seq, hidden]
    // 3. Reshape: [seq, hidden] → [seq, heads, head_dim]
    // 4. Transpose: [seq, heads, head_dim] → [heads, seq, head_dim]
    // 5. Attention core: Q @ K^T → Softmax → @ V
    // 6. Transpose back: [heads, seq, head_dim] → [seq, heads, head_dim]
    // 7. Reshape: [seq, heads, head_dim] → [seq, hidden]
    // 8. Output projection: Linear [seq, hidden] → [seq, hidden]
    use ny_propagate::layers::{
        LinearLayer, MatMulLayer, ReshapeLayer, SoftmaxLayer, TransposeLayer,
    };
    use ny_propagate::{GraphNetwork, GraphNode};
    use ny_tensor::BoundedTensor;

    // Whisper-tiny dimensions
    let seq_len = 4_usize;
    let hidden_dim = 384_usize;
    let num_heads = 6_usize;
    let head_dim = 64_usize; // hidden_dim / num_heads

    // Create weight matrices for projections (small values for stable bounds)
    let q_weights = Array2::from_elem((hidden_dim, hidden_dim), 0.01f32);
    let k_weights = Array2::from_elem((hidden_dim, hidden_dim), 0.01f32);
    let v_weights = Array2::from_elem((hidden_dim, hidden_dim), 0.01f32);
    let out_weights = Array2::from_elem((hidden_dim, hidden_dim), 0.01f32);

    // Build the full attention graph
    let mut graph = GraphNetwork::new();

    // Q projection: [seq, hidden] @ [hidden, hidden] = [seq, hidden]
    let q_proj = LinearLayer::new(q_weights, None).expect("Q projection");
    graph.add_node(GraphNode::from_input("q_proj", PropLayer::Linear(q_proj)));

    // K projection
    let k_proj = LinearLayer::new(k_weights, None).expect("K projection");
    graph.add_node(GraphNode::from_input("k_proj", PropLayer::Linear(k_proj)));

    // V projection
    let v_proj = LinearLayer::new(v_weights, None).expect("V projection");
    graph.add_node(GraphNode::from_input("v_proj", PropLayer::Linear(v_proj)));

    // Reshape Q: [seq, hidden] → [seq, heads, head_dim]
    let q_reshape = ReshapeLayer::new(vec![seq_len as i64, num_heads as i64, head_dim as i64]);
    graph.add_node(GraphNode::new(
        "q_reshape",
        PropLayer::Reshape(q_reshape),
        vec!["q_proj".to_string()],
    ));

    // Reshape K: [seq, hidden] → [seq, heads, head_dim]
    let k_reshape = ReshapeLayer::new(vec![seq_len as i64, num_heads as i64, head_dim as i64]);
    graph.add_node(GraphNode::new(
        "k_reshape",
        PropLayer::Reshape(k_reshape),
        vec!["k_proj".to_string()],
    ));

    // Reshape V: [seq, hidden] → [seq, heads, head_dim]
    let v_reshape = ReshapeLayer::new(vec![seq_len as i64, num_heads as i64, head_dim as i64]);
    graph.add_node(GraphNode::new(
        "v_reshape",
        PropLayer::Reshape(v_reshape),
        vec!["v_proj".to_string()],
    ));

    // Transpose Q: [seq, heads, head_dim] → [heads, seq, head_dim]
    let q_transpose = TransposeLayer::new(vec![1, 0, 2]); // swap dims 0 and 1
    graph.add_node(GraphNode::new(
        "q_transpose",
        PropLayer::Transpose(q_transpose),
        vec!["q_reshape".to_string()],
    ));

    // Transpose K: [seq, heads, head_dim] → [heads, seq, head_dim]
    let k_transpose = TransposeLayer::new(vec![1, 0, 2]);
    graph.add_node(GraphNode::new(
        "k_transpose",
        PropLayer::Transpose(k_transpose),
        vec!["k_reshape".to_string()],
    ));

    // Transpose V: [seq, heads, head_dim] → [heads, seq, head_dim]
    let v_transpose = TransposeLayer::new(vec![1, 0, 2]);
    graph.add_node(GraphNode::new(
        "v_transpose",
        PropLayer::Transpose(v_transpose),
        vec!["v_reshape".to_string()],
    ));

    // Attention scores: Q @ K^T with scaling
    // Shape: [heads, seq, head_dim] @ [heads, head_dim, seq] = [heads, seq, seq]
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores = MatMulLayer::new(true, Some(scale));
    graph.add_node(GraphNode::binary(
        "scores",
        PropLayer::MatMul(scores),
        "q_transpose",
        "k_transpose",
    ));

    // Softmax
    let softmax = SoftmaxLayer::new(-1);
    graph.add_node(GraphNode::new(
        "probs",
        PropLayer::Softmax(softmax),
        vec!["scores".to_string()],
    ));

    // Attention output: probs @ V
    // Shape: [heads, seq, seq] @ [heads, seq, head_dim] = [heads, seq, head_dim]
    let attn_out = MatMulLayer::new(false, None);
    graph.add_node(GraphNode::binary(
        "attn_out",
        PropLayer::MatMul(attn_out),
        "probs",
        "v_transpose",
    ));

    // Transpose back: [heads, seq, head_dim] → [seq, heads, head_dim]
    let out_transpose = TransposeLayer::new(vec![1, 0, 2]);
    graph.add_node(GraphNode::new(
        "out_transpose",
        PropLayer::Transpose(out_transpose),
        vec!["attn_out".to_string()],
    ));

    // Reshape back: [seq, heads, head_dim] → [seq, hidden]
    let out_reshape = ReshapeLayer::new(vec![seq_len as i64, hidden_dim as i64]);
    graph.add_node(GraphNode::new(
        "out_reshape",
        PropLayer::Reshape(out_reshape),
        vec!["out_transpose".to_string()],
    ));

    // Output projection: [seq, hidden] @ [hidden, hidden] = [seq, hidden]
    let out_proj = LinearLayer::new(out_weights, None).expect("Output projection");
    graph.add_node(GraphNode::new(
        "out_proj",
        PropLayer::Linear(out_proj),
        vec!["out_reshape".to_string()],
    ));

    graph.set_output("out_proj");

    println!("\n=== Testing Full Attention Subgraph IBP ===");
    println!("Graph has {} nodes", graph.num_nodes());
    println!("Input shape: [seq={}, hidden={}]", seq_len, hidden_dim);
    println!(
        "Whisper dimensions: {} heads, {} head_dim",
        num_heads, head_dim
    );

    // Create input tensor
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&[seq_len, hidden_dim]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid test input");

    match graph.propagate_ibp(&input) {
        Ok(output) => {
            println!("SUCCESS: Full attention subgraph IBP completed");
            println!("Output shape: {:?}", output.shape());
            println!("Max width: {:.6}", output.max_width());

            // Output should be [seq, hidden]
            assert_eq!(output.shape(), &[seq_len, hidden_dim]);

            // Bounds should be sound
            let sound = output
                .lower()
                .iter()
                .zip(output.upper().iter())
                .all(|(l, u)| l <= u);
            assert!(sound, "Bounds must be sound");
        }
        Err(e) => {
            println!("Full attention IBP failed: {:?}", e);
            // Print graph structure for debugging
            println!("\nGraph structure:");
            for node in graph.node_names() {
                println!("  {}", node);
            }
            panic!("Full attention subgraph should work with shape transformations");
        }
    }
}

fn parse_block_index(name: &str) -> Option<usize> {
    let name = name.to_ascii_lowercase();
    for token in [
        "blocks.", "layers.", "layer.", "block.", "blocks_", "layers_", "layer_", "block_",
        "blocks/", "layers/", "layer/", "block/",
    ] {
        if let Some(pos) = name.find(token) {
            let rest = &name[pos + token.len()..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(index) = num_str.parse() {
                return Some(index);
            }
        }
    }
    None
}

fn find_whisper_attn_weight_key(
    weights: &WeightStore,
    block_idx: usize,
    kind: &str,
) -> Option<String> {
    let kind_suffixes: &[&str] = match kind {
        "q" => &["q"],
        "k" => &["k"],
        "v" => &["v"],
        "out" => &["out", "o"],
        _ => &[kind],
    };
    for suffix in kind_suffixes {
        let direct_candidates = [
            format!("p_layers_{}_self_attn_{}_proj_weight", block_idx, suffix),
            format!("p_layers_{}_attn_{}_proj_weight", block_idx, suffix),
            format!("layers_{}_self_attn_{}_proj_weight", block_idx, suffix),
            format!("layers_{}_attn_{}_proj_weight", block_idx, suffix),
            format!("layer_{}_self_attn_{}_proj_weight", block_idx, suffix),
            format!("layer_{}_attn_{}_proj_weight", block_idx, suffix),
            format!("layers.{}.self_attn.{}_proj.weight", block_idx, suffix),
            format!("layers.{}.attn.{}_proj.weight", block_idx, suffix),
            format!("layer.{}.self_attn.{}_proj.weight", block_idx, suffix),
            format!("layer.{}.attn.{}_proj.weight", block_idx, suffix),
            format!(
                "encoder.layers.{}.self_attn.{}_proj.weight",
                block_idx, suffix
            ),
            format!(
                "encoder.layer.{}.self_attn.{}_proj.weight",
                block_idx, suffix
            ),
            format!("encoder.layers.{}.attn.{}_proj.weight", block_idx, suffix),
            format!("encoder.layer.{}.attn.{}_proj.weight", block_idx, suffix),
            format!(
                "model.encoder.layers.{}.self_attn.{}_proj.weight",
                block_idx, suffix
            ),
            format!(
                "model.encoder.layer.{}.self_attn.{}_proj.weight",
                block_idx, suffix
            ),
            format!(
                "model.encoder.layers.{}.attn.{}_proj.weight",
                block_idx, suffix
            ),
            format!(
                "model.encoder.layer.{}.attn.{}_proj.weight",
                block_idx, suffix
            ),
            format!(
                "encoder.blocks.{}.self_attn.{}_proj.weight",
                block_idx, suffix
            ),
            format!("encoder.blocks.{}.attn.{}_proj.weight", block_idx, suffix),
        ];
        for candidate in direct_candidates {
            if weights.contains_key(&candidate) {
                return Some(candidate);
            }
        }
    }

    let (kind_tokens, fallback_tokens): (&[&str], &[&str]) = match kind {
        "q" => (
            &["q_proj", "query_proj", "query", "qproj"],
            &["attn/q", "attn.q", "attn_q"],
        ),
        "k" => (
            &["k_proj", "key_proj", "key", "kproj"],
            &["attn/k", "attn.k", "attn_k"],
        ),
        "v" => (
            &["v_proj", "value_proj", "value", "vproj"],
            &["attn/v", "attn.v", "attn_v"],
        ),
        "out" => (
            &["out_proj", "o_proj", "output_proj"],
            &["attn/out", "attn.out", "attn_out"],
        ),
        _ => (&[], &[]),
    };

    let mut keys: Vec<&str> = weights.keys().collect();
    keys.sort_unstable();
    for name in keys {
        let name_lower = name.to_ascii_lowercase();
        let Some(found_idx) = parse_block_index(name) else {
            continue;
        };
        if found_idx != block_idx {
            continue;
        }
        let has_attn_token = name_lower.contains("attn")
            || name_lower.contains("self_attn")
            || name_lower.contains("attention");
        let has_proj_token = name_lower.contains("proj") || name_lower.contains("projection");
        if !has_attn_token && !has_proj_token {
            continue;
        }
        if !name_lower.contains("weight") && !name_lower.contains("matmul") {
            continue;
        }
        let kind_match = kind_tokens.iter().any(|token| name_lower.contains(token))
            || fallback_tokens
                .iter()
                .any(|token| name_lower.contains(token));
        if kind_match {
            return Some(name.to_string());
        }
    }
    None
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_block_index_layer_variants() {
    assert_eq!(parse_block_index("layer.0.attn.k_proj.weight"), Some(0));
    assert_eq!(parse_block_index("layer_3_attn_q_proj_weight"), Some(3));
    assert_eq!(parse_block_index("layer/5/attn/k_proj/weight"), Some(5));
}

#[ntest::timeout(10000)]
#[test]
fn test_find_whisper_attn_weight_key_layer_attn_direct() {
    let mut weights = WeightStore::new();
    weights.insert(
        "layer.0.attn.k_proj.weight".to_string(),
        ArrayD::from_elem(vec![1], 0.0),
    );
    let key = find_whisper_attn_weight_key(&weights, 0, "k");
    assert_eq!(key.as_deref(), Some("layer.0.attn.k_proj.weight"));
}

#[ntest::timeout(10000)]
#[test]
fn test_find_whisper_attn_weight_key_proj_without_attn() {
    let mut weights = WeightStore::new();
    weights.insert(
        "layer.1.k_proj.weight".to_string(),
        ArrayD::from_elem(vec![1], 0.0),
    );
    let key = find_whisper_attn_weight_key(&weights, 1, "k");
    assert_eq!(key.as_deref(), Some("layer.1.k_proj.weight"));
}

#[ntest::timeout(10000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_whisper_attention_with_real_weights() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    // Test attention subgraph using actual Whisper model weights.
    // This validates that the verification works with production-scale weights.
    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);

    let whisper = load_whisper(&path).expect("Failed to load model");

    // Whisper-tiny dimensions
    let batch = 1_usize;
    let seq_len = 4_usize;
    let hidden_dim = whisper.hidden_dim;
    let num_heads = whisper.num_heads;
    let head_dim = hidden_dim / num_heads;
    assert_eq!(
        hidden_dim % num_heads,
        0,
        "hidden_dim must be divisible by num_heads"
    );

    println!("\n=== Testing Attention with Real Whisper Weights ===");
    println!(
        "Hidden dim: {}, Heads: {}, Head dim: {}",
        hidden_dim, num_heads, head_dim
    );
    let graph = whisper
        .attention_subgraph(0)
        .expect("Failed to build attention subgraph");
    let input_shape = vec![batch, seq_len, hidden_dim];
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&input_shape), 0.0f32);
    let input = ny_tensor::BoundedTensor::from_epsilon(input_data, 0.1).expect("valid input");
    let output = graph
        .propagate_ibp(&input)
        .expect("Attention subgraph IBP failed");
    assert_eq!(output.shape(), input_shape.as_slice());
    let bounds_sound = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .all(|(l, u)| l <= u);
    assert!(bounds_sound, "Bounds must be sound");
    let bounds_finite = output
        .lower()
        .iter()
        .chain(output.upper().iter())
        .all(|v| v.is_finite());
    assert!(bounds_finite, "Bounds must be finite");
    println!(
        "Attention subgraph output shape: {:?}, max width: {:.6}",
        output.shape(),
        output.max_width()
    );
}

// This production-weight zonotope path takes about 7s alone and can exceed
// 10s under the full suite's parallel load. Keep a CI-safe margin while still
// bounding hangs.
#[ntest::timeout(60000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_whisper_attention_with_real_weights_zonotope_context_shape_regression_3464() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);

    let whisper = load_whisper(&path).expect("Failed to load model");
    let hidden_dim = whisper.hidden_dim;

    let graph = whisper
        .attention_subgraph(0)
        .expect("Failed to build attention subgraph");
    let input_shape = vec![1, 2, hidden_dim];
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&input_shape), 0.0f32);
    let input = ny_tensor::BoundedTensor::from_epsilon(input_data, 1e-4).expect("valid input");

    // #3464 fix: transpose_b check now precedes expand_to_match, so the context
    // matmul (softmax @ V, transpose_b=false) falls back to IBP instead of
    // hitting a spurious ShapeMismatch on expand_to_match.
    let output = graph
        .propagate_zonotope(&input, 0.0)
        .expect("zonotope attention should succeed after #3464 fix");

    assert_eq!(
        output.shape(),
        &input_shape[..],
        "attention output shape must match input residual shape [1, 2, {hidden_dim}]"
    );
}
