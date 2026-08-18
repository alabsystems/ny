// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Write as _;

use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_propagate::AlphaCrownConfig;
use ny_propagate::Layer as PropLayer;
use ny_tensor::BoundedTensor;

fn load_minimal_attention_fixture(path: &Path) -> OnnxModel {
    // Q @ K^T is encoded with an explicit standard-domain Transpose, so both
    // native ONNX schema validation and NY's raw preflight authenticate it.
    load_onnx(path).expect("Failed to load standard-ONNX minimal attention model")
}

#[ntest::timeout(10000)]
#[test]
fn test_load_minimal_attention_core_graph() {
    let path = require_test_model("minimal_attention_core.onnx");

    let model = load_minimal_attention_fixture(&path);

    let has_softmax = model
        .network
        .layers
        .iter()
        .any(|layer| layer.layer_type == LayerType::Softmax);
    assert!(
        has_softmax,
        "Expected Softmax layer in minimal attention model"
    );

    let prop = model
        .to_propagate_network()
        .expect("Failed to convert minimal attention model");

    let mut bilinear_count = 0;
    for layer in prop.layers() {
        if let PropLayer::BilinearCrown(_) = layer {
            bilinear_count += 1;
        }
    }
    // Since c93afde62, all activation-activation MatMuls (both Q@K^T and attn@V)
    // produce BilinearCrown layers, not MatMul.
    assert!(
        bilinear_count >= 2,
        "Expected at least 2 BilinearCrown layers (Q@K^T and attn@V), got {bilinear_count}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_simple_attention() {
    let path = require_test_model_with_hint("simple_attention.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load simple_attention model");

    // Check that the model loaded with some layers
    assert!(
        !model.network.layers.is_empty(),
        "Expected at least some layers to be loaded"
    );

    // Check for expected layer types
    let layer_types: Vec<_> = model.network.layers.iter().map(|l| &l.layer_type).collect();
    println!("Loaded layer types: {:?}", layer_types);

    // Attention model should have:
    // - Linear layers (Q, K, V, out projections from MatMul+Add)
    // - Softmax
    // - MatMul (bounded, for Q@K^T and attn@V)
    // - Add (for biases - recognized as part of linear or standalone)

    let has_linear_or_matmul = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::Linear || l.layer_type == LayerType::MatMul);
    assert!(has_linear_or_matmul, "Expected Linear or MatMul layers");

    let has_softmax = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::Softmax);
    assert!(has_softmax, "Expected Softmax layer");
}

#[ntest::timeout(10000)]
#[test]
fn test_load_causal_attention() {
    let path = require_test_model_with_hint("causal_attention.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load causal_attention model");

    // Check that the model loaded with some layers
    assert!(
        !model.network.layers.is_empty(),
        "Expected at least some layers to be loaded"
    );

    // Check for expected layer types
    let layer_types: Vec<_> = model.network.layers.iter().map(|l| &l.layer_type).collect();
    println!("Causal attention layer types: {:?}", layer_types);

    // Causal attention should have CausalSoftmax (fused from Trilu + Add + Softmax)
    let has_causal_softmax = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::CausalSoftmax);

    // Or if fusion didn't happen, it should at least have Softmax
    let has_softmax =
        model.network.layers.iter().any(|l| {
            l.layer_type == LayerType::Softmax || l.layer_type == LayerType::CausalSoftmax
        });
    assert!(has_softmax, "Expected Softmax or CausalSoftmax layer");

    if has_causal_softmax {
        println!("Causal softmax fusion detected");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_transformer_block() {
    let path = require_test_model_with_hint("transformer_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load transformer_block model");

    // Check that we have a good number of layers
    assert!(
        model.network.layers.len() >= 5,
        "Expected at least 5 layers in transformer block, got {}",
        model.network.layers.len()
    );

    // Check for expected transformer components
    let layer_types: Vec<_> = model.network.layers.iter().map(|l| &l.layer_type).collect();
    println!("Transformer block layer types: {:?}", layer_types);

    // The fixture contains an exact Erf decomposition of GELU.  Keep the
    // primitive instead of relying on the disabled canonical GELU fusion.
    let has_layer_norm = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::LayerNorm);
    let has_erf = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::Erf);
    let has_softmax = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::Softmax);

    // At least two of these should be present (depending on fusion)
    let transformer_markers = [has_layer_norm, has_erf, has_softmax]
        .iter()
        .filter(|&&x| x)
        .count();
    assert!(
        transformer_markers >= 2,
        "Expected at least 2 transformer markers (LayerNorm/Erf/Softmax)"
    );
    assert!(
        has_erf,
        "transformer block must preserve decomposed GELU Erf"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_transformer_mlp() {
    let path = require_test_model_with_hint("transformer_mlp.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load transformer_mlp model");

    // MLP should preserve the exact Linear -> decomposed-Erf GELU -> Linear graph.
    let layer_types: Vec<_> = model
        .network
        .layers
        .iter()
        .map(|l| l.layer_type.clone())
        .collect();
    println!("MLP layer types: {:?}", layer_types);

    let has_linear = layer_types
        .iter()
        .any(|t| *t == LayerType::Linear || *t == LayerType::MatMul);
    let has_erf = layer_types.contains(&LayerType::Erf);

    assert!(has_linear, "MLP should have Linear layers");
    assert!(
        has_erf,
        "MLP should preserve the decomposed GELU Erf primitive"
    );
    assert!(
        !layer_types.contains(&LayerType::GELU),
        "MLP must not use the disabled canonical GELU fusion"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_attention_structure() {
    // Finite additive masks remain an exact Add -> Softmax graph.  They must
    // not be rewritten to hard zero-probability causal semantics.
    let path = require_test_model_with_hint("causal_attention.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load causal attention model");
    let network = model.to_propagate_network().expect("Failed to convert");

    println!("\n=== Causal Attention Structure Test ===");
    println!("Network has {} layers", network.layers().len());

    // Print layer types
    for (i, layer) in network.layers().iter().enumerate() {
        println!("  Layer {}: {:?}", i, layer.layer_type());
    }

    // Verify CausalSoftmax is present (fusion worked)
    let has_causal_softmax = network
        .layers()
        .iter()
        .any(|l| l.layer_type() == "CausalSoftmax");
    let has_softmax = network.layers().iter().any(|l| l.layer_type() == "Softmax");
    // Since c93afde62, all activation-activation MatMuls produce BilinearCrown.
    let has_bilinear = network
        .layers()
        .iter()
        .any(|l| l.layer_type() == "BilinearCrown");

    println!("\nHas CausalSoftmax: {}", has_causal_softmax);
    println!("Has Softmax: {}", has_softmax);
    println!("Has BilinearCrown: {}", has_bilinear);

    assert!(!has_causal_softmax, "finite mask must not hard-fuse");
    assert!(has_softmax, "finite mask must preserve ordinary Softmax");
    assert!(
        has_bilinear,
        "Causal attention should have BilinearCrown (for Q@K^T and attn@V)"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_cross_attention_load() {
    // Test loading cross-attention model (encoder-decoder).
    let path = require_test_model_with_hint("cross_attention.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load cross attention model");
    let network = model.to_propagate_network().expect("Failed to convert");

    // Cross-attention model should produce a non-empty network
    assert!(
        !network.layers().is_empty(),
        "Cross attention model should have at least one layer"
    );

    // Cross-attention should include an attention normalization layer and
    // an attention score/value multiplication primitive.
    let has_attention_softmax = network.layers().iter().any(|layer| {
        let layer_type = layer.layer_type();
        layer_type == "Softmax" || layer_type == "CausalSoftmax"
    });
    let has_attention_matmul = network.layers().iter().any(|layer| {
        let layer_type = layer.layer_type();
        layer_type == "MatMul" || layer_type == "BilinearCrown"
    });

    assert!(
        has_attention_softmax,
        "Cross attention model should have Softmax or CausalSoftmax"
    );
    assert!(
        has_attention_matmul,
        "Cross attention model should have MatMul or BilinearCrown"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_encoder_decoder_block_load() {
    // Test loading the encoder-decoder block (Whisper decoder style).
    let path =
        require_test_model_with_hint("encoder_decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load encoder-decoder block model");

    println!("\n=== Encoder-Decoder Block Load Test ===");
    println!("Network has {} layers", model.network.layers.len());

    // Print layer types
    let layer_types: Vec<_> = model.network.layers.iter().map(|l| &l.layer_type).collect();
    println!("Layer types: {:?}", layer_types);

    // The finite additive self-attention mask must remain Add -> Softmax; it is
    // not equivalent to hard zero-probability causal semantics.
    let has_causal_softmax = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::CausalSoftmax);
    let has_softmax = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::Softmax);

    println!("Has CausalSoftmax: {}", has_causal_softmax);
    println!("Has Softmax: {}", has_softmax);

    assert!(!has_causal_softmax, "finite mask must not hard-fuse");
    assert!(has_softmax, "encoder-decoder block must preserve Softmax");
}

#[ntest::timeout(10000)]
#[test]
fn test_cross_attention_subgraph() {
    // Test cross-attention subgraph extraction for encoder-decoder models
    let path =
        require_test_model_with_hint("encoder_decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let decoder = load_decoder(&path).expect("Failed to load encoder-decoder model");
    println!(
        "Loaded encoder-decoder: {} blocks, hidden={}, heads={}",
        decoder.num_blocks, decoder.hidden_dim, decoder.num_heads
    );

    // Check that cross-attention is detected
    let has_cross = decoder
        .structure
        .blocks
        .iter()
        .any(|b| b.has_cross_attention);
    assert!(has_cross, "Model should report cross-attention support");

    // GraphNetwork cannot represent the independent encoder input, so the
    // extractor must fail rather than return a dangling two-input graph.
    match decoder.cross_attention_subgraph(0) {
        Err(NyError::UnsupportedConfiguration(message)) => {
            assert!(message.contains("single-input propagation contract"));
            assert!(message.contains("no graph or bounds were produced"));
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => panic!("cross-attention extractor returned an unusable two-input graph"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_with_attention() {
    // Test that to_graph_network handles attention patterns (binary MatMul)
    let path = require_test_model_with_hint("simple_attention.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load attention model");

    // Convert to graph network
    let graph = model
        .to_graph_network()
        .expect("Failed to convert attention model to graph");

    println!(
        "Attention model converted to graph with {} nodes",
        graph.num_nodes()
    );

    // The graph should have more nodes due to the branching structure
    assert!(
        graph.num_nodes() >= 3,
        "Attention graph should have at least Q/K/V projections"
    );

    // Test that IBP can propagate through the graph.
    // simple_attention.onnx was exported with input shape (1, 2, 4) — see
    // scripts/export_test_transformer.py line 249: SimpleAttention(dim=4), (1,2,4).
    // The model's linear layers are 4->4, so hidden_dim must match.
    let batch = 1;
    let seq_len = 2;
    let hidden_dim = 4;
    let shape = IxDyn(&[batch, seq_len, hidden_dim]);

    let center = ArrayD::from_elem(shape, 0.0_f32);
    let eps = 0.1;
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input = BoundedTensor::new(lower, upper).expect("Failed to create input");

    // This may fail for complex attention due to unsupported ops, but
    // should at least attempt propagation
    match graph.propagate_ibp(&input) {
        Ok(output) => {
            println!(
                "GraphNetwork IBP succeeded with output shape {:?}",
                output.shape()
            );
            // Bounds should be finite
            assert!(
                output.lower().iter().all(|v| v.is_finite()),
                "Lower bounds should be finite"
            );
            assert!(
                output.upper().iter().all(|v| v.is_finite()),
                "Upper bounds should be finite"
            );
        }
        Err(e) => {
            panic!("GraphNetwork IBP failed: {:?}", e);
        }
    }
}

/// Max bound width across all elements of a BoundedTensor.
fn max_bound_width(bt: &BoundedTensor) -> f32 {
    bt.lower()
        .iter()
        .zip(bt.upper().iter())
        .map(|(&l, &u)| u - l)
        .fold(0.0_f32, f32::max)
}

/// Measure bounds with soundness check for one method.
/// Returns (max_width, is_sound).
fn measure_with_soundness(
    bounds: &BoundedTensor,
    graph: &ny_propagate::GraphNetwork,
    center: &ArrayD<f32>,
    eps: f32,
) -> (f32, bool) {
    let mw = max_bound_width(bounds);
    let sound = verify_bounds_by_sampling(graph, bounds, center, eps);
    (mw, sound)
}

/// Single-eps measurement result for decision gate reporting.
struct OnnxEpsMeasurement {
    eps: f32,
    ibp_max: f32,
    crown_max: f32,
    alpha_max: f32,
    sound: bool,
    tighter: bool,
}

/// Run IBP/CROWN/alpha-CROWN comparison at one eps on a graph network.
fn measure_onnx_at_eps(
    graph: &ny_propagate::GraphNetwork,
    center: &ArrayD<f32>,
    eps: f32,
    alpha_config: &AlphaCrownConfig,
) -> OnnxEpsMeasurement {
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input =
        BoundedTensor::new(lower, upper).expect("invariant: center +/- eps produces valid bounds");

    let ibp = graph.propagate_ibp(&input).expect("IBP should succeed");
    let (ibp_max, sound_ibp) = measure_with_soundness(&ibp, graph, center, eps);

    let crown = graph
        .propagate_crown_batched(&input)
        .unwrap_or_else(|error| {
            panic!("CROWN must cover the attention fixture at eps={eps}: {error}")
        });
    let (crown_max, sound_crown) = measure_with_soundness(&crown, graph, center, eps);

    let alpha = graph
        .propagate_alpha_crown_with_config(&input, alpha_config)
        .unwrap_or_else(|error| {
            panic!("alpha-CROWN must cover the attention fixture at eps={eps}: {error}")
        });
    let (alpha_max, sound_alpha) = measure_with_soundness(&alpha, graph, center, eps);

    let tighter = alpha_max.is_finite() && alpha_max < ibp_max * 0.999;
    OnnxEpsMeasurement {
        eps,
        ibp_max,
        crown_max,
        alpha_max,
        sound: sound_ibp && sound_crown && sound_alpha,
        tighter,
    }
}

/// Format measurement results into a decision report.
fn format_onnx_decision_report(results: &[OnnxEpsMeasurement]) -> String {
    let mut report = String::from(
        "\n=== Phase 3: ONNX minimal_attention_core McCormick vs IBP ===\n\
         Model: x -> Linear_Q/K/V -> Q@K^T -> Softmax -> probs@V\n\
         Input: [1, 2], center=0.5\n",
    );
    let _ = writeln!(
        report,
        "  {:>7} | {:>10} {:>10} {:>10} | {:>8} {:>8}",
        "eps", "IBP max", "CROWN max", "alpha max", "CR/IBP", "decision"
    );
    let _ = writeln!(report, "  {}", "-".repeat(72));

    for r in results {
        let cr_ratio = if r.ibp_max > 0.0 && r.crown_max.is_finite() {
            r.crown_max / r.ibp_max
        } else {
            f32::NAN
        };
        let decision = if r.tighter {
            "TIGHTER"
        } else if r.alpha_max.is_nan() {
            "FAILED"
        } else {
            "~IBP"
        };
        let snd = if r.sound { "ok" } else { "FAIL" };
        let _ = writeln!(
            report,
            "  {:>7.4} | {:>10.6} {:>10.6} {:>10.6} | {:>8.4} {:>8} [{snd}]",
            r.eps, r.ibp_max, r.crown_max, r.alpha_max, cr_ratio, decision
        );
    }

    let any_tighter = results.iter().any(|r| r.tighter);
    let any_unsound = results.iter().any(|r| !r.sound);
    let _ = writeln!(
        report,
        "\n=== Decision: {} ===",
        if any_tighter && !any_unsound {
            "PASS: broadcast+alpha McCormick beats IBP on ONNX attention"
        } else if any_unsound {
            "BLOCKED: UNSOUND bounds detected — investigate"
        } else {
            "NEUTRAL: McCormick ~= IBP (expected for self-attention with correlated Q/K)"
        }
    );
    report
}

/// Phase 3 decision gate (#286): McCormick broadcast+alpha vs IBP on ONNX attention model.
///
/// Loads minimal_attention_core.onnx (self-attention with asymmetric Q/K weights)
/// and compares IBP, CROWN, and alpha-CROWN at eps = {0.001, 0.01, 0.05, 0.1}.
///
/// Reference: designs/2026-03-04-286-attention-bilinear-alternative.md Phase 3
/// Reference: auto_LiRPA operators/bivariate.py:39-75
#[ntest::timeout(120000)]
#[test]
fn test_phase3_mccormick_vs_ibp_onnx_attention() {
    let path = require_test_model("minimal_attention_core.onnx");
    let model = load_minimal_attention_fixture(&path);
    let graph = model
        .to_graph_network()
        .expect("Failed to convert minimal_attention_core to graph network");

    let center = ArrayD::from_elem(IxDyn(&[1, 2]), 0.5_f32);
    let alpha_config = AlphaCrownConfig {
        iterations: 30,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    };

    let results: Vec<_> = [0.001_f32, 0.01, 0.05, 0.1]
        .iter()
        .map(|&e| measure_onnx_at_eps(&graph, &center, e, &alpha_config))
        .collect();

    let report = format_onnx_decision_report(&results);
    eprint!("{report}");

    // Self-attention with shared input x: McCormick may be looser than IBP (expected).
    // McCormick treats Q/K as independent, losing correlation from shared x.
    // Synthetic tests in measurement_phase3.rs demonstrate tightness for asymmetric Q/K.
    assert!(
        results.iter().all(|r| r.sound),
        "All bound methods must be sound (contain concrete outputs within tolerance)"
    );
}

/// Verify soundness by grid-sampling concrete outputs and checking containment.
/// Samples 5^n grid points across the input domain.
fn verify_bounds_by_sampling(
    graph: &ny_propagate::GraphNetwork,
    bounds: &BoundedTensor,
    center: &ArrayD<f32>,
    eps: f32,
) -> bool {
    let n = center.len();
    let levels = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    let n_samples = levels.len().pow(n as u32);
    let lower_vals: Vec<f32> = center.iter().map(|&c| c - eps).collect();

    for sample_idx in 0..n_samples {
        let vals: Vec<f32> = (0..n)
            .map(|dim| {
                let level_idx = (sample_idx / levels.len().pow(dim as u32)) % levels.len();
                lower_vals[dim] + 2.0 * eps * levels[level_idx]
            })
            .collect();
        let point = ArrayD::from_shape_vec(IxDyn(center.shape()), vals)
            .expect("invariant: valid sample shape");
        let concrete = BoundedTensor::concrete(point).expect("invariant: valid concrete tensor");
        let exact = match graph.propagate_ibp(&concrete) {
            Ok(b) => b,
            Err(_) => return false,
        };
        for ((out_val, &lower), &upper) in exact
            .lower()
            .iter()
            .zip(bounds.lower().iter())
            .zip(bounds.upper().iter())
        {
            if *out_val < lower - 1e-4 || *out_val > upper + 1e-4 {
                return false;
            }
        }
    }
    true
}
