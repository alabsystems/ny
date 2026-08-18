// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use super::helpers::{assert_tensor_sound_finite, whisper_tiny_encoder, whisper_zero_input};
use ny_propagate::{ZonotopePropagationOptions, ZonotopeSoftmaxMode};
use ny_tensor::BoundedTensor;
use ny_test_utils::assert_bounded_tensor_close;

struct ContextRow {
    label: &'static str,
    context_width: f32,
}

fn expected_context_shape(whisper: &WhisperModel) -> Vec<usize> {
    vec![
        1,
        whisper.num_heads,
        2,
        whisper.hidden_dim / whisper.num_heads,
    ]
}

fn ibp_context_bounds(
    whisper: &WhisperModel,
    input: &BoundedTensor,
    layernorm_forward_mode: bool,
) -> BoundedTensor {
    let artifacts = whisper
        .attention_subgraph_artifacts(0)
        .expect("block 0 attention artifacts");
    let mut graph = artifacts.graph;
    graph.set_output(artifacts.context_node);
    if layernorm_forward_mode {
        graph.set_layernorm_forward_mode(true);
    }
    graph
        .propagate_ibp(input)
        .expect("block 0 IBP attention context")
}

fn attention_crown_context_bounds(
    whisper: &WhisperModel,
    input: &BoundedTensor,
    layernorm_forward_mode: bool,
) -> BoundedTensor {
    let artifacts = whisper
        .attention_subgraph_artifacts(0)
        .expect("block 0 attention artifacts");
    let mut graph = artifacts.graph;
    graph.set_output(artifacts.context_node);
    if layernorm_forward_mode {
        graph.set_layernorm_forward_mode(true);
    }
    graph
        .propagate_crown_batched_with_attention_full_composition(input)
        .map(|result| result.bounds)
        .expect("block 0 attention-CROWN context")
}

fn zonotope_context_bounds(whisper: &WhisperModel, input: &BoundedTensor) -> BoundedTensor {
    let artifacts = whisper
        .attention_suffix_subgraph_artifacts_from_layernorm_output(0)
        .expect("block 0 attention suffix artifacts");
    let ln_output = whisper
        .attention_layernorm_output_ibp(0, input, false)
        .expect("block 0 conservative ln_output");
    let mut graph = artifacts.graph;
    graph.set_output(artifacts.context_node);
    graph
        .propagate_zonotope_with_options(
            &ln_output,
            0.0,
            ZonotopePropagationOptions::new()
                .with_softmax_mode(ZonotopeSoftmaxMode::IntervalFallback),
        )
        .expect("block 0 zonotope context")
}

fn assert_context_row(bounds: &BoundedTensor, whisper: &WhisperModel, label: &str) {
    assert_eq!(
        bounds.shape(),
        expected_context_shape(whisper).as_slice(),
        "{label} shape mismatch"
    );
    assert_tensor_sound_finite(bounds, label);
}

fn print_context_rows(rows: &[ContextRow]) {
    println!("\n=== Whisper Block-0 Attention Context Matrix (#318) ===");
    for row in rows {
        println!("  [{:>20}] context={:.6e}", row.label, row.context_width);
    }
}

fn print_context_ratios(rows: &[ContextRow]) {
    let ibp = &rows[0];
    let zono = &rows[1];
    let attention_crown = &rows[2];
    println!("\n  === Context Ratios ===");
    println!(
        "  zonotope / graph_ibp       = {:.6}",
        zono.context_width / ibp.context_width
    );
    println!(
        "  attention_crown / graph_ibp = {:.6}",
        attention_crown.context_width / ibp.context_width
    );
    println!(
        "  attention_crown / zonotope  = {:.6}",
        attention_crown.context_width / zono.context_width
    );
}

#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_whisper_block0_attention_context_matrix_318() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);

    // Hold LayerNorm policy constant so this matrix compares the actual context
    // propagation methods rather than a mixture of method and normalization
    // policy changes.
    let ibp_context = ibp_context_bounds(whisper, &input, false);
    let zono_context = zonotope_context_bounds(whisper, &input);
    let attention_crown_context = attention_crown_context_bounds(whisper, &input, false);

    assert_context_row(&ibp_context, whisper, "graph_ibp_context");
    assert_context_row(&zono_context, whisper, "zonotope_context");
    assert_context_row(&attention_crown_context, whisper, "attention_crown_context");

    let rows = [
        ContextRow {
            label: "graph_ibp",
            context_width: ibp_context.max_width(),
        },
        ContextRow {
            label: "zonotope",
            context_width: zono_context.max_width(),
        },
        ContextRow {
            label: "attention_crown",
            context_width: attention_crown_context.max_width(),
        },
    ];
    print_context_rows(&rows);
    print_context_ratios(&rows);

    assert_bounded_tensor_close(
        &attention_crown_context,
        &ibp_context,
        1e-2,
        "attention-CROWN context should stay equivalent to IBP on current Whisper block-0 weights",
    );
    assert!(
        rows[1].context_width <= rows[0].context_width * 1.01,
        "zono context should stay no wider than the IBP context: zono={:.6e}, ibp={:.6e}",
        rows[1].context_width,
        rows[0].context_width
    );
}
