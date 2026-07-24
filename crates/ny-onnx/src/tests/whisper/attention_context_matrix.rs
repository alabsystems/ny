// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use super::helpers::{
    assert_block_width_tuple_finite, assert_tensor_sound_finite, run_whisper_block,
    run_whisper_block_attention_crown_seed, whisper_tiny_encoder, whisper_zero_input,
};
use ny_propagate::{ZonotopePropagationOptions, ZonotopeSoftmaxMode};
use ny_tensor::BoundedTensor;
use ny_test_utils::assert_bounded_tensor_close;

struct ContextRow {
    label: &'static str,
    context_width: f32,
    attn_output_width: f32,
    block_output_width: f32,
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
        println!(
            "  [{:>20}] context={:.6e}  attn_output={:.6e}  block_output={:.6e}",
            row.label, row.context_width, row.attn_output_width, row.block_output_width
        );
    }
}

fn print_context_ratios(rows: &[ContextRow]) {
    let mlp = &rows[0];
    let zono = &rows[1];
    let attention_crown = &rows[2];
    println!("\n  === Context Ratios ===");
    println!(
        "  zono_context / mlp_context             = {:.6}",
        zono.context_width / mlp.context_width
    );
    println!(
        "  attention_crown_context / mlp_context = {:.6}",
        attention_crown.context_width / mlp.context_width
    );
    println!(
        "  attention_crown_context / zono_context = {:.6}",
        attention_crown.context_width / zono.context_width
    );
    println!("\n  === Output-From-Context Ratios ===");
    println!(
        "  mlp_seed attn_output / context             = {:.6}",
        mlp.attn_output_width / mlp.context_width
    );
    println!(
        "  zono_seed attn_output / context            = {:.6}",
        zono.attn_output_width / zono.context_width
    );
    println!(
        "  attention_crown_seed attn_output / context = {:.6}",
        attention_crown.attn_output_width / attention_crown.context_width
    );
}

#[ntest::timeout(120000)]
#[test]
fn test_whisper_block0_attention_context_matrix_318() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);
    let mlp_seed_cfg = MultiBlockConfig::default().with_crown_block_wise(true);
    let zono_seed_cfg = MultiBlockConfig::sound_tight().with_crown_block_wise(true);

    let (mlp_seed_output, mlp_seed) =
        run_whisper_block(whisper, 0, &input, &mlp_seed_cfg, "mlp_seed");
    let (zono_seed_output, zono_seed) =
        run_whisper_block(whisper, 0, &input, &zono_seed_cfg, "zono_seed");
    let (attention_crown_output, attention_crown_seed) = run_whisper_block_attention_crown_seed(
        whisper,
        0,
        &input,
        &mlp_seed_cfg,
        "attention_crown_seed",
    );

    let ibp_context = ibp_context_bounds(whisper, &input, mlp_seed_cfg.layernorm_forward_mode);
    let zono_context = zonotope_context_bounds(whisper, &input);
    let attention_crown_context =
        attention_crown_context_bounds(whisper, &input, mlp_seed_cfg.layernorm_forward_mode);

    assert_context_row(&ibp_context, whisper, "mlp_seed_context");
    assert_context_row(&zono_context, whisper, "zono_seed_context");
    assert_context_row(&attention_crown_context, whisper, "attention_crown_context");
    assert_block_width_tuple_finite(&mlp_seed_output, &mlp_seed, "mlp_seed");
    assert_block_width_tuple_finite(&zono_seed_output, &zono_seed, "zono_seed");
    assert_block_width_tuple_finite(
        &attention_crown_output,
        &attention_crown_seed,
        "attention_crown_seed",
    );

    let rows = [
        ContextRow {
            label: "mlp_seed",
            context_width: ibp_context.max_width(),
            attn_output_width: mlp_seed.attention_delta_width,
            block_output_width: mlp_seed.output_width,
        },
        ContextRow {
            label: "zono_seed",
            context_width: zono_context.max_width(),
            attn_output_width: zono_seed.attention_delta_width,
            block_output_width: zono_seed.output_width,
        },
        ContextRow {
            label: "attention_crown_seed",
            context_width: attention_crown_context.max_width(),
            attn_output_width: attention_crown_seed.attention_delta_width,
            block_output_width: attention_crown_seed.output_width,
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
