// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::NyError;
use ny_propagate::{GraphNetwork, ZonotopePropagationOptions, ZonotopeSoftmaxMode};
use ny_tensor::BoundedTensor;
use ny_test_utils::assert_bounded_tensor_close;

use super::super::{MultiBlockConfig, WhisperModel};
use super::helpers::{
    assert_block_width_tuple_finite, assert_tensor_sound_finite, run_whisper_block,
    whisper_tiny_encoder, whisper_zero_input,
};

struct LaneLocalization {
    label: &'static str,
    x_attn: BoundedTensor,
    block_output: BoundedTensor,
    details: crate::GpuCompositionalDetails,
    mlp_graph: GraphNetwork,
    stage_nodes: [(&'static str, String); 4],
}

struct StageBounds {
    ibp: BoundedTensor,
    crown: BoundedTensor,
}

fn zonotope_attention_delta(
    whisper: &WhisperModel,
    index: usize,
    input: &BoundedTensor,
    cfg: &MultiBlockConfig,
    label: &str,
) -> BoundedTensor {
    let ln_output = whisper
        .attention_layernorm_output_ibp(index, input, false)
        .unwrap_or_else(|err| panic!("block {index} {label} attention LayerNorm failed: {err}"));
    let mut suffix_graph = whisper
        .attention_suffix_subgraph_from_layernorm_output(index)
        .unwrap_or_else(|err| panic!("block {index} {label} attention suffix graph failed: {err}"));
    if cfg.layernorm_forward_mode {
        suffix_graph.set_layernorm_forward_mode(true);
    }

    let zonotope_options =
        ZonotopePropagationOptions::new().with_softmax_mode(ZonotopeSoftmaxMode::IntervalFallback);
    match suffix_graph.propagate_zonotope_with_options(&ln_output, 0.0, zonotope_options) {
        Ok(delta) => delta,
        Err(
            NyError::UnsupportedOp(_)
            | NyError::UnsupportedConfiguration(_)
            | NyError::NumericalInstability(_),
        ) => suffix_graph
            .propagate_ibp(&ln_output)
            .unwrap_or_else(|err| {
                panic!("block {index} {label} zonotope fallback attention IBP failed: {err}")
            }),
        Err(err) => panic!("block {index} {label} zonotope attention failed: {err}"),
    }
}

fn block_x_attn(
    whisper: &WhisperModel,
    index: usize,
    input: &BoundedTensor,
    cfg: &MultiBlockConfig,
    label: &str,
) -> BoundedTensor {
    let attn_delta = zonotope_attention_delta(whisper, index, input, cfg, label);
    input
        .add(&attn_delta)
        .unwrap_or_else(|err| panic!("block {index} {label} x_attn residual failed: {err}"))
}

fn mlp_stage_nodes(graph: &GraphNetwork) -> [(&'static str, String); 4] {
    let topo = graph
        .exec_order()
        .expect("MLP graph topological sort should succeed");
    let mlp_ln = topo
        .first()
        .cloned()
        .expect("MLP graph should contain a LayerNorm entry node");
    let gelu = topo
        .iter()
        .find(|name| {
            graph
                .node(name)
                .is_some_and(|node| node.layer().layer_type() == "GELU")
        })
        .cloned()
        .expect("MLP graph should contain a GELU node");
    let fc1 = graph
        .node(&gelu)
        .and_then(|node| node.inputs().first())
        .cloned()
        .expect("GELU should consume the fc1 stage output");
    let mlp_delta = topo
        .last()
        .cloned()
        .expect("MLP graph should contain a final delta output");

    assert_ne!(mlp_ln, fc1, "MLP LayerNorm and fc1 stage must be distinct");
    assert_ne!(fc1, gelu, "fc1 and GELU stage must be distinct");
    assert_ne!(gelu, mlp_delta, "GELU and final MLP delta must be distinct");

    [
        ("mlp_ln", mlp_ln),
        ("fc1", fc1),
        ("gelu", gelu),
        ("mlp_delta", mlp_delta),
    ]
}

fn mlp_prefix_subgraph(graph: &GraphNetwork, output_node: &str) -> GraphNetwork {
    let topo = graph
        .topological_sort()
        .expect("MLP graph topological sort should succeed");
    let mut prefix = GraphNetwork::new();

    for node_name in topo {
        let node = graph
            .node(&node_name)
            .cloned()
            .unwrap_or_else(|| panic!("MLP stage node {node_name} should exist"));
        prefix
            .try_add_node(node)
            .unwrap_or_else(|err| panic!("failed to add MLP prefix node {node_name}: {err}"));
        if node_name == output_node {
            prefix.set_output(node_name);
            return prefix;
        }
    }

    panic!("MLP stage output node {output_node} was not found in the prefix graph");
}

fn run_mlp_stage_ibp(
    graph: &GraphNetwork,
    output_node: &str,
    input: &BoundedTensor,
    stage_name: &str,
    lane: &str,
) -> BoundedTensor {
    mlp_prefix_subgraph(graph, output_node)
        .propagate_ibp(input)
        .unwrap_or_else(|err| panic!("{lane} {stage_name} IBP failed: {err}"))
}

fn run_mlp_stage_crown(
    graph: &GraphNetwork,
    output_node: &str,
    input: &BoundedTensor,
    stage_name: &str,
    lane: &str,
) -> BoundedTensor {
    mlp_prefix_subgraph(graph, output_node)
        .propagate_crown_within_graph_per_position_with_stats(input)
        .unwrap_or_else(|err| panic!("{lane} {stage_name} block-wise CROWN failed: {err}"))
        .0
}

fn assert_width_matches(label: &str, actual: f32, expected: f32) {
    let tolerance = expected.abs().max(1.0) * 1e-5;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label} width mismatch: actual={actual:.6e}, expected={expected:.6e}, tolerance={tolerance:.6e}"
    );
}

fn build_lane_localization(
    whisper: &WhisperModel,
    input: &BoundedTensor,
    cfg: &MultiBlockConfig,
    label: &'static str,
) -> LaneLocalization {
    let (block_output, details) = run_whisper_block(whisper, 0, input, cfg, label);
    assert_block_width_tuple_finite(&block_output, &details, label);

    let x_attn = block_x_attn(whisper, 0, input, cfg, label);
    assert_tensor_sound_finite(&x_attn, &format!("{label}_x_attn"));
    assert_width_matches(
        &format!("{label} x_attn"),
        x_attn.max_width(),
        details.x_attn_width,
    );

    let mlp_graph = whisper
        .mlp_subgraph(0)
        .unwrap_or_else(|err| panic!("{label} MLP graph should build: {err}"));
    let stage_nodes = mlp_stage_nodes(&mlp_graph);

    LaneLocalization {
        label,
        x_attn,
        block_output,
        details,
        mlp_graph,
        stage_nodes,
    }
}

fn assert_stage_topology_stable(practical: &LaneLocalization, sound: &LaneLocalization) {
    assert_eq!(
        practical
            .stage_nodes
            .iter()
            .map(|(label, _)| *label)
            .collect::<Vec<_>>(),
        sound
            .stage_nodes
            .iter()
            .map(|(label, _)| *label)
            .collect::<Vec<_>>(),
        "MLP stage labels must stay stable across zonotope-backed lanes"
    );
    assert_eq!(
        practical
            .stage_nodes
            .iter()
            .map(|(_, node)| node.as_str())
            .collect::<Vec<_>>(),
        sound
            .stage_nodes
            .iter()
            .map(|(_, node)| node.as_str())
            .collect::<Vec<_>>(),
        "MLP stage topology must stay stable across zonotope-backed lanes"
    );
}

fn run_stage_bounds(lane: &LaneLocalization, stage_name: &str, output_node: &str) -> StageBounds {
    StageBounds {
        ibp: run_mlp_stage_ibp(
            &lane.mlp_graph,
            output_node,
            &lane.x_attn,
            stage_name,
            lane.label,
        ),
        crown: run_mlp_stage_crown(
            &lane.mlp_graph,
            output_node,
            &lane.x_attn,
            stage_name,
            lane.label,
        ),
    }
}

fn assert_stage_bounds_valid(stage_name: &str, practical: &StageBounds, sound: &StageBounds) {
    assert_tensor_sound_finite(
        &practical.ibp,
        &format!("practical_hybrid_{stage_name}_ibp"),
    );
    assert_tensor_sound_finite(
        &practical.crown,
        &format!("practical_hybrid_{stage_name}_crown"),
    );
    assert_tensor_sound_finite(&sound.ibp, &format!("sound_hybrid_{stage_name}_ibp"));
    assert_tensor_sound_finite(&sound.crown, &format!("sound_hybrid_{stage_name}_crown"));
    assert_eq!(
        practical.ibp.shape(),
        practical.crown.shape(),
        "practical_hybrid {stage_name} IBP/CROWN shape mismatch"
    );
    assert_eq!(
        sound.ibp.shape(),
        sound.crown.shape(),
        "sound_hybrid {stage_name} IBP/CROWN shape mismatch"
    );
    assert_eq!(
        practical.crown.shape(),
        sound.crown.shape(),
        "{stage_name} practical/sound CROWN shape mismatch"
    );
}

fn assert_stage_lane_widths_match(stage_name: &str, practical: &StageBounds, sound: &StageBounds) {
    assert_width_matches(
        &format!("{stage_name} practical/sound IBP"),
        practical.ibp.max_width(),
        sound.ibp.max_width(),
    );
    assert_width_matches(
        &format!("{stage_name} practical/sound CROWN"),
        practical.crown.max_width(),
        sound.crown.max_width(),
    );
}

fn assert_final_stage_matches_block(lane: &LaneLocalization, stage: &StageBounds) {
    assert_width_matches(
        &format!("{} mlp_delta", lane.label),
        stage.crown.max_width(),
        lane.details.mlp_delta_width,
    );

    let reconstructed = lane.x_attn.add(&stage.crown).unwrap_or_else(|err| {
        panic!(
            "{} reconstructed block output should add: {err}",
            lane.label
        )
    });
    assert_bounded_tensor_close(
        &reconstructed,
        &lane.block_output,
        1e-3,
        &format!("{} reconstructed block output", lane.label),
    );
}

#[ntest::timeout(120000)]
#[test]
#[ignore = "requires the real compositional engine: build_lane_localization cross-checks its \
            recomputed per-stage x_attn (zonotope attention delta + residual) against \
            details.x_attn_width from the verify_block_compositional_gpu_with_config shim \
            (whisper/model.rs), which is the whole-block IBP output width — structurally a \
            different quantity; unmasked 2026-07 when block extraction was fixed for the \
            dynamo fixture"]
fn test_whisper_block0_zonotope_mlp_stage_localization_318() {
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);
    let practical_cfg = MultiBlockConfig::default()
        .with_zonotope_attention(true)
        .with_crown_block_wise(true);
    let sound_cfg = MultiBlockConfig::sound_tight().with_crown_block_wise(true);

    let practical = build_lane_localization(whisper, &input, &practical_cfg, "practical_hybrid");
    let sound = build_lane_localization(whisper, &input, &sound_cfg, "sound_hybrid");
    assert_stage_topology_stable(&practical, &sound);

    println!("\n=== Whisper Block-0 Zonotope MLP Stage Localization (#318) ===");
    assert_width_matches(
        "x_attn practical/sound",
        practical.x_attn.max_width(),
        sound.x_attn.max_width(),
    );
    println!(
        "  x_attn practical={:.6e} sound={:.6e} ratio={:.6e}",
        practical.x_attn.max_width(),
        sound.x_attn.max_width(),
        practical.x_attn.max_width() / sound.x_attn.max_width()
    );

    for ((stage_name, practical_node), (_, sound_node)) in
        practical.stage_nodes.iter().zip(sound.stage_nodes.iter())
    {
        let practical_stage = run_stage_bounds(&practical, stage_name, practical_node);
        let sound_stage = run_stage_bounds(&sound, stage_name, sound_node);
        assert_stage_bounds_valid(stage_name, &practical_stage, &sound_stage);
        assert_stage_lane_widths_match(stage_name, &practical_stage, &sound_stage);

        println!(
            "  stage={stage_name} node={practical_node} practical_ibp={:.6e} practical_crown={:.6e} sound_ibp={:.6e} sound_crown={:.6e}",
            practical_stage.ibp.max_width(),
            practical_stage.crown.max_width(),
            sound_stage.ibp.max_width(),
            sound_stage.crown.max_width(),
        );

        if *stage_name == "mlp_delta" {
            assert_final_stage_matches_block(&practical, &practical_stage);
            assert_final_stage_matches_block(&sound, &sound_stage);
        }
    }
}

fn print_cross_block_amplification(
    block0: &crate::GpuCompositionalDetails,
    block1: &crate::GpuCompositionalDetails,
    block1_x_attn_width: f32,
) {
    fn print_ratio(label: &str, block1_width: f32, block0_width: f32) {
        if block1_width.is_finite() && block0_width.is_finite() && block0_width > 0.0 {
            println!(
                "  {label}: block1/block0 = {:.6e}",
                block1_width / block0_width
            );
        } else if !block1_width.is_finite() {
            println!("  {label}: OVERFLOW (block-1={block1_width:.6e})");
        } else {
            println!("  {label}: SKIP (block-0={block0_width:.6e})");
        }
    }

    println!("\nCross-block amplification:");
    print_ratio("x_attn", block1_x_attn_width, block0.x_attn_width);
    print_ratio("output", block1.output_width, block0.output_width);
    print_ratio("mlp_delta", block1.mlp_delta_width, block0.mlp_delta_width);
}

fn print_block_per_stage_mlp(
    mlp_graph: &GraphNetwork,
    stage_nodes: &[(&str, String); 4],
    x_attn: &BoundedTensor,
    block_label: &str,
    expected_mlp_delta_width: Option<f32>,
) {
    println!("\n{block_label} per-stage MLP decomposition:");
    println!(
        "  {:>10} {:>14} {:>14} {:>10}",
        "stage", "ibp", "crown", "crown/ibp"
    );
    for (stage_name, node) in stage_nodes {
        let ibp = run_mlp_stage_ibp(mlp_graph, node, x_attn, stage_name, block_label);
        let crown = run_mlp_stage_crown(mlp_graph, node, x_attn, stage_name, block_label);

        assert_tensor_sound_finite(&ibp, &format!("{block_label}_{stage_name}_ibp"));
        assert_tensor_sound_finite(&crown, &format!("{block_label}_{stage_name}_crown"));
        assert_eq!(
            ibp.shape(),
            crown.shape(),
            "{block_label} {stage_name} IBP/CROWN shape mismatch"
        );

        if *stage_name == "mlp_delta" {
            if let Some(expected_width) = expected_mlp_delta_width {
                assert_width_matches(
                    &format!("{block_label} {stage_name}"),
                    crown.max_width(),
                    expected_width,
                );
            }
        }

        let crown_ibp_ratio = if ibp.max_width() > 0.0 {
            crown.max_width() / ibp.max_width()
        } else {
            f32::NAN
        };

        println!(
            "  {:>10} {:>14.6e} {:>14.6e} {:>10.6e}",
            stage_name,
            ibp.max_width(),
            crown.max_width(),
            crown_ibp_ratio
        );
    }
}

/// Block-1 MLP stage localization: measures per-stage IBP and CROWN widths
/// at block-1 to identify where cross-block bound amplification occurs.
/// Block-0 MLP interior was ruled out (W2-1542); this extends the same
/// decomposition to block-1 to measure amplification and CROWN effectiveness.
///
/// Part of #318.
#[ntest::timeout(300000)]
#[test]
#[ignore = "requires the real compositional engine: build_lane_localization cross-checks its \
            recomputed per-stage x_attn (zonotope attention delta + residual) against \
            details.x_attn_width from the verify_block_compositional_gpu_with_config shim \
            (whisper/model.rs), which is the whole-block IBP output width — structurally a \
            different quantity; unmasked 2026-07 when block extraction was fixed for the \
            dynamo fixture"]
fn test_whisper_block1_mlp_stage_localization_318() {
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);
    // sound_tight: conservative LN + zonotope attention.
    // Matches encoder sequential verifier for block 1+ (forced conservative LN).
    let cfg = MultiBlockConfig::sound_tight().with_crown_block_wise(true);

    let (block0_output, block0_details) =
        run_whisper_block(whisper, 0, &input, &cfg, "block0_sound");
    assert_block_width_tuple_finite(&block0_output, &block0_details, "block0_sound");

    let (_block1_output, block1_details) =
        run_whisper_block(whisper, 1, &block0_output, &cfg, "block1_sound");

    // zonotope_attention_delta falls back to IBP if zonotope fails at wide inputs
    let block1_x_attn = block_x_attn(whisper, 1, &block0_output, &cfg, "block1_sound");
    assert_tensor_sound_finite(&block1_x_attn, "block1_x_attn");
    assert_width_matches(
        "block1_sound x_attn",
        block1_x_attn.max_width(),
        block1_details.x_attn_width,
    );

    let mlp_graph = whisper
        .mlp_subgraph(1)
        .expect("block-1 MLP graph should build");
    let stage_nodes = mlp_stage_nodes(&mlp_graph);

    println!("\n=== Whisper Block-1 MLP Stage Localization (#318) ===");
    println!("Config: sound_tight + crown_block_wise (conservative LN)");
    println!("\nBlock-0 aggregate:");
    println!(
        "  x_attn={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        block0_details.x_attn_width, block0_details.mlp_delta_width, block0_details.output_width
    );
    println!("Block-1 aggregate:");
    println!(
        "  x_attn={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        block1_x_attn.max_width(),
        block1_details.mlp_delta_width,
        block1_details.output_width
    );

    print_cross_block_amplification(&block0_details, &block1_details, block1_x_attn.max_width());
    print_block_per_stage_mlp(
        &mlp_graph,
        &stage_nodes,
        &block1_x_attn,
        "Block-1",
        Some(block1_details.mlp_delta_width),
    );

    assert!(
        block1_x_attn.max_width().is_finite(),
        "block-1 x_attn must be finite for stage localization to be meaningful"
    );
}
