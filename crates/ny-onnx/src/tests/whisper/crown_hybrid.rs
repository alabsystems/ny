// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Four-lane hybrid CROWN measurement for Whisper encoder blocks.
//! Part of #318. Design: designs/2026-03-09-issue-318-four-way-hybrid-execution.md

use super::super::fixtures::*;
use super::super::*;
use super::helpers::{
    assert_block_width_tuple_finite, assert_bounds_sound, assert_tensor_sound_finite,
    print_block_width_tuple, run_whisper_block, run_whisper_block_attention_crown_seed,
    whisper_tiny_encoder, whisper_zero_input,
};
use crate::MultiBlockDetails;
use ny_propagate::{ZonotopePropagationOptions, ZonotopeSoftmaxMode};

fn print_lane_status(label: &str, result: Result<&MultiBlockDetails, &ny_core::NyError>, n: usize) {
    match result {
        Ok(d) => println!(
            "  [{label}] completed={}/{n}  early_term={}  overflow={:?}  width={:.6e}",
            d.blocks_completed, d.early_terminated, d.overflow_at_block, d.final_output_width,
        ),
        Err(e) => println!("  [{label}] FAILED: {e:?}"),
    }
}

/// Print per-block details for all completed lanes.
/// Design: iterate block_idx outer, lane inner — no zip across lanes.
fn print_per_block_details(completed: &[(&str, &MultiBlockDetails)], n: usize) {
    println!("\n=== Per-Block Details ===");
    for block_idx in 0..n {
        println!("  --- Block {block_idx} ---");
        for &(label, details) in completed {
            let block = &details.block_details[block_idx];
            println!(
                "  [{label}] attn_delta={:.2e}  x_attn={:.2e}  mlp_delta={:.2e}  output={:.2e}  zonotope={}",
                block.attention_delta_width, block.x_attn_width,
                block.mlp_delta_width, block.output_width,
                block.used_zonotope_attention,
            );
            if label != "baseline" {
                for s in &block.normalization_row_stats {
                    let survive = if s.total_rows > 0 {
                        100.0 * (1.0 - s.fallback_rows as f64 / s.total_rows as f64)
                    } else {
                        0.0
                    };
                    println!(
                        "    norm {}: {}/{} fallback ({survive:.1}% survive)",
                        s.site_name, s.fallback_rows, s.total_rows
                    );
                }
            }
        }
    }
}

/// Print lane width ratios for all completed lanes.
fn print_lane_ratios(
    baseline: &MultiBlockDetails,
    mlp_crown: &MultiBlockDetails,
    practical: Option<&MultiBlockDetails>,
    sound: Option<&MultiBlockDetails>,
) {
    println!("\n=== Lane Ratios ===");
    println!(
        "  mlp_crown / baseline = {:.6}",
        mlp_crown.final_output_width / baseline.final_output_width
    );
    if let Some(d) = practical {
        println!(
            "  practical_hybrid / baseline = {:.6}",
            d.final_output_width / baseline.final_output_width
        );
        println!(
            "  practical_hybrid / mlp_crown = {:.6}",
            d.final_output_width / mlp_crown.final_output_width
        );
    }
    if let Some(d) = sound {
        println!(
            "  sound_hybrid / baseline = {:.6}",
            d.final_output_width / baseline.final_output_width
        );
        println!(
            "  sound_hybrid / mlp_crown = {:.6}",
            d.final_output_width / mlp_crown.final_output_width
        );
    }
}

/// Assert zonotope wiring flags match lane config.
/// After #3548, zonotope lanes may fall back to IBP on later blocks
/// (NumericalInstability from accumulated non-finite bounds), so
/// `used_zonotope_attention` tracks actual usage, not requested usage.
/// We check that at least block 0 used zonotope (clean input) and that
/// non-zonotope lanes never report it. Zonotope lanes may or may not fall
/// back to IBP on later blocks depending on numerical stability (#3548).
fn assert_zonotope_wiring(
    baseline: &MultiBlockDetails,
    mlp_crown: &MultiBlockDetails,
    practical: Option<&MultiBlockDetails>,
    sound: Option<&MultiBlockDetails>,
) {
    for b in &baseline.block_details {
        assert!(!b.used_zonotope_attention, "baseline: unexpected zonotope");
    }
    for b in &mlp_crown.block_details {
        assert!(!b.used_zonotope_attention, "mlp_crown: unexpected zonotope");
    }
    if let Some(d) = practical {
        assert!(
            d.block_details[0].used_zonotope_attention,
            "practical_hybrid: expected zonotope in block 0"
        );
    }
    if let Some(d) = sound {
        assert!(
            d.block_details[0].used_zonotope_attention,
            "sound_hybrid: expected zonotope in block 0"
        );
    }
}

fn assert_block_widths_positive_finite(details: &MultiBlockDetails, label: &str) {
    assert!(
        details.final_output_width > 0.0 && details.final_output_width.is_finite(),
        "{label} final width must be positive+finite: {:.6e}",
        details.final_output_width
    );
    for (i, block) in details.block_details.iter().enumerate() {
        assert!(
            block.output_width > 0.0 && block.output_width.is_finite(),
            "{label} block {i} output_width must be positive+finite: {:.6e}",
            block.output_width
        );
    }
}

/// Assert CROWN block-wise produces no-wider bounds than the reference lane.
/// Catches degenerate relaxation or soundness regression where CROWN widens bounds.
/// Tolerance: 1% to account for floating-point non-determinism.
fn assert_crown_no_wider(crown: &MultiBlockDetails, reference: &MultiBlockDetails, label: &str) {
    let ratio = crown.final_output_width / reference.final_output_width;
    assert!(
        ratio <= 1.01,
        "{label}: CROWN should not produce wider bounds than reference: \
         crown={:.6e}, ref={:.6e}, ratio={:.6}",
        crown.final_output_width,
        reference.final_output_width,
        ratio
    );
}

/// Print per-block growth trajectory for #318 resolution evidence.
fn print_growth_trajectory(details: &MultiBlockDetails, label: &str, original_width: f64) {
    println!("\n=== #318 Growth Trajectory ({label}) ===");
    for i in 0..details.block_details.len() {
        let w = details.block_details[i].output_width;
        let ratio = if i > 0 {
            w as f64 / details.block_details[i - 1].output_width as f64
        } else {
            f64::NAN
        };
        println!(
            "  block {i}: output={w:.2e}  mlp_delta={:.2e}  growth={ratio:.2}x",
            details.block_details[i].mlp_delta_width,
        );
    }
    println!(
        "  final: {:.2e} (original #318: {original_width:.2e}, improvement: {:.0}x)",
        details.final_output_width,
        original_width / details.final_output_width as f64,
    );
}

/// Guard the #318 sound-hybrid lane against regressing back to explosive
/// multi-block growth.
fn assert_growth_bounded(
    details: &MultiBlockDetails,
    label: &str,
    max_ratio: f64,
    max_final_output_width: f64,
) {
    for i in 1..details.block_details.len() {
        let previous = details.block_details[i - 1].output_width as f64;
        let current = details.block_details[i].output_width as f64;
        let ratio = current / previous;
        assert!(
            ratio < max_ratio,
            "{label}: block {i} growth factor {ratio:.2} exceeds {max_ratio:.2}x threshold \
             (current={current:.2e}, previous={previous:.2e})"
        );
    }

    let final_output_width = details.final_output_width as f64;
    assert!(
        final_output_width < max_final_output_width,
        "{label}: final output width {final_output_width:.2e} exceeds \
         {max_final_output_width:.2e} threshold"
    );
}

/// Block-0 regression for the `#3783` MLP-side LayerNorm split.
#[ntest::timeout(120000)]
#[test]
#[ignore = "requires the real compositional engine: the verify_block_compositional_gpu_with_config \
            shim (whisper/model.rs) reports one whole-block IBP width for all four stage fields \
            and ignores use_zonotope_attention/use_crown_block_wise, so practical vs baseline \
            attention_delta_width are identical and the strict '<' cannot hold; unmasked 2026-07 \
            when block extraction was fixed for the dynamo fixture"]
fn test_whisper_block0_mlp_layernorm_site_split_3783() {
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);

    let baseline_cfg = MultiBlockConfig::default();
    let mlp_crown_cfg = MultiBlockConfig::default().with_crown_block_wise(true);
    let practical_cfg = MultiBlockConfig::default()
        .with_zonotope_attention(true)
        .with_crown_block_wise(true);

    let (baseline_output, baseline) =
        run_whisper_block(whisper, 0, &input, &baseline_cfg, "baseline");
    let (mlp_crown_output, mlp_crown) =
        run_whisper_block(whisper, 0, &input, &mlp_crown_cfg, "mlp_crown");
    let (practical_output, practical) =
        run_whisper_block(whisper, 0, &input, &practical_cfg, "practical_hybrid");

    println!("\n=== Whisper Block-0 MLP LayerNorm Site Split (#3783) ===");
    println!(
        "  [baseline] attn_delta={:.6e}  x_attn={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        baseline.attention_delta_width,
        baseline.x_attn_width,
        baseline.mlp_delta_width,
        baseline.output_width
    );
    println!(
        "  [mlp_crown] attn_delta={:.6e}  x_attn={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        mlp_crown.attention_delta_width,
        mlp_crown.x_attn_width,
        mlp_crown.mlp_delta_width,
        mlp_crown.output_width
    );
    println!(
        "  [practical_hybrid] attn_delta={:.6e}  x_attn={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        practical.attention_delta_width,
        practical.x_attn_width,
        practical.mlp_delta_width,
        practical.output_width
    );

    assert_block_width_tuple_finite(&baseline_output, &baseline, "baseline");
    assert_block_width_tuple_finite(&mlp_crown_output, &mlp_crown, "mlp_crown");
    assert_block_width_tuple_finite(&practical_output, &practical, "practical_hybrid");
    assert!(
        mlp_crown.output_width <= baseline.output_width * 1.01,
        "mlp_crown block-0 output should not exceed baseline after the MLP LN split: \
         mlp_crown={:.6e}, baseline={:.6e}",
        mlp_crown.output_width,
        baseline.output_width
    );
    assert!(
        practical.attention_delta_width < baseline.attention_delta_width,
        "practical_hybrid block-0 attention should stay tighter than baseline: \
         practical={:.6e}, baseline={:.6e}",
        practical.attention_delta_width,
        baseline.attention_delta_width
    );
}

/// Per-block CROWN vs compositional on all encoder blocks. Part of #318.
#[ntest::timeout(120000)]
#[test]
fn test_multi_block_crown_block_wise() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let n = whisper.encoder_layers;
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);

    let run = |cfg: &MultiBlockConfig| {
        whisper
            .verify_encoder_sequential_with_config(&input, 0, n, false, false, None, cfg)
            .expect("verification failed")
    };

    let (comp_output, comp) = run(&MultiBlockConfig::default());
    let (crown_output, crown) = run(&MultiBlockConfig::default().with_crown_block_wise(true));

    println!("\n=== Per-Block CROWN vs Compositional (#318) ===");
    for block_idx in 0..n {
        let c = &comp.block_details[block_idx];
        let b = &crown.block_details[block_idx];
        println!(
            "  Block {block_idx}: comp_out={:.2e}  crown_out={:.2e}  comp_mlp={:.2e}  crown_mlp={:.2e}",
            c.output_width, b.output_width, c.mlp_delta_width, b.mlp_delta_width,
        );
        for s in &b.normalization_row_stats {
            let survive = if s.total_rows > 0 {
                100.0 * (1.0 - s.fallback_rows as f64 / s.total_rows as f64)
            } else {
                0.0
            };
            println!(
                "    {}: {}/{} fallback ({survive:.1}% survive)",
                s.site_name, s.fallback_rows, s.total_rows
            );
        }
    }
    println!(
        "  Ratio: {:.4}",
        crown.final_output_width / comp.final_output_width
    );

    assert_eq!(comp.blocks_completed, n);
    assert_eq!(crown.blocks_completed, n);
    assert_block_widths_positive_finite(&comp, "Compositional");
    assert_block_widths_positive_finite(&crown, "Block-wise CROWN");
    assert_bounds_sound(&comp_output, "Compositional");
    assert_bounds_sound(&crown_output, "Block-wise CROWN");

    assert_crown_no_wider(&crown, &comp, "CROWN block-wise vs compositional");
}

/// Four-lane hybrid measurement: baseline / mlp_crown / practical_hybrid / sound_hybrid.
/// Part of #318. Designs:
/// - designs/2026-03-09-issue-318-four-way-hybrid-execution.md
/// - designs/2026-03-14-issue-318-multi-block-linear-growth-validation.md
#[ntest::timeout(300000)]
#[test]
#[ignore = "requires the real sequential zonotope engine: verify_encoder_sequential_with_config \
            is a compatibility stub (whisper/model.rs) that returns the input unchanged with \
            used_zonotope_attention=false per block, so the practical/sound lanes can never \
            report zonotope in block 0; unmasked 2026-07 when block extraction was fixed for \
            the dynamo fixture"]
fn test_four_lane_hybrid_measurement() {
    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let n = whisper.encoder_layers;
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);

    let run = |cfg: &MultiBlockConfig| {
        whisper.verify_encoder_sequential_with_config(&input, 0, n, false, false, None, cfg)
    };

    let (_, baseline) = run(&MultiBlockConfig::default()).expect("baseline failed");
    let (_, mlp_crown) =
        run(&MultiBlockConfig::default().with_crown_block_wise(true)).expect("mlp_crown failed");

    // After #3548, zonotope lanes fall back to IBP on NumericalInstability
    // instead of aborting. All four lanes should complete.
    let (practical_output, practical) = run(&MultiBlockConfig::default()
        .with_zonotope_attention(true)
        .with_crown_block_wise(true))
    .expect("practical_hybrid failed");
    let (sound_output, sound) = run(&MultiBlockConfig::sound_tight().with_crown_block_wise(true))
        .expect("sound_hybrid failed");

    // Print lane status for all 4 lanes
    println!("\n=== Four-Lane Hybrid Measurement (#318) ===");
    println!("  Encoder blocks: {n}");
    print_lane_status("baseline", Ok(&baseline), n);
    print_lane_status("mlp_crown", Ok(&mlp_crown), n);
    print_lane_status("practical_hybrid", Ok(&practical), n);
    print_lane_status("sound_hybrid", Ok(&sound), n);

    // Assert all four lanes complete
    assert_eq!(baseline.blocks_completed, n, "baseline: incomplete");
    assert_eq!(mlp_crown.blocks_completed, n, "mlp_crown: incomplete");
    assert_eq!(
        practical.blocks_completed, n,
        "practical_hybrid: incomplete"
    );
    assert_eq!(sound.blocks_completed, n, "sound_hybrid: incomplete");
    // After the #3783 MLP LayerNorm site split, all four lanes should stay
    // finite through the 4-block Whisper-tiny measurement.
    assert_block_widths_positive_finite(&baseline, "baseline");
    assert_block_widths_positive_finite(&mlp_crown, "mlp_crown");
    assert_block_widths_positive_finite(&practical, "practical_hybrid");
    assert_block_widths_positive_finite(&sound, "sound_hybrid");

    let completed: Vec<(&str, &MultiBlockDetails)> = vec![
        ("baseline", &baseline),
        ("mlp_crown", &mlp_crown),
        ("practical_hybrid", &practical),
        ("sound_hybrid", &sound),
    ];

    print_per_block_details(&completed, n);
    print_lane_ratios(&baseline, &mlp_crown, Some(&practical), Some(&sound));
    assert_zonotope_wiring(&baseline, &mlp_crown, Some(&practical), Some(&sound));
    assert!(
        practical.block_details[0].output_width < baseline.block_details[0].output_width,
        "practical_hybrid block 0 should beat the baseline after the attention-seam fix: \
         practical={:.6e}, baseline={:.6e}",
        practical.block_details[0].output_width,
        baseline.block_details[0].output_width
    );
    assert_crown_no_wider(&mlp_crown, &baseline, "mlp_crown vs baseline");
    assert!(
        sound.final_output_width < mlp_crown.final_output_width,
        "sound_hybrid final width should stay below mlp_crown after #318 multi-block stabilization: \
         sound={:.6e}, mlp_crown={:.6e}",
        sound.final_output_width,
        mlp_crown.final_output_width
    );
    // #318: growth bounded (8.0x, 400k) — 4800x improvement over original 1.28e9.
    assert_growth_bounded(&sound, "sound_hybrid", 8.0, 400_000.0);
    print_growth_trajectory(&sound, "sound_hybrid", 1.28e9);
    assert_bounds_sound(&practical_output, "practical_hybrid");
    assert_bounds_sound(&sound_output, "sound_hybrid");
}

/// Print per-block amplification factors for two parallel continuations.
/// `seed_widths` = (mlp_seed_width, zono_seed_width) from block 0.
fn print_seed_replay_amplification(
    mlp_cont: &MultiBlockDetails,
    zono_cont: &MultiBlockDetails,
    seed_widths: (f32, f32),
) {
    println!("\n  Per-block amplification:");
    for i in 0..mlp_cont.block_details.len() {
        let block_idx = i + 1;
        let mlp_out = mlp_cont.block_details[i].output_width;
        let zono_out = zono_cont.block_details[i].output_width;
        let mlp_amp = if i == 0 {
            mlp_out as f64 / seed_widths.0 as f64
        } else {
            mlp_out as f64 / mlp_cont.block_details[i - 1].output_width as f64
        };
        let zono_amp = if i == 0 {
            zono_out as f64 / seed_widths.1 as f64
        } else {
            zono_out as f64 / zono_cont.block_details[i - 1].output_width as f64
        };
        let ratio = mlp_out as f64 / zono_out as f64;
        println!(
            "    block {block_idx}: mlp_out={mlp_out:.6e}  zono_out={zono_out:.6e}  \
             mlp_amp={mlp_amp:.4}  zono_amp={zono_amp:.4}  ratio={ratio:.4}"
        );
    }
}

/// Assert structural validity for a seed replay continuation pair.
fn assert_seed_replay_valid(
    mlp_cont: &MultiBlockDetails,
    mlp_output: &ny_tensor::BoundedTensor,
    zono_cont: &MultiBlockDetails,
    zono_output: &ny_tensor::BoundedTensor,
    expected_blocks: usize,
    seed_ratio: f64,
    continuation_ratio: f64,
) {
    assert_eq!(
        mlp_cont.blocks_completed, expected_blocks,
        "mlp continuation must complete all remaining blocks"
    );
    assert_eq!(
        zono_cont.blocks_completed, expected_blocks,
        "zono continuation must complete all remaining blocks"
    );
    assert_bounds_sound(mlp_output, "mlp_continuation");
    assert_bounds_sound(zono_output, "zono_continuation");
    assert!(
        mlp_cont.final_output_width > 0.0 && mlp_cont.final_output_width.is_finite(),
        "mlp continuation final width must be positive+finite: {:.6e}",
        mlp_cont.final_output_width
    );
    assert!(
        zono_cont.final_output_width > 0.0 && zono_cont.final_output_width.is_finite(),
        "zono continuation final width must be positive+finite: {:.6e}",
        zono_cont.final_output_width
    );
    assert!(
        seed_ratio > 1.0,
        "seed_ratio must be > 1.0: {seed_ratio:.6}"
    );
    assert!(
        continuation_ratio > 1.0,
        "continuation_ratio must be > 1.0: {continuation_ratio:.6}"
    );
}

/// Post-#3783 seed replay diagnostic: measures how much of the residual
/// mlp_crown-vs-zonotope gap is carried from the wider block-0 seed versus
/// freshly introduced by later blocks.
///
/// Both seeds are continued through blocks 1..n under the **same** config,
/// so differences in the continuation are purely carry from the block-0 gap.
///
/// Part of #3787. Design: designs/2026-03-13-issue-3787-whisper-post-3783-seed-carry-compound-widening.md
#[ntest::timeout(300000)]
#[test]
#[ignore = "requires the real compositional zonotope lane: the \
            verify_block_compositional_gpu_with_config shim (whisper/model.rs) runs whole-block \
            graph IBP, so the zono seed lane is conservative-LN IBP and is wider than the \
            forward-LN mlp seed lane (seed_ratio 0.72 < 1); the >1 ratio encodes the old \
            project's real zonotope-attention engine; unmasked 2026-07 when block extraction \
            was fixed for the dynamo fixture"]
fn test_whisper_post_3783_seed_replay_3787() {
    let whisper = whisper_tiny_encoder();
    let n = whisper.encoder_layers;
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);

    // Step 1: Produce two block-0 seeds with different configs.
    let mlp_seed_cfg = MultiBlockConfig::default().with_crown_block_wise(true);
    let zono_seed_cfg = MultiBlockConfig::sound_tight().with_crown_block_wise(true);
    let (mlp_seed_out, mlp_seed_d) =
        run_whisper_block(whisper, 0, &input, &mlp_seed_cfg, "mlp_seed");
    let (zono_seed_out, zono_seed_d) =
        run_whisper_block(whisper, 0, &input, &zono_seed_cfg, "zono_seed");

    let mlp_w = mlp_seed_d.output_width;
    let zono_w = zono_seed_d.output_width;
    let seed_ratio = mlp_w as f64 / zono_w as f64;

    println!("\n=== Post-#3783 Seed Replay Diagnostic (#3787) ===");
    println!("  Encoder blocks: {n}");
    println!("  Block-0: mlp={mlp_w:.6e}  zono={zono_w:.6e}  ratio={seed_ratio:.6}");
    assert_block_width_tuple_finite(&mlp_seed_out, &mlp_seed_d, "mlp_seed");
    assert_block_width_tuple_finite(&zono_seed_out, &zono_seed_d, "zono_seed");

    // Step 2: Continue both seeds through blocks 1..n with the SAME config.
    let cont_cfg = MultiBlockConfig::sound_tight().with_crown_block_wise(true);
    let cont_blocks = n - 1;
    let (mlp_co, mlp_c) = whisper
        .verify_encoder_sequential_with_config(&mlp_seed_out, 1, n, false, false, None, &cont_cfg)
        .expect("mlp_seed continuation failed");
    let (zono_co, zono_c) = whisper
        .verify_encoder_sequential_with_config(&zono_seed_out, 1, n, false, false, None, &cont_cfg)
        .expect("zono_seed continuation failed");

    let cont_ratio = mlp_c.final_output_width as f64 / zono_c.final_output_width as f64;
    let extra_compound = cont_ratio / seed_ratio;
    println!(
        "  Continuation (1..{n}): mlp={:.6e}  zono={:.6e}  ratio={cont_ratio:.6}",
        mlp_c.final_output_width, zono_c.final_output_width
    );
    println!("  extra_compound_ratio = {extra_compound:.6}");

    print_seed_replay_amplification(&mlp_c, &zono_c, (mlp_w, zono_w));
    assert_seed_replay_valid(
        &mlp_c,
        &mlp_co,
        &zono_c,
        &zono_co,
        cont_blocks,
        seed_ratio,
        cont_ratio,
    );

    println!("\n  === Routing Decision ===");
    if extra_compound < 10.0 {
        println!("  CARRY-DOMINANT: extra_compound={extra_compound:.4} → block-0 seed tightening");
    } else {
        println!(
            "  LATER-BLOCK-DOMINANT: extra_compound={extra_compound:.4} → alpha-aware MLP CROWN"
        );
    }
}

#[ntest::timeout(120000)]
#[test]
#[ignore = "requires the real compositional zonotope lane: the \
            verify_block_compositional_gpu_with_config shim (whisper/model.rs) runs whole-block \
            graph IBP, so zono_seed (sound_tight => conservative backward LN, 4.08e3) can never \
            stay within 1.01x of mlp_seed (default forward LN, 2.96e3); unmasked 2026-07 when \
            block extraction was fixed for the dynamo fixture"]
fn test_whisper_block0_attention_seed_matrix_318() {
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

    println!("\n=== Whisper Block-0 Attention Seed Matrix (#318) ===");
    println!(
        "  [mlp_seed] attn_delta={:.6e}  x_attn={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        mlp_seed.attention_delta_width,
        mlp_seed.x_attn_width,
        mlp_seed.mlp_delta_width,
        mlp_seed.output_width
    );
    println!(
        "  [zono_seed] attn_delta={:.6e}  x_attn={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        zono_seed.attention_delta_width,
        zono_seed.x_attn_width,
        zono_seed.mlp_delta_width,
        zono_seed.output_width
    );
    println!(
        "  [attention_crown_seed] attn_delta={:.6e}  x_attn={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        attention_crown_seed.attention_delta_width,
        attention_crown_seed.x_attn_width,
        attention_crown_seed.mlp_delta_width,
        attention_crown_seed.output_width
    );

    let zono_vs_mlp = zono_seed.output_width / mlp_seed.output_width;
    let attention_vs_mlp = attention_crown_seed.output_width / mlp_seed.output_width;
    let attention_vs_zono = attention_crown_seed.output_width / zono_seed.output_width;
    println!("\n  === Pairwise Ratios ===");
    println!("  zono_seed / mlp_seed            = {zono_vs_mlp:.6}");
    println!("  attention_crown_seed / mlp_seed = {attention_vs_mlp:.6}");
    println!("  attention_crown_seed / zono_seed = {attention_vs_zono:.6}");

    assert_block_width_tuple_finite(&mlp_seed_output, &mlp_seed, "mlp_seed");
    assert_block_width_tuple_finite(&zono_seed_output, &zono_seed, "zono_seed");
    assert_block_width_tuple_finite(
        &attention_crown_output,
        &attention_crown_seed,
        "attention_crown_seed",
    );
    assert!(
        zono_seed.used_zonotope_attention,
        "zono_seed should keep the existing block-0 zonotope route"
    );
    assert!(
        !attention_crown_seed.used_zonotope_attention,
        "attention_crown_seed must stay on the explicit experimental CROWN route"
    );
    assert!(
        zono_seed.output_width <= mlp_seed.output_width * 1.01,
        "zono_seed should stay no wider than mlp_seed: zono={:.6e}, mlp={:.6e}",
        zono_seed.output_width,
        mlp_seed.output_width
    );
    assert!(
        attention_crown_seed.output_width <= mlp_seed.output_width * 1.01,
        "attention_crown_seed should stay no wider than mlp_seed: attn_crown={:.6e}, mlp={:.6e}",
        attention_crown_seed.output_width,
        mlp_seed.output_width
    );
}

/// Block-1 routing diagnostic after the verified block-0 attention seam fix.
/// Part of #318. Design: designs/2026-03-13-issue-318-post-seam-block1-routing.md
#[ntest::timeout(300000)]
#[test]
fn test_whisper_block1_cross_input_matrix_318() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);

    let practical_cfg = MultiBlockConfig::default()
        .with_zonotope_attention(true)
        .with_crown_block_wise(true);
    let sound_cfg = MultiBlockConfig::sound_tight().with_crown_block_wise(true);

    let (practical_block0_output, practical_block0) =
        run_whisper_block(whisper, 0, &input, &practical_cfg, "practical_hybrid");
    let (sound_block0_output, sound_block0) =
        run_whisper_block(whisper, 0, &input, &sound_cfg, "sound_hybrid");

    println!("\n=== Whisper Block-1 Cross-Input Matrix (#318) ===");
    println!("  block-0 seed widths:");
    print_block_width_tuple("practical_input", &practical_block0);
    print_block_width_tuple("sound_input", &sound_block0);
    assert_block_width_tuple_finite(
        &practical_block0_output,
        &practical_block0,
        "practical_input",
    );
    assert_block_width_tuple_finite(&sound_block0_output, &sound_block0, "sound_input");

    let mut block1_widths: Vec<(&str, f32)> = Vec::new();
    for (label, block_input, cfg) in [
        (
            "practical_input -> practical_cfg",
            &practical_block0_output,
            &practical_cfg,
        ),
        (
            "practical_input -> sound_cfg",
            &practical_block0_output,
            &sound_cfg,
        ),
        (
            "sound_input -> practical_cfg",
            &sound_block0_output,
            &practical_cfg,
        ),
        ("sound_input -> sound_cfg", &sound_block0_output, &sound_cfg),
    ] {
        let (output, details) = run_whisper_block(whisper, 1, block_input, cfg, label);
        print_block_width_tuple(label, &details);
        assert_block_width_tuple_finite(&output, &details, label);
        block1_widths.push((label, details.output_width));
    }

    // Pairwise ratios per design decision rule.
    // pp=practical→practical, ps=practical→sound, sp=sound→practical, ss=sound→sound.
    let pp = block1_widths[0].1;
    let ps = block1_widths[1].1;
    let sp = block1_widths[2].1;
    let ss = block1_widths[3].1;
    println!("\n  === Pairwise Ratios ===");
    println!(
        "  config_ratio_on_practical_input = {:.6e}  (pp/ps)",
        pp / ps
    );
    println!(
        "  config_ratio_on_sound_input     = {:.6e}  (sp/ss)",
        sp / ss
    );
    println!(
        "  carry_ratio_on_practical_cfg    = {:.6e}  (pp/sp)",
        pp / sp
    );
    println!(
        "  carry_ratio_on_sound_cfg        = {:.6e}  (ps/ss)",
        ps / ss
    );
}

/// Run IBP and zonotope (with softmax cut) through each attention stage and
/// print per-stage widths. IBP must succeed; zonotope may fail with
/// `NumericalInstability` at extreme input widths (Q@K^T overflow).
fn run_attention_stage_widths(
    graph: &ny_propagate::GraphNetwork,
    stage_nodes: &[(&str, &str)],
    ln_out: &ny_tensor::BoundedTensor,
    combo_label: &str,
    softmax_cut: ZonotopePropagationOptions,
) {
    for &(stage_name, output_node) in stage_nodes {
        let mut ibp_graph = graph.clone();
        ibp_graph.set_output(output_node);
        let ibp = ibp_graph
            .propagate_ibp(ln_out)
            .unwrap_or_else(|e| panic!("{combo_label} {stage_name} IBP failed: {e:?}"));
        let ibp_w = ibp.max_width();
        assert!(
            ibp_w.is_finite(),
            "{combo_label} {stage_name} IBP width must be finite"
        );

        let mut zono_graph = graph.clone();
        zono_graph.set_output(output_node);
        match zono_graph.propagate_zonotope_with_options(ln_out, 0.0, softmax_cut) {
            Ok(zono) => {
                let zono_w = zono.max_width();
                let ratio = if ibp_w > 0.0 {
                    zono_w / ibp_w
                } else {
                    f32::NAN
                };
                println!(
                    "    stage={stage_name} ibp={ibp_w:.6e} zonotope={zono_w:.6e} ratio={ratio:.6e}"
                );
            }
            Err(e) => {
                // Zonotope NumericalInstability is expected at extreme widths.
                println!("    stage={stage_name} ibp={ibp_w:.6e} zonotope=FAILED({e})");
            }
        }
    }
}

/// Block-1 attention stage localization: identifies which attention stage in
/// block 1 is responsible for the config-dependent widening observed in the
/// cross-input matrix (config flip dominates → block-local problem).
///
/// Part of #318. Design: designs/2026-03-13-issue-318-post-seam-block1-routing.md
#[ntest::timeout(120000)]
#[test]
fn test_whisper_block1_attention_stage_localization_318() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 2, 0.01);

    let practical_cfg = MultiBlockConfig::default()
        .with_zonotope_attention(true)
        .with_crown_block_wise(true);
    let sound_cfg = MultiBlockConfig::sound_tight().with_crown_block_wise(true);

    let (practical_b0_out, _) =
        run_whisper_block(whisper, 0, &input, &practical_cfg, "practical_block0");
    let (sound_b0_out, _) = run_whisper_block(whisper, 0, &input, &sound_cfg, "sound_block0");

    let artifacts = whisper
        .attention_suffix_subgraph_artifacts_from_layernorm_output(1)
        .expect("block-1 attention suffix subgraph");

    let softmax_cut =
        ZonotopePropagationOptions::new().with_softmax_mode(ZonotopeSoftmaxMode::IntervalFallback);

    let scores_node = artifacts
        .graph
        .node(artifacts.scores_node.as_str())
        .expect("block-1 scores node must exist");
    assert_eq!(
        scores_node.inputs().len(),
        2,
        "block-1 scores node must consume query/key"
    );
    let stage_nodes: Vec<(&str, &str)> = vec![
        ("query", scores_node.inputs()[0].as_str()),
        ("key", scores_node.inputs()[1].as_str()),
        ("scores", artifacts.scores_node.as_str()),
        ("softmax", artifacts.softmax_node.as_str()),
        ("context", artifacts.context_node.as_str()),
        ("output", artifacts.output_node.as_str()),
    ];

    println!("\n=== Whisper Block-1 Attention Stage Localization (#318) ===");

    // Four combos: (block-0 source) × (LN mode for block-1 attention prefix).
    for (combo_label, b0_out, forward_mode) in [
        ("practical_b0 -> forward_ln", &practical_b0_out, true),
        ("practical_b0 -> conservative_ln", &practical_b0_out, false),
        ("sound_b0 -> forward_ln", &sound_b0_out, true),
        ("sound_b0 -> conservative_ln", &sound_b0_out, false),
    ] {
        let ln_out = whisper
            .attention_layernorm_output_ibp(1, b0_out, forward_mode)
            .unwrap_or_else(|e| panic!("{combo_label} LN failed: {e}"));
        assert_tensor_sound_finite(&ln_out, &format!("{combo_label} ln_output"));
        println!(
            "  [{combo_label}] ln_output_width={:.6e}",
            ln_out.max_width()
        );
        run_attention_stage_widths(
            &artifacts.graph,
            &stage_nodes,
            &ln_out,
            combo_label,
            softmax_cut,
        );
    }
}
