// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential Whisper encoder block verification.
//!
//! Verifies Whisper encoder blocks sequentially with configurable
//! block ranges, overflow handling, and GPU acceleration.

use anyhow::Result;
use ny_onnx::load_whisper;
use std::path::PathBuf;

use super::{make_multiblock_config, make_synthetic_input, resolve_whisper_backend};
use crate::BackendArg;

// Justification: Sequential Whisper verification requires model path, block range,
// architecture flags, tensor dimensions, perturbation config, and backend selection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_whisper_seq_command(
    model: PathBuf,
    start_block: usize,
    end_block: Option<usize>,
    include_stem: bool,
    include_ln_post: bool,
    batch: usize,
    seq_len: usize,
    n_mels: usize,
    time: usize,
    epsilon: f32,
    backend: BackendArg,
    gpu: bool,
    mode: String,
    max_bound_width: Option<f32>,
    terminate_on_overflow: Option<bool>,
    continue_after_overflow: Option<bool>,
    overflow_clamp_value: Option<f32>,
    reset_zonotope_blocks: Option<bool>,
    json: bool,
) -> Result<()> {
    let whisper = load_whisper(&model)?;
    let end_block = end_block.unwrap_or(whisper.encoder_layers);

    let config = make_multiblock_config(
        &mode,
        max_bound_width,
        terminate_on_overflow,
        continue_after_overflow,
        overflow_clamp_value,
        reset_zonotope_blocks,
    )?;

    let input = make_synthetic_input(
        &whisper,
        include_stem,
        batch,
        seq_len,
        n_mels,
        time,
        epsilon,
    )?;

    // Resolve backend and create device
    let (effective_backend, gpu_device) = resolve_whisper_backend(backend, gpu, json);
    let gpu_ref = gpu_device.as_ref();

    if !json {
        println!("Model: {}", model.display());
        println!(
            "Blocks: {}..{} ({} total in model)",
            start_block, end_block, whisper.encoder_layers
        );
        println!(
            "Input: {} (epsilon {})",
            if include_stem {
                format!("[batch={}, n_mels={}, time={}]", batch, n_mels, time)
            } else {
                format!(
                    "[batch={}, seq_len={}, hidden={}]",
                    batch, seq_len, whisper.hidden_dim
                )
            },
            epsilon
        );
        println!(
            "Config: mode={}, max_bound_width={:.2e}, terminate_on_overflow={}, continue_after_overflow={}, overflow_clamp_value={:.2e}, reset_zonotope_blocks={}",
            mode,
            config.max_bound_width,
            config.terminate_on_overflow,
            config.continue_after_overflow,
            config.overflow_clamp_value,
            config.reset_zonotope_between_blocks
        );
        println!(
            "Backend: {} (GPU: {})",
            effective_backend,
            if gpu_ref.is_some() {
                "enabled"
            } else {
                "disabled"
            }
        );
    }

    let (_out, details) = whisper.verify_encoder_sequential_with_config(
        &input,
        start_block,
        end_block,
        include_stem,
        include_ln_post,
        gpu_ref,
        &config,
    )?;

    if json {
        let blocks_json: Vec<_> = details
            .block_details
            .iter()
            .enumerate()
            .map(|(i, b)| {
                serde_json::json!({
                    "block": start_block + i,
                    "attention_delta_width": b.attention_delta_width,
                    "x_attn_width": b.x_attn_width,
                    "mlp_delta_width": b.mlp_delta_width,
                    "output_width": b.output_width,
                    "used_gpu_attention": b.used_gpu_attention,
                    "seq_len": b.seq_len
                })
            })
            .collect();

        println!(
            "{}",
            serde_json::json!({
                "model": model.display().to_string(),
                "start_block": start_block,
                "end_block": end_block,
                "include_stem": include_stem,
                "include_ln_post": include_ln_post,
                "epsilon": epsilon,
                "backend": effective_backend.to_string(),
                "gpu_enabled": gpu_ref.is_some(),
                "config": {
                    "mode": mode,
                    "max_bound_width": config.max_bound_width,
                    "terminate_on_overflow": config.terminate_on_overflow,
                    "continue_after_overflow": config.continue_after_overflow,
                    "overflow_clamp_value": config.overflow_clamp_value,
                    "reset_zonotope_blocks": config.reset_zonotope_between_blocks
                },
                "result": {
                    "blocks_completed": details.blocks_completed,
                    "num_blocks": details.num_blocks,
                    "early_terminated": details.early_terminated,
                    "overflow_at_block": details.overflow_at_block,
                    "termination_reason": details.termination_reason,
                    "final_output_width": details.final_output_width,
                    "total_time_ms": details.total_time_ms
                },
                "blocks": blocks_json
            })
        );
    } else {
        println!("\nResult:");
        println!(
            "blocks_completed={} / {}, early_terminated={}, overflow_at_block={:?}",
            details.blocks_completed,
            details.num_blocks,
            details.early_terminated,
            details.overflow_at_block
        );
        if let Some(reason) = &details.termination_reason {
            println!("termination_reason={}", reason);
        }
        println!("final_output_width={:.6e}", details.final_output_width);
        println!("total_time_ms={}", details.total_time_ms);

        if !details.block_details.is_empty() {
            println!("\nPer-block:");
            println!(
                "{:>6} {:>12} {:>12} {:>12} {:>12} {:>6} {:>6}",
                "block", "attn", "x+attn", "mlp", "out", "gpu", "seq"
            );
            for (i, b) in details.block_details.iter().enumerate() {
                println!(
                    "{:>6} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>6} {:>6}",
                    start_block + i,
                    b.attention_delta_width,
                    b.x_attn_width,
                    b.mlp_delta_width,
                    b.output_width,
                    if b.used_gpu_attention { "yes" } else { "no" },
                    b.seq_len
                );
            }
        }
    }

    Ok(())
}
