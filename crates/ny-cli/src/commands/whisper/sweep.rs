// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Whisper epsilon sweep verification.
//!
//! Sweeps over a range of epsilon values (linear or logarithmic) to
//! characterize how bound widths grow with perturbation magnitude.

use anyhow::Result;
use ny_onnx::load_whisper;
use std::path::PathBuf;
use tracing::info;

use super::{eps_sweep, make_multiblock_config, make_synthetic_input, resolve_whisper_backend};
use crate::BackendArg;

// Justification: Epsilon sweep over Whisper blocks needs model path, block range,
// architecture flags, tensor dimensions, sweep range, and backend — all from CLI.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_whisper_sweep_command(
    model: PathBuf,
    start_block: usize,
    end_block: Option<usize>,
    include_stem: bool,
    include_ln_post: bool,
    batch: usize,
    seq_len: usize,
    n_mels: usize,
    time: usize,
    epsilon_min: f32,
    epsilon_max: f32,
    steps: usize,
    linear: bool,
    backend: BackendArg,
    gpu: bool,
    mode: String,
    max_bound_width: Option<f32>,
    reset_zonotope_blocks: Option<bool>,
    per_block: bool,
    json: bool,
) -> Result<()> {
    let whisper = load_whisper(&model)?;
    let end_block = end_block.unwrap_or(whisper.encoder_layers);

    let mut config = make_multiblock_config(
        &mode,
        max_bound_width,
        None,
        None,
        None,
        reset_zonotope_blocks,
    )?;

    // If the caller left strict defaults but removed the threshold, enforce a reasonable default
    // so the sweep terminates before f32 overflow.
    if mode == "strict" && config.max_bound_width == f32::MAX {
        config.max_bound_width = 1e20;
    }

    // Resolve backend and create device
    let (effective_backend, gpu_device) = resolve_whisper_backend(backend, gpu, json);
    let gpu_ref = gpu_device.as_ref();

    let eps_list = eps_sweep(epsilon_min, epsilon_max, steps, linear)?;

    if !json {
        println!("Model: {}", model.display());
        println!(
            "Blocks: {}..{} ({} blocks requested)",
            start_block,
            end_block,
            end_block.saturating_sub(start_block)
        );
        println!(
            "Input: {}",
            if include_stem {
                format!("[batch={}, n_mels={}, time={}]", batch, n_mels, time)
            } else {
                format!(
                    "[batch={}, seq_len={}, hidden={}]",
                    batch, seq_len, whisper.hidden_dim
                )
            }
        );
        println!(
            "Sweep: {} points, {} space, eps in [{:.2e}, {:.2e}]",
            steps,
            if linear { "linear" } else { "log" },
            epsilon_min,
            epsilon_max
        );
        println!(
            "Config: mode={}, max_bound_width={:.2e}, terminate_on_overflow={}, continue_after_overflow={}, reset_zonotope_blocks={}",
            mode,
            config.max_bound_width,
            config.terminate_on_overflow,
            config.continue_after_overflow,
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

        println!(
            "\n{:>12} {:>8} {:>8} {:>12} {:>12} {:>10}",
            "epsilon", "done", "early", "overflow", "final_w", "time_ms"
        );
    }

    // Collect results for JSON output
    let mut sweep_results = Vec::new();

    for eps in &eps_list {
        let input =
            make_synthetic_input(&whisper, include_stem, batch, seq_len, n_mels, time, *eps)?;

        let res = whisper.verify_encoder_sequential_with_config(
            &input,
            start_block,
            end_block,
            include_stem,
            include_ln_post,
            gpu_ref,
            &config,
        );

        match res {
            Ok((_out, details)) => {
                if json {
                    let blocks_json: Vec<_> = if per_block {
                        details
                            .block_details
                            .iter()
                            .enumerate()
                            .map(|(i, b)| {
                                serde_json::json!({
                                    "block": start_block + i,
                                    "attention_delta_width": b.attention_delta_width,
                                    "x_attn_width": b.x_attn_width,
                                    "mlp_delta_width": b.mlp_delta_width,
                                    "output_width": b.output_width
                                })
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };

                    sweep_results.push(serde_json::json!({
                        "epsilon": eps,
                        "blocks_completed": details.blocks_completed,
                        "num_blocks": details.num_blocks,
                        "early_terminated": details.early_terminated,
                        "overflow_at_block": details.overflow_at_block,
                        "final_output_width": details.final_output_width,
                        "total_time_ms": details.total_time_ms,
                        "blocks": blocks_json,
                        "error": serde_json::Value::Null
                    }));
                } else {
                    println!(
                        "{:>12.3e} {:>8} {:>8} {:>12} {:>12.3e} {:>10}",
                        eps,
                        format!("{}/{}", details.blocks_completed, details.num_blocks),
                        if details.early_terminated {
                            "yes"
                        } else {
                            "no"
                        },
                        details
                            .overflow_at_block
                            .map(|b| b.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                        details.final_output_width,
                        details.total_time_ms
                    );

                    // Print per-block widths if requested
                    if per_block && !details.block_details.is_empty() {
                        for (i, b) in details.block_details.iter().enumerate() {
                            println!(
                                "      block[{}]: attn={:.2e} x+attn={:.2e} mlp={:.2e} out={:.2e}",
                                start_block + i,
                                b.attention_delta_width,
                                b.x_attn_width,
                                b.mlp_delta_width,
                                b.output_width
                            );
                        }
                    }
                }
            }
            Err(e) => {
                if json {
                    sweep_results.push(serde_json::json!({
                        "epsilon": eps,
                        "error": e.to_string()
                    }));
                } else {
                    println!(
                        "{:>12.3e} {:>8} {:>8} {:>12} {:>12} {:>10}",
                        eps, "err", "-", "-", "-", "-"
                    );
                    info!("epsilon {} failed: {:?}", eps, e);
                }
            }
        }
    }

    if json {
        println!(
            "{}",
            serde_json::json!({
                "model": model.display().to_string(),
                "start_block": start_block,
                "end_block": end_block,
                "include_stem": include_stem,
                "include_ln_post": include_ln_post,
                "backend": effective_backend.to_string(),
                "gpu_enabled": gpu_ref.is_some(),
                "sweep": {
                    "epsilon_min": epsilon_min,
                    "epsilon_max": epsilon_max,
                    "steps": steps,
                    "linear": linear
                },
                "config": {
                    "mode": mode,
                    "max_bound_width": config.max_bound_width,
                    "terminate_on_overflow": config.terminate_on_overflow,
                    "continue_after_overflow": config.continue_after_overflow,
                    "reset_zonotope_blocks": config.reset_zonotope_between_blocks
                },
                "results": sweep_results
            })
        );
    }

    Ok(())
}
