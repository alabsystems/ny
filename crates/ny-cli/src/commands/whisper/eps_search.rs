// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Binary search for maximum verifiable epsilon.
//!
//! Performs binary search in log-space to find the largest epsilon
//! value for which Whisper encoder verification completes successfully.

use anyhow::Result;
use ny_onnx::load_whisper;
use std::path::PathBuf;
use tracing::info;

use super::{make_multiblock_config, make_synthetic_input, resolve_whisper_backend};
use crate::BackendArg;

// Justification: Binary search for max verifiable epsilon requires all Whisper config
// parameters plus search-specific settings (min/max eps, tolerance, target blocks).
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_whisper_eps_search_command(
    model: PathBuf,
    start_block: usize,
    end_block: Option<usize>,
    target_blocks: Option<usize>,
    include_stem: bool,
    include_ln_post: bool,
    batch: usize,
    seq_len: usize,
    n_mels: usize,
    time: usize,
    epsilon_min: f32,
    epsilon_max: f32,
    iterations: usize,
    backend: BackendArg,
    gpu: bool,
    mode: String,
    max_bound_width: Option<f32>,
    reset_zonotope_blocks: Option<bool>,
    verbose_search: bool,
    json: bool,
) -> Result<()> {
    let whisper = load_whisper(&model)?;
    let end_block = end_block.unwrap_or(whisper.encoder_layers);
    let num_blocks = end_block.saturating_sub(start_block);
    let target_blocks = target_blocks.unwrap_or(num_blocks);

    if iterations == 0 {
        anyhow::bail!("iterations must be >= 1");
    }
    if !epsilon_min.is_finite() || !epsilon_max.is_finite() {
        anyhow::bail!(
            "epsilon_min and epsilon_max must be finite (got {}..{})",
            epsilon_min,
            epsilon_max
        );
    }
    if epsilon_min <= 0.0 || epsilon_max <= 0.0 {
        anyhow::bail!(
            "epsilon_min and epsilon_max must be > 0 (got {}..{})",
            epsilon_min,
            epsilon_max
        );
    }
    if epsilon_min > epsilon_max {
        anyhow::bail!(
            "epsilon_min must be <= epsilon_max (got {}..{})",
            epsilon_min,
            epsilon_max
        );
    }

    if target_blocks == 0 || target_blocks > num_blocks {
        anyhow::bail!(
            "target_blocks must be in [1, {}] (got {})",
            num_blocks,
            target_blocks
        );
    }

    let mut config = make_multiblock_config(
        &mode,
        max_bound_width,
        None,
        None,
        None,
        reset_zonotope_blocks,
    )?;
    if mode == "strict" && config.max_bound_width == f32::MAX {
        config.max_bound_width = 1e20;
    }

    // Resolve backend and create device
    let (effective_backend, gpu_device) = resolve_whisper_backend(backend, gpu, json);
    let gpu_ref = gpu_device.as_ref();

    if !json {
        println!("Model: {}", model.display());
        println!(
            "Blocks: {}..{} ({} blocks), target={} to complete",
            start_block, end_block, num_blocks, target_blocks
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
            "Search: {} iterations in [{:.2e}, {:.2e}]",
            iterations, epsilon_min, epsilon_max
        );
        println!(
            "Config: mode={}, max_bound_width={:.2e}, terminate_on_overflow={}, reset_zonotope_blocks={}",
            mode,
            config.max_bound_width,
            config.terminate_on_overflow,
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

    // Binary search in log space for numerical stability
    let mut log_low = epsilon_min.ln();
    let mut log_high = epsilon_max.ln();
    let mut best_eps: Option<f32> = None;
    let mut best_details: Option<ny_onnx::MultiBlockDetails> = None;

    // Collect search history for JSON output
    let mut search_history: Vec<serde_json::Value> = Vec::new();

    // Helper to test an epsilon value
    let test_eps = |eps: f32| -> Result<(bool, ny_onnx::MultiBlockDetails)> {
        let input =
            make_synthetic_input(&whisper, include_stem, batch, seq_len, n_mels, time, eps)?;
        let (_out, details) = whisper.verify_encoder_sequential_with_config(
            &input,
            start_block,
            end_block,
            include_stem,
            include_ln_post,
            gpu_ref,
            &config,
        )?;
        let success = details.blocks_completed >= target_blocks && !details.early_terminated;
        Ok((success, details))
    };

    if verbose_search && !json {
        println!(
            "\n{:>5} {:>12} {:>8} {:>8} {:>12} {:>10}",
            "iter", "epsilon", "done", "success", "final_w", "time_ms"
        );
    }

    for iter in 0..iterations {
        let log_mid = f32::midpoint(log_low, log_high);
        let eps_mid = log_mid.exp();

        match test_eps(eps_mid) {
            Ok((success, details)) => {
                if json && verbose_search {
                    search_history.push(serde_json::json!({
                        "iteration": iter,
                        "epsilon": eps_mid,
                        "blocks_completed": details.blocks_completed,
                        "num_blocks": details.num_blocks,
                        "success": success,
                        "final_output_width": details.final_output_width,
                        "total_time_ms": details.total_time_ms,
                        "error": serde_json::Value::Null
                    }));
                } else if verbose_search {
                    println!(
                        "{:>5} {:>12.3e} {:>8} {:>8} {:>12.3e} {:>10}",
                        iter,
                        eps_mid,
                        format!("{}/{}", details.blocks_completed, details.num_blocks),
                        if success { "yes" } else { "no" },
                        details.final_output_width,
                        details.total_time_ms
                    );
                }

                if success {
                    // Can go higher
                    best_eps = Some(eps_mid);
                    best_details = Some(details);
                    log_low = log_mid;
                } else {
                    // Need to go lower
                    log_high = log_mid;
                }
            }
            Err(e) => {
                if json && verbose_search {
                    search_history.push(serde_json::json!({
                        "iteration": iter,
                        "epsilon": eps_mid,
                        "success": false,
                        "error": e.to_string()
                    }));
                } else if verbose_search {
                    println!(
                        "{:>5} {:>12.3e} {:>8} {:>8} {:>12} {:>10}",
                        iter, eps_mid, "err", "no", "-", "-"
                    );
                }
                if !json {
                    info!("epsilon {} failed: {:?}", eps_mid, e);
                }
                // Treat errors as needing to go lower
                log_high = log_mid;
            }
        }
    }

    // Check if epsilon_max also succeeds
    let epsilon_max_succeeds = test_eps(epsilon_max).map(|(s, _)| s).unwrap_or(false);

    if json {
        let result_json = match (best_eps, &best_details) {
            (Some(eps), Some(details)) => {
                serde_json::json!({
                    "found": true,
                    "max_epsilon": eps,
                    "blocks_completed": details.blocks_completed,
                    "num_blocks": details.num_blocks,
                    "target_blocks": target_blocks,
                    "final_output_width": details.final_output_width,
                    "total_time_ms": details.total_time_ms,
                    "epsilon_max_succeeds": epsilon_max_succeeds
                })
            }
            _ => {
                serde_json::json!({
                    "found": false,
                    "target_blocks": target_blocks,
                    "suggestion": "Try lowering epsilon_min or increasing target_blocks tolerance"
                })
            }
        };

        println!(
            "{}",
            serde_json::json!({
                "model": model.display().to_string(),
                "start_block": start_block,
                "end_block": end_block,
                "num_blocks": num_blocks,
                "target_blocks": target_blocks,
                "include_stem": include_stem,
                "include_ln_post": include_ln_post,
                "backend": effective_backend.to_string(),
                "gpu_enabled": gpu_ref.is_some(),
                "search": {
                    "epsilon_min": epsilon_min,
                    "epsilon_max": epsilon_max,
                    "iterations": iterations
                },
                "config": {
                    "mode": mode,
                    "max_bound_width": config.max_bound_width,
                    "terminate_on_overflow": config.terminate_on_overflow,
                    "reset_zonotope_blocks": config.reset_zonotope_between_blocks
                },
                "history": if verbose_search { search_history } else { Vec::new() },
                "result": result_json
            })
        );
    } else {
        println!("\n--- Search Result ---");
        match (best_eps, best_details) {
            (Some(eps), Some(details)) => {
                println!("max_epsilon={:.6e}", eps);
                println!(
                    "blocks_completed={}/{}, target={}",
                    details.blocks_completed, details.num_blocks, target_blocks
                );
                println!("final_output_width={:.6e}", details.final_output_width);
                println!("total_time_ms={}", details.total_time_ms);

                // Check if we should also test bounds
                if epsilon_max_succeeds {
                    println!(
                        "\nNote: epsilon_max ({:.2e}) also succeeds. Max may be higher.",
                        epsilon_max
                    );
                }
            }
            _ => {
                println!(
                    "No epsilon found in [{:.2e}, {:.2e}] that completes {} blocks.",
                    epsilon_min, epsilon_max, target_blocks
                );
                println!("Try lowering epsilon_min or increasing target_blocks tolerance.");
            }
        }
    }

    Ok(())
}
