// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-block weight norm computation (Frobenius and spectral norms).
//!
//! Used by `ny weights norms` to analyze weight sensitivity per transformer block.

use anyhow::Result;
use ny_core::{nan_propagating_max_f64, nan_propagating_min_f64};
use serde_json::json;
use std::collections::BTreeMap;

/// Per-block weight norm statistics
struct BlockNorms {
    block_index: usize,
    attn_q_frobenius: f64,
    attn_q_spectral: f64,
    attn_k_frobenius: f64,
    attn_k_spectral: f64,
    attn_v_frobenius: f64,
    attn_v_spectral: f64,
    attn_output_frobenius: f64,
    attn_output_spectral: f64,
    ffn_up_frobenius: f64,
    ffn_up_spectral: f64,
    ffn_down_frobenius: f64,
    ffn_down_spectral: f64,
    ffn_gate_frobenius: f64,
    ffn_gate_spectral: f64,
    total_frobenius: f64,
    max_spectral: f64,
}

/// Compute Frobenius norm of a tensor: ||A||_F = sqrt(sum(a_ij^2))
pub(super) fn frobenius_norm(tensor: &ndarray::ArrayD<f32>) -> f64 {
    tensor
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Approximate spectral norm via power iteration (largest singular value)
/// For matrix A, spectral norm = max singular value = sqrt(max eigenvalue of A^T A)
pub(super) fn spectral_norm_approx(tensor: &ndarray::ArrayD<f32>, iterations: usize) -> f64 {
    // Only works for 2D matrices
    if tensor.ndim() != 2 {
        return frobenius_norm(tensor); // Fallback for non-2D
    }

    let shape = tensor.shape();
    let (m, n) = (shape[0], shape[1]);

    // Reshape to 2D for matrix operations
    let Ok(matrix) = tensor.view().into_shape_with_order((m, n)) else {
        return frobenius_norm(tensor);
    };

    // Initialize random vector
    let mut v: Vec<f64> = (0..n)
        .map(|i| ((i * 31337) % 1000) as f64 / 1000.0)
        .collect();

    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }

    // Power iteration: v = A^T A v / ||A^T A v||
    for _ in 0..iterations {
        // u = A v
        let mut u = vec![0.0f64; m];
        for i in 0..m {
            for j in 0..n {
                u[i] += matrix[[i, j]] as f64 * v[j];
            }
        }

        // v = A^T u
        let mut v_new = vec![0.0f64; n];
        for j in 0..n {
            for i in 0..m {
                v_new[j] += matrix[[i, j]] as f64 * u[i];
            }
        }

        // Normalize
        let norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return 0.0;
        }
        for x in &mut v_new {
            *x /= norm;
        }
        v = v_new;
    }

    // Compute ||Av|| as estimate of largest singular value
    let mut av = vec![0.0f64; m];
    for i in 0..m {
        for j in 0..n {
            av[i] += matrix[[i, j]] as f64 * v[j];
        }
    }
    av.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Extract block number from GGUF-style name (e.g., "blk.5.attn_q.weight" -> 5)
pub(super) fn extract_gguf_block_number(name: &str) -> Option<usize> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        num_str.parse().ok()
    } else {
        None
    }
}

/// Extract block number from Whisper/HF-style name (e.g., "encoder.layers.5.self_attn.q_proj.weight" -> 5)
pub(super) fn extract_hf_block_number(name: &str) -> Option<usize> {
    // Try patterns like "encoder.layers.N", "decoder.layers.N", "model.layers.N", "layers.N"
    let patterns = [
        "encoder.layers.",
        "decoder.layers.",
        "model.layers.",
        "layers.",
    ];
    for pattern in patterns {
        if let Some(rest) = name.find(pattern) {
            let after_pattern = &name[rest + pattern.len()..];
            let num_str: String = after_pattern
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = num_str.parse() {
                return Some(n);
            }
        }
    }
    None
}

/// Compute per-block weight norms for a model
fn compute_block_norms(weights: &ny_onnx::WeightStore) -> Vec<BlockNorms> {
    // Determine naming convention by checking first weight name
    let names: Vec<&str> = weights.keys().collect();
    let is_gguf_style = names.iter().any(|n| n.starts_with("blk."));

    // Group weights by block
    let mut block_weights: BTreeMap<usize, Vec<(&str, &ndarray::ArrayD<f32>)>> = BTreeMap::new();

    for (name, tensor) in weights.iter() {
        let block_num = if is_gguf_style {
            extract_gguf_block_number(name)
        } else {
            extract_hf_block_number(name)
        };

        if let Some(n) = block_num {
            block_weights.entry(n).or_default().push((name, tensor));
        }
    }

    // For each block, compute norms
    let mut results = Vec::new();

    for (&block_idx, tensors) in &block_weights {
        let mut norms = BlockNorms {
            block_index: block_idx,
            attn_q_frobenius: 0.0,
            attn_q_spectral: 0.0,
            attn_k_frobenius: 0.0,
            attn_k_spectral: 0.0,
            attn_v_frobenius: 0.0,
            attn_v_spectral: 0.0,
            attn_output_frobenius: 0.0,
            attn_output_spectral: 0.0,
            ffn_up_frobenius: 0.0,
            ffn_up_spectral: 0.0,
            ffn_down_frobenius: 0.0,
            ffn_down_spectral: 0.0,
            ffn_gate_frobenius: 0.0,
            ffn_gate_spectral: 0.0,
            total_frobenius: 0.0,
            max_spectral: 0.0,
        };

        for (name, tensor) in tensors {
            let frob = frobenius_norm(tensor);
            let spec = if tensor.ndim() == 2 {
                spectral_norm_approx(tensor, 20) // 20 iterations
            } else {
                frob // For non-2D, just use Frobenius
            };

            // Categorize by weight name pattern
            let name_lower = name.to_lowercase();

            if name_lower.contains("attn_q") || name_lower.contains("q_proj") {
                norms.attn_q_frobenius = frob;
                norms.attn_q_spectral = spec;
            } else if name_lower.contains("attn_k") || name_lower.contains("k_proj") {
                norms.attn_k_frobenius = frob;
                norms.attn_k_spectral = spec;
            } else if name_lower.contains("attn_v") || name_lower.contains("v_proj") {
                norms.attn_v_frobenius = frob;
                norms.attn_v_spectral = spec;
            } else if name_lower.contains("attn_output")
                || name_lower.contains("out_proj")
                || name_lower.contains("o_proj")
            {
                norms.attn_output_frobenius = frob;
                norms.attn_output_spectral = spec;
            } else if name_lower.contains("ffn_up")
                || name_lower.contains("up_proj")
                || name_lower.contains("fc1")
            {
                norms.ffn_up_frobenius = frob;
                norms.ffn_up_spectral = spec;
            } else if name_lower.contains("ffn_down")
                || name_lower.contains("down_proj")
                || name_lower.contains("fc2")
            {
                norms.ffn_down_frobenius = frob;
                norms.ffn_down_spectral = spec;
            } else if name_lower.contains("ffn_gate") || name_lower.contains("gate_proj") {
                norms.ffn_gate_frobenius = frob;
                norms.ffn_gate_spectral = spec;
            }

            norms.total_frobenius += frob * frob; // Sum squares for total
            norms.max_spectral = nan_propagating_max_f64(norms.max_spectral, spec);
        }

        norms.total_frobenius = norms.total_frobenius.sqrt(); // sqrt of sum of squares
        results.push(norms);
    }

    results
}

/// Handle the Weights Norms subcommand - compute per-block weight norms
pub(super) fn handle_weights_norms(file: &std::path::Path, json_output: bool) -> Result<()> {
    let weights = super::load_weights_from_file(file)?;

    // Detect block naming pattern and extract per-block norms
    let block_norms = compute_block_norms(&weights);

    if json_output {
        let blocks_json: Vec<_> = block_norms
            .iter()
            .map(|b| {
                json!({
                    "block": b.block_index,
                    "attn_q_frobenius": b.attn_q_frobenius,
                    "attn_q_spectral": b.attn_q_spectral,
                    "attn_k_frobenius": b.attn_k_frobenius,
                    "attn_k_spectral": b.attn_k_spectral,
                    "attn_v_frobenius": b.attn_v_frobenius,
                    "attn_v_spectral": b.attn_v_spectral,
                    "attn_output_frobenius": b.attn_output_frobenius,
                    "attn_output_spectral": b.attn_output_spectral,
                    "ffn_up_frobenius": b.ffn_up_frobenius,
                    "ffn_up_spectral": b.ffn_up_spectral,
                    "ffn_down_frobenius": b.ffn_down_frobenius,
                    "ffn_down_spectral": b.ffn_down_spectral,
                    "ffn_gate_frobenius": b.ffn_gate_frobenius,
                    "ffn_gate_spectral": b.ffn_gate_spectral,
                    "total_frobenius": b.total_frobenius,
                    "max_spectral": b.max_spectral,
                })
            })
            .collect();

        // Compute summary stats
        let total_frobenius_values: Vec<f64> =
            block_norms.iter().map(|b| b.total_frobenius).collect();
        let max_spectral_values: Vec<f64> = block_norms.iter().map(|b| b.max_spectral).collect();

        let max_total_frob = total_frobenius_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, nan_propagating_max_f64);
        let min_total_frob = total_frobenius_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, nan_propagating_min_f64);
        let max_spec = max_spectral_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, nan_propagating_max_f64);
        let min_spec = max_spectral_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, nan_propagating_min_f64);
        let frob_range = if min_total_frob > 0.0 {
            max_total_frob / min_total_frob
        } else {
            f64::INFINITY
        };
        let spec_range = if min_spec > 0.0 {
            max_spec / min_spec
        } else {
            f64::INFINITY
        };

        let output = json!({
            "file": file.to_string_lossy(),
            "block_count": block_norms.len(),
            "blocks": blocks_json,
            "summary": {
                "max_total_frobenius": max_total_frob,
                "min_total_frobenius": min_total_frob,
                "max_spectral": max_spec,
                "min_spectral": min_spec,
                "frobenius_range": frob_range,
                "spectral_range": spec_range,
            }
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Weight Norms Analysis");
        println!("File: {}", file.display());
        println!("Blocks: {}\n", block_norms.len());

        println!(
            "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "Block", "Attn_out_F", "FFN_down_F", "FFN_up_F", "Max_Spec", "Total_F"
        );
        println!("{}", "-".repeat(72));

        for b in &block_norms {
            println!(
                "{:>6} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}",
                b.block_index,
                b.attn_output_frobenius,
                b.ffn_down_frobenius,
                b.ffn_up_frobenius,
                b.max_spectral,
                b.total_frobenius
            );
        }

        // Summary
        let total_frobenius_values: Vec<f64> =
            block_norms.iter().map(|b| b.total_frobenius).collect();
        let max_spectral_values: Vec<f64> = block_norms.iter().map(|b| b.max_spectral).collect();

        let max_frob = total_frobenius_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, nan_propagating_max_f64);
        let min_frob = total_frobenius_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, nan_propagating_min_f64);
        let max_spec = max_spectral_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, nan_propagating_max_f64);
        let min_spec = max_spectral_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, nan_propagating_min_f64);

        println!("\nSummary:");
        println!(
            "  Total Frobenius range: {:.4e} - {:.4e} ({:.2}x)",
            min_frob,
            max_frob,
            max_frob / min_frob
        );
        println!(
            "  Max Spectral range: {:.4e} - {:.4e} ({:.2}x)",
            min_spec,
            max_spec,
            max_spec / min_spec
        );
    }

    Ok(())
}
