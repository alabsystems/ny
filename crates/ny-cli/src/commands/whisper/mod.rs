// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Whisper-specific command handlers.
//!
//! Handles Whisper model verification commands including:
//! - Component verification (encoder, attention)
//! - Sequential block verification
//! - Epsilon sweep and binary search
//! - ONNX export script generation

mod eps_search;
mod sequential;
mod sweep;
#[cfg(test)]
mod tests;

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_gpu::{Backend, ComputeDevice};
use ny_onnx::load_whisper;
use ny_tensor::BoundedTensor;
use std::path::PathBuf;
use tracing::info;

use super::backend::resolve_backend;
use crate::BackendArg;

pub(crate) use eps_search::handle_whisper_eps_search_command;
pub(crate) use sequential::handle_whisper_seq_command;
pub(crate) use sweep::handle_whisper_sweep_command;

// Justification: CLI handler — parameters map to clap arguments for Whisper verification.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_whisper_command(
    model: PathBuf,
    component: String,
    layer: Option<usize>,
    epsilon: f32,
    json: bool,
) -> Result<()> {
    if !json {
        info!("Verifying Whisper component: {}", component);
    }

    let whisper = load_whisper(&model)?;
    if !json {
        info!(
            "Loaded Whisper model: {} encoder layers, {} decoder layers",
            whisper.encoder_layers, whisper.decoder_layers
        );
    }

    let network = match component.as_str() {
        "encoder" => {
            if let Some(idx) = layer {
                whisper.encoder_layer(idx)?
            } else {
                whisper.encoder()?
            }
        }
        "attention" => {
            let layer_idx = layer.unwrap_or(0);
            whisper.attention_head(layer_idx, 0)?
        }
        _ => {
            anyhow::bail!("Unknown component: {}. Use encoder or attention", component);
        }
    };

    if json {
        println!(
            "{}",
            serde_json::json!({
                "model": model.display().to_string(),
                "component": component,
                "layer": layer,
                "epsilon": epsilon,
                "num_layers": network.num_layers(),
                "encoder_layers": whisper.encoder_layers,
                "decoder_layers": whisper.decoder_layers,
                "hidden_dim": whisper.hidden_dim,
                "status": "not_implemented",
                "message": "Verification not yet fully implemented for Whisper"
            })
        );
    } else {
        println!(
            "Component to verify: {} (layers: {})",
            component,
            network.num_layers()
        );
        println!("Perturbation epsilon: {}", epsilon);
        println!("\nVerification not yet fully implemented for Whisper");
    }

    Ok(())
}

pub(crate) fn handle_export_command(
    model_type: String,
    size: String,
    output: Option<PathBuf>,
) -> Result<()> {
    let script = match model_type.as_str() {
        "whisper" => ny_onnx::generate_whisper_export_script(&size),
        _ => {
            anyhow::bail!("Unknown model type: {}", model_type);
        }
    };

    if let Some(path) = output {
        std::fs::write(&path, &script)?;
        println!("Export script written to: {}", path.display());
    } else {
        println!("{}", script);
    }

    Ok(())
}

pub(super) fn make_synthetic_input(
    whisper: &ny_onnx::WhisperModel,
    include_stem: bool,
    batch: usize,
    seq_len: usize,
    n_mels: usize,
    time: usize,
    epsilon: f32,
) -> Result<BoundedTensor> {
    if include_stem {
        let data = ArrayD::from_elem(IxDyn(&[batch, n_mels, time]), 0.0f32);
        Ok(BoundedTensor::from_epsilon(data, epsilon)?)
    } else {
        let hidden_dim = whisper.hidden_dim;
        let data = ArrayD::from_elem(IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
        Ok(BoundedTensor::from_epsilon(data, epsilon)?)
    }
}

pub(super) fn make_multiblock_config(
    mode: &str,
    max_bound_width: Option<f32>,
    terminate_on_overflow: Option<bool>,
    continue_after_overflow: Option<bool>,
    overflow_clamp_value: Option<f32>,
    reset_zonotope_between_blocks: Option<bool>,
) -> Result<ny_onnx::MultiBlockConfig> {
    let mut config = match mode {
        "default" => ny_onnx::MultiBlockConfig::default(),
        "strict" => ny_onnx::MultiBlockConfig::strict(),
        "diagnostic" => ny_onnx::MultiBlockConfig::diagnostic(),
        "sound-tight" => ny_onnx::MultiBlockConfig::sound_tight(),
        _ => anyhow::bail!(
            "Unknown mode: {}. Use default, strict, diagnostic, or sound-tight",
            mode
        ),
    };

    if let Some(max_width) = max_bound_width {
        config.max_bound_width = max_width;
    }
    if let Some(v) = terminate_on_overflow {
        config.terminate_on_overflow = v;
    }
    if let Some(v) = continue_after_overflow {
        config.continue_after_overflow = v;
    }
    if let Some(v) = overflow_clamp_value {
        config.overflow_clamp_value = v;
    }
    if let Some(v) = reset_zonotope_between_blocks {
        config.reset_zonotope_between_blocks = v;
    }

    Ok(config)
}

pub(super) fn resolve_whisper_backend(
    backend: BackendArg,
    gpu: bool,
    json: bool,
) -> (BackendArg, Option<ComputeDevice>) {
    let mut effective_backend = resolve_backend(backend, gpu);
    let device = match effective_backend {
        BackendArg::Cpu => None,
        BackendArg::Wgpu => match ComputeDevice::new(Backend::Wgpu) {
            Ok(dev) => Some(dev),
            Err(e) => {
                if !json {
                    eprintln!("WGPU backend not available: {}. Using CPU.", e);
                }
                None
            }
        },
    };

    if device.is_none() && effective_backend != BackendArg::Cpu {
        effective_backend = BackendArg::Cpu;
    }

    (effective_backend, device)
}

pub(super) fn eps_sweep(
    epsilon_min: f32,
    epsilon_max: f32,
    steps: usize,
    linear: bool,
) -> Result<Vec<f32>> {
    if steps == 0 {
        anyhow::bail!("steps must be >= 1");
    }
    if steps == 1 {
        return Ok(vec![epsilon_min]);
    }
    if epsilon_min.is_nan() || epsilon_max.is_nan() || epsilon_min <= 0.0 || epsilon_max <= 0.0 {
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

    let mut eps = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        let v = if linear {
            epsilon_min + t * (epsilon_max - epsilon_min)
        } else {
            let ratio = epsilon_max / epsilon_min;
            epsilon_min * ratio.powf(t)
        };
        eps.push(v);
    }
    Ok(eps)
}
