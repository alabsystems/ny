// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Whisper export and unavailable verification compatibility handlers.
//!
//! Export script generation is supported. The verdict-oriented Whisper
//! compatibility commands retain their CLI shape but contain no verifier
//! fallback: they fail closed before model or device work.

mod eps_search;
mod sequential;
mod sweep;
#[cfg(test)]
mod tests;

use anyhow::Result;
use std::path::PathBuf;

pub(crate) use eps_search::handle_whisper_eps_search_command;
pub(crate) use sequential::handle_whisper_seq_command;
pub(crate) use sweep::handle_whisper_sweep_command;

pub(super) fn whisper_verification_unavailable() -> Result<()> {
    anyhow::bail!(
        "Whisper verification is unavailable: verifier execution is not implemented; \
         no verification verdict was produced"
    )
}

// Justification: CLI handler — parameters preserve the historical clap surface.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_whisper_command(
    _model: PathBuf,
    _component: String,
    _layer: Option<usize>,
    _epsilon: f32,
    _json: bool,
) -> Result<()> {
    whisper_verification_unavailable()
}

pub(crate) fn handle_export_command(
    model_type: String,
    size: String,
    output: Option<PathBuf>,
) -> Result<()> {
    const SUPPORTED_WHISPER_SIZES: &[&str] = &["tiny", "base", "small", "medium", "large"];
    let script = match model_type.as_str() {
        "whisper" if SUPPORTED_WHISPER_SIZES.contains(&size.as_str()) => {
            ny_onnx::generate_whisper_export_script(&size)
        }
        "whisper" => {
            anyhow::bail!(
                "Unsupported Whisper model size '{size}'; expected one of: {}",
                SUPPORTED_WHISPER_SIZES.join(", ")
            );
        }
        _ => {
            anyhow::bail!("Unsupported export model type '{model_type}'; expected 'whisper'");
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
