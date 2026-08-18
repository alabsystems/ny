// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared arguments for unavailable experimental Whisper compatibility commands.
//!
//! The WhisperSeq, WhisperSweep, and WhisperEpsSearch subcommands share
//! these fields to preserve their CLI shape. Their handlers fail closed before
//! using the model, backend, or verification controls.

use clap::Args;
use std::path::PathBuf;

use super::cli_types::BackendArg;

/// Compatibility arguments for unavailable experimental Whisper commands.
#[derive(Args, Clone, Debug)]
pub(crate) struct WhisperCommonArgs {
    /// Retained model path argument; the unavailable command does not open it
    pub model: PathBuf,

    /// Retained start block; ignored while verification is unavailable
    #[arg(long, default_value_t = 0)]
    pub start_block: usize,

    /// Retained end block; ignored while verification is unavailable
    #[arg(long)]
    pub end_block: Option<usize>,

    /// Retained stem flag; ignored while verification is unavailable
    #[arg(long, default_value_t = false)]
    pub include_stem: bool,

    /// Retained final-LayerNorm flag; ignored while verification is unavailable
    #[arg(long, default_value_t = false)]
    pub include_ln_post: bool,

    /// Retained batch size; ignored while verification is unavailable
    #[arg(long, default_value_t = 1)]
    pub batch: usize,

    /// Retained sequence length; ignored while verification is unavailable
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Retained mel-bin count; ignored while verification is unavailable
    #[arg(long, default_value_t = 80)]
    pub n_mels: usize,

    /// Retained time dimension; ignored while verification is unavailable
    #[arg(long, default_value_t = 3000)]
    pub time: usize,

    /// Retained backend selector; no backend is initialized
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    pub backend: BackendArg,

    /// Retained deprecated GPU flag; no device is initialized
    #[arg(long, default_value_t = false, hide = true)]
    pub gpu: bool,

    /// Retained width limit; ignored while verification is unavailable
    #[arg(long)]
    pub max_bound_width: Option<f32>,

    /// Retained zonotope reset flag; no zonotope lane executes
    #[arg(
        long,
        value_parser = clap::value_parser!(bool),
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub reset_zonotope_blocks: Option<bool>,

    /// Retained compatibility flag; the fail-closed error uses normal CLI text
    #[arg(long, default_value_t = false)]
    pub json: bool,
}
