// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared Whisper CLI arguments.
//!
//! The WhisperSeq, WhisperSweep, and WhisperEpsSearch subcommands share
//! ~14 identical fields. This struct is `#[command(flatten)]`-ed into each
//! variant to eliminate the triplication.

use clap::Args;
use std::path::PathBuf;

use super::cli_types::BackendArg;

/// Common arguments shared by WhisperSeq, WhisperSweep, and WhisperEpsSearch.
#[derive(Args, Clone, Debug)]
pub(crate) struct WhisperCommonArgs {
    /// Path to Whisper ONNX model
    pub model: PathBuf,

    /// First encoder block (0-indexed)
    #[arg(long, default_value_t = 0)]
    pub start_block: usize,

    /// End encoder block (exclusive). Defaults to all blocks.
    #[arg(long)]
    pub end_block: Option<usize>,

    /// Include encoder stem (mel -> hidden) before the first block
    #[arg(long, default_value_t = false)]
    pub include_stem: bool,

    /// Include final encoder LayerNorm (ln_post) after the last block
    #[arg(long, default_value_t = false)]
    pub include_ln_post: bool,

    /// Batch size for synthetic input (hidden states or mel)
    #[arg(long, default_value_t = 1)]
    pub batch: usize,

    /// Sequence length for synthetic hidden-state input (ignored if --include-stem)
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Mel bins for synthetic mel input (only used with --include-stem)
    #[arg(long, default_value_t = 80)]
    pub n_mels: usize,

    /// Time dimension for synthetic mel input (only used with --include-stem)
    #[arg(long, default_value_t = 3000)]
    pub time: usize,

    /// Compute backend (cpu, wgpu)
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    pub backend: BackendArg,

    /// Use wgpu GPU acceleration (deprecated, use --backend wgpu)
    #[arg(long, default_value_t = false, hide = true)]
    pub gpu: bool,

    /// Override: maximum bound width threshold before early termination
    #[arg(long)]
    pub max_bound_width: Option<f32>,

    /// Override: reset zonotope correlations at block boundaries (true/false)
    /// Normalizes input bounds and rescales output for deep transformers (28+ layers)
    #[arg(
        long,
        value_parser = clap::value_parser!(bool),
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub reset_zonotope_blocks: Option<bool>,

    /// Output as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,
}
