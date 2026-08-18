// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unavailable experimental sequential Whisper compatibility command.

use anyhow::Result;
use std::path::PathBuf;

use crate::BackendArg;

// Justification: Parameters preserve the historical clap compatibility surface.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_whisper_seq_command(
    _model: PathBuf,
    _start_block: usize,
    _end_block: Option<usize>,
    _include_stem: bool,
    _include_ln_post: bool,
    _batch: usize,
    _seq_len: usize,
    _n_mels: usize,
    _time: usize,
    _epsilon: f32,
    _backend: BackendArg,
    _gpu: bool,
    _mode: String,
    _max_bound_width: Option<f32>,
    _terminate_on_overflow: Option<bool>,
    _continue_after_overflow: Option<bool>,
    _overflow_clamp_value: Option<f32>,
    _reset_zonotope_blocks: Option<bool>,
    _json: bool,
) -> Result<()> {
    super::whisper_verification_unavailable()
}
