// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

pub(super) mod block_index;
mod load;
pub(super) mod scope;
pub(super) mod structure;

/// Whisper loader entry point for parsing encoder structure and block metadata.
pub use load::load_whisper;
