// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! File I/O for NNet format.

use ny_core::{NyError, Result};
use std::path::Path;
use tracing::info;

use super::parser::parse_nnet;
use super::NNetNetwork;

/// Load a neural network from NNet format.
///
/// # Arguments
///
/// * `path` - Path to the .nnet file
///
/// # Returns
///
/// A parsed `NNetNetwork` ready for verification.
pub fn load_nnet<P: AsRef<Path>>(path: P) -> Result<NNetNetwork> {
    let path = path.as_ref();
    info!("Loading NNet from: {}", path.display());

    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let content = std::fs::read_to_string(path)
        .map_err(|e| NyError::ModelLoad(format!("Failed to read file: {}", e)))?;

    parse_nnet(&content)
}
