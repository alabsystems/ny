// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::WeightStore;
use ny_core::{NyError, Result};
#[cfg(feature = "pytorch")]
use serde::Deserialize;
#[cfg(feature = "pytorch")]
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Load weights from a file or directory based on extension/type.
///
/// Supports:
/// - PyTorch: .pt, .pth, .bin
/// - SafeTensors: .safetensors (single file or sharded directory)
/// - GGUF: .gguf (llama.cpp format)
/// - CoreML: .mlmodel, .mlpackage
/// - HuggingFace model directories (config.json + *.safetensors shards)
///
/// # Example
///
/// ```rust,no_run
/// use ny_onnx::native::load_weights;
///
/// // Load from single file
/// let weights = load_weights("model.safetensors").unwrap();
///
/// // Load from sharded directory
/// let weights = load_weights("path/to/model/").unwrap();
/// ```
pub fn load_weights<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();

    // Handle directories (mlpackage, HuggingFace model directories)
    if path.is_dir() {
        return load_weights_from_directory(path);
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        // PyTorch formats
        "pt" | "pth" | "bin" => {
            #[cfg(feature = "pytorch")]
            {
                crate::pytorch::load_pytorch(path)
            }
            #[cfg(not(feature = "pytorch"))]
            {
                Err(NyError::ModelLoad(
                    "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
                ))
            }
        }

        // SafeTensors format (used by Hugging Face)
        "safetensors" => crate::safetensors::load_safetensors(path),

        // GGUF format (llama.cpp)
        "gguf" => {
            #[cfg(feature = "gguf")]
            {
                crate::gguf::load_gguf(path)
            }
            #[cfg(not(feature = "gguf"))]
            {
                Err(NyError::ModelLoad(
                    "GGUF support not enabled. Rebuild with --features gguf".to_string(),
                ))
            }
        }

        // CoreML formats
        "mlmodel" | "mlpackage" => {
            #[cfg(feature = "coreml")]
            {
                crate::coreml::load_coreml(path)
            }
            #[cfg(not(feature = "coreml"))]
            {
                Err(NyError::ModelLoad(
                    "CoreML support not enabled. Rebuild with --features coreml".to_string(),
                ))
            }
        }

        _ => Err(NyError::ModelLoad(format!(
            "Unknown file extension: {}. Supported: .pt, .pth, .bin, .safetensors, .gguf, .mlmodel, .mlpackage",
            ext
        ))),
    }
}

fn directory_read_error(dir: &Path, error: std::io::Error) -> NyError {
    NyError::ModelLoad(format!(
        "Failed to read directory {}: {}",
        dir.display(),
        error
    ))
}

fn directory_entry_error(dir: &Path, error: std::io::Error) -> NyError {
    NyError::ModelLoad(format!(
        "Failed to read directory entry in {}: {}",
        dir.display(),
        error
    ))
}

pub(super) fn directory_contains_extension_in_entries<I, P>(
    dir: &Path,
    entries: I,
    extension: &str,
) -> Result<bool>
where
    I: IntoIterator<Item = std::io::Result<P>>,
    P: AsRef<Path>,
{
    for entry in entries {
        let path = entry.map_err(|error| directory_entry_error(dir, error))?;
        if path.as_ref().extension().and_then(|ext| ext.to_str()) == Some(extension) {
            return Ok(true);
        }
    }

    Ok(false)
}

fn directory_contains_extension(dir: &Path, extension: &str) -> Result<bool> {
    let entries = std::fs::read_dir(dir).map_err(|error| directory_read_error(dir, error))?;
    directory_contains_extension_in_entries(
        dir,
        entries.map(|entry| entry.map(|entry| entry.path())),
        extension,
    )
}

fn read_directory_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir).map_err(|error| directory_read_error(dir, error))?;
    entries
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| directory_entry_error(dir, error))
        })
        .collect()
}

/// Load weights from a directory (HuggingFace models, mlpackage, sharded safetensors).
fn load_weights_from_directory(dir: &Path) -> Result<WeightStore> {
    info!("Loading weights from directory: {}", dir.display());

    // Check for HuggingFace model directory (has config.json + *.safetensors)
    let config_json = dir.join("config.json");
    let has_safetensors = if config_json.exists() {
        directory_contains_extension(dir, "safetensors")?
    } else {
        false
    };

    if config_json.exists() && has_safetensors {
        info!("Detected HuggingFace model directory");
        return load_sharded_safetensors(dir);
    }

    // Check for .mlpackage (CoreML)
    let model_mlmodel = dir.join("Data/com.apple.CoreML/model.mlmodel");
    if model_mlmodel.exists() || dir.extension().and_then(|s| s.to_str()) == Some("mlpackage") {
        #[cfg(feature = "coreml")]
        {
            return crate::coreml::load_coreml(dir);
        }
        #[cfg(not(feature = "coreml"))]
        {
            return Err(NyError::ModelLoad(
                "CoreML support not enabled. Rebuild with --features coreml".to_string(),
            ));
        }
    }

    // Check for PyTorch checkpoint directory
    let pytorch_index = dir.join("pytorch_model.bin.index.json");
    if pytorch_index.exists() {
        #[cfg(feature = "pytorch")]
        {
            return load_sharded_pytorch(dir, &pytorch_index);
        }
        #[cfg(not(feature = "pytorch"))]
        {
            return Err(NyError::ModelLoad(
                "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
            ));
        }
    }

    let pytorch_shards = find_pytorch_shard_files(dir)?;
    if !pytorch_shards.is_empty() {
        #[cfg(feature = "pytorch")]
        {
            return load_pytorch_shards(&pytorch_shards);
        }
        #[cfg(not(feature = "pytorch"))]
        {
            return Err(NyError::ModelLoad(
                "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
            ));
        }
    }

    let pytorch_bin = dir.join("pytorch_model.bin");
    let model_pt = dir.join("model.pt");
    if pytorch_bin.exists() {
        #[cfg(feature = "pytorch")]
        {
            return crate::pytorch::load_pytorch(&pytorch_bin);
        }
        #[cfg(not(feature = "pytorch"))]
        {
            return Err(NyError::ModelLoad(
                "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
            ));
        }
    }
    if model_pt.exists() {
        #[cfg(feature = "pytorch")]
        {
            return crate::pytorch::load_pytorch(&model_pt);
        }
        #[cfg(not(feature = "pytorch"))]
        {
            return Err(NyError::ModelLoad(
                "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
            ));
        }
    }

    Err(NyError::ModelLoad(format!(
        "Could not determine model format for directory: {}. \
         Expected: HuggingFace model (config.json + *.safetensors), .mlpackage, or PyTorch checkpoint (pytorch_model.bin, model.pt, or sharded pytorch_model-*.bin + index)",
        dir.display()
    )))
}

#[cfg(feature = "pytorch")]
#[derive(Debug, Deserialize)]
struct PytorchBinIndex {
    #[serde(default)]
    weight_map: HashMap<String, String>,
}

#[cfg(feature = "pytorch")]
fn load_sharded_pytorch(dir: &Path, index_path: &Path) -> Result<WeightStore> {
    use std::collections::HashSet;

    let index_data = std::fs::read_to_string(index_path).map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to read PyTorch shard index {}: {}",
            index_path.display(),
            e
        ))
    })?;

    let index: PytorchBinIndex = serde_json::from_str(&index_data).map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to parse PyTorch shard index {}: {}",
            index_path.display(),
            e
        ))
    })?;

    if index.weight_map.is_empty() {
        return Err(NyError::ModelLoad(format!(
            "PyTorch shard index {} has empty weight_map",
            index_path.display()
        )));
    }

    for shard_name in index.weight_map.values() {
        if shard_name.contains('\\') && !cfg!(windows) {
            return Err(NyError::ModelLoad(format!(
                "PyTorch shard index contains invalid path separator: {}",
                shard_name
            )));
        }
        let shard_path = Path::new(shard_name);
        let has_parent = shard_path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir));
        let has_prefix = shard_path
            .components()
            .any(|component| matches!(component, std::path::Component::Prefix(_)));
        if shard_path.is_absolute() || has_parent || has_prefix {
            return Err(NyError::ModelLoad(format!(
                "PyTorch shard index contains unsafe path: {}",
                shard_name
            )));
        }
    }

    let shard_names: HashSet<String> = index.weight_map.into_values().collect();
    let mut shard_paths: Vec<PathBuf> = shard_names.into_iter().map(|n| dir.join(n)).collect();
    shard_paths.sort();

    for shard in &shard_paths {
        if !shard.exists() {
            return Err(NyError::ModelLoad(format!(
                "PyTorch shard referenced by index is missing: {}",
                shard.display()
            )));
        }
    }

    info!(
        "Loading {} PyTorch shards from {}",
        shard_paths.len(),
        dir.display()
    );
    load_pytorch_shards(&shard_paths)
}

fn find_pytorch_shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shard_paths: Vec<PathBuf> = read_directory_paths(dir)?
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| {
                    n.starts_with("pytorch_model-")
                        && Path::new(n)
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
                })
                .unwrap_or(false)
        })
        .collect();

    shard_paths.sort();
    Ok(shard_paths)
}

#[cfg(feature = "pytorch")]
fn load_pytorch_shards(shard_paths: &[PathBuf]) -> Result<WeightStore> {
    let mut combined = WeightStore::new();

    for shard_path in shard_paths {
        debug!("Loading PyTorch shard: {}", shard_path.display());
        let shard_weights = crate::pytorch::load_pytorch(shard_path)?;

        for (name, tensor) in shard_weights.iter() {
            if combined.contains_key(name) {
                return Err(NyError::ModelLoad(format!(
                    "Duplicate tensor '{}' found across PyTorch shards (at {})",
                    name,
                    shard_path.display()
                )));
            }
            combined.insert(name.to_string(), tensor.clone());
        }
    }

    info!(
        "Loaded {} tensors from {} PyTorch shard file(s)",
        combined.len(),
        shard_paths.len()
    );
    // Defense-in-depth: validate combined store even though each shard was
    // individually validated. Catches NaN from future post-merge processing. (#2791)
    combined.validate_no_nan()?;
    Ok(combined)
}

/// Load sharded SafeTensors files from a directory.
fn load_sharded_safetensors(dir: &Path) -> Result<WeightStore> {
    let mut combined = WeightStore::new();

    // Find all .safetensors files
    let mut safetensor_files: Vec<_> = read_directory_paths(dir)?
        .into_iter()
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();

    safetensor_files.sort();

    if safetensor_files.is_empty() {
        return Err(NyError::ModelLoad(
            "No .safetensors files found in directory".to_string(),
        ));
    }

    info!("Loading {} SafeTensors shards", safetensor_files.len());

    for shard_path in safetensor_files {
        debug!("Loading shard: {}", shard_path.display());
        let shard_weights = crate::safetensors::load_safetensors(&shard_path)?;

        for (name, tensor) in shard_weights.iter() {
            if combined.contains_key(name) {
                return Err(NyError::ModelLoad(format!(
                    "Duplicate tensor '{}' found across SafeTensors shards (at {})",
                    name,
                    shard_path.display()
                )));
            }
            combined.insert(name.to_string(), tensor.clone());
        }
    }

    info!("Loaded {} tensors from sharded SafeTensors", combined.len());
    // Defense-in-depth: validate combined store even though each shard was
    // individually validated. Catches NaN from future post-merge processing. (#2791)
    combined.validate_no_nan()?;
    Ok(combined)
}
