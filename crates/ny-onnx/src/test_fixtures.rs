// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared test fixture infrastructure for all ny-onnx test modules.
//!
//! Provides model discovery, git-based restoration, and assertion helpers.
//! Previously duplicated across 5 test modules (tests/, nnet/tests/, diff/,
//! sensitivity/, profile/).

use serde::Deserialize;
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};
use thiserror::Error;

use crate::onnx_proto;

pub(crate) const TEST_MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models");
pub(crate) const TEST_MODELS_ENV: &str = "NY_TEST_MODELS_DIR";
pub(crate) const DEFAULT_FIXTURE_HINT: &str =
    "Generate fixtures with: python scripts/generate_test_models.py (or cargo run -p ny-onnx --bin gen_test_fixtures for minimal fixtures)";
pub(crate) const TRANSFORMER_TEST_MODEL_HINT: &str =
    "Generate with: python scripts/export_test_transformer.py";
pub(crate) const WHISPER_TEST_MODEL_HINT: &str =
    "Generate with: python scripts/export_whisper_encoder.py";
pub(crate) const AVOICE_TEST_MODEL_HINT: &str =
    "Set NY_TEST_MODELS_DIR to the directory containing the ONNX exports, or place speaker_encoder.onnx / talker_attention_layer0.onnx / kokoro_vocoder.onnx / kokoro_duration_predictor.onnx in tests/models/. Run the explicit AVoice lane with cargo run -p ny-onnx --bin ny_onnx_conformance -- avoice after staging its models. Optional adjacent <model>.contract.json sidecars are supported.";

const AVOICE_EXPORT_FILES: &[&str] = &[
    "speaker_encoder.onnx",
    "talker_attention_layer0.onnx",
    "kokoro_vocoder.onnx",
    "kokoro_duration_predictor.onnx",
];
const AVOICE_EXPORT_DIRS: &[&str] = &[
    "root/avoice/models/Qwen3-TTS-0.6B-Base/onnx",
    "root/avoice/models/Qwen3-TTS-1.7B/onnx",
    "root/avoice/models/kokoro-v1_0/onnx",
    "avoice/models/Qwen3-TTS-0.6B-Base/onnx",
    "avoice/models/Qwen3-TTS-1.7B/onnx",
    "avoice/models/kokoro-v1_0/onnx",
];

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub(crate) struct AvoiceFixtureContract {
    pub version: u32,
    pub model: String,
    pub activation_input: String,
    #[serde(default)]
    pub aux_inputs: Vec<String>,
    pub canonical_seq_len: Option<usize>,
    #[serde(default)]
    pub dynamic_axes: BTreeMap<String, BTreeMap<String, String>>,
    #[serde(default)]
    pub constraints: AvoiceFixtureConstraints,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub(crate) struct AvoiceFixtureConstraints {
    pub min_fixed_aux_t: Option<usize>,
    pub boundary_samples: Option<usize>,
    pub sample_rate_hz: Option<usize>,
    pub hidden_dim: Option<usize>,
    pub rope_dim: Option<usize>,
    pub rope_base: Option<f64>,
    pub mask_kind: Option<String>,
    pub duration_head: Option<String>,
    pub duration_bin_count: Option<usize>,
    pub min_duration_frames: Option<f32>,
    pub max_duration_frames: Option<f32>,
}

#[derive(Debug, Error)]
pub(crate) enum AvoiceContractLoadError {
    #[error("failed to check avoice contract sidecar at {path:?}: {source}")]
    Check {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read avoice contract sidecar at {path:?}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse avoice contract sidecar at {path:?}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

pub(crate) fn test_models_dir() -> PathBuf {
    match std::env::var(TEST_MODELS_ENV) {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir.trim()),
        _ => PathBuf::from(TEST_MODELS_DIR),
    }
}

pub(crate) fn test_model_path(name: &str) -> PathBuf {
    test_models_dir().join(name)
}

fn is_avoice_export(name: &str) -> bool {
    AVOICE_EXPORT_FILES.contains(&name)
}

fn avoice_export_search_paths() -> Vec<PathBuf> {
    let home = match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home),
        None => return Vec::new(),
    };
    AVOICE_EXPORT_DIRS
        .iter()
        .map(|relative| home.join(relative))
        .collect()
}

fn find_avoice_export(name: &str) -> Option<PathBuf> {
    if !is_avoice_export(name) {
        return None;
    }
    avoice_export_search_paths()
        .into_iter()
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

pub(crate) fn avoice_contract_path(model_path: &Path) -> PathBuf {
    let mut contract_path = model_path.to_path_buf();
    contract_path.set_extension("contract.json");
    contract_path
}

pub(crate) fn load_avoice_contract(
    model_path: &Path,
) -> Result<Option<AvoiceFixtureContract>, AvoiceContractLoadError> {
    let contract_path = avoice_contract_path(model_path);
    match contract_path.try_exists() {
        Ok(false) => return Ok(None),
        Ok(true) => {}
        Err(source) => {
            return Err(AvoiceContractLoadError::Check {
                path: contract_path,
                source,
            });
        }
    }

    let contract_bytes =
        std::fs::read(&contract_path).map_err(|source| AvoiceContractLoadError::Read {
            path: contract_path.clone(),
            source,
        })?;
    let contract =
        serde_json::from_slice::<AvoiceFixtureContract>(&contract_bytes).map_err(|source| {
            AvoiceContractLoadError::Parse {
                path: contract_path,
                source,
            }
        })?;
    Ok(Some(contract))
}

pub(crate) fn try_restore_test_model(path: &Path, name: &str) -> bool {
    if let Ok(metadata) = path.metadata() {
        if metadata.is_file() && metadata.len() > 0 {
            return true;
        }
    }

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let repo_root = match repo_root.canonicalize() {
        Ok(path) => path,
        Err(_) => return false,
    };
    if !repo_root.join(".git").exists() {
        return false;
    }

    let git_path = format!("tests/models/{}", name);
    let output = std::process::Command::new("git")
        .args(["show", &format!("HEAD:{}", git_path)])
        .current_dir(&repo_root)
        .output();

    let output = match output {
        Ok(output) if output.status.success() => output,
        _ => return false,
    };

    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return false;
        }
    }

    publish_test_model_atomically(path, &output.stdout)
}

fn publish_test_model_atomically(path: &Path, bytes: &[u8]) -> bool {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut staged = match tempfile::NamedTempFile::new_in(parent) {
        Ok(staged) => staged,
        Err(_) => return false,
    };
    if staged.write_all(bytes).is_err()
        || staged.flush().is_err()
        || staged.as_file().sync_all().is_err()
    {
        return false;
    }

    // Tests load fixtures concurrently and even separate cargo processes can
    // restore the same missing file. Publish a complete same-directory
    // snapshot atomically so readers never observe a protobuf prefix.
    if staged.persist(path).is_err() {
        // Another publisher may have won (or Windows may have refused a
        // replace). Count that as success only when the complete expected
        // fixture is now visible.
        return std::fs::read(path).is_ok_and(|published| published == bytes);
    }
    std::fs::read(path).is_ok_and(|published| published == bytes)
}

pub(crate) fn optional_test_model(name: &str) -> Option<PathBuf> {
    let path = test_model_path(name);
    if try_restore_test_model(&path, name) {
        return Some(path);
    }

    if std::env::var(TEST_MODELS_ENV)
        .ok()
        .is_none_or(|dir| dir.trim().is_empty())
    {
        return find_avoice_export(name);
    }

    None
}

fn missing_test_model_details(name: &str, hint: &str) -> String {
    let restore_hint =
        "If this fixture is tracked, tests attempt to restore it from git; otherwise generate it.";
    let env_hint = match std::env::var(TEST_MODELS_ENV) {
        Ok(dir) if !dir.trim().is_empty() => {
            format!("{} is set to {}.", TEST_MODELS_ENV, dir.trim())
        }
        _ => format!(
            "Set {} to override the fixtures directory.",
            TEST_MODELS_ENV
        ),
    };
    let avoice_hint = if is_avoice_export(name) {
        let searched = avoice_export_search_paths()
            .into_iter()
            .map(|dir| dir.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(" Auto-discovery also searched: {}.", searched)
    } else {
        String::new()
    };
    if hint.is_empty() {
        format!("{} {}{}", env_hint, restore_hint, avoice_hint)
    } else {
        format!("{} {} {}{}", hint, env_hint, restore_hint, avoice_hint)
    }
}

pub(crate) fn require_test_model_with_hint(name: &str, hint: &str) -> PathBuf {
    if let Some(path) = optional_test_model(name) {
        return path;
    }

    panic!(
        "Test model missing at {:?}. {}",
        test_model_path(name),
        missing_test_model_details(name, hint)
    );
}

pub(crate) fn require_test_model(name: &str) -> PathBuf {
    require_test_model_with_hint(name, DEFAULT_FIXTURE_HINT)
}

/// Hint matching what the hard-require call sites pass for this fixture, so
/// fixture diagnostics from bare guards stay consistent.
#[cfg(any(feature = "external-avoice", feature = "external-whisper"))]
pub(crate) fn default_hint_for(name: &str) -> &'static str {
    if is_avoice_export(name) {
        AVOICE_TEST_MODEL_HINT
    } else if name == "whisper_tiny_encoder.onnx" {
        WHISPER_TEST_MODEL_HINT
    } else {
        DEFAULT_FIXTURE_HINT
    }
}

/// Test-body guard for model-backed tests.
///
/// Every explicitly selected model-backed conformance test requires its
/// fixture. External suites are registered only by their explicit Cargo
/// feature, and an unavailable asset is always a hard failure.
#[cfg(any(feature = "external-avoice", feature = "external-whisper"))]
macro_rules! assert_test_model_available {
    ($name:expr) => {
        let name = $name;
        let _fixture_path = crate::test_fixtures::require_test_model_with_hint(
            name,
            crate::test_fixtures::default_hint_for(name),
        );
    };
}
#[cfg(any(feature = "external-avoice", feature = "external-whisper"))]
pub(crate) use assert_test_model_available;

pub(crate) fn specialize_kokoro_duration_predictor_for_lstm_unroll(
    proto: &mut onnx_proto::ModelProto,
    seq_len: i64,
) {
    concretize_symbolic_dim(proto, "T", seq_len);

    let Some(graph) = proto.graph.as_mut() else {
        return;
    };

    #[cfg(feature = "onnx-value-info")]
    {
        if !graph
            .value_info
            .iter()
            .any(|info| info.name == "/lstm/Transpose_output_0")
        {
            graph.value_info.push(f32_tensor_value_info(
                "/lstm/Transpose_output_0",
                &[seq_len, 1, 640],
            ));
        }
    }

    for node in &mut graph.node {
        if node.name == "/lstm/LSTM" {
            if let Some(initial_h) = node.input.get_mut(5) {
                initial_h.clear();
            }
            if let Some(initial_c) = node.input.get_mut(6) {
                initial_c.clear();
            }
        }
        if node.name == "/duration_proj/linear_layer/MatMul"
            && node.input.first().map(String::as_str) == Some("/lstm/Transpose_2_output_0")
        {
            node.input[0] = "/lstm/LSTM_output_0".to_string();
        }
    }

    graph.node.retain(|node| {
        !matches!(
            node.name.as_str(),
            "/lstm/Transpose_1" | "/lstm/Constant_3" | "/lstm/Reshape" | "/lstm/Transpose_2"
        )
    });
}

fn concretize_symbolic_dim(proto: &mut onnx_proto::ModelProto, symbol: &str, concrete_value: i64) {
    use onnx_proto::tensor_shape_proto::dimension::Value;

    let set_dim = |value_info: &mut onnx_proto::ValueInfoProto| {
        let Some(tensor_type) = value_info
            .r#type
            .as_mut()
            .and_then(|ty| ty.tensor_type.as_mut())
        else {
            return;
        };
        let Some(shape) = tensor_type.shape.as_mut() else {
            return;
        };
        for dim in &mut shape.dim {
            let is_target = matches!(
                dim.value.as_ref(),
                Some(Value::DimParam(param)) if param == symbol
            );
            if is_target {
                dim.value = Some(Value::DimValue(concrete_value));
            }
        }
    };

    let Some(graph) = proto.graph.as_mut() else {
        return;
    };
    for value_info in &mut graph.input {
        set_dim(value_info);
    }
    for value_info in &mut graph.output {
        set_dim(value_info);
    }
    #[cfg(feature = "onnx-value-info")]
    {
        for value_info in &mut graph.value_info {
            set_dim(value_info);
        }
    }
}

#[cfg(feature = "onnx-value-info")]
fn f32_tensor_value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
    use onnx_proto::tensor_shape_proto::{dimension::Value, Dimension};

    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type: 1,
                shape: Some(onnx_proto::TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&value| Dimension {
                            value: Some(Value::DimValue(value)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

#[cfg(test)]
mod tests;
