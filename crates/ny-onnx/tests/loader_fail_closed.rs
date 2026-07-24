// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "gguf")]
use ny_onnx::gguf::load_gguf;
#[cfg(feature = "internal-test-utils")]
use ny_onnx::native::test_support::directory_contains_extension_in_entries;
use ny_onnx::native::{load_weights, NativeModel};
use safetensors::tensor::TensorView;
use safetensors::{serialize, Dtype};
use std::collections::BTreeMap;
use std::fmt::Display;
#[cfg(feature = "internal-test-utils")]
use std::io;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use tempfile::tempdir;
#[cfg(feature = "gguf")]
use tempfile::NamedTempFile;

fn expect_error_message<T, E: Display>(result: Result<T, E>, context: &str) -> String {
    match result {
        Ok(_) => panic!("{context}"),
        Err(err) => err.to_string(),
    }
}

fn write_config_json(dir: &std::path::Path, contents: &str) -> std::path::PathBuf {
    let path = dir.join("config.json");
    std::fs::write(&path, contents).expect("Failed to write config.json");
    path
}

fn write_minimal_whisper_safetensors(
    dir: &std::path::Path,
    hidden_dim: usize,
    n_mels: usize,
) -> std::path::PathBuf {
    let conv1_len = hidden_dim * n_mels * 3;
    let conv2_len = hidden_dim * hidden_dim * 3;
    let conv1: Vec<f32> = (0..conv1_len).map(|i| i as f32 * 0.01).collect();
    let conv2: Vec<f32> = (0..conv2_len).map(|i| i as f32 * 0.01).collect();

    let conv1_view = TensorView::new(
        Dtype::F32,
        vec![hidden_dim, n_mels, 3],
        bytemuck::cast_slice(&conv1),
    )
    .expect("Failed to build conv1 TensorView");
    let conv2_view = TensorView::new(
        Dtype::F32,
        vec![hidden_dim, hidden_dim, 3],
        bytemuck::cast_slice(&conv2),
    )
    .expect("Failed to build conv2 TensorView");

    let mut tensors = BTreeMap::new();
    tensors.insert("conv1.weight".to_string(), conv1_view);
    tensors.insert("conv2.weight".to_string(), conv2_view);

    let data = serialize(tensors, None).expect("Failed to serialize safetensors");
    let path = dir.join("model.safetensors");
    std::fs::write(&path, data).expect("Failed to write safetensors file");
    path
}

#[ntest::timeout(10000)]
#[test]
fn test_native_model_load_rejects_invalid_parent_config_json() {
    let dir = tempdir().expect("Failed to create tempdir");
    write_config_json(dir.path(), "{ invalid json");
    let model_path = write_minimal_whisper_safetensors(dir.path(), 2, 1);

    let msg = expect_error_message(
        NativeModel::load(&model_path),
        "invalid parent Hugging Face config must fail",
    );
    assert!(
        msg.contains("Failed to parse config.json"),
        "expected config parse failure, got: {msg}"
    );
}

#[cfg(feature = "internal-test-utils")]
#[ntest::timeout(10000)]
#[test]
fn test_directory_contains_extension_propagates_entry_error() {
    let dir = tempdir().expect("Failed to create tempdir");
    let entries = vec![
        Err(io::Error::other("synthetic dir entry failure")),
        Ok(dir.path().join("weights.safetensors")),
    ];

    let msg = expect_error_message(
        directory_contains_extension_in_entries(dir.path(), entries, "safetensors"),
        "directory entry errors during safetensors detection must fail",
    );
    assert!(
        msg.contains("Failed to read directory entry"),
        "expected directory entry read failure, got: {msg}"
    );
}

#[cfg(unix)]
#[ntest::timeout(10000)]
#[test]
fn test_load_weights_rejects_unreadable_directory_during_safetensors_detection() {
    let dir = tempdir().expect("Failed to create tempdir");
    write_config_json(
        dir.path(),
        r#"{
  "architectures": ["WhisperForConditionalGeneration"],
  "model_type": "whisper",
  "d_model": 2,
  "encoder_layers": 0,
  "encoder_attention_heads": 1,
  "num_mel_bins": 1
}"#,
    );
    write_minimal_whisper_safetensors(dir.path(), 2, 1);

    let original_permissions = std::fs::metadata(dir.path())
        .expect("Failed to stat tempdir")
        .permissions();
    let mut execute_only = original_permissions.clone();
    execute_only.set_mode(0o111);
    std::fs::set_permissions(dir.path(), execute_only).expect("Failed to make tempdir unreadable");

    let result = load_weights(dir.path());

    std::fs::set_permissions(dir.path(), original_permissions)
        .expect("Failed to restore tempdir permissions");

    let msg = expect_error_message(
        result,
        "directory read failures during safetensors detection must fail",
    );
    assert!(
        msg.contains("Failed to read directory"),
        "expected directory read failure, got: {msg}"
    );
}

#[cfg(feature = "gguf")]
fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

#[cfg(feature = "gguf")]
fn push_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}

#[cfg(feature = "gguf")]
fn push_string(v: &mut Vec<u8>, s: &str) {
    push_u64(v, s.len() as u64);
    v.extend_from_slice(s.as_bytes());
}

#[cfg(feature = "gguf")]
#[ntest::timeout(10000)]
#[test]
fn test_load_gguf_fails_on_corrupted_supported_tensor_payload() {
    use gguf::GGMLType;

    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(b"GGUF");
    push_u32(&mut buf, 3);
    push_u64(&mut buf, 1);
    push_u64(&mut buf, 1);

    push_string(&mut buf, "general.alignment");
    push_u32(&mut buf, 4);
    push_u32(&mut buf, 32);

    push_string(&mut buf, "test.weight");
    push_u32(&mut buf, 1);
    push_u64(&mut buf, 4);
    push_u32(&mut buf, GGMLType::F32 as u32);
    push_u64(&mut buf, 0);

    let data_start = (buf.len() + 31) & !31;
    buf.resize(data_start, 0);

    for f in [1.0f32, 2.0] {
        buf.extend_from_slice(&f.to_le_bytes());
    }

    let file = NamedTempFile::new().expect("Failed to create temp file");
    std::fs::write(file.path(), &buf).expect("Failed to write GGUF file");

    let msg = expect_error_message(
        load_gguf(file.path()),
        "corrupted supported GGUF tensor payload must fail",
    );
    assert!(
        msg.contains("Failed to load GGUF tensor 'test.weight'"),
        "expected tensor load failure, got: {msg}"
    );
    assert!(
        msg.contains("out of bounds"),
        "expected out-of-bounds detail, got: {msg}"
    );
}
