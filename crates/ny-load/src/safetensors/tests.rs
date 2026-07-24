// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use safetensors::tensor::{serialize, TensorView};
use std::io::Write;
use std::path::Path;
use tempfile::{NamedTempFile, TempPath};

fn temp_path_ref(path: &TempPath) -> &Path {
    <TempPath as AsRef<Path>>::as_ref(path)
}

fn create_test_safetensors() -> TempPath {
    let data1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data2: Vec<f32> = vec![0.1, 0.2, 0.3];
    let bytes1: Vec<u8> = data1.iter().flat_map(|f| f.to_le_bytes()).collect();
    let bytes2: Vec<u8> = data2.iter().flat_map(|f| f.to_le_bytes()).collect();
    let view1 = TensorView::new(safetensors::Dtype::F32, vec![2, 3], &bytes1).unwrap();
    let view2 = TensorView::new(safetensors::Dtype::F32, vec![3], &bytes2).unwrap();
    let tensors = vec![("layer1.weight", &view1), ("layer1.bias", &view2)];
    let serialized = serialize(tensors, None).unwrap();
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&serialized).unwrap();
    file.flush().unwrap();
    file.into_temp_path()
}

#[ntest::timeout(5000)]
#[test]
fn test_load_safetensors() {
    let file = create_test_safetensors();
    let weights = load_safetensors(temp_path_ref(&file)).unwrap();
    assert_eq!(weights.len(), 2);
    let weight = weights.get("layer1.weight").unwrap();
    assert_eq!(weight.shape(), &[2, 3]);
    assert!((weight[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((weight[[1, 2]] - 6.0).abs() < 1e-6);
    let bias = weights.get("layer1.bias").unwrap();
    assert_eq!(bias.shape(), &[3]);
    assert!((bias[[0]] - 0.1).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_safetensors_info() {
    let file = create_test_safetensors();
    let info = safetensors_info(temp_path_ref(&file)).unwrap();
    assert_eq!(info.tensor_count, 2);
    assert_eq!(info.param_count, 9);
}

#[ntest::timeout(5000)]
#[test]
fn test_half_to_f32_basic() {
    assert_eq!(half_to_f32(0x0000), 0.0);
    assert_eq!(half_to_f32(0x8000), -0.0);
    assert!((half_to_f32(0x3C00) - 1.0).abs() < 1e-6);
    assert!((half_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
    assert!(half_to_f32(0x7C00).is_infinite());
    assert!(half_to_f32(0x7E00).is_nan());
}

#[ntest::timeout(5000)]
#[test]
fn test_bf16_to_f32_basic() {
    assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
    assert!((bf16_to_f32(0xBF80) - (-1.0)).abs() < 1e-6);
    assert_eq!(bf16_to_f32(0x0000), 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_file_not_found() {
    let result = load_safetensors("/nonexistent/path/model.safetensors");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("File not found"));
}

#[ntest::timeout(5000)]
#[test]
fn test_safetensors_info_file_not_found() {
    let result = safetensors_info("/nonexistent/path/model.safetensors");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("File not found"));
}

#[ntest::timeout(5000)]
#[test]
fn test_half_to_f32_subnormal() {
    let smallest = half_to_f32(0x0001);
    assert!(smallest > 0.0);
    assert!((smallest - 5.960_464_5e-8).abs() < 1e-10);
    let largest = half_to_f32(0x03FF);
    assert!(largest > 0.0 && largest < 6.1e-5);
    let neg = half_to_f32(0x8001);
    assert!((neg - (-5.960_464_5e-8)).abs() < 1e-10);
}

#[ntest::timeout(5000)]
#[test]
fn test_half_to_f32_negative_infinity() {
    let neg_inf = half_to_f32(0xFC00);
    assert!(neg_inf.is_infinite() && neg_inf < 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_half_to_f32_various_normals() {
    assert!((half_to_f32(0x4000) - 2.0).abs() < 1e-6);
    assert!((half_to_f32(0x3800) - 0.5).abs() < 1e-6);
    assert!((half_to_f32(0xC000) - (-2.0)).abs() < 1e-6);
    assert!((half_to_f32(0x3400) - 0.25).abs() < 1e-6);
    let max_finite = half_to_f32(0x7BFF);
    assert!((max_finite - 65504.0).abs() < 1e-6 && max_finite.is_finite());
    let min_normal = half_to_f32(0x0400);
    assert!((min_normal - 6.103_515_6e-5).abs() < 1e-8);
}

#[ntest::timeout(5000)]
#[test]
fn test_half_to_f32_nan_propagates() {
    assert!(half_to_f32(0x7E00).is_nan());
    assert!(half_to_f32(0x7C01).is_nan());
    assert!(half_to_f32(0x7FFF).is_nan());
    assert!(half_to_f32(0xFE00).is_nan());
}

#[ntest::timeout(5000)]
#[test]
fn test_bf16_to_f32_infinity() {
    let pos = bf16_to_f32(0x7F80);
    assert!(pos.is_infinite() && pos > 0.0);
    let neg = bf16_to_f32(0xFF80);
    assert!(neg.is_infinite() && neg < 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_bf16_to_f32_nan() {
    assert!(bf16_to_f32(0x7FC0).is_nan());
    assert!(bf16_to_f32(0x7F81).is_nan());
    assert!(bf16_to_f32(0xFFC0).is_nan());
}

#[ntest::timeout(5000)]
#[test]
fn test_bf16_to_f32_various_values() {
    assert!((bf16_to_f32(0x4000) - 2.0).abs() < 1e-5);
    assert!((bf16_to_f32(0x3F00) - 0.5).abs() < 1e-5);
    assert!((bf16_to_f32(0xC000) - (-2.0)).abs() < 1e-5);
    let neg_zero = bf16_to_f32(0x8000);
    assert_eq!(neg_zero, 0.0);
    assert!(neg_zero.is_sign_negative());
}

#[ntest::timeout(5000)]
#[test]
fn test_load_safetensors_invalid_file() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"not a valid safetensors file").unwrap();
    file.flush().unwrap();
    let path = file.into_temp_path();
    let err = load_safetensors(temp_path_ref(&path)).unwrap_err();
    assert!(err.to_string().contains("Failed to parse SafeTensors"));
}

#[ntest::timeout(5000)]
#[test]
fn test_safetensors_info_empty_file() {
    let file = NamedTempFile::new().unwrap().into_temp_path();
    let err = safetensors_info(temp_path_ref(&file)).unwrap_err();
    assert!(matches!(&err, NyError::ModelLoad(_)));
}

fn create_dtype_file(dtype: safetensors::Dtype, data: &[u8], shape: Vec<usize>) -> TempPath {
    let view = TensorView::new(dtype, shape, data).unwrap();
    let tensors = vec![("tensor", &view)];
    let serialized = serialize(tensors, None).unwrap();
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&serialized).unwrap();
    file.flush().unwrap();
    file.into_temp_path()
}

#[ntest::timeout(5000)]
#[test]
fn test_load_f16_conversion() {
    let data: Vec<u8> = [0x3C00u16, 0x4000u16, 0x4200u16]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let file = create_dtype_file(safetensors::Dtype::F16, &data, vec![3]);
    let weights = load_safetensors(temp_path_ref(&file)).unwrap();
    let t = weights.get("tensor").unwrap();
    assert!((t[[0]] - 1.0).abs() < 1e-3);
    assert!((t[[1]] - 2.0).abs() < 1e-3);
    assert!((t[[2]] - 3.0).abs() < 1e-3);
}

#[ntest::timeout(5000)]
#[test]
fn test_load_bf16_conversion() {
    let data: Vec<u8> = [0x3F80u16, 0xBF80u16, 0x0000u16]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let file = create_dtype_file(safetensors::Dtype::BF16, &data, vec![3]);
    let weights = load_safetensors(temp_path_ref(&file)).unwrap();
    let t = weights.get("tensor").unwrap();
    assert!((t[[0]] - 1.0).abs() < 1e-5);
    assert!((t[[1]] - (-1.0)).abs() < 1e-5);
    assert!((t[[2]] - 0.0).abs() < 1e-5);
}

#[ntest::timeout(5000)]
#[test]
fn test_load_i32_conversion() {
    let data: Vec<u8> = [1i32, -2i32, 1000i32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let file = create_dtype_file(safetensors::Dtype::I32, &data, vec![3]);
    let weights = load_safetensors(temp_path_ref(&file)).unwrap();
    let t = weights.get("tensor").unwrap();
    assert!((t[[0]] - 1.0).abs() < 1e-5);
    assert!((t[[1]] - (-2.0)).abs() < 1e-5);
    assert!((t[[2]] - 1000.0).abs() < 1e-5);
}

#[ntest::timeout(5000)]
#[test]
fn test_load_i64_conversion() {
    let data: Vec<u8> = [100i64, -50i64, 0i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let file = create_dtype_file(safetensors::Dtype::I64, &data, vec![3]);
    let weights = load_safetensors(temp_path_ref(&file)).unwrap();
    let t = weights.get("tensor").unwrap();
    assert!((t[[0]] - 100.0).abs() < 1e-5);
    assert!((t[[1]] - (-50.0)).abs() < 1e-5);
    assert!((t[[2]] - 0.0).abs() < 1e-5);
}

#[ntest::timeout(5000)]
#[test]
fn test_load_f64_conversion() {
    let data: Vec<u8> = [1.5f64, -2.5f64, 3.125f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let file = create_dtype_file(safetensors::Dtype::F64, &data, vec![3]);
    let weights = load_safetensors(temp_path_ref(&file)).unwrap();
    let t = weights.get("tensor").unwrap();
    assert!((t[[0]] - 1.5).abs() < 1e-5);
    assert!((t[[1]] - (-2.5)).abs() < 1e-5);
    assert!((t[[2]] - 3.125).abs() < 1e-5);
}

// ==========================================================================
// Overflow regression tests (#3280)
//
// Verify that checked_shape_product in safetensors_info rejects shapes
// whose element count overflows usize. Without this guard, integer overflow
// in the shape product could produce a small (wrapped) param_count and
// bypass downstream allocation checks.
// ==========================================================================

/// Regression test (#3280): safetensors_info rejects overflowing shape product.
///
/// Crafts raw safetensors bytes where the JSON header declares a shape of
/// [4294967296, 4294967296] (2^32 × 2^32 = 2^64, overflows usize on 64-bit).
/// The `checked_shape_product` guard inside `safetensors_info` must return
/// Err(ModelLoad) instead of wrapping to a small number.
#[ntest::timeout(5000)]
#[test]
fn test_safetensors_info_shape_overflow_rejected_3280() {
    // Craft raw safetensors bytes: 8-byte LE header_size + JSON header + no data.
    // The JSON header declares a tensor with shape [4294967296, 4294967296].
    // data_offsets [0, 0] means no actual tensor data.
    let header_json = r#"{"overflow_tensor":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut raw = Vec::new();
    raw.extend_from_slice(&header_len.to_le_bytes());
    raw.extend_from_slice(header_bytes);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&raw).unwrap();
    file.flush().unwrap();
    let path = file.into_temp_path();
    let result = safetensors_info(temp_path_ref(&path));
    assert!(
        result.is_err(),
        "#3280: safetensors_info must reject overflowing shape"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("overflow"),
        "#3280: error should mention overflow, got: {err_msg}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_safetensors_info_param_count() {
    let data1: Vec<u8> = vec![0u8; 6 * 4];
    let data2: Vec<u8> = vec![0u8; 4 * 4];
    let view1 = TensorView::new(safetensors::Dtype::F32, vec![2, 3], &data1).unwrap();
    let view2 = TensorView::new(safetensors::Dtype::F32, vec![4], &data2).unwrap();
    let tensors = vec![("tensor1", &view1), ("tensor2", &view2)];
    let serialized = serialize(tensors, None).unwrap();
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&serialized).unwrap();
    file.flush().unwrap();
    let path = file.into_temp_path();
    let info = safetensors_info(temp_path_ref(&path)).unwrap();
    assert_eq!(info.tensor_count, 2);
    assert_eq!(info.param_count, 10);
}
