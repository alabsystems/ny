// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for tensor decoding helpers and `onnx_elem_type_to_dtype`.

use super::{
    extract_constant_tensor, onnx_elem_type_to_dtype, tensor_proto_to_array,
    tensor_proto_to_loaded_tensor, value_info_to_tensor_spec,
};
use crate::loader::numeric_cast::{
    f64_to_f32_checked, i32_to_f32_warned, i64_to_f32_checked, i64_to_f32_warned,
};
use crate::onnx_proto::{attribute_type, AttributeProto, NodeProto};
use crate::onnx_proto::{TensorProto, TensorTypeProto, TypeProto, ValueInfoProto};
use crate::DataType;
use ny_core::NyError;

fn make_tensor_with_raw_data(
    name: &str,
    dims: &[i64],
    data_type: i32,
    raw_data: Vec<u8>,
) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type,
        name: name.to_string(),
        raw_data,
        float_data: Vec::new(),
        ..Default::default()
    }
}

#[test]
fn tensor_proto_to_array_rejects_raw_data_length_mismatch() {
    let raw = 1.0f32.to_le_bytes().to_vec();
    let tensor = make_tensor_with_raw_data("bad_len", &[2], 1, raw);

    let err = tensor_proto_to_array(&tensor).unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(msg.contains("expected 2 elements"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

#[test]
fn onnx_elem_type_to_dtype_maps_supported_types() {
    let cases = [
        (1, DataType::Float32),
        (6, DataType::Int32),
        (7, DataType::Int64),
    ];
    for (elem_type, expected) in cases {
        let dtype = onnx_elem_type_to_dtype(elem_type).expect("supported type should parse");
        assert_eq!(dtype, expected, "elem_type={elem_type}");
    }
}

/// Round-trip: FLOAT16 raw_data is correctly decoded to f32.
/// Reference: ONNX TensorProto.DataType FLOAT16 = 10, 2 bytes per element.
#[test]
fn tensor_proto_to_array_decodes_float16_raw_data() {
    use half::f16;
    // Encode [1.0, -0.5, 0.0] as FLOAT16 little-endian
    let values = [1.0f32, -0.5, 0.0];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("fp16", &[3], 10, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode FLOAT16");
    assert_eq!(arr.len(), 3);
    for (i, &expected) in values.iter().enumerate() {
        let got = arr[[i]];
        assert!(
            (got - expected).abs() < 1e-3,
            "element {i}: expected {expected}, got {got}"
        );
    }
}

/// Round-trip: BFLOAT16 raw_data is correctly decoded to f32.
#[test]
fn tensor_proto_to_array_decodes_bfloat16_raw_data() {
    use half::bf16;
    let values = [1.0f32, -2.0, 0.5];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("bf16", &[3], 16, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode BFLOAT16");
    assert_eq!(arr.len(), 3);
    for (i, &expected) in values.iter().enumerate() {
        let got = arr[[i]];
        assert!(
            (got - expected).abs() < 1e-2,
            "element {i}: expected {expected}, got {got}"
        );
    }
}

/// UINT8 raw_data is correctly decoded to f32.
#[test]
fn tensor_proto_to_array_decodes_uint8_raw_data() {
    let raw = vec![0u8, 127, 255];
    let tensor = make_tensor_with_raw_data("u8", &[3], 2, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode UINT8");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[[0]], 0.0);
    assert_eq!(arr[[1]], 127.0);
    assert_eq!(arr[[2]], 255.0);
}

/// INT8 raw_data is correctly decoded to f32 (signed).
#[test]
fn tensor_proto_to_array_decodes_int8_raw_data() {
    // 0x80 = -128i8, 0xFF = -1i8, 0x7F = 127i8
    let raw = vec![0x80u8, 0xFF, 0x7F, 0x00];
    let tensor = make_tensor_with_raw_data("i8", &[4], 3, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode INT8");
    assert_eq!(arr.len(), 4);
    assert_eq!(arr[[0]], -128.0);
    assert_eq!(arr[[1]], -1.0);
    assert_eq!(arr[[2]], 127.0);
    assert_eq!(arr[[3]], 0.0);
}

#[test]
fn tensor_proto_to_loaded_tensor_decodes_uint16_raw_with_integer_provenance() {
    let values = [0_u16, 1, 32_768, u16::MAX];
    let raw = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor = make_tensor_with_raw_data("u16", &[4], 4, raw);
    let loaded = tensor_proto_to_loaded_tensor(&tensor).expect("should decode UINT16");

    assert_eq!(
        loaded.float_data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 1.0, 32_768.0, 65_535.0]
    );
    assert_eq!(
        loaded
            .integer_data
            .expect("UINT16 must retain exact integer data")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![0, 1, 32_768, 65_535]
    );
    assert_eq!(loaded.integer_range, Some((0, 65_535)));
}

#[test]
fn tensor_proto_to_loaded_tensor_decodes_int16_raw_with_integer_provenance() {
    let values = [i16::MIN, -1, 0, i16::MAX];
    let raw = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor = make_tensor_with_raw_data("i16", &[4], 5, raw);
    let loaded = tensor_proto_to_loaded_tensor(&tensor).expect("should decode INT16");

    assert_eq!(
        loaded.float_data.iter().copied().collect::<Vec<_>>(),
        vec![-32_768.0, -1.0, 0.0, 32_767.0]
    );
    assert_eq!(
        loaded
            .integer_data
            .expect("INT16 must retain exact integer data")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![-32_768, -1, 0, 32_767]
    );
    assert_eq!(loaded.integer_range, Some((-32_768, 32_767)));
}

#[test]
fn tensor_proto_to_loaded_tensor_validates_widened_int16_payloads() {
    let uint16 = TensorProto {
        dims: vec![2],
        data_type: 4,
        name: "u16_wide".to_string(),
        int32_data: vec![0, i32::from(u16::MAX)],
        ..Default::default()
    };
    let loaded = tensor_proto_to_loaded_tensor(&uint16).expect("widened UINT16 should decode");
    assert_eq!(loaded.integer_range, Some((0, 65_535)));

    let mut invalid = uint16;
    invalid.int32_data = vec![0, -1];
    let error = match tensor_proto_to_loaded_tensor(&invalid) {
        Err(error) => error,
        Ok(_) => panic!("negative widened UINT16 must be rejected"),
    };
    assert!(error.to_string().contains("UINT16 int32_data value -1"));

    let int16 = TensorProto {
        dims: vec![2],
        data_type: 5,
        name: "i16_wide".to_string(),
        int32_data: vec![i32::from(i16::MIN), i32::from(i16::MAX)],
        ..Default::default()
    };
    let loaded = tensor_proto_to_loaded_tensor(&int16).expect("widened INT16 should decode");
    assert_eq!(loaded.integer_range, Some((-32_768, 32_767)));

    let mut invalid = int16;
    invalid.int32_data = vec![i32::from(i16::MIN) - 1, 0];
    let error = match tensor_proto_to_loaded_tensor(&invalid) {
        Err(error) => error,
        Ok(_) => panic!("out-of-range widened INT16 must be rejected"),
    };
    assert!(error.to_string().contains("INT16 int32_data value -32769"));
}

#[test]
fn empty_integer_tensors_retain_typed_sidecar_and_range() {
    let cases = [
        (2, (0, 255)),
        (3, (-128, 127)),
        (4, (0, 65_535)),
        (5, (-32_768, 32_767)),
        (6, (i32::MIN as i64, i32::MAX as i64)),
        (7, (i64::MIN, i64::MAX)),
    ];
    for (data_type, expected_range) in cases {
        let tensor = TensorProto {
            dims: vec![0, 3],
            data_type,
            name: format!("empty_{data_type}"),
            ..Default::default()
        };
        let loaded = tensor_proto_to_loaded_tensor(&tensor)
            .unwrap_or_else(|error| panic!("empty dtype {data_type} should decode: {error}"));
        assert!(loaded.float_data.is_empty());
        assert!(
            loaded
                .integer_data
                .as_ref()
                .is_some_and(|values| values.is_empty()),
            "dtype {data_type} lost its empty integer sidecar"
        );
        assert_eq!(loaded.integer_range, Some(expected_range));
    }
}

/// Exactly representable DOUBLE raw_data can be preserved as f32 constants.
#[test]
fn tensor_proto_to_array_decodes_double_raw_data() {
    let values = [1.0f64, -0.5, 1.5];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("f64", &[3], 11, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode DOUBLE");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[[0]], 1.0);
    assert_eq!(arr[[1]], -0.5);
    assert_eq!(arr[[2]], 1.5);
}

#[test]
fn tensor_proto_to_array_rejects_inexact_double_raw_data() {
    let value = 16_777_217.0f64;
    let tensor =
        make_tensor_with_raw_data("rounded_double", &[1], 11, value.to_le_bytes().to_vec());
    let err = tensor_proto_to_array(&tensor).expect_err("rounded DOUBLE must fail closed");
    assert!(
        err.to_string().contains("cannot be represented exactly"),
        "{err}"
    );
    assert!(err.to_string().contains("rounded_double"), "{err}");
}

#[test]
fn tensor_proto_to_array_rejects_double_raw_data_out_of_range_2360() {
    let values = [f64::MAX];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("f64_too_large", &[1], 11, raw);
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(msg.contains("f64→f32 out of range"), "msg = {msg}");
            assert!(msg.contains("f64_too_large"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

/// FLOAT32 raw_data round-trips exactly.
#[test]
fn tensor_proto_to_array_decodes_float32_raw_data() {
    let values = [1.0f32, -0.5, 0.0, f32::INFINITY, f32::NEG_INFINITY];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("f32", &[5], 1, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode FLOAT");
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[[0]], 1.0);
    assert_eq!(arr[[1]], -0.5);
    assert_eq!(arr[[2]], 0.0);
    assert_eq!(arr[[3]], f32::INFINITY);
    assert_eq!(arr[[4]], f32::NEG_INFINITY);
}

/// Unknown data_type in raw_data returns an error instead of garbage.
#[test]
fn tensor_proto_to_array_rejects_unknown_data_type_in_raw_data() {
    let raw = vec![0u8; 4]; // 4 bytes of zeros
    let tensor = make_tensor_with_raw_data("unknown_type", &[1], 99, raw);
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(msg.contains("unsupported ONNX data_type 99"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

/// INT32 raw_data is correctly decoded to f32.
/// Pre-existing handler; test added during self-audit round 2/2 for coverage.
#[test]
fn tensor_proto_to_array_decodes_int32_raw_data() {
    let values: Vec<i32> = vec![0, -1, i32::MAX, i32::MIN, 42];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("i32", &[5], 6, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode INT32");
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[[0]], 0.0);
    assert_eq!(arr[[1]], -1.0);
    // i32::MAX (2147483647) loses precision in f32 — verify it's close
    assert!(
        (arr[[2]] - i32::MAX as f32).abs() < 1.0,
        "i32::MAX: got {}",
        arr[[2]]
    );
    assert!(
        (arr[[3]] - i32::MIN as f32).abs() < 1.0,
        "i32::MIN: got {}",
        arr[[3]]
    );
    assert_eq!(arr[[4]], 42.0);
}

/// INT64 raw_data is correctly decoded to f32 (with precision loss for large values).
/// Pre-existing handler; test added during self-audit round 2/2 for coverage.
#[test]
fn tensor_proto_to_array_decodes_int64_raw_data() {
    let values: Vec<i64> = vec![0, -1, 1000000, -1000000];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("i64", &[4], 7, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode INT64");
    assert_eq!(arr.len(), 4);
    assert_eq!(arr[[0]], 0.0);
    assert_eq!(arr[[1]], -1.0);
    assert_eq!(arr[[2]], 1_000_000.0);
    assert_eq!(arr[[3]], -1_000_000.0);
}

/// INT64 values outside f32 exact-integer range produce warnings (#2848).
///
/// i64::MAX (9223372036854775807) is commonly used as a sentinel in ONNX Slice
/// start/end. When cast to f32, it round-trips to 9223372036854775808.0 (2^63),
/// which is incorrect. The warned conversion detects this.
#[test]
fn tensor_proto_to_array_int64_precision_loss_sentinel() {
    let values: Vec<i64> = vec![i64::MAX, i64::MIN, 1 << 24, -(1 << 24)];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("i64_sentinel", &[4], 7, raw);
    // Should succeed (warning is logged, not an error)
    let arr = tensor_proto_to_array(&tensor).expect("should decode INT64 with precision loss");
    assert_eq!(arr.len(), 4);
    // i64::MAX → f32 produces 2^63 (precision loss, but no panic)
    assert!(arr[[0]].is_finite(), "i64::MAX should produce finite f32");
    assert!(arr[[1]].is_finite(), "i64::MIN should produce finite f32");
    // 2^24 is exactly representable
    assert_eq!(arr[[2]], (1i64 << 24) as f32);
    // -(2^24) is exactly representable
    assert_eq!(arr[[3]], -(1i64 << 24) as f32);
}

/// INT32 values outside f32 exact-integer range produce warnings (#2848).
#[test]
fn tensor_proto_to_array_int32_precision_loss_boundary() {
    // 2^24 + 1 = 16777217, the first integer not exactly representable as f32
    let exact_limit = 1i32 << 24; // 16_777_216 — exactly representable
    let just_over = exact_limit + 1; // 16_777_217 — loses precision
    let values: Vec<i32> = vec![exact_limit, just_over, i32::MAX];
    let mut raw = Vec::new();
    for &v in &values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("i32_boundary", &[3], 6, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode INT32 with precision loss");
    assert_eq!(arr.len(), 3);
    // 2^24 is exact
    assert_eq!(arr[[0]], exact_limit as f32);
    // 2^24 + 1 rounds (precision loss, warned)
    assert!(arr[[1]].is_finite());
    // i32::MAX is finite but imprecise
    assert!(arr[[2]].is_finite());
}

#[test]
fn tensor_proto_to_loaded_tensor_preserves_exact_int64_values_2360() {
    let values: Vec<i64> = vec![i64::MAX, 16_777_217, -16_777_217];
    let mut raw = Vec::new();
    for &value in &values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    let tensor = make_tensor_with_raw_data("i64_exact", &[3], 7, raw);

    let loaded = tensor_proto_to_loaded_tensor(&tensor).expect("should decode INT64");

    let integer_data = loaded
        .integer_data
        .expect("INT64 tensor should preserve integer payload");
    let stored: Vec<i64> = integer_data.iter().copied().collect();
    assert_eq!(stored, values);
    assert!(
        loaded.float_data[[1]].is_finite(),
        "backward-compatible f32 view should still be populated"
    );
}

#[test]
fn extract_constant_tensor_preserves_value_ints_payload_2360() {
    let node = NodeProto {
        op_type: "Constant".to_string(),
        attribute: vec![AttributeProto {
            name: "value_ints".to_string(),
            r#type: attribute_type::INTS,
            ints: vec![i64::MAX, 16_777_217],
            ..Default::default()
        }],
        ..Default::default()
    };

    let loaded = extract_constant_tensor(&node)
        .expect("constant parse should succeed")
        .expect("constant payload should exist");

    let integer_data = loaded
        .integer_data
        .expect("value_ints should preserve integer payload");
    let stored: Vec<i64> = integer_data.iter().copied().collect();
    assert_eq!(stored, vec![i64::MAX, 16_777_217]);
}

/// FLOAT16 NaN propagates correctly through f16→f32 conversion.
/// Soundness-critical: NaN must not become a finite value during type conversion.
#[test]
fn tensor_proto_to_array_float16_nan_propagation() {
    use half::f16;
    let nan_f16 = f16::NAN;
    let inf_f16 = f16::INFINITY;
    let neg_inf_f16 = f16::NEG_INFINITY;
    let mut raw = Vec::new();
    raw.extend_from_slice(&nan_f16.to_le_bytes());
    raw.extend_from_slice(&inf_f16.to_le_bytes());
    raw.extend_from_slice(&neg_inf_f16.to_le_bytes());
    let tensor = make_tensor_with_raw_data("fp16_special", &[3], 10, raw);
    let arr = tensor_proto_to_array(&tensor).expect("should decode FLOAT16 special values");
    assert_eq!(arr.len(), 3);
    assert!(arr[[0]].is_nan(), "NaN must propagate through f16→f32");
    assert_eq!(arr[[1]], f32::INFINITY);
    assert_eq!(arr[[2]], f32::NEG_INFINITY);
}

/// Non-raw INT64 payload (int64_data, field tag 7) decoded from hand-encoded
/// protobuf bytes. Byte-level encoding (not a struct literal round-trip) so a
/// wrong tag number in the prost schema cannot pass.
#[test]
fn tensor_proto_decodes_non_raw_int64_data_from_bytes() {
    use prost::Message;
    let bytes: &[u8] = &[
        0x0A, 0x01, 0x02, // dims (tag 1, packed): [2]
        0x10, 0x07, // data_type (tag 2): 7 = INT64
        0x3A, 0x02, 0x04, 0x08, // int64_data (tag 7, packed): [4, 8]
        0x42, 0x05, b's', b'h', b'a', b'p', b'e', // name (tag 8): "shape"
    ];
    let tensor = TensorProto::decode(bytes).expect("valid TensorProto bytes");
    assert_eq!(tensor.int64_data, vec![4, 8]);

    let loaded = tensor_proto_to_loaded_tensor(&tensor).expect("should decode int64_data");
    let integer_data = loaded.integer_data.expect("integer payload preserved");
    let stored: Vec<i64> = integer_data.iter().copied().collect();
    assert_eq!(stored, vec![4, 8]);
    assert_eq!(loaded.float_data[[0]], 4.0);
    assert_eq!(loaded.float_data[[1]], 8.0);
}

/// Non-raw UINT8 payload (int32_data, field tag 5) decoded from hand-encoded
/// protobuf bytes — the encoding quantized exporters use for zero_points.
#[test]
fn tensor_proto_decodes_non_raw_uint8_int32_data_from_bytes() {
    use prost::Message;
    let bytes: &[u8] = &[
        0x0A, 0x01, 0x01, // dims (tag 1, packed): [1]
        0x10, 0x02, // data_type (tag 2): 2 = UINT8
        0x2A, 0x02, 0x83, 0x01, // int32_data (tag 5, packed): [131]
        0x42, 0x02, b'z', b'p', // name (tag 8): "zp"
    ];
    let tensor = TensorProto::decode(bytes).expect("valid TensorProto bytes");
    assert_eq!(tensor.int32_data, vec![131]);

    let loaded = tensor_proto_to_loaded_tensor(&tensor).expect("should decode int32_data");
    assert_eq!(loaded.float_data[[0]], 131.0);
    let integer_data = loaded.integer_data.expect("integer payload preserved");
    assert_eq!(integer_data[[0]], 131);
    assert_eq!(loaded.integer_range, Some((0, u8::MAX as i64)));
}

/// Non-raw INT8 payload keeps its sign and reports the i8 integer range.
#[test]
fn tensor_proto_to_array_decodes_int8_int32_data() {
    let tensor = TensorProto {
        dims: vec![3],
        data_type: 3,
        name: "i8_typed".to_string(),
        int32_data: vec![-128, -1, 127],
        ..Default::default()
    };
    let loaded = tensor_proto_to_loaded_tensor(&tensor).expect("should decode INT8 int32_data");
    assert_eq!(loaded.float_data[[0]], -128.0);
    assert_eq!(loaded.float_data[[1]], -1.0);
    assert_eq!(loaded.float_data[[2]], 127.0);
    assert_eq!(loaded.integer_range, Some((i8::MIN as i64, i8::MAX as i64)));
}

/// UINT8 int32_data values outside [0, 255] are rejected, not truncated.
#[test]
fn tensor_proto_to_array_rejects_out_of_range_uint8_int32_data() {
    let tensor = TensorProto {
        dims: vec![1],
        data_type: 2,
        name: "u8_bad".to_string(),
        int32_data: vec![300],
        ..Default::default()
    };
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(msg.contains("out of range"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

/// Non-raw DOUBLE payloads are accepted only when their values survive exactly.
#[test]
fn tensor_proto_to_array_decodes_double_data() {
    let tensor = TensorProto {
        dims: vec![2],
        data_type: 11,
        name: "f64_typed".to_string(),
        double_data: vec![1.5, -2.25],
        ..Default::default()
    };
    let arr = tensor_proto_to_array(&tensor).expect("should decode double_data");
    assert_eq!(arr[[0]], 1.5);
    assert_eq!(arr[[1]], -2.25);
}

/// A payload in a typed field that does not match data_type is rejected.
#[test]
fn tensor_proto_to_array_rejects_mismatched_typed_field() {
    let tensor = TensorProto {
        dims: vec![1],
        data_type: 7, // INT64, but payload is in int32_data
        name: "mismatch".to_string(),
        int32_data: vec![4],
        ..Default::default()
    };
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(msg.contains("int32_data"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

/// FLOAT payloads are only valid for FLOAT tensors. Previously this branch
/// bypassed data_type validation and silently loaded an INT64 tensor as f32.
#[test]
fn tensor_proto_to_array_rejects_mismatched_float_data() {
    let tensor = TensorProto {
        dims: vec![1],
        data_type: 7,
        name: "mismatched_float".to_string(),
        float_data: vec![4.0],
        ..Default::default()
    };
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(msg.contains("float_data"), "msg = {msg}");
            assert!(msg.contains("data_type 7"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

/// TensorProto uses exactly one data representation. Accepting two and picking
/// by field precedence makes malformed files decode ambiguously.
#[test]
fn tensor_proto_to_array_rejects_multiple_payload_fields() {
    let tensor = TensorProto {
        dims: vec![1],
        data_type: 1,
        name: "ambiguous".to_string(),
        raw_data: 1.0f32.to_le_bytes().to_vec(),
        float_data: vec![2.0],
        ..Default::default()
    };
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(
                msg.contains("multiple populated data fields"),
                "msg = {msg}"
            );
            assert!(msg.contains("raw_data"), "msg = {msg}");
            assert!(msg.contains("float_data"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

/// ONNX stores typed FLOAT16/BFLOAT16 values as uint16 bit patterns widened
/// into int32_data.
#[test]
fn tensor_proto_to_array_decodes_half_precision_int32_data() {
    let fp16 = TensorProto {
        dims: vec![3],
        data_type: 10,
        name: "fp16_typed".to_string(),
        int32_data: [1.0f32, -0.5, f32::INFINITY]
            .into_iter()
            .map(|value| i32::from(half::f16::from_f32(value).to_bits()))
            .collect(),
        ..Default::default()
    };
    let fp16_array = tensor_proto_to_array(&fp16).expect("typed FLOAT16 should decode");
    assert_eq!(
        fp16_array.iter().copied().collect::<Vec<_>>(),
        vec![1.0, -0.5, f32::INFINITY]
    );

    let bf16 = TensorProto {
        dims: vec![3],
        data_type: 16,
        name: "bf16_typed".to_string(),
        int32_data: [1.0f32, -0.5, f32::NEG_INFINITY]
            .into_iter()
            .map(|value| i32::from(half::bf16::from_f32(value).to_bits()))
            .collect(),
        ..Default::default()
    };
    let bf16_array = tensor_proto_to_array(&bf16).expect("typed BFLOAT16 should decode");
    assert_eq!(
        bf16_array.iter().copied().collect::<Vec<_>>(),
        vec![1.0, -0.5, f32::NEG_INFINITY]
    );
}

#[test]
fn tensor_proto_to_array_rejects_out_of_range_half_bit_pattern() {
    let tensor = TensorProto {
        dims: vec![1],
        data_type: 10,
        name: "fp16_bad_bits".to_string(),
        int32_data: vec![i32::from(u16::MAX) + 1],
        ..Default::default()
    };
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    assert!(
        err.to_string().contains("bit pattern") && err.to_string().contains("out of range"),
        "msg = {err}"
    );
}

/// A shape-carrying tensor with every data field empty must fail closed, never
/// load as zeros: the payload may live in an encoding the schema does not model.
#[test]
fn tensor_proto_to_array_rejects_shape_carrying_tensor_without_data() {
    for data_type in [2, 3, 6, 7] {
        let tensor = TensorProto {
            dims: vec![2],
            data_type,
            name: "empty_payload".to_string(),
            ..Default::default()
        };
        let err = tensor_proto_to_array(&tensor).unwrap_err();
        match err {
            NyError::ModelLoad(msg) => {
                assert!(
                    msg.contains("has no data"),
                    "data_type={data_type} msg={msg}"
                );
            }
            other => unreachable!("unexpected error: {other:?}"),
        }
    }
}

/// A zero-element tensor (a 0 in dims) is legitimately empty.
#[test]
fn tensor_proto_to_array_accepts_zero_element_tensor_without_data() {
    let tensor = TensorProto {
        dims: vec![0],
        data_type: 7,
        name: "empty_tensor".to_string(),
        ..Default::default()
    };
    let arr = tensor_proto_to_array(&tensor).expect("zero-element tensor should load");
    assert_eq!(arr.len(), 0);
}

/// Empty tensors still require a supported element type; otherwise unsupported
/// metadata could bypass type validation solely because no payload is present.
#[test]
fn tensor_proto_to_array_rejects_unsupported_zero_element_tensor() {
    let tensor = TensorProto {
        dims: vec![0],
        data_type: 99,
        name: "empty_unknown".to_string(),
        ..Default::default()
    };
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    assert!(
        err.to_string().contains("unsupported ONNX data_type 99"),
        "msg = {err}"
    );
}

#[test]
fn value_info_to_tensor_spec_rejects_missing_tensor_type() {
    for value_info in [
        ValueInfoProto {
            name: "missing_type".to_string(),
            r#type: None,
        },
        ValueInfoProto {
            name: "non_tensor_type".to_string(),
            r#type: Some(TypeProto { tensor_type: None }),
        },
    ] {
        let err = value_info_to_tensor_spec(&value_info).unwrap_err();
        assert!(
            err.to_string().contains("missing tensor type metadata"),
            "msg = {err}"
        );
        assert!(err.to_string().contains(&value_info.name), "msg = {err}");
    }
}

#[test]
fn value_info_to_tensor_spec_rejects_unknown_rank() {
    let value_info = ValueInfoProto {
        name: "unknown_rank".to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: None,
            }),
        }),
    };

    let err = value_info_to_tensor_spec(&value_info).unwrap_err();
    assert!(
        err.to_string().contains("missing tensor shape metadata"),
        "msg = {err}"
    );
    assert!(err.to_string().contains(&value_info.name), "msg = {err}");
}

/// data_location=EXTERNAL keeps the payload outside the model file; loading it
/// as if the inline fields were authoritative would fabricate data.
#[test]
fn tensor_proto_to_array_rejects_external_data_location() {
    let tensor = TensorProto {
        dims: vec![2],
        data_type: 1,
        name: "external".to_string(),
        data_location: 1,
        ..Default::default()
    };
    let err = tensor_proto_to_array(&tensor).unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(msg.contains("data_location"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

#[test]
fn onnx_elem_type_to_dtype_rejects_unsupported_graph_dtypes() {
    // These types are rejected as graph input/output dtypes even though
    // tensor_proto_to_array can decode their raw_data.
    let cases = [
        (0, "UNDEFINED"),
        (2, "UINT8"),
        (3, "INT8"),
        (9, "BOOL"),
        (10, "FLOAT16"),
        (11, "DOUBLE"),
        (16, "BFLOAT16"),
        (999, "Unknown ONNX element type"),
    ];
    for (elem_type, label) in cases {
        let err = onnx_elem_type_to_dtype(elem_type).unwrap_err();
        match err {
            NyError::ModelLoad(msg) => {
                assert!(msg.contains(label), "elem_type={elem_type} msg={msg}");
            }
            other => unreachable!("unexpected error: {other:?}"),
        }
    }
}

// -- i64_to_f32_warned / i32_to_f32_warned unit tests (#2848) --

/// Values within f32 exact-integer range convert exactly.
#[test]
fn i64_to_f32_warned_exact_range() {
    assert_eq!(i64_to_f32_warned(0, "test"), 0.0);
    assert_eq!(i64_to_f32_warned(1, "test"), 1.0);
    assert_eq!(i64_to_f32_warned(-1, "test"), -1.0);
    assert_eq!(i64_to_f32_warned(1 << 24, "test"), (1i64 << 24) as f32);
    assert_eq!(i64_to_f32_warned(-(1 << 24), "test"), -(1i64 << 24) as f32);
}

/// Values outside f32 exact-integer range still produce finite f32 (warned).
#[test]
fn i64_to_f32_warned_precision_loss() {
    // 2^24 + 1 = 16_777_217 — not exactly representable
    let result = i64_to_f32_warned((1 << 24) + 1, "test");
    assert!(result.is_finite());
    // i64::MAX — commonly used as ONNX Slice sentinel
    let result = i64_to_f32_warned(i64::MAX, "test");
    assert!(result.is_finite());
}

#[test]
fn i64_to_f32_checked_rejects_precision_loss_2360() {
    let err = i64_to_f32_checked((1 << 24) + 1, "shape tensor")
        .expect_err("checked conversion must reject rounded i64 values");
    assert!(
        err.to_string().contains("precision loss"),
        "error should mention precision loss, got: {err}"
    );
}

#[test]
fn i64_to_f32_checked_accepts_exact_range_2360() {
    let converted =
        i64_to_f32_checked(1 << 24, "shape tensor").expect("2^24 should remain exact in f32");
    assert_eq!(converted, (1i64 << 24) as f32);
}

/// i32 values within f32 exact-integer range convert exactly.
#[test]
fn i32_to_f32_warned_exact_range() {
    assert_eq!(i32_to_f32_warned(0, "test"), 0.0);
    assert_eq!(i32_to_f32_warned(1 << 24, "test"), (1i32 << 24) as f32);
    assert_eq!(i32_to_f32_warned(-(1 << 24), "test"), -(1i32 << 24) as f32);
}

/// i32 values outside f32 exact-integer range produce warned conversion.
#[test]
fn i32_to_f32_warned_precision_loss() {
    let result = i32_to_f32_warned(i32::MAX, "test");
    assert!(result.is_finite());
    let result = i32_to_f32_warned(i32::MIN, "test");
    assert!(result.is_finite());
}

#[test]
fn f64_to_f32_checked_reports_precision_loss_2360() {
    let (converted, loses_precision) =
        f64_to_f32_checked(std::f64::consts::PI, "test").expect("pi should stay within f32 range");
    assert!(
        loses_precision,
        "pi should lose precision on f64→f32 downcast"
    );
    assert!((converted - std::f32::consts::PI).abs() < 1e-6);
}

#[test]
fn f64_to_f32_checked_rejects_out_of_range_2360() {
    let err = f64_to_f32_checked(f64::MAX, "double weight").unwrap_err();
    match err {
        NyError::ModelLoad(msg) => {
            assert!(msg.contains("f64→f32 out of range"), "msg = {msg}");
            assert!(msg.contains("double weight"), "msg = {msg}");
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}
