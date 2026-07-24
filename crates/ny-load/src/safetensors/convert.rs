// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Dtype conversion helpers for SafeTensors tensor views.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use safetensors::tensor::TensorView;
use tracing::{debug, warn};

/// f32 can represent all integers in [-2^24, 2^24] exactly.
const F32_INT_EXACT_LIMIT: i64 = 1 << 24; // 16_777_216

/// Convert a SafeTensors tensor view to an f32 ndarray.
///
/// Returns an error for unsupported dtypes or shape/data mismatches (#2900).
pub(crate) fn tensor_view_to_f32_array(
    view: &TensorView<'_>,
    shape: &[usize],
    name: &str,
) -> Result<ArrayD<f32>> {
    let data = view.data();
    let floats: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => convert_f32(data),
        safetensors::Dtype::F16 => convert_f16(data),
        safetensors::Dtype::BF16 => convert_bf16(data),
        safetensors::Dtype::F64 => {
            debug!(
                "Converting f64 tensor '{}' to f32 with potential precision loss",
                name
            );
            convert_f64(data)
        }
        safetensors::Dtype::I64 => convert_i64(data),
        safetensors::Dtype::I32 => convert_i32(data),
        other => {
            return Err(NyError::ModelLoad(format!(
                "Tensor '{}' has unsupported dtype {:?}",
                name, other
            )));
        }
    };
    ArrayD::from_shape_vec(IxDyn(shape), floats).map_err(|e| {
        NyError::ModelLoad(format!(
            "Tensor '{}' shape {:?} doesn't match data: {}",
            name, shape, e
        ))
    })
}

fn convert_f32(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn convert_f16(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|c| half_to_f32(u16::from_le_bytes(*c)))
        .collect()
}

fn convert_bf16(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
        .collect()
}

// SAFETY(as f32): f64→f32 uses round-to-nearest-even. For weight loading
// this is acceptable (weights are stored as f64 but consumed as f32).
// The debug! log at the call site makes the conversion visible.
fn convert_f64(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c) as f32)
        .collect()
}

// SAFETY(as f32): Guarded — warns on values outside f32 exact-integer range.
// Matches ONNX loader's i64_to_f32_warned behavior (ny-onnx tensor.rs:29).
fn convert_i64(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            let v = i64::from_le_bytes(*c);
            if v.unsigned_abs() > F32_INT_EXACT_LIMIT as u64 {
                warn!(
                    "SafeTensors i64→f32 precision loss: {v} exceeds \
                     f32 exact-integer range ±{F32_INT_EXACT_LIMIT}"
                );
            }
            v as f32
        })
        .collect()
}

// SAFETY(as f32): Guarded — warns on values outside f32 exact-integer range.
// Matches ONNX loader's i32_to_f32_warned behavior (ny-onnx tensor.rs:46).
fn convert_i32(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            let v = i32::from_le_bytes(*c);
            if (v as i64).unsigned_abs() > F32_INT_EXACT_LIMIT as u64 {
                warn!(
                    "SafeTensors i32→f32 precision loss: {v} exceeds \
                     f32 exact-integer range ±{F32_INT_EXACT_LIMIT}"
                );
            }
            v as f32
        })
        .collect()
}

/// Convert IEEE 754 half-precision (f16) to f32.
///
/// Delegates to the `half` crate for correctness, eliminating divergence
/// between hand-rolled and crate implementations (#2772).
#[inline]
pub fn half_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Convert bfloat16 to f32.
/// BF16 has the same exponent range as f32, just truncated mantissa.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
