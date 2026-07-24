// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::numeric_cast::{f64_to_f32_checked, i32_to_f32_warned, i64_to_f32_warned};
use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::{DataType, TensorSpec, WeightStore};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use std::collections::HashMap;
use tracing::{debug, trace, warn};

pub(super) struct LoadedTensor {
    pub(super) float_data: ArrayD<f32>,
    pub(super) integer_data: Option<ArrayD<i64>>,
    pub(super) integer_range: Option<(i64, i64)>,
}

pub(super) fn node_attr_tensor_scalar_f32(node: &onnx_proto::NodeProto) -> Result<Option<f32>> {
    if node.op_type != "Constant" {
        return Ok(None);
    }
    if let Some(loaded) = extract_constant_tensor(node)? {
        if loaded.float_data.len() != 1 {
            return Ok(None);
        }
        return Ok(Some(
            loaded.float_data.iter().next().copied().unwrap_or_default(),
        ));
    }
    Ok(None)
}

/// Extract the tensor value from a Constant node's "value" attribute.
///
/// Returns `Ok(None)` when the node is not a Constant or has no recognized value attribute.
/// Returns `Err` when the value attribute exists but cannot be parsed (corrupt tensor data).
pub(super) fn extract_constant_value(node: &onnx_proto::NodeProto) -> Result<Option<ArrayD<f32>>> {
    Ok(extract_constant_tensor(node)?.map(|tensor| tensor.float_data))
}

pub(super) fn extract_constant_tensor(
    node: &onnx_proto::NodeProto,
) -> Result<Option<LoadedTensor>> {
    if node.op_type != "Constant" {
        return Ok(None);
    }
    for attr in &node.attribute {
        if attr.name == "value" && attr.r#type == attribute_type::TENSOR {
            // Type 4 is TENSOR
            let Some(t) = attr.t.as_ref() else {
                return Ok(None);
            };
            return Ok(Some(tensor_proto_to_loaded_tensor(t)?));
        }
        if attr.name == "value_int" && attr.r#type == attribute_type::INT {
            // Type 2 is INT (single value)
            return Ok(Some(LoadedTensor {
                float_data: ArrayD::from_elem(
                    IxDyn(&[]),
                    i64_to_f32_warned(attr.i, "Constant value_int"),
                ),
                integer_data: Some(ArrayD::from_elem(IxDyn(&[]), attr.i)),
                integer_range: None,
            }));
        }
        if attr.name == "value_ints" && attr.r#type == attribute_type::INTS {
            // Type 7 is INTS array
            let integer_values = attr.ints.clone();
            let values: Vec<f32> = attr
                .ints
                .iter()
                .map(|&v| i64_to_f32_warned(v, "Constant value_ints"))
                .collect();
            return Ok(Some(LoadedTensor {
                float_data: ArrayD::from_shape_vec(IxDyn(&[values.len()]), values)
                    .map_err(|e| NyError::ModelLoad(format!("shape error in value_ints: {e}")))?,
                integer_data: Some(
                    ArrayD::from_shape_vec(IxDyn(&[integer_values.len()]), integer_values)
                        .map_err(|e| {
                            NyError::ModelLoad(format!(
                                "shape error in value_ints integer payload: {e}"
                            ))
                        })?,
                ),
                integer_range: None,
            }));
        }
        if attr.name == "value_float" && attr.r#type == attribute_type::FLOAT {
            // Type 1 is FLOAT (single value)
            return Ok(Some(LoadedTensor {
                float_data: ArrayD::from_elem(IxDyn(&[]), attr.f),
                integer_data: None,
                integer_range: None,
            }));
        }
        if attr.name == "value_floats" && attr.r#type == attribute_type::FLOATS {
            // Type 6 is FLOATS array
            return Ok(Some(LoadedTensor {
                float_data: ArrayD::from_shape_vec(
                    IxDyn(&[attr.floats.len()]),
                    attr.floats.clone(),
                )
                .map_err(|e| NyError::ModelLoad(format!("shape error in value_floats: {e}")))?,
                integer_data: None,
                integer_range: None,
            }));
        }
    }
    Ok(None)
}

pub(super) fn scalar_from_weights(weights: &WeightStore, name: &str) -> Option<f32> {
    let weights = weights.get(name)?;
    if weights.len() != 1 {
        return None;
    }
    weights.iter().next().copied()
}

pub(super) fn scalar_for_input(
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
    name: &str,
) -> Option<f32> {
    if let Some(value) = scalar_from_weights(weights, name) {
        return Some(value);
    }
    let idx = *producer_by_output.get(name)?;
    node_attr_tensor_scalar_f32(&nodes[idx])
        .map_err(|e| {
            warn!("scalar_for_input({name}): tensor parse failed: {e}");
            e
        })
        .ok()
        .flatten()
}

/// Convert ONNX ValueInfoProto to TensorSpec.
///
/// Extracts tensor shape and dtype from the ONNX value info. Dynamic dimensions
/// (symbolic or param) are represented as -1 in the shape.
///
/// # Errors
/// Returns error for unsupported dtypes (see `onnx_elem_type_to_dtype`).
pub(super) fn value_info_to_tensor_spec(info: &onnx_proto::ValueInfoProto) -> Result<TensorSpec> {
    let tensor_type = info.r#type.as_ref().and_then(|t| t.tensor_type.as_ref());
    let shape = tensor_type
        .and_then(|tt| tt.shape.as_ref())
        .map(|s| {
            s.dim
                .iter()
                .map(|d| match &d.value {
                    Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
                    _ => -1, // Dynamic dimension
                })
                .collect()
        })
        .unwrap_or_default();
    let dtype = match tensor_type {
        Some(tt) => onnx_elem_type_to_dtype(tt.elem_type)?,
        None => {
            debug!(
                "ValueInfoProto '{}' missing type info, defaulting to Float32",
                info.name
            );
            DataType::Float32
        }
    };

    Ok(TensorSpec {
        name: info.name.clone(),
        shape,
        dtype,
    })
}

/// Convert ONNX element type to ny DataType for graph input/output specs.
///
/// Maps ONNX TensorProto.DataType enum values to ny DataType.
/// Returns an error for unsupported element types rather than silently defaulting.
///
/// Note: This function governs graph-level dtype metadata (input/output specs).
/// Tensor *data* decoding in `tensor_proto_to_array` supports additional types
/// (UINT8, INT8, DOUBLE, BFLOAT16) by converting to f32.
///
/// Supported graph dtypes: FLOAT (1), INT32 (6), INT64 (7), FLOAT16 (10), DOUBLE (11).
/// DOUBLE is mapped to Float32 (f64→f32 downcast, matching tensor data handling).
/// See: <https://onnx.ai/onnx/api/mapping.html#onnx-ml-tensor-type>
fn onnx_elem_type_to_dtype(elem_type: i32) -> Result<DataType> {
    match elem_type {
        0 => Err(NyError::ModelLoad(
            "ONNX tensor has UNDEFINED element type".to_string(),
        )),
        1 => Ok(DataType::Float32),  // FLOAT
        6 => Ok(DataType::Int32),    // INT32
        7 => Ok(DataType::Int64),    // INT64
        10 => Ok(DataType::Float16), // FLOAT16
        // Types supported for raw_data decode but not as graph dtypes
        2 => Err(NyError::ModelLoad(
            "Unsupported ONNX graph dtype: UINT8 (2). Graph inputs/outputs support FLOAT, FLOAT16, INT32, INT64".to_string(),
        )),
        3 => Err(NyError::ModelLoad(
            "Unsupported ONNX graph dtype: INT8 (3). Graph inputs/outputs support FLOAT, FLOAT16, INT32, INT64".to_string(),
        )),
        9 => Err(NyError::ModelLoad(
            "Unsupported ONNX graph dtype: BOOL (9). Graph inputs/outputs support FLOAT, FLOAT16, INT32, INT64".to_string(),
        )),
        11 => {
            // DOUBLE — downcast to Float32 (all tensor data is f32 internally).
            // Precision loss for f64 values outside f32 range; emits a trace-level
            // diagnostic so the decision is visible in debug logs.
            trace!("ONNX graph dtype DOUBLE (11) mapped to Float32 (f64→f32 downcast)");
            Ok(DataType::Float32)
        }
        16 => Err(NyError::ModelLoad(
            "Unsupported ONNX graph dtype: BFLOAT16 (16). Graph inputs/outputs support FLOAT, FLOAT16, INT32, INT64".to_string(),
        )),
        _ => Err(NyError::ModelLoad(format!(
            "Unknown ONNX element type: {}. Graph inputs/outputs support FLOAT (1), FLOAT16 (10), INT32 (6), INT64 (7)",
            elem_type
        ))),
    }
}

/// Convert ONNX TensorProto to ndarray.
pub(super) fn tensor_proto_to_array(tensor: &onnx_proto::TensorProto) -> Result<ArrayD<f32>> {
    Ok(tensor_proto_to_loaded_tensor(tensor)?.float_data)
}

pub(super) fn tensor_proto_to_loaded_tensor(
    tensor: &onnx_proto::TensorProto,
) -> Result<LoadedTensor> {
    // data_location=1 (EXTERNAL) keeps the payload in side files listed in
    // external_data; the in-file data fields are then legitimately empty and
    // must not be interpreted (e.g. zero-filled).
    if tensor.data_location != 0 {
        return Err(NyError::ModelLoad(format!(
            "Tensor {}: data_location {} is unsupported (external tensor data must be inlined into the model file)",
            tensor.name, tensor.data_location
        )));
    }
    let mut shape = Vec::with_capacity(tensor.dims.len());
    for &dim in &tensor.dims {
        if dim < 0 {
            return Err(NyError::ModelLoad(format!(
                "Tensor {} has negative dimension {}",
                tensor.name, dim
            )));
        }
        shape.push(dim as usize);
    }
    let expected_len = if shape.is_empty() {
        1
    } else {
        let mut total = 1usize;
        for dim in &shape {
            total = total.checked_mul(*dim).ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "Tensor {} shape {:?} is too large to materialize",
                    tensor.name, shape
                ))
            })?;
        }
        total
    };

    // ONNX data types: 1=FLOAT, 2=UINT8, 3=INT8, 6=INT32, 7=INT64, 10=FLOAT16, 11=DOUBLE, 16=BFLOAT16
    let data_type = tensor.data_type;
    let raw_len_checked = |bytes_per: usize| -> Result<usize> {
        if !tensor.raw_data.len().is_multiple_of(bytes_per) {
            return Err(NyError::ModelLoad(format!(
                "Tensor {} raw_data length {} is not divisible by {}",
                tensor.name,
                tensor.raw_data.len(),
                bytes_per
            )));
        }
        let elements = tensor.raw_data.len() / bytes_per;
        if expected_len != elements && !(expected_len == 0 && elements == 0) {
            return Err(NyError::ModelLoad(format!(
                "Tensor {} expected {} elements but raw_data has {}",
                tensor.name, expected_len, elements
            )));
        }
        Ok(elements)
    };

    // Data can be in raw_data or in one of the typed repeated fields
    // (float_data, int32_data, int64_data, double_data) depending on data_type.
    let (data, integer_data, integer_range): (Vec<f32>, Option<Vec<i64>>, Option<(i64, i64)>) =
        if !tensor.raw_data.is_empty() {
            // Raw data - interpret based on data type
            match data_type {
                1 => {
                    // FLOAT - 4 bytes per element
                    raw_len_checked(4)?;
                    (
                        tensor
                            .raw_data
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|chunk| f32::from_le_bytes(*chunk))
                            .collect(),
                        None,
                        None,
                    )
                }
                2 => {
                    // UINT8 - 1 byte per element
                    raw_len_checked(1)?;
                    (
                        tensor.raw_data.iter().map(|&b| b as f32).collect(),
                        Some(tensor.raw_data.iter().map(|&b| b as i64).collect()),
                        Some((0, u8::MAX as i64)),
                    )
                }
                3 => {
                    // INT8 - 1 byte per element
                    raw_len_checked(1)?;
                    (
                        tensor.raw_data.iter().map(|&b| (b as i8) as f32).collect(),
                        Some(tensor.raw_data.iter().map(|&b| (b as i8) as i64).collect()),
                        Some((i8::MIN as i64, i8::MAX as i64)),
                    )
                }
                6 => {
                    // INT32 - 4 bytes per element
                    // SAFETY(as f32): Guarded via i32_to_f32_warned — warns for |value| > 2^24.
                    raw_len_checked(4)?;
                    let name = &tensor.name;
                    let integer_values: Vec<i64> = tensor
                        .raw_data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| i32::from_le_bytes(*chunk) as i64)
                        .collect();
                    (
                        integer_values
                            .iter()
                            .map(|&value| i32_to_f32_warned(value as i32, name))
                            .collect(),
                        Some(integer_values),
                        Some((i32::MIN as i64, i32::MAX as i64)),
                    )
                }
                7 => {
                    // INT64 - 8 bytes per element
                    // SAFETY(as f32): Guarded via i64_to_f32_warned — warns for |value| > 2^24.
                    // ONNX uses i64::MAX as sentinel in Slice start/end; this round-trips
                    // incorrectly through f32 and the warning makes it diagnosable.
                    raw_len_checked(8)?;
                    let name = &tensor.name;
                    let integer_values: Vec<i64> = tensor
                        .raw_data
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|chunk| i64::from_le_bytes(*chunk))
                        .collect();
                    (
                        integer_values
                            .iter()
                            .map(|&value| i64_to_f32_warned(value, name))
                            .collect(),
                        Some(integer_values),
                        Some((i64::MIN, i64::MAX)),
                    )
                }
                10 => {
                    // FLOAT16 - 2 bytes per element
                    // Reference: ONNX spec TensorProto.DataType FLOAT16 = 10
                    raw_len_checked(2)?;
                    (
                        tensor
                            .raw_data
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
                            .collect(),
                        None,
                        None,
                    )
                }
                11 => {
                    // DOUBLE - 8 bytes per element
                    raw_len_checked(8)?;
                    let values = f64_values_to_f32_vec(
                        tensor
                            .raw_data
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .map(|chunk| f64::from_le_bytes(*chunk)),
                        &tensor.name,
                    )?;
                    (values, None, None)
                }
                16 => {
                    // BFLOAT16 - 2 bytes per element
                    // Reference: ONNX spec TensorProto.DataType BFLOAT16 = 16
                    raw_len_checked(2)?;
                    (
                        tensor
                            .raw_data
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
                            .collect(),
                        None,
                        None,
                    )
                }
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "Tensor {}: unsupported ONNX data_type {} in raw_data",
                        tensor.name, data_type
                    )));
                }
            }
        } else if !tensor.float_data.is_empty() {
            (tensor.float_data.clone(), None, None)
        } else if !tensor.int32_data.is_empty() {
            // Non-raw payload: per the ONNX spec, int32_data carries UINT8,
            // INT8, and INT32 tensors (among others) when raw_data is unused.
            match data_type {
                2 => {
                    // UINT8 stored as widened int32 values
                    for &value in &tensor.int32_data {
                        if !(0..=i32::from(u8::MAX)).contains(&value) {
                            return Err(NyError::ModelLoad(format!(
                                "Tensor {}: UINT8 int32_data value {} out of range",
                                tensor.name, value
                            )));
                        }
                    }
                    (
                        tensor.int32_data.iter().map(|&v| v as f32).collect(),
                        Some(tensor.int32_data.iter().map(|&v| i64::from(v)).collect()),
                        Some((0, u8::MAX as i64)),
                    )
                }
                3 => {
                    // INT8 stored as widened int32 values
                    for &value in &tensor.int32_data {
                        if !(i32::from(i8::MIN)..=i32::from(i8::MAX)).contains(&value) {
                            return Err(NyError::ModelLoad(format!(
                                "Tensor {}: INT8 int32_data value {} out of range",
                                tensor.name, value
                            )));
                        }
                    }
                    (
                        tensor.int32_data.iter().map(|&v| v as f32).collect(),
                        Some(tensor.int32_data.iter().map(|&v| i64::from(v)).collect()),
                        Some((i8::MIN as i64, i8::MAX as i64)),
                    )
                }
                6 => {
                    // INT32
                    // SAFETY(as f32): Guarded via i32_to_f32_warned — warns for |value| > 2^24.
                    let name = &tensor.name;
                    (
                        tensor
                            .int32_data
                            .iter()
                            .map(|&value| i32_to_f32_warned(value, name))
                            .collect(),
                        Some(tensor.int32_data.iter().map(|&v| i64::from(v)).collect()),
                        Some((i32::MIN as i64, i32::MAX as i64)),
                    )
                }
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "Tensor {}: unsupported ONNX data_type {} in int32_data",
                        tensor.name, data_type
                    )));
                }
            }
        } else if !tensor.int64_data.is_empty() {
            match data_type {
                7 => {
                    // INT64
                    // SAFETY(as f32): Guarded via i64_to_f32_warned — warns for |value| > 2^24.
                    let name = &tensor.name;
                    (
                        tensor
                            .int64_data
                            .iter()
                            .map(|&value| i64_to_f32_warned(value, name))
                            .collect(),
                        Some(tensor.int64_data.clone()),
                        Some((i64::MIN, i64::MAX)),
                    )
                }
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "Tensor {}: unsupported ONNX data_type {} in int64_data",
                        tensor.name, data_type
                    )));
                }
            }
        } else if !tensor.double_data.is_empty() {
            match data_type {
                11 => (
                    // DOUBLE
                    f64_values_to_f32_vec(tensor.double_data.iter().copied(), &tensor.name)?,
                    None,
                    None,
                ),
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "Tensor {}: unsupported ONNX data_type {} in double_data",
                        tensor.name, data_type
                    )));
                }
            }
        } else if expected_len == 0 {
            // A dim of 0 makes an empty tensor legitimately data-free.
            (Vec::new(), None, None)
        } else {
            // A shape-carrying tensor with every data field empty means the
            // payload lives in an encoding this decoder does not model (or the
            // file is corrupt). Substituting zeros would silently change the
            // network, so refuse to load. Genuinely-optional inputs (e.g. a
            // Quantize/DequantizeLinear zero_point) are omitted from the
            // node's input list rather than supplied as empty initializers.
            return Err(NyError::ModelLoad(format!(
                "Tensor {}: data_type {} expects {} elements but has no data in \
                 raw_data/float_data/int32_data/int64_data/double_data",
                tensor.name, data_type, expected_len
            )));
        };
    if expected_len != data.len() && !(expected_len == 0 && data.is_empty()) {
        return Err(NyError::ModelLoad(format!(
            "Tensor {} expected {} elements but got {}",
            tensor.name,
            expected_len,
            data.len()
        )));
    }

    let float_data = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to create array: {}", e)))?;
    let integer_data = integer_data
        .map(|values| {
            ArrayD::from_shape_vec(IxDyn(&shape), values)
                .map_err(|e| NyError::ModelLoad(format!("Failed to create integer array: {}", e)))
        })
        .transpose()?;

    Ok(LoadedTensor {
        float_data,
        integer_data,
        integer_range,
    })
}

/// Downcast DOUBLE payload values to f32, failing on out-of-range values and
/// aggregating precision-loss diagnostics into a single warning.
fn f64_values_to_f32_vec(
    values: impl ExactSizeIterator<Item = f64>,
    name: &str,
) -> Result<Vec<f32>> {
    let mut precision_loss_count = 0usize;
    let mut first_precision_loss = None;
    let mut out = Vec::with_capacity(values.len());

    for value in values {
        let (converted, loses_precision) = f64_to_f32_checked(value, name)?;
        if loses_precision {
            precision_loss_count += 1;
            if first_precision_loss.is_none() {
                first_precision_loss = Some((value, converted));
            }
        }
        out.push(converted);
    }

    if let Some((original, converted)) = first_precision_loss {
        warn!(
            "DOUBLE tensor {} loses precision on {} values during f64→f32 downcast; \
             first example: {} -> {}",
            name, precision_loss_count, original, converted
        );
    }

    Ok(out)
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
