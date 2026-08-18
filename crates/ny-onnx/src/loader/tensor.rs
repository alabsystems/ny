// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::attributes::validate_attribute_storage;
use super::numeric_cast::{f64_to_f32_checked, i32_to_f32_warned, i64_to_f32_warned};
use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::{DataType, TensorSpec, WeightStore};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use std::collections::HashMap;
use tracing::warn;

#[derive(Debug)]
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

pub(super) fn extract_constant_tensor(
    node: &onnx_proto::NodeProto,
) -> Result<Option<LoadedTensor>> {
    if node.op_type != "Constant" {
        return Ok(None);
    }
    validate_constant_payload_schema(node)?;
    let attr = &node.attribute[0];
    let loaded = match (attr.name.as_str(), attr.r#type) {
        ("value", attribute_type::TENSOR) => {
            // Type 4 is TENSOR
            let Some(t) = attr.t.as_ref() else {
                return Ok(None);
            };
            tensor_proto_to_loaded_tensor(t)?
        }
        ("value_int", attribute_type::INT) => {
            // Type 2 is INT (single value)
            LoadedTensor {
                float_data: ArrayD::from_elem(
                    IxDyn(&[]),
                    i64_to_f32_warned(attr.i_value(), "Constant value_int"),
                ),
                integer_data: Some(ArrayD::from_elem(IxDyn(&[]), attr.i_value())),
                integer_range: Some((i64::MIN, i64::MAX)),
            }
        }
        ("value_ints", attribute_type::INTS) => {
            // Type 7 is INTS array
            let integer_values = attr.ints.clone();
            let values: Vec<f32> = attr
                .ints
                .iter()
                .map(|&v| i64_to_f32_warned(v, "Constant value_ints"))
                .collect();
            LoadedTensor {
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
                integer_range: Some((i64::MIN, i64::MAX)),
            }
        }
        ("value_float", attribute_type::FLOAT) => {
            // Type 1 is FLOAT (single value)
            LoadedTensor {
                float_data: ArrayD::from_elem(IxDyn(&[]), attr.f_value()),
                integer_data: None,
                integer_range: None,
            }
        }
        ("value_floats", attribute_type::FLOATS) => {
            // Type 6 is FLOATS array
            LoadedTensor {
                float_data: ArrayD::from_shape_vec(
                    IxDyn(&[attr.floats.len()]),
                    attr.floats.clone(),
                )
                .map_err(|e| NyError::ModelLoad(format!("shape error in value_floats: {e}")))?,
                integer_data: None,
                integer_range: None,
            }
        }
        _ => unreachable!("validate_constant_payload_schema admitted an unknown payload"),
    };
    Ok(Some(loaded))
}

/// Require the exact Constant payload subset that NY can materialize without
/// erasing ambiguity.  Constant has a one-of attribute schema: accepting the
/// first recognized attribute would silently choose semantics for malformed
/// nodes with duplicate or competing payloads before graph conversion.
pub(super) fn validate_constant_payload_schema(node: &onnx_proto::NodeProto) -> Result<()> {
    if node.op_type != "Constant" {
        return Ok(());
    }
    let [attribute] = node.attribute.as_slice() else {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Constant node '{}' must have exactly one supported payload attribute, got {}",
            node.name,
            node.attribute.len()
        )));
    };
    let supported = matches!(
        (attribute.name.as_str(), attribute.r#type),
        ("value", attribute_type::TENSOR)
            | ("value_float", attribute_type::FLOAT)
            | ("value_floats", attribute_type::FLOATS)
            | ("value_int", attribute_type::INT)
            | ("value_ints", attribute_type::INTS)
    );
    if !supported {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Constant node '{}' has unsupported or malformed payload attribute '{}' of type {}",
            node.name, attribute.name, attribute.r#type
        )));
    }
    validate_attribute_storage(node, attribute)?;
    if attribute.name == "value" && attribute.t.is_none() {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Constant node '{}' has a tensor payload without TensorProto data",
            node.name
        )));
    }
    Ok(())
}

/// Apply the versioned part of Constant's one-of payload schema before any
/// materialization can erase which protobuf attribute authored the value.
pub(super) fn validate_constant_payload_for_opset(
    node: &onnx_proto::NodeProto,
    opset: i64,
) -> Result<()> {
    validate_constant_payload_schema(node)?;
    if node.op_type == "Constant" && opset < 12 && node.attribute[0].name != "value" {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Constant node '{}' uses '{}' at opset {opset}; scalar and list payload attributes require opset 12 or newer",
            node.name, node.attribute[0].name
        )));
    }
    Ok(())
}

/// Validate ConstantOfShape's exact standard schema before constant folding.
/// Its optional `value` is a TensorProto containing one scalar fill element;
/// accepting a list/scalar lookalike or selecting one of competing attributes
/// would otherwise let a malformed producer disappear during folding.
pub(super) fn validate_constant_of_shape_schema(node: &onnx_proto::NodeProto) -> Result<()> {
    if node.op_type != "ConstantOfShape" {
        return Ok(());
    }
    if node.input.len() != 1
        || node.input[0].is_empty()
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX ConstantOfShape node '{}' must have exactly one non-empty input and exactly one non-empty output; got inputs {:?} and outputs {:?}",
            node.name, node.input, node.output
        )));
    }

    let Some(attribute) = node.attribute.first() else {
        return Ok(());
    };
    if node.attribute.len() != 1
        || attribute.name != "value"
        || attribute.r#type != attribute_type::TENSOR
        || attribute.t.is_none()
    {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX ConstantOfShape node '{}' accepts only one optional tensor-valued 'value' attribute",
            node.name
        )));
    }
    validate_attribute_storage(node, attribute)?;

    let tensor = attribute.t.as_ref().expect("checked above");
    let mut element_count = 1_u64;
    for &dimension in &tensor.dims {
        let dimension = u64::try_from(dimension).map_err(|_| {
            NyError::ModelLoad(format!(
                "standard ONNX ConstantOfShape node '{}' has a negative value-tensor dimension {}",
                node.name, dimension
            ))
        })?;
        element_count = element_count.checked_mul(dimension).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "standard ONNX ConstantOfShape node '{}' value-tensor shape overflows its element count",
                node.name
            ))
        })?;
    }
    if element_count != 1 {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX ConstantOfShape node '{}' value tensor must represent exactly one scalar element, but shape {:?} represents {element_count}",
            node.name, tensor.dims
        )));
    }
    Ok(())
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
/// Returns an error when tensor type or shape metadata is missing, or the dtype
/// is unsupported (see `onnx_elem_type_to_dtype`). Graph inputs and outputs
/// cannot safely default a missing element type or conflate an unknown rank
/// with the empty shape of a scalar.
pub(super) fn value_info_to_tensor_spec(info: &onnx_proto::ValueInfoProto) -> Result<TensorSpec> {
    let tensor_type = info
        .r#type
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .ok_or_else(|| {
            NyError::ModelLoad(format!(
                "ONNX value '{}' is missing tensor type metadata",
                info.name
            ))
        })?;
    let tensor_shape = tensor_type.shape.as_ref().ok_or_else(|| {
        NyError::ModelLoad(format!(
            "ONNX value '{}' is missing tensor shape metadata",
            info.name
        ))
    })?;
    let shape = tensor_shape
        .dim
        .iter()
        .map(|d| match &d.value {
            Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
            _ => -1, // Dynamic dimension
        })
        .collect();
    let dtype = onnx_elem_type_to_dtype(tensor_type.elem_type)?;

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
/// for exact constants, but graph computation is accepted only when its
/// floating-point semantics match ny's f32 verifier.
///
/// Supported graph dtypes: FLOAT (1), INT32 (6), INT64 (7). FLOAT16 and DOUBLE
/// are rejected because ny does not model their per-operation rounding.
/// See: <https://onnx.ai/onnx/api/mapping.html#onnx-ml-tensor-type>
fn onnx_elem_type_to_dtype(elem_type: i32) -> Result<DataType> {
    match elem_type {
        0 => Err(NyError::ModelLoad(
            "ONNX tensor has UNDEFINED element type".to_string(),
        )),
        1 => Ok(DataType::Float32),  // FLOAT
        6 => Ok(DataType::Int32),    // INT32
        7 => Ok(DataType::Int64),    // INT64
        10 => Err(NyError::ModelLoad(
            "Unsupported ONNX graph dtype: FLOAT16 (10). ny verification models FLOAT32 operation rounding, not FLOAT16 execution; convert the graph to FLOAT32".to_string(),
        )),
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
        11 => Err(NyError::ModelLoad(
            "Unsupported ONNX graph dtype: DOUBLE (11). ny verification models FLOAT32 operation rounding, not DOUBLE execution; convert the graph to FLOAT32".to_string(),
        )),
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
    if tensor.data_location != 0 {
        return Err(NyError::ModelLoad(format!(
            "Tensor {}: data_location {} requires a model-origin external-data resolver",
            tensor.name, tensor.data_location
        )));
    }
    tensor_proto_to_loaded_tensor_impl(tensor, &tensor.raw_data)
}

/// Decode a tensor whose authoritative raw payload was read from an ONNX
/// external-data file. The resolver validates the external metadata and slice
/// length before this conversion boundary.
pub(super) fn tensor_proto_to_loaded_tensor_with_external_raw(
    tensor: &onnx_proto::TensorProto,
    raw_data: &[u8],
) -> Result<LoadedTensor> {
    if tensor.data_location != 1 {
        return Err(NyError::ModelLoad(format!(
            "Tensor {}: internal external-data decode requested for data_location {}",
            tensor.name, tensor.data_location
        )));
    }
    tensor_proto_to_loaded_tensor_impl(tensor, raw_data)
}

/// Exact number of raw bytes required by the tensor shape and element type.
///
/// External-data lengths are checked against this value before allocating or
/// reading a payload, preventing a crafted length from becoming an
/// attacker-controlled allocation size.
pub(super) fn expected_raw_data_byte_len(tensor: &onnx_proto::TensorProto) -> Result<usize> {
    let mut elements = 1usize;
    for &dim in &tensor.dims {
        let dim = usize::try_from(dim).map_err(|_| {
            NyError::ModelLoad(format!(
                "Tensor {} has invalid dimension {}",
                tensor.name, dim
            ))
        })?;
        elements = elements.checked_mul(dim).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "Tensor {} shape {:?} is too large to materialize",
                tensor.name, tensor.dims
            ))
        })?;
    }
    let bytes_per_element = match tensor.data_type {
        1 | 6 => 4usize,
        2 | 3 => 1,
        4 | 5 | 10 | 16 => 2,
        7 | 11 => 8,
        other => {
            return Err(NyError::ModelLoad(format!(
                "Tensor {}: unsupported ONNX data_type {}",
                tensor.name, other
            )));
        }
    };
    elements.checked_mul(bytes_per_element).ok_or_else(|| {
        NyError::ModelLoad(format!(
            "Tensor {} raw payload is too large to materialize",
            tensor.name
        ))
    })
}

fn tensor_proto_to_loaded_tensor_impl(
    tensor: &onnx_proto::TensorProto,
    raw_data: &[u8],
) -> Result<LoadedTensor> {
    if tensor.segment.is_some() {
        return Err(NyError::ModelLoad(format!(
            "Tensor {} uses deprecated segmented storage, which NY does not support",
            tensor.name
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

    // ONNX data types: 1=FLOAT, 2=UINT8, 3=INT8, 4=UINT16, 5=INT16,
    // 6=INT32, 7=INT64, 10=FLOAT16, 11=DOUBLE, 16=BFLOAT16. Validate the
    // type independently of payload presence so an empty tensor cannot
    // smuggle an unsupported element type through the data-free path.
    let data_type = tensor.data_type;
    if !matches!(data_type, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 | 11 | 16) {
        return Err(NyError::ModelLoad(format!(
            "Tensor {}: unsupported ONNX data_type {}",
            tensor.name, data_type
        )));
    }

    // TensorProto permits exactly one storage representation. Silently
    // choosing one field by precedence makes malformed input ambiguous and
    // can decode different values than another conforming implementation.
    let populated_data_fields: Vec<&str> = [
        (!raw_data.is_empty()).then_some("raw_data"),
        (!tensor.float_data.is_empty()).then_some("float_data"),
        (!tensor.int32_data.is_empty()).then_some("int32_data"),
        (!tensor.int64_data.is_empty()).then_some("int64_data"),
        (!tensor.double_data.is_empty()).then_some("double_data"),
        (!tensor.string_data.is_empty()).then_some("string_data"),
        (!tensor.uint64_data.is_empty()).then_some("uint64_data"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if populated_data_fields.len() > 1 {
        return Err(NyError::ModelLoad(format!(
            "Tensor {} has multiple populated data fields: {}",
            tensor.name,
            populated_data_fields.join(", ")
        )));
    }

    let raw_len_checked = |bytes_per: usize| -> Result<usize> {
        if !raw_data.len().is_multiple_of(bytes_per) {
            return Err(NyError::ModelLoad(format!(
                "Tensor {} raw_data length {} is not divisible by {}",
                tensor.name,
                raw_data.len(),
                bytes_per
            )));
        }
        let elements = raw_data.len() / bytes_per;
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
        if !raw_data.is_empty() {
            // Raw data - interpret based on data type
            match data_type {
                1 => {
                    // FLOAT - 4 bytes per element
                    raw_len_checked(4)?;
                    (
                        raw_data
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
                        raw_data.iter().map(|&b| b as f32).collect(),
                        Some(raw_data.iter().map(|&b| b as i64).collect()),
                        Some((0, u8::MAX as i64)),
                    )
                }
                3 => {
                    // INT8 - 1 byte per element
                    raw_len_checked(1)?;
                    (
                        raw_data.iter().map(|&b| (b as i8) as f32).collect(),
                        Some(raw_data.iter().map(|&b| (b as i8) as i64).collect()),
                        Some((i8::MIN as i64, i8::MAX as i64)),
                    )
                }
                4 => {
                    // UINT16 - 2 bytes per element
                    raw_len_checked(2)?;
                    let integer_values: Vec<i64> = raw_data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|chunk| u16::from_le_bytes(*chunk) as i64)
                        .collect();
                    (
                        integer_values.iter().map(|&value| value as f32).collect(),
                        Some(integer_values),
                        Some((0, u16::MAX as i64)),
                    )
                }
                5 => {
                    // INT16 - 2 bytes per element
                    raw_len_checked(2)?;
                    let integer_values: Vec<i64> = raw_data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|chunk| i16::from_le_bytes(*chunk) as i64)
                        .collect();
                    (
                        integer_values.iter().map(|&value| value as f32).collect(),
                        Some(integer_values),
                        Some((i16::MIN as i64, i16::MAX as i64)),
                    )
                }
                6 => {
                    // INT32 - 4 bytes per element
                    // SAFETY(as f32): Guarded via i32_to_f32_warned — warns for |value| > 2^24.
                    raw_len_checked(4)?;
                    let name = &tensor.name;
                    let integer_values: Vec<i64> = raw_data
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
                    let integer_values: Vec<i64> = raw_data
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
                        raw_data
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
                        raw_data
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
                        raw_data
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
                            .collect(),
                        None,
                        None,
                    )
                }
                _ => unreachable!("data_type validated before raw_data dispatch"),
            }
        } else if !tensor.float_data.is_empty() {
            match data_type {
                1 => (tensor.float_data.clone(), None, None),
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "Tensor {}: unsupported ONNX data_type {} in float_data",
                        tensor.name, data_type
                    )));
                }
            }
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
                4 => {
                    // UINT16 stored as widened int32 values.
                    for &value in &tensor.int32_data {
                        if !(0..=i32::from(u16::MAX)).contains(&value) {
                            return Err(NyError::ModelLoad(format!(
                                "Tensor {}: UINT16 int32_data value {} out of range",
                                tensor.name, value
                            )));
                        }
                    }
                    (
                        tensor.int32_data.iter().map(|&v| v as f32).collect(),
                        Some(tensor.int32_data.iter().map(|&v| i64::from(v)).collect()),
                        Some((0, u16::MAX as i64)),
                    )
                }
                5 => {
                    // INT16 stored as widened int32 values.
                    for &value in &tensor.int32_data {
                        if !(i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&value) {
                            return Err(NyError::ModelLoad(format!(
                                "Tensor {}: INT16 int32_data value {} out of range",
                                tensor.name, value
                            )));
                        }
                    }
                    (
                        tensor.int32_data.iter().map(|&v| v as f32).collect(),
                        Some(tensor.int32_data.iter().map(|&v| i64::from(v)).collect()),
                        Some((i16::MIN as i64, i16::MAX as i64)),
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
                10 | 16 => {
                    // ONNX stores FLOAT16 and BFLOAT16 typed payloads as
                    // widened uint16 bit patterns in int32_data.
                    let type_name = if data_type == 10 {
                        "FLOAT16"
                    } else {
                        "BFLOAT16"
                    };
                    let mut values = Vec::with_capacity(tensor.int32_data.len());
                    for &value in &tensor.int32_data {
                        let bits = u16::try_from(value).map_err(|_| {
                            NyError::ModelLoad(format!(
                                "Tensor {}: {} int32_data bit pattern {} out of range",
                                tensor.name, type_name, value
                            ))
                        })?;
                        values.push(if data_type == 10 {
                            f16::from_bits(bits).to_f32()
                        } else {
                            bf16::from_bits(bits).to_f32()
                        });
                    }
                    (values, None, None)
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
            // A dim of 0 makes an empty tensor legitimately data-free. Keep
            // authored integer provenance even though there are no values:
            // an absent sidecar/range would make an empty integer parameter
            // indistinguishable from an empty FLOAT tensor after loading.
            let integer_range = integer_range_for_data_type(data_type);
            (Vec::new(), integer_range.map(|_| Vec::new()), integer_range)
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

fn integer_range_for_data_type(data_type: i32) -> Option<(i64, i64)> {
    match data_type {
        2 => Some((0, u8::MAX as i64)),
        3 => Some((i8::MIN as i64, i8::MAX as i64)),
        4 => Some((0, u16::MAX as i64)),
        5 => Some((i16::MIN as i64, i16::MAX as i64)),
        6 => Some((i32::MIN as i64, i32::MAX as i64)),
        7 => Some((i64::MIN, i64::MAX)),
        _ => None,
    }
}

/// Convert a DOUBLE constant payload only when every value is exactly
/// representable as f32. Rounding an initializer would verify a different
/// network, so precision loss is a hard model-load error.
fn f64_values_to_f32_vec(
    values: impl ExactSizeIterator<Item = f64>,
    name: &str,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(values.len());

    for value in values {
        let (converted, loses_precision) = f64_to_f32_checked(value, name)?;
        if loses_precision {
            return Err(NyError::ModelLoad(format!(
                "DOUBLE tensor {name} cannot be represented exactly as FLOAT32: {value} would round to {converted}; refusing to verify a changed network"
            )));
        }
        out.push(converted);
    }

    Ok(out)
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
