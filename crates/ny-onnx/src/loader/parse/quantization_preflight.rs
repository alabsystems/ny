// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Raw-protobuf validation for ONNX linear quantization operators.
//!
//! This gate intentionally runs before constant folding.  WeightStore exposes
//! a normalized f32 mirror for several authored tensor dtypes, and a constant
//! Q/DQ node can disappear entirely.  Therefore only the raw graph can prove
//! that ny's FLOAT32 arithmetic is the arithmetic the model declared.

use crate::onnx_proto::{self, AttributeProto, GraphProto, NodeProto};
use ny_core::{NyError, Result};
use std::collections::{HashMap, HashSet};

use super::super::external_data::ExternalDataResolver;
use super::super::tensor::{
    extract_constant_tensor, tensor_proto_to_loaded_tensor,
    tensor_proto_to_loaded_tensor_with_external_raw, validate_constant_payload_schema,
    LoadedTensor,
};

const FLOAT: i32 = 1;
const UINT8: i32 = 2;
const INT8: i32 = 3;
const UINT16: i32 = 4;
const INT16: i32 = 5;
const INT32: i32 = 6;
const INT64: i32 = 7;
const BOOL: i32 = 9;

/// Validate every standard-domain QuantizeLinear/DequantizeLinear node before
/// any rewrite can erase its schema, attributes, or authored tensor dtypes.
#[cfg(test)]
pub(super) fn validate_quantization_schemas(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
) -> Result<()> {
    validate_quantization_schemas_with_external(graph, opset_imports, None)
}

/// File-backed models may keep exact integer control tensors in ONNX external
/// data. Resolve those bytes through the model-origin capability before
/// proving Q/DQ parameter shapes; byte-loaded models deliberately pass no
/// resolver and continue to fail closed on external references.
pub(super) fn validate_quantization_schemas_with_external(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
    external_data: Option<&mut ExternalDataResolver>,
) -> Result<()> {
    let mut dtypes = RawDtypeResolver::new(graph);
    let mut shapes = RawShapeResolver::new(graph, opset_imports, external_data);

    for node in &graph.node {
        if !is_standard_domain(&node.domain)
            || !matches!(node.op_type.as_str(), "QuantizeLinear" | "DequantizeLinear")
        {
            continue;
        }

        let opset = standard_opset(node, opset_imports)?;
        let attrs = validate_schema(node, opset)?;
        validate_dtypes(node, opset, &attrs, &mut dtypes)?;
        validate_parameter_shapes(node, opset, &mut shapes)?;
    }

    Ok(())
}

fn is_standard_domain(domain: &str) -> bool {
    matches!(domain, "" | "ai.onnx")
}

fn standard_opset(node: &NodeProto, opset_imports: &HashMap<String, i64>) -> Result<i64> {
    opset_imports
        .get(&node.domain)
        .or_else(|| {
            if node.domain.is_empty() {
                opset_imports.get("ai.onnx")
            } else {
                opset_imports.get("")
            }
        })
        .copied()
        .ok_or_else(|| {
            NyError::ModelLoad(format!(
                "standard ONNX {} node '{}' has no standard-domain opset import",
                node.op_type, node.name
            ))
        })
}

#[derive(Clone, Copy, Default)]
struct QuantizationAttrs {
    output_dtype: Option<i64>,
}

fn validate_schema(node: &NodeProto, opset: i64) -> Result<QuantizationAttrs> {
    if opset < 10 {
        return Err(node_error(
            node,
            format!("requires ONNX opset 10 or newer, but the model imports opset {opset}"),
        ));
    }
    if !(node.input.len() == 2 || node.input.len() == 3)
        || node.input[0].is_empty()
        || node.input[1].is_empty()
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return Err(node_error(
            node,
            format!(
                "must have exactly two required inputs, at most one optional zero-point input, and one non-empty output; got {} input(s) and {} output(s)",
                node.input.len(),
                node.output.len()
            ),
        ));
    }

    let mut seen = HashSet::new();
    let mut parsed = QuantizationAttrs::default();
    for attr in &node.attribute {
        if !seen.insert(attr.name.as_str()) {
            return Err(node_error(
                node,
                format!("has duplicate '{}' attributes", attr.name),
            ));
        }
        if attr.r#type != onnx_proto::attribute_type::INT {
            return Err(node_error(
                node,
                format!("attribute '{}' must have ONNX INT type", attr.name),
            ));
        }

        match (node.op_type.as_str(), attr.name.as_str()) {
            (_, "axis") => require_attr_opset(node, attr, opset, 13)?,
            ("QuantizeLinear", "saturate") => {
                require_attr_opset(node, attr, opset, 19)?;
                if !matches!(attr.i_value(), 0 | 1) {
                    return Err(node_error(
                        node,
                        format!(
                            "attribute 'saturate' must be 0 or 1, got {}",
                            attr.i_value()
                        ),
                    ));
                }
            }
            ("QuantizeLinear", "block_size") => {
                require_attr_opset(node, attr, opset, 21)?;
                require_zero_block_size(node, attr)?;
            }
            ("DequantizeLinear", "block_size") => {
                require_attr_opset(node, attr, opset, 21)?;
                require_zero_block_size(node, attr)?;
            }
            ("QuantizeLinear", "precision") => {
                require_attr_opset(node, attr, opset, 23)?;
                if attr.i_value() != 0 && attr.i_value() != i64::from(FLOAT) {
                    return Err(node_error(
                        node,
                        format!(
                            "attribute 'precision' must be default/0 or FLOAT ({FLOAT}), got {}",
                            attr.i_value()
                        ),
                    ));
                }
            }
            ("QuantizeLinear", "output_dtype") => {
                require_attr_opset(node, attr, opset, 21)?;
                if attr.i_value() != 0 && !is_quantized_integer_dtype_i64(attr.i_value()) {
                    return Err(node_error(
                        node,
                        format!(
                            "attribute 'output_dtype' must be default/0 or a supported integer dtype, got {}",
                            attr.i_value()
                        ),
                    ));
                }
                parsed.output_dtype = Some(attr.i_value());
            }
            ("DequantizeLinear", "output_dtype") => {
                require_attr_opset(node, attr, opset, 23)?;
                if attr.i_value() != 0 && attr.i_value() != i64::from(FLOAT) {
                    return Err(node_error(
                        node,
                        format!(
                            "attribute 'output_dtype' must be default/0 or FLOAT ({FLOAT}), got {}",
                            attr.i_value()
                        ),
                    ));
                }
                parsed.output_dtype = Some(attr.i_value());
            }
            _ => {
                return Err(node_error(
                    node,
                    format!("has unsupported attribute '{}'", attr.name),
                ));
            }
        }
    }

    Ok(parsed)
}

fn require_attr_opset(
    node: &NodeProto,
    attr: &AttributeProto,
    opset: i64,
    minimum: i64,
) -> Result<()> {
    if opset < minimum {
        return Err(node_error(
            node,
            format!(
                "attribute '{}' requires ONNX opset {minimum} or newer, but the model imports opset {opset}",
                attr.name
            ),
        ));
    }
    Ok(())
}

fn require_zero_block_size(node: &NodeProto, attr: &AttributeProto) -> Result<()> {
    if attr.i_value() != 0 {
        return Err(node_error(
            node,
            format!(
                "blocked quantization is unsupported; attribute 'block_size' must be 0, got {}",
                attr.i_value()
            ),
        ));
    }
    Ok(())
}

fn validate_dtypes(
    node: &NodeProto,
    opset: i64,
    attrs: &QuantizationAttrs,
    dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    let x_dtype = require_dtype(node, dtypes, &node.input[0], "input")?;
    let scale_dtype = require_dtype(node, dtypes, &node.input[1], "scale")?;
    if scale_dtype != FLOAT {
        return Err(node_error(
            node,
            format!(
                "scale '{}' must have authored FLOAT ({FLOAT}) dtype, got {scale_dtype}",
                node.input[1]
            ),
        ));
    }

    let zero_point_name = node.input.get(2).filter(|name| !name.is_empty());
    let output_dtype = if node.op_type == "QuantizeLinear" {
        if !matches!(x_dtype, FLOAT | INT32) {
            return Err(node_error(
                node,
                format!(
                    "input '{}' must have authored FLOAT ({FLOAT}) or INT32 ({INT32}) dtype, got {x_dtype}",
                    node.input[0]
                ),
            ));
        }

        let zero_point_dtype = match zero_point_name {
            Some(name) => {
                let dtype = require_dtype(node, dtypes, name, "zero point")?;
                if !is_quantized_integer_dtype_at_opset(dtype, opset) {
                    return Err(node_error(
                        node,
                        format!(
                            "zero point '{name}' has integer dtype {dtype}, which is unsupported at opset {opset}"
                        ),
                    ));
                }
                Some(dtype)
            }
            None => None,
        };

        quantize_output_dtype(node, attrs.output_dtype, zero_point_dtype)?
    } else {
        if !is_dequantize_input_dtype_at_opset(x_dtype, opset) {
            return Err(node_error(
                node,
                format!(
                    "input '{}' has quantized dtype {x_dtype}, which is unsupported at opset {opset}",
                    node.input[0],
                ),
            ));
        }
        if let Some(name) = zero_point_name {
            if x_dtype == INT32 {
                return Err(node_error(
                    node,
                    "INT32 DequantizeLinear input must not specify a zero point".to_string(),
                ));
            }
            let zero_point_dtype = require_dtype(node, dtypes, name, "zero point")?;
            if zero_point_dtype != x_dtype {
                return Err(node_error(
                    node,
                    format!(
                        "zero point '{name}' dtype {zero_point_dtype} must match input '{}' dtype {x_dtype}",
                        node.input[0]
                    ),
                ));
            }
        }
        FLOAT
    };

    // If the model authored output type metadata, require it to agree with the
    // operator's type rule.  This also memoizes Q output types so a following
    // DQ node can consume a Q result without value_info.
    let resolved_output = require_dtype(node, dtypes, &node.output[0], "output")?;
    if resolved_output != output_dtype {
        return Err(node_error(
            node,
            format!(
                "output '{}' has dtype {resolved_output}, but the operator declares dtype {output_dtype}",
                node.output[0]
            ),
        ));
    }

    Ok(())
}

fn validate_parameter_shapes(
    node: &NodeProto,
    opset: i64,
    shapes: &mut RawShapeResolver<'_, '_>,
) -> Result<()> {
    let scale_shape = require_shape(node, shapes, &node.input[1], "scale")?;
    if opset < 13 {
        if !scale_shape.is_empty() {
            return Err(node_error(
                node,
                format!(
                    "opset {opset} requires a true rank-0 scalar scale, but '{}' has shape {:?}",
                    node.input[1], scale_shape
                ),
            ));
        }
    } else if scale_shape.len() > 1 {
        return Err(node_error(
            node,
            format!(
                "scale '{}' has shape {:?}; ny supports only rank-0 per-tensor or rank-1 per-axis quantization, not blocked same-rank parameters",
                node.input[1], scale_shape
            ),
        ));
    }
    if scale_shape.len() == 1 && scale_shape[0] <= 0 {
        return Err(node_error(
            node,
            format!(
                "rank-1 scale '{}' must be non-empty, got shape {:?}",
                node.input[1], scale_shape
            ),
        ));
    }

    // A rank-1 parameter is per-axis, not a scalar merely because its
    // authored extent happens to be one.  Prove both that the selected axis is
    // legal for the raw input rank and that the parameter length is exactly
    // that axis extent.  Otherwise the later materializer's scalar fast path
    // could erase an invalid model by broadcasting a length-one vector.
    if scale_shape.len() == 1 {
        let input_shape = require_shape(node, shapes, &node.input[0], "input")?;
        let rank = i64::try_from(input_shape.len()).map_err(|_| {
            node_error(
                node,
                format!("input '{}' rank cannot be represented", node.input[0]),
            )
        })?;
        let authored_axis = node
            .attribute
            .iter()
            .find(|attribute| attribute.name == "axis")
            .map_or(1, |attribute| attribute.i_value());
        let normalized_axis = if authored_axis < 0 {
            authored_axis.checked_add(rank)
        } else {
            Some(authored_axis)
        }
        .filter(|axis| *axis >= 0 && *axis < rank)
        .ok_or_else(|| {
            node_error(
                node,
                format!(
                    "per-axis parameter selects axis {authored_axis}, outside input '{}' rank {}",
                    node.input[0],
                    input_shape.len()
                ),
            )
        })?;
        let axis = usize::try_from(normalized_axis).map_err(|_| {
            node_error(
                node,
                format!("normalized per-axis index {normalized_axis} cannot be represented"),
            )
        })?;
        if input_shape[axis] != scale_shape[0] {
            return Err(node_error(
                node,
                format!(
                    "per-axis scale '{}' extent {} must equal input '{}' axis {authored_axis} extent {}",
                    node.input[1], scale_shape[0], node.input[0], input_shape[axis]
                ),
            ));
        }
    }

    if let Some(zero_point) = node.input.get(2).filter(|name| !name.is_empty()) {
        let zero_point_shape = require_shape(node, shapes, zero_point, "zero point")?;
        if zero_point_shape != scale_shape {
            return Err(node_error(
                node,
                format!(
                    "zero point '{zero_point}' shape {:?} must exactly match scale '{}' shape {:?}",
                    zero_point_shape, node.input[1], scale_shape
                ),
            ));
        }
    }

    Ok(())
}

fn quantize_output_dtype(
    node: &NodeProto,
    output_dtype_attr: Option<i64>,
    zero_point_dtype: Option<i32>,
) -> Result<i32> {
    let explicit = output_dtype_attr.filter(|dtype| *dtype != 0);
    if let (Some(explicit), Some(zero_point_dtype)) = (explicit, zero_point_dtype) {
        if explicit != i64::from(zero_point_dtype) {
            return Err(node_error(
                node,
                format!("output_dtype {explicit} must match zero-point dtype {zero_point_dtype}"),
            ));
        }
    }

    match explicit {
        Some(dtype) => i32::try_from(dtype).map_err(|_| {
            node_error(
                node,
                format!("output_dtype {dtype} cannot be represented as an ONNX dtype"),
            )
        }),
        None => Ok(zero_point_dtype.unwrap_or(UINT8)),
    }
}

fn require_dtype(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    value: &str,
    role: &str,
) -> Result<i32> {
    dtypes.resolve(value)?.ok_or_else(|| {
        node_error(
            node,
            format!(
                "cannot prove the raw authored dtype of {role} tensor '{value}' before constant folding"
            ),
        )
    })
}

fn require_shape(
    node: &NodeProto,
    shapes: &mut RawShapeResolver<'_, '_>,
    value: &str,
    role: &str,
) -> Result<Vec<i64>> {
    shapes.resolve(value)?.ok_or_else(|| {
        node_error(
            node,
            format!(
                "cannot prove the raw authored shape of {role} tensor '{value}' before constant folding"
            ),
        )
    })
}

fn is_quantized_integer_dtype(dtype: i32) -> bool {
    matches!(dtype, UINT8 | INT8 | UINT16 | INT16)
}

fn is_quantized_integer_dtype_at_opset(dtype: i32, opset: i64) -> bool {
    matches!(dtype, UINT8 | INT8) || (opset >= 21 && matches!(dtype, UINT16 | INT16))
}

fn is_quantized_integer_dtype_i64(dtype: i64) -> bool {
    i32::try_from(dtype)
        .ok()
        .is_some_and(is_quantized_integer_dtype)
}

fn is_dequantize_input_dtype_at_opset(dtype: i32, opset: i64) -> bool {
    dtype == INT32 || is_quantized_integer_dtype_at_opset(dtype, opset)
}

fn node_error(node: &NodeProto, detail: String) -> NyError {
    NyError::ModelLoad(format!(
        "standard ONNX {} node '{}' {detail}",
        node.op_type, node.name
    ))
}

/// A deliberately narrow raw dtype proof engine.  It accepts authored tensor
/// metadata, initializer/Constant/Cast types, and operators whose output type
/// is unambiguously preserved from a selected data input.  Anything else is
/// unknown and makes Q/DQ fail closed.
pub(super) struct RawDtypeResolver<'a> {
    graph: &'a GraphProto,
    authored: HashMap<&'a str, Vec<i32>>,
    producers: HashMap<&'a str, Vec<usize>>,
    memo: HashMap<String, Option<i32>>,
}

impl<'a> RawDtypeResolver<'a> {
    pub(super) fn new(graph: &'a GraphProto) -> Self {
        let mut authored: HashMap<&str, Vec<i32>> = HashMap::new();
        for initializer in &graph.initializer {
            if !initializer.name.is_empty() {
                authored
                    .entry(initializer.name.as_str())
                    .or_default()
                    .push(initializer.data_type);
            }
        }
        for info in graph
            .input
            .iter()
            .chain(graph.output.iter())
            .chain(graph.value_info().iter())
        {
            let Some(dtype) = info
                .r#type
                .as_ref()
                .and_then(|ty| ty.tensor_type.as_ref())
                .map(|tensor| tensor.elem_type)
                .filter(|dtype| *dtype != 0)
            else {
                continue;
            };
            if !info.name.is_empty() {
                authored.entry(info.name.as_str()).or_default().push(dtype);
            }
        }

        let mut producers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, node) in graph.node.iter().enumerate() {
            for output in node.output.iter().filter(|name| !name.is_empty()) {
                producers.entry(output.as_str()).or_default().push(index);
            }
        }

        Self {
            graph,
            authored,
            producers,
            memo: HashMap::new(),
        }
    }

    pub(super) fn resolve(&mut self, value: &str) -> Result<Option<i32>> {
        self.resolve_inner(value, &mut HashSet::new())
    }

    fn resolve_inner(&mut self, value: &str, active: &mut HashSet<String>) -> Result<Option<i32>> {
        if let Some(dtype) = self.memo.get(value) {
            return Ok(*dtype);
        }
        if !active.insert(value.to_string()) {
            return Err(NyError::ModelLoad(format!(
                "cycle encountered while proving raw ONNX dtype of '{value}'"
            )));
        }

        let mut candidates = self.authored.get(value).cloned().unwrap_or_default();
        if let Some(producers) = self.producers.get(value).cloned() {
            if producers.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "cannot prove raw ONNX dtype of '{value}': it has {} producers",
                    producers.len()
                )));
            }
            if let Some(dtype) = self.infer_node_output_dtype(producers[0], value, active)? {
                candidates.push(dtype);
            }
        }

        active.remove(value);
        candidates.retain(|dtype| *dtype != 0);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.len() > 1 {
            return Err(NyError::ModelLoad(format!(
                "conflicting raw ONNX dtypes {:?} for value '{value}'",
                candidates
            )));
        }
        let dtype = candidates.first().copied();
        self.memo.insert(value.to_string(), dtype);
        Ok(dtype)
    }

    fn infer_node_output_dtype(
        &mut self,
        node_index: usize,
        output: &str,
        active: &mut HashSet<String>,
    ) -> Result<Option<i32>> {
        // Clone one untrusted protobuf node before recursing so dtype proofs
        // can mutably memoize through producer inputs without aliasing `self`.
        let node = self.graph.node[node_index].clone();
        if !is_standard_domain(&node.domain) {
            return Ok(None);
        }

        // Split and the standard recurrent operators are genuinely
        // multi-output, but every output shares the first data input's tensor
        // type. Handle them before the
        // single-output proof gate. Authentic NN4SYS graphs commonly reduce a
        // Split branch without ValueInfo metadata, and recurrent lowering
        // fixtures consume Y_h/Y_c before the recurrent node is expanded.
        if matches!(node.op_type.as_str(), "Split" | "RNN" | "GRU" | "LSTM")
            && node.output.iter().any(|name| name == output)
        {
            return self.first_input_dtype(&node, active);
        }
        if node.op_type == "TopK" {
            return match node.output.iter().position(|name| name == output) {
                Some(0) => self.first_input_dtype(&node, active),
                Some(1) => Ok(Some(INT64)),
                _ => Ok(None),
            };
        }

        // ONNX protobufs retain omitted trailing optional outputs as empty
        // strings. A node with one live first output is still a single-output
        // dtype proof; requiring `output.len() == 1` needlessly rejected common
        // Dropout/normalization exports whose optional outputs were omitted by
        // placeholder.
        let exactly_this_output = node.output.first().is_some_and(|name| name == output)
            && node.output.iter().skip(1).all(String::is_empty);
        if !exactly_this_output {
            return Ok(None);
        }

        match node.op_type.as_str() {
            "Constant" => Ok(constant_dtype(&node)),
            "Cast" => Ok(cast_target_dtype(&node)),
            "ConstantOfShape" => Ok(constant_of_shape_dtype(&node)),
            "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" => Ok(Some(BOOL)),
            "QuantizeLinear" => {
                let attrs = raw_output_dtype_attribute(&node)?;
                let zero_point_dtype = match node.input.get(2).filter(|name| !name.is_empty()) {
                    Some(name) => self.resolve_inner(name, active)?,
                    None => None,
                };
                quantize_output_dtype(&node, attrs, zero_point_dtype).map(Some)
            }
            "DequantizeLinear" => Ok(Some(FLOAT)),
            // Standard ONNX shape-query operators always produce INT64.
            "Shape" | "Size" | "NonZero" | "ArgMax" | "ArgMin" => Ok(Some(INT64)),

            // These operators preserve the dtype of their first data input.
            "Identity"
            | "Reshape"
            | "Transpose"
            | "Flatten"
            | "Squeeze"
            | "Unsqueeze"
            | "Expand"
            | "Tile"
            | "Slice"
            | "Gather"
            | "GatherElements"
            | "GatherND"
            | "ScatterND"
            | "Pad"
            | "Resize"
            | "Upsample"
            | "Conv"
            | "ConvTranspose"
            | "BatchNormalization"
            | "InstanceNormalization"
            | "LayerNormalization"
            | "SimplifiedLayerNormalization"
            | "GroupNormalization"
            | "RMSNormalization"
            | "Dropout"
            | "AveragePool"
            | "GlobalAveragePool"
            | "MaxPool"
            | "Clip"
            | "Abs"
            | "Neg"
            | "Relu"
            | "LeakyRelu"
            | "PRelu"
            | "Elu"
            | "Selu"
            | "HardSigmoid"
            | "HardSwish"
            | "Celu"
            | "ThresholdedRelu"
            | "Shrink"
            | "Mish"
            | "Gelu"
            | "Swish"
            | "Sigmoid"
            | "Tanh"
            | "Exp"
            | "Log"
            | "Sqrt"
            | "Erf"
            | "Reciprocal"
            | "Floor"
            | "Ceil"
            | "Round"
            | "Softplus"
            | "Softsign"
            | "Sign"
            | "Sin"
            | "Cos"
            | "Tan"
            | "Atan"
            | "Softmax"
            | "LogSoftmax"
            | "Pow"
            | "ReduceMean"
            | "ReduceSum"
            | "ReduceMax"
            | "ReduceMin"
            | "ReduceProd"
            | "ReduceL1"
            | "ReduceL2"
            | "CumSum" => self.first_input_dtype(&node, active),

            // ONNX constrains all data operands of these operators to one T.
            "Add" | "Sub" | "Mul" | "Div" | "Min" | "Max" | "Sum" | "MatMul" | "Concat"
            | "Range" => self.uniform_input_dtype(&node, active),
            // Gemm-11 makes C optional; exporters may encode its omission as
            // either two inputs or a trailing empty placeholder.
            "Gemm" => self.uniform_present_input_dtype(&node, active),
            "Where" => self.uniform_selected_input_dtype(&node, &[1, 2], active),
            _ => Ok(None),
        }
    }

    fn first_input_dtype(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<i32>> {
        let Some(input) = node.input.first().filter(|name| !name.is_empty()) else {
            return Ok(None);
        };
        self.resolve_inner(input, active)
    }

    fn uniform_input_dtype(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<i32>> {
        // Optional ONNX operands may be represented by an empty placeholder
        // (notably Gemm C).  The enclosing raw schema gate authenticates the
        // signature; only present operands participate in the shared T type.
        let indices: Vec<usize> = node
            .input
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (!value.is_empty()).then_some(index))
            .collect();
        self.uniform_selected_input_dtype(node, &indices, active)
    }

    fn uniform_present_input_dtype(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<i32>> {
        let mut dtype = None;
        for input in node.input.iter().filter(|name| !name.is_empty()) {
            let Some(candidate) = self.resolve_inner(input, active)? else {
                return Ok(None);
            };
            if dtype.is_some_and(|known| known != candidate) {
                return Ok(None);
            }
            dtype = Some(candidate);
        }
        Ok(dtype)
    }

    fn uniform_selected_input_dtype(
        &mut self,
        node: &NodeProto,
        indices: &[usize],
        active: &mut HashSet<String>,
    ) -> Result<Option<i32>> {
        let mut dtype = None;
        for index in indices {
            let Some(input) = node.input.get(*index).filter(|name| !name.is_empty()) else {
                return Ok(None);
            };
            let Some(candidate) = self.resolve_inner(input, active)? else {
                return Ok(None);
            };
            if dtype.is_some_and(|known| known != candidate) {
                return Ok(None);
            }
            dtype = Some(candidate);
        }
        Ok(dtype)
    }
}

/// Exact raw shape proof for quantization parameters.  Symbolic dimensions
/// are intentionally not guessed: a rank-1 scale and zero point must be shown
/// to have the same authored extent before either can be materialized as an
/// untyped WeightStore value.
struct RawShapeResolver<'graph, 'external> {
    graph: &'graph GraphProto,
    opset_imports: &'graph HashMap<String, i64>,
    authored: HashMap<&'graph str, Vec<Vec<i64>>>,
    producers: HashMap<&'graph str, Vec<usize>>,
    memo: HashMap<String, Option<Vec<i64>>>,
    integers: RawIntegerResolver<'graph, 'external>,
}

impl<'graph, 'external> RawShapeResolver<'graph, 'external> {
    fn new(
        graph: &'graph GraphProto,
        opset_imports: &'graph HashMap<String, i64>,
        external_data: Option<&'external mut ExternalDataResolver>,
    ) -> Self {
        let mut authored: HashMap<&str, Vec<Vec<i64>>> = HashMap::new();
        for initializer in &graph.initializer {
            if !initializer.name.is_empty() {
                if let Some(shape) = exact_tensor_shape(initializer) {
                    authored
                        .entry(initializer.name.as_str())
                        .or_default()
                        .push(shape);
                }
            }
        }
        for info in graph
            .input
            .iter()
            .chain(graph.output.iter())
            .chain(graph.value_info().iter())
        {
            if !info.name.is_empty() {
                if let Some(shape) = exact_value_info_shape(info) {
                    authored.entry(info.name.as_str()).or_default().push(shape);
                }
            }
        }

        let mut producers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, node) in graph.node.iter().enumerate() {
            for output in node.output.iter().filter(|name| !name.is_empty()) {
                producers.entry(output.as_str()).or_default().push(index);
            }
        }

        Self {
            graph,
            opset_imports,
            authored,
            producers,
            memo: HashMap::new(),
            integers: RawIntegerResolver::new(graph, external_data),
        }
    }

    fn resolve(&mut self, value: &str) -> Result<Option<Vec<i64>>> {
        self.resolve_inner(value, &mut HashSet::new())
    }

    fn resolve_inner(
        &mut self,
        value: &str,
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        if let Some(shape) = self.memo.get(value) {
            return Ok(shape.clone());
        }
        if !active.insert(value.to_string()) {
            return Err(NyError::ModelLoad(format!(
                "cycle encountered while proving raw ONNX shape of '{value}'"
            )));
        }

        let mut candidates = self.authored.get(value).cloned().unwrap_or_default();
        if let Some(producers) = self.producers.get(value).cloned() {
            if producers.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "cannot prove raw ONNX shape of '{value}': it has {} producers",
                    producers.len()
                )));
            }
            if let Some(shape) = self.infer_node_output_shape(producers[0], value, active)? {
                candidates.push(shape);
            }
        }

        active.remove(value);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.len() > 1 {
            return Err(NyError::ModelLoad(format!(
                "conflicting raw ONNX shapes {:?} for value '{value}'",
                candidates
            )));
        }
        let shape = candidates.into_iter().next();
        self.memo.insert(value.to_string(), shape.clone());
        Ok(shape)
    }

    fn infer_node_output_shape(
        &mut self,
        node_index: usize,
        output: &str,
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        let node = self.graph.node[node_index].clone();
        if !is_standard_domain(&node.domain) || node.output.len() != 1 || node.output[0] != output {
            return Ok(None);
        }

        match node.op_type.as_str() {
            "Constant" => Ok(constant_shape(&node)),

            // These shape-only transformations are admitted only when every
            // control tensor is an exact, inline INT64 constant. Dynamic
            // shape/axes values deliberately remain unknown at this raw-model
            // boundary.
            "Reshape" => self.reshape_output_shape(&node, active),
            "Squeeze" => self.squeeze_output_shape(&node, active),
            "Unsqueeze" => self.unsqueeze_output_shape(&node, active),
            "Transpose" => self.transpose_output_shape(&node, active),

            "Identity" if node.input.len() == 1 && node.attribute.is_empty() => {
                self.first_input_shape(&node, active)
            }
            "Cast"
                if node.input.len() == 1
                    && node.attribute.len() == 1
                    && cast_target_dtype(&node).is_some() =>
            {
                self.first_input_shape(&node, active)
            }

            // Q/DQ, normalizations, and pointwise unary operators retain
            // their first input's complete shape.  Operators with secondary
            // outputs are excluded by the exactly-one-output check above.
            "QuantizeLinear"
            | "DequantizeLinear"
            | "BatchNormalization"
            | "InstanceNormalization"
            | "LayerNormalization"
            | "GroupNormalization"
            | "RMSNormalization"
            | "Dropout"
            | "Clip"
            | "Abs"
            | "Neg"
            | "Relu"
            | "LeakyRelu"
            | "PRelu"
            | "Elu"
            | "Selu"
            | "HardSigmoid"
            | "HardSwish"
            | "ThresholdedRelu"
            | "Mish"
            | "Gelu"
            | "Sigmoid"
            | "Tanh"
            | "Exp"
            | "Log"
            | "Sqrt"
            | "Erf"
            | "Reciprocal"
            | "Floor"
            | "Ceil"
            | "Round"
            | "Softmax"
            | "LogSoftmax" => self.first_input_shape(&node, active),

            // Equal operand shapes are a sufficient (though intentionally not
            // necessary) proof of an elementwise output shape.
            "Add" | "Sub" | "Mul" | "Div" | "Min" | "Max" | "Sum" => {
                self.uniform_input_shape(&node, active)
            }
            "Where" => self.uniform_selected_input_shape(&node, &[1, 2], active),
            _ => Ok(None),
        }
    }

    fn reshape_output_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        if node.input.len() != 2 || node.input.iter().any(String::is_empty) {
            return Ok(None);
        }
        let opset = standard_opset(node, self.opset_imports)?;
        let allowzero = match (opset, node.attribute.as_slice()) {
            (_, []) => false,
            (14.., [attribute])
                if attribute.name == "allowzero"
                    && attribute.r#type == onnx_proto::attribute_type::INT
                    && matches!(attribute.i_value(), 0 | 1) =>
            {
                attribute.i_value() == 1
            }
            _ => return Ok(None),
        };
        let Some(input_shape) = self.resolve_inner(&node.input[0], active)? else {
            return Ok(None);
        };
        let Some(target) = self.exact_i64_vector(&node.input[1])? else {
            return Ok(None);
        };
        Ok(infer_exact_reshape_shape(&input_shape, &target, allowzero))
    }

    fn squeeze_output_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        if !matches!(node.input.len(), 1 | 2) || node.input[0].is_empty() {
            return Ok(None);
        }
        let opset = standard_opset(node, self.opset_imports)?;
        let Some(input_shape) = self.resolve_inner(&node.input[0], active)? else {
            return Ok(None);
        };
        let Some(axes) = self.squeeze_axes(node, opset, &input_shape)? else {
            return Ok(None);
        };
        Ok(squeeze_exact_shape(&input_shape, &axes))
    }

    fn squeeze_axes(
        &mut self,
        node: &NodeProto,
        opset: i64,
        input_shape: &[i64],
    ) -> Result<Option<Vec<i64>>> {
        if opset >= 13 {
            if !node.attribute.is_empty() {
                return Ok(None);
            }
            return match node.input.get(1).filter(|name| !name.is_empty()) {
                Some(name) => self.exact_i64_vector(name),
                None => Ok(Some(unit_extent_axes(input_shape))),
            };
        }
        if node.input.len() != 1 {
            return Ok(None);
        }
        match exact_axes_attribute(node)? {
            Some(axes) => Ok(Some(axes)),
            None if node.attribute.is_empty() => Ok(Some(unit_extent_axes(input_shape))),
            None => Ok(None),
        }
    }

    fn unsqueeze_output_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        if !matches!(node.input.len(), 1 | 2) || node.input[0].is_empty() {
            return Ok(None);
        }
        let opset = standard_opset(node, self.opset_imports)?;
        let Some(input_shape) = self.resolve_inner(&node.input[0], active)? else {
            return Ok(None);
        };
        let axes = if opset >= 13 {
            if !node.attribute.is_empty() || node.input.len() != 2 || node.input[1].is_empty() {
                return Ok(None);
            }
            let Some(axes) = self.exact_i64_vector(&node.input[1])? else {
                return Ok(None);
            };
            axes
        } else {
            if node.input.len() != 1 {
                return Ok(None);
            }
            match exact_axes_attribute(node)? {
                Some(axes) => axes,
                None => return Ok(None),
            }
        };
        Ok(unsqueeze_exact_shape(&input_shape, &axes))
    }

    fn transpose_output_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        if node.input.len() != 1 || node.input[0].is_empty() {
            return Ok(None);
        }
        let Some(input_shape) = self.resolve_inner(&node.input[0], active)? else {
            return Ok(None);
        };
        let permutation = match node.attribute.as_slice() {
            [] => (0..input_shape.len())
                .rev()
                .map(|axis| axis as i64)
                .collect(),
            [attribute]
                if attribute.name == "perm"
                    && attribute.r#type == onnx_proto::attribute_type::INTS =>
            {
                attribute.ints.clone()
            }
            _ => return Ok(None),
        };
        Ok(transpose_exact_shape(&input_shape, &permutation))
    }

    fn exact_i64_vector(&mut self, value: &str) -> Result<Option<Vec<i64>>> {
        let Some(tensor) = self.integers.resolve(value)? else {
            return Ok(None);
        };
        if tensor.dtype != 7 || tensor.shape.len() != 1 {
            return Ok(None);
        }
        Ok(Some(tensor.values))
    }

    fn first_input_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        let Some(input) = node.input.first().filter(|name| !name.is_empty()) else {
            return Ok(None);
        };
        self.resolve_inner(input, active)
    }

    fn uniform_input_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        let indices: Vec<usize> = (0..node.input.len()).collect();
        self.uniform_selected_input_shape(node, &indices, active)
    }

    fn uniform_selected_input_shape(
        &mut self,
        node: &NodeProto,
        indices: &[usize],
        active: &mut HashSet<String>,
    ) -> Result<Option<Vec<i64>>> {
        let mut shape: Option<Vec<i64>> = None;
        for index in indices {
            let Some(input) = node.input.get(*index).filter(|name| !name.is_empty()) else {
                return Ok(None);
            };
            let Some(candidate) = self.resolve_inner(input, active)? else {
                return Ok(None);
            };
            if shape.as_ref().is_some_and(|known| known != &candidate) {
                return Ok(None);
            }
            shape = Some(candidate);
        }
        Ok(shape)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExactIntegerTensor {
    dtype: i32,
    shape: Vec<i64>,
    values: Vec<i64>,
}

/// Exact-value proof used only for Reshape/Squeeze/Unsqueeze control tensors.
/// ValueInfo metadata cannot establish contents; only initializers, Constant
/// nodes, and exact integer Identity/Cast chains are admitted. File-backed
/// external payloads are read only through the retained model-origin resolver.
struct RawIntegerResolver<'graph, 'external> {
    graph: &'graph GraphProto,
    initializers: HashMap<&'graph str, Vec<usize>>,
    producers: HashMap<&'graph str, Vec<usize>>,
    memo: HashMap<String, Option<ExactIntegerTensor>>,
    external_data: Option<&'external mut ExternalDataResolver>,
}

impl<'graph, 'external> RawIntegerResolver<'graph, 'external> {
    fn new(
        graph: &'graph GraphProto,
        external_data: Option<&'external mut ExternalDataResolver>,
    ) -> Self {
        let mut initializers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, initializer) in graph.initializer.iter().enumerate() {
            if !initializer.name.is_empty() {
                initializers
                    .entry(initializer.name.as_str())
                    .or_default()
                    .push(index);
            }
        }
        let mut producers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, node) in graph.node.iter().enumerate() {
            for output in node.output.iter().filter(|name| !name.is_empty()) {
                producers.entry(output.as_str()).or_default().push(index);
            }
        }
        Self {
            graph,
            initializers,
            producers,
            memo: HashMap::new(),
            external_data,
        }
    }

    fn resolve(&mut self, value: &str) -> Result<Option<ExactIntegerTensor>> {
        self.resolve_inner(value, &mut HashSet::new())
    }

    fn resolve_inner(
        &mut self,
        value: &str,
        active: &mut HashSet<String>,
    ) -> Result<Option<ExactIntegerTensor>> {
        if let Some(tensor) = self.memo.get(value) {
            return Ok(tensor.clone());
        }
        if !active.insert(value.to_string()) {
            return Err(NyError::ModelLoad(format!(
                "cycle encountered while proving exact raw ONNX integer value '{value}'"
            )));
        }

        let mut candidates = Vec::new();
        if let Some(indices) = self.initializers.get(value).cloned() {
            if indices.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "cannot prove exact raw ONNX integer value '{value}': it has {} initializers",
                    indices.len()
                )));
            }
            if let Some(tensor) = exact_integer_initializer(
                &self.graph.initializer[indices[0]],
                self.external_data.as_deref_mut(),
            )? {
                candidates.push(tensor);
            }
        }
        if let Some(indices) = self.producers.get(value).cloned() {
            if indices.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "cannot prove exact raw ONNX integer value '{value}': it has {} producers",
                    indices.len()
                )));
            }
            if let Some(tensor) = self.infer_node_value(indices[0], value, active)? {
                candidates.push(tensor);
            }
        }

        active.remove(value);
        candidates.sort();
        candidates.dedup();
        if candidates.len() > 1 {
            return Err(NyError::ModelLoad(format!(
                "conflicting exact raw ONNX integer values for '{value}'"
            )));
        }
        let tensor = candidates.into_iter().next();
        self.memo.insert(value.to_string(), tensor.clone());
        Ok(tensor)
    }

    fn infer_node_value(
        &mut self,
        node_index: usize,
        output: &str,
        active: &mut HashSet<String>,
    ) -> Result<Option<ExactIntegerTensor>> {
        let node = self.graph.node[node_index].clone();
        if !is_standard_domain(&node.domain) || node.output.len() != 1 || node.output[0] != output {
            return Ok(None);
        }
        match node.op_type.as_str() {
            "Constant" => exact_integer_constant(&node, self.external_data.as_deref_mut()),
            "Identity"
                if node.input.len() == 1
                    && !node.input[0].is_empty()
                    && node.attribute.is_empty() =>
            {
                self.resolve_inner(&node.input[0], active)
            }
            "Cast"
                if node.input.len() == 1
                    && !node.input[0].is_empty()
                    && node.attribute.len() == 1 =>
            {
                let Some(target_dtype) = cast_target_dtype(&node) else {
                    return Ok(None);
                };
                let Some(mut tensor) = self.resolve_inner(&node.input[0], active)? else {
                    return Ok(None);
                };
                let Some(range) = exact_integer_dtype_range(target_dtype) else {
                    return Ok(None);
                };
                if !tensor
                    .values
                    .iter()
                    .all(|value| *value >= range.0 && *value <= range.1)
                {
                    return Ok(None);
                }
                tensor.dtype = target_dtype;
                Ok(Some(tensor))
            }
            _ => Ok(None),
        }
    }
}

fn exact_integer_initializer(
    tensor: &onnx_proto::TensorProto,
    external_data: Option<&mut ExternalDataResolver>,
) -> Result<Option<ExactIntegerTensor>> {
    if exact_integer_dtype_range(tensor.data_type).is_none() {
        return Ok(None);
    }
    let loaded = load_tensor_resolving_external(tensor, external_data)?;
    let Some(integer_data) = loaded.integer_data else {
        return Ok(None);
    };
    Ok(Some(ExactIntegerTensor {
        dtype: tensor.data_type,
        shape: tensor.dims.clone(),
        values: integer_data.iter().copied().collect(),
    }))
}

fn exact_integer_constant(
    node: &NodeProto,
    external_data: Option<&mut ExternalDataResolver>,
) -> Result<Option<ExactIntegerTensor>> {
    validate_constant_payload_schema(node)?;
    let Some(dtype) =
        constant_dtype(node).filter(|dtype| exact_integer_dtype_range(*dtype).is_some())
    else {
        return Ok(None);
    };
    let Some(shape) = constant_shape(node) else {
        return Ok(None);
    };
    let loaded = match node.attribute.as_slice() {
        [attribute]
            if attribute.name == "value"
                && attribute.r#type == onnx_proto::attribute_type::TENSOR =>
        {
            let Some(tensor) = attribute.t.as_ref() else {
                return Ok(None);
            };
            Some(load_tensor_resolving_external(tensor, external_data)?)
        }
        _ => extract_constant_tensor(node)?,
    };
    let Some(loaded) = loaded else {
        return Ok(None);
    };
    let Some(integer_data) = loaded.integer_data else {
        return Ok(None);
    };
    Ok(Some(ExactIntegerTensor {
        dtype,
        shape,
        values: integer_data.iter().copied().collect(),
    }))
}

fn load_tensor_resolving_external(
    tensor: &onnx_proto::TensorProto,
    external_data: Option<&mut ExternalDataResolver>,
) -> Result<LoadedTensor> {
    if let Some(raw_data) = external_data
        .map(|resolver| resolver.read_tensor(tensor))
        .transpose()?
        .flatten()
    {
        tensor_proto_to_loaded_tensor_with_external_raw(tensor, &raw_data)
    } else {
        tensor_proto_to_loaded_tensor(tensor)
    }
}

fn exact_integer_dtype_range(dtype: i32) -> Option<(i64, i64)> {
    match dtype {
        2 => Some((0, u8::MAX as i64)),
        3 => Some((i8::MIN as i64, i8::MAX as i64)),
        4 => Some((0, u16::MAX as i64)),
        5 => Some((i16::MIN as i64, i16::MAX as i64)),
        6 => Some((i32::MIN as i64, i32::MAX as i64)),
        7 => Some((i64::MIN, i64::MAX)),
        _ => None,
    }
}

fn exact_axes_attribute(node: &NodeProto) -> Result<Option<Vec<i64>>> {
    match node.attribute.as_slice() {
        [] => Ok(None),
        [attribute]
            if attribute.name == "axes" && attribute.r#type == onnx_proto::attribute_type::INTS =>
        {
            Ok(Some(attribute.ints.clone()))
        }
        _ => Ok(None),
    }
}

fn unit_extent_axes(shape: &[i64]) -> Vec<i64> {
    shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &extent)| (extent == 1).then_some(axis as i64))
        .collect()
}

fn infer_exact_reshape_shape(
    input_shape: &[i64],
    target: &[i64],
    allowzero: bool,
) -> Option<Vec<i64>> {
    let input_elements = exact_element_count(input_shape)?;
    let mut output = Vec::with_capacity(target.len());
    let mut inferred_axis = None;
    for (axis, &dimension) in target.iter().enumerate() {
        let resolved = match dimension {
            -1 if inferred_axis.replace(axis).is_none() => 1,
            -1 => return None,
            0 if !allowzero => *input_shape.get(axis)?,
            value if value >= 0 => value,
            _ => return None,
        };
        output.push(resolved);
    }

    if let Some(axis) = inferred_axis {
        // ONNX forbids combining allowzero=1, a literal zero, and -1 because
        // the inferred extent would be ambiguous.
        if allowzero && target.contains(&0) {
            return None;
        }
        let known_elements = exact_element_count(&output)?;
        if known_elements == 0 || input_elements % known_elements != 0 {
            return None;
        }
        output[axis] = i64::try_from(input_elements / known_elements).ok()?;
    } else if exact_element_count(&output)? != input_elements {
        return None;
    }
    Some(output)
}

fn exact_element_count(shape: &[i64]) -> Option<u128> {
    shape.iter().try_fold(1_u128, |product, &dimension| {
        let dimension = u128::try_from(dimension).ok()?;
        product.checked_mul(dimension)
    })
}

fn squeeze_exact_shape(input_shape: &[i64], axes: &[i64]) -> Option<Vec<i64>> {
    let rank = i64::try_from(input_shape.len()).ok()?;
    let mut normalized = HashSet::new();
    for &axis in axes {
        let axis = normalize_axis(axis, rank)?;
        if !normalized.insert(axis) || input_shape[axis] != 1 {
            return None;
        }
    }
    Some(
        input_shape
            .iter()
            .enumerate()
            .filter_map(|(axis, &extent)| (!normalized.contains(&axis)).then_some(extent))
            .collect(),
    )
}

fn unsqueeze_exact_shape(input_shape: &[i64], axes: &[i64]) -> Option<Vec<i64>> {
    let output_rank = input_shape.len().checked_add(axes.len())?;
    let output_rank_i64 = i64::try_from(output_rank).ok()?;
    let mut normalized = HashSet::new();
    for &axis in axes {
        let axis = normalize_axis(axis, output_rank_i64)?;
        if !normalized.insert(axis) {
            return None;
        }
    }
    let mut input = input_shape.iter().copied();
    (0..output_rank)
        .map(|axis| {
            if normalized.contains(&axis) {
                Some(1)
            } else {
                input.next()
            }
        })
        .collect()
}

fn transpose_exact_shape(input_shape: &[i64], permutation: &[i64]) -> Option<Vec<i64>> {
    if permutation.len() != input_shape.len() {
        return None;
    }
    let rank = i64::try_from(input_shape.len()).ok()?;
    let mut seen = HashSet::new();
    permutation
        .iter()
        .map(|&axis| {
            let axis = usize::try_from(axis)
                .ok()
                .filter(|axis| *axis < input_shape.len())?;
            (i64::try_from(axis).ok()? < rank && seen.insert(axis)).then_some(input_shape[axis])
        })
        .collect()
}

fn normalize_axis(axis: i64, rank: i64) -> Option<usize> {
    let axis = if axis < 0 {
        axis.checked_add(rank)?
    } else {
        axis
    };
    usize::try_from(axis)
        .ok()
        .filter(|axis| *axis < rank as usize)
}

fn exact_tensor_shape(tensor: &onnx_proto::TensorProto) -> Option<Vec<i64>> {
    tensor
        .dims
        .iter()
        .copied()
        .all(|dimension| dimension >= 0)
        .then(|| tensor.dims.clone())
}

fn exact_value_info_shape(info: &onnx_proto::ValueInfoProto) -> Option<Vec<i64>> {
    use onnx_proto::tensor_shape_proto::dimension::Value;

    let dimensions = &info
        .r#type
        .as_ref()?
        .tensor_type
        .as_ref()?
        .shape
        .as_ref()?
        .dim;
    dimensions
        .iter()
        .map(|dimension| match dimension.value.as_ref()? {
            Value::DimValue(value) if *value >= 0 => Some(*value),
            Value::DimValue(_) | Value::DimParam(_) => None,
        })
        .collect()
}

fn constant_shape(node: &NodeProto) -> Option<Vec<i64>> {
    if !node.input.is_empty() {
        return None;
    }
    let mut shapes = Vec::new();
    for attr in &node.attribute {
        let shape = match (attr.name.as_str(), attr.r#type) {
            ("value", onnx_proto::attribute_type::TENSOR) => {
                attr.t.as_ref().and_then(exact_tensor_shape)
            }
            ("value_float", onnx_proto::attribute_type::FLOAT)
            | ("value_int", onnx_proto::attribute_type::INT) => Some(Vec::new()),
            ("value_floats", onnx_proto::attribute_type::FLOATS) => {
                i64::try_from(attr.floats.len())
                    .ok()
                    .map(|length| vec![length])
            }
            ("value_ints", onnx_proto::attribute_type::INTS) => i64::try_from(attr.ints.len())
                .ok()
                .map(|length| vec![length]),
            _ => None,
        };
        if let Some(shape) = shape {
            shapes.push(shape);
        }
    }
    match shapes.as_slice() {
        [shape] => Some(shape.clone()),
        _ => None,
    }
}

fn constant_dtype(node: &NodeProto) -> Option<i32> {
    if !node.input.is_empty() {
        return None;
    }
    let mut dtypes = Vec::new();
    for attr in &node.attribute {
        let dtype = match (attr.name.as_str(), attr.r#type) {
            ("value", onnx_proto::attribute_type::TENSOR) => {
                attr.t.as_ref().map(|tensor| tensor.data_type)
            }
            ("value_float", onnx_proto::attribute_type::FLOAT)
            | ("value_floats", onnx_proto::attribute_type::FLOATS) => Some(FLOAT),
            ("value_int", onnx_proto::attribute_type::INT)
            | ("value_ints", onnx_proto::attribute_type::INTS) => Some(7),
            _ => None,
        };
        if let Some(dtype) = dtype {
            dtypes.push(dtype);
        }
    }
    match dtypes.as_slice() {
        [dtype] => Some(*dtype),
        _ => None,
    }
}

fn constant_of_shape_dtype(node: &NodeProto) -> Option<i32> {
    let values: Vec<i32> = node
        .attribute
        .iter()
        .filter(|attr| attr.name == "value" && attr.r#type == onnx_proto::attribute_type::TENSOR)
        .filter_map(|attr| attr.t.as_ref().map(|tensor| tensor.data_type))
        .collect();
    match values.as_slice() {
        [] => Some(FLOAT),
        [dtype] => Some(*dtype),
        _ => None,
    }
}

fn cast_target_dtype(node: &NodeProto) -> Option<i32> {
    if node.input.len() != 1 || node.input[0].is_empty() {
        return None;
    }
    let attrs: Vec<&AttributeProto> = node
        .attribute
        .iter()
        .filter(|attr| attr.name == "to")
        .collect();
    match attrs.as_slice() {
        [attr] if attr.r#type == onnx_proto::attribute_type::INT => {
            i32::try_from(attr.i_value()).ok()
        }
        _ => None,
    }
}

fn raw_output_dtype_attribute(node: &NodeProto) -> Result<Option<i64>> {
    let attrs: Vec<&AttributeProto> = node
        .attribute
        .iter()
        .filter(|attr| attr.name == "output_dtype")
        .collect();
    match attrs.as_slice() {
        [] => Ok(None),
        [attr] if attr.r#type == onnx_proto::attribute_type::INT => Ok(Some(attr.i_value())),
        _ => Err(node_error(
            node,
            "has ambiguous or non-INT output_dtype metadata".to_string(),
        )),
    }
}
