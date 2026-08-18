// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Raw schema and dtype authentication for affine and convolution operators.
//!
//! `WeightStore` deliberately exposes one `f32` view of several authored
//! tensor dtypes.  That representation is useful for exact integer shape
//! folding, but it must never authorize integer MatMul/Gemm/Conv arithmetic.
//! Validate these operators before folding can erase either their nodes or
//! their raw tensor-type provenance.

use crate::loader::const_fold::is_standard_onnx_domain;
use crate::onnx_proto::{self, AttributeProto, GraphProto, NodeProto};
use ny_core::{NyError, Result};
use std::collections::{HashMap, HashSet};

use super::quantization_preflight::RawDtypeResolver;

const FLOAT: i32 = 1;

pub(super) fn validate_linear_convolution_schemas(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
) -> Result<()> {
    let standard_opset = opset_imports
        .get("")
        .copied()
        .or_else(|| opset_imports.get("ai.onnx").copied());
    let mut raw_dtypes = RawDtypeResolver::new(graph);

    for node in graph
        .node
        .iter()
        .filter(|node| is_standard_onnx_domain(&node.domain))
    {
        let Some(kind) = represented_kind(&node.op_type) else {
            continue;
        };
        let opset = standard_opset
            .ok_or_else(|| node_error(node, "has no standard-domain opset import".to_string()))?;
        match kind {
            RepresentedKind::MatMul => validate_matmul_schema(node, opset)?,
            RepresentedKind::Gemm => validate_gemm_schema(node, opset)?,
            RepresentedKind::Convolution => validate_convolution_schema(graph, node, opset)?,
        }
        authenticate_float32_data_path(node, &mut raw_dtypes)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum RepresentedKind {
    MatMul,
    Gemm,
    Convolution,
}

fn represented_kind(op_type: &str) -> Option<RepresentedKind> {
    match op_type {
        "MatMul" => Some(RepresentedKind::MatMul),
        "Gemm" => Some(RepresentedKind::Gemm),
        "Conv" | "ConvTranspose" => Some(RepresentedKind::Convolution),
        _ => None,
    }
}

fn validate_matmul_schema(node: &NodeProto, opset: i64) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    require_io(node, 2, 1)?;
    if let Some(attribute) = node.attribute.first() {
        return Err(node_error(
            node,
            format!(
                "does not define attribute '{}' in the standard ONNX domain; ny's transpose_b/scale attributes are internal-only",
                attribute.name
            ),
        ));
    }
    Ok(())
}

fn validate_gemm_schema(node: &NodeProto, opset: i64) -> Result<()> {
    // Opsets 1 and 6 used the legacy `broadcast` attribute for C.  The modern
    // unidirectional-broadcast schema represented by ny begins at Gemm-7.
    require_minimum_opset(node, opset, 7)?;
    let valid_inputs = if opset >= 11 {
        matches!(node.input.len(), 2 | 3) && node.input[0..2].iter().all(|value| !value.is_empty())
    } else {
        node.input.len() == 3 && node.input.iter().all(|value| !value.is_empty())
    };
    if !valid_inputs || node.output.len() != 1 || node.output[0].is_empty() {
        return Err(node_error(
            node,
            format!(
                "requires {} and exactly one non-empty output at opset {opset}; got inputs {:?} and outputs {:?}",
                if opset >= 11 {
                    "A, B, and optional C"
                } else {
                    "non-empty A, B, and C"
                },
                node.input, node.output
            ),
        ));
    }

    for attribute in &node.attribute {
        let valid = match attribute.name.as_str() {
            "alpha" | "beta" => attribute.r#type == onnx_proto::attribute_type::FLOAT,
            "transA" | "transB" => {
                attribute.r#type == onnx_proto::attribute_type::INT
                    && matches!(attribute.i_value(), 0 | 1)
            }
            _ => false,
        };
        if !valid {
            return Err(node_error(
                node,
                format!(
                    "has unsupported or malformed '{}' attribute at opset {opset}",
                    attribute.name
                ),
            ));
        }
    }
    Ok(())
}

fn validate_convolution_schema(graph: &GraphProto, node: &NodeProto, opset: i64) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    if !matches!(node.input.len(), 2 | 3)
        || node.input[0..2].iter().any(String::is_empty)
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return Err(node_error(
            node,
            format!(
                "requires non-empty X and W, an optional B (which may be an empty placeholder), and exactly one non-empty output; got inputs {:?} and outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    // Conv and ConvTranspose have schema revisions 1, 11, and 22. Their
    // attribute sets are unchanged, but resolve the active revision explicitly
    // so future version-specific changes cannot silently inherit this allowlist.
    let schema_revision = if opset >= 22 {
        22
    } else if opset >= 11 {
        11
    } else {
        1
    };
    let transpose = node.op_type == "ConvTranspose";
    let mut names = HashSet::new();
    let mut kernel_shape = None;
    let mut dilations = None;
    let mut pads = None;
    let mut strides = None;
    let mut output_padding = None;
    let mut output_shape = None;
    let mut group = 1_i64;

    for attribute in &node.attribute {
        if !names.insert(attribute.name.as_str()) {
            return Err(node_error(
                node,
                format!("has duplicate '{}' attributes", attribute.name),
            ));
        }
        match attribute.name.as_str() {
            "auto_pad" => {
                require_attribute_type(
                    node,
                    attribute,
                    onnx_proto::attribute_type::STRING,
                    "STRING",
                )?;
                match attribute.s_value() {
                    b"NOTSET" => {}
                    b"SAME_UPPER" | b"SAME_LOWER" | b"VALID" => {
                        return Err(NyError::UnsupportedOp(format!(
                            "standard ONNX {} node '{}' uses auto_pad={}; ny represents only explicit symmetric pads (auto_pad=NOTSET)",
                            node.op_type,
                            node.name,
                            String::from_utf8_lossy(attribute.s_value())
                        )));
                    }
                    value => {
                        return Err(node_error(
                            node,
                            format!(
                                "has invalid auto_pad value {:?} for {}-{}",
                                String::from_utf8_lossy(value),
                                node.op_type,
                                schema_revision
                            ),
                        ));
                    }
                }
            }
            "dilations" => {
                let values = require_ints_attribute(node, attribute)?;
                require_nonempty_positive_values(node, attribute, values)?;
                dilations = Some(values);
            }
            "group" => {
                require_attribute_type(node, attribute, onnx_proto::attribute_type::INT, "INT")?;
                group = attribute.i_value();
                if group <= 0 {
                    return Err(node_error(
                        node,
                        format!("requires group >= 1, got {group}"),
                    ));
                }
            }
            "kernel_shape" => {
                let values = require_ints_attribute(node, attribute)?;
                require_nonempty_positive_values(node, attribute, values)?;
                kernel_shape = Some(values);
            }
            "pads" => {
                let values = require_ints_attribute(node, attribute)?;
                require_nonempty_nonnegative_values(node, attribute, values)?;
                pads = Some(values);
            }
            "strides" => {
                let values = require_ints_attribute(node, attribute)?;
                require_nonempty_positive_values(node, attribute, values)?;
                strides = Some(values);
            }
            "output_padding" if transpose => {
                let values = require_ints_attribute(node, attribute)?;
                require_nonempty_nonnegative_values(node, attribute, values)?;
                output_padding = Some(values);
            }
            "output_shape" if transpose => {
                let values = require_ints_attribute(node, attribute)?;
                require_nonempty_nonnegative_values(node, attribute, values)?;
                output_shape = Some(values);
            }
            _ => {
                return Err(node_error(
                    node,
                    format!(
                        "does not define attribute '{}' in the active {}-{} schema",
                        attribute.name, node.op_type, schema_revision
                    ),
                ));
            }
        }
    }

    let spatial_rank = authenticate_represented_conv_rank(
        graph,
        node,
        kernel_shape,
        dilations,
        pads,
        strides,
        output_padding,
        output_shape,
    )?;

    if let (Some(expected), Some(authored)) =
        (kernel_shape, authored_tensor_dims(graph, &node.input[1])?)
    {
        if authored.len() >= 2 {
            let actual = &authored[2..];
            if actual.iter().all(Option::is_some)
                && !actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| *actual == Some(*expected))
            {
                return Err(node_error(
                    node,
                    format!(
                        "kernel_shape {:?} does not match W spatial dimensions {:?}",
                        expected, actual
                    ),
                ));
            }
        }
    }

    if let (Some(rank), Some(values)) = (spatial_rank, pads) {
        if values[..rank] != values[rank..] {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX {} node '{}' uses asymmetric pads {:?}; ny's convolution layers represent only equal start/end padding",
                node.op_type, node.name, values
            )));
        }
    }

    if transpose {
        if let Some(values) = output_shape {
            // ONNX output_shape contains spatial dimensions only and causes
            // pads to be auto-generated. Those start/end pads can differ by
            // one, while ny's layer stores one symmetric pad per axis.
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX ConvTranspose node '{}' specifies spatial-only output_shape {:?}; its potentially asymmetric auto-generated pads are not represented",
                node.name, values
            )));
        }
        if spatial_rank == Some(2) && group != 1 {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX ConvTranspose node '{}' uses group={group} in 2D; grouped ConvTranspose2d is not represented",
                node.name
            )));
        }
        if let (Some(rank), Some(values)) = (spatial_rank, output_padding) {
            if rank == 1 && values.iter().any(|value| *value != 0) {
                return Err(NyError::UnsupportedOp(format!(
                    "standard ONNX ConvTranspose node '{}' uses nonzero 1D output_padding {:?}, which ny's ConvTranspose1d layer does not represent",
                    node.name, values
                )));
            }
            let default_strides;
            let represented_strides = match strides {
                Some(values) => values,
                None => {
                    default_strides = vec![1_i64; rank];
                    &default_strides
                }
            };
            if values
                .iter()
                .zip(represented_strides)
                .any(|(padding, stride)| padding >= stride)
            {
                return Err(NyError::UnsupportedOp(format!(
                    "standard ONNX ConvTranspose node '{}' uses output_padding {:?} with strides {:?}; ny's represented 2D subset requires output_padding < stride on every axis",
                    node.name, values, represented_strides
                )));
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn authenticate_represented_conv_rank(
    graph: &GraphProto,
    node: &NodeProto,
    kernel_shape: Option<&[i64]>,
    dilations: Option<&[i64]>,
    pads: Option<&[i64]>,
    strides: Option<&[i64]>,
    output_padding: Option<&[i64]>,
    output_shape: Option<&[i64]>,
) -> Result<Option<usize>> {
    let mut candidates = Vec::new();
    for (role, value) in [
        ("X", node.input[0].as_str()),
        ("W", node.input[1].as_str()),
        ("Y", node.output[0].as_str()),
    ] {
        if let Some(rank) = authored_tensor_rank(graph, value)? {
            let spatial = rank.checked_sub(2).ok_or_else(|| {
                node_error(
                    node,
                    format!("{role} tensor '{value}' has invalid rank {rank}"),
                )
            })?;
            candidates.push((format!("{role} rank"), spatial));
        }
    }
    for (name, values) in [
        ("kernel_shape", kernel_shape),
        ("dilations", dilations),
        ("strides", strides),
        ("output_padding", output_padding),
        ("output_shape", output_shape),
    ] {
        if let Some(values) = values {
            candidates.push((name.to_string(), values.len()));
        }
    }
    if let Some(values) = pads {
        if values.len() % 2 != 0 {
            return Err(node_error(
                node,
                format!(
                    "pads must contain begin/end values for every spatial axis, got {:?}",
                    values
                ),
            ));
        }
        candidates.push(("pads".to_string(), values.len() / 2));
    }

    let Some((first_source, rank)) = candidates.first() else {
        return Ok(None);
    };
    if !matches!(*rank, 1 | 2) {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' has spatial rank {} authenticated by {}; ny represents only 1D and 2D convolution",
            node.op_type, node.name, rank, first_source
        )));
    }
    if let Some((source, conflicting)) = candidates
        .iter()
        .skip(1)
        .find(|(_, candidate)| candidate != rank)
    {
        return Err(node_error(
            node,
            format!(
                "has inconsistent spatial rank: {} implies {}, but {} implies {}",
                first_source, rank, source, conflicting
            ),
        ));
    }
    Ok(Some(*rank))
}

fn authored_tensor_rank(graph: &GraphProto, value: &str) -> Result<Option<usize>> {
    Ok(authored_tensor_dims(graph, value)?.map(|dimensions| dimensions.len()))
}

/// Recover authored dimensions without inventing values for symbolic axes.
/// Initializer and Constant tensors provide exact dimensions; ValueInfo still
/// authenticates rank even when individual extents are symbolic.
fn authored_tensor_dims(graph: &GraphProto, value: &str) -> Result<Option<Vec<Option<i64>>>> {
    let mut candidates = Vec::new();
    for initializer in graph
        .initializer
        .iter()
        .filter(|initializer| initializer.name == value)
    {
        candidates.push(
            initializer
                .dims
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>(),
        );
    }
    for info in graph
        .input
        .iter()
        .chain(graph.output.iter())
        .chain(graph.value_info().iter())
        .filter(|info| info.name == value)
    {
        if let Some(shape) = info
            .r#type
            .as_ref()
            .and_then(|ty| ty.tensor_type.as_ref())
            .and_then(|tensor| tensor.shape.as_ref())
        {
            candidates.push(
                shape
                    .dim
                    .iter()
                    .map(|dimension| match dimension.value.as_ref() {
                        Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(value))
                            if *value >= 0 =>
                        {
                            Some(*value)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            );
        }
    }
    for producer in graph.node.iter().filter(|producer| {
        is_standard_onnx_domain(&producer.domain)
            && producer.op_type == "Constant"
            && producer.output.iter().any(|output| output == value)
    }) {
        for tensor in producer.attribute.iter().filter_map(|attribute| {
            (attribute.name == "value" && attribute.r#type == onnx_proto::attribute_type::TENSOR)
                .then_some(attribute.t.as_ref())
                .flatten()
        }) {
            candidates.push(tensor.dims.iter().copied().map(Some).collect::<Vec<_>>());
        }
    }

    let Some(first) = candidates.first().cloned() else {
        return Ok(None);
    };
    if candidates.iter().skip(1).any(|candidate| {
        candidate.len() != first.len()
            || candidate
                .iter()
                .zip(&first)
                .any(|(lhs, rhs)| lhs.is_some() && rhs.is_some() && lhs != rhs)
    }) {
        return Err(NyError::ModelLoad(format!(
            "conflicting authored shapes for ONNX value '{value}': {candidates:?}"
        )));
    }
    let mut merged = first;
    for candidate in candidates.iter().skip(1) {
        for (target, source) in merged.iter_mut().zip(candidate) {
            if target.is_none() {
                *target = *source;
            }
        }
    }
    Ok(Some(merged))
}

fn require_attribute_type(
    node: &NodeProto,
    attribute: &AttributeProto,
    expected: i32,
    expected_name: &str,
) -> Result<()> {
    if attribute.r#type != expected {
        return Err(node_error(
            node,
            format!(
                "attribute '{}' must have type {expected_name}, got AttributeProto type {}",
                attribute.name, attribute.r#type
            ),
        ));
    }
    Ok(())
}

fn require_ints_attribute<'a>(
    node: &NodeProto,
    attribute: &'a AttributeProto,
) -> Result<&'a [i64]> {
    require_attribute_type(node, attribute, onnx_proto::attribute_type::INTS, "INTS")?;
    Ok(&attribute.ints)
}

fn require_nonempty_positive_values(
    node: &NodeProto,
    attribute: &AttributeProto,
    values: &[i64],
) -> Result<()> {
    if values.is_empty() || values.iter().any(|value| *value <= 0) {
        return Err(node_error(
            node,
            format!(
                "attribute '{}' requires a non-empty list of positive integers, got {:?}",
                attribute.name, values
            ),
        ));
    }
    Ok(())
}

fn require_nonempty_nonnegative_values(
    node: &NodeProto,
    attribute: &AttributeProto,
    values: &[i64],
) -> Result<()> {
    if values.is_empty() || values.iter().any(|value| *value < 0) {
        return Err(node_error(
            node,
            format!(
                "attribute '{}' requires a non-empty list of non-negative integers, got {:?}",
                attribute.name, values
            ),
        ));
    }
    Ok(())
}

fn authenticate_float32_data_path(
    node: &NodeProto,
    raw_dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    for value in node.input.iter().filter(|value| !value.is_empty()) {
        let dtype = raw_dtypes.resolve(value)?.ok_or_else(|| {
            node_error(
                node,
                format!("cannot authenticate FLOAT32 dtype for input '{value}'"),
            )
        })?;
        if dtype != FLOAT {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX {} node '{}' input '{}' has ONNX dtype {dtype}; ny only represents the FLOAT32 arithmetic data path",
                node.op_type, node.name, value
            )));
        }
    }
    // All represented operators are type-preserving on their sole output.
    // Resolve the output as well so contradictory graph-output/value-info
    // metadata cannot silently relabel FLOAT32 arithmetic after conversion.
    let output = &node.output[0];
    let dtype = raw_dtypes.resolve(output)?.ok_or_else(|| {
        node_error(
            node,
            format!("cannot authenticate FLOAT32 dtype for output '{output}'"),
        )
    })?;
    if dtype != FLOAT {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' output '{}' has ONNX dtype {dtype}; ny only represents the FLOAT32 arithmetic data path",
            node.op_type, node.name, output
        )));
    }
    Ok(())
}

fn require_minimum_opset(node: &NodeProto, opset: i64, minimum: i64) -> Result<()> {
    if opset < minimum {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' uses opset {opset}; represented semantics require opset {minimum} or newer",
            node.op_type, node.name
        )));
    }
    Ok(())
}

fn require_io(node: &NodeProto, inputs: usize, outputs: usize) -> Result<()> {
    if node.input.len() != inputs
        || node.input.iter().any(String::is_empty)
        || node.output.len() != outputs
        || node.output.iter().any(String::is_empty)
    {
        return Err(node_error(
            node,
            format!(
                "requires exactly {inputs} non-empty input(s) and {outputs} non-empty output(s); got inputs {:?} and outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    Ok(())
}

fn node_error(node: &NodeProto, detail: String) -> NyError {
    NyError::ModelLoad(format!(
        "standard ONNX {} node '{}' {detail}",
        node.op_type, node.name
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::{
        attribute_type, AttributeProto, TensorProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };

    fn value_info(name: &str, dtype: i32) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: dtype,
                    shape: None,
                }),
            }),
        }
    }

    fn node(op_type: &str, inputs: &[&str]) -> NodeProto {
        NodeProto {
            name: "affine".to_string(),
            op_type: op_type.to_string(),
            input: inputs.iter().map(|value| (*value).to_string()).collect(),
            output: vec!["y".to_string()],
            ..Default::default()
        }
    }

    fn opsets(version: i64) -> HashMap<String, i64> {
        HashMap::from([(String::new(), version)])
    }

    fn initializer(name: &str, dtype: i32, dims: &[i64]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            dims: dims.to_vec(),
            data_type: dtype,
            ..Default::default()
        }
    }

    fn float_initializer(name: &str, dtype: i32) -> TensorProto {
        initializer(name, dtype, &[1, 1])
    }

    fn ints_attribute(name: &str, values: &[i64]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: attribute_type::INTS,
            ints: values.to_vec(),
            ..Default::default()
        }
    }

    fn int_attribute(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: attribute_type::INT,
            i: Some(value),
            ..Default::default()
        }
    }

    fn string_attribute(name: &str, value: &[u8]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: attribute_type::STRING,
            s: Some(value.to_vec()),
            ..Default::default()
        }
    }

    fn convolution_graph(
        op_type: &str,
        weight_dims: &[i64],
        attributes: Vec<AttributeProto>,
    ) -> GraphProto {
        let mut convolution = node(op_type, &["x", "w"]);
        convolution.attribute = attributes;
        GraphProto {
            node: vec![convolution],
            initializer: vec![initializer("w", FLOAT, weight_dims)],
            input: vec![value_info("x", FLOAT)],
            output: vec![value_info("y", FLOAT)],
            ..Default::default()
        }
    }

    #[test]
    fn matmul_rejects_internal_only_attributes_in_standard_domain() {
        let mut matmul = node("MatMul", &["a", "b"]);
        matmul.attribute.push(AttributeProto {
            name: "transpose_b".to_string(),
            r#type: attribute_type::INT,
            i: Some(1),
            ..Default::default()
        });
        assert!(validate_matmul_schema(&matmul, 13).is_err());
    }

    #[test]
    fn gemm_optional_c_is_versioned() {
        let gemm = node("Gemm", &["a", "b"]);
        assert!(validate_gemm_schema(&gemm, 9).is_err());
        validate_gemm_schema(&gemm, 11).expect("C became optional in Gemm-11");
        assert!(matches!(
            validate_gemm_schema(&gemm, 6),
            Err(NyError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn gemm_empty_optional_c_preserves_authenticated_output_dtype() {
        let graph = GraphProto {
            node: vec![node("Gemm", &["a", "b", ""])],
            input: vec![value_info("a", FLOAT)],
            initializer: vec![float_initializer("b", FLOAT)],
            ..Default::default()
        };
        let mut resolver = RawDtypeResolver::new(&graph);
        assert_eq!(resolver.resolve("y").unwrap(), Some(FLOAT));
    }

    #[test]
    fn integer_matmul_weight_is_rejected_before_folding() {
        const INT64: i32 = 7;
        let graph = GraphProto {
            node: vec![node("MatMul", &["x", "w"])],
            initializer: vec![float_initializer("w", INT64)],
            input: vec![value_info("x", FLOAT)],
            output: vec![value_info("y", FLOAT)],
            ..Default::default()
        };
        let error = validate_linear_convolution_schemas(&graph, &opsets(13))
            .expect_err("integer MatMul must not become f32 arithmetic");
        assert!(matches!(error, NyError::UnsupportedOp(_)), "{error}");
    }

    #[test]
    fn integer_conv_kernel_is_rejected_before_folding() {
        const INT64: i32 = 7;
        let graph = GraphProto {
            node: vec![node("Conv", &["x", "w"])],
            initializer: vec![initializer("w", INT64, &[2, 3, 3, 3])],
            input: vec![value_info("x", FLOAT)],
            output: vec![value_info("y", FLOAT)],
            ..Default::default()
        };
        let error = validate_linear_convolution_schemas(&graph, &opsets(22))
            .expect_err("integer Conv must not become f32 arithmetic");
        assert!(matches!(error, NyError::UnsupportedOp(_)), "{error}");
    }

    #[test]
    fn contradictory_affine_output_dtype_is_rejected_before_conversion() {
        const INT64: i32 = 7;
        let graph = GraphProto {
            node: vec![node("MatMul", &["x", "w"])],
            initializer: vec![float_initializer("w", FLOAT)],
            input: vec![value_info("x", FLOAT)],
            output: vec![value_info("y", INT64)],
            ..Default::default()
        };
        let error = validate_linear_convolution_schemas(&graph, &opsets(22))
            .expect_err("producer and authored output dtype must agree");
        assert!(
            error.to_string().contains("conflicting raw ONNX dtypes"),
            "{error}"
        );
    }

    #[test]
    fn float32_affine_and_convolution_inputs_are_admitted() {
        let matmul = GraphProto {
            node: vec![node("MatMul", &["x", "w"])],
            initializer: vec![float_initializer("w", FLOAT)],
            input: vec![value_info("x", FLOAT)],
            output: vec![value_info("y", FLOAT)],
            ..Default::default()
        };
        validate_linear_convolution_schemas(&matmul, &opsets(22)).expect("valid MatMul failed");

        for op_type in ["Conv", "ConvTranspose"] {
            let graph = convolution_graph(op_type, &[1, 1, 3, 3], vec![]);
            validate_linear_convolution_schemas(&graph, &opsets(22))
                .unwrap_or_else(|error| panic!("valid {op_type} failed: {error}"));
        }
    }

    #[test]
    fn convolution_admits_an_empty_optional_bias_placeholder() {
        for op_type in ["Conv", "ConvTranspose"] {
            let mut convolution = node(op_type, &["x", "w", ""]);
            convolution.name = "optional_bias".to_string();
            let graph = GraphProto {
                node: vec![convolution],
                initializer: vec![initializer("w", FLOAT, &[1, 1, 3, 3])],
                input: vec![value_info("x", FLOAT)],
                output: vec![value_info("y", FLOAT)],
                ..Default::default()
            };
            validate_linear_convolution_schemas(&graph, &opsets(22))
                .unwrap_or_else(|error| panic!("valid optional {op_type} B failed: {error}"));
        }
    }

    #[test]
    fn convolution_attributes_are_strictly_typed_and_allowlisted() {
        let scalar_strides =
            convolution_graph("Conv", &[2, 3, 3, 3], vec![int_attribute("strides", 2)]);
        let error = validate_linear_convolution_schemas(&scalar_strides, &opsets(22))
            .expect_err("ONNX strides is INTS, never scalar INT");
        assert!(error.to_string().contains("must have type INTS"), "{error}");

        let unknown = convolution_graph(
            "ConvTranspose",
            &[3, 2, 3, 3],
            vec![int_attribute("adj", 1)],
        );
        let error = validate_linear_convolution_schemas(&unknown, &opsets(11))
            .expect_err("unknown standard-domain attributes must fail closed");
        assert!(
            error.to_string().contains("active ConvTranspose-11 schema"),
            "{error}"
        );

        let scalar_output_padding = convolution_graph(
            "ConvTranspose",
            &[3, 2, 3, 3],
            vec![int_attribute("output_padding", 1)],
        );
        let error = validate_linear_convolution_schemas(&scalar_output_padding, &opsets(22))
            .expect_err("ONNX output_padding is INTS, never scalar INT");
        assert!(error.to_string().contains("must have type INTS"), "{error}");

        let malformed_auto_pad = convolution_graph(
            "Conv",
            &[2, 3, 3, 3],
            vec![string_attribute("auto_pad", b"same_upper")],
        );
        assert!(validate_linear_convolution_schemas(&malformed_auto_pad, &opsets(1)).is_err());
    }

    #[test]
    fn convolution_spatial_attributes_match_the_represented_rank_and_kernel() {
        let represented = convolution_graph(
            "Conv",
            &[2, 3, 3, 5],
            vec![
                string_attribute("auto_pad", b"NOTSET"),
                ints_attribute("kernel_shape", &[3, 5]),
                ints_attribute("dilations", &[1, 2]),
                ints_attribute("pads", &[1, 2, 1, 2]),
                ints_attribute("strides", &[2, 1]),
                int_attribute("group", 1),
            ],
        );
        validate_linear_convolution_schemas(&represented, &opsets(1))
            .expect("the represented Conv-1 attribute subset should pass");

        for malformed in [
            convolution_graph("Conv", &[2, 3, 3, 5], vec![ints_attribute("strides", &[2])]),
            convolution_graph(
                "Conv",
                &[2, 3, 3, 5],
                vec![ints_attribute("kernel_shape", &[3, 3])],
            ),
            convolution_graph(
                "Conv",
                &[2, 3, 3, 5],
                vec![ints_attribute("dilations", &[1, 0])],
            ),
            convolution_graph(
                "Conv",
                &[2, 3, 3, 5],
                vec![ints_attribute("pads", &[1, 2, 0, 2])],
            ),
            convolution_graph("Conv", &[2, 3, 3, 5], vec![int_attribute("group", 0)]),
        ] {
            assert!(
                validate_linear_convolution_schemas(&malformed, &opsets(22)).is_err(),
                "malformed or unrepresented convolution attributes must fail closed: {:?}",
                malformed.node[0].attribute
            );
        }
    }

    #[test]
    fn conv_transpose_output_controls_fail_closed_outside_exact_subset() {
        let represented = convolution_graph(
            "ConvTranspose",
            &[3, 2, 3, 3],
            vec![
                ints_attribute("strides", &[2, 2]),
                ints_attribute("output_padding", &[1, 1]),
            ],
        );
        validate_linear_convolution_schemas(&represented, &opsets(22))
            .expect("represented ConvTranspose2d output_padding should pass");

        let output_shape = convolution_graph(
            "ConvTranspose",
            &[3, 2, 3, 3],
            vec![ints_attribute("output_shape", &[10, 10])],
        );
        let error = validate_linear_convolution_schemas(&output_shape, &opsets(22))
            .expect_err("output_shape auto-generated padding is not represented");
        assert!(
            error.to_string().contains("spatial-only output_shape")
                && error.to_string().contains("potentially asymmetric"),
            "{error}"
        );

        for unsupported in [
            convolution_graph(
                "ConvTranspose",
                &[3, 2, 3],
                vec![
                    ints_attribute("strides", &[2]),
                    ints_attribute("output_padding", &[1]),
                ],
            ),
            convolution_graph(
                "ConvTranspose",
                &[3, 2, 3, 3],
                vec![
                    ints_attribute("strides", &[2, 2]),
                    ints_attribute("output_padding", &[2, 0]),
                ],
            ),
            convolution_graph(
                "ConvTranspose",
                &[3, 2, 3, 3],
                vec![int_attribute("group", 2)],
            ),
            convolution_graph(
                "ConvTranspose",
                &[3, 2, 3, 3],
                vec![ints_attribute("output_shape", &[1, 2, 10, 10])],
            ),
        ] {
            assert!(
                validate_linear_convolution_schemas(&unsupported, &opsets(22)).is_err(),
                "unrepresented ConvTranspose controls must fail closed: {:?}",
                unsupported.node[0].attribute
            );
        }
    }
}
