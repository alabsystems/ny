// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Raw schemas for represented data-transform operators.
//!
//! Pad, Resize, Upsample, and ScatterND admit many ONNX element types while
//! ny's propagated implementations are specifically FLOAT32.  Their integer
//! controls also have exact schema types.  Authenticate both before constant
//! folding can normalize integer payloads into `WeightStore`'s f32 view.

use crate::loader::const_fold::is_standard_onnx_domain;
use crate::onnx_proto::{self, GraphProto, NodeProto};
use ny_core::{NyError, Result};
use std::collections::HashMap;

use super::quantization_preflight::RawDtypeResolver;

const FLOAT: i32 = 1;
const INT32: i32 = 6;
const INT64: i32 = 7;

pub(super) fn validate_transform_schemas(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
) -> Result<()> {
    let standard_opset = opset_imports
        .get("")
        .copied()
        .or_else(|| opset_imports.get("ai.onnx").copied());
    let mut dtypes = RawDtypeResolver::new(graph);

    for node in graph
        .node
        .iter()
        .filter(|node| is_standard_onnx_domain(&node.domain))
    {
        if !matches!(
            node.op_type.as_str(),
            "Pad" | "Resize" | "Upsample" | "ScatterND"
        ) {
            continue;
        }
        let opset = standard_opset
            .ok_or_else(|| node_error(node, "has no standard-domain opset import"))?;
        match node.op_type.as_str() {
            "Pad" => validate_pad(node, opset, &mut dtypes)?,
            "Resize" => validate_resize(node, opset, &mut dtypes)?,
            "Upsample" => validate_upsample(node, opset, &mut dtypes)?,
            "ScatterND" => validate_scatter_nd(node, opset, &mut dtypes)?,
            _ => unreachable!("operator filtered above"),
        }
    }
    Ok(())
}

fn validate_pad(node: &NodeProto, opset: i64, dtypes: &mut RawDtypeResolver<'_>) -> Result<()> {
    // Pad-1 used a different experimental attribute schema.  Pad-2 is the
    // earliest attribute form represented by ny.
    require_minimum_opset(node, opset, 2)?;
    require_single_output(node)?;
    if opset < 11 {
        if node.input.len() != 1 || node.input[0].is_empty() {
            return Err(signature_error(node, opset));
        }
        let mut saw_pads = false;
        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "pads" if attribute.r#type == onnx_proto::attribute_type::INTS => {
                    saw_pads = true;
                }
                "mode" if attribute.r#type == onnx_proto::attribute_type::STRING => {
                    require_pad_mode(node, attribute.s_value())?;
                }
                "value" if attribute.r#type == onnx_proto::attribute_type::FLOAT => {}
                _ => return Err(unknown_or_malformed_attribute(node, attribute, opset)),
            }
        }
        if !saw_pads {
            return Err(node_error(
                node,
                "is missing required INTS attribute 'pads'",
            ));
        }
    } else {
        let maximum = if opset >= 18 { 4 } else { 3 };
        if !(2..=maximum).contains(&node.input.len())
            || node.input[0..2].iter().any(String::is_empty)
        {
            return Err(signature_error(node, opset));
        }
        for attribute in &node.attribute {
            if attribute.name == "mode" && attribute.r#type == onnx_proto::attribute_type::STRING {
                require_pad_mode(node, attribute.s_value())?;
            } else {
                return Err(unknown_or_malformed_attribute(node, attribute, opset));
            }
        }
        require_dtype(node, dtypes, &node.input[1], "pads", &[INT64])?;
        if let Some(value) = node.input.get(2).filter(|value| !value.is_empty()) {
            require_dtype(node, dtypes, value, "constant_value", &[FLOAT])?;
        }
        if let Some(value) = node.input.get(3).filter(|value| !value.is_empty()) {
            require_dtype(node, dtypes, value, "axes", &[INT32, INT64])?;
        }
    }
    require_dtype(node, dtypes, &node.input[0], "data", &[FLOAT])
}

fn require_pad_mode(node: &NodeProto, value: &[u8]) -> Result<()> {
    if matches!(value, b"constant" | b"reflect") {
        Ok(())
    } else {
        Err(NyError::UnsupportedOp(format!(
            "standard ONNX Pad node '{}' uses mode '{}'; ny represents only constant and reflect padding",
            node.name,
            String::from_utf8_lossy(value)
        )))
    }
}

fn validate_resize(node: &NodeProto, opset: i64, dtypes: &mut RawDtypeResolver<'_>) -> Result<()> {
    require_minimum_opset(node, opset, 10)?;
    require_single_output(node)?;
    if opset == 10 {
        if node.input.len() != 2 || node.input.iter().any(String::is_empty) {
            return Err(signature_error(node, opset));
        }
        require_dtype(node, dtypes, &node.input[0], "X", &[FLOAT])?;
        require_dtype(node, dtypes, &node.input[1], "scales", &[FLOAT])?;
    } else {
        if !(3..=4).contains(&node.input.len()) || node.input[0].is_empty() {
            return Err(signature_error(node, opset));
        }
        let scales = node.input.get(2).filter(|value| !value.is_empty());
        let sizes = node.input.get(3).filter(|value| !value.is_empty());
        if scales.is_some() == sizes.is_some() {
            return Err(node_error(
                node,
                "requires exactly one non-empty scales or sizes input",
            ));
        }
        require_dtype(node, dtypes, &node.input[0], "X", &[FLOAT])?;
        if let Some(roi) = node.input.get(1).filter(|value| !value.is_empty()) {
            require_dtype(node, dtypes, roi, "roi", &[FLOAT])?;
        }
        if let Some(scales) = scales {
            require_dtype(node, dtypes, scales, "scales", &[FLOAT])?;
        }
        if let Some(sizes) = sizes {
            require_dtype(node, dtypes, sizes, "sizes", &[INT64])?;
        }
    }

    for attribute in &node.attribute {
        let valid = match attribute.name.as_str() {
            "mode" => attribute.r#type == onnx_proto::attribute_type::STRING,
            "coordinate_transformation_mode" | "nearest_mode" if opset >= 11 => {
                attribute.r#type == onnx_proto::attribute_type::STRING
            }
            "cubic_coeff_a" | "extrapolation_value" if opset >= 11 => {
                attribute.r#type == onnx_proto::attribute_type::FLOAT
            }
            "exclude_outside" if opset >= 11 => {
                attribute.r#type == onnx_proto::attribute_type::INT
                    && matches!(attribute.i_value(), 0 | 1)
            }
            "antialias" if opset >= 18 => {
                attribute.r#type == onnx_proto::attribute_type::INT
                    && matches!(attribute.i_value(), 0 | 1)
            }
            "axes" if opset >= 18 => attribute.r#type == onnx_proto::attribute_type::INTS,
            "keep_aspect_ratio_policy" if opset >= 18 => {
                attribute.r#type == onnx_proto::attribute_type::STRING
            }
            _ => false,
        };
        if !valid {
            return Err(unknown_or_malformed_attribute(node, attribute, opset));
        }
    }
    Ok(())
}

fn validate_upsample(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    // Upsample-9 moved scales from an attribute to the tensor input consumed
    // by ny.  Upsample-7 is therefore a different representation.
    require_minimum_opset(node, opset, 9)?;
    require_exact_io(node, 2, 1)?;
    for attribute in &node.attribute {
        if attribute.name != "mode" || attribute.r#type != onnx_proto::attribute_type::STRING {
            return Err(unknown_or_malformed_attribute(node, attribute, opset));
        }
    }
    require_dtype(node, dtypes, &node.input[0], "X", &[FLOAT])?;
    require_dtype(node, dtypes, &node.input[1], "scales", &[FLOAT])
}

fn validate_scatter_nd(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 11)?;
    require_exact_io(node, 3, 1)?;
    for attribute in &node.attribute {
        if opset < 16
            || attribute.name != "reduction"
            || attribute.r#type != onnx_proto::attribute_type::STRING
        {
            return Err(unknown_or_malformed_attribute(node, attribute, opset));
        }
        if attribute.s_value() != b"none" {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX ScatterND node '{}' uses reduction '{}'; ny represents only overwrite reduction='none'",
                node.name,
                String::from_utf8_lossy(attribute.s_value())
            )));
        }
    }
    require_dtype(node, dtypes, &node.input[0], "data", &[FLOAT])?;
    require_dtype(node, dtypes, &node.input[1], "indices", &[INT64])?;
    require_dtype(node, dtypes, &node.input[2], "updates", &[FLOAT])
}

fn require_dtype(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    value: &str,
    role: &str,
    allowed: &[i32],
) -> Result<()> {
    let dtype = dtypes.resolve(value)?.ok_or_else(|| {
        node_error(
            node,
            &format!("cannot authenticate dtype of {role} tensor '{value}'"),
        )
    })?;
    if !allowed.contains(&dtype) {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' uses {role} tensor '{}' with dtype {dtype}; represented dtype(s) are {:?}",
            node.op_type, node.name, value, allowed
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

fn require_exact_io(node: &NodeProto, inputs: usize, outputs: usize) -> Result<()> {
    if node.input.len() != inputs
        || node.input.iter().any(String::is_empty)
        || node.output.len() != outputs
        || node.output.iter().any(String::is_empty)
    {
        return Err(signature_error(node, 0));
    }
    Ok(())
}

fn require_single_output(node: &NodeProto) -> Result<()> {
    if node.output.len() != 1 || node.output[0].is_empty() {
        return Err(node_error(node, "requires exactly one non-empty output"));
    }
    Ok(())
}

fn signature_error(node: &NodeProto, opset: i64) -> NyError {
    node_error(
        node,
        &format!(
            "has unsupported opset-{opset} signature: inputs {:?}, outputs {:?}",
            node.input, node.output
        ),
    )
}

fn unknown_or_malformed_attribute(
    node: &NodeProto,
    attribute: &onnx_proto::AttributeProto,
    opset: i64,
) -> NyError {
    node_error(
        node,
        &format!(
            "has unsupported or malformed '{}' attribute at opset {opset}",
            attribute.name
        ),
    )
}

fn node_error(node: &NodeProto, detail: &str) -> NyError {
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

    fn initializer(name: &str, dtype: i32) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            dims: vec![1],
            data_type: dtype,
            ..Default::default()
        }
    }

    fn node(op_type: &str, inputs: &[&str]) -> NodeProto {
        NodeProto {
            name: "transform".to_string(),
            op_type: op_type.to_string(),
            input: inputs.iter().map(|value| (*value).to_string()).collect(),
            output: vec!["y".to_string()],
            ..Default::default()
        }
    }

    fn opsets(version: i64) -> HashMap<String, i64> {
        HashMap::from([(String::new(), version)])
    }

    #[test]
    fn resize_requires_exact_control_dtypes() {
        for (scale_dtype, should_pass) in [(FLOAT, true), (INT64, false)] {
            let graph = GraphProto {
                node: vec![node("Resize", &["x", "", "scales"])],
                input: vec![value_info("x", FLOAT)],
                initializer: vec![initializer("scales", scale_dtype)],
                ..Default::default()
            };
            assert_eq!(
                validate_transform_schemas(&graph, &opsets(13)).is_ok(),
                should_pass
            );
        }
    }

    #[test]
    fn resize_sizes_must_be_int64() {
        let graph = GraphProto {
            node: vec![node("Resize", &["x", "", "", "sizes"])],
            input: vec![value_info("x", FLOAT)],
            initializer: vec![initializer("sizes", INT32)],
            ..Default::default()
        };
        assert!(validate_transform_schemas(&graph, &opsets(18)).is_err());
    }

    #[test]
    fn scatter_nd_requires_float_data_and_int64_indices() {
        for (data_dtype, index_dtype, should_pass) in [
            (FLOAT, INT64, true),
            (INT64, INT64, false),
            (FLOAT, INT32, false),
        ] {
            let graph = GraphProto {
                node: vec![node("ScatterND", &["data", "indices", "updates"])],
                input: vec![
                    value_info("data", data_dtype),
                    value_info("updates", data_dtype),
                ],
                initializer: vec![initializer("indices", index_dtype)],
                ..Default::default()
            };
            assert_eq!(
                validate_transform_schemas(&graph, &opsets(18)).is_ok(),
                should_pass
            );
        }
    }

    #[test]
    fn scatter_nd_reduction_is_versioned_and_fail_closed() {
        let mut scatter = node("ScatterND", &["data", "indices", "updates"]);
        scatter.attribute.push(AttributeProto {
            name: "reduction".to_string(),
            r#type: attribute_type::STRING,
            s: Some(b"add".to_vec()),
            ..Default::default()
        });
        let graph = GraphProto {
            node: vec![scatter],
            input: vec![value_info("data", FLOAT), value_info("updates", FLOAT)],
            initializer: vec![initializer("indices", INT64)],
            ..Default::default()
        };
        assert!(validate_transform_schemas(&graph, &opsets(15)).is_err());
        assert!(validate_transform_schemas(&graph, &opsets(18)).is_err());
    }

    #[test]
    fn scatter_nd_output_preserves_authenticated_float_dtype() {
        let mut scatter = node("ScatterND", &["data", "indices", "updates"]);
        scatter.output = vec!["scattered".to_string()];
        let graph = GraphProto {
            node: vec![scatter],
            input: vec![value_info("data", FLOAT), value_info("updates", FLOAT)],
            initializer: vec![initializer("indices", INT64)],
            ..Default::default()
        };
        let mut resolver = RawDtypeResolver::new(&graph);
        assert_eq!(resolver.resolve("scattered").unwrap(), Some(FLOAT));
    }

    #[test]
    fn pad_data_must_be_float32() {
        let graph = GraphProto {
            node: vec![node("Pad", &["x", "pads"])],
            input: vec![value_info("x", INT64)],
            initializer: vec![initializer("pads", INT64)],
            ..Default::default()
        };
        assert!(validate_transform_schemas(&graph, &opsets(18)).is_err());
    }
}
