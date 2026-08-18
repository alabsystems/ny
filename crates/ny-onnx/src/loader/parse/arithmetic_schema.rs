// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Raw, versioned schemas for represented standard-domain arithmetic ops.
//!
//! The converter implements NumPy-style broadcasting over `f32` tensors.
//! Legacy ONNX arithmetic used attribute-controlled unidirectional
//! broadcasting, and Min/Max before opset 8 required equal shapes.  Those
//! schemas must not be silently reinterpreted after attributes or constant
//! nodes disappear, so this gate admits only the exactly represented modern
//! subset.

use crate::onnx_proto::NodeProto;
use ny_core::{NyError, Result};

use super::literal_cone::{is_exact_integer_literal_node, LiteralCone, LiteralExemptions};
use super::quantization_preflight::RawDtypeResolver;

const FLOAT: i32 = 1;

/// Validate an arithmetic node if it is one of the operators owned here.
/// Returns `true` when the node was recognized.
pub(super) fn validate_arithmetic_schema(node: &NodeProto, opset: i64) -> Result<bool> {
    match node.op_type.as_str() {
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            // Opsets 1/6 used `broadcast` and `axis` to expand only B into A.
            // NY's binary layers use multidirectional broadcasting instead.
            require_minimum_opset(node, opset, 7)?;
            require_exact_io(node, 2, 1)?;
            require_no_attributes(node, opset)?;
        }
        "Min" | "Max" => {
            // Min/Max 1/6 required all inputs to have the same shape; opset 8
            // introduced multidirectional broadcasting.  ONNX permits one or
            // more inputs, while NY currently emits one binary layer, so only
            // the exact binary subset is represented.
            require_minimum_opset(node, opset, 8)?;
            require_exact_io(node, 2, 1)?;
            require_no_attributes(node, opset)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Require the `f32` arithmetic path represented by NY.  ONNX admits integer,
/// f16, bfloat16, and f64 variants at various opsets; converting those tensors
/// into `WeightStore` f32 values would change authored arithmetic semantics.
pub(super) fn validate_arithmetic_float32_data_path(
    node: &NodeProto,
    raw_dtypes: &mut RawDtypeResolver<'_>,
    literals: &LiteralCone,
    exemptions: &mut LiteralExemptions,
) -> Result<()> {
    for value in node
        .input
        .iter()
        .chain(node.output.iter())
        .filter(|value| !value.is_empty())
    {
        let dtype = raw_dtypes.resolve(value)?.ok_or_else(|| {
            node_error(
                node,
                format!("cannot authenticate FLOAT32 dtype for tensor '{value}'"),
            )
        })?;
        if dtype != FLOAT {
            let refusal = NyError::UnsupportedOp(format!(
                "standard ONNX {} node '{}' tensor '{}' has ONNX dtype {dtype}; ny only represents the FLOAT32 arithmetic data path",
                node.op_type, node.name, value
            ));
            // A node whose whole cone is authored constants of exact integer
            // type is not on the arithmetic data path at all: it is shape/index
            // bookkeeping the loader must erase into a literal before any
            // propagation runs. The refusal is deferred, not dropped — see
            // `literal_cone`.
            if !is_exact_integer_literal_node(node, literals, raw_dtypes)? {
                return Err(refusal);
            }
            exemptions.record(node, &refusal);
            return Ok(());
        }
    }
    Ok(())
}

fn require_minimum_opset(node: &NodeProto, opset: i64, minimum: i64) -> Result<()> {
    if opset < minimum {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' uses opset {opset}; ny requires opset {minimum} or newer so the authored broadcasting semantics match its multidirectional arithmetic layers",
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
        return Err(node_error(
            node,
            format!(
                "requires exactly {inputs} non-empty input(s) and {outputs} non-empty output(s) for the represented binary subset; got inputs {:?} and outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    Ok(())
}

fn require_no_attributes(node: &NodeProto, opset: i64) -> Result<()> {
    if let Some(attribute) = node.attribute.first() {
        return Err(node_error(
            node,
            format!(
                "does not define attribute '{}' at opset {opset}",
                attribute.name
            ),
        ));
    }
    Ok(())
}

fn node_error(node: &NodeProto, detail: impl Into<String>) -> NyError {
    NyError::ModelLoad(format!(
        "standard ONNX {} node '{}' {}",
        node.op_type,
        node.name,
        detail.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::{
        attribute_type, AttributeProto, GraphProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };

    fn node(op_type: &str, inputs: &[&str], attributes: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            name: "arithmetic".to_string(),
            op_type: op_type.to_string(),
            input: inputs.iter().map(|value| (*value).to_string()).collect(),
            output: vec!["y".to_string()],
            attribute: attributes,
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

    #[test]
    fn binary_arithmetic_requires_modern_broadcasting_schema() {
        for op_type in ["Add", "Sub", "Mul", "Div", "Pow"] {
            assert!(matches!(
                validate_arithmetic_schema(&node(op_type, &["a", "b"], vec![]), 6),
                Err(NyError::UnsupportedOp(_))
            ));
            validate_arithmetic_schema(&node(op_type, &["a", "b"], vec![]), 7)
                .expect("opset-7 multidirectional binary schema");
            assert!(validate_arithmetic_schema(
                &node(op_type, &["a", "b"], vec![int_attribute("broadcast", 1)]),
                7,
            )
            .is_err());
        }
    }

    #[test]
    fn min_max_admit_only_modern_binary_subset() {
        for op_type in ["Min", "Max"] {
            assert!(matches!(
                validate_arithmetic_schema(&node(op_type, &["a", "b"], vec![]), 6),
                Err(NyError::UnsupportedOp(_))
            ));
            validate_arithmetic_schema(&node(op_type, &["a", "b"], vec![]), 8)
                .expect("binary modern Min/Max subset");
            assert!(validate_arithmetic_schema(&node(op_type, &["a"], vec![]), 13).is_err());
            assert!(
                validate_arithmetic_schema(&node(op_type, &["a", "b", "c"], vec![]), 13).is_err()
            );
        }
    }

    #[test]
    fn malformed_binary_signature_and_modern_attributes_fail_closed() {
        assert!(validate_arithmetic_schema(&node("Add", &["a", ""], vec![]), 14).is_err());
        assert!(validate_arithmetic_schema(
            &node("Div", &["a", "b"], vec![int_attribute("axis", 0)]),
            14,
        )
        .is_err());
    }

    #[test]
    fn arithmetic_data_path_requires_float32_for_every_operand() {
        const INT64: i32 = 7;
        for (rhs_dtype, should_pass) in [(FLOAT, true), (INT64, false)] {
            let arithmetic = node("Pow", &["base", "exponent"], vec![]);
            let graph = GraphProto {
                node: vec![arithmetic],
                input: vec![value_info("base", FLOAT), value_info("exponent", rhs_dtype)],
                output: vec![value_info("y", FLOAT)],
                ..Default::default()
            };
            let arithmetic = &graph.node[0];
            let mut raw_dtypes = RawDtypeResolver::new(&graph);
            let literals = LiteralCone::new(&graph);
            let mut exemptions = LiteralExemptions::default();
            let accepted = validate_arithmetic_float32_data_path(
                arithmetic,
                &mut raw_dtypes,
                &literals,
                &mut exemptions,
            )
            .is_ok()
                // A deferred literal-cone refusal is not an acceptance: this
                // unit test never folds, so require it to re-raise.
                && exemptions
                    .require_all_folded(&crate::WeightStore::new())
                    .is_ok();
            assert_eq!(accepted, should_pass);
        }
    }
}
