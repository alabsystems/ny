// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Raw, versioned schemas for standard-domain elementwise activations.
//!
//! These checks deliberately run before constant folding.  Otherwise a
//! malformed all-constant node can disappear before its opset, arity, or
//! attribute spelling is authenticated.

use crate::onnx_proto::{attribute_type, AttributeProto, NodeProto};
use ny_core::{NyError, Result};

use super::quantization_preflight::RawDtypeResolver;

const FLOAT: i32 = 1;

#[derive(Clone, Copy)]
enum AttributeRule {
    FiniteFloat,
    Int,
    Ints,
    StringEnum(&'static [&'static [u8]]),
}

const NONE_OR_TANH: &[&[u8]] = &[b"none", b"tanh"];
const NO_ATTRIBUTES: &[(&str, AttributeRule)] = &[];
const ALPHA: &[(&str, AttributeRule)] = &[("alpha", AttributeRule::FiniteFloat)];
const ALPHA_GAMMA: &[(&str, AttributeRule)] = &[
    ("alpha", AttributeRule::FiniteFloat),
    ("gamma", AttributeRule::FiniteFloat),
];
const ALPHA_BETA: &[(&str, AttributeRule)] = &[
    ("alpha", AttributeRule::FiniteFloat),
    ("beta", AttributeRule::FiniteFloat),
];
const SHRINK_ATTRIBUTES: &[(&str, AttributeRule)] = &[
    ("bias", AttributeRule::FiniteFloat),
    ("lambd", AttributeRule::FiniteFloat),
];
const CLIP_ATTRIBUTES: &[(&str, AttributeRule)] = &[
    ("min", AttributeRule::FiniteFloat),
    ("max", AttributeRule::FiniteFloat),
];
const GELU_ATTRIBUTES: &[(&str, AttributeRule)] =
    &[("approximate", AttributeRule::StringEnum(NONE_OR_TANH))];
const SWISH_ATTRIBUTES: &[(&str, AttributeRule)] = &[("alpha", AttributeRule::FiniteFloat)];
const AXIS: &[(&str, AttributeRule)] = &[("axis", AttributeRule::Int)];
const CONSUMED_INPUTS: &[(&str, AttributeRule)] = &[("consumed_inputs", AttributeRule::Ints)];
const ALPHA_LEGACY: &[(&str, AttributeRule)] = &[
    ("alpha", AttributeRule::FiniteFloat),
    ("consumed_inputs", AttributeRule::Ints),
];
const ALPHA_BETA_LEGACY: &[(&str, AttributeRule)] = &[
    ("alpha", AttributeRule::FiniteFloat),
    ("beta", AttributeRule::FiniteFloat),
    ("consumed_inputs", AttributeRule::Ints),
];
const CLIP_ATTRIBUTES_LEGACY: &[(&str, AttributeRule)] = &[
    ("min", AttributeRule::FiniteFloat),
    ("max", AttributeRule::FiniteFloat),
    ("consumed_inputs", AttributeRule::Ints),
];

/// Validate an activation node if it is one of the operators owned here.
/// Returns `true` when the node was recognized.
pub(super) fn validate_activation_schema(node: &NodeProto, opset: i64) -> Result<bool> {
    match node.op_type.as_str() {
        "Relu" => {
            require_minimum_opset(node, opset, 1)?;
            require_unary(node)?;
            validate_attributes(
                node,
                legacy_consumed_inputs(opset, NO_ATTRIBUTES, CONSUMED_INPUTS),
            )?;
        }
        "LeakyRelu" | "Elu" => {
            require_minimum_opset(node, opset, 1)?;
            require_unary(node)?;
            validate_attributes(node, legacy_consumed_inputs(opset, ALPHA, ALPHA_LEGACY))?;
        }
        "Selu" => {
            // Selu-1 has materially different alpha/gamma defaults. LayerSpec
            // does not retain the authored opset, so admitting the absent v1
            // attributes would silently execute the Selu-6 constants.
            require_minimum_opset(node, opset, 6)?;
            require_unary(node)?;
            validate_attributes(node, ALPHA_GAMMA)?;
        }
        "HardSigmoid" => {
            require_minimum_opset(node, opset, 1)?;
            require_unary(node)?;
            validate_attributes(
                node,
                legacy_consumed_inputs(opset, ALPHA_BETA, ALPHA_BETA_LEGACY),
            )?;
        }
        "HardSwish" => {
            require_minimum_opset(node, opset, 14)?;
            require_unary(node)?;
            validate_attributes(node, NO_ATTRIBUTES)?;
        }
        "Celu" => {
            require_minimum_opset(node, opset, 12)?;
            require_unary(node)?;
            validate_attributes(node, ALPHA)?;
        }
        "ThresholdedRelu" => {
            require_minimum_opset(node, opset, 10)?;
            require_unary(node)?;
            validate_attributes(node, ALPHA)?;
        }
        "Shrink" => {
            require_minimum_opset(node, opset, 9)?;
            require_unary(node)?;
            validate_attributes(node, SHRINK_ATTRIBUTES)?;
        }
        "Clip" => validate_clip(node, opset)?,
        "PRelu" => {
            require_minimum_opset(node, opset, 1)?;
            require_exact_io(node, 2, 1)?;
            validate_attributes(
                node,
                legacy_consumed_inputs(opset, NO_ATTRIBUTES, CONSUMED_INPUTS),
            )?;
        }
        "Gelu" => {
            require_minimum_opset(node, opset, 20)?;
            require_unary(node)?;
            validate_attributes(node, GELU_ATTRIBUTES)?;
        }
        "Mish" => {
            require_minimum_opset(node, opset, 18)?;
            require_unary(node)?;
            validate_attributes(node, NO_ATTRIBUTES)?;
        }
        "Swish" => {
            require_minimum_opset(node, opset, 24)?;
            require_unary(node)?;
            validate_attributes(node, SWISH_ATTRIBUTES)?;
        }
        "Tanh" | "Sigmoid" | "Exp" | "Log" | "Floor" | "Ceil" | "Reciprocal" | "Abs" | "Neg"
        | "Sqrt" => {
            require_minimum_opset(node, opset, 1)?;
            require_unary(node)?;
            validate_attributes(
                node,
                legacy_consumed_inputs(opset, NO_ATTRIBUTES, CONSUMED_INPUTS),
            )?;
        }
        "Softplus" | "Softsign" => {
            require_minimum_opset(node, opset, 1)?;
            require_unary(node)?;
            validate_attributes(node, NO_ATTRIBUTES)?;
        }
        "Sin" | "Cos" | "Tan" | "Atan" => {
            require_minimum_opset(node, opset, 7)?;
            require_unary(node)?;
            validate_attributes(node, NO_ATTRIBUTES)?;
        }
        "Erf" | "Sign" => {
            require_minimum_opset(node, opset, 9)?;
            require_unary(node)?;
            validate_attributes(node, NO_ATTRIBUTES)?;
        }
        "Round" => {
            require_minimum_opset(node, opset, 11)?;
            require_unary(node)?;
            validate_attributes(node, NO_ATTRIBUTES)?;
        }
        "Softmax" | "LogSoftmax" => {
            // Before opset 13, ONNX coerced the input to 2D and normalized over
            // the flattened suffix beginning at `axis`. NY's layers normalize one
            // authored axis, which is the Softmax-13 contract. The two agree
            // EXACTLY when the authored axis denotes the FINAL dimension, because
            // then the flattened suffix is a single dimension and the coercion is
            // a no-op on the normalized axis.
            //
            // That equality is not assumed here — it is PROVED per node, downstream
            // and fail-closed, by `convert.rs::authenticate_standard_softmax_semantics`:
            // for `version < 13 && axis != -1` it demands an AUTHENTICATED input rank
            // (erroring when the rank cannot be recovered) and rejects the node unless
            // the resolved axis is `rank - 1`, with the message "flattens multiple
            // suffix dimensions, which ny's single-axis layer does not represent".
            // That check runs unconditionally for every converted node
            // (`convert.rs:467`), so nothing admitted here can reach a bound with
            // legacy suffix-flattening semantics.
            //
            // A blanket pre-13 rejection at THIS layer is therefore strictly stronger
            // than the semantics require, and it is not free. MEASURED 2026-08-06 on
            // GB10: `vit_2023`'s two graphs are opset 9 with `axis = 3` on rank-4
            // attention logits — precisely the equal case the downstream gate exists to
            // admit — so this line refused BOTH models and returned `unknown` in
            // 1.4-2.5 s of a 100 s budget on every one of the category's 200 instances
            // (`NY-HARNESS: MODEL-LOAD-FAILURE ... requires opset 13 or newer, got 9`).
            // Keep the arity/attribute authentication; defer the semantics to the gate
            // that can actually see the rank.
            require_minimum_opset(node, opset, 1)?;
            require_unary(node)?;
            validate_attributes(node, AXIS)?;
        }
        "SiLU" => {
            return Err(NyError::UnsupportedOp(format!(
                "standard-domain ONNX SiLU node '{}' is not a registered main-domain operator; use Swish-24 or an explicitly registered custom domain",
                node.name
            )));
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Require the exact FLOAT32 tensor path represented by WeightStore and NY's
/// propagators.  ONNX permits wider (and for some ops integer) type sets, but
/// normalizing those values to f32 would not preserve their authored
/// arithmetic semantics.
pub(super) fn validate_activation_float32_data_path(
    node: &NodeProto,
    raw_dtypes: &mut RawDtypeResolver<'_>,
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
                format!(
                    "cannot authenticate FLOAT32 dtype for tensor '{value}' before normalization"
                ),
            )
        })?;
        if dtype != FLOAT {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX {} node '{}' tensor '{}' has ONNX dtype {dtype}; ny only represents the FLOAT32 activation data path",
                node.op_type, node.name, value
            )));
        }
    }
    Ok(())
}

fn validate_clip(node: &NodeProto, opset: i64) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    if opset < 11 {
        require_unary(node)?;
        validate_attributes(
            node,
            legacy_consumed_inputs(opset, CLIP_ATTRIBUTES, CLIP_ATTRIBUTES_LEGACY),
        )?;
        return Ok(());
    }

    if !(1..=3).contains(&node.input.len())
        || node.input.first().is_none_or(String::is_empty)
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return Err(node_error(
            node,
            format!(
                "requires one non-empty data input, up to two optional scalar-bound inputs, and one non-empty output; got inputs {:?} and outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    validate_attributes(node, NO_ATTRIBUTES)
}

fn legacy_consumed_inputs(
    opset: i64,
    modern: &'static [(&'static str, AttributeRule)],
    legacy: &'static [(&'static str, AttributeRule)],
) -> &'static [(&'static str, AttributeRule)] {
    if opset >= 6 {
        modern
    } else {
        legacy
    }
}

fn validate_attributes(node: &NodeProto, rules: &[(&str, AttributeRule)]) -> Result<()> {
    for attribute in &node.attribute {
        let Some((_, rule)) = rules
            .iter()
            .find(|(name, _)| *name == attribute.name.as_str())
        else {
            return Err(node_error(
                node,
                format!("does not define attribute '{}'", attribute.name),
            ));
        };
        validate_attribute(node, attribute, *rule)?;
    }
    Ok(())
}

fn validate_attribute(
    node: &NodeProto,
    attribute: &AttributeProto,
    rule: AttributeRule,
) -> Result<()> {
    match rule {
        AttributeRule::FiniteFloat => {
            if attribute.r#type != attribute_type::FLOAT {
                return Err(attribute_type_error(node, attribute, "FLOAT"));
            }
            if !attribute.f_value().is_finite() {
                return Err(node_error(
                    node,
                    format!("attribute '{}' must be finite", attribute.name),
                ));
            }
        }
        AttributeRule::Int => {
            if attribute.r#type != attribute_type::INT {
                return Err(attribute_type_error(node, attribute, "INT"));
            }
        }
        AttributeRule::Ints => {
            if attribute.r#type != attribute_type::INTS {
                return Err(attribute_type_error(node, attribute, "INTS"));
            }
        }
        AttributeRule::StringEnum(values) => {
            if attribute.r#type != attribute_type::STRING {
                return Err(attribute_type_error(node, attribute, "STRING"));
            }
            if !values.contains(&attribute.s_value()) {
                return Err(node_error(
                    node,
                    format!(
                        "attribute '{}' has unsupported value {:?}",
                        attribute.name,
                        String::from_utf8_lossy(attribute.s_value())
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn require_minimum_opset(node: &NodeProto, opset: i64, minimum: i64) -> Result<()> {
    if opset < minimum {
        return Err(node_error(
            node,
            format!("requires opset {minimum} or newer, got {opset}"),
        ));
    }
    Ok(())
}

fn require_unary(node: &NodeProto) -> Result<()> {
    require_exact_io(node, 1, 1)
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
                "requires exactly {inputs} non-empty input(s) and {outputs} non-empty output(s); got inputs {:?} and outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    Ok(())
}

fn attribute_type_error(node: &NodeProto, attribute: &AttributeProto, expected: &str) -> NyError {
    node_error(
        node,
        format!("attribute '{}' must have type {expected}", attribute.name),
    )
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
    use crate::onnx_proto::{GraphProto, TensorTypeProto, TypeProto, ValueInfoProto};

    fn node(op_type: &str, inputs: &[&str], attributes: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            name: "activation".to_string(),
            op_type: op_type.to_string(),
            input: inputs.iter().map(|value| (*value).to_string()).collect(),
            output: vec!["y".to_string()],
            attribute: attributes,
            ..Default::default()
        }
    }

    fn float_attribute(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: attribute_type::FLOAT,
            f: Some(value),
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

    fn ints_attribute(name: &str, values: &[i64]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: attribute_type::INTS,
            ints: values.to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn modern_function_activations_enforce_first_opset_and_unary_signature() {
        for (op_type, first_opset) in [("HardSwish", 14), ("Mish", 18), ("Gelu", 20), ("Swish", 24)]
        {
            assert!(
                validate_activation_schema(&node(op_type, &["x"], vec![]), first_opset - 1)
                    .is_err()
            );
            assert!(
                validate_activation_schema(&node(op_type, &["x"], vec![]), first_opset)
                    .expect("first registered opset should validate")
            );
            assert!(validate_activation_schema(
                &node(op_type, &["x", "extra"], vec![]),
                first_opset
            )
            .is_err());
        }
    }

    #[test]
    fn gelu_approximate_is_a_closed_string_enum() {
        for value in [b"none".as_slice(), b"tanh".as_slice()] {
            validate_activation_schema(
                &node("Gelu", &["x"], vec![string_attribute("approximate", value)]),
                20,
            )
            .expect("registered Gelu approximation");
        }
        assert!(validate_activation_schema(
            &node(
                "Gelu",
                &["x"],
                vec![string_attribute("approximate", b"fast")],
            ),
            20,
        )
        .is_err());
        assert!(validate_activation_schema(
            &node("Gelu", &["x"], vec![float_attribute("approximate", 0.0)],),
            20,
        )
        .is_err());
    }

    #[test]
    fn swish_uses_standard_alpha_spelling_and_silu_fails_closed() {
        validate_activation_schema(
            &node("Swish", &["x"], vec![float_attribute("alpha", 1.0)]),
            24,
        )
        .expect("Swish-24 alpha");
        assert!(validate_activation_schema(
            &node("Swish", &["x"], vec![float_attribute("beta", 1.0)]),
            24,
        )
        .is_err());
        assert!(matches!(
            validate_activation_schema(&node("SiLU", &["x"], vec![]), 24),
            Err(NyError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn clip_schema_switches_from_attributes_to_optional_inputs_at_opset_11() {
        validate_activation_schema(
            &node("Clip", &["x"], vec![float_attribute("min", -1.0)]),
            10,
        )
        .expect("legacy Clip attributes");
        assert!(validate_activation_schema(
            &node("Clip", &["x"], vec![float_attribute("min", -1.0)]),
            11,
        )
        .is_err());
        validate_activation_schema(&node("Clip", &["x", "", "max"], vec![]), 11)
            .expect("modern Clip optional min placeholder");
        assert!(validate_activation_schema(
            &node("Clip", &["x", "min", "max", "extra"], vec![]),
            11
        )
        .is_err());
    }

    #[test]
    fn prelu_requires_live_slope_input_and_versioned_legacy_attribute() {
        assert!(validate_activation_schema(&node("PRelu", &["x"], vec![]), 16).is_err());
        validate_activation_schema(&node("PRelu", &["x", "slope"], vec![]), 16)
            .expect("two-input PRelu");

        let consumed = AttributeProto {
            name: "consumed_inputs".to_string(),
            r#type: attribute_type::INTS,
            ints: vec![0, 0],
            ..Default::default()
        };
        validate_activation_schema(&node("PRelu", &["x", "slope"], vec![consumed.clone()]), 1)
            .expect("PRelu-1 legacy attribute");
        assert!(
            validate_activation_schema(&node("PRelu", &["x", "slope"], vec![consumed]), 6).is_err()
        );
    }

    #[test]
    fn unknown_wrong_typed_and_non_finite_attributes_are_rejected() {
        assert!(validate_activation_schema(
            &node("Elu", &["x"], vec![float_attribute("unknown", 1.0)]),
            6,
        )
        .is_err());
        assert!(validate_activation_schema(
            &node("Elu", &["x"], vec![string_attribute("alpha", b"1")]),
            6,
        )
        .is_err());
        assert!(validate_activation_schema(
            &node("Elu", &["x"], vec![float_attribute("alpha", f32::NAN)]),
            6,
        )
        .is_err());
    }

    #[test]
    fn selu_rejects_v1_defaults_that_cannot_survive_layer_spec_lowering() {
        assert!(validate_activation_schema(&node("Selu", &["x"], vec![]), 1).is_err());
        validate_activation_schema(&node("Selu", &["x"], vec![]), 6)
            .expect("Selu-6 uses the implemented defaults");
    }

    #[test]
    fn otherwise_valid_non_float_activation_fails_closed() {
        const DOUBLE: i32 = 11;

        for (dtype, should_pass) in [(FLOAT, true), (DOUBLE, false)] {
            let activation = node("Relu", &["x"], vec![]);
            let graph = GraphProto {
                node: vec![activation],
                input: vec![value_info("x", dtype)],
                ..Default::default()
            };
            let activation = &graph.node[0];
            assert!(validate_activation_schema(activation, 14).expect("valid Relu schema"));
            let mut raw_dtypes = RawDtypeResolver::new(&graph);
            assert_eq!(
                validate_activation_float32_data_path(activation, &mut raw_dtypes).is_ok(),
                should_pass
            );
        }
    }

    #[test]
    fn activation_data_path_resolves_multi_output_recurrent_tensor_types() {
        for (op_type, outputs) in [
            ("RNN", vec!["y", "y_h"]),
            ("GRU", vec!["y", "y_h"]),
            ("LSTM", vec!["y", "y_h", "y_c"]),
        ] {
            let recurrent = NodeProto {
                name: "recurrent".to_string(),
                op_type: op_type.to_string(),
                input: vec!["x".to_string(), "w".to_string(), "r".to_string()],
                output: outputs.into_iter().map(str::to_string).collect(),
                ..Default::default()
            };
            let mut sigmoid = node("Sigmoid", &["y_h"], vec![]);
            sigmoid.output[0] = "probability".to_string();
            let graph = GraphProto {
                input: vec![value_info("x", FLOAT)],
                node: vec![recurrent, sigmoid.clone()],
                ..Default::default()
            };

            let mut raw_dtypes = RawDtypeResolver::new(&graph);
            validate_activation_float32_data_path(&sigmoid, &mut raw_dtypes).unwrap_or_else(
                |error| panic!("{op_type} outputs have the first X input's tensor type: {error}"),
            );
        }
    }

    #[test]
    fn activation_dtype_proof_accepts_omitted_optional_placeholders() {
        let gemm = NodeProto {
            name: "affine".to_string(),
            op_type: "Gemm".to_string(),
            input: vec!["x".to_string(), "w".to_string(), String::new()],
            output: vec!["logits".to_string()],
            ..Default::default()
        };
        let dropout = NodeProto {
            name: "inference_dropout".to_string(),
            op_type: "Dropout".to_string(),
            input: vec!["logits".to_string()],
            output: vec!["dropped".to_string(), String::new()],
            ..Default::default()
        };
        let relu = node("Relu", &["dropped"], vec![]);
        let graph = GraphProto {
            input: vec![value_info("x", FLOAT), value_info("w", FLOAT)],
            node: vec![gemm, dropout, relu.clone()],
            ..Default::default()
        };

        let mut raw_dtypes = RawDtypeResolver::new(&graph);
        validate_activation_float32_data_path(&relu, &mut raw_dtypes).expect(
            "empty optional Gemm C and Dropout mask placeholders do not change the FLOAT path",
        );
    }

    #[test]
    fn remaining_mapped_standard_unaries_are_versioned_and_attribute_closed() {
        for (op_type, first_opset) in [
            ("Tanh", 1),
            ("Sigmoid", 1),
            ("Softplus", 1),
            ("Exp", 1),
            ("Log", 1),
            ("Softsign", 1),
            ("Floor", 1),
            ("Ceil", 1),
            ("Reciprocal", 1),
            ("Abs", 1),
            ("Neg", 1),
            ("Sqrt", 1),
            ("Sin", 7),
            ("Cos", 7),
            ("Tan", 7),
            ("Atan", 7),
            ("Erf", 9),
            ("Sign", 9),
            ("Round", 11),
        ] {
            if first_opset > 1 {
                assert!(validate_activation_schema(
                    &node(op_type, &["x"], vec![]),
                    first_opset - 1,
                )
                .is_err());
            }
            validate_activation_schema(&node(op_type, &["x"], vec![]), first_opset)
                .expect("first registered unary schema");
            assert!(validate_activation_schema(
                &node(op_type, &["x", "extra"], vec![]),
                first_opset,
            )
            .is_err());
        }

        let consumed = ints_attribute("consumed_inputs", &[0]);
        validate_activation_schema(&node("Exp", &["x"], vec![consumed.clone()]), 1)
            .expect("Exp-1 legacy optimization attribute");
        assert!(validate_activation_schema(&node("Exp", &["x"], vec![consumed]), 6).is_err());
        assert!(validate_activation_schema(
            &node(
                "Softplus",
                &["x"],
                vec![ints_attribute("consumed_inputs", &[0])]
            ),
            1,
        )
        .is_err());
    }

    #[test]
    fn softmax_family_defers_legacy_flattened_suffix_semantics_to_the_rank_gate() {
        for op_type in ["Softmax", "LogSoftmax"] {
            // Pre-13 nodes are ADMITTED here: whether the authored axis denotes the
            // final dimension cannot be decided without the input rank, and
            // `convert.rs::authenticate_standard_softmax_semantics` decides it
            // fail-closed once the rank is authenticated. Rejecting at this layer
            // zeroed all 200 vit_2023 instances (opset 9, axis 3, rank 4).
            validate_activation_schema(&node(op_type, &["x"], vec![]), 12)
                .expect("pre-13 axis semantics are authenticated downstream, not here");
            validate_activation_schema(
                &node(
                    op_type,
                    &["x"],
                    vec![AttributeProto {
                        name: "axis".to_string(),
                        r#type: attribute_type::INT,
                        i: Some(3),
                        ..Default::default()
                    }],
                ),
                9,
            )
            .expect("vit_2023's authored form: opset 9 with an INT axis attribute");
            // Arity and attribute authentication still bite at every opset.
            assert!(validate_activation_schema(
                &node(op_type, &["x"], vec![float_attribute("axis", -1.0)]),
                9,
            )
            .is_err());
            assert!(validate_activation_schema(&node(op_type, &["x", "y"], vec![]), 9).is_err());
            validate_activation_schema(&node(op_type, &["x"], vec![]), 13)
                .expect("opset-13 axis-wise semantics");
            validate_activation_schema(
                &node(
                    op_type,
                    &["x"],
                    vec![AttributeProto {
                        name: "axis".to_string(),
                        r#type: attribute_type::INT,
                        i: Some(-1),
                        ..Default::default()
                    }],
                ),
                13,
            )
            .expect("INT axis attribute");
            assert!(validate_activation_schema(
                &node(op_type, &["x"], vec![float_attribute("axis", -1.0)]),
                13,
            )
            .is_err());
        }
    }
}
