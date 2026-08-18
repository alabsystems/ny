// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Versioned standard-domain schema checks that must run on the authored
//! protobuf graph, before constant folding or dtype normalization.

use crate::loader::attributes::validate_attribute_storage;
use crate::loader::const_fold::is_standard_onnx_domain;
use crate::loader::tensor::{
    validate_constant_of_shape_schema, validate_constant_payload_for_opset,
};
use crate::onnx_proto::{self, GraphProto, NodeProto};
use ny_core::{NyError, Result};
use std::collections::{HashMap, HashSet};

use super::literal_cone::{LiteralCone, LiteralExemptions};
use super::quantization_preflight::RawDtypeResolver;

const INT32: i32 = 6;
const INT64: i32 = 7;

/// Validate the authored standard-domain schemas.
///
/// Returns the raw-schema refusals that were deferred because their node is a
/// load-time literal computation (see `literal_cone`).  The caller MUST hand
/// them to `LiteralExemptions::require_all_folded` once constant folding has
/// run; that call re-raises every deferral whose node survived.
pub(super) fn validate_standard_schemas(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
) -> Result<LiteralExemptions> {
    let literals = LiteralCone::new(graph);
    let mut exemptions = LiteralExemptions::default();
    let standard_opset = opset_imports
        .get("")
        .copied()
        .or_else(|| opset_imports.get("ai.onnx").copied());

    // Validate Constant's versioned union before path audits inspect the
    // dtype it purports to produce.  Otherwise a malformed pre-opset-12
    // value_int Constant can be rejected for its consumer first, obscuring
    // the actual authored-schema defect.
    if let Some(opset) = standard_opset {
        for node in &graph.node {
            if is_standard_onnx_domain(&node.domain) && node.op_type == "Constant" {
                validate_constant_payload_for_opset(node, opset)?;
            }
        }
    }

    super::reduction_schema::validate_reduction_comparison_schemas(
        graph,
        opset_imports,
        &literals,
        &mut exemptions,
    )?;
    super::structural_schema::validate_structural_schemas(
        graph,
        opset_imports,
        &literals,
        &mut exemptions,
    )?;
    super::linear_schema::validate_linear_convolution_schemas(graph, opset_imports)?;
    super::transform_schema::validate_transform_schemas(graph, opset_imports)?;

    let mut raw_dtypes = RawDtypeResolver::new(graph);

    for node in &graph.node {
        // AttributeProto's union and per-node name uniqueness are ONNX IR
        // invariants, not standard-operator schema details.  Enforce them
        // before a registered custom handler can make an order-dependent
        // choice or lose a hidden payload during conversion.
        let mut attribute_names = HashSet::new();
        for attribute in &node.attribute {
            if !attribute_names.insert(attribute.name.as_str()) {
                return Err(NyError::ModelLoad(format!(
                    "ONNX {} node '{}' in domain '{}' has duplicate '{}' attributes",
                    node.op_type, node.name, node.domain, attribute.name
                )));
            }
            validate_attribute_storage(node, attribute)?;
        }
        if !is_standard_onnx_domain(&node.domain) {
            continue;
        }
        let opset = standard_opset.ok_or_else(|| {
            NyError::ModelLoad(format!(
                "standard ONNX {} node '{}' has no standard-domain opset import",
                node.op_type, node.name
            ))
        })?;
        if super::activation_schema::validate_activation_schema(node, opset)? {
            super::activation_schema::validate_activation_float32_data_path(node, &mut raw_dtypes)?;
            continue;
        }
        if super::arithmetic_schema::validate_arithmetic_schema(node, opset)? {
            super::arithmetic_schema::validate_arithmetic_float32_data_path(
                node,
                &mut raw_dtypes,
                &literals,
                &mut exemptions,
            )?;
            continue;
        }
        match node.op_type.as_str() {
            "Constant" => validate_constant_payload_for_opset(node, opset)?,
            "ConstantOfShape" => {
                if opset < 9 {
                    return Err(NyError::ModelLoad(format!(
                        "standard ONNX ConstantOfShape node '{}' requires opset 9 or newer, got {opset}",
                        node.name
                    )));
                }
                validate_constant_of_shape_schema(node)?;
                let shape_input = &node.input[0];
                let dtype = raw_dtypes.resolve(shape_input)?.ok_or_else(|| {
                    NyError::ModelLoad(format!(
                        "standard ONNX ConstantOfShape node '{}' cannot authenticate required INT64 dtype for shape input '{}' before folding",
                        node.name, shape_input
                    ))
                })?;
                if dtype != 7 {
                    return Err(NyError::ModelLoad(format!(
                        "standard ONNX ConstantOfShape node '{}' requires INT64 shape input '{}', got ONNX dtype {dtype}",
                        node.name, shape_input
                    )));
                }
            }
            "ArgMax" | "ArgMin" => validate_arg_extrema_schema(node, opset, &mut raw_dtypes)?,
            "TopK" => validate_topk_schema(graph, node, opset, &mut raw_dtypes)?,
            "Dropout" => validate_dropout_schema(node, opset, &mut raw_dtypes)?,
            "Identity" => {
                require_minimum_opset(node, opset, 1)?;
                require_exact_io(node, 1, 1)?;
                require_no_attributes(node)?;
            }
            "NonZero" => {
                require_minimum_opset(node, opset, 9)?;
                require_exact_io(node, 1, 1)?;
                require_no_attributes(node)?;
                // ONNX NonZero has a data-dependent second output dimension.
                // ny's fixed-shape tensor graph cannot represent the union of
                // those shapes: padding to the maximum count invents columns
                // that do not exist in the authored execution.
                return Err(NyError::UnsupportedOp(format!(
                    "standard ONNX NonZero node '{}' has a data-dependent output shape that ny cannot represent soundly",
                    node.name
                )));
            }
            "Gather" => validate_gather_schema(graph, node, opset, &mut raw_dtypes)?,
            "AveragePool" | "GlobalAveragePool" | "MaxPool" => {
                validate_pool_schema(node, opset, &mut raw_dtypes)?
            }
            "BatchNormalization"
            | "InstanceNormalization"
            | "LayerNormalization"
            | "GroupNormalization"
            | "RMSNormalization" => {
                for (index, value) in node
                    .input
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| !value.is_empty())
                {
                    require_float_data(node, &mut raw_dtypes, value, &format!("input {index}"))?;
                }
            }
            "Attention" => reject_unrepresented_standard_attention(node, opset)?,
            "MultiHeadAttention" => {
                return Err(NyError::UnsupportedOp(format!(
                    "standard-domain ONNX MultiHeadAttention node '{}' is not a registered main-domain operator; use the standard Attention operator or an explicitly registered custom domain",
                    node.name
                )));
            }
            "ArgSort"
            | "Snake"
            | "RoPE"
            | "RotaryPositionEmbedding"
            | "SimplifiedLayerNormalization"
            | "AdaIN"
            | "AdaptiveInstanceNorm"
            | "AdaptiveInstanceNormalization" => {
                return Err(NyError::UnsupportedOp(format!(
                    "standard-domain ONNX {} node '{}' is not a registered main-domain operator; use an explicitly imported and registered vendor domain",
                    node.op_type, node.name
                )));
            }
            _ => {}
        }
    }
    Ok(exemptions)
}

/// Test-only: run the preflight and immediately discharge its deferrals.
///
/// A unit test never runs constant folding, so every deferred literal-cone
/// refusal must re-raise here with its original message. That keeps the
/// schema-level assertions in `tests/` measuring the raw gate itself rather
/// than the deferral.
#[cfg(test)]
pub(super) fn validate_standard_schemas_unfolded(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
) -> Result<()> {
    validate_standard_schemas(graph, opset_imports)?.require_all_folded(&crate::WeightStore::new())
}

/// Do not reinterpret the standard ONNX Attention function as ny's much
/// smaller ternary SelfAttention layer.  Attention-23/24 additionally define
/// packed 3D head layouts, masks, KV caches, GQA/MQA head replication,
/// softcap/precision controls, and optional state/debug outputs.  Even its
/// simple 4D form scales Q and K before MatMul, which is not the same authored
/// floating-point graph as multiplying the completed score matrix.  An exact
/// subset can be admitted later, but only with a source-semantics certificate.
fn reject_unrepresented_standard_attention(node: &NodeProto, opset: i64) -> Result<()> {
    require_minimum_opset(node, opset, 23)?;
    Err(NyError::UnsupportedOp(format!(
        "standard ONNX Attention node '{}' cannot be mapped to ny's simplified SelfAttention layer without changing the authored Attention-{} semantics",
        node.name,
        if opset >= 24 { 24 } else { 23 }
    )))
}

fn node_error(node: &NodeProto, detail: impl Into<String>) -> NyError {
    NyError::ModelLoad(format!(
        "standard ONNX {} node '{}' {}",
        node.op_type,
        node.name,
        detail.into()
    ))
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

fn require_no_attributes(node: &NodeProto) -> Result<()> {
    if let Some(attribute) = node.attribute.first() {
        return Err(node_error(
            node,
            format!("does not define attribute '{}'", attribute.name),
        ));
    }
    Ok(())
}

fn require_int_attribute(node: &NodeProto, attribute: &onnx_proto::AttributeProto) -> Result<i64> {
    if attribute.r#type != onnx_proto::attribute_type::INT {
        return Err(node_error(
            node,
            format!("attribute '{}' must have type INT", attribute.name),
        ));
    }
    Ok(attribute.i_value())
}

fn require_bool_int_attribute(
    node: &NodeProto,
    attribute: &onnx_proto::AttributeProto,
) -> Result<i64> {
    let value = require_int_attribute(node, attribute)?;
    if !matches!(value, 0 | 1) {
        return Err(node_error(
            node,
            format!(
                "attribute '{}' must be encoded as INT 0 or 1, got {value}",
                attribute.name
            ),
        ));
    }
    Ok(value)
}

fn require_ints_attribute<'a>(
    node: &NodeProto,
    attribute: &'a onnx_proto::AttributeProto,
) -> Result<&'a [i64]> {
    if attribute.r#type != onnx_proto::attribute_type::INTS {
        return Err(node_error(
            node,
            format!("attribute '{}' must have type INTS", attribute.name),
        ));
    }
    Ok(&attribute.ints)
}

fn require_string_attribute<'a>(
    node: &NodeProto,
    attribute: &'a onnx_proto::AttributeProto,
) -> Result<&'a [u8]> {
    if attribute.r#type != onnx_proto::attribute_type::STRING {
        return Err(node_error(
            node,
            format!("attribute '{}' must have type STRING", attribute.name),
        ));
    }
    Ok(attribute.s_value())
}

/// Admit only the two-dimensional pooling subset represented by ny's native
/// layers.  In particular, ONNX defaults `strides` to one; validating the raw
/// schema here prevents a missing `kernel_shape` on AveragePool from being
/// mistaken for GlobalAveragePool and prevents a live MaxPool indices output
/// from being aliased to the values tensor.
fn validate_pool_schema(
    node: &NodeProto,
    opset: i64,
    raw_dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    if node.input.len() != 1
        || node.input[0].is_empty()
        || node.output.is_empty()
        || node.output[0].is_empty()
        || (node.op_type == "MaxPool" && node.output.len() > 2)
        || (node.op_type != "MaxPool" && node.output.len() != 1)
    {
        return Err(node_error(
            node,
            format!(
                "has an invalid pooling signature: inputs {:?}, outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    if node
        .output
        .get(1)
        .is_some_and(|indices| !indices.is_empty())
    {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX MaxPool node '{}' requests the Indices output, which ny's value-only pool layer does not represent",
            node.name
        )));
    }
    require_float_data(node, raw_dtypes, &node.input[0], "data")?;

    if node.op_type == "GlobalAveragePool" {
        return require_no_attributes(node);
    }

    let mut saw_kernel = false;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "kernel_shape" => {
                let values = require_ints_attribute(node, attribute)?;
                if values.len() != 2 || values.iter().any(|value| *value <= 0) {
                    return Err(node_error(
                        node,
                        format!(
                            "requires a positive two-dimensional kernel_shape, got {:?}",
                            values
                        ),
                    ));
                }
                saw_kernel = true;
            }
            "strides" => {
                let values = require_ints_attribute(node, attribute)?;
                if values.len() != 2 || values.iter().any(|value| *value <= 0) {
                    return Err(node_error(
                        node,
                        format!(
                            "requires positive two-dimensional strides, got {:?}",
                            values
                        ),
                    ));
                }
            }
            "pads" => {
                let values = require_ints_attribute(node, attribute)?;
                if values.len() != 4
                    || values.iter().any(|value| *value < 0)
                    || values[0] != values[2]
                    || values[1] != values[3]
                {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX {} node '{}' requires non-negative symmetric two-dimensional pads, got {:?}",
                        node.op_type, node.name, values
                    )));
                }
            }
            "auto_pad" => {
                let value = require_string_attribute(node, attribute)?;
                if value != b"NOTSET" {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX {} node '{}' uses unsupported auto_pad {:?}; only NOTSET is represented",
                        node.op_type,
                        node.name,
                        String::from_utf8_lossy(value)
                    )));
                }
            }
            "count_include_pad" if node.op_type == "AveragePool" && opset >= 7 => {
                require_bool_int_attribute(node, attribute)?;
            }
            "ceil_mode" if opset >= 10 => {
                if require_bool_int_attribute(node, attribute)? != 0 {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX {} node '{}' uses ceil_mode=1, which ny does not represent",
                        node.op_type, node.name
                    )));
                }
            }
            "dilations" if node.op_type == "MaxPool" && opset >= 10 => {
                let values = require_ints_attribute(node, attribute)?;
                if values != [1, 1] {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX MaxPool node '{}' uses unsupported dilations {:?}",
                        node.name, values
                    )));
                }
            }
            "dilations" if node.op_type == "AveragePool" && opset >= 19 => {
                let values = require_ints_attribute(node, attribute)?;
                if values != [1, 1] {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX AveragePool node '{}' uses unsupported dilations {:?}",
                        node.name, values
                    )));
                }
            }
            "storage_order" if node.op_type == "MaxPool" && opset >= 8 => {
                if require_bool_int_attribute(node, attribute)? != 0 {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX MaxPool node '{}' uses storage_order=1, which ny's value-only row-major pool does not represent",
                        node.name
                    )));
                }
            }
            _ => {
                return Err(node_error(
                    node,
                    format!(
                        "does not define attribute '{}' at opset {opset}",
                        attribute.name
                    ),
                ));
            }
        }
    }
    if !saw_kernel {
        return Err(node_error(
            node,
            "is missing required INTS attribute 'kernel_shape'",
        ));
    }
    Ok(())
}

fn validate_arg_extrema_schema(
    node: &NodeProto,
    opset: i64,
    raw_dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    require_exact_io(node, 1, 1)?;
    require_float_data(node, raw_dtypes, &node.input[0], "data")?;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "axis" | "keepdims" => {
                if attribute.name == "keepdims" {
                    require_bool_int_attribute(node, attribute)?;
                } else {
                    require_int_attribute(node, attribute)?;
                }
            }
            "select_last_index" if opset >= 12 => {
                require_bool_int_attribute(node, attribute)?;
            }
            "select_last_index" => {
                return Err(node_error(
                    node,
                    format!(
                        "attribute 'select_last_index' was introduced in opset 12, but the model imports {opset}"
                    ),
                ));
            }
            _ => {
                return Err(node_error(
                    node,
                    format!(
                        "does not define attribute '{}' at opset {opset}",
                        attribute.name
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_topk_schema(
    graph: &GraphProto,
    node: &NodeProto,
    opset: i64,
    raw_dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    require_exact_io(node, if opset < 10 { 1 } else { 2 }, 2)?;
    require_float_data(node, raw_dtypes, &node.input[0], "data")?;

    let mut legacy_k = None;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "axis" => {
                require_int_attribute(node, attribute)?;
            }
            "k" if opset < 10 => {
                legacy_k = Some(require_int_attribute(node, attribute)?);
            }
            "largest" | "sorted" if opset >= 11 => {
                let value = require_bool_int_attribute(node, attribute)?;
                if value == 0 {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX TopK node '{}' sets {}=0, which ny's descending, ordered TopK layer does not represent",
                        node.name, attribute.name
                    )));
                }
            }
            _ => {
                return Err(node_error(
                    node,
                    format!(
                        "does not define attribute '{}' at opset {opset}",
                        attribute.name
                    ),
                ));
            }
        }
    }

    if opset < 10 {
        match legacy_k {
            Some(k) if k > 0 => {}
            Some(k) => return Err(node_error(node, format!("requires positive k, got {k}"))),
            None => return Err(node_error(node, "is missing required INT attribute 'k'")),
        }
    } else {
        require_index_dtype(graph, node, &node.input[1], &[INT64], raw_dtypes, "K")?;
    }
    Ok(())
}

fn validate_dropout_schema(
    node: &NodeProto,
    opset: i64,
    raw_dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    let modern = opset >= 12;
    let expected_inputs = if modern { 1..=3 } else { 1..=1 };
    if !expected_inputs.contains(&node.input.len())
        || node.input.first().is_none_or(String::is_empty)
        || node.output.is_empty()
        || node.output.len() > 2
        || node.output[0].is_empty()
    {
        return Err(node_error(
            node,
            format!(
                "has an invalid opset-{opset} optional signature: inputs {:?}, outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    if node.output.get(1).is_some_and(|output| !output.is_empty()) {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Dropout node '{}' requests the mask output, which cannot be erased as an identity",
            node.name
        )));
    }
    require_float_data(node, raw_dtypes, &node.input[0], "data")?;

    let mut ratio = 0.5_f32;
    let mut is_test = 0_i64;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "ratio" if opset < 12 && attribute.r#type == onnx_proto::attribute_type::FLOAT => {
                ratio = attribute.f_value();
            }
            "is_test" if opset < 7 => {
                is_test = require_int_attribute(node, attribute)?;
            }
            "consumed_inputs"
                if opset < 6 && attribute.r#type == onnx_proto::attribute_type::INTS => {}
            "seed" if modern => {
                require_int_attribute(node, attribute)?;
            }
            _ => {
                return Err(node_error(
                    node,
                    format!(
                        "does not define a '{}' attribute with this type at opset {opset}",
                        attribute.name
                    ),
                ));
            }
        }
    }

    if modern {
        if node.input.get(2).is_some_and(|input| !input.is_empty()) {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX Dropout node '{}' supplies training_mode; ny only erases the authored default inference mode",
                node.name
            )));
        }
    } else if opset < 7 && is_test == 0 && ratio != 0.0 {
        // Dropout-1 and Dropout-6 DO carry an authored mode control: `is_test`,
        // whose schema default is 0 = TRAINING. An `is_test=0` node with a
        // non-zero ratio therefore says "train", and erasing it is unsound.
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Dropout node '{}' cannot be erased at opset {opset}: is_test defaults to training mode and ratio is {ratio}; only ratio=0 is unconditionally an identity",
            node.name
        )));
    }
    // Opsets 7..=11 fall through to the identity erasure for ANY `ratio`, and
    // that admission is narrow and normative rather than a blanket erase:
    //
    //  * Dropout-7 REMOVED `is_test`, and `training_mode` was not introduced
    //    until Dropout-12. Across 7..=11 the schema therefore has NO authored
    //    control — attribute or input — by which a graph could select training
    //    mode. The only attribute admitted above at those opsets is `ratio`
    //    (`seed` is rejected outside `modern`), so no such control can be
    //    hiding in this node either.
    //  * ONNX's own version adapter resolves the resulting prose ambiguity:
    //    converting an opset-8 `Dropout(data){ratio=0.5}` to opset 12 emits
    //    `Dropout(data, ratio)` with the `training_mode` input ABSENT, i.e. at
    //    its schema default `false`. Dropout-12 states normatively that when
    //    `training_mode` is false "ratio is ignored and the operation mimics
    //    inference mode where nothing will be dropped from the input data".
    //    So the exact opset-12 meaning of every opset-7..=11 Dropout is the
    //    identity, for every ratio.
    //  * The mask output is refused above, so no consumer can observe the
    //    all-ones mask that the inference-mode semantics would produce.
    //
    // Anything outside 7..=11, or any node carrying an explicit
    // `training_mode` operand, still fails closed.
    Ok(())
}

/// NY's propagated arithmetic is specifically ONNX tensor(float), not a
/// generic implementation of every numeric tensor type admitted by an ONNX
/// operator schema.  In particular, normalizing DOUBLE/FLOAT16/BFLOAT16 or an
/// integer activation into WeightStore's f32 view changes authored rounding or
/// values.  Fail before folding erases that dtype provenance.
fn require_float_data(
    node: &NodeProto,
    raw_dtypes: &mut RawDtypeResolver<'_>,
    value: &str,
    role: &str,
) -> Result<()> {
    let dtype = raw_dtypes.resolve(value)?.ok_or_else(|| {
        node_error(
            node,
            format!("cannot authenticate the ONNX dtype of {role} value '{value}'"),
        )
    })?;
    if dtype != 1 {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' uses {role} dtype {dtype}; ny represents only tensor(float) data-path arithmetic",
            node.op_type, node.name
        )));
    }
    Ok(())
}

fn validate_gather_schema(
    graph: &GraphProto,
    node: &NodeProto,
    opset: i64,
    raw_dtypes: &mut RawDtypeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    require_exact_io(node, 2, 1)?;
    for attribute in &node.attribute {
        if attribute.name != "axis" {
            return Err(node_error(
                node,
                format!("does not define attribute '{}'", attribute.name),
            ));
        }
        require_int_attribute(node, attribute)?;
    }
    let data_dtype = raw_dtypes.resolve(&node.input[0])?.ok_or_else(|| {
        node_error(
            node,
            format!(
                "cannot authenticate the ONNX dtype of data input '{}'",
                node.input[0]
            ),
        )
    })?;
    if !matches!(data_dtype, 1 | INT64) {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Gather node '{}' uses data dtype {data_dtype}; ny represents only tensor(float) runtime data or sealed tensor(int64) structural data",
            node.name
        )));
    }
    require_index_dtype(
        graph,
        node,
        &node.input[1],
        &[INT32, INT64],
        raw_dtypes,
        "indices",
    )
}

fn require_index_dtype(
    graph: &GraphProto,
    node: &NodeProto,
    value: &str,
    allowed: &[i32],
    raw_dtypes: &mut RawDtypeResolver<'_>,
    role: &str,
) -> Result<()> {
    let dtype =
        resolve_index_dtype(graph, value, raw_dtypes, &mut HashSet::new())?.ok_or_else(|| {
            node_error(
                node,
                format!("cannot authenticate the ONNX dtype of {role} input '{value}'"),
            )
        })?;
    if !allowed.contains(&dtype) {
        return Err(node_error(
            node,
            format!(
                "requires {role} input '{value}' to have dtype {:?}, got {dtype}",
                allowed
            ),
        ));
    }
    Ok(())
}

fn resolve_index_dtype(
    graph: &GraphProto,
    value: &str,
    raw_dtypes: &mut RawDtypeResolver<'_>,
    active: &mut HashSet<String>,
) -> Result<Option<i32>> {
    if let Some(dtype) = raw_dtypes.resolve(value)? {
        return Ok(Some(dtype));
    }
    if !active.insert(value.to_string()) {
        return Ok(None);
    }
    let mut producers = graph
        .node
        .iter()
        .filter(|producer| producer.output.iter().any(|output| output == value));
    let Some(producer) = producers.next() else {
        active.remove(value);
        return Ok(None);
    };
    if producers.next().is_some() || !is_standard_onnx_domain(&producer.domain) {
        active.remove(value);
        return Ok(None);
    }
    let output_index = producer.output.iter().position(|output| output == value);
    let dtype = match (producer.op_type.as_str(), output_index) {
        ("ArgMax" | "ArgMin" | "NonZero", Some(0)) => Some(INT64),
        ("TopK", Some(1)) => Some(INT64),
        ("TopK", Some(0))
        | ("Identity" | "Reshape" | "Transpose" | "Flatten" | "Squeeze" | "Unsqueeze", Some(0)) => {
            match producer.input.first() {
                Some(input) if !input.is_empty() => {
                    resolve_index_dtype(graph, input, raw_dtypes, active)?
                }
                _ => None,
            }
        }
        _ => None,
    };
    active.remove(value);
    Ok(dtype)
}
