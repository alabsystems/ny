// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::model::OriginalFloat32Initializer;
use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use std::collections::{HashMap, HashSet};
use tracing::debug;

// ONNX TensorProto.DataType FLOAT.  Keep this check at the raw protobuf
// boundary: after tensor loading, WeightStore intentionally normalizes several
// authored dtypes to f32 and can no longer establish original provenance.
const ONNX_TENSOR_FLOAT32: i32 = 1;
const ONNX_TENSOR_INT64: i64 = 7;

use super::super::const_fold::{
    fold_constant_nodes, int64_cast_has_raw_int64_provenance,
    int64_cast_has_static_reshape_shape_use, is_standard_onnx_domain, raw_int64_shape_values,
};
use super::super::external_data::ExternalDataResolver;
use super::super::numeric_cast::{i64_to_f32_checked, i64_to_f32_warned};
use super::super::tensor::{
    extract_constant_tensor, tensor_proto_to_loaded_tensor,
    tensor_proto_to_loaded_tensor_with_external_raw, validate_constant_of_shape_schema,
    validate_constant_payload_schema, LoadedTensor,
};

pub(super) fn prepare_graph(
    graph: &mut onnx_proto::GraphProto,
    weights: &mut WeightStore,
    inferred_shapes: &mut HashMap<String, Vec<i64>>,
    capture_raw_float32_initializer_provenance: bool,
    mut external_data: Option<&mut ExternalDataResolver>,
) -> Result<HashMap<String, OriginalFloat32Initializer>> {
    let mut original_float32_initializers = HashMap::new();
    let mut initializer_names = HashSet::new();

    validate_computational_dtypes(graph)?;

    // Establish unambiguous initializer names before inserting any value or
    // running a rewrite. ONNX graph values are SSA: an initializer name must
    // be non-empty and unique, and cannot also be produced by a node. Enforce
    // this for every load, not only provenance-enabled loads; otherwise
    // WeightStore's replacement semantics silently select the last payload.
    for init in &graph.initializer {
        if init.name.is_empty() {
            return Err(NyError::ModelLoad(
                "ONNX initializer name cannot be empty".to_string(),
            ));
        }
        if !initializer_names.insert(init.name.clone()) {
            return Err(NyError::ModelLoad(format!(
                "duplicate ONNX initializer name '{}'",
                init.name
            )));
        }
    }
    validate_graph_value_names(graph, &initializer_names)?;

    // Constant tensor attributes are consumed at several later folding sites,
    // so materialize those uncommon payloads once. Initializers below retain
    // the streaming path and never store a second raw copy in the graph.
    if let Some(resolver) = external_data.as_deref_mut() {
        resolver.materialize_attribute_tensors(&mut graph.node)?;
    }

    // Extract weights from initializers.
    for init in &graph.initializer {
        let name = init.name.clone();
        let external_raw = match external_data.as_deref_mut() {
            Some(resolver) => resolver.read_tensor(init)?,
            None => None,
        };
        let tensor = match external_raw.as_deref() {
            Some(raw_data) => tensor_proto_to_loaded_tensor_with_external_raw(init, raw_data)?,
            None => tensor_proto_to_loaded_tensor(init)?,
        };
        reject_authored_reshape_copy_sentinels(&name, &tensor)?;
        debug!(
            "Loaded initializer: {} shape {:?}",
            name,
            tensor.float_data.shape()
        );
        insert_loaded_tensor(weights, name.clone(), tensor);
        if capture_raw_float32_initializer_provenance && init.data_type == ONNX_TENSOR_FLOAT32 {
            let current = weights.get(&name).ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "raw ONNX FLOAT initializer '{name}' was not inserted"
                ))
            })?;
            let revision = weights.revision(&name).ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "raw ONNX FLOAT initializer '{name}' has no valid weight revision"
                ))
            })?;
            original_float32_initializers.insert(
                name,
                OriginalFloat32Initializer::from_tensor(current, revision),
            );
        }
    }

    extract_constant_nodes_as_weights(&graph.node, weights)?;
    fold_constant_nodes(graph, weights, inferred_shapes);
    infer_concat_reshape_shapes(graph, weights);

    // Lower composite reductions into primitives we already support.
    lower_reduce_l2_nodes(&mut graph.node);
    extract_constant_nodes_as_weights(&graph.node, weights)?;

    // Lower LSTM nodes into per-timestep cell operations (MatMul, Add,
    // Sigmoid, Tanh, Mul) for bound propagation. Must run after weights
    // are loaded and constants folded, since we need W, R, B values and
    // input shapes. Graph inputs/value_info are passed separately to avoid
    // borrowing graph mutably and immutably at the same time.
    {
        let graph_inputs = graph.input.clone();
        let graph_value_info = graph.value_info().to_vec();
        super::super::lstm_unroll::lower_lstm_nodes(
            &mut graph.node,
            weights,
            &graph_inputs,
            &graph_value_info,
            inferred_shapes,
        );
    }

    // Re-extract constants in case the lowering created new Constant nodes.
    extract_constant_nodes_as_weights(&graph.node, weights)?;
    validate_skipped_nodes_are_semantic_identities(graph, weights)?;
    rewrite_materialized_int64_casts_as_identities(graph, weights);
    // Lowerings above synthesize graph value names. Recheck after every
    // prepare-time rewrite so generated outputs cannot acquire an
    // initializer's identity.
    validate_graph_value_names(graph, &initializer_names)?;
    Ok(original_float32_initializers)
}

/// Reject authored arithmetic whose rounding semantics the f32 verifier does
/// not model. This runs before initializer extraction or constant folding:
/// otherwise a DOUBLE/f16 computation can be evaluated in f32, have its dtype
/// provenance erased, and later look like an ordinary FLOAT constant.
fn validate_computational_dtypes(graph: &onnx_proto::GraphProto) -> Result<()> {
    let reject_tensor = |tensor: &onnx_proto::TensorProto, context: &str| -> Result<()> {
        if is_non_f32_floating_dtype(tensor.data_type) {
            return Err(NyError::ModelLoad(format!(
                "{context} '{}' uses ONNX dtype {}; ny verification only models FLOAT32 operation rounding",
                tensor.name, tensor.data_type
            )));
        }
        Ok(())
    };

    for initializer in &graph.initializer {
        reject_tensor(initializer, "initializer")?;
    }
    validate_direct_normalization_parameter_provenance(graph)?;
    let initializer_names: HashSet<&str> = graph
        .initializer
        .iter()
        .map(|initializer| initializer.name.as_str())
        .collect();
    for info in &graph.input {
        if initializer_names.contains(info.name.as_str()) {
            continue;
        }
        let elem_type = info
            .r#type
            .as_ref()
            .and_then(|ty| ty.tensor_type.as_ref())
            .map(|ty| ty.elem_type);
        if let Some(dtype) = elem_type {
            if dtype != ONNX_TENSOR_FLOAT32 {
                return Err(NyError::ModelLoad(format!(
                    "ONNX runtime input '{}' uses dtype {dtype}; ny verification currently accepts only FLOAT32 inputs",
                    info.name
                )));
            }
        }
    }
    for info in graph.output.iter().chain(graph.value_info()) {
        if let Some(elem_type) = info
            .r#type
            .as_ref()
            .and_then(|ty| ty.tensor_type.as_ref())
            .map(|ty| ty.elem_type)
        {
            if is_non_f32_floating_dtype(elem_type) {
                return Err(NyError::ModelLoad(format!(
                    "ONNX value '{}' uses dtype {elem_type}; ny verification only models FLOAT32 operation rounding",
                    info.name
                )));
            }
        }
    }
    for node in &graph.node {
        validate_standard_node_signature(node)?;
        for attribute in &node.attribute {
            if let Some(tensor) = &attribute.t {
                reject_tensor(
                    tensor,
                    &format!(
                        "tensor attribute '{}' on node '{}' ({})",
                        attribute.name, node.name, node.op_type
                    ),
                )?;
            }
        }
        if node.op_type == "Cast" && is_standard_onnx_domain(&node.domain) {
            let mut target_attributes = node
                .attribute
                .iter()
                .filter(|attribute| attribute.name == "to");
            let target_attribute = target_attributes.next().ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "ONNX Cast node '{}' is missing its required 'to' dtype",
                    node.name
                ))
            })?;
            if target_attributes.next().is_some() {
                return Err(NyError::ModelLoad(format!(
                    "ONNX Cast node '{}' has duplicate 'to' dtype attributes",
                    node.name
                )));
            }
            if target_attribute.r#type != onnx_proto::attribute_type::INT {
                return Err(NyError::ModelLoad(format!(
                    "ONNX Cast node '{}' has a non-INT 'to' dtype attribute",
                    node.name
                )));
            }
            let target = target_attribute.i_value();
            if !cast_target_semantics_are_modeled(target) {
                return Err(NyError::ModelLoad(format!(
                    "ONNX Cast node '{}' targets unsupported dtype {target}; only FLOAT32, \
                     INT32/INT64 and BOOL casts have semantics ny models exactly",
                    node.name
                )));
            }
        }
    }
    Ok(())
}

/// Require authored direct-normalization parameters to retain unambiguous raw
/// FLOAT32 provenance.  WeightStore intentionally mirrors integer and several
/// floating dtypes as f32, so this check must run before extraction/folding;
/// otherwise an INT64 initializer or a Cast/computed parameter could become
/// indistinguishable from an authored FLOAT tensor at conversion time.
fn validate_direct_normalization_parameter_provenance(
    graph: &onnx_proto::GraphProto,
) -> Result<()> {
    let float32_initializers: HashSet<&str> = graph
        .initializer
        .iter()
        .filter(|initializer| initializer.data_type == ONNX_TENSOR_FLOAT32)
        .map(|initializer| initializer.name.as_str())
        .collect();
    let float32_constant_outputs: HashSet<&str> = graph
        .node
        .iter()
        .filter(|node| {
            node.op_type == "Constant"
                && is_standard_onnx_domain(&node.domain)
                && node.input.is_empty()
                && node.output.len() == 1
                && !node.output[0].is_empty()
                && node.attribute.len() == 1
                && node.attribute[0].name == "value"
                && node.attribute[0].r#type == onnx_proto::attribute_type::TENSOR
                && node.attribute[0]
                    .t
                    .as_ref()
                    .is_some_and(|tensor| tensor.data_type == ONNX_TENSOR_FLOAT32)
        })
        .map(|node| node.output[0].as_str())
        .collect();

    for node in graph
        .node
        .iter()
        .filter(|node| is_standard_onnx_domain(&node.domain))
    {
        let indices: &[usize] = match node.op_type.as_str() {
            "BatchNormalization" => &[1, 2, 3, 4],
            "InstanceNormalization" | "GroupNormalization" => &[1, 2],
            "LayerNormalization" => {
                if node.input.get(2).is_some_and(|input| !input.is_empty()) {
                    &[1, 2]
                } else {
                    &[1]
                }
            }
            "SimplifiedLayerNormalization" | "RMSNormalization" => &[1],
            _ => continue,
        };
        for &index in indices {
            let Some(parameter_name) = node.input.get(index).filter(|name| !name.is_empty()) else {
                // Exact operator signatures are diagnosed by the schema gate.
                continue;
            };
            if !float32_initializers.contains(parameter_name.as_str())
                && !float32_constant_outputs.contains(parameter_name.as_str())
            {
                return Err(NyError::ModelLoad(format!(
                    "ONNX {} node '{}' parameter input {} ('{}') must be a direct FLOAT32 initializer or FLOAT32 Constant tensor output; computed, cast, runtime, and non-FLOAT32 parameters are unsupported",
                    node.op_type, node.name, index, parameter_name
                )));
            }
        }
    }
    Ok(())
}

fn validate_standard_node_signature(node: &onnx_proto::NodeProto) -> Result<()> {
    if !is_standard_onnx_domain(&node.domain) {
        return Ok(());
    }
    if node.op_type == "Gemm" {
        let valid_inputs = matches!(node.input.len(), 2 | 3)
            && node.input[0..2].iter().all(|name| !name.is_empty());
        if !valid_inputs
            || node.output.len() != 1
            || node.output.first().is_none_or(String::is_empty)
        {
            return Err(NyError::ModelLoad(format!(
                "standard ONNX Gemm node '{}' must have A and B, an optional C placeholder, and exactly one non-empty output; got inputs {:?} and outputs {:?}",
                node.name, node.input, node.output
            )));
        }

        let mut seen = HashSet::new();
        for attribute in &node.attribute {
            if !seen.insert(attribute.name.as_str()) {
                return Err(NyError::ModelLoad(format!(
                    "standard ONNX Gemm node '{}' has duplicate '{}' attributes",
                    node.name, attribute.name
                )));
            }
            match attribute.name.as_str() {
                "alpha" | "beta" if attribute.r#type == onnx_proto::attribute_type::FLOAT => {}
                "transA" | "transB"
                    if attribute.r#type == onnx_proto::attribute_type::INT
                        && matches!(attribute.i_value(), 0 | 1) => {}
                "alpha" | "beta" | "transA" | "transB" => {
                    return Err(NyError::ModelLoad(format!(
                        "standard ONNX Gemm node '{}' has malformed '{}' attribute",
                        node.name, attribute.name
                    )));
                }
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "standard ONNX Gemm node '{}' has unsupported attribute '{}'",
                        node.name, attribute.name
                    )));
                }
            }
        }
        return Ok(());
    }

    if node.op_type == "BatchNormalization" {
        if node.input.len() != 5
            || node.input.iter().any(String::is_empty)
            || node.output.is_empty()
            || node.output.len() > 5
            || node.output[0].is_empty()
            || node.output.iter().skip(1).any(|output| !output.is_empty())
        {
            return Err(NyError::ModelLoad(format!(
                "standard ONNX BatchNormalization node '{}' requires exactly five non-empty inference inputs and Y followed only by empty optional output placeholders; got inputs {:?} and outputs {:?}",
                node.name, node.input, node.output
            )));
        }

        let mut seen = HashSet::new();
        for attribute in &node.attribute {
            if !seen.insert(attribute.name.as_str()) {
                return Err(NyError::ModelLoad(format!(
                    "standard ONNX BatchNormalization node '{}' has duplicate '{}' attributes",
                    node.name, attribute.name
                )));
            }
            match attribute.name.as_str() {
                "epsilon"
                    if attribute.r#type == onnx_proto::attribute_type::FLOAT
                        && attribute.f_value().is_finite()
                        && attribute.f_value() >= 0.0 => {}
                // Momentum changes training-state updates only; it is inert for
                // the single inference output represented by ny.
                "momentum"
                    if attribute.r#type == onnx_proto::attribute_type::FLOAT
                        && attribute.f_value().is_finite() => {}
                "training_mode"
                    if attribute.r#type == onnx_proto::attribute_type::INT
                        && attribute.i_value() == 0 => {}
                "epsilon" | "momentum" | "training_mode" => {
                    return Err(NyError::ModelLoad(format!(
                        "standard ONNX BatchNormalization node '{}' has unsupported '{}' value or type for inference",
                        node.name, attribute.name
                    )));
                }
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "standard ONNX BatchNormalization node '{}' has unsupported attribute '{}'",
                        node.name, attribute.name
                    )));
                }
            }
        }
        return Ok(());
    }

    let expected_inputs = match node.op_type.as_str() {
        "Constant" => Some((0, 0)),
        "ConstantOfShape" => Some((1, 1)),
        "Sqrt" | "Neg" | "Sin" | "Cos" | "Abs" | "Relu" | "Sigmoid" | "Tanh" | "Exp" | "Log"
        | "Cast" | "Identity" | "Shape" => Some((1, 1)),
        "Pow" | "Div" | "Mul" | "Add" | "Sub" | "MatMul" | "Equal" | "Greater"
        | "GreaterOrEqual" | "Less" | "LessOrEqual" | "Reshape" | "Gather" => Some((2, 2)),
        "Where" => Some((3, 3)),
        "Squeeze" | "Unsqueeze" => Some((1, 2)),
        "Concat" => Some((1, usize::MAX)),
        _ => None,
    };
    let Some((minimum, maximum)) = expected_inputs else {
        return Ok(());
    };
    let valid_input_names = match node.op_type.as_str() {
        "Squeeze" | "Unsqueeze" => node.input.first().is_some_and(|name| !name.is_empty()),
        _ => node.input.iter().all(|name| !name.is_empty()),
    };
    if node.input.len() < minimum
        || node.input.len() > maximum
        || !valid_input_names
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        let expected = if minimum == maximum {
            format!("exactly {minimum}")
        } else if maximum == usize::MAX {
            format!("at least {minimum}")
        } else {
            format!("between {minimum} and {maximum}")
        };
        return Err(NyError::ModelLoad(format!(
            "standard ONNX {} node '{}' must have {expected} input(s) and exactly one non-empty output; got {} input(s) and {} output(s)",
            node.op_type,
            node.name,
            node.input.len(),
            node.output.len()
        )));
    }
    if node.op_type == "Constant" {
        validate_constant_payload_schema(node)?;
    } else if node.op_type == "ConstantOfShape" {
        validate_constant_of_shape_schema(node)?;
    }
    Ok(())
}

fn cast_target(node: &onnx_proto::NodeProto) -> Option<i64> {
    let mut targets = node
        .attribute
        .iter()
        .filter(|attribute| attribute.name == "to");
    let target = targets.next()?;
    (target.r#type == onnx_proto::attribute_type::INT && targets.next().is_none())
        .then_some(target.i_value())
}

/// Prove that every use of an INT64 Cast is confined to a shape-expression
/// chain and ends specifically at Reshape's target-shape operand.  This is a
/// syntactic admission check only; [`validate_skipped_nodes_are_semantic_identities`]
/// separately requires every value in the chain to have been materialized as
/// an exact i64 constant before graph conversion.
fn int64_cast_has_shape_only_uses(
    graph: &onnx_proto::GraphProto,
    cast: &onnx_proto::NodeProto,
) -> bool {
    if cast.op_type != "Cast"
        || !is_standard_onnx_domain(&cast.domain)
        || cast_target(cast) != Some(ONNX_TENSOR_INT64)
        || cast.input.len() != 1
        || cast.output.len() != 1
        || cast.output[0].is_empty()
    {
        return false;
    }
    shape_value_has_only_reshape_shape_uses(graph, &cast.output[0], &mut HashSet::new(), 0)
}

const MAX_SHAPE_ONLY_USE_DEPTH: usize = 256;

fn shape_value_has_only_reshape_shape_uses(
    graph: &onnx_proto::GraphProto,
    value: &str,
    visiting: &mut HashSet<String>,
    depth: usize,
) -> bool {
    if depth > MAX_SHAPE_ONLY_USE_DEPTH
        || graph.output.iter().any(|output| output.name == value)
        || !visiting.insert(value.to_string())
    {
        return false;
    }

    let mut saw_use = false;
    let valid = graph.node.iter().all(|consumer| {
        consumer
            .input
            .iter()
            .enumerate()
            .filter(|(_, input)| input.as_str() == value)
            .all(|(input_index, _)| {
                saw_use = true;
                if !shape_structural_node_has_valid_signature(consumer) {
                    return false;
                }
                match consumer.op_type.as_str() {
                    "Reshape" => input_index == 1,
                    "Cast" => {
                        input_index == 0
                            && cast_target(consumer) == Some(ONNX_TENSOR_INT64)
                            && node_outputs_have_only_reshape_shape_uses(
                                graph,
                                consumer,
                                visiting,
                                depth + 1,
                            )
                    }
                    "Identity" | "Squeeze" | "Unsqueeze" => {
                        input_index == 0
                            && node_outputs_have_only_reshape_shape_uses(
                                graph,
                                consumer,
                                visiting,
                                depth + 1,
                            )
                    }
                    "Concat" => node_outputs_have_only_reshape_shape_uses(
                        graph,
                        consumer,
                        visiting,
                        depth + 1,
                    ),
                    _ => false,
                }
            })
    });
    visiting.remove(value);
    saw_use && valid
}

fn node_outputs_have_only_reshape_shape_uses(
    graph: &onnx_proto::GraphProto,
    node: &onnx_proto::NodeProto,
    visiting: &mut HashSet<String>,
    depth: usize,
) -> bool {
    node.output.len() == 1
        && !node.output[0].is_empty()
        && shape_value_has_only_reshape_shape_uses(graph, &node.output[0], visiting, depth)
}

fn shape_structural_node_has_valid_signature(node: &onnx_proto::NodeProto) -> bool {
    if !is_standard_onnx_domain(&node.domain) || node.output.len() != 1 || node.output[0].is_empty()
    {
        return false;
    }
    match node.op_type.as_str() {
        "Reshape" => node.input.len() == 2,
        "Cast" | "Identity" => node.input.len() == 1 && !node.input[0].is_empty(),
        "Squeeze" | "Unsqueeze" => (1..=2).contains(&node.input.len()) && !node.input[0].is_empty(),
        "Concat" => !node.input.is_empty() && node.input.iter().all(|input| !input.is_empty()),
        _ => false,
    }
}

fn shape_only_int64_cast_path_is_materialized(
    graph: &onnx_proto::GraphProto,
    cast: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> bool {
    cast.output.len() == 1
        && shape_value_path_is_materialized(graph, &cast.output[0], weights, &mut HashSet::new(), 0)
}

fn shape_value_path_is_materialized(
    graph: &onnx_proto::GraphProto,
    value: &str,
    weights: &WeightStore,
    visiting: &mut HashSet<String>,
    depth: usize,
) -> bool {
    if depth > MAX_SHAPE_ONLY_USE_DEPTH
        || weights.get_integers(value).is_none()
        || !visiting.insert(value.to_string())
    {
        return false;
    }
    let valid = graph.node.iter().all(|consumer| {
        consumer
            .input
            .iter()
            .enumerate()
            .filter(|(_, input)| input.as_str() == value)
            .all(|(input_index, _)| {
                if !shape_structural_node_has_valid_signature(consumer) {
                    return false;
                }
                match consumer.op_type.as_str() {
                    "Reshape" => input_index == 1,
                    "Cast" | "Identity" | "Squeeze" | "Unsqueeze" | "Concat" => {
                        consumer.output.first().is_some_and(|output| {
                            shape_value_path_is_materialized(
                                graph,
                                output,
                                weights,
                                visiting,
                                depth + 1,
                            )
                        })
                    }
                    _ => false,
                }
            })
    });
    visiting.remove(value);
    valid
}

/// Remove the now-dead dtype operation only after the exact post-fold proof.
/// Rewriting to Identity keeps graph value names and producer topology intact,
/// while ensuring conversion drops only this authenticated shape-only form;
/// runtime integer Casts remain present and lower to guarded `Trunc` layers.
fn rewrite_materialized_int64_casts_as_identities(
    graph: &mut onnx_proto::GraphProto,
    weights: &WeightStore,
) {
    let proven_indices: Vec<usize> = graph
        .node
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            (int64_cast_has_shape_only_uses(graph, node)
                && shape_only_int64_cast_path_is_materialized(graph, node, weights))
            .then_some(index)
        })
        .collect();
    for index in proven_indices {
        let node = &mut graph.node[index];
        node.op_type = "Identity".to_string();
        node.attribute.clear();
    }
}

fn validate_skipped_nodes_are_semantic_identities(
    graph: &onnx_proto::GraphProto,
    weights: &WeightStore,
) -> Result<()> {
    let graph_output_names: HashSet<String> = graph
        .output
        .iter()
        .filter(|output| !output.name.is_empty())
        .map(|output| output.name.clone())
        .collect();
    let raw_int64_shape_values = raw_int64_shape_values(graph, weights);
    for (node_index, node) in graph.node.iter().enumerate() {
        if !is_standard_onnx_domain(&node.domain) {
            continue;
        }
        match node.op_type.as_str() {
            "Cast"
                if node
                    .attribute
                    .iter()
                    .any(|attribute| attribute.name == "to" && attribute.i_value() == 7) =>
            {
                // An INT64 Cast leaves `loader::convert` one of exactly two
                // ways:
                //
                //  1. DROPPED, producing no layer at all — admissible only for
                //     the exact static cGAN shape-use cone, where the value was
                //     proven and materialized by constant folding and is
                //     consumed solely as `Reshape` input 1;
                //  2. LOWERED to `LayerType::Trunc`, the exact f32→f32 reading
                //     of round-toward-zero, which is what cctsdb_yolo_2023's
                //     patch-position gates need and 25dee0c5 took away.
                //
                // Route 2 is exact ONLY when the value the rest of the graph
                // reads really is that integer. Two ways it is not, both
                // rejected here at the raw protobuf boundary — ahead of any
                // proto fusion, because a fusion is allowed to consume Cast
                // nodes (`convert_graph_to_layers` additionally refuses to let
                // one consume a non-FLOAT32 Cast, so route 2 cannot vanish
                // silently either).
                let materialized_output = node
                    .output
                    .iter()
                    .any(|output| !output.is_empty() && weights.contains_key(output));
                if materialized_output {
                    // Constant folding already baked an f32 into the network.
                    // `Trunc` cannot undo a rounded or fabricated payload, so
                    // demand the FULL raw-INT64 provenance proof: every value
                    // back to the authored initializers is exactly materialized
                    // (i64 payload == bit-identical f32 mirror, no private
                    // reshape sentinel) and authored INT64. That is what
                    // rejects a fractional FLOAT source, an integer-looking
                    // FLOAT source, a FLOAT `Gather` index, a FLOAT `Range`
                    // sidecar masquerading as INT64 provenance, a malformed
                    // Gather/Concat axis, and ny's copy-axis sentinel.
                    if !int64_cast_has_raw_int64_provenance(node, weights, &raw_int64_shape_values)
                    {
                        return Err(NyError::UnsupportedOp(format!(
                            "ONNX Cast node '{}' targets dtype 7 and was materialized by constant folding without proven raw INT64 provenance, so it is not an exact constant INT64 shape cone",
                            node.name
                        )));
                    }
                    // A Cast that disappears into a static Reshape-shape cone
                    // owes the audit branch's complete forward-use proof in
                    // addition to the raw backward-provenance proof above.
                    if int64_cast_has_shape_only_uses(graph, node)
                        && (!shape_only_int64_cast_path_is_materialized(graph, node, weights)
                            || !int64_cast_has_static_reshape_shape_use(
                                &graph.node,
                                node_index,
                                weights,
                                &graph_output_names,
                                &raw_int64_shape_values,
                            ))
                    {
                        return Err(NyError::UnsupportedOp(format!(
                            "ONNX Cast node '{}' targets dtype 7 but did not fold end-to-end to an exact constant INT64 shape cone ending only at Reshape input 1",
                            node.name
                        )));
                    }
                } else if let Some(output) = node.output.first().filter(|o| !o.is_empty()) {
                    // A value only known at RUNTIME cannot be a static shape.
                    // `ny-build` refuses `Reshape` with a non-constant shape
                    // tensor anyway. Check the full structural use cone, not
                    // just a direct Reshape consumer, so Identity/Concat cannot
                    // launder a dynamic shape before conversion.
                    if int64_cast_has_shape_only_uses(graph, node) {
                        return Err(NyError::UnsupportedOp(format!(
                            "ONNX Cast node '{}' targets dtype 7 on runtime value '{output}' and did not fold end-to-end to an exact constant INT64 shape cone ending only at Reshape input 1",
                            node.name
                        )));
                    }
                }
            }
            "Constant" | "ConstantOfShape" | "Range"
                if node.output.iter().all(String::is_empty)
                    || node
                        .output
                        .iter()
                        .filter(|output| !output.is_empty())
                        .any(|output| !weights.contains_key(output)) =>
            {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX {} node '{}' survived constant folding on a live data path",
                    node.op_type, node.name
                )));
            }
            "Constant" | "ConstantOfShape" | "Range" => {}
            // Two malformed-shape-op hazards the INT64 Cast cone used to police
            // indirectly (0184b7c9): a Cast is no longer the choke point once it
            // lowers to `Trunc`, so police them where they actually occur —
            // which also catches them when no Cast is involved at all. Both
            // shapes are invalid ONNX, and both make the constant folder's
            // answer depend on which spelling it happens to prefer.
            "Reshape" if node.input.len() > 2 => {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX Reshape node '{}' has {} inputs; the op takes exactly (data, shape), and \
                     a third operand makes the folded target shape ambiguous",
                    node.name,
                    node.input.len()
                )));
            }
            "Concat"
                if {
                    let mut axes = node
                        .attribute
                        .iter()
                        .filter(|attribute| attribute.name == "axis");
                    match axes.next() {
                        Some(axis) => {
                            axes.next().is_some() || axis.r#type != onnx_proto::attribute_type::INT
                        }
                        None => false,
                    }
                } =>
            {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX Concat node '{}' has a duplicate or non-INT `axis` attribute; the folded \
                     concatenation axis is then ambiguous",
                    node.name
                )));
            }
            "Unsqueeze"
                if node.input.len() > 1
                    && node
                        .attribute
                        .iter()
                        .any(|attribute| attribute.name == "axes") =>
            {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX Unsqueeze node '{}' carries both the opset-13 `axes` input and the \
                     opset-11 `axes` attribute; the two can disagree and the folded axis is then \
                     ambiguous",
                    node.name
                )));
            }
            "Dropout" => {
                let has_training_mode = node.input.get(2).is_some_and(|input| !input.is_empty());
                let has_mask_output = node.output.get(1).is_some_and(|output| !output.is_empty());
                if has_training_mode || has_mask_output {
                    return Err(NyError::UnsupportedOp(format!(
                        "ONNX Dropout node '{}' is only an identity when training_mode is absent and no mask output is requested",
                        node.name
                    )));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn is_non_f32_floating_dtype(dtype: i32) -> bool {
    matches!(dtype, 10 | 11 | 14..=20 | 23 | 24)
}

/// Whether ny models the value semantics of `Cast(to = dtype)` EXACTLY.
///
/// This is an admit-list, so an ONNX dtype ny has never seen fails closed.
/// Three families are admitted, and nothing else:
///
/// * **FLOAT32 (1)** — an exact identity in ny's all-f32 internal graph.
/// * **INT32 (6) and INT64 (7)** — the cast truncates toward zero, which
///   `LayerType::Trunc` reproduces EXACTLY as an f32→f32 map (`trunc` is
///   monotone, so `[trunc(lo), trunc(hi)]` is the exact interval hull; and for
///   `|x| >= 2^24` an f32 is already integral, so `trunc` is representable at
///   every magnitude). `crates/ny-onnx/src/loader/convert.rs` performs that
///   lowering; the const-fold path applies the same `trunc`
///   (`const_fold/ops/cast.rs`). The narrow and unsigned integer dtypes
///   (2,3,4,5,12,13) are NOT admitted: `trunc` is the ONNX reading only for
///   in-range values, out-of-range is undefined in the Cast spec and ONNX
///   Runtime answers with wraparound, and those dtypes are precisely the ones
///   a real graph overflows. INT32/INT64 need `|x| > 2^31`/`2^63` to get
///   there, which no shape/index/coordinate cone reaches.
/// * **BOOL (9)** — `x != 0`, NOT truncation and NOT an identity in general.
///   Admitted at this boundary only so a provably `{0,1}`-valued operand can be
///   recognized later; `convert.rs` re-checks each BOOL cast against its
///   producer and fails closed when it cannot prove the operand is already
///   `{0,1}`. The const-fold path materializes the indicator exactly.
///
/// Everything else stays rejected, and the rejection must stay HERE at the raw
/// protobuf boundary, ahead of constant folding: f16/bf16/double targets round
/// (or widen) with error the folder would silently erase into an ordinary
/// FLOAT constant, and DOUBLE additionally changes the arithmetic dtype of
/// every downstream op. That is the hole commit 25dee0c5 legitimately closed —
/// it is kept closed here — but that commit also swept INT32/INT64 and BOOL in,
/// and those ny does model exactly (#cctsdb B1).
fn cast_target_semantics_are_modeled(dtype: i64) -> bool {
    matches!(dtype, 1 | 6 | 7 | 9)
}

fn validate_graph_value_names(
    graph: &onnx_proto::GraphProto,
    initializer_names: &HashSet<String>,
) -> Result<()> {
    let mut input_names = HashSet::new();
    for input in &graph.input {
        if input.name.is_empty() {
            return Err(NyError::ModelLoad(
                "ONNX graph input name cannot be empty".to_string(),
            ));
        }
        if !input_names.insert(input.name.as_str()) {
            return Err(NyError::ModelLoad(format!(
                "duplicate ONNX graph input name '{}'",
                input.name
            )));
        }
    }

    let mut output_names = HashSet::new();
    for output in &graph.output {
        if output.name.is_empty() {
            return Err(NyError::ModelLoad(
                "ONNX graph output name cannot be empty".to_string(),
            ));
        }
        if !output_names.insert(output.name.as_str()) {
            return Err(NyError::ModelLoad(format!(
                "duplicate ONNX graph output name '{}'",
                output.name
            )));
        }
    }

    let mut value_info_names = HashSet::new();
    for info in graph.value_info() {
        if info.name.is_empty() {
            return Err(NyError::ModelLoad(
                "ONNX intermediate value_info name cannot be empty".to_string(),
            ));
        }
        if !value_info_names.insert(info.name.as_str()) {
            return Err(NyError::ModelLoad(format!(
                "duplicate ONNX intermediate value_info name '{}'",
                info.name
            )));
        }
    }

    let mut node_output_names = HashSet::new();
    for node in &graph.node {
        for output in node.output.iter().filter(|output| !output.is_empty()) {
            if initializer_names.contains(output) {
                return Err(NyError::ModelLoad(format!(
                    "ONNX initializer '{}' collides with output of node '{}' ({})",
                    output, node.name, node.op_type
                )));
            }
            if input_names.contains(output.as_str()) {
                return Err(NyError::ModelLoad(format!(
                    "ONNX graph input '{}' collides with output of node '{}' ({})",
                    output, node.name, node.op_type
                )));
            }
            if !node_output_names.insert(output.as_str()) {
                return Err(NyError::ModelLoad(format!(
                    "duplicate ONNX node output value '{}' at node '{}' ({})",
                    output, node.name, node.op_type
                )));
            }
        }
    }
    Ok(())
}

fn infer_concat_reshape_shapes(graph: &onnx_proto::GraphProto, weights: &mut WeightStore) {
    // Build node output lookup for Reshape shape inference.
    let node_by_output: HashMap<&str, &onnx_proto::NodeProto> = graph
        .node
        .iter()
        .flat_map(|node| {
            node.output
                .iter()
                .filter(|output| !output.is_empty())
                .map(move |output| (output.as_str(), node))
        })
        .collect();

    // Infer shapes for Reshape nodes where shape comes from Concat of known values.
    // This handles ViT-style patterns: Shape -> Gather -> Unsqueeze -> Concat -> Reshape.
    for node in &graph.node {
        let Some((shape_input, concat_node)) =
            missing_concat_reshape_shape(node, weights, &node_by_output)
        else {
            continue;
        };
        let Some((tensor, all_known)) = infer_concat_shape_tensor(concat_node, weights) else {
            continue;
        };
        // A zero in an ONNX Reshape shape is not an arbitrary unknown: it
        // copies the data input's dimension at the same axis.  Only materialize
        // the Concat when every scalar operand was evaluated exactly.  A
        // missing runtime scalar may have any value and cannot soundly be
        // replaced by that copy-axis sentinel.
        if !all_known {
            continue;
        }

        debug!(
            "Inferred Reshape shape from Concat: {} -> {:?} (all_known: {})",
            shape_input, tensor.float_data, all_known
        );
        insert_loaded_tensor(weights, shape_input.to_string(), tensor);
    }
}

fn missing_concat_reshape_shape<'a>(
    node: &'a onnx_proto::NodeProto,
    weights: &WeightStore,
    node_by_output: &HashMap<&'a str, &'a onnx_proto::NodeProto>,
) -> Option<(&'a str, &'a onnx_proto::NodeProto)> {
    if node.op_type != "Reshape"
        || !is_standard_onnx_domain(&node.domain)
        || node.input.len() != 2
        || node.output.len() != 1
    {
        return None;
    }

    let shape_input = node.input.get(1)?.as_str();
    if weights.contains_key(shape_input) {
        return None;
    }

    let concat_node = node_by_output.get(shape_input)?;
    (concat_node.op_type == "Concat"
        && is_standard_onnx_domain(&concat_node.domain)
        && !concat_node.input.is_empty()
        && concat_node.input.iter().all(|input| !input.is_empty())
        && concat_node.output.len() == 1)
        .then_some((shape_input, *concat_node))
}

fn infer_concat_shape_tensor(
    concat_node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<(LoadedTensor, bool)> {
    let mut inferred_shape = Vec::new();
    let mut inferred_shape_i64 = Some(Vec::new());
    let mut all_known = true;

    for concat_input in concat_node.input.iter().filter(|input| !input.is_empty()) {
        let (dim, exact_dim, is_known) = infer_concat_shape_dim(weights, concat_input)?;
        inferred_shape.push(dim);
        all_known &= is_known;

        match (inferred_shape_i64.as_mut(), exact_dim) {
            (Some(inferred_shape_i64), Some(exact_dim)) => inferred_shape_i64.push(exact_dim),
            (Some(_), None) => inferred_shape_i64 = None,
            (None, _) => {}
        }
    }

    (!inferred_shape.is_empty()).then(|| {
        let float_data = ArrayD::from_shape_vec(IxDyn(&[inferred_shape.len()]), inferred_shape)
            .unwrap_or_else(|_| ArrayD::from_elem(IxDyn(&[0]), 0.0));
        let integer_data = inferred_shape_i64.and_then(|inferred_shape_i64| {
            ArrayD::from_shape_vec(IxDyn(&[inferred_shape_i64.len()]), inferred_shape_i64).ok()
        });
        (
            LoadedTensor {
                float_data,
                integer_data,
                integer_range: None,
            },
            all_known,
        )
    })
}

fn infer_concat_shape_dim(
    weights: &WeightStore,
    concat_input: &str,
) -> Option<(f32, Option<i64>, bool)> {
    if let Some(value) = weights.get_integers(concat_input) {
        if value.len() != 1 {
            return None;
        }
        if let Some(float_value) = weights.get(concat_input) {
            if float_value.shape() != value.shape() {
                return None;
            }
        }
        let dim = value.iter().next().copied()?;
        return Some((
            i64_to_f32_checked(dim, "prepare_graph inferred reshape shape")
                .unwrap_or_else(|_| i64_to_f32_warned(dim, "prepare_graph inferred reshape shape")),
            Some(dim),
            true,
        ));
    }

    if let Some(value) = weights.get(concat_input) {
        if value.len() != 1 {
            return None;
        }
        let dim = value.iter().next().copied()?;
        return Some((dim, parse_shape_scalar_i64(dim), true));
    }

    // Use 0 to preserve a dynamic dimension in ONNX Reshape.
    Some((0.0, Some(0), false))
}

fn parse_shape_scalar_i64(value: f32) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    // A reshape dimension is discrete.  Approximate matching silently changes
    // the graph for an adjacent non-integral f32, so require exact integrality.
    if value != rounded {
        return None;
    }
    if rounded < i64::MIN as f32 || rounded >= i64::MAX as f32 {
        return None;
    }
    Some(rounded as i64)
}

pub(super) fn lower_reduce_l2_nodes(nodes: &mut Vec<onnx_proto::NodeProto>) {
    let mut lowered = Vec::with_capacity(nodes.len());

    for node in nodes.drain(..) {
        if node.op_type != "ReduceL2"
            || !is_standard_onnx_domain(&node.domain)
            || node.input.is_empty()
            || node.output.is_empty()
        {
            lowered.push(node);
            continue;
        }

        let base_name = if node.name.is_empty() {
            node.output[0].clone()
        } else {
            node.name.clone()
        };
        let input = node.input[0].clone();
        let output = node.output[0].clone();
        let domain = node.domain.clone();
        let reduce_attrs = node.attribute.clone();
        let square_output = format!("{base_name}__reduce_l2_square");
        let exponent_output = format!("{base_name}__reduce_l2_exponent");
        let sum_output = format!("{base_name}__reduce_l2_sum");

        lowered.push(onnx_proto::NodeProto {
            input: Vec::new(),
            output: vec![exponent_output.clone()],
            name: format!("{base_name}__reduce_l2_exponent"),
            op_type: "Constant".to_string(),
            domain: domain.clone(),
            attribute: vec![onnx_proto::AttributeProto {
                name: "value_float".to_string(),
                f: Some(2.0),
                r#type: onnx_proto::attribute_type::FLOAT,
                ..Default::default()
            }],
        });
        lowered.push(onnx_proto::NodeProto {
            input: vec![input, exponent_output],
            output: vec![square_output.clone()],
            name: format!("{base_name}__reduce_l2_pow"),
            op_type: "Pow".to_string(),
            domain: domain.clone(),
            attribute: Vec::new(),
        });
        lowered.push(onnx_proto::NodeProto {
            input: vec![square_output],
            output: vec![sum_output.clone()],
            name: format!("{base_name}__reduce_l2_sum"),
            op_type: "ReduceSum".to_string(),
            domain: domain.clone(),
            attribute: reduce_attrs,
        });
        lowered.push(onnx_proto::NodeProto {
            input: vec![sum_output],
            output: vec![output],
            name: format!("{base_name}__reduce_l2_sqrt"),
            op_type: "Sqrt".to_string(),
            domain,
            attribute: Vec::new(),
        });
    }

    *nodes = lowered;
}

fn extract_constant_nodes_as_weights(
    nodes: &[onnx_proto::NodeProto],
    weights: &mut WeightStore,
) -> Result<()> {
    for node in nodes {
        if node.op_type != "Constant" || !is_standard_onnx_domain(&node.domain) {
            continue;
        }
        if node.output.len() != 1 {
            debug!(
                "Skipping Constant node {} with {} outputs",
                node.name,
                node.output.len()
            );
            continue;
        }
        if let Some(output_name) = node.output.first().filter(|name| !name.is_empty()) {
            if let Some(tensor) = extract_constant_tensor(node)? {
                reject_authored_reshape_copy_sentinels(output_name, &tensor)?;
                debug!(
                    "Loaded Constant node: {} shape {:?}",
                    output_name,
                    tensor.float_data.shape()
                );
                insert_loaded_tensor(weights, output_name.clone(), tensor);
            }
        }
    }
    Ok(())
}

fn reject_authored_reshape_copy_sentinels(name: &str, tensor: &LoadedTensor) -> Result<()> {
    let Some(integer_data) = tensor.integer_data.as_ref() else {
        return Ok(());
    };
    if let Some(value) = integer_data
        .iter()
        .copied()
        .find(|&value| ny_core::reshape_copy_axis_from_sentinel(value).is_some())
    {
        return Err(NyError::ModelLoad(format!(
            "authored integer tensor '{name}' contains reserved internal Reshape sentinel value {value}; it cannot participate in an exact constant INT64 shape cone"
        )));
    }
    Ok(())
}

fn insert_loaded_tensor(weights: &mut WeightStore, name: String, tensor: LoadedTensor) {
    let LoadedTensor {
        float_data,
        integer_data,
        integer_range,
    } = tensor;
    if let Some(integer_data) = integer_data {
        weights.insert_integers(name.clone(), integer_data);
    }
    if let Some((min, max)) = integer_range {
        weights.insert_integer_range(name.clone(), min, max);
    }
    weights.insert(name, float_data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::{
        attribute_type, tensor_shape_proto, AttributeProto, GraphProto, NodeProto, TensorProto,
        TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use crate::WeightStore;
    use std::collections::HashMap;

    fn gemm_node_for_signature_test() -> NodeProto {
        NodeProto {
            name: "gemm".to_string(),
            op_type: "Gemm".to_string(),
            input: vec!["a".to_string(), "b".to_string(), String::new()],
            output: vec!["y".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn gemm_raw_signature_preserves_optional_c_and_rejects_fail_open_forms() {
        let valid = gemm_node_for_signature_test();
        validate_standard_node_signature(&valid).expect("empty optional C is valid");

        let mut extra_input = valid.clone();
        extra_input.input.push("ignored".to_string());
        assert!(validate_standard_node_signature(&extra_input).is_err());

        let mut wrong_type = valid.clone();
        wrong_type.attribute.push(AttributeProto {
            name: "alpha".to_string(),
            i: Some(1),
            r#type: attribute_type::INT,
            ..Default::default()
        });
        assert!(validate_standard_node_signature(&wrong_type).is_err());

        let mut non_boolean = valid.clone();
        non_boolean.attribute.push(AttributeProto {
            name: "transB".to_string(),
            i: Some(2),
            r#type: attribute_type::INT,
            ..Default::default()
        });
        assert!(validate_standard_node_signature(&non_boolean).is_err());

        let mut duplicate = valid.clone();
        for _ in 0..2 {
            duplicate.attribute.push(AttributeProto {
                name: "beta".to_string(),
                f: Some(1.0),
                r#type: attribute_type::FLOAT,
                ..Default::default()
            });
        }
        assert!(validate_standard_node_signature(&duplicate).is_err());

        let mut unknown = valid;
        unknown.attribute.push(AttributeProto {
            name: "ignored".to_string(),
            f: Some(1.0),
            r#type: attribute_type::FLOAT,
            ..Default::default()
        });
        assert!(validate_standard_node_signature(&unknown).is_err());
    }

    #[test]
    fn batch_norm_raw_signature_accepts_only_inference_semantics() {
        let valid = NodeProto {
            name: "bn".to_string(),
            op_type: "BatchNormalization".to_string(),
            input: ["x", "scale", "bias", "mean", "var"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            output: vec!["y".to_string()],
            ..Default::default()
        };
        validate_standard_node_signature(&valid).expect("default inference BN is supported");

        let mut empty_optional_outputs = valid.clone();
        empty_optional_outputs.output.extend([
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ]);
        validate_standard_node_signature(&empty_optional_outputs).expect(
            "raw preflight should preserve legal empty output placeholders for opset validation",
        );

        let mut training = valid.clone();
        training.attribute.push(AttributeProto {
            name: "training_mode".to_string(),
            i: Some(1),
            r#type: attribute_type::INT,
            ..Default::default()
        });
        assert!(validate_standard_node_signature(&training).is_err());

        let mut statistics_output = valid.clone();
        statistics_output.output.push("running_mean".to_string());
        assert!(validate_standard_node_signature(&statistics_output).is_err());

        let mut malformed_epsilon = valid.clone();
        malformed_epsilon.attribute.push(AttributeProto {
            name: "epsilon".to_string(),
            f: Some(-1.0),
            r#type: attribute_type::FLOAT,
            ..Default::default()
        });
        assert!(validate_standard_node_signature(&malformed_epsilon).is_err());

        let mut duplicate = valid;
        for _ in 0..2 {
            duplicate.attribute.push(AttributeProto {
                name: "momentum".to_string(),
                f: Some(0.9),
                r#type: attribute_type::FLOAT,
                ..Default::default()
            });
        }
        assert!(validate_standard_node_signature(&duplicate).is_err());
    }

    #[test]
    fn inferred_reshape_dimension_requires_exact_integrality() {
        assert_eq!(parse_shape_scalar_i64(1.0), Some(1));
        assert_eq!(
            parse_shape_scalar_i64(f32::from_bits(1.0_f32.to_bits() - 1)),
            None
        );
        assert_eq!(
            parse_shape_scalar_i64(f32::from_bits(1.0_f32.to_bits() + 1)),
            None
        );
        assert_eq!(parse_shape_scalar_i64(f32::NAN), None);
        assert_eq!(parse_shape_scalar_i64(f32::INFINITY), None);
    }

    #[test]
    fn inferred_concat_shape_requires_single_element_matching_payloads() {
        let mut weights = WeightStore::new();
        weights.insert(
            "float_multi".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        );
        weights.insert(
            "integer_multi".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        );
        weights.insert_integers(
            "integer_multi".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1, 2]).unwrap(),
        );
        weights.insert(
            "mismatched".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap(),
        );
        weights.insert_integers(
            "mismatched".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1]).unwrap(),
        );
        weights.insert(
            "exact".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![16_777_216.0]).unwrap(),
        );
        weights.insert_integers(
            "exact".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![16_777_217]).unwrap(),
        );

        assert_eq!(infer_concat_shape_dim(&weights, "float_multi"), None);
        assert_eq!(infer_concat_shape_dim(&weights, "integer_multi"), None);
        assert_eq!(infer_concat_shape_dim(&weights, "mismatched"), None);

        let (_, exact_dim, is_known) =
            infer_concat_shape_dim(&weights, "exact").expect("matching scalar payloads");
        assert_eq!(exact_dim, Some(16_777_217));
        assert!(is_known);

        let concat = NodeProto {
            input: vec!["exact".to_string(), "integer_multi".to_string()],
            output: vec!["shape".to_string()],
            op_type: "Concat".to_string(),
            ..Default::default()
        };
        assert!(infer_concat_shape_tensor(&concat, &weights).is_none());
    }

    #[test]
    fn unresolved_concat_scalar_does_not_materialize_reshape_shape() {
        let graph = GraphProto {
            node: vec![
                NodeProto {
                    input: vec!["runtime_dim".to_string(), "minus_one".to_string()],
                    output: vec!["shape".to_string()],
                    op_type: "Concat".to_string(),
                    ..Default::default()
                },
                NodeProto {
                    input: vec!["data".to_string(), "shape".to_string()],
                    output: vec!["reshaped".to_string()],
                    op_type: "Reshape".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        weights.insert(
            "minus_one".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
        );
        weights.insert_integers(
            "minus_one".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1_i64]).unwrap(),
        );

        infer_concat_reshape_shapes(&graph, &mut weights);

        assert!(!weights.contains_key("shape"));
    }

    #[test]
    fn exact_concat_scalars_still_materialize_reshape_shape() {
        let graph = GraphProto {
            node: vec![
                NodeProto {
                    input: vec!["three".to_string(), "minus_one".to_string()],
                    output: vec!["shape".to_string()],
                    op_type: "Concat".to_string(),
                    ..Default::default()
                },
                NodeProto {
                    input: vec!["data".to_string(), "shape".to_string()],
                    output: vec!["reshaped".to_string()],
                    op_type: "Reshape".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        for (name, value) in [("three", 3_i64), ("minus_one", -1_i64)] {
            weights.insert(
                name.to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![value as f32]).unwrap(),
            );
            weights.insert_integers(
                name.to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![value]).unwrap(),
            );
        }

        infer_concat_reshape_shapes(&graph, &mut weights);

        assert_eq!(
            weights
                .get_integers("shape")
                .expect("exact Concat shape should materialize")
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![3, -1]
        );
    }

    fn tensor_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
        let dims = shape
            .iter()
            .map(|dim| tensor_shape_proto::Dimension {
                value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
            })
            .collect();
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto { dim: dims }),
                }),
            }),
        }
    }

    fn int64_initializer(name: &str, dims: &[i64], values: &[i64]) -> TensorProto {
        let mut raw_data = Vec::new();
        for value in values {
            raw_data.extend_from_slice(&value.to_le_bytes());
        }
        TensorProto {
            dims: dims.to_vec(),
            data_type: 7,
            name: name.to_string(),
            raw_data,
            float_data: Vec::new(),
            ..Default::default()
        }
    }

    fn float32_initializer(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
        TensorProto {
            dims: dims.to_vec(),
            data_type: ONNX_TENSOR_FLOAT32,
            name: name.to_string(),
            float_data: values.to_vec(),
            ..Default::default()
        }
    }

    fn node(
        name: &str,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: Vec<AttributeProto>,
    ) -> NodeProto {
        NodeProto {
            input: inputs.iter().map(|value| value.to_string()).collect(),
            output: outputs.iter().map(|value| value.to_string()).collect(),
            op_type: op_type.to_string(),
            name: name.to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    fn attr_int(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: attribute_type::INT,
            ..Default::default()
        }
    }

    #[test]
    fn direct_normalization_parameters_require_raw_float32_sources() {
        let bn = node(
            "bn",
            "BatchNormalization",
            &["x", "scale", "bias", "mean", "var"],
            &["y"],
            Vec::new(),
        );
        let make_initializers = || {
            ["scale", "bias", "mean", "var"]
                .into_iter()
                .map(|name| float32_initializer(name, &[2], &[1.0, 1.0]))
                .collect::<Vec<_>>()
        };
        let valid = GraphProto {
            node: vec![bn],
            initializer: make_initializers(),
            ..Default::default()
        };
        validate_direct_normalization_parameter_provenance(&valid)
            .expect("direct FLOAT32 initializers should authenticate");

        let mut integer_parameter = valid.clone();
        integer_parameter.initializer[3] = int64_initializer("var", &[2], &[1, 1]);
        assert!(
            validate_direct_normalization_parameter_provenance(&integer_parameter).is_err(),
            "an INT64 parameter must not acquire FLOAT32 provenance through WeightStore"
        );

        let mut computed_parameter = valid.clone();
        computed_parameter
            .initializer
            .retain(|initializer| initializer.name != "var");
        computed_parameter.node.insert(
            0,
            node(
                "computed_var",
                "Add",
                &["mean", "mean"],
                &["var"],
                Vec::new(),
            ),
        );
        assert!(
            validate_direct_normalization_parameter_provenance(&computed_parameter).is_err(),
            "computed normalization parameters require explicit provenance tracking"
        );

        let mut constant_parameter = valid;
        constant_parameter
            .initializer
            .retain(|initializer| initializer.name != "var");
        constant_parameter.node.insert(
            0,
            node(
                "constant_var",
                "Constant",
                &[],
                &["var"],
                vec![AttributeProto {
                    name: "value".to_string(),
                    r#type: attribute_type::TENSOR,
                    t: Some(float32_initializer("", &[2], &[1.0, 1.0])),
                    ..Default::default()
                }],
            ),
        );
        validate_direct_normalization_parameter_provenance(&constant_parameter)
            .expect("a direct FLOAT32 Constant tensor output should authenticate");
    }

    #[test]
    #[cfg(feature = "onnx-value-info")]
    fn dtype_preflight_rejects_non_f32_arithmetic_before_folding() {
        let cases = [
            GraphProto {
                initializer: vec![TensorProto {
                    name: "double_weight".to_string(),
                    dims: vec![1],
                    data_type: 11,
                    double_data: vec![1.0],
                    ..Default::default()
                }],
                ..Default::default()
            },
            GraphProto {
                node: vec![node(
                    "half_constant",
                    "Constant",
                    &[],
                    &["half"],
                    vec![AttributeProto {
                        name: "value".to_string(),
                        r#type: attribute_type::TENSOR,
                        t: Some(TensorProto {
                            name: "half_payload".to_string(),
                            dims: vec![1],
                            data_type: 10,
                            int32_data: vec![0x3c00],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }],
                )],
                ..Default::default()
            },
            GraphProto {
                value_info: vec![ValueInfoProto {
                    name: "double_intermediate".to_string(),
                    r#type: Some(TypeProto {
                        tensor_type: Some(TensorTypeProto {
                            elem_type: 11,
                            shape: Some(TensorShapeProto { dim: Vec::new() }),
                        }),
                    }),
                }],
                ..Default::default()
            },
        ];

        for graph in cases {
            let error = validate_computational_dtypes(&graph)
                .expect_err("non-FLOAT32 arithmetic must fail before constant folding");
            assert!(
                error
                    .to_string()
                    .contains("only models FLOAT32 operation rounding"),
                "{error}"
            );
        }
    }

    #[test]
    fn dtype_preflight_rejects_unmodeled_cast_targets_and_missing_target() {
        // UINT8(2), INT8(3), UINT16(4), INT16(5), STRING(8), FLOAT16(10),
        // DOUBLE(11), UINT32(12), UINT64(13), COMPLEX64/128(14/15),
        // BFLOAT16(16), FLOAT8*(17..20), 4-bit(21..23) and any unknown dtype
        // have no exact f32 reading ny models, and the rejection must land
        // here — before constant folding — so a fold cannot launder the
        // rounding into a plain FLOAT constant.
        for target in [
            Some(2_i64),
            Some(3),
            Some(4),
            Some(5),
            Some(8),
            Some(10),
            Some(11),
            Some(12),
            Some(13),
            Some(14),
            Some(15),
            Some(16),
            Some(17),
            Some(21),
            Some(999),
            None,
        ] {
            let attrs = target
                .map(|target| vec![attr_int("to", target)])
                .unwrap_or_default();
            let graph = GraphProto {
                node: vec![node("cast", "Cast", &["x"], &["y"], attrs)],
                ..Default::default()
            };
            let error = validate_computational_dtypes(&graph)
                .expect_err("unmodeled Cast must fail before folding");
            assert!(error.to_string().contains("Cast node 'cast'"), "{error}");
        }
    }

    /// FLOAT32 (identity), INT32/INT64 (exact trunc-toward-zero) and BOOL
    /// (`x != 0`, re-proved against the operand's producer in
    /// `loader::convert`) all have semantics ny models exactly, so the
    /// preflight must admit them. cctsdb_yolo_2023 carries INT64 and BOOL
    /// casts and previously loaded; cgan's small_transformer carries the
    /// exactly-constant-folded INT64 shape cone.
    #[test]
    fn dtype_preflight_admits_float32_int32_int64_and_bool_cast_targets() {
        for target in [1_i64, 6, 7, 9] {
            let graph = GraphProto {
                node: vec![node(
                    "cast",
                    "Cast",
                    &["x"],
                    &["y"],
                    vec![attr_int("to", target)],
                )],
                ..Default::default()
            };
            validate_computational_dtypes(&graph)
                .unwrap_or_else(|error| panic!("target {target} should be modeled: {error}"));
        }

        // Duplicate or mistyped `to` attributes still fail closed: the admit
        // list is only consulted once a single INT-typed target is proven.
        for attrs in [
            vec![attr_int("to", 1), attr_int("to", 6)],
            vec![AttributeProto {
                name: "to".to_string(),
                i: Some(7),
                r#type: attribute_type::FLOAT,
                ..Default::default()
            }],
        ] {
            let graph = GraphProto {
                node: vec![node("cast", "Cast", &["x"], &["y"], attrs)],
                ..Default::default()
            };
            let error = validate_computational_dtypes(&graph)
                .expect_err("duplicate or mistyped Cast targets must fail closed");
            assert!(error.to_string().contains("Cast node 'cast'"), "{error}");
        }
    }

    #[test]
    fn dtype_preflight_admits_runtime_int64_while_shape_proof_remains_narrow() {
        let shape_only = GraphProto {
            node: vec![
                node(
                    "cast_shape",
                    "Cast",
                    &["constant_extent"],
                    &["shape"],
                    vec![attr_int("to", ONNX_TENSOR_INT64)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["activation", "shape"],
                    &["output"],
                    vec![],
                ),
            ],
            ..Default::default()
        };
        validate_computational_dtypes(&shape_only)
            .expect("an INT64 Cast used only as a Reshape target may proceed to folding");
        assert!(
            int64_cast_has_shape_only_uses(&shape_only, &shape_only.node[0]),
            "the separate static-shape proof must recognize the direct Reshape shape cone"
        );

        let activation_use = GraphProto {
            node: vec![
                node(
                    "cast_data",
                    "Cast",
                    &["activation"],
                    &["integer_activation"],
                    vec![attr_int("to", ONNX_TENSOR_INT64)],
                ),
                node(
                    "add",
                    "Add",
                    &["integer_activation", "bias"],
                    &["output"],
                    vec![],
                ),
            ],
            ..Default::default()
        };
        validate_computational_dtypes(&activation_use)
            .expect("guarded runtime INT64 casts are modeled by the Trunc lowering");
        assert!(
            !int64_cast_has_shape_only_uses(&activation_use, &activation_use.node[0]),
            "an activation data path must never acquire the static Reshape-shape exemption"
        );
    }

    #[test]
    fn dtype_preflight_rejects_malformed_standard_node_signatures() {
        let cases = [
            node("add", "Add", &["a", "b", "c"], &["out"], vec![]),
            node("neg", "Neg", &["a", "b"], &["out"], vec![]),
            node("where", "Where", &["a", "b", "c", "d"], &["out"], vec![]),
            node("empty_input", "Add", &["a", ""], &["out"], vec![]),
        ];
        for malformed in cases {
            let graph = GraphProto {
                node: vec![malformed],
                ..Default::default()
            };
            let error = validate_computational_dtypes(&graph)
                .expect_err("malformed standard node must fail before folding/conversion");
            assert!(error.to_string().contains("must have"), "{error}");
        }
    }

    #[test]
    fn shape_only_int64_cast_proof_rejects_custom_domain_structural_consumer() {
        let mut custom_concat = node(
            "custom_concat",
            "Concat",
            &["cast_shape"],
            &["joined"],
            vec![attr_int("axis", 0)],
        );
        custom_concat.domain = "vendor.example".to_string();
        let graph = GraphProto {
            node: vec![
                node(
                    "cast_shape",
                    "Cast",
                    &["extent"],
                    &["cast_shape"],
                    vec![attr_int("to", ONNX_TENSOR_INT64)],
                ),
                custom_concat,
                node(
                    "reshape",
                    "Reshape",
                    &["activation", "joined"],
                    &["output"],
                    vec![],
                ),
            ],
            ..Default::default()
        };
        validate_computational_dtypes(&graph)
            .expect("Cast target preflight is separate from structural-use authentication");
        assert!(
            !int64_cast_has_shape_only_uses(&graph, &graph.node[0]),
            "a custom-domain lookalike must not enter the standard static-shape proof"
        );
    }

    #[test]
    fn skipped_node_validation_rejects_unfolded_shape_int64_cast() {
        let graph = GraphProto {
            node: vec![
                node(
                    "cast_shape",
                    "Cast",
                    &["dynamic_extent"],
                    &["shape"],
                    vec![attr_int("to", ONNX_TENSOR_INT64)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["activation", "shape"],
                    &["output"],
                    vec![],
                ),
            ],
            ..Default::default()
        };
        let error = validate_skipped_nodes_are_semantic_identities(&graph, &WeightStore::new())
            .expect_err("shape-only admission must still require exact constant folding");
        assert!(
            error.to_string().contains("did not fold end-to-end"),
            "{error}"
        );
    }

    #[test]
    fn constant_extraction_ignores_empty_and_custom_domain_outputs() {
        let mut custom = node(
            "custom_constant",
            "Constant",
            &[],
            &["custom_out"],
            vec![AttributeProto {
                name: "value_int".to_string(),
                i: Some(7),
                r#type: attribute_type::INT,
                ..Default::default()
            }],
        );
        custom.domain = "vendor.example".to_string();
        let empty = node(
            "empty_constant",
            "Constant",
            &[],
            &[""],
            vec![AttributeProto {
                name: "value_int".to_string(),
                i: Some(9),
                r#type: attribute_type::INT,
                ..Default::default()
            }],
        );
        let mut weights = WeightStore::new();

        extract_constant_nodes_as_weights(&[custom, empty], &mut weights)
            .expect("ignored constants should not error");

        assert!(!weights.contains_key("custom_out"));
        assert!(!weights.contains_key(""));
    }

    #[test]
    fn dtype_preflight_rejects_non_float_runtime_inputs() {
        let mut integer_input = tensor_value_info("indices", &[1]);
        integer_input
            .r#type
            .as_mut()
            .and_then(|ty| ty.tensor_type.as_mut())
            .expect("tensor type")
            .elem_type = 7;
        let graph = GraphProto {
            input: vec![integer_input],
            ..Default::default()
        };
        let error = validate_computational_dtypes(&graph)
            .expect_err("integer runtime inputs must fail closed");
        assert!(error.to_string().contains("only FLOAT32 inputs"), "{error}");
    }

    #[test]
    fn prepare_graph_rejects_authored_private_reshape_sentinels() {
        let sentinel = ny_core::reshape_copy_axis_sentinel(2).expect("axis sentinel");
        let initializer_graph = GraphProto {
            initializer: vec![int64_initializer("reserved_initializer", &[1], &[sentinel])],
            ..Default::default()
        };
        let constant_graph = GraphProto {
            node: vec![node(
                "reserved_constant_node",
                "Constant",
                &[],
                &["reserved_constant"],
                vec![AttributeProto {
                    name: "value".to_string(),
                    r#type: attribute_type::TENSOR,
                    t: Some(int64_initializer("reserved_payload", &[1], &[sentinel])),
                    ..Default::default()
                }],
            )],
            ..Default::default()
        };

        for mut graph in [initializer_graph, constant_graph] {
            let error = prepare_graph(
                &mut graph,
                &mut WeightStore::new(),
                &mut HashMap::new(),
                false,
                None,
            )
            .expect_err("authored values must not collide with ny's private Reshape encoding");
            assert!(
                error
                    .to_string()
                    .contains("reserved internal Reshape sentinel"),
                "{error}"
            );
        }
    }

    #[test]
    fn skipped_node_validation_requires_materialization_or_true_identity() {
        for op_type in ["Constant", "ConstantOfShape", "Range"] {
            let graph = GraphProto {
                node: vec![node("producer", op_type, &["dynamic"], &["value"], vec![])],
                ..Default::default()
            };
            let error = validate_skipped_nodes_are_semantic_identities(&graph, &WeightStore::new())
                .expect_err("live constant producer must not be dropped");
            assert!(error.to_string().contains("live data path"), "{error}");
        }

        let mut weights = WeightStore::new();
        weights.insert("value".to_string(), ArrayD::from_elem(IxDyn(&[1]), 1.0));
        let materialized = GraphProto {
            node: vec![node("range", "Range", &["a", "b", "c"], &["value"], vec![])],
            ..Default::default()
        };
        validate_skipped_nodes_are_semantic_identities(&materialized, &weights)
            .expect("materialized constant producer is safe to omit");

        for dropout in [
            node("dropout", "Dropout", &["x", "", "training"], &["y"], vec![]),
            node("dropout", "Dropout", &["x"], &["y", "mask"], vec![]),
        ] {
            let graph = GraphProto {
                node: vec![dropout],
                ..Default::default()
            };
            let error = validate_skipped_nodes_are_semantic_identities(&graph, &WeightStore::new())
                .expect_err("training/mask Dropout must not be erased");
            assert!(error.to_string().contains("only an identity"), "{error}");
        }

        let inference_dropout = GraphProto {
            node: vec![node("dropout", "Dropout", &["x"], &["y"], vec![])],
            ..Default::default()
        };
        validate_skipped_nodes_are_semantic_identities(&inference_dropout, &WeightStore::new())
            .expect("inference Dropout without mask is an identity");
    }

    #[test]
    fn prepare_graph_rejects_duplicate_initializer_names_on_every_load() {
        for capture_provenance in [false, true] {
            let mut graph = GraphProto {
                initializer: vec![
                    float32_initializer("weight", &[1], &[1.0]),
                    float32_initializer("weight", &[1], &[2.0]),
                ],
                ..Default::default()
            };
            let mut weights = WeightStore::new();
            if capture_provenance {
                assert!(weights.enable_revision_tracking());
            }
            let error = prepare_graph(
                &mut graph,
                &mut weights,
                &mut HashMap::new(),
                capture_provenance,
                None,
            )
            .expect_err("duplicate names must never be resolved by last-write-wins");
            assert!(
                matches!(&error, NyError::ModelLoad(message) if message.contains("duplicate ONNX initializer name 'weight'")),
                "{error}"
            );
        }
    }

    #[test]
    fn prepare_graph_rejects_empty_initializer_name_without_provenance() {
        let mut graph = GraphProto {
            initializer: vec![float32_initializer("", &[1], &[1.0])],
            ..Default::default()
        };
        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("empty initializer names are invalid ONNX");
        assert!(
            matches!(&error, NyError::ModelLoad(message) if message.contains("name cannot be empty")),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_initializer_node_output_collision_without_provenance() {
        let mut graph = GraphProto {
            initializer: vec![float32_initializer("weight", &[1], &[1.0])],
            node: vec![node(
                "producer",
                "Identity",
                &["input"],
                &["weight"],
                vec![],
            )],
            ..Default::default()
        };
        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("initializer and node output must not share an SSA value name");
        assert!(
            matches!(&error, NyError::ModelLoad(message) if message.contains("collides with output")),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_duplicate_graph_io_metadata_names() {
        for (inputs, outputs, expected) in [
            (
                vec![
                    tensor_value_info("input", &[1]),
                    tensor_value_info("input", &[1]),
                ],
                Vec::new(),
                "duplicate ONNX graph input",
            ),
            (
                Vec::new(),
                vec![
                    tensor_value_info("output", &[1]),
                    tensor_value_info("output", &[1]),
                ],
                "duplicate ONNX graph output",
            ),
        ] {
            let mut graph = GraphProto {
                input: inputs,
                output: outputs,
                ..Default::default()
            };
            let error = prepare_graph(
                &mut graph,
                &mut WeightStore::new(),
                &mut HashMap::new(),
                false,
                None,
            )
            .expect_err("duplicate graph I/O metadata is ambiguous");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn prepare_graph_rejects_duplicate_node_output_values() {
        let mut graph = GraphProto {
            node: vec![
                node("first", "Identity", &["input"], &["shared"], vec![]),
                node("second", "Identity", &["input"], &["shared"], vec![]),
            ],
            ..Default::default()
        };
        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("ONNX graph values must have a single producer");
        assert!(
            error.to_string().contains("duplicate ONNX node output"),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_preserves_concat_reshape_shape_integer_store_2360() {
        let mut graph = GraphProto {
            input: vec![tensor_value_info("activation", &[1, 16_777_217])],
            initializer: vec![
                int64_initializer("gather_index", &[], &[1]),
                int64_initializer("unsqueeze_axes", &[1], &[0]),
                int64_initializer("exact_prefix", &[1], &[2]),
            ],
            node: vec![
                node("shape", "Shape", &["activation"], &["shape_out"], vec![]),
                node(
                    "gather",
                    "Gather",
                    &["shape_out", "gather_index"],
                    &["axis_size"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "unsqueeze",
                    "Unsqueeze",
                    &["axis_size", "unsqueeze_axes"],
                    &["axis_size_vec"],
                    vec![],
                ),
                node(
                    "concat",
                    "Concat",
                    &["exact_prefix", "axis_size_vec"],
                    &["reshape_shape"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["activation", "reshape_shape"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let mut weights = WeightStore::new();
        prepare_graph(&mut graph, &mut weights, &mut HashMap::new(), false, None)
            .expect("prepare should succeed");

        let reshape_shape = weights
            .get_integers("reshape_shape")
            .expect("reshape shape should preserve integer payload");
        assert_eq!(
            reshape_shape.iter().copied().collect::<Vec<_>>(),
            vec![2, 16_777_217]
        );
    }

    #[test]
    fn prepare_graph_folds_int64_shape_div_cast_chain_exactly() {
        let mut graph = GraphProto {
            // The divided extent is deliberately one above f32's consecutive
            // integer range. Going through the compatibility float view would
            // round 16_777_217 down and synthesize the wrong Reshape target.
            input: vec![tensor_value_info("activation", &[1, 33_554_434])],
            initializer: vec![
                int64_initializer("gather_index", &[], &[1]),
                int64_initializer("divisor", &[], &[2]),
                int64_initializer("prefix", &[1], &[2]),
                int64_initializer("unsqueeze_axes", &[1], &[0]),
            ],
            node: vec![
                node("shape", "Shape", &["activation"], &["shape_out"], vec![]),
                node(
                    "gather",
                    "Gather",
                    &["shape_out", "gather_index"],
                    &["extent"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "divide_extent",
                    "Div",
                    &["extent", "divisor"],
                    &["half_extent"],
                    vec![],
                ),
                node(
                    "cast_extent",
                    "Cast",
                    &["half_extent"],
                    &["half_extent_i64"],
                    vec![attr_int("to", ONNX_TENSOR_INT64)],
                ),
                node(
                    "cast_extent_again",
                    "Cast",
                    &["half_extent_i64"],
                    &["half_extent_i64_again"],
                    vec![attr_int("to", ONNX_TENSOR_INT64)],
                ),
                node(
                    "unsqueeze_extent",
                    "Unsqueeze",
                    &["half_extent_i64_again", "unsqueeze_axes"],
                    &["half_extent_vec"],
                    vec![],
                ),
                node(
                    "concat_shape",
                    "Concat",
                    &["prefix", "half_extent_vec"],
                    &["reshape_shape"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["activation", "reshape_shape"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let mut weights = WeightStore::new();
        prepare_graph(&mut graph, &mut weights, &mut HashMap::new(), false, None)
            .expect("a constant Shape/Div/Cast/Unsqueeze/Concat chain should prepare");

        assert_eq!(
            weights
                .get_integers("half_extent")
                .expect("INT64 Div must preserve its exact payload")
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![16_777_217]
        );
        assert_eq!(
            weights
                .get_integers("reshape_shape")
                .expect("the complete target shape must remain exact")
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![2, 16_777_217]
        );
        for cast_name in ["cast_extent", "cast_extent_again"] {
            let rewritten = graph
                .node
                .iter()
                .find(|node| node.name == cast_name)
                .expect("cast node retained for producer topology");
            assert_eq!(
                rewritten.op_type, "Identity",
                "only the postvalidated loader path may erase a folded INT64 Cast"
            );
            assert!(rewritten.attribute.is_empty());
        }
    }

    #[test]
    fn prepare_graph_materializes_static_shape_mul_div_int64_cast_chain() {
        let mut graph = GraphProto {
            input: vec![tensor_value_info("activation", &[1, 64, 32, 32])],
            initializer: vec![
                int64_initializer("height_axis", &[], &[2]),
                int64_initializer("width_axis", &[], &[3]),
                int64_initializer("head_area", &[], &[128]),
                int64_initializer("reshape_prefix", &[1], &[8192]),
            ],
            node: vec![
                node("shape", "Shape", &["activation"], &["shape"], vec![]),
                node(
                    "gather_height",
                    "Gather",
                    &["shape", "height_axis"],
                    &["height"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "gather_width",
                    "Gather",
                    &["shape", "width_axis"],
                    &["width"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "multiply_spatial",
                    "Mul",
                    &["height", "width"],
                    &["spatial_area"],
                    vec![],
                ),
                node(
                    "divide_head_area",
                    "Div",
                    &["spatial_area", "head_area"],
                    &["head_count"],
                    vec![],
                ),
                node(
                    "cast_head_count",
                    "Cast",
                    &["head_count"],
                    &["head_count_i64"],
                    vec![attr_int("to", 7)],
                ),
                node(
                    "cast_head_count_again",
                    "Cast",
                    &["head_count_i64"],
                    &["head_count_i64_again"],
                    vec![attr_int("to", 7)],
                ),
                node(
                    "unsqueeze_head_count",
                    "Unsqueeze",
                    &["head_count_i64_again"],
                    &["head_count_vec"],
                    vec![AttributeProto {
                        name: "axes".to_string(),
                        ints: vec![0],
                        r#type: attribute_type::INTS,
                        ..Default::default()
                    }],
                ),
                node(
                    "concat_shape",
                    "Concat",
                    &["reshape_prefix", "head_count_vec"],
                    &["reshape_shape"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["activation", "reshape_shape"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let mut weights = WeightStore::new();
        prepare_graph(&mut graph, &mut weights, &mut HashMap::new(), false, None)
            .expect("the authored static shape Cast chain should fold exactly");

        for name in [
            "spatial_area",
            "head_count",
            "head_count_i64",
            "head_count_i64_again",
            "head_count_vec",
        ] {
            let expected = if name == "spatial_area" { 1024 } else { 8 };
            assert_eq!(
                weights
                    .get_integers(name)
                    .unwrap_or_else(|| panic!("{name} should retain exact INT64 provenance"))
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![expected]
            );
            assert_eq!(
                weights
                    .get(name)
                    .unwrap()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![expected as f32]
            );
        }
    }

    #[test]
    fn prepare_graph_rejects_dynamic_int64_cast_before_proto_fusion() {
        let mut graph = GraphProto {
            input: vec![
                tensor_value_info("data", &[1, 4]),
                tensor_value_info("dynamic_shape", &[2]),
            ],
            node: vec![
                node(
                    "dynamic_cast",
                    "Cast",
                    &["dynamic_shape"],
                    &["shape_i64"],
                    vec![attr_int("to", 7)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["data", "shape_i64"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("dynamic INT64 Cast must be rejected before any fusion can consume it");
        assert!(
            error
                .to_string()
                .contains("exact constant INT64 shape cone"),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_float_only_and_private_sentinel_int64_casts() {
        let sentinel = ny_core::reshape_copy_axis_sentinel(0).expect("axis in range");
        for (label, initializer) in [
            (
                "fractional FLOAT source",
                float32_initializer("shape_source", &[1], &[0.7]),
            ),
            (
                "integer-looking FLOAT source",
                float32_initializer("shape_source", &[1], &[3.0]),
            ),
            (
                "private reshape sentinel",
                int64_initializer("shape_source", &[1], &[sentinel]),
            ),
        ] {
            let mut graph = GraphProto {
                input: vec![tensor_value_info("data", &[1])],
                initializer: vec![initializer],
                node: vec![
                    node(
                        "cast",
                        "Cast",
                        &["shape_source"],
                        &["shape_i64"],
                        vec![attr_int("to", 7)],
                    ),
                    node(
                        "reshape",
                        "Reshape",
                        &["data", "shape_i64"],
                        &["reshaped"],
                        vec![],
                    ),
                ],
                ..Default::default()
            };

            let error = prepare_graph(
                &mut graph,
                &mut WeightStore::new(),
                &mut HashMap::new(),
                false,
                None,
            )
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("exact constant INT64 shape cone"),
                "{label}: {error}"
            );
        }
    }

    #[test]
    fn prepare_graph_keeps_float_range_division_untyped_and_rejects_int64_cast() {
        let mut graph = GraphProto {
            input: vec![tensor_value_info("data", &[1])],
            initializer: vec![
                float32_initializer("one", &[], &[1.0]),
                float32_initializer("two", &[], &[2.0]),
                float32_initializer("three", &[], &[3.0]),
                float32_initializer("delta", &[], &[1.0]),
            ],
            node: vec![
                node(
                    "numerator_range",
                    "Range",
                    &["one", "two", "delta"],
                    &["numerator"],
                    vec![],
                ),
                node(
                    "denominator_range",
                    "Range",
                    &["two", "three", "delta"],
                    &["denominator"],
                    vec![],
                ),
                node(
                    "floating_division",
                    "Div",
                    &["numerator", "denominator"],
                    &["quotient"],
                    vec![],
                ),
                node(
                    "cast_quotient",
                    "Cast",
                    &["quotient"],
                    &["shape_i64"],
                    vec![attr_int("to", 7)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["data", "shape_i64"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let mut weights = WeightStore::new();
        let error = prepare_graph(&mut graph, &mut weights, &mut HashMap::new(), false, None)
            .expect_err("FLOAT Range sidecars must not manufacture raw INT64 Cast provenance");
        assert_eq!(
            weights
                .get("quotient")
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0.5]
        );
        assert!(
            weights.get_integers("quotient").is_none(),
            "FLOAT Range and division must not invent an integer sidecar"
        );
        assert!(
            error
                .to_string()
                .contains("exact constant INT64 shape cone"),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_erased_upstream_sentinel_provenance() {
        let sentinel = ny_core::reshape_copy_axis_sentinel(0).expect("axis in range");
        let mut graph = GraphProto {
            input: vec![tensor_value_info("data", &[1])],
            initializer: vec![int64_initializer("sentinel", &[], &[sentinel])],
            node: vec![
                node(
                    "erase_sentinel",
                    "Div",
                    &["sentinel", "sentinel"],
                    &["one"],
                    vec![],
                ),
                node(
                    "cast",
                    "Cast",
                    &["one"],
                    &["shape_i64"],
                    vec![attr_int("to", 7)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["data", "shape_i64"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("arithmetic must not erase private dynamic-shape provenance");
        assert!(
            error
                .to_string()
                .contains("exact constant INT64 shape cone"),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_float_gather_index_in_int64_shape_proof() {
        let mut graph = GraphProto {
            input: vec![tensor_value_info("data", &[1, 4])],
            initializer: vec![float32_initializer("axis", &[], &[1.0])],
            node: vec![
                node("shape", "Shape", &["data"], &["shape"], vec![]),
                node(
                    "gather",
                    "Gather",
                    &["shape", "axis"],
                    &["dimension"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "cast",
                    "Cast",
                    &["dimension"],
                    &["shape_i64"],
                    vec![attr_int("to", 7)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["data", "shape_i64"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("FLOAT Gather indices must not enter the raw INT64 proof");
        assert!(
            error
                .to_string()
                .contains("exact constant INT64 shape cone"),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_malformed_gather_axis_in_int64_shape_cone() {
        for axis_attributes in [
            vec![attr_int("axis", 0), attr_int("axis", 0)],
            vec![AttributeProto {
                name: "axis".to_string(),
                i: Some(0),
                r#type: attribute_type::FLOAT,
                ..Default::default()
            }],
        ] {
            let mut graph = GraphProto {
                input: vec![tensor_value_info("data", &[4])],
                initializer: vec![int64_initializer("index", &[], &[0])],
                node: vec![
                    node("shape", "Shape", &["data"], &["shape"], vec![]),
                    node(
                        "gather",
                        "Gather",
                        &["shape", "index"],
                        &["dimension"],
                        axis_attributes,
                    ),
                    node(
                        "cast",
                        "Cast",
                        &["dimension"],
                        &["dimension_i64"],
                        vec![attr_int("to", 7)],
                    ),
                    node(
                        "unsqueeze",
                        "Unsqueeze",
                        &["dimension_i64"],
                        &["reshape_shape"],
                        vec![AttributeProto {
                            name: "axes".to_string(),
                            ints: vec![0],
                            r#type: attribute_type::INTS,
                            ..Default::default()
                        }],
                    ),
                    node(
                        "reshape",
                        "Reshape",
                        &["data", "reshape_shape"],
                        &["reshaped"],
                        vec![],
                    ),
                ],
                ..Default::default()
            };

            let error = prepare_graph(
                &mut graph,
                &mut WeightStore::new(),
                &mut HashMap::new(),
                false,
                None,
            )
            .expect_err("malformed Gather axis must invalidate the INT64 shape proof");
            assert!(
                error
                    .to_string()
                    .contains("exact constant INT64 shape cone"),
                "{error}"
            );
        }
    }

    #[test]
    fn prepare_graph_rejects_competing_constant_payloads_in_int64_shape_cone() {
        let mut graph = GraphProto {
            input: vec![tensor_value_info("data", &[4])],
            node: vec![
                node(
                    "constant",
                    "Constant",
                    &[],
                    &["dimension"],
                    vec![
                        AttributeProto {
                            name: "value_int".to_string(),
                            i: Some(4),
                            r#type: attribute_type::INT,
                            ..Default::default()
                        },
                        AttributeProto {
                            name: "value_float".to_string(),
                            f: Some(2.0),
                            r#type: attribute_type::FLOAT,
                            ..Default::default()
                        },
                    ],
                ),
                node(
                    "cast",
                    "Cast",
                    &["dimension"],
                    &["dimension_i64"],
                    vec![attr_int("to", 7)],
                ),
                node(
                    "unsqueeze",
                    "Unsqueeze",
                    &["dimension_i64"],
                    &["reshape_shape"],
                    vec![AttributeProto {
                        name: "axes".to_string(),
                        ints: vec![0],
                        r#type: attribute_type::INTS,
                        ..Default::default()
                    }],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["data", "reshape_shape"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("competing Constant payloads must invalidate the INT64 shape proof");
        assert!(
            error
                .to_string()
                .contains("must have exactly one supported payload attribute"),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_two_input_unsqueeze_with_axes_attribute() {
        let mut graph = GraphProto {
            input: vec![tensor_value_info("data", &[4])],
            initializer: vec![
                int64_initializer("dimension_i64", &[], &[4]),
                // The input form says axis 1, while the malformed competing
                // attribute says axis 0. The generic folder prefers the attr.
                int64_initializer("axes_input", &[1], &[1]),
            ],
            node: vec![
                node(
                    "unsqueeze",
                    "Unsqueeze",
                    &["dimension_i64", "axes_input"],
                    &["reshape_shape"],
                    vec![AttributeProto {
                        name: "axes".to_string(),
                        ints: vec![0],
                        r#type: attribute_type::INTS,
                        ..Default::default()
                    }],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["data", "reshape_shape"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("two-input Unsqueeze must not also carry an axes attribute");
        assert!(
            error.to_string().contains(
                "carries both the opset-13 `axes` input and the opset-11 `axes` attribute"
            ),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_noncanonical_terminal_reshape() {
        let mut graph = GraphProto {
            input: vec![tensor_value_info("data", &[4])],
            initializer: vec![
                int64_initializer("shape", &[1], &[4]),
                int64_initializer("unexpected", &[], &[0]),
            ],
            node: vec![
                node(
                    "cast",
                    "Cast",
                    &["shape"],
                    &["shape_i64"],
                    vec![attr_int("to", 7)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["data", "shape_i64", "unexpected"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let error = prepare_graph(
            &mut graph,
            &mut WeightStore::new(),
            &mut HashMap::new(),
            false,
            None,
        )
        .expect_err("terminal Reshape must have exactly two inputs and one output");
        assert!(
            error
                .to_string()
                .contains("must have exactly 2 input(s) and exactly one non-empty output"),
            "{error}"
        );
    }

    #[test]
    fn prepare_graph_rejects_malformed_concat_axis_in_int64_shape_cone() {
        for axis_attributes in [
            vec![attr_int("axis", 0), attr_int("axis", 0)],
            vec![AttributeProto {
                name: "axis".to_string(),
                i: Some(0),
                r#type: attribute_type::FLOAT,
                ..Default::default()
            }],
        ] {
            let mut graph = GraphProto {
                input: vec![tensor_value_info("data", &[1, 2])],
                initializer: vec![
                    int64_initializer("prefix", &[1], &[1]),
                    int64_initializer("dimension_i64", &[1], &[2]),
                ],
                node: vec![
                    node(
                        "concat",
                        "Concat",
                        &["prefix", "dimension_i64"],
                        &["shape"],
                        axis_attributes,
                    ),
                    node(
                        "reshape",
                        "Reshape",
                        &["data", "shape"],
                        &["reshaped"],
                        vec![],
                    ),
                ],
                ..Default::default()
            };

            let error = prepare_graph(
                &mut graph,
                &mut WeightStore::new(),
                &mut HashMap::new(),
                false,
                None,
            )
            .expect_err("malformed Concat axis must invalidate the INT64 shape proof");
            assert!(
                error
                    .to_string()
                    .contains("duplicate or non-INT `axis` attribute"),
                "{error}"
            );
        }
    }

    #[test]
    fn prepare_graph_preserves_symbolic_concat_reshape_shape_over_ort_placeholder() {
        let mut graph = GraphProto {
            input: vec![
                tensor_value_info("hidden_states", &[1, -1, 1024]),
                tensor_value_info("projection", &[1, -1, 2048]),
            ],
            initializer: vec![
                int64_initializer("gather_batch_index", &[], &[0]),
                int64_initializer("gather_seq_index", &[], &[1]),
                int64_initializer("unsqueeze_axes", &[1], &[0]),
                int64_initializer("num_heads", &[1], &[16]),
                int64_initializer("head_dim", &[1], &[128]),
            ],
            node: vec![
                node("shape", "Shape", &["hidden_states"], &["shape_out"], vec![]),
                node(
                    "gather_batch",
                    "Gather",
                    &["shape_out", "gather_batch_index"],
                    &["batch_dim"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "gather_seq",
                    "Gather",
                    &["shape_out", "gather_seq_index"],
                    &["seq_dim"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "unsqueeze_batch",
                    "Unsqueeze",
                    &["batch_dim", "unsqueeze_axes"],
                    &["batch_dim_vec"],
                    vec![],
                ),
                node(
                    "unsqueeze_seq",
                    "Unsqueeze",
                    &["seq_dim", "unsqueeze_axes"],
                    &["seq_dim_vec"],
                    vec![],
                ),
                node(
                    "concat",
                    "Concat",
                    &["batch_dim_vec", "seq_dim_vec", "num_heads", "head_dim"],
                    &["reshape_shape"],
                    vec![attr_int("axis", 0)],
                ),
                node(
                    "reshape",
                    "Reshape",
                    &["projection", "reshape_shape"],
                    &["reshaped"],
                    vec![],
                ),
            ],
            ..Default::default()
        };

        let inferred_shapes = HashMap::from([
            ("hidden_states".to_string(), vec![1, 1, 1024]),
            ("projection".to_string(), vec![1, 1, 2048]),
        ]);
        let mut weights = WeightStore::new();
        let mut inferred_shapes = inferred_shapes;
        prepare_graph(&mut graph, &mut weights, &mut inferred_shapes, false, None)
            .expect("prepare should succeed");

        let reshape_shape = weights
            .get_integers("reshape_shape")
            .expect("reshape shape should preserve integer payload");
        assert_eq!(
            reshape_shape.iter().copied().collect::<Vec<_>>(),
            vec![
                1,
                ny_core::reshape_copy_axis_sentinel(1).expect("axis in range"),
                16,
                128
            ],
            "symbolic sequence length must not be replaced by an ORT placeholder"
        );
    }
}
