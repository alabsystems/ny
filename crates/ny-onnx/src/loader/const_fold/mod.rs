// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod broadcast;
pub(super) mod common;
mod ops;
mod shape_inference;

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::ArrayD;
use shape_inference::ConstFoldLookups;
use std::collections::HashMap;
use tracing::debug;

pub(super) struct FoldedTensor {
    pub(super) float_data: ArrayD<f32>,
    pub(super) integer_data: Option<ArrayD<i64>>,
    pub(super) integer_range: Option<(i64, i64)>,
}

impl FoldedTensor {
    fn from_float(float_data: ArrayD<f32>) -> Self {
        Self {
            float_data,
            integer_data: None,
            integer_range: None,
        }
    }
}

/// Whether the model is globally UNBATCHED: every REAL graph input (not an
/// initializer-backed entry) has a declared rank <= 1. For such models no
/// tensor ever carries a batch axis, so constant folds must preserve ONNX
/// shapes VERBATIM (no leading-1 strip) — cctsdb_yolo_2023 (#cctsdb B5).
/// Unknown input shapes stay conservative (`false`, legacy behavior).
fn graph_model_is_unbatched(graph: &onnx_proto::GraphProto, weights: &WeightStore) -> bool {
    let mut saw_real_input = false;
    for input in &graph.input {
        if weights.contains_key(&input.name) {
            continue; // initializer-backed graph input, not a real activation
        }
        saw_real_input = true;
        let rank = input
            .r#type
            .as_ref()
            .and_then(|t| t.tensor_type.as_ref())
            .and_then(|t| t.shape.as_ref())
            .map(|shape| shape.dim.len());
        match rank {
            Some(rank) if rank <= 1 => {}
            _ => return false,
        }
    }
    saw_real_input
}

pub(super) fn fold_constant_nodes(
    graph: &onnx_proto::GraphProto,
    weights: &mut WeightStore,
    inferred_shapes: &mut HashMap<String, Vec<i64>>,
) {
    let model_unbatched = graph_model_is_unbatched(graph, weights);
    // Variable-start affine Slice extents are intentionally not promoted to
    // static shapes here. ONNX clamps starts/ends to the data dimension, so
    // `ends = starts + width` does not imply a constant runtime extent near an
    // edge. Any future specialized optimization must authenticate the complete
    // consumer cone (or carry a dynamic/clamped shape), not publish the
    // unclamped maximum as global Shape authority.
    fold_constant_nodes_once(graph, weights, inferred_shapes, model_unbatched);
}

fn fold_constant_nodes_once(
    graph: &onnx_proto::GraphProto,
    weights: &mut WeightStore,
    inferred_shapes: &HashMap<String, Vec<i64>>,
    model_unbatched: bool,
) {
    let lookups = ConstFoldLookups::new(graph, inferred_shapes, model_unbatched);
    let mut changed = true;
    while changed {
        changed = false;
        for node in &graph.node {
            if let Some((output_name, output_tensor)) =
                try_fold_node(node, graph, &lookups, weights, model_unbatched)
            {
                debug!(
                    "Constant folded {}: {} shape {:?}",
                    node.op_type,
                    output_name,
                    output_tensor.float_data.shape()
                );
                if let Some(integer_data) = output_tensor.integer_data {
                    weights.insert_integers(output_name.clone(), integer_data);
                }
                if let Some((min, max)) = output_tensor.integer_range {
                    weights.insert_integer_range(output_name.clone(), min, max);
                }
                weights.insert(output_name, output_tensor.float_data);
                changed = true;
            }
        }
    }
}

fn try_fold_node(
    node: &onnx_proto::NodeProto,
    graph: &onnx_proto::GraphProto,
    lookups: &ConstFoldLookups,
    weights: &WeightStore,
    model_unbatched: bool,
) -> Option<(String, FoldedTensor)> {
    if !is_standard_onnx_domain(&node.domain) {
        return None;
    }
    let output_name = try_get_output_name(node, weights)?;
    let output_tensor = if node.op_type == "Shape" {
        ops::try_fold_shape_node(node, graph, lookups, weights)
    } else if all_inputs_constant(node, weights) {
        ops::try_fold_all_const_node(node, weights, model_unbatched)
    } else {
        None
    }?;
    Some((output_name, output_tensor))
}

pub(super) fn is_standard_onnx_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}

fn try_get_output_name(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<String> {
    if node.output.len() != 1 {
        debug!(
            "Skipping constant fold for {} with {} outputs",
            node.op_type,
            node.output.len()
        );
        return None;
    }
    let output_name = node.output.first()?.clone();
    if output_name.is_empty() {
        debug!(
            "Skipping constant fold for {} with an empty output name",
            node.op_type
        );
        return None;
    }
    (!weights.contains_key(&output_name)).then_some(output_name)
}

fn all_inputs_constant(node: &onnx_proto::NodeProto, weights: &WeightStore) -> bool {
    node.input
        .iter()
        .filter(|input| !input.is_empty())
        .all(|input| weights.contains_key(input))
}

/// Whether an INT64 Cast has been eliminated into an exact constant value.
///
/// This is the only non-FLOAT Cast graph conversion may omit.  Requiring the
/// exact i64 payload proves the Cast ran on a constant path.  Values without
/// authenticated INT64 provenance additionally require a bit-identical,
/// exactly representable f32 mirror.  Authenticated shape arithmetic may
/// exceed f32's consecutive-integer range; its exact i64 view remains
/// authoritative because the complete use cone is separately restricted to
/// structural shape construction.
pub(super) fn is_exact_materialized_int64_cast(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> bool {
    if node.op_type != "Cast"
        || !matches!(node.domain.as_str(), "" | "ai.onnx")
        || node.input.len() != 1
        || node.input[0].is_empty()
        || node.output.len() != 1
        || node.output[0].is_empty()
        || node.attribute.len() != 1
    {
        return false;
    }

    let mut targets = node
        .attribute
        .iter()
        .filter(|attribute| attribute.name == "to");
    let Some(target) = targets.next() else {
        return false;
    };
    if target.r#type != onnx_proto::attribute_type::INT
        || target.i_value() != 7
        || targets.next().is_some()
    {
        return false;
    }

    let Some(input_integers) = exact_integer_weight_view(weights, &node.input[0]) else {
        return false;
    };
    let Some(output_integers) = exact_integer_weight_view(weights, &node.output[0]) else {
        return false;
    };
    input_integers.shape() == output_integers.shape()
        && input_integers
            .iter()
            .zip(output_integers.iter())
            .all(|(input, output)| input == output)
}

pub(super) fn integer_is_exactly_representable_as_f32(value: i64) -> bool {
    let magnitude = value.unsigned_abs();
    if magnitude == 0 {
        return true;
    }
    let significant_bits = u64::BITS - magnitude.leading_zeros();
    significant_bits <= f32::MANTISSA_DIGITS
        || magnitude.trailing_zeros() >= significant_bits - f32::MANTISSA_DIGITS
}

fn exact_integer_weight_view<'a>(weights: &'a WeightStore, name: &str) -> Option<&'a ArrayD<i64>> {
    let integers = weights.get_integers(name)?;
    let floats = weights.get(name)?;
    if integers.shape() != floats.shape() {
        return None;
    }
    let authenticated_int64 = weights.get_integer_range(name) == Some((i64::MIN, i64::MAX));
    integers
        .iter()
        .zip(floats.iter())
        .all(|(&integer, &float)| {
            ny_core::reshape_copy_axis_from_sentinel(integer).is_none()
                && (authenticated_int64
                    || (integer_is_exactly_representable_as_f32(integer)
                        && (integer as f32 as i64) == integer
                        && (integer as f32).to_bits() == float.to_bits()))
        })
        .then_some(integers)
}

/// Prove that a materialized INT64 Cast really carries the authored INT64
/// value: both its operand and its result are exactly materialized (i64 payload
/// and bit-identical f32 mirror, no private reshape sentinel), and both are
/// reachable through the raw-protobuf INT64 provenance recursion.
///
/// This is the BACKWARD half of `int64_cast_has_static_reshape_shape_use` — it
/// says the constant is trustworthy, and says nothing about where it is used.
/// Dropping the node requires the forward half too; LOWERING it to
/// `LayerType::Trunc` (`loader::convert`) does not, because the node still
/// exists and its constant pre-evaluation reproduces exactly this value. That
/// is what admits cctsdb_yolo_2023's `Cast -> Range` limits, whose use cone is
/// not the cGAN `Reshape`-input-1 cone.
pub(super) fn int64_cast_has_raw_int64_provenance(
    cast: &onnx_proto::NodeProto,
    weights: &WeightStore,
    raw_int64_shape_values: &std::collections::HashSet<String>,
) -> bool {
    is_exact_materialized_int64_cast(cast, weights)
        && raw_int64_shape_values.contains(&cast.input[0])
        && raw_int64_shape_values.contains(&cast.output[0])
}

/// Prove that the whole use cone of a materialized INT64 Cast is static shape
/// construction and terminates only at the shape input of Reshape.
///
/// This predicate is used authoritatively after constant folding and before
/// proto fusions, then repeated during conversion as defense in depth.
pub(super) fn int64_cast_has_static_reshape_shape_use(
    nodes: &[onnx_proto::NodeProto],
    cast_index: usize,
    weights: &WeightStore,
    graph_output_names: &std::collections::HashSet<String>,
    raw_int64_shape_values: &std::collections::HashSet<String>,
) -> bool {
    let Some(cast) = nodes.get(cast_index) else {
        return false;
    };
    if !int64_cast_has_raw_int64_provenance(cast, weights, raw_int64_shape_values) {
        return false;
    }

    let mut consumers_by_input: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, node) in nodes.iter().enumerate() {
        for input in node.input.iter().filter(|input| !input.is_empty()) {
            consumers_by_input
                .entry(input.as_str())
                .or_default()
                .push(index);
        }
    }

    let mut pending: Vec<&str> = cast.output.iter().map(String::as_str).collect();
    let mut seen = std::collections::HashSet::new();
    let mut reached_reshape_shape = false;

    while let Some(value) = pending.pop() {
        if value.is_empty()
            || graph_output_names.contains(value)
            || !raw_int64_shape_values.contains(value)
            || !seen.insert(value)
        {
            return false;
        }
        let Some(consumers) = consumers_by_input.get(value) else {
            return false;
        };
        if consumers.is_empty() {
            return false;
        }

        for &consumer_index in consumers {
            let Some(consumer) = nodes.get(consumer_index) else {
                return false;
            };
            let positions: Vec<usize> = consumer
                .input
                .iter()
                .enumerate()
                .filter_map(|(position, input)| (input == value).then_some(position))
                .collect();
            if positions.is_empty() {
                return false;
            }

            if consumer.op_type == "Reshape" {
                if matches!(consumer.domain.as_str(), "" | "ai.onnx")
                    && consumer.input.len() == 2
                    && !consumer.input[0].is_empty()
                    && positions == [1]
                    && consumer.output.len() == 1
                    && !consumer.output[0].is_empty()
                    && consumer.attribute.is_empty()
                {
                    reached_reshape_shape = true;
                    continue;
                }
                return false;
            }

            let allowed_constant_shape_node =
                match consumer.op_type.as_str() {
                    "Cast" => is_exact_materialized_int64_cast(consumer, weights),
                    "Unsqueeze" => {
                        matches!(consumer.domain.as_str(), "" | "ai.onnx")
                            && positions.iter().all(|&position| position == 0)
                    }
                    "Concat" => {
                        matches!(consumer.domain.as_str(), "" | "ai.onnx")
                            && consumer.input.iter().filter(|input| !input.is_empty()).all(
                                |input| {
                                    raw_int64_shape_values.contains(input)
                                        && exact_integer_weight_view(weights, input).is_some()
                                },
                            )
                    }
                    _ => false,
                };
            if !allowed_constant_shape_node || consumer.output.is_empty() {
                return false;
            }
            for output in consumer.output.iter().filter(|output| !output.is_empty()) {
                if !raw_int64_shape_values.contains(output)
                    || exact_integer_weight_view(weights, output).is_none()
                {
                    return false;
                }
                pending.push(output);
            }
        }
    }

    reached_reshape_shape
}

/// Collect graph values whose authored dtype is provably INT64 along the
/// narrow static shape language used by the cGAN transformer exporter.
///
/// WeightStore's integer sidecar is intentionally not evidence here: tolerant
/// parsing in unrelated folds (notably FLOAT Range) may synthesize one.  This
/// recursion consults only raw protobuf dtypes and dtype-preserving shape ops.
pub(super) fn raw_int64_shape_values(
    graph: &onnx_proto::GraphProto,
    weights: &WeightStore,
) -> std::collections::HashSet<String> {
    if !graph
        .node
        .iter()
        .any(|node| node.op_type == "Cast" && cast_has_unique_int64_target(node))
    {
        return std::collections::HashSet::new();
    }

    let initializer_types: HashMap<&str, i32> = graph
        .initializer
        .iter()
        .filter(|initializer| !initializer.name.is_empty())
        .map(|initializer| (initializer.name.as_str(), initializer.data_type))
        .collect();
    let mut producer_by_output = HashMap::new();
    for (index, node) in graph.node.iter().enumerate() {
        for output in node.output.iter().filter(|output| !output.is_empty()) {
            producer_by_output.insert(output.as_str(), index);
        }
    }

    let mut memo = HashMap::new();
    let mut proven = std::collections::HashSet::new();
    let candidates: Vec<&str> = initializer_types
        .keys()
        .copied()
        .chain(producer_by_output.keys().copied())
        .collect();
    for value in candidates {
        if raw_value_is_int64_shape(
            value,
            graph,
            weights,
            &initializer_types,
            &producer_by_output,
            &mut memo,
            &mut std::collections::HashSet::new(),
        ) {
            proven.insert(value.to_string());
        }
    }
    proven
}

fn raw_value_is_int64_shape<'a>(
    value: &'a str,
    graph: &'a onnx_proto::GraphProto,
    weights: &WeightStore,
    initializer_types: &HashMap<&'a str, i32>,
    producer_by_output: &HashMap<&'a str, usize>,
    memo: &mut HashMap<&'a str, bool>,
    active: &mut std::collections::HashSet<&'a str>,
) -> bool {
    if let Some(&known) = memo.get(value) {
        return known;
    }
    // This graph is authored input. Bound recursive proof depth so a deeply
    // nested constant-only DAG cannot exhaust the loader stack. The official
    // cGAN shape cones are fewer than ten nodes deep.
    const MAX_RAW_INT64_SHAPE_DEPTH: usize = 256;
    if active.len() >= MAX_RAW_INT64_SHAPE_DEPTH {
        return false;
    }
    if !active.insert(value) {
        return false;
    }

    // Require every visited integer value to be exactly materialized and free
    // of ny's private dynamic-shape sentinels. Checking only the final Cast
    // would let arithmetic erase a sentinel (for example `s / s -> 1`).
    if exact_integer_weight_view(weights, value).is_none() {
        active.remove(value);
        memo.insert(value, false);
        return false;
    }

    let result = if let Some(&dtype) = initializer_types.get(value) {
        dtype == 7
    } else if let Some(&producer_index) = producer_by_output.get(value) {
        let node = &graph.node[producer_index];
        let standard = matches!(node.domain.as_str(), "" | "ai.onnx");
        let one_output = node.output.len() == 1 && node.output[0] == value;
        standard
            && one_output
            && match node.op_type.as_str() {
                "Constant" if node.input.is_empty() => constant_is_raw_int64(node),
                "Shape"
                    if node.input.len() == 1
                        && !node.input[0].is_empty()
                        && node.attribute.is_empty() =>
                {
                    true
                }
                "Gather" if node.input.len() == 2 && gather_has_valid_axis_attr(node) => {
                    node.input.iter().all(|input| {
                        !input.is_empty()
                            && raw_value_is_int64_shape(
                                input,
                                graph,
                                weights,
                                initializer_types,
                                producer_by_output,
                                memo,
                                active,
                            )
                    })
                }
                "Mul" | "Div" if node.input.len() == 2 && node.attribute.is_empty() => {
                    node.input.iter().all(|input| {
                        !input.is_empty()
                            && raw_value_is_int64_shape(
                                input,
                                graph,
                                weights,
                                initializer_types,
                                producer_by_output,
                                memo,
                                active,
                            )
                    })
                }
                "Cast" if node.input.len() == 1 && cast_has_unique_int64_target(node) => {
                    raw_value_is_int64_shape(
                        &node.input[0],
                        graph,
                        weights,
                        initializer_types,
                        producer_by_output,
                        memo,
                        active,
                    )
                }
                "Unsqueeze" if node.input.len() == 1 && unsqueeze_has_unique_axes_attr(node) => {
                    raw_value_is_int64_shape(
                        &node.input[0],
                        graph,
                        weights,
                        initializer_types,
                        producer_by_output,
                        memo,
                        active,
                    )
                }
                "Unsqueeze" if node.input.len() == 2 && node.attribute.is_empty() => {
                    node.input.iter().all(|input| {
                        !input.is_empty()
                            && raw_value_is_int64_shape(
                                input,
                                graph,
                                weights,
                                initializer_types,
                                producer_by_output,
                                memo,
                                active,
                            )
                    })
                }
                "Concat" if !node.input.is_empty() && concat_has_unique_axis_attr(node) => {
                    node.input.iter().all(|input| {
                        !input.is_empty()
                            && raw_value_is_int64_shape(
                                input,
                                graph,
                                weights,
                                initializer_types,
                                producer_by_output,
                                memo,
                                active,
                            )
                    })
                }
                _ => false,
            }
    } else {
        false
    };

    active.remove(value);
    memo.insert(value, result);
    result
}

fn cast_has_unique_int64_target(node: &onnx_proto::NodeProto) -> bool {
    if node.attribute.len() != 1 {
        return false;
    }
    let mut targets = node
        .attribute
        .iter()
        .filter(|attribute| attribute.name == "to");
    let Some(target) = targets.next() else {
        return false;
    };
    target.r#type == onnx_proto::attribute_type::INT
        && target.i_value() == 7
        && targets.next().is_none()
}

fn gather_has_valid_axis_attr(node: &onnx_proto::NodeProto) -> bool {
    match node.attribute.as_slice() {
        // Gather's default axis is zero.
        [] => true,
        [axis] => axis.name == "axis" && axis.r#type == onnx_proto::attribute_type::INT,
        _ => false,
    }
}

fn unsqueeze_has_unique_axes_attr(node: &onnx_proto::NodeProto) -> bool {
    if node.attribute.len() != 1 {
        return false;
    }
    let mut axes_attributes = node
        .attribute
        .iter()
        .filter(|attribute| attribute.name == "axes");
    let Some(axes) = axes_attributes.next() else {
        return false;
    };
    axes.r#type == onnx_proto::attribute_type::INTS && axes_attributes.next().is_none()
}

fn concat_has_unique_axis_attr(node: &onnx_proto::NodeProto) -> bool {
    if node.attribute.len() != 1 {
        return false;
    }
    let mut axis_attributes = node
        .attribute
        .iter()
        .filter(|attribute| attribute.name == "axis");
    let Some(axis) = axis_attributes.next() else {
        return false;
    };
    axis.r#type == onnx_proto::attribute_type::INT && axis_attributes.next().is_none()
}

fn constant_is_raw_int64(node: &onnx_proto::NodeProto) -> bool {
    // A valid Constant selects exactly one payload attribute. Requiring it to
    // be the node's only attribute rejects malformed competing payloads (for
    // example both value_int and value_float) instead of relying on whichever
    // one the generic folder happens to inspect first.
    if node.attribute.len() != 1 {
        return false;
    }
    let payload = &node.attribute[0];
    match payload.name.as_str() {
        "value" => {
            payload.r#type == onnx_proto::attribute_type::TENSOR
                && payload
                    .t
                    .as_ref()
                    .is_some_and(|tensor| tensor.data_type == 7)
        }
        "value_int" => payload.r#type == onnx_proto::attribute_type::INT,
        "value_ints" => payload.r#type == onnx_proto::attribute_type::INTS,
        _ => false,
    }
}

#[cfg(test)]
mod tests;
