// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod affine_extent;
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
    loop {
        fold_constant_nodes_once(graph, weights, inferred_shapes, model_unbatched);
        // Affine-extent slice shapes (#cctsdb B2): variable-start slices whose
        // extent (ends - starts) is provably constant get static output shapes
        // recorded into `inferred_shapes`; re-run folding so Shape-of-slice
        // chains (and the shape clusters they feed) cascade. Terminates: each
        // augmentation round records at least one NEW shape entry, bounded by
        // the node count.
        if !affine_extent::augment_inferred_shapes_with_affine_slice_extents(
            graph,
            weights,
            inferred_shapes,
        ) {
            break;
        }
    }
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
    (!weights.contains_key(&output_name)).then_some(output_name)
}

fn all_inputs_constant(node: &onnx_proto::NodeProto, weights: &WeightStore) -> bool {
    node.input
        .iter()
        .filter(|input| !input.is_empty())
        .all(|input| weights.contains_key(input))
}

#[cfg(test)]
mod tests;
