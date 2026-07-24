// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{TensorSpec, WeightStore};
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

use super::super::shape_infer::DEFAULT_OPSET_VERSION;
use super::super::tensor::value_info_to_tensor_spec;

pub(super) struct ParseMetadata {
    pub(super) inputs: Vec<TensorSpec>,
    pub(super) outputs: Vec<TensorSpec>,
    pub(super) tensor_producer: HashMap<String, String>,
    pub(super) constant_tensors: HashSet<String>,
    pub(super) tensor_shapes: HashMap<String, Vec<i64>>,
}

pub(super) fn build_tensor_shapes(
    graph: &onnx_proto::GraphProto,
    weights: &WeightStore,
    inferred_shapes: &HashMap<String, Vec<i64>>,
) -> HashMap<String, Vec<i64>> {
    let mut tensor_shapes = HashMap::new();

    for info in graph.input.iter().chain(graph.output.iter()) {
        if !info.name.is_empty() && value_info_has_declared_shape(info) {
            if let Ok(spec) = value_info_to_tensor_spec(info) {
                tensor_shapes.insert(spec.name, spec.shape);
            }
        }
    }
    for info in graph.value_info() {
        if info.name.is_empty() {
            continue;
        }
        if value_info_has_declared_shape(info) {
            if let Ok(spec) = value_info_to_tensor_spec(info) {
                tensor_shapes.insert(spec.name, spec.shape);
            }
        }
    }
    for (name, weight) in weights.iter() {
        let shape = weight.shape().iter().map(|dim| *dim as i64).collect();
        tensor_shapes.insert(name.to_string(), shape);
    }

    if !inferred_shapes.is_empty() {
        let initializer_names: HashSet<&str> = graph
            .initializer
            .iter()
            .map(|initializer| initializer.name.as_str())
            .collect();
        let graph_io: HashSet<&str> = graph
            .input
            .iter()
            .chain(graph.output.iter())
            .map(|info| info.name.as_str())
            .collect();
        for (name, inferred) in inferred_shapes {
            if initializer_names.contains(name.as_str()) || graph_io.contains(name.as_str()) {
                match tensor_shapes.get_mut(name) {
                    Some(existing) => merge_tensor_shape(name, existing, inferred),
                    None => {
                        tensor_shapes.insert(name.clone(), inferred.clone());
                    }
                }
                continue;
            }

            if let Some(existing) = tensor_shapes.get(name) {
                if tensor_shape_conflicts(existing, inferred) {
                    warn!(
                        "Shape conflict on intermediate '{}': proto/value_info declares {:?} \
                         but ORT inferred {:?}; using ORT shape",
                        name, existing, inferred
                    );
                }
            }
            tensor_shapes.insert(name.clone(), inferred.clone());
        }
    }

    validate_matmul_shapes(&graph.node, weights, &mut tensor_shapes);
    tensor_shapes
}

pub(super) fn build_parse_metadata(
    graph: &onnx_proto::GraphProto,
    weights: &WeightStore,
    tensor_shapes: HashMap<String, Vec<i64>>,
) -> ny_core::Result<ParseMetadata> {
    let inputs: Vec<TensorSpec> = graph
        .input
        .iter()
        .filter(|input| !input.name.is_empty() && !weights.contains_key(&input.name))
        .map(value_info_to_tensor_spec)
        .collect::<ny_core::Result<Vec<_>>>()?;

    let outputs: Vec<TensorSpec> = graph
        .output
        .iter()
        .filter(|output| !output.name.is_empty())
        .map(value_info_to_tensor_spec)
        .collect::<ny_core::Result<Vec<_>>>()?;

    Ok(ParseMetadata {
        inputs,
        outputs,
        tensor_producer: build_tensor_producer(&graph.node, weights),
        constant_tensors: build_constant_tensors(graph, weights),
        tensor_shapes,
    })
}

fn build_tensor_producer(
    nodes: &[onnx_proto::NodeProto],
    weights: &WeightStore,
) -> HashMap<String, String> {
    let mut tensor_producer = HashMap::new();

    for node in nodes {
        // For each output tensor, map it to the first non-weight input (activation source).
        let activation_input = node
            .input
            .iter()
            .find(|input| !input.is_empty() && !weights.contains_key(input))
            .cloned();

        if let Some(source) = activation_input {
            for output in &node.output {
                if !output.is_empty() {
                    tensor_producer.insert(output.clone(), source.clone());
                }
            }
        }
    }

    tensor_producer
}

fn build_constant_tensors(
    graph: &onnx_proto::GraphProto,
    weights: &WeightStore,
) -> HashSet<String> {
    // Ops that produce constants without depending on activation values.
    // Concat is only considered constant when used as a shape input.
    // Slice is excluded because its output const-ness depends on its data input.
    let constant_producing_ops = ["ConstantOfShape", "Shape", "Concat"];

    let shape_input_tensors: HashSet<&str> = graph
        .node
        .iter()
        .filter_map(|node| {
            if node.op_type == "Reshape" && node.input.len() >= 2 {
                Some(node.input[1].as_str())
            } else if node.op_type == "ConstantOfShape" && !node.input.is_empty() {
                Some(node.input[0].as_str())
            } else {
                None
            }
        })
        .collect();

    let mut constant_tensors = HashSet::new();
    for node in &graph.node {
        if constant_producing_ops.contains(&node.op_type.as_str()) {
            for output in &node.output {
                if !output.is_empty() {
                    if node.op_type == "Concat" && !shape_input_tensors.contains(output.as_str()) {
                        continue;
                    }
                    constant_tensors.insert(output.clone());
                }
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for node in &graph.node {
            if !node.input.is_empty()
                && node.input.iter().all(|input| {
                    input.is_empty()
                        || weights.contains_key(input)
                        || constant_tensors.contains(input)
                })
                && matches!(
                    node.op_type.as_str(),
                    "Add"
                        | "Sub"
                        | "Mul"
                        | "Div"
                        | "Pow"
                        | "Neg"
                        | "Abs"
                        | "Sqrt"
                        | "Reciprocal"
                        | "Exp"
                        | "Log"
                        | "Sin"
                        | "Cos"
                        | "Tan"
                        | "Cast"
                        | "Floor"
                        | "Ceil"
                        | "Reshape"
                        | "Squeeze"
                        | "Unsqueeze"
                        | "Transpose"
                        | "Flatten"
                        | "Concat"
                        | "Slice"
                        | "Gather"
                        | "Shape"
                        | "ConstantOfShape"
                        | "Range"
                        | "Expand"
                        | "Tile"
                        | "NonZero"
                        | "ReduceMean"
                        | "ReduceSum"
                        | "ReduceMax"
                        | "ReduceMin"
                        | "ReduceProd"
                )
            {
                for output in &node.output {
                    if !output.is_empty() && !weights.contains_key(output) {
                        let inserted = constant_tensors.insert(output.clone());
                        if inserted {
                            debug!(
                                "Tracking {} output {} as constant tensor (transitive)",
                                node.op_type, output
                            );
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    constant_tensors
}

fn value_info_has_declared_shape(info: &onnx_proto::ValueInfoProto) -> bool {
    info.r#type
        .as_ref()
        .and_then(|type_proto| type_proto.tensor_type.as_ref())
        .and_then(|tensor_type| tensor_type.shape.as_ref())
        .is_some()
}

pub(super) fn collect_opset_imports(model: &onnx_proto::ModelProto) -> HashMap<String, i64> {
    let mut opset_imports = HashMap::new();
    let mut has_default_domain = false;

    for opset in &model.opset_import {
        let domain = opset.domain.clone();
        let version = if opset.version > 0 {
            opset.version
        } else if domain.is_empty() || domain == "ai.onnx" {
            DEFAULT_OPSET_VERSION
        } else {
            warn!(
                "Skipping opset import with invalid version {} for domain \"{}\"",
                opset.version, domain
            );
            continue;
        };
        if domain.is_empty() {
            has_default_domain = true;
        }
        if let Some(previous) = opset_imports.insert(domain.clone(), version) {
            if previous != version {
                warn!(
                    "Duplicate opset import for domain \"{}\": {} -> {}",
                    domain, previous, version
                );
            }
        }

        if domain.is_empty() && !opset_imports.contains_key("ai.onnx") {
            opset_imports.insert("ai.onnx".to_string(), version);
        } else if domain == "ai.onnx" && !opset_imports.contains_key("") {
            opset_imports.insert(String::new(), version);
        }
    }

    if !has_default_domain {
        opset_imports
            .entry(String::new())
            .or_insert(DEFAULT_OPSET_VERSION);
        opset_imports
            .entry("ai.onnx".to_string())
            .or_insert(DEFAULT_OPSET_VERSION);
    }

    opset_imports
}

pub(super) fn merge_tensor_shape(name: &str, existing: &mut Vec<i64>, inferred: &[i64]) {
    if existing.is_empty() {
        *existing = inferred.to_vec();
        return;
    }
    if existing.len() != inferred.len() {
        return;
    }
    for (existing_dim, inferred_dim) in existing.iter_mut().zip(inferred.iter()) {
        if *existing_dim > 0 && *inferred_dim > 0 && *existing_dim != *inferred_dim {
            // Defense-in-depth: ORT "lenient merge" can produce shapes that
            // conflict with proto-declared shapes. Log the conflict so
            // validate_matmul_shapes (which has weight ground truth) can
            // correct it. See #3277.
            warn!(
                "Shape dimension conflict on '{}': proto declares {} but ORT inferred {}. \
                 Keeping proto value; validate_matmul_shapes will verify against weights.",
                name, *existing_dim, *inferred_dim
            );
        }
        if *existing_dim <= 0 && *inferred_dim > 0 {
            *existing_dim = *inferred_dim;
        }
    }
}

fn tensor_shape_conflicts(existing: &[i64], inferred: &[i64]) -> bool {
    existing.len() != inferred.len()
        || existing
            .iter()
            .zip(inferred.iter())
            .any(|(existing_dim, inferred_dim)| {
                *existing_dim > 0 && *inferred_dim > 0 && existing_dim != inferred_dim
            })
}

/// Validate MatMul/Gemm output shapes against weight tensor dimensions.
///
/// ORT shape inference can produce wrong shapes via "lenient merge" when
/// proto-declared shapes conflict with weight-derived shapes. This function
/// cross-checks each MatMul/Gemm node's output shape against its weight
/// tensor and corrects mismatches.
///
/// Fix for #3277: sat_relu models where ORT lenient merge corrupts hidden
/// layer dimensions, causing PGD to report false counterexamples.
pub(super) fn validate_matmul_shapes(
    nodes: &[onnx_proto::NodeProto],
    weights: &WeightStore,
    tensor_shapes: &mut HashMap<String, Vec<i64>>,
) {
    for node in nodes {
        let (b_idx, is_gemm) = match node.op_type.as_str() {
            "MatMul" => (1, false),
            "Gemm" => (1, true),
            _ => continue,
        };

        if node.input.len() <= b_idx || node.output.is_empty() {
            continue;
        }

        let b_name = &node.input[b_idx];
        let y_name = &node.output[0];

        let b_weight = match weights.get(b_name) {
            Some(weight) => weight,
            None => continue,
        };

        let b_shape = b_weight.shape();
        if b_shape.len() < 2 {
            continue;
        }

        // Compute expected output last dimension from weight shape.
        // MatMul Y = A @ B: output last dim = B.shape[-1]
        // Gemm transB=0: B is [K, N], output last dim = N = B.shape[1]
        // Gemm transB=1: B is [N, K], output last dim = N = B.shape[0]
        let expected_n = if is_gemm {
            let trans_b = node
                .attribute
                .iter()
                .find(|attr| attr.name == "transB")
                .is_some_and(|attr| attr.i != 0);
            if trans_b {
                b_shape[0] as i64
            } else {
                b_shape[b_shape.len() - 1] as i64
            }
        } else {
            b_shape[b_shape.len() - 1] as i64
        };

        if let Some(y_shape) = tensor_shapes.get_mut(y_name) {
            if let Some(last) = y_shape.last_mut() {
                if *last > 0 && expected_n > 0 && *last != expected_n {
                    warn!(
                        "Shape conflict on {} output '{}': declared last dim {} \
                         but weight '{}' shape {:?} implies {}. \
                         Correcting to weight-derived shape. (#3277)",
                        node.op_type, y_name, *last, b_name, b_shape, expected_n
                    );
                    *last = expected_n;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: inputs.iter().map(|value| value.to_string()).collect(),
            output: outputs.iter().map(|value| value.to_string()).collect(),
            name: name.to_string(),
            op_type: op_type.to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }
    }

    #[test]
    fn build_constant_tensors_marks_sqrt_reciprocal_transitive_3500() {
        let graph = onnx_proto::GraphProto {
            node: vec![
                node("sqrt", "Sqrt", &["x"], &["sqrt_x"]),
                node("reciprocal", "Reciprocal", &["sqrt_x"], &["inv_sqrt_x"]),
            ],
            name: "constant_unary_chain".to_string(),
            initializer: Vec::new(),
            input: Vec::new(),
            output: Vec::new(),
            #[cfg(feature = "onnx-value-info")]
            value_info: Vec::new(),
        };
        let mut weights = WeightStore::new();
        weights.insert("x".to_string(), ArrayD::zeros(IxDyn(&[2])));

        let constant_tensors = build_constant_tensors(&graph, &weights);

        assert!(constant_tensors.contains("sqrt_x"));
        assert!(constant_tensors.contains("inv_sqrt_x"));
    }
}
