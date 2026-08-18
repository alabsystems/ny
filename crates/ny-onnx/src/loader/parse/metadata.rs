// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{TensorSpec, WeightStore};
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

use super::super::const_fold::is_standard_onnx_domain;
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

    // ValueInfo and ORT are both optional. Recover a Flatten output only when
    // its immediate authored input shape is already known and the complete
    // local Flatten schema can be authenticated. Do not promote the broader
    // constant-fold shape recursion to conversion authority: rules for ops
    // such as Conv are deliberately partial and are unsafe as general model
    // metadata (for example, they need not model auto_pad).
    for node in &graph.node {
        if !is_standard_onnx_domain(&node.domain)
            || node.op_type != "Flatten"
            || node.input.len() != 1
            || node.input[0].is_empty()
            || node.output.len() != 1
            || node.output[0].is_empty()
            || tensor_shapes.contains_key(&node.output[0])
        {
            continue;
        }
        let Some(input_shape) = tensor_shapes.get(&node.input[0]) else {
            continue;
        };
        if let Some(output_shape) = infer_authenticated_flatten_shape(node, input_shape) {
            tensor_shapes.insert(node.output[0].clone(), output_shape);
        }
    }

    validate_matmul_shapes(&graph.node, weights, &mut tensor_shapes);
    tensor_shapes
}

fn infer_authenticated_flatten_shape(
    node: &onnx_proto::NodeProto,
    input_shape: &[i64],
) -> Option<Vec<i64>> {
    if input_shape.iter().any(|&dimension| dimension < 0) {
        return None;
    }
    let axis = match node.attribute.as_slice() {
        [] => 1,
        [attribute]
            if attribute.name == "axis" && attribute.r#type == onnx_proto::attribute_type::INT =>
        {
            attribute.i_value()
        }
        _ => return None,
    };
    let rank = i64::try_from(input_shape.len()).ok()?;
    let axis = if axis < 0 {
        axis.checked_add(rank)?
    } else {
        axis
    };
    if !(0..=rank).contains(&axis) {
        return None;
    }
    let axis = usize::try_from(axis).ok()?;
    let product = |dimensions: &[i64]| {
        dimensions
            .iter()
            .try_fold(1_i64, |accumulator, &dimension| {
                accumulator.checked_mul(dimension)
            })
    };
    Some(vec![
        product(&input_shape[..axis])?,
        product(&input_shape[axis..])?,
    ])
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
            if !is_standard_onnx_domain(&node.domain) {
                None
            } else if node.op_type == "Reshape" && node.input.len() >= 2 {
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
        if is_standard_onnx_domain(&node.domain)
            && constant_producing_ops.contains(&node.op_type.as_str())
        {
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
            if is_standard_onnx_domain(&node.domain)
                && !node.input.is_empty()
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

pub(super) fn collect_opset_imports(
    model: &onnx_proto::ModelProto,
) -> ny_core::Result<HashMap<String, i64>> {
    let mut opset_imports = HashMap::new();
    let mut standard_opset = None;
    let mut custom_opset_authorities = HashMap::new();

    for opset in &model.opset_import {
        let domain = opset.domain.clone();
        let is_standard = is_standard_onnx_domain(&domain);
        if is_standard && opset.version <= 0 {
            return Err(ny_core::NyError::ModelLoad(format!(
                "standard ONNX opset import for domain '{}' must have a positive version, got {}",
                domain, opset.version
            )));
        }
        if !is_standard {
            if let Some(previous) = custom_opset_authorities.insert(domain.clone(), opset.version) {
                let kind = if previous == opset.version {
                    "duplicate"
                } else {
                    "conflicting"
                };
                return Err(ny_core::NyError::ModelLoad(format!(
                    "{kind} custom ONNX opset imports for domain '{domain}' (versions {previous} and {})",
                    opset.version
                )));
            }
        }
        if !is_standard && opset.version <= 0 {
            warn!(
                "Skipping opset import with invalid version {} for domain \"{}\"",
                opset.version, domain,
            );
            continue;
        }

        if is_standard {
            if let Some(previous) = standard_opset {
                // The empty domain and 'ai.onnx' are the SAME domain per the ONNX
                // spec, so a model may legally list both. Only DISAGREEING versions
                // are ambiguous — there we still fail closed, because picking either
                // one silently changes operator semantics.
                //
                // Listing both at the SAME version is redundant but perfectly
                // well-defined: every operator resolves to one core opset either
                // way. Rejecting it blocked the entire dist_shift_2023 benchmark
                // (72/72 rows, "versions 11 and 11"), which is a guaranteed 0 for a
                // model ny can otherwise handle.
                if previous != opset.version {
                    return Err(ny_core::NyError::ModelLoad(format!(
                        "conflicting standard ONNX opset imports: the empty and 'ai.onnx' domains are aliases for one core operator set (versions {previous} and {})",
                        opset.version
                    )));
                }
                debug!(
                    "Model declares the standard ONNX opset twice at the same version ({previous}); \
                     the empty and 'ai.onnx' domains are aliases, so this is redundant but unambiguous"
                );
                continue;
            }
            standard_opset = Some(opset.version);
            continue;
        }

        opset_imports.insert(domain, opset.version);
    }

    let uses_standard_domain = model.graph.as_ref().is_some_and(|graph| {
        graph
            .node
            .iter()
            .any(|node| is_standard_onnx_domain(&node.domain))
    });
    if uses_standard_domain && standard_opset.is_none() {
        return Err(ny_core::NyError::ModelLoad(
            "model uses standard ONNX operators but has no standard-domain opset import; refusing to guess an operator-set version"
                .to_string(),
        ));
    }

    if let Some(version) = standard_opset {
        opset_imports.insert(String::new(), version);
        opset_imports.insert("ai.onnx".to_string(), version);
    }

    Ok(opset_imports)
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
        if !is_standard_onnx_domain(&node.domain) {
            continue;
        }
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
                .is_some_and(|attr| attr.i_value() != 0);
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
            sparse_initializer: Vec::new(),
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

    #[test]
    fn metadata_does_not_apply_standard_constant_semantics_to_custom_lookalikes() {
        let mut custom_shape = node("custom_shape", "Shape", &["runtime"], &["custom_shape_out"]);
        custom_shape.domain = "vendor.example".to_string();
        let mut custom_add = node("custom_add", "Add", &["lhs", "rhs"], &["custom_add_out"]);
        custom_add.domain = "vendor.example".to_string();
        let mut explicit_standard_shape = node(
            "explicit_standard_shape",
            "Shape",
            &["runtime"],
            &["explicit_standard_shape_out"],
        );
        explicit_standard_shape.domain = "ai.onnx".to_string();
        let graph = onnx_proto::GraphProto {
            node: vec![
                custom_shape,
                custom_add,
                node(
                    "default_shape",
                    "Shape",
                    &["runtime"],
                    &["default_shape_out"],
                ),
                explicit_standard_shape,
            ],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        weights.insert("lhs".to_string(), ArrayD::zeros(IxDyn(&[1])));
        weights.insert("rhs".to_string(), ArrayD::zeros(IxDyn(&[1])));

        let constants = build_constant_tensors(&graph, &weights);

        assert!(!constants.contains("custom_shape_out"));
        assert!(!constants.contains("custom_add_out"));
        assert!(constants.contains("default_shape_out"));
        assert!(constants.contains("explicit_standard_shape_out"));
    }

    #[test]
    fn matmul_shape_correction_ignores_custom_domain_lookalikes() {
        let standard = node("standard", "MatMul", &["x", "weight"], &["standard_out"]);
        let mut custom = node("custom", "MatMul", &["x", "weight"], &["custom_out"]);
        custom.domain = "vendor.example".to_string();
        let mut weights = WeightStore::new();
        weights.insert("weight".to_string(), ArrayD::zeros(IxDyn(&[2, 4])));
        let mut shapes = HashMap::from([
            ("standard_out".to_string(), vec![1, 9]),
            ("custom_out".to_string(), vec![1, 9]),
        ]);

        validate_matmul_shapes(&[standard, custom], &weights, &mut shapes);

        assert_eq!(shapes.get("standard_out"), Some(&vec![1, 4]));
        assert_eq!(
            shapes.get("custom_out"),
            Some(&vec![1, 9]),
            "a custom handler owns its output-shape semantics"
        );
    }
}
