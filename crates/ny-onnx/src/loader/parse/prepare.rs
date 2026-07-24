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

use super::super::const_fold::fold_constant_nodes;
use super::super::numeric_cast::{i64_to_f32_checked, i64_to_f32_warned};
use super::super::tensor::{extract_constant_tensor, tensor_proto_to_loaded_tensor, LoadedTensor};

pub(super) fn prepare_graph(
    graph: &mut onnx_proto::GraphProto,
    weights: &mut WeightStore,
    inferred_shapes: &mut HashMap<String, Vec<i64>>,
    capture_raw_float32_initializer_provenance: bool,
) -> Result<HashMap<String, OriginalFloat32Initializer>> {
    let mut original_float32_initializers = HashMap::new();
    let mut initializer_names = HashSet::new();

    // Establish unambiguous initializer names before inserting any value or
    // running a rewrite. ONNX graph values are SSA: a raw initializer cannot
    // also be produced by a node. Treating such an invalid collision as raw
    // provenance would let the mutable graph and immutable record disagree.
    if capture_raw_float32_initializer_provenance {
        for init in &graph.initializer {
            if init.name.is_empty() {
                return Err(NyError::ModelLoad(
                    "ONNX initializer name cannot be empty when establishing provenance"
                        .to_string(),
                ));
            }
            if !initializer_names.insert(init.name.clone()) {
                return Err(NyError::ModelLoad(format!(
                    "duplicate ONNX initializer name '{}' cannot establish immutable provenance",
                    init.name
                )));
            }
        }
        reject_initializer_node_output_collisions(graph, &initializer_names)?;
    }

    // Extract weights from initializers.
    for init in &graph.initializer {
        let name = init.name.clone();
        let tensor = tensor_proto_to_loaded_tensor(init)?;
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
    // Lowerings above synthesize graph value names. Recheck after every
    // prepare-time rewrite so generated outputs cannot acquire a raw
    // initializer's identity without changing its weight revision.
    if capture_raw_float32_initializer_provenance {
        reject_initializer_node_output_collisions(graph, &initializer_names)?;
    }
    Ok(original_float32_initializers)
}

fn reject_initializer_node_output_collisions(
    graph: &onnx_proto::GraphProto,
    initializer_names: &HashSet<String>,
) -> Result<()> {
    for node in &graph.node {
        for output in node.output.iter().filter(|output| !output.is_empty()) {
            if initializer_names.contains(output) {
                return Err(NyError::ModelLoad(format!(
                    "ONNX initializer '{}' collides with output of node '{}' ({})",
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
    if node.op_type != "Reshape" || node.input.len() < 2 {
        return None;
    }

    let shape_input = node.input.get(1)?.as_str();
    if weights.contains_key(shape_input) {
        return None;
    }

    let concat_node = node_by_output.get(shape_input)?;
    (concat_node.op_type == "Concat").then_some((shape_input, *concat_node))
}

fn infer_concat_shape_tensor(
    concat_node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<(LoadedTensor, bool)> {
    let mut inferred_shape = Vec::new();
    let mut inferred_shape_i64 = Some(Vec::new());
    let mut all_known = true;

    for concat_input in concat_node.input.iter().filter(|input| !input.is_empty()) {
        let (dim, exact_dim, is_known) = infer_concat_shape_dim(weights, concat_input);
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

fn infer_concat_shape_dim(weights: &WeightStore, concat_input: &str) -> (f32, Option<i64>, bool) {
    if let Some(value) = weights.get_integers(concat_input) {
        let dim = value.iter().next().copied().unwrap_or(0);
        return (
            i64_to_f32_checked(dim, "prepare_graph inferred reshape shape")
                .unwrap_or_else(|_| i64_to_f32_warned(dim, "prepare_graph inferred reshape shape")),
            Some(dim),
            true,
        );
    }

    if let Some(value) = weights.get(concat_input) {
        let dim = value.iter().next().copied().unwrap_or(0.0);
        return (dim, parse_shape_scalar_i64(dim), true);
    }

    // Use 0 to preserve a dynamic dimension in ONNX Reshape.
    (0.0, Some(0), false)
}

fn parse_shape_scalar_i64(value: f32) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1.0e-4 {
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
        if node.op_type != "ReduceL2" || node.input.is_empty() || node.output.is_empty() {
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
                f: 2.0,
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
        if node.op_type != "Constant" {
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
        if let Some(output_name) = node.output.first() {
            if let Some(tensor) = extract_constant_tensor(node)? {
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
            i: value,
            r#type: attribute_type::INT,
            ..Default::default()
        }
    }

    #[test]
    fn prepare_graph_rejects_duplicate_initializer_names_for_provenance() {
        let mut graph = GraphProto {
            initializer: vec![
                float32_initializer("weight", &[1], &[1.0]),
                float32_initializer("weight", &[1], &[1.0]),
            ],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        let error = prepare_graph(&mut graph, &mut weights, &mut HashMap::new(), true)
            .expect_err("duplicate names must not create ambiguous raw provenance");
        assert!(
            matches!(&error, NyError::ModelLoad(message) if message.contains("duplicate ONNX initializer name 'weight'")),
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
                    &["dynamic_prefix", "axis_size_vec"],
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
        prepare_graph(&mut graph, &mut weights, &mut HashMap::new(), false)
            .expect("prepare should succeed");

        let reshape_shape = weights
            .get_integers("reshape_shape")
            .expect("reshape shape should preserve integer payload");
        assert_eq!(
            reshape_shape.iter().copied().collect::<Vec<_>>(),
            vec![0, 16_777_217]
        );
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
        prepare_graph(&mut graph, &mut weights, &mut inferred_shapes, false)
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
