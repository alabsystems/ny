// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pre-saturation surrogate surgery for models whose scored output is FLAT
//! (#flat-objective-surrogate).
//!
//! A saturated terminal activation is common — a Sigmoid, a Softmax, a Tanh, a
//! clamped output quantizer — and it has now cost the campaign twice in the same
//! way: on `traffic_signs_recognition_2023` the STE attack hill-climbed the
//! post-Softmax output where 41–43 of 43 outputs are exactly `0.0f`, and on
//! `smart_turn_multimodal_2026` the terminal
//! `QuantizeLinear(uint8, zero_point = 0)` clamps every negative logit to code
//! 0, so `Sigmoid(0.0) = 0.5` EXACTLY on the whole box and the property margin
//! is identically zero. A search that scores the terminal output there learns
//! nothing from any probe; it can only win by luck.
//!
//! The cure is the same both times: score the PRE-saturation tensor, which
//! varies smoothly and orders points the way the terminal output would if it
//! were not pinned. This module does the model surgery for that: it walks back
//! from the graph's scored output through a whitelist of ELEMENTWISE MONOTONE
//! (non-decreasing) and element-order-preserving ops, and rebuilds the model
//! with the tensor at the far end as its only output. Because every peeled op
//! is monotone in each element, improving the surrogate can only move the true
//! output toward (never away from) the violation the search wants — the
//! surrogate is a steering signal with the right sign, not a claim.
//!
//! SOUNDNESS IS UNAFFECTED BY CONSTRUCTION. Nothing downstream ever accepts a
//! witness on surrogate evidence: acceptance re-runs the ORIGINAL, unmodified
//! model and re-checks the real property, then the unchanged trusted-oracle
//! gate. A wrong surrogate costs budget, never a verdict.

use prost::Message;

use crate::onnx_proto::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto};

/// ONNX `TensorProto.DataType.FLOAT`.
const DATA_TYPE_FLOAT: i32 = 1;

/// A surgeried model whose only graph output is the pre-saturation tensor.
#[derive(Debug, Clone)]
pub struct TailSurrogate {
    /// Name of the pre-saturation tensor the surrogate model outputs.
    pub tensor: String,
    /// Op types peeled between the surrogate tensor and the original output,
    /// nearest-to-output first (for logging: the lever's own observable).
    pub peeled: Vec<String>,
    /// Whether a genuinely SATURATING op (Sigmoid / Tanh / QuantizeLinear) was
    /// peeled — a tail of pure shape ops re-orders nothing and saturates
    /// nothing, so it cannot explain a flat objective.
    pub peeled_saturating_op: bool,
    /// Serialized model whose only output is [`TailSurrogate::tensor`].
    pub model_bytes: Vec<u8>,
}

/// Walk back from the model's FIRST graph output through the monotone tail and
/// return a surrogate model that outputs the pre-saturation tensor instead.
///
/// Returns `Ok(None)` when there is nothing useful to peel: no whitelisted op
/// at the output, or a tail that never saturates (shape ops only).
///
/// The whitelist and its conditions:
/// * `Sigmoid`, `Tanh` — strictly increasing elementwise.
/// * `Identity`, `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze` — element-order
///   preserving (the flattened element sequence is unchanged).
/// * `DequantizeLinear` fed by a `QuantizeLinear`, both with every scale
///   element `> 0` — peeled as an ATOMIC PAIR straight to the quantizer's
///   float input, because the intermediate code tensor is an integer type the
///   f32 forward cannot score, and a non-positive scale would flip
///   monotonicity. A `DequantizeLinear` whose producer is not a peelable
///   `QuantizeLinear` (or a bare `QuantizeLinear`) stops the walk instead.
///
/// `Softmax` is deliberately ABSENT: it is not elementwise monotone (each
/// output falls when a competing logit rises), so "bigger surrogate ⇒ closer to
/// violation" does not hold through it and a peel here could steer the search
/// backwards on multi-output specs.
pub fn peel_saturating_monotone_tail(model_bytes: &[u8]) -> Result<Option<TailSurrogate>, String> {
    let mut model = ModelProto::decode(model_bytes)
        .map_err(|e| format!("failed to parse ONNX protobuf: {e}"))?;
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| "ONNX model has no graph".to_string())?;
    let scored_output = graph
        .output
        .first()
        .map(|o| o.name.clone())
        .ok_or_else(|| "ONNX graph declares no output".to_string())?;

    // SSA: each tensor has at most one producing node.
    let producer_of = |graph: &GraphProto, tensor: &str| -> Option<usize> {
        graph
            .node
            .iter()
            .position(|n| n.output.first().map(String::as_str) == Some(tensor))
    };

    let mut tensor = scored_output;
    let mut peeled: Vec<String> = Vec::new();
    let mut peeled_saturating_op = false;
    // graph input / initializer ends the walk: nothing left to peel.
    while let Some(node_idx) = producer_of(graph, &tensor) {
        let (op_type, data_input) = {
            let node = &graph.node[node_idx];
            (node.op_type.clone(), node.input.first().cloned())
        };
        let Some(data_input) = data_input else { break };
        match op_type.as_str() {
            "Sigmoid" | "Tanh" => {
                peeled.push(op_type);
                peeled_saturating_op = true;
                tensor = data_input;
            }
            "Identity" | "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" => {
                peeled.push(op_type);
                tensor = data_input;
            }
            "DequantizeLinear" => {
                // Peel Q->DQ as an atomic pair to stay on a float tensor.
                let Some(scale) = node_scale_all_positive(graph, node_idx) else {
                    break;
                };
                if !scale {
                    break;
                }
                let Some(q_idx) = producer_of(graph, &data_input) else {
                    break;
                };
                if graph.node[q_idx].op_type != "QuantizeLinear" {
                    break;
                }
                let Some(q_scale) = node_scale_all_positive(graph, q_idx) else {
                    break;
                };
                if !q_scale {
                    break;
                }
                let Some(q_input) = graph.node[q_idx].input.first().cloned() else {
                    break;
                };
                peeled.push("DequantizeLinear".to_string());
                peeled.push("QuantizeLinear".to_string());
                peeled_saturating_op = true; // the quantizer's code range clamps
                tensor = q_input;
            }
            _ => break,
        }
    }

    if peeled.is_empty() || !peeled_saturating_op {
        return Ok(None);
    }
    // A surrogate that IS a declared activation input scores nothing.
    if graph.input.iter().any(|i| i.name == tensor)
        || graph.initializer.iter().any(|i| i.name == tensor)
    {
        return Ok(None);
    }

    graph.output = vec![ValueInfoProto {
        name: tensor.clone(),
        r#type: None, // ONNX Runtime infers it, exactly as expose_intermediate_outputs relies on
    }];
    let mut buf = Vec::new();
    model
        .encode(&mut buf)
        .map_err(|e| format!("failed to serialize surrogate ONNX model: {e}"))?;
    Ok(Some(TailSurrogate {
        tensor,
        peeled,
        peeled_saturating_op,
        model_bytes: buf,
    }))
}

/// Is every element of the node's scale input (`input[1]`) a POSITIVE f32
/// initializer value? `None` when the scale is absent, not an initializer, or
/// not float — all of which refuse the peel rather than guess.
fn node_scale_all_positive(graph: &GraphProto, node_idx: usize) -> Option<bool> {
    let scale_name = graph.node[node_idx].input.get(1)?;
    let scale = initializer_f32s(graph, scale_name)?;
    if scale.is_empty() {
        return None;
    }
    Some(scale.iter().all(|&s| s > 0.0 && s.is_finite()))
}

/// Decode a float initializer's values from either storage spelling.
fn initializer_f32s(graph: &GraphProto, name: &str) -> Option<Vec<f32>> {
    let tensor = graph.initializer.iter().find(|t| t.name == name)?;
    if tensor.data_type != DATA_TYPE_FLOAT {
        return None;
    }
    if !tensor.float_data.is_empty() {
        return Some(tensor.float_data.clone());
    }
    if !tensor.raw_data.is_empty() && tensor.raw_data.len() % 4 == 0 {
        return Some(
            tensor
                .raw_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        );
    }
    None
}

/// An activation input to pin to constant values (its box centre) so ONNX
/// Runtime constant-folds the entire branch it feeds at session build.
#[derive(Debug, Clone)]
pub struct FrozenInput {
    /// Declared activation-input name.
    pub name: String,
    /// Concrete resolved shape (symbolic batch already resolved to 1).
    pub shape: Vec<usize>,
    /// Row-major element values; length must equal the shape's element count.
    pub values: Vec<f32>,
}

/// Convert the given activation inputs into f32 initializers pinned at the
/// given values, removing them from `graph.input`.
///
/// The point is probe cost, not semantics: a measurably inert input (on
/// `smart_turn_multimodal_2026`, 1,204,224 of 1,268,224 declared coordinates
/// feed a video branch whose embedding moves at most one quantization step over
/// the WHOLE box) still costs its full branch forward on every probe. With the
/// branch's input constant, ONNX Runtime's constant folding evaluates the
/// branch ONCE at session build, and each probe pays only for the live inputs
/// — measured 11× cheaper per probe on that model. The searched witness always
/// carries the frozen coordinates at exactly the pinned values, and acceptance
/// re-runs the ORIGINAL model on the full vector.
pub fn freeze_activation_inputs(
    model_bytes: &[u8],
    frozen: &[FrozenInput],
) -> Result<Vec<u8>, String> {
    let mut model = ModelProto::decode(model_bytes)
        .map_err(|e| format!("failed to parse ONNX protobuf: {e}"))?;
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| "ONNX model has no graph".to_string())?;
    for f in frozen {
        // Membership first: "not an activation input" is the more fundamental
        // refusal, and it must not be masked by a shape/value complaint about
        // a tensor this function was never allowed to touch.
        if !graph.input.iter().any(|i| i.name == f.name) {
            return Err(format!(
                "frozen input '{}' is not a declared activation input",
                f.name
            ));
        }
        let elements: usize = f.shape.iter().product();
        if elements != f.values.len() {
            return Err(format!(
                "frozen input '{}' shape {:?} has {elements} elements but {} values were supplied",
                f.name,
                f.shape,
                f.values.len()
            ));
        }
        graph.input.retain(|i| i.name != f.name);
        graph.initializer.push(TensorProto {
            dims: f.shape.iter().map(|&d| d as i64).collect(),
            data_type: DATA_TYPE_FLOAT,
            name: f.name.clone(),
            float_data: f.values.clone(),
            ..TensorProto::default()
        });
    }
    let mut buf = Vec::new();
    model
        .encode(&mut buf)
        .map_err(|e| format!("failed to serialize frozen-input ONNX model: {e}"))?;
    Ok(buf)
}

/// The constant frontier of a graph after input freezing: intermediate tensors
/// computable from initializers alone that feed at least one NON-constant node.
#[derive(Debug, Clone)]
pub struct ConstantFrontier {
    /// Frontier tensor names, in graph node order.
    pub frontier: Vec<String>,
    /// Serialized model whose graph outputs are exactly `frontier`, for a
    /// one-forward capture of their values.
    pub exposed_model: Vec<u8>,
    /// How many nodes compute only constant tensors (observability: the work
    /// every probe re-pays until the frontier is baked).
    pub constant_node_count: usize,
}

/// Find the constant frontier, or `Ok(None)` when there is nothing to fold.
///
/// WHY THIS EXISTS (#flat-objective-surrogate): freezing an inert input as an
/// initializer makes its whole branch a constant function, but ONNX Runtime's
/// own constant folding measurably does NOT fold this model's frozen video
/// branch — on `smart_turn_multimodal_2026` a probe against the frozen-input
/// session still cost 0.137 s on a quiesced box, ~15× the audio-only cost, and
/// that priced the Hadamard ladder out of its conversion depth inside the
/// official 100 s budget. The cure is to evaluate the constant region ONCE and
/// bake its frontier values into the graph as initializers
/// ([`bake_frontier`]), after which the branch's nodes are dead and removed.
///
/// A node is constant iff every input is constant (initializers seed the set;
/// the empty-string "absent optional input" counts as constant; a no-input
/// node is constant only for op `Constant` — zero-input RANDOM ops must not
/// be). Declines (`Ok(None)`) on any control-flow subgraph attribute, because
/// a subgraph can consume outer-scope tensors invisibly to this analysis.
pub fn constant_frontier(model_bytes: &[u8]) -> Result<Option<ConstantFrontier>, String> {
    use std::collections::HashSet;
    let mut model = ModelProto::decode(model_bytes)
        .map_err(|e| format!("failed to parse ONNX protobuf: {e}"))?;
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| "ONNX model has no graph".to_string())?;
    if graph.node.iter().any(|n| {
        n.attribute
            .iter()
            .any(|a| a.g.is_some() || !a.graphs.is_empty())
    }) {
        return Ok(None); // control flow: outer-scope reads are invisible here
    }
    let mut constant: HashSet<&str> = graph.initializer.iter().map(|t| t.name.as_str()).collect();
    let inputs: HashSet<&str> = graph.input.iter().map(|i| i.name.as_str()).collect();
    let mut node_is_constant = vec![false; graph.node.len()];
    // ONNX requires topological node order; the peel above already relies on it.
    for (idx, node) in graph.node.iter().enumerate() {
        let is_const = if node.input.is_empty() {
            node.op_type == "Constant"
        } else {
            node.input.iter().all(|i| {
                i.is_empty() || (constant.contains(i.as_str()) && !inputs.contains(i.as_str()))
            })
        };
        if is_const {
            node_is_constant[idx] = true;
            for out in &node.output {
                constant.insert(out);
            }
        }
    }
    let constant_node_count = node_is_constant.iter().filter(|&&c| c).count();
    if constant_node_count == 0 {
        return Ok(None);
    }
    // Frontier: constant NODE outputs consumed by at least one non-constant
    // node. (Initializers themselves are excluded — they need no capture.)
    let node_outputs: HashSet<&str> = graph
        .node
        .iter()
        .enumerate()
        .filter(|&(idx, _)| node_is_constant[idx])
        .flat_map(|(_, n)| n.output.iter().map(String::as_str))
        .collect();
    let mut frontier: Vec<String> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for (idx, node) in graph.node.iter().enumerate() {
        if node_is_constant[idx] {
            continue;
        }
        for input in &node.input {
            if node_outputs.contains(input.as_str()) && seen.insert(input.as_str()) {
                frontier.push(input.clone());
            }
        }
    }
    if frontier.is_empty() {
        return Ok(None);
    }
    graph.output = frontier
        .iter()
        .map(|name| ValueInfoProto {
            name: name.clone(),
            r#type: None, // ORT infers it, as the peel's re-rooting relies on
        })
        .collect();
    let mut buf = Vec::new();
    model
        .encode(&mut buf)
        .map_err(|e| format!("failed to serialize frontier-exposed model: {e}"))?;
    Ok(Some(ConstantFrontier {
        frontier,
        exposed_model: buf,
        constant_node_count,
    }))
}

/// Bake captured frontier values into the graph as f32 initializers, remove
/// their producer nodes, and dead-code-eliminate the now-unreachable constant
/// cone (plus initializers nothing references any more).
///
/// Returns `Ok(None)` — decline, model unchanged — when any baked tensor's
/// producer also produces a tensor that is still consumed but NOT baked
/// (removing the node would orphan it, and keeping it would break SSA against
/// the new initializer).
pub fn bake_frontier(model_bytes: &[u8], baked: &[FrozenInput]) -> Result<Option<Vec<u8>>, String> {
    use std::collections::HashSet;
    let mut model = ModelProto::decode(model_bytes)
        .map_err(|e| format!("failed to parse ONNX protobuf: {e}"))?;
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| "ONNX model has no graph".to_string())?;
    let baked_names: HashSet<&str> = baked.iter().map(|f| f.name.as_str()).collect();
    for f in baked {
        let elements: usize = f.shape.iter().product();
        if elements != f.values.len() {
            return Err(format!(
                "baked frontier tensor '{}' shape {:?} has {elements} elements but {} values \
                 were supplied",
                f.name,
                f.shape,
                f.values.len()
            ));
        }
        if graph.initializer.iter().any(|t| t.name == f.name) {
            return Err(format!(
                "baked frontier tensor '{}' is already an initializer",
                f.name
            ));
        }
    }
    // Producers of baked tensors must be removable outright.
    let graph_outputs: HashSet<&str> = graph.output.iter().map(|o| o.name.as_str()).collect();
    let consumed: HashSet<&str> = graph
        .node
        .iter()
        .flat_map(|n| n.input.iter().map(String::as_str))
        .collect();
    let mut remove = vec![false; graph.node.len()];
    for (idx, node) in graph.node.iter().enumerate() {
        if node.output.iter().any(|o| baked_names.contains(o.as_str())) {
            let orphaned = node.output.iter().any(|o| {
                !baked_names.contains(o.as_str())
                    && (consumed.contains(o.as_str()) || graph_outputs.contains(o.as_str()))
            });
            if orphaned {
                return Ok(None);
            }
            remove[idx] = true;
        }
    }
    if !remove.iter().any(|&r| r) {
        return Ok(None); // nothing was produced by a node: nothing to fold
    }
    let mut nodes: Vec<NodeProto> = graph
        .node
        .drain(..)
        .zip(remove)
        .filter_map(|(n, r)| (!r).then_some(n))
        .collect();
    // DCE the upstream constant cone: drop nodes none of whose outputs are
    // consumed by a remaining node or declared as a graph output.
    loop {
        let used: HashSet<String> = nodes
            .iter()
            .flat_map(|n| n.input.iter().cloned())
            .chain(graph.output.iter().map(|o| o.name.clone()))
            .collect();
        let before = nodes.len();
        nodes.retain(|n| n.output.iter().any(|o| used.contains(o)));
        if nodes.len() == before {
            break;
        }
    }
    graph.node = nodes;
    for f in baked {
        graph.initializer.push(TensorProto {
            dims: f.shape.iter().map(|&d| d as i64).collect(),
            data_type: DATA_TYPE_FLOAT,
            name: f.name.clone(),
            float_data: f.values.clone(),
            ..TensorProto::default()
        });
    }
    // Prune initializers nothing references any more (the frozen video pixels
    // alone are ~1.2M floats the probe session need not even load).
    let referenced: HashSet<String> = graph
        .node
        .iter()
        .flat_map(|n| n.input.iter().cloned())
        .chain(graph.output.iter().map(|o| o.name.clone()))
        .collect();
    graph.initializer.retain(|t| referenced.contains(&t.name));
    let mut buf = Vec::new();
    model
        .encode(&mut buf)
        .map_err(|e| format!("failed to serialize frontier-baked model: {e}"))?;
    Ok(Some(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::{
        tensor_shape_proto, AttributeProto, NodeProto, TensorShapeProto, TensorTypeProto, TypeProto,
    };

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: DATA_TYPE_FLOAT,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&d| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }

    fn scalar_f32(name: &str, v: f32) -> TensorProto {
        TensorProto {
            dims: vec![],
            data_type: DATA_TYPE_FLOAT,
            name: name.to_string(),
            float_data: vec![v],
            ..TensorProto::default()
        }
    }

    fn scalar_u8(name: &str, v: i32) -> TensorProto {
        TensorProto {
            dims: vec![],
            data_type: 2, // UINT8
            name: name.to_string(),
            int32_data: vec![v],
            ..TensorProto::default()
        }
    }

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            name: format!("{op}_{output}"),
            op_type: op.to_string(),
            domain: String::new(),
            attribute: Vec::<AttributeProto>::new(),
        }
    }

    /// x -> Gemm(w) -> QuantizeLinear -> DequantizeLinear -> Sigmoid -> y,
    /// the exact smart_turn tail shape.
    fn qdq_sigmoid_tail_model(q_scale: f32) -> Vec<u8> {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    node("Gemm", &["x", "w"], "gemm_out"),
                    node("QuantizeLinear", &["gemm_out", "scale", "zp"], "codes"),
                    node("DequantizeLinear", &["codes", "scale", "zp"], "deq"),
                    node("Sigmoid", &["deq"], "y"),
                ],
                name: "tail".to_string(),
                initializer: vec![
                    scalar_f32("w", 1.0),
                    scalar_f32("scale", q_scale),
                    scalar_u8("zp", 0),
                ],
                input: vec![value_info("x", &[1, 4])],
                output: vec![value_info("y", &[1, 1])],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        };
        model.encode_to_vec()
    }

    #[test]
    fn peels_sigmoid_and_qdq_pair_to_the_float_logit() {
        let bytes = qdq_sigmoid_tail_model(0.009_454_178);
        let surrogate = peel_saturating_monotone_tail(&bytes)
            .expect("parse")
            .expect("peel");
        assert_eq!(surrogate.tensor, "gemm_out");
        assert_eq!(
            surrogate.peeled,
            vec!["Sigmoid", "DequantizeLinear", "QuantizeLinear"]
        );
        assert!(surrogate.peeled_saturating_op);
        let peeled_model = ModelProto::decode(surrogate.model_bytes.as_slice()).expect("decode");
        let graph = peeled_model.graph.expect("graph");
        assert_eq!(graph.output.len(), 1);
        assert_eq!(graph.output[0].name, "gemm_out");
        // Nodes are untouched; only the outputs changed.
        assert_eq!(graph.node.len(), 4);
    }

    #[test]
    fn non_positive_quantizer_scale_refuses_the_qdq_peel() {
        // A negative scale flips monotonicity, so the walk must stop at the
        // Sigmoid alone — leaving the (still-integer-quantized) deq tensor.
        let bytes = qdq_sigmoid_tail_model(-1.0);
        let surrogate = peel_saturating_monotone_tail(&bytes)
            .expect("parse")
            .expect("the Sigmoid peel alone still stands");
        assert_eq!(surrogate.tensor, "deq");
        assert_eq!(surrogate.peeled, vec!["Sigmoid"]);
    }

    #[test]
    fn a_pure_shape_tail_is_not_a_surrogate() {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    node("Gemm", &["x", "w"], "gemm_out"),
                    node("Reshape", &["gemm_out", "shape"], "y"),
                ],
                name: "shapes".to_string(),
                initializer: vec![scalar_f32("w", 1.0)],
                input: vec![value_info("x", &[1, 4])],
                output: vec![value_info("y", &[1, 1])],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        };
        let bytes = model.encode_to_vec();
        assert!(peel_saturating_monotone_tail(&bytes)
            .expect("parse")
            .is_none());
    }

    #[test]
    fn softmax_is_never_peeled() {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    node("Gemm", &["x", "w"], "gemm_out"),
                    node("Softmax", &["gemm_out"], "y"),
                ],
                name: "softmax".to_string(),
                initializer: vec![scalar_f32("w", 1.0)],
                input: vec![value_info("x", &[1, 4])],
                output: vec![value_info("y", &[1, 4])],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        };
        let bytes = model.encode_to_vec();
        assert!(peel_saturating_monotone_tail(&bytes)
            .expect("parse")
            .is_none());
    }

    #[test]
    fn freeze_converts_an_input_into_a_pinned_initializer() {
        let bytes = qdq_sigmoid_tail_model(1.0);
        let frozen = freeze_activation_inputs(
            &bytes,
            &[FrozenInput {
                name: "x".to_string(),
                shape: vec![1, 4],
                values: vec![0.1, 0.2, 0.3, 0.4],
            }],
        )
        .expect("freeze");
        let model = ModelProto::decode(frozen.as_slice()).expect("decode");
        let graph = model.graph.expect("graph");
        assert!(graph.input.is_empty());
        let init = graph
            .initializer
            .iter()
            .find(|t| t.name == "x")
            .expect("pinned initializer");
        assert_eq!(init.dims, vec![1, 4]);
        assert_eq!(init.float_data, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn freeze_refuses_a_name_that_is_not_an_activation_input() {
        let bytes = qdq_sigmoid_tail_model(1.0);
        let err = freeze_activation_inputs(
            &bytes,
            &[FrozenInput {
                name: "w".to_string(),
                shape: vec![],
                values: vec![],
            }],
        )
        .expect_err("w is an initializer, not an activation input");
        assert!(err.contains("not a declared activation input"), "{err}");
    }

    #[test]
    fn freeze_refuses_a_shape_value_mismatch() {
        let bytes = qdq_sigmoid_tail_model(1.0);
        let err = freeze_activation_inputs(
            &bytes,
            &[FrozenInput {
                name: "x".to_string(),
                shape: vec![1, 4],
                values: vec![0.0; 3],
            }],
        )
        .expect_err("3 values cannot pin 4 elements");
        assert!(err.contains("4 elements"), "{err}");
    }

    /// audio -> Identity -> aemb; (vin, vw initializers) -> Mul -> vemb;
    /// Add(aemb, vemb) -> out. The frozen-video shape in miniature.
    fn frozen_branch_model() -> Vec<u8> {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    node("Identity", &["audio"], "aemb"),
                    node("Mul", &["vin", "vw"], "vemb"),
                    node("Add", &["aemb", "vemb"], "out"),
                ],
                name: "frozen_branch".to_string(),
                initializer: vec![
                    TensorProto {
                        dims: vec![1, 2],
                        data_type: DATA_TYPE_FLOAT,
                        name: "vin".to_string(),
                        float_data: vec![1.0, 2.0],
                        ..TensorProto::default()
                    },
                    TensorProto {
                        dims: vec![1, 2],
                        data_type: DATA_TYPE_FLOAT,
                        name: "vw".to_string(),
                        float_data: vec![3.0, 4.0],
                        ..TensorProto::default()
                    },
                ],
                input: vec![value_info("audio", &[1, 2])],
                output: vec![value_info("out", &[1, 2])],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        };
        model.encode_to_vec()
    }

    #[test]
    fn constant_frontier_finds_the_frozen_branch_boundary() {
        let cf = constant_frontier(&frozen_branch_model())
            .expect("parse")
            .expect("one constant node exists");
        assert_eq!(cf.frontier, vec!["vemb"]);
        assert_eq!(cf.constant_node_count, 1);
        let exposed = ModelProto::decode(cf.exposed_model.as_slice()).expect("decode");
        let outs: Vec<_> = exposed
            .graph
            .unwrap()
            .output
            .into_iter()
            .map(|o| o.name)
            .collect();
        assert_eq!(outs, vec!["vemb"]);
    }

    #[test]
    fn bake_frontier_excises_the_constant_cone() {
        let bytes = frozen_branch_model();
        let baked = bake_frontier(
            &bytes,
            &[FrozenInput {
                name: "vemb".to_string(),
                shape: vec![1, 2],
                values: vec![3.0, 8.0], // vin * vw
            }],
        )
        .expect("bake")
        .expect("the Mul producer is removable");
        let model = ModelProto::decode(baked.as_slice()).expect("decode");
        let graph = model.graph.expect("graph");
        let ops: Vec<_> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
        assert_eq!(ops, vec!["Identity", "Add"], "the Mul cone must be gone");
        let init_names: Vec<_> = graph.initializer.iter().map(|t| t.name.as_str()).collect();
        assert!(init_names.contains(&"vemb"), "{init_names:?}");
        assert!(
            !init_names.contains(&"vin") && !init_names.contains(&"vw"),
            "dead initializers must be pruned: {init_names:?}"
        );
        let vemb = graph.initializer.iter().find(|t| t.name == "vemb").unwrap();
        assert_eq!(vemb.float_data, vec![3.0, 8.0]);
    }

    #[test]
    fn bake_declines_when_a_producer_output_would_be_orphaned() {
        // Foo(vin) -> [vemb, vaux]; both consumed, only vemb baked: removing
        // Foo would orphan vaux, keeping it would break SSA. Must decline.
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    node("Identity", &["audio"], "aemb"),
                    NodeProto {
                        input: vec!["vin".to_string()],
                        output: vec!["vemb".to_string(), "vaux".to_string()],
                        name: "Foo_v".to_string(),
                        op_type: "Foo".to_string(),
                        domain: String::new(),
                        attribute: vec![],
                    },
                    node("Add", &["aemb", "vemb"], "o1"),
                    node("Add", &["aemb", "vaux"], "o2"),
                ],
                name: "orphan".to_string(),
                initializer: vec![TensorProto {
                    dims: vec![1, 2],
                    data_type: DATA_TYPE_FLOAT,
                    name: "vin".to_string(),
                    float_data: vec![1.0, 2.0],
                    ..TensorProto::default()
                }],
                input: vec![value_info("audio", &[1, 2])],
                output: vec![value_info("o1", &[1, 2]), value_info("o2", &[1, 2])],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        };
        let baked = bake_frontier(
            &model.encode_to_vec(),
            &[FrozenInput {
                name: "vemb".to_string(),
                shape: vec![1, 2],
                values: vec![0.0, 0.0],
            }],
        )
        .expect("bake analysis runs");
        assert!(baked.is_none(), "orphaning bake must decline");
    }

    #[test]
    fn constant_frontier_declines_control_flow_subgraphs() {
        use crate::onnx_proto::AttributeProto;
        let mut model = ModelProto::decode(frozen_branch_model().as_slice()).unwrap();
        model.graph.as_mut().unwrap().node[1]
            .attribute
            .push(AttributeProto {
                name: "body".to_string(),
                r#type: 5, // GRAPH
                g: Some(GraphProto::default()),
                ..AttributeProto::default()
            });
        assert!(constant_frontier(&model.encode_to_vec())
            .expect("parse")
            .is_none());
    }

    #[test]
    fn a_graph_input_is_never_constant() {
        // No initializer feeds anything: frontier must be empty even though
        // the Identity node's input list is "all in some set".
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![node("Identity", &["audio"], "out")],
                name: "live".to_string(),
                initializer: vec![],
                input: vec![value_info("audio", &[1, 2])],
                output: vec![value_info("out", &[1, 2])],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        };
        assert!(constant_frontier(&model.encode_to_vec())
            .expect("parse")
            .is_none());
    }
}
