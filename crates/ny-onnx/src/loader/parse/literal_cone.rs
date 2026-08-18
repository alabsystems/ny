// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Load-time literal tensors: values fixed by the model bytes alone.
//!
//! ny's raw schema gates require the FLOAT32 data path because ny propagates
//! `tensor(float)` activations and normalizing another authored dtype into
//! `WeightStore`'s f32 view would change the arithmetic being verified.  That
//! premise is about the DATA PATH.  A node whose complete transitive input cone
//! is authored constants computes a value that is a function of the model file
//! and of nothing else: it is not an activation, it never sees a verification
//! input, and the loader is obliged to erase it into a literal before any
//! propagation runs.
//!
//! This module authenticates exactly that shape of cone, and only for the
//! exact-integer dtypes ny's constant folder evaluates in `i64`:
//!
//! * seeds are graph initializers and `Constant` nodes — **never** a runtime
//!   graph input;
//! * `Shape` is deliberately NOT a literal producer.  `Shape(x)` reads shape
//!   metadata of a runtime tensor, so its value is only as static as ny's shape
//!   inference; that case belongs to the separate INT64 structural-control
//!   machinery in `structural_schema`, which authenticates it properly;
//! * only pure, single-valued standard-domain operators propagate the property
//!   (no subgraph attributes, no RNG, no data-dependent output shape);
//! * a value with more than one producer is never literal.
//!
//! An exemption granted here is NOT a promise that ny can evaluate the operator.
//! It is a deferral: [`LiteralExemptions::require_all_folded`] re-raises the
//! original refusal after constant folding unless the node was actually erased
//! into a `WeightStore` literal.  A cone that does not fold end to end therefore
//! still fails closed, with the same message it had before.

use crate::loader::const_fold::is_standard_onnx_domain;
use crate::onnx_proto::{GraphProto, NodeProto};
use crate::WeightStore;
use ny_core::{NyError, Result};
use std::collections::{HashMap, HashSet};

use super::quantization_preflight::RawDtypeResolver;

const INT32: i32 = 6;
const INT64: i32 = 7;
const BOOL: i32 = 9;

/// Standard-domain operators that are pure functions of their inputs, produce a
/// single tensor per output, carry no subgraph attribute, and are folded by
/// `loader::const_fold`.  Membership only ever ADDS names to the literal set,
/// and every consumer of that set additionally restricts the dtype, so an
/// operator listed here can never launder a float rounding decision.
///
/// `Shape` is excluded on purpose (see the module docs).
const LITERAL_PRODUCERS: &[&str] = &[
    "Constant",
    "ConstantOfShape",
    "Identity",
    "Cast",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Neg",
    "Abs",
    "Equal",
    "Less",
    "Greater",
    "LessOrEqual",
    "GreaterOrEqual",
    "Where",
    "Concat",
    "Gather",
    "Slice",
    "Squeeze",
    "Unsqueeze",
    "Reshape",
    "Transpose",
    "Expand",
    "Range",
];

/// The set of tensor names whose value is decided by the model file alone.
pub(super) struct LiteralCone {
    literals: HashSet<String>,
}

impl LiteralCone {
    pub(super) fn new(graph: &GraphProto) -> Self {
        let mut producer_count: HashMap<&str, usize> = HashMap::new();
        for node in &graph.node {
            for output in node.output.iter().filter(|output| !output.is_empty()) {
                *producer_count.entry(output.as_str()).or_default() += 1;
            }
        }

        let mut literals: HashSet<String> = graph
            .initializer
            .iter()
            .filter(|tensor| !tensor.name.is_empty())
            .map(|tensor| tensor.name.clone())
            .collect();

        // Fixpoint: a listed operator all of whose non-empty inputs are already
        // literal contributes its outputs.  `Constant` has no inputs, so it
        // enters on the first pass; a runtime graph input has no producer and
        // can never enter at all.
        loop {
            let mut changed = false;
            for node in &graph.node {
                if !Self::is_literal_producer(node) {
                    continue;
                }
                if node
                    .input
                    .iter()
                    .any(|input| !input.is_empty() && !literals.contains(input))
                {
                    continue;
                }
                for output in node.output.iter().filter(|output| !output.is_empty()) {
                    if producer_count.get(output.as_str()).copied() != Some(1) {
                        continue;
                    }
                    if literals.insert(output.clone()) {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        Self { literals }
    }

    fn is_literal_producer(node: &NodeProto) -> bool {
        is_standard_onnx_domain(&node.domain)
            && LITERAL_PRODUCERS.contains(&node.op_type.as_str())
            && node.output.len() == 1
            && node.output.iter().all(|output| !output.is_empty())
    }

    #[cfg(test)]
    pub(super) fn is_literal(&self, value: &str) -> bool {
        self.literals.contains(value)
    }

    /// Whether every non-empty tensor this node reads or writes is a load-time
    /// literal.
    pub(super) fn covers(&self, node: &NodeProto) -> bool {
        Self::is_literal_producer(node)
            && node
                .input
                .iter()
                .chain(node.output.iter())
                .filter(|value| !value.is_empty())
                .all(|value| self.literals.contains(value))
    }
}

/// Whether this node is a load-time literal computation over exact integers.
///
/// Both halves are required.  `covers` proves the value never depends on a
/// verification input; the dtype restriction proves the folder evaluates it in
/// `i64` (or as a `{0,1}` BOOL), which is why erasing the authored dtype into
/// `WeightStore`'s f32 mirror cannot change the value.  Float dtypes — including
/// DOUBLE, FLOAT16 and BFLOAT16 constant arithmetic — are deliberately NOT
/// exempted: their refusal is about rounding, which a constant cone does not
/// make moot.
pub(super) fn is_exact_integer_literal_node(
    node: &NodeProto,
    cone: &LiteralCone,
    dtypes: &mut RawDtypeResolver<'_>,
) -> Result<bool> {
    if !cone.covers(node) {
        return Ok(false);
    }
    for value in node
        .input
        .iter()
        .chain(node.output.iter())
        .filter(|value| !value.is_empty())
    {
        let Some(dtype) = dtypes.resolve(value)? else {
            return Ok(false);
        };
        if !matches!(dtype, INT32 | INT64 | BOOL) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A refusal that was deferred because the node looked like a load-time
/// literal, together with the message to re-raise if it turns out not to be one.
struct DeferredRefusal {
    output: String,
    node: String,
    op_type: String,
    original: String,
}

/// Every raw-schema refusal that the literal-cone exemption suspended.
#[derive(Default)]
pub(super) struct LiteralExemptions {
    deferred: Vec<DeferredRefusal>,
}

impl LiteralExemptions {
    pub(super) fn record(&mut self, node: &NodeProto, original: &NyError) {
        let Some(output) = node.output.first().filter(|output| !output.is_empty()) else {
            return;
        };
        self.deferred.push(DeferredRefusal {
            output: output.clone(),
            node: node.name.clone(),
            op_type: node.op_type.clone(),
            original: original.to_string(),
        });
    }

    /// Re-raise every deferred refusal whose node survived constant folding.
    ///
    /// This is the ratchet that makes the exemption safe: the preflight did not
    /// decide the node was representable, it decided the node should not exist
    /// by the time propagation runs.  A node still present here would reach
    /// graph conversion as a real f32 layer — exactly the situation the raw gate
    /// exists to prevent — so it fails with the original message.
    pub(super) fn require_all_folded(&self, weights: &WeightStore) -> Result<()> {
        for deferred in &self.deferred {
            if !weights.contains_key(&deferred.output) {
                return Err(NyError::UnsupportedOp(format!(
                    "{}; its inputs are load-time literals but constant folding did not \
                     materialize '{}' for {} node '{}', so the node would reach ny's f32 \
                     runtime graph",
                    deferred.original, deferred.output, deferred.op_type, deferred.node
                )));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.deferred.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::{
        attribute_type, tensor_shape_proto, AttributeProto, NodeProto, TensorProto,
        TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };

    fn int64_constant(name: &str, output: &str, values: &[i64]) -> NodeProto {
        NodeProto {
            name: name.to_string(),
            op_type: "Constant".to_string(),
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: attribute_type::TENSOR,
                t: Some(TensorProto {
                    data_type: 7,
                    dims: vec![values.len() as i64],
                    int64_data: values.to_vec(),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn float_input(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: vec![tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(4)),
                        }],
                    }),
                }),
            }),
        }
    }

    fn binary(op: &str, name: &str, a: &str, b: &str, out: &str) -> NodeProto {
        NodeProto {
            name: name.to_string(),
            op_type: op.to_string(),
            input: vec![a.to_string(), b.to_string()],
            output: vec![out.to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn constant_cone_is_literal_and_runtime_cone_is_not() {
        let graph = GraphProto {
            input: vec![float_input("x")],
            node: vec![
                int64_constant("k0", "k0_out", &[1, -1]),
                int64_constant("k1", "k1_out", &[-1, -1]),
                binary("Equal", "eq", "k0_out", "k1_out", "eq_out"),
                binary("Mul", "runtime_mul", "x", "k0_out", "runtime_out"),
                binary(
                    "Equal",
                    "runtime_eq",
                    "runtime_out",
                    "k1_out",
                    "runtime_eq_out",
                ),
            ],
            ..Default::default()
        };
        let cone = LiteralCone::new(&graph);
        assert!(cone.is_literal("k0_out"));
        assert!(cone.is_literal("eq_out"));
        assert!(!cone.is_literal("x"));
        assert!(!cone.is_literal("runtime_out"));
        assert!(cone.covers(&graph.node[2]));
        assert!(!cone.covers(&graph.node[4]));
    }

    #[test]
    fn shape_is_never_a_literal_producer() {
        let graph = GraphProto {
            input: vec![float_input("x")],
            node: vec![
                NodeProto {
                    name: "shape".to_string(),
                    op_type: "Shape".to_string(),
                    input: vec!["x".to_string()],
                    output: vec!["shape_out".to_string()],
                    ..Default::default()
                },
                int64_constant("k", "k_out", &[4]),
                binary("Equal", "eq", "shape_out", "k_out", "eq_out"),
            ],
            ..Default::default()
        };
        let cone = LiteralCone::new(&graph);
        assert!(!cone.is_literal("shape_out"));
        assert!(!cone.is_literal("eq_out"));
        assert!(!cone.covers(&graph.node[2]));
    }

    #[test]
    fn float_literal_cone_is_not_exempted() {
        let float_const = NodeProto {
            name: "f".to_string(),
            op_type: "Constant".to_string(),
            output: vec!["f_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: attribute_type::TENSOR,
                t: Some(TensorProto {
                    data_type: 11, // DOUBLE
                    dims: vec![1],
                    double_data: vec![0.1],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![
                float_const,
                binary("Mul", "mul", "f_out", "f_out", "mul_out"),
            ],
            ..Default::default()
        };
        let cone = LiteralCone::new(&graph);
        assert!(cone.covers(&graph.node[1]));
        let mut dtypes = RawDtypeResolver::new(&graph);
        assert!(
            !is_exact_integer_literal_node(&graph.node[1], &cone, &mut dtypes).unwrap(),
            "a DOUBLE constant cone must stay refused: folding it through f32 changes rounding"
        );
    }

    #[test]
    fn deferred_refusal_is_reraised_when_the_node_survives_folding() {
        let node = binary("Equal", "eq", "a", "b", "eq_out");
        let mut exemptions = LiteralExemptions::default();
        exemptions.record(
            &node,
            &NyError::UnsupportedOp("original refusal".to_string()),
        );
        assert_eq!(exemptions.len(), 1);

        let empty = WeightStore::new();
        let error = exemptions
            .require_all_folded(&empty)
            .expect_err("an unfolded literal node must re-raise its refusal");
        assert!(error.to_string().contains("original refusal"));

        let mut folded = WeightStore::new();
        folded.insert(
            "eq_out".to_string(),
            ndarray::ArrayD::zeros(ndarray::IxDyn(&[2])),
        );
        exemptions
            .require_all_folded(&folded)
            .expect("an erased literal node carries no refusal");
    }
}
