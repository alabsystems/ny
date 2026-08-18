// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::onnx_proto;
use crate::WeightStore;
use tracing::warn;

use super::common::{read_tensor_i64s, reshape_allowzero};
use super::is_standard_onnx_domain;

pub(crate) struct ConstFoldLookups {
    graph_shapes: HashMap<String, Vec<i64>>,
    node_by_output: HashMap<String, usize>,
    /// Globally-unbatched model (#cctsdb B5): a recorded ORT shape with
    /// symbolic (-1) dims may be recomputable EXACTLY through Reshape /
    /// broadcast recursion (safe here because rank<=1 inputs admit no
    /// symbolic batch that recursion could bake to a dummy size).
    model_unbatched: bool,
}

impl ConstFoldLookups {
    pub(crate) fn new(
        graph: &onnx_proto::GraphProto,
        inferred_shapes: &HashMap<String, Vec<i64>>,
        model_unbatched: bool,
    ) -> Self {
        Self {
            graph_shapes: build_graph_shape_lookup(graph, inferred_shapes),
            node_by_output: build_node_by_output(graph),
            model_unbatched,
        }
    }

    pub(crate) fn infer_tensor_shape(
        &self,
        tensor_name: &str,
        graph: &onnx_proto::GraphProto,
        weights: &WeightStore,
        depth: usize,
    ) -> Option<Vec<i64>> {
        if depth == 0 {
            return None;
        }
        if let Some(shape) = self.graph_shapes.get(tensor_name) {
            // A fully-static recorded shape is authoritative. A shape with
            // symbolic (-1 / 0) dims is only a FALLBACK in unbatched models:
            // recursive computation through Reshape/broadcast may recover the
            // true static dims (cctsdb coordinate-grid Shape_87 cluster, which
            // ORT reports as [1,-1,-1,64] but is statically [1,w,w,64] once
            // the affine-extent slice shapes are known).
            if shape.iter().all(|&dim| dim > 0) || !self.model_unbatched {
                return Some(shape.clone());
            }
            // No fallback to the symbolic entry: folding a Shape through the
            // -1 placeholders would STICK (outputs fold once), permanently
            // hiding the true static dims that a later pass recovers once the
            // affine-extent slice shapes land (fold_constant_nodes loops
            // folding and augmentation to a fixpoint). Declining here lets
            // the next pass fold the REAL values.
            return self.infer_tensor_shape_recursive(tensor_name, graph, weights, depth);
        }
        if let Some(weight) = weights.get(tensor_name) {
            return Some(weight.shape().iter().map(|&d| d as i64).collect());
        }
        if let Some(weight) = weights.get_integers(tensor_name) {
            return Some(weight.shape().iter().map(|&d| d as i64).collect());
        }
        self.infer_tensor_shape_recursive(tensor_name, graph, weights, depth)
    }

    /// Structural shape recursion through the producing node (no recorded-shape
    /// shortcut for the tensor itself; inputs go back through
    /// [`Self::infer_tensor_shape`]).
    fn infer_tensor_shape_recursive(
        &self,
        tensor_name: &str,
        graph: &onnx_proto::GraphProto,
        weights: &WeightStore,
        depth: usize,
    ) -> Option<Vec<i64>> {
        let node_idx = *self.node_by_output.get(tensor_name)?;
        let node = graph.node.get(node_idx)?;
        // Every rule below implements a standard ONNX operator's shape
        // semantics.  A custom-domain lookalike may use the same op_type with
        // unrelated semantics; inferring through it could make a downstream
        // standard Shape node fold to a false constant.
        if !is_standard_onnx_domain(&node.domain) {
            return None;
        }
        match node.op_type.as_str() {
            "Conv" if node.input.len() >= 2 => {
                self.infer_conv_shape(node, graph, weights, depth - 1)
            }
            "Gemm" if node.input.len() >= 2 => {
                self.infer_gemm_shape(node, graph, weights, depth - 1)
            }
            "Reshape" if node.input.len() >= 2 => {
                self.infer_reshape_shape(node, graph, weights, depth - 1)
            }
            "Slice" if !node.input.is_empty() => {
                self.infer_slice_shape(node, graph, weights, depth - 1)
            }
            // Binary elementwise ops BROADCAST: the output shape is the
            // NumPy-broadcast of both inputs, not input[0]'s shape. Returning
            // input[0]'s shape here would fold a downstream `Shape` node to a
            // WRONG constant whenever input[1] broadcasts it up (exposed by
            // the affine-extent slice shapes, #cctsdb B2 — grid construction
            // Adds like [1,w,1] + [w,1,1]). Require both shapes known.
            "Sub" | "Add" | "Mul" | "Div" if node.input.len() >= 2 => {
                let lhs = self.infer_tensor_shape(&node.input[0], graph, weights, depth - 1)?;
                let rhs = self.infer_tensor_shape(&node.input[1], graph, weights, depth - 1)?;
                broadcast_output_shape(&lhs, &rhs)
            }
            "Neg" | "Relu" | "Sigmoid" | "Tanh" | "Cos" | "Sin" if !node.input.is_empty() => {
                self.infer_tensor_shape(&node.input[0], graph, weights, depth - 1)
            }
            _ => None,
        }
    }

    fn infer_conv_shape(
        &self,
        node: &onnx_proto::NodeProto,
        graph: &onnx_proto::GraphProto,
        weights: &WeightStore,
        depth: usize,
    ) -> Option<Vec<i64>> {
        let mut auto_pad = node
            .attribute
            .iter()
            .filter(|attribute| attribute.name == "auto_pad");
        if let Some(attribute) = auto_pad.next() {
            if auto_pad.next().is_some()
                || attribute.r#type != onnx_proto::attribute_type::STRING
                || (!attribute.s_value().is_empty() && attribute.s_value() != b"NOTSET")
            {
                // SAME_*/VALID determine spatial extents by semantics not
                // represented in the explicit-pad formula below. Decline
                // rather than publishing a false Shape constant.
                return None;
            }
        }
        let input_shape = self.infer_tensor_shape(&node.input[0], graph, weights, depth)?;
        let weight = weights.get(&node.input[1])?;
        let weight_shape = weight.shape();
        if weight_shape.len() < 3 || input_shape.len() != weight_shape.len() {
            return None;
        }

        let spatial_rank = weight_shape.len().checked_sub(2)?;
        let kernel = node
            .attribute
            .iter()
            .find(|attr| attr.name == "kernel_shape")
            .map(|attr| attr.ints.clone())
            .unwrap_or_else(|| {
                weight_shape[2..]
                    .iter()
                    .map(|&dim| dim as i64)
                    .collect::<Vec<_>>()
            });
        let strides = node
            .attribute
            .iter()
            .find(|attr| attr.name == "strides")
            .map(|attr| attr.ints.clone())
            .unwrap_or_else(|| vec![1; spatial_rank]);
        let dilations = node
            .attribute
            .iter()
            .find(|attr| attr.name == "dilations")
            .map(|attr| attr.ints.clone())
            .unwrap_or_else(|| vec![1; spatial_rank]);
        let pads = node
            .attribute
            .iter()
            .find(|attr| attr.name == "pads")
            .map(|attr| attr.ints.clone())
            .unwrap_or_else(|| vec![0; spatial_rank * 2]);

        if kernel.len() != spatial_rank
            || strides.len() != spatial_rank
            || dilations.len() != spatial_rank
            || pads.len() != spatial_rank * 2
        {
            return None;
        }

        let mut output_shape = vec![*input_shape.first()?, weight_shape[0] as i64];
        for axis in 0..spatial_rank {
            let input_dim = *input_shape.get(axis + 2)?;
            let stride = *strides.get(axis)?;
            let dilation = *dilations.get(axis)?;
            let kernel_dim = *kernel.get(axis)?;
            let pad_begin = *pads.get(axis)?;
            let pad_end = *pads.get(axis + spatial_rank)?;
            if input_dim <= 0
                || stride <= 0
                || dilation <= 0
                || kernel_dim <= 0
                || pad_begin < 0
                || pad_end < 0
            {
                return None;
            }

            let effective_kernel = (kernel_dim - 1).checked_mul(dilation)?.checked_add(1)?;
            let padded_input = input_dim.checked_add(pad_begin)?.checked_add(pad_end)?;
            let output_dim = padded_input
                .checked_sub(effective_kernel)?
                .checked_div(stride)?
                .checked_add(1)?;
            output_shape.push(output_dim);
        }
        Some(output_shape)
    }

    fn infer_gemm_shape(
        &self,
        node: &onnx_proto::NodeProto,
        graph: &onnx_proto::GraphProto,
        weights: &WeightStore,
        depth: usize,
    ) -> Option<Vec<i64>> {
        let a_shape = self.infer_tensor_shape(&node.input[0], graph, weights, depth)?;
        let b_shape = self.infer_tensor_shape(&node.input[1], graph, weights, depth)?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return None;
        }
        if a_shape.iter().any(|&dim| dim <= 0) || b_shape.iter().any(|&dim| dim <= 0) {
            return None;
        }

        let trans_a = node
            .attribute
            .iter()
            .find(|attr| attr.name == "transA")
            .map(|attr| attr.i_value() != 0)
            .unwrap_or(false);
        let trans_b = node
            .attribute
            .iter()
            .find(|attr| attr.name == "transB")
            .map(|attr| attr.i_value() != 0)
            .unwrap_or(false);

        let (m, k_a) = if trans_a {
            (a_shape[1], a_shape[0])
        } else {
            (a_shape[0], a_shape[1])
        };
        let (k_b, n) = if trans_b {
            (b_shape[1], b_shape[0])
        } else {
            (b_shape[0], b_shape[1])
        };

        (k_a == k_b).then_some(vec![m, n])
    }

    fn infer_reshape_shape(
        &self,
        node: &onnx_proto::NodeProto,
        graph: &onnx_proto::GraphProto,
        weights: &WeightStore,
        depth: usize,
    ) -> Option<Vec<i64>> {
        let input_shape = self.infer_tensor_shape(&node.input[0], graph, weights, depth)?;
        if input_shape.iter().any(|&dim| dim <= 0) {
            return None;
        }
        let requested_shape = read_tensor_i64s(weights, &node.input[1])?;
        let allowzero = reshape_allowzero(node);
        reshape_output_shape(&input_shape, &requested_shape, allowzero)
    }

    fn infer_slice_shape(
        &self,
        node: &onnx_proto::NodeProto,
        graph: &onnx_proto::GraphProto,
        weights: &WeightStore,
        depth: usize,
    ) -> Option<Vec<i64>> {
        let mut output_shape = self.infer_tensor_shape(&node.input[0], graph, weights, depth)?;
        let starts_name = node.input.get(1)?;
        let ends_name = node.input.get(2)?;
        let starts_vec = read_tensor_i64s(weights, starts_name)?;
        let ends_vec = read_tensor_i64s(weights, ends_name)?;
        let axes_vec = match node.input.get(3) {
            Some(name) if !name.is_empty() => read_tensor_i64s(weights, name)?,
            _ => (0..starts_vec.len() as i64).collect(),
        };
        let steps_vec = match node.input.get(4) {
            Some(name) if !name.is_empty() => read_tensor_i64s(weights, name)?,
            _ => vec![1; starts_vec.len()],
        };
        if starts_vec.len() != ends_vec.len()
            || axes_vec.len() != starts_vec.len()
            || steps_vec.len() != starts_vec.len()
            || steps_vec.iter().any(|&step| step != 1)
        {
            return None;
        }

        for (i, &axis) in axes_vec.iter().enumerate() {
            let axis_idx = if axis < 0 {
                let resolved = (output_shape.len() as i64).checked_add(axis)?;
                if resolved < 0 {
                    return None;
                }
                resolved as usize
            } else {
                usize::try_from(axis).ok()?
            };
            if axis_idx >= output_shape.len() {
                return None;
            }
            let dim = output_shape[axis_idx];
            if dim <= 0 {
                return None;
            }
            let start = if i < starts_vec.len() {
                let value = starts_vec[i];
                if value < 0 {
                    (dim + value).max(0)
                } else {
                    value.min(dim)
                }
            } else {
                0
            };
            let end = if i < ends_vec.len() {
                let value = ends_vec[i];
                if value < 0 {
                    (dim + value).max(0)
                } else {
                    value.min(dim)
                }
            } else {
                dim
            };
            output_shape[axis_idx] = (end - start).max(0);
        }
        Some(output_shape)
    }
}

/// NumPy-style broadcast of two static shapes (right-aligned; dims equal or
/// one of them 1). `None` on incompatible or non-positive dims.
fn broadcast_output_shape(lhs: &[i64], rhs: &[i64]) -> Option<Vec<i64>> {
    if lhs.iter().any(|&dim| dim <= 0) || rhs.iter().any(|&dim| dim <= 0) {
        return None;
    }
    let rank = lhs.len().max(rhs.len());
    let mut output = Vec::with_capacity(rank);
    for i in 0..rank {
        let l = if i >= rank - lhs.len() {
            lhs[i - (rank - lhs.len())]
        } else {
            1
        };
        let r = if i >= rank - rhs.len() {
            rhs[i - (rank - rhs.len())]
        } else {
            1
        };
        let dim = if l == r || r == 1 {
            l
        } else if l == 1 {
            r
        } else {
            return None;
        };
        output.push(dim);
    }
    Some(output)
}

fn extract_shape_from_type(type_proto: &Option<onnx_proto::TypeProto>) -> Vec<i64> {
    type_proto
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .and_then(|tensor_type| tensor_type.shape.as_ref())
        .map(|shape| {
            shape
                .dim
                .iter()
                .map(|dim| match &dim.value {
                    Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(value)) => {
                        *value
                    }
                    _ => -1,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn build_graph_shape_lookup(
    graph: &onnx_proto::GraphProto,
    inferred_shapes: &HashMap<String, Vec<i64>>,
) -> HashMap<String, Vec<i64>> {
    let mut shapes = HashMap::new();
    let graph_inputs: std::collections::HashSet<&str> =
        graph.input.iter().map(|info| info.name.as_str()).collect();
    for input in &graph.input {
        let dims = extract_shape_from_type(&input.r#type);
        if !dims.is_empty() {
            shapes.insert(input.name.clone(), dims);
        }
    }
    for info in graph.value_info() {
        let dims = extract_shape_from_type(&info.r#type);
        if !dims.is_empty() {
            shapes.insert(info.name.clone(), dims);
        }
    }
    for (name, inferred) in inferred_shapes {
        if graph_inputs.contains(name.as_str()) {
            merge_graph_input_shape(name, shapes.entry(name.clone()).or_default(), inferred);
            continue;
        }
        if let Some(existing) = shapes.get(name) {
            if shapes_conflict(existing, inferred) {
                warn!(
                    "Shape conflict on intermediate '{}': proto/value_info declares {:?} \
                     but ORT inferred {:?}; using ORT shape for Shape const-folding",
                    name, existing, inferred
                );
            }
        }
        shapes.insert(name.clone(), inferred.clone());
    }
    shapes
}

fn merge_graph_input_shape(name: &str, existing: &mut Vec<i64>, inferred: &[i64]) {
    if existing.is_empty() {
        *existing = inferred.to_vec();
        return;
    }
    if existing.len() != inferred.len() {
        warn!(
            "Shape rank conflict on graph input '{}': declared {:?} but ORT inferred {:?}; \
             keeping declared input shape",
            name, existing, inferred
        );
        return;
    }
    // Preserve symbolic graph-input dimensions. ORT may report a positive
    // placeholder for them, but folding Shape through that placeholder bakes in
    // the exporter dummy size.
    for (existing_dim, inferred_dim) in existing.iter_mut().zip(inferred.iter()) {
        if *existing_dim > 0 && *inferred_dim > 0 && *existing_dim != *inferred_dim {
            warn!(
                "Shape dimension conflict on graph input '{}': declared {} but ORT inferred {}; \
                 keeping declared input dimension",
                name, *existing_dim, *inferred_dim
            );
        }
    }
}

fn shapes_conflict(existing: &[i64], inferred: &[i64]) -> bool {
    existing.len() != inferred.len()
        || existing
            .iter()
            .zip(inferred.iter())
            .any(|(left, right)| *left > 0 && *right > 0 && left != right)
}

fn build_node_by_output(graph: &onnx_proto::GraphProto) -> HashMap<String, usize> {
    let mut node_by_output = HashMap::new();
    for (idx, node) in graph.node.iter().enumerate() {
        for output in &node.output {
            if !output.is_empty() {
                node_by_output.insert(output.clone(), idx);
            }
        }
    }
    node_by_output
}

fn reshape_output_shape(
    input_shape: &[i64],
    requested_shape: &[i64],
    allowzero: bool,
) -> Option<Vec<i64>> {
    let mut resolved = Vec::with_capacity(requested_shape.len());
    let mut infer_index = None;
    let mut known_product = 1i64;

    for (idx, &dim) in requested_shape.iter().enumerate() {
        match dim {
            -1 => {
                if infer_index.is_some() {
                    return None;
                }
                infer_index = Some(idx);
                resolved.push(-1);
            }
            0 => {
                let copied = if allowzero { 0 } else { *input_shape.get(idx)? };
                known_product = known_product.checked_mul(copied)?;
                resolved.push(copied);
            }
            positive if positive > 0 => {
                known_product = known_product.checked_mul(positive)?;
                resolved.push(positive);
            }
            _ => return None,
        }
    }

    let total_elems = input_shape
        .iter()
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim))?;

    if let Some(infer_index) = infer_index {
        if known_product == 0 {
            if total_elems != 0 {
                return None;
            }
            resolved[infer_index] = 0;
            return Some(resolved);
        }
        if total_elems % known_product != 0 {
            return None;
        }
        resolved[infer_index] = total_elems / known_product;
        return Some(resolved);
    }

    (known_product == total_elems).then_some(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
        onnx_proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx_proto::TypeProto {
                tensor_type: Some(onnx_proto::TensorTypeProto {
                    elem_type: 1,
                    shape: Some(onnx_proto::TensorShapeProto {
                        dim: shape
                            .iter()
                            .map(|&value| onnx_proto::tensor_shape_proto::Dimension {
                                value: Some(
                                    onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                                        value,
                                    ),
                                ),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }

    fn add_node(name: &str, output: &str, domain: &str) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: vec!["lhs".to_string(), "rhs".to_string()],
            output: vec![output.to_string()],
            name: name.to_string(),
            op_type: "Add".to_string(),
            domain: domain.to_string(),
            attribute: Vec::new(),
        }
    }

    #[test]
    fn recursive_shape_inference_rejects_custom_domain_lookalikes() {
        let graph = onnx_proto::GraphProto {
            input: vec![value_info("lhs", &[2, 1]), value_info("rhs", &[1, 3])],
            node: vec![
                add_node("custom_add", "custom_out", "vendor.example"),
                add_node("default_add", "default_out", ""),
                add_node("explicit_standard_add", "standard_out", "ai.onnx"),
            ],
            ..Default::default()
        };
        let lookups = ConstFoldLookups::new(&graph, &HashMap::new(), false);
        let weights = WeightStore::new();

        assert_eq!(
            lookups.infer_tensor_shape("custom_out", &graph, &weights, 8),
            None,
            "a custom-domain Add must not borrow standard broadcast semantics"
        );
        for output in ["default_out", "standard_out"] {
            assert_eq!(
                lookups.infer_tensor_shape(output, &graph, &weights, 8),
                Some(vec![2, 3]),
                "both standard ONNX domain spellings must retain shape inference"
            );
        }
    }

    #[test]
    fn recursive_slice_shape_declines_non_unit_steps() {
        let graph = onnx_proto::GraphProto {
            input: vec![value_info("input", &[8])],
            node: vec![onnx_proto::NodeProto {
                input: vec![
                    "input".to_string(),
                    "starts".to_string(),
                    "ends".to_string(),
                    "axes".to_string(),
                    "steps".to_string(),
                ],
                output: vec!["slice".to_string()],
                name: "strided_slice".to_string(),
                op_type: "Slice".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            }],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        for (name, value) in [
            ("starts", 0.0),
            ("ends", 8.0),
            ("axes", 0.0),
            ("steps", 2.0),
        ] {
            weights.insert(
                name.to_string(),
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![value]).unwrap(),
            );
        }
        let lookups = ConstFoldLookups::new(&graph, &HashMap::new(), false);
        assert_eq!(
            lookups.infer_tensor_shape("slice", &graph, &weights, 8),
            None
        );
    }

    #[test]
    fn recursive_conv_shape_declines_non_explicit_auto_pad() {
        let graph = onnx_proto::GraphProto {
            input: vec![value_info("input", &[1, 1, 5, 5])],
            node: vec![onnx_proto::NodeProto {
                input: vec!["input".to_string(), "kernel".to_string()],
                output: vec!["conv".to_string()],
                name: "same_conv".to_string(),
                op_type: "Conv".to_string(),
                domain: String::new(),
                attribute: vec![onnx_proto::AttributeProto {
                    name: "auto_pad".to_string(),
                    r#type: onnx_proto::attribute_type::STRING,
                    s: Some(b"SAME_UPPER".to_vec()),
                    ..Default::default()
                }],
            }],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        weights.insert(
            "kernel".to_string(),
            ndarray::ArrayD::zeros(ndarray::IxDyn(&[1, 1, 3, 3])),
        );
        let lookups = ConstFoldLookups::new(&graph, &HashMap::new(), false);
        assert_eq!(
            lookups.infer_tensor_shape("conv", &graph, &weights, 8),
            None,
            "explicit-pad arithmetic must not infer SAME_* output shapes"
        );
    }
}
