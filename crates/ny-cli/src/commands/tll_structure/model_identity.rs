// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact authored-ONNX identity proof for the TLL structural lane.
//!
//! This module deliberately does not call NY's ONNX loader. It accepts only a
//! narrow, raw protobuf realization whose complete graph is
//!
//! `x -> affine L -> exact one-hot selection -> min gadgets -> max gadgets`.
//!
//! Each gadget is checked entry-by-entry against the algebraic identity
//!
//! `min/max(a,b) = .5 * (s +/- |a-b|)`, `s = a+b`,
//!
//! including its two exact MatMul tensors, zero biases, intervening ReLU, and
//! all edges. Consequently the returned selector groups describe the authored
//! graph on the whole domain; finite forward samples are not proof authority.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use ny_core::f32_to_f64_exact;
use ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value as DimValue;
use ny_onnx::onnx_proto::{
    AttributeProto, OperatorSetIdProto, SparseTensorProto, TensorProto, ValueInfoProto,
};
use prost::Message;
use sha2::{Digest, Sha256};

use super::TllStructure;

mod wire_audit;

const FLOAT: i32 = 1;
const MAX_MODEL_BYTES: u64 = 384 * 1024 * 1024;
const MAX_FLOAT_ELEMENTS: usize = 80 * 1024 * 1024;
const MAX_NODES: usize = 128;
const MAX_INITIALIZERS: usize = 64;

/// Local source-audit protobuf. NY's general-purpose minimal protobuf omits
/// newer semantically relevant fields such as `NodeProto.overload` and model
/// local functions. Prost would silently discard them, so verdict provenance
/// uses this stricter envelope and explicitly rejects those features.
#[derive(Clone, PartialEq, Message)]
struct AuditedModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, optional, tag = "7")]
    graph: Option<AuditedGraphProto>,
    #[prost(bytes = "vec", repeated, tag = "20")]
    training_info: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "25")]
    functions: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "26")]
    device_configurations: Vec<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
struct AuditedGraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<AuditedNodeProto>,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "15")]
    sparse_initializer: Vec<SparseTensorProto>,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
struct AuditedNodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op_type: String,
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    domain: String,
    #[prost(string, tag = "8")]
    overload: String,
    #[prost(bytes = "vec", repeated, tag = "10")]
    device_configurations: Vec<Vec<u8>>,
}

/// Authenticate and decode a canonical raw TLL model. `None` means that some
/// source-level identity obligation was unsupported or violated.
pub(super) fn authenticate_raw_tll_model(path: &Path) -> Option<AuthenticatedTllModel> {
    if path.extension().and_then(|s| s.to_str()) != Some("onnx") {
        return None;
    }
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_MODEL_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if u64::try_from(bytes.len()).ok()? != metadata.len() {
        return None;
    }
    wire_audit::audit_model_wire(&bytes)?;
    let mut source = bytes.as_slice();
    let model = AuditedModelProto::decode(&mut source).ok()?;
    if !source.is_empty() {
        return None;
    }
    let structure = authenticate_model(&model)?;
    Some(AuthenticatedTllModel {
        structure,
        source_sha256: Sha256::digest(&bytes).into(),
    })
}

/// Couples the decoded algebraic proof to the exact source bytes it audited.
/// The caller rechecks this seal immediately before publishing a verdict, so a
/// path replacement during diagnostic NY/ORT reloads cannot change authority.
pub(super) struct AuthenticatedTllModel {
    structure: TllStructure,
    source_sha256: [u8; 32],
}

impl AuthenticatedTllModel {
    pub(super) fn structure(&self) -> &TllStructure {
        &self.structure
    }

    pub(super) fn source_still_matches(&self, path: &Path) -> bool {
        let Ok(metadata) = std::fs::metadata(path) else {
            return false;
        };
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_MODEL_BYTES {
            return false;
        }
        let Ok(bytes) = std::fs::read(path) else {
            return false;
        };
        u64::try_from(bytes.len()).ok() == Some(metadata.len())
            && <[u8; 32]>::from(Sha256::digest(&bytes)) == self.source_sha256
    }
}

fn authenticate_model(model: &AuditedModelProto) -> Option<TllStructure> {
    // This is the exact standard-domain encoding used by the audited corpus.
    // Being narrower than ONNX's full compatibility range is intentional.
    if model.ir_version != 7
        || model.opset_import.len() != 1
        || !model.opset_import[0].domain.is_empty()
        || model.opset_import[0].version != 13
        || !model.training_info.is_empty()
        || !model.functions.is_empty()
        || !model.device_configurations.is_empty()
    {
        return None;
    }
    let graph = model.graph.as_ref()?;
    if graph.node.is_empty()
        || graph.node.len() > MAX_NODES
        || graph.initializer.is_empty()
        || graph.initializer.len() > MAX_INITIALIZERS
        || !graph.sparse_initializer.is_empty()
        || graph.input.len() != 1
        || graph.output.len() != 1
        || !float_matrix_endpoint(&graph.input[0], 2)
        || !float_matrix_endpoint(&graph.output[0], 1)
    {
        return None;
    }

    let input_name = graph.input[0].name.as_str();
    let output_name = graph.output[0].name.as_str();
    if input_name.is_empty() || output_name.is_empty() || input_name == output_name {
        return None;
    }

    // Duplicate initializer or node names are rejected instead of selecting an
    // arbitrary substring/name match. Every initializer is validated before
    // graph decoding, including unused ones (which are rejected below).
    let mut initializers: HashMap<&str, &TensorProto> = HashMap::new();
    let mut total_elements = 0usize;
    for tensor in &graph.initializer {
        if tensor.name.is_empty() || initializers.insert(tensor.name.as_str(), tensor).is_some() {
            return None;
        }
        let raw = RawF32Tensor::new(tensor)?;
        total_elements = total_elements.checked_add(raw.len())?;
        if total_elements > MAX_FLOAT_ELEMENTS {
            return None;
        }
    }
    if initializers.contains_key(input_name) || initializers.contains_key(output_name) {
        return None;
    }

    let mut node_names = HashSet::new();
    for node in &graph.node {
        if node.name.is_empty()
            || !node_names.insert(node.name.as_str())
            || !node.domain.is_empty()
            || !node.overload.is_empty()
            || !node.device_configurations.is_empty()
            || !node.attribute.is_empty()
            || node.output.len() != 1
            || node.output[0].is_empty()
            || node.input.iter().any(String::is_empty)
        {
            return None;
        }
    }

    let mut cursor = RawChain::new(graph, input_name, initializers.keys().copied())?;
    let mut used_initializers = HashSet::new();

    // Authored local affine functions L_i(x) = a_i*x + b_i.
    let linear_w = cursor.take_dense(&initializers, &mut used_initializers)?;
    let [input_dim, n] = linear_w.matrix_shape()?;
    if input_dim != 2 || !(2..=128).contains(&n) {
        return None;
    }
    let linear_b = cursor.take_bias(&initializers, &mut used_initializers)?;
    if linear_b.vector_len()? != n {
        return None;
    }
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        let a0 = linear_w.at2(0, i)?;
        let a1 = linear_w.at2(1, i)?;
        let bias = linear_b.at1(i)?;
        if !(a0.is_finite() && a1.is_finite() && bias.is_finite()) {
            return None;
        }
        a.push([f32_to_f64_exact(a0), f32_to_f64_exact(a1)]);
        b.push(f32_to_f64_exact(bias));
    }

    // The selection layer must be bit-exact one-hot with an exact zero bias.
    let selection_w = cursor.take_dense(&initializers, &mut used_initializers)?;
    let [selection_rows, selected_len] = selection_w.matrix_shape()?;
    if selection_rows != n || !(2..=16_384).contains(&selected_len) {
        return None;
    }
    let selection_b = cursor.take_bias(&initializers, &mut used_initializers)?;
    if selection_b.vector_len()? != selected_len || !selection_b.all_exact(0.0) {
        return None;
    }
    let mut semantics = Vec::with_capacity(selected_len);
    for column in 0..selected_len {
        let mut selected = None;
        for row in 0..n {
            match selection_w.at2(row, column)? {
                0.0 => {}
                1.0 if selected.is_none() => selected = Some(row),
                _ => return None,
            }
        }
        semantics.push(vec![selected?]);
    }

    let mut saw_min = false;
    let mut saw_max = false;
    let mut decoded_groups: Option<Vec<Vec<usize>>> = None;
    while cursor.position() < graph.node.len() {
        let first_w = cursor.take_dense(&initializers, &mut used_initializers)?;
        let first_b = cursor.take_bias(&initializers, &mut used_initializers)?;
        cursor.take_relu()?;
        let second_w = cursor.take_dense(&initializers, &mut used_initializers)?;
        let second_b = cursor.take_bias(&initializers, &mut used_initializers)?;
        let (kind, operands) =
            authenticate_lattice_stage(first_w, first_b, second_w, second_b, semantics.len())?;

        match kind {
            LatticeKind::Min if !saw_max => {
                saw_min = true;
            }
            LatticeKind::Min => return None, // max-then-min is not this TLL form
            LatticeKind::Max => {
                if !saw_min {
                    return None;
                }
                if !saw_max {
                    decoded_groups = Some(semantics.clone());
                    semantics = (0..semantics.len()).map(|i| vec![i]).collect();
                }
                saw_max = true;
            }
        }
        semantics = combine_semantics(&semantics, &operands)?;
    }

    let groups = decoded_groups?;
    if !saw_min
        || !saw_max
        || groups.is_empty()
        || groups.iter().any(Vec::is_empty)
        || semantics.len() != 1
        || semantics[0] != (0..groups.len()).collect::<Vec<_>>()
        || cursor.current() != output_name
        || used_initializers.len() != initializers.len()
    {
        return None;
    }

    Some(TllStructure {
        a,
        b,
        groups,
        max_of_min: true,
    })
}

fn float_matrix_endpoint(value: &ValueInfoProto, trailing: i64) -> bool {
    let Some(tensor) = value.r#type.as_ref().and_then(|t| t.tensor_type.as_ref()) else {
        return false;
    };
    let Some(shape) = tensor.shape.as_ref() else {
        return false;
    };
    if tensor.elem_type != FLOAT || shape.dim.len() != 2 {
        return false;
    }
    let batch_ok = match shape.dim[0].value.as_ref() {
        Some(DimValue::DimValue(v)) => *v > 0,
        Some(DimValue::DimParam(s)) => !s.is_empty(),
        None => false,
    };
    batch_ok && matches!(shape.dim[1].value, Some(DimValue::DimValue(v)) if v == trailing)
}

#[derive(Clone)]
struct RawF32Tensor<'a> {
    shape: Vec<usize>,
    bytes: &'a [u8],
}

impl<'a> RawF32Tensor<'a> {
    fn new(tensor: &'a TensorProto) -> Option<Self> {
        if tensor.data_type != FLOAT
            || tensor.segment.is_some()
            || tensor.data_location != 0
            || !tensor.external_data.is_empty()
            || !tensor.float_data.is_empty()
            || !tensor.int32_data.is_empty()
            || !tensor.int64_data.is_empty()
            || !tensor.double_data.is_empty()
            || !tensor.string_data.is_empty()
            || !tensor.uint64_data.is_empty()
            || tensor.dims.is_empty()
        {
            return None;
        }
        let mut shape = Vec::with_capacity(tensor.dims.len());
        let mut len = 1usize;
        for &dim in &tensor.dims {
            let dim = usize::try_from(dim).ok()?;
            if dim == 0 {
                return None;
            }
            len = len.checked_mul(dim)?;
            shape.push(dim);
        }
        if len > MAX_FLOAT_ELEMENTS || tensor.raw_data.len() != len.checked_mul(4)? {
            return None;
        }
        Some(Self {
            shape,
            bytes: &tensor.raw_data,
        })
    }

    fn len(&self) -> usize {
        self.bytes.len() / 4
    }

    fn matrix_shape(&self) -> Option<[usize; 2]> {
        (self.shape.len() == 2).then(|| [self.shape[0], self.shape[1]])
    }

    fn vector_len(&self) -> Option<usize> {
        (self.shape.len() == 1).then(|| self.shape[0])
    }

    fn at1(&self, index: usize) -> Option<f32> {
        (self.vector_len()? > index)
            .then(|| self.at_flat(index))
            .flatten()
    }

    fn at2(&self, row: usize, column: usize) -> Option<f32> {
        let [rows, columns] = self.matrix_shape()?;
        if row >= rows || column >= columns {
            return None;
        }
        self.at_flat(row.checked_mul(columns)?.checked_add(column)?)
    }

    fn at_flat(&self, index: usize) -> Option<f32> {
        let start = index.checked_mul(4)?;
        let bytes: [u8; 4] = self
            .bytes
            .get(start..start.checked_add(4)?)?
            .try_into()
            .ok()?;
        Some(f32::from_le_bytes(bytes))
    }

    fn all_exact(&self, expected: f32) -> bool {
        (0..self.len()).all(|i| self.at_flat(i) == Some(expected))
    }
}

/// Cursor over the sole activation chain. Outputs may never shadow graph
/// inputs, initializers, or prior outputs, and no branch/dead node can remain.
struct RawChain<'a> {
    nodes: &'a [AuditedNodeProto],
    position: usize,
    current: String,
    seen_values: HashSet<String>,
}

impl<'a> RawChain<'a> {
    fn new(
        graph: &'a AuditedGraphProto,
        input: &str,
        initializer_names: impl Iterator<Item = &'a str>,
    ) -> Option<Self> {
        let mut seen_values: HashSet<String> = initializer_names.map(str::to_owned).collect();
        if !seen_values.insert(input.to_owned()) {
            return None;
        }
        Some(Self {
            nodes: &graph.node,
            position: 0,
            current: input.to_owned(),
            seen_values,
        })
    }

    fn take_dense<'b>(
        &mut self,
        initializers: &HashMap<&'b str, &'b TensorProto>,
        used: &mut HashSet<String>,
    ) -> Option<RawF32Tensor<'b>> {
        let node = self.nodes.get(self.position)?;
        if node.op_type != "MatMul" || node.input.len() != 2 || node.input[0] != self.current {
            return None;
        }
        let tensor = *initializers.get(node.input[1].as_str())?;
        used.insert(node.input[1].clone());
        self.advance(node.output[0].clone())?;
        RawF32Tensor::new(tensor)
    }

    fn take_bias<'b>(
        &mut self,
        initializers: &HashMap<&'b str, &'b TensorProto>,
        used: &mut HashSet<String>,
    ) -> Option<RawF32Tensor<'b>> {
        let node = self.nodes.get(self.position)?;
        if node.op_type != "Add" || node.input.len() != 2 || node.input[0] != self.current {
            return None;
        }
        let tensor = *initializers.get(node.input[1].as_str())?;
        used.insert(node.input[1].clone());
        self.advance(node.output[0].clone())?;
        RawF32Tensor::new(tensor)
    }

    fn take_relu(&mut self) -> Option<()> {
        let node = self.nodes.get(self.position)?;
        if node.op_type != "Relu" || node.input.len() != 1 || node.input[0] != self.current {
            return None;
        }
        self.advance(node.output[0].clone())
    }

    fn advance(&mut self, output: String) -> Option<()> {
        if output.is_empty() || !self.seen_values.insert(output.clone()) {
            return None;
        }
        self.current = output;
        self.position = self.position.checked_add(1)?;
        Some(())
    }

    fn position(&self) -> usize {
        self.position
    }

    fn current(&self) -> &str {
        &self.current
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LatticeKind {
    Min,
    Max,
}

/// Prove one complete comparator stage and return the two prior-vector indices
/// consumed by each output. Every prior value must participate at least once;
/// overlapping pairs (the benchmark's exact odd-width construction) are safe.
fn authenticate_lattice_stage(
    first_w: RawF32Tensor<'_>,
    first_b: RawF32Tensor<'_>,
    second_w: RawF32Tensor<'_>,
    second_b: RawF32Tensor<'_>,
    input_len: usize,
) -> Option<(LatticeKind, Vec<(usize, usize)>)> {
    let [first_rows, first_columns] = first_w.matrix_shape()?;
    let [second_rows, output_len] = second_w.matrix_shape()?;
    if first_rows != input_len
        || output_len == 0
        || output_len >= input_len
        || output_len.checked_mul(2)? < input_len
        || first_columns != output_len.checked_mul(4)?
        || second_rows != first_columns
        || first_b.vector_len()? != first_columns
        || second_b.vector_len()? != output_len
        || !first_b.all_exact(0.0)
        || !second_b.all_exact(0.0)
    {
        return None;
    }

    let kind = match (second_w.at2(2, 0)?, second_w.at2(3, 0)?) {
        (-0.5, -0.5) => LatticeKind::Min,
        (0.5, 0.5) => LatticeKind::Max,
        _ => return None,
    };
    let tail = match kind {
        LatticeKind::Min => -0.5,
        LatticeKind::Max => 0.5,
    };

    // W1 must contain exactly one [.5,-.5,+/- .5,+/- .5] block per output.
    for row in 0..second_rows {
        for column in 0..output_len {
            let expected = if row / 4 == column {
                match row % 4 {
                    0 => 0.5,
                    1 => -0.5,
                    _ => tail,
                }
            } else {
                0.0
            };
            if second_w.at2(row, column)? != expected {
                return None;
            }
        }
    }

    let mut operands = Vec::with_capacity(output_len);
    let mut covered = vec![false; input_len];
    for output in 0..output_len {
        let base = output.checked_mul(4)?;
        let mut rows = Vec::with_capacity(2);
        for row in 0..input_len {
            match first_w.at2(row, base)? {
                0.0 => {}
                1.0 => rows.push(row),
                _ => return None,
            }
        }
        if rows.len() != 2 {
            return None;
        }
        let (left, right) = (rows[0], rows[1]);
        let orientation = match (first_w.at2(left, base + 2)?, first_w.at2(right, base + 2)?) {
            (-1.0, 1.0) => -1.0,
            (1.0, -1.0) => 1.0,
            _ => return None,
        };
        for row in 0..input_len {
            let selected = row == left || row == right;
            let expected0 = if selected { 1.0 } else { 0.0 };
            let expected1 = if selected { -1.0 } else { 0.0 };
            let expected2 = if row == left {
                orientation
            } else if row == right {
                -orientation
            } else {
                0.0
            };
            if first_w.at2(row, base)? != expected0
                || first_w.at2(row, base + 1)? != expected1
                || first_w.at2(row, base + 2)? != expected2
                || first_w.at2(row, base + 3)? != -expected2
            {
                return None;
            }
        }
        covered[left] = true;
        covered[right] = true;
        operands.push((left, right));
    }
    if covered.iter().any(|covered| !covered) {
        return None;
    }
    Some((kind, operands))
}

fn combine_semantics(prior: &[Vec<usize>], operands: &[(usize, usize)]) -> Option<Vec<Vec<usize>>> {
    operands
        .iter()
        .map(|&(left, right)| {
            let mut combined = prior.get(left)?.clone();
            combined.extend_from_slice(prior.get(right)?);
            combined.sort_unstable();
            combined.dedup();
            Some(combined)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "external-vnncomp")]
    use std::io::Read;

    #[cfg(feature = "external-vnncomp")]
    use flate2::read::GzDecoder;

    use super::*;
    use ny_onnx::onnx_proto::tensor_shape_proto::{dimension, Dimension};
    use ny_onnx::onnx_proto::{TensorShapeProto, TensorTypeProto, TypeProto};

    #[derive(Clone, PartialEq, prost::Message)]
    struct ModelWithRawGraph {
        #[prost(int64, tag = "1")]
        ir_version: i64,
        #[prost(message, repeated, tag = "8")]
        opset_import: Vec<OperatorSetIdProto>,
        // `bytes` and an embedded message share wire type 2, letting this test
        // preserve an otherwise canonical graph with one hand-authored tag.
        #[prost(bytes = "vec", tag = "7")]
        graph: Vec<u8>,
    }

    fn tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
        TensorProto {
            dims: dims.to_vec(),
            data_type: FLOAT,
            name: name.to_owned(),
            raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ..Default::default()
        }
    }

    fn value_info(name: &str, trailing: i64) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_owned(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: FLOAT,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            Dimension {
                                value: Some(dimension::Value::DimParam("batch".to_owned())),
                            },
                            Dimension {
                                value: Some(dimension::Value::DimValue(trailing)),
                            },
                        ],
                    }),
                }),
            }),
        }
    }

    fn node(name: &str, op: &str, inputs: &[&str], output: &str) -> AuditedNodeProto {
        AuditedNodeProto {
            input: inputs.iter().map(|s| (*s).to_owned()).collect(),
            output: vec![output.to_owned()],
            name: name.to_owned(),
            op_type: op.to_owned(),
            domain: String::new(),
            attribute: Vec::new(),
            overload: String::new(),
            device_configurations: Vec::new(),
        }
    }

    fn comparator_tensors(
        prefix: &str,
        input_len: usize,
        pairs: &[(usize, usize)],
        min: bool,
    ) -> Vec<TensorProto> {
        let q = pairs.len();
        let mut first = vec![0.0; input_len * 4 * q];
        let mut second = vec![0.0; 4 * q * q];
        for (j, &(left, right)) in pairs.iter().enumerate() {
            let base = 4 * j;
            for &(row, sign) in &[(left, -1.0f32), (right, 1.0)] {
                first[row * (4 * q) + base] = 1.0;
                first[row * (4 * q) + base + 1] = -1.0;
                first[row * (4 * q) + base + 2] = sign;
                first[row * (4 * q) + base + 3] = -sign;
            }
            for (offset, value) in [
                0.5,
                -0.5,
                if min { -0.5 } else { 0.5 },
                if min { -0.5 } else { 0.5 },
            ]
            .into_iter()
            .enumerate()
            {
                second[(base + offset) * q + j] = value;
            }
        }
        vec![
            tensor(
                &format!("{prefix}.w0"),
                &[input_len as i64, (4 * q) as i64],
                &first,
            ),
            tensor(
                &format!("{prefix}.b0"),
                &[(4 * q) as i64],
                &vec![0.0; 4 * q],
            ),
            tensor(
                &format!("{prefix}.w1"),
                &[(4 * q) as i64, q as i64],
                &second,
            ),
            tensor(&format!("{prefix}.b1"), &[q as i64], &vec![0.0; q]),
        ]
    }

    fn canonical_model() -> AuditedModelProto {
        let mut initializer = vec![
            tensor("linear.w", &[2, 2], &[1.0, -2.0, 0.5, 3.0]),
            tensor("linear.b", &[2], &[0.25, -0.75]),
            // [L0,L1,L1,L0]: two selector groups, both min(L0,L1).
            tensor(
                "select.w",
                &[2, 4],
                &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            ),
            tensor("select.b", &[4], &[0.0; 4]),
        ];
        initializer.extend(comparator_tensors("min", 4, &[(0, 1), (2, 3)], true));
        initializer.extend(comparator_tensors("max", 2, &[(0, 1)], false));

        let mut nodes = vec![
            node("linear.mm", "MatMul", &["X", "linear.w"], "v0"),
            node("linear.add", "Add", &["v0", "linear.b"], "v1"),
            node("select.mm", "MatMul", &["v1", "select.w"], "v2"),
            node("select.add", "Add", &["v2", "select.b"], "v3"),
        ];
        let mut current = "v3".to_owned();
        for (stage, prefix) in [("min", "m"), ("max", "x")] {
            let outputs = [
                format!("{prefix}0"),
                format!("{prefix}1"),
                format!("{prefix}2"),
                format!("{prefix}3"),
                if stage == "max" {
                    "Y".to_owned()
                } else {
                    format!("{prefix}4")
                },
            ];
            nodes.extend([
                node(
                    &format!("{stage}.mm0"),
                    "MatMul",
                    &[&current, &format!("{stage}.w0")],
                    &outputs[0],
                ),
                node(
                    &format!("{stage}.add0"),
                    "Add",
                    &[&outputs[0], &format!("{stage}.b0")],
                    &outputs[1],
                ),
                node(
                    &format!("{stage}.relu"),
                    "Relu",
                    &[&outputs[1]],
                    &outputs[2],
                ),
                node(
                    &format!("{stage}.mm1"),
                    "MatMul",
                    &[&outputs[2], &format!("{stage}.w1")],
                    &outputs[3],
                ),
                node(
                    &format!("{stage}.add1"),
                    "Add",
                    &[&outputs[3], &format!("{stage}.b1")],
                    &outputs[4],
                ),
            ]);
            current = outputs[4].clone();
        }
        AuditedModelProto {
            ir_version: 7,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(AuditedGraphProto {
                node: nodes,
                name: "canonical_tll".to_owned(),
                initializer,
                sparse_initializer: Vec::new(),
                input: vec![value_info("X", 2)],
                output: vec![value_info("Y", 1)],
            }),
            training_info: Vec::new(),
            functions: Vec::new(),
            device_configurations: Vec::new(),
        }
    }

    fn tensor_mut<'a>(model: &'a mut AuditedModelProto, name: &str) -> &'a mut TensorProto {
        model
            .graph
            .as_mut()
            .unwrap()
            .initializer
            .iter_mut()
            .find(|t| t.name == name)
            .unwrap()
    }

    #[test]
    fn exact_canonical_graph_is_authenticated() {
        let tll = authenticate_model(&canonical_model()).expect("canonical identity");
        assert_eq!(tll.a, vec![[1.0, 0.5], [-2.0, 3.0]]);
        assert_eq!(tll.b, vec![0.25, -0.75]);
        assert_eq!(tll.groups, vec![vec![0, 1], vec![0, 1]]);
        assert!(tll.max_of_min);
    }

    #[test]
    fn approximate_selector_bit_is_rejected() {
        let mut model = canonical_model();
        let selection = tensor_mut(&mut model, "select.w");
        selection.raw_data[0..4]
            .copy_from_slice(&f32::from_bits(1.0f32.to_bits() + 1).to_le_bytes());
        assert!(authenticate_model(&model).is_none());
    }

    #[test]
    fn approximate_gadget_coefficient_is_rejected() {
        let mut model = canonical_model();
        let weight = tensor_mut(&mut model, "min.w1");
        weight.raw_data[0..4].copy_from_slice(&f32::from_bits(0.5f32.to_bits() + 1).to_le_bytes());
        assert!(authenticate_model(&model).is_none());
    }

    #[test]
    fn duplicate_names_extra_branches_and_dead_initializers_are_rejected() {
        let mut duplicate = canonical_model();
        duplicate.graph.as_mut().unwrap().node[1].name =
            duplicate.graph.as_ref().unwrap().node[0].name.clone();
        assert!(authenticate_model(&duplicate).is_none());

        let mut branch = canonical_model();
        branch
            .graph
            .as_mut()
            .unwrap()
            .node
            .push(node("dead", "Relu", &["v1"], "dead.out"));
        assert!(authenticate_model(&branch).is_none());

        let mut dead_initializer = canonical_model();
        dead_initializer
            .graph
            .as_mut()
            .unwrap()
            .initializer
            .push(tensor("dead.weight", &[1], &[0.0]));
        assert!(authenticate_model(&dead_initializer).is_none());
    }

    #[test]
    fn endpoint_dtype_attributes_and_custom_ops_are_rejected() {
        let mut dtype = canonical_model();
        dtype.graph.as_mut().unwrap().input[0]
            .r#type
            .as_mut()
            .unwrap()
            .tensor_type
            .as_mut()
            .unwrap()
            .elem_type = 11;
        assert!(authenticate_model(&dtype).is_none());

        let mut attribute = canonical_model();
        attribute.graph.as_mut().unwrap().node[0]
            .attribute
            .push(Default::default());
        assert!(authenticate_model(&attribute).is_none());

        let mut custom = canonical_model();
        custom.graph.as_mut().unwrap().node[6].op_type = "LeakyRelu".to_owned();
        assert!(authenticate_model(&custom).is_none());

        let mut overload = canonical_model();
        overload.graph.as_mut().unwrap().node[0].overload = "local".to_owned();
        assert!(authenticate_model(&overload).is_none());

        let mut function = canonical_model();
        function.functions.push(vec![0]);
        assert!(authenticate_model(&function).is_none());
    }

    #[test]
    fn altered_edge_output_and_typed_payload_are_rejected() {
        let mut edge = canonical_model();
        edge.graph.as_mut().unwrap().node[5].input[0] = "v1".to_owned();
        assert!(authenticate_model(&edge).is_none());

        let mut output = canonical_model();
        output.graph.as_mut().unwrap().output[0].name = "m4".to_owned();
        assert!(authenticate_model(&output).is_none());

        let mut mixed_payload = canonical_model();
        tensor_mut(&mut mixed_payload, "linear.b")
            .float_data
            .push(0.0);
        assert!(authenticate_model(&mixed_payload).is_none());
    }

    #[test]
    fn unknown_raw_graph_field_is_rejected_before_prost_can_drop_it() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("unknown-graph-field.onnx");
        let model = canonical_model();
        let mut graph = model.graph.as_ref().unwrap().encode_to_vec();
        // Field 17, varint wire type, value 1. GraphProto currently defines
        // fields only through 16; an ordinary prost decode silently skips it.
        graph.extend_from_slice(&[0x88, 0x01, 0x01]);
        let wire_model = ModelWithRawGraph {
            ir_version: model.ir_version,
            opset_import: model.opset_import,
            graph,
        };
        std::fs::write(&path, wire_model.encode_to_vec()).expect("write raw model");
        assert!(authenticate_raw_tll_model(&path).is_none());
    }

    /// Explicit external-corpus qualification without baking a workstation
    /// path into CI. Selecting the lane without its requested source is an
    /// actionable failure, never a vacuous pass.
    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn requested_real_fixture_is_authenticated() {
        let requested = std::env::var("NY_TLL_IDENTITY_FIXTURE").expect(
            "external-vnncomp TLL identity conformance requires \
             NY_TLL_IDENTITY_FIXTURE=/path/to/model.onnx[.gz]",
        );
        let requested = Path::new(&requested);
        let _materialized_directory;
        let path = if requested.extension().and_then(|value| value.to_str()) == Some("gz") {
            let mut decoder = GzDecoder::new(std::fs::File::open(requested).expect("open gzip"));
            let mut bytes = Vec::new();
            decoder.read_to_end(&mut bytes).expect("decompress model");
            _materialized_directory = tempfile::tempdir().expect("tempdir");
            let materialized = _materialized_directory.path().join("fixture.onnx");
            std::fs::write(&materialized, bytes).expect("materialize model");
            materialized
        } else {
            requested.to_owned()
        };
        let authenticated = authenticate_raw_tll_model(&path).expect("authenticate real TLL model");
        let structure = authenticated.structure();
        eprintln!(
            "authenticated real TLL: affines={}, groups={}, memberships={}",
            structure.a.len(),
            structure.groups.len(),
            structure.groups.iter().map(Vec::len).sum::<usize>()
        );
    }

    #[test]
    fn source_seal_detects_path_replacement() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("tll.onnx");
        let model = canonical_model();
        std::fs::write(&path, model.encode_to_vec()).expect("write canonical model");
        let authenticated = authenticate_raw_tll_model(&path).expect("authenticate source");
        assert!(authenticated.source_still_matches(&path));
        assert_eq!(authenticated.structure().groups.len(), 2);

        let mut replacement = canonical_model();
        tensor_mut(&mut replacement, "linear.b").raw_data[0] ^= 1;
        std::fs::write(&path, replacement.encode_to_vec()).expect("replace source");
        assert!(!authenticated.source_still_matches(&path));
    }
}
