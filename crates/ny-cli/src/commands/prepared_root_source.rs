// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Authenticate the original ONNX surface around the actual twin extractors.
//! No parameter arithmetic is duplicated here. Original batch/layout, graph
//! outputs, inference semantics and exact Add shapes are checked before the
//! immutable source bytes enter that actual extractor.

use std::{collections::HashMap, time::Instant};

use anyhow::{anyhow, bail, ensure, Result};
use ny_onnx::onnx_proto::{
    tensor_shape_proto::dimension::Value, AttributeProto, ModelProto, NodeProto, TensorProto,
};
use ny_propagate::certify::work::{LeaseClass, WorkBudget};
use prost::Message;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    rank: usize,
    dims: [usize; 4],
}
impl Shape {
    fn nchw(c: usize, h: usize, w: usize) -> Self {
        Self {
            rank: 4,
            dims: [1, c, h, w],
        }
    }
    fn matrix(n: usize) -> Self {
        Self {
            rank: 2,
            dims: [1, n, 1, 1],
        }
    }
    fn elements(self) -> Result<usize> {
        self.dims[..self.rank].iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d)
                .ok_or_else(|| anyhow!("original source shape overflow"))
        })
    }
}

pub(super) fn authenticate_original_twin_surface(
    bytes: &[u8],
    budget: &WorkBudget,
    at: Instant,
) -> Result<()> {
    checkpoint(budget, at)?;
    let model = ModelProto::decode(bytes)?;
    checkpoint(budget, at)?;
    let mut opset = None;
    for (i, import) in model.opset_import.iter().enumerate() {
        poll(budget, at, i)?;
        let standard = matches!(import.domain.as_str(), "" | "ai.onnx");
        for (j, prior) in model.opset_import[..i].iter().enumerate() {
            poll(budget, at, j)?;
            ensure!(
                import.domain != prior.domain
                    && !(standard && matches!(prior.domain.as_str(), "" | "ai.onnx")),
                "prepared source rejects duplicate or ambiguous original opsets"
            );
        }
        ensure!(import.version > 0, "invalid original opset version");
        if standard {
            ensure!(
                (11..=21).contains(&import.version),
                "prepared source requires a supported standard ONNX opset"
            );
            opset = Some(import.version);
        }
        // Nonstandard imports carry no executed semantics: every actual
        // node below must use the standard domain. Authentic SafeNLP files
        // retain an unused ai.onnx.ml import from their exporter.
    }
    let opset = opset.ok_or_else(|| anyhow!("missing original standard ONNX opset"))?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow!("missing original graph"))?;
    ensure!(
        graph.input.len() == 1 && graph.output.len() == 1 && graph.sparse_initializer.is_empty(),
        "prepared source requires exactly one original input/output and dense constants"
    );
    let input = &graph.input[0];
    ensure!(
        !input.name.is_empty(),
        "prepared source requires a real original graph input name"
    );
    let tensor = input
        .r#type
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .ok_or_else(|| anyhow!("original input must be a tensor"))?;
    ensure!(
        tensor.elem_type == 1,
        "prepared original input must be binary32"
    );
    let dims = &tensor
        .shape
        .as_ref()
        .ok_or_else(|| anyhow!("missing original input shape"))?
        .dim;
    ensure!(
        matches!(dims.len(), 2 | 4),
        "original input must be NCHW or a matrix row"
    );
    let rank_two_input = dims.len() == 2;
    let symbolic_batch = rank_two_input
        && matches!(dims[0].value.as_ref(), Some(Value::DimParam(name)) if !name.is_empty());
    let mut input_dims = [1usize; 4];
    for (i, d) in dims.iter().enumerate() {
        input_dims[i] = match &d.value {
            Some(Value::DimValue(v)) => usize::try_from(*v)?,
            Some(Value::DimParam(name)) if i == 0 && symbolic_batch && !name.is_empty() => 1,
            _ => bail!("prepared source rejects symbolic original feature dimensions"),
        };
    }
    ensure!(
        input_dims[0] == 1 && input_dims.iter().all(|&d| d > 0),
        "prepared original batch must be exactly one"
    );
    let mut tensors: HashMap<&str, &TensorProto> = HashMap::new();
    tensors.try_reserve(graph.initializer.len() + graph.node.len())?;
    for (i, tensor) in graph.initializer.iter().enumerate() {
        poll(budget, at, i)?;
        validate_tensor(tensor)?;
        ensure!(
            !tensor.name.is_empty()
                && tensor.name != input.name
                && tensors.insert(&tensor.name, tensor).is_none(),
            "prepared source rejects duplicate or overridable initializers"
        );
    }
    let mut shapes: HashMap<&str, Shape> = HashMap::new();
    shapes.try_reserve(graph.node.len() + 1)?;
    shapes.insert(
        &input.name,
        Shape {
            rank: dims.len(),
            dims: input_dims,
        },
    );
    // A symbolic matrix batch is an explicit conditional single-row domain:
    // X_j maps to the actual original input[0,j], and the file issuer checks
    // the complete raw property input count against this feature width.
    // Only batch-separable operations may carry that specialization; it is
    // not an assertion that the original graph authored DimValue(1).
    let mut final_is_affine = false;
    for (i, node) in graph.node.iter().enumerate() {
        poll(budget, at, i)?;
        final_is_affine = false;
        ensure!(
            !symbolic_batch || matches!(node.op_type.as_str(), "MatMul" | "Add" | "Relu"),
            "symbolic original batch requires a batch-separable native affine graph"
        );
        ensure!(
            matches!(node.domain.as_str(), "" | "ai.onnx")
                && node.output.len() == 1
                && !node.output[0].is_empty(),
            "prepared source rejects custom domains or multiple operator outputs"
        );
        ensure!(
            !shapes.contains_key(node.output[0].as_str())
                && !tensors.contains_key(node.output[0].as_str()),
            "prepared source tensor names must be unique"
        );
        validate_attributes(node)?;
        ensure!(
            opset >= 14
                || !node.attribute.iter().any(|a| {
                    (node.op_type == "BatchNormalization" && a.name == "training_mode")
                        || (node.op_type == "Reshape" && a.name == "allowzero")
                }),
            "original source attribute is not part of its declared operator schema"
        );
        if node.op_type == "Constant" {
            ensure!(
                node.input.is_empty()
                    && node.attribute.len() == 1
                    && node.attribute[0].name == "value",
                "prepared Constant requires one actual tensor value"
            );
            let tensor = node.attribute[0]
                .t
                .as_ref()
                .ok_or_else(|| anyhow!("prepared shape Constant has no tensor"))?;
            validate_tensor(tensor)?;
            ensure!(
                tensor.data_type == 7,
                "prepared Constant is restricted to actual reshape dimensions"
            );
            tensors.insert(&node.output[0], tensor);
            continue;
        }
        let first = node
            .input
            .first()
            .and_then(|name| shapes.get(name.as_str()))
            .or_else(|| {
                (node.op_type == "Add")
                    .then(|| node.input.get(1).and_then(|name| shapes.get(name.as_str())))
                    .flatten()
            })
            .copied()
            .ok_or_else(|| anyhow!("prepared source activation is not a prior original tensor"))?;
        let shape = match node.op_type.as_str() {
            "Conv" | "ConvTranspose" => {
                ensure!(
                    (2..=3).contains(&node.input.len()) && first.rank == 4,
                    "prepared convolution input/layout mismatch"
                );
                let weight = tensors
                    .get(node.input[1].as_str())
                    .copied()
                    .ok_or_else(|| anyhow!("missing actual original convolution weight"))?;
                let dims = positive_dims(weight)?;
                ensure!(
                    weight.data_type == 1 && dims.len() == 4,
                    "original convolution weight layout mismatch"
                );
                let transpose = node.op_type == "ConvTranspose";
                let (ci, co) = if transpose {
                    (dims[0], dims[1])
                } else {
                    (dims[1], dims[0])
                };
                ensure!(
                    ci == first.dims[1],
                    "original convolution channels mismatch"
                );
                if node.input.len() == 3 {
                    let bias = tensors
                        .get(node.input[2].as_str())
                        .copied()
                        .ok_or_else(|| anyhow!("missing convolution bias"))?;
                    ensure!(
                        bias.data_type == 1 && positive_dims(bias)? == [co],
                        "original convolution bias must be one channel vector"
                    );
                }
                ensure!(
                    scalar_i(node, "group", 1)? == 1 && ints(node, "dilations", &[1, 1])? == [1, 1],
                    "prepared original convolution supports unit group/dilation"
                );
                if let Some(attr) = attribute(node, "auto_pad") {
                    ensure!(
                        !transpose && attr.s.as_deref() == Some(b"NOTSET".as_slice()),
                        "prepared source does not implement auto padding"
                    );
                }
                if let Some(attr) = attribute(node, "kernel_shape") {
                    ensure!(
                        attr.ints == [i64::try_from(dims[2])?, i64::try_from(dims[3])?],
                        "original kernel attribute disagrees with actual weight"
                    );
                }
                let strides = ints(node, "strides", &[1, 1])?;
                let pads = ints(node, "pads", &[0, 0, 0, 0])?;
                ensure!(
                    strides.len() == 2 && pads.len() == 4,
                    "original convolution geometry arity"
                );
                let sh = positive(strides[0])?;
                let sw = positive(strides[1])?;
                let pads = [
                    nonnegative(pads[0])?,
                    nonnegative(pads[1])?,
                    nonnegative(pads[2])?,
                    nonnegative(pads[3])?,
                ];
                let (h, w) = if transpose {
                    ensure!(
                        attribute(node, "output_shape").is_none(),
                        "prepared transpose output_shape is not implemented"
                    );
                    let opad = ints(node, "output_padding", &[0, 0])?;
                    ensure!(opad.len() == 2, "original transpose output-padding arity");
                    let ph = nonnegative(opad[0])?;
                    let pw = nonnegative(opad[1])?;
                    ensure!(
                        ph < sh && pw < sw,
                        "original transpose output padding must be below stride"
                    );
                    (
                        transpose_dim(first.dims[2], dims[2], sh, pads[0], pads[2], ph)?,
                        transpose_dim(first.dims[3], dims[3], sw, pads[1], pads[3], pw)?,
                    )
                } else {
                    (
                        conv_dim(first.dims[2], dims[2], sh, pads[0], pads[2])?,
                        conv_dim(first.dims[3], dims[3], sw, pads[1], pads[3])?,
                    )
                };
                Shape::nchw(co, h, w)
            }
            "BatchNormalization" => {
                ensure!(
                    node.input.len() == 5
                        && matches!(first.rank, 2 | 4)
                        && scalar_i(node, "training_mode", 0)? == 0,
                    "prepared source requires inference-only BatchNormalization"
                );
                // The actual extractor uses a binary64 literal for the
                // absent default. Bind this source to an explicit original
                // FLOAT epsilon, so schema/default conversion cannot silently
                // substitute a different real parameter.
                ensure!(
                    attribute(node, "epsilon").is_some(),
                    "prepared BatchNormalization requires explicit original float epsilon"
                );
                let epsilon = scalar_f(node, "epsilon", 1e-5)?;
                ensure!(
                    epsilon.is_finite() && epsilon >= 0.0,
                    "original BatchNormalization epsilon invalid"
                );
                for name in &node.input[1..] {
                    let t = tensors
                        .get(name.as_str())
                        .copied()
                        .ok_or_else(|| anyhow!("missing original BatchNormalization constant"))?;
                    ensure!(
                        t.data_type == 1 && positive_dims(t)? == [first.dims[1]],
                        "original BatchNormalization constant must match channel axis"
                    );
                }
                first
            }
            "Relu" => {
                ensure!(node.input.len() == 1, "original Relu arity");
                first
            }
            "Add" => {
                ensure!(node.input.len() == 2, "original Add arity");
                if let (Some(left), Some(right)) = (
                    shapes.get(node.input[0].as_str()),
                    shapes.get(node.input[1].as_str()),
                ) {
                    ensure!(
                        left == right,
                        "prepared Add requires exact original batch/layout shape, not equal volume"
                    );
                } else {
                    let (activation, constant) = if shapes.contains_key(node.input[0].as_str()) {
                        (&node.input[0], &node.input[1])
                    } else {
                        (&node.input[1], &node.input[0])
                    };
                    let bias = tensors
                        .get(constant.as_str())
                        .copied()
                        .ok_or_else(|| anyhow!("missing actual original affine bias"))?;
                    ensure!(
                        first.rank == 2
                            && bias.data_type == 1
                            && matches!(positive_dims(bias)?.as_slice(), [n] | [1,n] if *n == first.dims[1]),
                        "original affine bias must be its actual class row vector"
                    );
                    let mut producer = None;
                    for (j, candidate) in graph.node[..i].iter().enumerate() {
                        poll(budget, at, j)?;
                        if candidate.output.first() == Some(activation) {
                            producer = Some(candidate);
                            break;
                        }
                    }
                    let producer = producer
                        .ok_or_else(|| anyhow!("missing original affine MatMul producer"))?;
                    ensure!(
                        producer.op_type == "MatMul",
                        "constant bias Add must follow actual MatMul"
                    );
                    let mut uses = 0usize;
                    for (j, consumer) in graph.node.iter().enumerate() {
                        poll(budget, at, j)?;
                        for (k, name) in consumer.input.iter().enumerate() {
                            poll(budget, at, k)?;
                            if name == activation {
                                uses = uses
                                    .checked_add(1)
                                    .ok_or_else(|| anyhow!("original affine use count overflow"))?;
                            }
                        }
                    }
                    ensure!(
                        uses == 1,
                        "original affine bias fold requires a unique MatMul consumer"
                    );
                    final_is_affine = true;
                }
                first
            }
            "MatMul" => {
                ensure!(
                    node.input.len() == 2 && first.rank == 2,
                    "original MatMul requires one matrix-row activation"
                );
                let weight = tensors
                    .get(node.input[1].as_str())
                    .copied()
                    .ok_or_else(|| {
                        anyhow!("original MatMul requires actual constant-right weight")
                    })?;
                let weight_dims = positive_dims(weight)?;
                ensure!(
                    weight.data_type == 1
                        && weight_dims.len() == 2
                        && weight_dims[0] == first.dims[1],
                    "original MatMul weight orientation or feature width mismatch"
                );
                Shape::matrix(weight_dims[1])
            }
            "Flatten" => {
                ensure!(
                    node.input.len() == 1 && scalar_i(node, "axis", 1)? == 1,
                    "prepared Flatten requires original batch-preserving axis one"
                );
                Shape::matrix(first.elements()?)
            }
            "Reshape" => {
                ensure!(
                    node.input.len() == 2 && scalar_i(node, "allowzero", 0)? == 0,
                    "prepared Reshape requires ordinary copy-zero semantics"
                );
                let t = tensors
                    .get(node.input[1].as_str())
                    .copied()
                    .ok_or_else(|| anyhow!("missing actual original reshape shape"))?;
                let requested = int64_values(t)?;
                ensure!(
                    matches!(requested.len(), 2 | 4),
                    "prepared source only retains authored batch-one matrix/NCHW reshapes"
                );
                let mut dims = [1usize; 4];
                let mut inferred = None;
                for (axis, &d) in requested.iter().enumerate() {
                    dims[axis] = match d {
                        -1 => {
                            ensure!(
                                inferred.replace(axis).is_none(),
                                "multiple original inferred reshape dimensions"
                            );
                            1
                        }
                        0 => {
                            ensure!(
                                axis == 0,
                                "prepared reshape only supports copying exact batch one"
                            );
                            1
                        }
                        _ => positive(d)?,
                    };
                }
                let known = dims[..requested.len()].iter().try_fold(1usize, |n, &d| {
                    n.checked_mul(d)
                        .ok_or_else(|| anyhow!("reshape volume overflow"))
                })?;
                let total = first.elements()?;
                if let Some(axis) = inferred {
                    ensure!(total % known == 0, "original reshape does not divide");
                    dims[axis] = total / known;
                }
                let shape = Shape {
                    rank: requested.len(),
                    dims,
                };
                ensure!(
                    shape.dims[0] == 1 && shape.elements()? == total,
                    "original reshape changes batch or volume"
                );
                shape
            }
            "Gemm" => {
                final_is_affine = true;
                ensure!(
                    (2..=3).contains(&node.input.len())
                        && first.rank == 2
                        && scalar_i(node, "transA", 0)? == 0
                        && scalar_f(node, "alpha", 1.0)? == 1.0
                        && scalar_f(node, "beta", 1.0)? == 1.0,
                    "prepared original Gemm semantics mismatch"
                );
                let t = tensors
                    .get(node.input[1].as_str())
                    .copied()
                    .ok_or_else(|| anyhow!("missing original Gemm weight"))?;
                let dims = positive_dims(t)?;
                ensure!(
                    t.data_type == 1 && dims.len() == 2,
                    "original Gemm weight matrix mismatch"
                );
                let trans = scalar_i(node, "transB", 0)?;
                ensure!(matches!(trans, 0 | 1), "invalid original Gemm transpose");
                let (nin, nout) = if trans == 1 {
                    (dims[1], dims[0])
                } else {
                    (dims[0], dims[1])
                };
                ensure!(
                    nin == first.dims[1],
                    "original Gemm input dimension mismatch"
                );
                if node.input.len() == 3 {
                    let b = tensors
                        .get(node.input[2].as_str())
                        .copied()
                        .ok_or_else(|| anyhow!("missing original Gemm bias"))?;
                    ensure!(
                        b.data_type == 1
                            && matches!(positive_dims(b)?.as_slice(),[n] | [1,n] if *n==nout),
                        "prepared Gemm bias is restricted to its actual row vector"
                    );
                }
                Shape::matrix(nout)
            }
            _ => bail!("prepared source contains unsupported original operator"),
        };
        ensure!(shape.elements()? > 0, "empty original source tensor");
        shapes.insert(&node.output[0], shape);
    }
    let final_node = graph
        .node
        .last()
        .ok_or_else(|| anyhow!("empty original graph"))?;
    ensure!(
        final_is_affine && final_node.output[0] == graph.output[0].name,
        "prepared objective must be actual final original affine output"
    );
    let output_shape = shapes
        .get(graph.output[0].name.as_str())
        .ok_or_else(|| anyhow!("missing actual original output tensor"))?;
    ensure!(
        output_shape.rank == 2 && output_shape.dims[0] == 1,
        "prepared original output must be one class row"
    );
    let declared_output = graph.output[0]
        .r#type
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .ok_or_else(|| anyhow!("missing original output tensor type"))?;
    let declared_dims = &declared_output
        .shape
        .as_ref()
        .ok_or_else(|| anyhow!("missing original output shape"))?
        .dim;
    ensure!(
        declared_output.elem_type == 1
            && declared_dims.len() == 2
            && (matches!(declared_dims[0].value.as_ref(), Some(Value::DimValue(1)))
                || (rank_two_input
                    && matches!(declared_dims[0].value.as_ref(),Some(Value::DimParam(name)) if !name.is_empty())))
            && matches!(declared_dims[1].value.as_ref(),Some(Value::DimValue(n)) if usize::try_from(*n).ok()==Some(output_shape.dims[1])),
        "original declared output is not the actual final class tensor"
    );
    checkpoint(budget, at)
}

fn validate_tensor(t: &TensorProto) -> Result<()> {
    ensure!(
        t.segment.is_none()
            && t.data_location == 0
            && t.external_data.is_empty()
            && t.string_data.is_empty()
            && t.int32_data.is_empty()
            && t.double_data.is_empty()
            && t.uint64_data.is_empty(),
        "prepared source requires full inline original float/int64 tensors"
    );
    let dims = positive_dims(t)?;
    let count = dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| anyhow!("original tensor count overflow"))
    })?;
    let element_bytes = match t.data_type {
        1 => {
            ensure!(t.int64_data.is_empty(), "mixed original float payload");
            4
        }
        7 => {
            ensure!(t.float_data.is_empty(), "mixed original int64 payload");
            8
        }
        _ => bail!("unsupported original tensor data type"),
    };
    if t.raw_data.is_empty() {
        let actual = if t.data_type == 1 {
            t.float_data.len()
        } else {
            t.int64_data.len()
        };
        ensure!(actual == count, "original typed payload count mismatch");
    } else {
        ensure!(
            t.float_data.is_empty()
                && t.int64_data.is_empty()
                && t.raw_data.len()
                    == count
                        .checked_mul(element_bytes)
                        .ok_or_else(|| anyhow!("original raw payload overflow"))?,
            "ambiguous or partial original raw payload"
        );
    }
    Ok(())
}
fn positive_dims(t: &TensorProto) -> Result<Vec<usize>> {
    let mut dims = Vec::new();
    dims.try_reserve_exact(t.dims.len())?;
    ensure!(
        dims.capacity() <= t.dims.len(),
        "original shape allocator exceeded admitted capacity"
    );
    for &d in &t.dims {
        dims.push(positive(d)?);
    }
    Ok(dims)
}
fn positive(d: i64) -> Result<usize> {
    let n = usize::try_from(d)?;
    ensure!(n > 0, "zero original dimension/stride");
    Ok(n)
}
fn nonnegative(d: i64) -> Result<usize> {
    usize::try_from(d).map_err(Into::into)
}
fn int64_values(t: &TensorProto) -> Result<Vec<i64>> {
    ensure!(
        t.data_type == 7 && t.dims.len() == 1,
        "original reshape shape is not a one-dimensional int64 tensor"
    );
    if t.raw_data.is_empty() {
        Ok(t.int64_data.clone())
    } else {
        Ok(t.raw_data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| i64::from_le_bytes(*bytes))
            .collect())
    }
}
fn attribute<'a>(node: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}
fn scalar_i(node: &NodeProto, name: &str, default: i64) -> Result<i64> {
    Ok(attribute(node, name).map_or(default, |a| a.i.unwrap_or(0)))
}
fn scalar_f(node: &NodeProto, name: &str, default: f64) -> Result<f64> {
    Ok(attribute(node, name).map_or(default, |a| ny_core::f32_to_f64_exact(a.f.unwrap_or(0.0))))
}
fn ints<'a>(node: &'a NodeProto, name: &str, default: &'a [i64]) -> Result<&'a [i64]> {
    Ok(attribute(node, name).map_or(default, |a| a.ints.as_slice()))
}
fn validate_attributes(node: &NodeProto) -> Result<()> {
    for (i, a) in node.attribute.iter().enumerate() {
        ensure!(
            a.ref_attr_name.is_empty() && !node.attribute[..i].iter().any(|old| old.name == a.name),
            "prepared source rejects referenced or duplicate attributes"
        );
        let expected = match (node.op_type.as_str(), a.name.as_str()) {
            ("Conv" | "ConvTranspose", "group")
            | ("BatchNormalization", "training_mode")
            | ("Flatten", "axis")
            | ("Reshape", "allowzero")
            | ("Gemm", "transA" | "transB") => 2,
            ("BatchNormalization", "epsilon" | "momentum") | ("Gemm", "alpha" | "beta") => 1,
            ("Conv" | "ConvTranspose", "auto_pad") => 3,
            ("Conv" | "ConvTranspose", "kernel_shape" | "dilations" | "strides" | "pads")
            | ("ConvTranspose", "output_padding" | "output_shape") => 7,
            ("Constant", "value") => 4,
            _ => bail!("prepared source rejects unmodeled original attribute"),
        };
        ensure!(
            a.r#type == expected
                && a.g.is_none()
                && a.sparse_tensor.is_none()
                && a.tp.is_none()
                && a.floats.is_empty()
                && a.strings.is_empty()
                && a.tensors.is_empty()
                && a.graphs.is_empty()
                && a.sparse_tensors.is_empty()
                && a.type_protos.is_empty(),
            "original attribute payload kind mismatch"
        );
        ensure!(
            (expected == 1 || a.f.is_none())
                && (expected == 2 || a.i.is_none())
                && (expected == 3 || a.s.is_none())
                && (expected == 4 || a.t.is_none())
                && (expected == 7 || a.ints.is_empty()),
            "original attribute has inactive payload"
        );
    }
    Ok(())
}
fn conv_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    before: usize,
    after: usize,
) -> Result<usize> {
    let span = input
        .checked_add(before)
        .and_then(|n| n.checked_add(after))
        .and_then(|n| n.checked_sub(kernel))
        .ok_or_else(|| anyhow!("original convolution geometry overflow"))?;
    span.checked_div(stride)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow!("original convolution stride overflow"))
}
fn transpose_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    before: usize,
    after: usize,
    opad: usize,
) -> Result<usize> {
    input
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(kernel))
        .and_then(|n| n.checked_add(opad))
        .and_then(|n| n.checked_sub(before))
        .and_then(|n| n.checked_sub(after))
        .filter(|&n| n > 0)
        .ok_or_else(|| anyhow!("original transpose convolution geometry overflow"))
}
pub(super) fn checkpoint(budget: &WorkBudget, at: Instant) -> Result<()> {
    budget.checkpoint_for(LeaseClass::Fallback)?;
    ensure!(
        Instant::now() < at,
        "original model authentication deadline expired"
    );
    ny_core::require_f64_interval_proof_environment()?;
    Ok(())
}
pub(super) fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(256) {
        checkpoint(budget, at)?;
    }
    Ok(())
}
