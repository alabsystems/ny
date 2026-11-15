// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ORT-free STATIC shape inference — the fallback the loaders use ONLY when
//! ONNX Runtime shape inference is unavailable.
//!
//! # Why this exists
//!
//! ny asks ONNX Runtime for every tensor shape, and the CLI runs that request
//! in a self-exec child (`shape_infer::subprocess`) so a native ORT abort
//! cannot take the verifier down. When the child cannot run at all — measured
//! 2026-09-28 on the x86_64 build under qemu-user without binfmt, where every
//! load logged `shape-infer subprocess exited with exit status: 127` — the
//! loader used to proceed with NO inferred shapes. That is sound but costly:
//! linearizenn `AllInOne_120_120` then refuses its `Concat` ("input
//! '/controller/MatMul_output_0' has no recorded ONNX shape") and lsnc
//! `relu_quadrotor2d_state` keeps 6-8 `OpaqueSkip` nodes, so the lowdim lane
//! declines. Both are banked `unsat` rows that turn into `unknown`. The same
//! loss would follow from anything that blocks the self-exec: a sandbox, a
//! `noexec` mount, `ENOEXEC`, an fd limit.
//!
//! # Contract: a WRONG shape must be impossible
//!
//! The fallback derives a shape only where the ONNX rule is EXACT and every
//! input it depends on is itself exactly known. Anything else — an op it does
//! not model, a non-constant parameter, a symbolic dimension, an attribute it
//! does not recognise, an ill-formed graph, an inconsistent payload — leaves
//! the tensor UNKNOWN, which is precisely today's behaviour for every tensor
//! when ORT is unavailable. Concretely:
//!
//! * Seeds are only (a) initializer dims whose inline payload length matches
//!   those dims, (b) Constant-node tensors checked the same way, and (c)
//!   runtime graph-input declarations after the SAME batch pin the ORT path
//!   applies (`batch_resolve::resolve_batch_dim`), and only when every dim is
//!   a concrete positive `dim_value`. Intermediate `value_info` is never
//!   trusted — the ORT path strips it too, because exporters leave stale
//!   annotations there. An initializer that is ALSO a graph input is, from
//!   IR 4 (and at an unset IR, which the ORT path normalises to 9), only an
//!   overridable default: ORT types it from its declaration and never folds
//!   its values, so the pass seeds it only when that declaration pins exactly
//!   the payload's dims, and never tracks its integer values.
//! * A default-domain opset newer than the newest audited one
//!   (`MAX_AUDITED_OPSET`) refuses the model: a future op version could change
//!   a rule, and this pass runs precisely when ORT, which would know, cannot.
//! * No emitted shape contains a zero or negative dim. A rule that would yield
//!   one (an empty slice, say) yields "unknown" instead, because elsewhere in
//!   the loader `0` and `-1` are read as symbolic.
//! * The empty shape `[]` means PROVEN rank 0, the same meaning the ORT path's
//!   table has (there only after its rank probe, `resolve_zero_dim_ranks`: ORT
//!   reports zero dims for an untyped tensor too). Here every emitted shape,
//!   `[]` included, is an exact fact derived from exact inputs; a tensor
//!   whose rank is not established is absent, never `[]`. The Shape constant
//!   folders rely on this to fold `Shape(x)` of a recorded `[]` to the empty
//!   vector.
//! * Integer VALUES (needed for `Reshape`/`Slice`/`Squeeze`/`Unsqueeze`/
//!   `ReduceSum` parameters) are tracked only for small INT64/INT32 tensors,
//!   decoded bit-exactly from the protobuf — never through the f32 weight
//!   store, which cannot represent `INT64_MAX` sentinels.
//! * SSA violations (a name produced twice, or a node output shadowing an
//!   initializer / graph input) make that name unknown. Nodes are visited in
//!   file order, so a non-topological reference simply stays unknown.
//! * A computed terminal-output shape that disagrees with a concrete declared
//!   graph-output dim is dropped.
//! * Opset-dependent rules follow the default-domain opset: negative axes only
//!   from opset 11, legacy (`broadcast`-attribute) arithmetic is refused, and
//!   every `axes`/`starts`/`shape` input-vs-attribute switch is honoured.
//! * The whole pass is panic-contained: a bug here degrades to "no shapes",
//!   never to a crash (release builds unwind, see the workspace profile).
//!
//! The corpus test (`static_infer_corpus_tests.rs`) checks every shape the
//! fallback fills against the table ORT produced for the same bytes, across
//! every distinct model of eight VNN-COMP corpora; the unit tests pin each
//! rule, including broadcasting and negative axes.
//!
//! # Why not an in-process ORT fallback on spawn failure
//!
//! The measured failure was not a spawn failure in Rust's sense: the logged
//! message ("exited with exit status: 127", empty stderr) comes from the
//! post-spawn exit-status branch, so `spawn()` returned `Ok` and the child
//! then exited 127. (The likely mechanism — glibc's `posix_spawn` failing its
//! `exec` after qemu-user's `vfork` emulation kept the error from reaching the
//! parent — is not verified here.) A spawn-error hook would never have fired.
//! Widening it to "exit 127" guesses at an exit code, and running ORT
//! in-process in exactly the environments where the isolation boundary is
//! unavailable reintroduces the native-abort exposure the child exists to
//! contain (tract aborts on ACASXU; see `ShapeInferBackend`'s docs). This
//! fallback executes no native code and spawns nothing, so it adds no crash
//! surface and is used for every ORT-unavailable cause alike.

use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::onnx_proto::tensor_shape_proto::dimension::Value as Dim;
use prost::Message;
use std::collections::{HashMap, HashSet};
use tracing::warn;

use super::super::batch_resolve::{resolve_batch_dim, VERIFICATION_BATCH_SIZE};
use super::super::const_fold::is_standard_onnx_domain;
use super::DEFAULT_OPSET_VERSION;

// ONNX `TensorProto.DataType` codes consulted here.
const DT_FLOAT: i32 = 1;
const DT_UINT8: i32 = 2;
const DT_INT8: i32 = 3;
const DT_UINT16: i32 = 4;
const DT_INT16: i32 = 5;
const DT_INT32: i32 = 6;
const DT_INT64: i32 = 7;
const DT_BOOL: i32 = 9;
const DT_FLOAT16: i32 = 10;
const DT_DOUBLE: i32 = 11;
const DT_UINT32: i32 = 12;
const DT_UINT64: i32 = 13;
const DT_BFLOAT16: i32 = 16;

/// Integer values are tracked only for tensors at most this large: they exist
/// to carry shape/axis/index parameters, not data.
const MAX_TRACKED_INT_ELEMENTS: usize = 1024;

/// The newest default-domain opset whose schemas were audited against every
/// rule below (onnx 1.21's `onnx_opset_version()`: the full since-version
/// history of each modelled op through 26 changes attributes/types only in
/// ways these rules already honour). A newer opset may redefine an op, and
/// this fallback runs exactly when ORT — which would know — is unavailable,
/// so such a model is refused rather than extrapolated.
const MAX_AUDITED_OPSET: i64 = 26;

/// First ONNX IR version in which an initializer that is also a graph input
/// is only a DEFAULT the caller may override. From it on, ONNX Runtime types
/// such a tensor from the input DECLARATION and never reads its values.
const OVERRIDABLE_INITIALIZERS_SINCE_IR: i64 = 4;

/// Ops whose single output has exactly the shape of input 0, whatever their
/// attributes (activation parameters, `Softmax`'s axis, ...). Each takes one
/// input.
const UNARY_SHAPE_PRESERVING: &[&str] = &[
    "Relu",
    "Sigmoid",
    "Tanh",
    "LeakyRelu",
    "Elu",
    "Selu",
    "Celu",
    "Softplus",
    "Softsign",
    "HardSigmoid",
    "HardSwish",
    "Mish",
    "Erf",
    "Gelu",
    "ThresholdedRelu",
    "Abs",
    "Neg",
    "Ceil",
    "Floor",
    "Round",
    "Sign",
    "Sqrt",
    "Exp",
    "Log",
    "Reciprocal",
    "Not",
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Sinh",
    "Cosh",
    "Asinh",
    "Acosh",
    "Atanh",
    "IsNaN",
    "IsInf",
    "Softmax",
    "LogSoftmax",
    "Hardmax",
];

/// Two-input ops with multidirectional (numpy) broadcasting from opset 7.
const BROADCAST_BINARY: &[&str] = &[
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Equal",
    "Less",
    "Greater",
    "LessOrEqual",
    "GreaterOrEqual",
    "And",
    "Or",
    "Xor",
];

/// Variadic ops with multidirectional broadcasting from opset 8.
const BROADCAST_VARIADIC: &[&str] = &["Max", "Min", "Sum", "Mean"];

/// Result of the static pass, in the ORT table's format plus diagnostics.
#[derive(Debug, Default)]
pub(crate) struct StaticShapeTable {
    /// Tensor name -> exact shape, for runtime graph inputs and node outputs
    /// (never initializers — the ORT table does not carry them either).
    pub(crate) shapes: HashMap<String, Vec<i64>>,
    /// Runtime inputs and node outputs the pass left unknown.
    pub(crate) unresolved: Vec<String>,
    /// Why the whole model was refused, when it was (the table is then empty).
    pub(crate) refusal: Option<String>,
}

impl StaticShapeTable {
    fn refused(reason: String) -> Self {
        Self {
            refusal: Some(reason),
            ..Self::default()
        }
    }
}

/// Shapes to use once ONNX Runtime shape inference has failed: the static
/// pass's exact table, with its engagement written to the log
/// (`[shape-infer-fallback] ... filled=N unknown=M`) so a run can prove the
/// fallback actually engaged rather than infer it.
pub(crate) fn fallback_shapes_after_ort_failure(model_bytes: &[u8]) -> HashMap<String, Vec<i64>> {
    let table = infer_tensor_shapes_static(model_bytes);
    match &table.refusal {
        Some(reason) => warn!(
            "[shape-infer-fallback] ONNX Runtime unavailable and the static pass refused \
             the model ({reason}); loading with no inferred shapes"
        ),
        None => warn!(
            "[shape-infer-fallback] ONNX Runtime unavailable; static exact-rule shape \
             inference filled={} unknown={}",
            table.shapes.len(),
            table.unresolved.len()
        ),
    }
    table.shapes
}

/// Run the static pass over serialized ONNX model bytes (already
/// gz-decoded). Never panics: an internal fault yields a refused, empty table.
pub(crate) fn infer_tensor_shapes_static(model_bytes: &[u8]) -> StaticShapeTable {
    std::panic::catch_unwind(|| infer_unguarded(model_bytes)).unwrap_or_else(|_| {
        StaticShapeTable::refused(
            "static shape inference faulted internally; no shapes derived".to_string(),
        )
    })
}

fn infer_unguarded(model_bytes: &[u8]) -> StaticShapeTable {
    let model = match onnx_proto::ModelProto::decode(model_bytes) {
        Ok(model) => model,
        Err(err) => return StaticShapeTable::refused(format!("undecodable ONNX protobuf: {err}")),
    };
    let opset = match default_domain_opset(&model) {
        Ok(opset) => opset,
        Err(reason) => return StaticShapeTable::refused(reason),
    };
    // `normalize_model_for_ort` turns a non-positive IR version into 9, so
    // only an explicit 1..=3 keeps initializer-backed inputs constant.
    let initializers_overridable =
        model.ir_version <= 0 || model.ir_version >= OVERRIDABLE_INITIALIZERS_SINCE_IR;
    let Some(mut graph) = model.graph else {
        return StaticShapeTable::refused("model has no graph".to_string());
    };
    // The SAME pin the ORT path applies to the bytes it hands ORT, so a
    // symbolic batch dim resolves identically in both tables.
    resolve_batch_dim(&mut graph, VERIFICATION_BATCH_SIZE);
    infer_graph(&graph, opset, initializers_overridable)
}

/// The default-domain opset version, normalised exactly as the ORT path
/// normalises it (`normalize_model_for_ort`): a non-positive version, or no
/// entry at all, means [`DEFAULT_OPSET_VERSION`]. Conflicting entries, or an
/// explicit `ai.onnx` import (which the ORT normalisation would pair with a
/// second default-domain entry), are refused rather than guessed, and so is
/// an opset newer than [`MAX_AUDITED_OPSET`].
fn default_domain_opset(model: &onnx_proto::ModelProto) -> Result<i64, String> {
    let version = declared_default_domain_opset(model)?;
    if version > MAX_AUDITED_OPSET {
        return Err(format!(
            "default-domain opset {version} is newer than the newest audited opset \
             {MAX_AUDITED_OPSET}"
        ));
    }
    Ok(version)
}

fn declared_default_domain_opset(model: &onnx_proto::ModelProto) -> Result<i64, String> {
    let mut version: Option<i64> = None;
    for entry in &model.opset_import {
        if entry.domain == "ai.onnx" {
            return Err("explicit 'ai.onnx' opset import".to_string());
        }
        if !entry.domain.is_empty() {
            continue;
        }
        let this = if entry.version <= 0 {
            DEFAULT_OPSET_VERSION
        } else {
            entry.version
        };
        match version {
            None => version = Some(this),
            Some(previous) if previous == this => {}
            Some(previous) => {
                return Err(format!(
                    "conflicting default-domain opset imports {previous} and {this}"
                ))
            }
        }
    }
    Ok(version.unwrap_or(DEFAULT_OPSET_VERSION))
}

/// An exactly-known tensor: its shape (every dim >= 1; empty for a scalar)
/// and, for a small integer tensor whose elements are provably known, those
/// elements in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Known {
    dims: Vec<i64>,
    ints: Option<Vec<i64>>,
}

impl Known {
    /// A shape-only fact; `None` when any dim is not a positive extent.
    fn shape(dims: Vec<i64>) -> Option<Self> {
        dims.iter()
            .all(|&dim| dim >= 1)
            .then_some(Self { dims, ints: None })
    }

    /// A shape with integer values. Values that do not match the element
    /// count, or exceed the tracking cap, are dropped (the shape survives).
    fn with_ints(dims: Vec<i64>, ints: Option<Vec<i64>>) -> Option<Self> {
        let mut known = Self::shape(dims)?;
        if let Some(values) = ints {
            if values.len() <= MAX_TRACKED_INT_ELEMENTS && numel(&known.dims) == Some(values.len())
            {
                known.ints = Some(values);
            }
        }
        Some(known)
    }

    fn rank(&self) -> usize {
        self.dims.len()
    }
}

fn numel(dims: &[i64]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(usize::try_from(dim).ok()?)
    })
}

fn product(dims: &[i64]) -> Option<i64> {
    dims.iter().try_fold(1i64, |acc, &dim| acc.checked_mul(dim))
}

fn infer_graph(
    graph: &onnx_proto::GraphProto,
    opset: i64,
    initializers_overridable: bool,
) -> StaticShapeTable {
    let mut initializer_count: HashMap<&str, usize> = HashMap::new();
    for initializer in &graph.initializer {
        *initializer_count
            .entry(initializer.name.as_str())
            .or_default() += 1;
    }
    let sparse_names: HashSet<&str> = graph
        .sparse_initializer
        .iter()
        .filter_map(|sparse| sparse.values.as_ref().map(|values| values.name.as_str()))
        .collect();
    let mut input_count: HashMap<&str, usize> = HashMap::new();
    for input in &graph.input {
        *input_count.entry(input.name.as_str()).or_default() += 1;
    }
    let mut produced_count: HashMap<&str, usize> = HashMap::new();
    for node in &graph.node {
        for output in node.output.iter().filter(|name| !name.is_empty()) {
            *produced_count.entry(output.as_str()).or_default() += 1;
        }
    }
    let ambiguous = |name: &str| -> bool {
        let initializers = initializer_count.get(name).copied().unwrap_or(0);
        let inputs = input_count.get(name).copied().unwrap_or(0);
        let produced = produced_count.get(name).copied().unwrap_or(0);
        let sparse = sparse_names.contains(name);
        initializers > 1
            || inputs > 1
            || produced > 1
            || (produced > 0 && (initializers > 0 || inputs > 0 || sparse))
            || (sparse && initializers > 0)
    };

    // The ORT path returns an empty table for a model without runtime inputs
    // (`has_runtime_inputs`); mirror that rather than derive a table ORT
    // never would.
    let has_runtime_input = graph.input.iter().any(|input| {
        !input.name.is_empty() && !initializer_count.contains_key(input.name.as_str())
    });
    if !has_runtime_input {
        return StaticShapeTable::refused("model has no runtime graph inputs".to_string());
    }

    let mut known: HashMap<String, Known> = HashMap::new();
    let mut table = StaticShapeTable::default();

    for initializer in &graph.initializer {
        if initializer.name.is_empty() || ambiguous(&initializer.name) {
            continue;
        }
        let Some(mut fact) = known_from_tensor(initializer) else {
            continue;
        };
        let declarations: Vec<&onnx_proto::ValueInfoProto> = graph
            .input
            .iter()
            .filter(|input| input.name == initializer.name)
            .collect();
        // An initializer also listed as a graph input must not contradict its
        // declaration; the pass does not arbitrate which one ORT would use.
        if !declarations
            .iter()
            .all(|input| declaration_agrees(input, &fact.dims))
        {
            continue;
        }
        if initializers_overridable && !declarations.is_empty() {
            // From IR 4 such an initializer is only an overridable DEFAULT:
            // ORT types the tensor from its declaration (a symbolic or absent
            // declared shape stays unknown there) and never folds its values.
            // Mirror that: a shape only when the declaration pins every dim to
            // exactly the payload's dims, and never integer values.
            let pinned_exactly = declarations.iter().all(|input| {
                declared_runtime_input(input).is_some_and(|declared| declared.dims == fact.dims)
            });
            if !pinned_exactly {
                continue;
            }
            fact.ints = None;
        }
        known.insert(initializer.name.clone(), fact);
    }

    for input in &graph.input {
        if input.name.is_empty()
            || initializer_count.contains_key(input.name.as_str())
            || sparse_names.contains(input.name.as_str())
        {
            continue;
        }
        let fact = if ambiguous(&input.name) {
            None
        } else {
            declared_runtime_input(input)
        };
        match fact {
            Some(fact) => {
                table.shapes.insert(input.name.clone(), fact.dims.clone());
                known.insert(input.name.clone(), fact);
            }
            None => table.unresolved.push(input.name.clone()),
        }
    }

    for node in &graph.node {
        let derived = if is_standard_onnx_domain(&node.domain) {
            infer_node(node, opset, &known)
        } else {
            None
        };
        for (index, name) in node.output.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let fact = derived
                .as_ref()
                .and_then(|outputs| outputs.get(index).cloned().flatten())
                .filter(|_| !ambiguous(name));
            match fact {
                Some(fact) => {
                    table.shapes.insert(name.clone(), fact.dims.clone());
                    known.insert(name.clone(), fact);
                }
                None => table.unresolved.push(name.clone()),
            }
        }
    }

    // Terminal graph outputs keep their declared I/O contract in the ORT path
    // (only re-consumed outputs are stripped), so a computed shape that
    // contradicts a concrete declared dim is withdrawn rather than asserted.
    let consumed: HashSet<&str> = graph
        .node
        .iter()
        .flat_map(|node| node.input.iter())
        .map(String::as_str)
        .collect();
    for output in &graph.output {
        if consumed.contains(output.name.as_str()) || input_count.contains_key(output.name.as_str())
        {
            continue;
        }
        let agrees = table
            .shapes
            .get(&output.name)
            .map_or(true, |dims| declaration_agrees(output, dims));
        if !agrees {
            table.shapes.remove(&output.name);
            table.unresolved.push(output.name.clone());
        }
    }
    table
}

/// A declared shape is compatible with `dims` when it is absent, or has the
/// same rank and every concrete dim matches. Symbolic / unset dims are
/// compatible; a non-positive concrete dim never is.
fn declaration_agrees(info: &onnx_proto::ValueInfoProto, dims: &[i64]) -> bool {
    let Some(shape) = info
        .r#type
        .as_ref()
        .and_then(|ty| ty.tensor_type.as_ref())
        .and_then(|tensor| tensor.shape.as_ref())
    else {
        return true;
    };
    shape.dim.len() == dims.len()
        && shape
            .dim
            .iter()
            .zip(dims)
            .all(|(declared, &dim)| match &declared.value {
                Some(Dim::DimValue(value)) => *value == dim,
                Some(Dim::DimParam(_)) | None => true,
            })
}

/// A runtime graph input's declared shape (after the batch pin), when every
/// dim is a concrete positive value.
fn declared_runtime_input(info: &onnx_proto::ValueInfoProto) -> Option<Known> {
    let shape = info.r#type.as_ref()?.tensor_type.as_ref()?.shape.as_ref()?;
    let dims = shape
        .dim
        .iter()
        .map(|dim| match &dim.value {
            Some(Dim::DimValue(value)) if *value >= 1 => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<i64>>>()?;
    Known::shape(dims)
}

/// Shape (and small integer values) of an inline tensor, when its payload is
/// consistent with its declared dims.
fn known_from_tensor(tensor: &onnx_proto::TensorProto) -> Option<Known> {
    if tensor.segment.is_some() || tensor.data_location != 0 || !tensor.external_data.is_empty() {
        return None;
    }
    let dims = tensor.dims.clone();
    let count = numel(&dims)?;
    if dims.iter().any(|&dim| dim < 1) || !payload_matches(tensor, count) {
        return None;
    }
    let ints = if count <= MAX_TRACKED_INT_ELEMENTS {
        decode_ints(tensor, count)
    } else {
        None
    };
    Known::with_ints(dims, ints)
}

/// Whether the inline payload holds exactly `count` elements of the tensor's
/// data type, in exactly one storage field. Unknown / packed types fail.
fn payload_matches(tensor: &onnx_proto::TensorProto, count: usize) -> bool {
    let typed_fields = [
        !tensor.float_data.is_empty(),
        !tensor.int32_data.is_empty(),
        !tensor.int64_data.is_empty(),
        !tensor.double_data.is_empty(),
        !tensor.string_data.is_empty(),
        !tensor.uint64_data.is_empty(),
    ];
    let typed_in_use = typed_fields.iter().filter(|&&used| used).count();
    let raw = !tensor.raw_data.is_empty();
    if raw && typed_in_use > 0 {
        return false;
    }
    let (byte_width, typed_len, typed_field_in_use) = match tensor.data_type {
        DT_FLOAT => (4, tensor.float_data.len(), typed_fields[0]),
        DT_UINT8 | DT_INT8 | DT_BOOL => (1, tensor.int32_data.len(), typed_fields[1]),
        DT_UINT16 | DT_INT16 | DT_FLOAT16 | DT_BFLOAT16 => {
            (2, tensor.int32_data.len(), typed_fields[1])
        }
        DT_INT32 => (4, tensor.int32_data.len(), typed_fields[1]),
        DT_INT64 => (8, tensor.int64_data.len(), typed_fields[2]),
        DT_DOUBLE => (8, tensor.double_data.len(), typed_fields[3]),
        DT_UINT32 => (4, tensor.uint64_data.len(), typed_fields[5]),
        DT_UINT64 => (8, tensor.uint64_data.len(), typed_fields[5]),
        _ => return false,
    };
    if raw {
        return count
            .checked_mul(byte_width)
            .is_some_and(|bytes| bytes == tensor.raw_data.len());
    }
    // `count >= 1` always holds here (no zero dims), so an all-empty payload
    // is a missing payload, not an empty tensor.
    typed_in_use == 1 && typed_field_in_use && typed_len == count
}

/// Bit-exact integer elements of an INT64 / INT32 tensor whose payload has
/// already passed [`payload_matches`].
fn decode_ints(tensor: &onnx_proto::TensorProto, count: usize) -> Option<Vec<i64>> {
    match tensor.data_type {
        DT_INT64 if !tensor.raw_data.is_empty() => {
            let values: Vec<i64> = tensor
                .raw_data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(chunk);
                    i64::from_le_bytes(bytes)
                })
                .collect();
            (values.len() == count).then_some(values)
        }
        DT_INT64 => (tensor.int64_data.len() == count).then(|| tensor.int64_data.clone()),
        DT_INT32 if !tensor.raw_data.is_empty() => {
            let values: Vec<i64> = tensor
                .raw_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| {
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(chunk);
                    i64::from(i32::from_le_bytes(bytes))
                })
                .collect();
            (values.len() == count).then_some(values)
        }
        DT_INT32 => (tensor.int32_data.len() == count)
            .then(|| tensor.int32_data.iter().map(|&v| i64::from(v)).collect()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Attribute access. Every accessor is strict: a duplicated name, a
// function-body reference, or a type other than the one the schema defines is
// `Malformed`, which the rules turn into "unknown".
// ---------------------------------------------------------------------------

struct Malformed;

type AttrLookup<T> = Result<Option<T>, Malformed>;

/// An optional input that is present but whose values are not a known integer
/// constant (`Rule::const_ints`).
struct UnknownInts;

fn find_attr<'a>(
    node: &'a onnx_proto::NodeProto,
    name: &str,
) -> AttrLookup<&'a onnx_proto::AttributeProto> {
    let mut hits = node.attribute.iter().filter(|attr| attr.name == name);
    let first = hits.next();
    if hits.next().is_some() {
        return Err(Malformed);
    }
    match first {
        Some(attr) if !attr.ref_attr_name.is_empty() => Err(Malformed),
        other => Ok(other),
    }
}

fn int_attr(node: &onnx_proto::NodeProto, name: &str) -> AttrLookup<i64> {
    match find_attr(node, name)? {
        None => Ok(None),
        Some(attr) if attr.r#type == attribute_type::INT => Ok(Some(attr.i_value())),
        Some(_) => Err(Malformed),
    }
}

fn ints_attr(node: &onnx_proto::NodeProto, name: &str) -> AttrLookup<Vec<i64>> {
    match find_attr(node, name)? {
        None => Ok(None),
        Some(attr) if attr.r#type == attribute_type::INTS => Ok(Some(attr.ints.clone())),
        Some(_) => Err(Malformed),
    }
}

fn string_attr(node: &onnx_proto::NodeProto, name: &str) -> AttrLookup<String> {
    match find_attr(node, name)? {
        None => Ok(None),
        Some(attr) if attr.r#type == attribute_type::STRING => {
            String::from_utf8(attr.s_value().to_vec())
                .map(Some)
                .map_err(|_| Malformed)
        }
        Some(_) => Err(Malformed),
    }
}

/// Every attribute name is in `allowed` and none repeats.
fn attrs_within(node: &onnx_proto::NodeProto, allowed: &[&str]) -> bool {
    let mut seen = HashSet::new();
    node.attribute
        .iter()
        .all(|attr| allowed.contains(&attr.name.as_str()) && seen.insert(attr.name.as_str()))
}

/// Normalise an axis against `rank`; negative axes only from opset 11.
fn norm_axis(axis: i64, rank: usize, opset: i64) -> Option<usize> {
    let rank = i64::try_from(rank).ok()?;
    let axis = if axis < 0 {
        if opset < 11 {
            return None;
        }
        axis.checked_add(rank)?
    } else {
        axis
    };
    if (0..rank).contains(&axis) {
        usize::try_from(axis).ok()
    } else {
        None
    }
}

/// Normalise a list of axes; duplicates are malformed.
fn norm_axes(axes: &[i64], rank: usize, opset: i64) -> Option<Vec<usize>> {
    let mut seen = HashSet::new();
    axes.iter()
        .map(|&axis| norm_axis(axis, rank, opset).filter(|&norm| seen.insert(norm)))
        .collect()
}

/// Multidirectional (numpy) broadcast of several shapes.
fn broadcast(shapes: &[&[i64]]) -> Option<Vec<i64>> {
    let rank = shapes.iter().map(|shape| shape.len()).max()?;
    let mut out = vec![1i64; rank];
    for shape in shapes {
        let offset = rank - shape.len();
        for (index, &dim) in shape.iter().enumerate() {
            let slot = out.get_mut(offset + index)?;
            if *slot == 1 {
                *slot = dim;
            } else if dim != 1 && dim != *slot {
                return None;
            }
        }
    }
    Some(out)
}

/// Whether `from` broadcasts unidirectionally to `to`.
fn unidirectional(from: &[i64], to: &[i64]) -> bool {
    from.len() <= to.len()
        && from
            .iter()
            .rev()
            .zip(to.iter().rev())
            .all(|(&source, &target)| source == target || source == 1)
}

/// numpy `matmul` shape rule, including 1-D promotion and batch broadcast.
fn matmul_dims(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let (&a_last, a_rest) = a.split_last()?;
    let (&b_first, _) = b.split_first()?;
    let a_is_vector = a.len() == 1;
    let b_is_vector = b.len() == 1;
    let (a_batch, a_rows) = if a_is_vector {
        (&[][..], None)
    } else {
        let (&rows, batch) = a_rest.split_last()?;
        (batch, Some(rows))
    };
    let (b_batch, b_inner, b_cols) = if b_is_vector {
        (&[][..], b_first, None)
    } else {
        let (&cols, head) = b.split_last()?;
        let (&inner, batch) = head.split_last()?;
        (batch, inner, Some(cols))
    };
    if a_last != b_inner {
        return None;
    }
    let mut out = broadcast(&[a_batch, b_batch])?;
    out.extend(a_rows);
    out.extend(b_cols);
    Some(out)
}

// ---------------------------------------------------------------------------
// Per-node rules.
// ---------------------------------------------------------------------------

type NodeFacts = Option<Vec<Option<Known>>>;

struct Rule<'a> {
    node: &'a onnx_proto::NodeProto,
    opset: i64,
    known: &'a HashMap<String, Known>,
}

impl Rule<'_> {
    /// Wrap a single-output rule's fact; nodes with other output counts are
    /// not what the rule models.
    fn single(&self, fact: Option<Known>) -> NodeFacts {
        (self.node.output.len() == 1).then(|| vec![fact])
    }

    fn arity(&self) -> usize {
        self.node.input.len()
    }

    fn present(&self, index: usize) -> bool {
        self.node
            .input
            .get(index)
            .is_some_and(|name| !name.is_empty())
    }

    /// The known fact for input `index` (absent or unknown -> `None`).
    fn input(&self, index: usize) -> Option<&Known> {
        let name = self.node.input.get(index).filter(|name| !name.is_empty())?;
        self.known.get(name)
    }

    /// Integer values of optional input `index`: `Ok(None)` when the input is
    /// absent, `Ok(Some(v))` when its values are known, `Err(UnknownInts)` when
    /// it is present but not a known integer constant. The shape of
    /// `ints_attr`: callers write `.ok()?` for an optional input and
    /// `.ok()??` for a required one.
    fn const_ints(&self, index: usize) -> Result<Option<Vec<i64>>, UnknownInts> {
        if !self.present(index) {
            return Ok(None);
        }
        self.input(index)
            .and_then(|known| known.ints.clone())
            .map(Some)
            .ok_or(UnknownInts)
    }

    fn unary(&self) -> Option<Known> {
        if self.arity() != 1 {
            return None;
        }
        Known::shape(self.input(0)?.dims.clone())
    }

    fn clip(&self) -> Option<Known> {
        if !(1..=3).contains(&self.arity()) {
            return None;
        }
        Known::shape(self.input(0)?.dims.clone())
    }

    fn identity(&self) -> Option<Known> {
        if self.arity() != 1 || !self.node.attribute.is_empty() {
            return None;
        }
        self.input(0).cloned()
    }

    fn cast(&self) -> Option<Known> {
        if self.arity() != 1 || !attrs_within(self.node, &["to", "saturate", "round_mode"]) {
            return None;
        }
        let to = int_attr(self.node, "to").ok()??;
        let x = self.input(0)?;
        // Integer values survive only an integer-to-integer cast that
        // represents every one of them exactly.
        let ints = match &x.ints {
            Some(values) if to == i64::from(DT_INT64) => Some(values.clone()),
            Some(values)
                if to == i64::from(DT_INT32)
                    && values.iter().all(|&v| i32::try_from(v).is_ok()) =>
            {
                Some(values.clone())
            }
            _ => None,
        };
        Known::with_ints(x.dims.clone(), ints)
    }

    fn dropout(&self) -> NodeFacts {
        if !(1..=3).contains(&self.arity()) || !(1..=2).contains(&self.node.output.len()) {
            return None;
        }
        let x = Known::shape(self.input(0)?.dims.clone());
        Some(vec![x; self.node.output.len()])
    }

    fn batch_norm(&self) -> NodeFacts {
        if self.arity() != 5 || self.node.output.is_empty() {
            return None;
        }
        let mut facts = vec![None; self.node.output.len()];
        if let Some(first) = facts.first_mut() {
            *first = Known::shape(self.input(0)?.dims.clone());
        }
        Some(facts)
    }

    fn broadcast_ops(
        &self,
        min_opset: i64,
        inputs: std::ops::RangeInclusive<usize>,
    ) -> Option<Known> {
        if self.opset < min_opset
            || !inputs.contains(&self.arity())
            || !self.node.attribute.is_empty()
        {
            return None;
        }
        let facts = (0..self.arity())
            .map(|index| self.input(index).map(|fact| fact.dims.as_slice()))
            .collect::<Option<Vec<&[i64]>>>()?;
        Known::shape(broadcast(&facts)?)
    }

    fn prelu(&self) -> Option<Known> {
        if self.opset < 7 || self.arity() != 2 || !self.node.attribute.is_empty() {
            return None;
        }
        let x = self.input(0)?;
        let slope = self.input(1)?;
        if !unidirectional(&slope.dims, &x.dims) {
            return None;
        }
        Known::shape(x.dims.clone())
    }

    fn matmul(&self) -> Option<Known> {
        if self.arity() != 2 || !self.node.attribute.is_empty() {
            return None;
        }
        Known::shape(matmul_dims(&self.input(0)?.dims, &self.input(1)?.dims)?)
    }

    fn gemm(&self) -> Option<Known> {
        let allowed: &[&str] = if self.opset < 7 {
            &["alpha", "beta", "transA", "transB", "broadcast"]
        } else {
            &["alpha", "beta", "transA", "transB"]
        };
        if !(2..=3).contains(&self.arity()) || !attrs_within(self.node, allowed) {
            return None;
        }
        let a = self.input(0)?;
        let b = self.input(1)?;
        let (&[a0, a1], &[b0, b1]) = (a.dims.as_slice(), b.dims.as_slice()) else {
            return None;
        };
        let trans_a = int_attr(self.node, "transA").ok()?.unwrap_or(0) != 0;
        let trans_b = int_attr(self.node, "transB").ok()?.unwrap_or(0) != 0;
        let (m, k) = if trans_a { (a1, a0) } else { (a0, a1) };
        let (k_b, n) = if trans_b { (b1, b0) } else { (b0, b1) };
        if k != k_b {
            return None;
        }
        let out = vec![m, n];
        if self.present(2) {
            let c = self.input(2)?;
            let c_ok = if self.opset < 7 {
                c.dims == out
            } else {
                unidirectional(&c.dims, &out)
            };
            if !c_ok {
                return None;
            }
        } else if self.opset < 11 {
            // C was a required input before opset 11.
            return None;
        }
        Known::shape(out)
    }

    fn conv(&self) -> Option<Known> {
        if !(2..=3).contains(&self.arity())
            || !attrs_within(
                self.node,
                &[
                    "auto_pad",
                    "dilations",
                    "group",
                    "kernel_shape",
                    "pads",
                    "strides",
                ],
            )
        {
            return None;
        }
        let x = self.input(0)?;
        let w = self.input(1)?;
        if x.rank() < 3 || w.rank() != x.rank() {
            return None;
        }
        let spatial = x.rank() - 2;
        let (&batch, x_rest) = x.dims.split_first()?;
        let (&channels, x_spatial) = x_rest.split_first()?;
        let (&out_channels, w_rest) = w.dims.split_first()?;
        let (&w_channels, kernel) = w_rest.split_first()?;
        let group = int_attr(self.node, "group").ok()?.unwrap_or(1);
        if group < 1 || channels != w_channels.checked_mul(group)? || out_channels % group != 0 {
            return None;
        }
        if let Some(kernel_shape) = ints_attr(self.node, "kernel_shape").ok()? {
            if kernel_shape.as_slice() != kernel {
                return None;
            }
        }
        if self.present(2) {
            if self.input(2)?.dims != [out_channels] {
                return None;
            }
        }
        let dilations = ints_attr(self.node, "dilations")
            .ok()?
            .unwrap_or_else(|| vec![1; spatial]);
        let strides = ints_attr(self.node, "strides")
            .ok()?
            .unwrap_or_else(|| vec![1; spatial]);
        let pads_attr = ints_attr(self.node, "pads").ok()?;
        let auto_pad = string_attr(self.node, "auto_pad")
            .ok()?
            .unwrap_or_else(|| "NOTSET".to_string());
        if dilations.len() != spatial
            || strides.len() != spatial
            || dilations.iter().chain(&strides).any(|&v| v < 1)
        {
            return None;
        }
        let pads = match (auto_pad.as_str(), pads_attr) {
            ("NOTSET", Some(pads)) => pads,
            ("NOTSET", None) => vec![0; 2 * spatial],
            // Explicit pads cannot be combined with an auto_pad mode.
            ("VALID" | "SAME_UPPER" | "SAME_LOWER", None) => vec![0; 2 * spatial],
            _ => return None,
        };
        if pads.len() != 2 * spatial || pads.iter().any(|&p| p < 0) {
            return None;
        }
        let mut out = vec![batch, out_channels];
        for axis in 0..spatial {
            let size = *x_spatial.get(axis)?;
            let k = *kernel.get(axis)?;
            let dilation = *dilations.get(axis)?;
            let stride = *strides.get(axis)?;
            let extent = k.checked_sub(1)?.checked_mul(dilation)?.checked_add(1)?;
            let dim = match auto_pad.as_str() {
                "SAME_UPPER" | "SAME_LOWER" => size.checked_add(stride - 1)? / stride,
                _ => {
                    let padded = size
                        .checked_add(*pads.get(axis)?)?
                        .checked_add(*pads.get(axis + spatial)?)?;
                    let span = padded.checked_sub(extent)?;
                    if span < 0 {
                        return None;
                    }
                    span / stride + 1
                }
            };
            out.push(dim);
        }
        Known::shape(out)
    }

    fn flatten(&self) -> Option<Known> {
        if self.arity() != 1 || !attrs_within(self.node, &["axis"]) {
            return None;
        }
        let x = self.input(0)?;
        let rank = i64::try_from(x.rank()).ok()?;
        if rank == 0 {
            return None;
        }
        let mut axis = int_attr(self.node, "axis").ok()?.unwrap_or(1);
        if axis < 0 {
            if self.opset < 11 {
                return None;
            }
            axis = axis.checked_add(rank)?;
        }
        if !(0..=rank).contains(&axis) {
            return None;
        }
        let (outer, inner) = x.dims.split_at(usize::try_from(axis).ok()?);
        Known::shape(vec![product(outer)?, product(inner)?])
    }

    fn concat(&self) -> Option<Known> {
        if self.arity() == 0 || !attrs_within(self.node, &["axis"]) {
            return None;
        }
        let axis = match int_attr(self.node, "axis").ok()? {
            Some(axis) => axis,
            // Concat-1 defaulted the axis to 1; later versions require it.
            None if self.opset < 4 => 1,
            None => return None,
        };
        let facts = (0..self.arity())
            .map(|index| self.input(index))
            .collect::<Option<Vec<&Known>>>()?;
        let first = facts.first()?;
        let rank = first.rank();
        let axis = norm_axis(axis, rank, self.opset)?;
        let mut out = first.dims.clone();
        let mut total: i64 = 0;
        for fact in &facts {
            if fact.rank() != rank {
                return None;
            }
            for (index, (&dim, &reference)) in fact.dims.iter().zip(&first.dims).enumerate() {
                if index != axis && dim != reference {
                    return None;
                }
            }
            total = total.checked_add(*fact.dims.get(axis)?)?;
        }
        *out.get_mut(axis)? = total;
        let ints = if rank == 1 {
            facts
                .iter()
                .map(|fact| fact.ints.clone())
                .collect::<Option<Vec<Vec<i64>>>>()
                .map(|parts| parts.concat())
        } else {
            None
        };
        Known::with_ints(out, ints)
    }

    fn slice(&self) -> Option<Known> {
        let (starts, ends, axes, steps) = if self.opset < 10 {
            if self.arity() != 1 || !attrs_within(self.node, &["starts", "ends", "axes"]) {
                return None;
            }
            (
                ints_attr(self.node, "starts").ok()??,
                ints_attr(self.node, "ends").ok()??,
                ints_attr(self.node, "axes").ok()?,
                None,
            )
        } else {
            if !(3..=5).contains(&self.arity()) || !self.node.attribute.is_empty() {
                return None;
            }
            (
                self.const_ints(1).ok()??,
                self.const_ints(2).ok()??,
                self.const_ints(3).ok()?,
                self.const_ints(4).ok()?,
            )
        };
        let x = self.input(0)?;
        let rank = x.rank();
        if starts.len() != ends.len() || starts.is_empty() {
            return None;
        }
        let axes = match axes {
            Some(axes) if axes.len() == starts.len() => norm_axes(&axes, rank, self.opset)?,
            Some(_) => return None,
            None => {
                if starts.len() > rank {
                    return None;
                }
                (0..starts.len()).collect()
            }
        };
        let steps = match steps {
            Some(steps) if steps.len() == starts.len() && steps.iter().all(|&s| s != 0) => steps,
            Some(_) => return None,
            None => vec![1; starts.len()],
        };
        let mut out = x.dims.clone();
        // (axis, clamped start, step, length) — used for value slicing below.
        let mut plan = Vec::with_capacity(axes.len());
        for (((&axis, &start), &end), &step) in axes.iter().zip(&starts).zip(&ends).zip(&steps) {
            let dim = *x.dims.get(axis)?;
            let (first, length) = slice_extent(dim, start, end, step)?;
            *out.get_mut(axis)? = length;
            plan.push((axis, first, step, length));
        }
        let ints = match (&x.ints, plan.as_slice()) {
            (Some(values), [(0, first, step, length)]) if rank == 1 => (0..*length)
                .map(|offset| {
                    let index = first.checked_add(offset.checked_mul(*step)?)?;
                    values.get(usize::try_from(index).ok()?).copied()
                })
                .collect::<Option<Vec<i64>>>(),
            _ => None,
        };
        Known::with_ints(out, ints)
    }

    fn gather(&self) -> Option<Known> {
        if self.arity() != 2 || !attrs_within(self.node, &["axis"]) {
            return None;
        }
        let data = self.input(0)?;
        let indices = self.input(1)?;
        let axis = norm_axis(
            int_attr(self.node, "axis").ok()?.unwrap_or(0),
            data.rank(),
            self.opset,
        )?;
        let mut out = data.dims.get(..axis)?.to_vec();
        out.extend(&indices.dims);
        out.extend(data.dims.get(axis + 1..)?);
        let ints = match (&data.ints, &indices.ints) {
            (Some(values), Some(picks)) if data.rank() == 1 => {
                let length = i64::try_from(values.len()).ok()?;
                picks
                    .iter()
                    .map(|&pick| {
                        let index = if pick < 0 {
                            if self.opset < 11 {
                                return None;
                            }
                            pick.checked_add(length)?
                        } else {
                            pick
                        };
                        values.get(usize::try_from(index).ok()?).copied()
                    })
                    .collect::<Option<Vec<i64>>>()
            }
            _ => None,
        };
        Known::with_ints(out, ints)
    }

    fn squeeze(&self) -> Option<Known> {
        let axes = if self.opset < 13 {
            if self.arity() != 1 || !attrs_within(self.node, &["axes"]) {
                return None;
            }
            ints_attr(self.node, "axes").ok()?
        } else {
            if !(1..=2).contains(&self.arity()) || !self.node.attribute.is_empty() {
                return None;
            }
            self.const_ints(1).ok()?
        };
        let x = self.input(0)?;
        let out: Vec<i64> = match axes {
            // An explicitly empty list is ambiguous across versions: refuse.
            Some(axes) if axes.is_empty() => return None,
            Some(axes) => {
                let axes = norm_axes(&axes, x.rank(), self.opset)?;
                if axes.iter().any(|&axis| x.dims.get(axis) != Some(&1)) {
                    return None;
                }
                x.dims
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !axes.contains(index))
                    .map(|(_, &dim)| dim)
                    .collect()
            }
            None => x.dims.iter().copied().filter(|&dim| dim != 1).collect(),
        };
        Known::with_ints(out, x.ints.clone())
    }

    fn unsqueeze(&self) -> Option<Known> {
        let axes = if self.opset < 13 {
            if self.arity() != 1 || !attrs_within(self.node, &["axes"]) {
                return None;
            }
            ints_attr(self.node, "axes").ok()??
        } else {
            if self.arity() != 2 || !self.node.attribute.is_empty() {
                return None;
            }
            self.const_ints(1).ok()??
        };
        let x = self.input(0)?;
        if axes.is_empty() {
            return None;
        }
        let out_rank = x.rank().checked_add(axes.len())?;
        let axes = norm_axes(&axes, out_rank, self.opset)?;
        let mut source = x.dims.iter();
        let out = (0..out_rank)
            .map(|index| {
                if axes.contains(&index) {
                    Some(1)
                } else {
                    source.next().copied()
                }
            })
            .collect::<Option<Vec<i64>>>()?;
        Known::with_ints(out, x.ints.clone())
    }

    fn reshape(&self) -> Option<Known> {
        let (target, allow_zero) = if self.opset < 5 {
            if self.arity() != 1 || !attrs_within(self.node, &["shape"]) {
                return None;
            }
            (ints_attr(self.node, "shape").ok()??, false)
        } else {
            let allowed: &[&str] = if self.opset >= 14 {
                &["allowzero"]
            } else {
                &[]
            };
            if self.arity() != 2 || !attrs_within(self.node, allowed) {
                return None;
            }
            (
                self.const_ints(1).ok()??,
                int_attr(self.node, "allowzero").ok()?.unwrap_or(0) != 0,
            )
        };
        let x = self.input(0)?;
        let total = i64::try_from(numel(&x.dims)?).ok()?;
        let mut out = Vec::with_capacity(target.len());
        let mut inferred_at = None;
        for (index, &requested) in target.iter().enumerate() {
            let dim = match requested {
                -1 => {
                    if inferred_at.replace(index).is_some() {
                        return None;
                    }
                    1
                }
                0 if !allow_zero => *x.dims.get(index)?,
                dim if dim >= 1 => dim,
                // Negative sizes are malformed; an allowzero zero is an empty
                // tensor, which this pass never asserts.
                _ => return None,
            };
            out.push(dim);
        }
        let known_product = product(&out)?;
        match inferred_at {
            Some(index) => {
                if known_product <= 0 || total % known_product != 0 {
                    return None;
                }
                *out.get_mut(index)? = total / known_product;
            }
            None => {
                if known_product != total {
                    return None;
                }
            }
        }
        Known::with_ints(out, x.ints.clone())
    }

    fn transpose(&self) -> Option<Known> {
        if self.arity() != 1 || !attrs_within(self.node, &["perm"]) {
            return None;
        }
        let x = self.input(0)?;
        let perm = match ints_attr(self.node, "perm").ok()? {
            Some(perm) => perm,
            None => (0..i64::try_from(x.rank()).ok()?).rev().collect(),
        };
        if perm.len() != x.rank() {
            return None;
        }
        let mut seen = HashSet::new();
        let out = perm
            .iter()
            .map(|&axis| {
                let axis = usize::try_from(axis).ok().filter(|&a| seen.insert(a))?;
                x.dims.get(axis).copied()
            })
            .collect::<Option<Vec<i64>>>()?;
        Known::shape(out)
    }

    fn shape_op(&self) -> Option<Known> {
        let allowed: &[&str] = if self.opset >= 15 {
            &["start", "end"]
        } else {
            &[]
        };
        if self.arity() != 1 || !attrs_within(self.node, allowed) {
            return None;
        }
        let x = self.input(0)?;
        let rank = i64::try_from(x.rank()).ok()?;
        let clamp = |value: i64| -> i64 {
            let value = if value < 0 {
                value.saturating_add(rank)
            } else {
                value
            };
            value.clamp(0, rank)
        };
        let start = clamp(int_attr(self.node, "start").ok()?.unwrap_or(0));
        let end = clamp(int_attr(self.node, "end").ok()?.unwrap_or(rank));
        let values = x
            .dims
            .get(usize::try_from(start).ok()?..usize::try_from(end.max(start)).ok()?)?
            .to_vec();
        let length = i64::try_from(values.len()).ok()?;
        Known::with_ints(vec![length], Some(values))
    }

    fn constant(&self) -> Option<Known> {
        if self.arity() != 0 || self.node.attribute.len() != 1 {
            return None;
        }
        let attr = self.node.attribute.first()?;
        if !attr.ref_attr_name.is_empty() {
            return None;
        }
        match (attr.name.as_str(), attr.r#type) {
            ("value", attribute_type::TENSOR) => known_from_tensor(attr.t.as_ref()?),
            ("value_int", attribute_type::INT) => {
                Known::with_ints(Vec::new(), Some(vec![attr.i_value()]))
            }
            ("value_ints", attribute_type::INTS) => Known::with_ints(
                vec![i64::try_from(attr.ints.len()).ok()?],
                Some(attr.ints.clone()),
            ),
            ("value_float", attribute_type::FLOAT) => Known::shape(Vec::new()),
            ("value_floats", attribute_type::FLOATS) => {
                Known::shape(vec![i64::try_from(attr.floats.len()).ok()?])
            }
            _ => None,
        }
    }

    fn constant_of_shape(&self) -> Option<Known> {
        if self.arity() != 1 || !attrs_within(self.node, &["value"]) {
            return None;
        }
        let shape = self.input(0)?;
        if shape.rank() != 1 {
            return None;
        }
        Known::shape(shape.ints.clone()?)
    }

    fn expand(&self) -> Option<Known> {
        if self.arity() != 2 || !self.node.attribute.is_empty() {
            return None;
        }
        let x = self.input(0)?;
        let shape = self.input(1)?;
        if shape.rank() != 1 {
            return None;
        }
        let requested = shape.ints.clone()?;
        if requested.iter().any(|&dim| dim < 1) {
            return None;
        }
        Known::shape(broadcast(&[x.dims.as_slice(), requested.as_slice()])?)
    }

    fn reduce(&self, axes_input_since: i64) -> Option<Known> {
        let x = self.input(0)?;
        let (axes, noop_with_empty_axes) = if self.opset >= axes_input_since {
            if !(1..=2).contains(&self.arity())
                || !attrs_within(self.node, &["keepdims", "noop_with_empty_axes"])
            {
                return None;
            }
            (
                self.const_ints(1).ok()?,
                int_attr(self.node, "noop_with_empty_axes")
                    .ok()?
                    .unwrap_or(0)
                    != 0,
            )
        } else {
            if self.arity() != 1 || !attrs_within(self.node, &["keepdims", "axes"]) {
                return None;
            }
            let axes = ints_attr(self.node, "axes").ok()?;
            if axes.as_ref().is_some_and(Vec::is_empty) {
                return None;
            }
            (axes, false)
        };
        let keep_dims = int_attr(self.node, "keepdims").ok()?.unwrap_or(1) != 0;
        if x.rank() == 0 {
            return None;
        }
        let axes: Vec<usize> = match axes {
            Some(axes) => norm_axes(&axes, x.rank(), self.opset)?,
            None if noop_with_empty_axes => return Known::shape(x.dims.clone()),
            None => (0..x.rank()).collect(),
        };
        let out = x
            .dims
            .iter()
            .enumerate()
            .filter_map(|(index, &dim)| match (axes.contains(&index), keep_dims) {
                (false, _) => Some(dim),
                (true, true) => Some(1),
                (true, false) => None,
            })
            .collect();
        Known::shape(out)
    }
}

/// Clamp a Slice `(start, end, step)` against `dim` exactly as the ONNX
/// specification does (`INT64_MAX` / `INT64_MIN` sentinels included) and
/// return `(first index, length)`. `None` for a zero step or an empty result.
fn slice_extent(dim: i64, start: i64, end: i64, step: i64) -> Option<(i64, i64)> {
    if step == 0 || dim < 1 {
        return None;
    }
    let dim = i128::from(dim);
    let step = i128::from(step);
    let mut start = i128::from(start);
    let mut end = i128::from(end);
    if start < 0 {
        start += dim;
    }
    if end < 0 {
        end += dim;
    }
    let length = if step > 0 {
        start = start.clamp(0, dim);
        end = end.clamp(0, dim);
        if end > start {
            (end - start + step - 1) / step
        } else {
            0
        }
    } else {
        start = start.clamp(0, dim - 1);
        end = end.clamp(-1, dim - 1);
        if start > end {
            (start - end + (-step) - 1) / (-step)
        } else {
            0
        }
    };
    if length < 1 {
        return None;
    }
    Some((i64::try_from(start).ok()?, i64::try_from(length).ok()?))
}

/// The input-vs-attribute switch opset for `axes` on each Reduce op.
fn reduce_axes_input_since(op: &str) -> Option<i64> {
    match op {
        "ReduceSum" => Some(13),
        "ReduceMean" | "ReduceMax" | "ReduceMin" | "ReduceProd" | "ReduceL1" | "ReduceL2"
        | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceSumSquare" => Some(18),
        _ => None,
    }
}

fn infer_node(
    node: &onnx_proto::NodeProto,
    opset: i64,
    known: &HashMap<String, Known>,
) -> NodeFacts {
    let rule = Rule { node, opset, known };
    let op = node.op_type.as_str();
    if UNARY_SHAPE_PRESERVING.contains(&op) {
        return rule.single(rule.unary());
    }
    if BROADCAST_BINARY.contains(&op) {
        return rule.single(rule.broadcast_ops(7, 2..=2));
    }
    if BROADCAST_VARIADIC.contains(&op) {
        return rule.single(rule.broadcast_ops(8, 1..=usize::MAX));
    }
    if let Some(since) = reduce_axes_input_since(op) {
        return rule.single(rule.reduce(since));
    }
    match op {
        "Clip" => rule.single(rule.clip()),
        "Identity" => rule.single(rule.identity()),
        "Cast" => rule.single(rule.cast()),
        "Dropout" => rule.dropout(),
        "BatchNormalization" => rule.batch_norm(),
        "Where" => rule.single(rule.broadcast_ops(9, 3..=3)),
        "PRelu" => rule.single(rule.prelu()),
        "MatMul" => rule.single(rule.matmul()),
        "Gemm" => rule.single(rule.gemm()),
        "Conv" => rule.single(rule.conv()),
        "Flatten" => rule.single(rule.flatten()),
        "Concat" => rule.single(rule.concat()),
        "Slice" => rule.single(rule.slice()),
        "Gather" => rule.single(rule.gather()),
        "Squeeze" => rule.single(rule.squeeze()),
        "Unsqueeze" => rule.single(rule.unsqueeze()),
        "Reshape" => rule.single(rule.reshape()),
        "Transpose" => rule.single(rule.transpose()),
        "Shape" => rule.single(rule.shape_op()),
        "Constant" => rule.single(rule.constant()),
        "ConstantOfShape" => rule.single(rule.constant_of_shape()),
        "Expand" => rule.single(rule.expand()),
        _ => None,
    }
}
