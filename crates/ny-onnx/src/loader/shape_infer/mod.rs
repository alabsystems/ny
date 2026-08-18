// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "ort")]
use crate::onnx_proto;
#[cfg(feature = "ort")]
use ny_core::NyError;
use ny_core::Result;
#[cfg(feature = "ort")]
use ny_propagate::layers::normalize_transpose_perm_for_rank;
#[cfg(feature = "ort")]
use ort::session::Session;
#[cfg(feature = "ort")]
use prost::Message;
use std::collections::HashMap;
#[cfg(feature = "ort")]
use std::collections::HashSet;
#[cfg(feature = "ort")]
use std::io::Write;
use std::path::Path;
#[cfg(feature = "ort")]
use std::sync::mpsc;
#[cfg(feature = "ort")]
use std::time::Duration;
#[cfg(feature = "ort")]
use tempfile::TempPath;
#[cfg(feature = "ort")]
use tracing::debug;
use tracing::warn;

#[cfg(feature = "ort")]
use super::const_fold::is_standard_onnx_domain;

/// Maximum time to wait for ONNX Runtime shape inference session creation.
/// Models with problematic shape inference (e.g., Gemm with non-rank-2 inputs)
/// can cause ORT to hang indefinitely in its C++ optimization passes.
#[cfg(feature = "ort")]
const ORT_SHAPE_INFERENCE_TIMEOUT: Duration = Duration::from_secs(30);

#[cfg(feature = "ort")]
const DEFAULT_IR_VERSION: i64 = 9;
#[cfg(any(feature = "ort", test))]
pub(super) const DEFAULT_OPSET_VERSION: i64 = 13;

#[cfg(feature = "ort")]
fn normalize_model_for_ort(model: &mut onnx_proto::ModelProto) {
    if model.ir_version <= 0 {
        model.ir_version = DEFAULT_IR_VERSION;
    }
    let mut has_default_domain = false;
    for opset in &mut model.opset_import {
        if opset.domain.is_empty() {
            has_default_domain = true;
            if opset.version <= 0 {
                opset.version = DEFAULT_OPSET_VERSION;
            }
        }
    }
    if !has_default_domain {
        model.opset_import.push(onnx_proto::OperatorSetIdProto {
            version: DEFAULT_OPSET_VERSION,
            domain: String::new(),
        });
    }
}

/// Create an ORT session from in-memory bytes with a timeout guard.
///
/// The ONNX Runtime `commit_from_memory` call is a blocking FFI call into C++
/// that can hang indefinitely on models with problematic shape inference (e.g.,
/// Gemm nodes with non-rank-2 inputs). This spawns the call on a background
/// thread and enforces a deadline.
#[cfg(feature = "ort")]
fn create_ort_session_from_memory_with_timeout(bytes: Vec<u8>) -> Result<Session> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = Session::builder()
            .map_err(|e| NyError::ModelLoad(format!("ONNX Runtime init failed: {e}")))
            .and_then(|mut builder| {
                builder.commit_from_memory(&bytes).map_err(|e| {
                    NyError::ModelLoad(format!("ONNX Runtime shape inference failed: {e}"))
                })
            });
        let _ = tx.send(result);
    });
    match rx.recv_timeout(ORT_SHAPE_INFERENCE_TIMEOUT) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            warn!(
                "ONNX Runtime shape inference timed out after {}s — skipping",
                ORT_SHAPE_INFERENCE_TIMEOUT.as_secs()
            );
            Err(NyError::ModelLoad(format!(
                "ONNX Runtime shape inference timed out after {}s",
                ORT_SHAPE_INFERENCE_TIMEOUT.as_secs()
            )))
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            // The workspace's unwind profiles, including release, turn a Rust
            // panic on the worker into sender disconnection. Native aborts and
            // faults still require the subprocess backend for containment.
            Err(NyError::ModelLoad(
                "ONNX Runtime shape inference thread panicked".to_string(),
            ))
        }
    }
}

/// Create an ORT session from a file path with a timeout guard.
#[cfg(feature = "ort")]
fn create_ort_session_from_file_with_timeout(path: std::path::PathBuf) -> Result<Session> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = Session::builder()
            .map_err(|e| NyError::ModelLoad(format!("ONNX Runtime init failed: {e}")))
            .and_then(|mut builder| {
                builder.commit_from_file(&path).map_err(|e| {
                    NyError::ModelLoad(format!("ONNX Runtime shape inference failed: {e}"))
                })
            });
        let _ = tx.send(result);
    });
    match rx.recv_timeout(ORT_SHAPE_INFERENCE_TIMEOUT) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            warn!(
                "ONNX Runtime shape inference timed out after {}s — skipping",
                ORT_SHAPE_INFERENCE_TIMEOUT.as_secs()
            );
            Err(NyError::ModelLoad(format!(
                "ONNX Runtime shape inference timed out after {}s",
                ORT_SHAPE_INFERENCE_TIMEOUT.as_secs()
            )))
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            // The workspace's unwind profiles, including release, turn a Rust
            // panic on the worker into sender disconnection. Native aborts and
            // faults still require the subprocess backend for containment.
            Err(NyError::ModelLoad(
                "ONNX Runtime shape inference thread panicked".to_string(),
            ))
        }
    }
}

#[cfg(feature = "ort")]
pub(super) fn infer_tensor_shapes_from_ort(
    model_bytes: &[u8],
) -> Result<HashMap<String, Vec<i64>>> {
    let exposed = expose_intermediate_outputs(model_bytes)?;
    if !exposed.has_runtime_inputs {
        return Ok(HashMap::new());
    }
    if exposed.bytes.is_empty() {
        return Err(NyError::ModelLoad(
            "ONNX model bytes empty after exposing intermediate outputs".to_string(),
        ));
    }
    let normalized_bytes = normalize_bytes_for_ort(&exposed.bytes)?;
    let session = create_ort_session_from_memory_with_timeout(normalized_bytes)?;
    Ok(collect_shapes_from_session(&session))
}

#[cfg(not(feature = "ort"))]
pub(super) fn infer_tensor_shapes_from_ort(
    _model_bytes: &[u8],
) -> Result<HashMap<String, Vec<i64>>> {
    warn!("ONNX Runtime shape inference disabled; rebuilding without inferred tensor shapes");
    Ok(HashMap::new())
}

#[cfg(feature = "ort")]
pub(super) fn infer_tensor_shapes_from_ort_path(
    path: &Path,
    model_bytes: &[u8],
) -> Result<HashMap<String, Vec<i64>>> {
    let exposed = expose_intermediate_outputs(model_bytes)?;
    if !exposed.has_runtime_inputs {
        return Ok(HashMap::new());
    }
    if exposed.bytes.is_empty() {
        return Err(NyError::ModelLoad(
            "ONNX model bytes empty after exposing intermediate outputs".to_string(),
        ));
    }
    let normalized_bytes = normalize_bytes_for_ort(&exposed.bytes)?;
    if is_gz_path(path) {
        let session = create_ort_session_from_memory_with_timeout(normalized_bytes)?;
        return Ok(collect_shapes_from_session(&session));
    }

    let temp_path = write_temp_onnx_file(path, &normalized_bytes)?;
    let file_path = temp_path_ref(&temp_path).to_path_buf();
    let session = create_ort_session_from_file_with_timeout(file_path)?;

    Ok(collect_shapes_from_session(&session))
}

#[cfg(not(feature = "ort"))]
pub(super) fn infer_tensor_shapes_from_ort_path(
    _path: &Path,
    _model_bytes: &[u8],
) -> Result<HashMap<String, Vec<i64>>> {
    warn!("ONNX Runtime shape inference disabled; rebuilding without inferred tensor shapes");
    Ok(HashMap::new())
}

#[cfg(feature = "ort")]
fn write_temp_onnx_file(path: &Path, bytes: &[u8]) -> Result<TempPath> {
    if bytes.is_empty() {
        return Err(NyError::ModelLoad(
            "Refusing to write empty ONNX model bytes".to_string(),
        ));
    }
    // Drive-relative Windows paths (e.g. "C:foo.onnx") yield a non-directory parent.
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let is_drive_relative = is_drive_relative_windows_prefix(parent);
    let mut temp = if parent.is_dir() && !is_drive_relative {
        match temp_builder().tempfile_in(parent) {
            Ok(tempfile) => Ok(tempfile),
            Err(err) => {
                debug!(
                    "Temp parent {} unavailable or unwritable ({err}); using system temp dir",
                    parent.display()
                );
                temp_builder().tempfile()
            }
        }
    } else {
        debug!(
            "Temp parent {} unavailable; using system temp dir",
            parent.display()
        );
        temp_builder().tempfile()
    }
    .map_err(|e| NyError::ModelLoad(format!("Failed to create temp ONNX file: {e}")))?;
    temp.write_all(bytes)
        .map_err(|e| NyError::ModelLoad(format!("Failed to write temp ONNX file: {e}")))?;
    temp.flush()
        .map_err(|e| NyError::ModelLoad(format!("Failed to flush temp ONNX file: {e}")))?;
    temp.as_file()
        .sync_data()
        .map_err(|e| NyError::ModelLoad(format!("Failed to sync temp ONNX file: {e}")))?;
    Ok(temp.into_temp_path())
}

#[cfg(feature = "ort")]
fn is_drive_relative_windows_prefix(path: &Path) -> bool {
    let text = path.to_string_lossy();
    if text.len() != 2 {
        return false;
    }
    let mut chars = text.chars();
    matches!(chars.next(), Some(letter) if letter.is_ascii_alphabetic())
        && matches!(chars.next(), Some(':'))
}

#[cfg(feature = "ort")]
fn temp_path_ref(path: &TempPath) -> &Path {
    <TempPath as AsRef<Path>>::as_ref(path)
}

#[cfg(feature = "ort")]
fn normalize_bytes_for_ort(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut model = onnx_proto::ModelProto::decode(bytes)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse ONNX protobuf: {e}")))?;
    normalize_model_for_ort(&mut model);
    let mut buf = Vec::new();
    model.encode(&mut buf).map_err(|e| {
        NyError::ModelLoad(format!("Failed to serialize ONNX model for inference: {e}"))
    })?;
    Ok(buf)
}

#[cfg(feature = "ort")]
fn is_gz_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
}

#[cfg(feature = "ort")]
fn temp_builder() -> tempfile::Builder<'static, 'static> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("ny-onnx-shape-").suffix(".onnx");
    builder
}

#[cfg(feature = "ort")]
fn collect_shapes_from_session(session: &Session) -> HashMap<String, Vec<i64>> {
    let mut shapes = HashMap::new();

    for input in session.inputs() {
        if let Some(shape) = input.dtype().tensor_shape() {
            insert_shape_if_informative(&mut shapes, input.name(), shape);
        }
    }

    for output in session.outputs() {
        if let Some(shape) = output.dtype().tensor_shape() {
            insert_shape_if_informative(&mut shapes, output.name(), shape);
        }
    }

    shapes
}

#[cfg(feature = "ort")]
fn insert_shape_if_informative(shapes: &mut HashMap<String, Vec<i64>>, name: &str, shape: &[i64]) {
    if name.is_empty() {
        return;
    }
    if shape.is_empty() || shape.iter().any(|dim| *dim > 0) {
        shapes.insert(name.to_string(), shape.to_vec());
    }
}

#[cfg(feature = "ort")]
fn type_from_initializer(initializer: &onnx_proto::TensorProto) -> Option<onnx_proto::TypeProto> {
    if initializer.data_type <= 0 {
        return None;
    }
    let dims = initializer
        .dims
        .iter()
        .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *dim,
            )),
        })
        .collect();
    Some(onnx_proto::TypeProto {
        tensor_type: Some(onnx_proto::TensorTypeProto {
            elem_type: initializer.data_type,
            shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
        }),
    })
}

#[cfg(feature = "ort")]
struct ExposedModel {
    bytes: Vec<u8>,
    has_runtime_inputs: bool,
}

#[cfg(feature = "ort")]
/// Clear the `shape` of every intermediate `value_info` tensor (element types
/// preserved). Prevents stale/rank-incorrect annotations from corrupting ORT
/// shape inference via its lenient rank-mismatch merge. Graph `input`/`output`
/// value_info form the I/O contract and are intentionally left untouched.
fn clear_intermediate_value_info_shapes(graph: &mut onnx_proto::GraphProto) {
    for info in graph.value_info.iter_mut() {
        if let Some(ty) = info.r#type.as_mut() {
            if let Some(tensor) = ty.tensor_type.as_mut() {
                tensor.shape = None;
            }
        }
    }
}

#[cfg(feature = "ort")]
/// Clear the `shape` of any `graph.output` entry that is actually an
/// intermediate tensor (i.e. consumed as input by a downstream node), while
/// leaving the shape of terminal model outputs (the true I/O contract) intact.
///
/// Some exporters expose an intermediate activation as a graph output AND
/// annotate it with a stale/rank-incorrect shape (e.g. tinyimagenet's tensor
/// '120', an intermediate 4-D Conv output {-1,64,27,27} mis-declared rank-1
/// {64}, likely confused with a bias/scale param of length 64). Because that
/// shape lives in `graph.output` — not `value_info` —
/// [`clear_intermediate_value_info_shapes`] cannot reach it, so the bogus rank
/// reaches ORT, survives its "lenient merge", and aborts the WHOLE shape
/// inference pass at the next Conv ("Input tensor must have at least 2
/// dimensions"). Clearing only re-consumed outputs lets ORT re-derive the
/// correct shape with no contradictory hint; terminal outputs are untouched so
/// the verified model's I/O dimensions never change.
fn clear_reconsumed_output_shapes(graph: &mut onnx_proto::GraphProto) {
    // Collect into an owned set first to avoid borrowing `graph.node` while
    // mutating `graph.output`.
    let consumed: HashSet<String> = graph
        .node
        .iter()
        .flat_map(|node| node.input.iter())
        .filter(|name| !name.is_empty())
        .cloned()
        .collect();
    for info in graph.output.iter_mut() {
        if consumed.contains(&info.name) {
            if let Some(ty) = info.r#type.as_mut() {
                if let Some(tensor) = ty.tensor_type.as_mut() {
                    tensor.shape = None;
                }
            }
        }
    }
}

#[cfg(feature = "ort")]
/// Map every tensor name with a *definitively* known rank to that rank. Only
/// sources ONNX Runtime treats as ground truth are used: `initializer` dims
/// (the data's real shape) and `graph.input` declarations (the I/O contract,
/// which `clear_*` never strips). Intermediate `value_info`/`output` shapes are
/// intentionally excluded — they may have been cleared above, and a stale
/// annotation there is exactly what we must not trust.
fn known_tensor_ranks(graph: &onnx_proto::GraphProto) -> HashMap<String, usize> {
    let mut ranks = HashMap::new();
    for init in &graph.initializer {
        if !init.name.is_empty() {
            ranks.insert(init.name.clone(), init.dims.len());
        }
    }
    for input in &graph.input {
        if input.name.is_empty() {
            continue;
        }
        if let Some(shape) = input
            .r#type
            .as_ref()
            .and_then(|t| t.tensor_type.as_ref())
            .and_then(|tt| tt.shape.as_ref())
        {
            ranks.insert(input.name.clone(), shape.dim.len());
        }
    }
    ranks
}

#[cfg(feature = "ort")]
/// Rewrite `Transpose` `perm` attributes to match the actual rank of the tensor
/// the node is applied to, when that rank is definitively known (see
/// [`known_tensor_ranks`]). Mismatched perms make ORT shape inference abort the
/// whole pass; a rank-consistent perm lets ORT proceed and infer correct shapes
/// for the rest of the graph.
///
/// Soundness: the rewrite is delegated to
/// [`normalize_transpose_perm_for_rank`], which only ever emits a perm that
/// computes the *same logical transpose* as the original (identity for rank-≤1
/// tensors, the surviving-axis permutation when leading dims were dropped) and
/// returns `None` — leaving the node untouched — when no equivalence is provable.
/// In all cases this affects only the bytes used for ORT shape *extraction*, not
/// the graph that is converted to a network.
fn normalize_transpose_perms_for_ort(graph: &mut onnx_proto::GraphProto) {
    let ranks = known_tensor_ranks(graph);
    for node in graph.node.iter_mut() {
        if node.op_type != "Transpose"
            || !is_standard_onnx_domain(&node.domain)
            || node.input.is_empty()
        {
            continue;
        }
        let Some(&rank) = node.input.first().and_then(|name| ranks.get(name)) else {
            continue;
        };
        let Some(perm_attr) = node.attribute.iter_mut().find(|a| a.name == "perm") else {
            continue;
        };
        // Parse the existing perm; skip if any entry is negative (invalid axis).
        let Some(raw_perm) = perm_attr
            .ints
            .iter()
            .map(|&v| usize::try_from(v).ok())
            .collect::<Option<Vec<usize>>>()
        else {
            continue;
        };
        if raw_perm.len() == rank {
            continue; // already rank-consistent
        }
        if let Some(fixed) = normalize_transpose_perm_for_rank(&raw_perm, rank) {
            debug!(
                "Rewriting Transpose '{}' perm {:?} -> {:?} for ORT (input rank {})",
                node.name, raw_perm, fixed, rank
            );
            perm_attr.ints = fixed.into_iter().map(|v| v as i64).collect();
        }
    }
}

#[cfg(feature = "ort")]
fn expose_intermediate_outputs(model_bytes: &[u8]) -> Result<ExposedModel> {
    let mut model = onnx_proto::ModelProto::decode(model_bytes)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse ONNX protobuf: {e}")))?;
    normalize_model_for_ort(&mut model);

    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| NyError::ModelLoad("ONNX model has no graph".to_string()))?;

    // Pin a symbolic/dynamic leading (batch) dimension on runtime graph inputs
    // to the verification batch size. With a free leading symbol, ORT cannot
    // resolve ranks through transformer attention `Shape → Gather(axis 0) →
    // Concat → Reshape → Transpose` chains and aborts the ENTIRE pass (e.g. ViT's
    // `[TypeInferenceError] Invalid attribute perm {0, 2, 1}, input shape =
    // {48}`), dropping every downstream tensor to conservative bounds. A static
    // leading dim lets ORT infer the rank-correct attention shapes directly. This
    // matches the same rewrite applied to the conversion graph in
    // `parse::parse_onnx_data`, so both paths agree on the dim. Soundness: this
    // only edits the bytes used for ORT shape *extraction* (the graph converted
    // to a network is re-decoded from the original bytes), and even there it
    // changes only shape metadata, never the network function.
    if crate::loader::batch_resolve::resolve_batch_dim(
        graph,
        crate::loader::batch_resolve::VERIFICATION_BATCH_SIZE,
    ) {
        debug!(
            "Resolved symbolic batch dimension to {} for ORT shape inference",
            crate::loader::batch_resolve::VERIFICATION_BATCH_SIZE
        );
    }

    // Strip shapes from intermediate value_info before handing to ORT. Some
    // exporters carry stale/rank-incorrect annotations (e.g. tinyimagenet's
    // tensor '120' declared rank-1 {64} when it is really a 4-D Conv output
    // {-1,64,27,27}). ORT's "lenient merge" of the mismatch keeps the bogus
    // rank-1 shape, which then makes a downstream Conv abort the ENTIRE shape
    // inference pass ("Input tensor must have at least 2 dimensions"). Clearing
    // only intermediate shapes (element types preserved; graph I/O untouched)
    // lets ORT re-derive correct shapes with no contradictory hint. (#4xxx)
    clear_intermediate_value_info_shapes(graph);
    // Same hazard, but for intermediates exposed in `graph.output` (which the
    // value_info clear above intentionally skips). Clears the shape only for
    // outputs that are re-consumed downstream — never terminal model outputs.
    clear_reconsumed_output_shapes(graph);
    // Rewrite any `Transpose` whose `perm` does not match the rank of the tensor
    // it is applied to (e.g. a vit positional-embedding `{48}` rank-1 initializer
    // fed to a `perm={0,2,1}` Transpose). ONNX Runtime's shape inference rejects
    // such a node outright ("[TypeInferenceError] Invalid attribute perm ...,
    // input shape = {...}") and ABORTS the entire pass, so every downstream
    // tensor — even healthy ones — loses its inferred shape and the model can
    // only fall back to conservative (unknown) bounds. This only edits the bytes
    // handed to ORT for *shape extraction*; the graph that is later converted to
    // layers is re-decoded from the original bytes, so the network's
    // mathematical function is untouched.
    normalize_transpose_perms_for_ort(graph);

    let initializer_names: HashSet<String> =
        graph.initializer.iter().map(|i| i.name.clone()).collect();
    let has_runtime_inputs = graph
        .input
        .iter()
        .any(|input| !input.name.is_empty() && !initializer_names.contains(&input.name));
    if !has_runtime_inputs {
        return Ok(ExposedModel {
            bytes: Vec::new(),
            has_runtime_inputs,
        });
    }

    let existing_outputs: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();
    let input_names: HashSet<String> = graph.input.iter().map(|i| i.name.clone()).collect();
    let mut type_map: HashMap<String, onnx_proto::TypeProto> = HashMap::new();

    for info in graph
        .input
        .iter()
        .chain(graph.output.iter())
        .chain(graph.value_info().iter())
    {
        if let Some(info_type) = info.r#type.as_ref() {
            type_map.insert(info.name.clone(), info_type.clone());
        }
    }

    for initializer in &graph.initializer {
        if let Some(ty) = type_from_initializer(initializer) {
            type_map.insert(initializer.name.clone(), ty);
        }
    }

    // Ops that preserve input shape (element-wise). Safe to propagate type from input.
    let shape_preserving_ops: HashSet<&str> = [
        "Relu",
        "Sigmoid",
        "Tanh",
        "LeakyRelu",
        "Elu",
        "Selu",
        "Softplus",
        "Softsign",
        "HardSigmoid",
        "HardSwish",
        "Mish",
        "Erf",
        "Gelu",
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
        "Clip",
        "Dropout",
        "BatchNormalization",
        "InstanceNormalization",
        "GroupNormalization",
        "Identity",
    ]
    .into_iter()
    .collect();

    let mut new_outputs = Vec::new();
    let mut new_output_names = HashSet::new();
    for node in &graph.node {
        // Only propagate input type to output for shape-preserving ops.
        // For shape-changing ops (MatMul, Gemm, Conv, Reshape, etc.), propagating
        // the input type would give the output WRONG dimensions, which then causes
        // ORT to fail when validating downstream ops. Leave type as None and let
        // ORT's own shape inference compute the correct output shape.
        //
        // Propagate ONLY from the primary data input (input 0). Shape-preserving
        // ops carry the shape of their data tensor; their remaining inputs are
        // parameters (e.g. BatchNormalization's scale/bias/mean/var, each rank-1
        // `[C]`) whose shape is NOT the output shape. Scanning every input for the
        // first known type wrongly grabs such a rank-1 param when input 0's type
        // is not yet known, mis-annotating e.g. a ViT BatchNorm output as `{48}`;
        // ORT then aborts the whole pass at the next `Transpose` (perm rank
        // mismatch). Using input 0 only keeps the annotation rank-correct or
        // absent (letting ORT infer it).
        let inferred_type = if !is_standard_onnx_domain(&node.domain) {
            None
        } else {
            match node.op_type.as_str() {
                "Cast" => (|| {
                    let mut ty = node
                        .input
                        .first()
                        .filter(|name| !name.is_empty())
                        .and_then(|input| type_map.get(input).cloned())?;
                    let mut targets = node
                        .attribute
                        .iter()
                        .filter(|attribute| attribute.name == "to");
                    let target = targets.next()?;
                    if target.r#type != onnx_proto::attribute_type::INT || targets.next().is_some()
                    {
                        return None;
                    }
                    ty.tensor_type.as_mut()?.elem_type = i32::try_from(target.i_value()).ok()?;
                    Some(ty)
                })(),
                "CastLike" => (|| {
                    let mut ty = node
                        .input
                        .first()
                        .filter(|name| !name.is_empty())
                        .and_then(|input| type_map.get(input).cloned())?;
                    let target_elem_type = node
                        .input
                        .get(1)
                        .filter(|name| !name.is_empty())
                        .and_then(|input| type_map.get(input))?
                        .tensor_type
                        .as_ref()?
                        .elem_type;
                    ty.tensor_type.as_mut()?.elem_type = target_elem_type;
                    Some(ty)
                })(),
                op_type if shape_preserving_ops.contains(op_type) => node
                    .input
                    .first()
                    .filter(|name| !name.is_empty())
                    .and_then(|input| type_map.get(input).cloned()),
                _ => None,
            }
        };
        for output_name in &node.output {
            if output_name.is_empty()
                || existing_outputs.contains(output_name)
                || input_names.contains(output_name)
                || initializer_names.contains(output_name)
                || !new_output_names.insert(output_name.clone())
            {
                continue;
            }

            new_outputs.push(onnx_proto::ValueInfoProto {
                name: output_name.clone(),
                r#type: inferred_type.clone(),
            });
            if let Some(output_type) = inferred_type.as_ref() {
                type_map.insert(output_name.clone(), output_type.clone());
            }
        }
    }

    if !new_outputs.is_empty() {
        debug!(
            "Adding {} intermediate outputs for shape inference",
            new_outputs.len()
        );
        graph.output.extend(new_outputs);
    }

    let mut buf = Vec::new();
    model.encode(&mut buf).map_err(|e| {
        NyError::ModelLoad(format!("Failed to serialize ONNX model for inference: {e}"))
    })?;

    Ok(ExposedModel {
        bytes: buf,
        has_runtime_inputs,
    })
}

pub(crate) mod subprocess;

// NOTE: Keep test modules uniquely named to avoid duplicate module collisions.
// These exercise `expose_intermediate_outputs`, which is only compiled with the
// `ort` feature, so gate the test modules on it too.
#[cfg(all(test, feature = "ort"))]
mod expose_intermediate_outputs_serialization_tests;
#[cfg(all(test, feature = "ort"))]
mod expose_intermediate_outputs_tests;
// The subprocess protocol itself is feature-independent (without `ort` the
// server answers with an empty shape table), so its tests are not gated.
#[cfg(test)]
mod subprocess_tests;
