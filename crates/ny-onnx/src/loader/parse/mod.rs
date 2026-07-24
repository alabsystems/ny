// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod metadata;
mod prepare;
#[cfg(test)]
mod tests;

use crate::io;
use crate::onnx_proto;
use crate::WeightStore;
use ny_core::{NyError, Result};
use prost::Message;
use std::collections::HashMap;
use tracing::{debug, warn};

use super::batch_resolve::{resolve_batch_dim, VERIFICATION_BATCH_SIZE};
use super::convert::convert_graph_to_layers;
use super::shape_infer::subprocess::infer_tensor_shapes_via_subprocess;
use super::shape_infer::{infer_tensor_shapes_from_ort, infer_tensor_shapes_from_ort_path};
use super::{CustomOpRegistry, ParsedOnnx, ShapeInferBackend, ShapeInferencePolicy};
use metadata::{build_parse_metadata, build_tensor_shapes, collect_opset_imports};
use prepare::prepare_graph;

/// Parse ONNX file to extract weights, layers, and I/O specs.
pub(super) fn parse_onnx_file<P: AsRef<std::path::Path>>(
    path: P,
    registry: &CustomOpRegistry,
    shape_inference_policy: ShapeInferencePolicy,
    shape_infer_backend: &ShapeInferBackend,
    merge_linear_enabled: bool,
    capture_raw_float32_initializer_provenance: bool,
) -> Result<ParsedOnnx> {
    // Read the file (supports `.onnx` and `.onnx.gz`)
    let data = io::read_bytes_maybe_gzip(path.as_ref())?;
    let inferred_shapes = infer_file_shapes(
        path.as_ref(),
        &data,
        shape_inference_policy,
        shape_infer_backend,
    );
    parse_onnx_data(
        &data,
        registry,
        inferred_shapes,
        merge_linear_enabled,
        capture_raw_float32_initializer_provenance,
    )
}

/// Parse ONNX bytes to extract weights, layers, and I/O specs.
pub(super) fn parse_onnx_bytes(
    data: &[u8],
    registry: &CustomOpRegistry,
    shape_inference_policy: ShapeInferencePolicy,
    shape_infer_backend: &ShapeInferBackend,
    merge_linear_enabled: bool,
    capture_raw_float32_initializer_provenance: bool,
) -> Result<ParsedOnnx> {
    let inferred_shapes = infer_byte_shapes(data, shape_inference_policy, shape_infer_backend);
    parse_onnx_data(
        data,
        registry,
        inferred_shapes,
        merge_linear_enabled,
        capture_raw_float32_initializer_provenance,
    )
}

fn infer_file_shapes(
    path: &std::path::Path,
    data: &[u8],
    shape_inference_policy: ShapeInferencePolicy,
    backend: &ShapeInferBackend,
) -> HashMap<String, Vec<i64>> {
    match shape_inference_policy {
        ShapeInferencePolicy::Ort => {
            let inferred = match backend {
                ShapeInferBackend::InProcess => infer_tensor_shapes_from_ort_path(path, data),
                // The subprocess protocol streams the (already gz-decoded)
                // model bytes over stdin, so the file-path variant is not
                // needed there.
                ShapeInferBackend::Subprocess { exe } => {
                    debug!(
                        "Delegating ORT shape inference to subprocess {}",
                        exe.display()
                    );
                    infer_tensor_shapes_via_subprocess(exe, data)
                }
            };
            inferred.unwrap_or_else(|err| {
                warn!("ONNX Runtime shape inference skipped: {}", err);
                HashMap::new()
            })
        }
        ShapeInferencePolicy::Skip => HashMap::new(),
    }
}

fn infer_byte_shapes(
    data: &[u8],
    shape_inference_policy: ShapeInferencePolicy,
    backend: &ShapeInferBackend,
) -> HashMap<String, Vec<i64>> {
    match shape_inference_policy {
        ShapeInferencePolicy::Ort => {
            let inferred = match backend {
                ShapeInferBackend::InProcess => infer_tensor_shapes_from_ort(data),
                ShapeInferBackend::Subprocess { exe } => {
                    debug!(
                        "Delegating ORT shape inference to subprocess {}",
                        exe.display()
                    );
                    infer_tensor_shapes_via_subprocess(exe, data)
                }
            };
            inferred.unwrap_or_else(|err| {
                warn!("ONNX Runtime shape inference skipped: {}", err);
                HashMap::new()
            })
        }
        ShapeInferencePolicy::Skip => HashMap::new(),
    }
}

fn parse_onnx_data(
    data: &[u8],
    registry: &CustomOpRegistry,
    mut inferred_shapes: HashMap<String, Vec<i64>>,
    merge_linear_enabled: bool,
    capture_raw_float32_initializer_provenance: bool,
) -> Result<ParsedOnnx> {
    // Parse as ONNX ModelProto
    let mut model = onnx_proto::ModelProto::decode(data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse ONNX: {}", e)))?;

    let opset_imports = collect_opset_imports(&model);

    let mut graph = model
        .graph
        .take()
        .ok_or_else(|| NyError::ModelLoad("Model has no graph".to_string()))?;

    // Pin a symbolic/dynamic leading (batch) dimension on runtime graph inputs
    // to the concrete verification batch size BEFORE shape inference and
    // const-folding. This lets the const-fold `Shape` path read a static leading
    // dim and fold transformer attention reshape→transpose chains to the correct
    // rank, instead of emitting copy-axis sentinels for a `-1` batch axis. The
    // bytes handed to ORT for shape extraction are pinned with the same rule (see
    // `shape_infer::expose_intermediate_outputs`) so all consumers agree on the
    // dim. Soundness: shape metadata only — never a weight, op, or attribute;
    // see `batch_resolve` for the rule and the byte-identical-output argument.
    if resolve_batch_dim(&mut graph, VERIFICATION_BATCH_SIZE) {
        debug!(
            "Resolved symbolic batch dimension to {} on runtime graph inputs",
            VERIFICATION_BATCH_SIZE
        );
    }

    let mut weights = WeightStore::new();
    if capture_raw_float32_initializer_provenance && !weights.enable_revision_tracking() {
        return Err(NyError::ModelLoad(
            "failed to enable raw initializer revision tracking before weight loading".to_string(),
        ));
    }
    let original_float32_initializers = prepare_graph(
        &mut graph,
        &mut weights,
        &mut inferred_shapes,
        capture_raw_float32_initializer_provenance,
    )?;

    // Build tensor shapes before graph conversion because proto-level fusion
    // inspects the raw ONNX graph and the staged weight set.
    let tensor_shapes = build_tensor_shapes(&graph, &weights, &inferred_shapes);
    let graph_output_names = graph
        .output
        .iter()
        .filter(|output| !output.name.is_empty())
        .map(|output| output.name.clone())
        .collect();
    let layers = convert_graph_to_layers(
        &mut graph.node,
        &mut weights,
        registry,
        &opset_imports,
        &tensor_shapes,
        &graph_output_names,
        merge_linear_enabled,
    )?;

    let metadata = build_parse_metadata(&graph, &weights, tensor_shapes)?;

    // Validate no NaN in weights after all loading, folding, and fusion (#2791).
    weights.validate_no_nan()?;

    Ok((
        layers,
        weights,
        metadata.inputs,
        metadata.outputs,
        metadata.tensor_producer,
        metadata.constant_tensors,
        metadata.tensor_shapes,
        opset_imports,
        original_float32_initializers,
    ))
}
