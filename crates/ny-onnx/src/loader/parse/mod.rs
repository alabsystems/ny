// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod activation_schema;
mod arithmetic_schema;
mod linear_schema;
mod literal_cone;
mod metadata;
mod prepare;
mod quantization_preflight;
mod reduction_schema;
mod schema_preflight;
mod structural_schema;
#[cfg(test)]
mod tests;
mod transform_schema;

use crate::onnx_proto;
use crate::WeightStore;
use ny_core::{NyError, Result};
use prost::Message;
use std::collections::HashMap;
use tracing::{debug, warn};

use super::batch_resolve::{resolve_batch_dim, VERIFICATION_BATCH_SIZE};
use super::convert::convert_graph_to_layers;
use super::external_data::{reject_external_data_without_origin, ExternalDataResolver};
use super::shape_infer::subprocess::infer_tensor_shapes_via_subprocess;
use super::shape_infer::{infer_tensor_shapes_from_ort, infer_tensor_shapes_from_ort_path};
use super::{
    BatchNormFoldingPolicy, CustomOpRegistry, ParsedOnnx, ShapeInferBackend, ShapeInferencePolicy,
};
use metadata::{build_parse_metadata, build_tensor_shapes, collect_opset_imports};
use prepare::prepare_graph;

/// Parse ONNX file to extract weights, layers, and I/O specs.
pub(super) fn parse_onnx_file<P: AsRef<std::path::Path>>(
    path: P,
    registry: &CustomOpRegistry,
    shape_inference_policy: ShapeInferencePolicy,
    shape_infer_backend: &ShapeInferBackend,
    merge_linear_enabled: bool,
    batch_norm_folding: BatchNormFoldingPolicy,
    capture_raw_float32_initializer_provenance: bool,
) -> Result<ParsedOnnx> {
    let mut external_data = ExternalDataResolver::for_model_path(path.as_ref())?;
    // Read the model and every later sidecar through one retained directory
    // capability (supports `.onnx` and `.onnx.gz`).
    let data = external_data.read_model_bytes(path.as_ref())?;
    let model = decode_onnx_model(&data)?;
    // Validate every untrusted external path before considering ORT. ORT owns
    // an independent filesystem loader and cannot consume our capability-open
    // file handles, so external-data models intentionally use authored shape
    // metadata rather than re-opening side files through a second trust path.
    let has_external_data = external_data.validate_model(&model)?;
    let inferred_shapes = if has_external_data {
        if shape_inference_policy == ShapeInferencePolicy::Ort {
            warn!(
                "ONNX Runtime shape inference skipped for external-data model; \
                 using model-authored shape metadata"
            );
        }
        HashMap::new()
    } else {
        infer_file_shapes(
            path.as_ref(),
            &data,
            shape_inference_policy,
            shape_infer_backend,
        )
    };
    parse_onnx_model(
        model,
        registry,
        inferred_shapes,
        merge_linear_enabled,
        batch_norm_folding,
        capture_raw_float32_initializer_provenance,
        Some(&mut external_data),
    )
}

/// Parse ONNX bytes to extract weights, layers, and I/O specs.
pub(super) fn parse_onnx_bytes(
    data: &[u8],
    registry: &CustomOpRegistry,
    shape_inference_policy: ShapeInferencePolicy,
    shape_infer_backend: &ShapeInferBackend,
    merge_linear_enabled: bool,
    batch_norm_folding: BatchNormFoldingPolicy,
    capture_raw_float32_initializer_provenance: bool,
) -> Result<ParsedOnnx> {
    let model = decode_onnx_model(data)?;
    // The display-only `name` accepted by the public byte APIs must never be
    // interpreted as filesystem authority.
    reject_external_data_without_origin(&model)?;
    let inferred_shapes = infer_byte_shapes(data, shape_inference_policy, shape_infer_backend);
    parse_onnx_model(
        model,
        registry,
        inferred_shapes,
        merge_linear_enabled,
        batch_norm_folding,
        capture_raw_float32_initializer_provenance,
        None,
    )
}

fn decode_onnx_model(data: &[u8]) -> Result<onnx_proto::ModelProto> {
    onnx_proto::ModelProto::decode(data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse ONNX: {e}")))
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

fn parse_onnx_model(
    mut model: onnx_proto::ModelProto,
    registry: &CustomOpRegistry,
    mut inferred_shapes: HashMap<String, Vec<i64>>,
    merge_linear_enabled: bool,
    batch_norm_folding: BatchNormFoldingPolicy,
    capture_raw_float32_initializer_provenance: bool,
    mut external_data: Option<&mut ExternalDataResolver>,
) -> Result<ParsedOnnx> {
    let opset_imports = collect_opset_imports(&model)?;

    let mut graph = model
        .graph
        .take()
        .ok_or_else(|| NyError::ModelLoad("Model has no graph".to_string()))?;

    // Validate authored standard schemas while attribute variants and their
    // operator-set authority are still visible. Constant folding can erase
    // both the producer node and the dtype/attribute provenance.
    let literal_exemptions = schema_preflight::validate_standard_schemas(&graph, &opset_imports)?;

    // Quantization constants are deliberately inspected at the raw protobuf
    // boundary.  `prepare_graph` normalizes authored integer/floating tensor
    // types into WeightStore's f32 view and may erase a fully constant Q/DQ
    // node, after which unsupported precision or dtype semantics could no
    // longer be distinguished from ordinary FLOAT32 arithmetic.
    quantization_preflight::validate_quantization_schemas_with_external(
        &graph,
        &opset_imports,
        external_data.as_deref_mut(),
    )?;

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
        external_data,
    )?;

    // Discharge every raw-schema refusal that was deferred because its node
    // looked like a load-time literal. Constant folding has now run, so the
    // question the preflight could not answer — "does this node still exist?" —
    // is decidable. Any survivor re-raises its original refusal.
    literal_exemptions.require_all_folded(&weights)?;

    // Build tensor shapes before graph conversion because proto-level fusion
    // inspects the raw ONNX graph and the staged weight set.
    let tensor_shapes = build_tensor_shapes(&graph, &weights, &inferred_shapes);
    let graph_output_names = graph
        .output
        .iter()
        .filter(|output| !output.name.is_empty())
        .map(|output| output.name.clone())
        .collect();
    let raw_int64_shape_values = super::const_fold::raw_int64_shape_values(&graph, &weights);
    let layers = convert_graph_to_layers(
        &mut graph.node,
        &mut weights,
        registry,
        &opset_imports,
        &tensor_shapes,
        &graph_output_names,
        &raw_int64_shape_values,
        merge_linear_enabled,
        batch_norm_folding,
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
