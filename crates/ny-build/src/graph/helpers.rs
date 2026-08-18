// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::INPUT_NODE_NAME;
use crate::{AttributeValue, LayerSpec, TensorSpec, WeightStore};
use ndarray::{ArrayD, Axis, Slice};
use ny_core::{NyError, Result};
use ny_propagate::layers::{OpaqueSkipLayer, SliceLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

pub(super) fn resolve_tensor_node_name(
    tensor_name: &str,
    tensor_to_node: &HashMap<String, String>,
    tensor_producer: &HashMap<String, String>,
) -> Option<String> {
    let mut current = tensor_name;
    let mut seen = HashSet::new();
    loop {
        if let Some(node_name) = tensor_to_node.get(current) {
            return Some(node_name.clone());
        }
        if !seen.insert(current.to_string()) {
            return None;
        }
        current = tensor_producer.get(current)?;
    }
}

pub(super) fn find_activation_inputs(
    inputs: &[String],
    weights: &WeightStore,
    constant_tensors: &HashSet<String>,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
) -> Vec<String> {
    inputs
        .iter()
        .filter(|name| {
            weights.get(name).is_none()
                && weights.get_integers(name).is_none()
                && !constant_tensors.contains(*name)
                && !evaluated_constants.contains_key(*name)
        })
        .cloned()
        .collect()
}

pub(super) fn map_outputs_to_node(
    tensor_to_node: &mut HashMap<String, String>,
    outputs: &[String],
    src_node: &str,
) {
    // Clone src_node to avoid holding an immutable borrow during inserts.
    for output in outputs {
        tensor_to_node.insert(output.clone(), src_node.to_string());
    }
}

pub(super) fn map_outputs_to_activation_inputs_or_input(
    tensor_to_node: &mut HashMap<String, String>,
    outputs: &[String],
    activation_inputs: &[String],
    op_name: &str,
) -> Result<()> {
    if activation_inputs.is_empty() {
        map_outputs_to_node(tensor_to_node, outputs, INPUT_NODE_NAME);
        return Ok(());
    }

    // Resolve all activation inputs up front to avoid borrow conflicts with
    // the mutable tensor_to_node inserts below. Network inputs are pre-populated
    // in tensor_to_node, so None means a dangling reference.
    let resolved: Vec<String> = activation_inputs
        .iter()
        .map(|input| {
            tensor_to_node.get(input).cloned().ok_or_else(|| {
                warn!(
                    "Identity pass-through '{}': activation input '{}' not found in tensor_to_node",
                    op_name, input
                );
                NyError::ModelLoad(format!(
                    "Identity pass-through '{}' references unresolvable activation input '{}' \
                     — no producer found in graph",
                    op_name, input
                ))
            })
        })
        .collect::<Result<Vec<String>>>()?;

    if resolved.len() >= outputs.len() {
        for (output, src_node) in outputs.iter().zip(resolved.iter()) {
            tensor_to_node.insert(output.clone(), src_node.clone());
        }
        return Ok(());
    }

    for (idx, output) in outputs.iter().enumerate() {
        let src_node = &resolved[idx % resolved.len()];
        tensor_to_node.insert(output.clone(), src_node.clone());
    }
    Ok(())
}

fn unbatched_recorded_shape(shape: Vec<usize>) -> Vec<usize> {
    if shape.len() > 1 {
        shape[1..].to_vec()
    } else {
        shape
    }
}

fn tensor_spec_shape(input_specs: &[TensorSpec], name: &str) -> Option<Vec<usize>> {
    input_specs
        .iter()
        .find(|spec| spec.name == name)
        .and_then(|spec| {
            if spec.shape.iter().all(|&dim| dim > 0) {
                Some(unbatched_recorded_shape(
                    spec.shape.iter().map(|&dim| dim as usize).collect(),
                ))
            } else {
                None
            }
        })
}

fn tensor_shapes_map(tensor_shapes: &HashMap<String, Vec<i64>>, name: &str) -> Option<Vec<usize>> {
    tensor_shapes.get(name).and_then(|shape| {
        if shape.iter().all(|&dim| dim > 0) {
            Some(unbatched_recorded_shape(
                shape.iter().map(|&dim| dim as usize).collect(),
            ))
        } else {
            None
        }
    })
}

fn infer_input_shape(
    input_name: &str,
    weights: &WeightStore,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
    input_specs: &[TensorSpec],
    tensor_shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<usize>> {
    weights
        .get(input_name)
        .map(|tensor| tensor.shape().to_vec())
        .or_else(|| {
            evaluated_constants
                .get(input_name)
                .map(|tensor| tensor.shape().to_vec())
        })
        .or_else(|| tensor_spec_shape(input_specs, input_name))
        .or_else(|| tensor_shapes_map(tensor_shapes, input_name))
}

fn known_axis_len(
    spec: &LayerSpec,
    input_name: &str,
    axis: i32,
    weights: &WeightStore,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
    input_specs: &[TensorSpec],
    tensor_shapes: &HashMap<String, Vec<i64>>,
) -> Result<Option<usize>> {
    let Some(shape) = infer_input_shape(
        input_name,
        weights,
        evaluated_constants,
        input_specs,
        tensor_shapes,
    ) else {
        return Ok(None);
    };
    if shape.is_empty() {
        return Ok(None);
    }
    let resolved_axis = resolve_split_axis_for_shape(spec, axis, shape.len())?;
    Ok(Some(shape[resolved_axis]))
}

pub(super) fn infer_equal_split_sizes(
    input_name: &str,
    axis: i32,
    num_outputs: usize,
    weights: &WeightStore,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
    input_specs: &[TensorSpec],
    tensor_shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<usize>> {
    let shape = infer_input_shape(
        input_name,
        weights,
        evaluated_constants,
        input_specs,
        tensor_shapes,
    )?;
    if shape.is_empty() || num_outputs == 0 {
        return None;
    }
    let rank = shape.len() as i32;
    let axis = if axis < 0 { axis + rank } else { axis };
    if axis < 0 || axis >= rank {
        return None;
    }
    let axis_len = shape[axis as usize];
    if axis_len == 0 || axis_len % num_outputs != 0 {
        return None;
    }
    let chunk = axis_len / num_outputs;
    Some(vec![chunk; num_outputs])
}

#[derive(Debug)]
pub(super) enum SplitGraphBuildOutcome {
    Handled,
    Skipped,
}

type ConstantSplitOutputs = Vec<(String, ArrayD<f32>)>;

pub(super) struct SplitBuildContext<'a> {
    pub weights: &'a WeightStore,
    pub evaluated_constants: &'a HashMap<String, ArrayD<f32>>,
    pub constant_tensors: &'a HashSet<String>,
    pub inputs: &'a [TensorSpec],
    pub tensor_shapes: &'a HashMap<String, Vec<i64>>,
    pub graph: &'a mut GraphNetwork,
    pub tensor_to_node: &'a mut HashMap<String, String>,
    pub last_added_node: &'a mut Option<String>,
}

fn parse_constant_split_sizes(spec: &LayerSpec, split_tensor: &ArrayD<f32>) -> Result<Vec<usize>> {
    if split_tensor.ndim() != 1 {
        return Err(NyError::ModelLoad(format!(
            "Split '{}' partition sizes must be a 1-D tensor, got shape {:?}",
            spec.name,
            split_tensor.shape()
        )));
    }
    split_tensor
        .iter()
        .map(|&value| {
            if !value.is_finite() {
                return Err(NyError::ModelLoad(format!(
                    "Split '{}' has non-finite split size {value}",
                    spec.name
                )));
            }
            let rounded = value.round();
            if value != rounded || rounded < 0.0 {
                return Err(NyError::ModelLoad(format!(
                    "Split '{}' has invalid split size {value}",
                    spec.name
                )));
            }
            if rounded >= i64::MAX as f32 {
                return Err(NyError::ModelLoad(format!(
                    "Split '{}' split size {rounded} is outside the non-saturating i64 range",
                    spec.name
                )));
            }
            usize::try_from(rounded as i64).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Split '{}' split size {rounded} overflows usize",
                    spec.name
                ))
            })
        })
        .collect()
}

/// Parse partition sizes from an int64 constant tensor (ONNX opset>=13 Split input form).
fn parse_integer_constant_split_sizes(
    spec: &LayerSpec,
    split_tensor: &ArrayD<i64>,
) -> Result<Vec<usize>> {
    if split_tensor.ndim() != 1 {
        return Err(NyError::ModelLoad(format!(
            "Split '{}' partition sizes must be a 1-D tensor, got shape {:?}",
            spec.name,
            split_tensor.shape()
        )));
    }
    split_tensor
        .iter()
        .map(|&value| {
            if value < 0 {
                return Err(NyError::ModelLoad(format!(
                    "Split '{}': negative split size {}",
                    spec.name, value
                )));
            }
            usize::try_from(value).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Split '{}': split size {} overflows usize",
                    spec.name, value
                ))
            })
        })
        .collect()
}

/// Resolve a Split ONNX axis for unbatched propagation.
///
/// TRAILING-RELATIVE remap (mirrors `ConvertContext::remap_axis_trailing`,
/// inlined because graph helpers do not hold a `ConvertContext`): ny's
/// internal runtime tensor for an ONNX tensor of rank `r` either kept its
/// ONNX rank (leading size-1 retained, e.g. Flatten / rank-2 Gemm outputs)
/// or had its leading batch dim stripped (rank `r-1`). Both layouts share
/// the ONNX trailing dims, so a negative axis (`onnx_axis - r`) selects the
/// same semantic dimension in either layout; the lowered `SliceLayer`s and
/// the load-time size helpers (`resolve_split_axis_for_shape`,
/// `infer_equal_split_sizes`) all resolve negative axes against the actual
/// rank. The legacy `onnx_axis - 1` guess was only correct for the stripped
/// layout (#pensieve ReduceSum no-op defect class).
///
/// Fail-closed: unknown recorded rank, out-of-range axes, and the ambiguous
/// batch axis 0 of a rank>1 tensor refuse conversion. `onnx_axis == 0` on a
/// rank-≤1 input is a genuine data axis (cctsdb pattern) and maps to `-1`.
/// Load-time axis interpretation only — no bound math.
fn split_axis_from_spec(
    spec: &LayerSpec,
    weights: &WeightStore,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
    tensor_shapes: &HashMap<String, Vec<i64>>,
) -> Result<i32> {
    let onnx_axis = match spec.attributes.get("axis") {
        Some(AttributeValue::Int(axis)) => i32::try_from(*axis).map_err(|_| {
            NyError::ModelLoad(format!(
                "Split '{}': axis {} is outside the supported i32 range",
                spec.name, axis
            ))
        })?,
        None => 0,
        Some(other) => {
            return Err(NyError::ModelLoad(format!(
                "Split '{}': invalid axis attribute {:?}",
                spec.name, other
            )))
        }
    };
    // Recorded ONNX rank of the data input. Mirrors
    // `ConvertContext::recorded_onnx_rank`, including the source priority:
    // `tensor_shapes` (authoritative ONNX record) first — `evaluated_constants`
    // are materialized in the INTERNAL (possibly batch-stripped) convention and
    // can understate the ONNX rank.
    let recorded_rank = spec.inputs.first().and_then(|input| {
        tensor_shapes
            .get(input)
            .map(|shape| shape.len())
            .or_else(|| weights.get(input).map(|tensor| tensor.ndim()))
            .or_else(|| evaluated_constants.get(input).map(|tensor| tensor.ndim()))
    });
    if onnx_axis < 0 {
        if let Some(rank) = recorded_rank {
            if i64::from(onnx_axis) < -(rank as i64) {
                return Err(NyError::ModelLoad(format!(
                    "Split '{}': ONNX axis {} out of range for recorded rank {}",
                    spec.name, onnx_axis, rank
                )));
            }
        }
        return Ok(onnx_axis);
    }
    let Some(rank) = recorded_rank else {
        // Legacy-compatibility branch (mirrors `remap_axis_trailing`): tensors
        // without recorded ONNX shapes come from ny-synthesized internal
        // subgraphs written against the stripped-batch convention, for which
        // the legacy adjustment is correct by construction.
        if onnx_axis == 0 {
            return Err(NyError::ModelLoad(format!(
                "Split '{}': axis=0 targets batch dimension which does not exist in unbatched mode",
                spec.name
            )));
        }
        debug!(
            "Split '{}': input has no recorded ONNX shape; keeping legacy batch-squeeze \
             adjustment {} -> {} (synthesized-subgraph compatibility)",
            spec.name,
            onnx_axis,
            onnx_axis - 1
        );
        return Ok(onnx_axis - 1);
    };
    if onnx_axis as usize >= rank {
        return Err(NyError::ModelLoad(format!(
            "Split '{}': ONNX axis {} out of range for recorded rank {}",
            spec.name, onnx_axis, rank
        )));
    }
    if onnx_axis == 0 && rank > 1 {
        return Err(NyError::ModelLoad(format!(
            "Split '{}': axis=0 targets batch dimension which does not exist in unbatched mode",
            spec.name
        )));
    }
    Ok(onnx_axis - rank as i32)
}

fn resolve_split_axis_for_shape(spec: &LayerSpec, axis: i32, ndim: usize) -> Result<usize> {
    let resolved = if axis >= 0 {
        axis as isize
    } else {
        ndim as isize + axis as isize
    };
    if resolved < 0 || resolved as usize >= ndim {
        return Err(NyError::ModelLoad(format!(
            "Split '{}': axis {} is out of bounds for rank-{} input",
            spec.name, axis, ndim
        )));
    }
    Ok(resolved as usize)
}

fn resolve_split_sizes(
    spec: &LayerSpec,
    axis: i32,
    weights: &WeightStore,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
    inputs: &[TensorSpec],
    tensor_shapes: &HashMap<String, Vec<i64>>,
) -> Result<Option<Vec<usize>>> {
    let split_input_name = spec.inputs.get(1).filter(|name| !name.is_empty());
    if spec.attributes.contains_key("split") && split_input_name.is_some() {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Split '{}' supplies partition sizes as both an attribute and an input",
            spec.name
        )));
    }
    // Try attribute-based form first (older opsets or early opset>=13 fallback)
    match spec.attributes.get("split") {
        Some(AttributeValue::Ints(splits)) => splits
            .iter()
            .map(|&size| {
                usize::try_from(size).map_err(|_| {
                    NyError::ModelLoad(format!(
                        "Split '{}': negative split size {}",
                        spec.name, size
                    ))
                })
            })
            .collect::<Result<Vec<usize>>>()
            .map(Some),
        None => {
            // Try input-based form (opset>=13): split partition sizes are an input tensor
            if let Some(split_input_name) = split_input_name {
                // First try int64 integers store (opset>=13 Split uses int64)
                if let Some(split_tensor) = weights.get_integers(split_input_name) {
                    if weights
                        .get(split_input_name)
                        .is_some_and(|floats| floats.shape() != split_tensor.shape())
                    {
                        return Err(NyError::ModelLoad(format!(
                            "Split '{}': exact integer partition tensor '{}' has shape {:?}, but its float view has shape {:?}",
                            spec.name,
                            split_input_name,
                            split_tensor.shape(),
                            weights.get(split_input_name).map(|tensor| tensor.shape()).unwrap_or(&[])
                        )));
                    }
                    return Ok(Some(parse_integer_constant_split_sizes(
                        spec,
                        split_tensor,
                    )?));
                }
                // Fall back to f32 constant tensors (edge case from const folding)
                if let Some(split_tensor) = weights.get(split_input_name) {
                    return Ok(Some(parse_constant_split_sizes(spec, split_tensor)?));
                }
                if let Some(split_tensor) = evaluated_constants.get(split_input_name) {
                    return Ok(Some(parse_constant_split_sizes(spec, split_tensor)?));
                }
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Split '{}' requires partition-size input '{}' to be constant",
                    spec.name, split_input_name
                )));
            } else {
                warn!(
                    "Split op '{}' missing 'split' attribute, using equal splits",
                    spec.name
                );
            }

            let input_name = spec.inputs.first().map(String::as_str).unwrap_or("");
            Ok(infer_equal_split_sizes(
                input_name,
                axis,
                spec.outputs.len(),
                weights,
                evaluated_constants,
                inputs,
                tensor_shapes,
            ))
        }
        Some(other) => Err(NyError::ModelLoad(format!(
            "Split '{}': invalid split attribute {:?}",
            spec.name, other
        ))),
    }
}

fn checked_split_total(spec: &LayerSpec, split_sizes: &[usize]) -> Result<usize> {
    split_sizes.iter().try_fold(0usize, |total, &size| {
        total.checked_add(size).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "Split '{}': partition-size sum overflows usize",
                spec.name
            ))
        })
    })
}

pub(super) fn evaluate_constant_split_outputs(
    spec: &LayerSpec,
    weights: &WeightStore,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
    inputs: &[TensorSpec],
    tensor_shapes: &HashMap<String, Vec<i64>>,
) -> Result<Option<ConstantSplitOutputs>> {
    let input_name = match spec.inputs.first().filter(|name| !name.is_empty()) {
        Some(name) => name,
        None => return Ok(None),
    };
    let input = if let Some(weight) = weights.get(input_name) {
        weight
    } else if let Some(evaluated) = evaluated_constants.get(input_name) {
        evaluated
    } else {
        return Ok(None);
    };

    // Trailing-relative axis: resolving against the constant's FULL ONNX rank
    // yields exactly the ONNX axis (constants are never batch-stripped) — the
    // legacy `axis - 1` guess was off by one here.
    let axis = split_axis_from_spec(spec, weights, evaluated_constants, tensor_shapes)?;
    let resolved_axis = resolve_split_axis_for_shape(spec, axis, input.ndim())?;
    let Some(split_sizes) = resolve_split_sizes(
        spec,
        axis,
        weights,
        evaluated_constants,
        inputs,
        tensor_shapes,
    )?
    else {
        return Ok(None);
    };

    if split_sizes.len() != spec.outputs.len() {
        warn!(
            "Split op '{}' has {} split sizes but {} outputs - skipping constant evaluation",
            spec.name,
            split_sizes.len(),
            spec.outputs.len()
        );
        return Ok(None);
    }

    let axis_len = input.shape()[resolved_axis];
    let total_split = checked_split_total(spec, &split_sizes)?;
    if total_split != axis_len {
        return Err(NyError::ModelLoad(format!(
            "Split '{}': split sizes {:?} sum to {}, expected axis length {}",
            spec.name, split_sizes, total_split, axis_len
        )));
    }

    let mut start = 0usize;
    let mut outputs = Vec::with_capacity(spec.outputs.len());
    for (output_name, &size) in spec.outputs.iter().zip(split_sizes.iter()) {
        let end = start.checked_add(size).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "Split '{}': slice endpoint overflows usize",
                spec.name
            ))
        })?;
        let slice = input
            .slice_axis(Axis(resolved_axis), Slice::from(start..end))
            .to_owned();
        outputs.push((output_name.clone(), slice));
        start = end;
    }
    Ok(Some(outputs))
}

pub(super) fn handle_split_layer(
    spec: &LayerSpec,
    split_ctx: &mut SplitBuildContext<'_>,
) -> Result<SplitGraphBuildOutcome> {
    let axis = split_axis_from_spec(
        spec,
        split_ctx.weights,
        split_ctx.evaluated_constants,
        split_ctx.tensor_shapes,
    )?;
    let activation_inputs = find_activation_inputs(
        &spec.inputs,
        split_ctx.weights,
        split_ctx.constant_tensors,
        split_ctx.evaluated_constants,
    );

    let Some(split_sizes) = resolve_split_sizes(
        spec,
        axis,
        split_ctx.weights,
        split_ctx.evaluated_constants,
        split_ctx.inputs,
        split_ctx.tensor_shapes,
    )?
    else {
        let input_name = spec.inputs.first().map(String::as_str).unwrap_or("");
        warn!(
            "Split op '{}' could not infer axis length for '{}'; skipping",
            spec.name, input_name
        );
        let declared_shape =
            declared_output_shape(&spec.outputs, split_ctx.tensor_shapes, split_ctx.inputs);
        map_skipped_outputs(
            split_ctx.graph,
            split_ctx.tensor_to_node,
            &spec.outputs,
            &activation_inputs,
            &spec.name,
            split_ctx.last_added_node,
            declared_shape,
        )?;
        return Ok(SplitGraphBuildOutcome::Skipped);
    };

    if split_sizes.len() != spec.outputs.len() {
        warn!(
            "Split op '{}' has {} split sizes but {} outputs - skipping",
            spec.name,
            split_sizes.len(),
            spec.outputs.len()
        );
        let declared_shape =
            declared_output_shape(&spec.outputs, split_ctx.tensor_shapes, split_ctx.inputs);
        map_skipped_outputs(
            split_ctx.graph,
            split_ctx.tensor_to_node,
            &spec.outputs,
            &activation_inputs,
            &spec.name,
            split_ctx.last_added_node,
            declared_shape,
        )?;
        return Ok(SplitGraphBuildOutcome::Skipped);
    }

    let input_name = spec.inputs.first().map(String::as_str).unwrap_or("");
    if let Some(axis_len) = known_axis_len(
        spec,
        input_name,
        axis,
        split_ctx.weights,
        split_ctx.evaluated_constants,
        split_ctx.inputs,
        split_ctx.tensor_shapes,
    )? {
        let total_split = checked_split_total(spec, &split_sizes)?;
        if total_split != axis_len {
            return Err(NyError::ModelLoad(format!(
                "Split '{}': split sizes {:?} sum to {}, expected axis length {}",
                spec.name, split_sizes, total_split, axis_len
            )));
        }
    }

    let input_node = match activation_inputs.first() {
        Some(input_name) => split_ctx
            .tensor_to_node
            .get(input_name)
            .cloned()
            .ok_or_else(|| {
                warn!(
                    "Split '{}': input '{}' not in tensor_to_node",
                    spec.name, input_name
                );
                NyError::ModelLoad(format!(
                    "Split '{}' references unresolvable activation input '{}' \
                 — no producer found in graph",
                    spec.name, input_name
                ))
            })?,
        None => INPUT_NODE_NAME.to_string(),
    };

    let mut start = 0usize;
    for (index, (output_name, &size)) in spec.outputs.iter().zip(split_sizes.iter()).enumerate() {
        let end = start.checked_add(size).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "Split '{}': slice endpoint overflows usize",
                spec.name
            ))
        })?;
        let slice_name = format!("{}_slice_{}", spec.name, index);
        let slice_layer = Layer::Slice(SliceLayer::new(axis, start, end));
        let node = GraphNode::new(slice_name.clone(), slice_layer, vec![input_node.clone()]);
        split_ctx.graph.try_add_node(node)?;
        *split_ctx.last_added_node = Some(slice_name.clone());
        split_ctx
            .tensor_to_node
            .insert(output_name.clone(), slice_name.clone());
        debug!(
            "Split '{}' output {} -> Slice node '{}' (start={}, end={}, axis={})",
            spec.name, index, slice_name, start, end, axis
        );
        start = end;
    }

    Ok(SplitGraphBuildOutcome::Handled)
}

/// Map an ONNX/ORT-inferred tensor shape to the internal (unbatched) shape
/// convention used by `ny-propagate`.
///
/// - Unbatched models (every graph input has rank <= 1): no batch axis was
///   ever stripped, so the ONNX shape maps verbatim (cctsdb_yolo_2023).
/// - Batched models: the leading dim is the stripped batch axis when the
///   rank is > 1 and the dim is exactly 1; any other leading dim is
///   ambiguous and yields `None`.
/// - Dynamic/symbolic dims (encoded <= 0) yield `None`.
///
/// `None` means "shape unknown": callers keep the status-quo fallback (the
/// first input's shape). This is metadata for conservative `[-inf, +inf]`
/// substitutions only — it never shapes finite bound values.
pub(super) fn internal_shape_from_onnx_shape(
    onnx_shape: &[i64],
    model_unbatched: bool,
) -> Option<Vec<usize>> {
    if onnx_shape.iter().any(|&dim| dim <= 0) {
        return None;
    }
    let dims: &[i64] = if model_unbatched || onnx_shape.len() <= 1 {
        onnx_shape
    } else if onnx_shape[0] == 1 {
        &onnx_shape[1..]
    } else {
        return None;
    };
    Some(dims.iter().map(|&dim| dim as usize).collect())
}

/// Whether the model is "unbatched": every graph input has rank <= 1, so no
/// batch axis was ever stripped during conversion and ONNX shapes map to
/// internal shapes verbatim. Mirrors `ConvertContext::data_had_batch_axis`.
pub(super) fn model_is_unbatched(model_inputs: &[TensorSpec]) -> bool {
    crate::convert::model_is_unbatched(model_inputs)
}

/// Declared internal output shape for a skipped op, from load-time
/// (ORT-inferred) tensor shapes. `None` when unknown or ambiguous.
pub(super) fn declared_output_shape(
    outputs: &[String],
    tensor_shapes: &HashMap<String, Vec<i64>>,
    model_inputs: &[TensorSpec],
) -> Option<Vec<usize>> {
    let output = outputs.first()?;
    let onnx_shape = tensor_shapes.get(output)?;
    internal_shape_from_onnx_shape(onnx_shape, model_is_unbatched(model_inputs))
}

/// Insert an OpaqueSkipLayer for any skipped unsupported op.
///
/// Previously, single-input ops were treated as identity pass-through, which is
/// unsound for non-identity ops (e.g., Reciprocal, Pow, Erf). Now all skipped
/// ops get OpaqueSkipLayer with conservative [-inf, +inf] bounds.
///
/// When `declared_shape` is known (ORT shape inference), the OpaqueSkip emits
/// its unbounded bounds in that shape instead of echoing the first input's
/// shape, keeping downstream shape-sensitive ops consistent (#cctsdb).
pub(super) fn map_skipped_outputs(
    graph: &mut GraphNetwork,
    tensor_to_node: &mut HashMap<String, String>,
    outputs: &[String],
    activation_inputs: &[String],
    op_name: &str,
    last_added_node: &mut Option<String>,
    declared_shape: Option<Vec<usize>>,
) -> Result<()> {
    // Resolve input graph nodes (deduplicated)
    let mut input_nodes = Vec::with_capacity(activation_inputs.len().max(1));
    let mut seen = HashSet::with_capacity(activation_inputs.len().max(1));
    if activation_inputs.is_empty() {
        // No activation inputs — connect to graph input
        input_nodes.push(INPUT_NODE_NAME.to_string());
    } else {
        for input in activation_inputs {
            let node_name = tensor_to_node.get(input).cloned().ok_or_else(|| {
                warn!(
                    "Skipped op '{}': activation input '{}' not found in tensor_to_node",
                    op_name, input
                );
                NyError::ModelLoad(format!(
                    "Skipped op '{}' references unresolvable activation input '{}' \
                     — no producer found in graph",
                    op_name, input
                ))
            })?;
            if seen.insert(node_name.clone()) {
                input_nodes.push(node_name);
            }
        }
    }

    let skip_name = format!("{}__skip", op_name);
    if !graph.contains_node(&skip_name) {
        let layer = match &declared_shape {
            Some(shape) => OpaqueSkipLayer::with_output_shape(shape.clone()),
            None => OpaqueSkipLayer::new(),
        };
        let node = GraphNode::new(skip_name.clone(), Layer::OpaqueSkip(layer), input_nodes);
        graph.try_add_node(node)?;
        if let Some(shape) = declared_shape {
            graph.set_declared_shape(skip_name.clone(), shape);
        }
    }

    map_outputs_to_node(tensor_to_node, outputs, &skip_name);
    *last_added_node = Some(skip_name);
    Ok(())
}

#[cfg(test)]
mod split_axis0_tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use ny_propagate::layers::BoundPropagation;
    use ny_propagate::Layer;
    use ny_tensor::BoundedTensor;

    /// Split op spec with the given axis attribute and equal-size split sizes.
    fn split_spec(axis: i64, input: &str, splits: Vec<i64>, outputs: Vec<&str>) -> LayerSpec {
        LayerSpec {
            name: "split".to_string(),
            layer_type: ny_core::LayerType::Slice, // Split is lowered to Slice nodes
            inputs: vec![input.to_string()],
            outputs: outputs.into_iter().map(str::to_string).collect(),
            weights: None,
            attributes: HashMap::from([
                ("axis".to_string(), AttributeValue::Int(axis)),
                ("split".to_string(), AttributeValue::Ints(splits)),
            ]),
        }
    }

    #[test]
    fn split_rejects_adjacent_non_integer_size() {
        let spec = split_spec(1, "x", vec![], vec!["a"]);
        let value = f32::from_bits(1.0_f32.to_bits() + 1);
        let sizes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![value]).unwrap();
        assert!(parse_constant_split_sizes(&spec, &sizes).is_err());
    }

    #[test]
    fn split_rejects_float_size_that_would_saturate_to_i64_max() {
        let spec = split_spec(1, "x", vec![], vec!["a"]);
        let sizes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![i64::MAX as f32]).unwrap();
        assert!(parse_constant_split_sizes(&spec, &sizes).is_err());
    }

    #[test]
    fn split_rejects_axis_that_would_wrap_to_i32() {
        let spec = split_spec(4_294_967_297, "x", vec![1], vec!["a"]);
        let weights = WeightStore::new();
        let evaluated = HashMap::new();
        let shapes = HashMap::from([("x".to_string(), vec![1, 1])]);
        let err = split_axis_from_spec(&spec, &weights, &evaluated, &shapes).unwrap_err();
        assert!(err.to_string().contains("i32 range"));
    }

    #[test]
    fn split_rejects_dynamic_partition_input() {
        let mut spec = split_spec(1, "x", vec![], vec!["a", "b"]);
        spec.attributes.remove("split");
        spec.inputs.push("runtime_sizes".to_string());
        let weights = WeightStore::new();
        let evaluated = HashMap::new();
        let inputs = Vec::new();
        let shapes = HashMap::from([("x".to_string(), vec![1, 6])]);
        let err =
            resolve_split_sizes(&spec, -1, &weights, &evaluated, &inputs, &shapes).unwrap_err();
        assert!(err.to_string().contains("to be constant"));
    }

    #[test]
    fn split_rejects_mismatched_exact_and_float_shapes() {
        let mut spec = split_spec(1, "x", vec![], vec!["a", "b"]);
        spec.attributes.remove("split");
        spec.inputs.push("sizes".to_string());
        let mut weights = WeightStore::new();
        weights.insert(
            "sizes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
        );
        weights.insert_integers(
            "sizes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![3, 3]).unwrap(),
        );
        let evaluated = HashMap::new();
        let inputs = Vec::new();
        let shapes = HashMap::from([("x".to_string(), vec![1, 6])]);
        let err =
            resolve_split_sizes(&spec, -1, &weights, &evaluated, &inputs, &shapes).unwrap_err();
        assert!(err.to_string().contains("float view"));
    }

    #[test]
    fn split_rejects_partition_sum_overflow() {
        let spec = split_spec(1, "x", vec![], vec!["a", "b"]);
        let err = checked_split_total(&spec, &[usize::MAX, 1]).unwrap_err();
        assert!(err.to_string().contains("sum overflows"));
    }

    /// A rank-1 (no batch axis) tensor `[6]` with axis=0 Split loads WITHOUT the
    /// unbatched-mode error, lowering to Slice nodes on internal axis 0; each produced
    /// Slice propagates to the correct per-output shape.
    #[test]
    fn split_axis0_rank1_genuine_data_axis_loads_and_shapes_correctly() {
        let mut tensor_shapes: HashMap<String, Vec<i64>> = HashMap::new();
        // Recorded ONNX shape is rank-1 → `data_had_batch_axis == Some(false)`.
        tensor_shapes.insert("x".to_string(), vec![6]);

        let weights = WeightStore::new();
        let evaluated_constants: HashMap<String, ArrayD<f32>> = HashMap::new();
        let constant_tensors: HashSet<String> = HashSet::new();
        let inputs: Vec<TensorSpec> = Vec::new();
        let mut graph = GraphNetwork::new();
        let mut tensor_to_node: HashMap<String, String> = HashMap::new();
        // The Split input is produced by the graph input node.
        tensor_to_node.insert("x".to_string(), INPUT_NODE_NAME.to_string());
        let mut last_added_node: Option<String> = None;

        let spec = split_spec(0, "x", vec![2, 4], vec!["a", "b"]);
        let mut split_ctx = SplitBuildContext {
            weights: &weights,
            evaluated_constants: &evaluated_constants,
            constant_tensors: &constant_tensors,
            inputs: &inputs,
            tensor_shapes: &tensor_shapes,
            graph: &mut graph,
            tensor_to_node: &mut tensor_to_node,
            last_added_node: &mut last_added_node,
        };

        let outcome = handle_split_layer(&spec, &mut split_ctx)
            .expect("axis=0 Split on a genuine rank-1 data axis must load without error");
        assert!(matches!(outcome, SplitGraphBuildOutcome::Handled));

        // Two Slice nodes, both on the trailing-relative sole data axis (-1),
        // covering [0:2) and [2:6).
        let node_a = graph
            .node("split_slice_0")
            .expect("first Split output node");
        let node_b = graph
            .node("split_slice_1")
            .expect("second Split output node");
        let Layer::Slice(slice_a) = node_a.layer() else {
            panic!(
                "expected Slice layer for output a, got {:?}",
                node_a.layer()
            );
        };
        let Layer::Slice(slice_b) = node_b.layer() else {
            panic!(
                "expected Slice layer for output b, got {:?}",
                node_b.layer()
            );
        };
        assert_eq!((slice_a.axis, slice_a.start, slice_a.end), (-1, 0, 2));
        assert_eq!((slice_b.axis, slice_b.start, slice_b.end), (-1, 2, 6));

        // Propagate a rank-1 input through each Slice: correct per-output shapes.
        let lower = ArrayD::from_shape_vec(IxDyn(&[6]), vec![0.0_f32; 6]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[6]), vec![1.0_f32; 6]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();
        let out_a = slice_a.propagate_ibp(&input).expect("slice a ibp");
        let out_b = slice_b.propagate_ibp(&input).expect("slice b ibp");
        assert_eq!(out_a.shape(), &[2]);
        assert_eq!(out_b.shape(), &[4]);
    }

    /// A rank-2 input (genuine stripped batch axis) with axis=0 Split still rejects.
    #[test]
    fn split_axis0_rank2_still_rejects_batch_axis() {
        let mut tensor_shapes: HashMap<String, Vec<i64>> = HashMap::new();
        // Recorded ONNX shape is rank-2 → `data_had_batch_axis == Some(true)`.
        tensor_shapes.insert("x".to_string(), vec![1, 6]);

        let weights = WeightStore::new();
        let evaluated_constants: HashMap<String, ArrayD<f32>> = HashMap::new();
        let constant_tensors: HashSet<String> = HashSet::new();
        let inputs: Vec<TensorSpec> = Vec::new();
        let mut graph = GraphNetwork::new();
        let mut tensor_to_node: HashMap<String, String> = HashMap::new();
        tensor_to_node.insert("x".to_string(), INPUT_NODE_NAME.to_string());
        let mut last_added_node: Option<String> = None;

        let spec = split_spec(0, "x", vec![2, 4], vec!["a", "b"]);
        let mut split_ctx = SplitBuildContext {
            weights: &weights,
            evaluated_constants: &evaluated_constants,
            constant_tensors: &constant_tensors,
            inputs: &inputs,
            tensor_shapes: &tensor_shapes,
            graph: &mut graph,
            tensor_to_node: &mut tensor_to_node,
            last_added_node: &mut last_added_node,
        };

        let err = handle_split_layer(&spec, &mut split_ctx)
            .expect_err("axis=0 Split on a rank-2 (batch) tensor must still reject");
        assert!(
            err.to_string().contains("batch dimension"),
            "expected batch-dimension error, got: {err}"
        );
    }

    #[test]
    fn split_sizes_must_cover_known_axis_exactly() {
        let mut tensor_shapes: HashMap<String, Vec<i64>> = HashMap::new();
        // Internal activation shape is [6] after stripping the ONNX batch dim.
        tensor_shapes.insert("x".to_string(), vec![1, 6]);

        let weights = WeightStore::new();
        let evaluated_constants: HashMap<String, ArrayD<f32>> = HashMap::new();
        let constant_tensors: HashSet<String> = HashSet::new();
        let inputs: Vec<TensorSpec> = Vec::new();
        let mut graph = GraphNetwork::new();
        let mut tensor_to_node: HashMap<String, String> = HashMap::new();
        tensor_to_node.insert("x".to_string(), INPUT_NODE_NAME.to_string());
        let mut last_added_node: Option<String> = None;

        let spec = split_spec(1, "x", vec![2, 3], vec!["a", "b"]);
        let mut split_ctx = SplitBuildContext {
            weights: &weights,
            evaluated_constants: &evaluated_constants,
            constant_tensors: &constant_tensors,
            inputs: &inputs,
            tensor_shapes: &tensor_shapes,
            graph: &mut graph,
            tensor_to_node: &mut tensor_to_node,
            last_added_node: &mut last_added_node,
        };

        let err = handle_split_layer(&spec, &mut split_ctx)
            .expect_err("inexact Split partition must fail");
        assert!(
            err.to_string().contains("sum to 5, expected axis length 6"),
            "expected exact-coverage error, got: {err}"
        );
        assert_eq!(graph.num_nodes(), 0, "must not emit partial Slice nodes");
    }
}
