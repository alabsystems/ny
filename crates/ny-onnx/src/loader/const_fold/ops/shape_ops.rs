// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use tracing::{debug, warn};

use super::super::common::{
    normalize_axis, parse_attribute_or_input_ints, read_tensor_i64s, read_tensor_i64s_and_shape,
    reshape_allowzero, reshape_with_warning,
};
use super::super::shape_inference::ConstFoldLookups;
use super::super::FoldedTensor;
use super::cast::try_fold_cast;
use super::slice::{try_fold_slice, try_fold_slice_integer};

/// Read a scalar INT attribute (e.g. ONNX `Shape` `start`/`end`).
fn read_int_attribute(node: &onnx_proto::NodeProto, name: &str) -> Option<i64> {
    use crate::onnx_proto::attribute_type;
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .and_then(|attr| match attr.r#type {
            attribute_type::INT => Some(attr.i_value()),
            _ => None,
        })
}

pub(super) fn try_fold_shape_node(
    node: &onnx_proto::NodeProto,
    graph: &onnx_proto::GraphProto,
    lookups: &ConstFoldLookups,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    if node.op_type != "Shape" || node.input.is_empty() {
        return None;
    }

    let input_name = &node.input[0];
    let shape = lookups
        .infer_tensor_shape(input_name, graph, weights, 8)
        .or_else(|| {
            weights
                .get(input_name)
                .map(|weight| weight.shape().iter().map(|&d| d as i64).collect::<Vec<_>>())
        })?;

    // ONNX Shape (opset >= 15) supports optional `start`/`end` attributes that
    // slice the reported shape vector: negative values are rank-relative, the
    // range is clamped to [0, rank]. Ignoring them folds the WRONG constant
    // (e.g. a batch-extracting `Shape(start=0, end=1)` would report the full
    // shape), which then poisons every downstream Concat/Reshape fold.
    let rank = shape.len() as i64;
    let clamp_axis = |value: i64| -> i64 {
        let adjusted = if value < 0 { value + rank } else { value };
        adjusted.clamp(0, rank)
    };
    let start = clamp_axis(read_int_attribute(node, "start").unwrap_or(0)) as usize;
    let end = clamp_axis(read_int_attribute(node, "end").unwrap_or(rank)) as usize;
    // Shape-15 defines start > end as a valid empty one-dimensional result,
    // not an invalid range.  Preserve that distinction: refusing the fold can
    // strand an INT64 shape value on ny's FLOAT-only runtime path, while
    // clamping/reordering the endpoints would invent dimensions.
    let shape = if start <= end {
        &shape[start..end]
    } else {
        &shape[0..0]
    };

    // Encode symbolic dimensions as copy-axis sentinels. A literal -1 means
    // "infer exactly one dimension", and ONNX's public 0 sentinel only copies
    // the dimension at the same target index, so neither can represent a
    // Shape/Gather value that is moved to a different reshape target index.
    // Sentinels keep the ORIGINAL axis index (offset by `start`), since they
    // name the source axis to copy from, not the output position.
    let shape_dims: Vec<i64> = shape
        .iter()
        .enumerate()
        .map(|(offset, &dim)| {
            let axis = start + offset;
            if dim <= 0 {
                ny_core::reshape_copy_axis_sentinel(axis)
            } else {
                Some(dim)
            }
        })
        .collect::<Option<_>>()?;
    let values: Vec<f32> = shape_dims
        .iter()
        .map(|&dim| {
            if ny_core::reshape_copy_axis_from_sentinel(dim).is_some() {
                return 0.0;
            }
            crate::loader::numeric_cast::i64_to_f32_checked(dim, "Shape constant fold")
                .unwrap_or_else(|_| {
                    // #2360: Keep the legacy float view for compatibility, but preserve the
                    // exact payload in WeightStore::integers.
                    crate::loader::numeric_cast::i64_to_f32_warned(dim, "Shape constant fold")
                })
        })
        .collect();
    let float_data = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).ok()?;
    let integer_data = ArrayD::from_shape_vec(IxDyn(&[shape_dims.len()]), shape_dims).ok()?;
    debug!(
        "Constant folded Shape: {} -> {:?} (unbatched, from graph shape inference)",
        node.output.first()?,
        float_data
    );
    Some(FoldedTensor {
        float_data,
        integer_data: Some(integer_data),
        // Shape always produces INT64, irrespective of the activation dtype.
        // Preserve that dtype provenance so downstream shape arithmetic can
        // use exact checked i64 semantics rather than the lossy f32 view.
        integer_range: Some((i64::MIN, i64::MAX)),
    })
}

pub(super) fn try_fold(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    model_unbatched: bool,
) -> Option<FoldedTensor> {
    match node.op_type.as_str() {
        "Expand" if node.input.len() >= 2 => try_fold_expand(node, weights, model_unbatched),
        "Gather" if node.input.len() >= 2 => try_fold_gather_node(node, weights),
        "Squeeze" if !node.input.is_empty() => try_fold_squeeze(node, weights),
        "Unsqueeze" if !node.input.is_empty() => try_fold_unsqueeze(node, weights),
        "Reshape" if node.input.len() >= 2 => try_fold_reshape(node, weights),
        "Transpose" if !node.input.is_empty() => try_fold_transpose(node, weights),
        "Concat" if !node.input.is_empty() => try_fold_concat(node, weights),
        "Cast" if !node.input.is_empty() => try_fold_cast(node, weights),
        "Slice" if !node.input.is_empty() => try_fold_slice_node(node, weights),
        _ => None,
    }
}

fn try_fold_slice_node(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    let float_data = try_fold_slice(node, weights)?;
    let integer_data = try_fold_slice_integer(node, weights);
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: weights.get_integer_range(&node.input[0]),
    })
}

fn try_fold_expand(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    model_unbatched: bool,
) -> Option<FoldedTensor> {
    let data = weights.get(&node.input[0])?;
    // Parse as i64 to handle ONNX's -1 convention (meaning "use data dimension"),
    // which appears in ml4acopf models through ConstantOfShape→Mul→Equal→Where chains.
    let target_i64 = read_tensor_i64s(weights, &node.input[1])?;
    let float_data = expand_array(data, &target_i64, model_unbatched)?;
    let integer_data = weights
        .get_integers(&node.input[0])
        .and_then(|integer_data| expand_array(integer_data, &target_i64, model_unbatched));
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: weights.get_integer_range(&node.input[0]),
    })
}

fn expand_array<T: Clone>(
    data: &ArrayD<T>,
    target_i64: &[i64],
    model_unbatched: bool,
) -> Option<ArrayD<T>> {
    let data_shape = data.shape();
    let out_ndim = data_shape.len().max(target_i64.len());
    let mut out_shape = vec![1usize; out_ndim];

    for (i, dim) in out_shape.iter_mut().enumerate().rev() {
        let data_idx = i as isize - (out_ndim as isize - data_shape.len() as isize);
        let target_idx = i as isize - (out_ndim as isize - target_i64.len() as isize);
        let data_dim = if data_idx >= 0 {
            data_shape[data_idx as usize]
        } else {
            1
        };
        let target_dim = if target_idx >= 0 {
            let t = target_i64[target_idx as usize];
            if t == -1 {
                // ONNX convention: -1 means "copy from data dimension"
                data_dim
            } else if t <= 0 {
                return None;
            } else {
                t as usize
            }
        } else {
            1
        };
        *dim = data_dim.max(target_dim);
    }

    const MAX_EXPAND_ELEMENTS: usize = 10_000_000;
    let total = ny_core::checked_shape_product(&out_shape).unwrap_or(usize::MAX);
    if total > MAX_EXPAND_ELEMENTS {
        warn!(
            "Expand: refusing to allocate {} elements (shape {:?}, limit {})",
            total, out_shape, MAX_EXPAND_ELEMENTS
        );
        return None;
    }

    data.broadcast(IxDyn(&out_shape)).map(|view| {
        // Unbatched mode removes only the synthetic leading batch axis. Keep
        // later singleton axes so constant data paths preserve their real rank
        // (for example ViT's CLS token stays [1, H] instead of collapsing to
        // [H]). Globally-unbatched models (#cctsdb B5) have no batch axis at
        // all: keep the broadcast shape verbatim.
        let mut result = view.to_owned();
        if !model_unbatched && result.ndim() > 1 && result.shape()[0] == 1 {
            result = result.index_axis_move(Axis(0), 0);
        }
        result
    })
}

fn try_fold_gather_node(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    let data = weights.get(&node.input[0])?;
    let (indices, indices_shape) = read_tensor_i64s_and_shape(weights, &node.input[1])?;
    let axis_attr = node
        .attribute
        .iter()
        .find(|attr| attr.name == "axis")
        .map(|attr| attr.i_value())
        .unwrap_or(0);
    let axis = normalize_axis(axis_attr, data.ndim())?;
    let float_data = const_fold_gather(data, &indices, &indices_shape, axis)?;
    let integer_data = weights
        .get_integers(&node.input[0])
        .and_then(|integer_data| const_fold_gather(integer_data, &indices, &indices_shape, axis));
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: weights.get_integer_range(&node.input[0]),
    })
}

fn const_fold_gather<T: Clone>(
    data: &ArrayD<T>,
    indices: &[i64],
    indices_shape: &[usize],
    axis: usize,
) -> Option<ArrayD<T>> {
    let data_shape = data.shape();
    let axis_len = data_shape.get(axis).copied()? as i64;
    let normalized: Vec<usize> = indices
        .iter()
        .map(|&index| {
            let adjusted = if index < 0 { index + axis_len } else { index };
            if adjusted >= 0 && adjusted < axis_len {
                Some(adjusted as usize)
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;

    let selected = data.select(Axis(axis), &normalized);
    let mut output_shape =
        Vec::with_capacity(data_shape.len().saturating_sub(1) + indices_shape.len());
    output_shape.extend_from_slice(&data_shape[..axis]);
    output_shape.extend_from_slice(indices_shape);
    output_shape.extend_from_slice(&data_shape[axis + 1..]);

    if indices_shape.is_empty() && normalized.len() == 1 {
        return Some(data.index_axis(Axis(axis), normalized[0]).to_owned());
    }

    selected
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(IxDyn(&output_shape))
        .ok()
}

fn try_fold_squeeze(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<FoldedTensor> {
    let data = weights.get(&node.input[0])?;
    let ndim = data.ndim();

    // Opset 13+: axes from second input; opset < 13: axes from attribute.
    // If no axes specified, squeeze all dimensions of size 1.
    let axes = parse_attribute_or_input_ints(node, "axes", 1, weights);

    let resolved = if let Some(axes) = axes {
        let mut resolved = Vec::with_capacity(axes.len());
        for &axis in &axes {
            let axis = if axis < 0 {
                (ndim as i64)
                    .checked_add(axis)
                    .and_then(|v| usize::try_from(v).ok())?
            } else {
                usize::try_from(axis).ok()?
            };
            if axis >= ndim {
                return None;
            }
            if data.shape()[axis] != 1 {
                return None;
            }
            resolved.push(axis);
        }
        resolved
    } else {
        // No axes: squeeze all dimensions of size 1
        (0..ndim).filter(|&i| data.shape()[i] == 1).collect()
    };

    let mut shape: Vec<usize> = data.shape().to_vec();
    // Remove axes in reverse order to preserve indices.
    let mut sorted = resolved;
    sorted.sort_unstable();
    sorted.dedup();
    for &axis in sorted.iter().rev() {
        shape.remove(axis);
    }
    let float_data = reshape_with_warning(data.clone(), &shape, "Squeeze")?;
    let integer_data = weights
        .get_integers(&node.input[0])
        .cloned()
        .and_then(|integer_data| reshape_with_warning(integer_data, &shape, "Squeeze"));
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: weights.get_integer_range(&node.input[0]),
    })
}

fn try_fold_unsqueeze(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<FoldedTensor> {
    let data = weights.get(&node.input[0])?;
    let axes = parse_attribute_or_input_ints(node, "axes", 1, weights)?;
    let output_rank = data.ndim() + axes.len();
    let mut resolved = Vec::with_capacity(axes.len());
    let mut valid = true;

    for &axis in &axes {
        let axis = if axis < 0 {
            match (output_rank as i64).checked_add(axis) {
                Some(value) if value >= 0 => value as usize,
                _ => {
                    valid = false;
                    break;
                }
            }
        } else {
            match usize::try_from(axis) {
                Ok(value) => value,
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        };
        if axis >= output_rank {
            valid = false;
            break;
        }
        resolved.push(axis);
    }

    resolved.sort_unstable();
    if resolved.windows(2).any(|window| window[0] == window[1]) {
        valid = false;
    }

    let mut shape = data.shape().to_vec();
    for &axis in &resolved {
        if axis > shape.len() {
            valid = false;
            break;
        }
        shape.insert(axis, 1);
    }

    if !valid {
        return None;
    }
    let float_data = reshape_with_warning(data.clone(), &shape, "Unsqueeze")?;
    let integer_data = weights
        .get_integers(&node.input[0])
        .cloned()
        .and_then(|integer_data| reshape_with_warning(integer_data, &shape, "Unsqueeze"));
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: weights.get_integer_range(&node.input[0]),
    })
}

fn try_fold_reshape(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<FoldedTensor> {
    let data = weights.get(&node.input[0])?;
    let shape_i64 = read_tensor_i64s(weights, &node.input[1])?;
    let plan = build_reshape_plan(&shape_i64, data.shape(), reshape_allowzero(node))?;
    let shape = finalize_reshape_shape(plan, data.len())?;
    let float_data = reshape_with_warning(data.clone(), &shape, "Reshape")?;
    let integer_data = weights
        .get_integers(&node.input[0])
        .cloned()
        .and_then(|integer_data| reshape_with_warning(integer_data, &shape, "Reshape"));
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: weights.get_integer_range(&node.input[0]),
    })
}

fn try_fold_transpose(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<FoldedTensor> {
    let data = weights.get(&node.input[0])?;
    let ndim = data.ndim();
    let perm = node
        .attribute
        .iter()
        .find(|attr| attr.name == "perm")
        .map(|attr| {
            attr.ints
                .iter()
                .map(|&value| usize::try_from(value).ok().filter(|&axis| axis < ndim))
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| Some((0..ndim).rev().collect()))?;
    // Materialize a standard-layout array so a downstream constant Reshape can
    // consume the transposed tensor without tripping ndarray's layout checks.
    let float_data = data
        .view()
        .permuted_axes(perm.clone())
        .as_standard_layout()
        .into_owned();
    let integer_data = weights.get_integers(&node.input[0]).map(|integer_data| {
        integer_data
            .view()
            .permuted_axes(perm.clone())
            .as_standard_layout()
            .into_owned()
    });
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: weights.get_integer_range(&node.input[0]),
    })
}

fn try_fold_concat(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<FoldedTensor> {
    let axis_attr = node
        .attribute
        .iter()
        .find(|attr| attr.name == "axis")
        .map(|attr| attr.i_value())
        .unwrap_or(0);
    let arrays: Vec<&ArrayD<f32>> = node
        .input
        .iter()
        .filter(|name| !name.is_empty())
        .map(|name| weights.get(name))
        .collect::<Option<Vec<_>>>()?;
    if arrays.is_empty() {
        return None;
    }

    let ndim = arrays[0].ndim().max(1);
    let axis = if axis_attr < 0 {
        (ndim as i64 + axis_attr) as usize
    } else {
        axis_attr as usize
    };
    let float_data = concat_promoted(arrays, axis)?;
    let integer_arrays: Option<Vec<&ArrayD<i64>>> = node
        .input
        .iter()
        .filter(|name| !name.is_empty())
        .map(|name| weights.get_integers(name))
        .collect();
    let integer_data =
        integer_arrays.and_then(|integer_arrays| concat_promoted(integer_arrays, axis));
    let integer_range = integer_data.as_ref().and_then(|_| {
        let mut ranges = node
            .input
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| weights.get_integer_range(name));
        let first = ranges.next().flatten()?;
        ranges.all(|range| range == Some(first)).then_some(first)
    });
    Some(FoldedTensor {
        float_data,
        integer_data,
        // Concat is dtype-preserving.  Retain authenticated integer provenance
        // only when every input carries the same authored integer range.
        integer_range,
    })
}

fn concat_promoted<T: Clone>(arrays: Vec<&ArrayD<T>>, axis: usize) -> Option<ArrayD<T>> {
    let promoted: Vec<ArrayD<T>> = arrays
        .iter()
        .map(|array| {
            if array.ndim() == 0 {
                (*array).clone().insert_axis(Axis(0))
            } else {
                (*array).clone()
            }
        })
        .collect();
    let views: Vec<ArrayViewD<'_, T>> = promoted.iter().map(|array| array.view()).collect();
    ndarray::concatenate(Axis(axis), &views).ok()
}

struct ReshapePlan {
    shape: Vec<usize>,
    inferred_index: Option<usize>,
    known_prod: usize,
}

fn build_reshape_plan(
    shape_i64: &[i64],
    input_shape: &[usize],
    allowzero: bool,
) -> Option<ReshapePlan> {
    let mut shape = Vec::with_capacity(shape_i64.len());
    let mut inferred_index = None;
    let mut known_prod = 1usize;

    for (idx, dim) in shape_i64.iter().enumerate() {
        match *dim {
            -1 => {
                if inferred_index.is_some() {
                    return None;
                }
                inferred_index = Some(idx);
                shape.push(0);
            }
            0 => {
                let resolved_dim = if allowzero { 0 } else { *input_shape.get(idx)? };
                known_prod = known_prod.checked_mul(resolved_dim)?;
                shape.push(resolved_dim);
            }
            positive if positive > 0 => {
                let resolved_dim = usize::try_from(positive).ok()?;
                known_prod = known_prod.checked_mul(resolved_dim)?;
                shape.push(resolved_dim);
            }
            _ => return None,
        }
    }

    Some(ReshapePlan {
        shape,
        inferred_index,
        known_prod,
    })
}

fn finalize_reshape_shape(mut plan: ReshapePlan, total_elems: usize) -> Option<Vec<usize>> {
    if let Some(inferred_index) = plan.inferred_index {
        if plan.known_prod == 0 {
            if total_elems != 0 {
                return None;
            }
            plan.shape[inferred_index] = 0;
            return Some(plan.shape);
        }
        if !total_elems.is_multiple_of(plan.known_prod) {
            return None;
        }
        plan.shape[inferred_index] = total_elems / plan.known_prod;
        return Some(plan.shape);
    }

    (plan.known_prod == total_elems).then_some(plan.shape)
}
