// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::domain_list::types::DomainMetadata;
use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{checked_dim_product, NyError, Result};
use ny_tensor::{BoundedTensor, PooledArray, TensorPool};

/// Slice a batched array along the first (batch) dimension.
///
/// Takes array of shape [batch, ...rest] and returns [...rest] for given batch index.
pub(super) fn slice_batch_dim(arr: &ArrayD<f32>, batch_idx: usize) -> Option<ArrayD<f32>> {
    let shape = arr.shape();
    if shape.is_empty() || batch_idx >= shape[0] {
        return None;
    }

    Some(arr.index_axis(Axis(0), batch_idx).to_owned().into_dyn())
}

/// Stack arrays along a new batch dimension (dimension 0).
///
/// All arrays must have the same shape.
pub(super) fn stack_arrays_pooled(arrays: &[ArrayD<f32>]) -> Result<PooledArray> {
    if arrays.is_empty() {
        return Err(NyError::InvalidSpec(
            "stack_arrays_pooled called with empty input".to_string(),
        ));
    }

    let first_shape = arrays[0].shape().to_vec();
    let batch_size = arrays.len();

    // Build new shape: [batch_size, *original_shape]
    let mut new_shape = vec![batch_size];
    new_shape.extend_from_slice(&first_shape);

    // Allocate output array using pooled buffer
    let total_elements: usize = checked_dim_product(&new_shape, "stack_arrays_pooled")?;
    let mut buffer = TensorPool::acquire(total_elements);
    buffer.truncate(total_elements);
    let data = buffer.as_mut_slice();
    if data.len() != total_elements {
        return Err(NyError::InternalError(format!(
            "stack_arrays_pooled: pooled buffer length {} != expected {}",
            data.len(),
            total_elements
        )));
    }
    let mut offset = 0usize;

    // Copy data from each array
    for arr in arrays {
        if arr.shape() != first_shape.as_slice() {
            return Err(NyError::shape_mismatch(
                first_shape.clone(),
                arr.shape().to_vec(),
            ));
        }

        if let Some(slice) = arr.as_slice() {
            let end = offset + slice.len();
            if end > data.len() {
                return Err(NyError::InternalError(format!(
                    "stack_arrays_pooled: copy end {} exceeds buffer length {}",
                    end,
                    data.len()
                )));
            }
            data[offset..end].copy_from_slice(slice);
            offset = end;
        } else {
            for v in arr.iter() {
                data[offset] = *v;
                offset += 1;
            }
        }
    }

    if offset != total_elements {
        return Err(NyError::InternalError(format!(
            "stack_arrays_pooled: copied {} elements but expected {}",
            offset, total_elements
        )));
    }

    PooledArray::try_from_pooled_buffer(buffer, &new_shape)
}

/// Reject non-finite (NaN/Inf) values in global bounds.
///
/// Returns `Err(NumericalInstability)` if any lower or upper bound is NaN or infinite.
/// This guard prevents corrupted values from reaching sort ordering, branch selection,
/// and stability classification (#2246).
pub(super) fn validate_global_bounds_finite(
    lower_bounds: &[f32],
    upper_bounds: &[f32],
    context: &str,
) -> Result<()> {
    for (i, (lb, ub)) in lower_bounds.iter().zip(upper_bounds).enumerate() {
        if !lb.is_finite() || !ub.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "{context} domain {i} has non-finite global bounds (lb={lb}, ub={ub})"
            )));
        }
    }
    Ok(())
}

/// Reject non-finite (NaN/Inf) values in a batched tensor.
///
/// Treats axis 0 as the batch dimension. When `keep_mask` is `Some`, only
/// rows where `keep_mask[i] == true` are validated — dropped rows may contain
/// non-finite values without triggering an error (#3115).
///
/// Returns `Err(NumericalInstability)` if any validated element is NaN or infinite.
pub(super) fn validate_batched_tensor_finite(
    tensor: &ArrayD<f32>,
    context: &str,
    field_name: &str,
    keep_mask: Option<&[bool]>,
) -> Result<()> {
    let batch_size = tensor.shape().first().copied().unwrap_or(0);
    for batch_idx in 0..batch_size {
        if let Some(mask) = keep_mask {
            if batch_idx < mask.len() && !mask[batch_idx] {
                continue;
            }
        }
        let row = tensor.index_axis(Axis(0), batch_idx);
        for (elem_idx, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "{context} {field_name} domain {batch_idx} element {elem_idx} \
                     has non-finite value ({val})"
                )));
            }
        }
    }
    Ok(())
}

/// Reject non-finite (NaN/Inf) values in named per-layer batched tensors.
///
/// Calls [`validate_batched_tensor_finite`] for every entry in the map,
/// encoding the layer name and `bound_kind` (e.g. "lower"/"upper") in the
/// field name for diagnostics (#3115).
pub(super) fn validate_named_batched_tensors_finite(
    tensors: &std::collections::HashMap<String, ArrayD<f32>>,
    context: &str,
    bound_kind: &str,
    keep_mask: Option<&[bool]>,
) -> Result<()> {
    for (layer_name, tensor) in tensors {
        let field = format!("{bound_kind}[{layer_name}]");
        validate_batched_tensor_finite(tensor, context, &field, keep_mask)?;
    }
    Ok(())
}

fn validate_single_bounded_tensor_finite(
    tensor: &BoundedTensor,
    context: &str,
    field_name: &str,
) -> Result<()> {
    for (bound_kind, bound) in [("lower", tensor.lower()), ("upper", tensor.upper())] {
        for (elem_idx, &val) in bound.iter().enumerate() {
            if !val.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "{context} {field_name}.{bound_kind} element {elem_idx} \
                     has non-finite value ({val})"
                )));
            }
        }
    }
    Ok(())
}

/// Reject non-finite (NaN/Inf) values in a deferred node-bounds override map.
pub(super) fn validate_node_bounds_override_finite(
    node_bounds_override: &std::collections::HashMap<String, BoundedTensor>,
    context: &str,
) -> Result<()> {
    for (node_name, bounds) in node_bounds_override {
        let field_name = format!("node_bounds_override[{node_name}]");
        validate_single_bounded_tensor_finite(bounds, context, &field_name)?;
    }
    Ok(())
}

/// Reject non-finite (NaN/Inf) values in queued per-domain node-bounds overrides.
///
/// When `keep_mask` is `Some`, only kept metadata entries are validated so dropped
/// rows can still carry debugging/corrupted data without blocking `add()` (#3115).
pub(super) fn validate_metadata_node_bounds_overrides_finite(
    metadata: &[DomainMetadata],
    context: &str,
    keep_mask: Option<&[bool]>,
) -> Result<()> {
    for (domain_idx, meta) in metadata.iter().enumerate() {
        if let Some(mask) = keep_mask {
            if domain_idx < mask.len() && !mask[domain_idx] {
                continue;
            }
        }
        if let Some(node_bounds_override) = meta.node_bounds_override() {
            let domain_context = format!("{context} domain {domain_idx}");
            validate_node_bounds_override_finite(node_bounds_override, &domain_context)?;
        }
    }
    Ok(())
}

pub(super) fn validate_pick_out_metadata_finite(metadata: &[DomainMetadata]) -> Result<()> {
    validate_metadata_node_bounds_overrides_finite(metadata, "DomainList::pick_out", None)
}

pub(super) fn validate_add_metadata_finite(
    metadata: &[DomainMetadata],
    keep_mask: &[bool],
) -> Result<()> {
    validate_metadata_node_bounds_overrides_finite(metadata, "DomainList::add", Some(keep_mask))
}

pub(super) fn unstable_mask_from_bounds(
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
) -> Result<ArrayD<bool>> {
    if lower.shape() != upper.shape() {
        return Err(NyError::shape_mismatch(
            lower.shape().to_vec(),
            upper.shape().to_vec(),
        ));
    }

    let shape = lower.shape().to_vec();
    let mut data = Vec::with_capacity(lower.len());
    for (l, u) in lower.iter().zip(upper.iter()) {
        // NaN bounds are conservatively classified as unstable — without this,
        // both NaN < 0 and NaN > 0 return false, causing NaN-bounded neurons
        // to appear stable and preventing necessary branching (#2246).
        data.push((*l < 0.0 && *u > 0.0) || l.is_nan() || u.is_nan());
    }

    ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|_| {
        NyError::InvalidSpec("unstable_mask_from_bounds failed to build mask".to_string())
    })
}
