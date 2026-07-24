// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::LayerSpec;

use super::super::CostError;
use super::common::{normalize_indices, shape_inference_error};
use super::ShapeLookup;
use std::collections::BTreeSet;

#[derive(Debug)]
pub(super) struct SliceArgs {
    pub(super) starts: Vec<i64>,
    pub(super) ends: Vec<i64>,
    pub(super) axes: Vec<i64>,
    pub(super) steps: Vec<i64>,
}

pub(super) fn infer_slice_shape(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    input_shape: &[usize],
) -> Result<Vec<usize>, CostError> {
    let args = parse_args(lookup, layer)?;
    let axes = normalize_indices(layer, input_shape.len(), &args.axes, "axes")?;
    if axes.iter().collect::<BTreeSet<_>>().len() != axes.len() {
        return Err(shape_inference_error(
            layer,
            format!("slice axes must be unique, got {:?}", args.axes),
        ));
    }

    let mut output_shape = input_shape.to_vec();
    for ((axis, start), (end, step)) in axes
        .into_iter()
        .zip(args.starts)
        .zip(args.ends.into_iter().zip(args.steps))
    {
        output_shape[axis] = output_dim(start, end, step, input_shape[axis] as i64, layer, axis)?;
    }
    Ok(output_shape)
}

pub(super) fn parse_scalar_i64(value: f32, allow_positive_infinity: bool) -> Option<i64> {
    if value.is_nan() {
        return None;
    }
    if value.is_infinite() {
        return if allow_positive_infinity && value.is_sign_positive() {
            Some(i64::MAX)
        } else {
            None
        };
    }

    let truncated = value.trunc();
    (truncated.is_finite() && truncated >= i64::MIN as f32 && truncated <= i64::MAX as f32)
        .then_some(truncated as i64)
}

fn parse_args(lookup: &ShapeLookup<'_>, layer: &LayerSpec) -> Result<SliceArgs, CostError> {
    let starts = lookup.read_i64_tensor(layer, 1, false, "slice starts")?;
    let ends = lookup.read_i64_tensor(layer, 2, true, "slice ends")?;
    let axes = match layer.inputs.get(3).filter(|name| !name.is_empty()) {
        Some(_) => lookup.read_i64_tensor(layer, 3, false, "slice axes")?,
        None => (0..starts.len() as i64).collect(),
    };
    let steps = match layer.inputs.get(4).filter(|name| !name.is_empty()) {
        Some(_) => lookup.read_i64_tensor(layer, 4, false, "slice steps")?,
        None => vec![1; starts.len()],
    };

    if steps.contains(&0)
        || starts.len() != ends.len()
        || starts.len() != axes.len()
        || starts.len() != steps.len()
    {
        return Err(shape_inference_error(
            layer,
            format!(
                "slice parameter lengths must match and steps must be non-zero, got starts={} ends={} axes={} steps={}",
                starts.len(),
                ends.len(),
                axes.len(),
                steps.len()
            ),
        ));
    }

    Ok(SliceArgs {
        starts,
        ends,
        axes,
        steps,
    })
}

fn output_dim(
    start_raw: i64,
    end_raw: i64,
    step: i64,
    dim: i64,
    layer: &LayerSpec,
    axis: usize,
) -> Result<usize, CostError> {
    let indices = slice_indices(start_raw, end_raw, step, dim).ok_or_else(|| {
        shape_inference_error(
            layer,
            format!(
                "slice bounds start={start_raw} end={end_raw} step={step} are invalid for axis {axis} of size {dim}"
            ),
        )
    })?;
    Ok(indices.len())
}

fn slice_indices(start_raw: i64, end_raw: i64, step: i64, dim: i64) -> Option<Vec<usize>> {
    if dim < 0 || step == 0 {
        return None;
    }
    let dim_usize = usize::try_from(dim).ok()?;
    if dim_usize == 0 {
        return Some(Vec::new());
    }

    let (start, end) = normalize_bounds(start_raw, end_raw, step, dim);
    let mut indices = Vec::new();

    if step > 0 {
        let mut index = start;
        while index < end {
            indices.push(usize::try_from(index).ok()?);
            index = index.checked_add(step)?;
        }
    } else {
        let mut index = start;
        while index > end {
            indices.push(usize::try_from(index).ok()?);
            index = index.checked_add(step)?;
        }
    }

    Some(indices)
}

fn normalize_bounds(start_raw: i64, end_raw: i64, step: i64, dim: i64) -> (i64, i64) {
    if step > 0 {
        (
            normalize_positive_bound(start_raw, dim),
            normalize_positive_bound(end_raw, dim),
        )
    } else {
        (
            normalize_negative_bound(start_raw, dim),
            normalize_negative_bound(end_raw, dim),
        )
    }
}

fn normalize_positive_bound(bound: i64, dim: i64) -> i64 {
    let translated = if bound < 0 {
        dim.saturating_add(bound)
    } else {
        bound
    };
    translated.clamp(0, dim)
}

fn normalize_negative_bound(bound: i64, dim: i64) -> i64 {
    let translated = if bound < 0 {
        dim.saturating_add(bound)
    } else {
        bound
    };
    translated.clamp(-1, dim - 1)
}
