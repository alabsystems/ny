// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{ArrayD, Axis, IxDyn};

use super::super::common::{parse_scalar_i64, read_tensor_i64s};

fn fits_exact_i64_f32(value: f32) -> bool {
    value >= i64::MIN as f32 && value < i64::MAX as f32
}

fn parse_slice_scalar_i64(value: f32, allow_positive_infinity: bool) -> Option<i64> {
    if value.is_nan() {
        return None;
    }
    if value.is_infinite() {
        if allow_positive_infinity && value.is_sign_positive() {
            return Some(i64::MAX);
        }
        return None;
    }
    parse_scalar_i64(value).or_else(|| {
        let truncated = value.trunc();
        (truncated.is_finite() && fits_exact_i64_f32(truncated)).then_some(truncated as i64)
    })
}

pub(super) fn try_fold_slice(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<ArrayD<f32>> {
    let data = weights.get(&node.input[0])?;
    let args = parse_slice_args(node, weights)?;
    let slice_ops = resolve_slice_ops(&args, data.ndim())?;
    apply_slice_ops(data.clone(), slice_ops)
}

pub(super) fn try_fold_slice_integer(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<ArrayD<i64>> {
    let data = weights.get_integers(&node.input[0])?;
    let args = parse_slice_args(node, weights)?;
    let slice_ops = resolve_slice_ops(&args, data.ndim())?;
    apply_slice_ops(data.clone(), slice_ops)
}

struct SliceArgs {
    starts: Vec<i64>,
    ends: Vec<i64>,
    axes: Vec<i64>,
    steps: Vec<i64>,
}

type SliceOp = (usize, i64, i64, i64);

fn parse_slice_args(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<SliceArgs> {
    parse_slice_args_from_inputs(node, weights).or_else(|| parse_slice_args_from_attributes(node))
}

fn parse_slice_args_from_inputs(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<SliceArgs> {
    let starts_name = node.input.get(1)?;
    let ends_name = node.input.get(2)?;
    let axes_name = node.input.get(3).filter(|name| !name.is_empty());
    let steps_name = node.input.get(4).filter(|name| !name.is_empty());

    let starts = read_slice_input_i64s(weights, starts_name, false)?;
    let ends = read_slice_input_i64s(weights, ends_name, true)?;
    let axes: Vec<i64> = match axes_name {
        Some(name) => read_slice_input_i64s(weights, name, false)?,
        None => (0..starts.len() as i64).collect(),
    };
    let steps: Vec<i64> = match steps_name {
        Some(name) => read_slice_input_i64s(weights, name, false)?,
        None => vec![1; starts.len()],
    };

    if steps.contains(&0)
        || starts.len() != ends.len()
        || starts.len() != axes.len()
        || starts.len() != steps.len()
    {
        return None;
    }

    Some(SliceArgs {
        starts,
        ends,
        axes,
        steps,
    })
}

fn read_slice_input_i64s(
    weights: &WeightStore,
    name: &str,
    allow_positive_infinity: bool,
) -> Option<Vec<i64>> {
    read_tensor_i64s(weights, name).or_else(|| {
        weights.get(name).and_then(|array| {
            array
                .iter()
                .map(|&value| parse_slice_scalar_i64(value, allow_positive_infinity))
                .collect::<Option<Vec<_>>>()
        })
    })
}

fn parse_slice_args_from_attributes(node: &onnx_proto::NodeProto) -> Option<SliceArgs> {
    let mut starts = None;
    let mut ends = None;
    let mut axes = None;
    let mut steps = None;

    for attr in &node.attribute {
        match attr.name.as_str() {
            "starts" => starts = Some(attr.ints.clone()),
            "ends" => ends = Some(attr.ints.clone()),
            "axes" => axes = Some(attr.ints.clone()),
            "steps" => steps = Some(attr.ints.clone()),
            _ => {}
        }
    }

    let starts = starts?;
    let ends = ends?;
    let axes = axes.unwrap_or_else(|| (0..starts.len() as i64).collect());
    let steps = steps.unwrap_or_else(|| vec![1; starts.len()]);

    if steps.contains(&0)
        || starts.len() != ends.len()
        || starts.len() != axes.len()
        || starts.len() != steps.len()
    {
        return None;
    }

    Some(SliceArgs {
        starts,
        ends,
        axes,
        steps,
    })
}

fn resolve_slice_ops(args: &SliceArgs, ndim: usize) -> Option<Vec<SliceOp>> {
    let mut slice_ops = Vec::with_capacity(args.axes.len());
    for (i, &axis_raw) in args.axes.iter().enumerate() {
        let axis = if axis_raw < 0 {
            match (ndim as i64).checked_add(axis_raw) {
                Some(value) if value >= 0 => value as usize,
                _ => return None,
            }
        } else {
            axis_raw as usize
        };
        if axis >= ndim {
            return None;
        }
        slice_ops.push((axis, args.starts[i], args.ends[i], args.steps[i]));
    }
    slice_ops.sort_by_key(|op| std::cmp::Reverse(op.0));
    Some(slice_ops)
}

fn apply_slice_ops<T: Clone>(mut result: ArrayD<T>, slice_ops: Vec<SliceOp>) -> Option<ArrayD<T>> {
    for (axis, start_raw, end_raw, step) in slice_ops {
        let indices = match slice_indices(start_raw, end_raw, step, result.shape()[axis] as i64) {
            Some(indices) => indices,
            None => return Some(result),
        };
        if indices.is_empty() {
            let mut shape = result.shape().to_vec();
            shape[axis] = 0;
            result = ArrayD::from_shape_vec(IxDyn(&shape), Vec::new()).ok()?;
        } else {
            result = result.select(Axis(axis), &indices);
        }
    }
    Some(result)
}

fn slice_indices(start_raw: i64, end_raw: i64, step: i64, dim: i64) -> Option<Vec<usize>> {
    if dim < 0 || step == 0 {
        return None;
    }
    let dim_usize = usize::try_from(dim).ok()?;
    if dim_usize == 0 {
        return Some(Vec::new());
    }

    let (start, end) = normalize_slice_bounds(start_raw, end_raw, step, dim);
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

fn normalize_slice_bounds(start_raw: i64, end_raw: i64, step: i64, dim: i64) -> (i64, i64) {
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

#[cfg(test)]
mod tests {
    use super::try_fold_slice;
    use crate::onnx_proto::NodeProto;
    use crate::WeightStore;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn try_fold_slice_prefers_integer_store_end_2360() {
        let mut weights = WeightStore::new();
        weights.insert(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap(),
        );
        weights.insert(
            "starts".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        );
        weights.insert(
            "ends".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![i64::MAX as f32]).unwrap(),
        );
        weights.insert(
            "axes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        );
        weights.insert_integers(
            "ends".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![i64::MAX]).unwrap(),
        );

        let node = NodeProto {
            input: vec![
                "data".to_string(),
                "starts".to_string(),
                "ends".to_string(),
                "axes".to_string(),
            ],
            op_type: "Slice".to_string(),
            ..Default::default()
        };

        let out = try_fold_slice(&node, &weights).expect("Slice should use exact integer end");
        assert_eq!(out.iter().copied().collect::<Vec<_>>(), vec![20.0, 30.0]);
    }
}
