// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{arr0, ArrayD, Ix1, Ix2};

use super::super::broadcast::{broadcast_binop, broadcast_where};

pub(super) fn try_fold(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<ArrayD<f32>> {
    match node.op_type.as_str() {
        "Pow" if node.input.len() >= 2 => {
            let (base, exponent) = binary_inputs(node, weights)?;
            if exponent.len() != 1 {
                return None;
            }
            let exponent = exponent.iter().next().copied().unwrap_or(1.0);
            finite_only(Some(base.mapv(|value| value.powf(exponent))))
        }
        "Sqrt" if !node.input.is_empty() => finite_only(unary(node, weights, |value| value.sqrt())),
        "Div" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            finite_only(broadcast_binop(lhs, rhs, |x, y| x / y))
        }
        "Mul" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, |x, y| x * y)
        }
        "Add" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, |x, y| x + y)
        }
        "Sub" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, |x, y| x - y)
        }
        "Neg" if !node.input.is_empty() => unary(node, weights, |value| -value),
        "Sin" if !node.input.is_empty() => finite_only(unary(node, weights, |value| value.sin())),
        "Cos" if !node.input.is_empty() => finite_only(unary(node, weights, |value| value.cos())),
        "Abs" if !node.input.is_empty() => unary(node, weights, |value| value.abs()),
        "Relu" if !node.input.is_empty() => unary(node, weights, |value| value.max(0.0)),
        "Sigmoid" if !node.input.is_empty() => {
            finite_only(unary(node, weights, |value| 1.0 / (1.0 + (-value).exp())))
        }
        "Tanh" if !node.input.is_empty() => finite_only(unary(node, weights, |value| value.tanh())),
        "Exp" if !node.input.is_empty() => finite_only(unary(node, weights, |value| value.exp())),
        "Log" if !node.input.is_empty() => finite_only(unary(node, weights, |value| value.ln())),
        "MatMul" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            try_fold_matmul(lhs, rhs)
        }
        "Equal" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_eq)
        }
        "Greater" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_gt)
        }
        "GreaterOrEqual" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_ge)
        }
        "Less" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_lt)
        }
        "LessOrEqual" if node.input.len() >= 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_le)
        }
        "Where" if node.input.len() >= 3 => {
            let condition = weights.get(&node.input[0])?;
            let true_value = weights.get(&node.input[1])?;
            let false_value = weights.get(&node.input[2])?;
            broadcast_where(condition, true_value, false_value)
        }
        _ => None,
    }
}

fn binary_inputs<'a>(
    node: &'a onnx_proto::NodeProto,
    weights: &'a WeightStore,
) -> Option<(&'a ArrayD<f32>, &'a ArrayD<f32>)> {
    Some((weights.get(&node.input[0])?, weights.get(&node.input[1])?))
}

fn unary<F>(node: &onnx_proto::NodeProto, weights: &WeightStore, op: F) -> Option<ArrayD<f32>>
where
    F: FnMut(f32) -> f32,
{
    weights.get(&node.input[0]).map(|array| array.mapv(op))
}

fn compare_eq(x: f32, y: f32) -> f32 {
    if (x - y).abs() < f32::EPSILON {
        1.0
    } else {
        0.0
    }
}

fn compare_gt(x: f32, y: f32) -> f32 {
    if x > y {
        1.0
    } else {
        0.0
    }
}

fn compare_ge(x: f32, y: f32) -> f32 {
    if x >= y {
        1.0
    } else {
        0.0
    }
}

fn compare_lt(x: f32, y: f32) -> f32 {
    if x < y {
        1.0
    } else {
        0.0
    }
}

fn compare_le(x: f32, y: f32) -> f32 {
    if x <= y {
        1.0
    } else {
        0.0
    }
}

fn finite_only(result: Option<ArrayD<f32>>) -> Option<ArrayD<f32>> {
    result.filter(|array| array.iter().all(|value| value.is_finite()))
}

fn try_fold_matmul(lhs: &ArrayD<f32>, rhs: &ArrayD<f32>) -> Option<ArrayD<f32>> {
    if lhs.ndim() == 1 && rhs.ndim() == 1 {
        let lhs = lhs.clone().into_dimensionality::<Ix1>().ok()?;
        let rhs = rhs.clone().into_dimensionality::<Ix1>().ok()?;
        return (lhs.len() == rhs.len()).then(|| arr0(lhs.dot(&rhs)).into_dyn());
    }
    if lhs.ndim() == 2 && rhs.ndim() == 2 {
        let lhs = lhs.clone().into_dimensionality::<Ix2>().ok()?;
        let rhs = rhs.clone().into_dimensionality::<Ix2>().ok()?;
        return (lhs.ncols() == rhs.nrows()).then(|| lhs.dot(&rhs).into_dyn());
    }
    if lhs.ndim() == 1 && rhs.ndim() == 2 {
        let lhs = lhs.clone().into_dimensionality::<Ix1>().ok()?;
        let rhs = rhs.clone().into_dimensionality::<Ix2>().ok()?;
        return (lhs.len() == rhs.nrows()).then(|| lhs.dot(&rhs).into_dyn());
    }
    if lhs.ndim() == 2 && rhs.ndim() == 1 {
        let lhs = lhs.clone().into_dimensionality::<Ix2>().ok()?;
        let rhs = rhs.clone().into_dimensionality::<Ix1>().ok()?;
        return (lhs.ncols() == rhs.len()).then(|| lhs.dot(&rhs).into_dyn());
    }
    None
}
