// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{arr0, ArrayD, Ix1, Ix2, IxDyn};

use super::super::broadcast::{broadcast_binop, broadcast_binop_checked, broadcast_where};
use super::super::common::{exact_f32_product, exact_f32_quotient, exact_f32_sum};
use super::super::FoldedTensor;

pub(super) fn try_fold(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    // Shape arithmetic is authored as INT64 in ONNX.  Keep that path exact:
    // evaluating it through WeightStore's compatibility f32 view first loses
    // dimensions above 2^24 and gives floating (rather than integer) Div
    // semantics.  A narrower integer dtype has a recorded range, so only the
    // INT64/Shape provenance (the full i64 range) is admitted here. Checked
    // arithmetic fails closed on overflow and division by zero; importantly,
    // once both operands are proven INT64, failure does NOT fall through to
    // the compatibility f32 evaluator.
    if has_integer_elementwise_evidence(node, weights) {
        return try_fold_typed_int64(node, weights);
    }

    let float_data = try_fold_float(node, weights)?;
    // Some constant shape cones reach this evaluator before authored dtype
    // provenance has been attached to WeightStore.  Preserve a checked exact
    // Mul/Div sidecar when one is available, but keep the generic f32 result
    // authoritative: an integer sidecar alone is not proof that the ONNX
    // operation was authored as INT64.  The later Cast gate requires raw
    // protobuf provenance and exact agreement between both views.
    let integer_data = try_fold_untyped_integer_sidecar(node, weights);
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: None,
    })
}

fn try_fold_typed_int64(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    match node.op_type.as_str() {
        "Add" | "Sub" | "Mul" | "Div" if node.input.len() == 2 => {
            let integer_data = try_fold_int64_binary(node, weights)?;
            Some(folded_int64(integer_data, (i64::MIN, i64::MAX)))
        }
        "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual"
            if node.input.len() == 2 =>
        {
            let (lhs, rhs) =
                compatible_exact_integer_inputs(weights, &node.input[0], &node.input[1])?;
            if contains_private_sentinel(lhs) || contains_private_sentinel(rhs) {
                return None;
            }
            let compare: fn(i64, i64) -> Option<i64> = match node.op_type.as_str() {
                "Equal" => |x, y| Some(i64::from(x == y)),
                "Greater" => |x, y| Some(i64::from(x > y)),
                "GreaterOrEqual" => |x, y| Some(i64::from(x >= y)),
                "Less" => |x, y| Some(i64::from(x < y)),
                "LessOrEqual" => |x, y| Some(i64::from(x <= y)),
                _ => unreachable!(),
            };
            let integer_data = broadcast_binop_checked(lhs, rhs, compare)?;
            // ONNX comparison outputs are BOOL, not an arithmetic integer
            // dtype.  Publish an exact 0/1 float view without integer-range
            // provenance so a downstream Where can fold normally.
            Some(FoldedTensor::from_float(
                integer_data.mapv(|value| value as f32),
            ))
        }
        // `Where` selects, it does not compute, so the exact integer payload
        // survives untouched. Selecting through the f32 compatibility view
        // instead would silently round any |value| > 2^24 shape scalar, which is
        // exactly what the typed branch exists to prevent.
        "Where" if node.input.len() == 3 => {
            let condition = weights.get(&node.input[0])?;
            // A folded ONNX comparison publishes its BOOL result as an exact
            // {0.0, 1.0} float view. Anything else is not a proven boolean and
            // must not be reinterpreted as one.
            if condition.iter().any(|value| *value != 0.0 && *value != 1.0) {
                return None;
            }
            let (true_values, false_values) =
                compatible_exact_integer_inputs(weights, &node.input[1], &node.input[2])?;
            if contains_private_sentinel(true_values) || contains_private_sentinel(false_values) {
                return None;
            }
            let range = weights.get_integer_range(&node.input[1])?;
            let integer_data = broadcast_where(condition, true_values, false_values)?;
            if contains_private_sentinel(&integer_data) {
                return None;
            }
            Some(folded_int64(integer_data, range))
        }
        "Neg" | "Abs" if node.input.len() == 1 => {
            let input = exact_int64_input(weights, &node.input[0])?;
            if contains_private_sentinel(input) {
                return None;
            }
            let op: fn(i64) -> Option<i64> = if node.op_type == "Neg" {
                i64::checked_neg
            } else {
                i64::checked_abs
            };
            let values = input
                .iter()
                .map(|&value| op(value))
                .collect::<Option<Vec<_>>>()?;
            let integer_data = ArrayD::from_shape_vec(IxDyn(input.shape()), values).ok()?;
            if contains_private_sentinel(&integer_data) {
                return None;
            }
            Some(folded_int64(integer_data, (i64::MIN, i64::MAX)))
        }
        // Pow and all remaining elementwise operators lack audited INT64
        // semantics here.  Typed provenance selected this branch precisely so
        // they cannot fall through to the lossy f32 compatibility view.
        _ => None,
    }
}

fn folded_int64(integer_data: ArrayD<i64>, integer_range: (i64, i64)) -> FoldedTensor {
    let float_data = integer_data.mapv(|value| {
        crate::loader::numeric_cast::i64_to_f32_warned(value, "INT64 elementwise constant fold")
    });
    FoldedTensor {
        float_data,
        integer_data: Some(integer_data),
        integer_range: Some(integer_range),
    }
}

fn try_fold_float(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<ArrayD<f32>> {
    match node.op_type.as_str() {
        "Pow" if node.input.len() == 2 => {
            let (base, exponent) = binary_inputs(node, weights)?;
            broadcast_binop_checked(base, exponent, exact_integer_power)
        }
        "Sqrt" if node.input.len() == 1 => unary_checked(node, weights, |value| {
            if !value.is_finite() || value < 0.0 {
                return None;
            }
            let root = value.sqrt();
            ((root as f64) * (root as f64) == value as f64).then_some(root)
        }),
        "Div" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop_checked(lhs, rhs, exact_f32_quotient)
        }
        "Mul" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop_checked(lhs, rhs, exact_f32_product)
        }
        "Add" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop_checked(lhs, rhs, exact_f32_sum)
        }
        "Sub" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop_checked(lhs, rhs, |x, y| exact_f32_sum(x, -y))
        }
        "Neg" if node.input.len() == 1 => {
            unary_checked(node, weights, |value| value.is_finite().then_some(-value))
        }
        "Sin" if node.input.len() == 1 => {
            unary_checked(node, weights, |value| (value == 0.0).then_some(value))
        }
        "Cos" if node.input.len() == 1 => {
            unary_checked(node, weights, |value| (value == 0.0).then_some(1.0))
        }
        "Abs" if node.input.len() == 1 => unary_checked(node, weights, |value| {
            value.is_finite().then_some(value.abs())
        }),
        "Relu" if node.input.len() == 1 => unary_checked(node, weights, |value| {
            value.is_finite().then_some(value.max(0.0))
        }),
        "Sigmoid" if node.input.len() == 1 => {
            unary_checked(node, weights, |value| (value == 0.0).then_some(0.5))
        }
        "Tanh" if node.input.len() == 1 => {
            unary_checked(node, weights, |value| (value == 0.0).then_some(value))
        }
        "Exp" if node.input.len() == 1 => {
            unary_checked(node, weights, |value| (value == 0.0).then_some(1.0))
        }
        "Log" if node.input.len() == 1 => {
            unary_checked(node, weights, |value| (value == 1.0).then_some(0.0))
        }
        // A rounded f32 dot product is not the authored exact-real MatMul.
        // Keep it as a graph operation until interval-valued frozen constants
        // can carry the residual instead of exactifying the rounded center.
        "MatMul" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            try_fold_matmul_exact(lhs, rhs)
        }
        "Equal" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_eq)
        }
        "Greater" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_gt)
        }
        "GreaterOrEqual" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_ge)
        }
        "Less" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_lt)
        }
        "LessOrEqual" if node.input.len() == 2 => {
            let (lhs, rhs) = binary_inputs(node, weights)?;
            broadcast_binop(lhs, rhs, compare_le)
        }
        "Where" if node.input.len() == 3 => {
            let condition = weights.get(&node.input[0])?;
            let true_value = weights.get(&node.input[1])?;
            let false_value = weights.get(&node.input[2])?;
            broadcast_where(condition, true_value, false_value)
        }
        _ => None,
    }
}

fn exact_integer_power(base: f32, exponent: f32) -> Option<f32> {
    if !base.is_finite() || !exponent.is_finite() {
        return None;
    }
    if exponent == 0.0 {
        return Some(1.0);
    }
    if base == 1.0 {
        return Some(1.0);
    }
    if base == 0.0 && exponent > 0.0 {
        // IEEE-754 pow preserves a negative zero only for a positive odd
        // integer exponent.  Equality with 0.0 alone cannot distinguish the
        // sign, so do not return the base verbatim for even exponents.
        let odd_integer = exponent.fract() == 0.0
            && exponent.abs() <= 16_777_216.0
            && (exponent.abs() as u64) % 2 == 1;
        return Some(if odd_integer { base } else { 0.0 });
    }
    if exponent.fract() != 0.0 || exponent.abs() > 16_777_216.0 {
        return None;
    }

    let negative = exponent < 0.0;
    let mut remaining = exponent.abs() as u64;
    let mut factor = base;
    let mut result = 1.0_f32;
    while remaining != 0 {
        if remaining & 1 == 1 {
            result = exact_f32_product(result, factor)?;
        }
        remaining >>= 1;
        if remaining != 0 {
            factor = exact_f32_product(factor, factor)?;
        }
    }
    if negative {
        exact_f32_quotient(1.0, result)
    } else {
        Some(result)
    }
}

fn try_fold_int64_binary(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<ArrayD<i64>> {
    let op: fn(i64, i64) -> Option<i64> = match node.op_type.as_str() {
        "Add" if node.input.len() == 2 => i64::checked_add,
        "Sub" if node.input.len() == 2 => i64::checked_sub,
        "Mul" if node.input.len() == 2 => i64::checked_mul,
        "Div" if node.input.len() == 2 => i64::checked_div,
        _ => return None,
    };
    let lhs_name = &node.input[0];
    let rhs_name = &node.input[1];
    let lhs = exact_int64_input(weights, lhs_name)?;
    let rhs = exact_int64_input(weights, rhs_name)?;
    // Dynamic Shape dimensions are represented internally by private
    // copy-axis sentinels.  They are semantic placeholders, not ONNX integer
    // values, so arithmetic on them would corrupt which activation axis a
    // later Reshape must copy.  Once the inputs are known-INT64, rejecting here
    // also prevents the caller from falling through to lossy f32 arithmetic.
    if contains_private_sentinel(lhs) || contains_private_sentinel(rhs) {
        return None;
    }
    let result = broadcast_binop_checked(lhs, rhs, op)?;
    // Likewise, do not let ordinary authored integers synthesize a value in
    // the reserved range: a downstream Reshape would otherwise reinterpret it
    // as a private copy-axis marker.
    if result
        .iter()
        .any(|&value| ny_core::reshape_copy_axis_from_sentinel(value).is_some())
    {
        return None;
    }
    Some(result)
}

fn try_fold_untyped_integer_sidecar(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<ArrayD<i64>> {
    let op: fn(i64, i64) -> Option<i64> = match node.op_type.as_str() {
        "Mul" if node.input.len() == 2 => i64::checked_mul,
        "Div" if node.input.len() == 2 => i64::checked_div,
        _ => return None,
    };
    let lhs = weights.get_integers(&node.input[0])?;
    let rhs = weights.get_integers(&node.input[1])?;
    if contains_private_sentinel(lhs) || contains_private_sentinel(rhs) {
        return None;
    }
    let result = broadcast_binop_checked(lhs, rhs, op)?;
    (!contains_private_sentinel(&result)).then_some(result)
}

fn has_integer_elementwise_evidence(node: &onnx_proto::NodeProto, weights: &WeightStore) -> bool {
    matches!(
        node.op_type.as_str(),
        "Pow"
            | "Sqrt"
            | "Div"
            | "Mul"
            | "Add"
            | "Sub"
            | "Neg"
            | "Sin"
            | "Cos"
            | "Abs"
            | "Relu"
            | "Sigmoid"
            | "Tanh"
            | "Exp"
            | "Log"
            | "MatMul"
            | "Equal"
            | "Greater"
            | "GreaterOrEqual"
            | "Less"
            | "LessOrEqual"
            | "Where"
    ) && node.input.iter().any(|name| {
        weights.get_integers(name).is_some() || weights.get_integer_range(name).is_some()
    })
}

fn has_int64_provenance(weights: &WeightStore, name: &str) -> bool {
    weights.get_integer_range(name) == Some((i64::MIN, i64::MAX))
}

fn exact_int64_input<'a>(weights: &'a WeightStore, name: &str) -> Option<&'a ArrayD<i64>> {
    has_int64_provenance(weights, name)
        .then(|| weights.get_integers(name))
        .flatten()
}

fn compatible_exact_integer_inputs<'a>(
    weights: &'a WeightStore,
    lhs: &str,
    rhs: &str,
) -> Option<(&'a ArrayD<i64>, &'a ArrayD<i64>)> {
    let range = weights.get_integer_range(lhs)?;
    (weights.get_integer_range(rhs) == Some(range))
        .then_some((weights.get_integers(lhs)?, weights.get_integers(rhs)?))
}

fn contains_private_sentinel(values: &ArrayD<i64>) -> bool {
    values
        .iter()
        .any(|&value| ny_core::reshape_copy_axis_from_sentinel(value).is_some())
}

fn binary_inputs<'a>(
    node: &'a onnx_proto::NodeProto,
    weights: &'a WeightStore,
) -> Option<(&'a ArrayD<f32>, &'a ArrayD<f32>)> {
    Some((weights.get(&node.input[0])?, weights.get(&node.input[1])?))
}

fn unary_checked<F>(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    operation: F,
) -> Option<ArrayD<f32>>
where
    F: FnMut(f32) -> Option<f32>,
{
    let input = weights.get(&node.input[0])?;
    let values = input
        .iter()
        .copied()
        .map(operation)
        .collect::<Option<Vec<_>>>()?;
    ArrayD::from_shape_vec(input.raw_dim(), values).ok()
}

fn compare_eq(x: f32, y: f32) -> f32 {
    if x == y {
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

fn exact_dot<'a>(
    lhs: impl Iterator<Item = &'a f32>,
    rhs: impl Iterator<Item = &'a f32>,
) -> Option<f32> {
    let mut sum = 0.0_f32;
    for (&lhs, &rhs) in lhs.zip(rhs) {
        sum = exact_f32_sum(sum, exact_f32_product(lhs, rhs)?)?;
    }
    Some(sum)
}

fn try_fold_matmul_exact(lhs: &ArrayD<f32>, rhs: &ArrayD<f32>) -> Option<ArrayD<f32>> {
    if lhs.ndim() == 1 && rhs.ndim() == 1 {
        let lhs = lhs.clone().into_dimensionality::<Ix1>().ok()?;
        let rhs = rhs.clone().into_dimensionality::<Ix1>().ok()?;
        return (lhs.len() == rhs.len())
            .then(|| exact_dot(lhs.iter(), rhs.iter()))
            .flatten()
            .map(|value| arr0(value).into_dyn());
    }
    if lhs.ndim() == 2 && rhs.ndim() == 2 {
        let lhs = lhs.clone().into_dimensionality::<Ix2>().ok()?;
        let rhs = rhs.clone().into_dimensionality::<Ix2>().ok()?;
        if lhs.ncols() != rhs.nrows() {
            return None;
        }
        let mut values = Vec::with_capacity(lhs.nrows().checked_mul(rhs.ncols())?);
        for row in lhs.rows() {
            for column in rhs.columns() {
                values.push(exact_dot(row.iter(), column.iter())?);
            }
        }
        return ArrayD::from_shape_vec(IxDyn(&[lhs.nrows(), rhs.ncols()]), values).ok();
    }
    if lhs.ndim() == 1 && rhs.ndim() == 2 {
        let lhs = lhs.clone().into_dimensionality::<Ix1>().ok()?;
        let rhs = rhs.clone().into_dimensionality::<Ix2>().ok()?;
        if lhs.len() != rhs.nrows() {
            return None;
        }
        let values = rhs
            .columns()
            .into_iter()
            .map(|column| exact_dot(lhs.iter(), column.iter()))
            .collect::<Option<Vec<_>>>()?;
        return ArrayD::from_shape_vec(IxDyn(&[rhs.ncols()]), values).ok();
    }
    if lhs.ndim() == 2 && rhs.ndim() == 1 {
        let lhs = lhs.clone().into_dimensionality::<Ix2>().ok()?;
        let rhs = rhs.clone().into_dimensionality::<Ix1>().ok()?;
        if lhs.ncols() != rhs.len() {
            return None;
        }
        let values = lhs
            .rows()
            .into_iter()
            .map(|row| exact_dot(row.iter(), rhs.iter()))
            .collect::<Option<Vec<_>>>()?;
        return ArrayD::from_shape_vec(IxDyn(&[lhs.nrows()]), values).ok();
    }
    None
}
