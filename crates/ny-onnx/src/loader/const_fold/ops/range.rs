// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constant folding for the ONNX `Range` op.
//!
//! `Range(start, limit, delta)` produces the 1-D arithmetic sequence
//! `[start, start + delta, start + 2*delta, ...]` whose length is
//! `max(ceil((limit - start) / delta), 0)` (ONNX Range spec). When all three
//! scalar inputs are constants (weights), the output is a constant tensor whose
//! shape and contents are fully determined at load time. In a verification graph
//! with fixed input shapes this is exact — no bounds approximation is involved.
//!
//! The fold preserves an exact i64 payload (`FoldedTensor::integer_data`)
//! whenever the three scalars are integral, because `Range` outputs are almost
//! always shape/index sequences consumed by downstream `Slice`/`Gather`/`Expand`
//! ops via [`read_tensor_i64s`](super::super::common::read_tensor_i64s). Routing
//! the exact integer view avoids any f32 rounding for large indices.

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use tracing::{debug, warn};

use super::super::common::{
    exact_f32_product, exact_f32_quotient, exact_f32_sum, read_tensor_i64s,
};
use super::super::FoldedTensor;

/// Cap on the number of elements a folded `Range` may materialize. Mirrors the
/// limits used by `ConstantOfShape`/`Expand` so a malformed model cannot force
/// an unbounded allocation during loading.
const MAX_RANGE_ELEMENTS: usize = 10_000_000;

pub(super) fn try_fold(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    if node.op_type != "Range"
        || node.input.len() != 3
        || node.input.iter().any(String::is_empty)
        || !node.attribute.is_empty()
    {
        return None;
    }

    let integer_evidence = node.input[..3].iter().any(|name| {
        weights.get_integers(name).is_some() || weights.get_integer_range(name).is_some()
    });
    if integer_evidence {
        // Once any operand carries integer provenance, failure must not fall
        // through to its lossy f32 compatibility mirror.
        return try_fold_integer(node, weights);
    }

    try_fold_float(node, weights)
}

/// Read a scalar (single-element) constant input as f32.
fn read_scalar_f32(weights: &WeightStore, name: &str) -> Option<f32> {
    let arr = weights.get(name)?;
    if arr.len() != 1 {
        return None;
    }
    arr.iter().next().copied()
}

/// Read a scalar (single-element) constant input as i64, preferring the exact
/// integer payload and falling back to a value-preserving f32→i64 parse.
fn read_scalar_i64(weights: &WeightStore, name: &str) -> Option<i64> {
    let values = read_tensor_i64s(weights, name)?;
    if values.len() != 1 {
        return None;
    }
    values.into_iter().next()
}

fn try_fold_integer(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<FoldedTensor> {
    let integer_range = weights.get_integer_range(&node.input[0])?;
    if weights.get_integer_range(&node.input[1]) != Some(integer_range)
        || weights.get_integer_range(&node.input[2]) != Some(integer_range)
    {
        return None;
    }
    let start = read_scalar_i64(weights, &node.input[0])?;
    let limit = read_scalar_i64(weights, &node.input[1])?;
    let delta = read_scalar_i64(weights, &node.input[2])?;

    if delta == 0 {
        warn!("Range constant fold: delta=0 is invalid; skipping");
        return None;
    }

    // ONNX: number_of_elements = max(ceil((limit - start) / delta), 0).
    let span = (limit as i128) - (start as i128);
    let delta_i128 = delta as i128;
    // Ceiling division toward +inf of span/delta, clamped at 0.
    let count: i128 = if (span > 0 && delta_i128 > 0) || (span < 0 && delta_i128 < 0) {
        // Same sign → at least one element; ceil of positive ratio.
        (span.abs() + delta_i128.abs() - 1) / delta_i128.abs()
    } else {
        // span == 0, or opposite signs → empty sequence.
        0
    };
    let count = usize::try_from(count).ok()?;

    if count > MAX_RANGE_ELEMENTS {
        warn!(
            "Range constant fold: refusing to allocate {} elements (limit {})",
            count, MAX_RANGE_ELEMENTS
        );
        return None;
    }
    if count == 0 {
        debug!("Range constant fold: empty sequence (start={start}, limit={limit}, delta={delta})");
        // Keep an empty 1-D tensor so downstream shape ops still resolve.
        let empty_i64 = ArrayD::from_shape_vec(IxDyn(&[0]), Vec::<i64>::new()).ok()?;
        let empty_f32 = ArrayD::from_shape_vec(IxDyn(&[0]), Vec::<f32>::new()).ok()?;
        return Some(FoldedTensor {
            float_data: empty_f32,
            integer_data: Some(empty_i64),
            integer_range: Some(integer_range),
        });
    }

    let mut int_values = Vec::with_capacity(count);
    let mut value = start as i128;
    for _ in 0..count {
        int_values.push(value);
        value += delta_i128;
    }
    // Guard against any out-of-i64 element before committing.
    let int_values: Vec<i64> = int_values
        .into_iter()
        .map(i64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;

    let float_values: Vec<f32> = int_values
        .iter()
        .map(|&v| crate::loader::numeric_cast::i64_to_f32_warned(v, "Range constant fold"))
        .collect();

    let integer_data = ArrayD::from_shape_vec(IxDyn(&[count]), int_values).ok()?;
    let float_data = ArrayD::from_shape_vec(IxDyn(&[count]), float_values).ok()?;
    debug!(
        "Constant folded Range: {} elements (start={start}, limit={limit}, delta={delta})",
        count
    );
    Some(FoldedTensor {
        float_data,
        integer_data: Some(integer_data),
        integer_range: Some(integer_range),
    })
}

fn try_fold_float(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<FoldedTensor> {
    let start = read_scalar_f32(weights, &node.input[0])?;
    let limit = read_scalar_f32(weights, &node.input[1])?;
    let delta = read_scalar_f32(weights, &node.input[2])?;

    if !start.is_finite() || !limit.is_finite() || !delta.is_finite() {
        warn!("Range constant fold: non-finite start/limit/delta; skipping");
        return None;
    }
    if delta == 0.0 {
        warn!("Range constant fold: delta=0 is invalid; skipping");
        return None;
    }

    // Authenticate the exact-real span and quotient before using them to
    // decide sequence length. A rounded value on either side of an integer
    // boundary can otherwise add or remove a Range element.
    let span = exact_f32_sum(limit, -start)?;
    let ratio = exact_f32_quotient(span, delta)?;
    let count_f = ratio.ceil();
    if !count_f.is_finite() || count_f <= 0.0 {
        debug!("Range constant fold: empty/degenerate float sequence; producing empty tensor");
        let empty = ArrayD::from_shape_vec(IxDyn(&[0]), Vec::<f32>::new()).ok()?;
        return Some(FoldedTensor::from_float(empty));
    }
    if count_f > MAX_RANGE_ELEMENTS as f32 {
        warn!(
            "Range constant fold: refusing to allocate {} elements (limit {})",
            count_f, MAX_RANGE_ELEMENTS
        );
        return None;
    }
    let count = count_f as usize;

    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        // MAX_RANGE_ELEMENTS is below 2^24, so i is exactly binary32. Require
        // both the product and addition to be exactly representable before
        // publishing the sequence as frozen point constants.
        let offset = exact_f32_product(i as f32, delta)?;
        values.push(exact_f32_sum(start, offset)?);
    }
    let float_data = ArrayD::from_shape_vec(IxDyn(&[count]), values).ok()?;
    debug!("Constant folded Range (float): {} elements", count);
    Some(FoldedTensor::from_float(float_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, ArrayD, IxDyn};

    fn scalar_node(start: &str, limit: &str, delta: &str) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: vec![start.to_string(), limit.to_string(), delta.to_string()],
            output: vec!["out".to_string()],
            name: "Range_test".to_string(),
            op_type: "Range".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }
    }

    fn insert_i64_scalar(weights: &mut WeightStore, name: &str, value: i64) {
        weights.insert(name.to_string(), arr0(value as f32).into_dyn());
        weights.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value]).unwrap(),
        );
        weights.insert_integer_range(name.to_string(), i64::MIN, i64::MAX);
    }

    #[test]
    fn range_integer_basic_sequence() {
        let mut weights = WeightStore::new();
        insert_i64_scalar(&mut weights, "start", 0);
        insert_i64_scalar(&mut weights, "limit", 5);
        insert_i64_scalar(&mut weights, "delta", 1);
        let node = scalar_node("start", "limit", "delta");
        let folded = try_fold(&node, &weights).expect("integer Range should fold");
        assert_eq!(folded.integer_range, Some((i64::MIN, i64::MAX)));
        let ints = folded.integer_data.expect("integer payload preserved");
        assert_eq!(ints.as_slice().unwrap(), &[0, 1, 2, 3, 4]);
        assert_eq!(
            folded.float_data.as_slice().unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn range_integer_step_two() {
        let mut weights = WeightStore::new();
        insert_i64_scalar(&mut weights, "start", 1);
        insert_i64_scalar(&mut weights, "limit", 10);
        insert_i64_scalar(&mut weights, "delta", 2);
        let node = scalar_node("start", "limit", "delta");
        let folded = try_fold(&node, &weights).unwrap();
        // ceil((10-1)/2) = 5 → [1,3,5,7,9]
        assert_eq!(
            folded.integer_data.unwrap().as_slice().unwrap(),
            &[1, 3, 5, 7, 9]
        );
    }

    #[test]
    fn range_integer_negative_step() {
        let mut weights = WeightStore::new();
        insert_i64_scalar(&mut weights, "start", 5);
        insert_i64_scalar(&mut weights, "limit", 0);
        insert_i64_scalar(&mut weights, "delta", -1);
        let node = scalar_node("start", "limit", "delta");
        let folded = try_fold(&node, &weights).unwrap();
        assert_eq!(
            folded.integer_data.unwrap().as_slice().unwrap(),
            &[5, 4, 3, 2, 1]
        );
    }

    #[test]
    fn range_integer_empty_when_limit_not_reached() {
        let mut weights = WeightStore::new();
        insert_i64_scalar(&mut weights, "start", 5);
        insert_i64_scalar(&mut weights, "limit", 5);
        insert_i64_scalar(&mut weights, "delta", 1);
        let node = scalar_node("start", "limit", "delta");
        let folded = try_fold(&node, &weights).unwrap();
        assert_eq!(folded.integer_data.unwrap().len(), 0);
        assert_eq!(folded.float_data.len(), 0);
    }

    #[test]
    fn range_integer_empty_wrong_direction() {
        let mut weights = WeightStore::new();
        insert_i64_scalar(&mut weights, "start", 0);
        insert_i64_scalar(&mut weights, "limit", 5);
        insert_i64_scalar(&mut weights, "delta", -1);
        let node = scalar_node("start", "limit", "delta");
        let folded = try_fold(&node, &weights).unwrap();
        assert_eq!(folded.integer_data.unwrap().len(), 0);
    }

    #[test]
    fn range_delta_zero_rejected() {
        let mut weights = WeightStore::new();
        insert_i64_scalar(&mut weights, "start", 0);
        insert_i64_scalar(&mut weights, "limit", 5);
        insert_i64_scalar(&mut weights, "delta", 0);
        let node = scalar_node("start", "limit", "delta");
        assert!(try_fold(&node, &weights).is_none());
    }

    #[test]
    fn range_float_sequence() {
        let mut weights = WeightStore::new();
        // Non-integral values exercise the float path.
        weights.insert("start".to_string(), arr0(0.5_f32).into_dyn());
        weights.insert("limit".to_string(), arr0(2.0_f32).into_dyn());
        weights.insert("delta".to_string(), arr0(0.5_f32).into_dyn());
        let node = scalar_node("start", "limit", "delta");
        let folded = try_fold(&node, &weights).unwrap();
        // ceil((2.0-0.5)/0.5) = 3 → [0.5, 1.0, 1.5]
        assert!(folded.integer_data.is_none());
        assert_eq!(folded.float_data.as_slice().unwrap(), &[0.5, 1.0, 1.5]);
    }

    #[test]
    fn range_float_rejects_inexact_sequence_arithmetic() {
        let mut weights = WeightStore::new();
        weights.insert("start".to_string(), arr0(0.1_f32).into_dyn());
        weights.insert("limit".to_string(), arr0(0.5_f32).into_dyn());
        weights.insert("delta".to_string(), arr0(0.1_f32).into_dyn());
        let node = scalar_node("start", "limit", "delta");
        assert!(
            try_fold(&node, &weights).is_none(),
            "rounded float Range arithmetic must remain explicit"
        );
    }

    #[test]
    fn range_non_scalar_input_skipped() {
        let mut weights = WeightStore::new();
        weights.insert(
            "start".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, 1.0]).unwrap(),
        );
        weights.insert("limit".to_string(), arr0(5.0_f32).into_dyn());
        weights.insert("delta".to_string(), arr0(1.0_f32).into_dyn());
        let node = scalar_node("start", "limit", "delta");
        assert!(try_fold(&node, &weights).is_none());
    }

    #[test]
    fn range_missing_input_skipped() {
        let mut weights = WeightStore::new();
        insert_i64_scalar(&mut weights, "start", 0);
        insert_i64_scalar(&mut weights, "limit", 5);
        // delta absent
        let node = scalar_node("start", "limit", "delta");
        assert!(try_fold(&node, &weights).is_none());
    }

    // Integral FLOAT inputs still use the authenticated float path; they must
    // not acquire integer provenance merely because their values look whole.
    #[test]
    fn range_integer_via_f32_fallback() {
        let mut weights = WeightStore::new();
        // Only float storage (no integer payload) — read_tensor_i64s falls back
        // to parse_scalar_i64 on the f32 view.
        weights.insert("start".to_string(), arr0(0.0_f32).into_dyn());
        weights.insert("limit".to_string(), arr0(3.0_f32).into_dyn());
        weights.insert("delta".to_string(), arr0(1.0_f32).into_dyn());
        let node = scalar_node("start", "limit", "delta");
        let folded = try_fold(&node, &weights).unwrap();
        assert!(folded.integer_data.is_none());
        assert_eq!(folded.float_data.as_slice().unwrap(), &[0.0, 1.0, 2.0]);
    }
}
