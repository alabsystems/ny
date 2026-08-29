// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The FALLBACK constant folder: what runs only after the bit-exact folder has
//! already declined.
//!
//! # Ordering, which is the whole safety argument for the disarmed arm
//!
//! `fold_constant_nodes_once` calls the existing exact folder FIRST and
//! verbatim. This module is reached only when that call returned `None` AND
//! [`super::interval_carrier_armed`] is true. So with the lever unset nothing
//! here executes, no carrier is allocated, and `WeightStore`'s interval map
//! stays empty — which is what the dark-behaviour tests assert rather than
//! assume.
//!
//! # What it will and will not fold
//!
//! Exactly three operations have interval rules: `MatMul`, `Add` and `Relu`.
//! They are the complete constant cone that the lsnc_relu Lyapunov defect
//! traverses, and each has a one-line monotone widening argument (see
//! [`super::interval`]). EVERY other operation declines, which leaves the node
//! in the graph and keeps today's fail-closed behaviour.
//!
//! The default is refusal, not best effort. An operation without an interval
//! rule that silently picked an endpoint — a comparison, a `Where`, a `Div` by
//! an interval straddling zero — would choose a branch the true value might not
//! take, and no downstream widening can repair that.
//!
//! Integer-provenance tensors are refused outright: ONNX shape arithmetic is
//! authored as INT64 and is exact by construction, so an interval there would
//! be both meaningless and a way to smuggle a rounded extent into a shape.

use crate::onnx_proto;
use crate::WeightStore;

use super::interval::{self, IntervalTensor};
use super::is_standard_onnx_domain;

/// Fold one node into an interval-valued frozen constant, or decline.
///
/// Returns the output tensor name and its enclosure.
pub(super) fn try_fold_node_interval(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<(String, IntervalTensor)> {
    if !is_standard_onnx_domain(&node.domain) {
        return None;
    }
    if node.output.len() != 1 {
        return None;
    }
    let output_name = node.output.first()?;
    if output_name.is_empty()
        || weights.contains_key(output_name)
        || weights.contains_interval(output_name)
    {
        return None;
    }
    // Every input must already be a frozen constant, point or interval.
    let mut inputs = Vec::with_capacity(node.input.len());
    for name in node.input.iter().filter(|name| !name.is_empty()) {
        // INT64 shape arithmetic stays exact; never carry an interval there.
        if weights.get_integers(name).is_some() || weights.get_integer_range(name).is_some() {
            return None;
        }
        inputs.push(constant_operand(weights, name)?);
    }

    let folded = match node.op_type.as_str() {
        "MatMul" if inputs.len() == 2 => interval::matmul(&inputs[0], &inputs[1]),
        "Add" if inputs.len() == 2 => interval::add(&inputs[0], &inputs[1]),
        "Relu" if inputs.len() == 1 => interval::relu(&inputs[0]),
        // No rule: decline. The node stays in the graph and keeps today's
        // fail-closed lowering.
        _ => None,
    }?;
    Some((output_name.clone(), folded))
}

/// View a frozen constant as an interval, whether it is stored as a point or as
/// an enclosure.
fn constant_operand(weights: &WeightStore, name: &str) -> Option<IntervalTensor> {
    if let Some((lower, upper)) = weights.get_interval(name) {
        return Some(IntervalTensor {
            lower: lower.clone(),
            upper: upper.clone(),
        });
    }
    weights.get(name).map(IntervalTensor::point)
}
