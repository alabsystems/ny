// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional LSTM output wiring.
//!
//! Combines forward and reverse direction results:
//! - Y_h: `[2, batch, H]` (ONNX spec — direction dim stacked)
//! - Y_c: `[2, batch, H]` (ONNX spec — direction dim stacked)
//!
//! The bidirectional sequence output Y is rejected by configuration admission:
//! emitting a three-dimensional PyTorch convenience layout under its raw ONNX
//! name would change semantics, while ny does not yet lower the exact four-
//! dimensional ONNX representation.
//!
//! Reference: PyTorch nn.LSTM with bidirectional=True returns `[seq, batch, 2*H]`.
//! ONNX LSTM spec: https://onnx.ai/onnx/operators/onnx__LSTM.html

use crate::onnx_proto;

use super::node_builder::{make_int_attr, make_ints_attr, make_node, make_node_variadic};
use super::{DirectionResult, LstmConfig};

/// Wire bidirectional LSTM outputs by combining forward and reverse directions.
///
/// Output formats:
/// - Y_h: `[2, batch, H]` (ONNX spec — direction dim stacked)
/// - Y_c: `[2, batch, H]` (ONNX spec — direction dim stacked)
pub(super) fn wire_bidirectional_outputs(
    config: &LstmConfig,
    fwd: &DirectionResult,
    rev: &DirectionResult,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) {
    wire_bidirectional_final_states(config, fwd, rev, nodes);
    debug_assert!(config.y_name.is_none());
}

/// Stack forward and reverse final states into [2, batch, H] for Y_h, Y_c.
fn wire_bidirectional_final_states(
    config: &LstmConfig,
    fwd: &DirectionResult,
    rev: &DirectionResult,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) {
    let base = &config.base;
    let domain = &config.domain;
    let direction_axis = if config.layout == 0 { 0 } else { 1 };

    if let Some(ref y_h_out) = config.y_h_name {
        let fwd_us = format!("{base}__lstm_bidi_yh_fwd_us");
        nodes.push(make_node(
            "Unsqueeze",
            &[&fwd.final_h],
            &[&fwd_us],
            &format!("{base}__lstm_bidi_yh_fwd_unsq"),
            domain,
            vec![make_ints_attr("axes", &[direction_axis])],
        ));
        let rev_us = format!("{base}__lstm_bidi_yh_rev_us");
        nodes.push(make_node(
            "Unsqueeze",
            &[&rev.final_h],
            &[&rev_us],
            &format!("{base}__lstm_bidi_yh_rev_unsq"),
            domain,
            vec![make_ints_attr("axes", &[direction_axis])],
        ));
        nodes.push(make_node_variadic(
            "Concat",
            &[&fwd_us, &rev_us],
            &[y_h_out],
            &format!("{base}__lstm_bidi_concat_yh"),
            domain,
            vec![make_int_attr("axis", direction_axis)],
        ));
    }

    if let Some(ref y_c_out) = config.y_c_name {
        let fwd_us = format!("{base}__lstm_bidi_yc_fwd_us");
        nodes.push(make_node(
            "Unsqueeze",
            &[&fwd.final_c],
            &[&fwd_us],
            &format!("{base}__lstm_bidi_yc_fwd_unsq"),
            domain,
            vec![make_ints_attr("axes", &[direction_axis])],
        ));
        let rev_us = format!("{base}__lstm_bidi_yc_rev_us");
        nodes.push(make_node(
            "Unsqueeze",
            &[&rev.final_c],
            &[&rev_us],
            &format!("{base}__lstm_bidi_yc_rev_unsq"),
            domain,
            vec![make_ints_attr("axes", &[direction_axis])],
        ));
        nodes.push(make_node_variadic(
            "Concat",
            &[&fwd_us, &rev_us],
            &[y_c_out],
            &format!("{base}__lstm_bidi_concat_yc"),
            domain,
            vec![make_int_attr("axis", direction_axis)],
        ));
    }
}
