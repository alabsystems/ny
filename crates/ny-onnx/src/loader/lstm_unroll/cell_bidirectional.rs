// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional LSTM output wiring.
//!
//! Combines forward and reverse direction results:
//! - Y_h: `[2, batch, H]` (ONNX spec — direction dim stacked)
//! - Y_c: `[2, batch, H]` (ONNX spec — direction dim stacked)
//! - Y:   `[seq, batch, 2*H]` (PyTorch runtime format — directions concatenated along hidden dim)
//!
//! Y uses PyTorch runtime format instead of the ONNX spec format `[seq, 2, batch, H]`
//! because the graph converter's batch stripping assumes batch is always axis 0.
//! The 4D ONNX format has batch at axis 2, which breaks batch stripping and causes
//! downstream Transpose/Reshape failures. PyTorch exports include Transpose+Reshape
//! nodes to convert from ONNX format to this runtime format; by producing the runtime
//! format directly, we eliminate the need for those conversion nodes.
//!
//! Reference: PyTorch nn.LSTM with bidirectional=True returns `[seq, batch, 2*H]`.
//! ONNX LSTM spec: https://onnx.ai/onnx/operators/onnx__LSTM.html

use crate::onnx_proto;

use super::node_builder::{make_int_attr, make_ints_attr, make_node, make_node_variadic};
use super::{DirectionResult, LstmConfig};

/// Wire bidirectional LSTM outputs by combining forward and reverse directions.
///
/// Output formats:
/// - Y: `[seq, batch, 2*H]` (layout=0) or `[batch, seq, 2*H]` (layout=1).
///   PyTorch runtime format with fwd/rev hidden states concatenated along last axis.
/// - Y_h: `[2, batch, H]` (ONNX spec — direction dim stacked)
/// - Y_c: `[2, batch, H]` (ONNX spec — direction dim stacked)
pub(super) fn wire_bidirectional_outputs(
    config: &LstmConfig,
    fwd: &DirectionResult,
    rev: &DirectionResult,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) {
    wire_bidirectional_final_states(config, fwd, rev, nodes);
    if let Some(ref y_out) = config.y_name {
        wire_bidirectional_sequence_output(config, fwd, rev, y_out, nodes);
    }
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

    if let Some(ref y_h_out) = config.y_h_name {
        let fwd_us = format!("{base}__lstm_bidi_yh_fwd_us");
        nodes.push(make_node(
            "Unsqueeze",
            &[&fwd.final_h],
            &[&fwd_us],
            &format!("{base}__lstm_bidi_yh_fwd_unsq"),
            domain,
            vec![make_ints_attr("axes", &[0])],
        ));
        let rev_us = format!("{base}__lstm_bidi_yh_rev_us");
        nodes.push(make_node(
            "Unsqueeze",
            &[&rev.final_h],
            &[&rev_us],
            &format!("{base}__lstm_bidi_yh_rev_unsq"),
            domain,
            vec![make_ints_attr("axes", &[0])],
        ));
        nodes.push(make_node_variadic(
            "Concat",
            &[&fwd_us, &rev_us],
            &[y_h_out],
            &format!("{base}__lstm_bidi_concat_yh"),
            domain,
            vec![make_int_attr("axis", 0)],
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
            vec![make_ints_attr("axes", &[0])],
        ));
        let rev_us = format!("{base}__lstm_bidi_yc_rev_us");
        nodes.push(make_node(
            "Unsqueeze",
            &[&rev.final_c],
            &[&rev_us],
            &format!("{base}__lstm_bidi_yc_rev_unsq"),
            domain,
            vec![make_ints_attr("axes", &[0])],
        ));
        nodes.push(make_node_variadic(
            "Concat",
            &[&fwd_us, &rev_us],
            &[y_c_out],
            &format!("{base}__lstm_bidi_concat_yc"),
            domain,
            vec![make_int_attr("axis", 0)],
        ));
    }
}

/// Build bidirectional Y output in PyTorch runtime format: `[seq, batch, 2*H]`.
///
/// For each timestep t:
/// 1. Concat fwd_h_t `[batch, H]` and rev_h_t `[batch, H]` along last axis → `[batch, 2*H]`
/// 2. Unsqueeze to add time dim → `[1, batch, 2*H]` (layout=0) or `[batch, 1, 2*H]` (layout=1)
///
/// Then concat all timesteps along time axis → `[seq, batch, 2*H]` or `[batch, seq, 2*H]`.
///
/// This differs from the ONNX spec format `[seq, 2, batch, H]`. We produce the PyTorch
/// runtime format because ny-propagate's batch stripping assumes batch is axis 0.
/// The 4D ONNX format has batch at axis 2, which breaks batch stripping. PyTorch ONNX
/// exports add Transpose+Reshape nodes to convert ONNX→PyTorch format; by producing the
/// runtime format directly, those conversion nodes should be fused/removed during loading.
///
/// After batch stripping: `[seq, 2*H]` — the concatenated hidden states at each timestep.
fn wire_bidirectional_sequence_output(
    config: &LstmConfig,
    fwd: &DirectionResult,
    rev: &DirectionResult,
    y_out: &str,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) {
    let base = &config.base;
    let domain = &config.domain;
    let layout = config.layout;
    let mut time_concat_inputs = Vec::new();

    // Time dim: layout=0 -> axis 0 (before batch); layout=1 -> axis 1 (after batch)
    let time_axis: i64 = if layout == 0 { 0 } else { 1 };

    for t in 0..config.seq_len {
        // Concat fwd and rev hidden states along the last (hidden) axis.
        // fwd_h_t: [batch, H], rev_h_t: [batch, H] → [batch, 2*H]
        let dir_cat = format!("{base}__lstm_bidi_y_t{t}_dir_cat");
        nodes.push(make_node_variadic(
            "Concat",
            &[&fwd.all_h[t], &rev.all_h[t]],
            &[&dir_cat],
            &format!("{base}__lstm_bidi_y_t{t}_cat_hidden"),
            domain,
            vec![make_int_attr("axis", -1)],
        ));

        // Unsqueeze to add time dimension.
        let time_us = format!("{base}__lstm_bidi_y_t{t}_time_us");
        nodes.push(make_node(
            "Unsqueeze",
            &[&dir_cat],
            &[&time_us],
            &format!("{base}__lstm_bidi_y_t{t}_time_unsq"),
            domain,
            vec![make_ints_attr("axes", &[time_axis])],
        ));

        time_concat_inputs.push(time_us);
    }

    let refs: Vec<&str> = time_concat_inputs.iter().map(|s| s.as_str()).collect();
    nodes.push(make_node_variadic(
        "Concat",
        &refs,
        &[y_out],
        &format!("{base}__lstm_bidi_concat_Y"),
        domain,
        vec![make_int_attr("axis", time_axis)],
    ));
}
