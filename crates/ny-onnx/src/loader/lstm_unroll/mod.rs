// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LSTM node unrolling for ONNX model loading.
//!
//! Decomposes ONNX LSTM nodes into per-timestep cell operations
//! (MatMul, Add, Sigmoid, Tanh, Mul) that ny's bound propagation
//! framework can handle natively.
//!
//! Follows the alpha-beta-CROWN reference approach: unroll LSTMCell in a loop.
//! Reference: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/examples/sequence/lstm.py
//!
//! Supported scope:
//! - Forward and bidirectional directions (no reverse-only)
//! - No peepholes (P input ignored)
//! - Default activations (Sigmoid, Tanh, Tanh)
//! - Requires concrete (non-dynamic) sequence length
//! - layout=0 (seq_first) and layout=1 (batch_first)
//!
//! Bidirectional LSTM: the ONNX spec stores weights as `[num_directions, ...]`.
//! We split by direction, unroll forward (t=0→T-1) and reverse (t=T-1→0)
//! independently, then concatenate outputs along the direction dimension.
//! This enables the Kokoro duration predictor (#3497) BiLSTM + projection path.
//!
//! ONNX LSTM spec: https://onnx.ai/onnx/operators/onnx__LSTM.html

mod cell;
mod cell_bidirectional;
mod node_builder;
mod weights;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_bilstm;

use crate::onnx_proto;
use crate::WeightStore;
use tracing::{debug, info, warn};

use weights::{find_tensor_shape, precompute_lstm_weights, store_lstm_slice_params};

/// Configuration extracted from an LSTM node.
struct LstmConfig {
    hidden_size: usize,
    input_size: usize,
    seq_len: usize,
    layout: i64,
    base: String,
    domain: String,
    batch_size: usize,
    /// 1 for forward-only, 2 for bidirectional.
    num_directions: usize,
    y_name: Option<String>,
    y_h_name: Option<String>,
    y_c_name: Option<String>,
}

/// Names of pre-computed weight tensors stored in the WeightStore.
struct LstmWeightNames {
    w_t: String,
    r_t: String,
    bias: String,
    h0: String,
    c0: String,
    /// Pre-computed `h0 @ R_T`, stored as a weight so that the graph builder
    /// does not see a MatMul with two weight inputs at t=0.
    h0_hr: String,
    /// Reshape target `[batch, input_size]` for timestep extraction.
    /// Used instead of Squeeze because the propagation framework forbids
    /// Squeeze at axis 0 (which the time axis becomes after batch stripping).
    x_reshape: String,
    gate_axis: String,
    gate_step: String,
    time_axis: String,
    time_step: String,
}

/// Per-direction unrolling result: hidden states at each timestep and final state.
struct DirectionResult {
    /// Hidden state at each original timestep (indexed by original time position).
    all_h: Vec<String>,
    /// Final hidden state name.
    final_h: String,
    /// Final cell state name.
    final_c: String,
}

/// Lower ONNX LSTM nodes into primitive ops for bound propagation.
///
/// Replaces each LSTM node with per-timestep cell operations. Pre-computed
/// weights (transposed W, R, combined bias) are stored directly in the weight
/// store; per-timestep computation nodes reference these pre-computed values.
///
/// Takes graph inputs and value_info separately from nodes to avoid borrow
/// conflicts (nodes are mutated, but graph metadata is read-only).
pub(super) fn lower_lstm_nodes(
    nodes: &mut Vec<onnx_proto::NodeProto>,
    weights: &mut WeightStore,
    graph_inputs: &[onnx_proto::ValueInfoProto],
    graph_value_info: &[onnx_proto::ValueInfoProto],
    inferred_shapes: &std::collections::HashMap<String, Vec<i64>>,
) {
    if !nodes.iter().any(|n| n.op_type == "LSTM") {
        return;
    }

    let mut lowered = Vec::with_capacity(nodes.len());

    for node in nodes.drain(..) {
        if node.op_type != "LSTM" {
            lowered.push(node);
            continue;
        }

        match unroll_lstm_node(
            &node,
            weights,
            graph_inputs,
            graph_value_info,
            inferred_shapes,
        ) {
            Ok(new_nodes) => {
                info!(
                    "Unrolled LSTM '{}' into {} primitive nodes",
                    node.name,
                    new_nodes.len()
                );
                lowered.extend(new_nodes);
            }
            Err(reason) => {
                warn!(
                    "Cannot unroll LSTM '{}': {}. Keeping original node.",
                    node.name, reason
                );
                lowered.push(node);
            }
        }
    }

    *nodes = lowered;
}

/// Unroll a single LSTM node into primitive operations.
fn unroll_lstm_node(
    node: &onnx_proto::NodeProto,
    weights: &mut WeightStore,
    graph_inputs: &[onnx_proto::ValueInfoProto],
    graph_value_info: &[onnx_proto::ValueInfoProto],
    inferred_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<Vec<onnx_proto::NodeProto>, String> {
    let config = read_lstm_config(
        node,
        graph_inputs,
        graph_value_info,
        weights,
        inferred_shapes,
    )?;

    let mut new_nodes = Vec::new();

    // If layout=0, add Transpose: [seq, batch, input] → [batch, seq, input]
    let actual_x_name = add_layout_transpose(&config, node, &mut new_nodes);

    if config.num_directions == 1 {
        unroll_forward_only(&config, node, weights, &actual_x_name, &mut new_nodes)?;
    } else {
        unroll_bidirectional(&config, node, weights, &actual_x_name, &mut new_nodes)?;
    }

    debug!(
        "LSTM '{}' unrolled: {} timesteps × {} directions, {} nodes",
        node.name,
        config.seq_len,
        config.num_directions,
        new_nodes.len()
    );
    Ok(new_nodes)
}

/// Unroll a forward-only LSTM (num_directions=1).
fn unroll_forward_only(
    config: &LstmConfig,
    node: &onnx_proto::NodeProto,
    weights: &mut WeightStore,
    actual_x_name: &str,
    new_nodes: &mut Vec<onnx_proto::NodeProto>,
) -> Result<(), String> {
    let wn = precompute_lstm_weights(config, node, weights, 0, &config.base)?;
    store_lstm_slice_params(config, weights, &config.base);

    let result = unroll_direction_forward(config, &wn, actual_x_name, weights, new_nodes);
    cell::wire_outputs(
        config,
        &result.all_h,
        &result.final_h,
        &result.final_c,
        new_nodes,
    );
    Ok(())
}

/// Unroll a bidirectional LSTM (num_directions=2).
///
/// Forward direction processes t=0→T-1, reverse processes t=T-1→0.
/// Outputs are concatenated along the direction dimension per the ONNX spec.
fn unroll_bidirectional(
    config: &LstmConfig,
    node: &onnx_proto::NodeProto,
    weights: &mut WeightStore,
    actual_x_name: &str,
    new_nodes: &mut Vec<onnx_proto::NodeProto>,
) -> Result<(), String> {
    let fwd_base = format!("{}_fwd", config.base);
    let rev_base = format!("{}_rev", config.base);

    let fwd_wn = precompute_lstm_weights(config, node, weights, 0, &fwd_base)?;
    store_lstm_slice_params(config, weights, &fwd_base);

    let rev_wn = precompute_lstm_weights(config, node, weights, 1, &rev_base)?;
    store_lstm_slice_params(config, weights, &rev_base);

    // Forward direction: t = 0, 1, ..., T-1
    let fwd_config = LstmConfig {
        base: fwd_base,
        ..clone_config(config)
    };
    let fwd = unroll_direction_forward(&fwd_config, &fwd_wn, actual_x_name, weights, new_nodes);

    // Reverse direction: t = T-1, T-2, ..., 0
    let rev_config = LstmConfig {
        base: rev_base,
        ..clone_config(config)
    };
    let rev = unroll_direction_reverse(&rev_config, &rev_wn, actual_x_name, weights, new_nodes);

    cell_bidirectional::wire_bidirectional_outputs(config, &fwd, &rev, new_nodes);
    Ok(())
}

/// Unroll one direction in forward order (t=0→T-1).
fn unroll_direction_forward(
    config: &LstmConfig,
    wn: &LstmWeightNames,
    actual_x_name: &str,
    weights: &mut WeightStore,
    new_nodes: &mut Vec<onnx_proto::NodeProto>,
) -> DirectionResult {
    let mut prev_h = wn.h0.clone();
    let mut prev_c = wn.c0.clone();
    let mut all_h = Vec::with_capacity(config.seq_len);

    for t in 0..config.seq_len {
        let (h_t, c_t) = cell::generate_timestep_nodes(
            t,
            config,
            wn,
            actual_x_name,
            (&prev_h, &prev_c),
            weights,
            new_nodes,
        );
        all_h.push(h_t.clone());
        prev_h = h_t;
        prev_c = c_t;
    }

    DirectionResult {
        all_h,
        final_h: prev_h,
        final_c: prev_c,
    }
}

/// Unroll one direction in reverse order (t=T-1→0).
///
/// The reverse direction processes the input sequence backward. The cell
/// computation is identical to forward; only the timestep iteration order
/// differs. The output `all_h` is indexed by **original** timestep position
/// so that Y output concatenation aligns forward and reverse at each t.
fn unroll_direction_reverse(
    config: &LstmConfig,
    wn: &LstmWeightNames,
    actual_x_name: &str,
    weights: &mut WeightStore,
    new_nodes: &mut Vec<onnx_proto::NodeProto>,
) -> DirectionResult {
    let mut prev_h = wn.h0.clone();
    let mut prev_c = wn.c0.clone();
    let mut all_h = vec![String::new(); config.seq_len];

    for step in 0..config.seq_len {
        let t = config.seq_len - 1 - step;
        let (h_t, c_t) = cell::generate_timestep_nodes(
            t,
            config,
            wn,
            actual_x_name,
            (&prev_h, &prev_c),
            weights,
            new_nodes,
        );
        all_h[t] = h_t.clone();
        prev_h = h_t;
        prev_c = c_t;
    }

    DirectionResult {
        all_h,
        final_h: prev_h,
        final_c: prev_c,
    }
}

/// Clone config fields for direction-specific configs.
fn clone_config(config: &LstmConfig) -> LstmConfig {
    LstmConfig {
        hidden_size: config.hidden_size,
        input_size: config.input_size,
        seq_len: config.seq_len,
        layout: config.layout,
        base: config.base.clone(),
        domain: config.domain.clone(),
        batch_size: config.batch_size,
        num_directions: config.num_directions,
        y_name: config.y_name.clone(),
        y_h_name: config.y_h_name.clone(),
        y_c_name: config.y_c_name.clone(),
    }
}

/// Add a Transpose node if layout=0, returning the effective X tensor name.
fn add_layout_transpose(
    config: &LstmConfig,
    node: &onnx_proto::NodeProto,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) -> String {
    let x_name = node.input.first().map(String::as_str).unwrap_or("");
    if config.layout == 0 {
        let bf_name = format!("{}__lstm_x_bf", config.base);
        nodes.push(node_builder::make_node(
            "Transpose",
            &[x_name],
            &[&bf_name],
            &format!("{}__lstm_transpose_to_bf", config.base),
            &config.domain,
            vec![node_builder::make_ints_attr("perm", &[1, 0, 2])],
        ));
        bf_name
    } else {
        x_name.to_string()
    }
}

/// Extract LSTM configuration from a node's attributes and graph metadata.
fn read_lstm_config(
    node: &onnx_proto::NodeProto,
    graph_inputs: &[onnx_proto::ValueInfoProto],
    graph_value_info: &[onnx_proto::ValueInfoProto],
    weights: &WeightStore,
    inferred_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<LstmConfig, String> {
    let hidden_size = node
        .attribute
        .iter()
        .find(|a| a.name == "hidden_size")
        .map(|a| a.i as usize)
        .ok_or("missing hidden_size attribute")?;

    let direction = node
        .attribute
        .iter()
        .find(|a| a.name == "direction")
        .and_then(|a| std::str::from_utf8(&a.s).ok())
        .unwrap_or("forward");
    let num_directions = match direction {
        "forward" => 1,
        "bidirectional" => 2,
        other => return Err(format!("unsupported LSTM direction '{other}'")),
    };

    let layout = node
        .attribute
        .iter()
        .find(|a| a.name == "layout")
        .map(|a| a.i)
        .unwrap_or(0);

    let x_name = node
        .input
        .first()
        .filter(|s| !s.is_empty())
        .ok_or("LSTM missing X input")?;
    let x_shape = find_tensor_shape(
        x_name,
        graph_inputs,
        graph_value_info,
        weights,
        inferred_shapes,
    )
    .ok_or_else(|| format!("cannot determine shape of LSTM input '{x_name}'"))?;
    if x_shape.len() != 3 {
        return Err(format!(
            "expected 3D input for LSTM, got shape {:?}",
            x_shape
        ));
    }

    let (seq_len, _batch_size, input_size) = if layout == 0 {
        (x_shape[0], x_shape[1], x_shape[2])
    } else {
        (x_shape[1], x_shape[0], x_shape[2])
    };
    if seq_len <= 0 {
        return Err(format!(
            "LSTM requires concrete sequence length, got {seq_len}"
        ));
    }

    let base = if node.name.is_empty() {
        node.output
            .first()
            .map(String::as_str)
            .unwrap_or("lstm")
            .to_string()
    } else {
        node.name.clone()
    };

    let batch_size = if _batch_size > 0 {
        _batch_size as usize
    } else {
        1
    };

    Ok(LstmConfig {
        hidden_size,
        input_size: input_size as usize,
        seq_len: seq_len as usize,
        layout,
        base,
        domain: node.domain.clone(),
        batch_size,
        num_directions,
        y_name: node.output.first().filter(|s| !s.is_empty()).cloned(),
        y_h_name: node.output.get(1).filter(|s| !s.is_empty()).cloned(),
        y_c_name: node.output.get(2).filter(|s| !s.is_empty()).cloned(),
    })
}
