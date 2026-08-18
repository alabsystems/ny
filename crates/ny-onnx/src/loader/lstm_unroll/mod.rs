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
//! - No sequence lengths or peepholes (non-empty optional inputs are rejected)
//! - Default activations (Sigmoid, Tanh, Tanh)
//! - Requires concrete (non-dynamic) sequence length
//! - layout=0 (seq_first) and layout=1 (batch_first)
//! - Bidirectional final-state outputs only; the four-dimensional sequence
//!   output is rejected until it can be preserved exactly
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
use std::collections::HashSet;
use tracing::{debug, info, warn};

use super::const_fold::is_standard_onnx_domain;
use weights::{
    find_tensor_shape, precompute_lstm_weights, prepare_lstm_weights, store_lstm_slice_params,
};

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
    if !nodes
        .iter()
        .any(|n| n.op_type == "LSTM" && is_standard_onnx_domain(&n.domain))
    {
        return;
    }

    // Generated constants live only in WeightStore, not GraphProto.initializer.
    // Reserve every authored value name up front so a synthesized constant can
    // never shadow a runtime input or another node's value.
    let mut authored_value_names: HashSet<String> = graph_inputs
        .iter()
        .chain(graph_value_info)
        .map(|value| value.name.clone())
        .filter(|name| !name.is_empty())
        .collect();
    for authored_node in nodes.iter() {
        authored_value_names.extend(
            authored_node
                .input
                .iter()
                .chain(&authored_node.output)
                .filter(|name| !name.is_empty())
                .cloned(),
        );
    }

    let mut lowered = Vec::with_capacity(nodes.len());

    for node in nodes.drain(..) {
        if node.op_type != "LSTM" || !is_standard_onnx_domain(&node.domain) {
            lowered.push(node);
            continue;
        }

        match unroll_lstm_node(
            &node,
            weights,
            graph_inputs,
            graph_value_info,
            inferred_shapes,
            &authored_value_names,
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
    authored_value_names: &HashSet<String>,
) -> Result<Vec<onnx_proto::NodeProto>, String> {
    let config = read_lstm_config(
        node,
        graph_inputs,
        graph_value_info,
        weights,
        inferred_shapes,
        authored_value_names,
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

    // Prepare both directions before committing either one. If an exact-real
    // arithmetic certificate fails in the reverse direction, the authored
    // graph remains untouched rather than receiving a partial lowering.
    let fwd_prepared = prepare_lstm_weights(config, node, weights, 0, &fwd_base)?;
    let rev_prepared = prepare_lstm_weights(config, node, weights, 1, &rev_base)?;
    let fwd_wn = fwd_prepared.store(weights);
    store_lstm_slice_params(config, weights, &fwd_base);

    let rev_wn = rev_prepared.store(weights);
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
    authored_value_names: &HashSet<String>,
) -> Result<LstmConfig, String> {
    validate_lstm_signature(node)?;
    let (hidden_size, num_directions, layout) = read_lstm_attributes(node)?;

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

    let (seq_len, batch_size, input_size) = if layout == 0 {
        (x_shape[0], x_shape[1], x_shape[2])
    } else {
        (x_shape[1], x_shape[0], x_shape[2])
    };
    if seq_len <= 0 {
        return Err(format!(
            "LSTM requires concrete sequence length, got {seq_len}"
        ));
    }
    if batch_size <= 0 {
        return Err(format!(
            "LSTM requires a concrete positive batch size, got {batch_size}"
        ));
    }
    if input_size <= 0 {
        return Err(format!(
            "LSTM requires a concrete positive input size, got {input_size}"
        ));
    }

    let seq_len = usize::try_from(seq_len)
        .map_err(|_| "LSTM sequence length does not fit usize".to_string())?;
    let batch_size = usize::try_from(batch_size)
        .map_err(|_| "LSTM batch size does not fit usize".to_string())?;
    let input_size = usize::try_from(input_size)
        .map_err(|_| "LSTM input size does not fit usize".to_string())?;
    validate_exact_index_dimensions(hidden_size, seq_len, batch_size, input_size)?;

    let base = if node.name.is_empty() {
        node.output
            .first()
            .map(String::as_str)
            .unwrap_or("lstm")
            .to_string()
    } else {
        node.name.clone()
    };

    let config = LstmConfig {
        hidden_size,
        input_size,
        seq_len,
        layout,
        base,
        domain: node.domain.clone(),
        batch_size,
        num_directions,
        y_name: node.output.first().filter(|s| !s.is_empty()).cloned(),
        y_h_name: node.output.get(1).filter(|s| !s.is_empty()).cloned(),
        y_c_name: node.output.get(2).filter(|s| !s.is_empty()).cloned(),
    };

    if config.num_directions == 2 && config.y_name.is_some() {
        return Err(
            "bidirectional LSTM sequence output Y requires exact four-dimensional ONNX layout, which is not yet supported"
                .to_string(),
        );
    }

    validate_synthesized_weight_names(&config, weights, authored_value_names)?;
    Ok(config)
}

fn validate_synthesized_weight_names(
    config: &LstmConfig,
    weights: &WeightStore,
    authored_value_names: &HashSet<String>,
) -> Result<(), String> {
    let direction_bases = if config.num_directions == 1 {
        vec![config.base.clone()]
    } else {
        vec![
            format!("{}_fwd", config.base),
            format!("{}_rev", config.base),
        ]
    };
    let mut synthesized = Vec::new();
    for base in direction_bases {
        for suffix in [
            "W_T",
            "R_T",
            "bias",
            "h0",
            "c0",
            "h0_hR",
            "x_reshape",
            "gate_axis",
            "gate_step",
            "time_axis",
            "time_step",
        ] {
            synthesized.push(format!("{base}__lstm_{suffix}"));
        }
        for label in ["i", "o", "f", "c"] {
            synthesized.push(format!("{base}__lstm_gate_{label}_start"));
            synthesized.push(format!("{base}__lstm_gate_{label}_end"));
        }
        for timestep in 0..config.seq_len {
            synthesized.push(format!("{base}__lstm_t{timestep}_start"));
            synthesized.push(format!("{base}__lstm_t{timestep}_end"));
        }
    }

    if let Some(name) = synthesized
        .into_iter()
        .find(|name| weights.contains_key(name) || authored_value_names.contains(name))
    {
        return Err(format!(
            "LSTM lowering would shadow authored value '{name}'"
        ));
    }
    Ok(())
}

fn validate_lstm_signature(node: &onnx_proto::NodeProto) -> Result<(), String> {
    if !(3..=8).contains(&node.input.len()) {
        return Err(format!(
            "LSTM requires between three and eight inputs, got {}",
            node.input.len()
        ));
    }
    for (index, label) in [(0, "X"), (1, "W"), (2, "R")] {
        if node.input.get(index).is_none_or(String::is_empty) {
            return Err(format!("LSTM missing required {label} input[{index}]"));
        }
    }
    if node.input.get(4).is_some_and(|name| !name.is_empty()) {
        return Err("LSTM sequence_lens input[4] is not supported".to_string());
    }
    if node.input.get(7).is_some_and(|name| !name.is_empty()) {
        return Err("LSTM peephole P input[7] is not supported".to_string());
    }
    if node.output.len() > 3 {
        return Err(format!(
            "LSTM permits at most three outputs, got {}",
            node.output.len()
        ));
    }
    Ok(())
}

fn read_lstm_attributes(node: &onnx_proto::NodeProto) -> Result<(usize, usize, i64), String> {
    let mut names = HashSet::new();
    for attribute in &node.attribute {
        if !names.insert(attribute.name.as_str()) {
            return Err(format!(
                "LSTM has duplicate '{}' attributes",
                attribute.name
            ));
        }
    }

    let mut hidden_size = None;
    let mut num_directions = 1_usize;
    let mut layout = 0_i64;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "hidden_size" => {
                require_attribute_type(attribute, onnx_proto::attribute_type::INT)?;
                if attribute.i_value() <= 0 {
                    return Err(format!(
                        "LSTM hidden_size must be positive, got {}",
                        attribute.i_value()
                    ));
                }
                hidden_size = Some(usize::try_from(attribute.i_value()).map_err(|_| {
                    format!(
                        "LSTM hidden_size {} does not fit usize",
                        attribute.i_value()
                    )
                })?);
            }
            "direction" => {
                require_attribute_type(attribute, onnx_proto::attribute_type::STRING)?;
                let direction = std::str::from_utf8(attribute.s_value())
                    .map_err(|_| "LSTM direction must be valid UTF-8".to_string())?;
                num_directions = match direction {
                    "forward" => 1,
                    "bidirectional" => 2,
                    other => return Err(format!("unsupported LSTM direction '{other}'")),
                };
            }
            "layout" => {
                require_attribute_type(attribute, onnx_proto::attribute_type::INT)?;
                if !matches!(attribute.i_value(), 0 | 1) {
                    return Err(format!(
                        "LSTM layout must be 0 or 1, got {}",
                        attribute.i_value()
                    ));
                }
                layout = attribute.i_value();
            }
            "input_forget" => {
                require_attribute_type(attribute, onnx_proto::attribute_type::INT)?;
                if attribute.i_value() != 0 {
                    return Err(format!(
                        "unsupported LSTM input_forget={}, only the default 0 is supported",
                        attribute.i_value()
                    ));
                }
            }
            "activations" => {
                require_attribute_type(attribute, onnx_proto::attribute_type::STRINGS)?;
                return Err(
                    "explicit LSTM activations are not supported; omit the attribute to use Sigmoid/Tanh/Tanh"
                        .to_string(),
                );
            }
            "activation_alpha" | "activation_beta" => {
                require_attribute_type(attribute, onnx_proto::attribute_type::FLOATS)?;
                return Err(format!("LSTM {} is not supported", attribute.name));
            }
            "clip" => {
                require_attribute_type(attribute, onnx_proto::attribute_type::FLOAT)?;
                return Err("LSTM clip is not supported".to_string());
            }
            other => return Err(format!("unknown LSTM attribute '{other}'")),
        }
    }

    Ok((
        hidden_size.ok_or("missing hidden_size attribute")?,
        num_directions,
        layout,
    ))
}

fn require_attribute_type(
    attribute: &onnx_proto::AttributeProto,
    expected: i32,
) -> Result<(), String> {
    if attribute.r#type != expected {
        return Err(format!(
            "LSTM attribute '{}' has type {}, expected {}",
            attribute.name, attribute.r#type, expected
        ));
    }
    Ok(())
}

fn validate_exact_index_dimensions(
    hidden_size: usize,
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
) -> Result<(), String> {
    const MAX_EXACT_F32_INTEGER: usize = 1 << 24;
    let gate_size = hidden_size
        .checked_mul(4)
        .ok_or("LSTM gate dimension overflows usize")?;
    for (label, value) in [
        ("sequence length", seq_len),
        ("batch size", batch_size),
        ("input size", input_size),
        ("gate size", gate_size),
    ] {
        if value > MAX_EXACT_F32_INTEGER {
            return Err(format!(
                "LSTM {label} {value} cannot be represented exactly by synthesized FLOAT slice/shape constants"
            ));
        }
    }
    Ok(())
}
