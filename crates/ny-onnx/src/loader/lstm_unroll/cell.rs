// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-timestep LSTM cell node generation and output wiring.
//!
//! Each function generates ONNX nodes for one logical step of the LSTM cell
//! computation. Functions are kept under 80 lines per the code quality gate.

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

use super::node_builder::{make_int_attr, make_ints_attr, make_node, make_node_variadic};
use super::{LstmConfig, LstmWeightNames};

/// Generate all nodes for one LSTM timestep. Returns `(h_t_name, c_t_name)`.
///
/// `prev_state` is `(prev_h_name, prev_c_name)`.
pub(super) fn generate_timestep_nodes(
    t: usize,
    config: &LstmConfig,
    wn: &LstmWeightNames,
    actual_x_name: &str,
    prev_state: (&str, &str),
    weights: &mut WeightStore,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) -> (String, String) {
    let (prev_h, prev_c) = prev_state;
    let ts = format!("t{t}");
    let base = &config.base;
    let domain = &config.domain;

    store_timestep_params(t, base, weights);
    let x_t = extract_timestep_input(t, &ts, base, domain, actual_x_name, wn, nodes);
    let gates = compute_gate_pre_activations(&ts, base, domain, &x_t, prev_h, wn, nodes);
    let gate_slices = slice_gates(&ts, base, domain, &gates, wn, nodes);
    let (i_t, f_t, o_t, c_input) = apply_gate_activations(&ts, base, domain, &gate_slices, nodes);
    let c_t = update_cell_state(&ts, base, domain, (&f_t, prev_c), (&i_t, &c_input), nodes);
    let h_t = compute_hidden_state(&ts, base, domain, &o_t, &c_t, nodes);

    (h_t, c_t)
}

/// Store start/end weight tensors for this timestep's time-axis slice.
fn store_timestep_params(t: usize, base: &str, weights: &mut WeightStore) {
    let ts = format!("t{t}");
    weights.insert(
        format!("{base}__lstm_{ts}_start"),
        ArrayD::from_elem(IxDyn(&[1]), t as f32),
    );
    weights.insert(
        format!("{base}__lstm_{ts}_end"),
        ArrayD::from_elem(IxDyn(&[1]), (t + 1) as f32),
    );
}

/// Slice+Reshape to extract `x_t: [batch, input]` from `X: [batch, seq, input]`.
///
/// Uses Reshape instead of Squeeze because the propagation framework forbids
/// Squeeze at axis 0. After batch stripping, the time axis IS axis 0,
/// so we use Reshape([batch, input]) which the converter strips to [input].
fn extract_timestep_input(
    t: usize,
    ts: &str,
    base: &str,
    domain: &str,
    actual_x_name: &str,
    wn: &LstmWeightNames,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) -> String {
    let slice_start = format!("{base}__lstm_{ts}_start");
    let slice_end = format!("{base}__lstm_{ts}_end");
    let x_t_3d = format!("{base}__lstm_{ts}_x3d");
    nodes.push(make_node(
        "Slice",
        &[
            actual_x_name,
            &slice_start,
            &slice_end,
            &wn.time_axis,
            &wn.time_step,
        ],
        &[&x_t_3d],
        &format!("{base}__lstm_{ts}_slice_x"),
        domain,
        vec![],
    ));

    let x_t = format!("{base}__lstm_t{t}_x");
    nodes.push(make_node(
        "Reshape",
        &[&x_t_3d, &wn.x_reshape],
        &[&x_t],
        &format!("{base}__lstm_{ts}_reshape_x"),
        domain,
        vec![],
    ));
    x_t
}

/// Compute gate pre-activations: `gates = x_t @ W_T + h_{t-1} @ R_T + bias`.
///
/// For t=0, `prev_h` matches `wn.h0` (a stored weight). Both h0 and R_T are
/// weights, so emitting `MatMul(h0, R_T)` would give the graph builder a
/// binary op with zero activation inputs. Instead, we use the pre-computed
/// `wn.h0_hr` weight directly and emit only an Add (1 activation + 1 weight).
fn compute_gate_pre_activations(
    ts: &str,
    base: &str,
    domain: &str,
    x_t: &str,
    prev_h: &str,
    wn: &LstmWeightNames,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) -> String {
    let xw = format!("{base}__lstm_{ts}_xW");
    nodes.push(make_node(
        "MatMul",
        &[x_t, &wn.w_t],
        &[&xw],
        &format!("{base}__lstm_{ts}_matmul_xW"),
        domain,
        vec![],
    ));

    // For t=0, use pre-computed h0 @ R_T (both are weights → stored result).
    // For t>0, prev_h is an activation → emit the MatMul normally.
    let hr = if prev_h == wn.h0 {
        wn.h0_hr.clone()
    } else {
        let hr = format!("{base}__lstm_{ts}_hR");
        nodes.push(make_node(
            "MatMul",
            &[prev_h, &wn.r_t],
            &[&hr],
            &format!("{base}__lstm_{ts}_matmul_hR"),
            domain,
            vec![],
        ));
        hr
    };

    let gates_pre = format!("{base}__lstm_{ts}_gates_pre");
    nodes.push(make_node(
        "Add",
        &[&xw, &hr],
        &[&gates_pre],
        &format!("{base}__lstm_{ts}_add_gates"),
        domain,
        vec![],
    ));

    let gates = format!("{base}__lstm_{ts}_gates");
    nodes.push(make_node(
        "Add",
        &[&gates_pre, &wn.bias],
        &[&gates],
        &format!("{base}__lstm_{ts}_add_bias"),
        domain,
        vec![],
    ));
    gates
}

/// Slice the combined gate tensor `[batch, 4H]` into 4 gates of `[batch, H]`.
/// ONNX gate order: [i, o, f, c].
fn slice_gates(
    ts: &str,
    base: &str,
    domain: &str,
    gates: &str,
    wn: &LstmWeightNames,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) -> [String; 4] {
    let labels = ["i", "o", "f", "c"];
    let mut outputs = Vec::with_capacity(4);
    for label in &labels {
        let gate_out = format!("{base}__lstm_{ts}_gate_{label}");
        let start_w = format!("{base}__lstm_gate_{label}_start");
        let end_w = format!("{base}__lstm_gate_{label}_end");
        nodes.push(make_node(
            "Slice",
            &[gates, &start_w, &end_w, &wn.gate_axis, &wn.gate_step],
            &[&gate_out],
            &format!("{base}__lstm_{ts}_slice_{label}"),
            domain,
            vec![],
        ));
        outputs.push(gate_out);
    }
    // Safe: always exactly 4 elements
    [
        outputs[0].clone(),
        outputs[1].clone(),
        outputs[2].clone(),
        outputs[3].clone(),
    ]
}

/// Apply gate activations: Sigmoid(i), Sigmoid(f), Sigmoid(o), Tanh(c_input).
fn apply_gate_activations(
    ts: &str,
    base: &str,
    domain: &str,
    gate_slices: &[String; 4],
    nodes: &mut Vec<onnx_proto::NodeProto>,
) -> (String, String, String, String) {
    let i_t = format!("{base}__lstm_{ts}_i");
    nodes.push(make_node(
        "Sigmoid",
        &[&gate_slices[0]],
        &[&i_t],
        &format!("{base}__lstm_{ts}_sigmoid_i"),
        domain,
        vec![],
    ));

    let f_t = format!("{base}__lstm_{ts}_f");
    nodes.push(make_node(
        "Sigmoid",
        &[&gate_slices[2]],
        &[&f_t],
        &format!("{base}__lstm_{ts}_sigmoid_f"),
        domain,
        vec![],
    ));

    let o_t = format!("{base}__lstm_{ts}_o");
    nodes.push(make_node(
        "Sigmoid",
        &[&gate_slices[1]],
        &[&o_t],
        &format!("{base}__lstm_{ts}_sigmoid_o"),
        domain,
        vec![],
    ));

    let c_input = format!("{base}__lstm_{ts}_c_in");
    nodes.push(make_node(
        "Tanh",
        &[&gate_slices[3]],
        &[&c_input],
        &format!("{base}__lstm_{ts}_tanh_c"),
        domain,
        vec![],
    ));

    (i_t, f_t, o_t, c_input)
}

/// Cell state update: `C_t = f_t * C_{t-1} + i_t * c_input`.
///
/// `forget_pair` is `(f_t, prev_c)`, `input_pair` is `(i_t, c_input)`.
fn update_cell_state(
    ts: &str,
    base: &str,
    domain: &str,
    forget_pair: (&str, &str),
    input_pair: (&str, &str),
    nodes: &mut Vec<onnx_proto::NodeProto>,
) -> String {
    let (f_t, prev_c) = forget_pair;
    let (i_t, c_input) = input_pair;

    let fc = format!("{base}__lstm_{ts}_fc");
    nodes.push(make_node(
        "Mul",
        &[f_t, prev_c],
        &[&fc],
        &format!("{base}__lstm_{ts}_mul_fc"),
        domain,
        vec![],
    ));

    let ic = format!("{base}__lstm_{ts}_ic");
    nodes.push(make_node(
        "Mul",
        &[i_t, c_input],
        &[&ic],
        &format!("{base}__lstm_{ts}_mul_ic"),
        domain,
        vec![],
    ));

    let c_t = format!("{base}__lstm_{ts}_C");
    nodes.push(make_node(
        "Add",
        &[&fc, &ic],
        &[&c_t],
        &format!("{base}__lstm_{ts}_add_cell"),
        domain,
        vec![],
    ));
    c_t
}

/// Hidden state: `H_t = o_t * tanh(C_t)`.
fn compute_hidden_state(
    ts: &str,
    base: &str,
    domain: &str,
    o_t: &str,
    c_t: &str,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) -> String {
    let tanh_c = format!("{base}__lstm_{ts}_tanh_C");
    nodes.push(make_node(
        "Tanh",
        &[c_t],
        &[&tanh_c],
        &format!("{base}__lstm_{ts}_tanh_cell"),
        domain,
        vec![],
    ));

    let h_t = format!("{base}__lstm_{ts}_H");
    nodes.push(make_node(
        "Mul",
        &[o_t, &tanh_c],
        &[&h_t],
        &format!("{base}__lstm_{ts}_mul_h"),
        domain,
        vec![],
    ));
    h_t
}

// --- Output wiring ---

/// Wire LSTM output tensors (Y, Y_h, Y_c) from unrolled hidden/cell states.
pub(super) fn wire_outputs(
    config: &LstmConfig,
    all_h_names: &[String],
    prev_h: &str,
    prev_c: &str,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) {
    wire_final_states(config, prev_h, prev_c, nodes);
    if let Some(ref y_out) = config.y_name {
        wire_sequence_output(config, all_h_names, y_out, nodes);
    }
}

/// Unsqueeze last hidden/cell states to add direction dim for Y_h, Y_c.
fn wire_final_states(
    config: &LstmConfig,
    prev_h: &str,
    prev_c: &str,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) {
    let base = &config.base;
    let domain = &config.domain;
    let direction_axis = if config.layout == 0 { 0 } else { 1 };

    if let Some(ref y_h_out) = config.y_h_name {
        nodes.push(make_node(
            "Unsqueeze",
            &[prev_h],
            &[y_h_out],
            &format!("{base}__lstm_unsqueeze_yh"),
            domain,
            vec![make_ints_attr("axes", &[direction_axis])],
        ));
    }

    if let Some(ref y_c_out) = config.y_c_name {
        nodes.push(make_node(
            "Unsqueeze",
            &[prev_c],
            &[y_c_out],
            &format!("{base}__lstm_unsqueeze_yc"),
            domain,
            vec![make_ints_attr("axes", &[direction_axis])],
        ));
    }
}

/// Build Y output: Unsqueeze each H_t to add direction+time dims, then Concat.
fn wire_sequence_output(
    config: &LstmConfig,
    all_h_names: &[String],
    y_out: &str,
    nodes: &mut Vec<onnx_proto::NodeProto>,
) {
    let base = &config.base;
    let domain = &config.domain;
    let layout = config.layout;
    let mut concat_inputs = Vec::new();

    for (t, h_name) in all_h_names.iter().enumerate() {
        let (axis1, axis2) = if layout == 0 { (0, 0) } else { (1, 1) };

        let us1 = format!("{base}__lstm_y_t{t}_us1");
        nodes.push(make_node(
            "Unsqueeze",
            &[h_name],
            &[&us1],
            &format!("{base}__lstm_y_t{t}_unsqueeze1"),
            domain,
            vec![make_ints_attr("axes", &[axis1])],
        ));

        let us2 = format!("{base}__lstm_y_t{t}_us2");
        nodes.push(make_node(
            "Unsqueeze",
            &[&us1],
            &[&us2],
            &format!("{base}__lstm_y_t{t}_unsqueeze2"),
            domain,
            vec![make_ints_attr("axes", &[axis2])],
        ));

        concat_inputs.push(us2);
    }

    let concat_axis = if layout == 0 { 0 } else { 1 };
    let concat_input_refs: Vec<&str> = concat_inputs.iter().map(|s| s.as_str()).collect();
    nodes.push(make_node_variadic(
        "Concat",
        &concat_input_refs,
        &[y_out],
        &format!("{base}__lstm_concat_Y"),
        domain,
        vec![make_int_attr("axis", concat_axis)],
    ));
}
