// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Recurrent family: `Lstm`.
//!
//! Port of NN's LSTM cell decomposition
//! (`nn-verify/src/trace_to_graph_layerspec_lstm.rs` + its `_helpers.rs`).
//! One `TraceOp::Lstm` decomposes into ~21 primitive LayerSpecs:
//!
//!   4 Linear (ih) + 4 Linear (hh) + 4 Add + 3 Sigmoid + 1 Tanh (g) +
//!   2 Mul (f*c, i*g) + 1 Add (c_new) + 1 Tanh (c_new) + 1 Mul (h_new)
//!   + 0-2 Linear (zero-state h_prev / c_prev — omitted when the trace
//!     provides non-zero initial state)
//!
//! PyTorch gate order: (i, f, g, o). The combined bias (`bias_ih + bias_hh`)
//! is split per gate and halved 50/50 between the ih and hh Linear paths,
//! matching NN's additive decomposition exactly.
//!
//! Two deliberate, semantics-preserving deviations from NN's LayerSpec file
//! (both verified against NN's direct-GraphNode ground truth in
//! `graph_tensor_lstm.rs`, where `LinearLayer::new(w, bias)` receives the
//! bias directly):
//!
//! 1. **Bias wiring.** NN's LayerSpec emitter inserts each per-gate bias into
//!    the weight store as `{layer}_bias` but never lists it in the Linear
//!    spec's `inputs`, and `ny_build`'s `convert_linear` only reads a bias
//!    from `spec.inputs[2]` — so on the LayerSpec path the gate biases are
//!    silently dropped at graph-build time. Building bounds for a
//!    bias-stripped network is unsound (a "hold" would not transfer to the
//!    real LSTM), so this port wires every bias it inserts as the Linear
//!    spec's third input, restoring the decomposition NN's comments and
//!    direct-GraphNode path both specify.
//! 2. **Initial-state resolution.** NN resolves non-zero initial state from
//!    `input_tensors[1]`/`[2]` by position; the schema carries the recorded
//!    state node ids (`initial_hidden` / `initial_cell`), so this port
//!    resolves those ids through `node_names` (`super::resolve_input`) —
//!    identical on well-formed traces (the recorder puts the same ids at
//!    inputs 1 and 2) and fail-closed instead of silently zero-injecting on
//!    malformed ones.
//!
//! NN reads the intermediate tensor shape from `ctx.tensor_shapes[output]`,
//! pre-inserted by its dispatch; the bridge's dispatch inserts that entry
//! *after* translation, so this port derives the same value directly from the
//! traced `node.output_shape`. Single-cell LSTM: `[H]`. Sequence: `[S, B, H]`.
//!
//! Zero-state injection (`Linear(zeros_w, zeros_b)` fed by the LSTM input) is
//! sound for first-timestep verification where h_0 = 0, c_0 = 0.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_build::{AttributeValue, LayerSpec};
use ny_core::{LayerType, NyError, Result};

use crate::schema::{TraceNode, TraceOp, WeightPayload};

use super::{first_input, resolve_input, shape_to_i64, simple_spec, weight_f32, Ctx, NodeOutput};

/// Gate names for the 4 LSTM gates (PyTorch order: i, f, g, o).
const GATE_NAMES: [&str; 4] = ["i", "f", "g", "o"];

/// Translate a recurrent-family op (`Lstm`) node.
///
/// Decomposes the single-step LSTM cell into the primitive-layer graph NN
/// emits (see module docs). `input_tensors[0]` is the LSTM data input x;
/// initial hidden/cell state comes from the recorded `initial_hidden` /
/// `initial_cell` node ids when present, else is zero-injected.
///
/// # Errors
///
/// Fail-closed [`NyError::ModelLoad`] on any weight/bias shape or data
/// mismatch (wrong rank, rows != 4*hidden_size, length mismatches,
/// placeholder or non-finite data) and [`NyError::InternalError`] on
/// dimension overflow or a dangling initial-state node id.
pub(super) fn translate_lstm(
    node: &TraceNode,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let TraceOp::Lstm {
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        hidden_size,
        initial_hidden,
        initial_cell,
    } = &node.op
    else {
        return Err(NyError::InternalError(
            "translate_lstm called with non-LSTM op".to_string(),
        ));
    };
    let hidden_size = *hidden_size;

    // -- Weight validation (mirrors NN's checks; adds explicit rank/length
    //    checks where NN's flat indexing would panic instead of erroring) --
    if weight_ih.shape.len() != 2 || weight_hh.shape.len() != 2 {
        return Err(NyError::ModelLoad(format!(
            "Lstm: weight_ih/weight_hh must be 2-D, got {}-D and {}-D",
            weight_ih.shape.len(),
            weight_hh.shape.len()
        )));
    }
    let four_h = weight_ih.shape[0];
    let input_dim = weight_ih.shape[1];
    let expected_four_h = hidden_size.checked_mul(4).ok_or_else(|| {
        NyError::InternalError(format!(
            "Lstm: hidden_size {hidden_size} overflows when multiplied by 4"
        ))
    })?;
    if four_h != expected_four_h {
        return Err(NyError::ModelLoad(format!(
            "Lstm: weight_ih rows ({four_h}) != 4*hidden_size ({expected_four_h})"
        )));
    }
    if weight_hh.shape[0] != four_h || weight_hh.shape[1] != hidden_size {
        return Err(NyError::ModelLoad(format!(
            "Lstm: weight_hh shape [{}, {}] expected [{four_h}, {hidden_size}]",
            weight_hh.shape[0], weight_hh.shape[1]
        )));
    }

    let w_ih_data = checked_weight_data(weight_ih, four_h, input_dim, "Lstm weight_ih")?;
    let w_hh_data = checked_weight_data(weight_hh, four_h, hidden_size, "Lstm weight_hh")?;

    // Combine biases: bias = bias_ih + bias_hh (PyTorch convention).
    let combined_bias: Option<Vec<f32>> = match (bias_ih, bias_hh) {
        (Some(bih), Some(bhh)) => {
            let bih = checked_bias_data(bih, four_h, "Lstm bias_ih")?;
            let bhh = checked_bias_data(bhh, four_h, "Lstm bias_hh")?;
            Some(bih.iter().zip(&bhh).map(|(a, b)| a + b).collect())
        }
        (Some(b), None) => Some(checked_bias_data(b, four_h, "Lstm bias_ih")?),
        (None, Some(b)) => Some(checked_bias_data(b, four_h, "Lstm bias_hh")?),
        (None, None) => None,
    };

    // Intermediate tensor shape = the traced output shape (all gate ops —
    // Linear, Add, Sigmoid, Tanh, Mul — preserve it). NN reads this from
    // ctx.tensor_shapes (pre-inserted by its dispatch); derived directly here.
    let intermediate_shape = shape_to_i64(&node.output_shape, "Lstm output")?;

    let input_tensor = first_input(input_tensors, "Lstm")?;

    let mut specs: Vec<LayerSpec> = Vec::with_capacity(23);

    // Resolve hidden and cell state tensors: recorded initial-state node when
    // the trace carries one, zero-injection otherwise (h_0 = 0, c_0 = 0).
    let h_prev_tensor = match initial_hidden {
        Some(h_id) => resolve_input(h_id.get(), node_names)?,
        None => inject_zero_state(
            &mut specs,
            name,
            "h_prev",
            hidden_size,
            &intermediate_shape,
            input_dim,
            &input_tensor,
            ctx,
        )?,
    };
    let c_prev_tensor = match initial_cell {
        Some(c_id) => resolve_input(c_id.get(), node_names)?,
        None => inject_zero_state(
            &mut specs,
            name,
            "c_prev",
            hidden_size,
            &intermediate_shape,
            input_dim,
            &input_tensor,
            ctx,
        )?,
    };

    // -- Per-gate decomposition: ih Linear + hh Linear + Add ----------------
    let mut gate_sum_tensors = Vec::with_capacity(4);
    for (gate_idx, gate_label) in GATE_NAMES.iter().enumerate() {
        let start = gate_idx * hidden_size;

        // Per-gate weight sub-matrices [H, input_dim] and [H, hidden_size].
        let w_ih_gate = extract_submatrix(&w_ih_data, input_dim, start, hidden_size);
        let w_hh_gate = extract_submatrix(&w_hh_data, hidden_size, start, hidden_size);

        // Split the combined bias evenly between the ih and hh paths (NN's
        // additive decomposition: b/2 + b/2 = b after the gate-sum Add).
        let (bias_ih_gate, bias_hh_gate) = if let Some(ref bias) = combined_bias {
            let half: Vec<f32> = bias[start..start + hidden_size]
                .iter()
                .map(|v| v * 0.5)
                .collect();
            (Some(half.clone()), Some(half))
        } else {
            (None, None)
        };

        // ih Linear: W_ih[gate] @ x + bias/2
        let ih_out = push_gate_linear(
            &mut specs,
            &format!("{name}_{gate_label}_ih"),
            &w_ih_gate,
            hidden_size,
            input_dim,
            bias_ih_gate,
            &input_tensor,
            &intermediate_shape,
            ctx,
        )?;

        // hh Linear: W_hh[gate] @ h_prev + bias/2
        let hh_out = push_gate_linear(
            &mut specs,
            &format!("{name}_{gate_label}_hh"),
            &w_hh_gate,
            hidden_size,
            hidden_size,
            bias_hh_gate,
            &h_prev_tensor,
            &intermediate_shape,
            ctx,
        )?;

        // Sum: ih + hh
        let sum_name = format!("{name}_{gate_label}_sum");
        let sum_out = format!("{sum_name}_out");
        ctx.tensor_shapes
            .insert(sum_out.clone(), intermediate_shape.clone());
        specs.push(simple_spec(
            &sum_name,
            LayerType::Add,
            vec![ih_out, hh_out],
            &sum_out,
            HashMap::new(),
        ));
        gate_sum_tensors.push(sum_out);
    }

    // Gate activations: i = Sigmoid, f = Sigmoid, g = Tanh, o = Sigmoid.
    let i_gate_out = push_activation(
        &mut specs,
        name,
        "i_gate",
        LayerType::Sigmoid,
        &gate_sum_tensors[0],
        &intermediate_shape,
        ctx,
    );
    let f_gate_out = push_activation(
        &mut specs,
        name,
        "f_gate",
        LayerType::Sigmoid,
        &gate_sum_tensors[1],
        &intermediate_shape,
        ctx,
    );
    let g_cand_out = push_activation(
        &mut specs,
        name,
        "g_cand",
        LayerType::Tanh,
        &gate_sum_tensors[2],
        &intermediate_shape,
        ctx,
    );
    let o_gate_out = push_activation(
        &mut specs,
        name,
        "o_gate",
        LayerType::Sigmoid,
        &gate_sum_tensors[3],
        &intermediate_shape,
        ctx,
    );

    // Cell state update: c_new = f_gate * c_prev + i_gate * g_cand.
    let f_cell_name = format!("{name}_f_cell");
    let f_cell_out = format!("{f_cell_name}_out");
    ctx.tensor_shapes
        .insert(f_cell_out.clone(), intermediate_shape.clone());
    specs.push(simple_spec(
        &f_cell_name,
        LayerType::Mul,
        vec![f_gate_out, c_prev_tensor],
        &f_cell_out,
        HashMap::new(),
    ));

    let i_g_name = format!("{name}_i_g");
    let i_g_out = format!("{i_g_name}_out");
    ctx.tensor_shapes
        .insert(i_g_out.clone(), intermediate_shape.clone());
    specs.push(simple_spec(
        &i_g_name,
        LayerType::Mul,
        vec![i_gate_out, g_cand_out],
        &i_g_out,
        HashMap::new(),
    ));

    let c_new_name = format!("{name}_c_new");
    let c_new_out = format!("{c_new_name}_out");
    ctx.tensor_shapes
        .insert(c_new_out.clone(), intermediate_shape.clone());
    specs.push(simple_spec(
        &c_new_name,
        LayerType::Add,
        vec![f_cell_out, i_g_out],
        &c_new_out,
        HashMap::new(),
    ));

    // Hidden state output: h_new = o_gate * tanh(c_new).
    let tanh_c_name = format!("{name}_tanh_c");
    let tanh_c_out = format!("{tanh_c_name}_out");
    ctx.tensor_shapes
        .insert(tanh_c_out.clone(), intermediate_shape.clone());
    specs.push(simple_spec(
        &tanh_c_name,
        LayerType::Tanh,
        vec![c_new_out],
        &tanh_c_out,
        HashMap::new(),
    ));

    // Final output Mul uses the traced node's name/output tensor.
    ctx.tensor_shapes
        .entry(output_tensor.to_string())
        .or_insert(intermediate_shape);
    specs.push(simple_spec(
        name,
        LayerType::Mul,
        vec![o_gate_out, tanh_c_out],
        output_tensor,
        HashMap::new(),
    ));

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Local helpers (port of NN's trace_to_graph_layerspec_lstm_helpers.rs)
// ---------------------------------------------------------------------------

/// Validate + extract a 2-D weight payload's data, checking the element count
/// against the declared `[rows, cols]` shape (fail-closed where NN's flat
/// indexing would panic).
fn checked_weight_data(
    payload: &WeightPayload,
    rows: usize,
    cols: usize,
    context: &str,
) -> Result<Vec<f32>> {
    let data = weight_f32(payload, context)?;
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        NyError::InternalError(format!("{context}: {rows}x{cols} element count overflow"))
    })?;
    if data.len() != expected {
        return Err(NyError::ModelLoad(format!(
            "{context}: data length {} != {rows}x{cols} = {expected}",
            data.len()
        )));
    }
    Ok(data)
}

/// Validate + extract a 1-D bias payload's data of expected length `4*H`.
fn checked_bias_data(payload: &WeightPayload, four_h: usize, context: &str) -> Result<Vec<f32>> {
    let data = weight_f32(payload, context)?;
    if data.len() != four_h {
        return Err(NyError::ModelLoad(format!(
            "{context}: bias length ({}) != 4*H = {four_h}",
            data.len()
        )));
    }
    Ok(data)
}

/// Extract a `[num_rows, cols]` sub-matrix (rows `start_row..start_row+num_rows`)
/// from a row-major flat `[4*H, cols]` weight.
fn extract_submatrix(data: &[f32], cols: usize, start_row: usize, num_rows: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_rows * cols);
    for r in start_row..start_row + num_rows {
        let row_start = r * cols;
        result.extend_from_slice(&data[row_start..row_start + cols]);
    }
    result
}

/// Insert a `[rows, cols]` gate Linear weight (+ optional `[rows]` bias) and
/// push its `transB=1` Linear spec; returns the output tensor name.
///
/// The bias, when present, is wired as the spec's third input so
/// `ny_build`'s Linear conversion actually applies it (see module docs).
#[allow(clippy::too_many_arguments)]
fn push_gate_linear(
    specs: &mut Vec<LayerSpec>,
    layer_name: &str,
    weight_data: &[f32],
    rows: usize,
    cols: usize,
    bias_data: Option<Vec<f32>>,
    input_tensor: &str,
    intermediate_shape: &[i64],
    ctx: &mut Ctx,
) -> Result<String> {
    let w_name = format!("{layer_name}_weight");
    let arr = ArrayD::from_shape_vec(IxDyn(&[rows, cols]), weight_data.to_vec()).map_err(|e| {
        NyError::ModelLoad(format!(
            "Lstm: per-gate weight shape mismatch for {layer_name}: {e}"
        ))
    })?;
    ctx.insert_weight(&w_name, arr)?;

    let mut spec_inputs = vec![input_tensor.to_string(), w_name];
    if let Some(bias) = bias_data {
        let bias_name = format!("{layer_name}_bias");
        let bias_arr = ArrayD::from_shape_vec(IxDyn(&[rows]), bias).map_err(|e| {
            NyError::ModelLoad(format!(
                "Lstm: per-gate bias shape mismatch for {layer_name}: {e}"
            ))
        })?;
        ctx.insert_weight(&bias_name, bias_arr)?;
        spec_inputs.push(bias_name);
    }

    let out = format!("{layer_name}_out");
    ctx.tensor_shapes
        .insert(out.clone(), intermediate_shape.to_vec());
    // Linear uses transB=1: weight is [out, in], matmul is input @ Wᵀ.
    let mut attrs = HashMap::new();
    attrs.insert("transB".to_string(), AttributeValue::Int(1));
    specs.push(simple_spec(
        layer_name,
        LayerType::Linear,
        spec_inputs,
        &out,
        attrs,
    ));
    Ok(out)
}

/// Inject a zero-initialized state as a `Linear(zeros_w, zeros_b)` spec fed
/// by the LSTM input (zero weight ⇒ output is always the zero bias);
/// returns the state tensor name.
#[allow(clippy::too_many_arguments)]
fn inject_zero_state(
    specs: &mut Vec<LayerSpec>,
    prefix: &str,
    label: &str,
    hidden_size: usize,
    intermediate_shape: &[i64],
    input_dim: usize,
    input_tensor: &str,
    ctx: &mut Ctx,
) -> Result<String> {
    let zeros_w = vec![0.0_f32; hidden_size * input_dim];
    let zeros_b = vec![0.0_f32; hidden_size];
    push_gate_linear(
        specs,
        &format!("{prefix}_{label}"),
        &zeros_w,
        hidden_size,
        input_dim,
        Some(zeros_b),
        input_tensor,
        intermediate_shape,
        ctx,
    )
}

/// Push an activation spec and return its output tensor name.
fn push_activation(
    specs: &mut Vec<LayerSpec>,
    prefix: &str,
    label: &str,
    layer_type: LayerType,
    input_tensor: &str,
    intermediate_shape: &[i64],
    ctx: &mut Ctx,
) -> String {
    let act_name = format!("{prefix}_{label}");
    let act_out = format!("{act_name}_out");
    ctx.tensor_shapes
        .insert(act_out.clone(), intermediate_shape.to_vec());
    specs.push(simple_spec(
        &act_name,
        layer_type,
        vec![input_tensor.to_string()],
        &act_out,
        HashMap::new(),
    ));
    act_out
}

#[cfg(test)]
mod tests {
    use ny_build::{AttributeValue, GraphModel};
    use ny_core::{LayerType, NyError};

    use crate::schema::{ComputationGraph, DType, NodeId, TraceNode, TraceOp, WeightPayload};
    use crate::translate::translate;

    fn node(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
        TraceNode::new(
            NodeId(id),
            name,
            op,
            inputs.iter().map(|&i| NodeId(i)).collect(),
            shape.to_vec(),
            DType::F32,
        )
    }

    fn count(model: &GraphModel, lt: &LayerType) -> usize {
        model
            .network
            .layers
            .iter()
            .filter(|l| &l.layer_type == lt)
            .count()
    }

    fn layer<'a>(model: &'a GraphModel, name: &str) -> &'a ny_build::LayerSpec {
        model
            .network
            .layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("layer {name} not found"))
    }

    /// Deterministic pseudo-random weights (same generator as NN's tests).
    fn seeded_weights(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let t = ((state >> 33) as f32) / (u32::MAX as f32);
                t * 0.6 - 0.3
            })
            .collect()
    }

    fn lstm_op(
        input_size: usize,
        hidden_size: usize,
        with_bias: bool,
        initial_hidden: Option<u64>,
        initial_cell: Option<u64>,
    ) -> TraceOp {
        let four_h = 4 * hidden_size;
        TraceOp::Lstm {
            weight_ih: WeightPayload::f32(
                seeded_weights(42, four_h * input_size),
                vec![four_h, input_size],
            ),
            weight_hh: WeightPayload::f32(
                seeded_weights(43, four_h * hidden_size),
                vec![four_h, hidden_size],
            ),
            bias_ih: with_bias
                .then(|| WeightPayload::f32(seeded_weights(44, four_h), vec![four_h])),
            bias_hh: with_bias
                .then(|| WeightPayload::f32(seeded_weights(45, four_h), vec![four_h])),
            hidden_size,
            initial_hidden: initial_hidden.map(NodeId),
            initial_cell: initial_cell.map(NodeId),
        }
    }

    /// Input → Lstm (zero state, with biases): full decomposition layer census
    /// per NN — 8 gate Linear + 2 zero-state Linear, 4 gate-sum Add + 1 c_new
    /// Add (+1 input identity Add), 3 Sigmoid, 2 Tanh, 3 Mul — and the graph
    /// builds.
    #[test]
    fn lstm_zero_state_decomposes_like_nn() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "lstm", lstm_op(4, 3, true, None, None), &[0], &[3]),
        ]);
        let model = translate(&graph).expect("translate");

        assert_eq!(
            count(&model, &LayerType::Linear),
            10,
            "8 gate + 2 zero-state"
        );
        assert_eq!(
            count(&model, &LayerType::Add),
            6,
            "4 gate sums + c_new + input identity"
        );
        assert_eq!(count(&model, &LayerType::Sigmoid), 3, "i/f/o gates");
        assert_eq!(
            count(&model, &LayerType::Tanh),
            2,
            "g candidate + tanh(c_new)"
        );
        assert_eq!(count(&model, &LayerType::Mul), 3, "f*c, i*g, o*tanh(c)");

        // Final layer carries the traced node's name and output tensor.
        let final_mul = layer(&model, "layer0_trace_1");
        assert_eq!(final_mul.layer_type, LayerType::Mul);
        assert_eq!(final_mul.outputs, vec!["layer0_trace_1_out".to_string()]);
        assert_eq!(
            final_mul.inputs,
            vec![
                "layer0_trace_1_o_gate_out".to_string(),
                "layer0_trace_1_tanh_c_out".to_string()
            ]
        );

        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("graph builds");
    }

    /// Gate Linear specs: transB=1 attribute, weight+bias wired as inputs,
    /// and per-gate weight/bias values match NN's submatrix split with the
    /// combined bias halved 50/50 between ih and hh.
    #[test]
    fn lstm_gate_linears_match_nn_submatrix_and_half_bias() {
        let (input_size, hidden_size) = (4, 3);
        let four_h = 4 * hidden_size;
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[input_size]),
            node(
                1,
                "lstm",
                lstm_op(input_size, hidden_size, true, None, None),
                &[0],
                &[hidden_size],
            ),
        ]);
        let model = translate(&graph).expect("translate");

        let w_ih = seeded_weights(42, four_h * input_size);
        let b_ih = seeded_weights(44, four_h);
        let b_hh = seeded_weights(45, four_h);

        // f gate = gate index 1: rows [H, 2H) of weight_ih.
        let f_ih = layer(&model, "layer0_trace_1_f_ih");
        assert_eq!(f_ih.layer_type, LayerType::Linear);
        assert_eq!(
            f_ih.attributes.get("transB"),
            Some(&AttributeValue::Int(1)),
            "gate Linear uses transB=1"
        );
        assert_eq!(
            f_ih.inputs,
            vec![
                "layer0_trace_0_out".to_string(), // x identity out
                "layer0_trace_1_f_ih_weight".to_string(),
                "layer0_trace_1_f_ih_bias".to_string(),
            ]
        );

        let w = model
            .weights
            .get("layer0_trace_1_f_ih_weight")
            .expect("f_ih weight stored");
        assert_eq!(w.shape(), &[hidden_size, input_size]);
        let expected_rows = &w_ih[hidden_size * input_size..2 * hidden_size * input_size];
        let got: Vec<f32> = w.iter().copied().collect();
        assert_eq!(got, expected_rows, "f gate ih submatrix rows [H, 2H)");

        let b = model
            .weights
            .get("layer0_trace_1_f_ih_bias")
            .expect("f_ih bias stored");
        let got_b: Vec<f32> = b.iter().copied().collect();
        let expected_b: Vec<f32> = (hidden_size..2 * hidden_size)
            .map(|i| f32::midpoint(b_ih[i], b_hh[i]))
            .collect();
        assert_eq!(got_b, expected_b, "combined bias halved for the ih path");

        // hh path of the same gate reads h_prev (zero-state Linear output).
        let f_hh = layer(&model, "layer0_trace_1_f_hh");
        assert_eq!(
            f_hh.inputs[0], "layer0_trace_1_h_prev_out",
            "hh Linear consumes injected zero hidden state"
        );
    }

    /// Non-zero initial state: the recorded state nodes are consumed directly
    /// and no zero-state Linears are injected (8 Linear, not 10).
    #[test]
    fn lstm_nonzero_state_skips_zero_injection() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "h0", TraceOp::Input, &[], &[3]),
            node(2, "c0", TraceOp::Input, &[], &[3]),
            node(
                3,
                "lstm",
                lstm_op(4, 3, false, Some(1), Some(2)),
                &[0, 1, 2],
                &[3],
            ),
        ]);
        let model = translate(&graph).expect("translate");

        assert_eq!(
            count(&model, &LayerType::Linear),
            8,
            "no zero-state injection"
        );
        let i_hh = layer(&model, "layer0_trace_3_i_hh");
        assert_eq!(
            i_hh.inputs[0], "layer0_trace_1_out",
            "hh reads recorded h_0"
        );
        let f_cell = layer(&model, "layer0_trace_3_f_cell");
        assert_eq!(
            f_cell.inputs,
            vec![
                "layer0_trace_3_f_gate_out".to_string(),
                "layer0_trace_2_out".to_string()
            ],
            "f*c reads recorded c_0"
        );

        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("graph builds");
    }

    /// Bias-less LSTM: gate Linear specs carry exactly [input, weight] and no
    /// bias tensors are stored for the gates.
    #[test]
    fn lstm_without_bias_emits_two_input_linears() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "lstm", lstm_op(4, 3, false, None, None), &[0], &[3]),
        ]);
        let model = translate(&graph).expect("translate");

        let g_ih = layer(&model, "layer0_trace_1_g_ih");
        assert_eq!(g_ih.inputs.len(), 2, "no bias input without bias");
        assert!(!model.weights.contains_key("layer0_trace_1_g_ih_bias"));

        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("graph builds");
    }

    /// Weight-shape mismatch fails closed.
    #[test]
    fn lstm_bad_weight_rows_is_refused() {
        let mut op = lstm_op(4, 3, false, None, None);
        if let TraceOp::Lstm { weight_ih, .. } = &mut op {
            // 11 rows != 4*hidden_size = 12.
            *weight_ih = WeightPayload::f32(seeded_weights(7, 11 * 4), vec![11, 4]);
        }
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "lstm", op, &[0], &[3]),
        ]);
        let err = translate(&graph).expect_err("must refuse");
        assert!(
            matches!(err, NyError::ModelLoad(ref m) if m.contains("4*hidden_size")),
            "unexpected error: {err:?}"
        );
    }

    /// End-to-end value check against a manually computed LSTM cell: with a
    /// degenerate input box [x, x], IBP bounds must pin the exact
    /// h_1 = sigmoid(o)*tanh(c_1) — this catches dropped gate biases (the
    /// NN LayerSpec-path bug this port fixes) and any gate-order slip.
    #[test]
    fn lstm_ibp_on_point_box_matches_reference_cell() {
        let (input_size, hidden_size) = (4, 3);
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[input_size]),
            node(
                1,
                "lstm",
                lstm_op(input_size, hidden_size, true, None, None),
                &[0],
                &[hidden_size],
            ),
        ]);
        let model = translate(&graph).expect("translate");
        let gn = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("graph builds");

        let x = seeded_weights(99, input_size);
        let lo = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[input_size]), x.clone())
            .expect("shape");
        let bounds = ny_tensor::BoundedTensor::new(lo.clone(), lo).expect("valid bounds");
        let out = gn.propagate_ibp(&bounds).expect("IBP");

        // Reference single-cell LSTM (PyTorch gate order i, f, g, o;
        // h0 = c0 = 0, so weight_hh contributes nothing).
        let four_h = 4 * hidden_size;
        let w_ih = seeded_weights(42, four_h * input_size);
        let b = {
            let bih = seeded_weights(44, four_h);
            let bhh = seeded_weights(45, four_h);
            bih.iter()
                .zip(&bhh)
                .map(|(a, b)| a + b)
                .collect::<Vec<f32>>()
        };
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        let gate = |idx: usize, j: usize| {
            let row = idx * hidden_size + j;
            let mut acc = b[row];
            for (k, xv) in x.iter().enumerate() {
                acc += w_ih[row * input_size + k] * xv;
            }
            acc
        };
        let expected: Vec<f32> = (0..hidden_size)
            .map(|j| {
                let i_g = sigmoid(gate(0, j));
                let g_c = gate(2, j).tanh();
                let o_g = sigmoid(gate(3, j));
                // c0 = 0 ⇒ f gate contributes nothing.
                let c_new = i_g * g_c;
                o_g * c_new.tanh()
            })
            .collect();

        let (out_lo, out_hi) = out.lower_upper();
        for (j, &e) in expected.iter().enumerate() {
            let l = out_lo[[j]];
            let u = out_hi[[j]];
            assert!(
                l <= e + 1e-4 && e - 1e-4 <= u,
                "output {j}: reference {e} outside IBP [{l}, {u}]"
            );
            assert!(
                (u - l).abs() < 1e-3,
                "point box should give near-point bounds, got [{l}, {u}]"
            );
        }
    }
}
