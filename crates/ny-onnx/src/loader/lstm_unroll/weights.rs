// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Weight precomputation and storage for LSTM unrolling.
//!
//! Extracts ONNX LSTM weights (W, R, B, initial states) and pre-computes
//! transposed matrices, combined bias, and h0 @ R_T for efficient unrolling.
//!
//! For bidirectional LSTM, weights are stored as `[num_directions, ...]`.
//! The `direction_idx` parameter selects which direction's weights to extract.

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

use super::node_builder;
use super::{LstmConfig, LstmWeightNames};

/// Pre-compute and store transposed weight matrices and combined bias.
///
/// `direction_idx`: 0 for forward, 1 for reverse (bidirectional only).
/// `dir_base`: direction-specific name prefix for stored weights.
pub(super) fn precompute_lstm_weights(
    config: &LstmConfig,
    node: &onnx_proto::NodeProto,
    weights: &mut WeightStore,
    direction_idx: usize,
    dir_base: &str,
) -> Result<LstmWeightNames, String> {
    let h = config.hidden_size;

    let w_name = node
        .input
        .get(1)
        .filter(|s| !s.is_empty())
        .ok_or("LSTM missing W input")?;
    let r_name = node
        .input
        .get(2)
        .filter(|s| !s.is_empty())
        .ok_or("LSTM missing R input")?;
    let b_name = node.input.get(3).filter(|s| !s.is_empty());
    let initial_h = node.input.get(5).filter(|s| !s.is_empty());
    let initial_c = node.input.get(6).filter(|s| !s.is_empty());

    // W: [num_directions, 4H, I] → slice direction → [1, 4H, I] → squeeze → transpose
    let w_raw = weights
        .get(w_name)
        .ok_or_else(|| format!("LSTM weight W '{w_name}' not found"))?
        .clone();
    let w_dir = slice_direction_if_needed(&w_raw, direction_idx, config.num_directions)?;
    let w_t = node_builder::transpose_2d(&node_builder::squeeze_leading_dim(&w_dir)?);
    let w_t_name = format!("{dir_base}__lstm_W_T");
    weights.insert(w_t_name.clone(), w_t.into_dyn());

    // R: [num_directions, 4H, H] → slice direction → squeeze → transpose
    let r_raw = weights
        .get(r_name)
        .ok_or_else(|| format!("LSTM weight R '{r_name}' not found"))?
        .clone();
    let r_dir = slice_direction_if_needed(&r_raw, direction_idx, config.num_directions)?;
    let r_t = node_builder::transpose_2d(&node_builder::squeeze_leading_dim(&r_dir)?);
    let r_t_name = format!("{dir_base}__lstm_R_T");
    weights.insert(r_t_name.clone(), r_t.into_dyn());

    // Bias: [num_directions, 8H] → slice direction → flatten → Wb[4H] + Rb[4H]
    let bias_name = format!("{dir_base}__lstm_bias");
    if let Some(bn) = b_name {
        let b_raw = weights
            .get(bn)
            .ok_or_else(|| format!("LSTM bias B '{bn}' not found"))?
            .clone();
        let b_dir = slice_direction_bias(&b_raw, direction_idx, config.num_directions, h)?;
        let wb = b_dir.slice(ndarray::s![..4 * h]).to_owned();
        let rb = b_dir.slice(ndarray::s![4 * h..]).to_owned();
        weights.insert(bias_name.clone(), (&wb + &rb).into_dyn());
    } else {
        weights.insert(bias_name.clone(), ArrayD::zeros(IxDyn(&[4 * h])));
    }

    // Initial states: [num_directions, batch, H] → slice direction → squeeze
    let h0_name = format!("{dir_base}__lstm_h0");
    store_initial_state(initial_h, &h0_name, config, weights, direction_idx)?;
    let c0_name = format!("{dir_base}__lstm_c0");
    store_initial_state(initial_c, &c0_name, config, weights, direction_idx)?;

    // Pre-compute h0 @ R_T so the graph builder never sees a MatMul with two
    // weight inputs at t=0.
    let h0_hr_name = format!("{dir_base}__lstm_h0_hR");
    let h0 = weights
        .get(&h0_name)
        .ok_or_else(|| format!("h0 '{h0_name}' not found after store"))?
        .clone();
    let r_t_val = weights
        .get(&r_t_name)
        .ok_or_else(|| format!("R_T '{r_t_name}' not found"))?
        .clone();
    let r_t_2d = r_t_val
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| format!("R_T must be 2D: {e}"))?;
    let h0_hr_flat = if h0.ndim() == 1 {
        let h0_1d = h0
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| format!("h0 must be 1D: {e}"))?;
        h0_1d.dot(&r_t_2d).into_dyn()
    } else {
        let h0_2d = h0
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("h0 must be 2D: {e}"))?;
        h0_2d.dot(&r_t_2d).into_dyn()
    };
    weights.insert(h0_hr_name.clone(), h0_hr_flat);

    // Reshape target for timestep extraction: [batch, input_size].
    let x_reshape_name = format!("{dir_base}__lstm_x_reshape");
    weights.insert(
        x_reshape_name.clone(),
        ArrayD::from_shape_vec(
            IxDyn(&[2]),
            vec![config.batch_size as f32, config.input_size as f32],
        )
        .expect("x_reshape shape"),
    );

    Ok(LstmWeightNames {
        w_t: w_t_name,
        r_t: r_t_name,
        bias: bias_name,
        h0: h0_name,
        c0: c0_name,
        h0_hr: h0_hr_name,
        x_reshape: x_reshape_name,
        gate_axis: format!("{dir_base}__lstm_gate_axis"),
        gate_step: format!("{dir_base}__lstm_gate_step"),
        time_axis: format!("{dir_base}__lstm_time_axis"),
        time_step: format!("{dir_base}__lstm_time_step"),
    })
}

/// Slice a `[num_directions, M, N]` weight tensor to `[1, M, N]` for one direction.
/// For forward-only (num_directions=1), returns the tensor unchanged.
fn slice_direction_if_needed(
    arr: &ArrayD<f32>,
    direction_idx: usize,
    num_directions: usize,
) -> Result<ArrayD<f32>, String> {
    if num_directions == 1 {
        return Ok(arr.clone());
    }
    node_builder::slice_direction(arr, direction_idx)
}

/// Slice a `[num_directions, 8H]` bias tensor to a flat `[8H]` for one direction.
/// For forward-only (num_directions=1), falls back to flatten_to_1d directly.
fn slice_direction_bias(
    arr: &ArrayD<f32>,
    direction_idx: usize,
    num_directions: usize,
    hidden_size: usize,
) -> Result<ndarray::Array1<f32>, String> {
    if num_directions == 1 {
        return node_builder::flatten_to_1d(arr, 8 * hidden_size);
    }
    let sliced = node_builder::slice_direction(arr, direction_idx)?;
    // sliced is [1, 8H], flatten to [8H]
    let flat = sliced.into_raw_vec_and_offset().0;
    if flat.len() != 8 * hidden_size {
        return Err(format!(
            "expected {} bias elements for direction {direction_idx}, got {}",
            8 * hidden_size,
            flat.len()
        ));
    }
    Ok(ndarray::Array1::from_vec(flat))
}

/// Store an initial state (h0 or c0), slicing by direction and squeezing.
///
/// For batch=1 (the common case for verification), squeeze to 1D `[H]` so that
/// downstream MulConstant/AddConstant ops produce 1D outputs matching the gate
/// dimension.
fn store_initial_state(
    input_name: Option<&String>,
    target_name: &str,
    config: &LstmConfig,
    weights: &mut WeightStore,
    direction_idx: usize,
) -> Result<(), String> {
    if let Some(name) = input_name {
        let raw = weights
            .get(name)
            .ok_or_else(|| format!("LSTM initial state '{name}' not found"))?
            .clone();
        // For bidirectional: raw is [2, B, H], slice to [1, B, H] first.
        let dir_raw = slice_direction_if_needed(&raw, direction_idx, config.num_directions)?;
        let squeezed = node_builder::squeeze_leading_dim(&dir_raw)?;
        // For batch=1, squeeze [1, H] → [H] to avoid phantom batch dimension.
        let val = if config.batch_size == 1 && squeezed.ndim() == 2 && squeezed.shape()[0] == 1 {
            squeezed
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| format!("initial state must be 2D: {e}"))?
                .row(0)
                .to_owned()
                .into_dyn()
        } else {
            squeezed.into_dyn()
        };
        weights.insert(target_name.to_string(), val);
    } else if config.batch_size == 1 {
        weights.insert(
            target_name.to_string(),
            ArrayD::zeros(IxDyn(&[config.hidden_size])),
        );
    } else {
        weights.insert(
            target_name.to_string(),
            ArrayD::zeros(IxDyn(&[config.batch_size, config.hidden_size])),
        );
    }
    Ok(())
}

/// Store shared slice parameters for gate splitting and timestep extraction.
pub(super) fn store_lstm_slice_params(
    config: &LstmConfig,
    weights: &mut WeightStore,
    dir_base: &str,
) {
    let h = config.hidden_size;

    // Gate splitting: axis 1 (last dim of [batch, 4H])
    weights.insert(
        format!("{dir_base}__lstm_gate_axis"),
        ArrayD::from_elem(IxDyn(&[1]), 1.0),
    );
    weights.insert(
        format!("{dir_base}__lstm_gate_step"),
        ArrayD::from_elem(IxDyn(&[1]), 1.0),
    );

    for (idx, label) in [(0, "i"), (1, "o"), (2, "f"), (3, "c")] {
        weights.insert(
            format!("{dir_base}__lstm_gate_{label}_start"),
            ArrayD::from_elem(IxDyn(&[1]), (idx * h) as f32),
        );
        weights.insert(
            format!("{dir_base}__lstm_gate_{label}_end"),
            ArrayD::from_elem(IxDyn(&[1]), ((idx + 1) * h) as f32),
        );
    }

    // Timestep extraction: axis 1 (time dim, batch_first)
    weights.insert(
        format!("{dir_base}__lstm_time_axis"),
        ArrayD::from_elem(IxDyn(&[1]), 1.0),
    );
    weights.insert(
        format!("{dir_base}__lstm_time_step"),
        ArrayD::from_elem(IxDyn(&[1]), 1.0),
    );
}

// --- Shape lookup ---

pub(super) fn find_tensor_shape(
    name: &str,
    graph_inputs: &[onnx_proto::ValueInfoProto],
    graph_value_info: &[onnx_proto::ValueInfoProto],
    weights: &WeightStore,
    inferred_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    for input in graph_inputs {
        if input.name == name {
            if let Some(shape) = extract_shape_from_value_info(input) {
                if !shape.is_empty() {
                    return Some(shape);
                }
            }
        }
    }
    for info in graph_value_info {
        if info.name == name {
            if let Some(shape) = extract_shape_from_value_info(info) {
                if !shape.is_empty() {
                    return Some(shape);
                }
            }
        }
    }
    if let Some(shape) = inferred_shapes.get(name) {
        if !shape.is_empty() {
            return Some(shape.clone());
        }
    }
    if let Some(w) = weights.get(name) {
        return Some(w.shape().iter().map(|&d| d as i64).collect());
    }
    None
}

fn extract_shape_from_value_info(info: &onnx_proto::ValueInfoProto) -> Option<Vec<i64>> {
    let tensor_type = info.r#type.as_ref()?.tensor_type.as_ref()?;
    let shape = tensor_type.shape.as_ref()?;
    let dims: Vec<i64> = shape
        .dim
        .iter()
        .map(|d| match &d.value {
            Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
            _ => -1,
        })
        .collect();
    Some(dims)
}
