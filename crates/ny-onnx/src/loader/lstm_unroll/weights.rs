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
use ndarray::{ArrayD, Axis, IxDyn};

use super::node_builder;
use super::{LstmConfig, LstmWeightNames};

/// A direction's synthesized tensors, kept detached from `WeightStore` until
/// every exactness and shape check has passed. This prevents a rejected LSTM
/// from leaving a partially materialized lowering behind.
pub(super) struct PreparedLstmWeights {
    names: LstmWeightNames,
    values: Vec<(String, ArrayD<f32>)>,
}

impl PreparedLstmWeights {
    pub(super) fn store(self, weights: &mut WeightStore) -> LstmWeightNames {
        for (name, value) in self.values {
            weights.insert(name, value);
        }
        self.names
    }
}

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
    let prepared = prepare_lstm_weights(config, node, weights, direction_idx, dir_base)?;
    Ok(prepared.store(weights))
}

/// Prepare one direction's lowering constants without mutating `weights`.
///
/// Transposes and slices merely rearrange authored binary32 values. The bias
/// addition and initial-state matrix product are admitted only when every
/// scalar operation has an exact binary32 representation, so materializing
/// them cannot change the verifier's exact-real network semantics.
pub(super) fn prepare_lstm_weights(
    config: &LstmConfig,
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    direction_idx: usize,
    dir_base: &str,
) -> Result<PreparedLstmWeights, String> {
    let h = config.hidden_size;
    let gate_size = h
        .checked_mul(4)
        .ok_or("LSTM gate dimension overflows usize")?;
    let bias_size = h
        .checked_mul(8)
        .ok_or("LSTM bias dimension overflows usize")?;
    if direction_idx >= config.num_directions {
        return Err(format!(
            "LSTM direction index {direction_idx} is out of range for {} directions",
            config.num_directions
        ));
    }

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
        .ok_or_else(|| format!("LSTM weight W '{w_name}' not found"))?;
    validate_float_parameter(weights, w_name, "W")?;
    validate_shape(
        w_raw,
        &[config.num_directions, gate_size, config.input_size],
        "W",
    )?;
    validate_finite(w_raw, "W")?;
    let w_dir = slice_direction_if_needed(w_raw, direction_idx, config.num_directions)?;
    let w_t = node_builder::transpose_2d(&node_builder::squeeze_leading_dim(&w_dir)?);
    let w_t_name = format!("{dir_base}__lstm_W_T");

    // R: [num_directions, 4H, H] → slice direction → squeeze → transpose
    let r_raw = weights
        .get(r_name)
        .ok_or_else(|| format!("LSTM weight R '{r_name}' not found"))?;
    validate_float_parameter(weights, r_name, "R")?;
    validate_shape(r_raw, &[config.num_directions, gate_size, h], "R")?;
    validate_finite(r_raw, "R")?;
    let r_dir = slice_direction_if_needed(r_raw, direction_idx, config.num_directions)?;
    let r_t = node_builder::transpose_2d(&node_builder::squeeze_leading_dim(&r_dir)?);
    let r_t_name = format!("{dir_base}__lstm_R_T");

    // Bias: [num_directions, 8H] → slice direction → flatten → Wb[4H] + Rb[4H]
    let bias_name = format!("{dir_base}__lstm_bias");
    let bias = if let Some(bn) = b_name {
        let b_raw = weights
            .get(bn)
            .ok_or_else(|| format!("LSTM bias B '{bn}' not found"))?;
        validate_float_parameter(weights, bn, "B")?;
        validate_shape(b_raw, &[config.num_directions, bias_size], "B")?;
        validate_finite(b_raw, "B")?;
        let b_dir = slice_direction_bias(b_raw, direction_idx, config.num_directions, h)?;
        exact_combined_bias(&b_dir, gate_size)?.into_dyn()
    } else {
        ArrayD::zeros(IxDyn(&[gate_size]))
    };

    // Initial states: [num_directions, batch, H] → slice direction → squeeze
    let h0_name = format!("{dir_base}__lstm_h0");
    let h0 = initial_state_value(initial_h, "initial_h", config, weights, direction_idx)?;
    let c0_name = format!("{dir_base}__lstm_c0");
    let c0 = initial_state_value(initial_c, "initial_c", config, weights, direction_idx)?;

    // Pre-compute h0 @ R_T so the graph builder never sees a MatMul with two
    // weight inputs at t=0.
    let h0_hr_name = format!("{dir_base}__lstm_h0_hR");
    let h0_hr = exact_initial_recurrent_product(&h0, &r_t)?;

    // Reshape target for timestep extraction: [batch, input_size].
    let x_reshape_name = format!("{dir_base}__lstm_x_reshape");
    let x_reshape = ArrayD::from_shape_vec(
        IxDyn(&[2]),
        vec![config.batch_size as f32, config.input_size as f32],
    )
    .expect("x_reshape shape");

    let names = LstmWeightNames {
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
    };
    let values = vec![
        (names.w_t.clone(), w_t.into_dyn()),
        (names.r_t.clone(), r_t.into_dyn()),
        (names.bias.clone(), bias),
        (names.h0.clone(), h0),
        (names.c0.clone(), c0),
        (names.h0_hr.clone(), h0_hr),
        (names.x_reshape.clone(), x_reshape),
    ];
    Ok(PreparedLstmWeights { names, values })
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

/// Extract an initial state for one direction using the axis selected by the
/// ONNX layout. For batch=1, squeeze the remaining batch axis to `[H]` so the
/// generated primitive graph retains its established singleton-batch form.
fn initial_state_value(
    input_name: Option<&String>,
    label: &str,
    config: &LstmConfig,
    weights: &WeightStore,
    direction_idx: usize,
) -> Result<ArrayD<f32>, String> {
    if let Some(name) = input_name {
        let raw = weights
            .get(name)
            .ok_or_else(|| format!("LSTM {label} state '{name}' not found"))?;
        validate_float_parameter(weights, name, label)?;
        let expected = if config.layout == 0 {
            vec![config.num_directions, config.batch_size, config.hidden_size]
        } else {
            vec![config.batch_size, config.num_directions, config.hidden_size]
        };
        validate_shape(raw, &expected, label)?;
        validate_finite(raw, label)?;

        let direction_axis = if config.layout == 0 { 0 } else { 1 };
        let direction = raw
            .index_axis(Axis(direction_axis), direction_idx)
            .to_owned();
        if config.batch_size == 1 {
            Ok(direction.index_axis(Axis(0), 0).to_owned().into_dyn())
        } else {
            Ok(direction.into_dyn())
        }
    } else if config.batch_size == 1 {
        Ok(ArrayD::zeros(IxDyn(&[config.hidden_size])))
    } else {
        Ok(ArrayD::zeros(IxDyn(&[
            config.batch_size,
            config.hidden_size,
        ])))
    }
}

fn validate_float_parameter(weights: &WeightStore, name: &str, label: &str) -> Result<(), String> {
    if weights.get_integers(name).is_some() {
        return Err(format!(
            "LSTM {label} parameter '{name}' is integer-valued; FLOAT32 parameters are required"
        ));
    }
    Ok(())
}

fn validate_shape(array: &ArrayD<f32>, expected: &[usize], label: &str) -> Result<(), String> {
    if array.shape() != expected {
        return Err(format!(
            "LSTM {label} must have shape {expected:?}, got {:?}",
            array.shape()
        ));
    }
    Ok(())
}

fn validate_finite(array: &ArrayD<f32>, label: &str) -> Result<(), String> {
    if array.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "LSTM {label} contains a non-finite value, which cannot authenticate exact-real lowering"
        ));
    }
    Ok(())
}

fn exact_combined_bias(
    bias: &ndarray::Array1<f32>,
    gate_size: usize,
) -> Result<ndarray::Array1<f32>, String> {
    let mut combined = Vec::with_capacity(gate_size);
    for index in 0..gate_size {
        let wb = bias[index];
        let rb = bias[gate_size + index];
        let sum = exact_f32_sum(wb, rb).ok_or_else(|| {
            format!(
                "LSTM bias Wb+Rb at gate index {index} is not exactly representable as FLOAT32 ({wb} + {rb})"
            )
        })?;
        combined.push(sum);
    }
    Ok(ndarray::Array1::from_vec(combined))
}

fn exact_initial_recurrent_product(
    h0: &ArrayD<f32>,
    r_t: &ndarray::Array2<f32>,
) -> Result<ArrayD<f32>, String> {
    let (rows, inner, output_shape) = match h0.ndim() {
        1 => (1, h0.shape()[0], vec![r_t.shape()[1]]),
        2 => (
            h0.shape()[0],
            h0.shape()[1],
            vec![h0.shape()[0], r_t.shape()[1]],
        ),
        rank => {
            return Err(format!(
                "LSTM initial_h must have rank 1 or 2 after slicing, got {rank}"
            ))
        }
    };
    if inner != r_t.shape()[0] {
        return Err(format!(
            "LSTM initial_h width {inner} does not match recurrent weight height {}",
            r_t.shape()[0]
        ));
    }

    let h_values = h0
        .as_slice_memory_order()
        .ok_or_else(|| "LSTM initial_h is not contiguous after direction slicing".to_string())?;
    let mut result = Vec::with_capacity(rows * r_t.shape()[1]);
    for row in 0..rows {
        for column in 0..r_t.shape()[1] {
            let mut sum = 0.0_f32;
            for index in 0..inner {
                let lhs = h_values[row * inner + index];
                let rhs = r_t[[index, column]];
                let product = exact_f32_product(lhs, rhs).ok_or_else(|| {
                    format!(
                        "LSTM initial_h @ R_T product at ({row},{index},{column}) is not exactly representable as FLOAT32 ({lhs} * {rhs})"
                    )
                })?;
                sum = exact_f32_sum(sum, product).ok_or_else(|| {
                    format!(
                        "LSTM initial_h @ R_T accumulation at ({row},{column}) is not exactly representable as FLOAT32"
                    )
                })?;
            }
            result.push(sum);
        }
    }
    ArrayD::from_shape_vec(IxDyn(&output_shape), result)
        .map_err(|error| format!("cannot shape exact LSTM initial_h @ R_T result: {error}"))
}

fn exact_f32_sum(lhs: f32, rhs: f32) -> Option<f32> {
    if !lhs.is_finite() || !rhs.is_finite() {
        return None;
    }
    let lhs64 = f64::from(lhs);
    let rhs64 = f64::from(rhs);
    let sum64 = lhs64 + rhs64;
    let rhs_virtual = sum64 - lhs64;
    let error = (lhs64 - (sum64 - rhs_virtual)) + (rhs64 - rhs_virtual);
    let rounded = lhs + rhs;
    (error == 0.0 && rounded.is_finite() && f64::from(rounded) == sum64).then_some(rounded)
}

fn exact_f32_product(lhs: f32, rhs: f32) -> Option<f32> {
    if !lhs.is_finite() || !rhs.is_finite() {
        return None;
    }
    // The exact product of two binary32 significands has at most 48 bits, so
    // binary64 represents it exactly across binary32's finite exponent range.
    let product64 = f64::from(lhs) * f64::from(rhs);
    let rounded = lhs * rhs;
    (rounded.is_finite() && f64::from(rounded) == product64).then_some(rounded)
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
