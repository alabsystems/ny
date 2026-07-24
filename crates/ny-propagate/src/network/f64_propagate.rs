// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-precision (f64) network propagation for soundness-critical benchmarks.
//!
//! Provides `propagate_network_f64` which walks a sequential network
//! (Linear+Conv2D+ReLU) entirely in f64. Required for VNN-COMP
//! soundnessbench and sat_relu where f32 rounding causes incorrect verdicts.
//!
//! Reference: alpha-beta-CROWN `double_fp: true` (`abcrown.py:81-82`).

use ny_core::Result;
use ny_tensor::BoundedTensor64;

use crate::bounds::LinearBounds64;
use crate::layers::float64::conv2d::{
    propagate_conv2d_crown_backward_f64, propagate_conv2d_ibp_f64, Conv2dParams,
};
use crate::layers::float64::linear::{
    propagate_linear_crown_backward_f64, propagate_linear_ibp_f64, weights_to_f64,
};
use crate::layers::float64::relu::{propagate_relu_crown_backward_f64, propagate_relu_ibp_f64};

/// Propagation mode for f64 network verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64PropagationMode {
    /// Interval Bound Propagation — fast but loose.
    Ibp,
    /// CROWN — tighter bounds via backward linear relaxation.
    Crown,
}

/// f64 linear layer data (boxed to avoid enum size variance).
#[derive(Debug, Clone)]
pub struct LinearDataF64 {
    /// Weight matrix (out_features, in_features) in f64.
    pub weight: ndarray::Array2<f64>,
    /// Bias vector (out_features,) in f64.
    pub bias: ndarray::Array1<f64>,
}

/// f64 Conv2D layer data (boxed to avoid enum size variance).
#[derive(Debug, Clone)]
pub struct Conv2dDataF64 {
    /// Kernel (out_channels, in_channels, kh, kw) in f64.
    pub kernel: ndarray::Array4<f64>,
    /// Bias vector (out_channels,) in f64.
    pub bias: ndarray::Array1<f64>,
    /// Convolution parameters.
    pub params: Conv2dParams,
}

/// A sequential layer in the f64 propagation path.
///
/// Only supports the layer types needed for soundnessbench/sat_relu:
/// Linear (FC), ReLU, and Conv2D.
#[derive(Debug, Clone)]
pub enum SequentialLayerF64 {
    /// Fully-connected linear layer: y = Wx + b.
    Linear(Box<LinearDataF64>),
    /// Conv2D layer.
    Conv2d(Box<Conv2dDataF64>),
    /// ReLU activation: y = max(0, x).
    Relu,
}

/// Propagate a sequential network entirely in f64 using IBP.
///
/// Walks forward through all layers, producing interval bounds at each step.
pub fn propagate_ibp_f64(
    layers: &[SequentialLayerF64],
    input: &BoundedTensor64,
) -> Result<BoundedTensor64> {
    let mut current = input.clone();
    for layer in layers {
        current = match layer {
            SequentialLayerF64::Linear(data) => {
                propagate_linear_ibp_f64(&data.weight, &data.bias, &current)?
            }
            SequentialLayerF64::Conv2d(data) => {
                propagate_conv2d_ibp_f64(&data.kernel, &data.bias, &current, &data.params)?
            }
            SequentialLayerF64::Relu => propagate_relu_ibp_f64(&current)?,
        };
    }
    Ok(current)
}

/// Propagate a sequential network entirely in f64 using CROWN.
///
/// Algorithm:
/// 1. IBP forward pass to collect intermediate bounds at each layer
/// 2. Initialize LinearBounds64 as identity at the output
/// 3. CROWN backward pass through layers in reverse
/// 4. Concretize the final linear bounds against the input
///
/// Returns the intersection of IBP and CROWN bounds (element-wise tightest).
pub fn propagate_crown_f64(
    layers: &[SequentialLayerF64],
    input: &BoundedTensor64,
) -> Result<BoundedTensor64> {
    if layers.is_empty() {
        return Ok(input.clone());
    }

    // Step 1: IBP forward to get intermediate bounds
    let intermediate_bounds = collect_ibp_bounds_f64(layers, input)?;

    // Step 2: Initialize CROWN backward at output
    let output_dim = intermediate_bounds
        .last()
        .ok_or_else(|| ny_core::NyError::InvalidSpec("f64 CROWN: no intermediate bounds".into()))?
        .len();
    let mut linear_bounds = LinearBounds64::identity(output_dim);

    // Step 3: CROWN backward
    for (i, layer) in layers.iter().enumerate().rev() {
        match layer {
            SequentialLayerF64::Linear(data) => {
                linear_bounds =
                    propagate_linear_crown_backward_f64(&data.weight, &data.bias, &linear_bounds)?;
            }
            SequentialLayerF64::Conv2d(data) => {
                linear_bounds = propagate_conv2d_crown_backward_f64(
                    &data.kernel,
                    &data.bias,
                    &linear_bounds,
                    &data.params,
                )?;
            }
            SequentialLayerF64::Relu => {
                let pre_act = if i > 0 {
                    &intermediate_bounds[i - 1]
                } else {
                    input
                };
                linear_bounds = propagate_relu_crown_backward_f64(&linear_bounds, pre_act)?;
            }
        }
    }

    // Step 4: Concretize
    let crown_output = linear_bounds.concretize(input)?;

    // Step 5: Intersect with IBP (element-wise tightest)
    let ibp_output = intermediate_bounds
        .last()
        .ok_or_else(|| ny_core::NyError::InvalidSpec("f64 CROWN: no intermediate bounds".into()))?;
    intersect_bounds_f64(&crown_output, ibp_output)
}

/// Propagate a network in f64 with the specified mode.
pub fn propagate_network_f64(
    layers: &[SequentialLayerF64],
    input: &BoundedTensor64,
    mode: F64PropagationMode,
) -> Result<BoundedTensor64> {
    match mode {
        F64PropagationMode::Ibp => propagate_ibp_f64(layers, input),
        F64PropagationMode::Crown => propagate_crown_f64(layers, input),
    }
}

/// Collect IBP bounds at each layer (for CROWN pre-activation bounds).
fn collect_ibp_bounds_f64(
    layers: &[SequentialLayerF64],
    input: &BoundedTensor64,
) -> Result<Vec<BoundedTensor64>> {
    let mut bounds = Vec::with_capacity(layers.len());
    let mut current = input.clone();
    for layer in layers {
        current = match layer {
            SequentialLayerF64::Linear(data) => {
                propagate_linear_ibp_f64(&data.weight, &data.bias, &current)?
            }
            SequentialLayerF64::Conv2d(data) => {
                propagate_conv2d_ibp_f64(&data.kernel, &data.bias, &current, &data.params)?
            }
            SequentialLayerF64::Relu => propagate_relu_ibp_f64(&current)?,
        };
        bounds.push(current.clone());
    }
    Ok(bounds)
}

/// Per-element intersection of two BoundedTensor64 with union fallback.
///
/// For each element: if `max(lower_a, lower_b) <= min(upper_a, upper_b)`, uses
/// intersection (tighter bounds). Otherwise falls back to union (conservative,
/// preserves soundness). This matches the f32 `intersection_per_element` pattern
/// in `numeric.rs` and prevents inverted bounds from numerical imprecision.
fn intersect_bounds_f64(a: &BoundedTensor64, b: &BoundedTensor64) -> Result<BoundedTensor64> {
    let mut lower = a.lower().clone();
    let mut upper = a.upper().clone();

    ndarray::Zip::from(&mut lower)
        .and(&mut upper)
        .and(b.lower())
        .and(b.upper())
        .for_each(|al, au, &bl, &bu| {
            let int_lower = al.max(bl);
            let int_upper = au.min(bu);
            if int_lower <= int_upper {
                // Intersection is non-empty: use tighter bounds
                *al = int_lower;
                *au = int_upper;
            } else {
                // Disjoint: fall back to union (conservative, sound)
                *al = al.min(bl);
                *au = au.max(bu);
            }
        });

    // Use checked constructor: the intersection/union loop preserves matching shapes
    // and the union fallback guarantees lower <= upper for every element.
    // The f64 path is not performance-critical (soundnessbench only). (#4253)
    BoundedTensor64::new(lower, upper)
}

/// Evaluate the network in f64 at a concrete input.
///
/// Exact f64 reference evaluation of a network.
pub fn evaluate_network_f64(
    layers: &[SequentialLayerF64],
    input: &ndarray::ArrayD<f64>,
) -> Result<ndarray::ArrayD<f64>> {
    let concrete_input = BoundedTensor64::concrete(input.clone())?;
    let output = propagate_ibp_f64(layers, &concrete_input)?;
    Ok(output.lower().clone())
}

/// Convert network layers from f32 to f64.
///
/// Helper for the CLI integration path: given f32 layer weights from ONNX,
/// produce f64 layers for the double-precision propagation path.
pub fn convert_linear_layers_to_f64(
    layers: &[(ndarray::Array2<f32>, Option<ndarray::Array1<f32>>)],
    relu_indices: &[usize],
) -> Vec<SequentialLayerF64> {
    let mut result = Vec::new();
    for (i, (weight, bias)) in layers.iter().enumerate() {
        let (w64, b64) = weights_to_f64(weight, bias.as_ref());
        result.push(SequentialLayerF64::Linear(Box::new(LinearDataF64 {
            weight: w64,
            bias: b64,
        })));
        if relu_indices.contains(&i) {
            result.push(SequentialLayerF64::Relu);
        }
    }
    result
}

/// Convert a sequential `Network`'s layers to f64 for double-precision propagation.
///
/// Supports: Linear, Conv2d, ReLU, Flatten (no-op on bounds).
/// Returns `Err` for any unsupported layer type.
///
/// Reference: design step 6 of `designs/2026-03-04-f64-propagation-path.md`.
pub fn convert_network_to_f64(layers: &[crate::layers::Layer]) -> Result<Vec<SequentialLayerF64>> {
    use crate::layers::Layer;

    let mut result = Vec::with_capacity(layers.len());
    for layer in layers {
        match layer {
            Layer::Linear(lin) => {
                let weight = lin.weight.mapv(|x| x as f64);
                let bias = match &lin.bias {
                    Some(b) => b.mapv(|x| x as f64),
                    None => ndarray::Array1::zeros(lin.weight.nrows()),
                };
                result.push(SequentialLayerF64::Linear(Box::new(LinearDataF64 {
                    weight,
                    bias,
                })));
            }
            Layer::Conv2d(conv) => {
                let kernel_shape = conv.kernel.shape();
                let kernel_4d: ndarray::Array4<f64> = conv
                    .kernel
                    .mapv(|x| x as f64)
                    .into_shape_with_order((
                        kernel_shape[0],
                        kernel_shape[1],
                        kernel_shape[2],
                        kernel_shape[3],
                    ))
                    .map_err(|e| {
                        ny_core::NyError::InternalError(format!(
                            "Conv2d kernel reshape to 4D failed: {e}"
                        ))
                    })?;
                let bias = match &conv.bias {
                    Some(b) => b.mapv(|x| x as f64),
                    None => ndarray::Array1::zeros(kernel_shape[0]),
                };
                let input_hw = conv.input_shape.ok_or_else(|| {
                    ny_core::NyError::InternalError(
                        "Conv2d layer missing input_shape for f64 CROWN backward".to_string(),
                    )
                })?;
                result.push(SequentialLayerF64::Conv2d(Box::new(Conv2dDataF64 {
                    kernel: kernel_4d,
                    bias,
                    params: Conv2dParams {
                        stride: conv.stride,
                        padding: conv.padding,
                        input_hw,
                    },
                })));
            }
            Layer::ReLU(_) => {
                result.push(SequentialLayerF64::Relu);
            }
            Layer::Flatten(_) => {
                // Flatten is a no-op on bounds (just reshapes).
                // The f64 path uses flat arrays, so skip.
            }
            other => {
                return Err(ny_core::NyError::UnsupportedOp(format!(
                    "f64 propagation does not support layer type: {:?}. \
                     Only Linear, Conv2d, ReLU, and Flatten are supported.",
                    std::mem::discriminant(other)
                )));
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn linear_layer_f64(
        weight: ndarray::Array2<f64>,
        bias: ndarray::Array1<f64>,
    ) -> SequentialLayerF64 {
        SequentialLayerF64::Linear(Box::new(LinearDataF64 { weight, bias }))
    }

    #[test]
    fn test_ibp_single_linear() {
        let layers = vec![linear_layer_f64(arr2(&[[2.0f64, -1.0]]), arr1(&[1.0f64]))];
        let input = BoundedTensor64::new(
            arr1(&[0.0f64, 0.0]).into_dyn(),
            arr1(&[1.0f64, 1.0]).into_dyn(),
        )
        .unwrap();

        let result = propagate_ibp_f64(&layers, &input).unwrap();
        // lower = 2*0 + (-1)*1 + 1 = 0
        // upper = 2*1 + (-1)*0 + 1 = 3
        assert_eq!(result.lower()[0], 0.0);
        assert_eq!(result.upper()[0], 3.0);
    }

    #[test]
    fn test_crown_tighter_than_ibp() {
        // 2-layer network: W1 @ relu(W0 @ x + b0) + b1
        let layers = vec![
            linear_layer_f64(arr2(&[[1.0f64, -1.0], [-1.0, 1.0]]), arr1(&[0.0f64, 0.0])),
            SequentialLayerF64::Relu,
            linear_layer_f64(arr2(&[[1.0f64, 1.0]]), arr1(&[0.0f64])),
        ];
        let input = BoundedTensor64::new(
            arr1(&[-1.0f64, -1.0]).into_dyn(),
            arr1(&[1.0f64, 1.0]).into_dyn(),
        )
        .unwrap();

        let ibp = propagate_ibp_f64(&layers, &input).unwrap();
        let crown = propagate_crown_f64(&layers, &input).unwrap();

        // CROWN should be at least as tight as IBP
        assert!(
            crown.lower()[0] >= ibp.lower()[0] - 1e-10,
            "CROWN lower {} >= IBP lower {}",
            crown.lower()[0],
            ibp.lower()[0]
        );
        assert!(
            crown.upper()[0] <= ibp.upper()[0] + 1e-10,
            "CROWN upper {} <= IBP upper {}",
            crown.upper()[0],
            ibp.upper()[0]
        );
    }

    #[test]
    fn test_f64_propagation_mode_dispatch() {
        let layers = vec![linear_layer_f64(arr2(&[[1.0f64]]), arr1(&[0.0f64]))];
        let input =
            BoundedTensor64::new(arr1(&[-1.0f64]).into_dyn(), arr1(&[1.0f64]).into_dyn()).unwrap();

        let ibp = propagate_network_f64(&layers, &input, F64PropagationMode::Ibp).unwrap();
        let crown = propagate_network_f64(&layers, &input, F64PropagationMode::Crown).unwrap();

        assert_eq!(ibp.lower()[0], -1.0);
        assert_eq!(crown.lower()[0], -1.0);
    }

    #[test]
    fn test_intersect_bounds_f64_disjoint_falls_back_to_union() {
        // Simulate disjoint IBP and CROWN bounds (shouldn't happen in
        // correct propagation, but can due to numerical imprecision).
        let a = BoundedTensor64::new(
            arr1(&[1.0f64, 5.0]).into_dyn(),
            arr1(&[3.0f64, 7.0]).into_dyn(),
        )
        .unwrap();
        let b = BoundedTensor64::new(
            arr1(&[2.0f64, 8.0]).into_dyn(), // dim 1: disjoint (8 > 7)
            arr1(&[4.0f64, 10.0]).into_dyn(),
        )
        .unwrap();

        let result = intersect_bounds_f64(&a, &b).unwrap();

        // dim 0: intersection [max(1,2), min(3,4)] = [2, 3]
        assert_eq!(result.lower()[0], 2.0);
        assert_eq!(result.upper()[0], 3.0);
        // dim 1: disjoint → union [min(5,8), max(7,10)] = [5, 10]
        assert_eq!(result.lower()[1], 5.0);
        assert_eq!(result.upper()[1], 10.0);
    }
}
