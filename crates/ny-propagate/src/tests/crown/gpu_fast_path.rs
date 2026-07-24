// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for the sequential GPU CROWN fast-path (#3397).

use super::helpers::{assert_bounded_tensor_close, assert_bounds_finite, MockGpuCrownEngine};
use super::*;
use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use ny_core::{NaiveCpuGemmEngine, Result};

/// Build an ACAS-Xu-like network: SubConst → Linear(no bias) → AddConst → ReLU → Linear → AddConst
///
/// This pattern emerges from ONNX MatMul+Add where MatMul→Linear(w, None) and Add→AddConstant.
/// Part of #3460: GPU CROWN must handle AddConstant/SubConstant.
fn build_acasxu_like_network() -> Result<(Network, BoundedTensor)> {
    use crate::layers::arithmetic::{AddConstantLayer, SubConstantLayer};

    let mut network = Network::new();
    // Input normalization: x - mean
    network.add_layer(Layer::SubConstant(SubConstantLayer::new(
        arr1(&[0.2, -0.1]).into_dyn(),
    )));
    // Hidden layer (no bias — bias comes from AddConstant)
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.4, -0.1], [0.2, 0.3], [-0.5, 0.7]]),
        None,
    )?));
    // Bias addition via AddConstant
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(
        arr1(&[0.1, -0.2, 0.05]).into_dyn(),
    )));
    network.add_layer(Layer::ReLU(ReLULayer));
    // Output layer (no bias)
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]),
        None,
    )?));
    // Output bias via AddConstant
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(
        arr1(&[0.0, 0.15]).into_dyn(),
    )));

    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.75]).into_dyn(),
    )?;
    Ok((network, input))
}

fn build_linear_relu_network() -> Result<(Network, BoundedTensor)> {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.4, -0.1], [0.2, 0.3], [-0.5, 0.7]]),
        Some(arr1(&[0.1, -0.2, 0.05])),
    )?));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]),
        Some(arr1(&[0.0, 0.15])),
    )?));

    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.75]).into_dyn(),
    )?;
    Ok((network, input))
}

fn build_conv_relu_flatten_linear_network() -> Result<(Network, BoundedTensor)> {
    let mut network = Network::new();

    let conv_kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5, -0.25, 0.75, 0.1])
        .expect("conv kernel shape should be valid");
    let conv_bias = Array1::from_vec(vec![0.2]);
    let conv = Conv2dLayer::with_input_shape(conv_kernel, Some(conv_bias), (1, 1), (0, 0), 3, 3)?;

    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::flatten_all()));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.3, -0.2, 0.4, 0.1], [-0.5, 0.2, 0.1, 0.6]]),
        Some(arr1(&[0.1, -0.05])),
    )?));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3, 3]), vec![-0.5; 9])
            .expect("input lower shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[1, 3, 3]), vec![0.75; 9])
            .expect("input upper shape should be valid"),
    )?;

    Ok((network, input))
}

#[test]
fn test_propagate_crown_uses_gpu_fast_path_for_supported_conv_network() -> Result<()> {
    let (network, input) = build_conv_relu_flatten_linear_network()?;
    let cpu_engine = NaiveCpuGemmEngine;
    let expected = network.propagate_crown_with_engine(&input, Some(&cpu_engine))?;
    let mock_gpu = MockGpuCrownEngine::succeed(&expected);

    let actual = network.propagate_crown_with_engine(&input, Some(&mock_gpu))?;

    assert_bounds_finite(&actual, "gpu fast-path conv network output");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, "gpu vs cpu bounds");
    assert!(
        mock_gpu.gpu_calls() >= 1,
        "GPU fast-path should run at least once (also called from CROWN-IBP collection #3599)"
    );
    assert_eq!(mock_gpu.observed_num_specs(), Some(expected.len()));
    assert_eq!(
        mock_gpu.observed_layer_kinds(),
        Some(vec!["Linear", "Activation", "Conv2d"]),
        "Flatten should be skipped when extracting GPU layers"
    );
    Ok(())
}

#[test]
fn test_propagate_crown_skips_gpu_fast_path_for_unsupported_layers() -> Result<()> {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.2, -0.1], [0.4, 0.3], [-0.6, 0.5]]),
        Some(arr1(&[0.1, 0.0, -0.2])),
    )?));
    network.add_layer(Layer::GELU(GELULayer::default()));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.7, -0.3, 0.2]]),
        Some(arr1(&[0.05])),
    )?));

    let input = BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())?;
    let cpu_engine = NaiveCpuGemmEngine;
    let expected = network.propagate_crown_with_engine(&input, Some(&cpu_engine))?;
    let mock_gpu = MockGpuCrownEngine::succeed(&expected);

    let actual = network.propagate_crown_with_engine(&input, Some(&mock_gpu))?;

    assert_bounds_finite(&actual, "gpu fast-path unsupported layers output");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, "gpu vs cpu bounds");
    assert_eq!(
        mock_gpu.gpu_calls(),
        0,
        "unsupported layers should bypass GPU fast-path extraction"
    );
    Ok(())
}

#[test]
fn test_propagate_crown_falls_back_to_cpu_when_gpu_fast_path_errors() -> Result<()> {
    // Exercises the FAST (unsound f32) GPU CROWN path, which the
    // process-global soundness gate masks by default — hold the shared gate
    // lock (it sets the gate OFF) instead of depending on a gate-flipping
    // test elsewhere having leaked an OFF state.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    let (network, input) = build_linear_relu_network()?;
    let cpu_engine = NaiveCpuGemmEngine;
    let expected = network.propagate_crown_with_engine(&input, Some(&cpu_engine))?;
    let mock_gpu = MockGpuCrownEngine::fail();

    let actual = network.propagate_crown_with_engine(&input, Some(&mock_gpu))?;

    assert_bounds_finite(&actual, "gpu fast-path fallback output");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, "gpu vs cpu bounds");
    assert!(
        mock_gpu.gpu_calls() >= 1,
        "GPU fast-path should be attempted (also called from CROWN-IBP collection #3599)"
    );
    assert_eq!(
        mock_gpu.observed_layer_kinds(),
        Some(vec!["Linear", "Activation", "Linear"])
    );
    Ok(())
}

/// Verify GPU extraction succeeds for ACAS-Xu-like networks with AddConstant/SubConstant.
///
/// Part of #3460: constant-arithmetic layers expressed as Activation(slopes, intercepts).
#[test]
fn test_gpu_fast_path_extracts_add_sub_constant_layers() -> Result<()> {
    // Fast-GPU-path routing test: hold the shared gate lock (sets the
    // soundness gate OFF) — see test_propagate_crown_falls_back_to_cpu_*.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    let (network, input) = build_acasxu_like_network()?;
    let cpu_engine = NaiveCpuGemmEngine;
    let expected = network.propagate_crown_with_engine(&input, Some(&cpu_engine))?;
    let mock_gpu = MockGpuCrownEngine::succeed(&expected);

    let actual = network.propagate_crown_with_engine(&input, Some(&mock_gpu))?;

    assert_bounds_finite(&actual, "gpu fast-path AddConstant/SubConstant output");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, "gpu vs cpu bounds");
    assert!(
        mock_gpu.gpu_calls() >= 1,
        "GPU fast-path should succeed for AddConstant/SubConstant network"
    );
    // Backward order: AddConst(out) → Linear(out) → ReLU → AddConst(bias) → Linear(hidden) → SubConst(norm)
    // All constant-arithmetic layers become Activation in GPU representation.
    assert_eq!(
        mock_gpu.observed_layer_kinds(),
        Some(vec![
            "Activation", // AddConstant (output bias)
            "Linear",     // Linear (output weights)
            "Activation", // ReLU
            "Activation", // AddConstant (hidden bias)
            "Linear",     // Linear (hidden weights)
            "Activation", // SubConstant (input normalization)
        ])
    );
    Ok(())
}

/// Verify GPU extraction succeeds for Linear → Sigmoid → Linear network.
///
/// Part of #3460: Sigmoid extraction produces per-neuron linear relaxation
/// slopes/intercepts using the same sigmoid_linear_relaxation function as
/// CPU CROWN backward. The GPU activation shader applies these generically.
///
/// This test validates:
/// 1. GPU fast-path activates (not skipped) for Sigmoid layers
/// 2. Layer kinds include Activation for the Sigmoid
/// 3. CPU-via-mock bounds match the direct CPU CROWN baseline
#[test]
fn test_gpu_fast_path_extracts_sigmoid_layers() -> Result<()> {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.4, -0.1], [0.2, 0.3], [-0.5, 0.7]]),
        Some(arr1(&[0.1, -0.2, 0.05])),
    )?));
    network.add_layer(Layer::Sigmoid(SigmoidLayer::new()));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]),
        Some(arr1(&[0.0, 0.15])),
    )?));

    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.75]).into_dyn(),
    )?;

    let cpu_engine = NaiveCpuGemmEngine;
    let expected = network.propagate_crown_with_engine(&input, Some(&cpu_engine))?;
    let mock_gpu = MockGpuCrownEngine::succeed(&expected);

    let actual = network.propagate_crown_with_engine(&input, Some(&mock_gpu))?;

    assert_bounds_finite(&actual, "gpu fast-path Sigmoid output");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, "gpu vs cpu bounds");
    assert!(
        mock_gpu.gpu_calls() >= 1,
        "GPU fast-path should succeed for Sigmoid network"
    );
    // Backward order: Linear(output) → Sigmoid → Linear(hidden)
    assert_eq!(
        mock_gpu.observed_layer_kinds(),
        Some(vec!["Linear", "Activation", "Linear"]),
        "Sigmoid should be extracted as Activation"
    );
    Ok(())
}

/// Verify GPU extraction succeeds for Linear → Tanh → Linear network.
///
/// Part of #3460: Tanh extraction uses tanh_linear_relaxation (same function
/// as CPU CROWN backward) to compute per-neuron slopes/intercepts.
#[test]
fn test_gpu_fast_path_extracts_tanh_layers() -> Result<()> {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.3, -0.2], [0.5, 0.1], [-0.4, 0.6]]),
        Some(arr1(&[-0.1, 0.3, 0.0])),
    )?));
    network.add_layer(Layer::Tanh(TanhLayer::new()));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.7, -0.3, 0.2], [-0.1, 0.4, 0.5]]),
        Some(arr1(&[0.05, -0.1])),
    )?));

    let input = BoundedTensor::new(
        arr1(&[-0.5, -1.0]).into_dyn(),
        arr1(&[0.75, 1.0]).into_dyn(),
    )?;

    let cpu_engine = NaiveCpuGemmEngine;
    let expected = network.propagate_crown_with_engine(&input, Some(&cpu_engine))?;
    let mock_gpu = MockGpuCrownEngine::succeed(&expected);

    let actual = network.propagate_crown_with_engine(&input, Some(&mock_gpu))?;

    assert_bounds_finite(&actual, "gpu fast-path Tanh output");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, "gpu vs cpu bounds");
    assert!(
        mock_gpu.gpu_calls() >= 1,
        "GPU fast-path should succeed for Tanh network"
    );
    assert_eq!(
        mock_gpu.observed_layer_kinds(),
        Some(vec!["Linear", "Activation", "Linear"]),
        "Tanh should be extracted as Activation"
    );
    Ok(())
}

/// Verify GPU extraction succeeds for MulConstant and DivConstant layers.
///
/// Part of #3460: MulConstant(c) → Activation(slopes=c, intercepts=0),
/// DivConstant(c) → Activation(slopes=1/c, intercepts=0).
#[test]
fn test_gpu_fast_path_extracts_mul_div_constant_layers() -> Result<()> {
    // Fast-GPU-path routing test: hold the shared gate lock (sets the
    // soundness gate OFF) — see test_propagate_crown_falls_back_to_cpu_*.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    use crate::layers::arithmetic::{DivConstantLayer, MulConstantLayer};

    let mut network = Network::new();
    // Scale input by constant
    network.add_layer(Layer::MulConstant(MulConstantLayer::new(
        arr1(&[2.0, 0.5]).into_dyn(),
    )));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.4, -0.1], [0.2, 0.3]]),
        Some(arr1(&[0.1, -0.2])),
    )?));
    // Divide by constant (e.g., attention scaling)
    network.add_layer(Layer::DivConstant(DivConstantLayer::new(
        arr1(&[4.0, 4.0]).into_dyn(),
    )));

    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.75]).into_dyn(),
    )?;

    let cpu_engine = NaiveCpuGemmEngine;
    let expected = network.propagate_crown_with_engine(&input, Some(&cpu_engine))?;
    let mock_gpu = MockGpuCrownEngine::succeed(&expected);

    let actual = network.propagate_crown_with_engine(&input, Some(&mock_gpu))?;

    assert_bounds_finite(&actual, "gpu fast-path MulConstant/DivConstant output");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, "gpu vs cpu bounds");
    assert!(
        mock_gpu.gpu_calls() >= 1,
        "GPU fast-path should succeed for MulConstant/DivConstant network"
    );
    // Backward: DivConst → Linear → MulConst
    assert_eq!(
        mock_gpu.observed_layer_kinds(),
        Some(vec!["Activation", "Linear", "Activation"])
    );
    Ok(())
}
