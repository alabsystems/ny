// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conv1d-specific coverage for the sequential GPU CROWN fast-path (#3549).

use super::helpers::{assert_bounded_tensor_close, assert_bounds_finite, MockGpuCrownEngine};
use super::*;
use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use ny_core::{NaiveCpuGemmEngine, Result};

fn build_conv1d_relu_flatten_linear_network() -> Result<(Network, BoundedTensor)> {
    let mut network = Network::new();

    let conv_kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.5, -0.25, 0.75, -0.2, 0.4, 0.1])
            .expect("conv1d kernel shape should be valid");
    let conv_bias = Array1::from_vec(vec![0.15, -0.05]);
    let conv = Conv1dLayer::with_input_length(conv_kernel, Some(conv_bias), 1, 1, 6)?;

    network.add_layer(Layer::Conv1d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::flatten_all()));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[
            [
                0.2, -0.1, 0.05, 0.3, -0.25, 0.4, -0.2, 0.15, 0.35, -0.05, 0.1, 0.25,
            ],
            [
                -0.3, 0.25, 0.15, -0.2, 0.1, -0.35, 0.4, -0.1, 0.05, 0.2, -0.15, 0.3,
            ],
        ]),
        Some(arr1(&[0.05, -0.1])),
    )?));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![-0.5, -0.25, 0.0, -0.1, -0.2, -0.3])
            .expect("conv1d input lower shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![0.75, 0.5, 0.4, 0.6, 0.8, 0.7])
            .expect("conv1d input upper shape should be valid"),
    )?;

    Ok((network, input))
}

#[test]
fn test_propagate_crown_uses_gpu_fast_path_for_supported_conv1d_network() -> Result<()> {
    // Exercises the FAST (unsound f32) GPU CROWN path, which the
    // process-global soundness gate masks by default — hold the shared gate
    // lock (it sets the gate OFF) instead of depending on a gate-flipping
    // test elsewhere having leaked an OFF state.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    let (network, input) = build_conv1d_relu_flatten_linear_network()?;
    let cpu_engine = NaiveCpuGemmEngine;
    let expected = network.propagate_crown_with_engine(&input, Some(&cpu_engine))?;
    let mock_gpu = MockGpuCrownEngine::succeed(&expected);

    let actual = network.propagate_crown_with_engine(&input, Some(&mock_gpu))?;

    assert_bounds_finite(&actual, "gpu fast-path Conv1d output");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, "conv1d gpu vs cpu bounds");
    assert!(
        mock_gpu.gpu_calls() >= 1,
        "GPU fast-path should run for Conv1d networks once Conv1d extracts as height-1 Conv2d"
    );
    assert_eq!(mock_gpu.observed_num_specs(), Some(expected.len()));
    assert_eq!(
        mock_gpu.observed_layer_kinds(),
        Some(vec!["Linear", "Activation", "Conv2d"]),
        "Flatten should be skipped and Conv1d should reuse the Conv2d GPU descriptor"
    );
    Ok(())
}
