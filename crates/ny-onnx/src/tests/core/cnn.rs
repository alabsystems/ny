// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end CNN verification tests for Patches mode gate (#2613).
//!
//! Tests load ONNX CNN models, run CROWN propagation through the full
//! ONNX → GraphNetwork → propagate_crown pipeline, and verify soundness.

use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use ny_test_utils::workspace_root;

/// Assert CROWN bounds are no wider than IBP bounds (tightening invariant).
fn assert_crown_within_ibp(crown: &BoundedTensor, ibp: &BoundedTensor, label: &str) {
    for (i, (((cl, cu), il), iu)) in crown
        .lower()
        .iter()
        .zip(crown.upper().iter())
        .zip(ibp.lower().iter())
        .zip(ibp.upper().iter())
        .enumerate()
    {
        assert!(
            *cl >= *il - 1e-4,
            "{}: CROWN lower[{}]={} looser than IBP lower={}",
            label,
            i,
            cl,
            il
        );
        assert!(
            *cu <= *iu + 1e-4,
            "{}: CROWN upper[{}]={} looser than IBP upper={}",
            label,
            i,
            cu,
            iu
        );
    }
}

/// Assert concrete output values are within CROWN bounds (soundness check).
fn assert_concrete_within_crown(concrete_out: &BoundedTensor, crown: &BoundedTensor, label: &str) {
    for (i, ((val, cl), cu)) in concrete_out
        .lower()
        .iter()
        .zip(crown.lower().iter())
        .zip(crown.upper().iter())
        .enumerate()
    {
        assert!(
            *val >= *cl - 1e-4 && *val <= *cu + 1e-4,
            "{}: output[{}]={} outside CROWN bounds [{}, {}]",
            label,
            i,
            val,
            cl,
            cu
        );
    }
}

/// Assert CROWN bounds contain no NaN or Inf values.
fn assert_crown_finite(crown: &BoundedTensor, label: &str) {
    assert!(
        !crown.lower().iter().any(|v| v.is_nan() || v.is_infinite()),
        "{}: CROWN lower bounds contain NaN or Inf",
        label
    );
    assert!(
        !crown.upper().iter().any(|v| v.is_nan() || v.is_infinite()),
        "{}: CROWN upper bounds contain NaN or Inf",
        label
    );
}

/// Run a single CNN verification instance: IBP, CROWN, soundness, tightening.
fn verify_cnn_instance(model: &OnnxModel, center: &ArrayD<f32>, eps: f32, id: usize) {
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input = BoundedTensor::new(lower.clone(), upper.clone())
        .unwrap_or_else(|e| panic!("Instance {id}: BoundedTensor: {e}"));

    let graph = model
        .to_graph_network()
        .unwrap_or_else(|e| panic!("Instance {id}: to_graph_network: {e}"));
    let seq = model
        .to_propagate_network()
        .unwrap_or_else(|e| panic!("Instance {id}: to_propagate_network: {e}"));

    let ibp = graph
        .propagate_ibp(&input)
        .unwrap_or_else(|e| panic!("Instance {id}: IBP: {e}"));
    let crown = graph
        .propagate_crown(&input)
        .unwrap_or_else(|e| panic!("Instance {id}: CROWN: {e}"));

    let label = format!("Instance {id}");
    assert_crown_within_ibp(&crown, &ibp, &label);
    assert_crown_finite(&crown, &label);

    // Soundness: sample center, lower corner, upper corner
    for (j, point) in [center.clone(), lower, upper].iter().enumerate() {
        let concrete = BoundedTensor::new(point.clone(), point.clone()).unwrap();
        let out = seq
            .propagate_ibp(&concrete)
            .unwrap_or_else(|e| panic!("Instance {id}, sample {j}: {e}"));
        assert_concrete_within_crown(&out, &crown, &format!("Instance {id} sample {j}"));
    }
}

/// Gate test: MNIST-CONV classifier verified with 5 instances via CROWN.
///
/// Loads mnist_conv.onnx (Conv→ReLU→Flatten→Linear), creates 5 bounded input
/// regions with different centers and epsilon, verifies CROWN propagation
/// produces sound bounds that are tighter than IBP.
///
/// Primary gate criterion for #2613 Patches Mode.
#[ntest::timeout(60000)]
#[test]
fn test_mnist_conv_crown_5_instances_2613() {
    let path = require_test_model("mnist_conv.onnx");
    let model = load_onnx(&path).expect("Failed to load mnist_conv.onnx");

    let input_shape = &[1_usize, 8, 8]; // after batch dim removal
    let n = input_shape.iter().product::<usize>();

    let instances: Vec<(Vec<f32>, f32)> = vec![
        (vec![0.0; n], 0.1),
        (vec![0.5; n], 0.05),
        (vec![0.0; n], 0.3),
        (
            (0..n)
                .map(|i| ((i as f32) / (n as f32) - 0.5) * 2.0)
                .collect(),
            0.1,
        ),
        (vec![0.25; n], 0.2),
    ];

    for (idx, (vals, eps)) in instances.iter().enumerate() {
        let center = ArrayD::from_shape_vec(IxDyn(input_shape), vals.clone()).unwrap();
        verify_cnn_instance(&model, &center, *eps, idx);
    }
}

/// Test CROWN on conv_relu.onnx (spatial output) — directly triggers Patches mode.
///
/// conv_relu.onnx: Conv2d(1→2, 2×2) + ReLU, input (1,4,4) → output (2,3,3).
/// Output is 3D spatial, so CROWN backward starts in Patches mode.
#[ntest::timeout(10000)]
#[test]
fn test_conv_relu_crown_patches_spatial_output_2613() {
    let path = require_test_model("conv_relu.onnx");
    let model = load_onnx(&path).expect("Failed to load conv_relu.onnx");
    let shape = &[1_usize, 4, 4];

    let center = ArrayD::zeros(IxDyn(shape));
    let input =
        BoundedTensor::new(center.mapv(|v: f32| v - 0.5), center.mapv(|v: f32| v + 0.5)).unwrap();

    let graph = model.to_graph_network().expect("to_graph_network failed");
    let seq = model
        .to_propagate_network()
        .expect("to_propagate_network failed");

    let ibp = graph.propagate_ibp(&input).expect("IBP failed");
    let crown = graph.propagate_crown(&input).expect("CROWN failed");

    assert_crown_within_ibp(&crown, &ibp, "conv_relu");
    assert_crown_finite(&crown, "conv_relu");

    let concrete = BoundedTensor::new(center.clone(), center).unwrap();
    let out = seq
        .propagate_ibp(&concrete)
        .expect("Concrete propagation failed");
    assert_concrete_within_crown(&out, &crown, "conv_relu");
}

/// Test CROWN on cnn_with_flatten.onnx — Conv+ReLU+MaxPool+Flatten+Linear.
///
/// Exercises the full CNN classifier pipeline through ONNX loading.
#[ntest::timeout(60000)]
#[test]
fn test_cnn_with_flatten_crown_soundness_2613() {
    let path = require_test_model("cnn_with_flatten.onnx");
    let model = load_onnx(&path).expect("Failed to load cnn_with_flatten.onnx");

    let graph = model.to_graph_network().expect("to_graph_network failed");
    let seq = model
        .to_propagate_network()
        .expect("to_propagate_network failed");

    // Derive propagation input shape (strip batch dim)
    let input_spec = model.network.inputs.first().expect("No input spec");
    let prop_shape: Vec<usize> = input_spec.shape[1..]
        .iter()
        .map(|&d| if d > 0 { d as usize } else { 1 })
        .collect();

    let center = ArrayD::zeros(IxDyn(&prop_shape));
    let input =
        BoundedTensor::new(center.mapv(|v: f32| v - 0.1), center.mapv(|v: f32| v + 0.1)).unwrap();

    let ibp = graph.propagate_ibp(&input).expect("IBP failed");
    let crown = graph.propagate_crown(&input).expect("CROWN failed");

    assert_crown_finite(&crown, "cnn_with_flatten");
    assert_crown_within_ibp(&crown, &ibp, "cnn_with_flatten");

    let concrete = BoundedTensor::new(center.clone(), center).unwrap();
    let out = seq
        .propagate_ibp(&concrete)
        .expect("Concrete propagation failed");
    assert_concrete_within_crown(&out, &crown, "cnn_with_flatten");
}

/// Require a separately downloaded VNN-COMP benchmark model.
fn require_external_benchmark_model(relative_path: &str) -> String {
    let path = workspace_root().join(relative_path);
    assert!(
        path.is_file(),
        "external benchmark model fixture is missing at {}; \
         run benchmarks/download_benchmarks.sh",
        path.display()
    );
    path.display().to_string()
}

/// Gate test (#2613): CIFAR-10 ResNet 2-block loads and IBP verifies correctly.
///
/// Loads the VNN-COMP 2021 `resnet_2b.onnx` model (17 nodes, 2 residual
/// blocks with skip connections) and runs IBP with 3 input centers.
/// Verifies ONNX → GraphNetwork pipeline works for real ResNet architecture.
///
/// Model: Conv(3→16) → [ResBlock(16→16)] × 2 → Flatten → FC → ReLU → FC(10)
/// Input: [3, 32, 32], Output: [10], eps: 0.008 (VNN-COMP standard)
#[ntest::timeout(60000)]
#[test]
#[cfg(feature = "external-vnncomp")]
fn test_cifar10_resnet_2b_ibp_gate_2613() {
    let path = require_external_benchmark_model(
        "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx",
    );
    let model = load_onnx(&path).expect("Failed to load resnet_2b.onnx");
    let graph = model
        .to_graph_network()
        .expect("resnet_2b: to_graph_network failed");

    let input_shape = &[3_usize, 32, 32];
    let n = input_shape.iter().product::<usize>();
    let eps = 0.008_f32;

    let centers: Vec<(&str, Vec<f32>)> = vec![
        ("mid", vec![0.5; n]),
        ("gradient", (0..n).map(|i| i as f32 / n as f32).collect()),
        ("low", vec![0.1; n]),
    ];
    for (label, vals) in &centers {
        let center = ArrayD::from_shape_vec(IxDyn(input_shape), vals.clone()).unwrap();
        let lower = center.mapv(|v| (v - eps).max(0.0));
        let upper = center.mapv(|v| (v + eps).min(1.0));
        let input = BoundedTensor::new(lower, upper)
            .unwrap_or_else(|e| panic!("resnet_2b {label}: BoundedTensor: {e}"));

        let ibp = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("resnet_2b {label}: IBP: {e}"));
        assert_eq!(ibp.len(), 10, "resnet_2b {label}: expected 10 outputs");

        // Soundness: concrete output at center within IBP bounds
        let concrete = BoundedTensor::new(center.clone(), center.clone()).unwrap();
        let out = graph
            .propagate_ibp(&concrete)
            .unwrap_or_else(|e| panic!("resnet_2b {label}: concrete: {e}"));
        assert_concrete_within_crown(&out, &ibp, &format!("resnet_2b {label}"));
    }
}

/// Gate test (#2613): CIFAR-10 ResNet CROWN produces sound bounds.
///
/// Runs full CROWN backward on resnet_2b.onnx with a single input center.
/// This exercises the graph engine's Patches→Dense transitions at residual
/// connections and verifies CROWN bounds are sound and tighter than IBP.
///
/// Note: Full CROWN on a 3072-dim ResNet is slow in debug mode (~300s).
/// Use release mode for benchmarking. The timeout here allows debug builds.
#[ntest::timeout(600000)]
#[test]
#[cfg(feature = "external-vnncomp")]
fn test_cifar10_resnet_2b_crown_gate_2613() {
    let path = require_external_benchmark_model(
        "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx",
    );
    let model = load_onnx(&path).expect("Failed to load resnet_2b.onnx");
    let graph = model
        .to_graph_network()
        .expect("resnet_2b: to_graph_network failed");

    let input_shape = &[3_usize, 32, 32];
    let eps = 0.008_f32;
    let center = ArrayD::from_elem(IxDyn(input_shape), 0.5_f32);
    let lower = center.mapv(|v| (v - eps).max(0.0));
    let upper = center.mapv(|v| (v + eps).min(1.0));
    let input = BoundedTensor::new(lower, upper).expect("BoundedTensor");

    let ibp = graph.propagate_ibp(&input).expect("IBP failed");
    let crown = graph.propagate_crown(&input).expect("CROWN failed");

    assert_eq!(crown.len(), 10, "CROWN expected 10 outputs");
    assert_crown_finite(&crown, "resnet_2b");
    assert_crown_within_ibp(&crown, &ibp, "resnet_2b");

    // Soundness: concrete output within CROWN bounds
    let concrete_input = BoundedTensor::new(center.clone(), center).unwrap();
    let out = graph.propagate_ibp(&concrete_input).expect("concrete");
    assert_concrete_within_crown(&out, &crown, "resnet_2b concrete");

    // Non-trivial width
    let has_width = crown
        .lower()
        .iter()
        .zip(crown.upper().iter())
        .any(|(l, u)| u - l > 1e-6);
    assert!(has_width, "CROWN bounds have zero width (degenerate)");
}
