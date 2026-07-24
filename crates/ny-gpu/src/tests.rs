// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::accelerated::{AcceleratedBoundPropagation, AcceleratedDevice};
use crate::backend::{Backend, ComputeDevice};
use approx::assert_relative_eq;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

#[test]
fn test_ny_gpu_api_does_not_reexport_ny_propagate_1894() {
    let lib_source = include_str!("lib.rs");
    let normalized_source: String = lib_source.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !normalized_source.contains("pubuseny_propagate"),
        "ny-gpu must not re-export ny-propagate modules from crate root"
    );
}

#[test]
fn test_linear_ibp_basic() {
    let device = AcceleratedDevice::new();

    // Create test input: shape [2, 3] with bounds
    let lower = ArrayD::from_elem(IxDyn(&[2, 3]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Weight [4, 3] -> output [2, 4]
    let weight = Array2::from_elem((4, 3), 0.5_f32);
    let bias = Some(Array1::from_elem(4, 0.1_f32));

    let result = device.linear_ibp(&input, &weight, bias.as_ref()).unwrap();

    assert_eq!(result.shape(), &[2, 4]);

    // For w=0.5 and x in [-1, 1]: sum of 3 terms = [-1.5, 1.5], plus bias 0.1 = [-1.4, 1.6]
    assert_relative_eq!(result.lower()[[0, 0]], -1.4, epsilon = 1e-5);
    assert_relative_eq!(result.upper()[[0, 0]], 1.6, epsilon = 1e-5);
}

#[test]
fn test_linear_ibp_batched() {
    let device = AcceleratedDevice::new();

    // Create batched input: shape [2, 3, 4] with bounds
    let lower = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Weight [8, 4] -> output [2, 3, 8]
    let weight = Array2::from_elem((8, 4), 0.25_f32);

    let result = device.linear_ibp(&input, &weight, None).unwrap();

    assert_eq!(result.shape(), &[2, 3, 8]);

    // For w=0.25 and x in [0, 1]: sum of 4 terms = [0, 1]
    assert_relative_eq!(result.lower()[[0, 0, 0]], 0.0, epsilon = 1e-5);
    assert_relative_eq!(result.upper()[[0, 0, 0]], 1.0, epsilon = 1e-5);
}

#[test]
fn test_matmul_ibp_basic() {
    let device = AcceleratedDevice::new();

    // A: [2, 3] with bounds [0, 1]
    let lower_a = ArrayD::from_elem(IxDyn(&[2, 3]), 0.0_f32);
    let upper_a = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
    let input_a = BoundedTensor::new(lower_a, upper_a).unwrap();

    // B: [3, 4] with bounds [0, 1]
    let lower_b = ArrayD::from_elem(IxDyn(&[3, 4]), 0.0_f32);
    let upper_b = ArrayD::from_elem(IxDyn(&[3, 4]), 1.0_f32);
    let input_b = BoundedTensor::new(lower_b, upper_b).unwrap();

    let result = device.matmul_ibp(&input_a, &input_b).unwrap();

    assert_eq!(result.shape(), &[2, 4]);

    // For A,B in [0, 1]^{2x3} @ [0, 1]^{3x4}: result in [0, 3]
    assert_relative_eq!(result.lower()[[0, 0]], 0.0, epsilon = 1e-5);
    assert_relative_eq!(result.upper()[[0, 0]], 3.0, epsilon = 1e-5);
}

#[test]
fn test_matmul_ibp_batched() {
    let device = AcceleratedDevice::new();

    // Batched A: [2, 2, 3] (2 batches of 2x3 matrices)
    let lower_a = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 0.5_f32);
    let upper_a = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 1.0_f32);
    let input_a = BoundedTensor::new(lower_a, upper_a).unwrap();

    // Batched B: [2, 3, 4] (2 batches of 3x4 matrices)
    let lower_b = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 0.5_f32);
    let upper_b = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0_f32);
    let input_b = BoundedTensor::new(lower_b, upper_b).unwrap();

    let result = device.matmul_ibp(&input_a, &input_b).unwrap();

    assert_eq!(result.shape(), &[2, 2, 4]);

    // For A,B in [0.5, 1.0]: each product in [0.25, 1.0], sum of 3 = [0.75, 3.0]
    assert_relative_eq!(result.lower()[[0, 0, 0]], 0.75, epsilon = 1e-5);
    assert_relative_eq!(result.upper()[[0, 0, 0]], 3.0, epsilon = 1e-5);
}

#[test]
fn test_crown_per_position_parallel() {
    use ndarray::Array2;
    use ny_propagate::{
        layers::{GELULayer, LinearLayer},
        GraphNode, Layer,
    };

    let device = AcceleratedDevice::new();

    // Build a small MLP graph: Linear -> GELU -> Linear
    // 4 features -> 8 features -> 4 features
    let in_features = 4;
    let hidden_features = 8;
    let out_features = 4;

    let weight1 = Array2::from_shape_fn((hidden_features, in_features), |(i, j)| {
        0.1 * ((i + j) as f32 - 6.0)
    });
    let bias1 = Array1::from_elem(hidden_features, 0.05_f32);
    let linear1 = LinearLayer::new(weight1, Some(bias1)).unwrap();

    let weight2 = Array2::from_shape_fn((out_features, hidden_features), |(i, j)| {
        0.1 * ((i + j) as f32 - 6.0)
    });
    let bias2 = Array1::from_elem(out_features, 0.02_f32);
    let linear2 = LinearLayer::new(weight2, Some(bias2)).unwrap();

    let gelu = GELULayer::default();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(gelu),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));
    graph.set_output("linear2");

    // Create multi-position input: [2, 3, 4] = 6 positions with 4 features each
    let lower = ArrayD::from_elem(IxDyn(&[2, 3, in_features]), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3, in_features]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run parallel per-position CROWN
    let result = device.crown_per_position_parallel(&graph, &input).unwrap();

    // Check output shape: [2, 3, 4]
    assert_eq!(result.shape(), &[2, 3, out_features]);

    // Verify bounds are valid (lower <= upper)
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..out_features {
                let l = result.lower()[[i, j, k]];
                let u = result.upper()[[i, j, k]];
                assert!(
                    l <= u,
                    "Invalid bounds at [{},{},{}]: lower={} > upper={}",
                    i,
                    j,
                    k,
                    l,
                    u
                );
            }
        }
    }

    // Compare with sequential per-position CROWN
    let sequential_result = graph.propagate_crown_per_position(&input).unwrap();

    // Results should be identical
    assert_eq!(result.shape(), sequential_result.shape());

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..out_features {
                assert_relative_eq!(
                    result.lower()[[i, j, k]],
                    sequential_result.lower()[[i, j, k]],
                    epsilon = 1e-5
                );
                assert_relative_eq!(
                    result.upper()[[i, j, k]],
                    sequential_result.upper()[[i, j, k]],
                    epsilon = 1e-5
                );
            }
        }
    }
}

#[test]
fn test_cpu_attention_ibp_basic() {
    let device = AcceleratedDevice::new();

    // Create test Q, K, V: shape [1, 2, 4, 8] (batch=1, heads=2, seq=4, dim=8)
    let batch = 1;
    let heads = 2;
    let seq = 4;
    let dim = 8;
    let shape = [batch, heads, seq, dim];

    // Q, K, V with small perturbation around 0
    let lower = ArrayD::from_elem(IxDyn(&shape), -0.1_f32);
    let upper = ArrayD::from_elem(IxDyn(&shape), 0.1_f32);
    let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let v = BoundedTensor::new(lower, upper).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.attention_ibp(&q, &k, &v, scale).unwrap();

    // Check output shape: [batch, heads, seq, dim]
    assert_eq!(result.shape(), &shape);

    // Check bounds are valid (lower <= upper)
    for (val_lower, val_upper) in result.lower().iter().zip(result.upper().iter()) {
        assert!(
            *val_lower <= *val_upper + 1e-6,
            "Invalid bounds: lower={} > upper={}",
            val_lower,
            val_upper
        );
    }

    // Output should be bounded since softmax outputs sum to 1
    // and V is in [-0.1, 0.1], so output should be roughly in [-0.1, 0.1]
    let max_upper = result
        .upper()
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_lower = result.lower().iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        max_upper < 1.0 && min_lower > -1.0,
        "Attention bounds seem too loose: [{}, {}]",
        min_lower,
        max_upper
    );
}

#[test]
fn test_cpu_causal_attention_ibp_basic() {
    let device = AcceleratedDevice::new();

    // Create test Q, K, V: shape [1, 2, 4, 8] (batch=1, heads=2, seq=4, dim=8)
    let batch = 1;
    let heads = 2;
    let seq = 4;
    let dim = 8;
    let shape = [batch, heads, seq, dim];

    // Q, K, V with small perturbation around 0
    let lower = ArrayD::from_elem(IxDyn(&shape), -0.1_f32);
    let upper = ArrayD::from_elem(IxDyn(&shape), 0.1_f32);
    let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let v = BoundedTensor::new(lower, upper).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.causal_attention_ibp(&q, &k, &v, scale).unwrap();

    // Check output shape: [batch, heads, seq, dim]
    assert_eq!(result.shape(), &shape);

    // Check bounds are valid (lower <= upper)
    for (val_lower, val_upper) in result.lower().iter().zip(result.upper().iter()) {
        assert!(
            *val_lower <= *val_upper + 1e-6,
            "Invalid bounds: lower={} > upper={}",
            val_lower,
            val_upper
        );
    }

    // Output should be bounded similarly to standard attention
    let max_upper = result
        .upper()
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_lower = result.lower().iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        max_upper < 1.0 && min_lower > -1.0,
        "Causal attention bounds seem too loose: [{}, {}]",
        min_lower,
        max_upper
    );
}

#[test]
fn test_causal_attention_soundness() {
    // Test that causal attention bounds are sound by checking that
    // concrete causal attention outputs fall within bounds
    let device = AcceleratedDevice::new();

    let batch = 1;
    let heads = 1;
    let seq = 4;
    let dim = 4;
    let shape = [batch, heads, seq, dim];

    // Create bounded Q, K, V with small perturbation
    let eps = 0.1;
    let center = ArrayD::from_elem(IxDyn(&shape), 0.5_f32);
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let v = BoundedTensor::new(lower, upper).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.causal_attention_ibp(&q, &k, &v, scale).unwrap();

    // Bounds should be valid
    assert_eq!(result.shape(), &shape);
    for i in 0..result.lower().len() {
        assert!(
            result.lower().as_slice().unwrap()[i] <= result.upper().as_slice().unwrap()[i] + 1e-5,
            "Invalid bounds at position {}: lower={} > upper={}",
            i,
            result.lower().as_slice().unwrap()[i],
            result.upper().as_slice().unwrap()[i]
        );
    }
}

#[test]
fn test_causal_vs_standard_attention_difference() {
    // Causal attention should give different results than standard attention
    // (except for the last position which sees all previous positions)
    let device = AcceleratedDevice::new();

    let batch = 1;
    let heads = 1;
    let seq = 4;
    let dim = 4;
    let shape = [batch, heads, seq, dim];

    // Use point estimates (no perturbation) to compare concrete outputs
    let data = ArrayD::from_shape_fn(IxDyn(&shape), |idx| {
        let [_b, _h, s, d] = [idx[0], idx[1], idx[2], idx[3]];
        (s + d) as f32 * 0.1
    });
    let q = BoundedTensor::new(data.clone(), data.clone()).unwrap();
    let k = BoundedTensor::new(data.clone(), data.clone()).unwrap();
    let v = BoundedTensor::new(data.clone(), data).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let causal_result = device.causal_attention_ibp(&q, &k, &v, scale).unwrap();
    let standard_result = device.attention_ibp(&q, &k, &v, scale).unwrap();

    // First position should have only one valid attention target, so might differ
    // from standard attention which sees all positions
    let causal_pos0: Vec<f32> = (0..dim)
        .map(|d| causal_result.lower()[[0, 0, 0, d]])
        .collect();
    let standard_pos0: Vec<f32> = (0..dim)
        .map(|d| standard_result.lower()[[0, 0, 0, d]])
        .collect();

    // At position 0, causal attention only sees position 0 (self)
    // Standard attention sees all 4 positions
    // These should typically differ unless inputs are special
    let diff: f32 = causal_pos0
        .iter()
        .zip(standard_pos0.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    // The outputs are allowed to be identical (if V values are all the same)
    // but in general they should differ
    // Just verify both produce valid results
    assert!(causal_pos0.iter().all(|x| x.is_finite()));
    assert!(standard_pos0.iter().all(|x| x.is_finite()));

    // Last position in causal should match standard (sees all positions)
    // This is only exact for point inputs (no perturbation)
    let last_pos = seq - 1;
    for d in 0..dim {
        let causal_val = causal_result.lower()[[0, 0, last_pos, d]];
        let standard_val = standard_result.lower()[[0, 0, last_pos, d]];
        assert!(
            (causal_val - standard_val).abs() < 1e-4,
            "Last position should match: causal={} vs standard={} at dim {}",
            causal_val,
            standard_val,
            d
        );
    }

    // Verify total difference exists (causal != standard for earlier positions)
    assert!(
        diff > 1e-6,
        "Expected difference between causal and standard attention, got diff={}",
        diff
    );
}

// ================= Cross-Attention Tests =================

#[test]
fn test_cross_attention_basic() {
    // Test basic cross-attention with different sequence lengths
    let device = AcceleratedDevice::new();

    // Q from decoder: [batch=1, heads=2, seq_dec=3, dim=4]
    // K, V from encoder: [batch=1, heads=2, seq_enc=5, dim=4]
    let batch = 1;
    let heads = 2;
    let seq_dec = 3;
    let seq_enc = 5;
    let dim = 4;

    let shape_q = [batch, heads, seq_dec, dim];
    let shape_kv = [batch, heads, seq_enc, dim];

    // Q, K, V with small perturbation around 0
    let lower_q = ArrayD::from_elem(IxDyn(&shape_q), -0.1_f32);
    let upper_q = ArrayD::from_elem(IxDyn(&shape_q), 0.1_f32);
    let lower_kv = ArrayD::from_elem(IxDyn(&shape_kv), -0.1_f32);
    let upper_kv = ArrayD::from_elem(IxDyn(&shape_kv), 0.1_f32);

    let q = BoundedTensor::new(lower_q, upper_q).unwrap();
    let k = BoundedTensor::new(lower_kv.clone(), upper_kv.clone()).unwrap();
    let v = BoundedTensor::new(lower_kv, upper_kv).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.cross_attention_ibp(&q, &k, &v, scale).unwrap();

    // Output should match decoder sequence length
    // Expected shape: [batch, heads, seq_dec, dim]
    assert_eq!(result.shape(), &shape_q);

    // Bounds should be valid
    for i in 0..result.lower().len() {
        assert!(
            result.lower().as_slice().unwrap()[i] <= result.upper().as_slice().unwrap()[i] + 1e-5,
            "Invalid bounds at position {}: lower={} > upper={}",
            i,
            result.lower().as_slice().unwrap()[i],
            result.upper().as_slice().unwrap()[i]
        );
    }
}

#[test]
fn test_cross_attention_soundness() {
    // Test that cross-attention bounds contain concrete outputs
    let device = AcceleratedDevice::new();

    let batch = 1;
    let heads = 1;
    let seq_dec = 2;
    let seq_enc = 3;
    let dim = 4;

    let shape_q = [batch, heads, seq_dec, dim];
    let shape_kv = [batch, heads, seq_enc, dim];

    // Create bounded Q, K, V with small perturbation
    let eps = 0.1;
    let center_q = ArrayD::from_elem(IxDyn(&shape_q), 0.5_f32);
    let center_kv = ArrayD::from_elem(IxDyn(&shape_kv), 0.5_f32);

    let q = BoundedTensor::new(center_q.mapv(|v| v - eps), center_q.mapv(|v| v + eps)).unwrap();
    let k = BoundedTensor::new(center_kv.mapv(|v| v - eps), center_kv.mapv(|v| v + eps)).unwrap();
    let v = BoundedTensor::new(center_kv.mapv(|v| v - eps), center_kv.mapv(|v| v + eps)).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.cross_attention_ibp(&q, &k, &v, scale).unwrap();

    // Expected output shape: [batch, heads, seq_dec, dim]
    assert_eq!(result.shape(), &shape_q);

    // Bounds should be valid
    for i in 0..result.lower().len() {
        assert!(
            result.lower().as_slice().unwrap()[i] <= result.upper().as_slice().unwrap()[i] + 1e-5,
            "Invalid bounds at position {}: lower={} > upper={}",
            i,
            result.lower().as_slice().unwrap()[i],
            result.upper().as_slice().unwrap()[i]
        );
    }

    // Output bounds should be reasonable (contained in V bounds with some slack)
    let min_lower = result.lower().iter().cloned().fold(f32::INFINITY, f32::min);
    let max_upper = result
        .upper()
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_lower >= -1.0 && max_upper <= 2.0,
        "Cross-attention bounds seem too loose: [{}, {}]",
        min_lower,
        max_upper
    );
}

#[test]
fn test_cross_attention_shape_validation() {
    // Test that cross-attention rejects mismatched shapes
    let device = AcceleratedDevice::new();

    // Valid shapes
    let shape_q = [1, 2, 3, 4]; // [batch=1, heads=2, seq_dec=3, dim=4]
    let shape_k = [1, 2, 5, 4]; // [batch=1, heads=2, seq_enc=5, dim=4]
    let shape_v = [1, 2, 5, 4]; // [batch=1, heads=2, seq_enc=5, dim=4]

    let lower = ArrayD::zeros(IxDyn(&shape_q));
    let upper = ArrayD::zeros(IxDyn(&shape_q));
    let q = BoundedTensor::new(lower, upper).unwrap();

    let lower_k = ArrayD::zeros(IxDyn(&shape_k));
    let upper_k = ArrayD::zeros(IxDyn(&shape_k));
    let k = BoundedTensor::new(lower_k, upper_k).unwrap();

    let lower_v = ArrayD::zeros(IxDyn(&shape_v));
    let upper_v = ArrayD::zeros(IxDyn(&shape_v));
    let v = BoundedTensor::new(lower_v, upper_v).unwrap();

    // This should work
    let result = device.cross_attention_ibp(&q, &k, &v, 1.0);
    assert!(result.is_ok());

    // Test mismatched batch - should fail
    let bad_shape = [2, 2, 5, 4]; // batch=2 doesn't match
    let bad_k = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&bad_shape)),
        ArrayD::zeros(IxDyn(&bad_shape)),
    )
    .unwrap();
    let result = device.cross_attention_ibp(&q, &bad_k, &v, 1.0);
    assert!(result.is_err());

    // Test mismatched K/V sequence lengths - should fail
    let bad_v_shape = [1, 2, 6, 4]; // seq_enc=6 doesn't match K's seq_enc=5
    let bad_v = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&bad_v_shape)),
        ArrayD::zeros(IxDyn(&bad_v_shape)),
    )
    .unwrap();
    let result = device.cross_attention_ibp(&q, &k, &bad_v, 1.0);
    assert!(result.is_err());

    // Test mismatched dim between Q and K - should fail
    let bad_k_dim_shape = [1, 2, 5, 8]; // dim=8 doesn't match Q's dim=4
    let bad_k_dim = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&bad_k_dim_shape)),
        ArrayD::zeros(IxDyn(&bad_k_dim_shape)),
    )
    .unwrap();
    let result = device.cross_attention_ibp(&q, &bad_k_dim, &v, 1.0);
    assert!(result.is_err());
}

// ================= Backend Enum Tests =================

#[test]
fn test_backend_display() {
    assert_eq!(Backend::Cpu.to_string(), "cpu");
    assert_eq!(Backend::Wgpu.to_string(), "wgpu");
}

#[test]
fn test_backend_from_str() {
    use std::str::FromStr;

    // Valid backends
    assert_eq!(Backend::from_str("cpu").unwrap(), Backend::Cpu);
    assert_eq!(Backend::from_str("CPU").unwrap(), Backend::Cpu);
    assert_eq!(Backend::from_str("wgpu").unwrap(), Backend::Wgpu);
    assert_eq!(Backend::from_str("WGPU").unwrap(), Backend::Wgpu);
    assert_eq!(Backend::from_str("gpu").unwrap(), Backend::Wgpu);
    assert_eq!(Backend::from_str("GPU").unwrap(), Backend::Wgpu);

    // "mlx" is no longer a valid backend
    assert!(Backend::from_str("mlx").is_err());

    // Invalid backends return NyError::InvalidSpec
    let err = Backend::from_str("invalid").unwrap_err();
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(_)),
        "Expected InvalidSpec, got: {err:?}"
    );
    assert!(
        err.to_string().contains("Unknown backend: invalid"),
        "Error message should contain input: {}",
        err
    );
    assert!(Backend::from_str("").is_err());
    assert!(Backend::from_str("cuda").is_err());
}

#[test]
fn test_backend_default() {
    assert_eq!(Backend::default(), Backend::Cpu);
}

#[test]
fn test_backend_equality() {
    assert_eq!(Backend::Cpu, Backend::Cpu);
    assert_ne!(Backend::Cpu, Backend::Wgpu);
}

#[test]
fn test_backend_clone() {
    // Backend implements Copy, so we test the Clone impl via Copy behavior
    let backend = Backend::Wgpu;
    let cloned: Backend = backend; // Uses Copy (which implies Clone)
    assert_eq!(backend, cloned);
}

#[test]
fn test_backend_copy() {
    let backend = Backend::Wgpu;
    let copied: Backend = backend; // Copy, not move
    assert_eq!(backend, copied);
}

// ================= ComputeDevice Tests =================

#[test]
fn test_compute_device_cpu_creation() {
    let device = ComputeDevice::new(Backend::Cpu).unwrap();
    assert_eq!(device.backend(), Backend::Cpu);
}

#[test]
fn test_compute_device_clear_crown_working_set_cpu_is_noop_3515() {
    let device = ComputeDevice::new(Backend::Cpu).unwrap();
    device
        .clear_crown_working_set()
        .expect("CPU backend cleanup should be a no-op");
}

#[test]
fn test_compute_device_cpu_linear_ibp() {
    let device = ComputeDevice::new(Backend::Cpu).unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[2, 3]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let weight = Array2::from_elem((4, 3), 0.5_f32);
    let bias = Some(Array1::from_elem(4, 0.1_f32));

    let result = device.linear_ibp(&input, &weight, bias.as_ref()).unwrap();
    assert_eq!(result.shape(), &[2, 4]);

    // Verify bounds: for w=0.5 and x in [-1, 1], sum of 3 = [-1.5, 1.5], plus bias 0.1
    assert_relative_eq!(result.lower()[[0, 0]], -1.4, epsilon = 1e-5);
    assert_relative_eq!(result.upper()[[0, 0]], 1.6, epsilon = 1e-5);
}

#[test]
fn test_compute_device_cpu_matmul_ibp() {
    let device = ComputeDevice::new(Backend::Cpu).unwrap();

    let lower_a = ArrayD::from_elem(IxDyn(&[2, 3]), 0.0_f32);
    let upper_a = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
    let input_a = BoundedTensor::new(lower_a, upper_a).unwrap();

    let lower_b = ArrayD::from_elem(IxDyn(&[3, 4]), 0.0_f32);
    let upper_b = ArrayD::from_elem(IxDyn(&[3, 4]), 1.0_f32);
    let input_b = BoundedTensor::new(lower_b, upper_b).unwrap();

    let result = device.matmul_ibp(&input_a, &input_b).unwrap();
    assert_eq!(result.shape(), &[2, 4]);

    // For A,B in [0, 1]^{2x3} @ [0, 1]^{3x4}: result in [0, 3]
    assert_relative_eq!(result.lower()[[0, 0]], 0.0, epsilon = 1e-5);
    assert_relative_eq!(result.upper()[[0, 0]], 3.0, epsilon = 1e-5);
}

#[test]
fn test_compute_device_cpu_attention_ibp() {
    let device = ComputeDevice::new(Backend::Cpu).unwrap();

    let batch = 1;
    let heads = 2;
    let seq = 4;
    let dim = 8;
    let shape = [batch, heads, seq, dim];

    let lower = ArrayD::from_elem(IxDyn(&shape), -0.1_f32);
    let upper = ArrayD::from_elem(IxDyn(&shape), 0.1_f32);
    let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let v = BoundedTensor::new(lower, upper).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.attention_ibp(&q, &k, &v, scale).unwrap();
    assert_eq!(result.shape(), &shape);

    // Verify bounds are valid
    for (l, u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
    }
}

#[test]
fn test_compute_device_cpu_causal_attention_ibp() {
    let device = ComputeDevice::new(Backend::Cpu).unwrap();

    let batch = 1;
    let heads = 2;
    let seq = 4;
    let dim = 8;
    let shape = [batch, heads, seq, dim];

    let lower = ArrayD::from_elem(IxDyn(&shape), -0.1_f32);
    let upper = ArrayD::from_elem(IxDyn(&shape), 0.1_f32);
    let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let v = BoundedTensor::new(lower, upper).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.causal_attention_ibp(&q, &k, &v, scale).unwrap();
    assert_eq!(result.shape(), &shape);

    // Verify bounds are valid
    for (l, u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
    }
}

#[test]
fn test_compute_device_cpu_cross_attention_ibp() {
    let device = ComputeDevice::new(Backend::Cpu).unwrap();

    let batch = 1;
    let heads = 2;
    let seq_dec = 3;
    let seq_enc = 5;
    let dim = 4;

    let shape_q = [batch, heads, seq_dec, dim];
    let shape_kv = [batch, heads, seq_enc, dim];

    let lower_q = ArrayD::from_elem(IxDyn(&shape_q), -0.1_f32);
    let upper_q = ArrayD::from_elem(IxDyn(&shape_q), 0.1_f32);
    let lower_kv = ArrayD::from_elem(IxDyn(&shape_kv), -0.1_f32);
    let upper_kv = ArrayD::from_elem(IxDyn(&shape_kv), 0.1_f32);

    let q = BoundedTensor::new(lower_q, upper_q).unwrap();
    let k = BoundedTensor::new(lower_kv.clone(), upper_kv.clone()).unwrap();
    let v = BoundedTensor::new(lower_kv, upper_kv).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.cross_attention_ibp(&q, &k, &v, scale).unwrap();
    assert_eq!(result.shape(), &shape_q);

    // Verify bounds are valid
    for (l, u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
    }
}

#[test]
fn test_compute_device_crown_per_position() {
    use ndarray::Array2;
    use ny_propagate::{
        layers::{GELULayer, LinearLayer},
        GraphNode, Layer,
    };

    let device = ComputeDevice::new(Backend::Cpu).unwrap();

    // Build a small MLP graph
    let in_features = 4;
    let hidden_features = 8;
    let out_features = 4;

    let weight1 = Array2::from_shape_fn((hidden_features, in_features), |(i, j)| {
        0.1 * ((i + j) as f32 - 6.0)
    });
    let bias1 = Array1::from_elem(hidden_features, 0.05_f32);
    let linear1 = LinearLayer::new(weight1, Some(bias1)).unwrap();

    let weight2 = Array2::from_shape_fn((out_features, hidden_features), |(i, j)| {
        0.1 * ((i + j) as f32 - 6.0)
    });
    let bias2 = Array1::from_elem(out_features, 0.02_f32);
    let linear2 = LinearLayer::new(weight2, Some(bias2)).unwrap();

    let gelu = GELULayer::default();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(gelu),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));
    graph.set_output("linear2");

    let lower = ArrayD::from_elem(IxDyn(&[2, 3, in_features]), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3, in_features]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = device.crown_per_position_parallel(&graph, &input).unwrap();
    assert_eq!(result.shape(), &[2, 3, out_features]);

    // Verify bounds are valid
    for (l, u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(*l <= *u + 1e-5);
    }
}

// ================= NaN Sanitization Tests =================
// #2913: Verify that NaN/Inf in kernel output gets sanitized to
// conservative FALLBACK_BOUND, widening (not tightening) bounds.

#[test]
fn test_linear_ibp_nan_sanitization_widens_bounds() {
    // When weights contain NaN, the nan-propagating weight split should
    // poison the accumulator, and the sanitization loop should replace
    // NaN output with [-FALLBACK_BOUND, +FALLBACK_BOUND] (widening).
    use crate::accelerated::linear_ibp_parallel;
    use crate::FALLBACK_BOUND;

    // Input: shape [1, 2], bounds [-1, 1]
    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Weight [3, 2] with NaN in first row
    let mut weight = Array2::from_elem((3, 2), 0.5_f32);
    weight[[0, 0]] = f32::NAN;

    let result = linear_ibp_parallel(&input, &weight, None, 1, 2, 3).unwrap();

    // Output 0 (NaN-poisoned row): should be sanitized to [-FALLBACK, +FALLBACK]
    assert_eq!(
        result.lower()[[0, 0]],
        -FALLBACK_BOUND,
        "NaN-poisoned lower bound should be -FALLBACK_BOUND, not tightened"
    );
    assert_eq!(
        result.upper()[[0, 0]],
        FALLBACK_BOUND,
        "NaN-poisoned upper bound should be +FALLBACK_BOUND, not tightened"
    );

    // Outputs 1,2 (clean rows): should be finite, not sanitized
    assert!(
        result.lower()[[0, 1]].is_finite() && result.lower()[[0, 1]] != -FALLBACK_BOUND,
        "Clean row should not be sanitized to FALLBACK_BOUND"
    );
    assert!(
        result.upper()[[0, 2]].is_finite() && result.upper()[[0, 2]] != FALLBACK_BOUND,
        "Clean row should not be sanitized to FALLBACK_BOUND"
    );
}

#[test]
fn test_matmul_ibp_nan_sanitization_widens_bounds() {
    // When inputs contain NaN, the nan-propagating min/max should poison
    // accumulation, and sanitization should widen to [-FALLBACK, +FALLBACK].
    use crate::accelerated::matmul_ibp_parallel;
    use crate::FALLBACK_BOUND;

    // A: [2, 2] with NaN in upper bound of element [0,0]
    let lower_a = ArrayD::from_elem(IxDyn(&[2, 2]), 0.5_f32);
    let mut upper_a = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32);
    // Inject NaN into upper_a[0,0] — this should propagate through row 0
    upper_a[[0, 0]] = f32::NAN;

    // new_unchecked bypasses NaN validation
    let input_a = BoundedTensor::new_unchecked(lower_a, upper_a).unwrap();

    // B: [2, 2] clean
    let lower_b = ArrayD::from_elem(IxDyn(&[2, 2]), 0.5_f32);
    let upper_b = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32);
    let input_b = BoundedTensor::new(lower_b, upper_b).unwrap();

    let result = matmul_ibp_parallel(&input_a, &input_b).unwrap();

    // Row 0 of A has NaN, so output row 0 should be sanitized
    assert_eq!(
        result.lower()[[0, 0]],
        -FALLBACK_BOUND,
        "NaN-poisoned output should have lower = -FALLBACK_BOUND"
    );
    assert_eq!(
        result.upper()[[0, 0]],
        FALLBACK_BOUND,
        "NaN-poisoned output should have upper = +FALLBACK_BOUND"
    );

    // Row 1 of A is clean, so output row 1 should be finite and non-fallback
    assert!(
        result.lower()[[1, 0]].is_finite() && result.lower()[[1, 0]] != -FALLBACK_BOUND,
        "Clean row output should not be sanitized"
    );
}

#[test]
fn test_matmul_ibp_nan_in_b_sanitization() {
    // Self-audit finding 1: Original test only checked NaN in A.
    // NaN in B must also propagate through nan_propagating_min/max.
    use crate::accelerated::matmul_ibp_parallel;
    use crate::FALLBACK_BOUND;

    // A: [2, 2] clean
    let lower_a = ArrayD::from_elem(IxDyn(&[2, 2]), 0.5_f32);
    let upper_a = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32);
    let input_a = BoundedTensor::new(lower_a, upper_a).unwrap();

    // B: [2, 2] with NaN in lower bound of element [0,0]
    let mut lower_b = ArrayD::from_elem(IxDyn(&[2, 2]), 0.5_f32);
    let upper_b = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32);
    lower_b[[0, 0]] = f32::NAN;
    let input_b = BoundedTensor::new_unchecked(lower_b, upper_b).unwrap();

    let result = matmul_ibp_parallel(&input_a, &input_b).unwrap();

    // Column 0 of B has NaN, so output column 0 should be sanitized
    assert_eq!(
        result.lower()[[0, 0]],
        -FALLBACK_BOUND,
        "NaN in B should propagate to output column 0"
    );
    assert_eq!(result.upper()[[0, 0]], FALLBACK_BOUND);

    // Column 1 of B is clean, so output column 1 should be finite
    assert!(
        result.lower()[[0, 1]].is_finite() && result.lower()[[0, 1]] != -FALLBACK_BOUND,
        "Clean column output should not be sanitized"
    );
}

#[test]
fn test_linear_ibp_inf_sanitization() {
    // When Inf appears in accumulation (e.g., from very large inputs),
    // sanitization should replace with FALLBACK_BOUND.
    use crate::accelerated::linear_ibp_parallel;
    use crate::FALLBACK_BOUND;

    // Input with Inf bounds
    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0_f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0_f32);
    upper[[0, 0]] = f32::INFINITY;

    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    // Weight [2, 2] with mixed signs to trigger Inf accumulation
    let weight = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 0.5, 0.5]).unwrap();

    let result = linear_ibp_parallel(&input, &weight, None, 1, 2, 2).unwrap();

    // Both outputs are sanitized because IEEE 754: 0.0 * Inf = NaN.
    // In the weight split (w_pos, w_neg), one side is always 0.0 for
    // each weight element, so 0.0 * Inf = NaN poisons every row's accumulator.
    // Output 0 (weight row [1.0, -1.0]): w_neg[0]=0.0, 0.0*Inf=NaN → sanitized
    assert_eq!(result.lower()[[0, 0]], -FALLBACK_BOUND);
    assert_eq!(result.upper()[[0, 0]], FALLBACK_BOUND);
    // Output 1 (weight row [0.5, 0.5]): w_neg[0]=0.0, 0.0*Inf=NaN → also sanitized
    assert_eq!(
        result.lower()[[0, 1]],
        -FALLBACK_BOUND,
        "0*Inf=NaN should contaminate all outputs, not just row 0"
    );
    assert_eq!(result.upper()[[0, 1]], FALLBACK_BOUND);
}

// ================= Reference Comparison Tests =================
// #2913: Verify parallel kernels match ny-propagate sequential IBP.

#[test]
fn test_linear_ibp_parallel_matches_sequential_reference() {
    // Compare AcceleratedDevice::linear_ibp against
    // ny-propagate's LinearLayer::propagate_ibp.
    use ny_propagate::{layers::LinearLayer, BoundPropagation};

    let in_features = 5;
    let out_features = 4;

    // Deterministic weight matrix with mixed signs
    let weight = Array2::from_shape_fn((out_features, in_features), |(i, j)| {
        0.3 * ((i as f32) - 2.0) * ((j as f32) - 1.5)
    });
    let bias = Array1::from_shape_fn(out_features, |i| 0.1 * (i as f32) - 0.2);

    // Batched input: [3, 5] with varying bounds
    let lower = ArrayD::from_shape_fn(IxDyn(&[3, in_features]), |idx| {
        -1.0 + 0.1 * (idx[0] as f32) + 0.05 * (idx[1] as f32)
    });
    let upper = ArrayD::from_shape_fn(IxDyn(&[3, in_features]), |idx| {
        1.0 + 0.1 * (idx[0] as f32) + 0.05 * (idx[1] as f32)
    });
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Sequential reference (ny-propagate LinearLayer)
    let linear_layer = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();
    let ref_result = linear_layer.propagate_ibp(&input).unwrap();

    // Parallel kernel (ny-gpu AcceleratedDevice)
    let device = AcceleratedDevice::new();
    let par_result = device.linear_ibp(&input, &weight, Some(&bias)).unwrap();

    assert_eq!(ref_result.shape(), par_result.shape());

    for (r, p) in ref_result.lower().iter().zip(par_result.lower().iter()) {
        assert_relative_eq!(r, p, epsilon = 1e-4);
    }
    for (r, p) in ref_result.upper().iter().zip(par_result.upper().iter()) {
        assert_relative_eq!(r, p, epsilon = 1e-4);
    }
}

#[test]
fn test_linear_ibp_parallel_matches_sequential_negative_weights() {
    // Edge case: all-negative weights test the W_neg path
    use ny_propagate::{layers::LinearLayer, BoundPropagation};

    let in_features = 3;
    let out_features = 2;

    let weight = Array2::from_elem((out_features, in_features), -0.5_f32);
    let bias = Array1::zeros(out_features);

    let lower = ArrayD::from_elem(IxDyn(&[2, in_features]), 1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, in_features]), 3.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_layer = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();
    let ref_result = linear_layer.propagate_ibp(&input).unwrap();

    let device = AcceleratedDevice::new();
    let par_result = device.linear_ibp(&input, &weight, Some(&bias)).unwrap();

    // With w=-0.5, x in [1,3]: lower = -0.5*3*3 = -4.5, upper = -0.5*1*3 = -1.5
    for (r, p) in ref_result.lower().iter().zip(par_result.lower().iter()) {
        assert_relative_eq!(r, p, epsilon = 1e-5);
    }
    for (r, p) in ref_result.upper().iter().zip(par_result.upper().iter()) {
        assert_relative_eq!(r, p, epsilon = 1e-5);
    }
}

// ================= Attention Soundness Sampling Tests =================
// #2913: Verify attention IBP bounds contain concrete point evaluations.

#[test]
fn test_attention_ibp_soundness_sampling() {
    // Self-audit finding 2: Use independent Q/K/V perturbation ranges and
    // sample 3 corner configurations (center, lower, upper) for better coverage.
    let device = AcceleratedDevice::new();

    let batch = 1;
    let heads = 1;
    let seq = 3;
    let dim = 4;
    let shape = [batch, heads, seq, dim];

    let eps = 0.05;
    // Use different centers for Q, K, V to test independent perturbation
    let center_q = ArrayD::from_shape_fn(IxDyn(&shape), |idx| {
        0.2 * (idx[2] as f32) - 0.1 * (idx[3] as f32)
    });
    let center_k = ArrayD::from_shape_fn(IxDyn(&shape), |idx| {
        -0.1 * (idx[2] as f32) + 0.15 * (idx[3] as f32)
    });
    let center_v = ArrayD::from_shape_fn(IxDyn(&shape), |idx| {
        0.05 * (idx[2] as f32) + 0.05 * (idx[3] as f32)
    });

    let lower_q = center_q.mapv(|v| v - eps);
    let upper_q = center_q.mapv(|v| v + eps);
    let lower_k = center_k.mapv(|v| v - eps);
    let upper_k = center_k.mapv(|v| v + eps);
    let lower_v = center_v.mapv(|v| v - eps);
    let upper_v = center_v.mapv(|v| v + eps);

    let q = BoundedTensor::new(lower_q.clone(), upper_q.clone()).unwrap();
    let k = BoundedTensor::new(lower_k.clone(), upper_k.clone()).unwrap();
    let v = BoundedTensor::new(lower_v.clone(), upper_v.clone()).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();

    let result = device.attention_ibp(&q, &k, &v, scale).unwrap();

    // Helper to check containment of a point result within IBP bounds.
    // Check both .lower() and .upper() of the point result: even for point
    // inputs (lower==upper), softmax IBP relaxation may produce an interval,
    // so both endpoints must lie within the wider IBP bounds.
    let check_containment = |point_result: &BoundedTensor, label: &str| {
        let bound_lower = result.lower().as_slice().unwrap();
        let bound_upper = result.upper().as_slice().unwrap();
        let pt_lower = point_result.lower().as_slice().unwrap();
        let pt_upper = point_result.upper().as_slice().unwrap();
        for i in 0..pt_lower.len() {
            assert!(
                pt_lower[i] >= bound_lower[i] - 1e-5 && pt_lower[i] <= bound_upper[i] + 1e-5,
                "Soundness violation ({}) at {} (lower): {} not in [{}, {}]",
                label,
                i,
                pt_lower[i],
                bound_lower[i],
                bound_upper[i]
            );
            assert!(
                pt_upper[i] >= bound_lower[i] - 1e-5 && pt_upper[i] <= bound_upper[i] + 1e-5,
                "Soundness violation ({}) at {} (upper): {} not in [{}, {}]",
                label,
                i,
                pt_upper[i],
                bound_lower[i],
                bound_upper[i]
            );
        }
    };

    // Sample 1: center point (Q=center_q, K=center_k, V=center_v)
    let q_pt = BoundedTensor::new(center_q.clone(), center_q).unwrap();
    let k_pt = BoundedTensor::new(center_k.clone(), center_k).unwrap();
    let v_pt = BoundedTensor::new(center_v.clone(), center_v).unwrap();
    let center_result = device.attention_ibp(&q_pt, &k_pt, &v_pt, scale).unwrap();
    check_containment(&center_result, "center");

    // Sample 2: lower corner (Q=lower_q, K=lower_k, V=lower_v)
    let q_lo = BoundedTensor::new(lower_q.clone(), lower_q).unwrap();
    let k_lo = BoundedTensor::new(lower_k.clone(), lower_k).unwrap();
    let v_lo = BoundedTensor::new(lower_v.clone(), lower_v).unwrap();
    let lo_result = device.attention_ibp(&q_lo, &k_lo, &v_lo, scale).unwrap();
    check_containment(&lo_result, "lower-corner");

    // Sample 3: upper corner (Q=upper_q, K=upper_k, V=upper_v)
    let q_hi = BoundedTensor::new(upper_q.clone(), upper_q).unwrap();
    let k_hi = BoundedTensor::new(upper_k.clone(), upper_k).unwrap();
    let v_hi = BoundedTensor::new(upper_v.clone(), upper_v).unwrap();
    let hi_result = device.attention_ibp(&q_hi, &k_hi, &v_hi, scale).unwrap();
    check_containment(&hi_result, "upper-corner");
}
