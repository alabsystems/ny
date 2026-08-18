// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use approx::assert_relative_eq;
use ndarray::arr2;
use ny_core::LayerType;
use ny_propagate::Layer as PropLayer;

#[ntest::timeout(10000)]
#[test]
fn test_convert_matmul_attributes_transpose_b_and_scale() {
    let model = OnnxModel {
        network: Network {
            name: "test".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "matmul".to_string(),
        layer_type: LayerType::MatMul,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::from([
            ("transpose_b".to_string(), AttributeValue::Int(1)),
            ("scale".to_string(), AttributeValue::Float(0.25)),
        ]),
    };

    let layer = model.convert_layer(&spec).unwrap();
    // With transpose_b=true and both inputs being activations (not weights),
    // MatMul is converted to BilinearCrown for attention-style Q@K^T operations
    match layer {
        PropLayer::BilinearCrown(b) => {
            assert!(b.transpose_b());
            assert_eq!(b.scale(), Some(0.25));
        }
        other => panic!(
            "Expected BilinearCrown layer (for transpose_b=true), got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_matmul_weight_transpose_b_true_and_scale_applied() {
    let mut weights = WeightStore::new();
    weights.insert(
        "w".to_string(),
        arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(),
    );

    let model = OnnxModel {
        network: Network {
            name: "test".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights,
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "matmul".to_string(),
        layer_type: LayerType::MatMul,
        inputs: vec!["a".to_string(), "w".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::from([
            ("transpose_b".to_string(), AttributeValue::Int(1)),
            ("scale".to_string(), AttributeValue::Float(0.5)),
        ]),
    };

    let layer = model.convert_layer(&spec).unwrap();
    match layer {
        PropLayer::Linear(l) => {
            assert_eq!(l.weight().shape(), &[2, 3]);
            assert_relative_eq!(l.weight()[[0, 0]], 0.5, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[0, 1]], 1.0, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[0, 2]], 1.5, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[1, 0]], 2.0, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[1, 1]], 2.5, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[1, 2]], 3.0, epsilon = 1e-6);
        }
        other => panic!(
            "Expected Linear layer (constant weight), got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_matmul_weight_transpose_b_false_transposes_weight() {
    let mut weights = WeightStore::new();
    // W has shape (K, N) for A @ W, and should be transposed to (N, K) for Linear.
    weights.insert(
        "w".to_string(),
        arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn(),
    );

    let model = OnnxModel {
        network: Network {
            name: "test".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights,
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "matmul".to_string(),
        layer_type: LayerType::MatMul,
        inputs: vec!["a".to_string(), "w".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    };

    let layer = model.convert_layer(&spec).unwrap();
    match layer {
        PropLayer::Linear(l) => {
            assert_eq!(l.weight().shape(), &[2, 3]);
            assert_relative_eq!(l.weight()[[0, 0]], 1.0, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[0, 1]], 3.0, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[0, 2]], 5.0, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[1, 0]], 2.0, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[1, 1]], 4.0, epsilon = 1e-6);
            assert_relative_eq!(l.weight()[[1, 2]], 6.0, epsilon = 1e-6);
        }
        other => panic!(
            "Expected Linear layer (constant weight), got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_matmul_transpose_b_const_weight() {
    let path = require_test_model("matmul_transpose_b_const.onnx");

    // Standard ONNX represents B^T with an explicit Transpose node; constant
    // folding should preserve that exact matrix before MatMul becomes Linear.
    let model = load_onnx(&path).expect("Failed to load explicit-transpose MatMul model");
    let prop = model
        .to_propagate_network()
        .expect("Failed to convert MatMul transpose_b model");

    assert_eq!(prop.layers().len(), 1);
    match &prop.layers()[0] {
        PropLayer::Linear(layer) => {
            assert_eq!(layer.weight().shape(), &[2, 3]);
            assert_relative_eq!(layer.weight()[[0, 0]], 1.0, epsilon = 1e-6);
            assert_relative_eq!(layer.weight()[[0, 1]], 2.0, epsilon = 1e-6);
            assert_relative_eq!(layer.weight()[[0, 2]], 3.0, epsilon = 1e-6);
            assert_relative_eq!(layer.weight()[[1, 0]], 4.0, epsilon = 1e-6);
            assert_relative_eq!(layer.weight()[[1, 1]], 5.0, epsilon = 1e-6);
            assert_relative_eq!(layer.weight()[[1, 2]], 6.0, epsilon = 1e-6);
        }
        other => panic!(
            "Expected Linear layer from explicit-transpose MatMul const weight, got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_attention_matmul_detection() {
    // Test that the ONNX loader correctly identifies bounded MatMul vs Linear
    let path = require_test_model_with_hint("simple_attention.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load model");

    // The LayerSpec shows ONNX types, but conversion to propagate network
    // should correctly identify Linear vs bounded MatMul
    let network = model.to_propagate_network().expect("Failed to convert");

    // Count actual layer types in the propagate network
    let mut linear_count = 0;
    let mut bilinear_count = 0;
    let mut add_count = 0;
    let mut softmax_count = 0;

    for layer in network.layers() {
        match layer {
            PropLayer::Linear(_) => linear_count += 1,
            // Since c93afde62, all activation-activation MatMuls produce BilinearCrown.
            PropLayer::BilinearCrown(_) => bilinear_count += 1,
            PropLayer::Add(_) => add_count += 1,
            PropLayer::Softmax(_) => softmax_count += 1,
            _ => {}
        }
    }

    println!("Propagate network layers: {} total", network.layers().len());
    println!("  Linear: {}", linear_count);
    println!("  BilinearCrown (bounded MatMul): {}", bilinear_count);
    println!("  Add: {}", add_count);
    println!("  Softmax: {}", softmax_count);

    // Expected structure for attention:
    // - 4 Linear layers (Q, K, V, output projections) - each MatMul+Add converts to Linear
    // - 2 BilinearCrown (Q@K^T and attn@V) — since c93afde62
    // - 1 Softmax
    // - Some Add layers for biases

    // We expect at least 4 Linear layers from the projections
    assert!(
        linear_count >= 4,
        "Expected at least 4 Linear layers (Q/K/V/out projections), got {}",
        linear_count
    );

    // Both Q@K^T and attn@V produce BilinearCrown (since c93afde62).
    assert!(
        bilinear_count >= 2,
        "Expected at least 2 BilinearCrown (Q@K^T and attn@V), got {}",
        bilinear_count
    );

    // We expect exactly 1 Softmax
    assert_eq!(softmax_count, 1, "Expected exactly 1 Softmax layer");
}
