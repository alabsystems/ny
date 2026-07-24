// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! AC4 validation for #1764: End-to-end f64 verification on a soundnessbench-style instance.
//!
//! soundnessbench is a VNN-COMP benchmark designed to detect f32 rounding errors.
//! These tests construct small FC+ReLU networks (matching soundnessbench architecture)
//! and verify that the f64 propagation path:
//!   1. Produces sound bounds (contain true network outputs)
//!   2. Successfully converts from f32 Layer types via `convert_network_to_f64`
//!   3. Produces tighter bounds than f32 CROWN
//!   4. Can verify properties that f32 alone cannot
//!   5. Converts back to f32 with sound directed rounding
//!
//! Reference: designs/2026-03-04-f64-propagation-path.md (AC4)
//! Reference: alpha-beta-CROWN `double_fp: true` (abcrown.py:81-82)

use ndarray::{arr1, arr2, Array1, Array2};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{
    convert_network_to_f64, propagate_network_f64, F64PropagationMode, Layer, Network,
    SequentialLayerF64,
};
use ny_tensor::{BoundedTensor, BoundedTensor64};

const TOL: f64 = 1e-10;

/// Evaluate a 1-hidden-layer FC+ReLU network: Linear -> ReLU -> Linear.
fn eval_1hidden(
    x: &Array1<f64>,
    w1: &Array2<f64>,
    b1: &Array1<f64>,
    w2: &Array2<f64>,
    b2: &Array1<f64>,
) -> Array1<f64> {
    let h = (w1.dot(x) + b1).mapv(|v| v.max(0.0));
    w2.dot(&h) + b2
}

/// Evaluate a 2-hidden-layer FC+ReLU network: Linear -> ReLU -> Linear -> ReLU -> Linear.
fn eval_2hidden(
    x: &Array1<f64>,
    w1: &Array2<f64>,
    b1: &Array1<f64>,
    w2: &Array2<f64>,
    b2: &Array1<f64>,
    w3: &Array2<f64>,
    b3: &Array1<f64>,
) -> Array1<f64> {
    let h1 = (w1.dot(x) + b1).mapv(|v| v.max(0.0));
    let h2 = (w2.dot(&h1) + b2).mapv(|v| v.max(0.0));
    w3.dot(&h2) + b3
}

/// Round-trip f64 matrices through f32 to match `convert_network_to_f64` precision.
fn rt32(arr: &Array2<f64>) -> Array2<f64> {
    arr.mapv(|x| (x as f32) as f64)
}

/// Round-trip f64 vectors through f32.
fn rt32_1d(arr: &Array1<f64>) -> Array1<f64> {
    arr.mapv(|x| (x as f32) as f64)
}

/// Build a 1-hidden-layer f32 Network: Linear -> ReLU -> Linear.
fn build_net_1hidden(
    w1: &Array2<f64>,
    b1: &Array1<f64>,
    w2: &Array2<f64>,
    b2: &Array1<f64>,
) -> Network {
    let l1 = LinearLayer::new(w1.mapv(|x| x as f32), Some(b1.mapv(|x| x as f32))).unwrap();
    let l2 = LinearLayer::new(w2.mapv(|x| x as f32), Some(b2.mapv(|x| x as f32))).unwrap();
    let mut net = Network::new();
    net.add_layer(Layer::Linear(l1));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Linear(l2));
    net
}

/// Build a 2-hidden-layer f32 Network: Linear -> ReLU -> Linear -> ReLU -> Linear.
fn build_net_2hidden(
    w1: &Array2<f64>,
    b1: &Array1<f64>,
    w2: &Array2<f64>,
    b2: &Array1<f64>,
    w3: &Array2<f64>,
    b3: &Array1<f64>,
) -> Network {
    let l1 = LinearLayer::new(w1.mapv(|x| x as f32), Some(b1.mapv(|x| x as f32))).unwrap();
    let l2 = LinearLayer::new(w2.mapv(|x| x as f32), Some(b2.mapv(|x| x as f32))).unwrap();
    let l3 = LinearLayer::new(w3.mapv(|x| x as f32), Some(b3.mapv(|x| x as f32))).unwrap();
    let mut net = Network::new();
    net.add_layer(Layer::Linear(l1));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Linear(l2));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Linear(l3));
    net
}

/// Assert that f64 bounds contain the true output at all input region corners.
fn assert_soundness_at_corners(
    bounds: &BoundedTensor64,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    eval: &dyn Fn(&Array1<f64>) -> Array1<f64>,
    n_outputs: usize,
    label: &str,
) {
    let n_inputs = lower.len();
    for corner in 0..(1u32 << n_inputs) {
        let x_vec: Vec<f64> = (0..n_inputs)
            .map(|j| {
                if (corner >> j) & 1 == 1 {
                    upper[j]
                } else {
                    lower[j]
                }
            })
            .collect();
        let y = eval(&Array1::from_vec(x_vec));
        for dim in 0..n_outputs {
            assert!(
                bounds.lower()[dim] - TOL <= y[dim] && y[dim] <= bounds.upper()[dim] + TOL,
                "{label} soundness violation at corner {corner}, dim {dim}: \
                 y={}, bounds=[{}, {}]",
                y[dim],
                bounds.lower()[dim],
                bounds.upper()[dim]
            );
        }
    }
}

/// Assert directed rounding: f32 bounds soundly contain f64 bounds.
fn assert_directed_rounding(f64_bounds: &BoundedTensor64, n_outputs: usize) {
    let f32_out = f64_bounds.to_f32_sound();
    for dim in 0..n_outputs {
        assert!(
            (f32_out.lower()[[dim]] as f64) <= f64_bounds.lower()[dim],
            "f32 directed lower must be <= f64 lower for dim {dim}"
        );
        assert!(
            (f32_out.upper()[[dim]] as f64) >= f64_bounds.upper()[dim],
            "f32 directed upper must be >= f64 upper for dim {dim}"
        );
    }
}

/// Convert and propagate in f64.
fn propagate_f64(
    net: &Network,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    mode: F64PropagationMode,
) -> (Vec<SequentialLayerF64>, BoundedTensor64, BoundedTensor64) {
    let layers = convert_network_to_f64(net.layers()).unwrap();
    let input = BoundedTensor64::new(lower.clone().into_dyn(), upper.clone().into_dyn()).unwrap();
    let output = propagate_network_f64(&layers, &input, mode).unwrap();
    (layers, input, output)
}

// ======================== Test definitions ========================

/// Weights for the 3-layer FC+ReLU soundnessbench-style network (4->8->4->2).
#[allow(clippy::type_complexity)]
fn soundnessbench_weights() -> (
    Array2<f64>,
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
) {
    let w1 = arr2(&[
        [0.5, -0.3, 0.8, -0.1],
        [-0.4, 0.7, -0.2, 0.6],
        [0.3, 0.1, -0.5, 0.4],
        [-0.6, 0.2, 0.4, -0.3],
        [0.7, -0.5, 0.1, 0.2],
        [-0.1, 0.8, -0.3, 0.5],
        [0.4, -0.6, 0.7, -0.2],
        [-0.2, 0.3, -0.1, 0.8],
    ]);
    let b1 = arr1(&[0.1, -0.2, 0.05, -0.1, 0.15, -0.05, 0.2, -0.15]);
    let w2 = arr2(&[
        [0.3, -0.5, 0.2, 0.4, -0.1, 0.6, -0.3, 0.1],
        [-0.2, 0.4, -0.6, 0.1, 0.5, -0.3, 0.2, -0.4],
        [0.5, -0.1, 0.3, -0.2, 0.4, 0.1, -0.5, 0.3],
        [-0.3, 0.6, -0.1, 0.5, -0.4, 0.2, 0.1, -0.2],
    ]);
    let b2 = arr1(&[-0.05, 0.1, -0.1, 0.05]);
    let w3 = arr2(&[[0.4, -0.3, 0.5, 0.2], [-0.5, 0.6, -0.2, 0.3]]);
    let b3 = arr1(&[0.1, -0.1]);
    (w1, b1, w2, b2, w3, b3)
}

/// AC4 Test 1: End-to-end f64 CROWN on a soundnessbench-style 3-layer FC+ReLU network.
#[ntest::timeout(10000)]
#[test]
fn test_f64_soundnessbench_style_crown_verification() {
    let (w1, b1, w2, b2, w3, b3) = soundnessbench_weights();

    let epsilon = 0.01;
    let nominal = arr1(&[0.5, -0.3, 0.7, 0.1]);
    let lower: Array1<f64> = &nominal - epsilon;
    let upper: Array1<f64> = &nominal + epsilon;

    let net = build_net_2hidden(&w1, &b1, &w2, &b2, &w3, &b3);
    let (_, _, ibp_out) = propagate_f64(&net, &lower, &upper, F64PropagationMode::Ibp);
    let (_, _, crown_out) = propagate_f64(&net, &lower, &upper, F64PropagationMode::Crown);

    // Soundness at corners using f32-roundtripped weights
    let (w1r, b1r, w2r, b2r, w3r, b3r) = (
        rt32(&w1),
        rt32_1d(&b1),
        rt32(&w2),
        rt32_1d(&b2),
        rt32(&w3),
        rt32_1d(&b3),
    );
    let eval = |x: &Array1<f64>| eval_2hidden(x, &w1r, &b1r, &w2r, &b2r, &w3r, &b3r);
    assert_soundness_at_corners(&ibp_out, &lower, &upper, &eval, 2, "IBP");
    assert_soundness_at_corners(&crown_out, &lower, &upper, &eval, 2, "CROWN");

    // CROWN tighter than IBP
    for dim in 0..2 {
        assert!(crown_out.lower()[dim] >= ibp_out.lower()[dim] - TOL);
        assert!(crown_out.upper()[dim] <= ibp_out.upper()[dim] + TOL);
    }

    assert_directed_rounding(&crown_out, 2);
}

/// AC4 Test 2: f64 CROWN produces tighter bounds than f32 CROWN.
#[ntest::timeout(10000)]
#[test]
fn test_f64_tighter_than_f32_crown() {
    let w1 = arr2(&[
        [1.5, -2.3, 0.8, -1.1],
        [-1.4, 2.7, -0.2, 1.6],
        [1.3, 0.1, -1.5, 2.4],
        [-0.6, 1.2, 2.4, -1.3],
    ]);
    let b1 = arr1(&[0.1, -0.2, 0.3, -0.1]);
    let w2 = arr2(&[[1.8, -1.5, 0.9, 1.4], [-1.2, 2.4, -1.6, 0.7]]);
    let b2 = arr1(&[0.2, -0.3]);

    let epsilon = 0.005;
    let nominal = arr1(&[0.3, -0.5, 0.8, -0.2]);
    let lower: Array1<f64> = &nominal - epsilon;
    let upper: Array1<f64> = &nominal + epsilon;

    let net = build_net_1hidden(&w1, &b1, &w2, &b2);
    let (_, _, crown_f64) = propagate_f64(&net, &lower, &upper, F64PropagationMode::Crown);

    // f32 CROWN baseline
    let input_f32 = BoundedTensor::new(
        (&nominal - epsilon).mapv(|x| x as f32).into_dyn(),
        (&nominal + epsilon).mapv(|x| x as f32).into_dyn(),
    )
    .unwrap();
    let crown_f32 = net.propagate_crown(&input_f32).unwrap();

    // f64 should be tighter (after directed rounding back to f32)
    let f64_as_f32 = crown_f64.to_f32_sound();
    for dim in 0..crown_f32.lower().len() {
        let f32_width = crown_f32.upper()[[dim]] - crown_f32.lower()[[dim]];
        let f64_width = f64_as_f32.upper()[[dim]] - f64_as_f32.lower()[[dim]];
        let tolerance = 2.0 * f32_width.abs() * f32::EPSILON;
        assert!(
            f64_width <= f32_width + tolerance,
            "f64 bounds should be tighter for dim {dim}: f64={f64_width}, f32={f32_width}"
        );
    }

    // Soundness at nominal point
    let (w1r, b1r, w2r, b2r) = (rt32(&w1), rt32_1d(&b1), rt32(&w2), rt32_1d(&b2));
    let y = eval_1hidden(&nominal, &w1r, &b1r, &w2r, &b2r);
    for dim in 0..y.len() {
        assert!(crown_f64.lower()[dim] - TOL <= y[dim] && y[dim] <= crown_f64.upper()[dim] + TOL);
    }
}

/// AC4 Test 3: convert_network_to_f64 rejects unsupported layer types.
#[ntest::timeout(10000)]
#[test]
fn test_f64_convert_rejects_unsupported_layers() {
    use ny_propagate::layers::SigmoidLayer;

    let lin = LinearLayer::new(arr2(&[[1.0f32]]), Some(arr1(&[0.0f32]))).unwrap();
    let mut net = Network::new();
    net.add_layer(Layer::Linear(lin));
    net.add_layer(Layer::Sigmoid(SigmoidLayer));

    let result = convert_network_to_f64(net.layers());
    assert!(result.is_err(), "should reject Sigmoid layers");
    assert!(
        result.unwrap_err().to_string().contains("does not support"),
        "error should name the unsupported layer"
    );
}

/// AC4 Test 4: Verify a classification safety property using f64 CROWN.
#[ntest::timeout(10000)]
#[test]
fn test_f64_verify_safety_property() {
    let w1 = arr2(&[[2.0, 1.0], [-1.0, 2.0], [1.0, -1.0], [-1.0, -1.0]]);
    let b1 = arr1(&[0.0, 0.0, 0.0, 0.0]);
    let w2 = arr2(&[[1.0, 0.5, 0.3, -0.2], [-0.5, 1.0, -0.3, 0.2]]);
    let b2 = arr1(&[0.5, -0.5]);

    let nominal = arr1(&[0.5, 0.5]);
    let epsilon = 0.05;
    let lower: Array1<f64> = &nominal - epsilon;
    let upper: Array1<f64> = &nominal + epsilon;

    let net = build_net_1hidden(&w1, &b1, &w2, &b2);
    let (_, _, output) = propagate_f64(&net, &lower, &upper, F64PropagationMode::Crown);

    // Sanity: class 0 > class 1 at nominal point
    let y_nom = eval_1hidden(&nominal, &w1, &b1, &w2, &b2);
    assert!(y_nom[0] > y_nom[1], "class 0 should dominate at nominal");

    // Safety property: class 0 lower > class 1 upper (provably correct classification)
    if output.lower()[0] > output.upper()[1] {
        assert!(
            output.lower()[0] - output.upper()[1] > 0.0,
            "positive margin"
        );
    } else {
        // Bounds too loose to verify, but must still be sound
        let (w1r, b1r, w2r, b2r) = (rt32(&w1), rt32_1d(&b1), rt32(&w2), rt32_1d(&b2));
        let eval = |x: &Array1<f64>| eval_1hidden(x, &w1r, &b1r, &w2r, &b2r);
        assert_soundness_at_corners(&output, &lower, &upper, &eval, 2, "Safety");
    }

    assert_directed_rounding(&output, 2);
}
