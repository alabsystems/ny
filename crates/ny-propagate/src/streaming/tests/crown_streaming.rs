// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::streaming::*;

fn nan_propagating_max(acc: f32, next: f32) -> f32 {
    if acc.is_nan() || next.is_nan() {
        f32::NAN
    } else {
        acc.max(next)
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_memory_savings_calculation() {
    let (original, streaming, savings) = estimate_memory_savings(100, 1000, 10);

    // 100 layers, checkpoint every 10 = 11 checkpoints (including input)
    // Savings should be ~89%
    assert!(savings > 85.0);
    assert!(savings < 95.0);
    assert!(streaming < original);
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_crown_empty_network() {
    let network = Network::new();
    let input = create_input(10);

    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let result = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    assert_eq!(result.shape(), input.shape());
    assert_eq!(result.lower(), input.lower());
    assert_eq!(result.upper(), input.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_crown_single_layer() {
    let network = create_test_network(1, 10, 10);
    let input = create_input(10);

    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let streaming_result = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    // Compare with regular CROWN
    let regular_result = network.propagate_crown(&input).unwrap();

    assert_eq!(streaming_result.shape(), regular_result.shape());

    // Bound values must match within floating point tolerance
    for (i, (sl, rl)) in streaming_result
        .lower()
        .iter()
        .zip(regular_result.lower().iter())
        .enumerate()
    {
        assert!(
            (sl - rl).abs() < 1e-5,
            "Lower bound mismatch at {i}: streaming={sl}, regular={rl}"
        );
    }
    for (i, (su, ru)) in streaming_result
        .upper()
        .iter()
        .zip(regular_result.upper().iter())
        .enumerate()
    {
        assert!(
            (su - ru).abs() < 1e-5,
            "Upper bound mismatch at {i}: streaming={su}, regular={ru}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_crown_two_layers() {
    let network = create_test_network(2, 10, 10);
    let input = create_input(10);

    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let streaming_result = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    let regular_result = network.propagate_crown(&input).unwrap();

    assert_eq!(streaming_result.shape(), regular_result.shape());

    // Bound values must match within floating point tolerance
    for (i, (sl, rl)) in streaming_result
        .lower()
        .iter()
        .zip(regular_result.lower().iter())
        .enumerate()
    {
        assert!(
            (sl - rl).abs() < 1e-5,
            "Lower bound mismatch at {i}: streaming={sl}, regular={rl}"
        );
    }
    for (i, (su, ru)) in streaming_result
        .upper()
        .iter()
        .zip(regular_result.upper().iter())
        .enumerate()
    {
        assert!(
            (su - ru).abs() < 1e-5,
            "Upper bound mismatch at {i}: streaming={su}, regular={ru}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_crown_equivalence() {
    // Create network with non-trivial weights
    let mut network = Network::new();
    for i in 0..10 {
        // Use small random-ish weights to avoid bound explosion
        let mut weight = Array2::<f32>::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                // Deterministic "random" pattern
                let val = ((r * 7 + c * 11 + i * 13) % 10) as f32 * 0.01 - 0.05;
                weight[[r, c]] = val;
            }
        }
        let bias = Some(Array1::<f32>::zeros(8));
        let linear = LinearLayer::new(weight, bias).unwrap();
        network.add_layer(Layer::Linear(linear));
    }

    let lower = ArrayD::from_elem(ndarray::IxDyn(&[8]), -0.1_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[8]), 0.1_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Regular CROWN
    let regular_result = network.propagate_crown(&input).unwrap();

    // Streaming CROWN with different checkpoint intervals
    for interval in [1, 2, 5, 10] {
        let config = StreamingConfig {
            checkpoint_interval: interval,
            ..Default::default()
        };
        let verifier = StreamingVerifier::new(config);
        let streaming_result = verifier
            .propagate_crown_streaming(&network, &input)
            .unwrap();

        // Results should match within floating point tolerance
        assert_eq!(streaming_result.shape(), regular_result.shape());

        let max_lower_diff: f32 = streaming_result
            .lower()
            .iter()
            .zip(regular_result.lower().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, nan_propagating_max);

        let max_upper_diff: f32 = streaming_result
            .upper()
            .iter()
            .zip(regular_result.upper().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, nan_propagating_max);

        assert!(
            max_lower_diff < 1e-5,
            "Lower bounds differ by {} with interval {}",
            max_lower_diff,
            interval
        );
        assert!(
            max_upper_diff < 1e-5,
            "Upper bounds differ by {} with interval {}",
            max_upper_diff,
            interval
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_recomputation_correctness() {
    // Test that recomputing from checkpoints gives same result as direct computation
    let network = create_test_network(10, 8, 8);
    let input = create_input(8);

    // Collect all bounds (regular way)
    let all_bounds = network.collect_ibp_bounds(&input).unwrap();

    // Collect checkpointed bounds
    let config = StreamingConfig {
        checkpoint_interval: 3,
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);
    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Verify that get_bounds_at returns same bounds as direct computation
    for (i, expected) in all_bounds.iter().take(10).enumerate() {
        let recomputed = checkpointed.bounds_at(i, &network).unwrap();

        let max_lower_diff: f32 = recomputed
            .lower()
            .iter()
            .zip(expected.lower().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, nan_propagating_max);

        let max_upper_diff: f32 = recomputed
            .upper()
            .iter()
            .zip(expected.upper().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, nan_propagating_max);

        assert!(
            max_lower_diff < 1e-6,
            "Layer {} lower bounds differ by {}",
            i,
            max_lower_diff
        );
        assert!(
            max_upper_diff < 1e-6,
            "Layer {} upper bounds differ by {}",
            i,
            max_upper_diff
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_estimate_memory_savings_zero_layers() {
    let (original, streaming, savings) = estimate_memory_savings(0, 1000, 10);
    assert_eq!(original, 0);
    // With 0 layers, still store exactly the input tensor.
    assert_eq!(streaming, 1000 * 4 * 2);
    assert_eq!(savings, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_estimate_memory_savings_single_layer() {
    let (original, streaming, savings) = estimate_memory_savings(1, 1000, 10);
    // 1 layer, checkpoint every 10 = 1 checkpoint + input = 2
    // original = 1 * 8000 = 8000
    // streaming = 2 * 8000 = 16000
    // This actually increases memory for very small networks
    assert_eq!(original, 8000);
    assert!(streaming > 0, "streaming memory must be positive");
    // Savings can be negative when streaming uses more memory than original
    assert!(savings.is_finite(), "savings must be finite, got {savings}");
}

#[ntest::timeout(10000)]
#[test]
fn test_estimate_memory_savings_large_interval() {
    let (original, streaming, savings) = estimate_memory_savings(100, 1000, 100);
    // 100 layers, checkpoint every 100 = 1 checkpoint + input = 2
    // Savings should be very high (~98%)
    assert!(
        original > 0,
        "100 layers must have positive original memory"
    );
    assert!(
        streaming < original,
        "streaming ({streaming}) must be less than original ({original})"
    );
    assert!(savings > 95.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_estimate_memory_savings_zero_interval() {
    let (original, streaming, savings) = estimate_memory_savings(10, 100, 0);
    // Interval should clamp to 1, so streaming memory should be > 0.
    assert!(original > 0, "10 layers must have positive original memory");
    assert!(streaming > 0);
    assert!(savings.is_finite(), "savings must be finite, got {savings}");
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_crown_with_batchnorm() {
    use crate::layers::BatchNormLayer;

    // Create a simple network: Linear -> BatchNorm -> Linear
    // Input: 8 features, BatchNorm with 8 channels
    let weight1 = Array2::<f32>::eye(8);
    let bias1 = Some(Array1::<f32>::zeros(8));
    let linear1 = LinearLayer::new(weight1, bias1).unwrap();

    // BatchNorm with 8 channels (non-trivial parameters to exercise the math)
    let mut ny = ArrayD::from_elem(ndarray::IxDyn(&[8]), 1.0f32);
    let mut beta = ArrayD::from_elem(ndarray::IxDyn(&[8]), 0.0f32);
    let mut mean = ArrayD::from_elem(ndarray::IxDyn(&[8]), 0.0f32);
    let mut var = ArrayD::from_elem(ndarray::IxDyn(&[8]), 1.0f32);
    // Use non-trivial parameters: varying scale, bias, mean, variance
    for i in 0..8 {
        ny[[i]] = 0.5 + (i as f32) * 0.1; // scale: 0.5, 0.6, ..., 1.2
        beta[[i]] = -0.1 + (i as f32) * 0.025; // bias: -0.1, -0.075, ..., 0.075
        mean[[i]] = 0.1 * (i as f32 - 4.0); // mean: -0.4, -0.3, ..., 0.3
        var[[i]] = 0.5 + (i as f32) * 0.1; // variance: 0.5, 0.6, ..., 1.2
    }
    let bn = BatchNormLayer::new(&ny, &beta, &mean, &var, 1e-5).unwrap();

    let weight2 = Array2::<f32>::eye(8);
    let bias2 = Some(Array1::<f32>::zeros(8));
    let linear2 = LinearLayer::new(weight2, bias2).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::BatchNorm(bn));
    network.add_layer(Layer::Linear(linear2));

    let input = create_input(8);

    // Streaming CROWN should now work with BatchNorm
    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let streaming_result = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    // Compare with regular CROWN
    let regular_result = network.propagate_crown(&input).unwrap();

    // Results should match (shape and approximately same bounds)
    assert_eq!(streaming_result.shape(), regular_result.shape());

    // Check bounds are close (should match within floating point tolerance)
    let stream_lower: Vec<f32> = streaming_result.lower().iter().cloned().collect();
    let stream_upper: Vec<f32> = streaming_result.upper().iter().cloned().collect();
    let reg_lower: Vec<f32> = regular_result.lower().iter().cloned().collect();
    let reg_upper: Vec<f32> = regular_result.upper().iter().cloned().collect();

    for i in 0..stream_lower.len() {
        assert!(
            (stream_lower[i] - reg_lower[i]).abs() < 1e-5,
            "Lower bound mismatch at {}: {} vs {}",
            i,
            stream_lower[i],
            reg_lower[i]
        );
        assert!(
            (stream_upper[i] - reg_upper[i]).abs() < 1e-5,
            "Upper bound mismatch at {}: {} vs {}",
            i,
            stream_upper[i],
            reg_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_crown_batchnorm_negative_scale() {
    use crate::layers::BatchNormLayer;

    // Test BatchNorm with negative scale
    // NOTE: Current impl swaps bounds for negative scale - potentially unsound (#306)
    let weight = Array2::<f32>::eye(4);
    let bias = Some(Array1::<f32>::zeros(4));
    let linear = LinearLayer::new(weight, bias).unwrap();

    // BatchNorm with negative scale on channel 0: scale=-1.0
    let mut ny = ArrayD::from_elem(ndarray::IxDyn(&[4]), 1.0f32);
    ny[[0]] = -1.0; // Negative scale
    let beta = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.0f32);
    let mean = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.0f32);
    let var = ArrayD::from_elem(ndarray::IxDyn(&[4]), 1.0f32);
    let bn = BatchNormLayer::new(&ny, &beta, &mean, &var, 1e-5).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::BatchNorm(bn));

    let input = create_input(4);

    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let streaming_result = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    let regular_result = network.propagate_crown(&input).unwrap();

    assert_eq!(streaming_result.shape(), regular_result.shape());

    // Verify negative scale handling: channel 0 should have swapped bounds
    // With input [-1, 1] and scale=-1, output should be [-1, 1] (swapped and negated)
    let stream_lower: Vec<f32> = streaming_result.lower().iter().cloned().collect();
    let stream_upper: Vec<f32> = streaming_result.upper().iter().cloned().collect();
    let reg_lower: Vec<f32> = regular_result.lower().iter().cloned().collect();
    let reg_upper: Vec<f32> = regular_result.upper().iter().cloned().collect();

    for i in 0..stream_lower.len() {
        assert!(
            (stream_lower[i] - reg_lower[i]).abs() < 1e-5,
            "Lower bound mismatch at {}: {} vs {}",
            i,
            stream_lower[i],
            reg_lower[i]
        );
        assert!(
            (stream_upper[i] - reg_upper[i]).abs() < 1e-5,
            "Upper bound mismatch at {}: {} vs {}",
            i,
            stream_upper[i],
            reg_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_crown_batchnorm_with_relu() {
    use crate::layers::{BatchNormLayer, ReLULayer};

    // Test BatchNorm followed by ReLU activation (common pattern in CNNs)
    let weight1 = Array2::<f32>::eye(4);
    let bias1 = Some(Array1::<f32>::zeros(4));
    let linear1 = LinearLayer::new(weight1, bias1).unwrap();

    // BatchNorm with non-trivial parameters
    let mut ny = ArrayD::from_elem(ndarray::IxDyn(&[4]), 1.0f32);
    let beta = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.0f32);
    let mean = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.0f32);
    let var = ArrayD::from_elem(ndarray::IxDyn(&[4]), 1.0f32);
    ny[[0]] = 2.0; // Different scales
    ny[[1]] = 0.5;
    let bn = BatchNormLayer::new(&ny, &beta, &mean, &var, 1e-5).unwrap();

    let relu = ReLULayer;

    let weight2 = Array2::<f32>::eye(4);
    let bias2 = Some(Array1::<f32>::zeros(4));
    let linear2 = LinearLayer::new(weight2, bias2).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::BatchNorm(bn));
    network.add_layer(Layer::ReLU(relu));
    network.add_layer(Layer::Linear(linear2));

    let input = create_input(4);

    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let streaming_result = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    let regular_result = network.propagate_crown(&input).unwrap();

    assert_eq!(streaming_result.shape(), regular_result.shape());

    let stream_lower: Vec<f32> = streaming_result.lower().iter().cloned().collect();
    let stream_upper: Vec<f32> = streaming_result.upper().iter().cloned().collect();
    let reg_lower: Vec<f32> = regular_result.lower().iter().cloned().collect();
    let reg_upper: Vec<f32> = regular_result.upper().iter().cloned().collect();

    for i in 0..stream_lower.len() {
        assert!(
            (stream_lower[i] - reg_lower[i]).abs() < 1e-5,
            "Lower bound mismatch at {}: {} vs {}",
            i,
            stream_lower[i],
            reg_lower[i]
        );
        assert!(
            (stream_upper[i] - reg_upper[i]).abs() < 1e-5,
            "Upper bound mismatch at {}: {} vs {}",
            i,
            stream_upper[i],
            reg_upper[i]
        );
    }
}

/// Regression test for #2280: streaming CROWN sound path must produce
/// finite, non-inverted bounds. Exercises the NaN/inversion guard.
#[ntest::timeout(10000)]
#[test]
fn streaming_crown_sound_produces_valid_bounds_2280() {
    use crate::streaming::StreamingVerifier;

    let network = create_test_network(3, 5, 5);
    let input = create_input(5);

    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let result = verifier
        .propagate_crown_streaming(&network, &input)
        .expect("invariant: sound streaming CROWN on linear network");

    assert_eq!(result.shape(), input.shape());
    for (&l, &u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(l.is_finite(), "lower bound must be finite, got {l}");
        assert!(u.is_finite(), "upper bound must be finite, got {u}");
        assert!(l <= u, "inverted interval: lower {l} > upper {u}");
    }
}

/// Verify centralized invalid-bound repair via new_repaired(Widen) (#2287, #3423).
#[test]
fn streaming_invalid_bounds_repair_is_centralized() {
    use ny_tensor::RepairStrategy;

    // Valid bounds pass through unchanged.
    let valid = BoundedTensor::new_repaired(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.0, 0.0])
            .expect("invariant: valid shape"),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 2.0])
            .expect("invariant: valid shape"),
        RepairStrategy::Widen,
    )
    .expect("new_repaired should succeed on valid bounds");
    assert_eq!(valid.lower()[[0]], -1.0);
    assert_eq!(valid.upper()[[0]], 1.0);

    // Invalid bounds: Widen replaces NaN with ±inf, keeps Inf as-is, swaps inversions.
    let invalid = BoundedTensor::new_repaired(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[5]), vec![f32::NAN, 0.0, 5.0, 0.0, -1.0])
            .expect("invariant: valid shape"),
        ArrayD::from_shape_vec(
            ndarray::IxDyn(&[5]),
            vec![1.0, f32::INFINITY, 3.0, f32::NAN, 2.0],
        )
        .expect("invariant: valid shape"),
        RepairStrategy::Widen,
    )
    .expect("new_repaired(Widen) should succeed on invalid bounds");

    // Element 0: lower=NaN→-inf, upper=1.0 (finite preserved) → [-inf, 1.0].
    assert_eq!(invalid.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(invalid.upper()[[0]], 1.0);
    // Element 1: lower=0.0 (finite), upper=Inf (kept as-is in Widen) → [0.0, +inf].
    assert_eq!(invalid.lower()[[1]], 0.0);
    assert_eq!(invalid.upper()[[1]], f32::INFINITY);
    // Element 2: lower=5.0 > upper=3.0 → swapped to [3.0, 5.0].
    assert_eq!(invalid.lower()[[2]], 3.0);
    assert_eq!(invalid.upper()[[2]], 5.0);
    // Element 3: lower=0.0 (finite), upper=NaN→+inf → [0.0, +inf].
    assert_eq!(invalid.lower()[[3]], 0.0);
    assert_eq!(invalid.upper()[[3]], f32::INFINITY);
    // Valid element remains unchanged.
    assert_eq!(invalid.lower()[[4]], -1.0);
    assert_eq!(invalid.upper()[[4]], 2.0);
}

/// Soundness test: streaming CROWN bounds must contain all concrete network outputs.
///
/// Constructs a network with non-trivial weights and ReLU, runs streaming CROWN,
/// then evaluates the network at a grid of concrete points within the input domain
/// and asserts all outputs lie within the streaming CROWN bounds.
///
/// This is the only test in this file that checks concrete-output-inside-bounds
/// (as opposed to equivalence with regular CROWN or shape-only checks).
/// See #1928 for the broader test-quality gap this addresses.
fn assert_streaming_crown_contains_concrete_outputs(
    network: &Network,
    input: &BoundedTensor,
    samples_per_dim: usize,
) {
    assert_streaming_crown_contains_concrete_outputs_with_config(
        network,
        input,
        samples_per_dim,
        StreamingConfig::default(),
    );
}

fn assert_streaming_crown_contains_concrete_outputs_with_config(
    network: &Network,
    input: &BoundedTensor,
    samples_per_dim: usize,
    config: StreamingConfig,
) {
    assert_eq!(
        input.shape(),
        &[2],
        "test helper currently supports 2D input domains"
    );
    let sampled_points = (samples_per_dim + 1) * (samples_per_dim + 1);
    assert!(
        sampled_points >= 100,
        "concrete soundness tests must sample at least 100 points"
    );

    let verifier = StreamingVerifier::new(config);
    let crown_bounds = verifier
        .propagate_crown_streaming(network, input)
        .expect("streaming CROWN sound propagation should succeed");

    let input_lower = input.lower();
    let input_upper = input.upper();
    for i in 0..=samples_per_dim {
        for j in 0..=samples_per_dim {
            let x0 = input_lower[[0]]
                + (input_upper[[0]] - input_lower[[0]]) * (i as f32) / (samples_per_dim as f32);
            let x1 = input_lower[[1]]
                + (input_upper[[1]] - input_lower[[1]]) * (j as f32) / (samples_per_dim as f32);
            let point = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![x0, x1])
                .expect("invariant: sample point shape");
            let concrete_input = BoundedTensor::concrete(point).unwrap();
            let concrete_output = network
                .propagate_ibp(&concrete_input)
                .expect("invariant: concrete propagation should succeed");
            let output = concrete_output.lower();

            let tol = 1e-5;
            for k in 0..output.len() {
                assert!(
                    crown_bounds.lower()[[k]] <= output[[k]] + tol,
                    "Streaming CROWN lower[{k}] = {} > concrete output[{k}] = {} at ({x0}, {x1})",
                    crown_bounds.lower()[[k]],
                    output[[k]]
                );
                assert!(
                    crown_bounds.upper()[[k]] >= output[[k]] - tol,
                    "Streaming CROWN upper[{k}] = {} < concrete output[{k}] = {} at ({x0}, {x1})",
                    crown_bounds.upper()[[k]],
                    output[[k]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn streaming_crown_concrete_soundness() {
    use crate::layers::ReLULayer;

    // Build a 2-layer Linear->ReLU->Linear network with non-trivial weights.
    // Keep dimensions small (2->2) so the grid covers the domain densely.
    let w1 = Array2::from_shape_vec((2, 2), vec![0.5, -0.3, 0.2, 0.8]).unwrap();
    let b1 = Some(Array1::from_vec(vec![0.1, -0.2]));
    let linear1 = LinearLayer::new(w1, b1).unwrap();

    let w2 = Array2::from_shape_vec((2, 2), vec![-0.4, 0.6, 0.3, -0.5]).unwrap();
    let b2 = Some(Array1::from_vec(vec![0.05, -0.1]));
    let linear2 = LinearLayer::new(w2, b2).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.0, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // 11x11 grid = 121 concrete points.
    assert_streaming_crown_contains_concrete_outputs(&network, &input, 10);
}

#[ntest::timeout(10000)]
#[test]
fn streaming_crown_concrete_soundness_with_batchnorm() {
    use crate::layers::{BatchNormLayer, ReLULayer};

    let w1 = Array2::from_shape_vec((2, 2), vec![0.6, -0.25, 0.15, 0.8]).unwrap();
    let b1 = Some(Array1::from_vec(vec![0.1, -0.05]));
    let linear1 = LinearLayer::new(w1, b1).unwrap();

    let ny = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.2, 0.7]).unwrap();
    let beta = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.05, -0.08]).unwrap();
    let mean = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.2, -0.1]).unwrap();
    let var = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.9, 1.4]).unwrap();
    let bn = BatchNormLayer::new(&ny, &beta, &mean, &var, 1e-5).unwrap();

    let w2 = Array2::from_shape_vec((2, 2), vec![-0.45, 0.3, 0.35, -0.55]).unwrap();
    let b2 = Some(Array1::from_vec(vec![-0.02, 0.04]));
    let linear2 = LinearLayer::new(w2, b2).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::BatchNorm(bn));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.2, -0.8]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.9, 1.1]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // 11x11 grid = 121 concrete points.
    assert_streaming_crown_contains_concrete_outputs(&network, &input, 10);
}

#[ntest::timeout(10000)]
#[test]
fn streaming_crown_concrete_soundness_deep_network() {
    use crate::layers::ReLULayer;

    let l1 = LinearLayer::new(
        Array2::from_shape_vec((3, 2), vec![0.35, -0.2, -0.12, 0.4, 0.15, 0.27]).unwrap(),
        Some(Array1::from_vec(vec![0.02, -0.03, 0.05])),
    )
    .unwrap();
    let l2 = LinearLayer::new(
        Array2::from_shape_vec(
            (3, 3),
            vec![0.28, -0.1, 0.06, -0.2, 0.33, 0.12, 0.09, 0.14, -0.24],
        )
        .unwrap(),
        Some(Array1::from_vec(vec![0.01, 0.0, -0.02])),
    )
    .unwrap();
    let l3 = LinearLayer::new(
        Array2::from_shape_vec((2, 3), vec![0.31, -0.08, 0.22, -0.16, 0.27, 0.19]).unwrap(),
        Some(Array1::from_vec(vec![-0.04, 0.03])),
    )
    .unwrap();
    let l4 = LinearLayer::new(
        Array2::from_shape_vec((2, 2), vec![0.4, -0.26, 0.18, 0.29]).unwrap(),
        Some(Array1::from_vec(vec![0.0, -0.01])),
    )
    .unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(l1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(l2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(l3));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(l4));

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.0, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // 11x11 grid = 121 concrete points.
    assert_streaming_crown_contains_concrete_outputs(&network, &input, 10);
}

#[ntest::timeout(10000)]
#[test]
fn streaming_crown_concrete_soundness_recomputes_between_checkpoints() {
    use crate::layers::ReLULayer;

    let layers = [
        (vec![0.42, -0.18, 0.16, 0.37], vec![0.03, -0.02], true),
        (vec![0.31, 0.12, -0.24, 0.28], vec![-0.01, 0.05], true),
        (vec![0.27, -0.09, 0.14, 0.33], vec![0.02, 0.01], true),
        (vec![-0.22, 0.29, 0.18, 0.24], vec![0.04, -0.03], false),
    ];

    let mut network = Network::new();
    for (weights, bias, add_relu) in layers {
        let linear = LinearLayer::new(
            Array2::from_shape_vec((2, 2), weights).unwrap(),
            Some(Array1::from_vec(bias)),
        )
        .unwrap();
        network.add_layer(Layer::Linear(linear));
        if add_relu {
            network.add_layer(Layer::ReLU(ReLULayer));
        }
    }

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.1, -0.9]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.8, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // 7 layers total with interval 2 forces the backward pass to request
    // pre-activations that are not stored exactly as checkpoints.
    let config = StreamingConfig {
        checkpoint_interval: 2,
        ..Default::default()
    };
    assert_streaming_crown_contains_concrete_outputs_with_config(&network, &input, 10, config);
}

/// #3082: Streaming CROWN falls back to IBP when CROWN backward degrades.
///
/// Uses large weights that cause f32 overflow in CROWN backward, producing
/// non-finite bounds. The streaming path should detect degradation via
/// `has_degraded_bounds` and return the finite IBP checkpoint bounds instead.
#[ntest::timeout(10000)]
#[test]
fn streaming_crown_ibp_fallback_on_degradation_3082() {
    use crate::streaming::{StreamingConfig, StreamingVerifier};

    // Build a network with large weights that overflow during CROWN backward.
    // Three layers of large weights: the CROWN matrix product overflows f32.
    let big = 1e18_f32;
    let w1 = Array2::from_shape_vec((2, 2), vec![big, big, big, big]).unwrap();
    let b1 = Some(Array1::zeros(2));
    let l1 = LinearLayer::new(w1, b1).unwrap();

    let w2 = Array2::from_shape_vec((2, 2), vec![big, big, big, big]).unwrap();
    let b2 = Some(Array1::zeros(2));
    let l2 = LinearLayer::new(w2, b2).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(l1));
    network.add_layer(Layer::Linear(l2));

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.0, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // IBP should produce finite (possibly large) bounds
    let ibp_result = network.propagate_ibp(&input).unwrap();
    let ibp_finite = ibp_result
        .lower()
        .iter()
        .chain(ibp_result.upper().iter())
        .all(|v| v.is_finite());

    // Streaming CROWN should detect degradation and fall back to IBP
    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let streaming_result = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    // The result must be finite (either IBP fallback or tightened CROWN)
    for (&l, &u) in streaming_result
        .lower()
        .iter()
        .zip(streaming_result.upper().iter())
    {
        assert!(
            l.is_finite(),
            "streaming CROWN lower bound must be finite after IBP fallback, got {l}"
        );
        assert!(
            u.is_finite(),
            "streaming CROWN upper bound must be finite after IBP fallback, got {u}"
        );
        assert!(l <= u, "inverted interval: lower {l} > upper {u}");
    }

    // If IBP produced finite bounds, the streaming result should match
    // (since degraded CROWN falls back to IBP)
    if ibp_finite {
        for (&sl, &il) in streaming_result
            .lower()
            .iter()
            .zip(ibp_result.lower().iter())
        {
            assert!(
                (sl - il).abs() < 1e-5,
                "streaming lower should match IBP on fallback: streaming={sl}, ibp={il}"
            );
        }
        for (&su, &iu) in streaming_result
            .upper()
            .iter()
            .zip(ibp_result.upper().iter())
        {
            assert!(
                (su - iu).abs() < 1e-5,
                "streaming upper should match IBP on fallback: streaming={su}, ibp={iu}"
            );
        }
    }
}
