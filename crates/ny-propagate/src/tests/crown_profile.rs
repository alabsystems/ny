// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Profiling tests for CROWN performance analysis.
//!
//! These tests measure the breakdown of time spent in CROWN propagation.

use crate::layers::BoundPropagation;
use crate::{Layer, LinearBounds, LinearLayer, Network, ReLULayer};
use ndarray::{Array1, Array2, ArrayD};
use ny_tensor::BoundedTensor;
use std::sync::Once;
use std::time::Instant;

const PROFILE_ENV: &str = "NY_PROFILE_TESTS";
static SKIP_NOTICE: Once = Once::new();

fn profiling_enabled() -> bool {
    let value = match std::env::var(PROFILE_ENV) {
        Ok(value) => value,
        Err(_) => return false,
    };

    if value.is_empty() {
        return false;
    }

    matches!(
        value.to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn profiling_or_smoke() -> bool {
    if profiling_enabled() {
        return true;
    }

    SKIP_NOTICE.call_once(|| {
        eprintln!(
            "Profiling disabled; running smoke checks. Set {}=1 for full profiling.",
            PROFILE_ENV
        );
    });
    false
}

fn assert_linear_bounds_sane(bounds: &LinearBounds) {
    assert_eq!(bounds.lower_a.dim(), bounds.upper_a.dim());
    assert_eq!(bounds.lower_b.len(), bounds.upper_b.len());
    assert_eq!(bounds.lower_a.nrows(), bounds.lower_b.len());
    assert_eq!(bounds.upper_a.nrows(), bounds.upper_b.len());
    for value in bounds
        .lower_a
        .iter()
        .chain(bounds.upper_a.iter())
        .chain(bounds.lower_b.iter())
        .chain(bounds.upper_b.iter())
    {
        assert!(value.is_finite(), "linear bound contains non-finite value");
    }
}

fn assert_bounded_tensor_sane(bounds: &BoundedTensor) {
    assert_eq!(bounds.lower().shape(), bounds.upper().shape());
    for (lower, upper) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(
            lower.is_finite() && upper.is_finite(),
            "tensor bounds contain non-finite value"
        );
        assert!(
            *lower <= *upper + 1e-6,
            "tensor bounds are inverted: {} > {}",
            lower,
            upper
        );
    }
}

/// Profile LinearLayer::propagate_linear with various sizes.
#[ntest::timeout(10000)]
#[test]
fn profile_linear_propagate() {
    let profiling = profiling_or_smoke();

    println!(
        "
=== LinearLayer::propagate_linear Profile ===
"
    );

    // Test various matrix sizes typical for NN verification
    let sizes = if profiling {
        vec![
            (10, 100, 100),   // Small: 10 outputs, 100->100 layer
            (10, 256, 256),   // Medium: 10 outputs, 256->256 layer
            (10, 512, 512),   // Large: 10 outputs, 512->512 layer
            (10, 1024, 1024), // Very large: 10 outputs, 1024->1024 layer
            (100, 512, 512),  // Many outputs: 100 outputs, 512->512 layer
            (1000, 256, 256), // High output: 1000 outputs, 256->256 layer
            (3072, 256, 256), // CIFAR10 input: 3072 outputs
        ]
    } else {
        vec![(2, 4, 4)]
    };

    for (num_outputs, out_features, in_features) in sizes {
        // Create random linear layer
        let weight = Array2::<f32>::from_elem((out_features, in_features), 0.1);
        let bias = Array1::<f32>::zeros(out_features);
        let layer = LinearLayer::new(weight, Some(bias)).unwrap();

        // Create linear bounds
        let lower_a = Array2::<f32>::from_elem((num_outputs, out_features), 0.1);
        let upper_a = Array2::<f32>::from_elem((num_outputs, out_features), 0.2);
        let lower_b = Array1::<f32>::zeros(num_outputs);
        let upper_b = Array1::<f32>::zeros(num_outputs);
        let bounds = LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap();

        // Warm up
        let warm = layer.propagate_linear(&bounds).unwrap();
        assert_eq!(warm.num_outputs(), num_outputs);
        assert_eq!(warm.num_inputs(), in_features);
        assert_linear_bounds_sane(warm.as_ref());

        // Profile
        let iterations = if profiling { 100 } else { 1 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = layer.propagate_linear(&bounds);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "Size ({} outputs × {} → {}): {:.2}μs/call ({:.2}ms for 100 calls)",
            num_outputs,
            out_features,
            in_features,
            per_call,
            elapsed.as_millis()
        );
    }
}

/// Profile LinearBounds::concretize with various sizes.
#[ntest::timeout(10000)]
#[test]
fn profile_concretize() {
    let profiling = profiling_or_smoke();

    println!(
        "
=== LinearBounds::concretize Profile ===
"
    );

    let sizes = if profiling {
        vec![(10, 100), (10, 1024), (10, 3072), (100, 3072), (1000, 3072)]
    } else {
        vec![(2, 4)]
    };

    for (num_outputs, num_inputs) in sizes {
        // Create linear bounds
        let lower_a = Array2::<f32>::from_elem((num_outputs, num_inputs), 0.1);
        let upper_a = Array2::<f32>::from_elem((num_outputs, num_inputs), 0.2);
        let lower_b = Array1::<f32>::zeros(num_outputs);
        let upper_b = Array1::<f32>::zeros(num_outputs);
        let bounds = LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap();

        // Create input bounds
        let input_lower = ArrayD::<f32>::from_elem(vec![num_inputs], 0.0);
        let input_upper = ArrayD::<f32>::from_elem(vec![num_inputs], 1.0);
        let input = BoundedTensor::new(input_lower, input_upper).unwrap();

        // Warm up
        let warm = bounds.concretize(&input);
        assert_eq!(warm.lower().len(), num_outputs);
        assert_bounded_tensor_sane(&warm);

        // Profile
        let iterations = if profiling { 100 } else { 1 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = bounds.concretize(&input);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "Size ({} outputs × {} inputs): {:.2}μs/call ({:.2}ms for 100 calls)",
            num_outputs,
            num_inputs,
            per_call,
            elapsed.as_millis()
        );
    }
}

/// Profile ReLU propagate_linear_with_bounds.
#[ntest::timeout(10000)]
#[test]
fn profile_relu_propagate() {
    let profiling = profiling_or_smoke();

    println!(
        "
=== ReLULayer::propagate_linear_with_bounds Profile ===
"
    );

    let sizes = if profiling {
        vec![(10, 256), (10, 512), (10, 1024), (100, 512)]
    } else {
        vec![(2, 4)]
    };

    let relu = ReLULayer;

    for (num_outputs, layer_dim) in sizes {
        // Create linear bounds
        let lower_a = Array2::<f32>::from_elem((num_outputs, layer_dim), 0.1);
        let upper_a = Array2::<f32>::from_elem((num_outputs, layer_dim), 0.2);
        let lower_b = Array1::<f32>::zeros(num_outputs);
        let upper_b = Array1::<f32>::zeros(num_outputs);
        let bounds = LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap();

        // Create pre-activation bounds (some positive, some negative for crossing ReLU)
        let mut pre_lower = ArrayD::<f32>::from_elem(vec![layer_dim], -0.5);
        let mut pre_upper = ArrayD::<f32>::from_elem(vec![layer_dim], 0.5);
        // Make some definitely positive
        for i in 0..layer_dim / 4 {
            pre_lower[[i]] = 0.1;
            pre_upper[[i]] = 1.0;
        }
        // Make some definitely negative
        for i in layer_dim / 4..layer_dim / 2 {
            pre_lower[[i]] = -1.0;
            pre_upper[[i]] = -0.1;
        }
        let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

        // Warm up
        let warm = relu
            .propagate_linear_with_bounds(&bounds, &pre_activation)
            .unwrap();
        assert_eq!(warm.num_outputs(), num_outputs);
        assert_eq!(warm.num_inputs(), layer_dim);
        assert_linear_bounds_sane(&warm);

        // Profile
        let iterations = if profiling { 100 } else { 1 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = relu.propagate_linear_with_bounds(&bounds, &pre_activation);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "Size ({} outputs × {} neurons): {:.2}μs/call ({:.2}ms for 100 calls)",
            num_outputs,
            layer_dim,
            per_call,
            elapsed.as_millis()
        );
    }
}

/// Profile memory allocation in LinearBounds.
#[ntest::timeout(10000)]
#[test]
fn profile_linearbounds_allocation() {
    let profiling = profiling_or_smoke();

    println!(
        "
=== LinearBounds Allocation Profile ===
"
    );

    let sizes = if profiling {
        vec![(10, 1024), (100, 1024), (1000, 3072), (3072, 3072)]
    } else {
        vec![(4, 4)]
    };

    for (rows, cols) in sizes {
        let iterations = if profiling { 100 } else { 1 };
        let start = Instant::now();
        for _ in 0..iterations {
            let bounds = LinearBounds::identity(rows.min(cols));
            assert_eq!(bounds.num_outputs(), rows.min(cols));
            assert_eq!(bounds.num_inputs(), rows.min(cols));
            assert_linear_bounds_sane(&bounds);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "Identity({} × {}): {:.2}μs/call",
            rows.min(cols),
            rows.min(cols),
            per_call
        );
    }
}

/// Full CROWN pass on a sequential network.
#[ntest::timeout(10000)]
#[test]
fn profile_crown_sequential() {
    let profiling = profiling_or_smoke();

    println!(
        "
=== Full CROWN Pass (Sequential Network) ===
"
    );

    // Create a network similar to CIFAR10 scale
    let configs = if profiling {
        vec![
            // (input_dim, hidden_dims, output_dim)
            (100, vec![64, 64], 10),
            (784, vec![256, 256], 10),
            (3072, vec![256, 256], 10), // CIFAR10-like
            (3072, vec![512, 512], 10), // Larger CIFAR10
        ]
    } else {
        vec![(16, vec![8], 4)]
    };

    for (input_dim, hidden_dims, output_dim) in configs {
        let mut network = Network::new();
        let mut prev_dim = input_dim;

        for &hidden in &hidden_dims {
            let weight = Array2::<f32>::from_elem((hidden, prev_dim), 0.01);
            let bias = Array1::<f32>::zeros(hidden);
            network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
            network.add_layer(Layer::ReLU(ReLULayer));
            prev_dim = hidden;
        }

        // Output layer
        let weight = Array2::<f32>::from_elem((output_dim, prev_dim), 0.01);
        let bias = Array1::<f32>::zeros(output_dim);
        network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));

        // Create input bounds
        let input = BoundedTensor::new(
            ArrayD::<f32>::from_elem(vec![input_dim], 0.0),
            ArrayD::<f32>::from_elem(vec![input_dim], 1.0),
        )
        .unwrap();

        // Warm up
        let warm = network.propagate_crown(&input).unwrap();
        assert_bounded_tensor_sane(&warm);

        // Profile
        let iterations = if profiling { 10 } else { 1 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = network.propagate_crown(&input);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_millis() as f64 / iterations as f64;

        println!(
            "Network {} → {:?} → {}: {:.2}ms/call",
            input_dim, hidden_dims, output_dim, per_call
        );
    }
}

/// Breakdown of CROWN time: IBP vs CROWN-IBP vs final backward pass.
#[ntest::timeout(10000)]
#[test]
fn profile_crown_breakdown() {
    let profiling = profiling_or_smoke();

    println!(
        "
=== CROWN Time Breakdown ===
"
    );

    // Create a network similar to CIFAR10 scale
    let configs = if profiling {
        vec![
            // (input_dim, hidden_dims, output_dim)
            (784, vec![256, 256], 10),
            (3072, vec![256, 256], 10), // CIFAR10-like
            (3072, vec![512, 512], 10), // Larger CIFAR10
        ]
    } else {
        vec![(16, vec![8], 4)]
    };

    for (input_dim, hidden_dims, output_dim) in configs {
        let mut network = Network::new();
        let mut prev_dim = input_dim;

        for &hidden in &hidden_dims {
            let weight = Array2::<f32>::from_elem((hidden, prev_dim), 0.01);
            let bias = Array1::<f32>::zeros(hidden);
            network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
            network.add_layer(Layer::ReLU(ReLULayer));
            prev_dim = hidden;
        }

        // Output layer
        let weight = Array2::<f32>::from_elem((output_dim, prev_dim), 0.01);
        let bias = Array1::<f32>::zeros(output_dim);
        network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));

        // Create input bounds
        let input = BoundedTensor::new(
            ArrayD::<f32>::from_elem(vec![input_dim], 0.0),
            ArrayD::<f32>::from_elem(vec![input_dim], 1.0),
        )
        .unwrap();

        println!(
            "Network {} → {:?} → {}:",
            input_dim, hidden_dims, output_dim
        );

        // Profile IBP only
        let iterations = if profiling { 10 } else { 1 };
        let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
        let ibp_last = ibp_bounds.last().expect("IBP bounds should not be empty");
        assert_bounded_tensor_sane(ibp_last);
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = network.collect_ibp_bounds(&input);
        }
        let ibp_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Profile CROWN-IBP (the expensive part)
        let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();
        let crown_ibp_last = crown_ibp_bounds
            .last()
            .expect("CROWN-IBP bounds should not be empty");
        assert_bounded_tensor_sane(crown_ibp_last);
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = network.collect_crown_ibp_bounds(&input);
        }
        let crown_ibp_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Profile full CROWN
        let full_bounds = network.propagate_crown(&input).unwrap();
        assert_bounded_tensor_sane(&full_bounds);
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = network.propagate_crown(&input);
        }
        let full_crown_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Final backward pass time = full - crown_ibp (approximately)
        let backward_time = (full_crown_time - crown_ibp_time).max(0.0);

        println!(
            "  IBP forward:        {:.2}μs ({:.1}%)",
            ibp_time,
            100.0 * ibp_time / full_crown_time
        );
        println!(
            "  CROWN-IBP:          {:.2}μs ({:.1}%)",
            crown_ibp_time,
            100.0 * crown_ibp_time / full_crown_time
        );
        println!(
            "  Final backward:     {:.2}μs ({:.1}%)",
            backward_time,
            100.0 * backward_time / full_crown_time
        );
        println!("  Total:              {:.2}μs", full_crown_time);
        println!();
    }
}

/// Compare propagate_crown_fast vs propagate_crown: speed and bound quality.
#[ntest::timeout(10000)]
#[test]
fn profile_crown_fast_comparison() {
    let profiling = profiling_or_smoke();

    println!(
        "
=== CROWN Fast vs Full CROWN Comparison ===
"
    );

    // Create a network similar to CIFAR10 scale
    let configs = if profiling {
        vec![
            // (input_dim, hidden_dims, output_dim)
            (784, vec![256, 256], 10),
            (3072, vec![256, 256], 10), // CIFAR10-like
            (3072, vec![512, 512], 10), // Larger CIFAR10
        ]
    } else {
        vec![(16, vec![8], 4)]
    };

    for (config_idx, (input_dim, hidden_dims, output_dim)) in configs.into_iter().enumerate() {
        let mut network = Network::new();
        let mut prev_dim = input_dim;

        // Use a fixed seed for deterministic profiling.
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
        let seed =
            0xC0FFEE_u64 ^ (input_dim as u64) ^ ((output_dim as u64) << 32) ^ config_idx as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        for &hidden in &hidden_dims {
            let weight =
                Array2::<f32>::from_shape_fn((hidden, prev_dim), |_| rng.random_range(-0.1..0.1));
            let bias = Array1::<f32>::zeros(hidden);
            network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
            network.add_layer(Layer::ReLU(ReLULayer));
            prev_dim = hidden;
        }

        // Output layer
        let weight =
            Array2::<f32>::from_shape_fn((output_dim, prev_dim), |_| rng.random_range(-0.1..0.1));
        let bias = Array1::<f32>::zeros(output_dim);
        network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));

        // Create input bounds with small perturbation
        let input = BoundedTensor::new(
            ArrayD::<f32>::from_elem(vec![input_dim], 0.4),
            ArrayD::<f32>::from_elem(vec![input_dim], 0.6),
        )
        .unwrap();

        println!(
            "Network {} → {:?} → {}:",
            input_dim, hidden_dims, output_dim
        );

        // Profile fast CROWN
        let iterations = if profiling { 20 } else { 1 };
        let fast_result = network.propagate_crown_fast(&input).unwrap();
        assert_bounded_tensor_sane(&fast_result);
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = network.propagate_crown_fast(&input);
        }
        let fast_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Profile full CROWN
        let full_result = network.propagate_crown(&input).unwrap();
        assert_bounded_tensor_sane(&full_result);
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = network.propagate_crown(&input);
        }
        let full_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Compute bound width (tightness metric)
        let fast_width: f32 = (fast_result.upper() - fast_result.lower()).iter().sum();
        let full_width: f32 = (full_result.upper() - full_result.lower()).iter().sum();
        let safe_full_width = if full_width.abs() < f32::EPSILON {
            f32::EPSILON
        } else {
            full_width
        };
        let width_ratio = fast_width / safe_full_width;

        let safe_full_time = full_time.max(f64::EPSILON);
        let safe_fast_time = fast_time.max(f64::EPSILON);
        let speedup = safe_full_time / safe_fast_time;

        println!(
            "  Fast CROWN:  {:.2}μs ({:.2}ms)",
            fast_time,
            fast_time / 1000.0
        );
        println!(
            "  Full CROWN:  {:.2}μs ({:.2}ms)",
            full_time,
            full_time / 1000.0
        );
        println!("  Speedup:     {:.2}x", speedup);
        println!("  Fast width:  {:.4}", fast_width);
        println!("  Full width:  {:.4}", full_width);
        println!(
            "  Width ratio: {:.2}x (1.0 = same tightness, higher = looser)",
            width_ratio
        );
        println!();
    }
}
