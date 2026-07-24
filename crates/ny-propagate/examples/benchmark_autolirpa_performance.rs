// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Performance comparison benchmark: ny vs Auto-LiRPA references.
//!
//! Run with: cargo run --release --example benchmark_autolirpa_performance -p ny-propagate

use ndarray::{arr1, arr2};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{Layer, Network};
use ny_tensor::BoundedTensor;
use std::time::Instant;

fn build_toy_benchmark_fixture() -> (Network, BoundedTensor) {
    let w1 = arr2(&[[1.0f32, -1.0], [2.0, -1.0]]);
    let w2 = arr2(&[[1.0f32, -1.0]]);

    let mut toy_network = Network::new();
    toy_network.add_layer(Layer::Linear(
        LinearLayer::new(w1, None).expect("toy benchmark weight shape should be valid"),
    ));
    toy_network.add_layer(Layer::ReLU(ReLULayer));
    toy_network.add_layer(Layer::Linear(
        LinearLayer::new(w2, None).expect("toy benchmark weight shape should be valid"),
    ));

    let toy_input = BoundedTensor::new(
        arr1(&[-1.0f32, -2.0]).into_dyn(),
        arr1(&[2.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    (toy_network, toy_input)
}

fn main() {
    // Auto-LiRPA reference timing (Python, PyTorch 2.9.1):
    // Toy Model (100 iterations):
    //   IBP:   0.288 ms/iter
    //   CROWN: 0.731 ms/iter
    // Deep Model (100 iterations):
    //   IBP:   0.384 ms/iter
    //   CROWN: 0.768 ms/iter

    let (toy_network, toy_input) = build_toy_benchmark_fixture();

    // Deep model setup
    let fc1_weight = arr2(&[
        [0.16834518f32, 0.064_404_7, 0.11723118],
        [0.11516652, -0.561_428_2, -0.09316415],
        [1.104_100_7, -0.31899852, 0.23082861],
        [0.13367544, 0.26745233, 0.404_678_6],
    ]);
    let fc1_bias = arr1(&[0.11102903f32, -0.168_979_9, -0.09889599, 0.09579718]);
    let fc2_weight = arr2(&[
        [-0.692_337_1_f32, -0.43561807, -0.11168297, 0.85868055],
        [0.15943986, -0.21225949, 0.15286016, -0.38729626],
        [-0.778_786_1, 0.49781805, -0.43989292, -0.30057147],
        [-0.637_075_7, 1.061_392_5, -0.617_326_7, -0.24395694],
    ]);
    let fc2_bias = arr1(&[0.02815196f32, 0.00561635, 0.052_271_6, -0.02383569]);
    let fc3_weight = arr2(&[
        [-0.02495167f32, 0.26316848, -0.00424941, 0.364_530_3],
        [0.06657098, 0.43198884, -0.50783736, -0.44437426],
    ]);
    let fc3_bias = arr1(&[0.01497797f32, -0.02088939]);

    let mut deep_network = Network::new();
    deep_network.add_layer(Layer::Linear(
        LinearLayer::new(fc1_weight, Some(fc1_bias)).unwrap(),
    ));
    deep_network.add_layer(Layer::ReLU(ReLULayer));
    deep_network.add_layer(Layer::Linear(
        LinearLayer::new(fc2_weight, Some(fc2_bias)).unwrap(),
    ));
    deep_network.add_layer(Layer::ReLU(ReLULayer));
    deep_network.add_layer(Layer::Linear(
        LinearLayer::new(fc3_weight, Some(fc3_bias)).unwrap(),
    ));

    let deep_input = BoundedTensor::new(
        arr1(&[0.4f32, 0.4, 0.4]).into_dyn(),
        arr1(&[0.6f32, 0.6, 0.6]).into_dyn(),
    )
    .unwrap();

    let n_iters = 1000;

    // Warm-up
    for _ in 0..10 {
        let _ = toy_network.propagate_ibp(&toy_input);
        let _ = toy_network.propagate_crown(&toy_input);
        let _ = deep_network.propagate_ibp(&deep_input);
        let _ = deep_network.propagate_crown(&deep_input);
    }

    // Benchmark Toy Model IBP
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = toy_network.propagate_ibp(&toy_input);
    }
    let toy_ibp_us = start.elapsed().as_micros() as f64 / n_iters as f64;

    // Benchmark Toy Model CROWN
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = toy_network.propagate_crown(&toy_input);
    }
    let toy_crown_us = start.elapsed().as_micros() as f64 / n_iters as f64;

    // Benchmark Deep Model IBP
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = deep_network.propagate_ibp(&deep_input);
    }
    let deep_ibp_us = start.elapsed().as_micros() as f64 / n_iters as f64;

    // Benchmark Deep Model CROWN
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = deep_network.propagate_crown(&deep_input);
    }
    let deep_crown_us = start.elapsed().as_micros() as f64 / n_iters as f64;

    // Auto-LiRPA reference times (in microseconds)
    let ref_toy_ibp_us = 288.0;
    let ref_toy_crown_us = 731.0;
    let ref_deep_ibp_us = 384.0;
    let ref_deep_crown_us = 768.0;

    println!("\n=== Performance Comparison: ny vs Auto-LiRPA ===");
    println!("({} iterations)\n", n_iters);

    println!("Toy Model:");
    println!(
        "  IBP:   ny={:.1}us, Auto-LiRPA={:.0}us, speedup={:.1}x",
        toy_ibp_us,
        ref_toy_ibp_us,
        ref_toy_ibp_us / toy_ibp_us
    );
    println!(
        "  CROWN: ny={:.1}us, Auto-LiRPA={:.0}us, speedup={:.1}x",
        toy_crown_us,
        ref_toy_crown_us,
        ref_toy_crown_us / toy_crown_us
    );

    println!("\nDeep Model:");
    println!(
        "  IBP:   ny={:.1}us, Auto-LiRPA={:.0}us, speedup={:.1}x",
        deep_ibp_us,
        ref_deep_ibp_us,
        ref_deep_ibp_us / deep_ibp_us
    );
    println!(
        "  CROWN: ny={:.1}us, Auto-LiRPA={:.0}us, speedup={:.1}x",
        deep_crown_us,
        ref_deep_crown_us,
        ref_deep_crown_us / deep_crown_us
    );

    let slowdown_threshold = if cfg!(debug_assertions) { 50.0 } else { 10.0 };
    if toy_ibp_us >= ref_toy_ibp_us * slowdown_threshold {
        eprintln!(
            "Warning: toy IBP slower than {}x reference ({}us vs {}us).",
            slowdown_threshold, toy_ibp_us, ref_toy_ibp_us
        );
    }
    if toy_crown_us >= ref_toy_crown_us * slowdown_threshold {
        eprintln!(
            "Warning: toy CROWN slower than {}x reference ({}us vs {}us).",
            slowdown_threshold, toy_crown_us, ref_toy_crown_us
        );
    }
}
