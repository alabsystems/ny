//! Benchmark comparing ny bounds against Auto-LiRPA reference.
//!
//! This is meant to be run as a standalone Rust file to compare bounds.
//! Run with: cargo test --package ny-propagate -- benchmark --nocapture

use ndarray::{arr1, arr2, Array1, Array2};
use ny_propagate::{AlphaCrownConfig, Layer, LinearLayer, Network, ReLULayer};
use ny_tensor::BoundedTensor;
use std::time::Instant;

/// Toy model from Auto-LiRPA example:
/// - Linear: 2 -> 2 (w1=[[1, -1], [2, -1]], no bias)
/// - ReLU
/// - Linear: 2 -> 1 (w2=[[1, -1]], no bias)
///
/// Input bounds: lower=[-1, -2], upper=[2, 1]
///
/// Auto-LiRPA reference results:
/// - IBP: lower=-6.0, upper=4.0
/// - CROWN: lower=-3.0, upper=3.0
/// - alpha-CROWN: lower=-3.0, upper=3.0
fn benchmark_toy_model() {
    println!("=== Toy Model Benchmark ===\n");

    // Create network with exact weights from Auto-LiRPA
    let w1 = arr2(&[[1.0f32, -1.0], [2.0, -1.0]]);
    let w2 = arr2(&[[1.0f32, -1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), None).unwrap()));

    // Input bounds from Auto-LiRPA: lower=[-1, -2], upper=[2, 1]
    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0f32, -2.0]).into_dyn(),
        arr1(&[2.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    println!("Network weights:");
    println!("  w1 = {:?}", w1.to_vec());
    println!("  w2 = {:?}", w2.to_vec());
    println!("\nInput bounds:");
    println!("  lower = {:?}", input_bounds.lower.as_slice().unwrap());
    println!("  upper = {:?}", input_bounds.upper.as_slice().unwrap());

    // Reference bounds from Auto-LiRPA
    let ref_ibp_lower = -6.0f32;
    let ref_ibp_upper = 4.0f32;
    let ref_crown_lower = -3.0f32;
    let ref_crown_upper = 3.0f32;

    // IBP
    let start = Instant::now();
    let ibp_result = network.propagate_ibp(&input_bounds).unwrap();
    let ibp_time = start.elapsed();

    let ibp_lower = ibp_result.lower[[0]];
    let ibp_upper = ibp_result.upper[[0]];

    println!("\n--- IBP ---");
    println!("  ny:  lower={:.6}, upper={:.6}", ibp_lower, ibp_upper);
    println!(
        "  Auto-LiRPA:   lower={:.6}, upper={:.6}",
        ref_ibp_lower, ref_ibp_upper
    );
    println!(
        "  Match: lower={}, upper={}",
        (ibp_lower - ref_ibp_lower).abs() < 1e-5,
        (ibp_upper - ref_ibp_upper).abs() < 1e-5
    );
    println!("  Time: {:?}", ibp_time);

    // CROWN
    let start = Instant::now();
    let crown_result = network.propagate_crown(&input_bounds).unwrap();
    let crown_time = start.elapsed();

    let crown_lower = crown_result.lower[[0]];
    let crown_upper = crown_result.upper[[0]];

    println!("\n--- CROWN ---");
    println!("  ny:  lower={:.6}, upper={:.6}", crown_lower, crown_upper);
    println!(
        "  Auto-LiRPA:   lower={:.6}, upper={:.6}",
        ref_crown_lower, ref_crown_upper
    );
    println!(
        "  Match: lower={}, upper={}",
        (crown_lower - ref_crown_lower).abs() < 1e-5,
        (crown_upper - ref_crown_upper).abs() < 1e-5
    );
    println!("  Time: {:?}", crown_time);

    // alpha-CROWN
    let config = AlphaCrownConfig::default();
    let start = Instant::now();
    let alpha_result = network.propagate_alpha_crown(&input_bounds).unwrap();
    let alpha_time = start.elapsed();

    let alpha_lower = alpha_result.lower[[0]];
    let alpha_upper = alpha_result.upper[[0]];

    println!("\n--- alpha-CROWN ---");
    println!("  ny:  lower={:.6}, upper={:.6}", alpha_lower, alpha_upper);
    println!(
        "  Auto-LiRPA:   lower={:.6}, upper={:.6}",
        ref_crown_lower, ref_crown_upper
    );
    println!(
        "  Match: lower={}, upper={}",
        (alpha_lower - ref_crown_lower).abs() < 1e-5,
        (alpha_upper - ref_crown_upper).abs() < 1e-5
    );
    println!("  Time: {:?}", alpha_time);

    // Verify soundness by testing concrete points
    println!("\n--- Soundness Check ---");
    let test_inputs = [
        arr1(&[-1.0f32, -2.0]),
        arr1(&[2.0f32, 1.0]),
        arr1(&[0.5f32, -0.5]),
        arr1(&[-1.0f32, 1.0]),
        arr1(&[2.0f32, -2.0]),
    ];

    for x in &test_inputs {
        // Compute concrete output: Linear -> ReLU -> Linear
        let z1 = w1.dot(x);
        let a1 = z1.mapv(|v| v.max(0.0));
        let y = w2.dot(&a1);
        let output = y[0];

        let in_ibp = output >= ibp_lower - 1e-5 && output <= ibp_upper + 1e-5;
        let in_crown = output >= crown_lower - 1e-5 && output <= crown_upper + 1e-5;
        let in_alpha = output >= alpha_lower - 1e-5 && output <= alpha_upper + 1e-5;

        println!(
            "  x={:?} -> y={:.4} | IBP:{} CROWN:{} alpha:{}",
            x.to_vec(),
            output,
            if in_ibp { "OK" } else { "FAIL" },
            if in_crown { "OK" } else { "FAIL" },
            if in_alpha { "OK" } else { "FAIL" }
        );
    }
}

fn main() {
    benchmark_toy_model();
}
