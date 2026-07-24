// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// NOTE: split from tests.rs for maintainability.

use super::prelude::*;

// ==================== Optimizer Benchmark Tests ====================
//
// These tests compare different optimizer variants (Adam, AMSGrad, AdamW, RAdam, Lookahead)
// on fixed verification problems to measure:
// - Final lower bound (higher = tighter bounds)
// - Convergence speed (iterations to reach tolerance)
// - Stability (variance across runs)

/// Results from a single optimizer run.
#[derive(Debug, Clone)]
struct OptBenchmarkResult {
    /// Name of the optimizer variant.
    name: String,
    /// Final lower bound achieved.
    final_lower: f32,
    /// Number of domains explored until completion.
    iterations: usize,
    /// Whether optimization converged (verified) before domain/time limit.
    converged: bool,
}

/// Create a deeper network for more meaningful optimizer comparison.
/// Structure: Linear(4,8) -> ReLU -> Linear(8,4) -> ReLU -> Linear(4,1)
fn benchmark_network() -> Network {
    use ndarray::Array2;

    // Layer 1: Linear 4 -> 8 (creates more unstable neurons)
    let w1: Array2<f32> = arr2(&[
        [0.5, -0.3, 0.2, -0.4],
        [-0.2, 0.6, -0.1, 0.3],
        [0.4, -0.5, 0.3, -0.2],
        [-0.3, 0.2, -0.4, 0.5],
        [0.1, -0.6, 0.5, -0.1],
        [-0.5, 0.4, -0.3, 0.6],
        [0.6, -0.2, 0.1, -0.5],
        [-0.4, 0.3, -0.6, 0.2],
    ]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    // Layer 2: Linear 8 -> 4
    let w2: Array2<f32> = arr2(&[
        [0.3, -0.2, 0.4, -0.3, 0.2, -0.1, 0.5, -0.4],
        [-0.4, 0.5, -0.2, 0.3, -0.5, 0.4, -0.3, 0.2],
        [0.2, -0.4, 0.3, -0.5, 0.4, -0.3, 0.1, -0.2],
        [-0.3, 0.1, -0.5, 0.2, -0.2, 0.5, -0.4, 0.3],
    ]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    // Layer 3: Linear 4 -> 1 (output)
    let w3: Array2<f32> = arr2(&[[0.5, -0.3, 0.4, -0.2]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));
    network
}

/// Run optimization with a specific config and return metrics.
/// Uses a tighter threshold (-0.5) to require more domain exploration.
fn run_optimizer_benchmark(
    name: &str,
    adaptive_config: AdaptiveOptConfig,
    max_iterations: usize,
) -> OptBenchmarkResult {
    run_optimizer_benchmark_with_threshold(name, adaptive_config, max_iterations, -0.5)
}

/// Run optimization with a specific config and threshold.
fn run_optimizer_benchmark_with_threshold(
    name: &str,
    adaptive_config: AdaptiveOptConfig,
    max_iterations: usize,
    threshold: f32,
) -> OptBenchmarkResult {
    let network = benchmark_network();

    // Input bounds that create unstable neurons (wider range for harder problem)
    let input = BoundedTensor::new(
        arr1(&[-1.0, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0, 1.0, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();

    // Config with adaptive optimizer
    let config = BetaCrownConfig {
        max_domains: 100, // More domains for harder problems
        timeout: Duration::from_secs(30),
        use_alpha_crown: true,
        use_adaptive: true,
        adaptive_config,
        beta_lr: 0.1,
        alpha_lr: 0.5,
        alpha_momentum: true,
        beta_iterations: max_iterations,
        beta_tolerance: 1e-5,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    // Run verification - this exercises the optimizer
    let result = verifier.verify(&network, &input, threshold).unwrap();

    // Extract metrics from the result
    let final_lower = match &result.result {
        BabVerificationStatus::Verified => result
            .output_bounds
            .as_ref()
            .map(|b| b.lower_scalar())
            .unwrap_or(0.0),
        BabVerificationStatus::Unknown { .. } => result
            .output_bounds
            .as_ref()
            .map(|b| b.lower_scalar())
            .unwrap_or(-f32::INFINITY),
        _ => -f32::INFINITY,
    };

    OptBenchmarkResult {
        name: name.to_string(),
        final_lower,
        iterations: result.domains_explored,
        converged: matches!(result.result, BabVerificationStatus::Verified),
    }
}

#[ntest::timeout(60000)]
#[test]
fn test_benchmark_adam_baseline() {
    // Baseline Adam optimizer
    let config = AdaptiveOptConfig {
        beta_lr: 0.1,
        alpha_lr: 0.3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        bias_correction: true,
        ..Default::default()
    };

    let result = run_optimizer_benchmark("Adam", config, 20);
    println!(
        "Adam baseline: lower={:.6}, domains={}, converged={}",
        result.final_lower, result.iterations, result.converged
    );

    // Verifier should explore domains even if the problem doesn't converge.
    // After #1865, the verifier correctly returns Unknown when domains hit
    // depth/propagation limits instead of falsely claiming Verified.
    assert!(result.iterations > 0, "Should explore at least one domain");
}

#[ntest::timeout(60000)]
#[test]
fn test_benchmark_amsgrad() {
    // AMSGrad variant
    let config = AdaptiveOptConfig {
        beta_lr: 0.1,
        alpha_lr: 0.3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        bias_correction: true,
        amsgrad: true,
        ..Default::default()
    };

    let result = run_optimizer_benchmark("AMSGrad", config, 20);
    println!(
        "AMSGrad: lower={:.6}, domains={}, converged={}",
        result.final_lower, result.iterations, result.converged
    );

    assert!(result.iterations > 0, "Should explore at least one domain");
}

#[ntest::timeout(60000)]
#[test]
fn test_benchmark_adamw() {
    // AdamW variant with weight decay
    let config = AdaptiveOptConfig {
        beta_lr: 0.1,
        alpha_lr: 0.3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        bias_correction: true,
        weight_decay: 0.01,
        ..Default::default()
    };

    let result = run_optimizer_benchmark("AdamW", config, 20);
    println!(
        "AdamW: lower={:.6}, domains={}, converged={}",
        result.final_lower, result.iterations, result.converged
    );

    assert!(result.iterations > 0, "Should explore at least one domain");
}

#[ntest::timeout(60000)]
#[test]
fn test_benchmark_radam() {
    // RAdam variant
    // Note: RAdam underperforms on this β-CROWN benchmark due to its warmup phase.
    // RAdam uses SGD-with-momentum steps for early iterations (before variance
    // estimate is reliable), which may not be optimal for constraint optimization.
    let config = AdaptiveOptConfig {
        beta_lr: 0.1,
        alpha_lr: 0.3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        bias_correction: true,
        radam: true,
        ..Default::default()
    };

    let result = run_optimizer_benchmark("RAdam", config, 20);
    println!(
        "RAdam: lower={:.6}, domains={}, converged={}",
        result.final_lower, result.iterations, result.converged
    );

    assert!(result.iterations > 0, "Should explore at least one domain");
}

#[ntest::timeout(60000)]
#[test]
fn test_benchmark_lookahead() {
    // Lookahead + Adam
    let config = AdaptiveOptConfig {
        beta_lr: 0.1,
        alpha_lr: 0.3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        bias_correction: true,
        lookahead: LookaheadConfig::new(5, 0.5),
        ..Default::default()
    };

    let result = run_optimizer_benchmark("Lookahead+Adam", config, 20);
    println!(
        "Lookahead+Adam: lower={:.6}, domains={}, converged={}",
        result.final_lower, result.iterations, result.converged
    );

    assert!(result.iterations > 0, "Should explore at least one domain");
}

#[ntest::timeout(60000)]
#[test]
fn test_benchmark_radam_integration() {
    // Integration test: RAdam with joint α-β optimization
    // This specifically tests RAdam in the full verification pipeline
    let network = benchmark_network();

    let input = BoundedTensor::new(
        arr1(&[-0.3, -0.3, -0.3, -0.3]).into_dyn(),
        arr1(&[0.3, 0.3, 0.3, 0.3]).into_dyn(),
    )
    .unwrap();

    let config = BetaCrownConfig {
        max_domains: 30,
        timeout: Duration::from_secs(20),
        use_alpha_crown: true,
        use_adaptive: true,
        adaptive_config: AdaptiveOptConfig {
            beta_lr: 0.1,
            alpha_lr: 0.3,
            radam: true,
            bias_correction: true,
            ..Default::default()
        },
        beta_iterations: 15,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -1.5).unwrap();

    println!(
        "RAdam integration test: status={:?}, domains={}",
        result.result, result.domains_explored
    );

    // Should produce valid result
    assert!(
        result.domains_explored > 0,
        "Should explore at least one domain"
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_optimizer_comparison_report() {
    // Comprehensive comparison of all optimizer variants
    // Runs each optimizer and prints a comparison table

    let variants: Vec<(&str, AdaptiveOptConfig)> = vec![
        (
            "Adam (baseline)",
            AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                ..Default::default()
            },
        ),
        (
            "AMSGrad",
            AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                amsgrad: true,
                ..Default::default()
            },
        ),
        (
            "AdamW (λ=0.01)",
            AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                weight_decay: 0.01,
                ..Default::default()
            },
        ),
        (
            "RAdam",
            AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                radam: true,
                ..Default::default()
            },
        ),
        (
            "Lookahead+Adam",
            AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                lookahead: LookaheadConfig::new(5, 0.5),
                ..Default::default()
            },
        ),
        (
            "RAdam+AMSGrad",
            AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                radam: true,
                amsgrad: true,
                ..Default::default()
            },
        ),
        (
            "RAdam+Lookahead",
            AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                radam: true,
                lookahead: LookaheadConfig::new(5, 0.5),
                ..Default::default()
            },
        ),
    ];

    println!(
        "
=== Optimizer Comparison Report ===
"
    );
    println!(
        "{:<20} {:>12} {:>10} {:>10}",
        "Optimizer", "Lower Bound", "Domains", "Converged"
    );
    println!("{}", "-".repeat(54));

    let mut results = Vec::new();
    for (name, config) in variants {
        let result = run_optimizer_benchmark(name, config, 20);
        println!(
            "{:<20} {:>12.6} {:>10} {:>10}",
            result.name,
            result.final_lower,
            result.iterations,
            if result.converged { "yes" } else { "no" }
        );
        results.push(result);
    }

    println!(
        "
{}",
        "=".repeat(54)
    );

    // Find best converged result
    let converged_results: Vec<_> = results.iter().filter(|r| r.converged).collect();
    let non_converged: Vec<_> = results.iter().filter(|r| !r.converged).collect();

    if !converged_results.is_empty() {
        let best = converged_results
            .iter()
            .min_by_key(|r| r.iterations) // Fewer domains = faster
            .unwrap();
        println!(
            "Best (fewest domains): {} ({} domains)",
            best.name, best.iterations
        );
    }

    if !non_converged.is_empty() {
        println!(
            "
Did not converge within domain limit:"
        );
        for r in &non_converged {
            println!("  - {}", r.name);
        }
    }

    // Analysis summary
    let converged_count = converged_results.len();
    let total = results.len();
    println!(
        "
Convergence rate: {}/{} ({:.0}%)",
        converged_count,
        total,
        (converged_count as f32 / total as f32) * 100.0
    );

    // After #1865, the verifier correctly returns Unknown when domains hit
    // depth/propagation limits. Convergence is not guaranteed for hard problems.
    // The comparison table is still useful for relative optimizer analysis.
    // If any converge, verify their bounds are valid.
    for result in &converged_results {
        assert!(
            result.final_lower.is_finite(),
            "{} converged but has invalid lower bound",
            result.name
        );
    }

    // All optimizers should at least explore some domains.
    for result in &results {
        assert!(
            result.iterations > 0,
            "{} should explore at least one domain",
            result.name
        );
    }
}

#[ntest::timeout(60000)]
#[test]
fn test_optimizer_lr_schedule_comparison() {
    // Compare learning rate schedules with Adam
    let schedules: Vec<(&str, LRScheduler)> = vec![
        ("Constant", LRScheduler::Constant),
        (
            "Step (γ=0.5, s=5)",
            LRScheduler::StepDecay {
                ny: 0.5,
                step_size: 5,
            },
        ),
        (
            "Exponential (γ=0.95)",
            LRScheduler::ExponentialDecay { ny: 0.95 },
        ),
        (
            "Cosine (T=20)",
            LRScheduler::CosineAnnealing {
                t_max: 20,
                min_lr: 0.001,
            },
        ),
        (
            "WarmupCosine",
            LRScheduler::WarmupCosine {
                warmup_steps: 5,
                min_lr: 0.001,
                t_max: 20,
            },
        ),
    ];

    println!(
        "
=== LR Schedule Comparison ===
"
    );
    println!(
        "{:<25} {:>12} {:>10} {:>10}",
        "Schedule", "Lower Bound", "Domains", "Converged"
    );
    println!("{}", "-".repeat(59));

    let mut results = Vec::new();
    for (name, scheduler) in schedules {
        let config = AdaptiveOptConfig {
            beta_lr: 0.1,
            alpha_lr: 0.3,
            scheduler,
            ..Default::default()
        };

        let result = run_optimizer_benchmark(name, config, 20);
        println!(
            "{:<25} {:>12.6} {:>10} {:>10}",
            result.name,
            result.final_lower,
            result.iterations,
            if result.converged { "yes" } else { "no" }
        );
        results.push(result);
    }

    // After #1865, the verifier correctly returns Unknown when domains hit
    // depth/propagation limits. Convergence is not guaranteed for hard problems.
    let converged_count = results.iter().filter(|r| r.converged).count();
    println!(
        "
Convergence rate: {}/{}",
        converged_count,
        results.len()
    );

    // All schedules should at least explore domains.
    for result in &results {
        assert!(
            result.iterations > 0,
            "{} should explore at least one domain",
            result.name
        );
    }
}

// =========================================================================
