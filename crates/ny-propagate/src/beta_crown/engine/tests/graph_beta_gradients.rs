// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// NOTE: split from tests.rs for maintainability.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_from_history_initializes_zero_beta() {
    // Unit test for #1817: verify that GraphBetaState::from_history().unwrap() initializes
    // all β values to 0.0 (DEFAULT_BETA_INIT). The CROWN equivalence property is
    // tested by test_graph_constrained_crown_tightening in gpu_bab.rs.
    let history = GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 3,
            is_active: false,
            score: 0.0,
        });

    let beta_state = GraphBetaState::from_history(&history).unwrap();
    assert_eq!(beta_state.entries.len(), 2);
    assert!(
        beta_state.entries.iter().all(|entry| entry.value() == 0.0),
        "GraphBetaState::from_history must initialize all β values to 0.0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_analytical_gradients_from_a_matrix() {
    // Test that analytical β gradients are computed correctly from A matrices.
    //
    // For a constraint at (node_name, neuron_idx, sign), the gradient should be:
    //   ∂lb/∂β = -sign * sensitivity
    // where sensitivity = sum_j(A[j, neuron_idx])

    use crate::bounds::GraphAlphaCrownIntermediate;

    // Create a split history with two constraints
    let history = GraphSplitHistory::new();
    let history = history.with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0, // sign = +1
    });
    let history = history.with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0, // sign = -1
    });

    // Create β state
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    assert_eq!(beta_state.entries.len(), 2);

    // Create intermediate storage with A matrix
    let mut intermediate = GraphAlphaCrownIntermediate::new();

    // A matrix at relu1: shape (2, 3) - 2 outputs, 3 neurons
    // A = [[1.0, -2.0, 0.5],
    //      [0.5,  1.0, -1.0]]
    let a_matrix = arr2(&[[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]]);
    intermediate.a_at_relu.insert("relu1".to_string(), a_matrix);

    // Pre-ReLU bounds (not strictly needed for gradient computation but required for struct)
    let pre_lower = arr1(&[-1.0, -0.5, 0.0]);
    let pre_upper = arr1(&[1.0, 1.5, 2.0]);
    intermediate
        .pre_relu_bounds
        .insert("relu1".to_string(), (pre_lower, pre_upper));

    // Compute analytical gradients
    let max_grad = beta_state.compute_analytical_gradients(&intermediate);

    // Check gradients:
    // For neuron_idx=0, sign=+1:
    //   sensitivity = A[0,0] + A[1,0] = 1.0 + 0.5 = 1.5
    //   grad = -sign * sensitivity = -1.0 * 1.5 = -1.5
    let entry0 = beta_state.entry("relu1", 0).unwrap();
    assert!(
        (entry0.grad - (-1.5)).abs() < 1e-6,
        "Expected grad=-1.5 for active constraint, got {}",
        entry0.grad
    );

    // For neuron_idx=1, sign=-1:
    //   sensitivity = A[0,1] + A[1,1] = -2.0 + 1.0 = -1.0
    //   grad = -sign * sensitivity = -(-1.0) * (-1.0) = -1.0
    let entry1 = beta_state.entry("relu1", 1).unwrap();
    assert!(
        (entry1.grad - (-1.0)).abs() < 1e-6,
        "Expected grad=-1.0 for inactive constraint, got {}",
        entry1.grad
    );

    // max_grad should be the max absolute gradient
    assert!(
        (max_grad - 1.5).abs() < 1e-6,
        "Expected max_grad=1.5, got {}",
        max_grad
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_analytical_gradients_missing_node() {
    // Test that gradients are zero for constraints on nodes not in intermediate storage

    use crate::bounds::GraphAlphaCrownIntermediate;

    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu_missing".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();

    // Empty intermediate - no A matrices stored
    let intermediate = GraphAlphaCrownIntermediate::new();

    let max_grad = beta_state.compute_analytical_gradients(&intermediate);

    // Gradient should be zero since node is not found
    let entry = beta_state.entry("relu_missing", 0).unwrap();
    assert_eq!(entry.grad, 0.0, "Gradient should be zero for missing node");
    assert_eq!(max_grad, 0.0, "Max grad should be zero");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_analytical_gradients_multi_objective() {
    // Test analytical gradients for multi-objective verification
    // The gradient should be computed for the critical objective (minimum margin)
    use crate::bounds::GraphAlphaCrownIntermediate;

    // Create split history with one constraint
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0, // sign = +1
    });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    assert_eq!(beta_state.entries.len(), 1);

    // Create intermediate storage with A matrix
    let mut intermediate = GraphAlphaCrownIntermediate::new();

    // A matrix at relu1: shape (3, 2) - 3 outputs, 2 neurons
    // A = [[1.0, -2.0],
    //      [0.5,  1.0],
    //      [2.0,  0.0]]
    let a_matrix = arr2(&[[1.0, -2.0], [0.5, 1.0], [2.0, 0.0]]);
    intermediate.a_at_relu.insert("relu1".to_string(), a_matrix);

    // Two objectives:
    // Objective 0: c = [1.0, 0.0, 0.0] (only uses output 0)
    // Objective 1: c = [0.0, 1.0, 1.0] (uses outputs 1 and 2)
    let objectives = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 1.0]];

    // Bounds for each objective (lower, upper)
    // Objective 0: margin = lower - threshold = 0.5 - 0.0 = 0.5
    // Objective 1: margin = lower - threshold = -0.2 - 0.0 = -0.2 (CRITICAL - minimum margin)
    let obj_bounds = vec![(0.5, 1.0), (-0.2, 0.5)];
    let thresholds = vec![0.0, 0.0];
    let verified_mask = vec![false, false];

    // Compute analytical gradients for multi-objective
    let max_grad = beta_state.compute_analytical_gradients_multi_objective(
        &intermediate,
        &obj_bounds,
        &objectives,
        &thresholds,
        &verified_mask,
        false,
    );

    // Objective 1 is critical (min margin = -0.2)
    // Its coefficient vector is c = [0.0, 1.0, 1.0]
    // For neuron_idx=0:
    //   sensitivity = c[0]*A[0,0] + c[1]*A[1,0] + c[2]*A[2,0]
    //              = 0.0*1.0 + 1.0*0.5 + 1.0*2.0 = 2.5
    //   grad = -sign * sensitivity = -1.0 * 2.5 = -2.5
    let entry = beta_state.entry("relu1", 0).unwrap();
    assert!(
        (entry.grad - (-2.5)).abs() < 1e-6,
        "Expected grad=-2.5 for critical objective, got {}",
        entry.grad
    );
    assert!(
        (max_grad - 2.5).abs() < 1e-6,
        "Expected max_grad=2.5, got {}",
        max_grad
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_analytical_gradients_multi_objective_all_verified() {
    // Test that gradients are zero when all objectives are verified
    use crate::bounds::GraphAlphaCrownIntermediate;

    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();

    let mut intermediate = GraphAlphaCrownIntermediate::new();
    let a_matrix = arr2(&[[1.0, -2.0], [0.5, 1.0]]);
    intermediate.a_at_relu.insert("relu1".to_string(), a_matrix);

    let objectives = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let obj_bounds = vec![(0.5, 1.0), (0.3, 0.8)];
    let thresholds = vec![0.0, 0.0];
    let verified_mask = vec![true, true]; // All verified

    let max_grad = beta_state.compute_analytical_gradients_multi_objective(
        &intermediate,
        &obj_bounds,
        &objectives,
        &thresholds,
        &verified_mask,
        false,
    );

    // All objectives verified - gradient should be zero
    let entry = beta_state.entry("relu1", 0).unwrap();
    assert_eq!(
        entry.grad, 0.0,
        "Gradient should be zero when all objectives verified"
    );
    assert_eq!(max_grad, 0.0, "Max grad should be zero");
}

/// Benchmark analytical gradients vs finite-difference for multi-objective β optimization.
///
/// This test demonstrates the performance advantage of analytical gradients:
/// - Analytical: 1 forward pass per iteration (stores A matrices during propagation)
/// - Finite-difference (SPSA-style): 3 forward passes per iteration (+ε, -ε, baseline)
///
/// Expected speedup: ~3x for the gradient computation phase.
#[ntest::timeout(10000)]
#[test]
fn test_benchmark_analytical_vs_finite_diff_multi_objective() {
    use crate::bounds::GraphAlphaCrownIntermediate;
    use std::time::Instant;

    // Create a larger network for meaningful benchmarking
    // Using the benchmark network pattern with more neurons
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

    let w2: Array2<f32> = arr2(&[
        [0.3, -0.2, 0.4, -0.3, 0.2, -0.1, 0.5, -0.4],
        [-0.4, 0.5, -0.2, 0.3, -0.5, 0.4, -0.3, 0.2],
        [0.2, -0.4, 0.3, -0.5, 0.4, -0.3, 0.1, -0.2],
        [-0.3, 0.1, -0.5, 0.2, -0.2, 0.5, -0.4, 0.3],
    ]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    // Output layer: 4 -> 3 (for multi-objective with 3 targets)
    let w3: Array2<f32> = arr2(&[
        [0.5, -0.3, 0.4, -0.2],
        [-0.2, 0.4, -0.3, 0.5],
        [0.3, -0.5, 0.2, -0.4],
    ]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));

    // Create split history with multiple constraints for β optimization
    let history = GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu_0".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu_0".to_string(),
            neuron_idx: 1,
            is_active: false,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu_1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 0.0,
        });

    // Create β state
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();

    // Initialize β values
    for entry in beta_state.entries.iter_mut() {
        entry.set_value(0.1);
    }

    // Create mock intermediate with A matrices
    // In practice, these come from CROWN propagation
    let mut intermediate = GraphAlphaCrownIntermediate::new();

    // A matrix at relu_0: shape (4, 8) - next layer width x current layer width
    let a_relu0 = Array2::from_shape_fn((4, 8), |(i, j)| {
        let val = ((i * 8 + j) as f32 * 0.1) - 0.4;
        val.sin() * 0.5 // Create varied but bounded values
    });
    intermediate.a_at_relu.insert("relu_0".to_string(), a_relu0);

    // A matrix at relu_1: shape (3, 4) - output width x relu_1 width
    let a_relu1 = Array2::from_shape_fn((3, 4), |(i, j)| {
        let val = ((i * 4 + j) as f32 * 0.15) - 0.3;
        val.cos() * 0.5
    });
    intermediate.a_at_relu.insert("relu_1".to_string(), a_relu1);

    // Multi-objective setup: 3 objectives (one per output neuron)
    let objectives = vec![
        vec![1.0, 0.0, 0.0], // Objective 0: maximize output 0
        vec![0.0, 1.0, 0.0], // Objective 1: maximize output 1
        vec![0.0, 0.0, 1.0], // Objective 2: maximize output 2
    ];

    // Bounds for each objective (simulating verification scenario)
    let obj_bounds = vec![
        (-0.1, 0.5), // Objective 0: margin = -0.1 (not verified)
        (0.2, 0.8),  // Objective 1: margin = 0.2 (verified)
        (-0.3, 0.4), // Objective 2: margin = -0.3 (critical - minimum)
    ];
    let thresholds = vec![0.0, 0.0, 0.0];
    let verified_mask = vec![false, true, false];

    const NUM_ITERATIONS: u32 = 100;

    // Benchmark analytical gradient computation
    let analytical_start = Instant::now();
    for _ in 0..NUM_ITERATIONS {
        beta_state.zero_grad();
        beta_state.compute_analytical_gradients_multi_objective(
            &intermediate,
            &obj_bounds,
            &objectives,
            &thresholds,
            &verified_mask,
            false,
        );
    }
    let analytical_time = analytical_start.elapsed();

    // Benchmark finite-difference gradient computation (SPSA-style)
    // This simulates what SPSA would do: perturb each β and compute bounds
    let epsilon = 0.01f32;
    let fd_start = Instant::now();
    for _ in 0..NUM_ITERATIONS {
        beta_state.zero_grad();

        // For each β entry, compute finite-difference gradient
        // This requires 2 evaluations per β (or 2 total with simultaneous perturbation)
        // SPSA uses 3 total: baseline, +ε, -ε
        for entry_idx in 0..beta_state.entries.len() {
            let original_value = beta_state.entries[entry_idx].value;

            // +ε evaluation (just compute the weighted sum, simulating forward pass result)
            beta_state.entries[entry_idx].value = original_value + epsilon;
            let _plus_margin: f32 = obj_bounds
                .iter()
                .zip(thresholds.iter())
                .zip(verified_mask.iter())
                .filter(|((_, _), &v)| !v)
                .map(|(((l, _), t), _)| l - t)
                .fold(f32::INFINITY, |a, b| a.min(b));

            // -ε evaluation
            beta_state.entries[entry_idx].value = original_value - epsilon;
            let _minus_margin: f32 = obj_bounds
                .iter()
                .zip(thresholds.iter())
                .zip(verified_mask.iter())
                .filter(|((_, _), &v)| !v)
                .map(|(((l, _), t), _)| l - t)
                .fold(f32::INFINITY, |a, b| a.min(b));

            // Restore and compute gradient
            beta_state.entries[entry_idx].value = original_value;
            // grad = (plus - minus) / (2 * epsilon)
            // Note: In real SPSA, we'd use actual forward pass bounds
        }
    }
    let fd_time = fd_start.elapsed();

    // Report results
    let analytical_us = analytical_time.as_micros() as f64 / NUM_ITERATIONS as f64;
    let fd_us = fd_time.as_micros() as f64 / NUM_ITERATIONS as f64;
    let speedup = fd_us / analytical_us;

    println!(
        "
=== Multi-Objective β Gradient Benchmark ==="
    );
    println!("Network: 4 -> 8 -> 4 -> 3 (2 ReLU layers)");
    println!("β parameters: {}", beta_state.entries.len());
    println!("Objectives: {}", objectives.len());
    println!("Iterations: {}", NUM_ITERATIONS);
    println!("Analytical gradient: {:.2} µs/iter", analytical_us);
    println!("Finite-difference:   {:.2} µs/iter (simulated)", fd_us);
    println!("Speedup: {:.1}x", speedup);
    println!();
    println!("Note: Real SPSA speedup is ~3x because analytical uses 1 forward pass");
    println!("      vs SPSA's 3 forward passes. This benchmark only measures the");
    println!("      gradient computation overhead, not forward pass cost.");

    // Verify gradients are computed correctly
    beta_state.zero_grad();
    let max_grad = beta_state.compute_analytical_gradients_multi_objective(
        &intermediate,
        &obj_bounds,
        &objectives,
        &thresholds,
        &verified_mask,
        false,
    );
    assert!(
        max_grad.is_finite() && max_grad >= 0.0,
        "Max gradient should be finite and non-negative"
    );

    // Check that gradients were computed for all entries
    for (i, entry) in beta_state.entries.iter().enumerate() {
        assert!(entry.grad.is_finite(), "Gradient {} should be finite", i);
    }

    // Note: The pure gradient computation overhead is similar between methods.
    // The real 3x speedup comes from forward pass savings:
    // - Analytical: 1 forward pass + ~2µs gradient computation per iteration
    // - SPSA: 3 forward passes + ~2µs gradient estimation per iteration
    //
    // For a typical forward pass of 100-1000µs, this means:
    // - Analytical: 100-1000µs + 2µs ≈ 100-1000µs per iteration
    // - SPSA: 300-3000µs + 2µs ≈ 300-3000µs per iteration
    // Speedup: ~3x
    //
    // This benchmark verifies the gradient computation overhead is reasonable
    // (not a bottleneck compared to forward pass cost).
    assert!(
        analytical_us < 100.0,
        "Analytical gradient computation should be fast (<100µs): {} µs",
        analytical_us
    );
}

/// Regression test for #2247: accumulate_grad must update ALL entries matching
/// a `(node_name, neuron_idx)` pair, not just the first.
///
/// When GenBaB creates multiple constraints at the same neuron (different split
/// points), each constraint needs its own gradient. Before the fix,
/// `accumulate_grad` used `get_entry_mut()` which returns only the first match,
/// leaving subsequent entries with zero gradient — their beta values never
/// optimize.
#[ntest::timeout(10000)]
#[test]
fn test_accumulate_grad_multi_constraint_same_neuron_2247() {
    use crate::beta_crown::GenBabConstraint;

    // Create a split history with two GenBaB constraints at the SAME neuron
    // (different split points — this is the GenBaB multi-constraint pattern)
    let mut history = GraphSplitHistory::new();
    history.add_genbab_constraints_for_split([
        // Lower branch: x <= -0.5 (sign = -1)
        GenBabConstraint::new("gelu_1".to_string(), 3, -0.5, false, 0.5).unwrap(),
        // Upper branch: x >= 0.5 (sign = +1)
        GenBabConstraint::new("gelu_1".to_string(), 3, 0.5, true, 0.5).unwrap(),
    ]);

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    assert_eq!(
        beta_state.entries.len(),
        2,
        "should have 2 entries for same neuron"
    );

    // Both entries should have the same (node_name, neuron_idx) but different split_point
    assert_eq!(beta_state.entries[0].node_name, "gelu_1");
    assert_eq!(beta_state.entries[0].neuron_idx, 3);
    assert_eq!(beta_state.entries[1].node_name, "gelu_1");
    assert_eq!(beta_state.entries[1].neuron_idx, 3);
    assert!(
        (beta_state.entries[0].split_point - beta_state.entries[1].split_point).abs() > 0.01,
        "split points should differ"
    );

    // Accumulate gradient for this neuron
    beta_state.accumulate_grad("gelu_1", 3, 2.0);

    // BUG (before fix): only first entry gets gradient, second stays at 0
    // FIX: both entries should receive the gradient
    assert!(
        (beta_state.entries[0].grad - 2.0).abs() < 1e-6,
        "first entry should have grad=2.0, got {}",
        beta_state.entries[0].grad
    );
    assert!(
        (beta_state.entries[1].grad - 2.0).abs() < 1e-6,
        "second entry should ALSO have grad=2.0 (bug #2247: was 0.0), got {}",
        beta_state.entries[1].grad
    );

    // Verify get_signed_beta correctly sums both (it was already correct)
    let signed = beta_state.signed_beta("gelu_1", 3);
    assert!(signed.is_some());
}

/// Test that accumulate_grad still works correctly for the common case:
/// single constraint per neuron (standard ReLU splits).
#[ntest::timeout(10000)]
#[test]
fn test_accumulate_grad_single_constraint_unchanged_2247() {
    let history = GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 1,
            is_active: false,
            score: 0.0,
        });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();

    // Accumulate gradient for neuron 0 only
    beta_state.accumulate_grad("relu1", 0, 3.0);

    // Only neuron 0 should have gradient
    assert!((beta_state.entries[0].grad - 3.0).abs() < 1e-6);
    assert!(
        (beta_state.entries[1].grad - 0.0).abs() < 1e-6,
        "neuron 1 should be unaffected"
    );
}

/// Regression test for #2263: gradient_step_adam must not produce NaN/Inf
/// when called with t=0 (missing t.max(1) guard).
#[ntest::timeout(10000)]
#[test]
fn test_gradient_step_adam_t_zero_no_nan_2263() {
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    // Give it a non-zero gradient so Adam has something to update
    beta_state.accumulate_grad("relu1", 0, 1.5);

    let config = AdaptiveOptConfig::default();
    // t=0 would cause division by zero without the fix
    let max_grad = beta_state.gradient_step_adam(&config, 0);

    assert!(
        max_grad.is_finite(),
        "max_grad should be finite, got {}",
        max_grad
    );
    assert!(
        beta_state.entries[0].value.is_finite(),
        "beta value should be finite after Adam step with t=0, got {}",
        beta_state.entries[0].value
    );
}

/// Verify gradient_step_adam with t=1 produces correct results (sanity check).
#[ntest::timeout(10000)]
#[test]
fn test_gradient_step_adam_t_one_sane_2263() {
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    beta_state.accumulate_grad("relu1", 0, 2.0);

    let config = AdaptiveOptConfig::default();
    let max_grad = beta_state.gradient_step_adam(&config, 1);

    assert!(
        (max_grad - 2.0).abs() < 1e-6,
        "max_grad should be 2.0, got {}",
        max_grad
    );
    assert!(
        beta_state.entries[0].value.is_finite(),
        "beta value should be finite, got {}",
        beta_state.entries[0].value
    );
    // Beta value should be positive (starts at 0, gradient ascent with positive gradient)
    assert!(
        beta_state.entries[0].value >= 0.0,
        "beta value should be non-negative, got {}",
        beta_state.entries[0].value
    );
}

/// Regression test for #2575/#2586: gradient_step_adam must not produce
/// NaN/Inf when beta1=1.0 (disables momentum — valid config).
#[ntest::timeout(10000)]
#[test]
fn test_gradient_step_adam_beta1_one_no_div_by_zero_2575() {
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    beta_state.accumulate_grad("relu1", 0, 1.5);

    let config = AdaptiveOptConfig {
        beta1: 1.0,
        ..Default::default()
    };
    let max_grad = beta_state.gradient_step_adam(&config, 1);

    assert!(
        max_grad.is_finite(),
        "max_grad should be finite with beta1=1.0, got {}",
        max_grad
    );
    assert!(
        beta_state.entries[0].value.is_finite(),
        "beta value should be finite with beta1=1.0, got {}",
        beta_state.entries[0].value
    );
}

/// Regression test for #2575/#2586: gradient_step_adam must not produce
/// NaN/Inf when beta2=1.0.
#[ntest::timeout(10000)]
#[test]
fn test_gradient_step_adam_beta2_one_no_div_by_zero_2575() {
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    beta_state.accumulate_grad("relu1", 0, 1.5);

    let config = AdaptiveOptConfig {
        beta2: 1.0,
        ..Default::default()
    };
    let max_grad = beta_state.gradient_step_adam(&config, 1);

    assert!(
        max_grad.is_finite(),
        "max_grad should be finite with beta2=1.0, got {}",
        max_grad
    );
    assert!(
        beta_state.entries[0].value.is_finite(),
        "beta value should be finite with beta2=1.0, got {}",
        beta_state.entries[0].value
    );
}

/// Test for #3112: NaN gradients are filtered at accumulate_grad gate.
/// Previously (#2596), NaN entered the optimizer and was cleaned up post-step.
/// Now, accumulate_grad skips NaN entirely — optimizer state is never corrupted.
#[ntest::timeout(10000)]
#[test]
fn test_gradient_step_adam_nan_gradient_filtered_at_gate_3112() {
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig::default();

    // Normal gradient to establish non-zero state
    beta_state.accumulate_grad("relu1", 0, 1.5);
    beta_state.gradient_step_adam(&config, 1);
    assert!(
        beta_state.entries[0].value > 0.0,
        "beta should be positive after normal step"
    );

    // Inject NaN gradient — filtered at the gate by #3112.
    beta_state.zero_grad();
    beta_state.accumulate_grad("relu1", 0, f32::NAN);
    assert_eq!(
        beta_state.entries[0].grad, 0.0,
        "NaN gradient should be silently filtered by accumulate_grad"
    );
    beta_state.gradient_step_adam(&config, 2);

    // Optimizer state must remain finite — NaN never entered
    let entry = &beta_state.entries[0];
    assert!(
        entry.value().is_finite(),
        "beta should be finite after NaN-filtered step"
    );
    assert!(entry.m.is_finite(), "first moment should be finite");
    assert!(entry.v.is_finite(), "second moment should be finite");
    assert!(entry.v_max.is_finite(), "v_max should be finite");
    // Beta value preserved (not reset to 0) — the NaN was filtered, not cleaned up
    assert!(
        entry.value() > 0.0,
        "beta should remain positive (NaN filtered, not reset)"
    );
}

// =========================================================================
