// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for parallel bound propagation.

use super::*;
use crate::layers::{Layer, LinearLayer, ReLULayer};
use crate::network::{GraphNetwork, GraphNode};
use ndarray::{arr2, arr3, ArrayD, IxDyn};
use ny_test_utils::CountingGemmEngine;

fn create_simple_graph() -> GraphNetwork {
    // Simple 2-layer MLP: Linear -> ReLU -> Linear
    let mut graph = GraphNetwork::new();

    // Input layer (3 -> 4)
    let weight1 = arr2(&[
        [1.0_f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]);
    let linear1 = LinearLayer::new(weight1, None).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    // ReLU
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Output layer (4 -> 2)
    let weight2 = arr2(&[[1.0_f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]);
    let linear2 = LinearLayer::new(weight2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));

    graph.set_output("linear2");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_parallel_verifier_basic() {
    let graph = create_simple_graph();

    // Input: [batch=1, seq=8, hidden=3]
    let lower = arr3(&[[
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
        [0.5, 0.5, 0.5],
        [0.6, 0.6, 0.6],
        [0.7, 0.7, 0.7],
    ]])
    .into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 2, // Use parallel for 8 positions
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    let result = verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();

    // Check output shape: [batch=1, seq=8, output=2]
    assert_eq!(result.output_bounds.shape(), &[1, 8, 2]);
    assert_eq!(result.num_positions, 8);
    assert_eq!(result.parallel_positions, 8);
}

#[ntest::timeout(10000)]
#[test]
fn test_parallel_vs_serial_equivalence() {
    let graph = create_simple_graph();

    // Input: [batch=1, seq=4, hidden=3]
    let lower = arr3(&[[
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
    ]])
    .into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Serial verification (high threshold forces serial)
    let serial_config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 100,
        ..Default::default()
    };
    let serial_verifier = ParallelVerifier::new(serial_config);
    let serial_result = serial_verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();

    // Parallel verification
    let parallel_config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 1,
        ..Default::default()
    };
    let parallel_verifier = ParallelVerifier::new(parallel_config);
    let parallel_result = parallel_verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();

    // Results should be identical
    assert_eq!(
        serial_result.output_bounds.shape(),
        parallel_result.output_bounds.shape()
    );

    let serial_bounds = &serial_result.output_bounds;
    let parallel_bounds = &parallel_result.output_bounds;

    for (s, p) in serial_bounds
        .lower()
        .iter()
        .zip(parallel_bounds.lower().iter())
    {
        assert!((s - p).abs() < 1e-6, "Lower bounds differ: {} vs {}", s, p);
    }
    for (s, p) in serial_bounds
        .upper()
        .iter()
        .zip(parallel_bounds.upper().iter())
    {
        assert!((s - p).abs() < 1e-6, "Upper bounds differ: {} vs {}", s, p);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_convenience_function() {
    let graph = create_simple_graph();

    let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = verify_parallel(&graph, &input, 1).unwrap();
    assert_eq!(output.shape(), &[1, 2, 2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_batch_parallel() {
    let graph = create_simple_graph();

    // Input: [batch=4, hidden=3] - parallelize over batch
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[4, 3]),
        vec![0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
    )
    .unwrap();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 1,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    let result = verifier.verify_batch_parallel(&graph, &input, 0).unwrap();

    // Output: [batch=4, output=2]
    assert_eq!(result.output_bounds.shape(), &[4, 2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_axis_out_of_bounds() {
    let graph = create_simple_graph();

    let lower = arr3(&[[[0.0, 0.0, 0.0]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let verifier = ParallelVerifier::new(ParallelConfig::default());
    let result = verifier.verify_positions_parallel(&graph, &input, 10);

    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_sdp_crown_parallel_rejects_uniform_epsilon_box() {
    // Simple ReLU MLP with a uniform-epsilon ℓ∞ box input. Even in this
    // best case, SDP-CROWN must refuse: its bounds are valid only over an
    // ℓ2 ball, which no ℓ∞ box input can soundly be converted to (the ball
    // of the box's half-width misses the box corners; the ball containing
    // the box is no tighter than CROWN).
    let graph = create_simple_graph();

    // Shape: [batch=1, seq=4, hidden=3]
    let epsilon = 0.05_f32;
    let center = arr3(&[[
        [0.5, 0.5, 0.5],
        [0.6, 0.6, 0.6],
        [0.7, 0.7, 0.7],
        [0.8, 0.8, 0.8],
    ]])
    .into_dyn();
    let lower = center.mapv(|x| x - epsilon);
    let upper = center.mapv(|x| x + epsilon);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::SdpCrown,
        min_positions_for_parallel: 2,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);
    let result = verifier.verify_positions_parallel(&graph, &input, 1);

    assert!(
        matches!(result, Err(NyError::UnsupportedOp(_))),
        "SDP-CROWN must reject ℓ∞ box inputs, got {result:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sdp_crown_parallel_rejects_non_uniform_epsilon_box() {
    let graph = create_simple_graph();

    // Input with NON-uniform epsilon (different half-widths per dimension)
    // Shape: [batch=1, seq=2, hidden=3]
    // First position: epsilon=0.1, second position: epsilon=0.2
    let lower = arr3(&[[[0.4, 0.4, 0.4], [0.3, 0.3, 0.3]]]).into_dyn();
    let upper = arr3(&[
        [[0.6, 0.6, 0.6], [0.7, 0.7, 0.7]], // Different epsilon per position
    ])
    .into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::SdpCrown,
        min_positions_for_parallel: 1,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);
    let result = verifier.verify_positions_parallel(&graph, &input, 1);

    assert!(
        matches!(result, Err(NyError::UnsupportedOp(_))),
        "SDP-CROWN must reject ℓ∞ box inputs, got {result:?}"
    );
}

// ============== ParallelConfig Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_parallel_config_default() {
    let config = ParallelConfig::default();
    assert!(matches!(config.method, PropagationMethod::Ibp));
    assert_eq!(config.min_positions_for_parallel, 4);
    assert!(config.max_threads.is_none());
    assert!(!config.report_progress);
}

#[ntest::timeout(10000)]
#[test]
fn test_parallel_config_custom() {
    let config = ParallelConfig {
        method: PropagationMethod::Crown,
        min_positions_for_parallel: 10,
        max_threads: Some(4),
        report_progress: true,
    };

    assert!(matches!(config.method, PropagationMethod::Crown));
    assert_eq!(config.min_positions_for_parallel, 10);
    assert_eq!(config.max_threads, Some(4));
    assert!(config.report_progress);
}

// ============== ParallelVerificationResult Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_parallel_verification_result_fields() {
    let graph = create_simple_graph();

    let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 1,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    let result = verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();

    // Check all fields are populated correctly
    assert_eq!(result.num_positions, 3);
    assert_eq!(result.parallel_positions, 3);
    assert!(result.avg_position_time_ms >= 0.0);
    assert_eq!(result.output_bounds.shape(), &[1, 3, 2]);
}

// ============== verify_parallel_with_method Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_verify_parallel_with_method_ibp() {
    let graph = create_simple_graph();

    let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = verify_parallel_with_method(&graph, &input, 1, PropagationMethod::Ibp).unwrap();
    assert_eq!(output.shape(), &[1, 2, 2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_parallel_with_method_crown() {
    let graph = create_simple_graph();

    let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = verify_parallel_with_method(&graph, &input, 1, PropagationMethod::Crown).unwrap();
    assert_eq!(output.shape(), &[1, 2, 2]);
}

// ============== Serial Fallback Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_parallel_serial_fallback_for_few_positions() {
    let graph = create_simple_graph();

    // Only 2 positions, with threshold of 4 -> should use serial
    let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 4, // Higher than 2 positions
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    let result = verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();

    // Should use serial (parallel_positions = 0)
    assert_eq!(result.num_positions, 2);
    assert_eq!(result.parallel_positions, 0);
    assert_eq!(result.output_bounds.shape(), &[1, 2, 2]);
}

// ============== Max Threads Configuration Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_parallel_with_max_threads() {
    let graph = create_simple_graph();

    let lower = arr3(&[[
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
    ]])
    .into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 1,
        max_threads: Some(2), // Limit to 2 threads
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    let result = verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();

    // Should still work correctly with limited threads
    assert_eq!(result.output_bounds.shape(), &[1, 4, 2]);
    assert_eq!(result.num_positions, 4);
}

// ============== AlphaCrown and BetaCrown Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_parallel_alpha_crown_method() {
    let graph = create_simple_graph();

    let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::AlphaCrown,
        min_positions_for_parallel: 1,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    // Should work (falls back to CROWN or IBP internally)
    let result = verifier.verify_positions_parallel(&graph, &input, 1);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().output_bounds.shape(), &[1, 2, 2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_parallel_beta_crown_method() {
    let graph = create_simple_graph();

    let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::BetaCrown,
        min_positions_for_parallel: 1,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    // Should work (falls back to CROWN or IBP internally)
    let result = verifier.verify_positions_parallel(&graph, &input, 1);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().output_bounds.shape(), &[1, 2, 2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_parallel_verifier_with_stored_engine_threads_crown() {
    let graph = create_simple_graph();
    let lower = arr3(&[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]).into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();
    let engine = Arc::new(CountingGemmEngine::new());

    let verifier = ParallelVerifier::new_with_engine(
        ParallelConfig {
            method: PropagationMethod::Crown,
            min_positions_for_parallel: 1,
            ..Default::default()
        },
        engine.clone(),
    );

    let result = verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();

    assert_eq!(result.output_bounds.shape(), &[1, 2, 2]);
    assert!(
        engine.gemm_calls() > 0,
        "ParallelVerifier::new_with_engine should thread the GemmEngine into CROWN verification"
    );
}

// ============== Progress Reporting Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_parallel_with_progress_reporting() {
    let graph = create_simple_graph();

    let lower = arr3(&[[
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
    ]])
    .into_dyn();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 1,
        report_progress: true, // Enable progress reporting
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    // Should work with progress reporting enabled
    let result = verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();
    assert_eq!(result.output_bounds.shape(), &[1, 5, 2]);
}

// ============== Multidimensional Batch Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_parallel_over_different_axes() {
    let graph = create_simple_graph();

    // Input: [batch=2, seq=3, hidden=3]
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 3, 3]),
        vec![
            0.0, 0.0, 0.0, // batch 0, seq 0
            0.1, 0.1, 0.1, // batch 0, seq 1
            0.2, 0.2, 0.2, // batch 0, seq 2
            0.3, 0.3, 0.3, // batch 1, seq 0
            0.4, 0.4, 0.4, // batch 1, seq 1
            0.5, 0.5, 0.5, // batch 1, seq 2
        ],
    )
    .unwrap();
    let upper = lower.mapv(|x| x + 0.1);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = ParallelConfig {
        method: PropagationMethod::Ibp,
        min_positions_for_parallel: 1,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);

    // Parallel over batch axis (0)
    let result_batch = verifier
        .verify_positions_parallel(&graph, &input, 0)
        .unwrap();
    assert_eq!(result_batch.output_bounds.shape(), &[2, 3, 2]);

    // Parallel over seq axis (1)
    let result_seq = verifier
        .verify_positions_parallel(&graph, &input, 1)
        .unwrap();
    assert_eq!(result_seq.output_bounds.shape(), &[2, 3, 2]);
}
