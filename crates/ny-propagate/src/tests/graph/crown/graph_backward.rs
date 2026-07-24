// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph CROWN backward unit tests: targeted coverage for graph-level
//! CROWN backward on DAGs with skip connections.
//!
//! Part of #3463: these topologies were previously tested only through
//! high-level public API wrappers on sequential or FFN-only graphs.
//! This module tests:
//! - Soundness on DAGs with skip connections (residual blocks)
//! - `crown_backward_within_block_with_engine (alpha_state)` via the public API
//! - Spec-guided CROWN with identity/difference specs on DAGs
//! - Partial-CROWN fallback for unsupported layers in block-wise CROWN

use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};

use ny_tensor::BoundedTensor;

use crate::layers::binary_ops::AddLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::normalization::layer_norm::LayerNormLayer;
use crate::layers::normalization::InstanceNorm1dLayer;
use crate::types::BoundsProvenance;
use crate::*;

/// Build a minimal residual DAG: input -> linear1 -> relu -> linear2 + input -> output.
///
/// The skip connection makes this a true DAG (not sequential). CROWN backward
/// must handle the Add node's binary inputs and accumulate bounds from both paths.
fn build_residual_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    // Main path: input -> linear1 -> relu -> linear2
    let w1 = arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]);
    let b1 = arr1(&[0.1_f32, -0.05]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.6_f32, -0.2], [-0.4, 0.7]]);
    let b2 = arr1(&[0.0_f32, 0.0]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));

    // Skip connection: input + linear2 output
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        NETWORK_INPUT,
        "linear2",
    ));
    graph.set_output("residual");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// Evaluate the residual DAG at a concrete point: linear2(relu(linear1(x))) + x.
fn eval_residual_dag(x: &[f32; 2]) -> [f32; 2] {
    let l1 = [
        0.8 * x[0] + (-0.3) * x[1] + 0.1,
        0.4 * x[0] + 0.9 * x[1] + (-0.05),
    ];
    let r = [l1[0].max(0.0), l1[1].max(0.0)];
    let l2 = [0.6 * r[0] + (-0.2) * r[1], (-0.4) * r[0] + 0.7 * r[1]];
    [x[0] + l2[0], x[1] + l2[1]]
}

/// Assert all concrete sample outputs fall within bounds.
fn assert_dag_soundness(bounds: &BoundedTensor, points: &[[f32; 2]]) {
    for point in points {
        let output = eval_residual_dag(point);
        for (i, &value) in output.iter().enumerate() {
            assert!(
                value >= bounds.lower()[[i]] - 1e-5,
                "Soundness: point {:?} output[{}]={:.6} < lower={:.6}",
                point,
                i,
                value,
                bounds.lower()[[i]],
            );
            assert!(
                value <= bounds.upper()[[i]] + 1e-5,
                "Soundness: point {:?} output[{}]={:.6} > upper={:.6}",
                point,
                i,
                value,
                bounds.upper()[[i]],
            );
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
// 1. CROWN backward on a residual DAG: soundness and tightness
// ───────────────────────────────────────────────────────────────────────

/// Test CROWN backward on a residual DAG with skip connection.
/// Verifies soundness via concrete sampling at corners, center, and interior.
///
/// This covers the `GraphNetworkCrownExt` trait's handling of DAG topology --
/// the Add node requires accumulating linear bounds from both the main path
/// and the skip connection, which the existing sequential tests don't exercise.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_residual_dag_soundness() {
    let (graph, input) = build_residual_dag();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // Verify interval ordering.
    for i in 0..2 {
        assert!(
            crown_bounds.lower()[[i]] <= crown_bounds.upper()[[i]] + 1e-6,
            "Inverted CROWN bound at dim {}: [{}, {}]",
            i,
            crown_bounds.lower()[[i]],
            crown_bounds.upper()[[i]],
        );
    }

    let test_points = [
        [-0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [0.0, 0.0],
        [-0.3, 0.2],
        [0.1, -0.4],
    ];
    assert_dag_soundness(&crown_bounds, &test_points);
}

/// Test that CROWN with provenance returns `Crown` (not fallback) on a residual DAG.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_residual_dag_provenance() {
    let (graph, input) = build_residual_dag();
    let result = graph.propagate_crown_with_provenance(&input).unwrap();

    assert_eq!(
        result.provenance,
        BoundsProvenance::Crown,
        "Residual DAG with supported layers should produce Crown provenance, got {:?}",
        result.provenance,
    );
    assert!(!result.is_fallback());
}

/// Test that CROWN on a residual DAG is at least as tight as IBP.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_residual_dag_tighter_than_ibp() {
    let (graph, input) = build_residual_dag();
    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        let crown_width = crown_bounds.upper()[[i]] - crown_bounds.lower()[[i]];
        let ibp_width = ibp_bounds.upper()[[i]] - ibp_bounds.lower()[[i]];
        assert!(
            crown_width <= ibp_width + 1e-4,
            "CROWN should be no wider than IBP at dim {}: CROWN={:.6}, IBP={:.6}",
            i,
            crown_width,
            ibp_width,
        );
    }
}

/// Test empty graph returns input bounds unchanged with Crown provenance.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_empty_graph_returns_input() {
    let graph = GraphNetwork::new();
    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[3.0_f32, 4.0]).into_dyn(),
    )
    .unwrap();

    let result = graph.propagate_crown_with_provenance(&input).unwrap();
    assert_eq!(result.provenance, BoundsProvenance::Crown);
    assert_eq!(result.bounds.lower()[[0]], 1.0);
    assert_eq!(result.bounds.upper()[[1]], 4.0);
}

// ───────────────────────────────────────────────────────────────────────
// 2. Alpha-CROWN via public API plus fixed block-CROWN soundness
// ───────────────────────────────────────────────────────────────────────

/// Build a single FFN block: LayerNorm -> Linear_up -> GELU -> Linear_down -> Add(residual).
fn build_single_block_ffn(hidden: usize, expansion: usize) -> GraphNetwork {
    let scale = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let mut graph = GraphNetwork::new();

    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
    graph.add_node(GraphNode::new(
        "layer0_norm",
        Layer::LayerNorm(ln),
        vec![NETWORK_INPUT.to_string()],
    ));

    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        scale * ((i * 17 + j * 31) as f32).sin() * 0.15
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();
    graph.add_node(GraphNode::new(
        "layer0_ffn_up",
        Layer::Linear(linear_up),
        vec!["layer0_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer0_ffn_act",
        Layer::GELU(GELULayer::default()),
        vec!["layer0_ffn_up".to_string()],
    ));

    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        scale * ((i * 23 + j * 37) as f32).cos() * 0.15
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();
    graph.add_node(GraphNode::new(
        "layer0_ffn_down",
        Layer::Linear(linear_down),
        vec!["layer0_ffn_act".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer0_add",
        Layer::Add(AddLayer),
        vec![NETWORK_INPUT.to_string(), "layer0_ffn_down".to_string()],
    ));
    graph.set_output("layer0_add");
    graph
}

/// Helper: generate concrete sample points within an epsilon-ball.
fn generate_sample_points(hidden: usize, epsilon: f32, n_random: usize) -> Vec<ArrayD<f32>> {
    let mut points = Vec::new();
    // All 2^hidden corners.
    for corner_idx in 0..(1_usize << hidden) {
        let v: Vec<f32> = (0..hidden)
            .map(|d| {
                if corner_idx & (1 << d) != 0 {
                    epsilon
                } else {
                    -epsilon
                }
            })
            .collect();
        points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), v).unwrap());
    }
    // Center.
    points.push(ArrayD::zeros(IxDyn(&[hidden])));
    // Deterministic pseudo-random.
    for s in 0..n_random {
        let v: Vec<f32> = (0..hidden)
            .map(|d| {
                let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                (hash * 2.0 - 1.0) * epsilon
            })
            .collect();
        points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), v).unwrap());
    }
    points
}

/// Test alpha-CROWN via public API: at midpoint, alpha-CROWN should produce
/// bounds at least as tight as fixed CROWN. This exercises
/// `crown_backward_within_block_with_engine (alpha_state)` internally.
#[ntest::timeout(60000)]
#[test]
fn test_alpha_crown_at_least_as_tight_as_fixed_on_ffn_block() {
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let epsilon = 0.01_f32;
        let graph = build_single_block_ffn(hidden, 2);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();

        let fixed = graph.propagate_crown_block_wise(&input, epsilon).unwrap();
        let alpha = graph
            .propagate_alpha_crown_block_wise(&input, epsilon)
            .unwrap();

        assert_eq!(fixed.total_blocks, alpha.total_blocks);
        for (fb, ab) in fixed.blocks.iter().zip(alpha.blocks.iter()) {
            assert!(
                ab.crown_successful,
                "Alpha-CROWN failed for {}",
                ab.block_name
            );
            if let Some(aw) = ab.alpha_crown_max_width {
                assert!(
                    aw <= fb.crown_max_width + 1e-6,
                    "Alpha wider than fixed for {}: alpha={:.6} > fixed={:.6}",
                    ab.block_name,
                    aw,
                    fb.crown_max_width,
                );
                assert!(
                    aw.is_finite(),
                    "Non-finite alpha width for {}",
                    ab.block_name
                );
            }
        }
    });
}

/// Test fixed block-CROWN soundness on the same FFN block topology.
///
/// The alpha-path coverage for `crown_backward_within_block_with_engine (alpha_state)` lives in
/// `test_alpha_crown_at_least_as_tight_as_fixed_on_ffn_block`. This test calls
/// `crown_backward_within_block` directly to verify sampled concrete
/// evaluations stay within the fixed block-CROWN bounds.
#[ntest::timeout(60000)]
#[test]
fn test_fixed_block_crown_soundness_on_ffn() {
    let hidden = 4;
    let epsilon = 0.05_f32;
    let graph = build_single_block_ffn(hidden, 2);

    let exec_order = graph.exec_order().unwrap();
    let block_nodes_map = GraphNetwork::collect_block_nodes(exec_order);

    for nodes_in_block in block_nodes_map.values() {
        let block_input =
            BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();
        let block_node_bounds = graph
            .collect_block_ibp_bounds(nodes_in_block, &block_input)
            .unwrap();

        // Use fixed CROWN (which alpha-CROWN relaxes to be at least as tight).
        let (crown_bounds, _, provenance) = graph
            .crown_backward_within_block(nodes_in_block, &block_node_bounds, &block_input)
            .unwrap();

        // Happy-path CROWN must report Crown provenance, not ForwardFallback.
        // Regression guard: silent fallback would widen bounds without warning (#4256).
        assert_eq!(
            provenance,
            BoundsProvenance::Crown,
            "block-wise CROWN on identity residual DAG must not fall back to forward bounds"
        );

        let sample_points = generate_sample_points(hidden, epsilon, 30);
        let last_node = nodes_in_block.last().unwrap();

        for (idx, point) in sample_points.iter().enumerate() {
            let point_bt = BoundedTensor::new(point.clone(), point.clone()).unwrap();
            let concrete = graph
                .collect_block_ibp_bounds(nodes_in_block, &point_bt)
                .unwrap();
            let vals = concrete.get(last_node).unwrap().lower();

            for d in 0..crown_bounds.lower().len() {
                assert!(
                    vals[[d]] >= crown_bounds.lower()[[d]] - 1e-6
                        && vals[[d]] <= crown_bounds.upper()[[d]] + 1e-6,
                    "Block CROWN soundness: sample {} dim {}: val={:.8} \
                     not in [{:.8}, {:.8}]",
                    idx,
                    d,
                    vals[[d]],
                    crown_bounds.lower()[[d]],
                    crown_bounds.upper()[[d]],
                );
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
// 3. Spec-guided CROWN on DAG: identity spec matches non-spec
// ───────────────────────────────────────────────────────────────────────

/// Test that spec-guided CROWN with identity spec_matrix produces bounds
/// matching fixed-slope CROWN on a residual DAG.
///
/// Uses `propagate_crown_fixed_slope` (not `propagate_crown`) because the
/// public `propagate_crown` routes through DAG alpha-CROWN which optimizes
/// ReLU slopes. Spec-guided CROWN uses fixed heuristic slopes, so the
/// correct comparison target is fixed-slope CROWN. (#3619 alpha-CROWN)
#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_identity_matches_non_spec_on_dag() {
    let (graph, input) = build_residual_dag();
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    let identity_spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let spec_bounds = graph
        .propagate_crown_with_specs_and_engine(&input, &identity_spec, None)
        .unwrap();

    for i in 0..2 {
        assert!(
            (crown_bounds.lower()[[i]] - spec_bounds.lower()[[i]]).abs() < 1e-4,
            "Spec lower[{}] mismatch: crown={:.6} spec={:.6}",
            i,
            crown_bounds.lower()[[i]],
            spec_bounds.lower()[[i]],
        );
        assert!(
            (crown_bounds.upper()[[i]] - spec_bounds.upper()[[i]]).abs() < 1e-4,
            "Spec upper[{}] mismatch: crown={:.6} spec={:.6}",
            i,
            crown_bounds.upper()[[i]],
            spec_bounds.upper()[[i]],
        );
    }
}

/// Test spec-guided CROWN with difference spec on a residual DAG.
/// The difference spec [[1, -1]] bounds output[0] - output[1].
/// This must be tighter-or-equal to post-hoc interval arithmetic and sound.
///
/// Uses `propagate_crown_fixed_slope` for the post-hoc baseline because the
/// public `propagate_crown` routes through DAG alpha-CROWN with optimized
/// slopes — alpha-CROWN's tighter per-output bounds make the post-hoc
/// difference tighter than what fixed-slope spec-guided CROWN can achieve,
/// invalidating the tightness comparison. (#3619 alpha-CROWN)
#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_difference_tighter_than_posthoc_on_dag() {
    let (graph, input) = build_residual_dag();
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    // Post-hoc interval arithmetic on Y_0 - Y_1.
    let posthoc_lower = crown_bounds.lower()[[0]] - crown_bounds.upper()[[1]];
    let posthoc_upper = crown_bounds.upper()[[0]] - crown_bounds.lower()[[1]];

    // Spec-guided CROWN: direct bounds on Y_0 - Y_1.
    let diff_spec = arr2(&[[1.0_f32, -1.0]]);
    let spec_bounds = graph
        .propagate_crown_with_specs_and_engine(&input, &diff_spec, None)
        .unwrap();
    let spec_lo = spec_bounds.lower()[[0]];
    let spec_hi = spec_bounds.upper()[[0]];

    assert!(
        spec_lo >= posthoc_lower - 1e-4,
        "Spec lower ({:.6}) < post-hoc ({:.6})",
        spec_lo,
        posthoc_lower
    );
    assert!(
        spec_hi <= posthoc_upper + 1e-4,
        "Spec upper ({:.6}) > post-hoc ({:.6})",
        spec_hi,
        posthoc_upper
    );

    // Soundness: sample concrete differences.
    for point in &[
        [-0.5_f32, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [0.0, 0.0],
    ] {
        let out = eval_residual_dag(point);
        let diff = out[0] - out[1];
        assert!(
            diff >= spec_lo - 1e-5 && diff <= spec_hi + 1e-5,
            "Spec soundness: {:?} diff={:.6} not in [{:.6}, {:.6}]",
            point,
            diff,
            spec_lo,
            spec_hi,
        );
    }
}

// ───────────────────────────────────────────────────────────────────────
// 3b. Diagnostic: compare spec vs standard CROWN linear bounds
// ───────────────────────────────────────────────────────────────────────

// ───────────────────────────────────────────────────────────────────────
// 4. Decomposed InstanceNorm CROWN within a block
// ───────────────────────────────────────────────────────────────────────
// Part of #3830: InstanceNorm1d now dispatches through the decomposed
// primitive-chain backward path inside `crown_backward_within_block`, preserving
// soundness without falling back to bias-only IBP.

/// Build a single block with InstanceNorm1d using decomposed block-wise CROWN.
///
/// Block structure:
///   block_input [C, T] -> layer0_instnorm -> layer0_relu -> layer0_add
///                          (InstanceNorm1d)    (ReLU)         (Add with skip)
///                                                              ↑
///                                                         block_input (skip)
///
/// CROWN backward dispatches InstanceNorm1d through the decomposed helper,
/// then continues through ReLU and residual Add in the same block.
fn build_instnorm_block(num_channels: usize) -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let instnorm = InstanceNorm1dLayer::new_default(num_channels, 1e-5).unwrap();
    graph.add_node(GraphNode::new(
        "layer0_instnorm",
        Layer::InstanceNorm1d(instnorm),
        vec![NETWORK_INPUT.to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer0_relu",
        Layer::ReLU(ReLULayer),
        vec!["layer0_instnorm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "layer0_add",
        Layer::Add(AddLayer),
        vec![NETWORK_INPUT.to_string(), "layer0_relu".to_string()],
    ));
    graph.set_output("layer0_add");
    graph
}

/// Evaluate the InstanceNorm block at a concrete point.
///
/// output = x + relu(instnorm(x)) where instnorm normalizes per-channel
/// with ny=1, beta=0: y[c,t] = (x[c,t] - mean_c) / sqrt(var_c + eps).
fn eval_instnorm_block(x: &[f32], num_channels: usize, time_len: usize) -> Vec<f32> {
    let eps = 1e-5_f32;
    let mut instnorm_out = vec![0.0_f32; x.len()];

    for c in 0..num_channels {
        let offset = c * time_len;
        let channel_slice = &x[offset..offset + time_len];

        let mean: f32 = channel_slice.iter().sum::<f32>() / time_len as f32;
        let var: f32 = channel_slice
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            / time_len as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for t in 0..time_len {
            instnorm_out[offset + t] = (channel_slice[t] - mean) * inv_std;
        }
    }

    // relu(instnorm(x)) + x
    x.iter()
        .zip(instnorm_out.iter())
        .map(|(&xi, &ni)| xi + ni.max(0.0))
        .collect()
}

/// Generate deterministic sample points in [-epsilon, epsilon] for shape [C, T].
fn generate_2d_sample_points(
    num_channels: usize,
    time_len: usize,
    epsilon: f32,
    n_random: usize,
) -> Vec<Vec<f32>> {
    let dim = num_channels * time_len;
    let mut points = Vec::new();

    // Center point.
    points.push(vec![0.0_f32; dim]);

    // Axis-aligned extremes (positive/negative along each dimension).
    for d in 0..dim {
        let mut pos = vec![0.0_f32; dim];
        pos[d] = epsilon;
        points.push(pos);
        let mut neg = vec![0.0_f32; dim];
        neg[d] = -epsilon;
        points.push(neg);
    }

    // All-positive, all-negative corners.
    points.push(vec![epsilon; dim]);
    points.push(vec![-epsilon; dim]);

    // Mixed corners: alternate signs.
    let alt: Vec<f32> = (0..dim)
        .map(|d| if d % 2 == 0 { epsilon } else { -epsilon })
        .collect();
    points.push(alt);
    let alt_inv: Vec<f32> = (0..dim)
        .map(|d| if d % 2 == 0 { -epsilon } else { epsilon })
        .collect();
    points.push(alt_inv);

    // Deterministic pseudo-random.
    for s in 0..n_random {
        let v: Vec<f32> = (0..dim)
            .map(|d| {
                let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                (hash * 2.0 - 1.0) * epsilon
            })
            .collect();
        points.push(v);
    }

    points
}

fn assert_finite_ordered_bounds(bounds: &BoundedTensor) {
    assert!(
        bounds.lower().iter().all(|v| v.is_finite()),
        "CROWN lower bounds contain non-finite: {:?}",
        bounds.lower()
    );
    assert!(
        bounds.upper().iter().all(|v| v.is_finite()),
        "CROWN upper bounds contain non-finite: {:?}",
        bounds.upper()
    );

    for (lower, upper) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(
            lower <= upper,
            "Inverted CROWN bound: lower={:.6} > upper={:.6}",
            lower,
            upper
        );
    }
}

fn assert_instnorm_block_soundness(
    bounds: &BoundedTensor,
    sample_points: &[Vec<f32>],
    num_channels: usize,
    time_len: usize,
) {
    let dim = num_channels * time_len;
    let lower = bounds.lower().as_slice().unwrap();
    let upper = bounds.upper().as_slice().unwrap();

    for (sample_idx, point) in sample_points.iter().enumerate() {
        let output = eval_instnorm_block(point, num_channels, time_len);
        assert_eq!(output.len(), dim);

        for (dim_idx, &value) in output.iter().enumerate().take(dim) {
            assert!(
                value >= lower[dim_idx] - 1e-4 && value <= upper[dim_idx] + 1e-4,
                "Soundness failure: sample {} dim {}: val={:.6} not in [{:.6}, {:.6}]",
                sample_idx,
                dim_idx,
                value,
                lower[dim_idx],
                upper[dim_idx],
            );
        }
    }
}

/// Test decomposed InstanceNorm CROWN with InstanceNorm1d in a block.
///
/// Exercises `crown_backward_within_block` → decomposed InstanceNorm dispatch.
/// The returned validation stats should include the InstanceNorm site, proving
/// the block-wise path did not route through the generic IBP fallback arm.
///
/// Verifies soundness: all concrete evaluations within the epsilon-ball
/// must fall inside the CROWN block bounds.
#[ntest::timeout(10000)]
#[test]
fn test_decomposed_instnorm_block_crown_soundness_3830() {
    let num_channels = 2;
    let time_len = 4;
    let epsilon = 0.05_f32;
    let shape = [num_channels, time_len];

    let graph = build_instnorm_block(num_channels);

    let exec_order = graph.exec_order().unwrap();
    let block_nodes_map = GraphNetwork::collect_block_nodes(exec_order);
    assert_eq!(
        block_nodes_map.len(),
        1,
        "Expected 1 block from layer0_* nodes"
    );

    let nodes_in_block = block_nodes_map.values().next().unwrap();
    assert!(
        nodes_in_block.iter().any(|n| n.contains("instnorm")),
        "Block should contain instnorm node: {:?}",
        nodes_in_block
    );

    let block_input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&shape)), epsilon).unwrap();
    let block_node_bounds = graph
        .collect_block_ibp_bounds(nodes_in_block, &block_input)
        .unwrap();

    let (crown_bounds, stats, provenance) = graph
        .crown_backward_within_block(nodes_in_block, &block_node_bounds, &block_input)
        .unwrap();
    assert_eq!(
        provenance,
        BoundsProvenance::Crown,
        "block-wise CROWN on InstanceNorm block must not fall back to forward bounds"
    );
    assert!(
        stats.iter().any(|stat| stat.node_name == "layer0_instnorm"),
        "block-wise InstanceNorm should report decomposed validation stats, got {:?}",
        stats
    );
    assert_finite_ordered_bounds(&crown_bounds);

    // Soundness: all concrete evaluations must be within CROWN bounds.
    let sample_points = generate_2d_sample_points(num_channels, time_len, epsilon, 30);
    assert_instnorm_block_soundness(&crown_bounds, &sample_points, num_channels, time_len);
}

/// Test decomposed InstanceNorm through the public `propagate_crown_block_wise` API.
///
/// Verifies that:
/// 1. The block-wise API succeeds (no panics, no errors)
/// 2. CROWN is reported as successful
/// 3. CROWN/IBP ratio is bounded on the toy block
#[ntest::timeout(10000)]
#[test]
fn test_decomposed_instnorm_block_wise_api_3830() {
    let num_channels = 2;
    let time_len = 4;
    let epsilon = 0.05_f32;
    let shape = [num_channels, time_len];

    let graph = build_instnorm_block(num_channels);
    let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&shape)), epsilon).unwrap();

    let result = graph.propagate_crown_block_wise(&input, epsilon).unwrap();

    assert_eq!(result.total_blocks, 1, "Expected 1 block");

    let block = &result.blocks[0];
    assert!(
        block.crown_successful,
        "CROWN should succeed on decomposed InstanceNorm blocks"
    );
    assert!(block.ibp_max_width > 0.0, "IBP width should be positive");
    assert!(
        block.crown_max_width.is_finite(),
        "CROWN width should be finite, got: {}",
        block.crown_max_width
    );
    assert!(
        block.crown_ibp_ratio.is_finite(),
        "CROWN/IBP ratio should be finite, got: {:.6}",
        block.crown_ibp_ratio
    );
    assert!(
        block.crown_ibp_ratio < 5.0,
        "CROWN/IBP ratio should be bounded, got: {:.6}",
        block.crown_ibp_ratio
    );
}

/// Test that the decomposed InstanceNorm path also works for alpha-CROWN.
///
/// This block has no GELU nodes, so alpha-CROWN has no parameters to optimize;
/// the test only checks that the block-wise alpha entry point still succeeds.
///
#[ntest::timeout(60000)]
#[test]
fn test_decomposed_instnorm_alpha_crown_3830() {
    tests::with_crown_dense_budget_mb("2048", || {
        let num_channels = 2;
        let time_len = 4;
        let epsilon = 0.05_f32;
        let shape = [num_channels, time_len];

        let graph = build_instnorm_block(num_channels);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&shape)), epsilon).unwrap();

        let result = graph
            .propagate_alpha_crown_block_wise(&input, epsilon)
            .unwrap();

        assert_eq!(result.total_blocks, 1);

        let block = &result.blocks[0];
        assert!(
            block.crown_successful,
            "Alpha-CROWN should succeed with decomposed InstanceNorm"
        );
        // No GELU nodes → alpha optimization is skipped → alpha_crown_max_width must be None.
        assert!(
            block.alpha_crown_max_width.is_none(),
            "Block has no GELU nodes, so alpha_crown_max_width should be None, got: {:?}",
            block.alpha_crown_max_width,
        );
        // Fixed-CROWN bounds should still be valid.
        assert!(
            block.crown_max_width.is_finite(),
            "CROWN width should be finite"
        );
        assert!(
            block.crown_ibp_ratio < 5.0,
            "CROWN/IBP ratio should be bounded, got: {:.6}",
            block.crown_ibp_ratio
        );
    });
}
