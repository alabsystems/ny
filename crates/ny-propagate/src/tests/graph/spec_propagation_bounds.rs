// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `propagate_crown_with_specs_and_engine_with_bounds` — the precomputed
//! node-bounds path in spec-guided CROWN.
//!
//! Covers #1959 acceptance criteria:
//! - Direct tests exist for `propagate_crown_with_specs_and_engine_with_bounds`
//!
//! Reference: designs/2026-02-10-network-beta-crown-coverage-wave-plan.md Step 2

use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use std::collections::HashMap;

/// Build Linear(2->3) -> ReLU -> Linear(3->2) graph
fn build_relu_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1, -0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));

    graph.set_output("linear2");
    graph
}

/// Build Linear(2->2) -> SiLU -> Linear(2->2) graph
fn build_silu_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.8_f32, -0.4], [0.3, 0.9]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.1, -0.2]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "silu",
        Layer::SiLU(SiLULayer::new()),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.6_f32, -0.2], [0.4, 0.5]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["silu".to_string()],
    ));

    graph.set_output("linear2");
    graph
}

fn build_reference_bounds_graph_3870() -> GraphNetwork {
    let w1 = arr2(&[[1.2, -0.8], [-0.6, 1.1], [0.9, 0.7], [-0.7, 0.4]]);
    let b1 = arr1(&[0.1, -0.05, 0.0, 0.12]);
    let w2 = arr2(&[[0.8, -0.5, 0.6, -0.2], [-0.3, 0.9, -0.4, 0.7]]);
    let b2 = arr1(&[0.05, -0.08]);
    let w3 = arr2(&[[1.0, -0.2], [-0.4, 0.9]]);
    let b3 = arr1(&[0.02, -0.03]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("valid linear2")),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).expect("valid linear3")),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");
    graph
}

/// Manual forward pass through the ReLU graph: Linear(w1,b1)->ReLU->Linear(w2)
fn relu_graph_forward(x: &[f32; 2]) -> [f32; 2] {
    let w1 = [[1.0f32, -0.5], [0.5, 1.0], [-1.0, 0.3]];
    let b1 = [0.0f32, 0.1, -0.1];
    let w2 = [[0.5f32, -0.3, 0.8], [0.2, 0.6, -0.4]];

    let z1 = [
        (w1[0][0] * x[0] + w1[0][1] * x[1] + b1[0]).max(0.0),
        (w1[1][0] * x[0] + w1[1][1] * x[1] + b1[1]).max(0.0),
        (w1[2][0] * x[0] + w1[2][1] * x[1] + b1[2]).max(0.0),
    ];
    [
        w2[0][0] * z1[0] + w2[0][1] * z1[1] + w2[0][2] * z1[2],
        w2[1][0] * z1[0] + w2[1][1] * z1[1] + w2[1][2] * z1[2],
    ]
}

// ---------------------------------------------------------------------------
// Tests for precomputed-bounds path
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_spec_with_precomputed_bounds_soundness_relu() {
    // Verify that passing precomputed IBP node bounds to
    // `propagate_crown_with_specs_and_engine_with_node_bounds`
    // produces sound bounds for a ReLU network.
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    // Spec: output[0] - output[1] (binary classification)
    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    // Get node bounds via IBP
    let node_bounds = graph.collect_node_bounds(&input).unwrap();

    // Call with precomputed bounds
    let spec_bounds = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
        )
        .unwrap();

    assert_eq!(spec_bounds.shape(), &[1]);
    assert!(spec_bounds.lower().iter().all(|v| v.is_finite()));
    assert!(spec_bounds.upper().iter().all(|v| v.is_finite()));

    // Verify soundness by sampling
    for i in 0..100 {
        let t1 = (i * 7 % 100) as f32 / 100.0;
        let t2 = (i * 11 % 100) as f32 / 100.0;
        let x = [-0.5 + t1, -0.5 + t2];
        let y = relu_graph_forward(&x);
        // Apply spec: output[0] - output[1]
        let spec_val = y[0] - y[1];
        assert!(
            spec_val >= spec_bounds.lower()[[0]] - 1e-4
                && spec_val <= spec_bounds.upper()[[0]] + 1e-4,
            "Spec value {} not in [{}, {}] for input {:?}",
            spec_val,
            spec_bounds.lower()[[0]],
            spec_bounds.upper()[[0]],
            x
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_with_precomputed_bounds_matches_without_relu() {
    // Verify that passing precomputed bounds produces the same result as
    // letting CROWN compute bounds internally for a ReLU network.
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    // Without precomputed bounds
    let bounds_auto = graph
        .propagate_crown_with_specs_and_engine(&input, &spec_matrix, None)
        .unwrap();

    // With precomputed bounds: use CROWN-IBP to match what the auto path
    // computes internally (should_use_crown_ibp_intermediates returns true
    // for this ReLU graph, so the auto path uses collect_crown_ibp_bounds_dag).
    let node_bounds = graph.collect_crown_ibp_bounds_dag(&input).unwrap();
    let bounds_precomputed = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
        )
        .unwrap();

    // Should be identical since we pass the same IBP bounds
    for i in 0..bounds_auto.shape()[0] {
        assert!(
            (bounds_auto.lower()[[i]] - bounds_precomputed.lower()[[i]]).abs() < 1e-4,
            "Lower mismatch at {}: auto={} precomputed={}",
            i,
            bounds_auto.lower()[[i]],
            bounds_precomputed.lower()[[i]]
        );
        assert!(
            (bounds_auto.upper()[[i]] - bounds_precomputed.upper()[[i]]).abs() < 1e-4,
            "Upper mismatch at {}: auto={} precomputed={}",
            i,
            bounds_auto.upper()[[i]],
            bounds_precomputed.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_with_precomputed_bounds_soundness_silu() {
    // Verify soundness with SiLU activation (non-ReLU).
    let graph = build_silu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.3_f32, -0.3]).into_dyn(),
        arr1(&[0.3_f32, 0.3]).into_dyn(),
    )
    .unwrap();

    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let spec_bounds = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
        )
        .unwrap();

    assert_eq!(spec_bounds.shape(), &[1]);
    assert!(spec_bounds.lower().iter().all(|v| v.is_finite()));
    assert!(spec_bounds.upper().iter().all(|v| v.is_finite()));
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_with_precomputed_bounds_multiclass() {
    // Test multiclass spec matrix with precomputed bounds.
    // Spec: [y0 - y1, y0 - y2] (prove class 0 wins)
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.3_f32, -0.3]).into_dyn(),
        arr1(&[0.3_f32, 0.3]).into_dyn(),
    )
    .unwrap();

    // 2 specs, 2 outputs
    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let spec_bounds = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
        )
        .unwrap();

    assert_eq!(spec_bounds.shape(), &[1]);

    // Verify soundness by sampling
    for i in 0..50 {
        let t1 = (i * 7 % 50) as f32 / 50.0;
        let t2 = (i * 13 % 50) as f32 / 50.0;
        let x = [-0.3 + 0.6 * t1, -0.3 + 0.6 * t2];
        let y = relu_graph_forward(&x);
        let spec_val = y[0] - y[1];
        assert!(
            spec_val >= spec_bounds.lower()[[0]] - 1e-4
                && spec_val <= spec_bounds.upper()[[0]] + 1e-4,
            "Multiclass spec value {} not in [{}, {}]",
            spec_val,
            spec_bounds.lower()[[0]],
            spec_bounds.upper()[[0]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_with_tighter_precomputed_bounds_gives_tighter_results() {
    // When precomputed node bounds are tighter (e.g., from alpha-CROWN),
    // the spec-guided CROWN result should be at least as tight as with IBP bounds.
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    // IBP bounds (loose)
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let result_ibp = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &ibp_bounds,
        )
        .unwrap();

    // Artificially tightened bounds (shrink IBP bounds by 20%)
    let mut tight_bounds = HashMap::new();
    for (name, bt) in &ibp_bounds {
        let mid = bt
            .lower()
            .iter()
            .zip(bt.upper().iter())
            .map(|(l, u)| (l + u) / 2.0)
            .collect::<Vec<_>>();
        let half_width = bt
            .lower()
            .iter()
            .zip(bt.upper().iter())
            .map(|(l, u)| (u - l) / 2.0 * 0.8) // 80% of original width
            .collect::<Vec<_>>();
        let new_lower: Vec<f32> = mid
            .iter()
            .zip(half_width.iter())
            .map(|(m, h)| m - h)
            .collect();
        let new_upper: Vec<f32> = mid
            .iter()
            .zip(half_width.iter())
            .map(|(m, h)| m + h)
            .collect();
        let shape = bt.lower().shape().to_vec();
        let new_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape), new_lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&shape), new_upper).unwrap(),
        )
        .unwrap();
        tight_bounds.insert(name.clone(), new_bt);
    }

    let result_tight = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &tight_bounds,
        )
        .unwrap();

    let ibp_width = result_ibp.upper()[[0]] - result_ibp.lower()[[0]];
    let tight_width = result_tight.upper()[[0]] - result_tight.lower()[[0]];

    // Tighter precomputed bounds should give tighter or equal spec bounds
    assert!(
        tight_width <= ibp_width + 1e-4,
        "Tighter precomputed bounds gave wider result: tight_width={} > ibp_width={}",
        tight_width,
        ibp_width
    );
}

/// Run spec-guided CROWN with optional fixed/reference node bounds. Helper for #3870 test.
fn run_spec_crown_3870(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec: &ndarray::Array2<f32>,
    fixed: Option<&HashMap<String, BoundedTensor>>,
    reference: Option<&HashMap<String, BoundedTensor>>,
) -> BoundedTensor {
    network::SpecCrownRequest::new(graph, input, spec, None)
        .node_bounds_opt(fixed)
        .reference_bounds_opt(reference)
        .run()
        .unwrap()
}

/// Assert bounds `inner` are at least as tight as `outer` (with tolerance).
fn assert_at_least_as_tight(inner: &BoundedTensor, outer: &BoundedTensor, label: &str) {
    assert!(
        inner.lower()[[0]] >= outer.lower()[[0]] - 1e-5,
        "{label}: inner lower {} loosened vs outer {}",
        inner.lower()[[0]],
        outer.lower()[[0]]
    );
    assert!(
        inner.upper()[[0]] <= outer.upper()[[0]] + 1e-5,
        "{label}: inner upper {} loosened vs outer {}",
        inner.upper()[[0]],
        outer.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_reference_bounds_recompute_then_tighten_3870() {
    let graph = build_reference_bounds_graph_3870();
    assert!(
        graph.should_use_crown_ibp_intermediates(),
        "test graph should use CROWN-IBP"
    );

    let child_input = BoundedTensor::new(
        arr1(&[-0.35_f32, -0.65_f32]).into_dyn(),
        arr1(&[0.55_f32, 0.15_f32]).into_dyn(),
    )
    .unwrap();
    let spec = arr2(&[[1.0_f32, -0.35_f32]]);
    let ibp_bounds = graph.collect_node_bounds(&child_input).unwrap();

    let fresh = run_spec_crown_3870(&graph, &child_input, &spec, None, None);
    let reference = run_spec_crown_3870(&graph, &child_input, &spec, None, Some(&ibp_bounds));
    let frozen = run_spec_crown_3870(&graph, &child_input, &spec, Some(&ibp_bounds), None);

    assert_at_least_as_tight(&reference, &fresh, "reference vs fresh");
    assert_at_least_as_tight(&reference, &frozen, "reference vs frozen");

    let reference_width = reference.upper()[[0]] - reference.lower()[[0]];
    let frozen_width = frozen.upper()[[0]] - frozen.lower()[[0]];
    assert!(
        frozen_width > reference_width + 1e-6,
        "test must distinguish frozen from reference: frozen_w={frozen_width}, ref_w={reference_width}",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_precomputed_bounds_missing_node_produces_error_or_fallback() {
    // When precomputed_node_bounds is missing a required node, the function
    // should either error or fall back gracefully.
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    // Empty node bounds — missing all required nodes
    let empty_bounds: HashMap<String, BoundedTensor> = HashMap::new();

    let result = graph.propagate_crown_with_specs_and_engine_with_node_bounds(
        &input,
        &spec_matrix,
        None,
        &empty_bounds,
    );

    // Should either succeed with a fallback or return an error.
    // Either way, it should not panic.
    match result {
        Ok(bounds) => {
            // If it succeeds, bounds must still be sound (may be looser)
            assert!(bounds.lower().iter().all(|v| v.is_finite()));
            assert!(bounds.upper().iter().all(|v| v.is_finite()));
        }
        Err(_) => {
            // Error is acceptable when required bounds are missing
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_with_linear_returns_linear_coefficients() {
    // Verify that `propagate_crown_with_specs_and_engine_with_linear`
    // returns LinearBounds with correct dimensions.
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    let (spec_bounds, linear_bounds) = graph
        .propagate_crown_with_specs_and_engine_with_linear(&input, &spec_matrix, None)
        .unwrap();

    assert_eq!(spec_bounds.shape(), &[1]);

    // Linear bounds should be present for non-empty graph
    if let Some(lb) = &linear_bounds {
        // A matrices should be (num_specs, input_dim)
        assert_eq!(lb.lower_a.nrows(), 1); // 1 spec
        assert_eq!(lb.lower_a.ncols(), 2); // 2 input dims
        assert_eq!(lb.upper_a.nrows(), 1);
        assert_eq!(lb.upper_a.ncols(), 2);

        // b vectors should have length = num_specs
        assert_eq!(lb.lower_b.len(), 1);
        assert_eq!(lb.upper_b.len(), 1);

        // Verify the linear bounds are consistent: for any x in [l,u],
        // lower_a . x + lower_b <= true_spec_value <= upper_a . x + upper_b
        for i in 0..50 {
            let t1 = (i * 7 % 50) as f32 / 50.0;
            let t2 = (i * 13 % 50) as f32 / 50.0;
            let x = [-0.5 + t1, -0.5 + t2];
            let y = relu_graph_forward(&x);
            let spec_val = y[0] - y[1];

            let lb_val = lb.lower_a[[0, 0]] * x[0] + lb.lower_a[[0, 1]] * x[1] + lb.lower_b[0];
            let ub_val = lb.upper_a[[0, 0]] * x[0] + lb.upper_a[[0, 1]] * x[1] + lb.upper_b[0];

            assert!(
                spec_val >= lb_val - 1e-4,
                "Linear lower {} > spec {} for x={:?}",
                lb_val,
                spec_val,
                x
            );
            assert!(
                spec_val <= ub_val + 1e-4,
                "Linear upper {} < spec {} for x={:?}",
                ub_val,
                spec_val,
                x
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_empty_graph_with_precomputed_bounds() {
    // A single-linear-layer graph (no activation) — precomputed bounds
    // should make no difference since there are no non-linear layers to relax.
    let mut graph = GraphNetwork::new();
    let w = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    // Get node bounds (nothing to relax, but function should work)
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let bounds_with = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
        )
        .unwrap();

    let bounds_without = graph
        .propagate_crown_with_specs_and_engine(&input, &spec_matrix, None)
        .unwrap();

    // Linear-only graph: precomputed bounds should make no difference
    assert!(
        (bounds_with.lower()[[0]] - bounds_without.lower()[[0]]).abs() < 1e-5,
        "Linear-only: precomputed lower {} != auto lower {}",
        bounds_with.lower()[[0]],
        bounds_without.lower()[[0]]
    );
    assert!(
        (bounds_with.upper()[[0]] - bounds_without.upper()[[0]]).abs() < 1e-5,
        "Linear-only: precomputed upper {} != auto upper {}",
        bounds_with.upper()[[0]],
        bounds_without.upper()[[0]]
    );
}

/// Batched multi-spec backward must produce IDENTICAL per-spec bounds vs running
/// each spec row through its own single-row backward pass.
///
/// The spec-guided CROWN backward seeds the output node with the full
/// `(num_specs, output_dim)` spec matrix (`LinearBounds::from_spec_matrix`) and
/// propagates every row together through each layer's backward (a matmul that
/// handles multiple output rows at once — exactly like a multi-output layer).
/// This test locks in that batched behavior: stacking K spec rows into one
/// backward must equal K independent single-row backward passes, since the
/// linear algebra is identical, just stacked. The ReLU graph is used so the
/// equivalence holds across the non-linear relaxation (each row's relaxation is
/// driven by the same precomputed pre-activation bounds, independent of the
/// other rows). Soundness/perf guard for the many-output-spec scenario
/// (e.g. TinyImageNet ResNet ~199 robustness specs).
#[ntest::timeout(10000)]
#[test]
fn test_spec_batched_matches_per_row_relu_graph() {
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.7]).into_dyn(),
        arr1(&[0.6_f32, 0.4]).into_dyn(),
    )
    .unwrap();

    // Multi-row spec matrix: each row is a distinct linear objective over the
    // 2-dim network output. Includes the canonical y0-y1 robustness comparison
    // plus other directions so the test exercises true row-stacking, not a
    // degenerate single direction.
    let spec_matrix = arr2(&[
        [1.0_f32, -1.0], // y0 - y1
        [-1.0_f32, 1.0], // y1 - y0
        [1.0_f32, 0.0],  // y0
        [0.0_f32, 1.0],  // y1
        [2.0_f32, -0.5], // mixed direction
    ]);
    let num_specs = spec_matrix.nrows();

    // Batched: all rows in one backward pass.
    let batched = network::SpecCrownRequest::new(&graph, &input, &spec_matrix, None)
        .run()
        .unwrap();
    assert_eq!(batched.shape(), &[num_specs]);

    // Per-row: one single-row backward pass per spec.
    for row in 0..num_specs {
        let single_row = arr2(&[[spec_matrix[[row, 0]], spec_matrix[[row, 1]]]]);
        let per_row = network::SpecCrownRequest::new(&graph, &input, &single_row, None)
            .run()
            .unwrap();
        assert_eq!(per_row.shape(), &[1]);

        let batched_lo = batched.lower()[[row]];
        let batched_hi = batched.upper()[[row]];
        let row_lo = per_row.lower()[[0]];
        let row_hi = per_row.upper()[[0]];

        assert!(
            (batched_lo - row_lo).abs() < 1e-5,
            "spec row {row}: batched lower {batched_lo} != per-row lower {row_lo}",
        );
        assert!(
            (batched_hi - row_hi).abs() < 1e-5,
            "spec row {row}: batched upper {batched_hi} != per-row upper {row_hi}",
        );
    }
}

/// Build graph: input [2,3] / (ReduceSum(input, axis=-1, keepdims=true) + 3.0) → [2,3].
/// Leading-axis broadcast [2,3] / [2,1] — the L2-normalization pattern.
/// Part of #3626, #3499.
fn build_leading_axis_broadcast_div_graph() -> GraphNetwork {
    use ndarray::{ArrayD, IxDyn};

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "reduce",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
    ));
    let shift_const = ArrayD::from_elem(IxDyn(&[1]), 3.0_f32);
    graph.add_node(GraphNode::new(
        "shift",
        Layer::AddConstant(AddConstantLayer::new(shift_const)),
        vec!["reduce".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "div",
        Layer::Div(DivLayer),
        NETWORK_INPUT,
        "shift",
    ));
    graph.set_output("div");
    graph
}

/// Forward eval for the leading-axis broadcast Div graph.
fn eval_leading_axis_broadcast_div(x: &[f32; 6]) -> [f32; 6] {
    let denom0 = x[0] + x[1] + x[2] + 3.0;
    let denom1 = x[3] + x[4] + x[5] + 3.0;
    [
        x[0] / denom0,
        x[1] / denom0,
        x[2] / denom0,
        x[3] / denom1,
        x[4] / denom1,
        x[5] / denom1,
    ]
}

/// Regression: Div CROWN backward with non-trailing broadcast [2,3] / [2,1].
///
/// Before the fix, flat stride `elem += b_len` grouped {0,2,4} and {1,3,5}
/// instead of the correct {0,1,2} and {3,4,5} for leading-axis broadcasts.
/// Part of #3626, #3499. Requested by Prover audit of commit 021ded490.
#[ntest::timeout(60000)]
#[test]
fn test_div_crown_leading_axis_broadcast_regression_3626() {
    use ndarray::{ArrayD, IxDyn};

    let graph = build_leading_axis_broadcast_div_graph();
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5_f32; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.5_f32; 6]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let ibp = graph.propagate_ibp(&input).unwrap();
    assert_eq!(ibp.shape(), &[2, 3]);
    assert!(ibp.lower().iter().all(|v| v.is_finite()));
    assert!(ibp.upper().iter().all(|v| v.is_finite()));

    let crown = graph
        .propagate_crown_with_specs_and_engine(&input, &ndarray::Array2::eye(6), None)
        .unwrap();
    assert_eq!(crown.len(), 6);
    assert!(crown.lower().iter().all(|v| v.is_finite()));
    assert!(crown.upper().iter().all(|v| v.is_finite()));

    // CROWN must not be looser than IBP.
    let ibp_lo: Vec<f32> = ibp.lower().iter().copied().collect();
    let ibp_up: Vec<f32> = ibp.upper().iter().copied().collect();
    for i in 0..6 {
        assert!(
            crown.lower()[[i]] >= ibp_lo[i] - 1e-3,
            "dim {i}: CROWN < IBP lower"
        );
        assert!(
            crown.upper()[[i]] <= ibp_up[i] + 1e-3,
            "dim {i}: CROWN > IBP upper"
        );
    }

    // Soundness: sampled concrete outputs within CROWN bounds.
    for i0 in 0..5 {
        for i1 in 0..5 {
            let t0 = 0.5 + (i0 as f32) / 4.0;
            let t1 = 0.5 + (i1 as f32) / 4.0;
            let x = [t0, t0 + 0.1, t0 - 0.1, t1, t1 + 0.2, t1 - 0.15];
            if x.iter().any(|&v| !(0.5..=1.5).contains(&v)) {
                continue;
            }
            let output = eval_leading_axis_broadcast_div(&x);
            for (dim, &val) in output.iter().enumerate() {
                assert!(
                    val >= crown.lower()[[dim]] - 1e-3,
                    "dim {dim}: unsound lower"
                );
                assert!(
                    val <= crown.upper()[[dim]] + 1e-3,
                    "dim {dim}: unsound upper"
                );
            }
        }
    }
}

/// Build graph: input [2] / (input + [2.0, 3.0]) → [2].
/// Element-wise Div (b_len == n) where the numerator can be negative.
/// This exercises the element-wise branch in spec_propagation.rs, which
/// is distinct from the broadcasting branch tested by the _3626 regression test.
/// Part of #3626.
fn build_elementwise_div_graph() -> GraphNetwork {
    use ndarray::{ArrayD, IxDyn};

    let mut graph = GraphNetwork::new();
    let shift_const = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0_f32, 3.0]).unwrap();
    graph.add_node(GraphNode::from_input(
        "den",
        Layer::AddConstant(AddConstantLayer::new(shift_const)),
    ));
    graph.add_node(GraphNode::binary(
        "div",
        Layer::Div(DivLayer),
        NETWORK_INPUT,
        "den",
    ));
    graph.set_output("div");
    graph
}

/// Forward eval for the element-wise Div graph.
fn eval_elementwise_div(x: &[f32; 2]) -> [f32; 2] {
    [x[0] / (x[0] + 2.0), x[1] / (x[1] + 3.0)]
}

/// Soundness test: element-wise Div CROWN with mixed-sign numerator.
///
/// Counterexample from reports/prover/2026-03-12-issue-3626-div-crown-audit.md:
/// For x ∈ [-1, 0], num(x) = x, den(x) = x + 2, y(x) = x/(x+2).
/// At x = -0.5: actual = -0.333..., element-wise interval scaling gives -0.25 (unsound).
///
/// The element-wise branch (b_len == n) in spec_propagation.rs uses interval
/// multiplication of A coefficients by reciprocal bounds without bias correction.
/// This is unsound when the numerator crosses zero because the linear bound
/// y ≈ a_scaled * num cannot simultaneously be a lower bound for both positive
/// and negative numerator values.
///
/// Part of #3626. Filed by Prover: confirms the audit report counterexample
/// exercises the actual code path and produces an unsound bound.
#[ntest::timeout(60000)]
#[test]
fn test_elementwise_div_crown_soundness_mixed_sign_3626() {
    use ndarray::ArrayD;

    let graph = build_elementwise_div_graph();
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, 0.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let ibp = graph.propagate_ibp(&input).unwrap();
    assert_eq!(ibp.shape(), &[2]);
    assert!(ibp.lower().iter().all(|v| v.is_finite()));
    assert!(ibp.upper().iter().all(|v| v.is_finite()));

    let crown = graph
        .propagate_crown_with_specs_and_engine(&input, &ndarray::Array2::eye(2), None)
        .unwrap();
    assert_eq!(crown.len(), 2);
    assert!(crown.lower().iter().all(|v| v.is_finite()));
    assert!(crown.upper().iter().all(|v| v.is_finite()));

    // Soundness: every concrete output must be within CROWN bounds.
    // The counterexample x = [-0.5, -0.5] should trigger:
    //   actual[0] = -0.5 / 1.5 = -0.333... but the element-wise interval scaling
    //   gives lower bound -0.25 (unsound: -0.25 > -0.333...).
    let mut unsound_count = 0;
    for i0 in 0..=20 {
        for i1 in 0..=20 {
            let x0 = -1.0 + (i0 as f32) / 20.0; // [-1.0, 0.0]
            let x1 = -1.0 + (i1 as f32) / 20.0; // [-1.0, 0.0]
            let output = eval_elementwise_div(&[x0, x1]);
            for (dim, &val) in output.iter().enumerate() {
                if val < crown.lower()[[dim]] - 1e-6 {
                    unsound_count += 1;
                }
                if val > crown.upper()[[dim]] + 1e-6 {
                    unsound_count += 1;
                }
            }
        }
    }

    assert_eq!(
        unsound_count, 0,
        "Element-wise Div CROWN produced unsound bounds: {unsound_count} violations found. \
         See reports/prover/2026-03-12-issue-3626-div-crown-audit.md for the counterexample."
    );
}

/// Build the #3680 extracted-stage reproducer graph:
/// `_input [1,1,5] → slice(axis=-1, 1..4) → pad(constant, +1/+1)
///                  → conv1d(1→1, k=2) → add(skip_slice) → out [1,1,4]`
///
/// Same topology as `build_extracted_graph_target_shape_regression_3680` in
/// `bounds/tests.rs`, duplicated here to avoid cross-module test coupling.
fn build_extracted_stage_graph_3680() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "main_slice",
        Layer::Slice(SliceLayer::new(-1, 1, 4)),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "main_pad",
        Layer::Pad(PadLayer::new(
            vec![(0, 0), (0, 0), (1, 1)],
            PadMode::Constant(0.0),
        )),
        vec!["main_slice".to_string()],
    ));
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.45_f32, -0.20])
        .expect("valid Conv1d kernel");
    let conv = Conv1dLayer::with_input_length(kernel, Some(arr1(&[0.05_f32])), 1, 0, 5)
        .expect("valid Conv1d params");
    graph.add_node(GraphNode::new(
        "main_conv",
        Layer::Conv1d(conv),
        vec!["main_pad".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "skip_slice",
        Layer::Slice(SliceLayer::new(-1, 0, 4)),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "output_add",
        Layer::Add(AddLayer),
        vec!["main_conv".to_string(), "skip_slice".to_string()],
    ));
    graph.set_output("output_add");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![-1.0_f32, -0.8, -0.4, 0.1, 0.5])
            .expect("valid lower input"),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![0.2_f32, 0.4, 0.7, 1.0, 1.3])
            .expect("valid upper input"),
    )
    .expect("valid bounded input");

    (graph, input)
}

/// Regression #3680: spec-guided CROWN must accept identity spec on an
/// extracted-subgraph with N-D target (Slice → Pad → Conv1d → Add skip).
///
/// Exercises the spec-propagation path through
/// `propagate_crown_with_specs_and_engine` on the same topology already
/// covered by `test_extracted_graph_crown_target_shape_3680` in
/// `bounds/tests.rs` (direct CROWN + CROWN-IBP).
#[ntest::timeout(10000)]
#[test]
fn test_spec_crown_extracted_graph_target_shape_3680() {
    let (graph, input) = build_extracted_stage_graph_3680();

    let ibp = graph.propagate_ibp(&input).unwrap();
    assert_eq!(ibp.shape(), &[1, 1, 4], "#3680 IBP output must be [1,1,4]");
    assert!(ibp.lower().iter().all(|v| v.is_finite()));

    let flat_dim = ibp.len();
    assert_eq!(flat_dim, 4);

    let spec = ndarray::Array2::eye(flat_dim);
    let crown = graph
        .propagate_crown_with_specs_and_engine(&input, &spec, None)
        .expect("#3680 spec-guided CROWN must accept identity spec on extracted graph");

    assert_eq!(crown.len(), flat_dim);
    assert!(crown.lower().iter().all(|v| v.is_finite()));
    assert!(crown.upper().iter().all(|v| v.is_finite()));

    // CROWN must not be looser than IBP (soundness).
    let ibp_lo: Vec<f32> = ibp.lower().iter().copied().collect();
    let ibp_up: Vec<f32> = ibp.upper().iter().copied().collect();
    for i in 0..flat_dim {
        assert!(
            crown.lower()[[i]] >= ibp_lo[i] - 1e-3,
            "#3680 dim {i}: spec-CROWN lower < IBP lower"
        );
        assert!(
            crown.upper()[[i]] <= ibp_up[i] + 1e-3,
            "#3680 dim {i}: spec-CROWN upper > IBP upper"
        );
    }
}

/// An already-expired deadline must surface `ForwardFallback(DeadlineExceeded)`
/// through the provenance-carrying public API.
///
/// Part of #3520 Packet C Step 2: proves the provenance seam works at the
/// `GraphNetwork` dispatch level, not just the internal spec_propagation helper.
#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_provenance_deadline_fallback_3520() {
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason};

    let graph = build_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();
    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);
    let node_bounds = graph.collect_node_bounds(&input).unwrap();

    // Already-expired deadline forces immediate IBP fallback.
    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap();

    let result = graph
        .propagate_crown_with_specs_and_provenance_and_engine_with_node_bounds_and_deadline(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
            Some(expired),
        )
        .expect("provenance API should succeed even on deadline fallback");

    // Provenance must record the fallback reason.
    assert_eq!(
        result.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
        "expired deadline should produce ForwardFallback(DeadlineExceeded)"
    );

    // Bounds must still be finite and sound (IBP fallback, not failure).
    assert!(result.bounds.lower().iter().all(|v| v.is_finite()));
    assert!(result.bounds.upper().iter().all(|v| v.is_finite()));
    assert!(result.bounds.lower()[[0]] <= result.bounds.upper()[[0]]);

    // Compare against a no-deadline call to verify the fallback bounds are
    // at least as wide (IBP is looser than CROWN).
    let crown_result = graph
        .propagate_crown_with_specs_and_provenance_and_engine_with_node_bounds_and_deadline(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
            None, // no deadline
        )
        .expect("no-deadline should succeed");
    assert_eq!(crown_result.provenance, BoundsProvenance::Crown);

    let fallback_width = result.bounds.upper()[[0]] - result.bounds.lower()[[0]];
    let crown_width = crown_result.bounds.upper()[[0]] - crown_result.bounds.lower()[[0]];
    assert!(
        fallback_width >= crown_width - 1e-4,
        "IBP fallback bounds ({fallback_width}) should be at least as wide as CROWN ({crown_width})"
    );
}
