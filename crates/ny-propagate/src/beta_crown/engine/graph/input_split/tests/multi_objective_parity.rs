// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;
use std::{collections::HashMap, sync::Arc};

use super::*;
use crate::batched_domain::{BatchedDomainOptions, BatchedDomainsBuilder};
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::domain_batch::{
    DenseSpecBatchRequest, GraphDomainBatchExecutor,
};
use crate::beta_crown::engine::graph::propagation::BatchedBackwardContext;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::BranchingHeuristic;
use ny_core::NaiveCpuGemmEngine;

use crate::layers::linear::crown_single_gamma_n_f64;

use super::lane_gap::{
    assert_linear_bounds_within_gap, box_magnitudes, cross_lane_gap, LaneGap, LaneStep,
};

/// The child-parity fixture's weights, in ONE place: the graph builder and the
/// DERIVED cross-lane tolerance ([`child_parity_lane_gap_4116`]) both read them,
/// so a fixture edit cannot silently invalidate the derivation.
fn child_parity_w1_4116() -> Array2<f32> {
    arr2(&[[1.2, -0.8], [-0.6, 1.1], [0.9, 0.7], [-0.7, 0.4]])
}

fn child_parity_w2_4116() -> Array2<f32> {
    arr2(&[[0.8, -0.5, 0.6, -0.2], [-0.3, 0.9, -0.4, 0.7]])
}

fn child_parity_w3_4116() -> Array2<f32> {
    arr2(&[[1.0, -0.2], [-0.4, 0.9]])
}

fn child_parity_b1_4116() -> Array1<f32> {
    arr1(&[0.1, -0.05, 0.0, 0.12])
}

fn child_parity_b2_4116() -> Array1<f32> {
    arr1(&[0.05, -0.08])
}

fn child_parity_b3_4116() -> Array1<f32> {
    arr1(&[0.02, -0.03])
}

pub(super) fn build_multi_objective_child_parity_graph() -> GraphNetwork {
    let w1 = child_parity_w1_4116();
    let b1 = child_parity_b1_4116();
    let w2 = child_parity_w2_4116();
    let b2 = child_parity_b2_4116();
    let w3 = child_parity_w3_4116();
    let b3 = child_parity_b3_4116();

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

fn build_multi_objective_status_parity_graph_4354() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity linear")),
    ));
    graph.set_output("out");
    graph
}

fn build_input_split_status_config_4354() -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        input_split_ibp_enhancement: false,
        max_domains: 64,
        max_depth: 1,
        batch_size: 4,
        timeout: Duration::from_secs(5),
        reorder_bab: false,
        ..Default::default()
    }
}

fn assert_flat_bounds_close(label: &str, actual: &BoundedTensor, expected: &BoundedTensor) {
    let actual = actual.flatten();
    let expected = expected.flatten();
    assert_eq!(
        actual.lower().shape(),
        expected.lower().shape(),
        "{label}: lower-bound shape changed"
    );
    assert_eq!(
        actual.upper().shape(),
        expected.upper().shape(),
        "{label}: upper-bound shape changed"
    );

    for (idx, (actual, expected)) in actual
        .lower()
        .iter()
        .zip(expected.lower().iter())
        .enumerate()
    {
        assert!(
            (*actual - *expected).abs() <= 1e-6,
            "{label}: lower bound changed at index {idx}: actual={actual}, expected={expected}"
        );
    }

    for (idx, (actual, expected)) in actual
        .upper()
        .iter()
        .zip(expected.upper().iter())
        .enumerate()
    {
        assert!(
            (*actual - *expected).abs() <= 1e-6,
            "{label}: upper bound changed at index {idx}: actual={actual}, expected={expected}"
        );
    }
}

/// DERIVED cross-lane tolerance for the child-parity fixture over one domain.
///
/// Chain, output-first: `linear3` (contraction width `n = rows(W3) = 2`, input
/// node `relu2`) → `relu2` → `linear2` (`n = rows(W2) = 2`, input node `relu1`)
/// → `relu1` → `linear1` (`n = rows(W1) = 4`, input node = the network input).
/// The ReLU steps carry relaxation slopes in `[0, 1]` and so contribute no
/// growth. See [`super::lane_gap`] for the `γ_n^{f32}` vs `γ_n^{f64}` algebra;
/// this function only supplies this fixture's `|W|` matrices and box magnitudes.
///
/// `node_bounds` are the bounds the run itself used for the intermediate nodes
/// (the callers pass root/alpha node bounds to BOTH lanes, so those are the boxes
/// the relaxations were built from).
fn child_parity_lane_gap_4116(
    spec_matrix: &Array2<f32>,
    node_bounds: &HashMap<String, BoundedTensor>,
    domain: &BoundedTensor,
) -> LaneGap {
    let mag = |name: &str| {
        box_magnitudes(
            node_bounds
                .get(name)
                .unwrap_or_else(|| panic!("fixture must expose a '{name}' node")),
        )
    };
    let spec_abs = spec_matrix.mapv(|v| f64::from(v.abs()));
    cross_lane_gap(
        &spec_abs,
        &[
            LaneStep::new(&child_parity_w3_4116(), &mag("relu2")),
            LaneStep::new(&child_parity_w2_4116(), &mag("relu1")),
            LaneStep::new(&child_parity_w1_4116(), &box_magnitudes(domain)),
        ],
    )
}

/// Independent f64 forward evaluation of the child-parity fixture, built from
/// the same weight/bias accessors as the graph and touching NONE of the code
/// under test.
fn child_parity_forward_4116(x: &[f64; 2]) -> Vec<f64> {
    let layer = |w: &Array2<f32>, b: &Array1<f32>, input: &[f64]| -> Vec<f64> {
        (0..w.nrows())
            .map(|i| {
                (0..w.ncols())
                    .map(|j| f64::from(w[[i, j]]) * input[j])
                    .sum::<f64>()
                    + f64::from(b[i])
            })
            .collect()
    };
    let relu = |v: Vec<f64>| -> Vec<f64> { v.into_iter().map(|t| t.max(0.0)).collect() };

    let h1 = relu(layer(&child_parity_w1_4116(), &child_parity_b1_4116(), x));
    let h2 = relu(layer(&child_parity_w2_4116(), &child_parity_b2_4116(), &h1));
    layer(&child_parity_w3_4116(), &child_parity_b3_4116(), &h2)
}

/// Assert a certified spec box contains the network's actual spec outputs on a
/// deterministic grid over the domain.
///
/// This is NECESSARY, not sufficient: a grid cannot prove soundness. It is here
/// because the tolerance comparisons around it are two-lane AGREEMENT checks,
/// and two lanes can agree while both being wrong. A bound that has drifted onto
/// the wrong side of a reachable output fails here regardless of how well the
/// lanes agree with each other.
fn assert_spec_box_contains_forward_grid_4116(
    label: &str,
    bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    domain: &BoundedTensor,
) {
    let flat = bounds.flatten();
    let lower: Vec<f32> = flat.lower().iter().copied().collect();
    let upper: Vec<f32> = flat.upper().iter().copied().collect();
    assert_eq!(
        lower.len(),
        spec_matrix.nrows(),
        "{label}: one certified interval per spec row"
    );

    let dl: Vec<f64> = domain.lower().iter().map(|&v| f64::from(v)).collect();
    let du: Vec<f64> = domain.upper().iter().map(|&v| f64::from(v)).collect();
    assert_eq!(dl.len(), 2, "the grid is written for a 2-D domain");

    // f64 evaluation slack: at most 4 layers x (2 products + 1 sum) roundings on
    // magnitudes bounded by `scale` (Higham Thm 3.1 at u = 2^-53).
    let scale: f64 = 64.0;
    let slack = crown_single_gamma_n_f64(16) * scale;

    const STEPS: usize = 20;
    let mut failures = Vec::new();
    for i in 0..=STEPS {
        for j in 0..=STEPS {
            let t0 = i as f64 / STEPS as f64;
            let t1 = j as f64 / STEPS as f64;
            let x = [dl[0] + t0 * (du[0] - dl[0]), dl[1] + t1 * (du[1] - dl[1])];
            let out = child_parity_forward_4116(&x);
            for row in 0..spec_matrix.nrows() {
                let value: f64 = (0..spec_matrix.ncols())
                    .map(|c| f64::from(spec_matrix[[row, c]]) * out[c])
                    .sum();
                if value < f64::from(lower[row]) - slack {
                    failures.push(format!(
                        "{label}: spec row {row} at x={x:?} evaluates to {value}, BELOW the \
                         certified lower bound {} (slack {slack:e}) — FALSE PROOF",
                        lower[row]
                    ));
                }
                if value > f64::from(upper[row]) + slack {
                    failures.push(format!(
                        "{label}: spec row {row} at x={x:?} evaluates to {value}, ABOVE the \
                         certified upper bound {} (slack {slack:e}) — FALSE PROOF",
                        upper[row]
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} grid violations, first few:\n{}",
        failures.len(),
        failures
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    for row in 0..spec_matrix.nrows() {
        assert!(
            upper[row] - lower[row] > 0.0 && lower[row].is_finite() && upper[row].is_finite(),
            "{label}: spec row {row} certified a degenerate box [{}, {}] — the comparisons \
             around this check would be vacuous",
            lower[row],
            upper[row]
        );
    }
}

/// Bound comparison at an explicitly supplied tolerance, COLLECTING every
/// divergence. `assert_flat_bounds_close`'s flat `1e-6` aborts at the first
/// element, which leaves the rest of the vector unverified.
fn assert_flat_bounds_within(
    label: &str,
    actual: &BoundedTensor,
    expected: &BoundedTensor,
    tol: f32,
) {
    let actual = actual.flatten();
    let expected = expected.flatten();
    assert_eq!(
        actual.lower().shape(),
        expected.lower().shape(),
        "{label}: lower-bound shape changed"
    );
    assert_eq!(
        actual.upper().shape(),
        expected.upper().shape(),
        "{label}: upper-bound shape changed"
    );

    let mut failures = Vec::new();
    for (idx, (a, e)) in actual
        .lower()
        .iter()
        .zip(expected.lower().iter())
        .enumerate()
    {
        if (a - e).abs() > tol {
            failures.push(format!(
                "{label}: lower bound at {idx}: actual={a}, expected={e}, diff {} exceeds \
                 derived tolerance {tol:e}",
                (a - e).abs()
            ));
        }
    }
    for (idx, (a, e)) in actual
        .upper()
        .iter()
        .zip(expected.upper().iter())
        .enumerate()
    {
        if (a - e).abs() > tol {
            failures.push(format!(
                "{label}: upper bound at {idx}: actual={a}, expected={e}, diff {} exceeds \
                 derived tolerance {tol:e}",
                (a - e).abs()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn direct_mul_binary_bounds_4284(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    mul_binary_alphas: &HashMap<String, Array2<f32>>,
) -> (BoundedTensor, Option<LinearBounds>) {
    compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        input,
        spec_matrix,
        None,
        None,
        None,
        None,
        Some(mul_binary_alphas),
        None,
        None,
        false,
    )
    .expect("direct mulbinary child should succeed")
}

fn assert_mul_binary_dense_spec_matches_direct_4284(
    spec_result: &BatchedSpecBounds,
    expected_a: &(BoundedTensor, Option<LinearBounds>),
    expected_b: &(BoundedTensor, Option<LinearBounds>),
) {
    assert_eq!(
        spec_result.bounds.len(),
        2,
        "two mulbinary child domains should produce two batched results"
    );

    assert_flat_bounds_close(
        "mulbinary batched domain 0 vs direct call",
        &spec_result.bounds[0],
        &expected_a.0,
    );
    assert_flat_bounds_close(
        "mulbinary batched domain 1 vs direct call",
        &spec_result.bounds[1],
        &expected_b.0,
    );

    match (&spec_result.linear_bounds[0], &expected_a.1) {
        (Some(actual), Some(expected)) => assert_linear_bounds_match(actual, expected),
        (None, None) => {}
        (actual, expected) => panic!(
            "mulbinary child_a linear bounds diverged: batched={} direct={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
    match (&spec_result.linear_bounds[1], &expected_b.1) {
        (Some(actual), Some(expected)) => assert_linear_bounds_match(actual, expected),
        (None, None) => {}
        (actual, expected) => panic!(
            "mulbinary child_b linear bounds diverged: batched={} direct={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

#[test]
fn test_multi_objective_child_joint_spec_helper_matches_direct_crown_3870() {
    let graph = build_multi_objective_child_parity_graph();
    let root_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid root input");
    let child_input = BoundedTensor::new(
        arr1(&[-0.35_f32, -0.65_f32]).into_dyn(),
        arr1(&[0.55_f32, 0.15_f32]).into_dyn(),
    )
    .expect("valid child input");
    let root_node_bounds = graph
        .collect_node_bounds(&root_input)
        .expect("root node bounds should succeed");
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32], [-0.6_f32, 1.0_f32]]);

    let (baseline_bounds, baseline_linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &child_input,
            &spec_matrix,
            None,
            &root_node_bounds,
            None,
        )
        .expect("direct joint-spec CROWN should succeed");

    let (helper_bounds, helper_linear) = compute_crown_or_ibp_bounds_with_node_bounds(
        &graph,
        &child_input,
        &spec_matrix,
        None,
        Some(&root_node_bounds),
        None,
        None,
        None,
        None,
        None,
        false, // ibp_enhancement
    )
    .expect("single-domain helper should preserve joint multi-row semantics");

    assert_flat_bounds_close(
        "single-domain helper vs direct joint-spec CROWN",
        &helper_bounds,
        &baseline_bounds,
    );

    match (&helper_linear, &baseline_linear) {
        (Some(actual), Some(expected)) => assert_linear_bounds_match(actual, expected),
        (None, None) => {}
        (actual, expected) => panic!(
            "single-domain helper/direct linear bound availability diverged: helper={} direct={}",
            actual.is_some(),
            expected.is_some()
        ),
    }

    let err = compute_crown_or_ibp_bounds_batched(
        &graph,
        &[&child_input],
        &spec_matrix,
        None,
        Some(&root_node_bounds),
        None,
        None,
        None,
        None,
        false, // ibp_enhancement
    )
    .expect_err("scalar batched helper must reject multi-row spec matrices");

    assert!(
        err.to_string().contains("single-row spec matrices"),
        "guard should explain the single-objective contract, got: {err}"
    );
}

/// Positive parity: dense-spec batched helper matches direct joint-spec CROWN
/// for one child domain with a multi-row spec matrix.
///
/// This is the #4116 Packet A flip: the old test above proves the scalar helper
/// rejects multi-row specs, while this test proves the new dense-spec helper
/// accepts them and produces identical bounds.
///
/// Checks, for the two lanes:
///   * agreement within the DERIVED cross-lane certificate gap (bounds and
///     linear bounds), and
///   * soundness of each against an independent f64 forward evaluation.
///
/// Part of #4116 Packet A Step 5.
#[test]
fn test_dense_spec_batched_helper_matches_direct_crown_4116() {
    let graph = build_multi_objective_child_parity_graph();
    let root_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid root input");
    let child_input = BoundedTensor::new(
        arr1(&[-0.35_f32, -0.65_f32]).into_dyn(),
        arr1(&[0.55_f32, 0.15_f32]).into_dyn(),
    )
    .expect("valid child input");
    let root_node_bounds = graph
        .collect_node_bounds(&root_input)
        .expect("root node bounds should succeed");
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32], [-0.6_f32, 1.0_f32]]);

    // The two lanes no longer share one arithmetic, and the tolerance below says
    // so explicitly.
    //
    // The rationale this test used to carry ("both paths use the SAME GEMM engine
    // ... pinning both to NaiveCpuGemmEngine makes the linear bounds match
    // bit-exactly") was a measurement claim with no expiry, and `9f06e35b2`
    // falsified it: that commit moved the DOMAIN-BATCHED Linear arm onto
    // `propagate_linear_pollable_cpu` (certified f64 contraction, `γ_n^{f64}·S`),
    // while the direct joint-spec CROWN below still runs
    // `propagate_linear_with_engine` (round-to-nearest f32 GEMM, `γ_n^{f32}·S`).
    // Pinning the engine no longer puts them on one arithmetic, because the
    // batched lane does not call the engine at all. The observed failure was
    // `lower_b max diff 1.4305115e-6` against the old flat `1e-6` — ~12 ULP on the
    // bias, exactly the `γ^{f32}`-vs-`γ^{f64}` scale at this depth.
    //
    // What the two lanes owe each other is therefore NOT ULP equality: it is
    // structural equivalence (same shapes, same spec composition, same bias
    // presence) and agreement within the certificate gap their two `γ_n`
    // factors imply. That gap is DERIVED in `child_parity_lane_gap_4116` from
    // this fixture's `|W|` matrices, its node boxes and the SAME `γ_n`
    // functions the production code uses — never fitted to the diff above. Do
    // not reinstate a "bit-exact" claim without first putting both lanes back on
    // one arithmetic.
    let engine = NaiveCpuGemmEngine;

    // Baseline: direct joint-spec CROWN (ground truth), same engine as the batched path.
    let (baseline_bounds, baseline_linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &child_input,
            &spec_matrix,
            Some(&engine),
            &root_node_bounds,
            None,
        )
        .expect("direct joint-spec CROWN should succeed");

    // Dense-spec batched helper with one domain.
    let spec_result = GraphDomainBatchExecutor::execute_dense_specs(DenseSpecBatchRequest {
        graph: &graph,
        input_bounds_batch: &[&child_input],
        spec_matrix: &spec_matrix,
        engine: Some(&engine),
        alpha_node_bounds: Some(&root_node_bounds),
        alpha_state: None,
        mul_binary_alphas: None,
        deadline: None,
        crown_backward_layers: None,
        ibp_enhancement: false,
        stacked_rebound: false,
        nested_deadline: None,
    })
    .expect("dense-spec batched helper should accept multi-row spec matrices");

    assert_eq!(
        spec_result.bounds.len(),
        1,
        "one domain should produce one result"
    );

    let gap = child_parity_lane_gap_4116(&spec_matrix, &root_node_bounds, &child_input);

    // SOUNDNESS FIRST. An agreement check between two lanes cannot see a defect
    // that moved BOTH of them, and if it runs first it also masks the soundness
    // check behind it: a tightening defect is a disagreement too, so the
    // agreement assertion fires and the soundness assertion is never reached.
    // Both lanes must be SOUND, not merely in agreement.
    assert_spec_box_contains_forward_grid_4116(
        "dense-spec batched",
        &spec_result.bounds[0],
        &spec_matrix,
        &child_input,
    );
    assert_spec_box_contains_forward_grid_4116(
        "direct joint-spec CROWN",
        &baseline_bounds,
        &spec_matrix,
        &child_input,
    );

    // Verify output bounds agree within the derived cross-lane certificate gap.
    assert_flat_bounds_within(
        "dense-spec batched vs direct joint-spec CROWN",
        &spec_result.bounds[0],
        &baseline_bounds,
        gap.affine,
    );

    // Verify linear bounds match when both are present.
    match (&spec_result.linear_bounds[0], &baseline_linear) {
        (Some(actual), Some(expected)) => assert_linear_bounds_within_gap(
            "dense-spec batched vs direct joint-spec CROWN",
            actual,
            expected,
            &gap,
            &child_input,
        ),
        (None, None) => {}
        (actual, expected) => panic!(
            "dense-spec batched/direct linear bound availability diverged: batched={} direct={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

#[test]
fn test_multi_objective_build_batch_size_preserves_status_4354() {
    let graph = build_multi_objective_status_parity_graph_4354();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = vec![-0.1_f32, -0.1_f32];

    let baseline_config = build_input_split_status_config_4354();
    let chunked_config = BetaCrownConfig {
        build_batch_size: Some(1),
        ..baseline_config.clone()
    };
    let baseline_verifier = BetaCrownVerifier::new(baseline_config);
    let chunked_verifier = BetaCrownVerifier::new(chunked_config);

    let baseline_result = baseline_verifier
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("baseline multi-objective input split should not error");
    let chunked_result = chunked_verifier
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("chunked multi-objective input split should not error");

    assert_eq!(
        std::mem::discriminant(&baseline_result.result),
        std::mem::discriminant(&chunked_result.result),
        "build_batch_size should preserve the multi-objective verifier status: baseline={:?}, chunked={:?}",
        baseline_result.result,
        chunked_result.result,
    );
    assert_eq!(
        baseline_result.domains_explored, chunked_result.domains_explored,
        "build_batch_size should not perturb the explored-domain count on the parity harness"
    );
    assert_eq!(
        baseline_result.domains_verified, chunked_result.domains_verified,
        "build_batch_size should preserve verified-domain accounting"
    );
}

/// Two-domain dense-spec parity: two independent domains through the dense-spec
/// batched helper match two independent direct calls.
///
/// Part of #4116 Packet A Step 5.
#[test]
fn test_dense_spec_batched_two_domains_match_independent_calls_4116() {
    let graph = build_multi_objective_child_parity_graph();
    let root_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid root input");
    let root_node_bounds = graph
        .collect_node_bounds(&root_input)
        .expect("root node bounds should succeed");
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32], [-0.6_f32, 1.0_f32]]);

    // Two different child domains.
    let child_a = BoundedTensor::new(
        arr1(&[-0.35_f32, -0.65_f32]).into_dyn(),
        arr1(&[0.55_f32, 0.15_f32]).into_dyn(),
    )
    .expect("valid child_a");
    let child_b = BoundedTensor::new(
        arr1(&[-0.10_f32, -0.40_f32]).into_dyn(),
        arr1(&[0.80_f32, 0.50_f32]).into_dyn(),
    )
    .expect("valid child_b");

    // Independent direct calls (ground truth).
    let (baseline_a, _) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &child_a,
            &spec_matrix,
            None,
            &root_node_bounds,
            None,
        )
        .expect("direct CROWN for child_a should succeed");
    let (baseline_b, _) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &child_b,
            &spec_matrix,
            None,
            &root_node_bounds,
            None,
        )
        .expect("direct CROWN for child_b should succeed");

    // Batched call with both domains.
    let spec_result = GraphDomainBatchExecutor::execute_dense_specs(DenseSpecBatchRequest {
        graph: &graph,
        input_bounds_batch: &[&child_a, &child_b],
        spec_matrix: &spec_matrix,
        engine: None,
        alpha_node_bounds: Some(&root_node_bounds),
        alpha_state: None,
        mul_binary_alphas: None,
        deadline: None,
        crown_backward_layers: None,
        ibp_enhancement: false,
        stacked_rebound: false,
        nested_deadline: None,
    })
    .expect("dense-spec batched helper with two domains should succeed");

    assert_eq!(
        spec_result.bounds.len(),
        2,
        "two domains should produce two results"
    );

    assert_flat_bounds_close(
        "batched domain 0 vs independent direct call",
        &spec_result.bounds[0],
        &baseline_a,
    );
    assert_flat_bounds_close(
        "batched domain 1 vs independent direct call",
        &spec_result.bounds[1],
        &baseline_b,
    );
}

#[test]
fn test_dense_spec_batched_mul_binary_alphas_two_domains_match_independent_calls_4284() {
    let (graph, spec_matrix, mul_binary_alphas, child_a, child_b) =
        build_mul_binary_dense_spec_batch_fixture_4284();

    let baseline_a =
        direct_mul_binary_bounds_4284(&graph, &child_a, &spec_matrix, &mul_binary_alphas);
    let baseline_b =
        direct_mul_binary_bounds_4284(&graph, &child_b, &spec_matrix, &mul_binary_alphas);

    let spec_result = GraphDomainBatchExecutor::execute_dense_specs(DenseSpecBatchRequest {
        graph: &graph,
        input_bounds_batch: &[&child_a, &child_b],
        spec_matrix: &spec_matrix,
        engine: None,
        alpha_node_bounds: None,
        alpha_state: None,
        mul_binary_alphas: Some(&mul_binary_alphas),
        deadline: None,
        crown_backward_layers: None,
        ibp_enhancement: false,
        stacked_rebound: false,
        nested_deadline: None,
    })
    .expect("dense-spec batched helper should support shared mulbinary alphas");

    assert_mul_binary_dense_spec_matches_direct_4284(&spec_result, &baseline_a, &baseline_b);
}

#[test]
fn test_dense_spec_batched_capture_matches_direct_cache_4403() {
    fn assert_array_close_4403<D: ndarray::Dimension>(
        node_name: &str,
        field: &str,
        actual: &ndarray::Array<f32, D>,
        expected: &ndarray::Array<f32, D>,
    ) {
        assert_eq!(
            actual.shape(),
            expected.shape(),
            "{node_name}: {field} shape changed"
        );

        // The direct and domain-batched graph coordinators publish directed
        // endpoints and discharge certified coefficient error at different
        // boundaries. The underlying layer dispatch is bit-identical (checked
        // below), but those graph-level publication points can accumulate a few
        // binary32 ULPs across this five-node backward chain. Use a local-scale
        // ULP budget rather than a magnitude-independent decimal tolerance.
        //
        // The unit-scale floor also handles cancellation near zero: endpoint
        // error is governed by the scale of the accumulated terms, not by the
        // tiny residual alone. Sixteen ULPs is deliberately small enough to
        // catch a changed relaxation or dropped error term while covering the
        // observed directed-publication depth (currently at most nine ULPs).
        const MAX_ULPS: f32 = 16.0;
        for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            if actual.to_bits() == expected.to_bits() {
                continue;
            }
            assert!(
                actual.is_finite() && expected.is_finite(),
                "{node_name}: {field}[{index}] non-finite mismatch: \
                 actual={actual}, expected={expected}"
            );
            let scale = actual.abs().max(expected.abs()).max(1.0);
            let next = f32::from_bits(scale.to_bits() + 1);
            let ulp = if next.is_finite() {
                next - scale
            } else {
                scale - f32::from_bits(scale.to_bits() - 1)
            };
            let tolerance = MAX_ULPS * ulp;
            let difference = (actual - expected).abs();
            assert!(
                difference <= tolerance,
                "{node_name}: {field}[{index}] differs by {difference}: \
                 actual={actual}, expected={expected}, tolerance={tolerance} \
                 ({MAX_ULPS} ULPs at scale {scale})"
            );
        }
    }

    fn assert_node_linear_bounds_match_4403(
        node_name: &str,
        actual: &LinearBounds,
        expected: &LinearBounds,
    ) {
        assert_array_close_4403(node_name, "lower_a", actual.lower_a(), expected.lower_a());
        assert_array_close_4403(node_name, "lower_b", actual.lower_b(), expected.lower_b());
        assert_array_close_4403(node_name, "upper_a", actual.upper_a(), expected.upper_a());
        assert_array_close_4403(node_name, "upper_b", actual.upper_b(), expected.upper_b());
    }

    fn assert_array_bits_equal_4403<D: ndarray::Dimension>(
        label: &str,
        actual: &ndarray::Array<f32, D>,
        expected: &ndarray::Array<f32, D>,
    ) {
        assert_eq!(actual.shape(), expected.shape(), "{label}: shape changed");
        for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label}[{index}] bit mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    fn assert_node_linear_bounds_bits_equal_4403(
        node_name: &str,
        actual: &LinearBounds,
        expected: &LinearBounds,
    ) {
        assert_array_bits_equal_4403(
            &format!("{node_name} lower_a"),
            actual.lower_a(),
            expected.lower_a(),
        );
        assert_array_bits_equal_4403(
            &format!("{node_name} lower_b"),
            actual.lower_b(),
            expected.lower_b(),
        );
        assert_array_bits_equal_4403(
            &format!("{node_name} upper_a"),
            actual.upper_a(),
            expected.upper_a(),
        );
        assert_array_bits_equal_4403(
            &format!("{node_name} upper_b"),
            actual.upper_b(),
            expected.upper_b(),
        );
        match (actual.lower_a_err(), expected.lower_a_err()) {
            (Some(actual), Some(expected)) => {
                assert_array_bits_equal_4403(&format!("{node_name} lower_a_err"), actual, expected)
            }
            (None, None) => {}
            _ => panic!("{node_name}: lower_a_err availability changed"),
        }
        match (actual.upper_a_err(), expected.upper_a_err()) {
            (Some(actual), Some(expected)) => {
                assert_array_bits_equal_4403(&format!("{node_name} upper_a_err"), actual, expected)
            }
            (None, None) => {}
            _ => panic!("{node_name}: upper_a_err availability changed"),
        }
    }

    let graph = build_multi_objective_child_parity_graph();
    let root_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid root input");
    let child_input = BoundedTensor::new(
        arr1(&[-0.35_f32, -0.65_f32]).into_dyn(),
        arr1(&[0.55_f32, 0.15_f32]).into_dyn(),
    )
    .expect("valid child input");
    let root_node_bounds = graph
        .collect_node_bounds(&root_input)
        .expect("root node bounds should succeed");
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32], [-0.6_f32, 1.0_f32]]);

    // Pin every path in this test — direct cache, direct final-linear, the batched
    // standard/capture passes, and the layer-level scalar/batched dispatch — to the
    // SAME f32 GEMM engine. This removes backend-specific coefficient differences.
    // The complete graph paths are not bit-identical, however: their coordinators
    // discharge certified coefficient errors and publish directed f32 bias endpoints
    // at different boundaries. The comparison above therefore uses a small,
    // scale-aware ULP budget; the final scalar/batched Linear dispatch isolates and
    // checks the underlying layer implementation separately.
    let engine = NaiveCpuGemmEngine;

    let (_direct_bounds, direct_cache_opt) = graph
        .propagate_crown_with_specs_and_node_bounds_and_cache_and_deadline(
            &child_input,
            &spec_matrix,
            Some(&engine),
            &root_node_bounds,
            None,
        )
        .expect("direct joint-spec CROWN with cache should succeed");
    let direct_cache = direct_cache_opt.expect("direct path should capture cached lA");
    let (_direct_output_bounds, direct_input_linear_opt) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &child_input,
            &spec_matrix,
            Some(&engine),
            &root_node_bounds,
            None,
        )
        .expect("direct joint-spec CROWN with final input linear should succeed");
    let direct_input_linear =
        direct_input_linear_opt.expect("direct path should preserve final input linear bounds");

    let mut builder =
        BatchedDomainsBuilder::new_with_options(Vec::new(), BatchedDomainOptions::default());
    let empty_layer_bounds: HashMap<String, (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> =
        HashMap::new();
    builder.add_domain(
        &empty_layer_bounds,
        child_input.lower().clone(),
        child_input.upper().clone(),
        0.0,
        0.0,
        0,
        Vec::new(),
    );
    let batched = builder.build().expect("batched dense-spec domain");

    let shared_root_bounds: HashMap<String, Arc<BoundedTensor>> = root_node_bounds
        .iter()
        .map(|(name, bounds)| (name.clone(), Arc::new(bounds.clone())))
        .collect();
    let empty_history = GraphSplitHistory::new();
    let ctx = BatchedBackwardContext {
        batched: &batched,
        histories: vec![&empty_history],
        beta_states: vec![None],
        base_bounds: vec![Some(&shared_root_bounds)],
        delta_seeds: vec![None],
        alpha_states: vec![None],
        cached_la: vec![None],
        mul_binary_alphas: None,
    };

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let standard_results = verifier
        .propagate_crown_batched_with_context_specs(&graph, &ctx, &spec_matrix, &engine)
        .expect("batched dense-spec standard path should succeed");
    assert_eq!(
        standard_results.len(),
        1,
        "single-domain standard batched path should return one result"
    );
    let standard_input_linear = standard_results[0]
        .input_linear
        .as_ref()
        .expect("standard batched path should preserve final input linear bounds");
    assert_node_linear_bounds_match_4403(
        "standard _input",
        standard_input_linear,
        &direct_input_linear,
    );

    let batched_result = verifier
        .propagate_crown_batched_with_context_specs_capture_la(&graph, &ctx, &spec_matrix, &engine)
        .expect("batched dense-spec capture should succeed");
    let mut batched_caches = batched_result
        .intermediate_la
        .expect("batched path should capture cached lA");
    let batched_cache = batched_caches
        .pop()
        .expect("single-domain batched capture should produce one cache");
    let batched_input_linear = batched_result.results[0]
        .input_linear
        .as_ref()
        .expect("batched path should preserve final input linear bounds");

    for node_name in ["linear3", "relu2", "linear2", "relu1", "linear1"] {
        let actual = batched_cache
            .get(node_name)
            .unwrap_or_else(|| panic!("batched cache missing node {node_name}"));
        let expected = direct_cache
            .linear_bounds(node_name)
            .unwrap_or_else(|| panic!("direct cache missing node {node_name}"));
        assert_node_linear_bounds_match_4403(node_name, actual, &expected);
    }

    assert_node_linear_bounds_match_4403("_input", batched_input_linear, &direct_input_linear);

    let linear1_bounds = direct_cache
        .linear_bounds("linear1")
        .expect("direct cache should contain linear1");
    let linear1_layer = match &graph
        .nodes
        .get("linear1")
        .expect("graph should contain linear1")
        .layer
    {
        Layer::Linear(layer) => layer,
        other => panic!("linear1 should be Linear, got {}", other.layer_type()),
    };
    let scalar_input_linear = linear1_layer
        .propagate_linear_with_engine(&linear1_bounds, Some(&engine))
        .expect("scalar linear1 dispatch should succeed")
        .into_owned();
    let batched_input_linear_from_linear1 = linear1_layer
        .propagate_linear_batched_with_engine(&[&linear1_bounds], &engine)
        .expect("batched linear1 dispatch should succeed");
    assert_eq!(
        batched_input_linear_from_linear1.len(),
        1,
        "single-domain linear1 batched dispatch should return one result"
    );
    assert_node_linear_bounds_bits_equal_4403(
        "linear1->_input dispatch",
        &batched_input_linear_from_linear1[0],
        &scalar_input_linear,
    );
}
