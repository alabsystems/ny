// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Full-graph DAG CROWN parity regressions for deterministic Where routing (#3676).

use crate::types::BoundsProvenance;
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_test_utils::assert_bounded_tensor_close;

fn build_deterministic_where_graph_3676(cond_value: f32) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let identity = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::Linear(LinearLayer::new(identity, None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond_base",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
            IxDyn(&[]),
            cond_value,
        ))),
        vec!["cond_base".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5]).into_dyn(),
        arr1(&[2.0_f32, 1.5]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

fn build_mixed_where_matrix_graph_3676() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["x".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0_f32, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 1.0]).unwrap(),
    )
    .unwrap();
    (graph, input)
}

fn assert_where_tighter_than_ibp_3676(
    ibp_out: &BoundedTensor,
    crown_out: &BoundedTensor,
    label: &str,
) {
    let ibp_out = ibp_out.flatten();
    let crown_out = crown_out.flatten();
    let mut strictly_tighter = false;

    for dim in 0..ibp_out.lower().len() {
        let ibp_lower = ibp_out.lower()[[dim]];
        let ibp_upper = ibp_out.upper()[[dim]];
        let crown_lower = crown_out.lower()[[dim]];
        let crown_upper = crown_out.upper()[[dim]];
        assert!(
            crown_lower >= ibp_lower - 1e-5,
            "{label} lower[{dim}] must be no looser than IBP: crown={} ibp={}",
            crown_lower,
            ibp_lower
        );
        assert!(
            crown_upper <= ibp_upper + 1e-5,
            "{label} upper[{dim}] must be no looser than IBP: crown={} ibp={}",
            crown_upper,
            ibp_upper
        );
        strictly_tighter |= crown_lower > ibp_lower + 1e-5 || crown_upper < ibp_upper - 1e-5;
    }

    assert!(
        strictly_tighter,
        "{label} should tighten deterministic Where over the IBP union"
    );
}

fn assert_dag_where_exact_3676(cond_value: f32, expected_branch: &str, label: &str) {
    let (graph, input) = build_deterministic_where_graph_3676(cond_value);
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let ibp_out = ibp_bounds.get("out").unwrap().clone();
    let expected = ibp_bounds.get(expected_branch).unwrap().clone();

    let result = graph
        .propagate_crown_with_provenance(&input)
        .expect("DAG-CROWN should succeed on deterministic Where");

    assert_eq!(
        result.provenance,
        BoundsProvenance::Crown,
        "{label} should stay on the DAG-CROWN path"
    );
    assert_bounded_tensor_close(&result.bounds, &expected, 1e-5, label);
    assert_where_tighter_than_ibp_3676(&ibp_out, &result.bounds, label);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_deterministic_true_keeps_crown_provenance_3676() {
    assert_dag_where_exact_3676(1.0, "x", "DAG deterministic-true Where");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_deterministic_false_keeps_crown_provenance_3676() {
    assert_dag_where_exact_3676(0.0, "y", "DAG deterministic-false Where");
}

/// Build a Where whose condition is a *constant* per-element mask (mixed
/// true/false, but bound-independent). True branch = x, false branch = y = -x.
/// With mask [1, 0]: out[0] = x[0], out[1] = -x[1]. The exact per-element select
/// must be reproduced by CROWN (not the loose IBP union).
fn build_const_mask_where_graph(mask: &[f32]) -> (GraphNetwork, BoundedTensor) {
    let n = mask.len();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));
    // cond = 0 * x + mask  => a constant (bound-independent) per-element mask.
    graph.add_node(GraphNode::new(
        "cond_base",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), mask.to_vec()).unwrap(),
        )),
        vec!["cond_base".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), vec![-1.0_f32, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), vec![2.0_f32, 1.5]).unwrap(),
    )
    .unwrap();
    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_constant_mask_exact_select_tighter_than_ibp() {
    // mask = [1, 0]: out[0] = x[0] in [-1, 2]; out[1] = y[1] = -x[1] in [-1.5, -0.5].
    let (graph, input) = build_const_mask_where_graph(&[1.0, 0.0]);
    let ibp_out = graph.propagate_ibp(&input).unwrap();

    let result = graph
        .propagate_crown_with_provenance(&input)
        .expect("constant-mask Where should stay on CROWN");
    assert_eq!(result.provenance, BoundsProvenance::Crown);

    let out = result.bounds.flatten();
    // Exact per-element select.
    assert!(
        (out.lower()[[0]] - (-1.0)).abs() < 1e-5,
        "out0 lower {}",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[0]] - 2.0).abs() < 1e-5,
        "out0 upper {}",
        out.upper()[[0]]
    );
    assert!(
        (out.lower()[[1]] - (-1.5)).abs() < 1e-5,
        "out1 lower {}",
        out.lower()[[1]]
    );
    assert!(
        (out.upper()[[1]] - (-0.5)).abs() < 1e-5,
        "out1 upper {}",
        out.upper()[[1]]
    );

    // And strictly tighter than the IBP union on at least one bound.
    assert_where_tighter_than_ibp_3676(&ibp_out, &result.bounds, "constant-mask Where");
}

/// Build a constant-mask Where over an `n`-dim network input `z`:
///   x = z, y = (-1) * z, cond = constant `mask`.
/// Output: out[i] = z[i] if mask[i] else -z[i]. Network input box = [lo, hi].
fn build_const_mask_where_graph_n(
    mask: &[f32],
    lo: &[f32],
    hi: &[f32],
) -> (GraphNetwork, BoundedTensor) {
    let n = mask.len();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond_base",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), mask.to_vec()).unwrap(),
        )),
        vec!["cond_base".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lo.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), hi.to_vec()).unwrap(),
    )
    .unwrap();
    (graph, input)
}

/// Dense-sampling soundness for the constant-condition Where CROWN relaxation.
///
/// For many random constant 0/1 masks and random input boxes, the concretized
/// CROWN output bound must contain the *true* elementwise Where output
/// (out[i] = z[i] if mask[i] else -z[i]) at every sampled interior point AND at
/// every box corner. A wrong verdict is disqualifying, so this is the soundness
/// gate for the exact per-element select. We also require the CROWN bound to be
/// no looser than the IBP union (it is in fact exact here).
#[ntest::timeout(120000)]
#[test]
fn test_graph_crown_where_constant_mask_dense_sampling_soundness() {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = SmallRng::seed_from_u64(0x57_4845_5245); // "WHERE"
    const TRIALS: usize = 60;
    const SAMPLES_PER_TRIAL: usize = 64;

    for trial in 0..TRIALS {
        let n: usize = rng.random_range(1..=5);
        let mask: Vec<f32> = (0..n)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { 0.0 })
            .collect();
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for _ in 0..n {
            let a: f32 = rng.random_range(-3.0..3.0);
            let b: f32 = rng.random_range(-3.0..3.0);
            lo.push(a.min(b));
            hi.push(a.max(b));
        }

        let (graph, input) = build_const_mask_where_graph_n(&mask, &lo, &hi);

        let result = graph
            .propagate_crown_with_provenance(&input)
            .expect("constant-mask Where should stay on CROWN");
        assert_eq!(
            result.provenance,
            BoundsProvenance::Crown,
            "trial {trial}: constant-mask Where must stay on the CROWN path"
        );
        let out = result.bounds.flatten();
        let ibp = graph.propagate_ibp(&input).unwrap().flatten();

        // Helper: true Where output at concrete point z.
        let true_out = |z: &[f32]| -> Vec<f32> {
            (0..n)
                .map(|i| if mask[i] >= 0.5 { z[i] } else { -z[i] })
                .collect()
        };

        let check = |z: &[f32], label: &str| {
            let y = true_out(z);
            for i in 0..n {
                assert!(
                    y[i] >= out.lower()[[i]] - 1e-4 && y[i] <= out.upper()[[i]] + 1e-4,
                    "trial {trial} {label}: out[{i}]={} not in CROWN bound \
                     [{}, {}] (mask={}, z={:?})",
                    y[i],
                    out.lower()[[i]],
                    out.upper()[[i]],
                    mask[i],
                    z,
                );
            }
        };

        // All 2^n box corners (n <= 5 => <= 32 corners).
        for corner_bits in 0..(1usize << n) {
            let z: Vec<f32> = (0..n)
                .map(|i| {
                    if (corner_bits >> i) & 1 == 1 {
                        hi[i]
                    } else {
                        lo[i]
                    }
                })
                .collect();
            check(&z, "corner");
        }
        // Random interior points.
        for _ in 0..SAMPLES_PER_TRIAL {
            let z: Vec<f32> = (0..n)
                .map(|i| {
                    if (hi[i] - lo[i]).abs() < 1e-9 {
                        lo[i]
                    } else {
                        rng.random_range(lo[i]..hi[i])
                    }
                })
                .collect();
            check(&z, "interior");
        }

        // CROWN must be no looser than the IBP union (sanity on tightness side).
        for i in 0..n {
            assert!(
                out.lower()[[i]] >= ibp.lower()[[i]] - 1e-4,
                "trial {trial}: CROWN lower[{i}] looser than IBP"
            );
            assert!(
                out.upper()[[i]] <= ibp.upper()[[i]] + 1e-4,
                "trial {trial}: CROWN upper[{i}] looser than IBP"
            );
        }
    }
}

/// Parity test: a DATA-DEPENDENT condition must NOT be tightened — CROWN must
/// reproduce the sound IBP convex-hull bound exactly (no exact-select path). Here
/// cond = z[0] (a non-degenerate interval that straddles the 0.5 boundary), so
/// `where_constant_mask` returns None and the loose concretize fallback is used.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_data_dependent_cond_matches_ibp_hull() {
    let mut graph = GraphNetwork::new();
    // x = z, y = -z, cond = z (data-dependent: spans below and above 0.5).
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0_f32, 1.0]).unwrap(),
    )
    .unwrap();

    let ibp_out = graph.propagate_ibp(&input).unwrap();
    let result = graph.propagate_crown_with_provenance(&input).unwrap();
    // Data-dependent cond => CROWN concretizes to the IBP union (sound, not tighter).
    assert_bounded_tensor_close(
        &result.bounds,
        &ibp_out,
        1e-5,
        "data-dependent Where must match the IBP convex hull",
    );
}

/// Graph CROWN backward must SUCCEED (not InvalidSpec) on an embedded-constant
/// Where node (single `cond` input; both branches embedded constants), and the
/// bound must be sound and — when `cond` is constant — exactly the selected
/// branch constant. Regression for the `require_ternary_inputs` failure on the
/// 1-input embedded form (WhereLayer::const_true/const_false).
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_embedded_constants_constant_cond_exact_and_sound() {
    // cond = 0*z + mask (constant per-element mask), true=[10,20], false=[-10,-20].
    // out[i] = 10/20 where mask true, -10/-20 where mask false.
    let mask = [1.0_f32, 0.0];
    let const_true = ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0_f32, 20.0]).unwrap();
    let const_false = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-10.0_f32, -20.0]).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), mask.to_vec()).unwrap(),
        )),
        vec!["z".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::with_constants(
            Some(const_true),
            Some(const_false),
        )),
        vec!["cond".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).unwrap(),
    )
    .unwrap();

    let result = graph
        .propagate_crown_with_provenance(&input)
        .expect("embedded-constant Where must not fail CROWN backward");
    assert_eq!(
        result.provenance,
        BoundsProvenance::Crown,
        "embedded-constant Where should stay on the CROWN path"
    );
    let out = result.bounds.flatten();
    // Exact constant select (tighter than the IBP union [-10,10],[-20,20]).
    assert!(
        (out.lower()[[0]] - 10.0).abs() < 1e-4,
        "out0 lo {}",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[0]] - 10.0).abs() < 1e-4,
        "out0 hi {}",
        out.upper()[[0]]
    );
    assert!(
        (out.lower()[[1]] - (-20.0)).abs() < 1e-4,
        "out1 lo {}",
        out.lower()[[1]]
    );
    assert!(
        (out.upper()[[1]] - (-20.0)).abs() < 1e-4,
        "out1 hi {}",
        out.upper()[[1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_mixed_matrix_concretizes_without_shape_error_3676() {
    let (graph, input) = build_mixed_where_matrix_graph_3676();
    let ibp_out = graph.propagate_ibp(&input).unwrap();

    let result = graph
        .propagate_crown_with_provenance(&input)
        .expect("Mixed 2D Where should concretize instead of failing shape conversion");

    assert_eq!(
        result.provenance,
        BoundsProvenance::Crown,
        "Mixed 2D Where should remain on the DAG-CROWN path"
    );
    assert_bounded_tensor_close(
        &result.bounds,
        &ibp_out,
        1e-5,
        "Mixed 2D Where should concretize to the IBP union without shape errors",
    );
}
