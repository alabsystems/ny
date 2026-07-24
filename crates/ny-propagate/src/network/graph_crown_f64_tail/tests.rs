// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness tests for the f64 tail pass (docs/LSNC_F64_TAIL_DESIGN.md §6.6):
//! exact-integer reference on a pure-linear net (the f64 bound must enclose
//! the EXACT min and beat the f32 certified floor), sampled containment on a
//! ReLU + MulBinary net, corner-repair unit tests (including a deliberately
//! corrupted plane), and fail-closed declines.

use std::collections::HashMap;

use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::*;
use crate::layers::{AddConstantLayer, LinearLayer, MulBinaryLayer, ReLULayer, SubLayer};
use crate::network::core::{GraphNetwork, GraphNode};

fn bt(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lo.len()]), lo.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[hi.len()]), hi.to_vec()).unwrap(),
    )
    .expect("finite bounds")
}

/// Pure-linear 2-layer net with INTEGER weights of magnitude ~2^10 and heavy
/// cancellation. Every composite coefficient and the true min over an
/// integer-endpoint box are EXACTLY representable in f64 (products < 2^53),
/// so the reference below is exact-rational-grade.
fn build_cancellation_linear_net() -> GraphNetwork {
    // w1: 3x2, integers with large magnitude.
    let w1 = arr2(&[[1024.0_f32, -1023.0], [-2048.0, 2047.0], [512.0, 511.0]]);
    let b1 = arr1(&[1.0_f32, -1.0, 2.0]);
    // w2: 2x3, cancels most of w1's magnitude.
    let w2 = arr2(&[[1023.0_f32, 511.0, -2044.0], [-1024.0, -512.0, 2048.0]]);
    let b2 = arr1(&[0.0_f32, 1.0]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("l1")),
    ));
    graph.add_node(GraphNode::new(
        "l2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("l2")),
        vec!["l1".to_string()],
    ));
    graph.set_output("l2");
    graph
}

/// Exact composite spec-row coefficients + bias for the pure-linear net,
/// computed in integer-exact f64 arithmetic.
fn exact_linear_row(row: &[f64]) -> (Vec<f64>, f64) {
    let w1 = [[1024.0_f64, -1023.0], [-2048.0, 2047.0], [512.0, 511.0]];
    let b1 = [1.0_f64, -1.0, 2.0];
    let w2 = [[1023.0_f64, 511.0, -2044.0], [-1024.0, -512.0, 2048.0]];
    let b2 = [0.0_f64, 1.0];
    // row·(W2·(W1·x + b1) + b2) = (row·W2·W1)·x + row·(W2·b1 + b2)
    let mut rw2 = [0.0_f64; 3];
    let mut bias = 0.0_f64;
    for (o, &r) in row.iter().enumerate() {
        bias += r * b2[o];
        for k in 0..3 {
            rw2[k] += r * w2[o][k];
        }
    }
    let mut coeff = [0.0_f64; 2];
    for k in 0..3 {
        bias += rw2[k] * b1[k];
        for j in 0..2 {
            coeff[j] += rw2[k] * w1[k][j];
        }
    }
    (coeff.to_vec(), bias)
}

#[test]
fn linear_net_bound_encloses_exact_reference() {
    // Integer weights: EVERY quantity in the reference below is exactly
    // representable (products < 2^53, sums integer), so `exact_min` is an
    // exact-rational-grade reference (design §6.6a).
    let graph = build_cancellation_linear_net();
    let input = bt(&[-1.0, -1.0], &[1.0, 1.0]);
    let spec = arr2(&[[1.0_f32, 1.0], [0.0, 1.0]]);
    let node_bounds =
        crate::network::collect_intermediate_bounds(&graph, &input, None, None).expect("anchors");

    let outcome = f64_tail_verify(
        &graph,
        &input,
        &spec,
        &[-1e9_f32, -1e9],
        &[1, 1],
        None,
        Some(&node_bounds),
        None,
        None,
    );
    let row_lowers = match outcome {
        F64TailOutcome::Verified { ref row_lowers } => row_lowers.clone(),
        ref other => panic!("expected Verified against -1e9 thresholds, got {other:?}"),
    };

    for (r, &l_cert) in row_lowers.iter().enumerate() {
        let row: Vec<f64> = spec.row(r).iter().map(|&v| f64::from(v)).collect();
        let (coeff, bias) = exact_linear_row(&row);
        // Exact min of coeff·x + bias over [-1,1]^2: bias - Σ|coeff| (exact
        // integer-scale arithmetic, no rounding).
        let exact_min = bias - coeff.iter().map(|c| c.abs()).sum::<f64>();
        assert!(
            l_cert <= exact_min,
            "row {r}: certified lower {l_cert} must not exceed the EXACT min {exact_min}"
        );
        // The envelope is deliberately over-counted (per-push node-range
        // discharge + global gamma slop); at this fixture's 2^10/4e3 scales it
        // lands ~1e-6, at lsnc scales (~1e2 intermediates) it is ~1e-8 —
        // orders below the 1e-6 clearance the Lyapunov rows need.
        assert!(
            exact_min - l_cert < 1e-4,
            "row {r}: f64 envelope should be tiny at this scale: exact {exact_min} vs {l_cert}"
        );
    }
}

/// Full-mantissa weights of ~2^10 magnitude with near-cancelling `row·W2`
/// entries: the f32 backward's certified coefficient-error penalty (γ·S with
/// S ~ 1e3-1e4, amplified through W1) is material (~1e-1..1), while the f64
/// pass's envelope is ~1e-12 — the design §1.1 "f32 storage tax" in miniature.
fn build_f32_tax_linear_net() -> GraphNetwork {
    let w1 = arr2(&[
        [1013.77_f32, -1013.912],
        [-2027.54, 2027.108],
        [507.33, 506.77],
    ]);
    let b1 = arr1(&[0.013_f32, -0.027, 0.045]);
    let w2 = arr2(&[[1013.9_f32, 507.1, -2026.8], [-1013.55, -507.9, 2027.4]]);
    let b2 = arr1(&[0.0_f32, 0.5]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("l1")),
    ));
    graph.add_node(GraphNode::new(
        "l2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("l2")),
        vec!["l1".to_string()],
    ));
    graph.set_output("l2");
    graph
}

#[test]
fn f32_tax_net_bound_is_sound_and_beats_f32_floor() {
    let graph = build_f32_tax_linear_net();
    let input = bt(&[-1.0, -1.0], &[1.0, 1.0]);
    let spec = arr2(&[[1.0_f32, 1.0]]);
    let node_bounds =
        crate::network::collect_intermediate_bounds(&graph, &input, None, None).expect("anchors");

    let outcome = f64_tail_verify(
        &graph,
        &input,
        &spec,
        &[-1e9_f32],
        &[1],
        None,
        Some(&node_bounds),
        None,
        None,
    );
    let l_cert = match outcome {
        F64TailOutcome::Verified { row_lowers } => row_lowers[0],
        other => panic!("expected Verified against -1e9 threshold, got {other:?}"),
    };

    // Soundness: a linear function attains its min at a box corner — check
    // all 4 corners with an f64 forward evaluation.
    let w1 = [
        [1013.77_f64, -1013.912],
        [-2027.54, 2027.108],
        [507.33, 506.77],
    ];
    let b1 = [0.013_f64, -0.027, 0.045];
    let w2 = [[1013.9_f64, 507.1, -2026.8], [-1013.55, -507.9, 2027.4]];
    let b2 = [0.0_f64, 0.5];
    // Widen the constants exactly as the net stores them (f32 -> f64).
    let w1: Vec<Vec<f64>> = w1
        .iter()
        .map(|row| row.iter().map(|&v| f64::from(v as f32)).collect())
        .collect();
    let w2: Vec<Vec<f64>> = w2
        .iter()
        .map(|row| row.iter().map(|&v| f64::from(v as f32)).collect())
        .collect();
    let b1: Vec<f64> = b1.iter().map(|&v| f64::from(v as f32)).collect();
    let b2: Vec<f64> = b2.iter().map(|&v| f64::from(v as f32)).collect();
    let mut corner_min = f64::INFINITY;
    for &x0 in &[-1.0_f64, 1.0] {
        for &x1 in &[-1.0_f64, 1.0] {
            let h: Vec<f64> = (0..3)
                .map(|k| w1[k][0] * x0 + w1[k][1] * x1 + b1[k])
                .collect();
            let y: Vec<f64> = (0..2)
                .map(|o| w2[o][0] * h[0] + w2[o][1] * h[1] + w2[o][2] * h[2] + b2[o])
                .collect();
            corner_min = corner_min.min(y[0] + y[1]);
        }
    }
    assert!(
        l_cert <= corner_min + 1e-9,
        "certified lower {l_cert} must not exceed the corner min {corner_min}"
    );

    // Floor-beat: the f32 lane's certified bound over the SAME anchors must
    // be materially looser (the tax this design removes).
    let (f32_bounds, _) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &input,
            &spec,
            None,
            &node_bounds,
            None,
        )
        .expect("f32 spec backward");
    let f32_lower = f64::from(f32_bounds.flatten().lower()[[0]]);
    assert!(
        l_cert > f32_lower + 1e-6,
        "expected a material f32 tax: f64 {l_cert} vs f32 {f32_lower}"
    );
}

/// ReLU + MulBinary + Sub + AddConstant net: sampled-containment soundness.
/// input(2) -> l1(2->2) -> relu -> {left = relu, right = l2(2->2)}
///   -> mul = left * right -> sub = mul - relu -> out = sub + [c0, c1]
fn build_relu_mul_net() -> GraphNetwork {
    let w1 = arr2(&[[1.5_f32, -0.75], [0.5, 1.25]]);
    let b1 = arr1(&[0.25_f32, -0.5]);
    let w2 = arr2(&[[0.8_f32, -1.1], [1.3, 0.4]]);
    let b2 = arr1(&[-0.2_f32, 0.6]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("l1")),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "l2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("l2")),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        vec!["relu".to_string(), "l2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sub",
        Layer::Sub(SubLayer),
        vec!["mul".to_string(), "relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1_f32, -0.3]).unwrap(),
        )),
        vec!["sub".to_string()],
    ));
    graph.set_output("out");
    graph
}

/// f64 forward eval of the relu-mul net at a point (RN; sampling reference).
fn relu_mul_forward(x: &[f64; 2]) -> [f64; 2] {
    let w1 = [[1.5_f64, -0.75], [0.5, 1.25]];
    let b1 = [0.25_f64, -0.5];
    let w2 = [[0.8_f64, -1.1], [1.3, 0.4]];
    let b2 = [-0.2_f64, 0.6];
    let mut h = [0.0_f64; 2];
    for o in 0..2 {
        h[o] = w1[o][0] * x[0] + w1[o][1] * x[1] + b1[o];
        h[o] = h[o].max(0.0);
    }
    let mut g = [0.0_f64; 2];
    for o in 0..2 {
        g[o] = w2[o][0] * h[0] + w2[o][1] * h[1] + b2[o];
    }
    let c = [0.1_f64, -0.3];
    [h[0] * g[0] - h[0] + c[0], h[1] * g[1] - h[1] + c[1]]
}

#[test]
fn relu_mul_net_bound_is_sampled_contained() {
    let graph = build_relu_mul_net();
    let input = bt(&[-1.0, -0.5], &[0.75, 1.0]);
    let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, -1.0]]);
    let node_bounds =
        crate::network::collect_intermediate_bounds(&graph, &input, None, None).expect("anchors");

    // With and without MulBinary alphas (option sweep: alpha present/absent).
    let mut alphas: HashMap<String, Array2<f32>> = HashMap::new();
    alphas.insert("mul".to_string(), Array2::from_elem((2, 2), 0.3_f32));
    for alpha_opt in [None, Some(&alphas)] {
        let outcome = f64_tail_verify(
            &graph,
            &input,
            &spec,
            &[-1e9_f32, -1e9, -1e9],
            &[1, 1, 1],
            alpha_opt,
            Some(&node_bounds),
            None,
            None,
        );
        let row_lowers = match outcome {
            F64TailOutcome::Verified { ref row_lowers } => row_lowers.clone(),
            ref other => panic!("expected Verified against -1e9 thresholds, got {other:?}"),
        };

        // Dense grid + corners: every sampled spec value must be >= l_cert.
        let (lo, hi) = ([-1.0_f64, -0.5], [0.75_f64, 1.0]);
        let steps = 33usize;
        for r in 0..spec.nrows() {
            let row: Vec<f64> = spec.row(r).iter().map(|&v| f64::from(v)).collect();
            let l_cert = row_lowers[r];
            assert!(l_cert.is_finite(), "row {r}: expected a finite bound");
            let mut min_sample = f64::INFINITY;
            for i in 0..=steps {
                for j in 0..=steps {
                    let x = [
                        lo[0] + (hi[0] - lo[0]) * (i as f64) / (steps as f64),
                        lo[1] + (hi[1] - lo[1]) * (j as f64) / (steps as f64),
                    ];
                    let y = relu_mul_forward(&x);
                    let v = row[0] * y[0] + row[1] * y[1];
                    min_sample = min_sample.min(v);
                }
            }
            assert!(
                l_cert <= min_sample + 1e-12,
                "row {r} (alphas={}): certified lower {l_cert} above sampled min {min_sample}",
                alpha_opt.is_some()
            );
        }
    }
}

#[test]
fn verdict_mirrors_grouped_semantics() {
    let graph = build_relu_mul_net();
    let input = bt(&[-0.5, -0.25], &[0.5, 0.5]);
    let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let node_bounds =
        crate::network::collect_intermediate_bounds(&graph, &input, None, None).expect("anchors");

    // Impossible threshold on row 0 (single clause of 2 rows would pass via
    // row 1; force clause split so clause 0 = row 0 alone must fail).
    let outcome = f64_tail_verify(
        &graph,
        &input,
        &spec,
        &[1e9_f32, -1e9],
        &[1, 1],
        None,
        Some(&node_bounds),
        None,
        None,
    );
    match outcome {
        F64TailOutcome::NotVerified { min_gap_f64 } => {
            assert!(
                min_gap_f64 < 0.0 && min_gap_f64.is_finite(),
                "unreachable threshold must produce a finite negative gap, got {min_gap_f64}"
            );
        }
        other => panic!("expected NotVerified, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Corner-certify-and-repair gadget (design §5.4).
// ---------------------------------------------------------------------------

/// Dense-sample validity of a lower plane z >= a*x + b*y + nu over the box.
fn lower_plane_valid(alpha: f64, beta: f64, nu: f64, xl: f64, xu: f64, yl: f64, yu: f64) -> bool {
    let steps = 40;
    for i in 0..=steps {
        for j in 0..=steps {
            let x = xl + (xu - xl) * (i as f64) / (steps as f64);
            let y = yl + (yu - yl) * (j as f64) / (steps as f64);
            if x * y < alpha * x + beta * y + nu - 1e-12 {
                return false;
            }
        }
    }
    true
}

fn upper_plane_valid(alpha: f64, beta: f64, nu: f64, xl: f64, xu: f64, yl: f64, yu: f64) -> bool {
    let steps = 40;
    for i in 0..=steps {
        for j in 0..=steps {
            let x = xl + (xu - xl) * (i as f64) / (steps as f64);
            let y = yl + (yu - yl) * (j as f64) / (steps as f64);
            if x * y > alpha * x + beta * y + nu + 1e-12 {
                return false;
            }
        }
    }
    true
}

#[test]
fn corner_repair_keeps_valid_plane_nearly_unchanged() {
    // Exact McCormick L1 plane over an integer box: z >= yl*x + xl*y - xl*yl.
    let (xl, xu, yl, yu) = (1.0_f64, 2.0, 3.0, 4.0);
    let (alpha, beta, nu) = (yl, xl, -xl * yl);
    let repaired = repair_lower_plane(alpha, beta, nu, xl, xu, yl, yu).expect("finite repair");
    // Directed corner rounding may shave a few ulps but never more.
    assert!(repaired <= nu);
    assert!(
        nu - repaired < 1e-12,
        "valid plane over-repaired: {nu} -> {repaired}"
    );
    assert!(lower_plane_valid(alpha, beta, repaired, xl, xu, yl, yu));
}

#[test]
fn corner_repair_fixes_corrupted_lower_plane() {
    let (xl, xu, yl, yu) = (-1.5_f64, 0.5, -2.0, 1.0);
    // Start from the exact L2 facet, then CORRUPT the intercept upward by 0.1
    // (making the plane invalid: it cuts above z = x*y near a corner).
    let (alpha, beta) = (yu, xu);
    let nu_exact = -xu * yu;
    let nu_bad = nu_exact + 0.1;
    assert!(
        !lower_plane_valid(alpha, beta, nu_bad, xl, xu, yl, yu),
        "corruption must actually invalidate the plane"
    );
    let repaired = repair_lower_plane(alpha, beta, nu_bad, xl, xu, yl, yu).expect("finite repair");
    assert!(
        lower_plane_valid(alpha, beta, repaired, xl, xu, yl, yu),
        "repair must restore validity"
    );
    assert!(
        repaired <= nu_bad - 0.099,
        "repair must remove (at least) the injected corruption: {nu_bad} -> {repaired}"
    );
}

#[test]
fn corner_repair_fixes_corrupted_upper_plane() {
    let (xl, xu, yl, yu) = (-1.0_f64, 2.0, 0.5, 3.0);
    let (alpha, beta) = (yu, xl);
    let nu_exact = -xl * yu;
    let nu_bad = nu_exact - 0.25;
    assert!(!upper_plane_valid(alpha, beta, nu_bad, xl, xu, yl, yu));
    let repaired = repair_upper_plane(alpha, beta, nu_bad, xl, xu, yl, yu).expect("finite repair");
    assert!(
        upper_plane_valid(alpha, beta, repaired, xl, xu, yl, yu),
        "repair must restore validity"
    );
    assert!(repaired >= nu_bad + 0.249);
}

#[test]
fn corner_repair_declines_on_non_finite() {
    assert!(repair_lower_plane(1.0, 1.0, 0.0, f64::NEG_INFINITY, 1.0, 0.0, 1.0).is_none());
    assert!(repair_lower_plane(f64::NAN, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0).is_none());
    assert!(repair_upper_plane(1.0, f64::INFINITY, 0.0, 0.0, 1.0, 0.0, 1.0).is_none());
}

// ---------------------------------------------------------------------------
// Fail-closed declines.
// ---------------------------------------------------------------------------

#[test]
fn unsupported_op_declines_whole_pass() {
    // Tanh is outside the f64-tail op class -> the pass must decline.
    // (Sigmoid moved INTO the class with the #nn4sys-dual arms.)
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32], [2.0]]), None).expect("l1")),
    ));
    graph.add_node(GraphNode::new(
        "tanh",
        Layer::Tanh(crate::layers::TanhLayer::new()),
        vec!["l1".to_string()],
    ));
    graph.set_output("tanh");
    assert!(!graph_supports_f64_tail(&graph));

    let input = bt(&[-1.0], &[1.0]);
    let spec = arr2(&[[1.0_f32, 0.0]]);
    let outcome = f64_tail_verify(
        &graph,
        &input,
        &spec,
        &[-1e9_f32],
        &[1],
        None,
        None,
        None,
        None,
    );
    assert!(
        matches!(outcome, F64TailOutcome::Unsupported),
        "unsupported op must decline, got {outcome:?}"
    );
}

#[test]
fn malformed_clause_layout_declines() {
    let graph = build_cancellation_linear_net();
    let input = bt(&[-1.0, -1.0], &[1.0, 1.0]);
    let spec = arr2(&[[1.0_f32, 0.0]]);
    // clause_sizes sum != rows
    assert!(matches!(
        f64_tail_verify(&graph, &input, &spec, &[0.0], &[2], None, None, None, None),
        F64TailOutcome::Unsupported
    ));
    // empty clause
    assert!(matches!(
        f64_tail_verify(&graph, &input, &spec, &[0.0], &[], None, None, None, None),
        F64TailOutcome::Unsupported
    ));
}

#[test]
fn expired_deadline_declines() {
    let graph = build_cancellation_linear_net();
    let input = bt(&[-1.0, -1.0], &[1.0, 1.0]);
    let spec = arr2(&[[1.0_f32, 0.0]]);
    let past = Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .unwrap();
    assert!(matches!(
        f64_tail_verify(
            &graph,
            &input,
            &spec,
            &[0.0],
            &[1],
            None,
            None,
            None,
            Some(past)
        ),
        F64TailOutcome::Unsupported
    ));
}

#[test]
fn band_default_and_parse() {
    // Default without env: 5e-3 (do not mutate process env in tests; just
    // assert the default when the var is absent).
    if std::env::var("NY_F64_TAIL_BAND").is_err() {
        assert_eq!(f64_tail_band(), 5e-3);
    }
}

// ---------------------------------------------------------------------------
// Alpha-tail refresh (docs/LSNC_ALPHA_TAIL_DESIGN.md option A).
// ---------------------------------------------------------------------------

/// Sampled-containment soundness across the WHOLE alpha design space —
/// frozen maps at 0.0 / 0.5 / 1.0 and the refreshed pass — plus the
/// keep-best guarantee (refreshed rows can only meet-or-beat the warm
/// baseline) and seed determinism.
#[test]
fn refreshed_pass_is_sound_keeps_best_and_deterministic() {
    let graph = build_relu_mul_net();
    let input = bt(&[-1.0, -0.5], &[0.75, 1.0]);
    let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, -1.0]]);
    let node_bounds =
        crate::network::collect_intermediate_bounds(&graph, &input, None, None).expect("anchors");

    // Sampled true minima per spec row (dense grid; the sampling reference).
    let (lo, hi) = ([-1.0_f64, -0.5], [0.75_f64, 1.0]);
    let steps = 33usize;
    let sampled_min = |row: &[f64]| -> f64 {
        let mut min_sample = f64::INFINITY;
        for i in 0..=steps {
            for j in 0..=steps {
                let x = [
                    lo[0] + (hi[0] - lo[0]) * (i as f64) / (steps as f64),
                    lo[1] + (hi[1] - lo[1]) * (j as f64) / (steps as f64),
                ];
                let y = relu_mul_forward(&x);
                let v = row[0] * y[0] + row[1] * y[1];
                min_sample = min_sample.min(v);
            }
        }
        min_sample
    };
    let spec_rows: Vec<Vec<f64>> = spec
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let mins: Vec<f64> = spec_rows.iter().map(|r| sampled_min(r)).collect();

    // Fixed alpha maps at the family endpoints and midpoint: every value in
    // [0,1] must produce certified-contained bounds.
    for alpha_val in [0.0_f32, 0.5, 1.0] {
        let mut alphas: HashMap<String, Array2<f32>> = HashMap::new();
        alphas.insert("mul".to_string(), Array2::from_elem((2, 2), alpha_val));
        let outcome = f64_tail_verify(
            &graph,
            &input,
            &spec,
            &[-1e9_f32, -1e9, -1e9],
            &[1, 1, 1],
            Some(&alphas),
            Some(&node_bounds),
            None,
            None,
        );
        let row_lowers = match outcome {
            F64TailOutcome::Verified { ref row_lowers } => row_lowers.clone(),
            ref other => panic!("alpha={alpha_val}: expected Verified vs -1e9, got {other:?}"),
        };
        for (r, (&l_cert, &min_true)) in row_lowers.iter().zip(mins.iter()).enumerate() {
            assert!(
                l_cert <= min_true + 1e-12,
                "alpha={alpha_val} row {r}: certified {l_cert} above sampled min {min_true}"
            );
        }
    }

    // Warm baseline (0.5 map) via the plain pass.
    let mut warm: HashMap<String, Array2<f32>> = HashMap::new();
    warm.insert("mul".to_string(), Array2::from_elem((2, 2), 0.5_f32));
    let baseline = match f64_tail_verify(
        &graph,
        &input,
        &spec,
        &[-1e9_f32, -1e9, -1e9],
        &[1, 1, 1],
        Some(&warm),
        Some(&node_bounds),
        None,
        None,
    ) {
        F64TailOutcome::Verified { row_lowers } => row_lowers,
        other => panic!("baseline must verify vs -1e9, got {other:?}"),
    };

    // Refreshed pass against UNREACHABLE thresholds: every clause blocks, the
    // SPSA refresh runs on all rows, outcome stays NotVerified (sound).
    let eval = f64_tail_verify_refreshed(
        &graph,
        &input,
        &spec,
        &[1e9_f32, 1e9, 1e9],
        &[1, 1, 1],
        Some(&warm),
        Some(&node_bounds),
        None,
        None,
        20,
        0xA1FA_7A11,
    );
    assert!(matches!(eval.outcome, F64TailOutcome::NotVerified { .. }));
    assert!(eval.refreshed_alphas.is_some(), "refresh must have run");
    assert_eq!(eval.row_lowers.len(), 3);
    for (r, ((&l_ref, &l_base), &min_true)) in eval
        .row_lowers
        .iter()
        .zip(baseline.iter())
        .zip(mins.iter())
        .enumerate()
    {
        assert!(
            l_ref >= l_base,
            "row {r}: keep-best violated (refreshed {l_ref} < baseline {l_base})"
        );
        assert!(
            l_ref <= min_true + 1e-12,
            "row {r}: refreshed certified {l_ref} above sampled min {min_true}"
        );
    }
    assert!(
        eval.gap_refreshed >= eval.gap_baseline,
        "grouped gap must not regress under keep-best"
    );

    // Seed determinism: identical bits on a re-run.
    let eval2 = f64_tail_verify_refreshed(
        &graph,
        &input,
        &spec,
        &[1e9_f32, 1e9, 1e9],
        &[1, 1, 1],
        Some(&warm),
        Some(&node_bounds),
        None,
        None,
        20,
        0xA1FA_7A11,
    );
    for (r, (&a, &b)) in eval
        .row_lowers
        .iter()
        .zip(eval2.row_lowers.iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {r}: refresh must be deterministic under a fixed seed"
        );
    }
}

/// The x^2 relaxation-gap geometry (hook-test fixture, unit leg): over
/// [-d, d] the interpolated-McCormick family for `x*x` is capped at `-d^2`
/// (every r yields intercept `-d^2` and plane min `<= -d^2`), so the refresh
/// ALONE cannot clear a threshold above it — the named blocker for micro-BaB.
/// On the HALF boxes the family contains the exact plane (r -> 0 or 1) and
/// the refresh must find it.
#[test]
fn refresh_finds_exact_facet_on_half_box_but_not_parent() {
    let d = 0.02_f32;
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "id",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("id")),
    ));
    graph.add_node(GraphNode::new(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        vec!["id".to_string(), "id".to_string()],
    ));
    graph.set_output("mul");
    let spec = arr2(&[[1.0_f32]]);
    let mut warm: HashMap<String, Array2<f32>> = HashMap::new();
    warm.insert("mul".to_string(), Array2::from_elem((2, 1), 0.5_f32));
    let t = -1.2e-4_f32;

    // Parent box: refresh cannot beat the family cap -d^2 = -4e-4 < t.
    let parent = bt(&[-d], &[d]);
    let eval = f64_tail_verify_refreshed(
        &graph,
        &parent,
        &spec,
        &[t],
        &[1],
        Some(&warm),
        None,
        None,
        None,
        20,
        0xA1FA_7A11,
    );
    match eval.outcome {
        F64TailOutcome::NotVerified { min_gap_f64 } => {
            assert!(
                min_gap_f64 < 0.0,
                "parent must stay blocked, got gap {min_gap_f64}"
            );
            // The refreshed gap must sit near the family cap: -d^2 - t.
            let cap_gap = -f64::from(d) * f64::from(d) - f64::from(t);
            assert!(
                (eval.gap_refreshed - cap_gap).abs() < 5e-5,
                "refreshed gap {:.3e} should be near the family cap {:.3e}",
                eval.gap_refreshed,
                cap_gap
            );
        }
        other => panic!("parent must not verify, got {other:?}"),
    }

    // Half boxes: the family contains the EXACT facet (bound -> 0) and the
    // 20-iteration refresh must walk r far enough to clear t (needs r < 0.3
    // on the left half, r > 0.7 on the right half — a genuine per-domain
    // re-targeting in BOTH directions).
    for (lo, hi) in [(-d, 0.0_f32), (0.0_f32, d)] {
        let child = bt(&[lo], &[hi]);
        let eval = f64_tail_verify_refreshed(
            &graph,
            &child,
            &spec,
            &[t],
            &[1],
            Some(&warm),
            None,
            None,
            None,
            20,
            0xA1FA_7A11,
        );
        match eval.outcome {
            F64TailOutcome::Verified { ref row_lowers } => {
                assert!(row_lowers[0] > f64::from(t));
                // Soundness: never above the true min (0 at x=0).
                assert!(row_lowers[0] <= 1e-12);
            }
            ref other => panic!("half box [{lo},{hi}] must verify after refresh, got {other:?}"),
        }
    }
}

// ---- #nn4sys-dual arms: Sigmoid + tensor-Div interval substitution ---------

/// Certified sigmoid endpoints enclose the RN evaluation across magnitudes,
/// including the exp overflow/underflow extremes, and stay inside [0, 1].
#[test]
fn sigmoid_endpoint_helpers_enclose() {
    let xs: Vec<f64> = vec![
        -750.0, -100.0, -20.0, -5.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 5.0, 20.0, 100.0, 750.0,
    ];
    for &x in &xs {
        let rn = 1.0 / (1.0 + (-x).exp());
        let lo = sigmoid_round_down(x);
        let hi = sigmoid_round_up(x);
        assert!(
            (0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi),
            "x={x}: endpoints outside [0,1]: lo={lo} hi={hi}"
        );
        assert!(lo <= rn, "x={x}: lo {lo} above RN sigma {rn}");
        assert!(hi >= rn, "x={x}: hi {hi} below RN sigma {rn}");
        assert!(lo <= hi, "x={x}: lo {lo} > hi {hi}");
    }
}

/// Linear -> ReLU -> {num, POSITIVE den} -> Div -> Sigmoid net: the walk's
/// certified row lowers must contain a dense sampled minimum (the same
/// contract as `relu_mul_net_bound_is_sampled_contained`).
fn build_div_sigmoid_net() -> GraphNetwork {
    let w1 = arr2(&[[1.2_f32, -0.8], [0.4, 1.1]]);
    let b1 = arr1(&[0.3_f32, -0.2]);
    // num: mixed-sign weights.
    let wn = arr2(&[[0.9_f32, -1.3], [-0.6, 0.7]]);
    let bn = arr1(&[0.2_f32, -0.4]);
    // den: nonnegative weights + bias >= 1.5 over relu >= 0 keeps d >= 1.5 > 0.
    let wd = arr2(&[[0.5_f32, 0.25], [0.3, 0.6]]);
    let bd = arr1(&[1.5_f32, 2.0]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("l1")),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "num",
        Layer::Linear(LinearLayer::new(wn, Some(bn)).expect("num")),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "den",
        Layer::Linear(LinearLayer::new(wd, Some(bd)).expect("den")),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "div",
        Layer::Div(crate::layers::DivLayer),
        vec!["num".to_string(), "den".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(crate::layers::SigmoidLayer::new()),
        vec!["div".to_string()],
    ));
    graph.set_output("sig");
    graph
}

/// f64 forward of the div-sigmoid net at a point (RN; sampling reference).
fn div_sigmoid_forward(x: &[f64; 2]) -> [f64; 2] {
    let w1 = [[1.2_f64, -0.8], [0.4, 1.1]];
    let b1 = [0.3_f64, -0.2];
    let wn = [[0.9_f64, -1.3], [-0.6, 0.7]];
    let bn = [0.2_f64, -0.4];
    let wd = [[0.5_f64, 0.25], [0.3, 0.6]];
    let bd = [1.5_f64, 2.0];
    let mut h = [0.0_f64; 2];
    for o in 0..2 {
        h[o] = (w1[o][0] * x[0] + w1[o][1] * x[1] + b1[o]).max(0.0);
    }
    let mut out = [0.0_f64; 2];
    for o in 0..2 {
        let n = wn[o][0] * h[0] + wn[o][1] * h[1] + bn[o];
        let d = wd[o][0] * h[0] + wd[o][1] * h[1] + bd[o];
        let q = n / d;
        out[o] = 1.0 / (1.0 + (-q).exp());
    }
    out
}

#[test]
fn div_sigmoid_net_bound_is_sampled_contained() {
    let graph = build_div_sigmoid_net();
    let input = bt(&[-1.0, -0.75], &[0.9, 1.1]);
    let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, -1.0], [-1.0, 1.0]]);
    let node_bounds =
        crate::network::collect_intermediate_bounds(&graph, &input, None, None).expect("anchors");
    let outcome = f64_tail_verify(
        &graph,
        &input,
        &spec,
        &[-1e9_f32, -1e9, -1e9, -1e9],
        &[1, 1, 1, 1],
        None,
        Some(&node_bounds),
        None,
        None,
    );
    let row_lowers = match outcome {
        F64TailOutcome::Verified { ref row_lowers } => row_lowers.clone(),
        ref other => panic!("expected Verified against -1e9 thresholds, got {other:?}"),
    };
    let (lo, hi) = ([-1.0_f64, -0.75], [0.9_f64, 1.1]);
    let steps = 41usize;
    for r in 0..spec.nrows() {
        let row: Vec<f64> = spec.row(r).iter().map(|&v| f64::from(v)).collect();
        let l_cert = row_lowers[r];
        assert!(l_cert.is_finite(), "row {r}: expected a finite bound");
        let mut min_sample = f64::INFINITY;
        for i in 0..=steps {
            for j in 0..=steps {
                let x = [
                    lo[0] + (hi[0] - lo[0]) * (i as f64) / (steps as f64),
                    lo[1] + (hi[1] - lo[1]) * (j as f64) / (steps as f64),
                ];
                let y = div_sigmoid_forward(&x);
                let v = row[0] * y[0] + row[1] * y[1];
                min_sample = min_sample.min(v);
            }
        }
        assert!(
            l_cert <= min_sample + 1e-12,
            "row {r}: certified lower {l_cert} above sampled min {min_sample}"
        );
    }
}

/// A denominator whose IBP anchor spans zero must DECLINE the pass
/// (fail-closed `Unsupported`), never emit a bound.
#[test]
fn div_zero_spanning_denominator_declines() {
    let w1 = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    // den = raw linear of the INPUT (no relu, mixed signs over the box).
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "num",
        Layer::Linear(LinearLayer::new(w1.clone(), None).expect("num")),
    ));
    graph.add_node(GraphNode::from_input(
        "den",
        Layer::Linear(LinearLayer::new(w1, None).expect("den")),
    ));
    graph.add_node(GraphNode::new(
        "div",
        Layer::Div(crate::layers::DivLayer),
        vec!["num".to_string(), "den".to_string()],
    ));
    graph.set_output("div");
    let input = bt(&[-1.0, -1.0], &[1.0, 1.0]);
    // Fail-closed fires even EARLIER than the walk guard: the DivLayer IBP
    // contract rejects a zero-spanning divisor during anchor collection, so
    // the tail can never even build its ctx for such a box (the in-walk
    // Decline remains as defense-in-depth for anchors from other sources).
    let anchors = crate::network::collect_intermediate_bounds(&graph, &input, None, None);
    assert!(
        anchors.is_err(),
        "zero-spanning denominator must fail anchor collection"
    );
}
