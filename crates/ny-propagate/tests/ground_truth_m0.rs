// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M0 spike for `docs/GEOMETRIC_GROUND_TRUTH_PLAN.md`: verify a surrogate
//! network against a *symbolic geometric ground truth* by reducing dominance
//! `f(x) >= g(x)` on a box to nonnegativity of the difference network
//! `h(x) = f(x) - g(x)` built by `build_difference_network`, then bounding
//! `h` with CROWN.
//!
//! Ground truth: signed distance to the rational plane z = 5, i.e.
//! `g(x) = n·x + d` with `n = (0, 0, 1)`, `d = -5`, expressed as a
//! one-Linear-layer `GraphNetwork`. On the box `x ∈ [-10, 10]^3`,
//! `g ∈ [-15, 5]`.
//!
//! Surrogates are 2-layer FC ReLU networks (3 -> 4 -> 1) with hand-picked
//! exactly-representable integer weights so that all claims hold *exactly*:
//! - dominant surrogate:  `f  = g + 10`  (so `f - g ≡ 10 >= 0` — Verified)
//! - violating surrogate: `f2 = g - 10`  (so `f2 - g ≡ -10 < 0` — Falsified,
//!   with a concrete counterexample on the plane itself).
//!
//! This proves the M0 reduction end to end using only machinery that exists
//! today (`GraphNetwork` construction + `build_difference_network` +
//! IBP/CROWN + `Verifier::verify_graph`).

use ndarray::{arr1, arr2};
use ny_core::{Bound, VerificationResult, VerificationSpec};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{
    build_difference_network, GraphNetwork, GraphNode, Layer, PropagationConfig, PropagationMethod,
    Verifier,
};
use ny_tensor::BoundedTensor;

/// Half-width of the input box `x ∈ [-10, 10]^3`.
const BOX: f32 = 10.0;

/// Ground truth g(x) = n·x + d = z - 5: signed distance to the plane z = 5
/// (unit normal, so the affine form *is* the signed distance), as a
/// single-Linear-layer GraphNetwork. All constants are integers, hence
/// f64/f32-exact — the "rational parameters" case of the plan.
fn build_plane_signed_distance() -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "plane_dist",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.0_f32, 0.0, 1.0]]), Some(arr1(&[-5.0_f32])))
                .expect("plane distance layer should be valid"),
        ),
    ));
    g.set_output("plane_dist");
    g
}

/// Hand-picked 2-layer FC ReLU surrogate (3 -> 4 -> 1) computing
/// `f(x) = g(x) + 20 + final_bias` exactly on the box:
///
/// - `lin1` neuron 0 computes `g(x) + 20 = z + 15 ∈ [5, 25]` on the box, so
///   its ReLU is stably ACTIVE (identity) — the pass-through is exact.
/// - `lin1` neurons 1..3 have zero weights and bias `-1`, so they are stably
///   INACTIVE (output 0) and `lin2` ignores them anyway.
/// - `lin2` = `[1, 0, 0, 0]` with bias `final_bias`.
///
/// `final_bias = -10` gives the dominant surrogate `f = g + 10`;
/// `final_bias = -30` gives the violating surrogate `f2 = g - 10`.
fn build_surrogate(final_bias: f32) -> GraphNetwork {
    let mut f = GraphNetwork::new();
    f.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[
                    [0.0_f32, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]),
                Some(arr1(&[15.0_f32, -1.0, -1.0, -1.0])),
            )
            .expect("surrogate lin1 should be valid"),
        ),
    ));
    f.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    f.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32, 0.0, 0.0, 0.0]]), Some(arr1(&[final_bias])))
                .expect("surrogate out should be valid"),
        ),
        vec!["relu1".to_string()],
    ));
    f.set_output("out");
    f
}

/// The input region R = [-10, 10]^3 as a BoundedTensor.
fn input_box() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-BOX, -BOX, -BOX]).into_dyn(),
        arr1(&[BOX, BOX, BOX]).into_dyn(),
    )
    .expect("input box should be valid")
}

/// A concrete point x* as a degenerate (zero-width) BoundedTensor, so IBP
/// forward propagation is an exact concrete evaluation.
fn point(x: [f32; 3]) -> BoundedTensor {
    let arr = arr1(&x).into_dyn();
    BoundedTensor::new(arr.clone(), arr).expect("point tensor should be valid")
}

/// Dominance spec: every output of h = f - g must lie in [0, +inf).
fn dominance_spec() -> VerificationSpec {
    VerificationSpec::new(
        vec![Bound::new(-BOX, BOX); 3],
        vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
    )
    .expect("dominance spec should be valid")
}

fn crown_verifier() -> Verifier {
    Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    })
}

/// M0 Verified outcome: f = g + 10 dominates the plane distance on the box,
/// and CROWN on the difference network proves it (lower bound of h >= 0).
#[test]
fn m0_crown_verifies_surrogate_dominance_over_plane_distance() {
    let f = build_surrogate(-10.0); // f = g + 10
    let g = build_plane_signed_distance();
    let h = build_difference_network(&f, &g).expect("difference network should build");

    // Direct CROWN pass over the box: h ≡ 10 exactly, and every constant is
    // integer-valued, so the relational (backward) bound is exact.
    let bounds = h
        .propagate_crown(&input_box())
        .expect("CROWN on the difference network should succeed");
    assert_eq!(bounds.lower().len(), 1, "h must have a single output");
    let lo = bounds.lower()[0];
    let hi = bounds.upper()[0];
    assert!(
        lo >= 0.0,
        "dominance must be proved: CROWN lower bound of f - g is {lo}, expected >= 0"
    );
    assert!(
        (lo - 10.0).abs() <= 1e-4 && (hi - 10.0).abs() <= 1e-4,
        "f - g is identically 10; CROWN should be exact here but gave [{lo}, {hi}]"
    );

    // End-to-end verifier path (the shape verify_against_ground_truth will
    // take in M1): spec output bound [0, +inf) over the same box.
    let result = crown_verifier()
        .verify_graph(&h, &dominance_spec())
        .expect("verification should run");
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert!(
                output_bounds.iter().all(|b| b.lower() >= 0.0),
                "Verified dominance must certify nonnegative lower bounds, got {output_bounds:?}"
            );
        }
        other => panic!("expected Verified dominance, got {other:?}"),
    }
}

/// Documents *why* the difference network needs a relational method: plain
/// IBP decorrelates the shared input, so interval subtraction alone cannot
/// prove dominance even though f - g ≡ 10 (IBP sees f ∈ [-5, 15] minus
/// g ∈ [-15, 5], i.e. h ∈ [-10, 30]).
#[test]
fn m0_ibp_alone_is_too_loose_to_prove_dominance() {
    let f = build_surrogate(-10.0); // f = g + 10
    let g = build_plane_signed_distance();
    let h = build_difference_network(&f, &g).expect("difference network should build");

    let ibp = h
        .propagate_ibp(&input_box())
        .expect("IBP on the difference network should succeed");
    assert!(
        ibp.lower()[0] < 0.0,
        "IBP is expected to decorrelate the shared input (lower bound {} should be < 0); \
         if this ever fails, IBP got relational — update the M0 notes",
        ibp.lower()[0]
    );
}

/// M0 Falsified outcome: f2 = g - 10 dips below the ground truth everywhere.
/// The analysis must NOT verify dominance, the CROWN *upper* bound must be
/// negative (definite violation over the whole box), and a concrete
/// counterexample x* on the plane confirms f2(x*) < g(x*).
#[test]
fn m0_falsified_dominance_is_rejected_with_concrete_counterexample() {
    let f2 = build_surrogate(-30.0); // f2 = g - 10
    let g = build_plane_signed_distance();
    let h = build_difference_network(&f2, &g).expect("difference network should build");

    // CROWN over the box: h ≡ -10, so even the UPPER bound is < 0 — the
    // property is not just unproven, it is definitely violated on all of R.
    let bounds = h
        .propagate_crown(&input_box())
        .expect("CROWN on the difference network should succeed");
    let (lo, hi) = (bounds.lower()[0], bounds.upper()[0]);
    assert!(
        hi < 0.0,
        "f2 - g is identically -10; CROWN upper bound should be negative, got [{lo}, {hi}]"
    );

    // The verifier must not report Verified for the dominance spec.
    let result = crown_verifier()
        .verify_graph(&h, &dominance_spec())
        .expect("verification should run");
    assert!(
        !matches!(result, VerificationResult::Verified { .. }),
        "dominance must not verify for f2 = g - 10, got {result:?}"
    );

    // Concrete counterexample: x* = (0, 0, 5) lies ON the plane, so
    // g(x*) = 0 and f2(x*) = -10. Evaluate exactly via zero-width IBP.
    let x_star = [0.0_f32, 0.0, 5.0];
    let h_at = h
        .propagate_ibp(&point(x_star))
        .expect("concrete evaluation of h should succeed");
    assert!(
        h_at.upper()[0] < 0.0,
        "counterexample must witness f2(x*) - g(x*) < 0, got {}",
        h_at.upper()[0]
    );

    // Cross-check the witness against the original networks (not just the
    // merged graph): f2(x*) < g(x*) in a direct evaluation of each side.
    let f2_at = f2
        .propagate_ibp(&point(x_star))
        .expect("concrete evaluation of f2 should succeed");
    let g_at = g
        .propagate_ibp(&point(x_star))
        .expect("concrete evaluation of g should succeed");
    assert!(
        f2_at.upper()[0] < g_at.lower()[0],
        "witness must satisfy f2(x*) < g(x*): f2(x*)={}, g(x*)={}",
        f2_at.upper()[0],
        g_at.lower()[0]
    );
    // x* lies on the plane, so g(x*) = 0. The IBP evaluation returns a sound
    // enclosure (1-ULP directed rounding), so assert it tightly brackets 0
    // rather than demanding exact equality.
    assert!(
        g_at.lower()[0] <= 0.0
            && g_at.upper()[0] >= 0.0
            && (g_at.upper()[0] - g_at.lower()[0]) <= 1e-6,
        "x* lies on the plane, so g(x*) = 0; sound enclosure was [{}, {}]",
        g_at.lower()[0],
        g_at.upper()[0]
    );
}
