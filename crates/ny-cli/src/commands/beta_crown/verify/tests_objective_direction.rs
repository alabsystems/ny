// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression pins for the objective-ENCODING ↔ direction-FLAG contract
//! (#ml4acopf-clause-direction).
//!
//! `BetaCrownConfig::verify_upper_bound` selects the engine's stop test:
//! `upper < threshold` when set, `lower > threshold` when clear. Two objective
//! encodings reach the engine from this CLI and they need OPPOSITE flags:
//!
//!   * RAW (`verify_standard`): objective `+Y_idx`, threshold = the spec's own
//!     constant. `Y_i >= c` unsafe ⇒ prove `upper(Y_i) < c` ⇒ flag SET.
//!   * SIGN-NORMALIZED (`build_constraint_objective`, used by every lane under
//!     `verify_relational_constraints`): `Y_i >= c` ⇒ `(-1·Y_i, -c)`, and
//!     `Y_i <= c` ⇒ `(+1·Y_i, +c)`. Refuting the row is then ALWAYS
//!     `lower(spec·Y) > threshold` ⇒ flag CLEAR, for both comparators.
//!
//! Both forms are pinned here against the REAL engine stop test (via
//! `dispatch_graph_constraint`, the same call the per-constraint lane makes),
//! including the case that makes a wrong flag UNSOUND rather than merely
//! incomplete.

use std::time::Duration;

use ndarray::{arr1, Array2};
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{
    beta_crown::BetaCrownConfig, layers::LinearLayer, BabVerificationStatus, BetaCrownVerifier,
    GraphNetwork, GraphNode, Layer,
};
use ny_tensor::BoundedTensor;

use super::{
    build_constraint_objective, config_for_normalized_objectives, dispatch_graph_constraint,
};

/// Identity graph `Y_0 = X_0`, so the objective bounds over the input box are
/// exactly `spec_coeff * [lo, hi]` and every assertion below is arithmetic the
/// test states in full — no dependence on relaxation quality.
fn identity_graph() -> GraphNetwork {
    let linear = LinearLayer::new(Array2::<f32>::eye(1), None).expect("identity layer");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");
    graph
}

fn input_box(lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(arr1(&[lo]).into_dyn(), arr1(&[hi]).into_dyn()).expect("valid box")
}

fn base_config(verify_upper_bound: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        max_domains: 1,
        max_depth: 1,
        timeout: Duration::from_secs(5),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        verify_upper_bound,
        ..Default::default()
    }
}

/// Run the single-objective graph lane exactly as `iterate_constraints` does.
fn verdict(
    config: &BetaCrownConfig,
    input: &BoundedTensor,
    spec_coeffs: &[f32],
    threshold: f32,
) -> BabVerificationStatus {
    let graph = identity_graph();
    let verifier = BetaCrownVerifier::new(config.clone());
    dispatch_graph_constraint(
        &verifier,
        &graph,
        input,
        spec_coeffs,
        threshold,
        false, // use_relu_split — input-split lane, same stop test
        false, // gpu_bab
        None,  // precomputed_bounds
        None,  // gemm_engine
        None,  // deadline
    )
    .expect("dispatch must not error")
    .result
}

fn is_verified(status: &BabVerificationStatus) -> bool {
    matches!(status, BabVerificationStatus::Verified)
}

/// The sign-normalized planner really does emit BOTH comparators in the single
/// `lower > threshold` form. This is the premise the direction pin rests on.
#[test]
fn normalized_planner_emits_lower_gt_threshold_form_for_both_comparators() {
    // `(>= Y_0 0.01)` unsafe → refute with `-Y_0 > -0.01`.
    let ge = build_constraint_objective(&OutputConstraint::GreaterEqConst(0, 0.01), 1)
        .expect("objective builds");
    assert_eq!(ge.spec_coeffs(), &[-1.0_f32]);
    assert!((ge.threshold() - (-0.01)).abs() < 1e-7);

    // `(<= Y_0 -0.01)` unsafe → refute with `+Y_0 > -0.01`.
    let le = build_constraint_objective(&OutputConstraint::LessEqConst(0, -0.01), 1)
        .expect("objective builds");
    assert_eq!(le.spec_coeffs(), &[1.0_f32]);
    assert!((le.threshold() - (-0.01)).abs() < 1e-7);

    // Opposite comparators, opposite coefficient signs, IDENTICAL threshold —
    // which is exactly why one shared direction flag has to be `lower >`.
    assert!((ge.threshold() - le.threshold()).abs() < 1e-7);
}

/// The flag handed to the sign-normalized subtree is CLEAR no matter what the
/// raw spec said.
#[test]
fn normalized_objective_config_always_clears_verify_upper_bound() {
    for incoming in [true, false] {
        let routed = config_for_normalized_objectives(&base_config(incoming));
        assert!(
            !routed.verify_upper_bound,
            "sign-normalized objectives are always decided by `lower > threshold` \
             (incoming flag was {incoming})"
        );
        // Nothing else may be disturbed by the normalization.
        assert_eq!(routed.max_domains, base_config(incoming).max_domains);
        assert_eq!(routed.timeout, base_config(incoming).timeout);
    }
}

/// FORM 1 — sign-normalized `(>= Y_0 c)` clause, refutable case.
///
/// `Y_0 ∈ [-0.005, 0.005]` stays strictly below `c = 0.01`, so the clause is
/// impossible and the correct verdict is Verified. The normalized objective is
/// `(-1·Y_0, -0.01)`; `lower(-Y_0) = -0.005 > -0.01` decides it, while
/// `upper(-Y_0) = 0.005 < -0.01` does not.
#[test]
fn normalized_ge_clause_needs_lower_gt_threshold_ml4acopf_shape() {
    let obj = build_constraint_objective(&OutputConstraint::GreaterEqConst(0, 0.01), 1)
        .expect("objective");
    let input = input_box(-0.005, 0.005);

    let routed = config_for_normalized_objectives(&base_config(true));
    assert!(
        is_verified(&verdict(
            &routed,
            &input,
            obj.spec_coeffs(),
            obj.threshold()
        )),
        "clause `Y_0 >= 0.01` is impossible on Y_0 ∈ [-0.005, 0.005] and must verify"
    );

    // The pre-fix flag (inherited from the raw spec because the first constant
    // constraint is a `>=`) tests the negation and cannot decide it.
    assert!(
        !is_verified(&verdict(
            &base_config(true),
            &input,
            obj.spec_coeffs(),
            obj.threshold()
        )),
        "`upper < threshold` on a sign-normalized row is the wrong direction"
    );
}

/// FORM 1, SOUNDNESS — same clause, but now it holds EVERYWHERE.
///
/// `Y_0 ∈ [0.02, 0.03]` satisfies `Y_0 >= 0.01` at every point, so the clause is
/// satisfiable and the verifier must NOT report Verified. With the pre-fix flag
/// it does (`upper(-Y_0) = -0.02 < -0.01`), which is an unsound `unsat` — this
/// is the -150 scoring hazard the fix removes, so pin both directions.
#[test]
fn normalized_ge_clause_wrong_direction_would_falsely_verify_a_satisfiable_clause() {
    let obj = build_constraint_objective(&OutputConstraint::GreaterEqConst(0, 0.01), 1)
        .expect("objective");
    let input = input_box(0.02, 0.03);

    let routed = config_for_normalized_objectives(&base_config(true));
    assert!(
        !is_verified(&verdict(
            &routed,
            &input,
            obj.spec_coeffs(),
            obj.threshold()
        )),
        "clause `Y_0 >= 0.01` HOLDS on Y_0 ∈ [0.02, 0.03]; reporting Verified would be unsound"
    );

    assert!(
        is_verified(&verdict(
            &base_config(true),
            &input,
            obj.spec_coeffs(),
            obj.threshold()
        )),
        "documents the pre-fix hazard: the inverted test verifies a clause that is \
         satisfiable everywhere"
    );
}

/// FORM 1 sibling — sign-normalized `(<= Y_0 c)` clause, the other half of the
/// ml4acopf disjunction. Same shared threshold, opposite coefficient sign, and
/// still decided by `lower > threshold`.
#[test]
fn normalized_le_clause_needs_lower_gt_threshold_ml4acopf_shape() {
    let obj =
        build_constraint_objective(&OutputConstraint::LessEqConst(0, -0.01), 1).expect("objective");
    let input = input_box(-0.005, 0.005);

    let routed = config_for_normalized_objectives(&base_config(true));
    assert!(
        is_verified(&verdict(
            &routed,
            &input,
            obj.spec_coeffs(),
            obj.threshold()
        )),
        "clause `Y_0 <= -0.01` is impossible on Y_0 ∈ [-0.005, 0.005] and must verify"
    );

    assert!(
        !is_verified(&verdict(
            &base_config(true),
            &input,
            obj.spec_coeffs(),
            obj.threshold()
        )),
        "`upper < threshold` on a sign-normalized row is the wrong direction"
    );
}

/// FORM 2 — the RAW `verify_standard` encoding, which genuinely needs
/// `upper < threshold` and must be left alone.
///
/// Objective `+Y_0` with the spec's own constant `c = 0.01` as the threshold:
/// proving `Y_0 >= 0.01` impossible IS `upper(Y_0) < 0.01`. Clearing the flag
/// here would break it, so this pins the opposite form.
#[test]
fn raw_standard_objective_still_needs_upper_lt_threshold() {
    let input = input_box(-0.005, 0.005);
    let raw_objective = [1.0_f32];
    let raw_threshold = 0.01_f32;

    assert!(
        is_verified(&verdict(
            &base_config(true),
            &input,
            &raw_objective,
            raw_threshold
        )),
        "raw `+Y_0` vs c=0.01 is decided by `upper(Y_0) < 0.01`"
    );

    assert!(
        !is_verified(&verdict(
            &base_config(false),
            &input,
            &raw_objective,
            raw_threshold
        )),
        "`lower > threshold` on a RAW objective is the wrong direction — the \
         normalization must not leak onto the verify_standard path"
    );
}
