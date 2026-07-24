// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end Pillar-1 test: generate an exact-rational CROWN certificate for a
//! concrete ReLU network, confirm it passes the in-tree mirror of Clean's
//! verifier, and emit the canonical JSON that Clean's `cert verify-external`
//! command consumes.
//!
//! Set `NY_CERT_OUT_DIR` to a directory to dump the generated certificates for
//! the cross-repo round-trip against Clean's binary.

use ny_cert::{
    check_entailment, check_farkas, entailment_to_json, farkas_to_json, Rat, Relu1Problem,
};

fn r(n: i128, d: i128) -> Rat {
    Rat::new(n, d).unwrap()
}

/// The worked example:
/// `z0 = x0 + x1`, `z1 = x0 − x1`, `y = ReLU(z0) − ReLU(z1) + 5/2`, box `[−1,1]²`.
/// True minimum of `y` over the box is `1/2`; CROWN certifies `y ≥ 1/2 ≥ 0`.
fn worked_example() -> Relu1Problem {
    Relu1Problem {
        w1: vec![vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(-1, 1)]],
        b1: vec![Rat::ZERO, Rat::ZERO],
        w2: vec![r(1, 1), r(-1, 1)],
        b2: r(5, 2),
        input_lower: vec![r(-1, 1), r(-1, 1)],
        input_upper: vec![r(1, 1), r(1, 1)],
        alpha: Some(vec![r(1, 2), r(1, 2)]),
    }
}

#[test]
fn crown_lower_bound_is_tight_and_sound() {
    let problem = worked_example();
    let certified = problem.certify(Rat::ZERO).unwrap();
    // CROWN reproduces the exact true minimum here.
    assert_eq!(certified.lower_bound, r(1, 2));
    // Pre-activation intervals are [−2, 2] for both units (unstable).
    assert_eq!(certified.preact_lower, vec![r(-2, 1), r(-2, 1)]);
    assert_eq!(certified.preact_upper, vec![r(2, 1), r(2, 1)]);
}

#[test]
fn entailment_certificate_self_checks() {
    let certified = worked_example().certify(Rat::ZERO).unwrap();
    let (derived, claimed) = check_entailment(&certified.entailment).unwrap();
    // Derived bound is −1/2 in normalized (−y ≤ ·) space, i.e. y ≥ 1/2;
    // the claimed safety bound is y ≥ 0.
    assert_eq!(derived, r(-1, 2));
    assert_eq!(claimed, Rat::ZERO);
}

#[test]
fn farkas_certificate_self_checks() {
    let certified = worked_example().certify(Rat::ZERO).unwrap();
    let contradiction = check_farkas(&certified.farkas).unwrap();
    // Combination collapses to (strict) 0 < threshold − m = −1/2 ≤ 0.
    assert_eq!(contradiction, r(-1, 2));
}

#[test]
fn rejects_threshold_above_certified_bound() {
    // The certified bound is 1/2; asking to prove y ≥ 1 must fail honestly.
    let err = worked_example().certify(Rat::ONE).unwrap_err();
    assert!(matches!(
        err,
        ny_cert::CrownError::ThresholdAboveBound { .. }
    ));
}

#[test]
fn json_matches_clean_schema_and_round_trips_via_serde() {
    let certified = worked_example().certify(Rat::ZERO).unwrap();

    let ent = entailment_to_json(&certified.entailment).unwrap();
    assert_eq!(ent["type"], "entailment_certificate");
    assert_eq!(ent["version"], "1.0");
    assert_eq!(ent["conclusion"]["type"], "linear_constraint");
    assert_eq!(ent["conclusion"]["kind"], "ge");
    assert_eq!(
        ent["premises"].as_array().unwrap().len(),
        certified.entailment.premises.len()
    );

    let far = farkas_to_json(&certified.farkas).unwrap();
    assert_eq!(far["type"], "farkas_certificate");
    assert_eq!(far["conclusion"], "contradiction");

    // Optionally emit for the cross-repo round-trip against Clean's binary.
    if let Ok(dir) = std::env::var("NY_CERT_OUT_DIR") {
        std::fs::write(
            format!("{dir}/ny_relu1_entailment.json"),
            serde_json::to_string_pretty(&ent).unwrap(),
        )
        .unwrap();
        std::fs::write(
            format!("{dir}/ny_relu1_farkas.json"),
            serde_json::to_string_pretty(&far).unwrap(),
        )
        .unwrap();
    }
}

/// A second, independent network so the generator is exercised beyond the
/// hand-checked case: a single always-active unit and a negative output weight.
#[test]
fn second_network_certifies() {
    // z0 = 2·x0 (box x0 ∈ [1, 2] ⇒ z0 ∈ [2,4], always active), a0 = z0,
    // y = a0 − 3 ⇒ y ∈ [−1, 1]; certify y ≥ −1.
    let problem = Relu1Problem {
        w1: vec![vec![r(2, 1)]],
        b1: vec![Rat::ZERO],
        w2: vec![r(1, 1)],
        b2: r(-3, 1),
        input_lower: vec![r(1, 1)],
        input_upper: vec![r(2, 1)],
        alpha: None,
    };
    let certified = problem.certify(r(-1, 1)).unwrap();
    assert_eq!(certified.lower_bound, r(-1, 1));
    check_entailment(&certified.entailment).unwrap();
    check_farkas(&certified.farkas).unwrap();
}
