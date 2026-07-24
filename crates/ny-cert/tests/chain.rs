// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Composition (cut-rule) tests for NY's legacy linear-linkage schema. The first
//! block is a genuine CROWN certificate from the worked ReLU network; later
//! steps relay/weaken its conclusion. Clean never adopted the former
//! `verify_entailment_chain` proposal; its current composition API merges and
//! re-verifies certificates, so this NY-only compatibility surface makes no
//! cross-repo Clean claim.
//!
//! Scope note: this checker models a *linear* sequence in which each
//! step's single conclusion is consumed by the next. That captures scalar
//! bound-threading / progressive refinement. Full block-wise CROWN (threading
//! several intermediate bounds as a DAG) is a strict generalization and is left
//! as future work — we do not pretend the linear chain already covers it.

use ny_cert::schema::LinearConstraint;
use ny_cert::{
    chain_to_json, check_chain, check_entailment, ConstraintKind, EntailmentCertificate, Rat,
    Relu1Problem,
};

fn r(n: i128, d: i128) -> Rat {
    Rat::new(n, d).unwrap()
}

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

/// A weakening step: premise `y ≥ c` entails `y ≥ c'` for any `c' ≤ c`,
/// with the single non-negative multiplier 1.
fn weaken_step(c: Rat, c_prime: Rat) -> EntailmentCertificate {
    EntailmentCertificate {
        premises: vec![LinearConstraint::with_kind(
            ConstraintKind::Ge,
            &[("y", Rat::ONE)],
            c,
        )],
        multipliers: vec![Rat::ONE],
        conclusion: LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", Rat::ONE)], c_prime),
    }
}

#[test]
fn real_crown_block_then_weakening_composes() {
    // Block 0: the genuine CROWN certificate proving y ≥ 0 for the worked net.
    let block0 = worked_example().certify(Rat::ZERO).unwrap().entailment;
    assert_eq!(check_entailment(&block0).unwrap().0, r(-1, 2)); // proves y ≥ 1/2

    // Block 1: consume block0's conclusion (y ≥ 0) and relay y ≥ -1.
    let block1 = weaken_step(Rat::ZERO, r(-1, 1));

    let chain = vec![block0, block1];
    let (derived, claimed) = check_chain(&chain).unwrap();
    // Final step: premise y ≥ 0 (normalized -y ≤ 0) entails y ≥ -1 (claimed
    // bound -y ≤ 1). derived 0 ≤ claimed 1, so the step (and chain) holds.
    assert_eq!(derived, Rat::ZERO);
    assert_eq!(claimed, r(1, 1));

    // The chain serializes to Clean's chain JSON.
    let json = chain_to_json(&chain).unwrap();
    assert_eq!(json["version"], "1.0");
    assert_eq!(json["steps"].as_array().unwrap().len(), 2);
    // Steps are bare (no outer "type" tag) per Clean's ExternalEntailmentCert.
    assert!(json["steps"][0].get("type").is_none());
    assert_eq!(json["steps"][0]["conclusion"]["kind"], "ge");

    // Emit for the cross-repo composition check if requested.
    if let Ok(dir) = std::env::var("NY_CERT_OUT_DIR") {
        std::fs::write(
            format!("{dir}/ny_chain.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn three_step_refinement_composes() {
    // y ≥ 2  ⊨  y ≥ 1  ⊨  y ≥ 0  (telescoping weakening).
    let chain = vec![
        weaken_step(r(2, 1), r(1, 1)),
        weaken_step(r(1, 1), Rat::ZERO),
        weaken_step(Rat::ZERO, r(-1, 1)),
    ];
    let (_, claimed) = check_chain(&chain).unwrap();
    assert_eq!(claimed, r(1, 1)); // final y ≥ -1 → normalized claimed 1
}

#[test]
fn broken_linkage_is_rejected() {
    // Step 1's premise is y ≥ 5, not step 0's conclusion y ≥ 0.
    let chain = vec![
        weaken_step(r(1, 1), Rat::ZERO),
        weaken_step(r(5, 1), r(4, 1)),
    ];
    assert!(matches!(
        check_chain(&chain),
        Err(ny_cert::CheckError::ChainBreak(1))
    ));
}

#[test]
fn invalid_step_is_rejected() {
    // y ≥ 0 does NOT entail y ≥ 1 (strengthening is unsound).
    let bad = weaken_step(Rat::ZERO, r(1, 1));
    assert!(check_chain(&[bad]).is_err());
}

#[test]
fn empty_chain_is_rejected() {
    assert!(matches!(
        check_chain(&[]),
        Err(ny_cert::CheckError::EmptyChain)
    ));
}
