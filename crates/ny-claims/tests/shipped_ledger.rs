// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests against the REAL ledger in `claims/`.
//!
//! These are the tests that make the mechanism load-bearing rather than
//! decorative: the shipped records must be admissible, the artifacts they cite
//! must still be there and still hash to their pins, and — the point of design
//! target 4 — **lane membership is a test**. A change that moves a benchmark's
//! scored route breaks a build here instead of silently invalidating whatever
//! program of work was resting on it.

use std::path::PathBuf;

use ny_claims::{evaluate, lint_all, load_dir, BinaryFacts, ClaimKind, Status, World};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("crates/ny-claims -> repo root")
        .to_path_buf()
}

/// A deliberately synthetic "current binary" so these tests do not depend on
/// whether `target/release/ny` happens to be built, or on which stamp it has
/// today. No real stamp can be all-zero, so every binary-sensitive claim is
/// deterministically drifted.
fn hermetic_world() -> World {
    World::with_binary(
        repo_root(),
        Some(BinaryFacts {
            stamp: "0".repeat(64),
            version: "0.0.test".to_string(),
            source: "hermetic test world".to_string(),
        }),
    )
}

fn ledger() -> Vec<ny_claims::LoadedClaim> {
    load_dir(&repo_root().join("claims")).expect("load the shipped ledger")
}

#[test]
fn the_shipped_ledger_is_admissible() {
    let findings = lint_all(&ledger());
    assert!(
        findings.is_empty(),
        "the shipped ledger must lint clean:\n{}",
        findings
            .iter()
            .map(|finding| format!("  {}: {}", finding.id, finding.message))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_ledger_is_not_empty() {
    assert!(
        ledger().len() >= 5,
        "a ledger nobody adds to is documentation again"
    );
}

/// Every cited artifact must still exist and still hash to its pin. This is
/// the check that makes a claim's evidence non-repudiable: an artifact edited
/// after the fact cannot keep a green claim green.
#[test]
fn every_cited_artifact_is_present_and_matches_its_pin() {
    let root = repo_root();
    for entry in ledger() {
        for artifact in &entry.claim.artifacts {
            let path = root.join(&artifact.path);
            assert!(
                path.exists(),
                "{} cites a missing artifact {}",
                entry.claim.id,
                artifact.path
            );
            let bytes = std::fs::read(&path).expect("read artifact");
            use sha2::Digest as _;
            let mut hasher = sha2::Sha256::new();
            hasher.update(&bytes);
            let found = format!("{:x}", hasher.finalize());
            assert_eq!(
                found, artifact.sha256,
                "{} cites {} whose bytes moved; re-pin ONLY if the move was deliberate",
                entry.claim.id, artifact.path
            );
        }
    }
}

/// **Design target 4: lane membership is a test, not a belief.**
///
/// A route claim may be STALE (its receipt is old, or came from another host)
/// but it may never be FALSE or VACUOUS: FALSE means the scored path moved and
/// every conclusion resting on that lane needs re-deriving, and VACUOUS means
/// the receipt never observed the route at all. Either is a build breaker.
#[test]
fn no_lane_membership_claim_is_false_or_vacuous() {
    let world = hermetic_world();
    let mut checked = 0;
    for entry in ledger() {
        if entry.claim.kind != ClaimKind::LaneMembership {
            continue;
        }
        checked += 1;
        let outcome = evaluate(&entry.claim, &world);
        assert!(
            !matches!(outcome.status, Status::False | Status::Vacuous),
            "lane claim {} is {}: {}\nA benchmark's scored route moved (or was never observed). \
             Every result that assumed this lane needs re-deriving.",
            entry.claim.id,
            outcome.status.label(),
            outcome.headline
        );
    }
    assert!(checked > 0, "no lane-membership claim is under test");
}

/// The interventional rule, checked on the real ledger rather than on a
/// fixture: no A/B may be filed without an engagement observable.
#[test]
fn every_interventional_claim_declares_an_engagement_witness() {
    for entry in ledger() {
        if entry.claim.kind == ClaimKind::Interventional {
            assert!(
                !entry.claim.engagement.is_empty(),
                "{} is interventional with no witness",
                entry.claim.id
            );
        }
    }
}

/// The three-plus-two worked examples, pinned. These are REAL claims from the
/// 2026-08-21/22 session and each one stands in for a specific reversal; if a
/// classification changes, either the world moved (fine — update this with the
/// receipt that shows it) or the classifier regressed (not fine).
#[test]
fn the_worked_examples_classify_as_recorded() {
    let world = hermetic_world();
    let expected = [
        // A bank lookup, insensitive to the binary: survives every rebuild.
        ("cifar100-idx8600-banked-unsat", Status::True),
        // Reversal 1: a measured bound whose verifier moved underneath it.
        ("margin-row-root-bound-idx8600", Status::Stale),
        // Reversals 4+5: the predicate holds, the lane never ran.
        ("m3-authority-preserves-the-bank", Status::Vacuous),
        // A real ledger claim that is simply wrong at the bank.
        ("root-parity-front-is-on-a-failing-row", Status::False),
        // Lane membership from a receipt that cannot name its own binary.
        ("traffic-signs-scored-route", Status::Stale),
    ];
    let loaded = ledger();
    for (id, status) in expected {
        let entry = loaded
            .iter()
            .find(|entry| entry.claim.id == id)
            .unwrap_or_else(|| panic!("claim {id} is missing from the ledger"));
        let outcome = evaluate(&entry.claim, &world);
        assert_eq!(
            outcome.status,
            status,
            "{id} classified {} , expected {}: {}",
            outcome.status.label(),
            status.label(),
            outcome.headline
        );
    }
}

/// The VACUOUS example must name BOTH defects, not just the first one: the
/// lane produced no telemetry, and the harness never armed it. The second
/// witness is what turns "something is wrong" into "here is the line to fix".
#[test]
fn the_vacuous_example_names_the_harness_defect() {
    let world = hermetic_world();
    let loaded = ledger();
    let entry = loaded
        .iter()
        .find(|entry| entry.claim.id == "m3-authority-preserves-the-bank")
        .expect("claim present");
    let outcome = evaluate(&entry.claim, &world);

    let failed: Vec<&str> = outcome
        .witnesses
        .iter()
        .filter(|witness| !witness.holds)
        .map(|witness| witness.name.as_str())
        .collect();
    assert_eq!(
        failed,
        vec![
            "gpu_backward_lane_executed",
            "arming_lever_exported_by_the_harness"
        ],
        "both witnesses must be reported, not only the first"
    );
    // And the predicate the claim was originally believed on must still be
    // shown as holding, so nobody "fixes" the wrong thing.
    assert_eq!(outcome.observed.as_deref(), Some("9"));
}
