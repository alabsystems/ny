// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The ORDER of the checks in `verify::evaluate` is the mechanism, so it is
//! pinned here rather than left to a comment.
//!
//! Each test names the reversal it is standing in for. If one of these ever
//! goes red, the classifier has quietly reverted to a design that would have
//! shipped the M3 flip.

use ny_claims::{
    evaluate, ArtifactRef, ArtifactRole, BinaryFacts, Claim, ClaimKind, ClaimProvenance, Extractor,
    Occurrence, Predicate, Sensitivity, Status, Witness, World, SCHEMA,
};

const PINNED_STAMP: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const OTHER_STAMP: &str = "2222222222222222222222222222222222222222222222222222222222222222";

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    /// One results file with the exact shape of the voided M3 A/B: three
    /// repetitions, every one an `unsat` matching the bank, every one with
    /// `gpu_lines=0`.
    fn m3_shaped() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("results.csv"),
            "idx_8600,r1,unsat,expansions=39,root_bound=-0.176890,gpu_lines=0,,kill=0\n\
             idx_8600,r2,unsat,expansions=39,root_bound=-0.176890,gpu_lines=0,,kill=0\n\
             idx_8600,r3,unsat,expansions=39,root_bound=-0.176890,gpu_lines=0,,kill=0\n",
        )
        .expect("write");
        Self { dir }
    }

    fn sha(&self, name: &str) -> String {
        let bytes = std::fs::read(self.dir.path().join(name)).expect("read");
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    }

    fn world(&self, stamp: Option<&str>) -> World {
        World::with_binary(
            self.dir.path().to_path_buf(),
            stamp.map(|stamp| BinaryFacts {
                stamp: stamp.to_string(),
                version: "0.3.test".to_string(),
                source: "test".to_string(),
            }),
        )
    }
}

/// The M3 claim, reconstructed: predicate holds (three unsats), witness fails
/// (the lane never printed a line).
fn m3_claim(fixture: &Fixture) -> Claim {
    Claim {
        schema: SCHEMA.to_string(),
        id: "m3".to_string(),
        assertion: "the armed device lane reproduces the banked verdicts".to_string(),
        kind: ClaimKind::Interventional,
        asserted_on: "2026-08-22".to_string(),
        asserted_by: "test".to_string(),
        artifacts: vec![ArtifactRef {
            path: "results.csv".to_string(),
            sha256: fixture.sha("results.csv"),
            role: ArtifactRole::Evidence,
            note: String::new(),
        }],
        engagement: vec![Witness {
            name: "gpu_backward_lane_executed".to_string(),
            extract: Extractor::KeyValue {
                artifact: "results.csv".to_string(),
                key: "gpu_lines".to_string(),
                in_lines_containing: None,
                occurrence: Occurrence::Max,
            },
            require: Predicate::AtLeast { min: 1.0 },
            why: "the device lane printed at least one telemetry line".to_string(),
        }],
        extract: Extractor::CountLinesContaining {
            artifact: "results.csv".to_string(),
            needle: ",unsat,".to_string(),
        },
        predicate: Predicate::AtLeast { min: 3.0 },
        provenance: ClaimProvenance {
            binary_stamp: Some(PINNED_STAMP.to_string()),
            ..ClaimProvenance::default()
        },
        sensitivity: Sensitivity {
            binary: true,
            host: false,
            route: false,
            levers: false,
            preset: false,
        },
        refutation: "a run reports a verdict other than unsat".to_string(),
        notes: String::new(),
        supersedes: Vec::new(),
        retracted: None,
    }
}

/// **THE test.** Reversal 5. The predicate is satisfied — nine (here three)
/// runs, every verdict matching the bank — and it is satisfied because the
/// lane never ran. A classifier that checks the predicate first prints TRUE
/// and the M3 flip ships.
#[test]
fn engagement_is_checked_before_the_predicate() {
    let fixture = Fixture::m3_shaped();
    let claim = m3_claim(&fixture);

    // The predicate really does hold, in isolation.
    let bare = claim.predicate.test(
        &claim
            .extract
            .run(fixture.dir.path())
            .expect("extract")
            .value,
    );
    assert!(bare.holds, "the predicate must genuinely hold: {bare:?}");

    // And the claim is nonetheless VACUOUS.
    let outcome = evaluate(&claim, &fixture.world(Some(PINNED_STAMP)));
    assert_eq!(outcome.status, Status::Vacuous, "{outcome:#?}");
    assert!(outcome.headline.contains("gpu_backward_lane_executed"));
    assert!(
        outcome.headline.contains("fix the harness"),
        "the verdict must say what to do: {}",
        outcome.headline
    );
}

/// Vacuity outranks staleness. A stale verdict says "re-run it"; re-running a
/// vacuous experiment reproduces the vacuum, so the harness defect has to be
/// the thing that gets reported.
#[test]
fn vacuity_outranks_staleness() {
    let fixture = Fixture::m3_shaped();
    let claim = m3_claim(&fixture);
    // Binary moved AND the witness fails.
    let outcome = evaluate(&claim, &fixture.world(Some(OTHER_STAMP)));
    assert_eq!(outcome.status, Status::Vacuous, "{outcome:#?}");
}

/// Reversal 1. The number still extracts and the predicate still holds; only
/// the binary moved, and the claim declared itself binary-sensitive.
#[test]
fn a_moved_binary_makes_a_binary_sensitive_claim_stale() {
    let fixture = Fixture::m3_shaped();
    let mut claim = m3_claim(&fixture);
    claim.engagement.clear();
    claim.kind = ClaimKind::Observational;

    assert_eq!(
        evaluate(&claim, &fixture.world(Some(PINNED_STAMP))).status,
        Status::True
    );
    let outcome = evaluate(&claim, &fixture.world(Some(OTHER_STAMP)));
    assert_eq!(outcome.status, Status::Stale, "{outcome:#?}");
    assert!(outcome.headline.contains("binary moved"), "{outcome:#?}");
}

/// Sensitivity SCOPES staleness. Without this, everything is stale after the
/// first rebuild and the column stops being read — which is how a discipline
/// dies.
#[test]
fn sensitivity_scopes_staleness() {
    let fixture = Fixture::m3_shaped();
    let mut claim = m3_claim(&fixture);
    claim.engagement.clear();
    claim.kind = ClaimKind::Observational;
    claim.sensitivity.binary = false;

    let outcome = evaluate(&claim, &fixture.world(Some(OTHER_STAMP)));
    assert_eq!(
        outcome.status,
        Status::True,
        "a claim that does not depend on the binary must survive a rebuild: {outcome:#?}"
    );
}

/// "I could not tell" must never be scored as "it matches". This is the exact
/// pattern that made reversal 5's A/B look green.
#[test]
fn an_unknown_current_binary_is_drift_not_agreement() {
    let fixture = Fixture::m3_shaped();
    let mut claim = m3_claim(&fixture);
    claim.engagement.clear();
    claim.kind = ClaimKind::Observational;

    let outcome = evaluate(&claim, &fixture.world(None));
    assert_eq!(outcome.status, Status::Stale, "{outcome:#?}");
    assert!(
        outcome.headline.contains("unknown is not agreement"),
        "{outcome:#?}"
    );
}

/// Moved evidence beats everything below grounding: if the bytes are not the
/// bytes, nothing read out of them supports anything.
#[test]
fn moved_artifact_bytes_short_circuit_to_stale() {
    let fixture = Fixture::m3_shaped();
    let claim = m3_claim(&fixture);
    std::fs::write(
        fixture.dir.path().join("results.csv"),
        "idx_8600,r1,sat,expansions=39,root_bound=-9.9,gpu_lines=7,,kill=0\n",
    )
    .expect("rewrite");

    let outcome = evaluate(&claim, &fixture.world(Some(PINNED_STAMP)));
    assert_eq!(outcome.status, Status::Stale, "{outcome:#?}");
    assert!(
        outcome
            .headline
            .contains("not the evidence this claim was made from"),
        "{outcome:#?}"
    );
    // And specifically NOT vacuous/false on the strength of bytes it should
    // not have been reading.
    assert!(outcome.witnesses.is_empty());
}

/// A missing artifact is UNGROUNDED, never FALSE. "I cannot check this" is a
/// distinct and useful answer; rounding it to FALSE invites a pointless fix,
/// and rounding it to TRUE is the disease.
#[test]
fn a_missing_artifact_is_ungrounded() {
    let fixture = Fixture::m3_shaped();
    let claim = m3_claim(&fixture);
    std::fs::remove_file(fixture.dir.path().join("results.csv")).expect("remove");

    let outcome = evaluate(&claim, &fixture.world(Some(PINNED_STAMP)));
    assert_eq!(outcome.status, Status::Ungrounded, "{outcome:#?}");
}

/// An extractor may not read bytes the claim did not pin. Otherwise the
/// artifact-integrity check has a hole exactly the size of the number being
/// claimed.
#[test]
fn an_extractor_reading_an_uncited_artifact_is_ungrounded() {
    let fixture = Fixture::m3_shaped();
    std::fs::write(fixture.dir.path().join("other.csv"), "x=1\n").expect("write");
    let mut claim = m3_claim(&fixture);
    claim.extract = Extractor::KeyValue {
        artifact: "other.csv".to_string(),
        key: "x".to_string(),
        in_lines_containing: None,
        occurrence: Occurrence::Unique,
    };

    let outcome = evaluate(&claim, &fixture.world(Some(PINNED_STAMP)));
    assert_eq!(outcome.status, Status::Ungrounded, "{outcome:#?}");
    assert!(outcome.headline.contains("unpinned"), "{outcome:#?}");
}

/// A grounded, engaged, current claim whose predicate fails is FALSE — the
/// only path to FALSE there is.
#[test]
fn only_a_failing_predicate_yields_false() {
    let fixture = Fixture::m3_shaped();
    let mut claim = m3_claim(&fixture);
    claim.engagement[0].require = Predicate::AtMost { max: 0.0 };
    claim.predicate = Predicate::AtLeast { min: 4.0 };

    let outcome = evaluate(&claim, &fixture.world(Some(PINNED_STAMP)));
    assert_eq!(outcome.status, Status::False, "{outcome:#?}");
}
