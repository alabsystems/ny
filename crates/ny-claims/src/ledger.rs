// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The on-disk ledger: `claims/*.claim.json`, and the lint that decides which
//! records are even admissible.
//!
//! # The lint is the preregistration gate
//!
//! Three of the five reversals were *interventional* claims — an A/B, a lever,
//! a lane — whose intervention was never observed. One rule closes that class
//! structurally:
//!
//! > an [`crate::ClaimKind::Interventional`] claim MUST declare at least one
//! > engagement witness.
//!
//! Not "should". The record does not lint, so the claim cannot be filed, so
//! `ny claims verify` reports it as inadmissible and the gate is red. A
//! vacuous null stops being something you have to remember to look for and
//! becomes something you cannot write down.
//!
//! Preregistration is the same file, written BEFORE the run with its artifact
//! pins empty. `ny claims verify` then reports it UNGROUNDED — "declared, not
//! yet measured" — which is exactly the state a preregistered hypothesis is
//! in, and it is visible in the same table as everything else.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::claim::{Claim, ClaimKind, SCHEMA};

/// A claim plus where it was loaded from.
#[derive(Debug, Clone)]
pub struct LoadedClaim {
    pub path: PathBuf,
    pub claim: Claim,
}

/// One admissibility failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub id: String,
    pub message: String,
}

/// Load every `*.claim.json` in `dir`, sorted by id.
///
/// A file that does not parse is a hard error naming the file: a claim that
/// silently vanished from the ledger because its JSON was malformed would be
/// the same disease in a new place.
pub fn load_dir(dir: &Path) -> Result<Vec<LoadedClaim>> {
    let mut loaded = Vec::new();
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("read the claim ledger at {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        if !path.to_string_lossy().ends_with(".claim.json") {
            continue;
        }
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let claim: Claim = serde_json::from_str(&text)
            .with_context(|| format!("parse {} as {SCHEMA}", path.display()))?;
        loaded.push(LoadedClaim { path, claim });
    }
    loaded.sort_by(|left, right| left.claim.id.cmp(&right.claim.id));
    Ok(loaded)
}

/// Admissibility. Every finding here is a defect in the CLAIM, not in the
/// world — the difference between "this record is not a claim" and "this
/// claim is false".
pub fn lint_claim(loaded: &LoadedClaim) -> Vec<LintFinding> {
    let claim = &loaded.claim;
    let mut findings = Vec::new();
    let mut fail = |message: String| {
        findings.push(LintFinding {
            id: claim.id.clone(),
            message,
        });
    };

    if claim.schema != SCHEMA {
        fail(format!("schema is {:?}, expected {SCHEMA}", claim.schema));
    }
    let stem = loaded
        .path
        .file_name()
        .map(|name| name.to_string_lossy().replace(".claim.json", ""))
        .unwrap_or_default();
    if stem != claim.id {
        fail(format!(
            "id {:?} does not match the file name {:?}; prose cites the id, so the two must agree",
            claim.id, stem
        ));
    }
    if claim.assertion.trim().is_empty() {
        fail("assertion is empty".to_string());
    }
    if claim.refutation.trim().is_empty() {
        fail(
            "refutation is empty: a claim whose author cannot name what would falsify it is not a \
             claim"
                .to_string(),
        );
    }
    if claim.artifacts.is_empty() {
        fail(
            "no artifacts cited: a claim with no evidence is prose with extra steps (a \
             PREREGISTERED claim is the one legitimate case and is reported UNGROUNDED, not \
             inadmissible — remove this lint only by measuring)"
                .to_string(),
        );
    }
    for artifact in &claim.artifacts {
        if artifact.sha256.len() != 64 || !artifact.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
            fail(format!(
                "artifact {} has no 64-hex sha256 pin ({:?}); run `ny claims pin`",
                artifact.path, artifact.sha256
            ));
        }
        if Path::new(&artifact.path).is_absolute() {
            fail(format!(
                "artifact {} is an absolute path; nobody else can re-check it",
                artifact.path
            ));
        }
    }

    // THE RULE. See the module docs.
    if claim.kind == ClaimKind::Interventional && claim.engagement.is_empty() {
        fail(
            "an interventional claim must declare at least one engagement witness — the \
             observable proving the intervention actually occurred. Reversals 4 and 5 were both \
             interventional claims whose intervention never happened, and both would have been \
             refused here"
                .to_string(),
        );
    }

    // Every artifact an extractor reads must be cited and therefore pinned.
    let cited: Vec<&str> = claim.artifacts.iter().map(|a| a.path.as_str()).collect();
    for artifact in std::iter::once(claim.extract.artifact())
        .chain(claim.engagement.iter().map(|w| w.extract.artifact()))
    {
        if !cited.contains(&artifact) {
            fail(format!(
                "an extractor reads {artifact}, which the claim does not cite, so those bytes are \
                 unpinned"
            ));
        }
    }

    // A binary- or host-sensitive claim that recorded neither is not
    // checkable along the dimension it says it depends on.
    if claim.sensitivity.host && claim.provenance.host.is_none() {
        fail(
            "sensitivity.host is set but provenance.host is missing: the claim says it depends on \
             the machine and does not say which machine"
                .to_string(),
        );
    }

    findings
}

/// Lint the whole ledger, including cross-record checks.
pub fn lint_all(loaded: &[LoadedClaim]) -> Vec<LintFinding> {
    let mut findings: Vec<LintFinding> = loaded.iter().flat_map(lint_claim).collect();
    let ids: Vec<&str> = loaded.iter().map(|entry| entry.claim.id.as_str()).collect();
    for entry in loaded {
        for superseded in &entry.claim.supersedes {
            if !ids.contains(&superseded.as_str()) {
                findings.push(LintFinding {
                    id: entry.claim.id.clone(),
                    message: format!(
                        "supersedes {superseded:?}, which is not in the ledger; a reader chasing \
                         the old id would find nothing"
                    ),
                });
            }
        }
    }
    findings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim::{ArtifactRef, ArtifactRole, ClaimProvenance, Sensitivity, Witness};
    use crate::extract::{Extractor, Occurrence};
    use crate::predicate::Predicate;

    fn base(kind: ClaimKind) -> LoadedClaim {
        LoadedClaim {
            path: PathBuf::from("claims/demo.claim.json"),
            claim: Claim {
                schema: SCHEMA.to_string(),
                id: "demo".to_string(),
                assertion: "something".to_string(),
                kind,
                asserted_on: "2026-08-22".to_string(),
                asserted_by: "test".to_string(),
                artifacts: vec![ArtifactRef {
                    path: "a.csv".to_string(),
                    sha256: "0".repeat(64),
                    role: ArtifactRole::Evidence,
                    note: String::new(),
                }],
                engagement: Vec::new(),
                extract: Extractor::KeyValue {
                    artifact: "a.csv".to_string(),
                    key: "x".to_string(),
                    in_lines_containing: None,
                    occurrence: Occurrence::Unique,
                },
                predicate: Predicate::Present,
                provenance: ClaimProvenance::default(),
                sensitivity: Sensitivity {
                    binary: false,
                    host: false,
                    route: false,
                    levers: false,
                    preset: false,
                },
                refutation: "x is absent".to_string(),
                notes: String::new(),
                supersedes: Vec::new(),
                retracted: None,
            },
        }
    }

    /// THE structural rule. An interventional claim with no witness must not
    /// be admissible — this is the lint that would have refused the M3 A/B.
    #[test]
    fn an_interventional_claim_without_a_witness_is_inadmissible() {
        let findings = lint_claim(&base(ClaimKind::Interventional));
        assert!(
            findings
                .iter()
                .any(|finding| finding.message.contains("engagement witness")),
            "{findings:?}"
        );
        // The identical record as an observational claim is fine.
        assert!(lint_claim(&base(ClaimKind::Observational)).is_empty());
    }

    #[test]
    fn an_interventional_claim_with_a_witness_is_admissible() {
        let mut loaded = base(ClaimKind::Interventional);
        loaded.claim.engagement.push(Witness {
            name: "lane_executed".to_string(),
            extract: Extractor::KeyValue {
                artifact: "a.csv".to_string(),
                key: "gpu_lines".to_string(),
                in_lines_containing: None,
                occurrence: Occurrence::Max,
            },
            require: Predicate::AtLeast { min: 1.0 },
            why: "the device lane printed at least one line".to_string(),
        });
        assert!(lint_claim(&loaded).is_empty(), "{:?}", lint_claim(&loaded));
    }

    #[test]
    fn an_uncited_artifact_is_inadmissible() {
        let mut loaded = base(ClaimKind::Observational);
        loaded.claim.extract = Extractor::KeyValue {
            artifact: "elsewhere.log".to_string(),
            key: "x".to_string(),
            in_lines_containing: None,
            occurrence: Occurrence::First,
        };
        let findings = lint_claim(&loaded);
        assert!(
            findings.iter().any(|f| f.message.contains("unpinned")),
            "{findings:?}"
        );
    }

    #[test]
    fn an_unpinned_artifact_is_inadmissible() {
        let mut loaded = base(ClaimKind::Observational);
        loaded.claim.artifacts[0].sha256 = "unpinned".to_string();
        assert!(lint_claim(&loaded)
            .iter()
            .any(|f| f.message.contains("sha256")));
    }

    #[test]
    fn a_claim_with_no_named_refutation_is_inadmissible() {
        let mut loaded = base(ClaimKind::Observational);
        loaded.claim.refutation = String::new();
        assert!(lint_claim(&loaded)
            .iter()
            .any(|f| f.message.contains("refutation")));
    }

    /// `sensitivity` has no serde default on purpose: a record that omits it
    /// must fail to parse, not silently claim insensitivity to everything.
    #[test]
    fn a_record_without_a_sensitivity_set_does_not_parse() {
        let json = r#"{
            "schema": "NY_CLAIM_V1", "id": "x", "assertion": "a",
            "kind": "observational",
            "artifacts": [{"path":"a","sha256":"0","role":"evidence"}],
            "extract": {"op":"key_value","artifact":"a","key":"k"},
            "predicate": {"op":"present"}
        }"#;
        let error = serde_json::from_str::<Claim>(json).expect_err("must not parse");
        assert!(error.to_string().contains("sensitivity"), "{error}");
    }
}
