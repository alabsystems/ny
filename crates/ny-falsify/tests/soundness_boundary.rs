// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! REQUIRED TEST (c): the crate cannot name a verdict type.
//!
//! ny's one existing structural example is `ny_mip::SignSpaceOutcome`, pinned
//! by `the_lane_cannot_produce_a_verified_outcome`. That test proves the
//! technique works and that it was applied to exactly one lane. Here the same
//! property is an invariant of the whole crate, argued three independent ways:
//! the return TYPE (M1), the crate GRAPH (M2), and the candidate's PAYLOAD (M3).

use ny_falsify::{Decline, Effort, Proposal, SpecShape, Strategy};
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Rust source with `//`-comments removed, lower-cased. Comments must be
/// stripped: this crate's documentation necessarily discusses the verdict types
/// it is forbidden from naming.
fn code_only(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
        .lines()
        .map(|line| match line.find("//") {
            Some(index) => &line[..index],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase()
}

fn source_files() -> Vec<PathBuf> {
    fn walk(directory: &Path, into: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, into);
            } else if path.extension().is_some_and(|e| e == "rs") {
                into.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(&crate_root().join("src"), &mut files);
    assert!(
        files.len() >= 9,
        "expected the whole crate, found {files:?}"
    );
    files
}

#[test]
fn m1_the_return_type_cannot_express_a_verdict() {
    // An exhaustive match. Adding a `Verified`/`Unsat`/`Sat` variant to
    // `Proposal` breaks this compile, which is the point: the boundary is a
    // build failure rather than a review comment.
    fn is_falsification_only(proposal: &Proposal) -> bool {
        match proposal {
            Proposal::Candidate(_) | Proposal::Exhausted(_) | Proposal::Declined(_) => true,
        }
    }

    let variants = [
        Proposal::Exhausted(Effort::default()),
        Proposal::Declined(Decline::Disarmed),
        Proposal::Declined(Decline::SpecShapeUnsupported {
            want: SpecShape::BoxInputs,
            got: SpecShape::NonBoxInputAssertions {
                non_box_assertions: 7,
            },
        }),
    ];
    for variant in &variants {
        assert!(is_falsification_only(variant));
        assert!(variant.is_falsification_only());
    }
}

#[test]
fn m2_the_crate_graph_forbids_the_type() {
    // `VnncompResult` lives in `ny-cli`. This crate depends on NOTHING -- not
    // `ny-cli`, not anything that re-exports from it, not anything at all --
    // so the type is not nameable here by any import, alias or associated
    // type. `cargo tree -p ny-falsify -e normal` is one line.
    let manifest = std::fs::read_to_string(crate_root().join("Cargo.toml")).unwrap();

    let mut in_dependency_table = false;
    let mut entries: Vec<&str> = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_dependency_table = trimmed == "[dependencies]"
                || trimmed == "[dev-dependencies]"
                || trimmed == "[build-dependencies]"
                || trimmed.ends_with(".dependencies]");
            continue;
        }
        if in_dependency_table && !trimmed.is_empty() && !trimmed.starts_with('#') {
            entries.push(trimmed);
        }
    }
    assert!(
        entries.is_empty(),
        "ny-falsify must have no dependencies at all; found {entries:?}"
    );
    assert!(
        !manifest.contains("path = \"../"),
        "a workspace path dependency would make a verdict type nameable"
    );
}

#[test]
fn m3_no_source_file_names_a_verdict_anywhere() {
    // Belt and braces over M2: even a hand-written local `enum Verdict` or a
    // string-typed `"unsat"` would be a way for this crate to start expressing
    // conclusions. It has none.
    for file in source_files() {
        let code = code_only(&file);
        for forbidden in [
            "vnncompresult",
            "verdict",
            "unsat",
            "verified",
            "counterexample_proved",
            "is_sat",
        ] {
            assert!(
                !code.contains(forbidden),
                "`{forbidden}` appears in code (not a comment) in {}",
                file.display()
            );
        }
    }
}

#[test]
fn m3_a_candidate_carries_inputs_and_nothing_else() {
    // The `Y_j` coordinates of a published witness must come from a real ORT
    // forward on the ORIGINAL graph, performed once in the caller's publication
    // path. If a candidate could carry outputs, a strategy's own arithmetic
    // could reach a witness. It cannot: there is no field to put it in.
    let domain = ny_falsify::SearchBox::new(&[0.0, 0.0], &[1.0, 1.0]).unwrap();
    let mut oracle = Oracle;
    let mut state = ny_falsify::SearchState::at_centre(&domain);
    let budget = ny_falsify::Budget {
        deadline: std::time::Instant::now() + std::time::Duration::from_secs(5),
        batch: 8,
        params: ny_falsify::ParamSpace {
            free_dims_ceiling: usize::MAX,
            max_points: 64,
            max_restarts: 1,
        },
        stall_rule: ny_falsify::StallRule::new(ny_falsify::WorkUnit::BatchesWithoutNewBest, 1),
    };

    struct Oracle;
    impl ny_falsify::Oracle for Oracle {
        fn evaluate_batch(
            &mut self,
            points: &[Vec<f64>],
        ) -> Result<Vec<ny_falsify::Score>, ny_falsify::OracleError> {
            // Everything holds: the most permissive oracle possible, so if a
            // candidate could ever carry more than inputs, it would here.
            Ok(points
                .iter()
                .map(|_| ny_falsify::Score {
                    steer: 1.0,
                    holds: true,
                })
                .collect())
        }
        fn batch_limit(&self) -> usize {
            8
        }
    }

    let proposal =
        ny_falsify::strategies::SpecialPoints.search(&domain, &mut oracle, &budget, &mut state);
    let Proposal::Candidate(candidate) = proposal else {
        panic!("expected a candidate");
    };

    assert_eq!(candidate.inputs().len(), domain.dims());
    let rendered = format!("{candidate:?}").to_lowercase();
    assert!(rendered.starts_with("candidate { inputs:"));
    for forbidden in ["output", "margin:", "verdict", "sat"] {
        assert!(
            !rendered.contains(forbidden),
            "a Candidate rendered `{forbidden}`: {rendered}"
        );
    }
}

#[test]
fn m4_the_publication_seam_is_not_in_this_crate() {
    // The gate is `gate_sat_with_trusted_oracle` in `ny-cli`, and this work
    // does not touch it. Assert that no part of this crate so much as mentions
    // it in code: the seam stays exactly where it is, unchanged, with the same
    // ORT re-confirm, the same true-f64 recheck and the same downgrade path.
    for file in source_files() {
        let code = code_only(&file);
        for forbidden in ["gate_sat", "trusted_oracle", "rehydrate", "ort_forward"] {
            assert!(
                !code.contains(forbidden),
                "`{forbidden}` appears in code in {}",
                file.display()
            );
        }
    }
}
