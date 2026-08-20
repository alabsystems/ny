// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! REQUIRED TEST (d): defaults are unchanged.
//!
//! Asserted on CODE PATHS and on the BUILD GRAPH, never on a log line -- the
//! same standard `unset_lever_never_constructs_the_ste_pgd_search` holds the
//! BNN lanes to. Four independent witnesses:
//!
//! 1. the arming type's `Default` is `Dark`;
//! 2. an unarmed registry declines every strategy and runs no search at all --
//!    the oracle is never called, so the darkness is not a filtered result;
//! 3. `ny-cli` -- and ONLY `ny-cli` -- depends on this crate, the edge is one
//!    way, and the lane it feeds is dark behind a declared lever whose
//!    declaration default is `false`;
//! 4. the crate is not a workspace default member and reads no process
//!    environment, so neither the default build nor the shipped lever surface
//!    changed.

#[path = "fixtures/support.rs"]
mod support;

use ny_falsify::strategies::{SpecialPoints, Square};
use ny_falsify::{Admission, Arming, Decline, ObjectiveQuality, Registry, Score};
use std::path::PathBuf;
use std::time::Duration;
use support::{box_spec, CountingLadder, PredicateOracle};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn arming_defaults_to_dark() {
    assert_eq!(Arming::default(), Arming::Dark);
    assert_eq!(Registry::default().arming(), Arming::Dark);
    assert_eq!(Registry::new().arming(), Arming::Dark);
    assert!(Registry::new().names().is_empty());
    // Registering a strategy does not arm it.
    let registry = Registry::new()
        .with(Box::new(SpecialPoints))
        .with(Box::new(Square::default()));
    assert_eq!(registry.arming(), Arming::Dark);
    assert_eq!(registry.names().len(), 2);
}

#[test]
fn an_unarmed_registry_declines_everything_and_never_calls_the_oracle() {
    let domain = ny_falsify::SearchBox::new(&[0.0; 16], &[1.0; 16]).unwrap();
    let mut ladder = CountingLadder::new(box_spec(16));
    // An oracle for which EVERY point violates. If anything ran, it would find
    // a candidate immediately -- so "no candidate" cannot be a weak search.
    let mut oracle = PredicateOracle::new(|_: &[f64]| Score {
        steer: 1.0,
        holds: true,
    });

    let mut registry = Registry::new()
        .with(Box::new(SpecialPoints))
        .with(Box::new(Square::default()));

    let receipt = registry
        .run(
            &mut ladder,
            &domain,
            &mut oracle,
            Duration::from_secs(10),
            ObjectiveQuality::Informative,
        )
        .expect("the ladder still runs; it is admission that declines");

    assert_eq!(receipt.admissions.len(), 2);
    for admission in &receipt.admissions {
        assert_eq!(
            admission.admission,
            Admission::Declined(Decline::Disarmed),
            "{} was not dark by default",
            admission.strategy
        );
    }
    assert!(receipt.proposals.is_empty(), "no strategy may have run");
    assert!(receipt.candidate().is_none());
    assert_eq!(oracle.calls, 0, "the oracle was called by a dark lane");
    assert_eq!(oracle.points, 0);
    assert_eq!(ladder.graph_calls, 0);
    assert_eq!(ladder.model_calls, 0);

    // And the same registry, armed, does find it -- so the assertions above are
    // about arming and not about a broken wiring.
    let mut registry = registry.armed();
    let receipt = registry
        .run(
            &mut ladder,
            &domain,
            &mut oracle,
            Duration::from_secs(10),
            ObjectiveQuality::Informative,
        )
        .unwrap();
    assert!(receipt.candidate().is_some());
    assert!(oracle.calls > 0);
}

#[test]
fn only_ny_cli_may_route_this_to_a_verdict_and_the_edge_is_one_way() {
    // The crate IS wired in now: `ny-cli` owns the publication seam, so it is
    // the one crate allowed to hold this edge. What must never change is the
    // DIRECTION. `ny-falsify` declares no dependency at all -- not on `ny-cli`,
    // not on anything -- so `VnncompResult` and `gate_sat_with_trusted_oracle`
    // stay unnameable from a strategy no matter what the caller does.
    let own = std::fs::read_to_string(repo_root().join("crates/ny-falsify/Cargo.toml")).unwrap();
    let deps = own
        .split("[dependencies]")
        .nth(1)
        .expect("the manifest declares a dependencies table")
        .split("[dev-dependencies]")
        .next()
        .unwrap();
    assert!(
        deps.lines()
            .all(|line| line.trim().is_empty() || line.trim_start().starts_with('#')),
        "ny-falsify gained a dependency; the soundness argument (M2) rests on it having none:\n{deps}"
    );

    let manifest = std::fs::read_to_string(repo_root().join("crates/ny-cli/Cargo.toml")).unwrap();
    assert!(
        manifest.contains("ny-falsify = { path = \"../ny-falsify\" }"),
        "ny-cli is the crate that wires the portfolio to the scored path; it must hold the edge"
    );

    // No OTHER crate may route it there. A second route would be a second
    // publication seam, and the whole argument is that there is exactly one.
    for crate_name in ["ny-mip", "ny-propagate", "ny-onnx", "ny-python", "ny-api"] {
        let path = repo_root()
            .join("crates")
            .join(crate_name)
            .join("Cargo.toml");
        let manifest = std::fs::read_to_string(&path).unwrap();
        assert!(
            !manifest.contains("ny-falsify"),
            "{crate_name} gained a dependency on ny-falsify"
        );
    }
}

#[test]
fn the_scored_lane_that_now_exists_is_dark_in_its_declaration() {
    // The lane is wired, so "dark by default" is no longer a fact about the
    // build graph -- it is a fact about a declaration. Read the declaration
    // itself rather than the reader, so this test fails if someone flips the
    // shipped default without a sweep.
    let decls =
        std::fs::read_to_string(repo_root().join("crates/ny-levers/src/decls/dark_probes.rs"))
            .unwrap();
    let lane = decls
        .split("name: \"NY_FALSIFY_PORTFOLIO\",")
        .nth(1)
        .expect("the portfolio lane is declared in dark_probes.rs")
        .split("};")
        .next()
        .unwrap();
    assert!(
        lane.contains("default: DefaultSpec::Bool(false)"),
        "NY_FALSIFY_PORTFOLIO must ship dark; defaults change only on a measurement"
    );
    assert!(
        lane.contains("moat: MoatRisk::High"),
        "a new sat source on the scored path is MoatRisk::High"
    );

    // And there is no typed preset key for it, which is what keeps it off a
    // competition harness entirely: the harness exports no NY_* variables.
    let cli = std::fs::read_to_string(
        repo_root().join("crates/ny-cli/src/commands/vnncomp/falsify_portfolio.rs"),
    )
    .unwrap();
    assert!(
        !cli.contains("read_over_config"),
        "the portfolio lane must stay environment-only until a family sweep measures a win"
    );
}

#[test]
fn the_default_build_is_unchanged() {
    let workspace = std::fs::read_to_string(repo_root().join("Cargo.toml")).unwrap();
    let default_members = workspace
        .split("default-members = [")
        .nth(1)
        .expect("workspace declares default-members")
        .split(']')
        .next()
        .unwrap();
    assert!(
        !default_members.contains("ny-falsify"),
        "ny-falsify joined default-members; `cargo build` at the root now builds it"
    );
    assert!(
        workspace.contains("\"crates/ny-falsify\""),
        "the crate must still be a workspace member so `cargo test -p` reaches it"
    );
}

#[test]
fn the_crate_reads_no_process_environment() {
    // The lever ratchet forbids a raw process-environment read outside a
    // declaration in `ny-levers`. This crate sidesteps the question entirely:
    // arming arrives as a typed argument from the caller, so there is nothing
    // here for the ratchet to count and no new lever to declare until something
    // actually routes this onto the scored path.
    fn walk(directory: &std::path::Path, into: &mut Vec<PathBuf>) {
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
    walk(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    for file in files {
        let source = std::fs::read_to_string(&file).unwrap();
        for forbidden in ["std::env", "env::var", "var_os", "env!(", "option_env!"] {
            assert!(
                !source.contains(forbidden),
                "`{forbidden}` appears in {}",
                file.display()
            );
        }
    }
}
