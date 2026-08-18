// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "external-ay")]

//! End-to-end (dark/experimental): drive a live `ay` UNSAT, then re-establish
//! its `la_generic` arithmetic leaf under ny-cert's kernel-checked Farkas
//! obligation (`farkas_premise_combination`).
//!
//! This closes the *arithmetic* half of the `ay`-UNSAT → MipCert loop: an
//! independent replay of the solver's linear refutation with no trust in `ay`.
//! The Boolean/resolution glue that composes multiple `la_generic` leaves into a
//! whole-subdomain verdict (the `MipCert.pattern_tree_cover` case-split cover)
//! is the documented remaining piece — see `docs/AY_UNSAT_NY_CERT_LOOP.md`.
//!
//! This live-tool contract is an explicit `external-ay` conformance lane. When
//! selected, it requires the exact `ay` revision pinned by `ny-mip`; absence,
//! revision drift, and unsupported proof publication all fail the test.

use ny_cert::alethe_bridge::bridge_la_generic;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Select `$NY_AY` when explicitly set; otherwise let `Command` resolve `ay`
/// on `PATH`. Validation remains the responsibility of `require_pinned_ay`.
fn selected_ay() -> PathBuf {
    std::env::var_os("NY_AY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ay"))
}

fn pinned_ay_revision() -> &'static str {
    let manifest = include_str!("../../ny-mip/Cargo.toml");
    let dependency = manifest
        .lines()
        .find(|line| line.trim_start().starts_with("ay-milp ="))
        .expect("ny-mip must declare ay-milp");
    let revision = dependency
        .split_once("rev = \"")
        .and_then(|(_, tail)| tail.split_once('"'))
        .map(|(revision, _)| revision)
        .expect("ay-milp must remain revision-pinned");
    assert_eq!(revision.len(), 40, "AY revision must be a full commit SHA");
    revision
}

fn ay_build_commit(ay: &Path) -> Option<String> {
    let out = Command::new(ay).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("build.commit="))
        .map(str::to_owned)
}

fn require_pinned_ay() -> PathBuf {
    let ay = selected_ay();
    let expected = pinned_ay_revision();
    let actual = ay_build_commit(&ay);
    assert_eq!(
        actual.as_deref(),
        Some(expected),
        "live ay-unsat bridge requires pinned AY revision {expected} at {}; \
         set NY_AY to the pinned executable (got {actual:?})",
        ay.display()
    );
    ay
}

/// Original-assertion-only QF_LRA contradiction. Keeping the arithmetic leaf
/// free of Boolean preprocessing is intentional: AY must not publish a proof
/// whose reachable assumptions depend on preprocessing-derived terms.
const QF_LRA_UNSAT_SMT2: &str = r#"(set-logic QF_LRA)
(declare-const x Real)
(assert (>= x (/ 1.0 1.0)))
(assert (<= x (/ 2.0 1.0)))
(assert (< x (/ 1.0 2.0)))
(check-sat)
"#;

#[test]
#[cfg(feature = "external-ay")]
fn live_ay_unsat_la_generic_bridges_to_ny_cert_farkas() {
    let ay = require_pinned_ay();
    let dir = tempfile::tempdir().expect("scratch dir");
    let smt = dir.path().join("qf_lra_unsat.smt2");
    let alethe = dir.path().join("qf_lra_unsat.alethe");
    std::fs::write(&smt, QF_LRA_UNSAT_SMT2).expect("write query");

    let out = Command::new(&ay)
        .arg("solve")
        .arg("--proof")
        .arg(&alethe)
        .arg(&smt)
        .output()
        .expect("run ay");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.lines().any(|l| l.trim() == "unsat"),
        "ay did not publish UNSAT (status={}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status
    );

    let proof = std::fs::read_to_string(&alethe).expect("ay wrote an Alethe proof");
    // Independent replay: bridge ay's la_generic refutation to a ny-cert Farkas
    // obligation and check it with the corpus' kernel-checked combination.
    let witness = bridge_la_generic(&proof).expect("la_generic bridges + Farkas-checks");
    assert!(
        !witness.is_positive(),
        "Farkas contradiction constant must be ≤ 0, got {witness:?}"
    );
}
