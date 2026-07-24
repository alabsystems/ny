// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

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
//! Skips honestly (with a stderr note) when no `ay` binary is reachable
//! (`$NY_AY` or `ay` on `PATH`), so the suite stays green without the solver.

use ny_cert::alethe_bridge::bridge_la_generic;
use std::path::{Path, PathBuf};
use std::process::Command;

struct AyBinary {
    path: PathBuf,
    explicitly_selected: bool,
}

/// Locate `ay`: `$NY_AY`, then `ay` on `PATH`. Returns `None` (skip) otherwise.
fn locate_ay() -> Option<AyBinary> {
    if let Ok(p) = std::env::var("NY_AY") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(AyBinary {
                path: p,
                explicitly_selected: true,
            });
        }
    }
    let out = Command::new("sh")
        .arg("-c")
        .arg("command -v ay")
        .output()
        .ok()?;
    if out.status.success() {
        let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(AyBinary {
                path: PathBuf::from(path),
                explicitly_selected: false,
            });
        }
    }
    None
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
fn live_ay_unsat_la_generic_bridges_to_ny_cert_farkas() {
    let ay = match locate_ay() {
        Some(p) => p,
        None => {
            eprintln!("skipping ay-unsat bridge test: no `ay` solver (set NY_AY or PATH)");
            return;
        }
    };
    let expected_revision = pinned_ay_revision();
    let actual_revision = ay_build_commit(&ay.path);
    if actual_revision.as_deref() != Some(expected_revision) {
        assert!(
            !ay.explicitly_selected,
            "NY_AY must name the pinned AY revision {expected_revision}, got {actual_revision:?}"
        );
        eprintln!(
            "skipping ay-unsat bridge test: PATH ay is not pinned AY {expected_revision} (got {actual_revision:?})"
        );
        return;
    }

    let dir = std::env::temp_dir().join(format!("ny-ay-bridge-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let smt = dir.join("qf_lra_unsat.smt2");
    let alethe = dir.join("qf_lra_unsat.alethe");
    std::fs::write(&smt, QF_LRA_UNSAT_SMT2).expect("write query");

    let out = Command::new(&ay.path)
        .arg("solve")
        .arg("--proof")
        .arg(&alethe)
        .arg(&smt)
        .output()
        .expect("run ay");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success()
        && stderr.contains("atomic no-replace artifact publication is unsupported on this platform")
    {
        eprintln!(
            "skipping ay-unsat bridge test: pinned AY cannot securely publish proof artifacts on this platform"
        );
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
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

    let _ = std::fs::remove_dir_all(&dir);
}
