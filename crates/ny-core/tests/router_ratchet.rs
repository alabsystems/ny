// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ROUTER_PROGRAM §11.2 tests 18–22: the router is verdict-free BY SOURCE.
//!
//! The crate graph is necessary, not sufficient: `ny-core` cannot reach
//! `UnsatToken` or `VnncompResult`, but it DEFINES `VerificationResult`,
//! `Counterexample`, `EnclosureVerdict`, `VerificationProof` and the proof
//! profile itself. So each premise is a source check over
//! `crates/ny-core/src/router/`, in the `proof_profile_ratchet.rs` style:
//!
//! * needles are assembled at runtime, so this file cannot match itself;
//! * comment lines are stripped (documentation is not code);
//! * every scanned file must be longer than 200 bytes — an empty module
//!   compiles and scans clean, which would turn a check into a no-op;
//! * every negative carries a POSITIVE CONTROL that the same scan finds the
//!   needle where it does live, so a zero is an absence and not a typo.

use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        text.len() > 200,
        "{} is suspiciously short ({} bytes): an empty module compiles and scans clean",
        path.display(),
        text.len()
    );
    text
}

/// `text` without comment lines.
fn code_only(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The router's own sources (its unit-test module excluded: tests may spell
/// names in order to prove the policy ignores them).
fn router_sources() -> Vec<PathBuf> {
    let dir = crate_root().join("src/router");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .filter(|p| p.file_name().is_some_and(|n| n != "tests.rs"))
        .collect();
    files.sort();
    let names: Vec<String> = files
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    for expected in [
        "mod.rs",
        "decl.rs",
        "facts.rs",
        "bus.rs",
        "plan.rs",
        "checkpoint.rs",
        "project.rs",
        "hash.rs",
        "guard.rs",
        "acting.rs",
        "derive.rs",
        "ostop.rs",
        "lease.rs",
        "r4.rs",
        "safepoint.rs",
        "route.rs",
        "gap_contract.rs",
        "otrial.rs",
        "trial.rs",
        "priors.rs",
        "spec.rs",
        "admission.rs",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "router module {expected} not found in {names:?}: the scan would be VOID"
        );
    }
    files
}

/// The pure policy files (§4.1): no clock, no environment, no filesystem.
fn policy_sources() -> Vec<PathBuf> {
    // #router-r2: the acting commitments and the O-DERIVE caps are policy
    // too — replay re-derives them bit for bit. #router-r3: so are the O-STOP
    // commitments and CR-ATK's slope predicate. #router-r4: so are the lease
    // arithmetic, acting set 3's commitments and the safe-point rulings.
    // #router-r5: so are acting set 4's commitments and CR-G's arithmetic.
    // #router-r6: so are the O-TRIAL commitments and the trial state
    // machines (the lanes feed them their own clocks).
    // #router-r7: so is the offline prior's reader (the table is compiled in).
    // #admission-precision: so are the specification shape's archetype
    // mapping and the lanes' route checks (the lanes call them inline).
    [
        "plan.rs",
        "checkpoint.rs",
        "project.rs",
        "hash.rs",
        "acting.rs",
        "derive.rs",
        "ostop.rs",
        "lease.rs",
        "r4.rs",
        "safepoint.rs",
        "route.rs",
        "gap_contract.rs",
        "otrial.rs",
        "trial.rs",
        "priors.rs",
        "spec.rs",
        "admission.rs",
    ]
    .iter()
    .map(|f| crate_root().join("src/router").join(f))
    .collect()
}

fn assert_absent(files: &[PathBuf], needles: &[String], why: &str) {
    assert!(!files.is_empty(), "no files scanned: {why}");
    for file in files {
        let code = code_only(&read(file));
        for needle in needles {
            assert!(
                !code.contains(needle.as_str()),
                "{} names `{needle}`: {why}",
                file.display()
            );
        }
    }
}

fn assert_found_somewhere(rel: &str, needle: &str) {
    let text = read(&crate_root().join(rel));
    assert!(
        code_only(&text).contains(needle),
        "positive control: `{needle}` must be found in {rel}, or the scan proves nothing"
    );
}

#[test]
fn router_sources_name_no_verdict_bearing_type() {
    let needles = [
        format!("{}{}", "Verification", "Result"),
        format!("{}{}", "Counter", "example"),
        format!("{}{}", "Enclosure", "Verdict"),
        format!("{}{}", "Verification", "Proof"),
        format!("{}{}", "Unsat", "Token"),
        format!("{}{}", "Unsat", "Receipt"),
        format!("{}{}", "Certi", "fied"),
        format!("{}_{}", "authorize", "unsat"),
        format!("{}_{}", "gate", "sat"),
        format!("{}{}", "Vnncomp", "Result"),
        format!("{}_{}", "publish", "lane"),
        format!("{}{}", "wit", "ness"),
    ];
    assert_absent(
        &router_sources(),
        &needles,
        "the router is verdict-free: it may order, fund and observe lanes, never name a verdict, \
         a certificate or the publisher (ROUTER_PROGRAM §6.1 premises ii/iii)",
    );
    // Positive controls: the same scan finds the verdict types where they live.
    assert_found_somewhere("src/verification_result.rs", &needles[0]);
    assert_found_somewhere("src/counterexample.rs", &needles[1]);
    assert_found_somewhere("src/wgpu_verdict.rs", &needles[2]);
    assert_found_somewhere("src/proof.rs", &needles[3]);
}

#[test]
fn router_sources_never_name_proof_profile() {
    let needles = [
        format!("{}{}", "Proof", "Profile"),
        format!("{}{}", "Audit", "Rates"),
        format!("{}_{}", "proof", "profile"),
    ];
    assert_absent(
        &router_sources(),
        &needles,
        "plans are computed identically in both profiles (ROUTER_PROGRAM §7); the router may \
         never learn which profile runs",
    );
    assert_found_somewhere("src/proof_profile.rs", &needles[0]);
    assert_found_somewhere("src/proof_profile.rs", &needles[1]);
}

#[test]
fn router_policy_sources_have_no_clock_env_or_fs() {
    let needles = [
        format!("{}::{}", "Instant", "now"),
        format!("{}{}", "System", "Time"),
        format!("{}::{}", "std", "env"),
        format!("{}::{}", "std", "fs"),
        format!("{} {}", "static", "mut"),
        format!("{}{}", "Once", "Lock"),
        format!("{}{}", "Mu", "tex"),
        format!("{}{}", "Atomic", "U"),
    ];
    assert_absent(
        &policy_sources(),
        &needles,
        "plan / checkpoint / project / hash are PURE: the same input must give the same plan bit \
         for bit (replay), so no clock, environment, filesystem or global may enter",
    );
    // Positive control: the bus (not a policy file) does read a clock and
    // does hold a global, and the scan sees both.
    let bus = code_only(&read(&crate_root().join("src/router/bus.rs")));
    assert!(bus.contains("elapsed()") && bus.contains(needles[6].as_str()));
}

#[test]
fn router_sources_name_no_category_or_path() {
    let needles = [
        format!("{}{}", "cate", "gory"),
        format!("{}{}", "pre", "set"),
        format!("{}{}", "bench", "mark"),
        format!("{}{}", "file", "name"),
        format!("{}_{}", "file", "name"),
        format!("{}{}", "on", "nx"),
        format!("{}{}", "vnn", "lib"),
        format!("{}_{}", "instance", "name"),
    ];
    let files = router_sources();
    for file in &files {
        let code = code_only(&read(file)).to_ascii_lowercase();
        for needle in &needles {
            assert!(
                !code.contains(needle.as_str()),
                "{} names `{needle}` (case-insensitive): the policy is name-blind by \
                 construction — facts come from the model and the property, never from what \
                 the harness calls them (ROUTER_PROGRAM §4.1)",
                file.display()
            );
        }
    }
    // Positive control: the same lowercase code scan finds the words in the
    // scored entry point, which does route by the harness's arguments.
    let entry = code_only(&read(
        &crate_root().join("../ny-cli/src/commands/vnncomp.rs"),
    ))
    .to_ascii_lowercase();
    for needle in [&needles[0], &needles[5], &needles[6]] {
        assert!(
            entry.contains(needle.as_str()),
            "positive control: `{needle}` must be found in ny-cli's vnncomp.rs"
        );
    }
}

#[test]
fn ny_core_dependencies_exclude_verdict_crates() {
    let manifest = read(&crate_root().join("Cargo.toml"));
    let mut in_deps = false;
    let mut deps: Vec<String> = Vec::new();
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_deps = line == "[dependencies]";
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let key = line
            .split(['=', '.'])
            .next()
            .map(str::trim)
            .unwrap_or_default();
        if !key.is_empty() {
            deps.push(key.to_owned());
        }
    }
    deps.sort();
    assert_eq!(
        deps,
        ["half", "serde", "sha2", "thiserror", "trust"],
        "ny-core's dependencies decide what the router CAN name: a verdict crate here would make \
         `UnsatToken` / `VnncompResult` reachable from the policy (ROUTER_PROGRAM §6.1)"
    );
}
