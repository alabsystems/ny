// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! THE INDEPENDENCE RATCHET (ENGINE_PROGRAM §1.7 B2; brief §4, §9).
//!
//! A B2 checker that calls the producer's certification code for the value it
//! checks only re-runs the producer's bug. So `ny-proofcheck`:
//!
//! * depends on EXACTLY `ny-cert` (its total JSON reader/writer), `sha2` and
//!   `rayon` — never on `ny-propagate`, `ny-core`, `ny-onnx`, `prost` or
//!   `serde`, and has no dev-dependency either;
//! * names none of those crates in its sources;
//! * reads no environment variable, and never uses fused multiply-add;
//! * forbids `unsafe`.
//!
//! Each check reads the files from disk, reports every violation, and has a
//! positive control proving the scanner can see what it forbids. Needles are
//! assembled at runtime so this file cannot match itself.

use std::path::{Path, PathBuf};

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e} (the ratchet must see it)",
            path.display()
        )
    });
    assert!(
        text.len() > 200,
        "{} is suspiciously short: an empty module compiles and scans clean",
        path.display()
    );
    text
}

/// `(section, dependency name)` for every dependency table entry.
fn dependencies(manifest: &str) -> Vec<(String, String)> {
    let mut section = String::new();
    let mut out = Vec::new();
    for raw in manifest.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            section = line.trim_matches(|c| c == '[' || c == ']').to_string();
            continue;
        }
        if !section.ends_with("dependencies") {
            continue;
        }
        let Some((key, _)) = line.split_once('=') else {
            continue;
        };
        let name = key.trim().split('.').next().unwrap_or("").trim();
        if !name.is_empty() {
            out.push((section.clone(), name.trim_matches('"').to_string()));
        }
    }
    out
}

fn sources() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let dir = crate_dir().join("src");
    for e in std::fs::read_dir(&dir).expect("src dir") {
        let p = e.expect("entry").path();
        if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    out.sort();
    assert!(
        out.len() >= 10,
        "expected the checker's modules, found {out:?}"
    );
    out
}

fn code_only(text: &str) -> String {
    text.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn proof_check_ratchet_checker_depends_on_no_producer_crate() {
    // Dependencies: an exact allowlist, and no other table at all.
    let manifest = read(&crate_dir().join("Cargo.toml"));
    let deps = dependencies(&manifest);
    let names: Vec<&str> = deps.iter().map(|(_, n)| n.as_str()).collect();
    assert!(
        names.contains(&"ny-cert") && names.contains(&"sha2") && names.contains(&"rayon"),
        "positive control: the manifest reader must see the three allowed dependencies: {deps:?}"
    );
    let allowed = ["ny-cert", "sha2", "rayon"];
    let violations: Vec<&(String, String)> = deps
        .iter()
        .filter(|(s, n)| s != "dependencies" || !allowed.contains(&n.as_str()))
        .collect();
    assert!(
        violations.is_empty(),
        "ny-proofcheck may depend on {allowed:?} only, and carries no dev/build dependency: \
         {violations:?}"
    );
    // A forbidden dependency is visible to the reader (positive control).
    let fake = "[dependencies]\nny-propagate = { path = \"../ny-propagate\" }\n";
    assert_eq!(
        dependencies(fake),
        vec![("dependencies".to_string(), format!("ny-{}", "propagate"))]
    );

    // Sources: no producer crate, no env read, no FMA, forbid(unsafe).
    let forbidden = [
        format!("ny_{}", "propagate"),
        format!("ny_{}", "core"),
        format!("ny_{}", "onnx"),
        format!("{}::", "prost"),
        format!("{}::", "serde"),
        format!("{}_json", "serde"),
        format!("std::env::{}", "var"),
        format!("env::{}(", "var"),
        format!("{}{}", "mul_", "add"),
        format!(".{}(", "fma"),
    ];
    let mut hits = Vec::new();
    for f in sources() {
        let code = code_only(&read(&f));
        for needle in &forbidden {
            if code.contains(needle.as_str()) {
                hits.push(format!("{} contains `{needle}`", f.display()));
            }
        }
    }
    assert!(hits.is_empty(), "independence violations: {hits:#?}");
    // Positive control: the scanner sees a needle in code.
    let probe = format!("let x = a.{}(b, c);", "mul_add");
    assert!(code_only(&probe).contains(forbidden[8].as_str()));
    let lib = read(&crate_dir().join("src/lib.rs"));
    assert!(
        lib.contains(&format!("#![forbid({}_code)]", "unsafe")),
        "src/lib.rs must forbid unsafe code"
    );
}
