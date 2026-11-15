// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! THE B2 DOWNGRADE-ONLY RATCHET (ENGINE_PROGRAM §1.7 B2; B2.1 brief §9).
//!
//! `ny proof-check` may never become a verdict source. Mechanically:
//!
//! 1. across every `crates/*/Cargo.toml` (all dependency sections) only
//!    `ny-cli` depends on `ny-proofcheck`;
//! 2. in `ny-cli/src`, only `commands/proof_check.rs` names `ny_proofcheck`,
//!    and only the dispatch (`main.rs`) names `proof_check::`;
//! 3. no production source outside `crates/ny-proofcheck/` and
//!    `commands/proof_check.rs` spells the report schema or the report file
//!    suffix (nothing else can read a report);
//! 4. hint capture is switched on (`enable_hint_capture(`) only by the Proof
//!    arm in `commands/proof_bundle.rs`.
//!
//! Each check scans code (comment lines stripped), reports every violation,
//! and carries a positive control. Needles are assembled at runtime.

use std::path::{Path, PathBuf};

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e} (the ratchet must see it)",
            path.display()
        )
    })
}

fn code_only(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Every `(section, dependency name)` of a manifest.
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

fn crate_dirs() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(crates_dir())
        .expect("crates dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("Cargo.toml").is_file())
        .collect();
    v.sort();
    assert!(v.len() > 10, "expected the workspace crates, found {v:?}");
    v
}

#[test]
fn proof_check_ratchet_no_verdict_path_reads_checker_output() {
    let checker = format!("ny-{}", "proofcheck");
    let checker_ident = format!("ny_{}", "proofcheck");
    let mut violations: Vec<String> = Vec::new();

    // 1. Only ny-cli depends on the checker.
    let mut dependents = Vec::new();
    for dir in crate_dirs() {
        let deps = dependencies(&read(&dir.join("Cargo.toml")));
        if deps.iter().any(|(_, n)| *n == checker) {
            dependents.push(
                dir.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            );
        }
    }
    assert!(
        dependents.contains(&"ny-cli".to_string()),
        "positive control: ny-cli must depend on the checker: {dependents:?}"
    );
    for d in &dependents {
        if d != "ny-cli" {
            violations.push(format!("crate `{d}` depends on {checker}"));
        }
    }

    // 2. Inside ny-cli/src: who names the crate and the command module.
    let cli_src = crates_dir().join("ny-cli/src");
    let mut files = Vec::new();
    rust_sources(&cli_src, &mut files);
    files.sort();
    let module_path = format!("{}_check::", "proof");
    let allowed_ident = cli_src.join("commands/proof_check.rs");
    let allowed_module = cli_src.join("main.rs");
    let mut ident_seen = false;
    let mut module_seen = false;
    for f in &files {
        let code = code_only(&read(f));
        if code.contains(&checker_ident) {
            if *f == allowed_ident {
                ident_seen = true;
            } else {
                violations.push(format!("{} names `{checker_ident}`", f.display()));
            }
        }
        if code.contains(&module_path) {
            if *f == allowed_module {
                module_seen = true;
            } else {
                violations.push(format!("{} names `{module_path}`", f.display()));
            }
        }
    }
    assert!(
        ident_seen,
        "positive control: commands/proof_check.rs must name the checker"
    );
    assert!(
        module_seen,
        "positive control: main.rs must dispatch the command"
    );

    // 3. No production source outside the checker reads a report.
    let schema = format!("ny-proof-check-{}/", "report");
    let suffix = format!(".check.{}", "json");
    let mut schema_seen = false;
    for dir in crate_dirs() {
        let mut srcs = Vec::new();
        rust_sources(&dir.join("src"), &mut srcs);
        for f in srcs {
            let code = code_only(&read(&f));
            let inside_checker = f.starts_with(crates_dir().join(&checker));
            if inside_checker {
                schema_seen |= code.contains(&schema);
                continue;
            }
            if f == allowed_ident {
                continue;
            }
            for needle in [&schema, &suffix] {
                if code.contains(needle.as_str()) {
                    violations.push(format!("{} contains `{needle}`", f.display()));
                }
            }
        }
    }
    assert!(
        schema_seen,
        "positive control: the checker must spell its report schema"
    );

    // 4. Hint capture is switched on only by the Proof arm.
    let call = format!("enable_hint_{}(", "capture");
    let def = format!("fn {call}");
    let allowed_call = cli_src.join("commands/proof_bundle.rs");
    let mut call_seen = false;
    for dir in crate_dirs() {
        let mut srcs = Vec::new();
        rust_sources(&dir.join("src"), &mut srcs);
        for f in srcs {
            let code = code_only(&read(&f));
            let calls = code.matches(call.as_str()).count() - code.matches(def.as_str()).count();
            if calls == 0 {
                continue;
            }
            if f == allowed_call {
                call_seen = true;
            } else {
                violations.push(format!("{} calls `{call}`", f.display()));
            }
        }
    }
    assert!(
        call_seen,
        "positive control: proof_bundle.rs must switch capture on"
    );

    assert!(
        violations.is_empty(),
        "the checker could become a verdict source: {violations:#?}"
    );
}
