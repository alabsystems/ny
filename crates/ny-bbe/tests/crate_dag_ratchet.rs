// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! THE CRATE-DAG RATCHET (ENGINE_PROGRAM §1.3, rule R2, JE C9).
//!
//! The search/certify split is only as strong as the dependency graph that
//! enforces it:
//!
//! * `ny-bbe` (UNTRUSTED search) depends on `ny-core` and NOTHING else — so it
//!   cannot name `ny-propagate`, and therefore cannot name or construct any
//!   certified type;
//! * `ny-propagate` (the certified evaluators) never depends on `ny-bbe` or on
//!   any CUDA crate — it sees search backends only as
//!   `&dyn ny_core::search::SearchBackend`, injected by `ny-cli`;
//! * `ny-core` depends on neither.
//!
//! Every check reads the manifests and sources from disk, so an edit to a
//! `Cargo.toml` fails here before anything is built on it. Each check is its
//! own `#[test]` and reports every violation it finds.

use std::collections::BTreeSet;
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

/// Every `(section, dependency name)` in a manifest's dependency tables:
/// `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]` and their
/// `[target.'cfg(..)'.*]` forms. A deliberately small TOML reader: section
/// headers, `name = ...` / `name.workspace = true` keys, comments skipped.
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

/// Every `.rs` file under `dir`, recursively.
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

/// Source text with `//` comments removed (doc comments may NAME the other
/// side of the boundary; code may not).
fn code_only(text: &str) -> String {
    text.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn ny_bbe_depends_on_ny_core_only() {
    let manifest = read(&crates_dir().join("ny-bbe/Cargo.toml"));
    let deps = dependencies(&manifest);
    let names: BTreeSet<&str> = deps.iter().map(|(_, n)| n.as_str()).collect();
    assert!(
        deps.iter()
            .any(|(s, n)| s == "dependencies" && n == "ny-core"),
        "the reader found no ny-core dependency — the parser is VOID: {deps:?}"
    );
    let violations: Vec<&(String, String)> = deps.iter().filter(|(_, n)| n != "ny-core").collect();
    assert!(
        violations.is_empty(),
        "ny-bbe may depend on ny-core ONLY (R2: untrusted search must not be able to name a \
         certified type or pull a device runtime into the verifier). Found: {violations:?} \
         (all deps: {names:?})"
    );
}

#[test]
fn ny_propagate_never_depends_on_search_or_cuda() {
    let manifest = read(&crates_dir().join("ny-propagate/Cargo.toml"));
    let deps = dependencies(&manifest);
    assert!(
        deps.iter().any(|(_, n)| n == "ny-core"),
        "the reader found no ny-core dependency in ny-propagate — the parser is VOID: {deps:?}"
    );
    let forbidden = ["ny-bbe", "ny-cuda", "cudarc"];
    let violations: Vec<&(String, String)> = deps
        .iter()
        .filter(|(_, n)| forbidden.contains(&n.as_str()))
        .collect();
    assert!(
        violations.is_empty(),
        "ny-propagate must stay CUDA-free and must never depend on ny-bbe (it sees backends as \
         `&dyn SearchBackend`, injected by ny-cli). Found: {violations:?}"
    );
}

#[test]
fn ny_core_depends_on_neither_side() {
    let manifest = read(&crates_dir().join("ny-core/Cargo.toml"));
    let deps = dependencies(&manifest);
    let violations: Vec<&(String, String)> = deps
        .iter()
        .filter(|(_, n)| n.starts_with("ny-") && n != "ny-test-utils")
        .collect();
    assert!(
        violations.is_empty(),
        "ny-core is the shared root: it may not depend on another ny crate. Found: \
         {violations:?}"
    );
}

#[test]
fn ny_core_carries_no_device_runtime() {
    let manifest = read(&crates_dir().join("ny-core/Cargo.toml"));
    let deps = dependencies(&manifest);
    assert!(
        !deps.is_empty(),
        "the reader found no ny-core dependency at all — the parser is VOID"
    );
    let violations: Vec<&(String, String)> = deps
        .iter()
        .filter(|(_, n)| n == "cudarc" || n.starts_with("cuda"))
        .collect();
    assert!(
        violations.is_empty(),
        "ny-core is the shared root of search and certify: it may not pull a device runtime \
         (M4: backends are injected as `&dyn SearchBackend`). Found: {violations:?}"
    );
}

/// M4 (#wide-fd-search): the DEVICE search backend lives in ny-cuda. The
/// device never certifies, so ny-cuda may not depend on the certifier
/// (ny-propagate) or on the CPU search crate, and its search module may not
/// name the certified side.
#[test]
fn ny_cuda_never_depends_on_the_certifier() {
    let manifest = read(&crates_dir().join("ny-cuda/Cargo.toml"));
    let deps = dependencies(&manifest);
    assert!(
        deps.iter().any(|(_, n)| n == "ny-core"),
        "the reader found no ny-core dependency in ny-cuda — the parser is VOID: {deps:?}"
    );
    let violations: Vec<&(String, String)> = deps
        .iter()
        .filter(|(_, n)| n == "ny-propagate" || n == "ny-bbe")
        .collect();
    assert!(
        violations.is_empty(),
        "ny-cuda (the device search backend) may not depend on the certifier or the CPU \
         search: {violations:?}"
    );
    let search = crates_dir().join("ny-cuda/src/search.rs");
    let code = code_only(&read(&search));
    assert!(
        code.contains("impl SearchBackend for CudaSearchBackend"),
        "the scan did not find the device backend in {} — VOID",
        search.display()
    );
    for needle in ["ny_propagate", "Certified"] {
        assert!(
            !code.contains(needle),
            "{} names `{needle}`: the device search may not name the certified side",
            search.display()
        );
    }
    // The device search's submodules (context, kernels, planner, tests) are
    // held to the same line, and may not reach the CPU search either.
    let mut files = Vec::new();
    rust_sources(&crates_dir().join("ny-cuda/src/search"), &mut files);
    assert!(
        files.iter().any(|f| f.ends_with("device.rs"))
            && files.iter().any(|f| f.ends_with("plan.rs")),
        "the walk found no ny-cuda/src/search/{{device,plan}}.rs — the scan is VOID"
    );
    let mut violations = Vec::new();
    for file in &files {
        let code = code_only(&read(file));
        for needle in ["ny_propagate", "Certified", "ny_bbe"] {
            if code.contains(needle) {
                violations.push(format!("{} names `{needle}`", file.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "the device search may not name the certified side or the CPU search: {violations:?}"
    );
}

#[test]
fn search_sources_never_name_the_certified_side() {
    let mut files = Vec::new();
    rust_sources(&crates_dir().join("ny-bbe/src"), &mut files);
    assert!(
        files.iter().any(|f| f.ends_with("joint_root.rs"))
            && files.iter().any(|f| f.ends_with("fd.rs"))
            && files.iter().any(|f| f.ends_with("wave.rs")),
        "the source walk found no joint_root.rs / fd.rs / wave.rs — the scan is VOID"
    );
    let mut violations = Vec::new();
    for file in &files {
        let code = code_only(&read(file));
        for needle in ["ny_propagate", "Certified", "unsafe "] {
            if code.contains(needle) {
                violations.push(format!("{} names `{needle}`", file.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "untrusted search code may not name the certified side or use unsafe: {violations:?}"
    );
}

#[test]
fn certified_sources_never_name_the_search_crate() {
    let mut files = Vec::new();
    rust_sources(&crates_dir().join("ny-propagate/src"), &mut files);
    assert!(
        files.len() > 100,
        "the source walk found {} ny-propagate files — the scan is VOID",
        files.len()
    );
    let violations: Vec<String> = files
        .iter()
        .filter(|f| code_only(&read(f)).contains("ny_bbe"))
        .map(|f| f.display().to_string())
        .collect();
    assert!(
        violations.is_empty(),
        "ny-propagate may reach search only through `ny_core::search::SearchBackend`: \
         {violations:?}"
    );
}
