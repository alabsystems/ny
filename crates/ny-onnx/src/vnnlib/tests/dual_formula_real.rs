// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Full-coverage validation of the dual-network formula DNF extractor against
//! the REAL VNN-COMP 2026 relational ACAS benchmark files (100 instances).
//! Skips gracefully when the benchmark checkout is absent; when present,
//! EVERY instance must either extract completely (`formula_dnf: Some`) or be
//! recognizably fail-closed — and for this benchmark we assert full coverage,
//! since the gate flip depends on it.

use std::path::{Path, PathBuf};

use crate::vnnlib::load_vnnlib;

/// Locate the sparse-cloned benchmark root in the repo checkout, or `None`
/// to skip.
fn benchmark_root() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [manifest.join("../../benchmarks/vnncomp2026_benchmarks/benchmarks")];
    candidates.into_iter().find(|c| c.is_dir())
}

fn category_vnnlib_files(root: &Path, category: &str) -> Vec<PathBuf> {
    let dir = root.join(category).join("2.0/vnnlib");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "vnnlib"))
        .collect();
    files.sort();
    files
}

#[test]
fn extractor_fully_covers_the_real_2026_relational_benchmarks() {
    let Some(root) = benchmark_root() else {
        eprintln!("2026 relational benchmarks not present; skipping");
        return;
    };
    let mut total = 0usize;
    let mut covered = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for category in ["isomorphic_acasxu_2026", "monotonic_acasxu_2026"] {
        let files = category_vnnlib_files(&root, category);
        assert!(
            !files.is_empty(),
            "benchmark dir present but no vnnlib files under {category}"
        );
        for file in files {
            total += 1;
            let spec = match load_vnnlib(&file) {
                Ok(spec) => spec,
                Err(e) => {
                    failures.push(format!("{}: parse error {e}", file.display()));
                    continue;
                }
            };
            let Some(dual) = spec.dual_network.as_ref() else {
                failures.push(format!("{}: no dual-network spec", file.display()));
                continue;
            };
            match dual.formula_dnf.as_ref() {
                Some(dnf) => {
                    // Structural sanity: a real instance always asserts a box,
                    // couplings, and output atoms — never an empty formula.
                    assert!(
                        !dnf.clauses.is_empty() && dnf.num_asserts > 0,
                        "{}: extracted an empty DNF",
                        file.display()
                    );
                    assert!(
                        dnf.clauses.iter().all(|c| !c.is_empty()),
                        "{}: extracted an empty clause",
                        file.display()
                    );
                    covered += 1;
                }
                None => {
                    failures.push(format!("{}: extraction failed closed", file.display()));
                }
            }
        }
    }
    println!("dual-formula extractor coverage: {covered}/{total} real 2026 relational instances");
    assert!(
        failures.is_empty(),
        "extractor did not fully cover {} of {total} instances:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(covered, total);
}

/// Diagnostic (temporary-grade but harmless to keep): what does the dual
/// parser + extractor produce on the first real instance of each category?
#[test]
fn dual_parse_diagnostic_instance0() {
    let Some(root) = benchmark_root() else {
        return;
    };
    for category in ["isomorphic_acasxu_2026", "monotonic_acasxu_2026"] {
        let files = category_vnnlib_files(&root, category);
        let Some(file) = files.first() else { continue };
        let content = std::fs::read_to_string(file).unwrap();
        let cleaned = crate::vnnlib::syntax::strip_vnnlib_comments(&content);
        let tokens = crate::vnnlib::syntax::tokenize(&cleaned).unwrap();
        let exprs = crate::vnnlib::syntax::parse_expressions(&tokens).unwrap();
        let dnf = crate::vnnlib::dual_formula::extract_dual_formula_dnf(&exprs);
        println!(
            "{category}: extractor -> {:?}",
            dnf.as_ref().map(|d| (d.clauses.len(), d.num_asserts))
        );
        match crate::vnnlib::parser::parse_dual_network_spec(&exprs) {
            Ok(Some(d)) => println!("{category}: dual spec OK, property {:?}", d.property),
            Ok(None) => println!("{category}: dual spec -> None"),
            Err(e) => println!("{category}: dual spec ERR: {e}"),
        }
    }
}
