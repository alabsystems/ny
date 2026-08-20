// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Full-coverage validation of the dual-network formula DNF extractor against
//! the REAL VNN-COMP 2026 relational ACAS benchmark files (100 instances).
//! The explicit `external-vnncomp` conformance lane requires the benchmark
//! checkout. When selected, EVERY instance must extract completely
//! (`formula_dnf: Some`); absence or parser refusal is a hard failure.

use std::path::{Path, PathBuf};

use crate::vnnlib::load_vnnlib;

/// Require the sparse-cloned benchmark root in the repo checkout.
///
/// Both spellings are accepted: upstream's repo is `vnncomp2026_benchmarks`, so
/// a plain clone lands there, while the sibling corpora in this tree sit under
/// their bare year. Hardcoding one made this fixture "missing" against a corpus
/// that was present under the other name.
fn benchmark_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("../../benchmarks/vnncomp2026_benchmarks/benchmarks"),
        manifest.join("../../benchmarks/vnncomp2026/benchmarks"),
    ];
    let root = candidates
        .iter()
        .find(|candidate| candidate.is_dir())
        .unwrap_or(&candidates[0])
        .clone();
    assert!(
        root.is_dir(),
        "VNN-COMP 2026 benchmark fixture root is missing at {}; \
         run benchmarks/download_benchmarks.sh",
        root.display()
    );
    root
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
#[cfg(feature = "external-vnncomp")]
fn extractor_fully_covers_the_real_2026_relational_benchmarks() {
    let root = benchmark_root();
    let mut total = 0usize;
    let mut covered = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for category in ["isomorphic_acasxu_2026", "monotonic_acasxu_2026"] {
        let files = category_vnnlib_files(&root, category);
        assert!(
            !files.is_empty(),
            "VNN-COMP 2026 fixture has no VNN-LIB files under {category}; \
             run benchmarks/download_benchmarks.sh"
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

/// Pin extractor/parser agreement on the first real instance of each category.
#[test]
#[cfg(feature = "external-vnncomp")]
fn dual_parser_and_extractor_agree_on_real_instance_zero() {
    let root = benchmark_root();
    let mut checked = 0usize;
    for category in ["isomorphic_acasxu_2026", "monotonic_acasxu_2026"] {
        let files = category_vnnlib_files(&root, category);
        let file = files.first().unwrap_or_else(|| {
            panic!(
                "VNN-COMP 2026 fixture has no VNN-LIB files under {category}; \
                 run benchmarks/download_benchmarks.sh"
            )
        });
        let content = std::fs::read_to_string(file)
            .unwrap_or_else(|error| panic!("{}: read failed: {error}", file.display()));
        let cleaned = crate::vnnlib::syntax::strip_vnnlib_comments(&content);
        let tokens = crate::vnnlib::syntax::tokenize(&cleaned)
            .unwrap_or_else(|error| panic!("{}: tokenization failed: {error}", file.display()));
        let exprs = crate::vnnlib::syntax::parse_expressions(&tokens)
            .unwrap_or_else(|error| panic!("{}: expression parse failed: {error}", file.display()));
        let extracted = crate::vnnlib::dual_formula::extract_dual_formula_dnf(&exprs)
            .unwrap_or_else(|| panic!("{}: formula extractor declined", file.display()));
        let parsed = crate::vnnlib::parser::parse_dual_network_spec(&exprs)
            .unwrap_or_else(|error| panic!("{}: dual parser failed: {error}", file.display()))
            .unwrap_or_else(|| panic!("{}: dual parser declined", file.display()));
        let parsed_dnf = parsed
            .formula_dnf
            .as_ref()
            .unwrap_or_else(|| panic!("{}: parsed dual spec omitted its DNF", file.display()));
        assert_eq!(
            (parsed_dnf.clauses.len(), parsed_dnf.num_asserts),
            (extracted.clauses.len(), extracted.num_asserts),
            "{}: parser and direct extractor disagree",
            file.display()
        );
        assert!(
            !extracted.clauses.is_empty() && extracted.num_asserts > 0,
            "{}: extracted formula is empty",
            file.display()
        );
        checked += 1;
    }
    assert_eq!(checked, 2, "both relational categories must be checked");
}
