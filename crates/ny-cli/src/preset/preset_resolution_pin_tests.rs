// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Preset-resolution shadowing pins for the shipped `configs/` tree.
//!
//! `resolve_preset_path` (commands/vnncomp.rs) searches `configs/vnncomp*/`
//! NEWEST year directory first (descending directory-name sort), trying the
//! full category name before the year-stripped base name; the first existing
//! file wins. Its generic ordering is pinned by unit tests next to the
//! resolver; the tests HERE pin the consequence of that ordering on the real
//! shipped tree for categories present in more than one year directory. The
//! 2025 counterfactual scores are measured through this resolution, so a
//! silent flip - adding, removing, or renaming one of these YAMLs, or an
//! ordering change - must fail a test here instead of silently changing which
//! preset the scored run loads.

use super::load_preset;
use std::fs;
use std::path::{Path, PathBuf};

fn configs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs")
}

/// Faithful mirror of the resolver's search order over a real configs tree:
/// `vnncomp*` directories in DESCENDING name order (newest year first), full
/// category name before the year-stripped base name, first existing file
/// wins. Kept in sync with `resolve_preset_path`, whose own unit tests pin
/// the ordering generically on synthetic trees.
fn resolve_newest_first(configs_dir: &Path, category: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(configs_dir)
        .expect("configs dir must be readable")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("vnncomp"))
        })
        .collect();
    dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    let lower = category.to_ascii_lowercase();
    let mut candidates = vec![lower.clone()];
    if lower.len() > 5 {
        let (head, tail) = lower.split_at(lower.len() - 5);
        let tb = tail.as_bytes();
        if tb[0] == b'_'
            && tb[1] == b'2'
            && tb[2] == b'0'
            && tb[3].is_ascii_digit()
            && tb[4].is_ascii_digit()
        {
            candidates.push(head.to_string());
        }
    }
    for dir in dirs {
        for candidate in &candidates {
            let preset = dir.join(format!("{candidate}.yaml"));
            if preset.is_file() {
                return Some(preset);
            }
        }
    }
    None
}

fn canon(path: &Path) -> PathBuf {
    path.canonicalize()
        .unwrap_or_else(|e| panic!("canonicalize {}: {e}", path.display()))
}

/// safenlp_2024 exists in BOTH vnncomp25 and vnncomp26; the newest-first
/// resolver loads the vnncomp26 file, SHADOWING the vnncomp25 one. Any edit
/// meant to change safenlp behavior for the 2025 counterfactual must land in
/// (or re-point resolution at) the file pinned here - editing the vnncomp25
/// YAML alone is dead for the scored path.
#[test]
fn safenlp_2024_resolves_to_vnncomp26_shadowing_vnncomp25() {
    let configs = configs_dir();
    let shadowed = configs.join("vnncomp25/safenlp_2024.yaml");
    let winner = configs.join("vnncomp26/safenlp_2024.yaml");
    assert!(
        shadowed.is_file(),
        "expected the SHADOWED vnncomp25 safenlp_2024.yaml to exist; if it was \
         removed or renamed, re-pin this test to the new tree"
    );
    assert!(winner.is_file(), "vnncomp26 safenlp_2024.yaml must exist");

    let resolved =
        resolve_newest_first(&configs, "safenlp_2024").expect("safenlp_2024 preset must resolve");
    assert_eq!(
        canon(&resolved),
        canon(&winner),
        "newest-first resolution must load the vnncomp26 safenlp_2024 preset \
         (the vnncomp25 file is shadowed); a flip here silently changes 2025 \
         counterfactual scores"
    );

    // Both files must stay parseable: the shadowed one is still loadable by
    // explicit path (e.g. --config), and the winner is the scored default.
    load_preset(&winner).expect("winning vnncomp26 preset must parse");
    load_preset(&shadowed).expect("shadowed vnncomp25 preset must parse");
}

/// collins_aerospace_benchmark exists ONLY in vnncomp26; resolution loads it
/// from there. The vnncomp25 non-existence is pinned deliberately: adding a
/// vnncomp25 YAML with this name would be dead for resolution (vnncomp26
/// still wins) - this test failing is the signal to the author.
#[test]
fn collins_aerospace_benchmark_resolves_to_vnncomp26() {
    let configs = configs_dir();
    let winner = configs.join("vnncomp26/collins_aerospace_benchmark.yaml");
    assert!(
        winner.is_file(),
        "vnncomp26 collins_aerospace_benchmark.yaml must exist"
    );
    assert!(
        !configs
            .join("vnncomp25/collins_aerospace_benchmark.yaml")
            .exists(),
        "a vnncomp25 collins_aerospace_benchmark.yaml appeared: it is SHADOWED \
         by the vnncomp26 file under newest-first resolution and will never \
         load on the scored path; re-pin this test only if that is intended"
    );

    let resolved = resolve_newest_first(&configs, "collins_aerospace_benchmark")
        .expect("collins_aerospace_benchmark preset must resolve");
    assert_eq!(canon(&resolved), canon(&winner));

    load_preset(&winner).expect("winning vnncomp26 preset must parse");
}
