// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! THE RATCHET — two direct-literal `NY_*` read forms may only ever go down.
//!
//! Phase 0b of `docs/LEVER_DEBT_EXECUTION_PLAN_2026-08-11.md`. The workspace
//! has ~850 ad-hoc `std::env::var` reads of `NY_*` names. Migrating them is a
//! multi-phase job; what this test does is freeze the net occurrence count of
//! the two exact substrings below while that job is in flight. It is the same idiom as
//! `measured_gate_delivery.rs`, which already proved it works in this repo.
//!
//! Std-lib only, hand-rolled directory walk: this crate is a leaf and does not
//! take a `walkdir` dependency for one recursive read_dir.
//!
//! # Known blind spot (deliberate, documented, not worth fixing here)
//!
//! The scan is a literal substring match, so it MISSES:
//!
//! * names built by a macro or by `format!`/`concat!`;
//! * calls whose name argument is a constant or alias
//!   (`env::var(CPU_SOUND_F64_ENGINE_ENV)`) rather than a literal;
//! * calls split across lines by rustfmt, e.g.
//!   `std::env::var(\n    "NY_SOMETHING_WITH_A_VERY_LONG_NAME",\n)`;
//! * reads that go through a helper wrapper instead of `std::env` directly;
//! * `env!`/`option_env!` compile-time reads.
//!
//! A miss makes the count too LOW, and removing one counted occurrence while
//! adding another can leave the net count unchanged. This gate is therefore a
//! cheap, review-visible drift tripwire, not proof that raw reads cannot grow.
//! The complete callsite/name inventories that cover aliases and prevent
//! same-count replacement are later migration work in the design document.
//!
//! Note also that `ny-levers`' own choke point is invisible here by
//! construction: it calls `std::env::var(decl.name)` with a runtime name. That
//! is the shape every migrated read is supposed to end up as.

use std::path::{Path, PathBuf};

/// The checked-in baseline. Lowered by every migration, in the same commit.
const BASELINE_FILE: &str = include_str!("../src/ratchet_baseline.txt");

/// Directory fragments that are not first-party source. `/target/` is build
/// output, `/.lake/` is a vendored Lean toolchain tree, `/proofs/lean/` is
/// vendored proof source.
const SKIP_FRAGMENTS: &[&str] = &["/target/", "/.lake/", "/proofs/lean/"];

/// Built at runtime from fragments so that this file does not itself contain
/// the substrings it searches for — otherwise the ratchet would count its own
/// needles and the baseline would drift with the test.
fn needles() -> [String; 2] {
    let prefix = concat!("env", "::var");
    [format!("{prefix}(\"NY_"), format!("{prefix}_os(\"NY_")]
}

fn workspace_root() -> PathBuf {
    // crates/ny-levers -> crates -> workspace root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root must resolve")
}

fn parse_baseline(raw: &str) -> usize {
    raw.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .and_then(|l| l.parse::<usize>().ok())
        .expect("ratchet_baseline.txt must contain a bare integer line")
}

/// Recursive `*.rs` collector. Skips symlinks so a stray link cannot loop.
fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_symlink() {
            continue;
        }
        // Compare with forward slashes; the fragments are written that way and
        // this repo is developed on unix hosts.
        let as_str = path.to_string_lossy().replace('\\', "/");
        if kind.is_dir() {
            let probe = format!("{as_str}/");
            if SKIP_FRAGMENTS.iter().any(|f| probe.contains(f)) {
                continue;
            }
            collect_rs(&path, out);
        } else if kind.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
        {
            if SKIP_FRAGMENTS.iter().any(|f| as_str.contains(f)) {
                continue;
            }
            out.push(path);
        }
    }
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

struct Scan {
    files: usize,
    var: usize,
    var_os: usize,
    worst: Vec<(String, usize)>,
}

fn scan() -> Scan {
    let root = workspace_root();
    let crates = root.join("crates");
    assert!(crates.is_dir(), "expected {} to exist", crates.display());

    let [n_var, n_var_os] = needles();
    let mut files = Vec::new();
    collect_rs(&crates, &mut files);
    files.sort();

    let mut scan = Scan {
        files: files.len(),
        var: 0,
        var_os: 0,
        worst: Vec::new(),
    };
    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue; // non-UTF-8 .rs is not a lever read
        };
        let a = count(&text, &n_var);
        let b = count(&text, &n_var_os);
        scan.var += a;
        scan.var_os += b;
        if a + b > 0 {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();
            scan.worst.push((rel, a + b));
        }
    }
    scan.worst.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    scan
}

#[test]
fn raw_ny_env_reads_match_the_shrinking_baseline_exactly() {
    let baseline = parse_baseline(BASELINE_FILE);
    let s = scan();
    let total = s.var + s.var_os;

    println!(
        "ratchet: {total} raw NY_* reads ({} var + {} var_os) in {} .rs files under crates/ \
         (baseline {baseline})",
        s.var, s.var_os, s.files
    );

    assert!(
        total == baseline,
        "RATCHET TRIPPED: {total} raw NY_* environment reads, baseline is {baseline} \
         (signed delta {:+}).\n\
         \n\
         If the delta is positive, do NOT raise the baseline. A new \
         `env::var(\"NY_...\")` is exactly the debt this gate exists to stop: it is \
         unenumerable, its parser disagrees with the other ~850, and it will not \
         appear in the flight receipt.\n\
         \n\
         Instead, declare the lever in crates/ny-levers/src/decls/ (see \
         decls/root_alpha.rs for the shape) and read it from the frozen LeverSet. \
         Declaring it costs you a `doc` and a `Provenance`, which is the point.\n\
         \n\
         If the delta is negative, lower ratchet_baseline.txt to {total} in the SAME \
         commit. Slack would let a deleted raw read return without tripping the gate.\n\
         \n\
         Heaviest files in this scan:\n{}",
        total as i128 - baseline as i128,
        s.worst
            .iter()
            .take(10)
            .map(|(f, n)| format!("  {n:>4}  {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_scan_actually_scans_something() {
    // Guards the failure mode where a bad path or an over-eager skip rule
    // makes the ratchet vacuously green forever.
    let s = scan();
    assert!(
        s.files > 500,
        "expected to walk the whole crates/ tree, saw only {} .rs files",
        s.files
    );
    assert!(
        s.var + s.var_os > 100,
        "the workspace is known to have hundreds of raw NY_* reads; finding {} means \
         the scan is broken, not that the debt is gone",
        s.var + s.var_os
    );
}

#[test]
fn skip_fragments_exclude_vendored_and_build_trees() {
    let s = scan();
    for (file, _) in &s.worst {
        for fragment in SKIP_FRAGMENTS {
            let probe = format!("/{file}");
            assert!(
                !probe.contains(fragment),
                "{file} should have been skipped by {fragment}"
            );
        }
    }
}
