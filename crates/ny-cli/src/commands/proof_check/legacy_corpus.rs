// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! review(legacy-bundle): the writer's fallback `clauses` against the
//! CHECKER's own reader on the REAL corpus (external-vnncomp lane only: it
//! reads the gitignored `benchmarks/`, which a git worktree lacks unless the
//! year roots are linked in).
//!
//! Every legacy `unsat` whose property this module renders is completed with
//! [`instance_clauses`] (ids by position) under a v2 header, and `ny
//! proof-check` then matches each clause's sorted atoms against its OWN
//! parse's disjuncts and requires every disjunct to be matched
//! (`clause_mismatch` / `required_mismatch`). A disagreement turns a legacy
//! row into `invalid` — exit 20, the soundness-emergency code of
//! `proof_check_real` — for a bundle whose only fault is the writer's
//! rendering. The in-process lane tests use synthetic properties, so only the
//! corpus can show the two readers agree. Rows the writer cannot render are
//! the v2.1 `unavailable` form (`unconfirmed`), and rows the checker's reader
//! refuses are its own scope limit (exit 31, `vnnlib_form`, after the bundle
//! parsed): both well-formed, both counted, neither asserted away.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::test_faces::checker_disjuncts_if_parsed;
use crate::commands::proof_bundle::instance_clauses;

/// The corpus: `(tag, vnnlib)` of every distinct (by content) property named
/// by an `instances.csv` row of the VNN-COMP 2025 tree and the 2026 tree's
/// 1.0 / 2.0 layouts, plus `(tag, bytes)` of every row SKIPPED because its
/// file is larger than `cap` (never read, reported by the caller).
type Corpus = (Vec<(String, PathBuf)>, Vec<(String, u64)>);

fn corpus(cap: Option<u64>) -> Corpus {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks");
    let mut csvs: Vec<(String, PathBuf)> = Vec::new();
    for (year, layouts) in [
        ("vnncomp2025", &["."][..]),
        ("vnncomp2026", &[".", "1.0", "2.0"][..]),
    ] {
        let dir = root.join(year).join("benchmarks");
        assert!(
            dir.is_dir(),
            "{} is missing: this external-vnncomp test needs the MAIN tree's gitignored \
             benchmarks/ (in a worktree, link the year roots in: the CLAUDE.md worktree trap)",
            dir.display()
        );
        let mut cats: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("benchmarks dir")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        cats.sort();
        for cat in cats {
            let name = cat
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            for layout in layouts {
                let csv = cat.join(layout).join("instances.csv");
                if csv.is_file() {
                    csvs.push((format!("{year}/{name}/{layout}"), csv));
                }
            }
        }
    }
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut seen_path: HashSet<PathBuf> = HashSet::new();
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    for (tag, csv) in csvs {
        let base = csv.parent().expect("csv dir").to_path_buf();
        let text = std::fs::read_to_string(&csv).expect("instances.csv");
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            // A dual-network row quotes its ONNX tuple: the VNN-LIB follows it.
            let rest = if let Some(quoted) = line.strip_prefix('"') {
                match quoted.find("\",") {
                    Some(i) => &quoted[i + 2..],
                    None => continue,
                }
            } else {
                match line.find(',') {
                    Some(i) => &line[i + 1..],
                    None => continue,
                }
            };
            let Some(field) = rest.split(',').next() else {
                continue;
            };
            let path = base.join(field.trim());
            if !seen_path.insert(path.clone()) {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            if cap.is_some_and(|c| meta.len() > c) {
                skipped.push((tag.clone(), meta.len()));
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            if seen.insert(Sha256::digest(&bytes).to_vec()) {
                out.push((tag.clone(), path));
            }
        }
    }
    (out, skipped)
}

/// How `ny proof-check` would meet the writer's fallback clauses of `path`:
/// `Ok(kind)` for a well-formed outcome, `Err(why)` for a disagreement (the
/// checker would report `invalid`).
fn classify(path: &Path) -> Result<String, String> {
    let list = match instance_clauses(path) {
        Ok(list) => list,
        Err(why) => return Ok(format!("v2_1:{why}")),
    };
    if path.extension().is_some_and(|e| e == "gz") {
        return Ok("checker_scope:gz".to_string());
    }
    let text = std::fs::read_to_string(path).expect("the writer read it");
    // The checker's reader through this module's parent's test face (the
    // ratchet: only that file names the checker crate inside ny-cli).
    let Some(regions) = checker_disjuncts_if_parsed(&text) else {
        return Ok("checker_scope:vnnlib_form".to_string());
    };
    // `check_inner`'s clause matching, verbatim in effect: region 0 (the
    // writer's), sorted + deduplicated atoms joined by `&`, the FIRST
    // disjunct with that canonical text; then every disjunct of every region
    // must have been matched.
    let mut bad = Vec::new();
    let mut matched: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (id, atoms) in &list {
        let mut atoms = atoms.clone();
        atoms.sort();
        atoms.dedup();
        let canon = atoms.join("&");
        match regions
            .first()
            .and_then(|r| r.iter().position(|d| *d == canon))
        {
            Some(k) => {
                matched.insert((0, k));
            }
            None => bad.push(format!("clause_mismatch id={id} `{canon}`")),
        }
    }
    for (r, region) in regions.iter().enumerate() {
        for (k, d) in region.iter().enumerate() {
            if !matched.contains(&(r, k)) {
                bad.push(format!("required_mismatch region={r} disjunct={k} `{d}`"));
            }
        }
    }
    if bad.is_empty() {
        Ok("agree".to_string())
    } else {
        bad.truncate(3);
        Err(bad.join("; "))
    }
}

/// Every corpus property through [`classify`] on `threads` workers; prints
/// the per-tag histogram and asserts the pin (module docs). `cap` skips (and
/// reports) larger files; `min_rows` / `min_agree` are the VOID guards.
fn run_corpus(cap: Option<u64>, threads: usize, min_rows: usize, min_agree: usize) {
    let (rows, skipped) = corpus(cap);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<(usize, Result<String, String>)>> =
        std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some((_, path)) = rows.get(i) else {
                    break;
                };
                let r = classify(path);
                results
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push((i, r));
            });
        }
    });
    let mut results = results
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    results.sort_by_key(|(i, _)| *i);
    assert_eq!(results.len(), rows.len(), "every property classified");
    let mut hist: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut disagree: Vec<String> = Vec::new();
    for (i, r) in results {
        let (tag, path) = &rows[i];
        let kind = match r {
            Ok(kind) => kind,
            Err(why) => {
                disagree.push(format!("{tag} {}: {why}", path.display()));
                "DISAGREE".to_string()
            }
        };
        *hist
            .entry(tag.clone())
            .or_default()
            .entry(kind)
            .or_insert(0) += 1;
    }
    for (tag, _) in &skipped {
        *hist
            .entry(tag.clone())
            .or_default()
            .entry("skipped_over_cap".to_string())
            .or_insert(0) += 1;
    }
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    for (tag, kinds) in &hist {
        let row: Vec<String> = kinds.iter().map(|(k, n)| format!("{k}={n}")).collect();
        eprintln!("[legacy-corpus] {tag} {}", row.join(" "));
        for (k, n) in kinds {
            let family = k.split(':').next().unwrap_or(k).to_string();
            *totals.entry(family).or_insert(0) += n;
        }
    }
    eprintln!(
        "[legacy-corpus] summary cap={cap:?} properties={} skipped_over_cap={} \
         (largest {} B) {}",
        rows.len(),
        skipped.len(),
        skipped.iter().map(|(_, b)| *b).max().unwrap_or(0),
        totals
            .iter()
            .map(|(k, n)| format!("{k}={n}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let margin_row_unrendered: usize = hist
        .iter()
        .filter(|(tag, _)| {
            [
                "cifar100_2024",
                "tinyimagenet_2024",
                "challenging_certified_training_2026",
            ]
            .iter()
            .any(|c| tag.contains(&format!("/{c}/")))
        })
        .flat_map(|(_, kinds)| kinds.iter())
        .filter(|(k, _)| k.starts_with("v2_1:"))
        .map(|(_, n)| *n)
        .sum();
    eprintln!("[legacy-corpus] margin_row_categories_unrendered={margin_row_unrendered}");
    assert!(
        rows.len() >= min_rows,
        "only {} distinct properties classified: VOID",
        rows.len()
    );
    assert!(
        totals.get("agree").copied().unwrap_or(0) >= min_agree,
        "too few properties reached the clause match: VOID ({totals:?})"
    );
    assert!(
        disagree.is_empty(),
        "{} propert(ies) where the writer's fallback clauses would make a legacy bundle \
         INVALID in the checker:\n{}",
        disagree.len(),
        disagree
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// THE CORPUS PIN (debug-affordable): every property file of at most 8 MB
/// (the larger nn4sys / collins_aerospace / vggnet16 / smart_turn files are
/// counted `skipped_over_cap` and left to the release-only `_heavy` pin): no
/// property the writer renders makes a well-formed legacy bundle `invalid`
/// in the checker. Also printed: how many margin-row-category properties the
/// writer cannot render (harmless since the writer's lazy header, `Stream`
/// in `proof_bundle`: a lane's own clauses keep its bundle v2).
#[test]
fn legacy_fallback_clauses_agree_with_the_checker_reader_on_the_corpus() {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(16));
    run_corpus(Some(8 << 20), threads, 2000, 1000);
}

/// `legacy_fallback_clauses_agree_with_the_checker_reader_on_the_corpus`
/// on EVERY property file (no size cap; files up to ~250 MB): release test
/// builds only (`targo --unverified test --release -p ny-cli --features
/// external-vnncomp --bin ny legacy_fallback_clauses`), the precedent of
/// `frozen_dual_bab_agrees_with_the_p1_replica_node_counts_heavy`; four
/// workers, so the largest parses stay a few GiB.
#[cfg(not(debug_assertions))]
#[test]
fn legacy_fallback_clauses_agree_with_the_checker_reader_on_the_corpus_heavy() {
    run_corpus(None, 4, 2000, 1000);
}
