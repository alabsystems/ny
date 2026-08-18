// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Guard: every workspace crate that pins the AY git repository must pin the SAME
//! revision.
//!
//! This is a submission-integrity check, not a style preference. Two revisions of
//! one Git repo make Cargo resolve two copies of every `ay-*` crate, which breaks
//! the scored path at two independent gates:
//!
//!   1. `vnncomp_scripts/build_submission_binary.sh` builds with `--locked`, and a
//!      lockfile carrying both revisions no longer matches the manifests:
//!      error: cannot update the lock file ... because --locked was passed
//!      The required `mip,cuda` tier fails and the builder exits WITHOUT a binary.
//!   2. `vnncomp_scripts/submission_binary_receipt.sh` `receipt_ay_commit()` fails
//!      closed on multiple pins, so no provenance receipt is written.
//!
//! With no `target/release/ny`, `run_instance.sh` writes `error` for EVERY instance
//! and the official scorer gives 0 to anything that is not holds/violated. So a
//! split pin zeroes the entire submission, not one benchmark.
//!
//! It is invisible in normal development because a plain `cargo build` (no
//! `--locked`) resolves both revisions side by side and succeeds. Only the sealed
//! builder rejects it — which is why this cheap test exists at the `cargo test`
//! layer, where it is seen immediately.
//!
//! This has now happened twice, both times because one crate was bumped alone:
//!   * 2026-08-05, ny-mip -> 1f15fd8c while ny-cli stayed at 6bb7453d
//!   * 2026-08-07, ny-mip -> 2e41cf2b while ny-cli stayed at 38a5d4f9
//!
//! Both were caught only by running the sealed builder by hand.
//!
//! When bumping AY, bump EVERY pin below together and regenerate `Cargo.lock`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Manifests that carry an AY git pin. Add new ones here when they appear.
const AY_PINNED_MANIFESTS: &[&str] = &["crates/ny-cli/Cargo.toml", "crates/ny-mip/Cargo.toml"];

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <root>/crates/ny-cli.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root is two levels above crates/ny-cli")
        .to_path_buf()
}

/// Extract every 40-hex `rev = "..."` on a line that names the AY repository.
fn ay_revisions(manifest: &str) -> Vec<String> {
    manifest
        .lines()
        .filter(|line| line.contains("alabsystems/ay"))
        .filter_map(|line| {
            let start = line.find("rev = \"")? + "rev = \"".len();
            let rev = line.get(start..start + 40)?;
            rev.chars()
                .all(|c| c.is_ascii_hexdigit())
                .then(|| rev.to_string())
        })
        .collect()
}

#[test]
fn every_ay_pin_in_the_workspace_names_the_same_revision() {
    let root = workspace_root();
    let mut by_manifest: BTreeMap<&str, Vec<String>> = BTreeMap::new();

    for relative in AY_PINNED_MANIFESTS {
        let path = root.join(relative);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let revisions = ay_revisions(&text);
        assert!(
            !revisions.is_empty(),
            "{relative} no longer carries an AY git pin. If the dependency moved, update \
             AY_PINNED_MANIFESTS in this test — do not delete the guard, it protects the \
             sealed submission build."
        );
        by_manifest.insert(relative, revisions);
    }

    let distinct: std::collections::BTreeSet<&String> = by_manifest.values().flatten().collect();
    assert_eq!(
        distinct.len(),
        1,
        "AY pins disagree across the workspace: {by_manifest:#?}\n\
         Two revisions of one Git repo make Cargo resolve two copies of every ay-* crate. \
         The sealed builder (--locked) then fails, no binary and no receipt are produced, \
         run_instance.sh writes `error` for every instance, and the WHOLE SUBMISSION scores 0. \
         Bump every pin together and regenerate Cargo.lock."
    );
}

/// `Cargo.lock` must agree with the manifests. A stale lock fails `--locked` even
/// when the manifests themselves are consistent, which is the other half of the
/// same outage.
#[test]
fn the_lockfile_resolves_exactly_one_ay_revision() {
    let lock_path = workspace_root().join("Cargo.lock");
    let lock = std::fs::read_to_string(&lock_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", lock_path.display()));

    let revisions: std::collections::BTreeSet<String> = lock
        .lines()
        .filter(|line| line.contains("alabsystems/ay"))
        .filter_map(|line| {
            let start = line.find("?rev=")? + "?rev=".len();
            let rev = line.get(start..start + 40)?;
            rev.chars()
                .all(|c| c.is_ascii_hexdigit())
                .then(|| rev.to_string())
        })
        .collect();

    assert!(
        !revisions.is_empty(),
        "Cargo.lock names no AY revision; if AY is no longer a dependency, update this guard."
    );
    assert_eq!(
        revisions.len(),
        1,
        "Cargo.lock resolves {} distinct AY revisions: {revisions:#?}\n\
         Regenerate it after unifying the manifest pins; `cargo build --locked` (which the \
         sealed submission builder uses) refuses a lock that disagrees with the manifests.",
        revisions.len()
    );
}
