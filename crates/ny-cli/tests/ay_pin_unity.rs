// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Every AY dependency must share one immutable revision. A split pin can
//! resolve successfully, but the submission receipt refuses ambiguous AY
//! provenance. A stale lockfile separately fails the locked build.
//!
//! Use `scripts/bump_ay_pin.py` to update both manifests, the lockfile and
//! active identity claims together.

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
         The submission receipt requires one AY revision. \
         Use scripts/bump_ay_pin.py to update all identity surfaces together."
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
         Regenerate it after unifying the manifest pins; `targo --unverified build --locked` (which the \
         sealed submission builder uses) refuses a lock that disagrees with the manifests.",
        revisions.len()
    );
}
