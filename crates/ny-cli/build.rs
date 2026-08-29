// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Build stamp for the `ny` binary (#binary-identity).
//!
//! ## Why this exists
//!
//! `[workspace.package] version` sat at `0.3.0` for the whole VNN-COMP
//! campaign — it did not move until the `0.3.0 -> 0.4.0` release bump in
//! `b791c199a` (2026-08-27). Every measurable change — the charge fix, the GPU
//! EFT seams at impl-v9, the sweep-poison fix, safenlp — produced a *different
//! verifier* that still answered `ny 0.3.0`. Meanwhile the sibling `ay`
//! crate moved 0.13 -> 0.14 -> 0.15 in a day and ny pins it by version+rev,
//! which is exactly why AY drift is visible and ny's own drift is not.
//!
//! A manual bump gets skipped. An automatic one does not. So the PATCH
//! component is derived here from `git rev-list --count HEAD`: two different
//! commits can never report the same version, without anyone remembering to
//! edit a TOML file.
//!
//! Most measurement in this repo happens on a DIRTY tree, which is precisely
//! when provenance is silently lost, so a dirty build is marked explicitly and
//! carries a digest of its own working-tree diff. Two dirty builds at the same
//! commit with different edits get different versions.
//!
//! ## What it emits
//!
//! | env var | meaning |
//! |---|---|
//! | `NY_BUILD_VERSION` | the display version, `<major>.<minor>.<rev-list count>` plus an optional `+dirty.<commit>.<digest>` suffix |
//! | `NY_BUILD_REVISION` | `git rev-list --count HEAD`, the auto-derived patch |
//! | `NY_BUILD_COMMIT` | full 40-hex HEAD |
//! | `NY_BUILD_COMMIT_SHORT` | first 12 hex of HEAD |
//! | `NY_BUILD_DIRTY` | `true` / `false` / `unknown` |
//! | `NY_BUILD_SOURCE_DIGEST` | 16 hex over the build-relevant working-tree diff, or `clean` |
//! | `NY_BUILD_FEATURES_ALL` | every enabled cargo feature, sorted |
//! | `NY_BUILD_RUSTC` | `rustc -vV` release + short commit hash |
//! | `NY_BUILD_PROFILE` | cargo profile (`debug` / `release`) |
//! | `NY_BUILD_TARGET` | target triple |
//!
//! Nothing here can fail the build: a missing `git`, a tarball extraction, or
//! a shallow checkout degrades to `unknown`, which the runtime stamp reports
//! honestly rather than hiding.
//!
//! ## Staleness
//!
//! A working-tree edit does NOT touch `.git/index`, so watching the git
//! directory alone would leave `NY_BUILD_SOURCE_DIGEST` stale — i.e. exactly
//! the silent-wrong-provenance failure this file exists to stop. Every tracked
//! build-relevant path is therefore emitted as `cargo:rerun-if-changed`, the
//! same discipline `ty` uses in `crates/tla-cli/build.rs`.

use sha2::{Digest, Sha256};

/// Tracked prefixes whose contents can change the compiled binary. Docs,
/// reports, and benchmark corpora are deliberately excluded: dirtying a
/// markdown file must not invalidate a binary's identity or force a relink.
const BUILD_RELEVANT_PATHS: &[&str] =
    &["crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"];

fn main() {
    let root = workspace_root();

    let commit = root
        .as_deref()
        .and_then(|root| git_stdout(root, &["rev-parse", "HEAD"]))
        .filter(|commit| is_lower_hex(commit, 40));
    let revision = root
        .as_deref()
        .and_then(|root| git_stdout(root, &["rev-list", "--count", "HEAD"]))
        .filter(|count| !count.is_empty() && count.bytes().all(|byte| byte.is_ascii_digit()));

    // Tracked-file changes only. An untracked scratch file in the repo is not
    // part of the compiled binary and must not make every build "dirty"; a new
    // untracked SOURCE file is caught below, because a tracked file has to
    // reference it for the compiler to see it.
    let dirty = root.as_deref().and_then(|root| {
        let mut args = vec!["status", "--porcelain=v1", "--untracked-files=no", "--"];
        args.extend_from_slice(BUILD_RELEVANT_PATHS);
        git_stdout_raw(root, &args).map(|out| !out.is_empty())
    });

    let source_digest = match dirty {
        Some(true) => root
            .as_deref()
            .and_then(|root| {
                let mut args = vec!["diff", "--binary", "--no-ext-diff", "HEAD", "--"];
                args.extend_from_slice(BUILD_RELEVANT_PATHS);
                git_stdout_raw(root, &args)
            })
            .map_or_else(
                || "unknown".to_string(),
                |diff| {
                    let mut hasher = Sha256::new();
                    hasher.update(b"ny.build-source-diff.v1");
                    hasher.update(&diff);
                    hex16(&hasher.finalize())
                },
            ),
        Some(false) => "clean".to_string(),
        None => "unknown".to_string(),
    };

    let dirty_text = match dirty {
        Some(true) => "true",
        Some(false) => "false",
        None => "unknown",
    };

    let base = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    let major_minor = major_minor(&base);
    let version = match (&revision, dirty, &commit) {
        (Some(revision), Some(false), _) => format!("{major_minor}.{revision}"),
        (Some(revision), Some(true), Some(commit)) => format!(
            "{major_minor}.{revision}+dirty.{}.{}",
            &commit[..8],
            &source_digest[..source_digest.len().min(8)]
        ),
        (Some(revision), _, _) => format!("{major_minor}.{revision}+unknowngit"),
        (None, _, _) => format!("{base}+nogit"),
    };

    emit("NY_BUILD_VERSION", &version);
    emit(
        "NY_BUILD_REVISION",
        revision.as_deref().unwrap_or("unknown"),
    );
    emit("NY_BUILD_COMMIT", commit.as_deref().unwrap_or("unknown"));
    emit(
        "NY_BUILD_COMMIT_SHORT",
        commit.as_deref().map_or("unknown", |commit| &commit[..12]),
    );
    emit("NY_BUILD_DIRTY", dirty_text);
    emit("NY_BUILD_SOURCE_DIGEST", &source_digest);
    emit("NY_BUILD_FEATURES_ALL", &enabled_features());
    emit("NY_BUILD_RUSTC", &rustc_identity());
    emit(
        "NY_BUILD_PROFILE",
        &std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string()),
    );
    emit(
        "NY_BUILD_TARGET",
        &std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
    );

    watch_git_metadata(root.as_deref());
    watch_tracked_sources(root.as_deref());
}

fn emit(name: &str, value: &str) {
    // A newline would forge extra cargo directives; the values above cannot
    // contain one, but refuse rather than trust that.
    let sanitized = value.replace(['\n', '\r'], "");
    println!("cargo:rustc-env={name}={sanitized}");
}

fn workspace_root() -> Option<std::path::PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    std::path::PathBuf::from(manifest)
        .join("..")
        .join("..")
        .canonicalize()
        .ok()
}

fn git_stdout_raw(root: &std::path::Path, args: &[&str]) -> Option<Vec<u8>> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

fn git_stdout(root: &std::path::Path, args: &[&str]) -> Option<String> {
    let bytes = git_stdout_raw(root, args)?;
    Some(String::from_utf8(bytes).ok()?.trim().to_string())
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Every enabled cargo feature, read from the `CARGO_FEATURE_*` variables
/// Cargo sets for this build, lower-cased back to their manifest spelling and
/// sorted.
///
/// Derived rather than hand-listed so a feature added to `Cargo.toml` next
/// month is part of the binary's identity for free — a hand-maintained list is
/// exactly the kind of thing that silently stops covering reality.
fn enabled_features() -> String {
    let mut features: Vec<String> = std::env::vars()
        .filter_map(|(name, _)| {
            name.strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect();
    features.sort();
    features.dedup();
    if features.is_empty() {
        "none".to_string()
    } else {
        features.join(",")
    }
}

fn hex16(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    digest
        .iter()
        .take(8)
        .fold(String::with_capacity(16), |mut out, byte| {
            // `write!` to a String is infallible; the Result is discarded rather
            // than unwrapped so this build script carries no panic path.
            let _ = write!(out, "{byte:02x}");
            out
        })
}

fn major_minor(version: &str) -> String {
    let mut parts = version.split('.');
    let major = parts.next().unwrap_or("0");
    let minor = parts.next().unwrap_or("0");
    format!("{major}.{minor}")
}

/// `rustc -vV` release plus its short commit hash: a nightly and a stable of
/// the same release number are different compilers, and the Trust toolchain
/// (`build-ny-with-trust-toolchain`) is different again.
fn rustc_identity() -> String {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let Ok(output) = std::process::Command::new(&rustc).arg("-vV").output() else {
        return "unknown".to_string();
    };
    if !output.status.success() {
        return "unknown".to_string();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut release = "unknown";
    let mut commit = "";
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("release: ") {
            release = value.trim();
        } else if let Some(value) = line.strip_prefix("commit-hash: ") {
            commit = value.trim();
        }
    }
    if commit.is_empty() || commit == "unknown" {
        release.to_string()
    } else {
        format!("{release}-{}", &commit[..commit.len().min(9)])
    }
}

/// HEAD, the ref store, and the index: a commit, checkout, or `git add`
/// changes the revision count or the dirty flag without touching any source
/// file.
fn watch_git_metadata(root: Option<&std::path::Path>) {
    let Some(root) = root else { return };
    for path in ["HEAD", "refs", "index"] {
        if let Some(resolved) = git_stdout(root, &["rev-parse", "--git-path", path]) {
            if !resolved.is_empty() {
                println!("cargo:rerun-if-changed={resolved}");
            }
        }
    }
}

/// Every tracked build-relevant file, so an uncommitted edit refreshes the
/// dirty digest instead of leaving a stale stamp compiled into the binary.
fn watch_tracked_sources(root: Option<&std::path::Path>) {
    let Some(root) = root else { return };
    let mut args = vec!["ls-files", "-z", "--"];
    args.extend_from_slice(BUILD_RELEVANT_PATHS);
    let Some(tracked) = git_stdout_raw(root, &args) else {
        return;
    };
    for record in tracked.split(|byte| *byte == 0) {
        if record.is_empty() {
            continue;
        }
        let Ok(relative) = std::str::from_utf8(record) else {
            continue;
        };
        if relative.contains('\n') || relative.contains('\r') {
            continue;
        }
        println!("cargo:rerun-if-changed={}", root.join(relative).display());
    }
}
