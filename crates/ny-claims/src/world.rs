// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The world a claim is re-checked against.
//!
//! Two facts only, because these are the two the reversals turned on: WHICH
//! BINARY would reproduce the measurement, and WHICH HOST is asking.
//!
//! The binary fact is deliberately not typed by the caller. `ny claims verify`
//! hosted inside `ny` uses the running process's own stamp; the standalone
//! `ny-claims` binary ASKS the scored executable who it is, by running
//! `--build-stamp`, which is the free (no-hash) half of the identity that
//! `crates/ny-cli/src/build_id.rs` already emits. Neither path lets a human
//! assert a build id, because "the binary is the one I think it is" is
//! reversal 2 and reversal 3.

use std::path::{Path, PathBuf};

/// What the current verifier binary says it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryFacts {
    /// `stamp=` — the digest of the compiled stamp.
    pub stamp: String,
    /// `version=`.
    pub version: String,
    /// Where this came from, printed in every drift report so a reader can
    /// tell "the running process" from "the file at target/release/ny".
    pub source: String,
}

impl BinaryFacts {
    /// Parse a `NY_BUILD_STAMP_V1 …` or `NY_BUILD_ID_V1 …` line.
    pub fn from_identity_line(line: &str, source: &str) -> Option<Self> {
        if !line.starts_with("NY_BUILD_STAMP_V1") && !line.starts_with("NY_BUILD_ID_V1") {
            return None;
        }
        let fields: std::collections::BTreeMap<&str, &str> = line
            .split_whitespace()
            .skip(1)
            .filter_map(|pair| pair.split_once('='))
            .collect();
        Some(Self {
            stamp: (*fields.get("stamp")?).to_string(),
            version: fields
                .get("version")
                .copied()
                .unwrap_or("unknown")
                .to_string(),
            source: source.to_string(),
        })
    }

    /// Ask the scored executable who it is. Returns `None` when it is not
    /// built, or is too old to answer — both of which are honest answers and
    /// neither of which may be silently treated as "matches".
    pub fn probe_executable(path: &Path) -> Option<Self> {
        let output = std::process::Command::new(path)
            .arg("--build-stamp")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        let line = text
            .lines()
            .find(|line| line.starts_with("NY_BUILD_STAMP_V1"))?;
        Self::from_identity_line(line, &format!("{} --build-stamp", path.display()))
    }
}

#[derive(Debug, Clone)]
pub struct World {
    /// Repo root. Every artifact path is relative to this.
    pub root: PathBuf,
    pub binary: Option<BinaryFacts>,
    /// Hostname. The campaign has already lost a result to a figure that was
    /// true on one machine (`97/99` on an RTX 5080) and false on the machine
    /// it was being acted on from.
    pub host: String,
}

impl World {
    /// The default probe: the scored `target/release/ny`.
    pub fn probe(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let binary = BinaryFacts::probe_executable(&root.join("target/release/ny"));
        Self {
            root,
            binary,
            host: hostname(),
        }
    }

    /// For the in-`ny` host: the running process already knows its own stamp,
    /// and that is strictly better evidence than re-running a file by path.
    pub fn with_binary(root: impl Into<PathBuf>, binary: Option<BinaryFacts>) -> Self {
        Self {
            root: root.into(),
            binary,
            host: hostname(),
        }
    }
}

/// Hostname without a `libc` dependency: the two files/vars every Linux and
/// macOS box here actually has, then `uname -n`.
pub fn hostname() -> String {
    if let Ok(name) = std::fs::read_to_string("/etc/hostname") {
        let name = name.trim();
        if !name.is_empty() {
            return name.to_string();
        }
    }
    if let Ok(name) = std::env::var("HOSTNAME") {
        if !name.is_empty() {
            return name;
        }
    }
    std::process::Command::new("uname")
        .arg("-n")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |name| name.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_real_build_stamp_line_parses() {
        // Verbatim from `scratchpad/m3ab/identity.txt`, the sealed identity of
        // the binary that ran the voided M3 A/B.
        let line = "NY_BUILD_ID_V1 version=0.3.4760+dirty.c8171fe3.63f0465b features=mip,cuda \
                    backend=cuda+wgpu markers=margin-row-gpu-eft-impl-v9 \
                    commit=c8171fe3e01166ef26715dde60b4afd31dab495b dirty=true revision=4760 \
                    source=63f0465b6e936b26 profile=release target=aarch64-unknown-linux-gnu \
                    rustc=1.99.0-dev-1183ee6ed \
                    stamp=2e3c0e3b9c39b33604b254a56189493f4782462c85f0b7769fe44235f8559b29 \
                    exe=3a4d5b6baefc1c13a02a02cee2aa3e78ec894c1fb8324dab7c5516d37141a1cb \
                    id=a6e5d9e2a945c37c89ba946a6de6057b6d045a33a5ef8595f7b6b81ec3564ad4";
        let facts = BinaryFacts::from_identity_line(line, "test").expect("parse");
        assert_eq!(
            facts.stamp,
            "2e3c0e3b9c39b33604b254a56189493f4782462c85f0b7769fe44235f8559b29"
        );
        assert_eq!(facts.version, "0.3.4760+dirty.c8171fe3.63f0465b");
    }

    #[test]
    fn a_non_identity_line_is_refused() {
        assert!(BinaryFacts::from_identity_line("ny 0.3.0", "test").is_none());
    }
}
