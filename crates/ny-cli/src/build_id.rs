// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Binary identity (#binary-identity) for measurement provenance.
//!
//! `build.rs` derives the display version from the manifest's major/minor and
//! `git rev-list --count HEAD`, with a commit/diff suffix for dirty sources:
//!
//! ```text
//! <major>.<minor>.4745                         clean build at commit #4745
//! <major>.<minor>.4745+dirty.2c8538a3.9f1c2ab3  dirty tree: commit prefix + diff digest
//! ```
//!
//! Commit counts can coincide on different branches. The full stamp therefore
//! includes the commit, source digest, features, backend, markers and toolchain.
//! [`BinaryIdentity`] adds the executable SHA-256; [`BinaryIdentityGuard`]
//! rechecks its bytes to detect replacement at the same path.

use std::io::Read as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

/// Greppable prefix of the CONTENT-ADDRESSED identity: the compiled stamp plus
/// the sha256 of the executable file. Produced by `ny --build-id` and by the
/// harnesses that seal and re-verify a binary. Bump the suffix when a field is
/// removed or its meaning changes; adding a field at the end does not.
pub(crate) const SCHEMA: &str = "NY_BUILD_ID_V1";

/// Greppable prefix of the FREE identity: the compiled stamp only, no
/// filesystem access. Emitted by every run and carried by every flight
/// receipt. Shares its `stamp=` digest with [`SCHEMA`].
pub(crate) const STAMP_SCHEMA: &str = "NY_BUILD_STAMP_V1";

/// The version `ny --version` and clap report.
///
/// Falls back to the bare `Cargo.toml` version only when `build.rs` could not
/// run at all, which is itself worth seeing in a log.
pub(crate) const DISPLAY_VERSION: &str = match option_env!("NY_BUILD_VERSION") {
    Some(version) => version,
    None => concat!(env!("CARGO_PKG_VERSION"), "+nostamp"),
};

const fn stamped(value: Option<&'static str>) -> &'static str {
    match value {
        Some(value) => value,
        None => "unknown",
    }
}

/// The compile-time half of the identity: everything the binary knows about
/// itself without touching the filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(crate) struct BuildStamp {
    /// Auto-derived display version — the primary handle. See module docs.
    pub(crate) version: &'static str,
    /// `Cargo.toml`'s frozen `major.minor.patch`, retained so a reader can see
    /// which component the derivation replaced.
    pub(crate) cargo_version: &'static str,
    /// `git rev-list --count HEAD`: the derived patch component.
    pub(crate) revision: &'static str,
    pub(crate) commit: &'static str,
    pub(crate) commit_short: &'static str,
    /// `true` / `false` / `unknown` over build-relevant tracked paths.
    pub(crate) dirty: &'static str,
    /// 16-hex digest of the build-relevant working-tree diff, `clean`, or
    /// `unknown`. This is what separates two dirty builds at one commit.
    pub(crate) source_digest: &'static str,
    /// The verdict-relevant cargo feature tier, in the same spelling the
    /// submission receipt uses: `mip,cuda` / `mip` / `cuda` / `none`.
    pub(crate) features: &'static str,
    /// Every compiled cargo feature, sorted. Folded into the digest so a
    /// loader-feature flip still changes the id.
    pub(crate) features_all: &'static str,
    /// Which accelerator backends are COMPILED IN. The backend actually
    /// resolved on this host is a runtime fact and is reported separately by
    /// `compute_backend::detect()`, which every flight record already carries.
    pub(crate) backend: &'static str,
    /// Implementation provenance markers the campaign greps for under R8.
    pub(crate) markers: &'static str,
    pub(crate) profile: &'static str,
    pub(crate) target: &'static str,
    pub(crate) rustc: &'static str,
}

/// Canonical order matches `build_submission_binary.sh` and the submission
/// receipt (`features=mip,cuda`), so a human comparing the two never has to
/// normalize ordering.
const FEATURE_TIER: &str = match (cfg!(feature = "mip"), cfg!(feature = "cuda")) {
    (true, true) => "mip,cuda",
    (true, false) => "mip",
    (false, true) => "cuda",
    (false, false) => "none",
};

const BACKEND_CLASS: &str = if cfg!(feature = "cuda") {
    "cuda+wgpu"
} else {
    "wgpu"
};

impl BuildStamp {
    pub(crate) const fn current() -> Self {
        Self {
            version: DISPLAY_VERSION,
            cargo_version: env!("CARGO_PKG_VERSION"),
            revision: stamped(option_env!("NY_BUILD_REVISION")),
            commit: stamped(option_env!("NY_BUILD_COMMIT")),
            commit_short: stamped(option_env!("NY_BUILD_COMMIT_SHORT")),
            dirty: stamped(option_env!("NY_BUILD_DIRTY")),
            source_digest: stamped(option_env!("NY_BUILD_SOURCE_DIGEST")),
            features: FEATURE_TIER,
            features_all: stamped(option_env!("NY_BUILD_FEATURES_ALL")),
            backend: BACKEND_CLASS,
            markers: ny_propagate::margin_row::GPU_BACKWARD_PROVENANCE_MARKER,
            profile: stamped(option_env!("NY_BUILD_PROFILE")),
            target: stamped(option_env!("NY_BUILD_TARGET")),
            rustc: stamped(option_env!("NY_BUILD_RUSTC")),
        }
    }

    /// Digest of the compiled stamp alone. Newline-delimited `key=value` in a
    /// fixed order: adding a field at the end changes the digest, which is the
    /// intended behaviour — a different stamp is a different verifier.
    pub(crate) fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"ny.build-stamp.v1\n");
        for (key, value) in self.fields() {
            hasher.update(key.as_bytes());
            hasher.update(b"=");
            hasher.update(value.as_bytes());
            hasher.update(b"\n");
        }
        hex(&hasher.finalize())
    }

    fn fields(&self) -> [(&'static str, &'static str); 13] {
        [
            ("version", self.version),
            ("cargo_version", self.cargo_version),
            ("revision", self.revision),
            ("commit", self.commit),
            ("dirty", self.dirty),
            ("source", self.source_digest),
            ("features", self.features),
            ("features_all", self.features_all),
            ("backend", self.backend),
            ("markers", self.markers),
            ("profile", self.profile),
            ("target", self.target),
            ("rustc", self.rustc),
        ]
    }

    /// Compiled identity without filesystem I/O, for run logs and flight receipts.
    /// Its `stamp=` digest links to the executable identity in a sweep manifest.
    pub(crate) fn canonical_line(&self) -> String {
        format!(
            "{STAMP_SCHEMA} version={} features={} backend={} markers={} commit={} dirty={} \
             revision={} source={} profile={} target={} rustc={} stamp={}",
            self.version,
            self.features,
            self.backend,
            self.markers,
            self.commit,
            self.dirty,
            self.revision,
            self.source_digest,
            self.profile,
            self.target,
            self.rustc,
            self.digest(),
        )
    }
}

/// The compiled stamp bound to the exact bytes of one executable file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct BinaryIdentity {
    pub(crate) schema: &'static str,
    #[serde(flatten)]
    pub(crate) stamp: BuildStamp,
    pub(crate) path: PathBuf,
    pub(crate) size_bytes: u64,
    pub(crate) stamp_sha256: String,
    pub(crate) exe_sha256: String,
    /// `sha256(schema || stamp_sha256 || exe_sha256)`. The one value to
    /// compare: it moves when the source moves AND when the bytes move.
    pub(crate) id: String,
}

impl BinaryIdentity {
    /// Bind the running binary's stamp to the bytes at `path`.
    ///
    /// `path` is normally `current_exe()`, but a harness legitimately seals a
    /// binary it is about to invoke, so the path is explicit.
    pub(crate) fn of_file(path: &Path) -> Result<Self> {
        let stamp = BuildStamp::current();
        let (canonical, exe_sha256, size_bytes) = sha256_regular_file_nofollow(path)?;
        let stamp_sha256 = stamp.digest();
        let mut hasher = Sha256::new();
        hasher.update(SCHEMA.as_bytes());
        hasher.update(b"\n");
        hasher.update(stamp_sha256.as_bytes());
        hasher.update(b"\n");
        hasher.update(exe_sha256.as_bytes());
        hasher.update(b"\n");
        Ok(Self {
            schema: SCHEMA,
            stamp,
            path: canonical,
            size_bytes,
            stamp_sha256,
            exe_sha256,
            id: hex(&hasher.finalize()),
        })
    }

    /// The identity of the process that is running right now.
    pub(crate) fn current() -> Result<Self> {
        let exe = std::env::current_exe().context("locate the running ny executable")?;
        Self::of_file(&exe)
    }

    /// The one canonical line. Single-line, space-separated `key=value`, no
    /// value containing a space, prefixed by [`SCHEMA`] so `grep NY_BUILD_ID_V1`
    /// over any log or receipt finds exactly this.
    pub(crate) fn canonical_line(&self) -> String {
        format!(
            "{SCHEMA} version={} features={} backend={} markers={} commit={} dirty={} \
             revision={} source={} profile={} target={} rustc={} stamp={} exe={} id={}",
            self.stamp.version,
            self.stamp.features,
            self.stamp.backend,
            self.stamp.markers,
            self.stamp.commit,
            self.stamp.dirty,
            self.stamp.revision,
            self.stamp.source_digest,
            self.stamp.profile,
            self.stamp.target,
            self.stamp.rustc,
            self.stamp_sha256,
            self.exe_sha256,
            self.id,
        )
    }

    /// Short form for a mismatch report and for the sweep summary: enough to
    /// tell two verifiers apart at a glance, short enough to read twice.
    pub(crate) fn short(&self) -> String {
        format!(
            "ny {} [features={} backend={} exe={} id={}]",
            self.stamp.version,
            self.stamp.features,
            self.stamp.backend,
            &self.exe_sha256[..16.min(self.exe_sha256.len())],
            &self.id[..16.min(self.id.len())],
        )
    }

    pub(crate) fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("render build identity as JSON")
    }
}

/// Guards one measurement session against the binary changing underneath it.
///
/// Sealed once when the session starts; [`Self::reverify`] runs before EVERY
/// row, not once per manifest. That is the whole point: the re-sweep that
/// poisoned 330 rows hashed once and then trusted a mutable path forever.
#[derive(Debug, Clone)]
pub(crate) struct BinaryIdentityGuard {
    sealed: BinaryIdentity,
}

impl BinaryIdentityGuard {
    pub(crate) fn seal(path: &Path) -> Result<Self> {
        Ok(Self {
            sealed: BinaryIdentity::of_file(path)?,
        })
    }

    pub(crate) const fn sealed(&self) -> &BinaryIdentity {
        &self.sealed
    }

    /// Fail closed when the bytes at the sealed path are no longer the bytes
    /// that were sealed.
    ///
    /// `context` names what was about to happen, so the error says which row
    /// the bank stops at. A hard error, never a warning: a sweep whose binary
    /// changed must abort loudly rather than keep producing mixed data that
    /// later reads as a regression.
    pub(crate) fn reverify(&self, context: &str) -> Result<()> {
        let found = BinaryIdentity::of_file(&self.sealed.path).with_context(|| {
            format!(
                "re-verify the sealed ny executable {} before {context}",
                self.sealed.path.display()
            )
        })?;
        // `found` carries THIS process's compiled stamp, which is still the
        // sealed one; only the file bytes can have moved. Ask the file on disk
        // who IT thinks it is, so the report names the replacement rather than
        // just its digest.
        if found.exe_sha256 == self.sealed.exe_sha256 {
            return Ok(());
        }
        let replacement = describe_executable(&self.sealed.path)
            .unwrap_or_else(|| format!("{SCHEMA} <the replacement could not self-identify>"));
        bail!(
            "the ny executable changed underneath this measurement, before {context}.\n  \
             path:    {}\n  \
             sealed:  {}\n  \
             found:   exe_sha256={}\n  \
             on disk: {}\n\
             Every row measured after the change would be attributed to the sealed \
             binary and is not comparable with the rows before it. Refusing to \
             continue; start a new bank against the current binary.",
            self.sealed.path.display(),
            self.sealed.canonical_line(),
            found.exe_sha256,
            replacement,
        );
    }
}

/// Ask an executable to identify itself. Best effort: an older binary has no
/// `--build-id`, and that answer is itself informative.
fn describe_executable(path: &Path) -> Option<String> {
    let output = std::process::Command::new(path)
        .arg("--build-id")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().find(|line| line.starts_with(SCHEMA))?;
    Some(line.to_string())
}

/// Hash a regular file without following a final symlink, and report the
/// canonical path actually hashed.
///
/// Adopted from `ty`'s `sha256_regular_file_nofollow`: attribution must name
/// the file whose bytes were read, not the name someone used to reach it.
pub(crate) fn sha256_regular_file_nofollow(path: &Path) -> Result<(PathBuf, String, u64)> {
    let metadata =
        std::fs::symlink_metadata(path).with_context(|| format!("inspect {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!(
            "refusing to attribute a measurement to a symlink; name the real file: {}",
            path.display()
        );
    }
    if !metadata.is_file() {
        bail!("not a regular file: {}", path.display());
    }
    let canonical =
        std::fs::canonicalize(path).with_context(|| format!("resolve {}", path.display()))?;
    let mut file = std::fs::File::open(&canonical)
        .with_context(|| format!("open {} for hashing", canonical.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 128 * 1024];
    let mut size_bytes = 0_u64;
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("read {} while hashing", canonical.display()))?;
        if count == 0 {
            break;
        }
        size_bytes += count as u64;
        hasher.update(&buffer[..count]);
    }
    Ok((canonical, hex(&hasher.finalize()), size_bytes))
}

fn hex(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// The free stamp line, computed once per process.
///
/// Deliberately the STAMP and not the content-addressed id: this is called on
/// the scored per-instance path, where hashing a 133 MB executable would be
/// charged against budgets as small as 10 s — and the child has nothing to
/// compare a hash against anyway. Content addressing belongs to the harness
/// that owns the file across rows ([`BinaryIdentityGuard`]).
pub(crate) fn current_line() -> &'static str {
    static LINE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    LINE.get_or_init(|| BuildStamp::current().canonical_line())
}

/// Emit the identity line to stderr exactly once per process.
///
/// `eprintln!`, not `tracing`, for the same reason `compute_backend::log_once`
/// uses it: every scored entry point runs under `RUST_LOG=error`, so an
/// info-level line is invisible exactly where provenance matters.
pub(crate) fn log_once() {
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| eprintln!("{}", current_line()));
}

/// `ny --build-id` / `ny build-id` / `ny --build-stamp`, intercepted before
/// clap so they stay single stable stdout lines with no logging, preset, or
/// GPU side effects.
///
/// - `--build-stamp` is FREE (compiled fields only) and is what a
///   per-instance harness should call.
/// - `--build-id` additionally hashes the executable, which is the
///   content-address a swap detector needs. Pay it once per sweep, not once
///   per scored row.
///
/// Flags:
/// - `--json`   machine-readable form (`--build-id` only)
/// - `--expect-features <tier>` exit non-zero unless the compiled tier matches
pub(crate) fn run_cli(args: &[std::ffi::OsString], hash_executable: bool) -> Result<()> {
    let mut json = false;
    let mut expect: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy().into_owned();
        match arg.as_str() {
            "--json" => json = true,
            "--expect-features" => {
                index += 1;
                let value = args
                    .get(index)
                    .context("--expect-features needs a feature tier, e.g. mip,cuda")?;
                expect = Some(value.to_string_lossy().into_owned());
            }
            other => bail!("unknown build-id argument {other:?}"),
        }
        index += 1;
    }

    let stamp = BuildStamp::current();
    if hash_executable {
        let identity = BinaryIdentity::current()?;
        if json {
            println!("{}", identity.to_json()?);
        } else {
            println!("{}", identity.canonical_line());
        }
    } else if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema": STAMP_SCHEMA,
                "stamp": stamp,
                "stamp_sha256": stamp.digest(),
            }))
            .context("render build stamp as JSON")?
        );
    } else {
        println!("{}", stamp.canonical_line());
    }

    if let Some(expect) = expect {
        if expect != stamp.features {
            bail!(
                "feature tier mismatch: expected {expect:?}, this binary is {:?} (ny {})",
                stamp.features,
                stamp.version,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stamp must be present and parseable — not `unknown` placeholders —
    /// whenever the crate is built inside its own git checkout. A build.rs
    /// that silently stopped running is exactly the regression this catches.
    #[test]
    fn the_stamp_is_present_and_parseable() {
        let stamp = BuildStamp::current();
        assert_ne!(
            stamp.version, "unknown",
            "build.rs must stamp a version into the binary"
        );
        assert!(
            !stamp.version.contains("nostamp"),
            "build.rs did not run: version is {}",
            stamp.version
        );
        assert_eq!(
            stamp.commit.len(),
            40,
            "commit must be a full 40-hex sha, got {:?}",
            stamp.commit
        );
        assert!(
            stamp.revision.bytes().all(|byte| byte.is_ascii_digit()),
            "revision must be the decimal rev-list count, got {:?}",
            stamp.revision
        );
        assert!(
            matches!(stamp.dirty, "true" | "false"),
            "dirty must be decided inside a checkout, got {:?}",
            stamp.dirty
        );
        assert!(
            stamp.markers.starts_with("margin-row-gpu-eft-"),
            "the impl provenance marker must be carried, got {:?}",
            stamp.markers
        );
    }

    /// The derived patch must actually be the derived patch:
    /// `<major>.<minor>.<revision>`, never the frozen `Cargo.toml` patch. This
    /// is the whole point — a campaign of different binaries all reporting
    /// `0.3.0`.
    #[test]
    fn the_version_patch_is_the_git_revision_not_the_frozen_cargo_patch() {
        let stamp = BuildStamp::current();
        let mut cargo = stamp.cargo_version.split('.');
        let major = cargo.next().expect("major");
        let minor = cargo.next().expect("minor");
        assert!(
            stamp
                .version
                .starts_with(&format!("{major}.{minor}.{}", stamp.revision)),
            "version {:?} must be {major}.{minor}.<revision={}>",
            stamp.version,
            stamp.revision
        );
        assert_ne!(
            stamp.version, stamp.cargo_version,
            "a derived version that equals the frozen Cargo.toml version proves nothing"
        );
    }

    /// A dirty build must SAY it is dirty, in the version itself, and carry a
    /// digest of its own diff so two dirty builds at one commit differ.
    #[test]
    fn a_dirty_build_is_marked_in_the_version() {
        let stamp = BuildStamp::current();
        match stamp.dirty {
            "true" => {
                assert!(
                    stamp.version.contains("+dirty."),
                    "a dirty build must be visibly dirty, got {:?}",
                    stamp.version
                );
                assert_ne!(stamp.source_digest, "clean");
                assert_ne!(stamp.source_digest, "unknown");
            }
            "false" => {
                assert!(!stamp.version.contains("+dirty."));
                assert_eq!(stamp.source_digest, "clean");
            }
            other => panic!("unexpected dirty state {other:?}"),
        }
    }

    /// A FEATURE-SET change must change the id. The stamp digest is what the
    /// canonical id is built from, so exercising it on the digest covers both.
    #[test]
    fn a_feature_set_change_changes_the_id() {
        let mip = BuildStamp {
            features: "mip",
            features_all: "coreml,gguf,mip,pytorch",
            backend: "wgpu",
            ..BuildStamp::current()
        };
        let mip_cuda = BuildStamp {
            features: "mip,cuda",
            features_all: "coreml,cuda,gguf,mip,pytorch",
            backend: "cuda+wgpu",
            ..BuildStamp::current()
        };
        assert_ne!(
            mip.digest(),
            mip_cuda.digest(),
            "mip and mip,cuda must not share a stamp digest"
        );
        // And the tier alone is enough, even holding everything else equal:
        // the docs dispute is precisely `mip` vs `mip,cuda`.
        let only_tier_differs = BuildStamp {
            features: "mip,cuda",
            ..mip
        };
        assert_ne!(mip.digest(), only_tier_differs.digest());
    }

    /// Every stamp field must reach the digest. A field added to the struct
    /// but forgotten in `fields()` would be invisible to the id — a silent
    /// hole exactly like the one this module exists to close.
    #[test]
    fn every_stamp_field_reaches_the_digest() {
        let base = BuildStamp::current();
        let mutated = [
            BuildStamp {
                version: "x",
                ..base
            },
            BuildStamp {
                cargo_version: "x",
                ..base
            },
            BuildStamp {
                revision: "x",
                ..base
            },
            BuildStamp {
                commit: "x",
                ..base
            },
            BuildStamp {
                commit_short: "x",
                ..base
            },
            BuildStamp { dirty: "x", ..base },
            BuildStamp {
                source_digest: "x",
                ..base
            },
            BuildStamp {
                features: "x",
                ..base
            },
            BuildStamp {
                features_all: "x",
                ..base
            },
            BuildStamp {
                backend: "x",
                ..base
            },
            BuildStamp {
                markers: "x",
                ..base
            },
            BuildStamp {
                profile: "x",
                ..base
            },
            BuildStamp {
                target: "x",
                ..base
            },
            BuildStamp { rustc: "x", ..base },
        ];
        let baseline = base.digest();
        // `commit_short` is derived from `commit` and deliberately absent from
        // the digest; every other field must move it.
        for stamp in mutated {
            if stamp.commit_short != base.commit_short {
                continue;
            }
            assert_ne!(
                stamp.digest(),
                baseline,
                "a changed stamp field left the digest untouched: {stamp:?}"
            );
        }
    }

    /// A same-path different-bytes swap must be DETECTED. Simulated with two
    /// files, because a real swap is exactly "the path stayed, the bytes
    /// changed" — and mtime is not consulted anywhere in this module.
    #[test]
    fn a_same_path_byte_swap_is_detected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ny");
        std::fs::write(&path, b"binary-A").expect("write A");
        let first = BinaryIdentity::of_file(&path).expect("identity A");

        // Same path, same name, and the mtime deliberately restored to the
        // pre-swap value: `R8: verify provenance by mtime` would pass here, and
        // only content addressing catches it.
        let before = std::fs::metadata(&path).expect("metadata").modified();
        std::fs::write(&path, b"binary-B-with-different-length").expect("write B");
        if let Ok(before) = before {
            let handle = std::fs::File::options()
                .write(true)
                .open(&path)
                .expect("reopen for times");
            handle
                .set_times(std::fs::FileTimes::new().set_modified(before))
                .expect("restore mtime");
            assert_eq!(
                std::fs::metadata(&path).expect("metadata").modified().ok(),
                Some(before),
                "the swap must be mtime-invisible for this test to mean anything"
            );
        }
        let second = BinaryIdentity::of_file(&path).expect("identity B");

        assert_eq!(first.path, second.path, "the swap must be same-path");
        assert_ne!(first.exe_sha256, second.exe_sha256);
        assert_ne!(first.id, second.id, "the canonical id must move with bytes");
        assert_eq!(
            first.stamp_sha256, second.stamp_sha256,
            "the compiled stamp is identical; only content addressing catches this"
        );
    }

    /// The mid-run abort fires, and names both ids.
    #[test]
    fn the_mid_run_abort_fires_with_both_ids() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ny");
        std::fs::write(&path, b"the sealed binary").expect("seal");
        let guard = BinaryIdentityGuard::seal(&path).expect("guard");
        guard
            .reverify("row 1")
            .expect("an unchanged binary must pass");

        std::fs::write(&path, b"a different binary").expect("swap");
        let error = guard
            .reverify("row 2")
            .expect_err("a changed binary must be a hard error");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("changed underneath this measurement"));
        assert!(rendered.contains("before row 2"), "{rendered}");
        assert!(
            rendered.contains(&guard.sealed().exe_sha256),
            "the sealed id must be printed: {rendered}"
        );
        let found = BinaryIdentity::of_file(&path).expect("found");
        assert!(
            rendered.contains(&found.exe_sha256),
            "the found id must be printed side by side: {rendered}"
        );
    }

    /// The FREE stamp line must name the verifier without touching the
    /// filesystem, and must cross-link to the content-addressed id through a
    /// shared `stamp=` digest.
    ///
    /// The split is load-bearing: the scored path emits this line per
    /// instance, and hashing a 133 MB executable there would be charged
    /// against budgets as small as 10 s.
    #[test]
    fn the_free_stamp_line_names_the_verifier_and_cross_links_to_the_full_id() {
        let stamp = BuildStamp::current();
        let line = stamp.canonical_line();
        assert_eq!(line.lines().count(), 1, "{line}");
        assert!(line.starts_with(STAMP_SCHEMA), "{line}");
        assert!(
            !line.contains(" exe="),
            "the stamp must not claim a file hash"
        );

        let fields: std::collections::BTreeMap<&str, &str> = line
            .split_whitespace()
            .skip(1)
            .map(|pair| pair.split_once('=').expect("key=value"))
            .collect();
        assert_eq!(fields["version"], stamp.version);
        assert_eq!(fields["features"], stamp.features);
        assert_eq!(fields["commit"], stamp.commit);
        assert_eq!(fields["dirty"], stamp.dirty);
        assert_eq!(fields["source"], stamp.source_digest);
        assert_eq!(fields["markers"], stamp.markers);
        assert_eq!(fields["stamp"], stamp.digest());
        assert_eq!(fields.len(), 12, "field count changed: {line}");

        // Same digest on both sides, so a run log and a sweep manifest can be
        // matched to each other without re-hashing anything.
        let identity = BinaryIdentity::current().expect("identity");
        assert_eq!(identity.stamp_sha256, stamp.digest());
        assert_eq!(current_line(), line);

        // Compare explicit tiers so the test also runs in a CUDA build.
        let baseline_tier = BuildStamp {
            features: "mip",
            backend: "wgpu",
            ..stamp
        };
        let cuda_tier = BuildStamp {
            features: "mip,cuda",
            backend: "cuda+wgpu",
            ..stamp
        };
        assert_ne!(cuda_tier.canonical_line(), baseline_tier.canonical_line());
    }

    /// The canonical line is one line, greppable, and round-trips every field
    /// a consumer parses.
    #[test]
    fn the_canonical_line_is_one_greppable_line() {
        let identity = BinaryIdentity::current().expect("identity");
        let line = identity.canonical_line();
        assert_eq!(line.lines().count(), 1, "{line}");
        assert!(line.starts_with(SCHEMA), "{line}");
        let fields: std::collections::BTreeMap<&str, &str> = line
            .split_whitespace()
            .skip(1)
            .map(|pair| pair.split_once('=').expect("key=value"))
            .collect();
        assert_eq!(fields["version"], identity.stamp.version);
        assert_eq!(fields["features"], identity.stamp.features);
        assert_eq!(fields["commit"], identity.stamp.commit);
        assert_eq!(fields["exe"], identity.exe_sha256);
        assert_eq!(fields["id"], identity.id);
        assert_eq!(fields.len(), 14, "field count changed: {line}");
    }

    /// The JSON form carries the same identity as the line.
    #[test]
    fn the_json_form_is_machine_readable_and_agrees_with_the_line() {
        let identity = BinaryIdentity::current().expect("identity");
        let value: serde_json::Value =
            serde_json::from_str(&identity.to_json().expect("json")).expect("parse");
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["version"], identity.stamp.version);
        assert_eq!(value["features"], identity.stamp.features);
        assert_eq!(value["exe_sha256"], identity.exe_sha256);
        assert_eq!(value["id"], identity.id);
    }

    /// The feature tier spelling must match what the submission receipt and
    /// build script use, so a human comparison never needs normalizing.
    #[test]
    fn the_feature_tier_spelling_matches_the_submission_receipt() {
        assert!(matches!(FEATURE_TIER, "mip,cuda" | "mip" | "cuda" | "none"));
        assert_eq!(FEATURE_TIER, BuildStamp::current().features);
    }
}
