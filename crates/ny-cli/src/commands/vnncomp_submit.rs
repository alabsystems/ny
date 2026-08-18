// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP submission packaging helpers.

use anyhow::{anyhow, bail, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const REQUIRED_SCRIPTS: [&str; 3] = ["install_tool.sh", "prepare_instance.sh", "run_instance.sh"];
/// tar exclude globs for runtime-only dev data that rides along under the
/// packaged roots but is never referenced by the source build
/// (`cargo build -p ny-cli`) or the harness. `corpus/` holds MILP/SMT
/// differential-test instances (zstd/npz) — no `include_bytes!`/`build.rs`
/// pulls them in, so shipping them ~4x's the submission for zero build value.
///
/// Both globs MUST stay scoped to `crates/*` (NY's own first-party sources).
/// An unscoped glob can also match a future externally vendored crate whose
/// files are listed in `.cargo-checksum.json`. `verify_vendor_checksums`
/// remains the pack-time backstop for that packaging bug. Internal repositories
/// such as AY are Git-sourced and are never package roots here.
const EXCLUDE_GLOBS: [&str; 2] = ["crates/*/corpus", "crates/*/proptest-regressions"];
const PREBUILT_ARCHIVE: &str = "dist/bin/ny-x86_64-linux.xz";
const PREBUILT_CHECKSUM: &str = "dist/bin/ny-x86_64-linux.xz.sha256";
const PREBUILT_PROVENANCE: &str = "dist/bin/ny-x86_64-linux.provenance.txt";
const PREBUILT_BUILDER: &str = "scripts/vnncomp_trust_linux_build.sh";
const PREBUILT_VERIFIER: &str = "vnncomp_scripts/verify_prebuilt.py";
const SOURCE_PROVENANCE: &str = ".ny-vnncomp-source.txt";
const PREBUILT_FILES: [&str; 3] = [PREBUILT_ARCHIVE, PREBUILT_CHECKSUM, PREBUILT_PROVENANCE];
const PREBUILT_SCHEMA: &str = "ny-vnncomp-prebuilt-v1";
const SOURCE_PROVENANCE_SCHEMA: &str = "ny-vnncomp-source-v1";
const PREBUILT_TARGET: &str = "x86_64-unknown-linux-gnu";
const PREBUILT_FEATURES: &str = "mip,cuda";
const BUILD_PROVENANCE_PREFIX: &str = "ny.vnncomp.build.v1|";
const TRUST_GATE_RECEIPT_SCHEMA: &str = "ny-trust-gate-receipt-v1";
const TRUST_GATE_COMMANDS_V1: &str = "trust-types:json_digest\ntrust-clean:instantiator_ordering\ntrust-router:production_\nrust-ui:valtree-node-limit-unit-enum-array\nrust-ui:clean-island-collapsed-order\ncheck_all:check\ne2e_targo_trust_cli\ntrust_falsification_gate\ntargo-trust:version\ntargo-trust:doctor-json\n";
const MAX_PREBUILT_BINARY_BYTES: u64 = 512 * 1024 * 1024;
const MAX_PREBUILT_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_PREBUILT_CHECKSUM_BYTES: u64 = 1024;
const MAX_PREBUILT_PROVENANCE_BYTES: u64 = 16 * 1024;
const PREBUILT_MANIFEST_KEYS: [&str; 18] = [
    "schema",
    "target",
    "features",
    "trust_commit",
    "trust_bootstrap_mode",
    "trust_gate_status",
    "trust_gate_receipt_sha256",
    "trust_gate_commands_sha256",
    "trust_gate_log_sha256",
    "trustc_sha256",
    "trustc_version_sha256",
    "ny_commit",
    "cargo_lock_sha256",
    "ay_lock_commit",
    "builder_script_sha256",
    "onnxruntime_static_sha256",
    "binary_sha256",
    "package_sha256",
];
const PACKAGE_ROOTS: [&str; 18] = [
    ".cargo",
    "benchmarks/download_benchmarks.sh",
    "Cargo.lock",
    "Cargo.toml",
    "dist",
    "LICENSE",
    "README.md",
    "_typos.toml",
    "clippy.toml",
    "configs",
    "crates",
    "install_tool.sh",
    "prepare_instance.sh",
    "requirements.txt",
    "rust-toolchain.toml",
    "run_instance.sh",
    "scripts/vnncomp_coverage.py",
    "vnncomp_scripts",
];
/// Reviewed files that may be surfaced by the package-list audit before they
/// have been added to the index. This intentionally remains a narrow
/// allowlist: any other non-ignored untracked file under PACKAGE_ROOTS makes
/// even the audit fail closed. `write_tarball` separately requires every
/// packaged input to be committed before it emits commit-labelled provenance.
const REVIEWED_WORKTREE_FILES: [&str; 10] = [
    "crates/ny-cli/src/commands/vnncomp_benchmarks.rs",
    "crates/ny-cli/src/commands/vnncomp_submit.rs",
    "crates/ny-cli/src/commands/vnncomp_late_submit.rs",
    "crates/ny-cli/src/commands/vnncomp_matrix.rs",
    "crates/ny-cli/src/commands/vnncomp_2025_tracks.rs",
    "crates/ny-cli/src/commands/vnncomp_2026_tracks.rs",
    "crates/ny-cli/testdata/vnncomp2025_track_membership.csv",
    "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/bounded_shared_executor.rs",
    "crates/ny-propagate/src/beta_crown/engine/graph/objectives/cuda_beta_spsa.rs",
    PREBUILT_VERIFIER,
];

#[derive(Debug)]
struct ValidatedPrebuilt {
    files: BTreeMap<&'static str, Vec<u8>>,
    source_commit: String,
}

impl ValidatedPrebuilt {
    fn bytes(&self, relative: &str) -> &[u8] {
        self.files
            .get(relative)
            .map(Vec::as_slice)
            .expect("validated prebuilt contains every triplet member")
    }
}

/// Build a VNN-COMP submission tarball from the current working tree.
pub(crate) fn handle_vnncomp_submit_command(
    output: PathBuf,
    no_build: bool,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    let repo_root = find_repo_root(&std::env::current_dir()?)?;
    handle_vnncomp_submit_from_repo(&repo_root, output, no_build, dry_run, json_output)
}

fn handle_vnncomp_submit_from_repo(
    repo_root: &Path,
    output: PathBuf,
    no_build: bool,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    validate_harness(repo_root)?;

    if !no_build && !dry_run {
        run_build(repo_root)?;
    }

    let output = if output.is_absolute() {
        output
    } else {
        repo_root.join(output)
    };

    let included_paths = package_file_list(repo_root)?;
    if included_paths.is_empty() {
        bail!("no package paths found under {}", repo_root.display());
    }
    let prebuilt_included = included_paths
        .iter()
        .any(|path| path == "dist/bin/ny-x86_64-linux.xz");
    if !prebuilt_included {
        eprintln!("WARNING: optional dist/bin/ny-x86_64-linux.xz is absent; the package will ");
        eprintln!(
            "use a source build; authenticated access to exact Git-pinned AY, crates.io/ORT network access, and native build prerequisites remain required."
        );
    }

    if !dry_run {
        write_tarball(repo_root, &output, &included_paths)?;
    }

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "command": "vnncomp-submit",
                "repo_root": repo_root,
                "output": output,
                "dry_run": dry_run,
                "built": !no_build && !dry_run,
                "required_scripts": REQUIRED_SCRIPTS,
                "included_file_count": included_paths.len() + 1,
                "package_roots": PACKAGE_ROOTS,
                "prebuilt_included": prebuilt_included,
                "next_step": "Upload this tarball through the VNN-COMP evaluation website when tool submission opens."
            }))?
        );
    } else {
        println!("VNN-COMP submission package");
        println!("  repo:   {}", repo_root.display());
        println!("  output: {}", output.display());
        println!(
            "  build:  {}",
            if no_build {
                "skipped"
            } else if dry_run {
                "dry-run skipped"
            } else {
                "done"
            }
        );
        println!("  scripts: install_tool.sh, prepare_instance.sh, run_instance.sh");
        println!(
            "  prebuilt: {}",
            if prebuilt_included {
                "included"
            } else {
                "absent (source-build fallback)"
            }
        );
        if dry_run {
            println!("  dry-run: no tarball written");
        }
        println!("  files:  {}", included_paths.len() + 1);
        println!();
        println!("Upload the tarball through the VNN-COMP evaluation website when the tool submission window opens.");
    }

    Ok(())
}

pub(crate) fn find_repo_root(start: &Path) -> Result<PathBuf> {
    for ancestor in start.ancestors() {
        if ancestor.join("Cargo.toml").is_file() && ancestor.join("vnncomp_scripts").is_dir() {
            return Ok(ancestor.to_path_buf());
        }
    }
    Err(anyhow!(
        "could not find ny repo root from {}",
        start.display()
    ))
}

fn validate_harness(repo_root: &Path) -> Result<()> {
    for script in REQUIRED_SCRIPTS {
        let path = repo_root.join(script);
        if !path.is_file() {
            bail!("missing required VNN-COMP script: {}", path.display());
        }
        require_executable(&path)?;
    }

    for script in [
        "vnncomp_scripts/build_submission_binary.sh",
        "vnncomp_scripts/prepare_instance.sh",
        "vnncomp_scripts/run_instance.sh",
        "vnncomp_scripts/submission_binary_receipt.sh",
    ] {
        let path = repo_root.join(script);
        if !path.is_file() {
            bail!("missing VNN-COMP helper script: {}", path.display());
        }
        require_executable(&path)?;
    }
    let verifier = repo_root.join(PREBUILT_VERIFIER);
    if !verifier.is_file() {
        bail!("missing VNN-COMP prebuilt verifier: {}", verifier.display());
    }

    Ok(())
}

/// The VNN-COMP harness (and install_tool.sh) exec these scripts directly, so
/// a script that exists but lacks the exec bit still dies with "Permission
/// denied" on the eval image. Reject it at packaging time instead — including
/// on --dry-run, which runs no build that could mask the problem.
fn require_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(path)?.permissions().mode();
        if mode & 0o111 == 0 {
            bail!(
                "VNN-COMP script is not executable (mode {mode:o}): {}",
                path.display()
            );
        }
    }
    // Windows has no exec bit, so there is nothing to reject here. Packaging
    // from Windows therefore cannot catch a missing mode bit — the check that
    // matters runs on the Linux eval image, and on the Linux packaging host.
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn package_file_list(repo_root: &Path) -> Result<Vec<String>> {
    let output = Command::new("git")
        .arg("ls-files")
        .arg("--")
        .args(PACKAGE_ROOTS)
        .current_dir(repo_root)
        .output()?;
    if !output.status.success() {
        bail!("git ls-files failed with status {}", output.status);
    }

    let mut files: Vec<String> = String::from_utf8(output.stdout)?
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();

    for required in REQUIRED_SCRIPTS {
        include_existing_file(repo_root, &mut files, required);
    }
    for reviewed in REVIEWED_WORKTREE_FILES {
        include_existing_file(repo_root, &mut files, reviewed);
    }

    // The package-list audit surfaces reviewed worktree additions, because
    // silently omitting a newly-created source or compile-time input can yield
    // an archive that cannot build. Archive creation later rejects every
    // uncommitted package input: a Git commit label cannot honestly describe
    // bytes that exist only in the worktree. Ignored `dist/` artifacts stay on
    // their separate, strictly validated path below.
    let untracked = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard", "-z", "--"])
        .args(PACKAGE_ROOTS)
        .current_dir(repo_root)
        .output()?;
    if !untracked.status.success() {
        bail!(
            "git ls-files for untracked package inputs failed with status {}",
            untracked.status
        );
    }
    let unexpected: Vec<String> = String::from_utf8(untracked.stdout)?
        .split('\0')
        .filter(|line| !line.is_empty())
        .filter(|line| !files.iter().any(|included| included == line))
        .map(str::to_string)
        .collect();
    if !unexpected.is_empty() {
        bail!(
            "refusing to omit unreviewed untracked file(s) under VNN-COMP package roots: {}. \
             Commit them or explicitly review and add them to REVIEWED_WORKTREE_FILES",
            unexpected.join(", ")
        );
    }

    // `dist/` is intentionally ignored. A completely absent prebuilt is a
    // supported source-build package, but any partial, stale, unproven, or
    // mislabelled prebuilt is a release error rather than a warning/fallback.
    // Validation is static so an AArch64 release host can package the required
    // x86_64 evaluation artifact without executing foreign code.
    if validate_optional_prebuilt(repo_root)?.is_some() {
        files.extend(PREBUILT_FILES.map(str::to_string));
        // The installer independently authenticates the builder identity in
        // the sealed manifest. Include it only for prebuilt packages so the
        // source-only fallback's package surface remains unchanged.
        include_existing_file(repo_root, &mut files, PREBUILT_BUILDER);
    }

    files.sort();
    files.dedup();
    Ok(files)
}

fn include_existing_file(repo_root: &Path, files: &mut Vec<String>, path: &str) {
    if repo_root.join(path).is_file() && !files.iter().any(|existing| existing == path) {
        files.push(path.to_string());
    }
}

fn validate_optional_prebuilt(repo_root: &Path) -> Result<Option<ValidatedPrebuilt>> {
    validate_prebuilt_parent(repo_root)?;
    let mut present = Vec::new();
    for relative in PREBUILT_FILES {
        let path = repo_root.join(relative);
        match fs::symlink_metadata(&path) {
            Ok(metadata) => {
                if !metadata.file_type().is_file() {
                    bail!(
                        "prebuilt package member must be a regular file: {}",
                        path.display()
                    );
                }
                present.push(relative);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }

    if present.is_empty() {
        return Ok(None);
    }
    if present.len() != PREBUILT_FILES.len() {
        let missing: Vec<_> = PREBUILT_FILES
            .iter()
            .filter(|path| !present.contains(path))
            .copied()
            .collect();
        bail!("refusing partial VNN-COMP prebuilt: present={present:?}, missing={missing:?}");
    }

    let archive = repo_root.join(PREBUILT_ARCHIVE);
    let checksum = repo_root.join(PREBUILT_CHECKSUM);
    let provenance = repo_root.join(PREBUILT_PROVENANCE);
    let archive_bytes =
        read_bounded_file(&archive, MAX_PREBUILT_ARCHIVE_BYTES, "compressed prebuilt")?;
    let checksum_bytes = read_bounded_file(
        &checksum,
        MAX_PREBUILT_CHECKSUM_BYTES,
        "prebuilt checksum sidecar",
    )?;
    let provenance_bytes = read_bounded_file(
        &provenance,
        MAX_PREBUILT_PROVENANCE_BYTES,
        "prebuilt provenance",
    )?;
    let archive_sha256 = sha256_bytes(&archive_bytes);
    let expected_checksum = format!("{archive_sha256}  ny-x86_64-linux.xz\n");
    let checksum_contents = std::str::from_utf8(&checksum_bytes)?;
    if checksum_contents != expected_checksum {
        bail!(
            "prebuilt checksum sidecar is not the exact sha256sum record for {}",
            archive.display()
        );
    }

    let manifest = parse_prebuilt_manifest(&provenance_bytes)?;
    require_manifest_value(&manifest, "schema", PREBUILT_SCHEMA)?;
    require_manifest_value(&manifest, "target", PREBUILT_TARGET)?;
    require_manifest_value(&manifest, "features", PREBUILT_FEATURES)?;
    require_manifest_value(&manifest, "trust_bootstrap_mode", "seed")?;
    require_manifest_value(&manifest, "trust_gate_status", "passed")?;
    require_manifest_value(&manifest, "package_sha256", &archive_sha256)?;
    let commands_sha256 = sha256_bytes(TRUST_GATE_COMMANDS_V1.as_bytes());
    require_manifest_value(&manifest, "trust_gate_commands_sha256", &commands_sha256)?;

    require_lower_hex(
        manifest_value(&manifest, "trust_commit")?,
        40,
        "trust_commit",
    )?;
    for key in [
        "trust_gate_receipt_sha256",
        "trust_gate_commands_sha256",
        "trust_gate_log_sha256",
        "trustc_sha256",
        "trustc_version_sha256",
        "cargo_lock_sha256",
        "builder_script_sha256",
        "onnxruntime_static_sha256",
        "binary_sha256",
        "package_sha256",
    ] {
        require_lower_hex(manifest_value(&manifest, key)?, 64, key)?;
    }
    let receipt_sha256 = sha256_bytes(trust_gate_receipt_payload(&manifest)?.as_bytes());
    require_manifest_value(&manifest, "trust_gate_receipt_sha256", &receipt_sha256)?;

    let ny_commit = manifest_value(&manifest, "ny_commit")?.to_string();
    require_lower_hex(&ny_commit, 40, "ny_commit")?;
    ensure_prebuilt_source_binding(repo_root, &ny_commit)?;

    let lock_path = repo_root.join("Cargo.lock");
    let lock_bytes = fs::read(&lock_path)?;
    let lock_sha256 = sha256_bytes(&lock_bytes);
    require_manifest_value(&manifest, "cargo_lock_sha256", &lock_sha256)?;
    let ay_commit = exact_ay_lock_commit(&lock_bytes)?;
    require_manifest_value(&manifest, "ay_lock_commit", &ay_commit)?;

    let builder_sha256 = sha256_file(&repo_root.join("scripts/vnncomp_trust_linux_build.sh"))?;
    require_manifest_value(&manifest, "builder_script_sha256", &builder_sha256)?;

    let mut captured_archive = tempfile::NamedTempFile::new()?;
    captured_archive.write_all(&archive_bytes)?;
    captured_archive.flush()?;
    let integrity = Command::new("xz")
        .args(["-t", "--"])
        .arg(captured_archive.path())
        .status()?;
    if !integrity.success() {
        bail!("xz integrity validation failed for {}", archive.display());
    }

    let decompressed = tempfile::NamedTempFile::new()?;
    let output_file = decompressed.reopen()?;
    let status = Command::new("xz")
        .args(["-dc", "--"])
        .arg(captured_archive.path())
        .stdout(Stdio::from(output_file))
        .status()?;
    if !status.success() {
        bail!("failed to decompress prebuilt {}", archive.display());
    }
    let binary_len = decompressed.as_file().metadata()?.len();
    if binary_len > MAX_PREBUILT_BINARY_BYTES {
        bail!(
            "decompressed prebuilt is too large ({binary_len} bytes; limit {MAX_PREBUILT_BINARY_BYTES})"
        );
    }
    validate_x86_64_elf(decompressed.path())?;
    let binary_sha256 = sha256_file(decompressed.path())?;
    require_manifest_value(&manifest, "binary_sha256", &binary_sha256)?;
    validate_embedded_build_provenance(
        decompressed.path(),
        &expected_build_provenance(&manifest)?,
    )?;
    let embedded_commits = embedded_ay_build_commits(decompressed.path())?;
    if embedded_commits != vec![ay_commit.clone()] {
        bail!(
            "prebuilt embedded AY build.commit mismatch: expected {ay_commit}, found {embedded_commits:?}"
        );
    }

    Ok(Some(ValidatedPrebuilt {
        files: BTreeMap::from([
            (PREBUILT_ARCHIVE, archive_bytes),
            (PREBUILT_CHECKSUM, checksum_bytes),
            (PREBUILT_PROVENANCE, provenance_bytes),
        ]),
        source_commit: ny_commit,
    }))
}

fn trust_gate_receipt_payload(manifest: &BTreeMap<String, String>) -> Result<String> {
    Ok(format!(
        "schema={TRUST_GATE_RECEIPT_SCHEMA}\ntrust_commit={}\ntrustc_sha256={}\ntrustc_version_sha256={}\ntrust_gate_commands_sha256={}\ntrust_gate_log_sha256={}\nstatus=passed\n",
        manifest_value(manifest, "trust_commit")?,
        manifest_value(manifest, "trustc_sha256")?,
        manifest_value(manifest, "trustc_version_sha256")?,
        manifest_value(manifest, "trust_gate_commands_sha256")?,
        manifest_value(manifest, "trust_gate_log_sha256")?,
    ))
}

fn expected_build_provenance(manifest: &BTreeMap<String, String>) -> Result<String> {
    Ok(format!(
        "{BUILD_PROVENANCE_PREFIX}status=sealed|target={PREBUILT_TARGET}|features={PREBUILT_FEATURES}|ny_commit={}|cargo_lock_sha256={}|ay_commit={}|builder_script_sha256={}|trust_commit={}|trustc_sha256={}|trustc_version_sha256={}|trust_gate_receipt_sha256={}|onnxruntime_static_sha256={}|",
        manifest_value(manifest, "ny_commit")?,
        manifest_value(manifest, "cargo_lock_sha256")?,
        manifest_value(manifest, "ay_lock_commit")?,
        manifest_value(manifest, "builder_script_sha256")?,
        manifest_value(manifest, "trust_commit")?,
        manifest_value(manifest, "trustc_sha256")?,
        manifest_value(manifest, "trustc_version_sha256")?,
        manifest_value(manifest, "trust_gate_receipt_sha256")?,
        manifest_value(manifest, "onnxruntime_static_sha256")?,
    ))
}

fn validate_embedded_build_provenance(path: &Path, expected: &str) -> Result<()> {
    let bytes = fs::read(path)?;
    if !is_canonical_build_provenance_record(expected.as_bytes()) {
        bail!("internal error: expected NY/Trust build provenance is not canonical");
    }
    let canonical_records: Vec<_> = bytes
        .windows(BUILD_PROVENANCE_PREFIX.len())
        .enumerate()
        .filter_map(|(start, prefix)| {
            if prefix != BUILD_PROVENANCE_PREFIX.as_bytes() {
                return None;
            }
            let end = start.checked_add(expected.len())?;
            let record = bytes.get(start..end)?;
            is_canonical_build_provenance_record(record).then_some(record)
        })
        .collect();
    let exact_count = canonical_records
        .iter()
        .filter(|record| **record == expected.as_bytes())
        .count();
    if canonical_records.len() != 1 || exact_count != 1 {
        bail!(
            "prebuilt embedded NY/Trust build provenance mismatch: expected one exact sealed record, found canonical={}, exact={exact_count}",
            canonical_records.len()
        );
    }
    Ok(())
}

fn is_canonical_build_provenance_record(record: &[u8]) -> bool {
    fn take_literal(record: &[u8], cursor: &mut usize, literal: &str) -> bool {
        let Some(end) = cursor.checked_add(literal.len()) else {
            return false;
        };
        if record.get(*cursor..end) != Some(literal.as_bytes()) {
            return false;
        }
        *cursor = end;
        true
    }

    fn take_lower_hex_field(record: &[u8], cursor: &mut usize, label: &str, length: usize) -> bool {
        if !take_literal(record, cursor, label) {
            return false;
        }
        let Some(end) = cursor.checked_add(length) else {
            return false;
        };
        let Some(value) = record.get(*cursor..end) else {
            return false;
        };
        if !value
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
        {
            return false;
        }
        *cursor = end;
        take_literal(record, cursor, "|")
    }

    let mut cursor = 0;
    take_literal(record, &mut cursor, BUILD_PROVENANCE_PREFIX)
        && take_literal(record, &mut cursor, "status=sealed|")
        && take_literal(record, &mut cursor, "target=x86_64-unknown-linux-gnu|")
        && take_literal(record, &mut cursor, "features=mip,cuda|")
        && take_lower_hex_field(record, &mut cursor, "ny_commit=", 40)
        && take_lower_hex_field(record, &mut cursor, "cargo_lock_sha256=", 64)
        && take_lower_hex_field(record, &mut cursor, "ay_commit=", 40)
        && take_lower_hex_field(record, &mut cursor, "builder_script_sha256=", 64)
        && take_lower_hex_field(record, &mut cursor, "trust_commit=", 40)
        && take_lower_hex_field(record, &mut cursor, "trustc_sha256=", 64)
        && take_lower_hex_field(record, &mut cursor, "trustc_version_sha256=", 64)
        && take_lower_hex_field(record, &mut cursor, "trust_gate_receipt_sha256=", 64)
        && take_lower_hex_field(record, &mut cursor, "onnxruntime_static_sha256=", 64)
        && cursor == record.len()
}

fn validate_prebuilt_parent(repo_root: &Path) -> Result<()> {
    let canonical_root = fs::canonicalize(repo_root)?;
    let mut current = repo_root.to_path_buf();
    for component in ["dist", "bin"] {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
                    bail!(
                        "prebuilt package parent must be a real directory, not a symlink: {}",
                        current.display()
                    );
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        }
    }
    let canonical_parent = fs::canonicalize(&current)?;
    if canonical_parent != canonical_root.join("dist/bin") {
        bail!(
            "prebuilt package parent escapes the NY checkout: {}",
            current.display()
        );
    }
    Ok(())
}

fn parse_prebuilt_manifest(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let contents = std::str::from_utf8(bytes)?;
    let allowed: BTreeSet<_> = PREBUILT_MANIFEST_KEYS.into_iter().collect();
    let mut values = BTreeMap::new();
    for (index, line) in contents.lines().enumerate() {
        if line.is_empty() {
            bail!("empty line in prebuilt provenance at line {}", index + 1);
        }
        let Some((key, value)) = line.split_once('=') else {
            bail!("malformed prebuilt provenance at line {}", index + 1);
        };
        if !allowed.contains(key) {
            bail!("unknown prebuilt provenance key {key:?}");
        }
        if value.is_empty() || value.trim() != value {
            bail!("invalid value for prebuilt provenance key {key:?}");
        }
        if values.insert(key.to_string(), value.to_string()).is_some() {
            bail!("duplicate prebuilt provenance key {key:?}");
        }
    }
    let actual: BTreeSet<_> = values.keys().map(String::as_str).collect();
    if actual != allowed {
        let missing: Vec<_> = allowed.difference(&actual).copied().collect();
        bail!("prebuilt provenance is missing required keys: {missing:?}");
    }
    Ok(values)
}

fn manifest_value<'a>(manifest: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    manifest
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| anyhow!("prebuilt provenance is missing {key:?}"))
}

fn require_manifest_value(
    manifest: &BTreeMap<String, String>,
    key: &str,
    expected: &str,
) -> Result<()> {
    let actual = manifest_value(manifest, key)?;
    if actual != expected {
        bail!("prebuilt provenance {key} mismatch: expected {expected}, found {actual}");
    }
    Ok(())
}

fn require_lower_hex(value: &str, length: usize, label: &str) -> Result<()> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} must be exactly {length} lowercase hexadecimal characters");
    }
    Ok(())
}

/// Bind a sealed prebuilt to source commit `source_commit` without requiring a
/// commit hash to contain itself.
///
/// A locally packaged ignored triplet is valid directly at the clean source
/// commit. A Git-clone release may instead point at a descendant commit, but
/// its complete tree delta from the sealed source must be exactly the three
/// prebuilt members. Thus the artifact commit can carry its own bytes while
/// every compiled/package input remains byte-identical to the reviewed source.
fn ensure_prebuilt_source_binding(repo_root: &Path, source_commit: &str) -> Result<()> {
    require_lower_hex(source_commit, 40, "prebuilt source commit")?;
    let status = Command::new("git")
        .args(["diff", "--quiet", "HEAD", "--"])
        .current_dir(repo_root)
        .status()?;
    match status.code() {
        Some(0) => {}
        Some(1) => {
            bail!("refusing a commit-labelled submission while tracked package inputs are dirty")
        }
        _ => bail!("git diff failed while validating packaged source state: {status}"),
    }

    // Reviewed-untracked inputs are useful for source-only development
    // packages, but can never match a binary sealed to a Git commit.
    let untracked = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard", "-z", "--"])
        .args(PACKAGE_ROOTS)
        .current_dir(repo_root)
        .output()?;
    if !untracked.status.success() {
        bail!(
            "git ls-files failed while validating prebuilt source binding ({})",
            untracked.status
        );
    }
    let untracked = String::from_utf8(untracked.stdout)?;
    let untracked: Vec<_> = untracked
        .split('\0')
        .filter(|path| !path.is_empty())
        .collect();
    if !untracked.is_empty() {
        bail!(
            "refusing prebuilt while package inputs are untracked: {}",
            untracked.join(", ")
        );
    }

    let head = git_stdout(repo_root, &["rev-parse", "--verify", "HEAD^{commit}"])?;
    require_lower_hex(&head, 40, "current NY commit")?;
    if head == source_commit {
        return Ok(());
    }

    let source_object = format!("{source_commit}^{{commit}}");
    let source_exists = Command::new("git")
        .args(["cat-file", "-e", &source_object])
        .current_dir(repo_root)
        .status()?;
    if !source_exists.success() {
        bail!("prebuilt source commit {source_commit} is unavailable from release HEAD {head}");
    }
    let ancestry = Command::new("git")
        .args(["merge-base", "--is-ancestor", source_commit, &head])
        .current_dir(repo_root)
        .status()?;
    if !ancestry.success() {
        bail!("prebuilt source commit {source_commit} is not an ancestor of release HEAD {head}");
    }

    let changed = Command::new("git")
        .args(["diff", "--name-only", "-z", source_commit, &head, "--"])
        .current_dir(repo_root)
        .output()?;
    if !changed.status.success() {
        bail!(
            "git diff failed while validating artifact-only release commit ({})",
            changed.status
        );
    }
    let changed = String::from_utf8(changed.stdout)?;
    let changed: BTreeSet<_> = changed
        .split('\0')
        .filter(|path| !path.is_empty())
        .collect();
    let expected: BTreeSet<_> = PREBUILT_FILES.into_iter().collect();
    if changed != expected {
        bail!(
            "release HEAD differs from sealed source outside the exact prebuilt triplet: changed={changed:?}, expected={expected:?}"
        );
    }
    Ok(())
}

fn ensure_packaged_source_clean(repo_root: &Path) -> Result<()> {
    let status = Command::new("git")
        .args(["diff", "--quiet", "HEAD", "--"])
        .current_dir(repo_root)
        .status()?;
    match status.code() {
        Some(0) => {}
        Some(1) => {
            bail!("refusing a commit-labelled submission while tracked package inputs are dirty")
        }
        _ => bail!("git diff failed while validating packaged source state: {status}"),
    }

    let untracked = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard", "-z", "--"])
        .args(PACKAGE_ROOTS)
        .current_dir(repo_root)
        .output()?;
    if !untracked.status.success() {
        bail!(
            "git ls-files failed while validating commit-labelled package inputs ({})",
            untracked.status
        );
    }
    let untracked = String::from_utf8(untracked.stdout)?;
    let untracked: Vec<_> = untracked
        .split('\0')
        .filter(|path| !path.is_empty())
        .collect();
    if !untracked.is_empty() {
        bail!(
            "refusing a commit-labelled submission while package inputs are untracked: {}",
            untracked.join(", ")
        );
    }
    Ok(())
}

fn git_stdout(repo_root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()?;
    if !output.status.success() {
        bail!("git {args:?} failed with status {}", output.status);
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn exact_ay_lock_commit(lock_bytes: &[u8]) -> Result<String> {
    const MARKER: &str = "alabsystems/ay";
    const PREFIX: &str = "source = \"git+https://github.com/alabsystems/ay.git?rev=";
    let lock = std::str::from_utf8(lock_bytes)?;
    let mut pins = BTreeSet::new();
    let mut source_count = 0_usize;
    for line in lock.lines() {
        if !line.to_ascii_lowercase().contains(MARKER) {
            continue;
        }
        source_count += 1;
        let Some(source) = line
            .strip_prefix(PREFIX)
            .and_then(|source| source.strip_suffix('"'))
        else {
            bail!("non-canonical AY Cargo.lock source: {line}");
        };
        let Some((requested, resolved)) = source.split_once('#') else {
            bail!("AY Cargo.lock source is not revision-resolved: {line}");
        };
        if resolved.contains('#') {
            bail!("non-canonical AY Cargo.lock source: {line}");
        }
        require_lower_hex(requested, 40, "AY requested revision")?;
        require_lower_hex(resolved, 40, "AY resolved revision")?;
        if requested != resolved {
            bail!("AY requested revision {requested} resolved to {resolved}");
        }
        pins.insert(resolved.to_string());
    }
    if source_count == 0 {
        bail!("Cargo.lock contains no AY source entries");
    }
    if pins.len() != 1 {
        bail!("expected exactly one AY commit across Cargo.lock, found {pins:?}");
    }
    Ok(pins.pop_first().expect("one AY pin checked above"))
}

fn read_bounded_file(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>> {
    let mut file = fs::File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        bail!("{label} must be a regular file: {}", path.display());
    }
    if metadata.len() > limit {
        bail!(
            "{label} is too large ({} bytes; limit {limit}): {}",
            metadata.len(),
            path.display()
        );
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        bail!(
            "{label} grew beyond its size limit while reading: {}",
            path.display()
        );
    }
    Ok(bytes)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_x86_64_elf(path: &Path) -> Result<()> {
    let mut file = fs::File::open(path)?;
    let mut header = [0_u8; 20];
    file.read_exact(&mut header)?;
    if &header[..4] != b"\x7fELF"
        || header[4] != 2
        || header[5] != 1
        || u16::from_le_bytes([header[18], header[19]]) != 62
    {
        bail!(
            "prebuilt is not an ELF64 little-endian x86_64 binary: {}",
            path.display()
        );
    }
    Ok(())
}

fn embedded_ay_build_commits(path: &Path) -> Result<Vec<String>> {
    const PREFIX: &[u8] = b"build.commit=";
    let bytes = fs::read(path)?;
    let mut commits = Vec::new();
    let mut cursor = 0;
    while cursor + PREFIX.len() <= bytes.len() {
        let Some(relative) = bytes[cursor..]
            .windows(PREFIX.len())
            .position(|window| window == PREFIX)
        else {
            break;
        };
        let start = cursor + relative + PREFIX.len();
        let end = start.saturating_add(40);
        if end <= bytes.len()
            && bytes[start..end]
                .iter()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
        {
            let mut commit = String::from_utf8(bytes[start..end].to_vec())?;
            if bytes.get(end..end + 6) == Some(b"-dirty") {
                commit.push_str("-dirty");
            }
            commits.push(commit);
        }
        cursor = start;
    }
    commits.sort();
    commits.dedup();
    Ok(commits)
}

fn run_build(repo_root: &Path) -> Result<()> {
    let status = Command::new(repo_root.join("vnncomp_scripts/build_submission_binary.sh"))
        .current_dir(repo_root)
        .status()?;
    if !status.success() {
        bail!("submission binary build failed with status {status}");
    }
    Ok(())
}

fn packaged_source_provenance(repo_root: &Path) -> Result<Vec<u8>> {
    ensure_packaged_source_clean(repo_root)?;
    let ny_commit = git_stdout(repo_root, &["rev-parse", "--verify", "HEAD"])?;
    require_lower_hex(&ny_commit, 40, "packaged NY commit")?;
    let cargo_lock_sha256 = sha256_file(&repo_root.join("Cargo.lock"))?;
    Ok(format!(
        "schema={SOURCE_PROVENANCE_SCHEMA}\n\
         ny_commit={ny_commit}\n\
         cargo_lock_sha256={cargo_lock_sha256}\n"
    )
    .into_bytes())
}

/// The `gzip` executable to invoke.
///
/// On Linux — the host that builds the scored artifact — this is the bare name
/// and resolution is PATH's job, exactly as before. Nothing about the packaging
/// path changes there.
///
/// Windows has no system `gzip`. It ships with Git for Windows, whose
/// `usr/bin` is deliberately kept off PATH by the standard installer, so
/// packaging failed on a developer box that had the tool installed all along.
/// PATH still WINS when it resolves — an explicitly chosen gzip must not be
/// overridden — and these fallbacks are consulted only when it does not.
#[cfg(windows)]
fn gzip_program() -> std::ffi::OsString {
    if path_has_executable("gzip.exe") {
        return "gzip".into();
    }
    ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"]
        .iter()
        .filter_map(std::env::var_os)
        .flat_map(|root| {
            let root = PathBuf::from(root);
            [
                root.join(r"Git\usr\bin\gzip.exe"),
                root.join(r"Programs\Git\usr\bin\gzip.exe"),
            ]
        })
        .find(|candidate| candidate.is_file())
        .map_or_else(|| "gzip".into(), PathBuf::into_os_string)
}

/// Whether PATH already resolves `name`, so an explicit choice is respected.
#[cfg(windows)]
fn path_has_executable(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| directory.join(name).is_file())
    })
}

#[cfg(not(windows))]
fn gzip_program() -> std::ffi::OsString {
    "gzip".into()
}

/// A path an external tool can actually open.
///
/// `fs::canonicalize` returns a VERBATIM path on Windows — `\\?\C:\...` — and
/// the MSYS-built utilities this function shells out to do not understand that
/// prefix. `gzip` from Git for Windows mangles it into `\?C:Users...` and then
/// reports "No such file or directory", which is what made submission
/// packaging unrunnable there. (Windows' own `tar.exe` is bsdtar and accepts
/// the prefix, so only some of these calls ever failed.)
///
/// Stripped for ARGUMENT use only: the canonical form is still what the
/// overwrite-safety comparison above comes from, so that check is unchanged.
/// The prefix exists to exceed `MAX_PATH`, and dropping it reintroduces that
/// limit — acceptable here because these are short temp paths beside the
/// output, and because the scored artifact is packaged on Linux, where this is
/// the identity function.
#[cfg(windows)]
fn external_tool_path(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    text.strip_prefix(r"\\?\")
        .map_or_else(|| path.to_path_buf(), PathBuf::from)
}

#[cfg(not(windows))]
fn external_tool_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

fn write_tarball(repo_root: &Path, output: &Path, included_paths: &[String]) -> Result<()> {
    let output_parent = output
        .parent()
        .ok_or_else(|| anyhow!("submission output has no parent: {}", output.display()))?;
    fs::create_dir_all(output_parent)?;
    let canonical_output_parent = fs::canonicalize(output_parent)?;
    let output_name = output
        .file_name()
        .ok_or_else(|| anyhow!("submission output has no file name: {}", output.display()))?;
    let canonical_output = canonical_output_parent.join(output_name);
    for input in included_paths.iter().map(|path| repo_root.join(path)) {
        if fs::canonicalize(&input)? == canonical_output {
            bail!(
                "submission output must not overwrite a packaged input: {}",
                output.display()
            );
        }
    }

    let includes_prebuilt = included_paths.iter().any(|path| path == PREBUILT_ARCHIVE);
    let validated_prebuilt =
        if includes_prebuilt {
            Some(validate_optional_prebuilt(repo_root)?.ok_or_else(|| {
                anyhow!("validated prebuilt disappeared before submission archiving")
            })?)
        } else {
            None
        };
    // A tarball has no .git directory. Inject the exact commit/lock identity so
    // source fallback builds can issue the same runtime receipt as checkout
    // builds, rather than relabelling archive bytes as an unknowable revision.
    // Requiring a clean tracked and untracked package tree keeps that commit
    // claim exact.
    let source_provenance = packaged_source_provenance(repo_root)?;
    let source_snapshot = tempfile::tempdir()?;
    fs::write(
        source_snapshot.path().join(SOURCE_PROVENANCE),
        &source_provenance,
    )?;

    // Keep the repository capture in one archive root for BSD and GNU tar
    // alike. BSD tar does not honor GNU tar's positional `-C` semantics around
    // an earlier `-T`, so append the generated source marker to an uncompressed
    // tar in a second pass, then gzip the completed stream.
    let mut list_file = tempfile::NamedTempFile::new()?;
    for path in included_paths {
        writeln!(list_file, "{path}")?;
    }

    let raw_tar = tempfile::NamedTempFile::new_in(&canonical_output_parent)?;
    let staged_output = tempfile::NamedTempFile::new_in(&canonical_output_parent)?;

    // Excludes must precede -T on both BSD (macOS) and GNU tar.
    let mut command = Command::new("tar");
    command
        .arg("-cf")
        .arg(external_tool_path(raw_tar.path()))
        .arg("-C")
        .arg(repo_root);
    for glob in EXCLUDE_GLOBS {
        command.arg(format!("--exclude={glob}"));
    }
    command.arg("-T").arg(list_file.path());

    let status = command.status()?;
    if !status.success() {
        bail!("tar failed with status {status}");
    }
    let append_status = Command::new("tar")
        .arg("-rf")
        .arg(external_tool_path(raw_tar.path()))
        .arg("-C")
        .arg(source_snapshot.path())
        .arg(SOURCE_PROVENANCE)
        .status()?;
    if !append_status.success() {
        bail!("tar failed while appending source provenance: {append_status}");
    }
    let gzip_output = staged_output.reopen()?;
    // Name the tool on failure. A bare `?` here surfaced only "program not
    // found", which says nothing about WHICH of the several programs this
    // function shells out to (tar, gzip, git, xz) is missing — on a host
    // without gzip that error cost a full bisect to attribute.
    //
    // stderr is CAPTURED rather than inherited so a non-zero exit reports what
    // gzip actually said; `.status()` discarded it and left only the code.
    let gzip_result = Command::new(gzip_program())
        .arg("-c")
        .arg(external_tool_path(raw_tar.path()))
        .stdout(Stdio::from(gzip_output))
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| anyhow!("run `gzip` to compress the submission tarball: {error}"))?
        .wait_with_output()
        .map_err(|error| anyhow!("wait for `gzip` to compress the submission tarball: {error}"))?;
    if !gzip_result.status.success() {
        bail!(
            "gzip failed with status {}: {}",
            gzip_result.status,
            String::from_utf8_lossy(&gzip_result.stderr).trim()
        );
    }

    // Backstop for externally vendored sources: a stray exclude glob (or any
    // other packaging path) that omits a checksummed file breaks Cargo's
    // directory-source build. Verify the ACTUAL archived bytes here.
    verify_vendor_checksums(staged_output.path())?;

    let archived_source = Command::new("tar")
        .args(["-xOzf"])
        .arg(staged_output.path())
        .arg(SOURCE_PROVENANCE)
        .output()?;
    if !archived_source.status.success() || archived_source.stdout != source_provenance {
        bail!("captured archive source provenance is missing or changed");
    }
    if let Some(prebuilt) = &validated_prebuilt {
        for relative in PREBUILT_FILES {
            let archived = Command::new("tar")
                .args(["-xOzf"])
                .arg(staged_output.path())
                .arg(relative)
                .output()?;
            if !archived.status.success() {
                bail!(
                    "could not read captured prebuilt member {relative:?}: {}",
                    String::from_utf8_lossy(&archived.stderr).trim()
                );
            }
            if archived.stdout != prebuilt.bytes(relative) {
                bail!("captured tar member changed during archiving: {relative}");
            }
        }
        ensure_prebuilt_source_binding(repo_root, &prebuilt.source_commit)?;
    }
    let final_source_provenance = packaged_source_provenance(repo_root)?;
    if final_source_provenance != source_provenance {
        bail!("NY commit or Cargo.lock changed during submission archiving");
    }
    staged_output
        .persist(&canonical_output)
        .map_err(|error| error.error)?;
    Ok(())
}

/// Assert that every file listed in an externally vendored crate's
/// `.cargo-checksum.json` is actually present in the produced archive. Cargo's
/// directory-source integrity
/// check recomputes those hashes at build time and aborts the entire build if a
/// listed file is missing ("failed to calculate checksum of … No such file or
/// directory"). Because the source-build submission has no prebuilt fallback on
/// the evaluator, a single dropped vendored file turns the whole run into a
/// zero-scoring non-build. We verify the archived bytes (not the working tree) so
/// any packaging path that drops a checksummed vendored file fails the pack here.
fn verify_vendor_checksums(archive: &Path) -> Result<()> {
    let listing = Command::new("tar").arg("-tzf").arg(archive).output()?;
    if !listing.status.success() {
        bail!(
            "could not list submission archive for checksum verification: {}",
            String::from_utf8_lossy(&listing.stderr).trim()
        );
    }
    // tar may emit "./"-prefixed and/or trailing-slash directory entries; the
    // checksum manifests use bare crate-relative paths, so normalize to match.
    let normalize = |raw: &str| {
        raw.strip_prefix("./")
            .unwrap_or(raw)
            .trim_end_matches('/')
            .to_string()
    };
    let entries: BTreeSet<String> = String::from_utf8_lossy(&listing.stdout)
        .lines()
        .map(normalize)
        .collect();

    // (crate_dir, manifest_path) for every vendored .cargo-checksum.json.
    let manifests: Vec<(String, String)> = entries
        .iter()
        .filter_map(|entry| {
            entry
                .strip_suffix(".cargo-checksum.json")
                .and_then(|prefix| prefix.strip_suffix('/'))
                .filter(|dir| dir.starts_with("vendor/"))
                .map(|dir| (dir.to_string(), entry.clone()))
        })
        .collect();
    if manifests.is_empty() {
        return Ok(());
    }

    // One extra decompression pass: pull just the manifests out to a scratch dir.
    let scratch = tempfile::tempdir()?;
    let mut extract = Command::new("tar");
    extract
        .args(["-xzf"])
        .arg(archive)
        .arg("-C")
        .arg(scratch.path());
    for (_, manifest_path) in &manifests {
        extract.arg(manifest_path);
    }
    if !extract.status()?.success() {
        bail!("could not extract vendored checksum manifests for verification");
    }

    let mut missing: Vec<String> = Vec::new();
    for (crate_dir, manifest_path) in &manifests {
        let bytes = fs::read(scratch.path().join(manifest_path))?;
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            anyhow!("malformed vendored checksum manifest {manifest_path}: {error}")
        })?;
        let files = parsed
            .get("files")
            .and_then(|files| files.as_object())
            .ok_or_else(|| {
                anyhow!("vendored checksum manifest {manifest_path} has no files map")
            })?;
        for relative in files.keys() {
            let expected = format!("{crate_dir}/{relative}");
            if !entries.contains(&expected) {
                missing.push(expected);
            }
        }
    }

    if !missing.is_empty() {
        missing.sort();
        missing.dedup();
        bail!(
            "submission archive drops {} vendored file(s) still listed in a \
             .cargo-checksum.json — cargo's offline directory-source build would \
             abort. Check EXCLUDE_GLOBS scoping. Missing: {}",
            missing.len(),
            missing.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_AY_COMMIT: &str = "805725e47d734b0c72ea8c089bdf6245c1cbee16";
    const STALE_AY_COMMIT: &str = "1560972ade2b04a702dfbd13a2de5444ea216009";

    fn run_git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("git UTF-8")
            .trim()
            .to_string()
    }

    fn write_executable(path: &Path, contents: &str) {
        fs::write(path, contents).expect("write executable fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(path).expect("fixture metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).expect("chmod fixture");
        }
    }

    /// The pack-time backstop must reject an archive that drops a file still
    /// listed in an external vendor crate's `.cargo-checksum.json`, and must
    /// accept an intact external vendor tree.
    #[test]
    fn verify_vendor_checksums_flags_dropped_vendored_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("root");
        let crate_dir = root.join("vendor/external-demo");
        fs::create_dir_all(crate_dir.join("proptest-regressions")).expect("mk crate dir");
        fs::write(
            crate_dir.join("Cargo.toml"),
            "[package]\nname = \"external-demo\"\n",
        )
        .expect("toml");
        // The seed file exists on disk (as in the real working tree) — the bug is
        // that a glob drops it from the ARCHIVE while the manifest still lists it.
        fs::write(crate_dir.join("proptest-regressions/lib.txt"), "seed").expect("seed");
        fs::write(
            crate_dir.join(".cargo-checksum.json"),
            r#"{"files":{"Cargo.toml":"aa","proptest-regressions/lib.txt":"bb"},"package":null}"#,
        )
        .expect("manifest");

        let build = |archive: &Path, exclude: Option<&str>| {
            let mut command = Command::new("tar");
            command.arg("-czf").arg(archive).arg("-C").arg(&root);
            if let Some(glob) = exclude {
                command.arg(format!("--exclude={glob}"));
            }
            command.arg("vendor");
            assert!(command.status().expect("run tar").success(), "tar failed");
        };

        // Unscoped glob drops the listed seed file => the check must bail, naming it.
        let broken = dir.path().join("broken.tar.gz");
        build(&broken, Some("*/proptest-regressions"));
        let error = verify_vendor_checksums(&broken)
            .expect_err("dropped vendored file must fail the pack")
            .to_string();
        assert!(
            error.contains("vendor/external-demo/proptest-regressions/lib.txt"),
            "error should name the missing file, got: {error}"
        );

        // Intact vendored tree must pass.
        let intact = dir.path().join("intact.tar.gz");
        build(&intact, None);
        verify_vendor_checksums(&intact).expect("intact vendored tree must pass");
    }

    fn fixture_manifest(root: &Path, ny_commit: &str, binary_sha256: &str) -> String {
        let lock_sha256 = sha256_file(&root.join("Cargo.lock")).expect("lock hash");
        let builder_sha256 =
            sha256_file(&root.join("scripts/vnncomp_trust_linux_build.sh")).expect("builder hash");
        let package_sha256 = sha256_file(&root.join(PREBUILT_ARCHIVE)).expect("archive hash");
        let receipt_sha256 = fixture_receipt_sha256();
        [
            format!("schema={PREBUILT_SCHEMA}"),
            format!("target={PREBUILT_TARGET}"),
            format!("features={PREBUILT_FEATURES}"),
            format!("trust_commit={}", "a".repeat(40)),
            "trust_bootstrap_mode=seed".to_string(),
            "trust_gate_status=passed".to_string(),
            format!("trust_gate_receipt_sha256={receipt_sha256}"),
            format!(
                "trust_gate_commands_sha256={}",
                sha256_bytes(TRUST_GATE_COMMANDS_V1.as_bytes())
            ),
            format!("trust_gate_log_sha256={}", "b".repeat(64)),
            format!("trustc_sha256={}", "c".repeat(64)),
            format!("trustc_version_sha256={}", "d".repeat(64)),
            format!("ny_commit={ny_commit}"),
            format!("cargo_lock_sha256={lock_sha256}"),
            format!("ay_lock_commit={TEST_AY_COMMIT}"),
            format!("builder_script_sha256={builder_sha256}"),
            format!("onnxruntime_static_sha256={}", "e".repeat(64)),
            format!("binary_sha256={binary_sha256}"),
            format!("package_sha256={package_sha256}"),
        ]
        .join("\n")
            + "\n"
    }

    fn fixture_receipt_sha256() -> String {
        let mut values = BTreeMap::new();
        values.insert("trust_commit".to_string(), "a".repeat(40));
        values.insert("trustc_sha256".to_string(), "c".repeat(64));
        values.insert("trustc_version_sha256".to_string(), "d".repeat(64));
        values.insert(
            "trust_gate_commands_sha256".to_string(),
            sha256_bytes(TRUST_GATE_COMMANDS_V1.as_bytes()),
        );
        values.insert("trust_gate_log_sha256".to_string(), "b".repeat(64));
        sha256_bytes(
            trust_gate_receipt_payload(&values)
                .expect("fixture receipt")
                .as_bytes(),
        )
    }

    fn make_prebuilt_fixture(embedded_ay: &str, elf_machine: u16) -> tempfile::TempDir {
        make_prebuilt_fixture_with_seal(embedded_ay, elf_machine, None)
    }

    fn make_prebuilt_fixture_with_seal(
        embedded_ay: &str,
        elf_machine: u16,
        seal_override: Option<(&str, &str)>,
    ) -> tempfile::TempDir {
        let temp = tempfile::tempdir().expect("temp repo");
        let root = temp.path();
        assert!(Command::new("git")
            .args(["init", "-q"])
            .current_dir(root)
            .status()
            .expect("git init")
            .success());

        fs::create_dir_all(root.join(".cargo")).expect("cargo dir");
        fs::create_dir_all(root.join("crates/ny-mip")).expect("crate dir");
        fs::create_dir_all(root.join("scripts")).expect("scripts dir");
        fs::create_dir_all(root.join("vnncomp_scripts")).expect("harness dir");
        fs::write(root.join(".gitignore"), "/dist/\n").expect("gitignore");
        fs::write(root.join("Cargo.toml"), "[workspace]\nmembers = []\n").expect("manifest");
        fs::write(
            root.join("Cargo.lock"),
            format!(
                "version = 4\n\n[[package]]\nname = \"ay-milp\"\nversion = \"0.11.0\"\nsource = \"git+https://github.com/alabsystems/ay.git?rev={TEST_AY_COMMIT}#{TEST_AY_COMMIT}\"\n"
            ),
        )
        .expect("lock");
        fs::write(
            root.join("rust-toolchain.toml"),
            "[toolchain]\nchannel = \"1.95.0\"\n",
        )
        .expect("toolchain");
        fs::write(root.join(".cargo/config.toml"), "[net]\noffline = true\n")
            .expect("cargo config");
        fs::write(root.join("crates/ny-mip/placeholder"), "source\n").expect("source");
        write_executable(
            &root.join("scripts/vnncomp_trust_linux_build.sh"),
            "#!/bin/bash\nexit 0\n",
        );
        for relative in REQUIRED_SCRIPTS {
            write_executable(&root.join(relative), "#!/bin/bash\nexit 0\n");
        }
        for relative in [
            "vnncomp_scripts/build_submission_binary.sh",
            "vnncomp_scripts/prepare_instance.sh",
            "vnncomp_scripts/run_instance.sh",
        ] {
            write_executable(&root.join(relative), "#!/bin/bash\nexit 0\n");
        }
        fs::write(
            root.join(PREBUILT_VERIFIER),
            "#!/usr/bin/env python3\nraise SystemExit(0)\n",
        )
        .expect("prebuilt verifier");
        write_executable(
            &root.join("vnncomp_scripts/submission_binary_receipt.sh"),
            "#!/bin/bash\nexit 0\n",
        );
        run_git(root, &["add", "."]);
        run_git(
            root,
            &[
                "-c",
                "user.name=NY Test",
                "-c",
                "user.email=ny@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
        );
        let head = run_git(root, &["rev-parse", "HEAD"]);

        let mut seal_values = BTreeMap::new();
        seal_values.insert("ny_commit".to_string(), head.clone());
        seal_values.insert(
            "cargo_lock_sha256".to_string(),
            sha256_file(&root.join("Cargo.lock")).expect("lock hash"),
        );
        seal_values.insert("ay_lock_commit".to_string(), TEST_AY_COMMIT.to_string());
        seal_values.insert(
            "builder_script_sha256".to_string(),
            sha256_file(&root.join("scripts/vnncomp_trust_linux_build.sh")).expect("builder hash"),
        );
        seal_values.insert("trust_commit".to_string(), "a".repeat(40));
        seal_values.insert("trustc_sha256".to_string(), "c".repeat(64));
        seal_values.insert("trustc_version_sha256".to_string(), "d".repeat(64));
        seal_values.insert(
            "trust_gate_receipt_sha256".to_string(),
            fixture_receipt_sha256(),
        );
        seal_values.insert("onnxruntime_static_sha256".to_string(), "e".repeat(64));
        if let Some((key, value)) = seal_override {
            seal_values.insert(key.to_string(), value.to_string());
        }
        let build_provenance =
            expected_build_provenance(&seal_values).expect("fixture build provenance");

        fs::create_dir_all(root.join("dist/bin")).expect("dist dir");
        let raw_path = root.join("dist/bin/fixture-ny");
        let mut raw = vec![0_u8; 64];
        raw[..4].copy_from_slice(b"\x7fELF");
        raw[4] = 2;
        raw[5] = 1;
        raw[6] = 1;
        raw[18..20].copy_from_slice(&elf_machine.to_le_bytes());
        raw.extend_from_slice(format!(" build.commit={embedded_ay} ").as_bytes());
        // The real ny executable contains bare copies of the validator's scan
        // prefix. They are not release seals unless immediately followed by
        // the sealed-status field and must not make a valid binary ambiguous.
        raw.extend_from_slice(
            format!(" validator-needle:{BUILD_PROVENANCE_PREFIX}status=sealed|compiler-literal ")
                .as_bytes(),
        );
        raw.extend_from_slice(format!(" {build_provenance} ").as_bytes());
        fs::write(&raw_path, &raw).expect("raw fixture");
        let compressed = Command::new("xz")
            .args(["-c", "--"])
            .arg(&raw_path)
            .output()
            .expect("compress fixture");
        assert!(compressed.status.success());
        fs::write(root.join(PREBUILT_ARCHIVE), compressed.stdout).expect("archive");
        fs::remove_file(&raw_path).expect("remove raw fixture");
        let archive_sha256 = sha256_file(&root.join(PREBUILT_ARCHIVE)).expect("archive hash");
        fs::write(
            root.join(PREBUILT_CHECKSUM),
            format!("{archive_sha256}  ny-x86_64-linux.xz\n"),
        )
        .expect("checksum");
        let binary_sha256 = sha256_bytes(&raw);
        fs::write(
            root.join(PREBUILT_PROVENANCE),
            fixture_manifest(root, &head, &binary_sha256),
        )
        .expect("provenance");
        temp
    }

    fn commit_prebuilt_triplet(root: &Path) -> (String, String) {
        let manifest = parse_prebuilt_manifest(
            &fs::read(root.join(PREBUILT_PROVENANCE)).expect("read provenance"),
        )
        .expect("parse provenance");
        let source_commit = manifest_value(&manifest, "ny_commit")
            .expect("source commit")
            .to_string();
        let mut add = vec!["add", "-f", "--"];
        add.extend(PREBUILT_FILES);
        run_git(root, &add);
        run_git(
            root,
            &[
                "-c",
                "user.name=NY Test",
                "-c",
                "user.email=ny@example.invalid",
                "commit",
                "-q",
                "-m",
                "artifact-only release",
            ],
        );
        let release_commit = run_git(root, &["rev-parse", "HEAD"]);
        (source_commit, release_commit)
    }

    fn replace_manifest_value(root: &Path, key: &str, value: &str) {
        let path = root.join(PREBUILT_PROVENANCE);
        let contents = fs::read_to_string(&path).expect("read provenance");
        let prefix = format!("{key}=");
        let mut found = false;
        let updated = contents
            .lines()
            .map(|line| {
                if line.starts_with(&prefix) {
                    found = true;
                    format!("{prefix}{value}")
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        assert!(found, "manifest key not found: {key}");
        fs::write(path, updated).expect("rewrite provenance");
    }

    fn rewrite_prebuilt_binary(root: &Path, mutate: impl FnOnce(&mut Vec<u8>)) {
        let archive = root.join(PREBUILT_ARCHIVE);
        let decompressed = Command::new("xz")
            .args(["-dc", "--"])
            .arg(&archive)
            .output()
            .expect("decompress fixture");
        assert!(decompressed.status.success());
        let mut binary = decompressed.stdout;
        mutate(&mut binary);

        let mut raw = tempfile::NamedTempFile::new().expect("raw fixture");
        raw.write_all(&binary).expect("write raw fixture");
        raw.flush().expect("flush raw fixture");
        let compressed = Command::new("xz")
            .args(["-c", "--"])
            .arg(raw.path())
            .output()
            .expect("recompress fixture");
        assert!(compressed.status.success());
        fs::write(&archive, compressed.stdout).expect("replace archive");

        let archive_sha256 = sha256_file(&archive).expect("archive hash");
        fs::write(
            root.join(PREBUILT_CHECKSUM),
            format!("{archive_sha256}  ny-x86_64-linux.xz\n"),
        )
        .expect("replace checksum");
        replace_manifest_value(root, "binary_sha256", &sha256_bytes(&binary));
        replace_manifest_value(root, "package_sha256", &archive_sha256);
    }

    #[test]
    fn package_paths_include_required_scripts() {
        for script in REQUIRED_SCRIPTS {
            assert!(PACKAGE_ROOTS.contains(&script));
        }
    }

    #[test]
    fn package_paths_include_toolchain_and_dist_but_not_internal_vendor_sources() {
        assert!(PACKAGE_ROOTS.contains(&"rust-toolchain.toml"));
        assert!(PACKAGE_ROOTS.contains(&".cargo"));
        assert!(!PACKAGE_ROOTS.contains(&"vendor"));
        assert!(PACKAGE_ROOTS.contains(&"dist"));
    }

    #[test]
    fn valid_cross_arch_prebuilt_is_included_as_an_atomic_triplet() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let files = package_file_list(fixture.path()).expect("valid prebuilt");
        for expected in PREBUILT_FILES {
            assert_eq!(
                files
                    .iter()
                    .filter(|path| path.as_str() == expected)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn source_only_commit_label_rejects_reviewed_untracked_release_modules() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        for relative in PREBUILT_FILES {
            fs::remove_file(fixture.path().join(relative)).expect("remove prebuilt member");
        }
        let release_modules: [(&str, &[u8]); 4] = [
            (
                "crates/ny-cli/src/commands/vnncomp_2025_tracks.rs",
                b"pub const TRACKS_2025: &[&str] = &[];\n",
            ),
            (
                "crates/ny-cli/src/commands/vnncomp_2026_tracks.rs",
                b"pub const TRACKS_2026: &[&str] = &[];\n",
            ),
            (
                "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/bounded_shared_executor.rs",
                b"pub(crate) struct BoundedSharedExecutor;\n",
            ),
            (
                "crates/ny-propagate/src/beta_crown/engine/graph/objectives/cuda_beta_spsa.rs",
                b"pub(crate) struct CudaBetaSpsa;\n",
            ),
        ];
        for (relative, contents) in release_modules {
            let path = fixture.path().join(relative);
            fs::create_dir_all(path.parent().expect("module parent"))
                .expect("create module parent");
            fs::write(path, contents).expect("write untracked release module");
        }

        let included = package_file_list(fixture.path()).expect("package reviewed dirty tree");
        for (relative, _) in release_modules {
            assert_eq!(
                included
                    .iter()
                    .filter(|candidate| candidate.as_str() == relative)
                    .count(),
                1,
                "{relative} must appear exactly once in the package list"
            );
        }

        let output = fixture.path().join("submission.tar.gz");
        let error = write_tarball(fixture.path(), &output, &included)
            .expect_err("untracked source cannot be labelled as the current commit")
            .to_string();
        assert!(error.contains("package inputs are untracked"), "{error}");
        for (relative, _) in release_modules {
            assert!(error.contains(relative), "missing untracked path: {error}");
        }
        assert!(
            !output.exists(),
            "failed packaging must not publish an archive"
        );
    }

    #[test]
    fn prebuilt_rejects_even_reviewed_untracked_package_inputs() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let relative = "crates/ny-cli/src/commands/vnncomp_2025_tracks.rs";
        let path = fixture.path().join(relative);
        fs::create_dir_all(path.parent().expect("module parent")).expect("create module parent");
        fs::write(&path, "pub const TRACKS_2025: &[&str] = &[];\n")
            .expect("write reviewed untracked module");

        let error = validate_optional_prebuilt(fixture.path())
            .expect_err("a commit-sealed prebuilt cannot cover untracked source")
            .to_string();
        assert!(error.contains("package inputs are untracked"), "{error}");
        assert!(error.contains(relative), "{error}");
    }

    #[test]
    fn artifact_only_descendant_commit_preserves_prebuilt_source_binding() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let (source_commit, release_commit) = commit_prebuilt_triplet(fixture.path());
        assert_ne!(source_commit, release_commit);

        let validated = validate_optional_prebuilt(fixture.path())
            .expect("artifact-only descendant should validate")
            .expect("prebuilt should be present");
        assert_eq!(validated.source_commit, source_commit);

        let files = package_file_list(fixture.path()).expect("package artifact-only release");
        assert!(files.iter().any(|path| path == PREBUILT_BUILDER));
    }

    #[test]
    fn descendant_commit_with_source_drift_cannot_relabel_prebuilt() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let (source_commit, _) = commit_prebuilt_triplet(fixture.path());
        fs::write(
            fixture.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"changed-after-build\"]\n",
        )
        .expect("change compiled source");
        run_git(fixture.path(), &["add", "Cargo.toml"]);
        run_git(
            fixture.path(),
            &[
                "-c",
                "user.name=NY Test",
                "-c",
                "user.email=ny@example.invalid",
                "commit",
                "-q",
                "-m",
                "forbidden source drift",
            ],
        );

        let error = validate_optional_prebuilt(fixture.path())
            .expect_err("artifact descendant with source drift must fail")
            .to_string();
        assert!(
            error.contains("outside the exact prebuilt triplet"),
            "{error}"
        );
        assert!(error.contains("Cargo.toml"), "{error}");
        assert_ne!(
            run_git(fixture.path(), &["rev-parse", "HEAD"]),
            source_commit
        );
    }

    #[test]
    fn unreviewed_untracked_package_input_fails_closed() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let relative = "crates/ny-cli/src/unreviewed_release_input.rs";
        fs::create_dir_all(
            fixture
                .path()
                .join(relative)
                .parent()
                .expect("input parent"),
        )
        .expect("create input parent");
        fs::write(fixture.path().join(relative), "pub struct Surprise;\n")
            .expect("write unreviewed input");

        let error = package_file_list(fixture.path())
            .expect_err("unreviewed untracked package input must fail")
            .to_string();
        assert!(error.contains("unreviewed untracked file"));
        assert!(error.contains(relative));
    }

    #[test]
    fn completely_absent_prebuilt_keeps_source_fallback() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        for relative in PREBUILT_FILES {
            fs::remove_file(fixture.path().join(relative)).expect("remove fixture member");
        }
        assert!(validate_optional_prebuilt(fixture.path())
            .expect("absence is supported")
            .is_none());
    }

    #[test]
    fn source_only_tarball_carries_exact_commit_and_lock_identity() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        for relative in PREBUILT_FILES {
            fs::remove_file(fixture.path().join(relative)).expect("remove fixture member");
        }
        let included = package_file_list(fixture.path()).expect("source-only package list");
        let output = fixture.path().join("source-only-submission.tar.gz");
        write_tarball(fixture.path(), &output, &included).expect("write source-only tarball");

        let archived = Command::new("tar")
            .args(["-xOzf"])
            .arg(&output)
            .arg(SOURCE_PROVENANCE)
            .output()
            .expect("extract source provenance");
        assert!(archived.status.success());
        let expected = format!(
            "schema={SOURCE_PROVENANCE_SCHEMA}\nny_commit={}\ncargo_lock_sha256={}\n",
            run_git(fixture.path(), &["rev-parse", "HEAD"]),
            sha256_file(&fixture.path().join("Cargo.lock")).expect("lock hash")
        );
        assert_eq!(archived.stdout, expected.as_bytes());
    }

    #[test]
    fn every_nonempty_partial_prebuilt_set_is_rejected() {
        for mask in 1_u8..7 {
            let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
            for (index, relative) in PREBUILT_FILES.iter().enumerate() {
                if mask & (1 << index) == 0 {
                    fs::remove_file(fixture.path().join(relative)).expect("remove member");
                }
            }
            let error = validate_optional_prebuilt(fixture.path()).expect_err("partial must fail");
            assert!(error.to_string().contains("partial VNN-COMP prebuilt"));
        }
    }

    #[test]
    fn dry_run_command_seam_rejects_an_invalid_triplet_without_writing_output() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        fs::remove_file(fixture.path().join(PREBUILT_PROVENANCE)).expect("remove provenance");
        let output = fixture.path().join("submission.tar.gz");
        let error =
            handle_vnncomp_submit_from_repo(fixture.path(), output.clone(), true, true, false)
                .expect_err("dry-run must validate ignored prebuilts");
        assert!(error.to_string().contains("partial VNN-COMP prebuilt"));
        assert!(!output.exists());
    }

    #[test]
    fn stale_or_unproven_manifest_fields_are_rejected() {
        for (key, value) in [
            ("ny_commit", "1".repeat(40)),
            ("cargo_lock_sha256", "2".repeat(64)),
            ("ay_lock_commit", STALE_AY_COMMIT.to_string()),
            ("trust_bootstrap_mode", "genesis".to_string()),
            ("trust_gate_status", "not-run".to_string()),
            ("target", "aarch64-unknown-linux-gnu".to_string()),
            ("features", "cuda".to_string()),
        ] {
            let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
            replace_manifest_value(fixture.path(), key, &value);
            assert!(
                validate_optional_prebuilt(fixture.path()).is_err(),
                "{key} must fail"
            );
        }
    }

    #[test]
    fn relabelled_stale_embedded_ay_commit_is_rejected() {
        let fixture = make_prebuilt_fixture(STALE_AY_COMMIT, 62);
        let error = validate_optional_prebuilt(fixture.path()).expect_err("stale AY must fail");
        assert!(error
            .to_string()
            .contains("embedded AY build.commit mismatch"));
    }

    #[test]
    fn relabelled_ny_trust_builder_receipt_ay_and_ort_seals_are_rejected() {
        for (key, value) in [
            ("ny_commit", "1".repeat(40)),
            ("cargo_lock_sha256", "2".repeat(64)),
            ("ay_lock_commit", STALE_AY_COMMIT.to_string()),
            ("builder_script_sha256", "3".repeat(64)),
            ("trust_commit", "4".repeat(40)),
            ("trustc_sha256", "5".repeat(64)),
            ("trustc_version_sha256", "6".repeat(64)),
            ("trust_gate_receipt_sha256", "7".repeat(64)),
            ("onnxruntime_static_sha256", "8".repeat(64)),
        ] {
            let fixture =
                make_prebuilt_fixture_with_seal(TEST_AY_COMMIT, 62, Some((key, value.as_str())));
            let error = validate_optional_prebuilt(fixture.path())
                .expect_err("relabelled embedded build seal must fail");
            assert!(
                error
                    .to_string()
                    .contains("embedded NY/Trust build provenance mismatch"),
                "unexpected error for {key}: {error:#}"
            );
        }
    }

    #[test]
    fn receipt_field_tampering_is_rejected_before_binary_acceptance() {
        for (key, value) in [
            ("trust_commit", "1".repeat(40)),
            ("trustc_sha256", "2".repeat(64)),
            ("trustc_version_sha256", "3".repeat(64)),
            ("trust_gate_log_sha256", "4".repeat(64)),
        ] {
            let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
            replace_manifest_value(fixture.path(), key, &value);
            let error = validate_optional_prebuilt(fixture.path())
                .expect_err("tampered canonical receipt must fail");
            assert!(
                error.to_string().contains("trust_gate_receipt_sha256"),
                "unexpected error for {key}: {error:#}"
            );
        }
    }

    #[test]
    fn duplicate_exact_build_provenance_seal_is_rejected() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let manifest = parse_prebuilt_manifest(
            &fs::read(fixture.path().join(PREBUILT_PROVENANCE)).expect("read provenance"),
        )
        .expect("parse provenance");
        let exact_seal = expected_build_provenance(&manifest).expect("expected seal");
        rewrite_prebuilt_binary(fixture.path(), move |binary| {
            binary.extend_from_slice(exact_seal.as_bytes());
        });
        let error = validate_optional_prebuilt(fixture.path())
            .expect_err("multiple exact build provenance seals must fail");
        assert!(error.to_string().contains("canonical=2, exact=2"));
    }

    #[test]
    fn stale_canonical_seal_plus_appended_current_seal_is_rejected() {
        let stale_trust_commit = "4".repeat(40);
        let fixture = make_prebuilt_fixture_with_seal(
            TEST_AY_COMMIT,
            62,
            Some(("trust_commit", &stale_trust_commit)),
        );
        let manifest = parse_prebuilt_manifest(
            &fs::read(fixture.path().join(PREBUILT_PROVENANCE)).expect("read provenance"),
        )
        .expect("parse provenance");
        let current_seal = expected_build_provenance(&manifest).expect("current seal");
        rewrite_prebuilt_binary(fixture.path(), move |binary| {
            binary.extend_from_slice(current_seal.as_bytes());
        });

        let error = validate_optional_prebuilt(fixture.path())
            .expect_err("stale canonical seal plus current canonical seal must fail");
        assert!(error.to_string().contains("found canonical=2, exact=1"));
    }

    #[test]
    fn every_ay_source_line_must_use_the_canonical_revision_form() {
        for noncanonical in [
            format!(
                "source = \"git+ssh://git@github.com:alabsystems/ay.git?branch=main#{}\"",
                "1".repeat(40)
            ),
            format!(
                "source = \"git+https://github.com/alabsystems/ay?branch=main#{}\"",
                "2".repeat(40)
            ),
        ] {
            let lock = format!(
                "source = \"git+https://github.com/alabsystems/ay.git?rev={TEST_AY_COMMIT}#{TEST_AY_COMMIT}\"\n{noncanonical}\n"
            );
            let error = exact_ay_lock_commit(lock.as_bytes())
                .expect_err("additional non-canonical AY source must fail");
            assert!(error
                .to_string()
                .contains("non-canonical AY Cargo.lock source"));
        }
    }

    #[test]
    fn aarch64_elf_under_x86_name_is_rejected_without_execution() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 183);
        let error = validate_optional_prebuilt(fixture.path()).expect_err("wrong arch must fail");
        assert!(error
            .to_string()
            .contains("not an ELF64 little-endian x86_64"));
    }

    #[test]
    fn checksum_corruption_is_rejected_before_decompression() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        fs::write(fixture.path().join(PREBUILT_CHECKSUM), "0\n").expect("corrupt checksum");
        let error = validate_optional_prebuilt(fixture.path()).expect_err("checksum must fail");
        assert!(error.to_string().contains("checksum sidecar"));
    }

    #[test]
    fn tracked_source_drift_is_rejected() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        fs::write(
            fixture.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"dirty\"]\n",
        )
        .expect("dirty tracked input");
        let error = validate_optional_prebuilt(fixture.path()).expect_err("dirty source must fail");
        assert!(error.to_string().contains("inputs are dirty"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_prebuilt_member_is_rejected() {
        use std::os::unix::fs::symlink;

        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let checksum = fixture.path().join(PREBUILT_CHECKSUM);
        fs::remove_file(&checksum).expect("remove checksum");
        symlink(PREBUILT_ARCHIVE, &checksum).expect("symlink checksum");
        let error = validate_optional_prebuilt(fixture.path()).expect_err("symlink must fail");
        assert!(error.to_string().contains("regular file"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_prebuilt_parent_cannot_escape_the_checkout() {
        use std::os::unix::fs::symlink;

        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let real_bin = fixture.path().join("outside-bin");
        fs::rename(fixture.path().join("dist/bin"), &real_bin).expect("move real bin");
        symlink(&real_bin, fixture.path().join("dist/bin")).expect("symlink parent");
        let error = validate_optional_prebuilt(fixture.path()).expect_err("parent must fail");
        assert!(error
            .to_string()
            .contains("parent must be a real directory"));
    }

    #[test]
    fn duplicate_or_unknown_manifest_keys_are_rejected() {
        for extra in ["schema=ny-vnncomp-prebuilt-v1\n", "surprise=value\n"] {
            let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
            let path = fixture.path().join(PREBUILT_PROVENANCE);
            let mut contents = fs::read_to_string(&path).expect("read provenance");
            contents.push_str(extra);
            fs::write(path, contents).expect("mutate provenance");
            assert!(validate_optional_prebuilt(fixture.path()).is_err());
        }
    }

    #[test]
    fn tarball_contains_the_exact_validated_prebuilt_snapshot() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let included = package_file_list(fixture.path()).expect("package list");
        let output = fixture.path().join("submission.tar.gz");
        let expected: BTreeMap<_, _> = PREBUILT_FILES
            .into_iter()
            .map(|relative| {
                (
                    relative,
                    fs::read(fixture.path().join(relative)).expect("fixture member"),
                )
            })
            .collect();

        write_tarball(fixture.path(), &output, &included).expect("write tarball");
        for relative in PREBUILT_FILES {
            let archived = Command::new("tar")
                .args(["-xOzf"])
                .arg(&output)
                .arg(relative)
                .output()
                .expect("extract member");
            assert!(archived.status.success());
            assert_eq!(archived.stdout, expected[relative]);
        }
        let source = Command::new("tar")
            .args(["-xOzf"])
            .arg(&output)
            .arg(SOURCE_PROVENANCE)
            .output()
            .expect("extract source marker");
        assert!(source.status.success());
        assert!(String::from_utf8(source.stdout)
            .expect("source marker UTF-8")
            .contains(&format!(
                "ny_commit={}\n",
                run_git(fixture.path(), &["rev-parse", "HEAD"])
            )));
    }

    #[test]
    fn normalized_output_alias_cannot_overwrite_a_packaged_input() {
        let fixture = make_prebuilt_fixture(TEST_AY_COMMIT, 62);
        let included = package_file_list(fixture.path()).expect("package list");
        let alias = fixture.path().join("dist/bin/../bin/ny-x86_64-linux.xz");
        let error = write_tarball(fixture.path(), &alias, &included)
            .expect_err("normalized input alias must fail");
        assert!(error
            .to_string()
            .contains("must not overwrite a packaged input"));
    }

    #[test]
    fn find_repo_root_from_current_checkout() {
        let cwd = std::env::current_dir().expect("cwd");
        let root = find_repo_root(&cwd).expect("repo root");
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.join("vnncomp_scripts").is_dir());
    }
}
