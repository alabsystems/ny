// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cargo-owned black-box contracts for repository policy executables.
//!
//! The production policy implementations remain small Python command-line
//! tools. Cargo owns their invocation and pass/fail contract in the explicit
//! `python-tools` lane; Python is the program under test, not the test runner.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn python_executable() -> OsString {
    std::env::var_os("NY_TEST_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn run_policy_tool(script: &str, arguments: &[&str]) -> Output {
    let root = workspace_root();
    Command::new(python_executable())
        .args(["-B", "-s"])
        .arg(root.join(script))
        .args(arguments)
        .current_dir(&root)
        .env_remove("NY_TEST_PYTHON")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .unwrap_or_else(|error| panic!("the selected Python interpreter is unavailable: {error}"))
}

fn output_text(bytes: &[u8], stream: &str) -> String {
    String::from_utf8(bytes.to_vec())
        .unwrap_or_else(|error| panic!("policy tool {stream} was not UTF-8: {error}"))
}

fn assert_success(output: &Output, context: &str) -> (String, String) {
    let stdout = output_text(&output.stdout, "stdout");
    let stderr = output_text(&output.stderr, "stderr");
    assert!(
        output.status.success(),
        "{context} failed with {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    (stdout, stderr)
}

fn canonical_ay_revision(manifest: &str) -> &str {
    let marker = "git = \"https://github.com/alabsystems/ay.git\"";
    let dependency = manifest
        .split_once(marker)
        .expect("ny-mip must use the canonical AY Git source")
        .1;
    let revision = dependency
        .split_once("rev = \"")
        .expect("canonical AY dependency must be revision-pinned")
        .1
        .split_once('"')
        .expect("canonical AY revision must be a quoted string")
        .0;
    assert_eq!(revision.len(), 40, "AY revision must be a full Git SHA");
    assert!(
        revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "AY revision must contain only hexadecimal digits"
    );
    revision
}

fn cargo_metadata_package_names(metadata: &str) -> BTreeSet<&str> {
    let packages = metadata
        .split_once("\"packages\":[")
        .expect("cargo metadata must contain a packages array")
        .1
        .split_once("],\"workspace_members\"")
        .expect("cargo metadata must contain workspace_members after packages")
        .0;
    let mut names = BTreeSet::new();
    let mut cursor = 0_usize;
    let bytes = packages.as_bytes();
    let mut depth = 0_usize;
    let mut object_start = None;
    let mut in_string = false;
    let mut escaped = false;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            cursor += 1;
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    object_start = Some(cursor);
                }
                depth += 1;
            }
            b'}' => {
                depth = depth
                    .checked_sub(1)
                    .expect("cargo metadata package JSON has balanced objects");
                if depth == 0 {
                    let start = object_start.take().expect("package object has a start");
                    let object = &packages[start..=cursor];
                    let name = object
                        .split_once("\"name\":\"")
                        .expect("workspace package object must have a name")
                        .1
                        .split_once('"')
                        .expect("workspace package name must terminate")
                        .0;
                    assert!(
                        name.bytes().all(|byte| {
                            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
                        }),
                        "workspace package name uses unexpected JSON escaping: {name}"
                    );
                    assert!(names.insert(name), "duplicate Cargo package name {name}");
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    assert!(
        !names.is_empty(),
        "cargo metadata returned no workspace packages"
    );
    names
}

#[test]
fn architecture_layer_policy_accepts_the_complete_workspace() {
    let output = run_policy_tool("scripts/check_architecture_layers.py", &["--json"]);
    let (stdout, stderr) = assert_success(&output, "architecture layer policy");
    assert!(
        stderr.is_empty(),
        "successful policy check wrote stderr: {stderr}"
    );
    assert!(
        stdout.contains("\"violations\": []"),
        "successful policy check did not report an empty violation set: {stdout}"
    );
    assert!(
        stdout.contains("\"pass\": true"),
        "successful policy check did not emit a true pass field: {stdout}"
    );
}

#[test]
fn active_ay_claims_are_aligned_with_the_canonical_manifest_pin() {
    let root = workspace_root();
    let manifest = fs::read_to_string(root.join("crates/ny-mip/Cargo.toml"))
        .expect("ny-mip manifest must be readable");
    let revision = canonical_ay_revision(&manifest);

    let output = run_policy_tool("scripts/align_scored_audit_ay_pin.py", &["--check"]);
    let (stdout, stderr) = assert_success(&output, "AY claim alignment policy");
    assert!(
        stderr.is_empty(),
        "successful alignment check wrote stderr: {stderr}"
    );
    assert_eq!(
        stdout.trim(),
        format!("already aligned to AY {}", &revision[..8])
    );

    for relative in [
        "docs/SCORED_REPRO_AUDIT_2026-07-19.md",
        "docs/AY_BRANCH_HINT_CANARY.md",
    ] {
        let claim = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"));
        assert!(
            claim.contains(revision),
            "{relative} does not name canonical AY revision {revision}"
        );
    }
}

#[test]
fn documented_package_inventory_matches_cargo_metadata() {
    let root = workspace_root();
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--format-version", "1", "--no-deps"])
        .current_dir(&root)
        .output()
        .expect("Cargo must be available while Cargo-owned policy tests run");
    let (stdout, stderr) = assert_success(&output, "cargo metadata");
    assert!(
        stderr.is_empty(),
        "successful cargo metadata wrote stderr: {stderr}"
    );
    let workspace_packages = cargo_metadata_package_names(&stdout);

    let package_doc = fs::read_to_string(root.join("docs/PACKAGES.md"))
        .expect("package inventory must be readable");
    let documented: Vec<_> = package_doc
        .lines()
        .filter_map(|line| line.strip_prefix("- `"))
        .filter_map(|line| line.split_once('`').map(|(name, _)| name))
        .collect();
    let documented_set: BTreeSet<_> = documented.iter().copied().collect();
    assert_eq!(
        documented.len(),
        documented_set.len(),
        "package inventory contains duplicate entries"
    );
    assert_eq!(documented_set, workspace_packages);
}
