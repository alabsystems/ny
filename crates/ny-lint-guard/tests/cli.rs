// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Black-box contracts for the Unix shell-based lint driver.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "ny-lint-guard-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).expect("create fixture");
        Self(root)
    }

    fn git(&self, arguments: &[&str]) {
        let output = Command::new("git")
            .current_dir(&self.0)
            .args(arguments)
            .output()
            .expect("run git");
        assert!(output.status.success(), "{}", diagnostics(&output));
    }

    fn repository(&self) {
        self.git(&["init", "--quiet"]);
        fs::write(self.0.join("README.md"), "fixture\n").expect("write tracked file");
        self.git(&["add", "."]);
        self.git(&[
            "-c",
            "user.name=Lint test",
            "-c",
            "user.email=lint@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ]);
    }

    fn script(&self, name: &str, body: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write fake tool");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))
            .expect("make fake tool executable");
        path
    }

    fn run(&self, mode: &str, tool: Option<&Path>) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ny-lint-guard"));
        command
            .current_dir(&self.0)
            .env("LINT_GUARD_BASE", "HEAD")
            .arg(mode);
        if let Some(tool) = tool {
            command.env("CARGO", tool);
        }
        command.output().expect("run lint guard")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove fixture");
    }
}

fn diagnostics(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn git_failure_cannot_report_an_empty_clean_tree() {
    let fixture = Fixture::new();
    let output = fixture.run("check", None);
    assert!(!output.status.success());
    let text = diagnostics(&output);
    assert!(text.contains("git diff"), "{text}");
    assert!(!text.contains("nothing to gate"), "{text}");
}

#[test]
fn full_includes_every_workspace_member() {
    let fixture = Fixture::new();
    let tool = fixture.script("targo", "printf '%s\\n' \"$@\" > \"$0.args\"");
    let output = fixture.run("full", Some(&tool));
    assert!(output.status.success(), "{}", diagnostics(&output));
    let arguments = fs::read_to_string(fixture.0.join("targo.args")).expect("tool arguments");
    assert!(arguments.lines().any(|argument| argument == "--workspace"));
    assert!(arguments
        .lines()
        .any(|argument| argument == "--all-targets"));
    assert!(arguments
        .lines()
        .any(|argument| argument == "--all-features"));
    assert!(arguments.lines().any(|argument| argument == "warnings"));
}

#[test]
fn quoted_untracked_paths_still_select_their_crate() {
    let fixture = Fixture::new();
    fixture.repository();
    let source = fixture.0.join("crates/ny-core/src");
    fs::create_dir_all(&source).expect("create crate fixture");
    fs::write(source.join("quoted\"source.rs"), "// fixture\n").expect("write source");
    let output = fixture.run("crates", None);
    assert!(output.status.success(), "{}", diagnostics(&output));
    assert_eq!(String::from_utf8_lossy(&output.stdout), "ny-core\n");
}

#[test]
fn fix_reports_compiler_errors_even_without_warning_lines() {
    let fixture = Fixture::new();
    fixture.repository();
    fs::create_dir_all(fixture.0.join("crates/ny-core/src")).expect("create crate fixture");
    fs::write(fixture.0.join("crates/ny-core/src/lib.rs"), "fixture\n").expect("write source");
    let tool = fixture.script(
        "targo",
        "for arg do [ \"$arg\" != --fix ] || exit 0; done\n\
         echo 'error[E0425]: cannot find value' >&2\nexit 101",
    );
    let output = fixture.run("fix", Some(&tool));
    assert!(!output.status.success());
    let text = diagnostics(&output);
    assert!(text.contains("error[E0425]"), "{text}");
    assert!(!text.contains("changed crates are clean"), "{text}");
}

#[test]
fn fix_checks_remaining_warnings_as_errors() {
    let fixture = Fixture::new();
    fixture.repository();
    fs::create_dir_all(fixture.0.join("crates/ny-core/src")).expect("create crate fixture");
    fs::write(fixture.0.join("crates/ny-core/src/lib.rs"), "fixture\n").expect("write source");
    let tool = fixture.script("targo", "printf '%s\\n' \"$@\" > \"$0.args\"\nexit 0");
    let output = fixture.run("fix", Some(&tool));
    assert!(output.status.success(), "{}", diagnostics(&output));
    let arguments = fs::read_to_string(fixture.0.join("targo.args")).expect("tool arguments");
    assert!(arguments.contains("--\n-D\nwarnings\n"), "{arguments}");
}
