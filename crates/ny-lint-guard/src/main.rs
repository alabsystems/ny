// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! Local Tippy gate: `check` lints changed ny-* crates, `fix` applies fixes
//! except the numeric lints listed below, and `full` lints every workspace
//! member. Checks deny remaining warnings; discovery and final-check failures
//! fail the gate.
//!
//! `LINT_GUARD_BASE` selects the Git baseline (default `origin/main`, falling
//! back to `HEAD` if unavailable). Changed-file discovery includes committed,
//! staged, unstaged and untracked files.

use std::collections::BTreeSet;
use std::process::{Command, ExitCode};

/// Lints whose clippy autofix can change numeric/soundness behavior in a
/// verifier — never auto-applied. Allowed during `fix` so clippy leaves them
/// untouched, then re-reported so a human resolves them by hand.
const SOUNDNESS_LINTS: &[&str] = &[
    "clippy::manual_midpoint",
    "clippy::neg_cmp_op_on_partial_ord",
    "clippy::float_cmp",
    "clippy::excessive_precision",
];

fn main() -> ExitCode {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    let crates = match mode.as_str() {
        "crates" | "check" | "fix" => match changed_crates() {
            Ok(crates) => crates,
            Err(error) => {
                eprintln!("ny-lint-guard: {error}");
                return ExitCode::FAILURE;
            }
        },
        _ => Vec::new(),
    };
    match mode.as_str() {
        "-h" | "--help" | "help" => {
            print_help();
            ExitCode::SUCCESS
        }
        "crates" => {
            for c in crates {
                println!("{c}");
            }
            ExitCode::SUCCESS
        }
        "full" => {
            eprintln!("ny-lint-guard: full workspace Tippy gate (-D warnings)");
            run_gate(&[]) // empty package list => whole workspace
        }
        "check" => {
            if crates.is_empty() {
                println!(
                    "ny-lint-guard: no changed ny-* crates vs {} — nothing to gate.",
                    base()
                );
                return ExitCode::SUCCESS;
            }
            eprintln!(
                "ny-lint-guard: Tippy gate on changed crates: {}",
                crates.join(" ")
            );
            run_gate(&crates)
        }
        "fix" => {
            if crates.is_empty() {
                println!(
                    "ny-lint-guard: no changed ny-* crates vs {} — nothing to fix.",
                    base()
                );
                return ExitCode::SUCCESS;
            }
            run_fix(&crates)
        }
        other => {
            eprintln!(
                "ny-lint-guard: unknown mode '{other}' (use: check | fix | full | crates | --help)"
            );
            ExitCode::from(2)
        }
    }
}

fn print_help() {
    print!(
        "ny-lint-guard — incremental Tippy gate for the ny workspace\n\n\
         USAGE: ny-lint-guard [MODE]\n\n\
         MODES:\n\
         \x20 check   (default)  lint crates changed vs origin/main; fail on any warning\n\
         \x20 fix                apply fixes except listed numeric lints, then check the rest\n\
         \x20 full               the complete workspace gate\n\
         \x20 crates             print the changed ny-* crate names\n\n\
         ENV:  LINT_GUARD_BASE=<git-ref>   baseline for \"changed\" (default origin/main)\n"
    );
}

/// The baseline ref: `LINT_GUARD_BASE`, else `origin/main`, falling back to
/// `HEAD` when the chosen ref does not resolve (fresh clone, no remote).
fn base() -> String {
    let want = std::env::var("LINT_GUARD_BASE").unwrap_or_else(|_| "origin/main".into());
    if git(&["rev-parse", "--verify", "--quiet", &want])
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        want
    } else {
        "HEAD".into()
    }
}

/// The `ny-*` crates with changes versus the baseline: committed-since-base,
/// working-tree, staged, and untracked source files all count.
fn changed_crates() -> Result<Vec<String>, String> {
    let base = base();
    let mut files = String::new();
    for args in [
        vec!["diff", "--name-only", "-z", &format!("{base}...HEAD")],
        vec!["diff", "--name-only", "-z", "HEAD"],
        vec!["diff", "--name-only", "-z", "--cached"],
        vec!["ls-files", "--others", "--exclude-standard", "-z"],
    ] {
        let out =
            git(&args).map_err(|error| format!("could not inspect changed files: {error}"))?;
        if !out.status.success() {
            return Err(format!(
                "git {} failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        files.push_str(&String::from_utf8_lossy(&out.stdout));
    }
    let mut set = BTreeSet::new();
    for line in files.split('\0') {
        // "crates/ny-foo/src/..." -> "ny-foo"
        if let Some(rest) = line.strip_prefix("crates/") {
            if let Some(name) = rest.split('/').next() {
                if name.starts_with("ny-") && !name.is_empty() {
                    set.insert(name.to_string());
                }
            }
        }
    }
    Ok(set.into_iter().collect())
}

/// Run the clippy gate (`-D warnings`) over the given packages, or the whole
/// workspace when the list is empty. Returns clippy's exit status.
fn run_gate(pkgs: &[String]) -> ExitCode {
    let mut args = vec![lint_subcommand().to_string()];
    if pkgs.is_empty() {
        args.push("--workspace".into());
    }
    for p in pkgs {
        args.push("-p".into());
        args.push(p.clone());
    }
    args.extend(["--all-targets", "--all-features", "--", "-D", "warnings"].map(String::from));
    passthrough(cargo(&args))
}

/// Apply fixes with the listed numeric lints disabled, then check all lints.
fn run_fix(crates: &[String]) -> ExitCode {
    eprintln!(
        "ny-lint-guard: applying fixes except listed numeric lints on: {}",
        crates.join(" ")
    );
    let mut args = vec![
        lint_subcommand().to_string(),
        "--fix".into(),
        "--allow-dirty".into(),
        "--allow-staged".into(),
    ];
    for p in crates {
        args.push("-p".into());
        args.push(p.clone());
    }
    args.extend(["--all-targets", "--all-features", "--"].map(String::from));
    for l in SOUNDNESS_LINTS {
        args.push("-A".into());
        args.push((*l).to_string());
    }
    let _ = cargo(&args); // best-effort; report the residue regardless

    eprintln!("\nny-lint-guard: remaining diagnostics (numeric lints require manual review):");
    let mut report = vec![lint_subcommand().to_string()];
    for p in crates {
        report.push("-p".into());
        report.push(p.clone());
    }
    report.extend(
        [
            "--all-targets",
            "--all-features",
            "--message-format=short",
            "--",
            "-D",
            "warnings",
        ]
        .map(String::from),
    );
    match cargo_capture(&report) {
        Some(out) => {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            if out.status.success() {
                println!("  (none — changed crates are clean)");
                ExitCode::SUCCESS
            } else {
                eprint!("{text}");
                ExitCode::FAILURE
            }
        }
        None => {
            eprintln!("ny-lint-guard: failed to spawn the remaining-diagnostics check");
            ExitCode::FAILURE
        }
    }
}

// ---- subprocess helpers ----

fn git(args: &[&str]) -> std::io::Result<std::process::Output> {
    Command::new("git").args(args).output()
}

/// The build tool: `$CARGO` (whatever runs this, normally targo), else
/// `targo` from PATH. There is no bare-`cargo` fallback.
fn targo() -> std::ffi::OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| "targo".into())
}

/// The Trust toolchain ships its clippy as `targo tippy`; a stock cargo
/// spells it `clippy`.
fn lint_subcommand() -> &'static str {
    let tool = targo();
    let is_targo = std::path::Path::new(&tool)
        .file_stem()
        .is_some_and(|stem| stem == "targo");
    if is_targo {
        "tippy"
    } else {
        "clippy"
    }
}

fn cargo(args: &[String]) -> Option<std::process::ExitStatus> {
    Command::new(targo()).args(args).status().ok()
}

fn cargo_capture(args: &[String]) -> Option<std::process::Output> {
    Command::new(targo()).args(args).output().ok()
}

fn passthrough(status: Option<std::process::ExitStatus>) -> ExitCode {
    match status {
        Some(s) if s.success() => ExitCode::SUCCESS,
        Some(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        None => {
            eprintln!("ny-lint-guard: failed to spawn targo (the Trust toolchain's cargo)");
            ExitCode::FAILURE
        }
    }
}
