// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `ny-lint-guard` — a compiled, incremental clippy gate for the ny workspace.
//!
//! This repo has no CI, so the `-D warnings` lint gate (`make lint`) is only as
//! green as the last person to run it, and the full workspace gate is too slow
//! to run on every change. This tool runs stock clippy scoped to only the
//! `ny-*` crates changed versus a baseline (default `origin/main`), so it
//! finishes in seconds.
//!
//! Modes:
//!   check   (default)  clippy the changed crates, fail on any warning
//!   fix                auto-apply the mechanically-safe fixes, then list the rest
//!   full               the complete workspace gate (== `cargo clippy` -D warnings)
//!   crates             print the changed `ny-*` crate names and exit
//!
//! The `fix` mode holds the soundness-sensitive lints OUT of the autofix so
//! clippy can never rewrite a bound computation or a NaN-rejecting comparison
//! in this verifier; those are re-reported for a human to resolve with a scoped
//! `#[allow]` + reason.
//!
//! Baseline override: `LINT_GUARD_BASE=<git-ref>` (falls back to `HEAD` if the
//! ref does not resolve).

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
    match mode.as_str() {
        "-h" | "--help" | "help" => {
            print_help();
            ExitCode::SUCCESS
        }
        "crates" => {
            for c in changed_crates() {
                println!("{c}");
            }
            ExitCode::SUCCESS
        }
        "full" => {
            eprintln!("ny-lint-guard: full workspace clippy gate (-D warnings)");
            run_gate(&[]) // empty package list => whole workspace
        }
        "check" => {
            let crates = changed_crates();
            if crates.is_empty() {
                println!(
                    "ny-lint-guard: no changed ny-* crates vs {} — nothing to gate.",
                    base()
                );
                return ExitCode::SUCCESS;
            }
            eprintln!(
                "ny-lint-guard: clippy gate on changed crates: {}",
                crates.join(" ")
            );
            run_gate(&crates)
        }
        "fix" => {
            let crates = changed_crates();
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
        "ny-lint-guard — incremental clippy gate for the ny workspace\n\n\
         USAGE: ny-lint-guard [MODE]\n\n\
         MODES:\n\
         \x20 check   (default)  clippy the crates changed vs origin/main; fail on any warning\n\
         \x20 fix                auto-apply mechanically-safe fixes, then list what remains\n\
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
fn changed_crates() -> Vec<String> {
    let base = base();
    let mut files = String::new();
    for args in [
        vec!["diff", "--name-only", &format!("{base}...HEAD")],
        vec!["diff", "--name-only", "HEAD"],
        vec!["diff", "--name-only", "--cached"],
        vec!["ls-files", "--others", "--exclude-standard"],
    ] {
        if let Some(out) = git(&args) {
            files.push_str(&String::from_utf8_lossy(&out.stdout));
            files.push('\n');
        }
    }
    let mut set = BTreeSet::new();
    for line in files.lines() {
        // "crates/ny-foo/src/..." -> "ny-foo"
        if let Some(rest) = line.strip_prefix("crates/") {
            if let Some(name) = rest.split('/').next() {
                if name.starts_with("ny-") && !name.is_empty() {
                    set.insert(name.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// Run the clippy gate (`-D warnings`) over the given packages, or the whole
/// workspace when the list is empty. Returns clippy's exit status.
fn run_gate(pkgs: &[String]) -> ExitCode {
    let mut args = vec!["clippy".to_string()];
    for p in pkgs {
        args.push("-p".into());
        args.push(p.clone());
    }
    args.extend(["--all-targets", "--all-features", "--", "-D", "warnings"].map(String::from));
    passthrough(cargo(&args))
}

/// Autofix the mechanically-safe lints on the changed crates (soundness lints
/// held out), then report every diagnostic that remains.
fn run_fix(crates: &[String]) -> ExitCode {
    eprintln!(
        "ny-lint-guard: auto-fixing mechanically-safe lints on: {}",
        crates.join(" ")
    );
    let mut args = vec![
        "clippy".to_string(),
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

    eprintln!(
        "\nny-lint-guard: remaining diagnostics (incl. the soundness-sensitive lints that must be\n\
         resolved by hand — keep the literal and add a scoped #[allow] with a reason):"
    );
    let mut report = vec!["clippy".to_string()];
    for p in crates {
        report.push("-p".into());
        report.push(p.clone());
    }
    report.extend(["--all-targets", "--all-features", "--message-format=short"].map(String::from));
    match cargo_capture(&report) {
        Some(out) => {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let warnings: Vec<&str> = text.lines().filter(|l| l.contains(": warning:")).collect();
            if warnings.is_empty() {
                println!("  (none — changed crates are clean)");
                ExitCode::SUCCESS
            } else {
                for w in &warnings {
                    println!("{w}");
                }
                ExitCode::FAILURE
            }
        }
        None => ExitCode::FAILURE,
    }
}

// ---- subprocess helpers ----

fn git(args: &[&str]) -> Option<std::process::Output> {
    Command::new("git").args(args).output().ok()
}

fn cargo(args: &[String]) -> Option<std::process::ExitStatus> {
    Command::new("cargo").args(args).status().ok()
}

fn cargo_capture(args: &[String]) -> Option<std::process::Output> {
    Command::new("cargo").args(args).output().ok()
}

fn passthrough(status: Option<std::process::ExitStatus>) -> ExitCode {
    match status {
        Some(s) if s.success() => ExitCode::SUCCESS,
        Some(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        None => {
            eprintln!("ny-lint-guard: failed to spawn cargo");
            ExitCode::FAILURE
        }
    }
}
