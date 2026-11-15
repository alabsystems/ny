// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

fn invoke(input: Option<&Path>, options: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ny-maxsat"));
    command.env_clear();
    if let Some(path) = input {
        command.arg(path);
    }
    let mut child = command
        .args(options)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the built NY frontend");
    let hang = Instant::now() + Duration::from_secs(60);
    while child.try_wait().expect("observe frontend exit").is_none() {
        if Instant::now() >= hang {
            let _ = child.kill();
            let output = child.wait_with_output().expect("reap hung frontend");
            panic!("frontend hang detector: {output:?}");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    child.wait_with_output().expect("collect frontend output")
}

fn run(text: &str, options: &[&str]) -> Output {
    let directory = tempfile::tempdir().expect("private fixture directory");
    let path = directory.path().join("input.wcnf");
    std::fs::write(&path, text).expect("write original WCNF");
    invoke(Some(&path), options)
}

fn frames(output: &Output) -> Vec<&str> {
    std::str::from_utf8(&output.stdout)
        .expect("UTF-8 protocol output")
        .lines()
        .filter(|line| line.starts_with("o ") || line.starts_with("s ") || line.starts_with("v "))
        .collect()
}

#[test]
fn legacy_and_modern_inputs_export_the_original_model_and_cost() {
    for text in ["h 1 0\n7 -1 0\n", "p wcnf 1 2 10\n10 1 0\n7 -1 0\n"] {
        let output = run(text, &["--timeout", "5"]);
        assert!(output.status.success(), "{output:?}");
        // The hard clause forces x=true; the original soft clause costs seven.
        // This exercises the real in-process MIP route, not a fabricated result.
        assert_eq!(
            frames(&output),
            ["o 7", "s SATISFIABLE", "v 1"],
            "{output:?}"
        );
    }
}

#[test]
fn duplicate_hard_literals_are_enforced_by_the_native_adapter() {
    let output = run("h -1 -1 0\n7 1 0\n", &["--timeout", "5"]);
    assert!(output.status.success(), "{output:?}");
    // The repeated negative literal still forces x=false. This checks the
    // adapter as well as the independent original-clause replay.
    assert_eq!(
        frames(&output),
        ["o 7", "s SATISFIABLE", "v 0"],
        "{output:?}"
    );
}

#[test]
fn native_infeasibility_does_not_export_an_unsat_or_optimum_claim() {
    let output = run("h 1 0\nh -1 0\n", &["--timeout", "5"]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(frames(&output), ["s UNKNOWN"], "{output:?}");
}

#[test]
fn malformed_unsupported_and_empty_models_export_only_unknown() {
    for text in [
        "p cnf 1 1\n1 0\n",
        "1 1\n",
        "9007199254740993 1 0\n",
        "p wcnf 0 0\n",
    ] {
        let output = run(text, &["--timeout", "5"]);
        assert!(output.status.success(), "{text}: {output:?}");
        assert_eq!(frames(&output), ["s UNKNOWN"], "{text}: {output:?}");
    }
}

#[test]
fn invalid_arguments_emit_no_candidate_and_help_needs_no_input() {
    for options in [
        ["--timeout", "NaN"],
        ["--timeout", "inf"],
        ["--timeout", "0"],
        ["--timeout", "86401"],
        ["--not-timeout", "1"],
    ] {
        let output = run("h 1 0\n", &options);
        assert_eq!(output.status.code(), Some(4), "{output:?}");
        assert!(frames(&output).is_empty(), "{output:?}");
    }
    let missing = invoke(None, &[]);
    assert_eq!(missing.status.code(), Some(4), "{missing:?}");
    assert!(frames(&missing).is_empty(), "{missing:?}");
    let help = invoke(None, &["--help"]);
    assert!(help.status.success(), "{help:?}");
    assert!(frames(&help).is_empty(), "{help:?}");
}
