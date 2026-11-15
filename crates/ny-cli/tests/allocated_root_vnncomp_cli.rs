// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original scalar-V1 CLI controls for joint allocated signed root support.
#![cfg(feature = "mip")]

use std::fs;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use prost::Message;

#[path = "support/allocated_root_fixture.rs"]
mod fixture;

struct Case {
    directory: Option<tempfile::TempDir>,
    result: String,
    log: String,
}

impl Drop for Case {
    fn drop(&mut self) {
        if std::thread::panicking() {
            if let Some(directory) = self.directory.take() {
                eprintln!(
                    "failed original control retained at {}",
                    directory.keep().display()
                );
            }
        }
    }
}

fn kill_and_wait(child: &mut Child) {
    #[cfg(unix)]
    {
        let group = i32::try_from(child.id()).unwrap();
        // SAFETY: the owned, unreaped child was spawned with process_group(0),
        // so its positive PID names its isolated group, never our own group.
        // This runs only after an unfinished wait or wait error, before reap.
        unsafe {
            libc::kill(-group, libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn run(bias: f32, permuted: bool) -> Case {
    let mut case = Case {
        directory: Some(tempfile::tempdir().unwrap()),
        result: String::new(),
        log: String::new(),
    };
    let root = case.directory.as_ref().unwrap().path();
    let configs = root.join("configs/vnncomp99");
    fs::create_dir_all(&configs).unwrap();
    // CPU placement only. Directional/groups/phase policy uses shipped defaults.
    fs::write(
        configs.join("allocated_dense.yaml"),
        "margin_row:\n  gpu: false\n  gpu_batch: false\n  gpu_backward: false\n",
    )
    .unwrap();
    let onnx = root.join("original.onnx");
    let property = root.join("original.vnnlib");
    let result = root.join("result.txt");
    let model_bytes = fixture::model(bias, permuted).encode_to_vec();
    let property_bytes = fixture::property().into_bytes();
    fs::write(&onnx, &model_bytes).unwrap();
    fs::write(&property, &property_bytes).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_ny"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("NY_") {
            command.env_remove(key);
        }
    }
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .args(["vnncomp", "v1", "allocated_dense"])
        .arg(&onnx)
        .arg(&property)
        .arg(&result)
        .args(["20", "--configs-dir"])
        .arg(root.join("configs"))
        .args(["--proof-profile", "competition"])
        .env("NY_VNNCOMP_EXTERNAL_WATCHDOG", "0")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::null())
        .stdout(fs::File::create(root.join("stdout.log")).unwrap())
        .stderr(fs::File::create(root.join("stderr.log")).unwrap())
        .spawn()
        .expect("spawn original VNN-COMP entry point");
    let deadline = Instant::now() + Duration::from_secs(45);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => {
                kill_and_wait(&mut child);
                panic!("original control wait failed: {error}");
            }
        }
        if Instant::now() >= deadline {
            kill_and_wait(&mut child);
            panic!("original allocated control exceeded watchdog");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    case.log = format!(
        "{}\nstdout:\n{}",
        fs::read_to_string(root.join("stderr.log")).unwrap(),
        fs::read_to_string(root.join("stdout.log")).unwrap(),
    );
    assert!(status.success(), "{status}\n{}", case.log);
    assert_eq!(
        fs::read(&onnx).unwrap(),
        model_bytes,
        "original model changed"
    );
    assert_eq!(
        fs::read(&property).unwrap(),
        property_bytes,
        "original property changed"
    );
    case.result = fs::read_to_string(result).unwrap();
    case
}

fn field<'a>(line: &'a str, name: &str) -> &'a str {
    line.split_whitespace()
        .find_map(|word| word.strip_prefix(name).and_then(|s| s.strip_prefix('=')))
        .unwrap_or_else(|| panic!("missing {name}: {line}"))
}

fn count(line: &str, name: &str) -> usize {
    field(line, name).parse().unwrap()
}

fn bound(line: &str, name: &str) -> f64 {
    let value = f64::from_bits(u64::from_str_radix(field(line, name), 16).unwrap());
    assert!(value.is_finite(), "{line}");
    value
}

fn one_line<'a>(log: &'a str, marker: &str) -> &'a str {
    let mut lines = log.lines().filter(|line| line.starts_with(marker));
    let line = lines
        .next()
        .unwrap_or_else(|| panic!("{marker} did not engage (VOID):\n{log}"));
    assert!(lines.next().is_none(), "duplicate {marker}:\n{log}");
    line
}

fn assert_caps(log: &str, line: &str) {
    let portfolio = one_line(log, "[input-head-directional-portfolio]");
    assert!(count(portfolio, "rounds") <= 3, "{portfolio}");
    assert!(count(portfolio, "attempted") <= 32, "{portfolio}");
    assert!(count(line, "algebraic_masks") <= 8192, "{line}");
    assert!(count(line, "work_cells") <= 2_000_000, "{line}");
    assert!(count(line, "calls") <= 3, "{line}");
}

fn assert_safe(case: &Case) {
    assert_eq!(
        case.result.split_whitespace().next(),
        Some("unsat"),
        "{}\n{}",
        case.result,
        case.log
    );
    let admission = one_line(&case.log, "[input-head-admission]");
    for (name, expected) in [
        ("inputs", "30"),
        ("head", "78"),
        ("batch", "1"),
        ("binding", "original_property"),
        ("property_format", "scalar_v1_real"),
        ("phase_cover_max_heads", "0"),
    ] {
        assert_eq!(field(admission, name), expected, "{admission}");
    }
    let g3 = one_line(&case.log, "[g3] lane=margin_row_input_head ");
    assert_eq!(field(g3, "g3"), "ran", "{g3}");
    assert_eq!(field(g3, "receipt"), "ok", "{g3}");
    let allocated = one_line(&case.log, "[input-head-directional-allocated]");
    assert_eq!(count(allocated, "planned_groups"), 6, "{allocated}");
    assert_eq!(
        count(allocated, "covered_negative_heads"),
        fixture::NEGATIVE,
        "{allocated}"
    );
    assert_eq!(count(allocated, "omitted_negative_heads"), 0, "{allocated}");
    let admitted = count(allocated, "admitted");
    assert_eq!(admitted, 6, "full allocated portfolio missing: {allocated}");
    assert_eq!(count(allocated, "certified"), 6, "{allocated}");
    for name in [
        "completed_masks",
        "charged_masks",
        "attempted_source_replays",
    ] {
        assert_eq!(count(allocated, name), 1536, "{allocated}");
    }
    let rows: Vec<_> = case
        .log
        .lines()
        .filter(|line| line.starts_with("[input-head-directional-allocated-row]"))
        .filter(|line| count(line, "admitted") == 1)
        .collect();
    assert_eq!(rows.len(), admitted, "{}", case.log);
    for row in rows {
        assert_eq!(count(row, "certified"), 1, "{row}");
        assert_eq!(count(row, "completed_masks"), 256, "{row}");
        assert_eq!(count(row, "charged_masks"), 256, "{row}");
        assert_eq!(count(row, "attempted_source_replays"), 256, "{row}");
        bound(row, "lower_bits");
    }
    let directional = one_line(&case.log, "[input-head-directional]");
    assert_caps(&case.log, directional);
    let portfolio = one_line(&case.log, "[input-head-directional-portfolio]");
    assert!(count(portfolio, "rounds") >= 2, "{portfolio}");
    let rounds: Vec<_> = case
        .log
        .lines()
        .filter(|line| line.starts_with("[input-head-directional-round]"))
        .collect();
    assert_eq!(rounds.len(), count(portfolio, "rounds"), "{}", case.log);
    assert_eq!(
        rounds.iter().map(|line| count(line, "cuts")).sum::<usize>(),
        count(directional, "cuts"),
        "{}",
        case.log
    );
    for round in rounds {
        assert!(count(round, "cuts") <= 4, "{round}");
    }
    assert!(count(directional, "cuts") >= admitted, "{directional}");
    assert!(count(directional, "native_replayed") > 0, "{directional}");
    assert!(bound(directional, "before_bits") <= 0.0, "{directional}");
    let after = bound(directional, "after_bits");
    assert!(after > 0.4 && after <= 0.5, "{directional}");
}

fn assert_unsafe(case: &Case) {
    // The original zero assignment evaluates all authored heads to zero,
    // hence its output margin is exactly -0.5. Upfront trusted SAT may return
    // before directional admission; it remains a no-false-UNSAT control.
    assert!(
        matches!(
            case.result.split_whitespace().next(),
            Some("sat" | "unknown")
        ),
        "{}\n{}",
        case.result,
        case.log
    );
    assert!(
        !case.log.contains("[g3] lane=margin_row_input_head"),
        "{}",
        case.log
    );
    if let Some(line) = case
        .log
        .lines()
        .find(|line| line.starts_with("[input-head-directional]"))
    {
        assert_caps(&case.log, line);
        if field(line, "after_bits") != "none" {
            assert!(bound(line, "after_bits") <= -0.5, "{line}");
        }
    }
}

#[test]
fn allocated_signed_rows_close_original_scalar_v1_with_default_policy_and_g3() {
    fixture::assert_exact_reference();
    assert_eq!(fixture::INPUTS, 30);
    assert_eq!(fixture::HEADS, 78);
    for permuted in [false, true] {
        assert_safe(&run(0.5, permuted));
        assert_unsafe(&run(-0.5, permuted));
    }
}
