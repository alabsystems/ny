// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cargo-owned black-box contracts for repository tools implemented in Python.
//!
//! This target is selected only by the explicit `python-tools` feature. The
//! test harness, process expectations, and artifact assertions live in Rust;
//! Python is treated as the external program under test. Tests use synthetic
//! inputs or checked-in fixtures and never turn a missing benchmark corpus or
//! measurement packet into a passing result.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time must follow the Unix epoch")
            .as_nanos();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ny-python-tooling-{label}-{}-{nonce}-{counter}",
            std::process::id()
        ));
        fs::create_dir(&path)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display()));
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn python_executable() -> OsString {
    std::env::var_os("NY_TEST_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    fs::write(path, contents)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn run_repository_script(relative_script: &str, arguments: &[String], cwd: &Path) -> Output {
    let root = workspace_root();
    Command::new(python_executable())
        .args(["-B", "-s"])
        .arg(root.join(relative_script))
        .args(arguments)
        .current_dir(cwd)
        .env_remove("NY_TEST_PYTHON")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .unwrap_or_else(|error| panic!("the selected Python interpreter is unavailable: {error}"))
}

// Sole caller is the `#[cfg(unix)]` byte-path contract below; gate to match so
// this is not dead on Windows.
#[cfg(unix)]
fn run_repository_code(program: &str, test_root: &Path) -> Output {
    let root = workspace_root();
    Command::new(python_executable())
        .args(["-I", "-B", "-c", program])
        .current_dir(&root)
        .env_remove("NY_TEST_PYTHON")
        .env("NY_REPOSITORY_ROOT", &root)
        .env("NY_TEST_ROOT", test_root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .unwrap_or_else(|error| panic!("the selected Python interpreter is unavailable: {error}"))
}

// Captured text is NORMALIZED to LF. These tools print with the platform
// newline and the repository is checked out with CRLF on Windows, while every
// assertion here is written against LF — a split on "\n    {\n" then finds
// nothing and an inventory of six reads as zero. The contracts are about tool
// CONTENT, not line-ending convention, so the convention is removed at each
// point where text enters the test.
fn normalize_newlines(text: String) -> String {
    text.replace("\r\n", "\n")
}

fn stdout(output: &Output) -> String {
    normalize_newlines(String::from_utf8(output.stdout.clone()).expect("tool stdout must be UTF-8"))
}

fn stderr(output: &Output) -> String {
    normalize_newlines(String::from_utf8(output.stderr.clone()).expect("tool stderr must be UTF-8"))
}

/// Read a repository file with the same LF normalization as captured stdout.
fn read_repository_text(path: PathBuf) -> std::io::Result<String> {
    fs::read_to_string(path).map(normalize_newlines)
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout(output),
        stderr(output)
    );
}

fn rust_constant_rhs<'a>(source: &'a str, name: &str) -> &'a str {
    let declaration = format!("pub const {name}:");
    let start = source
        .find(&declaration)
        .unwrap_or_else(|| panic!("missing Rust workload constant {name}"));
    let after_declaration = &source[start + declaration.len()..];
    let equals = after_declaration
        .find('=')
        .unwrap_or_else(|| panic!("missing `=` in Rust workload constant {name}"));
    let rhs = &after_declaration[equals + 1..];
    let semicolon = rhs
        .find(';')
        .unwrap_or_else(|| panic!("missing `;` in Rust workload constant {name}"));
    rhs[..semicolon].trim()
}

fn rust_numbers(source: &str, name: &str) -> Vec<usize> {
    rust_constant_rhs(source, name)
        .split(|character: char| !character.is_ascii_digit())
        .filter(|token| !token.is_empty())
        .map(|token| {
            token
                .parse()
                .unwrap_or_else(|error| panic!("invalid number in {name}: {error}"))
        })
        .collect()
}

fn rust_string<'a>(source: &'a str, name: &str) -> &'a str {
    rust_constant_rhs(source, name)
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or_else(|| panic!("{name} must remain a simple Rust string literal"))
}

fn convolution_output_dimension(specification: &[usize]) -> usize {
    assert_eq!(specification.len(), 9);
    let output_channels = specification[0];
    let kernel = specification[2];
    let output_height = (specification[7] + 2 * specification[5] - kernel) / specification[3] + 1;
    let output_width = (specification[8] + 2 * specification[6] - kernel) / specification[4] + 1;
    output_channels * output_height * output_width
}

fn workload_metadata(source: &str, prefix: &str) -> (String, usize, usize) {
    let case_name = rust_string(source, &format!("{prefix}_CASE_NAME")).to_owned();
    let raw_specs = rust_numbers(source, &format!("{prefix}_CONV_SPECS"));
    assert_eq!(
        raw_specs.len() % 9,
        0,
        "{prefix} Conv specification changed shape"
    );
    let specifications: Vec<_> = raw_specs.chunks_exact(9).collect();
    let convolution_parameters: usize = specifications
        .iter()
        .map(|specification| {
            specification[0] * specification[1] * specification[2] * specification[2]
                + specification[0]
        })
        .sum();
    let maximum_dimension = specifications
        .iter()
        .map(|specification| convolution_output_dimension(specification))
        .max()
        .expect("representative workload must contain a convolution");
    let dense_peak_bytes = 4 * maximum_dimension * maximum_dimension * 4;

    let parameter_count = match prefix {
        "METAROOM" => {
            let hidden = rust_numbers(source, "METAROOM_HIDDEN_DIM")[0];
            let output = rust_numbers(source, "METAROOM_OUTPUT_DIM")[0];
            let flattened = convolution_output_dimension(
                specifications
                    .last()
                    .expect("metaroom workload must contain a convolution"),
            );
            convolution_parameters + hidden * flattened + hidden + output * hidden + output
        }
        "SOUNDNESSBENCH" => {
            let input = rust_numbers(source, "SOUNDNESSBENCH_INPUT_DIM")[0];
            let output = rust_numbers(source, "SOUNDNESSBENCH_OUTPUT_DIM")[0];
            let reshape: usize = rust_numbers(source, "SOUNDNESSBENCH_RESHAPE_SHAPE")
                .into_iter()
                .product();
            convolution_parameters + reshape * input + reshape + output * output + output
        }
        _ => panic!("unsupported workload prefix {prefix}"),
    };
    (case_name, parameter_count, dense_peak_bytes)
}

fn json_string_field<'a>(object: &'a str, field: &str) -> &'a str {
    let marker = format!("\"{field}\": \"");
    let value = object
        .split_once(&marker)
        .unwrap_or_else(|| panic!("missing JSON string field {field}"))
        .1;
    value
        .split_once('"')
        .unwrap_or_else(|| panic!("unterminated JSON string field {field}"))
        .0
}

fn json_usize_field(object: &str, field: &str) -> usize {
    let marker = format!("\"{field}\": ");
    let value = object
        .split_once(&marker)
        .unwrap_or_else(|| panic!("missing JSON integer field {field}"))
        .1;
    let digits: String = value.chars().take_while(char::is_ascii_digit).collect();
    digits
        .parse()
        .unwrap_or_else(|error| panic!("invalid JSON integer field {field}: {error}"))
}

#[test]
fn workload_policy_metadata_is_derived_from_the_rust_workloads() {
    let root = workspace_root();
    let source = fs::read_to_string(
        root.join("crates/ny-gpu/src/benchmark_support/crown_backward_workloads.rs"),
    )
    .expect("GPU workload source must be readable");
    let policy =
        read_repository_text(root.join("configs/benchmark_regressions/gpu_crown_backward.json"))
            .expect("GPU regression policy must be readable");

    let metaroom = workload_metadata(&source, "METAROOM");
    let soundnessbench = workload_metadata(&source, "SOUNDNESSBENCH");
    assert_eq!(
        metaroom,
        (
            "metaroom_6cnn_ry_like".to_owned(),
            7_410_996,
            52_613_349_376
        )
    );
    assert_eq!(
        soundnessbench,
        (
            "soundnessbench_exact_like".to_owned(),
            1_740_696,
            154_618_822_656,
        )
    );

    let checks: Vec<_> = policy
        .split("\n    {\n")
        .skip(1)
        .filter_map(|tail| tail.split_once("\n    }").map(|(object, _)| object))
        .collect();
    assert_eq!(checks.len(), 6, "GPU regression check inventory drifted");
    for (case_name, parameter_count, dense_peak_bytes) in [&metaroom, &soundnessbench] {
        let matching: Vec<_> = checks
            .iter()
            .filter(|check| json_string_field(check, "case") == case_name)
            .collect();
        assert!(
            !matching.is_empty(),
            "missing policy checks for {case_name}"
        );
        for check in matching {
            assert_eq!(
                json_usize_field(check, "expected_parameter_count"),
                *parameter_count,
                "parameter count drifted for {case_name}"
            );
            assert_eq!(
                json_usize_field(check, "expected_estimated_cpu_peak_bytes"),
                *dense_peak_bytes,
                "dense peak drifted for {case_name}"
            );
        }
    }
}

const GPU_POLICY: &str = r#"{
  "suite": "gpu_crown_backward",
  "checks": [
    {
      "name": "gpu_path",
      "case": "metaroom_6cnn_ry_like",
      "phase": "graph_crown_ibp_collection_engine",
      "expected_status": "measured",
      "expected_parameter_count": 7410996,
      "expected_estimated_cpu_peak_bytes": 52613349376,
      "expected_cpu_dense_budget_bytes": 2147483648,
      "baseline_seconds": 4.0,
      "max_regression_ratio": 2.0,
      "max_seconds": 9.0,
      "source_artifact": "reports/benchmarks/candidate_b.csv"
    }
  ]
}
"#;

fn gpu_candidate(seconds: &str, status: &str) -> String {
    format!(
        "case,phase,seconds,parameter_count,estimated_cpu_peak_bytes,cpu_dense_budget_bytes,status,detail\n\
         metaroom_6cnn_ry_like,graph_crown_ibp_collection_engine,{seconds},7410996,52613349376,2147483648,{status},\n"
    )
}

#[test]
fn gpu_checker_selects_the_pinned_source_and_rejects_ambiguous_evidence() {
    let temporary = TestDirectory::new("gpu-checker");
    let policy = temporary.path().join("configs/gpu.json");
    let candidate_a = temporary.path().join("reports/benchmarks/candidate_a.csv");
    let candidate_b = temporary.path().join("reports/benchmarks/candidate_b.csv");
    let report = temporary.path().join("reports/check.json");
    write(&policy, GPU_POLICY);
    write(&candidate_a, &gpu_candidate("12.000000", "measured"));
    write(&candidate_b, &gpu_candidate("6.500000", "measured"));

    let arguments = vec![
        "--policy".to_owned(),
        policy.to_string_lossy().into_owned(),
        "--output".to_owned(),
        report.to_string_lossy().into_owned(),
        "--candidate".to_owned(),
        candidate_a.to_string_lossy().into_owned(),
        "--candidate".to_owned(),
        candidate_b.to_string_lossy().into_owned(),
    ];
    let selected = run_repository_script(
        "scripts/check_gpu_crown_backward_regression.py",
        &arguments,
        temporary.path(),
    );
    assert_success(&selected, "source-pinned GPU regression check");
    let selected_report =
        read_repository_text(report.clone()).expect("checker must write its report");
    assert!(selected_report.contains("\"selection_mode\": \"source_artifact_match\""));
    // Assert on the PIN, not on the raw argument echo. `candidates` repeats the
    // CLI paths verbatim, so on Windows they appear JSON-escaped ("C:\\Users\\...")
    // and a `contains` of the native path can never match. `source_artifact` is
    // the value the checker actually selected, and it is POSIX on every host.
    assert!(selected_report.contains("\"source_artifact\": \"reports/benchmarks/candidate_b.csv\""));
    assert!(selected_report.contains("\"regression\": false"));

    let conflicting_policy = GPU_POLICY.replace(
        "reports/benchmarks/candidate_b.csv",
        "reports/benchmarks/missing.csv",
    );
    write(&policy, &conflicting_policy);
    let rejected = run_repository_script(
        "scripts/check_gpu_crown_backward_regression.py",
        &arguments,
        temporary.path(),
    );
    assert_eq!(
        rejected.status.code(),
        Some(1),
        "ambiguous evidence must fail"
    );
    assert!(stdout(&rejected).contains("source_artifact_missing"));
    let rejected_report =
        read_repository_text(report).expect("checker must write rejection report");
    assert!(rejected_report.contains("\"reasons\": [\n        \"source_artifact_missing\""));
    assert!(rejected_report.contains("\"regression\": true"));
}

#[test]
fn gpu_refresh_is_atomic_on_rejection_and_pins_successful_evidence() {
    let temporary = TestDirectory::new("gpu-refresh");
    let policy = temporary.path().join("configs/gpu.json");
    let candidate = temporary.path().join("reports/benchmarks/candidate.csv");
    let initial_policy = GPU_POLICY.replace(
        "reports/benchmarks/candidate_b.csv",
        "reports/benchmarks/candidate.csv",
    );
    write(&policy, &initial_policy);
    write(&candidate, &gpu_candidate("6.500000", "failed"));
    let arguments = vec![
        "--policy".to_owned(),
        policy.to_string_lossy().into_owned(),
        "--candidate".to_owned(),
        candidate.to_string_lossy().into_owned(),
    ];

    let rejected = run_repository_script(
        "scripts/refresh_gpu_crown_backward_baselines.py",
        &arguments,
        temporary.path(),
    );
    assert_eq!(rejected.status.code(), Some(1), "status mismatch must fail");
    assert!(stdout(&rejected).contains("status_mismatch"));
    assert_eq!(
        read_repository_text(policy.clone()).expect("policy must remain readable"),
        initial_policy,
        "failed refresh must not rewrite the policy"
    );

    write(&candidate, &gpu_candidate("6.500000", "measured"));
    let accepted = run_repository_script(
        "scripts/refresh_gpu_crown_backward_baselines.py",
        &arguments,
        temporary.path(),
    );
    assert_success(&accepted, "valid GPU baseline refresh");
    let refreshed = read_repository_text(policy).expect("refreshed policy must be readable");
    assert!(refreshed.contains("\"baseline_seconds\": 6.5"));
    assert!(refreshed.contains("\"source_artifact\": \"reports/benchmarks/candidate.csv\""));
}

#[test]
fn biccos_planner_manifest_can_never_authorize_execution() {
    let root = workspace_root();
    let temporary = TestDirectory::new("biccos-plan");
    let manifest = temporary.path().join("unsafe.json");
    let output = temporary.path().join("plan.json");
    let committed = fs::read_to_string(root.join("benchmarks/biccos_mts_factorial_v1.json"))
        .expect("BICCOS planner manifest must be readable");
    let unsafe_manifest = committed.replace(
        "\"execution_allowed\": false",
        "\"execution_allowed\": true",
    );
    assert_ne!(
        unsafe_manifest, committed,
        "manifest safety field must exist"
    );
    write(&manifest, &unsafe_manifest);
    let arguments = vec![
        "--manifest".to_owned(),
        manifest.to_string_lossy().into_owned(),
        "--abc-repo".to_owned(),
        temporary
            .path()
            .join("absent-abc")
            .to_string_lossy()
            .into_owned(),
        "--benchmark-root".to_owned(),
        temporary
            .path()
            .join("absent-benchmarks")
            .to_string_lossy()
            .into_owned(),
        "--output".to_owned(),
        output.to_string_lossy().into_owned(),
    ];
    let rejected = run_repository_script(
        "scripts/plan_biccos_mts_factorial.py",
        &arguments,
        temporary.path(),
    );
    assert_eq!(rejected.status.code(), Some(2));
    assert!(stderr(&rejected).contains("planner manifest must forbid execution"));
    assert!(
        !output.exists(),
        "rejected planner input must not emit a plan"
    );
}

#[test]
fn benchmark_report_cli_renders_valid_fixtures_and_rejects_duplicate_rows() {
    let root = workspace_root();
    let temporary = TestDirectory::new("benchmark-report");
    let fixtures = root.join("tests/fixtures/benchmark_reports/compare_backends");
    let valid_output = temporary.path().join("issue-4282-test-current.md");
    let valid_arguments = vec![
        "--metadata".to_owned(),
        fixtures
            .join("valid_metadata.json")
            .to_string_lossy()
            .into_owned(),
        "--csv".to_owned(),
        fixtures
            .join("valid_rows.csv")
            .to_string_lossy()
            .into_owned(),
        "--output".to_owned(),
        valid_output.to_string_lossy().into_owned(),
    ];
    let accepted = run_repository_script(
        "scripts/render_backend_benchmark_report.py",
        &valid_arguments,
        temporary.path(),
    );
    assert_success(&accepted, "valid benchmark report render");
    let rendered = fs::read_to_string(&valid_output).expect("renderer must write Markdown");
    let headings = [
        "## Summary",
        "## Commands",
        "## Artifacts",
        "## Row Identity",
        "## Derived Comparison",
        "## Divergence Gate",
        "## Verdict",
    ];
    let positions: Vec<_> = headings
        .iter()
        .map(|heading| {
            rendered
                .find(heading)
                .unwrap_or_else(|| panic!("missing {heading}"))
        })
        .collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));

    let invalid_output = temporary.path().join("issue-4282-invalid-current.md");
    let invalid_arguments = vec![
        "--metadata".to_owned(),
        fixtures
            .join("valid_metadata.json")
            .to_string_lossy()
            .into_owned(),
        "--csv".to_owned(),
        fixtures
            .join("duplicate_backend_rows.csv")
            .to_string_lossy()
            .into_owned(),
        "--output".to_owned(),
        invalid_output.to_string_lossy().into_owned(),
    ];
    let rejected = run_repository_script(
        "scripts/render_backend_benchmark_report.py",
        &invalid_arguments,
        temporary.path(),
    );
    assert!(
        !rejected.status.success(),
        "duplicate backend rows must fail"
    );
    let error = stderr(&rejected);
    assert!(error.contains("expected exactly one cpu row and one wgpu row"));
    assert!(!error.contains("Traceback"));
    assert!(
        !invalid_output.exists(),
        "invalid rows must not produce a report"
    );
}

#[test]
fn extended_bank_dependency_failure_has_a_distinct_non_evidence_exit() {
    let root = workspace_root();
    let temporary = TestDirectory::new("extended-bank-dependency");
    let blocker = temporary.path().join("blocker");
    fs::create_dir(&blocker).expect("dependency blocker directory must be created");
    write(
        &blocker.join("numpy.py"),
        "raise ImportError('numpy is deliberately blocked by the Cargo contract')\n",
    );
    let counterexample = temporary.path().join("ce.txt");
    write(&counterexample, "((X_0 0.0))\n");
    let output = Command::new(python_executable())
        .args(["-B", "-s"])
        .arg(root.join("scripts/extended_bank/vnnlib_ce.py"))
        .arg(temporary.path().join("missing.onnx"))
        .arg(temporary.path().join("missing.vnnlib"))
        .arg(&counterexample)
        .current_dir(temporary.path())
        .env_remove("NY_TEST_PYTHON")
        .env_remove("PYTHONHOME")
        .env("PYTHONPATH", &blocker)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .unwrap_or_else(|error| panic!("the selected Python interpreter is unavailable: {error}"));
    assert_eq!(output.status.code(), Some(3));
    assert!(stderr(&output).contains("ENVIRONMENT ERROR"));
    assert!(stderr(&output).contains("required runtime package 'numpy'"));
    assert!(!stdout(&output).contains("MOAT"));
}

#[cfg(unix)]
#[test]
fn provenance_unix_byte_paths_round_trip_without_normalization() {
    let temporary = TestDirectory::new("provenance-byte-paths");
    let output = run_repository_code(
        r#"
import importlib.util
import os
import sys
from pathlib import Path

repository = Path(os.environ["NY_REPOSITORY_ROOT"])
root = Path(os.environ["NY_TEST_ROOT"])
script = repository / "scripts" / "ny_measurement_provenance.py"
spec = importlib.util.spec_from_file_location("ny_measurement_provenance_cargo", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

names = [b"-leading", b"line\nbreak"]
descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
try:
    for index, name in enumerate(names):
        child = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=descriptor)
        try:
            os.write(child, f"value-{index}\n".encode())
        finally:
            os.close(child)
finally:
    os.close(descriptor)

first = module._tracked_path_states(root, names)
second = module._tracked_path_states(root, list(reversed(names)))
print("DETERMINISTIC=" + str(first == second))
for entry in first:
    print("ENTRY=" + os.fsencode(entry["path"]).hex() + ":" + entry["kind"] + ":" + str(entry["size_bytes"]))
"#,
        temporary.path(),
    );
    assert_success(&output, "Unix byte-path provenance capture");
    let captured = stdout(&output);
    assert!(captured.contains("DETERMINISTIC=True\n"));
    assert!(captured.contains("ENTRY=2d6c656164696e67:file:8\n"));
    assert!(captured.contains("ENTRY=6c696e650a627265616b:file:8\n"));
}

#[cfg(unix)]
#[test]
fn vnncomp_benchmark_routes_relusplitter_to_mip_without_forcing_a_solver() {
    use std::os::unix::fs::PermissionsExt;

    let root = workspace_root();
    let temporary = TestDirectory::new("benchmark-relusplitter-route");
    let category = "relusplitter";
    let category_dir = temporary
        .path()
        .join("benchmarks/vnncomp2025/benchmarks")
        .join(category);
    fs::create_dir_all(category_dir.join("onnx")).expect("ONNX fixture directory must be created");
    fs::write(category_dir.join("onnx/model.onnx"), b"\x08\x01\x12\x03foo")
        .expect("ONNX fixture must be written");
    write(&category_dir.join("vnnlib/prop.vnnlib"), "");
    write(
        &category_dir.join("instances.csv"),
        "onnx/model.onnx,vnnlib/prop.vnnlib,2\n",
    );

    let fake_ny = temporary.path().join("fake_ny.sh");
    write(
        &fake_ny,
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  printf 'ny fixture 0.0.0\\n'\n  exit 0\nfi\nprintf '%s\\n' \"$@\" > \"$(dirname \"$0\")/argv.txt\"\nprintf 'Status: VERIFIED\\nDomains explored: 3\\n'\n",
    );
    let mut permissions = fs::metadata(&fake_ny)
        .expect("fake ny metadata must be readable")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_ny, permissions).expect("fake ny must be executable");

    let output = Command::new("bash")
        .arg(root.join("scripts/benchmark_vnncomp.sh"))
        .arg(category)
        .current_dir(temporary.path())
        .env(
            "BENCH_ROOT",
            temporary.path().join("benchmarks/vnncomp2025/benchmarks"),
        )
        .env("NY_BIN", &fake_ny)
        .env("MAX_SIGNAL_RETRIES", "1")
        .output()
        .expect("bash is required for the benchmark routing contract");
    assert_success(&output, "relusplitter benchmark routing");

    let arguments = fs::read_to_string(temporary.path().join("argv.txt"))
        .expect("fake ny must record its arguments");
    let arguments: Vec<_> = arguments.lines().collect();
    let complete = arguments
        .windows(2)
        .any(|pair| pair == ["--complete-verifier", "mip"]);
    assert!(
        complete,
        "relusplitter must select the MIP complete verifier: {arguments:?}"
    );
    assert!(
        !arguments.contains(&"--mip-solver"),
        "auto-escalation, not this wrapper, owns the MIP solver choice: {arguments:?}"
    );
    assert!(
        stdout(&output).contains("Verifier: --complete-verifier mip"),
        "routing banner did not describe the effective verifier: {}",
        stdout(&output)
    );

    let report_dir = temporary.path().join("reports/benchmarks");
    let reports: Vec<_> = fs::read_dir(&report_dir)
        .expect("benchmark must create a report directory")
        .map(|entry| {
            entry
                .expect("report directory entry must be readable")
                .path()
        })
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("relusplitter_")
                        && Path::new(name)
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
                })
        })
        .collect();
    assert_eq!(reports.len(), 1, "expected exactly one result bank");
    let report = fs::read_to_string(&reports[0]).expect("result bank must be readable");
    let rows: Vec<_> = report.lines().collect();
    assert_eq!(rows.len(), 2, "result bank must contain one data row");
    let headers: Vec<_> = rows[0].split(',').collect();
    let values: Vec<_> = rows[1].split(',').collect();
    let status = headers
        .iter()
        .position(|header| *header == "status")
        .and_then(|index| values.get(index))
        .copied();
    assert_eq!(status, Some("verified"));
}
