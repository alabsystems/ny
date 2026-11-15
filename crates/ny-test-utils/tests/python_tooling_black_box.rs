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

// Run a repository module as an isolated external Python program while still
// allowing the Cargo-owned contract to pass explicit fixture paths.
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

// ---------------------------------------------------------------------------
// scripts/check_score_at_risk_drift.py (gate lane [n])
// ---------------------------------------------------------------------------

/// A minimal stand-in for scripts/ny_retroactive_scorecard.py: the drift
/// checker reads only its two benchmark lists and its two bank-directory
/// argparse defaults, by text.
const SCORE_DRIFT_SCORER_FIXTURE: &str = r#"REGULAR = [
    "alpha",
]
EXTENDED = [
    "beta",
]


def main():
    ap.add_argument("--measured", default="reports/measured")
    ap.add_argument(
        "--ext-measured",
        default="reports/measured-ext",
    )
"#;

const SCORE_DRIFT_YAML_FIXTURE: &str = r#"measured_on: "2026-10-04"
measured_at_sha: "fixture"
command: python3 scripts/ny_retroactive_scorecard.py --extended
totals:
  regular_16: 50.0
  extended_9: 0.0
  full_field_25: 50.0
roots:
  - root: benchmarks/vnncomp2025/benchmarks/alpha
    benchmark: alpha
    track: regular
    banked_normalized: 50.0
  - root: benchmarks/vnncomp2025/benchmarks/beta
    benchmark: beta
    track: extended
    banked_normalized: 0.0
"#;

/// A copy of the real checker under a synthetic repository root (the script
/// resolves every input relative to its own location).
fn score_drift_fixture(label: &str) -> TestDirectory {
    let directory = TestDirectory::new(label);
    let root = directory.path();
    let script =
        read_repository_text(workspace_root().join("scripts/check_score_at_risk_drift.py"))
            .expect("the drift checker must be readable");
    write(&root.join("scripts/check_score_at_risk_drift.py"), &script);
    write(
        &root.join("scripts/ny_retroactive_scorecard.py"),
        SCORE_DRIFT_SCORER_FIXTURE,
    );
    write(
        &root.join("configs/preset_score_at_risk.yaml"),
        SCORE_DRIFT_YAML_FIXTURE,
    );
    // The regular member's bank, in the 6-field shape (index 3 = "prepared").
    write(
        &root.join("reports/measured/alpha.csv"),
        "alpha,onnx/a.onnx,vnnlib/a.vnnlib,prepared,unsat,1.0\n",
    );
    // The extended directory exists (as in the repository) and holds no bank
    // for `beta`, which the fixture yaml prices at 0.0.
    write(&root.join("reports/measured-ext/README.md"), "fixture\n");
    directory
}

/// The fixture yaml with `beta` (extended) priced at 10.0 and the totals to match.
fn score_drift_yaml_with_a_beta_stake() -> String {
    SCORE_DRIFT_YAML_FIXTURE
        .replace("extended_9: 0.0", "extended_9: 10.0")
        .replace("full_field_25: 50.0", "full_field_25: 60.0")
        .replace(
            "    track: extended\n    banked_normalized: 0.0",
            "    track: extended\n    banked_normalized: 10.0",
        )
}

fn run_score_drift(root: &Path) -> Output {
    // An absolute script path: `Path::join` keeps it, so the shared runner (and
    // its interpreter isolation) executes the fixture COPY, not the repository's.
    let script = root.join("scripts/check_score_at_risk_drift.py");
    run_repository_script(
        script.to_str().expect("fixture path must be UTF-8"),
        &[],
        root,
    )
}

/// The yaml prices the 2025 retroactive scorecard, so a solved bank for the
/// same benchmark NAME in another field's ledger (reports/measured-2026: the
/// yolo_2023 case, whose 2026 instances share 10 of 72 with the 2025 set) is a
/// note, not drift — the scorer cannot credit it.
#[test]
fn score_at_risk_drift_ignores_another_fields_bank() {
    let fixture = score_drift_fixture("score-drift-other-field");
    let root = fixture.path();
    write(
        &root.join("reports/measured-2026/beta.csv"),
        "beta,onnx/b.onnx,vnnlib/b.vnnlib,0.0,unsat,1.0\n",
    );

    let output = run_score_drift(root);
    let out = stdout(&output);
    assert!(
        output.status.success(),
        "a 2026-field bank must not price a 2025 stake\nstdout:\n{out}\nstderr:\n{}",
        stderr(&output)
    );
    assert!(
        out.contains("(read from reports/measured + reports/measured-ext)"),
        "{out}"
    );
    assert!(
        out.contains(
            "note: beta: no bank under reports/measured-ext; reports/measured-2026/beta.csv is \
             not an input of the scorer for this extended row"
        ),
        "{out}"
    );
    assert!(out.contains("PASS:"), "{out}");
}

/// The check still fires for the scorer's OWN banks: a solved row in the
/// extended directory for a member priced 0.0 is a recovered category whose
/// stake was never restored.
#[test]
fn score_at_risk_drift_fires_on_the_scorers_own_recovered_bank() {
    let fixture = score_drift_fixture("score-drift-recovered");
    let root = fixture.path();
    write(
        &root.join("reports/measured-ext/beta.csv"),
        "beta,onnx/b.onnx,vnnlib/b.vnnlib,unsat,1.0\n",
    );

    let output = run_score_drift(root);
    let out = stdout(&output);
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout:\n{out}\nstderr:\n{}",
        stderr(&output)
    );
    assert!(
        out.contains("beta: bank has 1 solved rows but is priced at 0.0"),
        "{out}"
    );
}

/// The vit_2023 shape: a bank with zero solved rows priced above 0.0.
#[test]
fn score_at_risk_drift_fires_on_a_zero_solved_bank_with_a_stake() {
    let fixture = score_drift_fixture("score-drift-zeroed");
    let root = fixture.path();
    write(
        &root.join("reports/measured/alpha.csv"),
        "alpha,onnx/a.onnx,vnnlib/a.vnnlib,prepared,timeout,1.0\n",
    );

    let output = run_score_drift(root);
    let out = stdout(&output);
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout:\n{out}\nstderr:\n{}",
        stderr(&output)
    );
    assert!(
        out.contains("alpha: bank has ZERO sat/unsat rows but is priced at 50.0"),
        "{out}"
    );
}

/// The yaml's recorded command is CHECKED against the scorer, never followed:
/// it is part of the file under test, and a check that reads the bank from
/// wherever that file points lets one wrong token choose what the file is
/// compared against. Naming another ledger is itself drift, and the bank is
/// still read from the scorer's directory (so the 2026-field row below stays
/// a note, not a "recovered category").
#[test]
fn score_at_risk_drift_refuses_a_recorded_command_that_names_another_ledger() {
    let fixture = score_drift_fixture("score-drift-command");
    let root = fixture.path();
    let yaml = SCORE_DRIFT_YAML_FIXTURE.replace(
        "command: python3 scripts/ny_retroactive_scorecard.py --extended",
        "command: python3 scripts/ny_retroactive_scorecard.py --measured reports/measured \
         --ext-measured reports/measured-2026 --extended",
    );
    write(&root.join("configs/preset_score_at_risk.yaml"), &yaml);
    write(
        &root.join("reports/measured-2026/beta.csv"),
        "beta,onnx/b.onnx,vnnlib/b.vnnlib,0.0,unsat,1.0\n",
    );

    let output = run_score_drift(root);
    let out = stdout(&output);
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout:\n{out}\nstderr:\n{}",
        stderr(&output)
    );
    assert!(
        out.contains("(read from reports/measured + reports/measured-ext)"),
        "{out}"
    );
    assert!(
        out.contains(
            "recorded `command` passes `--ext-measured reports/measured-2026` but the scorer \
             reads `reports/measured-ext` by default"
        ),
        "{out}"
    );
    assert!(!out.contains("beta: bank has 1 solved rows"), "{out}");
    // The directory the scorer does read, spelled either way, is not drift.
    assert!(!out.contains("`--measured reports/measured`"), "{out}");
}

/// The reviewer's plant (2026-10-08): a member's bank loses every solved row
/// while its stake stays, and the yaml's recorded command carries a mistyped
/// bank directory. When the checker followed that command it read nothing for
/// the member, printed a note and PASSED. Both defects must be reported.
#[test]
fn score_at_risk_drift_a_mistyped_recorded_directory_cannot_narrow_the_check() {
    for (label, create_the_typo_directory) in [("absent", false), ("empty", true)] {
        let fixture = score_drift_fixture(&format!("score-drift-typo-{label}"));
        let root = fixture.path();
        let yaml = SCORE_DRIFT_YAML_FIXTURE.replace(
            "command: python3 scripts/ny_retroactive_scorecard.py --extended",
            "command: python3 scripts/ny_retroactive_scorecard.py --measured=reports/measured-typo \
             --extended",
        );
        write(&root.join("configs/preset_score_at_risk.yaml"), &yaml);
        write(
            &root.join("reports/measured/alpha.csv"),
            "alpha,onnx/a.onnx,vnnlib/a.vnnlib,prepared,timeout,1.0\n",
        );
        if create_the_typo_directory {
            write(&root.join("reports/measured-typo/README.md"), "fixture\n");
        }

        let output = run_score_drift(root);
        let out = stdout(&output);
        assert_eq!(
            output.status.code(),
            Some(1),
            "{label}\nstdout:\n{out}\nstderr:\n{}",
            stderr(&output)
        );
        assert!(
            out.contains("alpha: bank has ZERO sat/unsat rows but is priced at 50.0"),
            "{label}: {out}"
        );
        assert!(
            out.contains(
                "recorded `command` passes `--measured reports/measured-typo` but the scorer \
                 reads `reports/measured` by default"
            ),
            "{label}: {out}"
        );
    }
}

/// A scored member priced above 0.0 with no bank in the directory the scorer
/// reads for its track: the scorer credits an unmeasured benchmark 0, so the
/// stake has nothing behind it. (Before, a missing bank was skipped silently.)
#[test]
fn score_at_risk_drift_fires_on_a_stake_with_no_bank() {
    let fixture = score_drift_fixture("score-drift-no-bank");
    let root = fixture.path();
    write(
        &root.join("configs/preset_score_at_risk.yaml"),
        &score_drift_yaml_with_a_beta_stake(),
    );

    let output = run_score_drift(root);
    let out = stdout(&output);
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout:\n{out}\nstderr:\n{}",
        stderr(&output)
    );
    assert!(
        out.contains("beta: priced at 10.0 but has no bank under reports/measured-ext"),
        "{out}"
    );
}

/// Each row is read from the ONE directory the scorer reads for its track. An
/// extended benchmark's file under the REGULAR directory is not its bank (the
/// repository has seven such pairs; lsnc_relu holds 80 solved rows there and
/// 11 in reports/measured-ext): its extended bank losing every solved row is
/// the vit_2023 drift even while the other file is full.
#[test]
fn score_at_risk_drift_reads_an_extended_row_only_from_the_extended_bank() {
    let fixture = score_drift_fixture("score-drift-track");
    let root = fixture.path();
    write(
        &root.join("configs/preset_score_at_risk.yaml"),
        &score_drift_yaml_with_a_beta_stake(),
    );
    write(
        &root.join("reports/measured/beta.csv"),
        "beta,onnx/b.onnx,vnnlib/b.vnnlib,prepared,unsat,1.0\n",
    );
    write(
        &root.join("reports/measured-ext/beta.csv"),
        "beta,onnx/b.onnx,vnnlib/b.vnnlib,timeout,1.0\n",
    );

    let output = run_score_drift(root);
    let out = stdout(&output);
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout:\n{out}\nstderr:\n{}",
        stderr(&output)
    );
    assert!(
        out.contains("beta: bank has ZERO sat/unsat rows but is priced at 10.0"),
        "{out}"
    );

    // The same stake with its solved row in the extended bank is not drift,
    // under either file name `load_ny_ext` accepts.
    for name in ["beta.csv", "ext_beta_results.csv"] {
        let fixture = score_drift_fixture("score-drift-track-ok");
        let root = fixture.path();
        write(
            &root.join("configs/preset_score_at_risk.yaml"),
            &score_drift_yaml_with_a_beta_stake(),
        );
        write(
            &root.join("reports/measured-ext").join(name),
            "beta,onnx/b.onnx,vnnlib/b.vnnlib,unsat,1.0\n",
        );
        let output = run_score_drift(root);
        let out = stdout(&output);
        assert!(
            output.status.success() && out.contains("2 with banks") && out.contains("PASS:"),
            "{name}\nstdout:\n{out}\nstderr:\n{}",
            stderr(&output)
        );
    }
}

/// A bank directory the scorer reads that does not exist is VOID (exit 2: the
/// check could not look), never a pass and not the drift code.
#[test]
fn score_at_risk_drift_is_void_when_a_scorer_bank_directory_is_missing() {
    let fixture = score_drift_fixture("score-drift-void");
    let root = fixture.path();
    fs::remove_dir_all(root.join("reports/measured-ext")).expect("fixture directory");

    let output = run_score_drift(root);
    let out = stdout(&output);
    assert_eq!(
        output.status.code(),
        Some(2),
        "stdout:\n{out}\nstderr:\n{}",
        stderr(&output)
    );
    assert!(
        out.contains(
            "VOID: bank directory `reports/measured-ext` (the scorer's `--ext-measured` \
             default) does not exist"
        ),
        "{out}"
    );
    assert!(!out.contains("PASS:"), "{out}");
}

// ---------------------------------------------------------------------------
// The transitional pytest suite (gate lane [e]) under an inherited GIT_DIR
// ---------------------------------------------------------------------------

fn git_in(dir: &Path, arguments: &[&str]) -> Output {
    Command::new("git")
        .args(arguments)
        .current_dir(dir)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_COMMON_DIR")
        .output()
        .unwrap_or_else(|error| panic!("git is required for this contract: {error}"))
}

/// Git exports GIT_DIR to the commands it runs from a linked worktree (hooks,
/// `git bisect run`, `git rebase -x`). The suite's fixture repositories are
/// built with plain `git init/config/commit`, so before tests/conftest.py
/// dropped git's repository-local environment, every fixture command acted on
/// the ENCLOSING repository: on 2026-10-04 a `git bisect run` over this suite
/// set `core.bare = true` and `user.name = NY Test` in the shared clone's
/// config and committed the fixture onto the worktree's HEAD. A git-fixture
/// test must pass and leave an inherited GIT_DIR's repository untouched.
#[test]
fn pytest_suite_fixtures_never_touch_an_inherited_git_dir() {
    let directory = TestDirectory::new("inherited-git-dir");
    let outer = directory.path().join("outer");
    fs::create_dir_all(&outer).expect("outer repository directory must be created");
    for arguments in [
        &["init", "-q"][..],
        &[
            "-c",
            "user.name=outer",
            "-c",
            "user.email=outer@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "outer",
        ][..],
    ] {
        let output = git_in(&outer, arguments);
        assert!(
            output.status.success(),
            "git {arguments:?}: {}",
            stderr(&output)
        );
    }
    let config_before = fs::read(outer.join(".git/config")).expect("outer config");
    let head_before = stdout(&git_in(&outer, &["rev-parse", "HEAD"]));

    // The selected test clears every NY_* variable itself. Cargo adds
    // DYLD_FALLBACK_LIBRARY_PATH to a test binary's environment on macOS; the
    // provenance capture rightly refuses that as loader injection, and the
    // gate's own `pytest tests` never carries it, so it is not inherited here.
    let output = Command::new(python_executable())
        .args(["-B", "-s", "-m", "pytest", "-q", "-p", "no:cacheprovider"])
        .arg("tests/test_ny_measurement_provenance.py")
        .args([
            "-k",
            "test_start_manifest_refuses_current_evidence_without_a_predicate",
        ])
        .current_dir(workspace_root())
        .env_remove("DYLD_FALLBACK_LIBRARY_PATH")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .env("GIT_DIR", outer.join(".git"))
        .output()
        .unwrap_or_else(|error| panic!("the selected Python interpreter is unavailable: {error}"));
    let out = stdout(&output);
    assert!(
        output.status.success() && out.contains("1 passed"),
        "the git-fixture test must pass under an inherited GIT_DIR\nstdout:\n{out}\nstderr:\n{}",
        stderr(&output)
    );
    assert_eq!(
        String::from_utf8_lossy(&fs::read(outer.join(".git/config")).expect("outer config")),
        String::from_utf8_lossy(&config_before),
        "a fixture git command rewrote the inherited repository's config"
    );
    assert_eq!(
        stdout(&git_in(&outer, &["rev-parse", "HEAD"])),
        head_before,
        "a fixture commit landed on the inherited repository's HEAD"
    );
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
    let specifications: Vec<&[usize]> = raw_specs
        .as_chunks::<9>()
        .0
        .iter()
        .map(<[usize; 9]>::as_slice)
        .collect();
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
    assert_eq!(
        rejected.status.code(),
        Some(2),
        "unsafe planner manifest returned an unexpected status\nstdout:\n{}\nstderr:\n{}",
        stdout(&rejected),
        stderr(&rejected),
    );
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

#[test]
fn regular_bank_promotion_enforces_the_sealed_flight_event_contract() {
    let temporary = TestDirectory::new("regular-bank-flight-contract");
    let output = run_repository_code(
        r#"
import copy
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

repository = Path(os.environ["NY_REPOSITORY_ROOT"])
script = repository / "scripts" / "regular_bank_evidence.py"
spec = importlib.util.spec_from_file_location("regular_bank_evidence_cargo", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

solver_environment = {"NY_NO_CUDA": "1", "PATH": "/usr/bin:/bin"}
predicate = (
    "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
    "NY_PGD_FAST_GRAD.0=ran"
)
search_space_payload = {
    "schema": "ny-levers/search-space/v6",
    "axes": [
        {
            "name": "NY_PGD_FAST_GRAD",
            "domain": "bool",
            "values": ["0", "1"],
            "class": "VerdictAffecting",
            "deliver": "preset:attack.pgd_fast_grad",
            "why": "Cargo promotion fixture",
            "absence_baseline": {"kind": "declaration_default"},
            "category_prefixes": [],
            "preset_baseline_keys": ["attack.pgd_fast_grad"],
        }
    ],
    "edges": [],
    "unsafe": [],
    "instrument_only": [],
    "test_only": [],
    "declarations": {
        "NY_PGD_FAST_GRAD": {
            "kind": "Bool",
            "default": "1",
            "min": None,
            "max": None,
            "members": None,
        }
    },
}
search_space_contract = module.provenance._search_space_contract_from_payload(
    search_space_payload, source="Cargo promotion fixture"
)
measurement = dict.fromkeys(module.CURRENT_EXECUTION_MEASUREMENT_KEYS)
measurement.update({
    "result_file": "/tmp/result.txt",
    "flight_record_file": "/tmp/result.txt.flight.json",
    "flight_record_capture": module.FLIGHT_RECORD_CAPTURE_POLICY,
    "execution_evidence_profile": module.provenance.EXECUTION_EVIDENCE_PROFILE,
    "solver_environment": {
        "mode": "env-i-reviewed-record-v1",
        "values": solver_environment,
    },
})
solver_sha256 = "2" * 64
build_fields = {
    "version": "0.3.1",
    "features": "none",
    "backend": "wgpu",
    "markers": "cargo-promotion-fixture-v1",
    "commit": "a" * 40,
    "dirty": "false",
    "revision": "1",
    "source": "clean",
    "profile": "release",
    "target": "fixture-target",
    "rustc": "1.90.0",
    "stamp": "1" * 64,
    "exe": solver_sha256,
}
build_fields["id"] = hashlib.sha256(
    (
        f"{module.provenance.BUILD_ID_SCHEMA}\n"
        f"{build_fields['stamp']}\n{build_fields['exe']}\n"
    ).encode("ascii")
).hexdigest()
build_id = {
    "schema": module.provenance.BUILD_ID_SCHEMA,
    "line": " ".join(
        [module.provenance.BUILD_ID_SCHEMA]
        + [
            f"{field}={build_fields[field]}"
            for field in module.provenance.BUILD_ID_CANONICAL_FIELDS
        ]
    ),
    "fields": build_fields,
}
solver = dict.fromkeys(module.CURRENT_EXECUTION_SOLVER_BINARY_KEYS)
solver.update({
    "declared_build_features": [],
    "sha256": solver_sha256,
    "build_id": build_id,
    "fingerprint": {"mtime_ns": 3_000_000_000},
    "build_coherence": {
        "binary_mtime_epoch": 3,
        "build_inputs_last_commit_epoch": 1,
        "behaviour_inputs_last_commit_epoch": 2,
        "build_input_paths": list(module.provenance._BUILD_INPUT_PATHS),
        "behaviour_input_paths": list(module.provenance._BEHAVIOUR_INPUT_PATHS),
    },
})
start = dict.fromkeys(module.START_KEYS)
start.update({
    "measurement": measurement,
    "solver_binary": solver,
    "environment": {
        "allowlist_schema": "ny_measurement_environment_v1",
        "values": {
            "NY_MEASURE_REQUIRE_FLIGHT_EVENT": predicate,
        },
        "typed_values": {
            "NY_MEASURE_REQUIRE_FLIGHT_EVENT":
                module.provenance.parse_execution_predicate_requirement(predicate),
        },
    },
})
measurement["search_space_contract"] = search_space_contract
measurement["execution_predicate_assignment"] = (
    module.provenance.build_execution_predicate_assignment(
        start["environment"], search_space_contract
    )
)
flight_build_stamp = module.provenance.flight_build_stamp_for_binary_build_id(
    build_id
)
record = {
    "schema_version": 3,
    "ambient_env_schema": "ny-cli/ambient-env/v3",
    "build_id": flight_build_stamp,
    "backend_kind": "cpu-only",
    "backend_summary": "Cargo promotion fixture",
    "host": {
        "hostname": "fixture-host",
        "cpu_model": "fixture-cpu",
        "logical_cores": 1,
        "ram_bytes": 1024,
    },
    "category": "cgan2026",
    "budget_secs": 10,
    "ambient_env": {"NY_NO_CUDA": "1"},
    "levers": {
        "status": "resolved",
        "receipt": {
            "schema": "ny-levers/receipt/v3",
            "lever_count": 2,
            "env_present": 1,
            "env_accepted": 1,
            "env_rejected": 0,
            "env_ignored": 0,
            "levers": [
                {
                    "name": "NY_NO_CUDA",
                    "value": True,
                    "source": "legacy_env",
                    "bucket": "debug",
                    "moat": "high",
                    "provenance": "unmeasured",
                    "env_utf8": True,
                },
                {
                    "name": "NY_PGD_FAST_GRAD",
                    "value": False,
                    "source": "config",
                    "bucket": "debug",
                    "moat": "high",
                    "provenance": "unmeasured",
                },
            ],
        },
    },
    "events": [
        {
            "method": "pgd_fast_gradient_execution",
            "status": "ran",
            "reason": (
                "schema=ny-cli/pgd-fast-gradient-execution/v1 "
                "phase=terminal route=conjunctive configured=false decisions=1 "
                "fast_attempts=0 fast_selected=0 certified_selected=1 "
                "spsa_selected=0"
            ),
            "at_secs": 0.5,
        },
        {
            "method": "run_complete",
            "status": "complete",
            "reason": "sat",
            "at_secs": 1.0,
        },
    ],
}
source = module._flight_record_source_bytes(record)
capture = {
    "status": "captured",
    "source_sha256": module.sha256(source),
    "size_bytes": len(source),
    "record": record,
}
metadata = dict.fromkeys(module.METADATA_KEYS)
metadata.update({
    "category": "cgan2026",
    "timeout_seconds": 10,
    "solver_verdict": "sat",
    "flight_record": capture,
})

assert module.validate_flight_record_binding(start=start, metadata=metadata) == (
    module.CURRENT_EXECUTION_START_PROFILE
)
print("ACCEPT=matching")

def expect_rejected(
    label, candidate_start, candidate_metadata, message, *, exact=False
):
    try:
        module.validate_flight_record_binding(
            start=candidate_start,
            metadata=candidate_metadata,
        )
    except module.EvidenceError as error:
        observed = str(error)
        assert (observed == message if exact else message in observed), (
            label,
            observed,
        )
        print("REJECT=" + label)
    else:
        raise AssertionError(label + " unexpectedly passed promotion validation")

dark = copy.deepcopy(metadata)
dark_record = dark["flight_record"]["record"]
dark_record["levers"]["receipt"]["levers"][1]["value"] = True
dark_record["events"][0]["reason"] = (
    "schema=ny-cli/pgd-fast-gradient-execution/v1 "
    "phase=terminal route=conjunctive configured=true decisions=1 "
    "fast_attempts=1 fast_selected=1 certified_selected=0 spsa_selected=0"
)
dark_source = module._flight_record_source_bytes(dark_record)
dark["flight_record"]["source_sha256"] = module.sha256(dark_source)
dark["flight_record"]["size_bytes"] = len(dark_source)
expect_rejected(
    "dark",
    start,
    dark,
    "sealed required flight event did not earn current execution credit: "
    "governed NY_PGD_FAST_GRAD=0 was not accepted exactly from its sealed "
    "config or child-environment authority",
    exact=True,
)

malformed = copy.deepcopy(start)
malformed["environment"]["values"]["NY_MEASURE_REQUIRE_FLIGHT_EVENT"] = (
    predicate + "=extra"
)
expect_rejected("malformed", malformed, metadata, "current execution predicate")

legacy_credit = copy.deepcopy(start)
legacy_credit["environment"]["values"]["NY_MEASURE_REQUIRE_FLIGHT_EVENT"] = (
    "cgan_input_leaf_summary=ran"
)
legacy_credit["environment"]["typed_values"]["NY_MEASURE_REQUIRE_FLIGHT_EVENT"] = {
    "type": "flight_event_requirement",
    "method": "cgan_input_leaf_summary",
    "status": "ran",
}
expect_rejected(
    "legacy-credit",
    legacy_credit,
    metadata,
    "execution predicate assignment differs from its typed requirement",
)

config_only = copy.deepcopy(metadata)
config_event = config_only["flight_record"]["record"]["events"][0]
config_event["status"] = "skipped"
config_event["reason"] = (
    "schema=ny-cli/pgd-fast-gradient-execution/v1 "
    "phase=terminal route=conjunctive configured=false decisions=0 "
    "fast_attempts=0 fast_selected=0 certified_selected=0 spsa_selected=0"
)
config_source = module._flight_record_source_bytes(
    config_only["flight_record"]["record"]
)
config_only["flight_record"]["source_sha256"] = module.sha256(config_source)
config_only["flight_record"]["size_bytes"] = len(config_source)
expect_rejected(
    "config-only", start, config_only, "did not prove live engagement"
)

bad_values = copy.deepcopy(start)
bad_values["environment"]["values"] = []
expect_rejected(
    "bad-values", bad_values, metadata, "sealed assignment environment is incomplete"
)

missing = copy.deepcopy(metadata)
missing["flight_record"] = {"status": "missing"}
expect_rejected("missing", start, missing, "flight record is missing")

legacy = copy.deepcopy(start)
legacy["measurement"].pop("execution_evidence_profile")
legacy["measurement"].pop("execution_predicate_assignment")
legacy["measurement"].pop("search_space_contract")
legacy["environment"]["values"] = {}
legacy["environment"]["typed_values"] = {}
assert module.validate_flight_record_binding(start=legacy, metadata=metadata) == (
    module.FLIGHT_BUILD_ID_START_PROFILE
)
print("ACCEPT=no-contract")
"#,
        temporary.path(),
    );
    assert_success(&output, "regular-bank flight-event promotion contract");
    let captured = stdout(&output);
    for marker in [
        "ACCEPT=matching",
        "REJECT=dark",
        "REJECT=malformed",
        "REJECT=legacy-credit",
        "REJECT=config-only",
        "REJECT=bad-values",
        "REJECT=missing",
        "ACCEPT=no-contract",
    ] {
        assert!(
            captured.lines().any(|line| line == marker),
            "missing {marker:?} in promotion-contract output: {captured}"
        );
    }
}

#[test]
fn measured_2026_coverage_census_matches_the_committed_complete_banks() {
    let root = workspace_root();
    let output = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &["--check".to_owned()],
        &root,
    );
    assert_success(&output, "measured-2026 coverage census check");
    let captured = stdout(&output);
    assert!(
        captured.contains("reports/measured-2026/README.md matches ")
            && captured.contains(" complete result banks"),
        "coverage check did not report its complete-bank count: {captured}"
    );
}

fn measured_2026_index_fixture(
    label: &str,
    complete_bank: &str,
) -> (TestDirectory, PathBuf, Vec<String>) {
    let temporary = TestDirectory::new(label);
    let report_dir = temporary.path().join("measured-2026");
    let readme = report_dir.join("README.md");
    let manifest = report_dir.join("bank_manifest.json");
    write(
        &readme,
        "# fixture\n\n<!-- BEGIN GENERATED: measured-2026 coverage census -->\n\
         stale\n<!-- END GENERATED: measured-2026 coverage census -->\n",
    );
    write(&report_dir.join("fixture_complete.csv"), complete_bank);
    write(
        &manifest,
        r#"{
  "schema": "ny_measured_2026_bank_manifest_v1",
  "source_repository": "fixture",
  "source_commit": "cccccccccccccccccccccccccccccccccccccccc",
  "scored_categories": ["fixture_complete", "fixture_later"],
  "banks": {
    "fixture_complete": {
      "instances_csv": "benchmarks/fixture_complete/instances.csv",
      "instances_csv_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "bank_csv_sha256": "d856616765b5fadeec63a52fbdb03c11cde37cc94edda75a6c80945c1cee4835",
      "instances": [
        ["onnx/repeated.onnx", "vnnlib/repeated.vnnlib", "10"],
        ["onnx/repeated.onnx", "vnnlib/repeated.vnnlib", "10"]
      ]
    },
    "fixture_later": {
      "instances_csv": "benchmarks/fixture_later/instances.csv",
      "instances_csv_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "bank_csv_sha256": "49c148631b9041dce213ae24e118a4e325cafce5b51b75560ee59b8bbd21dcb6",
      "instances": [["onnx/later.onnx", "vnnlib/later.vnnlib", "5"]]
    }
  }
}
"#,
    );
    let arguments = vec![
        "--report-dir".to_owned(),
        report_dir.display().to_string(),
        "--readme".to_owned(),
        readme.display().to_string(),
        "--manifest".to_owned(),
        manifest.display().to_string(),
    ];
    (temporary, report_dir, arguments)
}

#[test]
fn measured_2026_coverage_census_detects_a_new_pinned_complete_bank() {
    let (temporary, report_dir, arguments) = measured_2026_index_fixture(
        "measured-2026-index",
        "fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,sat,1.0\n\
         fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,timeout,10.0\n",
    );
    let readme = report_dir.join("README.md");

    let updated = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &arguments,
        temporary.path(),
    );
    assert_success(&updated, "initial measured-2026 coverage census update");
    let census = fs::read_to_string(&readme).expect("generated README must be readable");
    assert!(census.contains("Complete banks cover **1 of 2**"));
    assert!(census.contains("1 sat, 1 timeout | 1 / 2 (50.0%)"));

    // Move, don't clone: this is the last use of `arguments` in the test, and
    // stock clippy's `redundant_clone` is denied by the publish gate.
    let mut check_arguments = arguments;
    check_arguments.push("--check".to_owned());
    let current = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &check_arguments,
        temporary.path(),
    );
    assert_success(&current, "unchanged measured-2026 coverage census check");

    write(
        &report_dir.join("fixture_later.csv"),
        "fixture_later,onnx/later.onnx,vnnlib/later.vnnlib,unsat,1.0\n",
    );
    let stale = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &check_arguments,
        temporary.path(),
    );
    assert_eq!(stale.status.code(), Some(1));
    assert!(stderr(&stale).contains("STALE:"));
    assert!(stderr(&stale).contains("does not match the complete result banks"));
}

#[test]
fn measured_2026_coverage_census_rejects_a_truncated_occurrence_bank() {
    let (temporary, _report_dir, mut arguments) = measured_2026_index_fixture(
        "measured-2026-truncated",
        "fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,sat,1.0\n",
    );
    arguments.push("--check".to_owned());

    let rejected = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &arguments,
        temporary.path(),
    );
    assert_eq!(rejected.status.code(), Some(2));
    assert!(stderr(&rejected).contains("pinned instances.csv has 2 occurrences"));
}

#[test]
fn measured_2026_coverage_census_rejects_an_over_budget_row() {
    let (temporary, _report_dir, mut arguments) = measured_2026_index_fixture(
        "measured-2026-over-budget",
        "fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,sat,1.0\n\
         fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,timeout,10.001\n",
    );
    arguments.push("--check".to_owned());

    let rejected = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &arguments,
        temporary.path(),
    );
    assert_eq!(rejected.status.code(), Some(2));
    assert!(stderr(&rejected).contains("exceeds official budget 10 s"));
}

#[test]
fn measured_2026_coverage_census_rejects_a_wrong_pinned_occurrence() {
    let (temporary, _report_dir, mut arguments) = measured_2026_index_fixture(
        "measured-2026-wrong-occurrence",
        "fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,sat,1.0\n\
         fixture_complete,onnx/different.onnx,vnnlib/repeated.vnnlib,timeout,2.0\n",
    );
    arguments.push("--check".to_owned());

    let rejected = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &arguments,
        temporary.path(),
    );
    assert_eq!(rejected.status.code(), Some(2));
    assert!(stderr(&rejected).contains("instance occurrence 2"));
    assert!(stderr(&rejected).contains("does not match the pinned instances.csv row"));
}

#[test]
fn measured_2026_coverage_census_rejects_an_unpinned_new_bank() {
    let (temporary, report_dir, mut arguments) = measured_2026_index_fixture(
        "measured-2026-unpinned",
        "fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,sat,1.0\n\
         fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,timeout,2.0\n",
    );
    write(
        &report_dir.join("unreviewed.csv"),
        "unreviewed,onnx/new.onnx,vnnlib/new.vnnlib,unsat,1.0\n",
    );
    arguments.push("--check".to_owned());

    let rejected = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &arguments,
        temporary.path(),
    );
    assert_eq!(rejected.status.code(), Some(2));
    assert!(stderr(&rejected).contains("lack a pinned official instances.csv contract"));
    assert!(stderr(&rejected).contains("unreviewed"));
}

#[test]
fn measured_2026_coverage_census_rejects_an_unreviewed_bank_byte_change() {
    let (temporary, _report_dir, mut arguments) = measured_2026_index_fixture(
        "measured-2026-bank-digest",
        "fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,unsat,1.0\n\
         fixture_complete,onnx/repeated.onnx,vnnlib/repeated.vnnlib,timeout,10.0\n",
    );
    arguments.push("--check".to_owned());

    let rejected = run_repository_script(
        "scripts/update_measured_2026_index.py",
        &arguments,
        temporary.path(),
    );
    assert_eq!(rejected.status.code(), Some(2));
    assert!(stderr(&rejected).contains("bank CSV sha256"));
    assert!(stderr(&rejected).contains("review the result change"));
}

// ---------------------------------------------------------------------------
// The M0 arming harness (scripts/m0/): its self-test with the runner under the
// host's STOCK shell, /bin/bash.
//
// macOS ships bash 3.2 as /bin/bash. The runner, moat_both_profiles.sh, used
// bash 4's `mapfile` and `declare -A`, so every one of its end-to-end tests
// failed on that shell while passing under Linux's bash 5, and the failure
// read as a host quirk. bash 3.2 does not merely reject `declare -A`: the
// assignment then runs on an INDEXED array whose string subscripts evaluate
// arithmetically (docs/DEFECT_REGISTER_2026-07-31.md D1 `#bash32-declare-A`,
// which sent nine A/B runs to one instance). The contract here runs the
// self-test with `--bash /bin/bash` and requires every test's `ok` by name, so
// on a host whose stock shell is bash 3.2 either construct is a red. On Linux
// /bin/bash is bash 5 and this run cannot see them; the lexical scan that
// refuses bash-4-only constructs on EVERY host, including in the Linux-only
// `--launch` path the self-test cannot execute, is in the default lane:
// `source_policy`'s `m0_runner_has_no_bash4_only_construct` (gate check [a]).
// The self-test needs requirements.txt's PyYAML, like the other tools here.
// ---------------------------------------------------------------------------

/// Every runner test the self-test's default lane defines. Each one drives
/// `moat_both_profiles.sh --run` end to end, so each one exercises the shell.
const M0_RUNNER_TESTS: &[&str] = &[
    "test_runner_clean_run_exits_zero_and_done_propagates_rc",
    "test_runner_fixture_isolates_inherited_verdict_env",
    "test_runner_stops_before_a_run_while_its_stop_file_exists",
    "test_runner_out_dir_must_live_under_its_root_and_the_root_is_overridable",
    "test_runner_stops_on_an_emergency_with_rc3",
    "test_runner_void_run_is_not_a_pass",
    "test_runner_profile_split_is_an_emergency",
    "test_runner_refuses_ny_env_before_measuring",
    "test_runner_refuses_a_wrong_pin",
    "test_runner_resume_skips_judged_runs",
    "test_runner_runs_under_the_competition_backstop",
    "test_runner_backstop_kill_is_judged_from_the_results_file",
    "test_runner_results_file_unsat_is_an_emergency_when_the_log_says_timeout",
    "test_runner_refuses_to_relaunch_after_an_emergency",
    "test_runner_refuses_verdict_affecting_non_ny_env",
    "test_runner_refuses_an_address_space_rlimit",
    "test_runner_resolver_checks_each_category_configs_preset_key_once",
    "test_runner_moves_a_stale_flight_record_aside",
];

/// Step-0 refusals of THIS HOST's memory envelope in a failed self-test's
/// output: the reason of every `M0MOAT_DONE rc=86` line, in first-seen order.
///
/// 16 of the 18 runner tests reach the runner's real step-0 memory check
/// (scripts/m0/README.md). On Linux it refuses a cgroup `memory.max` below
/// 96 GiB anywhere on the process's cgroup path, and an unreadable one; on
/// every host it refuses a finite `RLIMIT_AS`. A self-test run inside such a
/// cap fails those tests with rc 86 for a reason that has nothing to do with
/// the shell, so the contract below names it instead of leaving it in the
/// transcript. (The one test that expects rc 86 prints nothing when it
/// passes, so a printed rc-86 line is always a refusal a test did not expect.)
fn m0_host_envelope_refusals(transcript: &str) -> Vec<String> {
    let mut reasons: Vec<String> = Vec::new();
    for line in transcript.lines() {
        let Some((_, done)) = line.split_once("M0MOAT_DONE rc=86 ") else {
            continue;
        };
        let Some((_, reason)) = done.split_once("reason=") else {
            continue;
        };
        // `reason=<text, spaces allowed> <ISO timestamp>`
        let reason = reason
            .rsplit_once(' ')
            .map_or(reason, |(text, _timestamp)| text);
        if !reasons.iter().any(|seen| seen == reason) {
            reasons.push(reason.to_owned());
        }
    }
    reasons
}

/// The field value of `key=` in a `M0_SELFTEST_DONE` line.
fn done_field<'a>(done: &'a str, key: &str) -> &'a str {
    let prefix = format!("{key}=");
    done.split_whitespace()
        .find_map(|field| field.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("DONE line lacks {key}=: {done}"))
}

#[test]
fn m0_selftest_default_lane_is_green_with_the_runner_under_stock_bin_bash() {
    let root = workspace_root();
    let output = run_repository_script(
        "scripts/m0/m0_selftest.py",
        &["--bash".to_owned(), "/bin/bash".to_owned()],
        &root,
    );
    let printed = stdout(&output);
    let context = format!(
        "status {}\nstdout:\n{printed}\nstderr:\n{}",
        output.status,
        stderr(&output)
    );
    // A lane that cannot run its contract on this host says exactly why.
    let refusals = m0_host_envelope_refusals(&context);
    assert!(
        refusals.is_empty(),
        "the M0 runner's step-0 memory check refused THIS HOST (rc 86: {}), so its end-to-end \
         tests cannot run here. The refusal is the runner's contract, not a shell-portability \
         red: on Linux a cgroup memory.max below 96 GiB on the process's cgroup path (or an \
         unreadable one) moves ny's dense clamp, and a finite RLIMIT_AS is refused on every \
         host (scripts/m0/README.md). Run this lane outside the cap.\n{context}",
        refusals.join("; ")
    );
    assert!(output.status.success(), "m0 self-test failed: {context}");

    let done = printed
        .lines()
        .rfind(|line| line.starts_with("M0_SELFTEST_DONE "))
        .unwrap_or_else(|| panic!("no M0_SELFTEST_DONE line: {context}"));
    for (key, want) in [
        ("rc", "0"),
        ("failures", "0"),
        ("errors", "0"),
        ("skipped", "0"),
        ("lane", "default"),
        ("bash", "/bin/bash"),
    ] {
        assert_eq!(done_field(done, key), want, "{key} in {done}");
    }
    // The stock macOS shell is the bash 3.2 the runner must support. A newer
    // /bin/bash there would make this check stop exercising 3.2, so say so.
    #[cfg(target_os = "macos")]
    assert!(
        done_field(done, "bash_version").starts_with("3."),
        "/bin/bash on macOS is no longer bash 3.x, so this check no longer covers it: {done}"
    );

    // Every test line, by name: `ran=` must equal the `... ok` lines, so a
    // test that vanished or reported anything but ok cannot hide in a count.
    let test_lines: Vec<&str> = printed
        .lines()
        .filter(|line| line.starts_with("test_") && line.contains(" (__main__."))
        .collect();
    let ran: usize = done_field(done, "ran")
        .parse()
        .unwrap_or_else(|error| panic!("ran= is not a count ({error}): {done}"));
    assert_eq!(test_lines.len(), ran, "one result line per test: {context}");
    for line in &test_lines {
        assert!(line.ends_with(" ... ok"), "not ok: {line}");
    }
    for name in M0_RUNNER_TESTS {
        let prefix = format!("{name} (__main__.Runner.{name}) ... ok");
        assert!(
            test_lines.iter().any(|line| *line == prefix),
            "runner test {name} did not report ok: {context}"
        );
    }
}

#[test]
fn m0_host_envelope_refusal_is_named_from_a_failed_selftest_transcript() {
    // Positive control for the refusal the self-test contract names: the two
    // shapes a failing runner test prints (the DONE line alone, as an
    // assertion message; and inside the runner's log tail), one reason per
    // distinct refusal, the timestamp dropped, other DONE lines untouched.
    let transcript = "\
AssertionError: 86 != 0 : M0MOAT_DONE rc=86 WORST=86 reason=rlimit-as-set:1024(KiB; refused on every host) 2026-10-08T13:53:39-07:00
  cgroup: 0::/user.slice/x.scope memory.max=68719476736
M0MOAT_DONE rc=86 WORST=86 reason=cgroup-memory-cap-below-96G:68719476736 2026-10-08T13:53:40-07:00
M0MOAT_DONE rc=86 WORST=86 reason=cgroup-memory-cap-below-96G:68719476736 2026-10-08T13:53:41-07:00
M0MOAT_DONE rc=97 WORST=97 reason=spec has no rows 2026-10-08T13:53:42-07:00
M0MOAT_DONE rc=0 WORST=0 ran=2 box_refused=0 2026-10-08T13:53:43-07:00
";
    assert_eq!(
        m0_host_envelope_refusals(transcript),
        [
            "rlimit-as-set:1024(KiB; refused on every host)",
            "cgroup-memory-cap-below-96G:68719476736",
        ]
    );
    assert!(m0_host_envelope_refusals("M0_SELFTEST_DONE rc=0 ran=59 failures=0").is_empty());
}
