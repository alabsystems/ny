// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cargo-owned contracts for replay and evidence tools that remain Python.
//!
//! The assertions live here so `cargo test` sees these contracts.  The child
//! interpreter imports only Python's standard library and repository modules;
//! model execution belongs to the explicit external replay lane.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn python_executable() -> OsString {
    std::env::var_os("NY_TEST_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn run_python(program: &str) -> String {
    let root = repository_root();
    let output = Command::new(python_executable())
        .args(["-I", "-B", "-c", program])
        .current_dir(&root)
        .env_remove("NY_TEST_PYTHON")
        .env("NY_REPOSITORY_ROOT", &root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .unwrap_or_else(|error| panic!("the selected Python interpreter is unavailable: {error}"));

    assert!(
        output.status.success(),
        "Python evidence-tool contract failed (status {}):\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    // NORMALIZE LINE ENDINGS. Python's `print` terminates with the platform
    // newline, so every line of this stdout arrives CRLF-terminated on Windows
    // while every assertion below is written against "\n" — `contains("...\n")`
    // then matches nothing and all five contracts fail for a reason that has
    // nothing to do with what they check. These assertions are about the tool's
    // CONTENT, not its line-ending convention, so the convention is normalized
    // away here at the single point where output is captured.
    String::from_utf8(output.stdout)
        .expect("Python contract output must be UTF-8")
        .replace("\r\n", "\n")
}

#[test]
fn canonicalizer_preserves_x_and_rejects_malformed_assignments() {
    let stdout = run_python(
        r#"
import os
import sys
from pathlib import Path

root = Path(os.environ["NY_REPOSITORY_ROOT"])
sys.path.insert(0, str(root / "scripts"))
import canonicalize_vnncomp_ce_y as tool

source = b"sat\n((X_0 -0.0)\n(X_1 1.0000000000000002)\n(Y_0 9)\n(Y_1 -4))\n"
assignment = tool.parse_sat_result(source)
print("X=" + "|".join(assignment.x_tokens))
print(tool.render_sat_result(assignment, [1.5, -2.25]).decode(), end="")

bad_inputs = (
    b"unsat\n",
    b"sat\n((X_1 0)(Y_0 0))\n",
    b"sat\n((X_0 nan)(Y_0 0))\n",
    b"sat\n((X_0 0)(Y_0 0)(X_1 1))\n",
    b"sat\n((X_0 0)(Y_0 0)garbage)\n",
)
errors = []
for value in bad_inputs:
    try:
        tool.parse_sat_result(value)
    except tool.CanonicalizationError as error:
        errors.append(type(error).__name__)
    else:
        errors.append("ACCEPTED")
print("ERRORS=" + "|".join(errors))
"#,
    );

    assert!(stdout.contains("X=-0.0|1.0000000000000002\n"));
    assert!(stdout.contains("(X_0 -0.0)\n(X_1 1.0000000000000002)"));
    assert!(stdout.contains("(Y_0 1.5)\n(Y_1 -2.25)"));
    assert!(stdout.contains(
        "ERRORS=CanonicalizationError|CanonicalizationError|CanonicalizationError|CanonicalizationError|CanonicalizationError\n"
    ));
}

#[test]
fn falsification_audit_reads_both_bank_schemas_and_mixed_bounds() {
    let stdout = run_python(
        r#"
import os
import sys
import tempfile
from pathlib import Path

root = Path(os.environ["NY_REPOSITORY_ROOT"])
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "scripts" / "extended_bank"))
import audit_unsat_by_falsification as audit

with tempfile.TemporaryDirectory() as temporary:
    bank = Path(temporary)
    (bank / "wide.csv").write_text(
        "acasxu_2023,onnx/a.onnx,vnnlib/p.vnnlib,prepared,unsat,0.55,run-1\n"
        "acasxu_2023,onnx/test_nano.onnx,vnnlib/test_nano.vnnlib,prepared,unsat,0\n",
        encoding="utf-8",
    )
    (bank / "narrow.csv").write_text(
        "cat,onnx,vnnlib,verdict,secs\n"
        "vit_2023,onnx/v.onnx,vnnlib/v.vnnlib,unsat,9.8\n",
        encoding="utf-8",
    )
    rows = audit.read_bank([bank])
    print("ROWS=" + repr([(r.category, r.onnx, r.verdict, r.seconds) for r in rows]))

    malformed = bank / "malformed"
    malformed.mkdir()
    (malformed / "bad.csv").write_text(
        "vit_2023,onnx/good.onnx,vnnlib/good.vnnlib,unsat,1.0\n"
        "vit_2023,onnx/bad.onnx,vnnlib/bad.vnnlib,prepared,unsat,1.0,run,extra\n",
        encoding="utf-8",
    )
    try:
        audit.read_bank([malformed])
    except audit.BankFormatError as error:
        print("MALFORMED=" + str(error).replace("\n", "|"))
    else:
        print("MALFORMED=ACCEPTED")

mixed = ["and", [">=", "X_0", "0.1"], ["<=", "X_0", "0.2"], ["<=", "Y_0", "0.5"]]
print("SIMPLE=" + repr(audit.simple_input_bounds(mixed)))
print("WITHIN=" + repr(audit._input_bounds_within(mixed)))
"#,
    );

    assert!(stdout.contains(
        "ROWS=[('vit_2023', 'onnx/v.onnx', 'unsat', '9.8'), ('acasxu_2023', 'onnx/a.onnx', 'unsat', '0.55')]"
    ));
    assert!(stdout.contains("MALFORMED=bank rows could not be parsed:"));
    assert!(stdout.contains("unsupported 8-column row"));
    assert!(!stdout.contains("MALFORMED=ACCEPTED"));
    assert!(stdout.contains("SIMPLE=None\n"));
    assert!(stdout.contains("WITHIN=[(0, '>=', 0.1), (0, '<=', 0.2)]\n"));
}

#[test]
fn vnncomp_2025_replay_metrics_are_exact_and_fail_closed() {
    let stdout = run_python(
        r#"
import os
import sys
from pathlib import Path

root = Path(os.environ["NY_REPOSITORY_ROOT"])
sys.path.insert(0, str(root / "scripts"))
import replay_vnncomp2025_counterexample as replay

message = (
    "L-inf norm difference between onnx execution and CE file output: "
    "5.5e-07 (rel error: 2.25E-06);(rel_limit: 0.001)\n"
    "Checking if spec was actually violated"
)
response = {
    "result": "correct",
    "message": message,
    "diff": 5.5e-7,
    "rel_error": 2.25e-6,
}
print("METRICS=" + repr(replay._parse_official_metrics(message)))
print("RESULT=" + replay._validate_machine_response(response)["result"])

bad = dict(response)
bad["extra"] = "self-attestation"
try:
    replay._validate_machine_response(bad)
except replay.ReplayError as error:
    print("EXTRA=" + str(error))
else:
    print("EXTRA=ACCEPTED")

try:
    replay._parse_official_metrics("counterexample accepted without metrics")
except replay.ReplayError as error:
    print("MISSING=" + str(error))
else:
    print("MISSING=ACCEPTED")
"#,
    );

    assert!(stdout.contains("METRICS=(5.5e-07, 2.25e-06)\n"));
    assert!(stdout.contains("RESULT=correct\n"));
    assert!(
        stdout.contains("EXTRA=official machine response does not have the exact canonical keys")
    );
    assert!(stdout.contains("MISSING="));
    assert!(!stdout.contains("ACCEPTED"));
}

#[test]
fn ny_replay_preserves_the_submitted_assignment_boundary() {
    let stdout = run_python(
        r#"
import os
import sys
from pathlib import Path

root = Path(os.environ["NY_REPOSITORY_ROOT"])
sys.path.insert(0, str(root / "scripts"))
import replay_ny_counterexamples as replay

raw = b"sat\n((X_0 -0.0)\n(Y_0 1.0))\n"
assignment = replay._extract_assignment(raw)
print("ASSIGNMENT=" + assignment.decode().replace("\n", "|"))
print("V1=" + replay._infer_vnnlib_version(Path("/bench/vnnlib/property.vnnlib")))
print("V2=" + replay._infer_vnnlib_version(Path("/bench/2.0/vnnlib/property.vnnlib")))

for label, value in (("VERDICT", b"unsat\n"), ("EMPTY", b"sat\n")):
    try:
        replay._extract_assignment(value)
    except replay.ReplayError as error:
        print(label + "=" + str(error))
    else:
        print(label + "=ACCEPTED")
"#,
    );

    assert!(stdout.contains("ASSIGNMENT=((X_0 -0.0)|(Y_0 1.0))|\n"));
    assert!(stdout.contains("V1=1.0\n"));
    assert!(stdout.contains("V2=2.0\n"));
    assert!(stdout.contains("VERDICT=raw result does not start with a standalone SAT verdict\n"));
    assert!(stdout.contains("EMPTY=raw SAT result has no assignment\n"));
    assert!(!stdout.contains("ACCEPTED"));
}

#[test]
fn falsified_watchlist_loader_rejects_schema_drift() {
    let stdout = run_python(
        r#"
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(os.environ["NY_REPOSITORY_ROOT"])
sys.path.insert(0, str(root / "scripts"))
import emit_ce_falsified_watchlist as watchlist

with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "watchlist.json"
    payload = {
        "version": watchlist.WATCHLIST_VERSION,
        "categories": {
            "cifar100_2024": [
                {"onnx": "onnx/net.onnx", "vnnlib": "vnnlib/prop.vnnlib", "occurrence": 2}
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = watchlist.load(path)
    print("ROWS=" + repr(sorted(loaded["cifar100_2024"])))

    path.write_text(json.dumps({"version": 999, "categories": {}}), encoding="utf-8")
    try:
        watchlist.load(path)
    except ValueError as error:
        print("SCHEMA=" + str(error))
    else:
        print("SCHEMA=ACCEPTED")

    print("ABSENT=" + repr(watchlist.load(Path(temporary) / "absent.json")))
"#,
    );

    assert!(stdout.contains("ROWS=[('onnx/net.onnx', 'vnnlib/prop.vnnlib', 2)]\n"));
    assert!(stdout.contains("SCHEMA="));
    assert!(stdout.contains("is not the expected 1"));
    assert!(stdout.contains("ABSENT={}\n"));
    assert!(!stdout.contains("ACCEPTED"));
}
