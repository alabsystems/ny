// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LEGACY WELL-FORMEDNESS on the REAL scored entry point (`ny vnncomp v1`),
//! the `router_acting.rs` pattern: the tracked `tests/models/simple_mlp.onnx`,
//! a temporary `configs/vnncomp99/<category>.yaml`, no `benchmarks/`.
//!
//! Both properties are impossible for the network (`Y_0 <= -1e6` over the unit
//! box; the second splits the box into two per-clause input boxes), so the
//! published verdict is a deterministic `unsat`, and on this fixture it comes
//! through a LEGACY door with no lane tape (the sequential engine,
//! `[cover-gate] legacy-uncovered authority=beta_crown_verdict`). On the ENG14
//! pin (`ny_eng14_8c2100b7f`) both bundles were `header, verdict(required=[]),
//! published, end` and `ny proof-check` refused them `malformed`
//! (`missing_sites`, exit 30 — the soundness-emergency code). Here:
//!
//! * COMPETITION IS BIT-IDENTICAL: the default (Competition) run writes no
//!   bundle at all, and its RESULTS_FILE is byte-identical to the Proof run's;
//! * the Proof bundle is WELL-FORMED (`sites`, `clauses`, one whole-region
//!   claim per clause naming the authority, a `verdict` requiring every
//!   clause) and the checker reports it `unconfirmed` (exit 10, the classic
//!   sub-moat), never `malformed`: `ny-proof-bundle/v2` for the renderable
//!   property, `ny-proof-bundle/v2.1` with the `unavailable` clause for the
//!   per-clause-box property the producer cannot render.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ny_test_utils::workspace_root;
use serde_json::Value;

const CATEGORY: &str = "proof_bundle_legacy_2026";

const CONJ: &str = "(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 -1000000.0))
";

const BOXES: &str = "(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (or (and (>= X_0 -1.0) (<= X_0 0.0) (>= X_1 -1.0) (<= X_1 1.0) (<= Y_0 -1000000.0)) \
(and (>= X_0 0.0) (<= X_0 1.0) (>= X_1 -1.0) (<= X_1 1.0) (<= Y_0 -1000000.0))))
";

fn ny() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ny"));
    // As in `router_shadow.rs`: no lingering out-of-process watchdog.
    c.current_dir(workspace_root())
        .env("NY_VNNCOMP_EXTERNAL_WATCHDOG", "0");
    c
}

fn with_suffix(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// One scored run: `profile = None` is the default (Competition).
fn run(
    dir: &Path,
    configs: &Path,
    vnnlib: &Path,
    tag: &str,
    profile: Option<&str>,
) -> (Output, PathBuf) {
    let results = dir.join(format!("{tag}.txt"));
    let mut c = ny();
    c.args(["vnncomp", "v1", CATEGORY])
        .arg(workspace_root().join("tests/models/simple_mlp.onnx"))
        .arg(vnnlib)
        .arg(&results)
        .arg("5")
        .arg("--configs-dir")
        .arg(configs);
    if let Some(p) = profile {
        c.args(["--proof-profile", p]);
    }
    let out = c.output().expect("run ny vnncomp v1");
    assert!(
        out.status.success(),
        "{tag}: ny vnncomp v1 failed: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    (out, results)
}

fn lines(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("bundle")
        .lines()
        .map(|l| serde_json::from_str(l).expect("every bundle line is JSON"))
        .collect()
}

#[ntest::timeout(600000)]
#[test]
fn legacy_lane_bundles_are_well_formed_and_competition_stays_bit_identical() {
    let root = tempfile::tempdir().expect("tempdir");
    let configs = root.path().join("configs");
    fs::create_dir_all(configs.join("vnncomp99")).expect("config tree");
    fs::write(
        configs.join("vnncomp99").join(format!("{CATEGORY}.yaml")),
        "",
    )
    .expect("empty preset");
    for (tag, text, schema, extra_code) in [
        ("conj", CONJ, "ny-proof-bundle/v2", None),
        (
            "boxes",
            BOXES,
            "ny-proof-bundle/v2.1",
            Some("clauses_unavailable"),
        ),
    ] {
        let vnnlib = root.path().join(format!("{tag}.vnnlib"));
        fs::write(&vnnlib, text).expect("vnnlib");
        let (_, comp) = run(root.path(), &configs, &vnnlib, &format!("{tag}_comp"), None);
        let (proof_out, proof) = run(
            root.path(),
            &configs,
            &vnnlib,
            &format!("{tag}_proof"),
            Some("proof"),
        );
        let stderr = String::from_utf8_lossy(&proof_out.stderr).into_owned();
        // The lane must BE a legacy door (else this test is VOID).
        let authority = stderr
            .lines()
            .find_map(|l| l.strip_prefix("[cover-gate] legacy-uncovered authority="))
            .and_then(|rest| rest.split_whitespace().next())
            .unwrap_or_else(|| panic!("{tag}: no legacy door published (VOID):\n{stderr}"))
            .to_owned();
        // Competition is bit-identical and writes nothing.
        let comp_bytes = fs::read(&comp).expect("competition result");
        assert!(comp_bytes.starts_with(b"unsat"), "{tag}: {comp_bytes:?}");
        assert_eq!(
            comp_bytes,
            fs::read(&proof).expect("proof result"),
            "{tag}: the Proof profile changed the published RESULTS_FILE"
        );
        for suffix in [".proof.jsonl", ".proof.jsonl.partial"] {
            assert!(
                !with_suffix(&comp, suffix).exists(),
                "{tag}: Competition wrote {suffix}"
            );
        }
        // The Proof bundle is written and well-formed.
        let artifact = stderr
            .lines()
            .find(|l| l.starts_with("[proof-artifact] artifact="))
            .unwrap_or_else(|| panic!("{tag}: no artifact line:\n{stderr}"));
        assert!(
            artifact.starts_with("[proof-artifact] artifact=written ")
                && artifact.contains(&format!("schema={schema} claims=")),
            "{tag}: {artifact}"
        );
        let bundle = with_suffix(&proof, ".proof.jsonl");
        let body = lines(&bundle);
        let kinds: Vec<&str> = body
            .iter()
            .map(|l| l["kind"].as_str().unwrap_or("?"))
            .collect();
        assert_eq!(body[0]["schema"], schema, "{tag}: {body:?}");
        for kind in ["sites", "clauses", "node", "verdict", "published", "end"] {
            assert!(kinds.contains(&kind), "{tag}: no `{kind}`: {kinds:?}");
        }
        let verdict = body
            .iter()
            .find(|l| l["kind"] == "verdict")
            .expect("verdict");
        assert_eq!(verdict["source"], authority.as_str(), "{tag}");
        let required: Vec<u64> = verdict["required"]
            .as_array()
            .expect("required")
            .iter()
            .filter_map(Value::as_u64)
            .collect();
        assert!(!required.is_empty(), "{tag}: the verdict requires nothing");
        let route = format!("legacy:{authority}");
        for k in &required {
            assert!(
                body.iter().any(|l| l["kind"] == "node"
                    && l["src"] == route.as_str()
                    && l["lits"] == serde_json::json!([])
                    && l["dis"] == serde_json::json!([k])),
                "{tag}: clause {k} has no whole-region `{route}` claim: {body:?}"
            );
        }
        // The independent checker: well-formed, unconfirmed (exit 10).
        let out = ny()
            .arg("proof-check")
            .arg(&bundle)
            .args(["--threads", "2"])
            .output()
            .expect("run ny proof-check");
        let check_err = String::from_utf8_lossy(&out.stderr).into_owned();
        let done = check_err
            .lines()
            .find(|l| l.starts_with("PROOF_CHECK_DONE rc="))
            .unwrap_or("")
            .to_owned();
        assert_eq!(out.status.code(), Some(10), "{tag}: {done}\n{check_err}");
        assert!(
            done.starts_with("PROOF_CHECK_DONE rc=10 verdict=unconfirmed "),
            "{tag}: {done}"
        );
        if let Some(code) = extra_code {
            let report: Value = serde_json::from_slice(
                &fs::read(with_suffix(&bundle, ".check.json")).expect("report"),
            )
            .expect("report json");
            assert!(
                report["reasons"]
                    .as_array()
                    .is_some_and(|r| r.iter().any(|x| x["code"] == code)),
                "{tag}: {report}"
            );
        }
        eprintln!("[legacy-cli] {tag} authority={authority} {done}");
    }
}
