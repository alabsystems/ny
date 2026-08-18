// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scored-path contract for flight-v3 layered registry receipts.
//!
//! Unit tests pin the resolver matrix. This test pins the integration seam:
//! `ny vnncomp` must load and validate its exact typed preset, layer the raw
//! entry-time environment over it, and serialize that same resolved alpha
//! input. Runtime authority for other entries arrives as their readers migrate.

use ny_test_utils::workspace_root;
use serde_json::Value;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::process::Command;

const PRESET_CATEGORY: &str = "flight_receipt_preset_2026";
const DEFAULT_CATEGORY: &str = "flight_receipt_default_2026";
const INVALID_CATEGORY: &str = "flight_receipt_invalid_2026";
const RELATIONAL_CATEGORY: &str = "isomorphic_acasxu_2026";

fn write_fixture(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let configs = root.join("configs");
    let year = configs.join("vnncomp99");
    fs::create_dir_all(&year).expect("create temporary config tree");
    fs::write(
        year.join(format!("{PRESET_CATEGORY}.yaml")),
        "solver:\n  alpha_crown:\n    alpha_zero_yield_frac: 0.25\n",
    )
    .expect("write typed alpha preset");
    fs::write(
        year.join(format!("{INVALID_CATEGORY}.yaml")),
        "solver:\n  alpha_crown:\n    alpha_zero_yield_frac: 0.95\n",
    )
    .expect("write semantically invalid alpha preset");

    let property = root.join("property.vnnlib");
    fs::write(
        &property,
        "(declare-const X_0 Real)\n\
         (declare-const X_1 Real)\n\
         (declare-const Y_0 Real)\n\
         (declare-const Y_1 Real)\n\
         (assert (>= X_0 -1.0))\n\
         (assert (<= X_0 1.0))\n\
         (assert (>= X_1 -1.0))\n\
         (assert (<= X_1 1.0))\n\
         (assert (<= Y_0 Y_1))\n",
    )
    .expect("write VNN-LIB fixture");
    (configs, property)
}

fn run_scored_case(
    root: &Path,
    configs: &Path,
    property: &Path,
    category: &str,
    case: &str,
    env_override: Option<&OsStr>,
) -> Value {
    let result = root.join(format!("{case}.result"));
    let model = workspace_root().join("tests/models/simple_mlp.onnx");
    let mut command = Command::new(env!("CARGO_BIN_EXE_ny"));
    command
        .current_dir(workspace_root())
        .args([
            "vnncomp",
            "v1",
            category,
            model.to_str().expect("model path UTF-8"),
            property.to_str().expect("property path UTF-8"),
            result.to_str().expect("result path UTF-8"),
            "5",
            "--configs-dir",
            configs.to_str().expect("configs path UTF-8"),
        ])
        // Keep the alpha precedence assertion independent of the caller's
        // alpha override and avoid retaining an external watchdog after the
        // child exits. Other registered variables remain visible on purpose:
        // the receipt must capture the real entry-time environment.
        .env_remove("NY_ALPHA_ZERO_YIELD_FRAC")
        .env("NY_VNNCOMP_EXTERNAL_WATCHDOG", "0");
    if let Some(raw) = env_override {
        command.env("NY_ALPHA_ZERO_YIELD_FRAC", raw);
    }
    let output = command.output().expect("run scored fixture");
    assert!(
        output.status.success(),
        "scored fixture failed: status={}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let sidecar = result.with_file_name(format!(
        "{}.flight.json",
        result
            .file_name()
            .expect("result filename")
            .to_string_lossy()
    ));
    serde_json::from_slice(&fs::read(sidecar).expect("flight sidecar written"))
        .expect("flight sidecar JSON")
}

fn alpha_entry(flight: &Value) -> &Value {
    assert_eq!(flight["schema_version"], 3);
    assert_eq!(flight["levers"]["status"], "resolved");
    assert_eq!(
        flight["levers"]["receipt"]["schema"],
        "ny-levers/receipt/v2"
    );
    flight["levers"]["receipt"]["levers"]
        .as_array()
        .expect("lever entries")
        .iter()
        .find(|entry| entry["name"] == "NY_ALPHA_ZERO_YIELD_FRAC")
        .expect("alpha-zero-yield declaration receipted")
}

#[ntest::timeout(60000)]
#[test]
fn scored_flight_receipts_typed_preset_and_explicit_env_kill() {
    let temp = tempfile::tempdir().expect("temporary fixture root");
    let (configs, property) = write_fixture(temp.path());

    let preset = run_scored_case(
        temp.path(),
        &configs,
        &property,
        PRESET_CATEGORY,
        "preset",
        None,
    );
    let preset_alpha = alpha_entry(&preset);
    assert_eq!(preset_alpha["value"], 0.25);
    assert_eq!(preset_alpha["source"], "config");

    let killed = run_scored_case(
        temp.path(),
        &configs,
        &property,
        PRESET_CATEGORY,
        "killed",
        Some(OsStr::new("0")),
    );
    let killed_alpha = alpha_entry(&killed);
    assert!(killed_alpha["value"].is_null());
    assert_eq!(killed_alpha["source"], "legacy_env_rejected");
    assert_eq!(killed_alpha["rejected_raw"], "0");
    assert_eq!(killed_alpha["env_utf8"], true);

    let defaults = run_scored_case(
        temp.path(),
        &configs,
        &property,
        DEFAULT_CATEGORY,
        "defaults",
        None,
    );
    let default_alpha = alpha_entry(&defaults);
    assert!(default_alpha["value"].is_null());
    assert_eq!(default_alpha["source"], "default");

    let invalid = run_scored_case(
        temp.path(),
        &configs,
        &property,
        INVALID_CATEGORY,
        "invalid",
        None,
    );
    assert_eq!(invalid["schema_version"], 3);
    assert_eq!(invalid["levers"]["status"], "invalid_config");
    assert!(invalid["levers"]["reason"]
        .as_str()
        .is_some_and(|reason| reason.contains("alpha_zero_yield_frac")));

    let missing_relational_property = temp.path().join("missing-relational.vnnlib");
    let relational = run_scored_case(
        temp.path(),
        &configs,
        &missing_relational_property,
        RELATIONAL_CATEGORY,
        "relational",
        None,
    );
    let relational_alpha = alpha_entry(&relational);
    assert!(relational_alpha["value"].is_null());
    assert_eq!(relational_alpha["source"], "default");

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;

        let non_utf8 = std::ffi::OsString::from_vec(vec![b'0', 0xff]);
        let killed = run_scored_case(
            temp.path(),
            &configs,
            &property,
            PRESET_CATEGORY,
            "killed-non-utf8",
            Some(&non_utf8),
        );
        let alpha = alpha_entry(&killed);
        assert!(alpha["value"].is_null());
        assert_eq!(alpha["source"], "legacy_env_rejected");
        assert_eq!(alpha["env_utf8"], false);
    }
}
