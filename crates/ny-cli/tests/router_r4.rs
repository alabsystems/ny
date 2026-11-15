// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 — acting set 3 (CR-L memory lease, CR-F FAR hand-on) on the
//! REAL scored entry point (`ny vnncomp v1`), the `router_acting.rs` pattern:
//! the tracked `tests/models/simple_mlp.onnx`, a temporary
//! `configs/vnncomp99/<category>.yaml`, and the `<result>.flight.json`
//! sidecar. No `benchmarks/` dependency.
//!
//! EVERY router `NY_*` (the two modes and every rule switch, R4's two
//! included) is REMOVED from the child's environment by the declarations' own
//! names, so the typed yaml keys are the only route (yaml-route arrival with
//! zero `NY_*`).
//!
//! The property is `Y_0 <= -1e6` over a unit box: impossible for this network,
//! so the published verdict is a deterministic `unsat`, and the acting router
//! (lease armed, FAR authority armed) must publish exactly what `off`
//! publishes. The 60 s budget keeps the row out of CR-S (short rows take no
//! checkpoint action), so CR-F is eligible and arms in OBSERVE mode: no
//! registry entry is strictly sealed.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use ny_levers::decls::router::{
    OPTIMIZER_MODE, OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR, OPTIMIZER_RULE_IA_YIELD,
    OPTIMIZER_RULE_ROOT_ADMISSION, ROUTER_MODE, ROUTER_RULE_ADMISSION, ROUTER_RULE_CAP_AUTHORITY,
    ROUTER_RULE_CONTRACTS, ROUTER_RULE_DERIVED_CAPS, ROUTER_RULE_FAR_HAND_ON, ROUTER_RULE_HAND_ON,
    ROUTER_RULE_LEASE, ROUTER_RULE_PUBLISH_GRACE, ROUTER_RULE_TERMINAL,
};
use ny_test_utils::workspace_root;
use serde_json::Value;

const PROPERTY: &str = "(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 -1000000.0))
";

const OFF_YAML: &str = "router:\n  mode: \"off\"\n";
const SHADOW_YAML: &str = "router:\n  mode: shadow\n";
const ON_YAML: &str = "router:\n  mode: \"on\"\n";
const ABLATED_YAML: &str = "router:\n  mode: \"on\"\n  rules:\n    lease: false\n    \
                            far_hand_on: false\n";

/// Every router declaration: none may reach the child through the environment.
fn router_decls() -> [&'static ny_levers::LeverDecl; 15] {
    [
        &ROUTER_MODE,
        &OPTIMIZER_MODE,
        &ROUTER_RULE_ADMISSION,
        &ROUTER_RULE_CONTRACTS,
        &ROUTER_RULE_CAP_AUTHORITY,
        &ROUTER_RULE_TERMINAL,
        &ROUTER_RULE_HAND_ON,
        &ROUTER_RULE_PUBLISH_GRACE,
        &ROUTER_RULE_DERIVED_CAPS,
        &ROUTER_RULE_LEASE,
        &ROUTER_RULE_FAR_HAND_ON,
        &OPTIMIZER_RULE_GOVERNOR,
        &OPTIMIZER_RULE_IA_YIELD,
        &OPTIMIZER_RULE_ATTACK_DEFER,
        &OPTIMIZER_RULE_ROOT_ADMISSION,
    ]
}

struct Fixture {
    root: tempfile::TempDir,
    configs: PathBuf,
    property: PathBuf,
    model: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("temporary fixture root");
        let configs = root.path().join("configs");
        fs::create_dir_all(configs.join("vnncomp99")).expect("config tree");
        let property = root.path().join("property.vnnlib");
        fs::write(&property, PROPERTY).expect("write the property");
        Self {
            configs,
            property,
            model: workspace_root().join("tests/models/simple_mlp.onnx"),
            root,
        }
    }

    fn preset(&self, category: &str, yaml: &str) {
        fs::write(
            self.configs
                .join("vnncomp99")
                .join(format!("{category}.yaml")),
            yaml,
        )
        .expect("write the category preset");
    }
}

struct Run {
    output: Output,
    result: String,
    flight: Value,
    sidecar: PathBuf,
}

impl Run {
    fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.output.stderr).into_owned()
    }

    fn router_lines(&self) -> Vec<String> {
        self.stderr()
            .lines()
            .filter(|line| line.starts_with("[router]"))
            .map(str::to_owned)
            .collect()
    }

    fn line(&self, prefix: &str) -> String {
        self.router_lines()
            .into_iter()
            .find(|line| line.starts_with(prefix))
            .unwrap_or_else(|| panic!("no `{prefix}` line:\n{}", self.stderr()))
    }
}

fn run(fixture: &Fixture, category: &str, case: &str) -> Run {
    let result = fixture.root.path().join(format!("{case}.result"));
    let mut command = Command::new(env!("CARGO_BIN_EXE_ny"));
    command
        .current_dir(workspace_root())
        .args([
            "vnncomp",
            "v1",
            category,
            fixture.model.to_str().expect("model path UTF-8"),
            fixture.property.to_str().expect("property path UTF-8"),
            result.to_str().expect("result path UTF-8"),
            "60",
            "--configs-dir",
            fixture.configs.to_str().expect("configs path UTF-8"),
        ])
        // As in `router_shadow.rs`: no lingering out-of-process watchdog.
        .env("NY_VNNCOMP_EXTERNAL_WATCHDOG", "0");
    for decl in router_decls() {
        command.env_remove(decl.name);
    }
    let output = command.output().expect("run the scored fixture");
    assert!(
        output.status.success(),
        "scored fixture {case} failed: status={}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let result_text = fs::read_to_string(&result).expect("the result file is written");
    let sidecar = result.with_file_name(format!(
        "{}.flight.json",
        result.file_name().expect("result name").to_string_lossy()
    ));
    let flight: Value =
        serde_json::from_slice(&fs::read(&sidecar).expect("flight sidecar written"))
            .expect("flight sidecar JSON");
    Run {
        output,
        result: result_text,
        flight,
        sidecar,
    }
}

fn lever_entry<'a>(flight: &'a Value, name: &str) -> &'a Value {
    flight["levers"]["receipt"]["levers"]
        .as_array()
        .expect("lever entries")
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("{name} is not receipted"))
}

/// `router.mode: on` arrives by the yaml route with zero router `NY_*`; acting
/// set 3 arms the memory lease against the declared host's envelope and the
/// FAR authority in observe mode (no strictly sealed recipient), both close
/// after publication, the record replays, and the published verdict is
/// exactly `off`'s: the lease is admission control, never a verdict input.
#[ntest::timeout(240000)]
#[test]
fn router_r4_on_arrives_by_yaml_route_and_arms_the_lease_and_far() {
    let fixture = Fixture::new();
    fixture.preset("router_r4_off_2026", OFF_YAML);
    fixture.preset("router_r4_on_2026", ON_YAML);
    let off = run(&fixture, "router_r4_off_2026", "r4_off");
    let on = run(&fixture, "router_r4_on_2026", "r4_on");
    let stderr = on.stderr();
    assert!(
        stderr.contains("[router] v1 mode=on effective=acting"),
        "the yaml key must arm the acting router:\n{stderr}"
    );
    let cr_l = on.line("[router] act rule=CR-L ");
    assert!(cr_l.contains(" acts=1 "), "CR-L acts: {cr_l}");
    let cr_f = on.line("[router] act rule=CR-F ");
    assert!(
        cr_f.contains(" acts=0 blocked_by=no_sealed_recipient "),
        "CR-F has no strictly sealed recipient in R4: {cr_f}"
    );
    let armed = on.line("[router] lease armed rule=CR-L ");
    assert!(
        armed.contains(" floor=live_envelope") && armed.contains(" terminal=granted:"),
        "{armed}"
    );
    assert_eq!(
        on.line("[router] far armed rule=CR-F "),
        "[router] far armed rule=CR-F mode=observe routes=[-] ckpt=c2,c3 window=work_units"
    );
    // Both authorities close AFTER publication with their summaries.
    let lease_done = on.line("[router] lease done ");
    assert!(
        lease_done.starts_with("[router] lease done armed=1 "),
        "{lease_done}"
    );
    let far_done = on.line("[router] far done ");
    assert!(
        far_done.starts_with("[router] far done armed=1 observe_only=1 "),
        "{far_done}"
    );
    assert!(
        far_done.contains(" handed=0 "),
        "observe mode hands nothing on: {far_done}"
    );
    assert!(stderr.contains("[router] done faults=0"), "{stderr}");
    // Zero router NY_*: the mode came from the preset, the R4 switches from
    // their declaration defaults.
    let ambient = on.flight["ambient_env"].as_object().expect("ambient env");
    for decl in router_decls() {
        assert!(
            !ambient.contains_key(decl.name),
            "{} reached the child environment",
            decl.name
        );
    }
    assert_eq!(
        lever_entry(&on.flight, ROUTER_MODE.name)["source"],
        "config"
    );
    for decl in [&ROUTER_RULE_LEASE, &ROUTER_RULE_FAR_HAND_ON] {
        let entry = lever_entry(&on.flight, decl.name);
        assert_eq!(entry["value"], true, "{}", decl.name);
        assert_ne!(entry["source"], "legacy_env", "{}", decl.name);
    }
    // The flight record carries the pure R4 commitments and both snapshots.
    let record = &on.flight["router"];
    assert_eq!(record["r4_commitments"]["acting"], true);
    assert!(record["r4_commitments"]["lease"]["envelope_bytes"].as_u64() > Some(0));
    assert_eq!(record["r4_commitments"]["far"]["observe_only"], true);
    assert_eq!(record["lease"]["armed"], true);
    let snapshot = &record["lease"]["snapshot"];
    assert!(
        snapshot["outstanding_bytes"].as_u64()
            <= record["r4_commitments"]["lease"]["envelope_bytes"].as_u64(),
        "the aggregate never passes the envelope: {snapshot}"
    );
    assert_eq!(record["far"]["armed"], true);
    // The acting router publishes what `off` publishes on this row.
    assert!(off.result.starts_with("unsat"), "{:?}", off.result);
    assert_eq!(off.result, on.result, "acting set 3 changed the verdict");
    assert!(
        off.router_lines().is_empty(),
        "router off printed {:?}",
        off.router_lines()
    );
    // Replay re-derives the plan, R2's and R4's commitments, and every live
    // CR-F decision, bit for bit.
    let replay = Command::new(env!("CARGO_BIN_EXE_ny"))
        .current_dir(workspace_root())
        .args([
            "vnncomp",
            "plan",
            "--replay",
            on.sidecar.to_str().expect("sidecar UTF-8"),
        ])
        .output()
        .expect("run the replay");
    assert!(
        replay.status.success(),
        "replay of an R4 record failed: {}\n{}",
        String::from_utf8_lossy(&replay.stdout),
        String::from_utf8_lossy(&replay.stderr)
    );
}

/// Each R4 rule is its own capability: a yaml `router.rules.<rule>: false`
/// ablates exactly that rule (zero `NY_*`), which says why, and arms nothing.
#[ntest::timeout(240000)]
#[test]
fn router_r4_rule_keys_ablate_their_rule_by_yaml_route() {
    let fixture = Fixture::new();
    fixture.preset("router_r4_ablated_2026", ABLATED_YAML);
    let ablated = run(&fixture, "router_r4_ablated_2026", "r4_ablated");
    for rule in ["CR-L", "CR-F"] {
        let line = ablated.line(&format!("[router] act rule={rule} "));
        assert!(
            line.contains(" acts=0 blocked_by=key "),
            "{rule} must be ablated by its key: {line}"
        );
    }
    let stderr = ablated.stderr();
    // Positive control: the R2 rules still act on this row.
    assert!(
        ablated.line("[router] act rule=CR-1 ").contains(" acts=1 "),
        "{stderr}"
    );
    assert!(
        stderr.contains("[router] lease idle rule=CR-L reason=not_acting"),
        "{stderr}"
    );
    assert!(
        stderr.contains("[router] far idle rule=CR-F reason=not_eligible"),
        "{stderr}"
    );
    assert!(!stderr.contains("[router] lease armed"), "{stderr}");
    for decl in [&ROUTER_RULE_LEASE, &ROUTER_RULE_FAR_HAND_ON] {
        let entry = lever_entry(&ablated.flight, decl.name);
        assert_eq!(entry["value"], false, "{}", decl.name);
        assert_eq!(entry["source"], "config", "{}", decl.name);
    }
    assert!(ablated.result.starts_with("unsat"), "{:?}", ablated.result);
}

/// `shadow` never arms acting set 3: no lease or FAR line, nothing in the
/// record, and the published result equals `off`'s.
#[ntest::timeout(240000)]
#[test]
fn router_r4_shadow_never_arms_the_lease_or_far() {
    let fixture = Fixture::new();
    fixture.preset("router_r4_shadow_2026", SHADOW_YAML);
    fixture.preset("router_r4_shadow_off_2026", OFF_YAML);
    let shadow = run(&fixture, "router_r4_shadow_2026", "r4_shadow");
    let off = run(&fixture, "router_r4_shadow_off_2026", "r4_shadow_off");
    for line in shadow.router_lines() {
        for forbidden in [
            "[router] act ",
            "[router] lease ",
            "[router] far ",
            "[router] safepoint ",
        ] {
            assert!(
                !line.starts_with(forbidden),
                "shadow printed an acting line: {line}"
            );
        }
    }
    let record = &shadow.flight["router"];
    assert_eq!(record["effective"], "shadow");
    assert!(record["r4_commitments"].is_null(), "{record}");
    assert!(record["lease"].is_null(), "{record}");
    assert!(record["far"].is_null(), "{record}");
    assert_eq!(shadow.result, off.result);
}
