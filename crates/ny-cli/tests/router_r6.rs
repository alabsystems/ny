// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r6 — the O-TRIAL rules on the REAL scored entry point
//! (`ny vnncomp v1`), the `router_ostop.rs` pattern: the tracked
//! `tests/models/simple_mlp.onnx`, a temporary
//! `configs/vnncomp99/<category>.yaml`, and the `<result>.flight.json` sidecar.
//! No `benchmarks/` dependency.
//!
//! EVERY router and optimizer `NY_*` (the two modes and every rule switch,
//! R2-R6) is REMOVED from the child's environment by the declarations' own
//! names, so the typed yaml keys are the only route (yaml-route arrival with
//! zero `NY_*`).
//!
//! The property is `Y_0 <= -1e6` over a unit box: impossible for this
//! network, so the published verdict is a deterministic `unsat`, and the
//! acting optimizer must publish exactly what `off` publishes.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use ny_levers::decls::router::{
    OPTIMIZER_MODE, OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR, OPTIMIZER_RULE_IA_YIELD,
    OPTIMIZER_RULE_ROOT_ADMISSION, OPTIMIZER_RULE_TRIAL_ATTACK_HALVING,
    OPTIMIZER_RULE_TRIAL_LR_BACKOFF, OPTIMIZER_RULE_TRIAL_SPLIT_RESTART,
    OPTIMIZER_RULE_TRIAL_WIDTH, ROUTER_MODE, ROUTER_RULE_ADMISSION, ROUTER_RULE_CAP_AUTHORITY,
    ROUTER_RULE_CONTRACTS, ROUTER_RULE_DERIVED_CAPS, ROUTER_RULE_FAR_HAND_ON,
    ROUTER_RULE_GAP_CONTRACT, ROUTER_RULE_HAND_ON, ROUTER_RULE_LEASE, ROUTER_RULE_PUBLISH_GRACE,
    ROUTER_RULE_ROUTE, ROUTER_RULE_TERMINAL,
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

const OFF_YAML: &str = "router:\n  mode: \"off\"\noptimizer:\n  mode: \"off\"\n";
const ON_YAML: &str = "optimizer:\n  mode: \"on\"\n";
const ABLATED_YAML: &str = "optimizer:\n  mode: \"on\"\n  rules:\n    trial_width: false\n    \
                            trial_attack_halving: false\n";
// Review fix (router-r6): the wave tree's site is blocked by its Open ledger
// before any pin is consulted, so the pin fixture pins a SEALED lane's knob.
const PINNED_YAML: &str = "optimizer:\n  mode: \"on\"\nmargin_row:\n  alpha_lr: 0.1\n";
const SHADOW_YAML: &str = "optimizer:\n  mode: shadow\n";

const R6_RULES: [&str; 4] = [
    "rule.trial_width",
    "rule.trial_lr_backoff",
    "rule.trial_split_restart",
    "rule.trial_attack_halving",
];

const SITES: [(&str, &str); 6] = [
    ("wave-beta-bab", "margin_row.tree_wave_batch"),
    ("input_split", "bab.batch_size"),
    ("lowdim", "lowdim.batch"),
    ("margin-row.alpha", "margin_row.alpha_lr"),
    ("lowdim", "bab.branching.input_split.lowdim_split"),
    ("attack.upfront_apgd", "attack.upfront_apgd.step0"),
];

/// Every router and optimizer declaration: none may reach the child through
/// the environment.
fn router_decls() -> [&'static ny_levers::LeverDecl; 21] {
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
        &ROUTER_RULE_ROUTE,
        &ROUTER_RULE_GAP_CONTRACT,
        &OPTIMIZER_RULE_GOVERNOR,
        &OPTIMIZER_RULE_IA_YIELD,
        &OPTIMIZER_RULE_ATTACK_DEFER,
        &OPTIMIZER_RULE_ROOT_ADMISSION,
        &OPTIMIZER_RULE_TRIAL_WIDTH,
        &OPTIMIZER_RULE_TRIAL_LR_BACKOFF,
        &OPTIMIZER_RULE_TRIAL_SPLIT_RESTART,
        &OPTIMIZER_RULE_TRIAL_ATTACK_HALVING,
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

    fn line(&self, prefix: &str) -> String {
        self.stderr()
            .lines()
            .find(|line| line.starts_with(prefix))
            .map(str::to_owned)
            .unwrap_or_else(|| panic!("no `{prefix}` line:\n{}", self.stderr()))
    }

    fn act_line(&self, rule: &str) -> String {
        self.line(&format!("[optimizer] act rule={rule} "))
    }

    fn site_line(&self, cap: &str, knob: &str) -> String {
        self.line(&format!("[router] trial-site site={cap} knob={knob} "))
    }
}

fn run(fixture: &Fixture, category: &str, case: &str, budget: &str) -> Run {
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
            budget,
            "--configs-dir",
            fixture.configs.to_str().expect("configs path UTF-8"),
        ])
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

fn assert_no_router_env(run: &Run) {
    let ambient = run.flight["ambient_env"].as_object().expect("ambient env");
    for decl in router_decls() {
        assert!(
            !ambient.contains_key(decl.name),
            "{} reached the child environment",
            decl.name
        );
    }
}

/// `optimizer.mode: on` arrives by the yaml route with zero `NY_*`; the
/// session ACTS through all four O-TRIAL rules at every site, arms the trial
/// authority, closes it after publication, records the pure commitments for
/// replay, and publishes exactly what `off` publishes.
#[ntest::timeout(240000)]
#[test]
fn router_r6_trials_arrive_by_yaml_route_and_arm_every_site() {
    let fixture = Fixture::new();
    fixture.preset("router_r6_off_2026", OFF_YAML);
    fixture.preset("router_r6_on_2026", ON_YAML);
    let off = run(&fixture, "router_r6_off_2026", "r6_off", "60");
    let on = run(&fixture, "router_r6_on_2026", "r6_on", "60");
    for rule in R6_RULES {
        let line = on.act_line(rule);
        assert!(line.contains(" acts=1 "), "{rule} must act: {line}");
    }
    for (cap, knob) in SITES {
        let line = on.site_line(cap, knob);
        // Review fix (router-r6): the wave tree is `Open` in the registry
        // (S7+S8 empty until M6): its site is never trialled.
        let expected = if cap == "wave-beta-bab" {
            " armed=0 blocked_by=ledger:wave-beta-bab"
        } else {
            " armed=1 blocked_by=-"
        };
        assert!(line.ends_with(expected), "{line}");
    }
    let stderr = on.stderr();
    assert!(
        stderr.contains("[router] trial authority armed sha="),
        "{stderr}"
    );
    assert!(stderr.contains("[router] trial done armed=1 "), "{stderr}");
    assert_no_router_env(&on);
    let mode = lever_entry(&on.flight, OPTIMIZER_MODE.name);
    assert_eq!(mode["value"], "on");
    assert_eq!(
        mode["source"], "config",
        "the mode arrives from the typed preset"
    );
    let record = &on.flight["router"];
    assert_eq!(record["optimizer_effective"], "acting");
    assert_eq!(record["trial_commitments"]["acting"], true);
    assert_eq!(record["trial"]["armed"], true);
    assert!(record["trial"]["snapshot"]["events"].is_array());
    assert!(
        record["plan"]["emitted"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "R6 still emits no action-space value"
    );
    assert!(off.result.starts_with("unsat"), "{:?}", off.result);
    assert_eq!(
        off.result, on.result,
        "the acting optimizer changed the verdict"
    );
    assert!(
        !off.stderr().contains("[router] trial"),
        "optimizer off printed a trial line:\n{}",
        off.stderr()
    );
    // Replay re-derives the plan AND the O-TRIAL commitments bit for bit.
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
        "replay of an O-TRIAL record failed: {}\n{}",
        String::from_utf8_lossy(&replay.stdout),
        String::from_utf8_lossy(&replay.stderr)
    );
}

/// Each O-TRIAL rule is its own capability: `optimizer.rules.trial_<rule>:
/// false` ablates exactly that rule (zero `NY_*`), which says why.
#[ntest::timeout(240000)]
#[test]
fn router_r6_trial_rule_keys_ablate_their_rule_by_yaml_route() {
    let fixture = Fixture::new();
    fixture.preset("router_r6_ablated_2026", ABLATED_YAML);
    let ablated = run(&fixture, "router_r6_ablated_2026", "r6_ablated", "60");
    for rule in ["rule.trial_width", "rule.trial_attack_halving"] {
        let line = ablated.act_line(rule);
        assert!(line.contains(" acts=0 blocked_by=key "), "{rule}: {line}");
    }
    for rule in ["rule.trial_lr_backoff", "rule.trial_split_restart"] {
        assert!(ablated.act_line(rule).contains(" acts=1 "), "{rule}");
    }
    for decl in [
        &OPTIMIZER_RULE_TRIAL_WIDTH,
        &OPTIMIZER_RULE_TRIAL_ATTACK_HALVING,
    ] {
        let entry = lever_entry(&ablated.flight, decl.name);
        assert_eq!(entry["value"], false, "{}", decl.name);
        assert_eq!(entry["source"], "config", "{}", decl.name);
    }
    let kept = lever_entry(&ablated.flight, OPTIMIZER_RULE_TRIAL_LR_BACKOFF.name);
    assert_eq!(kept["value"], true);
    assert_no_router_env(&ablated);
    assert!(ablated.result.starts_with("unsat"), "{:?}", ablated.result);
}

/// A knob the row's file pins is FROZEN: its site is not armed
/// (`blocked_by=pin:<path>`), every other site still is.
#[ntest::timeout(240000)]
#[test]
fn router_r6_a_pinned_knob_freezes_its_trial_site() {
    let fixture = Fixture::new();
    fixture.preset("router_r6_pinned_2026", PINNED_YAML);
    let pinned = run(&fixture, "router_r6_pinned_2026", "r6_pinned", "60");
    let line = pinned.site_line("margin-row.alpha", "margin_row.alpha_lr");
    assert!(
        line.ends_with(" armed=0 blocked_by=pin:margin_row.alpha_lr"),
        "{line}"
    );
    let lowdim = pinned.site_line("lowdim", "lowdim.batch");
    assert!(lowdim.ends_with(" armed=1 blocked_by=-"), "{lowdim}");
    assert!(pinned.act_line("rule.trial_width").contains(" acts=1 "));
    let lr = pinned.act_line("rule.trial_lr_backoff");
    assert!(
        lr.contains(" acts=0 blocked_by=pin:margin_row.alpha_lr "),
        "its only site is pinned: {lr}"
    );
    assert_eq!(
        pinned.flight["router"]["r6_input"]["pinned_keys"],
        serde_json::json!(["margin_row.alpha_lr"])
    );
}

/// CR-S: a row of 30 s or less takes no trial; `shadow` never arms.
#[ntest::timeout(240000)]
#[test]
fn router_r6_short_rows_and_shadow_take_no_trial() {
    let fixture = Fixture::new();
    fixture.preset("router_r6_short_2026", ON_YAML);
    fixture.preset("router_r6_shadow_2026", SHADOW_YAML);
    let short = run(&fixture, "router_r6_short_2026", "r6_short", "5");
    for rule in R6_RULES {
        let line = short.act_line(rule);
        assert!(
            line.contains(" acts=0 blocked_by=short_row "),
            "{rule}: {line}"
        );
    }
    assert!(
        short
            .stderr()
            .contains("[router] trial authority idle reason=no_site_armed"),
        "{}",
        short.stderr()
    );
    let shadow = run(&fixture, "router_r6_shadow_2026", "r6_shadow", "5");
    assert!(
        !shadow.stderr().contains("[router] trial"),
        "shadow printed a trial line:\n{}",
        shadow.stderr()
    );
    let record = &shadow.flight["router"];
    assert!(record["trial_commitments"].is_null(), "{record}");
    assert!(record["trial"].is_null(), "{record}");
    assert_eq!(short.result, shadow.result);
}
