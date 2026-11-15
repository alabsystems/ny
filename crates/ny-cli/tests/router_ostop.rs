// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r3 — acting set 2 (the O-STOP rules) on the REAL scored entry
//! point (`ny vnncomp v1`), the `router_acting.rs` pattern: the tracked
//! `tests/models/simple_mlp.onnx`, a temporary
//! `configs/vnncomp99/<category>.yaml`, and the `<result>.flight.json` sidecar.
//! No `benchmarks/` dependency.
//!
//! EVERY router and optimizer `NY_*` (the two modes, the seven router rule
//! switches and the four optimizer rule switches) is REMOVED from the child's
//! environment by the declarations' own names, so the typed yaml keys are the
//! only route (yaml-route arrival with zero `NY_*`).
//!
//! The property is `Y_0 <= -1e6` over a unit box: impossible for this
//! network, so the published verdict is a deterministic `unsat`, and the
//! acting optimizer must publish exactly what `off` publishes.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use ny_levers::decls::router::{
    OPTIMIZER_MODE, OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR, OPTIMIZER_RULE_IA_YIELD,
    OPTIMIZER_RULE_ROOT_ADMISSION, ROUTER_MODE, ROUTER_RULE_ADMISSION, ROUTER_RULE_CAP_AUTHORITY,
    ROUTER_RULE_CONTRACTS, ROUTER_RULE_DERIVED_CAPS, ROUTER_RULE_HAND_ON,
    ROUTER_RULE_PUBLISH_GRACE, ROUTER_RULE_TERMINAL,
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
const SHADOW_YAML: &str = "optimizer:\n  mode: shadow\n";
const ON_YAML: &str = "router:\n  mode: \"on\"\noptimizer:\n  mode: \"on\"\n";
const OPTIMIZER_ONLY_YAML: &str = "optimizer:\n  mode: \"on\"\n";
const ABLATED_YAML: &str = "router:\n  mode: \"on\"\noptimizer:\n  mode: \"on\"\n  rules:\n    \
                            governor: false\n    attack_defer: false\n";

const R3_RULES: [&str; 4] = [
    "rule.governor",
    "rule.ia_yield",
    "rule.attack_defer",
    "rule.root_admission",
];

/// Every router and optimizer declaration: none may reach the child through
/// the environment.
fn router_decls() -> [&'static ny_levers::LeverDecl; 13] {
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

    fn optimizer_lines(&self) -> Vec<String> {
        self.stderr()
            .lines()
            .filter(|line| line.starts_with("[optimizer]"))
            .map(str::to_owned)
            .collect()
    }

    fn act_line(&self, rule: &str) -> String {
        let prefix = format!("[optimizer] act rule={rule} ");
        self.optimizer_lines()
            .into_iter()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("no `{prefix}` line:\n{}", self.stderr()))
    }
}

/// `budget` is the scored budget in seconds: 60 keeps the row out of CR-S's
/// short-row regime (B <= 30 s takes no defer); the row is decided by the
/// fast path either way, so the wall does not grow.
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

/// `optimizer.mode: on` (with `router.mode: on`) arrives by the yaml route
/// with zero `NY_*`, the session ACTS through all four O-STOP rules, arms the
/// stop authority, closes it after publication, records the pure commitments
/// for replay, and publishes exactly what `off` publishes.
#[ntest::timeout(240000)]
#[test]
fn router_optimizer_on_arrives_by_yaml_route_and_acts_through_the_four_stop_rules() {
    let fixture = Fixture::new();
    fixture.preset("router_ostop_off_2026", OFF_YAML);
    fixture.preset("router_ostop_on_2026", ON_YAML);
    let off = run(&fixture, "router_ostop_off_2026", "ostop_off", "60");
    let on = run(&fixture, "router_ostop_on_2026", "ostop_on", "60");
    let stderr = on.stderr();
    assert!(
        stderr.contains("[router] v1 mode=on effective=acting"),
        "the router acts too (CR-1 is what ia_yield / attack_defer ride on):\n{stderr}"
    );
    for rule in R3_RULES {
        let line = on.act_line(rule);
        assert!(line.contains(" acts=1 "), "{rule} must act: {line}");
    }
    assert!(
        stderr.contains("[optimizer] authority armed sha="),
        "the stop authority arms:\n{stderr}"
    );
    assert!(
        stderr.contains("governor=0.15/5 ia_yield=0.15/2 attack_defer=0.5/5000ms root_admission=1"),
        "the measured constants ship:\n{stderr}"
    );
    assert!(stderr.contains("[optimizer] done armed=1 "), "{stderr}");
    assert!(stderr.contains("[router] done faults=0"), "{stderr}");
    assert_no_router_env(&on);
    let mode = lever_entry(&on.flight, OPTIMIZER_MODE.name);
    assert_eq!(mode["value"], "on");
    assert_eq!(
        mode["source"], "config",
        "the mode arrives from the typed preset"
    );
    let record = &on.flight["router"];
    assert_eq!(record["optimizer_effective"], "acting");
    assert_eq!(record["stop_commitments"]["acting"], true);
    assert_eq!(record["stop"]["armed"], true);
    assert!(record["stop"]["snapshot"]["yields"].is_array());
    assert!(
        record["plan"]["emitted"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "R3 still emits no action-space value"
    );
    // The acting optimizer publishes what `off` publishes on this row.
    assert!(off.result.starts_with("unsat"), "{:?}", off.result);
    assert_eq!(
        off.result, on.result,
        "the acting optimizer changed the verdict"
    );
    assert!(
        off.optimizer_lines().is_empty(),
        "optimizer off printed {:?}",
        off.optimizer_lines()
    );
    // Replay re-derives the plan, R2's commitments AND the O-STOP commitments.
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
        "replay of an O-STOP record failed: {}\n{}",
        String::from_utf8_lossy(&replay.stdout),
        String::from_utf8_lossy(&replay.stderr)
    );
}

/// Each O-STOP rule is its own capability: a yaml `optimizer.rules.<rule>:
/// false` ablates exactly that rule (zero `NY_*`), which says why.
#[ntest::timeout(240000)]
#[test]
fn router_optimizer_rule_keys_ablate_their_rule_by_yaml_route() {
    let fixture = Fixture::new();
    fixture.preset("router_ostop_ablated_2026", ABLATED_YAML);
    let ablated = run(&fixture, "router_ostop_ablated_2026", "ostop_ablated", "60");
    for rule in ["rule.governor", "rule.attack_defer"] {
        let line = ablated.act_line(rule);
        assert!(
            line.contains(" acts=0 blocked_by=key "),
            "{rule} must be ablated by its key: {line}"
        );
    }
    for rule in ["rule.ia_yield", "rule.root_admission"] {
        assert!(ablated.act_line(rule).contains(" acts=1 "), "{rule}");
    }
    let stderr = ablated.stderr();
    assert!(
        stderr.contains("governor=- ia_yield=0.15/2 attack_defer=- root_admission=1"),
        "{stderr}"
    );
    for decl in [&OPTIMIZER_RULE_GOVERNOR, &OPTIMIZER_RULE_ATTACK_DEFER] {
        let entry = lever_entry(&ablated.flight, decl.name);
        assert_eq!(entry["value"], false, "{}", decl.name);
        assert_eq!(entry["source"], "config", "{}", decl.name);
    }
    let kept = lever_entry(&ablated.flight, OPTIMIZER_RULE_IA_YIELD.name);
    assert_eq!(kept["value"], true);
    assert_no_router_env(&ablated);
    assert!(ablated.result.starts_with("unsat"), "{:?}", ablated.result);
}

/// The optimizer without the router: `rule.ia_yield` and `rule.attack_defer`
/// ride on CR-1 and are blocked; the governor and root admission still act.
#[ntest::timeout(240000)]
#[test]
fn router_optimizer_without_the_router_blocks_the_cr1_riders() {
    let fixture = Fixture::new();
    fixture.preset("router_ostop_alone_2026", OPTIMIZER_ONLY_YAML);
    let alone = run(&fixture, "router_ostop_alone_2026", "ostop_alone", "60");
    for rule in ["rule.ia_yield", "rule.attack_defer"] {
        let line = alone.act_line(rule);
        assert!(line.contains(" acts=0 blocked_by=CR-1 "), "{rule}: {line}");
    }
    for rule in ["rule.governor", "rule.root_admission"] {
        assert!(alone.act_line(rule).contains(" acts=1 "), "{rule}");
    }
    let stderr = alone.stderr();
    assert!(
        !stderr.contains("[router] authority armed"),
        "the router itself stays off: {stderr}"
    );
    assert!(stderr.contains("[optimizer] done armed=1 "), "{stderr}");
    assert!(alone.result.starts_with("unsat"), "{:?}", alone.result);
}

/// `shadow` never acts: no act / authority / yield / defer line, no O-STOP
/// commitments in the record, and the published result equals `off`'s.
#[ntest::timeout(240000)]
#[test]
fn router_optimizer_shadow_never_arms_the_stop_authority() {
    let fixture = Fixture::new();
    fixture.preset("router_ostop_shadow_2026", SHADOW_YAML);
    fixture.preset("router_ostop_shadow_off_2026", OFF_YAML);
    let shadow = run(&fixture, "router_ostop_shadow_2026", "ostop_shadow", "5");
    let off = run(
        &fixture,
        "router_ostop_shadow_off_2026",
        "ostop_shadow_off",
        "5",
    );
    for line in shadow.optimizer_lines() {
        for forbidden in [
            "[optimizer] act ",
            "[optimizer] authority",
            "[optimizer] yield",
            "[optimizer] hand-on",
            "[optimizer] defer",
            "[optimizer] root-admission",
            "[optimizer] contract",
            "[optimizer] done",
        ] {
            assert!(
                !line.starts_with(forbidden),
                "shadow printed an acting line: {line}"
            );
        }
    }
    let record = &shadow.flight["router"];
    assert_eq!(record["optimizer_effective"], "shadow");
    assert!(record["stop_commitments"].is_null(), "{record}");
    assert!(record["stop"].is_null(), "{record}");
    assert_eq!(shadow.result, off.result);
}

/// CR-S: on a short row (B <= 30 s) CR-ATK takes no defer — the attack keeps
/// its whole slice — while the other three rules still act.
#[ntest::timeout(240000)]
#[test]
fn router_optimizer_short_row_takes_no_attack_defer() {
    let fixture = Fixture::new();
    fixture.preset("router_ostop_short_2026", ON_YAML);
    let short = run(&fixture, "router_ostop_short_2026", "ostop_short", "5");
    let line = short.act_line("rule.attack_defer");
    assert!(line.contains(" acts=0 blocked_by=short_row "), "{line}");
    for rule in ["rule.governor", "rule.ia_yield", "rule.root_admission"] {
        assert!(short.act_line(rule).contains(" acts=1 "), "{rule}");
    }
    assert!(short.result.starts_with("unsat"), "{:?}", short.result);
}
