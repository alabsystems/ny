// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r2 — acting set 1 on the REAL scored entry point (`ny vnncomp v1`),
//! the `router_shadow.rs` pattern: the tracked `tests/models/simple_mlp.onnx`,
//! a temporary `configs/vnncomp99/<category>.yaml`, and the
//! `<result>.flight.json` sidecar. No `benchmarks/` dependency.
//!
//! EVERY router `NY_*` (the two modes and the seven rule switches) is REMOVED
//! from the child's environment by the declarations' own names, so the typed
//! yaml keys are the only route (yaml-route arrival with zero `NY_*`).
//!
//! The property is `Y_0 <= -1e6` over a unit box: impossible for this network,
//! so the published verdict is a deterministic `unsat`, and the acting router
//! must publish exactly what `off` publishes.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use ny_levers::decls::router::{
    OPTIMIZER_MODE, ROUTER_MODE, ROUTER_RULE_ADMISSION, ROUTER_RULE_CAP_AUTHORITY,
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

const OFF_YAML: &str = "router:\n  mode: \"off\"\n";
const SHADOW_YAML: &str = "router:\n  mode: shadow\n";
const ON_YAML: &str = "router:\n  mode: \"on\"\n";
const ABLATED_YAML: &str = "router:\n  mode: \"on\"\n  rules:\n    publish_grace: false\n    \
                            derived_caps: false\n";

/// Every router declaration: none may reach the child through the environment.
fn router_decls() -> [&'static ny_levers::LeverDecl; 9] {
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
    ]
}

const R2_RULES: [&str; 7] = ["CR-1", "CR-A", "CR-C", "CR-T", "CR-H", "CR-R", "CR-K"];

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

    fn act_line(&self, rule: &str) -> String {
        let prefix = format!("[router] act rule={rule} ");
        self.router_lines()
            .into_iter()
            .find(|line| line.starts_with(&prefix))
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
            "5",
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

/// `router.mode: on` arrives by the yaml route with zero router `NY_*`, the
/// session ACTS through all seven rules, arms the one deadline authority,
/// closes it after publication, and publishes exactly what `off` publishes.
#[ntest::timeout(240000)]
#[test]
fn router_on_arrives_by_yaml_route_and_acts_through_the_seven_rules() {
    let fixture = Fixture::new();
    fixture.preset("router_acting_off_2026", OFF_YAML);
    fixture.preset("router_acting_on_2026", ON_YAML);
    let off = run(&fixture, "router_acting_off_2026", "acting_off");
    let on = run(&fixture, "router_acting_on_2026", "acting_on");
    let stderr = on.stderr();
    assert!(
        stderr.contains("[router] v1 mode=on effective=acting"),
        "the yaml key must arm the acting router:\n{stderr}"
    );
    for rule in R2_RULES {
        let line = on.act_line(rule);
        assert!(line.contains(" acts=1 "), "{rule} must act: {line}");
    }
    assert!(
        stderr.contains("[router] authority armed sha="),
        "CR-1 arms the deadline authority:\n{stderr}"
    );
    assert!(stderr.contains("terminal=verifier"), "{stderr}");
    assert!(
        stderr.contains("grace_ms=1500"),
        "CR-R at the 2 s join: {stderr}"
    );
    assert!(stderr.contains("[router] acting done armed=1"), "{stderr}");
    assert!(stderr.contains("[router] done faults=0"), "{stderr}");
    // review(router-r2): the authority is CONSULTED on the scored path, not
    // only armed. The terminal claims its cell at the verifier's start (CR-T),
    // and the verifier's configuration boundary applies the rule-owned values
    // (CR-H phase hand-on, CR-K clause passes) with the router as the source.
    // Without these the `on` run above could arm an authority no lane ever
    // read and still pass.
    for wired in [
        "[router] lane slot=verifier event=claim rule=CR-T terminal=1",
        "[router] derived rule=CR-H key=bab.initial_bounds_warmup_fallback from=false to=true \
         source=router",
        "[router] derived rule=CR-K key=bab.disjunctive_clause_retry_passes from=0 to=",
    ] {
        assert!(
            stderr.contains(wired),
            "the scored path never consulted the router: no `{wired}`:\n{stderr}"
        );
    }
    // Zero router NY_*: every rule switch and the mode came from the preset
    // or the declaration default.
    let ambient = on.flight["ambient_env"].as_object().expect("ambient env");
    for decl in router_decls() {
        assert!(
            !ambient.contains_key(decl.name),
            "{} reached the child environment",
            decl.name
        );
    }
    let mode = lever_entry(&on.flight, ROUTER_MODE.name);
    assert_eq!(mode["value"], "on");
    assert_eq!(
        mode["source"], "config",
        "the mode arrives from the typed preset"
    );
    // The flight record carries the pure commitments and the authority.
    let record = &on.flight["router"];
    assert_eq!(record["effective"], "acting");
    assert_eq!(record["commitments"]["acting"], true);
    assert_eq!(record["commitments"]["arm_authority"], true);
    assert_eq!(record["acting"]["armed"], true);
    assert!(record["acting"]["authority"]["slots"].is_array());
    assert!(
        record["plan"]["emitted"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "R2 still emits no action-space value"
    );
    // The acting router publishes what `off` publishes on this row.
    assert!(off.result.starts_with("unsat"), "{:?}", off.result);
    assert_eq!(
        off.result, on.result,
        "the acting router changed the verdict"
    );
    // Off stays dark: no router line of any kind.
    assert!(
        off.router_lines().is_empty(),
        "router off printed {:?}",
        off.router_lines()
    );
    // Replay re-derives the plan AND the acting commitments bit for bit.
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
        "replay of an acting record failed: {}\n{}",
        String::from_utf8_lossy(&replay.stdout),
        String::from_utf8_lossy(&replay.stderr)
    );
}

/// Each rule is its own capability: a yaml `router.rules.<rule>: false`
/// ablates exactly that rule (zero `NY_*`), which says why.
#[ntest::timeout(240000)]
#[test]
fn router_rule_keys_ablate_their_rule_by_yaml_route() {
    let fixture = Fixture::new();
    fixture.preset("router_acting_ablated_2026", ABLATED_YAML);
    let ablated = run(&fixture, "router_acting_ablated_2026", "acting_ablated");
    for rule in ["CR-R", "CR-K"] {
        let line = ablated.act_line(rule);
        assert!(
            line.contains(" acts=0 blocked_by=key "),
            "{rule} must be ablated by its key: {line}"
        );
    }
    for rule in ["CR-1", "CR-A", "CR-C", "CR-T", "CR-H"] {
        assert!(ablated.act_line(rule).contains(" acts=1 "), "{rule}");
    }
    let stderr = ablated.stderr();
    assert!(stderr.contains("grace_ms=-"), "no grace armed: {stderr}");
    // review(router-r2): positive control for the CR-K absence below — the
    // verifier's configuration boundary DID run on this row (CR-H, not
    // ablated, applied its value there), so a missing CR-K line is the
    // ablation, not a row that never reached the boundary.
    assert!(
        stderr.contains(
            "[router] derived rule=CR-H key=bab.initial_bounds_warmup_fallback from=false \
             to=true source=router"
        ),
        "positive control: the verifier boundary never ran, so the CR-K check below \
         would be vacuous:\n{stderr}"
    );
    assert!(
        stderr.contains("[router] lane slot=verifier event=claim rule=CR-T terminal=1"),
        "{stderr}"
    );
    assert!(
        !stderr.contains("[router] derived rule=CR-K"),
        "CR-K ablated derives nothing: {stderr}"
    );
    for decl in [&ROUTER_RULE_PUBLISH_GRACE, &ROUTER_RULE_DERIVED_CAPS] {
        let entry = lever_entry(&ablated.flight, decl.name);
        assert_eq!(entry["value"], false, "{}", decl.name);
        assert_eq!(entry["source"], "config", "{}", decl.name);
    }
    let kept = lever_entry(&ablated.flight, ROUTER_RULE_CAP_AUTHORITY.name);
    assert_eq!(kept["value"], true);
    assert!(ablated.result.starts_with("unsat"), "{:?}", ablated.result);
}

/// `shadow` never acts: no act/authority/lane/derived line, no commitments in
/// the record, and the published result equals `off`'s.
#[ntest::timeout(240000)]
#[test]
fn router_shadow_never_arms_the_authority() {
    let fixture = Fixture::new();
    fixture.preset("router_acting_shadow_2026", SHADOW_YAML);
    fixture.preset("router_acting_shadow_off_2026", OFF_YAML);
    let shadow = run(&fixture, "router_acting_shadow_2026", "acting_shadow");
    let off = run(
        &fixture,
        "router_acting_shadow_off_2026",
        "acting_shadow_off",
    );
    for line in shadow.router_lines() {
        for forbidden in [
            "[router] act ",
            "[router] authority",
            "[router] lane ",
            "[router] derived ",
            "[router] acting done",
        ] {
            assert!(
                !line.starts_with(forbidden),
                "shadow printed an acting line: {line}"
            );
        }
    }
    let record = &shadow.flight["router"];
    assert_eq!(record["effective"], "shadow");
    assert!(record["commitments"].is_null(), "{record}");
    assert!(record["acting"].is_null(), "{record}");
    assert_eq!(shadow.result, off.result);
}
