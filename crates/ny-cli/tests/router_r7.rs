// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r7 — the offline cost-model prior on the REAL scored entry point
//! (`ny vnncomp v1`), the `router_r5.rs` pattern: the tracked
//! `tests/models/simple_mlp.onnx`, a temporary `configs/vnncomp99/<cat>.yaml`
//! and the `<result>.flight.json` sidecar. No `benchmarks/` dependency.
//!
//! EVERY router `NY_*` (both modes, every rule switch and the prior switch) is
//! REMOVED from the child's environment by the declarations' own names, so
//! the typed yaml keys are the only route.
//!
//! The property is `Y_0 <= -1e6` over a unit box: impossible for this network,
//! so the published verdict is a deterministic `unsat`, and every arm must
//! publish exactly what `off` publishes.
//!
//! Host-agnostic: the compiled table is the GB10's, so on another host class
//! the prior prices nothing (`host:<class>`) — the tests assert the arrival
//! and the identity, and accept either pricing outcome by its own token.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use ny_levers::decls::router::{
    OPTIMIZER_MODE, OPTIMIZER_PRIORS, OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR,
    OPTIMIZER_RULE_IA_YIELD, OPTIMIZER_RULE_ROOT_ADMISSION, ROUTER_MODE, ROUTER_RULE_ADMISSION,
    ROUTER_RULE_CAP_AUTHORITY, ROUTER_RULE_CONTRACTS, ROUTER_RULE_DERIVED_CAPS,
    ROUTER_RULE_FAR_HAND_ON, ROUTER_RULE_GAP_CONTRACT, ROUTER_RULE_HAND_ON, ROUTER_RULE_LEASE,
    ROUTER_RULE_PUBLISH_GRACE, ROUTER_RULE_ROUTE, ROUTER_RULE_TERMINAL,
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
const SHADOW_PRIORS_FALSE_YAML: &str = "router:\n  mode: shadow\noptimizer:\n  priors: false\n";
const SHADOW_PRIORS_YAML: &str = "router:\n  mode: shadow\noptimizer:\n  priors: true\n";
const ON_PRIORS_YAML: &str = "router:\n  mode: \"on\"\noptimizer:\n  priors: true\n";
const OFF_PRIORS_YAML: &str = "router:\n  mode: \"off\"\noptimizer:\n  priors: true\n";

/// Every router declaration: none may reach the child through the environment.
fn router_decls() -> [&'static ny_levers::LeverDecl; 18] {
    [
        &ROUTER_MODE,
        &OPTIMIZER_MODE,
        &OPTIMIZER_PRIORS,
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

fn replay(run: &Run) {
    let out = Command::new(env!("CARGO_BIN_EXE_ny"))
        .current_dir(workspace_root())
        .args([
            "vnncomp",
            "plan",
            "--replay",
            run.sidecar.to_str().expect("sidecar UTF-8"),
        ])
        .output()
        .expect("run the replay");
    assert!(
        out.status.success(),
        "replay failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The key absent and the key `false` are the SAME run as R5's shadow: the
/// same plan hash, the same `uniform-v1`, no prior line, no prior field in
/// the recorded input, the same verdict.
#[ntest::timeout(240000)]
#[test]
fn router_r7_priors_absent_or_false_is_bit_identical_to_r5() {
    let fixture = Fixture::new();
    fixture.preset("router_r7_shadow_2026", SHADOW_YAML);
    fixture.preset("router_r7_false_2026", SHADOW_PRIORS_FALSE_YAML);
    let absent = run(&fixture, "router_r7_shadow_2026", "r7_absent");
    let explicit = run(&fixture, "router_r7_false_2026", "r7_false");
    for r in [&absent, &explicit] {
        let banner = r.line("[router] v1 ");
        assert!(banner.contains(" priors=uniform-v1 "), "{banner}");
        assert!(
            !r.stderr().contains("[router] priors"),
            "no prior is read unless the key asks:\n{}",
            r.stderr()
        );
        assert!(r.flight["router"]["input"].get("priors").is_none());
        assert!(!r.stderr().contains("rule=rule.priors"));
    }
    let sha = |r: &Run| {
        r.line("[router] plan sha=")
            .split_whitespace()
            .nth(2)
            .expect("plan sha token")
            .to_owned()
    };
    assert_eq!(
        sha(&absent),
        sha(&explicit),
        "an explicit `false` must plan exactly as the absent key"
    );
    assert_eq!(absent.result, explicit.result);
    assert_eq!(
        lever_entry(&absent.flight, OPTIMIZER_PRIORS.name)["source"],
        "default"
    );
    assert_eq!(
        lever_entry(&explicit.flight, OPTIMIZER_PRIORS.name)["source"],
        "config"
    );
    assert_eq!(
        lever_entry(&explicit.flight, OPTIMIZER_PRIORS.name)["value"],
        false
    );
    replay(&absent);
}

/// `optimizer.priors: true` arrives by the yaml route with zero router `NY_*`:
/// the banner carries `priors=cost-v1+<sha8>`, a `[router] priors` line says
/// what was priced (or why not, by its own token), the record carries the
/// prior view, the record replays, and the verdict is `off`'s.
#[ntest::timeout(240000)]
#[test]
fn router_r7_priors_arrive_by_yaml_route_and_change_no_verdict() {
    let fixture = Fixture::new();
    fixture.preset("router_r7_off_2026", OFF_YAML);
    fixture.preset("router_r7_priors_2026", SHADOW_PRIORS_YAML);
    fixture.preset("router_r7_on_priors_2026", ON_PRIORS_YAML);
    let off = run(&fixture, "router_r7_off_2026", "r7_off");
    let shadow = run(&fixture, "router_r7_priors_2026", "r7_priors");
    let acting = run(&fixture, "router_r7_on_priors_2026", "r7_on_priors");
    for r in [&shadow, &acting] {
        let banner = r.line("[router] v1 ");
        assert!(banner.contains(" priors=cost-v1+"), "{banner}");
        let line = r.line("[router] priors ");
        assert!(line.starts_with("[router] priors v=cost-v1+"), "{line}");
        let record = &r.flight["router"]["input"]["priors"];
        assert!(record["version"]
            .as_str()
            .is_some_and(|v| v.starts_with("cost-v1+")));
        let host = r.line("[router] v1 ");
        // v1 prices no first rung anywhere (G7 refused the attack phase, G6
        // the verifier's first window), so `priced=-` on every host.
        assert!(
            line.contains(" priced=- "),
            "v1 prices no first rung: {line}"
        );
        if host.contains(" host=gb10 ") {
            assert!(
                line.contains(" predict=verifier.setup:"),
                "on the measured host the shipped phase is predicted: {line}"
            );
            assert!(
                line.contains("attack.upfront_apgd:attack.upfront:refused"),
                "{line}"
            );
            assert!(
                line.contains("relu_split.graph:root.bound:refused"),
                "{line}"
            );
        } else {
            assert!(
                line.contains(" predict=- "),
                "off its host the prior predicts nothing: {line}"
            );
            assert!(line.contains(":host:"), "{line}");
        }
        assert!(
            r.stderr().contains("[router] done faults=0"),
            "{}",
            r.stderr()
        );
        assert_eq!(
            off.result, r.result,
            "reading the prior changed the verdict"
        );
        let ambient = r.flight["ambient_env"].as_object().expect("ambient env");
        for decl in router_decls() {
            assert!(
                !ambient.contains_key(decl.name),
                "{} reached the child",
                decl.name
            );
        }
        let entry = lever_entry(&r.flight, OPTIMIZER_PRIORS.name);
        assert_eq!(entry["value"], true);
        assert_eq!(entry["source"], "config");
        replay(r);
    }
    assert!(off.result.starts_with("unsat"), "{:?}", off.result);
    assert!(
        off.router_lines().is_empty(),
        "router off printed {:?}",
        off.router_lines()
    );
}

/// The key without a router does NOTHING: no session, no line, no record.
#[ntest::timeout(240000)]
#[test]
fn router_r7_priors_without_a_router_do_nothing() {
    let fixture = Fixture::new();
    fixture.preset("router_r7_off_2026", OFF_YAML);
    fixture.preset("router_r7_off_priors_2026", OFF_PRIORS_YAML);
    let off = run(&fixture, "router_r7_off_2026", "r7_plain_off");
    let keyed = run(&fixture, "router_r7_off_priors_2026", "r7_off_priors");
    assert!(
        keyed.router_lines().is_empty(),
        "{:?}",
        keyed.router_lines()
    );
    assert!(keyed.flight.get("router").is_none_or(Value::is_null));
    assert_eq!(off.result, keyed.result);
    assert_eq!(
        lever_entry(&keyed.flight, OPTIMIZER_PRIORS.name)["value"],
        true
    );
}
