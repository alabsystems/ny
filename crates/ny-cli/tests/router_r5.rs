// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r5 — acting set 4 (CR-RS capabilities routed only at SR.seal,
//! CR-G gap-order contracts) on the REAL scored entry point (`ny vnncomp
//! v1`), the `router_r4.rs` pattern: the tracked `tests/models/simple_mlp.onnx`,
//! a temporary `configs/vnncomp99/<category>.yaml`, and the
//! `<result>.flight.json` sidecar. No `benchmarks/` dependency.
//!
//! EVERY router `NY_*` (the two modes and every rule switch, R5's two
//! included) is REMOVED from the child's environment by the declarations' own
//! names, so the typed yaml keys are the only route.
//!
//! The property is `Y_0 <= -1e6` over a unit box: impossible for this network,
//! so the published verdict is a deterministic `unsat`. With the REAL
//! registry no capability is sealed, so the acting router routes NOTHING and
//! must publish exactly what `off` publishes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ny_levers::decls::router::{
    OPTIMIZER_MODE, OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR, OPTIMIZER_RULE_IA_YIELD,
    OPTIMIZER_RULE_ROOT_ADMISSION, ROUTER_MODE, ROUTER_RULE_ADMISSION, ROUTER_RULE_CAP_AUTHORITY,
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

const OFF_YAML: &str = "router:\n  mode: \"off\"\n";
const SHADOW_YAML: &str = "router:\n  mode: shadow\n";
const ON_YAML: &str = "router:\n  mode: \"on\"\n";
const ABLATED_YAML: &str = "router:\n  mode: \"on\"\n  rules:\n    route: false\n    \
                            gap_contract: false\n";
const PINNED_YAML: &str = "router:\n  mode: \"on\"\nmargin_row:\n  frozen_dual_bab: false\n";

/// The routable capabilities of the real registry, in registry order.
const ROUTABLE: [&str; 10] = [
    // #prepared-root-route: a lane of its own (registered among the provers).
    "prepared-root",
    "frozen-dual-bab",
    "wave-beta-bab-cuda",
    "wave-beta-bab",
    "bi-window-joint-refresh",
    "cgan-child-zono",
    "fwd-root-screen",
    "la-warm-frontier",
    "vit-cover-certify",
    // #kfsb-production.
    "kfsb-single-shot",
];

/// Every router declaration: none may reach the child through the environment.
fn router_decls() -> [&'static ny_levers::LeverDecl; 17] {
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

    /// A byte-identical copy of the model and the property under new names.
    fn renamed(&self, model_name: &str, property_name: &str) -> (PathBuf, PathBuf) {
        let model = self.root.path().join(model_name);
        fs::copy(&self.model, &model).expect("copy the model");
        let property = self.root.path().join(property_name);
        fs::copy(&self.property, &property).expect("copy the property");
        (model, property)
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

    fn route_lines(&self) -> Vec<String> {
        self.router_lines()
            .into_iter()
            .filter(|line| line.starts_with("[router] route "))
            .collect()
    }
}

fn run_named(fixture: &Fixture, category: &str, case: &str, model: &Path, property: &Path) -> Run {
    let result = fixture.root.path().join(format!("{case}.result"));
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

fn run(fixture: &Fixture, category: &str, case: &str) -> Run {
    run_named(fixture, category, case, &fixture.model, &fixture.property)
}

fn lever_entry<'a>(flight: &'a Value, name: &str) -> &'a Value {
    flight["levers"]["receipt"]["levers"]
        .as_array()
        .expect("lever entries")
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("{name} is not receipted"))
}

/// `router.mode: on` arrives by the yaml route with zero router `NY_*`; with
/// the REAL registry acting set 4 routes NOTHING (no capability is sealed):
/// one decision per routable capability, each refused at `sr:decl` with the
/// moat refusal; CR-G arms (it needs no seal); the record replays; and the
/// published verdict is exactly `off`'s.
#[ntest::timeout(240000)]
#[test]
fn router_r5_on_arrives_by_yaml_route_and_routes_nothing_with_the_real_registry() {
    let fixture = Fixture::new();
    fixture.preset("router_r5_off_2026", OFF_YAML);
    fixture.preset("router_r5_on_2026", ON_YAML);
    let off = run(&fixture, "router_r5_off_2026", "r5_off");
    let on = run(&fixture, "router_r5_on_2026", "r5_on");
    let stderr = on.stderr();
    assert!(
        stderr.contains("[router] v1 mode=on effective=acting"),
        "the yaml key must arm the acting router:\n{stderr}"
    );
    let cr_rs = on.line("[router] act rule=CR-RS ");
    assert!(
        cr_rs.contains(" acts=0 blocked_by=no_sealed_capability "),
        "CR-RS routes nothing with the real registry: {cr_rs}"
    );
    assert!(cr_rs.contains("routed=0"), "{cr_rs}");
    assert!(
        on.line("[router] act rule=CR-G ").contains(" acts=1 "),
        "CR-G arms under on"
    );
    for cap in ROUTABLE {
        let line = on.line(&format!("[router] route cap={cap} "));
        assert!(
            line.contains(" sr=decl routed=0 blocked_by=sr:decl pin_debt=0 "),
            "{cap}: {line}"
        );
        assert!(line.contains("moat:high_unsealed"), "{cap}: {line}");
    }
    assert_eq!(
        on.line("[router] route armed "),
        "[router] route armed rule=CR-RS routes=[-] gap_contract=1"
    );
    assert!(
        on.line("[router] route done ")
            .starts_with("[router] route done armed=1 routed=0 consults=[-] applied=0 "),
        "{stderr}"
    );
    assert!(
        !stderr.contains("[router] route applied"),
        "nothing routed, nothing applied:\n{stderr}"
    );
    assert!(stderr.contains("[router] done faults=0"), "{stderr}");
    // Zero router NY_*: the mode came from the preset, the R5 switches from
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
    for decl in [&ROUTER_RULE_ROUTE, &ROUTER_RULE_GAP_CONTRACT] {
        let entry = lever_entry(&on.flight, decl.name);
        assert_eq!(entry["value"], true, "{}", decl.name);
        assert_ne!(entry["source"], "legacy_env", "{}", decl.name);
    }
    let record = &on.flight["router"];
    assert_eq!(record["r5_commitments"]["acting"], true);
    assert_eq!(record["r5_commitments"]["gap_contract"], true);
    let routes = record["r5_commitments"]["routes"]
        .as_array()
        .expect("route decisions");
    assert_eq!(routes.len(), ROUTABLE.len());
    assert!(routes.iter().all(|d| d["routed"] == false));
    assert_eq!(record["r5_input"]["pinned_keys"], serde_json::json!([]));
    assert_eq!(record["route"]["armed"], true);
    // The facts the router read carry a COMPLETE operator signature (the
    // scan's tokens), so the refusal is the seal, never unknown facts.
    assert_eq!(record["input"]["facts"]["ops_complete"], true);
    // The acting router publishes what `off` publishes on this row.
    assert!(off.result.starts_with("unsat"), "{:?}", off.result);
    assert_eq!(off.result, on.result, "acting set 4 changed the verdict");
    assert!(
        off.router_lines().is_empty(),
        "router off printed {:?}",
        off.router_lines()
    );
    // Replay re-derives the plan and every acting set's commitments.
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
        "replay of an R5 record failed: {}\n{}",
        String::from_utf8_lossy(&replay.stdout),
        String::from_utf8_lossy(&replay.stderr)
    );
}

/// Each R5 rule is its own capability: a yaml `router.rules.<rule>: false`
/// ablates exactly that rule (zero `NY_*`), which says why, and arms nothing.
#[ntest::timeout(240000)]
#[test]
fn router_r5_rule_keys_ablate_their_rule_by_yaml_route() {
    let fixture = Fixture::new();
    fixture.preset("router_r5_ablated_2026", ABLATED_YAML);
    let ablated = run(&fixture, "router_r5_ablated_2026", "r5_ablated");
    for rule in ["CR-RS", "CR-G"] {
        let line = ablated.line(&format!("[router] act rule={rule} "));
        assert!(
            line.contains(" acts=0 blocked_by=key "),
            "{rule} must be ablated by its key: {line}"
        );
    }
    let stderr = ablated.stderr();
    assert!(
        ablated.line("[router] act rule=CR-1 ").contains(" acts=1 "),
        "positive control: the R2 rules still act on this row\n{stderr}"
    );
    assert_eq!(
        ablated.line("[router] route idle "),
        "[router] route idle rule=CR-RS reason=nothing_routed_and_no_contract"
    );
    for cap in ROUTABLE {
        assert!(
            ablated
                .line(&format!("[router] route cap={cap} "))
                .contains(" routed=0 blocked_by=key "),
            "{cap}"
        );
    }
    for decl in [&ROUTER_RULE_ROUTE, &ROUTER_RULE_GAP_CONTRACT] {
        let entry = lever_entry(&ablated.flight, decl.name);
        assert_eq!(entry["value"], false, "{}", decl.name);
        assert_eq!(entry["source"], "config", "{}", decl.name);
    }
    assert_eq!(ablated.flight["router"]["route"]["armed"], false);
    assert!(ablated.result.starts_with("unsat"), "{:?}", ablated.result);
}

/// `shadow` never routes: no route line, nothing in the record, and the
/// published result equals `off`'s.
#[ntest::timeout(240000)]
#[test]
fn router_r5_shadow_never_routes() {
    let fixture = Fixture::new();
    fixture.preset("router_r5_shadow_2026", SHADOW_YAML);
    fixture.preset("router_r5_shadow_off_2026", OFF_YAML);
    let shadow = run(&fixture, "router_r5_shadow_2026", "r5_shadow");
    let off = run(&fixture, "router_r5_shadow_off_2026", "r5_shadow_off");
    assert!(
        shadow.route_lines().is_empty(),
        "shadow printed {:?}",
        shadow.route_lines()
    );
    for line in shadow.router_lines() {
        assert!(
            !line.starts_with("[router] act rule=CR-RS")
                && !line.starts_with("[router] act rule=CR-G"),
            "shadow printed an acting line: {line}"
        );
    }
    let record = &shadow.flight["router"];
    assert_eq!(record["effective"], "shadow");
    assert!(record["r5_commitments"].is_null(), "{record}");
    assert!(record["route"].is_null(), "{record}");
    assert_eq!(shadow.result, off.result);
}

/// NAME-BLIND: the same model and property under different file names and
/// category names give the SAME route decisions, the same commitments and
/// the same static plan hash — nothing the harness calls them enters.
#[ntest::timeout(240000)]
#[test]
fn router_r5_routing_is_identical_under_random_category_and_file_names() {
    let fixture = Fixture::new();
    fixture.preset("router_r5_names_alpha_2026", ON_YAML);
    fixture.preset("zz_unrelated_benchmark_2025", ON_YAML);
    let (model_a, property_a) = fixture.renamed("alpha_net.onnx", "prop_alpha.vnnlib");
    let (model_b, property_b) = fixture.renamed("q9_model_final_v3.onnx", "zz_spec_7.vnnlib");
    let a = run_named(
        &fixture,
        "router_r5_names_alpha_2026",
        "r5_names_a",
        &model_a,
        &property_a,
    );
    let b = run_named(
        &fixture,
        "zz_unrelated_benchmark_2025",
        "r5_names_b",
        &model_b,
        &property_b,
    );
    assert_eq!(
        a.route_lines().len(),
        ROUTABLE.len() + 2,
        "{:?}",
        a.route_lines()
    );
    assert_eq!(
        a.route_lines(),
        b.route_lines(),
        "route lines differ by name"
    );
    let (ra, rb) = (&a.flight["router"], &b.flight["router"]);
    assert_eq!(ra["r5_commitments"], rb["r5_commitments"]);
    assert_eq!(ra["r5_input"], rb["r5_input"]);
    assert_eq!(ra["plan"]["static_hash"], rb["plan"]["static_hash"]);
    assert_eq!(a.result, b.result);
}

/// A category file's pin reaches the router: `margin_row.frozen_dual_bab`
/// set in the yaml (here to its default) is a pin, so the tier's route is
/// refused `pin:<key>` too — and it is NOT pin debt, because the tier is
/// not sealed.
#[ntest::timeout(240000)]
#[test]
fn router_r5_a_yaml_pin_is_read_by_the_router() {
    let fixture = Fixture::new();
    fixture.preset("router_r5_pinned_2026", PINNED_YAML);
    let pinned = run(&fixture, "router_r5_pinned_2026", "r5_pinned");
    let fd = pinned.line("[router] route cap=frozen-dual-bab ");
    assert!(fd.contains("pin:margin_row.frozen_dual_bab"), "{fd}");
    assert!(
        fd.contains(" pin_debt=0 "),
        "an unsealed capability's pin is not debt: {fd}"
    );
    assert_eq!(
        pinned.flight["router"]["r5_input"]["pinned_keys"],
        serde_json::json!(["margin_row.frozen_dual_bab"])
    );
    let wave = pinned.line("[router] route cap=wave-beta-bab ");
    assert!(!wave.contains("pin:"), "only the pinned key: {wave}");
    assert!(pinned.result.starts_with("unsat"), "{:?}", pinned.result);
}
