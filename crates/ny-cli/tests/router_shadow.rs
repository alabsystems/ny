// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r1 — ROUTER_PROGRAM §11.2 tests 26–32 and 34, on the REAL scored
//! entry point (`ny vnncomp v1`), the `flight_lever_receipt.rs` pattern: the
//! tracked `tests/models/simple_mlp.onnx`, a temporary
//! `configs/vnncomp99/<category>.yaml`, and the `<result>.flight.json` sidecar.
//! No `benchmarks/` dependency (the worktree trap).
//!
//! The router's two modes are REMOVED from every child's environment (by the
//! declarations' own names), so the typed yaml key is the only route — except
//! the one invalid-preset case that deliberately exercises the entry-time
//! environment authority.
//!
//! The property is `Y_0 <= -1e6` over a unit box: impossible for this network,
//! so the published verdict is a deterministic `unsat` and off/shadow can be
//! compared byte for byte.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ny_levers::decls::router::{OPTIMIZER_MODE, ROUTER_MODE};
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

/// A valid preset that does not mention the router (an empty file is not a
/// valid preset document).
const ABSENT_YAML: &str = "margin_row: {}\n";
const SHADOW_YAML: &str = "router:\n  mode: shadow\n";
const OFF_YAML: &str = "router:\n  mode: \"off\"\n";
const INVALID_YAML: &str =
    "router:\n  mode: shadow\nsolver:\n  alpha_crown:\n    alpha_zero_yield_frac: 0.95\n";

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
            .filter(|line| line.starts_with("[router]") || line.starts_with("[optimizer]"))
            .map(str::to_owned)
            .collect()
    }

    fn plan_sha(&self) -> String {
        let line = self
            .router_lines()
            .into_iter()
            .find(|line| line.starts_with("[router] plan sha="))
            .unwrap_or_else(|| panic!("no `[router] plan sha=` line:\n{}", self.stderr()));
        line.trim_start_matches("[router] plan sha=")
            .split_whitespace()
            .next()
            .expect("sha token")
            .to_owned()
    }

    fn events(&self) -> Vec<(String, String)> {
        self.flight["events"]
            .as_array()
            .expect("flight events")
            .iter()
            .map(|event| {
                (
                    event["method"].as_str().unwrap_or_default().to_owned(),
                    event["status"].as_str().unwrap_or_default().to_owned(),
                )
            })
            .collect()
    }

    fn event_index(&self, method: &str) -> Option<usize> {
        self.events().iter().position(|(m, _)| m == method)
    }
}

fn run_case(
    fixture: &Fixture,
    category: &str,
    case: &str,
    model: &Path,
    property: &Path,
    env: &[(&str, &str)],
    extra: &[&str],
) -> Run {
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
            "5",
            "--configs-dir",
            fixture.configs.to_str().expect("configs path UTF-8"),
        ])
        .args(extra)
        .env_remove(ROUTER_MODE.name)
        .env_remove(OPTIMIZER_MODE.name)
        // As in `flight_lever_receipt.rs`: retain no out-of-process watchdog
        // helper after the child exits (a lingering `ny` also trips the box's
        // `pgrep` quiescence gate). Identical in every arm, so it cannot
        // separate off from shadow.
        .env("NY_VNNCOMP_EXTERNAL_WATCHDOG", "0");
    for (name, value) in env {
        command.env(name, value);
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
    run_case(
        fixture,
        category,
        case,
        &fixture.model,
        &fixture.property,
        &[],
        &[],
    )
}

fn lever_entry<'a>(flight: &'a Value, name: &str) -> &'a Value {
    flight["levers"]["receipt"]["levers"]
        .as_array()
        .expect("lever entries")
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("{name} is not receipted"))
}

/// The receipt without the router's own two entries.
fn non_router_receipt(flight: &Value) -> Value {
    let mut receipt = flight["levers"]["receipt"].clone();
    if let Some(levers) = receipt["levers"].as_array_mut() {
        levers.retain(|entry| {
            entry["name"] != ROUTER_MODE.name && entry["name"] != OPTIMIZER_MODE.name
        });
    }
    receipt
}

/// Flight events as a multiset of (method, status), router events excluded.
fn non_router_events(run: &Run) -> BTreeMap<(String, String), usize> {
    let mut out = BTreeMap::new();
    for event in run.events() {
        if !event.0.starts_with("router") {
            *out.entry(event).or_insert(0) += 1;
        }
    }
    out
}

fn code_only(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[ntest::timeout(180000)]
#[test]
fn router_off_emits_nothing_and_extracts_no_facts() {
    let fixture = Fixture::new();
    fixture.preset("router_absent_2026", ABSENT_YAML);
    fixture.preset("router_off_2026", OFF_YAML);
    for category in ["router_absent_2026", "router_off_2026"] {
        let off = run(&fixture, category, category);
        assert!(
            off.router_lines().is_empty(),
            "{category}: the router printed with mode off: {:?}",
            off.router_lines()
        );
        assert!(
            off.flight.get("router").is_none(),
            "{category}: an off run carries no router record"
        );
        assert!(
            off.events()
                .iter()
                .all(|(method, _)| !method.starts_with("router")),
            "{category}: an off run notes no router event"
        );
        let mode = lever_entry(&off.flight, ROUTER_MODE.name);
        assert_eq!(mode["value"], "off");
    }
    // Structural half: off extracts no facts. The entry point builds a start
    // context (and stats the model) only behind `requested`, and `start`
    // returns on `is_off` before it computes the policy input.
    let entry = code_only(
        &fs::read_to_string(workspace_root().join("crates/ny-cli/src/commands/vnncomp.rs"))
            .expect("read vnncomp.rs"),
    );
    let starts = entry.matches("crate::router::start(").count();
    let gates = entry.matches("crate::router::requested").count();
    assert!(starts >= 2, "both routes start the router ({starts})");
    assert_eq!(starts, gates, "every start site is gated by `requested`");
    let router = code_only(
        &fs::read_to_string(workspace_root().join("crates/ny-cli/src/router/mod.rs"))
            .expect("read router/mod.rs"),
    );
    let start_fn = router.find("pub(crate) fn start(").expect("fn start");
    let off_return = router[start_fn..]
        .find("if policy.is_off()")
        .expect("the off early return");
    let facts = router[start_fn..]
        .find("policy_input(ctx)")
        .expect("the facts extraction");
    assert!(
        off_return < facts,
        "start must return on off BEFORE extracting facts"
    );
}

#[ntest::timeout(180000)]
#[test]
fn router_shadow_arrives_by_yaml_route() {
    let fixture = Fixture::new();
    fixture.preset("router_shadow_2026", SHADOW_YAML);
    let shadow = run(&fixture, "router_shadow_2026", "shadow");
    let stderr = shadow.stderr();
    assert!(
        stderr.contains("[router] v1 mode=shadow effective=shadow"),
        "the yaml key must arm the shadow router:\n{stderr}"
    );
    assert!(stderr.contains("authority=frozen"), "{stderr}");
    assert!(stderr.contains("[router] done faults=0"), "{stderr}");
    let mode = lever_entry(&shadow.flight, ROUTER_MODE.name);
    assert_eq!(mode["value"], "shadow");
    assert_eq!(
        mode["source"], "config",
        "the mode arrives from the typed preset"
    );
    let ambient = shadow.flight["ambient_env"]
        .as_object()
        .expect("ambient env");
    assert!(
        !ambient.contains_key(ROUTER_MODE.name) && !ambient.contains_key(OPTIMIZER_MODE.name),
        "no router NY_* may be in the child environment: {ambient:?}"
    );
    let record = &shadow.flight["router"];
    assert_eq!(record["schema"], "ny-router/v1");
    assert_eq!(record["mode"], "shadow");
    assert_eq!(record["effective"], "shadow");
    assert!(record["plan"]["emitted"]
        .as_array()
        .is_some_and(Vec::is_empty));
}

#[ntest::timeout(180000)]
#[test]
fn router_shadow_changes_no_verdict_and_no_non_router_receipt_field() {
    let fixture = Fixture::new();
    fixture.preset("router_identity_off_2026", OFF_YAML);
    fixture.preset("router_identity_shadow_2026", SHADOW_YAML);
    let off = run(&fixture, "router_identity_off_2026", "identity_off");
    let shadow = run(&fixture, "router_identity_shadow_2026", "identity_shadow");
    assert_eq!(
        off.result, shadow.result,
        "shadow changed the published result file"
    );
    assert!(
        off.result.starts_with("unsat"),
        "the fixture's verdict is a deterministic unsat: {:?}",
        off.result
    );
    assert_eq!(
        non_router_receipt(&off.flight),
        non_router_receipt(&shadow.flight),
        "shadow changed a non-router lever receipt field"
    );
    assert_eq!(
        non_router_events(&off),
        non_router_events(&shadow),
        "shadow changed the flight's non-router events"
    );
    assert_eq!(
        off.flight["levers"]["status"],
        shadow.flight["levers"]["status"]
    );
    assert_eq!(
        off.flight["schema_version"],
        shadow.flight["schema_version"]
    );
    assert!(off.flight.get("router").is_none());
    assert!(shadow.flight.get("router").is_some());
}

#[ntest::timeout(180000)]
#[test]
fn router_done_line_on_every_exit_path() {
    let fixture = Fixture::new();
    // Normal verdict path.
    fixture.preset("router_done_normal_2026", SHADOW_YAML);
    let normal = run(&fixture, "router_done_normal_2026", "done_normal");
    assert!(normal.stderr().contains("[router] done faults=0"));
    assert_eq!(normal.flight["router"]["done"]["verdict"], "unsat");

    // Invalid preset: no typed layer exists, the run publishes `unknown`, and
    // the router (asked for through the entry-time environment) still closes.
    fixture.preset("router_done_invalid_2026", INVALID_YAML);
    let invalid = run_case(
        &fixture,
        "router_done_invalid_2026",
        "done_invalid",
        &fixture.model,
        &fixture.property,
        &[(ROUTER_MODE.name, "shadow")],
        &[],
    );
    assert!(
        invalid.result.starts_with("unknown"),
        "{:?}",
        invalid.result
    );
    assert_eq!(invalid.flight["levers"]["status"], "invalid_config");
    let stderr = invalid.stderr();
    assert!(stderr.contains("authority=entry_env"), "{stderr}");
    assert!(stderr.contains("[router] done faults=0"), "{stderr}");
    assert_eq!(invalid.flight["router"]["done"]["verdict"], "unknown");

    // Every other verdict exit path (timeout, error after the freeze, the
    // lane early-publish) records its verdict through `crate::flight::finish`,
    // which closes the router after publication: structural, so a new exit
    // path cannot skip it without also skipping the verdict record.
    let flight_src = code_only(
        &fs::read_to_string(workspace_root().join("crates/ny-cli/src/flight.rs"))
            .expect("read flight.rs"),
    );
    let finish_fn = flight_src
        .find("pub(crate) fn finish(verdict: &str)")
        .expect("the free finish");
    let body = &flight_src[finish_fn..];
    let body = &body[..body.find("\n}").expect("closed fn")];
    let recorded = body
        .find("global().finish(verdict)")
        .expect("records the verdict");
    let closed = body
        .find("crate::router::finish_after_publication(verdict)")
        .expect("closes the router");
    assert!(
        recorded < closed,
        "the router closes AFTER the verdict is recorded"
    );
    let entry = code_only(
        &fs::read_to_string(workspace_root().join("crates/ny-cli/src/commands/vnncomp.rs"))
            .expect("read vnncomp.rs"),
    );
    assert!(
        entry.matches("crate::flight::finish(").count() >= 10,
        "positive control: the verdict exit paths call flight::finish"
    );
}

#[ntest::timeout(240000)]
#[test]
fn router_plan_is_identical_under_random_category_and_file_names() {
    let fixture = Fixture::new();
    fixture.preset("router_name_a_2026", SHADOW_YAML);
    fixture.preset("zq8f3k_unrelated_x_2026", SHADOW_YAML);
    let renamed_model = fixture.root.path().join("w7q2_totally_other_name.onnx");
    fs::copy(&fixture.model, &renamed_model).expect("copy the model");
    let renamed_property = fixture.root.path().join("p9_other.vnnlib");
    fs::copy(&fixture.property, &renamed_property).expect("copy the property");
    let a = run(&fixture, "router_name_a_2026", "names_a");
    let b = run_case(
        &fixture,
        "zq8f3k_unrelated_x_2026",
        "names_b",
        &renamed_model,
        &renamed_property,
        &[],
        &[],
    );
    assert_eq!(
        a.plan_sha(),
        b.plan_sha(),
        "the plan moved with the category or file names"
    );
    assert_eq!(a.flight["router"]["input"], b.flight["router"]["input"]);
    assert_eq!(a.flight["router"]["plan"], b.flight["router"]["plan"]);
}

#[ntest::timeout(240000)]
#[test]
fn router_plan_hash_identical_across_proof_profiles() {
    let fixture = Fixture::new();
    fixture.preset("router_profile_2026", SHADOW_YAML);
    let run_profile = |profile: &str| {
        run_case(
            &fixture,
            "router_profile_2026",
            &format!("profile_{profile}"),
            &fixture.model,
            &fixture.property,
            &[],
            &["--proof-profile", profile],
        )
    };
    let proof = run_profile("proof");
    let competition = run_profile("competition");
    assert_eq!(proof.flight["proof_profile"]["mode"], "proof");
    assert_eq!(competition.flight["proof_profile"]["mode"], "competition");
    assert_eq!(
        proof.plan_sha(),
        competition.plan_sha(),
        "the plan hash must not depend on the proof profile"
    );
}

#[ntest::timeout(180000)]
#[test]
fn router_replay_reproduces_logged_plan_and_amendments() {
    let fixture = Fixture::new();
    fixture.preset("router_replay_2026", SHADOW_YAML);
    let shadow = run(&fixture, "router_replay_2026", "replay");
    let replay = Command::new(env!("CARGO_BIN_EXE_ny"))
        .current_dir(workspace_root())
        .args([
            "vnncomp",
            "plan",
            "--replay",
            shadow.sidecar.to_str().expect("sidecar UTF-8"),
        ])
        .output()
        .expect("run the replay");
    let stdout = String::from_utf8_lossy(&replay.stdout);
    assert!(
        replay.status.success(),
        "replay failed: {}\n{}",
        stdout,
        String::from_utf8_lossy(&replay.stderr)
    );
    assert!(
        stdout.contains(&format!(
            "[router] replay ok plan_sha={}",
            shadow.plan_sha()
        )),
        "{stdout}"
    );
    // Negative control: a tampered plan must DIVERGE (non-zero exit).
    let mut tampered = shadow.flight;
    let window = tampered["router"]["plan"]["window_ms"]
        .as_u64()
        .expect("window_ms");
    tampered["router"]["plan"]["window_ms"] = Value::from(window + 1);
    let tampered_path = fixture.root.path().join("tampered.flight.json");
    fs::write(
        &tampered_path,
        serde_json::to_vec(&tampered).expect("serialize"),
    )
    .expect("write the tampered record");
    let diverged = Command::new(env!("CARGO_BIN_EXE_ny"))
        .current_dir(workspace_root())
        .args([
            "vnncomp",
            "plan",
            "--replay",
            tampered_path.to_str().expect("UTF-8"),
        ])
        .output()
        .expect("run the tampered replay");
    assert!(!diverged.status.success(), "a tampered plan replayed clean");
    assert!(String::from_utf8_lossy(&diverged.stderr).contains("DIVERGED"));
}

#[ntest::timeout(180000)]
#[test]
fn router_retrospective_checkpoints_run_after_publication() {
    let fixture = Fixture::new();
    fixture.preset("router_order_2026", SHADOW_YAML);
    let shadow = run(&fixture, "router_order_2026", "order");
    let plan = shadow
        .event_index("router_plan")
        .expect("router_plan event");
    let publish = shadow
        .event_index("result_publish")
        .expect("result_publish event");
    let checkpoint = shadow
        .event_index("router_checkpoint")
        .expect("router_checkpoint event");
    let done = shadow
        .event_index("router_done")
        .expect("router_done event");
    assert!(plan < publish, "the plan is logged before the lanes run");
    assert!(
        publish < checkpoint && checkpoint < done,
        "checkpoints run strictly AFTER publication: {:?}",
        shadow.events()
    );
    let record = &shadow.flight["router"];
    assert_eq!(record["done"]["published_before_ckpt"], true);
    assert_eq!(
        record["checkpoints"].as_array().map(Vec::len),
        Some(3),
        "c1, c2 and c3 are all evaluated retrospectively"
    );
    let stderr = shadow.stderr();
    let ckpt_line = stderr.find("[router] ckpt=c1").expect("the c1 line");
    let done_line = stderr.find("[router] done").expect("the done line");
    assert!(ckpt_line < done_line);
}

/// A validated two-network `isomorphic-to` property whose unsafe region is a
/// conjunction of contradictory deviation atoms (empty for every f and g): the
/// relational fixture of `flight_lever_receipt.rs`. It routes by the
/// property's own structure and finishes on the certified arithmetic shortcut
/// without reading the ONNX argument.
const RELATIONAL_PROPERTY: &str = "(vnnlib-version 2.0)
(declare-network f
  (declare-input X_f Float32 [2])
  (declare-output Y_f Float32 [2])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Float32 [2])
  (declare-output Y_g Float32 [2])
)
(assert (and (>= X_f[0] -1) (<= X_f[0] 1)))
(assert (and (>= X_g[0] -1) (<= X_g[0] 1)))
(assert (and (>= X_f[1] -1) (<= X_f[1] 1)))
(assert (and (>= X_g[1] -1) (<= X_g[1] 1)))
(assert (== X_f[0] X_g[0]))
(assert (== X_f[1] X_g[1]))
(assert (and (> Y_g[0] (+ Y_f[0] 0.05)) (< Y_g[0] (- Y_f[0] 0.05))))
(assert (and (> Y_g[1] (+ Y_f[1] 0.05)) (< Y_g[1] (- Y_f[1] 0.05))))
";

/// Extra (not one of the brief's 35): the relational route freezes only what
/// it executes, and it runs the router, so the yaml key must arrive there too.
/// Before `relational_router_config_value` the key validated and resolved
/// `off` from the default layer on every dual-network row.
#[ntest::timeout(180000)]
#[test]
fn router_shadow_arrives_by_yaml_route_on_the_relational_route() {
    let fixture = Fixture::new();
    fixture.preset("router_relational_2026", SHADOW_YAML);
    let property = fixture.root.path().join("relational.vnnlib");
    fs::write(&property, RELATIONAL_PROPERTY).expect("write the dual-network property");
    let shadow = run_case(
        &fixture,
        "router_relational_2026",
        "relational_shadow",
        &fixture.model,
        &property,
        &[],
        &[],
    );
    let routed = shadow.flight["events"]
        .as_array()
        .expect("flight events")
        .iter()
        .any(|event| {
            event["method"] == "relational_route"
                && event["status"] == "ran"
                && event["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains("declared_networks=2"))
        });
    assert!(routed, "the fixture must take the dual-network route");
    let stderr = shadow.stderr();
    assert!(
        stderr.contains("[router] v1 mode=shadow effective=shadow"),
        "the yaml key must arm the router on the relational route:\n{stderr}"
    );
    assert!(stderr.contains("authority=frozen"), "{stderr}");
    assert!(
        stderr.contains("facts=unknown:relational-route"),
        "{stderr}"
    );
    assert!(stderr.contains("[router] done faults=0"), "{stderr}");
    let mode = lever_entry(&shadow.flight, ROUTER_MODE.name);
    assert_eq!(mode["value"], "shadow");
    assert_eq!(mode["source"], "config");
    assert!(shadow.flight.get("router").is_some());
}
