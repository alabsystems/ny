// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r1 `ny-cli` unit tests (run with `--bin ny`, never `--lib`):
//! ROUTER_PROGRAM §11.2 tests 33 and 35, plus the admission-debt ratchet and
//! the policy's own resolution contract.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ny_core::router::decl::CapKind;
use ny_core::router::facts::HostClass;
use ny_core::router::plan;
use ny_core::router::{OptimizerMode, RouterMode};

use super::{registry, Authority, RouterPolicy, StartContext};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    let path = workspace_root().join(rel);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        text.len() > 200,
        "{} is suspiciously short: the check would be VOID",
        path.display()
    );
    text
}

fn frozen(config: &[(&'static ny_levers::LeverDecl, &str)]) -> ny_levers::LeverSet {
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
    ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        config
            .iter()
            .find(|(candidate, _)| std::ptr::eq(*candidate, decl))
            .map(|(_, token)| ny_levers::LeverValue::Text((*token).to_owned()))
    })
    .expect("admissible router configuration")
}

pub(super) fn printer_ctx() -> StartContext<'static> {
    StartContext {
        levers: None,
        authority: Authority::Frozen,
        model: None,
        model_bytes: Some(10_000_000),
        facts_reason: None,
        declared_networks: Some(1),
        relational_selected: Some(false),
        official_budget_secs: 100,
        internal_secs: 95,
        join_margin: Duration::from_secs(2),
        anchor: Instant::now(),
        // A declared host: no host probe (and no device probe) in a unit test.
        target_host: Some(HostClass::M5),
        pinned_keys: Vec::new(),
        trial_pinned_keys: Vec::new(),
        // #admission-precision: no property read (every site check defers).
        spec: crate::commands::beta_crown::spec_shape::SpecRead::Unknown("printer-test"),
    }
}

#[test]
fn router_policy_is_dark_by_default_and_typed_from_the_frozen_levers() {
    use ny_levers::decls::router::{OPTIMIZER_MODE, ROUTER_MODE};
    let dark = frozen(&[]);
    assert_eq!(
        RouterPolicy::from_levers(&dark).expect("dark"),
        RouterPolicy::OFF
    );
    assert!(!super::requested(&dark), "off rows build no context");
    let typed = frozen(&[(&ROUTER_MODE, "shadow"), (&OPTIMIZER_MODE, "on")]);
    let policy = RouterPolicy::from_levers(&typed).expect("typed");
    assert_eq!(policy.mode, RouterMode::Shadow);
    assert_eq!(policy.optimizer, OptimizerMode::On);
    assert!(super::requested(&typed));
    assert_eq!(
        typed.resolved(&ROUTER_MODE).expect("declared").source,
        ny_levers::Source::Config,
        "the mode arrives from the typed layer"
    );
    // The optimizer alone also starts a session.
    let optimizer_only = frozen(&[(&OPTIMIZER_MODE, "shadow")]);
    assert!(super::requested(&optimizer_only));
}

#[test]
fn router_registry_ids_match_capability_registry() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&read("docs/engine_program_capabilities.yaml"))
            .expect("the capability registry parses");
    let capabilities = yaml
        .get("capabilities")
        .and_then(serde_yaml::Value::as_sequence)
        .expect("capabilities list");
    let program_ids: BTreeSet<String> = capabilities
        .iter()
        .filter_map(|c| c.get("id").and_then(serde_yaml::Value::as_str))
        .map(str::to_owned)
        .collect();
    assert!(
        program_ids.contains("frozen-dual-bab"),
        "positive control: the capability list must be read"
    );

    // Every routable entry: capability `routable:` blocks, then routable_lanes.
    let mut routable: BTreeMap<String, (String, &serde_yaml::Value)> = BTreeMap::new();
    let mut duplicates = Vec::new();
    for cap in capabilities {
        let Some(block) = cap.get("routable") else {
            continue;
        };
        let id = cap
            .get("id")
            .and_then(serde_yaml::Value::as_str)
            .expect("capability id")
            .to_owned();
        if routable
            .insert(id.clone(), ("capability".to_owned(), block))
            .is_some()
        {
            duplicates.push(id);
        }
    }
    for lane in yaml
        .get("routable_lanes")
        .and_then(serde_yaml::Value::as_sequence)
        .expect("routable_lanes list")
    {
        let id = lane
            .get("id")
            .and_then(serde_yaml::Value::as_str)
            .expect("routable_lanes id")
            .to_owned();
        if program_ids.contains(&id) {
            duplicates.push(format!(
                "{id} is a capability id: declare it through that capability's routable: block"
            ));
        }
        if routable
            .insert(id.clone(), ("routable_lanes".to_owned(), lane))
            .is_some()
        {
            duplicates.push(id);
        }
    }
    assert!(
        duplicates.is_empty(),
        "duplicate routable ids: {duplicates:?}"
    );

    let mut violations = Vec::new();
    for decl in registry::REGISTRY {
        let id = decl.id.as_str();
        let Some((_, block)) = routable.get(id) else {
            violations.push(format!(
                "registry id {id} has no routable entry in the yaml"
            ));
            continue;
        };
        let kind = match decl.kind {
            CapKind::Lane { .. } => "lane",
            CapKind::Tier { .. } => "tier",
            CapKind::Phase { .. } => "phase",
            CapKind::Variant { .. } => "variant",
            CapKind::Rule { .. } => "rule",
        };
        let recorded = block.get("kind").and_then(serde_yaml::Value::as_str);
        if recorded != Some(kind) {
            violations.push(format!(
                "{id}: yaml kind {recorded:?}, registry kind {kind}"
            ));
        }
        let file = block
            .get("decl")
            .and_then(|d| d.get("file"))
            .and_then(serde_yaml::Value::as_str);
        let grep = block
            .get("decl")
            .and_then(|d| d.get("grep"))
            .and_then(serde_yaml::Value::as_str);
        match (file, grep) {
            (Some(file), Some(grep)) => {
                if !read(file).contains(grep) {
                    violations.push(format!("{id}: decl grep {grep:?} not found in {file}"));
                }
                if !grep.contains(id) {
                    violations.push(format!("{id}: decl grep {grep:?} does not name the id"));
                }
            }
            _ => violations.push(format!("{id}: routable entry has no decl {{file, grep}}")),
        }
    }
    let registered: BTreeSet<&str> = registry::REGISTRY
        .iter()
        .map(|decl| decl.id.as_str())
        .collect();
    for id in routable.keys() {
        if !registered.contains(id.as_str()) {
            violations.push(format!(
                "yaml routable id {id} is not in crates/ny-cli/src/router/registry.rs"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "the router registry and docs/engine_program_capabilities.yaml disagree (one registry, \
         one identity — ROUTER_PROGRAM §3.4):\n  {}",
        violations.join("\n  ")
    );
    assert_eq!(registry::REGISTRY.len(), routable.len());
}

#[test]
fn router_emits_no_values_and_writes_no_cell_in_v1() {
    let ctx = printer_ctx();
    let (input, router_plan) = super::printer_plan(&ctx).expect("the printer plan");
    assert!(router_plan.emitted.is_empty(), "R1 emits no knob value");
    assert_eq!(
        plan::plan(&input),
        router_plan,
        "the plan is a pure function of its input"
    );
    assert_eq!(
        input.capabilities.len(),
        registry::REGISTRY.len(),
        "every registered capability is asked"
    );
    // #router-r2: acting set 1 is in the binary; the router still emits no
    // action-space value (the plan's `emitted` above), and the ONE place it
    // writes a verifier value is the rule-owned boundary in `acting.rs`
    // (checked below), never a preset cell or gate.
    // #router-r4: acting set 3 (CR-L, CR-F) follows acting set 1.
    // #router-r5: acting set 4 (CR-RS, CR-G) follows acting set 3.
    let mut acting_set: Vec<ny_core::router::RuleId> = ny_core::router::acting::R2_RULES.to_vec();
    acting_set.extend(ny_core::router::r4::R4_RULES);
    acting_set.extend(ny_core::router::route::R5_RULES);
    assert_eq!(ny_core::router::ACTING_RULES, acting_set.as_slice());
    // #router-r3: acting set 2 (the O-STOP rules) is in the binary too; they
    // end phases early through the stop authority and emit no knob value.
    // #router-r6: acting set 2 is followed by the four O-TRIAL rules; they
    // move a lane's own search / effort value inside its declared range
    // through the trial authority and emit no action-space value.
    let mut optimizer_set: Vec<ny_core::router::RuleId> = ny_core::router::ostop::R3_RULES.to_vec();
    optimizer_set.extend(ny_core::router::otrial::R6_RULES);
    assert_eq!(
        ny_core::router::ACTING_OPTIMIZER_RULES,
        optimizer_set.as_slice()
    );

    // Source check: nothing in the router writes a preset cell or a gate.
    let needles = [
        format!("{}_{}", "preset", "gates"),
        format!("{}::{}", "ny_levers", "gate"),
        format!("{}{}", "BetaCrown", "Config"),
        format!("{}_{}", "apply", "preset"),
    ];
    let mut scanned = 0usize;
    for rel in [
        "crates/ny-cli/src/router/mod.rs",
        "crates/ny-cli/src/router/registry.rs",
        "crates/ny-cli/src/router/facts.rs",
        "crates/ny-cli/src/router/telemetry.rs",
        "crates/ny-core/src/router/mod.rs",
        "crates/ny-core/src/router/plan.rs",
        "crates/ny-core/src/router/checkpoint.rs",
        "crates/ny-core/src/router/bus.rs",
    ] {
        let code: String = read(rel)
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for needle in &needles {
            assert!(
                !code.contains(needle.as_str()),
                "{rel} names `{needle}`: the R1 router writes no cell and applies no value"
            );
        }
        scanned += 1;
    }
    assert_eq!(scanned, 8);
    // Positive control: the bridge the router must not call does exist.
    assert!(read("crates/ny-cli/src/commands/vnncomp.rs").contains(needles[0].as_str()));
    // #router-r2: the acting glue writes rule-owned verifier values at one
    // boundary, and still never a preset cell, a gate, or the preset applier.
    let acting_code: String = read("crates/ny-cli/src/router/acting.rs")
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for needle in [&needles[0], &needles[1], &needles[3]] {
        assert!(
            !acting_code.contains(needle.as_str()),
            "router/acting.rs names `{needle}`: rule values apply at the verifier's \
             configuration boundary only"
        );
    }
    assert!(
        acting_code.contains(needles[2].as_str()),
        "positive control: the one boundary names the verifier configuration"
    );
    for rel in [
        "crates/ny-core/src/router/acting.rs",
        "crates/ny-core/src/router/derive.rs",
        // #router-r3: the O-STOP policy and its ny-cli glue write no cell and
        // apply no value either (the stop authority is their only output).
        "crates/ny-core/src/router/ostop.rs",
        "crates/ny-cli/src/router/ostop.rs",
        // #router-r4: acting set 3's policy and glue, the lease and FAR
        // authorities and the safe-point rulings write no cell of a lane's
        // configuration and apply no value either.
        "crates/ny-core/src/router/r4.rs",
        "crates/ny-core/src/router/lease.rs",
        "crates/ny-core/src/router/safepoint.rs",
        "crates/ny-core/src/memory_lease.rs",
        "crates/ny-core/src/far_authority.rs",
        "crates/ny-cli/src/router/lease.rs",
        "crates/ny-cli/src/router/far.rs",
        // #router-r5: acting set 4's pure policy, CR-G's arithmetic and the
        // route authority write no cell and apply no value.
        "crates/ny-core/src/router/route.rs",
        "crates/ny-core/src/router/gap_contract.rs",
        "crates/ny-core/src/route_authority.rs",
    ] {
        let code: String = read(rel)
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for needle in &needles {
            assert!(!code.contains(needle.as_str()), "{rel} names `{needle}`");
        }
    }
    // #router-r5: the route glue CONFIGURES a routed capability at the
    // verifier's configuration boundary (like CR-K's glue) and on the
    // margin-row lane's frozen policy, and still never writes a typed cell,
    // a gate, or calls the category-file applier.
    let route_code: String = read("crates/ny-cli/src/router/route.rs")
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for needle in [&needles[0], &needles[1], &needles[3]] {
        assert!(
            !route_code.contains(needle.as_str()),
            "router/route.rs names `{needle}`"
        );
    }
    assert!(
        route_code.contains(needles[2].as_str()),
        "positive control: the route glue names the verifier configuration"
    );
}

#[test]
fn router_admission_debt_matches_manifest() {
    let manifest: BTreeSet<(String, String, String)> = include_str!("router_admission_debt.tsv")
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(fields.len(), 3, "malformed debt row {line:?}");
            (
                fields[0].to_owned(),
                fields[1].to_owned(),
                fields[2].to_owned(),
            )
        })
        .collect();
    let derived: BTreeSet<(String, String, String)> = registry::admission_debt()
        .into_iter()
        .map(|(id, kind, site)| (id.to_owned(), kind.to_owned(), site.to_owned()))
        .collect();
    let rendered = derived
        .iter()
        .map(|(id, kind, site)| format!("{id}\t{kind}\t{site}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        derived, manifest,
        "THE ROUTER ADMISSION DEBT CHANGED. It may only fall: remove a row when an admission \
         becomes a named predicate the lane and the registry both call; a new row is a \
         capability that shipped without one. The registry now derives:\n{rendered}"
    );
    assert!(
        !derived.is_empty(),
        "positive control: R1 has admission debt"
    );
}

#[test]
fn router_admissions_come_from_the_lanes_own_predicates() {
    use ny_core::router::decl::Admission;
    let facts = ny_core::router::facts::InstanceFacts::known_empty();
    let single = registry::Context {
        facts: &facts,
        relational_selected: Some(false),
        levers: None,
    };
    let dual = registry::Context {
        facts: &facts,
        relational_selected: Some(true),
        levers: None,
    };
    let relational = registry::find("relational").expect("registered");
    assert_eq!(registry::admit(relational, &dual), Admission::Admit);
    assert!(matches!(
        registry::admit(relational, &single),
        Admission::Decline(_)
    ));
    // Inline admissions are refusals, never guesses; tiers follow their parent.
    let margin_row = registry::find("margin-row").expect("registered");
    assert_eq!(
        registry::admit(margin_row, &single),
        Admission::Unknown("inline-admission")
    );
    let frozen_dual = registry::find("frozen-dual-bab").expect("registered");
    assert_eq!(
        registry::admit(frozen_dual, &single),
        Admission::Unknown("parent-unknown")
    );
    let cz = registry::find("cz_hybrid").expect("registered");
    assert!(matches!(
        registry::admit(cz, &single),
        Admission::Decline(_)
    ));
    let ibp = registry::find("ibp-fast").expect("registered");
    assert_eq!(registry::admit(ibp, &single), Admission::Admit);
}

// ============================================================================
// #router-r2 — the acting glue, rule by rule, against a LOCAL authority (the
// process-global one is never armed here: other in-process tests' lanes read
// it).
// ============================================================================

mod r2 {
    use std::time::{Duration, Instant};

    use ny_core::deadline_authority::{Authority, Frame, LaneSlot, SLOTS};
    use ny_core::router::acting::{DerivedPlan, RuleSwitches};

    use super::super::acting::{self, VerifierPins, COLLECT_SLACK};
    use super::super::RouterPolicy;
    use crate::commands::lane_allocation::{AttackSliceAllocation, LaneWindow, TodaySlices};

    fn frozen_bool(config: &[(&'static ny_levers::LeverDecl, bool)]) -> ny_levers::LeverSet {
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
        ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            config
                .iter()
                .find(|(candidate, _)| std::ptr::eq(*candidate, decl))
                .map(|(_, value)| ny_levers::LeverValue::Bool(*value))
        })
        .expect("admissible rule configuration")
    }

    fn armed(grace: Option<Duration>, governed: [bool; SLOTS]) -> (Authority, Instant) {
        let authority = Authority::new();
        let anchor = Instant::now();
        let frame = Frame::new(
            anchor,
            anchor + Duration::from_secs(100),
            [
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(3),
                Duration::from_secs(3),
            ],
            Duration::from_secs(2),
        );
        authority
            .arm(frame, governed, Some(LaneSlot::Verifier), grace)
            .expect("arm");
        (authority, anchor)
    }

    #[test]
    fn router_r2_policy_resolves_rule_switches_from_the_frozen_levers() {
        use ny_levers::decls::router::{ROUTER_RULE_DERIVED_CAPS, ROUTER_RULE_PUBLISH_GRACE};
        let dark = frozen_bool(&[]);
        let policy = RouterPolicy::from_levers(&dark).expect("dark");
        assert_eq!(policy.rules, RuleSwitches::ALL_ON, "every rule defaults ON");
        assert_eq!(
            policy,
            RouterPolicy::OFF,
            "and the router itself defaults off"
        );
        let ablated = frozen_bool(&[
            (&ROUTER_RULE_PUBLISH_GRACE, false),
            (&ROUTER_RULE_DERIVED_CAPS, false),
        ]);
        let policy = RouterPolicy::from_levers(&ablated).expect("typed");
        assert!(!policy.rules.publish_grace && !policy.rules.derived_caps);
        assert!(policy.rules.cap_authority && policy.rules.hand_on);
        assert_eq!(
            ablated
                .resolved(&ROUTER_RULE_PUBLISH_GRACE)
                .expect("declared")
                .source,
            ny_levers::Source::Config,
            "the switch arrives from the typed layer"
        );
    }

    #[test]
    fn router_r2_helpers_are_identity_when_the_authority_is_unarmed() {
        let authority = Authority::new();
        let now = Instant::now();
        for slot in LaneSlot::ALL {
            for today in [
                LaneWindow::Private,
                LaneWindow::Skip,
                LaneWindow::Cap(Duration::from_millis(7_321)),
            ] {
                assert_eq!(
                    acting::claim_attack_window_with(&authority, slot, today),
                    today
                );
            }
            assert_eq!(acting::release_lane_with(&authority, slot, false), None);
            assert_eq!(
                acting::claim_margin_row_with(&authority, slot, now),
                now,
                "{slot:?}"
            );
            assert_eq!(
                acting::collector_grace_with(&authority, slot),
                Duration::ZERO
            );
        }
        assert_eq!(acting::claim_terminal_with(&authority, 93), 93);
        assert_eq!(
            acting::final_join_margin_with(&authority, Duration::from_secs(2)),
            Duration::from_secs(2)
        );
        assert!(!acting::governs_attack_slice_with(&authority, true));
        // The scored-path wrappers read the process-global authority, which no
        // unit test arms: identity too.
        assert_eq!(acting::claim_terminal(41), 41);
        assert_eq!(
            acting::claim_attack_window(LaneSlot::AttackUpfront, LaneWindow::Private),
            LaneWindow::Private
        );
        assert_eq!(
            acting::final_join_margin(Duration::from_secs(2)),
            Duration::from_secs(2)
        );
        let mut config = ny_propagate::BetaCrownConfig::default();
        let before = format!("{config:?}");
        acting::apply_verifier_directives(&mut config, VerifierPins::default());
        assert_eq!(
            format!("{config:?}"),
            before,
            "no session: no derived value"
        );
    }

    #[test]
    fn router_r2_attack_contracts_flow_from_the_frame_into_the_cells() {
        let (authority, _) = armed(None, [true; SLOTS]);
        let frame = AttackSliceAllocation::frame(TodaySlices {
            sign_space: Some(Duration::from_secs(20)),
            ste_pgd: None,
            upfront: Some(Duration::from_secs(8)),
        })
        .expect("a frame");
        assert!(acting::governs_attack_slice_with(&authority, true));
        assert!(
            !acting::governs_attack_slice_with(&authority, false),
            "CR-C off: the router does not admit the frame"
        );
        acting::commit_attack_contracts_with(&authority, true, Some(&frame));
        assert_eq!(
            authority.contract(LaneSlot::AttackSignSpace),
            Some(Duration::from_secs(20))
        );
        assert_eq!(
            authority.contract(LaneSlot::AttackStePgd),
            Some(Duration::ZERO)
        );
        assert_eq!(
            authority.contract(LaneSlot::AttackUpfront),
            Some(Duration::from_secs(8))
        );
        // Each lane is handed its cell's window at its own start.
        assert_eq!(
            acting::claim_attack_window_with(
                &authority,
                LaneSlot::AttackSignSpace,
                LaneWindow::Cap(Duration::from_secs(20))
            ),
            LaneWindow::Cap(Duration::from_secs(20))
        );
        assert_eq!(
            acting::claim_attack_window_with(&authority, LaneSlot::AttackStePgd, LaneWindow::Skip),
            LaneWindow::Skip,
            "a zero contract is a SKIP, never a small cap"
        );
        // It declines at once: its unspent contract goes to the terminal.
        let hand_on = acting::release_lane_with(&authority, LaneSlot::AttackSignSpace, false)
            .expect("unspent contract");
        assert_eq!(hand_on.to, Some(LaneSlot::Verifier));
        assert!(hand_on.credited_ns > 19_000_000_000, "{hand_on:?}");
        // The terminal keeps exactly its own timeout and finds the credit in it.
        assert_eq!(acting::claim_terminal_with(&authority, 60), 60);
        let snapshot = authority.snapshot();
        assert_eq!(snapshot.unrealized, 0, "{snapshot:?}");
        assert_eq!(snapshot.hand_ons.len(), 1);
        // A second release is not a second hand-on.
        assert_eq!(
            acting::release_lane_with(&authority, LaneSlot::AttackSignSpace, false),
            None
        );
        // Without the frame (another cap authority, or none), nothing commits.
        let (fresh, _) = armed(None, [true; SLOTS]);
        acting::commit_attack_contracts_with(&fresh, true, None);
        assert_eq!(fresh.contract(LaneSlot::AttackUpfront), None);
        assert_eq!(
            acting::claim_attack_window_with(&fresh, LaneSlot::AttackUpfront, LaneWindow::Private),
            LaneWindow::Private,
            "no contract: the lane keeps its private live-clock window"
        );
    }

    #[test]
    fn router_r2_publish_grace_extends_every_collector_exactly() {
        let grace = Duration::from_millis(1_500);
        let (authority, anchor) = armed(Some(grace), [true; SLOTS]);
        assert_eq!(
            acting::collector_grace_with(&authority, LaneSlot::MarginRowInline),
            grace + COLLECT_SLACK
        );
        // The join waits until the concurrent lane's own publish deadline
        // (hard - 3 s + g) plus the slack, never less than the write reserve.
        let joined = acting::final_join_margin_with(&authority, Duration::from_secs(2));
        assert_eq!(joined, Duration::from_millis(3_000 - 1_500 - 250));
        assert!(joined >= Duration::from_millis(500));
        assert!(
            joined < Duration::from_secs(2),
            "the join waits LONGER than today"
        );
        // The lane itself keeps exactly its own search deadline.
        let today = anchor + Duration::from_secs(97);
        assert_eq!(
            acting::claim_margin_row_with(&authority, LaneSlot::MarginRowInline, today),
            today
        );
        // An ungoverned margin-row lane is collected exactly as today.
        let mut governed = [true; SLOTS];
        governed[LaneSlot::MarginRowConcurrent.index()] = false;
        let (partial, _) = armed(Some(grace), governed);
        assert_eq!(
            acting::final_join_margin_with(&partial, Duration::from_secs(2)),
            Duration::from_secs(2)
        );
        // No grace armed: every collector waits exactly as today.
        let (no_grace, _) = armed(None, [true; SLOTS]);
        assert_eq!(
            acting::collector_grace_with(&no_grace, LaneSlot::MarginRowInline),
            Duration::ZERO
        );
    }

    /// review(router-r2): the concurrent lane's final join may only wait
    /// LONGER than today (a smaller margin below the scored deadline), never
    /// shorter, whatever grace is armed and whatever margin today uses; it
    /// always keeps the write reserve that today keeps; and whenever it moves,
    /// it waits at least until the lane's own publish deadline plus the
    /// collect slack (or down to the write reserve). The 1.5 s case above
    /// never reaches the `min(today)` clamp, so without this sweep a join that
    /// waits LESS than today would pass every test.
    #[test]
    fn router_r2_final_join_never_waits_less_than_today() {
        let tail_ms = ny_core::router::derive::MARGIN_ROW_TAIL_MS;
        let floor = Duration::from_millis(ny_core::router::derive::WRITE_RESERVE_MS);
        let mut clamped_to_today = 0usize;
        for grace_ms in [
            1u64, 50, 100, 250, 500, 749, 750, 751, 1_000, 1_250, 1_499, 1_500, 1_749, 1_750,
            1_751, 1_999,
        ] {
            let grace = Duration::from_millis(grace_ms);
            let (authority, _) = armed(Some(grace), [true; SLOTS]);
            let covers = Duration::from_millis(tail_ms)
                .saturating_sub(grace + COLLECT_SLACK)
                .max(floor);
            for today_ms in [0u64, 250, 500, 1_000, 1_250, 2_000, 3_000, 5_000] {
                let today = Duration::from_millis(today_ms);
                let joined = acting::final_join_margin_with(&authority, today);
                assert!(
                    joined <= today,
                    "grace={grace_ms}ms today={today_ms}ms: the join waits LESS than today \
                     (margin {joined:?} > {today:?})"
                );
                assert!(
                    joined >= floor.min(today),
                    "grace={grace_ms}ms today={today_ms}ms: the write reserve is gone ({joined:?})"
                );
                assert!(
                    joined <= covers,
                    "grace={grace_ms}ms today={today_ms}ms: the join stops before the lane's \
                     publish deadline plus the slack ({joined:?} > {covers:?})"
                );
                if joined == today && covers > today {
                    clamped_to_today += 1;
                }
            }
        }
        assert!(
            clamped_to_today > 0,
            "positive control: some case must reach the never-less-than-today clamp"
        );
    }

    #[test]
    fn router_r2_derived_caps_respect_pins_and_the_lease() {
        let lease = 105u64 << 30;
        let derived = DerivedPlan {
            clause_passes: Some(593),
            max_depth: Some(593_000),
            lease_bytes: lease,
            warmup_hand_on: true,
        };
        // Unpinned, no resident-queue byte budget: the warmup hands on, the
        // passes derive, the depth is NOT lifted (it is still doing memory work).
        let mut config = ny_propagate::BetaCrownConfig::default();
        let today_depth = config.max_depth;
        let applied = acting::apply_derived(&mut config, VerifierPins::default(), derived);
        assert!(config.initial_bounds_warmup_fallback);
        assert_eq!(config.disjunctive_clause_retry_passes, 593);
        assert_eq!(config.max_depth, today_depth);
        let depth = applied
            .iter()
            .find(|d| d.key == "bab.max_depth")
            .expect("depth recorded");
        assert_eq!(depth.blocked_by.as_deref(), Some("no_queue_byte_budget"));
        // A byte budget inside the lease admits the lift.
        let mut config = ny_propagate::BetaCrownConfig {
            max_queue_bytes: 2 << 30,
            ..ny_propagate::BetaCrownConfig::default()
        };
        acting::apply_derived(&mut config, VerifierPins::default(), derived);
        assert_eq!(config.max_depth, 593_000);
        // A budget past half the lease does not.
        let mut config = ny_propagate::BetaCrownConfig {
            max_queue_bytes: usize::try_from(lease).expect("fits"),
            ..ny_propagate::BetaCrownConfig::default()
        };
        acting::apply_derived(&mut config, VerifierPins::default(), derived);
        assert_eq!(config.max_depth, today_depth);
        // Pins win, every key.
        let pins = VerifierPins {
            max_depth: true,
            clause_passes: true,
            warmup_fallback: true,
        };
        let mut config = ny_propagate::BetaCrownConfig {
            max_queue_bytes: 2 << 30,
            ..ny_propagate::BetaCrownConfig::default()
        };
        let before = format!("{config:?}");
        let applied = acting::apply_derived(&mut config, pins, derived);
        assert_eq!(format!("{config:?}"), before, "a pinned row is untouched");
        assert_eq!(applied.len(), 3);
        assert!(applied
            .iter()
            .all(|d| d.blocked_by.as_deref() == Some("pin") && d.from == d.to));
        // Never below the value already in force.
        let mut config = ny_propagate::BetaCrownConfig {
            disjunctive_clause_retry_passes: 5_000,
            max_queue_bytes: 2 << 30,
            max_depth: 9_000_000,
            ..ny_propagate::BetaCrownConfig::default()
        };
        acting::apply_derived(&mut config, VerifierPins::default(), derived);
        assert_eq!(config.disjunctive_clause_retry_passes, 5_000);
        assert_eq!(config.max_depth, 9_000_000);
        // CR-K / CR-H off: nothing derived.
        let mut config = ny_propagate::BetaCrownConfig::default();
        let before = format!("{config:?}");
        let none = acting::apply_derived(
            &mut config,
            VerifierPins::default(),
            DerivedPlan {
                lease_bytes: lease,
                ..DerivedPlan::default()
            },
        );
        assert_eq!((none).len(), 0);
        assert_eq!(format!("{config:?}"), before);
    }
}

/// #router-r3 (ROUTER_PROGRAM §10 R3): the ny-cli glue of acting set 2. Every
/// test drives LOCAL authorities (the `*_with` cores); none arms a
/// process-global one another in-process test's lane could read. The TLS
/// upfront scope is per-thread, so each test owns its own.
mod r3 {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    use ny_core::deadline_authority::{Authority, Frame, LaneSlot, SLOTS};
    use ny_core::router::ostop::{OptimizerSwitches, ATTACK_DEFER_RULE};
    use ny_core::stop_authority::{StopAuthority, StopDirectives};

    use super::super::ostop;
    use super::super::RouterPolicy;

    fn frozen_bool(config: &[(&'static ny_levers::LeverDecl, bool)]) -> ny_levers::LeverSet {
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
        ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            config
                .iter()
                .find(|(candidate, _)| std::ptr::eq(*candidate, decl))
                .map(|(_, value)| ny_levers::LeverValue::Bool(*value))
        })
        .expect("admissible rule configuration")
    }

    fn cells(upfront_governed: bool) -> Authority {
        let authority = Authority::new();
        let anchor = Instant::now();
        let mut governed = [true; SLOTS];
        governed[LaneSlot::AttackUpfront.index()] = upfront_governed;
        authority
            .arm(
                Frame::new(
                    anchor,
                    anchor + Duration::from_secs(100),
                    [Duration::from_secs(5); SLOTS],
                    Duration::from_secs(2),
                ),
                governed,
                Some(LaneSlot::Verifier),
                None,
            )
            .expect("arm");
        authority
    }

    fn defer_armed() -> StopAuthority {
        let stop = StopAuthority::new();
        stop.arm(StopDirectives {
            attack_defer: Some(ATTACK_DEFER_RULE),
            ..StopDirectives::NONE
        })
        .expect("arm");
        stop
    }

    #[test]
    fn router_r3_policy_resolves_optimizer_switches_from_the_frozen_levers() {
        use ny_levers::decls::router::{OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR};
        let dark = frozen_bool(&[]);
        let policy = RouterPolicy::from_levers(&dark).expect("dark");
        assert_eq!(
            policy.optimizer_rules,
            OptimizerSwitches::ALL_ON,
            "every O-STOP rule defaults ON"
        );
        assert_eq!(
            policy,
            RouterPolicy::OFF,
            "and the optimizer itself defaults off"
        );
        let ablated = frozen_bool(&[
            (&OPTIMIZER_RULE_GOVERNOR, false),
            (&OPTIMIZER_RULE_ATTACK_DEFER, false),
        ]);
        let policy = RouterPolicy::from_levers(&ablated).expect("typed");
        assert!(!policy.optimizer_rules.governor && !policy.optimizer_rules.attack_defer);
        assert!(policy.optimizer_rules.ia_yield && policy.optimizer_rules.root_admission);
        assert_eq!(
            ablated
                .resolved(&OPTIMIZER_RULE_GOVERNOR)
                .expect("declared")
                .source,
            ny_levers::Source::Config,
            "the switch arrives from the typed layer"
        );
    }

    /// Off / shadow bit-identity of the glue: with no armed rule every helper
    /// is `today` in, `today` out, prints nothing, and the CR-ATK admission
    /// predicate (a live memory probe on the scored path) is never evaluated.
    #[test]
    fn router_r3_helpers_are_identity_when_the_stop_authority_is_unarmed() {
        let stop = StopAuthority::new();
        let cells = cells(true);
        let evaluated = Cell::new(false);
        assert!(ostop::enter_upfront_scope_with(&stop, &cells, || {
            evaluated.set(true);
            true
        })
        .is_none());
        assert!(!evaluated.get(), "the predicate ran on an unarmed row");
        assert!(!ostop::upfront_scope_armed());
        assert!(!ostop::upfront_restart_boundary_at(
            &stop,
            Instant::now(),
            3,
            Some(60),
            -0.033,
            1e-4
        ));
        assert_eq!(ostop::deferred_reserve_secs_with(&stop, true), 0);
        ostop::note_defer_held_with(&stop, 14, 14, 50);
        ostop::note_late_attack_start_with(&stop, Duration::from_secs(20));
        let snap = stop.snapshot();
        assert!(snap.defers.is_empty() && snap.defer_outcomes.is_empty());
        // The process-global helpers agree on an unarmed process.
        assert!(ostop::enter_upfront_scope(|| panic!("never evaluated")).is_none());
        assert_eq!(ostop::deferred_reserve_secs(true), 0);
    }

    /// CR-ATK refuses to defer where the late reserve cannot be honoured, or
    /// where the router does not govern the upfront slot: the lane runs on.
    #[test]
    fn router_r3_attack_defer_needs_an_admissible_late_reserve() {
        let stop = defer_armed();
        assert!(ostop::enter_upfront_scope_with(&stop, &cells(true), || false).is_none());
        assert!(ostop::enter_upfront_scope_with(&stop, &cells(false), || true).is_none());
        assert!(!ostop::upfront_scope_armed());
        let scope = ostop::enter_upfront_scope_with(&stop, &cells(true), || true);
        assert!(scope.is_some(), "positive control");
        assert!(ostop::upfront_scope_armed());
        drop(scope);
        assert!(
            !ostop::upfront_scope_armed(),
            "the scope is cleared on drop"
        );
    }

    /// Synthetic progress stream through the glue: a climbing attack keeps its
    /// slice; a plateau defers at a restart boundary only once a full half of
    /// the slice is spanned AND >= 5 s is left; a plateau with too little
    /// left runs on (a defer would be a cut); the defer is sticky.
    #[test]
    fn router_r3_attack_defer_glue_defers_only_on_a_flat_slope() {
        let stop = defer_armed();
        let scope = ostop::enter_upfront_scope_with(&stop, &cells(true), || true).expect("scope");
        let start = Instant::now();
        let deadline = start + Duration::from_secs(30);
        ostop::upfront_slice(start, deadline);
        let at = |secs: u64| start + Duration::from_secs(secs);
        // Climbing: never defers.
        for (i, secs) in (1..=20u64).enumerate() {
            let best = -2.0 + 0.09 * secs as f64;
            assert!(
                !ostop::upfront_restart_boundary_at(&stop, at(secs), i, None, best, 1e-4),
                "a climbing attack keeps its slice (t={secs})"
            );
        }
        assert!(scope.deferred().is_none());
        drop(scope);

        // Plateau from t=1: flat once the window [t-15, t] is spanned.
        let stop = defer_armed();
        let scope = ostop::enter_upfront_scope_with(&stop, &cells(true), || true).expect("scope");
        ostop::upfront_slice(start, deadline);
        let mut deferred_at = None;
        for (i, secs) in (1..=24u64).enumerate() {
            if ostop::upfront_restart_boundary_at(&stop, at(secs), i, None, -0.033, 1e-4) {
                deferred_at = Some(secs);
                break;
            }
        }
        assert_eq!(deferred_at, Some(16), "flat over [1, 16] = half of 30 s");
        let event = scope.deferred().expect("deferred");
        assert_eq!(event.unspent_ms, 14_000);
        assert_eq!(event.slice_ms, 30_000);
        assert!(event.rise <= event.eps);
        assert!(
            ostop::upfront_restart_boundary_at(&stop, at(17), 99, None, 0.5, 1e-4),
            "sticky once deferred"
        );
        assert_eq!(stop.snapshot().defers.len(), 1);
        drop(scope);

        // Climbs until t=8, then plateaus: the half-slice window (10 s of a
        // 20 s slice) is first flat at t=18, with only 2 s left — under the
        // 5 s floor a defer would be a cut, so the lane runs on to today's cap.
        let stop = defer_armed();
        let scope = ostop::enter_upfront_scope_with(&stop, &cells(true), || true).expect("scope");
        let short_deadline = start + Duration::from_secs(20);
        ostop::upfront_slice(start, short_deadline);
        for (i, secs) in (1..=19u64).enumerate() {
            let best = if secs <= 8 {
                -2.0 + 0.2 * secs as f64
            } else {
                -0.4
            };
            assert!(
                !ostop::upfront_restart_boundary_at(&stop, at(secs), i, None, best, 1e-4),
                "t={secs}: climbing, or flat with under 5 s left"
            );
        }
        assert!(scope.deferred().is_none());
    }

    /// #router-r3 review: the upfront lane keeps a restart cap
    /// (`GRAD_REFINE_MAX_RESTARTS`), so it can end BEFORE its deadline today.
    /// A defer holds out of the verifier only what the lane would still have
    /// spent — `min(deadline - now, restarts_left x shortest restart)` — and a
    /// projection under the 5 s floor is no defer at all. Same plateau as the
    /// glue test above (1 s restarts, flat over [1, 16] of a 30 s slice).
    #[test]
    fn router_r3_review_defer_holds_no_more_than_the_restart_cap_would_spend() {
        let start = Instant::now();
        let deadline = start + Duration::from_secs(30);
        let run = |cap: usize| {
            let stop = defer_armed();
            let scope =
                ostop::enter_upfront_scope_with(&stop, &cells(true), || true).expect("scope");
            ostop::upfront_slice(start, deadline);
            let mut deferred = None;
            for (i, secs) in (1..=24u64).enumerate() {
                let left = Some(cap.saturating_sub(i + 1));
                let now = start + Duration::from_secs(secs);
                if ostop::upfront_restart_boundary_at(&stop, now, i, left, -0.033, 1e-4) {
                    deferred = scope.deferred();
                    break;
                }
            }
            let booked = stop.snapshot().defers.len();
            drop(scope);
            (deferred, booked)
        };
        // A cap far past the deadline: the deadline binds, exactly as uncapped.
        let (far, booked) = run(64);
        let far = far.expect("a plateau with 48 restarts left defers");
        assert_eq!((far.elapsed_ms, far.unspent_ms), (16_000, 14_000));
        assert_eq!(booked, 1);
        // Today's lane would stop after restart 23 (t = 24 s): it would have
        // spent 8 more seconds, so only 8 s is deferred, not the 14 s to the
        // deadline.
        let (capped, _) = run(24);
        let capped = capped.expect("8 s projected clears the 5 s floor");
        assert_eq!(capped.elapsed_ms, 16_000);
        assert_eq!(
            capped.unspent_ms, 8_000,
            "the hold never exceeds what the capped lane would still spend"
        );
        // The cap ends the lane 4 s after the window is first flat: under the
        // floor at every boundary, so the lane runs on to its own cap.
        let (none, booked) = run(20);
        assert!(none.is_none(), "{none:?}");
        assert_eq!(booked, 0, "nothing is booked when nothing is deferred");
    }

    /// #router-r3 review: the upfront scope is CLOSED as soon as the lane
    /// returns, before the trusted-oracle gate — whose ORT repair runs an
    /// attack restart loop on this same thread and must not read as the
    /// upfront lane (it could be deferred, losing the sat it exists to
    /// recover, and book a defer the lane never made).
    #[test]
    fn router_r3_review_upfront_scope_closes_before_the_trusted_oracle_gate() {
        // Behaviour: `close` reports the lane's own defer and clears the scope.
        let stop = defer_armed();
        let scope = ostop::enter_upfront_scope_with(&stop, &cells(true), || true);
        assert!(
            scope.is_some() && ostop::upfront_scope_armed(),
            "positive control"
        );
        assert!(!ostop::UpfrontDeferScope::close(scope), "no defer was made");
        assert!(
            !ostop::upfront_scope_armed(),
            "closed: a later attack loop on this thread is not the upfront lane"
        );
        assert!(!ostop::UpfrontDeferScope::close(None), "unarmed rows");
        let stop = defer_armed();
        let scope = ostop::enter_upfront_scope_with(&stop, &cells(true), || true).expect("scope");
        let start = Instant::now();
        ostop::upfront_slice(start, start + Duration::from_secs(30));
        for (i, secs) in (1..=16u64).enumerate() {
            let now = start + Duration::from_secs(secs);
            let _ = ostop::upfront_restart_boundary_at(&stop, now, i, None, -0.033, 1e-4);
        }
        assert!(
            ostop::UpfrontDeferScope::close(Some(scope)),
            "the lane deferred"
        );
        assert!(!ostop::upfront_scope_armed());
        assert!(
            !ostop::upfront_restart_boundary_at(&stop, Instant::now(), 17, None, -0.033, 1e-4),
            "after the close no boundary can defer (nor stay sticky)"
        );
        // Wiring on the scored entry point: opened once, closed once, after the
        // lane and BEFORE the gate, and never read again.
        let code = super::read("crates/ny-cli/src/commands/vnncomp.rs");
        let open = code
            .find("let upfront_defer_scope = crate::router::ostop::enter_upfront_scope(")
            .expect("the scope is opened on the scored path");
        let after = |needle: &str| {
            code[open..]
                .find(needle)
                .map(|at| open + at)
                .unwrap_or_else(|| panic!("`{needle}` not found after the scope opens"))
        };
        let lane = after("try_upfront_falsify(");
        let close = after("UpfrontDeferScope::close(upfront_defer_scope)");
        let gate = after("gate_sat_with_trusted_oracle(");
        assert!(
            lane < close && close < gate,
            "the scope must close after the upfront lane and before the trusted-oracle gate \
             (lane={lane} close={close} gate={gate})"
        );
        assert_eq!(
            code.matches("upfront_defer_scope").count(),
            2,
            "the scope is opened once and closed once, and nothing reads it after the close"
        );
    }

    /// The deferred slice is held out of the verifier and realized by the late
    /// attack; an unavailable late consumer and a decided row close it with
    /// the right outcome. Conservation: nothing held exceeds what was deferred.
    #[test]
    fn router_r3_deferred_slice_is_held_and_realized_by_the_late_attack() {
        use ny_core::stop_authority::DeferEvent;
        let event = DeferEvent {
            restart: 9,
            slice_ms: 30_000,
            elapsed_ms: 16_000,
            unspent_ms: 14_400,
            best_margin: -0.033,
            rise: 0.0,
            eps: 1e-4,
            window_ms: 15_000,
        };
        // Realized.
        let stop = defer_armed();
        stop.record_defer(event.clone());
        let secs = ostop::deferred_reserve_secs_with(&stop, true);
        assert_eq!(
            secs, 14,
            "whole seconds round DOWN: the verifier never loses more"
        );
        ostop::note_defer_held_with(&stop, secs, secs, 50);
        let (deferred, held) = stop.held_reserve().expect("held");
        assert_eq!(deferred, Duration::from_millis(14_400));
        assert_eq!(held, Duration::from_secs(14));
        assert!(held <= deferred);
        ostop::note_late_attack_start_with(&stop, Duration::from_millis(15_100));
        let snap = stop.snapshot();
        let outcome = snap.defer_outcomes.last().expect("outcome");
        assert!(outcome.realized && !outcome.moot, "{outcome:?}");
        assert_eq!(outcome.late_budget_ms, Some(15_100));
        // Unavailable at hold: recorded unrealized, nothing held.
        let stop = defer_armed();
        stop.record_defer(event.clone());
        assert_eq!(ostop::deferred_reserve_secs_with(&stop, false), 0);
        assert_eq!(stop.held_reserve(), None);
        let snap = stop.snapshot();
        assert_eq!(
            snap.defer_outcomes.last().map(|o| o.reason.as_str()),
            Some("late_consumer_unavailable_at_hold")
        );
        // A decided row makes a held slice MOOT; an undecided one UNREALIZED.
        for (verdict, moot) in [("unsat", true), ("sat", true), ("timeout", false)] {
            let stop = defer_armed();
            stop.record_defer(event.clone());
            let secs = ostop::deferred_reserve_secs_with(&stop, true);
            ostop::note_defer_held_with(&stop, secs, secs, 50);
            let record = ostop::finish_with(
                &stop,
                ostop::StopRecord {
                    armed: true,
                    refused: None,
                    snapshot: None,
                },
                verdict,
            );
            let snap = record.snapshot.clone().expect("snapshot");
            let outcome = snap.defer_outcomes.last().expect("closed");
            assert_eq!(outcome.moot, moot, "{verdict}");
            assert!(!outcome.realized, "{verdict}");
            assert!(!stop.is_armed(), "finish disarms");
            let line = ostop::done_line(&record);
            assert!(line.starts_with("[optimizer] done armed=1 "), "{line}");
            assert!(line.contains(" defers=1 "), "{line}");
        }
    }

    /// CR-ATK's release hands NOTHING to the terminal: the deferred slice is
    /// ledgered `deferred`, and the terminal's claim finds no credit.
    #[test]
    fn router_r3_deferred_release_credits_nothing_to_the_terminal() {
        use super::super::acting;
        let cells = cells(true);
        let anchor = Instant::now();
        cells
            .commit_contract(LaneSlot::AttackUpfront, Duration::from_secs(20), anchor)
            .expect("contract");
        cells
            .claim_contract(LaneSlot::AttackUpfront, anchor)
            .expect("claim");
        let hand_on =
            acting::release_lane_to_reserve_with(&cells, LaneSlot::AttackUpfront).expect("unspent");
        assert!(hand_on.deferred && hand_on.credited_ns == 0 && hand_on.to.is_none());
        let claim = cells
            .claim_deadline(
                LaneSlot::Verifier,
                Instant::now() + Duration::from_secs(60),
                Instant::now(),
            )
            .expect("terminal");
        assert_eq!(claim.credit, Duration::ZERO);
        // An ungoverned slot: nothing.
        let ungoverned = super::r3::cells(false);
        assert!(
            acting::release_lane_to_reserve_with(&ungoverned, LaneSlot::AttackUpfront).is_none()
        );
    }

    #[test]
    fn router_r3_noise_floor_is_the_confirmation_floor() {
        assert!((ostop::noise_floor(0.0) - 1e-4).abs() < 1e-18);
        assert!((ostop::noise_floor(1_000.0) - 1e-2).abs() < 1e-12);
        assert!((ostop::noise_floor(f64::NAN) - 1e-4).abs() < 1e-18);
        // A magnitude: the sign of the caller's value is ignored.
        assert!((ostop::noise_floor(-50.0) - 5e-4).abs() < 1e-15);
    }

    /// The stop authority arms only from acting commitments; an O-STOP set
    /// with no acting rule leaves it idle; a contained fault in the glue
    /// returns today's value.
    #[test]
    fn router_r3_arm_lines_name_every_rule_and_the_directives() {
        use ny_core::router::ostop::{self as policy, R3_RULES};
        let stop = StopAuthority::new();
        let input = super::super::policy_input(&super::printer_ctx());
        let plan = ny_core::router::plan::plan(&input);
        let r2 = ny_core::router::acting::commit(
            &plan,
            &input,
            ny_core::router::EffectiveMode::Acting,
            &ny_core::router::acting::RuleSwitches::ALL_ON,
        );
        let c = policy::commit(
            &plan,
            &input,
            ny_core::router::EffectiveMode::Acting,
            &OptimizerSwitches::ALL_ON,
            Some(&r2),
        );
        let lines = ostop::arm_with(&stop, &c);
        for rule in R3_RULES {
            assert!(
                lines.iter().any(
                    |l| l.starts_with(&format!("[optimizer] act rule={} acts=", rule.as_str()))
                ),
                "{rule:?}: {lines:?}"
            );
        }
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("[optimizer] authority armed sha=")),
            "{lines:?}"
        );
        assert!(stop.is_armed());
        assert!(stop.governor().is_some(), "rule.governor needs no cell");
        assert!(stop.root_admission());
        // Nothing acting: idle, nothing armed.
        let off = policy::commit(
            &plan,
            &input,
            ny_core::router::EffectiveMode::Shadow,
            &OptimizerSwitches::ALL_ON,
            Some(&r2),
        );
        let idle = StopAuthority::new();
        let lines = ostop::arm_with(&idle, &off);
        assert!(
            lines.iter().any(|l| l.contains("authority idle")),
            "{lines:?}"
        );
        assert!(!idle.is_armed());
    }
}

// ============================================================================
// #router-r4 — acting set 3 (CR-L memory lease, CR-F FAR hand-on) and the
// enforceable safe points, at the ny-cli glue.
// ============================================================================

mod r4 {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    use ny_core::deadline_authority::{Authority, Frame, LaneSlot, SLOTS};
    use ny_core::far_authority::FarAuthority;
    use ny_core::memory_lease::LeaseAuthority;
    use ny_core::router::decl::SafePoints;
    use ny_core::router::facts::{DeviceQual, HostClass, HostEnvelope};
    use ny_core::router::lease::{self as lease_policy, LeaseHolder};
    use ny_core::router::r4::{FarCommit, R4Switches};
    use ny_core::stop_authority::StopSite;

    use super::super::{acting, far, lease, registry, RouterPolicy};
    use super::{printer_ctx, read};
    use crate::commands::lane_allocation::{AttackSliceAllocation, LaneWindow, TodaySlices};

    const GIB: u64 = 1 << 30;

    fn frozen_bool(config: &[(&'static ny_levers::LeverDecl, bool)]) -> ny_levers::LeverSet {
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
        ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            config
                .iter()
                .find(|(candidate, _)| std::ptr::eq(*candidate, decl))
                .map(|(_, value)| ny_levers::LeverValue::Bool(*value))
        })
        .expect("admissible rule configuration")
    }

    fn small_lease() -> LeaseAuthority {
        // 48 GiB host: E = 32 GiB, the terminal holds 16, one heavy lane fits.
        let plan = lease_policy::lease_plan(&HostEnvelope {
            class: HostClass::Other,
            threads: 8,
            mem_bytes: 48 * GIB,
            reserve_bytes: 16 * GIB,
            device: DeviceQual::None,
            declared: false,
        })
        .expect("known envelope");
        let authority = LeaseAuthority::new();
        authority.arm(plan).expect("arm");
        authority
    }

    #[test]
    fn router_r4_policy_resolves_r4_switches_from_the_frozen_levers() {
        use ny_levers::decls::router::{ROUTER_RULE_FAR_HAND_ON, ROUTER_RULE_LEASE};
        let dark = frozen_bool(&[]);
        let policy = RouterPolicy::from_levers(&dark).expect("dark");
        assert_eq!(policy.r4_rules, R4Switches::ALL_ON, "both rules default ON");
        assert_eq!(policy, RouterPolicy::OFF, "and the router defaults off");
        let ablated = frozen_bool(&[(&ROUTER_RULE_LEASE, false)]);
        let policy = RouterPolicy::from_levers(&ablated).expect("typed");
        assert!(!policy.r4_rules.lease && policy.r4_rules.far_hand_on);
        for decl in [&ROUTER_RULE_LEASE, &ROUTER_RULE_FAR_HAND_ON] {
            let resolved = ablated.resolved(decl).expect("declared");
            assert_ne!(
                resolved.source,
                ny_levers::Source::LegacyEnv,
                "{}",
                decl.name
            );
        }
        assert_eq!(
            ablated
                .resolved(&ROUTER_RULE_LEASE)
                .expect("declared")
                .source,
            ny_levers::Source::Config,
            "the switch arrives from the typed layer"
        );
        // The two switches are router selectors, never plan pins.
        let pinned = frozen_bool(&[
            (&ROUTER_RULE_LEASE, false),
            (&ROUTER_RULE_FAR_HAND_ON, false),
        ]);
        assert_eq!((super::super::pins(Some(&pinned))).len(), 0);
    }

    /// Unarmed, every lease helper is EXACTLY today's live gate: the floor
    /// closure is called once and its answer returned, whatever it is.
    #[test]
    fn router_r4_lease_gate_is_todays_gate_when_unarmed() {
        let authority = LeaseAuthority::new();
        for today in [true, false] {
            for holder in LeaseHolder::ALL {
                let calls = Cell::new(0u32);
                let floor = || {
                    calls.set(calls.get() + 1);
                    today
                };
                assert_eq!(lease::admit_with(&authority, holder, floor), today);
                assert_eq!(calls.get(), 1, "admit evaluates today's gate exactly once");
                let calls = Cell::new(0u32);
                let floor = || {
                    calls.set(calls.get() + 1);
                    today
                };
                assert_eq!(lease::peek_with(&authority, holder, floor), today);
                assert_eq!(calls.get(), 1, "peek evaluates today's gate exactly once");
                assert_eq!(lease::release_with(&authority, holder), None);
            }
        }
        assert!(authority.snapshot().events.is_empty(), "nothing recorded");
    }

    /// Armed: the live envelope stays the FLOOR (a refusal there reserves
    /// nothing and is never overridden by a lease), a denied lease refuses
    /// the lane, a held lease is never debited twice, and a release returns
    /// it.
    #[test]
    fn router_r4_lease_gate_floor_and_denial_paths() {
        let authority = small_lease();
        // Floor false: refused, and the lease is never even asked.
        assert!(!lease::admit_with(
            &authority,
            LeaseHolder::MarginRowConcurrent,
            || false
        ));
        assert_eq!(authority.held(LeaseHolder::MarginRowConcurrent), 0);
        assert!(!lease::peek_with(
            &authority,
            LeaseHolder::PostVerifierTail,
            || false
        ));
        // Floor true: granted while the pool covers it ...
        assert!(lease::admit_with(
            &authority,
            LeaseHolder::MarginRowConcurrent,
            || true
        ));
        assert_eq!(authority.held(LeaseHolder::MarginRowConcurrent), 16 * GIB);
        // ... re-checked at a later stage without a second debit ...
        assert!(lease::admit_with(
            &authority,
            LeaseHolder::MarginRowConcurrent,
            || true
        ));
        assert_eq!(authority.snapshot().outstanding_bytes, 32 * GIB);
        // ... and DENIED once the aggregate would pass the envelope, even
        // though today's live gate says yes (the `Unbounded => true` arm).
        assert!(!lease::peek_with(
            &authority,
            LeaseHolder::PostVerifierTail,
            || true
        ));
        assert!(!lease::admit_with(
            &authority,
            LeaseHolder::PostVerifierTail,
            || true
        ));
        assert!(!lease::admit_with(
            &authority,
            LeaseHolder::OrtRefineWave,
            || true
        ));
        let snap = authority.snapshot();
        assert_eq!(snap.denied, 2);
        assert_eq!(
            snap.outstanding_bytes,
            32 * GIB,
            "a denial reserves nothing"
        );
        // The verifier returns its base lease: the tail fits now.
        assert_eq!(
            lease::release_with(&authority, LeaseHolder::Terminal),
            Some(16 * GIB)
        );
        assert!(lease::admit_with(
            &authority,
            LeaseHolder::PostVerifierTail,
            || true
        ));
        assert_eq!(
            lease::release_with(&authority, LeaseHolder::PostVerifierTail),
            Some(16 * GIB)
        );
        assert_eq!(
            lease::release_with(&authority, LeaseHolder::PostVerifierTail),
            None
        );
        // Arming lines name the envelope, the terminal and the floor.
        let plan = authority.plan().expect("armed");
        let (record, lines) = lease::arm_with(&LeaseAuthority::new(), Some(&plan));
        assert!(record.armed);
        assert!(
            lines[0].starts_with("[router] lease armed rule=CR-L envelope_gib=32.0 ")
                && lines[0].contains(" terminal=granted:16.0gib ")
                && lines[0].ends_with(" floor=live_envelope"),
            "{lines:?}"
        );
        let (idle, lines) = lease::arm_with(&LeaseAuthority::new(), None);
        assert!(!idle.armed);
        assert_eq!(lines, ["[router] lease idle rule=CR-L reason=not_acting"]);
    }

    /// The lease is never a verdict input: its glue and its authority name no
    /// bound, certificate, tolerance or publisher, and every scored gate
    /// passes it TODAY's live decision as the floor.
    #[test]
    fn router_r4_lease_never_alters_a_certified_value() {
        let needles = [
            format!("{}{}", "Bounded", "Tensor"),
            format!("{}{}", "certif", "icate"),
            format!("{}{}", "Certi", "fied"),
            format!("{}{}", "toler", "ance"),
            format!("{}_{}", "publish", "lane"),
            format!("{}{}", "Vnncomp", "Result"),
            format!("{}{}", "BetaCrown", "Config"),
        ];
        for rel in [
            "crates/ny-cli/src/router/lease.rs",
            "crates/ny-core/src/memory_lease.rs",
            "crates/ny-core/src/router/lease.rs",
        ] {
            let code: String = read(rel)
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            for needle in &needles {
                assert!(!code.contains(needle.as_str()), "{rel} names `{needle}`");
            }
        }
        // Every scored gate hands the lease today's live decision as its
        // floor: the live gate is never bypassed, only AND-ed, and no site
        // calls it around the lease.
        let entry = read("crates/ny-cli/src/commands/vnncomp.rs");
        let gates = entry.matches("crate::router::lease::admit(").count()
            + entry.matches("crate::router::lease::peek(").count();
        assert_eq!(gates, 7, "the wired gate census (5 admit, 2 peek)");
        // Six gates pass the live gate itself; the ORT VJP wave passes its own
        // today's gate (the live gate OR its isolated-run override), which
        // still calls the live gate inside.
        let floors = entry.matches("optional_heavy_memory_allowed_now,").count()
            + entry.matches("ort_refine_vjp_memory_allowed_now,").count();
        assert_eq!(
            floors, gates,
            "every lease gate carries today's gate as its floor"
        );
        let wrapper = entry
            .find("fn ort_refine_vjp_memory_allowed_now() -> bool {")
            .expect("the VJP wave's own gate");
        assert!(
            entry[wrapper..]
                .lines()
                .take(4)
                .any(|l| l.contains("optional_heavy_memory_allowed_now(),")),
            "the VJP wave's gate is today's live gate plus its override"
        );
        let called = entry.matches("optional_heavy_memory_allowed_now()").count();
        assert_eq!(
            called, 2,
            "the definition and the VJP wave's own gate; every other site passes it to the lease"
        );
    }

    /// The registry declares safe points for every capability an acting rule
    /// may end early, and a deadline slot is only as suspendable as its least
    /// suspendable capability.
    #[test]
    fn router_r4_registry_stop_sites_declare_safe_points() {
        for site in [
            StopSite::AlphaAscent,
            StopSite::IntermAlpha,
            StopSite::AttackUpfront,
        ] {
            let decl = registry::find(site.token())
                .unwrap_or_else(|| panic!("{} is undeclared", site.token()));
            assert_eq!(decl.safe_points, SafePoints::BetweenUnits, "{site:?}");
        }
        // The one wired CR-F source declares its wave boundary.
        let lowdim = registry::find("lowdim").expect("lowdim");
        assert_eq!(lowdim.safe_points, SafePoints::BetweenUnits);
        assert!(lowdim.accepts.hand_on_source);
        // Slots: the attack lanes follow their capability; the margin-row
        // workers and the STE-PGD lane have none.
        assert_eq!(
            registry::slot_safe_points(LaneSlot::AttackUpfront),
            SafePoints::BetweenUnits
        );
        assert_eq!(
            registry::slot_safe_points(LaneSlot::AttackSignSpace),
            SafePoints::BetweenUnits
        );
        for slot in [
            LaneSlot::AttackStePgd,
            LaneSlot::MarginRowConcurrent,
            LaneSlot::MarginRowInline,
            LaneSlot::Verifier,
        ] {
            assert_eq!(
                registry::slot_safe_points(slot),
                SafePoints::None,
                "{slot:?}"
            );
        }
        // No registry entry is STRICTLY sealed in R4: CR-F has no recipient.
        assert!(registry::REGISTRY
            .iter()
            .all(|d| d.ledger != ny_core::router::decl::LedgerStatus::Sealed));
    }

    /// R2's attack claim is a shortening only when the cell is shorter than
    /// today's window: allowed for a lane with safe points, REFUSED (today's
    /// window kept) for the STE-PGD lane, which has none.
    #[test]
    fn router_r4_attack_claim_never_shortens_a_lane_without_safe_points() {
        // A 30 s scored deadline, every slot capped 5 s before it: two 20 s
        // contracts cannot both fit the 25 s window, so the SECOND one is
        // trimmed to 5 s at commit and its cell is shorter than today's 20 s.
        let armed = || {
            let authority = Authority::new();
            let anchor = Instant::now();
            authority
                .arm(
                    Frame::new(
                        anchor,
                        anchor + Duration::from_secs(30),
                        [Duration::from_secs(5); SLOTS],
                        Duration::from_secs(2),
                    ),
                    [true; SLOTS],
                    Some(LaneSlot::Verifier),
                    None,
                )
                .expect("arm");
            authority
        };
        let today = LaneWindow::Cap(Duration::from_secs(20));
        // Control, with safe points: the upfront lane is trimmed and HANDED
        // its shorter cell.
        let with_points = armed();
        let frame = AttackSliceAllocation::frame(TodaySlices {
            sign_space: Some(Duration::from_secs(20)),
            ste_pgd: None,
            upfront: Some(Duration::from_secs(20)),
        })
        .expect("a frame");
        acting::commit_attack_contracts_with(&with_points, true, Some(&frame));
        let upfront =
            acting::claim_attack_window_with(&with_points, LaneSlot::AttackUpfront, today);
        assert!(
            matches!(upfront, LaneWindow::Cap(w) if w < Duration::from_secs(20)),
            "control: a lane with safe points is shortened to its cell: {upfront:?}"
        );
        // Without safe points (STE-PGD's fixed anneal): today's window, kept.
        let without = armed();
        let frame = AttackSliceAllocation::frame(TodaySlices {
            sign_space: Some(Duration::from_secs(20)),
            ste_pgd: Some(Duration::from_secs(20)),
            upfront: None,
        })
        .expect("a frame");
        acting::commit_attack_contracts_with(&without, true, Some(&frame));
        assert!(
            without
                .contract(LaneSlot::AttackStePgd)
                .is_some_and(|c| c < Duration::from_secs(20)),
            "the STE-PGD contract was trimmed below today's window"
        );
        assert_eq!(
            acting::claim_attack_window_with(&without, LaneSlot::AttackStePgd, today),
            today,
            "a lane without safe points is never shortened"
        );
    }

    /// The FAR glue arms OBSERVE mode when no route exists and records every
    /// live decision with the samples it read.
    #[test]
    fn router_r4_far_glue_arms_observe_mode_and_records_decisions() {
        let ctx = printer_ctx();
        let (_, plan) = super::super::printer_plan(&ctx).expect("printer plan");
        let authority = FarAuthority::new();
        let (record, lines) = far::arm_with(
            &authority,
            &plan,
            Some(&FarCommit {
                routes: Vec::new(),
                observe_only: true,
            }),
            Instant::now(),
        );
        assert!(record.armed && authority.is_armed());
        assert_eq!(
            lines,
            ["[router] far armed rule=CR-F mode=observe routes=[-] ckpt=c2,c3 window=work_units"]
        );
        let (idle, lines) = far::arm_with(&FarAuthority::new(), &plan, None, Instant::now());
        assert!(!idle.armed);
        assert_eq!(lines, ["[router] far idle rule=CR-F reason=not_eligible"]);
        let snap = authority.snapshot();
        let done = far::done_line(&far::FarRecord {
            armed: true,
            snapshot: Some(snap.clone()),
        });
        assert!(
            done.starts_with("[router] far done armed=1 observe_only=1 calls=0 "),
            "{done}"
        );
        assert_eq!((far::event_lines(&snap)).len(), 0);
    }

    /// The printer shows what acting set 3 would commit on a row: CR-L with
    /// the declared host's envelope, CR-F observe-only (no sealed recipient).
    #[test]
    fn router_r4_printer_names_acting_set_three() {
        let ctx = printer_ctx();
        let (input, plan) = super::super::printer_plan(&ctx).expect("printer plan");
        let lines = super::super::printer_lines(&input, &plan);
        let cr_l = lines
            .iter()
            .find(|l| l.starts_with("[router] act rule=CR-L "))
            .expect("CR-L line");
        assert!(cr_l.contains(" acts=1 "), "{cr_l}");
        assert!(cr_l.contains("declared_host_m5"), "{cr_l}");
        let cr_f = lines
            .iter()
            .find(|l| l.starts_with("[router] act rule=CR-F "))
            .expect("CR-F line");
        assert!(
            cr_f.contains(" acts=0 blocked_by=no_sealed_recipient "),
            "{cr_f}"
        );
    }

    /// Review (router-r4): CR-F is STRUCTURALLY inert as built, not merely
    /// waiting for a seal. Even with EVERY registry capability strictly
    /// `Sealed` and admitted, no route exists: the only source the lanes
    /// consult (lowdim) owns the verifier's deadline slot, its only possible
    /// sealed recipient (input_split) owns that same slot, the margin-row
    /// lanes own two slots, and the tiers / variants the program names as the
    /// first recipients (frozen-dual-bab, wave-beta-bab) are not prover lanes
    /// and own no slot. A route therefore needs a registry / slot-map change
    /// (a prover lane with its OWN slot after the source) — and that change
    /// must also re-derive time conservation, because lowdim runs INSIDE the
    /// verifier slot whose unspent cell it would hand on. This test fails
    /// when such a change lands, so the claim is re-examined then.
    #[test]
    fn router_r4_review_far_has_no_route_even_with_every_capability_sealed() {
        use ny_core::router::acting::{self as acting_policy, RuleSwitches};
        use ny_core::router::checkpoint::far_recipient;
        use ny_core::router::plan::{AdmissionState, LedgerState};
        use ny_core::router::r4 as r4_policy;
        use ny_core::router::EffectiveMode;

        let mut input = super::super::policy_input(&printer_ctx());
        assert!(
            input.capabilities.len() >= 20,
            "the registry projected: {}",
            input.capabilities.len()
        );
        for cap in &mut input.capabilities {
            cap.ledger = LedgerState::Sealed;
            cap.admission = AdmissionState::Admit;
        }
        let plan = ny_core::router::plan::plan(&input);
        assert!(!plan.short_row, "a 100 s row: CR-F is eligible");
        let r2 = acting_policy::commit(&plan, &input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        let r4 = r4_policy::commit(
            &plan,
            &input,
            EffectiveMode::Acting,
            &R4Switches::ALL_ON,
            &r2,
        );
        let far = r4
            .far
            .as_ref()
            .expect("CR-F is eligible (CR-1 acts, long row)");
        assert!(
            far.routes.is_empty() && far.observe_only,
            "a CR-F route exists now: re-derive the claim and time conservation: {:?}",
            far.routes
        );
        assert_eq!((r4_policy::far_routes(&plan, &r2)).len(), 0);
        // Non-vacuous: sealing DOES give lowdim a recipient; the slot map is
        // what refuses the route.
        let to = far_recipient(&plan, "lowdim").expect("a sealed recipient after lowdim");
        assert_eq!(to.id, "input_split");
        assert_eq!(r4_policy::unique_slot("lowdim"), Some(LaneSlot::Verifier));
        assert_eq!(
            r4_policy::unique_slot("input_split"),
            Some(LaneSlot::Verifier)
        );
        // Every (source, sealed recipient) pair the plan can form shares a
        // slot or has none.
        let mut pairs = 0usize;
        for source in &plan.caps {
            let Ok(recipient) = far_recipient(&plan, &source.id) else {
                continue;
            };
            pairs += 1;
            let (from, to) = (
                r4_policy::unique_slot(&source.id),
                r4_policy::unique_slot(&recipient.id),
            );
            assert!(
                from.is_none() || to.is_none() || from == to,
                "{} -> {} could route {from:?} -> {to:?}",
                source.id,
                recipient.id
            );
        }
        assert!(pairs >= 1, "the sealed plan forms at least one pair");
        // The named first recipients are not prover lanes and own no slot.
        for id in ["frozen-dual-bab", "wave-beta-bab"] {
            let cap = plan
                .caps
                .iter()
                .find(|c| c.id == id)
                .unwrap_or_else(|| panic!("{id} is in the plan's view"));
            assert!(!cap.prover, "{id} is not a prover lane");
            assert_eq!(r4_policy::unique_slot(id), None, "{id} owns no slot");
        }
    }
}

/// #router-r5 (ROUTER_PROGRAM §3.5, §10 R5): capabilities ROUTED, not armed.
mod r5 {
    use std::collections::BTreeSet;

    use ny_core::route_authority::RouteAuthority;
    use ny_core::router::decl::{ids, KnobRisk, LedgerStatus, OpKind, OpSet, SpecSet, SrLevel};
    use ny_core::router::plan::{LedgerState, PolicyInput};
    use ny_core::router::route::{self as route_policy, R5Input, R5Switches};
    use ny_core::router::EffectiveMode;

    use super::super::{policy_input, registry, route, RouterPolicy, StartContext};
    use super::{printer_ctx, read, workspace_root};
    use crate::plan_resolver::ModelFacts;

    fn frozen_bool(config: &[(&'static ny_levers::LeverDecl, bool)]) -> ny_levers::LeverSet {
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
        ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            config
                .iter()
                .find(|(candidate, _)| std::ptr::eq(*candidate, decl))
                .map(|(_, value)| ny_levers::LeverValue::Bool(*value))
        })
        .expect("admissible rule configuration")
    }

    /// A scanned model with a COMPLETE operator signature (conv / relu /
    /// linear / residual), so every precondition is evaluable.
    fn scanned_model() -> ModelFacts {
        ModelFacts {
            param_count: 2_536_344,
            conv_layers: 19,
            max_conv_out_channels: 128,
            file_size_bytes: 10_156_168,
            op_tokens: [
                "Conv",
                "Relu",
                // #router-r5 review: an initializer-weight Gemm scans as
                // `Gemm:const` (linear); a bare `Gemm` is two computed factors.
                "Gemm:const",
                "Add",
                "Flatten",
                "BatchNormalization",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        }
    }

    /// The real registry's policy input on the scanned model.
    fn real_input(model: &ModelFacts) -> PolicyInput {
        let ctx = StartContext {
            model: Some(model),
            ..printer_ctx()
        };
        policy_input(&ctx)
    }

    fn acting(input: &PolicyInput, pins: &[&str]) -> route_policy::R5Commitments {
        route_policy::commit(
            input,
            &R5Input {
                pinned_keys: pins.iter().map(|p| (*p).to_owned()).collect(),
            },
            EffectiveMode::Acting,
            &R5Switches::ALL_ON,
        )
    }

    /// The scan's operator tokens become the router's COMPLETE, conservative
    /// signature; without tokens (the printer path) it stays a lower bound.
    #[test]
    fn router_r5_facts_carry_a_complete_conservative_operator_signature() {
        let input = real_input(&scanned_model());
        assert!(input.facts.ops_complete);
        let expect = OpSet::EMPTY
            .with(OpKind::Conv)
            .with(OpKind::Relu)
            .with(OpKind::Linear)
            .with(OpKind::Residual)
            .with(OpKind::BatchNorm);
        assert_eq!(input.facts.ops, expect, "Flatten carries no kind");
        let mut untokened = scanned_model();
        untokened.op_tokens.clear();
        assert!(!real_input(&untokened).facts.ops_complete);
        // Ambiguous operators take the harder kind; unknown ones are `other`.
        use super::super::facts::op_kinds;
        assert_eq!(op_kinds("Add"), OpSet::EMPTY.with(OpKind::Residual));
        assert_eq!(op_kinds("Add:const"), OpSet::EMPTY.with(OpKind::Linear));
        assert_eq!(op_kinds("MatMul"), OpSet::EMPTY.with(OpKind::Bilinear));
        assert_eq!(op_kinds("MatMul:const"), OpSet::EMPTY.with(OpKind::Linear));
        assert_eq!(op_kinds("Gather"), OpSet::EMPTY.with(OpKind::Other));
        assert_eq!(op_kinds("Reshape"), OpSet::EMPTY);
    }

    /// #router-r5 review: the operand POSITION decides the kind. The
    /// builder's mapping read `Gemm` as linear whatever its factors, `c / x`
    /// (a reciprocal) as linear, a computed-kernel convolution as a plain
    /// convolution and `Cast` (a rounding cast when its target is an integer
    /// type; the scan does not read `to`) as no kind at all — each one an
    /// UNDER-report that would let a sealed envelope over linear evidence
    /// cover a row it never saw. Each now takes the harder kind.
    #[test]
    fn router_r5_review_op_kinds_take_the_harder_kind_by_operand_position() {
        use super::super::facts::op_kinds;
        let bilinear = OpSet::EMPTY.with(OpKind::Bilinear);
        assert_eq!(
            op_kinds("Gemm"),
            bilinear,
            "Gemm(x, y): two computed factors"
        );
        assert_eq!(op_kinds("Gemm:const"), OpSet::EMPTY.with(OpKind::Linear));
        assert_eq!(op_kinds("Div"), bilinear, "c / x and x / y are not linear");
        assert_eq!(op_kinds("Div:const"), OpSet::EMPTY.with(OpKind::Linear));
        assert_eq!(
            op_kinds("Conv:dynamic"),
            OpSet::EMPTY.with(OpKind::Conv).with(OpKind::Bilinear)
        );
        assert_eq!(
            op_kinds("ConvTranspose:dynamic"),
            OpSet::EMPTY
                .with(OpKind::ConvTranspose)
                .with(OpKind::Bilinear)
        );
        assert_eq!(op_kinds("Conv"), OpSet::EMPTY.with(OpKind::Conv));
        assert_eq!(op_kinds("Cast"), OpSet::EMPTY.with(OpKind::Other));
        // End to end: a scanned row whose only nonlinearity is a reciprocal
        // `c / x` is NOT inside an envelope of linear + relu evidence.
        let mut model = scanned_model();
        model.op_tokens = ["Gemm:const", "Relu", "Div"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        let input = real_input(&model);
        assert!(input.facts.ops_complete);
        let linear_relu = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
        assert!(
            !input.facts.ops.is_subset_of(linear_relu),
            "a reciprocal must not read as linear: {:?}",
            input.facts.ops
        );
    }

    /// SEAL GATING WITH THE REAL REGISTRY: no capability is sealed today, so
    /// with `router.mode: on`, complete facts and nothing pinned, CR-RS routes
    /// NOTHING on any row — each routable capability is refused at `sr:decl`
    /// and carries the moat refusal (every route key is High risk).
    #[test]
    fn router_r5_real_registry_routes_nothing() {
        let input = real_input(&scanned_model());
        let c = acting(&input, &[]);
        assert_eq!(c.routed().count(), 0, "nothing is sealed: nothing routes");
        assert!(!c.acts(route_policy::R5_RULES[0]));
        assert_eq!(
            c.rules[0].blocked_by.as_deref(),
            Some("no_sealed_capability")
        );
        let routable: Vec<&str> = registry::REGISTRY
            .iter()
            .filter(|decl| decl.routable.is_some())
            .map(|decl| decl.id.as_str())
            .collect();
        let decided: Vec<&str> = c.routes.iter().map(|d| d.cap.as_str()).collect();
        assert_eq!(decided, routable, "one decision per routable capability");
        assert_eq!(
            routable.len(),
            10,
            "the eight R5 capabilities, #kfsb-production's kfsb-single-shot and \
             #prepared-root-route's prepared-root are routable"
        );
        for d in &c.routes {
            assert_eq!(d.sr, SrLevel::Decl, "{}", d.cap);
            assert_eq!(d.blocked_by(), Some("sr:decl"), "{}", d.cap);
            assert!(
                d.reasons.contains(&"moat:high_unsealed".to_owned()),
                "{}: {:?}",
                d.cap,
                d.reasons
            );
            assert!(!d.pin_debt);
        }
        // CR-G still arms (it needs no seal): the gap contract funds tiers.
        assert!(c.gap_contract && c.acts(route_policy::R5_RULES[1]));
    }

    /// A TEST-ONLY SEALED FIXTURE proves the mechanism: the same real
    /// registry input with ONE capability's SR level, ledger and envelope
    /// set to sealed routes exactly that capability; the authority arms with
    /// its settings; the configuration boundary applies them (from -> to);
    /// and the authority observes the consult.
    #[test]
    fn router_r5_sealed_fixture_routes_configures_and_observes() {
        fn seal(input: &mut PolicyInput, id: &str) {
            let ops = input.facts.ops;
            let cap = input
                .capabilities
                .iter_mut()
                .find(|cap| cap.id == id)
                .unwrap_or_else(|| panic!("{id} is registered"));
            cap.ledger = LedgerState::Sealed;
            let spec = cap.route.as_mut().expect("routable");
            spec.sr = SrLevel::Seal;
            spec.envelope_ops = ops;
            spec.envelope_specs = SpecSet::ALL;
        }
        let model = scanned_model();
        // A verifier-configuration capability.
        let mut input = real_input(&model);
        seal(&mut input, ids::CGAN_CHILD_ZONO.as_str());
        let c = acting(&input, &[]);
        let routed: Vec<&str> = c.routed().map(|d| d.cap.as_str()).collect();
        assert_eq!(routed, [ids::CGAN_CHILD_ZONO.as_str()]);
        assert!(c.acts(route_policy::R5_RULES[0]));
        let authority = RouteAuthority::new();
        let (record, lines) = route::arm_with(&authority, &c);
        assert!(record.armed && authority.is_armed());
        assert!(lines.iter().any(|l| l.starts_with(
            "[router] route cap=cgan-child-zono sr=seal routed=1 blocked_by=- pin_debt=0"
        )));
        assert!(lines.iter().any(
            |l| l == "[router] route armed rule=CR-RS routes=[cgan-child-zono] gap_contract=1"
        ));
        let mut config = ny_propagate::BetaCrownConfig::default();
        assert!(!config.alpha_config.cgan_child_zono_refresh.enabled);
        let settings = authority
            .settings(ids::CGAN_CHILD_ZONO)
            .expect("the routed settings");
        let applied = route::apply_config_settings(&mut config, ids::CGAN_CHILD_ZONO, &settings);
        assert!(
            config.alpha_config.cgan_child_zono_refresh.enabled,
            "configured"
        );
        assert_eq!(applied.len(), 1);
        assert_eq!(
            route::applied_line(&applied[0]),
            "[router] route applied cap=cgan-child-zono \
             key=solver.alpha_crown.cgan_child_zono_refresh.enabled from=false to=true \
             source=router"
        );
        assert!(
            !authority.routed(ids::FROZEN_DUAL_BAB),
            "unrouted: today's key"
        );
        authority.note_applied(applied[0].clone());
        let snapshot = authority.snapshot();
        assert_eq!(snapshot.routes[0].consults, 1, "the consult is observed");
        assert_eq!(snapshot.applied.len(), 1);
        assert!(route::done_line(true, &snapshot).starts_with(
            "[router] route done armed=1 routed=1 consults=[cgan-child-zono:1] applied=1"
        ));

        // A margin-row capability (the device wave is refused on a host
        // without a device; the CPU wave takes the slot).
        let mut input = real_input(&model);
        input.host.device = ny_core::router::facts::DeviceQual::None;
        seal(&mut input, ids::WAVE_BETA_BAB_CUDA.as_str());
        seal(&mut input, ids::WAVE_BETA_BAB.as_str());
        let c = acting(&input, &[]);
        let routed: Vec<&str> = c.routed().map(|d| d.cap.as_str()).collect();
        assert_eq!(routed, [ids::WAVE_BETA_BAB.as_str()]);
        let cuda = c
            .routes
            .iter()
            .find(|d| d.cap == ids::WAVE_BETA_BAB_CUDA.as_str())
            .expect("decided");
        assert_eq!(cuda.blocked_by(), Some("admit:decline:no-device"));
        let mut policy = ny_propagate::margin_row::MarginRowRuntimePolicy::default();
        let applied = route::apply_margin_row_settings(
            &mut policy,
            ids::WAVE_BETA_BAB,
            &c.routed().next().expect("routed").settings,
        );
        assert_eq!(policy.wave.mode, ny_propagate::margin_row::TreeMode::Wave);
        assert_eq!(
            (applied[0].from.as_str(), applied[0].to.as_str()),
            ("legacy", "wave")
        );
        // With a device, the device variant takes the slot first.
        input.host.device = ny_core::router::facts::DeviceQual::Search;
        let c = acting(&input, &[]);
        let routed: Vec<&str> = c.routed().map(|d| d.cap.as_str()).collect();
        assert_eq!(routed, [ids::WAVE_BETA_BAB_CUDA.as_str()]);
        assert_eq!(
            c.routes
                .iter()
                .find(|d| d.cap == ids::WAVE_BETA_BAB.as_str())
                .and_then(|d| d.blocked_by()),
            Some("slot:tree_mode:wave-beta-bab-cuda")
        );

        // The frozen-dual tier's lane latch reads the authority directly.
        let mut input = real_input(&model);
        seal(&mut input, ids::FROZEN_DUAL_BAB.as_str());
        let c = acting(&input, &[]);
        let authority = RouteAuthority::new();
        let _ = route::arm_with(&authority, &c);
        assert!(authority.routed(ids::FROZEN_DUAL_BAB));
        // A pin blocks the sealed fixture and is reported as pin debt.
        let pinned = acting(&input, &["margin_row.frozen_dual_bab"]);
        assert_eq!(pinned.routed().count(), 0);
        // #prepared-root-route: by id (prepared-root is now the first routable).
        let fd = pinned
            .routes
            .iter()
            .find(|d| d.cap == ids::FROZEN_DUAL_BAB.as_str())
            .expect("decided");
        assert!(fd.pin_debt && fd.blocked_by() == Some("pin:margin_row.frozen_dual_bab"));
    }

    /// The cover certification routes ONLY as its bank arm RF (replay +
    /// containment fence + root ORT adjudication), never `ledger` alone.
    #[test]
    fn router_r5_vit_cover_routes_as_its_rf_bank_arm() {
        let decl = registry::find(ids::VIT_COVER_CERTIFY.as_str()).expect("registered");
        let settings: Vec<(String, String)> = decl
            .routable
            .expect("routable")
            .route
            .settings
            .iter()
            .map(|s| (s.key.to_owned(), s.value.to_owned()))
            .collect();
        assert!(settings
            .iter()
            .all(|(k, v)| !(k == "bab.mo_cover_certify" && v == "ledger")));
        let mut config = ny_propagate::BetaCrownConfig::default();
        let applied = route::apply_config_settings(&mut config, ids::VIT_COVER_CERTIFY, &settings);
        assert_eq!(applied.len(), 3);
        assert!(
            applied.iter().all(|a| a.blocked_by.is_none()),
            "{applied:?}"
        );
        assert_eq!(
            config.mo_cover_certify,
            ny_propagate::MoCoverCertify::Replay
        );
        assert!(config.mo_containment_fence && config.mo_root_ort_adjudication);
    }

    /// The bi-window route on a row whose `backward_interm` phase is off is
    /// applied but recorded INERT (the window crops that phase's passes).
    #[test]
    fn router_r5_bi_window_route_is_recorded_inert_without_backward_interm() {
        let mut policy = ny_propagate::margin_row::MarginRowRuntimePolicy::default();
        assert!(!policy.backward_interm.enabled);
        let applied = route::apply_margin_row_settings(
            &mut policy,
            ids::BI_WINDOW_JOINT_REFRESH,
            &[("margin_row.bi_window".to_owned(), "true".to_owned())],
        );
        assert!(policy.backward_interm.window);
        assert_eq!(
            applied[0].blocked_by.as_deref(),
            Some("inert:backward_interm_off")
        );
        // A value outside the vocabulary is refused, never coerced.
        let mut policy = ny_propagate::margin_row::MarginRowRuntimePolicy::default();
        let applied = route::apply_margin_row_settings(
            &mut policy,
            ids::WAVE_BETA_BAB,
            &[("margin_row.tree_mode".to_owned(), "warp".to_owned())],
        );
        assert_eq!(policy.wave.mode, ny_propagate::margin_row::TreeMode::Legacy);
        assert_eq!(applied[0].blocked_by.as_deref(), Some("bad_value"));
    }

    /// Unarmed (every row today), the two configuration boundaries are
    /// today's values, byte for byte.
    #[test]
    fn router_r5_adapters_are_todays_values_when_unarmed() {
        assert!(!ny_core::route_authority::global().is_armed());
        let mut config = ny_propagate::BetaCrownConfig::default();
        let before = format!("{config:?}");
        route::apply_to_verifier_config(&mut config);
        assert_eq!(format!("{config:?}"), before);
        let mut policy = ny_propagate::margin_row::MarginRowRuntimePolicy::default();
        let before = format!("{policy:?}");
        route::apply_to_margin_row(&mut policy);
        assert_eq!(format!("{policy:?}"), before);
    }

    /// Pins: the row's own file under any path the key may be written at, or
    /// a non-default lever source. Nothing else.
    #[test]
    fn router_r5_pinned_keys_come_from_the_file_and_non_default_levers() {
        let dark = frozen_bool(&[]);
        assert_eq!((route::pinned_keys(None, Some(&dark))).len(), 0);
        assert_eq!(
            (route::pinned_keys(Some(&crate::preset::PresetConfig::default()), None)).len(),
            0
        );
        let file: crate::preset::PresetConfig = serde_yaml::from_str(
            "margin_row:\n  frozen_dual_bab: false\nbab:\n  forward_root_screen: true\n  \
             alpha_crown:\n    cgan_child_zono_refresh:\n      enabled: true\n",
        )
        .expect("parses");
        assert_eq!(
            route::pinned_keys(Some(&file), None),
            [
                "bab.forward_root_screen",
                "margin_row.frozen_dual_bab",
                "solver.alpha_crown.cgan_child_zono_refresh.enabled",
            ],
            "a pin-off is a pin; the bab.alpha_crown twin pins the solver key"
        );
        // A lever resolved from the typed layer pins its key.
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
        let levers = ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            std::ptr::eq(
                decl,
                &raw const ny_levers::decls::margin_row::MARGIN_ROW_TREE_MODE,
            )
            .then(|| ny_levers::LeverValue::Text("legacy".to_owned()))
        })
        .expect("admissible");
        assert_eq!(
            route::pinned_keys(None, Some(&levers)),
            ["margin_row.tree_mode"]
        );
    }

    /// Every route key reaches a consumer (an adapter), every route's moat
    /// risk is at least its keys' declared risk (a key with no declaration
    /// counts High), and every routable registry row's SR ladder, ledger and
    /// route match `docs/engine_program_capabilities.yaml` (SR is DERIVED:
    /// seal needs S7 + S8 done and a recorded envelope).
    #[test]
    fn router_r5_registry_ladders_match_the_capability_registry() {
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&read("docs/engine_program_capabilities.yaml"))
                .expect("the capability registry parses");
        let capabilities = yaml
            .get("capabilities")
            .and_then(serde_yaml::Value::as_sequence)
            .expect("capabilities list");
        let mut violations = Vec::new();
        let mut checked = 0usize;
        for decl in registry::REGISTRY {
            let Some(routable) = decl.routable else {
                continue;
            };
            let id = decl.id.as_str();
            for setting in routable.route.settings {
                if route::adapter_for(setting.key).is_none() {
                    violations.push(format!("{id}: route key {} has no adapter", setting.key));
                }
                let declared = ny_levers::space::axes()
                    .iter()
                    .chain(ny_levers::space::unsafe_axes())
                    .find(|a| a.deliver == ny_levers::space::Deliver::PresetKey(setting.key))
                    .and_then(|a| ny_levers::all().get(a.name));
                let high = declared.is_none_or(|d| d.moat == ny_levers::MoatRisk::High);
                if high && routable.route.risk != KnobRisk::High {
                    violations.push(format!("{id}: {} is High risk", setting.key));
                }
            }
            let Some(entry) = capabilities
                .iter()
                .find(|c| c.get("id").and_then(serde_yaml::Value::as_str) == Some(id))
            else {
                violations.push(format!("{id}: no capability entry"));
                continue;
            };
            let record = entry.get("record").expect("record");
            let done =
                |stage: &str| record.get(stage).and_then(serde_yaml::Value::as_str) == Some("done");
            let sr = record.get("SR").and_then(serde_yaml::Value::as_str);
            let block = entry.get("routable").expect("routable block");
            let envelope_len = block
                .get("envelope")
                .and_then(serde_yaml::Value::as_sequence)
                .map_or(0, Vec::len);
            let ladder = routable.ladder;
            if ladder.s7 != done("S7") || ladder.s8 != done("S8") {
                violations.push(format!(
                    "{id}: compiled S7/S8 {}/{} vs yaml {}/{}",
                    ladder.s7,
                    ladder.s8,
                    done("S7"),
                    done("S8")
                ));
            }
            if ladder.obs != matches!(sr, Some("obs" | "seal" | "done")) {
                violations.push(format!(
                    "{id}: compiled obs {} vs yaml SR {sr:?}",
                    ladder.obs
                ));
            }
            let compiled_envelope = decl.envelope.ops.0 != 0;
            if compiled_envelope != (envelope_len > 0) {
                violations.push(format!("{id}: envelope recorded in one registry only"));
            }
            let ledger = block.get("ledger").and_then(serde_yaml::Value::as_str);
            let compiled = match decl.ledger {
                LedgerStatus::Open(_) => "open",
                LedgerStatus::Shadow => "shadow",
                LedgerStatus::SealedAsShipped => "sealed_as_shipped",
                LedgerStatus::Sealed => "sealed",
            };
            if ledger != Some(compiled) {
                violations.push(format!("{id}: ledger {ledger:?} vs {compiled}"));
            }
            if matches!(decl.ledger, LedgerStatus::Sealed) != (decl.sr_level() == SrLevel::Seal) {
                violations.push(format!("{id}: ledger Sealed iff SR.seal"));
            }
            let yaml_route: Vec<(String, String)> = block
                .get("route")
                .and_then(serde_yaml::Value::as_sequence)
                .map(|items| {
                    items
                        .iter()
                        .map(|item| {
                            let field = |k: &str| {
                                item.get(k)
                                    .and_then(serde_yaml::Value::as_str)
                                    .unwrap_or_default()
                                    .to_owned()
                            };
                            (field("key"), field("value"))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let compiled_route: Vec<(String, String)> = routable
                .route
                .settings
                .iter()
                .map(|s| (s.key.to_owned(), s.value.to_owned()))
                .collect();
            if yaml_route != compiled_route {
                violations.push(format!("{id}: route {yaml_route:?} vs {compiled_route:?}"));
            }
            checked += 1;
        }
        assert_eq!(
            checked, 10,
            "every routable capability was checked (#kfsb-production: + kfsb-single-shot; \
             #prepared-root-route: + prepared-root)"
        );
        assert!(
            violations.is_empty(),
            "the routable registry and docs/engine_program_capabilities.yaml disagree:\n  {}",
            violations.join("\n  ")
        );
    }

    /// The deny list: a routable capability's route never sets a router
    /// selector, a precision, a reserve or a tolerance key.
    #[test]
    fn router_r5_routes_set_no_denied_key() {
        for decl in registry::REGISTRY {
            if let Some(routable) = decl.routable {
                for setting in routable.route.settings {
                    assert!(
                        !ny_levers::router_action_space::denied(setting.key),
                        "{}: {} is on the router's deny list",
                        decl.id.as_str(),
                        setting.key
                    );
                }
            }
        }
    }

    #[test]
    fn router_r5_policy_resolves_r5_switches_from_the_frozen_levers() {
        use ny_levers::decls::router::{ROUTER_RULE_GAP_CONTRACT, ROUTER_RULE_ROUTE};
        let dark = frozen_bool(&[]);
        let policy = RouterPolicy::from_levers(&dark).expect("dark");
        assert_eq!(policy.r5_rules, R5Switches::ALL_ON, "both rules default ON");
        assert_eq!(policy, RouterPolicy::OFF, "and the router defaults off");
        let ablated = frozen_bool(&[(&ROUTER_RULE_ROUTE, false)]);
        let policy = RouterPolicy::from_levers(&ablated).expect("typed");
        assert!(!policy.r5_rules.route && policy.r5_rules.gap_contract);
        assert_eq!(
            ablated
                .resolved(&ROUTER_RULE_ROUTE)
                .expect("declared")
                .source,
            ny_levers::Source::Config
        );
        let pinned = frozen_bool(&[
            (&ROUTER_RULE_ROUTE, false),
            (&ROUTER_RULE_GAP_CONTRACT, false),
        ]);
        assert!(
            super::super::pins(Some(&pinned)).is_empty(),
            "the two switches are router selectors, never plan pins"
        );
    }

    /// The printer names acting set 4 and every routable capability's
    /// decision (nothing routes: nothing is sealed).
    #[test]
    fn router_r5_printer_names_acting_set_four() {
        let model = scanned_model();
        let ctx = StartContext {
            model: Some(&model),
            ..printer_ctx()
        };
        let (input, plan) = super::super::printer_plan(&ctx).expect("printer plan");
        let lines = super::super::printer_lines(&input, &plan);
        for prefix in [
            "[router] act rule=CR-RS acts=0 blocked_by=no_sealed_capability",
            "[router] act rule=CR-G acts=1",
            "[router] route cap=frozen-dual-bab sr=decl routed=0 blocked_by=sr:decl",
            "[router] route cap=vit-cover-certify sr=decl routed=0",
        ] {
            assert!(
                lines.iter().any(|l| l.starts_with(prefix)),
                "missing `{prefix}` in {lines:#?}"
            );
        }
    }

    /// THE YAML-DEBT RATCHET. Every shipped category file line that sets a
    /// routable capability's route key is a row: `debt` when the capability
    /// is SEALED (the router routes it per instance; the file decides
    /// instead) — a count that may only FALL — and `unsealed` otherwise (the
    /// file is today's only delivery). The census must equal
    /// `crates/ny-cli/src/router/router_yaml_debt.tsv` exactly.
    #[test]
    fn router_yaml_debt_matches_manifest() {
        /// Pin debt rows today. MAY ONLY FALL: lower it when a debt row is
        /// retired; a new debt row means a sealed capability is still armed
        /// by a file — retire the line (the router routes it), never raise
        /// this number.
        const BASELINE_DEBT: usize = 0;

        fn walk<'a>(node: &'a serde_yaml::Value, path: &str) -> Option<&'a serde_yaml::Value> {
            path.split('.').try_fold(node, |node, part| {
                node.as_mapping()?.iter().find_map(|(k, v)| {
                    let key = k.as_str()?.replace('-', "_");
                    (key == part).then_some(v)
                })
            })
        }
        fn token(value: &serde_yaml::Value) -> String {
            match value {
                serde_yaml::Value::Bool(b) => b.to_string(),
                serde_yaml::Value::String(s) => s.clone(),
                serde_yaml::Value::Number(n) => n.to_string(),
                other => format!("{other:?}").replace(char::is_whitespace, "_"),
            }
        }

        let root = workspace_root();
        let mut rows: BTreeSet<String> = BTreeSet::new();
        let mut scanned = 0usize;
        let mut control = 0usize;
        for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
            let dir = root.join("configs").join(year);
            let mut files: Vec<_> = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "yaml"))
                .collect();
            files.sort();
            for path in files {
                let text = std::fs::read_to_string(&path).expect("read yaml");
                let doc: serde_yaml::Value = serde_yaml::from_str(&text)
                    .unwrap_or_else(|e| panic!("{} does not parse: {e}", path.display()));
                scanned += 1;
                if walk(&doc, "margin_row.backward_interm").is_some() {
                    control += 1;
                }
                let rel = path
                    .strip_prefix(&root)
                    .expect("under the workspace")
                    .display()
                    .to_string();
                for decl in registry::REGISTRY {
                    let Some(routable) = decl.routable else {
                        continue;
                    };
                    let state = if decl.sr_level() == SrLevel::Seal {
                        "debt"
                    } else {
                        "unsealed"
                    };
                    for setting in routable.route.settings {
                        let mut paths = vec![setting.key.to_owned()];
                        if let Some(rest) = setting.key.strip_prefix("solver.alpha_crown.") {
                            paths.push(format!("bab.alpha_crown.{rest}"));
                        }
                        for p in paths {
                            if let Some(value) = walk(&doc, &p).filter(|v| !v.is_null()) {
                                rows.insert(format!(
                                    "{state}\t{rel}\t{p}\t{}\t{}",
                                    token(value),
                                    decl.id.as_str()
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            scanned >= 40,
            "the census read {scanned} category files: VOID"
        );
        assert!(
            control > 0,
            "positive control: some shipped file sets margin_row.backward_interm, and the walk \
             must find it or a zero census proves nothing"
        );
        let manifest: BTreeSet<String> = read("crates/ny-cli/src/router/router_yaml_debt.tsv")
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(str::to_owned)
            .collect();
        let added: Vec<&String> = rows.difference(&manifest).collect();
        let removed: Vec<&String> = manifest.difference(&rows).collect();
        assert!(
            added.is_empty() && removed.is_empty(),
            "router_yaml_debt.tsv is stale: added {added:#?}, removed {removed:#?}; the census \
             body is:\n{}",
            rows.iter().cloned().collect::<Vec<_>>().join("\n")
        );
        let debt = rows.iter().filter(|r| r.starts_with("debt\t")).count();
        assert_eq!(
            debt, BASELINE_DEBT,
            "pin debt may only FALL: lower BASELINE_DEBT when a row is retired; never raise it \
             (route the sealed capability and delete the file line)"
        );
        // Positive control of the classifier: a capability at seal would be
        // debt; none is today.
        assert!(registry::REGISTRY
            .iter()
            .filter(|d| d.routable.is_some())
            .all(|d| d.sr_level() == SrLevel::Decl));
    }
}

// ---------------------------------------------------------------------------
// #router-r6: the O-TRIAL glue (policy, arming lines, pins, registry)
// ---------------------------------------------------------------------------

mod r6 {
    use ny_core::router::decl::SafePoints;
    use ny_core::router::otrial::{self as otrial_policy, R6Input, TrialSwitches, R6_RULES};
    use ny_core::router::EffectiveMode;
    use ny_core::trial_authority::{TrialAuthority, TrialSite};

    use super::super::{otrial, policy_input, registry, RouterPolicy};
    use super::printer_ctx;

    fn frozen_bool(config: &[(&'static ny_levers::LeverDecl, bool)]) -> ny_levers::LeverSet {
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
        ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            config
                .iter()
                .find(|(candidate, _)| std::ptr::eq(*candidate, decl))
                .map(|(_, value)| ny_levers::LeverValue::Bool(*value))
        })
        .expect("admissible rule configuration")
    }

    #[test]
    fn router_r6_policy_resolves_trial_switches_from_the_frozen_levers() {
        use ny_levers::decls::router::{
            OPTIMIZER_RULE_TRIAL_ATTACK_HALVING, OPTIMIZER_RULE_TRIAL_WIDTH,
        };
        let dark = frozen_bool(&[]);
        let policy = RouterPolicy::from_levers(&dark).expect("dark");
        assert_eq!(
            policy.trial_rules,
            TrialSwitches::ALL_ON,
            "every trial rule defaults ON"
        );
        assert_eq!(
            policy,
            RouterPolicy::OFF,
            "and the optimizer itself defaults off"
        );
        let ablated = frozen_bool(&[
            (&OPTIMIZER_RULE_TRIAL_WIDTH, false),
            (&OPTIMIZER_RULE_TRIAL_ATTACK_HALVING, false),
        ]);
        let policy = RouterPolicy::from_levers(&ablated).expect("typed");
        assert!(!policy.trial_rules.width && !policy.trial_rules.attack_halving);
        assert!(policy.trial_rules.lr_backoff && policy.trial_rules.split_restart);
        assert_eq!(
            ablated
                .resolved(&OPTIMIZER_RULE_TRIAL_WIDTH)
                .expect("declared")
                .source,
            ny_levers::Source::Config,
            "the switch arrives from the typed layer"
        );
    }

    /// The real registry's input: every trial site's capability is declared
    /// with safe points between units, so under `optimizer.mode: on` every
    /// site of a SEALED lane arms, and the arming lines name every rule and
    /// every site. Review fix (router-r6): the wave tree (`wave-beta-bab`) is
    /// `Open` in the registry (S7+S8 empty until M6), so its width site is
    /// blocked `ledger:wave-beta-bab`; the rule still acts at its other sites.
    #[test]
    fn router_r6_arm_lines_name_every_rule_and_site() {
        let input = policy_input(&printer_ctx());
        let plan = ny_core::router::plan::plan(&input);
        let commitments = otrial_policy::commit(
            &plan,
            &input,
            &R6Input::default(),
            EffectiveMode::Acting,
            &TrialSwitches::ALL_ON,
        );
        let unsealed = |site: TrialSite| site == TrialSite::WaveWidth;
        for site in TrialSite::ALL {
            assert_eq!(
                commitments.directives.site_armed(site),
                !unsealed(site),
                "{site:?}: {:?}",
                commitments.site(site)
            );
        }
        assert_eq!(
            commitments
                .site(TrialSite::WaveWidth)
                .and_then(|s| s.blocked_by.as_deref()),
            Some("ledger:wave-beta-bab"),
            "the registry's wave tree is Open: never trialled"
        );
        let authority = TrialAuthority::new();
        let (record, lines) = otrial::arm_with(&authority, &commitments);
        assert!(record.armed && authority.is_armed());
        for rule in R6_RULES {
            let prefix = format!("[optimizer] act rule={} acts=1", rule.as_str());
            assert!(
                lines.iter().any(|l| l.starts_with(&prefix)),
                "{prefix}\n{lines:#?}"
            );
        }
        for site in TrialSite::ALL {
            let prefix = format!(
                "[router] trial-site site={} knob={} rule={} armed={} blocked_by={}",
                site.token(),
                site.knob(),
                site.rule(),
                u8::from(!unsealed(site)),
                if unsealed(site) {
                    "ledger:wave-beta-bab"
                } else {
                    "-"
                }
            );
            assert!(lines.iter().any(|l| l == &prefix), "{prefix}\n{lines:#?}");
        }
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("[router] trial authority armed sha=")
                    && l.contains("trial_width=1")
                    && l.contains("lowdim/lowdim.batch")
                    && !l.contains("wave-beta-bab/margin_row.tree_wave_batch")),
            "{lines:#?}"
        );
        let done = otrial::finish_with(&authority, record);
        assert!(!authority.is_armed(), "the session's close disarms");
        assert!(
            otrial::done_line(&done).starts_with("[router] trial done armed=1 events=0 started=0"),
            "{}",
            otrial::done_line(&done)
        );
        // Shadow commits nothing and arms nothing.
        let shadow = otrial_policy::commit(
            &plan,
            &input,
            &R6Input::default(),
            EffectiveMode::Shadow,
            &TrialSwitches::ALL_ON,
        );
        let quiet = TrialAuthority::new();
        let (record, lines) = otrial::arm_with(&quiet, &shadow);
        assert!(!record.armed && !quiet.is_armed());
        assert!(lines
            .iter()
            .any(|l| l == "[router] trial authority idle reason=no_site_armed"));
    }

    /// Every trial site is a registered capability that declares safe points
    /// between units (a trial changes a value only there), and every R6 rule
    /// is a registered rule.
    #[test]
    fn router_r6_registry_declares_the_rules_and_the_sites_safe_points() {
        for site in TrialSite::ALL {
            let decl = registry::find(site.token())
                .unwrap_or_else(|| panic!("{} is not registered", site.token()));
            assert_eq!(decl.safe_points, SafePoints::BetweenUnits, "{site:?}");
        }
        for rule in R6_RULES {
            assert!(registry::find(rule.as_str()).is_some(), "{rule:?}");
        }
    }

    /// A knob the row's file sets — under every path it may be written at —
    /// or a lever resolved from a non-default source PINS its site.
    #[test]
    fn router_r6_pinned_keys_come_from_the_file_and_non_default_levers() {
        let dark = frozen_bool(&[]);
        assert_eq!((otrial::pinned_keys(None, Some(&dark))).len(), 0);
        assert_eq!(
            (otrial::pinned_keys(Some(&crate::preset::PresetConfig::default()), None)).len(),
            0
        );
        let file: crate::preset::PresetConfig = serde_yaml::from_str(
            "margin_row:\n  tree_wave_batch: 256\nsolver:\n  batch_size: 64\nbab:\n  branching:\n    \
             input_split:\n      lowdim_split: sbwr\n",
        )
        .expect("parses");
        assert_eq!(
            otrial::pinned_keys(Some(&file), None),
            [
                "bab.branching.input_split.lowdim_split",
                "margin_row.tree_wave_batch",
                "solver.batch_size",
            ],
            "a pin at the declaration default is still a pin"
        );
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
        let levers = ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            std::ptr::eq(
                decl,
                &raw const ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_LR,
            )
            .then_some(ny_levers::LeverValue::F64(0.1))
        })
        .expect("admissible");
        assert_eq!(
            otrial::pinned_keys(None, Some(&levers)),
            ["margin_row.alpha_lr"]
        );
    }

    /// Unarmed, the upfront lane's step pool is `None` and the pool helpers
    /// are no-ops: every restart starts at today's step.
    #[test]
    fn router_r6_glue_is_identity_when_the_trial_authority_is_unarmed() {
        assert!(!ny_core::trial_authority::global().is_armed());
        assert!(otrial::upfront_step_pool().is_none());
        let mut pool = None;
        assert_eq!(otrial::pool_next(&mut pool), None);
        otrial::pool_done(&mut pool, 0, 1.0, 1.0, 0.1);
        assert_eq!(
            (ny_core::trial_authority::global().snapshot().events).len(),
            0
        );
    }

    /// Review fix (router-r6): the upfront lane's step pool FAILS OPEN like
    /// every lane-side machine. While the authority is armed the helpers hand
    /// out the pool's arms (restart 0 is today's 0.25) and record its events;
    /// once the authority is DISARMED (a router fault elsewhere, or the
    /// session's close) the very next restart DROPS the pool and takes
    /// today's step. (The lane used to call the machine bare and keep the
    /// pool after a disarm.)
    #[test]
    fn router_r6_step_pool_drops_when_the_authority_disarms() {
        use ny_core::router::trial::{Halving, HalvingRule, TrialDecision};
        use ny_core::trial_authority::TrialDirectives;
        let authority = TrialAuthority::new();
        authority
            .arm(TrialDirectives {
                attack_halving: Some(HalvingRule::SHIPPED),
                sites: TrialSite::AttackStep.bit(),
                ..TrialDirectives::NONE
            })
            .expect("arm the attack site");
        let mut pool = Halving::new(
            HalvingRule::SHIPPED,
            "rule.trial_attack_halving",
            TrialSite::AttackStep.token(),
            TrialSite::AttackStep.knob(),
        );
        assert!(pool.is_some());
        let mut steps = Vec::new();
        for _ in 0..8 {
            let (arm, step) = otrial::pool_next_with(&authority, &mut pool).expect("armed");
            steps.push(step);
            otrial::pool_done_with(&authority, &mut pool, arm, step, 1.0, 0.1);
        }
        assert_eq!(steps[0], 0.25, "restart 0 is today's restart");
        assert!(
            steps.contains(&0.5),
            "the pool is live while armed: {steps:?}"
        );
        assert!(
            authority
                .snapshot()
                .events
                .iter()
                .any(|e| e.decision == TrialDecision::Kept),
            "the round's elimination was recorded"
        );
        authority.disarm();
        assert_eq!(
            otrial::pool_next_with(&authority, &mut pool),
            None,
            "disarmed: today's step"
        );
        assert!(pool.is_none(), "and the pool is dropped for good");
        // A pool held while the authority is disarmed is dropped at the
        // restart's END too.
        let mut late = Halving::new(HalvingRule::SHIPPED, "r", "s", "k");
        otrial::pool_done_with(&authority, &mut late, 0, 1.0, 1.0, 0.1);
        assert!(late.is_none());
    }
}

/// #router-r7: the offline cost-model prior on the ny-cli side — the key's
/// resolution, the policy input with and without it, and the printer/replay.
mod r7 {
    use std::time::{Duration, Instant};

    use ny_core::router::facts::HostClass;
    use ny_core::router::hash;
    use ny_core::router::plan;
    use ny_core::router::priors::{PriorTable, RULE_PRIORS};
    use ny_core::router::PRIORS_VERSION;

    use super::super::{
        policy_input, policy_input_with, registry, Authority, RouterPolicy, StartContext,
    };
    use crate::plan_resolver::ModelFacts;

    fn frozen_bool(config: &[(&'static ny_levers::LeverDecl, bool)]) -> ny_levers::LeverSet {
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
        ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            config
                .iter()
                .find(|(candidate, _)| std::ptr::eq(*candidate, decl))
                .map(|(_, value)| ny_levers::LeverValue::Bool(*value))
        })
        .expect("admissible configuration")
    }

    /// An MLP scan with a complete signature (linear + relu).
    fn mlp() -> ModelFacts {
        ModelFacts {
            param_count: 65_536,
            conv_layers: 0,
            max_conv_out_channels: 0,
            file_size_bytes: 300_000,
            op_tokens: ["Gemm:const", "Relu", "Flatten"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }
    }

    fn ctx(model: &ModelFacts, host: HostClass) -> StartContext<'_> {
        StartContext {
            levers: None,
            authority: Authority::Frozen,
            model: Some(model),
            model_bytes: Some(model.file_size_bytes),
            facts_reason: None,
            declared_networks: Some(1),
            relational_selected: Some(false),
            official_budget_secs: 100,
            internal_secs: 95,
            join_margin: Duration::from_secs(2),
            anchor: Instant::now(),
            target_host: Some(host),
            pinned_keys: Vec::new(),
            trial_pinned_keys: Vec::new(),
            // #admission-precision: no property read (every site check defers).
            spec: crate::commands::beta_crown::spec_shape::SpecRead::Unknown("r7-test"),
        }
    }

    #[test]
    fn router_r7_priors_key_is_dark_by_default_and_typed_from_the_frozen_levers() {
        use ny_levers::decls::router::OPTIMIZER_PRIORS;
        let dark = frozen_bool(&[]);
        let policy = RouterPolicy::from_levers(&dark).expect("dark");
        assert!(!policy.priors, "the router reads no prior unless asked");
        assert_eq!(policy, RouterPolicy::OFF);
        assert_eq!(
            dark.resolved(&OPTIMIZER_PRIORS).expect("declared").source,
            ny_levers::Source::Default
        );
        let typed = frozen_bool(&[(&OPTIMIZER_PRIORS, true)]);
        let policy = RouterPolicy::from_levers(&typed).expect("typed");
        assert!(policy.priors);
        assert_eq!(
            typed.resolved(&OPTIMIZER_PRIORS).expect("declared").source,
            ny_levers::Source::Config,
            "the key arrives from the typed layer"
        );
        // The key alone never starts a session: it selects what a running
        // router reads.
        assert!(!super::super::requested(&typed));
        // ...and it is a selector, not a pin of the plan.
        let pins = super::super::pins(Some(&typed));
        assert!(!pins.iter().any(|p| p == OPTIMIZER_PRIORS.name), "{pins:?}");
    }

    #[test]
    fn router_r7_policy_input_without_priors_is_r5_bit_identical() {
        let model = mlp();
        for host in [HostClass::Gb10, HostClass::M5] {
            let c = ctx(&model, host);
            let off = policy_input(&c);
            let explicit = policy_input_with(&c, false);
            assert_eq!(off, explicit);
            assert_eq!(off.priors, None);
            assert_eq!(off.priors_version, PRIORS_VERSION);
            assert!(off
                .capabilities
                .iter()
                .all(|cap| cap.first_rung_ms.is_none()));
            let json = serde_json::to_value(&off).expect("serializes");
            assert!(
                json.get("priors").is_none(),
                "no new field in an R5-shaped record"
            );
            let p = plan::plan(&off);
            assert!(!p.rules_fired.iter().any(|r| r == RULE_PRIORS.as_str()));
        }
    }

    #[test]
    fn router_r7_policy_input_with_priors_prices_only_on_the_measured_host() {
        let table = PriorTable::compiled().expect("the compiled prior loads");
        let model = mlp();
        let on_gb10 = policy_input_with(&ctx(&model, HostClass::Gb10), true);
        let view = on_gb10
            .priors
            .as_ref()
            .expect("the prior's view rides the input");
        assert_eq!(view.version, table.version_token());
        assert_eq!(on_gb10.priors_version, table.version_token());
        assert_eq!(view.cell.as_deref(), Some("conv=0;nonpwl=0"));
        // The shipped phase is predicted on the measured host...
        assert!(
            view.predictions.iter().any(|p| p.phase == "verifier.setup"),
            "{view:?}"
        );
        // ...but v1 prices no first rung: the attack's phase was refused by
        // G7 and the verifier's first window (root.bound) by G6.
        assert!(view.priced.is_empty(), "{view:?}");
        assert!(view
            .unpriced
            .iter()
            .any(|u| u.cap == "attack.upfront_apgd" && u.reason == "attack.upfront:refused"));
        // Every capability's rung is exactly what the view priced (none).
        for cap in &on_gb10.capabilities {
            assert_eq!(cap.first_rung_ms, view.rung_ms(&cap.id), "{}", cap.id);
        }
        let p = plan::plan(&on_gb10);
        assert!(p.rules_fired.iter().any(|r| r == RULE_PRIORS.as_str()));
        assert_ne!(
            hash::static_hash(&on_gb10),
            hash::static_hash(&policy_input(&ctx(&model, HostClass::Gb10)))
        );
        // A different host class prices nothing (the table is GB10's).
        let on_m5 = policy_input_with(&ctx(&model, HostClass::M5), true);
        let view = on_m5.priors.as_ref().expect("the view still rides, empty");
        assert!(
            view.priced.is_empty() && view.predictions.is_empty(),
            "{view:?}"
        );
        assert!(
            view.unpriced.iter().all(|u| u.reason == "host:m5"),
            "{view:?}"
        );
        assert!(on_m5.capabilities.iter().all(|c| c.first_rung_ms.is_none()));
    }

    #[test]
    fn router_r7_every_binding_names_a_registered_capability() {
        let table = PriorTable::compiled().expect("the compiled prior loads");
        for binding in &table.bindings {
            assert!(
                registry::find(&binding.cap).is_some(),
                "the prior binds {} but the registry does not declare it",
                binding.cap
            );
        }
        assert!(
            registry::find(RULE_PRIORS.as_str()).is_some(),
            "rule.priors is a registered rule"
        );
    }

    #[test]
    fn router_r7_printer_lines_carry_the_prior() {
        let model = mlp();
        let input = policy_input_with(&ctx(&model, HostClass::Gb10), true);
        let plan = plan::plan(&input);
        let lines = super::super::printer_lines(&input, &plan);
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("[router] would rule=rule.priors priors=cost-v1+")),
            "{lines:#?}"
        );
        let banner = &lines[0];
        assert!(banner.contains("priors=cost-v1+"), "{banner}");
        let line = super::super::telemetry::priors_line(input.priors.as_ref().expect("view"));
        assert!(line.starts_with("[router] priors v=cost-v1+"), "{line}");
        assert!(line.contains(" priced=- "), "{line}");
        assert!(
            line.contains("unpriced=attack.upfront_apgd:attack.upfront:refused,"),
            "{line}"
        );
        assert!(
            line.contains("relu_split.graph:root.bound:refused"),
            "{line}"
        );
        assert!(line.contains(" predict=verifier.setup:"), "{line}");
    }
}

// ============================================================================
// #prepared-root-route — the default-on prepared-root route, registered with
// a callable admission (the lane's own, extracted).
// ============================================================================

mod prepared_root {
    use ny_core::router::decl::{
        Admission, AdmissionRef, CapKind, KnobRisk, LaneRole, LedgerStatus, SrLadder,
    };

    use super::super::census_tests::{call_sites, code_only, fn_body, read};
    use super::super::registry;
    use crate::commands::prepared_root_route::router_admission;

    fn armed(decl: &'static ny_levers::LeverDecl) -> ny_levers::LeverSet {
        ny_levers::LeverSet::freeze_with(ny_levers::all(), |key| {
            (key == decl.name).then(|| "1".to_owned())
        })
    }

    /// The declaration: a prover lane of its own that never takes the
    /// remainder, whose admission is a CALLABLE predicate (so it is no
    /// admission debt), routable with no key to set (it runs by default:
    /// CR-RS neither arms nor disarms it) and an honest ladder — nothing
    /// recorded beyond the declaration, its S8 moat open.
    #[test]
    fn prepared_root_is_registered_with_a_callable_admission() {
        let decl = registry::find("prepared-root").expect("prepared-root is registered");
        assert_eq!(
            decl.kind,
            CapKind::Lane {
                role: LaneRole::Prover { terminal_ok: false }
            }
        );
        assert_eq!(
            decl.admission,
            AdmissionRef::Predicate {
                function: "prepared_root_route::admission"
            }
        );
        assert!(
            matches!(decl.ledger, LedgerStatus::Open(_)),
            "{:?}",
            decl.ledger
        );
        let routable = decl.routable.expect("routable: the census scopes its moat");
        assert_eq!(routable.ladder, SrLadder::DECLARED, "S7/S8/obs all open");
        assert!(
            routable.route.settings.is_empty(),
            "no key: it runs by default"
        );
        assert_eq!(routable.route.risk, KnobRisk::High, "it can publish unsat");
        assert_eq!(routable.route.slot, None);
        assert!(
            !registry::admission_debt()
                .iter()
                .any(|(id, _, _)| *id == "prepared-root"),
            "a callable predicate is no admission debt"
        );
    }

    /// The registry asks the route's OWN admission on the frozen lever set
    /// the router started with: admitted on the defaults, declined with the
    /// lane's reason where the lane would decline, and unknown (a refusal)
    /// with no lever set.
    #[test]
    fn prepared_root_registry_admission_is_the_routes_own_predicate() {
        let decl = registry::find("prepared-root").expect("registered");
        let facts = ny_core::router::facts::InstanceFacts::known_empty();
        let ask = |levers: Option<&ny_levers::LeverSet>| {
            registry::admit(
                decl,
                &registry::Context {
                    facts: &facts,
                    relational_selected: Some(false),
                    levers,
                },
            )
        };
        assert_eq!(ask(None), Admission::Unknown("no-frozen-levers"));
        let defaults = super::super::census::no_category_levers();
        assert_eq!(ask(Some(&defaults)), Admission::Admit);
        assert_eq!(router_admission(&defaults), Admission::Admit);
        for decl in [
            &ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB,
            &ny_levers::decls::margin_row::MARGIN_ROW_BETA,
            &ny_levers::decls::sound_channel::MARGIN_ROW_CLIP,
        ] {
            let levers = armed(decl);
            let name = decl.name;
            assert_eq!(
                ask(Some(&levers)),
                Admission::Decline("unsupported-armed-consumer-dose"),
                "{name}"
            );
            assert_eq!(ask(Some(&levers)), router_admission(&levers), "{name}");
        }
    }

    /// On a default-policy row the router's plan admits it as a prover that
    /// is never the terminal (it falls through to the lanes after it).
    #[test]
    fn prepared_root_is_admitted_and_never_terminal_in_the_plan() {
        let defaults = super::super::census::no_category_levers();
        let ctx = super::super::StartContext {
            levers: Some(&defaults),
            ..super::printer_ctx()
        };
        let (input, plan) = super::super::printer_plan(&ctx).expect("printer plan");
        let cap = input
            .capabilities
            .iter()
            .find(|c| c.id == "prepared-root")
            .expect("asked");
        assert!(cap.admission.is_admit(), "{:?}", cap.admission);
        assert!(plan.admitted.iter().any(|id| id == "prepared-root"));
        assert_ne!(plan.terminal.as_deref(), Some("prepared-root"));
    }

    /// SOURCE RATCHET — ONE IMPLEMENTATION: the route runs only from the
    /// scored entry point and only on `admission`'s answer (it takes the
    /// token, never a lever set); the entry point's classwise read goes to the
    /// admission and nowhere else; the policy gate has exactly one caller, the
    /// admission; the router registry asks `router_admission`, which asks
    /// `admission`; and the census starts the router with a lever set.
    #[test]
    fn prepared_root_lane_router_and_census_call_the_one_admission() {
        let entry = (
            "crates/ny-cli/src/commands/vnncomp.rs".to_owned(),
            "run_and_translate".to_owned(),
        );
        let src = "crates/ny-cli/src";
        assert_eq!(
            call_sites(src, &["try_prepared_root_route("]),
            [entry.clone()]
        );
        assert_eq!(
            call_sites(src, &["prepared_root_route::admission("]),
            [entry]
        );
        assert_eq!(
            call_sites(src, &["router_admission("]),
            [(
                "crates/ny-cli/src/router/registry.rs".to_owned(),
                "predicate".to_owned()
            )]
        );
        assert_eq!(
            call_sites(src, &["policy_gate("]),
            [(
                "crates/ny-cli/src/commands/prepared_root_route.rs".to_owned(),
                "admission".to_owned()
            )]
        );
        let route = code_only(&read("crates/ny-cli/src/commands/prepared_root_route.rs"));
        let lane = fn_body(&route, "try_prepared_root_route");
        assert!(lane.contains("admission: PreparedRootAdmission,"), "{lane}");
        assert!(!lane.contains("LeverSet"), "the lane cannot decide itself");
        assert!(
            fn_body(&route, "router_admission").contains("admission(false, &policy, levers)"),
            "the router asks the one admission"
        );
        let vnncomp = code_only(&read("crates/ny-cli/src/commands/vnncomp.rs"));
        let scored = fn_body(&vnncomp, "run_and_translate");
        let source = scored
            .find("safenlp_input_head::try_input_head_with_continuation(")
            .expect("the bounded Source attempt remains on the scored path");
        let prepared = scored
            .find("prepared_root_route::try_prepared_root_route(")
            .expect("the prepared fallback remains on the scored path");
        assert!(source < prepared,
            "the default Source portfolio must receive its bounded opportunity before prepared search");
        assert_eq!(
            vnncomp.matches("prepared_classwise_requested").count(),
            2,
            "the classwise read is defined once and handed to the admission once"
        );
        assert!(
            vnncomp.contains(
                "super::prepared_root_route::admission(\n        prepared_classwise_requested,\n"
            ),
            "the classwise read is the admission's first argument"
        );
        assert!(vnncomp.contains("            prepared_root_admission,\n"));
        let census = code_only(&read("crates/ny-cli/src/router/census.rs"));
        assert!(census.contains("levers: Some(levers),"));
    }
}
