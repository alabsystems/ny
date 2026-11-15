// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ROUTER_PROGRAM §11.2 tests 23–25 (`ny-levers`).
//!
//! * `router_mode_keys_are_declared_and_delivered_by_preset` — the two router
//!   keys are LeverDecls with a `Deliver::PresetKey` axis, and a preset-fed
//!   value resolves from `Source::Config`, never `Source::LegacyEnv`.
//! * `router_action_space_matches_manifest` — the derived action space equals
//!   `tests/data/router_action_space.tsv`; any delta fails in both directions
//!   and the COMPLETE derived body is printed (regenerate only from it).
//! * `router_action_space_excludes_precision_and_reserve_keys` — the deny-list
//!   is real (every named key exists), does work (it removes keys the
//!   class/moat filter alone would admit), and nothing denied survives.

use std::collections::BTreeSet;

use ny_levers::decls::router::{
    OPTIMIZER_MODE, OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR, OPTIMIZER_RULE_IA_YIELD,
    OPTIMIZER_RULE_ROOT_ADMISSION, OPTIMIZER_RULE_TRIAL_ATTACK_HALVING,
    OPTIMIZER_RULE_TRIAL_LR_BACKOFF, OPTIMIZER_RULE_TRIAL_SPLIT_RESTART,
    OPTIMIZER_RULE_TRIAL_WIDTH, ROUTER_MODE, ROUTER_RULE_ADMISSION, ROUTER_RULE_CAP_AUTHORITY,
    ROUTER_RULE_CONTRACTS, ROUTER_RULE_DERIVED_CAPS, ROUTER_RULE_FAR_HAND_ON,
    ROUTER_RULE_GAP_CONTRACT, ROUTER_RULE_HAND_ON, ROUTER_RULE_LEASE, ROUTER_RULE_PUBLISH_GRACE,
    ROUTER_RULE_ROUTE, ROUTER_RULE_TERMINAL,
};
use ny_levers::router_action_space::{
    action_space, candidates, denied, render_rows, DENY_KEYS, DENY_PATTERNS,
};
use ny_levers::space::{self, Class, Deliver, Domain};
use ny_levers::{Bucket, DefaultSpec, LeverKind, LeverValue, MoatRisk, Source};

const MANIFEST: &str = include_str!("data/router_action_space.tsv");

#[test]
fn router_mode_keys_are_declared_and_delivered_by_preset() {
    for (decl, key) in [
        (&ROUTER_MODE, "router.mode"),
        (&OPTIMIZER_MODE, "optimizer.mode"),
    ] {
        let registered = ny_levers::all()
            .get(decl.name)
            .unwrap_or_else(|| panic!("{} is not in the merged registry", decl.name));
        assert!(
            std::ptr::eq(registered, decl),
            "{}: the registry must hold THIS declaration",
            decl.name
        );
        let LeverKind::Enum(tokens) = decl.kind else {
            panic!("{} must be an Enum lever", decl.name);
        };
        assert_eq!(tokens, &["off", "shadow", "on"]);
        assert_eq!(
            decl.default,
            DefaultSpec::Text("off"),
            "{} ships dark",
            decl.name
        );
        assert_eq!(decl.bucket, Bucket::Debug);
        // #router-r2: `router.mode: on` now ACTS (acting set 1 moves lane
        // deadlines and rule-owned caps), so its honest moat risk is High.
        // #router-r3: so does `optimizer.mode: on` (acting set 2 ends phases
        // early), so both modes are High.
        assert_eq!(decl.moat, MoatRisk::High, "{}", decl.name);

        let axis = space::axes()
            .iter()
            .find(|axis| axis.name == decl.name)
            .unwrap_or_else(|| {
                panic!(
                    "{} has no search-space axis: DARK on scored rows",
                    decl.name
                )
            });
        assert_eq!(axis.deliver, Deliver::PresetKey(key));
        assert_eq!(axis.class, Class::SafeToSearch);
        assert_eq!(
            axis.domain,
            Domain::Enum(&["off", "shadow"]),
            "automated search never picks the acting mode"
        );
        assert_eq!(space::preset_baseline_keys(decl.name), &[key]);

        // Absent: the declaration default, from the default layer.
        let dark = ny_levers::read_over_config_with(decl, |_| None, None).expect("dark resolves");
        assert_eq!(dark.value.as_str(), Some("off"));
        assert_eq!(dark.source, Source::Default);

        // Preset-fed: the typed value, from the CONFIG layer.
        for token in ["shadow", "on", "off"] {
            let fed = ny_levers::read_over_config_with(
                decl,
                |_| None,
                Some(LeverValue::Text(token.to_owned())),
            )
            .expect("a declared token is admissible");
            assert_eq!(fed.value.as_str(), Some(token));
            assert_eq!(fed.source, Source::Config, "{} {token}", decl.name);
            assert_ne!(fed.source, Source::LegacyEnv);
        }
        // A typo is an error at the preset boundary, never a silent default.
        assert!(ny_levers::read_over_config_with(
            decl,
            |_| None,
            Some(LeverValue::Text("shadw".to_owned())),
        )
        .is_err());
    }
}

#[test]
fn router_action_space_matches_manifest() {
    let derived = action_space();
    let rendered = render_rows(&derived);
    let derived_rows: BTreeSet<&str> = rendered.lines().collect();
    let manifest_rows: BTreeSet<&str> = MANIFEST
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .collect();
    let undeclared: Vec<&&str> = derived_rows.difference(&manifest_rows).collect();
    let stale: Vec<&&str> = manifest_rows.difference(&derived_rows).collect();
    assert!(
        undeclared.is_empty() && stale.is_empty(),
        "THE ROUTER ACTION SPACE CHANGED.\n\n{} UNDECLARED row(s) (derived, not in the \
         manifest):\n  {}\n\n{} STALE row(s) (in the manifest, no longer derived):\n  {}\n\n\
         Review every delta: an added row is a key a router-emitted value may now name. If the \
         change is intended, replace the body of crates/ny-levers/tests/data/\
         router_action_space.tsv with EXACTLY these {} derived rows (never from an out-of-band \
         parse):\n{rendered}",
        undeclared.len(),
        undeclared
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join("\n  "),
        stale.len(),
        stale
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join("\n  "),
        derived.len(),
    );
    // Positive control: the derivation is not vacuous — the measured lead axis
    // (`judge_srI_actionspace.out`) is in it.
    assert!(
        derived.iter().any(|axis| axis.key == "margin_row.parallel"),
        "margin_row.parallel (SafeToSearch / Low) must be in the action space: {rendered}"
    );
}

#[test]
fn router_action_space_excludes_precision_and_reserve_keys() {
    let delivered: BTreeSet<&str> = space::axes()
        .iter()
        .chain(space::unsafe_axes())
        .filter_map(|axis| match axis.deliver {
            Deliver::PresetKey(key) => Some(key),
            Deliver::EnvOnly => None,
        })
        .collect();
    // The deny-list is real: every named key is a preset-delivered axis.
    for key in DENY_KEYS {
        assert!(
            delivered.contains(key),
            "deny-list key {key} names no preset-delivered axis: a placebo entry"
        );
        assert!(denied(key));
    }
    assert!(DENY_PATTERNS.iter().all(|p| !p.is_empty()));

    let space_keys: BTreeSet<&str> = action_space().iter().map(|axis| axis.key).collect();
    for axis in action_space() {
        assert!(
            !denied(axis.key),
            "{} is denied but in the action space",
            axis.key
        );
        let source = space::axes()
            .iter()
            .find(|a| a.name == axis.lever)
            .expect("a derived axis is a searchable axis");
        assert_ne!(source.class, Class::Unsafe, "{} is Unsafe", axis.key);
        let decl = ny_levers::all().get(axis.lever).expect("declared");
        assert_ne!(decl.moat, MoatRisk::High, "{} is a High knob", axis.key);
    }
    for key in [
        "margin_row.root_f32",
        "margin_row.tableau_f32",
        "margin_row.err_envelope",
        "margin_row.reserve_secs",
        "margin_row.release_frac",
        "router.mode",
        "optimizer.mode",
    ] {
        assert!(
            !space_keys.contains(key),
            "{key} must never be router-emitted"
        );
    }
    for axis in space::unsafe_axes() {
        if let Deliver::PresetKey(key) = axis.deliver {
            assert!(!space_keys.contains(key), "Unsafe axis {key} leaked in");
        }
    }
    // Two independent exclusions of the router's own modes. #router-r2:
    // `router.mode` is High since `on` acts, so the moat filter removes it;
    // #router-r3: `optimizer.mode` is High since `on` acts too. The deny-list
    // still names both, so a future moat downgrade cannot readmit either.
    let raw: BTreeSet<&str> = candidates().iter().map(|axis| axis.key).collect();
    assert!(!raw.contains("router.mode"), "router.mode is High since R2");
    assert!(
        !raw.contains("optimizer.mode"),
        "optimizer.mode is High since R3"
    );
    assert!(denied("router.mode") && denied("optimizer.mode"));
    // Whatever the class/moat filter admits, the deny-list removes exactly
    // the denied keys and nothing else.
    for key in &raw {
        assert_eq!(space_keys.contains(key), !denied(key), "{key}");
    }
    assert!(raw.len() >= space_keys.len());
}

/// #router-r2: each acting rule is its own capability with its own typed key.
/// Every `router.rules.*` switch is declared (BoolOverride, default `true` =
/// participates when the router acts, Debug), delivered by preset through an
/// UNSAFE axis (never searched: a rule joins the shipped acting set by a
/// compiled move after its moat), denied from the router's own action space,
/// and resolves from the typed layer with zero `NY_*`.
#[test]
fn router_rule_keys_are_declared_delivered_and_never_searched() {
    let rules = [
        (
            &ROUTER_RULE_ADMISSION,
            "router.rules.admission",
            MoatRisk::High,
        ),
        (
            &ROUTER_RULE_CONTRACTS,
            "router.rules.contracts",
            MoatRisk::High,
        ),
        (
            &ROUTER_RULE_CAP_AUTHORITY,
            "router.rules.cap_authority",
            MoatRisk::High,
        ),
        (
            &ROUTER_RULE_TERMINAL,
            "router.rules.terminal",
            MoatRisk::Low,
        ),
        (&ROUTER_RULE_HAND_ON, "router.rules.hand_on", MoatRisk::High),
        (
            &ROUTER_RULE_PUBLISH_GRACE,
            "router.rules.publish_grace",
            MoatRisk::High,
        ),
        (
            &ROUTER_RULE_DERIVED_CAPS,
            "router.rules.derived_caps",
            MoatRisk::High,
        ),
    ];
    let action_keys: BTreeSet<&str> = action_space().iter().map(|axis| axis.key).collect();
    for (decl, key, moat) in rules {
        let registered = ny_levers::all()
            .get(decl.name)
            .unwrap_or_else(|| panic!("{} is not in the merged registry", decl.name));
        assert!(std::ptr::eq(registered, decl), "{}", decl.name);
        assert_eq!(decl.kind, LeverKind::BoolOverride, "{}", decl.name);
        assert_eq!(decl.default, DefaultSpec::Bool(true), "{}", decl.name);
        assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
        assert_eq!(decl.moat, moat, "{}: honest moat risk", decl.name);
        assert!(
            space::axes().iter().all(|axis| axis.name != decl.name),
            "{} must never be a searchable axis",
            decl.name
        );
        let axis = space::unsafe_axes()
            .iter()
            .find(|axis| axis.name == decl.name)
            .unwrap_or_else(|| panic!("{} has no axis: DARK on scored rows", decl.name));
        assert_eq!(axis.deliver, Deliver::PresetKey(key));
        assert_eq!(axis.class, Class::Unsafe);
        assert!(
            denied(key),
            "{key}: the router may never emit its own rule switch"
        );
        assert!(!action_keys.contains(key));
        let dark = ny_levers::read_over_config_with(decl, |_| None, None).expect("dark resolves");
        assert_eq!(dark.value, LeverValue::Bool(true), "{}", decl.name);
        assert_eq!(dark.source, Source::Default);
        let fed = ny_levers::read_over_config_with(decl, |_| None, Some(LeverValue::Bool(false)))
            .expect("a typed value is admissible");
        assert_eq!(fed.value, LeverValue::Bool(false), "{}", decl.name);
        assert_eq!(fed.source, Source::Config, "{}", decl.name);
        assert_ne!(fed.source, Source::LegacyEnv);
    }
}

/// #router-r3: each O-STOP optimizer rule is its own capability with its own
/// typed key. Every `optimizer.rules.*` switch is declared (BoolOverride,
/// default `true` = participates when the optimizer acts, Debug, honest
/// MoatRisk High: each ends a phase early), delivered by preset through an
/// UNSAFE axis (never searched), denied from the router's own action space,
/// and resolves from the typed layer with zero `NY_*`.
#[test]
fn optimizer_rule_keys_are_declared_delivered_and_never_searched() {
    let rules = [
        (&OPTIMIZER_RULE_GOVERNOR, "optimizer.rules.governor"),
        (&OPTIMIZER_RULE_IA_YIELD, "optimizer.rules.ia_yield"),
        (&OPTIMIZER_RULE_ATTACK_DEFER, "optimizer.rules.attack_defer"),
        (
            &OPTIMIZER_RULE_ROOT_ADMISSION,
            "optimizer.rules.root_admission",
        ),
    ];
    let action_keys: BTreeSet<&str> = action_space().iter().map(|axis| axis.key).collect();
    for (decl, key) in rules {
        let registered = ny_levers::all()
            .get(decl.name)
            .unwrap_or_else(|| panic!("{} is not in the merged registry", decl.name));
        assert!(std::ptr::eq(registered, decl), "{}", decl.name);
        assert_eq!(decl.kind, LeverKind::BoolOverride, "{}", decl.name);
        assert_eq!(decl.default, DefaultSpec::Bool(true), "{}", decl.name);
        assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
        assert_eq!(decl.moat, MoatRisk::High, "{}: honest moat risk", decl.name);
        assert!(
            space::axes().iter().all(|axis| axis.name != decl.name),
            "{} must never be a searchable axis",
            decl.name
        );
        let axis = space::unsafe_axes()
            .iter()
            .find(|axis| axis.name == decl.name)
            .unwrap_or_else(|| panic!("{} has no axis: DARK on scored rows", decl.name));
        assert_eq!(axis.deliver, Deliver::PresetKey(key));
        assert_eq!(axis.class, Class::Unsafe);
        assert!(
            denied(key),
            "{key}: the router may never emit a rule switch"
        );
        assert!(!action_keys.contains(key));
        let dark = ny_levers::read_over_config_with(decl, |_| None, None).expect("dark resolves");
        assert_eq!(dark.value, LeverValue::Bool(true), "{}", decl.name);
        assert_eq!(dark.source, Source::Default);
        let fed = ny_levers::read_over_config_with(decl, |_| None, Some(LeverValue::Bool(false)))
            .expect("a typed value is admissible");
        assert_eq!(fed.value, LeverValue::Bool(false), "{}", decl.name);
        assert_eq!(fed.source, Source::Config, "{}", decl.name);
        assert_ne!(fed.source, Source::LegacyEnv);
    }
}

/// #router-r4: acting set 3's two rules (CR-L memory lease, CR-F FAR hand-on)
/// are each their own capability with their own typed key: declared
/// (BoolOverride, default `true` = participates when the router acts, Debug,
/// honest MoatRisk High: a refused lane or a hand-on can cost a proof),
/// delivered by preset through an UNSAFE axis (never searched), denied from
/// the router's own action space, and resolved from the typed layer with
/// zero `NY_*`.
#[test]
fn router_r4_rule_keys_are_declared_delivered_and_never_searched() {
    let rules = [
        (&ROUTER_RULE_LEASE, "router.rules.lease"),
        (&ROUTER_RULE_FAR_HAND_ON, "router.rules.far_hand_on"),
    ];
    let action_keys: BTreeSet<&str> = action_space().iter().map(|axis| axis.key).collect();
    for (decl, key) in rules {
        let registered = ny_levers::all()
            .get(decl.name)
            .unwrap_or_else(|| panic!("{} is not in the merged registry", decl.name));
        assert!(std::ptr::eq(registered, decl), "{}", decl.name);
        assert_eq!(decl.kind, LeverKind::BoolOverride, "{}", decl.name);
        assert_eq!(decl.default, DefaultSpec::Bool(true), "{}", decl.name);
        assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
        assert_eq!(decl.moat, MoatRisk::High, "{}: honest moat risk", decl.name);
        assert!(
            space::axes().iter().all(|axis| axis.name != decl.name),
            "{} must never be a searchable axis",
            decl.name
        );
        let axis = space::unsafe_axes()
            .iter()
            .find(|axis| axis.name == decl.name)
            .unwrap_or_else(|| panic!("{} has no axis: DARK on scored rows", decl.name));
        assert_eq!(axis.deliver, Deliver::PresetKey(key));
        assert_eq!(axis.class, Class::Unsafe);
        assert!(
            denied(key),
            "{key}: the router may never emit a rule switch"
        );
        assert!(!action_keys.contains(key));
        let dark = ny_levers::read_over_config_with(decl, |_| None, None).expect("dark resolves");
        assert_eq!(dark.value, LeverValue::Bool(true), "{}", decl.name);
        assert_eq!(dark.source, Source::Default);
        let fed = ny_levers::read_over_config_with(decl, |_| None, Some(LeverValue::Bool(false)))
            .expect("a typed value is admissible");
        assert_eq!(fed.value, LeverValue::Bool(false), "{}", decl.name);
        assert_eq!(fed.source, Source::Config, "{}", decl.name);
        assert_ne!(fed.source, Source::LegacyEnv);
    }
}

/// #router-r5: acting set 4's two rules (CR-RS routing only at SR.seal, CR-G
/// gap-order contracts) are each their own capability with their own typed
/// key: declared (BoolOverride, default `true` = participates when the router
/// acts, Debug, honest MoatRisk High: a routed capability or a re-divided
/// tier can cost a proof), delivered by preset through an UNSAFE axis (never
/// searched), denied from the router's own action space, and resolved from
/// the typed layer with zero `NY_*`.
#[test]
fn router_r5_rule_keys_are_declared_delivered_and_never_searched() {
    let rules = [
        (&ROUTER_RULE_ROUTE, "router.rules.route"),
        (&ROUTER_RULE_GAP_CONTRACT, "router.rules.gap_contract"),
    ];
    let action_keys: BTreeSet<&str> = action_space().iter().map(|axis| axis.key).collect();
    for (decl, key) in rules {
        let registered = ny_levers::all()
            .get(decl.name)
            .unwrap_or_else(|| panic!("{} is not in the merged registry", decl.name));
        assert!(std::ptr::eq(registered, decl), "{}", decl.name);
        assert_eq!(decl.kind, LeverKind::BoolOverride, "{}", decl.name);
        assert_eq!(decl.default, DefaultSpec::Bool(true), "{}", decl.name);
        assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
        assert_eq!(decl.moat, MoatRisk::High, "{}: honest moat risk", decl.name);
        assert!(
            space::axes().iter().all(|axis| axis.name != decl.name),
            "{} must never be a searchable axis",
            decl.name
        );
        let axis = space::unsafe_axes()
            .iter()
            .find(|axis| axis.name == decl.name)
            .unwrap_or_else(|| panic!("{} has no axis: DARK on scored rows", decl.name));
        assert_eq!(axis.deliver, Deliver::PresetKey(key));
        assert_eq!(axis.class, Class::Unsafe);
        assert!(
            denied(key),
            "{key}: the router may never emit a rule switch"
        );
        assert!(!action_keys.contains(key));
        let dark = ny_levers::read_over_config_with(decl, |_| None, None).expect("dark resolves");
        assert_eq!(dark.value, LeverValue::Bool(true), "{}", decl.name);
        assert_eq!(dark.source, Source::Default);
        let fed = ny_levers::read_over_config_with(decl, |_| None, Some(LeverValue::Bool(false)))
            .expect("a typed value is admissible");
        assert_eq!(fed.value, LeverValue::Bool(false), "{}", decl.name);
        assert_eq!(fed.source, Source::Config, "{}", decl.name);
        assert_ne!(fed.source, Source::LegacyEnv);
    }
}

/// #router-r6: each O-TRIAL optimizer rule is its own capability with its own
/// typed key. Every `optimizer.rules.trial_*` switch is declared (BoolOverride,
/// default `true` = participates when the optimizer acts, Debug, honest
/// MoatRisk High: each moves a search / effort value inside a verdict-bearing
/// lane), delivered by preset through an UNSAFE axis (never searched), denied
/// from the router's own action space (so `router_action_space.tsv` gains no
/// row), and resolves from the typed layer with zero `NY_*`.
#[test]
fn optimizer_trial_rule_keys_are_declared_delivered_and_never_searched() {
    let rules = [
        (&OPTIMIZER_RULE_TRIAL_WIDTH, "optimizer.rules.trial_width"),
        (
            &OPTIMIZER_RULE_TRIAL_LR_BACKOFF,
            "optimizer.rules.trial_lr_backoff",
        ),
        (
            &OPTIMIZER_RULE_TRIAL_SPLIT_RESTART,
            "optimizer.rules.trial_split_restart",
        ),
        (
            &OPTIMIZER_RULE_TRIAL_ATTACK_HALVING,
            "optimizer.rules.trial_attack_halving",
        ),
    ];
    let action_keys: BTreeSet<&str> = action_space().iter().map(|axis| axis.key).collect();
    let candidate_keys: BTreeSet<&str> = candidates().iter().map(|axis| axis.key).collect();
    for (decl, key) in rules {
        let registered = ny_levers::all()
            .get(decl.name)
            .unwrap_or_else(|| panic!("{} is not in the merged registry", decl.name));
        assert!(std::ptr::eq(registered, decl), "{}", decl.name);
        assert_eq!(decl.kind, LeverKind::BoolOverride, "{}", decl.name);
        assert_eq!(decl.default, DefaultSpec::Bool(true), "{}", decl.name);
        assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
        assert_eq!(decl.moat, MoatRisk::High, "{}: honest moat risk", decl.name);
        assert!(
            space::axes().iter().all(|axis| axis.name != decl.name),
            "{} must never be a searchable axis",
            decl.name
        );
        let axis = space::unsafe_axes()
            .iter()
            .find(|axis| axis.name == decl.name)
            .unwrap_or_else(|| panic!("{} has no axis: DARK on scored rows", decl.name));
        assert_eq!(axis.deliver, Deliver::PresetKey(key));
        assert_eq!(axis.class, Class::Unsafe);
        assert!(
            denied(key),
            "{key}: the router may never emit a rule switch"
        );
        assert!(!action_keys.contains(key));
        assert!(
            !candidate_keys.contains(key),
            "{key}: an Unsafe axis is never even a candidate, so the action-space manifest \
             gains no row"
        );
        let dark = ny_levers::read_over_config_with(decl, |_| None, None).expect("dark resolves");
        assert_eq!(dark.value, LeverValue::Bool(true), "{}", decl.name);
        assert_eq!(dark.source, Source::Default);
        let fed = ny_levers::read_over_config_with(decl, |_| None, Some(LeverValue::Bool(false)))
            .expect("a typed value is admissible");
        assert_eq!(fed.value, LeverValue::Bool(false), "{}", decl.name);
        assert_eq!(fed.source, Source::Config, "{}", decl.name);
        assert_ne!(fed.source, Source::LegacyEnv);
    }
}

/// #router-r7: `optimizer.priors` — the switch that lets the router READ the
/// compiled offline cost-model prior — is declared (Bool, default `false` =
/// no prior is read, Debug, honest MoatRisk High: a priced rung can change
/// which slot the deadline authority governs), delivered by preset through an
/// UNSAFE axis (never searched), denied from the router's own action space,
/// and resolved from the typed layer with zero `NY_*`.
#[test]
fn router_r7_priors_key_is_declared_delivered_and_never_searched() {
    use ny_levers::decls::router::OPTIMIZER_PRIORS;
    let decl = &OPTIMIZER_PRIORS;
    let key = "optimizer.priors";
    let registered = ny_levers::all()
        .get(decl.name)
        .unwrap_or_else(|| panic!("{} is not in the merged registry", decl.name));
    assert!(std::ptr::eq(registered, decl), "{}", decl.name);
    assert_eq!(decl.kind, LeverKind::Bool, "{}", decl.name);
    assert_eq!(
        decl.default,
        DefaultSpec::Bool(false),
        "dark: no prior unless asked"
    );
    assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
    assert_eq!(decl.moat, MoatRisk::High, "{}: honest moat risk", decl.name);
    assert!(
        space::axes().iter().all(|axis| axis.name != decl.name),
        "{} must never be a searchable axis",
        decl.name
    );
    let axis = space::unsafe_axes()
        .iter()
        .find(|axis| axis.name == decl.name)
        .unwrap_or_else(|| panic!("{} has no axis: DARK on scored rows", decl.name));
    assert_eq!(axis.deliver, Deliver::PresetKey(key));
    assert_eq!(axis.class, Class::Unsafe);
    assert!(
        denied(key),
        "{key}: the router may never emit its own prior switch"
    );
    let action_keys: BTreeSet<&str> = action_space().iter().map(|axis| axis.key).collect();
    assert!(!action_keys.contains(key));
    let dark = ny_levers::read_over_config_with(decl, |_| None, None).expect("dark resolves");
    assert_eq!(dark.value, LeverValue::Bool(false), "{}", decl.name);
    assert_eq!(dark.source, Source::Default);
    let fed = ny_levers::read_over_config_with(decl, |_| None, Some(LeverValue::Bool(true)))
        .expect("a typed value is admissible");
    assert_eq!(fed.value, LeverValue::Bool(true), "{}", decl.name);
    assert_eq!(fed.source, Source::Config, "{}", decl.name);
    assert_ne!(fed.source, Source::LegacyEnv);
}
