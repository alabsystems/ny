// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #metaroom-mip-governor / #cz-hybrid-budget — the DELIVERY evidence for the
//! three keys the metaroom Row A / Row B work added.
//!
//! These are the "named evidence tests": each asserts that the key reaches its
//! consumer THROUGH THE YAML, not through an environment variable the scored
//! harness does not export (`vnncomp_scripts/run_instance.sh` exports exactly
//! one variable, `OMP_NUM_THREADS`). The check is on the resolution SOURCE, not
//! on the value — a value can agree by accident.

use super::*;
use ny_levers::{read_over_config_with, LeverValue, Source};

/// The exact section the metaroom CANARY config carries. Written out here so
/// the schema contract is testable without a file on disk, and so a rename in
/// `preset/mod.rs` breaks this rather than silently making the canary inert.
const CANARY: &str = r"
bab:
  mip:
    leaf_undecided_streak_cap: 1
    escalation_skips_certified_clauses: true
  phase_budget:
    post_bab_pgd_fraction: 0.0
cz_hybrid:
  max_budget_fraction: 0.6
";

#[test]
fn metaroom_canary_yaml_resolves_all_three_governor_keys() {
    let preset: PresetConfig = serde_yaml::from_str(CANARY).expect("canary yaml must parse");
    assert_eq!(preset.bab.mip.leaf_undecided_streak_cap, Some(1));
    assert_eq!(
        preset.bab.mip.escalation_skips_certified_clauses,
        Some(true)
    );
    assert_eq!(preset.bab.phase_budget.post_bab_pgd_fraction, Some(0.0));
    assert_eq!(preset.cz_hybrid.max_budget_fraction, Some(0.6));
    validate_preset(&preset).expect("the canary must pass semantic validation");
}

/// An ABSENT section is the shipped state and must leave every key unset —
/// this is one half of the dark proof.
#[test]
fn an_absent_mip_section_leaves_every_governor_key_unset() {
    let preset: PresetConfig = serde_yaml::from_str("bab:\n  batch_size: 8\n")
        .expect("a preset without the section must still parse");
    assert_eq!(preset.bab.mip.leaf_undecided_streak_cap, None);
    assert_eq!(preset.bab.mip.escalation_skips_certified_clauses, None);
    assert_eq!(preset.cz_hybrid.max_budget_fraction, None);
    validate_preset(&preset).expect("an absent section must validate");
}

/// The typed route must be the route: with NO environment value present, the
/// resolved source is `Config`, never `LegacyEnv`. Resolving against an
/// explicit empty lookup is what makes this a statement about the yaml.
#[test]
fn the_governor_keys_resolve_from_config_not_from_legacy_env() {
    let cap = read_over_config_with(
        &ny_levers::decls::graph_mip::MIP_LEAF_UNDECIDED_STREAK_CAP,
        |_| None,
        Some(LeverValue::U64(3)),
    )
    .expect("streak cap must resolve");
    assert_eq!(cap.source, Source::Config);
    assert_eq!(cap.value, LeverValue::U64(3));

    let skip = read_over_config_with(
        &ny_levers::decls::graph_mip::MIP_ESCALATE_SKIP_CERTIFIED,
        |_| None,
        Some(LeverValue::Bool(true)),
    )
    .expect("escalation skip must resolve");
    assert_eq!(skip.source, Source::Config);
    assert_eq!(skip.value, LeverValue::Bool(true));
}

/// With neither environment nor preset, both levers resolve to the SHIPPED
/// behaviour — cap 0 (no cap) and skip false (all clauses). The other half of
/// the dark proof.
#[test]
fn unset_governor_keys_resolve_to_the_shipped_behaviour() {
    let cap = read_over_config_with(
        &ny_levers::decls::graph_mip::MIP_LEAF_UNDECIDED_STREAK_CAP,
        |_| None,
        None,
    )
    .expect("streak cap must resolve");
    assert_eq!(cap.source, Source::Default);
    assert_eq!(cap.value, LeverValue::U64(0));

    let skip = read_over_config_with(
        &ny_levers::decls::graph_mip::MIP_ESCALATE_SKIP_CERTIFIED,
        |_| None,
        None,
    )
    .expect("escalation skip must resolve");
    assert_eq!(skip.source, Source::Default);
    assert_eq!(skip.value, LeverValue::Bool(false));
}

/// `cz_hybrid.max_budget_fraction` has no environment twin and therefore no
/// generic axis validator; it must still be REFUSED rather than clamped, so a
/// preset cannot claim a share it does not get.
#[test]
fn cz_hybrid_max_budget_fraction_is_refused_outside_the_unit_interval() {
    for bad in ["0.0", "-0.5", "1.5", ".nan", ".inf"] {
        let yaml = format!("cz_hybrid:\n  max_budget_fraction: {bad}\n");
        let preset: PresetConfig =
            serde_yaml::from_str(&yaml).expect("an out-of-range share is still valid yaml");
        let err =
            validate_preset(&preset).expect_err("an out-of-range cz_hybrid share must be refused");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("cz_hybrid.max_budget_fraction"),
            "the refusal must name the key; got {rendered}"
        );
    }
    // The endpoint 1.0 IS admissible: it is exactly today's behaviour.
    let preset: PresetConfig =
        serde_yaml::from_str("cz_hybrid:\n  max_budget_fraction: 1.0\n").expect("parse");
    validate_preset(&preset).expect("1.0 is today's behaviour and must be accepted");
}
