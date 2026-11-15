// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-open-disjuncts delivery: the six `attack.open_disjunct_*` keys
//! parse, reach `BetaCrownConfig::open_disjunct_attack` through
//! `apply_preset` (the only route — the stage has no environment spelling),
//! ship dark, reject out-of-range values, and are armed by NO shipped yaml
//! (arming lands only in the commit that carries its measurement).

use std::path::Path;

use super::*;
use ny_propagate::{
    BetaCrownConfig, OpenDisjunctAttackConfig, OPEN_DISJUNCT_DEFAULT_MAX_FRAC,
    OPEN_DISJUNCT_DEFAULT_MAX_SECS, OPEN_DISJUNCT_DEFAULT_MIN_SECS, OPEN_DISJUNCT_DEFAULT_RESTARTS,
};

const ARMED: &str = "attack:\n  open_disjunct_restrict: true\n  open_disjunct_min_secs: 0.25\n  \
                     open_disjunct_max_secs: 8.0\n  open_disjunct_restarts: 32\n  \
                     open_disjunct_root_wait_secs: 45.0\n  open_disjunct_max_frac: 0.5\n";

/// Plan test name: the preset value arrives as a TYPED config field, carried
/// by the preset (not an environment layer — none exists for this stage).
#[test]
fn attack_open_preset_source_is_typed() {
    let preset: PresetConfig = serde_yaml::from_str(ARMED).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.open_disjunct_attack,
        OpenDisjunctAttackConfig {
            restrict: true,
            min_secs: 0.25,
            max_secs: 8.0,
            restarts: 32,
            root_wait_secs: 45.0,
            max_frac: 0.5,
        }
    );
    // The mapping vnncomp arms the lane's root publication from is the SAME
    // one `apply` wrote.
    assert_eq!(
        open_disjunct_attack_from_preset(&preset).unwrap(),
        Some(config.open_disjunct_attack)
    );
    // A lone switch takes the engine defaults for every other field.
    let preset: PresetConfig =
        serde_yaml::from_str("attack:\n  open_disjunct_restrict: true\n").unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.open_disjunct_attack,
        OpenDisjunctAttackConfig {
            restrict: true,
            min_secs: OPEN_DISJUNCT_DEFAULT_MIN_SECS,
            max_secs: OPEN_DISJUNCT_DEFAULT_MAX_SECS,
            restarts: OPEN_DISJUNCT_DEFAULT_RESTARTS,
            root_wait_secs: 0.0,
            max_frac: OPEN_DISJUNCT_DEFAULT_MAX_FRAC,
        }
    );
}

/// Absent keys: the engine default (`restrict: false`) stands, bit for bit.
#[test]
fn attack_open_absent_ships_dark() {
    let preset = PresetConfig::default();
    assert_eq!(open_disjunct_attack_from_preset(&preset).unwrap(), None);
    let mut config = BetaCrownConfig::default();
    assert!(
        !config.open_disjunct_attack.restrict,
        "engine default is dark"
    );
    let before = config.open_disjunct_attack;
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.open_disjunct_attack, before);
    assert_eq!(before, OpenDisjunctAttackConfig::default());
    // Tuning keys without the switch configure a stage that still does not
    // exist.
    let preset: PresetConfig =
        serde_yaml::from_str("attack:\n  open_disjunct_min_secs: 1.0\n").unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(!config.open_disjunct_attack.restrict);
    assert_eq!(config.open_disjunct_attack.min_secs, 1.0);
}

/// An out-of-range value is a load error, never a silently dark stage.
#[test]
fn attack_open_rejects_out_of_range_values() {
    for bad in [
        "open_disjunct_min_secs: 0.0",
        "open_disjunct_min_secs: -1.0",
        "open_disjunct_max_secs: -0.5",
        "open_disjunct_max_secs: .nan",
        "open_disjunct_root_wait_secs: .inf",
        "open_disjunct_root_wait_secs: 4000.0",
        "open_disjunct_restarts: 0",
        "open_disjunct_max_frac: 0.0",
        "open_disjunct_max_frac: 1.5",
        "open_disjunct_max_frac: .nan",
    ] {
        let text = format!("attack:\n  open_disjunct_restrict: true\n  {bad}\n");
        let preset: PresetConfig = serde_yaml::from_str(&text).unwrap();
        let mut config = BetaCrownConfig::default();
        assert!(
            apply_preset(&mut config, &preset).is_err(),
            "`{bad}` must fail the preset load"
        );
        assert!(open_disjunct_attack_from_preset(&preset).is_err());
    }
}

/// A misspelled key is a load error (`AttackPreset` denies unknown fields).
#[test]
fn attack_open_rejects_unknown_keys() {
    let text = "attack:\n  open_disjunct_restric: true\n";
    assert!(serde_yaml::from_str::<PresetConfig>(text).is_err());
}

/// PROMOTION IS NOT ARMING: no shipped yaml names any of the keys (comments
/// ignored). Every `configs/vnncomp*` year is scanned, as the sibling
/// `#attack-vertex-geometry` pin does, so a year added later is covered too.
#[test]
fn no_shipped_preset_arms_attack_open_disjuncts() {
    let configs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    let mut years = 0usize;
    let listing = fs::read_dir(&configs).unwrap_or_else(|e| panic!("{}: {e}", configs.display()));
    for year in listing.flatten() {
        let dir = year.path();
        let is_vnncomp = dir
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("vnncomp"));
        if !is_vnncomp || !dir.is_dir() {
            continue;
        }
        years += 1;
        let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).unwrap();
            scanned += 1;
            for (line_no, line) in text.lines().enumerate() {
                let live = line.split('#').next().unwrap_or("");
                assert!(
                    !live.contains("open_disjunct_"),
                    "{}:{} arms #attack-open-disjuncts; arming belongs to the commit that \
                     carries its measurement",
                    path.display(),
                    line_no + 1
                );
            }
        }
    }
    assert!(
        years >= 2 && scanned > 0,
        "the scan read {years} config years / {scanned} yaml: VOID"
    );
}
