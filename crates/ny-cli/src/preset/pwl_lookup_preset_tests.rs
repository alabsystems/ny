// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #pwl-lookup: typed delivery of `bab.pwl_lookup_exact_range`, and its
//! dark-by-default state across every shipped preset.

use std::fs;
use std::path::Path;

use ny_propagate::BetaCrownConfig;

use super::apply::apply_preset;
use super::{load_preset, PresetConfig};

/// The key must reach `BetaCrownConfig` FROM A YAML (the scored route;
/// `BabPreset` is `deny_unknown_fields`, so a misspelt key fails loudly), both
/// polarities must arrive, and an omitted key must leave the default (dark).
#[test]
fn apply_preset_propagates_pwl_lookup_exact_range_and_defaults_dark() {
    let armed: PresetConfig = serde_yaml::from_str("bab:\n  pwl_lookup_exact_range: true\n")
        .expect("bab.pwl_lookup_exact_range should parse");
    assert_eq!(armed.bab.pwl_lookup_exact_range, Some(true));
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed preset should apply cleanly");
    assert!(config.pwl_lookup_exact_range);

    let disarmed: PresetConfig = serde_yaml::from_str("bab:\n  pwl_lookup_exact_range: false\n")
        .expect("an explicit false should parse");
    let mut config = BetaCrownConfig {
        pwl_lookup_exact_range: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut config, &disarmed).expect("disarmed preset should apply cleanly");
    assert!(
        !config.pwl_lookup_exact_range,
        "an explicit false must reach the config"
    );

    let omitted = PresetConfig::default();
    assert_eq!(omitted.bab.pwl_lookup_exact_range, None);
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &omitted).expect("omitted key should apply cleanly");
    assert!(
        !config.pwl_lookup_exact_range,
        "a preset that omits the key must leave the capability dark"
    );

    assert!(
        serde_yaml::from_str::<PresetConfig>("bab:\n  pwl_lookup_exact_rang: true\n").is_err(),
        "a misspelt key must be refused, not silently ignored"
    );
}

/// ARMED ONLY WHERE MEASURED: `vnncomp25/ml4acopf_2024.yaml` arms
/// `bab.pwl_lookup_exact_range: true` (measured 2026-09-26 on
/// ny_wave6_cf376ec20: the 12 `-linear-*` prop3/prop4 rows unsat in 1.0-4.1 s,
/// contradiction=0); no other shipped preset in any year may set it, and the
/// effective ml4acopf_2024 preset must resolve it armed.
#[test]
fn pwl_lookup_exact_range_stays_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    let mut configured = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let Ok(entries) = fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            scanned += 1;
            let is_ml4acopf = year == "vnncomp25"
                && path.file_name().and_then(|name| name.to_str()) == Some("ml4acopf_2024.yaml");
            if preset.bab.pwl_lookup_exact_range.is_some() && !is_ml4acopf {
                configured.push(path);
            }
        }
    }
    assert!(
        scanned > 0,
        "VOID: no shipped preset was scanned under {}",
        root.display()
    );
    assert!(
        configured.is_empty(),
        "bab.pwl_lookup_exact_range is set outside vnncomp25/ml4acopf_2024.yaml, where it \
         was never measured: {configured:?}"
    );

    let effective = crate::commands::vnncomp::resolve_preset_path(&root, "ml4acopf_2024")
        .expect("ml4acopf_2024 must resolve to a shipped preset");
    let preset = load_preset(&effective).expect("effective ml4acopf preset loads");
    assert_eq!(
        preset.bab.pwl_lookup_exact_range,
        Some(true),
        "the effective ml4acopf preset ({}) must arm the measured key",
        effective.display()
    );
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("effective ml4acopf preset applies");
    assert!(
        config.pwl_lookup_exact_range,
        "the armed key must reach BetaCrownConfig"
    );
}
