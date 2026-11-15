// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-concurrent: typed delivery of `attack.concurrent`, its dark
//! default, and the measured isomorphic-only shipped policy.

use std::path::Path;

use ny_propagate::BetaCrownConfig;

use super::apply::apply_preset;
use super::PresetConfig;

/// DARK BY DEFAULT: the key's default value is `false`, and a preset that
/// omits it — or names other attack keys — leaves it `false`; the race sites
/// read this one bool and run today's serial code (the site-level half of the
/// bit-identity argument is `attack_concurrent_tests::key_off_runs_no_race`).
#[test]
fn attack_concurrent_is_dark_by_default() {
    assert!(!BetaCrownConfig::default().attack_concurrent);
    let omitted = PresetConfig::default();
    assert_eq!(omitted.attack.concurrent, None);
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &omitted).expect("omitted key should apply cleanly");
    assert!(!config.attack_concurrent);

    // An attack section that does not name the key leaves it dark.
    let other_attack_key: PresetConfig =
        serde_yaml::from_str("attack:\n  pgd_restarts: 7\n").expect("parse");
    let mut with_section = BetaCrownConfig::default();
    apply_preset(&mut with_section, &other_attack_key).expect("apply");
    assert!(!with_section.attack_concurrent);
    assert_eq!(
        with_section.pgd_restarts, 7,
        "positive control: the section applied"
    );
}

/// REACHABILITY FROM THE YAML (the scored route): `attack.concurrent: true`
/// reaches `BetaCrownConfig::attack_concurrent` through `apply_preset` with no
/// environment variable involved (the key has no environment twin at all),
/// `false` overrides it back, and only booleans parse
/// (`AttackPreset` is `deny_unknown_fields`).
#[test]
fn attack_concurrent_reaches_the_sites_from_the_yaml() {
    let armed: PresetConfig = serde_yaml::from_str("attack:\n  concurrent: true\n")
        .expect("attack.concurrent: true should parse");
    assert_eq!(armed.attack.concurrent, Some(true));
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed preset should apply cleanly");
    assert!(
        config.attack_concurrent,
        "the yaml must arm the race sites' key"
    );

    let off: PresetConfig =
        serde_yaml::from_str("attack:\n  concurrent: false\n").expect("explicit false parses");
    let mut config = BetaCrownConfig {
        attack_concurrent: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut config, &off).expect("explicit false should apply cleanly");
    assert!(!config.attack_concurrent);

    for bad in [
        "attack:\n  concurrent: fast\n",
        "attack:\n  concurrent: 2\n",
        "attack:\n  concurent: true\n",
    ] {
        assert!(
            serde_yaml::from_str::<PresetConfig>(bad).is_err(),
            "`{bad}` must not parse"
        );
    }

    // No environment twin: the race module reads no environment variable at
    // all, so a yaml is the only way in. (The workspace-wide half is the
    // ny-levers raw-read ratchet, which this key does not move.)
    let race_module = include_str!("../commands/beta_crown/attack_concurrent.rs");
    assert!(
        race_module.contains("pub(crate) fn run_race<A, P>("),
        "positive control: the race module was read"
    );
    for read in ["env::var", "var_os("] {
        assert!(
            !race_module.contains(read),
            "attack.concurrent must stay a typed key with no environment read (`{read}`)"
        );
    }
}

/// Measured promotion: only the qualified isomorphic category may request the
/// worker. All other shipped presets keep the field absent; global defaults
/// and explicit-off semantics are covered independently above.
#[test]
fn only_qualified_isomorphic_preset_arms_attack_concurrent() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut read = 0usize;
    let mut armed = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let entries = std::fs::read_dir(root.join(year)).expect("shipped preset year directory");
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            read += 1;
            if preset.attack.concurrent.is_some() {
                armed.push((
                    path.strip_prefix(&root).unwrap().to_path_buf(),
                    preset.attack.concurrent,
                ));
            }
        }
    }
    assert!(
        read > 40,
        "expected to read the shipped presets, read {read} (positive control)"
    );
    assert_eq!(
        armed,
        vec![(
            Path::new("vnncomp26/isomorphic_acasxu_2026.yaml").to_path_buf(),
            Some(true)
        )],
        "only the measured isomorphic policy may request this worker"
    );
}
