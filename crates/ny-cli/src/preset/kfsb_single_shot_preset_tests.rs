// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #kfsb-single-shot: typed delivery of `bab.kfsb_single_shot`, and its
//! dark-by-default state across every shipped preset.

use std::path::Path;

use ny_propagate::BetaCrownConfig;

use super::apply::apply_preset;
use super::PresetConfig;

/// `bab.kfsb_single_shot` must reach `BetaCrownConfig` FROM A YAML (the scored
/// route), accept only booleans (`BabPreset` is `deny_unknown_fields`), be
/// overridable back to `false`, and be dark when omitted.
#[test]
fn kfsb_single_shot_key_reaches_the_beta_config_and_defaults_dark() {
    let armed: PresetConfig = serde_yaml::from_str("bab:\n  kfsb_single_shot: true\n")
        .expect("bab.kfsb_single_shot: true should parse");
    assert_eq!(armed.bab.kfsb_single_shot, Some(true));
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed preset should apply cleanly");
    assert!(config.kfsb_single_shot);

    let off: PresetConfig = serde_yaml::from_str("bab:\n  kfsb_single_shot: false\n")
        .expect("explicit false should parse");
    let mut config = BetaCrownConfig {
        kfsb_single_shot: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut config, &off).expect("explicit false should apply cleanly");
    assert!(!config.kfsb_single_shot);

    for bad in [
        "bab:\n  kfsb_single_shot: fast\n",
        "bab:\n  kfsb_single_shot: 2\n",
        "bab:\n  kfsb_single_shoot: true\n",
    ] {
        assert!(
            serde_yaml::from_str::<PresetConfig>(bad).is_err(),
            "`{bad}` must not parse"
        );
    }

    let omitted = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &omitted).expect("omitted key should apply cleanly");
    assert!(
        !config.kfsb_single_shot,
        "a preset that omits the key must leave the kFSB scorer exactly as it was"
    );
    assert!(!BetaCrownConfig::default().kfsb_single_shot);
}

/// DARK-BY-DEFAULT PIN: no shipped yaml sets `bab.kfsb_single_shot`. Arming it
/// is a measured decision (graph-lane throughput and a both-year moat) that
/// belongs in its own commit, and that commit must edit THIS test, naming the
/// categories it arms.
#[test]
fn no_shipped_preset_arms_the_kfsb_single_shot_key() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut read = 0usize;
    let mut armed = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let Ok(entries) = std::fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            read += 1;
            if preset.bab.kfsb_single_shot.is_some() {
                armed.push(path.display().to_string());
            }
        }
    }
    assert!(
        read > 40,
        "expected to read the shipped presets, read {read} (positive control)"
    );
    assert!(
        armed.is_empty(),
        "bab.kfsb_single_shot is dark until measured; armed in: {armed:?}"
    );
}
