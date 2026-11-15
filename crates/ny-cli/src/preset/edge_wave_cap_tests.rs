// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #iso-edge-cap delivery: `bab.branching.input_split.edge_wave_cap` parses,
//! reaches `BetaCrownConfig` through `apply_preset`, ships dark, and is armed
//! by NO shipped yaml (arming lands only in the commit that carries its
//! measurement).

use std::path::Path;

use super::*;
use ny_propagate::BetaCrownConfig;

const ARMED_EMPTY: &str = "bab:\n  branching:\n    input_split:\n      edge_wave_cap: {}\n";

/// Absent: the engine default stays dark (`None` = ungoverned edge work).
#[test]
fn edge_wave_cap_absent_ships_dark() {
    let preset = PresetConfig::default();
    assert!(preset.bab.branching.input_split.edge_wave_cap.is_none());
    let mut config = BetaCrownConfig::default();
    assert_eq!(config.input_split_edge_wave_cap, None, "engine default");
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.input_split_edge_wave_cap, None,
        "an absent key must leave the edge work ungoverned"
    );
}

/// Present, even EMPTY, arms it: unnamed fields stay unnamed so the lane's
/// own defaults (2.0 s, 0.25) govern.
#[test]
fn edge_wave_cap_empty_block_arms_with_lane_defaults() {
    let preset: PresetConfig = serde_yaml::from_str(ARMED_EMPTY).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    let cap = config
        .input_split_edge_wave_cap
        .expect("an empty block still arms — presence is the switch");
    assert_eq!(cap.max_secs, None);
    assert_eq!(cap.rebound_share, None);
}

/// Named fields flow through unchanged.
#[test]
fn edge_wave_cap_named_fields_flow_to_config() {
    let text = "bab:\n  branching:\n    input_split:\n      edge_wave_cap:\n        \
                max_secs: 0.75\n        rebound_share: 0.5\n";
    let preset: PresetConfig = serde_yaml::from_str(text).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    let cap = config.input_split_edge_wave_cap.expect("armed");
    assert_eq!(cap.max_secs, Some(0.75));
    assert_eq!(cap.rebound_share, Some(0.5));
    // The one mapping shared with the relational route agrees with `apply`.
    assert_eq!(
        preset
            .bab
            .branching
            .input_split
            .edge_wave_cap
            .as_ref()
            .map(InputSplitEdgeWaveCapPreset::to_config),
        Some(cap)
    );
}

/// A misspelled sub-key is a load error, never a silently dark key.
#[test]
fn edge_wave_cap_rejects_unknown_fields() {
    let text = "bab:\n  branching:\n    input_split:\n      edge_wave_cap:\n        max_sec: 2.0\n";
    assert!(serde_yaml::from_str::<PresetConfig>(text).is_err());
}

/// PROMOTION IS NOT ARMING: no shipped yaml names the key (comments ignored).
#[test]
fn no_shipped_preset_arms_the_edge_wave_cap() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut scanned = 0usize;
    for dir in ["configs/vnncomp25", "configs/vnncomp26"] {
        let dir = repo_root.join(dir);
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
                    !live.contains("edge_wave_cap"),
                    "{}:{} arms #iso-edge-cap; arming belongs to the commit that \
                     carries its measurement",
                    path.display(),
                    line_no + 1
                );
            }
        }
    }
    assert!(scanned > 0, "the scan read no yaml: VOID");
}
