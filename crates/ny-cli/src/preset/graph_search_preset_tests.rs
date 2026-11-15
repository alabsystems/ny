// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-lane-search (ENGINE_PROGRAM §2 M7): typed delivery of
//! `bab.graph_search`, and its dark-by-default state across every shipped
//! preset.

use std::path::Path;

use ny_propagate::{BetaCrownConfig, GraphLaneSearch};

use super::apply::apply_preset;
use super::PresetConfig;

/// `bab.graph_search` must reach `BetaCrownConfig` FROM A YAML (the scored
/// route), accept exactly `off | cpu_f32`, refuse misspellings (`BabPreset` is
/// `deny_unknown_fields` and the enum is closed), and be dark when omitted.
#[test]
fn graph_lane_search_key_reaches_the_beta_config_and_defaults_dark() {
    let armed: PresetConfig = serde_yaml::from_str("bab:\n  graph_search: cpu_f32\n")
        .expect("bab.graph_search: cpu_f32 should parse");
    assert_eq!(armed.bab.graph_search, Some(GraphLaneSearch::CpuF32));
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed preset should apply cleanly");
    assert_eq!(config.graph_search, GraphLaneSearch::CpuF32);
    assert!(config.graph_search.is_armed());

    let off: PresetConfig =
        serde_yaml::from_str("bab:\n  graph_search: off\n").expect("explicit off should parse");
    assert_eq!(off.bab.graph_search, Some(GraphLaneSearch::Off));
    let mut config = BetaCrownConfig {
        graph_search: GraphLaneSearch::CpuF32,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut config, &off).expect("explicit off should apply cleanly");
    assert_eq!(config.graph_search, GraphLaneSearch::Off);

    for bad in [
        "bab:\n  graph_search: CpuF32\n",
        "bab:\n  graph_search: cpu\n",
        "bab:\n  graph_search: cuda\n",
        "bab:\n  graph_search: true\n",
        "bab:\n  graph_serch: cpu_f32\n",
    ] {
        assert!(
            serde_yaml::from_str::<PresetConfig>(bad).is_err(),
            "`{bad}` must not parse"
        );
    }
    for token in GraphLaneSearch::TOKENS {
        let yaml = format!("bab:\n  graph_search: {token}\n");
        let parsed: PresetConfig = serde_yaml::from_str(&yaml).expect("declared token parses");
        assert_eq!(
            parsed.bab.graph_search.map(GraphLaneSearch::label),
            Some(token),
            "the token round-trips through its label"
        );
    }

    let omitted = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &omitted).expect("omitted key should apply cleanly");
    assert_eq!(
        config.graph_search,
        GraphLaneSearch::Off,
        "a preset that omits the key must leave the graph lane exactly as it was"
    );
    assert_eq!(
        BetaCrownConfig::default().graph_search,
        GraphLaneSearch::Off
    );
}

/// DARK-BY-DEFAULT PIN: no shipped yaml sets `bab.graph_search`. Arming it is
/// a measured decision (M7's measurement: cora / malbeware root-decided, the
/// relusplitter root bound) that belongs in its own commit, and that commit
/// must edit THIS test, naming the categories it arms.
#[test]
fn no_shipped_preset_arms_the_graph_search_key() {
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
            if preset.bab.graph_search.is_some() {
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
        "bab.graph_search is dark until measured; armed in: {armed:?}"
    );
}
