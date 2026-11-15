// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane stage 2 — REACHABILITY of `bab.branching.input_split.
//! lowdim_joint_quadratic`: the typed key must arrive through the SAME
//! projection a scored run uses (`preset_delivered_config_value` over the
//! serialized preset, with NO environment), with `Source::Config`, and reach
//! the lane's config (`BetaCrownConfig::lowdim_config`). The resolver's own
//! observable is `[lowdim-joint] armed=1 source=Config`.
//!
//! The same projection test covers stage 1's
//! `bab.branching.input_split.lowdim_interval_twin` (the key without which
//! the lane cannot bound `lsnc_relu` at all) and stage 3's search keys.

use super::preset_delivered_config_value;

fn projected_levers(preset: &crate::preset::PresetConfig) -> ny_levers::LeverSet {
    let serialized = serde_json::to_value(preset).expect("preset serializes");
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
    ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        preset_delivered_config_value(&serialized, decl)
    })
    .expect("the projected preset must match its declarations")
}

/// Absent the key, the joint bound resolves dark on the scored projection.
#[test]
fn the_joint_quadratic_key_is_dark_without_the_key() {
    let levers = projected_levers(&crate::preset::PresetConfig::default());
    let resolved = levers
        .resolved(&ny_levers::decls::lowdim_lane::LOWDIM_JOINT_QUADRATIC)
        .expect("the lever is declared");
    assert!(!resolved.value.as_bool());
    assert_eq!(resolved.source, ny_levers::Source::Default);
    assert!(!crate::commands::beta_crown::resolve_lowdim_joint_quadratic(Some(&levers), None));
}

/// The typed key arms it through the scored projection with
/// `Source::Config` — never a legacy env read — and the resolver the CLI uses
/// turns that into the lane's `LowDimConfig::joint_quadratic`.
#[test]
fn the_joint_quadratic_key_arrives_through_the_scored_projection() {
    let mut preset = crate::preset::PresetConfig::default();
    preset.bab.branching.input_split.lowdim_joint_quadratic = Some(true);
    let levers = projected_levers(&preset);
    let resolved = levers
        .resolved(&ny_levers::decls::lowdim_lane::LOWDIM_JOINT_QUADRATIC)
        .expect("the lever is declared");
    assert!(resolved.value.as_bool());
    assert_eq!(resolved.source, ny_levers::Source::Config);
    assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
    let armed = crate::commands::beta_crown::resolve_lowdim_joint_quadratic(Some(&levers), None);
    assert!(armed);
    let config = ny_propagate::BetaCrownConfig {
        input_split_lowdim_joint_quadratic: armed,
        ..Default::default()
    };
    assert!(config.lowdim_config().joint_quadratic);
}

/// ONLY the shipped lsnc_relu preset arms it and stage 1's interval twin (armed 2026-09-26 on the two-year
/// moat: 2026 79/80, 2025 80/80, 0 lost, 0 flips); no other shipped yaml may. Non-vacuity: a scan that found
/// no yaml (a wrong root) fails, and the lsnc file must actually carry both keys.
#[test]
fn no_shipped_preset_arms_the_joint_quadratic_key() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    let mut lsnc_hits = 0usize;
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            scanned += 1;
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let is_lsnc = year == "vnncomp25"
                && path.file_name().and_then(|n| n.to_str()) == Some("lsnc_relu.yaml");
            for key in ["lowdim_joint_quadratic", "lowdim_interval_twin"] {
                if is_lsnc {
                    lsnc_hits += usize::from(text.contains(&format!("{key}: true")));
                } else {
                    assert!(
                        !text.contains(key),
                        "{} arms the #lsnc-lane key {key}; only vnncomp25/lsnc_relu.yaml may",
                        path.display()
                    );
                }
            }
        }
    }
    assert!(scanned > 0, "VOID: no shipped yaml was scanned");
    assert_eq!(
        lsnc_hits, 2,
        "vnncomp25/lsnc_relu.yaml must arm both lane keys"
    );
}

// ---------------------------------------------------------------------------
// #lsnc-lane stage 1: lowdim_interval_twin
// ---------------------------------------------------------------------------

/// Absent the key, the twin resolves dark on the scored projection, and the
/// resolver the CLI calls before attaching a twin agrees.
#[test]
fn the_interval_twin_key_is_dark_without_the_key() {
    let levers = projected_levers(&crate::preset::PresetConfig::default());
    let resolved = levers
        .resolved(&ny_levers::decls::lowdim_lane::LOWDIM_INTERVAL_TWIN)
        .expect("the lever is declared");
    assert!(!resolved.value.as_bool());
    assert_eq!(resolved.source, ny_levers::Source::Default);
    assert_eq!(
        crate::commands::beta_crown::resolve_lowdim_interval_twin(Some(&levers), None),
        (false, ny_levers::Source::Default)
    );
}

/// The typed key arms the twin through the SAME projection a scored run uses
/// (the `space.rs` preset-key path over the serialized preset, no
/// environment), with `Source::Config` — never a legacy env read — and the
/// CLI resolver (`attach_lowdim_interval_twin`'s input) returns it armed. A
/// mistyped `Deliver::PresetKey` path would leave the lane unable to bound
/// `lsnc_relu` in competition while every resolver-level test stayed green.
#[test]
fn the_interval_twin_key_arrives_through_the_scored_projection() {
    let mut preset = crate::preset::PresetConfig::default();
    preset.bab.branching.input_split.lowdim_interval_twin = Some(true);
    let levers = projected_levers(&preset);
    let resolved = levers
        .resolved(&ny_levers::decls::lowdim_lane::LOWDIM_INTERVAL_TWIN)
        .expect("the lever is declared");
    assert!(resolved.value.as_bool());
    assert_eq!(resolved.source, ny_levers::Source::Config);
    assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
    assert_eq!(
        crate::commands::beta_crown::resolve_lowdim_interval_twin(Some(&levers), None),
        (true, ny_levers::Source::Config)
    );
}

// ---------------------------------------------------------------------------
// #lsnc-lane stage 3: lowdim_split / lowdim_clip / lowdim_parallel
// ---------------------------------------------------------------------------

/// Absent the keys, the scored projection resolves the historical lane:
/// `widest`, no clip, the `Bound` wave mode — from `Source::Default`.
#[test]
fn the_stage3_search_keys_resolve_to_the_historical_lane_without_the_keys() {
    let levers = projected_levers(&crate::preset::PresetConfig::default());
    let split = levers
        .resolved(&ny_levers::decls::lowdim_lane::LOWDIM_SPLIT)
        .expect("the split lever is declared");
    assert_eq!(split.value.as_str(), Some("widest"));
    assert_eq!(split.source, ny_levers::Source::Default);
    for decl in [
        &ny_levers::decls::lowdim_lane::LOWDIM_CLIP,
        &ny_levers::decls::lowdim_lane::LOWDIM_PARALLEL,
    ] {
        let resolved = levers.resolved(decl).expect("the lever is declared");
        assert!(!resolved.value.as_bool(), "{} must resolve dark", decl.name);
        assert_eq!(resolved.source, ny_levers::Source::Default);
    }
    let config = ny_propagate::BetaCrownConfig {
        input_split_lowdim_split: crate::commands::beta_crown::resolve_lowdim_split(
            Some(&levers),
            None,
        ),
        input_split_lowdim_clip: crate::commands::beta_crown::resolve_lowdim_clip(
            Some(&levers),
            None,
        ),
        input_split_lowdim_parallel: crate::commands::beta_crown::resolve_lowdim_parallel(
            Some(&levers),
            None,
        ),
        ..Default::default()
    };
    let lane = config.lowdim_config();
    assert_eq!(lane.split, ny_propagate::lowdim::LowDimSplit::Widest);
    assert!(!lane.clip);
    assert_eq!(lane.parallel, ny_propagate::lowdim::LowDimParallel::Bound);
}

/// Each typed key arrives through the scored projection with `Source::Config`
/// — never a legacy env read — and the resolvers the CLI uses turn it into the
/// lane's `LowDimConfig` (`split`, `clip`, `parallel = Full`). Every split
/// token is exercised.
#[test]
fn the_stage3_search_keys_arrive_through_the_scored_projection() {
    use ny_propagate::lowdim::{LowDimParallel, LowDimSplit};
    for split in [
        LowDimSplit::Widest,
        LowDimSplit::Gradient,
        LowDimSplit::SbSum,
        LowDimSplit::SbWr,
    ] {
        let mut preset = crate::preset::PresetConfig::default();
        preset.bab.branching.input_split.lowdim_split = Some(split);
        preset.bab.branching.input_split.lowdim_clip = Some(true);
        preset.bab.branching.input_split.lowdim_parallel = Some(true);
        let levers = projected_levers(&preset);
        let resolved = levers
            .resolved(&ny_levers::decls::lowdim_lane::LOWDIM_SPLIT)
            .expect("declared");
        assert_eq!(resolved.value.as_str(), Some(split.token()));
        assert_eq!(resolved.source, ny_levers::Source::Config);
        for decl in [
            &ny_levers::decls::lowdim_lane::LOWDIM_CLIP,
            &ny_levers::decls::lowdim_lane::LOWDIM_PARALLEL,
        ] {
            let resolved = levers.resolved(decl).expect("declared");
            assert!(resolved.value.as_bool(), "{} must arrive armed", decl.name);
            assert_eq!(resolved.source, ny_levers::Source::Config);
            assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
        }
        let config = ny_propagate::BetaCrownConfig {
            input_split_lowdim_split: crate::commands::beta_crown::resolve_lowdim_split(
                Some(&levers),
                None,
            ),
            input_split_lowdim_clip: crate::commands::beta_crown::resolve_lowdim_clip(
                Some(&levers),
                None,
            ),
            input_split_lowdim_parallel: crate::commands::beta_crown::resolve_lowdim_parallel(
                Some(&levers),
                None,
            ),
            ..Default::default()
        };
        let lane = config.lowdim_config();
        assert_eq!(lane.split, split);
        assert!(lane.clip);
        assert_eq!(lane.parallel, LowDimParallel::Full);
    }
}

/// The yaml spelling of each token parses into the typed key (the serde
/// tokens are the lever's enum members), and an unknown token is REFUSED at
/// parse time rather than silently resolving to the default.
#[test]
fn the_split_key_parses_its_yaml_tokens_and_refuses_others() {
    use ny_propagate::lowdim::LowDimSplit;
    for (token, split) in [
        ("widest", LowDimSplit::Widest),
        ("gradient", LowDimSplit::Gradient),
        ("sb", LowDimSplit::SbSum),
        ("sbwr", LowDimSplit::SbWr),
    ] {
        let parsed: crate::preset::InputSplitPreset =
            serde_yaml::from_str(&format!("lowdim_split: {token}")).expect("parses");
        assert_eq!(parsed.lowdim_split, Some(split));
        assert_eq!(LowDimSplit::from_token(token), Some(split));
        assert_eq!(split.token(), token);
    }
    assert!(
        serde_yaml::from_str::<crate::preset::InputSplitPreset>("lowdim_split: sb_sum").is_err(),
        "an unknown split token must fail the preset parse"
    );
}

/// The stage-3 search keys are armed ONLY where each was measured, and every
/// listed preset must carry all three measured values; every other shipped
/// preset stays dark. lsnc_relu: the two-year lsnc moat (2026 79/80, 2025
/// 80/80, 0 lost). cersyve + acasxu_2023: scratchpad/w6q M2 (2026-09-26) —
/// cersyve cart_pole_finetune_inv timeout 3/3 -> unsat 3/3; acasxu 1_1 p5,
/// 4_2 p2, 4_5 p10 timeout 3/3 -> unsat 3/3.
#[test]
fn no_shipped_preset_arms_the_stage3_search_keys() {
    const ARMED: [&str; 3] = ["lsnc_relu.yaml", "cersyve.yaml", "acasxu_2023.yaml"];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    let mut armed_hits = std::collections::BTreeMap::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let Ok(entries) = std::fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            scanned += 1;
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if year == "vnncomp25" && ARMED.contains(&name) {
                let hits = [
                    "lowdim_split: sbwr",
                    "lowdim_clip: true",
                    "lowdim_parallel: true",
                ]
                .iter()
                .filter(|armed| text.contains(*armed))
                .count();
                armed_hits.insert(name.to_string(), hits);
                continue;
            }
            for key in ["lowdim_split", "lowdim_clip", "lowdim_parallel"] {
                assert!(
                    !text.contains(key),
                    "{} arms the #lsnc-lane key {key}; only the measured presets {ARMED:?} may",
                    path.display()
                );
            }
        }
    }
    assert!(scanned > 0, "VOID: no shipped yaml was scanned");
    for name in ARMED {
        assert_eq!(
            armed_hits.get(name).copied(),
            Some(3),
            "vnncomp25/{name} must arm lowdim_split: sbwr, lowdim_clip and lowdim_parallel"
        );
    }
}

// ---------------------------------------------------------------------------
// #lowdim-bundle: lowdim_varying_dims / lowdim_alpha_iters /
// lowdim_leaf_attack_every / lowdim_shared_worklist
// ---------------------------------------------------------------------------

/// The four bundle levers, resolved into the lane config the way the CLI does.
fn bundle_lane(levers: &ny_levers::LeverSet) -> ny_propagate::lowdim::LowDimConfig {
    let config =
        ny_propagate::BetaCrownConfig {
            input_split_lowdim_varying_dims:
                crate::commands::beta_crown::resolve_lowdim_varying_dims(Some(levers), None),
            input_split_lowdim_alpha_iters: crate::commands::beta_crown::resolve_lowdim_alpha_iters(
                Some(levers),
                None,
            ),
            input_split_lowdim_leaf_attack_every:
                crate::commands::beta_crown::resolve_lowdim_leaf_attack_every(Some(levers), None),
            input_split_lowdim_shared_worklist:
                crate::commands::beta_crown::resolve_lowdim_shared_worklist(Some(levers), None),
            ..Default::default()
        };
    config.lowdim_config()
}

/// DARK BY DEFAULT: absent the keys, the scored projection resolves every
/// bundle lever at its declaration default (`Source::Default`) and the lane
/// config is the historical one.
#[test]
fn the_bundle_keys_are_dark_without_the_keys() {
    let levers = projected_levers(&crate::preset::PresetConfig::default());
    for decl in [
        &ny_levers::decls::lowdim_lane::LOWDIM_VARYING_DIMS,
        &ny_levers::decls::lowdim_lane::LOWDIM_SHARED_WORKLIST,
    ] {
        let resolved = levers.resolved(decl).expect("the lever is declared");
        assert!(!resolved.value.as_bool(), "{} must resolve dark", decl.name);
        assert_eq!(resolved.source, ny_levers::Source::Default);
    }
    for decl in [
        &ny_levers::decls::lowdim_lane::LOWDIM_ALPHA_ITERS,
        &ny_levers::decls::lowdim_lane::LOWDIM_LEAF_ATTACK_EVERY,
    ] {
        let resolved = levers.resolved(decl).expect("the lever is declared");
        assert_eq!(
            resolved.value.as_u64(),
            Some(0),
            "{} must resolve to 0",
            decl.name
        );
        assert_eq!(resolved.source, ny_levers::Source::Default);
    }
    let lane = bundle_lane(&levers);
    let historical = ny_propagate::lowdim::LowDimConfig::default();
    assert!(!lane.varying_dims);
    assert_eq!(lane.alpha_iters, 0);
    assert_eq!(lane.leaf_attack_every, 0);
    assert!(!lane.shared_worklist);
    assert_eq!(lane.varying_dims, historical.varying_dims);
    assert_eq!(lane.alpha_iters, historical.alpha_iters);
    assert_eq!(lane.leaf_attack_every, historical.leaf_attack_every);
    assert_eq!(lane.shared_worklist, historical.shared_worklist);
}

/// Each typed key arrives through the SAME projection a scored run uses, with
/// `Source::Config` — never a legacy env read — and the CLI resolvers turn it
/// into the lane's `LowDimConfig`.
#[test]
fn the_bundle_keys_arrive_through_the_scored_projection() {
    let mut preset = crate::preset::PresetConfig::default();
    preset.bab.branching.input_split.lowdim_varying_dims = Some(true);
    preset.bab.branching.input_split.lowdim_alpha_iters = Some(20);
    preset.bab.branching.input_split.lowdim_leaf_attack_every = Some(40);
    preset.bab.branching.input_split.lowdim_shared_worklist = Some(true);
    let levers = projected_levers(&preset);
    for decl in [
        &ny_levers::decls::lowdim_lane::LOWDIM_VARYING_DIMS,
        &ny_levers::decls::lowdim_lane::LOWDIM_SHARED_WORKLIST,
    ] {
        let resolved = levers.resolved(decl).expect("declared");
        assert!(resolved.value.as_bool(), "{} must arrive armed", decl.name);
        assert_eq!(resolved.source, ny_levers::Source::Config);
        assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
    }
    for (decl, want) in [
        (&ny_levers::decls::lowdim_lane::LOWDIM_ALPHA_ITERS, 20u64),
        (
            &ny_levers::decls::lowdim_lane::LOWDIM_LEAF_ATTACK_EVERY,
            40u64,
        ),
    ] {
        let resolved = levers.resolved(decl).expect("declared");
        assert_eq!(resolved.value.as_u64(), Some(want), "{}", decl.name);
        assert_eq!(resolved.source, ny_levers::Source::Config);
        assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
    }
    let lane = bundle_lane(&levers);
    assert!(lane.varying_dims);
    assert_eq!(lane.alpha_iters, 20);
    assert_eq!(lane.leaf_attack_every, 40);
    assert!(lane.shared_worklist);
}

/// The yaml spelling of each key parses into the typed preset (and a negative
/// count is refused at parse time).
#[test]
fn the_bundle_keys_parse_their_yaml_spelling() {
    let parsed: crate::preset::InputSplitPreset = serde_yaml::from_str(
        "lowdim_varying_dims: true\nlowdim_alpha_iters: 20\nlowdim_leaf_attack_every: 3\n\
         lowdim_shared_worklist: true",
    )
    .expect("parses");
    assert_eq!(parsed.lowdim_varying_dims, Some(true));
    assert_eq!(parsed.lowdim_alpha_iters, Some(20));
    assert_eq!(parsed.lowdim_leaf_attack_every, Some(3));
    assert_eq!(parsed.lowdim_shared_worklist, Some(true));
    assert!(
        serde_yaml::from_str::<crate::preset::InputSplitPreset>("lowdim_alpha_iters: -1").is_err(),
        "a negative count must fail the preset parse"
    );
}

/// No shipped preset arms any bundle key: every one is dark until its own
/// measurement and moat. Non-vacuity: a scan that found no yaml fails.
#[test]
fn no_shipped_preset_arms_the_bundle_keys() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let Ok(entries) = std::fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            scanned += 1;
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            for key in [
                "lowdim_varying_dims",
                "lowdim_alpha_iters",
                "lowdim_leaf_attack_every",
                "lowdim_shared_worklist",
            ] {
                assert!(
                    !text.contains(key),
                    "{} arms the #lowdim-bundle key {key}; no shipped preset may yet",
                    path.display()
                );
            }
        }
    }
    assert!(scanned > 0, "VOID: no shipped yaml was scanned");
}
