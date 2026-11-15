// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP benchmark preset loading tests.
//!
//! Extracted from `tests.rs` to stay under the 1000-line limit.
//! These tests load actual YAML configs from `configs/vnncomp25/` and verify
//! that parsed preset values flow correctly to `BetaCrownConfig`.

use super::apply::apply_preset;
use super::*;
use crate::commands::beta_crown::branching::{
    apply_resolved_auto_branching_runtime_policy, resolve_auto_branching, AutoBranchingReason,
    AutoBranchingRequest, ModelStructure,
};
use ny_propagate::{
    BetaCrownConfig, BranchingHeuristic, InputClipType, PgdAlphaMode, PgdOptimizer,
};
use std::{fs, path::Path};

#[test]
fn no_shipped_preset_arms_solver_dark_bound_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut configured = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.solver.const_matmul_interval.is_some()
                || preset.solver.joint_bilinear_quad.is_some()
            {
                configured.push(path);
            }
        }
    }

    assert!(
        configured.is_empty(),
        "the const-matmul and joint-bilinear bound paths must remain absent from every shipped \
         competition preset: {configured:?}"
    );
}

/// #lowdim-bundle: the four typed lowdim-lane keys (varying-dims admission +
/// Sigmoid, boundary alpha, in-lane leaf attack, the fold's shared worklist)
/// parse as their typed fields and stay ABSENT from every shipped preset, so
/// every scored run resolves them at their dark declaration defaults. Non-
/// vacuity: the scan must parse at least one preset, and a preset that sets a
/// key must round-trip it into the typed field (checked on a synthetic yaml).
#[test]
fn lowdim_bundle_keys_stay_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut parsed = 0usize;
    let mut armed = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            parsed += 1;
            let split = &preset.bab.branching.input_split;
            if split.lowdim_varying_dims.is_some()
                || split.lowdim_alpha_iters.is_some()
                || split.lowdim_leaf_attack_every.is_some()
                || split.lowdim_shared_worklist.is_some()
                || split.lowdim_batch_engine.is_some()
                || split.engine.is_some()
            {
                armed.push(path);
            }
        }
    }
    assert!(parsed > 0, "VOID: no shipped preset was parsed");
    assert!(
        armed.is_empty(),
        "the #lowdim-bundle keys must stay absent from every shipped preset until measured: \
         {armed:?}"
    );
    let synthetic: PresetConfig = serde_yaml::from_str(
        "bab:\n  branching:\n    input_split:\n      lowdim_varying_dims: true\n      \
         lowdim_alpha_iters: 20\n      lowdim_leaf_attack_every: 40\n      \
         lowdim_shared_worklist: true\n      lowdim_batch_engine: bbe\n      \
         engine: bbe\n",
    )
    .expect("the bundle keys parse");
    let split = &synthetic.bab.branching.input_split;
    assert_eq!(split.lowdim_varying_dims, Some(true));
    assert_eq!(split.lowdim_alpha_iters, Some(20));
    assert_eq!(split.lowdim_leaf_attack_every, Some(40));
    assert_eq!(split.lowdim_shared_worklist, Some(true));
    // M8a (#m8-input-split-batch): the two lane keys parse their tokens, and
    // an unknown token fails the preset parse (deny_unknown_fields does not
    // cover enum values; the serde enum does).
    assert_eq!(
        split.lowdim_batch_engine,
        Some(ny_propagate::lowdim::LowDimBatchEngine::Bbe)
    );
    assert_eq!(
        split.engine,
        Some(ny_propagate::lowdim::InputSplitEngine::Bbe)
    );
    assert!(serde_yaml::from_str::<PresetConfig>(
        "bab:\n  branching:\n    input_split:\n      lowdim_batch_engine: cuda\n",
    )
    .is_err());
    assert!(serde_yaml::from_str::<PresetConfig>(
        "bab:\n  branching:\n    input_split:\n      engine: graph\n",
    )
    .is_err());
}

#[test]
fn qdq_activation_relaxation_stays_dark_in_every_resolvable_preset() {
    // #smart-turn-qdq: the certified QDQ relaxation is REPORT-ONLY this round.
    // The only yaml allowed to arm `model.qdq_activation_relaxation` is the
    // explicitly-suffixed canary, which category resolution can never pick up
    // — asserted directly below, so a rename cannot silently promote it.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.model.qdq_activation_relaxation == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [
            // The multi-input canary (#multi-input-propagation) layers the
            // concat-view gate ON TOP of the QDQ gate, so the funnel's
            // report-only latch stays engaged when the bounds half finally
            // runs; it is equally suffixed and equally unreachable through
            // category resolution (asserted below).
            Path::new("vnncomp26/smart_turn_multimodal_2026_multiinput_canary.yaml").to_path_buf(),
            Path::new("vnncomp26/smart_turn_multimodal_2026_qdq_canary.yaml").to_path_buf(),
        ],
        "only the suffixed canaries may arm the dark QDQ relaxation gate"
    );

    // And the canary must be unreachable through category resolution: the
    // effective smart_turn preset (if any) must not arm the gate.
    if let Some(effective) =
        crate::commands::vnncomp::resolve_preset_path(&root, "smart_turn_multimodal_2026")
    {
        let preset = load_preset(&effective).expect("effective smart_turn preset loads");
        assert_ne!(
            preset.model.qdq_activation_relaxation,
            Some(true),
            "category resolution must never arm the QDQ canary: {}",
            effective.display()
        );
    }
}

#[test]
fn nn4sys_1d_cover_stays_dark_in_every_resolvable_preset() {
    // #nn4sys-1d-cover: the exact one-axis complete cover is DARK in
    // production. The only yaml allowed to arm any of its three keys is the
    // explicitly suffixed canary, which category resolution can never pick up
    // — asserted directly below, so a rename cannot silently promote it.
    //
    // Both halves matter and neither substitutes for the other: the first
    // says WHICH file arms the keys, the second says the file a scored nn4sys
    // run would actually load does not.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.bab.nn4sys_1d_cover.is_some()
                || preset.bab.nn4sys_1d_cover_depth.is_some()
                || preset.bab.nn4sys_1d_cover_slice.is_some()
            {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/nn4sys_1dcover_canary.yaml").to_path_buf()],
        "only the suffixed canary may arm the #nn4sys-1d-cover keys"
    );

    // The canary must actually arm all THREE, or the delivery proof it exists
    // to carry is vacuous — and the depth in particular, because the cover's
    // pass is all-or-nothing and a master without its depth is inert.
    let canary = load_preset(&root.join("vnncomp25/nn4sys_1dcover_canary.yaml"))
        .expect("the canary preset loads");
    assert_eq!(canary.bab.nn4sys_1d_cover, Some(true));
    assert_eq!(canary.bab.nn4sys_1d_cover_depth, Some(24));
    assert_eq!(canary.bab.nn4sys_1d_cover_slice, Some(0.25));

    // And the canary must be unreachable through category resolution: the
    // effective nn4sys preset must arm none of the three.
    let effective = crate::commands::vnncomp::resolve_preset_path(&root, "nn4sys")
        .expect("nn4sys resolves to a shipped preset");
    assert!(
        effective.ends_with("nn4sys.yaml"),
        "category resolution must land on the production preset, not a canary: {}",
        effective.display()
    );
    let preset = load_preset(&effective).expect("effective nn4sys preset loads");
    assert_eq!(preset.bab.nn4sys_1d_cover, None);
    assert_eq!(preset.bab.nn4sys_1d_cover_depth, None);
    assert_eq!(preset.bab.nn4sys_1d_cover_slice, None);
}

#[test]
fn multi_input_concat_stays_dark_in_every_resolvable_preset() {
    // #multi-input-propagation: the concat-view mechanism is DARK. The only
    // yaml allowed to arm `model.multi_input_concat` is the explicitly
    // suffixed canary, which category resolution can never pick up — asserted
    // directly below, so a rename cannot silently promote it.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.model.multi_input_concat == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp26/smart_turn_multimodal_2026_multiinput_canary.yaml").to_path_buf()],
        "only the suffixed canary may arm the dark multi-input concat gate"
    );

    // And the canary must be unreachable through category resolution: the
    // effective smart_turn preset (if any) must not arm the gate.
    if let Some(effective) =
        crate::commands::vnncomp::resolve_preset_path(&root, "smart_turn_multimodal_2026")
    {
        let preset = load_preset(&effective).expect("effective smart_turn preset loads");
        assert_ne!(
            preset.model.multi_input_concat,
            Some(true),
            "category resolution must never arm the multi-input canary: {}",
            effective.display()
        );
    }
}

#[test]
fn patches_engine_wiring_stays_dark_in_every_resolvable_preset() {
    // #patches-engine-refusal-containment / #fast-f32-prewarm-all-hosts
    // (reports/measured-2026/cct_bound_gap_2026-08-31): both bound-path keys
    // are DARK. The only yaml allowed to set either is the explicitly
    // suffixed cct canary, which category resolution can never pick up —
    // asserted directly below, so a rename cannot silently promote it.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.bab.patches_engine_refusal_containment.is_some()
                || preset.bab.fast_f32_prewarm.is_some()
            {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [
            Path::new("vnncomp26/challenging_certified_training_2026_patcheswiring_canary.yaml")
                .to_path_buf()
        ],
        "only the suffixed canary may set the dark cct engine-wiring keys"
    );

    // And the canary must be unreachable through category resolution: the
    // effective cct preset (if any) must not arm either gate.
    if let Some(effective) =
        crate::commands::vnncomp::resolve_preset_path(&root, "challenging_certified_training_2026")
    {
        let preset = load_preset(&effective).expect("effective cct preset loads");
        assert_ne!(
            preset.bab.patches_engine_refusal_containment,
            Some(true),
            "category resolution must never arm the patches-containment canary: {}",
            effective.display()
        );
        assert_ne!(
            preset.bab.fast_f32_prewarm,
            Some(true),
            "category resolution must never arm the fast-f32-prewarm canary: {}",
            effective.display()
        );
    }
}

/// The engine-wiring canary must differ from the shipped cct preset by
/// exactly its TWO typed treatment keys. Normalizing both and requiring full
/// serialized equality pins every other loader, attack, and solver knob to
/// the production values, so the canary measurement is attributable to the
/// wiring pair alone.
#[test]
fn patcheswiring_canary_differs_by_exactly_the_two_typed_keys() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production =
        load_preset(&repo_root.join("configs/vnncomp26/challenging_certified_training_2026.yaml"))
            .unwrap();
    let canary =
        load_preset(&repo_root.join(
            "configs/vnncomp26/challenging_certified_training_2026_patcheswiring_canary.yaml",
        ))
        .unwrap();

    assert_eq!(production.bab.patches_engine_refusal_containment, None);
    assert_eq!(production.bab.fast_f32_prewarm, None);
    assert_eq!(canary.bab.patches_engine_refusal_containment, Some(true));
    assert_eq!(canary.bab.fast_f32_prewarm, Some(true));

    let mut normalized_canary = canary;
    normalized_canary.bab.patches_engine_refusal_containment =
        production.bab.patches_engine_refusal_containment;
    normalized_canary.bab.fast_f32_prewarm = production.bab.fast_f32_prewarm;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "the patches-wiring canary may differ from the shipped cct preset only at \
         bab.patches_engine_refusal_containment and bab.fast_f32_prewarm"
    );
}

/// #dense-sparse-seed: `bab.crown_partial_dense_min_stable` is armed by the cct
/// preset alone (plus its patches-wiring canary, which must stay identical to
/// it outside its own two keys), at exactly alpha-beta-CROWN's 0.1 — and
/// category resolution hands that value to the scored cct route. Arming it on
/// any other category is a separate, measured decision that must edit this
/// list.
#[test]
fn only_cct_arms_crown_partial_dense_min_stable() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed: Vec<(PathBuf, f64)> = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if let Some(value) = preset.bab.crown_partial_dense_min_stable {
                armed.push((
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                    value,
                ));
            }
        }
    }
    armed.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        armed,
        vec![
            (
                Path::new("vnncomp26/challenging_certified_training_2026.yaml").to_path_buf(),
                0.1,
            ),
            (
                Path::new(
                    "vnncomp26/challenging_certified_training_2026_patcheswiring_canary.yaml"
                )
                .to_path_buf(),
                0.1,
            ),
        ],
        "only the cct preset (and its canary) may arm bab.crown_partial_dense_min_stable"
    );

    let effective =
        crate::commands::vnncomp::resolve_preset_path(&root, "challenging_certified_training_2026")
            .expect("the cct category resolves an explicit preset");
    let preset = load_preset(&effective).expect("effective cct preset loads");
    assert_eq!(
        preset.bab.crown_partial_dense_min_stable,
        Some(0.1),
        "category resolution must deliver the armed threshold: {}",
        effective.display()
    );
}

#[test]
fn only_vnncomp25_cifar100_arms_root_alpha_phase_checkpoint() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.bab.root_alpha_phase_checkpoint == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/cifar100_2024.yaml").to_path_buf()],
        "the root-alpha checkpoint production opt-in must remain scoped to measured CIFAR100"
    );
}

#[test]
fn effective_cifar100_2024_preset_arms_root_alpha_phase_checkpoint() {
    let configs_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let effective = crate::commands::vnncomp::resolve_preset_path(&configs_root, "cifar100_2024")
        .expect("CIFAR100 production preset resolves");
    let preset = load_preset(&effective).expect("effective CIFAR100 preset loads");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("effective CIFAR100 preset applies");
    assert!(
        config.root_alpha_phase_checkpoint,
        "newer vnncomp directories must not silently shadow the qualified CIFAR100 policy"
    );
}

#[test]
fn only_vnncomp25_cifar100_arms_kfsb_cert_reuse() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.bab.branching.kfsb_cert_reuse == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/cifar100_2024.yaml").to_path_buf()],
        "the proof-carrying kFSB reuse opt-in must remain scoped to qualified CIFAR100"
    );
}

#[test]
fn effective_cifar100_2024_preset_arms_kfsb_cert_reuse() {
    let configs_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let effective = crate::commands::vnncomp::resolve_preset_path(&configs_root, "cifar100_2024")
        .expect("CIFAR100 production preset resolves");
    assert!(
        effective.ends_with("vnncomp25/cifar100_2024.yaml"),
        "the missing vnncomp26 override must fall back to the vnncomp25 CIFAR100 preset: {}",
        effective.display()
    );
    let preset = load_preset(&effective).expect("effective CIFAR100 preset loads");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("effective CIFAR100 preset applies");
    assert!(
        config.kfsb_cert_reuse,
        "newer vnncomp directories must not silently shadow the CIFAR100 certificate policy"
    );
}

#[test]
fn breakpoint_category_defaults_resolve_on_both_competition_years() {
    let configs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    for category in ["cifar100_2024", "tinyimagenet_2024"] {
        let expected = configs.join("vnncomp25").join(format!("{category}.yaml"));
        let scored = crate::commands::vnncomp::resolve_preset_path(&configs, category)
            .expect("the scored category must resolve a preset");
        assert_eq!(
            scored, expected,
            "{category} must retain its effective category preset"
        );
        for year in [2025, 2026] {
            let sweep = crate::commands::vnncomp::sweep_preset_path(&configs, year, category)
                .expect("both competition years must resolve the category preset");
            assert_eq!(
                sweep, scored,
                "{year} {category}: sweep and scored defaults disagree"
            );
            let preset = load_preset(&sweep).expect("the effective category preset must load");
            assert_eq!(preset.margin_row.frozen_dual_bab, Some(true));
            assert_eq!(preset.margin_row.frozen_dual_refine_steps, Some(4));
            assert_eq!(preset.margin_row.frozen_dual_bab_secs, None);
            assert_eq!(preset.margin_row.frozen_dual_bab_node_cap, None);
            assert_eq!(preset.margin_row.concurrent, Some(true));
            assert_eq!(preset.general.device.as_deref(), Some("wgpu"));
        }
    }
}

#[test]
fn only_vnncomp25_acas_arms_relational_exact_gradient_wrapper() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.attack.vnncomp_upfront_relational_exact_gradient == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/acasxu_2023.yaml").to_path_buf()],
        "the default-off relational wrapper opt-in must remain ACAS-only"
    );
}

#[test]
fn no_shipped_preset_arms_mo_cuda_factory_engine_handoff() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.bab.mo_cuda_factory_engine_handoff.is_some() {
                armed.push(path);
            }
        }
    }

    assert!(
        armed.is_empty(),
        "the default-dark handoff must remain absent from every shipped preset: {armed:?}"
    );
}

#[test]
fn no_shipped_preset_arms_mo_cuda_bounded_shared_executor() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            if preset.bab.mo_cuda_bounded_shared_executor.is_some() {
                armed.push(path);
            }
        }
    }

    assert!(
        armed.is_empty(),
        "the default-dark bounded shared executor must remain absent from every shipped preset: \
         {armed:?}"
    );
}

/// #vit-bank-gate + #vit-margin-alpha-complex (P8): the containment fence
/// and its ORT adjudication are SCOPED TO vit_2023. They are verdict-surface
/// changes — the fence refuses proofs (and, armed, fails closed on an
/// unfenced root-terminal authority or any emptiness claim) — built as the
/// vit bank gate, so no other category may ever set either key; a preset
/// that does fails here. Arming them on vit_2023 itself is a measured
/// decision that lands in its own commit (with its moat); this test permits
/// it but does not require it, and prints the current vit state.
#[test]
fn mo_containment_fence_is_scoped_to_vit_2023() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut outside_scope = Vec::new();
    let mut vit_state = Vec::new();
    let mut parsed = 0usize;
    let mut saw_vit = false;

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            parsed += 1;
            let is_vit = path.file_stem().and_then(|stem| stem.to_str()) == Some("vit_2023");
            saw_vit |= is_vit;
            for (key, set) in [
                (
                    "bab.mo_containment_fence",
                    preset.bab.mo_containment_fence.is_some(),
                ),
                (
                    "bab.mo_root_ort_adjudication",
                    preset.bab.mo_root_ort_adjudication.is_some(),
                ),
            ] {
                match (set, is_vit) {
                    (true, true) => vit_state.push(format!("{} sets {key}", path.display())),
                    (true, false) => outside_scope.push(format!("{} sets {key}", path.display())),
                    (false, _) => {}
                }
            }
        }
    }

    assert!(parsed > 0, "the shipped preset directories must resolve");
    assert!(
        saw_vit,
        "the scope anchor configs/<year>/vit_2023.yaml must exist, or this scoping is vacuous"
    );
    eprintln!("[vit-bank-gate-scope] vit_2023 fence keys: {vit_state:?}");
    assert!(
        outside_scope.is_empty(),
        "the containment fence and its ORT adjudication are the vit_2023 bank gate; no other \
         shipped preset may set them: {outside_scope:?}"
    );
}

/// #vit-v3-bankable: `bab.mo_cover_certify` is armed by no shipped preset
/// until its measurement (the vit bank protocol, docs/engine_program/
/// VIT_BANKABILITY.md) commits an arming.
#[test]
fn no_shipped_preset_arms_the_mo_cover_certify_key() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();
    let mut parsed = 0usize;
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
            parsed += 1;
            if preset.bab.mo_cover_certify.is_some() {
                armed.push(format!(
                    "{year}/{}",
                    path.file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default()
                ));
            }
        }
    }
    assert!(parsed > 0, "the shipped preset directories must resolve");
    assert!(
        armed.is_empty(),
        "these shipped presets set bab.mo_cover_certify without a measurement commit: {armed:?}"
    );
}

/// #vit-collector-throughput / #vit-batched-bilinear: the four collector-kernel
/// keys ship DARK. They are throughput levers whose arming is a measured
/// decision (and, for vit, no vit yaml ships before the P8 bank gate), so every
/// shipped preset must omit all four until the commit that carries the
/// measurement edits this test.
#[test]
fn no_shipped_preset_arms_vit_collector_kernel_keys() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();
    let mut parsed = 0usize;

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
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
            parsed += 1;
            for (key, set) in [
                (
                    "bab.bilinear_rowpar_kernel",
                    preset.bab.bilinear_rowpar_kernel.is_some(),
                ),
                (
                    "bab.linear_position_stack",
                    preset.bab.linear_position_stack.is_some(),
                ),
                (
                    "bab.reciprocal_unbounded_upper",
                    preset.bab.reciprocal_unbounded_upper.is_some(),
                ),
                (
                    "bab.bilinear_gemm_kernel",
                    preset.bab.bilinear_gemm_kernel.is_some(),
                ),
            ] {
                if set {
                    armed.push(format!("{} sets {key}", path.display()));
                }
            }
        }
    }

    assert!(parsed > 0, "the shipped preset directories must resolve");
    assert!(
        armed.is_empty(),
        "the vit collector-kernel keys must remain absent from every shipped preset until \
         their own measured arming commit: {armed:?}"
    );
}

#[test]
fn cifar100_critical_alpha_bracket_uses_beta_crown_lr_not_adaptive_default() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    // #general-defaults 2026-08-31: the yaml no longer STATES lr_alpha, because
    // `BetaCrownConfig::default().alpha_lr` is now 0.1 -- the value 13 of the 14
    // scored configs that set it were all writing by hand. The key's absence is
    // therefore not a regression; what this test guards is the RESOLVED value,
    // asserted below after `apply_preset`, which is strictly stronger than a yaml
    // pin (it also catches a key that is present but overridden downstream).
    // If the engine default ever drifts off 0.1, the assertion below fails.
    assert!(
        cifar.bab.beta_crown.lr_alpha.is_none_or(|v| v == 0.1),
        "cifar100 may omit lr_alpha (default 0.1) or state 0.1, but not disagree"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cifar).unwrap();
    assert_eq!(
        config.alpha_lr.to_bits(),
        0.1_f32.to_bits(),
        "the bracket base must be the applied beta-CROWN alpha LR"
    );
    assert_eq!(
        config.adaptive_config.alpha_lr.to_bits(),
        0.01_f32.to_bits(),
        "the sealed adaptive default remains distinct and must not seed the bracket"
    );
    assert_eq!(
        [0.3_f32, 1.0, 2.0].map(|scale| (config.alpha_lr * scale).to_bits()),
        [0.1_f32 * 0.3, 0.1, 0.2].map(f32::to_bits),
        "the applied CIFAR base yields the staged 0.03/0.10/0.20 bracket"
    );
}

#[test]
fn cifar100_2024_arms_bounded_dense_head_crown_only_in_its_preset() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(
        cifar.general.device.as_deref(),
        Some("wgpu"),
        "the CIFAR preset must pass a WGPU engine into the ordinary verifier route"
    );
    assert_eq!(cifar.bab.root_crown_interm_dense_head, Some(true));
    assert_eq!(cifar.bab.root_crown_interm_max_secs, Some(2));
    assert_eq!(cifar.bab.root_crown_interm_max_dim, Some(512));

    let mut cifar_config = BetaCrownConfig::default();
    apply_preset(&mut cifar_config, &cifar).unwrap();
    assert!(cifar_config.root_crown_interm_dense_head);
    assert_eq!(cifar_config.root_crown_interm_max_secs, 2);
    assert_eq!(cifar_config.root_crown_interm_max_dim, 512);
    assert_eq!(
        cifar_config.alpha_config.iterations, 20,
        "CIFAR keeps its alpha-update schedule until a fast initial-CROWN route \
         preserves the useful state without regressing banked UNSAT rows"
    );

    let acas = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(acas.bab.root_crown_interm_dense_head, None);
    let mut acas_config = BetaCrownConfig::default();
    apply_preset(&mut acas_config, &acas).unwrap();
    assert!(
        !acas_config.root_crown_interm_dense_head,
        "representative non-cifar presets must remain default-off"
    );
}

#[test]
fn cifar100_2024_types_the_exact_open_row_adaptive_route() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.margin_row.adaptive_reserve, Some(true));

    let acas = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(
        acas.margin_row.adaptive_reserve, None,
        "unrelated categories retain the historical reserve policy"
    );
}

#[test]
fn cifar100_2024_enables_reference_adaptive_relu_split_depth() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.bab.min_batch_size_ratio, Some(0.1));
    assert_eq!(cifar.bab.max_split_depth, Some(4));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cifar).unwrap();
    assert_eq!(config.min_batch_fill_ratio, 0.1);
    assert_eq!(config.max_relu_split_depth, 4);
    assert_eq!(
        config.effective_relu_split_depth(1),
        4,
        "singleton CIFAR root should commit the reference 16-leaf split"
    );
}

/// #a2-parity-not-exercised (measured 2026-08-09, GB10, `--features mip,cuda`).
///
/// The asymmetry pinned below is NOT an oversight waiting to be corrected by a
/// copy from the sibling preset. Arming the whole cifar100 bundle on
/// tinyimagenet — `batch_size` 128 -> 256, `candidates` 10 -> 7, the `clip`
/// block, `kfsb_multi`, `max_split_depth`, `min_batch_size_ratio`,
/// `alpha_crown.lr_alpha` 0.2 -> 0.25, `beta_crown.iterations` 15 -> 10 and
/// `root_crown_interm_dense_head` — was A/B'd at the official 100 s budget,
/// arms interleaved per instance, on both halves of the field (the GT-decided
/// rows ny misses and the rows it banks).
///
/// The arms were confirmed LIVE from the run's own effective-config receipt,
/// and the result was still null, because the internal verifier reports
/// `domains_explored: 0` on this category: every key in the bundle except the
/// dense head is consumed inside the BaB loop, which never runs. The verdicts
/// come from the concurrent margin-row lane, whose entry point
/// (`ny_propagate::margin_row::run_margin_row_lane`) takes no `BetaCrownConfig`
/// at all, so no `bab.*` key can reach it by signature.
///
/// See `docs/TINYIMAGENET_PARITY_A2_NOT_EXERCISED_2026-08-09.md`. Re-attempt
/// only after `domains_explored > 0` is demonstrated here.
#[test]
fn complete_clip_experiment_is_scoped_to_cifar100_not_tinyimagenet() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    let tiny = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();

    assert_eq!(cifar.bab.clip.interm_domain, Some(true));
    assert_eq!(cifar.bab.clip.interm_topk, Some(20));
    assert_eq!(cifar.bab.max_split_depth, Some(4));
    assert_eq!(cifar.bab.branching.kfsb_multi, Some(true));
    assert_eq!(cifar.bab.branching.kfsb_cert_reuse, Some(true));
    assert_eq!(cifar.bab.root_alpha_phase_checkpoint, Some(true));

    let mut cifar_config = BetaCrownConfig::default();
    apply_preset(&mut cifar_config, &cifar).unwrap();
    assert!(cifar_config.enable_clip_interm_domain);
    assert_eq!(cifar_config.clip_interm_topk, 20);
    assert_eq!(cifar_config.max_relu_split_depth, 4);
    assert!(cifar_config.use_kfsb_multi_branching);
    assert!(cifar_config.kfsb_cert_reuse);
    assert!(cifar_config.root_alpha_phase_checkpoint);

    assert_eq!(tiny.bab.clip.interm_domain, None);
    assert_eq!(tiny.bab.clip.interm_topk, None);
    assert_eq!(tiny.bab.max_split_depth, None);
    assert_eq!(tiny.bab.branching.kfsb_multi, None);
    assert_eq!(tiny.bab.branching.kfsb_cert_reuse, None);
    assert_eq!(tiny.bab.root_alpha_phase_checkpoint, None);

    let mut tiny_config = BetaCrownConfig::default();
    apply_preset(&mut tiny_config, &tiny).unwrap();
    assert!(!tiny_config.enable_clip_interm_domain);
    assert_eq!(tiny_config.clip_interm_topk, 3);
    assert_eq!(tiny_config.max_relu_split_depth, 1);
    assert!(!tiny_config.use_kfsb_multi_branching);
    assert!(!tiny_config.kfsb_cert_reuse);
    assert!(!tiny_config.root_alpha_phase_checkpoint);
}

/// #margin-row-branch-width: pin the measured cifar100 width and its scoping.
///
/// k=16 converts `idx_8600_sidx_2721` (timeout -> unsat, 186 expansions,
/// frontier drains 424 -> 2) and k=4 LOSES a banked proof, so this key is
/// evidence-bearing in both directions: the value matters, and it must not
/// leak to categories where it was never measured (the tinyimagenet transfer
/// trap all over again — see the allocation-trio no-go below).
#[test]
fn cifar100_branch_width_is_pinned_and_scoped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.margin_row.k_head, Some(16));
    assert_eq!(cifar.margin_row.k_trunk, Some(16));

    let tiny = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();
    assert_eq!(
        tiny.margin_row.k_head, None,
        "k_head is still untransferred and unmeasured on tiny"
    );
    // #k-trunk-narrow: tiny arms k_trunk on ITS OWN evidence, not by transfer —
    // which is what this test exists to police. MEASURED 2026-09-16: the branch
    // shortlist was scoring seven trunk candidates that essentially never win
    // (`trunk_split=0` over 214 splits, and `score_candidates` is the LARGEST
    // phase on this category at 50.4%). 5 paired rows +49% expansions, 25
    // candidate rows -> 1 control-verified conversion (idx_2045_sidx_7367,
    // ctl 2/2 timeout vs arm 2/2 unsat), 18 banked unsats clean.
    assert_eq!(tiny.margin_row.k_trunk, Some(1));

    // #backward-interm ships to BOTH wall categories — each on its own
    // measured evidence and its own pre-registered ship gate (2026-08-19).
    assert_eq!(cifar.margin_row.backward_interm, Some(true));
    assert_eq!(tiny.margin_row.backward_interm, Some(true));
}

/// Names of every `margin_row` field this preset actually sets.
///
/// Field-wise rather than a derived comparison on purpose: `MarginRowPreset`
/// gains keys faster than presets adopt them, and a NEW key must show up here
/// as a compile error in [`margin_row_wall_preset_field_diff_is_deliberate`]
/// rather than slip through an `..` rest pattern unexamined.
fn margin_row_fields_set(m: &MarginRowPreset) -> Vec<&'static str> {
    let MarginRowPreset {
        reserve_secs,
        adaptive_reserve,
        reserve_max_frac,
        release_frac,
        concurrent,
        root_f32,
        tableau_f32,
        domain_stack,
        err_envelope,
        lru,
        parallel,
        k_head,
        k_trunk,
        k_adaptive,
        gpu,
        gpu_batch,
        gpu_backward,
        gpu_conv,
        frozen_dual_bab,
        frozen_dual_bab_node_cap,
        frozen_dual_bab_secs,
        frozen_dual_refine_steps,
        frozen_dual_bab_state,
        frozen_dual_bab_search,
        tree_mode,
        tree_wave_batch,
        tree_wave_cert_threads,
        alpha_opt,
        alpha_iters,
        alpha_secs,
        alpha_lr,
        alpha_cols,
        alpha_guard,
        spec_alpha,
        spec_alpha_width,
        spec_alpha_cands,
        spec_alpha_rounds,
        spec_alpha_entries,
        clip,
        clip_head,
        clip_interm,
        clip_rows,
        clip_topk,
        closure_thresh,
        closure_warmup,
        closure_window,
        closure_max_tier,
        bi_chunk,
        bi_chunk_secs,
        bi_topk,
        interm_alpha,
        beta,
        beta_eta,
        beta_iters,
        beta_per_column,
        beta_columns,
        column_retire,
        parallel_prebuild,
        beta_lambda,
        beta_polyak,
        beta_heads,
        rebake,
        rebake_rounds,
        rebake_secs,
        bi_window,
        joint_refresh,
        joint_refresh_secs,
        head_groups,
        head_directional_cuts,
        head_phase_cover,
        root_certify_once,
        root_certify_once_iters,
        root_certify_once_refresh,
        root_certify_once_secs,
        root_search,
        root_box_source,
        bi_gpu,
        bi_patch,
        bi_patch_ratio,
        bi_patch_mb,
        ibp_seed,
        closure_adaptive,
        branch_reduce_max,
        parent_lift,
        backward_interm,
        domain_alpha,
        domain_alpha_iters,
        domain_alpha_eta,
        domain_alpha_min_depth,
        domain_alpha_stride,
        domain_alpha_gap,
        clip_stateful,
        bi_secs,
        epoch_tier0,
        epoch_universe,
        epoch_retain,
        epoch_depth,
        epoch_attempts,
        epoch_parent_topk,
        early_exit,
        verifier_threads,
        concurrent_threads,
    } = m;
    let flags: [(&'static str, bool); 102] = [
        ("reserve_secs", reserve_secs.is_some()),
        ("adaptive_reserve", adaptive_reserve.is_some()),
        ("reserve_max_frac", reserve_max_frac.is_some()),
        ("release_frac", release_frac.is_some()),
        ("concurrent", concurrent.is_some()),
        ("root_f32", root_f32.is_some()),
        ("tableau_f32", tableau_f32.is_some()),
        ("domain_stack", domain_stack.is_some()),
        ("err_envelope", err_envelope.is_some()),
        ("lru", lru.is_some()),
        ("parallel", parallel.is_some()),
        ("k_head", k_head.is_some()),
        ("k_trunk", k_trunk.is_some()),
        ("k_adaptive", k_adaptive.is_some()),
        ("gpu", gpu.is_some()),
        ("gpu_batch", gpu_batch.is_some()),
        ("gpu_backward", gpu_backward.is_some()),
        ("gpu_conv", gpu_conv.is_some()),
        ("frozen_dual_bab", frozen_dual_bab.is_some()),
        (
            "frozen_dual_bab_node_cap",
            frozen_dual_bab_node_cap.is_some(),
        ),
        ("frozen_dual_bab_secs", frozen_dual_bab_secs.is_some()),
        (
            "frozen_dual_refine_steps",
            frozen_dual_refine_steps.is_some(),
        ),
        ("frozen_dual_bab_state", frozen_dual_bab_state.is_some()),
        ("frozen_dual_bab_search", frozen_dual_bab_search.is_some()),
        ("tree_mode", tree_mode.is_some()),
        ("tree_wave_batch", tree_wave_batch.is_some()),
        ("tree_wave_cert_threads", tree_wave_cert_threads.is_some()),
        ("alpha_opt", alpha_opt.is_some()),
        ("alpha_iters", alpha_iters.is_some()),
        ("alpha_secs", alpha_secs.is_some()),
        ("alpha_lr", alpha_lr.is_some()),
        ("alpha_cols", alpha_cols.is_some()),
        ("alpha_guard", alpha_guard.is_some()),
        ("spec_alpha", spec_alpha.is_some()),
        ("spec_alpha_width", spec_alpha_width.is_some()),
        ("spec_alpha_cands", spec_alpha_cands.is_some()),
        ("spec_alpha_rounds", spec_alpha_rounds.is_some()),
        ("spec_alpha_entries", spec_alpha_entries.is_some()),
        ("clip", clip.is_some()),
        ("clip_head", clip_head.is_some()),
        ("clip_interm", clip_interm.is_some()),
        ("clip_rows", clip_rows.is_some()),
        ("clip_topk", clip_topk.is_some()),
        ("closure_thresh", closure_thresh.is_some()),
        ("closure_warmup", closure_warmup.is_some()),
        ("closure_window", closure_window.is_some()),
        ("closure_max_tier", closure_max_tier.is_some()),
        ("bi_chunk", bi_chunk.is_some()),
        ("bi_chunk_secs", bi_chunk_secs.is_some()),
        ("bi_topk", bi_topk.is_some()),
        ("interm_alpha", interm_alpha.is_some()),
        ("beta", beta.is_some()),
        ("beta_eta", beta_eta.is_some()),
        ("beta_iters", beta_iters.is_some()),
        ("beta_per_column", beta_per_column.is_some()),
        ("beta_columns", beta_columns.is_some()),
        ("column_retire", column_retire.is_some()),
        ("parallel_prebuild", parallel_prebuild.is_some()),
        ("beta_lambda", beta_lambda.is_some()),
        ("beta_polyak", beta_polyak.is_some()),
        ("beta_heads", beta_heads.is_some()),
        ("rebake", rebake.is_some()),
        ("rebake_rounds", rebake_rounds.is_some()),
        ("rebake_secs", rebake_secs.is_some()),
        ("bi_window", bi_window.is_some()),
        ("joint_refresh", joint_refresh.is_some()),
        ("joint_refresh_secs", joint_refresh_secs.is_some()),
        ("head_groups", head_groups.is_some()),
        ("head_directional_cuts", head_directional_cuts.is_some()),
        ("head_phase_cover", head_phase_cover.is_some()),
        ("root_certify_once", root_certify_once.is_some()),
        ("root_certify_once_iters", root_certify_once_iters.is_some()),
        (
            "root_certify_once_refresh",
            root_certify_once_refresh.is_some(),
        ),
        ("root_certify_once_secs", root_certify_once_secs.is_some()),
        ("root_search", root_search.is_some()),
        ("root_box_source", root_box_source.is_some()),
        ("bi_gpu", bi_gpu.is_some()),
        ("bi_patch", bi_patch.is_some()),
        ("bi_patch_ratio", bi_patch_ratio.is_some()),
        ("bi_patch_mb", bi_patch_mb.is_some()),
        ("ibp_seed", ibp_seed.is_some()),
        ("closure_adaptive", closure_adaptive.is_some()),
        ("branch_reduce_max", branch_reduce_max.is_some()),
        ("parent_lift", parent_lift.is_some()),
        ("backward_interm", backward_interm.is_some()),
        ("domain_alpha", domain_alpha.is_some()),
        ("domain_alpha_iters", domain_alpha_iters.is_some()),
        ("domain_alpha_eta", domain_alpha_eta.is_some()),
        ("domain_alpha_min_depth", domain_alpha_min_depth.is_some()),
        ("domain_alpha_stride", domain_alpha_stride.is_some()),
        ("domain_alpha_gap", domain_alpha_gap.is_some()),
        ("clip_stateful", clip_stateful.is_some()),
        ("bi_secs", bi_secs.is_some()),
        ("epoch_tier0", epoch_tier0.is_some()),
        ("epoch_universe", epoch_universe.is_some()),
        ("epoch_retain", epoch_retain.is_some()),
        ("epoch_depth", epoch_depth.is_some()),
        ("epoch_attempts", epoch_attempts.is_some()),
        ("epoch_parent_topk", epoch_parent_topk.is_some()),
        ("early_exit", early_exit.is_some()),
        ("verifier_threads", verifier_threads.is_some()),
        ("concurrent_threads", concurrent_threads.is_some()),
    ];
    flags
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect()
}

/// The margin-row lane is the prover for BOTH wall categories, and the campaign
/// has repeatedly shipped a cifar100 key and then discovered nobody asked
/// whether tinyimagenet wanted it. This test is the field-by-field diff of the
/// two `margin_row:` sections in executable form, so a transfer is a deliberate
/// edit here rather than a silent copy there.
///
/// The decisions, each with the evidence recorded next to the key in the yaml:
///
/// | field | cifar100 | tinyimagenet | why they differ |
/// |---|---|---|---|
/// | `reserve_secs` | absent (45 default, then released) | `0` | see `adaptive_reserve` |
/// | `adaptive_reserve` | `true` | absent | both hand the verifier the full budget; the ONLY difference is that the adaptive route additionally arms the scored sparse root CROWN (`enables_scored_sparse_crown`), and this category's internal verifier reports `domains_explored: 0` |
/// | `concurrent` / `root_f32` | `true` | `true` | same twin-wall pair, measured separately on each |
/// | `tableau_f32` | absent | `true` | ARMED ON TINY'S OWN EVIDENCE 2026-08-25: conv 41.4->24.6 s, idx_5451 timeout->unsat 3/3, and the 11-row wall-hugger ship gate held 3/3. NOT transferred to cifar100, which sets root_f32 but has no A/B of its own (alloc-trio rule) |
/// | `lru` / `parallel` | absent (64 / all workers) | absent (64 / all workers) | typed delivery makes both measured throughput defaults searchable without silently arming a new YAML treatment; explicit parallel `0` is the serial control and typed `1` is invalid because legacy `1` means all |
/// | `k_head` / `k_trunk` | `16` | absent (8) | cifar's evidence is a frontier explosion; this lane runs a 2-expansion tree |
/// | `k_adaptive` | absent | absent | typed delivery exists; measured 3/3 on cifar, but it remains dark and unmeasured on tiny |
/// | `alpha_opt` | `true` | `true` | cifar100 retains its own timeout-to-UNSAT conversion and 9/9 moat. Tiny ARMED 2026-09-22 (#tiny-alpha-aligned): its 2026-08-30 loss had two causes, both now fixed — the MEAN-over-8-columns direction (`alpha_cols`) and the unset `alpha_secs` cap. Marginal rows convert 7/10 armed vs 1/10 shipped (p=0.0099), root bit-identical every rep |
/// | `alpha_cols` | `1` (measured) | `1` (measured) | the ascent's objective is the MIN column, its gradient was a MEAN over 8, so one column aligns them. Real root gain on both categories; on cifar100 it converts nothing ALONE (0 of 27), on tiny it is half of the pair above |
/// | `alpha_secs` | absent (20 default) | `35.0` | tiny's ascent SATURATES the 20 s default cap; cifar100 has no evidence either way, so it keeps the default |
/// | `gpu*`; other explicit alpha tuning; `clip*` / beta / rebake fields | absent | absent | typed search delivery only; no other explicit tuning or treatment is qualified to ship without its own sealed A/B |
/// | `closure_adaptive` (+ `closure_max_tier`) | `true` / `2` | absent | ARMED ON CIFAR100'S OWN EVIDENCE 2026-09-01. The ladder escalates on the lane's OWN trailing child-closure fraction and `bab.rs` fires the per-domain alpha ascent at `TIER_BETA_DEEP`, so this is a self-configuring gate, not a tuned constant: it refuses to escalate after 75% of budget (`NO_ESCALATE_AFTER_FRAC`), must afford 3x the current per-expansion cost, and rolls back an escalation that ran 4x slower without raising closure. 79-row moat vs shipped: held 79, lost 0, flips 0; fired on ONE row; converts idx_5127_sidx_993 (certified, 5/5). Armed GLOBALLY it is net -30 -- the gate is what makes it positive. `max_tier` 2 stops below `TIER_EPOCH`, whose root-gain discard is a measured regression. Tiny stays dark pending its own moat: 61 of its 74 banked unsats publish within 10 s of the 100 s wall, and the LOWER rung (`TIER_BETA`) was measured there at +1 conversion vs -4 proofs and refused |
/// | `backward_interm` | `true` | `true` | shipped to each on its own ship gate (2026-08-19) |
/// | `domain_alpha` (+ `_iters`, `_eta`, and the SCHEDULE `_min_depth` / `_stride` / `_gap`) | absent | absent | #margin-row-domain-alpha ships dark pending its pre-registered 2x2 (off / clip / alpha / BOTH) on idx_1298 and idx_4597 — and the schedule grid is MEASURED NULL (2026-08-24: zero conversions at every operating point; see margin_row/domain_alpha.rs); arming it on one category is exactly the transfer this test exists to make deliberate |
/// | `bi_secs` | `40` (measured) | ABSENT | the old "2.16x overrun" rationale confused root-build `secs=` with phase `work_secs=`. The later Tiny dose response solved idx_5185 only 1/4 and did not bank; idx_9611 failed under every dose on that dirty binary, so it is not a causal loss. Tableau-f32 later changed the timing frame again. Tiny stays at 20 pending a clean current-binary 20-vs-30 receipt-closed A/B; see docs/BI_DOSE_RESPONSE_5185_2026-08-24.md |
/// | `bi_chunk_secs` | ABSENT | ABSENT | #bi-adaptive-chunk ships DARK (default 0 = the historical fixed grain). The exploratory idx_9611 observation used unretained logs and a dirty binary, and its declaration cited an unrelated commit; it is context, not scored provenance. Any non-zero value remains an unmeasured experiment requiring a sealed execution receipt |
/// | `epoch_tier0` / `epoch_universe` / `epoch_retain` | absent | absent | typed delivery makes the ranker measurable from the clean scorecard, but promotion does not arm the July ranker result on any shipped category |
/// | `epoch_depth` / `epoch_attempts` / `epoch_parent_topk` | absent | absent | typed delivery exists so the clean scorecard can measure it, but the authoritative refresh remains dark pending a sealed A/B |
/// | `bi_secs` | `40` (measured) | absent (20 default) | the same-day cifar transfer was REVERTED by 3ce02e38a before it was ever scored — tiny's rows hug the publication wall |
/// | `branch_reduce_max` | `true` (measured) | absent | #branch-reduceop. abc's OWN tuned cifar100 and tinyimagenet configs both set `reduceop: max` (De Palma eq. 22), overriding abc's global `min` default on exactly the two benchmarks ny fails; ny hardcoded `min`. Armed on cifar100 2026-08-30 with #interm-alpha as a PAIR — neither converts alone. Tiny stays dark pending its own moat sweep: 60 of its 65 banked unsats publish at 89.5-97.1 s of 100 s |
/// | `interm_alpha` | `2` (measured) | absent | #interm-alpha, the other half of that pair. Dose 2 = chord-parallel + flipped area-heuristic vertex, intersected SHRINK-ONLY, so a wrong dose costs wall (`ia_secs`) and never soundness. Gate was the FULL bank: 124 banked-timeout + 47 banked-unsat rows, engagement 171/171, moat 47/47 with zero breaks, and a yaml-route proof with zero `NY_*` set (control: no `[interm-alpha]` line and timeout; armed: `armed=1`, `mode=max` 10/10, UNSAT) |
/// | `domain_stack` + `err_envelope` | `true` + `envelope` (measured) | `true` + `envelope` (measured) | ARMED 2026-09-26 as a PAIR (scratchpad/mrab, 3 reps x 3 arms, 100 s): cifar100 idx_2050/idx_4921 timeout 3/3 -> unsat 3/3, the other rows 22-26% faster; tiny +54-106% expansions. Stack ALONE loses idx_3894 3/3, so it never ships alone; the envelope only loosens the certified penalty (worst 6.0e-9) and stacking is bit-identical |
/// | `parent_lift` | `true` (measured) | absent | #branch-parent-lift. Converts idx_8739_sidx_1540 (a row the pair alone does not) at ZERO wall cost, moat 47/47, engagement 124/124. Tiny stays dark: it changes tree shape and 63 of 74 tiny unsats publish within 10 s of the wall |
/// | `early_exit` | absent (= ON) | absent (= ON) | default; the key exists only as a kill switch |
/// | `bi_topk` | ABSENT (the allocator owns it) | absent | #bi-topk-128 measured a real conversion (idx_9372_sidx_3515, 10/10, control 0/3, moat 80/80, tight rows +18.1s) but is NOT armed here: `#auto-alloc` narrows it per instance from the model's own work proxy. Arming it in the yaml would pre-narrow the value the allocator prices from, dropping the proxy ~4x so the root looks affordable and the allocator declines to shed `interm_alpha` — losing idx_1280_sidx_5310. The yaml states the CEILING; the allocator states the instance |
/// | `column_retire` | `true` (measured) | absent | #margin-row-col-retire. Retires a margin column once an ancestor certified it closed. Converts `idx_5831_sidx_5506` (+0.53); 15 banked unsats held with 8 genuine engagement tests; rarity ~1 in 57, so worth ONE row. resnet_large is unaffected by construction (`cr_lines=0`) |
/// | `verifier_threads` | absent | absent | Front B's seam, MEASURED NEGATIVE 2026-09-17: 9 runs, engagement 9/9, `exp=22` on every run, lane 34.87s (shared 20) vs 35.33s (private 18) vs 37.57s (private 16). Monotone in LANE width, not verifier contention — the wall presets run the verifier on `device: wgpu`, so there is no starvation to relieve and a cap only takes threads off the lane |
/// | `frozen_dual_bab` (+ `_node_cap`, `_secs`) | `true` (tuning keys absent) | `true` (tuning keys absent) | User-directed default activation of the certified tier: automatic mode uses at most 4 s / 5% remaining time, 8192 nodes per objective and 256 MiB state; dense row estimates >=64 MiB select one objective, otherwise at most two. Explicit positive seconds retain manual mode. Official-budget benefit and the full both-year comparison remain unmeasured |
/// | `frozen_dual_bab_state` | absent | absent | #fd-shared-state (ENGINE_PROGRAM M3) ships DARK on BOTH (`per_class`, bit-identical): `shared` builds ONE trunk of rows under slopes shared across the open classes before the fair shares (tiny t4863 spends 11.41 s of its 27.2 s tier in six per-class builds; t8853 refused all 15 on the ~1.3 s shares) and gives each class a one-column margin tail. Inert unless `frozen_dual_bab` is armed; arming needs the yaml-route `[frozen-dual] trunk state=shared` proof, the shared-lambda root-loss and classes-built measurement x3, the tier-engaged sat negative control and a both-year moat |
/// | `frozen_dual_bab_search` | absent | absent | #wide-fd-search (ENGINE_PROGRAM M4) ships DARK on BOTH (`f64`, bit-identical): `cpu_f32` / `cuda` / `auto` hand each frozen-dual level's scoring to an injected untrusted backend whose proposals the tier certifies in f64 and splits otherwise. Inert unless `frozen_dual_bab` is armed; arming needs the yaml-route `[bbe] armed=1 backend= probe=ok` proof, the S1 certified nodes/s, the refuted/claimed rate, a ctl-vs-armed x3 per category and a both-year moat |
/// | `tree_mode` (+ `tree_wave_batch`, `tree_wave_cert_threads`) | absent | absent | #wave-beta-bab (ENGINE_PROGRAM M6) ships DARK on BOTH (`legacy`, bit-identical): `wave` runs the margin-row outer frame as waves of injected untrusted output-only beta-CROWN search whose claims a private pool certifies with the lane's own Outward domain evaluation (frozen -> clip -> y-pack ladder; a refuted claim is split, never closed) behind the same path-based cover gate. Arming needs the yaml-route `[bbe] phase=wave armed=1 backend=` proof, the certified closes/s, the refuted/claimed rate (<= 20%), a ctl-vs-armed x3 per category with 0 banked rows lost, and a both-year moat |
/// | `root_box_source` | absent | absent | #root-box-source (ENGINE_PROGRAM M5b) ships DARK on BOTH (`tableau`, bit-identical): `patch_backward` builds every root walk's trunk boxes from a certified IBP reference plus a certified #bi-patch receptive-field backward over the reference-unstable neurons, intersected shrink-only, instead of the forward tableau (20.4-22.0 s of tiny's root). It cannot produce clip rows, Tier-0 retained rows or the f32 tableau walk, so it is REFUSED at validation next to `clip_rows` (cifar100), `root_f32` (both) or `tableau_f32` (tiny) and at runtime on any other route (`[root-box] refused reason=`); it subsumes backward_interm. Arming therefore first needs an A/B that attributes dropping those keys, then the yaml-route `[root-box] source=patch_backward` proof, the root-wall measurement against the kill line (certify_s > 0.5 x the tableau build keeps the tableau), a ctl-vs-armed x3 per category and a both-year moat |
/// | `root_search` | absent | absent | #bbe-joint-root (ENGINE_PROGRAM M5a) ships DARK on BOTH (`ascent`, bit-identical): `joint_f32` swaps #root-certify-once's untrusted ascent for the injected ny-bbe joint-alpha root search, whose slopes still reach the certified root only through the ingest gate and one certified evaluation. Inert unless `root_certify_once` is armed; arming needs the yaml-route `[bbe] phase=joint_root` proof, the S3 measurement, a ctl-vs-armed x3 per category and a both-year moat |
/// | `frozen_dual_refine_steps` | `4` | `4` | User-directed category default: untrusted multi-cut and coordinate search inside the bounded frozen-dual tier; every hint passes the unchanged node certifier and only strict certified improvements are retained. Competition gain remains unmeasured; generic default stays 0 |
/// | `root_certify_once` (+ `_iters`, `_refresh`, `_secs`) | absent | absent | #root-certify-once (Phase-0.5 root alpha, search-untrusted / certify-once) ships DARK on BOTH: armed, it runs the root ascent uncertified on a Parity scratch after every existing root-alpha owner and publishes ONE certified evaluation only if it strictly beats today's certified root, so it can only add root wall unless `alpha_opt`/`alpha_secs` are lowered in the same arm. Arming needs a ctl-vs-armed x3 per category (with the certified ascent kept AND with it replaced), the sat negative control and a both-year moat |
/// | `ibp_seed` | absent | absent | #mr-ibp-seed ships DARK on BOTH, and BY CONSTRUCTION it cannot engage on either: both categories' trunks are residual resnets, and phase 1 declines residual trunks (`declined=residual`, byte-identical). Its target is `challenging_certified_training_2026`'s IBP-trained eps8 CNN7 chain; arming it there needs the GAP conversion x3 (5323, 3144w, 9352, 1031), the 8090/8397 negative controls and the cct moat |
/// | `concurrent_threads` | absent | absent | unmeasured lane-width seam; omission uses the shared global pool without a verifier request, or derives only from observed host-minus-actual-global width with one; explicit `0` forces shared global |
#[test]
fn margin_row_wall_preset_field_diff_is_deliberate() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = repo_root.join("configs/vnncomp25");
    let cifar = load_preset(&dir.join("cifar100_2024.yaml")).unwrap();
    let tiny = load_preset(&dir.join("tinyimagenet_2024.yaml")).unwrap();

    assert_eq!(
        margin_row_fields_set(&cifar.margin_row),
        vec![
            "adaptive_reserve",
            "concurrent",
            "root_f32",
            // #mr-stack + #mr-errlane ARMED 2026-09-26 (scratchpad/mrab): converts
            // idx_2050 and idx_4921 3/3 vs control 0/3; only the PAIR ships.
            "domain_stack",
            "err_envelope",
            "k_head",
            "k_trunk",
            "frozen_dual_bab",
            "frozen_dual_refine_steps",
            "alpha_opt",
            // #alpha-cols-1 ARMED 2026-09-22 (commit 6b52dcdc5): aligns the
            // ascent's gradient with its MIN-column objective.
            "alpha_cols",
            "clip_interm",
            "clip_rows",
            // #dom-alpha-closure-gate ARMED 2026-09-01: the closure ladder now
            // escalates on the lane's own child-closure fraction, and bab.rs fires
            // the per-domain alpha ascent at TIER_BETA_DEEP. 79-row moat vs shipped:
            // held 79, lost 0, flips 0; the gate fired on ONE row. Converts
            // idx_5127_sidx_993 (certified, reproduced 5/5).
            "closure_max_tier",
            "interm_alpha",
            "column_retire",
            "closure_adaptive",
            "branch_reduce_max",
            "parent_lift",
            "backward_interm",
            "bi_secs",
        ],
        "cifar100 margin_row field set changed; update the diff table above"
    );
    assert_eq!(
        margin_row_fields_set(&tiny.margin_row),
        vec![
            "reserve_secs",
            "concurrent",
            "root_f32",
            "tableau_f32",
            // #mr-stack + #mr-errlane ARMED 2026-09-26 with cifar100: +54-106% tree
            // expansions on tiny idx_2146/7943; the full-category moat decides it.
            "domain_stack",
            "err_envelope",
            // #k-trunk-narrow, armed on tiny's OWN evidence 2026-09-16 (see
            // `cifar100_branch_width_is_pinned_and_scoped`): `trunk_split=0`
            // over 214 splits while `score_candidates` is this category's
            // LARGEST phase, +49% expansions over 5 paired rows, and one
            // control-verified conversion with 18 banked unsats clean.
            "k_trunk",
            "frozen_dual_bab",
            "frozen_dual_refine_steps",
            // #tiny-alpha-aligned ARMED 2026-09-22 (commit c8230d7f5): the root
            // ascent with the aligned gradient and the unpinned cost cap.
            // Marginal rows 7/10 armed vs 1/10 shipped (p=0.0099).
            "alpha_opt",
            "alpha_secs",
            "alpha_cols",
            "backward_interm",
        ],
        "tinyimagenet margin_row field set changed; update the diff table above"
    );

    // Values, not just presence.
    assert_eq!(cifar.margin_row.adaptive_reserve, Some(true));
    assert_eq!(cifar.margin_row.k_head, Some(16));
    assert_eq!(cifar.margin_row.k_trunk, Some(16));
    assert_eq!(cifar.margin_row.bi_secs, Some(40));
    assert_eq!(cifar.margin_row.alpha_cols, Some(1));
    assert_eq!(cifar.margin_row.domain_stack, Some(true));
    assert_eq!(tiny.margin_row.domain_stack, Some(true));
    assert_eq!(tiny.margin_row.alpha_opt, Some(true));
    assert_eq!(tiny.margin_row.alpha_cols, Some(1));
    assert_eq!(tiny.margin_row.alpha_secs, Some(35.0));
    assert_eq!(tiny.margin_row.reserve_secs, Some(0));
    // #backward-interm budget is CIFAR-ONLY and the asymmetry is deliberate.
    // A transplant to tinyimagenet was REVERTED before it was ever scored (the
    // yaml at tinyimagenet_2024.yaml:138 carries the reasoning and the
    // pre-registered A/B), so the two presets are asserted asymmetrically,
    // which is the whole point of this test.
    //
    // THE REVERT STANDS, AND IT HAS NOW BEEN MEASURED DIRECTLY — but both
    // justifications that stood here failed, the second by MISREADING THIS
    // PHASE'S OWN TELEMETRY:
    //
    // (1) The headroom argument (60 of 64 banked unsats publish at 89.5-97.1 s
    //     of 100 s) argued about the LANE, not this phase.
    // (2) The replacement said the phase "overruns its armed budget 2.11x
    //     median (195/195 logs, exhausted=true 195/195)" so "`bi_secs` does
    //     not control this phase". That read `[backward-interm] done: secs=`
    //     as phase cost; THAT CLOCK STARTS AT THE ROOT BUILD (`BiState::t0`),
    //     so it is build wall. Direct measurement 2026-08-24 (idx_5451,
    //     official 100 s): the phase's own compute is 2.83 s of a 48.9 s
    //     build, one 256-column pass is 0.64 s, and `exhausted=true` means
    //     the FORWARD BUILD spent the 20 s clock — the phase is STARVED at
    //     trunk layer 2 of ~10, not greedy. `work_secs=` on the done line now
    //     reports phase compute so the two cannot be confused again.
    //
    // The key then got its dose-response ON THIS CATEGORY
    // (docs/BI_DOSE_RESPONSE_5185_2026-08-24.md): the higher dose converted
    // banked-timeout idx_5185 only NONDETERMINISTICALLY (1/4; validation 0/3).
    // idx_9611 failed under the off, 20 s, and higher-dose arms alike on that
    // dirty binary, so it cannot be attributed to the dose. Tableau-f32 later
    // changed the root timing frame. Tiny stays absent pending a clean current-
    // binary 20-vs-30 A/B with the versioned execution receipt.
    assert_eq!(cifar.margin_row.bi_secs, Some(40), "cifar100 sets bi_secs");
    assert_eq!(
        tiny.margin_row.bi_secs, None,
        "tinyimagenet must NOT set bi_secs: the higher-dose 5185 result validated 0/3 \
         after one success, and current tableau-f32 timing needs a clean 20-vs-30 A/B \
         (docs/BI_DOSE_RESPONSE_5185_2026-08-24.md)"
    );
    let wall_presets = [
        ("cifar100_2024", &cifar.margin_row),
        ("tinyimagenet_2024", &tiny.margin_row),
    ];
    for (name, m) in wall_presets {
        assert_eq!(m.concurrent, Some(true), "{name} arms the concurrent lane");
        assert_eq!(m.root_f32, Some(true), "{name} arms the sound f32 root");
        assert_eq!(
            m.frozen_dual_bab,
            Some(true),
            "{name} enables the bounded tier"
        );
        assert_eq!(
            m.frozen_dual_refine_steps,
            Some(4),
            "{name} requests four certified refinement steps"
        );
        assert_eq!(
            m.frozen_dual_bab_node_cap, None,
            "{name} uses the automatic node ceiling"
        );
        assert_eq!(
            m.frozen_dual_bab_secs, None,
            "{name} uses bounded automatic mode"
        );
        assert_eq!(m.lru, None, "{name}: LRU stays on its measured default 64");
        assert_eq!(
            m.parallel, None,
            "{name}: parallel frontier stays on the measured all-worker default"
        );
        assert_eq!(
            m.backward_interm,
            Some(true),
            "{name} arms #backward-interm"
        );
        assert_eq!(
            m.k_adaptive, None,
            "{name}: typed delivery remains deliberately unarmed"
        );
        assert_eq!(
            m.epoch_tier0, None,
            "{name}: Tier-0 ranker remains default-off"
        );
        assert_eq!(
            m.epoch_universe, None,
            "{name}: Tier-0 universe stays on its declaration default"
        );
        assert_eq!(
            m.epoch_retain, None,
            "{name}: retention stays on its declaration default"
        );
        assert_eq!(m.epoch_depth, None, "{name}: epochs remain default-off");
        assert_eq!(
            m.err_envelope,
            Some(ny_propagate::margin_row::ErrLane::Envelope),
            "{name}: #mr-errlane ships armed as a pair with domain_stack (2026-09-26 A/B)"
        );
        assert_eq!(
            m.epoch_attempts, None,
            "{name}: epoch attempts stay defaulted"
        );
        assert_eq!(
            m.epoch_parent_topk, None,
            "{name}: parent refresh remains dark"
        );
        // `bi_secs` is cifar-only again: 3ce02e38a reverted tinyimagenet's
        // unmeasured 40 before it was ever scored (60 of its 65 banked unsats
        // publish at 89.5-97.1 s of 100 s, so extra root seconds come straight
        // out of publication headroom). That commit left this test asserting
        // the withdrawn value, so `-p ny-cli --bins` has not compiled-and-passed
        // since; repaired here rather than left as ambient red.
        assert_eq!(
            cifar.margin_row.bi_secs,
            Some(40),
            "cifar100 keeps the measured 40 s backward-interm budget"
        );
        assert_eq!(
            tiny.margin_row.bi_secs, None,
            "tinyimagenet rides the 20 s default (3ce02e38a)"
        );
        let _ = m.bi_secs;
        assert_eq!(m.k_adaptive, None, "{name}: the ratchet is still env-only");
        assert_eq!(
            m.early_exit, None,
            "{name}: early exit rides its ON default"
        );
        assert_eq!(
            m.verifier_threads, None,
            "{name}: #verifier-thread-cap is MEASURED NEGATIVE (2026-09-17) — \
             keep it unset. 9 runs on tinyimagenet idx_4330_sidx_3504, 3 arms x \
             3 reps, arm order rotated, engagement 9/9, exp=22 on every run \
             (identical trees): lane 34.87s on the shared 20-wide pool, 35.33s \
             at a private 18, 37.57s at a private 16. Lane time is monotone in \
             LANE width, not in verifier contention — the verifier runs on \
             device: wgpu and its host-CPU appetite is under two threads\u{2019} \
             worth, so partitioning the pool only takes threads off the lane"
        );
        assert_eq!(
            m.concurrent_threads, None,
            "{name}: concurrent lane width is unmeasured; omission preserves derived/global policy"
        );
        // #clip-stateful DEFAULT OFF pin. All three Clip-and-Verify keys are
        // newly DELIVERABLE (they were env-only, so no scored run could ever
        // have armed them) and all three ship DARK. Arming any of them is a
        // separate, measured decision — the depth-indexed gating-bound A/B on
        // idx_1298 and idx_4597 is pre-registered and unrun.
        assert_eq!(
            m.clip_stateful, None,
            "{name}: stateful clip accumulation ships dark"
        );
        // #clip-head DEFAULT OFF pin. MoatRisk::High and UNMEASURED: the
        // halfspace is a necessary condition of its split only because the
        // certified slack is subtracted, so this key ships dark until the
        // pre-registered engagement proof (`[clip-head] armed` plus a `call=`
        // line with `hs>0`, zero `NY_*` in the environment) and the wall-hugger
        // moat have both run. Arming it removes this pin, the PromotedKey and
        // the unarmed-promotions row, in the commit carrying the measurement.
        assert_eq!(
            m.clip_head, None,
            "{name}: head-split clip halfspaces ship dark until measured"
        );
        // #branch-reduceop and #interm-alpha are cifar-only, so they are
        // asserted per preset below rather than across the loop.
        let _ = m.branch_reduce_max;
        let _ = m.interm_alpha;
        let _ = m.parent_lift;
    }

    // #branch-reduceop + #interm-alpha: ARMED on cifar100 2026-08-30, dark on
    // tinyimagenet. The pair is the campaign's first measured conversion —
    // idx_5242_sidx_1208 timeout -> unsat, 5/5 reps, with neither lever
    // converting alone (base timeout 3/3, reduceop-only timeout 6/6). The
    // arming gate was the FULL bank: all 124 banked-timeout rows plus all 47
    // banked-unsat rows, engagement 171/171 armed so the nulls are real, and
    // the moat came back 47/47 unsat with zero breaks. The yaml route itself
    // was PROVEN with zero NY_* in the environment (control: no
    // [interm-alpha] line, reduceop mode=min, timeout; armed: armed=1,
    // mode=max 10/10, unsat) — required because run_instance.sh exports no
    // NY_* and this route measured DARK on 2026-08-27 (space.rs:1971).
    assert_eq!(
        cifar.margin_row.branch_reduce_max,
        Some(true),
        "cifar100 arms #branch-reduceop (De Palma eq. 22, abc's own tuned choice here)"
    );
    assert_eq!(
        cifar.margin_row.interm_alpha,
        Some(2),
        "cifar100 arms #interm-alpha at the measured dose 2"
    );
    // tinyimagenet is NOT armed: 60 of its 65 banked unsats publish at
    // 89.5-97.1 s of 100 s, so an extra per-start-node pass comes straight out
    // of publication headroom. Arming it needs its own moat sweep.
    assert_eq!(
        tiny.margin_row.branch_reduce_max, None,
        "tinyimagenet keeps #branch-reduceop dark pending its own sweep"
    );
    assert_eq!(
        tiny.margin_row.interm_alpha, None,
        "tinyimagenet keeps #interm-alpha dark pending its own sweep"
    );
    // #branch-parent-lift: ARMED on cifar100 2026-08-30. It converts
    // idx_8739_sidx_1540, a row the reduceop+interm_alpha pair alone does not,
    // at ZERO wall cost, with the moat coming back 47/47 unsat and engagement
    // on 124/124 pool rows. Dark on tinyimagenet: it changes tree shape, and
    // 63 of that benchmark's 74 banked unsats publish within 10 s of the wall,
    // so it needs its own moat sweep there.
    assert_eq!(
        cifar.margin_row.parent_lift,
        Some(true),
        "cifar100 arms #branch-parent-lift"
    );
    assert_eq!(
        tiny.margin_row.parent_lift, None,
        "tinyimagenet keeps #branch-parent-lift dark pending its own sweep"
    );
    // #clip-and-verify: ARMED on cifar100 2026-08-31. It converts
    // idx_1798_sidx_1061 (5 of 8 reps, certified) at ZERO wall cost, with the
    // moat 47/47 including all five thin-margin rows. Dark on tinyimagenet: no
    // clip row has ever been measured there and its preset forbids blind cifar
    // transfers by name.
    assert_eq!(
        cifar.margin_row.clip_interm,
        Some(true),
        "cifar100 arms #clip-and-verify intermediate tightening"
    );
    assert_eq!(
        cifar.margin_row.clip_rows,
        Some(true),
        "cifar100 retains the clip halfspace rows (clip_interm is inert without them)"
    );
    assert_eq!(
        tiny.margin_row.clip_interm, None,
        "tinyimagenet keeps clip dark pending its own measurement"
    );

    // The diff is COMPLETE: no other shipped preset carries a margin_row key,
    // so nothing outside this table can inherit one of these decisions.
    //
    // #cct-margin-row-admission: configs/vnncomp26 SHADOWS configs/vnncomp25
    // and vnncomp26 presets carry lane settings, so scan both years. The cct
    // preset and its patches-wiring canary are pinned field-for-field (and
    // canary-verbatim) by the cct test. SafeNLP's two explicitly directed
    // default-on switches are pinned by the head-directional test below.
    const PINNED_ELSEWHERE: &[&str] = &[
        "challenging_certified_training_2026",
        "challenging_certified_training_2026_patcheswiring_canary",
        "safenlp_2024",
    ];
    let mut others: Vec<String> = Vec::new();
    let mut pinned_elsewhere_seen: Vec<String> = Vec::new();
    for year_dir in [dir, repo_root.join("configs/vnncomp26")] {
        for entry in
            fs::read_dir(&year_dir).unwrap_or_else(|e| panic!("read {}: {e}", year_dir.display()))
        {
            let path = entry.expect("dir entry").path();
            if path.extension().is_none_or(|e| e != "yaml") {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("preset stem")
                .to_string();
            if stem == "cifar100_2024" || stem == "tinyimagenet_2024" {
                continue;
            }
            let cfg = load_preset(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
            let set = margin_row_fields_set(&cfg.margin_row);
            if set.is_empty() {
                continue;
            }
            if PINNED_ELSEWHERE.contains(&stem.as_str()) {
                pinned_elsewhere_seen.push(stem);
                continue;
            }
            others.push(format!("{}: {set:?}", path.display()));
        }
    }
    assert!(
        others.is_empty(),
        "another preset now sets margin_row keys, so the diff above is no longer \
         the whole story: {others:?}"
    );
    pinned_elsewhere_seen.sort();
    assert_eq!(
        pinned_elsewhere_seen, PINNED_ELSEWHERE,
        "the sanctioned vnncomp26 margin_row carriers moved or disarmed; update \
         PINNED_ELSEWHERE and its dedicated field-for-field pin tests together"
    );
}

#[test]
fn safenlp_head_directional_defaults_are_the_exact_shipped_margin_row_block() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp26/safenlp_2024.yaml")).unwrap();
    assert_eq!(
        margin_row_fields_set(&preset.margin_row),
        ["head_groups", "head_directional_cuts"],
        "the shipped SafeNLP margin block must contain only its two directed switches"
    );
    assert_eq!(preset.margin_row.head_groups, Some(true));
    assert_eq!(preset.margin_row.head_directional_cuts, Some(true));
}

/// #tinyimagenet-alloc-parity-no-go: the CIFAR allocation trio regressed the
/// official-budget TinyImageNet sample from 5/10 to 2/10, including three lost
/// SAT rows. Pin the measured TinyImageNet baseline so a sibling-preset transfer
/// cannot silently override category-specific evidence again.
#[test]
fn tinyimagenet_2024_rejects_the_regressive_cifar100_allocation_trio() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tiny = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();

    assert_eq!(tiny.bab.root_alpha_cap_secs, None);
    assert_eq!(tiny.bab.phase_budget.disjunctive_pgd_fraction, Some(0.40));
    assert_eq!(tiny.margin_row.adaptive_reserve, None);
    assert_eq!(
        tiny.margin_row.release_frac, None,
        "no margin-row reserve override should ride along with the rejected transfer"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &tiny).unwrap();
    assert_eq!(config.root_alpha_cap_secs, None);
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.40);

    // Bound-quality knobs remain independently scoped and are not part of this rollback.
    assert_eq!(tiny.bab.root_crown_interm_dense_head, None);
    assert_eq!(tiny.bab.clip.interm_domain, None);
    assert_eq!(tiny.bab.branching.kfsb_multi, None);
}

#[test]
fn cersyve_preset_enables_complete_clipping() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();

    assert_eq!(
        preset.bab.clip.clip_type.as_deref(),
        Some("complete"),
        "cersyve preset should enable complete clipping"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.input_clip_type,
        InputClipType::Complete,
        "complete clip_type should propagate to BetaCrownConfig"
    );
}

#[test]
fn lsnc_relu_stays_relaxed_while_cersyve_is_complete() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();

    assert_eq!(
        lsnc_relu.bab.clip.clip_type.as_deref(),
        None,
        "lsnc_relu should rely on the default relaxed clip type"
    );
    assert_eq!(
        cersyve.bab.clip.clip_type.as_deref(),
        Some("complete"),
        "cersyve should keep its explicit complete clip type"
    );

    let mut lsnc_config = BetaCrownConfig::default();
    apply_preset(&mut lsnc_config, &lsnc_relu).unwrap();
    assert_eq!(
        lsnc_config.input_clip_type,
        InputClipType::Relaxed,
        "lsnc_relu should stay on the relaxed runtime clip path"
    );

    let mut cersyve_config = BetaCrownConfig::default();
    apply_preset(&mut cersyve_config, &cersyve).unwrap();
    assert_eq!(
        cersyve_config.input_clip_type,
        InputClipType::Complete,
        "cersyve should stay on the complete runtime clip path"
    );
}

#[test]
fn nn4sys_preset_matches_reference_input_split_schedule_4354() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/nn4sys.yaml")).unwrap();

    // pgd_order: skip — PGD disabled for nn4sys
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("skip"));
    assert_eq!(
        preset.bab.phase_budget.post_bab_pgd_fraction,
        Some(0.0),
        "disabled NN4SYS PGD must not retain an engine tail reservation"
    );
    assert_eq!(
        preset.bab.phase_budget.vnncomp_post_bab_attack,
        Some(false),
        "the independent VNN-COMP post-BaB wrapper must follow the explicit proof-only policy"
    );
    assert_eq!(preset.bab.clip.relaxed, Some(true));
    assert_eq!(preset.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(preset.bab.branching.input_split.enable, Some(true));
    assert_eq!(preset.bab.branching.input_split.sb_coeff_thresh, Some(0.1));
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(
        config.enable_relaxed_clip,
        "nn4sys preset should propagate relaxed clipping into BetaCrownConfig"
    );
    assert_eq!(config.input_clip_type, InputClipType::Relaxed);
    assert_eq!(config.input_split_coeff_thresh, 0.1);
    assert!(config.reorder_bab);
    assert_eq!(
        config.phase_budget.post_bab_pgd_fraction, 0.0,
        "the applied NN4SYS engine schedule must give its full slice to proof"
    );
}

/// #nn4sys-seb-dark (B8): the shipped nn4sys preset arms Saturation-Escape
/// Branching through the TYPED key, so the dual-pool input brancher (and the
/// disjunctive precheck budget cap that funds it) is reachable without any
/// ambient `NY_SAT_ESCAPE_BRANCH` export — while a preset that never names the
/// key keeps the dark default byte-identically.
#[test]
fn nn4sys_preset_threads_sat_escape_branch_and_absence_stays_dark() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    // Key parses from the shipped nn4sys preset. It is `false`: `2e4af94b`
    // DISARMED it on a scored-wall-time A/B over the first 12 `_dual` rows —
    // armed proved 1/12 unsat, unarmed 8/12, so arming LOSES 7 rows. Arming
    // caps the per-clause precheck slice at `.min(0.5)` where the unarmed path
    // takes `.max(0.95)` (verify/disjunctive.rs:549-558), and on the mscn dual
    // band that precheck lane is what closes the disjuncts.
    //
    // This test previously asserted `Some(true)` and went stale the moment the
    // preset was corrected — it was pinning an intent the measurement refuted.
    // What is worth pinning is the WIRING in both directions, which is what it
    // now does.
    let nn4sys = load_preset(&repo_root.join("configs/vnncomp25/nn4sys.yaml")).unwrap();
    assert_eq!(
        nn4sys.bab.branching.input_split.sat_escape_branch,
        Some(false),
        "nn4sys deliberately DISARMS SEB (2e4af94b); NY_SAT_ESCAPE_BRANCH=1 still \
         forces it for experiments"
    );

    // …and an explicit `false` threads through to the engine field as disarmed.
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &nn4sys).unwrap();
    assert!(
        !config.sat_escape_branch,
        "an explicit `false` must leave BetaCrownConfig::sat_escape_branch disarmed"
    );

    // Absent key = today's behavior: another input-split preset without the
    // key leaves the field at its dark default.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(acasxu.bab.branching.input_split.sat_escape_branch, None);
    let mut acasxu_config = BetaCrownConfig::default();
    apply_preset(&mut acasxu_config, &acasxu).unwrap();
    assert!(
        !acasxu_config.sat_escape_branch,
        "a preset that does not name the key must stay byte-identical (dark default)"
    );
}

#[test]
fn cersyve_and_lsnc_relu_presets_propagate_reference_input_split_tuning() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    assert_eq!(cersyve.bab.branching.input_split.enable, Some(true));
    assert_eq!(
        cersyve.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(cersyve.bab.branching.input_split.sb_sum, Some(true));
    assert_eq!(
        cersyve.bab.branching.input_split.touch_zero_score,
        Some(0.1)
    );
    assert_eq!(
        cersyve.bab.branching.input_split.override_parallel,
        Some(true)
    );

    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    assert_eq!(lsnc_relu.bab.branching.input_split.enable, Some(true));
    assert_eq!(
        lsnc_relu.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(lsnc_relu.bab.branching.input_split.sb_sum, Some(true));
    assert_eq!(
        lsnc_relu.bab.branching.input_split.touch_zero_score,
        Some(0.1)
    );
    assert_eq!(lsnc_relu.bab.branching.input_split.override_parallel, None);

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cersyve).unwrap();
    assert!(config.input_split_sb_sum);
    assert_eq!(config.input_split_coeff_thresh, 1.0e-2);
    assert_eq!(config.input_split_touch_zero_score, 0.1);
    assert!(config.input_split_override_parallel);
}

#[test]
fn cersyve_and_lsnc_relu_presets_enable_pgd_alpha_scale() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    for preset_name in ["cersyve", "lsnc_relu"] {
        let preset =
            load_preset(&repo_root.join(format!("configs/vnncomp25/{preset_name}.yaml"))).unwrap();
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert_eq!(
            config.pgd_optimizer,
            PgdOptimizer::SignedGradient,
            "{preset_name} should switch PGD to signed-gradient when pgd_alpha_scale=true"
        );
        assert_eq!(
            config.pgd_alpha_mode,
            PgdAlphaMode::InputRangeScaled(0.01),
            "{preset_name} should propagate pgd_alpha_scale into the runtime PGD alpha mode"
        );
    }
}

#[test]
fn cora_2024_preset_enables_clip_in_alpha_crown() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cora_2024.yaml")).unwrap();

    assert_eq!(preset.bab.clip.in_alpha_crown, Some(true));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.clip_in_alpha_crown,
        "cora_2024 preset should propagate clip_in_alpha_crown into BetaCrownConfig"
    );
}

#[test]
fn cora_and_metaroom_presets_explicitly_disable_interm_transfer_4358() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cora = load_preset(&repo_root.join("configs/vnncomp25/cora_2024.yaml")).unwrap();
    let metaroom = load_preset(&repo_root.join("configs/vnncomp25/metaroom_2023.yaml")).unwrap();

    assert_eq!(cora.bab.interm_transfer, Some(false));
    assert_eq!(metaroom.bab.interm_transfer, Some(false));

    let mut cora_config = BetaCrownConfig::default();
    apply_preset(&mut cora_config, &cora).unwrap();
    assert!(
        !cora_config.enable_interm_transfer,
        "cora_2024 preset should override the enabled default and keep interm_transfer disabled"
    );

    let mut metaroom_config = BetaCrownConfig::default();
    apply_preset(&mut metaroom_config, &metaroom).unwrap();
    assert!(
        !metaroom_config.enable_interm_transfer,
        "metaroom_2023 preset should keep its explicit interm_transfer=false override"
    );
}

#[test]
fn metaroom_preset_propagates_dd_zonotope_admission_overrides() {
    // #metaroom-ddzono: the scored entry point sets no env vars, so the
    // category preset is the only way the dd-zonotope admission caps can
    // reach `DdZonoConfig` for metaroom's 5376-input / k=161 instances.
    //
    // MECHANISM test: an inline section must propagate every knob.
    let inline: PresetConfig = serde_yaml::from_str(
        r#"
dd_zonotope:
  min_input_numel: 5000
  max_k: 192
  max_generators: 4096
  interm_intersect: true
"#,
    )
    .expect("dd_zonotope preset section must parse");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &inline).unwrap();
    assert_eq!(config.dd_zonotope_min_input_numel, Some(5000));
    assert_eq!(config.dd_zonotope_max_k, Some(192));
    assert_eq!(config.dd_zonotope_max_generators, Some(4096));
    assert_eq!(config.dd_zonotope_collect_interm, Some(true));

    // SHIPPED-STATE test: the metaroom yaml keeps the section DISARMED until
    // the near-wall A/B (spec_idx_28-class rows vs the ~18s/row cost) and the
    // campaign-GPU interm-intersect value test are both recorded — the
    // population-evidence discipline. Re-point these at Some(..) ONLY together
    // with those measurements.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let metaroom = load_preset(&repo_root.join("configs/vnncomp25/metaroom_2023.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &metaroom).unwrap();
    assert_eq!(config.dd_zonotope_min_input_numel, None);
    assert_eq!(config.dd_zonotope_max_k, None);
    assert_eq!(config.dd_zonotope_max_generators, None);
    assert_eq!(config.dd_zonotope_collect_interm, None);

    // Every other shipped preset leaves the section absent, which must remain
    // byte-identical to the built-in caps: apply changes nothing.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    let mut untouched = BetaCrownConfig::default();
    apply_preset(&mut untouched, &acasxu).unwrap();
    assert_eq!(untouched.dd_zonotope_min_input_numel, None);
    assert_eq!(untouched.dd_zonotope_max_k, None);
    assert_eq!(untouched.dd_zonotope_max_generators, None);
    assert_eq!(untouched.dd_zonotope_collect_interm, None);
}

#[test]
fn metaroom_preset_arms_the_cz_hybrid_instance_allowlist() {
    // #cz-hybrid: the scored entry point sets no env vars, so the typed
    // preset section is the ONLY route by which the wired CZ+Box verdict lane
    // (`commands::cz_metaroom_wired`) can engage. The boundary's own unit
    // tests cover the admission mechanism; this pins the SHIPPED state:
    // metaroom_2023 arms exactly the qualified spec_idx_119 instance, and
    // every other preset leaves the section absent = lane dark.
    //
    // Changing either name here without re-qualifying the sealed replays
    // would make the lane engage on an instance it must fail closed on — the
    // catch_unwind boundary makes that a wasted-budget bug rather than a
    // soundness bug, but do not do it.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let metaroom = load_preset(&repo_root.join("configs/vnncomp25/metaroom_2023.yaml")).unwrap();
    assert_eq!(
        metaroom.cz_hybrid.only_onnx.as_deref(),
        Some("6cnn_ry_39_6_no_custom_OP.onnx")
    );
    assert_eq!(
        metaroom.cz_hybrid.only_vnnlib.as_deref(),
        Some("spec_idx_119_eps_0.00000436.vnnlib")
    );

    // Every other shipped preset must leave the lane dark.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(acasxu.cz_hybrid.only_onnx, None);
    assert_eq!(acasxu.cz_hybrid.only_vnnlib, None);
}

#[test]
fn cersyve_and_lsnc_relu_presets_propagate_min_batch_fill_ratio() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cersyve).unwrap();
    assert_eq!(
        config.min_batch_fill_ratio, 0.1,
        "cersyve min_batch_fill_ratio should match reference (0.1)"
    );

    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &lsnc_relu).unwrap();
    assert_eq!(
        config.min_batch_fill_ratio, 0.0,
        "lsnc_relu min_batch_fill_ratio should match reference (0.0)"
    );
}

#[test]
fn acasxu_2023_preset_matches_hardcoded_config() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();

    // Verify YAML fields parsed correctly
    assert_eq!(
        preset.solver.bound_prop_method.as_deref(),
        Some("crown"),
        "acasxu preset should use plain CROWN (not alpha-CROWN)"
    );
    assert_eq!(preset.solver.batch_size, Some(16384));
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("after"));
    assert_eq!(preset.attack.pgd_restarts, Some(10000));
    assert_eq!(preset.bab.branching.method, Some("input".to_string()));
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(preset.bab.clip.relaxed, Some(true));

    // Apply and verify config matches BetaCrownConfig::acas_xu()
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    let reference = BetaCrownConfig::acas_xu();
    assert_eq!(
        config.use_alpha_crown, reference.use_alpha_crown,
        "preset should disable alpha-CROWN like acas_xu() config"
    );
    assert_eq!(
        config.batch_size, reference.batch_size,
        "preset batch_size should match acas_xu() config"
    );
    assert_eq!(
        config.branching_heuristic, reference.branching_heuristic,
        "preset branching should match acas_xu() config"
    );
    assert_eq!(
        config.reorder_bab, reference.reorder_bab,
        "preset input-split reordering should match acas_xu() config"
    );
    assert_eq!(
        config.input_split_depth, reference.input_split_depth,
        "preset multi-dim input_split_depth should match acas_xu() config"
    );
    assert_eq!(
        config.enable_relaxed_clip, reference.enable_relaxed_clip,
        "preset relaxed_clip should match acas_xu() config"
    );
    assert_eq!(
        config.enable_pgd_attack, reference.enable_pgd_attack,
        "preset pgd_attack should match acas_xu() config"
    );
    assert_eq!(
        config.pgd_restarts, reference.pgd_restarts,
        "preset pgd_restarts should match acas_xu() config"
    );
}

/// vit_2023 regression pin: softmax "complex" stays DISABLED in the shipped
/// preset — measured net-negative at vit scale (00338ba) — while the Phase A
/// alpha/share_alphas knobs stay in place and apply_preset succeeds cleanly.
#[test]
fn vit_2023_preset_keeps_softmax_complex_disabled() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/vit_2023.yaml")).unwrap();

    // softmax: complex is deliberately NOT set — measured net-negative at vit
    // scale (00338ba: ~3x per-iteration cost, best bound flat while the
    // direct-LSE arm improves). The machinery stays in-tree; the preset
    // re-enables it only after the recorded follow-ups land.
    assert_eq!(
        preset.solver.alpha_crown.softmax.as_deref(),
        None,
        "vit_2023 preset keeps softmax-complex disabled (measured net-negative; see 00338ba)"
    );
    assert_eq!(preset.solver.alpha_crown.iterations, Some(50));
    assert_eq!(preset.solver.alpha_crown.lr_alpha, Some(0.5));
    assert_eq!(preset.solver.alpha_crown.share_alphas, Some(true));

    // The softmax field is consumed at model load (graph rewrite), not by
    // BetaCrownConfig; applying the preset must still succeed cleanly.
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.alpha_config.iterations, 50);
    assert_eq!(config.alpha_config.learning_rate, 0.5);

    // #disj-demand (measured 2026-08-26, official 100 s): both vit model
    // families are outside the exact-gradient fragment, so the 5% disjunctive
    // attack slice can only run SPSA over point-IBP forwards and has never
    // produced a counterexample on this family. The typed key must arm the
    // demand-zeroing through the yaml route (no NY_* involved), while the
    // configured fraction is retained as the disarmed fallback.
    assert_eq!(
        preset.bab.phase_budget.demand_sized_disjunctive_pgd,
        Some(true)
    );
    assert!(config.phase_budget.demand_sized_disjunctive_pgd);
    assert!((config.phase_budget.disjunctive_pgd_fraction - 0.05).abs() < 1e-6);
}

/// cgan_2023 root-collection scheduling caps (#cgan-bn11-budget):
/// the preset's per-node CROWN-IBP cap and aggregate α reference-refresh cap
/// flow through to the exact typed configs consumed by the collectors. The
/// per-node floor and refresh fraction retain their built-in defaults.
#[test]
fn cgan_2023_preset_raises_crown_ibp_per_node_cap() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let load_config =
        build_onnx_load_config(&preset).expect("official cGAN loader policy must be admitted");
    assert_eq!(
        load_config.batch_norm_folding_policy(),
        ny_onnx::BatchNormFoldingPolicy::PreserveRaw,
        "official cGAN verdicts must target the authored graph"
    );
    assert!(
        load_config.require_authored_float32_initializers(),
        "official cGAN loading must fail before verification on rewritten FLOAT initializers"
    );
    assert_eq!(
        preset.model.forward_linear_spec_alpha,
        Some(false),
        "the candidate remains typed and dark pending the physical moat"
    );

    assert_eq!(preset.bab.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(preset.bab.crown_ibp_per_node_floor_secs, None);
    // #cgan-abc-parity (2026-08-27): reorder_bab is production - both an abc
    // treatment field and the prerequisite of the exact-domain input clip.
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(preset.bab.branching.input_split.warm_parallel, None);
    assert_eq!(preset.bab.branching.input_split.alpha_iteration, Some(1));
    assert_eq!(preset.bab.branching.input_split.lr_alpha, Some(0.05));
    assert_eq!(preset.bab.alpha_crown.reference_refresh_max_secs, Some(12));
    assert_eq!(preset.bab.alpha_crown.reference_refresh_fraction, None);
    assert_eq!(
        preset
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true)
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(config.crown_ibp_per_node_floor_secs, None);
    assert!(config.reorder_bab);
    assert!(!config.input_split_warm_parallel);
    assert_eq!(config.input_split_alpha_iteration, 1);
    assert_eq!(config.input_split_lr_alpha, 0.05);
    assert_eq!(config.alpha_config.reference_refresh_max_secs, Some(12));
    assert!(config.alpha_config.forward_linear_deadline_fallback_to_ibp);
    assert_eq!(
        config.alpha_config.reference_refresh_fraction,
        ny_propagate::AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION
    );

    let budget = config.crown_ibp_per_node_time_budget();
    assert_eq!(budget.cap_secs, Some(150.0));
    assert_eq!(budget.floor_secs, None);

    // Companion phase-budget levers: retained SAT instances fall to PGD in
    // under 2 s, while the official holdout otherwise spent 85.1 s before BaB
    // and reserved another 10% for an unsuccessful post-BaB attack.
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.10);
    assert_eq!(config.phase_budget.disjunctive_pgd_max_secs, Some(5));
    assert_eq!(config.phase_budget.post_bab_pgd_fraction, 0.0);
    assert_eq!(config.phase_budget.initial_bounds_fraction, 0.45);
    // The forward-linear precheck does not return early: its 0.85 share is
    // SPENT TIME. Measured 2026-08-30 on cgan2026 row 10 (`cGAN_imgSz64_nCh_1`
    // `prop_2`, official 900 s): the phase runs to `elapsed_s=717.884` — 80% of
    // the whole budget, 94% of the 0.85*900 = 765 s ceiling — and returns
    // `verified=0`. A single-variable arm capping it to 150 s leaves BaB's
    // first six `[disjunctive-trajectory]` rows BIT-IDENTICAL, so the phase's
    // output is a pure discard on that row. 0.85 is therefore a REAL grant, not
    // a ceiling; the only absolute bound is `disjunctive_precheck_max_secs`,
    // which both cgan presets now arm at 60 s (see #precheck-abs-cap below).
    // See reports/measured-2026/precheck_discard_2026-08-30/.
    assert_eq!(config.phase_budget.disjunctive_precheck_fraction, 0.85);
}

/// The forward-alpha treatment is a typed, reviewable preset rather than an
/// environment overlay. Normalize its single treatment field and require the
/// full serialized presets to match, so sealed A/B runs cannot accidentally
/// drift a loader, attack, or solver knob.
#[test]
fn cgan_forward_alpha_surrogate_canary_differs_by_exactly_one_typed_field() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_forward_alpha_surrogate_canary.yaml"),
    )
    .unwrap();

    assert_eq!(production.model.forward_linear_spec_alpha, Some(false));
    assert_eq!(canary.model.forward_linear_spec_alpha, Some(true));

    for (name, preset) in [("production", &production), ("canary", &canary)] {
        let load_config = build_onnx_load_config(preset)
            .unwrap_or_else(|error| panic!("{name} cGAN loader policy must be admitted: {error}"));
        assert_eq!(
            load_config.batch_norm_folding_policy(),
            ny_onnx::BatchNormFoldingPolicy::PreserveRaw,
            "{name} cGAN preset must verify the authored graph"
        );
        assert!(
            load_config.require_authored_float32_initializers(),
            "{name} cGAN preset must reject rewritten authored FLOAT initializers"
        );
    }

    let mut normalized_canary = canary;
    normalized_canary.model.forward_linear_spec_alpha = production.model.forward_linear_spec_alpha;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "forward-alpha canary may differ from shipped cGAN only at forward_linear_spec_alpha"
    );
}

/// `#cgan-child-interm-refine`: the measurement canary must differ from the
/// shipped cgan2026 preset by exactly its TWO typed treatment fields — the
/// per-child wall budget and the REUSED `cgan_narrow_target_max_rows` ceiling
/// (#cgan-narrow-target-budget) that makes the child cascade refine only
/// narrow targets. Normalizing both and requiring full serialized equality
/// makes canary drift (the multi-variable-arm failure the narrow-target
/// canary was once caught in) a test failure instead of a silent measurement
/// hazard. The resolver-level apply assertions are necessary but NOT
/// sufficient reachability evidence: the lever's own `[cgan-child-interm-refine]
/// armed` line from a run with the armed yaml and no `NY_*` set is the proof
/// the Measure phase must still collect.
#[test]
fn cgan_child_interm_refine_canary_differs_by_exactly_two_typed_fields() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp26/cgan2026.yaml")).unwrap();
    let canary =
        load_preset(&repo_root.join("configs/vnncomp26/cgan2026_child_interm_refine_canary.yaml"))
            .unwrap();

    assert_eq!(
        production.bab.alpha_crown.cgan_child_interm_refine_max_secs, None,
        "the SHIPPED cgan2026 preset must keep child refinement dark"
    );
    assert_eq!(production.bab.alpha_crown.cgan_narrow_target_max_rows, None);
    assert_eq!(
        canary.bab.alpha_crown.cgan_child_interm_refine_max_secs,
        Some(120.0)
    );
    assert_eq!(
        canary.bab.alpha_crown.cgan_narrow_target_max_rows,
        Some(4096),
        "the child canary must arm the REUSED narrow-target ceiling at the \
         measured value-law boundary; without it every child would pay the \
         wide generator targets the leave-one-out priced at zero"
    );

    let mut normalized_canary = canary.clone();
    normalized_canary
        .bab
        .alpha_crown
        .cgan_child_interm_refine_max_secs = None;
    normalized_canary
        .bab
        .alpha_crown
        .cgan_narrow_target_max_rows = None;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "child-interm-refine canary may differ from shipped cgan2026 only at \
         its two typed treatment fields"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &canary).unwrap();
    assert_eq!(
        config.alpha_config.cgan_child_interm_refine_max_secs,
        Some(120.0)
    );
    assert_eq!(config.alpha_config.cgan_narrow_target_max_rows, Some(4096));
    assert!(
        config.alpha_config.cgan_complete_crown_ibp_root,
        "the child budget is inert by construction without the root arm; the \
         canary must carry both"
    );
}

/// `#cgan-child-seed`: the three child-cascade keys are MEASUREMENT keys. No
/// shipped preset (production or canary, either alpha block) may name them
/// until the commit that carries their measurement arms them, and the scan
/// must actually parse presets or it is a void pass.
#[test]
fn cgan_child_seed_keys_are_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut named = Vec::new();
    let mut parsed = 0usize;
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
            parsed += 1;
            for (block, alpha) in [
                ("solver.alpha_crown", &preset.solver.alpha_crown),
                ("bab.alpha_crown", &preset.bab.alpha_crown),
            ] {
                if alpha.cgan_child_interm_refine_seed_from_parent.is_some()
                    || alpha.cgan_child_interm_refine_parallel.is_some()
                    || alpha.cgan_child_interm_refine_fresh_clip_map.is_some()
                {
                    named.push(format!("{}: {block}", path.display()));
                }
            }
        }
    }
    assert!(
        parsed > 0,
        "no shipped preset was parsed, so this darkness check is VOID, not a pass"
    );
    assert!(
        named.is_empty(),
        "the #cgan-child-seed keys must stay dark in every shipped preset until their \
         measurement lands: {named:?}"
    );
}

/// `#cgan-child-seed`: each typed key reaches its `AlphaCrownConfig` field
/// from yaml, an absent key keeps the dark default, and an unhonourable
/// concurrency request is refused at load instead of silently clamped.
#[test]
fn cgan_child_seed_keys_apply_from_yaml_and_refuse_an_out_of_range_width() {
    let armed: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  alpha_crown:
    cgan_complete_crown_ibp_root: true
    cgan_child_interm_refine_max_secs: 120.0
    cgan_child_interm_refine_seed_from_parent: true
    cgan_child_interm_refine_parallel: 4
    cgan_child_interm_refine_fresh_clip_map: true
"#,
    )
    .expect("armed child-seed preset");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed child-seed preset should apply");
    assert!(
        config
            .alpha_config
            .cgan_child_interm_refine_seed_from_parent
    );
    assert_eq!(config.alpha_config.cgan_child_interm_refine_parallel, 4);
    assert!(config.alpha_config.cgan_child_interm_refine_fresh_clip_map);
    assert_eq!(
        config.alpha_config.cgan_child_interm_refine_max_secs,
        Some(120.0)
    );

    let budget_only: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  alpha_crown:
    cgan_complete_crown_ibp_root: true
    cgan_child_interm_refine_max_secs: 120.0
"#,
    )
    .expect("budget-only preset");
    let mut dark = BetaCrownConfig::default();
    apply_preset(&mut dark, &budget_only).expect("budget-only preset should apply");
    assert!(
        !dark.alpha_config.cgan_child_interm_refine_seed_from_parent,
        "arming the historical prepass alone must not seed it"
    );
    assert_eq!(dark.alpha_config.cgan_child_interm_refine_parallel, 0);
    assert!(!dark.alpha_config.cgan_child_interm_refine_fresh_clip_map);

    let too_wide: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  alpha_crown:
    cgan_child_interm_refine_parallel: 65
"#,
    )
    .expect("out-of-range width still parses");
    let mut refused = BetaCrownConfig::default();
    assert!(
        apply_preset(&mut refused, &too_wide).is_err(),
        "a width above AlphaCrownConfig::MAX_CGAN_CHILD_INTERM_REFINE_PARALLEL must be \
         refused at load"
    );
}

/// `#cgan-child-zono`: the per-child zonotope refresh is a MEASUREMENT key.
/// No shipped preset (production or canary, either alpha block) may name it
/// until the commit that carries its measurement arms it, and the scan must
/// actually parse presets or it is a void pass.
#[test]
fn cgan_child_zono_refresh_is_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut named = Vec::new();
    let mut parsed = 0usize;
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
            parsed += 1;
            for (block, alpha) in [
                ("solver.alpha_crown", &preset.solver.alpha_crown),
                ("bab.alpha_crown", &preset.bab.alpha_crown),
            ] {
                if alpha.cgan_child_zono_refresh.is_some() {
                    named.push(format!("{}: {block}", path.display()));
                }
            }
        }
    }
    assert!(
        parsed > 0,
        "no shipped preset was parsed, so this darkness check is VOID, not a pass"
    );
    assert!(
        named.is_empty(),
        "bab.alpha_crown.cgan_child_zono_refresh must stay dark in every shipped preset \
         until its measurement lands: {named:?}"
    );
}

/// `#cgan-child-zono`: every sub-key reaches `AlphaCrownConfig` from yaml, an
/// absent block keeps the dark default, a partial block keeps the other
/// defaults, and an inadmissible budget (`0` included — it never means
/// "unlimited") is refused at load instead of silently clamped.
#[test]
fn cgan_child_zono_refresh_applies_from_yaml_and_refuses_bad_values() {
    let armed: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  alpha_crown:
    cgan_child_zono_refresh:
      enabled: true
      max_generators: 4096
      max_secs: 2.5
"#,
    )
    .expect("armed zonotope preset");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed zonotope preset should apply");
    let refresh = config.alpha_config.cgan_child_zono_refresh;
    assert!(refresh.enabled);
    assert_eq!(refresh.max_generators, 4096);
    assert_eq!(refresh.max_secs, Some(2.5));

    let enabled_only: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  alpha_crown:
    cgan_child_zono_refresh:
      enabled: true
"#,
    )
    .expect("enabled-only preset");
    let mut partial = BetaCrownConfig::default();
    apply_preset(&mut partial, &enabled_only).expect("enabled-only preset should apply");
    assert!(partial.alpha_config.cgan_child_zono_refresh.enabled);
    assert_eq!(
        partial.alpha_config.cgan_child_zono_refresh.max_generators, 16_384,
        "an unnamed budget keeps the non-binding default"
    );
    assert_eq!(partial.alpha_config.cgan_child_zono_refresh.max_secs, None);

    let mut dark = BetaCrownConfig::default();
    apply_preset(&mut dark, &PresetConfig::default()).expect("empty preset");
    assert!(!dark.alpha_config.cgan_child_zono_refresh.enabled);

    for bad in [
        "max_generators: 0",
        "max_generators: 4",
        "max_generators: 2000000",
        "max_secs: 0.0",
        "max_secs: -1.0",
        "max_secs: .nan",
    ] {
        let yaml = format!("bab:\n  alpha_crown:\n    cgan_child_zono_refresh:\n      {bad}\n");
        let preset: PresetConfig = serde_yaml::from_str(&yaml).expect("bad value still parses");
        let mut refused = BetaCrownConfig::default();
        assert!(
            apply_preset(&mut refused, &preset).is_err(),
            "`{bad}` must be refused at load"
        );
    }
    let unknown: Result<PresetConfig, _> = serde_yaml::from_str(
        "bab:\n  alpha_crown:\n    cgan_child_zono_refresh:\n      budget: 5\n",
    );
    assert!(
        unknown.is_err(),
        "the block is closed: unknown sub-keys are rejected"
    );
}

/// Named V2 test 12: the zonotope refresh reaches the lane through the TYPED
/// preset key. Its plan records where the arming came from, and the key has
/// no environment reader at all, so an armed plan built from yaml must report
/// `Source::Config` — never `Source::LegacyEnv`.
#[test]
fn child_zono_preset_source_is_typed() {
    let armed: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  alpha_crown:
    cgan_child_zono_refresh:
      enabled: true
"#,
    )
    .expect("armed zonotope preset");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed zonotope preset should apply");
    let plan = ny_propagate::zono_core::cgan_child_zono_plan(&config)
        .expect("the yaml key must produce an armed plan");
    assert_ne!(plan.source, ny_levers::Source::LegacyEnv);
    assert_eq!(plan.source, ny_levers::Source::Config);
    assert_eq!(plan.max_generators, 16_384);
    assert!(
        ny_propagate::zono_core::cgan_child_zono_plan(&BetaCrownConfig::default()).is_none(),
        "without the key there is no plan and no work"
    );
}

/// `#fwd-root-screen`: the forward root screen is a MEASUREMENT key. No
/// shipped preset (any year, production or canary) may name either of its
/// keys until the commit carrying the measurement arms ml4acopf, and the scan
/// must actually parse presets or it is a void pass.
#[test]
fn forward_root_screen_is_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut named = Vec::new();
    let mut parsed = 0usize;
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
            parsed += 1;
            if preset.bab.forward_root_screen.is_some()
                || preset.bab.forward_root_screen_max_mb.is_some()
            {
                named.push(path.display().to_string());
            }
        }
    }
    assert!(
        parsed > 0,
        "no shipped preset was parsed, so this darkness check is VOID, not a pass"
    );
    assert!(
        named.is_empty(),
        "bab.forward_root_screen[_max_mb] must stay dark in every shipped preset until its \
         measurement lands: {named:?}"
    );
    let mut dark = BetaCrownConfig::default();
    apply_preset(&mut dark, &PresetConfig::default()).expect("empty preset");
    assert!(
        !dark.forward_root_screen,
        "an empty preset leaves the screen dark"
    );
    assert_eq!(dark.forward_root_screen_max_mb, 8192);
}

/// `#fwd-root-screen`: both keys reach `BetaCrownConfig` from yaml, an explicit
/// `false` disarms, an omitted ceiling keeps the default, and an inadmissible
/// ceiling (`0` included — it never means "unlimited") is refused at load.
#[test]
fn forward_root_screen_applies_from_yaml_and_refuses_bad_values() {
    let armed: PresetConfig = serde_yaml::from_str(
        "bab:\n  forward_root_screen: true\n  forward_root_screen_max_mb: 2048\n",
    )
    .expect("armed screen preset");
    assert_eq!(armed.bab.forward_root_screen, Some(true));
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed screen preset should apply");
    assert!(config.forward_root_screen);
    assert_eq!(config.forward_root_screen_max_mb, 2048);

    let enabled_only: PresetConfig =
        serde_yaml::from_str("bab:\n  forward_root_screen: true\n").expect("enabled only");
    let mut partial = BetaCrownConfig::default();
    apply_preset(&mut partial, &enabled_only).expect("enabled-only preset should apply");
    assert!(partial.forward_root_screen);
    assert_eq!(partial.forward_root_screen_max_mb, 8192);

    let disarmed: PresetConfig =
        serde_yaml::from_str("bab:\n  forward_root_screen: false\n").expect("disarmed");
    let mut config = BetaCrownConfig {
        forward_root_screen: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut config, &disarmed).expect("disarmed preset should apply");
    assert!(
        !config.forward_root_screen,
        "an explicit false must reach the config"
    );

    for bad in ["0", "63", "1048577"] {
        let yaml = format!("bab:\n  forward_root_screen_max_mb: {bad}\n");
        let preset: PresetConfig = serde_yaml::from_str(&yaml).expect("bad value still parses");
        let mut refused = BetaCrownConfig::default();
        assert!(
            apply_preset(&mut refused, &preset).is_err(),
            "`forward_root_screen_max_mb: {bad}` must be refused at load"
        );
    }
    assert!(
        serde_yaml::from_str::<PresetConfig>("bab:\n  forward_root_scren: true\n").is_err(),
        "a misspelt key must be refused, not silently ignored"
    );
}

/// Named V4 test 8: the screen reaches the lane through the TYPED preset key.
/// Its plan records where the arming came from, and the key has no
/// environment reader at all, so a plan built from yaml must report
/// `Source::Config` — never `Source::LegacyEnv`.
#[test]
fn fwd_root_screen_preset_source_is_typed() {
    let armed: PresetConfig =
        serde_yaml::from_str("bab:\n  forward_root_screen: true\n").expect("armed preset");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed preset should apply");
    let plan = ny_propagate::zono_core::forward_root_screen_plan(&config)
        .expect("the yaml key must produce an armed plan");
    assert_ne!(plan.source, ny_levers::Source::LegacyEnv);
    assert_eq!(plan.source, ny_levers::Source::Config);
    assert_eq!(plan.max_bytes, 8192u64 << 20);
    assert!(
        ny_propagate::zono_core::forward_root_screen_plan(&BetaCrownConfig::default()).is_none(),
        "without the key there is no plan and no work"
    );
}
/// `#relusplitter-lp-refine`: the rigorous LP root refiner is a MEASUREMENT
/// key. No shipped preset (any year, production or canary) may name it until
/// the commit carrying the relusplitter measurement arms it, and the scan must
/// actually parse presets or it is a void pass.
#[test]
fn mip_refine_intermediates_is_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut named = Vec::new();
    let mut parsed = 0usize;
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
            parsed += 1;
            if preset.bab.mip_refine_intermediates.is_some() {
                named.push(path.display().to_string());
            }
        }
    }
    assert!(
        parsed > 0,
        "no shipped preset was parsed, so this darkness check is VOID, not a pass"
    );
    assert!(
        named.is_empty(),
        "bab.mip_refine_intermediates must stay dark in every shipped preset until its \
         measurement lands: {named:?}"
    );
    let mut dark = BetaCrownConfig::default();
    apply_preset(&mut dark, &PresetConfig::default()).expect("empty preset");
    assert_eq!(
        dark.mip_refine_intermediates, None,
        "an empty preset attaches no refiner"
    );
}

/// `#relusplitter-lp-refine`: the key reaches `BetaCrownConfig` from yaml with
/// its defaults filled, explicit values survive, a misspelt sub-key or key is
/// refused at parse, and an inadmissible value is refused at load (never
/// clamped).
#[test]
fn mip_refine_intermediates_applies_from_yaml_and_refuses_bad_values() {
    let armed: PresetConfig = serde_yaml::from_str(
        "bab:\n  mip_refine_intermediates:\n    layers: 3\n    budget_frac: 0.25\n",
    )
    .expect("armed refiner preset");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed refiner preset should apply");
    assert_eq!(
        config.mip_refine_intermediates,
        Some(ny_propagate::MipRefineIntermediates {
            layers: 3,
            budget_frac: 0.25,
        })
    );

    let defaults: PresetConfig =
        serde_yaml::from_str("bab:\n  mip_refine_intermediates: {}\n").expect("defaults only");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &defaults).expect("defaults preset should apply");
    assert_eq!(
        config.mip_refine_intermediates,
        Some(ny_propagate::MipRefineIntermediates::default())
    );

    for bad in [
        "bab:\n  mip_refine_intermediates:\n    layers: 0\n",
        "bab:\n  mip_refine_intermediates:\n    layers: 65\n",
        "bab:\n  mip_refine_intermediates:\n    budget_frac: 0.0\n",
        "bab:\n  mip_refine_intermediates:\n    budget_frac: 0.95\n",
    ] {
        let preset: PresetConfig = serde_yaml::from_str(bad).expect("bad value still parses");
        let mut refused = BetaCrownConfig::default();
        assert!(
            apply_preset(&mut refused, &preset).is_err(),
            "`{bad}` must be refused at load"
        );
    }
    for misspelt in [
        "bab:\n  mip_refine_intermediates:\n    layer: 3\n",
        "bab:\n  mip_refine_intermediate:\n    layers: 3\n",
    ] {
        assert!(
            serde_yaml::from_str::<PresetConfig>(misspelt).is_err(),
            "`{misspelt}` must be refused, not silently ignored"
        );
    }
}

/// `#vit-zono-root`: the certified forward-zonotope root map is a MEASUREMENT
/// key. No shipped preset (any year, production or canary) may name any of its
/// three keys until the commit carrying the vit measurement (and the vit bank
/// gate) arms it, and the scan must actually parse presets or it is a void
/// pass.
#[test]
fn vit_zono_root_is_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut named = Vec::new();
    let mut parsed = 0usize;
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
            parsed += 1;
            if preset.bab.vit_zono_root.is_some()
                || preset.bab.vit_zono_root_max_mb.is_some()
                || preset.bab.vit_zono_root_max_secs.is_some()
            {
                named.push(path.display().to_string());
            }
        }
    }
    assert!(
        parsed > 0,
        "no shipped preset was parsed, so this darkness check is VOID, not a pass"
    );
    assert!(
        named.is_empty(),
        "bab.vit_zono_root[_max_mb|_max_secs] must stay dark in every shipped preset until its \
         measurement lands: {named:?}"
    );
    let mut dark = BetaCrownConfig::default();
    apply_preset(&mut dark, &PresetConfig::default()).expect("empty preset");
    assert!(
        !dark.vit_zono_root,
        "an empty preset leaves the root map dark"
    );
    assert_eq!(dark.vit_zono_root_max_mb, 8192);
    assert!((dark.vit_zono_root_max_secs - 30.0).abs() < f32::EPSILON);
    assert!(
        ny_propagate::zono_core::vit_zono_root_plan(&dark).is_none(),
        "without the key there is no plan and no work"
    );
}

/// `#vit-zono-root`: all three keys reach `BetaCrownConfig` from yaml, an
/// explicit `false` disarms, omitted ceilings keep their defaults, and an
/// inadmissible ceiling (`0` included — it never means "unlimited") is refused
/// at load, as is a misspelt key.
#[test]
fn vit_zono_root_applies_from_yaml_and_refuses_bad_values() {
    let armed: PresetConfig = serde_yaml::from_str(
        "bab:\n  vit_zono_root: true\n  vit_zono_root_max_mb: 2048\n  vit_zono_root_max_secs: 12.5\n",
    )
    .expect("armed root-map preset");
    assert_eq!(armed.bab.vit_zono_root, Some(true));
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed root-map preset should apply");
    assert!(config.vit_zono_root);
    assert_eq!(config.vit_zono_root_max_mb, 2048);
    assert!((config.vit_zono_root_max_secs - 12.5).abs() < f32::EPSILON);

    let enabled_only: PresetConfig =
        serde_yaml::from_str("bab:\n  vit_zono_root: true\n").expect("enabled only");
    let mut partial = BetaCrownConfig::default();
    apply_preset(&mut partial, &enabled_only).expect("enabled-only preset should apply");
    assert!(partial.vit_zono_root);
    assert_eq!(partial.vit_zono_root_max_mb, 8192);
    assert!((partial.vit_zono_root_max_secs - 30.0).abs() < f32::EPSILON);

    let disarmed: PresetConfig =
        serde_yaml::from_str("bab:\n  vit_zono_root: false\n").expect("disarmed");
    let mut config = BetaCrownConfig {
        vit_zono_root: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut config, &disarmed).expect("disarmed preset should apply");
    assert!(
        !config.vit_zono_root,
        "an explicit false must reach the config"
    );

    for bad in ["0", "63", "1048577"] {
        let yaml = format!("bab:\n  vit_zono_root_max_mb: {bad}\n");
        let preset: PresetConfig = serde_yaml::from_str(&yaml).expect("bad value still parses");
        let mut refused = BetaCrownConfig::default();
        assert!(
            apply_preset(&mut refused, &preset).is_err(),
            "`vit_zono_root_max_mb: {bad}` must be refused at load"
        );
    }
    for bad in ["0", "-1", "3601", ".nan", ".inf"] {
        let yaml = format!("bab:\n  vit_zono_root_max_secs: {bad}\n");
        let preset: PresetConfig = serde_yaml::from_str(&yaml).expect("bad value still parses");
        let mut refused = BetaCrownConfig::default();
        assert!(
            apply_preset(&mut refused, &preset).is_err(),
            "`vit_zono_root_max_secs: {bad}` must be refused at load"
        );
    }
    assert!(
        serde_yaml::from_str::<PresetConfig>("bab:\n  vit_zono_rot: true\n").is_err(),
        "a misspelt key must be refused, not silently ignored"
    );
}

/// `#vit-zono-root`: the root map reaches the lane through the TYPED preset
/// key. Its plan records where the arming came from, and the key has no
/// environment reader at all, so a plan built from yaml must report
/// `Source::Config` — never `Source::LegacyEnv`.
#[test]
fn vit_zono_root_preset_source_is_typed() {
    let armed: PresetConfig =
        serde_yaml::from_str("bab:\n  vit_zono_root: true\n").expect("armed preset");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed preset should apply");
    let plan = ny_propagate::zono_core::vit_zono_root_plan(&config)
        .expect("the yaml key must produce an armed plan");
    assert_ne!(plan.source, ny_levers::Source::LegacyEnv);
    assert_eq!(plan.source, ny_levers::Source::Config);
    assert_eq!(plan.max_bytes, 8192u64 << 20);
    assert!((plan.max_secs - 30.0).abs() < 1e-9);
}

#[test]
fn cgan_2026_preset_preserves_authored_graph_and_keeps_surrogate_dark() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp26/cgan2026.yaml")).unwrap();
    let load_config = build_onnx_load_config(&preset).expect("official 2026 cGAN loader policy");
    assert_eq!(
        load_config.batch_norm_folding_policy(),
        ny_onnx::BatchNormFoldingPolicy::PreserveRaw
    );
    assert!(load_config.require_authored_float32_initializers());
    assert_eq!(preset.model.forward_linear_spec_alpha, Some(false));
}

/// The warm-parallel route is intentionally absent from the shipped cGAN
/// preset. Its canary changes exactly the reordered-loop and scoped activation
/// fields, so a row-7 serial/parallel A/B cannot drift any other solver knob.
/// The target-complete root collector remains a sidecar experiment. Keep its
/// treatment surface attributable: it may receive more initial-bound time,
/// request exact root intermediates, and arm the new typed policy, but must
/// preserve every other production cGAN setting.
#[test]
fn cgan_sparse_target_complete_canary_differs_by_exactly_three_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_sparse_target_complete_canary.yaml"),
    )
    .unwrap();

    assert_eq!(
        production.bab.phase_budget.initial_bounds_fraction,
        Some(0.45)
    );
    assert_eq!(canary.bab.phase_budget.initial_bounds_fraction, Some(0.85));
    assert_eq!(production.bab.alpha_crown.fix_interm_bounds, None);
    assert_eq!(canary.bab.alpha_crown.fix_interm_bounds, Some(false));
    assert_eq!(
        production.bab.alpha_crown.cgan_sparse_target_complete_root,
        None
    );
    assert_eq!(
        canary.bab.alpha_crown.cgan_sparse_target_complete_root,
        Some(true)
    );
    assert_eq!(
        canary
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true),
        "the canary must preserve production's certified deadline fallback"
    );

    let mut normalized_canary = canary;
    normalized_canary.bab.phase_budget.initial_bounds_fraction =
        production.bab.phase_budget.initial_bounds_fraction;
    normalized_canary.bab.alpha_crown.fix_interm_bounds =
        production.bab.alpha_crown.fix_interm_bounds;
    normalized_canary
        .bab
        .alpha_crown
        .cgan_sparse_target_complete_root =
        production.bab.alpha_crown.cgan_sparse_target_complete_root;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "canary may differ only at initial_bounds_fraction, fix_interm_bounds, and \
         cgan_sparse_target_complete_root"
    );
}

/// #precheck-abs-cap: the absolute ceiling (seconds) cgan2026 puts on the
/// disjunctive CROWN / Alpha-CROWN precheck phases. Held in ONE place so the
/// yaml, the parsed preset, and the applied `BetaCrownConfig` cell cannot drift
/// apart. Value chosen from measurement, not preference — see
/// `reports/measured-2026/precheck_ship_2026-08-30/NOTES.md`.
const CGAN2026_PRECHECK_MAX_SECS: u64 = 60;

/// cgan2026 resamples properties over the same seven ONNX models as cgan_2023.
/// Keep model-cost and sound-fallback scheduling levers aligned without
/// transferring any result-count claim between competitions.
///
/// `bab.phase_budget.disjunctive_precheck_max_secs` (#precheck-abs-cap) was
/// armed on cgan2026 first (2026-08-30) and on cgan_2023 only after its own A/B,
/// a MOAT decision rather than a tuning taste:
///
///  * cgan2026 banks ZERO `unsat` (12 sat + 8 timeout + 1 unknown). All 12 sat
///    are decided by the upfront PGD / trusted-ORT attack in 0.1-1.3 s, which is
///    STRICTLY UPSTREAM of the precheck phase (`disjunctive.rs`: the attack lane
///    runs before `crown_precheck_clauses`). Bounding the precheck therefore
///    cannot lose a banked verdict here; it can only hand BaB more seconds.
///  * cgan_2023 banks TWO `unsat` — `imgSz32_nCh_1` prop_3 at 542.6 s and
///    `imgSz32_nCh_3` prop_1 at 808.0 s. Those are a live regression surface, so
///    the cap was not transferred there unmeasured; the A/B over those two rows
///    (2026-09-10, recorded in the test body) held both, and both presets now
///    arm the same value.
///
/// Measured basis: `reports/measured-2026/precheck_discard_2026-08-30/` (the
/// phase is a pure discard on row 10: capping it 4.8x leaves BaB's first six
/// `[disjunctive-trajectory]` rows BIT-IDENTICAL) and
/// `reports/measured-2026/precheck_ship_2026-08-30/` (the value, chosen across
/// all eight open rows).
///
/// The normalization below is deliberately NARROW — this one field and nothing
/// else. The final block of this test PROVES it is narrow by showing that a
/// second divergence, inside `phase_budget` and outside it, still trips the
/// equality. Without that block a widened normalization could silently license
/// arbitrary drift between the two presets.
#[test]
fn cgan2026_mirrors_byte_identical_model_schedule_levers() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cgan_2025 = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let cgan_2026 = load_preset(&repo_root.join("configs/vnncomp26/cgan2026.yaml")).unwrap();

    assert_ne!(cgan_2026.general.root_path, cgan_2025.general.root_path);
    assert!(
        cgan_2026
            .general
            .root_path
            .as_deref()
            .is_some_and(|path| path.ends_with("cgan2026/1.0")),
        "the regular-track cgan2026 preset must name the selected VNN-LIB 1.0 manifest"
    );
    assert_eq!(cgan_2026.general.device.as_deref(), Some("wgpu"));
    assert_eq!(
        cgan_2026.bab.crown_ibp_per_node_cap_secs,
        cgan_2025.bab.crown_ibp_per_node_cap_secs
    );
    // #precheck-abs-cap: state the cap positively on both presets, so the
    // normalization below can never quietly absorb some OTHER value. The
    // cgan_2023 A/B (2026-09-10, GB10, HEAD f8768e4cd, zero NY_*): both banked
    // unsat rows held and got faster under the cap (542.6 -> 308.5 s,
    // 808.0 -> 555.4 s), so cgan_2023 arms the same measured value.
    assert_eq!(
        cgan_2025.bab.phase_budget.disjunctive_precheck_max_secs,
        Some(CGAN2026_PRECHECK_MAX_SECS),
        "cgan_2023 must arm the measured precheck absolute cap (A/B 2026-09-10)"
    );
    assert_eq!(
        cgan_2026.bab.phase_budget.disjunctive_precheck_max_secs,
        Some(CGAN2026_PRECHECK_MAX_SECS),
        "cgan2026 must arm the measured precheck absolute cap"
    );

    let mut normalized_phase_budget = cgan_2026.bab.phase_budget.clone();
    normalized_phase_budget.disjunctive_precheck_max_secs =
        cgan_2025.bab.phase_budget.disjunctive_precheck_max_secs;
    assert_eq!(
        serde_json::to_value(&normalized_phase_budget).unwrap(),
        serde_json::to_value(&cgan_2025.bab.phase_budget).unwrap(),
        "byte-identical model schedules must retain the proven phase ceilings \
         apart from the measured precheck absolute cap"
    );
    assert_eq!(
        serde_json::to_value(&cgan_2026.bab.branching).unwrap(),
        serde_json::to_value(&cgan_2025.bab.branching).unwrap(),
        "byte-identical model schedules must retain the certified input-split options"
    );
    assert_eq!(
        cgan_2026.bab.alpha_crown.reference_refresh_max_secs,
        Some(12)
    );
    assert_eq!(cgan_2026.bab.alpha_crown.reference_refresh_fraction, None);
    assert_eq!(
        cgan_2026
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true)
    );
    assert_eq!(
        cgan_2026.bab.alpha_crown.cgan_complete_crown_ibp_root,
        cgan_2025.bab.alpha_crown.cgan_complete_crown_ibp_root,
        "byte-identical cGAN models must retain the certified complete-root policy"
    );

    let mut normalized_2026 = cgan_2026.clone();
    normalized_2026.general.root_path = cgan_2025.general.root_path.clone();
    normalized_2026.general.device = cgan_2025.general.device.clone();
    normalized_2026
        .bab
        .phase_budget
        .disjunctive_precheck_max_secs = cgan_2025.bab.phase_budget.disjunctive_precheck_max_secs;
    assert_eq!(
        serde_json::to_value(&normalized_2026).unwrap(),
        serde_json::to_value(&cgan_2025).unwrap(),
        "byte-identical cGAN presets may differ only in corpus path, selected \
         device, and the measured precheck absolute cap"
    );

    // ABLE TO FAIL. The three normalizations above are the whole licence this
    // test grants; anything else must still be caught. Demonstrate that with a
    // second divergence INSIDE phase_budget (the block the cap widened) and one
    // OUTSIDE it, both applied to the already-normalized preset.
    let mut second_divergence = normalized_2026.clone();
    second_divergence.bab.phase_budget.initial_bounds_fraction = Some(0.44);
    assert_ne!(
        serde_json::to_value(&second_divergence).unwrap(),
        serde_json::to_value(&cgan_2025).unwrap(),
        "the precheck-cap normalization must not mask any other phase_budget drift"
    );
    let mut third_divergence = normalized_2026;
    third_divergence.bab.batch_size = Some(63);
    assert_ne!(
        serde_json::to_value(&third_divergence).unwrap(),
        serde_json::to_value(&cgan_2025).unwrap(),
        "the precheck-cap normalization must not mask drift outside phase_budget"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cgan_2026).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.10);
    assert_eq!(config.phase_budget.disjunctive_pgd_max_secs, Some(5));
    assert_eq!(config.phase_budget.attack_extension_fraction, 0.0);
    assert_eq!(config.phase_budget.initial_bounds_fraction, 0.45);
    assert_eq!(config.phase_budget.disjunctive_precheck_fraction, 0.85);
    // #precheck-abs-cap: assert the APPLIED cell, not only the parsed preset.
    // `apply.rs:1130` is the bridge that writes it, and a bridge like this has
    // been silently clobbered in a merge before while the resolver-side test
    // stayed green. 0.85 * 900 = 765 s is a REAL grant here (measured 717.884 s
    // spent for `verified=0`), so this absolute ceiling is the only bound.
    assert_eq!(
        config.phase_budget.disjunctive_precheck_max_secs,
        Some(CGAN2026_PRECHECK_MAX_SECS)
    );
    assert_eq!(config.phase_budget.post_bab_pgd_fraction, 0.0);
    assert_eq!(config.branching_heuristic, BranchingHeuristic::InputSplit);
    assert_eq!(config.fsb_candidates, 5);
    assert!(config.reorder_bab);
    assert_eq!(config.input_split_alpha_iteration, 1);
    assert_eq!(config.input_split_lr_alpha, 0.05);
    assert!(config.input_split_ibp_enhancement);
    assert!(config.input_split_stacked_rebound);
    assert_eq!(config.alpha_config.reference_refresh_max_secs, Some(12));
    assert!(config.alpha_config.forward_linear_deadline_fallback_to_ibp);
    assert!(config.alpha_config.cgan_complete_crown_ibp_root);
}

/// The warm-parallel route is intentionally absent from the shipped cGAN
/// preset. Its canary changes exactly the measured reordered-loop batch and
/// scoped activation fields, so a row-7 A/B cannot drift any other solver
/// knob. Since #cgan-abc-parity (2026-08-27) promoted `reorder_bab: true`
/// into production, the canary's remaining treatment surface is batch_size
/// and warm_parallel.
#[test]
fn cgan_warm_parallel_batch8_canary_is_scoped_and_differs_by_exactly_two_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_reorder_warm_parallel_canary.yaml"),
    )
    .unwrap();

    assert_eq!(production.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(production.bab.branching.input_split.warm_parallel, None);
    assert_eq!(production.bab.batch_size, Some(64));
    assert_eq!(canary.bab.batch_size, Some(8));
    assert_eq!(canary.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(canary.bab.branching.input_split.warm_parallel, Some(true));

    let mut normalized_canary = canary.clone();
    normalized_canary.bab.batch_size = production.bab.batch_size;
    normalized_canary.bab.branching.input_split.warm_parallel =
        production.bab.branching.input_split.warm_parallel;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "canary may differ only at batch_size and warm_parallel"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();
    assert!(production_config.reorder_bab);
    assert!(!production_config.input_split_warm_parallel);
    assert_eq!(production_config.batch_size, 64);
    assert!(canary_config.reorder_bab);
    assert!(canary_config.input_split_warm_parallel);
    assert_eq!(canary_config.batch_size, 8);
}

/// This default-off canary began as a seven-knob translation of
/// alpha-beta-CROWN's current cGAN recipe. #cgan-abc-parity (2026-08-27)
/// promoted five of those knobs into production itself (conv_mode,
/// bound_prop_method, sb_coeff_thresh, reorder_bab, adv_check) after the
/// promoted set converted `nCh_1_prop_3` (banked timeout,895 -> unsat in
/// 513-533 s, 3 reproductions) with the 12-sat + test_nano moat clean. The
/// canary's remaining treatment surface is device and pgd_restarts. Keep it
/// exact: any extra drift would make a sealed row-7 A/B impossible to
/// attribute.
#[test]
fn cgan_abcrown_parity_canary_differs_by_exactly_two_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary =
        load_preset(&repo_root.join("configs/vnncomp25/cgan_2023_abcrown_parity_canary.yaml"))
            .unwrap();

    // The two remaining treatments.
    assert_eq!(production.general.device, None);
    assert_eq!(canary.general.device.as_deref(), Some("wgpu"));
    assert_eq!(production.attack.pgd_restarts, Some(200));
    assert_eq!(canary.attack.pgd_restarts, Some(100));

    // The five promoted knobs: now production values, shared by the canary.
    assert_eq!(
        production.general.conv_mode,
        Some(ny_propagate::ConvMode::Matrix)
    );
    assert_eq!(
        production.solver.bound_prop_method.as_deref(),
        Some("crown")
    );
    assert_eq!(
        production.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(production.bab.branching.input_split.reorder_bab, Some(true));
    // Sentinel semantics are intentionally counterintuitive: zero enables the
    // periodic SAT-only probe immediately, while -1 disables it.
    assert_eq!(production.bab.branching.input_split.adv_check, Some(0));
    assert_eq!(canary.general.conv_mode, production.general.conv_mode);
    assert_eq!(
        canary.solver.bound_prop_method,
        production.solver.bound_prop_method
    );
    assert_eq!(
        canary.bab.branching.input_split.sb_coeff_thresh,
        production.bab.branching.input_split.sb_coeff_thresh
    );
    assert_eq!(
        canary.bab.branching.input_split.reorder_bab,
        production.bab.branching.input_split.reorder_bab
    );
    assert_eq!(
        canary.bab.branching.input_split.adv_check,
        production.bab.branching.input_split.adv_check
    );

    let mut normalized_canary = canary.clone();
    normalized_canary.general.device = production.general.device.clone();
    normalized_canary.attack.pgd_restarts = production.attack.pgd_restarts;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "alpha-beta-CROWN parity canary may differ from shipped cGAN only at device and pgd_restarts"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();

    assert_eq!(production_config.conv_mode, ny_propagate::ConvMode::Matrix);
    assert!(!production_config.use_alpha_crown);
    assert_eq!(production_config.pgd_restarts, 200);
    assert_eq!(production_config.input_split_coeff_thresh, 1.0e-2);
    assert!(production_config.reorder_bab);
    assert_eq!(production_config.adv_check, 0);
    assert!(production_config.input_split_fresh_domain_clip);

    assert_eq!(canary_config.conv_mode, ny_propagate::ConvMode::Matrix);
    assert!(!canary_config.use_alpha_crown);
    assert_eq!(canary_config.pgd_restarts, 100);
    assert_eq!(canary_config.input_split_coeff_thresh, 1.0e-2);
    assert!(canary_config.reorder_bab);
    assert_eq!(canary_config.adv_check, 0);
    assert!(canary_config.input_split_fresh_domain_clip);
}

/// The default-off generator-prefix treatment must be a pure routing delta
/// from the alpha-beta-CROWN parity canary: explicit ReLU splitting replaces
/// explicit input splitting, and no input-split-only tuning leaks into the
/// treatment. Its sidecar filename also keeps normal cgan_2023 preset
/// discovery from selecting it.
#[test]
fn cgan_generator_relu_phase_split_canary_is_explicit_and_scoped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let parity_path = repo_root.join("configs/vnncomp25/cgan_2023_abcrown_parity_canary.yaml");
    let canary_path =
        repo_root.join("configs/vnncomp25/cgan_2023_generator_relu_phase_split_canary.yaml");

    assert_ne!(
        canary_path.file_name().and_then(|name| name.to_str()),
        Some("cgan_2023.yaml"),
        "default category discovery must not silently select the canary"
    );

    let parity = load_preset(&parity_path).unwrap();
    let canary = load_preset(&canary_path).unwrap();
    assert_eq!(parity.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(canary.bab.branching.method.as_deref(), Some("relu"));
    assert_eq!(
        serde_json::to_value(&canary.bab.branching.input_split).unwrap(),
        serde_json::to_value(InputSplitPreset::default()).unwrap(),
        "ReLU treatment must not retain inapplicable input-split tuning"
    );

    let resolved = resolve_branching(&canary)
        .unwrap()
        .expect("explicit relu method must resolve");
    assert_eq!(resolved.heuristic, BranchingHeuristic::LargestBoundWidth);
    assert!(
        resolved.use_relu_split,
        "explicit relu method must route to ReLU splitting, not auto-selection"
    );

    // The exact-domain input clip is InputSplit-only (`BetaCrownConfig::validate`
    // rejects it under any other branching heuristic), so the ReLU-splitting
    // canary must NOT carry the parity/production `input_split_fresh_domain_clip`
    // key: it is inapplicable input-split tuning of exactly the kind the
    // stanza-removal rule above exists for.
    assert_eq!(canary.bab.clip.input_split_fresh_domain_clip, None);

    let mut normalized_canary = canary.clone();
    normalized_canary.bab.branching.method = parity.bab.branching.method.clone();
    normalized_canary.bab.branching.input_split = parity.bab.branching.input_split.clone();
    normalized_canary.bab.clip.input_split_fresh_domain_clip =
        parity.bab.clip.input_split_fresh_domain_clip;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&parity).unwrap(),
        "generator-prefix canary may differ from the parity canary only at the explicit branching route and removed input-split-only tuning (stanza + exact-domain clip)"
    );

    assert_eq!(canary.general.device.as_deref(), Some("wgpu"));
    assert_eq!(
        canary.general.conv_mode,
        Some(ny_propagate::ConvMode::Matrix)
    );
    assert_eq!(canary.solver.bound_prop_method.as_deref(), Some("crown"));
    assert_eq!(canary.attack.pgd_restarts, Some(100));
    assert_eq!(canary.bab.batch_size, Some(64));
    assert_eq!(canary.bab.phase_budget.disjunctive_pgd_fraction, Some(0.10));
    assert_eq!(canary.bab.phase_budget.attack_extension_fraction, Some(0.0));
    assert_eq!(canary.bab.phase_budget.initial_bounds_fraction, Some(0.45));
    assert_eq!(
        canary.bab.phase_budget.disjunctive_precheck_fraction,
        Some(0.85)
    );
    assert_eq!(canary.bab.clip.relaxed, Some(true));
    assert_eq!(canary.bab.clip.relaxed_iterations, Some(20));
    assert_eq!(canary.bab.alpha_crown.lr_alpha, Some(0.2));
    // #general-defaults 2026-08-31: 100 IS the engine default
    // (alpha_config.rs, citing abc arguments.py:354), so stating it was a no-op in
    // 24 configs. Absent or 100 are the same resolved value; disagreement is not.
    assert!(canary.bab.alpha_crown.iterations.is_none_or(|v| v == 100));
    // #general-defaults: beta alpha_lr default corrected 0.01 -> 0.1 (13 of 14).
    assert!(canary.bab.beta_crown.lr_alpha.is_none_or(|v| v == 0.1));
    // #general-defaults: beta_lr default corrected 0.05 -> 0.15 (11 of 14).
    assert!(canary.bab.beta_crown.lr_beta.is_none_or(|v| v == 0.15));
    assert_eq!(canary.bab.beta_crown.iterations, Some(25));
}

/// The alpha-beta-CROWN VGG recipe is a default-off canary. Keep NY's shipped
/// WGPU/root phase-budget tuning and isolate exactly the five treatment knobs.
#[test]
fn vgg_abcrown_parity_canary_is_default_off_and_scoped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/vggnet16_2022.yaml")).unwrap();
    let canary =
        load_preset(&repo_root.join("configs/vnncomp25/vggnet16_2022_abcrown_parity_canary.yaml"))
            .unwrap();

    assert_eq!(production.model.vgg_abcrown_treatment, None);
    assert_eq!(canary.model.vgg_abcrown_treatment, Some(true));
    assert_eq!(
        canary.solver.bound_prop_method.as_deref(),
        Some("forward+backward")
    );
    assert_eq!(canary.bab.branching.method.as_deref(), Some("sb"));
    assert_eq!(canary.bab.branching.input_split.enable, Some(true));
    assert_eq!(canary.attack.pgd_order.as_deref(), Some("input_bab"));

    let mut normalized_canary = canary.clone();
    normalized_canary.model.vgg_abcrown_treatment = production.model.vgg_abcrown_treatment;
    normalized_canary.solver.bound_prop_method = production.solver.bound_prop_method.clone();
    normalized_canary.bab.branching.method = production.bab.branching.method.clone();
    normalized_canary.bab.branching.input_split.enable =
        production.bab.branching.input_split.enable;
    normalized_canary.attack.pgd_order = production.attack.pgd_order.clone();
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "VGG canary may differ from production only at its five treatment fields"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();

    assert!(
        production_config.use_alpha_crown,
        "shipped VGG root optimizations remain unchanged"
    );
    assert!(!production_config.use_forward_bounds);
    assert_eq!(
        production_config.branching_heuristic,
        BranchingHeuristic::LargestBoundWidth
    );

    assert!(!canary_config.use_alpha_crown);
    assert!(canary_config.use_forward_bounds);
    assert_eq!(
        canary_config.branching_heuristic,
        BranchingHeuristic::InputSplit
    );
    assert!(canary_config.enable_pgd_attack);
    assert_eq!(canary_config.batch_size, production_config.batch_size);
    assert_eq!(
        serde_json::to_value(&canary_config.phase_budget).unwrap(),
        serde_json::to_value(&production_config.phase_budget).unwrap(),
        "existing root phase budgets must survive the treatment"
    );
}

/// The production YOLO preset deliberately omits a branching method. Once its
/// loaded-model dimensions reach auto-resolution, that must select kFSB and
/// carry the typed critical-row policy into the final verifier config. The
/// companion ny-propagate selector test proves this config bit reaches the real
/// scorer with `NY_MO_SCORER_FIX` absent.
#[test]
fn yolo_2023_preset_auto_kfsb_arms_typed_critical_row_policy() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/yolo_2023.yaml")).unwrap();
    assert_eq!(
        preset.bab.branching.method, None,
        "YOLO must reach model-aware auto resolution"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        !config.use_multi_objective_critical_kfsb,
        "the reusable preset alone must not globally arm the costly scorer"
    );

    let resolved = resolve_auto_branching(
        AutoBranchingRequest {
            mip_complete_verifier: false,
            spec_input_count: Some(8112),
        },
        ModelStructure {
            param_count: 5_000_000,
            has_conv: true,
            relu_node_count: 40,
            is_dag: true,
        },
        8112,
    );
    assert_eq!(resolved.heuristic, BranchingHeuristic::Kfsb);
    assert_eq!(resolved.reason, AutoBranchingReason::HighDimOrManyRelu);
    assert!(resolved.use_multi_objective_critical_kfsb);

    // Mirrors the two runtime stamps in handle_beta_crown_command after preset
    // application and auto resolution.
    config.branching_heuristic = resolved.heuristic.clone();
    apply_resolved_auto_branching_runtime_policy(&mut config, &resolved);
    assert_eq!(config.branching_heuristic, BranchingHeuristic::Kfsb);
    assert!(config.use_multi_objective_critical_kfsb);
}

/// yolo_2023 carries NO per-node CROWN-IBP cap. The historical 12-second cap was
/// set because Conv_12/Conv_20 "consumed 51.772 and 31.142 seconds before falling
/// back to IBP" — but they fell back because the residual patches merge refused
/// on deadline PRESENCE (audit N6, regression 7b0140d09). With that fixed, the
/// cap was the only thing starving Conv_20/Conv_25: MEASURED 2026-09-23 on the
/// GB10, prop_000001 at the official 300 s, capped = 5/7 CROWN targets and a
/// watchdog SIGKILL at 308 s, uncapped = 7/7 CROWN and UNSAT at 76.3 s. Re-adding
/// a cap must re-measure that row first.
#[test]
fn yolo_2023_preset_leaves_crown_ibp_targets_uncapped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/yolo_2023.yaml")).unwrap();

    assert_eq!(preset.bab.crown_ibp_per_node_cap_secs, None);

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, None);
}

/// Presets that do not opt in carry no per-node overrides: the collector keeps
/// the 2.0-second floor and adaptive remaining-budget cap (e.g. cersyve,
/// dist_shift).
#[test]
fn presets_without_crown_ibp_budget_knob_keep_defaults() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["cersyve.yaml", "dist_shift_2023.yaml"] {
        let preset = load_preset(&repo_root.join("configs/vnncomp25").join(name)).unwrap();
        assert_eq!(
            preset.bab.crown_ibp_per_node_cap_secs, None,
            "{name} must not set the cgan-only cap knob"
        );
        assert_eq!(preset.bab.crown_ibp_per_node_floor_secs, None);

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        let budget = config.crown_ibp_per_node_time_budget();
        assert_eq!(budget, ny_propagate::CrownIbpPerNodeTimeBudget::default());
    }
}

/// TEETH for #reclaim-pgd on relusplitter.
///
/// The disjunctive PGD lane on this benchmark is RESTART-bound, not
/// fraction-bound: `pgd_restarts: 50` x the default `pgd_steps: 50` x 9 clauses
/// exhausts restarts before the fractional deadline is reached (measured
/// 2026-08-01: 11.5-27.5 s used of a 30 s/90 s slice on GT-unsat oval21 rows). So
/// `disjunctive_pgd_fraction` is the WRONG control here and an ABSOLUTE cap is the
/// right one.
///
/// Deleting the preset's cap fails the first assertion. "Fixing" it by lowering the
/// fraction instead fails the second. Reaching for any other phase knob — in
/// particular one that shortens a BOUND phase, which could change a verdict — fails
/// the whole-struct comparison, which is the assertion that actually pins the blast
/// radius.
#[test]
fn relusplitter_caps_the_restart_bound_disjunctive_pgd_phase() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/relusplitter.yaml")).unwrap();

    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_max_secs,
        Some(5),
        "relusplitter must cap the restart-bound disjunctive PGD phase in ABSOLUTE \
         seconds; the reclaimed budget flows to BaB via the ledger's remaining()"
    );
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_fraction, None,
        "the fraction is measured INERT on relusplitter (the lane never reaches its \
         fractional deadline) — it must not be used as the control here"
    );

    // #proof-tail (2026-08-21 receipts in the preset): the outer wrapper lane
    // is explicitly disabled alongside the zero engine fraction so the router
    // arms PresetProofTail (the cersyve precedent) and the solver keeps
    // everything but the fixed publication margin.
    assert_eq!(
        preset.bab.phase_budget.vnncomp_post_bab_attack,
        Some(false),
        "relusplitter must disable the outer post-BaB wrapper lane explicitly; \
         paired with post_bab_pgd_fraction 0.0 this arms the PresetProofTail route"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    // SCOPE, asserted as a whole-struct diff: the preset moves EXACTLY two phase
    // budget fields. Every other phase — initial bounds, reduced verification,
    // MIP, precheck — keeps its sealed default, so no bound-producing phase is
    // shortened and no verdict can move. Both moved fields bound ATTACK phases:
    // shortening an attack can only lose a counterexample (`sat` -> `unknown`,
    // a completeness cost); it can never mint an `unsat`, because no `unsat` is
    // ever concluded from a failed attack. The post-BaB tail is measured dead
    // weight on this category (full 220 sweep, 2026-08-21: all 20 GT-sat rows
    // land upfront at <=0.3-0.9s; the tail converted zero rows while holding
    // ~10s of every 60s row and ~21s of every 180s row).
    let expected = ny_propagate::PhaseBudgetConfig {
        disjunctive_pgd_max_secs: Some(5),
        post_bab_pgd_fraction: 0.0,
        ..Default::default()
    };
    assert_eq!(
        config.phase_budget, expected,
        "relusplitter's phase budget must differ from the sealed defaults in exactly \
         disjunctive_pgd_max_secs and post_bab_pgd_fraction"
    );
}

/// Collins' absolute cap suppresses an optional forward-linear warmer whose
/// cold build dominates the small CNN proof. The cap is schedule-only: neither
/// its attack work nor its optional bound-cache warmer can grant uncertified
/// authority, and the whole-struct assertion pins every other proof/attack
/// phase to its sealed default.
#[test]
fn collins_rul_caps_the_disjunctive_prewave_without_fraction_drift() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset =
        load_preset(&repo_root.join("configs/vnncomp25/collins_rul_cnn_2022.yaml")).unwrap();

    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_max_secs,
        Some(15),
        "Collins must keep the measured absolute prewave ceiling at every timeout tier"
    );
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_fraction, None,
        "a fractional override grows at long tiers and can re-admit the optional warmer"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    let expected = ny_propagate::PhaseBudgetConfig {
        disjunctive_pgd_max_secs: Some(15),
        ..Default::default()
    };
    assert_eq!(
        config.phase_budget, expected,
        "Collins may change only the attack prewave's absolute ceiling"
    );
}

/// The cap is SCOPED: presets that never measured the restart-bound lane must keep
/// the pure-fraction behaviour, so landing relusplitter's cap cannot silently
/// shorten falsification on an unrelated benchmark.
#[test]
fn disjunctive_pgd_cap_stays_scoped_to_presets_that_measured_it() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["acasxu_2023.yaml", "cersyve.yaml", "vit_2023.yaml"] {
        let preset = load_preset(&repo_root.join("configs/vnncomp25").join(name)).unwrap();
        assert_eq!(
            preset.bab.phase_budget.disjunctive_pgd_max_secs, None,
            "{name} must keep the pure-fraction disjunctive PGD behaviour"
        );

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert_eq!(
            config.phase_budget.disjunctive_pgd_max_secs, None,
            "{name} must not inherit relusplitter's absolute cap"
        );
    }
}

/// #sb-plane delivery: the typed key parses from yaml, reaches
/// `BetaCrownConfig` through `apply_preset`, defaults dark, and NO production
/// cgan preset arms it (the measurement runs on a canary copy). Fails if the
/// key stops flowing, if the default flips, or if someone arms production
/// without a measured promotion.
#[test]
fn sb_keep_plane_under_fresh_clip_parses_flows_to_config_and_ships_dark() {
    // Parse route: a minimal yaml naming only this key.
    let armed_text =
        "bab:\n  branching:\n    input_split:\n      sb_keep_plane_under_fresh_clip: true\n";
    let armed: PresetConfig = serde_yaml::from_str(armed_text).unwrap();
    assert_eq!(
        armed
            .bab
            .branching
            .input_split
            .sb_keep_plane_under_fresh_clip,
        Some(true)
    );
    let mut armed_config = BetaCrownConfig::default();
    assert!(
        !armed_config.input_split_sb_keep_plane_under_fresh_clip,
        "the engine default must be dark"
    );
    apply_preset(&mut armed_config, &armed).unwrap();
    assert!(
        armed_config.input_split_sb_keep_plane_under_fresh_clip,
        "the typed key must reach BetaCrownConfig"
    );

    // An explicit `false` is delivered too (not conflated with absent).
    let disarmed_text = armed_text.replace("true", "false");
    let disarmed: PresetConfig = serde_yaml::from_str(&disarmed_text).unwrap();
    assert_eq!(
        disarmed
            .bab
            .branching
            .input_split
            .sb_keep_plane_under_fresh_clip,
        Some(false)
    );

    // Production cgan presets ship DARK: key absent, resolved config false.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for relative in [
        "configs/vnncomp25/cgan_2023.yaml",
        "configs/vnncomp26/cgan2026.yaml",
    ] {
        let production = load_preset(&repo_root.join(relative)).unwrap();
        assert_eq!(
            production
                .bab
                .branching
                .input_split
                .sb_keep_plane_under_fresh_clip,
            None,
            "{relative} must not arm #sb-plane without a measured promotion"
        );
        let mut production_config = BetaCrownConfig::default();
        apply_preset(&mut production_config, &production).unwrap();
        assert!(
            !production_config.input_split_sb_keep_plane_under_fresh_clip,
            "{relative} must resolve #sb-plane dark"
        );
    }
}

/// #cap-extend delivery: PROMOTED 2026-09-10. Production cgan2026 (and
/// cgan_2023, kept byte-identical by the mirror test) now arm the disjunctive
/// domain-cap extension, measured on the GB10 (HEAD f8768e4cd, zero NY_*):
/// nCh_1 prop_1 timeout -> unsat 833-838 s (3/3), both banked unsat rows held.
/// The canary yaml is retained and must be byte-identical to production, so
/// the key still flows to `BetaCrownConfig` and nobody can silently re-darken
/// production while the canary keeps reading as the armed arm.
#[test]
fn cgan2026_cap_extend_is_promoted_and_flows_to_config() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp26/cgan2026.yaml")).unwrap();
    let canary =
        load_preset(&repo_root.join("configs/vnncomp26/cgan2026_capextend_canary.yaml")).unwrap();

    // Production is ARMED (promotion 2026-09-10), and the key is DELIVERED.
    assert_eq!(
        production.bab.branching.input_split.cap_extend_near_closure,
        Some(true),
        "production cgan2026 must arm #cap-extend (promoted 2026-09-10)"
    );
    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    assert!(production_config.input_split_cap_extend_near_closure);

    // Canary is ARMED, and the key is DELIVERED to the engine config.
    assert_eq!(
        canary.bab.branching.input_split.cap_extend_near_closure,
        Some(true)
    );
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();
    assert!(canary_config.input_split_cap_extend_near_closure);

    // Since the promotion the canary is byte-identical to production.
    assert_eq!(
        serde_json::to_value(&canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "cap-extend canary must equal production cgan2026 after the 2026-09-10 promotion"
    );
}

/// #cct-margin-row-admission: the cct preset arms EXACTLY cifar100_2024's
/// margin-row lane levers at cifar100's values, with tinyimagenet_2024's
/// reserve pattern (`reserve_secs: 0` + `concurrent`) in place of cifar100's
/// `adaptive_reserve` — cct's 120 s / 550 s tiers would otherwise hold a 45 s
/// reserve AND run the concurrent lane (see
/// `concurrent_presets_never_also_hold_a_reserve`). Category resolution must
/// deliver the block, and the patches-wiring canary must carry it verbatim.
#[test]
fn cct_margin_row_block_is_cifar100_lane_levers_with_zero_reserve() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    let cct =
        load_preset(&repo_root.join("configs/vnncomp26/challenging_certified_training_2026.yaml"))
            .unwrap();

    // `domain_stack` + `err_envelope` were armed on cifar100 (and tiny) on
    // their OWN A/B (2026-09-26, scratchpad/mrab); cct has no measurement of
    // the pair yet (WAVE6_PLAN M12), so it is a deliberate, recorded divergence
    // until a cct A/B arms it here. User-directed bounded frozen-dual defaults
    // are scoped to CIFAR100/TinyImageNet; CCT keeps the parent and refinement
    // keys absent as another explicit divergence.
    let mut want: Vec<&str> = margin_row_fields_set(&cifar.margin_row)
        .into_iter()
        .filter(|f| {
            !matches!(
                *f,
                "adaptive_reserve"
                    | "domain_stack"
                    | "err_envelope"
                    | "frozen_dual_bab"
                    | "frozen_dual_refine_steps"
            )
        })
        .collect();
    want.push("reserve_secs");
    want.sort_unstable();
    let mut got = margin_row_fields_set(&cct.margin_row);
    got.sort_unstable();
    assert_eq!(
        got, want,
        "cct margin_row preserves its recorded divergences from CIFAR100's lane levers"
    );

    let (c, x) = (&cct.margin_row, &cifar.margin_row);
    assert_eq!(c.reserve_secs, Some(0));
    assert_eq!(c.adaptive_reserve, None);
    assert_eq!(c.frozen_dual_bab, None);
    assert_eq!(c.frozen_dual_refine_steps, None);
    assert_eq!(c.concurrent, Some(true));
    assert_eq!(c.concurrent, x.concurrent);
    assert_eq!(c.root_f32, x.root_f32);
    assert_eq!(c.k_head, x.k_head);
    assert_eq!(c.k_trunk, x.k_trunk);
    assert_eq!(c.alpha_opt, x.alpha_opt);
    assert_eq!(c.alpha_cols, x.alpha_cols);
    assert_eq!(c.clip_interm, x.clip_interm);
    assert_eq!(c.clip_rows, x.clip_rows);
    assert_eq!(c.closure_max_tier, x.closure_max_tier);
    assert_eq!(c.interm_alpha, x.interm_alpha);
    assert_eq!(c.column_retire, x.column_retire);
    assert_eq!(c.closure_adaptive, x.closure_adaptive);
    assert_eq!(c.branch_reduce_max, x.branch_reduce_max);
    assert_eq!(c.parent_lift, x.parent_lift);
    assert_eq!(c.backward_interm, x.backward_interm);
    assert_eq!(c.bi_secs, x.bi_secs);

    let effective = crate::commands::vnncomp::resolve_preset_path(
        &repo_root.join("configs"),
        "challenging_certified_training_2026",
    )
    .expect("the cct category resolves an explicit preset");
    let resolved = load_preset(&effective).expect("effective cct preset loads");
    assert_eq!(resolved.margin_row.concurrent, Some(true));
    assert_eq!(resolved.margin_row.reserve_secs, Some(0));

    let canary =
        load_preset(&repo_root.join(
            "configs/vnncomp26/challenging_certified_training_2026_patcheswiring_canary.yaml",
        ))
        .unwrap();
    assert_eq!(
        serde_json::to_value(&canary.margin_row).unwrap(),
        serde_json::to_value(&cct.margin_row).unwrap(),
        "the patches-wiring canary must carry the production margin_row block verbatim"
    );
}

/// `#vgg-gpu-crown`: the certified GPU patches-CROWN root pass is a
/// MEASUREMENT key. No shipped preset (any year, production or canary) may
/// name it until the commit carrying the vggnet16 spec15-17 measurement arms
/// it, and the scan must actually parse presets or it is a void pass.
#[test]
fn gpu_patches_crown_is_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut named = Vec::new();
    let mut parsed = 0usize;
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
            parsed += 1;
            if preset.bab.gpu_patches_crown.is_some() {
                named.push(path.display().to_string());
            }
        }
    }
    assert!(
        parsed > 0,
        "no shipped preset was parsed, so this darkness check is VOID, not a pass"
    );
    assert!(
        named.is_empty(),
        "bab.gpu_patches_crown must stay dark in every shipped preset until its \
         measurement lands: {named:?}"
    );
    let mut dark = BetaCrownConfig::default();
    apply_preset(&mut dark, &PresetConfig::default()).expect("empty preset");
    assert_eq!(
        dark.gpu_patches_crown, None,
        "an empty preset arms no GPU patches-CROWN pass"
    );
}

/// `#vgg-gpu-crown`: the key reaches `BetaCrownConfig` from yaml with its
/// defaults filled, explicit values (including `precision: f32`) survive, a
/// misspelt sub-key or key is refused at parse, and an inadmissible value is
/// refused at load (never clamped).
#[test]
fn gpu_patches_crown_applies_from_yaml_and_refuses_bad_values() {
    let armed: PresetConfig = serde_yaml::from_str(
        "bab:\n  gpu_patches_crown:\n    chunk_rows: 128\n    max_mb: 4096\n    \
         gpu_tile_mb: 512\n    precision: f32\n    depth: 4\n    screen: false\n    \
         budget_frac: 0.8\n    max_secs: 1000.0\n    min_perturbed: 100000\n",
    )
    .expect("armed GPU patches-CROWN preset");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("armed preset should apply");
    let pass = config.gpu_patches_crown.expect("the key must land");
    assert_eq!(pass.chunk_rows, 128);
    assert_eq!(pass.max_mb, 4096);
    assert_eq!(pass.gpu_tile_mb, 512);
    assert_eq!(pass.precision, ny_propagate::GpuPatchesPrecision::F32);
    assert_eq!(pass.depth, 4);
    assert!(!pass.screen);
    assert!((pass.budget_frac - 0.8).abs() < 1e-12);
    assert!((pass.max_secs - 1000.0).abs() < 1e-12);
    assert_eq!(pass.min_perturbed, 100_000);
    assert!(
        (pass.headroom_frac - ny_propagate::GpuPatchesCrownConfig::default().headroom_frac).abs()
            < 1e-12,
        "an unset sub-key keeps its default"
    );

    let defaults: PresetConfig =
        serde_yaml::from_str("bab:\n  gpu_patches_crown: {}\n").expect("defaults only");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &defaults).expect("defaults preset should apply");
    assert_eq!(
        config.gpu_patches_crown,
        Some(ny_propagate::GpuPatchesCrownConfig::default())
    );

    for bad in [
        "bab:\n  gpu_patches_crown:\n    chunk_rows: 0\n",
        "bab:\n  gpu_patches_crown:\n    max_mb: 0\n",
        "bab:\n  gpu_patches_crown:\n    headroom_frac: 1.5\n",
        "bab:\n  gpu_patches_crown:\n    gpu_tile_mb: 0\n",
        "bab:\n  gpu_patches_crown:\n    depth: 65\n",
        "bab:\n  gpu_patches_crown:\n    budget_frac: 0.99\n",
        "bab:\n  gpu_patches_crown:\n    max_secs: 0.0\n",
    ] {
        let preset: PresetConfig = serde_yaml::from_str(bad).expect("bad value still parses");
        let mut refused = BetaCrownConfig::default();
        assert!(
            apply_preset(&mut refused, &preset).is_err(),
            "`{bad}` must be refused at load"
        );
    }
    for misspelt in [
        "bab:\n  gpu_patches_crown:\n    chunk_row: 3\n",
        "bab:\n  gpu_patches_crown:\n    precision: f16\n",
        "bab:\n  gpu_patch_crown: {}\n",
    ] {
        assert!(
            serde_yaml::from_str::<PresetConfig>(misspelt).is_err(),
            "`{misspelt}` must be refused, not silently ignored"
        );
    }
}

/// `#disj-odi`: the disjunctive exact-VJP ODI and focus-lane keys are
/// MEASUREMENT keys. No shipped preset (any year, production or canary) may
/// name either until the commit carrying the tinyimagenet idx_1831/idx_5529
/// conversion and the both-year moat arms cifar100/tinyimagenet, and the scan
/// must actually parse presets or it is a void pass.
#[test]
fn disj_odi_keys_are_dark_in_every_shipped_preset() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut named = Vec::new();
    let mut parsed = 0usize;
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
            parsed += 1;
            if preset.attack.pgd_disj_odi.is_some() || preset.attack.pgd_disj_focus_lanes.is_some()
            {
                named.push(path.display().to_string());
            }
            let mut config = BetaCrownConfig::default();
            if apply_preset(&mut config, &preset).is_ok() {
                assert!(
                    !config.pgd_exact_vjp.disj_odi && config.pgd_exact_vjp.disj_focus_lanes == 0,
                    "{} resolves an armed #disj-odi policy",
                    path.display()
                );
            }
        }
    }
    assert!(
        parsed > 0,
        "no shipped preset was parsed, so this darkness check is VOID, not a pass"
    );
    assert!(
        named.is_empty(),
        "attack.pgd_disj_odi / attack.pgd_disj_focus_lanes must stay dark in every shipped \
         preset until their measurement lands: {named:?}"
    );
    let mut dark = BetaCrownConfig::default();
    apply_preset(&mut dark, &PresetConfig::default()).expect("empty preset");
    assert!(!dark.pgd_exact_vjp.disj_odi);
    assert_eq!(dark.pgd_exact_vjp.disj_focus_lanes, 0);
}
