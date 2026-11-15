// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #fl-f32-seam-key receipt agreement (#fl-alpha-composition review): the
//! f32 value-GEMM seam mode the CLI installs
//! (`beta_crown::resolve_forward_linear_f32_seam`, whose value wins inside
//! ny-propagate's `forward_linear_f32_seam_armed`) must be the mode the scored
//! run's flight receipt records for `NY_FORWARD_LINEAR_F32`.
//!
//! The receipt layers the environment over the preset key
//! (`LeverSet::resolve_layered` over `preset_delivered_config_value`, which
//! maps `model.forward_linear_f32_seam` through `space.rs`). The original
//! bridge installed the RAW key, which then beat the environment, so with
//! `NY_FORWARD_LINEAR_F32=1` and the key `false` the receipt said armed while
//! the run used f64. The variable is named through its declaration only.

use super::preset_delivered_config_value;

fn seam_decl() -> &'static ny_levers::LeverDecl {
    &ny_levers::decls::forward_linear::FORWARD_LINEAR_F32
}

fn preset_with_key(key: Option<bool>) -> crate::preset::PresetConfig {
    let mut preset = crate::preset::PresetConfig::default();
    preset.model.forward_linear_f32_seam = key;
    preset
}

/// The LeverSet a scored run freezes for its receipt: `env` is the captured
/// `NY_FORWARD_LINEAR_F32` (every other lever absent), over the preset's own
/// projection.
fn receipt_levers(preset: &crate::preset::PresetConfig, env: Option<&str>) -> ny_levers::LeverSet {
    let serialized = serde_json::to_value(preset).expect("preset serializes");
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |name| {
        if name == seam_decl().name {
            env.map(std::ffi::OsString::from)
        } else {
            None
        }
    });
    ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        preset_delivered_config_value(&serialized, decl)
    })
    .expect("the projected preset must match its declarations")
}

/// ny's layering for an exact-`"1"` Bool lever: an admissible environment
/// value wins, a present inadmissible one suppresses the key and resolves to
/// the default (f64), absence falls through to the key, then the default.
fn layered(env: Option<&str>, key: Option<bool>) -> bool {
    match env {
        Some("1") => true,
        Some(_) => false,
        None => key.unwrap_or(false),
    }
}

const ENV_INPUTS: [Option<&str>; 4] = [None, Some("0"), Some("1"), Some("true")];
const KEY_INPUTS: [Option<bool>; 3] = [None, Some(false), Some(true)];

/// Scored route: on every environment x key combination the installed seam
/// mode is exactly the receipt's, and the receipt is ny's env-over-config
/// layering (so this is not merely the resolver reading back its own input).
#[test]
fn scored_seam_install_matches_the_flight_receipt() {
    for env in ENV_INPUTS {
        for key in KEY_INPUTS {
            let preset = preset_with_key(key);
            let levers = receipt_levers(&preset, env);
            let receipt = levers
                .resolved(seam_decl())
                .expect("the lever is declared")
                .value
                .as_bool();
            assert_eq!(
                receipt,
                layered(env, key),
                "receipt layering, env={env:?} key={key:?}"
            );
            assert_eq!(
                crate::commands::beta_crown::resolve_forward_linear_f32_seam(
                    Some(&levers),
                    Some(&preset)
                ),
                Some(receipt),
                "installed seam mode disagrees with the flight receipt, env={env:?} key={key:?}"
            );
        }
    }
}

/// The reviewer's case, spelled out: environment `1`, key `false`. The
/// receipt records the seam ARMED from the environment, so the run must arm
/// it too (the original bridge installed `false`).
#[test]
fn an_admissible_environment_value_beats_the_key_in_the_run_as_in_the_receipt() {
    let preset = preset_with_key(Some(false));
    let levers = receipt_levers(&preset, Some("1"));
    let resolved = levers.resolved(seam_decl()).expect("the lever is declared");
    assert!(resolved.value.as_bool());
    assert_eq!(resolved.source, ny_levers::Source::LegacyEnv);
    assert_eq!(
        crate::commands::beta_crown::resolve_forward_linear_f32_seam(Some(&levers), Some(&preset)),
        Some(true)
    );
}

/// The typed key still arrives through the scored projection with
/// `Source::Config` when no environment is set, and is installed armed.
#[test]
fn the_seam_key_arrives_through_the_scored_projection() {
    let preset = preset_with_key(Some(true));
    let levers = receipt_levers(&preset, None);
    let resolved = levers.resolved(seam_decl()).expect("the lever is declared");
    assert!(resolved.value.as_bool());
    assert_eq!(resolved.source, ny_levers::Source::Config);
    assert_eq!(
        crate::commands::beta_crown::resolve_forward_linear_f32_seam(Some(&levers), Some(&preset)),
        Some(true)
    );
}

/// Interactive route (no frozen LeverSet): the environment is layered over
/// the key by the same rule, and without the key nothing is installed, so the
/// seam's own declared environment read decides exactly as with no preset.
#[test]
fn interactive_seam_install_layers_the_environment_over_the_key() {
    ny_test_utils::env::with_env_edits(|edits| {
        for env in ENV_INPUTS {
            match env {
                Some(value) => edits.set(seam_decl().name, value),
                None => edits.remove(seam_decl().name),
            }
            assert_eq!(
                crate::commands::beta_crown::resolve_forward_linear_f32_seam(None, None),
                None,
                "no preset installs nothing, env={env:?}"
            );
            for key in KEY_INPUTS {
                let preset = preset_with_key(key);
                let expected = key.map(|key| layered(env, Some(key)));
                assert_eq!(
                    crate::commands::beta_crown::resolve_forward_linear_f32_seam(
                        None,
                        Some(&preset)
                    ),
                    expected,
                    "interactive install, env={env:?} key={key:?}"
                );
            }
        }
    });
}
