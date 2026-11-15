// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #root-box-source (ENGINE_PROGRAM M5b) at the CLI boundary.
//!
//! * `root_box_source_is_dark_without_the_key_and_typed_with_it` — the typed
//!   key `margin_row.root_box_source` resolves to the bit-identical `tableau`
//!   without the key and arrives through the scored run's own preset
//!   projection (`preset_delivered_config_value`, NO environment) with
//!   `Source::Config`, all the way into the lane's frozen
//!   `RootPrecisionRuntimePolicy` — the policy the initial root, every rebake
//!   round and every epoch rebuild receive — and moves nothing else.
//!
//! This is the RESOLVER half of the arrival proof. The runtime half (an armed
//! yaml, zero `NY_*`, `[root-box] source=patch_backward` on the row's own
//! stderr) is the integrator's, on a pinned binary.

use ny_propagate::margin_row::{MarginRowRuntimePolicy, RootBoxSource};

use super::preset_delivered_config_value;

fn projected_levers(preset: &crate::preset::PresetConfig) -> ny_levers::LeverSet {
    let serialized = serde_json::to_value(preset).expect("preset serializes");
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
    ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        preset_delivered_config_value(&serialized, decl)
    })
    .expect("the projected preset must match its declarations")
}

#[test]
fn root_box_source_is_dark_without_the_key_and_typed_with_it() {
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_ROOT_BOX_SOURCE;
    let dark = projected_levers(&crate::preset::PresetConfig::default());
    let resolved = dark.resolved(decl).expect("the lever is declared");
    assert_eq!(
        resolved.value.as_str(),
        Some(RootBoxSource::Tableau.token())
    );
    assert_eq!(resolved.source, ny_levers::Source::Default);
    let dark_policy = MarginRowRuntimePolicy::from_levers(&dark).expect("policy");
    assert_eq!(
        dark_policy.root_precision.box_source,
        RootBoxSource::Tableau
    );

    let mut preset = crate::preset::PresetConfig::default();
    preset.margin_row.root_box_source = Some(RootBoxSource::PatchBackward);
    let typed = projected_levers(&preset);
    let resolved = typed.resolved(decl).expect("the lever is declared");
    assert_eq!(
        resolved.value.as_str(),
        Some(RootBoxSource::PatchBackward.token())
    );
    assert_eq!(resolved.source, ny_levers::Source::Config);
    assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
    // The yaml route reaches the lane's frozen root-precision policy with zero
    // `NY_*` set, and moves no other field of it.
    let typed_policy = MarginRowRuntimePolicy::from_levers(&typed).expect("policy");
    assert_eq!(
        typed_policy.root_precision.box_source,
        RootBoxSource::PatchBackward
    );
    assert_eq!(
        (
            typed_policy.root_precision.root_f32,
            typed_policy.root_precision.tableau_f32,
            typed_policy.root_precision.err_lane,
            typed_policy.root_precision.ibp_seed,
        ),
        (
            dark_policy.root_precision.root_f32,
            dark_policy.root_precision.tableau_f32,
            dark_policy.root_precision.err_lane,
            dark_policy.root_precision.ibp_seed,
        )
    );
    assert_eq!(typed_policy.backward_interm, dark_policy.backward_interm);
}
