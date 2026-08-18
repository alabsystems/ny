// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Talker attention output-width smoke and ranking tests (#3520 Packet B).
//!
//! Proves the weak-region mining pipeline works on a multi-input model
//! after auxiliary inputs (cos/sin/mask) are frozen via `freeze_inputs`.
//! Uses the short-seq surface (T=4) to minimize CROWN runtime.

use super::super::super::training_signal_support::{
    assert_hotspot_contract, assert_output_width_metrics_and_provenance, assert_report_artifacts,
    HotspotContract, OutputWidthProvenanceContract,
};
use super::super::fixtures::{
    bounded_hidden_states_input, load_talker_attention_with_fixed_aux_for_seq_len,
    TALKER_ATTENTION_SHORT_SEQ_LEN,
};
use super::*;

/// Build a single-region spec for the talker attention short-seq surface.
///
/// Uses the short sequence length (4) and tight epsilon to minimize CROWN
/// runtime while still exercising the full pipeline on real weights.
fn talker_smoke_region_spec(seq_len: usize, epsilon: f32) -> Vec<RegionSpec> {
    let bt = bounded_hidden_states_input(seq_len, epsilon);
    vec![RegionSpec {
        label: format!("seq{seq_len}_eps{epsilon:.0e}"),
        lower: bt.lower().to_owned(),
        upper: bt.upper().to_owned(),
        metadata: Some(serde_json::json!({
            "sequence_len": seq_len,
            "epsilon": epsilon,
        })),
    }]
}

/// Smoke test: mines 1 region on the talker attention model with frozen
/// cos/sin/mask, then verifies metrics, hotspots, provenance, and artifacts.
///
/// Uses short seq_len=4 and tight epsilon=5e-4 to minimize CROWN runtime.
/// The key proof: `mine_weak_regions_model` works on a multi-input model
/// after `freeze_inputs` reduces it to a single activation input.
///
/// Part of #3520 Packet B.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_talker_training_signal_smoke_3520() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let seq_len = TALKER_ATTENTION_SHORT_SEQ_LEN;
    let epsilon = 5e-4;
    let model = load_talker_attention_with_fixed_aux_for_seq_len(seq_len);

    // After freeze_inputs, exactly 1 input (hidden_states) remains.
    assert_eq!(
        model.network.inputs.len(),
        1,
        "frozen talker should have 1 activation input, got {}",
        model.network.inputs.len()
    );
    let primary_input = model.network.inputs[0].name.clone();
    assert_eq!(primary_input, "hidden_states");

    let config = RegionSweepConfig {
        primary_input,
        objective: SweepObjective::OutputWidth,
        regions: talker_smoke_region_spec(seq_len, epsilon),
        deadline: None,
        top_k_bounds: 1,
        hotspot_limit: 3,
    };
    let source = SweepModelSource {
        model_name: "talker_attention_layer0".to_string(),
        model_path: None,
        model_digest: None,
    };

    let report = mine_weak_regions_model(&model, &source, &config)
        .expect("weak-region mining should succeed on frozen talker attention model");

    assert_eq!(report.regions.len(), 1, "expected 1 scored region");
    let record = &report.regions[0];
    assert_eq!(record.label, format!("seq{seq_len}_eps{epsilon:.0e}"));
    assert_eq!(record.primary_input, "hidden_states");

    assert_output_width_metrics_and_provenance(
        record,
        OutputWidthProvenanceContract::CrownOrForwardFallback,
    );
    assert_hotspot_contract(
        record,
        HotspotContract {
            min_count: 0,
            max_count: 3,
        },
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    write_weak_region_report(&report, tmp.path()).expect("write should succeed");
    assert_report_artifacts(record, tmp.path(), 1);

    // Verify manifest records the correct model name and primary input.
    let manifest_text =
        std::fs::read_to_string(tmp.path().join("manifest.json")).expect("read manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).expect("parse manifest");
    assert_eq!(manifest["model_name"], "talker_attention_layer0");
    assert_eq!(manifest["primary_input"], "hidden_states");
    assert_eq!(manifest["region_count"], 1);
}

/// Multi-region ranking: mines 2 regions with different epsilons to prove
/// the talker sweep ranks wider bounds higher. Deterministic ordering
/// is already proven by unit tests; this confirms it on real attention weights.
///
/// Part of #3520 Packet B.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_talker_training_signal_ranking_3520() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let seq_len = TALKER_ATTENTION_SHORT_SEQ_LEN;
    let model = load_talker_attention_with_fixed_aux_for_seq_len(seq_len);
    let primary_input = model.network.inputs[0].name.clone();

    let tight = bounded_hidden_states_input(seq_len, 1e-4);
    let wide = bounded_hidden_states_input(seq_len, 5e-4);

    let config = RegionSweepConfig {
        primary_input,
        objective: SweepObjective::OutputWidth,
        regions: vec![
            RegionSpec {
                label: "tight_1e-4".to_string(),
                lower: tight.lower().to_owned(),
                upper: tight.upper().to_owned(),
                metadata: None,
            },
            RegionSpec {
                label: "wide_5e-4".to_string(),
                lower: wide.lower().to_owned(),
                upper: wide.upper().to_owned(),
                metadata: None,
            },
        ],
        deadline: None,
        top_k_bounds: 1,
        hotspot_limit: 3,
    };
    let source = SweepModelSource {
        model_name: "talker_attention_layer0".to_string(),
        model_path: None,
        model_digest: None,
    };

    let report =
        mine_weak_regions_model(&model, &source, &config).expect("talker ranking should succeed");

    assert_eq!(report.regions.len(), 2, "expected 2 scored regions");

    // Wider epsilon region should rank first (wider output bounds).
    assert_eq!(
        report.regions[0].label, "wide_5e-4",
        "wider epsilon region should rank first; got '{}' then '{}'",
        report.regions[0].label, report.regions[1].label
    );
    assert!(
        report.regions[0].output_width_max >= report.regions[1].output_width_max,
        "first-ranked region should have wider bounds: {} >= {}",
        report.regions[0].output_width_max,
        report.regions[1].output_width_max,
    );

    // Only the top-1 winner should have a bounds_file.
    assert!(
        report.regions[0].bounds_file.is_some(),
        "winner must have bounds_file"
    );
    assert!(
        report.regions[1].bounds_file.is_none(),
        "non-winner must not have bounds_file"
    );
}
