// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Real speaker encoder integration tests for training_signal (#3520 Packet A).
//!
//! Uses the seq5 epsilon ladder from the fixture contract:
//! designs/2026-03-12-issue-3520-speaker-region-fixture-contract.md
//!
//! Consolidated single-region smoke test: proves the full end-to-end pipeline
//! on real ONNX within the 600s cargo wrapper timeout. Multi-region ranking
//! is already covered by unit tests in `training_signal/runner_tests`.

use super::super::training_signal_support::{
    assert_hotspot_contract, assert_output_width_metrics_and_provenance, assert_report_artifacts,
    HotspotContract, OutputWidthProvenanceContract,
};
use super::*;
use crate::training_signal::{
    mine_weak_regions_model, write_weak_region_report, RegionSpec, RegionSweepConfig,
    SweepModelSource, SweepObjective,
};

/// Build a single-region spec using the tightest epsilon (fastest CROWN).
fn speaker_smoke_region_spec(model: &OnnxModel) -> Vec<RegionSpec> {
    let bt = shared::bounded_speaker_encoder_input(model, SPEAKER_ENCODER_SEQUENCE_LEN, 5e-4);
    vec![RegionSpec {
        label: "seq5_eps5e-4".to_string(),
        lower: bt.lower().to_owned(),
        upper: bt.upper().to_owned(),
        metadata: Some(serde_json::json!({
            "sequence_len": SPEAKER_ENCODER_SEQUENCE_LEN,
            "epsilon": 5e-4_f32,
        })),
    }]
}

/// Comprehensive smoke test: mines 1 region on the real speaker encoder,
/// then verifies metrics, hotspots, provenance, and report artifact layout.
///
/// Uses the tightest epsilon (5e-4) to minimize CROWN runtime. With 1 region
/// and top_k_bounds=1, the pipeline runs 3 heavy passes (CROWN + profile +
/// node-bound collection) instead of the 8+ the 3-region config needs.
/// Multi-region ranking is proven by synthetic unit tests in runner_tests.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_speaker_training_signal_smoke() {
    crate::test_fixtures::require_test_model_or_skip!("speaker_encoder.onnx");
    let model = shared::avoice_speaker_encoder();
    let config = RegionSweepConfig {
        primary_input: model.network.inputs[0].name.clone(),
        objective: SweepObjective::OutputWidth,
        regions: speaker_smoke_region_spec(model),
        deadline: None,
        top_k_bounds: 1,
        hotspot_limit: 3,
    };
    let source = SweepModelSource {
        model_name: "speaker_encoder".to_string(),
        model_path: None,
        model_digest: None,
    };

    let report = mine_weak_regions_model(model, &source, &config)
        .expect("weak-region mining should succeed on speaker_encoder.onnx");

    assert_eq!(report.regions.len(), 1, "expected 1 scored region");
    let record = &report.regions[0];
    assert_eq!(record.label, "seq5_eps5e-4");

    assert_output_width_metrics_and_provenance(record, OutputWidthProvenanceContract::StrictCrown);
    assert_hotspot_contract(
        record,
        HotspotContract {
            min_count: 1,
            max_count: 3,
        },
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    write_weak_region_report(&report, tmp.path()).expect("write should succeed");
    assert_report_artifacts(record, tmp.path(), 1);
}
