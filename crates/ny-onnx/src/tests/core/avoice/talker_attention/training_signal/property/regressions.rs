// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Packet C regression tests for property-guided centroid-monotonicity (#3520).
//!
//! Part of #4089 property lane decomposition.

use super::fixture::{
    build_talker_property_sweep_fixture, build_talker_property_sweep_fixture_with_deadline,
};
use super::metrics::{
    assert_property_provenance_and_metrics, assert_property_record_matches_direct_metrics,
    direct_property_metrics,
};
use super::*;

/// Packet C smoke test: property-guided mining on the talker softmax surface.
///
/// Builds a centroid-monotonicity spec matrix and runs `mine_weak_regions_graph`
/// with `SweepObjective::Linear` on the short-seq softmax subgraph.
///
/// Part of #3520 Packet C.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_talker_training_signal_property_smoke_3520() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let seq_len = TALKER_ATTENTION_SHORT_SEQ_LEN;
    let epsilon = 5e-4;
    let fixture = build_talker_property_sweep_fixture(seq_len, epsilon);

    assert_eq!(fixture.report.regions.len(), 1, "expected 1 scored region");
    let record = &fixture.report.regions[0];
    assert_property_provenance_and_metrics(record);

    assert_eq!(
        fixture.report.manifest.ranking_lane, "property",
        "Packet C should use property ranking lane"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    write_weak_region_report(&fixture.report, tmp.path()).expect("write should succeed");
    assert_report_artifacts(record, tmp.path(), 1);
}

/// Packet C regression: the sweep-level deadline must thread through the
/// property lane so `mine_weak_regions_graph` can surface deadline fallback
/// provenance, not just the lower-level graph API.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_talker_training_signal_property_deadline_fallback_3520() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let fixture = build_talker_property_sweep_fixture_with_deadline(
        TALKER_ATTENTION_SHORT_SEQ_LEN,
        5e-4,
        Some(Duration::ZERO),
    );
    let record = fixture
        .report
        .regions
        .first()
        .expect("property-guided mining should return one scored region");

    assert_eq!(record.method_requested, "spec_guided_crown");
    assert_eq!(record.method_actual, "forward_fallback");
    assert_eq!(
        record.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
        "deadline-budgeted property sweeps must surface deadline fallback provenance"
    );
    assert_eq!(fixture.report.manifest.ranking_lane, "property");
    assert!(
        record.certified_slack_min.is_some()
            && record.objective_width_max.is_some()
            && record.objective_width_mean.is_some(),
        "fallback property sweeps must still populate property metrics from the IBP-backed result"
    );
}

/// Packet C regression: `certified_slack_min` must match the direct
/// spec-guided centroid-monotonicity lower bound and preserve the same
/// certification status as the conservative centroid-gap interval check.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_talker_training_signal_property_slack_tracks_centroid_status_3520() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let seq_len = TALKER_ATTENTION_SHORT_SEQ_LEN;
    let epsilon = 5e-4;
    let fixture = build_talker_property_sweep_fixture(seq_len, epsilon);
    let record = fixture
        .report
        .regions
        .first()
        .expect("property-guided mining should return one scored region");
    let (direct_provenance, direct_slack, direct_obj_max, direct_obj_mean) =
        direct_property_metrics(&fixture);
    assert_property_record_matches_direct_metrics(
        record,
        direct_provenance,
        direct_slack,
        direct_obj_max,
        direct_obj_mean,
    );
    let recorded_slack = record
        .certified_slack_min
        .expect("certified_slack_min must be populated for property objective");

    let (centroid_lower, centroid_upper, query_seq_len) = centroid_bounds_from_softmax(
        fixture
            .node_bounds
            .get(fixture.graph.output_name())
            .expect("output node bounds must exist"),
        "short-seq talker Packet C output bounds",
    );
    let conservative_max_gap =
        centroid_monotonicity_gaps(&centroid_lower, &centroid_upper, query_seq_len)
            .into_iter()
            .fold(f32::NEG_INFINITY, f32::max);
    let conservative_slack = -conservative_max_gap;

    assert!(
        recorded_slack + 1e-4 >= conservative_slack,
        "spec-guided slack should be at least as strong as the conservative centroid-gap bound: record={recorded_slack}, conservative={conservative_slack}"
    );

    if conservative_slack > 1e-4 {
        assert!(
            recorded_slack > 1e-4,
            "positive conservative slack must imply certified_slack_min certifies monotonicity too: record={recorded_slack}, conservative={conservative_slack}"
        );
    }
}
