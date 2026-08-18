// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro vocoder deep-prefix property-lane canary (#3755).
//!
//! Moves Kokoro from the 2-node OutputWidth prefix smoke to the 12-node deep
//! prefix (through first Conv1d after ConvTranspose1d) and exercises the
//! SweepObjective::Linear property lane with boundary-spec objectives.
//!
//! This proves: (1) the property lane works on an audio-facing Kokoro surface,
//! (2) spec-guided CROWN propagates through the ResBlock cycle, and (3) the
//! recorded property metrics match the direct oracle.
//!
//! Budget: ~80s IBP node bounds + ~20-60s spec-guided CROWN = ~100-140s.
//! 600s accommodates variance under CPU contention.

use super::super::super::training_signal_support::assert_report_artifacts;
use super::super::crown_ibp_tightening::support::boundary_spec_matrix_range;
use super::super::graph_support::vocoder_prefix_subgraph;
use super::super::model::{
    bounded_kokoro_features_input, load_kokoro_vocoder_with_fixed_aux,
    KOKORO_VOCODER_MIN_FIXED_AUX_T,
};
use super::super::prefix::first_conv1d_after_conv_transpose;
use crate::training_signal::{
    mine_weak_regions_graph, write_weak_region_report, RegionSpec, RegionSweepConfig,
    SweepModelSource, SweepObjective, WeakRegionRecord, WeakRegionReport,
};
use ndarray::Array2;
use ny_propagate::types::BoundsProvenance;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;

const KOKORO_PROPERTY_BOUNDARY_SPECS: usize = 4;

struct KokoroPropertySweepFixture {
    graph: GraphNetwork,
    input: BoundedTensor,
    spec_matrix: Array2<f32>,
    node_bounds: HashMap<String, BoundedTensor>,
    report: WeakRegionReport,
}

fn build_kokoro_property_sweep_fixture() -> KokoroPropertySweepFixture {
    let dynamic_t = KOKORO_VOCODER_MIN_FIXED_AUX_T;
    let epsilon = 1e-3;
    let model = load_kokoro_vocoder_with_fixed_aux(dynamic_t);

    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");
    let cut_node = first_conv1d_after_conv_transpose(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &cut_node);

    let input = bounded_kokoro_features_input(&model, dynamic_t, epsilon);

    let node_bounds = prefix
        .collect_node_bounds(&input)
        .expect("deep prefix IBP node-bound collection should succeed");

    let output_bounds = node_bounds
        .get(prefix.output_name())
        .expect("output node bounds must exist for deep prefix");
    let flat_len = output_bounds.lower().len();

    let spec_matrix = boundary_spec_matrix_range(
        flat_len,
        KOKORO_PROPERTY_BOUNDARY_SPECS,
        0,
        KOKORO_PROPERTY_BOUNDARY_SPECS,
    );

    let regions = vec![{
        let bt = bounded_kokoro_features_input(&model, dynamic_t, epsilon);
        RegionSpec {
            label: format!("features_t{dynamic_t}_eps{epsilon:.0e}"),
            lower: bt.lower().to_owned(),
            upper: bt.upper().to_owned(),
            metadata: Some(serde_json::json!({
                "dynamic_t": dynamic_t,
                "epsilon": epsilon,
            })),
        }
    }];

    let config = RegionSweepConfig {
        primary_input: "features".to_string(),
        objective: SweepObjective::Linear {
            spec_matrix: Box::new(spec_matrix.clone()),
            thresholds: None,
        },
        regions,
        deadline: None,
        top_k_bounds: 1,
        hotspot_limit: 3,
    };
    let source = SweepModelSource {
        model_name: "kokoro_vocoder_deep_prefix".to_string(),
        model_path: None,
        model_digest: None,
    };

    let report = mine_weak_regions_graph(&prefix, &source, &config)
        .expect("property-guided mining should succeed on kokoro deep prefix");

    KokoroPropertySweepFixture {
        graph: prefix,
        input,
        spec_matrix,
        node_bounds,
        report,
    }
}

fn direct_kokoro_property_metrics(
    fixture: &KokoroPropertySweepFixture,
) -> (BoundsProvenance, f32, f32, f32) {
    let direct_spec = fixture
        .graph
        .propagate_crown_with_specs_and_provenance_and_engine_with_node_bounds_and_deadline(
            &fixture.input,
            &fixture.spec_matrix,
            None,
            &fixture.node_bounds,
            None,
        )
        .expect("direct spec-guided CROWN should succeed on kokoro deep prefix");

    let direct_slack = direct_spec
        .bounds
        .lower()
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let mut direct_obj_max = f32::NEG_INFINITY;
    let mut direct_width_total = 0.0_f32;
    let mut direct_width_count = 0usize;
    for (&upper, &lower) in direct_spec
        .bounds
        .upper()
        .iter()
        .zip(direct_spec.bounds.lower().iter())
    {
        let width = upper - lower;
        direct_obj_max = direct_obj_max.max(width);
        direct_width_total += width;
        direct_width_count += 1;
    }
    assert!(
        direct_width_count > 0,
        "direct property oracle must produce at least one spec bound"
    );
    (
        direct_spec.provenance,
        direct_slack,
        direct_obj_max,
        direct_width_total / direct_width_count as f32,
    )
}

fn assert_kokoro_property_record_basics(record: &WeakRegionRecord) {
    assert_eq!(record.method_requested, "spec_guided_crown");
    assert!(
        record.method_actual == "spec_guided_crown" || record.method_actual == "forward_fallback",
        "unexpected method_actual: {}",
        record.method_actual
    );
    match record.provenance {
        BoundsProvenance::Crown => assert_eq!(record.method_actual, "spec_guided_crown"),
        BoundsProvenance::ForwardFallback(_) => {
            assert_eq!(record.method_actual, "forward_fallback")
        }
    }
    let slack = record
        .certified_slack_min
        .expect("certified_slack_min must be populated for property objective");
    assert!(slack.is_finite(), "certified_slack_min not finite: {slack}");
    let obj_max = record
        .objective_width_max
        .expect("objective_width_max must be populated for property objective");
    assert!(
        obj_max.is_finite(),
        "objective_width_max not finite: {obj_max}"
    );
    let obj_mean = record
        .objective_width_mean
        .expect("objective_width_mean must be populated for property objective");
    assert!(
        obj_mean.is_finite(),
        "objective_width_mean not finite: {obj_mean}"
    );
}

fn assert_kokoro_property_matches_oracle(
    record: &WeakRegionRecord,
    direct_provenance: BoundsProvenance,
    direct_slack: f32,
    direct_obj_max: f32,
    direct_obj_mean: f32,
) {
    assert_eq!(
        record.provenance, direct_provenance,
        "report provenance must match direct spec-guided propagation"
    );
    let expected_method = match direct_provenance {
        BoundsProvenance::Crown => "spec_guided_crown",
        BoundsProvenance::ForwardFallback(_) => "forward_fallback",
    };
    assert_eq!(record.method_actual, expected_method);
    let recorded_slack = record.certified_slack_min.expect("slack");
    let recorded_obj_max = record.objective_width_max.expect("obj_max");
    let recorded_obj_mean = record.objective_width_mean.expect("obj_mean");
    assert!(
        (recorded_slack - direct_slack).abs() <= 1e-5,
        "certified_slack_min: record={recorded_slack}, direct={direct_slack}"
    );
    assert!(
        (recorded_obj_max - direct_obj_max).abs() <= 1e-5,
        "objective_width_max: record={recorded_obj_max}, direct={direct_obj_max}"
    );
    assert!(
        (recorded_obj_mean - direct_obj_mean).abs() <= 1e-5,
        "objective_width_mean: record={recorded_obj_mean}, direct={direct_obj_mean}"
    );
}

/// Deep-prefix property-lane smoke test (#3755).
///
/// Proves that the SweepObjective::Linear property lane works on the Kokoro
/// vocoder's 12-node deep prefix (through first Conv1d after ConvTranspose1d).
/// This exercises spec-guided CROWN through the ResBlock cycle with boundary-
/// spec objectives and validates the recorded metrics match a direct oracle.
///
/// Part of #3755.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_kokoro_deep_prefix_training_signal_property_smoke_3755() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    let fixture = build_kokoro_property_sweep_fixture();

    assert_eq!(fixture.report.regions.len(), 1, "expected 1 scored region");
    let record = &fixture.report.regions[0];

    assert_eq!(
        fixture.report.manifest.ranking_lane, "property",
        "deep prefix property sweep should use property ranking lane"
    );
    assert_kokoro_property_record_basics(record);

    let (direct_provenance, direct_slack, direct_obj_max, direct_obj_mean) =
        direct_kokoro_property_metrics(&fixture);
    assert_eq!(
        direct_provenance,
        BoundsProvenance::Crown,
        "direct oracle should use CROWN provenance on the deep prefix"
    );
    assert_kokoro_property_matches_oracle(
        record,
        direct_provenance,
        direct_slack,
        direct_obj_max,
        direct_obj_mean,
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    write_weak_region_report(&fixture.report, tmp.path()).expect("write should succeed");
    assert_report_artifacts(record, tmp.path(), 1);

    let manifest_text =
        std::fs::read_to_string(tmp.path().join("manifest.json")).expect("read manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).expect("parse manifest");
    assert_eq!(manifest["model_name"], "kokoro_vocoder_deep_prefix");

    eprintln!(
        "#3755 property smoke: method_actual={}, slack={:.4e}, obj_max={:.4e}, obj_mean={:.4e}",
        record.method_actual,
        record.certified_slack_min.unwrap_or(f32::NAN),
        record.objective_width_max.unwrap_or(f32::NAN),
        record.objective_width_mean.unwrap_or(f32::NAN),
    );
}
