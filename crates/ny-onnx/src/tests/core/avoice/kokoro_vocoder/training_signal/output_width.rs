// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro vocoder prefix output-width smoke and ranking tests (#3520 Packet D).
//!
//! Proves the weak-region mining pipeline works on the Kokoro vocoder's
//! CPU-viable prefix subgraph. Full-graph vocoder CROWN is too expensive
//! for CPU tests (~180s+ timeout), so we extract the first ConvTranspose1d
//! prefix (2 nodes, ~45s IBP) and mine on that surface.

use super::super::super::training_signal_support::{
    assert_hotspot_contract, assert_output_width_metrics_and_provenance, assert_report_artifacts,
    HotspotContract, OutputWidthProvenanceContract,
};
use super::super::graph_support::{first_conv_transpose_node, vocoder_prefix_subgraph};
use super::super::model::{
    bounded_kokoro_features_input, load_kokoro_vocoder_with_fixed_aux,
    KOKORO_VOCODER_MIN_FIXED_AUX_T,
};
use crate::training_signal::{
    mine_weak_regions_graph, write_weak_region_report, RegionSpec, RegionSweepConfig,
    SweepModelSource, SweepObjective,
};

/// Build region specs for the kokoro vocoder prefix surface.
///
/// Uses the minimum fixed-aux temporal window (T=1) with `features` shape
/// [512, 1] (unbatched) so the prefix IBP+CROWN stays within CPU budget.
fn kokoro_prefix_region_spec(
    model: &crate::OnnxModel,
    dynamic_t: usize,
    epsilon: f32,
) -> Vec<RegionSpec> {
    let bt = bounded_kokoro_features_input(model, dynamic_t, epsilon);
    vec![RegionSpec {
        label: format!("features_t{dynamic_t}_eps{epsilon:.0e}"),
        lower: bt.lower().to_owned(),
        upper: bt.upper().to_owned(),
        metadata: Some(serde_json::json!({
            "dynamic_t": dynamic_t,
            "epsilon": epsilon,
        })),
    }]
}

/// Smoke test: mines 1 region on the Kokoro vocoder prefix subgraph,
/// then verifies metrics, hotspots, provenance, and artifacts.
///
/// Uses the shallow ConvTranspose1d prefix (2 nodes) to keep runtime
/// within CPU budget (~45s IBP + batched CROWN). The key proof:
/// `mine_weak_regions_graph` works on a subgraph extracted from a
/// multi-input model after `freeze_inputs` + prefix extraction.
///
/// Part of #3520 Packet D.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_kokoro_prefix_training_signal_smoke_3520() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    let dynamic_t = KOKORO_VOCODER_MIN_FIXED_AUX_T;
    let epsilon = 1e-3;
    let model = load_kokoro_vocoder_with_fixed_aux(dynamic_t);

    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");
    let cut_node = first_conv_transpose_node(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &cut_node);

    eprintln!(
        "Packet D smoke: prefix {} nodes (full: {}), cut: {}",
        prefix.num_nodes(),
        graph.num_nodes(),
        cut_node
    );

    let config = RegionSweepConfig {
        primary_input: "features".to_string(),
        objective: SweepObjective::OutputWidth,
        regions: kokoro_prefix_region_spec(&model, dynamic_t, epsilon),
        deadline: None,
        top_k_bounds: 1,
        hotspot_limit: 3,
    };
    let source = SweepModelSource {
        model_name: "kokoro_vocoder_prefix".to_string(),
        model_path: None,
        model_digest: None,
    };

    let report = mine_weak_regions_graph(&prefix, &source, &config)
        .expect("weak-region mining should succeed on kokoro vocoder prefix");

    assert_eq!(report.regions.len(), 1, "expected 1 scored region");
    let record = &report.regions[0];
    assert_eq!(
        record.label,
        format!("features_t{dynamic_t}_eps{epsilon:.0e}")
    );
    assert_eq!(record.primary_input, "features");

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

    // Verify manifest records prefix model name and features input.
    let manifest_text =
        std::fs::read_to_string(tmp.path().join("manifest.json")).expect("read manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).expect("parse manifest");
    assert_eq!(manifest["model_name"], "kokoro_vocoder_prefix");
    assert_eq!(manifest["primary_input"], "features");
    assert_eq!(manifest["region_count"], 1);

    eprintln!(
        "Packet D smoke: method_actual={}, width_max={:.4e}, width_mean={:.4e}",
        record.method_actual, record.output_width_max, record.output_width_mean
    );
}

/// Multi-region ranking: mines 2 regions with different epsilons on the
/// Kokoro vocoder prefix to prove the sweep ranks wider bounds higher.
///
/// Part of #3520 Packet D.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_kokoro_prefix_training_signal_ranking_3520() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    let dynamic_t = KOKORO_VOCODER_MIN_FIXED_AUX_T;
    let model = load_kokoro_vocoder_with_fixed_aux(dynamic_t);

    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");
    let cut_node = first_conv_transpose_node(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &cut_node);

    let tight = bounded_kokoro_features_input(&model, dynamic_t, 1e-4);
    let wide = bounded_kokoro_features_input(&model, dynamic_t, 1e-3);

    let config = RegionSweepConfig {
        primary_input: "features".to_string(),
        objective: SweepObjective::OutputWidth,
        regions: vec![
            RegionSpec {
                label: "tight_1e-4".to_string(),
                lower: tight.lower().to_owned(),
                upper: tight.upper().to_owned(),
                metadata: None,
            },
            RegionSpec {
                label: "wide_1e-3".to_string(),
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
        model_name: "kokoro_vocoder_prefix".to_string(),
        model_path: None,
        model_digest: None,
    };

    let report = mine_weak_regions_graph(&prefix, &source, &config)
        .expect("kokoro prefix ranking should succeed");

    assert_eq!(report.regions.len(), 2, "expected 2 scored regions");

    // Wider epsilon region should rank first (wider output bounds).
    assert_eq!(
        report.regions[0].label, "wide_1e-3",
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
