// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runner for weak-region mining: scores regions with batched CROWN,
//! profiles hotspots, sorts deterministically, and caches top-K winners.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use ny_tensor::BoundedTensor;
use tracing::info;

use crate::analysis_error::AnalysisError;

use super::report::assign_winner_bounds_files;
use super::scoring::{score_region_linear, score_region_output_width, LinearSweepArgs};
use super::types::{
    RegionSweepConfig, SweepManifest, SweepModelSource, SweepObjective, WeakRegionRecord,
    WeakRegionReport,
};

#[derive(Debug)]
struct RankedWeakRegion {
    source_region_index: usize,
    record: WeakRegionRecord,
}

/// Mine weak regions from a model file path.
///
/// Loads the ONNX model, converts to a graph, and delegates to
/// `mine_weak_regions_model`. Fills in `model_path` and `model_digest`
/// from the file automatically.
pub fn mine_weak_regions(
    path: impl AsRef<Path>,
    config: &RegionSweepConfig,
) -> Result<WeakRegionReport, AnalysisError> {
    let path = path.as_ref();
    let model = crate::load_onnx(path).map_err(|e| AnalysisError::load("training_signal", e))?;

    let digest = ny_propagate::types::compute_model_hash(path).ok();

    let source = SweepModelSource {
        model_name: path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        model_path: Some(path.to_path_buf()),
        model_digest: digest,
    };

    mine_weak_regions_model(&model, &source, config)
}

/// Mine weak regions from an already-loaded model.
///
/// Validates the model has exactly one input matching `config.primary_input`,
/// then scores each region with batched CROWN, profiles hotspots, sorts
/// deterministically, and caches top-K winner per-node bounds.
pub fn mine_weak_regions_model(
    model: &crate::OnnxModel,
    source: &SweepModelSource,
    config: &RegionSweepConfig,
) -> Result<WeakRegionReport, AnalysisError> {
    // Validate single-input contract
    if model.network.inputs.len() != 1 {
        return Err(AnalysisError::invalid_input_shape(
            "training_signal",
            format!(
                "requires single-input model, got {} inputs",
                model.network.inputs.len()
            ),
        ));
    }
    let actual_input = &model.network.inputs[0].name;
    if actual_input != &config.primary_input {
        return Err(AnalysisError::invalid_input_shape(
            "training_signal",
            format!(
                "primary_input '{}' does not match model input '{}'",
                config.primary_input, actual_input
            ),
        ));
    }

    let graph = model
        .to_graph_network()
        .map_err(|e| AnalysisError::propagation("training_signal", e))?;

    mine_weak_regions_graph(&graph, source, config)
}

/// Mine weak regions from a pre-built graph network.
///
/// This is the graph-level core of the weak-region mining pipeline. Use this
/// when you have a subgraph (e.g., a prefix extracted from a large model)
/// that is already converted to a `GraphNetwork`.
///
/// Callers must supply `SweepModelSource` metadata explicitly because a
/// `GraphNetwork` does not retain source-path provenance.
pub fn mine_weak_regions_graph(
    graph: &ny_propagate::GraphNetwork,
    source: &SweepModelSource,
    config: &RegionSweepConfig,
) -> Result<WeakRegionReport, AnalysisError> {
    let graph_output = graph.output_name().to_string();

    info!(
        "Mining {} regions for model '{}' (input: '{}')",
        config.regions.len(),
        source.model_name,
        config.primary_input
    );

    // Determine ranking lane from objective.
    let ranking_lane = match &config.objective {
        SweepObjective::OutputWidth => "uncertainty",
        SweepObjective::Linear { .. } => "property",
    };

    // Score each region
    let mut ranked_records: Vec<RankedWeakRegion> = Vec::with_capacity(config.regions.len());

    for (source_region_index, region) in config.regions.iter().enumerate() {
        let input = BoundedTensor::new(region.lower.clone(), region.upper.clone())
            .map_err(|e| AnalysisError::propagation("training_signal", e))?;
        let deadline = absolute_deadline(config.deadline)?;

        let record = match &config.objective {
            SweepObjective::OutputWidth => score_region_output_width(
                graph,
                &input,
                region,
                &config.primary_input,
                config.hotspot_limit,
            )?,
            SweepObjective::Linear {
                spec_matrix,
                thresholds,
            } => score_region_linear(
                graph,
                &input,
                region,
                &config.primary_input,
                config.hotspot_limit,
                LinearSweepArgs {
                    spec_matrix,
                    thresholds: thresholds.as_ref(),
                    deadline,
                },
            )?,
        };

        ranked_records.push(RankedWeakRegion {
            source_region_index,
            record,
        });
    }

    match &config.objective {
        SweepObjective::OutputWidth => sort_ranked_records(&mut ranked_records),
        SweepObjective::Linear { .. } => sort_ranked_records_property(&mut ranked_records),
    }

    // Re-collect node bounds only for top-K winners
    let top_k = config.top_k_bounds.min(ranked_records.len());
    let mut winner_node_bounds = Vec::with_capacity(top_k);
    for source_region_index in winner_source_indices(&ranked_records, top_k) {
        let region = &config.regions[source_region_index];
        let input = BoundedTensor::new(region.lower.clone(), region.upper.clone())
            .map_err(|e| AnalysisError::propagation("training_signal", e))?;
        let node_bounds = graph
            .collect_node_bounds(&input)
            .map_err(|e| AnalysisError::propagation("training_signal", e))?;
        winner_node_bounds.push(node_bounds);
    }

    // Assign bounds files and build private cache
    let mut records: Vec<WeakRegionRecord> = ranked_records
        .into_iter()
        .map(|ranked| ranked.record)
        .collect();
    let exported = assign_winner_bounds_files(&mut records, top_k, winner_node_bounds);

    let manifest = SweepManifest {
        schema_version: 1,
        generator: "ny".to_string(),
        model_name: source.model_name.clone(),
        model_path: source.model_path.as_ref().map(|p| p.display().to_string()),
        model_digest: source.model_digest.clone(),
        graph_output,
        primary_input: config.primary_input.clone(),
        ranking_lane: ranking_lane.to_string(),
        top_k_bounds: config.top_k_bounds,
        hotspot_limit: config.hotspot_limit,
        weak_regions_file: "weak_regions.jsonl".to_string(),
        top_bounds_dir: "top_bounds".to_string(),
        region_count: records.len(),
        top_bounds_count: exported.len(),
    };

    Ok(WeakRegionReport::new(manifest, records, exported))
}

fn absolute_deadline(
    budget: Option<std::time::Duration>,
) -> Result<Option<Instant>, AnalysisError> {
    budget
        .map(|duration| {
            Instant::now().checked_add(duration).ok_or_else(|| {
                AnalysisError::invalid_input_shape(
                    "training_signal",
                    format!("deadline budget {duration:?} cannot be represented as an Instant"),
                )
            })
        })
        .transpose()
}

/// NaN-safe descending f32 comparison (NaN sorts last).
fn cmp_f32_desc(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b.total_cmp(&a),
    }
}

fn sort_ranked_records(records: &mut [RankedWeakRegion]) {
    // Preserve source-region identity through sorting so duplicate labels do
    // not collapse when we re-collect top-K node bounds.
    records.sort_by(|a, b| {
        cmp_f32_desc(a.record.output_width_max, b.record.output_width_max)
            .then_with(|| cmp_f32_desc(a.record.output_width_mean, b.record.output_width_mean))
            .then_with(|| {
                let a_worst = a
                    .record
                    .top_hotspots
                    .iter()
                    .map(|h| h.growth_ratio)
                    .fold(f32::NEG_INFINITY, f32::max);
                let b_worst = b
                    .record
                    .top_hotspots
                    .iter()
                    .map(|h| h.growth_ratio)
                    .fold(f32::NEG_INFINITY, f32::max);
                cmp_f32_desc(a_worst, b_worst)
            })
            .then_with(|| a.record.label.cmp(&b.record.label))
    });
}

/// NaN-safe ascending f32 comparison (NaN sorts last).
fn cmp_f32_asc(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}

/// Property-guided sort: certified_slack ascending, then objective_width
/// descending, then output_width descending, then label ascending.
///
/// Lower slack means harder to certify (weakest region first). When slacks
/// tie, wider objective bounds break the tie (more uncertain property).
fn sort_ranked_records_property(records: &mut [RankedWeakRegion]) {
    records.sort_by(|a, b| {
        let a_slack = a.record.certified_slack_min.unwrap_or(f32::INFINITY);
        let b_slack = b.record.certified_slack_min.unwrap_or(f32::INFINITY);
        cmp_f32_asc(a_slack, b_slack)
            .then_with(|| {
                let a_obj = a.record.objective_width_max.unwrap_or(f32::NEG_INFINITY);
                let b_obj = b.record.objective_width_max.unwrap_or(f32::NEG_INFINITY);
                cmp_f32_desc(a_obj, b_obj)
            })
            .then_with(|| cmp_f32_desc(a.record.output_width_max, b.record.output_width_max))
            .then_with(|| a.record.label.cmp(&b.record.label))
    });
}

fn winner_source_indices(
    records: &[RankedWeakRegion],
    top_k: usize,
) -> impl Iterator<Item = usize> + '_ {
    records
        .iter()
        .take(top_k)
        .map(|ranked| ranked.source_region_index)
}

#[cfg(test)]
mod runner_tests {
    use super::*;
    use ny_propagate::types::BoundsProvenance;

    fn make_ranked_record(source_region_index: usize, label: &str, width: f32) -> RankedWeakRegion {
        RankedWeakRegion {
            source_region_index,
            record: WeakRegionRecord {
                region_id: format!("region:{source_region_index:016x}"),
                label: label.to_string(),
                primary_input: "speaker".to_string(),
                lower_shape: vec![2],
                upper_shape: vec![2],
                method_requested: "batched_crown".to_string(),
                method_actual: "batched_crown".to_string(),
                provenance: BoundsProvenance::Crown,
                output_width_max: width,
                output_width_mean: width / 2.0,
                certified_slack_min: None,
                objective_width_max: None,
                objective_width_mean: None,
                top_hotspots: Vec::new(),
                bounds_file: None,
                metadata: None,
            },
        }
    }

    #[test]
    fn test_winner_source_indices_preserve_duplicate_labels() {
        let mut ranked_records = vec![
            make_ranked_record(0, "duplicate", 1.0),
            make_ranked_record(1, "duplicate", 3.0),
            make_ranked_record(2, "unique", 5.0),
        ];

        sort_ranked_records(&mut ranked_records);
        let winners: Vec<usize> = winner_source_indices(&ranked_records, 3).collect();

        assert_eq!(
            winners,
            vec![2, 1, 0],
            "duplicate labels must retain their distinct source regions after ranking"
        );
    }

    fn make_property_record(
        idx: usize,
        label: &str,
        slack: f32,
        obj_width: f32,
        out_width: f32,
    ) -> RankedWeakRegion {
        RankedWeakRegion {
            source_region_index: idx,
            record: WeakRegionRecord {
                region_id: format!("region:{idx:016x}"),
                label: label.to_string(),
                primary_input: "hidden_states".to_string(),
                lower_shape: vec![2],
                upper_shape: vec![2],
                method_requested: "spec_guided_crown".to_string(),
                method_actual: "spec_guided_crown".to_string(),
                provenance: BoundsProvenance::Crown,
                output_width_max: out_width,
                output_width_mean: out_width / 2.0,
                certified_slack_min: Some(slack),
                objective_width_max: Some(obj_width),
                objective_width_mean: Some(obj_width / 2.0),
                top_hotspots: Vec::new(),
                bounds_file: None,
                metadata: None,
            },
        }
    }

    /// Lower slack should sort before higher slack (#3520 Packet C).
    #[test]
    fn test_property_ranking_prefers_lower_slack_3520() {
        let mut records = vec![
            make_property_record(0, "safe", 2.0, 1.0, 5.0),
            make_property_record(1, "unsafe", -1.5, 3.0, 8.0),
            make_property_record(2, "borderline", 0.1, 2.0, 6.0),
        ];

        sort_ranked_records_property(&mut records);
        let labels: Vec<&str> = records.iter().map(|r| r.record.label.as_str()).collect();
        assert_eq!(
            labels,
            vec!["unsafe", "borderline", "safe"],
            "lower slack (weaker) must rank first"
        );
    }

    /// When slacks tie, wider objective_width_max breaks the tie.
    #[test]
    fn test_property_ranking_tiebreak_by_objective_width_3520() {
        let mut records = vec![
            make_property_record(0, "narrow", -1.0, 1.0, 5.0),
            make_property_record(1, "wide", -1.0, 4.0, 3.0),
        ];

        sort_ranked_records_property(&mut records);
        assert_eq!(
            records[0].record.label, "wide",
            "same slack: wider objective bounds should rank first"
        );
    }

    /// When slack and objective tie, output_width_max breaks the tie.
    #[test]
    fn test_property_ranking_tiebreak_by_output_width_3520() {
        let mut records = vec![
            make_property_record(0, "small_out", -1.0, 2.0, 1.0),
            make_property_record(1, "big_out", -1.0, 2.0, 9.0),
        ];

        sort_ranked_records_property(&mut records);
        assert_eq!(
            records[0].record.label, "big_out",
            "same slack and obj width: wider output should rank first"
        );
    }
}
