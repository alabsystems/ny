// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Schema types for weak-region mining and training signal export (#3520).

use ndarray::{Array1, Array2, ArrayD};
use ny_propagate::types::BoundsProvenance;
use ny_tensor::BoundedTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Source provenance metadata for a sweep run.
///
/// Callers must supply this explicitly because `OnnxModel` does not retain
/// its source path or digest after loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepModelSource {
    pub model_name: String,
    pub model_path: Option<PathBuf>,
    pub model_digest: Option<String>,
}

/// Objective function for weak-region mining (#3520 Packet C).
///
/// `OutputWidth` preserves Packet A/B/D behavior: rank regions by output-bound
/// width (widest = weakest). `Linear` runs spec-guided CROWN with a caller-
/// supplied property matrix and ranks by certified slack.
#[derive(Debug, Clone)]
pub enum SweepObjective {
    /// Rank by output-node bound width (default Packet A/B/D behavior).
    OutputWidth,
    /// Rank by certified slack under a linear property `Cx >= tau`.
    ///
    /// `spec_matrix` is the C matrix (num_properties x output_dim).
    /// `thresholds` is the tau vector; when `None`, interpreted as all-zero.
    Linear {
        spec_matrix: Box<Array2<f32>>,
        thresholds: Option<Array1<f32>>,
    },
}

/// Configuration for a weak-region mining sweep.
#[derive(Debug, Clone)]
pub struct RegionSweepConfig {
    /// Name of the primary bounded input tensor.
    pub primary_input: String,
    /// Objective function controlling ranking and scoring (#3520 Packet C).
    pub objective: SweepObjective,
    /// Regions to score.
    pub regions: Vec<RegionSpec>,
    /// Optional wall-clock deadline (not used in Packet A).
    pub deadline: Option<Duration>,
    /// Number of top-ranked winners to export per-node bounds for.
    pub top_k_bounds: usize,
    /// Max hotspots to extract per region from profiling.
    pub hotspot_limit: usize,
}

/// A single bounded region to score.
#[derive(Debug, Clone)]
pub struct RegionSpec {
    pub label: String,
    pub lower: ArrayD<f32>,
    pub upper: ArrayD<f32>,
    pub metadata: Option<serde_json::Value>,
}

/// Per-node hotspot extracted from profiling a scored region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakRegionHotspot {
    pub name: String,
    pub layer_type: String,
    pub max_width: f32,
    pub mean_width: f32,
    pub growth_ratio: f32,
    pub status: String,
}

/// Scored record for one mined region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakRegionRecord {
    pub region_id: String,
    pub label: String,
    pub primary_input: String,
    pub lower_shape: Vec<usize>,
    pub upper_shape: Vec<usize>,
    pub method_requested: String,
    pub method_actual: String,
    pub provenance: BoundsProvenance,
    pub output_width_max: f32,
    pub output_width_mean: f32,
    /// Minimum certified slack across all properties: min(lower_i - tau_i).
    /// Present only when objective is `Linear` (#3520 Packet C).
    pub certified_slack_min: Option<f32>,
    /// Max property-bound width across all properties: max(upper_i - lower_i).
    /// Present only when objective is `Linear` (#3520 Packet C).
    pub objective_width_max: Option<f32>,
    /// Mean property-bound width across all properties.
    /// Present only when objective is `Linear` (#3520 Packet C).
    pub objective_width_mean: Option<f32>,
    pub top_hotspots: Vec<WeakRegionHotspot>,
    pub bounds_file: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Manifest describing a complete weak-region mining run.
///
/// This is the single entrypoint for downstream consumers. All artifact
/// references are relative to the output directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepManifest {
    pub schema_version: u32,
    pub generator: String,
    pub model_name: String,
    pub model_path: Option<String>,
    pub model_digest: Option<String>,
    pub graph_output: String,
    pub primary_input: String,
    pub ranking_lane: String,
    pub top_k_bounds: usize,
    pub hotspot_limit: usize,
    pub weak_regions_file: String,
    pub top_bounds_dir: String,
    pub region_count: usize,
    pub top_bounds_count: usize,
}

/// Cached node-bound payload for a winning region (module-private).
#[derive(Debug, Clone)]
pub(super) struct SelectedRegionBounds {
    pub(super) bounds_file: String,
    pub(super) node_bounds: HashMap<String, BoundedTensor>,
}

/// Complete report from a weak-region mining sweep.
///
/// `manifest` and `regions` are the public serializable surface.
/// `exported_bounds` caches only the top-K winners' per-node bound maps
/// so the writer can emit safetensors without recomputing.
#[derive(Debug, Clone)]
pub struct WeakRegionReport {
    pub manifest: SweepManifest,
    pub regions: Vec<WeakRegionRecord>,
    /// Private: cached top-K winner payloads for the writer.
    pub(super) exported_bounds: Vec<SelectedRegionBounds>,
}

impl WeakRegionReport {
    /// Create a new report with the given manifest, regions, and cached winner bounds.
    pub(super) fn new(
        manifest: SweepManifest,
        regions: Vec<WeakRegionRecord>,
        exported_bounds: Vec<SelectedRegionBounds>,
    ) -> Self {
        Self {
            manifest,
            regions,
            exported_bounds,
        }
    }
}
