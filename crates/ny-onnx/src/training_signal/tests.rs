// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Packet-local unit tests for training_signal (#3520).
//!
//! These tests use synthetic data and do NOT require the real avoice fixture.

use std::collections::HashMap;

use ndarray::arr1;
use ny_propagate::types::BoundsProvenance;
use ny_tensor::BoundedTensor;

use super::report::{compute_region_id, write_weak_region_report};
use super::types::{
    SelectedRegionBounds, SweepManifest, WeakRegionHotspot, WeakRegionRecord, WeakRegionReport,
};

fn make_record(label: &str, lower: &[f32], upper: &[f32], width_max: f32) -> WeakRegionRecord {
    WeakRegionRecord {
        region_id: compute_region_id(
            "input",
            &[lower.len()],
            &[upper.len()],
            lower.iter().copied(),
            upper.iter().copied(),
        ),
        label: label.to_string(),
        primary_input: "input".to_string(),
        lower_shape: vec![lower.len()],
        upper_shape: vec![upper.len()],
        method_requested: "batched_crown".to_string(),
        method_actual: "batched_crown".to_string(),
        provenance: BoundsProvenance::Crown,
        output_width_max: width_max,
        output_width_mean: width_max * 0.5,
        certified_slack_min: None,
        objective_width_max: None,
        objective_width_mean: None,
        top_hotspots: vec![WeakRegionHotspot {
            name: "layer_0".to_string(),
            layer_type: "Linear".to_string(),
            max_width: width_max * 0.8,
            mean_width: width_max * 0.4,
            growth_ratio: width_max,
            status: if width_max > 5.0 { "WIDE" } else { "MODERATE" }.to_string(),
        }],
        bounds_file: None,
        metadata: None,
    }
}

fn make_winner_cache(regions: &mut [WeakRegionRecord], top_k: usize) -> Vec<SelectedRegionBounds> {
    let mut exported = Vec::new();
    for i in 0..top_k.min(regions.len()) {
        let safe: String = regions[i].region_id.replace(
            |c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '_' && c != '-',
            "_",
        );
        let bounds_file = format!("top_bounds/{}.safetensors", safe);
        regions[i].bounds_file = Some(bounds_file.clone());
        let mut nb = HashMap::new();
        nb.insert(
            "layer_0".to_string(),
            BoundedTensor::new(arr1(&[-1.0, -0.5]).into_dyn(), arr1(&[1.0, 0.5]).into_dyn())
                .unwrap(),
        );
        exported.push(SelectedRegionBounds {
            bounds_file,
            node_bounds: nb,
        });
    }
    exported
}

fn make_manifest(top_k: usize, region_count: usize, top_bounds_count: usize) -> SweepManifest {
    SweepManifest {
        schema_version: 1,
        generator: "ny".to_string(),
        model_name: "test_model".to_string(),
        model_path: Some("/tmp/test.onnx".to_string()),
        model_digest: Some("abc123".to_string()),
        graph_output: "output".to_string(),
        primary_input: "input".to_string(),
        ranking_lane: "uncertainty".to_string(),
        top_k_bounds: top_k,
        hotspot_limit: 3,
        weak_regions_file: "weak_regions.jsonl".to_string(),
        top_bounds_dir: "top_bounds".to_string(),
        region_count,
        top_bounds_count,
    }
}

/// Build a synthetic report for writer tests.
fn make_synthetic_report(top_k: usize) -> WeakRegionReport {
    let mut regions = vec![
        make_record("wide_region", &[-1.0, 0.0], &[1.0, 0.5], 10.0),
        make_record("narrow_region", &[-0.5, 0.0], &[0.5, 0.25], 3.0),
    ];
    let exported = make_winner_cache(&mut regions, top_k);
    let manifest = make_manifest(top_k, regions.len(), exported.len());
    WeakRegionReport::new(manifest, regions, exported)
}

#[ntest::timeout(10000)]
#[test]
fn test_writer_creates_manifest_and_regions() {
    let report = make_synthetic_report(1);
    let tmp = tempfile::tempdir().expect("tempdir");
    write_weak_region_report(&report, tmp.path()).expect("write");

    // manifest.json must exist and be valid JSON
    let manifest_path = tmp.path().join("manifest.json");
    assert!(manifest_path.exists(), "manifest.json must exist");
    let manifest_text = std::fs::read_to_string(&manifest_path).expect("read manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).expect("parse manifest");

    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["generator"], "ny");
    assert_eq!(manifest["weak_regions_file"], "weak_regions.jsonl");
    assert_eq!(manifest["top_bounds_dir"], "top_bounds");
    assert_eq!(manifest["region_count"], 2);
    assert_eq!(manifest["top_bounds_count"], 1);

    // weak_regions.jsonl must exist with correct line count
    let regions_path = tmp.path().join("weak_regions.jsonl");
    assert!(regions_path.exists(), "weak_regions.jsonl must exist");
    let lines: Vec<String> = std::fs::read_to_string(&regions_path)
        .expect("read regions")
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    assert_eq!(lines.len(), 2, "expected 2 region records");

    // Verify each line is valid JSON
    for (i, line) in lines.iter().enumerate() {
        let _: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("line {} invalid JSON: {}", i, e));
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_writer_top_bounds_files() {
    let report = make_synthetic_report(1);
    let tmp = tempfile::tempdir().expect("tempdir");
    write_weak_region_report(&report, tmp.path()).expect("write");

    let bounds_dir = tmp.path().join("top_bounds");
    assert!(bounds_dir.exists(), "top_bounds dir must exist");

    let entries: Vec<_> = std::fs::read_dir(&bounds_dir)
        .expect("read bounds dir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(entries.len(), 1, "exactly 1 winner safetensors");
    assert!(
        entries[0]
            .file_name()
            .to_str()
            .unwrap()
            .ends_with(".safetensors"),
        "winner file must end in .safetensors"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_winner_bounds_file_is_relative_and_no_colon() {
    let report = make_synthetic_report(2);

    for record in &report.regions {
        if let Some(ref bf) = record.bounds_file {
            assert!(
                bf.starts_with("top_bounds/"),
                "bounds_file must be relative: {}",
                bf
            );
            assert!(
                !bf.contains(':'),
                "bounds_file must not contain ':': {}",
                bf
            );
        }
    }

    // With top_k=2 and 2 regions, both are winners
    assert!(report.regions[0].bounds_file.is_some());
    assert!(report.regions[1].bounds_file.is_some());
}

#[ntest::timeout(10000)]
#[test]
fn test_non_winner_has_no_bounds_file() {
    let report = make_synthetic_report(1);
    assert!(
        report.regions[0].bounds_file.is_some(),
        "winner must have bounds_file"
    );
    assert!(
        report.regions[1].bounds_file.is_none(),
        "non-winner must have None bounds_file"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_writer_manifest_contract_fields() {
    let report = make_synthetic_report(2);
    let tmp = tempfile::tempdir().expect("tempdir");
    write_weak_region_report(&report, tmp.path()).expect("write");

    let manifest_text =
        std::fs::read_to_string(tmp.path().join("manifest.json")).expect("read manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).expect("parse manifest");

    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["generator"], "ny");
    assert!(manifest["model_name"].is_string());
    assert_eq!(manifest["ranking_lane"], "uncertainty");
    assert_eq!(manifest["top_k_bounds"], 2);
    assert_eq!(manifest["hotspot_limit"], 3);
    assert_eq!(manifest["weak_regions_file"], "weak_regions.jsonl");
    assert_eq!(manifest["top_bounds_dir"], "top_bounds");
    assert_eq!(manifest["region_count"], 2);
    assert_eq!(manifest["top_bounds_count"], 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_writer_bounds_files_resolve() {
    let report = make_synthetic_report(1);
    let tmp = tempfile::tempdir().expect("tempdir");
    write_weak_region_report(&report, tmp.path()).expect("write");

    for record in &report.regions {
        if let Some(ref bf) = record.bounds_file {
            let full_path = tmp.path().join(bf);
            assert!(
                full_path.exists(),
                "bounds_file reference must resolve: {}",
                full_path.display()
            );
        }
    }
}
