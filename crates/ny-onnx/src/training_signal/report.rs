// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Report writing for weak-region mining results (#3520).
//!
//! Owns `manifest.json`, `weak_regions.jsonl`, and `top_bounds/*.safetensors`
//! emission. Does NOT call profiling or CROWN propagation — only serialization.

use ny_core::Result;
use std::path::Path;

use super::types::WeakRegionReport;

/// Derive a file-safe filename stem from a `region_id`.
///
/// Replaces every non-`[A-Za-z0-9._-]` byte with `_`.
fn bounds_file_name(region_id: &str) -> String {
    let safe_stem: String = region_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("top_bounds/{}.safetensors", safe_stem)
}

/// Compute a deterministic region ID from the primary input name and bounds.
///
/// Hashes `primary_input`, `lower.shape()`, `upper.shape()`, and all lower/upper
/// f32 values in little-endian order using a deterministic weighted-sum byte hash.
/// Does NOT include model digest — region IDs are stable across checkpoint updates.
pub(super) fn compute_region_id(
    primary_input: &str,
    lower_shape: &[usize],
    upper_shape: &[usize],
    lower: impl IntoIterator<Item = f32>,
    upper: impl IntoIterator<Item = f32>,
) -> String {
    let mut hash: u64 = 0;
    let mut pos: u64 = 1;

    // Hash primary input name bytes
    for &b in primary_input.as_bytes() {
        hash = hash.wrapping_add((b as u64).wrapping_mul(pos));
        pos = pos.wrapping_add(1);
    }

    // Hash lower shape
    for &dim in lower_shape {
        for &b in &(dim as u64).to_le_bytes() {
            hash = hash.wrapping_add((b as u64).wrapping_mul(pos));
            pos = pos.wrapping_add(1);
        }
    }

    // Hash upper shape
    for &dim in upper_shape {
        for &b in &(dim as u64).to_le_bytes() {
            hash = hash.wrapping_add((b as u64).wrapping_mul(pos));
            pos = pos.wrapping_add(1);
        }
    }

    // Hash lower values
    for val in lower {
        for &b in &val.to_le_bytes() {
            hash = hash.wrapping_add((b as u64).wrapping_mul(pos));
            pos = pos.wrapping_add(1);
        }
    }

    // Hash upper values
    for val in upper {
        for &b in &val.to_le_bytes() {
            hash = hash.wrapping_add((b as u64).wrapping_mul(pos));
            pos = pos.wrapping_add(1);
        }
    }

    format!("region:{:016x}", hash)
}

/// Write a weak-region report to an output directory.
///
/// Creates:
/// - `manifest.json` — self-describing run manifest (pretty JSON)
/// - `weak_regions.jsonl` — one compact JSON object per scored region
/// - `top_bounds/<safe_stem>.safetensors` — per-node bounds for top-K winners
///
/// All artifact references are relative to `output_dir`.
pub fn write_weak_region_report(
    report: &WeakRegionReport,
    output_dir: impl AsRef<Path>,
) -> Result<()> {
    use ny_core::NyError;
    use std::io::Write;

    let out = output_dir.as_ref();

    // Create output directory and top_bounds subdirectory
    std::fs::create_dir_all(out).map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to create output dir {}: {}",
            out.display(),
            e
        ))
    })?;
    let bounds_dir = out.join("top_bounds");
    std::fs::create_dir_all(&bounds_dir).map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to create top_bounds dir {}: {}",
            bounds_dir.display(),
            e
        ))
    })?;

    // Write manifest.json (pretty)
    let manifest_json = serde_json::to_string_pretty(&report.manifest)
        .map_err(|e| NyError::ModelLoad(format!("Failed to serialize manifest: {}", e)))?;
    let manifest_path = out.join("manifest.json");
    std::fs::write(&manifest_path, manifest_json).map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to write manifest {}: {}",
            manifest_path.display(),
            e
        ))
    })?;

    // Write weak_regions.jsonl (one compact JSON per line)
    let regions_path = out.join("weak_regions.jsonl");
    let mut regions_file = std::fs::File::create(&regions_path).map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to create regions file {}: {}",
            regions_path.display(),
            e
        ))
    })?;
    for record in &report.regions {
        let line = serde_json::to_string(record)
            .map_err(|e| NyError::ModelLoad(format!("Failed to serialize region record: {}", e)))?;
        writeln!(regions_file, "{}", line).map_err(|e| {
            NyError::ModelLoad(format!(
                "Failed to write region record to {}: {}",
                regions_path.display(),
                e
            ))
        })?;
    }

    // Write top_bounds safetensors for winners
    for selected in &report.exported_bounds {
        let safetensors_path = out.join(&selected.bounds_file);
        crate::bound_export::export_bounds_safetensors(&selected.node_bounds, &safetensors_path)?;
    }

    Ok(())
}

/// Assign `bounds_file` to winning regions and build the private winner cache.
///
/// Called by the runner after sorting to populate the report's private
/// `exported_bounds` field and backfill `bounds_file` on winning records.
pub(super) fn assign_winner_bounds_files(
    regions: &mut [super::types::WeakRegionRecord],
    top_k: usize,
    winner_node_bounds: Vec<std::collections::HashMap<String, ny_tensor::BoundedTensor>>,
) -> Vec<super::types::SelectedRegionBounds> {
    let mut exported = Vec::new();
    for (i, node_bounds) in winner_node_bounds.into_iter().enumerate() {
        if i >= regions.len() || i >= top_k {
            break;
        }
        let file = bounds_file_name(&regions[i].region_id);
        regions[i].bounds_file = Some(file.clone());
        exported.push(super::types::SelectedRegionBounds {
            bounds_file: file,
            node_bounds,
        });
    }
    exported
}

#[cfg(test)]
mod report_tests {
    use super::*;
    use ndarray::{arr2, IxDyn};

    #[test]
    fn test_region_id_deterministic() {
        let id1 = compute_region_id("mel", &[2], &[2], [0.0, 1.0], [0.5, 1.5]);
        let id2 = compute_region_id("mel", &[2], &[2], [0.0, 1.0], [0.5, 1.5]);
        assert_eq!(id1, id2, "region_id must be deterministic");
        assert!(
            id1.starts_with("region:"),
            "region_id must start with 'region:'"
        );
    }

    #[test]
    fn test_region_id_changes_with_bounds() {
        let id1 = compute_region_id("mel", &[2], &[2], [0.0, 1.0], [0.5, 1.5]);
        let id2 = compute_region_id("mel", &[2], &[2], [0.0, 1.0], [0.5, 2.0]);
        assert_ne!(id1, id2, "different bounds must produce different IDs");
    }

    #[test]
    fn test_region_id_changes_with_shape() {
        let id1 = compute_region_id("mel", &[1, 2], &[1, 2], [0.0, 1.0], [0.5, 1.5]);
        let id2 = compute_region_id("mel", &[2, 1], &[2, 1], [0.0, 1.0], [0.5, 1.5]);
        assert_ne!(id1, id2, "different shapes must produce different IDs");
    }

    #[test]
    fn test_region_id_ignores_model_digest() {
        // Same bounds, same input name → same ID regardless of model
        let id1 = compute_region_id("mel", &[2], &[2], [-1.0, 0.0], [1.0, 0.5]);
        let id2 = compute_region_id("mel", &[2], &[2], [-1.0, 0.0], [1.0, 0.5]);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_region_id_matches_logical_order_for_non_contiguous_arrays() {
        let lower = arr2(&[[0.0_f32, 1.0], [2.0, 3.0]])
            .into_dyn()
            .permuted_axes(IxDyn(&[1, 0]));
        let upper = arr2(&[[0.5_f32, 1.5], [2.5, 3.5]])
            .into_dyn()
            .permuted_axes(IxDyn(&[1, 0]));
        assert!(
            lower.as_slice().is_none(),
            "test must use a strided lower tensor"
        );
        assert!(
            upper.as_slice().is_none(),
            "test must use a strided upper tensor"
        );

        let expected = compute_region_id(
            "mel",
            lower.shape(),
            upper.shape(),
            lower.iter().copied().collect::<Vec<_>>(),
            upper.iter().copied().collect::<Vec<_>>(),
        );
        let actual = compute_region_id(
            "mel",
            lower.shape(),
            upper.shape(),
            lower.iter().copied(),
            upper.iter().copied(),
        );

        assert_eq!(
            actual, expected,
            "region_id must hash logical tensor order even for strided arrays"
        );
    }

    #[test]
    fn test_bounds_file_name_sanitizes_colon() {
        let file = bounds_file_name("region:9f14c2ab77d0e1f4");
        assert_eq!(file, "top_bounds/region_9f14c2ab77d0e1f4.safetensors");
        assert!(!file.contains(':'), "bounds_file must not contain ':'");
    }

    #[test]
    fn test_bounds_file_name_preserves_safe_chars() {
        let file = bounds_file_name("my-region_01.test");
        assert_eq!(file, "top_bounds/my-region_01.test.safetensors");
    }
}
