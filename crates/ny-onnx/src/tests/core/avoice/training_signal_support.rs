// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared output-width report assertion helpers for avoice training_signal
//! tests (#3834).
//!
//! All three avoice model families (speaker, talker, Kokoro) duplicate the
//! same weak-region mining scaffold: metrics+provenance checks, hotspot contract,
//! and report artifact layout assertions. This module centralizes those into a
//! single surface with enum-based contract selectors.

use crate::training_signal::WeakRegionRecord;
use ny_propagate::types::BoundsProvenance;

/// Provenance contract for output-width assertions.
///
/// Speaker requires strict Crown provenance; talker and Kokoro accept
/// `ForwardFallback(_)` as a valid runtime outcome on real-weight surfaces.
pub(super) enum OutputWidthProvenanceContract {
    StrictCrown,
    CrownOrForwardFallback,
}

/// Hotspot count contract for output-width assertions.
pub(super) struct HotspotContract {
    pub(super) min_count: usize,
    pub(super) max_count: usize,
}

/// Assert output metrics are finite and positive, and provenance matches
/// the given contract.
pub(super) fn assert_output_width_metrics_and_provenance(
    record: &WeakRegionRecord,
    contract: OutputWidthProvenanceContract,
) {
    assert!(
        record.output_width_max.is_finite(),
        "output_width_max must be finite, got {}",
        record.output_width_max
    );
    assert!(
        record.output_width_mean.is_finite(),
        "output_width_mean must be finite, got {}",
        record.output_width_mean
    );
    assert!(
        record.output_width_max > 0.0,
        "output_width_max must be positive for non-trivial bounds"
    );
    assert_eq!(record.method_requested, "batched_crown");

    match contract {
        OutputWidthProvenanceContract::StrictCrown => {
            assert_eq!(record.method_actual, "batched_crown");
            assert_eq!(record.provenance, BoundsProvenance::Crown);
        }
        OutputWidthProvenanceContract::CrownOrForwardFallback => {
            assert!(
                record.method_actual == "batched_crown"
                    || record.method_actual == "forward_fallback",
                "unexpected method_actual: {}",
                record.method_actual
            );
            match record.provenance {
                BoundsProvenance::Crown => {
                    assert_eq!(record.method_actual, "batched_crown")
                }
                BoundsProvenance::ForwardFallback(_) => {
                    assert_eq!(record.method_actual, "forward_fallback")
                }
            }
        }
    }
}

/// Assert hotspots satisfy the contract: count within [min, max],
/// with finite metrics and non-empty metadata fields.
pub(super) fn assert_hotspot_contract(record: &WeakRegionRecord, contract: HotspotContract) {
    assert!(
        record.top_hotspots.len() >= contract.min_count,
        "expected at least {} hotspot(s), got {}",
        contract.min_count,
        record.top_hotspots.len()
    );
    assert!(
        record.top_hotspots.len() <= contract.max_count,
        "hotspot count {} exceeds limit {}",
        record.top_hotspots.len(),
        contract.max_count
    );
    for hs in &record.top_hotspots {
        assert!(!hs.name.is_empty(), "hotspot name must be non-empty");
        assert!(
            !hs.layer_type.is_empty(),
            "hotspot layer_type must be non-empty"
        );
        assert!(hs.max_width.is_finite(), "hotspot max_width must be finite");
        assert!(
            hs.mean_width.is_finite(),
            "hotspot mean_width must be finite"
        );
        assert!(
            hs.growth_ratio.is_finite(),
            "hotspot growth_ratio must be finite"
        );
        assert!(!hs.status.is_empty(), "hotspot status must be non-empty");
    }
}

/// Assert report artifacts: manifest.json, weak_regions.jsonl, top_bounds/.
pub(super) fn assert_report_artifacts(
    record: &WeakRegionRecord,
    dir: &std::path::Path,
    winners: usize,
) {
    assert!(
        dir.join("manifest.json").exists(),
        "manifest.json must be written"
    );
    assert!(
        dir.join("weak_regions.jsonl").exists(),
        "weak_regions.jsonl must be written"
    );

    let entries: Vec<_> = std::fs::read_dir(dir.join("top_bounds"))
        .expect("read bounds dir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        entries.len(),
        winners,
        "expected {winners} winner safetensors files"
    );

    let bf = record
        .bounds_file
        .as_ref()
        .expect("winner should carry bounds_file");
    assert!(
        bf.starts_with("top_bounds/"),
        "bounds_file must be relative: {bf}"
    );
    assert!(
        dir.join(bf).exists(),
        "bounds_file must resolve: {}",
        dir.join(bf).display()
    );
}
