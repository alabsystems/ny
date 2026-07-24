// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Weak-region mining for verification-guided training (#3520).
//!
//! This module identifies which input regions produce the widest (weakest)
//! CROWN output bounds, enabling downstream training to focus on model
//! weaknesses.
//!
//! Two ranking lanes are supported via [`SweepObjective`]:
//!
//! - **Uncertainty** (Packet A/B/D): rank by output-node bound width.
//! - **Property** (Packet C): rank by certified slack under a linear spec
//!   `Cx >= tau`, using spec-guided CROWN with provenance.

mod report;
mod runner;
mod scoring;
mod types;

pub use report::write_weak_region_report;
pub use runner::{mine_weak_regions, mine_weak_regions_graph, mine_weak_regions_model};
pub use types::{
    RegionSpec, RegionSweepConfig, SweepManifest, SweepModelSource, SweepObjective,
    WeakRegionHotspot, WeakRegionRecord, WeakRegionReport,
};

#[cfg(test)]
mod tests;
