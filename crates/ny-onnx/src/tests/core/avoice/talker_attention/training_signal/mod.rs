// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Talker attention training_signal integration tests (Packets B+C).
//!
//! Split from the monolithic `training_signal.rs` into lane-specific leaves:
//! - `output_width.rs` — output-width smoke/ranking (Packet B)
//! - `property/` — property-guided centroid-monotonicity (Packet C)

use super::*;
use crate::training_signal::{
    mine_weak_regions_graph, mine_weak_regions_model, write_weak_region_report, RegionSpec,
    RegionSweepConfig, SweepModelSource, SweepObjective, WeakRegionRecord, WeakRegionReport,
};
use ndarray::Array2;
use ny_propagate::types::{BoundsProvenance, CrownIbpFallbackReason};
use ny_tensor::BoundedTensor;
use std::time::Duration;

mod output_width;
mod property;
