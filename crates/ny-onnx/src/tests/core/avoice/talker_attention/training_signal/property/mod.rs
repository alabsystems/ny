// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Talker attention property-guided centroid-monotonicity tests (#3520 Packet C).
//!
//! Builds a centroid-monotonicity spec matrix and runs `mine_weak_regions_graph`
//! with `SweepObjective::Linear` on the short-seq softmax subgraph.
//!
//! Split from flat `property.rs` into lane-specific leaves:
//! - `fixture.rs` — spec construction and sweep fixture assembly
//! - `metrics.rs` — direct oracle and metric parity contract helpers
//! - `regressions.rs` — Packet C regression tests (#3520)
//!
//! Part of #4089 property lane decomposition.

use super::super::super::training_signal_support::assert_report_artifacts;
use super::super::centroid::{centroid_bounds_from_softmax, centroid_monotonicity_gaps};
use super::super::fixtures::{
    bounded_hidden_states_input, talker_attention_softmax_output_graph_for_seq_len,
    TALKER_ATTENTION_SHORT_SEQ_LEN,
};
use super::*;

mod fixture;
mod metrics;
mod regressions;
