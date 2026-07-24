// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod bounds;
mod distance;
mod graphs;
mod progress;

pub(super) use self::bounds::{
    build_component_node_bounds, scalar_spec_bounds_with_node_bounds, scalar_width,
};
pub(super) use self::distance::speaker_cosine_distance_upper;
pub(super) use self::graphs::{
    build_speaker_cosine_component_graphs, build_speaker_cosine_component_graphs_round_trip,
};
