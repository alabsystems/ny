// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::centroid::{
    centroid_bounds_from_softmax, centroid_monotonicity_gaps, centroid_monotonicity_stats,
    CentroidMonotonicityStats,
};
use super::super::fixtures::{
    bounded_hidden_states_input, compute_qwen3_rope_tables, talker_attention_softmax_output_graph,
    talker_attention_softmax_output_graph_real_rope, QWEN3_TTS_ROPE_BASE, TALKER_ATTENTION_EPSILON,
    TALKER_ATTENTION_ROPE_DIM, TALKER_ATTENTION_SEQ_LEN,
};
use super::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::ArrayD;

mod named_node;
mod sweep;
mod tables;
