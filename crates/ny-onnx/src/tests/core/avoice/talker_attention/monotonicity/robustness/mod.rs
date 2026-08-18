// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::centroid::{centroid_bounds_from_softmax, centroid_monotonicity_gaps};
use super::super::fixtures::{
    bounded_hidden_states_input, bounded_hidden_states_input_centered,
    configure_sound_softmax_modes, first_talker_attention_softmax_node,
    load_talker_attention_with_real_rope_seq_len, talker_attention_softmax_output_graph,
    TALKER_ATTENTION_SEQ_LEN,
};
use super::*;

mod center;
mod scaling;
