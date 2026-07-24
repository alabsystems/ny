// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common;
use super::*;

mod aux_freeze;
mod contract;
mod graphs;

pub(super) use aux_freeze::{
    compute_qwen3_rope_tables, load_talker_attention_with_fixed_aux,
    load_talker_attention_with_fixed_aux_for_seq_len, load_talker_attention_with_real_rope_seq_len,
};
pub(super) use contract::{
    avoice_talker_attention_raw, QWEN3_TTS_ROPE_BASE, TALKER_ATTENTION_EPSILON,
    TALKER_ATTENTION_HIDDEN_DIM, TALKER_ATTENTION_ROPE_DIM, TALKER_ATTENTION_SEQ_LEN,
    TALKER_ATTENTION_SHORT_SEQ_LEN,
};
pub(super) use graphs::{
    bounded_hidden_states_input, bounded_hidden_states_input_centered,
    configure_sound_softmax_modes, first_talker_attention_softmax_node,
    talker_attention_graph_with_fixed_aux_for_seq_len, talker_attention_softmax_output_graph,
    talker_attention_softmax_output_graph_for_seq_len,
    talker_attention_softmax_output_graph_real_rope,
};
