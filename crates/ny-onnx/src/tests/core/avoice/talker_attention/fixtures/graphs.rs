// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::aux_freeze::{
    load_talker_attention_with_fixed_aux_for_seq_len, load_talker_attention_with_real_rope_seq_len,
};
use super::contract::{talker_fixture_contract, TALKER_ATTENTION_SEQ_LEN};
use super::*;

pub(in super::super) fn configure_sound_softmax_modes(mut graph: GraphNetwork) -> GraphNetwork {
    graph.set_softmax_sound_mode(true);
    graph.set_causal_softmax_sound_mode(true);
    graph
}

pub(in super::super) fn first_talker_attention_softmax_node(graph: &GraphNetwork) -> String {
    let softmax_nodes = common::node_names_by_layer_types(graph, &["Softmax", "CausalSoftmax"]);
    assert!(
        !softmax_nodes.is_empty(),
        "expected at least one Softmax/CausalSoftmax node in talker attention graph; \
         softmax hits={:?}, causal hits={:?}",
        common::node_name_hits(graph, "softmax"),
        common::node_name_hits(graph, "causal")
    );
    softmax_nodes[0].clone()
}

pub(in super::super) fn talker_attention_softmax_output_graph_real_rope() -> (GraphNetwork, String)
{
    let model = load_talker_attention_with_real_rope_seq_len(TALKER_ATTENTION_SEQ_LEN);
    let graph = configure_sound_softmax_modes(
        model
            .to_graph_network()
            .expect("talker attention with real RoPE should convert to GraphNetwork"),
    );
    let softmax_name = first_talker_attention_softmax_node(&graph);
    let mut softmax_graph = graph;
    softmax_graph.set_output(softmax_name.clone());
    (softmax_graph, softmax_name)
}

pub(in super::super) fn talker_attention_graph_with_fixed_aux_for_seq_len(
    seq_len: usize,
) -> Result<GraphNetwork, String> {
    let model = load_talker_attention_with_fixed_aux_for_seq_len(seq_len);
    let graph = model
        .to_graph_network()
        .map_err(|e| format!("graph conversion failed at seq_len={seq_len}: {e}"))?;
    Ok(configure_sound_softmax_modes(graph))
}

pub(in super::super) fn talker_attention_softmax_output_graph() -> (GraphNetwork, String) {
    talker_attention_softmax_output_graph_for_seq_len(TALKER_ATTENTION_SEQ_LEN).unwrap_or_else(
        |e| {
            panic!(
                "exported talker-attention softmax graph should build at seq_len={}: {e}",
                TALKER_ATTENTION_SEQ_LEN
            )
        },
    )
}

pub(in super::super) fn talker_attention_softmax_output_graph_for_seq_len(
    seq_len: usize,
) -> Result<(GraphNetwork, String), String> {
    let graph = talker_attention_graph_with_fixed_aux_for_seq_len(seq_len)?;
    let softmax_name = first_talker_attention_softmax_node(&graph);
    let mut softmax_graph = graph;
    softmax_graph.set_output(softmax_name.clone());
    Ok((softmax_graph, softmax_name))
}

/// Build a bounded hidden_states input for the talker attention model.
///
/// Shape: [seq_len, hidden_dim] (batch axis stripped for propagation).
/// Center: all zeros. Bounds: center +/- epsilon.
pub(in super::super) fn bounded_hidden_states_input(seq_len: usize, epsilon: f32) -> BoundedTensor {
    let contract = talker_fixture_contract();
    let shape = [seq_len, contract.hidden_dim];
    let center = ArrayD::zeros(IxDyn(&shape));
    BoundedTensor::from_epsilon(center, epsilon)
        .expect("bounded_hidden_states_input should build a valid epsilon box")
}

/// Build a bounded hidden_states input centered at a uniform non-zero value.
///
/// Used to verify that monotonicity results are not artifacts of the zero-center
/// choice. A non-zero center produces content-dependent attention patterns
/// (Q*center != 0), exercising the real-weight attention computation.
pub(in super::super) fn bounded_hidden_states_input_centered(
    seq_len: usize,
    center_value: f32,
    epsilon: f32,
) -> BoundedTensor {
    let contract = talker_fixture_contract();
    let shape = [seq_len, contract.hidden_dim];
    let center = ArrayD::from_elem(IxDyn(&shape), center_value);
    BoundedTensor::from_epsilon(center, epsilon)
        .expect("bounded_hidden_states_input_centered should build a valid epsilon box")
}
