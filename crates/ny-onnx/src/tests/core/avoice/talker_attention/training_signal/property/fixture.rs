// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Centroid-monotonicity spec construction and property sweep fixture assembly.
//!
//! Part of #4089 property lane decomposition.

use super::*;

/// Build a centroid monotonicity spec matrix for softmax output bounds.
///
/// For softmax output shape `[..., Q, K]`, the centroid at query position `t`
/// is `centroid(t) = Σ_j j * softmax[..., t, j]`. The monotonicity property
/// `centroid(t) >= centroid(t-1)` is encoded as one spec row per consecutive
/// (chunk, position) pair:
///
///   `Σ_j j * out[row_t, j] - Σ_j j * out[row_{t-1}, j] >= 0`
///
/// Uses the same positional weighting as `centroid_monotonicity_gaps` in
/// `talker_attention/mod.rs:321-345`.
fn build_centroid_monotonicity_spec(output_shape: &[usize]) -> Array2<f32> {
    assert!(
        output_shape.len() >= 2,
        "softmax output must have at least query/key axes, got {:?}",
        output_shape
    );
    let query_len = output_shape[output_shape.len() - 2];
    let key_len = *output_shape.last().expect("output_shape non-empty");
    let output_dim: usize = output_shape.iter().product();
    let num_rows = output_dim / key_len;
    assert!(
        query_len >= 2 && num_rows.is_multiple_of(query_len),
        "rows={num_rows} should decompose into query_len={query_len}"
    );

    let num_chunks = num_rows / query_len;
    let num_properties = num_chunks * (query_len - 1);
    let mut spec = Array2::zeros((num_properties, output_dim));
    let mut prop = 0;
    for chunk in 0..num_chunks {
        for t in 1..query_len {
            let row_cur = chunk * query_len + t;
            let row_prev = chunk * query_len + (t - 1);
            for j in 0..key_len {
                spec[[prop, row_cur * key_len + j]] = j as f32;
                spec[[prop, row_prev * key_len + j]] = -(j as f32);
            }
            prop += 1;
        }
    }
    spec
}

pub(super) struct TalkerPropertySweepFixture {
    pub(super) graph: GraphNetwork,
    pub(super) input: BoundedTensor,
    pub(super) spec_matrix: Array2<f32>,
    pub(super) node_bounds: HashMap<String, BoundedTensor>,
    pub(super) report: WeakRegionReport,
}

pub(super) fn build_talker_property_sweep_fixture(
    seq_len: usize,
    epsilon: f32,
) -> TalkerPropertySweepFixture {
    build_talker_property_sweep_fixture_with_deadline(seq_len, epsilon, None)
}

pub(super) fn build_talker_property_sweep_fixture_with_deadline(
    seq_len: usize,
    epsilon: f32,
    deadline: Option<Duration>,
) -> TalkerPropertySweepFixture {
    let (graph, _) = talker_attention_softmax_output_graph_for_seq_len(seq_len)
        .expect("short-seq talker softmax graph should build");
    let input = bounded_hidden_states_input(seq_len, epsilon);
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("node bounds should succeed on softmax subgraph");
    let output_bounds = node_bounds
        .get(graph.output_name())
        .expect("output node bounds must exist");
    let spec_matrix = build_centroid_monotonicity_spec(output_bounds.lower().shape());
    let config = RegionSweepConfig {
        primary_input: "hidden_states".to_string(),
        objective: SweepObjective::Linear {
            spec_matrix: Box::new(spec_matrix.clone()),
            thresholds: None,
        },
        regions: vec![RegionSpec {
            label: format!("seq{seq_len}_monotonicity"),
            lower: input.lower().to_owned(),
            upper: input.upper().to_owned(),
            metadata: Some(serde_json::json!({
                "sequence_len": seq_len,
                "objective": "centroid_monotonicity",
            })),
        }],
        deadline,
        top_k_bounds: 1,
        hotspot_limit: 3,
    };
    let source = SweepModelSource {
        model_name: "talker_attention_softmax".to_string(),
        model_path: None,
        model_digest: None,
    };
    let report = mine_weak_regions_graph(&graph, &source, &config)
        .expect("property-guided mining should succeed on talker softmax graph");

    TalkerPropertySweepFixture {
        graph,
        input,
        spec_matrix,
        node_bounds,
        report,
    }
}
