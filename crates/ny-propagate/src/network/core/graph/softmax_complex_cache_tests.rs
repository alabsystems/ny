// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Adversarial proof-cache coverage for the softmax structural rewrite.

use super::*;
use crate::layers::SoftmaxLayer;
use crate::network::core::graph::{
    CrownIbpCollectionCacheEntry, ForwardLinearCacheEntry, ForwardLinearCacheFingerprint,
    MarginOptMemoFingerprint,
};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

fn box_tensor(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape"),
    )
    .expect("valid box")
}

fn softmax_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "sm",
        Layer::Softmax(SoftmaxLayer::new(-1)),
    ));
    graph.set_output("sm");
    graph
}

fn install_stale_bound_caches(graph: &mut GraphNetwork) {
    let mut stale_map = HashMap::new();
    // Softmax outputs live in [0, 1], so this cannot be mistaken for a
    // plausible post-rewrite enclosure if it ever escapes the cache.
    stale_map.insert(
        "sm".to_string(),
        box_tensor(&[2, 3], vec![7.0; 6], vec![9.0; 6]),
    );
    let entry = ForwardLinearCacheEntry {
        fingerprint: ForwardLinearCacheFingerprint {
            hash: 0xdead_beef_dead_beef,
            canonical: Arc::from(&b"stale-pre-rewrite-entry"[..]),
        },
        map: Arc::new(stale_map),
        output_lb: None,
        build_cost: Duration::from_millis(1),
    };
    *graph
        .cached_forward_linear_map
        .fixed
        .get_mut()
        .expect("fixed cache lock") = Some(entry.clone());
    *graph
        .cached_forward_linear_map
        .alpha
        .get_mut()
        .expect("alpha cache lock") = Some(entry);
    *graph
        .cached_forward_linear_map
        .alpha_opt
        .get_mut()
        .expect("alpha-opt cache lock") = Some((
        MarginOptMemoFingerprint {
            hash: 0x0bad_c0de_0bad_c0de,
            canonical: b"stale-alpha-opt-entry".to_vec(),
        },
        None,
    ));
    graph
        .cached_crown_ibp_collection
        .slots
        .get_mut()
        .expect("CROWN-IBP cache lock")
        .push(CrownIbpCollectionCacheEntry {
            key: 0xabad_1dea,
            fingerprint: Arc::from(&b"stale-crown-ibp-entry"[..]),
            result: Arc::new(crate::types::GraphCrownIbpBoundsResult {
                bounds: HashMap::new(),
                provenance: HashMap::new(),
                fallback_events: Vec::new(),
            }),
            complete: true,
            crown_count: 0,
            truncated_reuse_scope: None,
        });
}

fn cache_occupancy(graph: &GraphNetwork) -> (bool, bool, bool, usize) {
    (
        graph
            .cached_forward_linear_map
            .fixed
            .read()
            .expect("fixed cache lock")
            .is_some(),
        graph
            .cached_forward_linear_map
            .alpha
            .read()
            .expect("alpha cache lock")
            .is_some(),
        graph
            .cached_forward_linear_map
            .alpha_opt
            .read()
            .expect("alpha-opt cache lock")
            .is_some(),
        graph
            .cached_crown_ibp_collection
            .slots
            .read()
            .expect("CROWN-IBP cache lock")
            .len(),
    )
}

/// The public rewrite changes both graph structure and the function computed
/// by node `sm`. All four bound-cache slots must be empty afterwards; clearing
/// only execution metadata leaves a pre-rewrite proof object reusable under a
/// request key that contains no graph-structure bytes.
#[test]
fn rewrite_clears_forward_linear_and_crown_ibp_bound_caches() {
    let mut graph = softmax_graph();
    install_stale_bound_caches(&mut graph);
    assert_eq!(
        cache_occupancy(&graph),
        (true, true, true, 1),
        "adversarial fixture must populate every bound-cache family"
    );

    let input = box_tensor(
        &[2, 3],
        vec![-1.0, 1.0, -2.0, 4.0, -4.0, 0.0],
        vec![1.0, 3.0, 0.0, 6.0, -2.0, 2.0],
    );
    let report = graph
        .decompose_softmax_complex(&input)
        .expect("eligible rewrite");
    assert_eq!(
        report.decomposed,
        vec!["sm".to_string()],
        "test must exercise a real structural rewrite"
    );
    assert_eq!(graph.num_nodes(), 5, "four helper nodes must be inserted");
    assert_eq!(
        cache_occupancy(&graph),
        (false, false, false, 0),
        "post-rewrite graph retained a proof cache for the old function"
    );
}
