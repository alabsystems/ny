// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::super::crown::CrownIbpCollectOptions;
use super::*;
use crate::layers::{BatchNormLayer, LinearLayer, ReLULayer};
use crate::network::GraphNode;
use crate::types::{BoundsProvenance, CrownIbpFallbackReason, GraphCrownIbpBoundsResult};
use ndarray::{arr1, arr2, Array2};

// Genuine source: p=ReLU(a),q=ReLU(-a). Only h0=p+q-1.5 crosses;
// h1=p-q+2 is stable. No injected row limit manufactures an unselected row.
fn fixture(raw_bn: bool) -> (GraphNetwork, BoundedTensor, &'static str) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "signed",
        Layer::Linear(LinearLayer::new(arr2(&[[1f32], [-1.]]), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "pq",
        Layer::ReLU(ReLULayer),
        vec!["signed".into()],
    ));
    graph.add_node(GraphNode::new(
        "head",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1., 1.], [1., -1.]]), Some(arr1(&[-1.5, 2.]))).unwrap(),
        ),
        vec!["pq".into()],
    ));
    let target = if raw_bn {
        let one = arr1(&[1f32, 1.]).into_dyn();
        let zero = arr1(&[0f32, 0.]).into_dyn();
        graph.add_node(GraphNode::new(
            "bn",
            Layer::BatchNorm(BatchNormLayer::new(&one, &zero, &zero, &one, 0.).unwrap()),
            vec!["head".into()],
        ));
        "bn"
    } else {
        "head"
    };
    graph.add_node(GraphNode::new(
        "act",
        Layer::ReLU(ReLULayer),
        vec![target.into()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1f32, -1.], [-1., 1.]]), None).unwrap()),
        vec!["act".into()],
    ));
    graph.set_output("out");
    let input = BoundedTensor::new(arr1(&[-1f32]).into_dyn(), arr1(&[1f32]).into_dyn()).unwrap();
    (graph, input, target)
}
fn demand(target: &str) -> HashSet<String> {
    HashSet::from([target.to_string()])
}
fn bits(bound: &BoundedTensor) -> (Vec<u32>, Vec<u32>) {
    (
        bound.lower().iter().map(|v| v.to_bits()).collect(),
        bound.upper().iter().map(|v| v.to_bits()).collect(),
    )
}
fn transaction(target: &str) -> HeadRowsTransaction {
    HeadRowsTransaction {
        target: target.into(),
        rows: [0; 8],
        selected: 1,
        eligible: 1,
        total: 2,
        entered: true,
        stage: "entered",
        committed: None,
        seed_bytes: 16,
        estimated_macs: 1,
        elapsed_secs: 0.,
        grant_secs: 0.,
        rate_macs_per_sec: 0,
        gpu: bounded_gpu::Observation::default(),
    }
}
fn box2(lo: [f32; 2], hi: [f32; 2]) -> BoundedTensor {
    BoundedTensor::new(arr1(&lo).into_dyn(), arr1(&hi).into_dyn()).unwrap()
}

#[test]
fn selected_head_genuine_source_and_raw_bn_complete_one_crossing_identity_row() {
    for raw_bn in [false, true] {
        let (graph, input, target) = fixture(raw_bn);
        assert!(
            graph.use_patches_mode,
            "default graph mode must not make the new route dormant"
        );
        let original = graph.collect_node_bounds(&input).unwrap();
        let before: HashMap<_, _> = original.iter().map(|(n, b)| (n.clone(), bits(b))).collect();
        let base = &original[target];
        assert!(base.lower()[[0]] <= -1.5 && base.upper()[[0]] >= 0.5);
        assert!(base.lower()[[1]] > 0.99);
        let end = Instant::now() + Duration::from_secs(30);
        let mut got = run(
            &graph,
            &input,
            &original,
            &demand(target),
            None,
            end,
            32 << 20,
            true,
        )
        .unwrap();
        assert_eq!((got.selected, got.eligible, got.total), (1, 1, 2));
        assert_eq!(got.rows[0], 0);
        assert!(got.entered);
        assert_eq!(got.stage, "committed");
        let (published, count) = got.publish(base, base, Some(end));
        assert_eq!(count, 1);
        // Upper = -1/2 in exact arithmetic; permit only conservative outward loss.
        assert!(published.upper()[[0]] >= -0.5 && published.upper()[[0]] < -0.49);
        assert!(published.lower()[[0]] <= -1.5 && published.lower()[[0]] > -1.501);
        assert_eq!(
            published.lower()[[1]].to_bits(),
            base.lower()[[1]].to_bits()
        );
        assert_eq!(
            published.upper()[[1]].to_bits(),
            base.upper()[[1]].to_bits()
        );
        assert!(
            original.iter().all(|(n, b)| before[n] == bits(b)),
            "read-only source map custody"
        );
        assert_eq!(
            got.publish(base, base, Some(end)).1,
            0,
            "private commit is consumed once"
        );
    }
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep the original fixture owner intact while mutating an independent candidate."
)]
fn selected_head_planner_bounds_whole_prefix_rows_names_and_source_shape() {
    let (mut graph, input, target) = fixture(false);
    let original = graph.collect_node_bounds(&input).unwrap();
    let end = Instant::now() + Duration::from_secs(30);
    assert!(plan(&graph, &original, &HashSet::new(), end).is_none());
    let mut wrong = original.clone();
    wrong.insert(
        target.into(),
        BoundedTensor::new(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 1, 1]), -1.),
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 1, 1]), 1.),
        )
        .unwrap(),
    );
    assert!(plan(&graph, &wrong, &demand(target), end).is_none());
    for i in graph.num_nodes()..=MAX_PREFIX_NODES {
        graph.add_node(GraphNode::from_input(
            format!("unused-{i}"),
            Layer::Linear(LinearLayer::new(arr2(&[[1f32]]), None).unwrap()),
        ));
    }
    assert!(
        plan(&graph, &original, &demand(target), end).is_none(),
        "whole-prefix inspection is bounded"
    );
    let mut wide = GraphNetwork::new();
    wide.add_node(GraphNode::from_input(
        "head",
        Layer::Linear(LinearLayer::new(Array2::ones((9, 1)), None).unwrap()),
    ));
    wide.add_node(GraphNode::new(
        "act",
        Layer::ReLU(ReLULayer),
        vec!["head".into()],
    ));
    wide.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(Array2::ones((1, 9)), None).unwrap()),
        vec!["act".into()],
    ));
    wide.set_output("out");
    let boxes = wide.collect_node_bounds(&input).unwrap();
    let ranked = plan(&wide, &boxes, &demand("head"), end).unwrap();
    assert_eq!(
        (ranked.selected, ranked.eligible, ranked.ranked),
        (8, 9, [0, 1, 2, 3, 4, 5, 6, 7])
    );
    let mut malformed = boxes.clone();
    malformed.remove("act");
    assert!(plan(&wide, &malformed, &demand("head"), end).is_none());
}

#[test]
fn selected_head_reservation_preserves_original_width_floor_cap_and_absolute_end() {
    let now = Instant::now();
    let end = now + Duration::from_secs(40);
    let policy = CrownIbpPerNodeTimeBudget::default();
    assert_eq!(
        reservation(now, end, 512, &policy),
        Some(now + Duration::from_secs(2))
    );
    assert!(reservation(now, now + Duration::from_secs(7), 512, &policy).is_none());
    let capped = CrownIbpPerNodeTimeBudget {
        cap_secs: Some(1.),
        ..policy
    };
    assert!(reservation(now, end, 512, &capped).is_none());
    let small_floor = CrownIbpPerNodeTimeBudget {
        floor_secs: Some(0.1),
        ..policy
    };
    assert_eq!(
        reservation(now, now + Duration::from_secs(4), 512, &small_floor),
        Some(now + Duration::from_secs(1))
    );
    assert!(reservation(
        now,
        now.checked_sub(Duration::from_millis(1)).unwrap(),
        512,
        &small_floor
    )
    .is_none());
    assert!(local_bytes(usize::MAX, 8192).is_none());
}

#[test]
fn selected_head_memory_floor_and_expired_admission_never_enter_backward() {
    let (mut graph, input, target) = fixture(false);
    let original = graph.collect_node_bounds(&input).unwrap();
    let end = Instant::now() + Duration::from_secs(30);
    let mut refused = run(
        &graph,
        &input,
        &original,
        &demand(target),
        None,
        end,
        0,
        true,
    )
    .unwrap();
    assert!(!refused.entered);
    assert_eq!(refused.stage, "memory");
    let (published, n) = refused.publish(&original[target], &original[target], Some(end));
    assert_eq!(n, 0);
    assert_eq!(bits(&published), bits(&original[target]));
    graph.crown_ibp_per_node_time_budget.floor_secs = Some(3.);
    let refused = run(
        &graph,
        &input,
        &original,
        &demand(target),
        None,
        end,
        32 << 20,
        true,
    )
    .unwrap();
    assert!(!refused.entered);
    assert_eq!(refused.stage, "budget");
    assert!(run(
        &graph,
        &input,
        &original,
        &demand(target),
        None,
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
        32 << 20,
        true
    )
    .is_none());
}

#[test]
fn selected_head_atomic_commit_refuses_late_disjoint_error_and_wrong_identity_count() {
    let base = box2([-1.5, 1.], [0.5, 3.]);
    let candidate = box2([-1.5, -99.], [-0.5, 99.]);
    let live_end = Instant::now() + Duration::from_secs(30);
    let mut late = transaction("head");
    late.finish(
        &base,
        RowRefineOutcome::Refined {
            bounds: Box::new(candidate.clone()),
            k: 1,
            secs: 0.,
        },
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    assert_eq!(late.stage, "publication-deadline");
    assert!(late.committed.is_none());
    for outcome in [
        RowRefineOutcome::Declined("typed-refusal"),
        RowRefineOutcome::Disjoint { count: 1 },
        RowRefineOutcome::Failed(ny_core::NyError::DeadlineExceeded("late".into())),
        RowRefineOutcome::Failed(ny_core::NyError::InvalidSpec("shape".into())),
        RowRefineOutcome::Refined {
            bounds: Box::new(candidate.clone()),
            k: 2,
            secs: 0.,
        },
    ] {
        let mut state = transaction("head");
        state.finish(&base, outcome, live_end);
        assert!(state.committed.is_none());
        let (published, n) = state.publish(&base, &base, Some(live_end));
        assert_eq!(n, 0);
        assert_eq!(bits(&published), bits(&base));
    }
    assert!(selected_intersection(&base, &box2([1., 1.], [2., 3.]), &[0]).is_none());
    assert!(selected_intersection(&base, &candidate, &[0, 0]).is_none());
    assert!(selected_intersection(&base, &candidate, &[2]).is_none());
}

#[test]
fn selected_head_on_time_commit_survives_late_visit_but_new_composition_requires_time() {
    let base = box2([-1.5, 1.], [0.5, 3.]);
    let candidate = box2([-1.5, -99.], [-0.5, 99.]);
    let live_end = Instant::now() + Duration::from_secs(30);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    let fresh = || {
        let mut s = transaction("head");
        s.finish(
            &base,
            RowRefineOutcome::Refined {
                bounds: Box::new(candidate.clone()),
                k: 1,
                secs: 0.,
            },
            live_end,
        );
        s
    };
    let mut state = fresh();
    let (kept, n) = state.publish(&base, &base, Some(expired));
    assert_eq!(n, 1);
    assert_eq!(kept.upper()[[0]], -0.5);
    let composed = box2([-1.4, 1.25], [0.4, 2.75]);
    let mut state = fresh();
    let (merged, n) = state.publish(&composed, &base, Some(live_end));
    assert_eq!(n, 1);
    assert_eq!(merged.lower()[[0]], -1.4);
    assert_eq!(merged.upper()[[0]], -0.5);
    assert_eq!(
        merged.lower()[[1]].to_bits(),
        composed.lower()[[1]].to_bits()
    );
    assert_eq!(
        merged.upper()[[1]].to_bits(),
        composed.upper()[[1]].to_bits()
    );
    let mut state = fresh();
    let (late, n) = state.publish(&composed, &base, Some(expired));
    assert_eq!(n, 0);
    assert_eq!(bits(&late), bits(&composed));
}

#[test]
fn selected_head_default_scope_bypasses_complete_cache_and_never_retries_declined_target() {
    let (mut graph, input, target) = fixture(false);
    // Populate a real complete same-source cache first. Enabled policy must
    // still perform its own bounded opportunity instead of serving that map.
    graph
        .collect_crown_ibp_bounds_dag_with_options(&input, CrownIbpCollectOptions::default())
        .unwrap();
    let hits = graph.crown_ibp_collection_cache_hits();
    graph.crown_ibp_per_node_time_budget.floor_secs = Some(3.);
    let original = graph.collect_node_bounds(&input).unwrap();
    let result = graph
        .collect_crown_ibp_bounds_dag_with_options(
            &input,
            CrownIbpCollectOptions {
                deadline: Some(Instant::now() + Duration::from_secs(30)),
                deadline_includes_ibp: true,
                precomputed_ibp: Some(original.clone()),
                original_root_head_rows: Some(OriginalRootHeadRows),
                ..CrownIbpCollectOptions::default()
            },
        )
        .unwrap();
    assert_eq!(graph.crown_ibp_collection_cache_hits(), hits);
    assert_eq!(bits(&result.bounds[target]), bits(&original[target]));
    assert_eq!(
        result.provenance[target],
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::SelectedHeadRows)
    );
    let events: Vec<_> = result
        .fallback_events
        .iter()
        .filter(|e| e.reason == CrownIbpFallbackReason::SelectedHeadRows)
        .collect();
    assert_eq!(events.len(), 1);
    assert!(events[0].details.contains("entered=false"));
    assert!(events[0].details.contains("full_width_retry=0"));
    assert!(!result.all_requested_crown_targets_completed());
}

#[test]
fn selected_head_partial_provenance_and_event_each_refuse_complete_credit() {
    let reason = CrownIbpFallbackReason::SelectedHeadRows;
    let mut result = GraphCrownIbpBoundsResult {
        bounds: HashMap::new(),
        provenance: HashMap::from([("head".into(), BoundsProvenance::ForwardFallback(reason))]),
        fallback_events: Vec::new(),
    };
    assert!(!result.all_requested_crown_targets_completed());
    result.provenance.clear();
    result
        .fallback_events
        .push(crate::types::CrownIbpFallbackEvent {
            layer_index: 0,
            layer_type: "Linear".into(),
            reason,
            details: "completed=1 total=2".into(),
        });
    assert!(!result.all_requested_crown_targets_completed());
    assert_eq!(
        super::super::ref_map_telemetry::classify(BoundsProvenance::ForwardFallback(reason)),
        super::super::ref_map_telemetry::ProvenanceClass::OtherFallback
    );
}
