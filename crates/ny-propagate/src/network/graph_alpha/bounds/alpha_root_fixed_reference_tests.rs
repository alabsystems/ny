// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::crown::RootFixedReferenceCollectionCounter;
use super::*;
use crate::layers::{
    AddLayer, BatchNormLayer, Conv2dLayer, FlattenLayer, Layer, LinearLayer, ReLULayer,
};
use crate::network::GraphNode;
use crate::NETWORK_INPUT;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

fn config() -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations: 1,
        gradient_method: GradientMethod::AnalyticChain,
        fix_interm_bounds: true,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        ..AlphaCrownConfig::default()
    }
}

fn dag_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "left",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -0.4], [0.7, 0.9]]),
                Some(arr1(&[0.1, -0.2])),
            )
            .unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "left_relu",
        Layer::ReLU(ReLULayer),
        vec!["left".into()],
    ));
    graph.add_node(GraphNode::from_input(
        "right",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[-0.6_f32, 1.1], [0.8, -0.5]]),
                Some(arr1(&[-0.15, 0.05])),
            )
            .unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "right_relu",
        Layer::ReLU(ReLULayer),
        vec!["right".into()],
    ));
    graph.add_node(GraphNode::binary(
        "sum",
        Layer::Add(AddLayer),
        "left_relu",
        "right_relu",
    ));
    graph.add_node(GraphNode::new(
        "output",
        Layer::Linear(LinearLayer::new(arr2(&[[1.2_f32, -0.9]]), Some(arr1(&[0.07]))).unwrap()),
        vec!["sum".into()],
    ));
    graph.set_output("output");
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.7]).into_dyn(),
        arr1(&[1.3_f32, 1.1]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

fn sequential_raw_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![1.0_f32, -1.0]).unwrap(),
        None,
        (1, 1),
        (0, 0),
        1,
        1,
    )
    .unwrap();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    let ones = arr1(&[1.0_f32, 1.0]).into_dyn();
    let zeros = arr1(&[0.0_f32, 0.0]).into_dyn();
    let bn = BatchNormLayer::new(&ones, &zeros, &zeros, &ones, 0.0).unwrap();
    graph.add_node(GraphNode::new(
        "raw_bn",
        Layer::BatchNorm(bn),
        vec!["conv".into()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["raw_bn".into()],
    ));
    graph.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec!["relu".into()],
    ));
    graph.add_node(GraphNode::new(
        "output",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, 1.0]]), None).unwrap()),
        vec!["flat".into()],
    ));
    graph.set_output("output");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0_f32),
    )
    .unwrap();
    (graph, input)
}

fn map_bits(map: &HashMap<String, BoundedTensor>) -> BTreeMap<String, (Vec<u32>, Vec<u32>)> {
    map.iter()
        .map(|(name, bounds)| {
            (
                name.clone(),
                (
                    bounds.lower().iter().map(|x| x.to_bits()).collect(),
                    bounds.upper().iter().map(|x| x.to_bits()).collect(),
                ),
            )
        })
        .collect()
}

fn actual_artifact(graph: &GraphNetwork, input: &BoundedTensor) -> DagAlphaCollectionArtifact {
    let cfg = config();
    let reference = graph
        .collect_crown_ibp_bounds_dag_with_options(
            input,
            reference_options(&cfg, None, Duration::from_secs(5)),
        )
        .expect("same-call certified reference");
    let collected = graph
        .propagate_dag_alpha_crown_collect_root_fixed_reference(
            input,
            &cfg,
            None,
            PrecomputedAlphaReferenceBounds {
                bounds: reference.bounds,
                source: AlphaReferenceBoundsSource::CrownIbp,
            },
        )
        .expect("actual fixed-reference optimization");
    let RootFixedReferenceCollection::Optimized(artifact) = collected else {
        panic!("unstable ReLU fixture must optimize");
    };
    artifact
}

fn assert_fresh_sample_enclosure(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    result: super::super::GraphAlphaCollectionResult,
) {
    let (map, alpha) = result;
    let evaluated = graph
        .collect_crown_bounds_with_alpha(input, &map, &alpha, None, None)
        .expect("fresh certified objective evaluation");
    let output = &evaluated["output"];
    let lo = *output.lower().iter().next().unwrap();
    let hi = *output.upper().iter().next().unwrap();
    assert!(lo.is_finite() && hi.is_finite() && lo <= hi);
    for x in [-1.0_f32, 0.15, 1.3] {
        for y in [-0.7_f32, 0.2, 1.1] {
            let l0 = (x - 0.4 * y + 0.1).max(0.0);
            let l1 = (0.7 * x + 0.9 * y - 0.2).max(0.0);
            let r0 = (-0.6 * x + 1.1 * y - 0.15).max(0.0);
            let r1 = (0.8 * x - 0.5 * y + 0.05).max(0.0);
            let value = 1.2 * (l0 + r0) - 0.9 * (l1 + r1) + 0.07;
            assert!(lo <= value && value <= hi, "{value} outside [{lo}, {hi}]");
        }
    }
}

#[ntest::timeout(30000)]
#[test]
fn root_fixed_reference_dag_collects_once_and_supports_fresh_evaluation() {
    let (graph, input) = dag_fixture();
    assert!(!graph.is_sequential_graph(graph.exec_order().unwrap()));
    let count = RootFixedReferenceCollectionCounter::start();
    let outcome = graph
        .collect_root_fixed_reference_alpha_with_engine(
            &input,
            &config(),
            None,
            Duration::from_secs(5),
        )
        .expect("root collection");
    assert_eq!(
        count.entries(),
        1,
        "pre-loop and tail must not repeat CROWN-IBP collection"
    );
    let GraphAlphaCollectionOutcome::Complete(result) = outcome else {
        panic!("unexpired tiny phase");
    };
    assert_fresh_sample_enclosure(&graph, &input, result);
}

#[ntest::timeout(30000)]
#[test]
fn root_fixed_reference_sequential_raw_conv_bn_uses_explicit_dag_core() {
    let (graph, input) = sequential_raw_fixture();
    assert!(graph.is_sequential_graph(graph.exec_order().unwrap()));
    let count = RootFixedReferenceCollectionCounter::start();
    let outcome = graph
        .collect_root_fixed_reference_alpha_with_engine(
            &input,
            &config(),
            None,
            Duration::from_secs(5),
        )
        .expect("sequential Graph is supported by the explicit precomputed DAG entry");
    assert_eq!(count.entries(), 1);
    let GraphAlphaCollectionOutcome::Complete((map, alpha)) = outcome else {
        panic!("unexpired phase");
    };
    assert!(map.contains_key("raw_bn") && alpha.alphas.contains_key("relu"));
    let evaluated = graph
        .collect_crown_bounds_with_alpha(&input, &map, &alpha, None, None)
        .unwrap();
    let output = &evaluated["output"];
    let lo = *output.lower().iter().next().unwrap();
    let hi = *output.upper().iter().next().unwrap();
    // Authored conv + identity raw BN computes |x|, over x in [-1, 1].
    assert!(lo <= 0.0 && hi >= 1.0 && lo.is_finite() && hi.is_finite());
}

#[ntest::timeout(30000)]
#[test]
fn root_fixed_reference_completed_artifact_moves_without_recollection_even_after_expiry() {
    for expired in [false, true] {
        let (graph, input) = dag_fixture();
        let count = RootFixedReferenceCollectionCounter::start();
        let artifact = actual_artifact(&graph, &input);
        assert_eq!(artifact.completed_iterations, 1);
        assert_eq!(artifact.optimizer_updates_completed, 1);
        let expected = map_bits(&artifact.reference_bounds);
        let allocation = artifact.reference_bounds["left"].lower().as_ptr();
        let outcome = publish_completed_collection(Ok(Some(artifact)), expired).unwrap();
        let result = match outcome {
            GraphAlphaCollectionOutcome::ReferenceOnly { .. } => {
                panic!("completed optimizer must not be downgraded")
            }
            GraphAlphaCollectionOutcome::Complete(result) => {
                assert!(!expired);
                result
            }
            GraphAlphaCollectionOutcome::PhaseCapCheckpoint {
                result,
                completed_iterations,
                optimizer_updates_completed,
            } => {
                assert!(expired);
                assert_eq!((completed_iterations, optimizer_updates_completed), (1, 1));
                result
            }
        };
        assert_eq!(map_bits(&result.0), expected);
        assert_eq!(
            result.0["left"].lower().as_ptr(),
            allocation,
            "published map is moved, not deep-cloned"
        );
        assert_eq!(count.entries(), 1);
        assert_fresh_sample_enclosure(&graph, &input, result);
        // Positive control: a repeated reference request really is observable,
        // even if the existing complete-map cache can satisfy it cheaply.
        graph
            .collect_crown_ibp_bounds_dag_with_options(
                &input,
                reference_options(&config(), None, Duration::from_secs(5)),
            )
            .unwrap();
        assert_eq!(count.entries(), 2);
    }
}

#[test]
fn root_fixed_reference_rejects_unsupported_requests_before_collection() {
    let (graph, input) = dag_fixture();
    let count = RootFixedReferenceCollectionCounter::start();
    for case in 0..6 {
        let mut cfg = config();
        let mut cap = Duration::from_secs(5);
        match case {
            0 => cfg.fix_interm_bounds = false,
            1 => cfg.gradient_method = GradientMethod::Spsa,
            2 => cfg.invprop.enabled = true,
            3 => cfg.deadline = None,
            4 => cfg.iterations = 0,
            5 => cap = Duration::ZERO,
            _ => unreachable!(),
        }
        assert!(matches!(
            graph.collect_root_fixed_reference_alpha_with_engine(&input, &cfg, None, cap),
            Err(NyError::InvalidSpec(_))
        ));
        assert_eq!(count.entries(), 0);
    }
}

#[test]
fn root_fixed_reference_expired_phase_cannot_start_or_publish_zero_folds() {
    let (graph, input) = dag_fixture();
    let count = RootFixedReferenceCollectionCounter::start();
    let mut cfg = config();
    cfg.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    assert!(matches!(
        graph.collect_root_fixed_reference_alpha_with_engine(
            &input,
            &cfg,
            None,
            Duration::from_secs(5),
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(count.entries(), 0);
    assert!(matches!(
        publish_completed_collection(Ok(None), true),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        publish_completed_collection(Ok(None), false),
        Err(NyError::InvalidSpec(_))
    ));
}

#[ntest::timeout(30000)]
#[test]
fn root_fixed_reference_zero_count_wrong_source_and_numeric_error_fail_closed() {
    for expired in [false, true] {
        let (graph, input) = dag_fixture();
        let mut artifact = actual_artifact(&graph, &input);
        artifact.completed_iterations = 0;
        let result = publish_completed_collection(Ok(Some(artifact)), expired);
        assert!(match result {
            Err(NyError::DeadlineExceeded(_)) => expired,
            Err(NyError::InvalidSpec(_)) => !expired,
            _ => false,
        });
        let result = publish_completed_collection(
            Err(NyError::NumericalInstability(
                "actual collection error channel".into(),
            )),
            expired,
        );
        assert!(
            matches!(result, Err(NyError::NumericalInstability(message)) if message == "actual collection error channel")
        );
    }
    let (graph, input) = dag_fixture();
    let mut artifact = actual_artifact(&graph, &input);
    artifact.reference_bounds_source = AlphaReferenceBoundsSource::Ibp;
    assert!(matches!(
        publish_completed_collection(Ok(Some(artifact)), true),
        Err(NyError::InvalidSpec(_))
    ));
}

#[ntest::timeout(30000)]
#[test]
fn root_fixed_reference_new_domain_is_collected_under_its_own_input() {
    let (graph, first) = dag_fixture();
    let second = BoundedTensor::new(
        arr1(&[-2.0_f32, -1.4]).into_dyn(),
        arr1(&[2.6_f32, 2.2]).into_dyn(),
    )
    .unwrap();
    let mut prior_left = None;
    for input in [&first, &second] {
        let result = graph
            .collect_root_fixed_reference_alpha_with_engine(
                input,
                &config(),
                None,
                Duration::from_secs(5),
            )
            .unwrap();
        let GraphAlphaCollectionOutcome::Complete((map, _)) = result else {
            panic!("unexpired phase");
        };
        assert_eq!(map[NETWORK_INPUT].lower(), input.lower());
        assert_eq!(map[NETWORK_INPUT].upper(), input.upper());
        let left = map_bits(&map)["left"].clone();
        if let Some(prior) = prior_left {
            assert_ne!(
                prior, left,
                "a different domain must not reuse the first box's reference"
            );
        }
        prior_left = Some(left);
    }
}

#[test]
fn root_fixed_reference_options_cover_foundation_and_preserve_exact_phase_cap() {
    let mut cfg = config();
    cfg.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let cap = Duration::from_millis(17);
    let options = reference_options(&cfg, None, cap);
    assert_eq!(options.deadline, cfg.deadline);
    assert_eq!(options.tightening_cap, Some(cap));
    assert!(options.deadline_includes_ibp && options.complete_cache_lookup_only);
    assert!(options.precomputed_ibp.is_none());
    assert!(
        options.original_root_head_rows.is_some(),
        "eligible original root route enables the transaction by default"
    );
    assert!(
        CrownIbpCollectOptions::default()
            .original_root_head_rows
            .is_none(),
        "unrelated collectors keep their historical policy"
    );
    let (graph, input) = dag_fixture();
    assert!(
        matches!(
            graph.collect_crown_ibp_bounds_dag_with_options(&input, options),
            Err(NyError::DeadlineExceeded(_))
        ),
        "cold foundation must honor expired authority too"
    );
}

#[ntest::timeout(30000)]
#[test]
fn root_fixed_reference_no_activation_returns_reference_only_without_recollection() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "output",
        Layer::Linear(LinearLayer::new(arr2(&[[2.0_f32]]), Some(arr1(&[0.5]))).unwrap()),
    ));
    graph.set_output("output");
    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let count = GraphNetwork::root_fixed_reference_collection_counter_for_test();
    let outcome = graph
        .collect_root_fixed_reference_alpha_with_engine(
            &input,
            &config(),
            None,
            Duration::from_secs(5),
        )
        .unwrap();
    assert_eq!(count(), 1);
    let GraphAlphaCollectionOutcome::ReferenceOnly {
        result: (map, alpha),
    } = outcome
    else {
        panic!("no activation has no optimizer credit");
    };
    assert!(alpha.alphas.is_empty());
    let evaluated = graph
        .collect_crown_bounds_with_alpha(&input, &map, &alpha, None, None)
        .unwrap();
    let output = &evaluated["output"];
    assert!(*output.lower().iter().next().unwrap() <= -1.5);
    assert!(*output.upper().iter().next().unwrap() >= 2.5);
}

#[ntest::timeout(30000)]
#[test]
fn root_fixed_reference_stable_and_adaptive_skip_are_reference_only() {
    for adaptive in [false, true] {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "pre",
            Layer::Linear(
                LinearLayer::new(
                    arr2(&[[1.0_f32]]),
                    Some(arr1(&[if adaptive { 0.0 } else { 2.0 }])),
                )
                .unwrap(),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["pre".into()],
        ));
        graph.set_output("relu");
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let mut cfg = config();
        cfg.adaptive_skip = adaptive;
        cfg.adaptive_skip_depth_threshold = 0;
        let count = GraphNetwork::root_fixed_reference_collection_counter_for_test();
        let outcome = graph
            .collect_root_fixed_reference_alpha_with_engine(
                &input,
                &cfg,
                None,
                Duration::from_secs(5),
            )
            .unwrap();
        assert_eq!(count(), 1);
        let GraphAlphaCollectionOutcome::ReferenceOnly {
            result: (map, alpha),
        } = outcome
        else {
            panic!("stable/explicit skip has no completed optimizer fold");
        };
        let evaluated = graph
            .collect_crown_bounds_with_alpha(&input, &map, &alpha, None, None)
            .unwrap();
        let output = &evaluated["relu"];
        let (lo, hi) = if adaptive { (0.0, 1.0) } else { (1.0, 3.0) };
        assert!(*output.lower().iter().next().unwrap() <= lo);
        assert!(*output.upper().iter().next().unwrap() >= hi);
    }
}
