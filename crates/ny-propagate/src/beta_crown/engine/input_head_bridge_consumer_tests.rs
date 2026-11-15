// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{LinearLayer, MaxPool2dLayer, ReLULayer, ReshapeLayer};
use crate::margin_row::{
    verify_input_head_retained, HeadGroupPolicy, InputHeadOutcome, InputHeadSpec,
};
use crate::{BabVerificationStatus, BetaCrownConfig, BranchingHeuristic, Layer};
use ndarray::{arr1, arr2};
use ny_test_utils::CountingGemmEngine;
use std::time::Duration;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(3)
}
fn network(tail_sign: f32) -> Network {
    let mut net = Network::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0], [0.0]]), Some(arr1(&[0.5, 0.0]))).unwrap(),
    ));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[tail_sign, -tail_sign]]), None).unwrap(),
    ));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, 1])));
    net.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (1, 1),
        (1, 1),
        (0, 0),
    )));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1])));
    net
}
fn configured() -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        max_domains: 4,
        max_depth: 1,
        timeout: Duration::from_secs(3),
        ..Default::default()
    }
}
fn verifier(config: BetaCrownConfig) -> (BetaCrownVerifier, BoundedTensor, Instant) {
    let spec = InputHeadSpec::new(
        1,
        1,
        vec![1.0],
        vec![0.0],
        vec![-1.0, 0.0],
        vec![0.5, 0.0],
        [91; 32],
    )
    .unwrap();
    let end = deadline();
    let (outcome, source) = verify_input_head_retained(
        &spec,
        &[-1.0],
        &[1.0],
        0,
        1,
        HeadGroupPolicy::new(1.0, 1, 1).unwrap(),
        end,
        &[-1.0],
        &[1.0],
        end,
    );
    assert!(
        matches!(outcome, InputHeadOutcome::Unknown(_)),
        "M(1)=-1/2 is genuinely unsafe"
    );
    let root = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    (
        BetaCrownVerifier::new(config)
            .with_input_head_continuation(source.expect("native completed open Source")),
        root,
        end,
    )
}

#[test]
fn source_owned_child_prunes_before_generic_work_but_unsafe_sibling_prevents_instance() {
    let (v, root, end) = verifier(configured());
    let net = network(1.0);
    let parent = BabDomain::root_with_input(vec![], -1.0, 1.0, &root).unwrap();
    let engine = CountingGemmEngine::new();
    assert!(v
        .create_input_split_child(
            &net,
            &root,
            &parent,
            0,
            -1.0,
            -0.25,
            0.0,
            Some(end),
            Some(&engine)
        )
        .unwrap()
        .is_none());
    assert_eq!(
        engine.gemm_calls(),
        0,
        "the completed original child skips the generic collector"
    );
    assert!(v.input_head_continuation().unwrap().dependent_use());
    assert!(v
        .input_head_continuation()
        .unwrap()
        .try_finish(end)
        .is_none());
    let unsafe_child = v
        .create_input_split_child(&net, &root, &parent, 0, 0.0, 1.0, 0.0, Some(end), None)
        .unwrap()
        .expect("an unsafe sibling cannot disappear");
    assert!(unsafe_child.lower_bound <= -0.5);
    assert!(v
        .input_head_continuation()
        .unwrap()
        .try_finish(end)
        .is_none());
    let result = v
        .verify_with_engine(&net, &root, 0.0, None, Some(end))
        .unwrap();
    assert!(!matches!(result.result, BabVerificationStatus::Verified));
}

#[test]
fn consumer_declines_wrong_tail_threshold_upper_direction_and_relu_split() {
    let (v, root, end) = verifier(configured());
    let safe_child =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[-0.25]).into_dyn()).unwrap();
    assert!(
        v.input_head_support(&network(1.0), &safe_child, 0.0, Some(end))
            .unwrap()
            > 0.0
    );
    assert!(v
        .input_head_support(&network(-1.0), &safe_child, 0.0, Some(end))
        .is_none());
    assert!(v
        .input_head_support(&network(1.0), &safe_child, 0.125, Some(end))
        .is_none());
    let mut c = configured();
    c.verify_upper_bound = true;
    let upper = v.with_config_from(c);
    assert!(upper
        .input_head_support(&network(1.0), &safe_child, 0.0, Some(end))
        .is_none());
    let mut c = configured();
    c.branching_heuristic = BranchingHeuristic::LargestBoundWidth;
    let relu = upper.with_config_from(c);
    assert!(relu
        .input_head_support(&network(1.0), &safe_child, 0.0, Some(end))
        .is_none());
    assert!(
        !v.input_head_continuation().unwrap().dependent_use(),
        "a support proposal is not numerical MAX consumption"
    );
    assert!(v.input_head_continuation().unwrap().root_box_matches(
        root.lower().as_slice().unwrap(),
        root.upper().as_slice().unwrap()
    ));
}

#[test]
fn expired_phase_or_absent_cap_cannot_supply_a_delayed_floor() {
    let (v, _root, end) = verifier(configured());
    let safe_child =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[-0.25]).into_dyn()).unwrap();
    let old = v
        .input_head_support(&network(1.0), &safe_child, 0.0, Some(end))
        .unwrap();
    assert!(old > 0.0);
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(!v.input_head_live(Some(expired)));
    assert!(v
        .input_head_support(&network(1.0), &safe_child, 0.0, Some(expired))
        .is_none());
    assert!(v
        .input_head_support(&network(1.0), &safe_child, 0.0, None)
        .is_none());
    assert!(!v.input_head_continuation().unwrap().dependent_use());
}

#[test]
fn actual_two_child_consumer_completes_original_instance_that_scalar_root_left_open() {
    // M(x)=ReLU(x)+1/2 ReLU(1-x)-1/4 has true minimum1/4.
    // The native root line 1/2*x+1/4 is negative at x=-1; on the left
    // child the retained complete Q residual gives at least1/4 instead.
    let spec = InputHeadSpec::new(
        1,
        2,
        vec![1.0, -1.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0, 0.5, 0.0],
        vec![-0.25, 0.0],
        [92; 32],
    )
    .unwrap();
    let end = deadline();
    let (outcome, source) = verify_input_head_retained(
        &spec,
        &[-1.0],
        &[2.0],
        0,
        1,
        HeadGroupPolicy::new(1.0, 1, 1).unwrap(),
        end,
        &[-1.0],
        &[2.0],
        end,
    );
    let InputHeadOutcome::Unknown(report) = outcome else {
        panic!("unoptimized complete root portfolios should stay open")
    };
    assert!(report.lower.unwrap() <= -0.25);
    assert_eq!(report.advice_calls, 0);
    let source = source.unwrap();
    let mut net = Network::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [-1.0]]), Some(arr1(&[0.0, 1.0]))).unwrap(),
    ));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.5], [0.0, 0.0]]), Some(arr1(&[-0.25, 0.0]))).unwrap(),
    ));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, -1.0]]), None).unwrap(),
    ));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, 1])));
    net.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (1, 1),
        (1, 1),
        (0, 0),
    )));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1])));
    assert!(source.matches_scalar_network(&net));
    let root = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    let parent = BabDomain::root_with_input(vec![], -1.0, 3.0, &root).unwrap();
    let v = BetaCrownVerifier::new(configured()).with_input_head_continuation(Arc::clone(&source));
    let engine = CountingGemmEngine::new();
    assert!(v
        .create_input_split_child(
            &net,
            &root,
            &parent,
            0,
            -1.0,
            0.0,
            0.0,
            Some(end),
            Some(&engine)
        )
        .unwrap()
        .is_none());
    assert!(
        source.try_finish(end).is_none(),
        "a first sibling never proves the whole instance"
    );
    assert!(v
        .create_input_split_child(
            &net,
            &root,
            &parent,
            0,
            0.0,
            2.0,
            0.0,
            Some(end),
            Some(&engine)
        )
        .unwrap()
        .is_none());
    assert_eq!(engine.gemm_calls(), 0);
    assert!(source.dependent_use());
    let token = source
        .try_finish(end)
        .expect("both actual children cover the original evaluation root");
    assert_eq!(token.scope(), crate::certify::TokenScope::Instance);
    assert_eq!(token.classes(), &[1]);
    assert!(token.min_margin() > 0.125 && token.min_margin() <= 0.25);
}
