// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{arr1, arr2};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::*;
use crate::beta_crown::config::{BetaCrownConfig, InputClipType};
use crate::bounds::LinearBounds;
use crate::layers::ReLULayer;
use crate::{GraphNetwork, GraphNode, Layer, LinearLayer};

fn complete_reorder_test_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "hidden",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("hidden linear")),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["hidden".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("out linear")),
        vec!["relu".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn parent_linear_bounds() -> LinearBounds {
    LinearBounds {
        lower_a: arr2(&[[1.0_f32]]),
        lower_b: arr1(&[0.0_f32]),
        upper_a: arr2(&[[1.0_f32]]),
        upper_b: arr1(&[0.0_f32]),
        lower_a_err: None,
        upper_a_err: None,
    }
}

fn parent_domain(child_input: &BoundedTensor) -> MultiObjInputDomain {
    MultiObjInputDomain {
        input_bounds: Arc::new(child_input.clone()),
        obj_bounds: vec![(-1.0, 1.0)],
        linear_bounds: Some(parent_linear_bounds()),
        depth: 0,
        priority: 1.0,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

fn run_reorder_child_screen(
    verifier: &BetaCrownVerifier,
    compute_bounds: &impl Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<MultiObjBounds>,
) -> MultiObjInputDomain {
    let graph = complete_reorder_test_graph();
    let child_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite child bounds");
    let parent_domain = parent_domain(&child_input);
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
    let mut domains_verified_by_clip = 0usize;

    screen_multi_obj_child(
        verifier,
        &graph,
        child_input,
        &arr2(&[[1.0_f32]]),
        &[0.2],
        None,
        compute_bounds,
        None,
        &parent_domain,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("reorder child screen should not error");

    queue.pop().expect("child should remain unresolved")
}

#[test]
fn test_screen_multi_obj_child_reorder_complete_defers_with_node_bounds_override_4116() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Complete,
        reorder_bab: true,
        input_split_ibp_enhancement: false,
        ..Default::default()
    });

    let child = run_reorder_child_screen(&verifier, &|_, _| {
        panic!("reorder multi-objective screening must not eagerly re-run child CROWN")
    });

    assert!(
        child.needs_bounding,
        "reorder_bab must still defer child bounding in complete mode"
    );
    assert!(
        child.node_bounds_override.is_some(),
        "complete clipping should carry node-bounds overrides into the deferred child"
    );
    assert!(
        child.linear_bounds.is_none(),
        "deferred children should clear linear bounds until the batch rebound pass"
    );
}

#[test]
fn test_screen_multi_obj_child_reorder_uses_parent_linear_without_eager_crown_4116() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        reorder_bab: true,
        input_split_ibp_enhancement: false,
        ..Default::default()
    });
    let compute_calls = Cell::new(0usize);

    let child = run_reorder_child_screen(&verifier, &|_, _| {
        compute_calls.set(compute_calls.get() + 1);
        Ok((vec![(-1.0, 1.0)], Some(parent_linear_bounds())))
    });

    assert_eq!(
        compute_calls.get(),
        0,
        "reorder path should reuse parent linear bounds instead of eagerly bounding the child"
    );
    assert!(
        child.needs_bounding,
        "reorder child must stay deferred after parent-linear clipping"
    );
    assert!(
        child.node_bounds_override.is_none(),
        "relaxed clipping should not synthesize node-bounds overrides"
    );
}

#[test]
fn test_screen_multi_obj_child_eager_applies_parent_floor_4354() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: false,
        input_clip_type: InputClipType::Relaxed,
        reorder_bab: false,
        input_split_ibp_enhancement: false,
        ..Default::default()
    });
    let graph = complete_reorder_test_graph();
    let child_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite child bounds");
    let parent_domain = MultiObjInputDomain {
        input_bounds: Arc::new(child_input.clone()),
        obj_bounds: vec![(0.3_f32, 1.0_f32)],
        linear_bounds: None,
        depth: 0,
        priority: 1.0,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    };
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
    let mut domains_verified_by_clip = 0usize;

    screen_multi_obj_child(
        &verifier,
        &graph,
        child_input,
        &arr2(&[[1.0_f32]]),
        &[0.4_f32],
        None,
        &|_, _| Ok((vec![(0.1_f32, 0.8_f32)], None)),
        None,
        &parent_domain,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("eager multi-objective child screen should not error");

    let child = queue.pop().expect("child should remain unresolved");
    assert_eq!(
        child.obj_bounds,
        vec![(0.3_f32, 0.8_f32)],
        "eager multi-objective screening should clamp the child lower bound to the parent floor"
    );
    assert_eq!(
        child.priority,
        multi_obj_domain_priority(&child.obj_bounds, &[0.4_f32]),
        "eager multi-objective screening should recompute priority from the clamped bounds"
    );
    assert!(!child.needs_bounding);
    assert_eq!(domains_verified_by_clip, 0);
    assert_eq!(lifecycle.domains_verified, 0);
}

fn eager_no_clip_verifier() -> BetaCrownVerifier {
    BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: false,
        input_clip_type: InputClipType::Relaxed,
        reorder_bab: false,
        input_split_ibp_enhancement: false,
        ..Default::default()
    })
}

/// cgan step-2 port: when the warm closure is supplied AND the parent carries
/// refined α slopes, the eager multi-objective screen must take the warm path
/// (never the frozen `compute_bounds`) and store the refined α on the child so
/// its own children warm-start from it.
#[test]
fn test_screen_multi_obj_child_eager_warm_alpha_path_refines_and_stores_alpha() {
    let verifier = eager_no_clip_verifier();
    let graph = complete_reorder_test_graph();
    let child_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite child bounds");
    let parent_alpha = Arc::new(GraphAlphaState::default());
    let parent_domain = MultiObjInputDomain {
        input_bounds: Arc::new(child_input.clone()),
        obj_bounds: vec![(0.3_f32, 1.0_f32)],
        linear_bounds: None,
        depth: 0,
        priority: 1.0,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: Some(parent_alpha.clone()),
    };
    let warm_calls = Cell::new(0usize);
    let warm = |_: &BoundedTensor,
                _: Option<&HashMap<String, BoundedTensor>>,
                _: &GraphAlphaState|
     -> Result<WarmMultiObjBoundsResult> {
        warm_calls.set(warm_calls.get() + 1);
        Ok((vec![(0.1_f32, 0.8_f32)], None, GraphAlphaState::default()))
    };
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
    let mut domains_verified_by_clip = 0usize;

    screen_multi_obj_child(
        &verifier,
        &graph,
        child_input,
        &arr2(&[[1.0_f32]]),
        &[0.4_f32],
        None,
        &|_, _| -> Result<MultiObjBounds> {
            panic!("frozen compute_bounds must NOT run when the warm-alpha path is enabled")
        },
        Some(&warm),
        &parent_domain,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("warm-alpha multi-objective child screen should not error");

    assert_eq!(warm_calls.get(), 1, "warm closure must run exactly once");
    let child = queue.pop().expect("child should remain unresolved");
    assert_eq!(
        child.obj_bounds,
        vec![(0.3_f32, 0.8_f32)],
        "warm path must still clamp the child lower bound to the parent floor"
    );
    let child_alpha = child
        .inherited_alpha_state
        .as_ref()
        .expect("warm path must store the refined α on the child");
    assert!(
        !Arc::ptr_eq(child_alpha, &parent_alpha),
        "child must carry the REFINED α, not the parent's frozen Arc"
    );
}

/// Frozen default (input_split_alpha_iteration = 0 => no warm closure): the
/// eager screen must use `compute_bounds` and pass the parent's α state (if
/// any) through unchanged — byte-identical historical behavior.
#[test]
fn test_screen_multi_obj_child_eager_frozen_passes_parent_alpha_through() {
    let verifier = eager_no_clip_verifier();
    let graph = complete_reorder_test_graph();
    let child_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite child bounds");
    let parent_alpha = Arc::new(GraphAlphaState::default());
    let parent_domain = MultiObjInputDomain {
        input_bounds: Arc::new(child_input.clone()),
        obj_bounds: vec![(-1.0_f32, 1.0_f32)],
        linear_bounds: None,
        depth: 0,
        priority: 1.0,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: Some(parent_alpha.clone()),
    };
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
    let mut domains_verified_by_clip = 0usize;

    screen_multi_obj_child(
        &verifier,
        &graph,
        child_input,
        &arr2(&[[1.0_f32]]),
        &[0.4_f32],
        None,
        &|_, _| Ok((vec![(0.1_f32, 0.8_f32)], None)),
        None,
        &parent_domain,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("frozen multi-objective child screen should not error");

    let child = queue.pop().expect("child should remain unresolved");
    let child_alpha = child
        .inherited_alpha_state
        .as_ref()
        .expect("frozen path must still carry the parent α forward");
    assert!(
        Arc::ptr_eq(child_alpha, &parent_alpha),
        "frozen path must pass the parent's α Arc through UNCHANGED"
    );
}

/// Reorder mode defers bounding, so per-domain refinement must not run there;
/// the parent's α slopes are carried forward unchanged for the deferred pass.
#[test]
fn test_screen_multi_obj_child_reorder_carries_parent_alpha_unchanged() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: false,
        input_clip_type: InputClipType::Relaxed,
        reorder_bab: true,
        input_split_ibp_enhancement: false,
        ..Default::default()
    });
    let graph = complete_reorder_test_graph();
    let child_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite child bounds");
    let parent_alpha = Arc::new(GraphAlphaState::default());
    let mut parent = parent_domain(&child_input);
    parent.inherited_alpha_state = Some(parent_alpha.clone());
    let warm = |_: &BoundedTensor,
                _: Option<&HashMap<String, BoundedTensor>>,
                _: &GraphAlphaState|
     -> Result<WarmMultiObjBoundsResult> {
        panic!("reorder mode must NOT invoke per-domain α refinement at screening time")
    };
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
    let mut domains_verified_by_clip = 0usize;

    screen_multi_obj_child(
        &verifier,
        &graph,
        child_input,
        &arr2(&[[1.0_f32]]),
        &[0.2_f32],
        None,
        &|_, _| -> Result<MultiObjBounds> {
            panic!("reorder mode must not eagerly bound the child")
        },
        Some(&warm),
        &parent,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("reorder child screen should not error");

    let child = queue.pop().expect("child should be deferred");
    assert!(child.needs_bounding);
    let child_alpha = child
        .inherited_alpha_state
        .as_ref()
        .expect("reorder path must carry the parent α forward for the deferred pass");
    assert!(
        Arc::ptr_eq(child_alpha, &parent_alpha),
        "reorder path must carry the parent's α Arc UNCHANGED"
    );
}
