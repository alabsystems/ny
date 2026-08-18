// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-child-batch (S1) parity: the consolidated child pipeline must be
//! BIT-IDENTICAL to the historical `FlatPendingChild` clone chain across the
//! whole split -> prescreen -> clip -> push stage: identical lifecycle
//! counters, identical verified-by-clip count, and an identical queue —
//! priority/depth/obj-bounds/box compared as RAW f32 BITS in pop order (heap
//! tie-breaking is part of the parity criterion). Fixture follows the
//! LSNC_PARITY_CHECKLIST Part 3 A shape: multi-clause spec, 6-dim inputs,
//! mixed-sign / exact-zero / near-zero coefficients, carried coefficient
//! error (finite AND non-finite), an infeasible-by-clip child, a
//! grouped-verified child, an IBP-prescreen-verified child, a NaN priority,
//! +/-inf objective bounds, and every skip lane of the per-domain loop head.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array2};
use ny_core::{NaiveCpuGemmEngine, Result};
use ny_tensor::BoundedTensor;

use super::super::super::fresh_domain_clip::FreshDomainClipTelemetry;
use super::super::super::shared::{MultiObjBounds, MultiObjInputDomain};
use super::super::process_batch::process_disjunctive_domain_batch;
use super::super::screen_child::WarmAlphaTelemetry;
use super::{force_child_batch, force_clip_planes};
use crate::beta_crown::config::{BetaCrownConfig, InputClipType};
use crate::beta_crown::engine::graph::propagation::batched::SPEC_GATE_TEST_LOCK;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::{Layer, LinearLayer, ReLULayer};
use crate::{GraphNetwork, GraphNode};

/// 6 -> 8 (ReLU) -> 4 graph: identity-ish first 6 hidden rows plus two
/// negated rows so hidden neurons flip stability class per sub-box.
fn build_lsnc_shaped_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(
        arr2(&[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        Some(arr1(&[0.1, -0.1, 0.05, 0.0, 0.0, 0.0, 0.2, -0.2])),
    )
    .expect("valid linear1");
    let linear2 = LinearLayer::new(
        arr2(&[
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        Some(arr1(&[0.0, 0.1, -0.1, 0.0])),
    )
    .expect("valid linear2");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

fn spec_matrix_6x4() -> Array2<f32> {
    arr2(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-0.5, 0.0, 0.25, 1.0],
    ])
}

fn box6(lo: f32, hi: f32) -> Arc<BoundedTensor> {
    Arc::new(
        BoundedTensor::new(arr1(&[lo; 6]).into_dyn(), arr1(&[hi; 6]).into_dyn())
            .expect("finite box"),
    )
}

fn box_dims(lo: [f32; 6], hi: [f32; 6]) -> Arc<BoundedTensor> {
    Arc::new(BoundedTensor::new(arr1(&lo).into_dyn(), arr1(&hi).into_dyn()).expect("finite box"))
}

fn unresolved_obj_bounds() -> Vec<(f32, f32)> {
    vec![
        (-0.3, 1.0),
        (f32::NEG_INFINITY, f32::INFINITY),
        (-0.7, 0.4),
        (-0.2, f32::INFINITY),
        (-1.5, 2.0),
        (-0.9, 0.1),
    ]
}

/// Mixed-sign / exact-zero / near-zero coefficient planes (6 spec rows x 6
/// input dims).
fn lb_mixed() -> LinearBounds {
    LinearBounds::new(
        arr2(&[
            [0.8, -0.3, 0.0, 1e-12, 0.4, -0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.7, 0.2, -0.1, 1e-11, 0.3],
            [0.9, 0.0, -0.4, 0.6, -0.2, 0.1],
            [-1e-12, 0.5, 0.5, -0.5, 0.5, -0.5],
            [0.3, -0.3, 0.3, -0.3, 0.3, -0.3],
        ]),
        arr1(&[-0.1, 0.2, 0.0, -0.3, 0.4, f32::NEG_INFINITY]),
        arr2(&[
            [0.8, -0.3, 0.0, 1e-12, 0.4, -0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.7, 0.2, -0.1, 1e-11, 0.3],
            [0.9, 0.0, -0.4, 0.6, -0.2, 0.1],
            [-1e-12, 0.5, 0.5, -0.5, 0.5, -0.5],
            [0.3, -0.3, 0.3, -0.3, 0.3, -0.3],
        ]),
        arr1(&[0.1, 0.3, 0.2, 0.5, 0.6, f32::INFINITY]),
    )
    .expect("valid mixed lb")
}

/// Same planes plus carried coefficient error: finite entries straddling the
/// coefficients on most rows, one NON-FINITE entry (carry-forever row that
/// degrades to a +/-inf bias on fold and can never verify).
fn lb_with_coeff_err() -> LinearBounds {
    let mut lb = lb_mixed();
    let mut lower_err = Array2::<f32>::zeros((6, 6));
    let mut upper_err = Array2::<f32>::zeros((6, 6));
    for r in 0..6 {
        for c in 0..6 {
            // Deterministic small magnitudes straddling the coefficients.
            lower_err[[r, c]] = 1e-3 * ((r * 6 + c) % 5) as f32;
            upper_err[[r, c]] = 2e-3 * ((r + c) % 3) as f32;
        }
    }
    lower_err[[2, 4]] = f32::INFINITY;
    lb.set_coeff_err(lower_err, upper_err);
    lb
}

/// Contradictory rows on dim 0 under thresholds 0.5 (keep-regions
/// x0 <= 0.2 AND x0 >= 0.8): opposite-direction clips invert the box, so
/// children go infeasible under the sequential-threshold clip (the
/// `lb_infeasible_4366` pattern shifted for threshold 0.5).
fn lb_infeasible() -> LinearBounds {
    let mut lower_a = Array2::<f32>::zeros((6, 6));
    let mut lower_b = ndarray::Array1::<f32>::zeros(6);
    lower_a[[0, 0]] = 10.0;
    lower_b[0] = -1.5;
    lower_a[[1, 0]] = -10.0;
    lower_b[1] = 8.5;
    for r in 2..6 {
        lower_a[[r, r]] = 0.05;
        lower_b[r] = -10.0;
    }
    LinearBounds::new(lower_a.clone(), lower_b.clone(), lower_a, lower_b)
        .expect("valid infeasible lb")
}

/// Strong single-direction rows (`10*x0 + 5` per row, the `lb_verified_4366`
/// pattern): the clip cannot invert the box (one direction only), but the
/// post-clip concretized lower `>= 10*x0_l + 5 > 0.5` verifies every row —
/// the grouped disjunctive lane.
fn lb_grouped_verified() -> LinearBounds {
    let mut a = Array2::<f32>::zeros((6, 6));
    for r in 0..6 {
        a[[r, 0]] = 10.0;
    }
    let b = ndarray::Array1::from_elem(6, 5.0_f32);
    LinearBounds::new(a.clone(), b.clone(), a, b).expect("valid verified lb")
}

fn make_parent_domains(seed_alpha: &Arc<GraphAlphaState>) -> Vec<MultiObjInputDomain> {
    vec![
        // 0: plain mixed-coefficient parent — children clip + queue.
        MultiObjInputDomain {
            input_bounds: box6(-1.0, 1.0),
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: Some(lb_mixed()),
            depth: 1,
            priority: 1.0,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
        // 1: already grouped-verified at the loop head (no children).
        MultiObjInputDomain {
            input_bounds: box6(-0.5, 0.5),
            obj_bounds: vec![(1.0e9, f32::INFINITY); 6],
            linear_bounds: Some(lb_mixed()),
            depth: 2,
            priority: 0.5,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
        // 2: at max_depth — unresolved_due_to_depth, no children.
        MultiObjInputDomain {
            input_bounds: box6(-0.25, 0.25),
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: Some(lb_mixed()),
            depth: 10,
            priority: 0.25,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
        // 3: zero-width box — unresolved_due_to_unsplittable.
        MultiObjInputDomain {
            input_bounds: box6(0.3, 0.3),
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: Some(lb_mixed()),
            depth: 3,
            priority: 0.75,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
        // 4: NO LinearBounds + NaN priority — children pushed unclipped, in
        // the without-lb group, NaN priority bits must survive verbatim.
        MultiObjInputDomain {
            input_bounds: box6(-0.8, 0.6),
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: None,
            depth: 1,
            priority: f32::NAN,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
        // 5: carried coefficient error (finite + one non-finite entry) —
        // exercises the per-child clone-then-fold discharge (I-A10). Equal
        // priority to parent 0 exercises heap tie-breaking.
        MultiObjInputDomain {
            input_bounds: box6(-0.9, 0.9),
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: Some(lb_with_coeff_err()),
            depth: 2,
            priority: 1.0,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
        // 6: contradictory clip rows — children verified by box infeasibility.
        MultiObjInputDomain {
            input_bounds: box6(0.0, 1.0),
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: Some(lb_infeasible()),
            depth: 1,
            priority: 2.0,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
        // 7: positive far-from-origin box — children verified by the IBP
        // prescreen (grouped criterion holds on interval bounds alone).
        MultiObjInputDomain {
            input_bounds: box6(2.0, 3.0),
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: Some(lb_mixed()),
            depth: 1,
            priority: 3.0,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
        // 8: survives the IBP prescreen (box straddles zero on dims 1-5 so
        // clause 2 is not interval-verified) but every post-clip concretized
        // row exceeds the threshold — the grouped disjunctive lane.
        MultiObjInputDomain {
            input_bounds: box_dims(
                [0.5, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: Some(lb_grouped_verified()),
            depth: 2,
            priority: 4.0,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(Arc::clone(seed_alpha)),
        },
    ]
}

/// Bit-exact queue snapshot in pop order.
type QueueEntry = (
    u32,             // priority bits
    usize,           // depth
    bool,            // needs_bounding
    bool,            // linear_bounds.is_some()
    bool,            // node_bounds_override.is_some()
    Vec<(u32, u32)>, // obj_bounds bits
    Vec<u32>,        // input lower bits (flattened)
    Vec<u32>,        // input upper bits (flattened)
    bool,            // inherited_alpha_state is the exact parent Arc
);

fn drain_queue(
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    seed_alpha: &Arc<GraphAlphaState>,
) -> Vec<QueueEntry> {
    let mut entries = Vec::with_capacity(queue.len());
    while let Some(domain) = queue.pop() {
        let flat = domain.input_bounds.flatten();
        entries.push((
            domain.priority.to_bits(),
            domain.depth,
            domain.needs_bounding,
            domain.linear_bounds.is_some(),
            domain.node_bounds_override.is_some(),
            domain
                .obj_bounds
                .iter()
                .map(|(l, u)| (l.to_bits(), u.to_bits()))
                .collect(),
            flat.lower().iter().map(|v| v.to_bits()).collect(),
            flat.upper().iter().map(|v| v.to_bits()).collect(),
            domain
                .inherited_alpha_state
                .as_ref()
                .is_some_and(|alpha| Arc::ptr_eq(alpha, seed_alpha)),
        ));
    }
    entries
}

struct LegOutcome {
    explored: usize,
    verified: usize,
    max_depth: usize,
    unresolved_depth: bool,
    unresolved_unsplittable: bool,
    clipped: usize,
    queue: Vec<QueueEntry>,
}

/// `clip_planes`: `None` leaves the S5 gate at its env default (irrelevant on
/// the reference leg); `Some(_)` forces it for the child-batch fast path.
/// Callers hold `SPEC_GATE_TEST_LOCK` (the force atomics are process-global).
fn run_leg(force: bool, clip_planes: Option<bool>, config: BetaCrownConfig) -> LegOutcome {
    force_child_batch(Some(force));
    force_clip_planes(clip_planes);
    let verifier = BetaCrownVerifier::new(config);
    let graph = build_lsnc_shaped_graph();
    let spec_matrix = spec_matrix_6x4();
    let thresholds = [0.5_f32; 6];
    let clause_sizes = [3usize, 3usize];
    let engine = NaiveCpuGemmEngine;
    let seed_alpha = Arc::new(GraphAlphaState::new());

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;
    let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);
    let fresh_domain_clip_telemetry = FreshDomainClipTelemetry::new(false);

    let result = process_disjunctive_domain_batch(
        &verifier,
        &graph,
        make_parent_domains(&seed_alpha),
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        Some(&engine),
        &|_input, _node_bounds| -> Result<MultiObjBounds> {
            panic!("reorder prescreen lane must not call compute_bounds")
        },
        None,
        &warm_alpha_telemetry,
        &fresh_domain_clip_telemetry,
        // No MulBinary alphas in this fixture (pre-existing merge fixup:
        // ddb123c1 merged S1's call site without the alphas parameter).
        None,
        Duration::from_mins(1),
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    );
    force_child_batch(None);
    force_clip_planes(None);
    let result = result.expect("batch processing should not error");
    assert!(result.is_none(), "no early termination expected");
    let queue_entries = drain_queue(&mut queue, &seed_alpha);
    assert!(
        queue_entries.iter().all(|entry| entry.8),
        "every reference/S1/S5/fallback survivor must preserve the exact parent alpha Arc"
    );

    LegOutcome {
        explored: lifecycle.domains_explored,
        verified: lifecycle.domains_verified,
        max_depth: lifecycle.max_depth_reached,
        unresolved_depth: lifecycle.unresolved_due_to_depth,
        unresolved_unsplittable: lifecycle.unresolved_due_to_unsplittable,
        clipped: domains_verified_by_clip,
        queue: queue_entries,
    }
}

fn lsnc_reorder_config() -> BetaCrownConfig {
    BetaCrownConfig {
        reorder_bab: true,
        input_split_ibp_enhancement: true,
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 3,
        max_depth: 10,
        ..Default::default()
    }
}

/// The S1 parity test: reference (`FlatPendingChild` chain) vs consolidated
/// `ChildBatch` fast path, everything compared exactly.
#[test]
fn test_child_batch_reorder_prescreen_parity_lsnc_s1() {
    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let reference = run_leg(false, None, lsnc_reorder_config());
    let fast = run_leg(true, Some(false), lsnc_reorder_config());

    assert_eq!(reference.explored, fast.explored, "domains_explored");
    assert_eq!(reference.verified, fast.verified, "domains_verified");
    assert_eq!(reference.max_depth, fast.max_depth, "max_depth_reached");
    assert_eq!(
        reference.unresolved_depth, fast.unresolved_depth,
        "unresolved_due_to_depth"
    );
    assert_eq!(
        reference.unresolved_unsplittable, fast.unresolved_unsplittable,
        "unresolved_due_to_unsplittable"
    );
    assert_eq!(reference.clipped, fast.clipped, "domains_verified_by_clip");
    assert_eq!(
        reference.queue.len(),
        fast.queue.len(),
        "queue length must match"
    );
    for (i, (r, f)) in reference.queue.iter().zip(fast.queue.iter()).enumerate() {
        assert_eq!(r, f, "queue entry {i} must be bit-identical in pop order");
    }

    // Fixture-coverage guards (a fixture that stops exercising a lane would
    // make the parity vacuous):
    assert_eq!(reference.explored, 9, "all 9 parents processed");
    assert!(reference.unresolved_depth, "depth-capped parent exercised");
    assert!(
        reference.unresolved_unsplittable,
        "unsplittable parent exercised"
    );
    assert!(
        reference.clipped >= 2,
        "clip/grouped verification lanes exercised (got {})",
        reference.clipped
    );
    assert!(
        reference.verified > reference.clipped,
        "loop-head/prescreen verification exercised beyond clip verifies"
    );
    assert!(
        !reference.queue.is_empty(),
        "surviving children must reach the queue"
    );
    assert!(
        reference.queue.iter().any(|e| f32::from_bits(e.0).is_nan()),
        "NaN-priority child must survive verbatim"
    );
}

/// Ineligible configuration (relaxed clip disabled): the gate must decline and
/// take the unchanged reference path — forced ON and forced OFF are trivially
/// identical AND well-behaved (fallback actually runs: `push_fallback_survivors`
/// queues children without clip verification).
#[test]
fn test_child_batch_gate_declines_non_relaxed_lane_s1() {
    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let config = || BetaCrownConfig {
        reorder_bab: true,
        input_split_ibp_enhancement: true,
        enable_relaxed_clip: false,
        max_depth: 10,
        ..Default::default()
    };
    let reference = run_leg(false, None, config());
    let forced = run_leg(true, None, config());

    assert_eq!(reference.explored, forced.explored);
    assert_eq!(reference.verified, forced.verified);
    assert_eq!(reference.clipped, forced.clipped);
    assert_eq!(
        reference.clipped, 0,
        "no clip verifies without relaxed clip"
    );
    assert_eq!(reference.queue.len(), forced.queue.len());
    for (r, f) in reference.queue.iter().zip(forced.queue.iter()) {
        assert_eq!(r, f);
    }
    assert!(
        !reference.queue.is_empty(),
        "fallback push lane must queue surviving children"
    );
}

/// #lsnc-clip-planes (S5) pipeline parity: reference `FlatPendingChild` chain
/// vs S1 stacked path vs S5 planes path — all three consumer-surface
/// bit-identical (lifecycle counters, verified-by-clip count, and the queue
/// compared as raw f32 bits in pop order) on the S1 adversarial fixture
/// (mixed-sign / exact-zero / near-zero coefficients, carried coefficient
/// error finite AND non-finite, infeasible-by-clip children, grouped-verified
/// children, prescreen-verified children, NaN priority).
#[test]
fn test_clip_planes_reorder_prescreen_parity_lsnc_s5() {
    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let reference = run_leg(false, None, lsnc_reorder_config());
    let stacked = run_leg(true, Some(false), lsnc_reorder_config());
    let handled_before = super::CLIP_PLANES_HANDLED.load(std::sync::atomic::Ordering::Relaxed);
    let planes = run_leg(true, Some(true), lsnc_reorder_config());
    assert!(
        super::CLIP_PLANES_HANDLED.load(std::sync::atomic::Ordering::Relaxed) > handled_before,
        "planes leg must actually run the S5 path (silent decline would make this parity vacuous)"
    );

    for (name, leg) in [("stacked", &stacked), ("planes", &planes)] {
        assert_eq!(reference.explored, leg.explored, "{name}: domains_explored");
        assert_eq!(reference.verified, leg.verified, "{name}: domains_verified");
        assert_eq!(
            reference.max_depth, leg.max_depth,
            "{name}: max_depth_reached"
        );
        assert_eq!(
            reference.unresolved_depth, leg.unresolved_depth,
            "{name}: unresolved_due_to_depth"
        );
        assert_eq!(
            reference.unresolved_unsplittable, leg.unresolved_unsplittable,
            "{name}: unresolved_due_to_unsplittable"
        );
        assert_eq!(
            reference.clipped, leg.clipped,
            "{name}: domains_verified_by_clip"
        );
        assert_eq!(
            reference.queue.len(),
            leg.queue.len(),
            "{name}: queue length"
        );
        for (i, (r, l)) in reference.queue.iter().zip(leg.queue.iter()).enumerate() {
            assert_eq!(
                r, l,
                "{name}: queue entry {i} must be bit-identical in pop order"
            );
        }
    }

    // Fixture-coverage guards (mirror the S1 test: a fixture that stops
    // exercising a lane would make the parity vacuous).
    assert!(reference.clipped >= 2, "clip/grouped lanes exercised");
    assert!(!reference.queue.is_empty(), "survivors reach the queue");
    assert!(
        reference.queue.iter().any(|e| f32::from_bits(e.0).is_nan()),
        "NaN-priority child must survive verbatim"
    );
}

/// #lsnc-clip-planes (S5) core parity: `batched_relaxed_clip_from_planes` +
/// `concretize_postclip_lower_bounds_planes` vs the S1 stacked entries on the
/// FULL output surface — every clipped bound of every child (including
/// latched-infeasible children whose bounds the consumer discards) compared as
/// raw f32 bits, plus the verified latch and every concretized post-clip pair.
/// Fixture: shared parent planes with two children per parent, mixed-sign /
/// exact-zero / near-zero coefficients, ±inf biases, an infeasible
/// (latch-mid-sequence) parent, a short plane (out-of-range threshold rows),
/// a zero-width child box, both clip directions, iteration counts {1, 20}.
#[test]
fn test_batched_clip_planes_matches_stacked_s5() {
    use super::super::super::batched_clip::{
        batched_relaxed_clip_from_planes, batched_relaxed_clip_from_stacked,
        concretize_postclip_lower_bounds_planes, ParentClipPlane,
    };
    use super::super::push_survivors::concretize_postclip_lower_bounds;
    use ndarray::{ArrayD, IxDyn};
    use std::borrow::Cow;

    let x_dim = 6usize;
    let thresholds = [0.5_f32, -0.1, 0.0, 0.25, 0.4];
    let n_thr = thresholds.len();

    // Parent 0: mixed coefficients, ±inf biases (5 rows).
    let p0 = LinearBounds::new(
        arr2(&[
            [0.8, -0.3, 0.0, 1e-12, 0.4, -0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.7, 0.2, -0.1, 1e-11, 0.3],
            [0.9, 0.0, -0.4, 0.6, -0.2, 0.1],
            [-1e-12, 0.5, 0.5, -0.5, 0.5, -0.5],
        ]),
        arr1(&[-0.1, 0.2, f32::NEG_INFINITY, -0.3, 0.4]),
        arr2(&[
            [0.7, -0.2, 0.1, 0.0, 0.3, -0.5],
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [-0.4, 0.6, 0.3, -0.2, 0.0, 0.2],
            [0.8, 0.1, -0.3, 0.5, -0.1, 0.2],
            [0.0, 0.4, 0.6, -0.4, 0.4, -0.6],
        ]),
        arr1(&[0.1, 0.3, 0.2, f32::INFINITY, 0.6]),
    )
    .expect("p0");
    // Parent 1: SHORT plane (3 rows) — threshold rows 3, 4 are out-of-range.
    let p1 = LinearBounds::new(
        arr2(&[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
        ]),
        arr1(&[-0.4, 0.1, 0.0]),
        arr2(&[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
        ]),
        arr1(&[-0.2, 0.2, 0.1]),
    )
    .expect("p1");
    // Parent 2: contradictory rows on dim 0 — children latch mid-sequence and
    // the later rows must see the midpoint collapse.
    let p2 = {
        let mut lower_a = Array2::<f32>::zeros((5, 6));
        let mut lower_b = ndarray::Array1::<f32>::zeros(5);
        lower_a[[0, 0]] = 10.0;
        lower_b[0] = -1.5;
        lower_a[[1, 0]] = -10.0;
        lower_b[1] = 8.5;
        for r in 2..5 {
            lower_a[[r, r]] = 0.05;
            lower_b[r] = -10.0;
        }
        LinearBounds::new(lower_a.clone(), lower_b.clone(), lower_a, lower_b).expect("p2")
    };
    let parents = [&p0, &p1, &p2];

    // Children: two per parent 0/1, one for parent 2; varied boxes including a
    // zero-width child (the verified-collapse shape).
    let child_plane: Vec<usize> = vec![0, 0, 1, 1, 2];
    let m = child_plane.len();
    #[rustfmt::skip]
    let orig_l: Vec<f32> = vec![
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -0.2,  0.1, -0.6, -0.4, -0.9,  0.0,
        -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
         0.3,  0.3,  0.3,  0.3,  0.3,  0.3, // zero-width
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    ];
    #[rustfmt::skip]
    let orig_u: Vec<f32> = vec![
         1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
         0.9,  1.2,  0.4,  0.5,  0.1,  0.8,
         2.0,  2.0,  2.0,  2.0,  2.0,  2.0,
         0.3,  0.3,  0.3,  0.3,  0.3,  0.3, // zero-width
         1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
    ];

    for &verify_upper in &[false, true] {
        for &iters in &[1usize, 20] {
            // Reference leg: the S1 stacked entry over per-child parent refs.
            let stacked_l = ArrayD::from_shape_vec(IxDyn(&[m, x_dim]), orig_l.clone()).unwrap();
            let stacked_u = ArrayD::from_shape_vec(IxDyn(&[m, x_dim]), orig_u.clone()).unwrap();
            let lb_refs: Vec<&LinearBounds> = child_plane.iter().map(|&s| parents[s]).collect();
            // Whole-spec single clause: this leg exercises the sequential
            // cross-row latch + midpoint collapse bit-for-bit (unchanged by the
            // clause-aware driver, which delegates the single-clause case to the
            // whole-spec core). Multi-clause stacked-vs-planes parity is covered
            // by `test_batched_clip_grouped_multiclause_stacked_planes_parity`.
            let whole_spec = [n_thr];
            let (ref_l, ref_u, ref_v) = batched_relaxed_clip_from_stacked(
                &stacked_l,
                &stacked_u,
                &lb_refs,
                &thresholds,
                &whole_spec,
                verify_upper,
                iters,
            )
            .expect("stacked clip");

            // Planes leg: shared per-parent planes + per-child biases in the
            // clip sign convention (no coefficient error in this fixture, so
            // biases are the base rows; the err-fold lane is covered by the
            // pipeline parity test's err-carrying parents).
            let planes: Vec<ParentClipPlane<'_>> = parents
                .iter()
                .map(|lb| {
                    let used = if verify_upper {
                        lb.upper_a()
                    } else {
                        lb.lower_a()
                    };
                    let flat = used.as_slice().expect("standard layout");
                    ParentClipPlane {
                        coeffs: if verify_upper {
                            Cow::Owned(flat.iter().map(|v| -v).collect())
                        } else {
                            Cow::Borrowed(flat)
                        },
                        nrows: used.nrows(),
                    }
                })
                .collect();
            let mut bias_used = vec![0f32; m * n_thr];
            for k in 0..m {
                let lb = parents[child_plane[k]];
                let base = if verify_upper {
                    lb.upper_b()
                } else {
                    lb.lower_b()
                };
                for i in 0..base.len().min(n_thr) {
                    bias_used[k * n_thr + i] = if verify_upper { -base[i] } else { base[i] };
                }
            }
            let (pl_l, pl_u, pl_v) = batched_relaxed_clip_from_planes(
                &orig_l,
                &orig_u,
                &planes,
                &child_plane,
                &bias_used,
                &thresholds,
                &whole_spec,
                verify_upper,
                iters,
                m,
                x_dim,
            )
            .expect("planes clip");

            assert_eq!(
                ref_v, pl_v,
                "verified latch mismatch (verify_upper={verify_upper}, iters={iters})"
            );
            assert!(ref_v.iter().any(|&v| v), "fixture must latch someone");
            assert!(!ref_v.iter().all(|&v| v), "fixture must keep survivors");
            let ref_l = ref_l.as_slice().unwrap();
            let ref_u = ref_u.as_slice().unwrap();
            for i in 0..m * x_dim {
                assert_eq!(
                    ref_l[i].to_bits(),
                    pl_l[i].to_bits(),
                    "x_l bit mismatch at {i} (verify_upper={verify_upper}, iters={iters}): ref={} planes={}",
                    ref_l[i],
                    pl_l[i]
                );
                assert_eq!(
                    ref_u[i].to_bits(),
                    pl_u[i].to_bits(),
                    "x_u bit mismatch at {i} (verify_upper={verify_upper}, iters={iters}): ref={} planes={}",
                    ref_u[i],
                    pl_u[i]
                );
            }

            // Post-clip concretize: planes variant vs the per-child reference
            // on every non-latched child, every row pair bit-identical.
            let mut planes_out: Vec<(f32, f32)> = Vec::new();
            for k in 0..m {
                if ref_v[k] {
                    continue;
                }
                let row_l = ArrayD::from_shape_vec(
                    IxDyn(&[x_dim]),
                    ref_l[k * x_dim..(k + 1) * x_dim].to_vec(),
                )
                .unwrap();
                let row_u = ArrayD::from_shape_vec(
                    IxDyn(&[x_dim]),
                    ref_u[k * x_dim..(k + 1) * x_dim].to_vec(),
                )
                .unwrap();
                let reference = concretize_postclip_lower_bounds(
                    &row_l,
                    &row_u,
                    lb_refs[k],
                    &thresholds,
                    verify_upper,
                );
                concretize_postclip_lower_bounds_planes(
                    &pl_l[k * x_dim..(k + 1) * x_dim],
                    &pl_u[k * x_dim..(k + 1) * x_dim],
                    &planes[child_plane[k]],
                    &bias_used[k * n_thr..(k + 1) * n_thr],
                    n_thr,
                    &mut planes_out,
                );
                assert_eq!(reference.len(), planes_out.len(), "child {k} row count");
                for (i, (r, p)) in reference.iter().zip(planes_out.iter()).enumerate() {
                    assert_eq!(
                        r.0.to_bits(),
                        p.0.to_bits(),
                        "child {k} row {i} lower mismatch (verify_upper={verify_upper}, iters={iters})"
                    );
                    assert_eq!(r.1.to_bits(), p.1.to_bits(), "child {k} row {i} upper");
                }
            }
        }
    }
}

#[test]
fn postclip_concretizers_round_each_addition_down_under_cancellation() {
    use super::super::super::batched_clip::{
        concretize_postclip_lower_bounds_planes, ParentClipPlane,
    };
    use super::super::push_survivors::concretize_postclip_lower_bounds;
    use ndarray::{ArrayD, IxDyn};
    use std::borrow::Cow;

    let large = 2.0_f32.powi(50);
    let row = arr2(&[[large, 1.0, -large]]);
    let bounds =
        LinearBounds::new(row.clone(), arr1(&[0.0]), row, arr1(&[0.0])).expect("linear row");
    let point = vec![large, -1.0, large];
    let clipped_lower = ArrayD::from_shape_vec(IxDyn(&[3]), point.clone()).unwrap();
    let clipped_upper = clipped_lower.clone();

    let reference =
        concretize_postclip_lower_bounds(&clipped_lower, &clipped_upper, &bounds, &[0.0], false);
    let coeffs = bounds.lower_a().as_slice().expect("standard layout");
    let plane = ParentClipPlane {
        coeffs: Cow::Borrowed(coeffs),
        nrows: 1,
    };
    let mut projected = Vec::new();
    concretize_postclip_lower_bounds_planes(&point, &point, &plane, &[0.0], 1, &mut projected);

    // Exact binary value: 2^100 - 1 - 2^100 = -1. A nearest-f64 fold
    // returns zero, whose final next-down f32 is still inward and could prune.
    assert!(reference[0].0 <= -1.0, "stacked lower={}", reference[0].0);
    assert!(projected[0].0 <= -1.0, "plane lower={}", projected[0].0);
}

/// #disj-cross-clause-clip-unsat: the clause-aware (grouped) stacked and planes
/// clip drivers must agree bit-for-bit on MULTI-CLAUSE specs too — the grouped
/// combination (per-clause intersection from the original box + union bounding
/// box + all-clauses-empty verified) is implemented once over the stacked core
/// and once over the planes row loop, so their outputs (clipped boxes, verified
/// latch) must be identical for the same partition.
#[test]
fn test_batched_clip_grouped_multiclause_stacked_planes_parity() {
    use super::super::super::batched_clip::{
        batched_relaxed_clip_from_planes, batched_relaxed_clip_from_stacked, ParentClipPlane,
    };
    use ndarray::{ArrayD, IxDyn};
    use std::borrow::Cow;

    let x_dim = 6usize;
    let thresholds = [0.5_f32, -0.1, 0.0, 0.25, 0.4];
    let n_thr = thresholds.len();

    // Two parents: one mixed-coefficient, one with a contradictory pair on dim 0
    // (rows 0 and 1) so a SINGLE clause containing both rows can go infeasible
    // while a partition that splits them keeps every clause feasible.
    let p0 = lb_mixed();
    let p1 = {
        let mut lower_a = Array2::<f32>::zeros((6, 6));
        let mut lower_b = ndarray::Array1::<f32>::zeros(6);
        lower_a[[0, 0]] = 10.0;
        lower_b[0] = -1.5;
        lower_a[[1, 0]] = -10.0;
        lower_b[1] = 8.5;
        for r in 2..6 {
            lower_a[[r, r]] = 0.05;
            lower_b[r] = -10.0;
        }
        LinearBounds::new(lower_a.clone(), lower_b.clone(), lower_a, lower_b).expect("p1")
    };
    let parents = [&p0, &p1];
    let child_plane: Vec<usize> = vec![0, 0, 1, 1];
    let m = child_plane.len();
    #[rustfmt::skip]
    let orig_l: Vec<f32> = vec![
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -0.5,  0.0, -0.6, -0.4, -0.9,  0.0,
         0.0, -0.5, -0.5, -0.5, -0.5, -0.5,
         0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
    ];
    #[rustfmt::skip]
    let orig_u: Vec<f32> = vec![
         1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
         0.9,  1.0,  0.4,  0.5,  0.1,  0.8,
         1.0,  0.5,  0.5,  0.5,  0.5,  0.5,
         0.9,  0.9,  0.9,  0.9,  0.9,  0.9,
    ];

    let mut latched_any = false;
    let mut survivor_any = false;
    for clause_sizes in [vec![2usize, 3], vec![1, 1, 1, 1, 1], vec![3, 2], vec![5]] {
        for &verify_upper in &[false, true] {
            for &iters in &[1usize, 8] {
                let stacked_l = ArrayD::from_shape_vec(IxDyn(&[m, x_dim]), orig_l.clone()).unwrap();
                let stacked_u = ArrayD::from_shape_vec(IxDyn(&[m, x_dim]), orig_u.clone()).unwrap();
                let lb_refs: Vec<&LinearBounds> = child_plane.iter().map(|&s| parents[s]).collect();
                let (ref_l, ref_u, ref_v) = batched_relaxed_clip_from_stacked(
                    &stacked_l,
                    &stacked_u,
                    &lb_refs,
                    &thresholds,
                    &clause_sizes,
                    verify_upper,
                    iters,
                )
                .expect("stacked clip");

                let planes: Vec<ParentClipPlane<'_>> = parents
                    .iter()
                    .map(|lb| {
                        let used = if verify_upper {
                            lb.upper_a()
                        } else {
                            lb.lower_a()
                        };
                        let flat = used.as_slice().expect("standard layout");
                        ParentClipPlane {
                            coeffs: if verify_upper {
                                Cow::Owned(flat.iter().map(|v| -v).collect())
                            } else {
                                Cow::Borrowed(flat)
                            },
                            nrows: used.nrows(),
                        }
                    })
                    .collect();
                let mut bias_used = vec![0f32; m * n_thr];
                for k in 0..m {
                    let lb = parents[child_plane[k]];
                    let base = if verify_upper {
                        lb.upper_b()
                    } else {
                        lb.lower_b()
                    };
                    for i in 0..base.len().min(n_thr) {
                        bias_used[k * n_thr + i] = if verify_upper { -base[i] } else { base[i] };
                    }
                }
                let (pl_l, pl_u, pl_v) = batched_relaxed_clip_from_planes(
                    &orig_l,
                    &orig_u,
                    &planes,
                    &child_plane,
                    &bias_used,
                    &thresholds,
                    &clause_sizes,
                    verify_upper,
                    iters,
                    m,
                    x_dim,
                )
                .expect("planes clip");

                assert_eq!(
                    ref_v, pl_v,
                    "verified latch mismatch (clauses={clause_sizes:?}, verify_upper={verify_upper}, iters={iters})"
                );
                latched_any |= ref_v.iter().any(|&v| v);
                survivor_any |= ref_v.iter().any(|&v| !v);
                let ref_l = ref_l.as_slice().unwrap();
                let ref_u = ref_u.as_slice().unwrap();
                for i in 0..m * x_dim {
                    assert_eq!(
                        ref_l[i].to_bits(),
                        pl_l[i].to_bits(),
                        "x_l bit mismatch at {i} (clauses={clause_sizes:?}, verify_upper={verify_upper}, iters={iters})"
                    );
                    assert_eq!(
                        ref_u[i].to_bits(),
                        pl_u[i].to_bits(),
                        "x_u bit mismatch at {i} (clauses={clause_sizes:?}, verify_upper={verify_upper}, iters={iters})"
                    );
                }
            }
        }
    }
    // Teeth: the fixture must exercise both a verified (all-clauses-empty) child
    // and a survivor across the partitions.
    assert!(
        latched_any,
        "fixture must verify some child under some partition"
    );
    assert!(survivor_any, "fixture must keep some survivor");
}

/// #lsnc-clip-planes (S5) decline leg: a parent whose plane width does not
/// match the child boxes must DECLINE to the reference body, which surfaces
/// its historical error — both legs must return the identical error (the
/// planes path erroring differently, or succeeding, would prove the decline
/// did not fall back).
#[test]
fn test_clip_planes_decline_ncols_mismatch_s5() {
    use std::collections::BinaryHeap;
    use std::time::Instant;

    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());

    // 5-column planes against 6-dim child rows.
    let lb_narrow = LinearBounds::new(
        Array2::<f32>::from_elem((6, 5), 0.5),
        arr1(&[0.0; 6]),
        Array2::<f32>::from_elem((6, 5), 0.5),
        arr1(&[0.0; 6]),
    )
    .expect("narrow lb");

    let run = |planes: Option<bool>| {
        force_clip_planes(planes);
        let verifier = BetaCrownVerifier::new(lsnc_reorder_config());
        let parents = vec![super::ChildBatchParent {
            obj_bounds: unresolved_obj_bounds(),
            linear_bounds: Some(lb_narrow.clone()),
            child_depth: 2,
            priority: 1.0,
            inherited_alpha_state: None,
        }];
        let x_dim = 6usize;
        let lower_data = vec![-1.0f32; 2 * x_dim];
        let upper_data = vec![1.0f32; 2 * x_dim];
        let parent_idx = vec![0usize, 0];
        let with_lb = vec![0usize, 1];
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clipped = 0usize;
        let result = super::push_child_batch_clip_survivors(
            &verifier,
            &with_lb,
            &lower_data,
            &upper_data,
            x_dim,
            &parent_idx,
            &parents,
            &[x_dim],
            &[0.5_f32; 6],
            &[3usize, 3usize],
            &mut queue,
            &mut lifecycle,
            &mut clipped,
        );
        force_clip_planes(None);
        (result, queue.len(), lifecycle.domains_verified, clipped)
    };

    let (ref_result, ref_queue, ref_verified, ref_clipped) = run(Some(false));
    let (pl_result, pl_queue, pl_verified, pl_clipped) = run(Some(true));

    let ref_err = ref_result.expect_err("reference must reject the width mismatch");
    let pl_err = pl_result.expect_err("planes leg must decline INTO the same rejection");
    assert_eq!(
        ref_err.to_string(),
        pl_err.to_string(),
        "decline must reproduce the reference error"
    );
    assert!(
        ref_err.to_string().contains("x_dim"),
        "expected the reference width-mismatch error, got: {ref_err}"
    );
    assert_eq!(
        (ref_queue, ref_verified, ref_clipped),
        (pl_queue, pl_verified, pl_clipped)
    );
}
