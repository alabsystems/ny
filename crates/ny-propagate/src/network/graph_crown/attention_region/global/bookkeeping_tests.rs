// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Genuine long original DAGs and same-ledger control-storage frontiers.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::layers::{BoundPropagation, ReshapeLayer};
use crate::network::core::graph::dispatch_plan::CrownDispatchPlan;
use crate::types::BoundsProvenance;
use ndarray::{ArrayD, IxDyn};
use num_rational::BigRational;
use num_traits::Signed;

const SHIFT: [f32; 6] = [0.125, -0.25, 0.375, 0.0, -0.125, 0.25];
const CAP: usize = 32 << 20;

struct Chain {
    graph: GraphNetwork,
    input: BoundedTensor,
    boxes: HashMap<String, BoundedTensor>,
    order: Vec<String>,
}

fn chain(groups: usize, reduce: bool) -> Chain {
    let mut graph = GraphNetwork::new();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-0.75, -0.5, -0.25, -0.75, -0.5, -0.25])
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, 0.75, 1.0, 0.5, 0.75, 1.0]).unwrap(),
    )
    .unwrap();
    let mut prior = input.clone();
    let mut source = NETWORK_INPUT.to_string();
    let mut boxes = HashMap::new();
    let mut order = Vec::new();
    for group in 0..groups {
        for (name, layer) in [
            (
                format!("reshape-{group}"),
                Layer::Reshape(ReshapeLayer::new(vec![3, 2])),
            ),
            (
                format!("transpose-{group}"),
                Layer::Transpose(TransposeLayer::new(vec![1, 0])),
            ),
            (
                format!("shift-{group}"),
                Layer::AddConstant(AddConstantLayer::new(
                    ArrayD::from_shape_vec(IxDyn(&[2, 3]), SHIFT.to_vec()).unwrap(),
                )),
            ),
        ] {
            let output = layer.propagate_ibp(&prior).unwrap();
            graph.add_node(GraphNode::new(&name, layer, vec![source]));
            source = name.clone();
            boxes.insert(name.clone(), output.clone());
            order.push(name);
            prior = output;
        }
    }
    if reduce {
        let layer = Layer::ReduceMean(ReduceMeanLayer::new(vec![-2], false));
        let output = layer.propagate_ibp(&prior).unwrap();
        graph.add_node(GraphNode::new("mean", layer, vec![source]));
        boxes.insert("mean".into(), output);
        order.push("mean".into());
        source = "mean".into();
    }
    graph.set_output(&source);
    Chain {
        graph,
        input,
        boxes,
        order,
    }
}

fn authority(
    cap: usize,
    work: u64,
    deadline: Option<Instant>,
) -> (
    AttentionRegionAuthority,
    WorkBudget,
    MemoryAuthority,
    CancellationToken,
) {
    let memory = MemoryAuthority::new(cap);
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline,
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: work,
            max_proposals: MAX_REGIONS as u64,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (
        AttentionRegionAuthority {
            budget: Some(budget.clone()),
            cap,
            deadline,
        },
        budget,
        memory,
        cancellation,
    )
}

fn at() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

fn rat(x: f32) -> BigRational {
    BigRational::from_float(f32_to_f64_exact(x)).unwrap()
}

fn point(mask: usize) -> Vec<BigRational> {
    (0..6)
        .map(|i| {
            rat(if mask & (1 << i) == 0 {
                [-0.75, -0.5, -0.25][i % 3]
            } else {
                [0.5, 0.75, 1.0][i % 3]
            })
        })
        .collect()
}

/// Authored [2,3] -> reshape[3,2] -> transpose[2,3] -> add, using
/// independent exact rational indexing and constants, no production map.
fn original_forward(mut x: Vec<BigRational>, groups: usize) -> Vec<BigRational> {
    for _ in 0..groups {
        let old = x;
        x = (0..6)
            .map(|i| old[(i % 3) * 2 + i / 3].clone() + rat(SHIFT[i]))
            .collect();
    }
    x
}

fn incoming() -> LinearBounds {
    let mut row = LinearBounds::new(
        Array2::from_shape_fn((4, 6), |(r, c)| {
            if r >= 2 {
                0.0
            } else {
                ((r * 3 + c * 5) % 9) as f32 * 0.125 - 0.5
            }
        }),
        Array1::from_vec(vec![-0.125, 0.75, 0.0, -0.0]),
        Array2::from_shape_fn((4, 6), |(r, c)| {
            if r >= 2 {
                -0.0
            } else {
                ((r * 7 + c * 2) % 11) as f32 * 0.0625 - 0.375
            }
        }),
        // Independent constants must never be averaged or reordered.
        Array1::from_vec(vec![0.25, 0.125, 0.0, 0.0]),
    )
    .unwrap();
    row.lower_a_err = Some(Array2::from_shape_fn((4, 6), |(r, c)| {
        if r == 3 {
            0.0
        } else {
            ((r + c) % 5) as f32 * 0.0078125
        }
    }));
    row.upper_a_err = Some(Array2::from_shape_fn((4, 6), |(r, c)| {
        if r == 3 {
            0.0
        } else {
            ((r + 2 * c) % 7) as f32 * 0.00390625
        }
    }));
    row
}

fn exact(row: &LinearBounds, r: usize, x: &[BigRational], upper: bool) -> BigRational {
    let (a, b, e) = if upper {
        (
            &row.upper_a,
            &row.upper_b,
            row.upper_a_err.as_ref().unwrap(),
        )
    } else {
        (
            &row.lower_a,
            &row.lower_b,
            row.lower_a_err.as_ref().unwrap(),
        )
    };
    x.iter().enumerate().fold(rat(b[r]), |sum, (c, x)| {
        let eta = rat(e[[r, c]]) * x.abs();
        sum + rat(a[[r, c]]) * x + if upper { eta } else { -eta }
    })
}

fn take(accumulator: &mut BatchedCrownAccumulator, source: &str) -> BatchedLinearBounds {
    match accumulator.take(source).unwrap() {
        BatchedCrownBounds::Dense(bounds) => bounds,
        _ => panic!("original full-column row must remain dense"),
    }
}

#[test]
fn attention_region_global_bookkeeping_normal_batched_route_completes_long_original_chain() {
    let _env = crate::tests::lock_env_shared();
    let chain = chain(24, true);
    assert_eq!(chain.graph.num_nodes(), 73);
    let (authority, budget, memory, _) = authority(CAP, MAX_WORK, Some(at()));
    let _scope = AttentionRegionScope::install(authority);
    let (result, receipt) = chain
        .graph
        .propagate_crown_batched_with_attention_region_receipt_and_deadline(
            &chain.input,
            Some(at()),
        )
        .unwrap();
    assert_eq!(result.provenance, BoundsProvenance::Crown);
    assert_eq!(result.bounds.shape(), [3]);
    assert_eq!(receipt.composed, 0);
    for mask in 0..64 {
        let exact = original_forward(point(mask), 24);
        for c in 0..3 {
            let y = (&exact[c] + &exact[c + 3]) / BigRational::from_integer(2.into());
            assert!(rat(result.bounds.lower().as_slice().unwrap()[c]) <= y);
            assert!(rat(result.bounds.upper().as_slice().unwrap()[c]) >= y);
        }
    }
    assert!(budget.receipt().unwrap().completed >= chain.graph.num_nodes() as u64);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}

#[test]
fn attention_region_global_bookkeeping_long_full_planes_contain_independent_signed_rationals() {
    let _env = crate::tests::lock_env_shared();
    let chain = chain(24, false);
    let (authority, budget, memory, _) = authority(CAP, MAX_WORK, Some(at()));
    let _scope = AttentionRegionScope::install(authority);
    let mut query = AttentionRegionQuery::new_for_graph(&chain.graph, Some(at()));
    assert_eq!(query.global_step_limit, chain.graph.num_nodes());
    let capacity = query.held.capacity();
    let original = incoming();
    let incoming = global_source(original.clone(), &[2, 3], query.deadline, &budget).unwrap();
    let plan = CrownDispatchPlan::build(&chain.graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    accumulator.insert(
        chain.order.last().unwrap(),
        BatchedCrownBounds::Dense(incoming),
    );
    for name in chain.order.iter().rev() {
        let incoming = take(&mut accumulator, name);
        assert!(propagate(
            &mut query,
            &chain.graph.nodes[name],
            &incoming,
            &chain.input,
            &chain.boxes,
            &mut accumulator,
        )
        .unwrap());
    }
    let result = take(&mut accumulator, NETWORK_INPUT);
    assert_eq!(query.global_steps, 72);
    assert_eq!(query.held.capacity(), capacity);
    assert_eq!(query.held.len(), 73);
    assert_eq!(result.lower_a.shape(), [4, 6]);
    assert_eq!(result.input_shape(), [2, 3]);
    assert_eq!(result.output_shape(), [4]);
    let actual = consume::flat_incoming(&result, 6, query.deadline, &budget).unwrap();
    for mask in 0..64 {
        let x = point(mask);
        let y = original_forward(x.clone(), 24);
        for r in 0..4 {
            assert!(exact(&actual, r, &x, false) <= exact(&original, r, &y, false));
            assert!(exact(&actual, r, &x, true) >= exact(&original, r, &y, true));
        }
    }
    assert!(actual.lower_a.row(2).iter().all(|&v| v == 0.0));
    assert!(actual
        .lower_a_err
        .as_ref()
        .unwrap()
        .row(2)
        .iter()
        .any(|&v| v > 0.0));
    assert!(actual.lower_a.row(3).iter().all(|&v| v == 0.0));
    assert!(actual
        .lower_a_err
        .as_ref()
        .unwrap()
        .row(3)
        .iter()
        .all(|&v| v == 0.0));
    let before = budget.receipt().unwrap();
    assert!(!propagate(
        &mut query,
        &chain.graph.nodes[chain.order.last().unwrap()],
        &result,
        &chain.input,
        &chain.boxes,
        &mut accumulator,
    )
    .unwrap());
    assert_eq!(query.global_steps, 72);
    assert_eq!(query.held.capacity(), capacity);
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    drop(query);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}

#[test]
fn attention_region_global_bookkeeping_exact_header_cap_and_same_authority_legacy_fallback() {
    let _env = crate::tests::lock_env_shared();
    let chain = chain(192, false);
    let header = HeldPlan::new(chain.graph.num_nodes(), true).unwrap();
    assert!(header.bytes > HEADER_BYTES);
    for (cap, expected) in [
        (header.bytes, chain.graph.num_nodes()),
        (header.bytes - 1, 64),
    ] {
        let (authority, budget, memory, _) = authority(cap, MAX_WORK, Some(at()));
        let _scope = AttentionRegionScope::install(authority);
        let query = AttentionRegionQuery::new_for_graph(&chain.graph, Some(at()));
        assert!(query.budget.as_ref().unwrap().shares_ledger(&budget));
        assert_eq!(query.global_step_limit, expected);
        let plan = HeldPlan::new(expected, expected != 64).unwrap();
        assert_eq!(query.held.len(), 1);
        assert!(query.held.capacity() >= plan.slots);
        assert!(plan.actual_bytes(query.held.capacity()).unwrap() <= plan.bytes);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, plan.bytes);
        assert_eq!(budget.receipt().unwrap().work_reserved, plan.work);
        drop(query);
        let released = memory.receipt().unwrap();
        assert_eq!(
            released.live_materialized_bytes + released.live_pending_bytes,
            0
        );
    }
}

#[test]
fn attention_region_global_bookkeeping_nested_queries_share_header_peak_and_never_refill() {
    let _env = crate::tests::lock_env_shared();
    let chain = chain(192, false);
    let header = HeldPlan::new(chain.graph.num_nodes(), true).unwrap();
    let cap = header.bytes + HEADER_BYTES;
    let (authority, budget, memory, _) = authority(cap, MAX_WORK, Some(at()));
    let _scope = AttentionRegionScope::install(authority);
    let parent = AttentionRegionQuery::new_for_graph(&chain.graph, Some(at()));
    assert_eq!(parent.global_step_limit, chain.graph.num_nodes());
    {
        let nested = AttentionRegionQuery::new_for_graph(&chain.graph, Some(at()));
        assert_eq!(nested.global_step_limit, 64);
        assert!(nested.budget.as_ref().unwrap().shares_ledger(&budget));
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, cap);
        assert_eq!(budget.receipt().unwrap().work_reserved, header.work);
    }
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        header.bytes
    );
    budget
        .lease(WorkRequest {
            work: MAX_WORK - header.work,
            ..WorkRequest::default()
        })
        .unwrap();
    {
        let nested = AttentionRegionQuery::new_for_graph(&chain.graph, Some(at()));
        assert_eq!(nested.global_step_limit, 64);
        assert!(nested.budget.as_ref().unwrap().shares_ledger(&budget));
        assert_eq!(budget.receipt().unwrap().work_reserved, MAX_WORK);
        assert!(nested
            .budget
            .as_ref()
            .unwrap()
            .lease(WorkRequest {
                work: 1,
                ..WorkRequest::default()
            })
            .is_err());
    }
    drop(parent);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}

#[test]
fn attention_region_global_bookkeeping_original_work_refusal_retains_paid_legacy_header() {
    let _env = crate::tests::lock_env_shared();
    let chain = chain(192, false);
    let header = HeldPlan::new(chain.graph.num_nodes(), true).unwrap();
    // Memory admits the graph-derived header. Only the original work ledger
    // refuses its bookkeeping work, so fallback cannot hide a new allowance.
    let (authority, budget, memory, _) = authority(CAP, header.work - 1, Some(at()));
    let _scope = AttentionRegionScope::install(authority);
    let query = AttentionRegionQuery::new_for_graph(&chain.graph, Some(at()));
    assert_eq!(query.global_step_limit, 64);
    assert!(query.budget.as_ref().unwrap().shares_ledger(&budget));
    let receipt = budget.receipt().unwrap();
    assert_eq!(receipt.live_persistent_bytes, HEADER_BYTES);
    assert_eq!(receipt.work_reserved, 0);
    budget
        .lease(WorkRequest {
            work: header.work - 1,
            ..WorkRequest::default()
        })
        .unwrap();
    assert!(query
        .budget
        .as_ref()
        .unwrap()
        .lease(WorkRequest {
            work: 1,
            ..WorkRequest::default()
        })
        .is_err());
    assert_eq!(budget.receipt().unwrap().work_reserved, header.work - 1);
    drop(query);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}

#[test]
fn attention_region_global_bookkeeping_earliest_deadline_and_shared_cancellation_are_preserved() {
    let _env = crate::tests::lock_env_shared();
    let chain = chain(24, false);
    let earliest = at();
    let (authority, budget, memory, cancellation) = authority(CAP, MAX_WORK, Some(earliest));
    let _scope = AttentionRegionScope::install(authority);
    let query = AttentionRegionQuery::new_for_graph(
        &chain.graph,
        Some(earliest + Duration::from_secs(300)),
    );
    assert_eq!(query.deadline, Some(earliest));
    assert_eq!(
        query._scope.as_ref().unwrap().authority().deadline,
        Some(earliest)
    );
    let before = budget.receipt().unwrap();
    {
        let expired = AttentionRegionQuery::new_for_graph(
            &chain.graph,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        );
        assert!(expired.budget.is_none());
        assert_eq!(expired.held.capacity(), 0);
        assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    }
    cancellation.cancel();
    {
        let cancelled = AttentionRegionQuery::new_for_graph(
            &chain.graph,
            Some(earliest + Duration::from_secs(300)),
        );
        assert!(cancelled.budget.is_none());
        assert_eq!(cancelled.held.capacity(), 0);
        assert!(cancelled
            ._scope
            .as_ref()
            .unwrap()
            .authority()
            .budget
            .as_ref()
            .unwrap()
            .shares_ledger(&budget));
        assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            before.work_reserved
        );
    }
    drop(query);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}

#[test]
fn attention_region_global_bookkeeping_occupied_header_cap_refuses_before_any_allocation() {
    let _env = crate::tests::lock_env_shared();
    let chain = chain(192, false);
    let header = HeldPlan::new(chain.graph.num_nodes(), true).unwrap();
    let (authority, budget, memory, _) = authority(header.bytes, MAX_WORK, Some(at()));
    let _scope = AttentionRegionScope::install(authority);
    let mut held = budget
        .lease(WorkRequest {
            persistent_bytes: header.bytes - HEADER_BYTES + 1,
            ..WorkRequest::default()
        })
        .unwrap();
    held.mark_materialized().unwrap();
    let before = budget.receipt().unwrap();
    let memory_before = memory.receipt().unwrap();
    let query = AttentionRegionQuery::new_for_graph(&chain.graph, Some(at()));
    assert!(query.budget.is_none());
    assert_eq!(query.held.capacity(), 0);
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert_eq!(memory.receipt().unwrap(), memory_before);
    drop(query);
    drop(held);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}

#[test]
fn attention_region_global_bookkeeping_checked_geometry_and_legacy_limit_remain_bounded() {
    let _env = crate::tests::lock_env_shared();
    assert!(HeldPlan::new(usize::MAX, true).is_none());
    assert!(HeldPlan::new(usize::MAX - MAX_REGIONS, true).is_none());
    assert!(HeldPlan::new(64, false)
        .unwrap()
        .actual_bytes(usize::MAX)
        .is_none());
    let chain = chain(24, false);
    let mut query = AttentionRegionQuery::with_limits(None, CAP, MAX_WORK);
    assert_eq!(query.global_step_limit, MAX_GLOBAL_STEPS);
    let budget = query.budget.clone().unwrap();
    let original = incoming();
    let mut incoming = global_source(original, &[2, 3], None, &budget).unwrap();
    let plan = CrownDispatchPlan::build(&chain.graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    let capacity = query.held.capacity();
    for (step, name) in chain.order.iter().rev().enumerate() {
        let before = budget.receipt().unwrap();
        let accepted = propagate(
            &mut query,
            &chain.graph.nodes[name],
            &incoming,
            &chain.input,
            &chain.boxes,
            &mut accumulator,
        )
        .unwrap();
        if step == 64 {
            assert!(!accepted);
            assert_eq!(query.global_steps, 64);
            assert_eq!(query.held.capacity(), capacity);
            assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
            return;
        }
        assert!(accepted);
        incoming = take(
            &mut accumulator,
            chain.graph.nodes[name].require_unary_input().unwrap(),
        );
    }
    panic!("real long graph must reach the preserved legacy64 refusal");
}
