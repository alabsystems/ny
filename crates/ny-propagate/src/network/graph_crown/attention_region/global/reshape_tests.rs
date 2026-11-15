// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::layers::{BoundPropagation, ReshapeLayer};
use crate::network::core::graph::dispatch_plan::CrownDispatchPlan;
use ndarray::{ArrayD, IxDyn};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

fn authority() -> WorkBudget {
    WorkBudget::new(None, 32 << 20, 32 << 20, 100_000_000, 64, Duration::ZERO).unwrap()
}

fn original(rows: usize, columns: usize) -> LinearBounds {
    let mut result = LinearBounds::new(
        Array2::from_shape_fn((rows, columns), |(r, c)| {
            if r == rows - 1 {
                0.0
            } else {
                ((r * 3 + c * 5) % 11) as f32 * 0.125 - 0.625
            }
        }),
        Array1::from_shape_fn(rows, |r| r as f32 * 0.0625 - 0.125),
        Array2::from_shape_fn((rows, columns), |(r, c)| {
            if r == rows - 1 {
                -0.0
            } else {
                ((r * 7 + c * 3) % 13) as f32 * 0.0625 - 0.375
            }
        }),
        Array1::from_shape_fn(rows, |r| 0.25 - r as f32 * 0.03125),
    )
    .unwrap();
    result.lower_a_err = Some(Array2::from_shape_fn((rows, columns), |(r, c)| {
        ((r + 2 * c) % 5) as f32 * 0.015625
    }));
    result.upper_a_err = Some(Array2::from_shape_fn((rows, columns), |(r, c)| {
        ((3 * r + c) % 7) as f32 * 0.0078125
    }));
    result
}

fn graph(
    pre: &[usize],
    target: &[usize],
    axes: &[usize],
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "reshape",
        Layer::Reshape(ReshapeLayer::new(
            target.iter().map(|&n| n as i64).collect(),
        )),
    ));
    graph.add_node(GraphNode::new(
        "transpose",
        Layer::Transpose(TransposeLayer::new(axes.to_vec())),
        vec!["reshape".into()],
    ));
    graph.set_output("transpose");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(pre), -0.5),
        ArrayD::from_elem(IxDyn(pre), 0.75),
    )
    .unwrap();
    let reshape = graph.nodes["reshape"]
        .layer()
        .propagate_ibp(&input)
        .unwrap();
    let transpose = graph.nodes["transpose"]
        .layer()
        .propagate_ibp(&reshape)
        .unwrap();
    let boxes = HashMap::from([("reshape".into(), reshape), ("transpose".into(), transpose)]);
    (graph, input, boxes)
}

fn take(accumulator: &mut BatchedCrownAccumulator, source: &str) -> BatchedLinearBounds {
    match accumulator.take(source).unwrap() {
        BatchedCrownBounds::Dense(bounds) => bounds,
        _ => panic!("complete global contribution must remain dense"),
    }
}

fn bits(left: impl Iterator<Item = f32>, right: impl Iterator<Item = f32>) {
    let left: Vec<_> = left.map(f32::to_bits).collect();
    let right: Vec<_> = right.map(f32::to_bits).collect();
    assert_eq!(left, right);
}

#[test]
fn attention_region_global_transpose_reshape_original_vit_geometry_preserves_every_plane() {
    // Original Q/V authored rank4 permutation with the model batch removed:
    // projection [17,48] -> [17,3,16] -> [3,17,16]. Also cover authored Kt.
    for axes in [vec![0, 2, 1, 3], vec![0, 2, 3, 1]] {
        let (graph, input, boxes) = graph(&[17, 48], &[17, 3, 16], &axes);
        let budget = authority();
        let original = original(9, 816);
        let incoming =
            global_source(original.clone(), boxes["transpose"].shape(), None, &budget).unwrap();
        let legacy = graph.nodes["transpose"]
            .layer()
            .propagate_crown_backward_batched(&incoming, Some(&boxes["reshape"]), None)
            .unwrap();
        assert_eq!(legacy.input_shape(), boxes["transpose"].shape());
        assert!(
            matches!(
                graph.nodes["reshape"]
                    .layer()
                    .propagate_crown_backward_batched(&legacy, Some(&input), None),
                Err(NyError::InvalidSpec(_))
            ),
            "stale original transpose metadata reproduces the authored reshape refusal"
        );
        let plan = CrownDispatchPlan::build(&graph).unwrap();
        let mut accumulator = BatchedCrownAccumulator::new(&plan);
        let mut query = AttentionRegionQuery::with_limits(None, 32 << 20, 100_000_000);
        assert!(propagate(
            &mut query,
            &graph.nodes["transpose"],
            &incoming,
            &input,
            &boxes,
            &mut accumulator
        )
        .unwrap());
        let reshaped = take(&mut accumulator, "reshape");
        assert_eq!(reshaped.input_shape(), [17, 3, 16]);
        assert_eq!(reshaped.output_shape(), [9]);
        assert_eq!(reshaped.lower_a.shape(), [9, 816]);
        assert!(propagate(
            &mut query,
            &graph.nodes["reshape"],
            &reshaped,
            &input,
            &boxes,
            &mut accumulator
        )
        .unwrap());
        let result = take(&mut accumulator, NETWORK_INPUT);
        assert_eq!(result.input_shape(), [17, 48]);
        assert_eq!(result.output_shape(), [9]);
        let permutation = if axes[2] == 1 { [1, 0, 2] } else { [1, 2, 0] };
        // Independent authored tensor coordinates, not TransposeIndex.
        for s in 0..17 {
            for h in 0..3 {
                for d in 0..16 {
                    let coordinates = [s, h, d];
                    let out_shape = boxes["transpose"].shape();
                    let output_column = (coordinates[permutation[0]] * out_shape[1]
                        + coordinates[permutation[1]])
                        * out_shape[2]
                        + coordinates[permutation[2]];
                    let source_column = (s * 3 + h) * 16 + d;
                    for row in 0..9 {
                        for (actual, expected) in [
                            (&result.lower_a, &original.lower_a),
                            (&result.upper_a, &original.upper_a),
                            (
                                result.lower_a_err.as_ref().unwrap(),
                                original.lower_a_err.as_ref().unwrap(),
                            ),
                            (
                                result.upper_a_err.as_ref().unwrap(),
                                original.upper_a_err.as_ref().unwrap(),
                            ),
                        ] {
                            assert_eq!(
                                actual[[row, source_column]].to_bits(),
                                expected[[row, output_column]].to_bits()
                            );
                        }
                    }
                }
            }
        }
        bits(
            result.lower_b.iter().copied(),
            original.lower_b.iter().copied(),
        );
        bits(
            result.upper_b.iter().copied(),
            original.upper_b.iter().copied(),
        );
        assert_eq!(query.global_steps, 2);
        assert_eq!(query.work_receipt().completed, 2);
    }
}

fn rational(value: f32) -> BigRational {
    BigRational::from_float(f32_to_f64_exact(value)).unwrap()
}

fn exact(bounds: &LinearBounds, row: usize, point: &[BigRational], upper: bool) -> BigRational {
    let (a, b, error) = if upper {
        (
            &bounds.upper_a,
            &bounds.upper_b,
            bounds.upper_a_err.as_ref(),
        )
    } else {
        (
            &bounds.lower_a,
            &bounds.lower_b,
            bounds.lower_a_err.as_ref(),
        )
    };
    point
        .iter()
        .enumerate()
        .fold(rational(b[row]), |sum, (column, x)| {
            let nominal = rational(a[[row, column]]) * x;
            let error = error.map_or_else(BigRational::zero, |error| {
                rational(error[[row, column]]) * x.abs()
            });
            sum + nominal + if upper { error } else { -error }
        })
}

#[test]
fn attention_region_global_transpose_reshape_signed_error_only_exact_vertices_and_final_concretization(
) {
    let (graph, input, boxes) = graph(&[2, 3], &[2, 3, 1], &[0, 2, 1, 3]);
    let budget = authority();
    let mut original = original(9, 6);
    // The geometry-only fixture's zero-nominal row had lower bias .375 and
    // upper bias 0. Its exact box supports were .22265625 and .111328125:
    // an incompatible interval, correctly widened by final concretization.
    // Supply a compatible upper bias while retaining every signed nominal
    // and error coefficient, then require a finite concrete enclosure below.
    original.upper_b[8] = 1.0;
    let incoming = global_source(original.clone(), &[3, 2, 1], None, &budget).unwrap();
    let plan = CrownDispatchPlan::build(&graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    let mut query = AttentionRegionQuery::with_limits(None, 32 << 20, 100_000_000);
    assert!(propagate(
        &mut query,
        &graph.nodes["transpose"],
        &incoming,
        &input,
        &boxes,
        &mut accumulator
    )
    .unwrap());
    let reshaped = take(&mut accumulator, "reshape");
    assert!(propagate(
        &mut query,
        &graph.nodes["reshape"],
        &reshaped,
        &input,
        &boxes,
        &mut accumulator
    )
    .unwrap());
    let result = take(&mut accumulator, NETWORK_INPUT);
    let flat = consume::flat_incoming(&result, 6, None, &budget).unwrap();
    assert!(flat.lower_a.row(8).iter().all(|&v| v.to_bits() == 0));
    assert!(flat
        .upper_a
        .row(8)
        .iter()
        .all(|&v| v.to_bits() == (-0.0f32).to_bits()));
    assert!(flat
        .lower_a_err
        .as_ref()
        .unwrap()
        .row(8)
        .iter()
        .any(|&v| v > 0.0));
    for deadline in [None, Some(Instant::now() + Duration::from_secs(60))] {
        let concrete = result
            .concretize_sound_with_deadline(&input, deadline)
            .unwrap();
        for row in 0..9 {
            let lower = concrete.lower().as_slice().unwrap()[row];
            let upper = concrete.upper().as_slice().unwrap()[row];
            assert!(
                lower.is_finite() && upper.is_finite() && lower <= upper,
                "compatible original row {row} must retain finite supports: {lower}..{upper}, finite_deadline={}",
                deadline.is_some()
            );
        }
        for mask in 0..64 {
            let x: Vec<_> = (0..6)
                .map(|c| rational(if mask & (1 << c) == 0 { -0.5 } else { 0.75 }))
                .collect();
            // Independently apply original reshape followed by [1,0,2].
            let y: Vec<_> = (0..6).map(|c| x[(c % 2) * 3 + c / 2].clone()).collect();
            for row in 0..9 {
                assert_eq!(
                    exact(&flat, row, &x, false),
                    exact(&original, row, &y, false)
                );
                assert_eq!(exact(&flat, row, &x, true), exact(&original, row, &y, true));
                assert!(
                    rational(concrete.lower().as_slice().unwrap()[row])
                        <= exact(&original, row, &y, false)
                );
                assert!(
                    rational(concrete.upper().as_slice().unwrap()[row])
                        >= exact(&original, row, &y, true)
                );
            }
        }
    }
}

#[test]
fn attention_region_global_transpose_declines_wrong_permutation_scope_and_poison_before_mutation() {
    let (graph, input, mut boxes) = graph(&[2, 3], &[2, 3, 1], &[0, 2, 1, 3]);
    let budget = authority();
    let incoming = global_source(original(9, 6), &[3, 2, 1], None, &budget).unwrap();
    let plan = CrownDispatchPlan::build(&graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    accumulator.insert(NETWORK_INPUT, BatchedCrownBounds::Dense(incoming.clone()));
    let before = accumulator.total_memory_bytes();
    for case in 0..6 {
        let mut candidate = incoming.clone();
        let mut node = graph.nodes["transpose"].clone();
        match case {
            0 => {
                node = GraphNode::new(
                    "transpose",
                    Layer::Transpose(TransposeLayer::new(vec![1, 1, 2])),
                    vec!["reshape".into()],
                )
            }
            1 => {
                candidate.input_shape = vec![2, 3, 1];
            }
            2 => {
                boxes.insert(
                    "transpose".into(),
                    BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[3, 1, 2]))).unwrap(),
                );
                candidate.input_shape = vec![3, 1, 2];
            }
            3 => candidate.lower_a[[0, 0]] = f32::INFINITY,
            4 => candidate.upper_a_err.as_mut().unwrap()[[8, 3]] = f32::from_bits(0x8000_0001),
            5 => candidate = BatchedLinearBounds::identity(&[3, 2, 1]).unwrap(),
            _ => unreachable!(),
        }
        let mut query = AttentionRegionQuery::with_limits(None, 32 << 20, 100_000_000);
        assert!(!propagate(
            &mut query,
            &node,
            &candidate,
            &input,
            &boxes,
            &mut accumulator
        )
        .unwrap());
        assert_eq!(query.global_steps, 0);
        assert_eq!(accumulator.total_memory_bytes(), before);
        assert!(accumulator.contains_key(NETWORK_INPUT));
        // Restore only the test-owned original box; production never mutates it.
        boxes.insert(
            "transpose".into(),
            graph.nodes["transpose"]
                .layer()
                .propagate_ibp(&boxes["reshape"])
                .unwrap(),
        );
    }
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn attention_region_global_transpose_cancellation_and_cap_refusal_preserve_original_ledger() {
    let (graph, input, boxes) = graph(&[2, 3], &[2, 3, 1], &[0, 2, 1, 3]);
    let incoming = global_source(original(9, 6), &[3, 2, 1], None, &authority()).unwrap();
    let plan = CrownDispatchPlan::build(&graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    accumulator.insert(NETWORK_INPUT, BatchedCrownBounds::Dense(incoming.clone()));
    let before = accumulator.total_memory_bytes();
    let cancellation = CancellationToken::default();
    let memory = MemoryAuthority::new(1 << 20);
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: 1 << 20,
            persistent_bytes: 1 << 20,
            max_work: 100_000_000,
            max_proposals: 64,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let mut query = AttentionRegionQuery::from_authority(
        AttentionRegionAuthority {
            budget: Some(budget.clone()),
            cap: 1 << 20,
            deadline: None,
        },
        None,
        None,
    );
    let receipt = budget.receipt().unwrap();
    // flat_incoming validates 4*54 +2*9=234 real cells. Cancel during
    // the second authenticated original scan, after admission/copies occurred.
    let (accepted, inspected) =
        consume::cancel_during_validation(cancellation.clone(), 235, || {
            propagate(
                &mut query,
                &graph.nodes["transpose"],
                &incoming,
                &input,
                &boxes,
                &mut accumulator,
            )
        });
    assert!(!accepted.unwrap());
    assert!(inspected >= 235);
    assert_eq!(query.global_steps, 0);
    let after = budget.receipt().unwrap();
    assert!(after.work_reserved > receipt.work_reserved);
    assert_eq!(after.completed, receipt.completed);
    assert_eq!(after.live_scratch_bytes, receipt.live_scratch_bytes);
    assert_eq!(after.live_persistent_bytes, receipt.live_persistent_bytes);
    drop(query);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_pending_bytes + released.live_materialized_bytes,
        0
    );
    for (deadline, work) in [
        (None, 1),
        (
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
            100_000_000,
        ),
    ] {
        let mut query = AttentionRegionQuery::with_limits(deadline, 32 << 20, work);
        assert!(!propagate(
            &mut query,
            &graph.nodes["transpose"],
            &incoming,
            &input,
            &boxes,
            &mut accumulator
        )
        .unwrap());
        assert_eq!(query.global_steps, 0);
        assert_eq!(accumulator.total_memory_bytes(), before);
        assert!(accumulator.contains_key(NETWORK_INPUT));
    }
}
