// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Complete original Concat slots, exact independent coordinate arithmetic,
//! and admission/publication controls. No property row becomes a tensor axis.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::layers::{ConcatLayer, ReshapeLayer};
use crate::network::core::graph::dispatch_plan::CrownDispatchPlan;
use crate::types::BoundsProvenance;
use ndarray::{ArrayD, IxDyn};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

const CAP: usize = 32 << 20;
fn authority() -> (
    AttentionRegionAuthority,
    WorkBudget,
    MemoryAuthority,
    CancellationToken,
) {
    let memory = MemoryAuthority::new(CAP);
    let cancellation = CancellationToken::default();
    let deadline = Some(Instant::now() + Duration::from_secs(45));
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline,
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: MAX_WORK,
            max_proposals: MAX_REGIONS as u64,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (
        AttentionRegionAuthority {
            budget: Some(budget.clone()),
            cap: CAP,
            deadline,
        },
        budget,
        memory,
        cancellation,
    )
}
fn tensor(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), lo),
        ArrayD::from_elem(IxDyn(shape), hi),
    )
    .unwrap()
}
fn rat(value: f32) -> BigRational {
    BigRational::from_float(f32_to_f64_exact(value)).unwrap()
}
fn incoming(columns: usize) -> LinearBounds {
    let mut row = LinearBounds::new(
        Array2::from_shape_fn((4, columns), |(r, c)| {
            if r >= 2 {
                0.0
            } else {
                ((r * 3 + c * 5) % 9) as f32 * 0.125 - 0.5
            }
        }),
        Array1::from_vec(vec![-0.125, 0.5, 0.0, -0.0]),
        Array2::from_shape_fn((4, columns), |(r, c)| {
            if r >= 2 {
                -0.0
            } else {
                ((r * 7 + c * 2) % 11) as f32 * 0.0625 - 0.375
            }
        }),
        Array1::from_vec(vec![0.25, 0.125, 0.0, 0.0]),
    )
    .unwrap();
    row.lower_a_err = Some(Array2::from_shape_fn((4, columns), |(r, c)| {
        if r == 3 {
            0.0
        } else {
            ((r + c) % 5) as f32 * 0.0078125
        }
    }));
    row.upper_a_err = Some(Array2::from_shape_fn((4, columns), |(r, c)| {
        if r == 3 {
            0.0
        } else {
            ((r + 2 * c) % 7) as f32 * 0.00390625
        }
    }));
    row
}
fn value(row: &LinearBounds, r: usize, x: &[BigRational], upper: bool) -> BigRational {
    let (a, b, e) = if upper {
        (&row.upper_a, &row.upper_b, row.upper_a_err.as_ref())
    } else {
        (&row.lower_a, &row.lower_b, row.lower_a_err.as_ref())
    };
    x.iter().enumerate().fold(rat(b[r]), |sum, (c, x)| {
        let error = e.map_or_else(BigRational::zero, |e| rat(e[[r, c]])) * x.abs();
        sum + rat(a[[r, c]]) * x + if upper { error } else { -error }
    })
}
fn take(accumulator: &mut BatchedCrownAccumulator, source: &str) -> BatchedLinearBounds {
    match accumulator.take(source).unwrap() {
        BatchedCrownBounds::Dense(bounds) => bounds,
        _ => panic!("original full-column Concat must retain independent dense planes"),
    }
}

struct Fixture {
    graph: GraphNetwork,
    boxes: HashMap<String, BoundedTensor>,
    names: Vec<String>,
    slots: Vec<BoundedTensor>,
    dynamic: Vec<bool>,
    input: BoundedTensor,
}
fn fixture(layer: ConcatLayer, slots: Vec<BoundedTensor>, dynamic: Vec<bool>) -> Fixture {
    let input = slots[dynamic.iter().position(|&v| v).unwrap()].clone();
    let mut graph = GraphNetwork::new();
    let mut boxes = HashMap::new();
    let mut names = Vec::new();
    for (i, slot) in slots.iter().enumerate() {
        if dynamic[i] {
            let name = format!("dynamic-{i}");
            graph.add_node(GraphNode::from_input(
                &name,
                Layer::Reshape(ReshapeLayer::new(
                    slot.shape().iter().map(|&s| s as i64).collect(),
                )),
            ));
            boxes.insert(name.clone(), slot.clone());
            names.push(name);
        }
    }
    let output = layer
        .propagate_ibp_nary(&slots.iter().collect::<Vec<_>>())
        .unwrap();
    boxes.insert("concat".into(), output);
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(layer),
        names.clone(),
    ));
    graph.set_output("concat");
    Fixture {
        graph,
        boxes,
        names,
        slots,
        dynamic,
        input,
    }
}

// Independent authored coordinate oracle: decode every output coordinate,
// select its original slot, then encode the remaining original coordinates.
// This uses no production Slot, output_column, shape preparer, or split map.
fn original_concat(
    shape: &[usize],
    axis: usize,
    slot_shapes: &[Vec<usize>],
    points: &[Vec<BigRational>],
) -> Vec<BigRational> {
    (0..shape.iter().product::<usize>())
        .map(|mut column| {
            let mut coordinates = vec![0; shape.len()];
            for d in (0..shape.len()).rev() {
                coordinates[d] = column % shape[d];
                column /= shape[d];
            }
            let mut slot = 0;
            while coordinates[axis] >= slot_shapes[slot][axis] {
                coordinates[axis] -= slot_shapes[slot][axis];
                slot += 1;
            }
            let source = coordinates
                .iter()
                .zip(&slot_shapes[slot])
                .fold(0usize, |index, (&c, &extent)| index * extent + c);
            points[slot][source].clone()
        })
        .collect()
}
fn check_fixture(fixture: Fixture, axis: usize, normalized: Vec<Vec<usize>>) {
    let (authority, budget, memory, _) = authority();
    let mut query = AttentionRegionQuery::from_authority(authority, None, None);
    let shape = fixture.boxes["concat"].shape();
    let original = incoming(fixture.boxes["concat"].len());
    let carrier = global_source(original.clone(), shape, query.deadline, &budget).unwrap();
    let plan = CrownDispatchPlan::build(&fixture.graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    assert!(propagate(
        &mut query,
        &fixture.graph.nodes["concat"],
        &carrier,
        &fixture.input,
        &fixture.boxes,
        &mut accumulator
    )
    .unwrap());
    let parts: Vec<_> = fixture
        .names
        .iter()
        .map(|name| {
            let part = take(&mut accumulator, name);
            consume::flat_incoming(
                &part,
                part.input_shape().iter().product(),
                query.deadline,
                &budget,
            )
            .unwrap()
        })
        .collect();
    assert!(accumulator.is_empty());
    // Signed and error-only rows are compared against exact original slots,
    // including both endpoints of embedded interval constants.
    let total_columns: usize = fixture.slots.iter().map(BoundedTensor::len).sum();
    assert!(
        total_columns <= 12,
        "independent fixture enumerates every corner"
    );
    for mask in 0_usize..(1_usize << total_columns) {
        let mut offset = 0_usize;
        let points: Vec<Vec<_>> = fixture
            .slots
            .iter()
            .map(|slot| {
                let point = slot
                    .lower()
                    .iter()
                    .zip(slot.upper())
                    .map(|(&lo, &hi)| {
                        let x = rat(if (mask & (1_usize << offset)) == 0_usize {
                            lo
                        } else {
                            hi
                        });
                        offset += 1;
                        x
                    })
                    .collect();
                point
            })
            .collect();
        let original_output = original_concat(shape, axis, &normalized, &points);
        for r in 0..4 {
            for upper in [false, true] {
                let mut p = 0;
                let actual = fixture
                    .dynamic
                    .iter()
                    .enumerate()
                    .filter(|(_, dynamic)| **dynamic)
                    .fold(BigRational::zero(), |sum, (slot, _)| {
                        let next = value(&parts[p], r, &points[slot], upper);
                        p += 1;
                        sum + next
                    });
                let expected = value(&original, r, &original_output, upper);
                assert!(
                    if upper {
                        actual >= expected
                    } else {
                        actual <= expected
                    },
                    "mask={mask} row={r} upper={upper}"
                );
            }
        }
    }
    // Unique original column labels independently authenticate every split
    // permutation. Numerical widening of a bias cannot conceal a wrong
    // coefficient/error map, including signed-zero payload bits.
    let mut next = 0_usize;
    let labels: Vec<Vec<_>> = fixture
        .slots
        .iter()
        .map(|slot| {
            (0..slot.len())
                .map(|_| {
                    let label = rat(next as f32);
                    next += 1;
                    label
                })
                .collect()
        })
        .collect();
    let output_labels = original_concat(shape, axis, &normalized, &labels);
    let mut part = 0_usize;
    for (slot, &dynamic) in fixture.dynamic.iter().enumerate() {
        if !dynamic {
            continue;
        }
        for (column, label) in labels[slot].iter().enumerate() {
            let output = output_labels
                .iter()
                .position(|candidate| candidate == label)
                .unwrap();
            for row in 0..4 {
                for (a, expected) in [
                    (&parts[part].lower_a, &original.lower_a),
                    (&parts[part].upper_a, &original.upper_a),
                    (
                        parts[part].lower_a_err.as_ref().unwrap(),
                        original.lower_a_err.as_ref().unwrap(),
                    ),
                    (
                        parts[part].upper_a_err.as_ref().unwrap(),
                        original.upper_a_err.as_ref().unwrap(),
                    ),
                ] {
                    assert_eq!(
                        a[[row, column]].to_bits(),
                        expected[[row, output]].to_bits()
                    );
                }
            }
        }
        part += 1;
    }
    assert!(parts
        .iter()
        .all(|p| p.lower_a.row(2).iter().all(|&v| v == 0.0)));
    assert!(parts.iter().any(|p| p
        .lower_a_err
        .as_ref()
        .unwrap()
        .row(2)
        .iter()
        .any(|&v| v > 0.0)));
    assert_eq!(query.global_steps, 1);
    assert!(budget.receipt().unwrap().live_persistent_bytes > 0);
    drop(query);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}

#[test]
fn attention_region_global_concat_interleaved_original_constant_and_three_dynamic_slots() {
    let _env = crate::tests::lock_env_shared();
    let shapes = vec![vec![2, 1], vec![2, 1], vec![2, 2], vec![2, 1]];
    let constant = tensor(&shapes[1], -0.25, 0.5);
    let layer = ConcatLayer::with_constants(
        -1,
        shapes.clone(),
        vec![None, Some(constant.clone()), None, None],
    );
    check_fixture(
        fixture(
            layer,
            vec![
                tensor(&shapes[0], -0.5, 0.75),
                constant,
                tensor(&shapes[2], -0.75, 0.5),
                tensor(&shapes[3], -0.25, 1.0),
            ],
            vec![true, false, true, true],
        ),
        1,
        shapes,
    );
}
#[test]
fn attention_region_global_concat_original_nonlast_tensor_axis_keeps_property_rows() {
    let _env = crate::tests::lock_env_shared();
    let shapes = vec![vec![2, 1, 2], vec![2, 2, 2]];
    let constant = tensor(&shapes[0], -0.25, 0.25);
    let layer = ConcatLayer::with_constants(1, shapes.clone(), vec![Some(constant.clone()), None]);
    check_fixture(
        fixture(
            layer,
            vec![constant, tensor(&shapes[1], -0.5, 0.75)],
            vec![false, true],
        ),
        1,
        shapes,
    );
}
#[test]
fn attention_region_global_concat_original_cls_sequence_and_restored_singleton_batch() {
    let _env = crate::tests::lock_env_shared();
    for restored in [false, true] {
        let shapes = if restored {
            vec![vec![1, 1], vec![2]]
        } else {
            vec![vec![1, 2], vec![3, 2]]
        };
        let constant = tensor(&shapes[0], 0.125, 0.125);
        let layer =
            ConcatLayer::with_constants(0, shapes.clone(), vec![Some(constant.clone()), None])
                .with_restored_batch_axis_shift(restored);
        check_fixture(
            fixture(
                layer,
                vec![constant, tensor(&shapes[1], -0.5, 0.75)],
                vec![false, true],
            ),
            usize::from(restored),
            if restored {
                vec![vec![1, 1], vec![1, 2]]
            } else {
                shapes
            },
        );
    }
}
#[test]
fn attention_region_global_concat_constant_subnormal_dust_and_error_are_not_dropped() {
    let _env = crate::tests::lock_env_shared();
    let constant = tensor(&[1], f32::from_bits(1), f32::from_bits(1));
    let layer = ConcatLayer::with_constants(
        0,
        vec![vec![1], vec![1]],
        vec![Some(constant.clone()), None],
    );
    check_fixture(
        fixture(
            layer,
            vec![constant, tensor(&[1], -0.5, 0.75)],
            vec![false, true],
        ),
        0,
        vec![vec![1], vec![1]],
    );
}
#[test]
fn attention_region_global_concat_default_batched_route_reaches_embedded_original_constant() {
    let _env = crate::tests::lock_env_shared();
    let input = tensor(&[2, 2], -0.5, 0.75);
    let constant = tensor(&[2, 1], 0.25, 0.25);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "concat",
        Layer::Concat(ConcatLayer::with_constants(
            1,
            vec![vec![2, 1], vec![2, 2]],
            vec![Some(constant), None],
        )),
    ));
    graph.add_node(GraphNode::new(
        "mean",
        Layer::ReduceMean(ReduceMeanLayer::new(vec![1], false)),
        vec!["concat".into()],
    ));
    graph.set_output("mean");
    let original_ibp = graph.propagate_ibp(&input).unwrap();
    let (authority, budget, memory, _) = authority();
    let _scope = AttentionRegionScope::install(authority);
    let (result, _) = graph
        .propagate_crown_batched_with_attention_region_receipt_and_deadline(
            &input,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .unwrap();
    assert_eq!(result.provenance, BoundsProvenance::Crown);
    assert_eq!(result.bounds.shape(), [2]);
    for mask in 0_usize..16 {
        for r in 0..2 {
            let x = (0..2)
                .map(|c| {
                    rat(if mask & (1_usize << (r * 2 + c)) == 0_usize {
                        -0.5
                    } else {
                        0.75
                    })
                })
                .sum::<BigRational>();
            let y = (x + rat(0.25)) / BigRational::from_integer(3.into());
            assert!(
                rat(result.bounds.lower().as_slice().unwrap()[r]) <= y,
                "original Concat lower containment: row={r} corner={mask} exact={y} crown={:?} ibp={:?}",
                result.bounds,
                original_ibp,
            );
            assert!(
                rat(result.bounds.upper().as_slice().unwrap()[r]) >= y,
                "original Concat upper containment: row={r} corner={mask} exact={y} crown={:?} ibp={:?}",
                result.bounds,
                original_ibp,
            );
        }
    }
    assert!(budget.receipt().unwrap().completed >= 3);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}
fn small() -> Fixture {
    let c = tensor(&[1, 1], 0.25, 0.25);
    fixture(
        ConcatLayer::with_constants(1, vec![vec![1, 1], vec![1, 2]], vec![Some(c.clone()), None]),
        vec![c, tensor(&[1, 2], -0.5, 0.75)],
        vec![false, true],
    )
}
#[test]
fn attention_region_global_concat_relabelled_geometry_and_slot_count_publish_nothing() {
    let _env = crate::tests::lock_env_shared();
    for case in 0..3 {
        let mut fixture = small();
        match case {
            0 => {
                fixture
                    .boxes
                    .insert("concat".into(), tensor(&[3, 1], -1.0, 1.0));
            }
            1 => {
                fixture
                    .graph
                    .nodes
                    .get_mut("concat")
                    .unwrap()
                    .inputs
                    .push(fixture.names[0].clone());
            }
            _ => {
                fixture.boxes.remove(&fixture.names[0]);
            }
        }
        let (authority, budget, _, _) = authority();
        let mut query = AttentionRegionQuery::from_authority(authority, None, None);
        let carrier = global_source(incoming(3), &[1, 3], query.deadline, &budget).unwrap();
        let plan = CrownDispatchPlan::build(&fixture.graph).unwrap();
        let mut accumulator = BatchedCrownAccumulator::new(&plan);
        assert!(!propagate(
            &mut query,
            &fixture.graph.nodes["concat"],
            &carrier,
            &fixture.input,
            &fixture.boxes,
            &mut accumulator
        )
        .unwrap());
        assert!(accumulator.is_empty());
        assert_eq!(query.global_steps, 0);
    }
}
#[test]
fn attention_region_global_concat_occupied_original_ledger_declines_without_partial_outputs() {
    let _env = crate::tests::lock_env_shared();
    let fixture = small();
    let (authority, budget, memory, _) = authority();
    let mut query = AttentionRegionQuery::from_authority(authority, None, None);
    let carrier = global_source(incoming(3), &[1, 3], query.deadline, &budget).unwrap();
    let live = budget.receipt().unwrap().live_persistent_bytes;
    let occupied = budget
        .lease(WorkRequest {
            persistent_bytes: CAP - live,
            ..WorkRequest::default()
        })
        .unwrap();
    let plan = CrownDispatchPlan::build(&fixture.graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    assert!(!propagate(
        &mut query,
        &fixture.graph.nodes["concat"],
        &carrier,
        &fixture.input,
        &fixture.boxes,
        &mut accumulator
    )
    .unwrap());
    assert!(accumulator.is_empty());
    assert_eq!(query.global_steps, 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, CAP);
    drop(occupied);
    drop(query);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_materialized_bytes + released.live_pending_bytes,
        0
    );
}
#[test]
fn attention_region_global_concat_same_cancellation_and_expired_clock_publish_nothing() {
    let _env = crate::tests::lock_env_shared();
    for cancel in [false, true] {
        let fixture = small();
        let (authority, budget, _, cancellation) = authority();
        let mut query = AttentionRegionQuery::from_authority(authority, None, None);
        let carrier = global_source(incoming(3), &[1, 3], query.deadline, &budget).unwrap();
        let plan = CrownDispatchPlan::build(&fixture.graph).unwrap();
        let mut accumulator = BatchedCrownAccumulator::new(&plan);
        if cancel {
            cancellation.cancel();
        } else {
            query.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
        }
        let result = propagate(
            &mut query,
            &fixture.graph.nodes["concat"],
            &carrier,
            &fixture.input,
            &fixture.boxes,
            &mut accumulator,
        );
        assert!(!matches!(result, Ok(true)));
        assert!(accumulator.is_empty());
        assert_eq!(query.global_steps, 0);
    }
}
