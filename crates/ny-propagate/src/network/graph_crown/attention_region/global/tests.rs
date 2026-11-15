// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{
    AddLayer, BilinearCrownLayer, MulConstantLayer, ReLULayer, ReshapeLayer, SoftmaxLayer,
    TransposeLayer,
};
use crate::network::core::graph::dispatch_plan::CrownDispatchPlan;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

fn rat(value: f32) -> BigRational {
    BigRational::from_float(f32_to_f64_exact(value)).unwrap()
}
fn budget() -> WorkBudget {
    WorkBudget::new(None, 32 << 20, 32 << 20, 100_000_000, 64, Duration::ZERO).unwrap()
}
fn incoming(rows: usize, columns: usize) -> LinearBounds {
    let mut bounds = LinearBounds::new(
        Array2::from_shape_fn((rows, columns), |(r, c)| {
            ((r * 5 + c * 3) % 11) as f32 * 0.125 - 0.625
        }),
        Array1::from_shape_fn(rows, |r| r as f32 * 0.0625 - 0.125),
        Array2::from_shape_fn((rows, columns), |(r, c)| {
            ((r * 7 + c * 2) % 13) as f32 * 0.0625 - 0.375
        }),
        Array1::from_shape_fn(rows, |r| 0.25 - r as f32 * 0.03125),
    )
    .unwrap();
    bounds.lower_a_err = Some(Array2::from_shape_fn((rows, columns), |(r, c)| {
        ((r + c) % 3) as f32 * 0.015625
    }));
    bounds.upper_a_err = Some(Array2::from_shape_fn((rows, columns), |(r, c)| {
        ((r + 2 * c) % 5) as f32 * 0.0078125
    }));
    bounds
}
fn value(bounds: &LinearBounds, row: usize, x: &[BigRational], upper: bool) -> BigRational {
    let (a, b, e) = if upper {
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
    let mut sum = rat(b[row]);
    for (c, x) in x.iter().enumerate() {
        sum += rat(a[[row, c]]) * x;
        let error = e.map_or_else(BigRational::zero, |e| rat(e[[row, c]])) * x.abs();
        if upper {
            sum += error;
        } else {
            sum -= error;
        }
    }
    sum
}

#[test]
fn attention_region_global_mean_original_nondyadic_axis_and_error_only_rows() {
    let budget = budget();
    let mut original = incoming(9, 4);
    original.lower_a.row_mut(8).fill(0.0);
    original.upper_a.row_mut(8).fill(0.0);
    let out = mean(&original, 12, 3, 2, None, &budget).unwrap();
    // Original shape [2,3,2], reduce axis1, keepdims=false => [2,2].
    for mask in 0..4096usize {
        let x: Vec<_> = (0..12)
            .map(|c| rat(if mask & (1 << c) == 0 { -0.5 } else { 0.75 }))
            .collect();
        let y: Vec<_> = (0..4)
            .map(|c| {
                (0..3)
                    .map(|k| x[(c / 2) * 6 + k * 2 + c % 2].clone())
                    .sum::<BigRational>()
                    / BigRational::from_integer(3.into())
            })
            .collect();
        for row in 0..9 {
            assert!(value(&out, row, &x, false) <= value(&original, row, &y, false));
            assert!(value(&out, row, &x, true) >= value(&original, row, &y, true));
        }
    }
    assert!(out.lower_a.row(8).iter().all(|&v| v == 0.0));
    assert!(out
        .lower_a_err
        .as_ref()
        .unwrap()
        .row(8)
        .iter()
        .any(|&v| v > 0.0));
    assert!(mean_profile(&ReduceMeanLayer::new(vec![-2], false), &[2, 3, 2], &[2, 2]).is_some());
    assert!(mean_profile(&ReduceMeanLayer::new(vec![-2], false), &[2, 3, 2], &[3, 2]).is_none());
}

#[test]
fn attention_region_global_linear_original_signed_products_bias_once_and_error_planes() {
    let budget = budget();
    let layer = LinearLayer::new(
        arr2(&[[0.25, -0.5], [-0.75, 0.125], [0.5, 0.375]]),
        Some(arr1(&[0.125, -0.25, 0.375])),
    )
    .unwrap();
    let original = incoming(9, 6);
    let result = linear(&original, &layer, 2, None, &budget).unwrap();
    for mask in 0..16usize {
        let x: Vec<_> = (0..4)
            .map(|c| rat(if mask & (1 << c) == 0 { -0.75 } else { 0.625 }))
            .collect();
        let y: Vec<_> = (0..6)
            .map(|k| {
                (0..2).fold(rat(layer.bias().unwrap()[k % 3]), |sum, c| {
                    sum + rat(layer.weight()[[k % 3, c]]) * &x[(k / 3) * 2 + c]
                })
            })
            .collect();
        for row in 0..9 {
            assert!(
                value(&result, row, &x, false) <= value(&original, row, &y, false),
                "lower mask={mask} row={row}"
            );
            assert!(
                value(&result, row, &x, true) >= value(&original, row, &y, true),
                "upper mask={mask} row={row}"
            );
        }
    }
    let zero = LinearBounds::new(
        Array2::zeros((2, 6)),
        arr1(&[-0.5, 0.0]),
        Array2::zeros((2, 6)),
        arr1(&[0.25, 0.0]),
    )
    .unwrap();
    let zero = linear(&zero, &layer, 2, None, &budget).unwrap();
    assert!(zero.lower_a.iter().all(|&v| v == 0.0));
    assert!(zero.lower_a_err.as_ref().unwrap().iter().all(|&v| v == 0.0));
    assert!(zero.lower_b[0] <= -0.5 && zero.upper_b[0] >= 0.25);
}

#[test]
fn attention_region_global_subnormal_nominal_and_error_publish_normal_enclosures() {
    let budget = budget();
    let tiny = f32::from_bits(1);
    let mut original = LinearBounds::new(
        arr2(&[[tiny]]),
        arr1(&[0.0]),
        arr2(&[[-tiny]]),
        arr1(&[0.0]),
    )
    .unwrap();
    original.lower_a_err = Some(arr2(&[[tiny]]));
    original.upper_a_err = Some(arr2(&[[tiny]]));
    let result = mean(&original, 2, 2, 1, None, &budget).unwrap();
    for array in [
        &result.lower_a,
        &result.upper_a,
        result.lower_a_err.as_ref().unwrap(),
        result.upper_a_err.as_ref().unwrap(),
    ] {
        assert!(array.iter().all(|v| {
            let m = v.to_bits() & 0x7fff_ffff;
            m == 0 || m >= 0x0080_0000
        }));
    }
    assert!(result
        .lower_a_err
        .as_ref()
        .unwrap()
        .iter()
        .all(|&v| v >= f32::MIN_POSITIVE));
    for point in [[-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]] {
        let x: Vec<_> = point.into_iter().map(rat).collect();
        let y = vec![(&x[0] + &x[1]) / BigRational::from_integer(2.into())];
        assert!(value(&result, 0, &x, false) <= value(&original, 0, &y, false));
        assert!(value(&result, 0, &x, true) >= value(&original, 0, &y, true));
    }
}

#[test]
fn attention_region_global_source_preserves_all_cells_and_error_planes_without_batching_properties()
{
    let budget = budget();
    let original = incoming(9, 6);
    let saved = original.clone();
    let pointers = [
        original.lower_a.as_ptr(),
        original.upper_a.as_ptr(),
        original.lower_b.as_ptr(),
        original.upper_b.as_ptr(),
        original.lower_a_err.as_ref().unwrap().as_ptr(),
        original.upper_a_err.as_ref().unwrap().as_ptr(),
    ];
    let result = global_source(original, &[1, 3, 2], None, &budget).unwrap();
    assert_eq!(result.lower_a.shape(), &[9, 6]);
    assert_eq!(result.lower_b.shape(), &[9]);
    assert_eq!(result.input_shape(), &[1, 3, 2]);
    assert_eq!(result.output_shape(), &[9]);
    assert_eq!(
        pointers,
        [
            result.lower_a.as_ptr(),
            result.upper_a.as_ptr(),
            result.lower_b.as_ptr(),
            result.upper_b.as_ptr(),
            result.lower_a_err.as_ref().unwrap().as_ptr(),
            result.upper_a_err.as_ref().unwrap().as_ptr()
        ]
    );
    for (actual, original) in [
        (&result.lower_a, saved.lower_a.into_dyn()),
        (&result.upper_a, saved.upper_a.into_dyn()),
        (
            result.lower_a_err.as_ref().unwrap(),
            saved.lower_a_err.unwrap().into_dyn(),
        ),
        (
            result.upper_a_err.as_ref().unwrap(),
            saved.upper_a_err.unwrap().into_dyn(),
        ),
    ] {
        for (&a, &o) in actual.iter().zip(original.iter()) {
            assert_eq!(a.to_bits(), o.to_bits());
        }
    }
    // Exercise the SAME legacy and finite public concretization boundaries
    // against all six original input cells, not just final feature width2.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 3, 2]), -0.5),
        ArrayD::from_elem(IxDyn(&[1, 3, 2]), 0.75),
    )
    .unwrap();
    for deadline in [None, Some(Instant::now() + Duration::from_secs(30))] {
        let output = result
            .concretize_sound_with_deadline(&input, deadline)
            .unwrap();
        assert_eq!(output.shape(), &[9]);
        for mask in 0..64usize {
            let x: Vec<_> = (0..6)
                .map(|c| rat(if mask & (1 << c) == 0 { -0.5 } else { 0.75 }))
                .collect();
            for row in 0..9 {
                let mut lower = rat(saved.lower_b[row]);
                let mut upper = rat(saved.upper_b[row]);
                for c in 0..6 {
                    lower += rat(result.lower_a[[row, c]]) * &x[c]
                        - rat(result.lower_a_err.as_ref().unwrap()[[row, c]]) * x[c].abs();
                    upper += rat(result.upper_a[[row, c]]) * &x[c]
                        + rat(result.upper_a_err.as_ref().unwrap()[[row, c]]) * x[c].abs();
                }
                assert!(rat(output.lower()[[row]]) <= lower);
                assert!(rat(output.upper()[[row]]) >= upper);
            }
        }
    }
}

#[test]
fn attention_region_global_add_constant_original_broadcast_signed_and_error_only_rows() {
    let budget = budget();
    for (shape, constant_shape, constants, expected) in [
        (
            vec![3, 2],
            vec![2],
            vec![0.25, -0.5],
            vec![0.25, -0.5, 0.25, -0.5, 0.25, -0.5],
        ),
        (
            vec![2, 3],
            vec![2, 1],
            vec![-0.25, 0.5],
            vec![-0.25, -0.25, -0.25, 0.5, 0.5, 0.5],
        ),
        (
            vec![2, 1, 3],
            vec![2],
            vec![-0.25, 0.5],
            vec![-0.25, -0.25, -0.25, 0.5, 0.5, 0.5],
        ),
        (
            vec![2, 3],
            vec![1, 2, 3],
            vec![0.25, -0.5, 0.125, -0.25, 0.5, -0.125],
            vec![0.25, -0.5, 0.125, -0.25, 0.5, -0.125],
        ),
        (vec![2, 3], vec![], vec![-0.25], vec![-0.25; 6]),
    ] {
        let layer = AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&constant_shape), constants).unwrap(),
        );
        let effective = layer.effective_broadcast_shape(&shape).unwrap();
        let output = crate::shape::broadcast_shapes(&shape, &effective).unwrap();
        let indexing = BroadcastIndex::new(&layer, &shape, &output)
            .unwrap()
            .unwrap();
        for (column, &value) in expected.iter().enumerate() {
            let value: f32 = value;
            assert_eq!(
                layer.constant().as_slice().unwrap()[indexing.index(column)].to_bits(),
                value.to_bits()
            );
        }
        let mut original = incoming(9, 6);
        original.lower_a.row_mut(8).fill(0.0);
        original.upper_a.row_mut(8).fill(0.0);
        let saved = original.clone();
        let pointers = [
            original.lower_a.as_ptr(),
            original.upper_a.as_ptr(),
            original.lower_a_err.as_ref().unwrap().as_ptr(),
            original.upper_a_err.as_ref().unwrap().as_ptr(),
        ];
        let result = add_constant(original, &layer, indexing, None, &budget).unwrap();
        assert_eq!(
            pointers,
            [
                result.lower_a.as_ptr(),
                result.upper_a.as_ptr(),
                result.lower_a_err.as_ref().unwrap().as_ptr(),
                result.upper_a_err.as_ref().unwrap().as_ptr()
            ]
        );
        assert_eq!(result.lower_a, saved.lower_a);
        assert_eq!(result.upper_a, saved.upper_a);
        assert_eq!(result.lower_a_err, saved.lower_a_err);
        assert_eq!(result.upper_a_err, saved.upper_a_err);
        assert!(result.lower_b[8] < saved.lower_b[8]);
        assert!(result.upper_b[8] > saved.upper_b[8]);
        for mask in 0..64usize {
            let x: Vec<_> = (0..6)
                .map(|c| rat(if mask & (1 << c) == 0 { -0.75 } else { 0.5 }))
                .collect();
            let y: Vec<_> = x.iter().zip(&expected).map(|(x, &k)| x + rat(k)).collect();
            for row in 0..9 {
                assert!(
                    value(&result, row, &x, false) <= value(&saved, row, &y, false),
                    "lower shape={shape:?} mask={mask} row={row}"
                );
                assert!(
                    value(&result, row, &x, true) >= value(&saved, row, &y, true),
                    "upper shape={shape:?} mask={mask} row={row}"
                );
            }
        }
    }
}

#[test]
fn attention_region_global_add_constant_declines_expansion_ambiguity_shape_and_nonfinite_originals()
{
    let broadcast =
        AddConstantLayer::new(ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.25, -0.5]).unwrap());
    assert!(BroadcastIndex::new(&broadcast, &[1, 3], &[2, 3])
        .unwrap()
        .is_none());
    assert!(BroadcastIndex::new(&broadcast, &[2, 3], &[3, 2])
        .unwrap()
        .is_none());
    let ambiguous = AddConstantLayer::new(arr1(&[0.25, -0.5]).into_dyn());
    assert!(BroadcastIndex::new(&ambiguous, &[2, 3, 2], &[2, 3, 2])
        .unwrap()
        .is_none());
    let mut transposed = Array2::from_shape_vec((2, 3), vec![0.125; 6])
        .unwrap()
        .into_dyn();
    transposed.swap_axes(0, 1);
    let transposed = AddConstantLayer::new(transposed);
    assert!(BroadcastIndex::new(&transposed, &[3, 2], &[3, 2])
        .unwrap()
        .is_none());
    for poisoned in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let layer = AddConstantLayer {
            constant: ArrayD::from_elem(IxDyn(&[2]), poisoned),
        };
        let indexing = BroadcastIndex::new(&layer, &[3, 2], &[3, 2])
            .unwrap()
            .unwrap();
        assert!(add_constant(incoming(9, 6), &layer, indexing, None, &budget()).is_err());
    }
}

#[test]
fn attention_region_global_add_constant_cancellation_retains_original_pending_and_work() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "shift",
        Layer::AddConstant(AddConstantLayer::new(arr1(&[0.25, -0.5]).into_dyn())),
    ));
    graph.set_output("shift");
    let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[3, 2]))).unwrap();
    let output = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.25, -0.5, 0.25, -0.5, 0.25, -0.5]).unwrap(),
    )
    .unwrap();
    let boxes = HashMap::from([("shift".into(), output)]);
    let mut bounds = global_source(incoming(9, 6), &[3, 2], None, &budget()).unwrap();
    bounds.output_shape = vec![9];
    let plan = CrownDispatchPlan::build(&graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    accumulator.insert(NETWORK_INPUT, BatchedCrownBounds::Dense(bounds.clone()));
    let before = accumulator.total_memory_bytes();
    use crate::certify::residency::{CancellationToken, MemoryAuthority};
    use crate::certify::work::{WorkLimits, WorkReserve};
    let cancellation = CancellationToken::default();
    let memory = MemoryAuthority::new(1 << 20);
    let authority = WorkBudget::new_in(
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
            budget: Some(authority.clone()),
            cap: 1 << 20,
            deadline: None,
        },
        None,
        None,
    );
    let original_receipt = authority.receipt().unwrap();
    // flat_incoming checks 4*54+2*9 actual values; cancellation occurs
    // after that complete scan, during the shifted carrier's final validation.
    let (accepted, inspected) =
        consume::cancel_during_validation(cancellation.clone(), 235, || {
            propagate(
                &mut query,
                graph.nodes.get("shift").unwrap(),
                &bounds,
                &input,
                &boxes,
                &mut accumulator,
            )
        });
    assert!(!accepted.unwrap());
    assert!(inspected > 234);
    assert!(accumulator.contains_key(NETWORK_INPUT));
    assert_eq!(accumulator.total_memory_bytes(), before);
    assert_eq!(query.global_steps, 0);
    assert!(cancellation.is_cancelled());
    let after = authority.receipt().unwrap();
    assert!(after.work_reserved > original_receipt.work_reserved);
    assert_eq!(
        after.live_scratch_bytes,
        original_receipt.live_scratch_bytes
    );
    assert_eq!(
        after.live_persistent_bytes,
        original_receipt.live_persistent_bytes
    );
}

fn production_graph(with_constant: bool) -> (GraphNetwork, LinearLayer, LinearLayer, LinearLayer) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
    ));
    let mut kt = TransposeLayer::new(vec![0, 2, 1]);
    kt.set_input_shape(vec![1, 3, 2]);
    graph.add_node(GraphNode::from_input("kt", Layer::Transpose(kt)));
    graph.add_node(GraphNode::binary(
        "qk",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        "q",
        "kt",
    ));
    graph.add_node(GraphNode::new(
        "scale",
        Layer::MulConstant(MulConstantLayer::scalar(0.25)),
        vec!["qk".into()],
    ));
    graph.add_node(GraphNode::new(
        "prob",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["scale".into()],
    ));
    graph.add_node(GraphNode::binary(
        "pv",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        "prob",
        NETWORK_INPUT,
    ));
    graph.add_node(GraphNode::new(
        "reshape",
        Layer::Reshape(ReshapeLayer::new(vec![1, 3, 2])),
        vec!["pv".into()],
    ));
    let first = LinearLayer::new(
        arr2(&[[0.5, -0.25], [-0.125, 0.375], [0.25, 0.125]]),
        Some(arr1(&[0.5, 0.5, 0.5])),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "first",
        Layer::Linear(first.clone()),
        vec!["reshape".into()],
    ));
    if with_constant {
        graph.add_node(GraphNode::new(
            "constant_bias",
            Layer::AddConstant(AddConstantLayer::new(
                arr1(&[0.25, -0.125, 0.0625]).into_dyn(),
            )),
            vec!["first".into()],
        ));
    }
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec![if with_constant {
            "constant_bias".into()
        } else {
            "first".into()
        }],
    ));
    let second = LinearLayer::new(
        arr2(&[[0.5, -0.25, 0.125], [-0.25, 0.375, 0.5]]),
        Some(arr1(&[0.125, -0.25])),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "second",
        Layer::Linear(second.clone()),
        vec!["relu".into()],
    ));
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "second",
        "reshape",
    ));
    graph.add_node(GraphNode::new(
        "mean",
        Layer::ReduceMean(ReduceMeanLayer::new(vec![-2], false)),
        vec!["residual".into()],
    ));
    let head = LinearLayer::new(
        Array2::from_shape_fn((9, 2), |(r, c)| ((r * 3 + c * 5) % 9) as f32 * 0.125 - 0.5),
        Some(Array1::from_shape_fn(9, |r| r as f32 * 0.03125 - 0.125)),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "head",
        Layer::Linear(head.clone()),
        vec!["mean".into()],
    ));
    graph.set_output("head");
    (graph, first, second, head)
}

#[test]
fn attention_region_default_global_mean_affine_reshape_corridor_reaches_original_pv() {
    check_production_corridor(false);
}

#[test]
fn attention_region_default_global_original_broadcast_constant_corridor_reaches_pv() {
    check_production_corridor(true);
}

fn check_production_corridor(with_constant: bool) {
    let (graph, first, second, head) = production_graph(with_constant);
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 3, 2]), -0.125),
        ArrayD::from_elem(IxDyn(&[1, 3, 2]), 0.25),
    )
    .unwrap();
    let (report, receipt) = graph
        .propagate_crown_batched_with_attention_region_receipt(&input)
        .unwrap();
    assert_eq!(
        receipt.composed, 1,
        "ordinary original corridor must reach PV"
    );
    assert_eq!(receipt.batched, 1);
    assert_eq!(receipt.rows, 9);
    for mask in 0..64usize {
        let x: Vec<_> = (0..6)
            .map(|c| rat(if mask & (1 << c) == 0 { -0.125 } else { 0.25 }))
            .collect();
        let v: Vec<_> = (0..2)
            .map(|c| {
                (0..3).map(|k| x[k * 2 + c].clone()).sum::<BigRational>()
                    / BigRational::from_integer(3.into())
            })
            .collect();
        let first_value: Vec<_> = (0..3)
            .map(|r| {
                let value = (0..2).fold(rat(first.bias().unwrap()[r]), |s, c| {
                    s + rat(first.weight()[[r, c]]) * &v[c]
                });
                value
                    + rat(if with_constant {
                        [0.25, -0.125, 0.0625][r]
                    } else {
                        0.0
                    })
            })
            .collect();
        assert!(first_value.iter().all(|v| v >= &BigRational::zero()));
        let y: Vec<_> = (0..2)
            .map(|r| {
                (0..3).fold(rat(second.bias().unwrap()[r]), |s, c| {
                    s + rat(second.weight()[[r, c]]) * &first_value[c]
                }) + &v[r]
            })
            .collect();
        for row in 0..9 {
            let exact = (0..2).fold(rat(head.bias().unwrap()[row]), |s, c| {
                s + rat(head.weight()[[row, c]]) * &y[c]
            });
            assert!(
                rat(report.bounds.lower().as_slice().unwrap()[row]) <= exact,
                "lower mask={mask} row={row}"
            );
            assert!(
                rat(report.bounds.upper().as_slice().unwrap()[row]) >= exact,
                "upper mask={mask} row={row}"
            );
        }
    }
    let ordinary = graph.propagate_crown_batched(&input).unwrap();
    assert_eq!(ordinary.lower(), report.bounds.lower());
    assert_eq!(ordinary.upper(), report.bounds.upper());
}

#[test]
fn attention_region_global_corridor_low_budget_keeps_pending_and_no_new_allowance() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "mean",
        Layer::ReduceMean(ReduceMeanLayer::new(vec![0], false)),
    ));
    graph.set_output("mean");
    let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[3, 2]))).unwrap();
    let output = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[2]))).unwrap();
    let boxes = HashMap::from([("mean".into(), output)]);
    let bounds = BatchedLinearBounds::identity(&[2]).unwrap();
    let plan = CrownDispatchPlan::build(&graph).unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(&plan);
    accumulator.insert(
        NETWORK_INPUT,
        BatchedCrownBounds::Dense(BatchedLinearBounds::identity(&[3, 2]).unwrap()),
    );
    let before = accumulator.total_memory_bytes();
    let mut query = AttentionRegionQuery::with_limits(None, 1 << 20, 1);
    let budget = query.budget.clone().unwrap();
    assert!(!propagate(
        &mut query,
        graph.nodes.get("mean").unwrap(),
        &bounds,
        &input,
        &boxes,
        &mut accumulator
    )
    .unwrap());
    assert!(accumulator.contains_key(NETWORK_INPUT));
    assert_eq!(accumulator.total_memory_bytes(), before);
    assert_eq!(query.global_steps, 0);
    assert!(budget.shares_ledger(query.budget.as_ref().unwrap()));
    let mut query = AttentionRegionQuery::with_limits(
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap(),
        ),
        1 << 20,
        1_000_000,
    );
    assert!(!propagate(
        &mut query,
        graph.nodes.get("mean").unwrap(),
        &bounds,
        &input,
        &boxes,
        &mut accumulator
    )
    .unwrap());
    assert_eq!(accumulator.total_memory_bytes(), before);
}
