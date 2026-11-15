// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{
    AddLayer, BilinearCrownLayer, FlattenLayer, LinearLayer, MulConstantLayer, SoftmaxLayer,
    TransposeLayer,
};
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

fn box_for(shape: &[usize], radius: f32) -> BoundedTensor {
    let count = shape.iter().product();
    let center: Vec<f32> = (0..count)
        .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.125)
        .collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), center.iter().map(|v| v - radius).collect()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), center.iter().map(|v| v + radius).collect()).unwrap(),
    )
    .unwrap()
}

fn graph(zero_q: bool, extra_paths: bool) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::MulConstant(MulConstantLayer::scalar(if zero_q { 0.0 } else { 1.0 })),
    ));
    let mut transpose = TransposeLayer::new(vec![0, 2, 1]);
    transpose.set_input_shape(vec![1, 2, 2]);
    graph.add_node(GraphNode::from_input("kt", Layer::Transpose(transpose)));
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
    let final_name = if extra_paths {
        graph.add_node(GraphNode::binary(
            "prob_residual",
            Layer::Add(AddLayer),
            "pv",
            "prob",
        ));
        graph.add_node(GraphNode::binary(
            "q_residual",
            Layer::Add(AddLayer),
            "prob_residual",
            "q",
        ));
        graph.add_node(GraphNode::binary(
            "kt_residual",
            Layer::Add(AddLayer),
            "q_residual",
            "kt",
        ));
        graph.add_node(GraphNode::binary(
            "score_residual",
            Layer::Add(AddLayer),
            "kt_residual",
            "scale",
        ));
        graph.add_node(GraphNode::binary(
            "input_residual",
            Layer::Add(AddLayer),
            "score_residual",
            NETWORK_INPUT,
        ));
        "input_residual"
    } else {
        "pv"
    };
    graph.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec![final_name.into()],
    ));
    graph.add_node(GraphNode::new(
        "head",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0, -0.5, 0.25, -1.0], [-0.5, 0.25, 1.0, 0.5]]),
                Some(arr1(&[0.125, -0.25])),
            )
            .unwrap(),
        ),
        vec!["flat".into()],
    ));
    graph.set_output("head");
    graph
}

fn rat(value: f32) -> BigRational {
    BigRational::from_float(value as f64).unwrap()
}

fn original_uniform(x: &[f32; 4], extra: bool) -> Vec<BigRational> {
    let half = rat(0.5);
    (0..4)
        .map(|i| {
            let average = (rat(x[i % 2]) + rat(x[2 + i % 2])) * &half;
            if extra {
                average + &half + rat(x[i]) + rat(x[(i % 2) * 2 + i / 2])
            } else {
                average
            }
        })
        .collect()
}

fn assert_exact_containment(result: &BoundedTensor, input: &BoundedTensor, extra: bool) {
    let lower = input.lower().as_slice().unwrap();
    let upper = input.upper().as_slice().unwrap();
    let weights = [[1.0, -0.5, 0.25, -1.0], [-0.5, 0.25, 1.0, 0.5]];
    for mask in 0..16 {
        let x = std::array::from_fn(|i| {
            if mask & (1 << i) == 0 {
                lower[i]
            } else {
                upper[i]
            }
        });
        let y = original_uniform(&x, extra);
        for row in 0..2 {
            let value = weights[row]
                .iter()
                .zip(&y)
                .fold(rat(if row == 0 { 0.125 } else { -0.25 }), |sum, (&w, v)| {
                    sum + rat(w) * v
                });
            assert!(rat(result.lower().as_slice().unwrap()[row]) <= value,
                "original lower containment extra={extra} mask={mask} row={row} result={:?} exact={value}", result.lower());
            assert!(rat(result.upper().as_slice().unwrap()[row]) >= value,
                "original upper containment extra={extra} mask={mask} row={row} result={:?} exact={value}", result.upper());
        }
    }
}

#[test]
fn attention_region_default_batched_original_topology_preserves_extra_pending_paths() {
    let input = box_for(&[1, 2, 2], 0.03125);
    for extra in [false, true] {
        let graph = graph(true, extra);
        let (result, receipt) = graph
            .propagate_crown_batched_with_attention_region_receipt(&input)
            .unwrap();
        assert_eq!(
            receipt.batched, 1,
            "a positive default coordinator arrival is required"
        );
        assert_eq!(receipt.composed, 1);
        assert_eq!(receipt.rows, 2);
        assert!(receipt.returned_coefficient_elements > 0);
        assert_exact_containment(&result.bounds, &input, extra);
        // The unchanged public default uses precisely the same production
        // runtime policy as the receipt seam, not an experimental retry arm.
        let default = graph.propagate_crown_batched(&input).unwrap();
        assert_eq!(default.lower(), result.bounds.lower());
        assert_eq!(default.upper(), result.bounds.upper());
    }
}

#[test]
fn attention_region_signed_error_only_rows_and_exact_k_inverse_match_original_oracle() {
    let graph = graph(true, false);
    let input = box_for(&[1, 2, 2], 0.03125);
    let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let mut incoming = LinearBounds::new(
        arr2(&[[-1.0, 0.5, 0.25, -0.75], [0.0, 0.0, 0.0, 0.0]]),
        arr1(&[-0.3, 0.2]),
        arr2(&[[-1.0, 0.5, 0.25, -0.75], [0.0, 0.0, 0.0, 0.0]]),
        arr1(&[0.4, 0.7]),
    )
    .unwrap();
    incoming.set_coeff_err(
        arr2(&[[0.0, 0.0, 0.0, 0.0], [0.125, 0.0, 0.0, 0.0]]),
        arr2(&[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.25]]),
    );
    let mut query = AttentionRegionQuery::with_limits(None, MAX_BYTES, MAX_WORK);
    let candidate = query
        .candidate(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &incoming,
            &input,
            &boxes,
            0,
            None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(candidate.sources, ["q", "kt", NETWORK_INPUT]);
    let lo = input.lower().as_slice().unwrap();
    let hi = input.upper().as_slice().unwrap();
    for mask in 0..16 {
        let x: [f32; 4] = std::array::from_fn(|i| if mask & (1 << i) == 0 { lo[i] } else { hi[i] });
        let y = original_uniform(&x, false);
        let source_points = [[0.0; 4], [x[0], x[2], x[1], x[3]], x];
        for row in 0..2 {
            let nominal = (0..4).fold(BigRational::zero(), |sum, i| {
                sum + rat(incoming.lower_a()[[row, i]]) * &y[i]
            });
            let lower_error = (0..4).fold(BigRational::zero(), |sum, i| {
                sum + rat(incoming.lower_a_err().unwrap()[[row, i]]) * y[i].abs()
            });
            let upper_error = (0..4).fold(BigRational::zero(), |sum, i| {
                sum + rat(incoming.upper_a_err().unwrap()[[row, i]]) * y[i].abs()
            });
            let mut actual_lower = rat(candidate.lower_bias[row]);
            let mut actual_upper = rat(candidate.upper_bias[row]);
            for (bounds, point) in candidate.bounds.iter().zip(&source_points) {
                for i in 0..4 {
                    actual_lower += rat(bounds.lower_a()[[row, i]]) * rat(point[i]);
                    actual_upper += rat(bounds.upper_a()[[row, i]]) * rat(point[i]);
                }
            }
            assert!(actual_lower <= &nominal + rat(incoming.lower_b()[row]) - lower_error);
            assert!(actual_upper >= nominal + rat(incoming.upper_b()[row]) + upper_error);
        }
    }
    assert_eq!(query.receipt.checked, 1);
    assert!(query.work_receipt().work_reserved > RECOGNITION_WORK);
}

#[test]
fn attention_region_scalar_named_alias_sources_bias_once_and_baseline_control() {
    let graph = graph(true, false);
    let input = box_for(&[1, 2, 2], 0.03125);
    let result = graph.propagate_crown_fixed_slope(&input).unwrap();
    assert_exact_containment(&result, &input, false);
    let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let incoming = LinearBounds::new(
        arr2(&[[1.0, -0.5, 0.25, -1.0]]),
        arr1(&[0.125]),
        arr2(&[[1.0, -0.5, 0.25, -1.0]]),
        arr1(&[0.125]),
    )
    .unwrap();
    let mut query = AttentionRegionQuery::with_limits(None, MAX_BYTES, MAX_WORK);
    let mut accumulator = CrownMergeAccumulator::new();
    let mut flag = false;
    assert!(query
        .try_scalar(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &incoming,
            &input,
            &boxes,
            &mut accumulator,
            1,
            input.len(),
            &mut flag,
            None
        )
        .unwrap());
    assert!(flag);
    assert!(
        accumulator.contains_key("q")
            && accumulator.contains_key("kt")
            && accumulator.contains_key(NETWORK_INPUT)
    );
    assert_eq!(query.receipt.scalar, 1);
    assert_eq!(query.receipt.composed, 1);
}

#[test]
fn attention_region_semantic_and_capacity_declines_leave_original_accumulator_untouched() {
    let input = box_for(&[1, 2, 2], 0.03125);
    for mutation in 0..5 {
        let mut graph = graph(true, false);
        match mutation {
            0 => {
                if let Layer::Softmax(sm) = &mut graph.nodes.get_mut("prob").unwrap().layer {
                    sm.axis = 1;
                }
            }
            1 => {
                if let Layer::MulConstant(mul) = &mut graph.nodes.get_mut("scale").unwrap().layer {
                    mul.constant = ArrayD::from_elem(IxDyn(&[2]), 0.25);
                }
            }
            2 => {
                if let Layer::BilinearCrown(product) = &mut graph.nodes.get_mut("qk").unwrap().layer
                {
                    product.scale = Some(1.0);
                }
            }
            3 => {
                if let Layer::Transpose(transpose) = &mut graph.nodes.get_mut("kt").unwrap().layer {
                    transpose.axes = vec![0, 0, 2];
                }
            }
            _ => {}
        }
        // Collect valid boxes from the unmodified original, then independently
        // poison topology. The private recognizer must bind BOTH, not guess.
        let original = self::graph(true, false);
        let boxes = original
            .collect_node_bounds_with_engine(&input, None)
            .unwrap();
        let incoming = LinearBounds::identity(4);
        let mut query = AttentionRegionQuery::with_limits(
            None,
            if mutation == 4 {
                HEADER_BYTES + 1
            } else {
                MAX_BYTES
            },
            MAX_WORK,
        );
        let mut accumulator = CrownMergeAccumulator::new();
        let mut flag = false;
        assert!(!query
            .try_scalar(
                &graph,
                graph.nodes.get("pv").unwrap(),
                &incoming,
                &input,
                &boxes,
                &mut accumulator,
                4,
                input.len(),
                &mut flag,
                None
            )
            .unwrap());
        assert_eq!(accumulator.logical_frontier_payload_bytes(), 0);
        assert!(!flag);
        assert_eq!(query.receipt.composed, 0);
    }
}

#[test]
fn attention_region_batched_rejects_local_head_rows_and_retains_every_error_cell() {
    let budget = WorkBudget::new(None, MAX_BYTES, MAX_BYTES, MAX_WORK, 64, Duration::ZERO).unwrap();
    let local = BatchedLinearBounds::identity(&[3, 2, 2]).unwrap();
    assert!(consume::flat_incoming(&local, 12, None, &budget).is_err());
    let mut global = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 4]), vec![1.0; 8]).unwrap(),
        ArrayD::zeros(IxDyn(&[1, 2])),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 4]), vec![-0.5; 8]).unwrap(),
        ArrayD::zeros(IxDyn(&[1, 2])),
        vec![1, 2, 2],
        vec![1, 2],
    )
    .unwrap();
    global.set_coeff_err(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 4]),
            (1..=8).map(|i| i as f32 * 0.015625).collect(),
        )
        .unwrap(),
        ArrayD::from_elem(IxDyn(&[1, 2, 4]), 0.125),
    );
    let flat = consume::flat_incoming(&global, 4, None, &budget).unwrap();
    assert_eq!(flat.num_outputs(), 2);
    assert_eq!(
        flat.lower_a_err().unwrap().as_slice().unwrap(),
        global.lower_a_err.as_ref().unwrap().as_slice().unwrap()
    );
    assert_eq!(
        flat.upper_a_err().unwrap().as_slice().unwrap(),
        global.upper_a_err.as_ref().unwrap().as_slice().unwrap()
    );
}

#[test]
fn attention_region_pinned_vit_producer_metadata_includes_cls_and_authored_head_permutation() {
    // Original commit and decompressed publisher hashes are fixture identity,
    // not a claim that this synthetic graph has run an official property.
    const OFFICIAL_COMMIT: &str = "303f2327a8e8411efc4457da03d2016ce42733cf";
    const ORIGINAL_MODELS: [(usize, usize, &str); 2] = [
        (
            17,
            3,
            "9cc53b9edb35d40d70de1d816008a434c7e60b0404cf3cb38b9aefc189692883",
        ),
        (
            5,
            2,
            "246326387574617a274b6f73f7f52771df1195e03c99065ce4ad18c46b8e0984",
        ),
    ];
    assert_eq!(OFFICIAL_COMMIT.len(), 40);
    for (seq, blocks, hash) in ORIGINAL_MODELS {
        assert_eq!(hash.len(), 64);
        assert!(blocks >= 2);
        let input = box_for(&[seq, 3, 16], 0.001);
        let mut graph = GraphNetwork::new();
        for (name, perm) in [
            ("q", vec![1, 0, 2]),
            ("kt", vec![1, 2, 0]),
            ("v", vec![1, 0, 2]),
        ] {
            let mut transpose = TransposeLayer::new(perm);
            transpose.set_input_shape(input.shape().to_vec());
            graph.add_node(GraphNode::from_input(name, Layer::Transpose(transpose)));
        }
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
            "v",
        ));
        graph.set_output("pv");
        let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
        let region = recognize::recognize(&graph, graph.nodes.get("pv").unwrap(), &input, &boxes)
            .unwrap()
            .unwrap();
        assert_eq!(region.q.shape(), &[3, seq, 16]);
        assert_eq!(region.k.shape(), &[3, 16, seq]);
        assert_eq!(region.v.shape(), &[3, seq, 16]);
        assert_eq!(region.scale.to_bits(), 0.25f32.to_bits());
        assert!(region.authored_kt);
        assert!(graph
            .nodes
            .values()
            .all(|n| !matches!(n.layer, Layer::SelfAttention(_))));
    }
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn attention_region_nested_fallback_and_worker_context_never_refill_allowance() {
    let authority = AttentionRegionAuthority {
        budget: Some(
            WorkBudget::new(
                None,
                MAX_BYTES,
                MAX_BYTES,
                3 * RECOGNITION_WORK,
                64,
                Duration::ZERO,
            )
            .unwrap(),
        ),
        cap: MAX_BYTES,
        deadline: None,
    };
    let budget = authority.budget.as_ref().unwrap().clone();
    {
        let _outer = AttentionRegionScope::install(authority.clone());
        let _parent_query = AttentionRegionQuery::new(None);
        budget
            .lease(WorkRequest {
                work: RECOGNITION_WORK,
                ..WorkRequest::default()
            })
            .unwrap();
        {
            let nested = AttentionRegionQuery::new(None);
            assert!(nested.budget.as_ref().unwrap().shares_ledger(&budget));
            nested
                .budget
                .as_ref()
                .unwrap()
                .lease(WorkRequest {
                    work: RECOGNITION_WORK,
                    ..WorkRequest::default()
                })
                .unwrap();
        }
        // Explicit worker propagation: TLS cannot inherit thread creation.
        let worker_authority = authority.clone();
        std::thread::spawn(move || {
            let _scope = AttentionRegionScope::install(worker_authority);
            let worker = AttentionRegionQuery::new(None);
            worker
                .budget
                .as_ref()
                .unwrap()
                .lease(WorkRequest {
                    work: RECOGNITION_WORK,
                    ..WorkRequest::default()
                })
                .unwrap();
            assert!(worker
                .budget
                .as_ref()
                .unwrap()
                .lease(WorkRequest {
                    work: 1,
                    ..WorkRequest::default()
                })
                .is_err());
        })
        .join()
        .unwrap();
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            3 * RECOGNITION_WORK
        );
        let fallback = AttentionRegionQuery::new(None);
        assert!(fallback
            .budget
            .as_ref()
            .unwrap()
            .lease(WorkRequest {
                work: 1,
                ..WorkRequest::default()
            })
            .is_err());
    }
    let next_query = AttentionRegionQuery::new(None);
    assert!(!next_query.budget.as_ref().unwrap().shares_ledger(&budget));
    assert_eq!(next_query.work_receipt().work_reserved, 0);
}

#[test]
fn attention_region_scope_unwind_restores_parent_and_earlier_deadline() {
    let outer = AttentionRegionScope::enter(None);
    let authority = outer.authority();
    let budget = authority.budget.as_ref().unwrap().clone();
    let _ = std::panic::catch_unwind(|| {
        let inner_authority = AttentionRegionAuthority {
            budget: Some(
                WorkBudget::new(None, MAX_BYTES, MAX_BYTES, 1, 1, Duration::ZERO).unwrap(),
            ),
            cap: MAX_BYTES,
            deadline: None,
        };
        let _inner = AttentionRegionScope::install(inner_authority);
        panic!("control unwind");
    });
    let inherited = AttentionRegionQuery::new(Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    ));
    assert!(inherited
        ._scope
        .as_ref()
        .unwrap()
        .authority()
        .budget
        .as_ref()
        .unwrap()
        .shares_ledger(&budget));
    assert!(
        inherited.budget.is_none(),
        "expired construction declines before control allocation"
    );
    assert!(poll(
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap()
        ),
        &budget
    )
    .is_err());
    assert!(
        budget.checkpoint().is_ok(),
        "an earlier node deadline cannot replace the original ledger deadline"
    );
    drop(inherited);
    drop(outer);
}

fn authored_heads(zero_q: bool) -> (GraphNetwork, ndarray::Array2<f32>) {
    // Original [tokens, heads, channels] -> authored Q/V head permutation and
    // K [heads, channels, tokens]. Tokens != channels and properties != heads.
    let mut graph = GraphNetwork::new();
    for (name, perm) in [
        ("qh", vec![1, 0, 2]),
        ("kt", vec![1, 2, 0]),
        ("v", vec![1, 0, 2]),
    ] {
        let mut transpose = TransposeLayer::new(perm);
        transpose.set_input_shape(vec![3, 3, 2]);
        graph.add_node(GraphNode::from_input(name, Layer::Transpose(transpose)));
    }
    graph.add_node(GraphNode::new(
        "q",
        Layer::MulConstant(MulConstantLayer::scalar(if zero_q { 0.0 } else { 1.0 })),
        vec!["qh".into()],
    ));
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
        "v",
    ));
    graph.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec!["pv".into()],
    ));
    let weights = ndarray::Array2::from_shape_fn((9, 18), |(r, c)| {
        ((r * 7 + c * 3) % 11) as f32 * 0.125 - 0.625
    });
    graph.add_node(GraphNode::new(
        "head",
        Layer::Linear(
            LinearLayer::new(
                weights.clone(),
                Some(Array1::from_shape_fn(9, |r| r as f32 * 0.03125 - 0.125)),
            )
            .unwrap(),
        ),
        vec!["flat".into()],
    ));
    graph.set_output("head");
    (graph, weights)
}

fn uniform_head_support(
    weights: &ndarray::Array2<f32>,
    input: &BoundedTensor,
    row: usize,
) -> (BigRational, BigRational) {
    let mut lower = rat(row as f32 * 0.03125 - 0.125);
    let mut upper = lower.clone();
    let third = BigRational::new(1.into(), 3.into());
    for token in 0..3 {
        for head in 0..3 {
            for channel in 0..2 {
                let coefficient = (0..3).fold(BigRational::zero(), |sum, q| {
                    sum + rat(weights[[row, head * 6 + q * 2 + channel]])
                }) * &third;
                let raw = token * 6 + head * 2 + channel;
                let lo = rat(input.lower().as_slice().unwrap()[raw]);
                let hi = rat(input.upper().as_slice().unwrap()[raw]);
                if coefficient.is_negative() {
                    lower += &coefficient * &hi;
                    upper += coefficient * lo;
                } else {
                    lower += &coefficient * &lo;
                    upper += coefficient * hi;
                }
            }
        }
    }
    (lower, upper)
}

#[test]
fn attention_region_authored_asymmetric_permutation_nine_signed_rows_three_heads() {
    let (graph, weights) = authored_heads(true);
    let input = box_for(&[3, 3, 2], 0.03125);
    let (result, receipt) = graph
        .propagate_crown_batched_with_attention_region_receipt(&input)
        .unwrap();
    assert_eq!((receipt.batched, receipt.composed, receipt.rows), (1, 1, 9));
    assert_eq!(receipt.authored_k_direct_regions, 1);
    assert_eq!(receipt.elided_k_endpoint_copy_elements, 36);
    assert_eq!(receipt.elided_k_coefficient_copy_elements, 324);
    for row in 0..9 {
        // Q=0 makes the ORIGINAL score/exp/divide/PV expression exactly the
        // uniform rational mean, including non-dyadic 1/3. This checks the
        // whole input box via its exact affine support, not sampled libm.
        let (lower, upper) = uniform_head_support(&weights, &input, row);
        assert!(rat(result.bounds.lower().as_slice().unwrap()[row]) <= lower);
        assert!(rat(result.bounds.upper().as_slice().unwrap()[row]) >= upper);
    }
    // Inspect the exact inverse K map with distinguishable signed/error cells.
    // A square-only fixture or Q=0 containment alone could hide a K swap bug.
    let budget = WorkBudget::new(None, MAX_BYTES, MAX_BYTES, MAX_WORK, 64, Duration::ZERO).unwrap();
    let k = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 3]), (0..18).map(|i| i as f32 - 8.0).collect())
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 3]), (0..18).map(|i| i as f32 - 7.5).collect())
            .unwrap(),
    )
    .unwrap();
    let canonical = consume::transpose_box(&k, None, &budget).unwrap();
    for head in 0..3 {
        for key in 0..3 {
            for ch in 0..2 {
                assert_eq!(
                    canonical.lower()[[head, key, ch]].to_bits(),
                    k.lower()[[head, ch, key]].to_bits()
                );
                assert_eq!(
                    canonical.upper()[[head, key, ch]].to_bits(),
                    k.upper()[[head, ch, key]].to_bits()
                );
            }
        }
    }
    let a = ndarray::Array2::from_shape_fn((9, 18), |(r, c)| (r * 18 + c) as f32 * 0.015625 - 1.0);
    let err = a.mapv(|x| x.abs() * 0.03125);
    let mut carrier =
        LinearBounds::new(a.clone(), Array1::zeros(9), -a.clone(), Array1::zeros(9)).unwrap();
    carrier.set_coeff_err(err.clone(), err.clone());
    consume::transpose_coefficients(&mut carrier, &[3, 2, 3], None, &budget).unwrap();
    for row in 0..9 {
        for head in 0..3 {
            for ch in 0..2 {
                for key in 0..3 {
                    let old = head * 6 + key * 2 + ch;
                    let new = head * 6 + ch * 3 + key;
                    assert_eq!(
                        carrier.lower_a()[[row, new]].to_bits(),
                        a[[row, old]].to_bits()
                    );
                    assert_eq!(
                        carrier.upper_a()[[row, new]].to_bits(),
                        (-a[[row, old]]).to_bits()
                    );
                    assert_eq!(
                        carrier.lower_a_err().unwrap()[[row, new]].to_bits(),
                        err[[row, old]].to_bits()
                    );
                    assert_eq!(
                        carrier.upper_a_err().unwrap()[[row, new]].to_bits(),
                        err[[row, old]].to_bits()
                    );
                }
            }
        }
    }
    // Nonzero Q and identical original K token rows produce exactly equal
    // original scores at a singleton, so uniformity is independently known.
    let (nonzero_graph, weights) = authored_heads(false);
    let center =
        ArrayD::from_shape_fn(IxDyn(&[3, 3, 2]), |i| (i[1] * 2 + i[2] + 1) as f32 * 0.0625);
    let point = BoundedTensor::new(center.clone(), center).unwrap();
    let (point_result, point_receipt) = nonzero_graph
        .propagate_crown_batched_with_attention_region_receipt(&point)
        .unwrap();
    assert_eq!((point_receipt.composed, point_receipt.rows), (1, 9));
    for row in 0..9 {
        let (exact, _) = uniform_head_support(&weights, &point, row);
        assert!(rat(point_result.bounds.lower().as_slice().unwrap()[row]) <= exact);
        assert!(rat(point_result.bounds.upper().as_slice().unwrap()[row]) >= exact);
    }
}

#[test]
fn attention_region_authored_kt_direct_original_candidate_matches_old_transposes_bit_exact() {
    let (graph, weights) = authored_heads(false);
    let input = box_for(&[3, 3, 2], 0.03125);
    let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let mut incoming = LinearBounds::new(
        weights.clone(),
        Array1::from_shape_fn(9, |r| r as f32 * 0.03125 - 0.125),
        -weights,
        Array1::from_shape_fn(9, |r| 0.25 - r as f32 * 0.015625),
    )
    .unwrap();
    incoming.set_coeff_err(
        ndarray::Array2::from_shape_fn((9, 18), |(r, c)| ((r + c) % 3) as f32 * 0.00390625),
        ndarray::Array2::from_shape_fn((9, 18), |(r, c)| ((r + 2 * c) % 5) as f32 * 0.001953125),
    );
    let region = recognize::recognize(&graph, graph.nodes.get("pv").unwrap(), &input, &boxes)
        .unwrap()
        .unwrap();
    assert!(region.authored_kt);
    assert_eq!(region.k.shape(), &[3, 2, 3]);
    let expected_request = region
        .request(9, input.len(), incoming.memory_bytes())
        .unwrap();
    let count = (region.q.len() + region.k.len() + region.v.len() + input.len()) * 9;
    assert_eq!(
        expected_request.scratch_bytes,
        count * 64 + incoming.memory_bytes() * 4 + HEADER_BYTES
    );
    let control_budget =
        WorkBudget::new(None, MAX_BYTES, MAX_BYTES, MAX_WORK, 64, Duration::ZERO).unwrap();
    let canonical_k = consume::transpose_box(region.k, None, &control_budget).unwrap();
    let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(region.scale));
    let (old_outputs, old_lower, old_upper) = layer
        .propagate_crown_ternary_with_budget_and_deadline(
            &incoming,
            region.q,
            &canonical_k,
            region.v,
            &control_budget,
            None,
        )
        .unwrap();
    let mut old: Vec<_> = old_outputs.into_iter().map(|b| b.unwrap()).collect();
    consume::transpose_coefficients(&mut old[1], region.k.shape(), None, &control_budget).unwrap();
    let mut query = AttentionRegionQuery::with_limits(None, MAX_BYTES, MAX_WORK);
    let direct = query
        .candidate(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &incoming,
            &input,
            &boxes,
            0,
            None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(direct.sources, ["q", "kt", "v"]);
    let bits = |a: &[f32]| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(direct.lower_bias.as_slice().unwrap()),
        bits(old_lower.as_slice().unwrap())
    );
    assert_eq!(
        bits(direct.upper_bias.as_slice().unwrap()),
        bits(old_upper.as_slice().unwrap())
    );
    for (old, new) in old.iter().zip(&direct.bounds) {
        assert_eq!(
            bits(old.lower_a().as_slice().unwrap()),
            bits(new.lower_a().as_slice().unwrap())
        );
        assert_eq!(
            bits(old.upper_a().as_slice().unwrap()),
            bits(new.upper_a().as_slice().unwrap())
        );
        assert_eq!(
            bits(old.lower_b().as_slice().unwrap()),
            bits(new.lower_b().as_slice().unwrap())
        );
        assert_eq!(
            bits(old.upper_b().as_slice().unwrap()),
            bits(new.upper_b().as_slice().unwrap())
        );
        assert_eq!(old.lower_a_err(), new.lower_a_err());
        assert_eq!(old.upper_a_err(), new.upper_a_err());
    }
    assert!(direct.bounds[1].lower_a().iter().any(|&a| a != 0.0));
    let receipt = query.receipt();
    assert_eq!(receipt.authored_k_direct_regions, 1);
    assert_eq!(receipt.elided_k_endpoint_copy_elements, 36);
    assert_eq!(receipt.elided_k_coefficient_copy_elements, 324);
    // Returned carriers stay on the ORIGINAL query ledger until final drop.
    let budget = query.budget.as_ref().unwrap().clone();
    let before_drop = budget.receipt().unwrap();
    assert!(before_drop.live_persistent_bytes >= expected_request.persistent_bytes);
    drop(direct);
    drop(query);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        before_drop.work_reserved
    );
}

#[test]
fn attention_region_public_memory_fallback_and_expired_node_preserve_original_ledger() {
    let graph = graph(true, false);
    let input = box_for(&[1, 2, 2], 0.03125);
    let budget = WorkBudget::new(
        None,
        MAX_BYTES,
        MAX_BYTES,
        RECOGNITION_WORK,
        64,
        Duration::ZERO,
    )
    .unwrap();
    let authority = AttentionRegionAuthority {
        budget: Some(budget.clone()),
        cap: MAX_BYTES,
        deadline: None,
    };
    let _scope = AttentionRegionScope::install(authority);
    budget
        .lease(WorkRequest {
            work: RECOGNITION_WORK,
            ..WorkRequest::default()
        })
        .unwrap();
    crate::tests::with_crown_dense_budget_mb("0", || {
        let (result, receipt) = graph
            .propagate_crown_batched_with_attention_region_receipt(&input)
            .unwrap();
        assert!(matches!(
            result.provenance,
            crate::BoundsProvenance::ForwardFallback(
                crate::CrownIbpFallbackReason::MemoryBudgetExceeded
            )
        ));
        assert_eq!(receipt.composed, 0);
        let after = AttentionRegionScope::enter(None);
        let inherited = after.authority().budget.unwrap();
        assert!(inherited.shares_ledger(&budget));
        assert!(inherited
            .lease(WorkRequest {
                work: 1,
                ..WorkRequest::default()
            })
            .is_err());
    });
    let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let mut query = AttentionRegionQuery::new(Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    ));
    let mut accumulator = CrownMergeAccumulator::new();
    let mut flag = false;
    assert!(!query
        .try_scalar(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &LinearBounds::identity(4),
            &input,
            &boxes,
            &mut accumulator,
            4,
            input.len(),
            &mut flag,
            None
        )
        .unwrap());
    assert_eq!(accumulator.logical_frontier_payload_bytes(), 0);
    assert!(!flag);
    assert!(budget.checkpoint().is_ok());
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn attention_region_worker_handoff_retains_earlier_nested_deadline() {
    let budget = WorkBudget::new(None, MAX_BYTES, MAX_BYTES, MAX_WORK, 64, Duration::ZERO).unwrap();
    let outer_authority = AttentionRegionAuthority {
        budget: Some(budget.clone()),
        cap: MAX_BYTES,
        deadline: None,
    };
    let _outer = AttentionRegionScope::install(outer_authority);
    let _inner = AttentionRegionScope::enter(Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    ));
    let worker_authority = _inner.authority();
    let same = budget.clone();
    std::thread::spawn(move || {
        let _worker = AttentionRegionScope::install(worker_authority);
        let query = AttentionRegionQuery::new(None);
        assert!(query
            ._scope
            .as_ref()
            .unwrap()
            .authority()
            .budget
            .as_ref()
            .unwrap()
            .shares_ledger(&same));
        assert!(query.budget.is_none());
        assert!(poll(query.deadline, &same).is_err());
        assert!(
            same.checkpoint().is_ok(),
            "deadline restriction does not mint or replace the original ledger"
        );
    })
    .join()
    .unwrap();
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep the original fixture owner intact while mutating an independent candidate."
)]
fn attention_region_batched_failed_topology_lookups_share_the_attempt_ceiling() {
    let original = graph(true, false);
    let input = box_for(&[1, 2, 2], 0.03125);
    let boxes = original
        .collect_node_bounds_with_engine(&input, None)
        .unwrap();
    let mut graph = original.clone();
    if let Layer::Softmax(sm) = &mut graph.nodes.get_mut("prob").unwrap().layer {
        sm.axis = 1;
    }
    let mut query = AttentionRegionQuery::with_limits(None, MAX_BYTES, MAX_WORK);
    let incoming = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![1.0; 4]).unwrap(),
        ArrayD::zeros(IxDyn(&[1, 1])),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![1.0; 4]).unwrap(),
        ArrayD::zeros(IxDyn(&[1, 1])),
        vec![1, 2, 2],
        vec![1, 1],
    )
    .unwrap();
    let mut accumulator = BatchedCrownAccumulator::new(graph.dispatch_plan().unwrap());
    for _ in 0..MAX_ATTEMPTS + 5 {
        assert!(!query
            .try_batched(
                &graph,
                graph.nodes.get("pv").unwrap(),
                &incoming,
                &input,
                &boxes,
                &mut accumulator,
                None
            )
            .unwrap());
    }
    assert_eq!(query.receipt.attempted, MAX_ATTEMPTS);
    assert_eq!(query.receipt.declined, MAX_ATTEMPTS + 5);
    assert_eq!(query.receipt.composed, 0);
    assert!(accumulator.is_empty());
    let expected = MAX_ATTEMPTS as u64 * (RECOGNITION_WORK + incoming.memory_bytes() as u64 * 8);
    assert_eq!(
        query.work_receipt().work_reserved,
        expected,
        "no unrecorded lookup or refreshed recognition allowance"
    );
}

#[test]
fn attention_region_batched_nine_signed_error_rows_match_scalar_carriers() {
    let (graph, mut weights) = authored_heads(true);
    let input = box_for(&[3, 3, 2], 0.03125);
    weights.row_mut(8).fill(0.0); // a nonzero-error-only property direction
    let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let bias = Array1::from_shape_fn(9, |r| r as f32 * 0.03125 - 0.125);
    let error = ndarray::Array2::from_shape_fn((9, 18), |(r, c)| ((r + c) % 5) as f32 * 0.015625);
    let mut incoming =
        LinearBounds::new(weights.clone(), bias.clone(), weights.clone(), bias).unwrap();
    incoming.set_coeff_err(error.clone(), error.mapv(|x| x * 2.0));
    let mut scalar = AttentionRegionQuery::with_limits(None, MAX_BYTES, MAX_WORK);
    let candidate = scalar
        .candidate(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &incoming,
            &input,
            &boxes,
            0,
            None,
        )
        .unwrap()
        .unwrap();
    let convert = |array: &ndarray::Array2<f32>| {
        array
            .clone()
            .into_shape_with_order(IxDyn(&[1, 9, 18]))
            .unwrap()
    };
    let mut batched = BatchedLinearBounds::new(
        convert(incoming.lower_a()),
        incoming
            .lower_b()
            .clone()
            .into_shape_with_order(IxDyn(&[1, 9]))
            .unwrap(),
        convert(incoming.upper_a()),
        incoming
            .upper_b()
            .clone()
            .into_shape_with_order(IxDyn(&[1, 9]))
            .unwrap(),
        vec![3, 3, 2],
        vec![1, 9],
    )
    .unwrap();
    batched.set_coeff_err(
        convert(incoming.lower_a_err().unwrap()),
        convert(incoming.upper_a_err().unwrap()),
    );
    let mut query = AttentionRegionQuery::with_limits(None, MAX_BYTES, MAX_WORK);
    let mut accumulator = BatchedCrownAccumulator::new(graph.dispatch_plan().unwrap());
    assert!(query
        .try_batched(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &batched,
            &input,
            &boxes,
            &mut accumulator,
            None
        )
        .unwrap());
    assert_eq!((query.receipt.batched, query.receipt.rows), (1, 9));
    for (source, expected) in candidate.sources.into_iter().zip(candidate.bounds) {
        let actual = accumulator
            .take(source)
            .unwrap()
            .into_batched_dense_checked("attention region signed parity test")
            .unwrap();
        for (a, b) in actual.lower_a().iter().zip(expected.lower_a()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in actual.upper_a().iter().zip(expected.upper_a()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert!(actual
            .lower_b()
            .iter()
            .chain(actual.upper_b())
            .all(|x| *x == 0.0));
        assert_eq!(actual.input_shape, boxes.get(source).unwrap().shape());
        assert_eq!(actual.output_shape, vec![1, 9]);
        assert_eq!(
            actual.lower_a_err.is_some(),
            expected.lower_a_err().is_some()
        );
        assert_eq!(
            actual.upper_a_err.is_some(),
            expected.upper_a_err().is_some()
        );
        if let (Some(a), Some(b)) = (&actual.lower_a_err, expected.lower_a_err()) {
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        if let (Some(a), Some(b)) = (&actual.upper_a_err, expected.upper_a_err()) {
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
    let shared = accumulator
        .take(NETWORK_INPUT)
        .unwrap()
        .into_batched_dense_checked("attention region bias parity test")
        .unwrap();
    for (a, b) in shared.lower_b().iter().zip(candidate.lower_bias) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in shared.upper_b().iter().zip(candidate.upper_bias) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(shared
        .lower_a()
        .iter()
        .chain(shared.upper_a())
        .all(|x| *x == 0.0));
    assert!(
        accumulator.is_empty(),
        "exactly3 original sources and one shared bias must be published"
    );
}

fn validation_carrier(rows: usize, columns: usize) -> BatchedLinearBounds {
    let shape = IxDyn(&[1, rows, columns]);
    let matrix = |offset: usize| {
        ArrayD::from_shape_vec(
            shape.clone(),
            (0..rows * columns)
                .map(|i| ((i + offset) % 17) as f32 * 0.03125 - 0.25)
                .collect(),
        )
        .unwrap()
    };
    let mut result = BatchedLinearBounds::new(
        matrix(0),
        ArrayD::from_elem(IxDyn(&[1, rows]), -0.125),
        matrix(3),
        ArrayD::from_elem(IxDyn(&[1, rows]), 0.25),
        vec![1, columns],
        vec![1, rows],
    )
    .unwrap();
    result.lower_a_err = Some(
        ArrayD::from_shape_vec(
            shape.clone(),
            (0..rows * columns)
                .map(|i| (i % 7) as f32 * 0.015625)
                .collect(),
        )
        .unwrap(),
    );
    result.upper_a_err = Some(
        ArrayD::from_shape_vec(
            shape,
            (0..rows * columns)
                .map(|i| (i % 11) as f32 * 0.0078125)
                .collect(),
        )
        .unwrap(),
    );
    result
}

fn validation_budget() -> (
    WorkBudget,
    crate::certify::residency::CancellationToken,
    crate::certify::residency::MemoryAuthority,
) {
    use crate::certify::residency::{CancellationToken, MemoryAuthority};
    use crate::certify::work::{WorkLimits, WorkReserve};
    let cancellation = CancellationToken::default();
    let memory = MemoryAuthority::new(MAX_BYTES);
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: MAX_BYTES,
            persistent_bytes: MAX_BYTES,
            max_work: MAX_WORK,
            max_proposals: 64,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, cancellation, memory)
}

#[test]
fn attention_region_final_carrier_validation_cancellation_leaves_pending_paths_untouched() {
    let graph = graph(true, true);
    let input = box_for(&[1, 2, 2], 0.03125);
    let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let incoming = validation_carrier(2, 4);
    let (budget, cancellation, memory) = validation_budget();
    let mut query = AttentionRegionQuery::from_authority(
        AttentionRegionAuthority {
            budget: Some(budget.clone()),
            cap: MAX_BYTES,
            deadline: None,
        },
        None,
        None,
    );
    let mut accumulator = BatchedCrownAccumulator::new(graph.dispatch_plan().unwrap());
    let pending = validation_carrier(2, 4);
    accumulator.accumulate_dense("q", pending.clone()).unwrap();
    let pending_bytes = accumulator.total_memory_bytes();
    let before = budget.receipt().unwrap();
    let (result, inspected) = consume::cancel_during_validation(cancellation.clone(), 1, || {
        query.try_batched(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &incoming,
            &input,
            &boxes,
            &mut accumulator,
            None,
        )
    });
    // Recognition and the full incoming/error copy completed before the first
    // real final numeric scan cancelled the ORIGINAL ledger. No contribution,
    // bias or error plane was published to the pending accumulator.
    assert!(!result.unwrap());
    assert_eq!(inspected, 8);
    assert!(cancellation.is_cancelled());
    assert_eq!(accumulator.total_memory_bytes(), pending_bytes);
    let untouched = accumulator
        .take("q")
        .unwrap()
        .into_batched_dense_checked("cancelled attention region pending control")
        .unwrap();
    for (actual, original) in [
        (&untouched.lower_a, &pending.lower_a),
        (&untouched.upper_a, &pending.upper_a),
        (&untouched.lower_b, &pending.lower_b),
        (&untouched.upper_b, &pending.upper_b),
        (
            untouched.lower_a_err.as_ref().unwrap(),
            pending.lower_a_err.as_ref().unwrap(),
        ),
        (
            untouched.upper_a_err.as_ref().unwrap(),
            pending.upper_a_err.as_ref().unwrap(),
        ),
    ] {
        assert_eq!(actual.shape(), original.shape());
        for (&actual, &original) in actual.iter().zip(original) {
            assert_eq!(actual.to_bits(), original.to_bits());
        }
    }
    assert!(accumulator.is_empty());
    assert_eq!(query.receipt.recognized, 1);
    assert_eq!(query.receipt.composed, 0);
    assert_eq!(query.receipt.declined, 1);
    let after = budget.receipt().unwrap();
    assert_eq!(
        after.work_reserved - before.work_reserved,
        incoming.memory_bytes() as u64 * 8 + RECOGNITION_WORK,
    );
    assert_eq!(after.live_scratch_bytes, 0);
    assert_eq!(after.live_persistent_bytes, HEADER_BYTES);
    assert_eq!(after.completed, before.completed);
    drop(query);
    let memory = memory.receipt().unwrap();
    assert_eq!(
        memory.live_pending_bytes + memory.live_materialized_bytes,
        0
    );
}

#[test]
fn attention_region_final_carrier_validation_is_bounded_and_preserves_numeric_firewall() {
    let incoming = validation_carrier(2, 2048);
    let (budget, cancellation, _) = validation_budget();
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: incoming.memory_bytes() * 4,
            work: incoming.memory_bytes() as u64 * 8,
            ..WorkRequest::default()
        })
        .unwrap();
    let (result, inspected) = consume::cancel_during_validation(cancellation, 1024, || {
        consume::flat_incoming(&incoming, 2048, None, &budget)
    });
    assert!(
        matches!(result, Err(NyError::UnsupportedOp(ref message)) if message.contains("Cancelled"))
    );
    assert_eq!(inspected, 1024, "stop within one final-validation chunk");
    assert!(lease.complete().is_err());
    drop(lease);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);

    let (budget, _, _) = validation_budget();
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: incoming.memory_bytes() * 4,
            work: incoming.memory_bytes() as u64 * 8,
            ..WorkRequest::default()
        })
        .unwrap();
    let flat = consume::flat_incoming(&incoming, 2048, None, &budget).unwrap();
    let restored = consume::batched_source(flat, &[1, 32, 64], &incoming, None, &budget).unwrap();
    for (actual, original) in [
        (&restored.lower_a, &incoming.lower_a),
        (&restored.upper_a, &incoming.upper_a),
        (&restored.lower_b, &incoming.lower_b),
        (&restored.upper_b, &incoming.upper_b),
        (
            restored.lower_a_err.as_ref().unwrap(),
            incoming.lower_a_err.as_ref().unwrap(),
        ),
        (
            restored.upper_a_err.as_ref().unwrap(),
            incoming.upper_a_err.as_ref().unwrap(),
        ),
    ] {
        assert_eq!(actual.shape(), original.shape());
        for (&actual, &original) in actual.iter().zip(original) {
            assert_eq!(actual.to_bits(), original.to_bits());
        }
    }
    assert_eq!(restored.input_shape, vec![1, 32, 64]);
    assert_eq!(restored.output_shape, incoming.output_shape);
    for poison in 0..6 {
        let mut malformed = incoming.clone();
        match poison {
            0 => malformed.lower_a.as_slice_mut().unwrap()[2049] = f32::INFINITY,
            1 => malformed.upper_b.as_slice_mut().unwrap()[1] = f32::NAN,
            2 => {
                malformed
                    .lower_a_err
                    .as_mut()
                    .unwrap()
                    .as_slice_mut()
                    .unwrap()[2049] = f32::from_bits(0x8000_0001)
            }
            3 => malformed.upper_a_err = Some(ArrayD::zeros(IxDyn(&[1, 1, 2048]))),
            4 => malformed.upper_a = ArrayD::zeros(IxDyn(&[1, 2, 1024])),
            _ => malformed.lower_a = ArrayD::zeros(IxDyn(&[4096])),
        }
        assert!(consume::flat_incoming(&malformed, 2048, None, &budget).is_err());
    }
    // Biases and conservative errors retain their previous non-NaN profile;
    // signed zero is accepted, without sanitizing a negative subnormal.
    let mut conservative = incoming.clone();
    conservative.lower_b.as_slice_mut().unwrap()[0] = f32::NEG_INFINITY;
    conservative
        .upper_a_err
        .as_mut()
        .unwrap()
        .as_slice_mut()
        .unwrap()[0] = f32::INFINITY;
    conservative
        .lower_a_err
        .as_mut()
        .unwrap()
        .as_slice_mut()
        .unwrap()[0] = -0.0;
    let flat = consume::flat_incoming(&conservative, 2048, None, &budget).unwrap();
    assert_eq!(
        flat.lower_a_err.as_ref().unwrap()[(0, 0)].to_bits(),
        (-0.0f32).to_bits()
    );
    let mut invalid = flat;
    invalid.upper_a_err.as_mut().unwrap()[(1, 1)] = f32::from_bits(0x8000_0001);
    assert!(consume::batched_source(invalid, &[1, 32, 64], &incoming, None, &budget).is_err());
    lease.complete().unwrap();
}

#[test]
fn attention_region_final_box_validation_is_polled_and_retains_finite_profile() {
    let original = box_for(&[1, 32, 64], 0.03125);
    let (budget, cancellation, _) = validation_budget();
    let lease = budget
        .lease(WorkRequest {
            scratch_bytes: original.len() * 32,
            work: original.len() as u64 * 64,
            ..WorkRequest::default()
        })
        .unwrap();
    let (result, inspected) = consume::cancel_during_validation(cancellation, 1024, || {
        consume::transpose_box(&original, None, &budget)
    });
    assert!(
        matches!(result, Err(NyError::UnsupportedOp(ref message)) if message.contains("Cancelled"))
    );
    assert_eq!(inspected, 1024);
    drop(lease);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);

    let (budget, _, _) = validation_budget();
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: original.len() * 32,
            work: original.len() as u64 * 64,
            ..WorkRequest::default()
        })
        .unwrap();
    let transposed = consume::transpose_box(&original, None, &budget).unwrap();
    assert_eq!(transposed.shape(), &[1, 64, 32]);
    for (actual, expected) in [
        (transposed.lower(), original.lower()),
        (transposed.upper(), original.upper()),
    ] {
        for key in 0..64 {
            for channel in 0..32 {
                assert_eq!(
                    actual[[0, key, channel]].to_bits(),
                    expected[[0, channel, key]].to_bits(),
                );
            }
        }
    }
    let nonfinite = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[1, 32, 64]), f32::NEG_INFINITY),
        original.upper().clone(),
    )
    .unwrap();
    assert!(consume::transpose_box(&nonfinite, None, &budget).is_err());
    lease.complete().unwrap();
}

fn scalar_pending_control(accumulator: &mut CrownMergeAccumulator) -> LinearBounds {
    let mut pending = LinearBounds::identity(4);
    pending.set_coeff_err(
        ndarray::Array2::from_elem((4, 4), 0.015625),
        ndarray::Array2::from_elem((4, 4), 0.03125),
    );
    accumulator.insert(
        "q".into(),
        crate::bounds::patches::CrownBounds::Dense(pending.clone()),
    );
    pending
}

fn assert_scalar_pending_untouched(
    accumulator: &mut CrownMergeAccumulator,
    expected: &LinearBounds,
) {
    let Some(crate::bounds::patches::CrownBounds::Dense(actual)) = accumulator.take("q").unwrap()
    else {
        panic!("scalar attention refusal changed the original pending carrier");
    };
    for (actual, expected) in [
        (
            actual.lower_a().as_slice().unwrap(),
            expected.lower_a().as_slice().unwrap(),
        ),
        (
            actual.upper_a().as_slice().unwrap(),
            expected.upper_a().as_slice().unwrap(),
        ),
        (
            actual.lower_b().as_slice().unwrap(),
            expected.lower_b().as_slice().unwrap(),
        ),
        (
            actual.upper_b().as_slice().unwrap(),
            expected.upper_b().as_slice().unwrap(),
        ),
        (
            actual.lower_a_err().unwrap().as_slice().unwrap(),
            expected.lower_a_err().unwrap().as_slice().unwrap(),
        ),
        (
            actual.upper_a_err().unwrap().as_slice().unwrap(),
            expected.upper_a_err().unwrap().as_slice().unwrap(),
        ),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    assert_eq!(accumulator.logical_frontier_payload_bytes(), 0);
}

#[test]
fn attention_region_scalar_bias_firewall_declines_nan_and_preserves_infinities() {
    let graph = graph(true, true);
    let input = box_for(&[1, 2, 2], 0.03125);
    let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    for lower in [true, false] {
        let mut incoming = LinearBounds::identity(4);
        if lower {
            incoming.lower_b_mut()[1] = f32::NAN;
        } else {
            incoming.upper_b_mut()[1] = f32::NAN;
        }
        let mut query = AttentionRegionQuery::with_limits(None, MAX_BYTES, MAX_WORK);
        let mut accumulator = CrownMergeAccumulator::new();
        let expected = scalar_pending_control(&mut accumulator);
        let before = query.work_receipt();
        let mut accumulated = false;
        assert!(!query
            .try_scalar(
                &graph,
                graph.nodes.get("pv").unwrap(),
                &incoming,
                &input,
                &boxes,
                &mut accumulator,
                4,
                input.len(),
                &mut accumulated,
                None,
            )
            .unwrap());
        assert!(!accumulated);
        assert_scalar_pending_untouched(&mut accumulator, &expected);
        assert_eq!(query.receipt.checked, 0);
        assert_eq!(query.receipt.composed, 0);
        let after = query.work_receipt();
        assert!(after.work_reserved > before.work_reserved + RECOGNITION_WORK);
        assert_eq!(after.live_scratch_bytes, 0);
        assert_eq!(after.live_persistent_bytes, HEADER_BYTES);
    }
    // The ordinary scalar profile permits conservative signed infinities.
    // They must remain untouched rather than rejected or sanitized to zero.
    let mut incoming = LinearBounds::identity(4);
    incoming.lower_b_mut().fill(f32::NEG_INFINITY);
    incoming.upper_b_mut().fill(f32::INFINITY);
    let mut query = AttentionRegionQuery::with_limits(None, MAX_BYTES, MAX_WORK);
    let candidate = query
        .candidate(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &incoming,
            &input,
            &boxes,
            0,
            None,
        )
        .unwrap()
        .expect("signed conservative biases retain the original scalar profile");
    assert!(candidate
        .lower_bias
        .iter()
        .all(|x| x.to_bits() == f32::NEG_INFINITY.to_bits()));
    assert!(candidate
        .upper_bias
        .iter()
        .all(|x| x.to_bits() == f32::INFINITY.to_bits()));
    assert_eq!(query.receipt.checked, 1);
}

#[test]
fn attention_region_scalar_bias_validation_cancellation_preserves_original_ledger() {
    let graph = graph(true, true);
    let input = box_for(&[1, 2, 2], 0.03125);
    let boxes = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let incoming = LinearBounds::identity(4);
    let (budget, cancellation, memory) = validation_budget();
    let mut query = AttentionRegionQuery::from_authority(
        AttentionRegionAuthority {
            budget: Some(budget.clone()),
            cap: MAX_BYTES,
            deadline: None,
        },
        None,
        None,
    );
    let mut accumulator = CrownMergeAccumulator::new();
    let expected = scalar_pending_control(&mut accumulator);
    let before = budget.receipt().unwrap();
    let mut accumulated = false;
    let (result, inspected) = consume::cancel_during_validation(cancellation.clone(), 1, || {
        query.try_scalar(
            &graph,
            graph.nodes.get("pv").unwrap(),
            &incoming,
            &input,
            &boxes,
            &mut accumulator,
            4,
            input.len(),
            &mut accumulated,
            None,
        )
    });
    assert!(!result.unwrap());
    assert_eq!(
        inspected, 4,
        "original bias cells trigger cancellation before K/kernel work"
    );
    assert!(cancellation.is_cancelled());
    assert!(!accumulated);
    assert_scalar_pending_untouched(&mut accumulator, &expected);
    assert_eq!(query.receipt.checked, 0);
    assert_eq!(query.receipt.composed, 0);
    let after = budget.receipt().unwrap();
    assert!(after.work_reserved > before.work_reserved + RECOGNITION_WORK);
    assert_eq!(after.live_scratch_bytes, 0);
    assert_eq!(after.live_persistent_bytes, HEADER_BYTES);
    assert_eq!(after.completed, before.completed);
    drop(query);
    let memory = memory.receipt().unwrap();
    assert_eq!(
        memory.live_pending_bytes + memory.live_materialized_bytes,
        0
    );
}
