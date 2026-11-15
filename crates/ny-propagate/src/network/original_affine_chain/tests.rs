// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use ndarray::{array, Array1, Array2};
use num_bigint::BigInt;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::scope::NumericalSemantics;
use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkReserve};
use crate::layers::{BoundPropagation, GELULayer, PointLinearBuildPolicy, ReLULayer};
use crate::{GraphNetwork, GraphNode, Network};

use super::*;

fn budget() -> WorkBudget {
    WorkBudget::new(None, 1 << 24, 1 << 26, 1 << 32, 4096, Duration::ZERO).unwrap()
}
fn options(budget: &WorkBudget) -> OriginalAffineOptions<'_> {
    OriginalAffineOptions {
        budget,
        class: LeaseClass::Optional,
        deadline: None,
    }
}
fn ids(index: usize) -> OriginalAffineIds {
    let digest = |prefix: &[u8], index: usize| Digest::of_bytes(prefix, &index.to_le_bytes());
    OriginalAffineIds {
        lowering: Digest::of_bytes(b"test-chain-lowering", b"ordered original chain"),
        node: digest(b"test-chain-node", index),
        input: digest(b"test-chain-edge", index),
        output: digest(b"test-chain-edge", index + 1),
    }
}
fn owner(
    weights: &Array2<f32>,
    bias: &Array1<f32>,
    ledger: &WorkBudget,
    deadline: Option<Instant>,
) -> Layer {
    Layer::PointLinear(
        PointLinearLayer::from_original_view(
            weights.view(),
            false,
            Some(bias.view().into_dyn()),
            PointLinearBuildPolicy {
                budget: ledger,
                deadline,
                max_resident_bytes: 1 << 26,
                max_scratch_bytes: 1 << 24,
                max_work: 1 << 32,
            },
        )
        .unwrap(),
    )
}
fn layers(ledger: &WorkBudget) -> Vec<Layer> {
    vec![
        owner(
            &array![[0.5, -1.0], [0.1, 0.3]],
            &array![0.125, -0.25],
            ledger,
            None,
        ),
        Layer::ReLU(ReLULayer::new()),
        owner(
            &array![[1.5, -0.5], [0.1, 0.3], [0.1, 0.3]],
            &array![-0.125, 0.25, 0.25],
            ledger,
            None,
        ),
    ]
}
fn nodes(layers: &[Layer]) -> Vec<OriginalChainNode<'_>> {
    layers
        .iter()
        .enumerate()
        .map(|(index, layer)| OriginalChainNode {
            ids: ids(index),
            layer,
        })
        .collect()
}
fn input() -> BoundedTensor {
    BoundedTensor::new(
        array![[-1.0, -0.5], [-0.25, 0.25]].into_dyn(),
        array![[1.0, 0.75], [0.5, 1.0]].into_dyn(),
    )
    .unwrap()
}
fn q(value: f32) -> Q {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    assert_ne!(exponent, 0xff);
    let fraction = bits & 0x7fffff;
    let (mantissa, power) = if exponent == 0 {
        (fraction, -149i32)
    } else {
        ((1 << 23) | fraction, exponent as i32 - 127 - 23)
    };
    let mut numerator = BigInt::from(mantissa);
    if bits >> 31 != 0 {
        numerator = -numerator;
    }
    if power >= 0 {
        Q::from_integer(numerator << power as usize)
    } else {
        Q::new(numerator, BigInt::from(1) << (-power) as usize)
    }
}
fn exact_forward(layers: &[Layer], input: &[Q], positions: usize) -> Vec<Q> {
    let mut values = input.to_vec();
    for layer in layers {
        match layer {
            Layer::PointLinear(owner) => {
                let mut next = Vec::new();
                for position in 0..positions {
                    for row in 0..owner.out_features() {
                        let mut value = owner.bias().map_or(Q::zero(), |b| q(b[row]));
                        for column in 0..owner.in_features() {
                            value += q(owner.weight()[[row, column]])
                                * &values[position * owner.in_features() + column];
                        }
                        next.push(value);
                    }
                }
                values = next;
            }
            Layer::ReLU(_) => values
                .iter_mut()
                .for_each(|value| *value = value.clone().max(Q::zero())),
            _ => panic!("test oracle only covers supported chain"),
        }
    }
    values
}
fn eval_row(bounds: &LinearBounds, row: usize, values: &[Q], lower: bool) -> Q {
    let (a, b, errors) = if lower {
        (bounds.lower_a(), bounds.lower_b(), bounds.lower_a_err())
    } else {
        (bounds.upper_a(), bounds.upper_b(), bounds.upper_a_err())
    };
    let mut result = q(b[row]);
    for (column, value) in values.iter().enumerate() {
        result += q(a[[row, column]]) * value;
        if let Some(errors) = errors {
            let charge = q(errors[[row, column]]) * value.abs();
            result += if lower { -charge } else { charge };
        }
    }
    result
}
fn incoming() -> LinearBounds {
    LinearBounds::from_prevalidated_parts_with_optional_err(
        array![
            [1.0, -0.5, 0.25, -0.75, 0.5, -0.25],
            [-0.01, 0.0, 1.0, 0.5, -1.0, 0.0]
        ],
        array![-0.125, 0.0],
        array![
            [-0.75, 0.5, -0.25, 0.5, -1.0, 0.25],
            [0.01, -1.0, 0.0, -0.5, 0.25, 1.0]
        ],
        array![0.5, 0.125],
        Some(Array2::from_elem((2, 6), 0.125)),
        Some(Array2::from_elem((2, 6), 0.0625)),
    )
    .unwrap()
}

#[test]
fn original_affine_chain_signed_and_sparse_rows_contain_rational_original_network() {
    let ledger = budget();
    let layers = layers(&ledger);
    let nodes = nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    assert_eq!(context.scope().numerical, NumericalSemantics::ExactReal);
    let input = input();
    let facts = context.forward(&input, options(&ledger)).unwrap();
    let incoming = incoming();
    let signed = context
        .signed_linear(&facts, &incoming, options(&ledger))
        .unwrap();
    let differences = [
        OriginalLogitDifference {
            target: 0,
            competitor: 1,
            position: 0,
        },
        OriginalLogitDifference {
            target: 1,
            competitor: 2,
            position: 1,
        },
    ];
    let sparse = context
        .sparse_margins(&facts, &differences, options(&ledger))
        .unwrap();
    assert_eq!(facts.bounds().shape(), &[2, 3]);
    assert_eq!(signed.bounds().lower_a().dim(), (2, 4));
    assert_eq!(sparse.bounds().lower_a().dim(), (2, 4));
    assert_eq!(facts.num_nodes(), 3);
    assert_eq!(signed.scope(), context.scope());
    assert_ne!(signed.binding().query, sparse.binding().query);
    // Five dyadic samples per original dimension include every box vertex and
    // interior activation crossings. The oracle performs no host float math.
    for encoded in 0..625usize {
        let mut digits = encoded;
        let mut x = Vec::new();
        for index in 0..4 {
            let digit = digits % 5;
            digits /= 5;
            let l = q(input.lower()[[index / 2, index % 2]]);
            let u = q(input.upper()[[index / 2, index % 2]]);
            x.push(l.clone() + (u - l) * Q::new(BigInt::from(digit), BigInt::from(4)));
        }
        let y = exact_forward(&layers, &x, 2);
        for (index, value) in y.iter().enumerate() {
            assert!(q(facts.bounds().lower()[[index / 3, index % 3]]) <= *value);
            assert!(q(facts.bounds().upper()[[index / 3, index % 3]]) >= *value);
        }
        for row in 0..2 {
            assert!(eval_row(signed.bounds(), row, &x, true) <= eval_row(&incoming, row, &y, true));
            assert!(
                eval_row(signed.bounds(), row, &x, false) >= eval_row(&incoming, row, &y, false)
            );
            let difference = differences[row];
            let exact = &y[difference.position * 3 + difference.target]
                - &y[difference.position * 3 + difference.competitor];
            assert!(eval_row(sparse.bounds(), row, &x, true) <= exact);
            assert!(eval_row(sparse.bounds(), row, &x, false) >= exact);
            if row == 1 {
                assert!(
                    exact.is_zero(),
                    "duplicate logits remain an exact unknown tie"
                );
            }
        }
    }
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn original_affine_chain_binds_nodes_edges_input_boxes_and_exact_demand_ids() {
    let ledger = budget();
    let layers = layers(&ledger);
    let nodes = nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    let facts = context.forward(&input(), options(&ledger)).unwrap();
    let mut altered = nodes.clone();
    altered[1].ids.node = Digest::of_bytes(b"changed-node", b"relu");
    let changed = PreparedOriginalAffineChain::prepare(&altered, options(&ledger)).unwrap();
    assert_eq!(context.scope().model, changed.scope().model);
    assert_ne!(context.scope().coordinates, changed.scope().coordinates);
    assert!(changed
        .signed_linear(&facts, &incoming(), options(&ledger))
        .is_err());
    let same = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    assert_eq!(context.scope(), same.scope());
    assert!(
        same.signed_linear(&facts, &incoming(), options(&ledger))
            .is_err(),
        "no cross-context fact authority"
    );
    let other_input = BoundedTensor::new(
        array![[-0.5, -0.25], [0.0, 0.5]].into_dyn(),
        array![[0.5, 0.5], [0.25, 0.75]].into_dyn(),
    )
    .unwrap();
    let other_facts = context.forward(&other_input, options(&ledger)).unwrap();
    let a = context
        .signed_linear(&facts, &incoming(), options(&ledger))
        .unwrap();
    let b = context
        .signed_linear(&other_facts, &incoming(), options(&ledger))
        .unwrap();
    assert_ne!(a.binding().query, b.binding().query);
    let first = [OriginalLogitDifference {
        target: 1,
        competitor: 2,
        position: 0,
    }];
    let second = [OriginalLogitDifference {
        target: 2,
        competitor: 1,
        position: 0,
    }];
    let a = context
        .sparse_margins(&facts, &first, options(&ledger))
        .unwrap();
    let b = context
        .sparse_margins(&facts, &second, options(&ledger))
        .unwrap();
    assert_ne!(
        a.binding().query,
        b.binding().query,
        "even duplicate-row ties bind their original IDs"
    );
    let before = ledger.receipt().unwrap().live_persistent_bytes;
    for invalid in [
        OriginalLogitDifference {
            target: 3,
            competitor: 1,
            position: 0,
        },
        OriginalLogitDifference {
            target: 0,
            competitor: 1,
            position: 2,
        },
    ] {
        assert!(context
            .sparse_margins(&facts, &[invalid], options(&ledger))
            .is_err());
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, before);
    }
}

#[test]
fn original_affine_chain_refuses_unsupported_branches_shapes_and_poison() {
    let ledger = budget();
    let layers = layers(&ledger);
    let ordinary = nodes(&layers);
    let unsupported = Layer::GELU(GELULayer::new(crate::layers::GeluApproximation::Tanh));
    let mut changed = ordinary.clone();
    changed[1].layer = &unsupported;
    assert!(matches!(
        PreparedOriginalAffineChain::prepare(&changed, options(&ledger)),
        Err(NyError::UnsupportedOp(_))
    ));
    changed = ordinary.clone();
    changed[2].ids.input = ids(0).input;
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    changed = ordinary.clone();
    changed[2].ids.output = ids(0).input;
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    changed = ordinary.clone();
    changed[2].ids.node = ids(0).node;
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    changed = ordinary.clone();
    changed[2].ids.lowering = Digest::of_bytes(b"other-lowering", b"x");
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    assert!(PreparedOriginalAffineChain::prepare(&ordinary[1..], options(&ledger)).is_err());
    let wrong = owner(&array![[1.0, 0.0, 0.0]], &array![0.0], &ledger, None);
    changed = ordinary.clone();
    changed[2].layer = &wrong;
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    let context = PreparedOriginalAffineChain::prepare(&ordinary, options(&ledger)).unwrap();
    let before = ledger.receipt().unwrap().live_persistent_bytes;
    let infinite = BoundedTensor::new_allow_infinite(
        array![f32::NEG_INFINITY, -1.0].into_dyn(),
        array![1.0, f32::INFINITY].into_dyn(),
    )
    .unwrap();
    assert!(context.forward(&infinite, options(&ledger)).is_err());
    let wrong_box = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap();
    assert!(context.forward(&wrong_box, options(&ledger)).is_err());
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, before);
    let facts = context.forward(&input(), options(&ledger)).unwrap();
    let mut poisoned = incoming();
    poisoned = LinearBounds::from_prevalidated_parts_with_optional_err(
        poisoned.lower_a().clone(),
        poisoned.lower_b().clone(),
        poisoned.upper_a().clone(),
        poisoned.upper_b().clone(),
        Some(Array2::from_elem((2, 6), -f32::from_bits(1))),
        None,
    )
    .unwrap();
    let before = ledger.receipt().unwrap().live_persistent_bytes;
    assert!(context
        .signed_linear(&facts, &poisoned, options(&ledger))
        .is_err());
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, before);
}

#[test]
fn original_affine_chain_same_ledger_resource_decline_has_no_refill_or_live_escape() {
    let ledger = budget();
    let layers = layers(&ledger);
    let ordinary = nodes(&layers);
    let substitute = budget();
    let before = substitute.receipt().unwrap();
    assert!(PreparedOriginalAffineChain::prepare(&ordinary, options(&substitute)).is_err());
    assert_eq!(substitute.receipt().unwrap(), before);
    let context = PreparedOriginalAffineChain::prepare(&ordinary, options(&ledger)).unwrap();
    let before = ledger.receipt().unwrap();
    assert!(context.forward(&input(), options(&substitute)).is_err());
    assert_eq!(ledger.receipt().unwrap(), before);
    let blocked = ledger
        .lease(WorkRequest {
            scratch_bytes: 1 << 24,
            ..WorkRequest::default()
        })
        .unwrap();
    assert!(context.forward(&input(), options(&ledger)).is_err());
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    drop(blocked);
    let before = ledger.receipt().unwrap();
    let blocked = ledger
        .lease(WorkRequest {
            persistent_bytes: (1 << 26) - before.live_persistent_bytes - 1024,
            ..WorkRequest::default()
        })
        .unwrap();
    assert!(context.forward(&input(), options(&ledger)).is_err());
    drop(blocked);
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert!(ledger.receipt().unwrap().refused > before.refused);
    let facts = context.forward(&input(), options(&ledger)).unwrap();
    let after_forward = ledger.receipt().unwrap().work_reserved;
    let result = context
        .signed_linear(
            &facts,
            &incoming(),
            OriginalAffineOptions {
                class: LeaseClass::Replay,
                ..options(&ledger)
            },
        )
        .unwrap();
    assert!(
        ledger.receipt().unwrap().work_reserved > after_forward,
        "Replay uses the original cumulative counters"
    );
    drop(result);
    let work = ledger.receipt().unwrap().work_reserved;
    drop(facts);
    assert_eq!(
        ledger.receipt().unwrap().work_reserved,
        work,
        "resident cleanup never refunds work"
    );
}

#[test]
fn original_affine_chain_cancellation_deadlines_and_last_clone_keep_actual_owners() {
    let memory = MemoryAuthority::new(1 << 28);
    let cancellation = CancellationToken::default();
    let ledger = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: 1 << 24,
            persistent_bytes: 1 << 26,
            max_work: 1 << 32,
            max_proposals: 4096,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let layers = layers(&ledger);
    let nodes = nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    let facts = context.forward(&input(), options(&ledger)).unwrap();
    let result = context
        .signed_linear(&facts, &incoming(), options(&ledger))
        .unwrap();
    let clone = result.clone();
    assert_eq!(
        result.bounds().lower_a().as_ptr(),
        clone.bounds().lower_a().as_ptr()
    );
    let before = ledger.receipt().unwrap().live_persistent_bytes;
    let expired = OriginalAffineOptions {
        deadline: Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        ..options(&ledger)
    };
    assert!(context.signed_linear(&facts, &incoming(), expired).is_err());
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, before);
    cancellation.cancel();
    assert!(context
        .sparse_margins(
            &facts,
            &[OriginalLogitDifference {
                target: 0,
                competitor: 1,
                position: 0
            }],
            options(&ledger)
        )
        .is_err());
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, before);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
    drop(result);
    drop(facts);
    drop(context);
    drop(nodes);
    drop(layers);
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        clone.original_owner_bytes() + clone.reserved_bytes(),
        "the final report accounts for every retained preparation and output reservation"
    );
    assert!(memory.receipt().unwrap().live_materialized_bytes > 0);
    assert_eq!(
        clone.bounds().lower_a().dim(),
        (2, 4),
        "completed report remains valid after cancellation"
    );
    drop(clone);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn original_affine_chain_relu_subnormal_chord_and_error_sign_flip_are_rationally_checked() {
    let ledger = budget();
    let tiny = f32::from_bits(1);
    let lower = array![-tiny, -1.0, -f32::MAX, 0.0, -2.0];
    let upper = array![tiny, 3.0, f32::MAX, tiny, -tiny];
    let pre = BoundedTensor::new(lower.clone().into_dyn(), upper.clone().into_dyn()).unwrap();
    let mut la = Array2::<f32>::zeros((5, 5));
    let mut ua = Array2::<f32>::zeros((5, 5));
    let mut errors = Array2::<f32>::zeros((5, 5));
    for row in 0..5 {
        la[[row, row]] = -0.25;
        ua[[row, row]] = 0.25;
    }
    errors[[0, 0]] = tiny;
    errors[[1, 1]] = 0.5;
    errors[[3, 3]] = tiny;
    let incoming = LinearBounds::from_prevalidated_parts_with_optional_err(
        la,
        Array1::zeros(5),
        ua,
        Array1::zeros(5),
        Some(errors.clone()),
        Some(errors),
    )
    .unwrap();
    let result = relu::backward(&incoming, &pre, options(&ledger)).unwrap();
    assert_eq!(
        result.bounds.lower_a_err().unwrap()[[3, 3]].to_bits(),
        tiny.to_bits(),
        "stable original error is retained exactly"
    );
    for row in 0..5 {
        for point in [lower[row], upper[row], 0.0] {
            if q(point) < q(lower[row]) || q(point) > q(upper[row]) {
                continue;
            }
            let mut x: Vec<_> = lower.iter().copied().map(q).collect();
            x[row] = q(point);
            let y: Vec<_> = x.iter().map(|value| value.clone().max(Q::zero())).collect();
            assert!(eval_row(&result.bounds, row, &x, true) <= eval_row(&incoming, row, &y, true));
            assert!(
                eval_row(&result.bounds, row, &x, false) >= eval_row(&incoming, row, &y, false)
            );
        }
    }
    let invalid = LinearBounds::from_prevalidated_parts_with_optional_err(
        incoming.lower_a().clone(),
        incoming.lower_b().clone(),
        incoming.upper_a().clone(),
        incoming.upper_b().clone(),
        Some(Array2::from_elem((5, 5), -tiny)),
        None,
    )
    .unwrap();
    let before = ledger.receipt().unwrap().live_persistent_bytes;
    assert!(relu::backward(&invalid, &pre, options(&ledger)).is_err());
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, before);
}

#[test]
fn original_affine_chain_tied_original_owner_has_one_resident_lifetime() {
    let ledger = budget();
    let original = owner(
        &array![[0.5, -0.25], [0.25, 0.5]],
        &array![0.125, -0.125],
        &ledger,
        None,
    );
    let relu = Layer::ReLU(ReLULayer::new());
    let nodes = [
        OriginalChainNode {
            ids: ids(0),
            layer: &original,
        },
        OriginalChainNode {
            ids: ids(1),
            layer: &relu,
        },
        OriginalChainNode {
            ids: ids(2),
            layer: &original,
        },
    ];
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    let owner_bytes = match &original {
        Layer::PointLinear(owner) => owner.reserved_bytes(),
        _ => unreachable!(),
    };
    let affine_preparations: Vec<_> = context
        .stages
        .iter()
        .filter_map(|stage| match stage {
            Stage::Affine(prepared) => Some(prepared.preparation_reserved_bytes()),
            Stage::Relu | Stage::GeluErf | Stage::SiLu => None,
        })
        .collect();
    assert_eq!(affine_preparations.len(), 2);
    assert_eq!(
        context.shared.preparation_metadata_bytes,
        affine_preparations.iter().sum::<usize>(),
        "each node preparation is retained once despite its tied coefficient owner"
    );
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        context.preparation_reserved_bytes() + owner_bytes,
        "preparation metadata is complete without counting tied original coefficients twice"
    );
    let facts = context.forward(&input(), options(&ledger)).unwrap();
    assert_eq!(
        facts.original_owner_bytes(),
        owner_bytes,
        "tied nodes retain one actual coefficient allocation"
    );
    drop(context);
    drop(original);
    drop(relu);
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        facts.reserved_bytes() + owner_bytes,
        "only the report survives, retaining all preparation reservations exactly once"
    );
    drop(facts);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn original_affine_chain_local_adapter_does_not_relax_ordinary_network_quarantine() {
    let ledger = budget();
    let layers = layers(&ledger);
    let original = match &layers[0] {
        Layer::PointLinear(owner) => owner.clone(),
        _ => unreachable!(),
    };
    let input = input();
    assert!(matches!(
        original.propagate_ibp(&input),
        Err(NyError::UnsupportedOp(_))
    ));
    let mut network = Network::new();
    for layer in &layers {
        network.add_layer(layer.clone());
    }
    assert!(matches!(
        network.ensure_proof_eligible(),
        Err(NyError::UnsupportedOp(_))
    ));
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("original", layers[0].clone()));
    graph.set_output("original");
    assert!(matches!(
        graph.propagate_ibp(&input),
        Err(NyError::UnsupportedOp(_))
    ));
    let context = PreparedOriginalAffineChain::prepare(&nodes(&layers), options(&ledger)).unwrap();
    assert!(context.forward(&input, options(&ledger)).is_ok());
}

#[test]
fn original_affine_chain_compact_stage_prices_actual_capacity_and_last_alias() {
    let ledger = budget();
    let layers = layers(&ledger);
    let node_list = nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&node_list, options(&ledger)).unwrap();
    let expected = META
        + context.stages.capacity() * size_of::<Stage<'_>>()
        + context.shared.owners.capacity() * size_of::<PointLinearLayer>();
    assert_eq!(context.shared.reservation.bytes(), expected);
    assert!(size_of::<Stage<'_>>() <= size_of::<PreparedOriginalAffine<'_>>() + size_of::<usize>());
    assert!(size_of::<PreparedOriginalAffine<'_>>() <= 64);
    let original = context
        .shared
        .owners
        .iter()
        .map(PointLinearLayer::reserved_bytes)
        .sum::<usize>();
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        context.preparation_reserved_bytes() + original
    );
    eprintln!(
        "ORIGINAL_AFFINE_STAGE_LAYOUT view={} arrays={} prepared={} stage={} stage_capacity={} owner_capacity={} chain_reservation={} per_node_preparation={}",
        size_of::<crate::layers::linear::original_affine::OriginalAffineView<'_>>(),
        size_of::<crate::layers::linear::original_affine::OriginalAffineArrays<'_>>(),
        size_of::<PreparedOriginalAffine<'_>>(),
        size_of::<Stage<'_>>(),
        context.stages.capacity(),
        context.shared.owners.capacity(),
        context.shared.reservation.bytes(),
        context.shared.preparation_metadata_bytes,
    );
    let facts = context.forward(&input(), options(&ledger)).unwrap();
    let clone = facts.clone();
    assert_eq!(
        std::ptr::from_ref(facts.scope()),
        std::ptr::from_ref(clone.scope())
    );
    assert_eq!(
        facts.bounds().lower().as_ptr(),
        clone.bounds().lower().as_ptr()
    );
    let held = clone.reserved_bytes() + clone.original_owner_bytes();
    drop(facts);
    drop(context);
    drop(node_list);
    drop(layers);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, held);
    let spent = ledger.receipt().unwrap().work_reserved;
    drop(clone);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, spent);
}

#[test]
fn original_affine_chain_stage_retained_storage_observation() {
    for kind in ["affine", "relu", "gelu", "silu", "tied"] {
        let ledger = budget();
        let mut chain_layers = layers(&ledger);
        match kind {
            "affine" => {
                chain_layers.remove(1);
            }
            "gelu" => {
                chain_layers[1] =
                    Layer::GELU(GELULayer::new(crate::layers::GeluApproximation::Erf));
            }
            "silu" => {
                chain_layers[1] = Layer::SiLU(crate::layers::SiLULayer::new());
            }
            "tied" => {
                chain_layers[2] = chain_layers[0].clone();
            }
            _ => {}
        }
        let node_list = nodes(&chain_layers);
        let context = PreparedOriginalAffineChain::prepare(&node_list, options(&ledger)).unwrap();
        let fixed = META
            + context.stages.capacity() * size_of::<Stage<'_>>()
            + context.shared.owners.capacity() * size_of::<PointLinearLayer>();
        assert_eq!(context.shared.reservation.bytes(), fixed);
        let original_bytes = context
            .shared
            .owners
            .iter()
            .map(PointLinearLayer::reserved_bytes)
            .sum::<usize>();
        let preparation = context.preparation_reserved_bytes();
        let before = ledger.receipt().unwrap();
        assert_eq!(before.live_persistent_bytes, preparation + original_bytes);
        assert_eq!(Arc::strong_count(&context.shared), 1);
        let facts = context.forward(&input(), options(&ledger)).unwrap();
        let clone = facts.clone();
        assert_eq!(Arc::strong_count(&context.shared), 2);
        assert_eq!(
            std::ptr::from_ref(context.scope()),
            std::ptr::from_ref(facts.scope())
        );
        assert_eq!(
            facts.bounds().lower().as_ptr(),
            clone.bounds().lower().as_ptr()
        );
        let held = clone.reserved_bytes() + clone.original_owner_bytes();
        let after = ledger.receipt().unwrap();
        assert_eq!(after.live_persistent_bytes, held);
        eprintln!(
            "ORIGINAL_AFFINE_STAGE_RETAINED kind={} nodes={} view={} prepared={} stage={} stage_capacity={} owner_capacity={} unique_owners={} chain_reservation={} per_node_preparation={} original_bytes={} preparation={} prepare_live={} held={} work_prepare={} work_forward={} proposals={} peak={}",
            kind, context.stages.len(),
            size_of::<crate::layers::linear::original_affine::OriginalAffineView<'_>>(),
            size_of::<PreparedOriginalAffine<'_>>(), size_of::<Stage<'_>>(),
            context.stages.capacity(), context.shared.owners.capacity(), context.shared.owners.len(),
            context.shared.reservation.bytes(), context.shared.preparation_metadata_bytes,
            original_bytes, preparation, before.live_persistent_bytes, held,
            before.work_reserved, after.work_reserved, after.proposals_reserved, after.peak_leased_bytes,
        );
        drop(facts);
        drop(context);
        drop(node_list);
        drop(chain_layers);
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, held);
        drop(clone);
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(ledger.receipt().unwrap().work_reserved, after.work_reserved);
    }
}

#[test]
fn original_affine_chain_fact_retained_storage_observation() {
    for kind in ["affine", "relu", "gelu", "silu", "tied"] {
        let ledger = budget();
        let mut chain_layers = layers(&ledger);
        match kind {
            "affine" => {
                chain_layers.remove(1);
            }
            "gelu" => {
                chain_layers[1] =
                    Layer::GELU(GELULayer::new(crate::layers::GeluApproximation::Erf));
            }
            "silu" => {
                chain_layers[1] = Layer::SiLU(crate::layers::SiLULayer::new());
            }
            "tied" => {
                chain_layers[2] = chain_layers[0].clone();
            }
            _ => {}
        }
        let node_list = nodes(&chain_layers);
        let context = PreparedOriginalAffineChain::prepare(&node_list, options(&ledger)).unwrap();
        let fixed = META
            + context.stages.capacity() * size_of::<Stage<'_>>()
            + context.shared.owners.capacity() * size_of::<PointLinearLayer>();
        assert_eq!(context.shared.reservation.bytes(), fixed);
        let original_bytes = context
            .shared
            .owners
            .iter()
            .map(PointLinearLayer::reserved_bytes)
            .sum::<usize>();
        let preparation = context.preparation_reserved_bytes();
        let before = ledger.receipt().unwrap();
        assert_eq!(before.live_persistent_bytes, preparation + original_bytes);
        assert_eq!(Arc::strong_count(&context.shared), 1);
        let facts = context.forward(&input(), options(&ledger)).unwrap();
        let clone = facts.clone();
        assert_eq!(Arc::strong_count(&context.shared), 2);
        assert_eq!(
            std::ptr::from_ref(context.scope()),
            std::ptr::from_ref(facts.scope())
        );
        assert_eq!(
            facts.bounds().lower().as_ptr(),
            clone.bounds().lower().as_ptr()
        );
        let held = clone.reserved_bytes() + clone.original_owner_bytes();
        let after = ledger.receipt().unwrap();
        assert_eq!(after.live_persistent_bytes, held);
        let (forward_owner, chain_owner, fact_capacity, fact_bytes) =
            facts.retained_storage_observation();
        eprintln!(
            "ORIGINAL_CHAIN_FACT_RETAINED kind={} nodes={} fact={} fact_capacity={} fact_vec_bytes={} relu_header={} gelu_header={} relu_data_header={} gelu_data_header={} relu_owner_bound={} gelu_owner_bound={} fact_meta={} forward_owner={} chain_owner={} stage={} stage_capacity={} original_bytes={} held={} work={} proposals={} peak={}",
            kind, facts.num_nodes(), fact_bytes, fact_capacity,
            fact_bytes.checked_mul(fact_capacity).unwrap(),
            size_of::<OwnedBox>(), size_of::<activation_stage::OwnedActivationFact>(),
            size_of::<OwnedBoxData>(), size_of::<activation_stage::ActivationFactData>(),
            RetainedOwner::<OwnedBoxData>::metadata_bytes().unwrap(),
            RetainedOwner::<activation_stage::ActivationFactData>::metadata_bytes().unwrap(), META,
            forward_owner, chain_owner, size_of::<Stage<'_>>(), context.stages.capacity(),
            original_bytes, held, after.work_reserved, after.proposals_reserved,
            after.peak_leased_bytes,
        );
        drop(facts);
        drop(context);
        drop(node_list);
        drop(chain_layers);
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, held);
        drop(clone);
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(ledger.receipt().unwrap().work_reserved, after.work_reserved);
    }
}
