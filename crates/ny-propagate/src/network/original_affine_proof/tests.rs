// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use ndarray::{array, Array1, Array2};
use num_bigint::BigInt;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

use crate::certify::scope::NumericalSemantics;
use crate::certify::work::{WorkReceipt, WorkReserve};
use crate::layers::{BoundPropagation, Layer, PointLinearBuildPolicy, PointLinearLayer};
use crate::{GraphNetwork, GraphNode, Network};

use super::*;

fn budget() -> WorkBudget {
    WorkBudget::new(None, 1 << 24, 1 << 26, 1 << 32, 1024, Duration::ZERO).unwrap()
}
fn options(budget: &WorkBudget) -> OriginalAffineOptions<'_> {
    OriginalAffineOptions {
        budget,
        class: LeaseClass::Optional,
        deadline: None,
    }
}
fn ids() -> OriginalAffineIds {
    OriginalAffineIds {
        lowering: Digest::of_bytes(b"test-lowering", b"original one affine map"),
        node: Digest::of_bytes(b"test-node", b"affine"),
        input: Digest::of_bytes(b"test-input", b"original input"),
        output: Digest::of_bytes(b"test-output", b"original output"),
    }
}
fn owner(
    weights: &Array2<f32>,
    bias: Option<&Array1<f32>>,
    ledger: &WorkBudget,
    deadline: Option<Instant>,
) -> PointLinearLayer {
    PointLinearLayer::from_original_view(
        weights.view(),
        false,
        bias.map(|b| b.view().into_dyn()),
        PointLinearBuildPolicy {
            budget: ledger,
            deadline,
            max_resident_bytes: 1 << 26,
            max_scratch_bytes: 1 << 24,
            max_work: 1 << 32,
        },
    )
    .unwrap()
}
fn q(value: f32) -> Q {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    assert_ne!(exponent, 0xff);
    let fraction = bits & 0x7f_ffff;
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
fn unchanged_bytes(before: WorkReceipt, after: WorkReceipt) {
    assert_eq!(before.live_persistent_bytes, after.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, 0);
}

#[test]
fn original_affine_context_original_owner_interval_and_full_rows_replay_rational() {
    let large = 2f32.powi(30);
    let small = 2f32.powi(-30);
    let weights = array![[large, 1.0, -large], [-0.5, 0.25, 2.0]];
    let biases = array![-0.25, 0.125];
    let ledger = budget();
    let original = owner(&weights, Some(&biases), &ledger, None);
    let context = PreparedOriginalAffine::prepare(&original, ids(), options(&ledger)).unwrap();
    assert_eq!(context.scope().model, original.original_digest());
    assert_eq!(context.scope().numerical, NumericalSemantics::ExactReal);
    let input = BoundedTensor::new(
        array![[1.0, small, 1.0], [-0.75, -0.5, -1.0]].into_dyn(),
        array![[1.0, small, 1.0], [-0.5, 0.75, -0.75]].into_dyn(),
    )
    .unwrap();
    let interval = context.interval(&input, options(&ledger)).unwrap();
    for position in 0..2 {
        for row in 0..2 {
            let mut lo = q(biases[row]);
            let mut hi = lo.clone();
            for j in 0..3 {
                let a = q(weights[[row, j]]) * q(input.lower()[[position, j]]);
                let b = q(weights[[row, j]]) * q(input.upper()[[position, j]]);
                lo += a.clone().min(b.clone());
                hi += a.max(b);
            }
            assert!(q(interval.bounds().lower()[[position, row]]) <= lo);
            assert!(q(interval.bounds().upper()[[position, row]]) >= hi);
        }
    }
    let incoming = LinearBounds::new(
        array![[1.0, -0.5, -0.25, 0.75]],
        array![-0.125],
        array![[-0.75, 0.5, 0.125, 1.0]],
        array![0.5],
    )
    .unwrap();
    let signed = context.signed_linear(&incoming, options(&ledger)).unwrap();
    assert_eq!(signed.bounds().lower_a().dim(), (1, 6));
    assert!(signed.bounds().lower_a_err().is_some() && signed.bounds().upper_a_err().is_some());
    for vertex in 0..64 {
        let x: Vec<_> = (0..6)
            .map(|j| {
                q(if vertex & (1 << j) == 0 {
                    input.lower()[[j / 3, j % 3]]
                } else {
                    input.upper()[[j / 3, j % 3]]
                })
            })
            .collect();
        let mut y = Vec::new();
        for position in 0..2 {
            for row in 0..2 {
                let mut value = q(biases[row]);
                for j in 0..3 {
                    value += q(weights[[row, j]]) * &x[position * 3 + j];
                }
                y.push(value);
            }
        }
        let mut original_lower = q(incoming.lower_b()[0]);
        let mut original_upper = q(incoming.upper_b()[0]);
        for j in 0..4 {
            original_lower += q(incoming.lower_a()[[0, j]]) * &y[j];
            original_upper += q(incoming.upper_a()[[0, j]]) * &y[j];
        }
        let mut stored_lower = q(signed.bounds().lower_b()[0]);
        let mut stored_upper = q(signed.bounds().upper_b()[0]);
        for j in 0..6 {
            stored_lower += q(signed.bounds().lower_a()[[0, j]]) * &x[j]
                - q(signed.bounds().lower_a_err().unwrap()[[0, j]]) * x[j].abs();
            stored_upper += q(signed.bounds().upper_a()[[0, j]]) * &x[j]
                + q(signed.bounds().upper_a_err().unwrap()[[0, j]]) * x[j].abs();
        }
        assert!(stored_lower <= original_lower && stored_upper >= original_upper);
    }
    assert_ne!(interval.binding().query, signed.binding().query);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn original_affine_context_sparse_rows_bind_every_id_position_and_original_error() {
    let weights = array![[0.1, -0.5], [0.3, 0.25], [0.1, -0.5]];
    let biases = array![0.125, -0.25, 0.125];
    let ledger = budget();
    let original = owner(&weights, Some(&biases), &ledger, None);
    let context = PreparedOriginalAffine::prepare(&original, ids(), options(&ledger)).unwrap();
    let directions = [
        OriginalLogitDifference {
            target: 1,
            competitor: 0,
            position: 0,
        },
        OriginalLogitDifference {
            target: 0,
            competitor: 2,
            position: 1,
        },
    ];
    let report = context
        .sparse_margins(2, &directions, options(&ledger))
        .unwrap();
    assert_eq!(report.bounds().lower_a().dim(), (2, 4));
    for (row, direction) in directions.iter().enumerate() {
        for position in 0..2 {
            for j in 0..2 {
                let index = position * 2 + j;
                let exact = if position == direction.position {
                    q(weights[[direction.target, j]]) - q(weights[[direction.competitor, j]])
                } else {
                    Q::zero()
                };
                let central = q(report.bounds().lower_a()[[row, index]]);
                let error = q(report.bounds().lower_a_err().unwrap()[[row, index]]);
                assert!((central - exact).abs() <= error);
                if position != direction.position {
                    assert_eq!(report.bounds().lower_a()[[row, index]].to_bits(), 0);
                    assert_eq!(
                        report.bounds().lower_a_err().unwrap()[[row, index]].to_bits(),
                        0
                    );
                }
            }
        }
        let exact = q(biases[direction.target]) - q(biases[direction.competitor]);
        assert!(q(report.bounds().lower_b()[row]) <= exact);
        assert!(q(report.bounds().upper_b()[row]) >= exact);
    }
    // Equal original rows still have distinct demanded IDs and query scopes.
    let changed = [OriginalLogitDifference {
        target: 2,
        competitor: 0,
        position: 1,
    }];
    let other = context
        .sparse_margins(2, &changed, options(&ledger))
        .unwrap();
    let equal = context
        .sparse_margins(2, &directions[1..], options(&ledger))
        .unwrap();
    assert_ne!(equal.binding().query, other.binding().query);
    let before = ledger.receipt().unwrap();
    let invalid_id = [OriginalLogitDifference {
        target: 3,
        competitor: 0,
        position: 0,
    }];
    assert!(context
        .sparse_margins(2, &invalid_id, options(&ledger))
        .is_err());
    let invalid_position = [OriginalLogitDifference {
        target: 0,
        competitor: 1,
        position: 2,
    }];
    assert!(context
        .sparse_margins(2, &invalid_position, options(&ledger))
        .is_err());
    unchanged_bytes(before, ledger.receipt().unwrap());
}

#[test]
fn original_affine_context_reports_keep_one_output_original_owner_and_last_lease() {
    let ledger = budget();
    let original = owner(&array![[0.5, -0.25]], None, &ledger, None);
    let original_bytes = original.reserved_bytes();
    let context = PreparedOriginalAffine::prepare(&original, ids(), options(&ledger)).unwrap();
    let prepared_bytes = context.preparation_reserved_bytes();
    let report = context
        .interval(
            &BoundedTensor::concrete(array![1.0, -2.0].into_dyn()).unwrap(),
            options(&ledger),
        )
        .unwrap();
    let held = original_bytes + prepared_bytes + report.reserved_bytes();
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, held);
    let copied = report.clone();
    assert_eq!(
        copied.bounds().lower().as_ptr(),
        report.bounds().lower().as_ptr()
    );
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, held);
    drop(context);
    drop(original);
    drop(report);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, held);
    assert_eq!(copied.bounds().shape(), &[1]);
    drop(copied);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn original_affine_context_refuses_substitute_ledger_deadline_and_poison_without_escape() {
    let ledger = budget();
    let original_deadline = Instant::now() + Duration::from_secs(120);
    let original = owner(&array![[1.0, -0.5]], None, &ledger, Some(original_deadline));
    assert_eq!(original.original_deadline(), Some(original_deadline));
    let substitute = budget();
    assert!(PreparedOriginalAffine::prepare(&original, ids(), options(&substitute)).is_err());
    assert_eq!(substitute.receipt().unwrap().attempted, 0);
    let prepared_deadline = Instant::now() + Duration::from_secs(60);
    let context = PreparedOriginalAffine::prepare(
        &original,
        ids(),
        OriginalAffineOptions {
            deadline: Some(prepared_deadline),
            ..options(&ledger)
        },
    )
    .unwrap();
    assert_eq!(context.shared.deadline, Some(prepared_deadline));
    let input = BoundedTensor::concrete(array![1.0, -1.0].into_dyn()).unwrap();
    let before = ledger.receipt().unwrap();
    assert!(context.interval(&input, options(&substitute)).is_err());
    assert_eq!(substitute.receipt().unwrap().attempted, 0);
    assert!(matches!(
        context.interval(
            &input,
            OriginalAffineOptions {
                deadline: Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
                ..options(&ledger)
            }
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    let mut incoming =
        LinearBounds::new(array![[-1.0]], array![0.0], array![[1.0]], array![0.0]).unwrap();
    incoming.lower_a_err = Some(array![[-f32::from_bits(1)]]);
    assert!(context.signed_linear(&incoming, options(&ledger)).is_err());
    incoming.lower_a_err = None;
    incoming.lower_b_mut()[0] = f32::NAN;
    assert!(context.signed_linear(&incoming, options(&ledger)).is_err());
    unchanged_bytes(before, ledger.receipt().unwrap());
}

#[test]
fn original_affine_context_phase_capacity_refuses_before_output_allocation() {
    let weights = array![[1.0, -0.5]];
    let roomy = budget();
    let sample = owner(&weights, None, &roomy, None);
    let context = PreparedOriginalAffine::prepare(&sample, ids(), options(&roomy)).unwrap();
    let input = BoundedTensor::concrete(array![1.0, 0.0].into_dyn()).unwrap();
    let cold = PointLinearLayer::construction_request(1, 2, false).unwrap();
    let prepared = PreparedOriginalAffine::preparation_request(&sample).unwrap();
    let query = context.interval_request(&input).unwrap();
    for cap in [1, 2] {
        let persistent = cold.persistent_bytes + prepared.persistent_bytes + query.persistent_bytes;
        let work = cold.work + prepared.work + query.work;
        let ledger = WorkBudget::new(
            None,
            1 << 24,
            if cap == 1 { persistent - 1 } else { persistent },
            if cap == 2 { work - 1 } else { work },
            10,
            Duration::ZERO,
        )
        .unwrap();
        let original = owner(&weights, None, &ledger, None);
        let prepared = PreparedOriginalAffine::prepare(&original, ids(), options(&ledger)).unwrap();
        let before = ledger.receipt().unwrap();
        assert!(prepared.interval(&input, options(&ledger)).is_err());
        let after = ledger.receipt().unwrap();
        unchanged_bytes(before, after);
        assert_eq!(after.work_reserved, before.work_reserved);
        assert_eq!(after.refused, before.refused + 1);
    }
    let before = roomy.receipt().unwrap();
    let busy = roomy
        .lease(WorkRequest {
            scratch_bytes: (1 << 24) - query.scratch_bytes + 1,
            ..WorkRequest::default()
        })
        .unwrap();
    assert!(context.interval(&input, options(&roomy)).is_err());
    assert_eq!(
        roomy.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    drop(busy);
    unchanged_bytes(before, roomy.receipt().unwrap());
}

#[test]
fn original_affine_context_replay_reserve_shares_counters_and_preserves_graph_quarantine() {
    let weights = array![[1.0, -0.5]];
    let roomy = budget();
    let sample = owner(&weights, None, &roomy, None);
    let prepared = PreparedOriginalAffine::prepare(&sample, ids(), options(&roomy)).unwrap();
    let input = BoundedTensor::concrete(array![1.0, -1.0].into_dyn()).unwrap();
    let cold = PointLinearLayer::construction_request(1, 2, false).unwrap();
    let preparation = PreparedOriginalAffine::preparation_request(&sample).unwrap();
    let query = prepared.interval_request(&input).unwrap();
    let ledger = WorkBudget::new_reserved(
        None,
        1 << 24,
        1 << 26,
        cold.work + preparation.work + query.work,
        10,
        Duration::ZERO,
        WorkReserve {
            work: query.work,
            ..WorkReserve::default()
        },
    )
    .unwrap();
    let original = owner(&weights, None, &ledger, None);
    let context = PreparedOriginalAffine::prepare(&original, ids(), options(&ledger)).unwrap();
    assert!(context.interval(&input, options(&ledger)).is_err());
    let report = context
        .interval(
            &input,
            OriginalAffineOptions {
                class: LeaseClass::Replay,
                ..options(&ledger)
            },
        )
        .unwrap();
    assert!(report.bounds().lower()[0] <= 1.5 && report.bounds().upper()[0] >= 1.5);
    assert!(
        context
            .interval(
                &input,
                OriginalAffineOptions {
                    class: LeaseClass::Fallback,
                    ..options(&ledger)
                }
            )
            .is_err(),
        "fallback cannot manufacture a fresh work allowance"
    );
    assert!(matches!(
        original.propagate_ibp(&input),
        Err(NyError::UnsupportedOp(_))
    ));
    let mut network = Network::new();
    network.add_layer(Layer::PointLinear(original.clone()));
    assert!(matches!(
        network.ensure_proof_eligible(),
        Err(NyError::UnsupportedOp(_))
    ));
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "affine",
        Layer::PointLinear(original),
    ));
    graph.set_output("affine");
    assert!(matches!(
        graph.ensure_proof_eligible(),
        Err(NyError::UnsupportedOp(_))
    ));
}

#[test]
fn original_affine_context_compact_handle_keeps_identity_after_caller_drop() {
    let ledger = budget();
    let original = owner(&array![[0.5, -0.25]], Some(&array![-0.0]), &ledger, None);
    let context = PreparedOriginalAffine::prepare(&original, ids(), options(&ledger)).unwrap();
    let weight = original.weight().as_ptr();
    let scope = std::ptr::from_ref(context.scope());
    let prepared = context.preparation_reserved_bytes();
    let owner_bytes = original.reserved_bytes();
    assert_eq!(context.shared.owner.weight().as_ptr(), weight);
    let report = context
        .interval(
            &BoundedTensor::concrete(array![-0.75, 1.25].into_dyn()).unwrap(),
            options(&ledger),
        )
        .unwrap();
    assert_eq!(std::ptr::from_ref(report.scope()), scope);
    let held = owner_bytes + prepared + report.reserved_bytes();
    let clone = report.clone();
    assert_eq!(std::ptr::from_ref(clone.scope()), scope);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, held);
    drop(context);
    drop(original);
    drop(report);
    assert_eq!(std::ptr::from_ref(clone.scope()), scope);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, held);
    assert!(q(clone.bounds().lower()[0]) <= q(-0.6875));
    assert!(q(clone.bounds().upper()[0]) >= q(-0.6875));
    let spent = ledger.receipt().unwrap().work_reserved;
    drop(clone);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, spent);
}
