// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

fn budget() -> WorkBudget {
    WorkBudget::new(None, 8 << 20, 1 << 20, 1 << 30, 100, Duration::ZERO).unwrap()
}
fn geometry(lower: &[f64], upper: &[f64], epsilon: f64, centered: bool) -> GeometryF64 {
    // The caller's already-retained geometry is separate from this private
    // kernel's incremental phase. These tiny fixtures test that same boundary.
    GeometryF64::new(lower, upper, epsilon, centered, &mut |_: usize| {
        Ok::<_, ()>(())
    })
    .unwrap()
}
fn run(
    geometry: &GeometryF64,
    request: &GroupSupportRequest<'_>,
    ledger: &WorkBudget,
) -> Result<CheckedGroupUpper> {
    let prediction = group_support_request(request)?;
    let phase = GroupSupportPhase::admit(ledger, LeaseClass::Optional, prediction, None)?;
    group_upper(geometry, request, phase)
}
fn run_member(
    geometry: &GeometryF64,
    request: &GroupSupportRequest<'_>,
    ledger: &WorkBudget,
) -> Result<CheckedMemberGroupUpper> {
    let prediction = member_group_support_request(request)?;
    let phase = GroupSupportPhase::admit(ledger, LeaseClass::Optional, prediction, None)?;
    member_group_upper(geometry, request, phase)
}
fn q(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}

/// Independent original rational evaluation. Squared signed comparisons
/// eliminate sqrt without sharing interval arithmetic or floating proposals.
fn original_containment(
    bound: f64,
    request: &GroupSupportRequest<'_>,
    point: &[f64],
    epsilon: f64,
    centered: bool,
) {
    let h = request.hidden;
    let nf = BigRational::from_integer(BigInt::from(h));
    let mut z: Vec<_> = point.iter().copied().map(q).collect();
    if centered {
        let mean = z.iter().cloned().sum::<BigRational>() / &nf;
        for value in &mut z {
            *value -= &mean;
        }
    }
    let denominator2 = q(epsilon)
        + z.iter()
            .map(|value| value * value / &nf)
            .sum::<BigRational>();
    assert!(denominator2 > BigRational::zero());
    let target = &request.weights[request.target * h..(request.target + 1) * h];
    for member in request.start..request.end {
        if member == request.target {
            continue;
        }
        let row = &request.weights[member * h..(member + 1) * h];
        let mut numerator = BigRational::zero();
        let mut constant = q(request.bias.map_or(0.0, |bias| bias[member]))
            - q(request.bias.map_or(0.0, |bias| bias[request.target]));
        for i in 0..h {
            let difference = q(row[i]) - q(target[i]);
            numerator += &difference * q(request.gamma[i]) * &z[i];
            if let Some(beta) = request.beta {
                constant += difference * q(beta[i]);
            }
        }
        let allowance = q(bound) - constant;
        if numerator >= BigRational::zero() {
            assert!(
                allowance >= BigRational::zero(),
                "member {member}: wrong upper sign"
            );
            assert!(
                &allowance * &allowance * &denominator2 >= &numerator * &numerator,
                "member {member}: original positive margin exceeds {bound}"
            );
        } else if allowance < BigRational::zero() {
            assert!(
                &allowance * &allowance * &denominator2 <= &numerator * &numerator,
                "member {member}: original negative margin exceeds {bound}"
            );
        }
    }
}

#[test]
fn decoder_n12_group_original_rational_signed_zero_gamma_and_nondyadic_ln_mean() {
    let weights = [
        1.0 + 2f64.powi(-40),
        -0.5,
        0.25,
        -0.75,
        0.125,
        0.8,
        1.25,
        -0.25,
        -0.1 - 2f64.powi(-42),
        0.25,
        0.375,
        -0.5,
    ];
    let bias = [0.25, -2.0, -1.5, -2.25];
    let gamma = [-1.5, 0.0, 2.0 + 2f64.powi(-41)];
    let beta = [0.125, -0.25, 2f64.powi(-30)];
    let lower = [-0.375, 0.1875, -0.25];
    let upper = [0.3125, 0.4375, 0.125];
    let request = GroupSupportRequest {
        weights: &weights,
        bias: Some(&bias),
        hidden: 3,
        target: 0,
        start: 1,
        end: 4,
        gamma: &gamma,
        beta: Some(&beta),
    };
    for centered in [false, true] {
        let geometry = geometry(&lower, &upper, 1e-5, centered);
        let bound = run(&geometry, &request, &budget()).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let choices = [a, b, c];
                    let point: Vec<f64> = (0..3)
                        .map(|i| match choices[i] {
                            0 => lower[i],
                            1 => lower[i] * 0.5 + upper[i] * 0.5,
                            _ => upper[i],
                        })
                        .collect();
                    original_containment(bound.upper(), &request, &point, 1e-5, centered);
                }
            }
        }
        assert!(bound.raw_radius_upper() > 0.0);
        assert!(bound.projected_anchor_norm_upper() > 0.0);
    }
}

#[test]
fn decoder_n12_group_h1_ln_beta_and_bias_are_paid_once_for_the_complete_box() {
    let weights = [2.0, 1.0, -3.0];
    let bias = [0.0, -0.125, 1.0];
    let request = GroupSupportRequest {
        weights: &weights,
        bias: Some(&bias),
        hidden: 1,
        target: 0,
        start: 1,
        end: 3,
        gamma: &[-2.0],
        beta: Some(&[0.5]),
    };
    let geometry = geometry(&[-2.0], &[3.0], 1e-5, true);
    let bound = run(&geometry, &request, &budget()).unwrap();
    assert_eq!(bound.displacement_upper(), 0.0);
    assert!(bound.upper() < -0.62 && bound.upper() >= -0.625);
    for point in [-2.0, -0.1, 0.0, 3.0] {
        original_containment(bound.upper(), &request, &[point], 1e-5, true);
    }
}

#[test]
fn decoder_n12_group_positive_epsilon_rms_does_not_assume_saturated_norm() {
    let request = GroupSupportRequest {
        weights: &[0.0, 1.0, -0.25],
        bias: Some(&[0.0, -1.0, -1.0]),
        hidden: 1,
        target: 0,
        start: 1,
        end: 3,
        gamma: &[1.0],
        beta: None,
    };
    let geometry = geometry(&[0.125], &[0.125], 0.25, false);
    let bound = run(&geometry, &request, &budget()).unwrap();
    assert_eq!(bound.displacement_upper(), 0.0);
    assert!(bound.upper() < -0.75 && bound.upper() > -1.0);
    original_containment(bound.upper(), &request, &[0.125], 0.25, false);
}

#[test]
fn decoder_n12_group_nonvacuous_narrow_original_box_has_strict_negative_support() {
    let weights = [0.0, 0.0, 0.0, 0.25, -0.5, 0.125, 0.3, -0.45, 0.15];
    let request = GroupSupportRequest {
        weights: &weights,
        bias: Some(&[0.0, -2.0, -1.75]),
        hidden: 3,
        target: 0,
        start: 1,
        end: 3,
        gamma: &[1.0, -0.75, 0.5],
        beta: Some(&[0.1, -0.1, 0.05]),
    };
    let lower = [0.2499, -0.3751, 0.1249];
    let upper = [0.2501, -0.3749, 0.1251];
    let geometry = geometry(&lower, &upper, 1e-5, true);
    let bound = run(&geometry, &request, &budget()).unwrap();
    assert!(bound.upper() < -1.0);
    assert!(bound.displacement_upper() > 0.0);
    for mask in 0..8 {
        let point: Vec<_> = (0..3)
            .map(|i| {
                if mask & (1 << i) == 0 {
                    lower[i]
                } else {
                    upper[i]
                }
            })
            .collect();
        original_containment(bound.upper(), &request, &point, 1e-5, true);
    }
}

#[test]
fn decoder_n12_group_target_exclusion_and_last_unsafe_member_prevent_false_closure() {
    let weights = vec![0.0; 257];
    let mut bias = vec![-1.0; 257];
    bias[0] = 0.0;
    bias[256] = 0.125;
    let request = GroupSupportRequest {
        weights: &weights,
        bias: Some(&bias),
        hidden: 1,
        target: 0,
        start: 0,
        end: 257,
        gamma: &[0.0],
        beta: None,
    };
    let geometry = geometry(&[0.0], &[0.0], 1e-5, false);
    let bound = run(&geometry, &request, &budget()).unwrap();
    assert_eq!(bound.range(), (0, 257));
    assert_eq!(bound.target(), 0);
    assert_eq!(bound.competitors(), 256);
    assert_eq!(bound.upper(), 0.125);
    assert_eq!(bound.stats().original_member_visits, 257);
    original_containment(bound.upper(), &request, &[0.0], 1e-5, false);
    let tie = GroupSupportRequest {
        weights: &[0.0, 0.0],
        bias: None,
        hidden: 1,
        target: 0,
        start: 1,
        end: 2,
        gamma: &[0.0],
        beta: None,
    };
    assert_eq!(run(&geometry, &tie, &budget()).unwrap().upper(), 0.0);
    let empty = GroupSupportRequest { end: 1, ..request };
    assert_eq!(
        group_support_request(&empty),
        Err(GroupSupportError::Input("group has no original competitor"))
    );
}

#[test]
fn decoder_n12_group_subnormal_singletons_keep_original_anchor_and_finite_support() {
    for point in [f64::from_bits(1), -f64::from_bits(1)] {
        let request = GroupSupportRequest {
            weights: &[0.0, 1.0, -1.0],
            bias: Some(&[0.0, -1.0, -1.0]),
            hidden: 1,
            target: 0,
            start: 1,
            end: 3,
            gamma: &[-1.0],
            beta: None,
        };
        for centered in [false, true] {
            let geometry = geometry(&[point], &[point], 1e-5, centered);
            let bound = run(&geometry, &request, &budget()).unwrap();
            assert_eq!(bound.displacement_upper(), 0.0);
            assert!(bound.upper() < 0.0);
            original_containment(bound.upper(), &request, &[point], 1e-5, centered);
        }
    }
}

#[test]
fn decoder_n12_group_poisoned_originals_are_checked_before_zero_shortcuts() {
    let geometry = geometry(&[0.0], &[0.0], 1e-5, false);
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let weights = [0.0, 0.0, poison];
        let request = GroupSupportRequest {
            weights: &weights,
            bias: None,
            hidden: 1,
            target: 0,
            start: 1,
            end: 3,
            gamma: &[0.0],
            beta: None,
        };
        let ledger = budget();
        assert!(matches!(
            run(&geometry, &request, &ledger),
            Err(GroupSupportError::Arithmetic(
                "group finite original member"
            ))
        ));
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
        let poisoned_gamma = GroupSupportRequest {
            weights: &[0.0, 0.0],
            end: 2,
            gamma: &[poison],
            ..request
        };
        assert!(matches!(
            run(&geometry, &poisoned_gamma, &budget()),
            Err(GroupSupportError::Arithmetic(
                "support finite anchor parameters"
            ))
        ));
        let poisoned_beta = GroupSupportRequest {
            weights: &[0.0, 0.0],
            end: 2,
            beta: Some(&[poison]),
            ..request
        };
        assert!(matches!(
            run(&geometry, &poisoned_beta, &budget()),
            Err(GroupSupportError::Arithmetic(
                "support finite anchor parameters"
            ))
        ));
        let poisoned_bias = GroupSupportRequest {
            weights: &[0.0, 0.0],
            end: 2,
            bias: Some(&[0.0, poison]),
            ..request
        };
        assert!(matches!(
            run(&geometry, &poisoned_bias, &budget()),
            Err(GroupSupportError::Arithmetic(
                "group finite original member bias"
            ))
        ));
        let poisoned_target = GroupSupportRequest {
            weights: &[poison, 0.0],
            end: 2,
            ..request
        };
        assert!(matches!(
            run(&geometry, &poisoned_target, &budget()),
            Err(GroupSupportError::Arithmetic(
                "support finite group direction"
            ))
        ));
    }
}

#[test]
fn decoder_n12_group_distant_radius_and_overflow_refuse_without_a_scalar_claim() {
    let geometry = geometry(&[0.125], &[0.25], 1e-5, false);
    let request = GroupSupportRequest {
        weights: &[0.0, 1.0, 100.0],
        bias: Some(&[0.0, -10.0, -10.0]),
        hidden: 1,
        target: 0,
        start: 1,
        end: 3,
        gamma: &[2.0],
        beta: None,
    };
    let bound = run(&geometry, &request, &budget()).unwrap();
    assert!(bound.raw_radius_upper() >= 99.0);
    original_containment(bound.upper(), &request, &[0.25], 1e-5, false);
    let overflow = GroupSupportRequest {
        weights: &[0.0, 0.0, f64::MAX],
        gamma: &[0.0],
        ..request
    };
    let ledger = budget();
    assert!(matches!(
        run(&geometry, &overflow, &ledger),
        Err(GroupSupportError::Arithmetic(_))
    ));
    let receipt = ledger.receipt().unwrap();
    assert_eq!(
        (receipt.live_scratch_bytes, receipt.live_persistent_bytes),
        (0, 0)
    );
    // The raw radius is representable, while its learned scale is not. Even
    // an exactly zero anchor direction/image cannot hide this residual term.
    let scaled_overflow = GroupSupportRequest {
        weights: &[0.0, 0.0, 1e100],
        gamma: &[f64::MAX],
        ..request
    };
    let zero_anchor = self::geometry(&[-0.125], &[0.125], 1e-5, false);
    assert!(matches!(
        run(&zero_anchor, &scaled_overflow, &budget()),
        Err(GroupSupportError::Arithmetic("group learned residual norm"))
    ));
}

#[test]
fn decoder_n12_group_actual_visits_retained_capacity_and_nonrefundable_work_are_truthful() {
    let weights = [0.0; 12];
    let request = GroupSupportRequest {
        weights: &weights,
        bias: Some(&[0.0, -1.0, -2.0, -3.0]),
        hidden: 3,
        target: 0,
        start: 1,
        end: 4,
        gamma: &[1.0; 3],
        beta: None,
    };
    let geometry = geometry(&[0.1; 3], &[0.2; 3], 1e-5, true);
    let ledger = budget();
    let prediction = group_support_request(&request).unwrap();
    let bound = run(&geometry, &request, &ledger).unwrap();
    let stats = bound.stats();
    assert_eq!(stats.coordinate_visits, (3 + 4) * 3);
    assert_eq!(stats.original_member_visits, 9);
    assert_eq!(stats.reserved_work, prediction.work);
    assert_eq!(
        stats.consumed_work,
        stats.coordinate_visits * 32 + 3 * 128 + 256
    );
    assert!(stats.consumed_work <= stats.reserved_work);
    assert_eq!(stats.scratch_bytes, prediction.scratch_bytes);
    assert_eq!(stats.persistent_bytes, prediction.persistent_bytes);
    let live = ledger.receipt().unwrap();
    assert_eq!(live.live_scratch_bytes, 0);
    assert_eq!(live.live_persistent_bytes, 512);
    assert_eq!(live.completed, 1);
    drop(bound);
    let released = ledger.receipt().unwrap();
    assert_eq!(released.live_persistent_bytes, 0);
    assert_eq!(released.work_reserved, prediction.work);
}

#[test]
fn decoder_n12_group_original_phase_refuses_spent_work_memory_and_earlier_deadline() {
    let request = GroupSupportRequest {
        weights: &[0.0, 1.0],
        bias: None,
        hidden: 1,
        target: 0,
        start: 1,
        end: 2,
        gamma: &[1.0],
        beta: None,
    };
    let geometry = geometry(&[0.0], &[1.0], 1e-5, false);
    let prediction = group_support_request(&request).unwrap();
    let ledger = WorkBudget::new(
        None,
        prediction.scratch_bytes,
        512,
        prediction.work,
        1,
        Duration::ZERO,
    )
    .unwrap();
    let bound = run(&geometry, &request, &ledger).unwrap();
    drop(bound);
    assert!(matches!(
        run(&geometry, &request, &ledger),
        Err(GroupSupportError::Work(WorkRefusal::WorkLimit))
    ));
    let small = WorkBudget::new(
        None,
        prediction.scratch_bytes - 1,
        512,
        prediction.work,
        1,
        Duration::ZERO,
    )
    .unwrap();
    assert!(matches!(
        run(&geometry, &request, &small),
        Err(GroupSupportError::Work(WorkRefusal::ScratchLimit))
    ));
    let tiny_work = WorkRequest {
        work: 1,
        ..prediction
    };
    let ledger = budget();
    let phase = GroupSupportPhase::admit(&ledger, LeaseClass::Optional, tiny_work, None).unwrap();
    assert!(matches!(
        group_upper(&geometry, &request, phase),
        Err(GroupSupportError::Work(WorkRefusal::WorkLimit))
    ));
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert!(matches!(
        GroupSupportPhase::admit(
            &budget(),
            LeaseClass::Optional,
            prediction,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
        ),
        Err(GroupSupportError::Deadline)
    ));
}

#[test]
fn decoder_n12_group_same_authority_cancellation_between_admission_and_scan_refuses() {
    let request = GroupSupportRequest {
        weights: &[0.0, 1.0],
        bias: None,
        hidden: 1,
        target: 0,
        start: 1,
        end: 2,
        gamma: &[1.0],
        beta: None,
    };
    let prediction = group_support_request(&request).unwrap();
    let memory = MemoryAuthority::new(1 << 20);
    let cancel = CancellationToken::default();
    let ledger = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: prediction.scratch_bytes,
            persistent_bytes: 512,
            max_work: prediction.work,
            max_proposals: 1,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let phase = GroupSupportPhase::admit(&ledger, LeaseClass::Replay, prediction, None).unwrap();
    cancel.cancel();
    let geometry = geometry(&[0.0], &[1.0], 1e-5, false);
    assert!(matches!(
        group_upper(&geometry, &request, phase),
        Err(GroupSupportError::Work(WorkRefusal::Cancelled))
    ));
    let receipt = ledger.receipt().unwrap();
    assert_eq!(
        (receipt.live_scratch_bytes, receipt.live_persistent_bytes),
        (0, 0)
    );
    assert_eq!(receipt.work_reserved, prediction.work);
}

#[test]
fn decoder_n12_group_shape_and_original_region_mismatch_refuse_before_allocation() {
    let request = GroupSupportRequest {
        weights: &[0.0, 1.0],
        bias: None,
        hidden: 1,
        target: 0,
        start: 1,
        end: 2,
        gamma: &[1.0],
        beta: None,
    };
    let wrong = GroupSupportRequest { end: 3, ..request };
    assert!(matches!(
        group_support_request(&wrong),
        Err(GroupSupportError::Input(_))
    ));
    let geometry = geometry(&[0.0, 0.0], &[1.0, 1.0], 1e-5, false);
    let ledger = budget();
    assert_eq!(
        run(&geometry, &request, &ledger).unwrap_err(),
        GroupSupportError::Input("group checked region shape")
    );
    let receipt = ledger.receipt().unwrap();
    assert_eq!(
        (receipt.live_scratch_bytes, receipt.live_persistent_bytes),
        (0, 0)
    );
}

#[test]
fn decoder_n12_group_large_original_scans_stop_at_bounded_work_and_poll_quanta() {
    let h = 257;
    let weights = vec![0.25; 3 * h];
    let gamma = vec![1.0; h];
    let request = GroupSupportRequest {
        weights: &weights,
        bias: None,
        hidden: h,
        target: 0,
        start: 1,
        end: 3,
        gamma: &gamma,
        beta: None,
    };
    let geometry = geometry(&vec![0.0; h], &vec![1.0; h], 1e-5, false);
    let prediction = group_support_request(&request).unwrap();
    // A phase with only partial scan credits refuses at the entry prediction
    // before allocating/scanning; its admitted work is still nonrefundable.
    let work = (4 * h + 256) as u64 * 32 + 128 + 64;
    let reduced = WorkRequest { work, ..prediction };
    let ledger = budget();
    let phase = GroupSupportPhase::admit(&ledger, LeaseClass::Optional, reduced, None).unwrap();
    assert!(matches!(
        group_upper(&geometry, &request, phase),
        Err(GroupSupportError::Work(WorkRefusal::WorkLimit))
    ));
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.completed, 0);
    assert_eq!(receipt.work_reserved, work);
    assert_eq!(
        (receipt.live_scratch_bytes, receipt.live_persistent_bytes),
        (0, 0)
    );
    let mut largest_block = 0;
    let mut visited = 0;
    let mut stop = |count: usize| {
        largest_block = largest_block.max(count);
        visited += count;
        if visited >= 256 {
            Err(())
        } else {
            Ok(())
        }
    };
    assert!(matches!(
        geometry.normalized_anchor_image(&gamma, None, &mut stop),
        Err(SupportError::Poll(()))
    ));
    assert_eq!((largest_block, visited), (256, 256));
}

#[test]
fn decoder_n12_member_group_pairs_original_centers_and_norms_to_close_a_loose_group() {
    let request = GroupSupportRequest {
        weights: &[0.0, 0.0, 2.0],
        bias: Some(&[0.0, -0.125, -2.0]),
        hidden: 1,
        target: 0,
        start: 1,
        end: 3,
        gamma: &[1.0],
        beta: None,
    };
    let geometry = geometry(&[-0.25], &[0.25], 1.0, false);
    let old = run(&geometry, &request, &budget()).unwrap();
    let ledger = budget();
    let prediction = member_group_support_request(&request).unwrap();
    let bound = run_member(&geometry, &request, &ledger).unwrap();
    // The maximum center belongs to a constant row; the maximum norm belongs
    // to a different, strongly negative row. Combining those maxima loses it.
    assert!(old.upper() > 0.0 && bound.unpaired_upper() > 0.0);
    assert!(bound.upper() < 0.0 && bound.upper() < bound.unpaired_upper());
    for point in [-0.25, -0.125, 0.0, 0.125, 0.25] {
        original_containment(bound.upper(), &request, &[point], 1.0, false);
    }
    let stats = bound.stats();
    assert_eq!(stats.original_member_visits, 6); // center plus two norm scans
    assert!(stats.consumed_work <= prediction.work);
    assert_eq!(stats.reserved_work, prediction.work);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 512);
    drop(bound);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, prediction.work);
}

#[test]
fn decoder_n12_member_group_ln_uniform_and_zero_gamma_exclude_target_from_all_maxima() {
    let request = GroupSupportRequest {
        weights: &[0.0, 0.0, 10.0, 10.0, 20.0, 20.0],
        bias: Some(&[0.0, -1.0, -2.0]),
        hidden: 2,
        target: 0,
        start: 0,
        end: 3,
        gamma: &[1.0, 1.0],
        beta: None,
    };
    let geometry = geometry(&[-0.25; 2], &[0.25; 2], 1.0, true);
    let old = run(&geometry, &request, &budget()).unwrap();
    assert!(old.upper() > 0.0);
    let bound = run_member(&geometry, &request, &budget()).unwrap();
    assert!(bound.upper() < -0.99);
    assert_eq!(bound.competitors(), 2);
    assert_eq!(bound.range(), (0, 3));
    assert_eq!(bound.stats().original_member_visits, 14);
    for a in [-0.25, 0.0, 0.25] {
        for b in [-0.25, 0.0, 0.25] {
            original_containment(bound.upper(), &request, &[a, b], 1.0, true);
        }
    }
    // Zero learned scale retains the original beta and biases. The in-range
    // target has margin zero and must never enter even this degenerate maximum.
    let zero = GroupSupportRequest {
        gamma: &[0.0, -0.0],
        beta: Some(&[0.125, -0.125]),
        ..request
    };
    let bound = run_member(&geometry, &zero, &budget()).unwrap();
    assert!(bound.upper() < -0.99);
    for point in [[-0.25, 0.25], [0.25, -0.25], [0.0, 0.0]] {
        original_containment(bound.upper(), &zero, &point, 1.0, true);
    }
}

#[test]
fn decoder_n12_member_group_signed_gamma_subnormals_and_poison_have_original_rational_controls() {
    let weights = [1.0, -0.5, 0.25, -0.75, 0.125, 0.8, 1.25, -0.25, -0.1];
    let request = GroupSupportRequest {
        weights: &weights,
        bias: Some(&[0.25, -2.0, -1.5]),
        hidden: 3,
        target: 0,
        start: 0,
        end: 3,
        gamma: &[-1.5, 0.0, 2.0 + 2f64.powi(-41)],
        beta: Some(&[0.125, -0.25, 2f64.powi(-30)]),
    };
    let lower = [-0.375, 0.1875, -0.25];
    let upper = [0.3125, 0.4375, 0.125];
    for centered in [false, true] {
        let checked = geometry(&lower, &upper, 1e-5, centered);
        let bound = run_member(&checked, &request, &budget()).unwrap();
        for mask in 0..8 {
            let point: Vec<_> = (0..3)
                .map(|i| {
                    if mask & (1 << i) == 0 {
                        lower[i]
                    } else {
                        upper[i]
                    }
                })
                .collect();
            original_containment(bound.upper(), &request, &point, 1e-5, centered);
        }
    }
    let tiny = f64::from_bits(1);
    let tiny_request = GroupSupportRequest {
        weights: &[0.0, tiny, -tiny],
        bias: Some(&[0.0, -1.0, -1.0]),
        hidden: 1,
        target: 0,
        start: 0,
        end: 3,
        gamma: &[-1.0],
        beta: None,
    };
    for centered in [false, true] {
        for point in [-tiny, tiny] {
            let checked = geometry(&[point], &[point], 1e-5, centered);
            let bound = run_member(&checked, &tiny_request, &budget()).unwrap();
            assert!(bound.upper() < 0.0);
            original_containment(bound.upper(), &tiny_request, &[point], 1e-5, centered);
        }
    }
    let checked = geometry(&[-0.25], &[0.25], 1.0, true);
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // Validate the target too, even with zero gamma and an excluded target
        // leaf. No shortcut can turn its poisoned original row into authority.
        for weights in [[poison, 0.0, 0.0], [0.0, poison, 0.0]] {
            let bad = GroupSupportRequest {
                weights: &weights,
                gamma: &[0.0],
                ..tiny_request
            };
            let ledger = budget();
            let work = member_group_support_request(&bad).unwrap().work;
            assert!(matches!(
                run_member(&checked, &bad, &ledger),
                Err(GroupSupportError::Arithmetic(_))
            ));
            assert_eq!(ledger.receipt().unwrap().work_reserved, work);
            assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
        }
        for bad in [
            GroupSupportRequest {
                gamma: &[poison],
                ..tiny_request
            },
            GroupSupportRequest {
                beta: Some(&[poison]),
                ..tiny_request
            },
            GroupSupportRequest {
                bias: Some(&[0.0, poison, -1.0]),
                ..tiny_request
            },
        ] {
            assert!(matches!(
                run_member(&checked, &bad, &budget()),
                Err(GroupSupportError::Arithmetic(_))
            ));
        }
    }
}

#[test]
fn decoder_n12_member_group_caps_cancellation_and_phase_predictions_do_not_refill() {
    let h = 600;
    let weights = vec![0.0; 3 * h];
    let gamma = vec![1.0; h];
    let request = GroupSupportRequest {
        weights: &weights,
        bias: Some(&[0.0, -1.0, -2.0]),
        hidden: h,
        target: 0,
        start: 1,
        end: 3,
        gamma: &gamma,
        beta: None,
    };
    let checked = geometry(&vec![-0.25; h], &vec![0.25; h], 1.0, false);
    let prediction = member_group_support_request(&request).unwrap();
    let small = WorkBudget::new(
        None,
        prediction.scratch_bytes - 1,
        512,
        prediction.work,
        1,
        Duration::ZERO,
    )
    .unwrap();
    assert!(matches!(
        run_member(&checked, &request, &small),
        Err(GroupSupportError::Work(WorkRefusal::ScratchLimit))
    ));
    assert_eq!(small.receipt().unwrap().work_reserved, 0);
    // Partial credits do not admit even the first scan/allocation. The phase
    // has already spent its reservation and cannot refund or refill that work.
    let work = (h + 256) as u64 * 32 + 128 + 64;
    let ledger = budget();
    let phase = GroupSupportPhase::admit(
        &ledger,
        LeaseClass::Optional,
        WorkRequest { work, ..prediction },
        None,
    )
    .unwrap();
    assert!(matches!(
        member_group_upper(&checked, &request, phase),
        Err(GroupSupportError::Work(WorkRefusal::WorkLimit))
    ));
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.work_reserved, work);
    assert_eq!(
        (
            receipt.completed,
            receipt.live_scratch_bytes,
            receipt.live_persistent_bytes
        ),
        (0, 0, 0)
    );
    // A private zero-proposal phase cannot bypass either kernel's prediction.
    // The original production request stays unchanged at one proposal.
    for member in [false, true] {
        let required = if member {
            prediction
        } else {
            group_support_request(&request).unwrap()
        };
        let ledger = budget();
        let phase = GroupSupportPhase::admit(
            &ledger,
            LeaseClass::Optional,
            WorkRequest {
                proposals: 0,
                ..required
            },
            None,
        )
        .unwrap();
        let refused = if member {
            member_group_upper(&checked, &request, phase).map(|_| ())
        } else {
            group_upper(&checked, &request, phase).map(|_| ())
        };
        assert!(matches!(
            refused,
            Err(GroupSupportError::Work(WorkRefusal::ProposalLimit))
        ));
        let receipt = ledger.receipt().unwrap();
        assert_eq!(receipt.work_reserved, required.work);
        assert_eq!(receipt.proposals_reserved, 0);
        assert_eq!(
            (
                receipt.completed,
                receipt.live_scratch_bytes,
                receipt.live_persistent_bytes
            ),
            (0, 0, 0)
        );
    }
    assert!(matches!(
        GroupSupportPhase::admit(
            &budget(),
            LeaseClass::Replay,
            prediction,
            Some(Instant::now())
        ),
        Err(GroupSupportError::Deadline)
    ));
    let authority = MemoryAuthority::new(1 << 20);
    let cancel = CancellationToken::default();
    let ledger = WorkBudget::new_in(
        &authority,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: prediction.scratch_bytes,
            persistent_bytes: 512,
            max_work: prediction.work,
            max_proposals: 1,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let phase = GroupSupportPhase::admit(&ledger, LeaseClass::Replay, prediction, None).unwrap();
    cancel.cancel();
    assert!(matches!(
        member_group_upper(&checked, &request, phase),
        Err(GroupSupportError::Work(WorkRefusal::Cancelled))
    ));
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.work_reserved, prediction.work);
    assert_eq!(
        (receipt.live_scratch_bytes, receipt.live_persistent_bytes),
        (0, 0)
    );
}
