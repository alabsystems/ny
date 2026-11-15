// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use ndarray::{array, Array2};

use super::*;

fn budget(bytes: usize) -> WorkBudget {
    WorkBudget::new(None, 1 << 25, bytes, 1 << 32, 1_024, Duration::ZERO).unwrap()
}
fn policy(budget: &WorkBudget) -> PointLinearBuildPolicy<'_> {
    PointLinearBuildPolicy {
        budget,
        deadline: None,
        max_resident_bytes: 1 << 30,
        max_scratch_bytes: 1 << 25,
        max_work: 1 << 32,
    }
}

#[test]
fn decoder_n10_point_owner_shares_actual_coefficients_and_last_reservation() {
    let weights = Array2::from_shape_fn((256, 128), |(row, col)| (row as f32 - col as f32) * 0.125);
    let ledger = budget(1 << 26);
    let compact =
        PointLinearLayer::from_original_view(weights.view(), false, None, policy(&ledger)).unwrap();
    assert!(compact.weight().is_standard_layout());
    assert_eq!(compact.coefficient_terms(), weights.len());
    assert_eq!(compact.retained_bytes(), compact.reserved_bytes());
    assert!(compact.retained_bytes() < weights.len() * 8);
    let actual = compact.retained_bytes();
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, actual);
    let cloned = compact.clone();
    assert_eq!(cloned.weight().as_ptr(), compact.weight().as_ptr());
    assert_eq!(cloned.original_digest(), compact.original_digest());
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, actual);
    drop(compact);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, actual);
    drop(cloned);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn decoder_n10_point_owner_preserves_strided_signed_bits_orientation_and_bias() {
    let source = array![
        [0.0_f32, -0.0],
        [f32::from_bits(1), -f32::from_bits(1)],
        [f32::from_bits(0x3f80_0001), -1.0,]
    ];
    let ledger = budget(1 << 24);
    let bias = array![[-0.0_f32, f32::from_bits(1), 1.0]];
    let compact = PointLinearLayer::from_original_view(
        source.t(),
        true,
        Some(bias.view().into_dyn()),
        policy(&ledger),
    )
    .unwrap();
    assert_eq!(compact.weight().dim(), source.dim());
    for (&actual, &expected) in compact.weight().iter().zip(source.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for (&actual, &expected) in compact.bias().unwrap().iter().zip(bias.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    let contiguous = PointLinearLayer::from_original_view(
        source.view(),
        false,
        Some(bias.view().into_dyn()),
        policy(&ledger),
    )
    .unwrap();
    assert_eq!(compact.original_digest(), contiguous.original_digest());
    let mut changed = source.clone();
    changed[[0, 1]] = 0.0;
    let changed = PointLinearLayer::from_original_view(
        changed.view(),
        false,
        Some(bias.view().into_dyn()),
        policy(&ledger),
    )
    .unwrap();
    assert_ne!(compact.original_digest(), changed.original_digest());
    let scalar = ndarray::arr0(-0.25_f32);
    let broadcast = PointLinearLayer::from_original_view(
        source.view(),
        false,
        Some(scalar.view().into_dyn()),
        policy(&ledger),
    )
    .unwrap();
    assert!(broadcast.bias().unwrap().iter().all(|&v| v == -0.25));
}

#[test]
fn decoder_n10_point_owner_refuses_before_copy_and_preserves_other_live_owners() {
    let source = Array2::<f32>::zeros((32, 64));
    let required = PointLinearLayer::construction_request(32, 64, false).unwrap();
    let ledger = budget(required.persistent_bytes - 1);
    assert!(
        PointLinearLayer::from_original_view(source.view(), false, None, policy(&ledger)).is_err()
    );
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.attempted, 1);
    assert_eq!(receipt.admitted, 0);
    assert_eq!(receipt.live_persistent_bytes, 0);
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert_eq!(receipt.work_reserved, 0);

    let ledger = budget(1 << 24);
    let good =
        PointLinearLayer::from_original_view(source.view(), false, None, policy(&ledger)).unwrap();
    let retained = ledger.receipt().unwrap().live_persistent_bytes;
    let mut limits = policy(&ledger);
    limits.max_resident_bytes = required.persistent_bytes - 1;
    let attempts = ledger.receipt().unwrap().attempted;
    assert!(PointLinearLayer::from_original_view(source.view(), false, None, limits).is_err());
    assert_eq!(ledger.receipt().unwrap().attempted, attempts);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, retained);
    assert!(good.weight().iter().all(|&v| v == 0.0));
}

#[test]
fn decoder_n10_point_owner_partial_nonfinite_and_deadline_release_without_completion() {
    let ledger = budget(1 << 24);
    let mut source = Array2::<f32>::zeros((32, 256));
    source[[20, 100]] = f32::NAN;
    assert!(
        PointLinearLayer::from_original_view(source.view(), false, None, policy(&ledger)).is_err()
    );
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.admitted, 1);
    assert_eq!(receipt.completed, 0);
    assert_eq!(receipt.live_persistent_bytes, 0);
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert!(receipt.work_reserved > 0);

    source[[20, 100]] = 0.0;
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CONSTRUCTION_POLLS_FOR_TEST.with(|polls| polls.set(None));
        }
    }
    let _reset = Reset;
    // Entry, allocation, first row poll, then refuse during partial copying.
    CONSTRUCTION_POLLS_FOR_TEST.with(|polls| polls.set(Some(3)));
    assert!(matches!(
        PointLinearLayer::from_original_view(source.view(), false, None, policy(&ledger)),
        Err(NyError::DeadlineExceeded(_))
    ));
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.completed, 0);
    assert_eq!(receipt.live_persistent_bytes, 0);
    assert_eq!(receipt.live_scratch_bytes, 0);
}

#[test]
fn decoder_n10_point_only_never_mints_interval_or_crown_evidence() {
    let ledger = budget(1 << 24);
    let source = array![[1.0_f32, -1.0]];
    let layer =
        PointLinearLayer::from_original_view(source.view(), false, None, policy(&ledger)).unwrap();
    let point = BoundedTensor::concrete(array![1.0_f32, 1.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    assert!(matches!(
        layer.propagate_ibp(&point),
        Err(NyError::UnsupportedOp(_))
    ));
    assert!(matches!(
        layer.propagate_linear(&bounds),
        Err(NyError::UnsupportedOp(_))
    ));
    assert!(matches!(
        layer.propagate_linear_with_bounds(&bounds, &point),
        Err(NyError::UnsupportedOp(_))
    ));
    assert!(matches!(
        layer.propagate_crown_backward(&bounds, None),
        Err(NyError::UnsupportedOp(_))
    ));
    assert!(matches!(
        layer.propagate_crown_backward(&bounds, Some(&point)),
        Err(NyError::UnsupportedOp(_))
    ));
    let layer = crate::Layer::PointLinear(layer);
    assert!(layer.is_diagnostic_point_only());
    // This helper is EXACT constant-fold authority, not diagnostic execution.
    assert!(matches!(
        layer.propagate_concrete(array![1.0_f32, 1.0].into_dyn()),
        Err(NyError::UnsupportedOp(_))
    ));
}

#[test]
fn decoder_n10_point_scalar_original_bits_match_eager_adversarial_controls() {
    let tiny = f32::from_bits(1);
    let matrices = [
        array![[16_777_216.0_f32, 1.0, -16_777_216.0], [-1.0, 0.0, 1.0]],
        array![
            [tiny, -tiny, -0.0],
            [f32::from_bits(0x3f80_0001), -1.0, 0.0]
        ],
        array![[0.25, -0.5, 0.125], [-0.125, 0.5, -0.25]],
    ];
    for weight in matrices {
        let ledger = budget(1 << 24);
        let bias = array![-0.0_f32, tiny];
        let compact = PointLinearLayer::from_original_view(
            weight.view(),
            false,
            Some(bias.view().into_dyn()),
            policy(&ledger),
        )
        .unwrap();
        let eager = crate::layers::LinearLayer::new(weight, Some(bias)).unwrap();
        let input =
            BoundedTensor::concrete(array![[1.0_f32, 1.0, 1.0], [-1.0, 0.5, 1.0]].into_dyn())
                .unwrap();
        let deadline = Some(Instant::now() + Duration::from_secs(10));
        let (actual, receipt) = compact
            .propagate_concrete_point_with_receipt(&input, None, deadline)
            .unwrap();
        let expected = eager
            .propagate_concrete_point_with_engine_and_deadline(&input, None, deadline)
            .unwrap();
        assert_eq!(receipt.arm, PointLinearArm::ScalarF64OnceF32);
        assert_eq!(actual.shape(), expected.shape());
        for (&a, &e) in actual.lower().iter().zip(expected.lower().iter()) {
            assert_eq!(a.to_bits(), e.to_bits());
        }
        let retained = compact.retained_bytes();
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, retained);
        // A point result never causes a signed/absolute/transpose cache to appear.
        assert_eq!(compact.retained_bytes(), retained);
    }
}

#[test]
fn decoder_n10_point_original_view_matvec_has_distinct_honest_arm_and_bounded_storage() {
    let ledger = budget(1 << 26);
    let weight = Array2::from_shape_fn((128, 256), |(o, i)| {
        ((o * 17 + i * 13) % 31) as f32 * 0.015625 - 0.25
    });
    let bias = Array1::from_shape_fn(128, |i| i as f32 * 0.000_976_562_5);
    let compact = PointLinearLayer::from_original_view(
        weight.view(),
        false,
        Some(bias.view().into_dyn()),
        policy(&ledger),
    )
    .unwrap();
    let eager = crate::layers::LinearLayer::new(weight.clone(), Some(bias)).unwrap();
    assert!(compact.retained_bytes() * 5 < eager.retained_bytes().unwrap());
    let input = BoundedTensor::concrete(
        Array2::from_shape_fn((3, 256), |(r, c)| {
            ((r * 3 + c) % 19) as f32 * 0.03125 - 0.25
        })
        .into_dyn(),
    )
    .unwrap();
    let original_bytes = compact.retained_bytes();
    let (actual, receipt) = compact
        .propagate_concrete_point_with_receipt(&input, None, None)
        .unwrap();
    let expected = eager
        .propagate_concrete_point_with_engine_and_deadline(&input, None, None)
        .unwrap();
    assert_eq!(receipt.arm, PointLinearArm::OriginalRowMajorFaerMatvecF32);
    assert_ne!(
        receipt.arm.identity_name(),
        crate::layers::linear::LinearPointArm::CpuFaerF32.identity_name()
    );
    // Different physical layouts can choose different binary32 reductions.
    // This gate establishes the existing diagnostic tolerance, not bit parity.
    for (&a, &e) in actual.lower().iter().zip(expected.lower().iter()) {
        assert!(
            (a - e).abs() <= 2e-4 + 2e-4 * e.abs(),
            "compact={a} eager={e}"
        );
    }
    assert_eq!(compact.retained_bytes(), original_bytes);
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        original_bytes
    );
    let request = PointLinearLayer::execution_request(128, 256, 3, true).unwrap();
    assert!(request.work < 4 * 3 * weight.len() as u64);
    assert!(request.scratch_bytes < weight.len() * 4);
}

#[test]
fn decoder_n10_point_refuses_nondegenerate_nonfinite_and_expired_authority() {
    let ledger = budget(1 << 24);
    let weight = array![[f32::MAX]];
    let compact =
        PointLinearLayer::from_original_view(weight.view(), false, None, policy(&ledger)).unwrap();
    let interval =
        BoundedTensor::new(array![0.0_f32].into_dyn(), array![1.0_f32].into_dyn()).unwrap();
    assert!(matches!(
        compact.propagate_concrete_point_with_receipt(&interval, None, None),
        Err(NyError::InvalidSpec(_))
    ));
    let overflow = BoundedTensor::concrete(array![f32::MAX].into_dyn()).unwrap();
    assert!(matches!(
        compact.propagate_concrete_point_with_receipt(&overflow, None, None),
        Err(NyError::NumericalInstability(_))
    ));
    let point = BoundedTensor::concrete(array![1.0_f32].into_dyn()).unwrap();
    assert!(matches!(
        compact.propagate_concrete_point_with_receipt(&point, None, Some(Instant::now())),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        compact.retained_bytes()
    );
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn decoder_d4_qwen_head_finite_deadline_geometry_admits_27_refuses_28() {
    // Original Qwen0.5B head geometry. This is request admission only: no
    // coefficient/output arrays, head multiplication or numerical claim.
    const OUT: usize = 151_936;
    const INNER: usize = 896;
    const SCRATCH: usize = 256 * 1024 * 1024;
    const WORK: u64 = 1 << 34;
    let deadline = Some(Instant::now() + Duration::from_secs(60));
    assert_eq!(FINITE_POINT_MAX_ELEMENTS, 4_194_304);
    assert_eq!(27 * OUT, 4_102_272);
    assert_eq!(28 * OUT, 4_254_208);
    let request = PointLinearLayer::execution_request_with_limits(
        OUT, INNER, 27, false, deadline, SCRATCH, WORK,
    )
    .unwrap();
    assert_eq!(
        request,
        PointLinearLayer::execution_request(OUT, INNER, 27, false).unwrap()
    );
    assert_eq!(
        request.scratch_bytes,
        27 * INNER * 4 + 27 * OUT * 16 + 32 * 1024
    );
    assert_eq!(
        request.work,
        (27 * OUT * INNER * 2 + 27 * INNER * 3 + 27 * OUT * 6 + POINT_MAX_RANK * 4) as u64
    );
    assert_eq!(request.persistent_bytes, 0);
    assert!(matches!(
        PointLinearLayer::execution_request_with_limits(
            OUT, INNER, 28, false, deadline, SCRATCH, WORK,
        ),
        Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "PointLinear finite-deadline concrete point buffer",
        }) if required_bytes == 4_254_208 * 4 && budget_bytes == 4_194_304 * 4
    ));
    // The historical no-deadline arm has no finite-profile element guard.
    assert!(PointLinearLayer::execution_request_with_limits(
        OUT, INNER, 28, false, None, SCRATCH, WORK,
    )
    .is_ok());

    let original_ledger = WorkBudget::new(
        deadline,
        request.scratch_bytes,
        0,
        request.work,
        1,
        Duration::ZERO,
    )
    .unwrap();
    let phase = original_ledger.lease(request).unwrap();
    let receipt = original_ledger.receipt().unwrap();
    assert_eq!(receipt.admitted, 1);
    assert_eq!(receipt.completed, 0, "request admission is not execution");
    assert_eq!(receipt.live_scratch_bytes, request.scratch_bytes);
    assert_eq!(receipt.live_persistent_bytes, 0);
    assert_eq!(receipt.work_reserved, request.work);
    drop(phase);
    assert_eq!(original_ledger.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(
        original_ledger.receipt().unwrap().work_reserved,
        request.work
    );
    assert!(matches!(
        original_ledger.clone().lease(request),
        Err(crate::certify::work::WorkRefusal::WorkLimit)
    ));
}

#[test]
fn decoder_d4_point_execution_geometry_retains_caps_and_checks_empty_overflow() {
    let deadline = Some(Instant::now() + Duration::from_secs(60));
    let request = PointLinearLayer::execution_request(151_936, 896, 27, false).unwrap();
    for (scratch, work) in [
        (request.scratch_bytes - 1, request.work),
        (request.scratch_bytes, request.work - 1),
    ] {
        assert!(matches!(
            PointLinearLayer::execution_request_with_limits(
                151_936, 896, 27, false, deadline, scratch, work,
            ),
            Err(NyError::UnsupportedConfiguration(_))
        ));
    }
    for (out, inner, rows) in [
        (0, 896, 27),
        (151_936, 0, 27),
        (151_936, 896, 0),
        (usize::MAX, 2, 1),
        (1, usize::MAX, 2),
    ] {
        assert!(matches!(
            PointLinearLayer::execution_request_with_limits(
                out,
                inner,
                rows,
                false,
                deadline,
                usize::MAX,
                u64::MAX,
            ),
            Err(NyError::InvalidSpec(_))
        ));
    }
    // Both input and output buffers use the same inclusive 4Mi profile.
    for (out, inner) in [
        (FINITE_POINT_MAX_ELEMENTS, 1),
        (1, FINITE_POINT_MAX_ELEMENTS),
    ] {
        assert!(PointLinearLayer::execution_request_with_limits(
            out,
            inner,
            1,
            true,
            deadline,
            usize::MAX,
            u64::MAX,
        )
        .is_ok());
    }
    for (out, inner) in [
        (FINITE_POINT_MAX_ELEMENTS + 1, 1),
        (1, FINITE_POINT_MAX_ELEMENTS + 1),
    ] {
        assert!(matches!(
            PointLinearLayer::execution_request_with_limits(
                out,
                inner,
                1,
                true,
                deadline,
                usize::MAX,
                u64::MAX,
            ),
            Err(NyError::CpuMemoryExceeded { .. })
        ));
    }
}

#[test]
fn decoder_d4_point_execution_inherits_original_deadline_and_expiry_refuses() {
    let original_ledger = budget(1 << 24);
    let weights = array![[1.0_f32]];
    let original_deadline = Some(Instant::now() + Duration::from_secs(60));
    let owner = PointLinearLayer::from_original_view(
        weights.view(),
        false,
        None,
        PointLinearBuildPolicy {
            deadline: original_deadline,
            ..policy(&original_ledger)
        },
    )
    .unwrap();
    let point = BoundedTensor::concrete(array![0.25_f32].into_dyn()).unwrap();
    assert_eq!(owner.original_deadline(), original_deadline);
    let (output, receipt) = owner
        .propagate_concrete_point_with_receipt(&point, None, None)
        .unwrap();
    assert_eq!(receipt.arm, PointLinearArm::ScalarF64OnceF32);
    assert_eq!(output.lower()[[0]].to_bits(), 0.25_f32.to_bits());
    let before = original_ledger.receipt().unwrap();
    assert!(matches!(
        owner.propagate_concrete_point_with_receipt(
            &point,
            None,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(original_ledger.receipt().unwrap(), before);
    assert_eq!(owner.original_deadline(), original_deadline);
}
