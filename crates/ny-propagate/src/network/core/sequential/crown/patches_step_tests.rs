// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for patches-aware CROWN backward-step dispatch.
//!
//! Tests the dispatch logic in `crown_backward_step_patches`:
//! - Dense mode delegates correctly to `crown_backward_step`
//! - Patches→Dense termination for structural layers (Linear, Flatten)
//! - Patches-native activation dispatch (ReLU stays in Patches mode)

use super::*;
use crate::bounds::patches::{
    CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds, UnstableIdx,
};
use crate::bounds::LinearBounds;
use crate::layers::{
    AddLayer, BatchNormLayer, Conv2dLayer, FlattenLayer, Layer, LinearLayer, MulBinaryLayer,
    ReLULayer, SkipMergeLayer,
};
use crate::BoundPropagation;
use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;
use std::time::{Duration, Instant};

/// Helper: create pre-activation BoundedTensor from 1D arrays.
fn bounded_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_vec(lower.to_vec()).into_dyn(),
        Array1::from_vec(upper.to_vec()).into_dyn(),
    )
    .expect("valid bounds")
}

/// Helper: create pre-activation BoundedTensor from 3D shape (C, H, W).
fn bounded_3d(shape: (usize, usize, usize), lower_val: f32, upper_val: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[shape.0, shape.1, shape.2]), lower_val),
        ArrayD::from_elem(IxDyn(&[shape.0, shape.1, shape.2]), upper_val),
    )
    .expect("valid bounds")
}

// ── Dense-mode pass-through tests ─────────────────────────────────────────
// When CrownBounds is already Dense, patches_step delegates to crown_backward_step.

/// Dense + ReLU fully active: identity pass-through (same as backward_step).
///
/// Verifies that crown_backward_step_patches correctly delegates Dense-mode
/// ReLU backward to crown_backward_step, which uses the trait path.
#[test]
fn test_patches_step_dense_relu_fully_active() -> Result<()> {
    let layer = Layer::ReLU(ReLULayer::new());
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(2));
    let pre_act = bounded_1d(&[1.0, 2.0], &[3.0, 4.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Dense ReLU fully active should Continue"
    );

    // Verify bounds are still Dense.
    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense, got Patches"),
    };
    // Fully active ReLU = identity: A-matrices unchanged.
    assert!((lb.lower_a()[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((lb.lower_a()[[1, 1]] - 1.0).abs() < 1e-6);
    assert!(lb.lower_a()[[0, 1]].abs() < 1e-6);
    assert!(lb.lower_a()[[1, 0]].abs() < 1e-6);
    Ok(())
}

/// Dense + Linear: weight matrix composition through patches dispatch.
///
/// Verifies that the patches step correctly delegates Dense-mode Linear
/// backward to crown_backward_step, producing A_new = I @ W = W.
#[test]
fn test_patches_step_dense_linear_composes_weight() -> Result<()> {
    let weight = arr2(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let bias = arr1(&[0.1f32, 0.2, 0.3]);
    let layer = Layer::Linear(LinearLayer::new(weight.clone(), Some(bias))?);
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(3));
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Dense Linear should Continue"
    );

    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense, got Patches"),
    };
    assert_eq!(lb.num_outputs(), 3);
    assert_eq!(lb.num_inputs(), 2);

    // A_new = I @ W = W.
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (lb.lower_a()[[i, j]] - weight[[i, j]]).abs() < 1e-5,
                "lower_a[{i},{j}] = {} != weight = {}",
                lb.lower_a()[[i, j]],
                weight[[i, j]]
            );
        }
    }
    Ok(())
}

#[test]
fn dense_linear_preserves_expired_deadline_without_engine_launch() -> Result<()> {
    let layer = Layer::Linear(LinearLayer::new(arr2(&[[1.0f32]]), None)?);
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(1));
    let pre_act = bounded_1d(&[-1.0], &[1.0]);
    let engine = CountingGemmEngine::new();
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("one millisecond fits before the current instant");

    let error = match crown_backward_step_patches(
        &layer,
        &mut bounds,
        &pre_act,
        Some(&engine),
        0,
        "test",
        Some(expired),
    ) {
        Err(error) => error,
        Ok(_) => panic!("expired dense Linear dispatch must remain structured"),
    };

    assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
    assert_eq!(
        engine.gemm_calls(),
        0,
        "expired Dense delegation must not launch the caller engine"
    );
    Ok(())
}

#[test]
fn finite_deadline_conv_uses_typed_fallback_and_preserves_expiry() -> Result<()> {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0f32]).expect("valid Conv2d kernel");
    let layer = Layer::Conv2d(Conv2dLayer::with_input_shape(
        kernel,
        None,
        (1, 1),
        (0, 0),
        1,
        1,
    )?);
    let make_bounds = || {
        CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
            (1, 2, 2),
            (1, 2, 2),
        )))
    };
    // A 1x1 spatial fixture crosses the separate Patches-to-Dense memory
    // boundary (kernel area == input area). Use 2x2 so this test specifically
    // reaches the finite-authority Dense dispatcher and its typed refusal.
    let pre_act = bounded_3d((1, 2, 2), -1.0, 1.0);
    let engine = CountingGemmEngine::new();

    let mut expired_bounds = make_bounds();
    let error = match crown_backward_step_patches(
        &layer,
        &mut expired_bounds,
        &pre_act,
        Some(&engine),
        0,
        "test",
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("one millisecond fits before the current instant"),
        ),
    ) {
        Err(error) => error,
        Ok(_) => panic!("expired finite Conv2d must remain structured"),
    };
    assert!(error.is_deadline_exceeded(), "unexpected error: {error}");

    let mut bounds = make_bounds();
    let result = crown_backward_step_patches(
        &layer,
        &mut bounds,
        &pre_act,
        Some(&engine),
        0,
        "test",
        Some(Instant::now() + Duration::from_secs(30)),
    )?;
    match result {
        CrownStepResult::IbpFallback(fallback) => {
            assert_eq!(
                fallback.reason,
                CrownIbpFallbackReason::CrownPropagationError
            );
            assert!(
                fallback.details.contains("finite") && fallback.details.contains("Conv2d"),
                "typed finite refusal must retain operator context: {}",
                fallback.details
            );
        }
        CrownStepResult::Continue => {
            panic!("finite Conv2d must not enter the partially cooperative dense kernel")
        }
    }
    assert_eq!(
        engine.gemm_calls(),
        0,
        "finite Conv2d refusal must not launch the caller engine"
    );
    Ok(())
}

#[test]
fn explicit_row_relu_deadline_gate_is_exact_and_default_dark() {
    crate::tests::with_env_edits(|env| {
        env.remove("NY_PATCHES_DEADLINE_RELU");
        assert!(!patches_deadline_relu_enabled());
        env.set("NY_PATCHES_DEADLINE_RELU", "0");
        assert!(!patches_deadline_relu_enabled());
        env.set("NY_PATCHES_DEADLINE_RELU", "");
        assert!(!patches_deadline_relu_enabled());
        env.set("NY_PATCHES_DEADLINE_RELU", "true");
        assert!(!patches_deadline_relu_enabled());
        env.set("NY_PATCHES_DEADLINE_RELU", "1");
        assert!(patches_deadline_relu_enabled());
    });
}

fn explicit_row_relu_bounds() -> CrownBounds {
    let make_side = |value: f32| PatchesData {
        coeff_err: Some(Array1::from_vec(vec![1.0e-4])),
        patches: Some(ArrayD::from_elem(IxDyn(&[1, 1, 1, 1, 1, 1, 1]), value)),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: (1, 1, 1),
        input_shape: (1, 1, 1),
        unstable_idx: None,
    };
    CrownBounds::Patches(Box::new(PatchesLinearBounds {
        row_count: 1,
        lower_a: make_side(-0.75),
        lower_b: Array1::from_vec(vec![0.25]),
        upper_a: make_side(1.25),
        upper_b: Array1::from_vec(vec![-0.5]),
    }))
}

fn f32_bits<'a>(values: impl IntoIterator<Item = &'a f32>) -> Vec<u32> {
    values.into_iter().map(|value| value.to_bits()).collect()
}

fn assert_patches_data_bitwise_eq(actual: &PatchesData, expected: &PatchesData, label: &str) {
    assert_eq!(actual.geometry, expected.geometry, "{label}: geometry");
    assert_eq!(actual.identity, expected.identity, "{label}: identity");
    assert_eq!(
        actual.output_shape, expected.output_shape,
        "{label}: output shape"
    );
    assert_eq!(
        actual.input_shape, expected.input_shape,
        "{label}: input shape"
    );
    match (&actual.patches, &expected.patches) {
        (Some(actual), Some(expected)) => {
            assert_eq!(actual.shape(), expected.shape(), "{label}: patches shape");
            assert_eq!(
                f32_bits(actual),
                f32_bits(expected),
                "{label}: patches bits"
            );
        }
        (None, None) => {}
        _ => panic!("{label}: patches materialization changed"),
    }
    match (&actual.coeff_err, &expected.coeff_err) {
        (Some(actual), Some(expected)) => {
            assert_eq!(
                f32_bits(actual),
                f32_bits(expected),
                "{label}: coefficient-error bits"
            );
        }
        (None, None) => {}
        _ => panic!("{label}: coefficient-error presence changed"),
    }
    match (&actual.unstable_idx, &expected.unstable_idx) {
        (Some(actual), Some(expected)) => {
            assert_eq!(actual.channels, expected.channels, "{label}: channels");
            assert_eq!(actual.heights, expected.heights, "{label}: heights");
            assert_eq!(actual.widths, expected.widths, "{label}: widths");
        }
        (None, None) => {}
        _ => panic!("{label}: sparse-index presence changed"),
    }
}

fn assert_patches_bounds_bitwise_eq(actual: &CrownBounds, expected: &CrownBounds, label: &str) {
    let (CrownBounds::Patches(actual), CrownBounds::Patches(expected)) = (actual, expected) else {
        panic!("{label}: expected Patches on both sides");
    };
    assert_eq!(actual.row_count, expected.row_count, "{label}: row count");
    assert_patches_data_bitwise_eq(&actual.lower_a, &expected.lower_a, "lower coefficients");
    assert_patches_data_bitwise_eq(&actual.upper_a, &expected.upper_a, "upper coefficients");
    assert_eq!(
        f32_bits(&actual.lower_b),
        f32_bits(&expected.lower_b),
        "{label}: lower-bias bits"
    );
    assert_eq!(
        f32_bits(&actual.upper_b),
        f32_bits(&expected.upper_b),
        "{label}: upper-bias bits"
    );
}

fn one_mib_pair_anchored_bounds() -> CrownBounds {
    // One row over 131,072 input columns is exactly a 1 MiB lower/upper f32
    // Dense pair. The 131,072-tap anchored unfold plan is larger than that, so
    // the pair-only outer gate admits equality while the full materialization
    // preflight must refuse under the same 1 MiB budget.
    const INPUT_HEIGHT: usize = 131_072;
    let geometry = PatchGeometry::anchored(vec![0], vec![0]).expect("fixture axes are non-empty");
    let make_side = |value: f32| PatchesData {
        coeff_err: None,
        patches: Some(ArrayD::from_elem(
            IxDyn(&[1, 1, 1, 1, INPUT_HEIGHT, 1]),
            value,
        )),
        geometry: geometry.clone(),
        identity: false,
        output_shape: (1, 1, 1),
        input_shape: (1, INPUT_HEIGHT, 1),
        unstable_idx: None,
    };
    CrownBounds::Patches(Box::new(PatchesLinearBounds {
        row_count: 1,
        lower_a: make_side(-0.75),
        lower_b: arr1(&[0.25]),
        upper_a: make_side(1.25),
        upper_b: arr1(&[-0.5]),
    }))
}

#[test]
fn ordinary_wrapper_maps_full_peak_memory_refusal_without_mutating_anchored_carrier() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "1");

        let mut bounds = one_mib_pair_anchored_bounds();
        let before = bounds.clone();
        let layer = Layer::Flatten(FlattenLayer::new(0));
        let pre_activation = bounded_3d((1, 131_072, 1), -1.0, 1.0);

        let result = crown_backward_step_patches(
            &layer,
            &mut bounds,
            &pre_activation,
            None,
            0,
            "test",
            None,
        )
        .expect("ordinary wrapper must convert a typed memory refusal into IBP fallback");
        match result {
            CrownStepResult::IbpFallback(fallback) => assert_eq!(
                fallback.reason,
                CrownIbpFallbackReason::MemoryBudgetExceeded
            ),
            CrownStepResult::Continue => panic!("over-budget full peak must not continue"),
        }
        assert_patches_bounds_bitwise_eq(&bounds, &before, "ordinary full-peak memory refusal");
    });
}

#[test]
fn batchnorm_no_deadline_memory_refusal_never_retries_dense_and_preserves_carrier() {
    crate::tests::with_env_edits(|env| {
        use crate::bounds::patches::{
            patches_to_dense_call_sites, reset_patches_to_dense_call_count,
        };

        // At 65,000 taps, Dense needs only 520,000 bytes and the generic retry
        // gate would admit it under 1 MiB. Anchored BN's authoritative
        // total-live receipt includes the 520,000-byte source plus ~585,000
        // bytes of output/map scratch, so the native operation must refuse.
        const WIDTH: usize = 65_000;
        env.set("NY_DENSE_BUDGET_MB", "1");
        let geometry = PatchGeometry::anchored(vec![0], vec![0]).unwrap();
        let make_side = |value: f32| PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[1, 1, 1, 1, 1, WIDTH]), value)),
            geometry: geometry.clone(),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, WIDTH),
            unstable_idx: None,
        };
        let mut bounds = CrownBounds::Patches(Box::new(PatchesLinearBounds {
            row_count: 1,
            lower_a: make_side(-0.75),
            lower_b: arr1(&[0.25]),
            upper_a: make_side(1.25),
            upper_b: arr1(&[-0.5]),
        }));
        let before = bounds.clone();
        let layer = Layer::BatchNorm(
            BatchNormLayer::from_scale_bias(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.75]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.125]).unwrap(),
            )
            .unwrap(),
        );
        let pre_activation = bounded_3d((1, 1, WIDTH), -1.0, 1.0);

        reset_patches_to_dense_call_count();
        let result = crown_backward_step_patches(
            &layer,
            &mut bounds,
            &pre_activation,
            None,
            0,
            "test",
            None,
        )
        .expect("typed BN memory refusal must map directly to IBP fallback");
        match result {
            CrownStepResult::IbpFallback(fallback) => assert_eq!(
                fallback.reason,
                CrownIbpFallbackReason::MemoryBudgetExceeded
            ),
            CrownStepResult::Continue => panic!("BN memory refusal must not continue via Dense"),
        }
        assert!(
            patches_to_dense_call_sites().is_empty(),
            "authoritative BN receipt refusal attempted Dense materialization"
        );
        assert_patches_bounds_bitwise_eq(&bounds, &before, "no-deadline BN memory refusal");
    });
}

/// The graph collector always gives patches-startable nodes an aggregate
/// scheduling timestamp, even when the caller supplied no outer deadline. That
/// soft timestamp must remain visible to cooperative kernels without turning
/// into authority to densify an otherwise memory-admissible native Conv2d walk.
#[test]
fn collector_soft_deadline_keeps_native_conv_patches_route() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "1");

        let conv = Conv2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 0.25_f32),
            Some(arr1(&[0.0_f32])),
            (1, 1),
            (0, 0),
            33,
            33,
        )
        .expect("valid spatial conv");
        let layer = Layer::Conv2d(conv);
        let pre_activation = bounded_3d((1, 33, 33), -1.0, 1.0);
        let identity = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
            (1, 32, 32),
            (1, 32, 32),
        )));
        let soft_deadline = Some(Instant::now() + Duration::from_secs(30));

        let mut soft = identity.clone();
        let result = crown_backward_step_patches_with_deadline_authority(
            &layer,
            &mut soft,
            &pre_activation,
            None,
            0,
            "collector-soft-budget",
            soft_deadline,
            false,
        )
        .expect("soft collector budget must keep the native Patches route");
        assert!(matches!(result, CrownStepResult::Continue));
        assert!(matches!(soft, CrownBounds::Patches(_)));

        let mut hard = identity;
        let result = crown_backward_step_patches_with_deadline_authority(
            &layer,
            &mut hard,
            &pre_activation,
            None,
            0,
            "caller-hard-deadline",
            soft_deadline,
            true,
        )
        .expect("hard authority must retain the typed memory fallback");
        assert!(matches!(
            result,
            CrownStepResult::IbpFallback(CrownStepFallback {
                reason: CrownIbpFallbackReason::MemoryBudgetExceeded,
                ..
            })
        ));
    });
}

#[test]
fn collector_soft_deadline_keeps_native_affine_batchnorm_route() -> Result<()> {
    use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};

    let layer = Layer::BatchNorm(BatchNormLayer::from_scale_bias(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.75]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.125]).unwrap(),
    )?);
    let pre_activation = bounded_3d((1, 2, 2), -1.0, 1.0);
    let identity = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (1, 2, 2),
        (1, 2, 2),
    )));

    let mut historical = identity.clone();
    let historical_result = crown_backward_step_patches_with_deadline_authority(
        &layer,
        &mut historical,
        &pre_activation,
        None,
        0,
        "historical-no-deadline",
        None,
        false,
    )?;
    assert!(matches!(historical_result, CrownStepResult::Continue));
    assert!(matches!(historical, CrownBounds::Patches(_)));

    reset_patches_to_dense_call_count();
    let mut soft = identity;
    let soft_result = crown_backward_step_patches_with_deadline_authority(
        &layer,
        &mut soft,
        &pre_activation,
        None,
        0,
        "collector-soft-budget",
        Some(Instant::now() + Duration::from_secs(30)),
        false,
    )?;
    assert!(matches!(soft_result, CrownStepResult::Continue));
    assert_patches_bounds_bitwise_eq(&soft, &historical, "soft affine BN route");
    assert!(
        patches_to_dense_call_sites().is_empty(),
        "soft affine BN scheduling authority attempted Dense materialization"
    );
    Ok(())
}

/// A collector-local Patches budget is scheduling authority, not permission to
/// replace the historical affine ReLU walk with an O(rows * columns) Dense
/// materialization. The soft route must remain bit-identical to no deadline,
/// while an already-expired timestamp must refuse before publishing anything.
#[test]
fn collector_soft_deadline_keeps_historical_relu_patches_route_atomically() -> Result<()> {
    use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};

    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "2");

        // A Dense 1024x1024 lower/upper pair is 8 MiB, so any accidental
        // generic Dense dispatch is refused under this 2 MiB budget. The
        // native identity-ReLU Patches result remains only O(spatial).
        let spatial = (1, 32, 32);
        let identity =
            CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(spatial, spatial)));
        let layer = Layer::ReLU(ReLULayer::new());
        let pre_activation = bounded_3d(spatial, -1.0, 2.0);

        let mut historical = identity.clone();
        let historical_result = crown_backward_step_patches_with_deadline_authority(
            &layer,
            &mut historical,
            &pre_activation,
            None,
            0,
            "historical-no-deadline",
            None,
            false,
        )?;
        assert!(matches!(historical_result, CrownStepResult::Continue));
        assert!(matches!(historical, CrownBounds::Patches(_)));

        reset_patches_to_dense_call_count();
        let mut soft = identity.clone();
        let soft_result = crown_backward_step_patches_with_deadline_authority(
            &layer,
            &mut soft,
            &pre_activation,
            None,
            0,
            "collector-soft-budget",
            Some(Instant::now() + Duration::from_secs(30)),
            false,
        )?;
        assert!(matches!(soft_result, CrownStepResult::Continue));
        assert_patches_bounds_bitwise_eq(&soft, &historical, "soft ReLU route");
        assert!(
            patches_to_dense_call_sites().is_empty(),
            "soft ReLU scheduling authority attempted Dense materialization"
        );

        reset_patches_to_dense_call_count();
        let mut expired = identity;
        let before = expired.clone();
        let limit = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before the current instant");
        let error = match crown_backward_step_patches_with_deadline_authority(
            &layer,
            &mut expired,
            &pre_activation,
            None,
            0,
            "collector-expired-soft-budget",
            Some(limit),
            false,
        ) {
            Err(error) => error,
            Ok(_) => panic!("expired soft scheduling authority must be terminal"),
        };
        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert_patches_bounds_bitwise_eq(&expired, &before, "expired soft ReLU route");
        assert!(
            patches_to_dense_call_sites().is_empty(),
            "expired soft ReLU route attempted Dense materialization"
        );
        Ok(())
    })
}

#[test]
fn ordinary_face_honors_expiry_atomically_and_ignores_spec_gate_without_deadline() -> Result<()> {
    crate::tests::with_env_edits(|env| {
        let layer = Layer::ReLU(ReLULayer::new());
        let pre_act = bounded_3d((1, 1, 1), -1.0, 2.0);
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before the current instant");

        env.remove("NY_PATCHES_DEADLINE_RELU");
        let mut historical = explicit_row_relu_bounds();
        let historical_result = crown_backward_step_patches(
            &layer,
            &mut historical,
            &pre_act,
            None,
            0,
            "SPEC-CROWN",
            None,
        )?;
        assert!(matches!(historical_result, CrownStepResult::Continue));

        env.set("NY_PATCHES_DEADLINE_RELU", "1");
        let mut actual = explicit_row_relu_bounds();
        let actual_result = crown_backward_step_patches(
            &layer,
            &mut actual,
            &pre_act,
            None,
            0,
            "SPEC-CROWN",
            None,
        )?;
        assert!(matches!(actual_result, CrownStepResult::Continue));
        assert_patches_bounds_bitwise_eq(&actual, &historical, "ordinary ReLU face");

        env.remove("NY_PATCHES_DEADLINE_RELU");
        let mut expired_bounds = explicit_row_relu_bounds();
        let before = expired_bounds.clone();
        let error = match crown_backward_step_patches(
            &layer,
            &mut expired_bounds,
            &pre_act,
            None,
            0,
            "SPEC-CROWN",
            Some(expired),
        ) {
            Err(error) => error,
            Ok(_) => panic!("expired ordinary ReLU dispatch must remain structured"),
        };
        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert_patches_bounds_bitwise_eq(
            &expired_bounds,
            &before,
            "expired ordinary ReLU transaction",
        );
        Ok(())
    })
}

#[test]
fn spec_crown_wrapper_expiry_preserves_input_bitwise() -> Result<()> {
    crate::tests::with_env_edits(|env| {
        env.set("NY_PATCHES_DEADLINE_RELU", "1");
        let mut bounds = explicit_row_relu_bounds();
        let before = bounds.clone();
        let layer = Layer::ReLU(ReLULayer::new());
        let pre_act = bounded_3d((1, 1, 1), -1.0, 2.0);
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before the current instant");

        let error = match crown_backward_step_patches_spec_crown(
            &layer,
            &mut bounds,
            &pre_act,
            None,
            0,
            "SPEC-CROWN",
            Some(expired),
        ) {
            Err(error) => error,
            Ok(_) => panic!("gated expired ReLU must preserve structured deadline authority"),
        };
        assert!(matches!(error, SpecPatchesStepError::ReluDeadlineExceeded));
        assert_patches_bounds_bitwise_eq(&bounds, &before, "expired Spec-CROWN transaction");
        Ok(())
    })
}

/// Dense + Add: multi-input ops return IbpFallback through patches dispatch.
///
/// Sequential networks can't resolve second inputs for binary ops.
/// Verifies that patches_step delegates correctly and the IbpFallback propagates.
#[test]
fn test_patches_step_dense_add_returns_ibp_fallback() -> Result<()> {
    let layer = Layer::Add(AddLayer);
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(2));
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::IbpFallback(_)),
        "Dense Add must return IbpFallback in sequential CROWN"
    );
    Ok(())
}

/// Dense + MulBinary: binary ops return IbpFallback through patches dispatch.
#[test]
fn test_patches_step_dense_mul_binary_returns_ibp_fallback() -> Result<()> {
    let layer = Layer::MulBinary(MulBinaryLayer);
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(2));
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::IbpFallback(_)),
        "Dense MulBinary must return IbpFallback in sequential CROWN"
    );
    Ok(())
}

/// Dense + SkipMerge: identity pass-through preserves bounds unchanged.
#[test]
fn test_patches_step_dense_skip_merge_passthrough() -> Result<()> {
    let layer = Layer::SkipMerge(SkipMergeLayer::new());
    let original = LinearBounds::new(
        arr2(&[[2.0, 3.0]]),
        arr1(&[0.5]),
        arr2(&[[4.0, 5.0]]),
        arr1(&[0.7]),
    )?;
    let mut bounds = CrownBounds::Dense(original.clone());
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Dense SkipMerge should Continue"
    );

    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense, got Patches"),
    };
    assert_eq!(lb.lower_a(), original.lower_a(), "lower_a unchanged");
    assert_eq!(lb.upper_a(), original.upper_a(), "upper_a unchanged");
    assert_eq!(lb.lower_b(), original.lower_b(), "lower_b unchanged");
    assert_eq!(lb.upper_b(), original.upper_b(), "upper_b unchanged");
    Ok(())
}

// ── Patches→Dense termination tests ───────────────────────────────────────
// Structural layers (Linear, Flatten, Reshape) terminate Patches mode by
// converting to Dense before dispatch.

/// Patches + Linear → Patches→Dense termination, then standard Linear backward.
///
/// Verifies that crown_backward_step_patches converts Patches to Dense
/// at a Linear layer boundary, then dispatches the Dense backward.
#[test]
fn test_patches_step_patches_to_dense_termination_linear() -> Result<()> {
    // Identity patches for spatial tensor (C=1, H=1, W=2) → 2 elements.
    let spatial = (1, 1, 2);
    let pb = PatchesLinearBounds::identity(spatial, spatial);
    let mut bounds = CrownBounds::Patches(Box::new(pb));

    // Linear: 2→2 identity weight, zero bias.
    let weight = arr2(&[[1.0f32, 0.0], [0.0, 1.0]]);
    let layer = Layer::Linear(LinearLayer::new(weight, None)?);
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Patches→Dense Linear should Continue"
    );

    // After termination: bounds must be Dense.
    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense after Linear termination"),
    };

    // Identity Linear backward through identity patches = identity.
    assert_eq!(lb.num_outputs(), 2);
    assert_eq!(lb.num_inputs(), 2);
    assert!(
        (lb.lower_a()[[0, 0]] - 1.0).abs() < 1e-5,
        "identity preserved: lower_a[0,0] = {}",
        lb.lower_a()[[0, 0]]
    );
    assert!(
        (lb.lower_a()[[1, 1]] - 1.0).abs() < 1e-5,
        "identity preserved: lower_a[1,1] = {}",
        lb.lower_a()[[1, 1]]
    );
    Ok(())
}

// ── Patches-native activation dispatch tests ──────────────────────────────
// Element-wise activations in Patches mode use patches-native backward,
// preserving the sparse structure.

/// Patches + ReLU fully active: stays in Patches mode (patches-native dispatch).
///
/// When pre-activation bounds are all positive, ReLU is identity and the
/// patches activation backward should preserve Patches structure.
/// Reference: crown_elementwise_backward_patches applies per-element slopes.
#[test]
fn test_patches_step_patches_relu_fully_active_stays_patches() -> Result<()> {
    let spatial = (1, 1, 2);
    let pb = PatchesLinearBounds::identity(spatial, spatial);
    let mut bounds = CrownBounds::Patches(Box::new(pb));

    let layer = Layer::ReLU(ReLULayer::new());
    // Fully active: all lower bounds positive.
    let pre_act = bounded_3d(spatial, 1.0, 3.0);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Patches ReLU fully active should Continue"
    );

    // Fully active ReLU should stay in Patches mode (no Dense conversion needed).
    assert!(
        matches!(bounds, CrownBounds::Patches(_)),
        "ReLU fully active should preserve Patches mode"
    );
    Ok(())
}

/// Patches + ReLU fully inactive: stays in Patches mode with zero coefficients.
///
/// When all pre-activation upper bounds are non-positive, ReLU output is zero.
/// The patches backward should zero out the A-matrices while staying in Patches mode.
#[test]
fn test_patches_step_patches_relu_fully_inactive_stays_patches() -> Result<()> {
    let spatial = (1, 1, 2);
    let pb = PatchesLinearBounds::identity(spatial, spatial);
    let mut bounds = CrownBounds::Patches(Box::new(pb));

    let layer = Layer::ReLU(ReLULayer::new());
    // Fully inactive: all upper bounds non-positive.
    let pre_act = bounded_3d(spatial, -3.0, -1.0);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Patches ReLU fully inactive should Continue"
    );

    // Should stay in Patches mode.
    assert!(
        matches!(bounds, CrownBounds::Patches(_)),
        "ReLU fully inactive should preserve Patches mode"
    );

    // Verify zero output: convert to Dense and check A-matrices are zero.
    let lb = bounds.ensure_dense()?;
    assert!(
        lb.lower_a().iter().all(|&v| v.abs() < 1e-6),
        "fully inactive ReLU: lower_a should be zero, got {:?}",
        lb.lower_a()
    );
    assert!(
        lb.upper_a().iter().all(|&v| v.abs() < 1e-6),
        "fully inactive ReLU: upper_a should be zero, got {:?}",
        lb.upper_a()
    );
    Ok(())
}

#[test]
fn sparse_relu_refusal_falls_back_to_the_same_dense_result() -> Result<()> {
    let sparse = PatchesLinearBounds::sparse_identity(
        (1, 1, 3),
        (1, 1, 3),
        UnstableIdx {
            channels: vec![0, 0],
            heights: vec![0, 0],
            widths: vec![0, 2],
        },
    );
    let layer = Layer::ReLU(ReLULayer::new());
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![-1.0, -2.0, -3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![2.0, 4.0, 6.0]).unwrap(),
    )
    .expect("valid three-neuron pre-activation box");

    let mut expected = CrownBounds::Patches(Box::new(sparse.clone()));
    let expected_dense = expected
        .ensure_dense()
        .expect("valid sparse identity must materialize for the reference");
    crown_backward_step(&layer, expected_dense, &pre_act, None, 0, "test", None)
        .expect("dense ReLU reference must compose");

    let mut actual = CrownBounds::Patches(Box::new(sparse));
    let disposition =
        crown_backward_step_patches(&layer, &mut actual, &pre_act, None, 0, "test", None)
            .expect("sparse refusal must reach the dense fallback");
    assert!(matches!(disposition, CrownStepResult::Continue));

    let (CrownBounds::Dense(actual), CrownBounds::Dense(expected)) = (&actual, &expected) else {
        panic!("sparse activation refusal must publish the dense fallback");
    };
    assert_eq!(
        actual.lower_a().mapv(f32::to_bits),
        expected.lower_a().mapv(f32::to_bits)
    );
    assert_eq!(
        actual.upper_a().mapv(f32::to_bits),
        expected.upper_a().mapv(f32::to_bits)
    );
    assert_eq!(
        actual.lower_b().mapv(f32::to_bits),
        expected.lower_b().mapv(f32::to_bits)
    );
    assert_eq!(
        actual.upper_b().mapv(f32::to_bits),
        expected.upper_b().mapv(f32::to_bits)
    );
    Ok(())
}

// ── Dense-mode end-to-end soundness ───────────────────────────────────────

/// Verify CROWN bounds contain true ReLU(Wx+b) output for 25 sampled inputs.
fn assert_linear_relu_soundness(
    crown_flat: &BoundedTensor,
    weight: &ndarray::Array2<f32>,
    bias: &Array1<f32>,
) {
    for xi in 0..5 {
        for xj in 0..5 {
            let x0 = -1.0 + 2.0 * (xi as f32) / 4.0;
            let x1 = -1.0 + 2.0 * (xj as f32) / 4.0;
            let z0 = weight[[0, 0]] * x0 + weight[[0, 1]] * x1 + bias[0];
            let z1 = weight[[1, 0]] * x0 + weight[[1, 1]] * x1 + bias[1];
            let y0 = z0.max(0.0);
            let y1 = z1.max(0.0);
            assert!(
                crown_flat.lower()[[0]] <= y0 + 1e-5,
                "lower[0]={} > true y0={y0} at ({x0},{x1})",
                crown_flat.lower()[[0]]
            );
            assert!(
                crown_flat.upper()[[0]] >= y0 - 1e-5,
                "upper[0]={} < true y0={y0} at ({x0},{x1})",
                crown_flat.upper()[[0]]
            );
            assert!(
                crown_flat.lower()[[1]] <= y1 + 1e-5,
                "lower[1]={} > true y1={y1} at ({x0},{x1})",
                crown_flat.lower()[[1]]
            );
            assert!(
                crown_flat.upper()[[1]] >= y1 - 1e-5,
                "upper[1]={} < true y1={y1} at ({x0},{x1})",
                crown_flat.upper()[[1]]
            );
        }
    }
}

/// Dense-mode Linear→ReLU backward through patches dispatch must produce
/// sound bounds (contain true output for all inputs in domain).
///
/// Same soundness check as backward_step_tests but going through the
/// patches dispatch layer to verify no information is lost in the wrapper.
#[test]
fn test_patches_step_dense_linear_relu_soundness() -> Result<()> {
    let weight = arr2(&[[1.0f32, -1.0], [-2.0, 1.0]]);
    let bias = arr1(&[0.5f32, -0.5]);
    let linear_layer = Layer::Linear(LinearLayer::new(weight.clone(), Some(bias.clone()))?);
    let relu_layer = Layer::ReLU(ReLULayer::new());
    let input = bounded_1d(&[-1.0, -1.0], &[1.0, 1.0]);

    // Forward IBP for pre-activation bounds.
    let post_linear = linear_layer.propagate_ibp(&input)?;

    // CROWN backward through patches dispatch (Dense mode).
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(post_linear.flatten().len()));
    let r1 = crown_backward_step_patches(
        &relu_layer,
        &mut bounds,
        &post_linear,
        None,
        1,
        "test",
        None,
    )?;
    assert!(matches!(r1, CrownStepResult::Continue));
    let r2 =
        crown_backward_step_patches(&linear_layer, &mut bounds, &input, None, 0, "test", None)?;
    assert!(matches!(r2, CrownStepResult::Continue));

    // Concretize and verify soundness against 25 sampled inputs.
    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense"),
    };
    let crown_flat = lb.concretize(&input).flatten();
    assert_linear_relu_soundness(&crown_flat, &weight, &bias);
    Ok(())
}

// ── #patches-zero-pad-identity ────────────────────────────────────────────
// A Pad whose every (before, after) is (0, 0) adds no elements, so its output
// tensor IS its input tensor. The patches relation must therefore pass through
// UNTOUCHED rather than being materialized dense.

/// A zero-pad `Pad` in Patches mode must stay in Patches and leave the relation
/// bit-identical.
///
/// Regression guard for the TinyYOLO / yolo_2023 defect: `Pad_10` and `Pad_17`
/// are both `pads=[0,0,0,0,0,0,0,0]` — pure no-ops — yet each fell through to
/// `generic_dense_dispatch`, demanding a 3_743_547_392-byte dense pair against a
/// 2 GiB budget. The guard refused, the CROWN backward for `Conv_12`/`Add_15`
/// returned the conservative relation, and both targets silently reverted to IBP
/// width while still being counted as CROWN successes.
#[test]
fn zero_pad_stays_in_patches_and_does_not_materialize_dense() -> Result<()> {
    use crate::layers::{PadLayer, PadMode};

    let layer = Layer::Pad(PadLayer::new(
        vec![(0, 0), (0, 0), (0, 0)],
        PadMode::Constant(0.0),
    ));
    let mut bounds = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (1, 2, 2),
        (1, 2, 2),
    )));
    let before = match &bounds {
        CrownBounds::Patches(pb) => (pb.row_count, pb.lower_b.clone(), pb.upper_b.clone()),
        CrownBounds::Dense(_) => panic!("fixture must start in Patches"),
    };
    let pre_act = bounded_3d((1, 2, 2), -1.0, 1.0);
    let engine = CountingGemmEngine::new();

    let result = crown_backward_step_patches(
        &layer,
        &mut bounds,
        &pre_act,
        Some(&engine),
        0,
        "test",
        None,
    )?;

    assert!(matches!(result, CrownStepResult::Continue));
    match &bounds {
        CrownBounds::Patches(pb) => {
            assert_eq!(pb.row_count, before.0, "zero pad must not change row_count");
            assert_eq!(pb.lower_b, before.1, "zero pad must not change lower_b");
            assert_eq!(pb.upper_b, before.2, "zero pad must not change upper_b");
        }
        CrownBounds::Dense(_) => {
            panic!("zero-pad Pad must NOT materialize dense — that is the defect")
        }
    }
    assert_eq!(
        engine.gemm_calls(),
        0,
        "an identity pass-through must not launch the caller engine"
    );
    Ok(())
}

#[test]
fn expired_soft_deadline_preempts_zero_pad_identity_before_publication() {
    use crate::layers::{PadLayer, PadMode};

    let layer = Layer::Pad(PadLayer::new(
        vec![(0, 0), (0, 0), (0, 0)],
        PadMode::Constant(0.0),
    ));
    let mut bounds = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (1, 2, 2),
        (1, 2, 2),
    )));
    let before = bounds.clone();
    let pre_act = bounded_3d((1, 2, 2), -1.0, 1.0);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("one millisecond fits before now");

    let error = match crown_backward_step_patches_with_deadline_authority(
        &layer,
        &mut bounds,
        &pre_act,
        None,
        0,
        "collector-expired-soft-budget",
        Some(expired),
        false,
    ) {
        Err(error) => error,
        Ok(_) => {
            panic!("expired soft scheduling authority must refuse before zero-Pad pass-through")
        }
    };

    assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
    assert_patches_bounds_bitwise_eq(&bounds, &before, "expired zero-Pad entry");
}

/// A Pad with a NON-zero pad must NOT take the identity path — it still needs
/// real work, so it falls through to the standard dense dispatch.
#[test]
fn nonzero_pad_does_not_take_the_identity_shortcut() -> Result<()> {
    use crate::layers::{PadLayer, PadMode};

    let layer = Layer::Pad(PadLayer::new(
        vec![(0, 0), (1, 0), (0, 0)],
        PadMode::Constant(0.0),
    ));
    let mut bounds = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (1, 2, 2),
        (1, 2, 2),
    )));
    let pre_act = bounded_3d((1, 2, 2), -1.0, 1.0);

    // Whatever this does (succeed via dense, or refuse), it must NOT silently
    // pass the relation through unchanged as if the pad were absent.
    let _ = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None);
    assert!(
        !matches!(&bounds, CrownBounds::Patches(pb) if pb.row_count == 4
            && pb.lower_a.identity),
        "a non-zero pad must not be treated as an identity pass-through"
    );
    Ok(())
}
