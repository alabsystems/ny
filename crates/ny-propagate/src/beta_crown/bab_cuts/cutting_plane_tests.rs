// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`CuttingPlane`] construction and constraint strengthening.

use ny_core::NyError;

use super::cutting_plane::CuttingPlane;
use super::types::{CutKind, CutMetadata, CutTerm};
use crate::beta_crown::branching::{NeuronConstraint, SplitHistory};
use crate::beta_crown::state::BetaState;

#[ntest::timeout(5000)]
#[test]
fn test_strengthened_cut_drops_low_influence_constraints() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.1,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.9,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 2,
        neuron_idx: 0,
        is_active: true,
        score: 0.2,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(2, 0, 0.5); // Ensure at least one beta-positive constraint

    let strengthened = CuttingPlane::from_verified_domain_strengthened(&history, &beta_state, 0.5)
        .unwrap()
        .expect("strengthened cut should be Some for valid history");

    assert!(strengthened.dropped_constraints > 0);
    assert_eq!(strengthened.history.constraints.len(), 2);
    assert_eq!(strengthened.cut.terms.len(), 2);
    assert_eq!(strengthened.cut.bias, 0.0); // 1 active - 1
}

#[ntest::timeout(5000)]
#[test]
fn test_strengthened_cut_keeps_unranked_constraints() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: f32::NAN,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: f32::NAN,
    });

    let beta_state = BetaState::from_history(&history).unwrap();
    let strengthened = CuttingPlane::from_verified_domain_strengthened(&history, &beta_state, 0.5)
        .unwrap()
        .expect("strengthened cut should be Some for valid history");

    assert_eq!(strengthened.history.constraints.len(), 2);
    assert_eq!(strengthened.cut.terms.len(), 2);
}

#[ntest::timeout(5000)]
#[test]
fn test_strengthened_cut_handles_non_finite_drop_ratio() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.1,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.9,
    });

    let beta_state = BetaState::from_history(&history).unwrap();
    let strengthened =
        CuttingPlane::from_verified_domain_strengthened(&history, &beta_state, f32::NAN)
            .unwrap()
            .expect("strengthened cut should be Some for valid history");

    assert_eq!(strengthened.history.constraints.len(), 2);
    assert_eq!(strengthened.cut.terms.len(), 2);
}

/// Regression test for #2998 Slice B: from_verified_domain returns Result<Option<_>>
/// and propagates Ok(Some(...)) for valid histories.
#[ntest::timeout(5000)]
#[test]
fn test_from_verified_domain_returns_result_ok_some_2998() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    let result = CuttingPlane::from_verified_domain(&history);
    assert!(result.is_ok(), "valid history should not produce an error");
    let opt = result.unwrap();
    assert!(opt.is_some(), "non-empty history should produce a cut");
    let cut = opt.unwrap();
    assert_eq!(cut.terms.len(), 2);
    assert_eq!(cut.bias, 0.0); // 1 active - 1
}

/// Regression test for #2998 Slice B: from_verified_domain returns Ok(None)
/// for empty histories.
#[ntest::timeout(5000)]
#[test]
fn test_from_verified_domain_returns_result_ok_none_for_empty_2998() {
    let history = SplitHistory::new();
    let result = CuttingPlane::from_verified_domain(&history);
    assert!(result.is_ok(), "empty history should not produce an error");
    assert!(
        result.unwrap().is_none(),
        "empty history should produce None"
    );
}

/// Regression test for #2998 Slice B: from_verified_domain_strengthened returns
/// Result<Option<_>> and propagates Ok(Some(...)) for valid histories.
#[ntest::timeout(5000)]
#[test]
fn test_from_verified_domain_strengthened_returns_result_2998() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.5,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.9,
    });

    let beta_state = BetaState::from_history(&history).unwrap();
    let result = CuttingPlane::from_verified_domain_strengthened(&history, &beta_state, 0.0);
    assert!(result.is_ok(), "valid history should not produce an error");
    assert!(
        result.unwrap().is_some(),
        "valid history with drop_ratio=0 should produce a cut"
    );
}

/// Regression test for #2998 Slice B: non-finite bias must surface as
/// `NyError::NumericalInstability` instead of a panic.
#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_new_rejects_non_finite_bias_2998() {
    let terms = vec![CutTerm {
        layer_idx: 0,
        neuron_idx: 0,
        coefficient: 1.0,
    }];

    for bias in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = CuttingPlane::new(
            terms.clone(),
            bias,
            0.0,
            1,
            CutMetadata::new(0, CutKind::Verified),
        )
        .expect_err("non-finite bias should return NyError");
        match err {
            NyError::NumericalInstability(msg) => {
                assert!(
                    msg.contains("bias must be finite"),
                    "expected bias validation context, got: {msg}"
                );
            }
            other => panic!("expected NumericalInstability, got: {other:?}"),
        }
    }
}

/// Regression test for #2998 Slice B: non-finite lambda must surface as
/// `NyError::NumericalInstability` instead of a panic.
#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_new_rejects_non_finite_lambda_2998() {
    let terms = vec![CutTerm {
        layer_idx: 0,
        neuron_idx: 0,
        coefficient: 1.0,
    }];

    for lambda in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = CuttingPlane::new(
            terms.clone(),
            0.0,
            lambda,
            1,
            CutMetadata::new(0, CutKind::Verified),
        )
        .expect_err("non-finite lambda should return NyError");
        match err {
            NyError::NumericalInstability(msg) => {
                assert!(
                    msg.contains("lambda must be finite"),
                    "expected lambda validation context, got: {msg}"
                );
            }
            other => panic!("expected NumericalInstability, got: {other:?}"),
        }
    }
}

/// Regression test for #3076: CuttingPlane::new must reject NaN term coefficients.
#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_new_rejects_nan_coefficient_3076() {
    let terms = vec![
        CutTerm {
            layer_idx: 0,
            neuron_idx: 0,
            coefficient: 1.0,
        },
        CutTerm {
            layer_idx: 0,
            neuron_idx: 1,
            coefficient: f32::NAN, // NaN coefficient
        },
    ];
    let result = CuttingPlane::new(terms, 0.0, 0.0, 2, CutMetadata::new(0, CutKind::Verified));
    assert!(result.is_err(), "NaN coefficient should be rejected");
}

/// Regression test for #3076: CuttingPlane::new must reject Inf term coefficients.
#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_new_rejects_inf_coefficient_3076() {
    let terms = vec![CutTerm {
        layer_idx: 0,
        neuron_idx: 0,
        coefficient: f32::INFINITY,
    }];
    let result = CuttingPlane::new(terms, 0.0, 0.0, 1, CutMetadata::new(0, CutKind::Verified));
    assert!(result.is_err(), "Inf coefficient should be rejected");
}

/// Regression test for #3076: set_lambda_grad sanitizes NaN to 0.0.
#[ntest::timeout(5000)]
#[test]
fn test_set_lambda_grad_sanitizes_nan_3076() {
    let mut cut = CuttingPlane::new(
        vec![CutTerm {
            layer_idx: 0,
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        0.0,
        0.0,
        1,
        CutMetadata::new(0, CutKind::Verified),
    )
    .expect("test cut: finite coefficient=1.0, bias=0.0, lambda=0.0");

    cut.set_lambda_grad(f32::NAN);
    assert_eq!(
        cut.lambda_grad(),
        0.0,
        "NaN grad should be sanitized to 0.0"
    );

    cut.set_lambda_grad(f32::INFINITY);
    assert_eq!(
        cut.lambda_grad(),
        0.0,
        "Inf grad should be sanitized to 0.0"
    );

    cut.set_lambda_grad(1.5);
    assert_eq!(cut.lambda_grad(), 1.5, "Finite grad should be preserved");
}

/// Regression test for #3076: Adam NaN guard catches lambda NaN even when m/v are finite.
#[ntest::timeout(5000)]
#[test]
fn test_adam_catches_lambda_nan_3076() {
    use crate::beta_crown::config::AdaptiveOptConfig;

    let mut cut = CuttingPlane::new(
        vec![CutTerm {
            layer_idx: 0,
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        0.0,
        1.0, // Start with valid lambda
        1,
        CutMetadata::new(0, CutKind::Verified),
    )
    .expect("test cut: finite coefficient=1.0, bias=0.0, lambda=1.0");

    // Manually corrupt lambda to NaN (simulates lr=NaN, grad=0 edge case)
    cut.lambda = f32::NAN;

    let config = AdaptiveOptConfig::default();
    cut.gradient_step_adam(&config, 1);

    assert_eq!(cut.lambda(), 0.0, "NaN lambda should be reset to 0.0");
    assert_eq!(cut.lambda_grad(), 0.0, "Grad should be reset");
}

/// Regression test for #2860: evaluate must not panic when neuron_idx exceeds
/// the pre_activations slice length.
#[test]
fn test_evaluate_oob_neuron_idx_no_panic_2860() {
    let cut = CuttingPlane::new(
        vec![
            CutTerm {
                layer_idx: 0,
                neuron_idx: 0,
                coefficient: 1.0,
            },
            CutTerm {
                layer_idx: 0,
                neuron_idx: 999, // out of bounds
                coefficient: -0.5,
            },
        ],
        0.0,
        1.0,
        2,
        CutMetadata::new(0, CutKind::Verified),
    )
    .expect("test cut: finite bias=0.0, lambda=1.0");

    // Only 2 elements — neuron_idx=999 is out of bounds
    let pre_activations = vec![(-1.0_f32, 2.0_f32), (-0.5, 1.5)];
    // Should not panic; OOB lookup returns (NEG_INFINITY, INFINITY),
    // producing non-finite intermediate → NaN guard returns 0.0 (#2598).
    let result = cut.evaluate(&pre_activations);
    assert_eq!(
        result, 0.0,
        "OOB index should return 0.0 via NaN guard; got {result}"
    );
}
