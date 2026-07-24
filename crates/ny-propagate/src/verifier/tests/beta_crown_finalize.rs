// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for beta-CROWN result finalization: counterexample validation
//! and status downgrade logic.

use super::super::*;
use crate::beta_crown::{BabVerificationStatus, BetaCrownResult};
use crate::layers::{Layer, LinearLayer};
use crate::{PropagationConfig, PropagationMethod, Verifier};
use ndarray::{arr1, arr2};
use ny_core::UnknownReason;
use std::time::Duration;

fn identity_single_output_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("1x1 identity linear"),
    ));
    network
}

fn beta_crown_verifier() -> Verifier {
    Verifier::new(PropagationConfig {
        method: PropagationMethod::BetaCrown,
        ..Default::default()
    })
}

fn single_input_bounds() -> ny_tensor::BoundedTensor {
    ny_tensor::BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid input bounds")
}

fn lower_bound_spec(lower: f32) -> ny_core::VerificationSpec {
    ny_core::VerificationSpec::from_parts(
        vec![ny_core::Bound::new(-1.0, 1.0)],
        vec![ny_core::Bound::new_allow_infinite(lower, f32::INFINITY)],
        Some(5_000),
        None,
    )
    .expect("valid verification spec")
}

#[ntest::timeout(10000)]
#[test]
fn beta_crown_invalid_counterexample_is_downgraded_to_unknown() {
    let verifier = beta_crown_verifier();
    let network = identity_single_output_network();
    let input = single_input_bounds();
    let spec = lower_bound_spec(0.0);

    let result = BetaCrownResult {
        result: BabVerificationStatus::Violated {
            counterexample: vec![0.5],
            output: vec![-999.0],
        },
        domains_explored: 1,
        time_elapsed: Duration::from_millis(1),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };

    let finalized = verifier
        .finalize_beta_crown_result(&network, &input, &spec, result, 0.0)
        .expect("finalization should succeed");

    match finalized {
        ny_core::VerificationResult::Unknown { reason, bounds, .. } => {
            assert_eq!(reason, UnknownReason::PotentialViolation);
            assert_eq!(bounds.len(), 1);
            assert!(
                bounds[0].lower().is_infinite() && bounds[0].lower().is_sign_negative(),
                "invalid witness must not leak reported concrete lower bound into Unknown: {:?}",
                bounds
            );
            assert_eq!(bounds[0].upper(), 0.0);
        }
        other => panic!("expected Unknown after invalid witness rejection, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn beta_crown_valid_counterexample_uses_revalidated_output() {
    let verifier = beta_crown_verifier();
    let network = identity_single_output_network();
    let input = single_input_bounds();
    let spec = lower_bound_spec(0.0);

    let result = BetaCrownResult {
        result: BabVerificationStatus::Violated {
            counterexample: vec![-0.25],
            output: vec![123.0],
        },
        domains_explored: 1,
        time_elapsed: Duration::from_millis(1),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };

    let finalized = verifier
        .finalize_beta_crown_result(&network, &input, &spec, result, 0.0)
        .expect("finalization should succeed");

    match finalized {
        ny_core::VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => {
            assert_eq!(counterexample, vec![-0.25]);
            assert_eq!(output.len(), 1);
            assert!(
                (output[0] + 0.25).abs() <= 1e-5,
                "expected revalidated output near -0.25, got {:?}",
                output
            );
        }
        other => panic!("expected Violated with revalidated output, got {other:?}"),
    }
}
