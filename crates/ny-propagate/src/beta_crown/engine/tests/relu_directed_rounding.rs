// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::prelude::*;
use ny_tensor::{next_down_f32, next_up_f32};
use proptest::prelude::*;

fn alpha_state_with_pairs(pairs: &[(usize, f32)]) -> DomainAlphaState {
    let neurons = pairs
        .iter()
        .map(|&(neuron_idx, alpha)| ((1, neuron_idx), AlphaNeuronState::new(alpha)))
        .collect::<HashMap<_, _>>();
    DomainAlphaState {
        neurons,
        slow_alphas: None,
    }
}

fn one_neuron_output_bounds(
    lower_a: [[f32; 1]; 2],
    upper_a: [[f32; 1]; 2],
) -> Result<LinearBounds, ny_core::NyError> {
    LinearBounds::new_or_conservative(
        arr2(&lower_a),
        arr1(&[0.0, 0.0]),
        arr2(&upper_a),
        arr1(&[0.0, 0.0]),
    )
}

fn unstable_interval_strategy() -> impl Strategy<Value = (f32, f32)> {
    (-5.0f32..-0.05, 0.05f32..5.0).prop_map(|(l, u)| (l, u))
}

fn nonzero_positive_coeff_strategy() -> impl Strategy<Value = f32> {
    0.25f32..3.0
}

fn nonzero_negative_coeff_strategy() -> impl Strategy<Value = f32> {
    -3.0f32..-0.25
}

#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_rounds_coefficients_4188() {
    let pre_bounds = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn()).unwrap();
    let output_bounds = one_neuron_output_bounds([[1.3], [-1.7]], [[1.1], [-1.2]]).unwrap();
    let alpha_state = alpha_state_with_pairs(&[(0, 0.3)]);

    let result = BetaCrownVerifier::new(BetaCrownConfig::default())
        .relu_backward_with_alpha_beta(
            &output_bounds,
            &pre_bounds,
            None,
            &BetaState::empty(),
            &alpha_state,
            1,
        )
        .unwrap();

    let lower_slope = 0.3f32;
    let upper_slope = 3.0f32 / 5.0f32;

    assert_eq!(
        result.lower_a()[[0, 0]],
        next_down_f32(1.3f32 * lower_slope)
    );
    assert_eq!(
        result.lower_a()[[1, 0]],
        next_down_f32(-1.7f32 * upper_slope)
    );
    assert_eq!(result.upper_a()[[0, 0]], next_up_f32(1.1f32 * upper_slope));
    assert_eq!(result.upper_a()[[1, 0]], next_up_f32(-1.2f32 * lower_slope));

    assert!((result.lower_a()[[0, 0]] as f64) <= 1.3f64 * lower_slope as f64);
    assert!((result.lower_a()[[1, 0]] as f64) <= -1.7f64 * upper_slope as f64);
    assert!((result.upper_a()[[0, 0]] as f64) >= 1.1f64 * upper_slope as f64);
    assert!((result.upper_a()[[1, 0]] as f64) >= -1.2f64 * lower_slope as f64);
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1024) })]

    #[test]
    fn proptest_relu_directed_rounding_brackets_reference_4188(
        (l, u) in unstable_interval_strategy(),
        alpha in 0.0f32..1.0,
        lower_coeff_pos in nonzero_positive_coeff_strategy(),
        lower_coeff_neg in nonzero_negative_coeff_strategy(),
        upper_coeff_pos in nonzero_positive_coeff_strategy(),
        upper_coeff_neg in nonzero_negative_coeff_strategy(),
    ) {
        let pre_bounds = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn())
            .map_err(|e| TestCaseError::fail(format!("{e}")))?;
        let alpha_state = alpha_state_with_pairs(&[(0, alpha)]);
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let upper_slope = u / (u - l).max(crate::layers::activations::RELU_RELAX_MIN_WIDTH);

        let alpha_beta_bounds = one_neuron_output_bounds(
            [[lower_coeff_pos], [lower_coeff_neg]],
            [[upper_coeff_pos], [upper_coeff_neg]],
        )
        .map_err(|e| TestCaseError::fail(format!("{e}")))?;
        let alpha_beta_result = verifier
            .relu_backward_with_alpha_beta(
                &alpha_beta_bounds,
                &pre_bounds,
                None,
                &BetaState::empty(),
                &alpha_state,
                1,
            )
            .map_err(|e| TestCaseError::fail(format!("{e}")))?;

        let lower_pos_ref = lower_coeff_pos as f64 * alpha as f64;
        let lower_neg_ref = lower_coeff_neg as f64 * upper_slope as f64;
        let upper_pos_ref = upper_coeff_pos as f64 * upper_slope as f64;
        let upper_neg_ref = upper_coeff_neg as f64 * alpha as f64;

        prop_assert!(
            (alpha_beta_result.lower_a()[[0, 0]] as f64) <= lower_pos_ref,
            "alpha-beta lower positive unsound: {} > {}",
            alpha_beta_result.lower_a()[[0, 0]],
            lower_pos_ref
        );
        prop_assert!(
            (alpha_beta_result.lower_a()[[1, 0]] as f64) <= lower_neg_ref,
            "alpha-beta lower negative unsound: {} > {}",
            alpha_beta_result.lower_a()[[1, 0]],
            lower_neg_ref
        );
        prop_assert!(
            (alpha_beta_result.upper_a()[[0, 0]] as f64) >= upper_pos_ref,
            "alpha-beta upper positive unsound: {} < {}",
            alpha_beta_result.upper_a()[[0, 0]],
            upper_pos_ref
        );
        prop_assert!(
            (alpha_beta_result.upper_a()[[1, 0]] as f64) >= upper_neg_ref,
            "alpha-beta upper negative unsound: {} < {}",
            alpha_beta_result.upper_a()[[1, 0]],
            upper_neg_ref
        );

        prop_assert_eq!(
            alpha_beta_result.lower_a()[[0, 0]],
            next_down_f32(lower_coeff_pos * alpha)
        );
        prop_assert_eq!(
            alpha_beta_result.lower_a()[[1, 0]],
            next_down_f32(lower_coeff_neg * upper_slope)
        );
        prop_assert_eq!(
            alpha_beta_result.upper_a()[[0, 0]],
            next_up_f32(upper_coeff_pos * upper_slope)
        );
        prop_assert_eq!(
            alpha_beta_result.upper_a()[[1, 0]],
            next_up_f32(upper_coeff_neg * alpha)
        );
    }
}
