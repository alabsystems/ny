// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::common::BoundPropagation;
use crate::layers::{CeluLayer, EluLayer, HardSwishLayer, MishLayer, SeluLayer, SiLULayer};
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{
    hardswish_eval, mish_eval, sample_points, silu_eval, valid_interval, FP_TOLERANCE, SELU_ALPHA,
    SELU_LAMBDA,
};

// =============================================================================
// ADDITIONAL ACTIVATION FUNCTION SOUNDNESS TESTS (ELU, SELU, Mish, HardSwish)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// ELU IBP soundness: for any x in [l, u], ELU(x) is in computed bounds.
    /// ELU(x) = x if x >= 0, else alpha * (exp(x) - 1)
#[ntest::timeout(10000)]
    #[test]
    fn soundness_elu_ibp((l, u) in valid_interval(10.0), alpha in 0.5f32..2.0) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let elu = EluLayer::new(alpha);
        let output = elu.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let elu_x = if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) };
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= elu_x && elu_x <= output.upper()[[0]] + FP_TOLERANCE,
                "ELU soundness violation: ELU({}, alpha={})={} not in [{}, {}]",
                x, alpha, elu_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// SELU IBP soundness: for any x in [l, u], SELU(x) is in computed bounds.
    /// SELU(x) = lambda * x if x >= 0, else lambda * alpha * (exp(x) - 1)
#[ntest::timeout(10000)]
    #[test]
    fn soundness_selu_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let selu = SeluLayer::new();
        let output = selu.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let selu_x = if x >= 0.0 {
                SELU_LAMBDA * x
            } else {
                SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0)
            };
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= selu_x && selu_x <= output.upper()[[0]] + FP_TOLERANCE,
                "SELU soundness violation: SELU({})={} not in [{}, {}]",
                x, selu_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// CELU IBP soundness: for any x in [l, u], CELU(x) is in computed bounds.
    /// CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
#[ntest::timeout(10000)]
    #[test]
    fn soundness_celu_ibp((l, u) in valid_interval(10.0), alpha in 0.5f32..2.0) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let celu = CeluLayer::new(alpha);
        let output = celu.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let celu_x = x.max(0.0) + (alpha * ((x / alpha).exp() - 1.0)).min(0.0);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= celu_x && celu_x <= output.upper()[[0]] + FP_TOLERANCE,
                "CELU soundness violation: CELU({}, alpha={})={} not in [{}, {}]",
                x, alpha, celu_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Mish IBP soundness: for any x in [l, u], Mish(x) is in computed bounds.
    /// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
#[ntest::timeout(10000)]
    #[test]
    fn soundness_mish_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let mish = MishLayer::new();
        let output = mish.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let mish_x = mish_eval(x);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= mish_x && mish_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Mish soundness violation: Mish({})={} not in [{}, {}]",
                x, mish_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// HardSwish IBP soundness: for any x in [l, u], HardSwish(x) is in computed bounds.
    /// HardSwish(x) = x * max(0, min(1, (x + 3) / 6))
#[ntest::timeout(10000)]
    #[test]
    fn soundness_hardswish_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let hardswish = HardSwishLayer::new();
        let output = hardswish.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let hs_x = hardswish_eval(x);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= hs_x && hs_x <= output.upper()[[0]] + FP_TOLERANCE,
                "HardSwish soundness violation: HardSwish({})={} not in [{}, {}]",
                x, hs_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// SiLU IBP soundness: for any x in [l, u], SiLU(x) is in computed bounds.
    /// SiLU(x) = x * sigmoid(x)
#[ntest::timeout(10000)]
    #[test]
    fn soundness_silu_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let silu = SiLULayer::new();
        let output = silu.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let silu_x = silu_eval(x);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= silu_x && silu_x <= output.upper()[[0]] + FP_TOLERANCE,
                "SiLU soundness violation: SiLU({})={} not in [{}, {}]",
                x, silu_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }
}
