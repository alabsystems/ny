// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sliding-window causal softmax soundness regressions.

use crate::layers::common::BoundPropagation;
use crate::layers::softmax::CausalSoftmaxLayer;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::softmax;

/// Tolerance for sliding-window causal softmax IBP bounds.
const SLIDING_CAUSAL_IBP_TOLERANCE: f32 = 1e-4;

fn sliding_causal_softmax(x: &Array2<f32>, window_size: usize) -> Array2<f32> {
    let (seq_q, seq_k) = (x.nrows(), x.ncols());
    let mut out = Array2::<f32>::zeros((seq_q, seq_k));
    for i in 0..seq_q {
        let active_end = (i + 1).min(seq_k);
        let active_start = active_end.saturating_sub(window_size + 1);
        let active: Vec<f32> = (active_start..active_end).map(|j| x[[i, j]]).collect();
        if active.is_empty() {
            continue;
        }
        let sm = softmax(&Array1::from_vec(active));
        for (offset, j) in (active_start..active_end).enumerate() {
            out[[i, j]] = sm[offset];
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Sliding-window causal softmax IBP remains sound and masks positions
    /// outside the local causal window.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sliding_window_causal_softmax_ibp_3x3(
        (l00, u00) in super::valid_interval(3.0),
        (l01, u01) in super::valid_interval(3.0),
        (l02, u02) in super::valid_interval(3.0),
        (l10, u10) in super::valid_interval(3.0),
        (l11, u11) in super::valid_interval(3.0),
        (l12, u12) in super::valid_interval(3.0),
        (l20, u20) in super::valid_interval(3.0),
        (l21, u21) in super::valid_interval(3.0),
        (l22, u22) in super::valid_interval(3.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![l00, l01, l02, l10, l11, l12, l20, l21, l22],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![u00, u01, u02, u10, u11, u12, u20, u21, u22],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = CausalSoftmaxLayer::new(-1).with_window_size(1);
        let output = layer
            .propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("sliding causal propagate_ibp failed: {e}")
            ))?;

        let x = Array2::from_shape_vec(
            (3, 3),
            vec![
                f32::midpoint(l00, u00), f32::midpoint(l01, u01), f32::midpoint(l02, u02),
                f32::midpoint(l10, u10), f32::midpoint(l11, u11), f32::midpoint(l12, u12),
                f32::midpoint(l20, u20), f32::midpoint(l21, u21), f32::midpoint(l22, u22),
            ],
        ).unwrap();
        let expected = sliding_causal_softmax(&x, 1);

        for i in 0..3 {
            for j in 0..3 {
                prop_assert!(
                    output.lower()[[i, j]] <= expected[[i, j]] + SLIDING_CAUSAL_IBP_TOLERANCE,
                    "Sliding causal IBP lower violated: [{},{}]: lb={} > actual={}",
                    i, j, output.lower()[[i, j]], expected[[i, j]]
                );
                prop_assert!(
                    output.upper()[[i, j]] >= expected[[i, j]] - SLIDING_CAUSAL_IBP_TOLERANCE,
                    "Sliding causal IBP upper violated: [{},{}]: ub={} < actual={}",
                    i, j, output.upper()[[i, j]], expected[[i, j]]
                );
            }
        }

        prop_assert_eq!(output.lower()[[2, 0]], 0.0);
        prop_assert_eq!(output.upper()[[2, 0]], 0.0);
    }
}
