// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-based verification of batched CROWN memory budget guards (#3550).
//!
//! Verifies:
//! 1. Budget estimation arithmetic is correct (no overflow wrap-around)
//! 2. Overflow-inducing dimensions saturate to `usize::MAX` (never wrap to small)
//! 3. With zero budget, batched CROWN fallback bounds are sound (contain all outputs)
//! 4. With zero budget, batched CROWN bounds are at least as wide as IBP
//!
//! WALL-CLOCK POLICY FOR THIS FILE: the `#[ntest::timeout(..)]` guards below are
//! HANG SENTINELS, not performance assertions. These two properties cost 0.09s
//! and 0.06s ISOLATED -- they are not slow, they WAIT.
//!
//! They are the WRITER side of the `ny-test-utils` env lock: they set
//! `NY_DENSE_BUDGET_MB` to 0 process-wide, so they take the exclusive half and
//! must first drain every reader holding the shared half. That is the whole
//! point -- this file's zero-budget mutation is exactly the leak that was
//! surfacing elsewhere in the suite as `crown=-inf` and `budget_bytes: 0`, in
//! tests that had nothing to do with budgets. Excluding readers is correct; the
//! wait it creates is correct; a 10s wall turned that correct wait into the last
//! 2 failures of a 10,489-test run at --test-threads=8.
//!
//! 300s is ~3000x the isolated cost, and still catches an infinite loop.
//! MEASURE BEFORE LOWERING THEM.

use crate::network::crown_memory::{
    batched_dense_pair_bytes, batched_identity_pair_bytes, check_batched_identity_budget,
};
use crate::{Layer, LinearLayer, Network, ReLULayer};
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

// =============================================================================
// BUDGET ESTIMATION ARITHMETIC
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Budget estimation matches the mathematical formula for non-overflowing inputs.
    ///
    /// For small dimensions: `batched_dense_pair_bytes(b, r, c) == 2 * b * r * c * 4`
    #[test]
    fn budget_estimation_matches_formula(
        batch in 1usize..100,
        rows in 1usize..100,
        cols in 1usize..100,
    ) {
        let result = batched_dense_pair_bytes(batch, rows, cols);
        let expected = 2 * batch * rows * cols * size_of::<f32>();
        prop_assert_eq!(result, Some(expected));
    }

    /// Identity estimation extracts dim correctly from shape.
    ///
    /// For shape `[b1, b2, dim]`: bytes = 2 * b1 * b2 * dim * dim * sizeof(f32)
    #[test]
    fn identity_estimation_extracts_shape(
        b1 in 1usize..20,
        b2 in 1usize..20,
        dim in 1usize..20,
    ) {
        let shape = vec![b1, b2, dim];
        let result = batched_identity_pair_bytes(&shape);
        let batch_positions = b1 * b2;
        let expected = 2 * batch_positions * dim * dim * size_of::<f32>();
        prop_assert_eq!(result, Some(expected));
    }

    /// Overflow-inducing dimensions must return None (never wrap to a small value).
    ///
    /// If `batched_dense_pair_bytes` returned `Some(small)` for huge inputs, the
    /// budget guard would incorrectly pass, allowing an allocation that exceeds
    /// physical memory.
    #[test]
    fn overflow_returns_none(
        batch in (1usize << 30)..(1usize << 40),
        rows in (1usize << 20)..(1usize << 30),
        cols in (1usize << 20)..(1usize << 30),
    ) {
        let result = batched_dense_pair_bytes(batch, rows, cols);
        // With batch >= 2^30, rows >= 2^20, cols >= 2^20:
        // product >= 2^70, which overflows usize on 64-bit.
        // Must be None (overflow detected) — never Some(wrapped_small_value).
        prop_assert!(
            result.is_none(),
            "Expected None for overflowing dimensions ({batch} x {rows} x {cols}), got {:?}",
            result
        );
    }

    /// Zero budget rejects every positive-dimensional identity shape.
    ///
    /// With budget=0 bytes, any non-trivial identity allocation must be rejected.
    #[test]
    fn zero_budget_rejects_all_nontrivial(dim in 1usize..1000) {
        let shape = vec![dim];
        crate::tests::with_crown_dense_budget_mb("0", || {
            let result = check_batched_identity_budget("proptest:zero_budget", &shape);
            prop_assert!(
                result.is_err(),
                "dim={dim}: zero budget should reject identity allocation"
            );
            Ok(())
        })?;
    }
}

// =============================================================================
// FALLBACK SOUNDNESS: batched CROWN with zero budget
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// With zero budget, `propagate_crown_batched` falls back to IBP and bounds
    /// contain all true network outputs (soundness).
    ///
    /// Network: Linear(2x2) -> ReLU -> Linear(1x2).
    /// This is the critical invariant: budget exhaustion must never produce
    /// unsound bounds.
    #[ntest::timeout(300000)]
    #[test]
    fn batched_crown_zero_budget_fallback_is_sound(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(w2.clone(), Some(b2_arr.clone())).unwrap(),
        ));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        )
        .unwrap();

        let batched_output = crate::tests::with_crown_dense_budget_mb("0", || {
            network.propagate_crown_batched(&input)
        })
        .unwrap();

        // Verify all concrete outputs are within the bounds.
        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let y1 = w1.dot(&x) + &b1;
                let relu_out = y1.mapv(|v| v.max(0.0));
                let final_out = w2.dot(&relu_out) + &b2_arr;

                prop_assert!(
                    batched_output.lower()[[0]] - FP_TOLERANCE <= final_out[0]
                        && final_out[0] <= batched_output.upper()[[0]] + FP_TOLERANCE,
                    "Zero-budget batched CROWN soundness violation: output={} not in [{}, {}]",
                    final_out[0],
                    batched_output.lower()[[0]],
                    batched_output.upper()[[0]]
                );
            }
        }
    }

    /// With zero budget, batched CROWN bounds are at least as wide as IBP.
    ///
    /// Since the budget guard forces fallback to sequential CROWN (which itself
    /// falls back to IBP under zero budget), the final bounds must be no tighter
    /// than IBP. This tests the fallback chain doesn't silently narrow bounds.
    #[ntest::timeout(300000)]
    #[test]
    fn batched_crown_zero_budget_at_least_as_wide_as_ibp(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(w1, Some(b1)).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(w2, Some(b2_arr)).unwrap(),
        ));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        )
        .unwrap();

        let ibp_output = network.propagate_ibp(&input).unwrap();

        let batched_output = crate::tests::with_crown_dense_budget_mb("0", || {
            network.propagate_crown_batched(&input)
        })
        .unwrap();

        // Fallback bounds must be at least as wide as IBP (lower <= ibp_lower, upper >= ibp_upper).
        prop_assert!(
            batched_output.lower()[[0]] <= ibp_output.lower()[[0]] + FP_TOLERANCE,
            "Zero-budget batched lower ({}) is tighter than IBP lower ({}) — unsound narrowing",
            batched_output.lower()[[0]],
            ibp_output.lower()[[0]]
        );
        prop_assert!(
            batched_output.upper()[[0]] >= ibp_output.upper()[[0]] - FP_TOLERANCE,
            "Zero-budget batched upper ({}) is tighter than IBP upper ({}) — unsound narrowing",
            batched_output.upper()[[0]],
            ibp_output.upper()[[0]]
        );
    }
}
