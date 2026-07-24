// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NaN-freedom property tests for IBP bound propagation.
//!
//! Validates that finite-width intervals produce non-NaN, non-inverted
//! output bounds. These tests verify the NaN guards and `nan_propagating_max`
//! fixes from #3316.
//!
//! Pattern from softsign/tests.rs `proptest_softsign_ibp_no_nan_for_finite_intervals`.
//! See also: mish, hardswish, gelu which have this test inline.

use crate::layers::common::BoundPropagation;
use crate::layers::{AbsLayer, CeluLayer, EluLayer, ExpLayer, SeluLayer};
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(2000) })]

    // #2435: finite-width IBP intervals must not produce NaN bounds.
    // Validates W3's NaN-propagating clamp fix (747d1c9) for CELU.
    #[test]
    fn proptest_celu_ibp_no_nan_for_finite_intervals(
        a in -1.0e6f32..1.0e6,
        b in -1.0e6f32..1.0e6,
    ) {
        let (l, u) = (a.min(b), a.max(b));
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();
        let layer = CeluLayer::default_alpha();
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        prop_assert!(!lower.is_nan(), "CELU IBP lower is NaN for [{l}, {u}]");
        prop_assert!(!upper.is_nan(), "CELU IBP upper is NaN for [{l}, {u}]");
        prop_assert!(lower <= upper, "CELU IBP bounds inverted for [{l}, {u}]: {lower} > {upper}");
    }

    // #2435: finite-width IBP intervals must not produce NaN bounds.
    // Validates W3's NaN-propagating clamp fix (747d1c9) for ELU.
    #[test]
    fn proptest_elu_ibp_no_nan_for_finite_intervals(
        a in -1.0e6f32..1.0e6,
        b in -1.0e6f32..1.0e6,
    ) {
        let (l, u) = (a.min(b), a.max(b));
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();
        let layer = EluLayer::default_alpha();
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        prop_assert!(!lower.is_nan(), "ELU IBP lower is NaN for [{l}, {u}]");
        prop_assert!(!upper.is_nan(), "ELU IBP upper is NaN for [{l}, {u}]");
        prop_assert!(lower <= upper, "ELU IBP bounds inverted for [{l}, {u}]: {lower} > {upper}");
    }

    // #2435: finite-width IBP intervals must not produce NaN bounds.
    // Validates W3's NaN-propagating clamp fix (747d1c9) for SELU.
    #[test]
    fn proptest_selu_ibp_no_nan_for_finite_intervals(
        a in -1.0e6f32..1.0e6,
        b in -1.0e6f32..1.0e6,
    ) {
        let (l, u) = (a.min(b), a.max(b));
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();
        let layer = SeluLayer;
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        prop_assert!(!lower.is_nan(), "SELU IBP lower is NaN for [{l}, {u}]");
        prop_assert!(!upper.is_nan(), "SELU IBP upper is NaN for [{l}, {u}]");
        prop_assert!(lower <= upper, "SELU IBP bounds inverted for [{l}, {u}]: {lower} > {upper}");
    }

    // #2435: finite-width IBP intervals must not produce NaN bounds.
    // Exp has an overflow threshold at 88.0 (exp(88) ≈ f32::MAX).
    // Range restricted to [-100, 88] to stay within valid Exp domain.
    #[test]
    fn proptest_exp_ibp_no_nan_for_finite_intervals(
        a in -100.0f32..88.0,
        b in -100.0f32..88.0,
    ) {
        let (l, u) = (a.min(b), a.max(b));
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();
        let layer = ExpLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        prop_assert!(!lower.is_nan(), "Exp IBP lower is NaN for [{l}, {u}]");
        prop_assert!(!upper.is_nan(), "Exp IBP upper is NaN for [{l}, {u}]");
        prop_assert!(lower <= upper, "Exp IBP bounds inverted for [{l}, {u}]: {lower} > {upper}");
        // Exp output is always >= 0
        prop_assert!(lower >= 0.0, "Exp IBP lower {lower} < 0 for [{l}, {u}]");
    }

    // #2435, #3316: finite-width IBP intervals must not produce NaN bounds for Abs.
    // Validates the NaN guard and nan_propagating_max fix in abs.rs.
    #[test]
    fn proptest_abs_ibp_no_nan_for_finite_intervals(
        a in -1.0e6f32..1.0e6,
        b in -1.0e6f32..1.0e6,
    ) {
        let (l, u) = (a.min(b), a.max(b));
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();
        let layer = AbsLayer;
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        prop_assert!(!lower.is_nan(), "Abs IBP lower is NaN for [{l}, {u}]");
        prop_assert!(!upper.is_nan(), "Abs IBP upper is NaN for [{l}, {u}]");
        prop_assert!(lower <= upper, "Abs IBP bounds inverted for [{l}, {u}]: {lower} > {upper}");
        // |x| >= 0 for all x
        prop_assert!(lower >= 0.0, "Abs IBP lower {lower} < 0 for [{l}, {u}]");
    }
}
