// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward equivalence proptests.
//!
//! For each layer: create identical incoming bounds as `LinearBounds` (scalar)
//! and `BatchedLinearBounds` (batched), run both CROWN backward paths, and
//! assert element-by-element equivalence within FP tolerance.
//!
//! Pattern from `batchnorm_batched_matches_scalar` (commit a073f40).
//!
//! Part of #3247.

use crate::layers::arithmetic::{
    AddConstantLayer, DivConstantLayer, MulConstantLayer, SubConstantLayer,
};
use crate::layers::common::BoundPropagation;
use crate::layers::convolution::conv1d::Conv1dLayer;
use crate::layers::convolution::conv2d::Conv2dLayer;
use crate::layers::trigonometric::SigmoidLayer;
use crate::layers::{Layer, LinearLayer, ReshapeLayer, TransposeLayer};
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// Separate tolerances for coefficient matrices vs bias vectors.
///
/// Most batched/scalar pairs reuse the exact same math and should match bit-for-bit.
/// `AddConstant`/`SubConstant` bias paths are the exception: scalar uses `dot_bias_f64`
/// while batched does `sum_axis` then one multiply, so f64 evaluation order can differ
/// by about one f32 ULP after directed rounding. (#3255)
#[derive(Clone, Copy)]
struct EquivTolerances {
    coefficient: f32,
    bias: f32,
}

const EXACT_EQUIV_TOLERANCES: EquivTolerances = EquivTolerances {
    coefficient: 0.0,
    bias: 0.0,
};
const BIAS_F64_EVAL_ORDER_TOLERANCES: EquivTolerances = EquivTolerances {
    coefficient: 0.0,
    bias: 1e-6,
};

fn assert_iter_equiv(
    actual: impl Iterator<Item = f32>,
    expected: impl Iterator<Item = f32>,
    tolerance: f32,
    layer_name: &str,
    field_name: &str,
) -> Result<(), TestCaseError> {
    for (idx, (a, e)) in actual.zip(expected).enumerate() {
        prop_assert!(
            (a - e).abs() <= tolerance,
            "{layer_name} {field_name} mismatch at {idx}: batched={a}, scalar={e}, tolerance={tolerance}"
        );
    }
    Ok(())
}

/// Assert element-by-element equivalence of all four bound fields.
///
/// Verifies element counts match before comparing values — zip() alone
/// would silently truncate to the shorter iterator and hide shape bugs.
fn assert_batched_equiv(
    actual: &BatchedLinearBounds,
    expected: &LinearBounds,
    layer_name: &str,
    tolerances: EquivTolerances,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(
        actual.lower_a.len(),
        expected.lower_a.len(),
        "{} lower_a element count mismatch",
        layer_name,
    );
    prop_assert_eq!(
        actual.upper_a.len(),
        expected.upper_a.len(),
        "{} upper_a element count mismatch",
        layer_name,
    );
    prop_assert_eq!(
        actual.lower_b.len(),
        expected.lower_b.len(),
        "{} lower_b element count mismatch",
        layer_name,
    );
    prop_assert_eq!(
        actual.upper_b.len(),
        expected.upper_b.len(),
        "{} upper_b element count mismatch",
        layer_name,
    );

    assert_iter_equiv(
        actual.lower_a.iter().copied(),
        expected.lower_a.iter().copied(),
        tolerances.coefficient,
        layer_name,
        "lower_a",
    )?;
    assert_iter_equiv(
        actual.upper_a.iter().copied(),
        expected.upper_a.iter().copied(),
        tolerances.coefficient,
        layer_name,
        "upper_a",
    )?;
    assert_iter_equiv(
        actual.lower_b.iter().copied(),
        expected.lower_b.iter().copied(),
        tolerances.bias,
        layer_name,
        "lower_b",
    )?;
    assert_iter_equiv(
        actual.upper_b.iter().copied(),
        expected.upper_b.iter().copied(),
        tolerances.bias,
        layer_name,
        "upper_b",
    )?;
    Ok(())
}

/// Build scalar and batched bounds from raw coefficient/bias vectors.
fn build_bounds_pair(
    num_out: usize,
    num_in: usize,
    lower_a_vals: Vec<f32>,
    lower_b_vals: Vec<f32>,
    upper_a_vals: Vec<f32>,
    upper_b_vals: Vec<f32>,
) -> (LinearBounds, BatchedLinearBounds) {
    let scalar = LinearBounds::new(
        Array2::from_shape_vec((num_out, num_in), lower_a_vals).unwrap(),
        Array1::from_vec(lower_b_vals),
        Array2::from_shape_vec((num_out, num_in), upper_a_vals).unwrap(),
        Array1::from_vec(upper_b_vals),
    )
    .unwrap();

    let batched = BatchedLinearBounds::from_parts_unchecked(
        scalar.lower_a.clone().into_dyn(),
        scalar.lower_b.clone().into_dyn(),
        scalar.upper_a.clone().into_dyn(),
        scalar.upper_b.clone().into_dyn(),
        vec![num_in],
        vec![num_out],
    );

    (scalar, batched)
}

// =============================================================================
// MULCONSTANT BATCHED CROWN EQUIV
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// MulConstant batched CROWN matches scalar CROWN.
    ///
    /// y = x * c. CROWN backward scales coefficient columns by c.
    /// Both paths should produce identical results modulo FP ordering.
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn mulconstant_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        c in -5.0f32..5.0,
    ) {
        prop_assume!(c.abs() > 0.001);

        let num_in = 3;
        let num_out = 2;

        let layer = MulConstantLayer::scalar(c);

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, num_in,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "MulConstant", EXACT_EQUIV_TOLERANCES)?;
    }
}

// =============================================================================
// DIVCONSTANT BATCHED CROWN EQUIV
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// DivConstant batched CROWN matches scalar CROWN.
    ///
    /// y = x / c. DivConstant delegates to MulConstant(1/c) internally.
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn divconstant_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        c in -5.0f32..5.0,
    ) {
        prop_assume!(c.abs() > 0.1);

        let num_in = 3;
        let num_out = 2;

        let layer = DivConstantLayer::scalar(c);

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, num_in,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "DivConstant", EXACT_EQUIV_TOLERANCES)?;
    }
}

// =============================================================================
// ADDCONSTANT BATCHED CROWN EQUIV
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// AddConstant batched CROWN matches scalar CROWN.
    ///
    /// y = x + c. CROWN backward: A unchanged, bias += A @ c.
    /// The batched path uses f64 accumulation + directed rounding for the
    /// row-sum (A @ c) computation. Scalar path should match.
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn addconstant_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        c in -5.0f32..5.0,
    ) {
        let num_in = 3;
        let num_out = 2;

        let layer = AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), c));

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, num_in,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(
            &actual,
            &expected,
            "AddConstant",
            BIAS_F64_EVAL_ORDER_TOLERANCES,
        )?;
    }
}

// =============================================================================
// SUBCONSTANT BATCHED CROWN EQUIV
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// SubConstant (y = x - c) batched CROWN matches scalar CROWN.
    ///
    /// CROWN backward: A unchanged, bias -= A @ c.
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn subconstant_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        c in -5.0f32..5.0,
    ) {
        let num_in = 3;
        let num_out = 2;

        let layer = SubConstantLayer::scalar(c);

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, num_in,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(
            &actual,
            &expected,
            "SubConstant",
            BIAS_F64_EVAL_ORDER_TOLERANCES,
        )?;
    }

    /// SubConstant reverse (y = c - x) batched CROWN matches scalar CROWN.
    ///
    /// Reverse negates coefficients and shifts bias. Tests sign-flip path.
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn subconstant_reverse_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        c in -5.0f32..5.0,
    ) {
        let num_in = 3;
        let num_out = 2;

        let layer = SubConstantLayer::new_reverse(ArrayD::from_elem(IxDyn(&[]), c));

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, num_in,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(
            &actual,
            &expected,
            "SubConstant_reverse",
            BIAS_F64_EVAL_ORDER_TOLERANCES,
        )?;
    }
}

// =============================================================================
// LINEAR BATCHED CROWN EQUIV
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Linear layer batched CROWN matches scalar CROWN.
    ///
    /// For W=[4x3], bias=[4], input dim=3, output dim=4.
    /// Incoming bounds: [2x4] (2 output rows, 4 intermediate dims).
    /// After backward through Linear: new A is [2x3], new b is [2].
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn linear_batched_matches_scalar(
        weights in prop::collection::vec(-3.0f32..3.0, 12),   // 4*3 = 12
        biases in prop::collection::vec(-3.0f32..3.0, 4),
        lower_a_vals in prop::collection::vec(-3.0f32..3.0, 8),  // 2*4 = 8
        upper_a_vals in prop::collection::vec(-3.0f32..3.0, 8),
        lower_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
        upper_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
    ) {
        let in_features = 3;
        let out_features = 4;
        let num_out = 2;

        let weight = Array2::from_shape_vec((out_features, in_features), weights).unwrap();
        let bias = Array1::from_vec(biases);
        let layer = LinearLayer::new(weight, Some(bias))
            .map_err(|e| TestCaseError::fail(format!("LinearLayer::new failed: {e}")))?;

        // Incoming bounds operate on the Linear layer's OUTPUT (out_features=4)
        let scalar_bounds = LinearBounds::new(
            Array2::from_shape_vec((num_out, out_features), lower_a_vals).unwrap(),
            Array1::from_vec(lower_b_vals),
            Array2::from_shape_vec((num_out, out_features), upper_a_vals).unwrap(),
            Array1::from_vec(upper_b_vals),
        ).unwrap();

        let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
            scalar_bounds.lower_a.clone().into_dyn(),
            scalar_bounds.lower_b.clone().into_dyn(),
            scalar_bounds.upper_a.clone().into_dyn(),
            scalar_bounds.upper_b.clone().into_dyn(),
            vec![out_features],
            vec![num_out],
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Linear", EXACT_EQUIV_TOLERANCES)?;
    }

    /// Linear layer no-bias batched CROWN matches scalar.
    ///
    /// Tests the path where bias is None — only weight multiplication.
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn linear_nobias_batched_matches_scalar(
        weights in prop::collection::vec(-3.0f32..3.0, 6),   // 2*3 = 6
        lower_a_vals in prop::collection::vec(-3.0f32..3.0, 4),  // 2*2 = 4
        upper_a_vals in prop::collection::vec(-3.0f32..3.0, 4),
        lower_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
        upper_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
    ) {
        let in_features = 3;
        let out_features = 2;
        let num_out = 2;

        let weight = Array2::from_shape_vec((out_features, in_features), weights).unwrap();
        let layer = LinearLayer::new(weight, None)
            .map_err(|e| TestCaseError::fail(format!("LinearLayer::new failed: {e}")))?;

        let scalar_bounds = LinearBounds::new(
            Array2::from_shape_vec((num_out, out_features), lower_a_vals).unwrap(),
            Array1::from_vec(lower_b_vals),
            Array2::from_shape_vec((num_out, out_features), upper_a_vals).unwrap(),
            Array1::from_vec(upper_b_vals),
        ).unwrap();

        let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
            scalar_bounds.lower_a.clone().into_dyn(),
            scalar_bounds.lower_b.clone().into_dyn(),
            scalar_bounds.upper_a.clone().into_dyn(),
            scalar_bounds.upper_b.clone().into_dyn(),
            vec![out_features],
            vec![num_out],
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Linear_nobias", EXACT_EQUIV_TOLERANCES)?;
    }
}

// =============================================================================
// RESHAPE BATCHED CROWN EQUIV
// =============================================================================
//
// Reshape is a passthrough in batched mode (bounds.clone()). The scalar path
// also doesn't modify coefficient matrices for Reshape. This test confirms
// that the batched passthrough produces the same result as scalar propagation
// by going through the Layer enum's `propagate_crown_backward_batched` dispatch,
// which includes `validate_flatten_like` validation.
//
// Fix for #3253: previous test was vacuous — it compared raw input against
// itself without calling propagate_linear_batched.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Reshape batched CROWN matches scalar CROWN via Layer dispatch.
    ///
    /// Exercises the actual `propagate_crown_backward_batched` path on the
    /// Layer enum, which runs `validate_flatten_like` before the passthrough.
    /// Part of #3247, Fixes #3253.
    #[ntest::timeout(10000)]
    #[test]
    fn reshape_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 12),  // 2*6 = 12
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 12),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
    ) {
        let num_in = 6;  // flattened: 2*3 = 6
        let num_out = 2;

        let layer = ReshapeLayer::new(vec![3, 2]);

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, num_in,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();

        // Call through the Layer enum dispatch to exercise the real batched path
        // including validate_flatten_like validation.
        let layer_enum = Layer::Reshape(layer);
        let pre_act_lower = ArrayD::from_elem(IxDyn(&[num_in]), -1.0f32);
        let pre_act_upper = ArrayD::from_elem(IxDyn(&[num_in]), 1.0f32);
        let pre_activation = BoundedTensor::new(pre_act_lower, pre_act_upper)
            .map_err(|e| TestCaseError::fail(format!("BoundedTensor::new failed: {e}")))?;

        let actual = layer_enum
            .propagate_crown_backward_batched(&batched_bounds, Some(&pre_activation), None)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Reshape", EXACT_EQUIV_TOLERANCES)?;
    }
}

// =============================================================================
// TRANSPOSE BATCHED CROWN EQUIV
// =============================================================================
//
// Transpose needs input_shape set and operates on flattened indices.
// It permutes columns of the A matrices.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Transpose(perm=[1,0]) batched CROWN matches scalar CROWN.
    ///
    /// For input shape [2, 3] (6 elements), Transpose([1,0]) produces [3, 2].
    /// The scalar path uses propagate_linear, the batched path uses
    /// propagate_linear_batched with input_shape set.
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn transpose_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 12),  // 2*6 = 12
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 12),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
    ) {
        let num_in = 6;  // 2*3 flattened
        let num_out = 2;

        let mut layer = TransposeLayer::new(vec![1, 0]);
        layer.set_input_shape(vec![2, 3]);

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, num_in,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Transpose", EXACT_EQUIV_TOLERANCES)?;
    }

    /// Transpose(perm=[2,0,1]) 3D batched CROWN matches scalar CROWN.
    ///
    /// For input shape [2, 3, 2] (12 elements), Transpose([2,0,1]) -> [2, 2, 3].
    /// Tests a more complex permutation pattern.
    /// Part of #3247.
    #[ntest::timeout(10000)]
    #[test]
    fn transpose_3d_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 24),  // 2*12 = 24
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 24),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
    ) {
        let num_in = 12;  // 2*3*2 flattened
        let num_out = 2;

        let mut layer = TransposeLayer::new(vec![2, 0, 1]);
        layer.set_input_shape(vec![2, 3, 2]);

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, num_in,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Transpose_3d", EXACT_EQUIV_TOLERANCES)?;
    }
}

// =============================================================================
// TRANSPOSE GROUPED MULTI-DIMENSIONAL BATCHED CROWN EQUIV (#4171)
// =============================================================================
//
// When incoming BatchedLinearBounds have grouped (multi-dimensional) batch
// dimensions, TransposeLayer must flatten to block-diagonal before permuting.
// This proptest verifies the flat-grouped path introduced in #4171 Packet A
// matches the scalar oracle.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(100) })]

    /// Grouped transpose: BatchedLinearBounds::identity([2, 3]) through
    /// Transpose([1,0]) with input_shape=[3, 2] must match the scalar oracle.
    ///
    /// identity shape [2, 3] produces grouped bounds with batch_dims=[2], dim=3.
    /// Transpose([1,0]) swaps axis 0 and 1 on the input shape [3, 2] → output [2, 3].
    /// The flatten_to_block_diagonal path produces a flat [6, 6] A matrix.
    /// Part of #4171 Packet B.
    #[ntest::timeout(10000)]
    #[test]
    fn transpose_grouped_2d_identity_matches_scalar_4171(
        // Use a seed but don't need random coefficients — identity is deterministic.
        // The proptest verifies the path doesn't panic for different layer configs.
        perm_choice in 0u8..2,
    ) {
        let (input_shape, output_shape, perm) = if perm_choice == 0 {
            (vec![3usize, 2], vec![2, 3], vec![1, 0])
        } else {
            (vec![2, 3], vec![3, 2], vec![1, 0])
        };
        let total_elems: usize = input_shape.iter().product();

        let mut layer = TransposeLayer::new(perm);
        layer.set_input_shape(input_shape);

        let identity_grouped = BatchedLinearBounds::identity(&output_shape)
            .expect("grouped identity should construct");
        let identity_scalar = LinearBounds::identity(total_elems);

        let actual = layer.propagate_linear_batched(&identity_grouped)
            .map_err(|e| TestCaseError::fail(format!("grouped batched CROWN failed: {e}")))?;
        let expected = layer.propagate_linear(&identity_scalar)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();

        // After flattening, both should produce [total_elems, total_elems] A matrices.
        let actual_la = actual.lower_a();
        let expected_la = expected.lower_a().clone().into_dyn();
        prop_assert_eq!(actual_la.shape(), expected_la.shape(),
            "lower_a shape mismatch: grouped={:?} scalar={:?}", actual_la.shape(), expected_la.shape());
        for (idx, (&a, &e)) in actual_la.iter().zip(expected_la.iter()).enumerate() {
            prop_assert!((a - e).abs() < 1e-6,
                "lower_a[{idx}]: grouped={a} scalar={e}");
        }

        let actual_ua = actual.upper_a();
        let expected_ua = expected.upper_a().clone().into_dyn();
        for (idx, (&a, &e)) in actual_ua.iter().zip(expected_ua.iter()).enumerate() {
            prop_assert!((a - e).abs() < 1e-6,
                "upper_a[{idx}]: grouped={a} scalar={e}");
        }
    }

    /// Grouped transpose 3D: BatchedLinearBounds::identity([1, 3, 4]) through
    /// batched_transpose() with input_shape=[1, 4, 3] matches the scalar oracle.
    ///
    /// Tests a 3D grouped identity which exercises the full flatten path with
    /// batch_dims=[1, 3], dim=4 (Kokoro-shaped). Uses batched_transpose() which
    /// auto-swaps the last two dimensions: [1, 4, 3] -> [1, 3, 4].
    /// Part of #4171 Packet B.
    #[ntest::timeout(10000)]
    #[test]
    fn transpose_grouped_3d_identity_matches_scalar_4171(
        _dummy in 0u8..1,
    ) {
        let input_shape = vec![1usize, 4, 3];
        let output_shape = vec![1, 3, 4];
        let total_elems: usize = input_shape.iter().product();

        let mut layer = TransposeLayer::batched_transpose();
        layer.set_input_shape(input_shape);

        let identity_grouped = BatchedLinearBounds::identity(&output_shape)
            .expect("grouped 3D identity should construct");
        let identity_scalar = LinearBounds::identity(total_elems);

        let actual = layer.propagate_linear_batched(&identity_grouped)
            .map_err(|e| TestCaseError::fail(format!("grouped 3D batched CROWN failed: {e}")))?;
        let expected = layer.propagate_linear(&identity_scalar)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();

        let actual_la = actual.lower_a();
        let expected_la = expected.lower_a().clone().into_dyn();
        prop_assert_eq!(actual_la.shape(), expected_la.shape(),
            "lower_a shape: grouped={:?} scalar={:?}", actual_la.shape(), expected_la.shape());
        for (idx, (&a, &e)) in actual_la.iter().zip(expected_la.iter()).enumerate() {
            prop_assert!((a - e).abs() < 1e-6,
                "lower_a[{idx}]: grouped={a} scalar={e}");
        }

        let actual_ua = actual.upper_a();
        let expected_ua = expected.upper_a().clone().into_dyn();
        for (idx, (&a, &e)) in actual_ua.iter().zip(expected_ua.iter()).enumerate() {
            prop_assert!((a - e).abs() < 1e-6,
                "upper_a[{idx}]: grouped={a} scalar={e}");
        }
    }
}

// =============================================================================
// CONV2D BATCHED CROWN EQUIV
// =============================================================================
//
// Conv2d CROWN backward applies transposed convolution per coefficient row.
// Both scalar and batched paths call the same conv2d_transpose function;
// the batched path adds a batch loop but produces identical per-row results.
// Bias path uses f64 accumulation in both paths.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Conv2d (with bias) batched CROWN matches scalar CROWN.
    ///
    /// kernel=(1,1,2,2), stride=(1,1), pad=(0,0), input=(3,3), output=(2,2).
    /// conv_out_size=4, conv_in_size=9. Incoming bounds: [2,4] -> output: [2,9].
    /// Part of #3247.
    #[ntest::timeout(60000)]
    #[test]
    fn conv2d_batched_matches_scalar(
        kernel_vals in prop::collection::vec(-3.0f32..3.0, 4),  // (1,1,2,2)
        bias_val in -3.0f32..3.0,
        lower_a_vals in prop::collection::vec(-3.0f32..3.0, 8),  // 2*4
        upper_a_vals in prop::collection::vec(-3.0f32..3.0, 8),
        lower_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
        upper_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
    ) {
        let in_h = 3;
        let in_w = 3;
        let num_out = 2;
        // output = (3+0-2)/1+1 = 2 in each dim, conv_out_size = 1*2*2 = 4
        let conv_out_size = 4;

        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), kernel_vals).unwrap();
        let bias = Array1::from_vec(vec![bias_val]);
        let layer = Conv2dLayer::with_input_shape(
            kernel, Some(bias), (1, 1), (0, 0), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv2dLayer::new failed: {e}")))?;

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, conv_out_size,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        // Pin the CPU dense budget. NY_DENSE_BUDGET_MB is process-global and
        // sibling tests set it to 1 MiB; when one of these two conv paths sees
        // that budget and the other does not, they legitimately diverge and this
        // equivalence assertion fails. It reproduced only in the full parallel
        // suite (passing in isolation and in small filtered runs), which is
        // exactly the signature of that race.
        let (expected, actual) = crate::tests::with_crown_dense_budget_mb("2048", || {
            let expected = layer.propagate_linear(&scalar_bounds).map(|b| b.into_owned());
            let actual = layer.propagate_linear_batched(&batched_bounds, None);
            (expected, actual)
        });
        let expected = expected
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;
        let actual = actual
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Conv2d", EXACT_EQUIV_TOLERANCES)?;
    }

    /// Conv2d (no bias) batched CROWN matches scalar CROWN.
    ///
    /// Tests the path where bias is None — only conv_transpose affects coefficients.
    /// Part of #3247.
    #[ntest::timeout(60000)]
    #[test]
    fn conv2d_nobias_batched_matches_scalar(
        kernel_vals in prop::collection::vec(-3.0f32..3.0, 4),
        lower_a_vals in prop::collection::vec(-3.0f32..3.0, 8),
        upper_a_vals in prop::collection::vec(-3.0f32..3.0, 8),
        lower_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
        upper_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
    ) {
        let in_h = 3;
        let in_w = 3;
        let num_out = 2;
        let conv_out_size = 4;

        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), kernel_vals).unwrap();
        let layer = Conv2dLayer::with_input_shape(
            kernel, None, (1, 1), (0, 0), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv2dLayer::new failed: {e}")))?;

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, conv_out_size,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        // See the bias variant above: NY_DENSE_BUDGET_MB is process-global and a
        // sibling test setting it to 1 MiB makes these two paths diverge.
        let (expected, actual) = crate::tests::with_crown_dense_budget_mb("2048", || {
            let expected = layer.propagate_linear(&scalar_bounds).map(|b| b.into_owned());
            let actual = layer.propagate_linear_batched(&batched_bounds, None);
            (expected, actual)
        });
        let expected = expected
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;
        let actual = actual
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Conv2d_nobias", EXACT_EQUIV_TOLERANCES)?;
    }
}

// =============================================================================
// CONV1D BATCHED CROWN EQUIV
// =============================================================================
//
// Conv1d CROWN backward applies transposed 1D convolution per coefficient row.
// Both paths call conv1d_transpose with the same data.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Conv1d (with bias) batched CROWN matches scalar CROWN.
    ///
    /// kernel=(1,1,3), stride=1, pad=0, input_len=5, output_len=3.
    /// conv_out_size=3, conv_in_size=5. Incoming bounds: [2,3] -> output: [2,5].
    /// Part of #3247.
    #[ntest::timeout(60000)]
    #[test]
    fn conv1d_batched_matches_scalar(
        kernel_vals in prop::collection::vec(-3.0f32..3.0, 3),  // (1,1,3)
        bias_val in -3.0f32..3.0,
        lower_a_vals in prop::collection::vec(-3.0f32..3.0, 6),  // 2*3
        upper_a_vals in prop::collection::vec(-3.0f32..3.0, 6),
        lower_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
        upper_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
    ) {
        let input_len = 5;
        let num_out = 2;
        // output_len = (5-3)/1+1 = 3, conv_out_size = 1*3 = 3
        let conv_out_size = 3;

        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), kernel_vals).unwrap();
        let bias = Array1::from_vec(vec![bias_val]);
        let layer = Conv1dLayer::with_input_length(
            kernel, Some(bias), 1, 0, input_len,
        ).map_err(|e| TestCaseError::fail(format!("Conv1dLayer::new failed: {e}")))?;

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, conv_out_size,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Conv1d", EXACT_EQUIV_TOLERANCES)?;
    }

    /// Conv1d (no bias) batched CROWN matches scalar CROWN.
    ///
    /// Tests the no-bias path where only conv1d_transpose affects coefficients.
    /// Part of #3247.
    #[ntest::timeout(60000)]
    #[test]
    fn conv1d_nobias_batched_matches_scalar(
        kernel_vals in prop::collection::vec(-3.0f32..3.0, 3),
        lower_a_vals in prop::collection::vec(-3.0f32..3.0, 6),
        upper_a_vals in prop::collection::vec(-3.0f32..3.0, 6),
        lower_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
        upper_b_vals in prop::collection::vec(-3.0f32..3.0, 2),
    ) {
        let input_len = 5;
        let num_out = 2;
        let conv_out_size = 3;

        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), kernel_vals).unwrap();
        let layer = Conv1dLayer::with_input_length(
            kernel, None, 1, 0, input_len,
        ).map_err(|e| TestCaseError::fail(format!("Conv1dLayer::new failed: {e}")))?;

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out, conv_out_size,
            lower_a_vals, lower_b_vals,
            upper_a_vals, upper_b_vals,
        );

        let expected = layer.propagate_linear(&scalar_bounds)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?
            .into_owned();
        let actual = layer.propagate_linear_batched(&batched_bounds)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Conv1d_nobias", EXACT_EQUIV_TOLERANCES)?;
    }
}

// =============================================================================
// SIGMOID (NONLINEAR ACTIVATION) BATCHED CROWN EQUIV
// =============================================================================
//
// First batched-vs-scalar equivalence proptest for a nonlinear elementwise
// activation. Tests the `crown_elementwise_backward_batched` code path that
// ALL elementwise activations share via the `impl_elementwise_activation!` macro.
// Part of #3400.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Sigmoid batched CROWN matches scalar CROWN.
    ///
    /// This is the first proptest exercising `crown_elementwise_backward_batched`
    /// for a nonlinear activation. All 29+ elementwise activations share this code
    /// path, so verifying Sigmoid also validates the shared batched infrastructure.
    ///
    /// Tests with 3 input neurons and 2 output neurons (bound rows).
    /// Pre-activation bounds are drawn from [-5, 5] to cover the full sigmoid
    /// transition (near-zero sigmoid, mid-range, and near-one sigmoid).
    /// Part of #3400.
    #[ntest::timeout(10000)]
    #[test]
    fn sigmoid_batched_matches_scalar(
        // Pre-activation bounds: lower in [-5, 5], delta in [0, 3]
        pre_l0 in -5.0f32..5.0, pre_d0 in 0.0f32..3.0,
        pre_l1 in -5.0f32..5.0, pre_d1 in 0.0f32..3.0,
        pre_l2 in -5.0f32..5.0, pre_d2 in 0.0f32..3.0,
        // Incoming bound coefficients and biases
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
    ) {
        let num_in = 3;
        let num_out = 2;

        let pre_activation = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[num_in]), vec![pre_l0, pre_l1, pre_l2]).unwrap(),
            ArrayD::from_shape_vec(
                IxDyn(&[num_in]),
                vec![pre_l0 + pre_d0, pre_l1 + pre_d1, pre_l2 + pre_d2],
            )
            .unwrap(),
        )
        .expect("valid pre-activation bounds");

        let layer = SigmoidLayer::new();

        let (scalar_bounds, batched_bounds) = build_bounds_pair(
            num_out,
            num_in,
            lower_a_vals,
            lower_b_vals,
            upper_a_vals,
            upper_b_vals,
        );

        let expected = layer
            .propagate_linear_with_bounds(&scalar_bounds, &pre_activation)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;
        let actual = layer
            .propagate_linear_batched_with_bounds(&batched_bounds, &pre_activation)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        assert_batched_equiv(&actual, &expected, "Sigmoid", EXACT_EQUIV_TOLERANCES)?;
    }
}
