// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for sequential CROWN propagation.
//!
//! Extracted from `crown.rs` inline tests as part of #4233 Packet A.

use super::bounds_validation::has_nan_bounds;
use super::{has_degraded_bounds, tighten_crown_output, try_extract_single_gpu_layer, Network};
use crate::layers::{Conv1dLayer, Layer, LinearLayer, ReLULayer, SkipMergeLayer};
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::{GpuCrownLayer, Result};
use ny_tensor::BoundedTensor;

#[test]
fn propagate_crown_skip_merge_identity() -> Result<()> {
    let mut network = Network::new();
    network.add_layer(Layer::SkipMerge(SkipMergeLayer::new()));

    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    let output = network.propagate_crown(&input)?;

    // propagate_crown uses concretize_sound which applies directed rounding
    // (next_down_f32 / next_up_f32). For exact 0.0 input: next_down_f32(0.0)
    // = -1e-45 (smallest negative subnormal). Soundness: output must contain
    // the true range [0.0, 1.0].
    assert!(
        output.lower()[[0]] <= 0.0,
        "lower bound must be <= true lower 0.0, got {}",
        output.lower()[[0]]
    );
    assert!(
        output.upper()[[0]] >= 1.0,
        "upper bound must be >= true upper 1.0, got {}",
        output.upper()[[0]]
    );
    // Bounds should be tight (within 1 ULP of directed rounding).
    assert!(
        (output.lower()[[0]] - 0.0).abs() < 1e-6,
        "lower bound should be close to 0.0, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-6,
        "upper bound should be close to 1.0, got {}",
        output.upper()[[0]]
    );
    Ok(())
}

/// Verify that propagate_crown falls back to IBP output bounds when CROWN
/// backward produces NaN coefficients. A linear layer with f32::INFINITY
/// weight causes NaN during concretization (inf * 0 = NaN), triggering
/// the post-concretize guard.
#[test]
fn propagate_crown_inf_weight_per_element_intersection() -> Result<()> {
    let weight = arr2(&[[f32::INFINITY]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None)?));

    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    let output = network.propagate_crown(&input)?;

    // Flow: CROWN backward detects non-finite A-matrix → zeroes row, sets
    // bias to ±inf → concretize_sound sanitizes to [-inf, +inf] →
    // tighten_crown_output's has_nan_bounds returns false (no NaN) →
    // per-element intersection with IBP: max(-inf, ibp_lower) = ibp_lower,
    // min(+inf, ibp_upper) = ibp_upper → result equals IBP bounds.
    //
    // IBP through linear with inf weight and input [0,1]:
    // w_pos=inf, w_neg=0: lower = inf*0 + 0*1 = NaN → new_repaired(Conservative)
    // repairs NaN to -inf; upper = inf*1 + 0*0 = +inf is PRESERVED (an inf
    // weight makes the true output unbounded in f32 — every f(x) for x > 0
    // overflows to +inf — so any finite upper bound would be unsound).
    // Per-element intersection with CROWN [-inf, +inf] yields [-inf, +inf].
    assert!(
        has_degraded_bounds(&output),
        "inf-weight output is honestly unbounded; the degradation predicate must flag it"
    );
    assert_eq!(
        output.lower()[[0]],
        f32::NEG_INFINITY,
        "NaN lower (inf*0) must repair to the conservative -inf, not a finite clamp"
    );
    assert_eq!(
        output.upper()[[0]],
        f32::INFINITY,
        "+inf upper must be preserved: the true output overflows f32 for any x > 0"
    );

    assert!(
        !output.lower()[[0]].is_nan(),
        "lower bound should not be NaN"
    );
    assert!(
        !output.upper()[[0]].is_nan(),
        "upper bound should not be NaN"
    );
    Ok(())
}

/// #2681: Soundness regression test — CROWN through a network with
/// differentiated output rows produces sound, non-NaN bounds.
///
/// Note: A single linear layer does NOT trigger CROWN A-matrix overflow
/// (A = I @ W = W, all entries finite). True per-row overflow requires
/// deep networks where A-matrices compound. The per-row IBP fallback
/// is tested at the function level in `tighten_crown_output_per_row_ibp_fallback_2681`.
#[test]
fn propagate_crown_soundness_regression_2681() -> Result<()> {
    // 2-input, 2-output linear layer:
    //   row 0: [1e20, 1e20] → large weights, CROWN handles correctly
    //   row 1: [1.0, -0.5] → moderate weights, different function
    let weight = arr2(&[[1e20_f32, 1e20_f32], [1.0_f32, -0.5_f32]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None)?));

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )?;

    let crown_output = network.propagate_crown(&input)?;
    let ibp_output = network.propagate_ibp(&input)?;

    // Both outputs should be non-NaN
    assert!(
        !crown_output.lower()[[0]].is_nan(),
        "row 0 lower should not be NaN"
    );
    assert!(
        !crown_output.upper()[[0]].is_nan(),
        "row 0 upper should not be NaN"
    );

    // Both rows: CROWN should be at least as tight as IBP.
    // For a single linear layer, CROWN = exact = IBP.
    for row in 0..2 {
        assert!(
            crown_output.lower()[[row]] >= ibp_output.lower()[[row]] - 1e-5,
            "row {row} CROWN lower ({}) should be >= IBP lower ({})",
            crown_output.lower()[[row]],
            ibp_output.lower()[[row]]
        );
        assert!(
            crown_output.upper()[[row]] <= ibp_output.upper()[[row]] + 1e-5,
            "row {row} CROWN upper ({}) should be <= IBP upper ({})",
            crown_output.upper()[[row]],
            ibp_output.upper()[[row]]
        );
    }

    Ok(())
}

/// Same test for propagate_crown_ibp — verify the guard works on that path too.
#[test]
fn propagate_crown_ibp_nan_falls_back_to_ibp() -> Result<()> {
    let weight = arr2(&[[f32::INFINITY]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None)?));

    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    let output = network.propagate_crown_ibp(&input)?;

    // The IBP fallback must eliminate NaN (repaired to the conservative ±inf
    // by new_repaired(Conservative)) without inverting the interval. The ±inf
    // endpoints themselves are the correct result: an inf weight makes the
    // true output overflow f32 for any x > 0, so no finite bound is provable
    // in either direction.
    assert!(
        !has_nan_bounds(&output),
        "CROWN-IBP output must not contain NaN after IBP fallback"
    );
    assert!(
        output.lower()[[0]] <= output.upper()[[0]],
        "CROWN-IBP output must not contain inverted intervals after IBP fallback"
    );
    assert_eq!(
        output.lower()[[0]],
        f32::NEG_INFINITY,
        "NaN lower (inf*0) must repair to the conservative -inf, not a finite clamp"
    );
    assert_eq!(
        output.upper()[[0]],
        f32::INFINITY,
        "+inf upper must be preserved: the true output overflows f32 for any x > 0"
    );
    Ok(())
}

/// Regression: sequential network CROWN with Tile layer should succeed.
///
/// Before the explicit `Layer::Tile` arm in `crown_backward_step`, Tile fell
/// to the wildcard dispatch which called `propagate_linear()` without
/// `set_input_shape()`, causing an `UnsupportedConfiguration` hard error
/// instead of proper CROWN backward propagation.
#[test]
fn propagate_crown_with_tile_layer_succeeds() -> Result<()> {
    use crate::layers::TileLayer;
    use crate::ReLULayer;

    // Linear(2->4) -> Tile(axis=-1, reps=2) -> ReLU -> Linear(8->1)
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.4], [0.8, 0.1]]);
    let b1 = arr1(&[0.0, 0.0, 0.0, 0.0]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let tile = TileLayer::new(-1, 2); // axis=-1 on 1D(4) => output dim 8

    let w2 = arr2(&[[0.3, -0.2, 0.5, 0.1, -0.4, 0.6, -0.1, 0.2]]);
    let b2 = arr1(&[0.0]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::Tile(tile));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    let input = BoundedTensor::new(arr1(&[-1.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn())?;

    // This should not error — previously it would fail with UnsupportedConfiguration
    let crown_output = network.propagate_crown(&input)?;
    let ibp_output = network.propagate_ibp(&input)?;

    // Basic soundness: CROWN output should be finite
    assert!(
        crown_output.lower()[[0]].is_finite(),
        "CROWN lower bound should be finite"
    );
    assert!(
        crown_output.upper()[[0]].is_finite(),
        "CROWN upper bound should be finite"
    );
    // CROWN should be at least as tight as IBP (after #2990 intersection)
    assert!(
        crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - 1e-5,
        "CROWN lower ({}) should be >= IBP lower ({})",
        crown_output.lower()[[0]],
        ibp_output.lower()[[0]]
    );
    assert!(
        crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + 1e-5,
        "CROWN upper ({}) should be <= IBP upper ({})",
        crown_output.upper()[[0]],
        ibp_output.upper()[[0]]
    );

    Ok(())
}

/// Regression: sequential network CROWN with Slice layer should succeed.
///
/// Before the explicit `Layer::Slice` arm in `crown_backward_step`, Slice fell
/// to the wildcard dispatch. The explicit arm matches the Tile/Transpose pattern
/// for consistency and defense in depth.
///
/// Reference: #3105
#[test]
fn propagate_crown_with_slice_layer_succeeds() -> Result<()> {
    use crate::layers::SliceLayer;
    use crate::ReLULayer;

    // Linear(2->4) -> Slice(axis=0, 1..3) -> ReLU -> Linear(2->1)
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.4], [0.8, 0.1]]);
    let b1 = arr1(&[0.0, 0.0, 0.0, 0.0]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let slice = SliceLayer::new(0, 1, 3); // axis=0, [1..3) on 1D(4) → output dim 2

    let w2 = arr2(&[[0.3, -0.2]]);
    let b2 = arr1(&[0.0]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::Slice(slice));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    let input = BoundedTensor::new(arr1(&[-1.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn())?;

    let crown_output = network.propagate_crown(&input)?;
    let ibp_output = network.propagate_ibp(&input)?;

    assert!(
        crown_output.lower()[[0]].is_finite(),
        "CROWN lower bound should be finite"
    );
    assert!(
        crown_output.upper()[[0]].is_finite(),
        "CROWN upper bound should be finite"
    );
    assert!(
        crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - 1e-5,
        "CROWN lower ({}) should be >= IBP lower ({})",
        crown_output.lower()[[0]],
        ibp_output.lower()[[0]]
    );
    assert!(
        crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + 1e-5,
        "CROWN upper ({}) should be <= IBP upper ({})",
        crown_output.upper()[[0]],
        ibp_output.upper()[[0]]
    );

    Ok(())
}

/// Verify that has_degraded_bounds detects non-finite values (NaN, Inf).
/// Used by fast.rs (#2287) and SDP-CROWN for paths that need full
/// non-finite detection. The main tightening path uses `has_nan_bounds` (#2681).
#[test]
fn has_degraded_bounds_detects_non_finite() -> Result<()> {
    let valid = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    assert!(
        !has_degraded_bounds(&valid),
        "valid finite bounds should not be flagged"
    );

    // NaN detection
    let nan_lower =
        BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    assert!(
        has_degraded_bounds(&nan_lower),
        "NaN lower should be detected"
    );

    // Inf detection — the primary post-#2287 case (repaired elements become [-inf, +inf])
    let inf_bounds = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )?;
    assert!(
        has_degraded_bounds(&inf_bounds),
        "[-inf, +inf] repaired bounds should be flagged (used by fast.rs, SDP-CROWN)"
    );
    Ok(())
}

/// Verify that has_nan_bounds detects NaN but not ±Inf (#2681).
#[test]
fn has_nan_bounds_distinguishes_nan_from_inf() -> Result<()> {
    let valid = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    assert!(
        !has_nan_bounds(&valid),
        "valid finite bounds should not be flagged"
    );

    // NaN should be detected
    let nan_lower =
        BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    assert!(has_nan_bounds(&nan_lower), "NaN should be detected");

    // ±Inf should NOT be detected — handled by per-element intersection (#2681)
    let inf_bounds = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )?;
    assert!(
        !has_nan_bounds(&inf_bounds),
        "±Inf should not be flagged by has_nan_bounds — per-element intersection handles it"
    );
    Ok(())
}

/// #2681: When CROWN produces ±inf for some rows (from non-finite A-matrix
/// row fallback), tighten_crown_output should do per-row IBP fallback —
/// not fall back to IBP for the entire network.
///
/// This test verifies that healthy rows preserve their CROWN tightness
/// while overflow rows get replaced with IBP bounds.
#[test]
fn tighten_crown_output_per_row_ibp_fallback_2681() -> Result<()> {
    // CROWN output: row 0 overflowed → [-inf, +inf], row 1 has tight bounds
    let crown_output = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY, 2.0]).into_dyn(),
        arr1(&[f32::INFINITY, 4.0]).into_dyn(),
    )?;

    // IBP forward bounds: looser than row 1's CROWN bounds
    let forward_bounds =
        BoundedTensor::new(arr1(&[1.0, 0.0]).into_dyn(), arr1(&[5.0, 6.0]).into_dyn())?;

    let result = tighten_crown_output(crown_output, &forward_bounds, "test")?;

    // Row 0: was [-inf, +inf], intersection with IBP [1.0, 5.0] → [1.0, 5.0]
    assert_eq!(
        result.lower()[[0]],
        1.0,
        "overflow row lower should be IBP lower"
    );
    assert_eq!(
        result.upper()[[0]],
        5.0,
        "overflow row upper should be IBP upper"
    );

    // Row 1: CROWN [2.0, 4.0] is tighter than IBP [0.0, 6.0] → [2.0, 4.0]
    assert_eq!(
        result.lower()[[1]],
        2.0,
        "healthy row should keep CROWN lower"
    );
    assert_eq!(
        result.upper()[[1]],
        4.0,
        "healthy row should keep CROWN upper"
    );

    Ok(())
}

/// #2681: NaN bounds still trigger full IBP fallback (not per-element).
#[test]
fn tighten_crown_output_nan_falls_back_to_ibp() -> Result<()> {
    let crown_output = BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, 2.0]).into_dyn(),
        arr1(&[1.0, 4.0]).into_dyn(),
    )?;
    let forward_bounds =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[5.0, 6.0]).into_dyn())?;

    let result = tighten_crown_output(crown_output, &forward_bounds, "test")?;

    // NaN → fall back entirely to IBP
    assert_eq!(
        result.lower()[[0]],
        0.0,
        "NaN case: should be full IBP fallback"
    );
    assert_eq!(
        result.upper()[[0]],
        5.0,
        "NaN case: should be full IBP fallback"
    );
    assert_eq!(
        result.lower()[[1]],
        0.0,
        "NaN case: even healthy rows get IBP"
    );
    assert_eq!(
        result.upper()[[1]],
        6.0,
        "NaN case: even healthy rows get IBP"
    );
    Ok(())
}

/// #3301: Shape mismatch with matching element count → reshape CROWN to
/// match forward shape, then intersect. Verifies the #3300 defensive
/// reshape path (crown.rs lines 570-587).
///
/// Uses non-uniform per-element bounds to verify that the reshape preserves
/// row-major element ordering (element [i,j] maps to [i*3+j] in flat layout).
#[test]
fn tighten_crown_output_shape_mismatch_reshape_3301() -> Result<()> {
    // CROWN output: shape [2, 3], non-uniform bounds per element
    // Row-major layout: [-10,10], [-8,8], [-6,6], [-4,4], [-2,2], [-1,1]
    let crown_output = BoundedTensor::new(
        arr2(&[[-10.0, -8.0, -6.0], [-4.0, -2.0, -1.0]]).into_dyn(),
        arr2(&[[10.0, 8.0, 6.0], [4.0, 2.0, 1.0]]).into_dyn(),
    )?;

    // Forward bounds: shape [6], tighter per-element bounds
    // Element 0: [0,3], 1: [0,3], 2: [0,3], 3: [0,3], 4: [0,1.5], 5: [0,0.5]
    let forward_bounds = BoundedTensor::new(
        arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[3.0, 3.0, 3.0, 3.0, 1.5, 0.5]).into_dyn(),
    )?;

    // Same element count (6), different shapes → reshape and intersect
    let result = tighten_crown_output(crown_output, &forward_bounds, "test_reshape")?;

    // After reshape + intersection: max(crown_lower, fwd_lower), min(crown_upper, fwd_upper)
    assert_eq!(
        result.shape(),
        forward_bounds.shape(),
        "result should have forward shape"
    );

    // Element 0: max(-10,0)=0, min(10,3)=3  → [0, 3]
    assert_eq!(result.lower()[[0]], 0.0, "elem 0 lower");
    assert_eq!(result.upper()[[0]], 3.0, "elem 0 upper");

    // Element 4: max(-2,0)=0, min(2,1.5)=1.5  → [0, 1.5]
    assert_eq!(result.lower()[[4]], 0.0, "elem 4 lower");
    assert_eq!(
        result.upper()[[4]],
        1.5,
        "elem 4 upper: min(CROWN=2.0, fwd=1.5)"
    );

    // Element 5: max(-1,0)=0, min(1,0.5)=0.5  → [0, 0.5]
    assert_eq!(result.lower()[[5]], 0.0, "elem 5 lower");
    assert_eq!(
        result.upper()[[5]],
        0.5,
        "elem 5 upper: min(CROWN=1.0, fwd=0.5)"
    );

    Ok(())
}

/// #3301: Different element counts → skip intersection entirely, return
/// CROWN output unchanged. Verifies the logging-only skip path
/// (crown.rs lines 598-607).
#[test]
fn tighten_crown_output_different_element_count_skips_3301() -> Result<()> {
    // CROWN output: shape [4]
    let crown_output = BoundedTensor::new(
        arr1(&[-2.0, -3.0, -4.0, -5.0]).into_dyn(),
        arr1(&[2.0, 3.0, 4.0, 5.0]).into_dyn(),
    )?;

    // Forward bounds: shape [3] — different element count, no reshape possible
    let forward_bounds = BoundedTensor::new(
        arr1(&[0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[1.0, 1.0, 1.0]).into_dyn(),
    )?;

    let result = tighten_crown_output(crown_output.clone(), &forward_bounds, "test_skip")?;

    // Should return CROWN output unchanged — no intersection
    assert_eq!(
        result.shape(),
        crown_output.shape(),
        "result should keep CROWN shape"
    );
    for i in 0..4 {
        assert_eq!(
            result.lower()[[i]],
            crown_output.lower()[[i]],
            "element {i}: lower should be unchanged from CROWN"
        );
        assert_eq!(
            result.upper()[[i]],
            crown_output.upper()[[i]],
            "element {i}: upper should be unchanged from CROWN"
        );
    }
    Ok(())
}

#[test]
fn propagate_crown_with_cached_layer_bounds_matches_fresh_crown() -> Result<()> {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[1.0, -0.5], [0.25, 2.0]]),
        Some(arr1(&[0.1, -0.2])),
    )?));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.75, -1.25]]),
        Some(arr1(&[0.3])),
    )?));

    let input = BoundedTensor::new(arr1(&[-1.0, -0.5]).into_dyn(), arr1(&[1.5, 2.0]).into_dyn())?;

    let cached_layer_bounds =
        network.collect_crown_ibp_bounds_with_engine_and_deadline(&input, None, None)?;
    let cached_output = network
        .propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
            &input,
            &cached_layer_bounds,
            None,
            None,
            None,
        )?;
    let fresh_output = network.propagate_crown(&input)?;

    assert_eq!(cached_output.shape(), fresh_output.shape());
    for (cached, fresh) in cached_output
        .lower()
        .iter()
        .zip(fresh_output.lower().iter())
    {
        assert!(
            (cached - fresh).abs() < 1e-6,
            "lower mismatch: cached={cached} fresh={fresh}"
        );
    }
    for (cached, fresh) in cached_output
        .upper()
        .iter()
        .zip(fresh_output.upper().iter())
    {
        assert!(
            (cached - fresh).abs() < 1e-6,
            "upper mismatch: cached={cached} fresh={fresh}"
        );
    }

    Ok(())
}

#[test]
fn try_extract_single_gpu_layer_conv1d_maps_to_conv2d_descriptor() -> Result<()> {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.25, -0.5, 0.75, -1.0, 0.5, 0.125])
            .expect("conv1d kernel shape should be valid");
    let bias = arr1(&[0.2, -0.3]);
    let conv = Conv1dLayer::with_input_length(kernel, Some(bias), 2, 1, 7)?;
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 7]), vec![-1.0; 7])
            .expect("conv1d lower input shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[1, 7]), vec![1.0; 7])
            .expect("conv1d upper input shape should be valid"),
    )?;

    let mut gpu_layers = Vec::new();
    let extracted =
        try_extract_single_gpu_layer(&Layer::Conv1d(conv), &pre_activation, &mut gpu_layers);

    assert!(extracted.is_some(), "Conv1d should be GPU-extractable");
    assert_eq!(
        gpu_layers.len(),
        1,
        "Conv1d should add exactly one GPU layer"
    );

    match &gpu_layers[0] {
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
        } => {
            assert_eq!(weight_col.as_ref(), &[0.25, -0.5, 0.75, -1.0, 0.5, 0.125]);
            assert_eq!(
                bias_expanded.as_deref(),
                Some(&[0.2, 0.2, 0.2, 0.2, -0.3, -0.3, -0.3, -0.3][..])
            );
            assert_eq!(*out_channels, 2);
            assert_eq!(*in_channels, 1);
            assert_eq!(*kernel_h, 1);
            assert_eq!(*kernel_w, 3);
            assert_eq!(*stride_h, 1);
            assert_eq!(*stride_w, 2);
            assert_eq!(*pad_h, 0);
            assert_eq!(*pad_w, 1);
            assert_eq!(*out_h, 1);
            assert_eq!(*out_w, 4);
            assert_eq!(*in_h, 1);
            assert_eq!(*in_w, 7);
        }
        _ => panic!("expected Conv2d GPU descriptor"),
    }

    Ok(())
}

#[test]
fn try_extract_single_gpu_layer_conv1d_uses_input_length_for_flattened_input() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![0.1; 12])
        .expect("flattened conv1d kernel shape should be valid");
    let conv = Conv1dLayer::with_input_length(kernel, None, 1, 0, 4)?;
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[8]), vec![-0.5; 8])
            .expect("flattened lower input shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[8]), vec![0.5; 8])
            .expect("flattened upper input shape should be valid"),
    )?;

    let mut gpu_layers = Vec::new();
    let extracted =
        try_extract_single_gpu_layer(&Layer::Conv1d(conv), &pre_activation, &mut gpu_layers);

    assert!(
        extracted.is_some(),
        "Conv1d extraction should fall back to input_length when shape is flattened"
    );
    assert_eq!(gpu_layers.len(), 1);

    match &gpu_layers[0] {
        GpuCrownLayer::Conv2d {
            in_h,
            in_w,
            out_h,
            out_w,
            ..
        } => {
            assert_eq!(*in_h, 1);
            assert_eq!(*in_w, 4);
            assert_eq!(*out_h, 1);
            assert_eq!(*out_w, 2);
        }
        _ => panic!("expected Conv2d GPU descriptor"),
    }

    Ok(())
}

#[test]
fn try_extract_single_gpu_layer_conv1d_rejects_grouped_and_dilated_configs() -> Result<()> {
    let grouped_kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.1; 6])
        .expect("grouped conv1d kernel shape should be valid");
    let grouped = Conv1dLayer::with_input_length_full(grouped_kernel, None, 1, 0, 1, 2, 4)?;
    let dilated_kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.2; 6])
        .expect("dilated conv1d kernel shape should be valid");
    let dilated = Conv1dLayer::with_input_length_full(dilated_kernel, None, 1, 0, 2, 1, 6)?;
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 6]), vec![-1.0; 12])
            .expect("unsupported lower input shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[2, 6]), vec![1.0; 12])
            .expect("unsupported upper input shape should be valid"),
    )?;
    let mut gpu_layers = Vec::new();

    assert!(
        try_extract_single_gpu_layer(&Layer::Conv1d(grouped), &pre_activation, &mut gpu_layers)
            .is_none(),
        "grouped Conv1d must stay on CPU until grouped GPU backward exists"
    );
    assert!(
        try_extract_single_gpu_layer(&Layer::Conv1d(dilated), &pre_activation, &mut gpu_layers)
            .is_none(),
        "dilated Conv1d must stay on CPU until dilated GPU backward exists"
    );
    assert!(
        gpu_layers.is_empty(),
        "unsupported Conv1d variants should not push partial GPU descriptors"
    );

    Ok(())
}
