// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for GPU CROWN layer extraction (`gpu_extraction.rs`).
//!
//! These tests cover constant expansion, constant-arithmetic layer lowering,
//! Conv1d -> height-1 Conv2d metadata translation, unsupported-layer fallback,
//! and dynamic activation refresh through the extraction cache.
//!
//! Part of #4205.

use super::*;
use crate::layers::{
    AddConstantLayer, Conv1dLayer, DivConstantLayer, Layer, LinearLayer, MaxPool2dLayer,
    MulConstantLayer, ReLULayer, SubConstantLayer,
};
use crate::network::Network;
use ndarray::{arr0, arr1, arr2, Array1, Array2, Array3};
use ny_core::{GpuCrownLayer, Result};
use ny_tensor::BoundedTensor;
use std::sync::Mutex;

fn bounded_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_vec(lower.to_vec()).into_dyn(),
        Array1::from_vec(upper.to_vec()).into_dyn(),
    )
    .expect("valid bounds")
}

fn bounded_channels(channels: usize, length: usize, lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(
        Array2::from_elem((channels, length), lower).into_dyn(),
        Array2::from_elem((channels, length), upper).into_dyn(),
    )
    .expect("valid bounds")
}

fn bounded_spatial(
    lower: &[f32],
    upper: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> BoundedTensor {
    BoundedTensor::new(
        Array3::from_shape_vec((channels, height, width), lower.to_vec())
            .expect("lower spatial shape should be valid")
            .into_dyn(),
        Array3::from_shape_vec((channels, height, width), upper.to_vec())
            .expect("upper spatial shape should be valid")
            .into_dyn(),
    )
    .expect("valid spatial bounds")
}

fn assert_activation_descriptor(
    layer: &GpuCrownLayer,
    lower_slope: &[f32],
    upper_slope: &[f32],
    lower_intercept: &[f32],
    upper_intercept: &[f32],
) {
    match layer {
        GpuCrownLayer::Activation {
            lower_slope: actual_lower_slope,
            upper_slope: actual_upper_slope,
            lower_intercept: actual_lower_intercept,
            upper_intercept: actual_upper_intercept,
            num_neurons,
        } => {
            assert_eq!(*num_neurons, lower_slope.len());
            assert_eq!(actual_lower_slope, lower_slope);
            assert_eq!(actual_upper_slope, upper_slope);
            assert_eq!(actual_lower_intercept, lower_intercept);
            assert_eq!(actual_upper_intercept, upper_intercept);
        }
        _ => panic!("expected GPU extraction to yield Activation"),
    }
}

// Retained for dual-alpha cases (e.g. divergent lower/upper alpha); the
// stable/no-mask tests now expect the compact `Activation` descriptor (#4313).
#[allow(dead_code)]
fn assert_dual_alpha_activation_descriptor(
    layer: &GpuCrownLayer,
    lower_pos_slope: &[f32],
    cross_slope: &[f32],
    upper_neg_slope: &[f32],
    cross_intercept: &[f32],
) {
    match layer {
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: actual_lower_pos_slope,
            cross_slope: actual_cross_slope,
            upper_neg_slope: actual_upper_neg_slope,
            cross_intercept: actual_cross_intercept,
            num_neurons,
        } => {
            assert_eq!(*num_neurons, lower_pos_slope.len());
            assert_eq!(actual_lower_pos_slope, lower_pos_slope);
            assert_eq!(actual_cross_slope, cross_slope);
            assert_eq!(actual_upper_neg_slope, upper_neg_slope);
            assert_eq!(actual_cross_intercept, cross_intercept);
        }
        _ => panic!("expected GPU extraction to yield ActivationReluDualAlpha"),
    }
}

fn assert_conv1d_height_one_conv2d_descriptor(layer: &GpuCrownLayer) {
    match layer {
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
            ..
        } => {
            assert_eq!(weight_col.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            assert_eq!(*out_channels, 2);
            assert_eq!(*in_channels, 1);
            assert_eq!(*kernel_h, 1);
            assert_eq!(*kernel_w, 3);
            assert_eq!(*stride_h, 1);
            assert_eq!(*stride_w, 1);
            assert_eq!(*pad_h, 0);
            assert_eq!(*pad_w, 0);
            assert_eq!(*out_h, 1);
            assert_eq!(*out_w, 3);
            assert_eq!(*in_h, 1);
            assert_eq!(*in_w, 5);

            let bias_expanded = bias_expanded.as_ref().expect("Conv1d bias should expand");
            assert_eq!(bias_expanded.as_ref(), &[0.5, 0.5, 0.5, -0.5, -0.5, -0.5]);
        }
        _ => panic!("expected Conv1d GPU extraction to reuse Conv2d descriptor"),
    }
}

fn assert_unstable_relu_descriptor(layer: &GpuCrownLayer) {
    match layer {
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        } => {
            assert_eq!(*num_neurons, 1);
            assert_eq!(lower_slope, &vec![0.0]);
            assert_eq!(lower_intercept, &vec![0.0]);
            assert!(
                upper_slope[0] > 0.0 && upper_slope[0] < 1.0,
                "unstable ReLU upper slope should be a crossing chord, got {}",
                upper_slope[0]
            );
            assert!(
                upper_intercept[0] > 0.0,
                "unstable ReLU upper intercept should be positive, got {}",
                upper_intercept[0]
            );
        }
        _ => panic!("expected cached extraction to start with ReLU activation"),
    }
}

fn assert_linear_identity_descriptor(layer: &GpuCrownLayer) {
    match layer {
        GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            ..
        } => {
            assert_eq!(weight.as_ref(), &[1.0]);
            assert!(bias.is_none());
            assert_eq!(*out_features, 1);
            assert_eq!(*in_features, 1);
        }
        _ => panic!("expected linear suffix to remain cached"),
    }
}

fn assert_maxpool_descriptor(
    layer: &GpuCrownLayer,
    routing: &[u32],
    ibp_lower: &[f32],
    ibp_upper: &[f32],
    input_dim: usize,
    output_dim: usize,
) {
    match layer {
        GpuCrownLayer::MaxPool2d {
            routing: actual_routing,
            ibp_lower: actual_ibp_lower,
            ibp_upper: actual_ibp_upper,
            input_dim: actual_input_dim,
            output_dim: actual_output_dim,
        } => {
            assert_eq!(actual_routing, routing);
            assert_eq!(actual_ibp_lower, ibp_lower);
            assert_eq!(actual_ibp_upper, ibp_upper);
            assert_eq!(*actual_input_dim, input_dim);
            assert_eq!(*actual_output_dim, output_dim);
        }
        _ => panic!("expected GPU extraction to yield MaxPool2d"),
    }
}

#[test]
fn test_expand_constant_handles_exact_scalar_and_mismatch() {
    assert_eq!(
        expand_constant(&arr1(&[1.0f32, 2.0, 3.0]).into_dyn(), 3),
        Some(vec![1.0, 2.0, 3.0])
    );
    assert_eq!(
        expand_constant(&arr0(1.5f32).into_dyn(), 4),
        Some(vec![1.5; 4])
    );
    assert_eq!(expand_constant(&arr1(&[1.0f32, 2.0]).into_dyn(), 3), None);
}

#[test]
fn test_try_extract_single_gpu_layer_add_constant_uses_identity_slopes() {
    let layer = Layer::AddConstant(AddConstantLayer::new(arr0(1.5f32).into_dyn()));
    let pre_activation = bounded_1d(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
    let mut gpu_layers = Vec::new();

    assert_eq!(
        try_extract_single_gpu_layer(&layer, &pre_activation, &mut gpu_layers),
        Some(())
    );
    assert_eq!(gpu_layers.len(), 1);
    assert_activation_descriptor(
        &gpu_layers[0],
        &[1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0],
        &[1.5, 1.5, 1.5],
        &[1.5, 1.5, 1.5],
    );
}

#[test]
fn test_try_extract_single_gpu_layer_sub_constant_reverse_negates_slopes() {
    let layer = Layer::SubConstant(SubConstantLayer::new_reverse(arr0(2.0f32).into_dyn()));
    let pre_activation = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);
    let mut gpu_layers = Vec::new();

    assert_eq!(
        try_extract_single_gpu_layer(&layer, &pre_activation, &mut gpu_layers),
        Some(())
    );
    assert_activation_descriptor(
        &gpu_layers[0],
        &[-1.0, -1.0],
        &[-1.0, -1.0],
        &[2.0, 2.0],
        &[2.0, 2.0],
    );
}

#[test]
fn test_try_extract_single_gpu_layer_mul_and_div_constant_scale_slopes() {
    let pre_activation = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let mut mul_gpu_layers = Vec::new();
    let mul_layer = Layer::MulConstant(MulConstantLayer::new(arr0(2.0f32).into_dyn()));
    assert_eq!(
        try_extract_single_gpu_layer(&mul_layer, &pre_activation, &mut mul_gpu_layers),
        Some(())
    );
    assert_activation_descriptor(
        &mul_gpu_layers[0],
        &[2.0, 2.0],
        &[2.0, 2.0],
        &[0.0, 0.0],
        &[0.0, 0.0],
    );

    let mut div_gpu_layers = Vec::new();
    let div_layer = Layer::DivConstant(DivConstantLayer::new(arr0(4.0f32).into_dyn()));
    assert_eq!(
        try_extract_single_gpu_layer(&div_layer, &pre_activation, &mut div_gpu_layers),
        Some(())
    );
    assert_activation_descriptor(
        &div_gpu_layers[0],
        &[0.25, 0.25],
        &[0.25, 0.25],
        &[0.0, 0.0],
        &[0.0, 0.0],
    );
}

#[test]
fn test_try_extract_single_gpu_layer_div_constant_zero_returns_none() {
    let mut div = DivConstantLayer::new(arr0(1.0f32).into_dyn());
    div.constant = arr0(0.0f32).into_dyn();
    let layer = Layer::DivConstant(div);
    let pre_activation = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);
    let mut gpu_layers = Vec::new();

    assert_eq!(
        try_extract_single_gpu_layer(&layer, &pre_activation, &mut gpu_layers),
        None
    );
    assert!(
        gpu_layers.is_empty(),
        "unsupported extraction must not push layers"
    );
}

#[test]
fn test_try_extract_single_gpu_layer_conv1d_maps_to_height_one_conv2d() -> Result<()> {
    let kernel = ndarray::array![[[1.0f32, 2.0, 3.0]], [[4.0, 5.0, 6.0]]].into_dyn();
    let bias = arr1(&[0.5f32, -0.5]);
    let conv1d = Conv1dLayer::with_input_length(kernel, Some(bias), 1, 0, 5)?;
    let layer = Layer::Conv1d(conv1d);
    let pre_activation = bounded_channels(1, 5, -1.0, 1.0);
    let mut gpu_layers = Vec::new();

    assert_eq!(
        try_extract_single_gpu_layer(&layer, &pre_activation, &mut gpu_layers),
        Some(())
    );
    assert_eq!(gpu_layers.len(), 1);
    assert_conv1d_height_one_conv2d_descriptor(&gpu_layers[0]);

    Ok(())
}

#[test]
fn test_try_extract_single_gpu_layer_grouped_conv1d_returns_none() -> Result<()> {
    let kernel = ndarray::array![[[1.0f32, 2.0, 3.0]], [[4.0, 5.0, 6.0]]].into_dyn();
    let conv1d = Conv1dLayer::with_input_length_full(kernel, None, 1, 0, 1, 2, 5)?;
    let layer = Layer::Conv1d(conv1d);
    let pre_activation = bounded_channels(2, 5, -1.0, 1.0);
    let mut gpu_layers = Vec::new();

    assert_eq!(
        try_extract_single_gpu_layer(&layer, &pre_activation, &mut gpu_layers),
        None
    );
    assert!(
        gpu_layers.is_empty(),
        "grouped Conv1d must stay on CPU fallback path"
    );
    Ok(())
}

#[test]
fn test_extract_gpu_crown_layers_cached_refreshes_dynamic_relu_from_intermediate_bounds(
) -> Result<()> {
    let layers = vec![
        Layer::Linear(LinearLayer::new(arr2(&[[1.0f32]]), None)?),
        Layer::ReLU(ReLULayer::new()),
    ];
    let input = bounded_1d(&[0.0], &[0.0]);
    let cache = Mutex::new(None);

    let unstable_intermediate = vec![bounded_1d(&[-2.0], &[1.0])];
    let first = extract_gpu_crown_layers_cached(&layers, &unstable_intermediate, &input, &cache)
        .expect("initial extraction should succeed");
    assert_eq!(first.len(), 2);
    assert_unstable_relu_descriptor(&first[0]);

    let active_intermediate = vec![bounded_1d(&[1.0], &[2.0])];
    let second = extract_gpu_crown_layers_cached(&layers, &active_intermediate, &input, &cache)
        .expect("cache refresh should succeed");
    assert_activation_descriptor(&second[0], &[1.0], &[1.0], &[0.0], &[0.0]);
    assert_linear_identity_descriptor(&second[1]);

    Ok(())
}

#[test]
fn network_mutation_invalidates_static_gpu_crown_extraction() -> Result<()> {
    fn assert_linear(
        layer: &GpuCrownLayer,
        expected_weight: &[f32],
        expected_bias: Option<&[f32]>,
    ) {
        let GpuCrownLayer::Linear { weight, bias, .. } = layer else {
            panic!("expected GPU extraction to yield Linear");
        };
        assert_eq!(weight.as_ref(), expected_weight);
        assert_eq!(bias.as_deref(), expected_bias);
    }

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[1.0_f32]]),
        Some(arr1(&[0.25_f32])),
    )?));
    let input = bounded_1d(&[-1.0], &[1.0]);

    let first =
        extract_gpu_crown_layers_cached(network.layers(), &[], &input, &network.gpu_crown_cache)
            .expect("initial extraction should populate the static cache");
    assert_linear(&first[0], &[1.0], Some(&[0.25]));

    let Layer::Linear(linear) = &mut network.layers_mut()[0] else {
        panic!("test network should retain its Linear layer");
    };
    linear.replace_parameters(arr2(&[[2.0_f32]]), Some(arr1(&[0.5_f32])))?;

    let second =
        extract_gpu_crown_layers_cached(network.layers(), &[], &input, &network.gpu_crown_cache)
            .expect("post-mutation extraction should rebuild the static cache");
    assert_linear(&second[0], &[2.0], Some(&[0.5]));

    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[3.0_f32]]),
        Some(arr1(&[0.75_f32])),
    )?));
    let intermediate = vec![bounded_1d(&[-2.5], &[2.5])];
    let third = extract_gpu_crown_layers_cached(
        network.layers(),
        &intermediate,
        &input,
        &network.gpu_crown_cache,
    )
    .expect("post-append extraction should rebuild the static topology");
    assert_eq!(third.len(), 2);
    assert_linear(&third[0], &[3.0], Some(&[0.75]));
    assert_linear(&third[1], &[2.0], Some(&[0.5]));

    Ok(())
}

#[test]
fn test_try_extract_single_gpu_layer_maxpool2d_encodes_definite_winner_and_fallback() {
    let layer = Layer::MaxPool2d(MaxPool2dLayer::new((2, 2), (2, 2), (0, 0)));
    // Window 0 has a definite winner because idx 1 has lower=0.6 while every
    // competing upper bound is <= 0.5. Window 1 still falls back because its
    // winner candidate lower bound (0.3) is below another input's upper bound.
    let pre_activation = bounded_spatial(
        &[0.1, 0.6, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3],
        &[0.4, 0.8, 0.5, 0.6, 0.4, 0.5, 0.8, 0.9],
        1,
        2,
        4,
    );
    let mut gpu_layers = Vec::new();

    assert_eq!(
        try_extract_single_gpu_layer(&layer, &pre_activation, &mut gpu_layers),
        Some(())
    );
    assert_eq!(gpu_layers.len(), 1);
    assert_maxpool_descriptor(
        &gpu_layers[0],
        &[1, u32::MAX],
        &[0.6, 0.3],
        &[0.8, 0.9],
        8,
        2,
    );
}

#[test]
fn test_extract_gpu_crown_layers_cached_refreshes_dynamic_maxpool_routing() -> Result<()> {
    let layers = vec![Layer::MaxPool2d(MaxPool2dLayer::new(
        (2, 2),
        (2, 2),
        (0, 0),
    ))];
    let input = bounded_spatial(&[0.1, 0.6, 0.2, 0.3], &[0.4, 0.8, 0.5, 0.6], 1, 2, 2);
    let cache = Mutex::new(None);

    let first = extract_gpu_crown_layers_cached(&layers, &[], &input, &cache)
        .expect("initial maxpool extraction should succeed");
    assert_eq!(first.len(), 1);
    assert_maxpool_descriptor(&first[0], &[1], &[0.6], &[0.8], 4, 1);

    let updated_input = bounded_spatial(&[0.1, 0.3, 0.2, 0.25], &[0.4, 0.9, 0.5, 0.8], 1, 2, 2);
    let second = extract_gpu_crown_layers_cached(&layers, &[], &updated_input, &cache)
        .expect("maxpool cache refresh should succeed");
    assert_maxpool_descriptor(&second[0], &[u32::MAX], &[0.3], &[0.9], 4, 1);

    Ok(())
}

// --- Alpha-aware ReLU GPU extraction tests (#4312) ---

#[test]
fn test_extract_relu_gpu_layer_with_alpha_stable_positive_uses_identity() {
    // l >= 0: always-active neurons ignore alpha entirely.
    //
    // No neuron has diverging lower/upper alpha (all are stable), so the
    // four-slice dual-alpha upgrade (#4313) is unnecessary and extraction emits
    // the compact single-slope `Activation` descriptor — which encodes the exact
    // same identity relaxation (slope 1, intercept 0).
    let layer = extract_relu_gpu_layer_with_alpha(
        &[1.0, 2.0],
        &[3.0, 4.0],
        &[0.5, 0.5], // alpha values should be ignored
        &[0.5, 0.5],
        &[false, false],
    );
    assert_activation_descriptor(&layer, &[1.0, 1.0], &[1.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
}

#[test]
fn test_extract_relu_gpu_layer_with_alpha_stable_negative_uses_zero() {
    // u <= 0: always-inactive neurons ignore alpha entirely.
    //
    // No neuron needs dual alpha, so extraction emits the compact `Activation`
    // descriptor encoding the exact zero relaxation (slope 0, intercept 0).
    let layer = extract_relu_gpu_layer_with_alpha(
        &[-3.0, -2.0],
        &[-1.0, -0.5],
        &[0.5, 0.5],
        &[0.5, 0.5],
        &[false, false],
    );
    assert_activation_descriptor(&layer, &[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]);
}

#[test]
fn test_extract_relu_gpu_layer_with_alpha_crossing_with_active_alpha() {
    // Crossing neuron with active alpha: dual-alpha keeps alpha_lower, chord,
    // alpha_upper, and chord intercept as distinct slots.
    let layer = extract_relu_gpu_layer_with_alpha(
        &[-2.0],
        &[1.0],
        &[0.7], // optimized alpha
        &[0.3],
        &[true], // unstable
    );
    match &layer {
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            num_neurons,
        } => {
            assert_eq!(*num_neurons, 1);
            assert_eq!(
                lower_pos_slope[0], 0.7,
                "lower positive slope should be alpha_lower"
            );
            assert_eq!(
                upper_neg_slope[0], 0.3,
                "upper negative slope should be alpha_upper"
            );
            // Chord: u/(u-l) = 1/3, intercept = 2/3 (with conservative rounding)
            assert!(
                (cross_slope[0] - 1.0 / 3.0).abs() < 1e-5,
                "cross slope should be chord ~0.333, got {}",
                cross_slope[0]
            );
            assert!(
                (cross_intercept[0] - 2.0 / 3.0).abs() < 1e-5,
                "cross intercept should be chord ~0.667, got {}",
                cross_intercept[0]
            );
        }
        _ => panic!("expected ActivationReluDualAlpha"),
    }
}

#[test]
fn test_extract_relu_gpu_layer_with_alpha_crossing_without_mask_uses_heuristic() {
    // Crossing neuron but unstable_mask=false: falls back to heuristic.
    let layer_alpha = extract_relu_gpu_layer_with_alpha(
        &[-1.0],
        &[2.0],
        &[0.9],
        &[0.1],
        &[false], // not in unstable mask -> use heuristic
    );
    // Also extract with the heuristic directly for comparison.
    let mut heuristic_layers = Vec::new();
    let heuristic_pre = bounded_1d(&[-1.0], &[2.0]);
    try_extract_single_gpu_layer(
        &Layer::ReLU(ReLULayer::new()),
        &heuristic_pre,
        &mut heuristic_layers,
    );
    // With no neuron in the unstable mask, no dual-alpha divergence exists, so
    // alpha extraction emits the compact single-slope `Activation` descriptor
    // (#4313) — identical to the standalone heuristic extraction. Both must be
    // `Activation` with element-wise equal slopes/intercepts.
    match (&layer_alpha, &heuristic_layers[0]) {
        (
            GpuCrownLayer::Activation {
                lower_slope: a_ls,
                upper_slope: a_us,
                lower_intercept: a_li,
                upper_intercept: a_ui,
                ..
            },
            GpuCrownLayer::Activation {
                lower_slope: h_ls,
                upper_slope: h_us,
                lower_intercept: h_li,
                upper_intercept: h_ui,
                ..
            },
        ) => {
            assert_eq!(a_ls, h_ls, "lower slope should match heuristic lower slope");
            assert_eq!(a_us, h_us, "upper slope should match heuristic upper slope");
            assert_eq!(
                a_li, h_li,
                "lower intercept should match heuristic lower intercept"
            );
            assert_eq!(
                a_ui, h_ui,
                "upper intercept should match heuristic upper intercept"
            );
            assert_eq!(
                h_li,
                &vec![0.0],
                "heuristic lower intercept stays at zero for ReLU"
            );
        }
        _ => panic!("expected both extractions to be compact Activation descriptors"),
    }
}

#[test]
fn test_extract_relu_gpu_layer_with_alpha_mixed_neurons() {
    // Mix: neuron 0 = stable positive, neuron 1 = crossing with alpha,
    // neuron 2 = stable negative, neuron 3 = crossing without alpha mask.
    let layer = extract_relu_gpu_layer_with_alpha(
        &[1.0, -1.0, -2.0, -0.5],
        &[3.0, 2.0, -0.1, 1.5],
        &[0.0, 0.8, 0.0, 0.6],
        &[0.0, 0.2, 0.0, 0.4],
        &[false, true, false, false],
    );
    match &layer {
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            num_neurons,
        } => {
            assert_eq!(*num_neurons, 4);
            // Neuron 0: stable positive -> identity
            assert_eq!(lower_pos_slope[0], 1.0);
            assert_eq!(cross_slope[0], 1.0);
            assert_eq!(upper_neg_slope[0], 1.0);
            assert_eq!(cross_intercept[0], 0.0);
            // Neuron 1: crossing with alpha
            assert_eq!(lower_pos_slope[1], 0.8, "alpha_lower for unstable neuron 1");
            assert_eq!(upper_neg_slope[1], 0.2, "alpha_upper for unstable neuron 1");
            assert!(cross_slope[1] > 0.0 && cross_slope[1] < 1.0, "chord slope");
            assert!(cross_intercept[1] > 0.0, "chord intercept");
            // Neuron 2: stable negative -> zero
            assert_eq!(lower_pos_slope[2], 0.0);
            assert_eq!(cross_slope[2], 0.0);
            assert_eq!(upper_neg_slope[2], 0.0);
            assert_eq!(cross_intercept[2], 0.0);
            // Neuron 3: crossing but not masked -> heuristic
            // Heuristic for l=-0.5, u=1.5: u > -l -> alpha=1.0
            assert_eq!(lower_pos_slope[3], 1.0, "heuristic alpha for neuron 3");
            assert_eq!(
                upper_neg_slope[3], 1.0,
                "heuristic lower slope reused for upper-neg"
            );
        }
        _ => panic!("expected ActivationReluDualAlpha"),
    }
}
