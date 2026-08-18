// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN backward tests for seeded asymmetric suffixes.

use ny_core::{GpuCrownSeed, FALLBACK_BOUND};

use super::conv::build_small_conv_linear_case;
use super::*;

fn make_identity_seed(spec: &[f32], num_specs: usize, current_dim: usize) -> GpuCrownSeed {
    GpuCrownSeed {
        lower_a: spec.to_vec().into(),
        upper_a: spec.to_vec().into(),
        lower_b: vec![0.0; num_specs].into(),
        upper_b: vec![0.0; num_specs].into(),
        num_specs,
        current_dim,
    }
}

fn assert_seeded_gpu_matches_cpu(
    device: &WgpuDevice,
    layers: &[GpuCrownLayer],
    seed: &GpuCrownSeed,
    inp_l: &[f32],
    inp_u: &[f32],
    eps: f32,
) -> GpuCrownResult {
    let gpu = device
        .crown_backward_gpu_seeded(layers, seed, inp_l, inp_u)
        .expect("seeded GPU CROWN backward should succeed");
    let (cpu_l, cpu_u) = cpu_crown_backward_seeded(layers, seed, inp_l, inp_u);
    for i in 0..seed.num_specs {
        let dl = (gpu.lower_bounds[i] - cpu_l[i]).abs();
        let du = (gpu.upper_bounds[i] - cpu_u[i]).abs();
        assert!(
            dl <= eps,
            "seeded lower[{i}] GPU={} CPU={} diff={dl}",
            gpu.lower_bounds[i],
            cpu_l[i]
        );
        assert!(
            du <= eps,
            "seeded upper[{i}] GPU={} CPU={} diff={du}",
            gpu.upper_bounds[i],
            cpu_u[i]
        );
        assert!(
            gpu.lower_bounds[i] <= gpu.upper_bounds[i] + eps,
            "seeded spec {i}: lower {} > upper {}",
            gpu.lower_bounds[i],
            gpu.upper_bounds[i]
        );
    }
    gpu
}

fn cpu_crown_backward_seeded(
    layers: &[GpuCrownLayer],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut a_l = seed.lower_a.to_vec();
    let mut a_u = seed.upper_a.to_vec();
    let mut b_l = seed.lower_b.to_vec();
    let mut b_u = seed.upper_b.to_vec();
    let mut dim = seed.current_dim;
    let num_specs = seed.num_specs;

    for layer in layers {
        match layer {
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                cpu_activation_backward(
                    &mut a_l,
                    &mut a_u,
                    &mut b_l,
                    &mut b_u,
                    lower_slope,
                    upper_slope,
                    lower_intercept,
                    upper_intercept,
                    num_specs,
                    *num_neurons,
                );
            }
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
                ..
            } => {
                cpu_linear_backward(
                    &mut a_l,
                    &mut a_u,
                    &mut b_l,
                    &mut b_u,
                    weight,
                    bias.as_deref(),
                    num_specs,
                    *out_features,
                    *in_features,
                );
                dim = *in_features;
            }
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
                let spatial = out_h * out_w;
                let total_spatial = num_specs * spatial;
                let kernel_cols = in_channels * kernel_h * kernel_w;
                let flat_input_dim = in_channels * in_h * in_w;

                if let Some(expanded_bias) = bias_expanded {
                    for s in 0..num_specs {
                        let (mut lb, mut ub) = (0.0f32, 0.0f32);
                        for j in 0..(out_channels * spatial) {
                            lb += a_l[s * out_channels * spatial + j] * expanded_bias[j];
                            ub += a_u[s * out_channels * spatial + j] * expanded_bias[j];
                        }
                        b_l[s] += lb;
                        b_u[s] += ub;
                    }
                }

                let mut reshaped_l = vec![0.0f32; total_spatial * out_channels];
                let mut reshaped_u = vec![0.0f32; total_spatial * out_channels];
                for s in 0..num_specs {
                    for pos in 0..spatial {
                        for oc in 0..*out_channels {
                            let src_idx = s * out_channels * spatial + oc * spatial + pos;
                            let dst_idx = (s * spatial + pos) * out_channels + oc;
                            reshaped_l[dst_idx] = a_l[src_idx];
                            reshaped_u[dst_idx] = a_u[src_idx];
                        }
                    }
                }

                let mut gemm_l = vec![0.0f32; total_spatial * kernel_cols];
                let mut gemm_u = vec![0.0f32; total_spatial * kernel_cols];
                for row in 0..total_spatial {
                    for col in 0..kernel_cols {
                        let (mut sl, mut su) = (0.0f32, 0.0f32);
                        for k in 0..*out_channels {
                            let w = weight_col[k * kernel_cols + col];
                            sl += reshaped_l[row * out_channels + k] * w;
                            su += reshaped_u[row * out_channels + k] * w;
                        }
                        gemm_l[row * kernel_cols + col] = sl;
                        gemm_u[row * kernel_cols + col] = su;
                    }
                }

                let mut new_l = vec![0.0f32; num_specs * flat_input_dim];
                let mut new_u = vec![0.0f32; num_specs * flat_input_dim];
                for s in 0..num_specs {
                    for ic in 0..*in_channels {
                        for ih in 0..*in_h {
                            for iw_pos in 0..*in_w {
                                let flat_idx = ic * in_h * in_w + ih * in_w + iw_pos;
                                let (mut sum_l, mut sum_u) = (0.0f32, 0.0f32);
                                for ki in 0..*kernel_h {
                                    let ih_plus_ph = ih + pad_h;
                                    if ih_plus_ph < ki {
                                        continue;
                                    }
                                    let numerator_h = ih_plus_ph - ki;
                                    if numerator_h % stride_h != 0 {
                                        continue;
                                    }
                                    let gy = numerator_h / stride_h;
                                    if gy >= *out_h {
                                        continue;
                                    }
                                    for kj in 0..*kernel_w {
                                        let iw_plus_pw = iw_pos + pad_w;
                                        if iw_plus_pw < kj {
                                            continue;
                                        }
                                        let numerator_w = iw_plus_pw - kj;
                                        if numerator_w % stride_w != 0 {
                                            continue;
                                        }
                                        let gx = numerator_w / stride_w;
                                        if gx >= *out_w {
                                            continue;
                                        }
                                        let gemm_row = s * spatial + gy * out_w + gx;
                                        let gemm_col =
                                            ic * kernel_h * kernel_w + ki * kernel_w + kj;
                                        sum_l += gemm_l[gemm_row * kernel_cols + gemm_col];
                                        sum_u += gemm_u[gemm_row * kernel_cols + gemm_col];
                                    }
                                }
                                new_l[s * flat_input_dim + flat_idx] = sum_l;
                                new_u[s * flat_input_dim + flat_idx] = sum_u;
                            }
                        }
                    }
                }
                a_l = new_l;
                a_u = new_u;
                dim = flat_input_dim;
            }
            GpuCrownLayer::ActivationReluDualAlpha {
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
                num_neurons,
            } => {
                cpu_dual_alpha_activation_backward(
                    &mut a_l,
                    &mut a_u,
                    &mut b_l,
                    &mut b_u,
                    lower_pos_slope,
                    cross_slope,
                    upper_neg_slope,
                    cross_intercept,
                    num_specs,
                    *num_neurons,
                );
            }
            GpuCrownLayer::MaxPool2d {
                routing,
                ibp_lower,
                ibp_upper,
                input_dim,
                output_dim,
            } => {
                cpu_maxpool2d_backward(
                    &mut a_l,
                    &mut a_u,
                    &mut b_l,
                    &mut b_u,
                    routing,
                    ibp_lower,
                    ibp_upper,
                    num_specs,
                    *input_dim,
                    *output_dim,
                );
                dim = *input_dim;
            }
        }
    }

    cpu_concretize(
        &a_l,
        &a_u,
        &b_l,
        &b_u,
        input_lower,
        input_upper,
        num_specs,
        dim,
    )
}

#[test]
fn test_crown_backward_gpu_seeded_identity_matches_spec_entrypoint_3813() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    let case = build_small_conv_linear_case();
    let spec = identity_spec(case.out_dim);
    let seed = make_identity_seed(&spec, case.out_dim, case.out_dim);

    let seeded = device
        .crown_backward_gpu_seeded(&case.layers, &seed, &case.input_lower, &case.input_upper)
        .expect("seeded GPU identity path should succeed");
    let legacy = device
        .crown_backward_gpu(
            &case.layers,
            &spec,
            case.out_dim,
            &case.input_lower,
            &case.input_upper,
        )
        .expect("legacy GPU identity path should succeed");

    for i in 0..case.out_dim {
        assert!(
            (seeded.lower_bounds[i] - legacy.lower_bounds[i]).abs() <= 1e-4,
            "identity seeded lower[{i}] mismatch: seeded={} legacy={}",
            seeded.lower_bounds[i],
            legacy.lower_bounds[i]
        );
        assert!(
            (seeded.upper_bounds[i] - legacy.upper_bounds[i]).abs() <= 1e-4,
            "identity seeded upper[{i}] mismatch: seeded={} legacy={}",
            seeded.upper_bounds[i],
            legacy.upper_bounds[i]
        );
    }
}

#[test]
fn test_crown_backward_gpu_seeded_conv2d_asymmetric_suffix_matches_cpu_3813() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    let case = build_small_conv_linear_case();
    let seed = GpuCrownSeed {
        lower_a: vec![1.0, -0.5, 0.2, 0.7].into(),
        upper_a: vec![0.8, -0.3, 0.4, 1.1].into(),
        lower_b: vec![0.05, -0.02].into(),
        upper_b: vec![0.1, 0.03].into(),
        num_specs: case.out_dim,
        current_dim: case.out_dim,
    };

    assert_seeded_gpu_matches_cpu(
        &device,
        &case.layers,
        &seed,
        &case.input_lower,
        &case.input_upper,
        1e-3,
    );
}

#[test]
fn test_crown_backward_gpu_seeded_fallback_bound_coeff_degrades_row_2708() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let layers = vec![GpuCrownLayer::Linear {
        weight: vec![1.0].into(),
        bias: None,
        out_features: 1,
        in_features: 1,
        cert_err: Default::default(),
    }];
    let seed = GpuCrownSeed {
        lower_a: vec![FALLBACK_BOUND].into(),
        upper_a: vec![FALLBACK_BOUND].into(),
        lower_b: vec![0.0].into(),
        upper_b: vec![0.0].into(),
        num_specs: 1,
        current_dim: 1,
    };

    let result = device
        .crown_backward_gpu_seeded(&layers, &seed, &[0.25], &[0.5])
        .expect("GPU seeded fallback-bound sentinel should execute");

    assert_eq!(
        result.lower_bounds,
        vec![-FALLBACK_BOUND],
        "exact FALLBACK_BOUND lower coefficient is the GPU overflow sentinel and must degrade"
    );
    assert_eq!(
        result.upper_bounds,
        vec![FALLBACK_BOUND],
        "exact FALLBACK_BOUND upper coefficient is the GPU overflow sentinel and must degrade"
    );
}
