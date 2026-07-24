// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest: GPU CROWN backward vs CPU reference implementation (#3397).
//!
//! Verifies that `crown_backward_gpu` produces bounds matching a pure-CPU
//! reference for small random networks. This catches GPU-specific bugs like
//! slopes buffer aliasing (#3444) and ping buffer sizing (#3446).

#[cfg(feature = "gpu-tests")]
mod gpu_crown_backward {
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};
    use crate::WgpuDevice;
    use ny_core::{GpuCrownBackward, GpuCrownLayer, GpuCrownResult};
    use proptest::prelude::*;

    mod budget;
    mod conv;
    mod dual_alpha;
    mod linear;
    mod maxpool;
    mod plan_cache;
    mod seeded;
    mod timing;

    use maxpool::cpu_maxpool2d_backward;

    /// IBP forward through a single linear layer to get pre-activation bounds.
    fn ibp_forward_linear(
        weight: &[f32],
        bias: &[f32],
        inp_l: &[f32],
        inp_u: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut pre_l = vec![0.0f32; out_dim];
        let mut pre_u = vec![0.0f32; out_dim];
        for j in 0..out_dim {
            let (mut lb, mut ub) = (bias[j], bias[j]);
            for k in 0..in_dim {
                let w = weight[j * in_dim + k];
                if w >= 0.0 {
                    lb += w * inp_l[k];
                    ub += w * inp_u[k];
                } else {
                    lb += w * inp_u[k];
                    ub += w * inp_l[k];
                }
            }
            pre_l[j] = lb;
            pre_u[j] = ub;
        }
        (pre_l, pre_u)
    }

    /// ReLU linear relaxation matching ny-propagate's `relu_linear_relaxation`.
    fn relu_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
        if l >= 0.0 {
            (1.0, 1.0, 0.0, 0.0)
        } else if u <= 0.0 {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let us = u / (u - l);
            let ui = -us * l;
            let ls = if u > -l { 1.0 } else { 0.0 };
            (ls, us, 0.0, ui)
        }
    }

    /// Compute ReLU slopes/intercepts from pre-activation bounds.
    fn relu_slopes(pre_l: &[f32], pre_u: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = pre_l.len();
        let (mut ls, mut us, mut li, mut ui) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for j in 0..n {
            let (a, b, c, d) = relu_relaxation(pre_l[j], pre_u[j]);
            ls.push(a);
            us.push(b);
            li.push(c);
            ui.push(d);
        }
        (ls, us, li, ui)
    }

    /// Identity spec matrix: (dim x dim) row-major.
    fn identity_spec(dim: usize) -> Vec<f32> {
        let mut spec = vec![0.0f32; dim * dim];
        for i in 0..dim {
            spec[i * dim + i] = 1.0;
        }
        spec
    }

    /// Run GPU CROWN backward and compare with CPU reference.
    fn assert_gpu_matches_cpu(
        device: &WgpuDevice,
        layers: &[GpuCrownLayer],
        num_specs: usize,
        inp_l: &[f32],
        inp_u: &[f32],
        eps: f32,
    ) -> GpuCrownResult {
        let spec = identity_spec(num_specs);
        let gpu = device
            .crown_backward_gpu(layers, &spec, num_specs, inp_l, inp_u)
            .expect("GPU CROWN backward should succeed");
        let (cpu_l, cpu_u) = cpu_crown_backward(layers, &spec, num_specs, inp_l, inp_u);
        for i in 0..num_specs {
            let dl = (gpu.lower_bounds[i] - cpu_l[i]).abs();
            let du = (gpu.upper_bounds[i] - cpu_u[i]).abs();
            assert!(
                dl <= eps,
                "lower[{i}] GPU={} CPU={} diff={dl}",
                gpu.lower_bounds[i],
                cpu_l[i]
            );
            assert!(
                du <= eps,
                "upper[{i}] GPU={} CPU={} diff={du}",
                gpu.upper_bounds[i],
                cpu_u[i]
            );
            assert!(
                gpu.lower_bounds[i] <= gpu.upper_bounds[i] + eps,
                "spec {i}: lower {} > upper {}",
                gpu.lower_bounds[i],
                gpu.upper_bounds[i]
            );
        }
        gpu
    }

    /// CPU reference CROWN backward: A-matrix propagation + concretization in f32.
    fn cpu_crown_backward(
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let first_dim = match &layers[0] {
            GpuCrownLayer::Linear { out_features, .. } => *out_features,
            GpuCrownLayer::Activation { num_neurons, .. }
            | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => *num_neurons,
            GpuCrownLayer::MaxPool2d { output_dim, .. } => *output_dim,
            GpuCrownLayer::Conv2d {
                out_channels,
                out_h,
                out_w,
                ..
            } => out_channels * out_h * out_w,
        };
        let mut a_l: Vec<f32> = spec.to_vec();
        let mut a_u: Vec<f32> = spec.to_vec();
        let mut b_l = vec![0.0f32; num_specs];
        let mut b_u = vec![0.0f32; num_specs];
        let mut dim = first_dim;

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
                    let spatial = out_h * out_w;
                    let total_spatial = num_specs * spatial;
                    let kernel_cols = in_channels * kernel_h * kernel_w;
                    let flat_input_dim = in_channels * in_h * in_w;

                    // 1. Bias accumulate
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

                    // 2. Reshape: (S, OC*spatial) -> (S*spatial, OC)
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

                    // 3. GEMM: (total_spatial, OC) x (OC, kernel_cols)
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

                    // 4. Col2im: (S*spatial, kernel_cols) -> (S, IC*IH*IW)
                    let mut new_l = vec![0.0f32; num_specs * flat_input_dim];
                    let mut new_u = vec![0.0f32; num_specs * flat_input_dim];
                    for s in 0..num_specs {
                        for ic in 0..*in_channels {
                            for ih in 0..*in_h {
                                for iw_pos in 0..*in_w {
                                    let flat_idx = ic * in_h * in_w + ih * in_w + iw_pos;
                                    let mut sum_l = 0.0f32;
                                    let mut sum_u = 0.0f32;
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

    #[allow(clippy::too_many_arguments)] // test helper, clarity > API design
    fn cpu_activation_backward(
        a_l: &mut Vec<f32>,
        a_u: &mut Vec<f32>,
        b_l: &mut [f32],
        b_u: &mut [f32],
        ls: &[f32],
        us: &[f32],
        li: &[f32],
        ui: &[f32],
        num_specs: usize,
        n: usize,
    ) {
        let mut new_l = vec![0.0f32; num_specs * n];
        let mut new_u = vec![0.0f32; num_specs * n];
        for s in 0..num_specs {
            let (mut lb, mut ub) = (0.0f32, 0.0f32);
            for j in 0..n {
                let idx = s * n + j;
                let (al, au) = (a_l[idx], a_u[idx]);
                if al >= 0.0 {
                    new_l[idx] = al * ls[j];
                    lb += al * li[j];
                } else {
                    new_l[idx] = al * us[j];
                    lb += al * ui[j];
                }
                if au >= 0.0 {
                    new_u[idx] = au * us[j];
                    ub += au * ui[j];
                } else {
                    new_u[idx] = au * ls[j];
                    ub += au * li[j];
                }
            }
            b_l[s] += lb;
            b_u[s] += ub;
        }
        *a_l = new_l;
        *a_u = new_u;
    }

    use dual_alpha::cpu_dual_alpha_activation_backward;

    #[allow(clippy::too_many_arguments)] // test helper
    fn cpu_linear_backward(
        a_l: &mut Vec<f32>,
        a_u: &mut Vec<f32>,
        b_l: &mut [f32],
        b_u: &mut [f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        num_specs: usize,
        out_f: usize,
        in_f: usize,
    ) {
        if let Some(layer_bias) = bias {
            for s in 0..num_specs {
                let (mut lb, mut ub) = (0.0f32, 0.0f32);
                for j in 0..out_f {
                    lb += a_l[s * out_f + j] * layer_bias[j];
                    ub += a_u[s * out_f + j] * layer_bias[j];
                }
                b_l[s] += lb;
                b_u[s] += ub;
            }
        }
        let mut new_l = vec![0.0f32; num_specs * in_f];
        let mut new_u = vec![0.0f32; num_specs * in_f];
        for s in 0..num_specs {
            for c in 0..in_f {
                let (mut sl, mut su) = (0.0f32, 0.0f32);
                for k in 0..out_f {
                    sl += a_l[s * out_f + k] * weight[k * in_f + c];
                    su += a_u[s * out_f + k] * weight[k * in_f + c];
                }
                new_l[s * in_f + c] = sl;
                new_u[s * in_f + c] = su;
            }
        }
        *a_l = new_l;
        *a_u = new_u;
    }

    fn cpu_concretize(
        a_l: &[f32],
        a_u: &[f32],
        b_l: &[f32],
        b_u: &[f32],
        x_l: &[f32],
        x_u: &[f32],
        num_specs: usize,
        dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut lower = vec![0.0f32; num_specs];
        let mut upper = vec![0.0f32; num_specs];
        for s in 0..num_specs {
            let (mut lb, mut ub) = (b_l[s], b_u[s]);
            for j in 0..dim {
                let (al, au) = (a_l[s * dim + j], a_u[s * dim + j]);
                lb += al.max(0.0) * x_l[j] + al.min(0.0) * x_u[j];
                ub += au.max(0.0) * x_u[j] + au.min(0.0) * x_l[j];
            }
            lower[s] = lb;
            upper[s] = ub;
        }
        (lower, upper)
    }
}
