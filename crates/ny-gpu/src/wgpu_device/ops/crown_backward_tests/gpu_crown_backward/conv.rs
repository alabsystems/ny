// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN backward tests for Conv2d networks.

use super::*;

pub(super) struct SmallConvLinearCase {
    pub(super) layers: Vec<GpuCrownLayer>,
    pub(super) input_lower: Vec<f32>,
    pub(super) input_upper: Vec<f32>,
    pub(super) out_dim: usize,
}

/// Build GpuCrownLayer list for Conv2d -> ReLU -> Linear (backward order).
///
/// Conv2d: (IC, IH, IW) -> (OC, OH, OW), then flatten + Linear -> out_dim.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_conv_layers(
    weight_col: Vec<f32>,
    conv_bias: Vec<f32>,
    linear_weight: Vec<f32>,
    linear_bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    in_h: usize,
    in_w: usize,
    out_dim: usize,
    inp_l: &[f32],
    inp_u: &[f32],
) -> Vec<GpuCrownLayer> {
    let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    let spatial = out_h * out_w;
    let conv_flat = out_channels * spatial;

    // Expand per-channel bias to (OC * spatial)
    let bias_expanded: Vec<f32> = (0..out_channels)
        .flat_map(|oc| std::iter::repeat_n(conv_bias[oc], spatial))
        .collect();

    // IBP forward through conv to get pre-activation bounds for ReLU.
    let pre_bounds = ibp_forward_conv(
        &weight_col,
        &conv_bias,
        inp_l,
        inp_u,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        in_h,
        in_w,
        out_h,
        out_w,
    );
    let (relu_ls, relu_us, relu_li, relu_ui) = relu_slopes(&pre_bounds.0, &pre_bounds.1);

    vec![
        GpuCrownLayer::Linear {
            weight: linear_weight.into(),
            bias: Some(linear_bias.into()),
            out_features: out_dim,
            in_features: conv_flat,
            cert_err: Default::default(),
        },
        GpuCrownLayer::Activation {
            lower_slope: relu_ls,
            upper_slope: relu_us,
            lower_intercept: relu_li,
            upper_intercept: relu_ui,
            num_neurons: conv_flat,
        },
        GpuCrownLayer::Conv2d {
            weight_col: weight_col.into(),
            bias_expanded: Some(bias_expanded.into()),
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
            cert_err: Default::default(),
        },
    ]
}

pub(super) fn build_small_conv_linear_case() -> SmallConvLinearCase {
    let in_channels = 1;
    let out_channels = 2;
    let (kernel_h, kernel_w) = (2, 2);
    let (stride_h, stride_w) = (1, 1);
    let (pad_h, pad_w) = (0, 0);
    let (in_h, in_w) = (3, 3);
    let out_dim = 2;

    let weight_col = vec![0.3f32, -0.2, 0.1, 0.5, -0.1, 0.4, -0.3, 0.2];
    let conv_bias = vec![0.1f32, -0.1];

    let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    let conv_flat = out_channels * out_h * out_w;
    let linear_weight: Vec<f32> = (0..out_dim * conv_flat)
        .map(|i| 0.1 * (i as f32 - 4.0))
        .collect();
    let linear_bias = vec![0.05f32, -0.05];

    let in_flat = in_channels * in_h * in_w;
    let input_lower = vec![-1.0f32; in_flat];
    let input_upper = vec![1.0f32; in_flat];
    let layers = build_conv_layers(
        weight_col,
        conv_bias,
        linear_weight,
        linear_bias,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        in_h,
        in_w,
        out_dim,
        &input_lower,
        &input_upper,
    );
    SmallConvLinearCase {
        layers,
        input_lower,
        input_upper,
        out_dim,
    }
}

/// IBP forward pass through Conv2d for computing pre-activation bounds.
#[allow(clippy::too_many_arguments)]
fn ibp_forward_conv(
    weight_col: &[f32],
    bias: &[f32],
    inp_l: &[f32],
    inp_u: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let spatial = out_h * out_w;
    let kernel_cols = in_channels * kernel_h * kernel_w;
    let flat_out = out_channels * spatial;
    let mut pre_l = vec![0.0f32; flat_out];
    let mut pre_u = vec![0.0f32; flat_out];

    for oc in 0..out_channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let out_idx = oc * spatial + oh * out_w + ow;
                let (mut lb, mut ub) = (bias[oc], bias[oc]);
                for ic in 0..in_channels {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = oh * stride_h + kh;
                            let iw_pos = ow * stride_w + kw;
                            let ih_actual = ih as isize - pad_h as isize;
                            let iw_actual = iw_pos as isize - pad_w as isize;
                            if ih_actual < 0
                                || ih_actual >= in_h as isize
                                || iw_actual < 0
                                || iw_actual >= in_w as isize
                            {
                                continue;
                            }
                            let in_idx =
                                ic * in_h * in_w + ih_actual as usize * in_w + iw_actual as usize;
                            let w_idx =
                                oc * kernel_cols + ic * kernel_h * kernel_w + kh * kernel_w + kw;
                            let w = weight_col[w_idx];
                            if w >= 0.0 {
                                lb += w * inp_l[in_idx];
                                ub += w * inp_u[in_idx];
                            } else {
                                lb += w * inp_u[in_idx];
                                ub += w * inp_l[in_idx];
                            }
                        }
                    }
                }
                pre_l[out_idx] = lb;
                pre_u[out_idx] = ub;
            }
        }
    }
    (pre_l, pre_u)
}

// ---- Deterministic Tests ----

#[test]
fn test_crown_backward_gpu_conv2d_deterministic() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    let case = build_small_conv_linear_case();
    assert_gpu_matches_cpu(
        &device,
        &case.layers,
        case.out_dim,
        &case.input_lower,
        &case.input_upper,
        1e-3,
    );
}

#[test]
fn test_crown_backward_gpu_conv2d_stride_pad() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Conv2d(1->1, 3x3, stride=2, pad=1) on 5x5 input -> 3x3 output
    let in_channels = 1;
    let out_channels = 1;
    let (kernel_h, kernel_w) = (3, 3);
    let (stride_h, stride_w) = (2, 2);
    let (pad_h, pad_w) = (1, 1);
    let (in_h, in_w) = (5, 5);
    let out_dim = 1;

    let weight_col = vec![0.1f32, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.1, 0.5];
    let conv_bias = vec![0.05f32];

    let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    let conv_flat = out_channels * out_h * out_w;

    let linear_weight: Vec<f32> = (0..out_dim * conv_flat)
        .map(|i| 0.1 * (i as f32 - 4.0))
        .collect();
    let linear_bias = vec![0.0f32];

    let in_flat = in_channels * in_h * in_w;
    let inp_l = vec![-0.5f32; in_flat];
    let inp_u = vec![0.5f32; in_flat];

    let layers = build_conv_layers(
        weight_col,
        conv_bias,
        linear_weight,
        linear_bias,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        in_h,
        in_w,
        out_dim,
        &inp_l,
        &inp_u,
    );
    assert_gpu_matches_cpu(&device, &layers, out_dim, &inp_l, &inp_u, 1e-3);
}

// ---- Proptests ----

// Random Conv2d->ReLU->Linear networks, GPU matches CPU reference.
// Expanded per Prover P1:1125/P1:1126: vary stride {1,2}, pad {0,1}, IC {1,2}.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(50) })]
    #[test]
    fn proptest_crown_backward_gpu_conv2d_vs_cpu(
        in_channels in 1usize..=2,
        out_channels in 1usize..=3,
        stride in 1usize..=2,
        pad in 0usize..=1,
        out_dim in 1usize..=2,
        // Seed arrays sized for max dimensions:
        // max weight_col = OC(3) * IC(2) * KH(2) * KW(2) = 24
        w_seed in proptest::collection::vec(-1.0f32..1.0, 128),
        b_seed in proptest::collection::vec(-0.5f32..0.5, 8),
        // max linear_weight = out_dim(2) * OC(3) * OH(6) * OW(6) = 216
        lw_seed in proptest::collection::vec(-1.0f32..1.0, 256),
        lb_seed in proptest::collection::vec(-0.5f32..0.5, 4),
        // max in_flat = IC(2) * IH(5) * IW(5) = 50
        inp_center in proptest::collection::vec(-1.0f32..1.0, 64),
        inp_radius in proptest::collection::vec(0.05f32..0.5, 64),
    ) {
        let _gpu_serial = gpu_test_serial_guard();
        let device = require_device();

        // Fixed kernel 2x2, input 5x5. All stride/pad combos yield valid output:
        // s=1,p=0: OH=4  s=1,p=1: OH=6  s=2,p=0: OH=2  s=2,p=1: OH=3
        let (kernel_h, kernel_w) = (2, 2);
        let (stride_h, stride_w) = (stride, stride);
        let (pad_h, pad_w) = (pad, pad);
        let (in_h, in_w) = (5, 5);
        let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
        let kernel_cols = in_channels * kernel_h * kernel_w;
        let spatial = out_h * out_w;
        let conv_flat = out_channels * spatial;
        let in_flat = in_channels * in_h * in_w;

        let weight_col: Vec<f32> = (0..out_channels * kernel_cols)
            .map(|i| w_seed[i % w_seed.len()])
            .collect();
        let conv_bias: Vec<f32> = (0..out_channels)
            .map(|i| b_seed[i % b_seed.len()])
            .collect();
        let linear_weight: Vec<f32> = (0..out_dim * conv_flat)
            .map(|i| lw_seed[i % lw_seed.len()])
            .collect();
        let linear_bias: Vec<f32> = (0..out_dim)
            .map(|i| lb_seed[i % lb_seed.len()])
            .collect();
        let inp_l: Vec<f32> = (0..in_flat)
            .map(|i| inp_center[i % inp_center.len()] - inp_radius[i % inp_radius.len()])
            .collect();
        let inp_u: Vec<f32> = (0..in_flat)
            .map(|i| inp_center[i % inp_center.len()] + inp_radius[i % inp_radius.len()])
            .collect();

        let layers = build_conv_layers(
            weight_col,
            conv_bias,
            linear_weight,
            linear_bias,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            in_h,
            in_w,
            out_dim,
            &inp_l,
            &inp_u,
        );
        let spec = identity_spec(out_dim);

        let gpu = device.crown_backward_gpu(&layers, &spec, out_dim, &inp_l, &inp_u)
            .map_err(|e| TestCaseError::fail(format!(
                "GPU Conv2d (IC={in_channels},s={stride},p={pad}) error: {e}"
            )))?;
        let (cpu_l, cpu_u) = cpu_crown_backward(&layers, &spec, out_dim, &inp_l, &inp_u);

        let eps = 1e-2;
        for i in 0..out_dim {
            let dl = (gpu.lower_bounds[i] - cpu_l[i]).abs();
            let du = (gpu.upper_bounds[i] - cpu_u[i]).abs();
            prop_assert!(dl <= eps,
                "Conv2d (IC={in_channels},s={stride},p={pad}) lower[{i}] GPU={} CPU={} diff={dl}",
                gpu.lower_bounds[i], cpu_l[i]);
            prop_assert!(du <= eps,
                "Conv2d (IC={in_channels},s={stride},p={pad}) upper[{i}] GPU={} CPU={} diff={du}",
                gpu.upper_bounds[i], cpu_u[i]);
            prop_assert!(gpu.lower_bounds[i] <= gpu.upper_bounds[i] + eps,
                "Conv2d spec {i}: lower {} > upper {}", gpu.lower_bounds[i], gpu.upper_bounds[i]);
        }
    }
}
