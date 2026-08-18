// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN layer extraction helpers.

use crate::layers::activations::exp::exp_linear_relaxation;
use crate::layers::activations::log::log_linear_relaxation;
use crate::layers::activations::relu::relu_linear_relaxation;
use crate::layers::activations::LinearRelaxation;
use crate::layers::trigonometric::{sigmoid_linear_relaxation, tanh_linear_relaxation};
use crate::layers::Layer;
use ndarray::ArrayD;
use ny_core::GpuCrownLayer;
use ny_tensor::BoundedTensor;
use std::sync::{Arc, Mutex};

mod alpha;
mod maxpool;

pub(crate) use alpha::{
    extract_relu_gpu_layer_with_alpha, gpu_relu_affine_cell, GpuReluAffineVariant,
};
use maxpool::extract_maxpool_gpu_layer;

/// Build a `GpuCrownLayer::Activation` from per-neuron linear relaxation bounds.
///
/// Used by all elementwise activation layers (ReLU, Sigmoid, Tanh, Exp, Log, etc.)
/// in `try_extract_gpu_crown_layers`. The caller provides a closure that computes
/// the `LinearRelaxation` for a single neuron given its pre-activation bounds [l, u].
fn extract_activation_gpu_layer(
    pre_l: &[f32],
    pre_u: &[f32],
    relax: impl Fn(f32, f32) -> LinearRelaxation,
) -> GpuCrownLayer {
    let num_neurons = pre_l.len();
    let mut lower_slope = Vec::with_capacity(num_neurons);
    let mut upper_slope = Vec::with_capacity(num_neurons);
    let mut lower_intercept = Vec::with_capacity(num_neurons);
    let mut upper_intercept = Vec::with_capacity(num_neurons);

    for (&l, &u) in pre_l.iter().zip(pre_u.iter()) {
        let r = relax(l, u);
        lower_slope.push(r.lower_slope);
        upper_slope.push(r.upper_slope);
        lower_intercept.push(r.lower_intercept);
        upper_intercept.push(r.upper_intercept);
    }

    GpuCrownLayer::Activation {
        lower_slope,
        upper_slope,
        lower_intercept,
        upper_intercept,
        num_neurons,
    }
}

/// Flatten an ndarray constant to a Vec<f32>, broadcasting scalars to `dim`.
///
/// Returns `Some(vec)` if the constant has exactly `dim` elements or is a
/// scalar (broadcast to `dim`). Returns `None` otherwise.
fn expand_constant(constant: &ArrayD<f32>, dim: usize) -> Option<Vec<f32>> {
    let flat: Vec<f32> = constant.iter().copied().collect();
    if flat.len() == dim {
        Some(flat)
    } else if flat.len() == 1 {
        Some(vec![flat[0]; dim])
    } else {
        None
    }
}

/// Return type for full GPU layer extraction with dynamic tracking.
type GpuExtractionResult = (Vec<GpuCrownLayer>, Vec<(usize, usize)>);

/// Cached static GPU CROWN layer data (#3397 plan cache Step 1).
///
/// Separates static model data (weights, topology) from dynamic activation
/// relaxations so BaB iterations only rebuild the nonlinear activation parts.
/// Linear/Conv2d weights use `Arc<[f32]>` so cloning the cached list is O(1)
/// per static layer instead of O(weight_elements).
pub(crate) struct GpuCrownStaticCache {
    /// Full layer list from first extraction (backward order).
    /// Static entries (Linear/Conv2d/constant-arithmetic) are shared via Arc.
    layers: Vec<GpuCrownLayer>,
    /// Indices of dynamic activation entries: (gpu_layer_index, network_layer_index).
    dynamic_entries: Vec<(usize, usize)>,
}

/// Cached GPU CROWN layer extraction (#3397 plan cache Step 1).
///
/// On first call, performs full extraction and populates the cache.
/// On subsequent calls, clones the cached layer list (O(1) per Arc weight)
/// and only recomputes nonlinear activation relaxations from current bounds.
pub(super) fn extract_gpu_crown_layers_cached(
    network_layers: &[Layer],
    layer_bounds: &[BoundedTensor],
    input: &BoundedTensor,
    cache: &Mutex<Option<GpuCrownStaticCache>>,
) -> Option<Vec<GpuCrownLayer>> {
    let guard = cache.lock().ok()?;

    if let Some(cached) = guard.as_ref() {
        // Cache hit: clone the cached layers (cheap for Arc-backed static entries),
        // then refresh dynamic entries from current pre-activation bounds.
        let mut gpu_layers = cached.layers.clone();
        for &(gpu_idx, net_idx) in &cached.dynamic_entries {
            let pre = if net_idx == 0 {
                input
            } else {
                &layer_bounds[net_idx - 1]
            };
            let new_layer = match &network_layers[net_idx] {
                Layer::ReLU(_) => {
                    let pre_l = pre.lower().as_slice()?;
                    let pre_u = pre.upper().as_slice()?;
                    extract_activation_gpu_layer(pre_l, pre_u, relu_linear_relaxation)
                }
                Layer::Sigmoid(_) => {
                    let pre_l = pre.lower().as_slice()?;
                    let pre_u = pre.upper().as_slice()?;
                    extract_activation_gpu_layer(pre_l, pre_u, sigmoid_linear_relaxation)
                }
                Layer::Tanh(_) => {
                    let pre_l = pre.lower().as_slice()?;
                    let pre_u = pre.upper().as_slice()?;
                    extract_activation_gpu_layer(pre_l, pre_u, tanh_linear_relaxation)
                }
                Layer::Exp(_) => {
                    let pre_l = pre.lower().as_slice()?;
                    let pre_u = pre.upper().as_slice()?;
                    extract_activation_gpu_layer(pre_l, pre_u, exp_linear_relaxation)
                }
                Layer::Log(_) => {
                    let pre_l = pre.lower().as_slice()?;
                    let pre_u = pre.upper().as_slice()?;
                    extract_activation_gpu_layer(pre_l, pre_u, log_linear_relaxation)
                }
                Layer::MaxPool2d(pool) => extract_maxpool_gpu_layer(pool, pre)?,
                // Structurally unreachable: dynamic_entries only holds activation/pool indices.
                // Return None defensively instead of panicking (#4205).
                _ => return None,
            };
            gpu_layers[gpu_idx] = new_layer;
        }
        return Some(gpu_layers);
    }

    // Cache miss: full extraction, then populate cache.
    drop(guard);
    let (gpu_layers, dynamic_entries) =
        try_extract_gpu_crown_layers_with_dynamic_tracking(network_layers, layer_bounds, input)?;

    if let Ok(mut guard) = cache.lock() {
        *guard = Some(GpuCrownStaticCache {
            layers: gpu_layers.clone(),
            dynamic_entries,
        });
    }

    Some(gpu_layers)
}

/// Full extraction that also tracks which entries are dynamic activations.
fn try_extract_gpu_crown_layers_with_dynamic_tracking(
    layers: &[Layer],
    layer_bounds: &[BoundedTensor],
    input: &BoundedTensor,
) -> Option<GpuExtractionResult> {
    let mut gpu_layers = Vec::with_capacity(layers.len());
    let mut dynamic_entries = Vec::new();

    for (i, layer) in layers.iter().enumerate().rev() {
        let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };
        let is_dynamic_layer = matches!(
            layer,
            Layer::ReLU(_)
                | Layer::Sigmoid(_)
                | Layer::Tanh(_)
                | Layer::Exp(_)
                | Layer::Log(_)
                | Layer::MaxPool2d(_)
        );

        // Track dynamic entries before pushing (index = current length).
        if is_dynamic_layer {
            dynamic_entries.push((gpu_layers.len(), i));
        }

        try_extract_single_gpu_layer(layer, pre_activation, &mut gpu_layers)?;
    }

    Some((gpu_layers, dynamic_entries))
}

/// Extract a single layer's GPU CROWN descriptor and push it to `gpu_layers`.
///
/// Returns `Some(())` on success, `None` if the layer is unsupported for the GPU
/// fast-path (caller falls back to CPU backward loop).
///
/// Supported layers:
/// - `Linear` → `GpuCrownLayer::Linear` (weight + bias, Arc-backed)
/// - `Conv1d`/`Conv2d` → `GpuCrownLayer::Conv2d` (Conv1d uses height=1 equivalence)
/// - `MaxPool2d` → `GpuCrownLayer::MaxPool2d` (winner routing or IBP fallback)
/// - `ReLU`/`Sigmoid`/`Tanh`/`Exp`/`Log` → `GpuCrownLayer::Activation` (relaxation slopes/intercepts)
/// - `AddConstant`/`SubConstant`/`MulConstant`/`DivConstant` → `GpuCrownLayer::Activation`
/// - `Flatten`/`Reshape` → skipped (no-op for flat A-matrices in Dense mode)
pub(crate) fn try_extract_single_gpu_layer(
    layer: &Layer,
    pre_activation: &BoundedTensor,
    gpu_layers: &mut Vec<GpuCrownLayer>,
) -> Option<()> {
    match layer {
        Layer::Linear(linear) => {
            gpu_layers.push(GpuCrownLayer::Linear {
                weight: linear.weight.as_slice()?.to_vec().into(),
                bias: linear
                    .bias
                    .as_ref()
                    .and_then(|b| Some(b.as_slice()?.to_vec().into())),
                out_features: linear.out_features(),
                in_features: linear.in_features(),
                cert_err: Default::default(),
            });
        }
        // Elementwise activations: compute per-neuron linear relaxation
        // (slopes/intercepts) and encode as GpuCrownLayer::Activation.
        // The GPU activation backward shader is already generic — it applies
        // element-wise slopes/intercepts regardless of the activation type.
        // Reference: alpha-beta-CROWN BoundOptimizableActivation.
        Layer::ReLU(_) | Layer::Sigmoid(_) | Layer::Tanh(_) | Layer::Exp(_) | Layer::Log(_) => {
            let pre_l = pre_activation.lower().as_slice()?;
            let pre_u = pre_activation.upper().as_slice()?;
            let gpu_layer = match layer {
                Layer::ReLU(_) => {
                    extract_activation_gpu_layer(pre_l, pre_u, relu_linear_relaxation)
                }
                Layer::Sigmoid(_) => {
                    extract_activation_gpu_layer(pre_l, pre_u, sigmoid_linear_relaxation)
                }
                Layer::Tanh(_) => {
                    extract_activation_gpu_layer(pre_l, pre_u, tanh_linear_relaxation)
                }
                Layer::Exp(_) => extract_activation_gpu_layer(pre_l, pre_u, exp_linear_relaxation),
                Layer::Log(_) => extract_activation_gpu_layer(pre_l, pre_u, log_linear_relaxation),
                // Structurally unreachable: outer match restricts to ReLU/Sigmoid/Tanh/Exp/Log.
                // Return None defensively instead of panicking (#4205).
                _ => return None,
            };
            gpu_layers.push(gpu_layer);
        }
        Layer::Conv2d(c) => {
            // Soundness gate (#4205, depthwise-separable CROWN crash): the
            // `GpuCrownLayer::Conv2d` descriptor and its shader model a DENSE
            // convolution — `weight_col` is interpreted as `(out_c, in_c*kh*kw)`
            // with no `groups`/`dilation` representation. A grouped (e.g. depthwise)
            // kernel is stored compactly as `(out_c, in_c/groups, kh, kw)`, so
            // feeding it through the dense descriptor both panics the kernel-length
            // `debug_assert` below AND, in release, would silently propagate WRONG
            // (unsound) bounds / read out of bounds. Reject grouped or dilated
            // Conv2d here so the caller falls back to the proven-sound CPU backward
            // (`Conv2dLayer::propagate_linear_batched`, which handles groups via
            // `conv2d_transpose_batched_gemm_grouped`). Mirrors the Conv1d guard below.
            if c.groups > 1 || c.dilation != (1, 1) {
                return None;
            }

            let input_shape = pre_activation.shape();
            let (in_h, in_w) = if input_shape.len() >= 3 {
                (
                    input_shape[input_shape.len() - 2],
                    input_shape[input_shape.len() - 1],
                )
            } else {
                return None;
            };

            let out_c = c.out_channels();
            let in_c = c.in_channels();
            let (kh, kw) = c.kernel_size();
            let (sh, sw) = c.stride;
            let (ph, pw) = c.padding;
            let (oh, ow) = c.output_size(in_h, in_w).ok()?;

            // Reshape kernel from (out_c, in_c, kh, kw) to W_col (out_c, in_c*kh*kw)
            // row-major. This is the same layout used by CPU batched GEMM path.
            let kernel_slice = c.kernel.as_slice()?;
            let kernel_cols = in_c * kh * kw;
            debug_assert_eq!(kernel_slice.len(), out_c * kernel_cols);
            let weight_col: Arc<[f32]> = kernel_slice.to_vec().into();

            // Expand per-channel bias to full spatial size (out_c * oh * ow).
            // GPU bias_accumulate shader expects element-wise bias matching
            // the A-matrix column dimension.
            let bias_expanded: Option<Arc<[f32]>> = c.bias.as_ref().map(|b| {
                let spatial = oh * ow;
                let mut expanded = Vec::with_capacity(out_c * spatial);
                for ch in 0..out_c {
                    for _ in 0..spatial {
                        expanded.push(b[ch]);
                    }
                }
                expanded.into()
            });

            gpu_layers.push(GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels: out_c,
                in_channels: in_c,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                pad_h: ph,
                pad_w: pw,
                out_h: oh,
                out_w: ow,
                in_h,
                in_w,
                cert_err: Default::default(),
            });
        }
        Layer::Conv1d(c) => {
            // Reuse the Conv2d GPU path via the Conv1d(IC, OC, K) ≡
            // Conv2d(IC, OC, (1, K)) equivalence on height-1 inputs.
            // Reference: designs/2026-03-13-conv1d-gpu-crown-backward.md
            if c.groups > 1 || c.dilation > 1 {
                return None;
            }

            let input_shape = pre_activation.shape();
            let in_len = if input_shape.len() >= 2 {
                input_shape[input_shape.len() - 1]
            } else {
                c.input_length?
            };

            let out_c = c.out_channels();
            let in_c = c.in_channels();
            let kernel_w = c.kernel_size();
            let out_len = c.output_length(in_len).ok()?;

            let kernel_slice = c.kernel.as_slice()?;
            debug_assert_eq!(kernel_slice.len(), out_c * in_c * kernel_w);
            let weight_col: Arc<[f32]> = kernel_slice.to_vec().into();

            let bias_expanded: Option<Arc<[f32]>> = c.bias.as_ref().map(|bias| {
                let mut expanded = Vec::with_capacity(out_c * out_len);
                for ch in 0..out_c {
                    for _ in 0..out_len {
                        expanded.push(bias[ch]);
                    }
                }
                expanded.into()
            });

            gpu_layers.push(GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels: out_c,
                in_channels: in_c,
                kernel_h: 1,
                kernel_w,
                stride_h: 1,
                stride_w: c.stride,
                pad_h: 0,
                pad_w: c.padding,
                out_h: 1,
                out_w: out_len,
                in_h: 1,
                in_w: in_len,
                cert_err: Default::default(),
            });
        }
        Layer::MaxPool2d(pool) => {
            gpu_layers.push(extract_maxpool_gpu_layer(pool, pre_activation)?);
        }
        // Constant-arithmetic layers expressed as Activation (slopes/intercepts).
        // These reuse the GPU activation shader with appropriate parameters:
        //   AddConstant(c): y = x + c -> slopes=1, intercepts=c
        //   SubConstant(c): y = x - c -> slopes=1, intercepts=-c
        //   SubConstant(c, reverse): y = c - x -> slopes=-1, intercepts=c
        //   MulConstant(c): y = x * c -> slopes=c, intercepts=0
        //   DivConstant(c): y = x / c -> slopes=1/c, intercepts=0
        // Reference: designs/2026-03-06-gpu-crown-backward.md, #3460
        Layer::AddConstant(layer) => {
            let num_neurons = pre_activation.len();
            let intercept = expand_constant(&layer.constant, num_neurons)?;
            gpu_layers.push(GpuCrownLayer::Activation {
                lower_slope: vec![1.0; num_neurons],
                upper_slope: vec![1.0; num_neurons],
                lower_intercept: intercept.clone(),
                upper_intercept: intercept,
                num_neurons,
            });
        }
        Layer::SubConstant(layer) => {
            let num_neurons = pre_activation.len();
            let flat = expand_constant(&layer.constant, num_neurons)?;
            if layer.reverse {
                // y = c - x: negate A, shift bias by +c
                gpu_layers.push(GpuCrownLayer::Activation {
                    lower_slope: vec![-1.0; num_neurons],
                    upper_slope: vec![-1.0; num_neurons],
                    lower_intercept: flat.clone(),
                    upper_intercept: flat,
                    num_neurons,
                });
            } else {
                // y = x - c: identity on A, shift bias by -c
                let neg: Vec<f32> = flat.iter().map(|&v| -v).collect();
                gpu_layers.push(GpuCrownLayer::Activation {
                    lower_slope: vec![1.0; num_neurons],
                    upper_slope: vec![1.0; num_neurons],
                    lower_intercept: neg.clone(),
                    upper_intercept: neg,
                    num_neurons,
                });
            }
        }
        Layer::MulConstant(layer) => {
            let num_neurons = pre_activation.len();
            let slopes = expand_constant(&layer.constant, num_neurons)?;
            gpu_layers.push(GpuCrownLayer::Activation {
                lower_slope: slopes.clone(),
                upper_slope: slopes,
                lower_intercept: vec![0.0; num_neurons],
                upper_intercept: vec![0.0; num_neurons],
                num_neurons,
            });
        }
        Layer::DivConstant(layer) => {
            let num_neurons = pre_activation.len();
            let flat = expand_constant(&layer.constant, num_neurons)?;
            if flat.contains(&0.0) {
                return None;
            }
            let inv: Vec<f32> = flat.iter().map(|&v| 1.0 / v).collect();
            gpu_layers.push(GpuCrownLayer::Activation {
                lower_slope: inv.clone(),
                upper_slope: inv,
                lower_intercept: vec![0.0; num_neurons],
                upper_intercept: vec![0.0; num_neurons],
                num_neurons,
            });
        }
        // Flatten and Reshape are no-ops for flat A-matrices in Dense mode.
        // The GPU backward operates on flat (num_specs × dim) matrices, so
        // dimension rearrangement layers have no effect.
        Layer::Flatten(_) | Layer::Reshape(_) => {}
        _ => return None,
    }

    Some(())
}

#[cfg(test)]
#[path = "gpu_extraction_tests.rs"]
mod tests;

/// #cgan-bn-gpu-extract: BatchNorm (inference affine y = s*x + b) as an EXACT
/// 1x1 diagonal [`GpuCrownLayer::Conv2d`], plus the per-position
/// precompute-error discharge `werr_y` the CALLER MUST fold into the feeding
/// ReLU's intercepts (fail-closed otherwise; see `apply_bn_werr_to_host_relu`).
/// Spatial is flattened to (1 x EPC): for a 1x1 stride-1 pad-0 kernel the HxW
/// factorization is irrelevant and the conv backward reduces to the exact
/// per-position column scale a_y[j] = a_z[j]*s[c(j)] — the same signed
/// substitution as the CPU reference, so a NEGATIVE per-channel scale is
/// handled as a general affine (no monotonicity assumption anywhere).
///
/// SOUNDNESS vs the real BatchNorm: real_z(y) = f32_z(y) + d(y) with
/// |d(y)| <= scale_err[c]*|y| + bias_err[c] <= werr_z[j] over the BN input box.
/// The needed outward widen is |a_z|*werr_z; the host ReLU sees a_y = a_z*s
/// exactly, so +/-werr_y on its intercepts with werr_y = up(werr_z/|s|) gives
/// |a_y|*werr_y >= |a_z|*werr_z. s == 0.0 cannot route the discharge -> refuse.
/// Rounding rigor: werr_z accumulates in f64; `next_up_f32` after the f32 cast
/// is rigorously outward (the f64 deficit is dominated by the f32 ulp step).
pub(crate) fn try_extract_batch_norm_conv1x1(
    bn: &crate::layers::normalization::BatchNormLayer,
    pre_activation: &BoundedTensor,
) -> Option<(GpuCrownLayer, Vec<f32>)> {
    use ny_tensor::next_up_f32;
    let total = pre_activation.len();
    let (c, epc) = bn
        .gpu_extraction_layout(pre_activation.shape(), Some(total))
        .ok()?;
    let scale = bn.scale.as_slice()?;
    let bias = bn.bias.as_slice()?;
    let scale_err = bn.scale_err.as_slice()?;
    let bias_err = bn.bias_err.as_slice()?;
    if scale.len() != c || bias.len() != c || scale_err.len() != c || bias_err.len() != c {
        return None;
    }
    if scale.iter().any(|s| *s == 0.0 || !s.is_finite()) || bias.iter().any(|b| !b.is_finite()) {
        return None;
    }
    // weight_col: (out_c = C, in_c*kh*kw = C) row-major diagonal.
    let mut weight_col = vec![0.0f32; c * c];
    for (o, &s) in scale.iter().enumerate() {
        weight_col[o * c + o] = s;
    }
    // bias_expanded: (C*EPC) channel-major — matches BN's flat layout AND the
    // Conv2d descriptor's (out_c * oh * ow) expectation.
    let mut bias_expanded = Vec::with_capacity(c * epc);
    for &b in bias {
        bias_expanded.extend(std::iter::repeat_n(b, epc));
    }
    let pre_l = pre_activation.lower().as_slice()?;
    let pre_u = pre_activation.upper().as_slice()?;
    if pre_l.len() != c * epc || pre_u.len() != c * epc {
        return None;
    }
    let mut werr_y = Vec::with_capacity(c * epc);
    for j in 0..c * epc {
        let ch = j / epc;
        let ymag = f64::from(pre_l[j].abs().max(pre_u[j].abs()));
        if !ymag.is_finite() {
            return None; // unbounded pre-activation: discharge uncertifiable
        }
        let werr_z = f64::from(scale_err[ch]) * ymag + f64::from(bias_err[ch]);
        let w = next_up_f32((werr_z / f64::from(scale[ch]).abs()) as f32);
        if !w.is_finite() {
            return None;
        }
        werr_y.push(w);
    }
    Some((
        GpuCrownLayer::Conv2d {
            weight_col: weight_col.into(),
            bias_expanded: Some(bias_expanded.into()),
            out_channels: c,
            in_channels: c,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 1,
            out_w: epc,
            in_h: 1,
            in_w: epc,
            cert_err: Default::default(),
        },
        werr_y,
    ))
}

/// Fold a pending BatchNorm precompute-error discharge OUTWARD into the host
/// ReLU's just-extracted plain `Activation` intercepts (li -= w, ui += w).
/// `ActivationReluDualAlpha` (through-origin branches, no intercept slots) and
/// width mismatches refuse — fail-closed.
pub(crate) fn apply_bn_werr_to_host_relu(layer: &mut GpuCrownLayer, werr: &[f32]) -> Option<()> {
    use ny_tensor::{next_down_f32, next_up_f32};
    let GpuCrownLayer::Activation {
        lower_intercept,
        upper_intercept,
        num_neurons,
        ..
    } = layer
    else {
        return None;
    };
    if *num_neurons != werr.len() {
        return None;
    }
    for (li, &w) in lower_intercept.iter_mut().zip(werr) {
        *li = next_down_f32((f64::from(*li) - f64::from(w)) as f32);
    }
    for (ui, &w) in upper_intercept.iter_mut().zip(werr) {
        *ui = next_up_f32((f64::from(*ui) + f64::from(w)) as f32);
    }
    Some(())
}
