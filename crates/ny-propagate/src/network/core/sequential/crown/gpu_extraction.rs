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

/// Cached static GPU CROWN layer data.
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

/// Cached GPU CROWN layer extraction.
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
                // Return None defensively instead of panicking.
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
            // ONE copy per call instead of two (#gpu-descriptor-single-copy).
            // An allocation cut, NOT weight sharing — see WHAT REMAINS below.
            //
            // `Arc::<[f32]>::from(&[f32])` makes one ref-counted allocation and
            // fills it from the slice in a single pass (`ArcFromSlice` in
            // `alloc::sync`; for a trivially-cloneable element such as `f32`
            // that pass specializes to a `copy_nonoverlapping`). The
            // `.to_vec().into()` it replaces copied the weight TWICE: `to_vec`
            // allocated and filled a `Vec`, then std's
            // `impl From<Vec<T>> for Arc<[T]>` allocated the `Arc`,
            // `copy_nonoverlapping`d the whole `Vec` into it and freed the
            // `Vec`'s buffer. Same bytes, same order, same length, so no
            // descriptor float changes; what goes away is one full weight-sized
            // copy plus the `Vec`'s allocate/free pair.
            //
            // WHAT REMAINS: the descriptor still gets a PRIVATE copy of the
            // weight, rebuilt on every call of this function — the `Arc` is
            // born with refcount 1 and shares nothing with the layer or with any
            // earlier descriptor. Whether that copy is paid once or once per
            // backward pass is the CALLER's property, not this function's: the
            // sequential path memoizes the descriptor list in
            // `GpuCrownStaticCache` above (cloning it copies the `Arc` handles,
            // not the weights) and the graph resnet path has the
            // `ResnetSegmentSkeleton` twin, while callers such as
            // `network/ibp/crown_partial_gpu.rs` and
            // `network/graph_alpha/backward/gpu_suffix.rs` build a fresh
            // descriptor list — hence a fresh copy of every weight — on each of
            // their own invocations.
            //
            // Removing the copy itself needs an edit OUTSIDE this file: an
            // `Arc<[f32]>` handle over the row-major parameters held ON the
            // immutable layer, which extraction would clone instead of copy —
            // i.e. a new field on `LinearLayer` (`layers/linear/mod.rs`, over
            // `weight` and `bias`), on `Conv2dLayer`
            // (`layers/convolution/conv2d/types.rs`) and on `Conv1dLayer`
            // (`layers/convolution/conv1d/types/forward.rs`, both over
            // `kernel`). `LinearLayer` already caches derived arrays exactly
            // that way (`weight_t_rm`, `weight_f64`, both `Arc<OnceLock<..>>`,
            // resting on its documented invariant that `weight` is immutable
            // after construction); the two conv types carry no cache field at
            // all today. Not attempted here — those files are outside this
            // change.
            gpu_layers.push(GpuCrownLayer::Linear {
                weight: Arc::<[f32]>::from(linear.weight.as_slice()?),
                bias: linear
                    .bias
                    .as_ref()
                    .and_then(|b| Some(Arc::<[f32]>::from(b.as_slice()?))),
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
                // Return None defensively instead of panicking.
                _ => return None,
            };
            gpu_layers.push(gpu_layer);
        }
        Layer::Conv2d(c) => {
            // Soundness gate (depthwise-separable CROWN crash): the
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
            // One copy instead of two, and the per-call copy of the kernel
            // remains; see the `Layer::Linear` arm.
            let weight_col: Arc<[f32]> = Arc::from(kernel_slice);

            // Expand per-channel bias to full spatial size (out_c * oh * ow).
            // GPU bias_accumulate shader expects element-wise bias matching
            // the A-matrix column dimension.
            //
            // Collected STRAIGHT into the `Arc<[f32]>` instead of through an
            // intermediate `Vec` (#gpu-descriptor-single-copy). The source is a
            // `Map` over a `Range<usize>`, hence `TrustedLen`, and std's
            // `impl FromIterator<T> for Arc<[T]>` documents a single allocation
            // for an exact-size `TrustedLen` iterator (`ToArcSlice` →
            // `Arc::from_iter_exact` in `alloc::sync`, which writes the j-th
            // yielded item at index j).
            // The `Vec` this replaces was filled and then copied wholesale into
            // the `Arc`, so one full `out_c * oh * ow` copy and one
            // allocate/free pair go away. That specialization is a std internal
            // rather than a stability guarantee, but nothing here depends on it:
            // its fallback is `collect::<Vec<_>>().into()`, which is exactly the
            // code this replaced.
            //
            // The expansion is still recomputed on every call, and unlike the
            // kernel it cannot be hoisted onto the layer behind a plain handle:
            // its length follows the OUTPUT geometry derived from the caller's
            // `pre_activation` (`oh * ow`), so a layer-side cache for it would
            // have to be keyed by that geometry.
            //
            // Element `i` still holds `b[i / spatial]`, exactly the
            // channel-major order the nested loop emitted; the running
            // `channel_end` boundary advances the channel without paying an
            // integer division per element.
            let bias_expanded: Option<Arc<[f32]>> = c.bias.as_ref().map(|b| {
                let spatial = oh * ow;
                let mut channel = 0usize;
                let mut channel_end = spatial;
                (0..out_c * spatial)
                    .map(|i| {
                        if i == channel_end {
                            channel += 1;
                            channel_end += spatial;
                        }
                        b[channel]
                    })
                    .collect()
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
            // One copy instead of two, and the per-call copy of the kernel
            // remains; see the `Layer::Linear` arm.
            let weight_col: Arc<[f32]> = Arc::from(kernel_slice);

            // Same single-pass channel-major expansion as the Conv2d arm — same
            // `Arc` collect, same per-call rebuild — with `out_len` as the
            // per-channel run: element `i` holds `bias[i / out_len]`.
            let bias_expanded: Option<Arc<[f32]>> = c.bias.as_ref().map(|bias| {
                let mut channel = 0usize;
                let mut channel_end = out_len;
                (0..out_c * out_len)
                    .map(|i| {
                        if i == channel_end {
                            channel += 1;
                            channel_end += out_len;
                        }
                        bias[channel]
                    })
                    .collect()
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

/// Value pins for the descriptor buffers this module builds
/// (#gpu-descriptor-single-copy).
///
/// They sit inline rather than in `gpu_extraction_tests.rs` so the module that
/// owns the buffer arithmetic owns its pins.
///
/// WHAT THEY DO AND DO NOT COVER. They pin buffer CONTENT, which is what the
/// bias-expansion rewrite can get wrong (a run length taken from the wrong
/// dimension, a channel boundary off by one, a channel that never advances).
/// They cannot distinguish `Arc::from(slice)` from the `to_vec().into()` it
/// replaced: that swap is byte-identical by construction, so no value assertion
/// can see it, and only an allocation-count or pointer assertion could — a claim
/// about allocation strategy, not about behaviour. Read them as a pin on the
/// contract the rewrite had to preserve, not as evidence for the rewrite.
///
/// The Conv2d arm's bias expansion had no VALUE pin before this one. The two
/// `bias_expanded` value assertions in this crate —
/// `gpu_extraction_tests.rs::assert_conv1d_height_one_conv2d_descriptor` and
/// `crown_tests.rs::try_extract_single_gpu_layer_conv1d_maps_to_conv2d_descriptor`
/// — both drive the CONV1D arm, whose per-channel run is `out_len`; neither
/// reaches the Conv2d arm, whose run is `oh * ow`. The other in-crate readers of
/// `bias_expanded` — `graph/propagation/batched/tests_skeleton.rs` (fold-vs-legacy
/// fingerprints) and `margin_row/gpu_seam/batch/tests.rs` (`Arc::ptr_eq`) —
/// compare two extraction paths or allocation identities rather than pinning
/// values, so a consistently-wrong expansion passes them.
#[cfg(test)]
mod descriptor_buffer_tests {
    use super::*;
    use crate::layers::{Conv2dLayer, LinearLayer};
    use ndarray::{arr1, arr2, IxDyn};
    use ny_core::Result;

    fn uniform_bounds(shape: &[usize]) -> BoundedTensor {
        let len: usize = shape.iter().product();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), vec![-1.0f32; len])
                .expect("lower shape should be valid"),
            ArrayD::from_shape_vec(IxDyn(shape), vec![1.0f32; len])
                .expect("upper shape should be valid"),
        )
        .expect("valid bounds")
    }

    /// The Linear descriptor carries the weight row-major and the bias intact.
    ///
    /// Fails on a copy that truncates, transposes or reorders the buffer, and on
    /// a bias arm that drops the bias. The Linear descriptor pins in the sibling
    /// `gpu_extraction_tests.rs` are all 1x1 weights (`[1.0]`, `[2.0]`, `[3.0]`),
    /// in which a transposition is invisible; this one is 2x3 with distinct
    /// entries. It does NOT witness the `Arc::from(slice)` swap itself, which is
    /// byte-identical to what it replaced — see the module comment.
    #[test]
    fn linear_descriptor_carries_row_major_weight_and_bias() -> Result<()> {
        let linear = LinearLayer::new(
            arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            Some(arr1(&[-0.25f32, 0.75])),
        )?;
        let pre_activation = uniform_bounds(&[3]);
        let mut gpu_layers = Vec::new();

        assert_eq!(
            try_extract_single_gpu_layer(&Layer::Linear(linear), &pre_activation, &mut gpu_layers),
            Some(())
        );
        assert_eq!(gpu_layers.len(), 1);
        let GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            ..
        } = &gpu_layers[0]
        else {
            panic!("expected a Linear GPU descriptor");
        };
        assert_eq!((*out_features, *in_features), (2, 3));
        assert_eq!(weight.as_ref(), &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(bias.as_deref(), Some(&[-0.25f32, 0.75][..]));
        Ok(())
    }

    /// The Conv2d descriptor expands the bias channel-major with a run of
    /// `oh * ow`, and carries the kernel verbatim as `weight_col`.
    ///
    /// Fails closed on the expansion arithmetic: the geometry is 3 output
    /// channels over `oh = 2, ow = 3`, so a run length taken from the wrong
    /// dimension (`out_c = 3`, `kh * kw = 4`, `in_h * in_w = 12` all differ from
    /// 6), a channel boundary off by one, or a channel that fails to advance
    /// each produce a different 18-element vector. It is deliberately blind to
    /// which iteration form produced that vector: the nested-push loop and the
    /// `(0..n).map(..).collect()` that replaced it are required to agree here.
    #[test]
    fn conv2d_descriptor_expands_bias_channel_major_over_output_spatial() -> Result<()> {
        // kernel (out_c = 3, in_c = 1, kh = 2, kw = 2)
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[3, 1, 2, 2]),
            vec![
                1.0f32, 2.0, 3.0, 4.0, // out channel 0
                5.0, 6.0, 7.0, 8.0, // out channel 1
                9.0, 10.0, 11.0, 12.0, // out channel 2
            ],
        )
        .expect("conv2d kernel shape should be valid");
        let conv = Conv2dLayer::new(kernel, Some(arr1(&[0.5f32, -1.5, 2.25])), (1, 1), (0, 0))?;
        // A 1x3x4 input under a 2x2 stride-1 pad-0 kernel gives oh = 2, ow = 3,
        // so the per-channel run is 6 and the whole plane is 18 elements.
        let pre_activation = uniform_bounds(&[1, 3, 4]);
        let mut gpu_layers = Vec::new();

        assert_eq!(
            try_extract_single_gpu_layer(&Layer::Conv2d(conv), &pre_activation, &mut gpu_layers),
            Some(())
        );
        assert_eq!(gpu_layers.len(), 1);
        let GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            out_h,
            out_w,
            ..
        } = &gpu_layers[0]
        else {
            panic!("expected a Conv2d GPU descriptor");
        };
        assert_eq!((*out_channels, *out_h, *out_w), (3, 2, 3));
        assert_eq!(
            weight_col.as_ref(),
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        let expected: Vec<f32> = [0.5f32, -1.5, 2.25]
            .into_iter()
            .flat_map(|b| std::iter::repeat_n(b, 6))
            .collect();
        assert_eq!(
            bias_expanded.as_deref().expect("Conv2d bias should expand"),
            expected.as_slice()
        );
        Ok(())
    }

    /// A Conv2d without a bias still produces `bias_expanded: None` — the
    /// `Option` arm must survive the collect rewrite rather than becoming an
    /// all-zero plane.
    #[test]
    fn conv2d_descriptor_without_bias_expands_nothing() -> Result<()> {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2, 2]), vec![1.0f32; 8])
            .expect("conv2d kernel shape should be valid");
        let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0))?;
        let pre_activation = uniform_bounds(&[1, 3, 4]);
        let mut gpu_layers = Vec::new();

        assert_eq!(
            try_extract_single_gpu_layer(&Layer::Conv2d(conv), &pre_activation, &mut gpu_layers),
            Some(())
        );
        let GpuCrownLayer::Conv2d { bias_expanded, .. } = &gpu_layers[0] else {
            panic!("expected a Conv2d GPU descriptor");
        };
        assert!(
            bias_expanded.is_none(),
            "biasless Conv2d must expand no bias"
        );
        Ok(())
    }
}
