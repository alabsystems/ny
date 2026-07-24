// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-aware CROWN backward step dispatch.
//!
//! Extracted from `crown.rs` as Packet 1 of the three-module extraction (#4005).
//! Contains [`crown_backward_step_patches`] and the shared
//! [`try_patches_or_dense_fallback`] helper that deduplicates the
//! BatchNorm/AveragePool/MaxPool2d dense-fallback arms.

use crate::bounds::patches::CrownBounds;
use crate::layers::common::PatchesPropagation;
use crate::layers::Layer;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use std::time::Instant;
use tracing::{debug, info};

use super::{
    crown_backward_step, guard_dense_materialization_budget, CrownStepFallback, CrownStepResult,
};
use crate::types::CrownIbpFallbackReason;

/// Try patches-native backward, fall back to Dense + [`crown_backward_step`] on error.
///
/// Used by BatchNorm, AveragePool, MaxPool2d arms — each supports patches natively
/// but falls back to Dense when propagation fails (shape mismatch, numerical issue).
///
/// `patches_fn` abstracts the method dispatch since BatchNorm uses
/// `propagate_patches(pb)` while pool layers use
/// `propagate_patches_with_bounds(pb, pre_activation)`.
#[allow(clippy::too_many_arguments)] // CROWN backward context: each param is a distinct propagation concern
fn try_patches_or_dense_fallback(
    patches_fn: impl FnOnce(&mut crate::bounds::patches::PatchesLinearBounds) -> Result<CrownBounds>,
    crown_bounds: &mut CrownBounds,
    pre_activation: &BoundedTensor,
    layer: &Layer,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    fallback_site: &'static str,
    dispatch_site: &'static str,
) -> Result<CrownStepResult> {
    if let CrownBounds::Patches(pb) = crown_bounds {
        match patches_fn(pb) {
            Ok(result) => {
                *crown_bounds = result;
                return Ok(CrownStepResult::Continue);
            }
            Err(e) => {
                debug!(
                    "{}: {} Patches backward failed at layer {}: {}, \
                     falling back to Dense",
                    label,
                    layer.layer_type(),
                    layer_idx,
                    e
                );
                if let Some(fallback) = guard_dense_materialization_budget(
                    crown_bounds,
                    fallback_site,
                    layer,
                    layer_idx,
                    label,
                )? {
                    return Ok(fallback);
                }
                let lb = crown_bounds.ensure_dense()?;
                return crown_backward_step(layer, lb, pre_activation, engine, layer_idx, label);
            }
        }
    }
    // Dense mode: use standard Dense dispatch
    if let Some(fallback) =
        guard_dense_materialization_budget(crown_bounds, dispatch_site, layer, layer_idx, label)?
    {
        return Ok(fallback);
    }
    let lb = crown_bounds.ensure_dense()?;
    crown_backward_step(layer, lb, pre_activation, engine, layer_idx, label)
}

/// Patches-aware CROWN backward step for sequential networks.
///
/// Wraps [`CrownBounds`] to dispatch layers through the Patches path when
/// already in Patches mode, falling back to Dense for unsupported layers.
///
/// Patches-aware layers (Phase 1: Conv2d, Phase 2: ReLU/activations) process
/// Patches coefficients natively, preserving the sparse structure. All other
/// layers call `ensure_dense()` first, then delegate to [`crown_backward_step`].
///
/// Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md
/// Part of #2613
pub(crate) fn crown_backward_step_patches(
    layer: &Layer,
    crown_bounds: &mut CrownBounds,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    deadline: Option<Instant>,
) -> Result<CrownStepResult> {
    // SOUNDNESS SAFETY NET (#vnncomp-aw-soundness). The incoming Dense bounds may
    // carry a certified coefficient-error interval (from a linear/conv backward
    // composed earlier in the chain). Layers taught to propagate that error keep
    // it sound:
    //   - `propagates_coeff_err()` (Linear, Conv2d, element-wise activations,
    //     pass-through reshapes) — handled inside their own backward;
    //   - `is_exact_linear_coeff_err_carrier()` (Slice/Transpose/Tile/Gather/Pad/
    //     constant-arithmetic/Conv1d/ConvTranspose…) — carried by the EXACT-linear
    //     transform path in `dispatch_backward_layer`;
    //   - `SkipMerge` — a pure identity pass-through (`Continue`) that leaves the
    //     error matrices untouched.
    // For any OTHER layer, soundly discharge the error by degrading the affected
    // rows to conservative `[-inf, +inf]` before dispatch, so the layer proceeds
    // on plain (error-free) bounds and never silently drops the penalty.
    let carries_err = layer.propagates_coeff_err()
        || layer.is_exact_linear_coeff_err_carrier()
        || matches!(layer, Layer::SkipMerge(_));
    if !carries_err {
        if let CrownBounds::Dense(lb) = crown_bounds {
            if lb.has_coeff_err() {
                debug!(
                    "{}: layer {} ({}) does not propagate certified coefficient error; \
                     discharging affected rows to conservative bounds (#vnncomp-aw-soundness)",
                    label,
                    layer_idx,
                    layer.layer_type()
                );
                lb.discharge_coeff_err_to_conservative();
            }
        }
    }
    // Sparse patches fallback (#2613 Phase 4): structural layers (Conv2d,
    // BatchNorm, pooling, Linear, Flatten, Reshape) don't support sparse
    // patches natively yet. Convert sparse patches to dense before dispatch.
    // Element-wise activations handle sparse patches natively via
    // backward_patches_sparse in crown_patches.rs.
    let is_structural = matches!(
        layer,
        Layer::Conv2d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::BatchNorm(_)
            | Layer::AveragePool(_)
            | Layer::MaxPool2d(_)
            | Layer::Linear(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
    );
    let sparse_structural_to_dense = is_structural
        && matches!(
            crown_bounds,
            CrownBounds::Patches(pb)
                if pb.lower_a.unstable_idx.is_some() || pb.upper_a.unstable_idx.is_some()
        );
    if sparse_structural_to_dense {
        if let Some(fallback) = guard_dense_materialization_budget(
            crown_bounds,
            "sparse_structural_to_dense",
            layer,
            layer_idx,
            label,
        )? {
            return Ok(fallback);
        }
        if let CrownBounds::Patches(pb) = crown_bounds {
            debug!(
                "{}: layer {} ({}) converting sparse patches to dense",
                label,
                layer_idx,
                layer.layer_type()
            );
            let dense = pb.to_dense()?;
            *crown_bounds = CrownBounds::Dense(dense);
        }
    }

    match layer {
        Layer::Conv2d(c) => {
            let input_shape = pre_activation.shape();
            let (in_h, in_w) = if input_shape.len() >= 3 {
                (
                    input_shape[input_shape.len() - 2],
                    input_shape[input_shape.len() - 1],
                )
            } else {
                debug!(
                    "{}: Conv2d input shape too small: {:?}, using IBP",
                    label, input_shape
                );
                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                    reason: CrownIbpFallbackReason::CrownPropagationError,
                    details: format!("Conv2d input shape too small: {input_shape:?}"),
                }));
            };
            let mut conv_with_shape = c.clone();
            conv_with_shape.set_input_shape(in_h, in_w);

            // #3813 / #hotpath: Pre-check — if patches backward would produce a
            // composed kernel whose receptive-field AREA reaches the dense
            // A-matrix element count (the true memory crossover), convert to Dense
            // early. This avoids the expensive O(positions × channels × kernel²)
            // patches composition only when patches would no longer save memory
            // and `should_fallback_to_dense` would convert anyway. Patches stay
            // active for all earlier conv layers where new_kh*new_kw < in_h*in_w,
            // even past the old fixed 75%-per-dimension threshold.
            let bail_patches_to_dense = if let CrownBounds::Patches(pb) = &*crown_bounds {
                let (kh, kw) = c.kernel_size();
                pb.lower_a.would_conv_compose_cover_input(c.stride, (kh, kw), in_h, in_w)
                    || pb.upper_a.would_conv_compose_cover_input(c.stride, (kh, kw), in_h, in_w)
            } else {
                false
            };
            if bail_patches_to_dense {
                info!(
                    "{}: Conv2d Patches->Dense early bail at layer {}: \
                     composed kernel would cover {}x{} input (#3813)",
                    label, layer_idx, in_h, in_w
                );
                if let Some(fallback) = guard_dense_materialization_budget(
                    crown_bounds,
                    "conv2d_patches_early_bail",
                    layer,
                    layer_idx,
                    label,
                )? {
                    return Ok(fallback);
                }
                crown_bounds.ensure_dense()?;
            }

            match crown_bounds {
                CrownBounds::Patches(pb) => {
                    // Patches Conv2d backward. On UnsupportedConfiguration (dilation,
                    // or the soundness guard against composing through padded incoming
                    // patches — #hotpath), fall back to the exact dense CROWN path
                    // rather than propagating the error.
                    match conv_with_shape.propagate_patches_engine(pb, engine) {
                        Ok(result) => *crown_bounds = result,
                        Err(NyError::UnsupportedConfiguration(msg)) => {
                            debug!(
                                "{}: Conv2d Patches->Dense fallback at layer {}: {}",
                                label, layer_idx, msg
                            );
                            if let Some(fallback) = guard_dense_materialization_budget(
                                crown_bounds,
                                "conv2d_patches_unsupported_fallback",
                                layer,
                                layer_idx,
                                label,
                            )? {
                                return Ok(fallback);
                            }
                            let lb = crown_bounds.ensure_dense()?;
                            let next = conv_with_shape
                                .propagate_linear_with_engine_and_deadline(lb, engine, deadline)?;
                            if let Cow::Owned(next) = next {
                                *lb = next;
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                CrownBounds::Dense(lb) => {
                    // Thread deadline to enable intra-GEMM deadline checking (#3795).
                    let next = conv_with_shape
                        .propagate_linear_with_engine_and_deadline(lb, engine, deadline)?;
                    if let Cow::Owned(next) = next {
                        *lb = next;
                    }
                }
            }
            Ok(CrownStepResult::Continue)
        }
        // STRIDE-1 patches-native ConvTranspose2d CROWN backward (LEVER 2 stage
        // 2a). Mirrors the Conv2d arm: set the input shape, try the patches
        // path, and on UnsupportedConfiguration (stride>1, dilation, padding >
        // kernel-1, output_padding, or the Conv2d composition soundness guards)
        // fall back to the exact dense CROWN path. ConvTranspose2d is
        // `propagates_coeff_err = true`, so the patches arm emits the certified
        // coeff_err (inherited from the delegated Conv2d patches path).
        Layer::ConvTranspose2d(ct) => {
            let input_shape = pre_activation.shape();
            let (in_h, in_w) = if input_shape.len() >= 3 {
                (
                    input_shape[input_shape.len() - 2],
                    input_shape[input_shape.len() - 1],
                )
            } else {
                debug!(
                    "{}: ConvTranspose2d input shape too small: {:?}, using IBP",
                    label, input_shape
                );
                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                    reason: CrownIbpFallbackReason::CrownPropagationError,
                    details: format!("ConvTranspose2d input shape too small: {input_shape:?}"),
                }));
            };
            let mut ct_with_shape = ct.clone();
            ct_with_shape.set_input_shape(in_h, in_w);

            match crown_bounds {
                CrownBounds::Patches(pb) => {
                    match ct_with_shape.propagate_patches_engine(pb, engine) {
                        Ok(result) => *crown_bounds = result,
                        Err(NyError::UnsupportedConfiguration(msg)) => {
                            debug!(
                                "{}: ConvTranspose2d Patches->Dense fallback at layer {}: {}",
                                label, layer_idx, msg
                            );
                            if let Some(fallback) = guard_dense_materialization_budget(
                                crown_bounds,
                                "convtranspose2d_patches_unsupported_fallback",
                                layer,
                                layer_idx,
                                label,
                            )? {
                                return Ok(fallback);
                            }
                            let lb = crown_bounds.ensure_dense()?;
                            let next = ct_with_shape.propagate_linear_with_engine(lb, engine)?;
                            if let Cow::Owned(next) = next {
                                *lb = next;
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                CrownBounds::Dense(lb) => {
                    let next = ct_with_shape.propagate_linear_with_engine(lb, engine)?;
                    if let Cow::Owned(next) = next {
                        *lb = next;
                    }
                }
            }
            Ok(CrownStepResult::Continue)
        }
        // Phase 2: BatchNorm per-channel scaling natively on Patches.
        // BatchNorm is linear (y = scale*x + bias), so no pre-activation bounds
        // needed for the relaxation (there is no relaxation — exact substitution).
        // Part of #2613
        Layer::BatchNorm(bn) => try_patches_or_dense_fallback(
            |pb| bn.propagate_patches(pb),
            crown_bounds,
            pre_activation,
            layer,
            engine,
            layer_idx,
            label,
            "batchnorm_dense_fallback",
            "batchnorm_dense_dispatch",
        ),
        // Phase 3: Pooling layers natively support Patches backward.
        // AvgPool upsamples patches by pool kernel and divides by pool_size.
        // MaxPool applies relaxation slopes (argmax-lower / definite-winner).
        // Reference: designs/2026-03-01-patches-phase3-pooling-termination.md
        // Part of #2613
        Layer::AveragePool(ap) => try_patches_or_dense_fallback(
            |pb| ap.propagate_patches_with_bounds(pb, pre_activation),
            crown_bounds,
            pre_activation,
            layer,
            engine,
            layer_idx,
            label,
            "averagepool_dense_fallback",
            "averagepool_dense_dispatch",
        ),
        Layer::MaxPool2d(mp) => try_patches_or_dense_fallback(
            |pb| mp.propagate_patches_with_bounds(pb, pre_activation),
            crown_bounds,
            pre_activation,
            layer,
            engine,
            layer_idx,
            label,
            "maxpool_dense_fallback",
            "maxpool_dense_dispatch",
        ),
        // Phase 3: Linear/Flatten/Reshape terminate Patches mode — convert to Dense.
        // These layers flatten spatial structure, so Patches representation has no benefit.
        // Making this explicit rather than relying on the catch-all improves clarity.
        // Part of #2613 Phase 3.
        Layer::Linear(_) | Layer::Flatten(_) | Layer::Reshape(_) | Layer::Resize(_) => {
            if matches!(crown_bounds, CrownBounds::Patches(_)) {
                debug!(
                    "{}: Patches→Dense termination at layer {} ({})",
                    label,
                    layer_idx,
                    match layer {
                        Layer::Linear(_) => "Linear",
                        Layer::Flatten(_) => "Flatten",
                        Layer::Reshape(_) => "Reshape",
                        Layer::Resize(_) => "Resize",
                        _ => unreachable!(),
                    }
                );
            }
            if let Some(fallback) = guard_dense_materialization_budget(
                crown_bounds,
                "linear_flatten_reshape_resize_termination",
                layer,
                layer_idx,
                label,
            )? {
                return Ok(fallback);
            }
            let lb = crown_bounds.ensure_dense()?;
            crown_backward_step(layer, lb, pre_activation, engine, layer_idx, label)
        }
        // All remaining layers: try Patches activation dispatch, then Dense fallback.
        // Exhaustive listing — adding a new Layer variant is a compile error (#3424).
        //
        // Elementwise activations:
        Layer::ReLU(_) | Layer::GELU(_) | Layer::SiLU(_) | Layer::Tanh(_)
        | Layer::Sigmoid(_) | Layer::Exp(_) | Layer::Log(_) | Layer::Sqrt(_)
        | Layer::Reciprocal(_) | Layer::Softplus(_) | Layer::HardSwish(_)
        | Layer::Mish(_) | Layer::Selu(_) | Layer::Softsign(_) | Layer::Arctan(_)
        | Layer::Tan(_) | Layer::Sin(_) | Layer::Cos(_) | Layer::Elu(_)
        | Layer::Celu(_) | Layer::LeakyReLU(_) | Layer::HardSigmoid(_)
        | Layer::Clip(_) | Layer::ThresholdedRelu(_) | Layer::Abs(_)
        | Layer::PowConstant(_) | Layer::Floor(_) | Layer::Ceil(_)
        | Layer::Round(_) | Layer::Trunc(_) | Layer::Sign(_) | Layer::PRelu(_) | Layer::Shrink(_)
        | Layer::Snake(_) | Layer::Compare(_)
        // Softmax family:
        | Layer::Softmax(_) | Layer::CausalSoftmax(_) | Layer::LogSoftmax(_)
        | Layer::LogSumExp(_)
        // Normalization:
        | Layer::LayerNorm(_) | Layer::RmsNorm(_) | Layer::InstanceNorm1d(_)
        | Layer::GroupNorm(_) | Layer::AdaIN1d(_)
        // Convolutions (non-2D):
        | Layer::Conv1d(_) | Layer::ConvTranspose1d(_)
        // Constant arithmetic:
        | Layer::AddConstant(_) | Layer::MulConstant(_) | Layer::DivConstant(_)
        | Layer::SubConstant(_)
        // Reductions:
        | Layer::ReduceMean(_) | Layer::ReduceSum(_) | Layer::CumSum(_)
        | Layer::ReduceMax(_) | Layer::ReduceMin(_)
        | Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_)
        // Shape transforms:
        | Layer::Transpose(_) | Layer::Tile(_) | Layer::Gather(_)
        | Layer::ScatterAdd(_) | Layer::IndexAdd(_) | Layer::ScatterNd(_)
        | Layer::Slice(_) | Layer::Pad(_)
        | Layer::Squeeze(_) | Layer::Unsqueeze(_)
        | Layer::QdqPerturbation(_)
        // Positional encoding:
        | Layer::RoPE(_)
        // Binary / multi-input ops:
        | Layer::MatMul(_) | Layer::MulBinary(_) | Layer::Add(_) | Layer::Concat(_)
        | Layer::Sub(_) | Layer::Div(_) | Layer::Atan2(_) | Layer::BilinearCrown(_)
        | Layer::MinBinary(_)
        | Layer::MaxBinary(_) | Layer::ExpandLikeLastAxis(_)
        | Layer::CompareTensor(_)
        // Special:
        | Layer::SelfAttention(_) | Layer::Where(_) | Layer::NonZero(_)
        | Layer::SkipMerge(_) | Layer::OpaqueSkip(_) => {
            // Phase 2: Element-wise activations natively support Patches backward.
            // When in Patches mode, use propagate_patches_with_bounds to keep sparse
            // structure. When in Dense mode or for unsupported layers, fall through to
            // standard Dense dispatch. Generated via for_each_patches_capable_activation!
            // Part of #2613 Phase 2 step 11.
            if let CrownBounds::Patches(pb) = crown_bounds {
                macro_rules! patches_activation_dispatch {
                    ($($Variant:ident),*) => {
                        match layer {
                            $(Layer::$Variant(inner) => {
                                match inner.propagate_patches_with_bounds(pb, pre_activation) {
                                    Ok(result) => {
                                        *crown_bounds = result;
                                        return Ok(CrownStepResult::Continue);
                                    }
                                    Err(NyError::NumericalInstability(_)) => {
                                        debug!(
                                            "{}: {} Patches backward NumericalInstability \
                                             at layer {}, falling back to Dense",
                                            label, stringify!($Variant), layer_idx
                                        );
                                        if let Some(fallback) = guard_dense_materialization_budget(
                                            crown_bounds,
                                            "activation_numerical_dense_fallback",
                                            layer,
                                            layer_idx,
                                            label,
                                        )? {
                                            return Ok(fallback);
                                        }
                                        let lb = crown_bounds.ensure_dense()?;
                                        return crown_backward_step(
                                            layer, lb, pre_activation, engine,
                                            layer_idx, label,
                                        );
                                    }
                                    Err(e) => return Err(e),
                                }
                            })*
                            _ => {} // Not a patches-capable activation, fall through
                        }
                    };
                }
                crate::layers::layer_enum::for_each_patches_capable_activation!(
                    patches_activation_dispatch
                );
            }
            // All other layers (or Dense mode): ensure Dense, then delegate.
            // ensure_dense() is a no-op when already Dense.
            if let Some(fallback) = guard_dense_materialization_budget(
                crown_bounds,
                "generic_dense_dispatch",
                layer,
                layer_idx,
                label,
            )? {
                return Ok(fallback);
            }
            let lb = crown_bounds.ensure_dense()?;
            crown_backward_step(layer, lb, pre_activation, engine, layer_idx, label)
        }
    }
}

#[cfg(test)]
#[path = "patches_step_tests.rs"]
mod tests;
