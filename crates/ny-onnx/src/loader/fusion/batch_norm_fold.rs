// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::bn_fold_interval::{
    channel_affine_interval, emit_report, fold_interval_report, interval_report_enabled,
    ChannelAffineInterval, ChannelAxis,
};
use crate::loader::const_fold::common::read_tensor_i64s;
use crate::loader::numeric_cast::i64_to_f32_warned;
use crate::loader::BatchNormFoldingPolicy;
use crate::onnx_proto::{self, attribute_type};
use crate::WeightStore;
use ndarray::{Array1, ArrayD, Axis};
use std::collections::{HashMap, HashSet};
use tracing::debug;

const DEFAULT_BATCH_NORM_EPSILON: f32 = 1.0e-5;

/// Extended BN folds (#cgan-structural-fold): ConvTranspose+BN,
/// Gemm->Reshape->BN, and BN->Reshape->Gemm. Default ON;
/// `NY_BN_FOLD_EXT=0` disables ONLY the three extended patterns (the landed
/// Conv/Gemm fold is unconditional, as before).
fn extended_bn_folds_enabled() -> bool {
    std::env::var("NY_BN_FOLD_EXT").ok().as_deref() != Some("0")
}

struct BatchNormAffine {
    scale: Vec<f32>,
    shift: Vec<f32>,
    /// Rigorous f64 enclosures of the same per-channel `scale`/`shift`, present
    /// only under the `NY_BN_FOLD_INTERVAL_REPORT=1` dark gate
    /// (#bn-interval-fold). Purely observational: nothing in the fold reads
    /// this, so the f32 `scale`/`shift` above — and therefore every stored
    /// weight — are bit-identical whether it is populated or not.
    intervals: Option<Vec<ChannelAffineInterval>>,
}

#[cfg(test)]
pub(crate) fn fold_batch_norm_into_conv_linear(
    nodes: &mut [onnx_proto::NodeProto],
    weights: &mut WeightStore,
) -> HashSet<usize> {
    fold_batch_norm_into_conv_linear_with_context(
        nodes,
        weights,
        &HashMap::new(),
        &HashSet::new(),
        BatchNormFoldingPolicy::LegacyEnvironment,
    )
}

/// Context-aware entry used by graph conversion. Exact tensor shapes are
/// required by the BN -> Reshape -> Gemm fold, while graph output names prevent
/// removal of a BN value that remains externally observable.
pub(crate) fn fold_batch_norm_into_conv_linear_with_context(
    nodes: &mut [onnx_proto::NodeProto],
    weights: &mut WeightStore,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    graph_output_names: &HashSet<String>,
    policy: BatchNormFoldingPolicy,
) -> HashSet<usize> {
    let mut consumed = HashSet::new();
    if policy == BatchNormFoldingPolicy::PreserveRaw {
        return consumed;
    }

    for bn_idx in 0..nodes.len() {
        if nodes[bn_idx].op_type != "BatchNormalization" {
            continue;
        }
        if nodes[bn_idx].input.len() < 5 {
            continue;
        }

        let bn_output_name = match inference_batch_norm_output(&nodes[bn_idx]) {
            Some(name) => name,
            None => continue,
        };

        let predecessor_output_name = match nodes[bn_idx]
            .input
            .first()
            .filter(|value| !value.is_empty())
        {
            Some(name) => name.clone(),
            None => continue,
        };

        let predecessor_idx = match producer_index(nodes, predecessor_output_name.as_str()) {
            Some(idx) => idx,
            None => {
                if extended_bn_folds_enabled()
                    && try_fold_bn_reshape_gemm(
                        nodes,
                        weights,
                        tensor_shapes,
                        graph_output_names,
                        bn_idx,
                    )
                {
                    consumed.insert(bn_idx);
                }
                continue;
            }
        };
        let predecessor = &nodes[predecessor_idx];
        let predecessor_op_type = predecessor.op_type.clone();

        // #cgan-structural-fold: Gemm -> Reshape -> BN across-Reshape fold
        // (mirrors alpha-beta-CROWN's `merge_gemm_reshape_bn`). Handled by a
        // dedicated routine because the weight-carrying node sits BEHIND the
        // Reshape; the shared Conv/Gemm/ConvTranspose flow below assumes the
        // immediate predecessor carries the weights.
        if predecessor_op_type == "Reshape" {
            if extended_bn_folds_enabled()
                && try_fold_gemm_reshape_bn(
                    nodes,
                    weights,
                    tensor_shapes,
                    graph_output_names,
                    bn_idx,
                    predecessor_idx,
                )
            {
                consumed.insert(bn_idx);
            }
            continue;
        }

        // #cgan-structural-fold: ConvTranspose predecessors join the shared
        // flow. BN's per-output-channel affine acts AFTER the transposed-conv
        // summation, so it folds as a per-channel scale of the kernel exactly
        // like Conv — only the channel axis differs (ONNX ConvTranspose kernels
        // are [C_in, C_out/group, k...], so C_out is axis 1, not axis 0).
        // Grouped kernels are skipped: with group>1 the output-channel index
        // depends on the input-channel group, not axis 1 alone.
        let is_conv_transpose = predecessor_op_type == "ConvTranspose";
        if is_conv_transpose
            && (!extended_bn_folds_enabled() || conv_transpose_group(predecessor) != 1)
        {
            continue;
        }
        if predecessor_op_type != "Conv" && predecessor_op_type != "Gemm" && !is_conv_transpose {
            // alpha-beta-CROWN's cGAN recipe also folds the discriminator
            // tail BN -> Reshape -> Gemm forward into the Gemm. Restrict this
            // successor fold to BNs the historical predecessor folds cannot
            // consume, preserving their established rewrite priority.
            if extended_bn_folds_enabled()
                && try_fold_bn_reshape_gemm(
                    nodes,
                    weights,
                    tensor_shapes,
                    graph_output_names,
                    bn_idx,
                )
            {
                consumed.insert(bn_idx);
            }
            continue;
        }
        // Guard: skip Gemm+BN fusion when Gemm has non-default parameters.
        // transA=1 changes input layout (#2320), alpha/beta!=1.0 changes the
        // matmul/bias scaling that BN fusion equations assume (#2319).
        if predecessor_op_type == "Gemm" {
            if gemm_trans_a(predecessor) {
                debug!(
                    "Skipping BN fusion for Gemm node {} — transA=1 not supported",
                    predecessor_idx
                );
                continue;
            }
            if !gemm_has_exact_default_affine(predecessor) {
                let alpha = gemm_alpha(predecessor);
                let beta = gemm_beta(predecessor);
                debug!(
                    "Skipping BN fusion for Gemm node {} — alpha={}, beta={} (non-default)",
                    predecessor_idx, alpha, beta
                );
                continue;
            }
        }
        let predecessor_primary_output = match predecessor.output.first() {
            Some(output) if !output.is_empty() => output.clone(),
            _ => continue,
        };
        // This fold replaces the predecessor's output name with BN's Y. A
        // separately exposed predecessor value would otherwise disappear.
        if graph_output_names.contains(&predecessor_primary_output) {
            continue;
        }
        let consumers = consumer_indices(nodes, predecessor_primary_output.as_str());
        if consumers.as_slice() != [bn_idx] {
            continue;
        }

        let affine = match batch_norm_affine(&nodes[bn_idx], weights) {
            Some(value) => value,
            None => continue,
        };

        let weight_name = match predecessor.input.get(1).filter(|name| !name.is_empty()) {
            Some(name) => name.clone(),
            None => continue,
        };
        // Initializers may legally be graph outputs. Since this pass mutates
        // the name-keyed tensor in place, an observable initializer must retain
        // its authored value and blocks the fold.
        if graph_output_names.contains(&weight_name) {
            continue;
        }
        // The fold rewrites the weight (and any existing bias) in place in the
        // name-keyed WeightStore. If another node references the same
        // initializer — weight tying, Siamese/twin branches — that node would
        // silently read the BN-scaled tensor (and a twin branch with its own
        // BN would scale it a second time), so only fold when this node is the
        // initializer's sole consumer.
        if consumer_indices(nodes, weight_name.as_str()).as_slice() != [predecessor_idx] {
            debug!(
                "Skipping BN fusion for node {} — weight {} has other consumers",
                predecessor_idx, weight_name
            );
            continue;
        }
        let predecessor_weight = match weights.get(weight_name.as_str()) {
            Some(weight) => weight.clone(),
            None => continue,
        };

        let existing_bias_name = predecessor
            .input
            .get(2)
            .filter(|name| !name.is_empty())
            .cloned();
        if existing_bias_name.as_deref() == Some(weight_name.as_str()) {
            debug!(
                "Skipping BN fusion for node {} — weight B and bias C share name {}",
                predecessor_idx, weight_name
            );
            continue;
        }
        if existing_bias_name
            .as_ref()
            .is_some_and(|name| graph_output_names.contains(name))
        {
            continue;
        }
        if let Some(bias_name) = existing_bias_name.as_deref() {
            if consumer_indices(nodes, bias_name).as_slice() != [predecessor_idx] {
                debug!(
                    "Skipping BN fusion for node {} — bias {} has other consumers",
                    predecessor_idx, bias_name
                );
                continue;
            }
        }
        let mut existing_bias = match existing_bias_name.as_ref() {
            Some(name) => {
                let Some(bias) = weights.get(name).cloned() else {
                    // A dynamic bias/C input cannot be replaced by a
                    // synthesized constant without changing the model.
                    continue;
                };
                Some(bias)
            }
            None => None,
        };
        if predecessor_op_type == "Gemm" {
            existing_bias = match existing_bias.as_ref() {
                Some(bias) => match normalize_gemm_c(bias, affine.scale.len()) {
                    Some(normalized) => Some(normalized),
                    None => continue,
                },
                None => None,
            };
        }

        let fused_weight = if predecessor_op_type == "Conv" {
            fuse_conv_weight(&predecessor_weight, &affine.scale)
        } else if is_conv_transpose {
            fuse_conv_transpose_weight(&predecessor_weight, &affine.scale)
        } else {
            // Gemm: transB determines weight layout.
            //   transB=1 (default PyTorch): weight is (out, in) → scale axis 0
            //   transB=0: weight is (in, out) → scale axis 1
            // Fix for #2309: without transB, square weights always matched axis 0.
            let trans_b = gemm_trans_b(predecessor);
            fuse_gemm_weight(&predecessor_weight, &affine.scale, trans_b)
        };
        let fused_weight = match fused_weight {
            Some(weight) => weight,
            None => continue,
        };

        let fused_bias = match fuse_bias(existing_bias.as_ref(), &affine.scale, &affine.shift) {
            Some(bias) => bias,
            None => continue,
        };

        let bias_name = match existing_bias_name {
            Some(name) => name,
            None => match fresh_synthetic_bias_name(
                nodes,
                weights,
                tensor_shapes,
                graph_output_names,
                &nodes[predecessor_idx],
                bn_idx,
            ) {
                Some(name) => name,
                None => continue,
            },
        };

        // #bn-interval-fold (report-only, dark-gated). Runs AFTER the fold has
        // fully validated and committed to these exact tensors, so the reported
        // width describes what the loader will actually verify against. Placed
        // before the inserts purely because `fused_weight`/`fused_bias` are
        // moved by them.
        if let Some(intervals) = affine.intervals.as_ref() {
            let channel_axis = ChannelAxis {
                axis: if predecessor_op_type == "Conv" {
                    0
                } else if is_conv_transpose {
                    1
                } else if gemm_trans_b(&nodes[predecessor_idx]) {
                    0
                } else {
                    1
                },
                block: 1,
            };
            if let Some(report) = fold_interval_report(
                &predecessor_weight,
                &fused_weight,
                existing_bias.as_ref(),
                &fused_bias,
                intervals,
                channel_axis,
            ) {
                emit_report(&predecessor_op_type, predecessor_idx, bn_idx, &report);
            }
        }

        weights.insert(weight_name.clone(), fused_weight);
        weights.insert(bias_name.clone(), fused_bias);

        if let Some(predecessor_node) = nodes.get_mut(predecessor_idx) {
            if predecessor_node.output.is_empty() {
                predecessor_node.output.push(bn_output_name.clone());
            } else {
                predecessor_node.output[0] = bn_output_name.clone();
            }
            if predecessor_node.input.len() >= 3 {
                predecessor_node.input[2] = bias_name.clone();
            } else {
                predecessor_node.input.push(bias_name.clone());
            }
        }

        if let Some(batch_norm_node) = nodes.get_mut(bn_idx) {
            batch_norm_node.input.clear();
            batch_norm_node.output.clear();
        }
        consumed.insert(bn_idx);

        debug!(
            "Fused {} node {} with BatchNormalization node {}",
            predecessor_op_type, predecessor_idx, bn_idx
        );
    }

    consumed
}

/// BatchNormalization folding is inference-only. ONNX permits optional
/// training outputs after Y; consuming the node is valid only when Y is its
/// sole nonempty output and `training_mode` is absent/zero.
fn inference_batch_norm_output(node: &onnx_proto::NodeProto) -> Option<String> {
    if node.attribute.iter().any(|attr| {
        attr.name == "training_mode" && (attr.r#type != attribute_type::INT || attr.i_value() != 0)
    }) {
        return None;
    }
    let output = node.output.first().filter(|name| !name.is_empty())?;
    (node.output.iter().filter(|name| !name.is_empty()).count() == 1).then(|| output.clone())
}

fn batch_norm_affine(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<BatchNormAffine> {
    if node.input.len() < 5 {
        return None;
    }

    let ny = weight_vector(weights, node.input.get(1)?.as_str())?;
    let beta = weight_vector(weights, node.input.get(2)?.as_str())?;
    let mean = weight_vector(weights, node.input.get(3)?.as_str())?;
    let var = weight_vector(weights, node.input.get(4)?.as_str())?;

    if ny.is_empty() || ny.len() != beta.len() || ny.len() != mean.len() || ny.len() != var.len() {
        return None;
    }

    let epsilon = batch_norm_epsilon(node);
    let mut scale = Vec::with_capacity(ny.len());
    let mut shift = Vec::with_capacity(ny.len());
    // #bn-interval-fold: side-channel enclosures, gated OFF by default. Built
    // in the same loop so the report cannot drift out of sync with the f32
    // values it certifies.
    let mut intervals = interval_report_enabled().then(|| Vec::with_capacity(ny.len()));

    // Fold equations from alpha-beta-CROWN's Conv/BN merge logic:
    // complete_verifier/onnx_opt.py (fuse_conv_and_bn + merge_bn branches).
    for idx in 0..ny.len() {
        // #bn-fold-restore: compose in f64 with one final rounding, so the
        // uncertified residual on the folded parameters is at most the f32
        // storage rounding (<= 0.5 ulp) rather than accumulated f32 arithmetic
        // error. The reference fold (alpha-beta-CROWN onnx_opt.py) composes in
        // f32; this is strictly tighter.
        let denominator = f64::from(var[idx]) + f64::from(epsilon);
        let denominator = denominator.sqrt();
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        let channel_scale = (f64::from(ny[idx]) / denominator) as f32;
        let channel_shift =
            (f64::from(beta[idx]) - f64::from(ny[idx]) * f64::from(mean[idx]) / denominator) as f32;
        if !channel_scale.is_finite() || !channel_shift.is_finite() {
            return None;
        }
        scale.push(channel_scale);
        shift.push(channel_shift);
        if let Some(intervals) = intervals.as_mut() {
            // A channel that cannot be enclosed is recorded as an all-real
            // interval so the report shows it as unenclosable rather than
            // silently shrinking the reported width.
            intervals.push(
                channel_affine_interval(ny[idx], beta[idx], mean[idx], var[idx], epsilon)
                    .unwrap_or_else(ChannelAffineInterval::unenclosable),
            );
        }
    }

    Some(BatchNormAffine {
        scale,
        shift,
        intervals,
    })
}

fn batch_norm_epsilon(node: &onnx_proto::NodeProto) -> f32 {
    node.attribute
        .iter()
        .find(|attr| attr.name == "epsilon")
        .and_then(|attr| match attr.r#type {
            attribute_type::FLOAT => Some(attr.f_value()),
            attribute_type::INT => Some(i64_to_f32_warned(attr.i_value(), "BatchNorm epsilon INT")),
            attribute_type::FLOATS => attr.floats.first().copied(),
            attribute_type::INTS => attr
                .ints
                .first()
                .map(|value| i64_to_f32_warned(*value, "BatchNorm epsilon INTS")),
            _ => None,
        })
        .unwrap_or(DEFAULT_BATCH_NORM_EPSILON)
}

fn producer_index(nodes: &[onnx_proto::NodeProto], output_name: &str) -> Option<usize> {
    nodes
        .iter()
        .position(|node| node.output.iter().any(|name| name == output_name))
}

fn consumer_indices(nodes: &[onnx_proto::NodeProto], input_name: &str) -> Vec<usize> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| {
            if node.input.iter().any(|name| name == input_name) {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

fn weight_vector(weights: &WeightStore, name: &str) -> Option<Vec<f32>> {
    let tensor = weights.get(name)?;
    Some(tensor.iter().copied().collect())
}

fn fuse_conv_weight(weight: &ArrayD<f32>, scale: &[f32]) -> Option<ArrayD<f32>> {
    if weight.ndim() < 1 || weight.shape().first().copied().unwrap_or_default() != scale.len() {
        return None;
    }

    let mut fused = weight.clone();
    for (channel_idx, mut channel) in fused.axis_iter_mut(Axis(0)).enumerate() {
        channel *= scale[channel_idx];
    }
    Some(fused)
}

/// Extract the `group` attribute from a ConvTranspose node (default 1).
///
/// With `group > 1` the kernel is `[C_in, C_out/group, k...]` and output
/// channel `o = g*(C_out/group) + j` draws from kernel column `j` of group `g`
/// only — a per-axis-1 scale can no longer express the per-output-channel BN
/// scale, so fusion is skipped for grouped transposed convolutions.
fn conv_transpose_group(node: &onnx_proto::NodeProto) -> i64 {
    node.attribute
        .iter()
        .find(|attr| attr.name == "group")
        .map(|attr| attr.i_value())
        .unwrap_or(1)
}

/// Fuse ConvTranspose weight with per-output-channel BN scale
/// (#cgan-structural-fold).
///
/// ONNX ConvTranspose kernels are laid out `[C_in, C_out/group, kH, kW]`
/// (or `[C_in, C_out, kL]` for 1-d) — the OUTPUT channel axis is axis 1,
/// unlike Conv where it is axis 0. Callers must have already rejected
/// `group != 1`, so `weight.shape()[1] == C_out == scale.len()`.
///
/// Correctness: `ConvTranspose(x)[o, :, :] = sum_i x[i] * W[i, o, ...] + b[o]`,
/// so a per-`o` affine after the op distributes over the sum:
/// `s[o] * ConvTranspose(x)[o] + t[o] = ConvTranspose_with(W', b')` where
/// `W'[i, o, ...] = s[o] * W[i, o, ...]` and `b'[o] = s[o]*b[o] + t[o]` —
/// the same equations as the landed Conv fold, transposed to axis 1.
fn fuse_conv_transpose_weight(weight: &ArrayD<f32>, scale: &[f32]) -> Option<ArrayD<f32>> {
    if weight.ndim() < 2 || weight.shape()[1] != scale.len() {
        return None;
    }

    let mut fused = weight.clone();
    for (channel_idx, mut channel) in fused.axis_iter_mut(Axis(1)).enumerate() {
        channel *= scale[channel_idx];
    }
    Some(fused)
}

/// Extract the `transA` attribute from a Gemm node.
///
/// ONNX Gemm spec: transA defaults to 0 (no transpose). When transA=1, the
/// first input matrix A is transposed before multiplication. BN fusion logic
/// assumes standard (non-transposed) input layout and produces incorrect fused
/// weights when transA=1. Guard: skip fusion for transA=1 (#2320).
fn gemm_trans_a(node: &onnx_proto::NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attr| attr.name == "transA")
        .is_some_and(|attr| attr.i_value() != 0)
}

/// Extract the `transB` attribute from a Gemm node.
///
/// ONNX Gemm spec: transB defaults to 0 (no transpose). PyTorch exporters
/// typically set transB=1 so weight is stored as (out_features, in_features).
fn gemm_trans_b(node: &onnx_proto::NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attr| attr.name == "transB")
        .is_some_and(|attr| attr.i_value() != 0)
}

/// Extract the `alpha` scalar attribute from a Gemm node.
///
/// ONNX Gemm computes: Y = alpha * A' * B' + beta * C.
/// Default alpha=1.0. When alpha!=1.0, the BN fusion equations change because
/// BN sees alpha-scaled matmul output. Guard: skip fusion for alpha!=1.0 (#2319).
fn gemm_alpha(node: &onnx_proto::NodeProto) -> f32 {
    node.attribute
        .iter()
        .find(|attr| attr.name == "alpha")
        .map(|attr| attr.f_value())
        .unwrap_or(1.0)
}

/// Extract the `beta` scalar attribute from a Gemm node.
///
/// ONNX Gemm computes: Y = alpha * A' * B' + beta * C.
/// Default beta=1.0. When beta!=1.0, the existing bias is scaled by beta before
/// BN sees it, so fuse_bias produces incorrect results. Guard: skip fusion for
/// beta!=1.0 (#2319).
fn gemm_beta(node: &onnx_proto::NodeProto) -> f32 {
    node.attribute
        .iter()
        .find(|attr| attr.name == "beta")
        .map(|attr| attr.f_value())
        .unwrap_or(1.0)
}

/// Every fold in this module uses algebra for the exact ONNX default
/// `Y = A' * B' + C`. Approximate comparisons are unsound here: even an
/// adjacent f32 changes the function when the factor is dropped, and NaN must
/// fail closed rather than passing an ordered tolerance check.
pub(crate) fn gemm_has_exact_default_affine(node: &onnx_proto::NodeProto) -> bool {
    gemm_alpha(node) == 1.0 && gemm_beta(node) == 1.0
}

/// Fuse Gemm weight with per-channel BN scale, respecting transB.
///
/// - `trans_b=true`: weight is (out, in) — scale along axis 0 (rows)
/// - `trans_b=false`: weight is (in, out) — scale along axis 1 (columns)
///
/// Fix for #2309: without transB, square weights always matched axis 0.
fn fuse_gemm_weight(weight: &ArrayD<f32>, scale: &[f32], trans_b: bool) -> Option<ArrayD<f32>> {
    if weight.ndim() != 2 {
        return None;
    }

    let scale_axis = if trans_b { 0 } else { 1 };
    if weight.shape()[scale_axis] != scale.len() {
        return None;
    }

    let mut fused = weight.clone();
    for (idx, mut slice) in fused.axis_iter_mut(Axis(scale_axis)).enumerate() {
        slice *= scale[idx];
    }
    Some(fused)
}

/// Fold `Gemm -> Reshape -> BatchNormalization` into the Gemm across the
/// Reshape (#cgan-structural-fold; mirrors alpha-beta-CROWN's
/// `merge_gemm_reshape_bn` / `fuse_cgan_gemm_reshape_bn` in
/// complete_verifier/onnx_opt.py).
///
/// Shape algebra: the Gemm produces `[N, F]`; the Reshape views it as
/// `[N, C, d2, d3, ...]` in row-major (C) order with `F = C * block`,
/// `block = prod(d2..)`. Output feature `f` therefore lands in BN channel
/// `c(f) = f / block`, so BN's per-channel affine `y = s[c]*x + t[c]` becomes
/// a per-FEATURE (block-diagonal) affine on the Gemm output:
///   `W'[f, :] = s[f/block] * W[f, :]`   (transB=1; column scale when transB=0)
///   `b'[f]    = s[f/block] * b[f] + t[f/block]`
/// The Reshape node itself is preserved (it still re-ranks `[N, F]` to
/// `[N, C, d2, ...]`); only the BN node is consumed. Soundness convention is
/// identical to the landed Conv/Gemm fold: exact fold equations evaluated in
/// f32 (matching abc's reference rewrite), guarded by sole-consumer checks on
/// the rewritten initializers.
///
/// Returns true when the fold fired (caller marks the BN node consumed).
fn try_fold_gemm_reshape_bn(
    nodes: &mut [onnx_proto::NodeProto],
    weights: &mut WeightStore,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    graph_output_names: &HashSet<String>,
    bn_idx: usize,
    reshape_idx: usize,
) -> bool {
    let Some(bn_output_name) = inference_batch_norm_output(&nodes[bn_idx]) else {
        return false;
    };

    // Reshape guards: static shape input, single consumer (the BN).
    let reshape = &nodes[reshape_idx];
    if reshape.op_type != "Reshape" || reshape.input.len() < 2 {
        return false;
    }
    let Some(reshape_output) = reshape
        .output
        .first()
        .filter(|value| !value.is_empty())
        .cloned()
    else {
        return false;
    };
    // The fold changes this intermediate from raw Gemm output layout to the
    // post-BN value (and renames it), so it cannot be separately observable.
    if graph_output_names.contains(&reshape_output) {
        return false;
    }
    if consumer_indices(nodes, reshape_output.as_str()).as_slice() != [bn_idx] {
        return false;
    }
    let reshape_data_input = match nodes[reshape_idx]
        .input
        .first()
        .filter(|value| !value.is_empty())
    {
        Some(name) => name.clone(),
        None => return false,
    };
    let shape_name = nodes[reshape_idx].input[1].clone();
    let Some(target_shape) = read_tensor_i64s(weights, shape_name.as_str()) else {
        return false;
    };
    if target_shape.len() < 2 {
        return false;
    }

    // Gemm guards: same non-default-attribute rejections as the direct fold.
    let Some(gemm_idx) = producer_index(nodes, reshape_data_input.as_str()) else {
        return false;
    };
    let gemm = &nodes[gemm_idx];
    if gemm.op_type != "Gemm" {
        return false;
    }
    if gemm_trans_a(gemm) {
        debug!(
            "Skipping Gemm->Reshape->BN fold for Gemm node {} — transA=1 not supported",
            gemm_idx
        );
        return false;
    }
    if !gemm_has_exact_default_affine(gemm) {
        let alpha = gemm_alpha(gemm);
        let beta = gemm_beta(gemm);
        debug!(
            "Skipping Gemm->Reshape->BN fold for Gemm node {} — alpha={}, beta={}",
            gemm_idx, alpha, beta
        );
        return false;
    }
    let Some(gemm_primary_output) = gemm
        .output
        .first()
        .filter(|value| !value.is_empty())
        .cloned()
    else {
        return false;
    };
    // The Gemm output remains named but its value becomes BN-pretransformed.
    if graph_output_names.contains(&gemm_primary_output) {
        return false;
    }
    if consumer_indices(nodes, gemm_primary_output.as_str()).as_slice() != [reshape_idx] {
        return false;
    }

    let Some(affine) = batch_norm_affine(&nodes[bn_idx], weights) else {
        return false;
    };
    let channels = affine.scale.len();

    let gemm = &nodes[gemm_idx];
    let weight_name = match gemm.input.get(1).filter(|name| !name.is_empty()) {
        Some(name) => name.clone(),
        None => return false,
    };
    if graph_output_names.contains(&weight_name) {
        return false;
    }
    // Same sole-consumer guards as the direct fold: the rewrite mutates the
    // name-keyed store in place, so shared initializers must block it.
    if consumer_indices(nodes, weight_name.as_str()).as_slice() != [gemm_idx] {
        debug!(
            "Skipping Gemm->Reshape->BN fold for node {} — weight {} has other consumers",
            gemm_idx, weight_name
        );
        return false;
    }
    let Some(gemm_weight) = weights.get(weight_name.as_str()).cloned() else {
        return false;
    };
    if gemm_weight.ndim() != 2 {
        return false;
    }
    let trans_b = gemm_trans_b(&nodes[gemm_idx]);
    let features = if trans_b {
        gemm_weight.shape()[0]
    } else {
        gemm_weight.shape()[1]
    };

    // Reshape-shape guards. `target_shape[0]` is the batch axis and may be
    // symbolic (-1 / 0 / N) — the product check below pins the NON-batch axes
    // to exactly `F`, which is what makes the row-major channel map
    // `c(f) = f / block` valid for any batch size. All non-batch entries must
    // be positive literals: `0` (copy input dim) and `-1` (inferred) entries
    // would need input-shape knowledge this pass does not have.
    if target_shape[1] != channels as i64 {
        return false;
    }
    if target_shape[2..].iter().any(|dim| *dim <= 0) {
        return false;
    }
    let mut non_batch_product: i64 = 1;
    for dim in &target_shape[1..] {
        non_batch_product = match non_batch_product.checked_mul(*dim) {
            Some(value) => value,
            None => return false,
        };
    }
    if channels == 0 || non_batch_product != features as i64 {
        return false;
    }
    let block = features / channels;
    debug_assert_eq!(block as i64, target_shape[2..].iter().product::<i64>());

    let existing_bias_name = nodes[gemm_idx]
        .input
        .get(2)
        .filter(|name| !name.is_empty())
        .cloned();
    if existing_bias_name.as_deref() == Some(weight_name.as_str()) {
        return false;
    }
    if existing_bias_name
        .as_ref()
        .is_some_and(|name| graph_output_names.contains(name))
    {
        return false;
    }
    if let Some(bias_name) = existing_bias_name.as_deref() {
        if consumer_indices(nodes, bias_name).as_slice() != [gemm_idx] {
            debug!(
                "Skipping Gemm->Reshape->BN fold for node {} — bias {} has other consumers",
                gemm_idx, bias_name
            );
            return false;
        }
    }
    let existing_bias = match existing_bias_name.as_ref() {
        Some(name) => {
            let Some(bias) = weights.get(name).cloned() else {
                // A dynamic Gemm C input cannot be replaced by a synthesized
                // constant without changing the model.
                return false;
            };
            let Some(normalized) = normalize_gemm_c(&bias, features) else {
                return false;
            };
            Some(normalized)
        }
        None => None,
    };

    let Some(fused_weight) = fuse_gemm_reshape_weight(&gemm_weight, &affine.scale, trans_b, block)
    else {
        return false;
    };
    let Some(fused_bias) = fuse_gemm_reshape_bias(
        existing_bias.as_ref(),
        &affine.scale,
        &affine.shift,
        block,
        features,
    ) else {
        return false;
    };

    let bias_name = match existing_bias_name {
        Some(name) => name,
        None => {
            let Some(name) = fresh_synthetic_bias_name(
                nodes,
                weights,
                tensor_shapes,
                graph_output_names,
                &nodes[gemm_idx],
                bn_idx,
            ) else {
                return false;
            };
            name
        }
    };

    // #bn-interval-fold (report-only, dark-gated). This fold's channel map is
    // `c(f) = f / block`, which is exactly what `ChannelAxis.block` models, so
    // the same reporter covers it. The BN->Reshape->Gemm tail fold is
    // deliberately NOT reported: its fused bias is an inner product over all
    // features rather than a per-channel affine, so its enclosure needs a
    // summation interval this increment does not implement.
    if let Some(intervals) = affine.intervals.as_ref() {
        let channel_axis = ChannelAxis {
            axis: if trans_b { 0 } else { 1 },
            block,
        };
        if let Some(report) = fold_interval_report(
            &gemm_weight,
            &fused_weight,
            existing_bias.as_ref(),
            &fused_bias,
            intervals,
            channel_axis,
        ) {
            emit_report("Gemm->Reshape->BN", gemm_idx, bn_idx, &report);
        }
    }

    weights.insert(weight_name, fused_weight);
    weights.insert(bias_name.clone(), fused_bias);

    if let Some(gemm_node) = nodes.get_mut(gemm_idx) {
        if gemm_node.input.len() >= 3 {
            gemm_node.input[2] = bias_name;
        } else {
            gemm_node.input.push(bias_name);
        }
    }
    // The Reshape keeps doing its rank change on the (now BN-scaled) Gemm
    // output; it adopts the BN's output name so downstream consumers and graph
    // outputs are preserved.
    if let Some(reshape_node) = nodes.get_mut(reshape_idx) {
        reshape_node.output[0] = bn_output_name;
    }
    if let Some(batch_norm_node) = nodes.get_mut(bn_idx) {
        batch_norm_node.input.clear();
        batch_norm_node.output.clear();
    }

    debug!(
        "Fused Gemm node {} + Reshape node {} with BatchNormalization node {} (C={}, block={})",
        gemm_idx, reshape_idx, bn_idx, channels, block
    );
    true
}

/// Fold `BatchNormalization -> Reshape -> Gemm` into the Gemm across the
/// channel-major flatten (#cgan-structural-fold; alpha-beta-CROWN
/// `merge_bn_reshape_gemm`). For `F = C * block`, feature `f` belongs to BN
/// channel `f / block`, hence
/// `W'[o,f] = W[o,f] * scale[f/block]` and
/// `b'[o] = b[o] + sum_f W[o,f] * shift[f/block]` (transposed for transB=0).
fn try_fold_bn_reshape_gemm(
    nodes: &mut [onnx_proto::NodeProto],
    weights: &mut WeightStore,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    graph_output_names: &HashSet<String>,
    bn_idx: usize,
) -> bool {
    let Some(bn_input) = nodes[bn_idx]
        .input
        .first()
        .filter(|name| !name.is_empty())
        .cloned()
    else {
        return false;
    };
    let Some(bn_output) = inference_batch_norm_output(&nodes[bn_idx]) else {
        return false;
    };
    // Unlike predecessor folds, this rewrite bypasses Y rather than moving its
    // name to an equivalent producer. If Y is a graph output, it must remain
    // observable and the fold cannot fire.
    if graph_output_names.contains(&bn_output) {
        return false;
    }
    let bn_consumers = consumer_indices(nodes, &bn_output);
    let [reshape_idx] = bn_consumers.as_slice() else {
        return false;
    };
    let reshape_idx = *reshape_idx;
    let reshape = &nodes[reshape_idx];
    if reshape.op_type != "Reshape"
        || reshape.input.len() < 2
        || reshape.input.first() != Some(&bn_output)
    {
        return false;
    }
    let Some(reshape_output) = reshape
        .output
        .first()
        .filter(|name| !name.is_empty())
        .cloned()
    else {
        return false;
    };
    // This name survives the rewrite but its value changes from flattened BN Y
    // to flattened raw BN input. Only the final Gemm output is preserved.
    if graph_output_names.contains(&reshape_output) {
        return false;
    }
    let reshape_consumers = consumer_indices(nodes, &reshape_output);
    let [gemm_idx] = reshape_consumers.as_slice() else {
        return false;
    };
    let gemm_idx = *gemm_idx;
    let gemm = &nodes[gemm_idx];
    if gemm.op_type != "Gemm"
        || gemm.input.first() != Some(&reshape_output)
        || gemm_trans_a(gemm)
        || !gemm_has_exact_default_affine(gemm)
    {
        return false;
    }

    let Some(target_shape) = read_tensor_i64s(weights, &reshape.input[1]) else {
        return false;
    };
    let Some(affine) = batch_norm_affine(&nodes[bn_idx], weights) else {
        return false;
    };
    let Some(weight_name) = gemm.input.get(1).filter(|name| !name.is_empty()).cloned() else {
        return false;
    };
    if graph_output_names.contains(&weight_name) {
        return false;
    }
    if consumer_indices(nodes, &weight_name).as_slice() != [gemm_idx] {
        return false;
    }
    let Some(weight) = weights.get(&weight_name).cloned() else {
        return false;
    };
    if weight.ndim() != 2 || affine.scale.is_empty() {
        return false;
    }
    let trans_b = gemm_trans_b(gemm);
    let (features, outputs) = if trans_b {
        (weight.shape()[1], weight.shape()[0])
    } else {
        (weight.shape()[0], weight.shape()[1])
    };
    if features == 0 || features % affine.scale.len() != 0 {
        return false;
    }
    let Ok(channels_i64) = i64::try_from(affine.scale.len()) else {
        return false;
    };
    let Ok(features_i64) = i64::try_from(features) else {
        return false;
    };
    // Cover the two exporter forms in the official cGAN pool, but only after
    // proving the BN source is exact NCHW/channel-major storage with exactly F
    // elements per batch item. Shape syntax alone is insufficient: e.g.
    // `[1,2,2] -> [-1,2]` changes the Gemm row partition and cannot use a
    // feature-wise channel scale.
    let Some(source_shape) = tensor_shapes.get(&bn_input) else {
        return false;
    };
    if source_shape.len() < 2
        || source_shape.iter().any(|dim| *dim <= 0)
        || source_shape[1] != channels_i64
    {
        return false;
    }
    let Some(non_batch_elements) = source_shape[1..]
        .iter()
        .try_fold(1_i64, |product, dim| product.checked_mul(*dim))
    else {
        return false;
    };
    let safe_target = target_shape.as_slice() == [-1, features_i64]
        || (target_shape.as_slice() == [1, -1] && source_shape[0] == 1);
    if !safe_target || non_batch_elements != features_i64 {
        return false;
    }
    let block = features / affine.scale.len();

    let existing_bias_name = gemm.input.get(2).filter(|name| !name.is_empty()).cloned();
    if existing_bias_name.as_deref() == Some(weight_name.as_str()) {
        return false;
    }
    if existing_bias_name
        .as_ref()
        .is_some_and(|name| graph_output_names.contains(name))
    {
        return false;
    }
    if let Some(name) = existing_bias_name.as_deref() {
        if consumer_indices(nodes, name).as_slice() != [gemm_idx] {
            return false;
        }
    }
    let existing_bias = match existing_bias_name.as_ref() {
        Some(name) => {
            let Some(bias) = weights.get(name).cloned() else {
                // A dynamic Gemm C input cannot be replaced by a synthesized
                // constant without changing the model.
                return false;
            };
            let Some(normalized) = normalize_gemm_c(&bias, outputs) else {
                return false;
            };
            Some(normalized)
        }
        None => None,
    };
    let feature_axis = if trans_b { 1 } else { 0 };
    let mut fused_weight = weight.clone();
    for (feature, mut slice) in fused_weight.axis_iter_mut(Axis(feature_axis)).enumerate() {
        slice *= affine.scale[feature / block];
    }
    let mut fused_bias = Array1::<f32>::zeros(outputs);
    for output in 0..outputs {
        let mut value = existing_bias
            .as_ref()
            .map_or(0.0, |bias| bias.iter().nth(output).copied().unwrap_or(0.0));
        for feature in 0..features {
            let coefficient = if trans_b {
                weight[[output, feature]]
            } else {
                weight[[feature, output]]
            };
            value += coefficient * affine.shift[feature / block];
        }
        if !value.is_finite() {
            return false;
        }
        fused_bias[output] = value;
    }
    if fused_weight.iter().any(|value| !value.is_finite()) {
        return false;
    }

    let bias_name = match existing_bias_name {
        Some(name) => name,
        None => {
            let Some(name) = fresh_synthetic_bias_name(
                nodes,
                weights,
                tensor_shapes,
                graph_output_names,
                &nodes[gemm_idx],
                bn_idx,
            ) else {
                return false;
            };
            name
        }
    };
    weights.insert(weight_name, fused_weight);
    weights.insert(bias_name.clone(), fused_bias.into_dyn());
    if nodes[gemm_idx].input.len() >= 3 {
        nodes[gemm_idx].input[2] = bias_name;
    } else {
        nodes[gemm_idx].input.push(bias_name);
    }
    nodes[reshape_idx].input[0] = bn_input;
    nodes[bn_idx].input.clear();
    nodes[bn_idx].output.clear();
    debug!(
        "Fused BatchNormalization node {} + Reshape node {} into Gemm node {} (C={}, block={})",
        bn_idx,
        reshape_idx,
        gemm_idx,
        affine.scale.len(),
        block
    );
    true
}

/// Scale Gemm weight feature `f` by `scale[f / block]` (#cgan-structural-fold).
///
/// - `trans_b=true`: weight is `(F, in)` — features are rows (axis 0)
/// - `trans_b=false`: weight is `(in, F)` — features are columns (axis 1)
fn fuse_gemm_reshape_weight(
    weight: &ArrayD<f32>,
    scale: &[f32],
    trans_b: bool,
    block: usize,
) -> Option<ArrayD<f32>> {
    if weight.ndim() != 2 || block == 0 {
        return None;
    }
    let feature_axis = if trans_b { 0 } else { 1 };
    if weight.shape()[feature_axis] != scale.len() * block {
        return None;
    }

    let mut fused = weight.clone();
    for (feature_idx, mut slice) in fused.axis_iter_mut(Axis(feature_axis)).enumerate() {
        slice *= scale[feature_idx / block];
    }
    Some(fused)
}

/// Per-feature fused bias for the Gemm->Reshape->BN fold:
/// `b'[f] = b[f] * scale[f/block] + shift[f/block]` (zero `b` when absent).
/// Unlike the direct fold, the fused bias has length `F = C * block` — the BN
/// channel shift is replicated across each channel's `block` features.
fn fuse_gemm_reshape_bias(
    existing_bias: Option<&ArrayD<f32>>,
    scale: &[f32],
    shift: &[f32],
    block: usize,
    features: usize,
) -> Option<ArrayD<f32>> {
    if scale.len() != shift.len() || block == 0 || scale.len() * block != features {
        return None;
    }

    let mut fused = Array1::<f32>::zeros(features);
    if let Some(bias) = existing_bias {
        if bias.len() != features {
            return None;
        }
        for (idx, value) in bias.iter().enumerate() {
            fused[idx] = *value * scale[idx / block] + shift[idx / block];
        }
    } else {
        for idx in 0..features {
            fused[idx] = shift[idx / block];
        }
    }
    Some(fused.into_dyn())
}

fn fuse_bias(
    existing_bias: Option<&ArrayD<f32>>,
    scale: &[f32],
    shift: &[f32],
) -> Option<ArrayD<f32>> {
    if scale.len() != shift.len() {
        return None;
    }

    let mut fused = Array1::<f32>::zeros(scale.len());
    if let Some(bias) = existing_bias {
        if bias.len() != scale.len() {
            return None;
        }
        for (idx, value) in bias.iter().enumerate() {
            fused[idx] = *value * scale[idx] + shift[idx];
        }
    } else {
        for idx in 0..scale.len() {
            fused[idx] = shift[idx];
        }
    }
    Some(fused.into_dyn())
}

/// Normalize legal ONNX Gemm C broadcasts to the output-feature vector used by
/// the fusion algebra. Scalars (`[]`, `[1]`, `[1,1]`), `[N]`, and `[1,N]` are
/// independent of the runtime row count M. In particular `[M,1]` for M > 1
/// cannot be folded into one bias vector because its value varies by row.
fn normalize_gemm_c(bias: &ArrayD<f32>, outputs: usize) -> Option<ArrayD<f32>> {
    if outputs == 0 {
        return None;
    }
    let values = match bias.shape() {
        [] if bias.len() == 1 => vec![*bias.iter().next()?; outputs],
        [one] if *one == 1 => vec![*bias.iter().next()?; outputs],
        [n] if *n == outputs => bias.iter().copied().collect(),
        [one_a, one_b] if *one_a == 1 && *one_b == 1 => {
            vec![*bias.iter().next()?; outputs]
        }
        [one, n] if *one == 1 && *n == outputs => bias.iter().copied().collect(),
        _ => return None,
    };
    Some(Array1::from_vec(values).into_dyn())
}

fn synthetic_bias_base(predecessor: &onnx_proto::NodeProto, bn_idx: usize) -> String {
    if let Some(output) = predecessor.output.first() {
        if !output.is_empty() {
            return format!("{output}__bn_fused_bias");
        }
    }
    format!("bn_fused_bias_{bn_idx}")
}

/// Allocate a deterministic tensor name that is fresh across every value name
/// visible to this pass and both typed stores. This is computed before any
/// mutation, preserving the fold's validate-then-commit boundary.
fn fresh_synthetic_bias_name(
    nodes: &[onnx_proto::NodeProto],
    weights: &WeightStore,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    graph_output_names: &HashSet<String>,
    predecessor: &onnx_proto::NodeProto,
    bn_idx: usize,
) -> Option<String> {
    let base = synthetic_bias_base(predecessor, bn_idx);
    let in_use = |candidate: &str| {
        weights.contains_key(candidate)
            || tensor_shapes.contains_key(candidate)
            || graph_output_names.contains(candidate)
            || nodes.iter().any(|node| {
                node.input
                    .iter()
                    .chain(&node.output)
                    .any(|name| name == candidate)
            })
    };
    if !in_use(&base) {
        return Some(base);
    }
    let mut suffix = 1_usize;
    loop {
        let candidate = format!("{base}__{suffix}");
        if !in_use(&candidate) {
            return Some(candidate);
        }
        suffix = suffix.checked_add(1)?;
    }
}
