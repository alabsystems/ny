// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-level decomposition of RmsNorm and InstanceNorm into primitive ops.
//!
//! Companion to `compound_nodes.rs` which handles LayerNorm. Split out to
//! keep each file under 500 lines (#4399).

use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;

use super::compound_nodes::{
    generated_attributes_for, generated_node_with_marker, read_positive_epsilon,
    reduce_keepdims_last_axis, reduce_mean_attributes, reduce_mean_multi_axes, unique_tensor_name,
    RewriteState, GENERATED_INSTANCENORM_MARKER, GENERATED_RMSNORM_MARKER,
};
use crate::{LayerSpec, WeightStore};

/// Decompose RmsNorm into: x^2 -> mean(x^2) -> +eps -> sqrt -> reciprocal -> x*inv_std -> *ny
/// RmsNorm = x * ny / sqrt(mean(x^2) + eps)
pub(super) fn try_rewrite_rms_norm(
    spec: &LayerSpec,
    weights: &mut WeightStore,
    tensor_shapes: &mut HashMap<String, Vec<i64>>,
    reserved_layer_names: &mut HashSet<String>,
    reserved_tensor_names: &mut HashSet<String>,
) -> Option<Vec<LayerSpec>> {
    if spec.layer_type != LayerType::RMSNorm || spec.inputs.is_empty() || spec.outputs.len() != 1 {
        return None;
    }

    let ny_name = spec.inputs.get(1)?;
    if weights.get(ny_name)?.ndim() != 1 {
        return None;
    }

    let input_name = spec.inputs.first()?.clone();
    let input_shape = tensor_shapes.get(&input_name)?.clone();
    let reduced_shape = reduce_keepdims_last_axis(&input_shape)?;
    let epsilon = read_positive_epsilon(spec)?;

    let epsilon_name = unique_tensor_name(
        &format!("{}__epsilon", spec.name),
        reserved_tensor_names,
        weights,
    );
    weights.insert(epsilon_name.clone(), ArrayD::from_elem(IxDyn(&[]), epsilon));
    tensor_shapes.insert(epsilon_name.clone(), Vec::new());

    let mut state = RewriteState {
        tensor_shapes,
        reserved_layer_names,
        reserved_tensor_names,
        weights,
    };

    // x^2
    let square = generated_node_with_marker(
        &spec.name,
        "square",
        LayerType::Mul,
        vec![input_name.clone(), input_name.clone()],
        GENERATED_RMSNORM_MARKER,
        input_shape.clone(),
        &mut state,
    );
    // mean(x^2) over last axis
    let mean_sq = generated_node_with_marker(
        &spec.name,
        "mean_sq",
        LayerType::ReduceMean,
        vec![square.output.clone()],
        GENERATED_RMSNORM_MARKER,
        reduced_shape.clone(),
        &mut state,
    );
    // mean(x^2) + eps
    let var_eps = generated_node_with_marker(
        &spec.name,
        "var_eps",
        LayerType::Add,
        vec![mean_sq.output.clone(), epsilon_name],
        GENERATED_RMSNORM_MARKER,
        reduced_shape.clone(),
        &mut state,
    );
    // sqrt(mean(x^2) + eps)
    let rms = generated_node_with_marker(
        &spec.name,
        "rms",
        LayerType::Sqrt,
        vec![var_eps.output.clone()],
        GENERATED_RMSNORM_MARKER,
        reduced_shape.clone(),
        &mut state,
    );
    // 1 / sqrt(...)
    let inv_rms = generated_node_with_marker(
        &spec.name,
        "inv_rms",
        LayerType::Reciprocal,
        vec![rms.output.clone()],
        GENERATED_RMSNORM_MARKER,
        reduced_shape,
        &mut state,
    );
    // x * inv_rms (broadcasts reduced_shape over input_shape)
    let normalized = generated_node_with_marker(
        &spec.name,
        "normalized",
        LayerType::Mul,
        vec![input_name, inv_rms.output.clone()],
        GENERATED_RMSNORM_MARKER,
        input_shape.clone(),
        &mut state,
    );

    // ny * normalized — final node reuses the original LayerSpec name/output
    let last_axis = (input_shape.len() - 1) as i64;
    state
        .tensor_shapes
        .insert(spec.outputs[0].clone(), input_shape);
    let final_scale = LayerSpec {
        name: spec.name.clone(),
        layer_type: LayerType::Mul,
        inputs: vec![normalized.output, ny_name.clone()],
        outputs: spec.outputs.clone(),
        weights: None,
        attributes: generated_attributes_for(GENERATED_RMSNORM_MARKER),
    };

    // mean_sq needs reduce axes attribute
    let mut mean_sq_spec = mean_sq.spec;
    mean_sq_spec
        .attributes
        .extend(reduce_mean_attributes(last_axis));

    Some(vec![
        square.spec,
        mean_sq_spec,
        var_eps.spec,
        rms.spec,
        inv_rms.spec,
        normalized.spec,
        final_scale,
    ])
}

/// Decompose InstanceNorm into per-channel normalization over spatial dims.
/// InstanceNorm = ny * (x - mean(x)) / sqrt(var(x) + eps) + beta
/// where mean/var are computed per (batch, channel) over spatial dims [2..].
pub(super) fn try_rewrite_instance_norm(
    spec: &LayerSpec,
    weights: &mut WeightStore,
    tensor_shapes: &mut HashMap<String, Vec<i64>>,
    reserved_layer_names: &mut HashSet<String>,
    reserved_tensor_names: &mut HashSet<String>,
) -> Option<Vec<LayerSpec>> {
    if spec.layer_type != LayerType::InstanceNorm
        || spec.inputs.len() < 3
        || spec.outputs.len() != 1
    {
        return None;
    }

    let ny_name = spec.inputs.get(1)?;
    let beta_name = spec.inputs.get(2)?;

    let input_name = spec.inputs.first()?.clone();
    let input_shape = tensor_shapes.get(&input_name)?.clone();
    // InstanceNorm requires at least 3D [B, C, spatial...]
    if input_shape.len() < 3 {
        return None;
    }
    let num_channels = usize::try_from(*input_shape.get(1)?).ok()?;
    if num_channels == 0 {
        return None;
    }

    // ONNX stores InstanceNormalization affine parameters as `[C]`, but the
    // graph builder strips the leading batch dimension from activations.  A
    // raw `[N,C,H,W]` activation therefore becomes internal `[C,H,W]`; using
    // `[C]` directly would right-align it with W (and silently scale width when
    // `W == C`).  Materialize exact-bit reshaped copies `[C,1,...]` at the
    // internal activation rank so generic elementwise broadcasting applies
    // parameters by channel for every supported spatial rank.
    let ny = weights.get(ny_name)?;
    let beta = weights.get(beta_name)?;
    if ny.shape() != [num_channels] || beta.shape() != [num_channels] {
        return None;
    }
    let mut affine_shape = vec![num_channels];
    affine_shape.resize(input_shape.len() - 1, 1);
    let reshaped_ny =
        ArrayD::from_shape_vec(IxDyn(&affine_shape), ny.iter().copied().collect()).ok()?;
    let reshaped_beta =
        ArrayD::from_shape_vec(IxDyn(&affine_shape), beta.iter().copied().collect()).ok()?;
    let reshaped_ny_name = unique_tensor_name(
        &format!("{}__channel_scale", spec.name),
        reserved_tensor_names,
        weights,
    );
    let reshaped_beta_name = unique_tensor_name(
        &format!("{}__channel_bias", spec.name),
        reserved_tensor_names,
        weights,
    );
    weights.insert(reshaped_ny_name.clone(), reshaped_ny);
    weights.insert(reshaped_beta_name.clone(), reshaped_beta);
    let affine_shape_i64 = affine_shape
        .iter()
        .map(|&dimension| i64::try_from(dimension).ok())
        .collect::<Option<Vec<_>>>()?;
    tensor_shapes.insert(reshaped_ny_name.clone(), affine_shape_i64.clone());
    tensor_shapes.insert(reshaped_beta_name.clone(), affine_shape_i64);
    // Reduced shape: keep batch and channel dims, reduce spatial to 1
    let reduced_shape = reduce_keepdims_spatial(&input_shape)?;
    let spatial_axes: Vec<i64> = (2..input_shape.len() as i64).collect();
    let epsilon = read_positive_epsilon(spec)?;

    let epsilon_name = unique_tensor_name(
        &format!("{}__epsilon", spec.name),
        reserved_tensor_names,
        weights,
    );
    weights.insert(epsilon_name.clone(), ArrayD::from_elem(IxDyn(&[]), epsilon));
    tensor_shapes.insert(epsilon_name.clone(), Vec::new());

    let mut state = RewriteState {
        tensor_shapes,
        reserved_layer_names,
        reserved_tensor_names,
        weights,
    };

    // mean(x) over spatial axes
    let mean = generated_node_with_marker(
        &spec.name,
        "mean",
        LayerType::ReduceMean,
        vec![input_name.clone()],
        GENERATED_INSTANCENORM_MARKER,
        reduced_shape.clone(),
        &mut state,
    );
    // x - mean(x)
    let centered = generated_node_with_marker(
        &spec.name,
        "centered",
        LayerType::Sub,
        vec![input_name, mean.output.clone()],
        GENERATED_INSTANCENORM_MARKER,
        input_shape.clone(),
        &mut state,
    );
    // (x - mean)^2
    let square = generated_node_with_marker(
        &spec.name,
        "square",
        LayerType::Mul,
        vec![centered.output.clone(), centered.output.clone()],
        GENERATED_INSTANCENORM_MARKER,
        input_shape.clone(),
        &mut state,
    );
    // var = mean((x - mean)^2) over spatial axes
    let variance = generated_node_with_marker(
        &spec.name,
        "variance",
        LayerType::ReduceMean,
        vec![square.output.clone()],
        GENERATED_INSTANCENORM_MARKER,
        reduced_shape.clone(),
        &mut state,
    );
    // var + eps
    let var_eps = generated_node_with_marker(
        &spec.name,
        "var_eps",
        LayerType::Add,
        vec![variance.output.clone(), epsilon_name],
        GENERATED_INSTANCENORM_MARKER,
        reduced_shape.clone(),
        &mut state,
    );
    // sqrt(var + eps)
    let std_dev = generated_node_with_marker(
        &spec.name,
        "std",
        LayerType::Sqrt,
        vec![var_eps.output.clone()],
        GENERATED_INSTANCENORM_MARKER,
        reduced_shape.clone(),
        &mut state,
    );
    // 1 / sqrt(var + eps)
    let inv_std = generated_node_with_marker(
        &spec.name,
        "inv_std",
        LayerType::Reciprocal,
        vec![std_dev.output.clone()],
        GENERATED_INSTANCENORM_MARKER,
        reduced_shape,
        &mut state,
    );
    // (x - mean) * inv_std
    let normalized = generated_node_with_marker(
        &spec.name,
        "normalized",
        LayerType::Mul,
        vec![centered.output.clone(), inv_std.output.clone()],
        GENERATED_INSTANCENORM_MARKER,
        input_shape.clone(),
        &mut state,
    );
    // ny * normalized
    let scaled = generated_node_with_marker(
        &spec.name,
        "scaled",
        LayerType::Mul,
        vec![normalized.output.clone(), reshaped_ny_name],
        GENERATED_INSTANCENORM_MARKER,
        input_shape.clone(),
        &mut state,
    );

    // ny * normalized + beta — final node reuses the original name/output
    state
        .tensor_shapes
        .insert(spec.outputs[0].clone(), input_shape);
    let final_add = LayerSpec {
        name: spec.name.clone(),
        layer_type: LayerType::Add,
        inputs: vec![scaled.output, reshaped_beta_name],
        outputs: spec.outputs.clone(),
        weights: None,
        attributes: generated_attributes_for(GENERATED_INSTANCENORM_MARKER),
    };

    // Patch ReduceMean specs with the correct spatial axes
    let mut mean_spec = mean.spec;
    mean_spec
        .attributes
        .extend(reduce_mean_multi_axes(&spatial_axes));
    let mut variance_spec = variance.spec;
    variance_spec
        .attributes
        .extend(reduce_mean_multi_axes(&spatial_axes));

    Some(vec![
        mean_spec,
        centered.spec,
        square.spec,
        variance_spec,
        var_eps.spec,
        std_dev.spec,
        inv_std.spec,
        normalized.spec,
        scaled.spec,
        final_add,
    ])
}

/// Reduce spatial dimensions (axes 2+) to 1, keeping batch and channel dims.
fn reduce_keepdims_spatial(input_shape: &[i64]) -> Option<Vec<i64>> {
    if input_shape.len() < 3 {
        return None;
    }
    let mut reduced = input_shape.to_vec();
    for dim in reduced.iter_mut().skip(2) {
        *dim = 1;
    }
    Some(reduced)
}
