// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_propagate::layers::LayerNormMode;

use super::norm_decompose::{try_rewrite_instance_norm, try_rewrite_rms_norm};
use crate::graph_options::CompoundNodePolicy;
use crate::layernorm_mode_from_attrs;
use crate::{AttributeValue, LayerSpec, WeightStore};

pub(super) const COMPOUND_GENERATED_ATTR: &str = "__compound_generated";
const GENERATED_LAYERNORM_MARKER: &str = "layernorm";
pub(super) const GENERATED_RMSNORM_MARKER: &str = "rmsnorm";
pub(super) const GENERATED_INSTANCENORM_MARKER: &str = "instancenorm";

#[derive(Debug, Clone)]
pub(super) struct CompoundRewriteResult {
    pub(super) layers: Vec<LayerSpec>,
    pub(super) weights: WeightStore,
    pub(super) tensor_shapes: HashMap<String, Vec<i64>>,
}

pub(super) fn rewrite_compound_nodes(
    layers: &[LayerSpec],
    weights: &WeightStore,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    policy: CompoundNodePolicy,
) -> Option<CompoundRewriteResult> {
    if matches!(policy, CompoundNodePolicy::Preserve) {
        return None;
    }

    let mut rewritten_layers = Vec::with_capacity(layers.len());
    let mut rewritten_weights = weights.clone();
    let mut rewritten_shapes = tensor_shapes.clone();
    let mut reserved_layer_names: HashSet<String> =
        layers.iter().map(|spec| spec.name.clone()).collect();
    let mut reserved_tensor_names: HashSet<String> = tensor_shapes.keys().cloned().collect();
    for spec in layers {
        reserved_tensor_names.extend(spec.inputs.iter().cloned());
        reserved_tensor_names.extend(spec.outputs.iter().cloned());
    }

    for spec in layers {
        if let Some(rewritten) = try_rewrite_standard_layernorm(
            spec,
            &mut rewritten_weights,
            &mut rewritten_shapes,
            &mut reserved_layer_names,
            &mut reserved_tensor_names,
        ) {
            rewritten_layers.extend(rewritten);
        } else if let Some(rewritten) = try_rewrite_rms_norm(
            spec,
            &mut rewritten_weights,
            &mut rewritten_shapes,
            &mut reserved_layer_names,
            &mut reserved_tensor_names,
        ) {
            rewritten_layers.extend(rewritten);
        } else if let Some(rewritten) = try_rewrite_instance_norm(
            spec,
            &mut rewritten_weights,
            &mut rewritten_shapes,
            &mut reserved_layer_names,
            &mut reserved_tensor_names,
        ) {
            rewritten_layers.extend(rewritten);
        } else {
            rewritten_layers.push(spec.clone());
        }
    }

    Some(CompoundRewriteResult {
        layers: rewritten_layers,
        weights: rewritten_weights,
        tensor_shapes: rewritten_shapes,
    })
}

pub(super) fn is_compound_generated(spec: &LayerSpec) -> bool {
    matches!(
        spec.attributes.get(COMPOUND_GENERATED_ATTR),
        Some(AttributeValue::String(kind))
            if kind == GENERATED_LAYERNORM_MARKER
            || kind == GENERATED_RMSNORM_MARKER
            || kind == GENERATED_INSTANCENORM_MARKER
    )
}

fn try_rewrite_standard_layernorm(
    spec: &LayerSpec,
    weights: &mut WeightStore,
    tensor_shapes: &mut HashMap<String, Vec<i64>>,
    reserved_layer_names: &mut HashSet<String>,
    reserved_tensor_names: &mut HashSet<String>,
) -> Option<Vec<LayerSpec>> {
    if spec.layer_type != LayerType::LayerNorm || spec.inputs.len() < 3 || spec.outputs.len() != 1 {
        return None;
    }
    if layernorm_mode_from_attrs(spec) != LayerNormMode::Standard {
        return None;
    }

    let ny_name = spec.inputs.get(1)?;
    let beta_name = spec.inputs.get(2)?;
    if weights.get(ny_name)?.ndim() != 1 || weights.get(beta_name)?.ndim() != 1 {
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

    let mean = generated_node(
        &spec.name,
        "mean",
        LayerType::ReduceMean,
        vec![input_name.clone()],
        reduce_mean_attributes((input_shape.len() - 1) as i64),
        reduced_shape.clone(),
        &mut state,
    );
    let centered = generated_node(
        &spec.name,
        "centered",
        LayerType::Sub,
        vec![input_name, mean.output.clone()],
        generated_attributes(),
        input_shape.clone(),
        &mut state,
    );
    let square = generated_node(
        &spec.name,
        "square",
        LayerType::Mul,
        vec![centered.output.clone(), centered.output.clone()],
        generated_attributes(),
        input_shape.clone(),
        &mut state,
    );
    let variance = generated_node(
        &spec.name,
        "variance",
        LayerType::ReduceMean,
        vec![square.output.clone()],
        reduce_mean_attributes((input_shape.len() - 1) as i64),
        reduced_shape.clone(),
        &mut state,
    );
    let var_eps = generated_node(
        &spec.name,
        "var_eps",
        LayerType::Add,
        vec![variance.output.clone(), epsilon_name],
        generated_attributes(),
        reduced_shape.clone(),
        &mut state,
    );
    let std = generated_node(
        &spec.name,
        "std",
        LayerType::Sqrt,
        vec![var_eps.output.clone()],
        generated_attributes(),
        reduced_shape.clone(),
        &mut state,
    );
    let inv_std = generated_node(
        &spec.name,
        "inv_std",
        LayerType::Reciprocal,
        vec![std.output.clone()],
        generated_attributes(),
        reduced_shape,
        &mut state,
    );
    let normalized = generated_node(
        &spec.name,
        "normalized",
        LayerType::Mul,
        vec![centered.output.clone(), inv_std.output.clone()],
        generated_attributes(),
        input_shape.clone(),
        &mut state,
    );
    let scaled = generated_node(
        &spec.name,
        "scaled",
        LayerType::Mul,
        vec![normalized.output.clone(), ny_name.clone()],
        generated_attributes(),
        input_shape.clone(),
        &mut state,
    );

    state
        .tensor_shapes
        .insert(spec.outputs[0].clone(), input_shape);
    let final_add = LayerSpec {
        name: spec.name.clone(),
        layer_type: LayerType::Add,
        inputs: vec![scaled.output, beta_name.clone()],
        outputs: spec.outputs.clone(),
        weights: None,
        attributes: generated_attributes(),
    };

    Some(vec![
        mean.spec,
        centered.spec,
        square.spec,
        variance.spec,
        var_eps.spec,
        std.spec,
        inv_std.spec,
        normalized.spec,
        scaled.spec,
        final_add,
    ])
}

// --- Shared types and helpers used by norm_decompose.rs ---

#[derive(Debug)]
pub(super) struct GeneratedNodeSpec {
    pub(super) spec: LayerSpec,
    pub(super) output: String,
}

pub(super) struct RewriteState<'a> {
    pub(super) tensor_shapes: &'a mut HashMap<String, Vec<i64>>,
    pub(super) reserved_layer_names: &'a mut HashSet<String>,
    pub(super) reserved_tensor_names: &'a mut HashSet<String>,
    pub(super) weights: &'a WeightStore,
}

fn generated_node(
    source_name: &str,
    suffix: &str,
    layer_type: LayerType,
    inputs: Vec<String>,
    attributes: HashMap<String, AttributeValue>,
    output_shape: Vec<i64>,
    state: &mut RewriteState<'_>,
) -> GeneratedNodeSpec {
    let name = unique_layer_name(
        &format!("{}__{}", source_name, suffix),
        state.reserved_layer_names,
    );
    let output = unique_tensor_name(
        &format!("{}__{}_out", source_name, suffix),
        state.reserved_tensor_names,
        state.weights,
    );
    state.tensor_shapes.insert(output.clone(), output_shape);
    GeneratedNodeSpec {
        spec: LayerSpec {
            name,
            layer_type,
            inputs,
            outputs: vec![output.clone()],
            weights: None,
            attributes,
        },
        output,
    }
}

fn generated_attributes() -> HashMap<String, AttributeValue> {
    generated_attributes_for(GENERATED_LAYERNORM_MARKER)
}

pub(super) fn generated_attributes_for(marker: &str) -> HashMap<String, AttributeValue> {
    HashMap::from([(
        COMPOUND_GENERATED_ATTR.to_string(),
        AttributeValue::String(marker.to_string()),
    )])
}

pub(super) fn generated_node_with_marker(
    source_name: &str,
    suffix: &str,
    layer_type: LayerType,
    inputs: Vec<String>,
    marker: &str,
    output_shape: Vec<i64>,
    state: &mut RewriteState<'_>,
) -> GeneratedNodeSpec {
    let name = unique_layer_name(
        &format!("{}__{}", source_name, suffix),
        state.reserved_layer_names,
    );
    let output = unique_tensor_name(
        &format!("{}__{}_out", source_name, suffix),
        state.reserved_tensor_names,
        state.weights,
    );
    state.tensor_shapes.insert(output.clone(), output_shape);
    GeneratedNodeSpec {
        spec: LayerSpec {
            name,
            layer_type,
            inputs,
            outputs: vec![output.clone()],
            weights: None,
            attributes: generated_attributes_for(marker),
        },
        output,
    }
}

pub(super) fn reduce_mean_attributes(axis: i64) -> HashMap<String, AttributeValue> {
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), AttributeValue::Ints(vec![axis]));
    attributes.insert("keepdims".to_string(), AttributeValue::Int(1));
    attributes
}

pub(super) fn reduce_mean_multi_axes(axes: &[i64]) -> HashMap<String, AttributeValue> {
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), AttributeValue::Ints(axes.to_vec()));
    attributes.insert("keepdims".to_string(), AttributeValue::Int(1));
    attributes
}

pub(super) fn reduce_keepdims_last_axis(input_shape: &[i64]) -> Option<Vec<i64>> {
    let mut reduced = input_shape.to_vec();
    let last_dim = reduced.last_mut()?;
    *last_dim = 1;
    Some(reduced)
}

pub(super) fn read_positive_epsilon(spec: &LayerSpec) -> Option<f32> {
    match spec.attributes.get("epsilon") {
        None => Some(1e-5),
        Some(AttributeValue::Float(value)) if value.is_finite() && *value > 0.0 => Some(*value),
        _ => None,
    }
}

fn unique_layer_name(base: &str, reserved_layer_names: &mut HashSet<String>) -> String {
    let mut candidate = base.to_string();
    let mut suffix = 0usize;
    while reserved_layer_names.contains(&candidate) {
        suffix += 1;
        candidate = format!("{base}_{suffix}");
    }
    reserved_layer_names.insert(candidate.clone());
    candidate
}

pub(super) fn unique_tensor_name(
    base: &str,
    reserved_tensor_names: &mut HashSet<String>,
    weights: &WeightStore,
) -> String {
    let mut candidate = base.to_string();
    let mut suffix = 0usize;
    while reserved_tensor_names.contains(&candidate) || weights.contains_key(&candidate) {
        suffix += 1;
        candidate = format!("{base}_{suffix}");
    }
    reserved_tensor_names.insert(candidate.clone());
    candidate
}
