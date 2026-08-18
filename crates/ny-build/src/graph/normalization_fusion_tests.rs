// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_tensor::BoundedTensor;

use super::super::builder::{build_graph_network, GraphBuildInputs};
use super::*;
use crate::{AttributeValue, DataType, TensorSpec, WeightStore};

fn tensor_spec(name: &str, shape: &[i64]) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: DataType::Float32,
    }
}

fn layer_spec(
    name: &str,
    layer_type: LayerType,
    inputs: &[&str],
    outputs: &[&str],
    attributes: HashMap<String, AttributeValue>,
) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: inputs.iter().copied().map(str::to_owned).collect(),
        outputs: outputs.iter().copied().map(str::to_owned).collect(),
        weights: None,
        attributes,
    }
}

fn reduce_attrs(axes: i64) -> HashMap<String, AttributeValue> {
    HashMap::from([
        ("axes".to_string(), AttributeValue::Ints(vec![axes])),
        ("keepdims".to_string(), AttributeValue::Int(1)),
    ])
}

/// Shared variance chain: ReduceMean(x) → Sub → Mul(self) → ReduceMean → Add(eps) → Sqrt.
/// Returns layers for the shared centering + variance computation prefix.
fn variance_chain_layers(axes: i64) -> Vec<LayerSpec> {
    let attrs = reduce_attrs(axes);
    vec![
        layer_spec(
            "mean_a",
            LayerType::ReduceMean,
            &["input"],
            &["mean_a_out"],
            attrs.clone(),
        ),
        layer_spec(
            "mean_b",
            LayerType::ReduceMean,
            &["input"],
            &["mean_b_out"],
            attrs.clone(),
        ),
        layer_spec(
            "center_var",
            LayerType::Sub,
            &["input", "mean_b_out"],
            &["center_var_out"],
            HashMap::new(),
        ),
        layer_spec(
            "square",
            LayerType::Mul,
            &["center_var_out", "center_var_out"],
            &["square_out"],
            HashMap::new(),
        ),
        layer_spec(
            "var_mean",
            LayerType::ReduceMean,
            &["square_out"],
            &["var_mean_out"],
            attrs,
        ),
        layer_spec(
            "var_eps",
            LayerType::Add,
            &["var_mean_out", "eps"],
            &["var_eps_out"],
            HashMap::new(),
        ),
        layer_spec(
            "std",
            LayerType::Sqrt,
            &["var_eps_out"],
            &["std_out"],
            HashMap::new(),
        ),
        layer_spec(
            "center_norm",
            LayerType::Sub,
            &["input", "mean_a_out"],
            &["center_norm_out"],
            HashMap::new(),
        ),
    ]
}

fn decomposed_instance_norm_layers(axes: i64) -> Vec<LayerSpec> {
    let mut layers = variance_chain_layers(axes);
    layers.push(layer_spec(
        "norm_div",
        LayerType::Div,
        &["center_norm_out", "std_out"],
        &["norm_out"],
        HashMap::new(),
    ));
    layers
}

fn decomposed_instance_norm_reciprocal_mul_layers(axes: i64) -> Vec<LayerSpec> {
    let mut layers = variance_chain_layers(axes);
    layers.push(layer_spec(
        "inv_std",
        LayerType::Reciprocal,
        &["std_out"],
        &["inv_std_out"],
        HashMap::new(),
    ));
    layers.push(layer_spec(
        "norm_mul",
        LayerType::Mul,
        &["center_norm_out", "inv_std_out"],
        &["norm_out"],
        HashMap::new(),
    ));
    layers
}

fn generated_layernorm_reciprocal_mul_layers(axes: i64) -> Vec<LayerSpec> {
    let mut layers = decomposed_instance_norm_reciprocal_mul_layers(axes);
    for layer in &mut layers {
        layer.attributes.insert(
            "__compound_generated".to_string(),
            AttributeValue::String("layernorm".to_string()),
        );
    }
    layers
}

fn base_tensor_shapes() -> HashMap<String, Vec<i64>> {
    HashMap::from([
        ("input".to_string(), vec![1, 2, 4]),
        ("mean_a_out".to_string(), vec![1, 2, 1]),
        ("mean_b_out".to_string(), vec![1, 2, 1]),
        ("center_var_out".to_string(), vec![1, 2, 4]),
        ("square_out".to_string(), vec![1, 2, 4]),
        ("var_mean_out".to_string(), vec![1, 2, 1]),
        ("var_eps_out".to_string(), vec![1, 2, 1]),
        ("std_out".to_string(), vec![1, 2, 1]),
        ("center_norm_out".to_string(), vec![1, 2, 4]),
        ("norm_out".to_string(), vec![1, 2, 4]),
        ("eps".to_string(), vec![]),
    ])
}

fn rank_two_tensor_shapes() -> HashMap<String, Vec<i64>> {
    HashMap::from([
        ("input".to_string(), vec![1, 4]),
        ("mean_a_out".to_string(), vec![1, 1]),
        ("mean_b_out".to_string(), vec![1, 1]),
        ("center_var_out".to_string(), vec![1, 4]),
        ("square_out".to_string(), vec![1, 4]),
        ("var_mean_out".to_string(), vec![1, 1]),
        ("var_eps_out".to_string(), vec![1, 1]),
        ("std_out".to_string(), vec![1, 1]),
        ("center_norm_out".to_string(), vec![1, 4]),
        ("norm_out".to_string(), vec![1, 4]),
        ("eps".to_string(), vec![]),
    ])
}

fn build_graph_from_layers(
    layers: Vec<LayerSpec>,
    tensor_shapes: HashMap<String, Vec<i64>>,
) -> ny_propagate::GraphNetwork {
    build_graph_from_layers_with_eps(layers, tensor_shapes, 1e-5)
}

fn build_graph_from_layers_with_eps(
    layers: Vec<LayerSpec>,
    tensor_shapes: HashMap<String, Vec<i64>>,
    eps: f32,
) -> ny_propagate::GraphNetwork {
    let inputs = vec![tensor_spec(
        "input",
        tensor_shapes.get("input").expect("input shape"),
    )];
    let outputs = vec![tensor_spec(
        "norm_out",
        tensor_shapes.get("norm_out").expect("output shape"),
    )];
    let mut weights = WeightStore::new();
    weights.insert(
        "eps".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![eps]).expect("scalar eps"),
    );
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    build_graph_network(&data, crate::GraphNetworkOptions::default())
        .expect("graph conversion should succeed")
}

fn build_decomposed_instance_norm_graph(axes: i64) -> ny_propagate::GraphNetwork {
    build_graph_from_layers(decomposed_instance_norm_layers(axes), base_tensor_shapes())
}

fn build_decomposed_instance_norm_reciprocal_mul_graph(axes: i64) -> ny_propagate::GraphNetwork {
    let mut shapes = base_tensor_shapes();
    shapes.insert("inv_std_out".to_string(), vec![1, 2, 1]);
    build_graph_from_layers(decomposed_instance_norm_reciprocal_mul_layers(axes), shapes)
}

fn pow_square_pattern_matches(exponent: f32) -> bool {
    let mut layers = variance_chain_layers(2);
    layers[3] = layer_spec(
        "square",
        LayerType::Pow,
        &["center_var_out", "exponent"],
        &["square_out"],
        HashMap::new(),
    );
    let output_to_spec = layers
        .iter()
        .enumerate()
        .flat_map(|(index, spec)| spec.outputs.iter().cloned().map(move |name| (name, index)))
        .collect::<HashMap<_, _>>();
    let mut weights = WeightStore::new();
    weights.insert(
        "exponent".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![exponent]).expect("scalar exponent"),
    );
    let shapes = base_tensor_shapes();
    let constants = HashSet::new();
    let context = ConvertContext::new(&weights, &shapes, &constants);

    extract_squared_centered_input_tensor("square_out", &layers, &output_to_spec, &context)
        .is_some()
}

#[test]
fn pow_square_fusion_requires_exact_exponent() {
    assert!(pow_square_pattern_matches(2.0));
    assert!(!pow_square_pattern_matches(f32::from_bits(
        2.0_f32.to_bits() - 1
    )));
    assert!(!pow_square_pattern_matches(f32::from_bits(
        2.0_f32.to_bits() + 1
    )));
}

#[test]
fn fuses_decomposed_instance_norm_div_to_monolithic_layer_3591() {
    let graph = build_decomposed_instance_norm_graph(2);
    let norm_div = graph
        .node("norm_div")
        .expect("fused norm node should exist");
    assert!(
        matches!(norm_div.layer(), Layer::InstanceNorm1d(_)),
        "expected norm_div to fuse to InstanceNorm1d, got {:?}",
        norm_div.layer()
    );
    assert_eq!(
        norm_div.inputs(),
        &["_input".to_string()],
        "fused InstanceNorm should read directly from the original activation input"
    );

    let center = ArrayD::zeros(IxDyn(&[2, 4]));
    let input = BoundedTensor::from_epsilon(center, 1e-3).expect("bounded input");
    let output = graph
        .propagate_ibp(&input)
        .expect("fused InstanceNorm graph IBP should succeed");
    assert_eq!(output.lower().shape(), &[2, 4]);
    assert!(output.lower().iter().all(|value| value.is_finite()));
    assert!(output.upper().iter().all(|value| value.is_finite()));
}

#[test]
fn does_not_fuse_reduce_mean_over_non_terminal_axis_3591() {
    let graph = build_decomposed_instance_norm_graph(1);
    let norm_div = graph.node("norm_div").expect("norm_div node should exist");
    assert!(
        matches!(norm_div.layer(), Layer::Div(_)),
        "expected non-matching pattern to remain a Div node, got {:?}",
        norm_div.layer()
    );
}

#[test]
fn does_not_fuse_rank_two_last_axis_normalization_to_instance_norm() {
    let graph =
        build_graph_from_layers(decomposed_instance_norm_layers(1), rank_two_tensor_shapes());
    let norm_div = graph.node("norm_div").expect("norm_div node should exist");
    assert!(
        matches!(norm_div.layer(), Layer::Div(_)),
        "rank-2 last-axis normalization must remain decomposed, got {:?}",
        norm_div.layer()
    );
}

#[test]
fn does_not_fuse_epsilon_below_supported_normalization_range() {
    let unsupported_eps = f32::from_bits(NORMALIZATION_MIN_EPS.to_bits() - 1);
    let graph = build_graph_from_layers_with_eps(
        decomposed_instance_norm_layers(2),
        base_tensor_shapes(),
        unsupported_eps,
    );
    let norm_div = graph.node("norm_div").expect("norm_div node should exist");
    assert!(
        matches!(norm_div.layer(), Layer::Div(_)),
        "a semantically distinct epsilon must remain decomposed, got {:?}",
        norm_div.layer()
    );
}

#[test]
fn fuses_decomposed_instance_norm_reciprocal_mul_to_monolithic_layer_3591() {
    let graph = build_decomposed_instance_norm_reciprocal_mul_graph(2);
    let norm_mul = graph
        .node("norm_mul")
        .expect("fused norm node should exist");
    assert!(
        matches!(norm_mul.layer(), Layer::InstanceNorm1d(_)),
        "expected Reciprocal+Mul pattern to fuse to InstanceNorm1d, got {:?}",
        norm_mul.layer()
    );
    assert_eq!(
        norm_mul.inputs(),
        &["_input".to_string()],
        "fused InstanceNorm should read directly from the original activation input"
    );

    let center = ArrayD::zeros(IxDyn(&[2, 4]));
    let input = BoundedTensor::from_epsilon(center, 1e-3).expect("bounded input");
    let output = graph
        .propagate_ibp(&input)
        .expect("fused InstanceNorm graph IBP should succeed");
    assert_eq!(output.lower().shape(), &[2, 4]);
    assert!(output.lower().iter().all(|value| value.is_finite()));
    assert!(output.upper().iter().all(|value| value.is_finite()));
}

#[test]
fn does_not_fuse_reciprocal_mul_over_non_terminal_axis_3591() {
    let graph = build_decomposed_instance_norm_reciprocal_mul_graph(1);
    let norm_mul = graph.node("norm_mul").expect("norm_mul node should exist");
    assert!(
        !matches!(norm_mul.layer(), Layer::InstanceNorm1d(_)),
        "expected non-matching Reciprocal+Mul pattern to remain unfused, got {:?}",
        norm_mul.layer()
    );
}

/// Verify Div-variant and Reciprocal+Mul-variant fused graphs produce identical
/// IBP bounds. Both patterns fuse to InstanceNorm1d(ny=1, beta=0, eps=1e-5),
/// so their output bounds must be element-wise identical for the same input.
///
/// Part of #3591: Prover verification that both fusion paths are equivalent.
#[test]
fn fused_div_and_reciprocal_mul_ibp_parity_3591() {
    let graph_div = build_decomposed_instance_norm_graph(2);
    let graph_recip = build_decomposed_instance_norm_reciprocal_mul_graph(2);

    // Non-trivial input: C=2, T=4, center has varying values across channels.
    let center = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![0.5, -0.3, 1.0, 0.2, -0.7, 0.8, -0.1, 0.4],
    )
    .expect("center");
    let input = BoundedTensor::from_epsilon(center, 0.05).expect("bounded input");

    let out_div = graph_div.propagate_ibp(&input).expect("Div graph IBP");
    let out_recip = graph_recip
        .propagate_ibp(&input)
        .expect("Reciprocal+Mul graph IBP");

    assert_eq!(out_div.lower().shape(), out_recip.lower().shape());

    let tol = 1e-6;
    for (i, (&d, &r)) in out_div
        .lower()
        .iter()
        .zip(out_recip.lower().iter())
        .enumerate()
    {
        assert!(
            (d - r).abs() <= tol,
            "Lower bound mismatch at index {i}: div={d}, recip={r}, diff={}",
            (d - r).abs()
        );
    }
    for (i, (&d, &r)) in out_div
        .upper()
        .iter()
        .zip(out_recip.upper().iter())
        .enumerate()
    {
        assert!(
            (d - r).abs() <= tol,
            "Upper bound mismatch at index {i}: div={d}, recip={r}, diff={}",
            (d - r).abs()
        );
    }
}

#[test]
fn generated_chain_skips_instancenorm_fusion_4172() {
    let mut shapes = base_tensor_shapes();
    shapes.insert("inv_std_out".to_string(), vec![1, 2, 1]);
    let graph = build_graph_from_layers(generated_layernorm_reciprocal_mul_layers(2), shapes);
    let norm_mul = graph.node("norm_mul").expect("norm_mul node should exist");
    assert!(
        matches!(norm_mul.layer(), Layer::MulBinary(_)),
        "generated LayerNorm fragment must stay a binary multiply, got {:?}",
        norm_mul.layer()
    );
}

/// Compute exact InstanceNorm per channel: (x - mean) / sqrt(var + eps)
/// with ny=1, beta=0. Input is flat [C*T] with C=2, T=4.
fn eval_instance_norm_exact(point: &[f32], norm_eps: f32) -> [f32; 8] {
    let mut output = [0.0_f32; 8];
    for ch in 0..2_usize {
        let offset = ch * 4;
        let ch_vals = &point[offset..offset + 4];
        let mean: f32 = ch_vals.iter().sum::<f32>() / 4.0;
        let var: f32 = ch_vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        let std = (var + norm_eps).sqrt();
        for t in 0..4_usize {
            output[offset + t] = (ch_vals[t] - mean) / std;
        }
    }
    output
}

/// Assert that all elements of `exact` fall within `[ibp_lower, ibp_upper]`.
fn assert_ibp_soundness(exact: &[f32], ibp_out: &BoundedTensor, point_idx: usize) {
    let soundness_tol = 1e-5;
    let lo_slice = ibp_out.lower().as_slice().unwrap();
    let hi_slice = ibp_out.upper().as_slice().unwrap();
    for (i, &exact_val) in exact.iter().enumerate() {
        assert!(
            exact_val >= lo_slice[i] - soundness_tol,
            "Soundness violation (lower) at point {point_idx}, index {i}: \
             exact={exact_val}, lower_bound={}, diff={}",
            lo_slice[i],
            lo_slice[i] - exact_val
        );
        assert!(
            exact_val <= hi_slice[i] + soundness_tol,
            "Soundness violation (upper) at point {point_idx}, index {i}: \
             exact={exact_val}, upper_bound={}, diff={}",
            hi_slice[i],
            exact_val - hi_slice[i]
        );
    }
}

/// Verify fused InstanceNorm graph IBP is sound: for concrete inputs within the
/// bounded interval, the exact InstanceNorm output falls within IBP bounds.
///
/// Part of #3591: graph-level soundness check for fused InstanceNorm IBP.
/// Complements layer-level proptest coverage in normalization_ibp.rs.
#[test]
fn fused_instance_norm_graph_ibp_soundness_3591() {
    let graph = build_decomposed_instance_norm_graph(2);

    // Non-trivial input with varying values. C=2, T=4.
    let center_vals = vec![0.5, -0.3, 1.0, 0.2, -0.7, 0.8, -0.1, 0.4];
    let eps_val = 0.1_f32;
    let center = ArrayD::from_shape_vec(IxDyn(&[2, 4]), center_vals.clone()).expect("center");
    let input = BoundedTensor::from_epsilon(center, eps_val).expect("bounded input");

    let ibp_out = graph.propagate_ibp(&input).expect("fused graph IBP");
    let norm_eps = 1e-5_f32; // eps from build_graph_from_layers

    // Sample concrete points: center, all-lower, all-upper, two mixed corners.
    let test_points: Vec<Vec<f32>> = vec![
        center_vals.clone(),
        center_vals.iter().map(|v| v - eps_val).collect(),
        center_vals.iter().map(|v| v + eps_val).collect(),
        // ch0 at lower, ch1 at upper
        vec![
            0.5 - eps_val,
            -0.3 - eps_val,
            1.0 - eps_val,
            0.2 - eps_val,
            -0.7 + eps_val,
            0.8 + eps_val,
            -0.1 + eps_val,
            0.4 + eps_val,
        ],
        // ch0 at upper, ch1 at lower
        vec![
            0.5 + eps_val,
            -0.3 + eps_val,
            1.0 + eps_val,
            0.2 + eps_val,
            -0.7 - eps_val,
            0.8 - eps_val,
            -0.1 - eps_val,
            0.4 - eps_val,
        ],
    ];

    for (pi, point) in test_points.iter().enumerate() {
        let exact = eval_instance_norm_exact(point, norm_eps);
        assert_ibp_soundness(&exact, &ibp_out, pi);
    }
}
