// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for InstanceNorm/LayerNorm discrimination (#3591).
//!
//! The discriminator `try_discriminate_instance_norm` takes a `LayerSpec`
//! already typed as `LayerNorm` (from `try_fuse_layer_norm`) and remaps it
//! to `InstanceNorm` when ny shape matches the channel dim (axis 1)
//! rather than the last (reduced) dim.

use crate::loader::fusion::instance_norm::try_discriminate_instance_norm;
use crate::model::WeightStore;
use crate::LayerSpec;
use ndarray::arr1;
use ny_core::LayerType;
use std::collections::{HashMap, HashSet};

fn make_layer_norm_spec(input_name: &str, ny_name: &str, beta_name: &str) -> LayerSpec {
    LayerSpec {
        name: "test_norm".to_string(),
        layer_type: LayerType::LayerNorm,
        inputs: vec![
            input_name.to_string(),
            ny_name.to_string(),
            beta_name.to_string(),
        ],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    }
}

/// 3D input [B, 512, T] with ny [512] → InstanceNorm (channel dim).
#[test]
fn test_discriminate_3d_channel_dim_remaps_to_instance_norm() {
    let mut spec = make_layer_norm_spec("x", "ny", "beta");
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 512, 100]);

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&vec![1.0; 512]).into_dyn());

    assert!(try_discriminate_instance_norm(&mut spec, &shapes, &weights));
    assert_eq!(spec.layer_type, LayerType::InstanceNorm);
}

/// 3D input [B, T, D] with ny [D] → stays LayerNorm (last dim).
#[test]
fn test_discriminate_3d_last_dim_stays_layer_norm() {
    let mut spec = make_layer_norm_spec("x", "ny", "beta");
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 32, 768]);

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&vec![1.0; 768]).into_dyn());

    assert!(!try_discriminate_instance_norm(
        &mut spec, &shapes, &weights
    ));
    assert_eq!(spec.layer_type, LayerType::LayerNorm);
}

/// Ambiguous case: channel_dim == last_dim → conservative, stays LayerNorm.
#[test]
fn test_discriminate_ambiguous_same_dims_stays_layer_norm() {
    let mut spec = make_layer_norm_spec("x", "ny", "beta");
    let mut shapes = HashMap::new();
    // [B, 256, 256]: channel_dim == last_dim
    shapes.insert("x".to_string(), vec![1, 256, 256]);

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&vec![1.0; 256]).into_dyn());

    assert!(!try_discriminate_instance_norm(
        &mut spec, &shapes, &weights
    ));
    assert_eq!(spec.layer_type, LayerType::LayerNorm);
}

/// 2D input [B, D] → not discriminated (needs 3D+).
#[test]
fn test_discriminate_2d_input_not_discriminated() {
    let mut spec = make_layer_norm_spec("x", "ny", "beta");
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 512]);

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&vec![1.0; 512]).into_dyn());

    assert!(!try_discriminate_instance_norm(
        &mut spec, &shapes, &weights
    ));
    assert_eq!(spec.layer_type, LayerType::LayerNorm);
}

/// Missing shape info → not discriminated.
#[test]
fn test_discriminate_missing_shape_not_discriminated() {
    let mut spec = make_layer_norm_spec("x", "ny", "beta");
    let shapes = HashMap::new(); // no shapes

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&vec![1.0; 512]).into_dyn());

    assert!(!try_discriminate_instance_norm(
        &mut spec, &shapes, &weights
    ));
    assert_eq!(spec.layer_type, LayerType::LayerNorm);
}

/// Missing ny weight → not discriminated.
#[test]
fn test_discriminate_missing_ny_not_discriminated() {
    let mut spec = make_layer_norm_spec("x", "ny", "beta");
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 512, 100]);

    let weights = WeightStore::new(); // no weights

    assert!(!try_discriminate_instance_norm(
        &mut spec, &shapes, &weights
    ));
    assert_eq!(spec.layer_type, LayerType::LayerNorm);
}

/// Spec with fewer than 2 inputs → not discriminated.
#[test]
fn test_discriminate_single_input_not_discriminated() {
    let mut spec = LayerSpec {
        name: "test_norm".to_string(),
        layer_type: LayerType::LayerNorm,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let shapes = HashMap::new();
    let weights = WeightStore::new();

    assert!(!try_discriminate_instance_norm(
        &mut spec, &shapes, &weights
    ));
}

/// 4D input [B, C, H, W] with ny [C] → InstanceNorm (channel dim).
/// This exercises the 4D case (e.g., image normalization).
#[test]
fn test_discriminate_4d_channel_dim_remaps_to_instance_norm() {
    let mut spec = make_layer_norm_spec("x", "ny", "beta");
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 64, 32, 32]);

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&vec![1.0; 64]).into_dyn());

    assert!(try_discriminate_instance_norm(&mut spec, &shapes, &weights));
    assert_eq!(spec.layer_type, LayerType::InstanceNorm);
}

// ---------------------------------------------------------------------------
// End-to-end proto-level integration tests (#3591)
//
// These exercise the full ONNX proto → try_fuse_layer_norm →
// try_discriminate_instance_norm pipeline. The discriminator unit tests
// above start from a pre-built LayerSpec; these start from raw ONNX
// NodeProto objects to verify the fusion + discrimination pipeline works
// end-to-end.
// ---------------------------------------------------------------------------

use super::{make_axes_attr, make_const_scalar, make_node};
use crate::loader::fusion::try_fuse_layer_norm;

/// Build producer/consumer maps for a node list (shared test helper).
fn build_maps(
    nodes: &[crate::onnx_proto::NodeProto],
) -> (HashMap<&str, usize>, HashMap<&str, Vec<usize>>) {
    let mut producer_by_output = HashMap::new();
    let mut consumers_by_input: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for output in &node.output {
            producer_by_output.insert(output.as_str(), idx);
        }
        for input in &node.input {
            consumers_by_input
                .entry(input.as_str())
                .or_default()
                .push(idx);
        }
    }
    (producer_by_output, consumers_by_input)
}

/// Full pipeline: decomposed normalization on a [B, C, T] input with
/// ny [C] (channel dim) should produce InstanceNorm after discrimination.
///
/// This is the exact pattern the kokoro vocoder's AdaIN produces after
/// freezing the style auxiliary input.
///
/// Part of #3591: integration test for proto-level InstanceNorm fusion.
#[test]
fn test_full_pipeline_3d_channel_ny_produces_instance_norm_3591() {
    // Build decomposed ReduceMean→Sub→Mul(self)→ReduceMean→Add(eps)→Sqrt→
    // Reciprocal→Mul→Mul(ny)→Add(beta) pattern
    let mut mean1 = make_node("ReduceMean", &["x"], &["mean1"]);
    mean1.attribute.push(make_axes_attr(&[-1]));

    let sub = make_node("Sub", &["x", "mean1"], &["centered"]);
    let square = make_node("Mul", &["centered", "centered"], &["squared"]);

    let mut mean2 = make_node("ReduceMean", &["squared"], &["mean2"]);
    mean2.attribute.push(make_axes_attr(&[-1]));

    let eps = make_const_scalar("eps", 1e-5);
    let add_eps = make_node("Add", &["mean2", "eps"], &["var_eps"]);
    let sqrt = make_node("Sqrt", &["var_eps"], &["std"]);
    let inv = make_node("Reciprocal", &["std"], &["inv_std"]);
    let mul_norm = make_node("Mul", &["centered", "inv_std"], &["norm"]);
    let mul_gamma = make_node("Mul", &["norm", "ny"], &["scaled"]);
    let add_beta = make_node("Add", &["scaled", "beta"], &["out"]);

    let nodes = vec![
        mean1, sub, square, mean2, eps, add_eps, sqrt, inv, mul_norm, mul_gamma, add_beta,
    ];
    let (producer_by_output, consumers_by_input) = build_maps(&nodes);

    // ny [C=512] for input [B=1, C=512, T=100]
    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&vec![1.0; 512]).into_dyn());
    weights.insert("beta".to_string(), arr1(&vec![0.0; 512]).into_dyn());

    // Step 1: LayerNorm fusion
    let (_start_idx, mut spec, _fused_nodes) = try_fuse_layer_norm(
        &nodes,
        0,
        &producer_by_output,
        &consumers_by_input,
        &weights,
        &HashSet::new(),
    )
    .expect("try_fuse_layer_norm should match the decomposed normalization pattern");
    assert_eq!(
        spec.layer_type,
        LayerType::LayerNorm,
        "before discrimination, the fused spec should be LayerNorm"
    );

    // Step 2: InstanceNorm discrimination
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 512, 100]); // [B, C, T]

    let discriminated = try_discriminate_instance_norm(&mut spec, &shapes, &weights);
    assert!(
        discriminated,
        "ny [512] on input [1, 512, 100] should discriminate as InstanceNorm"
    );
    assert_eq!(
        spec.layer_type,
        LayerType::InstanceNorm,
        "after discrimination, the fused spec should be InstanceNorm"
    );
    assert_eq!(spec.inputs[0], "x", "activation input should be preserved");
}

/// Full pipeline: same normalization pattern but with ny matching the
/// last (reduced) dim should stay as LayerNorm after discrimination.
///
/// Part of #3591: negative integration test for proto-level discrimination.
#[test]
fn test_full_pipeline_3d_last_dim_ny_stays_layer_norm_3591() {
    let mut mean1 = make_node("ReduceMean", &["x"], &["mean1"]);
    mean1.attribute.push(make_axes_attr(&[-1]));

    let sub = make_node("Sub", &["x", "mean1"], &["centered"]);
    let two = make_const_scalar("two", 2.0);
    let pow = make_node("Pow", &["centered", "two"], &["squared"]);

    let mut mean2 = make_node("ReduceMean", &["squared"], &["mean2"]);
    mean2.attribute.push(make_axes_attr(&[-1]));

    let eps = make_const_scalar("eps", 1e-5);
    let add_eps = make_node("Add", &["mean2", "eps"], &["var_eps"]);
    let sqrt = make_node("Sqrt", &["var_eps"], &["std"]);
    let div = make_node("Div", &["centered", "std"], &["norm"]);
    let mul_gamma = make_node("Mul", &["norm", "ny"], &["scaled"]);
    let add_beta = make_node("Add", &["scaled", "beta"], &["out"]);

    let nodes = vec![
        mean1, sub, two, pow, mean2, eps, add_eps, sqrt, div, mul_gamma, add_beta,
    ];
    let (producer_by_output, consumers_by_input) = build_maps(&nodes);

    // ny [D=768] for input [B=1, T=32, D=768] — last dim, not channel
    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&vec![1.0; 768]).into_dyn());
    weights.insert("beta".to_string(), arr1(&vec![0.0; 768]).into_dyn());

    let (_start_idx, mut spec, _fused_nodes) = try_fuse_layer_norm(
        &nodes,
        0,
        &producer_by_output,
        &consumers_by_input,
        &weights,
        &HashSet::new(),
    )
    .expect("try_fuse_layer_norm should match the Pow+Div normalization pattern");

    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 32, 768]); // [B, T, D]

    let discriminated = try_discriminate_instance_norm(&mut spec, &shapes, &weights);
    assert!(
        !discriminated,
        "ny [768] on input [1, 32, 768] should NOT discriminate as InstanceNorm"
    );
    assert_eq!(
        spec.layer_type,
        LayerType::LayerNorm,
        "pattern with ny matching last dim should stay as LayerNorm"
    );
}
