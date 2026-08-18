// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::loader::scope::detect_block_scope;
use super::subgraph::attention::DiscoveredAttentionNodes;
use super::*;
use crate::model::{DataType, Network, TensorSpec, WeightStore};
use ndarray::{ArrayD, IxDyn};
use ny_propagate::layers::{AddConstantLayer, AddLayer, ConcatLayer};
use std::collections::{HashMap, HashSet};

fn minimal_whisper_with_weights(
    constants: &[&str],
    weights: &[(&str, ArrayD<f32>)],
) -> WhisperModel {
    let constant_tensors = constants
        .iter()
        .map(|name| (*name).to_string())
        .collect::<HashSet<_>>();
    let mut weight_store = WeightStore::new();
    for (name, tensor) in weights {
        weight_store.insert((*name).to_string(), tensor.clone());
    }
    let model = OnnxModel {
        network: Network {
            name: "whisper-test".to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            layers: Vec::new(),
            param_count: 0,
        },
        weights: weight_store,
        tensor_producer: HashMap::new(),
        constant_tensors,
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    };

    WhisperModel {
        model,
        structure: WhisperEncoderStructure {
            stem_end_idx: 0,
            blocks: Vec::new(),
            ln_post_start_idx: 0,
        },
        encoder_layers: 0,
        decoder_layers: 0,
        hidden_dim: 0,
        num_heads: 0,
    }
}

fn minimal_whisper_with_constants(constants: &[&str]) -> WhisperModel {
    minimal_whisper_with_weights(constants, &[])
}

#[test]
fn direct_block_input_shape_validation_is_fixture_free() {
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    let mut model = minimal_whisper_with_constants(&[]);
    model.hidden_dim = 8;

    let rank_two = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4, 8])), 0.1)
        .expect("well-formed rank-two bounds");
    assert!(matches!(
        model.validate_direct_block_input(&rank_two),
        Err(NyError::InvalidSpec(message)) if message.contains("[batch, sequence, hidden]")
    ));

    let wrong_hidden = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[1, 4, 7])), 0.1)
        .expect("well-formed wrong-hidden bounds");
    assert!(matches!(
        model.validate_direct_block_input(&wrong_hidden),
        Err(NyError::ShapeMismatch { expected, got })
            if expected == vec![1, 4, 8] && got == vec![1, 4, 7]
    ));
}

fn layer_spec(inputs: &[&str], layer_type: LayerType) -> LayerSpec {
    LayerSpec {
        name: "test_layer".to_string(),
        layer_type,
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    }
}

fn layer_named(name: &str) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Add,
        inputs: Vec::new(),
        outputs: Vec::new(),
        weights: None,
        attributes: HashMap::new(),
    }
}

fn layer_named_with_type(name: &str, layer_type: LayerType) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: Vec::new(),
        outputs: Vec::new(),
        weights: None,
        attributes: HashMap::new(),
    }
}

fn layer_named_layernorm(name: &str, norm_size: usize) -> LayerSpec {
    let mut spec = layer_named_with_type(name, LayerType::LayerNorm);
    spec.attributes.insert(
        "normalized_shape".to_string(),
        AttributeValue::Ints(vec![norm_size as i64]),
    );
    spec
}

fn attention_plumbing_spec(
    name: &str,
    layer_type: LayerType,
    inputs: &[&str],
    output: &str,
    attributes: &[(&str, AttributeValue)],
) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: vec![output.to_string()],
        weights: None,
        attributes: attributes
            .iter()
            .map(|(name, value)| ((*name).to_string(), value.clone()))
            .collect(),
    }
}

/// A minimal exporter-shaped attention block used to exercise the structural
/// rewrite gate without loading the 33MB Whisper fixture.
fn attention_rewrite_fixture() -> WhisperModel {
    let layers = vec![
        attention_plumbing_spec(
            "ln",
            LayerType::LayerNorm,
            &["input"],
            "ln_out",
            &[("normalized_shape", AttributeValue::Ints(vec![8]))],
        ),
        attention_plumbing_spec(
            "q",
            LayerType::MatMul,
            &["ln_out", "encoder.blocks.0.self_attn.q_proj.weight"],
            "q_out",
            &[],
        ),
        attention_plumbing_spec(
            "k",
            LayerType::MatMul,
            &["ln_out", "encoder.blocks.0.self_attn.k_proj.weight"],
            "k_out",
            &[],
        ),
        attention_plumbing_spec(
            "v",
            LayerType::MatMul,
            &["ln_out", "encoder.blocks.0.self_attn.v_proj.weight"],
            "v_out",
            &[],
        ),
        attention_plumbing_spec(
            "q_reshape",
            LayerType::Reshape,
            &["q_out", "q_shape"],
            "q_r",
            &[],
        ),
        attention_plumbing_spec(
            "q_transpose",
            LayerType::Transpose,
            &["q_r"],
            "q_t",
            &[("perm", AttributeValue::Ints(vec![0, 2, 1, 3]))],
        ),
        // Dynamo-style score scaling: one head_dim^-0.5 Mul on Q.
        attention_plumbing_spec(
            "q_scale",
            LayerType::Mul,
            &["q_t", "score_scale"],
            "q_s",
            &[],
        ),
        attention_plumbing_spec(
            "k_reshape",
            LayerType::Reshape,
            &["k_out", "k_shape"],
            "k_r",
            &[],
        ),
        attention_plumbing_spec(
            "k_transpose",
            LayerType::Transpose,
            &["k_r"],
            "k_t",
            &[("perm", AttributeValue::Ints(vec![0, 2, 3, 1]))],
        ),
        attention_plumbing_spec(
            "scores",
            LayerType::MatMul,
            &["q_s", "k_t"],
            "scores_out",
            &[],
        ),
        attention_plumbing_spec("softmax", LayerType::Softmax, &["scores_out"], "probs", &[]),
        attention_plumbing_spec(
            "v_reshape",
            LayerType::Reshape,
            &["v_out", "v_shape"],
            "v_r",
            &[],
        ),
        attention_plumbing_spec(
            "v_transpose",
            LayerType::Transpose,
            &["v_r"],
            "v_t",
            &[("perm", AttributeValue::Ints(vec![0, 2, 1, 3]))],
        ),
        attention_plumbing_spec("context", LayerType::MatMul, &["probs", "v_t"], "ctx", &[]),
        attention_plumbing_spec(
            "ctx_transpose",
            LayerType::Transpose,
            &["ctx"],
            "ctx_t",
            &[("perm", AttributeValue::Ints(vec![0, 2, 1, 3]))],
        ),
        attention_plumbing_spec(
            "ctx_reshape",
            LayerType::Reshape,
            &["ctx_t", "ctx_shape"],
            "ctx_r",
            &[],
        ),
        attention_plumbing_spec(
            "out",
            LayerType::MatMul,
            &["ctx_r", "encoder.blocks.0.self_attn.out_proj.weight"],
            "attn_out",
            &[],
        ),
    ];
    let layer_count = layers.len();
    let mut weight_store = WeightStore::new();
    for name in [
        "encoder.blocks.0.self_attn.q_proj.weight",
        "encoder.blocks.0.self_attn.k_proj.weight",
        "encoder.blocks.0.self_attn.v_proj.weight",
        "encoder.blocks.0.self_attn.out_proj.weight",
    ] {
        weight_store.insert(name.to_string(), ArrayD::zeros(IxDyn(&[8usize, 8usize])));
    }
    weight_store.insert(
        "score_scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1]), 0.5),
    );
    for name in ["q_shape", "k_shape", "v_shape"] {
        weight_store.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1, 1500, 2, 4]).unwrap(),
        );
    }
    weight_store.insert_integers(
        "ctx_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1, 1500, 8]).unwrap(),
    );
    WhisperModel {
        model: OnnxModel {
            network: Network {
                name: "attention-rewrite-fixture".to_string(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                layers,
                param_count: 0,
            },
            weights: weight_store,
            tensor_producer: HashMap::new(),
            constant_tensors: HashSet::new(),
            tensor_shapes: HashMap::from([
                ("input".to_string(), vec![1, 1500, 8]),
                ("ln_out".to_string(), vec![1, 1500, 8]),
                ("q_out".to_string(), vec![1, 1500, 8]),
                ("q_r".to_string(), vec![1, 1500, 2, 4]),
                ("q_t".to_string(), vec![1, 2, 1500, 4]),
                ("q_s".to_string(), vec![1, 2, 1500, 4]),
                ("k_out".to_string(), vec![1, 1500, 8]),
                ("k_r".to_string(), vec![1, 1500, 2, 4]),
                ("k_t".to_string(), vec![1, 2, 4, 1500]),
                ("scores_out".to_string(), vec![1, 2, 1500, 1500]),
                ("probs".to_string(), vec![1, 2, 1500, 1500]),
                ("v_out".to_string(), vec![1, 1500, 8]),
                ("v_r".to_string(), vec![1, 1500, 2, 4]),
                ("v_t".to_string(), vec![1, 2, 1500, 4]),
                ("ctx".to_string(), vec![1, 2, 1500, 4]),
                ("ctx_t".to_string(), vec![1, 1500, 2, 4]),
                ("ctx_r".to_string(), vec![1, 1500, 8]),
                ("attn_out".to_string(), vec![1, 1500, 8]),
            ]),
            original_float32_initializers: HashMap::new(),
            original_network_topology: None,
            opset_imports: HashMap::new(),
        },
        structure: WhisperEncoderStructure {
            stem_end_idx: 0,
            blocks: vec![WhisperBlockInfo {
                index: 0,
                start_layer_idx: 0,
                end_layer_idx: layer_count,
                num_layers: layer_count,
            }],
            ln_post_start_idx: layer_count,
        },
        encoder_layers: 1,
        decoder_layers: 0,
        hidden_dim: 8,
        num_heads: 2,
    }
}

fn attention_rewrite_nodes(model: &WhisperModel) -> DiscoveredAttentionNodes<'_> {
    let find = |name: &str| {
        model
            .model
            .network
            .layers
            .iter()
            .find(|spec| spec.name == name)
            .unwrap_or_else(|| panic!("missing fixture layer {name}"))
    };
    DiscoveredAttentionNodes {
        attn_ln: find("ln"),
        q_matmul: find("q"),
        q_add: None,
        k_matmul: find("k"),
        k_add: None,
        v_matmul: find("v"),
        v_add: None,
        attn_scores: find("scores"),
        attn_softmax: find("softmax"),
        attn_ctx: find("context"),
        out_matmul: find("out"),
        out_add: None,
    }
}

fn rewrite_plan_error(model: &WhisperModel) -> String {
    model
        .attention_rewrite_plan(0, &attention_rewrite_nodes(model))
        .expect_err("mutated attention fixture must fail closed")
        .to_string()
}

#[test]
fn attention_rewrite_plan_accepts_dynamo_layout_and_scale() {
    let model = attention_rewrite_fixture();
    let plan = model
        .attention_rewrite_plan(0, &attention_rewrite_nodes(&model))
        .expect("known dynamo layout/scale must be structurally equivalent");
    assert_eq!(plan.replaced_nodes.len(), 9);
    for expected in [
        "q_reshape",
        "q_transpose",
        "q_scale",
        "k_reshape",
        "k_transpose",
        "v_reshape",
        "v_transpose",
        "ctx_transpose",
        "ctx_reshape",
    ] {
        assert!(plan.replaced_nodes.contains(expected), "missing {expected}");
    }
}

#[test]
fn attention_rewrite_plan_proves_native_sequence_inference_matches_synthetic_shapes() {
    let mut model = attention_rewrite_fixture();
    // Q: [1,1500,8] -> infer batch in [-1,1500,2,4].
    // Context: [1,1500,2,4] -> infer sequence in [1,-1,8].
    // Both resolve exactly to the synthetic [0,0,...] shapes at the concrete
    // native export sequence, including element-product semantics.
    model.model.weights.insert_integers(
        "q_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1, 1500, 2, 4]).unwrap(),
    );
    model.model.weights.insert_integers(
        "ctx_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1, -1, 8]).unwrap(),
    );
    model
        .attention_rewrite_plan(0, &attention_rewrite_nodes(&model))
        .expect("concrete native-sequence inference must equal synthetic shapes");
}

#[test]
fn attention_rewrite_plan_accepts_legacy_split_qk_scale_equivalence() {
    let mut model = attention_rewrite_fixture();
    let split_scale = 4.0f32.powf(-0.25);
    model.model.weights.insert(
        "score_scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1]), split_scale),
    );
    model.model.weights.insert(
        "key_scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1]), split_scale),
    );
    let scores = model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "scores")
        .unwrap();
    scores.inputs[1] = "k_s".to_string();
    model.model.network.layers.push(attention_plumbing_spec(
        "k_scale",
        LayerType::Mul,
        &["k_t", "key_scale"],
        "k_s",
        &[],
    ));
    let len = model.model.network.layers.len();
    model.structure.blocks[0].end_layer_idx = len;
    model.structure.blocks[0].num_layers = len;
    model.structure.ln_post_start_idx = len;

    let plan = model
        .attention_rewrite_plan(0, &attention_rewrite_nodes(&model))
        .expect("two head_dim^-0.25 factors must equal one head_dim^-0.5 factor");
    assert!(plan.replaced_nodes.contains("q_scale"));
    assert!(plan.replaced_nodes.contains("k_scale"));
}

#[test]
fn attention_rewrite_plan_accepts_dynamo_sdpa_flatten_unflatten_key_path() {
    // torch dynamo lowers scaled_dot_product_attention's K arm by flattening
    // batch*heads for a 3-D batched matmul and unflattening it again:
    //   [B,S,H,D] -transpose[0,2,1,3]-> [B,H,S,D]
    //   -reshape[-1,S,D]-> [B*H,S,D] -transpose[0,2,1]-> [B*H,D,S]
    //   -reshape[1,H,D,S]-> [B,H,D,S]
    // The net axis map is [0,2,3,1] (B,H,D,S) — identical to the single-transpose
    // Key form — so the merge/unmerge Reshapes are proven to only *reorder* whole
    // head axes (a pure regrouping), never reshuffle within one.
    let mut model = attention_rewrite_fixture();

    // K now transposes to B,H,S,D (was a direct B,H,D,S transpose).
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "k_transpose")
        .unwrap()
        .attributes
        .insert("perm".to_string(), AttributeValue::Ints(vec![0, 2, 1, 3]));
    // scores reads the unflattened K instead of k_t directly.
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "scores")
        .unwrap()
        .inputs[1] = "k_u".to_string();
    model.model.network.layers.push(attention_plumbing_spec(
        "k_flatten",
        LayerType::Reshape,
        &["k_t", "k_flat_shape"],
        "k_flat",
        &[],
    ));
    model.model.network.layers.push(attention_plumbing_spec(
        "k_swap",
        LayerType::Transpose,
        &["k_flat"],
        "k_swap_out",
        &[("perm", AttributeValue::Ints(vec![0, 2, 1]))],
    ));
    model.model.network.layers.push(attention_plumbing_spec(
        "k_unflatten",
        LayerType::Reshape,
        &["k_swap_out", "k_unflat_shape"],
        "k_u",
        &[],
    ));
    model.model.weights.insert_integers(
        "k_flat_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1, 1500, 4]).unwrap(),
    );
    model.model.weights.insert_integers(
        "k_unflat_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1, 2, 4, 1500]).unwrap(),
    );
    for (name, shape) in [
        ("k_t", vec![1, 2, 1500, 4]),
        ("k_flat", vec![2, 1500, 4]),
        ("k_swap_out", vec![2, 4, 1500]),
        ("k_u", vec![1, 2, 4, 1500]),
    ] {
        model.model.tensor_shapes.insert(name.to_string(), shape);
    }
    let len = model.model.network.layers.len();
    model.structure.blocks[0].end_layer_idx = len;
    model.structure.blocks[0].num_layers = len;
    model.structure.ln_post_start_idx = len;

    let plan = model
        .attention_rewrite_plan(0, &attention_rewrite_nodes(&model))
        .expect("dynamo SDPA flatten/unflatten K path is a proven pure permutation");
    for expected in [
        "k_reshape",
        "k_transpose",
        "k_flatten",
        "k_swap",
        "k_unflatten",
    ] {
        assert!(plan.replaced_nodes.contains(expected), "missing {expected}");
    }
}

#[test]
fn attention_rewrite_plan_rejects_reshape_that_splits_a_head_axis() {
    // A reshape that cuts *through* a head axis (head_dim 4 -> 2x2) is a real
    // element reshuffle, not a pure regrouping, and must fail closed even though
    // the total element count is preserved.
    let mut model = attention_rewrite_fixture();
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "k_transpose")
        .unwrap()
        .attributes
        .insert("perm".to_string(), AttributeValue::Ints(vec![0, 2, 1, 3]));
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "scores")
        .unwrap()
        .inputs[1] = "k_u".to_string();
    // Split head_dim (4) into 2x2 — misaligned with the atomic head axes.
    model.model.network.layers.push(attention_plumbing_spec(
        "k_split",
        LayerType::Reshape,
        &["k_t", "k_split_shape"],
        "k_u",
        &[],
    ));
    model.model.weights.insert_integers(
        "k_split_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![1, 2, 1500, 2, 2]).unwrap(),
    );
    for (name, shape) in [
        ("k_t", vec![1, 2, 1500, 4]),
        ("k_u", vec![1, 2, 1500, 2, 2]),
    ] {
        model.model.tensor_shapes.insert(name.to_string(), shape);
    }
    let len = model.model.network.layers.len();
    model.structure.blocks[0].end_layer_idx = len;
    model.structure.blocks[0].num_layers = len;
    model.structure.ln_post_start_idx = len;
    assert!(rewrite_plan_error(&model).contains("splits an atomic head axis"));
}

#[test]
fn attention_rewrite_plan_rejects_unproven_reshape_target() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert_integers(
        "q_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1, 1500, 4, 2]).unwrap(),
    );
    assert!(rewrite_plan_error(&model).contains("not proven B,S,heads,head_dim"));
}

#[test]
fn attention_rewrite_plan_rejects_inferred_qkv_layout_with_same_element_count() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert_integers(
        "q_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1, 750, 2, 4]).unwrap(),
    );
    let err = rewrite_plan_error(&model);
    assert!(err.contains("not synthetic B,S,heads,head_dim"), "{err}");
    assert!(err.contains("[2, 750, 2, 4]"), "{err}");
}

#[test]
fn attention_rewrite_plan_rejects_inferred_context_layout_with_same_element_count() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert_integers(
        "ctx_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1, 750, 8]).unwrap(),
    );
    let err = rewrite_plan_error(&model);
    assert!(err.contains("not synthetic B,S,hidden_dim"), "{err}");
    assert!(err.contains("[2, 750, 8]"), "{err}");
}

#[test]
fn attention_rewrite_plan_rejects_missing_source_shape_metadata() {
    let mut model = attention_rewrite_fixture();
    model.model.tensor_shapes.remove("q_out");
    let err = rewrite_plan_error(&model);
    assert!(err.contains("has no shape metadata"), "{err}");
}

#[test]
fn attention_rewrite_plan_rejects_unrepresentable_float_reshape_target() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert(
        "q_shape_float".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, i64::MAX as f32, 2.0, 4.0]).unwrap(),
    );
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_reshape")
        .unwrap()
        .inputs[1] = "q_shape_float".to_string();
    assert!(rewrite_plan_error(&model).contains("unrepresentable target value"));
}

#[test]
fn attention_rewrite_plan_rejects_two_inferred_reshape_dimensions() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert_integers(
        "q_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1, -1, 2, 4]).unwrap(),
    );
    assert!(rewrite_plan_error(&model).contains("not proven B,S,heads,head_dim"));
}

#[test]
fn attention_rewrite_plan_rejects_conflicting_attribute_and_input_shape() {
    let mut model = attention_rewrite_fixture();
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_reshape")
        .unwrap()
        .attributes
        .insert(
            "shape".to_string(),
            AttributeValue::Ints(vec![1, 1500, 2, 4]),
        );
    assert!(rewrite_plan_error(&model).contains("ambiguous/conflicting input shape"));
}

#[test]
fn attention_rewrite_plan_accepts_allowzero_without_explicit_zero() {
    // torch dynamo emits allowzero=1 on head-split reshapes whose targets carry
    // no explicit 0 (e.g. [1,1500,-1,64]); there allowzero=1 and the default are
    // identical, so the rewrite is proven equivalent and accepted.
    let mut model = attention_rewrite_fixture();
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_reshape")
        .unwrap()
        .attributes
        .insert("allowzero".to_string(), AttributeValue::Int(1));
    let plan = model
        .attention_rewrite_plan(0, &attention_rewrite_nodes(&model))
        .expect("allowzero=1 without an explicit 0 target is equivalent to the default");
    assert!(plan.replaced_nodes.contains("q_reshape"));
}

#[test]
fn attention_rewrite_plan_rejects_allowzero_reshape() {
    // allowzero=1 *with* an explicit 0 is genuinely different (a literal empty
    // axis, not "copy source axis"), so it must fail closed.
    let mut model = attention_rewrite_fixture();
    let q_reshape = model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_reshape")
        .unwrap();
    q_reshape
        .attributes
        .insert("allowzero".to_string(), AttributeValue::Int(1));
    model.model.weights.insert_integers(
        "q_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0, 1500, 2, 4]).unwrap(),
    );
    assert!(rewrite_plan_error(&model).contains("allowzero"));
}

#[test]
fn attention_rewrite_plan_rejects_wrong_transpose_permutation() {
    let mut model = attention_rewrite_fixture();
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_transpose")
        .unwrap()
        .attributes
        .insert("perm".to_string(), AttributeValue::Ints(vec![0, 1, 2, 3]));
    assert!(rewrite_plan_error(&model).contains("transpose composition"));
}

#[test]
fn attention_rewrite_plan_rejects_unknown_transpose_attribute() {
    let mut model = attention_rewrite_fixture();
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_transpose")
        .unwrap()
        .attributes
        .insert("unproven".to_string(), AttributeValue::Int(0));
    assert!(rewrite_plan_error(&model).contains("unproven attribute"));
}

#[test]
fn attention_rewrite_plan_rejects_reverse_division() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert(
        "score_scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1]), 2.0),
    );
    let scale = model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_scale")
        .unwrap();
    scale.layer_type = LayerType::Div;
    scale.inputs = vec!["score_scale".to_string(), "q_t".to_string()];
    assert!(rewrite_plan_error(&model).contains("activation / constant"));
}

#[test]
fn attention_rewrite_plan_rejects_zero_divisor() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert(
        "score_scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1]), 0.0),
    );
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_scale")
        .unwrap()
        .layer_type = LayerType::Div;
    assert!(rewrite_plan_error(&model).contains("divides by zero"));
}

#[test]
fn attention_rewrite_plan_rejects_nan_scale() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert(
        "score_scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1]), f32::NAN),
    );
    assert!(rewrite_plan_error(&model).contains("must be finite"));
}

#[test]
fn attention_rewrite_plan_rejects_unknown_scalar_attribute() {
    let mut model = attention_rewrite_fixture();
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "q_scale")
        .unwrap()
        .attributes
        .insert("unproven".to_string(), AttributeValue::Int(0));
    assert!(rewrite_plan_error(&model).contains("unproven attributes"));
}

#[test]
fn attention_rewrite_plan_rejects_scale_outside_two_f32_ulps() {
    let mut model = attention_rewrite_fixture();
    // The old 1e-4 relative tolerance accepted this materially different
    // graph; the synthetic f32 score scale is exactly 0.5 for head_dim=4.
    model.model.weights.insert(
        "score_scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1]), 0.50001),
    );
    assert!(rewrite_plan_error(&model).contains("two-ULP tolerance"));
}

#[test]
fn attention_rewrite_plan_rejects_unit_scale_outside_two_f32_ulps() {
    let mut model = attention_rewrite_fixture();
    let scale = f32::from_bits(1.0f32.to_bits() + 3);
    model.model.weights.insert(
        "unit_scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1]), scale),
    );
    let context_index = model
        .model
        .network
        .layers
        .iter()
        .position(|spec| spec.name == "context")
        .unwrap();
    model.model.network.layers.insert(
        context_index,
        attention_plumbing_spec(
            "v_unit_scale",
            LayerType::Mul,
            &["v_t", "unit_scale"],
            "v_unit_scaled",
            &[],
        ),
    );
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "context")
        .unwrap()
        .inputs[1] = "v_unit_scaled".to_string();
    let len = model.model.network.layers.len();
    model.structure.blocks[0].end_layer_idx = len;
    model.structure.blocks[0].num_layers = len;
    model.structure.ln_post_start_idx = len;

    assert!(rewrite_plan_error(&model).contains("two-ULP unit tolerance"));
}

#[test]
fn attention_rewrite_plan_rejects_unknown_core_matmul_attribute() {
    let mut model = attention_rewrite_fixture();
    model
        .model
        .network
        .layers
        .iter_mut()
        .find(|spec| spec.name == "scores")
        .unwrap()
        .attributes
        .insert("unproven".to_string(), AttributeValue::Int(0));
    assert!(rewrite_plan_error(&model).contains("unproven attribute"));
}

#[test]
fn attention_rewrite_plan_rejects_replaced_output_fanout() {
    let mut model = attention_rewrite_fixture();
    model.model.network.layers.push(attention_plumbing_spec(
        "q_t_extra_consumer",
        LayerType::ReLU,
        &["q_t"],
        "unused",
        &[],
    ));
    let len = model.model.network.layers.len();
    model.structure.blocks[0].end_layer_idx = len;
    model.structure.blocks[0].num_layers = len;
    model.structure.ln_post_start_idx = len;
    let err = rewrite_plan_error(&model);
    assert!(err.contains("consumers"), "unexpected error: {err}");
}

#[test]
fn attention_rewrite_plan_rejects_outside_block_fanout() {
    let mut model = attention_rewrite_fixture();
    // Deliberately leave block bounds/ln_post unchanged: this consumer is not
    // returned by block_layers_for_index, but deleting q_transpose would still
    // corrupt it in the full model.
    model.model.network.layers.push(attention_plumbing_spec(
        "outside_block_consumer",
        LayerType::ReLU,
        &["q_t"],
        "outside",
        &[],
    ));
    let err = rewrite_plan_error(&model);
    assert!(err.contains("consumers"), "unexpected error: {err}");
}

#[test]
fn attention_rewrite_plan_rejects_duplicate_traced_layer_identity() {
    let mut model = attention_rewrite_fixture();
    // This unrelated out-of-block node reuses a traced plumbing name. A
    // name-keyed replacement set must not be allowed to identify both nodes.
    model.model.network.layers.push(attention_plumbing_spec(
        "q_transpose",
        LayerType::ReLU,
        &["outside_input"],
        "outside_duplicate",
        &[],
    ));
    let err = rewrite_plan_error(&model);
    assert!(err.contains("duplicate name 'q_transpose'"), "{err}");

    let fallback = model
        .encoder_layer_graph_full(0)
        .expect("duplicate identity must force the original full-block graph");
    assert!(fallback.contains_node("q_transpose"));
    assert!(!fallback.contains_node("q::__reshape_bshd"));
    let attention_err = match model.attention_subgraph(0) {
        Ok(_) => panic!("attention-only builder must reject duplicate replacement identity"),
        Err(err) => err.to_string(),
    };
    assert!(
        attention_err.contains("duplicate name 'q_transpose'"),
        "{attention_err}"
    );
}

#[test]
fn attention_rewrite_plan_rejects_empty_layer_identity() {
    let mut model = attention_rewrite_fixture();
    model.model.network.layers.push(attention_plumbing_spec(
        "",
        LayerType::ReLU,
        &["outside_input"],
        "outside_empty_name",
        &[],
    ));
    let err = rewrite_plan_error(&model);
    assert!(err.contains("nonempty name"), "{err}");
}

#[test]
fn attention_rewrite_plan_rejects_replaced_network_output() {
    let mut model = attention_rewrite_fixture();
    model.model.network.outputs.push(TensorSpec {
        name: "q_t".to_string(),
        shape: vec![1, 2, 1500, 4],
        dtype: DataType::Float32,
    });
    let err = rewrite_plan_error(&model);
    assert!(err.contains("network output"), "unexpected error: {err}");
}

#[test]
fn attention_rewrite_builders_accept_proven_fixture() {
    let model = attention_rewrite_fixture();
    let full = model
        .encoder_layer_graph_full(0)
        .expect("full-block builder must use a proven structural rewrite");
    assert!(full.contains_node("q::__reshape_bshd"));
    assert!(full.contains_node("context::__reshape_bsd"));
    assert!(!full.contains_node("q_reshape"));

    let attention = model
        .attention_subgraph(0)
        .expect("attention-only builder must accept the same proven rewrite");
    assert!(attention.contains_node("q::__reshape_bshd"));
    assert!(attention.contains_node("context::__reshape_bsd"));
    assert!(!attention.contains_node("q_reshape"));
}

#[test]
fn attention_rewrite_builders_fail_closed_on_non_equivalent_inference() {
    let mut model = attention_rewrite_fixture();
    model.model.weights.insert_integers(
        "q_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1, 750, 2, 4]).unwrap(),
    );

    let fallback = model
        .encoder_layer_graph_full(0)
        .expect("full-block builder must fall back to the original graph");
    assert!(fallback.contains_node("q_reshape"));
    assert!(!fallback.contains_node("q::__reshape_bshd"));

    let err = match model.attention_subgraph(0) {
        Ok(_) => panic!("attention-only builder must reject an unproven rewrite"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("not synthetic B,S,heads,head_dim"), "{err}");
}

#[ntest::timeout(10000)]
#[test]
fn test_find_input_nodes_concat_keeps_constant_tensors() {
    let whisper = minimal_whisper_with_constants(&["const_in"]);
    assert!(whisper.model.constant_tensors.contains("const_in"));
    let spec = layer_spec(&["act_in", "const_in"], LayerType::Concat);
    let layer = Layer::Concat(ConcatLayer::new(0));

    let mut tensor_to_node = HashMap::new();
    tensor_to_node.insert("act_in".to_string(), "act_node".to_string());
    tensor_to_node.insert("const_in".to_string(), "const_node".to_string());

    let mut external_tensors = HashSet::new();
    let input_nodes = whisper
        .find_input_nodes(
            &spec,
            &layer,
            &tensor_to_node,
            &mut external_tensors,
            &whisper.model.constant_tensors,
            &HashMap::new(),
        )
        .expect("concat input nodes should resolve");

    assert_eq!(
        input_nodes,
        vec!["act_node".to_string(), "const_node".to_string()]
    );
    assert!(external_tensors.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_find_input_nodes_concat_skips_evaluated_constants() {
    // Evaluated constants (pre-computed by evaluate_block_constants) should be
    // filtered from graph edges because they're embedded in ConcatLayer::constant_inputs
    // by convert_concat_with_evaluated (#696).
    let whisper = minimal_whisper_with_constants(&[]);
    let spec = layer_spec(&["act_in", "eval_const"], LayerType::Concat);
    let layer = Layer::Concat(ConcatLayer::new(0));

    let mut tensor_to_node = HashMap::new();
    tensor_to_node.insert("act_in".to_string(), "act_node".to_string());

    let mut evaluated = HashMap::new();
    evaluated.insert(
        "eval_const".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("shape"),
    );

    let mut external_tensors = HashSet::new();
    let input_nodes = whisper
        .find_input_nodes(
            &spec,
            &layer,
            &tensor_to_node,
            &mut external_tensors,
            &whisper.model.constant_tensors,
            &evaluated,
        )
        .expect("concat input nodes should resolve");

    // Only activation input appears as graph edge; evaluated constant is embedded in layer.
    assert_eq!(input_nodes, vec!["act_node".to_string()]);
    assert!(external_tensors.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_mixed_encoder_decoder_blocks() {
    let network = Network {
        name: "whisper-mixed".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("stem/conv"),
            layer_named("/encoder/blocks.0/attn"),
            layer_named("/encoder/blocks.0/mlp"),
            layer_named("/encoder/blocks.1/attn"),
            layer_named("/encoder/blocks.1/mlp"),
            layer_named("/encoder/ln_post"),
            layer_named("/decoder/blocks.0/attn"),
            layer_named("/decoder/blocks.0/mlp"),
            layer_named("/decoder/ln_post"),
        ],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert_eq!(structure.stem_end_idx, 1);
    assert_eq!(structure.blocks.len(), 2);
    assert_eq!(structure.blocks[0].index, 0);
    assert_eq!(structure.blocks[0].start_layer_idx, 1);
    assert_eq!(structure.blocks[0].end_layer_idx, 3);
    assert_eq!(structure.blocks[1].index, 1);
    assert_eq!(structure.blocks[1].start_layer_idx, 3);
    assert_eq!(structure.blocks[1].end_layer_idx, 5);
    assert_eq!(structure.ln_post_start_idx, 5);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_rejects_nonzero_first_block() {
    let network = Network {
        name: "whisper-block-one-only".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("/encoder/blocks.1/attn"),
            layer_named("/encoder/blocks.1/mlp"),
        ],
        param_count: 0,
    };

    let error = parse_whisper_structure(&network)
        .expect_err("a block-one-only model must not alias vector position zero");
    assert!(error.to_string().contains("expected 0, got 1"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_rejects_block_index_gap() {
    let network = Network {
        name: "whisper-block-gap".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("/encoder/blocks.0/attn"),
            layer_named("/encoder/blocks.0/mlp"),
            layer_named("/encoder/blocks.2/attn"),
            layer_named("/encoder/blocks.2/mlp"),
        ],
        param_count: 0,
    };

    let error =
        parse_whisper_structure(&network).expect_err("a gapped block sequence must fail closed");
    assert!(error.to_string().contains("expected 1, got 2"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_unscoped_ln_post() {
    let network = Network {
        name: "whisper-unscoped-ln-post".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("stem/conv"),
            layer_named("/encoder/blocks.0/attn"),
            layer_named("/encoder/blocks.0/mlp"),
            layer_named("ln_post"),
        ],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert_eq!(structure.stem_end_idx, 1);
    assert_eq!(structure.blocks.len(), 1);
    assert_eq!(structure.blocks[0].start_layer_idx, 1);
    assert_eq!(structure.blocks[0].end_layer_idx, 3);
    assert_eq!(structure.ln_post_start_idx, 3);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_ln_post_with_block_prefix() {
    let network = Network {
        name: "whisper-block-prefix-ln-post".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("stem/conv"),
            layer_named("/encoder/blocks.0/attn"),
            layer_named("/encoder/blocks.0/mlp"),
            layer_named("/encoder/blocks.1/attn"),
            layer_named("/encoder/blocks.1/mlp"),
            layer_named("/encoder/blocks.1/ln_post"),
        ],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert_eq!(structure.stem_end_idx, 1);
    assert_eq!(structure.blocks.len(), 2);
    assert_eq!(structure.blocks[0].start_layer_idx, 1);
    assert_eq!(structure.blocks[0].end_layer_idx, 3);
    assert_eq!(structure.blocks[1].start_layer_idx, 3);
    assert_eq!(structure.blocks[1].end_layer_idx, 5);
    assert_eq!(structure.ln_post_start_idx, 5);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_ln_f_marker() {
    let mut ln_f = layer_named("/encoder/ln_f");
    ln_f.layer_type = LayerType::LayerNorm;

    let network = Network {
        name: "whisper-ln-f-marker".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("stem/conv"),
            layer_named("/encoder/blocks.0/attn"),
            layer_named("/encoder/blocks.0/mlp"),
            layer_named("/encoder/blocks.1/attn"),
            layer_named("/encoder/blocks.1/mlp"),
            ln_f,
        ],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert_eq!(structure.stem_end_idx, 1);
    assert_eq!(structure.blocks.len(), 2);
    assert_eq!(structure.blocks[0].start_layer_idx, 1);
    assert_eq!(structure.blocks[0].end_layer_idx, 3);
    assert_eq!(structure.blocks[1].start_layer_idx, 3);
    assert_eq!(structure.blocks[1].end_layer_idx, 5);
    assert_eq!(structure.ln_post_start_idx, 5);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_layernorm_type_fallback() {
    let mut ln_post = layer_named("encoder/post_norm");
    ln_post.layer_type = LayerType::LayerNorm;

    let network = Network {
        name: "whisper-layernorm-fallback".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("/encoder/blocks.0/attn"),
            layer_named("/encoder/blocks.0/mlp"),
            layer_named("/encoder/blocks.1/attn"),
            layer_named("/encoder/blocks.1/mlp"),
            ln_post,
        ],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert_eq!(structure.blocks.len(), 2);
    assert_eq!(structure.blocks[1].start_layer_idx, 2);
    assert_eq!(structure.blocks[1].end_layer_idx, 4);
    assert_eq!(structure.ln_post_start_idx, 4);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_prefers_ln_post_after_last_block() {
    let network = Network {
        name: "whisper-ln-post-after-last-block".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("stem/conv"),
            layer_named("/encoder/blocks.0/attn"),
            layer_named("/encoder/blocks.0/mlp"),
            layer_named("layer_norm"),
            layer_named("/encoder/blocks.1/attn"),
            layer_named("/encoder/blocks.1/mlp"),
            layer_named("ln_post"),
        ],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert_eq!(structure.blocks.len(), 2);
    assert_eq!(structure.blocks[0].start_layer_idx, 1);
    assert_eq!(structure.blocks[0].end_layer_idx, 3);
    assert_eq!(structure.blocks[1].start_layer_idx, 4);
    assert_eq!(structure.blocks[1].end_layer_idx, 6);
    assert_eq!(structure.ln_post_start_idx, 6);
}

#[ntest::timeout(10000)]
#[test]
fn test_detect_block_scope_variants() {
    let encoder_network = Network {
        name: "whisper-encoder-scope".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![layer_named("/encoder/blocks.0/attn")],
        param_count: 0,
    };
    assert_eq!(
        detect_block_scope(&encoder_network),
        WhisperBlockScope::Encoder
    );

    let decoder_network = Network {
        name: "whisper-decoder-scope".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![layer_named("/decoder/layers.1/attn")],
        param_count: 0,
    };
    assert_eq!(
        detect_block_scope(&decoder_network),
        WhisperBlockScope::Decoder
    );

    let mixed_network = Network {
        name: "whisper-mixed-scope".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("/encoder/blocks.2/attn"),
            layer_named("decoder.blocks.3.attn"),
        ],
        param_count: 0,
    };
    assert_eq!(
        detect_block_scope(&mixed_network),
        WhisperBlockScope::Encoder
    );

    let unscoped_network = Network {
        name: "whisper-unscoped".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![layer_named("blocks.4/attn")],
        param_count: 0,
    };
    assert_eq!(
        detect_block_scope(&unscoped_network),
        WhisperBlockScope::All
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_block_index_with_scope_variants() {
    assert_eq!(
        parse_block_index_with_scope("/encoder/blocks.7/attn", WhisperBlockScope::Encoder),
        Some(7)
    );
    assert_eq!(
        parse_block_index_with_scope("encoder.blocks.3.attn", WhisperBlockScope::Encoder),
        Some(3)
    );
    assert_eq!(
        parse_block_index_with_scope("/decoder/layers.2/attn", WhisperBlockScope::Decoder),
        Some(2)
    );
    assert_eq!(
        parse_block_index_with_scope("decoder.layers.9.attn", WhisperBlockScope::Decoder),
        Some(9)
    );
    assert_eq!(
        parse_block_index_with_scope("/decoder/blocks.4/attn", WhisperBlockScope::Encoder),
        None
    );
    assert_eq!(
        parse_block_index_with_scope("blocks.11.attn", WhisperBlockScope::Encoder),
        Some(11)
    );
    assert_eq!(
        parse_block_index_with_scope("layers.5.self_attn", WhisperBlockScope::Decoder),
        Some(5)
    );
    assert_eq!(
        parse_block_index_with_scope("blocks.11.attn", WhisperBlockScope::All),
        Some(11)
    );
    assert_eq!(
        parse_block_index_with_scope("layers.5.self_attn", WhisperBlockScope::All),
        Some(5)
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_block_layers_for_index_skips_other_block_indices_in_range() {
    let mut whisper = minimal_whisper_with_constants(&[]);
    whisper.model.network.layers = vec![
        layer_named("stem/conv"),
        layer_named("/encoder/blocks.0/attn"),
        layer_named("unscoped/intermediate"),
        layer_named("/encoder/blocks.1/attn"),
        layer_named("/encoder/ln_post"),
    ];
    whisper.structure = WhisperEncoderStructure {
        stem_end_idx: 1,
        blocks: vec![WhisperBlockInfo {
            index: 0,
            start_layer_idx: 1,
            end_layer_idx: 4,
            num_layers: 3,
        }],
        ln_post_start_idx: 4,
    };
    whisper.encoder_layers = 1;

    let layers = whisper
        .block_layers_for_index(0)
        .expect("block layers should resolve");
    let names: Vec<&str> = layers.iter().map(|spec| spec.name.as_str()).collect();

    assert!(names.contains(&"/encoder/blocks.0/attn"));
    assert!(names.contains(&"unscoped/intermediate"));
    assert!(!names.contains(&"/encoder/blocks.1/attn"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_no_blocks_with_ln_post() {
    let network = Network {
        name: "whisper-no-blocks-ln-post".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("stem/conv"),
            layer_named("ln_post"),
            layer_named("tail/fc"),
        ],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert!(structure.blocks.is_empty());
    assert_eq!(structure.stem_end_idx, 0);
    assert_eq!(structure.ln_post_start_idx, 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_no_blocks_no_ln_post() {
    let network = Network {
        name: "whisper-no-blocks-no-ln-post".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![layer_named("stem/conv"), layer_named("tail/fc")],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert!(structure.blocks.is_empty());
    assert_eq!(structure.stem_end_idx, 0);
    assert_eq!(structure.ln_post_start_idx, 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_whisper_structure_blocks_no_ln_post() {
    let network = Network {
        name: "whisper-blocks-no-ln-post".to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        layers: vec![
            layer_named("stem/conv"),
            layer_named("/encoder/blocks.0/attn"),
            layer_named("/encoder/blocks.0/mlp"),
            layer_named("/encoder/blocks.1/attn"),
            layer_named("/encoder/blocks.1/mlp"),
        ],
        param_count: 0,
    };

    let structure = parse_whisper_structure(&network).expect("structure parse should succeed");

    assert_eq!(structure.stem_end_idx, 1);
    assert_eq!(structure.blocks.len(), 2);
    assert_eq!(structure.blocks[0].start_layer_idx, 1);
    assert_eq!(structure.blocks[0].end_layer_idx, 3);
    assert_eq!(structure.blocks[1].start_layer_idx, 3);
    assert_eq!(structure.blocks[1].end_layer_idx, 5);
    assert_eq!(structure.ln_post_start_idx, network.layers.len());
}

#[ntest::timeout(10000)]
#[test]
fn test_encoder_layer_from_network_prefers_last_pre_block_layernorm() {
    let mut whisper = minimal_whisper_with_constants(&[]);
    whisper.model.network.layers = vec![
        layer_named_layernorm("stem/conv", 4),
        layer_named_layernorm("stem/ln_early", 4),
        layer_named_layernorm("stem/ln_pre_block", 4),
        layer_named_layernorm("/encoder/blocks.0/attn", 4),
        layer_named_layernorm("/encoder/blocks.0/mlp", 4),
    ];
    whisper.structure = WhisperEncoderStructure {
        stem_end_idx: 1,
        blocks: vec![WhisperBlockInfo {
            index: 0,
            start_layer_idx: 3,
            end_layer_idx: 5,
            num_layers: 2,
        }],
        ln_post_start_idx: 5,
    };
    whisper.encoder_layers = 1;

    let full_network = whisper
        .model
        .to_propagate_network()
        .expect("propagate network should build");
    let block = whisper
        .encoder_layer_from_network(&full_network, 0)
        .expect("block extraction should succeed");

    assert_eq!(block.num_layers(), 3);
}

#[ntest::timeout(10000)]
#[test]
fn test_find_input_nodes_binary_filters_constant_tensors() {
    let whisper = minimal_whisper_with_constants(&["const_in"]);
    assert!(whisper.model.constant_tensors.contains("const_in"));
    let spec = layer_spec(&["act_in", "const_in"], LayerType::Add);
    assert!(whisper
        .model
        .constant_tensors
        .contains(spec.inputs[1].as_str()));
    assert!(whisper.model.constant_tensors.contains(&spec.inputs[1]));
    let layer = Layer::Add(AddLayer);

    let mut tensor_to_node = HashMap::new();
    tensor_to_node.insert("act_in".to_string(), "act_node".to_string());
    tensor_to_node.insert("const_in".to_string(), "const_node".to_string());

    let mut external_tensors = HashSet::new();
    let err = whisper
        .find_input_nodes(
            &spec,
            &layer,
            &tensor_to_node,
            &mut external_tensors,
            &whisper.model.constant_tensors,
            &HashMap::new(),
        )
        .expect_err("binary input nodes should reject constant tensors");

    assert!(err
        .to_string()
        .contains("constant tensor inputs without a weight constant"));
    assert!(external_tensors.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_find_input_nodes_concat_skips_weight_inputs() {
    // After #696: weights are embedded in ConcatLayer::constant_inputs by
    // convert_concat_with_evaluated, so find_input_nodes skips them from graph
    // edges. Only activation inputs produce graph edges.
    let weight = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).expect("weight shape");
    let whisper = minimal_whisper_with_weights(&[], &[("const_w", weight)]);
    assert!(whisper.model.weights.contains_key("const_w"));

    let spec = layer_spec(&["act_in", "const_w"], LayerType::Concat);
    let layer = Layer::Concat(ConcatLayer::new(0));

    let mut tensor_to_node = HashMap::new();
    tensor_to_node.insert("act_in".to_string(), "act_node".to_string());

    let mut external_tensors = HashSet::new();
    let input_nodes = whisper
        .find_input_nodes(
            &spec,
            &layer,
            &tensor_to_node,
            &mut external_tensors,
            &whisper.model.constant_tensors,
            &HashMap::new(),
        )
        .expect("concat input nodes should resolve");

    // Only activation input appears as graph edge; weight is embedded in layer.
    assert_eq!(input_nodes, vec!["act_node".to_string()]);
    assert!(external_tensors.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_find_activation_input_skips_constant_tensor() {
    let whisper = minimal_whisper_with_constants(&["const_in"]);
    let inputs = vec!["const_in".to_string(), "act_in".to_string()];

    let activation = whisper
        .find_activation_input(&inputs, &whisper.model.constant_tensors, &HashMap::new())
        .expect("activation input should ignore constants");

    assert_eq!(activation, "act_in");
}

/// Regression test for #697: find_activation_input must also skip evaluated_constants.
/// A tensor only present in evaluated_constants (not in weights or constant_tensors)
/// must not be returned as an activation input.
#[ntest::timeout(10000)]
#[test]
fn test_find_activation_input_skips_evaluated_constants() {
    let whisper = minimal_whisper_with_constants(&[]);
    let inputs = vec!["eval_const".to_string(), "act_in".to_string()];

    // "eval_const" is not in weights or constant_tensors, but IS in evaluated_constants
    let mut evaluated = HashMap::new();
    evaluated.insert("eval_const".to_string(), ArrayD::<f32>::zeros(vec![1]));

    let activation = whisper
        .find_activation_input(&inputs, &whisper.model.constant_tensors, &evaluated)
        .expect("activation input should skip evaluated constants");

    assert_eq!(activation, "act_in");
}

/// Regression test for #697: evaluated_constants must be filtered from activation
/// inputs. A tensor in `evaluated_constants` (pre-evaluated constant chain) but NOT
/// in `constant_tensors` or `weights` must be skipped by `find_input_nodes`, not
/// treated as an activation input.
#[ntest::timeout(10000)]
#[test]
fn test_find_input_nodes_skips_evaluated_constants() {
    // Create a model with no weights and no constant_tensors for "eval_const".
    // The tensor "eval_const" exists only in evaluated_constants.
    let whisper = minimal_whisper_with_constants(&[]);
    assert!(!whisper.model.weights.contains_key("eval_const"));
    assert!(!whisper.model.constant_tensors.contains("eval_const"));

    // Layer with two inputs: one activation, one pre-evaluated constant.
    // Use AddConstant (unary op with embedded constant) since the converter
    // would create this variant when one Add input is a known constant.
    let spec = layer_spec(&["act_in", "eval_const"], LayerType::Add);
    let layer = Layer::AddConstant(AddConstantLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![42.0]).unwrap(),
    ));

    let mut tensor_to_node = HashMap::new();
    tensor_to_node.insert("act_in".to_string(), "act_node".to_string());

    // Put "eval_const" in evaluated_constants only.
    let mut evaluated = HashMap::new();
    evaluated.insert(
        "eval_const".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![42.0]).unwrap(),
    );

    let mut external_tensors = HashSet::new();
    let input_nodes = whisper
        .find_input_nodes(
            &spec,
            &layer,
            &tensor_to_node,
            &mut external_tensors,
            &whisper.model.constant_tensors,
            &evaluated,
        )
        .expect("should resolve with evaluated constant filtered");

    // "eval_const" should be skipped (line 774 in find_input_nodes), leaving only "act_in".
    assert_eq!(input_nodes, vec!["act_node".to_string()]);
    assert!(external_tensors.is_empty());
}
