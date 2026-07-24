// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Consumer-side integration: traced `GraphModel` → `GraphNetwork` → `Verifier`.
//!
//! The ny-api `graph_model_handoff.rs` tests prove the `GraphModel` contract
//! from inside the ny-api crate. This test proves the same contract is usable
//! from ny-cli — the primary downstream consumer — via the curated
//! `ny_api::model` surface.
//!
//! This is the narrower integration smoke that `#3288` needs: the programmatic
//! traced-producer path works end-to-end from a consumer that is NOT ny-api.

use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, IxDyn};
use ny_api::graph::NETWORK_INPUT;
use ny_api::model::{
    AttributeValue, DataType, GraphModel, GraphModelBuilder, GraphNetworkOptions, LayerSpec,
    LayerType, NetworkSpec, TensorSpec, WeightStore,
};
use ny_api::verify::{PropagationConfig, PropagationMethod, Verifier};
use ny_api::{Bound, VerificationResult, VerificationSpec};

fn style_gate_linear_layer() -> LayerSpec {
    LayerSpec {
        name: "style_linear".to_string(),
        layer_type: LayerType::Linear,
        inputs: vec![
            "style".to_string(),
            "linear_weight".to_string(),
            "linear_bias".to_string(),
        ],
        outputs: vec!["linear_out".to_string()],
        weights: None,
        attributes: HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
    }
}

fn style_gate_split_layer() -> LayerSpec {
    LayerSpec {
        name: "style_split".to_string(),
        layer_type: LayerType::Slice,
        inputs: vec!["reshaped".to_string(), "split_sizes".to_string()],
        outputs: vec!["style_gate".to_string(), "style_residual".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(1))]),
    }
}

fn style_gate_builder_with_weights(builder: GraphModelBuilder) -> GraphModelBuilder {
    builder
        .weight(
            "linear_weight",
            ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0])
                .expect("valid linear weight"),
        )
        .weight(
            "linear_bias",
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).expect("valid bias"),
        )
        .weight(
            "reshape_shape",
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 2.0]).expect("valid reshape shape"),
        )
        .weight(
            "split_sizes",
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).expect("valid split sizes"),
        )
}

fn style_gate_builder_with_layers(builder: GraphModelBuilder) -> GraphModelBuilder {
    builder
        .layer(style_gate_linear_layer())
        .layer(LayerSpec {
            name: "style_reshape".to_string(),
            layer_type: LayerType::Reshape,
            inputs: vec!["linear_out".to_string(), "reshape_shape".to_string()],
            outputs: vec!["reshaped".to_string()],
            weights: None,
            attributes: HashMap::new(),
        })
        .layer(style_gate_split_layer())
        .layer(LayerSpec {
            name: "mixed_add".to_string(),
            layer_type: LayerType::Add,
            inputs: vec!["activation".to_string(), "style_gate".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        })
}

fn style_gate_graph_model_builder(name: &str) -> GraphModelBuilder {
    let builder = GraphModelBuilder::new(name)
        .input("activation", &[1, 1, 2], DataType::Float32)
        .output("out", &[1, 1, 2], DataType::Float32)
        .frozen_input(
            "style",
            &[1, 2],
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).expect("valid style tensor"),
        )
        .tensor_shape("activation", &[1, 1, 2])
        .tensor_shape("linear_out", &[1, 4])
        .tensor_shape("reshaped", &[1, 2, 2])
        .tensor_shape("style_gate", &[1, 1, 2])
        .tensor_shape("style_residual", &[1, 1, 2]);
    style_gate_builder_with_layers(style_gate_builder_with_weights(builder))
}

/// Build a style-gate constant model: a traced producer pattern where a
/// constant-weight linear prelude folds away, leaving only the live activation
/// path through a single Add node.
///
/// Architecture:
///   activation ──────────────────────────> Add → out
///   style(const) → Linear → Reshape → Split ─┘
///
/// After constant folding the graph collapses to: activation → Add(constant) → out.
fn style_gate_constant_graph_model() -> GraphModel {
    style_gate_graph_model_builder("style-gate-consumer").build()
}

fn talker_like_rotary_bias_graph_model() -> GraphModel {
    GraphModelBuilder::new("talker-like-rotary-bias-consumer")
        .input("hidden_states", &[1, 4, 2], DataType::Float32)
        .output("out", &[1, 4, 2], DataType::Float32)
        .frozen_input(
            "cos",
            &[1, 4, 2],
            ArrayD::from_elem(IxDyn(&[4, 2]), 1.0_f32),
        )
        .frozen_input(
            "sin",
            &[1, 4, 2],
            ArrayD::from_elem(IxDyn(&[4, 2]), 2.0_f32),
        )
        .frozen_input(
            "mask",
            &[1, 4, 4],
            ArrayD::from_shape_vec(
                IxDyn(&[4, 4]),
                vec![
                    1.0_f32, 1.0, 9.0, 9.0, //
                    1.0, 1.0, 9.0, 9.0, //
                    1.0, 1.0, 9.0, 9.0, //
                    1.0, 1.0, 9.0, 9.0,
                ],
            )
            .expect("valid mask tensor"),
        )
        .layer(LayerSpec {
            name: "cos_merge".to_string(),
            layer_type: LayerType::Add,
            inputs: vec!["hidden_states".to_string(), "cos".to_string()],
            outputs: vec!["hidden_plus_cos".to_string()],
            weights: None,
            attributes: HashMap::new(),
        })
        .layer(LayerSpec {
            name: "pre_mask_merge".to_string(),
            layer_type: LayerType::Add,
            inputs: vec!["hidden_plus_cos".to_string(), "sin".to_string()],
            outputs: vec!["pre_mask_out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        })
        .layer(LayerSpec {
            name: "mask_gate".to_string(),
            layer_type: LayerType::Slice,
            inputs: vec!["mask".to_string()],
            outputs: vec!["mask_gate_out".to_string()],
            weights: None,
            // Axis 2 in producer-declared [1, 4, 4] space becomes runtime axis 1
            // after batch stripping, yielding the intended [4, 2] mask gate.
            attributes: HashMap::from([
                ("axis".to_string(), AttributeValue::Int(2)),
                ("start".to_string(), AttributeValue::Int(0)),
                ("end".to_string(), AttributeValue::Int(2)),
            ]),
        })
        .layer(LayerSpec {
            name: "output_merge".to_string(),
            layer_type: LayerType::Add,
            inputs: vec!["pre_mask_out".to_string(), "mask_gate_out".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        })
        .tensor_shape("hidden_states", &[1, 4, 2])
        .tensor_shape("hidden_plus_cos", &[1, 4, 2])
        .tensor_shape("pre_mask_out", &[1, 4, 2])
        .tensor_shape("mask_gate_out", &[1, 4, 2])
        .tensor_shape("out", &[1, 4, 2])
        .build()
}

/// Build a sequence-shaped phoneme encoder that mirrors the `#3519`
/// adversarial robustness contract:
///
/// `phoneme_embeddings[seq, dim] -> Linear -> ReLU -> Linear`
///
/// This stays on the owned `GraphModel -> build_graph_network()` boundary while
/// matching the same input/output rank the downstream `verify_robustness()`
/// packet uses.
fn adversarial_phoneme_graph_model() -> GraphModel {
    GraphModelBuilder::new("adversarial-phoneme-consumer")
        .input("phoneme_embeddings", &[2, 2], DataType::Float32)
        .output("robustness_scores", &[2, 1], DataType::Float32)
        .weight(
            "encoder_weight",
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0])
                .expect("valid encoder weight"),
        )
        .weight(
            "encoder_bias",
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.25]).expect("valid encoder bias"),
        )
        .weight(
            "projection_weight",
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0])
                .expect("valid projection weight"),
        )
        .weight(
            "projection_bias",
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).expect("valid projection bias"),
        )
        .layer(LayerSpec {
            name: "phoneme_encoder".to_string(),
            layer_type: LayerType::Linear,
            inputs: vec![
                "phoneme_embeddings".to_string(),
                "encoder_weight".to_string(),
                "encoder_bias".to_string(),
            ],
            outputs: vec!["encoded_phonemes".to_string()],
            weights: None,
            attributes: HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
        })
        .layer(LayerSpec {
            name: "phoneme_relu".to_string(),
            layer_type: LayerType::ReLU,
            inputs: vec!["encoded_phonemes".to_string()],
            outputs: vec!["encoded_relu".to_string()],
            weights: None,
            attributes: HashMap::new(),
        })
        .layer(LayerSpec {
            name: "phoneme_projection".to_string(),
            layer_type: LayerType::Linear,
            inputs: vec![
                "encoded_relu".to_string(),
                "projection_weight".to_string(),
                "projection_bias".to_string(),
            ],
            outputs: vec!["robustness_scores".to_string()],
            weights: None,
            attributes: HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
        })
        .tensor_shape("phoneme_embeddings", &[2, 2])
        .tensor_shape("encoded_phonemes", &[2, 2])
        .tensor_shape("encoded_relu", &[2, 2])
        .tensor_shape("robustness_scores", &[2, 1])
        .build()
}

fn adversarial_phoneme_input_bounds() -> Vec<Bound> {
    vec![
        Bound::new(0.0, 0.25),
        Bound::new(0.5, 0.75),
        Bound::new(1.0, 1.25),
        Bound::new(1.5, 1.75),
    ]
}

/// Exact interval image of the #3519 phoneme encoder over
/// [`adversarial_phoneme_input_bounds`]: row-wise `x0 + x1 + 1.75`.
fn adversarial_phoneme_exact_bounds() -> Vec<Bound> {
    vec![Bound::new(2.25, 2.75), Bound::new(4.25, 4.75)]
}

/// Padding admitting the sound CROWN reductions: they accumulate in f64 and
/// cast outward (`next_down`/`next_up`), so each reported endpoint can sit a
/// few f32 ULP outside the exact value (measured: <= 5 ULP at this scale,
/// ~1.2e-6). 16 ULP at magnitude 4.75 keeps the spec tight while never
/// tripping on the directed rounding.
const PHONEME_BOUND_ULP_PAD: f32 = 16.0 * 4.75 * f32::EPSILON;

fn adversarial_phoneme_expected_bounds() -> Vec<Bound> {
    adversarial_phoneme_exact_bounds()
        .iter()
        .map(|bound| {
            Bound::new(
                bound.lower() - PHONEME_BOUND_ULP_PAD,
                bound.upper() + PHONEME_BOUND_ULP_PAD,
            )
        })
        .collect()
}

fn repeated_bounds(lower: f32, upper: f32, count: usize) -> Vec<Bound> {
    (0..count).map(|_| Bound::new(lower, upper)).collect()
}

/// Consumer-side smoke: GraphModel → GraphNetwork → Verifier::verify_graph().
///
/// Proves the traced-producer contract is usable from ny-cli (downstream
/// consumer) via the curated ny-api surface. The constant prelude folds
/// away, and the resulting graph verifies under IBP with known bounds.
#[ntest::timeout(10000)]
#[test]
fn test_graph_model_consumer_verify_through_curated_api() {
    let graph_model = style_gate_constant_graph_model();

    // Phase 1: build — the consumer can construct and build a GraphNetwork.
    let graph = graph_model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("traced GraphModel should build from consumer crate (ny-cli)");

    // Phase 2: verify — the consumer can drive Verifier::verify_graph().
    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });

    // The constant prelude computes style_gate = [10, 20] (from the linear
    // transform of style=[10,20] with identity-like weights, reshaped and split).
    // With activation in [0,1]², the output is activation + [10, 20], so
    // output bounds are [10,11] × [20,21].
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)],
        vec![Bound::new(10.0, 11.0), Bound::new(20.0, 21.0)],
        Some(5_000),
        Some(vec![1, 1, 2]),
    )
    .expect("valid verification spec");

    let result = verifier
        .verify_graph(&graph, &spec)
        .expect("consumer-side verify_graph should succeed on folded traced model");

    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            assert_eq!(
                output_bounds,
                vec![Bound::new(10.0, 11.0), Bound::new(20.0, 21.0)],
                "consumer-side verification bounds must match the folded constant-prelude contract"
            );
            assert_eq!(
                actual_method.as_deref(),
                Some("Ibp"),
                "IBP should be the actual method used"
            );
        }
        other => panic!("expected consumer-side folded graph to verify, got {other:?}"),
    }
}

/// Consumer-side graph structure: constant prelude folds away from the consumer's
/// perspective, not just inside ny-api tests.
#[test]
fn test_graph_model_consumer_constant_folding_collapses_prelude() {
    let graph_model = style_gate_constant_graph_model();

    let graph = graph_model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("consumer-side GraphModel build should succeed");

    // The constant linear prelude should fold away.
    assert!(
        graph.node("style_linear").is_none(),
        "consumer-side: constant linear prelude should fold away"
    );

    // The Add node should remain as the sole computation node.
    let add = graph
        .node("mixed_add")
        .expect("consumer-side: mixed_add node should survive constant folding");

    assert_eq!(
        add.inputs(),
        &[NETWORK_INPUT.to_string()],
        "consumer-side: folded Add should only retain the live activation input"
    );
}

#[test]
fn test_graph_model_consumer_builds_adversarial_phoneme_encoder_3519() {
    let graph = adversarial_phoneme_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("consumer-side GraphModel should build the #3519 phoneme encoder");

    let projection = graph
        .node("phoneme_projection")
        .expect("projection node should exist for the #3519 phoneme encoder");
    assert_eq!(
        projection.inputs(),
        &["phoneme_relu".to_string()],
        "consumer-side GraphModel should preserve the Linear -> ReLU -> Linear phoneme path"
    );
    assert_eq!(
        graph.output_name(),
        "phoneme_projection",
        "consumer-side GraphModel should expose the final projection as the graph output"
    );
}

#[test]
fn test_graph_model_consumer_verifies_adversarial_phoneme_bounds_3519() {
    let graph = adversarial_phoneme_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("consumer-side GraphModel should build the #3519 phoneme encoder");
    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    });
    let spec = VerificationSpec::from_parts(
        adversarial_phoneme_input_bounds(),
        adversarial_phoneme_expected_bounds(),
        Some(5_000),
        Some(vec![2, 2]),
    )
    .expect("valid #3519 verification spec");

    let result = verifier
        .verify_graph(&graph, &spec)
        .expect("consumer-side CROWN should verify the #3519 phoneme encoder");

    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            assert_eq!(
                actual_method.as_deref(),
                Some("Crown"),
                "the #3519 consumer smoke should remain on CROWN"
            );
            let exact = adversarial_phoneme_exact_bounds();
            assert_eq!(output_bounds.len(), exact.len());
            for (got, exact) in output_bounds.iter().zip(&exact) {
                // Sound enclosure of the exact interval image...
                assert!(
                    got.lower() <= exact.lower() && got.upper() >= exact.upper(),
                    "CROWN bounds {got:?} must enclose the exact image {exact:?}"
                );
                // ...and tight up to the directed-rounding pad (f64 accumulate
                // + outward next_down/next_up casts on the sound reductions).
                assert!(
                    got.lower() >= exact.lower() - PHONEME_BOUND_ULP_PAD
                        && got.upper() <= exact.upper() + PHONEME_BOUND_ULP_PAD,
                    "CROWN bounds {got:?} drifted more than the directed-rounding pad from {exact:?}"
                );
            }
            assert!(
                output_bounds
                    .iter()
                    .all(|bound| bound.lower().is_finite() && bound.upper().is_finite()),
                "consumer-side adversarial phoneme bounds must remain finite"
            );
        }
        other => panic!("expected the #3519 phoneme encoder to verify, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_model_consumer_verifies_talker_like_multi_frozen_packet_3924() {
    let graph = talker_like_rotary_bias_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("consumer-side GraphModel should build the direct talker-like multi-frozen packet");
    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });
    let expected_bounds = repeated_bounds(4.0_f32, 5.0_f32, 8);
    let spec = VerificationSpec::from_parts(
        repeated_bounds(0.0_f32, 1.0_f32, 8),
        expected_bounds.clone(),
        Some(5_000),
        Some(vec![1, 4, 2]),
    )
    .expect("valid talker-like multi-frozen verification spec");

    let result = verifier
        .verify_graph(&graph, &spec)
        .expect("consumer-side verify_graph should succeed on the direct talker-like packet");

    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            assert_eq!(
                output_bounds,
                expected_bounds,
                "consumer-side talker-like packet should verify the exact [4, 5] elementwise bounds"
            );
            assert_eq!(
                actual_method.as_deref(),
                Some("Ibp"),
                "the direct talker-like consumer smoke should remain on IBP"
            );
        }
        other => panic!("expected the direct talker-like packet to verify, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Owned `GraphModel::new(...)` consumer smoke (#3958)
//
// Proves that a downstream consumer crate can use the owned `NetworkSpec`
// construction path — the shape a translator naturally has after it already
// produced owned LayerSpec, TensorSpec, weight, and metadata collections.
// ---------------------------------------------------------------------------

fn style_gate_constant_owned_graph_model() -> GraphModel {
    let network_spec = NetworkSpec {
        name: "style-gate-consumer-owned".to_string(),
        inputs: vec![TensorSpec {
            name: "activation".to_string(),
            shape: vec![1, 1, 2],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1, 1, 2],
            dtype: DataType::Float32,
        }],
        layers: vec![
            style_gate_linear_layer(),
            LayerSpec {
                name: "style_reshape".to_string(),
                layer_type: LayerType::Reshape,
                inputs: vec!["linear_out".to_string(), "reshape_shape".to_string()],
                outputs: vec!["reshaped".to_string()],
                weights: None,
                attributes: HashMap::new(),
            },
            style_gate_split_layer(),
            LayerSpec {
                name: "mixed_add".to_string(),
                layer_type: LayerType::Add,
                inputs: vec!["activation".to_string(), "style_gate".to_string()],
                outputs: vec!["out".to_string()],
                weights: None,
                attributes: HashMap::new(),
            },
        ],
        param_count: 0,
    };

    let mut weights = WeightStore::default();
    // Frozen auxiliary input stored unbatched
    weights.insert(
        "style".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).expect("valid style tensor"),
    );
    weights.insert(
        "linear_weight".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0])
            .expect("valid linear weight"),
    );
    weights.insert(
        "linear_bias".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).expect("valid bias"),
    );
    weights.insert(
        "reshape_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 2.0]).expect("valid reshape shape"),
    );
    weights.insert(
        "split_sizes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).expect("valid split sizes"),
    );

    GraphModel::new(network_spec, weights)
        .with_constant_tensors(HashSet::from(["style".to_string()]))
        .with_tensor_shapes(HashMap::from([
            ("activation".to_string(), vec![1, 1, 2]),
            ("linear_out".to_string(), vec![1, 4]),
            ("reshaped".to_string(), vec![1, 2, 2]),
            ("style_gate".to_string(), vec![1, 1, 2]),
            ("style_residual".to_string(), vec![1, 1, 2]),
            ("style".to_string(), vec![1, 2]),
        ]))
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_model_consumer_verify_owned_traced_contract_3958() {
    let graph = style_gate_constant_owned_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("owned GraphModel::new should build from consumer crate (ny-cli)");

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)],
        vec![Bound::new(10.0, 11.0), Bound::new(20.0, 21.0)],
        Some(5_000),
        Some(vec![1, 1, 2]),
    )
    .expect("valid verification spec");

    let result = verifier
        .verify_graph(&graph, &spec)
        .expect("owned consumer-side verify_graph should succeed on folded traced model");

    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            assert_eq!(
                output_bounds,
                vec![Bound::new(10.0, 11.0), Bound::new(20.0, 21.0)],
                "owned consumer-side bounds must match the builder-based contract"
            );
            assert_eq!(
                actual_method.as_deref(),
                Some("Ibp"),
                "IBP should be the actual method used"
            );
        }
        other => panic!("expected owned consumer-side graph to verify, got {other:?}"),
    }
}

#[test]
fn test_graph_model_consumer_owned_constant_folding_matches_builder_3958() {
    let owned_graph = style_gate_constant_owned_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("owned GraphModel::new should build from consumer crate");
    let builder_graph = style_gate_constant_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("builder GraphModel should build from consumer crate");

    // Both paths should fold away the constant linear prelude
    assert!(
        owned_graph.node("style_linear").is_none(),
        "owned: constant linear prelude should fold away"
    );
    assert!(
        builder_graph.node("style_linear").is_none(),
        "builder: constant linear prelude should fold away"
    );

    // Both paths should leave only the live activation input on mixed_add
    let owned_add = owned_graph
        .node("mixed_add")
        .expect("owned: mixed_add should survive folding");
    let builder_add = builder_graph
        .node("mixed_add")
        .expect("builder: mixed_add should survive folding");

    assert_eq!(
        owned_add.inputs(),
        builder_add.inputs(),
        "owned and builder Add node inputs should be identical after constant folding"
    );
    assert_eq!(
        owned_add.inputs(),
        &[NETWORK_INPUT.to_string()],
        "folded Add should retain only the live activation input"
    );
}
