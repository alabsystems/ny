// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Compile-contract test for the feature-gated `propagate` facade.
//!
//! This test verifies that the curated external API surface exposed through
//! `ny_api` with the `propagate` feature remains present and importable for
//! intended external consumers. If any of these imports break, the curated
//! facade contract is violated.
#![allow(unused_qualifications)]

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use std::collections::HashMap;

// Always-on surface (no feature gate needed).
use ny_api::VerificationBoundsSource;
use ny_api::VerificationSoundnessMode;
use ny_api::{Bound, BoundedTensor, GenericBounds, HeuristicUsed, MethodUsed, SoundnessProvenance};
use ny_api::{VerificationResult, VerificationSpec};

// Feature-gated surface (requires `propagate` feature, enabled in test config).
use ny_api::composition::{
    check_ducking_snr, check_priority_routing, check_spatial_ild, compose_linear_mix,
    BoundCertificate, BoundCertificationResult, BoundProvenance, MixerSpec, PipelineCertificate,
    PipelineStage, PipelineVerifier, PropertyResult,
};
use ny_api::graph::{GraphNetwork, GraphNode, SequentialNetwork, NETWORK_INPUT};
use ny_api::layers::{
    AttentionMask, BoundPropagation, Conv1dLayer, Conv2dLayer, ConvTranspose1dLayer, CumsumLayer,
    GELULayer, GeluApproximation, Layer, LayerNormCrownMode, LayerNormLayer, LinearLayer,
    ReLULayer, RmsNormLayer, SelfAttentionLayer, SiLULayer, SigmoidLayer, SoftmaxLayer,
};
use ny_api::model::{
    AttributeValue, CompoundNodePolicy, DataType, GraphModel, GraphModelBuilder,
    GraphNetworkOptions, LayerSpec, LayerType, MissingOutputPolicy, NetworkSpec, TensorSpec,
    WeightRef, WeightStore,
};
use ny_api::parallel::{
    verify_parallel, verify_parallel_with_engine, verify_parallel_with_method,
    verify_parallel_with_method_and_engine, ParallelConfig, ParallelVerificationResult,
    ParallelVerifier,
};
use ny_api::prelude as api_prelude;
use ny_api::verify::{
    GemmEngine, NaiveCpuGemmEngine, PropagationConfig, PropagationMethod, Verifier,
};

type ParallelWithEngineFn = fn(
    &GraphNetwork,
    &BoundedTensor,
    usize,
    std::sync::Arc<dyn GemmEngine>,
) -> ny_api::Result<BoundedTensor>;
type ParallelWithPreludeEngineFn = fn(
    &api_prelude::GraphNetwork,
    &BoundedTensor,
    usize,
    std::sync::Arc<dyn api_prelude::GemmEngine>,
) -> ny_api::Result<BoundedTensor>;
type ParallelWithMethodAndEngineFn = fn(
    &GraphNetwork,
    &BoundedTensor,
    usize,
    PropagationMethod,
    std::sync::Arc<dyn GemmEngine>,
) -> ny_api::Result<BoundedTensor>;
type ComposeLinearMixFn =
    fn(&[BoundCertificate], &MixerSpec) -> ny_api::Result<(BoundedTensor, BoundedTensor)>;
type SpatialIldFn =
    fn(&BoundedTensor, &BoundedTensor, (f32, f32), (f32, f32), f64) -> PropertyResult;

/// Compile-contract: all curated facade types are importable and nameable.
///
/// This test does not exercise runtime behavior. Its purpose is to fail at
/// compile time if any curated path is removed or renamed.
#[test]
fn facade_types_are_importable() {
    // Always-on types
    fn _bound(_: Bound) {}
    fn _bounded_tensor(_: BoundedTensor) {}
    fn _generic_bounds(_: GenericBounds<f64>) {}
    fn _heuristic(_: HeuristicUsed) {}
    fn _method_used(_: MethodUsed) {}
    fn _provenance(_: SoundnessProvenance) {}
    fn _soundness_mode(_: VerificationSoundnessMode) {}
    fn _result(_: VerificationResult) {}
    fn _spec(_: VerificationSpec) {}
    fn _bounds_source<T: VerificationBoundsSource>(_: &T) {}

    // Composition types
    fn _bound_certification_result(_: BoundCertificationResult) {}
    fn _bound_certificate(_: BoundCertificate) {}
    fn _bound_provenance(_: BoundProvenance) {}
    fn _mixer_spec(_: MixerSpec) {}
    fn _pipeline_certificate(_: PipelineCertificate) {}
    fn _pipeline_stage(_: PipelineStage) {}
    fn _pipeline_verifier(_: PipelineVerifier) {}
    fn _property_result(_: PropertyResult) {}

    // Graph types
    fn _graph_network(_: GraphNetwork) {}
    fn _graph_node(_: GraphNode) {}
    fn _sequential_network(_: SequentialNetwork) {}
    let _ = NETWORK_INPUT;

    // Model-build types
    fn _attribute_value(_: AttributeValue) {}
    fn _data_type(_: DataType) {}
    fn _graph_model(_: GraphModel) {}
    fn _graph_model_builder(_: GraphModelBuilder) {}
    fn _network_spec(_: NetworkSpec) {}
    fn _layer_spec(_: LayerSpec) {}
    fn _layer_type(_: LayerType) {}
    fn _tensor_spec(_: TensorSpec) {}
    fn _weight_ref(_: WeightRef) {}
    fn _compound_node_policy(_: CompoundNodePolicy) {}
    let _ = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    let _ = MissingOutputPolicy::Error;
    let _ = WeightStore::new();

    // Layer types
    fn _attention_mask(_: AttentionMask) {}
    fn _conv1d(_: Conv1dLayer) {}
    fn _conv2d(_: Conv2dLayer) {}
    fn _conv_transpose1d(_: ConvTranspose1dLayer) {}
    fn _cumsum(_: CumsumLayer) {}
    fn _gelu(_: GELULayer) {}
    fn _gelu_approximation(_: GeluApproximation) {}
    fn _layer(_: Layer) {}
    fn _layer_norm_crown_mode(_: LayerNormCrownMode) {}
    fn _layer_norm(_: LayerNormLayer) {}
    fn _linear(_: LinearLayer) {}
    fn _relu(_: ReLULayer) {}
    fn _rms_norm(_: RmsNormLayer) {}
    fn _self_attention(_: SelfAttentionLayer) {}
    fn _silu(_: SiLULayer) {}
    fn _sigmoid(_: SigmoidLayer) {}
    fn _softmax(_: SoftmaxLayer) {}

    // Verify types
    fn _engine(_: NaiveCpuGemmEngine) {}
    fn _config(_: PropagationConfig) {}
    fn _method(_: PropagationMethod) {}

    // Parallel types
    fn _parallel_config(_: ParallelConfig) {}
    fn _parallel_result(_: ParallelVerificationResult) {}

    // Struct types are importable
    fn _verifier(_: Verifier) {}
    fn _parallel_verifier(_: ParallelVerifier) {}

    // Trait is importable
    fn _bound_prop<T: BoundPropagation>(_: T) {}

    // Prelude types stay importable through the curated wildcard surface.
    fn _prelude_graph_network(_: api_prelude::GraphNetwork) {}
    fn _prelude_graph_node(_: api_prelude::GraphNode) {}
    fn _prelude_sequential_network(_: api_prelude::SequentialNetwork) {}
    let _ = api_prelude::NETWORK_INPUT;
    fn _prelude_layer<T: api_prelude::BoundPropagation>(_: T) {}
    fn _prelude_layer_enum(_: api_prelude::Layer) {}
    fn _prelude_graph_model(_: api_prelude::GraphModel) {}
    fn _prelude_graph_model_builder(_: api_prelude::GraphModelBuilder) {}
    fn _prelude_graph_network_options(_: api_prelude::GraphNetworkOptions) {}
    fn _prelude_layer_spec(_: api_prelude::LayerSpec) {}
    fn _prelude_layer_type(_: api_prelude::LayerType) {}
    fn _prelude_missing_output_policy(_: api_prelude::MissingOutputPolicy) {}
    fn _prelude_network_spec(_: api_prelude::NetworkSpec) {}
    fn _prelude_tensor_spec(_: api_prelude::TensorSpec) {}
    fn _prelude_weight_ref(_: api_prelude::WeightRef) {}
    fn _prelude_weight_store(_: api_prelude::WeightStore) {}
    fn _prelude_unknown_reason(_: api_prelude::UnknownReason) {}
    fn _prelude_engine(_: api_prelude::NaiveCpuGemmEngine) {}
    fn _prelude_config(_: api_prelude::PropagationConfig) {}
    fn _prelude_method(_: api_prelude::PropagationMethod) {}
    fn _prelude_verifier(_: api_prelude::Verifier) {}
    fn _prelude_parallel_config(_: api_prelude::ParallelConfig) {}
    fn _prelude_parallel_result(_: api_prelude::ParallelVerificationResult) {}
    fn _prelude_parallel_verifier(_: api_prelude::ParallelVerifier) {}

    // Free functions are importable
    let _: fn(&GraphNetwork, &BoundedTensor, usize) -> ny_api::Result<BoundedTensor> =
        verify_parallel;
    let _: ParallelWithEngineFn = verify_parallel_with_engine;
    let _: ParallelWithPreludeEngineFn = verify_parallel_with_engine;
    let _: fn(
        &GraphNetwork,
        &BoundedTensor,
        usize,
        PropagationMethod,
    ) -> ny_api::Result<BoundedTensor> = verify_parallel_with_method;
    let _: ParallelWithMethodAndEngineFn = verify_parallel_with_method_and_engine;
    let _: fn(&api_prelude::GraphNetwork, &BoundedTensor, usize) -> ny_api::Result<BoundedTensor> =
        api_prelude::verify_parallel;
    let _: ParallelWithPreludeEngineFn = api_prelude::verify_parallel_with_engine;
    let _: fn(
        &api_prelude::GraphNetwork,
        &BoundedTensor,
        usize,
        api_prelude::PropagationMethod,
    ) -> ny_api::Result<BoundedTensor> = api_prelude::verify_parallel_with_method;
    let _: fn(
        &api_prelude::GraphNetwork,
        &BoundedTensor,
        usize,
        api_prelude::PropagationMethod,
        std::sync::Arc<dyn api_prelude::GemmEngine>,
    ) -> ny_api::Result<BoundedTensor> = api_prelude::verify_parallel_with_method_and_engine;
    let _: ComposeLinearMixFn = compose_linear_mix;
    let _: fn(&BoundedTensor, &[&BoundedTensor]) -> PropertyResult = check_priority_routing;
    let _: fn(&BoundedTensor, &BoundedTensor, f64) -> PropertyResult = check_ducking_snr;
    let _: SpatialIldFn = check_spatial_ild;
    let _ = GraphModelBuilder::new("facade-builder");
    let _ = GraphModelBuilder::new("frozen-facade-builder").frozen_input(
        "style",
        &[1, 2],
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("valid frozen input"),
    );
    let _: fn(NetworkSpec, WeightStore) -> GraphModel = GraphModel::new;
    let _: fn(&GraphModel, GraphNetworkOptions) -> ny_api::Result<GraphNetwork> =
        GraphModel::build_graph_network;
}

fn tensor_spec(name: &str, shape: &[i64]) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: DataType::Float32,
    }
}

fn certified_bounds(result: BoundCertificationResult, label: &str) -> BoundCertificate {
    match result {
        BoundCertificationResult::Certified(cert) => cert,
        other => panic!("expected Certified {label} bounds, got {other:?}"),
    }
}

fn assert_pipeline_stage_contract(
    pipeline_cert: &PipelineCertificate,
    decoder_input: &BoundedTensor,
) {
    assert_eq!(pipeline_cert.stages().len(), 2);
    assert_eq!(pipeline_cert.overall_provenance(), BoundProvenance::Ibp);
    assert_eq!(
        pipeline_cert.overall_soundness().mode(),
        VerificationSoundnessMode::Sound
    );
    assert!(
        pipeline_cert
            .overall_soundness()
            .heuristics_used()
            .is_empty(),
        "sound pipeline should not accumulate heuristics, got {:?}",
        pipeline_cert.overall_soundness().heuristics_used()
    );
    assert_eq!(
        pipeline_cert.stages()[1].certificate().model_id(),
        "decoder",
        "final pipeline should preserve per-stage certificates"
    );
    assert_eq!(
        pipeline_cert.stages()[1].input_bounds().lower(),
        decoder_input.lower(),
        "pipeline certificate should preserve downstream input witnesses"
    );
    assert_eq!(pipeline_cert.final_bounds().shape(), &[1]);
}

fn make_linear_network(weight: f32, bias: f32) -> SequentialNetwork {
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[weight]]), Some(arr1(&[bias]))).expect("valid linear layer"),
    ));
    network
}

fn assert_sound_metadata(soundness: &SoundnessProvenance, label: &str) {
    assert_eq!(soundness.mode(), VerificationSoundnessMode::Sound);
    assert!(
        soundness.heuristics_used().is_empty(),
        "{label} should stay sound, got {:?}",
        soundness.heuristics_used()
    );
}

fn assert_sound_certificate(
    certificate: &BoundCertificate,
    actual_method: MethodUsed,
    provenance: BoundProvenance,
    label: &str,
) {
    assert_eq!(certificate.actual_method(), &actual_method);
    assert_eq!(certificate.provenance(), provenance);
    assert_sound_metadata(certificate.soundness(), label);
}

fn assert_verified_single_output(result: VerificationResult, label: &str) {
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
        }
        other => panic!("expected Verified {label} result, got {other:?}"),
    }
}

#[test]
fn graph_model_builds_graph_network_via_curated_facade() {
    let network = NetworkSpec {
        name: "relu".to_string(),
        inputs: vec![tensor_spec("input", &[1, 2])],
        outputs: vec![tensor_spec("relu_out", &[1, 2])],
        layers: vec![LayerSpec {
            name: "relu".to_string(),
            layer_type: LayerType::ReLU,
            inputs: vec!["input".to_string()],
            outputs: vec!["relu_out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }],
        param_count: 0,
    };

    let graph_model = GraphModel::new(network, WeightStore::new())
        .with_tensor_producer(HashMap::from([(
            "relu_out".to_string(),
            "input".to_string(),
        )]))
        .with_tensor_shapes(HashMap::from([
            ("input".to_string(), vec![1, 2]),
            ("relu_out".to_string(), vec![1, 2]),
        ]));

    let graph = graph_model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("graph model should build through ny_api facade");
    let relu = graph.node("relu").expect("relu node should exist");
    assert_eq!(
        relu.inputs(),
        &[NETWORK_INPUT.to_string()],
        "facade-built graph should still route declared inputs through NETWORK_INPUT"
    );
}

#[test]
fn composition_facade_certifies_and_reuses_bounds_3920() {
    let encoder = make_linear_network(1.0, 0.5);
    let decoder = make_linear_network(1.0, -0.25);

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        use_gpu: false,
        ..Default::default()
    });

    let encoder_input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid encoder input bounds");
    let encoder_cert = verifier
        .certify_network_bounds("encoder", &encoder, &encoder_input, None)
        .expect("encoder certification should succeed");

    let encoder_cert = certified_bounds(encoder_cert, "encoder");
    assert_sound_certificate(
        &encoder_cert,
        MethodUsed::Ibp,
        BoundProvenance::Ibp,
        "linear encoder certification",
    );

    let decoder_spec = ny_api::SpecBuilder::default()
        .try_input_source(&encoder_cert)
        .expect("certificate should plug directly into downstream input builder via VerificationBoundsSource")
        .output_bounds(vec![Bound::new_allow_infinite(
            f32::NEG_INFINITY,
            f32::INFINITY,
        )])
        .build()
        .expect("downstream spec should build");

    assert_eq!(decoder_spec.input_shape(), Some(&[1][..]));
    assert_eq!(
        decoder_spec.input_bounds()[0],
        Bound::new(
            encoder_cert.output_bounds().lower().as_slice().unwrap()[0],
            encoder_cert.output_bounds().upper().as_slice().unwrap()[0]
        )
    );

    let decoder_result = verifier
        .verify(&decoder, &decoder_spec)
        .expect("decoder verification should accept source-built spec");
    assert_verified_single_output(decoder_result, "decoder");

    let decoder_input = encoder_cert.output_bounds().clone();
    let decoder_cert = verifier
        .certify_network_bounds("decoder", &decoder, &decoder_input, None)
        .expect("decoder certification should succeed");
    let decoder_cert = certified_bounds(decoder_cert, "decoder");
    assert_sound_certificate(
        &decoder_cert,
        MethodUsed::Ibp,
        BoundProvenance::Ibp,
        "linear decoder certification",
    );

    let mut pipeline = PipelineVerifier::new();
    pipeline
        .push_stage(encoder_input, encoder_cert)
        .expect("pipeline should accept first certified stage");
    pipeline
        .push_stage(decoder_input.clone(), decoder_cert)
        .expect("pipeline should accept downstream certified stage");

    let pipeline_cert = pipeline
        .finalize()
        .expect("pipeline certificate should finalize from certified stages");
    assert_pipeline_stage_contract(&pipeline_cert, &decoder_input);
}

#[test]
fn composition_facade_exposes_timeout_metadata_3920() {
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).expect("valid layer"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        use_gpu: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid input bounds");

    let certification = verifier
        .certify_network_bounds("timeout_stage", &network, &input_bounds, Some(0))
        .expect("timeout should remain structured control flow");

    match certification {
        BoundCertificationResult::Timeout {
            partial,
            actual_method,
            soundness,
        } => {
            assert!(
                partial.is_none(),
                "expected no partial certificate at timeout"
            );
            assert_eq!(actual_method, MethodUsed::Crown);
            assert_sound_metadata(&soundness, "timeout metadata");
        }
        other => panic!("expected structured timeout metadata, got {other:?}"),
    }
}
