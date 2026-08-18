// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::ConvertContext;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ndarray::ArrayD;
use ny_core::{LayerType, NyError};
use ny_propagate::layers::SeluLayer;
use std::collections::{HashMap, HashSet};

fn make_context() -> ConvertContext<'static> {
    let weights = Box::leak(Box::new(WeightStore::new()));
    let tensor_shapes = Box::leak(Box::new(HashMap::new()));
    let constant_tensors = Box::leak(Box::new(HashSet::new()));
    let evaluated_constants = Box::leak(Box::new(HashMap::<String, ArrayD<f32>>::new()));
    ConvertContext {
        weights,
        tensor_shapes,
        constant_tensors,
        evaluated_constants,
        model_unbatched: false,
    }
}

#[test]
fn convert_elu_rejects_invalid_alpha_2551() {
    let ctx = make_context();
    let spec = LayerSpec {
        name: "elu_bad_alpha".to_string(),
        layer_type: LayerType::Elu,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("alpha".to_string(), AttributeValue::Float(0.0))]),
    };

    let err = ctx
        .convert_elementwise(&spec)
        .expect_err("invalid ELU alpha should fail conversion");
    let NyError::ModelLoad(message) = err else {
        panic!("expected ModelLoad error, got {:?}", err);
    };
    assert!(message.contains("Elu elu_bad_alpha invalid alpha 0"));
}

#[test]
fn convert_clip_rejects_invalid_bounds_2551() {
    let ctx = make_context();
    let spec = LayerSpec {
        name: "clip_bad_bounds".to_string(),
        layer_type: LayerType::Clip,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([
            ("min".to_string(), AttributeValue::Float(2.0)),
            ("max".to_string(), AttributeValue::Float(1.0)),
        ]),
    };

    let err = ctx
        .convert_elementwise(&spec)
        .expect_err("invalid Clip bounds should fail conversion");
    let NyError::ModelLoad(message) = err else {
        panic!("expected ModelLoad error, got {:?}", err);
    };
    assert!(message.contains("Clip clip_bad_bounds invalid bounds min=2"));
}

#[test]
fn convert_celu_rejects_invalid_alpha_2551() {
    let ctx = make_context();
    let spec = LayerSpec {
        name: "celu_bad_alpha".to_string(),
        layer_type: LayerType::Celu,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("alpha".to_string(), AttributeValue::Float(0.0))]),
    };

    let err = ctx
        .convert_elementwise(&spec)
        .expect_err("invalid CELU alpha should fail conversion");
    let NyError::ModelLoad(message) = err else {
        panic!("expected ModelLoad error, got {:?}", err);
    };
    assert!(message.contains("Celu celu_bad_alpha invalid alpha 0"));
}

#[test]
fn convert_hard_sigmoid_rejects_invalid_alpha_2551() {
    let ctx = make_context();
    let spec = LayerSpec {
        name: "hard_sigmoid_bad_alpha".to_string(),
        layer_type: LayerType::HardSigmoid,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([
            ("alpha".to_string(), AttributeValue::Float(0.0)),
            ("beta".to_string(), AttributeValue::Float(0.5)),
        ]),
    };

    let err = ctx
        .convert_elementwise(&spec)
        .expect_err("invalid HardSigmoid alpha should fail conversion");
    let NyError::ModelLoad(message) = err else {
        panic!("expected ModelLoad error, got {:?}", err);
    };
    assert!(message.contains("HardSigmoid hard_sigmoid_bad_alpha invalid alpha 0"));
}

#[test]
fn convert_shrink_rejects_invalid_lambda_2551() {
    let ctx = make_context();
    let spec = LayerSpec {
        name: "shrink_bad_lambda".to_string(),
        layer_type: LayerType::Shrink,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("lambd".to_string(), AttributeValue::Float(-0.5))]),
    };

    let err = ctx
        .convert_elementwise(&spec)
        .expect_err("invalid Shrink lambda should fail conversion");
    let NyError::ModelLoad(message) = err else {
        panic!("expected ModelLoad error, got {:?}", err);
    };
    assert!(message.contains("Shrink shrink_bad_lambda invalid bias=0"));
}

fn silu_spec(beta: f32) -> LayerSpec {
    LayerSpec {
        name: "silu_beta".to_string(),
        layer_type: LayerType::SiLU,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("beta".to_string(), AttributeValue::Float(beta))]),
    }
}

fn swish_spec(alpha: AttributeValue) -> LayerSpec {
    LayerSpec {
        name: "swish_alpha".to_string(),
        layer_type: LayerType::SiLU,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("alpha".to_string(), alpha)]),
    }
}

#[test]
fn convert_silu_accepts_exact_unit_beta() {
    let layer = make_context()
        .convert_elementwise(&silu_spec(1.0))
        .expect("the implemented beta must load");
    assert!(layer.is_some());
}

#[test]
fn convert_silu_rejects_adjacent_beta_above_one() {
    let beta = f32::from_bits(1.0_f32.to_bits() + 1);
    let err = make_context()
        .convert_elementwise(&silu_spec(beta))
        .expect_err("an ignored non-unit beta must be rejected");
    assert!(err.to_string().contains("unsupported (expected 1.0)"));
}

#[test]
fn convert_silu_rejects_adjacent_beta_below_one() {
    let beta = f32::from_bits(1.0_f32.to_bits() - 1);
    let err = make_context()
        .convert_elementwise(&silu_spec(beta))
        .expect_err("an ignored non-unit beta must be rejected");
    assert!(err.to_string().contains("unsupported (expected 1.0)"));
}

#[test]
fn convert_standard_swish_accepts_exact_unit_alpha() {
    let layer = make_context()
        .convert_elementwise(&swish_spec(AttributeValue::Float(1.0)))
        .expect("the implemented Swish alpha must load");
    assert!(layer.is_some());
}

#[test]
fn convert_standard_swish_rejects_non_unit_or_wrong_typed_alpha() {
    let adjacent = f32::from_bits(1.0_f32.to_bits() + 1);
    let error = make_context()
        .convert_elementwise(&swish_spec(AttributeValue::Float(adjacent)))
        .expect_err("an ignored non-unit alpha must be rejected");
    assert!(error.to_string().contains("unsupported (expected 1.0)"));

    let error = make_context()
        .convert_elementwise(&swish_spec(AttributeValue::Int(1)))
        .expect_err("wrong-typed alpha must be rejected");
    assert!(error.to_string().contains("must be FLOAT"));
}

#[test]
fn convert_selu_authenticates_effective_coefficients() {
    let spec = |attributes| LayerSpec {
        name: "selu".to_string(),
        layer_type: LayerType::Selu,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes,
    };

    make_context()
        .convert_elementwise(&spec(HashMap::new()))
        .expect("ONNX Selu-6 defaults are implemented exactly");
    make_context()
        .convert_elementwise(&spec(HashMap::from([
            ("alpha".to_string(), AttributeValue::Float(SeluLayer::ALPHA)),
            (
                "gamma".to_string(),
                AttributeValue::Float(SeluLayer::LAMBDA),
            ),
        ])))
        .expect("explicit implemented coefficients");

    let adjacent_alpha = f32::from_bits(SeluLayer::ALPHA.to_bits() + 1);
    let error = make_context()
        .convert_elementwise(&spec(HashMap::from([(
            "alpha".to_string(),
            AttributeValue::Float(adjacent_alpha),
        )])))
        .expect_err("ignored custom alpha must be rejected");
    assert!(error.to_string().contains("coefficients"));

    let error = make_context()
        .convert_elementwise(&spec(HashMap::from([(
            "gamma".to_string(),
            AttributeValue::Int(1),
        )])))
        .expect_err("wrong-typed gamma must be rejected");
    assert!(error.to_string().contains("must be FLOAT"));
}
