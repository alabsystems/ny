// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::ConvertContext;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ndarray::ArrayD;
use ny_core::{LayerType, NyError};
use ny_propagate::layers::{Layer, SeluLayer};
use ny_propagate::BoundPropagation;
use ny_tensor::BoundedTensor;
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

fn elu_spec(name: &str, alpha: f32) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Elu,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("alpha".to_string(), AttributeValue::Float(alpha))]),
    }
}

/// ELU alpha = 0 CONVERTS now: the layer degenerates to exactly ReLU
/// (`y = x` for `x >= 0`, `y = 0·(exp(x) − 1) = 0` below), which
/// `EluLayer::try_new` admits and IBP handles exactly. This test previously
/// pinned the opposite — that alpha = 0 was refused — which turned an ONNX
/// model the verifier can handle into a load failure.
#[test]
fn convert_elu_admits_zero_alpha_as_exact_relu() {
    let ctx = make_context();
    let layer = ctx
        .convert_elementwise(&elu_spec("elu_zero_alpha", 0.0))
        .expect("ELU alpha=0 must convert (exact ReLU)")
        .expect("ELU is an elementwise layer");
    assert!(
        matches!(layer, Layer::Elu(_)),
        "converted to an Elu layer, got {layer:?}"
    );
    // Behavioural pin (the fields are private): alpha = 0 must propagate as
    // ReLU, not as the alpha = 1 default. On [-2, 3] ReLU gives [0, 3];
    // ELU(alpha = 1) would give a lower endpoint near -0.865.
    let out = layer
        .propagate_ibp(
            &BoundedTensor::new(
                ArrayD::from_elem(ndarray::IxDyn(&[1]), -2.0f32),
                ArrayD::from_elem(ndarray::IxDyn(&[1]), 3.0f32),
            )
            .expect("valid bounds"),
        )
        .expect("ELU alpha=0 IBP");
    assert!(
        out.lower()[0] <= 0.0 && out.lower()[0] >= -f32::EPSILON,
        "alpha=0 negative branch is exactly 0 (ReLU), got lower {}",
        out.lower()[0]
    );
    assert!(
        out.upper()[0] >= 3.0 && out.upper()[0] <= 3.0 + 1e-5,
        "positive branch is the identity, got upper {}",
        out.upper()[0]
    );
}

/// Negative / non-finite ELU alpha stays refused: negative alpha makes the
/// negative branch DECREASING, so ELU is no longer monotone and the pointwise
/// IBP endpoint mapping would invert the bounds (#2779).
#[test]
fn convert_elu_rejects_invalid_alpha_2551() {
    let ctx = make_context();
    for (name, alpha) in [
        ("elu_neg_alpha", -1.0f32),
        ("elu_nan_alpha", f32::NAN),
        ("elu_inf_alpha", f32::INFINITY),
        ("elu_neg_inf_alpha", f32::NEG_INFINITY),
    ] {
        let err = ctx
            .convert_elementwise(&elu_spec(name, alpha))
            .expect_err("invalid ELU alpha should fail conversion");
        let NyError::ModelLoad(message) = err else {
            panic!("expected ModelLoad error for {name}, got {err:?}");
        };
        assert!(
            message.contains(&format!("Elu {name} invalid alpha"))
                && message.contains("must be finite and >= 0"),
            "{name}: got {message}"
        );
    }
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

fn hard_sigmoid_spec(name: &str, alpha: f32) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::HardSigmoid,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([
            ("alpha".to_string(), AttributeValue::Float(alpha)),
            ("beta".to_string(), AttributeValue::Float(0.5)),
        ]),
    }
}

/// HardSigmoid alpha = 0 CONVERTS now: the layer becomes the exact constant
/// `y = clamp(beta, 0, 1)`, which both the IBP and CROWN paths handle
/// exactly. Same pin-flip rationale as the ELU case above.
#[test]
fn convert_hard_sigmoid_admits_zero_alpha_as_exact_constant() {
    let ctx = make_context();
    let layer = ctx
        .convert_elementwise(&hard_sigmoid_spec("hard_sigmoid_zero_alpha", 0.0))
        .expect("HardSigmoid alpha=0 must convert (exact constant)")
        .expect("HardSigmoid is an elementwise layer");
    assert!(
        matches!(layer, Layer::HardSigmoid(_)),
        "converted to a HardSigmoid layer, got {layer:?}"
    );
    // Behavioural pin (private fields): alpha = 0 collapses to the exact
    // constant clamp(beta, 0, 1) = 0.5 for EVERY input, so a wide input box
    // must come back as the degenerate point [0.5, 0.5].
    let out = layer
        .propagate_ibp(
            &BoundedTensor::new(
                ArrayD::from_elem(ndarray::IxDyn(&[1]), -100.0f32),
                ArrayD::from_elem(ndarray::IxDyn(&[1]), 100.0f32),
            )
            .expect("valid bounds"),
        )
        .expect("HardSigmoid alpha=0 IBP");
    assert_eq!(out.lower()[0], 0.5, "constant clamp(beta, 0, 1)");
    assert_eq!(out.upper()[0], 0.5, "constant clamp(beta, 0, 1)");
}

/// Negative / non-finite HardSigmoid alpha stays refused: negative alpha
/// flips monotonicity, and the piecewise-region analysis in both the IBP and
/// CROWN paths assumes an increasing ramp.
#[test]
fn convert_hard_sigmoid_rejects_invalid_alpha_2551() {
    let ctx = make_context();
    for (name, alpha) in [
        ("hs_neg_alpha", -0.2f32),
        ("hs_nan_alpha", f32::NAN),
        ("hs_inf_alpha", f32::INFINITY),
        ("hs_neg_inf_alpha", f32::NEG_INFINITY),
    ] {
        let err = ctx
            .convert_elementwise(&hard_sigmoid_spec(name, alpha))
            .expect_err("invalid HardSigmoid alpha should fail conversion");
        let NyError::ModelLoad(message) = err else {
            panic!("expected ModelLoad error for {name}, got {err:?}");
        };
        assert!(
            message.contains(&format!("HardSigmoid {name} invalid alpha"))
                && message.contains("must be finite and >= 0"),
            "{name}: got {message}"
        );
    }
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
