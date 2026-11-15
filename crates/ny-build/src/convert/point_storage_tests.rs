// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Construction-policy arrival and refusal controls; no proof is minted here.
use super::ConvertContext;
use crate::{
    AttributeValue, DataType, GraphModel, GraphNetworkOptions, LayerSpec, LinearStoragePolicy,
    Network, TensorSpec, WeightStore,
};
use ndarray::{array, Array2};
use ny_core::{LayerType, NyError};
use ny_propagate::certify::work::WorkBudget;
use ny_propagate::layers::{PointLinearBuildPolicy, PointLinearLayer};
use ny_propagate::Layer;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

fn budget(bytes: usize) -> WorkBudget {
    WorkBudget::new(None, 1 << 25, bytes, 1 << 32, 1_024, Duration::ZERO).unwrap()
}
fn policy(ledger: &WorkBudget) -> LinearStoragePolicy<'_> {
    LinearStoragePolicy::OriginalPoint(PointLinearBuildPolicy {
        budget: ledger,
        deadline: None,
        max_resident_bytes: 1 << 30,
        max_scratch_bytes: 1 << 25,
        max_work: 1 << 32,
    })
}
fn spec(kind: LayerType, inputs: &[&str]) -> LayerSpec {
    LayerSpec {
        name: "projection".into(),
        layer_type: kind,
        inputs: inputs.iter().map(|s| (*s).into()).collect(),
        outputs: vec!["y".into()],
        weights: None,
        attributes: HashMap::new(),
    }
}
fn fixture() -> GraphModel {
    let mut weights = WeightStore::new();
    weights.insert(
        "w".into(),
        Array2::from_shape_fn((64, 32), |(r, c)| (r as f32 - c as f32) * 0.03125).into_dyn(),
    );
    weights.insert("b".into(), ndarray::Array1::from_elem(64, 0.125).into_dyn());
    let mut projection = spec(LayerType::Linear, &["x", "w", "b"]);
    projection
        .attributes
        .insert("transB".into(), AttributeValue::Int(1));
    let tensor = |name: &str, shape: &[i64]| TensorSpec {
        name: name.into(),
        shape: shape.to_vec(),
        dtype: DataType::Float32,
    };
    GraphModel::new(
        Network {
            name: "compact-control".into(),
            inputs: vec![tensor("x", &[1, 32])],
            outputs: vec![tensor("y", &[1, 64])],
            layers: vec![projection],
            param_count: 64 * 32 + 64,
        },
        weights,
    )
    .with_tensor_shapes(HashMap::from([
        ("x".into(), vec![1, 32]),
        ("y".into(), vec![1, 64]),
    ]))
}

#[test]
fn decoder_n10_typed_builder_keeps_eager_default_and_shares_compact_owner() {
    let source = fixture();
    let eager = source
        .build_graph_network(GraphNetworkOptions::default())
        .unwrap();
    assert!(matches!(
        eager.node("projection").unwrap().layer(),
        Layer::Linear(_)
    ));
    let ledger = budget(1 << 24);
    let compact = crate::build_graph_network_with_linear_storage(
        &source.graph_build_inputs(),
        GraphNetworkOptions::default(),
        policy(&ledger),
    )
    .unwrap();
    let Layer::PointLinear(original) = compact.node("projection").unwrap().layer() else {
        panic!("typed policy did not reach the pre-copy conversion boundary")
    };
    assert_eq!(original.weight().shape(), [64, 32]);
    for (&actual, &source) in original
        .weight()
        .iter()
        .zip(source.weights.get("w").unwrap())
    {
        assert_eq!(actual.to_bits(), source.to_bits());
    }
    let held = original.clone();
    let live = ledger.receipt().unwrap().live_persistent_bytes;
    assert_eq!(live, original.retained_bytes());
    let Layer::Linear(eager) = eager.node("projection").unwrap().layer() else {
        unreachable!()
    };
    assert!(original.retained_bytes() < eager.retained_bytes().unwrap());
    drop(compact);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, live);
    drop(held);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n10_builder_refuses_owner_cap_before_any_coefficients_are_retained() {
    let source = fixture();
    let request = PointLinearLayer::construction_request(64, 32, true).unwrap();
    let ledger = budget(request.persistent_bytes - 1);
    assert!(crate::build_graph_network_with_linear_storage(
        &source.graph_build_inputs(),
        GraphNetworkOptions::default(),
        policy(&ledger),
    )
    .is_err());
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.attempted, 1);
    assert_eq!(receipt.admitted, 0);
    assert_eq!(receipt.work_reserved, 0);
    assert_eq!(receipt.live_persistent_bytes, 0);
    assert_eq!(receipt.live_scratch_bytes, 0);
}

#[test]
fn decoder_n10_conversion_preserves_gemm_orientation_and_scalar_bias_bits() {
    let ledger = budget(1 << 24);
    let source = array![[0.0_f32, -0.0], [f32::from_bits(1), -1.0], [3.0, 5.0]];
    for trans_b in [0, 1] {
        let mut weights = WeightStore::new();
        weights.insert(
            "w".into(),
            if trans_b == 1 {
                source.clone().into_dyn()
            } else {
                source.t().to_owned().into_dyn()
            },
        );
        weights.insert("b".into(), ndarray::arr0(-0.0_f32).into_dyn());
        let shapes = HashMap::new();
        let constants = HashSet::new();
        let context =
            ConvertContext::new(&weights, &shapes, &constants).with_linear_storage(policy(&ledger));
        let mut projection = spec(LayerType::Linear, &["x", "w", "b"]);
        projection
            .attributes
            .insert("transB".into(), AttributeValue::Int(trans_b));
        let Layer::PointLinear(compact) = context.convert_layer(&projection).unwrap() else {
            panic!("no compact affine arrival")
        };
        assert_eq!(compact.weight().shape(), [3, 2]);
        for (&actual, &expected) in compact.weight().iter().zip(source.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert!(compact
            .bias()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == (-0.0_f32).to_bits()));
    }
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n10_unqualified_static_and_quantized_consumers_refuse_dynamic_matmul_survives() {
    let ledger = budget(1 << 24);
    let mut weights = WeightStore::new();
    weights.insert("w".into(), Array2::<f32>::eye(2).into_dyn());
    let shapes = HashMap::new();
    let constants = HashSet::new();
    let context =
        ConvertContext::new(&weights, &shapes, &constants).with_linear_storage(policy(&ledger));
    let before = ledger.receipt().unwrap();
    for rejected in [
        spec(LayerType::MatMul, &["x", "w"]),
        spec(LayerType::QuantizeLinear, &["x", "scale"]),
        spec(LayerType::DequantizeLinear, &["x", "scale"]),
    ] {
        assert!(matches!(
            context.convert_layer(&rejected),
            Err(NyError::UnsupportedConfiguration(_))
        ));
    }
    assert!(matches!(
        context.convert_layer(&spec(LayerType::MatMul, &["x", "activation_b"])),
        Ok(Layer::BilinearCrown(_))
    ));
    assert_eq!(
        ledger.receipt().unwrap(),
        before,
        "no refused derived view can mint another allowance"
    );
}
