// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Differential-translation oracle: one model, two build routes, required
//! agreement.
//!
//! Every model is loaded once and translated through both graph-build
//! routes:
//!
//! 1. **Direct**: `OnnxModel::to_graph_network()` (borrowed
//!    `GraphBuildInputs`).
//! 2. **GraphModel**: `OnnxModel::to_graph_model()` →
//!    `ny_build::GraphModel::build_graph_network()` — the owned contract
//!    that external traced producers target.
//!
//! The routes must produce structurally identical graphs, element-wise
//! matching IBP and CROWN bounds, and identical sign-certification
//! verdicts. Any divergence is a mistranslation on the verdict path.
//!
//! Tolerance policy: bounds are expected to agree bit-for-bit because both
//! routes feed the same specification into `ny_build::build_graph_network`.
//! The documented ceiling is [`DIFFERENTIAL_TOLERANCE`]; excess over it is
//! a real translation bug to investigate, never a reason to loosen the
//! tolerance.

use super::*;
use crate::load_onnx_bytes;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use prost::Message;

/// Maximum acceptable element-wise bound divergence between the two
/// routes: `1e-5`, applied relative to `max(|a|, |b|, 1)`.
///
/// Today the routes agree exactly; this ceiling only bounds acceptable
/// future refactoring noise (e.g. summation-order changes inside a shared
/// helper). A mismatch above it means the routes translated the model
/// differently and must be diagnosed, not absorbed.
const DIFFERENTIAL_TOLERANCE: f32 = 1e-5;

/// Sign-certification verdict for one output element — the shape of a
/// robustness verdict derived from output bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignVerdict {
    /// `lower >= 0`: certified nonnegative.
    NonNegative,
    /// `upper <= 0`: certified nonpositive.
    NonPositive,
    /// Bounds cross zero: no certificate.
    Undetermined,
}

fn sign_verdicts(bounds: &BoundedTensor, label: &str) -> Vec<SignVerdict> {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .enumerate()
        .map(|(idx, (&lower, &upper))| {
            assert!(
                lower.is_finite() && upper.is_finite() && lower <= upper,
                "{label}: bounds at index {idx} must be finite and ordered, got [{lower}, {upper}]"
            );
            if lower >= 0.0 {
                SignVerdict::NonNegative
            } else if upper <= 0.0 {
                SignVerdict::NonPositive
            } else {
                SignVerdict::Undetermined
            }
        })
        .collect()
}

/// Build the same loaded model through both translation routes.
///
/// Consumes the model so both routes are guaranteed to translate the
/// identical loaded instance (same weights, same graph metadata maps).
fn build_both_routes(model: OnnxModel, label: &str) -> (GraphNetwork, GraphNetwork) {
    let direct = model
        .to_graph_network()
        .unwrap_or_else(|e| panic!("{label}: direct GraphNetwork build failed: {e}"));
    let routed = model
        .to_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .unwrap_or_else(|e| panic!("{label}: GraphModel-route build failed: {e}"));
    (direct, routed)
}

/// Node-by-node structural parity: same output, same topological order,
/// and per node the same layer type and input wiring.
fn assert_structural_parity(direct: &GraphNetwork, routed: &GraphNetwork, label: &str) {
    assert_eq!(
        routed.output_name(),
        direct.output_name(),
        "{label}: output node mismatch between routes"
    );
    let direct_topo = direct
        .topological_sort()
        .unwrap_or_else(|e| panic!("{label}: direct topo sort failed: {e}"));
    let routed_topo = routed
        .topological_sort()
        .unwrap_or_else(|e| panic!("{label}: GraphModel-route topo sort failed: {e}"));
    assert_eq!(
        routed_topo, direct_topo,
        "{label}: topological node-name order mismatch between routes"
    );
    for name in &direct_topo {
        let direct_node = direct
            .node(name)
            .unwrap_or_else(|| panic!("{label}: node '{name}' missing from direct graph"));
        let routed_node = routed
            .node(name)
            .unwrap_or_else(|| panic!("{label}: node '{name}' missing from GraphModel route"));
        assert_eq!(
            routed_node.layer().layer_type(),
            direct_node.layer().layer_type(),
            "{label}: node '{name}' layer-type mismatch between routes"
        );
        assert_eq!(
            routed_node.inputs(),
            direct_node.inputs(),
            "{label}: node '{name}' input-wiring mismatch between routes"
        );
    }
}

/// Element-wise bound agreement within [`DIFFERENTIAL_TOLERANCE`].
fn assert_bounds_agree(routed: &BoundedTensor, direct: &BoundedTensor, label: &str) {
    assert_eq!(
        routed.lower().shape(),
        direct.lower().shape(),
        "{label}: lower-bound shape mismatch between routes"
    );
    assert_eq!(
        routed.upper().shape(),
        direct.upper().shape(),
        "{label}: upper-bound shape mismatch between routes"
    );
    for (side, routed_side, direct_side) in [
        ("lower", routed.lower(), direct.lower()),
        ("upper", routed.upper(), direct.upper()),
    ] {
        for (idx, (&routed_value, &direct_value)) in
            routed_side.iter().zip(direct_side.iter()).enumerate()
        {
            if routed_value.to_bits() == direct_value.to_bits() {
                continue;
            }
            let tol = DIFFERENTIAL_TOLERANCE * routed_value.abs().max(direct_value.abs()).max(1.0);
            assert!(
                (routed_value - direct_value).abs() <= tol,
                "{label}: {side}[{idx}] diverges between routes: \
                 GraphModel route={routed_value}, direct={direct_value}, tol={tol}"
            );
        }
    }
}

/// Feed the same input box through both routes with IBP and CROWN and
/// require structural, bound, and verdict agreement.
fn assert_differential_translation(model: OnnxModel, input: &BoundedTensor, label: &str) {
    let (direct, routed) = build_both_routes(model, label);
    assert_structural_parity(&direct, &routed, label);

    let direct_ibp = direct
        .propagate_ibp(input)
        .unwrap_or_else(|e| panic!("{label}: direct IBP failed: {e}"));
    let routed_ibp = routed
        .propagate_ibp(input)
        .unwrap_or_else(|e| panic!("{label}: GraphModel-route IBP failed: {e}"));
    assert_bounds_agree(&routed_ibp, &direct_ibp, &format!("{label} IBP"));
    assert_eq!(
        sign_verdicts(&routed_ibp, &format!("{label} GraphModel-route IBP")),
        sign_verdicts(&direct_ibp, &format!("{label} direct IBP")),
        "{label}: IBP sign-certification verdicts diverge between routes"
    );

    let direct_crown = direct
        .propagate_crown(input)
        .unwrap_or_else(|e| panic!("{label}: direct CROWN failed: {e}"));
    let routed_crown = routed
        .propagate_crown(input)
        .unwrap_or_else(|e| panic!("{label}: GraphModel-route CROWN failed: {e}"));
    assert_bounds_agree(&routed_crown, &direct_crown, &format!("{label} CROWN"));
    assert_eq!(
        sign_verdicts(&routed_crown, &format!("{label} GraphModel-route CROWN")),
        sign_verdicts(&direct_crown, &format!("{label} direct CROWN")),
        "{label}: CROWN sign-certification verdicts diverge between routes"
    );
}

fn uniform_box(shape: &[usize], lower: f32, upper: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(shape), lower);
    let upper = ArrayD::from_elem(IxDyn(shape), upper);
    BoundedTensor::new(lower, upper).expect("valid uniform input box")
}

/// Asymmetric, per-element varied box so channel/axis mix-ups cannot hide
/// behind a symmetric input.
fn patterned_box(shape: &[usize]) -> BoundedTensor {
    let n: usize = shape.iter().product();
    let lower_data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) / 4.0).collect();
    let upper_data: Vec<f32> = lower_data
        .iter()
        .enumerate()
        .map(|(i, &lower)| lower + 0.25 + (i % 3) as f32 * 0.1)
        .collect();
    let lower = ArrayD::from_shape_vec(IxDyn(shape), lower_data).expect("valid lower");
    let upper = ArrayD::from_shape_vec(IxDyn(shape), upper_data).expect("valid upper");
    BoundedTensor::new(lower, upper).expect("valid patterned input box")
}

/// MLP: simple_mlp.onnx is Linear(2→4) + ReLU + Linear(4→2).
#[ntest::timeout(60000)]
#[test]
fn test_differential_translation_simple_mlp() {
    let path = require_test_model("simple_mlp.onnx");
    let model = load_onnx(&path).expect("Failed to load simple_mlp.onnx");
    let input = uniform_box(&[2], 0.0, 1.0);
    assert_differential_translation(model, &input, "simple_mlp");
}

/// Spatial conv: conv_relu.onnx is Conv2d(1→2, 2×2) + ReLU with a 3D
/// spatial output, so CROWN runs its Patches-mode backward.
#[ntest::timeout(60000)]
#[test]
fn test_differential_translation_conv_relu() {
    let path = require_test_model("conv_relu.onnx");
    let model = load_onnx(&path).expect("Failed to load conv_relu.onnx");
    let input = uniform_box(&[1, 4, 4], -0.5, 0.5);
    assert_differential_translation(model, &input, "conv_relu");
}

/// Conv classifier: mnist_conv.onnx is Conv → ReLU → Flatten → Linear.
#[ntest::timeout(60000)]
#[test]
fn test_differential_translation_mnist_conv() {
    let path = require_test_model("mnist_conv.onnx");
    let model = load_onnx(&path).expect("Failed to load mnist_conv.onnx");
    let input = uniform_box(&[1, 8, 8], -0.1, 0.1);
    assert_differential_translation(model, &input, "mnist_conv");
}

/// Pooling: test_cnn_maxpool.onnx is Conv → ReLU → MaxPool → Flatten →
/// Gemm, driven with the same `[0.4, 0.6]` per-pixel box as its companion
/// vnnlib spec (a box root CROWN cannot decide, so both certified and
/// undetermined verdict regimes stay exercised).
#[ntest::timeout(60000)]
#[test]
fn test_differential_translation_cnn_maxpool() {
    let path = require_test_model("test_cnn_maxpool.onnx");
    let model = load_onnx(&path).expect("Failed to load test_cnn_maxpool.onnx");
    let input = uniform_box(&[1, 8, 8], 0.4, 0.6);
    assert_differential_translation(model, &input, "test_cnn_maxpool");
}

// --- standalone BatchNorm fixture -------------------------------------
//
// The tracked conv fixtures fold BatchNormalization into a preceding
// Conv/Gemm at load time, so no .onnx file in tests/models/ carries a live
// BatchNorm LayerSpec into the graph build. Mirror the fusion-test proto
// builders to make a model whose BatchNormalization consumes the graph
// input directly — the fold pass leaves it standalone.

fn tensor_value_info(name: &str, shape: &[i64], elem_type: i32) -> onnx_proto::ValueInfoProto {
    let dims = shape
        .iter()
        .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *dim,
            )),
        })
        .collect();
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type,
                shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

fn tensor_f32(name: &str, shape: &[i64], data: &[f32]) -> onnx_proto::TensorProto {
    assert_eq!(shape.iter().product::<i64>() as usize, data.len());
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 1,
        name: name.to_string(),
        raw_data: Vec::new(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn float_attr(name: &str, value: f32) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        f: value,
        i: 0,
        s: Vec::new(),
        t: None,
        r#type: onnx_proto::attribute_type::FLOAT,
        floats: Vec::new(),
        ints: Vec::new(),
    }
}

/// Build an input → BatchNormalization → ReLU model as ONNX bytes.
fn standalone_bn_relu_model_bytes() -> Vec<u8> {
    let batch_norm = onnx_proto::NodeProto {
        input: vec![
            "input".to_string(),
            "bn_scale".to_string(),
            "bn_bias".to_string(),
            "bn_mean".to_string(),
            "bn_var".to_string(),
        ],
        output: vec!["bn_out".to_string()],
        name: "bn".to_string(),
        op_type: "BatchNormalization".to_string(),
        domain: String::new(),
        attribute: vec![float_attr("epsilon", 1.0e-3)],
    };
    let relu = onnx_proto::NodeProto {
        input: vec!["bn_out".to_string()],
        output: vec!["out".to_string()],
        name: "relu".to_string(),
        op_type: "Relu".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    };

    let graph = onnx_proto::GraphProto {
        node: vec![batch_norm, relu],
        name: "standalone_bn_relu".to_string(),
        initializer: vec![
            tensor_f32("bn_scale", &[2], &[1.5, -0.75]),
            tensor_f32("bn_bias", &[2], &[0.25, 0.5]),
            tensor_f32("bn_mean", &[2], &[0.1, -0.2]),
            tensor_f32("bn_var", &[2], &[0.9, 1.6]),
        ],
        input: vec![tensor_value_info("input", &[1, 2, 2, 2], 1)],
        output: vec![tensor_value_info("out", &[1, 2, 2, 2], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model_proto = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        producer_name: "ny-onnx-fixture".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    };

    model_proto.encode_to_vec()
}

/// Normalization: a standalone BatchNorm (not foldable into a preceding
/// Conv/Gemm) followed by ReLU, on a squeezed `[C, H, W]` feature map.
#[ntest::timeout(60000)]
#[test]
fn test_differential_translation_standalone_batch_norm() {
    let bytes = standalone_bn_relu_model_bytes();
    let model =
        load_onnx_bytes("standalone_bn_relu.onnx", &bytes).expect("standalone bn+relu model loads");
    assert!(
        model
            .network
            .layers
            .iter()
            .any(|layer| layer.layer_type == ny_core::LayerType::BatchNorm),
        "fixture must keep a live BatchNorm layer (not folded away)"
    );
    let input = patterned_box(&[2, 2, 2]);
    assert_differential_translation(model, &input, "standalone_bn_relu");
}
