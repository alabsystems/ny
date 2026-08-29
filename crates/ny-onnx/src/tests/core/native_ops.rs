// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end front-end parity for ONNX ops whose ny layer already existed and
//! was reachable only through the trace bridge: `Trilu` and
//! `ScatterElements(reduction="add")`.
//!
//! Every acceptance test proves the SAME two things:
//!
//! 1. the model LOADS to the intended `LayerType` (the layer the trace bridge
//!    reaches), and
//! 2. ny's IBP at a concrete (zero-width) input ENCLOSES an independent
//!    runtime's forward over the identical model bytes — here with zero slack,
//!    since both lowerings are pure masking / accumulation.
//!
//! Every refused configuration has a pin naming it, so a later change cannot
//! quietly widen the admitted set.

use super::onnx_fixture::*;
use crate::load_onnx_bytes;
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_tensor::BoundedTensor;

fn concrete(shape: &[usize], data: Vec<f32>) -> BoundedTensor {
    let array = ArrayD::from_shape_vec(IxDyn(shape), data).expect("fixture shape");
    BoundedTensor::new(array.clone(), array).expect("a concrete tensor is a valid bound")
}

/// Assert ny's bound is CONCRETE (zero width) and equals `expected`.
fn assert_exact(output: &BoundedTensor, expected: &[f32], what: &str) {
    let lower: Vec<f32> = output.lower().iter().copied().collect();
    let upper: Vec<f32> = output.upper().iter().copied().collect();
    assert_eq!(lower, upper, "{what}: bound must stay concrete");
    assert_eq!(lower, expected, "{what}: value mismatch");
}

// ---------------------------------------------------------------------------
// Trilu
// ---------------------------------------------------------------------------

fn trilu_model(upper: Option<i64>, k: Option<i64>) -> Vec<u8> {
    let mut inputs = vec!["x"];
    let mut initializer = Vec::new();
    if let Some(k) = k {
        inputs.push("k");
        initializer.push(i64_initializer("k", &[], &[k]));
    }
    model_bytes(
        "trilu",
        17,
        vec![node(
            "mask",
            "Trilu",
            &inputs,
            &["out"],
            upper.into_iter().map(|v| int_attr("upper", v)).collect(),
        )],
        initializer,
        vec![value_info("x", &[1, 3, 3])],
        vec![value_info("out", &[1, 3, 3])],
    )
}

#[ntest::timeout(20000)]
#[test]
fn trilu_on_live_data_loads_to_the_triangular_layers() {
    for (upper, k, expected_type) in [
        (Some(1), None, LayerType::Triu),
        (Some(0), None, LayerType::Tril),
        (None, None, LayerType::Triu),
        (Some(1), Some(1), LayerType::Triu),
        (Some(0), Some(-1), LayerType::Tril),
    ] {
        let bytes = trilu_model(upper, k);
        let model = load_onnx_bytes("trilu.onnx", &bytes)
            .unwrap_or_else(|err| panic!("Trilu(upper={upper:?}, k={k:?}) must load: {err}"));
        assert_eq!(model.network.layers.len(), 1);
        assert_eq!(model.network.layers[0].layer_type, expected_type);
    }
}

#[cfg(feature = "ort")]
#[ntest::timeout(20000)]
#[test]
fn trilu_bounds_match_an_independent_runtime_on_the_same_bytes() {
    let values: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    for (upper, k) in [
        (Some(1), None),
        (Some(0), None),
        (None, None),
        (Some(1), Some(1)),
        (Some(1), Some(-1)),
        (Some(0), Some(-1)),
        (Some(0), Some(2)),
    ] {
        let bytes = trilu_model(upper, k);
        let model = load_onnx_bytes("trilu.onnx", &bytes).expect("Trilu must load");
        let graph = model
            .to_graph_network()
            .expect("Trilu must build a graph network");
        let output = graph
            .propagate_ibp(&concrete(&[3, 3], values.clone()))
            .expect("Trilu IBP must succeed");

        let reference = trusted_forward(
            &bytes,
            &ArrayD::from_shape_vec(IxDyn(&[1, 3, 3]), values.clone()).unwrap(),
        );
        assert_exact(
            &output,
            &reference,
            &format!("Trilu(upper={upper:?}, k={k:?})"),
        );
    }
}

#[ntest::timeout(20000)]
#[test]
fn trilu_with_a_runtime_diagonal_is_refused() {
    // `k` is a graph INPUT, not an initializer: the retained-entry set would be
    // data-dependent, and ny's fixed triangular mask cannot enclose that.
    //
    // The input is declared FLOAT because ny refuses an INT64 graph input
    // outright, several layers earlier ("accepts only FLOAT32 inputs") — which
    // would pin someone else's gate rather than this one. A FLOAT `k` reaches
    // the canonicalizer and exercises exactly the intended refusal: no
    // authenticated INT64 constant is available for the diagonal.
    let bytes = model_bytes(
        "trilu_dynamic_k",
        17,
        vec![node(
            "mask",
            "Trilu",
            &["x", "k"],
            &["out"],
            vec![int_attr("upper", 1)],
        )],
        Vec::new(),
        vec![value_info("x", &[1, 3, 3]), value_info("k", &[])],
        vec![value_info("out", &[1, 3, 3])],
    );
    let error = load_onnx_bytes("trilu_dynamic_k.onnx", &bytes)
        .expect_err("a data-dependent Trilu diagonal must be refused");
    let message = error.to_string();
    assert!(
        message.contains("Trilu") && message.contains("INT64"),
        "the refusal must name the op and the missing exact INT64 diagonal, got: {message}"
    );
}

#[ntest::timeout(20000)]
#[test]
fn trilu_with_a_non_boolean_upper_is_refused() {
    let bytes = trilu_model(Some(7), None);
    let error = load_onnx_bytes("trilu_bad_upper.onnx", &bytes)
        .expect_err("upper outside {0, 1} must be refused");
    assert!(error.to_string().contains("upper"), "{error}");
}

// ---------------------------------------------------------------------------
// ScatterElements(reduction = "add")
// ---------------------------------------------------------------------------

/// `data` and `indices` are initializers; `updates` is the live input, so the
/// emitted `ScatterAddLayer` embeds two operands and keeps one activation edge
/// — the configuration `ny-build`'s `resolve_accumulate_operands` selects.
fn scatter_elements_model(
    reduction: Option<&str>,
    indices: &[i64],
    opset: i64,
    op_type: &str,
) -> Vec<u8> {
    let mut attribute = vec![int_attr("axis", 0)];
    if let Some(reduction) = reduction {
        attribute.push(string_attr("reduction", reduction));
    }
    model_bytes(
        "scatter_elements",
        opset,
        vec![node(
            "scatter",
            op_type,
            &["data", "indices", "updates"],
            &["out"],
            attribute,
        )],
        vec![
            f32_initializer("data", &[4], &[10.0, 20.0, 30.0, 40.0]),
            i64_initializer("indices", &[2], indices),
        ],
        vec![value_info("updates", &[2])],
        vec![value_info("out", &[4])],
    )
}

#[ntest::timeout(20000)]
#[test]
fn scatter_elements_add_loads_to_scatter_add() {
    let bytes = scatter_elements_model(Some("add"), &[1, 3], 18, "ScatterElements");
    let model = load_onnx_bytes("scatter_elements.onnx", &bytes)
        .expect("ScatterElements(reduction='add') must load");
    assert_eq!(model.network.layers.len(), 1);
    assert_eq!(model.network.layers[0].layer_type, LayerType::ScatterAdd);
}

#[cfg(feature = "ort")]
#[ntest::timeout(20000)]
#[test]
fn scatter_elements_add_bounds_match_an_independent_runtime() {
    // Distinct destinations, a duplicate destination (both contributions must
    // SUM, which is what makes the additive family order-independent), and a
    // negative index (counted from the end).
    for indices in [vec![1_i64, 3], vec![2, 2], vec![-1, 0]] {
        let bytes = scatter_elements_model(Some("add"), &indices, 18, "ScatterElements");
        let model =
            load_onnx_bytes("scatter_elements.onnx", &bytes).expect("ScatterElements must load");
        let graph = model
            .to_graph_network()
            .expect("ScatterElements must build a graph network");
        let updates = vec![1.5_f32, -2.0];
        let output = graph
            .propagate_ibp(&concrete(&[2], updates.clone()))
            .expect("ScatterAdd IBP must succeed");

        let reference = trusted_forward(
            &bytes,
            &ArrayD::from_shape_vec(IxDyn(&[2]), updates).unwrap(),
        );
        assert_exact(
            &output,
            &reference,
            &format!("ScatterElements add {indices:?}"),
        );
    }
}

#[ntest::timeout(20000)]
#[test]
fn every_non_additive_scatter_elements_reduction_is_refused_by_name() {
    for reduction in [None, Some("none"), Some("mul"), Some("max"), Some("min")] {
        let bytes = scatter_elements_model(reduction, &[1, 3], 18, "ScatterElements");
        let error = load_onnx_bytes("scatter_elements.onnx", &bytes)
            .expect_err("a non-additive reduction must be refused");
        let message = error.to_string();
        assert!(
            message.contains(reduction.unwrap_or("none")) && message.contains("ScatterElements"),
            "the refusal must name the op and its reduction, got: {message}"
        );
    }
}

#[ntest::timeout(20000)]
#[test]
fn the_deprecated_overwrite_scatter_is_refused_by_name() {
    let bytes = scatter_elements_model(None, &[1, 3], 10, "Scatter");
    let error = load_onnx_bytes("scatter.onnx", &bytes)
        .expect_err("the deprecated overwrite Scatter must be refused");
    assert!(error.to_string().contains("Scatter"), "{error}");
}

#[ntest::timeout(20000)]
#[test]
fn scatter_elements_requires_float_data_and_discrete_indices() {
    // FLOAT indices are not the authored INT32/INT64 index dtype; the raw
    // schema gate catches it before folding normalizes the payload to f32.
    let bytes = model_bytes(
        "scatter_elements_float_indices",
        18,
        vec![node(
            "scatter",
            "ScatterElements",
            &["data", "indices", "updates"],
            &["out"],
            vec![int_attr("axis", 0), string_attr("reduction", "add")],
        )],
        vec![
            f32_initializer("data", &[4], &[0.0, 0.0, 0.0, 0.0]),
            f32_initializer("indices", &[2], &[1.0, 3.0]),
        ],
        vec![value_info("updates", &[2])],
        vec![value_info("out", &[4])],
    );
    assert!(load_onnx_bytes("scatter_elements_float_indices.onnx", &bytes).is_err());
}

// ---------------------------------------------------------------------------
// The deleted `op_map` arms: refusal pins at the REAL load boundary
// ---------------------------------------------------------------------------

/// The op types whose `op_map` arms were deleted must still be refused on a
/// real load, by the authority that always shadowed those arms: the raw
/// standard-schema preflight.
///
/// This is the load-path half of
/// `convert::op_map::tests::vendor_domain_op_types_are_not_claimed_by_the_standard_map`
/// and of `convert::tests::legacy_simplified_layer_normalization_is_refused_at_conversion`.
#[ntest::timeout(20000)]
#[test]
fn vendor_domain_op_types_are_refused_on_a_real_standard_domain_load() {
    for op_type in [
        "ArgSort",
        "Snake",
        "RoPE",
        "RotaryPositionEmbedding",
        "SimplifiedLayerNormalization",
        "AdaIN",
        "AdaptiveInstanceNorm",
        "AdaptiveInstanceNormalization",
    ] {
        let bytes = model_bytes(
            "vendor_lookalike",
            23,
            vec![node("vendor", op_type, &["x", "w"], &["out"], Vec::new())],
            vec![f32_initializer("w", &[3], &[1.0, 1.0, 1.0])],
            vec![value_info("x", &[1, 3])],
            vec![value_info("out", &[1, 3])],
        );
        let error = match load_onnx_bytes("vendor_lookalike.onnx", &bytes) {
            Ok(_) => panic!(
                "'{op_type}' is not a main-domain ONNX operator; a standard-domain graph \
                 declaring it must not load"
            ),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(op_type),
            "the refusal must name the op type, got: {error}"
        );
    }
}
