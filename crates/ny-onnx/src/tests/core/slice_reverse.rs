// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end front-end parity for a REVERSING ONNX `Slice` (`steps = [-1]`),
//! lowered to the unit-`Slice` + `Concat` decomposition `ny-trace-bridge`
//! already runs for `TraceOp::Flip`.
//!
//! A reversal is a pure reindexing, so the acceptance tests assert ny's bound
//! is CONCRETE (zero width) and bit-equal to an independent runtime's forward
//! over the same model bytes — no slack is admissible here.

use super::onnx_fixture::*;
use crate::load_onnx_bytes;
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_tensor::BoundedTensor;

const AXIS_LENGTH: i64 = 4;

/// `Slice(x, starts, ends, axes, steps)` over a `[1, 4]` input.
///
/// Axis 1 is used throughout: ONNX axis 0 is the (possibly batch-stripped)
/// leading dimension, which the lowering deliberately refuses.
fn slice_model(start: i64, end: i64, axis: i64, step: i64, out_len: i64) -> Vec<u8> {
    model_bytes(
        "reverse_slice",
        17,
        vec![node(
            "rev",
            "Slice",
            &["x", "starts", "ends", "axes", "steps"],
            &["out"],
            Vec::new(),
        )],
        vec![
            i64_initializer("starts", &[1], &[start]),
            i64_initializer("ends", &[1], &[end]),
            i64_initializer("axes", &[1], &[axis]),
            i64_initializer("steps", &[1], &[step]),
        ],
        vec![value_info("x", &[1, AXIS_LENGTH])],
        vec![value_info("out", &[1, out_len])],
    )
}

fn concrete(shape: &[usize], data: Vec<f32>) -> BoundedTensor {
    let array = ArrayD::from_shape_vec(IxDyn(shape), data).expect("fixture shape");
    BoundedTensor::new(array.clone(), array).expect("a concrete tensor is a valid bound")
}

#[ntest::timeout(20000)]
#[test]
fn a_full_reversal_lowers_to_unit_slices_and_one_concat() {
    let bytes = slice_model(-1, i64::MIN, 1, -1, AXIS_LENGTH);
    let model = load_onnx_bytes("reverse_slice.onnx", &bytes).expect("a reversing Slice must load");
    let types: Vec<LayerType> = model
        .network
        .layers
        .iter()
        .map(|layer| layer.layer_type.clone())
        .collect();
    assert_eq!(
        types,
        vec![
            LayerType::Slice,
            LayerType::Slice,
            LayerType::Slice,
            LayerType::Slice,
            LayerType::Concat,
        ]
    );
}

#[ntest::timeout(20000)]
#[test]
fn a_forward_slice_keeps_its_single_layer_lowering() {
    let bytes = slice_model(1, 3, 1, 1, 2);
    let model = load_onnx_bytes("forward_slice.onnx", &bytes)
        .expect("an ordinary forward Slice must still load");
    assert_eq!(
        model.network.layers.len(),
        1,
        "the reversal canonicalizer must not touch a forward Slice"
    );
    assert_eq!(model.network.layers[0].layer_type, LayerType::Slice);
}

#[cfg(feature = "ort")]
#[ntest::timeout(20000)]
#[test]
fn reversal_bounds_match_an_independent_runtime_on_the_same_bytes() {
    let values = vec![1.0_f32, 2.0, 3.0, 4.0];
    for (start, end, out_len) in [
        (-1_i64, i64::MIN, 4_i64), // full reversal
        (2, 0, 2),                 // x[2:0:-1]
        (-1, -3, 2),               // negative start and end
        (99, -5, 4),               // an oversized start clamps to dim - 1
        (-1, -2, 1),               // a single retained position: no Concat
    ] {
        let bytes = slice_model(start, end, 1, -1, out_len);
        let model = load_onnx_bytes("reverse_slice.onnx", &bytes)
            .unwrap_or_else(|err| panic!("Slice[{start}:{end}:-1] must load: {err}"));
        let graph = model
            .to_graph_network()
            .expect("the reversal must build a graph network");
        let output = graph
            .propagate_ibp(&concrete(&[4], values.clone()))
            .expect("reversal IBP must succeed");

        let reference = trusted_forward(
            &bytes,
            &ArrayD::from_shape_vec(IxDyn(&[1, 4]), values.clone()).unwrap(),
        );
        let lower: Vec<f32> = output.lower().iter().copied().collect();
        let upper: Vec<f32> = output.upper().iter().copied().collect();
        assert_eq!(lower, upper, "a reindexing must stay concrete");
        assert_eq!(
            lower, reference,
            "Slice[{start}:{end}:-1] must reproduce the runtime's reversal exactly"
        );
    }
}

#[cfg(feature = "ort")]
#[ntest::timeout(20000)]
#[test]
fn reversal_encloses_a_widened_input_without_extra_slack() {
    // A reindexing must carry both endpoints through untouched, so the output
    // interval is exactly the input interval permuted.
    let bytes = slice_model(-1, i64::MIN, 1, -1, AXIS_LENGTH);
    let model = load_onnx_bytes("reverse_slice.onnx", &bytes).expect("must load");
    let graph = model.to_graph_network().expect("must build");
    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0_f32, 0.0, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.5_f32, 0.5, 1.5, 2.5]).unwrap();
    let output = graph
        .propagate_ibp(&BoundedTensor::new(lower, upper).unwrap())
        .expect("IBP must succeed");
    assert_eq!(
        output.lower().iter().copied().collect::<Vec<_>>(),
        vec![2.0, 1.0, 0.0, -1.0]
    );
    assert_eq!(
        output.upper().iter().copied().collect::<Vec<_>>(),
        vec![2.5, 1.5, 0.5, -0.5]
    );

    // And it encloses the concrete evaluation of an interior point.
    let interior = vec![-0.75_f32, 0.25, 1.25, 2.25];
    let reference = trusted_forward(
        &bytes,
        &ArrayD::from_shape_vec(IxDyn(&[1, 4]), interior).unwrap(),
    );
    for (index, value) in reference.iter().enumerate() {
        assert!(
            output.lower()[index] <= *value && *value <= output.upper()[index],
            "position {index}: {value} escapes [{}, {}]",
            output.lower()[index],
            output.upper()[index]
        );
    }
}

#[ntest::timeout(20000)]
#[test]
fn strides_other_than_minus_one_are_refused_by_name() {
    let bytes = slice_model(-1, i64::MIN, 1, -2, 2);
    let error = load_onnx_bytes("stride_slice.onnx", &bytes)
        .expect_err("a strided reversal must be refused");
    assert!(error.to_string().contains("step=-2"), "{error}");
}

#[ntest::timeout(20000)]
#[test]
fn reversing_the_leading_axis_is_refused() {
    let bytes = slice_model(-1, i64::MIN, 0, -1, 1);
    let error = load_onnx_bytes("batch_reverse.onnx", &bytes)
        .expect_err("reversing the batch axis must be refused");
    assert!(error.to_string().contains("axis 0"), "{error}");
}

#[ntest::timeout(20000)]
#[test]
fn an_empty_reversal_is_refused() {
    // start == end after clamping selects nothing, and ny has no empty-extent
    // tensor representation.
    let bytes = slice_model(0, 0, 1, -1, 1);
    assert!(load_onnx_bytes("empty_reverse.onnx", &bytes).is_err());
}
