// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end parity for ONNX `Resize` with `mode="linear"`.
//!
//! The lowering (`loader/convert/resize_linear.rs`) re-derives the
//! source-coordinate map per authored `coordinate_transformation_mode` rather
//! than reusing the trace bridge's single half-pixel kernel. The load-bearing
//! evidence for that is here: for EVERY admitted convention, ny's IBP at a
//! concrete input must reproduce an independent runtime's forward over the
//! same model bytes — and the conventions must not collapse into one another,
//! which is what a wrong reuse would look like.

use super::onnx_fixture::*;
use super::*;
use crate::load_onnx_bytes;
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_tensor::BoundedTensor;

const IN_SHAPE: [i64; 4] = [1, 1, 2, 2];
const IN_VALUES: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

fn resize_model(
    attributes: Vec<onnx_proto::AttributeProto>,
    scales: &[f32],
    out_shape: &[i64],
) -> Vec<u8> {
    model_bytes(
        "resize_linear",
        18,
        vec![node(
            "resize",
            "Resize",
            &["input", "roi", "scales"],
            &["out"],
            attributes,
        )],
        vec![
            f32_initializer("roi", &[0], &[]),
            f32_initializer("scales", &[scales.len() as i64], scales),
        ],
        vec![value_info("input", &IN_SHAPE)],
        vec![value_info("out", out_shape)],
    )
}

fn linear_attributes(coordinate: &str) -> Vec<onnx_proto::AttributeProto> {
    vec![
        string_attr("mode", "linear"),
        string_attr("coordinate_transformation_mode", coordinate),
    ]
}

fn concrete_input() -> BoundedTensor {
    let array = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), IN_VALUES.to_vec()).unwrap();
    BoundedTensor::new(array.clone(), array).unwrap()
}

const ADMITTED_MODES: [&str; 4] = [
    "half_pixel",
    "pytorch_half_pixel",
    "align_corners",
    "asymmetric",
];

#[ntest::timeout(30000)]
#[test]
fn a_linear_resize_lowers_to_the_separated_linear_pair() {
    let bytes = resize_model(
        linear_attributes("half_pixel"),
        &[1.0, 1.0, 2.0, 2.0],
        &[1, 1, 4, 4],
    );
    let model = load_onnx_bytes("resize_linear.onnx", &bytes).expect("a linear Resize must load");
    let types: Vec<LayerType> = model
        .network
        .layers
        .iter()
        .map(|layer| layer.layer_type.clone())
        .collect();
    assert_eq!(
        types,
        vec![
            LayerType::Linear,
            LayerType::Transpose,
            LayerType::Linear,
            LayerType::Transpose,
        ],
        "W resample, transpose, H resample, transpose back"
    );

    // Only the width axis changes: one Linear, no transposes.
    let bytes = resize_model(
        linear_attributes("half_pixel"),
        &[1.0, 1.0, 1.0, 2.0],
        &[1, 1, 2, 4],
    );
    let model = load_onnx_bytes("resize_linear_w.onnx", &bytes).expect("must load");
    assert_eq!(model.network.layers.len(), 1);
    assert_eq!(model.network.layers[0].layer_type, LayerType::Linear);
}

#[ntest::timeout(30000)]
#[test]
fn a_nearest_resize_keeps_its_existing_lowering() {
    let bytes = resize_model(
        vec![
            string_attr("mode", "nearest"),
            string_attr("coordinate_transformation_mode", "asymmetric"),
            string_attr("nearest_mode", "floor"),
        ],
        &[1.0, 1.0, 2.0, 2.0],
        &[1, 1, 4, 4],
    );
    let model = load_onnx_bytes("resize_nearest.onnx", &bytes).expect("must load");
    assert_eq!(model.network.layers.len(), 1);
    assert_eq!(model.network.layers[0].layer_type, LayerType::Resize);
}

#[cfg(feature = "ort")]
#[ntest::timeout(60000)]
#[test]
fn every_admitted_convention_matches_an_independent_runtime() {
    for coordinate in ADMITTED_MODES {
        for (scales, out_shape) in [
            (vec![1.0_f32, 1.0, 2.0, 2.0], vec![1_i64, 1, 4, 4]), // upscale both
            (vec![1.0, 1.0, 1.0, 2.0], vec![1, 1, 2, 4]),         // width only
            (vec![1.0, 1.0, 2.0, 1.0], vec![1, 1, 4, 2]),         // height only
            (vec![1.0, 1.0, 3.0, 3.0], vec![1, 1, 6, 6]),         // non-power-of-two
        ] {
            let bytes = resize_model(linear_attributes(coordinate), &scales, &out_shape);
            let model = load_onnx_bytes("resize_linear.onnx", &bytes).unwrap_or_else(|err| {
                panic!("Resize(linear, {coordinate}, scales={scales:?}) must load: {err}")
            });
            let graph = model
                .to_graph_network()
                .expect("the resampler must build a graph network");
            let output = graph
                .propagate_ibp(&concrete_input())
                .expect("resampler IBP must succeed");

            let reference = trusted_forward(
                &bytes,
                &ArrayD::from_shape_vec(IxDyn(&IN_SHAPE.map(|d| d as usize)), IN_VALUES.to_vec())
                    .unwrap(),
            );
            let lower: Vec<f32> = output.lower().iter().copied().collect();
            let upper: Vec<f32> = output.upper().iter().copied().collect();
            assert_eq!(
                lower.len(),
                reference.len(),
                "{coordinate} {scales:?}: element count mismatch"
            );
            for (index, expected) in reference.iter().enumerate() {
                // ENCLOSURE is the soundness obligation: the independent
                // runtime's value for this cell must lie inside ny's bound.
                assert!(
                    lower[index] <= *expected && *expected <= upper[index],
                    "{coordinate} {scales:?}: cell {index} value {expected} escapes \
                     [{}, {}]",
                    lower[index],
                    upper[index]
                );
                // NO-LOOSENESS is the second claim, and the interval is not
                // exactly a point: `LinearLayer`'s IBP rounds each endpoint
                // OUTWARD, so a resampled cell carries a few ULPs of certified
                // slack — the same slack any authored Gemm carries. What must
                // NOT appear is slack of the size a wrong coordinate map would
                // produce (whole source cells apart).
                let width = upper[index] - lower[index];
                assert!(
                    width <= 1e-4 * expected.abs().max(1.0),
                    "{coordinate} {scales:?}: cell {index} bound [{}, {}] is far wider than \
                     the Linear tier's rounding slack",
                    lower[index],
                    upper[index]
                );
            }
        }
    }
}

#[cfg(feature = "ort")]
#[ntest::timeout(30000)]
#[test]
fn the_conventions_do_not_collapse_into_one_another() {
    // The whole reason the bridge's fixed half-pixel kernel cannot be reused:
    // on the SAME geometry, three of the admitted conventions read different
    // source cells. If a future refactor lowered them all through one map,
    // this fails.
    let mut seen: Vec<(String, Vec<f32>)> = Vec::new();
    for coordinate in ["half_pixel", "align_corners", "asymmetric"] {
        let bytes = resize_model(
            linear_attributes(coordinate),
            &[1.0, 1.0, 2.0, 2.0],
            &[1, 1, 4, 4],
        );
        let model = load_onnx_bytes("resize_linear.onnx", &bytes).expect("must load");
        let graph = model.to_graph_network().expect("must build");
        let output = graph.propagate_ibp(&concrete_input()).expect("IBP");
        seen.push((
            coordinate.to_string(),
            output.lower().iter().copied().collect(),
        ));
    }
    for i in 0..seen.len() {
        for j in (i + 1)..seen.len() {
            assert_ne!(
                seen[i].1, seen[j].1,
                "{} and {} must not produce the same resample",
                seen[i].0, seen[j].0
            );
        }
    }
}

#[cfg(feature = "ort")]
#[ntest::timeout(30000)]
#[test]
fn the_resampler_encloses_a_widened_input() {
    // A row-substochastic convex-combination map cannot leave the input hull,
    // and every concrete point inside the box must land inside the output box.
    let bytes = resize_model(
        linear_attributes("half_pixel"),
        &[1.0, 1.0, 2.0, 2.0],
        &[1, 1, 4, 4],
    );
    let model = load_onnx_bytes("resize_linear.onnx", &bytes).expect("must load");
    let graph = model.to_graph_network().expect("must build");
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.5_f32, 1.5, 2.5, 3.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.5_f32, 2.5, 3.5, 4.5]).unwrap();
    let output = graph
        .propagate_ibp(&BoundedTensor::new(lower, upper).unwrap())
        .expect("IBP must succeed");

    // Each matrix row is a convex combination, so the exact image cannot leave
    // [0.5, 4.5]. The emitted endpoints may sit a few ULPs outside it because
    // `LinearLayer`'s IBP rounds OUTWARD — which is the sound direction. What
    // must not happen is a bound escaping by anything the size of a cell.
    for value in output.lower().iter().chain(output.upper().iter()) {
        assert!(
            (0.5 - 1e-4..=4.5 + 1e-4).contains(value),
            "a convex combination cannot leave the input hull [0.5, 4.5], got {value}"
        );
    }

    for interior in [
        vec![0.5_f32, 1.5, 2.5, 3.5],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.5, 2.5, 3.5, 4.5],
        vec![0.75, 2.25, 2.75, 4.25],
    ] {
        let reference = trusted_forward(
            &bytes,
            &ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), interior.clone()).unwrap(),
        );
        let lower: Vec<f32> = output.lower().iter().copied().collect();
        let upper: Vec<f32> = output.upper().iter().copied().collect();
        for (index, value) in reference.iter().enumerate() {
            assert!(
                lower[index] <= *value && *value <= upper[index],
                "interior {interior:?} cell {index}: {value} escapes [{}, {}]",
                lower[index],
                upper[index]
            );
        }
    }
}

#[ntest::timeout(30000)]
#[test]
fn unrepresented_conventions_and_kernels_are_refused_by_name() {
    let cases: Vec<(Vec<onnx_proto::AttributeProto>, &str)> = vec![
        (
            linear_attributes("tf_crop_and_resize"),
            "tf_crop_and_resize",
        ),
        (
            linear_attributes("half_pixel_symmetric"),
            "half_pixel_symmetric",
        ),
        (vec![string_attr("mode", "cubic")], "cubic"),
        (
            {
                let mut attributes = linear_attributes("half_pixel");
                attributes.push(int_attr("exclude_outside", 1));
                attributes
            },
            "exclude_outside",
        ),
        (
            {
                let mut attributes = linear_attributes("half_pixel");
                attributes.push(int_attr("antialias", 1));
                attributes
            },
            "antialias",
        ),
        (
            {
                let mut attributes = linear_attributes("half_pixel");
                attributes.push(string_attr("keep_aspect_ratio_policy", "not_larger"));
                attributes
            },
            "keep_aspect_ratio_policy",
        ),
    ];
    for (attributes, needle) in cases {
        let bytes = resize_model(attributes, &[1.0, 1.0, 2.0, 2.0], &[1, 1, 4, 4]);
        let error = load_onnx_bytes("resize_refused.onnx", &bytes)
            .expect_err("an unrepresented Resize configuration must be refused");
        assert!(
            error.to_string().contains(needle),
            "the refusal must name '{needle}', got: {error}"
        );
    }
}

#[ntest::timeout(30000)]
#[test]
fn resizing_a_non_spatial_axis_is_refused() {
    let bytes = resize_model(
        linear_attributes("half_pixel"),
        &[1.0, 2.0, 2.0, 2.0],
        &[1, 2, 4, 4],
    );
    let error = load_onnx_bytes("resize_channels.onnx", &bytes)
        .expect_err("resizing the channel axis must be refused");
    assert!(error.to_string().contains("non-spatial axis"), "{error}");
}

#[ntest::timeout(30000)]
#[test]
fn the_deprecated_linear_upsample_is_refused_because_its_convention_is_not_on_the_wire() {
    let bytes = model_bytes(
        "upsample_linear",
        9,
        vec![node(
            "up",
            "Upsample",
            &["input", "scales"],
            &["out"],
            vec![string_attr("mode", "linear")],
        )],
        vec![f32_initializer("scales", &[4], &[1.0, 1.0, 2.0, 2.0])],
        vec![value_info("input", &IN_SHAPE)],
        vec![value_info("out", &[1, 1, 4, 4])],
    );
    let error = load_onnx_bytes("upsample_linear.onnx", &bytes)
        .expect_err("a linear Upsample must be refused");
    assert!(
        error.to_string().contains("coordinate_transformation_mode"),
        "{error}"
    );
}
