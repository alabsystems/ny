// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, ArrayD, Axis, IxDyn};
use ny_propagate::layers::{AveragePoolLayer, Conv1dLayer, ConvTranspose1dLayer};

fn make_conv1d_graph() -> GraphNetwork {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0, -0.5]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Conv1d(
        Conv1dLayer::new(kernel, Some(arr1(&[0.25])), 1, 0).expect("conv1d kernel should be valid"),
    ));
    GraphNetwork::from_sequential(&network).expect("single conv1d network should convert to graph")
}

fn make_conv_transpose1d_graph() -> GraphNetwork {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0, 0.5]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::ConvTranspose1d(
        ConvTranspose1dLayer::new(kernel, Some(arr1(&[-0.1])), 1, 0)
            .expect("conv_transpose1d kernel should be valid"),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("single conv_transpose1d network should convert to graph")
}

fn make_average_pool_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::AveragePool(AveragePoolLayer::new(
        (2, 2),
        (2, 2),
        (0, 0),
        false,
    )));
    GraphNetwork::from_sequential(&network)
        .expect("single average-pool network should convert to graph")
}

fn squeeze_unit_axes(mut array: ArrayD<f32>) -> ArrayD<f32> {
    while let Some(axis) = array.shape().iter().position(|&dim| dim == 1) {
        array = array.index_axis_move(Axis(axis), 0).into_dyn();
    }
    array
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_reshape_4096() {
    let graph = make_fixed_reshape_linear_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, -0.5, 0.25, 0.75]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0, 2.0, -0.25, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 0.5, 1.0, -1.5]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_flatten_4096() {
    let graph = make_fixed_flatten_linear_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = [
        arr1(&[1.0, -0.5, 0.25, 0.75]).into_dyn(),
        arr1(&[-1.0, 2.0, -0.25, 0.5]).into_dyn(),
        arr1(&[0.0, 0.5, 1.0, -1.5]).into_dyn(),
    ];

    let sample_views: Vec<_> = samples.iter().map(|sample| sample.view()).collect();
    let batched_samples = ndarray::stack(Axis(0), &sample_views)
        .expect("stacked test inputs should form a valid restart batch");
    let batched_input =
        BoundedTensor::concrete(batched_samples).expect("batched concrete input should be valid");
    let batched_output = graph
        .propagate_ibp_with_engine_preserve_leading_axis(&batched_input, Some(&engine))
        .expect("flatten preserve-leading-axis IBP should succeed");

    for (batch_index, sample) in samples.iter().enumerate() {
        let sequential_input =
            BoundedTensor::concrete(sample.clone()).expect("concrete sample should be valid");
        let sequential_output = graph
            .propagate_ibp_with_engine(&sequential_input, Some(&engine))
            .expect("sequential IBP should succeed");
        let batched_lower = batched_output
            .lower()
            .index_axis(Axis(0), batch_index)
            .to_owned()
            .into_dyn();
        let batched_upper = batched_output
            .upper()
            .index_axis(Axis(0), batch_index)
            .to_owned()
            .into_dyn();

        // Flatten(0) intentionally drops the synthetic singleton axis from the
        // sample-space output under preserve-leading-axis mode. Compare after
        // squeezing unit axes so the regression matches the approved #4093 contract.
        assert_arrays_close(
            &squeeze_unit_axes(batched_lower),
            &squeeze_unit_axes(sequential_output.lower().clone()),
            "flatten lower bounds should match sequential output after squeezing unit axes",
        );
        assert_arrays_close(
            &squeeze_unit_axes(batched_upper),
            &squeeze_unit_axes(sequential_output.upper().clone()),
            "flatten upper bounds should match sequential output after squeezing unit axes",
        );
    }
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_conv1d_4096() {
    let graph = make_conv1d_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, -0.5, 0.25, 0.75]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0, 2.0, -0.25, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 0.5, 1.0, -1.5]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_conv_transpose1d_4096() {
    let graph = make_conv_transpose1d_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, -0.5, 0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-1.0, 2.0, -0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0.0, 0.5, 1.0]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_average_pool_4096() {
    let graph = make_average_pool_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, -0.5, 0.25, 0.75]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0, 2.0, 0.5, -0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.0, 0.5, 1.0, -1.5]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}
