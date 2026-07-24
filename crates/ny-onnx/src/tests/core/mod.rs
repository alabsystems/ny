// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::fixtures::*;
use super::*;
use approx::relative_eq;
use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_propagate::Network as PropNetwork;
use ny_tensor::BoundedTensor;
use std::path::Path;

fn concrete_input_for_model(model: &OnnxModel) -> BoundedTensor {
    let input_spec = model
        .network
        .inputs
        .first()
        .expect("Model should have at least one input spec");
    let shape: Vec<usize> = input_spec
        .shape
        .iter()
        .map(|&dim| if dim > 0 { dim as usize } else { 1 })
        .collect();
    let data = ArrayD::zeros(IxDyn(&shape));
    BoundedTensor::new(data.clone(), data).expect("Failed to create concrete input tensor")
}

fn assert_round_trip_sequential(path: impl AsRef<Path>) {
    let model = load_onnx(path).expect("Failed to load model");
    let seq = model
        .to_propagate_network()
        .expect("Failed to convert to sequential network");
    let graph = model
        .to_graph_network()
        .expect("Failed to convert to graph network");

    let roundtrip_seq = graph
        .try_to_sequential_network()
        .expect("Graph should be convertible back to sequential network");

    assert_eq!(
        seq.layers().len(),
        roundtrip_seq.layers().len(),
        "Round-trip layer count mismatch"
    );

    for (idx, (orig, roundtrip)) in seq
        .layers()
        .iter()
        .zip(roundtrip_seq.layers().iter())
        .enumerate()
    {
        assert_eq!(
            std::mem::discriminant(orig),
            std::mem::discriminant(roundtrip),
            "Layer type mismatch at index {}",
            idx
        );
    }

    let concrete_input = concrete_input_for_model(&model);

    let orig_output = seq
        .propagate_ibp(&concrete_input)
        .expect("IBP failed on original sequential network");
    let roundtrip_output = roundtrip_seq
        .propagate_ibp(&concrete_input)
        .expect("IBP failed on round-trip sequential network");

    assert_eq!(
        orig_output.lower().shape(),
        roundtrip_output.lower().shape(),
        "Round-trip lower shape mismatch"
    );
    assert_eq!(
        orig_output.upper().shape(),
        roundtrip_output.upper().shape(),
        "Round-trip upper shape mismatch"
    );

    for (idx, (lower, upper)) in orig_output
        .lower()
        .iter()
        .zip(orig_output.upper().iter())
        .enumerate()
    {
        assert!(
            relative_eq!(*lower, *upper, epsilon = 1e-5),
            "Original output bounds should collapse for concrete input at index {}: {} vs {}",
            idx,
            lower,
            upper
        );
    }

    for (idx, (lower, upper)) in roundtrip_output
        .lower()
        .iter()
        .zip(roundtrip_output.upper().iter())
        .enumerate()
    {
        assert!(
            relative_eq!(*lower, *upper, epsilon = 1e-5),
            "Round-trip output bounds should collapse for concrete input at index {}: {} vs {}",
            idx,
            lower,
            upper
        );
    }

    for (idx, (orig, roundtrip)) in orig_output
        .lower()
        .iter()
        .zip(roundtrip_output.lower().iter())
        .enumerate()
    {
        assert!(
            relative_eq!(*orig, *roundtrip, epsilon = 1e-5),
            "Round-trip lower bound mismatch at index {}: {} vs {}",
            idx,
            orig,
            roundtrip
        );
    }

    for (idx, (orig, roundtrip)) in orig_output
        .upper()
        .iter()
        .zip(roundtrip_output.upper().iter())
        .enumerate()
    {
        assert!(
            relative_eq!(*orig, *roundtrip, epsilon = 1e-5),
            "Round-trip upper bound mismatch at index {}: {} vs {}",
            idx,
            orig,
            roundtrip
        );
    }
}

fn assert_dynamic_reshape_error(result: Result<PropNetwork>, layer_name: &str) {
    match result {
        Err(NyError::UnsupportedOp(msg)) => {
            assert!(
                msg.contains("dynamic shape"),
                "unexpected error message: {}",
                msg
            );
            assert!(
                msg.contains("PropagateNetworkOptions::permissive"),
                "missing permissive hint: {}",
                msg
            );
            assert!(
                msg.contains(layer_name),
                "missing layer name {}: {}",
                layer_name,
                msg
            );
        }
        other => panic!("expected dynamic reshape UnsupportedOp, got {:?}", other),
    }
}

mod acasxu_1923;
mod attention;
mod avoice;
mod cnn;
mod custom_ops;
mod decoder;
mod differential_translation;
mod fusion;
mod gpu_crown;
mod gpu_crown_graph;
#[cfg(feature = "benchmarks")]
mod gpu_crown_timing;
mod graph;
mod load;
mod matmul;
mod merge_linear;
mod mul;
mod probabilistic_integration;
mod propagate;
mod reduce_batch_axis;
mod reshape;
mod resize;
mod scatter_nd;
mod shape_inference_policy;
mod slice_batch_axis;
