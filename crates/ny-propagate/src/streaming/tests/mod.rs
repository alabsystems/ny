// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `streaming` module, split by behavior family.
//!
//! - `config`: StreamingConfig construction and defaults
//! - `checkpoint`: CheckpointedBounds API — add, find, sort, memory, edge cases
//! - `f16_compression`: f16 compressed checkpoint storage, soundness, overflow
//! - `crown_streaming`: Streaming CROWN equivalence vs regular CROWN
//! - `batched_streaming`: Batched streaming CROWN parity and activation coverage

pub(super) use crate::layers::{Layer, LinearLayer};
pub(super) use crate::network::Network;
pub(super) use ndarray::{Array1, Array2, ArrayD};
pub(super) use ny_tensor::BoundedTensor;

/// Create a linear-only test network with zero weights and biases.
pub(super) fn create_test_network(num_layers: usize, in_dim: usize, out_dim: usize) -> Network {
    let mut network = Network::new();
    for i in 0..num_layers {
        let (layer_in, layer_out) = if i == 0 {
            (in_dim, out_dim)
        } else {
            (out_dim, out_dim)
        };
        let weight = Array2::<f32>::zeros((layer_out, layer_in));
        let bias = Some(Array1::<f32>::zeros(layer_out));
        let linear = LinearLayer::new(weight, bias).unwrap();
        network.add_layer(Layer::Linear(linear));
    }
    network
}

/// Create a test input with bounds [-1, 1] in each dimension.
pub(super) fn create_input(dim: usize) -> BoundedTensor {
    let lower = ArrayD::from_elem(ndarray::IxDyn(&[dim]), -1.0_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[dim]), 1.0_f32);
    BoundedTensor::new(lower, upper).unwrap()
}

mod batched_streaming;
mod checkpoint;
mod config;
mod crown_streaming;
mod engine_streaming;
mod f16_compression;
