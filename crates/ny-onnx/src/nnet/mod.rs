// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NNet format support for loading neural network verification benchmarks.
//!
//! NNet is a simple text format for ReLU networks, commonly used in the
//! VNN-COMP (Verification of Neural Networks Competition) benchmarks,
//! particularly for ACAS-Xu collision avoidance networks.
//!
//! # Format Specification
//!
//! The NNet format (Kyle Julian, Stanford 2016) stores fully-connected
//! ReLU networks as plain text:
//!
//! - Comment lines starting with `//`
//! - Header: numLayers, inputSize, outputSize, maxLayerSize
//! - Layer sizes (comma-separated)
//! - Symmetric flag (typically 0, unused)
//! - Input bounds: minimums, maximums
//! - Normalization: means, ranges (for inputs + 1 for output)
//! - For each layer: weight matrix (row-major), then bias vector
//!
//! The network uses ReLU activations for hidden layers and linear output.
//!
//! # Module Structure
//!
//! - `convert` — Conversion to ny Network and PropNetwork formats
//! - `io` — File loading (`load_nnet`)
//! - `model` — NNetNetwork runtime methods (evaluate, normalize, param_count)
//! - `parser` — NNet text format parsing
//!
//! # Usage
//!
//! ```rust,no_run
//! use ny_onnx::nnet::{load_nnet, NNetNetwork};
//!
//! let network = load_nnet("model.nnet").unwrap();
//! println!("Layers: {}, Inputs: {}, Outputs: {}",
//!          network.num_layers(), network.input_size(), network.output_size());
//! ```

mod convert;
mod io;
mod model;
pub(crate) mod parser;

#[cfg(test)]
mod tests;

use ndarray::{Array1, Array2};

pub use io::load_nnet;
pub use parser::parse_nnet;

/// A parsed NNet network with all metadata.
#[derive(Debug, Clone)]
pub struct NNetNetwork {
    /// Number of layers (not including input layer).
    pub(crate) num_layers: usize,
    /// Size of input layer.
    pub(crate) input_size: usize,
    /// Size of output layer.
    pub(crate) output_size: usize,
    /// Maximum size of any hidden layer.
    pub(crate) max_layer_size: usize,
    /// Sizes of all layers including input and output.
    pub(crate) layer_sizes: Vec<usize>,
    /// Minimum input values (for normalization/clipping).
    pub(crate) input_minimums: Vec<f32>,
    /// Maximum input values (for normalization/clipping).
    pub(crate) input_maximums: Vec<f32>,
    /// Mean values for input normalization.
    pub(crate) input_means: Vec<f32>,
    /// Range values for input normalization.
    pub(crate) input_ranges: Vec<f32>,
    /// Mean value for output denormalization.
    pub(crate) output_mean: f32,
    /// Range value for output denormalization.
    pub(crate) output_range: f32,
    /// Weight matrices for each layer (`layer_sizes[i+1] x layer_sizes[i]`).
    pub(crate) weights: Vec<Array2<f32>>,
    /// Bias vectors for each layer (`layer_sizes[i+1]`).
    pub(crate) biases: Vec<Array1<f32>>,
}

impl NNetNetwork {
    /// Number of layers (not including input layer).
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Size of input layer.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Size of output layer.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Maximum size of any hidden layer.
    pub fn max_layer_size(&self) -> usize {
        self.max_layer_size
    }

    /// Sizes of all layers including input and output.
    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    /// Minimum input values (for normalization/clipping).
    pub fn input_minimums(&self) -> &[f32] {
        &self.input_minimums
    }

    /// Maximum input values (for normalization/clipping).
    pub fn input_maximums(&self) -> &[f32] {
        &self.input_maximums
    }

    /// Mean values for input normalization.
    pub fn input_means(&self) -> &[f32] {
        &self.input_means
    }

    /// Range values for input normalization.
    pub fn input_ranges(&self) -> &[f32] {
        &self.input_ranges
    }

    /// Mean value for output denormalization.
    pub fn output_mean(&self) -> f32 {
        self.output_mean
    }

    /// Range value for output denormalization.
    pub fn output_range(&self) -> f32 {
        self.output_range
    }

    /// Weight matrices for each layer.
    pub fn weights(&self) -> &[Array2<f32>] {
        &self.weights
    }

    /// Bias vectors for each layer.
    pub fn biases(&self) -> &[Array1<f32>] {
        &self.biases
    }
}
