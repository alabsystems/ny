// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conversion from NNetNetwork to ny Network and PropNetwork formats.

use crate::{DataType, Network, TensorSpec};
use ndarray::Array1;
use ny_core::Result;
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{Layer, Network as PropNetwork};

use super::NNetNetwork;

impl NNetNetwork {
    /// Convert to ny's Network format for inspection.
    ///
    /// Note: For verification, use `to_prop_network()` instead.
    pub fn to_ny_network(&self) -> Network {
        Network {
            name: "nnet_model".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, self.input_size as i64],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![1, self.output_size as i64],
                dtype: DataType::Float32,
            }],
            layers: Vec::new(), // LayerSpec requires complex setup; use to_prop_network for verification
            param_count: self.param_count(),
        }
    }

    /// Convert to ny's PropNetwork for verification.
    pub fn to_prop_network(&self) -> Result<PropNetwork> {
        let mut network = PropNetwork::new();

        for (layer_idx, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            let is_output = layer_idx == self.num_layers - 1;

            // Create bias Array1
            let bias = Array1::from_vec(b.iter().cloned().collect());

            // Add Linear layer
            let linear = LinearLayer::new(w.clone(), Some(bias))?;
            network.add_layer(Layer::Linear(linear));

            // Add ReLU for hidden layers
            if !is_output {
                network.add_layer(Layer::ReLU(ReLULayer));
            }
        }

        Ok(network)
    }
}
