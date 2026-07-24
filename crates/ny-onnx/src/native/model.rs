// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::builders::build_network;
use super::detect::detect_architecture;
use super::weights::load_weights;
use super::{HfConfig, ModelConfig};
use crate::{Network, OnnxModel, WeightStore};
use ny_core::{NyError, Result};
use ny_propagate::Network as PropNetwork;
use std::path::Path;
use tracing::info;

/// A model loaded from native format (PyTorch/SafeTensors).
pub struct NativeModel {
    /// Network specification (graph structure).
    pub network: Network,
    /// Weight storage.
    pub weights: WeightStore,
    /// Detected or specified configuration.
    pub config: ModelConfig,
}

impl NativeModel {
    /// Load a model from a native format file or directory.
    ///
    /// If loading from a HuggingFace model directory with config.json,
    /// uses the config to determine architecture. Otherwise falls back
    /// to weight-name-based detection.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading native model from: {}", path.display());

        // Try to load HfConfig if this is a directory with config.json
        let hf_config = if path.is_dir() {
            HfConfig::from_directory(path)?
        } else {
            // Check parent directory for config.json. Parse/read failures must
            // remain hard loader errors rather than silently falling back to
            // heuristic architecture detection.
            path.parent()
                .map(HfConfig::from_directory)
                .transpose()?
                .flatten()
        };

        // Load weights based on extension
        let weights = load_weights(path)?;

        // Detect architecture - prefer HfConfig over weight-based detection
        let config = if let Some(hf_cfg) = &hf_config {
            info!(
                "Using HfConfig: architecture={:?}, model_type={}",
                hf_cfg.architecture_name(),
                hf_cfg.model_type
            );
            hf_cfg.to_model_config()
        } else {
            detect_architecture(&weights)?
        };
        info!("Detected architecture: {:?}", config.architecture);

        // Build network from weights
        let network = build_network(&weights, &config)?;

        Ok(Self {
            network,
            weights,
            config,
        })
    }

    /// Load a model from a HuggingFace model directory with explicit config.json.
    ///
    /// This is the preferred method for loading HuggingFace models as it
    /// uses the config.json to accurately determine architecture.
    pub fn load_from_hf_directory<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Err(NyError::ModelLoad(format!(
                "Expected directory, got file: {}",
                path.display()
            )));
        }

        let hf_config = HfConfig::from_directory(path)?.ok_or_else(|| {
            NyError::ModelLoad(format!(
                "No config.json found in directory: {}",
                path.display()
            ))
        })?;

        info!(
            "Loading HuggingFace model: {} ({})",
            hf_config.architecture_name().unwrap_or("unknown"),
            hf_config.model_type
        );

        let weights = load_weights(path)?;
        let config = hf_config.to_model_config();
        let network = build_network(&weights, &config)?;

        Ok(Self {
            network,
            weights,
            config,
        })
    }

    /// Load a model with explicit configuration.
    pub fn load_with_config<P: AsRef<Path>>(path: P, config: ModelConfig) -> Result<Self> {
        let path = path.as_ref();
        info!(
            "Loading native model from: {} with config {:?}",
            path.display(),
            config.architecture
        );

        // Load weights based on extension
        let weights = load_weights(path)?;

        // Build network from weights using provided config
        let network = build_network(&weights, &config)?;

        Ok(Self {
            network,
            weights,
            config,
        })
    }

    /// Get network specification.
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Get weights.
    pub fn weights(&self) -> &WeightStore {
        &self.weights
    }

    /// Convert to a propagate-compatible network.
    ///
    /// This creates a `ny_propagate::Network` that can be used for
    /// bound propagation and verification.
    pub fn to_propagate_network(&self) -> Result<PropNetwork> {
        // Reuse OnnxModel's conversion logic by creating a temporary OnnxModel
        let onnx_model = OnnxModel::empty_with_network(self.network.clone(), self.weights.clone());
        onnx_model.to_propagate_network()
    }

    /// Convert to a GraphNetwork for DAG-based bound propagation.
    ///
    /// Unlike `to_propagate_network()` which creates a sequential network,
    /// this builds a proper directed acyclic graph (DAG) that can handle
    /// binary operations like attention MatMul (Q@K^T) where both inputs
    /// are bounded tensors.
    ///
    /// Use this for models with attention (self-attention, cross-attention)
    /// or other branching/merging patterns.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ny_onnx::native::NativeModel;
    /// let model = NativeModel::load("whisper.safetensors").unwrap();
    /// let graph = model.to_graph_network().unwrap();
    /// // let output_bounds = graph.propagate_ibp(&input_bounds).unwrap();
    /// ```
    pub fn to_graph_network(&self) -> Result<ny_propagate::GraphNetwork> {
        // Reuse OnnxModel's graph network conversion
        let onnx_model = OnnxModel::empty_with_network(self.network.clone(), self.weights.clone());
        onnx_model.to_graph_network()
    }
}
