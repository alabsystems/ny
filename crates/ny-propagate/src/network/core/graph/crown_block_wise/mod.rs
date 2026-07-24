// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-block CROWN backward propagation for transformer networks.
//!
//! Applies CROWN backward propagation within each transformer block (between
//! normalization boundaries) while using IBP at block boundaries. This exploits
//! CROWN's tightness for activation relaxation (ReLU/GELU/SiLU) within blocks
//! while accepting that normalization layers act as correlation firewalls.
//!
//! # Motivation
//!
//! Whole-network CROWN through normalization layers produces bounds identical to
//! IBP (CROWN/IBP = 1.000) because normalization's dense Jacobian destroys the
//! correlation structure CROWN maintains through activation linearization.
//! Per-block CROWN captures within-block correlations where CROWN/IBP ≈ 0.04.
//!
//! # Design
//!
//! See `designs/2026-03-03-per-block-crown-transformer-verification.md`.
//! Part of #3221, #287.

use std::collections::{HashMap, HashSet};

use crate::bounds::BatchedLinearBounds;
use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::GraphNetwork;

mod alpha;
mod backward;
mod entrypoints;
mod ibp;

/// Explicit block specification for consumer-supplied block boundaries.
///
/// Allows traced or imported graphs whose node names do not follow the
/// `layer{N}` convention to supply block boundaries directly, preserving
/// consumer-chosen block indices and block names through to the
/// `BlockWiseCrownResult`.
///
/// Part of #4024.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// Ordered list of block entries (must be in topological order).
    pub blocks: Vec<BlockSpecEntry>,
}

/// A single block in a `BlockSpec`.
#[derive(Debug, Clone)]
pub struct BlockSpecEntry {
    /// Consumer-chosen block index (need not be contiguous).
    pub block_index: usize,
    /// Consumer-chosen block name (e.g., "encoder_block_3").
    pub block_name: String,
    /// Node names belonging to this block, in topological order.
    pub node_names: Vec<String>,
}

impl BlockSpec {
    /// Validate this block spec against the given graph.
    ///
    /// Checks:
    /// 1. Every block is non-empty.
    /// 2. Every node name exists in the graph.
    /// 3. No node appears in multiple blocks.
    /// 4. Concatenated node order is strictly increasing in graph topological order.
    fn validate(&self, graph: &GraphNetwork, exec_order: &[String]) -> Result<()> {
        // Build position lookup from topological order.
        let topo_pos: HashMap<&str, usize> = exec_order
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        let mut seen_nodes: HashSet<&str> = HashSet::new();
        let mut prev_max_pos: Option<usize> = None;

        for entry in &self.blocks {
            if entry.node_names.is_empty() {
                return Err(NyError::InvalidSpec(format!(
                    "BlockSpec: block '{}' (index {}) is empty",
                    entry.block_name, entry.block_index
                )));
            }

            for node_name in &entry.node_names {
                // Check node exists in graph.
                let pos = topo_pos.get(node_name.as_str()).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "BlockSpec: node '{}' in block '{}' does not exist in graph",
                        node_name, entry.block_name
                    ))
                })?;
                let _ = graph.node(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "BlockSpec: node '{}' in block '{}' not found in graph nodes",
                        node_name, entry.block_name
                    ))
                })?;

                // Check no duplicates across blocks.
                if !seen_nodes.insert(node_name.as_str()) {
                    return Err(NyError::InvalidSpec(format!(
                        "BlockSpec: node '{}' appears in multiple blocks",
                        node_name
                    )));
                }

                // Check topological ordering across concatenated spec.
                if let Some(prev) = prev_max_pos {
                    if *pos <= prev {
                        return Err(NyError::InvalidSpec(format!(
                            "BlockSpec: node '{}' (topo pos {}) violates strictly \
                             increasing topological order (previous max pos {})",
                            node_name, pos, prev
                        )));
                    }
                }
                prev_max_pos = Some(*pos);
            }
        }

        Ok(())
    }
}

/// Per-block GELU alpha state for alpha-CROWN optimization.
///
/// Stores per-neuron alpha parameters (tangent point control) for each GELU
/// layer within a block. Alpha in [0, 1] controls where the tangent line is
/// drawn for the lower bound in the convex region.
///
/// Part of #3221 Phase 4.
#[derive(Debug, Clone)]
pub struct BlockAlphaState {
    /// Map from GELU node name to per-neuron alpha array.
    /// Alpha = 0.5 is the midpoint tangent (default sound relaxation behavior).
    pub gelu_alphas: HashMap<String, Array1<f32>>,
}

impl BlockAlphaState {
    /// Create alpha state initialized at midpoint (alpha=0.5) for all GELU nodes.
    pub(crate) fn new_midpoint(gelu_nodes: &[(String, usize)]) -> Self {
        let mut gelu_alphas = HashMap::with_capacity(gelu_nodes.len());
        for (name, dim) in gelu_nodes {
            gelu_alphas.insert(name.clone(), Array1::from_elem(*dim, 0.5_f32));
        }
        Self { gelu_alphas }
    }
}

/// Result of per-block CROWN verification comparing CROWN vs IBP per block.
#[derive(Debug, Clone)]
pub struct BlockWiseCrownResult {
    /// Per-block comparison of CROWN vs IBP bound widths.
    pub blocks: Vec<BlockCrownComparison>,
    /// Total blocks processed.
    pub total_blocks: usize,
    /// Epsilon used for each block's fresh input.
    pub block_epsilon: f32,
}

/// Per-block comparison of CROWN vs IBP bound widths.
#[derive(Debug, Clone)]
pub struct BlockCrownComparison {
    /// Block index (from node naming: layer0, layer1, ...).
    pub block_index: usize,
    /// Block name (e.g., "layer0").
    pub block_name: String,
    /// Maximum output width from IBP (block-wise reset).
    pub ibp_max_width: f32,
    /// Maximum output width from per-block CROWN.
    pub crown_max_width: f32,
    /// CROWN/IBP ratio (< 1.0 means CROWN is tighter).
    pub crown_ibp_ratio: f32,
    /// Whether CROWN succeeded for this block (false = fell back to IBP).
    pub crown_successful: bool,
    /// Maximum output width from alpha-CROWN (optimized GELU slopes), if computed.
    pub alpha_crown_max_width: Option<f32>,
    /// Alpha-CROWN/IBP ratio, if computed.
    pub alpha_crown_ibp_ratio: Option<f32>,
}

/// Per-site decomposed-normalization validation stats aggregated across positions.
///
/// Used to surface `fallback_rows/total_rows` per LayerNorm/RmsNorm/
/// InstanceNorm1d site through the Whisper measurement path. The type name is
/// retained for API stability. Part of #318, #3892.
#[derive(Debug, Clone)]
pub struct LayerNormValidationStats {
    pub node_name: String,
    pub fallback_rows: usize,
    pub total_rows: usize,
}

// --- Utility methods used by backward.rs ---

impl GraphNetwork {
    /// Create bias-only BatchedLinearBounds from an interval BoundedTensor.
    ///
    /// Used when a sub-path is concretized via partial fallback (e.g., at
    /// LayerNorm) and the resulting interval needs to be accumulated as a
    /// constant contribution alongside other linear paths (e.g., residual).
    ///
    /// The result has A=0 (no dependence on input) and b=interval bounds.
    pub(super) fn bias_only_bounds_from_interval(
        interval: &BoundedTensor,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<BatchedLinearBounds> {
        let in_dim = input_shape.last().copied().unwrap_or(1);
        let out_dim = output_shape.last().copied().unwrap_or(1);
        let out_batch: Vec<usize> = output_shape[..output_shape.len().saturating_sub(1)].to_vec();

        // A shape: [out_batch..., out_dim, in_dim]
        let mut a_shape = out_batch.clone();
        a_shape.push(out_dim);
        a_shape.push(in_dim);

        let lower_a = ArrayD::zeros(IxDyn(&a_shape));
        let upper_a = ArrayD::zeros(IxDyn(&a_shape));

        // Bias = the interval's lower/upper bounds, reshaped to [out_batch..., out_dim].
        let mut b_shape = out_batch;
        b_shape.push(out_dim);

        let lower_b = interval
            .lower()
            .clone()
            .into_shape_with_order(IxDyn(&b_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "bias_only_bounds_from_interval: reshape lower failed: {}",
                    e
                ))
            })?;
        let upper_b = interval
            .upper()
            .clone()
            .into_shape_with_order(IxDyn(&b_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "bias_only_bounds_from_interval: reshape upper failed: {}",
                    e
                ))
            })?;

        BatchedLinearBounds::new(
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            input_shape.to_vec(),
            output_shape.to_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    use super::GraphNetwork;

    fn bounded_tensor(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape valid"),
            ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape valid"),
        )
        .expect("bounds valid")
    }

    #[test]
    fn test_bias_only_bounds_scalar_interval() {
        let interval = bounded_tensor(&[2], vec![1.0, 2.0], vec![3.0, 4.0]);

        let bounds =
            GraphNetwork::bias_only_bounds_from_interval(&interval, &[3], &[2]).expect("valid");

        assert_eq!(bounds.input_shape(), &[3]);
        assert_eq!(bounds.output_shape(), &[2]);
        assert_eq!(bounds.lower_a().shape(), &[2, 3]);
        assert_eq!(bounds.upper_a().shape(), &[2, 3]);
        assert!(
            bounds.lower_a().iter().all(|value| *value == 0.0),
            "lower_a should be all zeros for bias-only bounds"
        );
        assert!(
            bounds.upper_a().iter().all(|value| *value == 0.0),
            "upper_a should be all zeros for bias-only bounds"
        );
        assert_eq!(bounds.lower_b().shape(), &[2]);
        assert_eq!(bounds.upper_b().shape(), &[2]);
        assert_eq!(
            bounds.lower_b(),
            &ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("shape valid")
        );
        assert_eq!(
            bounds.upper_b(),
            &ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).expect("shape valid")
        );
    }

    #[test]
    fn test_bias_only_bounds_batched_interval() {
        let interval = bounded_tensor(
            &[4, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        );

        let bounds = GraphNetwork::bias_only_bounds_from_interval(&interval, &[4, 3], &[4, 2])
            .expect("valid");

        assert_eq!(bounds.input_shape(), &[4, 3]);
        assert_eq!(bounds.output_shape(), &[4, 2]);
        assert_eq!(bounds.lower_a().shape(), &[4, 2, 3]);
        assert_eq!(bounds.upper_a().shape(), &[4, 2, 3]);
        assert!(
            bounds.lower_a().iter().all(|value| *value == 0.0),
            "lower_a should be all zeros for batched bias-only bounds"
        );
        assert!(
            bounds.upper_a().iter().all(|value| *value == 0.0),
            "upper_a should be all zeros for batched bias-only bounds"
        );
        assert_eq!(bounds.lower_b().shape(), &[4, 2]);
        assert_eq!(bounds.upper_b().shape(), &[4, 2]);
        assert_eq!(
            bounds.lower_b(),
            &ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                .expect("shape valid")
        );
        assert_eq!(
            bounds.upper_b(),
            &ArrayD::from_shape_vec(
                IxDyn(&[4, 2]),
                vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
            )
            .expect("shape valid")
        );
    }

    #[test]
    fn test_bias_only_bounds_shape_mismatch_returns_error() {
        let interval = bounded_tensor(&[2], vec![1.0, 2.0], vec![3.0, 4.0]);

        let error = GraphNetwork::bias_only_bounds_from_interval(&interval, &[3], &[5])
            .expect_err("shape mismatch should fail");

        match error {
            NyError::InternalError(message) => {
                assert!(
                    message.contains("reshape lower failed"),
                    "expected lower reshape failure, got: {message}"
                );
            }
            other => panic!("expected InternalError, got {other:?}"),
        }
    }

    #[test]
    fn test_bias_only_bounds_empty_shapes_default_to_one() {
        let interval = bounded_tensor(&[], vec![1.5], vec![2.5]);

        let bounds =
            GraphNetwork::bias_only_bounds_from_interval(&interval, &[], &[]).expect("valid");

        assert!(bounds.input_shape().is_empty());
        assert!(bounds.output_shape().is_empty());
        assert_eq!(bounds.lower_a().shape(), &[1, 1]);
        assert_eq!(bounds.upper_a().shape(), &[1, 1]);
        assert_eq!(bounds.lower_b().shape(), &[1]);
        assert_eq!(bounds.upper_b().shape(), &[1]);
        assert_eq!(bounds.lower_a()[[0, 0]], 0.0);
        assert_eq!(bounds.upper_a()[[0, 0]], 0.0);
        assert_eq!(bounds.lower_b()[[0]], 1.5);
        assert_eq!(bounds.upper_b()[[0]], 2.5);
    }
}
