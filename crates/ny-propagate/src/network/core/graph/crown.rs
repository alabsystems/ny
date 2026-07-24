// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Non-delegating CROWN methods for GraphNetwork.
//!
//! Delegation methods (that imported from sibling `graph_crown`) moved to
//! `network/dispatch/graph_crown.rs` (#2380). This file retains only methods
//! that are self-contained (no sibling imports).

use std::borrow::Cow;

use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::crown_block_wise::LayerNormValidationStats;
use super::GraphNetwork;

impl GraphNetwork {
    /// Propagate CROWN bounds through the graph, treating each position independently.
    ///
    /// For N-D inputs where the last dimension is the feature dimension, this runs
    /// CROWN separately on each position (flattened batch dimensions) and combines
    /// the results. This is useful for transformer MLPs which operate independently
    /// on each position.
    ///
    /// # Arguments
    /// * `input` - Bounded tensor of shape [...batch_dims..., hidden_dim]
    ///
    /// # Returns
    /// * Bounded tensor of shape [...batch_dims..., output_dim]
    ///
    /// # Algorithm
    /// 1. Flatten input from [...batch, hidden] to [num_positions, hidden]
    /// 2. For each position, extract \[hidden\] slice and run CROWN
    /// 3. Stack outputs and reshape to [...batch, output_dim]
    #[inline]
    pub fn propagate_crown_per_position(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_per_position_with_engine(input, None)
    }

    /// Engine-aware variant of `propagate_crown_per_position` (#3772).
    pub fn propagate_crown_per_position_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();

        // Guard: scalar input (ndim==0) would underflow shape[ndim-1] indexing (#2690).
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "propagate_crown_per_position: scalar input (ndim=0) is not supported".into(),
            ));
        }

        // For 1-D input, just use regular CROWN
        if ndim == 1 {
            return self.propagate_crown_with_engine(input, engine);
        }

        // Extract batch dimensions and hidden dimension
        let hidden_dim = shape[ndim - 1];
        let batch_shape: Vec<usize> = shape[..ndim - 1].to_vec();
        let num_positions = checked_shape_product(&batch_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Graph CROWN per-position: batch shape product overflows: {:?}",
                batch_shape
            ))
        })?;

        // Guard: zero-batch input would panic at flat_lower.row(0) (#2690).
        if num_positions == 0 {
            return Err(NyError::InvalidSpec(format!(
                "propagate_crown_per_position: zero-batch input (shape={shape:?}, \
                 num_positions=0) is not supported"
            )));
        }

        debug!(
            "Per-position CROWN: {} positions x {} hidden, batch shape {:?}",
            num_positions, hidden_dim, batch_shape
        );

        // Flatten input to [num_positions, hidden_dim]
        // Use Cow to avoid cloning when data is already contiguous (zero-copy path).
        // Only convert to owned when memory layout requires it.
        let lower_cow: Cow<'_, ndarray::ArrayD<f32>> = if input.lower().is_standard_layout() {
            Cow::Borrowed(input.lower())
        } else {
            Cow::Owned(input.lower().as_standard_layout().to_owned())
        };
        let upper_cow: Cow<'_, ndarray::ArrayD<f32>> = if input.upper().is_standard_layout() {
            Cow::Borrowed(input.upper())
        } else {
            Cow::Owned(input.upper().as_standard_layout().to_owned())
        };

        let target_shape = (num_positions, hidden_dim);
        let flat_lower = lower_cow
            .view()
            .into_shape_with_order(target_shape)
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Failed to reshape lower from {:?} to {:?}: {:?}",
                    shape, target_shape, e
                ))
            })?;
        let flat_upper = upper_cow
            .view()
            .into_shape_with_order(target_shape)
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Failed to reshape upper from {:?} to {:?}: {:?}",
                    shape, target_shape, e
                ))
            })?;

        // Run CROWN on first position to determine output dimension
        let first_lower = flat_lower.row(0).to_owned().into_dyn();
        let first_upper = flat_upper.row(0).to_owned().into_dyn();
        let first_input = BoundedTensor::new(first_lower, first_upper)?;
        let first_output = self.propagate_crown_with_engine(&first_input, engine)?;
        let output_dim = first_output.len();

        // Allocate output arrays
        let mut out_lower = ndarray::Array2::<f32>::zeros((num_positions, output_dim));
        let mut out_upper = ndarray::Array2::<f32>::zeros((num_positions, output_dim));

        // Copy first result
        {
            let first_out_lower = first_output
                .lower()
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    NyError::shape_mismatch(vec![output_dim], first_output.lower().shape().to_vec())
                })?;
            let first_out_upper = first_output
                .upper()
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    NyError::shape_mismatch(vec![output_dim], first_output.upper().shape().to_vec())
                })?;
            out_lower.row_mut(0).assign(&first_out_lower);
            out_upper.row_mut(0).assign(&first_out_upper);
        }

        // Process remaining positions
        for pos in 1..num_positions {
            let pos_lower = flat_lower.row(pos).to_owned().into_dyn();
            let pos_upper = flat_upper.row(pos).to_owned().into_dyn();
            let pos_input = BoundedTensor::new(pos_lower, pos_upper)?;

            let pos_output = self.propagate_crown_with_engine(&pos_input, engine)?;

            let pos_out_lower = pos_output
                .lower()
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    NyError::shape_mismatch(vec![output_dim], pos_output.lower().shape().to_vec())
                })?;
            let pos_out_upper = pos_output
                .upper()
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    NyError::shape_mismatch(vec![output_dim], pos_output.upper().shape().to_vec())
                })?;

            out_lower.row_mut(pos).assign(&pos_out_lower);
            out_upper.row_mut(pos).assign(&pos_out_upper);
        }

        // Reshape output to [...batch_dims..., output_dim]
        let mut output_shape = batch_shape;
        output_shape.push(output_dim);

        let out_lower_nd = out_lower
            .into_dyn()
            .into_shape_with_order(ndarray::IxDyn(&output_shape))
            .map_err(|_| {
                NyError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
            })?;
        let out_upper_nd = out_upper
            .into_dyn()
            .into_shape_with_order(ndarray::IxDyn(&output_shape))
            .map_err(|_| {
                NyError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
            })?;

        BoundedTensor::new(out_lower_nd, out_upper_nd)
    }

    /// Per-position CROWN through block-wise backward with decomposed normalization.
    ///
    /// Like `propagate_crown_per_position`, but uses `propagate_crown_within_graph`
    /// which attempts decomposed LayerNorm/RmsNorm backward inside each block
    /// before falling back to interval-derived bias-only bounds.
    ///
    /// This path is primarily for #318-style transformer experiments. Current
    /// measurements show that forward-mode LayerNorm decomposition behaves like
    /// Cut plus forward-mode IBP, and the fully sound decomposition can still be
    /// looser than fused LayerNorm IBP when normalization dominates block width.
    ///
    /// Part of #318.
    pub fn propagate_crown_within_graph_per_position(
        &self,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_within_graph_per_position_with_engine(input, None)
    }

    /// Engine-aware variant of `propagate_crown_within_graph_per_position` (#3772).
    pub fn propagate_crown_within_graph_per_position_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let (result, _stats) =
            self.propagate_crown_within_graph_per_position_with_stats_and_engine(input, engine)?;
        Ok(result)
    }

    /// Per-position decomposed-norm CROWN with row-collapse stats aggregation.
    ///
    /// Same as `propagate_crown_within_graph_per_position` but returns per-site
    /// `LayerNormValidationStats` with `fallback_rows`/`total_rows` summed across
    /// all positions for LayerNorm/RmsNorm/InstanceNorm1d sites. Sorted by
    /// `node_name` for deterministic output. Part of #318, #3892.
    pub fn propagate_crown_within_graph_per_position_with_stats(
        &self,
        input: &BoundedTensor,
    ) -> Result<(BoundedTensor, Vec<LayerNormValidationStats>)> {
        self.propagate_crown_within_graph_per_position_with_stats_and_engine(input, None)
    }

    /// Engine-aware variant of `propagate_crown_within_graph_per_position_with_stats` (#3772).
    pub fn propagate_crown_within_graph_per_position_with_stats_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, Vec<LayerNormValidationStats>)> {
        use std::collections::HashMap;

        let shape = input.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "propagate_crown_within_graph_per_position: scalar input not supported".into(),
            ));
        }

        // For 1-D input, use block-wise CROWN directly.
        if ndim == 1 {
            return self.propagate_crown_within_graph_with_stats_and_engine(input, engine);
        }

        let hidden_dim = shape[ndim - 1];
        let batch_shape: Vec<usize> = shape[..ndim - 1].to_vec();
        let num_positions: usize = checked_shape_product(&batch_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "crown_within_graph_per_position: batch shape {batch_shape:?} overflow",
            ))
        })?;

        if num_positions == 0 {
            return Err(NyError::InvalidSpec(
                "crown_within_graph_per_position: zero-batch input".into(),
            ));
        }

        debug!(
            "Per-position decomposed-norm CROWN: {} positions x {} hidden",
            num_positions, hidden_dim,
        );

        let lower_cow: Cow<'_, ndarray::ArrayD<f32>> = if input.lower().is_standard_layout() {
            Cow::Borrowed(input.lower())
        } else {
            Cow::Owned(input.lower().as_standard_layout().to_owned())
        };
        let upper_cow: Cow<'_, ndarray::ArrayD<f32>> = if input.upper().is_standard_layout() {
            Cow::Borrowed(input.upper())
        } else {
            Cow::Owned(input.upper().as_standard_layout().to_owned())
        };

        let target_shape = (num_positions, hidden_dim);
        let flat_lower = lower_cow
            .view()
            .into_shape_with_order(target_shape)
            .map_err(|e| {
                NyError::InvalidSpec(format!("reshape lower from {:?}: {:?}", shape, e))
            })?;
        let flat_upper = upper_cow
            .view()
            .into_shape_with_order(target_shape)
            .map_err(|e| {
                NyError::InvalidSpec(format!("reshape upper from {:?}: {:?}", shape, e))
            })?;

        // Aggregated stats: sum fallback_rows and total_rows per node_name.
        let mut stats_map: HashMap<String, (usize, usize)> = HashMap::new();

        // Process first position to determine output dimension.
        let first_lower = flat_lower.row(0).to_owned().into_dyn();
        let first_upper = flat_upper.row(0).to_owned().into_dyn();
        let first_input = BoundedTensor::new(first_lower, first_upper)?;
        let (first_output, first_stats) =
            self.propagate_crown_within_graph_with_stats_and_engine(&first_input, engine)?;
        let output_dim = first_output.len();

        for s in &first_stats {
            let entry = stats_map.entry(s.node_name.clone()).or_insert((0, 0));
            entry.0 += s.fallback_rows;
            entry.1 += s.total_rows;
        }

        let mut out_lower = ndarray::Array2::<f32>::zeros((num_positions, output_dim));
        let mut out_upper = ndarray::Array2::<f32>::zeros((num_positions, output_dim));

        {
            let first_out_lower = first_output
                .lower()
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    NyError::shape_mismatch(vec![output_dim], first_output.lower().shape().to_vec())
                })?;
            let first_out_upper = first_output
                .upper()
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    NyError::shape_mismatch(vec![output_dim], first_output.upper().shape().to_vec())
                })?;
            out_lower.row_mut(0).assign(&first_out_lower);
            out_upper.row_mut(0).assign(&first_out_upper);
        }

        for pos in 1..num_positions {
            let pos_lower = flat_lower.row(pos).to_owned().into_dyn();
            let pos_upper = flat_upper.row(pos).to_owned().into_dyn();
            let pos_input = BoundedTensor::new(pos_lower, pos_upper)?;

            let (pos_output, pos_stats) =
                self.propagate_crown_within_graph_with_stats_and_engine(&pos_input, engine)?;

            for s in &pos_stats {
                let entry = stats_map.entry(s.node_name.clone()).or_insert((0, 0));
                entry.0 += s.fallback_rows;
                entry.1 += s.total_rows;
            }

            let pos_out_lower = pos_output
                .lower()
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    NyError::shape_mismatch(vec![output_dim], pos_output.lower().shape().to_vec())
                })?;
            let pos_out_upper = pos_output
                .upper()
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    NyError::shape_mismatch(vec![output_dim], pos_output.upper().shape().to_vec())
                })?;

            out_lower.row_mut(pos).assign(&pos_out_lower);
            out_upper.row_mut(pos).assign(&pos_out_upper);
        }

        let mut output_shape = batch_shape;
        output_shape.push(output_dim);

        let out_lower_nd = out_lower
            .into_dyn()
            .into_shape_with_order(ndarray::IxDyn(&output_shape))
            .map_err(|_| {
                NyError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
            })?;
        let out_upper_nd = out_upper
            .into_dyn()
            .into_shape_with_order(ndarray::IxDyn(&output_shape))
            .map_err(|_| {
                NyError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
            })?;

        // Aggregate stats sorted by node_name for deterministic output.
        let mut aggregated_stats: Vec<LayerNormValidationStats> = stats_map
            .into_iter()
            .map(|(name, (fb, total))| LayerNormValidationStats {
                node_name: name,
                fallback_rows: fb,
                total_rows: total,
            })
            .collect();
        aggregated_stats.sort_by(|a, b| a.node_name.cmp(&b.node_name));

        let result = BoundedTensor::new(out_lower_nd, out_upper_nd)?;
        Ok((result, aggregated_stats))
    }
}
