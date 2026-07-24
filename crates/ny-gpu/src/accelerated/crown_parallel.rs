// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-position CROWN bound propagation with parallel and sequential modes.
//!
//! For transformer verification, each position in the sequence can be verified
//! independently since position-independent layers (Linear, LayerNorm, GELU,
//! etc.) don't create cross-position dependencies.

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use tracing::debug;

/// Parallel CROWN for per-position bound propagation.
///
/// For transformer verification, each position in the sequence can be
/// verified independently since position-independent layers (Linear, LayerNorm,
/// GELU, etc.) don't create cross-position dependencies.
///
/// This function parallelizes CROWN execution across positions using Rayon,
/// providing significant speedup proportional to the number of CPU cores.
pub fn crown_per_position_parallel(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Result<BoundedTensor> {
    crown_per_position_parallel_with_engine(graph, input, None)
}

/// Sequential per-position CROWN with optional GEMM acceleration.
///
/// This is primarily used by GPU engines that cannot safely participate in
/// Rayon parallel execution due to internal buffer reuse or thread-affinity
/// constraints. It still provides acceleration by offloading GEMM operations
/// within CROWN backward propagation to the provided `engine`.
pub fn crown_per_position_sequential_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = shape.len();

    // For 1-D input, just use regular CROWN (no per-position structure)
    if ndim == 1 {
        return graph.propagate_crown_with_engine(input, engine);
    }

    // Extract batch dimensions and hidden dimension
    let hidden_dim = shape[ndim - 1];
    let batch_shape: Vec<usize> = shape[..ndim - 1].to_vec();
    let num_positions: usize = checked_shape_product(&batch_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "crown_per_position_sequential: batch shape {batch_shape:?} overflow usize",
        ))
    })?;

    // Guard: zero-batch input → no positions to process (#2870, #2920 WP-C).
    if num_positions == 0 {
        return Err(NyError::InvalidSpec(
            "crown_per_position_sequential: zero-batch input (num_positions=0)".into(),
        ));
    }

    debug!(
        "Sequential per-position CROWN{}: {} positions x {} hidden, batch shape {:?}",
        if engine.is_some() {
            " (engine-accelerated)"
        } else {
            ""
        },
        num_positions,
        hidden_dim,
        batch_shape
    );

    // Flatten input to [num_positions, hidden_dim]
    // Make arrays contiguous first to avoid reshape failures due to memory layout.
    let lower_contiguous = if input.lower().is_standard_layout() {
        input.lower().clone()
    } else {
        input.lower().as_standard_layout().to_owned()
    };
    let upper_contiguous = if input.upper().is_standard_layout() {
        input.upper().clone()
    } else {
        input.upper().as_standard_layout().to_owned()
    };

    let target_shape = (num_positions, hidden_dim);
    let flat_lower = lower_contiguous
        .into_shape_with_order(target_shape)
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "Failed to reshape lower from {:?} to {:?}: {:?}",
                shape, target_shape, e
            ))
        })?;
    let flat_upper = upper_contiguous
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
    let first_output = graph.propagate_crown_with_engine(&first_input, engine)?;
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

        let pos_output = graph.propagate_crown_with_engine(&pos_input, engine)?;

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

    // Sanitize NaN/Inf values before creating BoundedTensor
    // CROWN can produce overflow when bound widths explode through deep networks.
    // Replace NaN/Inf with conservative fallback bounds to maintain soundness.
    let mut sanitized_count = 0;
    use crate::FALLBACK_BOUND;

    for pos in 0..num_positions {
        for i in 0..output_dim {
            let l = out_lower[[pos, i]];
            let u = out_upper[[pos, i]];
            if l.is_nan() || l.is_infinite() || u.is_nan() || u.is_infinite() {
                out_lower[[pos, i]] = -FALLBACK_BOUND;
                out_upper[[pos, i]] = FALLBACK_BOUND;
                sanitized_count += 1;
            }
        }
    }

    if sanitized_count > 0 {
        debug!(
            "crown_per_position_sequential_with_engine: sanitized {} NaN/Inf values ({}% of output)",
            sanitized_count,
            100.0 * sanitized_count as f64 / (num_positions * output_dim) as f64
        );
    }

    // Reshape output to [...batch_dims..., output_dim]
    let mut output_shape = batch_shape;
    output_shape.push(output_dim);

    let out_lower_nd = out_lower
        .into_dyn()
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| {
            NyError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
        })?;
    let out_upper_nd = out_upper
        .into_dyn()
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| {
            NyError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
        })?;

    BoundedTensor::new(out_lower_nd, out_upper_nd)
}

/// Parallel CROWN for per-position bound propagation with optional GPU acceleration.
///
/// Same as `crown_per_position_parallel` but accepts an optional `GemmEngine` for
/// GPU-accelerated matrix operations within CROWN backward propagation.
///
/// When `engine` is `Some`, the GEMM operations within each position's CROWN
/// propagation will be accelerated using the provided engine (e.g., GPU via wgpu).
pub fn crown_per_position_parallel_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = shape.len();

    // For 1-D input, just use regular CROWN (no parallelization benefit)
    if ndim == 1 {
        return graph.propagate_crown_with_engine(input, engine);
    }

    // Extract batch dimensions and hidden dimension
    let hidden_dim = shape[ndim - 1];
    let batch_shape: Vec<usize> = shape[..ndim - 1].to_vec();
    let num_positions: usize = checked_shape_product(&batch_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "crown_per_position_parallel: batch shape {batch_shape:?} overflow usize",
        ))
    })?;

    // Guard: zero-batch input → no positions to process (#2870, #2920 WP-C).
    if num_positions == 0 {
        return Err(NyError::InvalidSpec(
            "crown_per_position_parallel: zero-batch input (num_positions=0)".into(),
        ));
    }

    debug!(
        "Parallel per-position CROWN{}: {} positions x {} hidden, batch shape {:?}",
        if engine.is_some() {
            " (GPU-accelerated)"
        } else {
            ""
        },
        num_positions,
        hidden_dim,
        batch_shape
    );

    // Flatten input to [num_positions, hidden_dim]
    let flat_lower = input
        .lower()
        .clone()
        .into_shape_with_order((num_positions, hidden_dim))
        .map_err(|_| NyError::shape_mismatch(vec![num_positions, hidden_dim], shape.to_vec()))?;
    let flat_upper = input
        .upper()
        .clone()
        .into_shape_with_order((num_positions, hidden_dim))
        .map_err(|_| NyError::shape_mismatch(vec![num_positions, hidden_dim], shape.to_vec()))?;

    // Run CROWN on first position to determine output dimension
    let first_lower = flat_lower.row(0).to_owned().into_dyn();
    let first_upper = flat_upper.row(0).to_owned().into_dyn();
    let first_input = BoundedTensor::new(first_lower, first_upper)?;
    let first_output = graph.propagate_crown_with_engine(&first_input, engine)?;
    let output_dim = first_output.len();

    // Run CROWN in parallel across all positions
    // Access rows directly from flat_lower/flat_upper to avoid O(num_positions * hidden_dim) staging
    let results: Vec<Result<(Vec<f32>, Vec<f32>)>> = (0..num_positions)
        .into_par_iter()
        .map(|pos| {
            // Create BoundedTensor for this position by copying row data
            // Note: row() returns an ArrayView, to_vec() copies into Vec<f32>
            let lower_row = flat_lower.row(pos).to_vec();
            let upper_row = flat_upper.row(pos).to_vec();
            let pos_lower = ArrayD::from_shape_vec(IxDyn(&[hidden_dim]), lower_row)
                .map_err(|_| NyError::shape_mismatch(vec![hidden_dim], vec![hidden_dim]))?;
            let pos_upper = ArrayD::from_shape_vec(IxDyn(&[hidden_dim]), upper_row)
                .map_err(|_| NyError::shape_mismatch(vec![hidden_dim], vec![hidden_dim]))?;
            let pos_input = BoundedTensor::new(pos_lower, pos_upper)?;

            // Run CROWN with optional engine
            let pos_output = graph.propagate_crown_with_engine(&pos_input, engine)?;

            // Extract results as Vec<f32>
            let out_lower = pos_output
                .lower()
                .as_slice()
                .ok_or_else(|| {
                    NyError::InternalError("crown_per_position: output lower not contiguous".into())
                })?
                .to_vec();
            let out_upper = pos_output
                .upper()
                .as_slice()
                .ok_or_else(|| {
                    NyError::InternalError("crown_per_position: output upper not contiguous".into())
                })?
                .to_vec();

            Ok((out_lower, out_upper))
        })
        .collect();

    // Check for errors and collect results
    let mut out_lower = ndarray::Array2::<f32>::zeros((num_positions, output_dim));
    let mut out_upper = ndarray::Array2::<f32>::zeros((num_positions, output_dim));

    for (pos, result) in results.into_iter().enumerate() {
        let (lower_row, upper_row) = result?;
        for (i, (&l, &u)) in lower_row.iter().zip(upper_row.iter()).enumerate() {
            out_lower[[pos, i]] = l;
            out_upper[[pos, i]] = u;
        }
    }

    // Sanitize NaN/Inf values before creating BoundedTensor
    // CROWN can produce overflow when bound widths explode through deep networks.
    // Replace NaN/Inf with conservative fallback bounds to maintain soundness.
    let mut sanitized_count = 0;
    use crate::FALLBACK_BOUND;

    for pos in 0..num_positions {
        for i in 0..output_dim {
            let l = out_lower[[pos, i]];
            let u = out_upper[[pos, i]];
            if l.is_nan() || l.is_infinite() || u.is_nan() || u.is_infinite() {
                out_lower[[pos, i]] = -FALLBACK_BOUND;
                out_upper[[pos, i]] = FALLBACK_BOUND;
                sanitized_count += 1;
            }
        }
    }

    if sanitized_count > 0 {
        debug!(
            "crown_per_position_parallel_with_engine: sanitized {} NaN/Inf values ({}% of output)",
            sanitized_count,
            100.0 * sanitized_count as f64 / (num_positions * output_dim) as f64
        );
    }

    // Reshape output to [...batch_dims..., output_dim]
    let mut output_shape = batch_shape;
    output_shape.push(output_dim);

    let out_lower_nd = out_lower
        .into_dyn()
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| {
            NyError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
        })?;
    let out_upper_nd = out_upper
        .into_dyn()
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| {
            NyError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
        })?;

    BoundedTensor::new(out_lower_nd, out_upper_nd)
}
