// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{s, Array1, Array2, Axis};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use super::shared::compute_crown_or_ibp_bounds;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::GraphNetwork;

/// Root-build wrapper that chunks large spec matrices into smaller batches.
///
/// Mirrors alpha-beta-CROWN `solver.build_batch_size` for the graph
/// input-split warmup path: large multi-row spec matrices can be evaluated in
/// smaller slices, then concatenated back into one dense-spec result for the
/// rest of the BaB loop. This keeps the child/rebound paths unchanged while
/// reducing initialization pressure on large nn4sys-style properties.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds_in_build_batches(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    build_batch_size: Option<usize>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    let chunk_size = match build_batch_size {
        Some(0) => {
            return Err(NyError::InvalidConfig(
                "build_batch_size must be >= 1 when set".to_string(),
            ));
        }
        Some(size) if spec_matrix.nrows() > size => size,
        _ => {
            return compute_crown_or_ibp_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                alpha_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
                ibp_enhancement,
            );
        }
    };

    tracing::info!(
        spec_rows = spec_matrix.nrows(),
        build_batch_size = chunk_size,
        "Chunking graph input-split root spec build"
    );

    let mut bounds_chunks = Vec::new();
    let mut linear_chunks = Vec::new();
    let mut missing_linear_bounds = false;

    for start in (0..spec_matrix.nrows()).step_by(chunk_size) {
        let end = (start + chunk_size).min(spec_matrix.nrows());
        let spec_chunk = spec_matrix.slice(s![start..end, ..]).to_owned();
        let (chunk_bounds, chunk_linear) = compute_crown_or_ibp_bounds(
            graph,
            input_bounds,
            &spec_chunk,
            engine,
            alpha_node_bounds,
            alpha_state,
            mul_binary_alphas,
            deadline,
            crown_backward_layers,
            ibp_enhancement,
        )?;
        bounds_chunks.push(chunk_bounds);
        match chunk_linear {
            Some(linear) => linear_chunks.push(linear),
            None => missing_linear_bounds = true,
        }
    }

    let combined_bounds = BoundedTensor::concat(&bounds_chunks, 0)?;
    let combined_linear = if missing_linear_bounds {
        None
    } else {
        Some(stack_linear_bounds_row_batches_4354(&linear_chunks)?)
    };
    Ok((combined_bounds, combined_linear))
}

fn stack_linear_bounds_row_batches_4354(chunks: &[LinearBounds]) -> Result<LinearBounds> {
    let Some(first) = chunks.first() else {
        return Err(NyError::InvalidSpec(
            "cannot stack empty LinearBounds chunk list".to_string(),
        ));
    };

    let lower_a_views: Vec<_> = chunks.iter().map(|chunk| chunk.lower_a().view()).collect();
    let upper_a_views: Vec<_> = chunks.iter().map(|chunk| chunk.upper_a().view()).collect();
    let lower_a = ndarray::concatenate(Axis(0), &lower_a_views)
        .map_err(|e| NyError::InvalidSpec(format!("stack build_batch_size lower_a: {e}")))?;
    let upper_a = ndarray::concatenate(Axis(0), &upper_a_views)
        .map_err(|e| NyError::InvalidSpec(format!("stack build_batch_size upper_a: {e}")))?;

    let mut lower_b_data = Vec::with_capacity(chunks.iter().map(LinearBounds::num_outputs).sum());
    let mut upper_b_data = Vec::with_capacity(chunks.iter().map(LinearBounds::num_outputs).sum());
    for chunk in chunks {
        if chunk.num_inputs() != first.num_inputs() {
            return Err(NyError::InvalidSpec(format!(
                "build_batch_size LinearBounds input width mismatch: expected {}, got {}",
                first.num_inputs(),
                chunk.num_inputs()
            )));
        }
        lower_b_data.extend(chunk.lower_b().iter().copied());
        upper_b_data.extend(chunk.upper_b().iter().copied());
    }

    LinearBounds::new(
        lower_a,
        Array1::from_vec(lower_b_data),
        upper_a,
        Array1::from_vec(upper_b_data),
    )
}
