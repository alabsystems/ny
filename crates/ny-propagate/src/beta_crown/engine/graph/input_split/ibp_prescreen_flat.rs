// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Flat-row variant of batched IBP pre-screen for input-split BaB.
//!
//! Accepts flat 1D lower/upper arrays per child and a shared shape, avoiding
//! per-child BoundedTensor construction before the prescreen pass.
//! Part of #4366 Packet A.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::GraphNetwork;

use super::ibp_prescreen::{evaluate_batched_output, run_ibp_forward, validate_prescreen_layout};

/// Like [`super::ibp_prescreen::batched_ibp_prescreen`] but accepts flat 1D
/// lower/upper arrays per child and a shared shape, avoiding per-child
/// BoundedTensor construction. Builds a single stacked `(N, ...shape)` tensor
/// directly from the flat rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batched_ibp_prescreen_from_flat(
    graph: &GraphNetwork,
    flat_lowers: &[ArrayD<f32>],
    flat_uppers: &[ArrayD<f32>],
    child_shape: &[usize],
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: Option<&[usize]>,
    verify_upper_bound: bool,
    engine: Option<&dyn GemmEngine>,
) -> Result<Vec<bool>> {
    let n = flat_lowers.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if flat_uppers.len() != n {
        return Err(NyError::InvalidSpec(format!(
            "batched_ibp_prescreen_from_flat: lower count {} != upper count {}",
            n,
            flat_uppers.len()
        )));
    }

    validate_prescreen_layout(spec_matrix, thresholds, clause_sizes)?;

    let stacked_input = stack_flat_rows(flat_lowers, flat_uppers, child_shape)?;
    let sanitized = run_ibp_forward(graph, &stacked_input, engine)?;

    evaluate_batched_output(
        &sanitized,
        n,
        spec_matrix,
        thresholds,
        clause_sizes,
        verify_upper_bound,
    )
}

/// #lsnc-child-batch (S1): stacked-row variant of
/// [`batched_ibp_prescreen_from_flat`]. Accepts the PRE-STACKED
/// `(N, ...child_shape)` lower/upper arrays directly (the `ChildBatch` split
/// kernel writes child rows contiguously), skipping the per-child flat-array
/// clones and the `stack_flat_rows` re-copy. The stacked buffers still enter
/// through `BoundedTensor::new` (the NaN/Inf/inversion firewall, I-E4) and the
/// SAME `run_ibp_forward` + `evaluate_batched_output` leaf helpers, so given
/// bit-identical stacked rows the verified mask is bit-identical to the flat
/// entry. Parity: `test_child_batch_reorder_prescreen_parity_lsnc_s1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batched_ibp_prescreen_from_stacked(
    graph: &GraphNetwork,
    stacked_lower: ArrayD<f32>,
    stacked_upper: ArrayD<f32>,
    n: usize,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: Option<&[usize]>,
    verify_upper_bound: bool,
    engine: Option<&dyn GemmEngine>,
) -> Result<Vec<bool>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    validate_prescreen_layout(spec_matrix, thresholds, clause_sizes)?;

    let stacked_input = BoundedTensor::new(stacked_lower, stacked_upper)?;
    let sanitized = run_ibp_forward(graph, &stacked_input, engine)?;

    evaluate_batched_output(
        &sanitized,
        n,
        spec_matrix,
        thresholds,
        clause_sizes,
        verify_upper_bound,
    )
}

/// Build a stacked `(N, ...shape)` BoundedTensor from flat 1D lower/upper rows.
///
/// Concatenates all flat data into two contiguous buffers and reshapes once,
/// avoiding N intermediate allocations.
fn stack_flat_rows(
    flat_lowers: &[ArrayD<f32>],
    flat_uppers: &[ArrayD<f32>],
    child_shape: &[usize],
) -> Result<BoundedTensor> {
    let n = flat_lowers.len();
    let flat_dim: usize = child_shape.iter().product();

    let mut lower_data = Vec::with_capacity(n * flat_dim);
    let mut upper_data = Vec::with_capacity(n * flat_dim);
    for i in 0..n {
        if flat_lowers[i].len() != flat_dim {
            return Err(NyError::InvalidSpec(format!(
                "stack_flat_rows: lower[{}] len {} != flat_dim {}",
                i,
                flat_lowers[i].len(),
                flat_dim
            )));
        }
        if flat_uppers[i].len() != flat_dim {
            return Err(NyError::InvalidSpec(format!(
                "stack_flat_rows: upper[{}] len {} != flat_dim {}",
                i,
                flat_uppers[i].len(),
                flat_dim
            )));
        }
        lower_data.extend(flat_lowers[i].iter());
        upper_data.extend(flat_uppers[i].iter());
    }

    let mut stacked_shape = Vec::with_capacity(1 + child_shape.len());
    stacked_shape.push(n);
    stacked_shape.extend_from_slice(child_shape);

    let lower = ArrayD::from_shape_vec(IxDyn(&stacked_shape), lower_data).map_err(|e| {
        NyError::InvalidSpec(format!("stack_flat_rows: reshape stacked lower: {}", e))
    })?;
    let upper = ArrayD::from_shape_vec(IxDyn(&stacked_shape), upper_data).map_err(|e| {
        NyError::InvalidSpec(format!("stack_flat_rows: reshape stacked upper: {}", e))
    })?;

    BoundedTensor::new(lower, upper)
}
