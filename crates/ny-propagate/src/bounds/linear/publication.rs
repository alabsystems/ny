// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Final numeric firewall over private, already owned output buffers.
//! No payload is allocated, copied or detached from its caller's claim here.
//! This does not account for source captures, allocations, or whole kernels.

use super::{bias_nan, coefficient_non_finite, sanitize_coefficient_error, LinearBounds};
use crate::amortise::{CooperativeWork, POLL_CHUNK_UNITS};
use ndarray::{Array, Array1, Array2, Dimension};
use ny_core::Result;

// Conservative fixed cost for one value predicate, assignment or sanitization.
// Every batch is admitted before its first value is read or written. Charging
// a whole scan batch can include its unvisited suffix after a short circuit.
const VALUE_UNITS: usize = 8;
const VALUES_PER_BATCH: usize = POLL_CHUNK_UNITS / VALUE_UNITS;

fn any_with_work<D: Dimension, P: CooperativeWork>(
    array: &Array<f32, D>,
    predicate: fn(&f32) -> bool,
    work: &mut P,
) -> Result<bool> {
    if let Some(values) = array.as_slice() {
        for batch in values.chunks(VALUES_PER_BATCH) {
            work.before(batch.len() * VALUE_UNITS)?;
            if batch.iter().any(predicate) {
                return Ok(true);
            }
        }
    } else {
        // A view's stride/order is never repaired or materialized. Iteration
        // follows the same logical order as the original ndarray validator.
        for value in array {
            work.before(VALUE_UNITS)?;
            if predicate(value) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn map_with_work<D: Dimension, P: CooperativeWork>(
    array: &mut Array<f32, D>,
    operation: fn(f32) -> f32,
    work: &mut P,
) -> Result<()> {
    if let Some(values) = array.as_slice_mut() {
        for batch in values.chunks_mut(VALUES_PER_BATCH) {
            work.before(batch.len() * VALUE_UNITS)?;
            for value in batch {
                *value = operation(*value);
            }
        }
    } else {
        for value in array {
            work.before(VALUE_UNITS)?;
            *value = operation(*value);
        }
    }
    Ok(())
}

impl LinearBounds {
    /// Apply the existing `new_or_conservative_with_err` numeric policy in
    /// place under the caller's original cooperative authority.
    ///
    /// This consumes PRIVATE, unpublished arrays. Any refusal drops them and
    /// returns no partial bound. The caller must retain their actual-capacity
    /// claims through this call and the returned bound's lifetime. No memory
    /// allowance, reservation or allocation provenance is created here.
    ///
    /// Base shape errors remain hard errors. A non-finite coefficient or NaN
    /// bias degrades every row; malformed error-plane shapes drop both errors
    /// and degrade every row. Otherwise both error planes remain present and
    /// use the exact original sanitizer. Valid ±Inf biases are preserved.
    /// All surviving arrays keep their owners, offsets, strides and capacities,
    /// including when the original allocating constructor would build a fresh
    /// conservative fallback or copy an error plane through `mapv`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_or_conservative_with_err_polled_in_place<P: CooperativeWork>(
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
        mut lower_a_err: Array2<f32>,
        mut upper_a_err: Array2<f32>,
        work: &mut P,
    ) -> Result<Self> {
        work.checkpoint()?;
        work.before(64)?;
        let mut base = Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: None,
            upper_a_err: None,
        };
        base.validate_internal_shapes()?;
        work.checkpoint()?;

        let invalid_values = any_with_work(&base.lower_a, coefficient_non_finite, work)?
            || any_with_work(&base.upper_a, coefficient_non_finite, work)?
            || any_with_work(&base.lower_b, bias_nan, work)?
            || any_with_work(&base.upper_b, bias_nan, work)?;
        let errors_match = lower_a_err.shape() == base.lower_a.shape()
            && upper_a_err.shape() == base.upper_a.shape();
        work.checkpoint()?;

        if invalid_values || !errors_match {
            // Same conservative object values as the original constructor.
            // Reuse only this private output's original owners; no replacement
            // arrays or second, uncharged output commitment is allocated.
            map_with_work(&mut base.lower_a, |_| 0.0, work)?;
            map_with_work(&mut base.lower_b, |_| f32::NEG_INFINITY, work)?;
            map_with_work(&mut base.upper_a, |_| 0.0, work)?;
            map_with_work(&mut base.upper_b, |_| f32::INFINITY, work)?;
        }
        if errors_match {
            map_with_work(&mut lower_a_err, sanitize_coefficient_error, work)?;
            map_with_work(&mut upper_a_err, sanitize_coefficient_error, work)?;
            base.lower_a_err = Some(lower_a_err);
            base.upper_a_err = Some(upper_a_err);
        }
        work.checkpoint()?;
        Ok(base)
    }
}

#[cfg(test)]
#[path = "publication_tests.rs"]
mod tests;
