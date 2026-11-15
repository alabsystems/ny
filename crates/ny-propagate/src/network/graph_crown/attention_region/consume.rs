// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{Array2, ArrayD, IxDyn};

fn allocation() -> NyError {
    NyError::UnsupportedOp("attention region allocation admission".into())
}

#[derive(Clone, Copy)]
enum NumericKind {
    Finite,
    Bias,
    Error,
}

// Every caller holds the existing conversion/region lease before these scans.
// Their work predictions include copying and validating all four carrier planes;
// this helper does not create a new allowance or sanitize any original value.
fn validate_values(
    values: &[f32],
    kind: NumericKind,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<()> {
    poll(deadline, budget)?;
    for chunk in values.chunks(1024) {
        poll(deadline, budget)?;
        for &value in chunk {
            let bits = value.to_bits();
            let magnitude = bits & 0x7fff_ffff;
            let nan = magnitude > 0x7f80_0000;
            let valid = match kind {
                NumericKind::Finite => magnitude < 0x7f80_0000,
                NumericKind::Bias => !nan,
                // Inspect the sign bits so DAZ cannot erase a negative subnormal
                // error. Signed zero remains nonnegative, as in the old profile.
                NumericKind::Error => !nan && (bits >> 31 == 0 || magnitude == 0),
            };
            if !valid {
                return Err(NyError::NumericalInstability(
                    "attention region carrier numeric validation".into(),
                ));
            }
        }
        validation_checkpoint(chunk.len(), deadline, budget)?;
    }
    poll(deadline, budget)
}

pub(super) fn validate_linear(
    bounds: &LinearBounds,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<()> {
    bounds.validate_internal_shapes()?;
    validate_parts(
        &bounds.lower_a,
        &bounds.lower_b,
        &bounds.upper_a,
        &bounds.upper_b,
        bounds.lower_a_err.as_ref(),
        bounds.upper_a_err.as_ref(),
        deadline,
        budget,
    )
}

pub(super) fn validate_biases(
    lower: &Array1<f32>,
    upper: &Array1<f32>,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<()> {
    for array in [lower, upper] {
        validate_values(
            array.as_slice().ok_or_else(allocation)?,
            NumericKind::Bias,
            deadline,
            budget,
        )?;
    }
    poll(deadline, budget)
}

#[allow(clippy::too_many_arguments)]
fn validate_parts(
    lower_a: &Array2<f32>,
    lower_b: &Array1<f32>,
    upper_a: &Array2<f32>,
    upper_b: &Array1<f32>,
    lower_a_err: Option<&Array2<f32>>,
    upper_a_err: Option<&Array2<f32>>,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<()> {
    for array in [lower_a, upper_a] {
        validate_values(
            array.as_slice().ok_or_else(allocation)?,
            NumericKind::Finite,
            deadline,
            budget,
        )?;
    }
    for array in [lower_b, upper_b] {
        validate_values(
            array.as_slice().ok_or_else(allocation)?,
            NumericKind::Bias,
            deadline,
            budget,
        )?;
    }
    for array in lower_a_err.into_iter().chain(upper_a_err) {
        validate_values(
            array.as_slice().ok_or_else(allocation)?,
            NumericKind::Error,
            deadline,
            budget,
        )?;
    }
    poll(deadline, budget)
}

fn validation_checkpoint(
    _inspected: usize,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<()> {
    #[cfg(test)]
    VALIDATION_STOP.with(|slot| {
        if let Some(stop) = slot.borrow_mut().as_mut() {
            stop.inspected = stop.inspected.saturating_add(_inspected);
            if stop.inspected >= stop.after {
                stop.cancellation.cancel();
            }
        }
    });
    poll(deadline, budget)
}

#[cfg(test)]
struct ValidationStop {
    cancellation: crate::certify::residency::CancellationToken,
    after: usize,
    inspected: usize,
}

#[cfg(test)]
thread_local! {
    static VALIDATION_STOP: RefCell<Option<ValidationStop>> = const { RefCell::new(None) };
}

/// Test seam cancels the ORIGINAL ledger after actual final-validation cells,
/// never during allocation/copy. RAII also restores a nested hook on unwind.
#[cfg(test)]
pub(super) fn cancel_during_validation<T>(
    cancellation: crate::certify::residency::CancellationToken,
    after: usize,
    action: impl FnOnce() -> T,
) -> (T, usize) {
    struct Restore(Option<ValidationStop>);
    impl Drop for Restore {
        fn drop(&mut self) {
            VALIDATION_STOP.with(|slot| {
                slot.replace(self.0.take());
            });
        }
    }
    let _restore = Restore(VALIDATION_STOP.with(|slot| {
        slot.replace(Some(ValidationStop {
            cancellation,
            after,
            inspected: 0,
        }))
    }));
    let result = action();
    let inspected = VALIDATION_STOP.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|stop| stop.inspected)
            .unwrap_or(0)
    });
    (result, inspected)
}

pub(super) fn copied(
    values: &[f32],
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<Vec<f32>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|_| allocation())?;
    if result.capacity() > values.len().saturating_mul(2).max(16) {
        return Err(allocation());
    }
    for chunk in values.chunks(1024) {
        poll(deadline, budget)?;
        result.extend_from_slice(chunk);
    }
    Ok(result)
}

pub(super) fn zeroes(
    len: usize,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<Vec<f32>> {
    let mut result = Vec::new();
    result.try_reserve_exact(len).map_err(|_| allocation())?;
    if result.capacity() > len.saturating_mul(2).max(16) {
        return Err(allocation());
    }
    while result.len() < len {
        poll(deadline, budget)?;
        let count = (len - result.len()).min(1024);
        result.extend(std::iter::repeat_n(0.0, count));
    }
    Ok(result)
}

// Kept only as the old-layout numerical/test control. Production consumes the
// authenticated original Kt storage directly and allocates neither plane.
#[cfg(test)]
pub(super) fn transpose_box(
    original: &BoundedTensor,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<BoundedTensor> {
    let shape = original.shape().to_vec();
    let rank = shape.len();
    let (dk, sk) = (shape[rank - 2], shape[rank - 1]);
    let swap = |array: &ArrayD<f32>| -> Result<ArrayD<f32>> {
        let source = array.as_slice().ok_or_else(allocation)?;
        let mut result = zeroes(source.len(), deadline, budget)?;
        for (index, value) in result.iter_mut().enumerate() {
            if index % 1024 == 0 {
                poll(deadline, budget)?;
            }
            let slice = index / (sk * dk);
            let key = (index / dk) % sk;
            let channel = index % dk;
            *value = source[slice * sk * dk + channel * sk + key];
        }
        let mut target_shape = shape.clone();
        target_shape.swap(rank - 2, rank - 1);
        ArrayD::from_shape_vec(IxDyn(&target_shape), result).map_err(|_| allocation())
    };
    let lower = swap(original.lower())?;
    let upper = swap(original.upper())?;
    // Preserve the old strict finite-box profile. The existing polled tensor
    // factory then authenticates shape, NaNs and endpoint order without an
    // uninterruptible final pass over a potentially large box.
    for array in [&lower, &upper] {
        validate_values(
            array.as_slice().ok_or_else(allocation)?,
            NumericKind::Finite,
            deadline,
            budget,
        )?;
    }
    BoundedTensor::new_allow_infinite_with_poll(lower, upper, || poll(deadline, budget))
}

#[cfg(test)]
pub(super) fn transpose_coefficients(
    bounds: &mut LinearBounds,
    original_shape: &[usize],
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<()> {
    let rank = original_shape.len();
    let (dk, sk) = (original_shape[rank - 2], original_shape[rank - 1]);
    let permute = |array: &Array2<f32>| -> Result<Array2<f32>> {
        let source = array.as_slice().ok_or_else(allocation)?;
        let columns = array.ncols();
        let mut result = zeroes(source.len(), deadline, budget)?;
        for (index, value) in result.iter_mut().enumerate() {
            if index % 1024 == 0 {
                poll(deadline, budget)?;
            }
            let row = index / columns;
            let column = index % columns;
            let slice = column / (sk * dk);
            let channel = (column / sk) % dk;
            let key = column % sk;
            *value = source[row * columns + slice * sk * dk + key * dk + channel];
        }
        Array2::from_shape_vec(array.raw_dim(), result).map_err(|_| allocation())
    };
    bounds.lower_a = permute(&bounds.lower_a)?;
    bounds.upper_a = permute(&bounds.upper_a)?;
    if let Some(error) = &bounds.lower_a_err {
        bounds.lower_a_err = Some(permute(error)?);
    }
    if let Some(error) = &bounds.upper_a_err {
        bounds.upper_a_err = Some(permute(error)?);
    }
    poll(deadline, budget)
}

pub(super) fn flat_incoming(
    bounds: &BatchedLinearBounds,
    columns: usize,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<LinearBounds> {
    bounds.validate_internal_shapes()?;
    let shape = bounds.lower_a.shape();
    // This first profile accepts complete global property directions (possibly
    // with leading singleton batch axes). A per-head/local row is not a global
    // objective: do not tile it or reinterpret heads as property rows.
    if shape.len() < 2
        || shape.len() > 10
        || bounds.output_shape.len() > 8
        || bounds.input_shape.len() > 8
        || shape[shape.len() - 1] != columns
        || ny_core::checked_shape_product(&shape[..shape.len() - 2]) != Some(1)
    {
        return Err(allocation());
    }
    let rows = shape[shape.len() - 2];
    if bounds.lower_b.len() != rows || rows == 0 {
        return Err(allocation());
    }
    let matrix = |a: &ArrayD<f32>| -> Result<Array2<f32>> {
        Array2::from_shape_vec(
            (rows, columns),
            copied(a.as_slice().ok_or_else(allocation)?, deadline, budget)?,
        )
        .map_err(|_| allocation())
    };
    let la = matrix(&bounds.lower_a)?;
    let ua = matrix(&bounds.upper_a)?;
    let lb = Array1::from_vec(copied(
        bounds.lower_b.as_slice().ok_or_else(allocation)?,
        deadline,
        budget,
    )?);
    let ub = Array1::from_vec(copied(
        bounds.upper_b.as_slice().ok_or_else(allocation)?,
        deadline,
        budget,
    )?);
    // Directly retain and validate the exact error cells; no sanitizing
    // factory may make a malformed incoming error look like an exact zero.
    let le = bounds.lower_a_err.as_ref().map(matrix).transpose()?;
    let ue = bounds.upper_a_err.as_ref().map(matrix).transpose()?;
    validate_parts(
        &la,
        &lb,
        &ua,
        &ub,
        le.as_ref(),
        ue.as_ref(),
        deadline,
        budget,
    )?;
    let result = LinearBounds::from_prevalidated_parts_with_optional_err(la, lb, ua, ub, le, ue)?;
    poll(deadline, budget)?;
    Ok(result)
}

pub(super) fn batched_source(
    bounds: LinearBounds,
    source: &[usize],
    incoming: &BatchedLinearBounds,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<BatchedLinearBounds> {
    incoming.validate_internal_shapes()?;
    validate_linear(&bounds, deadline, budget)?;
    let mut a_shape = incoming.lower_a.shape().to_vec();
    let last = a_shape.len() - 1;
    a_shape[last] = bounds.num_inputs();
    let bias_shape = &a_shape[..last];
    let la = bounds
        .lower_a
        .into_dyn()
        .into_shape_with_order(IxDyn(&a_shape))
        .map_err(|_| allocation())?;
    let ua = bounds
        .upper_a
        .into_dyn()
        .into_shape_with_order(IxDyn(&a_shape))
        .map_err(|_| allocation())?;
    let lb = bounds
        .lower_b
        .into_dyn()
        .into_shape_with_order(IxDyn(bias_shape))
        .map_err(|_| allocation())?;
    let ub = bounds
        .upper_b
        .into_dyn()
        .into_shape_with_order(IxDyn(bias_shape))
        .map_err(|_| allocation())?;
    let reshape_error = |error: Array2<f32>| {
        error
            .into_dyn()
            .into_shape_with_order(IxDyn(&a_shape))
            .map_err(|_| allocation())
    };
    // Only shape metadata changes. All numeric parts and optional error cells
    // were authenticated by the polled scan above; do not invoke a constructor
    // that repeats unpolled numeric scans (including in debug builds).
    let result = BatchedLinearBounds {
        lower_a: la,
        lower_b: lb,
        upper_a: ua,
        upper_b: ub,
        input_shape: source.to_vec(),
        output_shape: incoming.output_shape.clone(),
        lower_a_err: bounds.lower_a_err.map(reshape_error).transpose()?,
        upper_a_err: bounds.upper_a_err.map(reshape_error).transpose()?,
    };
    result.validate_internal_shapes()?;
    poll(deadline, budget)?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batched(
    query: &mut AttentionRegionQuery,
    graph: &GraphNetwork,
    node: &GraphNode,
    incoming: &BatchedLinearBounds,
    input: &BoundedTensor,
    boxes: &HashMap<String, BoundedTensor>,
    accumulator: &mut BatchedCrownAccumulator,
    deadline: Option<Instant>,
) -> Result<bool> {
    if !query.begin_attempt(node) {
        return Ok(false);
    }
    let deadline = earlier(deadline, query.deadline);
    let Some(budget) = query.budget.clone() else {
        return Ok(false);
    };
    let attempt = (|| -> Result<_> {
        // Pay the bounded original-node lookup before doing it. This lease also
        // prices a complete incoming carrier copy, including both error arrays.
        let incoming_bytes = incoming.memory_bytes();
        let request = WorkRequest {
            scratch_bytes: incoming_bytes
                .checked_mul(4)
                .and_then(|n| n.checked_add(HEADER_BYTES))
                .ok_or_else(allocation)?,
            work: u64::try_from(incoming_bytes)
                .ok()
                .and_then(|n| n.checked_mul(8))
                .and_then(|n| n.checked_add(RECOGNITION_WORK))
                .ok_or_else(allocation)?,
            ..WorkRequest::default()
        };
        query.admit_peak(accumulator.total_memory_bytes(), request, 0)?;
        let mut conversion = budget.lease(request).map_err(refusal)?;
        poll(deadline, &budget)?;
        let Some(region) = recognize::recognize(graph, node, input, boxes)? else {
            return Ok(None);
        };
        query.receipt.recognized += 1;
        let dense = flat_incoming(incoming, region.output_len, deadline, &budget)?;
        let Some(candidate) = query.compose_region(
            region,
            &dense,
            input.len(),
            accumulator
                .total_memory_bytes()
                .saturating_add(incoming_bytes),
            deadline,
            &budget,
        )?
        else {
            return Ok(None);
        };
        let mut children = Vec::with_capacity(3);
        for (source, bounds) in candidate.sources.into_iter().zip(candidate.bounds) {
            let source_box = if source == NETWORK_INPUT {
                input
            } else {
                boxes.get(source).ok_or_else(allocation)?
            };
            children.push((
                source,
                batched_source(bounds, source_box.shape(), incoming, deadline, &budget)?,
            ));
        }
        let rows = candidate.lower_bias.len();
        let count = rows.checked_mul(input.len()).ok_or_else(allocation)?;
        // The zero arrays are finite by construction. Authenticate the kernel
        // biases before the shape-only constructor; batched_source also polls
        // the final complete carrier validation before publication.
        for array in [&candidate.lower_bias, &candidate.upper_bias] {
            validate_values(
                array.as_slice().ok_or_else(allocation)?,
                NumericKind::Bias,
                deadline,
                &budget,
            )?;
        }
        let bias = LinearBounds::from_prevalidated_parts_with_optional_err(
            Array2::from_shape_vec((rows, input.len()), zeroes(count, deadline, &budget)?)
                .map_err(|_| allocation())?,
            candidate.lower_bias,
            Array2::from_shape_vec((rows, input.len()), zeroes(count, deadline, &budget)?)
                .map_err(|_| allocation())?,
            candidate.upper_bias,
            None,
            None,
        )?;
        let bias = batched_source(bias, input.shape(), incoming, deadline, &budget)?;
        poll(deadline, &budget)?;
        conversion.complete().map_err(refusal)?;
        Ok(Some((children, bias)))
    })();
    let staged = match attempt {
        Ok(Some(staged)) => staged,
        Ok(None) => {
            query.receipt.declined += 1;
            return Ok(false);
        }
        Err(error) => {
            query.receipt.declined += 1;
            tracing::debug!(node = node.name(), %error, "batched attention region declined");
            return Ok(false);
        }
    };
    if poll(deadline, &budget).is_err() {
        return Ok(false);
    }
    // Publication is all-or-abort for the surrounding walk. No candidate error
    // after the first merge may return false and retry the original PV path.
    accumulator.accumulate_dense(NETWORK_INPUT, staged.1)?;
    for (source, bounds) in staged.0 {
        accumulator.accumulate_dense(source, bounds)?;
    }
    query.receipt.batched += 1;
    query.receipt.composed += 1;
    Ok(true)
}
