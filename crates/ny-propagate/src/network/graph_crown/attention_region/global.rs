// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Checked original full-column corridor feeding a decomposed attention region.
//! The property axis is never a tensor batch axis. Original tensor boxes and
//! typed layer parameters authenticate each column transform before allocation.

use super::*;
use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::{certified_affine_sum_f32_with_poll, OutwardDirection};
use crate::layers::linear::bias::{
    f64_to_f32_down_no_subnormal, f64_to_f32_up_no_subnormal, publish_error_up_normal,
};
use crate::layers::{AddConstantLayer, LinearLayer, ReduceMeanLayer, TransposeLayer};
use ndarray::Array2;
use ny_core::dd::{next_down_f64, next_up_f64};
use ny_core::{checked_shape_product, f32_to_f64_exact};

mod concat;

#[derive(Clone, Copy)]
enum Transform<'a> {
    Mean {
        axis: usize,
        width: usize,
        inner: usize,
    },
    Linear {
        layer: &'a LinearLayer,
        positions: usize,
    },
    AddConstant {
        layer: &'a AddConstantLayer,
        indexing: BroadcastIndex,
    },
    Reshape,
    Transpose(TransposeIndex),
}

/// Exact original output-column -> producer-column map. Tensor axes never
/// become property axes, and no full-sized index or permutation matrix exists.
#[derive(Clone, Copy)]
struct TransposeIndex {
    rank: usize,
    output: [usize; 8],
    source_axis: [usize; 8],
    source_strides: [usize; 8],
}

impl TransposeIndex {
    fn new(layer: &TransposeLayer, pre: &[usize], output: &[usize]) -> Option<Self> {
        if pre.is_empty()
            || pre.len() > 8
            || output.len() != pre.len()
            || pre.contains(&0)
            || layer.axes.len() > 9
            || checked_shape_product(pre) != checked_shape_product(output)
        {
            return None;
        }
        // Reuse the original typed layer's normalization of an authored batch
        // axis that the loader removed, including ONNX [0,2,1,3] at rank3.
        let perm = layer.resolve_perm(pre.len()).ok()?;
        if perm.len() != pre.len() {
            return None;
        }
        let mut result = Self {
            rank: pre.len(),
            output: [1; 8],
            source_axis: [0; 8],
            source_strides: [0; 8],
        };
        let mut seen = [false; 8];
        for (axis, &source) in perm.iter().enumerate() {
            if source >= pre.len() || seen[source] || output[axis] != pre[source] {
                return None;
            }
            seen[source] = true;
            result.output[axis] = output[axis];
            result.source_axis[axis] = source;
        }
        let mut stride = 1usize;
        for axis in (0..pre.len()).rev() {
            result.source_strides[axis] = stride;
            stride = stride.checked_mul(pre[axis])?;
        }
        Some(result)
    }

    fn source_column(&self, mut column: usize) -> usize {
        let mut source = 0;
        for axis in (0..self.rank).rev() {
            let coordinate = column % self.output[axis];
            column /= self.output[axis];
            source += coordinate * self.source_strides[self.source_axis[axis]];
        }
        source
    }
}

/// Fixed-size original broadcast map; no expanded constant/index matrix.
#[derive(Clone, Copy)]
struct BroadcastIndex {
    rank: usize,
    output: [usize; 8],
    constant: [usize; 8],
    strides: [usize; 8],
}

impl BroadcastIndex {
    fn new(layer: &AddConstantLayer, pre: &[usize], output: &[usize]) -> Result<Option<Self>> {
        if pre.len() > 8
            || output.len() > 8
            || layer.constant().ndim() > 8
            || layer.constant().is_empty()
        {
            return Ok(None);
        }
        // This is the SAME original convention as forward IBP, including its
        // refusal of ambiguous rank1 CNN-vs-right-aligned channel biases.
        let effective = match layer.effective_broadcast_shape(pre) {
            Ok(shape) => shape,
            Err(NyError::UnsupportedOp(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        let Some(broadcast) = crate::shape::broadcast_shapes(pre, &effective) else {
            return Ok(None);
        };
        if broadcast != output
            || output.contains(&0)
            || checked_shape_product(output) != checked_shape_product(pre)
            || effective.len() > output.len()
            || checked_shape_product(&effective) != Some(layer.constant().len())
            || !layer.constant().is_standard_layout()
        {
            return Ok(None);
        }
        let mut result = Self {
            rank: output.len(),
            output: [1; 8],
            constant: [1; 8],
            strides: [0; 8],
        };
        result.output[..output.len()].copy_from_slice(output);
        let pad = output.len() - effective.len();
        result.constant[pad..output.len()].copy_from_slice(&effective);
        let mut stride = 1usize;
        for axis in (0..output.len()).rev() {
            result.strides[axis] = stride;
            stride = stride
                .checked_mul(result.constant[axis])
                .ok_or_else(declined)?;
        }
        Ok(Some(result))
    }

    fn index(&self, mut column: usize) -> usize {
        let mut result = 0;
        for axis in (0..self.rank).rev() {
            let coordinate = column % self.output[axis];
            column /= self.output[axis];
            if self.constant[axis] != 1 {
                result += coordinate * self.strides[axis];
            }
        }
        result
    }
}

fn declined() -> NyError {
    NyError::UnsupportedOp("attention original global-column corridor".into())
}

fn full_columns(bounds: &BatchedLinearBounds, shape: &[usize]) -> bool {
    let a = bounds.lower_a.shape();
    a.len() >= 2
        && a.len() <= 10
        && shape.len() <= 8
        && checked_shape_product(&a[..a.len() - 2]) == Some(1)
        && a[a.len() - 2] > 0
        && checked_shape_product(shape) == Some(a[a.len() - 1])
        && bounds.input_shape() == shape
        && bounds.lower_b.len() == a[a.len() - 2]
        && checked_shape_product(bounds.output_shape()) == Some(a[a.len() - 2])
}

fn mean_profile(
    layer: &ReduceMeanLayer,
    pre: &[usize],
    output: &[usize],
) -> Option<Transform<'static>> {
    if pre.is_empty() || pre.len() > 8 || layer.axes.len() != 1 {
        return None;
    }
    let axis = layer.axes[0];
    let axis = if axis < 0 {
        axis.checked_add(pre.len() as i64)?
    } else {
        axis
    };
    let axis = usize::try_from(axis).ok()?;
    let width = *pre.get(axis)?;
    if width == 0 || width > (1usize << 24) {
        return None;
    }
    let mut expected = pre.to_vec();
    if layer.keepdims {
        expected[axis] = 1;
    } else {
        expected.remove(axis);
    }
    if expected != output {
        return None;
    }
    let inner = checked_shape_product(&pre[axis + 1..])?;
    Some(Transform::Mean { axis, width, inner })
}

/// All optional preparation is private. Only a fully validated original
/// contribution is published; a declined candidate preserves prior paths.
pub(super) fn propagate(
    query: &mut AttentionRegionQuery,
    node: &GraphNode,
    incoming: &BatchedLinearBounds,
    input: &BoundedTensor,
    boxes: &HashMap<String, BoundedTensor>,
    accumulator: &mut BatchedCrownAccumulator,
) -> Result<bool> {
    if matches!(node.layer, Layer::Concat(_)) {
        return concat::propagate(query, node, incoming, input, boxes, accumulator);
    }
    if !matches!(
        node.layer,
        Layer::ReduceMean(_)
            | Layer::Linear(_)
            | Layer::Reshape(_)
            | Layer::Transpose(_)
            | Layer::AddConstant(_)
    ) {
        return Ok(false);
    }
    let source = node.require_unary_input()?;
    let pre = if source == NETWORK_INPUT {
        input
    } else {
        boxes.get(source).ok_or_else(declined)?
    };
    let output = boxes.get(node.name()).ok_or_else(declined)?;
    incoming.validate_internal_shapes()?;
    if !full_columns(incoming, output.shape()) || pre.shape().len() > 8 || pre.is_empty() {
        tracing::debug!(node=node.name(), coefficients=?incoming.lower_a.shape(), incoming_input=?incoming.input_shape(), incoming_output=?incoming.output_shape(), source=?pre.shape(), output=?output.shape(), "original global corridor geometry declined");
        return Ok(false);
    }
    let transform = match &node.layer {
        Layer::ReduceMean(layer) => {
            let Some(profile) = mean_profile(layer, pre.shape(), output.shape()) else {
                return Ok(false);
            };
            profile
        }
        Layer::Linear(layer) => {
            let i = layer.in_features();
            let o = layer.out_features();
            let ps = pre.shape();
            let os = output.shape();
            if ps.is_empty()
                || ps.len() != os.len()
                || i == 0
                || o == 0
                || ps[ps.len() - 1] != i
                || os[os.len() - 1] != o
                || ps[..ps.len() - 1] != os[..os.len() - 1]
                || layer.weight().shape() != [o, i]
                || layer.bias().is_some_and(|b| b.len() != o)
            {
                return Ok(false);
            }
            let positions = pre.len() / i;
            // The ordinary GEMM path remains unchanged for genuine local rows.
            if positions == 1 {
                return Ok(false);
            }
            Transform::Linear { layer, positions }
        }
        Layer::Reshape(layer) => {
            // The legacy resolver multiplies known target extents. Authenticate
            // their bounded product first, including authored copy-axis forms.
            let mut known = 1usize;
            for (index, &extent) in layer.target_shape.iter().enumerate().take(9) {
                let factor = if extent == -1 {
                    1
                } else if extent == 0 {
                    *pre.shape().get(index).ok_or_else(declined)?
                } else if let Some(axis) = ny_core::reshape_copy_axis_from_sentinel(extent) {
                    *pre.shape().get(axis).ok_or_else(declined)?
                } else {
                    match usize::try_from(extent) {
                        Ok(extent) => extent,
                        Err(_) => return Ok(false),
                    }
                };
                let Some(product) = known.checked_mul(factor) else {
                    return Ok(false);
                };
                known = product;
            }
            if layer.target_shape.len() > 8
                || layer.compute_output_shape(pre.shape())? != output.shape()
                || pre.len() != output.len()
            {
                return Ok(false);
            }
            Transform::Reshape
        }
        Layer::Transpose(layer) => {
            let Some(indexing) = TransposeIndex::new(layer, pre.shape(), output.shape()) else {
                return Ok(false);
            };
            Transform::Transpose(indexing)
        }
        Layer::AddConstant(layer) => {
            // Genuine last-axis/local rows retain the established dispatcher.
            // This adapter closes only the complete multi-position column gap.
            if pre.shape().len() < 2 || pre.shape().last().copied() == Some(pre.len()) {
                return Ok(false);
            }
            let Some(indexing) = BroadcastIndex::new(layer, pre.shape(), output.shape())? else {
                tracing::debug!(node=node.name(), constant=?layer.constant().shape(), source=?pre.shape(), output=?output.shape(), "original global corridor broadcast declined");
                return Ok(false);
            };
            Transform::AddConstant { layer, indexing }
        }
        _ => return Ok(false),
    };
    if query.global_steps >= query.global_step_limit || !query.has_held_slot() {
        tracing::debug!(
            node = node.name(),
            completed = query.global_steps,
            limit = query.global_step_limit,
            "original global corridor step limit declined"
        );
        return Ok(false);
    }
    let Some(budget) = query.budget.clone() else {
        tracing::debug!(
            node = node.name(),
            "original global corridor unavailable budget"
        );
        return Ok(false);
    };
    let rows = incoming.lower_b.len();
    let cells = rows.checked_mul(pre.len()).ok_or_else(declined)?;
    let macs = match transform {
        Transform::Linear { layer, positions } => rows
            .checked_mul(positions)
            .and_then(|n| n.checked_mul(layer.in_features()))
            .and_then(|n| n.checked_mul(layer.out_features()))
            .ok_or_else(declined)?,
        // Price the bounded logical index arithmetic for every plane, in
        // addition to the same copy/validation terms used by the old corridor.
        Transform::Transpose(indexing) => {
            cells.checked_mul(indexing.rank + 1).ok_or_else(declined)?
        }
        _ => cells,
    };
    let weights = match transform {
        Transform::Linear { layer, .. } => layer.weight().len(),
        Transform::AddConstant { layer, .. } => layer.constant().len(),
        _ => 0,
    };
    // Capacity slack for four result planes, copied original carrier, bias and
    // bounded shape metadata. There is no packed GEMM/TLS or dense block matrix.
    let persistent = cells
        .checked_mul(32)
        .and_then(|n| n.checked_add(rows.saturating_mul(32)))
        .and_then(|n| n.checked_add(1024))
        .ok_or_else(declined)?;
    let scratch = incoming
        .memory_bytes()
        .checked_mul(2)
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(declined)?;
    let work = macs
        .checked_mul(40)
        .and_then(|n| n.checked_add(cells.saturating_mul(64)))
        .and_then(|n| n.checked_add(incoming.lower_a.len().saturating_mul(64)))
        .and_then(|n| n.checked_add(weights.saturating_mul(4)))
        .and_then(|n| n.checked_add(4096))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(declined)?;
    let request = WorkRequest {
        persistent_bytes: persistent,
        scratch_bytes: scratch,
        work,
        ..WorkRequest::default()
    };
    if let Err(error) = query.admit_peak(accumulator.total_memory_bytes(), request, 0) {
        tracing::debug!(node=node.name(), %error, ?request, "original global corridor frontier admission declined");
        return Ok(false);
    }
    let mut lease = match budget.lease(request) {
        Ok(lease) => lease,
        Err(error) => {
            tracing::debug!(node=node.name(), %error, ?request, "original global corridor lease declined");
            return Ok(false);
        }
    };
    let candidate = (|| {
        poll(query.deadline, &budget)?;
        ny_core::require_f64_interval_proof_environment()?;
        let flat = consume::flat_incoming(incoming, output.len(), query.deadline, &budget)?;
        let result = match transform {
            Transform::Mean { axis, width, inner } => {
                debug_assert!(axis < pre.shape().len());
                mean(&flat, pre.len(), width, inner, query.deadline, &budget)?
            }
            Transform::Linear { layer, positions } => {
                linear(&flat, layer, positions, query.deadline, &budget)?
            }
            Transform::AddConstant { layer, indexing } => {
                add_constant(flat, layer, indexing, query.deadline, &budget)?
            }
            Transform::Reshape => flat,
            Transform::Transpose(indexing) => transpose(flat, indexing, query.deadline, &budget)?,
        };
        global_source(result, pre.shape(), query.deadline, &budget)
    })();
    let candidate = match candidate {
        Ok(candidate) => candidate,
        Err(error) => {
            tracing::debug!(node=node.name(), %error, "original global corridor declined");
            return Ok(false);
        }
    };
    poll(query.deadline, &budget)?;
    lease.complete().map_err(refusal)?;
    // Held until query drop, including accumulation overlap. The bounded header
    // was reserved before allocating this vector; no unpriced push can grow it.
    query.hold_completed(lease)?;
    query.global_steps += 1;
    accumulator.accumulate(source, BatchedCrownBounds::Dense(candidate))?;
    tracing::debug!(
        node = node.name(),
        completed = query.global_steps,
        "original global corridor propagated"
    );
    Ok(true)
}

/// The authenticated global property axis has NO tensor-batch prefix. Keep it
/// two-dimensional through later consumers/final concretization; a leading
/// singleton is not permission to index only the original tensor's last axis.
/// Numeric storage moves unchanged. Final graph tightening restores the
/// original output tensor shape after computing every complete property row.
fn global_source(
    bounds: LinearBounds,
    source: &[usize],
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<BatchedLinearBounds> {
    consume::validate_linear(&bounds, deadline, budget)?;
    if source.len() > 8 || checked_shape_product(source) != Some(bounds.num_inputs()) {
        return Err(declined());
    }
    let rows = bounds.num_outputs();
    let result = BatchedLinearBounds {
        lower_a: bounds.lower_a.into_dyn(),
        lower_b: bounds.lower_b.into_dyn(),
        upper_a: bounds.upper_a.into_dyn(),
        upper_b: bounds.upper_b.into_dyn(),
        lower_a_err: bounds.lower_a_err.map(|a| a.into_dyn()),
        upper_a_err: bounds.upper_a_err.map(|a| a.into_dyn()),
        input_shape: source.to_vec(),
        output_shape: vec![rows],
    };
    result.validate_internal_shapes()?;
    poll(deadline, budget)?;
    Ok(result)
}

fn transpose(
    incoming: LinearBounds,
    indexing: TransposeIndex,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<LinearBounds> {
    consume::validate_linear(&incoming, deadline, budget)?;
    let rows = incoming.num_outputs();
    let columns = incoming.num_inputs();
    if checked_shape_product(&indexing.output[..indexing.rank]) != Some(columns) {
        return Err(declined());
    }
    let permute = |original: Array2<f32>| -> Result<Array2<f32>> {
        let source = original.as_slice().ok_or_else(declined)?;
        let mut result = consume::zeroes(source.len(), deadline, budget)?;
        for (cell, &value) in source.iter().enumerate() {
            if cell % 1024 == 0 {
                poll(deadline, budget)?;
            }
            let row = cell / columns;
            let column = cell % columns;
            result[row * columns + indexing.source_column(column)] = value;
        }
        Array2::from_shape_vec((rows, columns), result).map_err(|_| declined())
    };
    let result = LinearBounds::from_prevalidated_parts_with_optional_err(
        permute(incoming.lower_a)?,
        incoming.lower_b,
        permute(incoming.upper_a)?,
        incoming.upper_b,
        incoming.lower_a_err.map(permute).transpose()?,
        incoming.upper_a_err.map(permute).transpose()?,
    )?;
    // The coefficient and error bits only change positions; biases move once.
    // Revalidate on the original lease before publishing producer metadata.
    consume::validate_linear(&result, deadline, budget)?;
    poll(deadline, budget)?;
    Ok(result)
}

fn up_add(a: f64, b: f64) -> f64 {
    if a == 0.0 && b == 0.0 {
        0.0
    } else {
        next_up_f64(a + b)
    }
}

fn coefficient(lo: f64, hi: f64, carried: f64) -> Result<(f32, f32)> {
    if !lo.is_finite() || !hi.is_finite() || lo > hi || carried.is_nan() || carried < 0.0 {
        return Err(declined());
    }
    let nearest = (lo * 0.5 + hi * 0.5) as f32;
    // New carriers must not depend on downstream binary32 gradual underflow.
    // Pay the full original interval gap when suppressing a subnormal center.
    let nominal = if nearest.to_bits() & 0x7fff_ffff < 0x0080_0000 {
        0.0
    } else {
        nearest
    };
    if !nominal.is_finite() {
        return Err(declined());
    }
    let stored = f32_to_f64_exact(nominal);
    let gap = if lo == hi && stored == lo {
        0.0
    } else {
        next_up_f64((stored - lo).abs().max((stored - hi).abs()))
    };
    Ok((nominal, publish_error_up_normal(up_add(gap, carried))))
}

fn finish(
    rows: usize,
    columns: usize,
    la: Vec<f32>,
    ua: Vec<f32>,
    lb: Vec<f32>,
    ub: Vec<f32>,
    le: Vec<f32>,
    ue: Vec<f32>,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<LinearBounds> {
    let matrix = |values| Array2::from_shape_vec((rows, columns), values).map_err(|_| declined());
    let result = LinearBounds::from_prevalidated_parts_with_optional_err(
        matrix(la)?,
        Array1::from_vec(lb),
        matrix(ua)?,
        Array1::from_vec(ub),
        Some(matrix(le)?),
        Some(matrix(ue)?),
    )?;
    consume::validate_linear(&result, deadline, budget)?;
    Ok(result)
}

fn mean(
    incoming: &LinearBounds,
    columns: usize,
    width: usize,
    inner: usize,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<LinearBounds> {
    let rows = incoming.num_outputs();
    let mut la = consume::zeroes(rows * columns, deadline, budget)?;
    let mut ua = consume::zeroes(rows * columns, deadline, budget)?;
    let mut le = consume::zeroes(rows * columns, deadline, budget)?;
    let mut ue = consume::zeroes(rows * columns, deadline, budget)?;
    let divisor = width as f64; // width <= 2^24: exact positive integer.
    for row in 0..rows {
        for column in 0..columns {
            if column % 256 == 0 {
                poll(deadline, budget)?;
            }
            let original = (column / (width * inner)) * inner + column % inner;
            for (a, error, nominal, stored_error) in [
                (
                    &incoming.lower_a,
                    incoming.lower_a_err.as_ref(),
                    &mut la,
                    &mut le,
                ),
                (
                    &incoming.upper_a,
                    incoming.upper_a_err.as_ref(),
                    &mut ua,
                    &mut ue,
                ),
            ] {
                let value = f32_to_f64_exact(a[[row, original]]);
                let quotient = value / divisor;
                let (lo, hi) = if value == 0.0 {
                    (0.0, 0.0)
                } else {
                    (next_down_f64(quotient), next_up_f64(quotient))
                };
                let e = error.map_or(0.0, |e| f32_to_f64_exact(e[[row, original]]));
                let carried = if e == 0.0 {
                    0.0
                } else {
                    next_up_f64(e / divisor)
                };
                let (n, e) = coefficient(lo, hi, carried)?;
                nominal[row * columns + column] = n;
                stored_error[row * columns + column] = e;
            }
        }
    }
    finish(
        rows,
        columns,
        la,
        ua,
        consume::copied(
            incoming.lower_b.as_slice().ok_or_else(declined)?,
            deadline,
            budget,
        )?,
        consume::copied(
            incoming.upper_b.as_slice().ok_or_else(declined)?,
            deadline,
            budget,
        )?,
        le,
        ue,
        deadline,
        budget,
    )
}

fn linear(
    incoming: &LinearBounds,
    layer: &LinearLayer,
    positions: usize,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<LinearBounds> {
    let rows = incoming.num_outputs();
    let i = layer.in_features();
    let o = layer.out_features();
    let columns = positions * i;
    for chunk in layer.weight().as_slice().ok_or_else(declined)?.chunks(1024) {
        poll(deadline, budget)?;
        if chunk.iter().any(|v| !v.is_finite()) {
            return Err(declined());
        }
    }
    if let Some(b) = layer.bias() {
        for chunk in b.as_slice().ok_or_else(declined)?.chunks(1024) {
            poll(deadline, budget)?;
            if chunk.iter().any(|v| !v.is_finite()) {
                return Err(declined());
            }
        }
    }
    let mut la = consume::zeroes(rows * columns, deadline, budget)?;
    let mut ua = consume::zeroes(rows * columns, deadline, budget)?;
    let mut le = consume::zeroes(rows * columns, deadline, budget)?;
    let mut ue = consume::zeroes(rows * columns, deadline, budget)?;
    let mut lb = consume::zeroes(rows, deadline, budget)?;
    let mut ub = consume::zeroes(rows, deadline, budget)?;
    for row in 0..rows {
        for position in 0..positions {
            for channel in 0..i {
                poll(deadline, budget)?;
                for (a, error, nominal, stored_error) in [
                    (
                        &incoming.lower_a,
                        incoming.lower_a_err.as_ref(),
                        &mut la,
                        &mut le,
                    ),
                    (
                        &incoming.upper_a,
                        incoming.upper_a_err.as_ref(),
                        &mut ua,
                        &mut ue,
                    ),
                ] {
                    let terms = || {
                        (0..o).map(|k| (a[[row, position * o + k]], layer.weight()[[k, channel]]))
                    };
                    let lo = certified_affine_sum_f32_with_poll(
                        0.0,
                        terms(),
                        OutwardDirection::Lower,
                        |_| poll(deadline, budget),
                    )?;
                    let hi = certified_affine_sum_f32_with_poll(
                        0.0,
                        terms(),
                        OutwardDirection::Upper,
                        |_| poll(deadline, budget),
                    )?;
                    let carry = if let Some(error) = error {
                        certified_affine_sum_f32_with_poll(
                            0.0,
                            (0..o).map(|k| {
                                (
                                    error[[row, position * o + k]],
                                    layer.weight()[[k, channel]].abs(),
                                )
                            }),
                            OutwardDirection::Upper,
                            |_| poll(deadline, budget),
                        )?
                    } else {
                        0.0
                    };
                    let (n, e) = coefficient(lo, hi, carry)?;
                    nominal[row * columns + position * i + channel] = n;
                    stored_error[row * columns + position * i + channel] = e;
                }
            }
        }
        for (a, error, original_bias, stored, direction) in [
            (
                &incoming.lower_a,
                incoming.lower_a_err.as_ref(),
                incoming.lower_b[row],
                &mut lb,
                OutwardDirection::Lower,
            ),
            (
                &incoming.upper_a,
                incoming.upper_a_err.as_ref(),
                incoming.upper_b[row],
                &mut ub,
                OutwardDirection::Upper,
            ),
        ] {
            let bias = |column: usize| layer.bias().map_or(0.0, |b| b[column % o]);
            let sum = certified_affine_sum_f32_with_poll(
                original_bias,
                (0..positions * o).map(|k| (a[[row, k]], bias(k))),
                direction,
                |_| poll(deadline, budget),
            )?;
            let carry = if let Some(error) = error {
                certified_affine_sum_f32_with_poll(
                    0.0,
                    (0..positions * o).map(|k| (error[[row, k]], bias(k).abs())),
                    OutwardDirection::Upper,
                    |_| poll(deadline, budget),
                )?
            } else {
                0.0
            };
            stored[row] = match direction {
                OutwardDirection::Lower => f64_to_f32_down_no_subnormal(if carry == 0.0 {
                    sum
                } else {
                    next_down_f64(sum - carry)
                }),
                OutwardDirection::Upper => f64_to_f32_up_no_subnormal(if carry == 0.0 {
                    sum
                } else {
                    next_up_f64(sum + carry)
                }),
            };
        }
    }
    finish(rows, columns, la, ua, lb, ub, le, ue, deadline, budget)
}

fn add_constant(
    mut incoming: LinearBounds,
    layer: &AddConstantLayer,
    indexing: BroadcastIndex,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<LinearBounds> {
    let constant = layer.constant().as_slice().ok_or_else(declined)?;
    for chunk in constant.chunks(1024) {
        poll(deadline, budget)?;
        if chunk.iter().any(|v| !v.is_finite()) {
            return Err(declined());
        }
    }
    let columns = incoming.num_inputs();
    for row in 0..incoming.num_outputs() {
        poll(deadline, budget)?;
        for (a, error, bias, direction) in [
            (
                &incoming.lower_a,
                incoming.lower_a_err.as_ref(),
                &mut incoming.lower_b,
                OutwardDirection::Lower,
            ),
            (
                &incoming.upper_a,
                incoming.upper_a_err.as_ref(),
                &mut incoming.upper_b,
                OutwardDirection::Upper,
            ),
        ] {
            let sum = certified_affine_sum_f32_with_poll(
                bias[row],
                (0..columns).map(|column| (a[[row, column]], constant[indexing.index(column)])),
                direction,
                |_| poll(deadline, budget),
            )?;
            // |error*(x+c)| <= error*|x| + error*|c|. Keep the
            // ORIGINAL coefficient-error planes, discharging only the extra
            // constant term once into this row's directed shared bias.
            let carry = if let Some(error) = error {
                certified_affine_sum_f32_with_poll(
                    0.0,
                    (0..columns).map(|column| {
                        (error[[row, column]], constant[indexing.index(column)].abs())
                    }),
                    OutwardDirection::Upper,
                    |_| poll(deadline, budget),
                )?
            } else {
                0.0
            };
            bias[row] = match direction {
                OutwardDirection::Lower => f64_to_f32_down_no_subnormal(if carry == 0.0 {
                    sum
                } else {
                    next_down_f64(sum - carry)
                }),
                OutwardDirection::Upper => f64_to_f32_up_no_subnormal(if carry == 0.0 {
                    sum
                } else {
                    next_up_f64(sum + carry)
                }),
            };
        }
    }
    consume::validate_linear(&incoming, deadline, budget)?;
    Ok(incoming)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod reshape_tests;

#[cfg(test)]
mod bookkeeping_tests;
