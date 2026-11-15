// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Direction-first normalization for the mathematical real LayerNorm/RMSNorm.
//!
//! For `z=P x`, `t=sqrt(eps+||z||²/n)`, and `a=P(C gamma)`, the requested
//! scalar is `a·z/t+C beta`. Its gradient is `a/t-(a·z)z/(n t³)`.
//! On the entire expansion tube, a Taylor remainder is at most
//! `3 ||a|| rho²/(sqrt(n) t_min²)`. A separately checked directional Hessian
//! bound can reduce that remainder. Binary64 interval arithmetic checks the
//! center, gradient, remainder, and every stored binary32 coefficient/bias.
//!
//! Default adapters retain the old affine row unless this row dominates it
//! pointwise over the input box, including its coefficient-error envelope.
//! Consequently upstream correlations cannot invalidate the selection. This
//! does not certify a frontend's unspecified floating point normalization.

use ndarray::{Array1, ArrayD, ArrayView1, ArrayViewMut1, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::certify::work::{WorkBudget, WorkRefusal, WorkRequest};
use crate::layers::softmax::weighted_support::Interval as I;
use crate::BatchedLinearBounds;

const MAX_CONSTRUCTION_BYTES: usize = 256 * 1024 * 1024;
const MAX_DEFAULT_WORK: u64 = 1_000_000_000;
pub(crate) mod baseline;
mod bridges;
mod demand;
// Private checked group prototype; no decoder/default route consumes it yet.
#[allow(dead_code)]
pub(crate) mod f64_group_support;
pub(crate) mod f64_support;
mod geometry;
pub(crate) use bridges::{
    admit_bridge, batched_to_scalar, batched_to_scalar_multi, complete_bridge, scalar_to_batched,
    scalar_to_batched_multi,
};
mod primitives;
mod rows;
use geometry::{sqrt_interval, sqrt_point, square, Geometry};
use rows::{error_charge, row};

fn refusal(message: &str) -> NyError {
    NyError::UnsupportedOp(format!("checked radial normalization: {message}"))
}

/// Qualify the current thread at every mathematical normalization proof entry;
/// construction on another thread or a decoder-only guard is insufficient.
fn require_environment() -> Result<()> {
    #[cfg(test)]
    if environment_tests::refused_for_test() {
        return Err(NyError::SoundnessRefusal(
            "radial normalization f64 interval environment refused".into(),
        ));
    }
    ny_core::require_f64_interval_proof_environment()
}

fn checkpoint(budget: Option<&WorkBudget>) -> Result<()> {
    if ny_core::instance_budget::deadline().is_some_and(|d| std::time::Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded("radial normalization".into()));
    }
    budget.map_or(Ok(()), |budget| {
        budget.checkpoint().map_err(|error| match error {
            WorkRefusal::Deadline => NyError::DeadlineExceeded("radial normalization".into()),
            _ => refusal("shared work admission refused"),
        })
    })
}

const ZERO_WRITE_CHUNK: usize = 256;

fn write_zero_row_with_poll(
    row: ArrayViewMut1<'_, f32>,
    mut poll: impl FnMut() -> Result<()>,
) -> Result<()> {
    for (index, value) in row.into_iter().enumerate() {
        if index % ZERO_WRITE_CHUNK == 0 {
            poll()?;
        }
        *value = 0.0;
    }
    poll()
}

fn write_zero_row(row: ArrayViewMut1<'_, f32>, budget: Option<&WorkBudget>) -> Result<()> {
    write_zero_row_with_poll(row, || checkpoint(budget))
}

fn append_zero_row_with_poll(
    output: &mut Vec<f32>,
    n: usize,
    mut poll: impl FnMut() -> Result<()>,
) -> Result<()> {
    let zeros = [0.0f32; ZERO_WRITE_CHUNK];
    let mut remaining = n;
    while remaining != 0 {
        poll()?;
        let count = remaining.min(ZERO_WRITE_CHUNK);
        output.extend_from_slice(&zeros[..count]);
        remaining -= count;
    }
    poll()
}

fn append_zero_row(output: &mut Vec<f32>, n: usize, budget: Option<&WorkBudget>) -> Result<()> {
    append_zero_row_with_poll(output, n, || checkpoint(budget))
}

/// A present error plane must remain present even when its last-axis row is
/// strided. At most two row buffers (8*n bytes) coexist; both the baseline's
/// 512*n and radial construction's 256*n scratch terms include that workspace.
/// Logical iteration preserves F-order and sliced layouts without changing
/// their coefficient meaning. Invalid errors are refused rather than dropped.
fn logical_error_row(
    row: Option<ArrayView1<'_, f32>>,
    budget: Option<&WorkBudget>,
) -> Result<Option<Vec<f32>>> {
    let Some(row) = row else {
        return Ok(None);
    };
    checkpoint(budget)?;
    let mut values = Vec::with_capacity(row.len());
    for (i, &value) in row.iter().enumerate() {
        if i % 256 == 0 {
            checkpoint(budget)?;
        }
        if !value.is_finite() || value < 0.0 {
            return Err(refusal("invalid incoming coefficient error"));
        }
        values.push(value);
    }
    checkpoint(budget)?;
    Ok(Some(values))
}

/// Convert only the error carrier, in logical index order. ndarray's default
/// reshape requires C-order even when adding/removing singleton dimensions.
/// The baseline's full-plane copies fit its 24*elements term (8*elements of
/// errors, 8*elements of output A and at most8*elements of output bias).
/// Radial uses this on one cloned plane at a time: its input copies, output
/// clone and one transient plane occupy at most36*elements, below40*elements.
fn logical_error_plane(error: &ArrayD<f32>, budget: Option<&WorkBudget>) -> Result<ArrayD<f32>> {
    checkpoint(budget)?;
    let mut values = Vec::with_capacity(error.len());
    for (i, &value) in error.iter().enumerate() {
        if i % 256 == 0 {
            checkpoint(budget)?;
        }
        if !value.is_finite() || value < 0.0 {
            return Err(refusal("invalid incoming coefficient error"));
        }
        values.push(value);
    }
    checkpoint(budget)?;
    ArrayD::from_shape_vec(error.raw_dim(), values).map_err(|_| refusal("logical error shape"))
}

fn error_plane_for_reshape<'a>(
    error: &'a ArrayD<f32>,
    budget: Option<&WorkBudget>,
) -> Result<std::borrow::Cow<'a, ArrayD<f32>>> {
    if error.is_standard_layout() {
        Ok(std::borrow::Cow::Borrowed(error))
    } else {
        logical_error_plane(error, budget).map(std::borrow::Cow::Owned)
    }
}

/// Minimum candidate-minus-baseline over the box; error encloses every
/// intended baseline coefficient. This gate is deliberately pointwise.
fn dominates(
    candidate: &[f32],
    candidate_b: f32,
    baseline: &[f32],
    baseline_b: f32,
    error: Option<&[f32]>,
    lower: &[f32],
    upper: &[f32],
    is_lower: bool,
) -> bool {
    if baseline.iter().any(|x| !x.is_finite()) || baseline_b.is_nan() {
        return false;
    }
    if (is_lower && baseline_b == f32::NEG_INFINITY) || (!is_lower && baseline_b == f32::INFINITY) {
        return true;
    }
    if !baseline_b.is_finite() {
        return false;
    }
    let sign = if is_lower { 1.0 } else { -1.0 };
    let mut difference =
        I::point(sign * candidate_b as f64).sub(I::point(sign * baseline_b as f64));
    for i in 0..candidate.len() {
        let coefficient =
            I::point(sign * candidate[i] as f64).sub(I::point(sign * baseline[i] as f64));
        difference = difference.add(coefficient.mul(I::new(lower[i] as f64, upper[i] as f64)));
        if let Some(error) = error {
            if !error[i].is_finite() || error[i] < 0.0 {
                return false;
            }
            difference = difference.sub(I::point(error[i] as f64).mul(I::point(
                (lower[i] as f64).abs().max((upper[i] as f64).abs()),
            )));
        }
    }
    difference.finite() && difference.lo >= 0.0
}

fn lower_cast(x: f64) -> f32 {
    let result = x as f32;
    if result as f64 > x {
        next_down_f32(result)
    } else {
        result
    }
}
fn upper_cast(x: f64) -> f32 {
    let result = x as f32;
    if (result as f64) < x {
        next_up_f32(result)
    } else {
        result
    }
}

pub(crate) fn refine_batched(
    bounds: &BatchedLinearBounds,
    baseline: BatchedLinearBounds,
    pre: &BoundedTensor,
    gamma: &Array1<f32>,
    beta: Option<&Array1<f32>>,
    eps: f32,
    centered: bool,
) -> Result<BatchedLinearBounds> {
    checkpoint(None)?;
    require_environment()?;
    // Optional arithmetic/admission failures preserve the completed baseline;
    // expired query deadlines still propagate rather than publish late work.
    match calculate(
        bounds,
        Some(&baseline),
        pre,
        gamma,
        beta,
        eps,
        centered,
        None,
    ) {
        Ok(result) => Ok(result),
        Err(error @ NyError::DeadlineExceeded(_)) => Err(error),
        Err(_) => {
            checkpoint(None)?;
            Ok(baseline)
        }
    }
}

pub(crate) fn candidate_batched(
    bounds: &BatchedLinearBounds,
    pre: &BoundedTensor,
    gamma: &Array1<f32>,
    beta: Option<&Array1<f32>>,
    eps: f32,
    centered: bool,
    budget: Option<&WorkBudget>,
) -> Result<BatchedLinearBounds> {
    calculate(bounds, None, pre, gamma, beta, eps, centered, budget)
}

fn calculate(
    bounds: &BatchedLinearBounds,
    baseline: Option<&BatchedLinearBounds>,
    pre: &BoundedTensor,
    gamma: &Array1<f32>,
    beta: Option<&Array1<f32>>,
    eps: f32,
    centered: bool,
    budget: Option<&WorkBudget>,
) -> Result<BatchedLinearBounds> {
    checkpoint(budget)?;
    require_environment()?;
    bounds.validate_internal_shapes()?;
    if !eps.is_finite() || eps < super::NORMALIZATION_MIN_EPS {
        return Err(refusal("authored epsilon below supported minimum"));
    }
    let shape = bounds.lower_a.shape();
    let n = shape[shape.len() - 1];
    let outputs = shape[shape.len() - 2];
    let groups = checked_shape_product(&shape[..shape.len() - 2])
        .ok_or_else(|| refusal("batch size overflow"))?;
    if n == 0
        || groups == 0
        || n > (1usize << 24)
        || gamma.len() != n
        || beta.is_some_and(|b| b.len() != n)
    {
        return Err(refusal("unsupported normalization shape"));
    }
    if pre.shape().last() != Some(&n)
        || pre.len()
            != groups
                .checked_mul(n)
                .ok_or_else(|| refusal("input size overflow"))?
    {
        return Err(refusal("input box does not match normalization groups"));
    }
    let elements = groups
        .checked_mul(outputs)
        .and_then(|v| v.checked_mul(n))
        .ok_or_else(|| refusal("output size overflow"))?;
    let scratch = elements
        .checked_mul(40)
        .and_then(|v| {
            groups
                .checked_mul(n)
                .and_then(|k| k.checked_mul(16))
                .and_then(|k| v.checked_add(k))
        })
        .and_then(|v| n.checked_mul(256).and_then(|k| v.checked_add(k)))
        .and_then(|v| {
            groups
                .checked_mul(outputs)
                .and_then(|k| k.checked_mul(32))
                .and_then(|k| v.checked_add(k))
        })
        .and_then(|v| v.checked_add(16 * 1024))
        .ok_or_else(|| refusal("scratch prediction overflow"))?;
    if scratch
        .checked_add(demand::mask_bytes(groups, outputs)?)
        .is_none_or(|bytes| bytes > MAX_CONSTRUCTION_BYTES)
    {
        return Err(refusal("finite construction admission refused"));
    }
    let demand = demand::inspect(bounds, pre, gamma, beta, groups, outputs, n, true, budget)?;
    let work = demand::construction_work(&demand, elements, groups * outputs, pre.len(), n)?;
    let total_work = work
        .checked_add(demand.scan_work)
        .ok_or_else(|| refusal("work prediction overflow"))?;
    if budget.is_none() && total_work > MAX_DEFAULT_WORK {
        return Err(refusal("finite construction admission refused"));
    }
    let mut lease = budget
        .map(|b| {
            b.lease(WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: 0,
                work,
                proposals: 1,
            })
        })
        .transpose()
        .map_err(|e| match e {
            WorkRefusal::Deadline => NyError::DeadlineExceeded("radial normalization".into()),
            _ => refusal("shared admission refused"),
        })?;
    checkpoint(budget)?;
    let lower: Vec<f32> = pre.lower().iter().copied().collect();
    let upper: Vec<f32> = pre.upper().iter().copied().collect();
    let gamma: Vec<f32> = gamma.iter().copied().collect();
    let beta = beta.map(|b| b.iter().copied().collect::<Vec<_>>());
    let c_lower: Vec<f32> = bounds.lower_a.iter().copied().collect();
    let c_upper: Vec<f32> = bounds.upper_a.iter().copied().collect();
    let e_lower = bounds
        .lower_a_err
        .as_ref()
        .map(|e| e.iter().copied().collect::<Vec<_>>());
    let e_upper = bounds
        .upper_a_err
        .as_ref()
        .map(|e| e.iter().copied().collect::<Vec<_>>());
    let bias_lower: Vec<f32> = bounds.lower_b.iter().copied().collect();
    let bias_upper: Vec<f32> = bounds.upper_b.iter().copied().collect();
    let mut result = if let Some(baseline) = baseline {
        if baseline.lower_a.shape() != shape {
            return Err(refusal("baseline shape mismatch"));
        }
        baseline.clone()
    } else {
        BatchedLinearBounds::new(
            ArrayD::zeros(IxDyn(shape)),
            ArrayD::zeros(bounds.lower_b.raw_dim()),
            ArrayD::zeros(IxDyn(shape)),
            ArrayD::zeros(bounds.upper_b.raw_dim()),
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?
    };
    for error in [&mut result.lower_a_err, &mut result.upper_a_err] {
        if let Some(plane) = error.as_ref().filter(|plane| !plane.is_standard_layout()) {
            let standard = logical_error_plane(plane, budget)?;
            *error = Some(standard);
        }
    }
    let mut out_lower = result
        .lower_a
        .view_mut()
        .into_shape_with_order((groups * outputs, n))
        .map_err(|_| refusal("baseline lower layout"))?;
    let mut out_upper = result
        .upper_a
        .view_mut()
        .into_shape_with_order((groups * outputs, n))
        .map_err(|_| refusal("baseline upper layout"))?;
    let mut out_bias_lower = result
        .lower_b
        .view_mut()
        .into_shape_with_order(groups * outputs)
        .map_err(|_| refusal("baseline lower bias layout"))?;
    let mut out_bias_upper = result
        .upper_b
        .view_mut()
        .into_shape_with_order(groups * outputs)
        .map_err(|_| refusal("baseline upper bias layout"))?;
    let mut out_error_lower = result
        .lower_a_err
        .as_mut()
        .map(|e| e.view_mut().into_shape_with_order((groups * outputs, n)))
        .transpose()
        .map_err(|_| refusal("baseline lower error layout"))?;
    let mut out_error_upper = result
        .upper_a_err
        .as_mut()
        .map(|e| e.view_mut().into_shape_with_order((groups * outputs, n)))
        .transpose()
        .map_err(|_| refusal("baseline upper error layout"))?;
    for group in 0..groups {
        checkpoint(budget)?;
        let lo = &lower[group * n..(group + 1) * n];
        let hi = &upper[group * n..(group + 1) * n];
        let geometry = if demand.zero_groups[group] {
            None
        } else {
            Some(Geometry::new(lo, hi, eps, centered, budget)?)
        };
        for output in 0..outputs {
            if output % 16 == 0 {
                checkpoint(budget)?;
            }
            let r = group * outputs + output;
            let start = r * n;
            if demand.zero_rows[r] {
                // The original affine demand is exactly the original bias.
                // No geometry, learned scale or uncertainty term is omitted.
                write_zero_row(out_lower.row_mut(r), budget)?;
                write_zero_row(out_upper.row_mut(r), budget)?;
                out_bias_lower[r] = bias_lower[r];
                out_bias_upper[r] = bias_upper[r];
                if let Some(error) = out_error_lower.as_mut() {
                    write_zero_row(error.row_mut(r), budget)?;
                }
                if let Some(error) = out_error_upper.as_mut() {
                    write_zero_row(error.row_mut(r), budget)?;
                }
                continue;
            }
            let geometry = geometry
                .as_ref()
                .ok_or_else(|| refusal("nonzero row lacks geometry"))?;
            let el = e_lower.as_ref().map(|e| &e[start..start + n]);
            let eu = e_upper.as_ref().map(|e| &e[start..start + n]);
            let lower_row = row(
                &c_lower[start..start + n],
                el,
                bias_lower[r],
                &gamma,
                beta.as_deref(),
                centered,
                geometry,
                budget,
            )?;
            let upper_row = row(
                &c_upper[start..start + n],
                eu,
                bias_upper[r],
                &gamma,
                beta.as_deref(),
                centered,
                geometry,
                budget,
            )?;
            let lb = lower_cast(lower_row.bias.lo);
            let ub = upper_cast(upper_row.bias.hi);
            let baseline_l = out_lower.row(r);
            let baseline_u = out_upper.row(r);
            let baseline_el =
                logical_error_row(out_error_lower.as_ref().map(|e| e.row(r)), budget)?;
            let baseline_eu =
                logical_error_row(out_error_upper.as_ref().map(|e| e.row(r)), budget)?;
            let take_lower = baseline.is_none()
                || dominates(
                    &lower_row.a,
                    lb,
                    baseline_l
                        .as_slice()
                        .ok_or_else(|| refusal("lower row layout"))?,
                    out_bias_lower[r],
                    baseline_el.as_deref(),
                    lo,
                    hi,
                    true,
                );
            let take_upper = baseline.is_none()
                || dominates(
                    &upper_row.a,
                    ub,
                    baseline_u
                        .as_slice()
                        .ok_or_else(|| refusal("upper row layout"))?,
                    out_bias_upper[r],
                    baseline_eu.as_deref(),
                    lo,
                    hi,
                    false,
                );
            if take_lower {
                for i in 0..n {
                    out_lower[[r, i]] = lower_row.a[i];
                }
                out_bias_lower[r] = lb;
                if let Some(e) = out_error_lower.as_mut() {
                    write_zero_row(e.row_mut(r), budget)?;
                }
            }
            if take_upper {
                for i in 0..n {
                    out_upper[[r, i]] = upper_row.a[i];
                }
                out_bias_upper[r] = ub;
                if let Some(e) = out_error_upper.as_mut() {
                    write_zero_row(e.row_mut(r), budget)?;
                }
            }
        }
    }
    checkpoint(budget)?;
    result.validate_no_nan()?;
    checkpoint(budget)?;
    if let Some(lease) = lease.as_mut() {
        lease
            .complete()
            .map_err(|_| refusal("completion refused"))?;
    }
    Ok(result)
}

#[cfg(test)]
mod demand_tests;
#[cfg(test)]
mod environment_tests;
#[cfg(test)]
mod layout_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pointwise_gate_rejects_local_minimum_improvement_that_loses_correlation() {
        // Baseline x versus constant 0.5 on [0,1]: constant has a larger
        // box minimum but loses at x=1. The default must keep the baseline.
        assert!(!dominates(
            &[0.0],
            0.5,
            &[1.0],
            0.0,
            None,
            &[0.0],
            &[1.0],
            true
        ));
        assert!(dominates(
            &[1.0],
            0.1,
            &[1.0],
            0.0,
            None,
            &[-1.0],
            &[1.0],
            true
        ));
        assert!(!dominates(
            &[1.0],
            0.1,
            &[1.0],
            0.0,
            Some(&[0.25]),
            &[-1.0],
            &[1.0],
            true
        ));
    }
    #[test]
    fn checked_sqrt_includes_squares_and_zero() {
        for x in [0.0, 1e-24, 0.5, 1.0, 3.0, 1e76] {
            let r = sqrt_point(x).unwrap();
            assert!(r.lo * r.lo <= x && r.hi * r.hi >= x);
        }
        assert!(sqrt_point(-1.0).is_none());
    }
}
