// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Checked scalar/batched normalization bridges and their allocation leases.

use super::{checkpoint, lower_cast, refusal, upper_cast, I, MAX_CONSTRUCTION_BYTES};
use crate::certify::work::{WorkBudget, WorkRefusal, WorkRequest};
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array1, ArrayD};
use ny_core::{NyError, Result};

pub(crate) fn scalar_to_batched(bounds: &LinearBounds) -> Result<BatchedLinearBounds> {
    let _admission = admit_bridge(bounds, None)?;
    let mut result = super::super::decomposed::scalar_bounds_to_batched(bounds)?;
    result.lower_a_err = bounds.lower_a_err.as_ref().map(|e| e.clone().into_dyn());
    result.upper_a_err = bounds.upper_a_err.as_ref().map(|e| e.clone().into_dyn());
    Ok(result)
}

pub(crate) fn scalar_to_batched_multi(
    bounds: &LinearBounds,
    groups: usize,
    n: usize,
) -> Result<BatchedLinearBounds> {
    let _admission = admit_bridge(bounds, None)?;
    let mut result =
        super::super::decomposed::scalar_bounds_to_batched_multi_dim(bounds, groups, n)?;
    let transform = |e: &ndarray::Array2<f32>| -> Result<ArrayD<f32>> {
        Ok(e.view()
            .into_shape_with_order((bounds.num_outputs(), groups, n))
            .map_err(|_| refusal("scalar error shape"))?
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_owned()
            .into_dyn())
    };
    result.lower_a_err = bounds.lower_a_err.as_ref().map(transform).transpose()?;
    result.upper_a_err = bounds.upper_a_err.as_ref().map(transform).transpose()?;
    Ok(result)
}

pub(crate) fn batched_to_scalar(bounds: &BatchedLinearBounds) -> Result<LinearBounds> {
    let mut result = super::super::decomposed::batched_bounds_to_scalar(bounds)?;
    result.lower_a_err = bounds
        .lower_a_err
        .as_ref()
        .map(|e| e.clone().into_dimensionality())
        .transpose()
        .map_err(|_| refusal("lower error bridge"))?;
    result.upper_a_err = bounds
        .upper_a_err
        .as_ref()
        .map(|e| e.clone().into_dimensionality())
        .transpose()
        .map_err(|_| refusal("upper error bridge"))?;
    Ok(result)
}

/// Merge normalization groups with outward bias summation and preserved
/// coefficient error. The historical bridge used nearest binary32 summation.
pub(crate) fn batched_to_scalar_multi(
    bounds: &BatchedLinearBounds,
    original_lower: &Array1<f32>,
    original_upper: &Array1<f32>,
) -> Result<LinearBounds> {
    let shape = bounds.lower_a.shape();
    if shape.len() != 3 {
        return Err(refusal("multi-group scalar bridge needs rank three"));
    }
    let (groups, outputs, n) = (shape[0], shape[1], shape[2]);
    let flat = groups
        .checked_mul(n)
        .ok_or_else(|| refusal("scalar bridge size overflow"))?;
    let transform = |a: &ArrayD<f32>| -> Result<ndarray::Array2<f32>> {
        Ok(a.view()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| refusal("scalar bridge rank"))?
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape_with_order((outputs, flat))
            .map_err(|_| refusal("scalar bridge layout"))?
            .into_owned())
    };
    let mut lower = Array1::zeros(outputs);
    let mut upper = Array1::zeros(outputs);
    let bl = bounds
        .lower_b
        .view()
        .into_shape_with_order((groups, outputs))
        .map_err(|_| refusal("scalar bridge bias layout"))?;
    let bu = bounds
        .upper_b
        .view()
        .into_shape_with_order((groups, outputs))
        .map_err(|_| refusal("scalar bridge bias layout"))?;
    for row in 0..outputs {
        if row % 16 == 0 {
            checkpoint(None)?;
        }
        let mut l = I::point(original_lower[row] as f64);
        let mut u = I::point(original_upper[row] as f64);
        for group in 0..groups {
            l = l.add(I::point(bl[[group, row]] as f64));
            u = u.add(I::point(bu[[group, row]] as f64));
        }
        lower[row] = lower_cast(l.lo);
        upper[row] = upper_cast(u.hi);
    }
    let mut result = LinearBounds::new(
        transform(&bounds.lower_a)?,
        lower,
        transform(&bounds.upper_a)?,
        upper,
    )?;
    result.lower_a_err = bounds.lower_a_err.as_ref().map(transform).transpose()?;
    result.upper_a_err = bounds.upper_a_err.as_ref().map(transform).transpose()?;
    Ok(result)
}

/// Scalar bridges retain their own reservation while the inner candidate
/// uses a separate construction lease; neither allocation escapes admission.
pub(crate) fn admit_bridge(
    bounds: &LinearBounds,
    budget: Option<&WorkBudget>,
) -> Result<Option<crate::certify::work::WorkLease>> {
    checkpoint(budget)?;
    let scratch = bounds
        .lower_a
        .len()
        .checked_mul(16)
        .and_then(|v| {
            bounds
                .lower_b
                .len()
                .checked_mul(16)
                .and_then(|k| v.checked_add(k))
        })
        .and_then(|v| v.checked_add(16 * 1024))
        .ok_or_else(|| refusal("scalar bridge prediction overflow"))?;
    if scratch > MAX_CONSTRUCTION_BYTES {
        return Err(refusal("scalar bridge admission"));
    }
    budget
        .map(|b| {
            b.lease(WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: 0,
                work: bounds.lower_a.len() as u64,
                proposals: 0,
            })
        })
        .transpose()
        .map_err(|e| match e {
            WorkRefusal::Deadline => {
                NyError::DeadlineExceeded("radial normalization bridge".into())
            }
            _ => refusal("shared bridge admission"),
        })
}

pub(crate) fn complete_bridge(lease: &mut Option<crate::certify::work::WorkLease>) -> Result<()> {
    checkpoint(None)?;
    if let Some(lease) = lease {
        lease.complete().map_err(|e| match e {
            WorkRefusal::Deadline => {
                NyError::DeadlineExceeded("radial normalization bridge".into())
            }
            _ => refusal("bridge completion"),
        })?;
    }
    Ok(())
}
