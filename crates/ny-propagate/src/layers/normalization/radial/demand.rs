// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//! Exact demand inspection, admitted separately from subsequent construction.
use super::{checkpoint, refusal, WorkBudget, WorkRefusal, WorkRequest, MAX_CONSTRUCTION_BYTES};
use crate::BatchedLinearBounds;
use ndarray::Array1;
use ny_core::Result;
use ny_tensor::BoundedTensor;

pub(super) struct DemandPlan {
    pub(super) zero_rows: Vec<bool>,
    pub(super) zero_groups: Vec<bool>,
    pub(super) active_rows: usize,
    pub(super) scan_work: u64,
    // A completed scan still owns its mask capacity until the plan is dropped.
    _lease: Option<crate::certify::work::WorkLease>,
}

fn admitted<T>(value: std::result::Result<T, WorkRefusal>) -> Result<T> {
    value.map_err(|error| match error {
        WorkRefusal::Deadline => {
            ny_core::NyError::DeadlineExceeded("radial demand inspection".into())
        }
        _ => refusal("demand inspection admission refused"),
    })
}

pub(super) fn mask_bytes(groups: usize, outputs: usize) -> Result<usize> {
    groups
        .checked_mul(outputs)
        .and_then(|rows| rows.checked_add(groups))
        .and_then(|bytes| bytes.checked_add(size_of::<DemandPlan>()))
        .ok_or_else(|| refusal("demand mask overflow"))
}

pub(super) fn signed_zero(value: f32) -> bool {
    value.to_bits() & 0x7fff_ffff == 0
}

/// Callers validate shape, epsilon and dense-memory admission first. Inspection
/// visits original logical values; no tolerance, scale cancellation or f32
/// arithmetic can classify a nonzero/subnormal coefficient as absent demand.
pub(super) fn inspect(
    bounds: &BatchedLinearBounds,
    pre: &BoundedTensor,
    gamma: &Array1<f32>,
    beta: Option<&Array1<f32>>,
    groups: usize,
    outputs: usize,
    n: usize,
    finite_bias: bool,
    budget: Option<&WorkBudget>,
) -> Result<DemandPlan> {
    checkpoint(budget)?;
    let rows = groups
        .checked_mul(outputs)
        .ok_or_else(|| refusal("demand rows overflow"))?;
    let mask_bytes = mask_bytes(groups, outputs)?;
    if mask_bytes > MAX_CONSTRUCTION_BYTES {
        return Err(refusal("demand mask admission"));
    }
    let scan_work = bounds
        .lower_a
        .len()
        .checked_mul(2)
        .and_then(|w| w.checked_add(bounds.lower_a_err.as_ref().map_or(0, |e| e.len())))
        .and_then(|w| w.checked_add(bounds.upper_a_err.as_ref().map_or(0, |e| e.len())))
        .and_then(|w| w.checked_add(pre.len().checked_mul(2)?))
        .and_then(|w| w.checked_add(gamma.len()))
        .and_then(|w| w.checked_add(beta.map_or(0, |b| b.len())))
        .and_then(|w| w.checked_add(rows.checked_mul(4)?))
        .and_then(|w| w.checked_add(groups))
        .ok_or_else(|| refusal("demand scan overflow"))?;
    let scan_work = u64::try_from(scan_work).map_err(|_| refusal("demand work overflow"))?;
    // A count describes scalar/ID visits, not seconds or process RSS.
    let mut lease = budget
        .map(|budget| {
            admitted(budget.lease(WorkRequest {
                scratch_bytes: mask_bytes,
                persistent_bytes: 0,
                work: scan_work,
                // Mandatory inspection is not another refinement proposal.
                proposals: 0,
            }))
        })
        .transpose()?;
    let mut zero_rows = vec![true; rows];
    for coefficients in [&bounds.lower_a, &bounds.upper_a] {
        for (index, &value) in coefficients.iter().enumerate() {
            if index % 256 == 0 {
                checkpoint(budget)?;
            }
            if !value.is_finite() {
                return Err(refusal("finite direction required for demand"));
            }
            zero_rows[index / n] &= signed_zero(value);
        }
    }
    for errors in bounds.lower_a_err.iter().chain(bounds.upper_a_err.iter()) {
        for (index, &value) in errors.iter().enumerate() {
            if index % 256 == 0 {
                checkpoint(budget)?;
            }
            if !value.is_finite() || value < 0.0 {
                return Err(refusal("invalid incoming coefficient error"));
            }
            zero_rows[index / n] &= signed_zero(value);
        }
    }
    for (index, (&lo, &hi)) in pre.lower().iter().zip(pre.upper().iter()).enumerate() {
        if index % 256 == 0 {
            checkpoint(budget)?;
        }
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(refusal("finite ordered input box required"));
        }
    }
    for (index, &value) in gamma
        .iter()
        .chain(beta.into_iter().flat_map(|b| b.iter()))
        .enumerate()
    {
        if index % 256 == 0 {
            checkpoint(budget)?;
        }
        if !value.is_finite() {
            return Err(refusal("finite scale and shift required"));
        }
    }
    for (index, &value) in bounds
        .lower_b
        .iter()
        .chain(bounds.upper_b.iter())
        .enumerate()
    {
        if index % 256 == 0 {
            checkpoint(budget)?;
        }
        if value.is_nan() || (finite_bias && !value.is_finite()) {
            return Err(refusal("finite direction and bias required"));
        }
    }
    let mut zero_groups = Vec::with_capacity(groups);
    let mut active_rows = 0;
    for group in 0..groups {
        if group % 256 == 0 {
            checkpoint(budget)?;
        }
        let rows = &zero_rows[group * outputs..(group + 1) * outputs];
        zero_groups.push(rows.iter().all(|&zero| zero));
        active_rows += rows.iter().filter(|&&zero| !zero).count();
    }
    checkpoint(budget)?;
    if let Some(lease) = lease.as_mut() {
        admitted(lease.complete())?;
    }
    Ok(DemandPlan {
        zero_rows,
        zero_groups,
        active_rows,
        scan_work,
        _lease: lease,
    })
}

/// Dense copies, returned carriers and zero-row writes remain full-size work.
/// The old512-per-coordinate refinement allowance now applies only to active
/// rows, which also cover their groups' geometry (at least one active row per
/// constructed group). This is a conservative reservation, not observed work.
pub(super) fn construction_work(
    plan: &DemandPlan,
    elements: usize,
    rows: usize,
    input: usize,
    n: usize,
) -> Result<u64> {
    let work = plan
        .active_rows
        .checked_mul(n)
        .and_then(|w| w.checked_mul(512))
        .and_then(|w| w.checked_add(elements.checked_mul(16)?))
        .and_then(|w| w.checked_add(rows.checked_mul(8)?))
        .and_then(|w| w.checked_add(input.checked_mul(4)?))
        .and_then(|w| w.checked_add(n.checked_mul(4)?))
        .ok_or_else(|| refusal("demand construction work overflow"))?;
    u64::try_from(work).map_err(|_| refusal("demand construction work overflow"))
}
