// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::certify::work::{LeaseClass, WorkBudget, WorkLease, WorkRefusal, WorkRequest};
use ny_core::{checked_shape_product, NyError, Result};

const MAX_OUTPUT_BYTES: usize = 256 * 1024 * 1024;
const MAX_SCRATCH_BYTES: usize = 512 * 1024 * 1024;
const MAX_DEFAULT_WORK: u64 = 10_000_000_000;
const WORK_PER_VISIBLE_KEY: u64 = 16_384;

pub(super) fn refusal(r: WorkRefusal) -> NyError {
    if r == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("checked causal softmax work budget".into())
    } else {
        NyError::UnsupportedOp(format!("causal softmax work admission: {r:?}"))
    }
}

pub(super) fn checkpoint(budget: Option<&WorkBudget>) -> Result<()> {
    if ny_core::instance_budget::deadline().is_some_and(|d| std::time::Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded("checked causal softmax".into()));
    }
    budget.map_or(Ok(()), |b| {
        b.checkpoint_for(LeaseClass::Fallback).map_err(refusal)
    })
}

pub(super) struct Admission {
    pub(super) checked: bool,
    pub(super) lease: Option<WorkLease>,
}

impl Admission {
    pub(super) fn new(
        shape: &[usize],
        window: Option<usize>,
        budget: Option<&WorkBudget>,
    ) -> Result<Self> {
        checkpoint(budget)?;
        let overflow = || NyError::UnsupportedOp("causal softmax admission overflow".into());
        let elements = checked_shape_product(shape).ok_or_else(overflow)?;
        let output = elements.checked_mul(8).ok_or_else(overflow)?;
        if output > MAX_OUTPUT_BYTES {
            return Err(NyError::UnsupportedOp(
                "causal softmax output memory admission".into(),
            ));
        }
        let queries = shape[shape.len() - 2];
        let slices = checked_shape_product(&shape[..shape.len() - 2]).ok_or_else(overflow)?;
        let width = window.map_or(queries, |w| queries.min(w.saturating_add(1)));
        let triangular = |n: usize| n.checked_mul(n.checked_add(1)?).map(|p| p / 2);
        let row_pairs = triangular(width)
            .and_then(|p| {
                (queries - width)
                    .checked_mul(width)
                    .and_then(|q| p.checked_add(q))
            })
            .ok_or_else(overflow)?;
        let pairs = row_pairs.checked_mul(slices).ok_or_else(overflow)? as u64;
        let work = pairs
            .checked_mul(WORK_PER_VISIBLE_KEY)
            .and_then(|n| {
                (elements as u64)
                    .checked_mul(16)
                    .and_then(|e| n.checked_add(e))
            })
            .ok_or_else(overflow)?;
        // Two endpoint caches, input vectors, exponential construction scratch,
        // excluded-max scans and returned row tuples coexist with dense output.
        let scratch = width
            .checked_mul(192)
            .and_then(|n| n.checked_add(output))
            .and_then(|n| n.checked_add(32 * 1024))
            .ok_or_else(overflow)?;
        let mut checked =
            scratch <= MAX_SCRATCH_BYTES && (budget.is_some() || work <= MAX_DEFAULT_WORK);
        let mut request = WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: 0,
            work,
            proposals: 0,
        };
        let mut lease = if checked {
            budget
                .map(|b| b.lease_for(LeaseClass::Fallback, request))
                .transpose()
        } else {
            Ok(None)
        };
        // A bounded simplex-box fallback can still finish when checked exp
        // refinement cannot be admitted. It never runs the old raw-exp loops.
        if matches!(
            lease,
            Err(WorkRefusal::WorkLimit | WorkRefusal::ScratchLimit)
        ) {
            checked = false;
        } else if let Err(r) = lease {
            return Err(refusal(r));
        }
        if !checked {
            request.scratch_bytes = width
                .checked_mul(16)
                .and_then(|row| output.checked_add(row))
                .and_then(|n| n.checked_add(32 * 1024))
                .ok_or_else(overflow)?;
            request.work = pairs
                .checked_mul(64)
                .and_then(|n| {
                    (elements as u64)
                        .checked_mul(16)
                        .and_then(|e| n.checked_add(e))
                })
                .ok_or_else(overflow)?;
            if request.scratch_bytes > MAX_SCRATCH_BYTES
                || (budget.is_none() && request.work > MAX_DEFAULT_WORK)
            {
                return Err(NyError::UnsupportedOp(
                    "causal softmax fallback admission".into(),
                ));
            }
            lease = budget
                .map(|b| b.lease_for(LeaseClass::Fallback, request))
                .transpose();
        }
        checkpoint(budget)?;
        Ok(Self {
            checked,
            lease: lease.map_err(refusal)?,
        })
    }
}
