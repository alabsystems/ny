// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Demand is exact zero detection on requested carriers, never a value/score
//! heuristic. The scan is admitted before allocation and retains its lease
//! while the candidate uses its own lease from the same ledger.

use super::{
    checkpoint, work_refusal, AttentionMask, LinearBounds, NyError, Result, SelfAttentionLayer,
    SliceDims, WorkBudget, WorkRequest, MAX_CONSTRUCTION_BYTES, MAX_DEFAULT_WORK,
};
use crate::certify::work::WorkLease;
use std::time::Instant;

pub(super) struct Demand {
    active: Vec<u8>,
    pub(super) summary: Summary,
    _lease: Option<WorkLease>,
}
#[derive(Clone, Copy)]
pub(super) struct Summary {
    pub(super) active_queries: usize,
    pub(super) visible_pairs: usize,
    pub(super) max_visible_keys: usize,
    pub(super) scan_work: u64,
    pub(super) map_bytes: usize,
}

pub(super) fn visible_range(layer: &SelfAttentionLayer, query: usize, sk: usize) -> (usize, usize) {
    if layer.mask == AttentionMask::Standard {
        return (0, sk);
    }
    // The caller retains the existing causal sq<=sk and window shape gates.
    let end = query.saturating_add(1).min(sk);
    let start = layer
        .window_size
        .map_or(0, |w| end.saturating_sub(w.saturating_add(1)));
    (start, end)
}

impl Demand {
    pub(super) fn active(&self, slice: usize, query: usize, sq: usize) -> bool {
        self.active[slice * sq + query] != 0
    }
    pub(super) fn scan(
        layer: &SelfAttentionLayer,
        node: &LinearBounds,
        dims: &SliceDims,
        deadline: Option<Instant>,
        budget: Option<&WorkBudget>,
        skip_zero: bool,
    ) -> Result<Self> {
        checkpoint(deadline, budget)?;
        node.validate_internal_shapes()?;
        let queries = dims
            .n_slices
            .checked_mul(dims.sq)
            .ok_or_else(|| NyError::UnsupportedOp("attention demand shape overflow".into()))?;
        let arrays = 2usize
            + usize::from(node.lower_a_err().is_some())
            + usize::from(node.upper_a_err().is_some());
        // A full scan, including all coefficient/error validations, plus both
        // per-query scan and routing passes. Exact zeros include negative zero.
        let scan_work = node
            .lower_a()
            .len()
            .checked_mul(arrays)
            .and_then(|n| queries.checked_mul(16).and_then(|q| n.checked_add(q)))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| NyError::UnsupportedOp("attention demand work overflow".into()))?;
        let scratch = queries
            .checked_add(16 * 1024)
            .ok_or_else(|| NyError::UnsupportedOp("attention demand scratch overflow".into()))?;
        if scratch > MAX_CONSTRUCTION_BYTES || (budget.is_none() && scan_work > MAX_DEFAULT_WORK) {
            return Err(NyError::UnsupportedOp(
                "attention demand scan admission".into(),
            ));
        }
        let mut lease = budget
            .map(|b| {
                b.lease(WorkRequest {
                    scratch_bytes: scratch,
                    persistent_bytes: 0,
                    work: scan_work,
                    proposals: 0,
                })
                .map_err(work_refusal)
            })
            .transpose()?;
        checkpoint(deadline, budget)?;
        let mut active = vec![0u8; queries];
        let mut summary = Summary {
            active_queries: 0,
            visible_pairs: 0,
            max_visible_keys: 0,
            scan_work,
            map_bytes: queries,
        };
        let mut scanned = 0usize;
        for slice in 0..dims.n_slices {
            for query in 0..dims.sq {
                checkpoint(deadline, budget)?;
                let mut demanded = false;
                let start = (slice * dims.sq + query) * dims.dv;
                for output in 0..node.num_outputs() {
                    for channel in 0..dims.dv {
                        if scanned.is_multiple_of(1024) {
                            checkpoint(deadline, budget)?;
                        }
                        scanned += 1;
                        let column = start + channel;
                        for a in [node.lower_a(), node.upper_a()] {
                            let value = a[[output, column]];
                            if !value.is_finite() {
                                return Err(NyError::UnsupportedOp(
                                    "nonfinite attention demand coefficient".into(),
                                ));
                            }
                            // Compare stored bits: DAZ must not erase a tiny
                            // requested coefficient during zero detection.
                            demanded |= value.to_bits() & 0x7fff_ffff != 0;
                        }
                        for error in [node.lower_a_err(), node.upper_a_err()]
                            .into_iter()
                            .flatten()
                        {
                            let value = error[[output, column]];
                            let nonzero = value.to_bits() & 0x7fff_ffff != 0;
                            if !value.is_finite() || (nonzero && value.is_sign_negative()) {
                                return Err(NyError::UnsupportedOp(
                                    "invalid attention demand coefficient error".into(),
                                ));
                            }
                            demanded |= nonzero;
                        }
                    }
                }
                if demanded || !skip_zero {
                    active[slice * dims.sq + query] = 1;
                    let (start, end) = visible_range(layer, query, dims.sk);
                    let visible = end - start;
                    summary.active_queries =
                        summary.active_queries.checked_add(1).ok_or_else(|| {
                            NyError::UnsupportedOp("attention demand count overflow".into())
                        })?;
                    summary.visible_pairs =
                        summary.visible_pairs.checked_add(visible).ok_or_else(|| {
                            NyError::UnsupportedOp("attention visible-pair count overflow".into())
                        })?;
                    summary.max_visible_keys = summary.max_visible_keys.max(visible);
                }
            }
        }
        checkpoint(deadline, budget)?;
        if let Some(lease) = lease.as_mut() {
            lease.complete().map_err(work_refusal)?;
        }
        Ok(Self {
            active,
            summary,
            _lease: lease,
        })
    }
}

#[cfg(test)]
mod tests;
