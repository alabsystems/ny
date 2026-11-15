// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pre-allocation compilation bounds and original-DAG reference liveness.
//!
//! These are allocation-capacity reservations, not RSS measurements. New
//! vectors use exact checked capacities; original parameter capacities move
//! into the canonical model. No advisory root census grants admission here.

use std::mem::size_of;
use std::time::Instant;

use ny_core::Result;

use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkRequest};

use super::super::net::{ConvOp, TwinNet, TwinOp};
use super::super::spec::{TwinOpSpec, TwinSpec};
use super::compiler::{reserved, OriginalModel};
use super::ir::{checkpoint, invalid, refusal};

fn add(total: &mut usize, count: usize, width: usize) -> Result<()> {
    *total = count
        .checked_mul(width)
        .and_then(|n| total.checked_add(n))
        .ok_or_else(|| invalid("original capacity/work overflow"))?;
    Ok(())
}

fn floats(total: &mut usize, values: &[&Vec<f64>]) -> Result<()> {
    for values in values {
        add(total, values.capacity(), size_of::<f64>())?;
    }
    Ok(())
}

pub(super) fn spec_capacity_bytes(
    spec: &TwinSpec,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<usize> {
    let mut bytes = size_of::<TwinSpec>();
    add(&mut bytes, spec.ops.capacity(), size_of::<TwinOpSpec>())?;
    for op in &spec.ops {
        checkpoint(budget, deadline)?;
        match op {
            TwinOpSpec::Conv {
                weight,
                bias,
                bias_err,
                ..
            }
            | TwinOpSpec::ConvTranspose {
                weight,
                bias,
                bias_err,
                ..
            }
            | TwinOpSpec::Gemm {
                weight,
                bias,
                bias_err,
                ..
            } => floats(&mut bytes, &[weight, bias, bias_err])?,
            TwinOpSpec::ChannelAffine {
                scale,
                shift,
                shift_err,
                ..
            } => floats(&mut bytes, &[scale, shift, shift_err])?,
            _ => (),
        }
    }
    checkpoint(budget, deadline)?;
    Ok(bytes)
}

pub(super) fn compile_prediction(
    spec: &TwinSpec,
    sizes: &[usize],
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<WorkRequest> {
    // Include raw spec capacity as a conservative upper bound on every moved
    // parameter vector. Released source-only vectors/headers remain reserved;
    // this intentionally avoids underpricing allocator growth or ownership.
    let source = spec_capacity_bytes(spec, budget, deadline)?;
    let mut bytes = source
        .checked_add(size_of::<TwinNet>())
        .and_then(|n| n.checked_add(1024))
        .ok_or_else(|| invalid("compiled owner bound overflow"))?;
    add(&mut bytes, spec.ops.len(), size_of::<TwinOp>())?;
    add(&mut bytes, sizes.len(), size_of::<usize>())?;
    add(&mut bytes, spec.ops.len(), size_of::<usize>())?;
    let mut work = 4096usize;
    add(&mut work, source, 32)?;
    add(&mut work, spec.ops.len(), 2048)?;
    for (op_id, op) in spec.ops.iter().enumerate() {
        checkpoint(budget, deadline)?;
        match op {
            TwinOpSpec::Conv {
                weight,
                kernel,
                ishape,
                oshape,
                ..
            }
            | TwinOpSpec::ConvTranspose {
                weight,
                kernel,
                ishape,
                oshape,
                ..
            } => {
                let taps = kernel
                    .2
                    .checked_mul(kernel.3)
                    .ok_or_else(|| invalid("kernel bound overflow"))?;
                let input = ishape
                    .1
                    .checked_mul(ishape.2)
                    .ok_or_else(|| invalid("input plane bound overflow"))?;
                let output = oshape
                    .1
                    .checked_mul(oshape.2)
                    .ok_or_else(|| invalid("output plane bound overflow"))?;
                let gather = kernel
                    .1
                    .checked_mul(taps)
                    .and_then(|n| n.checked_mul(output))
                    .ok_or_else(|| invalid("gather bound overflow"))?;
                // For each fixed kernel tap, the valid input/output coordinate
                // relation is injective even with stride and output padding.
                // Thus at most min(input plane, output plane) records per tap.
                let records = input
                    .min(output)
                    .checked_mul(taps)
                    .ok_or_else(|| invalid("tap bound overflow"))?;
                let visits = input
                    .checked_mul(taps)
                    .ok_or_else(|| invalid("tap work overflow"))?;
                add(&mut bytes, 1, size_of::<ConvOp>())?;
                add(&mut bytes, weight.len(), size_of::<f64>())?;
                add(&mut bytes, gather, size_of::<usize>())?;
                add(&mut bytes, input, size_of::<Vec<(usize, usize, usize)>>())?;
                add(&mut bytes, records, size_of::<(usize, usize, usize)>())?;
                add(&mut work, gather, 256)?;
                add(&mut work, visits, 384)?;
                add(&mut work, records, 256)?;
                add(&mut work, weight.len(), 128)?;
                add(&mut work, input, 512)?;
            }
            TwinOpSpec::ChannelAffine { .. } => {
                let neurons = *sizes
                    .get(op_id + 1)
                    .ok_or_else(|| invalid("missing original size"))?;
                add(&mut bytes, neurons, 3 * size_of::<f64>())?;
                add(&mut work, neurons, 384)?;
            }
            TwinOpSpec::Gemm {
                bias_err, shape, ..
            } if bias_err.is_empty() => {
                add(&mut bytes, shape.0, size_of::<f64>())?;
                add(&mut work, shape.0, 128)?;
            }
            _ => (),
        }
    }
    checkpoint(budget, deadline)?;
    Ok(WorkRequest {
        persistent_bytes: bytes,
        scratch_bytes: 16 * 1024,
        work: u64::try_from(work).map_err(|_| invalid("work bound exceeds u64"))?,
        proposals: 0,
    })
}

pub(super) fn actual_compiled_bytes(
    net: &TwinNet,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<usize> {
    let mut bytes = size_of::<TwinNet>();
    add(&mut bytes, net.ops.capacity(), size_of::<TwinOp>())?;
    add(&mut bytes, net.tsize.capacity(), size_of::<usize>())?;
    add(&mut bytes, net.trunk_relus.capacity(), size_of::<usize>())?;
    for op in &net.ops {
        checkpoint(budget, deadline)?;
        match op {
            TwinOp::Conv(c) => {
                add(&mut bytes, 1, size_of::<ConvOp>())?;
                floats(&mut bytes, &[&c.wmat, &c.wt, &c.bias, &c.bias_err])?;
                add(&mut bytes, c.gather.capacity(), size_of::<usize>())?;
                add(
                    &mut bytes,
                    c.back_taps.capacity(),
                    size_of::<Vec<(usize, usize, usize)>>(),
                )?;
                for taps in &c.back_taps {
                    checkpoint(budget, deadline)?;
                    add(
                        &mut bytes,
                        taps.capacity(),
                        size_of::<(usize, usize, usize)>(),
                    )?;
                }
            }
            TwinOp::ChannelAffine {
                scale,
                shift,
                shift_err,
                ..
            } => floats(&mut bytes, &[scale, shift, shift_err])?,
            TwinOp::Gemm {
                weight,
                bias,
                bias_err,
                ..
            } => floats(&mut bytes, &[weight, bias, bias_err])?,
            _ => (),
        }
    }
    checkpoint(budget, deadline)?;
    Ok(bytes)
}

/// Shared immutable release schedule for reference buffers. A retained
/// schedule keeps the model alive, but never retains numerical reference boxes.
/// Count original edges including both arms of Add, even if they coincide.
pub(in crate::margin_row) struct OriginalLiveness {
    model: OriginalModel,
    uses: Vec<usize>,
    offsets: Vec<usize>,
    releases: Vec<usize>,
    _resident: PersistentReservation,
}

impl OriginalLiveness {
    pub(in crate::margin_row) fn new(
        model: OriginalModel,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let deadline = model.deadline(requested)?;
        let budget = model.budget();
        let n = model
            .net()
            .ops
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid("liveness size overflow"))?;
        let offset_count = n
            .checked_add(1)
            .ok_or_else(|| invalid("liveness offset overflow"))?;
        let mut persistent = 512usize;
        add(&mut persistent, n, 3 * size_of::<usize>())?;
        add(&mut persistent, 1, size_of::<usize>())?;
        let mut scratch = 512usize;
        add(&mut scratch, n, 2 * size_of::<usize>())?;
        let mut work = 1024usize;
        add(&mut work, n, 512)?;
        let mut lease = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: persistent,
                    scratch_bytes: scratch,
                    work: u64::try_from(work).map_err(|_| invalid("liveness work overflow"))?,
                    proposals: 0,
                },
            )
            .map_err(refusal)?;
        let deadline = super::ir::earliest(deadline, lease.deadline());
        let mut uses = reserved(n, budget, deadline)?;
        let mut last = reserved(n, budget, deadline)?;
        for tensor in 0..n {
            if tensor.is_multiple_of(4096) {
                checkpoint(budget, deadline)?;
            }
            uses.push(0usize);
            last.push(tensor.saturating_sub(1));
        }
        for (op_id, op) in model.net().ops.iter().enumerate() {
            checkpoint(budget, deadline)?;
            let mut edge = |tensor: usize| -> Result<()> {
                if tensor > op_id {
                    return Err(invalid("liveness future edge"));
                }
                uses[tensor] = uses[tensor]
                    .checked_add(1)
                    .ok_or_else(|| invalid("consumer count overflow"))?;
                last[tensor] = op_id;
                Ok(())
            };
            match op {
                TwinOp::Conv(c) => edge(c.input)?,
                TwinOp::ChannelAffine { input, .. }
                | TwinOp::Relu { input, .. }
                | TwinOp::Flatten { input }
                | TwinOp::Gemm { input, .. } => edge(*input)?,
                TwinOp::Add { lhs, rhs } => {
                    edge(*lhs)?;
                    edge(*rhs)?;
                }
            }
        }
        // The final output lives through query/result consumption; its release
        // belongs to the separate finish boundary, never the last producer.
        last[n - 1] = n - 1;
        let mut offsets = reserved(offset_count, budget, deadline)?;
        for i in 0..=n {
            if i.is_multiple_of(4096) {
                checkpoint(budget, deadline)?;
            }
            offsets.push(0usize);
        }
        for (i, &release) in last.iter().enumerate() {
            if i.is_multiple_of(4096) {
                checkpoint(budget, deadline)?;
            }
            offsets[release + 1] += 1;
        }
        for i in 1..=n {
            if i.is_multiple_of(4096) {
                checkpoint(budget, deadline)?;
            }
            offsets[i] += offsets[i - 1];
        }
        let mut cursors = reserved(n, budget, deadline)?;
        for (i, &offset) in offsets[..n].iter().enumerate() {
            if i.is_multiple_of(4096) {
                checkpoint(budget, deadline)?;
            }
            cursors.push(offset);
        }
        let mut releases = reserved(n, budget, deadline)?;
        for i in 0..n {
            if i.is_multiple_of(4096) {
                checkpoint(budget, deadline)?;
            }
            releases.push(usize::MAX);
        }
        for (tensor, &boundary) in last.iter().enumerate() {
            if tensor.is_multiple_of(4096) {
                checkpoint(budget, deadline)?;
            }
            releases[cursors[boundary]] = tensor;
            cursors[boundary] += 1;
        }
        let mut actual = size_of::<Self>();
        for capacity in [uses.capacity(), offsets.capacity(), releases.capacity()] {
            add(&mut actual, capacity, size_of::<usize>())?;
        }
        if actual > persistent {
            return Err(invalid("liveness capacity reconciliation failed"));
        }
        drop(last);
        drop(cursors);
        checkpoint(budget, deadline)?;
        lease.mark_materialized().map_err(refusal)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        Ok(Self {
            model,
            uses,
            offsets,
            releases,
            _resident: resident,
        })
    }

    pub(in crate::margin_row) fn same_model(&self, model: &OriginalModel) -> bool {
        self.model.same_model(model)
    }
    pub(in crate::margin_row) fn consumer_count(&self, tensor: usize) -> Option<usize> {
        self.uses.get(tensor).copied()
    }
    pub(in crate::margin_row) fn release_after(&self, op: usize) -> Option<&[usize]> {
        if op >= self.model.net().ops.len() {
            return None;
        }
        Some(&self.releases[self.offsets[op]..self.offsets[op + 1]])
    }
    pub(in crate::margin_row) fn finish_releases(&self) -> &[usize] {
        let boundary = self.model.net().ops.len();
        &self.releases[self.offsets[boundary]..self.offsets[boundary + 1]]
    }
}

#[cfg(test)]
#[path = "liveness_tests.rs"]
mod tests;
